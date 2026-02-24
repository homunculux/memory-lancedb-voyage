/**
 * Voyage AI Embedding Layer
 * Uses native fetch — no OpenAI SDK dependency.
 */

import { createHash } from "node:crypto";
import { vectorDimsForModel } from "./config.js";

// ============================================================================
// Embedding Cache (LRU with TTL)
// ============================================================================

interface CacheEntry {
  vector: number[];
  createdAt: number;
}

class EmbeddingCache {
  private cache = new Map<string, CacheEntry>();
  private readonly maxSize: number;
  private readonly ttlMs: number;
  public hits = 0;
  public misses = 0;

  constructor(maxSize = 256, ttlMinutes = 30) {
    this.maxSize = maxSize;
    this.ttlMs = ttlMinutes * 60_000;
  }

  private key(text: string, inputType?: string): string {
    return createHash("sha256").update(`${inputType || ""}:${text}`).digest("hex").slice(0, 24);
  }

  get(text: string, inputType?: string): number[] | undefined {
    const k = this.key(text, inputType);
    const entry = this.cache.get(k);
    if (!entry) {
      this.misses++;
      return undefined;
    }
    if (Date.now() - entry.createdAt > this.ttlMs) {
      this.cache.delete(k);
      this.misses++;
      return undefined;
    }
    // Move to end (most recently used)
    this.cache.delete(k);
    this.cache.set(k, entry);
    this.hits++;
    return entry.vector;
  }

  set(text: string, inputType: string | undefined, vector: number[]): void {
    const k = this.key(text, inputType);
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(k, { vector, createdAt: Date.now() });
  }

  get size(): number { return this.cache.size; }
  get stats(): { size: number; hits: number; misses: number; hitRate: string } {
    const total = this.hits + this.misses;
    return {
      size: this.cache.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? `${((this.hits / total) * 100).toFixed(1)}%` : "N/A",
    };
  }
}

// ============================================================================
// Types
// ============================================================================

export interface EmbeddingConfig {
  apiKey: string;
  model: string;
  dimensions?: number;
}

interface VoyageEmbeddingResponse {
  object: string;
  data: Array<{ object: string; embedding: number[]; index: number }>;
  model: string;
  usage: { total_tokens: number };
}

// ============================================================================
// Embedder Class
// ============================================================================

const VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings";

export class Embedder {
  public readonly dimensions: number;
  private readonly _cache: EmbeddingCache;
  private readonly _apiKey: string;
  private readonly _model: string;

  constructor(config: EmbeddingConfig) {
    this._apiKey = config.apiKey;
    this._model = config.model;
    this.dimensions = vectorDimsForModel(config.model, config.dimensions);
    this._cache = new EmbeddingCache(256, 30);
  }

  // --------------------------------------------------------------------------
  // Backward-compatible API
  // --------------------------------------------------------------------------

  async embed(text: string): Promise<number[]> {
    return this.embedPassage(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return this.embedBatchPassage(texts);
  }

  // --------------------------------------------------------------------------
  // Task-aware API (Voyage uses input_type: "query" | "document")
  // --------------------------------------------------------------------------

  async embedQuery(text: string): Promise<number[]> {
    return this.embedSingle(text, "query");
  }

  async embedPassage(text: string): Promise<number[]> {
    return this.embedSingle(text, "document");
  }

  async embedBatchQuery(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts, "query");
  }

  async embedBatchPassage(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts, "document");
  }

  // --------------------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------------------

  private validateEmbedding(embedding: number[]): void {
    if (!Array.isArray(embedding)) {
      throw new Error(`Embedding is not an array (got ${typeof embedding})`);
    }
    if (embedding.length !== this.dimensions) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimensions}, got ${embedding.length}`,
      );
    }
  }

  private async embedSingle(text: string, inputType: "query" | "document"): Promise<number[]> {
    if (!text || text.trim().length === 0) {
      throw new Error("Cannot embed empty text");
    }

    const cached = this._cache.get(text, inputType);
    if (cached) return cached;

    const body: Record<string, unknown> = {
      model: this._model,
      input: [text],
      input_type: inputType,
    };

    const response = await this.callVoyageAPI(body);
    const embedding = response.data[0]?.embedding;
    if (!embedding) {
      throw new Error("No embedding returned from Voyage AI");
    }

    this.validateEmbedding(embedding);
    this._cache.set(text, inputType, embedding);
    return embedding;
  }

  private async embedMany(texts: string[], inputType: "query" | "document"): Promise<number[][]> {
    if (!texts || texts.length === 0) return [];

    const validTexts: string[] = [];
    const validIndices: number[] = [];

    texts.forEach((text, index) => {
      if (text && text.trim().length > 0) {
        validTexts.push(text);
        validIndices.push(index);
      }
    });

    if (validTexts.length === 0) return texts.map(() => []);

    // Voyage AI supports up to 128 texts per batch; chunk if needed
    const BATCH_SIZE = 128;
    const results: number[][] = new Array(texts.length);

    for (let i = 0; i < validTexts.length; i += BATCH_SIZE) {
      const batchTexts = validTexts.slice(i, i + BATCH_SIZE);
      const batchIndices = validIndices.slice(i, i + BATCH_SIZE);

      const body: Record<string, unknown> = {
        model: this._model,
        input: batchTexts,
        input_type: inputType,
      };

      const response = await this.callVoyageAPI(body);

      response.data.forEach((item, idx) => {
        const originalIndex = batchIndices[idx];
        const embedding = item.embedding;
        this.validateEmbedding(embedding);
        results[originalIndex] = embedding;
      });
    }

    // Fill empty arrays for invalid texts
    for (let i = 0; i < texts.length; i++) {
      if (!results[i]) results[i] = [];
    }

    return results;
  }

  private async callVoyageAPI(body: Record<string, unknown>): Promise<VoyageEmbeddingResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    try {
      const response = await fetch(VOYAGE_EMBEDDINGS_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this._apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => "");
        throw new Error(`Voyage AI embedding API returned ${response.status}: ${errorText}`);
      }

      return await response.json() as VoyageEmbeddingResponse;
    } finally {
      clearTimeout(timeout);
    }
  }

  get model(): string {
    return this._model;
  }

  async test(): Promise<{ success: boolean; error?: string; dimensions?: number }> {
    try {
      const testEmbedding = await this.embedPassage("test");
      return { success: true, dimensions: testEmbedding.length };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  get cacheStats() {
    return this._cache.stats;
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createEmbedder(config: EmbeddingConfig): Embedder {
  return new Embedder(config);
}

export { vectorDimsForModel as getVectorDimensions };
