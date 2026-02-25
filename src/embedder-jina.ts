/**
 * Jina AI Embedding Provider
 * Supports jina-embeddings-v3, jina-embeddings-v2-base-en.
 * Uses native fetch — no SDK dependency.
 */

import { vectorDimsForModel } from "./config.js";
import { EmbeddingCache } from "./embedder.js";
import type { IEmbedder } from "./embedder-interface.js";

// ============================================================================
// Types
// ============================================================================

export interface JinaEmbeddingConfig {
  apiKey: string;
  model: string;
  dimensions?: number;
  baseUrl?: string;
}

interface JinaEmbeddingResponse {
  model: string;
  object: string;
  data: Array<{ object: string; embedding: number[]; index: number }>;
  usage: { total_tokens: number; prompt_tokens: number };
}

// ============================================================================
// JinaEmbedder Class
// ============================================================================

const DEFAULT_JINA_URL = "https://api.jina.ai/v1/embeddings";

export class JinaEmbedder implements IEmbedder {
  public readonly dimensions: number;
  private readonly _cache: EmbeddingCache;
  private readonly _apiKey: string;
  private readonly _model: string;
  private readonly _baseUrl: string;
  private readonly _supportsTaskParam: boolean;

  constructor(config: JinaEmbeddingConfig) {
    this._apiKey = config.apiKey;
    this._model = config.model;
    this.dimensions = vectorDimsForModel(config.model, config.dimensions);
    this._cache = new EmbeddingCache(256, 30);
    this._baseUrl = config.baseUrl || DEFAULT_JINA_URL;
    // jina-embeddings-v3 supports the task parameter
    this._supportsTaskParam = config.model.includes("v3");
  }

  // --------------------------------------------------------------------------
  // IEmbedder API
  // --------------------------------------------------------------------------

  async embed(text: string): Promise<number[]> {
    return this.embedPassage(text);
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.embedSingle(text, "retrieval.query");
  }

  async embedPassage(text: string): Promise<number[]> {
    return this.embedSingle(text, "retrieval.passage");
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return this.embedBatchPassage(texts);
  }

  async embedBatchQuery(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts, "retrieval.query");
  }

  async embedBatchPassage(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts, "retrieval.passage");
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

  private async embedSingle(text: string, task: string): Promise<number[]> {
    if (!text || text.trim().length === 0) {
      throw new Error("Cannot embed empty text");
    }

    const cached = this._cache.get(text, task);
    if (cached) return cached;

    const body: Record<string, unknown> = {
      model: this._model,
      input: [text],
    };

    if (this._supportsTaskParam) {
      body.task = task;
    }

    const response = await this.callJinaAPI(body);
    const embedding = response.data[0]?.embedding;
    if (!embedding) {
      throw new Error("No embedding returned from Jina AI");
    }

    this.validateEmbedding(embedding);
    this._cache.set(text, task, embedding);
    return embedding;
  }

  private async embedMany(texts: string[], task: string): Promise<number[][]> {
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

    // Jina supports up to 2048 texts per batch; chunk at 128 for consistency
    const BATCH_SIZE = 128;
    const results: number[][] = new Array(texts.length);

    for (let i = 0; i < validTexts.length; i += BATCH_SIZE) {
      const batchTexts = validTexts.slice(i, i + BATCH_SIZE);
      const batchIndices = validIndices.slice(i, i + BATCH_SIZE);

      const body: Record<string, unknown> = {
        model: this._model,
        input: batchTexts,
      };

      if (this._supportsTaskParam) {
        body.task = task;
      }

      const response = await this.callJinaAPI(body);

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

  private async callJinaAPI(body: Record<string, unknown>): Promise<JinaEmbeddingResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    try {
      const response = await fetch(this._baseUrl, {
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
        throw new Error(`Jina AI embedding API returned ${response.status}: ${errorText}`);
      }

      return await response.json() as JinaEmbeddingResponse;
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
