/**
 * OpenAI Embedding Provider
 * Supports text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002.
 * Uses native fetch — no openai SDK dependency.
 */

import { vectorDimsForModel } from "./config.js";
import { EmbeddingCache } from "./embedder.js";
import type { IEmbedder } from "./embedder-interface.js";

// ============================================================================
// Types
// ============================================================================

export interface OpenAIEmbeddingConfig {
  apiKey: string;
  model: string;
  dimensions?: number;
  baseUrl?: string;
}

interface OpenAIEmbeddingResponse {
  object: string;
  data: Array<{ object: string; embedding: number[]; index: number }>;
  model: string;
  usage: { prompt_tokens: number; total_tokens: number };
}

// ============================================================================
// OpenAIEmbedder Class
// ============================================================================

const DEFAULT_OPENAI_URL = "https://api.openai.com/v1/embeddings";

export class OpenAIEmbedder implements IEmbedder {
  public readonly dimensions: number;
  private readonly _cache: EmbeddingCache;
  private readonly _apiKey: string;
  private readonly _model: string;
  private readonly _baseUrl: string;
  private readonly _supportsDimensionsParam: boolean;

  constructor(config: OpenAIEmbeddingConfig) {
    this._apiKey = config.apiKey;
    this._model = config.model;
    this.dimensions = vectorDimsForModel(config.model, config.dimensions);
    this._cache = new EmbeddingCache(256, 30);
    this._baseUrl = config.baseUrl || DEFAULT_OPENAI_URL;
    // text-embedding-3-* models support the dimensions parameter for truncation
    this._supportsDimensionsParam = config.model.startsWith("text-embedding-3-");
  }

  // --------------------------------------------------------------------------
  // IEmbedder API
  // OpenAI has no input_type distinction, so all methods delegate to the same logic
  // --------------------------------------------------------------------------

  async embed(text: string): Promise<number[]> {
    return this.embedSingle(text);
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.embedSingle(text, "query");
  }

  async embedPassage(text: string): Promise<number[]> {
    return this.embedSingle(text, "passage");
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts);
  }

  async embedBatchQuery(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts, "query");
  }

  async embedBatchPassage(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts, "passage");
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

  private async embedSingle(text: string, inputType?: string): Promise<number[]> {
    if (!text || text.trim().length === 0) {
      throw new Error("Cannot embed empty text");
    }

    const cached = this._cache.get(text, inputType);
    if (cached) return cached;

    const body: Record<string, unknown> = {
      model: this._model,
      input: [text],
    };

    if (this._supportsDimensionsParam && this.dimensions) {
      body.dimensions = this.dimensions;
    }

    const response = await this.callOpenAIAPI(body);
    const embedding = response.data[0]?.embedding;
    if (!embedding) {
      throw new Error("No embedding returned from OpenAI");
    }

    this.validateEmbedding(embedding);
    this._cache.set(text, inputType, embedding);
    return embedding;
  }

  private async embedMany(texts: string[], inputType?: string): Promise<number[][]> {
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

    // OpenAI supports up to 2048 texts per batch; chunk at 128 for consistency
    const BATCH_SIZE = 128;
    const results: number[][] = new Array(texts.length);

    for (let i = 0; i < validTexts.length; i += BATCH_SIZE) {
      const batchTexts = validTexts.slice(i, i + BATCH_SIZE);
      const batchIndices = validIndices.slice(i, i + BATCH_SIZE);

      const body: Record<string, unknown> = {
        model: this._model,
        input: batchTexts,
      };

      if (this._supportsDimensionsParam && this.dimensions) {
        body.dimensions = this.dimensions;
      }

      const response = await this.callOpenAIAPI(body);

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

  private async callOpenAIAPI(body: Record<string, unknown>): Promise<OpenAIEmbeddingResponse> {
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
        throw new Error(`OpenAI embedding API returned ${response.status}: ${errorText}`);
      }

      return await response.json() as OpenAIEmbeddingResponse;
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
