/**
 * Embedding Provider Interface
 * Abstraction layer for switching between Voyage AI, OpenAI, and Jina embedding providers.
 */

// ============================================================================
// Configuration
// ============================================================================

export interface EmbedderConfig {
  provider: "voyage" | "openai" | "jina";
  apiKey: string;
  model: string;
  dimensions?: number;
  baseUrl?: string; // custom endpoint override
}

// ============================================================================
// Embedder Interface
// ============================================================================

export interface IEmbedder {
  readonly dimensions: number;
  readonly model: string;
  embed(text: string): Promise<number[]>;
  embedQuery(text: string): Promise<number[]>;
  embedPassage(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  embedBatchQuery(texts: string[]): Promise<number[][]>;
  embedBatchPassage(texts: string[]): Promise<number[][]>;
  test(): Promise<{ success: boolean; error?: string; dimensions?: number }>;
  get cacheStats(): { size: number; hits: number; misses: number; hitRate: string };
}
