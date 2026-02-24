/**
 * Hybrid Retrieval System
 * Vector search + BM25 full-text search with RRF fusion, Voyage AI reranking
 */

import type { MemoryStore, MemorySearchResult } from "./store.js";
import type { Embedder } from "./embedder.js";
import { filterNoise } from "./noise-filter.js";

// ============================================================================
// Types & Configuration
// ============================================================================

export interface RetrievalConfig {
  mode: "hybrid" | "vector";
  vectorWeight: number;
  bm25Weight: number;
  minScore: number;
  rerank: "cross-encoder" | "lightweight" | "none";
  candidatePoolSize: number;
  recencyHalfLifeDays: number;
  recencyWeight: number;
  filterNoise: boolean;
  /** Voyage AI reranker model (default: rerank-2) */
  rerankModel: string;
  lengthNormAnchor: number;
  hardMinScore: number;
  timeDecayHalfLifeDays: number;
}

export interface RetrievalContext {
  query: string;
  limit: number;
  scopeFilter?: string[];
  category?: string;
}

export interface RetrievalResult extends MemorySearchResult {
  sources: {
    vector?: { score: number; rank: number };
    bm25?: { score: number; rank: number };
    fused?: { score: number };
    reranked?: { score: number };
  };
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_RETRIEVAL_CONFIG: RetrievalConfig = {
  mode: "hybrid",
  vectorWeight: 0.7,
  bm25Weight: 0.3,
  minScore: 0.3,
  rerank: "cross-encoder",
  candidatePoolSize: 20,
  recencyHalfLifeDays: 14,
  recencyWeight: 0.10,
  filterNoise: true,
  rerankModel: "rerank-2",
  lengthNormAnchor: 500,
  hardMinScore: 0.35,
  timeDecayHalfLifeDays: 60,
};

// ============================================================================
// Utility Functions
// ============================================================================

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.floor(value)));
}

function clamp01(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return Number.isFinite(fallback) ? fallback : 0;
  return Math.min(1, Math.max(0, value));
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error("Vector dimensions must match for cosine similarity");
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const norm = Math.sqrt(normA) * Math.sqrt(normB);
  return norm === 0 ? 0 : dotProduct / norm;
}

// ============================================================================
// Memory Retriever
// ============================================================================

const VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank";

export class MemoryRetriever {
  constructor(
    private store: MemoryStore,
    private embedder: Embedder,
    private config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
    private voyageApiKey?: string,
  ) {}

  async retrieve(context: RetrievalContext): Promise<RetrievalResult[]> {
    const { query, limit, scopeFilter, category } = context;
    const safeLimit = clampInt(limit, 1, 20);

    if (this.config.mode === "vector" || !this.store.hasFtsSupport) {
      return this.vectorOnlyRetrieval(query, safeLimit, scopeFilter, category);
    }

    return this.hybridRetrieval(query, safeLimit, scopeFilter, category);
  }

  private async vectorOnlyRetrieval(
    query: string, limit: number, scopeFilter?: string[], category?: string,
  ): Promise<RetrievalResult[]> {
    const queryVector = await this.embedder.embedQuery(query);
    const results = await this.store.vectorSearch(queryVector, limit, this.config.minScore, scopeFilter);

    const filtered = category ? results.filter(r => r.entry.category === category) : results;
    const mapped = filtered.map((result, index) => ({
      ...result,
      sources: { vector: { score: result.score, rank: index + 1 } },
    } as RetrievalResult));

    return this.applyPostProcessing(mapped, limit);
  }

  private async hybridRetrieval(
    query: string, limit: number, scopeFilter?: string[], category?: string,
  ): Promise<RetrievalResult[]> {
    const candidatePoolSize = Math.max(this.config.candidatePoolSize, limit * 2);
    const queryVector = await this.embedder.embedQuery(query);

    const [vectorResults, bm25Results] = await Promise.all([
      this.runVectorSearch(queryVector, candidatePoolSize, scopeFilter, category),
      this.runBM25Search(query, candidatePoolSize, scopeFilter, category),
    ]);

    const fusedResults = this.fuseResults(vectorResults, bm25Results);
    const filtered = fusedResults.filter(r => r.score >= this.config.minScore);

    const reranked = this.config.rerank !== "none"
      ? await this.rerankResults(query, queryVector, filtered.slice(0, limit * 2))
      : filtered;

    return this.applyPostProcessing(reranked, limit);
  }

  private applyPostProcessing(results: RetrievalResult[], limit: number): RetrievalResult[] {
    const boosted = this.applyRecencyBoost(results);
    const weighted = this.applyImportanceWeight(boosted);
    const lengthNormalized = this.applyLengthNormalization(weighted);
    const timeDecayed = this.applyTimeDecay(lengthNormalized);
    const hardFiltered = timeDecayed.filter(r => r.score >= this.config.hardMinScore);
    const denoised = this.config.filterNoise
      ? filterNoise(hardFiltered, r => r.entry.text)
      : hardFiltered;
    const deduplicated = this.applyMMRDiversity(denoised);
    return deduplicated.slice(0, limit);
  }

  private async runVectorSearch(
    queryVector: number[], limit: number, scopeFilter?: string[], category?: string,
  ): Promise<Array<MemorySearchResult & { rank: number }>> {
    const results = await this.store.vectorSearch(queryVector, limit, 0.1, scopeFilter);
    const filtered = category ? results.filter(r => r.entry.category === category) : results;
    return filtered.map((result, index) => ({ ...result, rank: index + 1 }));
  }

  private async runBM25Search(
    query: string, limit: number, scopeFilter?: string[], category?: string,
  ): Promise<Array<MemorySearchResult & { rank: number }>> {
    const results = await this.store.bm25Search(query, limit, scopeFilter);
    const filtered = category ? results.filter(r => r.entry.category === category) : results;
    return filtered.map((result, index) => ({ ...result, rank: index + 1 }));
  }

  private fuseResults(
    vectorResults: Array<MemorySearchResult & { rank: number }>,
    bm25Results: Array<MemorySearchResult & { rank: number }>,
  ): RetrievalResult[] {
    const vectorMap = new Map<string, MemorySearchResult & { rank: number }>();
    const bm25Map = new Map<string, MemorySearchResult & { rank: number }>();

    vectorResults.forEach(result => vectorMap.set(result.entry.id, result));
    bm25Results.forEach(result => bm25Map.set(result.entry.id, result));

    const allIds = new Set([...vectorMap.keys(), ...bm25Map.keys()]);
    const fusedResults: RetrievalResult[] = [];

    for (const id of allIds) {
      const vectorResult = vectorMap.get(id);
      const bm25Result = bm25Map.get(id);
      const baseResult = vectorResult || bm25Result!;

      const vectorScore = vectorResult ? vectorResult.score : 0;
      const bm25Hit = bm25Result ? 1 : 0;

      const fusedScore = vectorResult
        ? clamp01(vectorScore + (bm25Hit * 0.15 * vectorScore), 0.1)
        : clamp01(Math.max(bm25Result!.score, 0.5), 0.1);

      fusedResults.push({
        entry: baseResult.entry,
        score: fusedScore,
        sources: {
          vector: vectorResult ? { score: vectorResult.score, rank: vectorResult.rank } : undefined,
          bm25: bm25Result ? { score: bm25Result.score, rank: bm25Result.rank } : undefined,
          fused: { score: fusedScore },
        },
      });
    }

    return fusedResults.sort((a, b) => b.score - a.score);
  }

  /**
   * Rerank using Voyage AI rerank API (cross-encoder).
   * Falls back to cosine similarity if API unavailable.
   */
  private async rerankResults(query: string, queryVector: number[], results: RetrievalResult[]): Promise<RetrievalResult[]> {
    if (results.length === 0) return results;

    if (this.config.rerank === "cross-encoder" && this.voyageApiKey) {
      try {
        const documents = results.map(r => r.entry.text);
        const model = this.config.rerankModel || "rerank-2";

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(VOYAGE_RERANK_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${this.voyageApiKey}`,
          },
          body: JSON.stringify({
            model,
            query,
            documents,
            top_k: results.length,
          }),
          signal: controller.signal,
        });

        clearTimeout(timeout);

        if (response.ok) {
          const data = await response.json() as {
            data?: Array<{ index: number; relevance_score: number }>;
          };

          if (!Array.isArray(data.data)) {
            console.warn("Voyage rerank: invalid response shape, falling back to cosine");
          } else {
            const returnedIndices = new Set(data.data.map(r => r.index));

            const reranked = data.data
              .filter(item => item.index >= 0 && item.index < results.length)
              .map(item => {
                const original = results[item.index];
                const blendedScore = clamp01(
                  item.relevance_score * 0.6 + original.score * 0.4,
                  original.score * 0.5,
                );
                return {
                  ...original,
                  score: blendedScore,
                  sources: {
                    ...original.sources,
                    reranked: { score: item.relevance_score },
                  },
                };
              });

            const unreturned = results
              .filter((_, idx) => !returnedIndices.has(idx))
              .map(r => ({ ...r, score: r.score * 0.8 }));

            return [...reranked, ...unreturned].sort((a, b) => b.score - a.score);
          }
        } else {
          console.warn(`Voyage rerank API returned ${response.status}, falling back to cosine`);
        }
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          console.warn("Voyage rerank timed out (5s), falling back to cosine");
        } else {
          console.warn("Voyage rerank failed, falling back to cosine:", error);
        }
      }
    }

    // Fallback: lightweight cosine similarity rerank
    try {
      const reranked = results.map(result => {
        const cosineScore = cosineSimilarity(queryVector, result.entry.vector);
        const combinedScore = (result.score * 0.7) + (cosineScore * 0.3);
        return {
          ...result,
          score: clamp01(combinedScore, result.score),
          sources: { ...result.sources, reranked: { score: cosineScore } },
        };
      });
      return reranked.sort((a, b) => b.score - a.score);
    } catch (error) {
      console.warn("Reranking failed, returning original results:", error);
      return results;
    }
  }

  private applyRecencyBoost(results: RetrievalResult[]): RetrievalResult[] {
    const { recencyHalfLifeDays, recencyWeight } = this.config;
    if (!recencyHalfLifeDays || recencyHalfLifeDays <= 0 || !recencyWeight) return results;

    const now = Date.now();
    const boosted = results.map(r => {
      const ts = (r.entry.timestamp && r.entry.timestamp > 0) ? r.entry.timestamp : now;
      const ageDays = (now - ts) / 86_400_000;
      const boost = Math.exp(-ageDays / recencyHalfLifeDays) * recencyWeight;
      return { ...r, score: clamp01(r.score + boost, r.score) };
    });

    return boosted.sort((a, b) => b.score - a.score);
  }

  private applyImportanceWeight(results: RetrievalResult[]): RetrievalResult[] {
    const baseWeight = 0.7;
    const weighted = results.map(r => {
      const importance = r.entry.importance ?? 0.7;
      const factor = baseWeight + (1 - baseWeight) * importance;
      return { ...r, score: clamp01(r.score * factor, r.score * baseWeight) };
    });
    return weighted.sort((a, b) => b.score - a.score);
  }

  private applyLengthNormalization(results: RetrievalResult[]): RetrievalResult[] {
    const anchor = this.config.lengthNormAnchor;
    if (!anchor || anchor <= 0) return results;

    const normalized = results.map(r => {
      const charLen = r.entry.text.length;
      const ratio = charLen / anchor;
      const logRatio = Math.log2(Math.max(ratio, 1));
      const factor = 1 / (1 + 0.5 * logRatio);
      return { ...r, score: clamp01(r.score * factor, r.score * 0.3) };
    });

    return normalized.sort((a, b) => b.score - a.score);
  }

  private applyTimeDecay(results: RetrievalResult[]): RetrievalResult[] {
    const halfLife = this.config.timeDecayHalfLifeDays;
    if (!halfLife || halfLife <= 0) return results;

    const now = Date.now();
    const decayed = results.map(r => {
      const ts = (r.entry.timestamp && r.entry.timestamp > 0) ? r.entry.timestamp : now;
      const ageDays = (now - ts) / 86_400_000;
      const factor = 0.5 + 0.5 * Math.exp(-ageDays / halfLife);
      return { ...r, score: clamp01(r.score * factor, r.score * 0.5) };
    });

    return decayed.sort((a, b) => b.score - a.score);
  }

  private applyMMRDiversity(results: RetrievalResult[], similarityThreshold = 0.85): RetrievalResult[] {
    if (results.length <= 1) return results;

    const selected: RetrievalResult[] = [];
    const deferred: RetrievalResult[] = [];

    for (const candidate of results) {
      const tooSimilar = selected.some(s => {
        const sVec = s.entry.vector;
        const cVec = candidate.entry.vector;
        if (!sVec?.length || !cVec?.length) return false;
        const sArr = Array.from(sVec as Iterable<number>);
        const cArr = Array.from(cVec as Iterable<number>);
        return cosineSimilarity(sArr, cArr) > similarityThreshold;
      });

      if (tooSimilar) deferred.push(candidate);
      else selected.push(candidate);
    }
    return [...selected, ...deferred];
  }

  updateConfig(newConfig: Partial<RetrievalConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  getConfig(): RetrievalConfig {
    return { ...this.config };
  }

  async test(query = "test query"): Promise<{
    success: boolean; mode: string; hasFtsSupport: boolean; error?: string;
  }> {
    try {
      await this.retrieve({ query, limit: 1 });
      return { success: true, mode: this.config.mode, hasFtsSupport: this.store.hasFtsSupport };
    } catch (error) {
      return {
        success: false,
        mode: this.config.mode,
        hasFtsSupport: this.store.hasFtsSupport,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createRetriever(
  store: MemoryStore,
  embedder: Embedder,
  config?: Partial<RetrievalConfig>,
  voyageApiKey?: string,
): MemoryRetriever {
  const fullConfig = { ...DEFAULT_RETRIEVAL_CONFIG, ...config };
  return new MemoryRetriever(store, embedder, fullConfig, voyageApiKey);
}
