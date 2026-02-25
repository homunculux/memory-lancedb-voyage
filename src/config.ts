/**
 * Configuration for memory-lancedb-voyage plugin.
 * Uses a manual parse() pattern matching the built-in memory-lancedb config.ts.
 */

import fs from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// ============================================================================
// Types
// ============================================================================

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

export type EmbeddingProvider = "voyage" | "openai" | "jina";

export interface PluginConfig {
  embedding: {
    provider: EmbeddingProvider;
    apiKey: string;
    model: string;
    dimensions?: number;
    baseUrl?: string;
  };
  dbPath: string;
  autoCapture: boolean;
  autoRecall: boolean;
  captureAssistant: boolean;
  captureMaxChars: number;
  captureLlm: boolean;
  captureLlmModel: string;
  captureLlmUrl: string;
  enableManagementTools: boolean;
  retrieval: {
    mode: "hybrid" | "vector";
    vectorWeight: number;
    bm25Weight: number;
    minScore: number;
    rerank: "cross-encoder" | "lightweight" | "none";
    rerankModel: string;
    candidatePoolSize: number;
    recencyHalfLifeDays: number;
    recencyWeight: number;
    filterNoise: boolean;
    lengthNormAnchor: number;
    hardMinScore: number;
    timeDecayHalfLifeDays: number;
  };
  sessionMemory: { enabled: boolean; messageCount: number };
  scopes?: {
    default?: string;
    definitions?: Record<string, { description: string }>;
    agentAccess?: Record<string, string[]>;
  };
}

// ============================================================================
// Defaults
// ============================================================================

const DEFAULT_MODEL = "voyage-3-large";
const DEFAULT_PROVIDER: EmbeddingProvider = "voyage";
export const DEFAULT_CAPTURE_MAX_CHARS = 500;

const VALID_PROVIDERS = new Set(["voyage", "openai", "jina"]);

// Embedding model dimensions by provider
const EMBEDDING_DIMENSIONS: Record<string, number> = {
  // Voyage AI
  "voyage-3-large": 1024,
  "voyage-3": 1024,
  "voyage-3-lite": 512,
  "voyage-code-3": 1024,
  "voyage-finance-2": 1024,
  "voyage-law-2": 1024,
  "voyage-multilingual-2": 1024,
  // OpenAI
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
  // Jina
  "jina-embeddings-v3": 1024,
  "jina-embeddings-v2-base-en": 768,
};

export function vectorDimsForModel(model: string, overrideDims?: number): number {
  if (overrideDims && overrideDims > 0) {
    return overrideDims;
  }
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(
      `Unknown embedding model: ${model}. Set embedding.dimensions in config. Known models: ${Object.keys(EMBEDDING_DIMENSIONS).join(", ")}`,
    );
  }
  return dims;
}

function resolveDefaultDbPath(): string {
  const home = homedir();
  const preferred = join(home, ".openclaw", "memory", "lancedb-voyage");
  try {
    if (fs.existsSync(preferred)) {
      return preferred;
    }
  } catch {
    // best-effort
  }
  return preferred;
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

// ============================================================================
// Provider-specific env var fallback
// ============================================================================

const PROVIDER_ENV_VARS: Record<EmbeddingProvider, string> = {
  voyage: "VOYAGE_API_KEY",
  openai: "OPENAI_API_KEY",
  jina: "JINA_API_KEY",
};

const PROVIDER_DEFAULT_MODELS: Record<EmbeddingProvider, string> = {
  voyage: "voyage-3-large",
  openai: "text-embedding-3-small",
  jina: "jina-embeddings-v3",
};

// ============================================================================
// Config Parser (matches built-in plugin pattern)
// ============================================================================

export const memoryConfigSchema = {
  parse(value: unknown): PluginConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory-lancedb-voyage config required");
    }
    const cfg = value as Record<string, unknown>;

    // Embedding config (required)
    const embedding = cfg.embedding as Record<string, unknown> | undefined;
    if (!embedding) {
      throw new Error("embedding config is required");
    }

    // Provider (default: "voyage" for backward compatibility)
    const provider = (typeof embedding.provider === "string" && VALID_PROVIDERS.has(embedding.provider)
      ? embedding.provider
      : DEFAULT_PROVIDER) as EmbeddingProvider;

    if (typeof embedding.provider === "string" && !VALID_PROVIDERS.has(embedding.provider)) {
      throw new Error(
        `Unknown embedding provider: ${embedding.provider}. Supported providers: voyage, openai, jina`,
      );
    }

    // API key: try config value, then provider-specific env var, then VOYAGE_API_KEY fallback
    const envVarName = PROVIDER_ENV_VARS[provider];
    const apiKey = typeof embedding.apiKey === "string"
      ? embedding.apiKey
      : process.env[envVarName] || process.env.VOYAGE_API_KEY || "";
    if (!apiKey) {
      throw new Error(`embedding.apiKey is required (set directly or via ${envVarName} env var)`);
    }

    const defaultModel = PROVIDER_DEFAULT_MODELS[provider];
    const model = typeof embedding.model === "string" ? embedding.model : defaultModel;
    const overrideDims = typeof embedding.dimensions === "number" ? embedding.dimensions : undefined;
    const baseUrl = typeof embedding.baseUrl === "string" ? embedding.baseUrl : undefined;
    vectorDimsForModel(model, overrideDims); // validate

    // Retrieval config
    const ret = (typeof cfg.retrieval === "object" && cfg.retrieval !== null
      ? cfg.retrieval
      : {}) as Record<string, unknown>;

    // Session memory config
    const sm = (typeof cfg.sessionMemory === "object" && cfg.sessionMemory !== null
      ? cfg.sessionMemory
      : {}) as Record<string, unknown>;

    const captureMaxChars =
      typeof cfg.captureMaxChars === "number" ? Math.floor(cfg.captureMaxChars) : undefined;
    if (
      typeof captureMaxChars === "number" &&
      (captureMaxChars < 100 || captureMaxChars > 10_000)
    ) {
      throw new Error("captureMaxChars must be between 100 and 10000");
    }

    return {
      embedding: {
        provider,
        apiKey: resolveEnvVars(apiKey),
        model,
        dimensions: overrideDims,
        baseUrl,
      },
      dbPath: typeof cfg.dbPath === "string" ? cfg.dbPath : resolveDefaultDbPath(),
      autoCapture: cfg.autoCapture !== false,
      autoRecall: cfg.autoRecall !== false,
      captureAssistant: cfg.captureAssistant === true,
      captureMaxChars: captureMaxChars ?? DEFAULT_CAPTURE_MAX_CHARS,
      captureLlm: cfg.captureLlm !== false,
      captureLlmModel: typeof cfg.captureLlmModel === "string"
        ? cfg.captureLlmModel
        : "anthropic/claude-haiku-4-5-20251001",
      captureLlmUrl: typeof cfg.captureLlmUrl === "string"
        ? cfg.captureLlmUrl
        : "",
      enableManagementTools: cfg.enableManagementTools === true,
      retrieval: {
        mode: ret.mode === "vector" ? "vector" : "hybrid",
        vectorWeight: typeof ret.vectorWeight === "number" ? ret.vectorWeight : 0.7,
        bm25Weight: typeof ret.bm25Weight === "number" ? ret.bm25Weight : 0.3,
        minScore: typeof ret.minScore === "number" ? ret.minScore : 0.3,
        rerank: (ret.rerank === "lightweight" || ret.rerank === "none") ? ret.rerank : "cross-encoder",
        rerankModel: typeof ret.rerankModel === "string" ? ret.rerankModel : "rerank-2",
        candidatePoolSize: typeof ret.candidatePoolSize === "number" ? ret.candidatePoolSize : 20,
        recencyHalfLifeDays: typeof ret.recencyHalfLifeDays === "number" ? ret.recencyHalfLifeDays : 14,
        recencyWeight: typeof ret.recencyWeight === "number" ? ret.recencyWeight : 0.10,
        filterNoise: ret.filterNoise !== false,
        lengthNormAnchor: typeof ret.lengthNormAnchor === "number" ? ret.lengthNormAnchor : 500,
        hardMinScore: typeof ret.hardMinScore === "number" ? ret.hardMinScore : 0.35,
        timeDecayHalfLifeDays: typeof ret.timeDecayHalfLifeDays === "number" ? ret.timeDecayHalfLifeDays : 60,
      },
      sessionMemory: {
        enabled: sm.enabled === true,
        messageCount: typeof sm.messageCount === "number" ? sm.messageCount : 15,
      },
      scopes: typeof cfg.scopes === "object" && cfg.scopes !== null
        ? cfg.scopes as PluginConfig["scopes"]
        : undefined,
    };
  },
};
