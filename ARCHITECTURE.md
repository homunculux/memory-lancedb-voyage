# Architecture: memory-lancedb-voyage

OpenClaw memory plugin using LanceDB for storage and Voyage AI for embeddings + reranking.

## Module Dependency Map

```
index.ts (Plugin Entry Point)
├── src/config.ts          — Config parser (memoryConfigSchema.parse)
├── src/embedder.ts        — Voyage AI embedding (fetch-based, no OpenAI SDK)
│   └── src/config.ts      — vectorDimsForModel()
├── src/store.ts           — LanceDB storage layer (vector search + BM25)
├── src/retriever.ts       — Hybrid retrieval (RRF fusion, Voyage reranking, MMR)
│   ├── src/store.ts
│   ├── src/embedder.ts
│   └── src/noise-filter.ts
├── src/scopes.ts          — Multi-scope access control
├── src/migrate.ts         — Legacy DB migration
│   └── src/store.ts
├── src/tools.ts           — Agent tools (recall, store, forget, update, stats, list)
│   ├── src/retriever.ts
│   ├── src/store.ts
│   ├── src/scopes.ts
│   ├── src/embedder.ts
│   └── src/noise-filter.ts
├── src/adaptive-retrieval.ts — Skip retrieval for greetings/commands
└── cli.ts                 — CLI commands (list, search, stats, delete, export, import, reembed, migrate)
    ├── src/store.ts
    ├── src/retriever.ts
    ├── src/scopes.ts
    ├── src/migrate.ts
    └── src/embedder.ts
```

## Key Design Decisions

### Voyage AI (vs OpenAI/Jina)
- Single API key (`VOYAGE_API_KEY`) for both embedding and reranking
- Embeddings: `https://api.voyageai.com/v1/embeddings` with `input_type: "query"|"document"`
- Reranking: `https://api.voyageai.com/v1/rerank` with `model: "rerank-2"`
- Uses native `fetch()` — no `openai` package dependency

### Hybrid Retrieval Pipeline
```
Query → Adaptive Skip Check
      → Vector Search (LanceDB ANN)  ─┐
      → BM25 Full-Text Search        ─┤
                                       ├→ RRF Fusion
                                       ├→ Rerank (Voyage cross-encoder or cosine fallback)
                                       ├→ Recency Boost
                                       ├→ Importance Weighting
                                       ├→ Length Normalization
                                       ├→ Time Decay
                                       ├→ Hard Min Score Filter
                                       ├→ Noise Filter
                                       └→ MMR Diversity
```

### Plugin Interface
Matches the built-in `memory-lancedb` plugin pattern:
- `openclaw.plugin.json` with `configSchema` + `uiHints`
- `configSchema.parse()` method on the config object
- `register(api: OpenClawPluginApi)` with tools, CLI, hooks, service
- `kind: "memory"` plugin type

### Dependencies
- `@lancedb/lancedb` — Vector database (dynamic import for graceful failure)
- `@sinclair/typebox` — Tool parameter schemas
- NO `openai` package — Voyage AI accessed via native `fetch()`
