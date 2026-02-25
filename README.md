# Vidya 🧠

**Hybrid Memory for OpenClaw Agents**

> *विद्या (Vidya) — Sanskrit for "knowledge"*

A memory plugin for [OpenClaw](https://github.com/openclaw/openclaw) that gives AI agents persistent long-term memory with hybrid retrieval. Combines LanceDB vector search with BM25 full-text search, Voyage AI embeddings and reranking, multi-scope isolation, and intelligent auto-capture.

## Features

- **Hybrid Search** — Vector similarity + BM25 keyword search fused via Reciprocal Rank Fusion (RRF)
- **Voyage AI Integration** — Embeddings (`voyage-3-large`) and cross-encoder reranking (`rerank-2`) from a single API key
- **Multi-Scope Isolation** — Separate memory spaces per agent, project, or user with access control
- **Auto-Capture** — Automatically stores important information from conversations using LLM judgment (with heuristic fallback)
- **Auto-Recall** — Injects relevant memories into agent context before each turn
- **Noise Filtering** — Filters out agent denials, meta-questions, and boilerplate
- **MMR Diversity** — Maximal Marginal Relevance deduplication prevents redundant results
- **Post-Processing Pipeline** — Recency boost, importance weighting, length normalization, time decay
- **Daily Backups** — Automatic JSONL exports with 7-day rotation
- **CLI Tools** — List, search, export, import, re-embed, and migrate memories
- **6 Agent Tools** — `memory_recall`, `memory_store`, `memory_forget`, `memory_update`, `memory_stats`, `memory_list`

## Quick Start

### 1. Install

```bash
cd ~/.openclaw/plugins
git clone https://github.com/homunculux/Vidya.git memory-lancedb-voyage
cd memory-lancedb-voyage
npm install
```

### 2. Configure

Add to your `openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-lancedb-voyage"
    },
    "entries": {
      "memory-lancedb-voyage": {
        "enabled": true,
        "config": {
          "embedding": {
            "apiKey": "${VOYAGE_API_KEY}",
            "model": "voyage-3-large"
          },
          "autoCapture": true,
          "autoRecall": true,
          "retrieval": {
            "mode": "hybrid",
            "rerank": "cross-encoder"
          }
        }
      }
    }
  }
}
```

### 3. Set API Key

```bash
export VOYAGE_API_KEY="pa-..."
```

Get your key at [dash.voyageai.com](https://dash.voyageai.com/).

### 4. Restart OpenClaw

```bash
openclaw gateway restart
```

See [`config.example.json`](config.example.json) for all available options.

## Configuration

| Section | Key Options | Description |
|---------|-------------|-------------|
| `embedding` | `apiKey`, `model`, `dimensions` | Voyage AI embedding model. Supports `voyage-3-large` (1024d), `voyage-3`, `voyage-3-lite` (512d), `voyage-code-3` |
| `retrieval` | `mode`, `rerank`, `minScore`, `hardMinScore` | `hybrid` (vector+BM25) or `vector` only. Rerank: `cross-encoder`, `lightweight`, or `none` |
| `autoCapture` | `captureLlm`, `captureLlmModel` | LLM judges capture-worthiness via OpenClaw gateway. Falls back to regex heuristic if LLM unavailable |
| `scopes` | `default`, `definitions`, `agentAccess` | Memory isolation. Define scopes and restrict agent access |
| `sessionMemory` | `enabled`, `messageCount` | Store session summaries on `/new` command |
| `enableManagementTools` | — | Enables `memory_stats` and `memory_list` tools |

## Retrieval Pipeline

```
Query
 ├─ Adaptive Skip Check (greetings, commands, short text → skip)
 ├─ Vector Search (LanceDB ANN, cosine distance)
 └─ BM25 Full-Text Search (LanceDB FTS index)
      │
      ├─ RRF Fusion (weighted combination)
      ├─ Voyage Rerank (cross-encoder, blended 60/40 with fusion score)
      ├─ Recency Boost (exponential decay, configurable half-life)
      ├─ Importance Weighting (per-memory importance score)
      ├─ Length Normalization (penalize overly long entries)
      ├─ Time Decay (gradual score reduction for old memories, floor 0.5x)
      ├─ Hard Min Score Filter (discard low-confidence results)
      ├─ Noise Filter (remove denials, meta-questions, boilerplate)
      └─ MMR Diversity (deduplicate similar results, cosine threshold 0.85)
```

## Agent Tools

| Tool | Description |
|------|-------------|
| `memory_recall` | Search memories with hybrid retrieval. Supports scope/category filters. |
| `memory_store` | Save information with category, importance, and scope. Deduplicates against existing memories. |
| `memory_forget` | Delete by ID or search query. Shows candidates for ambiguous matches. |
| `memory_update` | Update text, importance, or category in-place. Supports ID prefix matching. |
| `memory_stats` | Memory count by scope and category, retrieval config info. *(requires `enableManagementTools`)* |
| `memory_list` | List recent memories with scope/category/offset filters. *(requires `enableManagementTools`)* |

## CLI

```bash
# List memories
openclaw memory list [--scope global] [--category fact] [--limit 20]

# Search
openclaw memory search "your query" [--limit 5]

# Stats
openclaw memory stats

# Delete
openclaw memory delete <memory-id>

# Export to JSONL
openclaw memory export [--output memories.jsonl]

# Import from JSONL
openclaw memory import <file.jsonl>

# Re-embed all memories (after model change)
openclaw memory reembed [--batch-size 10]

# Migrate from legacy DB
openclaw memory migrate <old-db-path>
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module dependency map.

```
index.ts          → Plugin entry, lifecycle hooks, auto-capture/recall
src/config.ts     → Config parser and validation
src/embedder.ts   → Voyage AI embedding (native fetch, no SDK)
src/store.ts      → LanceDB storage (vector + BM25 search)
src/retriever.ts  → Hybrid retrieval, RRF fusion, reranking, post-processing
src/scopes.ts     → Multi-scope access control
src/tools.ts      → Agent tool definitions
src/noise-filter.ts → Low-quality memory filtering
src/adaptive-retrieval.ts → Skip retrieval for trivial queries
src/migrate.ts    → Legacy DB migration
cli.ts            → CLI commands
```

## Benchmarks

Tested with 32 integration tests on production data:

| Metric | Score |
|--------|-------|
| Exact noun recall | 92% |
| Semantic recall | 100% |
| Hybrid recall | 100% |
| Avg retrieval latency | ~400ms |

## Known Issues

### jiti Cache

OpenClaw loads `.ts` files via jiti. After modifying source files, clear the cache:

```bash
rm -rf /tmp/jiti/
openclaw gateway restart
```

### Vector Dimension Lock

Once a database is created with a specific embedding model, changing models requires a new `dbPath` or re-embedding all memories via `openclaw memory reembed`.

## Dependencies

- [`@lancedb/lancedb`](https://github.com/lancedb/lancedb) — Vector database (embedded, no server needed)
- [`@sinclair/typebox`](https://github.com/sinclairzx81/typebox) — Tool parameter schemas
- [Voyage AI API](https://voyageai.com/) — Embeddings and reranking (API key required)
- No `openai` package — all API calls use native `fetch()`

## License

[MIT](LICENSE) © 2026 homunculux
