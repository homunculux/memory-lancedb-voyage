/**
 * Integration Tests for memory-lancedb-voyage
 *
 * Uses real Voyage AI API for embedding/rerank tests.
 * Uses temp directories for LanceDB storage (cleaned up after).
 *
 * Run: npx tsx --test test/integration.test.ts
 */

import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { Embedder } from "../src/embedder.js";
import { MemoryStore, type MemoryEntry } from "../src/store.js";
import { MemoryRetriever, createRetriever, DEFAULT_RETRIEVAL_CONFIG } from "../src/retriever.js";
import { isNoise, filterNoise } from "../src/noise-filter.js";
import { shouldSkipRetrieval } from "../src/adaptive-retrieval.js";

// ============================================================================
// Setup: Voyage API key from environment
// ============================================================================

const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;
if (!VOYAGE_API_KEY) {
  console.error("VOYAGE_API_KEY is not set. Run with:\n  source .openclaw/1password-config.sh && export VOYAGE_API_KEY=...");
  process.exit(1);
}

const VECTOR_DIM = 1024;
const MODEL = "voyage-3-large";

// Shared embedder (reused across tests to minimize API calls)
let embedder: Embedder;

before(() => {
  embedder = new Embedder({ apiKey: VOYAGE_API_KEY, model: MODEL });
});

// Helper: create a temp LanceDB directory
async function makeTempDb(): Promise<string> {
  return mkdtemp(join(tmpdir(), "lance-test-"));
}

// Helper: clean up a temp directory
async function cleanupDir(dir: string): Promise<void> {
  await rm(dir, { recursive: true, force: true });
}

// ============================================================================
// 1. Embedder Tests
// ============================================================================

describe("Embedder", () => {
  it("should generate an embedding with dimension=1024", async () => {
    const vector = await embedder.embedPassage("TypeScript is a typed superset of JavaScript.");
    assert.equal(vector.length, VECTOR_DIM, `Expected ${VECTOR_DIM} dimensions, got ${vector.length}`);
    // Verify it's a real numeric vector
    assert.ok(vector.every(v => typeof v === "number" && Number.isFinite(v)), "All elements should be finite numbers");
    // Verify it's not all zeros
    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    assert.ok(norm > 0.5, `Vector norm should be substantial, got ${norm}`);
  });

  it("should produce different embeddings for query vs passage input_type", async () => {
    const text = "What is the capital of France?";
    const queryVec = await embedder.embedQuery(text);
    const passageVec = await embedder.embedPassage(text);

    assert.equal(queryVec.length, VECTOR_DIM);
    assert.equal(passageVec.length, VECTOR_DIM);

    // They should differ since Voyage uses different input_type for query vs document
    let diffCount = 0;
    for (let i = 0; i < VECTOR_DIM; i++) {
      if (Math.abs(queryVec[i] - passageVec[i]) > 1e-6) diffCount++;
    }
    assert.ok(diffCount > 100, `Query and passage vectors should differ significantly (${diffCount} dimensions differ)`);
  });

  it("should batch embed multiple texts", async () => {
    const texts = [
      "The sky is blue.",
      "Rust is a systems programming language.",
      "Coffee is a popular beverage worldwide.",
    ];
    const vectors = await embedder.embedBatchPassage(texts);

    assert.equal(vectors.length, 3, "Should return 3 vectors");
    for (const vec of vectors) {
      assert.equal(vec.length, VECTOR_DIM, `Each vector should have ${VECTOR_DIM} dims`);
    }

    // Verify they're all different from each other
    for (let i = 0; i < vectors.length; i++) {
      for (let j = i + 1; j < vectors.length; j++) {
        const cos = cosineSim(vectors[i], vectors[j]);
        assert.ok(cos < 0.99, `Vectors ${i} and ${j} should be different (cosine=${cos.toFixed(4)})`);
      }
    }
  });

  it("should pass the test() self-check", async () => {
    const result = await embedder.test();
    assert.ok(result.success, `Embedder test failed: ${result.error}`);
    assert.equal(result.dimensions, VECTOR_DIM);
  });

  it("should use cache for repeated embeddings", async () => {
    const text = "Cache test: identical text should hit cache.";
    await embedder.embedPassage(text);
    const statsBefore = embedder.cacheStats;
    await embedder.embedPassage(text);
    const statsAfter = embedder.cacheStats;
    assert.ok(statsAfter.hits > statsBefore.hits, "Cache hits should increase on repeated embed");
  });
});

// ============================================================================
// 2. Store Tests
// ============================================================================

describe("Store", () => {
  let tmpDir: string;
  let store: MemoryStore;

  before(async () => {
    tmpDir = await makeTempDb();
    store = new MemoryStore({ dbPath: tmpDir, vectorDim: VECTOR_DIM });
  });

  after(async () => {
    await cleanupDir(tmpDir);
  });

  it("should store 5 entries and persist them", async () => {
    const texts = [
      "User prefers dark mode in all applications.",
      "Project uses PostgreSQL 16 as the primary database.",
      "Team decided to use Bun as the JavaScript runtime.",
      "User's email address is test@example.com.",
      "Always run lint before committing code.",
    ];

    const categories: MemoryEntry["category"][] = ["preference", "fact", "decision", "entity", "other"];
    const storedIds: string[] = [];

    for (let i = 0; i < texts.length; i++) {
      const vector = await embedder.embedPassage(texts[i]);
      const entry = await store.store({
        text: texts[i],
        vector,
        category: categories[i],
        scope: "global",
        importance: 0.5 + i * 0.1,
      });
      assert.ok(entry.id, "Entry should have a UUID");
      assert.ok(entry.timestamp > 0, "Entry should have a timestamp");
      storedIds.push(entry.id);
    }

    // Verify persistence via list
    const listed = await store.list(["global"]);
    assert.ok(listed.length >= 5, `Expected at least 5 entries, got ${listed.length}`);

    // Verify stats
    const stats = await store.stats(["global"]);
    assert.ok(stats.totalCount >= 5, `Stats should show >= 5 entries, got ${stats.totalCount}`);
    assert.ok(stats.categoryCounts["preference"] >= 1, "Should have at least 1 preference");
    assert.ok(stats.categoryCounts["fact"] >= 1, "Should have at least 1 fact");

    // Verify hasId
    const exists = await store.hasId(storedIds[0]);
    assert.ok(exists, "First stored ID should exist");
  });

  it("should support vector search", async () => {
    const queryVec = await embedder.embedQuery("What database does the project use?");
    const results = await store.vectorSearch(queryVec, 3, 0.1, ["global"]);

    assert.ok(results.length > 0, "Vector search should return results");
    // The PostgreSQL entry should be highly ranked
    const topTexts = results.map(r => r.entry.text);
    const pgResult = results.find(r => r.entry.text.includes("PostgreSQL"));
    assert.ok(pgResult, `PostgreSQL entry should appear in results. Got: ${topTexts.join(" | ")}`);
  });

  it("should update an entry", async () => {
    const listed = await store.list(["global"], "preference", 1);
    assert.ok(listed.length > 0, "Should have a preference entry to update");

    const id = listed[0].id;
    const newVector = await embedder.embedPassage("User prefers light mode in all editors.");
    const updated = await store.update(id, {
      text: "User prefers light mode in all editors.",
      vector: newVector,
    });

    assert.ok(updated, "Update should return the updated entry");
    assert.equal(updated!.text, "User prefers light mode in all editors.");
    assert.equal(updated!.id, id, "ID should remain the same");
  });

  it("should delete an entry by ID", async () => {
    const listed = await store.list(["global"], "other", 1);
    assert.ok(listed.length > 0, "Should have an 'other' entry to delete");

    const id = listed[0].id;
    const deleted = await store.delete(id, ["global"]);
    assert.ok(deleted, "Delete should return true");

    const stillExists = await store.hasId(id);
    assert.ok(!stillExists, "Deleted entry should no longer exist");
  });
});

// ============================================================================
// 3. Retriever Tests (hybrid retrieval with vector + BM25)
// ============================================================================

describe("Retriever", () => {
  let tmpDir: string;
  let store: MemoryStore;
  let retriever: MemoryRetriever;

  before(async () => {
    tmpDir = await makeTempDb();
    store = new MemoryStore({ dbPath: tmpDir, vectorDim: VECTOR_DIM });

    retriever = createRetriever(store, embedder, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      rerank: "none",          // Disable reranking for baseline retrieval test
      hardMinScore: 0.1,       // Lower threshold for test
      minScore: 0.1,
      filterNoise: false,      // Don't filter noise in this test
    }, VOYAGE_API_KEY);

    // Seed with diverse entries
    const entries = [
      { text: "User prefers vim keybindings in VS Code.", category: "preference" as const, importance: 0.9 },
      { text: "The deployment pipeline uses GitHub Actions with Docker.", category: "fact" as const, importance: 0.8 },
      { text: "Team decided to migrate from REST to GraphQL.", category: "decision" as const, importance: 0.85 },
      { text: "The main database is PostgreSQL 16 hosted on AWS RDS.", category: "fact" as const, importance: 0.7 },
      { text: "User's preferred programming language is TypeScript.", category: "preference" as const, importance: 0.9 },
    ];

    for (const e of entries) {
      const vector = await embedder.embedPassage(e.text);
      await store.store({ ...e, vector, scope: "global" });
    }
  });

  after(async () => {
    await cleanupDir(tmpDir);
  });

  it("should retrieve relevant results for a semantic query", async () => {
    const results = await retriever.retrieve({
      query: "What editor keybindings does the user prefer?",
      limit: 5,
      scopeFilter: ["global"],
    });

    assert.ok(results.length > 0, "Should return results");

    // The vim keybindings entry should be in the top results
    const topResult = results[0];
    assert.ok(
      topResult.entry.text.includes("vim") || topResult.entry.text.includes("keybinding"),
      `Top result should be about vim keybindings, got: "${topResult.entry.text}"`,
    );
  });

  it("should return results sorted by score descending", async () => {
    const results = await retriever.retrieve({
      query: "What technologies does the team use?",
      limit: 5,
      scopeFilter: ["global"],
    });

    assert.ok(results.length >= 2, "Should return multiple results");

    for (let i = 1; i < results.length; i++) {
      assert.ok(
        results[i - 1].score >= results[i].score,
        `Results should be sorted by score: ${results[i - 1].score} >= ${results[i].score}`,
      );
    }
  });

  it("should filter by category", async () => {
    const results = await retriever.retrieve({
      query: "user preferences",
      limit: 5,
      scopeFilter: ["global"],
      category: "preference",
    });

    for (const r of results) {
      assert.equal(r.entry.category, "preference", `All results should be 'preference', got '${r.entry.category}'`);
    }
  });

  it("should include source tracking in results", async () => {
    const results = await retriever.retrieve({
      query: "database hosting",
      limit: 3,
      scopeFilter: ["global"],
    });

    assert.ok(results.length > 0, "Should return results");
    const r = results[0];
    assert.ok(r.sources, "Result should have sources");
    assert.ok(r.sources.vector || r.sources.bm25 || r.sources.fused, "Should have at least one source type");
  });
});

// ============================================================================
// 4. Reranking Tests
// ============================================================================

describe("Reranking", () => {
  let tmpDir: string;
  let store: MemoryStore;

  before(async () => {
    tmpDir = await makeTempDb();
    store = new MemoryStore({ dbPath: tmpDir, vectorDim: VECTOR_DIM });

    // Seed entries where semantic similarity and keyword overlap diverge
    const entries = [
      // Decoy: lots of keyword overlap with "python programming language" but different meaning
      { text: "A python is a large nonvenomous snake found in Asia, Africa, and Australia.", category: "fact" as const, importance: 0.7 },
      // Target: actual answer about Python the programming language
      { text: "Guido van Rossum created the Python programming language in 1991.", category: "fact" as const, importance: 0.7 },
      // Another decoy
      { text: "Ball pythons are popular pet reptiles known for their docile temperament.", category: "fact" as const, importance: 0.7 },
      // Partially relevant
      { text: "TypeScript was created by Microsoft as a superset of JavaScript.", category: "fact" as const, importance: 0.7 },
    ];

    for (const e of entries) {
      const vector = await embedder.embedPassage(e.text);
      await store.store({ ...e, vector, scope: "global" });
    }
  });

  after(async () => {
    await cleanupDir(tmpDir);
  });

  it("should produce results with reranked scores when cross-encoder is enabled", async () => {
    const retrieverWithRerank = createRetriever(store, embedder, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      rerank: "cross-encoder",
      rerankModel: "rerank-2",
      hardMinScore: 0.05,
      minScore: 0.05,
      filterNoise: false,
    }, VOYAGE_API_KEY);

    const results = await retrieverWithRerank.retrieve({
      query: "Who created the Python programming language?",
      limit: 4,
      scopeFilter: ["global"],
    });

    assert.ok(results.length > 0, "Should return results");

    // Check that reranked source scores exist
    const rerankedResults = results.filter(r => r.sources.reranked);
    assert.ok(rerankedResults.length > 0, "At least some results should have reranked scores");

    // The Guido van Rossum entry should ideally be #1 after reranking
    const guidoResult = results.find(r => r.entry.text.includes("Guido"));
    assert.ok(guidoResult, "Guido van Rossum entry should appear in results");

    // Verify the top result is about the programming language, not the snake
    const topResult = results[0];
    assert.ok(
      topResult.entry.text.includes("Guido") || topResult.entry.text.includes("programming"),
      `Top reranked result should be about Python the language, got: "${topResult.entry.text}"`,
    );
  });

  it("lightweight reranking should also produce reranked scores", async () => {
    const retrieverLightweight = createRetriever(store, embedder, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      rerank: "lightweight",
      hardMinScore: 0.05,
      minScore: 0.05,
      filterNoise: false,
    }, VOYAGE_API_KEY);

    const results = await retrieverLightweight.retrieve({
      query: "Who created the Python programming language?",
      limit: 4,
      scopeFilter: ["global"],
    });

    assert.ok(results.length > 0, "Should return results");
    const rerankedResults = results.filter(r => r.sources.reranked);
    assert.ok(rerankedResults.length > 0, "Lightweight reranking should produce reranked source scores");
  });
});

// ============================================================================
// 5. Noise Filter Tests
// ============================================================================

describe("Noise Filter", () => {
  it("should detect greetings as noise", () => {
    assert.ok(isNoise("Hello"), "Hello should be noise");
    assert.ok(isNoise("Hi there"), "Hi there should be noise");
    assert.ok(isNoise("hey"), "hey should be noise");
    assert.ok(isNoise("Good morning everyone"), "Good morning should be noise");
  });

  it("should detect denials as noise", () => {
    assert.ok(isNoise("I don't have any information about that topic."), "Denial should be noise");
    assert.ok(isNoise("I wasn't able to find anything relevant."), "Unable to find should be noise");
    assert.ok(isNoise("No relevant memories found for your query."), "No memories should be noise");
    assert.ok(isNoise("I don't recall that detail."), "Don't recall should be noise");
  });

  it("should detect meta-questions as noise", () => {
    assert.ok(isNoise("Do you remember what I said yesterday?"), "Meta-question should be noise");
    assert.ok(isNoise("Can you recall my preferences?"), "Recall question should be noise");
    assert.ok(isNoise("Did I mention my email address?"), "Mention question should be noise");
  });

  it("should detect boilerplate as noise", () => {
    assert.ok(isNoise("fresh session starting up"), "Fresh session should be noise");
    assert.ok(isNoise("HEARTBEAT check"), "HEARTBEAT should be noise");
    assert.ok(isNoise("new session begins"), "New session should be noise");
  });

  it("should NOT flag legitimate content as noise", () => {
    assert.ok(!isNoise("User prefers dark mode in VS Code."), "Preference should not be noise");
    assert.ok(!isNoise("The project uses PostgreSQL for data storage."), "Fact should not be noise");
    assert.ok(!isNoise("Team decided to switch from REST to GraphQL."), "Decision should not be noise");
    assert.ok(!isNoise("Contact email: user@example.com for support inquiries."), "Entity should not be noise");
  });

  it("should detect very short text as noise", () => {
    assert.ok(isNoise("ok"), "Very short text should be noise");
    assert.ok(isNoise("hi"), "Very short greeting should be noise");
    assert.ok(isNoise("yes"), "Very short confirmation should be noise");
  });

  it("should filter noise from an array of items", () => {
    const items = [
      { id: 1, text: "User prefers TypeScript over JavaScript." },
      { id: 2, text: "Hello there!" },
      { id: 3, text: "The API endpoint is /api/v2/users." },
      { id: 4, text: "I don't have any information about that." },
      { id: 5, text: "HEARTBEAT signal received." },
    ];

    const filtered = filterNoise(items, item => item.text);
    const filteredIds = filtered.map(item => item.id);

    assert.ok(filteredIds.includes(1), "Preference should survive filtering");
    assert.ok(filteredIds.includes(3), "API endpoint fact should survive filtering");
    assert.ok(!filteredIds.includes(2), "Greeting should be filtered out");
    assert.ok(!filteredIds.includes(4), "Denial should be filtered out");
    assert.ok(!filteredIds.includes(5), "HEARTBEAT should be filtered out");
  });
});

// ============================================================================
// 6. Adaptive Retrieval Tests
// ============================================================================

describe("Adaptive Retrieval", () => {
  it("should skip retrieval for greetings", () => {
    assert.ok(shouldSkipRetrieval("hello"), "hello should skip");
    assert.ok(shouldSkipRetrieval("Hi"), "Hi should skip");
    assert.ok(shouldSkipRetrieval("hey"), "hey should skip");
    assert.ok(shouldSkipRetrieval("Good morning"), "Good morning should skip");
    assert.ok(shouldSkipRetrieval("yo"), "yo should skip");
  });

  it("should skip retrieval for commands", () => {
    assert.ok(shouldSkipRetrieval("/help"), "Slash commands should skip");
    assert.ok(shouldSkipRetrieval("git status"), "git commands should skip");
    assert.ok(shouldSkipRetrieval("npm install"), "npm commands should skip");
    assert.ok(shouldSkipRetrieval("ls -la"), "ls should skip");
  });

  it("should skip retrieval for confirmations", () => {
    assert.ok(shouldSkipRetrieval("yes"), "yes should skip");
    assert.ok(shouldSkipRetrieval("no"), "no should skip");
    assert.ok(shouldSkipRetrieval("ok"), "ok should skip");
    assert.ok(shouldSkipRetrieval("thanks"), "thanks should skip");
    assert.ok(shouldSkipRetrieval("sure"), "sure should skip");
    assert.ok(shouldSkipRetrieval("go ahead"), "go ahead should skip");
  });

  it("should skip retrieval for emoji-only messages", () => {
    assert.ok(shouldSkipRetrieval("\u{1F44D}"), "Thumbs up should skip");
    assert.ok(shouldSkipRetrieval("\u{1F44D}\u{1F44D}"), "Double thumbs up should skip");
  });

  it("should skip retrieval for short non-question text", () => {
    assert.ok(shouldSkipRetrieval("fix bug"), "Short text without ? should skip");
    assert.ok(shouldSkipRetrieval("do it"), "do it should skip");
  });

  it("should NOT skip retrieval for memory-related queries", () => {
    assert.ok(!shouldSkipRetrieval("do you remember my email?"), "Memory query should not skip");
    assert.ok(!shouldSkipRetrieval("what did I tell you last time?"), "Last time query should not skip");
    assert.ok(!shouldSkipRetrieval("recall my preferences"), "Recall query should not skip");
    assert.ok(!shouldSkipRetrieval("what is my name?"), "My name query should not skip");
  });

  it("should NOT skip retrieval for substantive questions", () => {
    assert.ok(!shouldSkipRetrieval("How do I configure the database connection pooling?"), "Substantive question should not skip");
    assert.ok(!shouldSkipRetrieval("What is the best practice for error handling in TypeScript?"), "Best practice question should not skip");
    assert.ok(!shouldSkipRetrieval("Explain the differences between REST and GraphQL APIs."), "Explanation request should not skip");
  });

  it("should handle HEARTBEAT signals", () => {
    assert.ok(shouldSkipRetrieval("HEARTBEAT"), "HEARTBEAT should skip");
    assert.ok(shouldSkipRetrieval("HEARTBEAT check"), "HEARTBEAT check should skip");
  });
});

// ============================================================================
// 7. JSONL Backup Tests
// ============================================================================

describe("JSONL Backup", () => {
  let tmpDir: string;
  let store: MemoryStore;
  let backupDir: string;

  before(async () => {
    tmpDir = await makeTempDb();
    backupDir = join(tmpDir, "backups");
    store = new MemoryStore({ dbPath: tmpDir, vectorDim: VECTOR_DIM });

    // Seed some entries
    const entries = [
      { text: "User prefers dark mode.", category: "preference" as const, importance: 0.8 },
      { text: "Project uses TypeScript.", category: "fact" as const, importance: 0.7 },
      { text: "API key stored in environment variables.", category: "decision" as const, importance: 0.6 },
    ];

    for (const e of entries) {
      const vector = await embedder.embedPassage(e.text);
      await store.store({ ...e, vector, scope: "global" });
    }
  });

  after(async () => {
    await cleanupDir(tmpDir);
  });

  it("should create a valid JSONL backup file", async () => {
    // Replicate the backup logic from index.ts
    const { mkdir: mkdirFs, writeFile: writeFileFs, readdir: readdirFs } = await import("node:fs/promises");
    await mkdirFs(backupDir, { recursive: true });

    const allMemories = await store.list(undefined, undefined, 10000, 0);
    assert.ok(allMemories.length >= 3, `Expected at least 3 entries, got ${allMemories.length}`);

    const dateStr = new Date().toISOString().split("T")[0];
    const backupFile = join(backupDir, `memory-backup-${dateStr}.jsonl`);

    const lines = allMemories.map(m => JSON.stringify({
      id: m.id,
      text: m.text,
      category: m.category,
      scope: m.scope,
      importance: m.importance,
      timestamp: m.timestamp,
      metadata: m.metadata,
    }));

    await writeFileFs(backupFile, lines.join("\n") + "\n");

    // Verify backup file exists
    const files = await readdirFs(backupDir);
    const backupFiles = files.filter(f => f.startsWith("memory-backup-") && f.endsWith(".jsonl"));
    assert.ok(backupFiles.length >= 1, "Should have at least 1 backup file");

    // Verify JSONL content
    const content = await readFile(backupFile, "utf-8");
    const jsonLines = content.trim().split("\n");
    assert.ok(jsonLines.length >= 3, `Backup should have at least 3 lines, got ${jsonLines.length}`);

    // Verify each line is valid JSON with expected fields
    for (const line of jsonLines) {
      const parsed = JSON.parse(line);
      assert.ok(parsed.id, "Each line should have an id");
      assert.ok(parsed.text, "Each line should have text");
      assert.ok(parsed.category, "Each line should have category");
      assert.ok(parsed.scope, "Each line should have scope");
      assert.ok(typeof parsed.importance === "number", "Each line should have numeric importance");
      assert.ok(typeof parsed.timestamp === "number", "Each line should have numeric timestamp");
    }
  });

  it("should include all required fields in backup entries", async () => {
    const allMemories = await store.list(undefined, undefined, 10000, 0);
    const firstMemory = allMemories[0];

    const backupEntry = JSON.parse(JSON.stringify({
      id: firstMemory.id,
      text: firstMemory.text,
      category: firstMemory.category,
      scope: firstMemory.scope,
      importance: firstMemory.importance,
      timestamp: firstMemory.timestamp,
      metadata: firstMemory.metadata,
    }));

    // UUID format check
    assert.match(backupEntry.id, /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i, "ID should be a valid UUID");
    assert.ok(backupEntry.text.length > 0, "Text should be non-empty");
    assert.ok(["preference", "fact", "decision", "entity", "other"].includes(backupEntry.category), "Category should be valid");
    assert.equal(backupEntry.scope, "global", "Scope should be global");
    assert.ok(backupEntry.importance >= 0 && backupEntry.importance <= 1, "Importance should be 0-1");
    assert.ok(backupEntry.timestamp > 0, "Timestamp should be positive");
  });
});

// ============================================================================
// Utility: cosine similarity
// ============================================================================

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
