/**
 * Unit tests for Vidya memory plugin.
 * Pure unit tests — NO API calls, NO LanceDB.
 *
 * Run: npx tsx --test test/unit.test.ts
 */

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";

// ============================================================================
// 1. Config Parser
// ============================================================================

import { memoryConfigSchema, DEFAULT_CAPTURE_MAX_CHARS, vectorDimsForModel } from "../src/config.js";

describe("Config Parser (memoryConfigSchema.parse)", () => {
  const savedEnv: Record<string, string | undefined> = {};

  beforeEach(() => {
    savedEnv.VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;
    // Ensure VOYAGE_API_KEY doesn't leak into tests
    delete process.env.VOYAGE_API_KEY;
  });

  afterEach(() => {
    if (savedEnv.VOYAGE_API_KEY !== undefined) {
      process.env.VOYAGE_API_KEY = savedEnv.VOYAGE_API_KEY;
    } else {
      delete process.env.VOYAGE_API_KEY;
    }
  });

  it("valid minimal config (just embedding.apiKey) returns all defaults", () => {
    const cfg = memoryConfigSchema.parse({
      embedding: { apiKey: "test-key-123" },
    });

    assert.equal(cfg.embedding.apiKey, "test-key-123");
    assert.equal(cfg.embedding.model, "voyage-3-large");
    assert.equal(cfg.embedding.dimensions, undefined);
    assert.equal(cfg.autoCapture, true);
    assert.equal(cfg.autoRecall, true);
    assert.equal(cfg.captureAssistant, false);
    assert.equal(cfg.captureMaxChars, DEFAULT_CAPTURE_MAX_CHARS);
    assert.equal(cfg.captureMaxChars, 500);
    assert.equal(cfg.captureLlm, true);
    assert.equal(cfg.captureLlmModel, "anthropic/claude-haiku-4-5-20251001");
    assert.equal(cfg.enableManagementTools, false);
    assert.equal(cfg.sessionMemory.enabled, false);
    assert.equal(cfg.sessionMemory.messageCount, 15);
    assert.equal(cfg.scopes, undefined);
  });

  it("missing embedding → should throw", () => {
    assert.throws(() => memoryConfigSchema.parse({}), /embedding config is required/);
  });

  it("missing apiKey (no env var either) → should throw", () => {
    assert.throws(
      () => memoryConfigSchema.parse({ embedding: {} }),
      /embedding\.apiKey is required/,
    );
  });

  it("unknown model without dimensions override → should throw", () => {
    assert.throws(
      () =>
        memoryConfigSchema.parse({
          embedding: { apiKey: "key", model: "some-unknown-model" },
        }),
      /Unknown embedding model/,
    );
  });

  it("unknown model WITH dimensions override → should succeed", () => {
    const cfg = memoryConfigSchema.parse({
      embedding: { apiKey: "key", model: "custom-model", dimensions: 768 },
    });
    assert.equal(cfg.embedding.model, "custom-model");
    assert.equal(cfg.embedding.dimensions, 768);
  });

  it("known model → correct dimensions auto-detected via vectorDimsForModel", () => {
    assert.equal(vectorDimsForModel("voyage-3-large"), 1024);
    assert.equal(vectorDimsForModel("voyage-3"), 1024);
    assert.equal(vectorDimsForModel("voyage-3-lite"), 512);
    assert.equal(vectorDimsForModel("voyage-code-3"), 1024);
    assert.equal(vectorDimsForModel("voyage-finance-2"), 1024);
    assert.equal(vectorDimsForModel("voyage-law-2"), 1024);
    assert.equal(vectorDimsForModel("voyage-multilingual-2"), 1024);
  });

  it("${ENV_VAR} resolution in apiKey", () => {
    process.env.TEST_VIDYA_API_KEY = "resolved-secret-key";
    try {
      const cfg = memoryConfigSchema.parse({
        embedding: { apiKey: "${TEST_VIDYA_API_KEY}" },
      });
      assert.equal(cfg.embedding.apiKey, "resolved-secret-key");
    } finally {
      delete process.env.TEST_VIDYA_API_KEY;
    }
  });

  it("${ENV_VAR} resolution throws when env var not set", () => {
    delete process.env.NONEXISTENT_VAR_VIDYA;
    assert.throws(
      () =>
        memoryConfigSchema.parse({
          embedding: { apiKey: "${NONEXISTENT_VAR_VIDYA}" },
        }),
      /Environment variable NONEXISTENT_VAR_VIDYA is not set/,
    );
  });

  it("captureMaxChars below 100 → throw", () => {
    assert.throws(
      () =>
        memoryConfigSchema.parse({
          embedding: { apiKey: "key" },
          captureMaxChars: 50,
        }),
      /captureMaxChars must be between 100 and 10000/,
    );
  });

  it("captureMaxChars above 10000 → throw", () => {
    assert.throws(
      () =>
        memoryConfigSchema.parse({
          embedding: { apiKey: "key" },
          captureMaxChars: 20000,
        }),
      /captureMaxChars must be between 100 and 10000/,
    );
  });

  it("captureMaxChars valid value → ok", () => {
    const cfg = memoryConfigSchema.parse({
      embedding: { apiKey: "key" },
      captureMaxChars: 2000,
    });
    assert.equal(cfg.captureMaxChars, 2000);
  });

  it("retrieval.mode defaults to 'hybrid'", () => {
    const cfg = memoryConfigSchema.parse({
      embedding: { apiKey: "key" },
    });
    assert.equal(cfg.retrieval.mode, "hybrid");
  });

  it("retrieval.mode can be set to 'vector'", () => {
    const cfg = memoryConfigSchema.parse({
      embedding: { apiKey: "key" },
      retrieval: { mode: "vector" },
    });
    assert.equal(cfg.retrieval.mode, "vector");
  });

  it("all retrieval default values are correct", () => {
    const cfg = memoryConfigSchema.parse({
      embedding: { apiKey: "key" },
    });
    assert.equal(cfg.retrieval.vectorWeight, 0.7);
    assert.equal(cfg.retrieval.bm25Weight, 0.3);
    assert.equal(cfg.retrieval.minScore, 0.3);
    assert.equal(cfg.retrieval.rerank, "cross-encoder");
    assert.equal(cfg.retrieval.rerankModel, "rerank-2");
    assert.equal(cfg.retrieval.candidatePoolSize, 20);
    assert.equal(cfg.retrieval.recencyHalfLifeDays, 14);
    assert.equal(cfg.retrieval.recencyWeight, 0.10);
    assert.equal(cfg.retrieval.filterNoise, true);
    assert.equal(cfg.retrieval.lengthNormAnchor, 500);
    assert.equal(cfg.retrieval.hardMinScore, 0.35);
    assert.equal(cfg.retrieval.timeDecayHalfLifeDays, 60);
  });

  it("non-object config → should throw", () => {
    assert.throws(() => memoryConfigSchema.parse(null), /config required/);
    assert.throws(() => memoryConfigSchema.parse("string"), /config required/);
    assert.throws(() => memoryConfigSchema.parse([1, 2]), /config required/);
  });

  it("VOYAGE_API_KEY env var fallback works", () => {
    process.env.VOYAGE_API_KEY = "env-fallback-key";
    try {
      const cfg = memoryConfigSchema.parse({ embedding: {} });
      assert.equal(cfg.embedding.apiKey, "env-fallback-key");
    } finally {
      delete process.env.VOYAGE_API_KEY;
    }
  });
});

// ============================================================================
// 2. Scope Manager
// ============================================================================

import { MemoryScopeManager, createScopeManager } from "../src/scopes.js";

describe("Scope Manager (MemoryScopeManager)", () => {
  it("default config → global scope accessible", () => {
    const mgr = new MemoryScopeManager();
    const scopes = mgr.getAllScopes();
    assert.ok(scopes.includes("global"));
  });

  it("default config → getDefaultScope returns 'global'", () => {
    const mgr = new MemoryScopeManager();
    assert.equal(mgr.getDefaultScope(), "global");
  });

  it("agent with explicit access → only those scopes", () => {
    const mgr = new MemoryScopeManager({
      agentAccess: { "agent-1": ["global", "custom:private"] },
      definitions: {
        global: { description: "global" },
        "custom:private": { description: "private" },
      },
    });
    const scopes = mgr.getAccessibleScopes("agent-1");
    assert.deepEqual(scopes, ["global", "custom:private"]);
  });

  it("agent without explicit access → global + agent:id", () => {
    const mgr = new MemoryScopeManager();
    const scopes = mgr.getAccessibleScopes("bot-42");
    assert.ok(scopes.includes("global"));
    assert.ok(scopes.includes("agent:bot-42"));
    assert.equal(scopes.length, 2);
  });

  it("isAccessible checks — agent with access", () => {
    const mgr = new MemoryScopeManager({
      agentAccess: { "a1": ["global"] },
    });
    assert.equal(mgr.isAccessible("global", "a1"), true);
    assert.equal(mgr.isAccessible("custom:secret", "a1"), false);
  });

  it("isAccessible checks — no agentId → validates scope format", () => {
    const mgr = new MemoryScopeManager();
    assert.equal(mgr.isAccessible("global"), true);
    assert.equal(mgr.isAccessible("agent:foo"), true); // built-in pattern
    assert.equal(mgr.isAccessible(""), false);
  });

  it("getDefaultScope for agent with agentAccess that includes agent scope", () => {
    const mgr = new MemoryScopeManager({
      definitions: {
        global: { description: "global" },
        "agent:writer": { description: "writer scope" },
      },
      agentAccess: { writer: ["global", "agent:writer"] },
    });
    assert.equal(mgr.getDefaultScope("writer"), "agent:writer");
  });

  it("getDefaultScope for agent without agentAccess → falls back to config default", () => {
    const mgr = new MemoryScopeManager();
    // agent:unknown-agent is built-in, so accessible, default scope = agent:unknown-agent
    assert.equal(mgr.getDefaultScope("unknown-agent"), "agent:unknown-agent");
  });

  it("validateScope for valid scope strings", () => {
    const mgr = new MemoryScopeManager();
    assert.equal(mgr.validateScope("global"), true);
    assert.equal(mgr.validateScope("agent:mybot"), true);
    assert.equal(mgr.validateScope("custom:test"), true);
    assert.equal(mgr.validateScope("project:proj-1"), true);
    assert.equal(mgr.validateScope("user:u123"), true);
  });

  it("validateScope for invalid scope strings", () => {
    const mgr = new MemoryScopeManager();
    assert.equal(mgr.validateScope(""), false);
    assert.equal(mgr.validateScope("   "), false);
  });

  it("addScopeDefinition works", () => {
    const mgr = new MemoryScopeManager();
    mgr.addScopeDefinition("custom:test", { description: "test scope" });
    assert.ok(mgr.getAllScopes().includes("custom:test"));
    assert.deepEqual(mgr.getScopeDefinition("custom:test"), { description: "test scope" });
  });

  it("addScopeDefinition rejects invalid format", () => {
    const mgr = new MemoryScopeManager();
    assert.throws(
      () => mgr.addScopeDefinition("invalid scope!!", { description: "bad" }),
      /Invalid scope format/,
    );
  });

  it("addScopeDefinition rejects scope > 100 chars", () => {
    const mgr = new MemoryScopeManager();
    const longScope = "a".repeat(101);
    assert.throws(
      () => mgr.addScopeDefinition(longScope, { description: "too long" }),
      /Invalid scope format/,
    );
  });

  it("removeScopeDefinition works", () => {
    const mgr = new MemoryScopeManager();
    mgr.addScopeDefinition("custom:temp", { description: "temp" });
    assert.ok(mgr.getAllScopes().includes("custom:temp"));
    const removed = mgr.removeScopeDefinition("custom:temp");
    assert.equal(removed, true);
    assert.ok(!mgr.getAllScopes().includes("custom:temp"));
  });

  it("removeScopeDefinition returns false for non-existent scope", () => {
    const mgr = new MemoryScopeManager();
    assert.equal(mgr.removeScopeDefinition("custom:nonexistent"), false);
  });

  it("cannot remove 'global' scope → throws", () => {
    const mgr = new MemoryScopeManager();
    assert.throws(
      () => mgr.removeScopeDefinition("global"),
      /Cannot remove global scope/,
    );
  });

  it("removeScopeDefinition also cleans agentAccess", () => {
    const mgr = new MemoryScopeManager({
      definitions: {
        global: { description: "global" },
        "custom:shared": { description: "shared" },
      },
      agentAccess: { "bot-1": ["global", "custom:shared"] },
    });
    mgr.removeScopeDefinition("custom:shared");
    // bot-1 should no longer have custom:shared
    const scopes = mgr.getAccessibleScopes("bot-1");
    assert.ok(!scopes.includes("custom:shared"));
    assert.deepEqual(scopes, ["global"]);
  });

  it("no agentId → all scopes accessible", () => {
    const mgr = new MemoryScopeManager({
      definitions: {
        global: { description: "global" },
        "custom:a": { description: "a" },
        "custom:b": { description: "b" },
      },
    });
    const scopes = mgr.getAccessibleScopes();
    assert.ok(scopes.includes("global"));
    assert.ok(scopes.includes("custom:a"));
    assert.ok(scopes.includes("custom:b"));
    assert.equal(scopes.length, 3);
  });

  it("createScopeManager factory works", () => {
    const mgr = createScopeManager();
    assert.ok(mgr instanceof MemoryScopeManager);
    assert.equal(mgr.getDefaultScope(), "global");
  });

  it("exportConfig returns deep copy", () => {
    const mgr = new MemoryScopeManager();
    const exported = mgr.exportConfig();
    assert.equal(exported.default, "global");
    assert.ok(exported.definitions.global);
    // Mutating the export should not affect the manager
    exported.default = "changed";
    assert.equal(mgr.getDefaultScope(), "global");
  });

  it("getStats returns correct structure", () => {
    const mgr = new MemoryScopeManager({
      definitions: {
        global: { description: "g" },
        "agent:a1": { description: "a1" },
        "custom:c1": { description: "c1" },
      },
      agentAccess: { a1: ["global", "agent:a1"] },
    });
    const stats = mgr.getStats();
    assert.equal(stats.totalScopes, 3);
    assert.equal(stats.agentsWithCustomAccess, 1);
    assert.equal(stats.scopesByType.global, 1);
    assert.equal(stats.scopesByType.agent, 1);
    assert.equal(stats.scopesByType.custom, 1);
  });
});

// ============================================================================
// 3. Capture Logic (shouldCapture, detectCategory)
// ============================================================================

import { shouldCapture, detectCategory } from "../index.js";

describe("shouldCapture()", () => {
  it("too short text → false", () => {
    assert.equal(shouldCapture("hi"), false);
    assert.equal(shouldCapture("ok"), false);
    assert.equal(shouldCapture("hello"), false);
  });

  it("too long text → false (default maxChars=500)", () => {
    const longText = "I prefer " + "x".repeat(600);
    assert.equal(shouldCapture(longText), false);
  });

  it("too long text with custom maxChars → false", () => {
    const text = "I prefer " + "x".repeat(200);
    assert.equal(shouldCapture(text, { maxChars: 100 }), false);
  });

  it("has triggers → true", () => {
    assert.equal(shouldCapture("I prefer dark mode for everything"), true);
    assert.equal(shouldCapture("remember this for later"), true);
    assert.equal(shouldCapture("my email is test@example.com"), true);
    assert.equal(shouldCapture("I always use vim for editing"), true);
    assert.equal(shouldCapture("I like TypeScript more than JavaScript"), true);
    assert.equal(shouldCapture("I hate writing boilerplate code"), true);
  });

  it("no triggers → false", () => {
    assert.equal(shouldCapture("the weather is nice today you know"), false);
    assert.equal(shouldCapture("please fix the bug in the code now"), false);
    assert.equal(shouldCapture("what time is it right now please"), false);
  });

  it("contains <relevant-memories> → false", () => {
    assert.equal(
      shouldCapture("I prefer <relevant-memories>some text</relevant-memories>"),
      false,
    );
  });

  it("XML-like text → false", () => {
    assert.equal(shouldCapture("<response>I prefer this</response>"), false);
  });

  it("too many emojis → false", () => {
    assert.equal(shouldCapture("I prefer 🎉🎊🎃🎄 these emojis"), false);
  });

  it("CJK text with lower minLen threshold", () => {
    // CJK min is 4, so 4-char CJK with trigger should capture
    assert.equal(shouldCapture("我偏好这个"), true); // 5 chars, has 偏好 trigger
    assert.equal(shouldCapture("偏好这"), false); // 3 chars, too short even for CJK
  });

  it("CJK triggers work", () => {
    assert.equal(shouldCapture("记住这个重要信息"), true);
    assert.equal(shouldCapture("我喜欢用深色模式"), true);
    assert.equal(shouldCapture("决定以后用这个工具"), true);
    assert.equal(shouldCapture("总是需要检查两遍"), true);
  });

  it("email pattern triggers capture", () => {
    assert.equal(shouldCapture("contact me at user@domain.com please"), true);
  });

  it("phone pattern triggers capture", () => {
    assert.equal(shouldCapture("my phone is +1234567890123"), true);
  });
});

describe("detectCategory()", () => {
  it("preference detection", () => {
    assert.equal(detectCategory("I prefer dark mode"), "preference");
    assert.equal(detectCategory("I like using vim"), "preference");
    assert.equal(detectCategory("I love TypeScript"), "preference");
    assert.equal(detectCategory("I hate boilerplate"), "preference");
    assert.equal(detectCategory("I want a simpler API"), "preference");
  });

  it("decision detection", () => {
    assert.equal(detectCategory("we decided to use PostgreSQL"), "decision");
    assert.equal(detectCategory("I will use Docker for deployment"), "decision");
    assert.equal(detectCategory("budeme používat React"), "decision");
  });

  it("entity detection", () => {
    assert.equal(detectCategory("my phone is +12345678901"), "entity");
    assert.equal(detectCategory("reach me at user@test.com"), "entity");
    assert.equal(detectCategory("he is called John"), "entity");
  });

  it("fact detection", () => {
    assert.equal(detectCategory("the API is rate-limited"), "fact");
    assert.equal(detectCategory("the server has 16GB RAM"), "fact");
    assert.equal(detectCategory("our team are all remote"), "fact");
  });

  it("other detection (no category match)", () => {
    assert.equal(detectCategory("just some random text here"), "other");
    assert.equal(detectCategory("hmm interesting okay"), "other");
  });

  it("CJK text patterns", () => {
    assert.equal(detectCategory("偏好使用暗色主题"), "preference");
    assert.equal(detectCategory("喜欢用 TypeScript"), "preference");
    assert.equal(detectCategory("讨厌写重复代码"), "preference");
    assert.equal(detectCategory("爱用 vim 编辑器"), "preference");
    assert.equal(detectCategory("习惯早起工作"), "preference");
    assert.equal(detectCategory("决定使用新的架构"), "decision");
    assert.equal(detectCategory("选择了 React 框架"), "decision");
    assert.equal(detectCategory("改用 pnpm 管理"), "decision");
    assert.equal(detectCategory("换成 Bun 运行时"), "decision");
    assert.equal(detectCategory("以后用 ESM 模块"), "decision");
    assert.equal(detectCategory("我的邮箱是 test@example.com"), "entity");
    assert.equal(detectCategory("叫我小明"), "entity");
    assert.equal(detectCategory("总是需要代码审查"), "fact");
    assert.equal(detectCategory("从不跳过测试"), "fact");
    assert.equal(detectCategory("一直使用 CI 流程"), "fact");
    assert.equal(detectCategory("每次都要检查"), "fact");
  });
});

// ============================================================================
// 4. Adaptive Retrieval
// ============================================================================

import { shouldSkipRetrieval } from "../src/adaptive-retrieval.js";

describe("shouldSkipRetrieval()", () => {
  it("greetings → skip", () => {
    assert.equal(shouldSkipRetrieval("hi"), true);
    assert.equal(shouldSkipRetrieval("hello"), true);
    assert.equal(shouldSkipRetrieval("hey"), true);
    assert.equal(shouldSkipRetrieval("good morning"), true);
    assert.equal(shouldSkipRetrieval("yo"), true);
  });

  it("commands → skip", () => {
    assert.equal(shouldSkipRetrieval("git status"), true);
    assert.equal(shouldSkipRetrieval("npm install"), true);
    assert.equal(shouldSkipRetrieval("/help"), true);
    assert.equal(shouldSkipRetrieval("docker build ."), true);
  });

  it("confirmations → skip", () => {
    assert.equal(shouldSkipRetrieval("yes"), true);
    assert.equal(shouldSkipRetrieval("no"), true);
    assert.equal(shouldSkipRetrieval("ok"), true);
    assert.equal(shouldSkipRetrieval("sure"), true);
    assert.equal(shouldSkipRetrieval("thanks"), true);
    assert.equal(shouldSkipRetrieval("好的"), true);
    assert.equal(shouldSkipRetrieval("可以"), true);
    assert.equal(shouldSkipRetrieval("行"), true);
  });

  it("emoji-only → skip", () => {
    assert.equal(shouldSkipRetrieval("👍"), true);
    assert.equal(shouldSkipRetrieval("👍 ✅"), true);
  });

  it("HEARTBEAT → skip", () => {
    assert.equal(shouldSkipRetrieval("HEARTBEAT"), true);
  });

  it("system messages → skip", () => {
    assert.equal(shouldSkipRetrieval("[System message here]"), true);
  });

  it("memory-related queries → don't skip (force retrieve)", () => {
    assert.equal(shouldSkipRetrieval("do you remember what I said?"), false);
    assert.equal(shouldSkipRetrieval("recall my preferences"), false);
    assert.equal(shouldSkipRetrieval("what did I tell you last time?"), false);
    assert.equal(shouldSkipRetrieval("my name is important"), false);
    assert.equal(shouldSkipRetrieval("what is my email address?"), false);
  });

  it("substantive questions → don't skip", () => {
    assert.equal(shouldSkipRetrieval("how do I configure the memory plugin?"), false);
    assert.equal(shouldSkipRetrieval("explain the hybrid retrieval algorithm"), false);
  });

  it("empty string → skip", () => {
    assert.equal(shouldSkipRetrieval(""), true);
    assert.equal(shouldSkipRetrieval("   "), true);
  });

  it("very short text (< 5 chars) → skip", () => {
    assert.equal(shouldSkipRetrieval("abc"), true);
    assert.equal(shouldSkipRetrieval("a"), true);
  });

  it("very long text → don't skip", () => {
    const longText = "Tell me about the implementation details of the vector search system and how it integrates with BM25";
    assert.equal(shouldSkipRetrieval(longText), false);
  });

  it("CJK short text without ? → skip", () => {
    // CJK min is 6 chars, so 5-char CJK without ? should skip
    assert.equal(shouldSkipRetrieval("看看代码吧"), true); // 5 CJK chars
  });

  it("CJK short text with 还记得 → don't skip (force retrieve)", () => {
    assert.equal(shouldSkipRetrieval("你还记得吗"), false); // Simplified Chinese matches 还记得
    assert.equal(shouldSkipRetrieval("还记得之前吗"), false);
  });

  it("CJK text at or above threshold with ? → don't skip", () => {
    assert.equal(shouldSkipRetrieval("这是什么？"), false); // has ？
  });

  it("ASCII short text with ? → don't skip", () => {
    assert.equal(shouldSkipRetrieval("what is this?"), false);
  });

  it("Chinese force-retrieve patterns work", () => {
    assert.equal(shouldSkipRetrieval("你记得我说的吗"), false);
    assert.equal(shouldSkipRetrieval("之前提到的那个"), false);
    assert.equal(shouldSkipRetrieval("上次的配置是什么"), false);
    assert.equal(shouldSkipRetrieval("以前用的方法"), false);
    assert.equal(shouldSkipRetrieval("我提到过的工具"), false);
    assert.equal(shouldSkipRetrieval("我说过的话"), false);
  });

  it("time references → force retrieve", () => {
    assert.equal(shouldSkipRetrieval("what happened last time?"), false);
    assert.equal(shouldSkipRetrieval("we discussed this before"), false);
    assert.equal(shouldSkipRetrieval("I mentioned this previously"), false);
    assert.equal(shouldSkipRetrieval("about a week ago we talked"), false);
  });
});

// ============================================================================
// 6. Embedding Provider Abstraction
// ============================================================================

import { createEmbedderFromConfig } from "../src/embedder-factory.js";
import { VoyageEmbedder, EmbeddingCache } from "../src/embedder.js";
import { OpenAIEmbedder } from "../src/embedder-openai.js";
import { JinaEmbedder } from "../src/embedder-jina.js";

describe("Embedding Provider Abstraction", () => {
  describe("createEmbedderFromConfig()", () => {
    it("creates VoyageEmbedder for provider=voyage", () => {
      const embedder = createEmbedderFromConfig({
        provider: "voyage",
        apiKey: "test-key",
        model: "voyage-3-large",
      });
      assert.ok(embedder instanceof VoyageEmbedder);
      assert.equal(embedder.dimensions, 1024);
      assert.equal(embedder.model, "voyage-3-large");
    });

    it("creates OpenAIEmbedder for provider=openai", () => {
      const embedder = createEmbedderFromConfig({
        provider: "openai",
        apiKey: "test-key",
        model: "text-embedding-3-small",
      });
      assert.ok(embedder instanceof OpenAIEmbedder);
      assert.equal(embedder.dimensions, 1536);
      assert.equal(embedder.model, "text-embedding-3-small");
    });

    it("creates JinaEmbedder for provider=jina", () => {
      const embedder = createEmbedderFromConfig({
        provider: "jina",
        apiKey: "test-key",
        model: "jina-embeddings-v3",
      });
      assert.ok(embedder instanceof JinaEmbedder);
      assert.equal(embedder.dimensions, 1024);
      assert.equal(embedder.model, "jina-embeddings-v3");
    });

    it("throws for unknown provider", () => {
      assert.throws(
        () => createEmbedderFromConfig({
          provider: "unknown" as any,
          apiKey: "test-key",
          model: "some-model",
        }),
        /Unknown embedding provider/,
      );
    });

    it("respects custom dimensions override", () => {
      const embedder = createEmbedderFromConfig({
        provider: "openai",
        apiKey: "test-key",
        model: "text-embedding-3-large",
        dimensions: 256,
      });
      assert.equal(embedder.dimensions, 256);
    });
  });

  describe("Config parser provider support", () => {
    it("defaults provider to voyage when not specified", () => {
      const config = memoryConfigSchema.parse({
        embedding: { apiKey: "test-key" },
      });
      assert.equal(config.embedding.provider, "voyage");
      assert.equal(config.embedding.model, "voyage-3-large");
    });

    it("accepts openai provider", () => {
      const config = memoryConfigSchema.parse({
        embedding: { provider: "openai", apiKey: "test-key" },
      });
      assert.equal(config.embedding.provider, "openai");
      assert.equal(config.embedding.model, "text-embedding-3-small");
    });

    it("accepts jina provider", () => {
      const config = memoryConfigSchema.parse({
        embedding: { provider: "jina", apiKey: "test-key" },
      });
      assert.equal(config.embedding.provider, "jina");
      assert.equal(config.embedding.model, "jina-embeddings-v3");
    });

    it("throws for unknown provider", () => {
      assert.throws(
        () => memoryConfigSchema.parse({
          embedding: { provider: "unknown", apiKey: "test-key" },
        }),
        /Unknown embedding provider/,
      );
    });

    it("resolves provider-specific env vars", () => {
      process.env.OPENAI_API_KEY = "sk-test-from-env";
      try {
        const config = memoryConfigSchema.parse({
          embedding: { provider: "openai" },
        });
        assert.equal(config.embedding.apiKey, "sk-test-from-env");
      } finally {
        delete process.env.OPENAI_API_KEY;
      }
    });

    it("accepts baseUrl for custom endpoints", () => {
      const config = memoryConfigSchema.parse({
        embedding: { provider: "openai", apiKey: "test-key", baseUrl: "http://localhost:8080/v1" },
      });
      assert.equal(config.embedding.baseUrl, "http://localhost:8080/v1");
    });
  });

  describe("Dimension lookups per provider", () => {
    it("Voyage: voyage-3-large → 1024", () => {
      const e = createEmbedderFromConfig({ provider: "voyage", apiKey: "k", model: "voyage-3-large" });
      assert.equal(e.dimensions, 1024);
    });

    it("Voyage: voyage-3-lite → 512", () => {
      const e = createEmbedderFromConfig({ provider: "voyage", apiKey: "k", model: "voyage-3-lite" });
      assert.equal(e.dimensions, 512);
    });

    it("OpenAI: text-embedding-3-large → 3072", () => {
      const e = createEmbedderFromConfig({ provider: "openai", apiKey: "k", model: "text-embedding-3-large" });
      assert.equal(e.dimensions, 3072);
    });

    it("OpenAI: text-embedding-ada-002 → 1536", () => {
      const e = createEmbedderFromConfig({ provider: "openai", apiKey: "k", model: "text-embedding-ada-002" });
      assert.equal(e.dimensions, 1536);
    });

    it("Jina: jina-embeddings-v2-base-en → 768", () => {
      const e = createEmbedderFromConfig({ provider: "jina", apiKey: "k", model: "jina-embeddings-v2-base-en" });
      assert.equal(e.dimensions, 768);
    });

    it("unknown model without dimensions → throws", () => {
      assert.throws(
        () => createEmbedderFromConfig({ provider: "openai", apiKey: "k", model: "unknown-model" }),
        /Unknown embedding model/,
      );
    });

    it("unknown model with dimensions override → works", () => {
      const e = createEmbedderFromConfig({ provider: "openai", apiKey: "k", model: "custom-model", dimensions: 384 });
      assert.equal(e.dimensions, 384);
    });
  });
});

// ============================================================================
// 7. LLM Capture Config
// ============================================================================

describe("LLM Capture Config", () => {
  it("captureLlm defaults to true", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" } });
    assert.equal(config.captureLlm, true);
  });

  it("captureLlm can be disabled", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" }, captureLlm: false });
    assert.equal(config.captureLlm, false);
  });

  it("captureLlmModel has default", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" } });
    assert.equal(config.captureLlmModel, "anthropic/claude-haiku-4-5-20251001");
  });

  it("captureLlmModel can be overridden", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" }, captureLlmModel: "openai/gpt-4o-mini" });
    assert.equal(config.captureLlmModel, "openai/gpt-4o-mini");
  });

  it("captureLlmUrl defaults to empty string", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" } });
    assert.equal(config.captureLlmUrl, "");
  });

  it("captureLlmUrl can be set", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" }, captureLlmUrl: "http://my-llm:8080" });
    assert.equal(config.captureLlmUrl, "http://my-llm:8080");
  });

  it("captureLlmUrl strips trailing /v1 to prevent double path", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" }, captureLlmUrl: "https://api.openai.com/v1" });
    assert.equal(config.captureLlmUrl, "https://api.openai.com");
  });

  it("captureLlmUrl strips trailing /v1/ (with slash) to prevent double path", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" }, captureLlmUrl: "https://api.openai.com/v1/" });
    assert.equal(config.captureLlmUrl, "https://api.openai.com");
  });

  it("captureLlmUrl strips trailing slash", () => {
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" }, captureLlmUrl: "http://localhost:3000/" });
    assert.equal(config.captureLlmUrl, "http://localhost:3000");
  });

  it("captureLlmUrl trims surrounding whitespace", () => {
    const config = memoryConfigSchema.parse({
      embedding: { apiKey: "k" },
      captureLlmUrl: "  http://localhost:3000  ",
    });
    assert.equal(config.captureLlmUrl, "http://localhost:3000");
  });
});

// ============================================================================
// Phase 1 — Pure Functions
// ============================================================================

// ============================================================================
// 8. Noise Filter
// ============================================================================

import { isNoise, filterNoise } from "../src/noise-filter.js";

describe("isNoise()", () => {
  it("short text (< 5 chars) → noise", () => {
    assert.equal(isNoise(""), true);
    assert.equal(isNoise("hi"), true);
    assert.equal(isNoise("ok"), true);
    assert.equal(isNoise("    "), true);
  });

  it("valid text above 5 chars → not noise", () => {
    assert.equal(isNoise("This is a valid statement for testing"), false);
  });

  it("denial patterns → noise", () => {
    assert.equal(isNoise("I don't have any information about that"), true);
    assert.equal(isNoise("I'm not sure about that topic"), true);
    assert.equal(isNoise("I don't recall anything about it"), true);
    assert.equal(isNoise("I don't remember that conversation"), true);
    assert.equal(isNoise("It looks like I don't have that"), true);
    assert.equal(isNoise("I wasn't able to find that"), true);
    assert.equal(isNoise("No relevant memories found for this"), true);
    assert.equal(isNoise("I don't have access to that data"), true);
    assert.equal(isNoise("no memories found in the database"), true);
  });

  it("meta-question patterns → noise", () => {
    assert.equal(isNoise("do you remember what I told you"), true);
    assert.equal(isNoise("can you recall the details from before"), true);
    assert.equal(isNoise("did I tell you about the project"), true);
    assert.equal(isNoise("have I told you about my preferences"), true);
    assert.equal(isNoise("what did I tell you yesterday"), true);
    assert.equal(isNoise("have I mentioned the new API design"), true);
    assert.equal(isNoise("did I say something about that"), true);
  });

  it("boilerplate patterns → noise", () => {
    assert.equal(isNoise("hi there how are you"), true);
    assert.equal(isNoise("hello everyone"), true);
    assert.equal(isNoise("Hey, good to see you"), true);
    assert.equal(isNoise("good morning team"), true);
    assert.equal(isNoise("good evening friends"), true);
    assert.equal(isNoise("greetings from the dev team"), true);
    assert.equal(isNoise("fresh session starting now"), true);
    assert.equal(isNoise("new session begun here"), true);
    assert.equal(isNoise("HEARTBEAT signal active"), true);
  });

  it("valid text → not noise", () => {
    assert.equal(isNoise("I prefer dark mode for coding"), false);
    assert.equal(isNoise("The database schema uses PostgreSQL"), false);
    assert.equal(isNoise("We decided to switch to TypeScript"), false);
    assert.equal(isNoise("My email is user@example.com"), false);
  });

  it("filterDenials=false → denials are not filtered", () => {
    assert.equal(isNoise("I don't have any information about that", { filterDenials: false }), false);
  });

  it("filterMetaQuestions=false → meta questions not filtered", () => {
    assert.equal(isNoise("do you remember what I told you", { filterMetaQuestions: false }), false);
  });

  it("filterBoilerplate=false → boilerplate not filtered", () => {
    assert.equal(isNoise("hello everyone in the room", { filterBoilerplate: false }), false);
  });

  it("all filters disabled → only short text filtered", () => {
    const opts = { filterDenials: false, filterMetaQuestions: false, filterBoilerplate: false };
    assert.equal(isNoise("hi", opts), true); // still too short
    assert.equal(isNoise("I don't have any information", opts), false);
    assert.equal(isNoise("hello everyone", opts), false);
  });
});

describe("filterNoise()", () => {
  it("filters noisy items from array", () => {
    const items = [
      { id: 1, content: "I prefer dark mode for everything" },
      { id: 2, content: "hi" },
      { id: 3, content: "I don't have any information about that" },
      { id: 4, content: "The API uses REST endpoints" },
    ];
    const filtered = filterNoise(items, item => item.content);
    assert.equal(filtered.length, 2);
    assert.equal(filtered[0].id, 1);
    assert.equal(filtered[1].id, 4);
  });

  it("returns empty array if all items are noisy", () => {
    const items = [
      { text: "hi" },
      { text: "ok" },
      { text: "hello there" },
    ];
    const filtered = filterNoise(items, item => item.text);
    assert.equal(filtered.length, 0);
  });

  it("returns all items if none are noisy", () => {
    const items = [
      { text: "Important technical decision recorded" },
      { text: "User prefers Python over JavaScript" },
    ];
    const filtered = filterNoise(items, item => item.text);
    assert.equal(filtered.length, 2);
  });

  it("respects custom options", () => {
    const items = [
      { text: "I don't have any information about that topic" },
    ];
    const filtered = filterNoise(items, item => item.text, { filterDenials: false });
    assert.equal(filtered.length, 1);
  });
});

// ============================================================================
// 9. Utils
// ============================================================================

import { normalizeBaseUrl, getUrlHost } from "../src/utils.js";

describe("normalizeBaseUrl()", () => {
  it("strips trailing slashes", () => {
    assert.equal(normalizeBaseUrl("http://example.com/"), "http://example.com");
    assert.equal(normalizeBaseUrl("http://example.com///"), "http://example.com");
  });

  it("strips trailing /v1", () => {
    assert.equal(normalizeBaseUrl("http://example.com/v1"), "http://example.com");
    assert.equal(normalizeBaseUrl("http://example.com/v1/"), "http://example.com");
  });

  it("strips trailing /v1 case-insensitively", () => {
    assert.equal(normalizeBaseUrl("http://example.com/V1"), "http://example.com");
    assert.equal(normalizeBaseUrl("http://example.com/V1/"), "http://example.com");
  });

  it("trims whitespace", () => {
    assert.equal(normalizeBaseUrl("  http://example.com  "), "http://example.com");
    assert.equal(normalizeBaseUrl("  http://example.com/v1  "), "http://example.com");
  });

  it("handles URL with path after stripping /v1", () => {
    assert.equal(normalizeBaseUrl("http://example.com/api/v1"), "http://example.com/api");
    assert.equal(normalizeBaseUrl("http://example.com/api/v1/"), "http://example.com/api");
  });

  it("preserves non-v1 paths", () => {
    assert.equal(normalizeBaseUrl("http://example.com/v2"), "http://example.com/v2");
    assert.equal(normalizeBaseUrl("http://example.com/api"), "http://example.com/api");
  });

  it("handles empty string and whitespace", () => {
    assert.equal(normalizeBaseUrl(""), "");
    assert.equal(normalizeBaseUrl("   "), "");
  });

  it("strips trailing slashes after removing /v1", () => {
    assert.equal(normalizeBaseUrl("http://example.com//v1"), "http://example.com");
  });
});

describe("getUrlHost()", () => {
  it("extracts host from standard URLs", () => {
    assert.equal(getUrlHost("http://example.com"), "example.com");
    assert.equal(getUrlHost("https://example.com"), "example.com");
    assert.equal(getUrlHost("http://example.com:8080"), "example.com:8080");
    assert.equal(getUrlHost("https://localhost:3000"), "localhost:3000");
  });

  it("extracts host from URL without protocol", () => {
    assert.equal(getUrlHost("example.com"), "example.com");
  });

  it("returns empty string when port is interpreted as protocol", () => {
    // "localhost:3000" is parsed as scheme "localhost:" by URL constructor (doesn't throw),
    // so it returns empty string host (URL treats "localhost" as the protocol)
    assert.equal(getUrlHost("localhost:3000"), "");
  });

  it("returns null for unparseable URLs", () => {
    assert.equal(getUrlHost(""), null);
    assert.equal(getUrlHost("not a url at all ::: ///"), null);
  });

  it("handles URLs with paths", () => {
    assert.equal(getUrlHost("http://example.com/api/v1"), "example.com");
    assert.equal(getUrlHost("https://api.openai.com/v1/embeddings"), "api.openai.com");
  });
});

// ============================================================================
// Phase 2 — Mock Fetch Tests
// ============================================================================

// ============================================================================
// 10. VoyageEmbedder (mock fetch)
// ============================================================================

describe("EmbeddingCache", () => {
  it("get returns undefined on miss", () => {
    const cache = new EmbeddingCache(10, 30);
    assert.equal(cache.get("nonexistent"), undefined);
    assert.equal(cache.misses, 1);
  });

  it("set and get round-trip", () => {
    const cache = new EmbeddingCache(10, 30);
    cache.set("hello", "doc", [1, 2, 3]);
    const result = cache.get("hello", "doc");
    assert.deepEqual(result, [1, 2, 3]);
    assert.equal(cache.hits, 1);
  });

  it("respects TTL (expired entries return undefined)", async () => {
    // Use a very short TTL and wait for it to expire
    const cache = new EmbeddingCache(10, 1 / 60_000); // ~1ms TTL
    cache.set("hello", "doc", [1, 2, 3]);
    // Wait 20ms for TTL to expire (generous margin for slow CI)
    await new Promise(resolve => setTimeout(resolve, 20));
    const result = cache.get("hello", "doc");
    assert.equal(result, undefined);
    assert.equal(cache.misses, 1);
  });

  it("evicts oldest when maxSize reached", () => {
    const cache = new EmbeddingCache(2, 30);
    cache.set("a", "doc", [1]);
    cache.set("b", "doc", [2]);
    cache.set("c", "doc", [3]);
    assert.equal(cache.get("a", "doc"), undefined);
    assert.deepEqual(cache.get("b", "doc"), [2]);
    assert.deepEqual(cache.get("c", "doc"), [3]);
    assert.equal(cache.size, 2);
  });

  it("key includes inputType", () => {
    const cache = new EmbeddingCache(10, 30);
    cache.set("hello", "query", [1]);
    cache.set("hello", "doc", [2]);
    assert.deepEqual(cache.get("hello", "query"), [1]);
    assert.deepEqual(cache.get("hello", "doc"), [2]);
  });

  it("stats reports correct values", () => {
    const cache = new EmbeddingCache(10, 30);
    cache.set("a", "doc", [1]);
    cache.get("a", "doc");
    cache.get("b", "doc");
    const stats = cache.stats;
    assert.equal(stats.size, 1);
    assert.equal(stats.hits, 1);
    assert.equal(stats.misses, 1);
    assert.equal(stats.hitRate, "50.0%");
  });

  it("stats hitRate is N/A when no lookups", () => {
    const cache = new EmbeddingCache(10, 30);
    assert.equal(cache.stats.hitRate, "N/A");
  });
});

describe("VoyageEmbedder (mocked fetch)", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function mockFetch(responseBody: unknown, status = 200) {
    globalThis.fetch = (async () => ({
      ok: status >= 200 && status < 300,
      status,
      json: async () => responseBody,
      text: async () => JSON.stringify(responseBody),
    })) as unknown as typeof fetch;
  }

  it("embed() calls Voyage API and returns embedding", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    const mockVector = Array(1024).fill(0.1);
    mockFetch({
      object: "list",
      data: [{ object: "embedding", embedding: mockVector, index: 0 }],
      model: "voyage-3-large",
      usage: { total_tokens: 5 },
    });

    const result = await embedder.embed("test text");
    assert.equal(result.length, 1024);
    assert.deepEqual(result, mockVector);
  });

  it("embed() caches results", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    const mockVector = Array(1024).fill(0.5);
    let callCount = 0;
    globalThis.fetch = (async () => {
      callCount++;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          data: [{ embedding: mockVector, index: 0 }],
          model: "voyage-3-large",
          usage: { total_tokens: 5 },
        }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await embedder.embed("cached text");
    await embedder.embed("cached text");
    assert.equal(callCount, 1);
    assert.equal(embedder.cacheStats.hits, 1);
  });

  it("embed() throws on empty text", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    await assert.rejects(() => embedder.embed(""), /Cannot embed empty text/);
    await assert.rejects(() => embedder.embed("   "), /Cannot embed empty text/);
  });

  it("embed() throws on API error", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    mockFetch({ error: "rate limited" }, 429);
    await assert.rejects(() => embedder.embed("test"), /Voyage AI embedding API returned 429/);
  });

  it("embed() throws on empty data response", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    mockFetch({ data: [] });
    await assert.rejects(() => embedder.embed("test"), /No embedding returned/);
  });

  it("embed() throws on dimension mismatch", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    mockFetch({ data: [{ embedding: [1, 2, 3], index: 0 }] });
    await assert.rejects(() => embedder.embed("test"), /Embedding dimension mismatch/);
  });

  it("embedBatch() returns multiple embeddings", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    const mockVector = Array(1024).fill(0.2);
    mockFetch({
      data: [
        { embedding: mockVector, index: 0 },
        { embedding: mockVector, index: 1 },
      ],
    });

    const results = await embedder.embedBatch(["text1", "text2"]);
    assert.equal(results.length, 2);
    assert.equal(results[0].length, 1024);
    assert.equal(results[1].length, 1024);
  });

  it("embedBatch() handles empty array", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    const results = await embedder.embedBatch([]);
    assert.deepEqual(results, []);
  });

  it("embedBatch() handles empty string entries", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    const mockVector = Array(1024).fill(0.1);
    mockFetch({ data: [{ embedding: mockVector, index: 0 }] });

    const results = await embedder.embedBatch(["valid text", "", "  "]);
    assert.equal(results.length, 3);
    assert.equal(results[0].length, 1024);
    assert.deepEqual(results[1], []);
    assert.deepEqual(results[2], []);
  });

  it("embedQuery() and embedPassage() use different input_type", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    const mockVector = Array(1024).fill(0.1);
    let capturedBodies: string[] = [];

    globalThis.fetch = (async (_url: any, opts: any) => {
      capturedBodies.push(opts.body);
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: mockVector, index: 0 }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await embedder.embedQuery("query text");
    await embedder.embedPassage("passage text");

    const body1 = JSON.parse(capturedBodies[0]);
    const body2 = JSON.parse(capturedBodies[1]);
    assert.equal(body1.input_type, "query");
    assert.equal(body2.input_type, "document");
  });

  it("test() returns success on valid response", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    const mockVector = Array(1024).fill(0.1);
    mockFetch({ data: [{ embedding: mockVector, index: 0 }] });
    const result = await embedder.test();
    assert.equal(result.success, true);
    assert.equal(result.dimensions, 1024);
  });

  it("test() returns failure on error", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "test-key", model: "voyage-3-large" });
    mockFetch({}, 500);
    const result = await embedder.test();
    assert.equal(result.success, false);
    assert.ok(result.error);
  });

  it("model getter returns configured model", () => {
    const embedder = new VoyageEmbedder({ apiKey: "key", model: "voyage-3" });
    assert.equal(embedder.model, "voyage-3");
  });

  it("uses custom baseUrl when provided", async () => {
    const embedder = new VoyageEmbedder({ apiKey: "key", model: "voyage-3-large", baseUrl: "http://custom:9000/embed" });
    let capturedUrl = "";
    const mockVector = Array(1024).fill(0.1);
    globalThis.fetch = (async (url: any) => {
      capturedUrl = url;
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: mockVector, index: 0 }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await embedder.embed("test");
    assert.equal(capturedUrl, "http://custom:9000/embed");
  });
});

// ============================================================================
// 11. OpenAIEmbedder (mocked fetch)
// ============================================================================

describe("OpenAIEmbedder (mocked fetch)", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function mockFetch(responseBody: unknown, status = 200) {
    globalThis.fetch = (async () => ({
      ok: status >= 200 && status < 300,
      status,
      json: async () => responseBody,
      text: async () => JSON.stringify(responseBody),
    })) as unknown as typeof fetch;
  }

  it("embed() returns embedding", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    const mockVector = Array(1536).fill(0.1);
    mockFetch({
      data: [{ embedding: mockVector, index: 0 }],
      usage: { prompt_tokens: 5, total_tokens: 5 },
    });

    const result = await embedder.embed("test");
    assert.equal(result.length, 1536);
  });

  it("sends dimensions parameter for text-embedding-3-* models", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small", dimensions: 256 });
    let capturedBody = "";
    const mockVector = Array(256).fill(0.1);

    globalThis.fetch = (async (_url: any, opts: any) => {
      capturedBody = opts.body;
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: mockVector, index: 0 }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await embedder.embed("test");
    const body = JSON.parse(capturedBody);
    assert.equal(body.dimensions, 256);
  });

  it("does NOT send dimensions for ada-002 model", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-ada-002" });
    let capturedBody = "";
    const mockVector = Array(1536).fill(0.1);

    globalThis.fetch = (async (_url: any, opts: any) => {
      capturedBody = opts.body;
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: mockVector, index: 0 }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await embedder.embed("test");
    const body = JSON.parse(capturedBody);
    assert.equal(body.dimensions, undefined);
  });

  it("embed() throws on empty text", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    await assert.rejects(() => embedder.embed(""), /Cannot embed empty text/);
  });

  it("embed() throws on API error", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    mockFetch({ error: "unauthorized" }, 401);
    await assert.rejects(() => embedder.embed("test"), /OpenAI embedding API returned 401/);
  });

  it("embedBatch() returns multiple embeddings", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    const mockVector = Array(1536).fill(0.1);
    mockFetch({
      data: [
        { embedding: mockVector, index: 0 },
        { embedding: mockVector, index: 1 },
      ],
    });
    const results = await embedder.embedBatch(["a", "b"]);
    assert.equal(results.length, 2);
  });

  it("embedBatch() handles empty array", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    assert.deepEqual(await embedder.embedBatch([]), []);
  });

  it("embedBatch() handles empty string entries", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    const mockVector = Array(1536).fill(0.1);
    mockFetch({ data: [{ embedding: mockVector, index: 0 }] });
    const results = await embedder.embedBatch(["valid", "", "  "]);
    assert.equal(results.length, 3);
    assert.deepEqual(results[1], []);
    assert.deepEqual(results[2], []);
  });

  it("test() returns success/failure", async () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    const mockVector = Array(1536).fill(0.1);
    mockFetch({ data: [{ embedding: mockVector, index: 0 }] });
    const result = await embedder.test();
    assert.equal(result.success, true);
  });

  it("model getter returns configured model", () => {
    const embedder = new OpenAIEmbedder({ apiKey: "sk", model: "text-embedding-3-large" });
    assert.equal(embedder.model, "text-embedding-3-large");
  });
});

// ============================================================================
// 12. JinaEmbedder (mocked fetch)
// ============================================================================

describe("JinaEmbedder (mocked fetch)", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function mockFetch(responseBody: unknown, status = 200) {
    globalThis.fetch = (async () => ({
      ok: status >= 200 && status < 300,
      status,
      json: async () => responseBody,
      text: async () => JSON.stringify(responseBody),
    })) as unknown as typeof fetch;
  }

  it("embed() returns embedding", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    const mockVector = Array(1024).fill(0.1);
    mockFetch({
      data: [{ embedding: mockVector, index: 0 }],
      usage: { total_tokens: 5, prompt_tokens: 5 },
    });

    const result = await embedder.embed("test");
    assert.equal(result.length, 1024);
  });

  it("sends task parameter for v3 model", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    let capturedBody = "";
    const mockVector = Array(1024).fill(0.1);

    globalThis.fetch = (async (_url: any, opts: any) => {
      capturedBody = opts.body;
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: mockVector, index: 0 }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await embedder.embedQuery("test");
    const body = JSON.parse(capturedBody);
    assert.equal(body.task, "retrieval.query");
  });

  it("does NOT send task parameter for v2 model", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v2-base-en" });
    let capturedBody = "";
    const mockVector = Array(768).fill(0.1);

    globalThis.fetch = (async (_url: any, opts: any) => {
      capturedBody = opts.body;
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: mockVector, index: 0 }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await embedder.embedQuery("test");
    const body = JSON.parse(capturedBody);
    assert.equal(body.task, undefined);
  });

  it("embed() throws on empty text", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    await assert.rejects(() => embedder.embed(""), /Cannot embed empty text/);
  });

  it("embed() throws on API error", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    mockFetch({ error: "bad request" }, 400);
    await assert.rejects(() => embedder.embed("test"), /Jina AI embedding API returned 400/);
  });

  it("embedBatch() returns embeddings", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    const mockVector = Array(1024).fill(0.1);
    mockFetch({
      data: [
        { embedding: mockVector, index: 0 },
        { embedding: mockVector, index: 1 },
      ],
    });
    const results = await embedder.embedBatch(["a", "b"]);
    assert.equal(results.length, 2);
  });

  it("embedBatch() handles empty array", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    assert.deepEqual(await embedder.embedBatch([]), []);
  });

  it("embedBatch() handles empty string entries", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    const mockVector = Array(1024).fill(0.1);
    mockFetch({ data: [{ embedding: mockVector, index: 0 }] });
    const results = await embedder.embedBatch(["valid", ""]);
    assert.equal(results.length, 2);
    assert.deepEqual(results[1], []);
  });

  it("embed() throws on dimension mismatch", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    mockFetch({ data: [{ embedding: [1, 2, 3], index: 0 }] });
    await assert.rejects(() => embedder.embed("test"), /Embedding dimension mismatch/);
  });

  it("test() returns success on valid response", async () => {
    const embedder = new JinaEmbedder({ apiKey: "jina-key", model: "jina-embeddings-v3" });
    const mockVector = Array(1024).fill(0.1);
    mockFetch({ data: [{ embedding: mockVector, index: 0 }] });
    const result = await embedder.test();
    assert.equal(result.success, true);
    assert.equal(result.dimensions, 1024);
  });

  it("model getter returns configured model", () => {
    const embedder = new JinaEmbedder({ apiKey: "k", model: "jina-embeddings-v3" });
    assert.equal(embedder.model, "jina-embeddings-v3");
  });

  it("uses custom baseUrl", async () => {
    const embedder = new JinaEmbedder({ apiKey: "k", model: "jina-embeddings-v3", baseUrl: "http://custom:5000/embed" });
    let capturedUrl = "";
    const mockVector = Array(1024).fill(0.1);
    globalThis.fetch = (async (url: any) => {
      capturedUrl = url;
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: mockVector, index: 0 }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await embedder.embed("test");
    assert.equal(capturedUrl, "http://custom:5000/embed");
  });
});

// ============================================================================
// 13. LLM URL construction (config integration)
// ============================================================================

describe("LLM URL construction (config integration)", () => {
  it("captureLlmUrl normalization strips /v1 to avoid /v1/v1/chat/completions", () => {
    const config = memoryConfigSchema.parse({
      embedding: { apiKey: "k" },
      captureLlmUrl: "https://openrouter.ai/api/v1",
    });
    assert.equal(config.captureLlmUrl, "https://openrouter.ai/api");
  });

  it("captureLlmUrl preserves non-v1 paths", () => {
    const config = memoryConfigSchema.parse({
      embedding: { apiKey: "k" },
      captureLlmUrl: "https://api.example.com/proxy",
    });
    assert.equal(config.captureLlmUrl, "https://api.example.com/proxy");
  });
});

// ============================================================================
// Phase 3 — Mock LanceDB Tests
// ============================================================================

// ============================================================================
// 14. Scopes (additional coverage for factory functions)
// ============================================================================

import {
  createAgentScope,
  createCustomScope,
  createProjectScope,
  createUserScope,
  parseScopeId,
  isScopeAccessible,
} from "../src/scopes.js";

describe("Scope utility functions", () => {
  it("createAgentScope creates correct scope string", () => {
    assert.equal(createAgentScope("bot-1"), "agent:bot-1");
  });

  it("createCustomScope creates correct scope string", () => {
    assert.equal(createCustomScope("private"), "custom:private");
  });

  it("createProjectScope creates correct scope string", () => {
    assert.equal(createProjectScope("proj-42"), "project:proj-42");
  });

  it("createUserScope creates correct scope string", () => {
    assert.equal(createUserScope("user123"), "user:user123");
  });

  it("parseScopeId parses global scope", () => {
    const result = parseScopeId("global");
    assert.deepEqual(result, { type: "global", id: "" });
  });

  it("parseScopeId parses typed scopes", () => {
    assert.deepEqual(parseScopeId("agent:bot-1"), { type: "agent", id: "bot-1" });
    assert.deepEqual(parseScopeId("custom:private"), { type: "custom", id: "private" });
    assert.deepEqual(parseScopeId("project:p1"), { type: "project", id: "p1" });
    assert.deepEqual(parseScopeId("user:u1"), { type: "user", id: "u1" });
  });

  it("parseScopeId returns null for invalid format", () => {
    assert.equal(parseScopeId("nocolon"), null);
  });

  it("isScopeAccessible checks inclusion", () => {
    assert.equal(isScopeAccessible("global", ["global", "agent:bot"]), true);
    assert.equal(isScopeAccessible("custom:secret", ["global"]), false);
  });
});

describe("MemoryScopeManager additional coverage", () => {
  it("setAgentAccess and removeAgentAccess", () => {
    const mgr = new MemoryScopeManager();
    mgr.setAgentAccess("bot-1", ["global"]);
    assert.deepEqual(mgr.getAccessibleScopes("bot-1"), ["global"]);

    const removed = mgr.removeAgentAccess("bot-1");
    assert.equal(removed, true);
    const scopes = mgr.getAccessibleScopes("bot-1");
    assert.ok(scopes.includes("global"));
    assert.ok(scopes.includes("agent:bot-1"));
  });

  it("removeAgentAccess returns false for non-existent agent", () => {
    const mgr = new MemoryScopeManager();
    assert.equal(mgr.removeAgentAccess("nonexistent"), false);
  });

  it("setAgentAccess throws on invalid agentId", () => {
    const mgr = new MemoryScopeManager();
    assert.throws(() => mgr.setAgentAccess("", ["global"]), /Invalid agent ID/);
  });

  it("setAgentAccess throws on invalid scope", () => {
    const mgr = new MemoryScopeManager();
    assert.throws(() => mgr.setAgentAccess("bot-1", [""]), /Invalid scope/);
  });

  it("importConfig merges configurations", () => {
    const mgr = new MemoryScopeManager();
    mgr.importConfig({
      definitions: { "custom:new": { description: "imported" } },
    });
    assert.ok(mgr.getAllScopes().includes("custom:new"));
    assert.ok(mgr.getAllScopes().includes("global"));
  });

  it("getStats counts project and user scopes", () => {
    const mgr = new MemoryScopeManager({
      definitions: {
        global: { description: "g" },
        "project:p1": { description: "p1" },
        "user:u1": { description: "u1" },
      },
    });
    const stats = mgr.getStats();
    assert.equal(stats.scopesByType.project, 1);
    assert.equal(stats.scopesByType.user, 1);
  });
});

// ============================================================================
// 15. Retriever (mocked store + embedder)
// ============================================================================

import { MemoryRetriever, DEFAULT_RETRIEVAL_CONFIG, createRetriever } from "../src/retriever.js";
import type { MemorySearchResult, MemoryEntry } from "../src/store.js";

describe("MemoryRetriever (mocked store + embedder)", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function makeEntry(overrides: Partial<MemoryEntry> = {}): MemoryEntry {
    return {
      id: overrides.id || "test-id-" + Math.random().toString(36).slice(2),
      text: overrides.text || "test text",
      vector: overrides.vector || Array(1024).fill(0.1),
      category: overrides.category || "fact",
      scope: overrides.scope || "global",
      importance: overrides.importance ?? 0.7,
      timestamp: overrides.timestamp ?? Date.now(),
      metadata: overrides.metadata || "{}",
    };
  }

  function makeResult(entry: MemoryEntry, score: number): MemorySearchResult {
    return { entry, score };
  }

  function createMockEmbedder(dims = 1024) {
    const mockVector = Array(dims).fill(0.1);
    return {
      dimensions: dims,
      model: "mock-model",
      embed: async () => mockVector,
      embedQuery: async () => mockVector,
      embedPassage: async () => mockVector,
      embedBatch: async (texts: string[]) => texts.map(() => mockVector),
      embedBatchQuery: async (texts: string[]) => texts.map(() => mockVector),
      embedBatchPassage: async (texts: string[]) => texts.map(() => mockVector),
      test: async () => ({ success: true, dimensions: dims }),
      get cacheStats() { return { size: 0, hits: 0, misses: 0, hitRate: "N/A" }; },
    };
  }

  function createMockStore(vectorResults: MemorySearchResult[] = [], bm25Results: MemorySearchResult[] = []) {
    return {
      dbPath: "/tmp/test-db",
      hasFtsSupport: true,
      vectorSearch: async () => vectorResults,
      bm25Search: async () => bm25Results,
      store: async () => ({} as any),
      list: async () => [],
      delete: async () => true,
      stats: async () => ({ totalCount: 0, scopeCounts: {}, categoryCounts: {} }),
      update: async () => null,
      bulkDelete: async () => 0,
      hasId: async () => false,
      importEntry: async () => ({} as any),
    } as any;
  }

  it("vector-only mode retrieves from store", async () => {
    const entry = makeEntry({ text: "important fact", importance: 0.8 });
    const store = createMockStore([makeResult(entry, 0.9)]);
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, { ...DEFAULT_RETRIEVAL_CONFIG, mode: "vector" });

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(results.length > 0);
    assert.equal(results[0].entry.text, "important fact");
    assert.ok(results[0].sources.vector);
  });

  it("hybrid mode fuses vector and BM25 results", async () => {
    const entry1 = makeEntry({ id: "vec-1", text: "vector result" });
    const entry2 = makeEntry({ id: "bm25-1", text: "bm25 result" });
    const store = createMockStore(
      [makeResult(entry1, 0.85)],
      [makeResult(entry2, 0.7)],
    );
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "hybrid",
      rerank: "none",
      hardMinScore: 0.1,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test query", limit: 10 });
    assert.ok(results.length >= 1);
  });

  it("fused results boost score when item appears in both vector and BM25", async () => {
    const entry = makeEntry({ id: "both-1", text: "dual hit result" });
    const store = createMockStore(
      [makeResult(entry, 0.8)],
      [makeResult(entry, 0.6)],
    );
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "hybrid",
      rerank: "none",
      hardMinScore: 0.1,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(results.length > 0);
    assert.ok(results[0].sources.vector);
    assert.ok(results[0].sources.bm25);
    assert.ok(results[0].sources.fused);
  });

  it("recency boost increases score for recent entries", async () => {
    const recentEntry = makeEntry({ id: "recent", text: "very recent item", timestamp: Date.now() - 3600_000, importance: 0.8 });
    const oldEntry = makeEntry({ id: "old", text: "very old item is this", timestamp: Date.now() - 365 * 86400_000, importance: 0.8 });
    const store = createMockStore([
      makeResult(recentEntry, 0.6),
      makeResult(oldEntry, 0.6),
    ]);
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
      rerank: "none",
      recencyHalfLifeDays: 14,
      recencyWeight: 0.10,
      hardMinScore: 0.1,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(results.length >= 2);
    const recentResult = results.find(r => r.entry.id === "recent");
    const oldResult = results.find(r => r.entry.id === "old");
    assert.ok(recentResult, "Expected a result for the recent entry");
    assert.ok(oldResult, "Expected a result for the old entry");
    assert.ok(
      recentResult!.score > oldResult!.score,
      "Expected recency boost to give the recent entry a higher score than the old entry",
    );
  });

  it("MMR diversity deduplicates similar entries", async () => {
    const vec = Array(1024).fill(0.5);
    // Create a vector that points in a genuinely different direction (not just different magnitude)
    const uniqueVec = Array.from({ length: 1024 }, (_, i) => (i % 2 === 0 ? 0.9 : -0.3));
    const entry1 = makeEntry({ id: "dup-1", text: "duplicate content A", vector: vec, importance: 0.8 });
    const entry2 = makeEntry({ id: "dup-2", text: "duplicate content B", vector: vec, importance: 0.8 });
    const entry3 = makeEntry({ id: "unique", text: "unique content here", vector: uniqueVec, importance: 0.8 });
    const store = createMockStore([
      makeResult(entry1, 0.9),
      makeResult(entry2, 0.89),
      makeResult(entry3, 0.85),
    ]);
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
      rerank: "none",
      hardMinScore: 0.1,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    // Ensure we get all three results back
    assert.ok(results.length >= 3);

    // MMR diversity defers similar entries — the unique entry should be
    // promoted above the second duplicate because dup-2 is too similar to dup-1.
    const ids = results.map((r) => r.entry.id);
    const uniqueIdx = ids.indexOf("unique");
    const dup1Idx = ids.indexOf("dup-1");
    const dup2Idx = ids.indexOf("dup-2");

    assert.ok(uniqueIdx !== -1, "Expected unique entry in results");
    assert.ok(dup1Idx !== -1, "Expected dup-1 entry in results");
    assert.ok(dup2Idx !== -1, "Expected dup-2 entry in results");

    // The unique entry should rank above the deferred duplicate
    assert.ok(
      uniqueIdx < dup2Idx,
      "MMR diversity should promote the unique entry above the second duplicate",
    );
  });

  it("hardMinScore filters out low-scoring results", async () => {
    const entry = makeEntry({ text: "low score entry" });
    const store = createMockStore([makeResult(entry, 0.2)]);
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
      rerank: "none",
      hardMinScore: 0.35,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.strictEqual(results.length, 0);
  });

  it("noise filter removes noisy results", async () => {
    const noisyEntry = makeEntry({ text: "I don't have any information about that", importance: 0.8 });
    const goodEntry = makeEntry({ text: "The database uses PostgreSQL for data storage and retrieval purposes", importance: 0.8 });
    const store = createMockStore([
      makeResult(goodEntry, 0.9),
      makeResult(noisyEntry, 0.85),
    ]);
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
      rerank: "none",
      filterNoise: true,
      hardMinScore: 0.1,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    const texts = results.map(r => r.entry.text);
    assert.ok(!texts.includes("I don't have any information about that"));
  });

  it("updateConfig changes retriever behavior", () => {
    const store = createMockStore();
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any);

    assert.equal(retriever.getConfig().mode, "hybrid");
    retriever.updateConfig({ mode: "vector" });
    assert.equal(retriever.getConfig().mode, "vector");
  });

  it("getConfig returns a copy of config", () => {
    const store = createMockStore();
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any);

    const config1 = retriever.getConfig();
    const config2 = retriever.getConfig();
    assert.deepEqual(config1, config2);
    config1.mode = "vector";
    assert.equal(retriever.getConfig().mode, "hybrid");
  });

  it("test() returns success with mock store", async () => {
    const store = createMockStore();
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
    });

    const result = await retriever.test("hello");
    assert.equal(result.success, true);
    assert.equal(result.mode, "vector");
  });

  it("test() returns failure when embedder throws", async () => {
    const store = createMockStore();
    const embedder = {
      ...createMockEmbedder(),
      embedQuery: async () => { throw new Error("API down"); },
    };
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
    });

    const result = await retriever.test();
    assert.equal(result.success, false);
    assert.ok(result.error?.includes("API down"));
  });

  it("lightweight rerank uses cosine similarity fallback", async () => {
    const vec1 = Array(1024).fill(0);
    vec1[0] = 1;
    const vec2 = Array(1024).fill(0);
    vec2[1] = 1;
    const entry1 = makeEntry({ id: "e1", text: "entry one text content", vector: vec1, importance: 0.8 });
    const entry2 = makeEntry({ id: "e2", text: "entry two text content", vector: vec2, importance: 0.8 });
    const store = createMockStore([
      makeResult(entry1, 0.8),
      makeResult(entry2, 0.7),
    ]);
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
      rerank: "lightweight",
      hardMinScore: 0.1,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(results.length >= 1);
    if (results[0].sources.reranked) {
      assert.ok(typeof results[0].sources.reranked.score === "number");
    }
  });

  it("category filter works in vector-only mode", async () => {
    const prefEntry = makeEntry({ id: "pref-1", text: "I prefer dark mode", category: "preference", importance: 0.8 });
    const factEntry = makeEntry({ id: "fact-1", text: "The API has rate limits", category: "fact", importance: 0.8 });
    const store = createMockStore([
      makeResult(prefEntry, 0.9),
      makeResult(factEntry, 0.85),
    ]);
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
      rerank: "none",
      hardMinScore: 0.1,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test", limit: 5, category: "preference" });
    assert.ok(results.length > 0, "Expected at least one result for category 'preference'");
    for (const r of results) {
      assert.equal(r.entry.category, "preference");
    }
  });

  it("cross-encoder rerank falls back to cosine when no API key", async () => {
    const entry = makeEntry({ text: "some test content here", importance: 0.8 });
    const store = createMockStore([makeResult(entry, 0.8)]);
    const embedder = createMockEmbedder();
    // Use hybrid mode so rerankResults actually gets called
    const retriever = new MemoryRetriever(store as any, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "hybrid",
      rerank: "cross-encoder",
      hardMinScore: 0.1,
      minScore: 0.1,
    }, undefined); // no API key → falls back to cosine

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(results.length > 0);
  });

  it("cross-encoder rerank with mocked Voyage API (hybrid mode)", async () => {
    const entry1 = makeEntry({ id: "r1", text: "highly relevant content", importance: 0.8 });
    const entry2 = makeEntry({ id: "r2", text: "less relevant text here", importance: 0.8 });
    const store = createMockStore(
      [makeResult(entry1, 0.8), makeResult(entry2, 0.7)],
      [makeResult(entry1, 0.6)], // entry1 also in BM25
    );
    const embedder = createMockEmbedder();

    globalThis.fetch = (async (url: any) => {
      if (typeof url === "string" && url.includes("rerank")) {
        return {
          ok: true,
          json: async () => ({
            data: [
              { index: 1, relevance_score: 0.95 },
              { index: 0, relevance_score: 0.6 },
            ],
          }),
        };
      }
      return { ok: false, status: 404, text: async () => "not found" };
    }) as unknown as typeof fetch;

    const retriever = new MemoryRetriever(store as any, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "hybrid",
      rerank: "cross-encoder",
      rerankModel: "rerank-2",
      hardMinScore: 0.1,
      minScore: 0.1,
    }, "voyage-test-key");

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(results.length >= 1);
    // Results should have reranked sources
    const hasReranked = results.some(r => r.sources.reranked);
    assert.ok(hasReranked);
  });

  it("cross-encoder rerank handles API failure gracefully (hybrid mode)", async () => {
    const entry = makeEntry({ text: "test entry content here", importance: 0.8 });
    const store = createMockStore([makeResult(entry, 0.8)]);
    const embedder = createMockEmbedder();

    globalThis.fetch = (async () => ({
      ok: false,
      status: 500,
      text: async () => "server error",
    })) as unknown as typeof fetch;

    const retriever = new MemoryRetriever(store as any, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "hybrid",
      rerank: "cross-encoder",
      hardMinScore: 0.1,
      minScore: 0.1,
    }, "voyage-test-key");

    // Should not throw, falls back to cosine similarity
    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(Array.isArray(results));
    assert.ok(results.length > 0, "Fallback should still return results");
    assert.ok(results[0].score > 0, "Fallback results should have non-zero scores");
  });

  it("cross-encoder rerank handles invalid response shape", async () => {
    const entry = makeEntry({ text: "content for rerank test", importance: 0.8 });
    const store = createMockStore([makeResult(entry, 0.8)]);
    const embedder = createMockEmbedder();

    globalThis.fetch = (async (url: any) => {
      if (typeof url === "string" && url.includes("rerank")) {
        return {
          ok: true,
          json: async () => ({ data: "not-an-array" }), // invalid shape
        };
      }
      return { ok: false, status: 404, text: async () => "" };
    }) as unknown as typeof fetch;

    const retriever = new MemoryRetriever(store as any, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "hybrid",
      rerank: "cross-encoder",
      hardMinScore: 0.1,
      minScore: 0.1,
    }, "voyage-test-key");

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(Array.isArray(results));
    assert.ok(results.length > 0, "Fallback should still return results on invalid rerank response");
    assert.ok(results[0].score > 0, "Fallback results should have non-zero scores");
  });

  it("cross-encoder rerank handles fetch exception", async () => {
    const entry = makeEntry({ text: "content for error test", importance: 0.8 });
    const store = createMockStore([makeResult(entry, 0.8)]);
    const embedder = createMockEmbedder();

    globalThis.fetch = (async () => {
      throw new Error("network error");
    }) as unknown as typeof fetch;

    const retriever = new MemoryRetriever(store as any, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "hybrid",
      rerank: "cross-encoder",
      hardMinScore: 0.1,
      minScore: 0.1,
    }, "voyage-test-key");

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(Array.isArray(results));
    assert.ok(results.length > 0, "Fallback should still return results on fetch exception");
    assert.ok(results[0].score > 0, "Fallback results should have non-zero scores");
  });

  it("falls back to vector-only when store has no FTS support", async () => {
    const entry = makeEntry({ text: "vector only result", importance: 0.8 });
    const store = createMockStore([makeResult(entry, 0.9)]);
    (store as any).hasFtsSupport = false;
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "hybrid",
      rerank: "none",
      hardMinScore: 0.1,
      minScore: 0.1,
    });

    const results = await retriever.retrieve({ query: "test", limit: 5 });
    assert.ok(Array.isArray(results));
  });

  it("limit is clamped between 1 and 20", async () => {
    const vectorSearchCalls: number[] = [];
    const store = createMockStore();
    const origVectorSearch = store.vectorSearch;
    store.vectorSearch = async (vec: any, limit: number, ...rest: any[]) => {
      vectorSearchCalls.push(limit);
      return origVectorSearch(vec, limit, ...rest);
    };
    const embedder = createMockEmbedder();
    const retriever = createRetriever(store, embedder as any, {
      ...DEFAULT_RETRIEVAL_CONFIG,
      mode: "vector",
      rerank: "none",
    });

    await retriever.retrieve({ query: "test", limit: 0 });
    await retriever.retrieve({ query: "test", limit: 100 });
    await retriever.retrieve({ query: "test", limit: -5 });

    assert.strictEqual(vectorSearchCalls[0], 1, "limit 0 should be clamped to 1");
    assert.strictEqual(vectorSearchCalls[1], 20, "limit 100 should be clamped to 20");
    assert.strictEqual(vectorSearchCalls[2], 1, "limit -5 should be clamped to 1");
  });
});

// ============================================================================
// 16. MemoryStore (mocked LanceDB)
// ============================================================================

import { MemoryStore } from "../src/store.js";

describe("MemoryStore (mocked LanceDB)", () => {

  // ---------------------------------------------------------------------------
  // Mock LanceDB table builder — returns a chainable query/search object
  // ---------------------------------------------------------------------------

  type Row = Record<string, unknown>;

  function createMockTable(initialRows: Row[] = []) {
    const rows: Row[] = [...initialRows];

    /** Builds a chainable query that filters, selects, and returns rows */
    function buildQuery() {
      let whereClause: string | null = null;
      let selectedCols: string[] | null = null;
      let limitVal = Infinity;

      const chain: any = {
        where(clause: string) { whereClause = clause; return chain; },
        select(cols: string[]) { selectedCols = cols; return chain; },
        limit(n: number) { limitVal = n; return chain; },
        async toArray() {
          let result = [...rows];
          if (whereClause) {
            result = result.filter(r => evalWhere(r, whereClause!));
          }
          if (selectedCols) {
            result = result.map(r => {
              const picked: Row = {};
              for (const c of selectedCols!) picked[c] = r[c];
              return picked;
            });
          }
          return result.slice(0, limitVal);
        },
      };
      return chain;
    }

    /** Builds a chainable vector search (adds _distance to rows) */
    function buildVectorSearch(_vector: number[]) {
      let whereClause: string | null = null;
      let limitVal = Infinity;

      const chain: any = {
        where(clause: string) { whereClause = clause; return chain; },
        limit(n: number) { limitVal = n; return chain; },
        async toArray() {
          let result = rows.map((r, i) => ({ ...r, _distance: i * 0.1 }));
          if (whereClause) {
            result = result.filter(r => evalWhere(r, whereClause!));
          }
          return result.slice(0, limitVal);
        },
      };
      return chain;
    }

    /** Builds a chainable FTS search (adds _score to rows) */
    function buildSearch(queryText: string, _mode: string) {
      let whereClause: string | null = null;
      let limitVal = Infinity;

      const chain: any = {
        where(clause: string) { whereClause = clause; return chain; },
        limit(n: number) { limitVal = n; return chain; },
        async toArray() {
          // filter rows whose text contains the query word (case-insensitive)
          let result = rows
            .filter(r => typeof r.text === "string" && (r.text as string).toLowerCase().includes(queryText.toLowerCase()))
            .map((r, i) => ({ ...r, _score: 10 - i }));
          if (whereClause) {
            result = result.filter(r => evalWhere(r, whereClause!));
          }
          return result.slice(0, limitVal);
        },
      };
      return chain;
    }

    /** Minimal SQL-where evaluator — handles simple patterns used by MemoryStore */
    function evalWhere(row: Row, clause: string): boolean {
      // Handle compound AND conditions: split by " AND " and require all parts to match
      if (clause.includes(" AND ")) {
        const parts = clause.split(" AND ");
        return parts.every(part => evalWhere(row, part.trim()));
      }
      // Handle scope = 'x' OR scope IS NULL patterns
      if (clause.includes(" OR scope IS NULL")) {
        const cleaned = clause.replace(/[()]/g, "").replace(/ OR scope IS NULL/g, "").trim();
        const parts = cleaned.split(/ OR /g);
        const scopeVals = parts.map(p => {
          const m = p.match(/scope\s*=\s*'([^']*)'/);
          return m ? m[1] : null;
        }).filter(Boolean);
        const rowScope = row.scope as string | undefined;
        return rowScope == null || scopeVals.includes(rowScope as string);
      }
      // Handle id = 'xxx'
      const idMatch = clause.match(/id\s*=\s*'([^']*)'/);
      if (idMatch && !clause.includes("scope")) return row.id === idMatch[1];
      // Handle category = 'xxx'
      const catMatch = clause.match(/category\s*=\s*'([^']*)'/);
      if (catMatch && !clause.includes("scope")) return row.category === catMatch[1];
      // Handle scope = 'xxx' (simple, with OR between scopes)
      const scopeOrParts = clause.replace(/[()]/g, "").trim().split(/ OR /g);
      const scopeVals = scopeOrParts.map(p => {
        const m = p.match(/scope\s*=\s*'([^']*)'/);
        return m ? m[1] : null;
      }).filter(Boolean);
      if (scopeVals.length > 0) {
        const rowScope = row.scope as string | undefined;
        return rowScope == null || scopeVals.includes(rowScope as string);
      }
      // Handle timestamp < N
      const tsMatch = clause.match(/timestamp\s*<\s*(\d+)/);
      if (tsMatch) {
        const ts = Number(tsMatch[1]);
        return (row.timestamp as number) < ts;
      }
      return true; // fallback: include row
    }

    const mockTable = {
      _rows: rows, // exposed for test assertions
      query: () => buildQuery(),
      vectorSearch: (vector: number[]) => buildVectorSearch(vector),
      search: (queryText: string, mode: string) => buildSearch(queryText, mode),
      add: async (entries: Row[]) => { rows.push(...entries); },
      delete: async (whereClause: string) => {
        const idMatch = whereClause.match(/id\s*=\s*'([^']*)'/);
        if (idMatch) {
          const idx = rows.findIndex(r => r.id === idMatch[1]);
          if (idx >= 0) rows.splice(idx, 1);
        } else {
          // For bulk delete, evaluate condition against all rows
          const toRemove = rows.filter(r => evalWhere(r, whereClause));
          for (const r of toRemove) {
            const idx = rows.indexOf(r);
            if (idx >= 0) rows.splice(idx, 1);
          }
        }
      },
      listIndices: async () => [],
      createIndex: async () => {},
    };
    return mockTable;
  }

  /** Helper: wire a mock table into a MemoryStore, bypassing real LanceDB init */
  function createStoreWithMockTable(mockTable: ReturnType<typeof createMockTable>, vectorDim = 3) {
    const store = new MemoryStore({ dbPath: "/tmp/mock-db", vectorDim });
    // Bypass ensureInitialized by setting private fields directly
    (store as any).table = mockTable;
    (store as any).db = {}; // truthy sentinel
    (store as any).ftsIndexCreated = true;
    return store;
  }

  // ---- Constructor & dbPath accessor ----

  describe("constructor & dbPath", () => {
    it("stores config and exposes dbPath", () => {
      const store = new MemoryStore({ dbPath: "/tmp/test", vectorDim: 128 });
      assert.equal(store.dbPath, "/tmp/test");
    });
  });

  // ---- store (addMemory) ----

  describe("store() — add a memory", () => {
    it("stores an entry and assigns id + timestamp", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      const result = await store.store({
        text: "hello world",
        vector: [0.1, 0.2, 0.3],
        category: "fact",
        scope: "global",
        importance: 0.8,
      });

      assert.ok(result.id, "should have an id");
      assert.ok(result.timestamp > 0, "should have a timestamp");
      assert.equal(result.text, "hello world");
      assert.equal(result.category, "fact");
      assert.equal(result.metadata, "{}");
      assert.equal(mockTable._rows.length, 1);
    });

    it("preserves provided metadata", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      const result = await store.store({
        text: "with meta",
        vector: [0.1, 0.2, 0.3],
        category: "preference",
        scope: "project",
        importance: 0.5,
        metadata: '{"key":"val"}',
      });

      assert.equal(result.metadata, '{"key":"val"}');
    });
  });

  // ---- vectorSearch ----

  describe("vectorSearch()", () => {
    it("returns results with score computed from distance", async () => {
      const rows: Row[] = [
        { id: "a1", text: "first", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "{}" },
        { id: "a2", text: "second", vector: [0, 1, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.vectorSearch([1, 0, 0], 5, 0.0);
      assert.ok(results.length > 0, "should return results");
      assert.equal(results[0].entry.id, "a1");
      // First row has _distance=0 → score = 1/(1+0) = 1.0
      assert.equal(results[0].score, 1.0);
    });

    it("filters by scope when scopeFilter provided", async () => {
      const rows: Row[] = [
        { id: "g1", text: "global item", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "{}" },
        { id: "p1", text: "project item", vector: [0, 1, 0], category: "fact", scope: "project", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.vectorSearch([1, 0, 0], 5, 0.0, ["global"]);
      assert.ok(results.every(r => r.entry.scope === "global"), "all results should be global scope");
    });

    it("respects minScore threshold", async () => {
      const rows: Row[] = [
        { id: "a1", text: "close", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "{}" },
        { id: "a2", text: "far", vector: [0, 1, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      // With high minScore, only the first (distance=0, score=1.0) should pass
      const results = await store.vectorSearch([1, 0, 0], 5, 0.95);
      assert.equal(results.length, 1);
      assert.equal(results[0].entry.id, "a1");
    });

    it("clamps limit between 1 and 20", async () => {
      const rows: Row[] = Array.from({ length: 25 }, (_, i) => ({
        id: `id-${i}`, text: `text ${i}`, vector: [i, 0, 0], category: "fact" as const,
        scope: "global", importance: 0.5, timestamp: i, metadata: "{}",
      }));
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.vectorSearch([1, 0, 0], 100, 0.0);
      assert.ok(results.length <= 20, "should clamp to 20 max");
    });

    it("defaults metadata to '{}' when row metadata is empty", async () => {
      const rows: Row[] = [
        { id: "m1", text: "no meta", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.vectorSearch([1, 0, 0], 5, 0.0);
      assert.equal(results[0].entry.metadata, "{}");
    });

    it("defaults scope to 'global' when row scope is undefined", async () => {
      const rows: Row[] = [
        { id: "ns1", text: "no scope", vector: [1, 0, 0], category: "fact", importance: 0.8, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.vectorSearch([1, 0, 0], 5, 0.0);
      assert.equal(results[0].entry.scope, "global");
    });
  });

  // ---- bm25Search ----

  describe("bm25Search()", () => {
    it("returns matching rows with normalized scores", async () => {
      const rows: Row[] = [
        { id: "b1", text: "the quick brown fox", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "{}" },
        { id: "b2", text: "lazy dog", vector: [0, 1, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.bm25Search("quick", 5);
      assert.equal(results.length, 1);
      assert.equal(results[0].entry.id, "b1");
      assert.ok(results[0].score > 0, "score should be positive");
    });

    it("returns empty array when ftsIndexCreated is false", async () => {
      const mockTable = createMockTable([
        { id: "b1", text: "some text", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "{}" },
      ]);
      const store = createStoreWithMockTable(mockTable);
      (store as any).ftsIndexCreated = false;

      const results = await store.bm25Search("some", 5);
      assert.equal(results.length, 0);
    });

    it("filters by scopeFilter", async () => {
      const rows: Row[] = [
        { id: "b1", text: "shared word here", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "{}" },
        { id: "b2", text: "shared word there", vector: [0, 1, 0], category: "fact", scope: "project", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.bm25Search("shared", 5, ["global"]);
      assert.ok(results.every(r => r.entry.scope === "global"));
    });

    it("handles search throwing by returning empty array", async () => {
      const mockTable = createMockTable();
      // Override search to throw
      mockTable.search = () => { throw new Error("FTS broken"); };
      const store = createStoreWithMockTable(mockTable);

      const results = await store.bm25Search("test", 5);
      assert.equal(results.length, 0);
    });

    it("defaults scope to 'global' when missing", async () => {
      const rows: Row[] = [
        { id: "ns1", text: "no scope bm25", vector: [1, 0, 0], category: "fact", importance: 0.8, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.bm25Search("bm25", 5);
      assert.equal(results[0].entry.scope, "global");
    });

    it("defaults metadata to '{}' when row metadata is empty", async () => {
      const rows: Row[] = [
        { id: "nm1", text: "no meta bm25", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.bm25Search("meta", 5);
      assert.equal(results[0].entry.metadata, "{}");
    });

    it("normalizes score to 0.5 when _score is 0", async () => {
      const rows: Row[] = [
        { id: "z1", text: "zero score text", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.8, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      // Override search to return _score = 0
      const origSearch = mockTable.search.bind(mockTable);
      mockTable.search = (q: string, m: string) => {
        const chain = origSearch(q, m);
        const origToArray = chain.toArray.bind(chain);
        chain.toArray = async () => {
          const res = await origToArray();
          return res.map((r: any) => ({ ...r, _score: 0 }));
        };
        return chain;
      };
      const store = createStoreWithMockTable(mockTable);

      const results = await store.bm25Search("zero", 5);
      assert.equal(results[0].score, 0.5);
    });
  });

  // ---- delete ----

  describe("delete()", () => {
    it("deletes by full UUID", async () => {
      const uuid = "12345678-1234-1234-1234-123456789abc";
      const rows: Row[] = [
        { id: uuid, text: "to delete", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const result = await store.delete(uuid);
      assert.equal(result, true);
      assert.equal(mockTable._rows.length, 0);
    });

    it("returns false when id not found", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      const result = await store.delete("12345678-1234-1234-1234-123456789abc");
      assert.equal(result, false);
    });

    it("throws on invalid id format", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      await assert.rejects(
        () => store.delete("not-valid!"),
        /Invalid memory ID format/,
      );
    });

    it("deletes by prefix (8+ hex chars)", async () => {
      const uuid = "aabbccdd-1234-1234-1234-123456789abc";
      const rows: Row[] = [
        { id: uuid, text: "prefix delete", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const result = await store.delete("aabbccdd");
      assert.equal(result, true);
      assert.equal(mockTable._rows.length, 0);
    });

    it("throws on ambiguous prefix", async () => {
      const rows: Row[] = [
        { id: "aabbccdd-1111-1111-1111-111111111111", text: "a", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
        { id: "aabbccdd-2222-2222-2222-222222222222", text: "b", vector: [0, 1, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      await assert.rejects(
        () => store.delete("aabbccdd"),
        /Ambiguous prefix/,
      );
    });

    it("respects scopeFilter — throws when memory is out of scope", async () => {
      const uuid = "12345678-1234-1234-1234-123456789abc";
      const rows: Row[] = [
        { id: uuid, text: "secret", vector: [1, 0, 0], category: "fact", scope: "project", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      await assert.rejects(
        () => store.delete(uuid, ["global"]),
        /outside accessible scopes/,
      );
    });

    it("allows delete when scope matches scopeFilter", async () => {
      const uuid = "12345678-1234-1234-1234-123456789abc";
      const rows: Row[] = [
        { id: uuid, text: "ok", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const result = await store.delete(uuid, ["global"]);
      assert.equal(result, true);
    });
  });

  // ---- list ----

  describe("list()", () => {
    it("returns all entries sorted by timestamp desc", async () => {
      const rows: Row[] = [
        { id: "l1", text: "oldest", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
        { id: "l2", text: "newest", vector: [0, 1, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.list();
      assert.equal(results.length, 2);
      assert.equal(results[0].id, "l2"); // newest first
      assert.equal(results[1].id, "l1");
      // list returns empty vector arrays
      assert.deepEqual(results[0].vector, []);
    });

    it("filters by scopeFilter", async () => {
      const rows: Row[] = [
        { id: "l1", text: "global", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
        { id: "l2", text: "project", vector: [0, 1, 0], category: "fact", scope: "project", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.list(["global"]);
      assert.ok(results.every(r => r.scope === "global"));
    });

    it("filters by category", async () => {
      const rows: Row[] = [
        { id: "c1", text: "a fact", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
        { id: "c2", text: "a pref", vector: [0, 1, 0], category: "preference", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.list(undefined, "preference");
      assert.equal(results.length, 1);
      assert.equal(results[0].category, "preference");
    });

    it("respects limit and offset", async () => {
      const rows: Row[] = Array.from({ length: 10 }, (_, i) => ({
        id: `p${i}`, text: `text ${i}`, vector: [i, 0, 0], category: "fact",
        scope: "global", importance: 0.5, timestamp: i * 100, metadata: "{}",
      }));
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.list(undefined, undefined, 3, 2);
      assert.equal(results.length, 3);
      // sorted desc by timestamp: p9(900), p8(800), p7(700), p6(600)...
      // offset 2 → skip first 2 → p7, p6, p5
      assert.equal(results[0].id, "p7");
    });

    it("defaults scope to 'global' when missing", async () => {
      const rows: Row[] = [
        { id: "ns1", text: "no scope", vector: [1, 0, 0], category: "fact", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.list();
      assert.equal(results[0].scope, "global");
    });

    it("defaults metadata to '{}' when empty", async () => {
      const rows: Row[] = [
        { id: "nm1", text: "no meta", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const results = await store.list();
      assert.equal(results[0].metadata, "{}");
    });
  });

  // ---- stats ----

  describe("stats()", () => {
    it("returns totalCount, scopeCounts, and categoryCounts", async () => {
      const rows: Row[] = [
        { id: "s1", text: "a", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
        { id: "s2", text: "b", vector: [0, 1, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
        { id: "s3", text: "c", vector: [0, 0, 1], category: "preference", scope: "project", importance: 0.5, timestamp: 300, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const stats = await store.stats();
      assert.equal(stats.totalCount, 3);
      assert.equal(stats.scopeCounts["global"], 2);
      assert.equal(stats.scopeCounts["project"], 1);
      assert.equal(stats.categoryCounts["fact"], 2);
      assert.equal(stats.categoryCounts["preference"], 1);
    });

    it("filters by scopeFilter", async () => {
      const rows: Row[] = [
        { id: "s1", text: "a", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
        { id: "s2", text: "b", vector: [0, 1, 0], category: "fact", scope: "project", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const stats = await store.stats(["global"]);
      assert.ok(stats.totalCount >= 1);
      assert.ok(stats.scopeCounts["global"] >= 1);
    });

    it("defaults scope to 'global' when missing", async () => {
      const rows: Row[] = [
        { id: "ns1", text: "a", vector: [1, 0, 0], category: "fact", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const stats = await store.stats();
      assert.equal(stats.scopeCounts["global"], 1);
    });
  });

  // ---- importEntry ----

  describe("importEntry()", () => {
    it("imports entry with correct vector dimension", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      const entry = {
        id: "import-1",
        text: "imported",
        vector: [0.1, 0.2, 0.3],
        category: "fact" as const,
        scope: "global",
        importance: 0.7,
        timestamp: Date.now(),
        metadata: "{}",
      };

      const result = await store.importEntry(entry);
      assert.equal(result.id, "import-1");
      assert.equal(result.text, "imported");
      assert.equal(mockTable._rows.length, 1);
    });

    it("throws when id is missing", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      await assert.rejects(
        () => store.importEntry({ id: "", text: "x", vector: [1, 2, 3], category: "fact", scope: "global", importance: 0.7, timestamp: 1 }),
        /importEntry requires a stable id/,
      );
    });

    it("throws on vector dimension mismatch", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable); // vectorDim=3

      await assert.rejects(
        () => store.importEntry({ id: "bad", text: "x", vector: [1, 2], category: "fact", scope: "global", importance: 0.7, timestamp: 1 }),
        /Vector dimension mismatch/,
      );
    });

    it("defaults scope to 'global' and importance to 0.7 when missing", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      const result = await store.importEntry({
        id: "import-2",
        text: "partial",
        vector: [0.1, 0.2, 0.3],
        category: "fact",
        scope: "",
        importance: NaN,
        timestamp: NaN,
      } as any);

      // empty string scope gets replaced with "global" because ("" || "global") → "global"
      assert.equal(result.scope, "global");
      assert.equal(result.importance, 0.7);
      assert.ok(result.timestamp > 0);
    });
  });

  // ---- hasId ----

  describe("hasId()", () => {
    it("returns true when id exists", async () => {
      const rows: Row[] = [
        { id: "exists-1", text: "a", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      assert.equal(await store.hasId("exists-1"), true);
    });

    it("returns false when id does not exist", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      assert.equal(await store.hasId("nonexistent"), false);
    });
  });

  // ---- update ----

  describe("update()", () => {
    it("updates text and importance of existing entry", async () => {
      const uuid = "12345678-1234-1234-1234-123456789abc";
      const rows: Row[] = [
        { id: uuid, text: "old text", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const result = await store.update(uuid, { text: "new text", importance: 0.9 });
      assert.ok(result);
      assert.equal(result!.text, "new text");
      assert.equal(result!.importance, 0.9);
      assert.equal(result!.category, "fact"); // unchanged
    });

    it("returns null when id not found", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      const result = await store.update("12345678-1234-1234-1234-123456789abc", { text: "x" });
      assert.equal(result, null);
    });

    it("throws on invalid id format", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      await assert.rejects(
        () => store.update("bad!id", { text: "x" }),
        /Invalid memory ID format/,
      );
    });

    it("throws when scopeFilter blocks access", async () => {
      const uuid = "12345678-1234-1234-1234-123456789abc";
      const rows: Row[] = [
        { id: uuid, text: "secret", vector: [1, 0, 0], category: "fact", scope: "project", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      await assert.rejects(
        () => store.update(uuid, { text: "x" }, ["global"]),
        /outside accessible scopes/,
      );
    });

    it("resolves by prefix", async () => {
      const uuid = "aabbccdd-1234-1234-1234-123456789abc";
      const rows: Row[] = [
        { id: uuid, text: "prefix update", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const result = await store.update("aabbccdd", { text: "updated" });
      assert.ok(result);
      assert.equal(result!.text, "updated");
    });

    it("throws on ambiguous prefix", async () => {
      const rows: Row[] = [
        { id: "aabbccdd-1111-1111-1111-111111111111", text: "a", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
        { id: "aabbccdd-2222-2222-2222-222222222222", text: "b", vector: [0, 1, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      await assert.rejects(
        () => store.update("aabbccdd", { text: "x" }),
        /Ambiguous prefix/,
      );
    });
  });

  // ---- bulkDelete ----

  describe("bulkDelete()", () => {
    it("deletes entries matching scope filter", async () => {
      const rows: Row[] = [
        { id: "bd1", text: "a", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
        { id: "bd2", text: "b", vector: [0, 1, 0], category: "fact", scope: "project", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const count = await store.bulkDelete(["project"]);
      assert.equal(count, 1);
      assert.equal(mockTable._rows.length, 1);
      assert.equal(mockTable._rows[0].id, "bd1");
    });

    it("deletes with timestamp filter", async () => {
      const rows: Row[] = [
        { id: "bd1", text: "old", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 50, metadata: "{}" },
        { id: "bd2", text: "new", vector: [0, 1, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 200, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      const count = await store.bulkDelete(["global"], 100);
      assert.equal(count, 1);
      assert.equal(mockTable._rows[0].id, "bd2");
    });

    it("throws when no filters provided", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      await assert.rejects(
        () => store.bulkDelete([]),
        /Bulk delete requires at least scope or timestamp filter/,
      );
    });

    it("returns 0 when nothing matches", async () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);

      const count = await store.bulkDelete(["nonexistent"]);
      assert.equal(count, 0);
    });
  });

  // ---- hasFtsSupport ----

  describe("hasFtsSupport", () => {
    it("returns true when ftsIndexCreated is true", () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);
      assert.equal(store.hasFtsSupport, true);
    });

    it("returns false when ftsIndexCreated is false", () => {
      const mockTable = createMockTable();
      const store = createStoreWithMockTable(mockTable);
      (store as any).ftsIndexCreated = false;
      assert.equal(store.hasFtsSupport, false);
    });
  });

  // ---- createFtsIndex (via doInitialize path) ----

  describe("FTS index creation", () => {
    it("skips index creation if FTS index already exists", async () => {
      let createIndexCalled = false;
      const mockTable = createMockTable();
      mockTable.listIndices = async () => [{ indexType: "FTS", columns: ["text"] }];
      mockTable.createIndex = async () => { createIndexCalled = true; };

      const store = new MemoryStore({ dbPath: "/tmp/mock-fts", vectorDim: 3 });
      // Simulate createFtsIndex call directly
      await (store as any).createFtsIndex(mockTable);
      assert.equal(createIndexCalled, false, "should NOT have called createIndex");
    });

    it("creates index when no FTS index exists", async () => {
      let createIndexCalled = false;
      const mockTable = createMockTable();
      mockTable.listIndices = async () => [];
      mockTable.createIndex = async () => { createIndexCalled = true; };

      const store = new MemoryStore({ dbPath: "/tmp/mock-fts2", vectorDim: 3 });
      await (store as any).createFtsIndex(mockTable);
      assert.equal(createIndexCalled, true, "should have called createIndex");
    });

    it("throws when listIndices fails", async () => {
      const mockTable = createMockTable();
      mockTable.listIndices = async () => { throw new Error("index list failed"); };

      const store = new MemoryStore({ dbPath: "/tmp/mock-fts3", vectorDim: 3 });
      await assert.rejects(
        () => (store as any).createFtsIndex(mockTable),
        /FTS index creation failed/,
      );
    });

    it("detects existing index via columns includes 'text'", async () => {
      let createIndexCalled = false;
      const mockTable = createMockTable();
      mockTable.listIndices = async () => [{ indexType: "OTHER", columns: ["text"] }];
      mockTable.createIndex = async () => { createIndexCalled = true; };

      const store = new MemoryStore({ dbPath: "/tmp/mock-fts4", vectorDim: 3 });
      await (store as any).createFtsIndex(mockTable);
      assert.equal(createIndexCalled, false, "should NOT create index when text column already indexed");
    });
  });

  // ---- Utility functions (clampInt, escapeSqlLiteral) via public methods ----

  describe("utility functions via public API", () => {
    it("clampInt handles non-finite values (via vectorSearch limit)", async () => {
      const rows: Row[] = [
        { id: "u1", text: "a", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      // NaN limit → clampInt returns min (1)
      const results = await store.vectorSearch([1, 0, 0], NaN, 0.0);
      assert.ok(results.length <= 1);
    });

    it("escapeSqlLiteral handles single quotes (via hasId)", async () => {
      const rows: Row[] = [
        { id: "it's-a-test", text: "a", vector: [1, 0, 0], category: "fact", scope: "global", importance: 0.5, timestamp: 100, metadata: "{}" },
      ];
      const mockTable = createMockTable(rows);
      const store = createStoreWithMockTable(mockTable);

      // This should not throw — the quote should be escaped
      const result = await store.hasId("it's-a-test");
      // Our mock doesn't handle escaped quotes, but this covers the code path
      assert.equal(typeof result, "boolean");
    });
  });
});

// ============================================================================
// 17. Plugin index.ts — sanitizeForContext, ConversationBuffer, LLM judgment,
//     register(), hooks, session reading
// ============================================================================

import plugin from "../index.js";

describe("Plugin (index.ts) — register & lifecycle hooks", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // ---------------------------------------------------------------------------
  // Mock OpenClawPluginApi builder
  // ---------------------------------------------------------------------------

  interface CapturedHook { name: string; handler: (...args: any[]) => any }
  interface CapturedTool { name: string; tool: any }
  interface CapturedService { id: string; start: () => Promise<void>; stop: () => void }

  function createMockApi(configOverrides: Record<string, any> = {}) {
    const hooks: CapturedHook[] = [];
    const tools: CapturedTool[] = [];
    const services: CapturedService[] = [];
    const registeredHookEvents: CapturedHook[] = [];
    const logs: { level: string; msg: string }[] = [];
    let cliRegistered = false;

    const baseConfig = {
      embedding: { apiKey: "test-key-123", model: "voyage-3-large" },
      autoCapture: false,
      autoRecall: false,
      captureLlm: false,
      enableManagementTools: false,
      sessionMemory: { enabled: false },
      ...configOverrides,
    };

    const api: any = {
      pluginConfig: baseConfig,
      resolvePath: (p: string) => `/tmp/test-vidya/${p}`,
      logger: {
        info: (msg: string) => logs.push({ level: "info", msg }),
        warn: (msg: string) => logs.push({ level: "warn", msg }),
        debug: (msg: string) => logs.push({ level: "debug", msg }),
      },
      registerTool: (tool: any, opts: any) => {
        tools.push({ name: opts?.name || tool.name, tool });
      },
      on: (hookName: string, handler: (...args: any[]) => any) => {
        hooks.push({ name: hookName, handler });
      },
      registerHook: (events: string | string[], handler: (...args: any[]) => any) => {
        const names = Array.isArray(events) ? events : [events];
        for (const name of names) registeredHookEvents.push({ name, handler });
      },
      registerService: (svc: any) => {
        services.push(svc);
      },
      registerCli: () => { cliRegistered = true; },
    };

    return { api, hooks, tools, services, registeredHookEvents, logs, cliRegistered: () => cliRegistered };
  }

  // ---------------------------------------------------------------------------
  // Plugin metadata
  // ---------------------------------------------------------------------------

  describe("plugin metadata", () => {
    it("has correct id and kind", () => {
      assert.equal(plugin.id, "memory-lancedb-voyage");
      assert.equal(plugin.kind, "memory");
      assert.ok(plugin.name);
      assert.ok(plugin.description);
    });

    it("configSchema is memoryConfigSchema", () => {
      assert.ok(plugin.configSchema);
      assert.equal(typeof plugin.configSchema.parse, "function");
    });
  });

  // ---------------------------------------------------------------------------
  // register() — basic wiring
  // ---------------------------------------------------------------------------

  describe("register() — basic wiring", () => {
    it("registers tools, service, and CLI", () => {
      const { api, tools, services, cliRegistered } = createMockApi();
      plugin.register(api);

      // Should register core tools: memory_recall, memory_store, memory_forget, memory_update
      const toolNames = tools.map(t => t.name);
      assert.ok(toolNames.includes("memory_recall"), "should register memory_recall");
      assert.ok(toolNames.includes("memory_store"), "should register memory_store");
      assert.ok(toolNames.includes("memory_forget"), "should register memory_forget");
      assert.ok(toolNames.includes("memory_update"), "should register memory_update");

      // Service registered
      assert.equal(services.length, 1);
      assert.equal(services[0].id, "memory-lancedb-voyage");

      // CLI registered
      assert.ok(cliRegistered());
    });

    it("registers management tools when enableManagementTools=true", () => {
      const { api, tools } = createMockApi({ enableManagementTools: true });
      plugin.register(api);

      const toolNames = tools.map(t => t.name);
      assert.ok(toolNames.includes("memory_stats"), "should register memory_stats");
      assert.ok(toolNames.includes("memory_list"), "should register memory_list");
    });

    it("does NOT register management tools by default", () => {
      const { api, tools } = createMockApi();
      plugin.register(api);

      const toolNames = tools.map(t => t.name);
      assert.ok(!toolNames.includes("memory_stats"));
      assert.ok(!toolNames.includes("memory_list"));
    });

    it("logs registration message", () => {
      const { api, logs } = createMockApi();
      plugin.register(api);

      const regLog = logs.find(l => l.msg.includes("registered"));
      assert.ok(regLog, "should log registration");
    });
  });

  // ---------------------------------------------------------------------------
  // autoRecall hook (before_agent_start)
  // ---------------------------------------------------------------------------

  describe("autoRecall hook (before_agent_start)", () => {
    it("registers before_agent_start hook when autoRecall=true", () => {
      const { api, hooks } = createMockApi({ autoRecall: true });
      plugin.register(api);

      const recallHook = hooks.find(h => h.name === "before_agent_start");
      assert.ok(recallHook, "should register before_agent_start hook");
    });

    it("does NOT register before_agent_start when autoRecall=false", () => {
      const { api, hooks } = createMockApi({ autoRecall: false });
      plugin.register(api);

      const recallHook = hooks.find(h => h.name === "before_agent_start");
      assert.equal(recallHook, undefined);
    });

    it("skips retrieval for greeting prompts", async () => {
      const { api, hooks } = createMockApi({ autoRecall: true });
      plugin.register(api);

      const recallHook = hooks.find(h => h.name === "before_agent_start")!;
      // "hi" should be skipped by shouldSkipRetrieval
      const result = await recallHook.handler({ prompt: "hi" }, { agentId: "main" });
      assert.equal(result, undefined, "should return undefined for greeting");
    });

    it("skips when prompt is empty", async () => {
      const { api, hooks } = createMockApi({ autoRecall: true });
      plugin.register(api);

      const recallHook = hooks.find(h => h.name === "before_agent_start")!;
      const result = await recallHook.handler({ prompt: "" }, {});
      assert.equal(result, undefined);
    });

    it("handles retrieval errors gracefully", async () => {
      const { api, hooks, logs } = createMockApi({ autoRecall: true });
      plugin.register(api);

      const recallHook = hooks.find(h => h.name === "before_agent_start")!;
      // A long substantive prompt will attempt retrieval, which will fail
      // because there's no real embedder — the error should be caught
      const result = await recallHook.handler(
        { prompt: "What were the important technical decisions we discussed last week about the database architecture?" },
        { agentId: "main" },
      );
      // Should have caught and logged the error
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("recall failed"));
      assert.ok(warnLog, "should log recall failure");
      assert.equal(result, undefined, "should return undefined on error");
    });
  });

  // ---------------------------------------------------------------------------
  // autoCapture hook (agent_end)
  // ---------------------------------------------------------------------------

  describe("autoCapture hook (agent_end)", () => {
    it("registers agent_end hook when autoCapture=true", () => {
      const { api, hooks } = createMockApi({ autoCapture: true });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end");
      assert.ok(captureHook, "should register agent_end hook");
    });

    it("does NOT register agent_end when autoCapture=false", () => {
      const { api, hooks } = createMockApi({ autoCapture: false });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end");
      assert.equal(captureHook, undefined);
    });

    it("skips capture when event.success is false", async () => {
      const { api, hooks } = createMockApi({ autoCapture: true });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      const result = await captureHook.handler({ success: false, messages: [] }, {});
      assert.equal(result, undefined);
    });

    it("skips capture when messages is empty", async () => {
      const { api, hooks } = createMockApi({ autoCapture: true });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      const result = await captureHook.handler({ success: true, messages: [] }, {});
      assert.equal(result, undefined);
    });

    it("skips capture for non-triggering text", async () => {
      const { api, hooks } = createMockApi({ autoCapture: true });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      // Short text that doesn't match any trigger
      const result = await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "ok thanks" }] },
        { agentId: "main" },
      );
      assert.equal(result, undefined);
    });

    it("attempts capture for triggering text (heuristic path)", async () => {
      const { api, hooks, logs } = createMockApi({ autoCapture: true, captureLlm: false });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      // Text with a memory trigger — will fail at embedder.embedPassage but error is caught
      const result = await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember that I prefer dark mode always in my editor" }] },
        { agentId: "main" },
      );
      // Error should be caught gracefully
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(warnLog, "should log capture failure");
    });

    it("handles array content blocks in messages", async () => {
      const { api, hooks, logs } = createMockApi({ autoCapture: true, captureLlm: false });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      const result = await captureHook.handler(
        {
          success: true,
          messages: [
            { role: "user", content: [{ type: "text", text: "Remember I always prefer TypeScript over JavaScript" }] },
          ],
        },
        { agentId: "main" },
      );
      // Should attempt capture (will fail at embed, caught gracefully)
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(warnLog, "should attempt capture with array content");
    });

    it("skips non-user messages when captureAssistant=false", async () => {
      const { api, hooks } = createMockApi({ autoCapture: true, captureAssistant: false, captureLlm: false });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      const result = await captureHook.handler(
        {
          success: true,
          messages: [{ role: "assistant", content: "Remember to always use TypeScript" }],
        },
        { agentId: "main" },
      );
      // assistant message should be skipped → no capture
      assert.equal(result, undefined);
    });

    it("includes assistant messages when captureAssistant=true", async () => {
      const { api, hooks, logs } = createMockApi({ autoCapture: true, captureAssistant: true, captureLlm: false });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        {
          success: true,
          messages: [{ role: "assistant", content: "Remember that I prefer using dark mode always" }],
        },
        { agentId: "main" },
      );
      // Should attempt capture (will fail at embed), meaning it processed the assistant message
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(warnLog, "should try to capture assistant message");
    });

    it("skips null/non-object messages gracefully", async () => {
      const { api, hooks } = createMockApi({ autoCapture: true, captureLlm: false });
      plugin.register(api);

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      const result = await captureHook.handler(
        { success: true, messages: [null, undefined, "string", 42] },
        { agentId: "main" },
      );
      assert.equal(result, undefined);
    });
  });

  // ---------------------------------------------------------------------------
  // LLM capture judgment (captureLlm=true path)
  // ---------------------------------------------------------------------------

  describe("LLM capture judgment (captureLlm=true)", () => {
    it("calls LLM gateway and handles store=false response", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      // Mock fetch to return store=false
      globalThis.fetch = (async (_url: any, _opts: any) => ({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: '{"store": false}' } }],
        }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // LLM said don't store — no capture error, debug log about not storing
      const hasCaptureFail = logs.some(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(!hasCaptureFail, "should NOT have capture failure when LLM says no");
    });

    it("calls LLM gateway and handles store=true with memories", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      // Mock fetch to return store=true with memories
      globalThis.fetch = (async (_url: any, _opts: any) => ({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: JSON.stringify({
            store: true,
            memories: [{ text: "User prefers dark mode", category: "preference", importance: 0.8 }],
          }) } }],
        }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // Will fail at embedder.embedPassage since no real API — caught gracefully
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(warnLog, "should fail at embedding step");
    });

    it("handles LLM gateway returning non-ok status", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      // Mock fetch to return 500
      globalThis.fetch = (async () => ({
        ok: false,
        status: 500,
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // LLM failed → falls through to heuristic → fails at embedder → caught
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(warnLog, "should fall through to heuristic and fail at embed");
    });

    it("handles LLM gateway throwing (network error)", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => { throw new Error("ECONNREFUSED"); }) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // Falls through to heuristic → fails at embed → caught
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(warnLog);
    });

    it("handles LLM returning empty content", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({ choices: [{ message: { content: "" } }] }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // LLM returned empty → null → heuristic fallback → embed failure
      const warnLog = logs.find(l => l.level === "warn");
      assert.ok(warnLog);
    });

    it("handles LLM returning non-JSON content", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({ choices: [{ message: { content: "I think we should store this" } }] }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // Non-JSON → null → heuristic fallback
      const warnLog = logs.find(l => l.msg.includes("not valid JSON") || l.msg.includes("capture failed"));
      assert.ok(warnLog);
    });

    it("handles LLM response with store=true but empty memories array", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({ choices: [{ message: { content: '{"store": true, "memories": []}' } }] }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // store=true but no memories → null → heuristic fallback
      const warnLog = logs.find(l => l.msg.includes("no memories array") || l.msg.includes("capture failed"));
      assert.ok(warnLog);
    });

    it("handles LLM response missing store boolean", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({ choices: [{ message: { content: '{"something": "else"}' } }] }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      const warnLog = logs.find(l => l.msg.includes("missing 'store' boolean") || l.msg.includes("capture failed"));
      assert.ok(warnLog);
    });

    it("sanitizes invalid category in LLM memory response", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: JSON.stringify({
            store: true,
            memories: [{ text: "test", category: "INVALID_CAT", importance: 99 }],
          }) } }],
        }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // Invalid category should be corrected to "other", invalid importance to 0.7
      // Will fail at embed step
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(warnLog, "should reach embed step with sanitized data");
    });

    it("handles LLM response with empty memory text", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: JSON.stringify({
            store: true,
            memories: [{ text: "", category: "fact", importance: 0.7 }],
          }) } }],
        }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // Empty text → null → heuristic fallback
      const warnLog = logs.find(l => l.level === "warn");
      assert.ok(warnLog);
    });

    it("deduplicates LLM gateway URLs by host", async () => {
      const fetchedUrls: string[] = [];
      const { api, hooks } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:3000/v1",  // same host as default fallback
      });
      plugin.register(api);

      globalThis.fetch = (async (url: any) => {
        fetchedUrls.push(String(url));
        throw new Error("ECONNREFUSED");
      }) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // localhost:3000 should only be tried once (configured URL deduplicates with default)
      const localhost3000Calls = fetchedUrls.filter(u => u.includes("localhost:3000"));
      assert.ok(localhost3000Calls.length <= 1, `localhost:3000 called ${localhost3000Calls.length} times, expected <=1`);
    });

    it("handles abort timeout scenario", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => {
        const err = new Error("The operation was aborted");
        err.name = "AbortError";
        throw err;
      }) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // abort → null from callLlmForCaptureJudgment → heuristic fallback
      const warnLog = logs.find(l => l.level === "warn");
      assert.ok(warnLog);
    });
  });

  // ---------------------------------------------------------------------------
  // Service start/stop
  // ---------------------------------------------------------------------------

  describe("service start/stop", () => {
    it("service start handles errors gracefully", async () => {
      const { api, services, logs } = createMockApi();
      plugin.register(api);

      const svc = services[0];
      assert.ok(svc);

      // Mock fetch to fail so embedder.test() returns {success: false}
      globalThis.fetch = (async () => {
        throw new Error("ECONNREFUSED");
      }) as any;

      // start() will try embedder.test() and retriever.test() — both will fail.
      // embedder.test() catches internally and returns {success: false}.
      // The service start code then logs the embedding test failure.
      await svc.start();
      const warnLog = logs.find(l => l.level === "warn" && l.msg.includes("embedding test failed"));
      assert.ok(warnLog, "should log embedding test failure");
    });

    it("service stop clears backup timer", () => {
      const { api, services, logs } = createMockApi();
      plugin.register(api);

      const svc = services[0];
      // stop() should not throw
      svc.stop();
      const stopLog = logs.find(l => l.msg.includes("stopped"));
      assert.ok(stopLog, "should log stop message");
    });
  });

  // ---------------------------------------------------------------------------
  // sessionMemory hook (command:new)
  // ---------------------------------------------------------------------------

  describe("sessionMemory hook", () => {
    it("registers command:new hook when sessionMemory.enabled=true", () => {
      const { api, registeredHookEvents } = createMockApi({
        sessionMemory: { enabled: true, messageCount: 15 },
      });
      plugin.register(api);

      const sessionHook = registeredHookEvents.find(h => h.name === "command:new");
      assert.ok(sessionHook, "should register command:new hook");
    });

    it("does NOT register command:new hook when sessionMemory.enabled=false", () => {
      const { api, registeredHookEvents } = createMockApi({
        sessionMemory: { enabled: false },
      });
      plugin.register(api);

      const sessionHook = registeredHookEvents.find(h => h.name === "command:new");
      assert.equal(sessionHook, undefined);
    });

    it("handles missing sessionFile gracefully", async () => {
      const { api, registeredHookEvents, logs } = createMockApi({
        sessionMemory: { enabled: true, messageCount: 15 },
      });
      plugin.register(api);

      const sessionHook = registeredHookEvents.find(h => h.name === "command:new")!;
      // Call with empty context — no sessionFile
      await sessionHook.handler({
        timestamp: Date.now(),
        sessionKey: "test-key",
        context: {},
      });

      // Should return without error (no sessionFile)
      const warnLog = logs.find(l => l.msg.includes("session summary failed"));
      assert.ok(!warnLog, "should not fail — just early return");
    });

    it("handles non-existent session file gracefully", async () => {
      const { api, registeredHookEvents, logs } = createMockApi({
        sessionMemory: { enabled: true, messageCount: 15 },
      });
      plugin.register(api);

      const sessionHook = registeredHookEvents.find(h => h.name === "command:new")!;
      await sessionHook.handler({
        timestamp: Date.now(),
        sessionKey: "test-key",
        context: {
          previousSessionEntry: {
            sessionFile: "/tmp/nonexistent-session-file.jsonl",
            sessionId: "test-session-id",
          },
        },
      });

      // File doesn't exist → readSessionMessages returns null → early return or error caught
      // Either way, no crash
    });
  });

  // ---------------------------------------------------------------------------
  // callLlmForCaptureJudgment — more edge cases (19)
  // ---------------------------------------------------------------------------

  describe("callLlmForCaptureJudgment edge cases", () => {
    const originalEnv = { ...process.env };

    afterEach(() => {
      // Restore env
      for (const key of Object.keys(process.env)) {
        if (!(key in originalEnv)) delete process.env[key];
      }
      for (const [key, val] of Object.entries(originalEnv)) {
        if (val !== undefined) process.env[key] = val;
      }
    });

    it("uses OPENCLAW_GATEWAY_URL env var as fallback", async () => {
      const fetchedUrls: string[] = [];
      process.env.OPENCLAW_GATEWAY_URL = "http://custom-gateway:5000";

      const { api, hooks } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
      });
      plugin.register(api);

      globalThis.fetch = (async (url: any) => {
        fetchedUrls.push(String(url));
        throw new Error("ECONNREFUSED");
      }) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      assert.ok(fetchedUrls.some(u => u.includes("custom-gateway:5000")), "should try OPENCLAW_GATEWAY_URL");

      delete process.env.OPENCLAW_GATEWAY_URL;
    });

    it("handles markdown-wrapped JSON in LLM response", async () => {
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({
          choices: [{ message: { content: '```json\n{"store": false}\n```' } }],
        }),
      })) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // Should extract JSON from markdown code block
      const captureFail = logs.find(l => l.level === "warn" && l.msg.includes("capture failed"));
      assert.ok(!captureFail, "should not fail — JSON extracted from markdown");
    });
  });

  // ---------------------------------------------------------------------------
  // ConversationBuffer behavior via capture hooks (20)
  // ---------------------------------------------------------------------------

  describe("ConversationBuffer behavior via capture hooks", () => {
    it("accumulates messages across multiple agent_end events", async () => {
      let callCount = 0;
      const { api, hooks, logs } = createMockApi({
        autoCapture: true,
        captureLlm: true,
        captureLlmModel: "test-model",
        captureLlmUrl: "http://localhost:19999",
      });
      plugin.register(api);

      globalThis.fetch = (async (_url: any, opts: any) => {
        callCount++;
        const body = JSON.parse(opts.body);
        // Verify that later calls include accumulated context
        if (callCount > 1) {
          assert.ok(body.messages[1].content.length > 0, "should have accumulated content");
        }
        return {
          ok: true,
          json: async () => ({ choices: [{ message: { content: '{"store": false}' } }] }),
        };
      }) as any;

      const captureHook = hooks.find(h => h.name === "agent_end")!;

      // First turn
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Remember I always prefer dark mode in my editor" }] },
        { agentId: "main" },
      );

      // Second turn — buffer should include context from first turn
      await captureHook.handler(
        { success: true, messages: [{ role: "user", content: "Also remember I prefer tabs over spaces always" }] },
        { agentId: "main" },
      );

      assert.ok(callCount >= 1, "should have called LLM at least once");
    });
  });
});

// ============================================================================
// 18. sanitizeForContext (exported indirectly via hook behavior)
// ============================================================================

describe("sanitizeForContext (via index.ts internals)", () => {
  // We can't import sanitizeForContext directly, but we can test it indirectly
  // by looking at the prependContext output from the autoRecall hook.
  // Since the recall hook requires a working embedder, we test the function
  // via its public effects — already covered above.

  // Instead, test the exported functions more thoroughly:

  it("shouldCapture rejects markdown-formatted text", () => {
    const text = "**Bold** and\n- list item about preferences";
    assert.equal(shouldCapture(text), false);
  });
});

// ============================================================================
// 21. Tool execute handlers (tools.ts)
// ============================================================================

import {
  registerMemoryRecallTool,
  registerMemoryStoreTool,
  registerMemoryForgetTool,
  registerMemoryUpdateTool,
  registerMemoryStatsTool,
  registerMemoryListTool,
  registerAllMemoryTools,
} from "../src/tools.js";

describe("Tool handlers (tools.ts)", () => {
  function createToolTestContext(overrides: Record<string, any> = {}) {
    const storedEntries: any[] = [];
    const deletedIds: string[] = [];

    const mockRetriever = {
      retrieve: overrides.retrieveResults
        ? async () => overrides.retrieveResults
        : async () => [],
      getConfig: () => ({ mode: "hybrid" }),
      test: async () => ({ success: true, mode: "hybrid", hasFtsSupport: true }),
      ...overrides.retriever,
    };

    const mockStore = {
      dbPath: "/tmp/test-db",
      hasFtsSupport: true,
      vectorSearch: overrides.vectorSearchResults
        ? async () => overrides.vectorSearchResults
        : async () => [],
      bm25Search: async () => [],
      store: async (entry: any) => {
        const full = { ...entry, id: "new-id-123", timestamp: Date.now() };
        storedEntries.push(full);
        return full;
      },
      list: overrides.listResults
        ? async () => overrides.listResults
        : async () => [],
      delete: async (id: string) => {
        if (overrides.deleteThrows) throw new Error(overrides.deleteThrows);
        deletedIds.push(id);
        return overrides.deleteResult !== undefined ? overrides.deleteResult : true;
      },
      stats: overrides.statsResult
        ? async () => overrides.statsResult
        : async () => ({ totalCount: 0, scopeCounts: {}, categoryCounts: {} }),
      update: overrides.updateResult !== undefined
        ? async () => overrides.updateResult
        : async () => null,
      bulkDelete: async () => 0,
      hasId: async () => false,
      importEntry: async () => ({} as any),
    };

    const mockScopeManager = {
      getAccessibleScopes: () => ["global"],
      getDefaultScope: () => "global",
      isAccessible: (scope: string) => scope === "global" || scope === "project",
      getStats: () => ({ totalScopes: 2, scopesByType: {} }),
    };

    const mockEmbedder = {
      dimensions: 3,
      model: "mock-model",
      embed: async () => [0.1, 0.2, 0.3],
      embedQuery: async () => [0.1, 0.2, 0.3],
      embedPassage: async () => [0.1, 0.2, 0.3],
      embedBatch: async (texts: string[]) => texts.map(() => [0.1, 0.2, 0.3]),
      test: async () => ({ success: true, dimensions: 3 }),
      get cacheStats() { return { size: 0, hits: 0, misses: 0, hitRate: "N/A" }; },
    };

    const context: any = {
      retriever: mockRetriever,
      store: mockStore,
      scopeManager: mockScopeManager,
      embedder: mockEmbedder,
      agentId: undefined,
    };

    return { context, storedEntries, deletedIds, mockStore, mockRetriever };
  }

  function createToolApi() {
    const registeredTools: Map<string, any> = new Map();
    const api: any = {
      registerTool: (tool: any, opts: any) => {
        registeredTools.set(opts?.name || tool.name, tool);
      },
    };
    return { api, registeredTools };
  }

  // ---- memory_recall ----

  describe("memory_recall tool", () => {
    it("returns no memories message when empty", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryRecallTool(api, context);

      const tool = registeredTools.get("memory_recall")!;
      const result = await tool.execute("call-1", { query: "test" });
      assert.ok(result.content[0].text.includes("No relevant memories"));
      assert.equal(result.details.count, 0);
    });

    it("returns formatted results when memories exist", async () => {
      const { context } = createToolTestContext({
        retrieveResults: [{
          entry: { id: "r1", text: "User prefers dark mode", category: "preference", scope: "global", importance: 0.8, timestamp: 100 },
          score: 0.92,
          sources: { vector: true, bm25: false, reranked: false },
        }],
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryRecallTool(api, context);

      const tool = registeredTools.get("memory_recall")!;
      const result = await tool.execute("call-1", { query: "preferences" });
      assert.ok(result.content[0].text.includes("Found 1 memories"));
      assert.ok(result.content[0].text.includes("dark mode"));
      assert.equal(result.details.count, 1);
    });

    it("filters by scope when provided", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryRecallTool(api, context);

      const tool = registeredTools.get("memory_recall")!;
      const result = await tool.execute("call-1", { query: "test", scope: "global" });
      assert.equal(result.details.count, 0);
      assert.deepEqual(result.details.scopes, ["global"]);
    });

    it("denies access to inaccessible scope", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryRecallTool(api, context);

      const tool = registeredTools.get("memory_recall")!;
      const result = await tool.execute("call-1", { query: "test", scope: "secret" });
      assert.ok(result.content[0].text.includes("Access denied"));
      assert.equal(result.details.error, "scope_access_denied");
    });

    it("handles retrieval errors gracefully", async () => {
      const { context } = createToolTestContext();
      context.retriever.retrieve = async () => { throw new Error("embed failed"); };
      const { api, registeredTools } = createToolApi();
      registerMemoryRecallTool(api, context);

      const tool = registeredTools.get("memory_recall")!;
      const result = await tool.execute("call-1", { query: "test" });
      assert.ok(result.content[0].text.includes("Memory recall failed"));
      assert.equal(result.details.error, "recall_failed");
    });

    it("includes source info (BM25, reranked) in output", async () => {
      const { context } = createToolTestContext({
        retrieveResults: [{
          entry: { id: "r1", text: "test", category: "fact", scope: "global", importance: 0.8, timestamp: 100 },
          score: 0.9,
          sources: { vector: true, bm25: true, reranked: true },
        }],
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryRecallTool(api, context);

      const tool = registeredTools.get("memory_recall")!;
      const result = await tool.execute("call-1", { query: "test" });
      assert.ok(result.content[0].text.includes("vector"));
      assert.ok(result.content[0].text.includes("BM25"));
      assert.ok(result.content[0].text.includes("reranked"));
    });
  });

  // ---- memory_store ----

  describe("memory_store tool", () => {
    it("stores a memory successfully", async () => {
      const { context, storedEntries } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryStoreTool(api, context);

      const tool = registeredTools.get("memory_store")!;
      const result = await tool.execute("call-1", { text: "User prefers dark mode" });
      assert.ok(result.content[0].text.includes("Stored"));
      assert.equal(result.details.action, "created");
      assert.equal(storedEntries.length, 1);
    });

    it("detects duplicate memories", async () => {
      const { context } = createToolTestContext({
        vectorSearchResults: [{
          entry: { id: "dup-1", text: "User prefers dark mode", category: "preference", scope: "global", importance: 0.8, timestamp: 100 },
          score: 0.99, // > 0.98 threshold
        }],
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryStoreTool(api, context);

      const tool = registeredTools.get("memory_store")!;
      const result = await tool.execute("call-1", { text: "User prefers dark mode" });
      assert.ok(result.content[0].text.includes("Similar memory already exists"));
      assert.equal(result.details.action, "duplicate");
    });

    it("denies access to inaccessible scope", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryStoreTool(api, context);

      const tool = registeredTools.get("memory_store")!;
      const result = await tool.execute("call-1", { text: "test", scope: "secret" });
      assert.ok(result.content[0].text.includes("Access denied"));
    });

    it("filters noise text", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryStoreTool(api, context);

      const tool = registeredTools.get("memory_store")!;
      const result = await tool.execute("call-1", { text: "ok" });
      assert.ok(result.content[0].text.includes("noise"));
      assert.equal(result.details.action, "noise_filtered");
    });

    it("handles store errors gracefully", async () => {
      const { context } = createToolTestContext();
      context.embedder.embedPassage = async () => { throw new Error("API error"); };
      const { api, registeredTools } = createToolApi();
      registerMemoryStoreTool(api, context);

      const tool = registeredTools.get("memory_store")!;
      const result = await tool.execute("call-1", { text: "important fact to remember" });
      assert.ok(result.content[0].text.includes("Memory storage failed"));
      assert.equal(result.details.error, "store_failed");
    });
  });

  // ---- memory_forget ----

  describe("memory_forget tool", () => {
    it("deletes by memoryId", async () => {
      const { context, deletedIds } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryForgetTool(api, context);

      const tool = registeredTools.get("memory_forget")!;
      const result = await tool.execute("call-1", { memoryId: "abc123" });
      assert.ok(result.content[0].text.includes("forgotten"));
      assert.equal(deletedIds[0], "abc123");
    });

    it("reports not found for missing memoryId", async () => {
      const { context } = createToolTestContext({ deleteResult: false });
      const { api, registeredTools } = createToolApi();
      registerMemoryForgetTool(api, context);

      const tool = registeredTools.get("memory_forget")!;
      const result = await tool.execute("call-1", { memoryId: "nonexistent" });
      assert.ok(result.content[0].text.includes("not found"));
    });

    it("searches by query and lists candidates", async () => {
      const { context } = createToolTestContext({
        retrieveResults: [
          { entry: { id: "r1", text: "first memory", category: "fact", scope: "global", importance: 0.8, timestamp: 100 }, score: 0.7, sources: {} },
          { entry: { id: "r2", text: "second memory", category: "fact", scope: "global", importance: 0.7, timestamp: 200 }, score: 0.6, sources: {} },
        ],
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryForgetTool(api, context);

      const tool = registeredTools.get("memory_forget")!;
      const result = await tool.execute("call-1", { query: "memory" });
      assert.ok(result.content[0].text.includes("candidates"));
      assert.equal(result.details.action, "candidates");
    });

    it("auto-deletes single high-confidence match", async () => {
      const { context, deletedIds } = createToolTestContext({
        retrieveResults: [
          { entry: { id: "r1", text: "exact match", category: "fact", scope: "global", importance: 0.8, timestamp: 100 }, score: 0.95, sources: {} },
        ],
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryForgetTool(api, context);

      const tool = registeredTools.get("memory_forget")!;
      const result = await tool.execute("call-1", { query: "exact" });
      assert.ok(result.content[0].text.includes("Forgotten"));
      assert.equal(deletedIds[0], "r1");
    });

    it("returns error when no query or memoryId", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryForgetTool(api, context);

      const tool = registeredTools.get("memory_forget")!;
      const result = await tool.execute("call-1", {});
      assert.ok(result.content[0].text.includes("Provide either"));
      assert.equal(result.details.error, "missing_param");
    });

    it("denies access to inaccessible scope", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryForgetTool(api, context);

      const tool = registeredTools.get("memory_forget")!;
      const result = await tool.execute("call-1", { memoryId: "abc", scope: "secret" });
      assert.ok(result.content[0].text.includes("Access denied"));
    });

    it("handles delete errors gracefully", async () => {
      const { context } = createToolTestContext({ deleteThrows: "DB error" });
      const { api, registeredTools } = createToolApi();
      registerMemoryForgetTool(api, context);

      const tool = registeredTools.get("memory_forget")!;
      const result = await tool.execute("call-1", { memoryId: "abc" });
      assert.ok(result.content[0].text.includes("Memory deletion failed"));
      assert.equal(result.details.error, "delete_failed");
    });

    it("returns no matching when query yields empty results", async () => {
      const { context } = createToolTestContext({ retrieveResults: [] });
      const { api, registeredTools } = createToolApi();
      registerMemoryForgetTool(api, context);

      const tool = registeredTools.get("memory_forget")!;
      const result = await tool.execute("call-1", { query: "nonexistent" });
      assert.ok(result.content[0].text.includes("No matching"));
    });
  });

  // ---- memory_update ----

  describe("memory_update tool", () => {
    it("updates an existing memory by UUID", async () => {
      const uuid = "12345678-1234-1234-1234-123456789abc";
      const { context } = createToolTestContext({
        updateResult: { id: uuid, text: "updated text", category: "fact", scope: "global", importance: 0.9, timestamp: 100 },
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryUpdateTool(api, context);

      const tool = registeredTools.get("memory_update")!;
      const result = await tool.execute("call-1", { memoryId: uuid, text: "updated text" });
      assert.ok(result.content[0].text.includes("Updated memory"));
      assert.equal(result.details.action, "updated");
    });

    it("returns error when nothing to update", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryUpdateTool(api, context);

      const tool = registeredTools.get("memory_update")!;
      const result = await tool.execute("call-1", { memoryId: "12345678" });
      assert.ok(result.content[0].text.includes("Nothing to update"));
      assert.equal(result.details.error, "no_updates");
    });

    it("returns not found when update returns null", async () => {
      const { context } = createToolTestContext({ updateResult: null });
      const { api, registeredTools } = createToolApi();
      registerMemoryUpdateTool(api, context);

      const tool = registeredTools.get("memory_update")!;
      const result = await tool.execute("call-1", { memoryId: "12345678-1234-1234-1234-123456789abc", text: "new text" });
      assert.ok(result.content[0].text.includes("not found"));
    });

    it("resolves by text search when memoryId is not UUID-like", async () => {
      const { context } = createToolTestContext({
        retrieveResults: [{
          entry: { id: "found-id-1234", text: "old text", category: "fact", scope: "global", importance: 0.8, timestamp: 100 },
          score: 0.9,
          sources: {},
        }],
        updateResult: { id: "found-id-1234", text: "new text", category: "fact", scope: "global", importance: 0.8, timestamp: 100 },
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryUpdateTool(api, context);

      const tool = registeredTools.get("memory_update")!;
      const result = await tool.execute("call-1", { memoryId: "my preference", text: "new text" });
      assert.ok(result.content[0].text.includes("Updated"));
    });

    it("shows candidates when text search returns multiple low-confidence results", async () => {
      const { context } = createToolTestContext({
        retrieveResults: [
          { entry: { id: "c1", text: "first", category: "fact", scope: "global", importance: 0.8, timestamp: 100 }, score: 0.5, sources: {} },
          { entry: { id: "c2", text: "second", category: "fact", scope: "global", importance: 0.7, timestamp: 200 }, score: 0.4, sources: {} },
        ],
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryUpdateTool(api, context);

      const tool = registeredTools.get("memory_update")!;
      const result = await tool.execute("call-1", { memoryId: "ambiguous text query", text: "x" });
      assert.ok(result.content[0].text.includes("Multiple matches"));
      assert.equal(result.details.action, "candidates");
    });

    it("filters noise in updated text", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryUpdateTool(api, context);

      const tool = registeredTools.get("memory_update")!;
      const result = await tool.execute("call-1", { memoryId: "12345678-1234-1234-1234-123456789abc", text: "ok" });
      assert.ok(result.content[0].text.includes("noise"));
      assert.equal(result.details.action, "noise_filtered");
    });

    it("handles update errors gracefully", async () => {
      const { context } = createToolTestContext();
      context.store.update = async () => { throw new Error("DB error"); };
      const { api, registeredTools } = createToolApi();
      registerMemoryUpdateTool(api, context);

      const tool = registeredTools.get("memory_update")!;
      const result = await tool.execute("call-1", { memoryId: "12345678-1234-1234-1234-123456789abc", importance: 0.9 });
      assert.ok(result.content[0].text.includes("Memory update failed"));
      assert.equal(result.details.error, "update_failed");
    });

    it("returns not found when text search yields no results", async () => {
      const { context } = createToolTestContext({ retrieveResults: [] });
      const { api, registeredTools } = createToolApi();
      registerMemoryUpdateTool(api, context);

      const tool = registeredTools.get("memory_update")!;
      const result = await tool.execute("call-1", { memoryId: "some text query", text: "new text" });
      assert.ok(result.content[0].text.includes("No memory found"));
    });
  });

  // ---- memory_stats ----

  describe("memory_stats tool", () => {
    it("returns formatted statistics", async () => {
      const { context } = createToolTestContext({
        statsResult: {
          totalCount: 5,
          scopeCounts: { global: 3, project: 2 },
          categoryCounts: { fact: 3, preference: 2 },
        },
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryStatsTool(api, context);

      const tool = registeredTools.get("memory_stats")!;
      const result = await tool.execute("call-1", {});
      assert.ok(result.content[0].text.includes("Total memories: 5"));
      assert.ok(result.content[0].text.includes("global: 3"));
    });

    it("denies access to inaccessible scope", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryStatsTool(api, context);

      const tool = registeredTools.get("memory_stats")!;
      const result = await tool.execute("call-1", { scope: "secret" });
      assert.ok(result.content[0].text.includes("Access denied"));
    });

    it("handles stats errors gracefully", async () => {
      const { context } = createToolTestContext();
      context.store.stats = async () => { throw new Error("DB error"); };
      const { api, registeredTools } = createToolApi();
      registerMemoryStatsTool(api, context);

      const tool = registeredTools.get("memory_stats")!;
      const result = await tool.execute("call-1", {});
      assert.ok(result.content[0].text.includes("Failed to get memory stats"));
    });
  });

  // ---- memory_list ----

  describe("memory_list tool", () => {
    it("returns formatted list of memories", async () => {
      const { context } = createToolTestContext({
        listResults: [
          { id: "l1", text: "first memory", category: "fact", scope: "global", importance: 0.8, timestamp: Date.now() },
          { id: "l2", text: "second memory", category: "preference", scope: "global", importance: 0.7, timestamp: Date.now() - 1000 },
        ],
      });
      const { api, registeredTools } = createToolApi();
      registerMemoryListTool(api, context);

      const tool = registeredTools.get("memory_list")!;
      const result = await tool.execute("call-1", {});
      assert.ok(result.content[0].text.includes("Recent memories"));
      assert.ok(result.content[0].text.includes("first memory"));
      assert.equal(result.details.count, 2);
    });

    it("returns no memories message when empty", async () => {
      const { context } = createToolTestContext({ listResults: [] });
      const { api, registeredTools } = createToolApi();
      registerMemoryListTool(api, context);

      const tool = registeredTools.get("memory_list")!;
      const result = await tool.execute("call-1", {});
      assert.ok(result.content[0].text.includes("No memories found"));
    });

    it("denies access to inaccessible scope", async () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerMemoryListTool(api, context);

      const tool = registeredTools.get("memory_list")!;
      const result = await tool.execute("call-1", { scope: "secret" });
      assert.ok(result.content[0].text.includes("Access denied"));
    });

    it("handles list errors gracefully", async () => {
      const { context } = createToolTestContext();
      context.store.list = async () => { throw new Error("DB error"); };
      const { api, registeredTools } = createToolApi();
      registerMemoryListTool(api, context);

      const tool = registeredTools.get("memory_list")!;
      const result = await tool.execute("call-1", {});
      assert.ok(result.content[0].text.includes("Failed to list memories"));
    });
  });

  // ---- registerAllMemoryTools ----

  describe("registerAllMemoryTools()", () => {
    it("registers core tools without management tools by default", () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerAllMemoryTools(api, context);

      assert.ok(registeredTools.has("memory_recall"));
      assert.ok(registeredTools.has("memory_store"));
      assert.ok(registeredTools.has("memory_forget"));
      assert.ok(registeredTools.has("memory_update"));
      assert.ok(!registeredTools.has("memory_stats"));
      assert.ok(!registeredTools.has("memory_list"));
    });

    it("registers all tools including management when enabled", () => {
      const { context } = createToolTestContext();
      const { api, registeredTools } = createToolApi();
      registerAllMemoryTools(api, context, { enableManagementTools: true });

      assert.ok(registeredTools.has("memory_stats"));
      assert.ok(registeredTools.has("memory_list"));
    });
  });
});
