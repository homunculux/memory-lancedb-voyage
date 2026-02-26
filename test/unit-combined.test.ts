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
import { VoyageEmbedder } from "../src/embedder.js";
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

import { VoyageEmbedder, EmbeddingCache } from "../src/embedder.js";

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

import { OpenAIEmbedder } from "../src/embedder-openai.js";

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

import { JinaEmbedder } from "../src/embedder-jina.js";

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
// Phase 2: Extended coverage tests for store, tools, cli, migrate, index
// ============================================================================

// Dynamic imports for modules not yet imported at the top of unit.test.ts
const storeModule = await import("../src/store.js");
const { MemoryStore } = storeModule;

const toolsModule = await import("../src/tools.js");
const {
  registerMemoryRecallTool,
  registerMemoryStoreTool,
  registerMemoryForgetTool,
  registerMemoryUpdateTool,
  registerMemoryStatsTool,
  registerMemoryListTool,
  registerAllMemoryTools,
} = toolsModule;

const cliModule = await import("../cli.js");
const { registerMemoryCLI } = cliModule;

const migrateModule = await import("../src/migrate.js");
const { MemoryMigrator, createMigrator, checkForLegacyData } = migrateModule;

const indexModule = await import("../index.js");
const memoryLanceDBVoyagePlugin = indexModule.default;

// ---------------------------------------------------------------------------
// Helpers: mock table builder for MemoryStore
// ---------------------------------------------------------------------------

function createMockTable(overrides: Record<string, any> = {}) {
  const addedRecords: any[] = [];
  const deletedWheres: string[] = [];

  const defaultTable: any = {
    add: async (records: any[]) => { addedRecords.push(...records); },
    delete: async (where: string) => { deletedWheres.push(where); },
    query: () => {
      let _where: string | undefined;
      let _limit: number | undefined;
      const chain: any = {
        select: () => chain,
        where: (w: string) => { _where = w; return chain; },
        limit: (n: number) => { _limit = n; return chain; },
        toArray: async () => [],
      };
      return chain;
    },
    vectorSearch: (vector: number[]) => {
      const chain: any = {
        limit: () => chain,
        where: () => chain,
        distanceType: () => chain,
        toArray: async () => [],
      };
      return chain;
    },
    search: (query: string, type: string) => {
      const chain: any = {
        limit: () => chain,
        where: () => chain,
        toArray: async () => [],
      };
      return chain;
    },
    listIndices: async () => [{ indexType: "FTS" }],
    createIndex: async () => {},
    countRows: async () => 0,
    ...overrides,
  };

  return { table: defaultTable, addedRecords, deletedWheres };
}

function createInitializedStore(mockTable: any): InstanceType<typeof MemoryStore> {
  const store = new MemoryStore({ dbPath: "/tmp/test-store", vectorDim: 4 });
  (store as any).table = mockTable;
  (store as any).db = {};
  (store as any).ftsIndexCreated = true;
  return store;
}

// ---------------------------------------------------------------------------
// FILE 1: index.ts — plugin object shape and register()
// ---------------------------------------------------------------------------

describe("index.ts — plugin export shape", () => {
  it("should export an object with correct id", () => {
    assert.equal(memoryLanceDBVoyagePlugin.id, "memory-lancedb-voyage");
  });

  it("should export an object with correct name", () => {
    assert.equal(memoryLanceDBVoyagePlugin.name, "Memory (LanceDB + Voyage AI)");
  });

  it("should have kind 'memory'", () => {
    assert.equal(memoryLanceDBVoyagePlugin.kind, "memory");
  });

  it("should have a non-empty description", () => {
    assert.ok(memoryLanceDBVoyagePlugin.description.length > 10);
  });

  it("should have a configSchema with .parse() method", () => {
    assert.ok(memoryLanceDBVoyagePlugin.configSchema);
    assert.equal(typeof memoryLanceDBVoyagePlugin.configSchema.parse, "function");
  });

  it("should have a register function", () => {
    assert.equal(typeof memoryLanceDBVoyagePlugin.register, "function");
  });
});

describe("index.ts — configSchema validation via plugin", () => {
  it("should validate a minimal valid config", () => {
    const config = memoryLanceDBVoyagePlugin.configSchema.parse({
      embedding: { apiKey: "test-key-123" },
    });
    assert.equal(config.embedding.apiKey, "test-key-123");
  });

  it("should reject config missing required fields", () => {
    assert.throws(() => {
      memoryLanceDBVoyagePlugin.configSchema.parse({});
    });
  });

  it("should accept config with optional fields set", () => {
    const config = memoryLanceDBVoyagePlugin.configSchema.parse({
      embedding: { apiKey: "test-key" },
      autoRecall: true,
      autoCapture: true,
      enableManagementTools: true,
    });
    assert.equal(config.autoRecall, true);
    assert.equal(config.autoCapture, true);
    assert.equal(config.enableManagementTools, true);
  });

  it("should apply default values for optional fields", () => {
    const config = memoryLanceDBVoyagePlugin.configSchema.parse({
      embedding: { apiKey: "test-key" },
    });
    assert.equal(typeof config.autoRecall, "boolean");
    assert.equal(typeof config.autoCapture, "boolean");
  });
});

describe("index.ts — register() with mocked API", () => {
  let registeredTools: any[];
  let registeredHooks: Map<string, Function>;
  let registeredServices: any[];
  let registeredClis: any[];
  let registeredEventHandlers: Map<string, Function>;
  let mockApi: any;

  function buildMockApi(configOverrides: Record<string, any> = {}) {
    registeredTools = [];
    registeredHooks = new Map();
    registeredServices = [];
    registeredClis = [];
    registeredEventHandlers = new Map();

    return {
      pluginConfig: {
        embedding: { apiKey: "test-voyage-key" },
        dbPath: "/tmp/vidya-test-register",
        autoRecall: false,
        autoCapture: false,
        sessionMemory: { enabled: false },
        ...configOverrides,
      },
      resolvePath: (p: string) => p,
      logger: {
        info: () => {},
        warn: () => {},
        debug: () => {},
      },
      registerTool: (toolDef: any, opts: any) => {
        registeredTools.push({ toolDef, opts });
      },
      registerHook: (hookName: string, handler: Function) => {
        registeredHooks.set(hookName, handler);
      },
      registerCli: (cliFactory: any, opts: any) => {
        registeredClis.push({ cliFactory, opts });
      },
      registerService: (svc: any) => {
        registeredServices.push(svc);
      },
      on: (event: string, handler: Function) => {
        registeredEventHandlers.set(event, handler);
      },
    };
  }

  it("should register core tools when called with valid config", () => {
    mockApi = buildMockApi();
    memoryLanceDBVoyagePlugin.register(mockApi);
    // 4 core tools: recall, store, forget, update
    assert.ok(registeredTools.length >= 4, `Expected at least 4 tools, got ${registeredTools.length}`);
  });

  it("should register a CLI", () => {
    mockApi = buildMockApi();
    memoryLanceDBVoyagePlugin.register(mockApi);
    assert.ok(registeredClis.length >= 1);
  });

  it("should register a service with start/stop", () => {
    mockApi = buildMockApi();
    memoryLanceDBVoyagePlugin.register(mockApi);
    assert.ok(registeredServices.length >= 1);
    assert.equal(typeof registeredServices[0].start, "function");
    assert.equal(typeof registeredServices[0].stop, "function");
  });

  it("should register before_agent_start when autoRecall is true", () => {
    mockApi = buildMockApi({ autoRecall: true });
    memoryLanceDBVoyagePlugin.register(mockApi);
    assert.ok(registeredEventHandlers.has("before_agent_start"));
  });

  it("should NOT register before_agent_start when autoRecall is false", () => {
    mockApi = buildMockApi({ autoRecall: false });
    memoryLanceDBVoyagePlugin.register(mockApi);
    assert.ok(!registeredEventHandlers.has("before_agent_start"));
  });

  it("should register agent_end when autoCapture is true", () => {
    mockApi = buildMockApi({ autoCapture: true });
    memoryLanceDBVoyagePlugin.register(mockApi);
    assert.ok(registeredEventHandlers.has("agent_end"));
  });

  it("should NOT register agent_end when autoCapture is false", () => {
    mockApi = buildMockApi({ autoCapture: false });
    memoryLanceDBVoyagePlugin.register(mockApi);
    assert.ok(!registeredEventHandlers.has("agent_end"));
  });

  it("should register command:new hook when sessionMemory is enabled", () => {
    mockApi = buildMockApi({ sessionMemory: { enabled: true } });
    memoryLanceDBVoyagePlugin.register(mockApi);
    assert.ok(registeredHooks.has("command:new"));
  });

  it("should NOT register command:new hook when sessionMemory is disabled", () => {
    mockApi = buildMockApi({ sessionMemory: { enabled: false } });
    memoryLanceDBVoyagePlugin.register(mockApi);
    assert.ok(!registeredHooks.has("command:new"));
  });

  it("should register 6 tools when enableManagementTools is true", () => {
    mockApi = buildMockApi({ enableManagementTools: true });
    memoryLanceDBVoyagePlugin.register(mockApi);
    // 4 core + 2 management (stats, list)
    assert.ok(registeredTools.length >= 6, `Expected at least 6 tools, got ${registeredTools.length}`);
  });
});

// ---------------------------------------------------------------------------
// FILE 2: src/store.ts — MemoryStore
// ---------------------------------------------------------------------------

describe("src/store.ts — MemoryStore constructor", () => {
  it("should construct with minimal config", () => {
    const store = new MemoryStore({ dbPath: "/tmp/t1", vectorDim: 1024 });
    assert.ok(store !== null);
  });

  it("should expose dbPath getter", () => {
    const store = new MemoryStore({ dbPath: "/tmp/t2", vectorDim: 512 });
    assert.equal(store.dbPath, "/tmp/t2");
  });

  it("should have hasFtsSupport false initially", () => {
    const store = new MemoryStore({ dbPath: "/tmp/t3", vectorDim: 4 });
    assert.equal(store.hasFtsSupport, false);
  });
});

describe("src/store.ts — store()", () => {
  it("should add a record with generated id and timestamp", async () => {
    const { table, addedRecords } = createMockTable();
    const store = createInitializedStore(table);

    const result = await store.store({
      text: "Hello world",
      vector: [0.1, 0.2, 0.3, 0.4],
      category: "fact",
      scope: "agent:test",
      importance: 0.8,
    });
    assert.ok(addedRecords.length >= 1);
    assert.ok(result.id.length > 0);
    assert.equal(result.text, "Hello world");
    assert.equal(result.category, "fact");
    assert.ok(result.timestamp > 0);
  });

  it("should default metadata to empty JSON object", async () => {
    const { table, addedRecords } = createMockTable();
    const store = createInitializedStore(table);

    await store.store({
      text: "test",
      vector: [0.1, 0.2, 0.3, 0.4],
      category: "other",
      scope: "global",
      importance: 0.5,
    });
    assert.equal(addedRecords[0].metadata, "{}");
  });
});

describe("src/store.ts — importEntry()", () => {
  it("should add an imported entry to the table", async () => {
    const { table, addedRecords } = createMockTable();
    const store = createInitializedStore(table);

    const result = await store.importEntry({
      id: "imported-1",
      text: "Imported content",
      vector: [0.1, 0.2, 0.3, 0.4],
      category: "fact",
      scope: "user:global",
      importance: 0.7,
      timestamp: 1700000000000,
    });
    assert.ok(addedRecords.length >= 1);
    assert.equal(result.id, "imported-1");
  });

  it("should throw if id is missing", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);

    await assert.rejects(
      () => store.importEntry({ id: "", text: "test", vector: [0.1, 0.2, 0.3, 0.4], category: "fact", scope: "global", importance: 0.7, timestamp: 0 }),
      /id/i,
    );
  });

  it("should throw on vector dimension mismatch", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);

    await assert.rejects(
      () => store.importEntry({ id: "bad-vec", text: "test", vector: [0.1, 0.2], category: "fact", scope: "global", importance: 0.7, timestamp: 0 }),
      /dimension/i,
    );
  });
});

describe("src/store.ts — hasId()", () => {
  it("should return true when ID exists", async () => {
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => [{ id: "existing" }],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    assert.equal(await store.hasId("existing"), true);
  });

  it("should return false when ID does not exist", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    assert.equal(await store.hasId("nonexistent"), false);
  });
});

describe("src/store.ts — vectorSearch()", () => {
  it("should return mapped results with scores", async () => {
    const fakeRows = [
      { id: "r1", text: "result 1", vector: [0.1, 0.2, 0.3, 0.4], category: "fact", scope: "agent:test", importance: 0.8, timestamp: 1700000000000, metadata: "{}", _distance: 0.05 },
      { id: "r2", text: "result 2", vector: [0.5, 0.6, 0.7, 0.8], category: "preference", scope: "agent:test", importance: 0.6, timestamp: 1700000001000, metadata: "{}", _distance: 0.2 },
    ];
    const { table } = createMockTable();
    table.vectorSearch = () => {
      const chain: any = {
        limit: () => chain,
        where: () => chain,
        distanceType: () => chain,
        toArray: async () => fakeRows,
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const results = await store.vectorSearch([0.1, 0.2, 0.3, 0.4], 5, 0.1);
    assert.equal(results.length, 2);
    assert.equal(results[0].entry.id, "r1");
    assert.ok(results[0].score > 0);
  });

  it("should filter results below minScore", async () => {
    const fakeRows = [
      { id: "r1", text: "result", vector: [0.1, 0.2, 0.3, 0.4], category: "fact", scope: "global", importance: 0.5, timestamp: 0, metadata: "{}", _distance: 100 },
    ];
    const { table } = createMockTable();
    table.vectorSearch = () => {
      const chain: any = {
        limit: () => chain,
        where: () => chain,
        distanceType: () => chain,
        toArray: async () => fakeRows,
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const results = await store.vectorSearch([0.1, 0.2, 0.3, 0.4], 5, 0.9);
    assert.equal(results.length, 0);
  });

  it("should apply scope filter when provided", async () => {
    let capturedWhere: string | undefined;
    const { table } = createMockTable();
    table.vectorSearch = () => {
      const chain: any = {
        limit: () => chain,
        where: (w: string) => { capturedWhere = w; return chain; },
        distanceType: () => chain,
        toArray: async () => [],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await store.vectorSearch([0.1, 0.2, 0.3, 0.4], 5, 0.3, ["agent:test"]);
    assert.ok(capturedWhere !== undefined);
    assert.ok(capturedWhere!.includes("agent:test"));
  });

  it("should return empty array when no results", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    const results = await store.vectorSearch([0, 0, 0, 0], 5, 0.3);
    assert.deepEqual(results, []);
  });
});

describe("src/store.ts — bm25Search()", () => {
  it("should return results for text query", async () => {
    const fakeRows = [
      { id: "b1", text: "bm25 result", vector: [0.1, 0.2, 0.3, 0.4], category: "fact", scope: "global", importance: 0.8, timestamp: 0, metadata: "{}", _score: 5.0 },
    ];
    const { table } = createMockTable();
    table.search = () => {
      const chain: any = {
        limit: () => chain,
        where: () => chain,
        toArray: async () => fakeRows,
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const results = await store.bm25Search("test query", 10);
    assert.equal(results.length, 1);
    assert.equal(results[0].entry.id, "b1");
    assert.ok(results[0].score > 0);
  });

  it("should return empty array when FTS not created", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    (store as any).ftsIndexCreated = false;
    const results = await store.bm25Search("test", 5);
    assert.deepEqual(results, []);
  });

  it("should apply scope filter", async () => {
    let capturedWhere: string | undefined;
    const { table } = createMockTable();
    table.search = () => {
      const chain: any = {
        limit: () => chain,
        where: (w: string) => { capturedWhere = w; return chain; },
        toArray: async () => [],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await store.bm25Search("query", 10, ["user:global"]);
    assert.ok(capturedWhere !== undefined);
    assert.ok(capturedWhere!.includes("user:global"));
  });

  it("should gracefully handle search errors and return empty", async () => {
    const { table } = createMockTable();
    table.search = () => { throw new Error("FTS broken"); };
    const store = createInitializedStore(table);
    // bm25Search catches errors and returns []
    const results = await store.bm25Search("test", 5);
    assert.deepEqual(results, []);
  });
});

describe("src/store.ts — delete()", () => {
  it("should delete by full UUID", async () => {
    const { table, deletedWheres } = createMockTable();
    const uuid = "550e8400-e29b-41d4-a716-446655440000";
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => [{ id: uuid, scope: "global" }],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const result = await store.delete(uuid);
    assert.equal(result, true);
    assert.ok(deletedWheres.length >= 1);
  });

  it("should return false when ID not found", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    const result = await store.delete("550e8400-e29b-41d4-a716-446655440000");
    assert.equal(result, false);
  });

  it("should throw on invalid ID format", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    await assert.rejects(
      () => store.delete("not-a-valid-id!"),
      /Invalid memory ID/,
    );
  });

  it("should throw when scope filter denies access", async () => {
    const uuid = "550e8400-e29b-41d4-a716-446655440000";
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => [{ id: uuid, scope: "private:other" }],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await assert.rejects(
      () => store.delete(uuid, ["agent:test"]),
      /outside accessible scopes/,
    );
  });

  it("should delete by prefix", async () => {
    const fullId = "abcdef01-2345-6789-abcd-ef0123456789";
    const { table, deletedWheres } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => [{ id: fullId, scope: "global" }],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const result = await store.delete("abcdef01");
    assert.equal(result, true);
  });

  it("should throw on ambiguous prefix", async () => {
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => [
          { id: "abcdef01-1111-1111-1111-111111111111", scope: "global" },
          { id: "abcdef01-2222-2222-2222-222222222222", scope: "global" },
        ],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await assert.rejects(
      () => store.delete("abcdef01"),
      /Ambiguous prefix/,
    );
  });
});

describe("src/store.ts — list()", () => {
  it("should return entries sorted by timestamp descending", async () => {
    const fakeEntries = [
      { id: "l1", text: "entry 1", category: "fact", scope: "agent:test", importance: 0.5, timestamp: 1000, metadata: "{}" },
      { id: "l2", text: "entry 2", category: "preference", scope: "agent:test", importance: 0.8, timestamp: 2000, metadata: "{}" },
    ];
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => fakeEntries,
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const results = await store.list();
    assert.equal(results.length, 2);
    // Should be sorted: newer first
    assert.equal(results[0].id, "l2");
    assert.equal(results[1].id, "l1");
  });

  it("should apply scope filter", async () => {
    let capturedWhere: string | undefined;
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: (w: string) => { capturedWhere = w; return chain; },
        limit: () => chain,
        toArray: async () => [],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await store.list(["agent:test"]);
    assert.ok(capturedWhere !== undefined);
    assert.ok(capturedWhere!.includes("agent:test"));
  });

  it("should apply category filter", async () => {
    let capturedWhere: string | undefined;
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: (w: string) => { capturedWhere = w; return chain; },
        limit: () => chain,
        toArray: async () => [],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await store.list(undefined, "fact");
    assert.ok(capturedWhere !== undefined);
    assert.ok(capturedWhere!.includes("fact"));
  });

  it("should apply pagination via offset and limit", async () => {
    const entries = Array.from({ length: 10 }, (_, i) => ({
      id: `e${i}`, text: `entry ${i}`, category: "fact", scope: "global", importance: 0.5, timestamp: i * 1000, metadata: "{}",
    }));
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => entries,
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const results = await store.list(undefined, undefined, 3, 2);
    assert.equal(results.length, 3);
  });
});

describe("src/store.ts — stats()", () => {
  it("should return aggregated stats", async () => {
    const fakeRows = [
      { scope: "agent:test", category: "fact" },
      { scope: "agent:test", category: "fact" },
      { scope: "user:global", category: "preference" },
    ];
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => fakeRows,
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const stats = await store.stats();
    assert.equal(stats.totalCount, 3);
    assert.equal(stats.scopeCounts["agent:test"], 2);
    assert.equal(stats.scopeCounts["user:global"], 1);
    assert.equal(stats.categoryCounts["fact"], 2);
    assert.equal(stats.categoryCounts["preference"], 1);
  });

  it("should apply scope filter to stats query", async () => {
    let capturedWhere: string | undefined;
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: (w: string) => { capturedWhere = w; return chain; },
        limit: () => chain,
        toArray: async () => [],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await store.stats(["agent:test"]);
    assert.ok(capturedWhere !== undefined);
    assert.ok(capturedWhere!.includes("agent:test"));
  });

  it("should return zero counts on empty table", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    const stats = await store.stats();
    assert.equal(stats.totalCount, 0);
    assert.deepEqual(stats.scopeCounts, {});
    assert.deepEqual(stats.categoryCounts, {});
  });
});

describe("src/store.ts — update()", () => {
  it("should delete old entry and store updated one", async () => {
    const original = { id: "550e8400-e29b-41d4-a716-446655440000", text: "old text", vector: [0.1, 0.2, 0.3, 0.4], category: "fact", scope: "agent:test", importance: 0.5, timestamp: 1000, metadata: "{}" };
    const { table, addedRecords, deletedWheres } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => [original],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const result = await store.update(original.id, { text: "new text", vector: [0.5, 0.6, 0.7, 0.8] });
    assert.ok(deletedWheres.length >= 1);
    assert.ok(addedRecords.length >= 1);
    assert.equal(result!.text, "new text");
  });

  it("should update importance without changing text", async () => {
    const original = { id: "550e8400-e29b-41d4-a716-446655440001", text: "keep text", vector: [0.1, 0.2, 0.3, 0.4], category: "fact", scope: "global", importance: 0.5, timestamp: 1000, metadata: "{}" };
    const { table, addedRecords } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => [original],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const result = await store.update(original.id, { importance: 0.95 });
    assert.equal(result!.importance, 0.95);
    assert.equal(result!.text, "keep text");
  });

  it("should return null when entry not found", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    const result = await store.update("550e8400-e29b-41d4-a716-446655440099", { text: "nope" });
    assert.equal(result, null);
  });

  it("should throw on invalid ID format", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    await assert.rejects(
      () => store.update("invalid-id!!", { text: "test" }),
      /Invalid memory ID/,
    );
  });

  it("should throw when scope filter denies access", async () => {
    const original = { id: "550e8400-e29b-41d4-a716-446655440002", text: "secret", vector: [0.1, 0.2, 0.3, 0.4], category: "fact", scope: "private:other", importance: 0.5, timestamp: 1000, metadata: "{}" };
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => [original],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await assert.rejects(
      () => store.update(original.id, { text: "updated" }, ["agent:test"]),
      /outside accessible scopes/,
    );
  });
});

describe("src/store.ts — bulkDelete()", () => {
  it("should delete entries matching scope filter", async () => {
    const matchingRows = [{ id: "d1" }, { id: "d2" }];
    const { table, deletedWheres } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: () => chain,
        limit: () => chain,
        toArray: async () => matchingRows,
      };
      return chain;
    };
    const store = createInitializedStore(table);
    const count = await store.bulkDelete(["agent:test"]);
    assert.equal(count, 2);
    assert.ok(deletedWheres.length >= 1);
  });

  it("should return zero when no entries match", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    const count = await store.bulkDelete(["empty:scope"]);
    assert.equal(count, 0);
  });

  it("should throw when no filters provided", async () => {
    const { table } = createMockTable();
    const store = createInitializedStore(table);
    await assert.rejects(
      () => store.bulkDelete([]),
      /Bulk delete requires/,
    );
  });

  it("should accept beforeTimestamp filter", async () => {
    let capturedWhere: string | undefined;
    const { table } = createMockTable();
    table.query = () => {
      const chain: any = {
        select: () => chain,
        where: (w: string) => { capturedWhere = w; return chain; },
        limit: () => chain,
        toArray: async () => [],
      };
      return chain;
    };
    const store = createInitializedStore(table);
    await store.bulkDelete(["agent:test"], 1700000000000);
    assert.ok(capturedWhere !== undefined);
    assert.ok(capturedWhere!.includes("timestamp"));
  });
});

// ---------------------------------------------------------------------------
// FILE 3: src/tools.ts — tool registrations and execute functions
// ---------------------------------------------------------------------------

function createToolContext(overrides: Record<string, any> = {}) {
  return {
    retriever: {
      retrieve: async (ctx: any) => [],
      getConfig: () => ({ mode: "hybrid" }),
    },
    store: {
      store: async (entry: any) => ({ ...entry, id: "new-id", timestamp: Date.now() }),
      hasId: async () => false,
      vectorSearch: async () => [],
      bm25Search: async () => [],
      list: async () => [],
      stats: async () => ({ totalCount: 0, scopeCounts: {}, categoryCounts: {} }),
      delete: async () => true,
      update: async (id: string, updates: any) => ({ id, text: updates.text || "existing", vector: [0.1, 0.2, 0.3, 0.4], category: "fact", scope: "global", importance: 0.7, timestamp: Date.now(), metadata: "{}" }),
      hasFtsSupport: true,
    },
    scopeManager: {
      getAccessibleScopes: (agentId?: string) => ["agent:test", "user:global"],
      isAccessible: (scope: string, agentId?: string) => scope !== "forbidden:scope",
      resolveScope: (s: string) => s,
      getDefaultScope: (agentId?: string) => "agent:test",
      getStats: () => ({ totalScopes: 2, agentsWithCustomAccess: 0, scopesByType: { agent: 1, user: 1 } }),
    },
    embedder: {
      embed: async (text: string) => [0.1, 0.2, 0.3, 0.4],
      embedPassage: async (text: string) => [0.1, 0.2, 0.3, 0.4],
      embedBatchPassage: async (texts: string[]) => texts.map(() => [0.1, 0.2, 0.3, 0.4]),
    },
    agentId: "agent-test-123",
    ...overrides,
  };
}

describe("src/tools.ts — registerMemoryRecallTool", () => {
  let tool: any;

  beforeEach(() => {
    tool = null;
    const api = { registerTool: (def: any, opts: any) => { tool = def; } };
    registerMemoryRecallTool(api as any, createToolContext({
      retriever: {
        retrieve: async (ctx: any) => {
          if (ctx.query === "empty") return [];
          if (ctx.query === "fail") throw new Error("Retrieval failed");
          return [
            {
              entry: { id: "m1", text: "Found memory", category: "fact", scope: "agent:test", importance: 0.8, timestamp: 1000, metadata: "{}", vector: [] },
              score: 0.95,
              sources: { vector: { score: 0.95, rank: 1 } },
            },
          ];
        },
        getConfig: () => ({ mode: "hybrid" }),
      },
    }) as any);
  });

  it("should register a tool named memory_recall", () => {
    assert.ok(tool !== null);
    assert.equal(tool.name, "memory_recall");
  });

  it("should return results for a valid query", async () => {
    const result = await tool.execute("c1", { query: "test query" });
    assert.ok(result.content[0].text.includes("Found memory"));
    assert.equal(result.details.count, 1);
  });

  it("should handle empty results", async () => {
    const result = await tool.execute("c2", { query: "empty" });
    assert.ok(result.content[0].text.includes("No relevant memories"));
    assert.equal(result.details.count, 0);
  });

  it("should handle retrieval errors", async () => {
    const result = await tool.execute("c3", { query: "fail" });
    assert.ok(result.content[0].text.includes("failed"));
  });

  it("should deny access to forbidden scope", async () => {
    const result = await tool.execute("c4", { query: "test", scope: "forbidden:scope" });
    assert.ok(result.content[0].text.includes("Access denied"));
  });
});

describe("src/tools.ts — registerMemoryStoreTool", () => {
  let tool: any;
  let storedEntries: any[];

  beforeEach(() => {
    tool = null;
    storedEntries = [];
    const api = { registerTool: (def: any, opts: any) => { tool = def; } };
    registerMemoryStoreTool(api as any, createToolContext({
      store: {
        store: async (entry: any) => { storedEntries.push(entry); return { ...entry, id: "new-id", timestamp: Date.now() }; },
        vectorSearch: async () => [],
      },
    }) as any);
  });

  it("should register a tool named memory_store", () => {
    assert.equal(tool.name, "memory_store");
  });

  it("should store a valid memory entry", async () => {
    const result = await tool.execute("s1", { text: "The user prefers dark mode for all editors", category: "preference", importance: 0.8 });
    assert.ok(result.content[0].text.includes("Stored"));
    assert.ok(storedEntries.length >= 1);
  });

  it("should reject noise content", async () => {
    const result = await tool.execute("s2", { text: "ok" });
    assert.ok(result.content[0].text.includes("noise"));
  });

  it("should reject access to forbidden scope", async () => {
    const result = await tool.execute("s3", { text: "Secret data that needs storing somewhere", scope: "forbidden:scope" });
    assert.ok(result.content[0].text.includes("Access denied"));
  });

  it("should detect duplicates", async () => {
    let storeTool: any = null;
    const api2 = { registerTool: (def: any) => { storeTool = def; } };
    registerMemoryStoreTool(api2 as any, createToolContext({
      store: {
        store: async (entry: any) => ({ ...entry, id: "x", timestamp: Date.now() }),
        vectorSearch: async () => [{ entry: { id: "dup-1", text: "Already stored", category: "fact", scope: "agent:test", importance: 0.8, timestamp: 0, metadata: "{}", vector: [] }, score: 0.99 }],
      },
    }) as any);
    const result = await storeTool.execute("s4", { text: "Already stored memory content" });
    assert.ok(result.content[0].text.includes("Similar memory already exists"));
  });

  it("should handle store errors", async () => {
    let errTool: any = null;
    const api3 = { registerTool: (def: any) => { errTool = def; } };
    registerMemoryStoreTool(api3 as any, createToolContext({
      store: {
        store: async () => { throw new Error("DB write failed"); },
        vectorSearch: async () => [],
      },
    }) as any);
    const result = await errTool.execute("s5", { text: "Will fail to store because of DB error", category: "fact" });
    assert.ok(result.content[0].text.includes("failed"));
  });
});

describe("src/tools.ts — registerMemoryForgetTool", () => {
  let tool: any;
  let deletedIds: string[];

  beforeEach(() => {
    tool = null;
    deletedIds = [];
    const api = { registerTool: (def: any) => { tool = def; } };
    registerMemoryForgetTool(api as any, createToolContext({
      store: {
        delete: async (id: string) => { deletedIds.push(id); return true; },
      },
      retriever: {
        retrieve: async (ctx: any) => {
          if (ctx.query === "single-match") return [{
            entry: { id: "found-1", text: "Memory to delete", category: "fact", scope: "agent:test", importance: 0.5, timestamp: 0, metadata: "{}", vector: [] },
            score: 0.99,
            sources: { vector: { score: 0.99, rank: 1 } },
          }];
          if (ctx.query === "multi-match") return [
            { entry: { id: "m1", text: "Match 1", category: "fact", scope: "agent:test", importance: 0.5, timestamp: 0, metadata: "{}", vector: [] }, score: 0.85, sources: {} },
            { entry: { id: "m2", text: "Match 2", category: "fact", scope: "agent:test", importance: 0.5, timestamp: 0, metadata: "{}", vector: [] }, score: 0.80, sources: {} },
          ];
          return [];
        },
        getConfig: () => ({ mode: "hybrid" }),
      },
    }) as any);
  });

  it("should register a tool named memory_forget", () => {
    assert.equal(tool.name, "memory_forget");
  });

  it("should delete by memoryId", async () => {
    const result = await tool.execute("f1", { memoryId: "550e8400-e29b-41d4-a716-446655440000" });
    assert.ok(result.content[0].text.includes("forgotten"));
    assert.ok(deletedIds.length >= 1);
  });

  it("should auto-delete a single high-score query match", async () => {
    const result = await tool.execute("f2", { query: "single-match" });
    assert.ok(result.content[0].text.includes("Forgotten") || result.content[0].text.includes("forgotten"));
  });

  it("should list candidates for multiple query matches", async () => {
    const result = await tool.execute("f3", { query: "multi-match" });
    assert.ok(result.content[0].text.includes("candidates") || result.content[0].text.includes("Specify"));
  });

  it("should return no matches for empty query results", async () => {
    const result = await tool.execute("f4", { query: "no-results" });
    assert.ok(result.content[0].text.includes("No matching"));
  });

  it("should require query or memoryId", async () => {
    const result = await tool.execute("f5", {});
    assert.ok(result.content[0].text.includes("Provide"));
  });

  it("should handle delete errors", async () => {
    let errTool: any = null;
    const api2 = { registerTool: (def: any) => { errTool = def; } };
    registerMemoryForgetTool(api2 as any, createToolContext({
      store: { delete: async () => { throw new Error("Delete failed"); } },
    }) as any);
    const result = await errTool.execute("f6", { memoryId: "550e8400-e29b-41d4-a716-446655440000" });
    assert.ok(result.content[0].text.includes("failed"));
  });
});

describe("src/tools.ts — registerMemoryUpdateTool", () => {
  let tool: any;
  let updateCalls: any[];
  let embedCalls: string[];

  beforeEach(() => {
    tool = null;
    updateCalls = [];
    embedCalls = [];
    const api = { registerTool: (def: any) => { tool = def; } };
    registerMemoryUpdateTool(api as any, createToolContext({
      store: {
        update: async (id: string, updates: any, scopeFilter?: string[]) => {
          updateCalls.push({ id, updates });
          return { id, text: updates.text || "existing", vector: [0.5, 0.6, 0.7, 0.8], category: updates.category || "fact", scope: "global", importance: updates.importance || 0.7, timestamp: 1000, metadata: "{}" };
        },
      },
      embedder: {
        embedPassage: async (text: string) => { embedCalls.push(text); return [0.5, 0.6, 0.7, 0.8]; },
      },
    }) as any);
  });

  it("should register a tool named memory_update", () => {
    assert.equal(tool.name, "memory_update");
  });

  it("should update content and re-embed", async () => {
    const result = await tool.execute("u1", { memoryId: "550e8400-e29b-41d4-a716-446655440000", text: "Updated content" });
    assert.ok(result.content[0].text.includes("Updated"));
    assert.ok(embedCalls.length >= 1);
    assert.ok(updateCalls.length >= 1);
    assert.equal(updateCalls[0].updates.text, "Updated content");
  });

  it("should update importance without re-embedding", async () => {
    const result = await tool.execute("u2", { memoryId: "550e8400-e29b-41d4-a716-446655440000", importance: 0.99 });
    assert.ok(result.content[0].text.includes("Updated"));
    assert.equal(embedCalls.length, 0);
    assert.ok(updateCalls.length >= 1);
    assert.ok(updateCalls[0].updates.importance !== undefined);
  });

  it("should return error when no updates provided", async () => {
    const result = await tool.execute("u3", { memoryId: "550e8400-e29b-41d4-a716-446655440000" });
    assert.ok(result.content[0].text.includes("Nothing to update"));
  });

  it("should handle update errors", async () => {
    let errTool: any = null;
    const api2 = { registerTool: (def: any) => { errTool = def; } };
    registerMemoryUpdateTool(api2 as any, createToolContext({
      store: { update: async () => { throw new Error("Update failed"); } },
    }) as any);
    const result = await errTool.execute("u4", { memoryId: "550e8400-e29b-41d4-a716-446655440000", text: "will fail" });
    assert.ok(result.content[0].text.includes("failed"));
  });

  it("should resolve non-UUID memoryId via search", async () => {
    let resolveTool: any = null;
    const api3 = { registerTool: (def: any) => { resolveTool = def; } };
    registerMemoryUpdateTool(api3 as any, createToolContext({
      retriever: {
        retrieve: async (ctx: any) => [{
          entry: { id: "550e8400-e29b-41d4-a716-446655440011", text: "Found entry", category: "fact", scope: "global", importance: 0.5, timestamp: 0, metadata: "{}", vector: [] },
          score: 0.95,
          sources: {},
        }],
        getConfig: () => ({ mode: "hybrid" }),
      },
      store: {
        update: async (id: string, updates: any) => ({ id, text: updates.text || "test", vector: [], category: "fact", scope: "global", importance: 0.7, timestamp: 0, metadata: "{}" }),
      },
    }) as any);
    const result = await resolveTool.execute("u5", { memoryId: "some natural language query", text: "Updated text" });
    // Should either update the resolved entry or show candidates
    assert.ok(result.content[0].text.includes("Updated") || result.content[0].text.includes("matches") || result.content[0].text.includes("Multiple"));
  });

  it("should reject noise text updates", async () => {
    const result = await tool.execute("u6", { memoryId: "550e8400-e29b-41d4-a716-446655440000", text: "ok" });
    assert.ok(result.content[0].text.includes("noise"));
  });
});

describe("src/tools.ts — registerMemoryStatsTool", () => {
  let tool: any;

  beforeEach(() => {
    tool = null;
    const api = { registerTool: (def: any) => { tool = def; } };
    registerMemoryStatsTool(api as any, createToolContext({
      store: {
        stats: async () => ({
          totalCount: 42,
          scopeCounts: { "agent:test": 30, "user:global": 12 },
          categoryCounts: { fact: 20, preference: 15, decision: 7 },
        }),
        hasFtsSupport: true,
      },
    }) as any);
  });

  it("should register a tool named memory_stats", () => {
    assert.equal(tool.name, "memory_stats");
  });

  it("should return stats text with totals", async () => {
    const result = await tool.execute("st1", {});
    assert.ok(result.content[0].text.includes("42"));
    assert.ok(result.content[0].text.includes("Memory Statistics"));
  });

  it("should handle scope denied", async () => {
    const result = await tool.execute("st2", { scope: "forbidden:scope" });
    assert.ok(result.content[0].text.includes("Access denied"));
  });

  it("should handle stats errors", async () => {
    let errTool: any = null;
    const api2 = { registerTool: (def: any) => { errTool = def; } };
    registerMemoryStatsTool(api2 as any, createToolContext({
      store: { stats: async () => { throw new Error("Stats failed"); }, hasFtsSupport: false },
    }) as any);
    const result = await errTool.execute("st3", {});
    assert.ok(result.content[0].text.includes("Failed"));
  });
});

describe("src/tools.ts — registerMemoryListTool", () => {
  let tool: any;

  beforeEach(() => {
    tool = null;
    const api = { registerTool: (def: any) => { tool = def; } };
    registerMemoryListTool(api as any, createToolContext({
      store: {
        list: async (scopeFilter?: string[], category?: string, limit?: number, offset?: number) => [
          { id: "l1", text: "Entry one text content", category: "fact", scope: "agent:test", importance: 0.8, timestamp: 1700000000000, metadata: "{}" },
          { id: "l2", text: "Entry two text content", category: "preference", scope: "agent:test", importance: 0.6, timestamp: 1700000001000, metadata: "{}" },
        ],
      },
    }) as any);
  });

  it("should register a tool named memory_list", () => {
    assert.equal(tool.name, "memory_list");
  });

  it("should return entries", async () => {
    const result = await tool.execute("l1", {});
    assert.ok(result.content[0].text.includes("Recent memories"));
    assert.equal(result.details.count, 2);
  });

  it("should handle empty list", async () => {
    let emptyTool: any = null;
    const api2 = { registerTool: (def: any) => { emptyTool = def; } };
    registerMemoryListTool(api2 as any, createToolContext({
      store: { list: async () => [] },
    }) as any);
    const result = await emptyTool.execute("l2", {});
    assert.ok(result.content[0].text.includes("No memories"));
  });

  it("should handle scope denied", async () => {
    const result = await tool.execute("l3", { scope: "forbidden:scope" });
    assert.ok(result.content[0].text.includes("Access denied"));
  });

  it("should handle list errors", async () => {
    let errTool: any = null;
    const api3 = { registerTool: (def: any) => { errTool = def; } };
    registerMemoryListTool(api3 as any, createToolContext({
      store: { list: async () => { throw new Error("List failed"); } },
    }) as any);
    const result = await errTool.execute("l4", {});
    assert.ok(result.content[0].text.includes("Failed"));
  });
});

describe("src/tools.ts — registerAllMemoryTools", () => {
  it("should register 4 core tools by default", () => {
    const tools: any[] = [];
    const api = { registerTool: (def: any, opts: any) => { tools.push(def); } };
    registerAllMemoryTools(api as any, createToolContext() as any);
    assert.equal(tools.length, 4);
  });

  it("should register 6 tools when enableManagementTools is true", () => {
    const tools: any[] = [];
    const api = { registerTool: (def: any, opts: any) => { tools.push(def); } };
    registerAllMemoryTools(api as any, createToolContext() as any, { enableManagementTools: true });
    assert.equal(tools.length, 6);
  });

  it("should register tools with correct names", () => {
    const tools: any[] = [];
    const api = { registerTool: (def: any, opts: any) => { tools.push(def); } };
    registerAllMemoryTools(api as any, createToolContext() as any, { enableManagementTools: true });
    const names = tools.map((t) => t.name);
    assert.ok(names.includes("memory_recall"));
    assert.ok(names.includes("memory_store"));
    assert.ok(names.includes("memory_forget"));
    assert.ok(names.includes("memory_update"));
    assert.ok(names.includes("memory_stats"));
    assert.ok(names.includes("memory_list"));
  });
});

// ---------------------------------------------------------------------------
// FILE 4: cli.ts — registerMemoryCLI
// ---------------------------------------------------------------------------

describe("cli.ts — registerMemoryCLI", () => {
  let consoleLogOutput: string[];
  let consoleErrorOutput: string[];
  let originalLog: typeof console.log;
  let originalError: typeof console.error;
  let originalExit: typeof process.exit;
  let mockContext: any;

  beforeEach(() => {
    consoleLogOutput = [];
    consoleErrorOutput = [];
    originalLog = console.log;
    originalError = console.error;
    originalExit = process.exit;
    console.log = (...args: any[]) => { consoleLogOutput.push(args.map(String).join(" ")); };
    console.error = (...args: any[]) => { consoleErrorOutput.push(args.map(String).join(" ")); };
    process.exit = (() => { throw new Error("process.exit called"); }) as any;

    mockContext = {
      store: {
        list: async () => [
          { id: "cli-1", text: "CLI test entry", category: "fact", scope: "agent:test", importance: 0.7, timestamp: 1700000000000, metadata: "{}" },
        ],
        stats: async () => ({
          totalCount: 10,
          scopeCounts: { "agent:test": 10 },
          categoryCounts: { fact: 5, preference: 3, decision: 2 },
        }),
        delete: async (id: string) => id === "cli-1",
        bulkDelete: async () => 3,
        importEntry: async () => {},
        hasId: async () => false,
        store: async (entry: any) => ({ ...entry, id: "new", timestamp: Date.now() }),
        vectorSearch: async () => [],
        hasFtsSupport: true,
      },
      retriever: {
        retrieve: async (ctx: any) => [
          {
            entry: { id: "s1", text: "Search result", category: "fact", scope: "agent:test", importance: 0.8, timestamp: 1700000000000, metadata: "{}", vector: [] },
            score: 0.9,
            sources: { vector: { score: 0.9, rank: 1 } },
          },
        ],
        getConfig: () => ({ mode: "hybrid" }),
      },
      scopeManager: {
        getAccessibleScopes: () => ["agent:test", "user:global"],
        isAccessible: () => true,
        resolveScope: (s: string) => s,
        getDefaultScope: () => "agent:test",
        getStats: () => ({ totalScopes: 2, agentsWithCustomAccess: 0, scopesByType: { agent: 1, user: 1 } }),
      },
      embedder: {
        embedPassage: async () => [0.1, 0.2, 0.3, 0.4],
        embedBatchPassage: async (texts: string[]) => texts.map(() => [0.1, 0.2, 0.3, 0.4]),
      },
      migrator: {
        checkMigrationNeeded: async () => ({ needed: false, sourceFound: false }),
        migrate: async () => ({ success: true, migratedCount: 0, skippedCount: 0, errors: [], summary: "OK" }),
        verifyMigration: async () => ({ valid: true, sourceCount: 0, targetCount: 0, issues: [] }),
      },
    };
  });

  afterEach(() => {
    console.log = originalLog;
    console.error = originalError;
    process.exit = originalExit;
  });

  it("should register memory commands on the program", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    const memoryCmd = program.commands.find((c: any) => c.name() === "memory");
    assert.ok(memoryCmd !== undefined, "Expected 'memory' command");
  });

  it("should list memories", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "list"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("CLI test entry") || output.includes("1 memories"));
  });

  it("should show stats", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "stats"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("10") || output.includes("Memory Statistics"));
  });

  it("should search memories", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "search", "test query"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("Search result") || output.includes("1 memories"));
  });

  it("should delete a memory", async () => {
    let deletedId: string | undefined;
    mockContext.store.delete = async (id: string) => { deletedId = id; return true; };
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "delete", "cli-1"], { from: "user" });
    } catch {}
    assert.equal(deletedId, "cli-1");
    assert.ok(consoleLogOutput.join("\n").includes("deleted"));
  });

  it("should export memories", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "export"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("version") || output.includes("memories"));
  });

  it("should check migration status", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "migrate", "check"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("Legacy database found") || output.includes("Migration needed"));
  });

  it("should run migration", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "migrate", "run"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("Status") || output.includes("Migrated"));
  });

  it("should verify migration", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "migrate", "verify"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("Valid") || output.includes("Source"));
  });

  it("should handle list error gracefully", async () => {
    mockContext.store.list = async () => { throw new Error("DB connection lost"); };
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "list"], { from: "user" });
    } catch {}
    const errOutput = consoleErrorOutput.join("\n");
    assert.ok(errOutput.includes("Failed") || errOutput.includes("error") || errOutput.includes("Error"));
  });

  it("should handle stats error gracefully", async () => {
    mockContext.store.stats = async () => { throw new Error("Stats failed"); };
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "stats"], { from: "user" });
    } catch {}
    const errOutput = consoleErrorOutput.join("\n");
    assert.ok(errOutput.includes("Failed") || errOutput.includes("error"));
  });

  it("should handle search with no results", async () => {
    mockContext.retriever.retrieve = async () => [];
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "search", "nothing"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("No relevant memories"));
  });

  it("should handle delete-bulk with scope", async () => {
    const { Command } = await import("commander");
    const program = new Command();
    program.exitOverride();
    registerMemoryCLI(program, mockContext);
    try {
      await program.parseAsync(["node", "test", "memory", "delete-bulk", "--scope", "agent:test"], { from: "user" });
    } catch {}
    const output = consoleLogOutput.join("\n");
    assert.ok(output.includes("Deleted") || output.includes("3"));
  });
});

// ---------------------------------------------------------------------------
// FILE 5: src/migrate.ts — MemoryMigrator
// ---------------------------------------------------------------------------

describe("src/migrate.ts — MemoryMigrator", () => {
  it("should construct from a store", () => {
    const mockStore = { store: async () => {}, stats: async () => ({ totalCount: 0, scopeCounts: {}, categoryCounts: {} }) };
    const migrator = new MemoryMigrator(mockStore as any);
    assert.ok(migrator !== null);
  });

  it("checkMigrationNeeded should return false when no legacy data", async () => {
    const mockStore = { stats: async () => ({ totalCount: 0, scopeCounts: {}, categoryCounts: {} }) };
    const migrator = new MemoryMigrator(mockStore as any);
    const result = await migrator.checkMigrationNeeded();
    assert.equal(result.sourceFound, false);
    assert.equal(result.needed, false);
  });

  it("checkMigrationNeeded with non-existent explicit path should return false", async () => {
    const mockStore = { stats: async () => ({ totalCount: 0, scopeCounts: {}, categoryCounts: {} }) };
    const migrator = new MemoryMigrator(mockStore as any);
    const result = await migrator.checkMigrationNeeded("/tmp/definitely-nonexistent-path-xyz-42");
    assert.equal(result.sourceFound, false);
    assert.equal(result.needed, false);
  });

  it("migrate should report no source when none exists", async () => {
    const mockStore = { store: async () => {}, stats: async () => ({ totalCount: 0, scopeCounts: {}, categoryCounts: {} }) };
    const migrator = new MemoryMigrator(mockStore as any);
    const result = await migrator.migrate();
    assert.equal(result.success, false);
    assert.ok(result.errors.length > 0);
    assert.ok(result.errors[0].includes("No legacy database"));
  });

  it("verifyMigration should report source not found when none exists", async () => {
    const mockStore = { stats: async () => ({ totalCount: 0, scopeCounts: {}, categoryCounts: {} }) };
    const migrator = new MemoryMigrator(mockStore as any);
    const result = await migrator.verifyMigration();
    assert.equal(result.valid, false);
    assert.ok(result.issues.some((i: string) => i.includes("not found")));
  });
});

describe("src/migrate.ts — createMigrator()", () => {
  it("should create a MemoryMigrator instance", () => {
    const mockStore = {} as any;
    const migrator = createMigrator(mockStore);
    assert.ok(migrator instanceof MemoryMigrator);
  });
});

describe("src/migrate.ts — checkForLegacyData()", () => {
  it("should return found=false when no legacy data exists in default paths", async () => {
    // This will attempt to connect to LanceDB at default paths, which will fail
    // The function catches all errors and returns found=false
    const result = await checkForLegacyData();
    assert.equal(typeof result.found, "boolean");
    assert.ok(Array.isArray(result.paths));
    assert.equal(typeof result.totalEntries, "number");
    // In test environment, no legacy data should exist
    assert.equal(result.found, false);
  });
});
