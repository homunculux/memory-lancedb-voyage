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
    const config = memoryConfigSchema.parse({ embedding: { apiKey: "k" }, captureLlmUrl: "http://my-llm:8080/v1" });
    assert.equal(config.captureLlmUrl, "http://my-llm:8080/v1");
  });
});
