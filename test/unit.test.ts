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

  // captureLlmApiKey tests
  it("captureLlmApiKey defaults to empty string when no config and no env vars", () => {
    const saved1 = process.env.OPENCLAW_LLM_API_KEY;
    const saved2 = process.env.OPENAI_API_KEY;
    delete process.env.OPENCLAW_LLM_API_KEY;
    delete process.env.OPENAI_API_KEY;
    try {
      const cfg = memoryConfigSchema.parse({
        embedding: { apiKey: "key" },
      });
      assert.equal(cfg.captureLlmApiKey, "");
    } finally {
      if (saved1 !== undefined) process.env.OPENCLAW_LLM_API_KEY = saved1; else delete process.env.OPENCLAW_LLM_API_KEY;
      if (saved2 !== undefined) process.env.OPENAI_API_KEY = saved2; else delete process.env.OPENAI_API_KEY;
    }
  });

  it("captureLlmApiKey uses config value when provided", () => {
    const cfg = memoryConfigSchema.parse({
      embedding: { apiKey: "key" },
      captureLlmApiKey: "sk-my-llm-key",
    });
    assert.equal(cfg.captureLlmApiKey, "sk-my-llm-key");
  });

  it("captureLlmApiKey supports ${ENV_VAR} resolution", () => {
    process.env.TEST_CAPTURE_LLM_KEY = "resolved-capture-key";
    try {
      const cfg = memoryConfigSchema.parse({
        embedding: { apiKey: "key" },
        captureLlmApiKey: "${TEST_CAPTURE_LLM_KEY}",
      });
      assert.equal(cfg.captureLlmApiKey, "resolved-capture-key");
    } finally {
      delete process.env.TEST_CAPTURE_LLM_KEY;
    }
  });

  it("captureLlmApiKey ${ENV_VAR} throws when env var not set", () => {
    delete process.env.NONEXISTENT_CAPTURE_KEY;
    assert.throws(
      () =>
        memoryConfigSchema.parse({
          embedding: { apiKey: "key" },
          captureLlmApiKey: "${NONEXISTENT_CAPTURE_KEY}",
        }),
      /Environment variable NONEXISTENT_CAPTURE_KEY is not set/,
    );
  });

  it("captureLlmApiKey falls back to OPENCLAW_LLM_API_KEY env var", () => {
    const saved1 = process.env.OPENCLAW_LLM_API_KEY;
    const saved2 = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;
    process.env.OPENCLAW_LLM_API_KEY = "openclaw-llm-key";
    try {
      const cfg = memoryConfigSchema.parse({
        embedding: { apiKey: "key" },
      });
      assert.equal(cfg.captureLlmApiKey, "openclaw-llm-key");
    } finally {
      if (saved1 !== undefined) process.env.OPENCLAW_LLM_API_KEY = saved1; else delete process.env.OPENCLAW_LLM_API_KEY;
      if (saved2 !== undefined) process.env.OPENAI_API_KEY = saved2; else delete process.env.OPENAI_API_KEY;
    }
  });

  it("captureLlmApiKey falls back to OPENAI_API_KEY env var", () => {
    const saved1 = process.env.OPENCLAW_LLM_API_KEY;
    const saved2 = process.env.OPENAI_API_KEY;
    delete process.env.OPENCLAW_LLM_API_KEY;
    process.env.OPENAI_API_KEY = "openai-fallback-key";
    try {
      const cfg = memoryConfigSchema.parse({
        embedding: { apiKey: "key" },
      });
      assert.equal(cfg.captureLlmApiKey, "openai-fallback-key");
    } finally {
      if (saved1 !== undefined) process.env.OPENCLAW_LLM_API_KEY = saved1; else delete process.env.OPENCLAW_LLM_API_KEY;
      if (saved2 !== undefined) process.env.OPENAI_API_KEY = saved2; else delete process.env.OPENAI_API_KEY;
    }
  });

  it("captureLlmApiKey: OPENCLAW_LLM_API_KEY takes priority over OPENAI_API_KEY", () => {
    const saved1 = process.env.OPENCLAW_LLM_API_KEY;
    const saved2 = process.env.OPENAI_API_KEY;
    process.env.OPENCLAW_LLM_API_KEY = "openclaw-wins";
    process.env.OPENAI_API_KEY = "openai-loses";
    try {
      const cfg = memoryConfigSchema.parse({
        embedding: { apiKey: "key" },
      });
      assert.equal(cfg.captureLlmApiKey, "openclaw-wins");
    } finally {
      if (saved1 !== undefined) process.env.OPENCLAW_LLM_API_KEY = saved1; else delete process.env.OPENCLAW_LLM_API_KEY;
      if (saved2 !== undefined) process.env.OPENAI_API_KEY = saved2; else delete process.env.OPENAI_API_KEY;
    }
  });

  it("captureLlmApiKey: explicit config value overrides env vars", () => {
    const saved1 = process.env.OPENCLAW_LLM_API_KEY;
    const saved2 = process.env.OPENAI_API_KEY;
    process.env.OPENCLAW_LLM_API_KEY = "env-key";
    process.env.OPENAI_API_KEY = "env-key-2";
    try {
      const cfg = memoryConfigSchema.parse({
        embedding: { apiKey: "key" },
        captureLlmApiKey: "explicit-key-wins",
      });
      assert.equal(cfg.captureLlmApiKey, "explicit-key-wins");
    } finally {
      if (saved1 !== undefined) process.env.OPENCLAW_LLM_API_KEY = saved1; else delete process.env.OPENCLAW_LLM_API_KEY;
      if (saved2 !== undefined) process.env.OPENAI_API_KEY = saved2; else delete process.env.OPENAI_API_KEY;
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
// 7b. callLlmForCaptureJudgment Authorization header
// ============================================================================

import { callLlmForCaptureJudgment } from "../index.js";

describe("callLlmForCaptureJudgment Authorization header", () => {
  const originalFetch = globalThis.fetch;
  const savedGatewayUrl = process.env.OPENCLAW_GATEWAY_URL;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    if (savedGatewayUrl !== undefined) {
      process.env.OPENCLAW_GATEWAY_URL = savedGatewayUrl;
    } else {
      delete process.env.OPENCLAW_GATEWAY_URL;
    }
  });

  const noop = () => {};
  const logger = { info: noop, warn: noop, debug: noop };

  const llmOkResponse = {
    choices: [{ message: { content: JSON.stringify({ store: false }) } }],
  };

  it("no Authorization header when captureLlmApiKey is empty", async () => {
    let capturedHeaders: Record<string, string> = {};
    // Prevent env-based fallback URLs from interfering
    delete process.env.OPENCLAW_GATEWAY_URL;

    globalThis.fetch = (async (_url: any, opts: any) => {
      capturedHeaders = opts.headers;
      return { ok: true, status: 200, json: async () => llmOkResponse };
    }) as unknown as typeof fetch;

    await callLlmForCaptureJudgment(
      "test conversation",
      "test-model",
      logger,
      "http://localhost:9999",
      "",
    );

    assert.equal(capturedHeaders["Content-Type"], "application/json");
    assert.equal(capturedHeaders["Authorization"], undefined);
  });

  it("no Authorization header when captureLlmApiKey is undefined", async () => {
    let capturedHeaders: Record<string, string> = {};
    delete process.env.OPENCLAW_GATEWAY_URL;

    globalThis.fetch = (async (_url: any, opts: any) => {
      capturedHeaders = opts.headers;
      return { ok: true, status: 200, json: async () => llmOkResponse };
    }) as unknown as typeof fetch;

    await callLlmForCaptureJudgment(
      "test conversation",
      "test-model",
      logger,
      "http://localhost:9999",
      undefined,
    );

    assert.equal(capturedHeaders["Content-Type"], "application/json");
    assert.equal(capturedHeaders["Authorization"], undefined);
  });

  it("sends Authorization: Bearer <key> when captureLlmApiKey is set", async () => {
    let capturedHeaders: Record<string, string> = {};
    delete process.env.OPENCLAW_GATEWAY_URL;

    globalThis.fetch = (async (_url: any, opts: any) => {
      capturedHeaders = opts.headers;
      return { ok: true, status: 200, json: async () => llmOkResponse };
    }) as unknown as typeof fetch;

    await callLlmForCaptureJudgment(
      "test conversation",
      "test-model",
      logger,
      "http://localhost:9999",
      "sk-test-secret-key",
    );

    assert.equal(capturedHeaders["Content-Type"], "application/json");
    assert.equal(capturedHeaders["Authorization"], "Bearer sk-test-secret-key");
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
