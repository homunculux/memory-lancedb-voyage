
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
