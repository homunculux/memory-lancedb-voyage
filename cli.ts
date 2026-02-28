/**
 * CLI Commands for Memory Management
 */

import type { Command } from "commander";
import { loadLanceDB, type MemoryEntry, type MemoryStore } from "./src/store.js";
import type { MemoryRetriever } from "./src/retriever.js";
import type { MemoryScopeManager } from "./src/scopes.js";
import type { MemoryMigrator } from "./src/migrate.js";

// ============================================================================
// Types
// ============================================================================

interface CLIContext {
  store: MemoryStore;
  retriever: MemoryRetriever;
  scopeManager: MemoryScopeManager;
  migrator: MemoryMigrator;
  embedder?: import("./src/embedder-interface.js").IEmbedder;
}

// ============================================================================
// Utility Functions
// ============================================================================

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.floor(value)));
}

function formatJson(obj: any): string {
  return JSON.stringify(obj, null, 2);
}

// ============================================================================
// CLI Registration
// ============================================================================

export function registerMemoryCLI(program: Command, context: CLIContext): void {
  const memory = program
    .command("memory")
    .description("Voyage AI memory management commands");

  // List
  memory
    .command("list")
    .description("List memories with optional filtering")
    .option("--scope <scope>", "Filter by scope")
    .option("--category <category>", "Filter by category")
    .option("--limit <n>", "Maximum number of results", "20")
    .option("--offset <n>", "Number of results to skip", "0")
    .option("--json", "Output as JSON")
    .action(async (options) => {
      try {
        const limit = parseInt(options.limit) || 20;
        const offset = parseInt(options.offset) || 0;
        const scopeFilter = options.scope ? [options.scope] : undefined;
        const memories = await context.store.list(scopeFilter, options.category, limit, offset);

        if (options.json) {
          console.log(formatJson(memories));
        } else if (memories.length === 0) {
          console.log("No memories found.");
        } else {
          console.log(`Found ${memories.length} memories:\n`);
          memories.forEach((memory, i) => {
            const date = new Date(memory.timestamp || Date.now()).toISOString().split("T")[0];
            const text = memory.text.slice(0, 100) + (memory.text.length > 100 ? "..." : "");
            console.log(`${offset + i + 1}. [${memory.category}:${memory.scope}] ${text} (${date})`);
          });
        }
      } catch (error) {
        console.error("Failed to list memories:", error);
        process.exit(1);
      }
    });

  // Search
  memory
    .command("search <query>")
    .description("Search memories using hybrid retrieval")
    .option("--scope <scope>", "Search within specific scope")
    .option("--category <category>", "Filter by category")
    .option("--limit <n>", "Maximum number of results", "10")
    .option("--json", "Output as JSON")
    .action(async (query, options) => {
      try {
        const limit = parseInt(options.limit) || 10;
        const scopeFilter = options.scope ? [options.scope] : undefined;
        const results = await context.retriever.retrieve({ query, limit, scopeFilter, category: options.category });

        if (options.json) {
          console.log(formatJson(results));
        } else if (results.length === 0) {
          console.log("No relevant memories found.");
        } else {
          console.log(`Found ${results.length} memories:\n`);
          results.forEach((result, i) => {
            const sources = [];
            if (result.sources.vector) sources.push("vector");
            if (result.sources.bm25) sources.push("BM25");
            if (result.sources.reranked) sources.push("reranked");
            console.log(
              `${i + 1}. [${result.entry.category}:${result.entry.scope}] ${result.entry.text} ` +
              `(${(result.score * 100).toFixed(0)}%, ${sources.join("+")})`,
            );
          });
        }
      } catch (error) {
        console.error("Search failed:", error);
        process.exit(1);
      }
    });

  // Stats
  memory
    .command("stats")
    .description("Show memory statistics")
    .option("--scope <scope>", "Stats for specific scope")
    .option("--json", "Output as JSON")
    .action(async (options) => {
      try {
        const scopeFilter = options.scope ? [options.scope] : undefined;
        const stats = await context.store.stats(scopeFilter);
        const scopeStats = context.scopeManager.getStats();
        const retrievalConfig = context.retriever.getConfig();

        if (options.json) {
          console.log(formatJson({ memory: stats, scopes: scopeStats, retrieval: { mode: retrievalConfig.mode, hasFtsSupport: context.store.hasFtsSupport } }));
        } else {
          console.log(`Memory Statistics:`);
          console.log(`\u2022 Total memories: ${stats.totalCount}`);
          console.log(`\u2022 Retrieval mode: ${retrievalConfig.mode}`);
          console.log(`\u2022 FTS support: ${context.store.hasFtsSupport ? "Yes" : "No"}`);
          console.log();
          console.log("Memories by scope:");
          Object.entries(stats.scopeCounts).forEach(([scope, count]) => console.log(`  \u2022 ${scope}: ${count}`));
          console.log();
          console.log("Memories by category:");
          Object.entries(stats.categoryCounts).forEach(([category, count]) => console.log(`  \u2022 ${category}: ${count}`));
        }
      } catch (error) {
        console.error("Failed to get statistics:", error);
        process.exit(1);
      }
    });

  // Delete
  memory
    .command("delete <id>")
    .description("Delete a specific memory by ID")
    .option("--scope <scope>", "Scope for access control")
    .action(async (id, options) => {
      try {
        const scopeFilter = options.scope ? [options.scope] : undefined;
        const deleted = await context.store.delete(id, scopeFilter);
        if (deleted) console.log(`Memory ${id} deleted.`);
        else { console.log(`Memory ${id} not found.`); process.exit(1); }
      } catch (error) {
        console.error("Failed to delete memory:", error);
        process.exit(1);
      }
    });

  // Bulk delete
  memory
    .command("delete-bulk")
    .description("Bulk delete memories with filters")
    .option("--scope <scopes...>", "Scopes to delete from (required)")
    .option("--before <date>", "Delete before date (YYYY-MM-DD)")
    .option("--dry-run", "Show what would be deleted")
    .action(async (options) => {
      try {
        if (!options.scope || options.scope.length === 0) {
          console.error("At least one scope must be specified.");
          process.exit(1);
        }
        let beforeTimestamp: number | undefined;
        if (options.before) {
          const date = new Date(options.before);
          if (isNaN(date.getTime())) { console.error("Invalid date format."); process.exit(1); }
          beforeTimestamp = date.getTime();
        }
        if (options.dryRun) {
          const stats = await context.store.stats(options.scope);
          console.log(`DRY RUN: Would delete from ${stats.totalCount} memories.`);
        } else {
          const count = await context.store.bulkDelete(options.scope, beforeTimestamp);
          console.log(`Deleted ${count} memories.`);
        }
      } catch (error) {
        console.error("Bulk delete failed:", error);
        process.exit(1);
      }
    });

  // Export
  memory
    .command("export")
    .description("Export memories to JSON")
    .option("--scope <scope>", "Export specific scope")
    .option("--category <category>", "Export specific category")
    .option("--output <file>", "Output file (default: stdout)")
    .action(async (options) => {
      try {
        const scopeFilter = options.scope ? [options.scope] : undefined;
        const memories = await context.store.list(scopeFilter, options.category, 1000);
        const exportData = {
          version: "1.0",
          exportedAt: new Date().toISOString(),
          count: memories.length,
          memories: memories.map(m => ({ ...m, vector: undefined })),
        };
        const output = formatJson(exportData);
        if (options.output) {
          const fs = await import("node:fs/promises");
          await fs.writeFile(options.output, output);
          console.log(`Exported ${memories.length} memories to ${options.output}`);
        } else {
          console.log(output);
        }
      } catch (error) {
        console.error("Export failed:", error);
        process.exit(1);
      }
    });

  // Import
  memory
    .command("import <file>")
    .description("Import memories from JSON file")
    .option("--scope <scope>", "Import into specific scope")
    .option("--dry-run", "Show what would be imported")
    .action(async (file, options) => {
      try {
        const fs = await import("node:fs/promises");
        const content = await fs.readFile(file, "utf-8");
        const data = JSON.parse(content);
        if (!data.memories || !Array.isArray(data.memories)) throw new Error("Invalid import file");

        if (options.dryRun) {
          console.log(`DRY RUN: Would import ${data.memories.length} memories`);
          return;
        }
        if (!context.embedder) {
          console.error("Import requires an embedder.");
          return;
        }

        const targetScope = options.scope || context.scopeManager.getDefaultScope();
        let imported = 0, skipped = 0;

        for (const memory of data.memories) {
          try {
            const text = memory.text;
            if (!text || typeof text !== "string" || text.length < 2) { skipped++; continue; }
            const existing = await context.retriever.retrieve({ query: text, limit: 1, scopeFilter: [targetScope] });
            if (existing.length > 0 && existing[0].score > 0.95) { skipped++; continue; }
            const vector = await context.embedder.embedPassage(text);
            await context.store.store({ text, vector, importance: memory.importance ?? 0.7, category: memory.category || "other", scope: targetScope });
            imported++;
          } catch { skipped++; }
        }

        console.log(`Import completed: ${imported} imported, ${skipped} skipped`);
      } catch (error) {
        console.error("Import failed:", error);
        process.exit(1);
      }
    });

  // Re-embed
  memory
    .command("reembed")
    .description("Re-embed memories from a source LanceDB")
    .requiredOption("--source-db <path>", "Source LanceDB directory")
    .option("--batch-size <n>", "Batch size", "32")
    .option("--limit <n>", "Limit rows")
    .option("--dry-run", "Show what would be re-embedded")
    .option("--skip-existing", "Skip existing IDs")
    .action(async (options) => {
      try {
        if (!context.embedder) {
          console.error("Re-embed requires an embedder.");
          return;
        }

        const sourceDbPath = options.sourceDb as string;
        const batchSize = clampInt(parseInt(options.batchSize, 10) || 32, 1, 128);
        const limit = options.limit ? clampInt(parseInt(options.limit, 10) || 0, 1, 1000000) : undefined;

        const lancedb = await loadLanceDB();
        const db = await lancedb.connect(sourceDbPath);
        const table = await db.openTable("memories");

        let query = table.query().select(["id", "text", "category", "scope", "importance", "timestamp", "metadata"]);
        if (limit) query = query.limit(limit);

        const rows = (await query.toArray())
          .filter((r: any) => r && typeof r.text === "string" && r.text.trim().length > 0)
          .filter((r: any) => r.id && r.id !== "__schema__");

        if (rows.length === 0) { console.log("No source memories found."); return; }

        console.log(`Re-embedding ${rows.length} memories (batchSize=${batchSize})`);
        if (options.dryRun) { console.log("DRY RUN"); return; }

        let processed = 0, imported = 0, skipped = 0;

        for (let i = 0; i < rows.length; i += batchSize) {
          const batch = rows.slice(i, i + batchSize);
          const texts = batch.map((r: any) => String(r.text));
          const vectors = await context.embedder.embedBatchPassage(texts);

          for (let j = 0; j < batch.length; j++) {
            processed++;
            const row = batch[j];
            const vector = vectors[j];
            if (!vector || vector.length === 0) { skipped++; continue; }

            const id = String(row.id);
            if (options.skipExisting) {
              const exists = await context.store.hasId(id);
              if (exists) { skipped++; continue; }
            }

            await context.store.importEntry({
              id,
              text: String(row.text),
              vector,
              category: (row.category as any) || "other",
              scope: (row.scope as string) || "global",
              importance: typeof row.importance === "number" ? row.importance : 0.7,
              timestamp: typeof row.timestamp === "number" ? row.timestamp : Date.now(),
              metadata: typeof row.metadata === "string" ? row.metadata : "{}",
            });
            imported++;
          }

          if (processed % 100 === 0 || processed === rows.length) {
            console.log(`Progress: ${processed}/${rows.length} processed, ${imported} imported, ${skipped} skipped`);
          }
        }

        console.log(`Re-embed completed: ${imported} imported, ${skipped} skipped`);
      } catch (error) {
        console.error("Re-embed failed:", error);
        process.exit(1);
      }
    });

  // Migration commands
  const migrate = memory.command("migrate").description("Migration utilities");

  migrate
    .command("check")
    .description("Check if migration is needed")
    .option("--source <path>", "Source database path")
    .action(async (options) => {
      try {
        const check = await context.migrator.checkMigrationNeeded(options.source);
        console.log(`Legacy database found: ${check.sourceFound ? "Yes" : "No"}`);
        if (check.sourceDbPath) console.log(`Source path: ${check.sourceDbPath}`);
        console.log(`Migration needed: ${check.needed ? "Yes" : "No"}`);
      } catch (error) {
        console.error("Migration check failed:", error);
        process.exit(1);
      }
    });

  migrate
    .command("run")
    .description("Run migration from legacy plugin")
    .option("--source <path>", "Source database path")
    .option("--default-scope <scope>", "Default scope", "global")
    .option("--dry-run", "Show what would be migrated")
    .option("--skip-existing", "Skip existing entries")
    .action(async (options) => {
      try {
        const result = await context.migrator.migrate({
          sourceDbPath: options.source,
          defaultScope: options.defaultScope,
          dryRun: options.dryRun,
          skipExisting: options.skipExisting,
        });
        console.log(`Status: ${result.success ? "Success" : "Failed"}`);
        console.log(`Migrated: ${result.migratedCount}, Skipped: ${result.skippedCount}`);
        if (result.errors.length > 0) result.errors.forEach(e => console.log(`  Error: ${e}`));
        if (!result.success) process.exit(1);
      } catch (error) {
        console.error("Migration failed:", error);
        process.exit(1);
      }
    });

  migrate
    .command("verify")
    .description("Verify migration results")
    .option("--source <path>", "Source database path")
    .action(async (options) => {
      try {
        const result = await context.migrator.verifyMigration(options.source);
        console.log(`Valid: ${result.valid ? "Yes" : "No"}`);
        console.log(`Source: ${result.sourceCount}, Target: ${result.targetCount}`);
        if (result.issues.length > 0) result.issues.forEach(i => console.log(`  Issue: ${i}`));
        if (!result.valid) process.exit(1);
      } catch (error) {
        console.error("Verification failed:", error);
        process.exit(1);
      }
    });
}

// ============================================================================
// Factory
// ============================================================================

export function createMemoryCLI(context: CLIContext) {
  return ({ program }: { program: Command }) => registerMemoryCLI(program, context);
}
