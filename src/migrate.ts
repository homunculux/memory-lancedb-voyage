/**
 * Migration Utilities
 * Migrates data from old memory-lancedb plugin to memory-lancedb-voyage
 */

import { homedir } from "node:os";
import { join } from "node:path";
import fs from "node:fs/promises";
import type { MemoryStore, MemoryEntry } from "./store.js";
import { loadLanceDB } from "./store.js";

// ============================================================================
// Types
// ============================================================================

interface LegacyMemoryEntry {
  id: string;
  text: string;
  vector: number[];
  importance: number;
  category: "preference" | "fact" | "decision" | "entity" | "other";
  createdAt: number;
  scope?: string;
}

interface MigrationResult {
  success: boolean;
  migratedCount: number;
  skippedCount: number;
  errors: string[];
  summary: string;
}

interface MigrationOptions {
  sourceDbPath?: string;
  dryRun?: boolean;
  defaultScope?: string;
  skipExisting?: boolean;
}

// ============================================================================
// Default Paths
// ============================================================================

function getDefaultLegacyPaths(): string[] {
  const home = homedir();
  return [
    join(home, ".openclaw", "memory", "lancedb"),
    join(home, ".openclaw", "memory", "lancedb-pro"),
    join(home, ".claude", "memory", "lancedb"),
  ];
}

// ============================================================================
// Migration
// ============================================================================

export class MemoryMigrator {
  constructor(private targetStore: MemoryStore) {}

  async migrate(options: MigrationOptions = {}): Promise<MigrationResult> {
    const result: MigrationResult = {
      success: false,
      migratedCount: 0,
      skippedCount: 0,
      errors: [],
      summary: "",
    };

    try {
      const sourceDbPath = await this.findSourceDatabase(options.sourceDbPath);
      if (!sourceDbPath) {
        result.errors.push("No legacy database found to migrate from");
        result.summary = "Migration failed: No source database found";
        return result;
      }

      console.log(`Migrating from: ${sourceDbPath}`);

      const legacyEntries = await this.loadLegacyData(sourceDbPath);
      if (legacyEntries.length === 0) {
        result.summary = "Migration completed: No data to migrate";
        result.success = true;
        return result;
      }

      console.log(`Found ${legacyEntries.length} entries to migrate`);

      if (!options.dryRun) {
        const stats = await this.migrateEntries(legacyEntries, options);
        result.migratedCount = stats.migrated;
        result.skippedCount = stats.skipped;
        result.errors.push(...stats.errors);
      } else {
        result.summary = `Dry run: Would migrate ${legacyEntries.length} entries`;
        result.success = true;
        return result;
      }

      result.success = result.errors.length === 0;
      result.summary = `Migration ${result.success ? "completed" : "completed with errors"}: ${result.migratedCount} migrated, ${result.skippedCount} skipped`;
    } catch (error) {
      result.errors.push(`Migration failed: ${error instanceof Error ? error.message : String(error)}`);
      result.summary = "Migration failed due to unexpected error";
    }

    return result;
  }

  private async findSourceDatabase(explicitPath?: string): Promise<string | null> {
    if (explicitPath) {
      try {
        await fs.access(explicitPath);
        return explicitPath;
      } catch {
        return null;
      }
    }

    for (const path of getDefaultLegacyPaths()) {
      try {
        await fs.access(path);
        const files = await fs.readdir(path);
        if (files.some(f => f.endsWith(".lance") || f === "memories.lance")) {
          return path;
        }
      } catch {
        continue;
      }
    }

    return null;
  }

  private async loadLegacyData(sourceDbPath: string, limit?: number): Promise<LegacyMemoryEntry[]> {
    const lancedb = await loadLanceDB();
    const db = await lancedb.connect(sourceDbPath);

    try {
      const table = await db.openTable("memories");
      let query = table.query();
      if (limit) query = query.limit(limit);
      const entries = await query.toArray();

      return entries.map((row): LegacyMemoryEntry => ({
        id: row.id as string,
        text: row.text as string,
        vector: row.vector as number[],
        importance: row.importance as number,
        category: (row.category as LegacyMemoryEntry["category"]) || "other",
        createdAt: row.createdAt as number,
        scope: row.scope as string | undefined,
      }));
    } catch (error) {
      console.warn(`Failed to load legacy data: ${error}`);
      return [];
    }
  }

  private async migrateEntries(
    legacyEntries: LegacyMemoryEntry[],
    options: MigrationOptions,
  ): Promise<{ migrated: number; skipped: number; errors: string[] }> {
    let migrated = 0;
    let skipped = 0;
    const errors: string[] = [];
    const defaultScope = options.defaultScope || "global";

    for (const legacy of legacyEntries) {
      try {
        if (options.skipExisting) {
          const existing = await this.targetStore.vectorSearch(
            legacy.vector, 1, 0.9, [legacy.scope || defaultScope],
          );
          if (existing.length > 0 && existing[0].score > 0.95) {
            skipped++;
            continue;
          }
        }

        await this.targetStore.store({
          text: legacy.text,
          vector: legacy.vector,
          category: legacy.category,
          scope: legacy.scope || defaultScope,
          importance: legacy.importance,
          metadata: JSON.stringify({
            migratedFrom: "memory-lancedb",
            originalId: legacy.id,
            originalCreatedAt: legacy.createdAt,
          }),
        });
        migrated++;

        if (migrated % 100 === 0) {
          console.log(`Migrated ${migrated}/${legacyEntries.length} entries...`);
        }
      } catch (error) {
        errors.push(`Failed to migrate entry ${legacy.id}: ${error}`);
        skipped++;
      }
    }

    return { migrated, skipped, errors };
  }

  async checkMigrationNeeded(sourceDbPath?: string): Promise<{
    needed: boolean;
    sourceFound: boolean;
    sourceDbPath?: string;
    entryCount?: number;
  }> {
    const sourcePath = await this.findSourceDatabase(sourceDbPath);
    if (!sourcePath) return { needed: false, sourceFound: false };

    try {
      const entries = await this.loadLegacyData(sourcePath, 1);
      return { needed: entries.length > 0, sourceFound: true, sourceDbPath: sourcePath };
    } catch {
      return { needed: false, sourceFound: true, sourceDbPath: sourcePath };
    }
  }

  async verifyMigration(sourceDbPath?: string): Promise<{
    valid: boolean;
    sourceCount: number;
    targetCount: number;
    issues: string[];
  }> {
    const issues: string[] = [];

    try {
      const sourcePath = await this.findSourceDatabase(sourceDbPath);
      if (!sourcePath) {
        return { valid: false, sourceCount: 0, targetCount: 0, issues: ["Source database not found"] };
      }

      const sourceEntries = await this.loadLegacyData(sourcePath);
      const targetStats = await this.targetStore.stats();

      if (targetStats.totalCount < sourceEntries.length) {
        issues.push(`Target has fewer entries (${targetStats.totalCount}) than source (${sourceEntries.length})`);
      }

      return { valid: issues.length === 0, sourceCount: sourceEntries.length, targetCount: targetStats.totalCount, issues };
    } catch (error) {
      return { valid: false, sourceCount: 0, targetCount: 0, issues: [`Verification failed: ${error}`] };
    }
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createMigrator(targetStore: MemoryStore): MemoryMigrator {
  return new MemoryMigrator(targetStore);
}

export async function migrateFromLegacy(targetStore: MemoryStore, options: MigrationOptions = {}): Promise<MigrationResult> {
  return createMigrator(targetStore).migrate(options);
}

export async function checkForLegacyData(): Promise<{ found: boolean; paths: string[]; totalEntries: number }> {
  const paths: string[] = [];
  let totalEntries = 0;

  for (const path of getDefaultLegacyPaths()) {
    try {
      const lancedb = await loadLanceDB();
      const db = await lancedb.connect(path);
      const table = await db.openTable("memories");
      const entries = await table.query().select(["id"]).toArray();
      if (entries.length > 0) {
        paths.push(path);
        totalEntries += entries.length;
      }
    } catch {
      continue;
    }
  }

  return { found: paths.length > 0, paths, totalEntries };
}
