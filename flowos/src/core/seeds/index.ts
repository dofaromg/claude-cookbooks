import { SeedDefinition, SeedMigration } from '../../types';
import { now, randomId } from '../../utils';

export class SeedRegistry {
  private seeds = new Map<string, SeedDefinition>();

  registerSeed(seed: Omit<SeedDefinition, 'id' | 'migrations'> & { migrations?: SeedMigration[]; id?: string }): SeedDefinition {
    const definition: SeedDefinition = {
      ...seed,
      id: seed.id ?? randomId(),
      migrations: seed.migrations ?? [],
    };
    this.seeds.set(definition.id, definition);
    return definition;
  }

  addMigration(seedId: string, migration: Omit<SeedMigration, 'id' | 'createdAt'>): SeedDefinition {
    const seed = this.seeds.get(seedId);
    if (!seed) {
      throw new Error(`Seed ${seedId} not found`);
    }

    const record: SeedMigration = { ...migration, id: randomId(), createdAt: now() };
    seed.migrations.push(record);
    return seed;
  }

  listSeeds(): SeedDefinition[] {
    return [...this.seeds.values()];
  }
}
