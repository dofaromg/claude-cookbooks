import { MemoryEntry } from '../../types';
import { MemoryStorage } from '../../storage';
import { now, randomId } from '../../utils';

export class MemorySystem {
  constructor(private readonly storage: MemoryStorage) {}

  remember(scope: MemoryEntry['scope'], topic: string, payload: Record<string, unknown>): MemoryEntry {
    const entry: MemoryEntry = {
      id: randomId(),
      scope,
      topic,
      payload,
      createdAt: now(),
    };
    return this.storage.upsertMemory(entry);
  }

  recall(scope?: MemoryEntry['scope']): MemoryEntry[] {
    return this.storage.listMemories(scope);
  }
}
