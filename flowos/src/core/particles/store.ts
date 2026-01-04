import { FlowParticleHistory, FlowParticleSnapshot, FlowParticleState, FlowContext } from '../../types';
import { randomId, now } from '../../utils';

export class ParticleStore {
  private snapshots = new Map<string, FlowParticleSnapshot>();

  create(content: string, context: FlowContext, summary?: string): FlowParticleSnapshot {
    const state: FlowParticleState = {
      id: randomId(),
      status: 'draft',
      content,
      summary,
      context,
      createdAt: now(),
      updatedAt: now(),
    };

    const snapshot: FlowParticleSnapshot = {
      id: state.id,
      history: [{ state, by: 'system', note: 'created' }],
      latest: state,
    };

    this.snapshots.set(snapshot.id, snapshot);
    return snapshot;
  }

  collapse(id: string, by: string, note?: string): FlowParticleSnapshot {
    const current = this.snapshots.get(id);
    if (!current) {
      throw new Error(`Particle ${id} not found`);
    }

    const nextState: FlowParticleState = {
      ...current.latest,
      status: 'collapsed',
      updatedAt: now(),
    };

    const historyEntry: FlowParticleHistory = { state: nextState, by, note };
    const updated: FlowParticleSnapshot = {
      ...current,
      latest: nextState,
      history: [...current.history, historyEntry],
    };

    this.snapshots.set(id, updated);
    return updated;
  }

  archive(id: string, by: string, note?: string): FlowParticleSnapshot {
    const current = this.snapshots.get(id);
    if (!current) {
      throw new Error(`Particle ${id} not found`);
    }

    const nextState: FlowParticleState = {
      ...current.latest,
      status: 'archived',
      updatedAt: now(),
    };

    const historyEntry: FlowParticleHistory = { state: nextState, by, note };
    const updated: FlowParticleSnapshot = {
      ...current,
      latest: nextState,
      history: [...current.history, historyEntry],
    };

    this.snapshots.set(id, updated);
    return updated;
  }

  get(id: string): FlowParticleSnapshot | undefined {
    return this.snapshots.get(id);
  }

  list(): FlowParticleSnapshot[] {
    return [...this.snapshots.values()];
  }
}
