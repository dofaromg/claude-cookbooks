import { FlowParticleSnapshot, FlowContext } from '../../types';
import { ParticleStore } from './store';

export class ParticleEngine {
  constructor(private readonly store: ParticleStore) {}

  createParticle(content: string, context: FlowContext, summary?: string): FlowParticleSnapshot {
    return this.store.create(content, context, summary);
  }

  collapseParticle(id: string, by: string, note?: string): FlowParticleSnapshot {
    return this.store.collapse(id, by, note);
  }

  archiveParticle(id: string, by: string, note?: string): FlowParticleSnapshot {
    return this.store.archive(id, by, note);
  }

  getParticle(id: string): FlowParticleSnapshot | undefined {
    return this.store.get(id);
  }

  listParticles(): FlowParticleSnapshot[] {
    return this.store.list();
  }
}
