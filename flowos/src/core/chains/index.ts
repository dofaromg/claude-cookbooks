import crypto from 'crypto';
import { FlowEvent, FlowContext, MerkleLink } from '../../types';
import { hashPayload } from '../../utils';

export class MerkleChain {
  private links: MerkleLink[] = [];

  append(event: FlowEvent, context: FlowContext): MerkleLink {
    const parent = this.links.length ? this.links[this.links.length - 1].hash : undefined;
    const payload = { event, context, parent };
    const hash = crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
    const link: MerkleLink = { hash, parent, context, event, createdAt: Date.now() };
    this.links.push(link);
    return link;
  }

  verify(): boolean {
    for (let i = 0; i < this.links.length; i += 1) {
      const link = this.links[i];
      const expectedHash = crypto
        .createHash('sha256')
        .update(JSON.stringify({ event: link.event, context: link.context, parent: link.parent }))
        .digest('hex');

      if (expectedHash !== link.hash) {
        return false;
      }
      if (i > 0 && this.links[i - 1].hash !== link.parent) {
        return false;
      }
    }
    return true;
  }

  trace(limit = 10): MerkleLink[] {
    return this.links.slice(-limit);
  }

  digest(): string {
    if (!this.links.length) return hashPayload({});
    return this.links[this.links.length - 1].hash;
  }
}
