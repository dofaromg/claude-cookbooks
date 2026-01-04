import { PersonaRecord, PersonaProfile, PersonaTriangle } from '../../types';
import { randomId } from '../../utils';

export class PersonaRegistry {
  private personas = new Map<string, PersonaRecord>();

  registerPersona(profile: Omit<PersonaProfile, 'id'> & { id?: string }, triangle?: PersonaTriangle): PersonaRecord {
    const persona: PersonaRecord = {
      profile: { ...profile, id: profile.id ?? randomId() },
      triangle,
      seeds: [],
    };
    this.personas.set(persona.profile.id, persona);
    return persona;
  }

  linkSeed(personaId: string, seedId: string): PersonaRecord {
    const persona = this.personas.get(personaId);
    if (!persona) {
      throw new Error(`Persona ${personaId} not found`);
    }
    if (!persona.seeds.includes(seedId)) {
      persona.seeds.push(seedId);
    }
    return persona;
  }

  listPersonas(): PersonaRecord[] {
    return [...this.personas.values()];
  }
}
