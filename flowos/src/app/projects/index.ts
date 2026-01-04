import { ProjectMetadata } from '../../types';
import { MemoryStorage } from '../../storage';
import { now, randomId } from '../../utils';

export class ProjectRegistry {
  constructor(private readonly storage: MemoryStorage) {}

  registerProject(name: string, description?: string): ProjectMetadata {
    const project: ProjectMetadata = {
      id: randomId(),
      name,
      description,
      createdAt: now(),
    };
    return this.storage.upsertProject(project);
  }

  listProjects(): ProjectMetadata[] {
    return this.storage.listProjects();
  }
}
