export type UUID = string;

export interface FlowContext {
  id: UUID;
  project?: string;
  persona?: UUID;
  seed?: UUID;
  createdAt: number;
  metadata?: Record<string, unknown>;
}

export interface FlowEvent {
  id: UUID;
  type: string;
  payload: Record<string, unknown>;
  createdAt: number;
}

export interface FlowParticleState {
  id: UUID;
  status: 'draft' | 'collapsed' | 'archived';
  content: string;
  summary?: string;
  context: FlowContext;
  createdAt: number;
  updatedAt: number;
}

export interface FlowParticleHistory {
  state: FlowParticleState;
  by: string;
  note?: string;
}

export interface FlowParticleSnapshot {
  id: UUID;
  history: FlowParticleHistory[];
  latest: FlowParticleState;
}

export interface MerkleLink {
  hash: string;
  parent?: string;
  context: FlowContext;
  event: FlowEvent;
  createdAt: number;
}

export interface PersonaProfile {
  id: UUID;
  name: string;
  description?: string;
  tone?: string;
  traits?: string[];
}

export interface PersonaTriangle {
  primary: string;
  secondary?: string;
  adversarial?: string;
}

export interface PersonaRecord {
  profile: PersonaProfile;
  triangle?: PersonaTriangle;
  seeds: UUID[];
}

export interface SeedMigration {
  id: UUID;
  from: string;
  to: string;
  summary: string;
  createdAt: number;
}

export interface SeedDefinition {
  id: UUID;
  name: string;
  version: string;
  description?: string;
  migrations: SeedMigration[];
  payload: Record<string, unknown>;
}

export interface ConversationMessage {
  id: UUID;
  author: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: number;
  context: FlowContext;
}

export interface ConversationThread {
  id: UUID;
  messages: ConversationMessage[];
  persona?: UUID;
  project?: UUID;
  createdAt: number;
}

export interface ProjectMetadata {
  id: UUID;
  name: string;
  description?: string;
  repository?: string;
  knowledgeBase?: string[];
  createdAt: number;
}

export interface ArtifactVersion {
  id: UUID;
  artifact: UUID;
  version: string;
  createdAt: number;
  metadata?: Record<string, unknown>;
}

export interface ArtifactRecord {
  id: UUID;
  project: UUID;
  name: string;
  description?: string;
  versions: ArtifactVersion[];
}

export interface MemoryEntry {
  id: UUID;
  scope: 'conversation' | 'project' | 'global';
  topic: string;
  embedding?: number[];
  payload: Record<string, unknown>;
  createdAt: number;
}

export interface ToolRegistration {
  id: UUID;
  name: string;
  description?: string;
  handler: (payload: Record<string, unknown>) => Promise<unknown> | unknown;
  metadata?: Record<string, unknown>;
}

export interface FlowSnapshot {
  particles: FlowParticleSnapshot[];
  personas: PersonaRecord[];
  seeds: SeedDefinition[];
  conversations: ConversationThread[];
  projects: ProjectMetadata[];
  artifacts: ArtifactRecord[];
  memories: MemoryEntry[];
}

export interface FlowLawViolation {
  code: string;
  message: string;
  context?: FlowContext;
}

export interface FlowLawResult {
  passed: boolean;
  violations: FlowLawViolation[];
}
