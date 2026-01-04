import { FlowOS } from './src';

const flow = new FlowOS();
const context = flow.createContext({ persona: 'demo-persona' });

const particle = flow.particles.createParticle('Hello FlowOS', context, 'intro message');
flow.particles.collapseParticle(particle.id, 'demo');

const conversation = flow.conversations.startConversation(context);
flow.conversations.appendMessage(conversation.id, 'user', 'Hello there', context);

const project = flow.projects.registerProject('FlowOS Sandbox', 'Playground for FlowOS runtime');
const artifact = flow.artifacts.registerArtifact(project.id, 'Transcript');
flow.artifacts.addVersion(artifact.id, 'v1');

flow.memory.remember('project', 'first-run', { project: project.name, conversation: conversation.id });

console.log('Flow snapshot:', JSON.stringify(flow.snapshot(), null, 2));
console.log('Chain digest:', flow.chain.digest());
console.log('FlowLaw:', flow.enforce());
