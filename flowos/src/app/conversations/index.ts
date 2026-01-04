import { ConversationMessage, ConversationThread, FlowContext } from '../../types';
import { MemoryStorage } from '../../storage';
import { now, randomId } from '../../utils';

export class ConversationManager {
  constructor(private readonly storage: MemoryStorage) {}

  startConversation(context: FlowContext): ConversationThread {
    const thread: ConversationThread = {
      id: randomId(),
      messages: [],
      persona: context.persona,
      project: context.project,
      createdAt: now(),
    };
    return this.storage.upsertConversation(thread);
  }

  appendMessage(threadId: string, author: ConversationMessage['author'], content: string, context: FlowContext): ConversationThread {
    const thread = this.storage.getConversation(threadId);
    if (!thread) {
      throw new Error(`Conversation ${threadId} not found`);
    }

    const message: ConversationMessage = {
      id: randomId(),
      author,
      content,
      createdAt: now(),
      context,
    };

    const updated: ConversationThread = { ...thread, messages: [...thread.messages, message] };
    return this.storage.upsertConversation(updated);
  }

  list(): ConversationThread[] {
    return this.storage.listConversations();
  }
}
