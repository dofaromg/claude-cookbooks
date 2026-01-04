import { ToolRegistration } from '../../types';
import { randomId } from '../../utils';

export class ToolRegistry {
  private tools = new Map<string, ToolRegistration>();

  registerTool(tool: Omit<ToolRegistration, 'id'> & { id?: string }): ToolRegistration {
    const record: ToolRegistration = { ...tool, id: tool.id ?? randomId() };
    this.tools.set(record.id, record);
    return record;
  }

  invokeTool(id: string, payload: Record<string, unknown>): Promise<unknown> {
    const tool = this.tools.get(id);
    if (!tool) {
      return Promise.reject(new Error(`Tool ${id} not found`));
    }
    return Promise.resolve(tool.handler(payload));
  }

  listTools(): ToolRegistration[] {
    return [...this.tools.values()];
  }
}
