import crypto from 'crypto';
import { UUID } from '../types';

export const now = (): number => Date.now();

export const randomId = (): UUID => crypto.randomUUID();

export function chunkText(text: string, size: number): string[] {
  if (size <= 0) return [text];
  const normalized = text.trim();
  const result: string[] = [];
  for (let i = 0; i < normalized.length; i += size) {
    result.push(normalized.slice(i, i + size));
  }
  return result;
}

export function shallowMerge<T extends object, U extends object>(base: T, incoming: U): T & U {
  return { ...base, ...incoming } as T & U;
}

export function ensure<T>(value: T | undefined | null, message: string): T {
  if (value === undefined || value === null) {
    throw new Error(message);
  }
  return value;
}

export function hashPayload(payload: unknown): string {
  const buffer = Buffer.from(JSON.stringify(payload));
  return crypto.createHash('sha256').update(buffer).digest('hex');
}
