import { API_BASE_URL } from '../config/api';

export type ChatEventType = 'status' | 'task_start' | 'task_complete' | 'token' | 'error' | 'defect_found';

export interface ChatEvent {
  type: ChatEventType;
  content: string;
  payload: Record<string, unknown>;
}

export interface ChatMessageInput {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  message: string;
  history: ChatMessageInput[];
  planning_mode?: boolean;
}

const CHAT_STREAM_ENDPOINT = `${API_BASE_URL}/api/chat/stream`;

const parseSseChunk = (buffer: string, onEvent: (event: ChatEvent) => void): string => {
  let remaining = buffer;
  while (true) {
    const separatorIndex = remaining.indexOf('\n\n');
    if (separatorIndex === -1) break;
    const block = remaining.slice(0, separatorIndex);
    remaining = remaining.slice(separatorIndex + 2);
    if (!block.trim()) continue;
    const lines = block.split('\n');
    const dataLines = lines
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart());
    if (dataLines.length === 0) continue;
    const dataText = dataLines.join('\n');
    try {
      const parsed = JSON.parse(dataText) as ChatEvent;
      if (parsed?.type) {
        onEvent(parsed);
      }
    } catch {
      continue;
    }
  }
  return remaining;
};

export const chatService = {
  streamChat: async (
    body: ChatRequest,
    onEvent: (event: ChatEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const response = await fetch(CHAT_STREAM_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok || !response.body) {
      const errorText = await response.text().catch(() => '');
      throw new Error(errorText || 'Failed to start chat stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSseChunk(buffer, onEvent);
    }
  },
};
