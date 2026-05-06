import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { MessageCircle, Send, X, Loader2, ChevronDown, ChevronRight, CheckCircle2, AlertCircle, Terminal, Info, User, Bot, AlertTriangle } from 'lucide-react';
import { chatService, type ChatEvent, type ChatMessageInput } from '../services/chatService';
import type { Defect } from '../services/agentService';

type ChatRole = 'user' | 'assistant' | 'event' | 'defect';

interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  meta?: Record<string, unknown>;
  turnId?: string;
}

interface ChatPanelProps {
  fileName?: string | null;
  onClose: () => void;
  onDefectSelect?: (defect: Defect | null) => void;
}

const createId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
const DEFAULT_CHAT_WIDTH = 360;
const MIN_CHAT_WIDTH = 280;
const MAX_CHAT_WIDTH = 520;

const MarkdownRenderer = ({ content }: { content: string }) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => <h1 className="text-base font-semibold text-white mt-3 mb-2 first:mt-0 border-b border-white/10 pb-1">{children}</h1>,
        h2: ({ children }) => <h2 className="text-sm font-semibold text-white mt-3 mb-1.5 first:mt-0">{children}</h2>,
        h3: ({ children }) => <h3 className="text-xs font-semibold text-neutral-100 mt-2 mb-1 first:mt-0">{children}</h3>,
        p: ({ children }) => <p className="text-[13px] text-neutral-300 leading-relaxed mb-2 last:mb-0 whitespace-pre-wrap">{children}</p>,
        ul: ({ children }) => <ul className="list-disc list-outside pl-5 space-y-1 my-2 text-[13px] text-neutral-300 leading-relaxed">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-outside pl-5 space-y-1 my-2 text-[13px] text-neutral-300 leading-relaxed">{children}</ol>,
        li: ({ children }) => <li className="pl-1 marker:text-neutral-500">{children}</li>,
        strong: ({ children }) => <strong className="text-white font-semibold">{children}</strong>,
        a: ({ children, href }) => (
          <a className="text-blue-400 hover:text-blue-300 hover:underline" href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
        code: ({ children, className }) => {
          const isBlock = typeof className === 'string' && className.length > 0;
          return isBlock ? (
            <div className="my-2 rounded-lg overflow-hidden border border-white/10 bg-[#1e1e1e]">
              <div className="px-2 py-1 bg-white/5 border-b border-white/5 flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-red-500/20"></div>
                <div className="w-2 h-2 rounded-full bg-yellow-500/20"></div>
                <div className="w-2 h-2 rounded-full bg-green-500/20"></div>
              </div>
              <pre className="p-2 overflow-x-auto text-[11px] text-neutral-200 font-mono custom-scrollbar">
                <code>{children}</code>
              </pre>
            </div>
          ) : (
            <code className="bg-white/10 px-1 py-0.5 rounded text-[11px] font-mono text-blue-200">{children}</code>
          );
        },
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-blue-500/50 pl-2 italic text-neutral-400 my-2">{children}</blockquote>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto border border-white/10 rounded-lg my-2 bg-white/[0.02]">
            <table className="w-full text-left border-collapse text-[11px]">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-white/5 text-neutral-100 font-medium border-b border-white/10">{children}</thead>,
        tbody: ({ children }) => <tbody className="divide-y divide-white/5">{children}</tbody>,
        tr: ({ children }) => <tr className="hover:bg-white/[0.02] transition-colors">{children}</tr>,
        th: ({ children }) => <th className="px-2 py-1.5 font-semibold">{children}</th>,
        td: ({ children }) => <td className="px-2 py-1.5 text-neutral-300 whitespace-pre-wrap">{children}</td>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

const ThinkingSection = ({ events }: { events: ChatMessage[] }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (events.length === 0) return null;

  return (
    <div className="rounded-2xl border border-white/10 bg-[#121a23] overflow-hidden shadow-lg">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2 text-[11px] font-semibold !text-white hover:text-white bg-[#17212b] transition-colors"
        style={{ color: 'white' }}
      >
        <span className="flex items-center gap-2">
          {isOpen ? <ChevronDown className="w-3 h-3 text-white" /> : <ChevronRight className="w-3 h-3 text-white" />}
          Process ({events.length})
        </span>
      </button>

      {isOpen && (
        <div className="px-3 py-2.5 space-y-2">
          {events.map((msg, index) => {
            let Icon = Info;
            let colorClass = 'text-neutral-400';
            const isTaskStart = msg.content.startsWith('Task started');

            if (msg.content.startsWith('Error')) {
              Icon = AlertCircle;
              colorClass = 'text-red-400';
            } else if (msg.content.startsWith('Task complete')) {
              Icon = CheckCircle2;
              colorClass = 'text-emerald-400';
            } else if (isTaskStart) {
              Icon = Terminal;
              colorClass = 'text-blue-400';
            }

            return (
              <div 
                key={msg.id} 
                className={`flex items-start gap-2 text-[10px] text-neutral-300 font-mono ${isTaskStart && index > 0 ? 'mt-6 pt-6 border-t border-white/10' : ''}`}
              >
                <div className="mt-0.5 shrink-0">
                  <Icon className={`w-3 h-3 ${colorClass}`} />
                </div>
                <div className="break-words whitespace-pre-wrap leading-relaxed">{msg.content}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

const DefectSummaryCard = ({ defects, onSelect }: { defects: Defect[]; onSelect?: (defect: Defect) => void }) => {
  const [expanded, setExpanded] = useState(true);

  if (!defects || defects.length === 0) return null;

  const stats = defects.reduce(
    (acc, d) => {
      const severity = String(d.severity).toLowerCase();
      if (severity === 'high') acc.high++;
      else if (severity === 'medium') acc.medium++;
      else acc.low++;
      return acc;
    },
    { high: 0, medium: 0, low: 0 }
  );

  const getSeverityColor = (severity: string) => {
    const s = severity.toLowerCase();
    if (s === 'high') return { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', dot: 'bg-red-500' };
    if (s === 'medium') return { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', dot: 'bg-yellow-500' };
    return { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', dot: 'bg-blue-500' };
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-[#121a23] overflow-hidden shadow-lg">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 text-[11px] font-semibold bg-[#17212b] hover:bg-[#1e2a36] transition-colors"
      >
        <span className="flex items-center gap-2 text-red-400">
          {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          <AlertTriangle className="w-3.5 h-3.5 text-orange-400" />
          Defects Detected ({defects.length})
        </span>
        <div className="flex items-center gap-2">
          {stats.high > 0 && <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400">{stats.high} High</span>}
          {stats.medium > 0 && <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400">{stats.medium} Med</span>}
          {stats.low > 0 && <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400">{stats.low} Low</span>}
        </div>
      </button>

      {expanded && (
        <div className="p-3 space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
          {defects.map((defect, index) => {
            const colors = getSeverityColor(String(defect.severity));
            return (
              <button
                key={defect.id ?? index}
                className={`p-2.5 rounded-xl ${colors.bg} ${colors.border} border transition-colors hover:brightness-110 text-left w-full ${onSelect ? 'cursor-pointer' : ''}`}
                onClick={() => onSelect?.(defect)}
              >
                <div className="flex items-start gap-2">
                  <div className={`w-1.5 h-1.5 rounded-full ${colors.dot} mt-1.5 shrink-0`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span className={`text-xs font-semibold ${colors.text}`}>
                        {defect.type || `Defect #${defect.id}`}
                      </span>
                      <span className={`text-[9px] uppercase font-bold ${colors.text} opacity-70`}>
                        {defect.severity}
                      </span>
                    </div>
                    <p className="text-[11px] text-neutral-300 leading-relaxed line-clamp-2">
                      {defect.description}
                    </p>
                    {(defect.layout_id !== undefined || defect.chunk_id !== undefined) && (
                      <div className="mt-1.5 text-[9px] text-neutral-500 font-mono">
                        {defect.layout_id !== undefined ? `Layout ${defect.layout_id}` : ''}
                        {defect.layout_id !== undefined && defect.chunk_id !== undefined ? ' • ' : ''}
                        {defect.chunk_id !== undefined ? `Chunk ${defect.chunk_id}` : ''}
                      </div>
                    )}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default function ChatPanel({ fileName, onClose, onDefectSelect }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [turnOrder, setTurnOrder] = useState<string[]>([]);
  const [turnStatus, setTurnStatus] = useState<Record<string, string>>({});
  const [turnDefects, setTurnDefects] = useState<Record<string, Defect[]>>({});
  const [chatWidth, setChatWidth] = useState(DEFAULT_CHAT_WIDTH);
  const abortRef = useRef<AbortController | null>(null);
  const streamMessageIdRef = useRef<Record<string, string>>({});
  const currentTurnIdRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const isResizingRef = useRef(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(DEFAULT_CHAT_WIDTH);
  const widthRef = useRef(DEFAULT_CHAT_WIDTH);

  useEffect(() => {
    const stored = Number(localStorage.getItem('chatPanelWidth'));
    if (Number.isFinite(stored) && stored >= MIN_CHAT_WIDTH && stored <= MAX_CHAT_WIDTH) {
      setChatWidth(stored);
      widthRef.current = stored;
    }
  }, []);

  useEffect(() => {
    widthRef.current = chatWidth;
  }, [chatWidth]);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!isResizingRef.current) return;
      const delta = startXRef.current - event.clientX;
      const nextWidth = Math.min(MAX_CHAT_WIDTH, Math.max(MIN_CHAT_WIDTH, startWidthRef.current + delta));
      widthRef.current = nextWidth;
      setChatWidth(nextWidth);
    };
    const handleMouseUp = () => {
      if (!isResizingRef.current) return;
      isResizingRef.current = false;
      localStorage.setItem('chatPanelWidth', String(widthRef.current));
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const handleResizeStart = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    isResizingRef.current = true;
    startXRef.current = event.clientX;
    startWidthRef.current = widthRef.current;
  }, []);

  const history = useMemo<ChatMessageInput[]>(
    () => {
      const turnMap = new Map<string, { user?: ChatMessage; assistant?: ChatMessage }>();
      messages.forEach((msg) => {
        if (!msg.turnId) return;
        const entry = turnMap.get(msg.turnId) ?? {};
        if (msg.role === 'user') entry.user = msg;
        if (msg.role === 'assistant') entry.assistant = msg;
        turnMap.set(msg.turnId, entry);
      });
      const ordered: ChatMessageInput[] = [];
      if (fileName) {
        ordered.push({ role: 'system', content: `Active file: ${fileName}` });
      }
      turnOrder.forEach((turnId) => {
        const entry = turnMap.get(turnId);
        if (!entry) return;
        if (entry.user) ordered.push({ role: 'user', content: entry.user.content });
        if (entry.assistant) ordered.push({ role: 'assistant', content: entry.assistant.content });
      });
      return ordered;
    },
    [messages, turnOrder, fileName],
  );

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, turnOrder, isStreaming, turnStatus]);

  const ensureActiveTurn = useCallback(() => {
    const current = currentTurnIdRef.current;
    if (current) return current;
    const nextId = createId();
    currentTurnIdRef.current = nextId;
    setTurnOrder((prev) => [...prev, nextId]);
    return nextId;
  }, []);

  const handleEvent = useCallback((event: ChatEvent) => {
    if (event.type === 'token') {
      const turnId = ensureActiveTurn();
      setMessages((prev) => {
        const lastId = streamMessageIdRef.current[turnId];
        if (!lastId) {
          const nextId = createId();
          streamMessageIdRef.current[turnId] = nextId;
          return [...prev, { id: nextId, role: 'assistant', content: event.content, turnId }];
        }
        return prev.map((msg) =>
          msg.id === lastId ? { ...msg, content: msg.content + event.content } : msg,
        );
      });
      return;
    }

    if (event.type === 'status') {
      const turnId = ensureActiveTurn();
      setTurnStatus((prev) => ({ ...prev, [turnId]: event.content || '' }));
    }

    if (event.type === 'task_start' || event.type === 'task_complete' || event.type === 'error') {
      const turnId = ensureActiveTurn();
      const label =
        event.type === 'task_start'
          ? 'Task started'
          : event.type === 'task_complete'
            ? 'Task complete'
            : 'Error';
      const toolName = typeof event.payload?.tool === 'string' ? event.payload.tool : '';
      const summary = typeof event.payload?.summary === 'string' ? event.payload.summary : '';
      const detail = event.content || (event.type === 'task_start' ? toolName : summary);
      const content = detail ? `${label}: ${detail}`.trim() : label;
      setMessages((prev) => [
        ...prev,
        { id: createId(), role: 'event', content, meta: event.payload, turnId },
      ]);
    }

    if (event.type === 'defect_found') {
      const turnId = ensureActiveTurn();
      const defects = (event.payload?.defects as Defect[]) || [];
      if (defects.length > 0) {
        setTurnDefects((prev) => ({
          ...prev,
          [turnId]: [...(prev[turnId] || []), ...defects],
        }));
      }
    }
  }, [ensureActiveTurn]);

  const finalizeStream = useCallback(() => {
    setIsStreaming(false);
    const currentTurnId = currentTurnIdRef.current;
    if (currentTurnId) {
      delete streamMessageIdRef.current[currentTurnId];
    }
    currentTurnIdRef.current = null;
    abortRef.current = null;
  }, []);

  const sendMessage = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;
    const turnId = createId();
    currentTurnIdRef.current = turnId;
    setTurnOrder((prev) => [...prev, turnId]);
    setInput('');
    setTurnStatus((prev) => {
      const next = { ...prev };
      delete next[turnId];
      return next;
    });
    setMessages((prev) => [...prev, { id: createId(), role: 'user', content: trimmed, turnId }]);
    setIsStreaming(true);

    const abortController = new AbortController();
    abortRef.current = abortController;

    try {
      await chatService.streamChat(
        {
          message: trimmed,
          history,
        },
        handleEvent,
        abortController.signal,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Chat stream failed';
      const fallbackTurnId = currentTurnIdRef.current ?? createId();
      if (!currentTurnIdRef.current) {
        currentTurnIdRef.current = fallbackTurnId;
        setTurnOrder((prev) => [...prev, fallbackTurnId]);
      }
      setMessages((prev) => [...prev, { id: createId(), role: 'event', content: message, turnId: fallbackTurnId }]);
    } finally {
      finalizeStream();
    }
  }, [handleEvent, history, input, isStreaming, finalizeStream]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    },
    [sendMessage],
  );

  return (
    <div
      className="relative h-full bg-[#0b1118] border-l border-white/10 flex flex-col z-20 transition-all duration-300"
      style={{ width: chatWidth }}
    >
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-ew-resize bg-gradient-to-r from-blue-500/20 via-blue-500/5 to-transparent"
        onMouseDown={handleResizeStart}
      />
      <div className="m-2 rounded-3xl border border-white/10 bg-gradient-to-b from-[#111a23] via-[#0f161f] to-[#0b1118] shadow-2xl overflow-hidden flex flex-col h-full">
        <div className="px-4 py-3 border-b border-white/10 bg-[#0f161f]">
          <div className="grid grid-cols-3 items-center">
            <div className="text-[11px] font-semibold text-neutral-300 flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <MessageCircle className="w-3.5 h-3.5 text-blue-400" />
              </div>
              AI Assistant
            </div>
            <div className="text-center text-[11px] font-semibold text-white tracking-wider uppercase">Chat</div>
            <div className="flex justify-end">
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-white/10 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden">
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4 custom-scrollbar">
            {turnOrder.length === 0 && messages.length === 0 && (
              <div className="text-[11px] text-neutral-500 text-center mt-6">
                Ask about {fileName ? fileName : 'your drawing'} or run targeted checks.
              </div>
            )}
            {turnOrder.map((turnId) => {
              const turnMessages = messages.filter((msg) => msg.turnId === turnId);
              const userMessage = turnMessages.find((msg) => msg.role === 'user');
              const assistantMessage = turnMessages.find((msg) => msg.role === 'assistant');
              const eventMessages = turnMessages.filter((msg) => msg.role === 'event');
              const statusText = turnStatus[turnId];
              const defects = turnDefects[turnId] || [];
              return (
                <div key={turnId} className="space-y-3">
                  {userMessage && (
                    <div className="flex flex-col items-end gap-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] uppercase tracking-widest text-neutral-500 font-semibold">You</span>
                        <div className="p-1.5 rounded-full bg-[#202a36] border border-white/10 shadow-sm">
                          <User className="w-3 h-3 text-blue-300" />
                        </div>
                      </div>
                      <div className="max-w-[85%] px-3 py-2 rounded-2xl bg-[#1b2430] border border-white/10 text-[12px] text-neutral-100 whitespace-pre-wrap shadow-md">
                        {userMessage.content}
                      </div>
                    </div>
                  )}
                  {(statusText || eventMessages.length > 0) && (
                    <div className="space-y-2">
                      {statusText && (
                        <div className="text-[10px] text-blue-300 flex items-center gap-1.5 opacity-80">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          {statusText}
                        </div>
                      )}
                      <ThinkingSection events={eventMessages} />
                    </div>
                  )}
                  {defects.length > 0 && <DefectSummaryCard defects={defects} onSelect={onDefectSelect ? (defect) => onDefectSelect(defect) : undefined} />}
                  {assistantMessage && (
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-2">
                        <div className="p-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 shadow-sm">
                          <Bot className="w-3 h-3 text-emerald-300" />
                        </div>
                        <span className="text-[10px] uppercase tracking-widest text-neutral-400 font-medium">AI Assistant</span>
                      </div>
                      <div className="max-w-[95%] px-3 py-2 rounded-2xl bg-[#121b24] border border-white/10 text-[12px] text-neutral-200 shadow-md">
                        <MarkdownRenderer content={assistantMessage.content} />
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="border-t border-white/10 px-4 py-3 bg-[#0f161f]">
          <div className="relative bg-[#0b1118] border border-white/10 rounded-2xl focus-within:border-blue-500/40 focus-within:bg-[#101824] transition-all">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question, or use @filename..."
              className="w-full pl-4 pr-11 py-3.5 rounded-2xl bg-transparent text-[12px] text-white placeholder:text-neutral-500 focus:outline-none resize-none custom-scrollbar leading-relaxed"
              style={{ color: 'white' }}
              disabled={isStreaming}
              rows={Math.min(Math.max(2, input.split('\n').length), 8)}
            />
            <button
              onClick={sendMessage}
              className="absolute right-2 bottom-2 p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/30 transition-all disabled:opacity-40 disabled:scale-95"
              disabled={isStreaming || !input.trim()}
            >
              {isStreaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
