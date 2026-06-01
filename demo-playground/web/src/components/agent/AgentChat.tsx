'use client';

import { useChat } from 'ai/react';
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import Output from '@/components/notebook/Output';
import type { OutputItem } from '@/lib/execution';
import { loadConfig } from '@/components/ui/Settings';
import { createSession } from '@/lib/execution';
import { connectionFromConfig } from '@/lib/server-connection';
import { agentApiKeyHelp } from '@/lib/agent-config';
import { isStaticHosting } from '@/lib/hosting';
import { useSettings } from '@/components/ui/SettingsContext';

interface AgentChatProps {
  onServerStatusChange?: (status: 'unknown' | 'online' | 'offline') => void;
}

export default function AgentChat({ onServerStatusChange }: AgentChatProps) {
  const cfg = useRef(loadConfig());
  const { openSettings } = useSettings();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'unknown' | 'online' | 'offline'>('unknown');
  const [apiKeyStatus, setApiKeyStatus] = useState<'unknown' | 'configured' | 'missing'>('unknown');
  const [apiKeyMessage, setApiKeyMessage] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Bootstrap a kernel session for the agent to use
  useEffect(() => {
    if (isStaticHosting) return;
    const conn = connectionFromConfig(cfg.current);
    if (!conn.httpUrl) {
      setServerStatus('offline');
      onServerStatusChange?.('offline');
      return;
    }
    (async () => {
      try {
        const id = await createSession(conn);
        setSessionId(id);
        setServerStatus('online');
        onServerStatusChange?.('online');
      } catch {
        setServerStatus('offline');
        onServerStatusChange?.('offline');
      }
    })();
  }, [onServerStatusChange]);

  useEffect(() => {
    if (isStaticHosting) {
      setApiKeyStatus('missing');
      setApiKeyMessage(
        'The agent needs a self-hosted playground. Run pnpm start locally and add an API key in web/.env.local or Settings (Ctrl+/).',
      );
      return;
    }

    (async () => {
      try {
        const res = await fetch('/api/agent/status', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            provider: cfg.current.aiProvider,
            customBaseUrl: cfg.current.aiCustomBaseUrl || undefined,
            customApiKey: cfg.current.aiCustomApiKey || undefined,
          }),
        });
        if (!res.ok) {
          setApiKeyStatus('missing');
          setApiKeyMessage(agentApiKeyHelp(cfg.current.aiProvider));
          return;
        }
        const data = (await res.json()) as { configured: boolean; message?: string | null };
        setApiKeyStatus(data.configured ? 'configured' : 'missing');
        setApiKeyMessage(data.configured ? null : (data.message ?? agentApiKeyHelp(cfg.current.aiProvider)));
      } catch {
        setApiKeyStatus('missing');
        setApiKeyMessage(agentApiKeyHelp(cfg.current.aiProvider));
      }
    })();
  }, []);

  const agentBlocked = isStaticHosting || apiKeyStatus === 'missing';
  const canSend = !agentBlocked && Boolean(sessionId);

  const { messages, input, handleInputChange, handleSubmit, isLoading, stop } = useChat({
    api: '/api/agent',
    body: {
      provider: cfg.current.aiProvider,
      model: cfg.current.aiModel,
      customBaseUrl: cfg.current.aiCustomBaseUrl || undefined,
      customApiKey: cfg.current.aiCustomApiKey || undefined,
      serverHttpUrl: cfg.current.serverHttpUrl,
      serverWsUrl: cfg.current.serverWsUrl,
      serverBackend: cfg.current.serverBackend,
      serverToken: cfg.current.serverToken || undefined,
      sessionId,
    },
    onError: (e) => console.error('Agent error:', e),
  });

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const examples = [
    'Simplify x + 0 with alkahest and show the Lean certificate',
    'Differentiate x³·sin(x) with alkahest and verify the proof in Lean',
    'Compare alkahest vs SymPy: integrate e^x·cos(x)',
    'Solve the system x + y = 5, x - y = 1 using alkahest',
    'Plot the derivatives of sin(x) up to order 4 with matplotlib',
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-57px)]">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-6 space-y-6">
          {messages.length === 0 && (
            <div className="space-y-8">
              <div className="text-center space-y-2">
                <h2 className="text-xl font-semibold">Alkahest AI</h2>
                <p className="text-sm text-ak-muted max-w-md mx-auto">
                  Ask me to compute symbolic math, compare libraries, or demonstrate Alkahest features.
                  I have access to a live Python kernel.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {examples.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => {
                      handleInputChange({ target: { value: ex } } as never);
                    }}
                    className="text-left rounded-lg border border-ak-border p-3 text-sm hover:bg-ak-code-bg hover:border-ak-muted/40 transition-colors"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role !== 'user' && (
                <div className="h-7 w-7 rounded-full bg-ak-brand flex items-center justify-center text-white text-xs font-bold shrink-0 mt-0.5">
                  A
                </div>
              )}

              <div className={`max-w-[85%] space-y-3 ${msg.role === 'user' ? 'items-end' : 'items-start'} flex flex-col`}>
                {/* Text content */}
                {typeof msg.content === 'string' && msg.content && (
                  <div
                    className={`rounded-lg px-4 py-2.5 text-sm ${
                      msg.role === 'user'
                        ? 'bg-ak-brand text-white'
                        : 'bg-ak-code-bg border border-ak-border'
                    }`}
                  >
                    {msg.role === 'user' ? (
                      <p>{msg.content}</p>
                    ) : (
                      <div className="prose text-sm">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                )}

                {/* Tool invocations */}
                {msg.toolInvocations?.map((inv) => (
                  <div key={inv.toolCallId} className="w-full space-y-1">
                    {/* Code the agent wrote */}
                    <div className="rounded-lg border border-ak-border overflow-hidden">
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-ak-bg border-b border-ak-border">
                        <span className="text-xs font-mono text-ak-muted">
                          {inv.toolName === 'verify_lean' ? 'verify_lean' : 'run_python'}
                        </span>
                        <span
                          className={`ml-auto text-xs px-1.5 py-0.5 rounded font-mono ${
                            inv.state === 'result'
                              ? 'bg-green-100 text-green-700'
                              : 'bg-ak-brand/10 text-ak-brand'
                          }`}
                        >
                          {inv.state === 'result' ? 'done' : 'running…'}
                        </span>
                      </div>
                      <pre className="bg-ak-code-bg px-4 py-3 text-xs font-mono overflow-x-auto whitespace-pre text-ak-fg max-h-48">
                        {inv.toolName === 'verify_lean'
                          ? ((inv.args as { source?: string }).source ?? '').slice(0, 2000) +
                            (((inv.args as { source?: string }).source?.length ?? 0) > 2000 ? '\n…' : '')
                          : (inv.args as { code?: string }).code ?? ''}
                      </pre>
                    </div>

                    {/* Output */}
                    {inv.state === 'result' && inv.toolName === 'verify_lean' && (
                      <div className="rounded-lg border border-ak-border px-4 py-3 text-xs font-mono space-y-1">
                        {(() => {
                          const r = inv.result as { ok?: boolean; stderr?: string; duration_ms?: number };
                          return (
                            <>
                              <p className={r.ok ? 'text-green-700' : 'text-red-700'}>
                                {r.ok ? `Lean verification passed (${r.duration_ms ?? 0}ms)` : 'Lean verification failed'}
                              </p>
                              {r.stderr && <pre className="text-red-600 whitespace-pre-wrap">{r.stderr}</pre>}
                            </>
                          );
                        })()}
                      </div>
                    )}
                    {inv.state === 'result' && inv.toolName !== 'verify_lean' && (
                      <div className="rounded-lg border border-ak-border overflow-hidden">
                        <Output items={(inv.result as { outputs: OutputItem[] }).outputs ?? []} />
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {msg.role === 'user' && (
                <div className="h-7 w-7 rounded-full bg-ak-muted flex items-center justify-center text-white text-xs font-bold shrink-0 mt-0.5">
                  U
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="h-7 w-7 rounded-full bg-ak-brand flex items-center justify-center text-white text-xs font-bold shrink-0">
                A
              </div>
              <div className="flex items-center gap-1 px-4 py-2.5 rounded-lg bg-ak-code-bg border border-ak-border">
                <span className="h-1.5 w-1.5 rounded-full bg-ak-muted animate-bounce [animation-delay:0ms]" />
                <span className="h-1.5 w-1.5 rounded-full bg-ak-muted animate-bounce [animation-delay:150ms]" />
                <span className="h-1.5 w-1.5 rounded-full bg-ak-muted animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar */}
      <div className="border-t border-ak-border bg-ak-bg px-4 py-3">
        {agentBlocked && apiKeyMessage && (
          <div className="mx-auto max-w-3xl mb-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
            <p>{apiKeyMessage}</p>
            {!isStaticHosting && (
              <button
                type="button"
                onClick={openSettings}
                className="mt-2 text-xs font-medium underline hover:no-underline"
              >
                Open Settings (Ctrl+/)
              </button>
            )}
          </div>
        )}
        <form
          onSubmit={handleSubmit}
          className="mx-auto max-w-3xl flex gap-2"
        >
          <input
            value={input}
            onChange={handleInputChange}
            placeholder={
              agentBlocked
                ? 'Add an API key to use the agent…'
                : 'Ask the agent to compute something with Alkahest…'
            }
            disabled={isLoading || !canSend}
            className="flex-1 rounded-lg border border-ak-border bg-ak-code-bg px-4 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-ak-brand disabled:opacity-50"
          />
          {isLoading ? (
            <button
              type="button"
              onClick={stop}
              className="rounded-lg border border-ak-border px-4 py-2.5 text-sm font-medium hover:bg-ak-code-bg"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim() || !canSend}
              className="rounded-lg bg-ak-brand px-4 py-2.5 text-sm font-medium text-white hover:bg-ak-brand-dark disabled:opacity-40 transition-colors"
            >
              Send
            </button>
          )}
        </form>
        <p className="text-center text-xs text-ak-muted mt-2">
          Provider:{' '}
          <span className="font-mono">
            {cfg.current.aiProvider} / {cfg.current.aiModel}
            {cfg.current.aiProvider === 'openai-compatible' && cfg.current.aiCustomBaseUrl
              ? ` @ ${cfg.current.aiCustomBaseUrl}`
              : ''}
          </span>
          {' · '}Kernel: <span className="font-mono">{sessionId ? sessionId.slice(0, 8) + '…' : 'none'}</span>
          {apiKeyStatus === 'missing' && !isStaticHosting && (
            <>
              {' · '}
              <span className="text-amber-700">API key required</span>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
