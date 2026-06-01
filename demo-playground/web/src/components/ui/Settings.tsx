'use client';

import { useEffect, useState } from 'react';
import {
  AI_PROVIDERS,
  defaultModelForProvider,
  type ProviderId,
} from '@/lib/ai-providers';
import { alkahestAuthHeaders, healthPath, type ServerBackend } from '@/lib/server-connection';
import { isStaticHosting } from '@/lib/hosting';

const EXECUTION_MODES = ['auto', 'wasm', 'server'] as const;

export type ExecutionMode = (typeof EXECUTION_MODES)[number];

export interface PlaygroundConfig {
  serverBackend: ServerBackend;
  serverHttpUrl: string;
  serverWsUrl: string;
  serverToken: string;
  executionMode: ExecutionMode;
  aiProvider: ProviderId;
  aiModel: string;
  aiCustomBaseUrl: string;
  aiCustomApiKey: string;
  /** When true, code cells hide CodeMirror gutter line numbers. */
  hideLineNumbers: boolean;
}

const DEFAULT_CONFIG: PlaygroundConfig = isStaticHosting
  ? {
      serverBackend: 'alkahest',
      serverHttpUrl: '',
      serverWsUrl: '',
      serverToken: '',
      executionMode: 'auto',
      aiProvider: 'anthropic',
      aiModel: 'claude-sonnet-4-6',
      aiCustomBaseUrl: '',
      aiCustomApiKey: '',
      hideLineNumbers: false,
    }
  : {
      serverBackend: 'alkahest',
      serverHttpUrl: 'http://localhost:8000',
      serverWsUrl: 'ws://localhost:8000',
      serverToken: '',
      executionMode: 'auto',
      aiProvider: 'anthropic',
      aiModel: 'claude-sonnet-4-6',
      aiCustomBaseUrl: '',
      aiCustomApiKey: '',
      hideLineNumbers: false,
    };

const STORAGE_KEY = 'alkahest-playground-config';

export function loadConfig(): PlaygroundConfig {
  if (typeof window === 'undefined') return DEFAULT_CONFIG;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_CONFIG;
    const parsed = JSON.parse(raw) as Partial<PlaygroundConfig>;
    return { ...DEFAULT_CONFIG, ...parsed };
  } catch {
    return DEFAULT_CONFIG;
  }
}

export function saveConfig(cfg: PlaygroundConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
}

interface SettingsProps {
  onClose: () => void;
  onExportNotebook?: () => void;
  onImportNotebook?: (file: File) => void | Promise<void>;
  /** Show notebook display options even when UI is in zen/recording layout */
  showNotebookOptions?: boolean;
}

export default function Settings({
  onClose,
  onExportNotebook,
  onImportNotebook,
  showNotebookOptions = true,
}: SettingsProps) {
  const [cfg, setCfg] = useState<PlaygroundConfig>(DEFAULT_CONFIG);
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'ok' | 'fail'>('idle');
  const [leanStatus, setLeanStatus] = useState<string | null>(null);
  const isCustomEndpoint = cfg.aiProvider === 'openai-compatible';

  useEffect(() => {
    setCfg(loadConfig());
  }, []);

  function update<K extends keyof PlaygroundConfig>(key: K, value: PlaygroundConfig[K]) {
    setCfg((prev) => {
      const next = { ...prev, [key]: value };
      if (key === 'serverHttpUrl') {
        next.serverWsUrl = (value as string).replace(/^https?/, (p) => (p === 'https' ? 'wss' : 'ws'));
      }
      if (key === 'aiProvider') {
        const defaultModel = defaultModelForProvider(value as string);
        if (defaultModel && (!prev.aiModel || prev.aiModel === defaultModelForProvider(prev.aiProvider))) {
          next.aiModel = defaultModel;
        }
      }
      return next;
    });
  }

  async function testConnection() {
    if (!cfg.serverHttpUrl.trim()) {
      setTestStatus('fail');
      return;
    }
    setTestStatus('testing');
    try {
      let res: Response;
      if (cfg.serverBackend === 'jupyter') {
        res = await fetch('/api/jupyter-proxy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            baseUrl: cfg.serverHttpUrl,
            token: cfg.serverToken || undefined,
            path: healthPath('jupyter'),
            method: 'GET',
          }),
          signal: AbortSignal.timeout(5000),
        });
      } else {
        res = await fetch(`${cfg.serverHttpUrl}${healthPath('alkahest')}`, {
          headers: alkahestAuthHeaders(cfg.serverToken),
          signal: AbortSignal.timeout(3000),
        });
      }
      setTestStatus(res.ok ? 'ok' : 'fail');
      if (cfg.serverBackend === 'alkahest' && res.ok) {
        try {
          const leanRes = await fetch(`${cfg.serverHttpUrl}/lean/status`, {
            headers: alkahestAuthHeaders(cfg.serverToken),
            signal: AbortSignal.timeout(5000),
          });
          if (leanRes.ok) {
            const s = (await leanRes.json()) as { available: boolean; project_dir: string };
            setLeanStatus(
              s.available
                ? 'Lean verifier ready'
                : `Lean verifier unavailable (project: ${s.project_dir})`,
            );
          } else {
            setLeanStatus(null);
          }
        } catch {
          setLeanStatus(null);
        }
      } else {
        setLeanStatus(null);
      }
    } catch {
      setTestStatus('fail');
      setLeanStatus(null);
    }
  }

  function handleSave() {
    saveConfig(cfg);
    onClose();
    window.location.reload();
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm" onClick={onClose}>
      <div
        className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-lg border border-ak-border bg-ak-bg p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-5 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold">Settings</h2>
            <p className="text-xs text-ak-muted">Ctrl+/ to toggle</p>
          </div>
          <button onClick={onClose} className="text-ak-muted hover:text-ak-fg" aria-label="Close settings">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-4">
          {/* Backend */}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ak-muted">
              Execution backend
            </h3>
            <p className="mb-3 text-xs text-ak-muted">
              Alkahest server or remote Jupyter (URL + token from jupyter server list). Token stays in this browser.
            </p>
            {isStaticHosting &&
              cfg.serverBackend === 'alkahest' &&
              /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/i.test(cfg.serverHttpUrl.trim()) && (
                <p className="mb-3 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-950">
                  This hosted playground is HTTPS and cannot call <code className="rounded bg-white/80 px-1">http://localhost</code>{' '}
                  (mixed content), even with an SSH tunnel. Use an <strong>https://</strong> backend URL, or run the playground
                  locally at <code className="rounded bg-white/80 px-1">http://localhost:3000</code> with the tunnel.
                </p>
              )}
            <label className="block text-sm mb-1">Backend type</label>
            <select
              value={cfg.serverBackend}
              onChange={(e) => update('serverBackend', e.target.value as ServerBackend)}
              className="mb-3 w-full rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ak-brand"
            >
              <option value="alkahest">Alkahest server</option>
              <option value="jupyter">Jupyter Server</option>
            </select>
            <label className="block text-sm mb-1">Server URL</label>
            <div className="mb-3 flex gap-2">
              <input
                type="url"
                value={cfg.serverHttpUrl}
                onChange={(e) => update('serverHttpUrl', e.target.value)}
                placeholder={cfg.serverBackend === 'jupyter' ? 'https://host:8888' : 'http://localhost:8000'}
                className="flex-1 rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ak-brand"
              />
              <button
                type="button"
                onClick={testConnection}
                className="rounded border border-ak-border px-3 py-1.5 text-xs hover:bg-ak-code-bg"
              >
                {testStatus === 'testing' ? '…' : testStatus === 'ok' ? '✓ OK' : testStatus === 'fail' ? '✗ fail' : 'Test'}
              </button>
            </div>
            <label className="block text-sm mb-1">Access token</label>
            <input
              type="password"
              value={cfg.serverToken}
              onChange={(e) => update('serverToken', e.target.value)}
              placeholder={cfg.serverBackend === 'jupyter' ? 'Jupyter token' : 'Optional Bearer token'}
              autoComplete="off"
              className="w-full rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ak-brand"
            />
            {leanStatus && (
              <p className="mt-2 text-xs text-ak-muted">{leanStatus}</p>
            )}
          </section>

          <section>
            <label className="block text-sm mb-1">Execution mode</label>
            <select
              value={cfg.executionMode}
              onChange={(e) => update('executionMode', e.target.value as ExecutionMode)}
              className="w-full rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ak-brand"
            >
              <option value="auto">Auto (WASM for pure Python, server for alkahest)</option>
              <option value="wasm">WASM only (Pyodide)</option>
              <option value="server">Server only</option>
            </select>
          </section>

          {/* AI */}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ak-muted">
              Agent AI
            </h3>
            <p className="mb-3 text-xs text-ak-muted">
              API keys are read from server environment variables unless you override them below
              (OpenAI-compatible only). Keys saved in settings stay in this browser only.
            </p>
            <label className="block text-sm mb-1">Provider</label>
            <select
              value={cfg.aiProvider}
              onChange={(e) => update('aiProvider', e.target.value as ProviderId)}
              className="mb-3 w-full rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ak-brand"
            >
              {AI_PROVIDERS.map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
            {isCustomEndpoint && (
              <div className="mb-3 space-y-3 rounded border border-ak-border bg-ak-code-bg/50 p-3">
                <div>
                  <label className="block text-sm mb-1">API base URL</label>
                  <input
                    type="url"
                    value={cfg.aiCustomBaseUrl}
                    onChange={(e) => update('aiCustomBaseUrl', e.target.value)}
                    placeholder="https://api.example.com/v1"
                    className="w-full rounded border border-ak-border bg-ak-bg px-3 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ak-brand"
                  />
                  <p className="mt-1 text-xs text-ak-muted">
                    Or set OPENAI_COMPATIBLE_BASE_URL in web/.env.local
                  </p>
                </div>
                <div>
                  <label className="block text-sm mb-1">API key (optional)</label>
                  <input
                    type="password"
                    value={cfg.aiCustomApiKey}
                    onChange={(e) => update('aiCustomApiKey', e.target.value)}
                    placeholder="sk-..."
                    autoComplete="off"
                    className="w-full rounded border border-ak-border bg-ak-bg px-3 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ak-brand"
                  />
                  <p className="mt-1 text-xs text-ak-muted">
                    Or set OPENAI_COMPATIBLE_API_KEY in web/.env.local
                  </p>
                </div>
              </div>
            )}
            <label className="block text-sm mb-1">Model</label>
            <input
              type="text"
              value={cfg.aiModel}
              onChange={(e) => update('aiModel', e.target.value)}
              className="w-full rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ak-brand"
              placeholder={isCustomEndpoint ? 'e.g. gpt-4o or your-model-id' : 'e.g. claude-sonnet-4-6'}
            />
          </section>

          {showNotebookOptions && (
            <section>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ak-muted">
                Notebook display
              </h3>
              <label className="flex cursor-pointer items-start gap-2 rounded border border-ak-border px-3 py-2 text-sm hover:bg-ak-code-bg">
                <input
                  type="checkbox"
                  checked={cfg.hideLineNumbers}
                  onChange={(e) => update('hideLineNumbers', e.target.checked)}
                  className="mt-0.5"
                />
                <span>
                  <span className="font-medium">Hide code line numbers</span>
                  <span className="mt-0.5 block text-xs text-ak-muted">
                    Cleaner layout for demos and recordings. Also available via{' '}
                    <code className="rounded bg-ak-code-bg px-1">?hideLineNumbers=1</code> in the URL.
                  </span>
                </span>
              </label>
            </section>
          )}

          {(onExportNotebook || onImportNotebook) && (
            <section>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ak-muted">
                Import &amp; export
              </h3>
              <div className="space-y-2">
                {onImportNotebook && (
                  <label className="flex w-full cursor-pointer items-center gap-2 rounded border border-ak-border px-3 py-2 text-sm hover:bg-ak-code-bg transition-colors">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5-5 5 5M12 15V3" />
                    </svg>
                    Upload notebook (.ipynb or .json)
                    <input
                      type="file"
                      accept=".ipynb,.json,application/json"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        e.target.value = '';
                        if (file) void onImportNotebook(file);
                      }}
                    />
                  </label>
                )}
                {onExportNotebook && (
                  <button
                    type="button"
                    onClick={() => { onExportNotebook(); onClose(); }}
                    className="flex w-full items-center gap-2 rounded border border-ak-border px-3 py-2 text-sm text-left hover:bg-ak-code-bg transition-colors"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 3v12" />
                    </svg>
                    Download as Jupyter notebook (.ipynb)
                  </button>
                )}
              </div>
            </section>
          )}
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded border border-ak-border px-4 py-1.5 text-sm hover:bg-ak-code-bg"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="rounded bg-ak-brand px-4 py-1.5 text-sm font-medium text-white hover:bg-ak-brand-dark"
          >
            Save & reload
          </button>
        </div>
      </div>
    </div>
  );
}
