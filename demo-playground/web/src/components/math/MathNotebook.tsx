'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { v4 as uuid } from 'uuid';
import { loadConfig } from '@/components/ui/Settings';
import { useSettings } from '@/components/ui/SettingsContext';
import { connectionFromConfig, type ServerConnection } from '@/lib/server-connection';
import {
  interpretMath,
  isWebKitOnly,
  LOCAL_MATH_URL,
  LOCAL_SETUP_COMMANDS,
  localConnection,
  loopbackPermission,
  pageIsLoopback,
  probeLocalMath,
  probeMathBackend,
  runMath,
  type MathBackendState,
  type MathDefinition,
  type MathResult,
} from '@/lib/math-api';
import { MATH_MORE_EXAMPLES, MATH_PALETTE, MATH_STARTER_CELLS, type PaletteItem } from '@/lib/math-starter';
import { PRECOMPUTED_VERSION, precomputedResults } from '@/lib/math-precomputed';
import MathCell, { type MathCellData } from './MathCell';

const STORAGE_KEY = 'alkahest-playground-math';
/** Set once a local `python -m alkahest.playground` has answered in this browser. */
const LOCAL_SEEN_KEY = 'alkahest-playground-math-local';
const PREVIEW_DEBOUNCE_MS = 180;
const FOCUS_PROBE_MIN_GAP_MS = 1500;

function rememberedLocal(): boolean {
  try {
    return localStorage.getItem(LOCAL_SEEN_KEY) === '1';
  } catch {
    return false;
  }
}

function rememberLocal() {
  try {
    localStorage.setItem(LOCAL_SEEN_KEY, '1');
  } catch {
    /* ignore */
  }
}

/** Fill in precomputed results for cells that have none (static site, no backend). */
function withPrecomputed(cells: MathCellData[]): MathCellData[] {
  const pre = precomputedResults(cells.map((c) => c.input));
  return cells.map((c, i) => {
    const r = pre[i];
    if (!r || (c.result && !c.precomputed)) return c;
    return { ...c, result: r, ranInput: c.input, status: 'done', precomputed: true };
  });
}

function newCell(input = ''): MathCellData {
  return { id: uuid(), input, status: 'idle' };
}

function loadInputs(): string[] | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed) && parsed.every((s) => typeof s === 'string') && parsed.length > 0) {
      return parsed as string[];
    }
  } catch {
    /* storage unavailable or corrupt: fall back to the starter cells */
  }
  return null;
}

function saveInputs(cells: MathCellData[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cells.map((c) => c.input)));
  } catch {
    /* ignore */
  }
}

/** Definitions made by the cells before `index`, last one per name winning. */
function definitionsBefore(cells: MathCellData[], index: number): MathDefinition[] {
  const byName = new Map<string, MathDefinition>();
  for (const c of cells.slice(0, index)) {
    const d = c.result?.ok && c.ranInput === c.input ? c.result.definition : undefined;
    if (d) {
      byName.delete(d.name);
      byName.set(d.name, d);
    }
  }
  return [...byName.values()];
}

interface MathNotebookProps {
  onServerStatusChange?: (status: 'unknown' | 'online' | 'offline', label?: string) => void;
  onReady?: () => void;
}

interface ProbeOptions {
  /** From a click: may show the browser's permission prompt. */
  userInitiated?: boolean;
  /** Background re-check: do not flash the "checking" state. */
  quiet?: boolean;
}

export default function MathNotebook({ onServerStatusChange, onReady }: MathNotebookProps) {
  const { openSettings } = useSettings();
  const [cells, setCells] = useState<MathCellData[]>(() =>
    withPrecomputed((loadInputs() ?? MATH_STARTER_CELLS).map((s) => newCell(s))),
  );
  const cellsRef = useRef(cells);
  cellsRef.current = cells;
  const [backend, setBackend] = useState<MathBackendState>({ status: 'checking' });
  const connRef = useRef<ServerConnection | null>(null);
  const inputRefs = useRef(new Map<string, HTMLInputElement>());
  const activeRef = useRef<{ id: string; start: number; end: number } | null>(null);
  const previewTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const autoRan = useRef(false);
  const lastProbe = useRef(0);
  const probing = useRef(false);
  const [runningAll, setRunningAll] = useState(false);

  const online = backend.status === 'online';

  useEffect(() => saveInputs(cells), [cells]);

  const patch = useCallback((id: string, fn: (c: MathCellData) => MathCellData) => {
    setCells((prev) => prev.map((c) => (c.id === id ? fn(c) : c)));
  }, []);

  const focusCell = useCallback((id: string, caret?: 'end') => {
    requestAnimationFrame(() => {
      const el = inputRefs.current.get(id);
      if (!el) return;
      el.focus();
      if (caret === 'end') el.setSelectionRange(el.value.length, el.value.length);
    });
  }, []);

  // ── backend ────────────────────────────────────────────────────────────────
  // With a server URL in settings, use it. Otherwise look for a local
  // `python -m alkahest.playground` on 127.0.0.1:8765. From a public page,
  // Chrome/Edge and Firefox ask the user before a site may reach this device,
  // so we only look without a click when that cannot prompt: the permission is
  // already granted, this browser has connected before, or the page itself is
  // served locally. Everyone else sees the setup card and its "Check again".
  const probe = useCallback(
    async ({ userInitiated = false, quiet = false }: ProbeOptions = {}): Promise<MathBackendState> => {
      probing.current = true;
      lastProbe.current = Date.now();
      if (!quiet) {
        setBackend({ status: 'checking' });
        onServerStatusChange?.('unknown');
      }
      let state: MathBackendState;
      const conn = connectionFromConfig(loadConfig());
      if (conn.httpUrl) {
        connRef.current = conn;
        state = await probeMathBackend(conn);
      } else {
        let permission = await loopbackPermission();
        const attempt =
          userInitiated ||
          (permission !== 'denied' && (permission === 'granted' || pageIsLoopback() || rememberedLocal()));
        const found = attempt ? await probeLocalMath(userInitiated ? 8000 : 1500) : null;
        if (found) {
          connRef.current = localConnection();
          rememberLocal();
          state = { status: 'online', local: true, version: found.version };
        } else {
          connRef.current = null;
          if (attempt) permission = await loopbackPermission();
          state = { status: 'none', tried: attempt, permission };
        }
      }
      probing.current = false;
      setBackend(state);
      onServerStatusChange?.(
        state.status === 'online' ? 'online' : 'offline',
        state.status === 'online' && state.local ? 'Computing on this machine' : undefined,
      );
      return state;
    },
    [onServerStatusChange],
  );

  // ── preview ────────────────────────────────────────────────────────────────
  const schedulePreview = useCallback((id: string) => {
    const timers = previewTimers.current;
    const t = timers.get(id);
    if (t) clearTimeout(t);
    timers.set(
      id,
      setTimeout(async () => {
        timers.delete(id);
        const conn = connRef.current;
        const list = cellsRef.current;
        const index = list.findIndex((c) => c.id === id);
        if (!conn || index < 0) return;
        const input = list[index].input;
        if (!input.trim()) {
          patch(id, (c) => ({ ...c, preview: null }));
          return;
        }
        try {
          const res = await interpretMath(conn, input, definitionsBefore(list, index));
          patch(id, (c) => (c.input === input ? { ...c, preview: res } : c));
        } catch {
          /* preview is best-effort */
        }
      }, PREVIEW_DEBOUNCE_MS),
    );
  }, [patch]);

  // ── running ────────────────────────────────────────────────────────────────
  const runOne = useCallback(
    async (id: string, list: MathCellData[]): Promise<MathResult | null> => {
      const conn = connRef.current;
      const index = list.findIndex((c) => c.id === id);
      if (!conn || index < 0) return null;
      const input = list[index].input;
      if (!input.trim()) return null;
      patch(id, (c) => ({ ...c, status: 'running' }));
      let result: MathResult;
      try {
        result = await runMath(conn, input, definitionsBefore(list, index));
      } catch (e) {
        const local = conn.httpUrl === LOCAL_MATH_URL;
        result = {
          ok: false,
          error_code: 'NETWORK',
          error_message: local
            ? 'The local Alkahest server stopped answering. Start it again with python -m alkahest.playground.'
            : `Could not reach the backend (${e instanceof Error ? e.message : String(e)}).`,
          error_source: 'playground',
        };
        void probe({ quiet: true });
      }
      patch(id, (c) => ({ ...c, status: 'done', result, ranInput: input, precomputed: false }));
      return result;
    },
    [patch, probe],
  );

  const runAll = useCallback(async () => {
    if (!connRef.current) return;
    setRunningAll(true);
    // Work on a local copy so each cell sees the definitions made just above it.
    let list = cellsRef.current.map((c) => ({ ...c }));
    for (const cell of list) {
      if (!cell.input.trim()) continue;
      const result = await runOne(cell.id, list);
      if (result) {
        list = list.map((c) => (c.id === cell.id ? { ...c, result, ranInput: c.input } : c));
      }
    }
    setRunningAll(false);
  }, [runOne]);

  /** Probe, and the first time a backend answers, replace precomputed results with live ones. */
  const connect = useCallback(
    async (opts: ProbeOptions = {}) => {
      const state = await probe(opts);
      if (state.status === 'online' && !autoRan.current) {
        autoRan.current = true;
        await runAll();
      }
      return state;
    },
    [probe, runAll],
  );

  useEffect(() => {
    let cancelled = false;
    void connect().then(() => {
      if (!cancelled) onReady?.();
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-check when the window regains focus (e.g. after starting the server in a terminal).
  useEffect(() => {
    const onFocus = () => {
      if (document.visibilityState === 'hidden' || probing.current) return;
      if (Date.now() - lastProbe.current < FOCUS_PROBE_MIN_GAP_MS) return;
      void connect({ quiet: true });
    };
    window.addEventListener('focus', onFocus);
    document.addEventListener('visibilitychange', onFocus);
    return () => {
      window.removeEventListener('focus', onFocus);
      document.removeEventListener('visibilitychange', onFocus);
    };
  }, [connect]);

  const runAndAdvance = useCallback(
    async (id: string) => {
      const list = cellsRef.current;
      const index = list.findIndex((c) => c.id === id);
      if (index < 0) return;
      let nextId: string;
      if (index === list.length - 1) {
        const added = newCell();
        nextId = added.id;
        setCells((prev) => [...prev, added]);
      } else {
        nextId = list[index + 1].id;
      }
      focusCell(nextId, 'end');
      if (online) await runOne(id, list);
    },
    [focusCell, online, runOne],
  );

  // ── editing ────────────────────────────────────────────────────────────────
  const setInput = (id: string, value: string) => {
    patch(id, (c) => ({ ...c, input: value }));
    if (online) schedulePreview(id);
  };

  const insertSnippet = (item: PaletteItem) => {
    let target = activeRef.current;
    const list = cellsRef.current;
    if (!target || !list.some((c) => c.id === target?.id)) {
      const last = list[list.length - 1];
      target = { id: last.id, start: last.input.length, end: last.input.length };
    }
    const cell = list.find((c) => c.id === target!.id)!;
    const { start, end } = target;
    const selected = cell.input.slice(start, end);
    const after = item.after ?? '';
    const value = cell.input.slice(0, start) + item.before + selected + after + cell.input.slice(end);
    const caret = start + item.before.length + selected.length;
    setInput(cell.id, value);
    activeRef.current = { id: cell.id, start: caret, end: caret };
    requestAnimationFrame(() => {
      const el = inputRefs.current.get(cell.id);
      if (!el) return;
      el.focus();
      el.setSelectionRange(caret, caret);
    });
  };

  const addCell = (input = '', focus = true) => {
    const c = newCell(input);
    setCells((prev) => [...prev, c]);
    if (focus) focusCell(c.id, 'end');
    return c;
  };

  const tryExample = async (example: string) => {
    const list = cellsRef.current;
    const last = list[list.length - 1];
    let target: MathCellData;
    if (last && !last.input.trim()) {
      target = { ...last, input: example };
    } else {
      target = newCell(example);
    }
    const next = [...list.filter((c) => c.id !== target.id), target];
    if (!online) {
      // No backend: show the precomputed answer if there is one.
      const filled = withPrecomputed(next);
      target = filled[filled.length - 1];
    }
    if (last && !last.input.trim()) {
      const t = target;
      patch(last.id, () => t);
    } else {
      setCells((prev) => [...prev, target]);
    }
    focusCell(target.id, 'end');
    if (online) await runOne(target.id, next);
  };

  const deleteCell = (id: string) => {
    const list = cellsRef.current;
    const index = list.findIndex((c) => c.id === id);
    if (list.length === 1) {
      setCells([newCell()]);
      return;
    }
    setCells((prev) => prev.filter((c) => c.id !== id));
    const neighbour = list[index - 1] ?? list[index + 1];
    if (neighbour) focusCell(neighbour.id, 'end');
  };

  const onKeyDown = (id: string) => (e: React.KeyboardEvent<HTMLInputElement>) => {
    const list = cellsRef.current;
    const index = list.findIndex((c) => c.id === id);
    if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void runAndAdvance(id);
    } else if (e.key === 'ArrowUp' && index > 0) {
      e.preventDefault();
      focusCell(list[index - 1].id, 'end');
    } else if (e.key === 'ArrowDown' && index < list.length - 1) {
      e.preventDefault();
      focusCell(list[index + 1].id, 'end');
    } else if (e.key === 'Backspace' && !list[index].input && list.length > 1) {
      e.preventDefault();
      deleteCell(id);
    } else if (e.key === 'Escape') {
      (e.target as HTMLInputElement).blur();
    }
  };

  const resetExamples = () => {
    if (!window.confirm('Replace all cells with the starter examples?')) return;
    const starter = MATH_STARTER_CELLS.map((s) => newCell(s));
    const fresh = online ? starter : withPrecomputed(starter);
    setCells(fresh);
    cellsRef.current = fresh;
    if (online) void runAll();
  };

  return (
    <div className="mx-auto max-w-3xl px-4 pb-[40vh] pt-8 sm:pt-10">
      <header className="mb-6">
        <p className="text-[13px] font-medium uppercase tracking-[0.08em] text-ak-brand">Math</p>
        <h1 className="mt-1 text-[28px] font-semibold leading-tight tracking-[-0.02em] text-ak-fg sm:text-[34px]">
          Write mathematics. Get exact answers.
        </h1>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-ak-fg2">
          Each line is read as ordinary maths and computed by Alkahest. Results it checked are marked{' '}
          <span className="rounded-full bg-ak-ok/10 px-1.5 py-px font-mono text-[12px] text-ak-ok">verified</span>; when it
          cannot stand behind an answer it declines with a coded reason instead of guessing.
        </p>
        {backend.status === 'online' && backend.local && (
          <p
            className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-ak-ok/30 bg-ak-ok/[0.07] px-2.5 py-0.5 text-xs font-medium text-ak-ok"
            data-testid="math-local-pill"
            title={`Connected to python -m alkahest.playground at ${LOCAL_MATH_URL}${backend.version ? ` (Alkahest ${backend.version})` : ''}`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-ak-ok" aria-hidden />
            Computing on this machine
          </p>
        )}
      </header>

      {!online &&
        (backend.status === 'none' ? (
          <LocalSetupCard
            state={backend}
            hasPrecomputed={cells.some((c) => c.precomputed)}
            onCheck={() => connect({ userInitiated: true })}
            onSettings={openSettings}
          />
        ) : (
          <BackendNotice state={backend} onRetry={() => void connect({ userInitiated: true })} onSettings={openSettings} />
        ))}

      <div className="sticky top-[60px] z-30 -mx-4 mb-4 border-b border-ak-border bg-ak-bg/90 px-4 py-2 backdrop-blur-md">
        <div
          className="flex items-center gap-1 overflow-x-auto no-scrollbar sm:flex-wrap sm:gap-y-1 sm:overflow-visible"
          role="toolbar"
          aria-label="Insert symbols"
        >
          {MATH_PALETTE.map((group, gi) => (
            <div key={group.name} className="flex shrink-0 items-center gap-1">
              {gi > 0 && <span className="mx-1.5 h-4 w-px shrink-0 bg-ak-border" aria-hidden />}
              {group.items.map((item) => (
                <button
                  key={group.name + item.label}
                  type="button"
                  title={item.title}
                  aria-label={`Insert ${item.title}`}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => insertSnippet(item)}
                  className="h-8 min-w-[2rem] shrink-0 rounded-md px-2 font-mono text-[13.5px] text-ak-fg2 transition-colors hover:bg-ak-sunk hover:text-ak-fg"
                >
                  {item.label}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-2.5" aria-label="Math cells">
        {cells.map((cell, i) => (
          <MathCell
            key={cell.id}
            cell={cell}
            index={i}
            canRun={online}
            precomputedVersion={PRECOMPUTED_VERSION}
            inputRef={(el) => {
              if (el) inputRefs.current.set(cell.id, el);
              else inputRefs.current.delete(cell.id);
            }}
            onChange={(v) => setInput(cell.id, v)}
            onRun={() => void runOne(cell.id, cellsRef.current)}
            onKeyDown={onKeyDown(cell.id)}
            onFocus={() => {
              const el = inputRefs.current.get(cell.id);
              const end = el?.selectionStart ?? cell.input.length;
              activeRef.current = { id: cell.id, start: end, end: el?.selectionEnd ?? end };
              if (online && !cell.preview && cell.input.trim() && !cell.result) schedulePreview(cell.id);
            }}
            onSelect={(start, end) => {
              activeRef.current = { id: cell.id, start, end };
            }}
            onDelete={() => deleteCell(cell.id)}
          />
        ))}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <ToolbarButton onClick={() => addCell()}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg>
          Add line
        </ToolbarButton>
        <ToolbarButton onClick={() => void runAll()} disabled={!online || runningAll}>
          <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z" /></svg>
          {runningAll ? 'Running…' : 'Run all'}
        </ToolbarButton>
        <ToolbarButton onClick={resetExamples}>Reset examples</ToolbarButton>
        <span className="ml-auto hidden text-xs text-ak-muted sm:inline">
          <Kbd>Enter</Kbd> runs and moves on · <Kbd>↑</Kbd> <Kbd>↓</Kbd> between lines
        </span>
      </div>

      <section className="mt-8">
        <h2 className="text-[13px] font-medium uppercase tracking-[0.08em] text-ak-muted">More to try</h2>
        <div className="mt-2.5 flex flex-wrap gap-2">
          {MATH_MORE_EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => void tryExample(ex)}
              className="rounded-md border border-ak-border bg-ak-elev px-2.5 py-1 font-mono text-[12.5px] text-ak-fg2 transition-colors hover:border-ak-border-strong hover:text-ak-fg"
            >
              {ex}
            </button>
          ))}
        </div>
        <p className="mt-5 text-xs leading-relaxed text-ak-muted">
          Commands: <Code>integrate</Code> (<Code>dx</Code>, <Code>from a to b</Code>), <Code>d/dx</Code>,{' '}
          <Code>derivative of</Code>, <Code>solve … for x, y</Code>, <Code>simplify</Code>, <Code>factor</Code>,{' '}
          <Code>expand</Code>, <Code>limit … as x -&gt; a</Code>, <Code>series … at x = a to order n</Code>,{' '}
          <Code>sum … for k = a to b</Code>, <Code>numeric … to n digits</Code>. Define names with{' '}
          <Code>f = …</Code> or <Code>g(t) = …</Code> and use them in later lines. Need loops or plots? Use the Python
          tab.
        </p>
      </section>
    </div>
  );
}

/** No backend configured: how to compute on this machine, in two commands. */
function LocalSetupCard({
  state,
  hasPrecomputed,
  onCheck,
  onSettings,
}: {
  state: Extract<MathBackendState, { status: 'none' }>;
  hasPrecomputed: boolean;
  onCheck: () => Promise<unknown>;
  onSettings: () => void;
}) {
  const [checking, setChecking] = useState(false);
  const webkit = isWebKitOnly();
  const denied = state.permission === 'denied';
  return (
    <section
      className="mb-5 rounded-lg border border-ak-border bg-ak-elev px-4 py-4 text-sm sm:px-5"
      aria-labelledby="math-local-title"
      data-testid="math-local-card"
    >
      <h2 id="math-local-title" className="text-[15px] font-semibold text-ak-fg">
        Compute on your own machine
      </h2>
      <p className="mt-1 leading-relaxed text-ak-fg2">
        Alkahest is native code and doesn&apos;t run in the browser yet, so this tab talks to a small Alkahest
        server on your computer. Everything is computed locally; nothing you type is sent to a server.
      </p>
      {webkit && (
        <p className="mt-3 rounded-md border border-ak-warn/30 bg-ak-warn/[0.07] px-3 py-2 text-[13px] leading-relaxed text-ak-fg">
          Safari (and every browser on iPhone and iPad) blocks secure pages from reaching a server on your computer,
          so this page can&apos;t connect from here. Use Chrome, Edge or Firefox on a computer, or run the playground
          itself locally.
        </p>
      )}
      <ol className="mt-3 space-y-2">
        {LOCAL_SETUP_COMMANDS.map((cmd, i) => (
          <li key={cmd} className="flex items-center gap-3">
            <span
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-ak-sunk font-mono text-xs text-ak-fg2"
              aria-hidden
            >
              {i + 1}
            </span>
            <CopyCommand command={cmd} />
          </li>
        ))}
      </ol>
      <p className="mt-3 text-[13px] leading-relaxed text-ak-fg2">
        This page then finds it at <Code>{LOCAL_MATH_URL.replace('http://', '')}</Code>. Your browser may ask
        whether this site can connect to apps on your device: choose <strong>Allow</strong>. Needs Python 3.9 or
        newer.
      </p>
      {denied ? (
        <p className="mt-2 text-[13px] leading-relaxed text-ak-warn" role="status">
          Your browser is blocking this page from connecting to apps on this device. Allow it in the site settings
          (the icon to the left of the address bar), then check again.
        </p>
      ) : (
        state.tried &&
        !checking && (
          <p className="mt-2 text-[13px] text-ak-muted" role="status" data-testid="math-local-status">
            No local server answered at {LOCAL_MATH_URL}.
          </p>
        )
      )}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={async () => {
            setChecking(true);
            try {
              await onCheck();
            } finally {
              setChecking(false);
            }
          }}
          disabled={checking}
          className="rounded-md bg-ak-brand px-3 py-1.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-60"
        >
          {checking ? 'Checking…' : 'Check again'}
        </button>
        <ToolbarButton onClick={onSettings}>Use another server</ToolbarButton>
      </div>
      {!webkit && (
        <p className="mt-3 text-xs leading-relaxed text-ak-muted">
          Works in Chrome, Edge and Firefox. Safari blocks this page from reaching your computer; there, run the
          playground itself locally.
        </p>
      )}
      {hasPrecomputed && (
        <p className="mt-1 text-xs leading-relaxed text-ak-muted">
          Until then, the results below were computed in advance with Alkahest {PRECOMPUTED_VERSION} when this page
          was built.
        </p>
      )}
    </section>
  );
}

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex min-w-0 flex-1 items-center rounded-md border border-ak-border bg-ak-code-bg">
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap px-3 py-1.5 font-mono text-[12px] text-ak-fg no-scrollbar sm:text-[13px]">
        {command}
      </code>
      <button
        type="button"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(command);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          } catch {
            /* ignore */
          }
        }}
        aria-label={`Copy: ${command}`}
        className="shrink-0 border-l border-ak-border px-2.5 py-1.5 text-xs text-ak-muted transition-colors hover:text-ak-fg"
      >
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  );
}

function BackendNotice({
  state,
  onRetry,
  onSettings,
}: {
  state: MathBackendState;
  onRetry: () => void;
  onSettings: () => void;
}) {
  if (state.status === 'checking') {
    return (
      <div className="mb-5 rounded-lg border border-ak-border bg-ak-elev px-4 py-3 text-sm text-ak-muted">
        Looking for an Alkahest backend…
      </div>
    );
  }
  let title = "The Alkahest server didn't answer";
  let body: React.ReactNode;
  if (state.status === 'jupyter') {
    title = 'The Math tab needs an Alkahest server, not Jupyter';
    body = <>Your settings point at a Jupyter server. The Math tab talks to the playground&apos;s own Alkahest server (it has a restricted <Code>/math</Code> endpoint). Switch the backend type in settings.</>;
  } else if (state.status === 'outdated') {
    title = 'This server is too old for the Math tab';
    body = <>The server answered, but it has no <Code>/math</Code> endpoint. Update <Code>demo-playground/server</Code> and restart it.</>;
  } else {
    body = (
      <>
        The server in your settings did not answer
        {state.status === 'offline' ? <> ({state.reason})</> : null}. Start it, or clear the backend URL in
        settings to use a local <Code>python -m alkahest.playground</Code> instead.
      </>
    );
  }
  return (
    <div className="mb-5 rounded-lg border border-ak-warn/30 bg-ak-warn/[0.06] px-4 py-3.5 text-sm" role="status">
      <p className="font-medium text-ak-fg">{title}</p>
      <p className="mt-1 leading-relaxed text-ak-fg2">{body}</p>
      <div className="mt-3 flex gap-2">
        <ToolbarButton onClick={onSettings}>Backend settings</ToolbarButton>
        <ToolbarButton onClick={onRetry}>Retry</ToolbarButton>
      </div>
    </div>
  );
}

function ToolbarButton({
  children,
  onClick,
  disabled,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex items-center gap-1.5 rounded-md border border-ak-border bg-ak-elev px-3 py-1.5 text-xs font-medium text-ak-fg2 transition-colors hover:border-ak-border-strong hover:text-ak-fg disabled:opacity-40"
    >
      {children}
    </button>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border border-ak-border bg-ak-elev px-1 py-px font-mono text-[11px] text-ak-fg2">
      {children}
    </kbd>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return <code className="rounded bg-ak-sunk px-1 py-px font-mono text-[0.92em] text-ak-fg2">{children}</code>;
}
