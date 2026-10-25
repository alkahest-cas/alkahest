'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { v4 as uuid } from 'uuid';
import { loadConfig } from '@/components/ui/Settings';
import { useSettings } from '@/components/ui/SettingsContext';
import { connectionFromConfig, type ServerConnection } from '@/lib/server-connection';
import { isStaticHosting } from '@/lib/hosting';
import {
  interpretMath,
  probeMathBackend,
  runMath,
  type MathBackendState,
  type MathDefinition,
  type MathResult,
} from '@/lib/math-api';
import { MATH_MORE_EXAMPLES, MATH_PALETTE, MATH_STARTER_CELLS, type PaletteItem } from '@/lib/math-starter';
import MathCell, { type MathCellData } from './MathCell';

const STORAGE_KEY = 'alkahest-playground-math';
const PREVIEW_DEBOUNCE_MS = 180;

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
  onServerStatusChange?: (status: 'unknown' | 'online' | 'offline') => void;
  onReady?: () => void;
}

export default function MathNotebook({ onServerStatusChange, onReady }: MathNotebookProps) {
  const { openSettings } = useSettings();
  const [cells, setCells] = useState<MathCellData[]>(() =>
    (loadInputs() ?? MATH_STARTER_CELLS).map((s) => newCell(s)),
  );
  const cellsRef = useRef(cells);
  cellsRef.current = cells;
  const [backend, setBackend] = useState<MathBackendState>({ status: 'checking' });
  const connRef = useRef<ServerConnection | null>(null);
  const inputRefs = useRef(new Map<string, HTMLInputElement>());
  const activeRef = useRef<{ id: string; start: number; end: number } | null>(null);
  const previewTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const autoRan = useRef(false);
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
  const probe = useCallback(async () => {
    setBackend({ status: 'checking' });
    onServerStatusChange?.('unknown');
    const conn = connectionFromConfig(loadConfig());
    connRef.current = conn;
    const state = await probeMathBackend(conn);
    setBackend(state);
    onServerStatusChange?.(state.status === 'online' ? 'online' : 'offline');
    return state;
  }, [onServerStatusChange]);

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
        result = {
          ok: false,
          error_code: 'NETWORK',
          error_message: `Could not reach the backend (${e instanceof Error ? e.message : String(e)}).`,
          error_source: 'playground',
        };
      }
      patch(id, (c) => ({ ...c, status: 'done', result, ranInput: input }));
      return result;
    },
    [patch],
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

  useEffect(() => {
    let cancelled = false;
    probe().then((state) => {
      if (cancelled) return;
      if (state.status === 'online' && !autoRan.current) {
        autoRan.current = true;
        void runAll().then(() => onReady?.());
      } else {
        onReady?.();
      }
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      patch(last.id, (c) => ({ ...c, input: example }));
    } else {
      target = newCell(example);
      setCells((prev) => [...prev, target]);
    }
    focusCell(target.id, 'end');
    if (online) {
      const next = [...list.filter((c) => c.id !== target.id), target];
      await runOne(target.id, next);
    }
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
    const fresh = MATH_STARTER_CELLS.map((s) => newCell(s));
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
      </header>

      {!online && <BackendNotice state={backend} onRetry={() => void probe()} onSettings={openSettings} />}

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
        Connecting to the Alkahest backend…
      </div>
    );
  }
  let title = 'Computation needs an Alkahest backend';
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
        This page does not compute anything in your browser, and the Math tab will not pretend to:
        every answer here comes from Alkahest running on a server.{' '}
        {state.status === 'offline' ? <>The configured server did not answer ({state.reason}). </> : null}
        To run one locally, clone the repository and start the playground:
      </>
    );
  }
  return (
    <div className="mb-5 rounded-lg border border-ak-warn/30 bg-ak-warn/[0.06] px-4 py-3.5 text-sm" role="status">
      <p className="font-medium text-ak-fg">{title}</p>
      <p className="mt-1 leading-relaxed text-ak-fg2">{body}</p>
      {(state.status === 'none' || state.status === 'offline') && (
        <pre className="mt-2.5 overflow-x-auto rounded-md border border-ak-border bg-ak-elev px-3 py-2 font-mono text-[12.5px] text-ak-fg">
{`cd alkahest/demo-playground
pnpm install && pnpm start   # web on :3000, Alkahest server on :8000`}
        </pre>
      )}
      {isStaticHosting && (state.status === 'none' || state.status === 'offline') && (
        <p className="mt-2 text-xs leading-relaxed text-ak-muted">
          This hosted page is served over HTTPS, so it can only reach an <Code>https://</Code> server. With a local
          server, use the playground at <Code>http://localhost:3000</Code> instead.
        </p>
      )}
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
