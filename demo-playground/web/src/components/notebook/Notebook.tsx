'use client';

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { v4 as uuid } from 'uuid';
import Cell, { type CellData } from './Cell';
import type { OutputItem, ExecutionMode } from '@/lib/execution';
import {
  needsServer,
  createSession,
  destroySession,
  installWheel,
  executeOnServer,
  executeInWasm,
} from '@/lib/execution';
import { postprocessOutputItems } from '@/lib/lean';
import { loadConfig } from '@/components/ui/Settings';
import { connectionFromConfig } from '@/lib/server-connection';
import { isStaticHosting } from '@/lib/hosting';

// ── State ──────────────────────────────────────────────────────────────────

type Action =
  | { type: 'ADD_CELL'; afterId?: string; cellType?: CellData['cellType'] }
  | { type: 'REMOVE_CELL'; id: string }
  | { type: 'SET_CODE'; id: string; code: string }
  | { type: 'SET_STATUS'; id: string; status: CellData['status'] }
  | { type: 'SET_BACKEND'; id: string; backend: CellData['backend'] }
  | { type: 'APPEND_OUTPUT'; id: string; item: OutputItem }
  | { type: 'CLEAR_OUTPUTS'; id: string }
  | { type: 'SET_OUTPUTS'; id: string; outputs: OutputItem[] }
  | { type: 'POSTPROCESS_OUTPUTS'; id: string }
  | { type: 'SET_EXEC_COUNT'; id: string; count: number }
  | { type: 'MOVE_UP'; id: string }
  | { type: 'MOVE_DOWN'; id: string }
  | { type: 'TOGGLE_CELL_TYPE'; id: string };

function newCell(code = '', cellType: CellData['cellType'] = 'code'): CellData {
  return { id: uuid(), code, outputs: [], status: 'idle', executionCount: null, backend: null, cellType };
}

function reducer(state: CellData[], action: Action): CellData[] {
  switch (action.type) {
    case 'ADD_CELL': {
      const idx = action.afterId ? state.findIndex((c) => c.id === action.afterId) : state.length - 1;
      const next = [...state];
      next.splice(idx + 1, 0, newCell('', action.cellType ?? 'code'));
      return next;
    }
    case 'REMOVE_CELL':
      return state.length === 1 ? [newCell()] : state.filter((c) => c.id !== action.id);
    case 'SET_CODE':
      return state.map((c) => (c.id === action.id ? { ...c, code: action.code } : c));
    case 'SET_STATUS':
      return state.map((c) => (c.id === action.id ? { ...c, status: action.status } : c));
    case 'SET_BACKEND':
      return state.map((c) => (c.id === action.id ? { ...c, backend: action.backend } : c));
    case 'APPEND_OUTPUT':
      return state.map((c) =>
        c.id === action.id ? { ...c, outputs: [...c.outputs, action.item] } : c,
      );
    case 'CLEAR_OUTPUTS':
      return state.map((c) => (c.id === action.id ? { ...c, outputs: [] } : c));
    case 'SET_OUTPUTS':
      return state.map((c) => (c.id === action.id ? { ...c, outputs: action.outputs } : c));
    case 'POSTPROCESS_OUTPUTS':
      return state.map((c) =>
        c.id === action.id ? { ...c, outputs: postprocessOutputItems(c.outputs) } : c,
      );
    case 'SET_EXEC_COUNT':
      return state.map((c) => (c.id === action.id ? { ...c, executionCount: action.count } : c));
    case 'MOVE_UP': {
      const i = state.findIndex((c) => c.id === action.id);
      if (i <= 0) return state;
      const next = [...state];
      [next[i - 1], next[i]] = [next[i], next[i - 1]];
      return next;
    }
    case 'MOVE_DOWN': {
      const i = state.findIndex((c) => c.id === action.id);
      if (i >= state.length - 1) return state;
      const next = [...state];
      [next[i], next[i + 1]] = [next[i + 1], next[i]];
      return next;
    }
    case 'TOGGLE_CELL_TYPE':
      return state.map((c) =>
        c.id === action.id
          ? { ...c, cellType: c.cellType === 'markdown' ? 'code' : 'markdown', outputs: [], status: 'idle', executionCount: null, backend: null }
          : c,
      );
    default:
      return state;
  }
}

// Avoid JS template literals for Python code that contains ${} — use string concat instead
const INITIAL_CELLS: CellData[] = [
  newCell(
    'import alkahest as ak\n' +
    'from alkahest import latex\n' +
    'from playground_helpers import display_lean_cert\n\n' +
    'pool = ak.ExprPool()\n' +
    'x = pool.symbol("x")\n' +
    'zero = pool.integer(0)\n\n' +
    'expr = x + zero\n' +
    'result = ak.simplify(pool, expr)\n' +
    'print("simplify(x + 0) = $$" + latex(result.value) + "$$")\n' +
    'print(result.derivation)\n' +
    'display_lean_cert(result, operation="simplify")\n',
  ),
  newCell(
    '# Compare with SymPy\n' +
    'from sympy import symbols, diff, latex as sp_latex\n\n' +
    'x = symbols("x")\n' +
    'result = diff(x**2, x)\n' +
    'print("SymPy: $$" + sp_latex(result) + "$$")\n',
  ),
];

// ── URL-param cell injection (for CLI-driven demos) ───────────────────────

function cellsFromUrlParam(): CellData[] | null {
  if (typeof window === 'undefined') return null;
  const params = new URLSearchParams(window.location.search);
  const encoded = params.get('demo');
  if (!encoded) return null;
  try {
    const codes: string[] = JSON.parse(atob(encoded));
    return codes.filter(Boolean).map((code) => newCell(code));
  } catch {
    return null;
  }
}

// ── Component ─────────────────────────────────────────────────────────────

interface NotebookProps {
  zenMode?: boolean;
  onServerStatusChange?: (status: 'unknown' | 'online' | 'offline') => void;
}

export default function Notebook({ zenMode, onServerStatusChange }: NotebookProps = {}) {
  const [cells, dispatch] = useReducer(reducer, null, () => cellsFromUrlParam() ?? INITIAL_CELLS);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'unknown' | 'online' | 'offline'>('unknown');
  const [execCount, setExecCount] = useState(0);
  const [autoRunPending, setAutoRunPending] = useState(false);
  const cfg = useRef(loadConfig());
  const cleanupFns = useRef<Map<string, () => void>>(new Map());

  const autoRun = typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('autorun') === '1';

  // ?mode=server|wasm|auto overrides the saved config for this page load
  if (typeof window !== 'undefined') {
    const modeParam = new URLSearchParams(window.location.search).get('mode') as ExecutionMode | null;
    if (modeParam && ['auto', 'wasm', 'server'].includes(modeParam)) {
      cfg.current = { ...cfg.current, executionMode: modeParam };
    }
  }

  // Create kernel session on mount; optionally auto-run all cells
  useEffect(() => {
    const conn = connectionFromConfig(cfg.current);
    if (!conn.httpUrl) {
      setServerStatus(isStaticHosting ? 'unknown' : 'offline');
      onServerStatusChange?.(isStaticHosting ? 'unknown' : 'offline');
      if (autoRun) setTimeout(() => setAutoRunPending(true), 800);
      return;
    }
    (async () => {
      try {
        const id = await createSession(conn);
        setSessionId(id);
        setServerStatus('online');
        onServerStatusChange?.('online');
        if (autoRun) {
          // Small delay to let the UI settle, then run all cells sequentially
          setTimeout(() => {
            setAutoRunPending(true);
          }, 800);
        }
      } catch {
        setServerStatus('offline');
        onServerStatusChange?.('offline');
      }
    })();
    return () => {
      if (sessionId) destroySession(connectionFromConfig(cfg.current), sessionId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-run all cells sequentially when triggered by ?autorun=1
  useEffect(() => {
    if (!autoRunPending || !sessionId) return;
    setAutoRunPending(false);
    let cancelled = false;
    (async () => {
      for (const cell of cells) {
        if (cancelled) break;
        runCell(cell.id);
        // Wait for this cell to finish before running the next
        await new Promise<void>((resolve) => {
          const interval = setInterval(() => {
            // We poll — cells state captured here may be stale but runCell updates it
            resolve();
          }, 2500);
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          void interval;
        });
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRunPending, sessionId]);

  const runCell = useCallback(
    (id: string) => {
      const cell = cells.find((c) => c.id === id);
      if (!cell || cell.status === 'running' || cell.cellType === 'markdown') return;

      // Cancel any existing execution for this cell
      cleanupFns.current.get(id)?.();
      cleanupFns.current.delete(id);

      dispatch({ type: 'CLEAR_OUTPUTS', id });
      dispatch({ type: 'SET_STATUS', id, status: 'running' });

      const mode = cfg.current.executionMode as ExecutionMode;
      const useServer = mode === 'server' || (mode !== 'wasm' && needsServer(cell.code));

      if (useServer) {
        dispatch({ type: 'SET_BACKEND', id, backend: 'server' });

        if (!sessionId) {
          dispatch({
            type: 'APPEND_OUTPUT',
            id,
            item: { type: 'error', ename: 'NoSession', evalue: 'Server not connected. Check settings.', traceback: [] },
          });
          dispatch({ type: 'SET_STATUS', id, status: 'error' });
          return;
        }

        const count = execCount + 1;
        setExecCount(count);

        const cancel = executeOnServer(
          connectionFromConfig(cfg.current),
          sessionId,
          cell.code,
          (item) => dispatch({ type: 'APPEND_OUTPUT', id, item }),
          (n) => {
            dispatch({ type: 'POSTPROCESS_OUTPUTS', id });
            dispatch({ type: 'SET_EXEC_COUNT', id, count: n });
            dispatch({ type: 'SET_STATUS', id, status: 'done' });
            cleanupFns.current.delete(id);
          },
          (err) => {
            dispatch({
              type: 'APPEND_OUTPUT',
              id,
              item: { type: 'error', ename: 'ExecutionError', evalue: err, traceback: [] },
            });
            dispatch({ type: 'SET_STATUS', id, status: 'error' });
          },
        );
        cleanupFns.current.set(id, cancel);
      } else {
        dispatch({ type: 'SET_BACKEND', id, backend: 'wasm' });

        executeInWasm(cell.code)
          .then((outputs) => {
            outputs.forEach((item) => dispatch({ type: 'APPEND_OUTPUT', id, item }));
            const n = execCount + 1;
            setExecCount(n);
            dispatch({ type: 'SET_EXEC_COUNT', id, count: n });
            dispatch({ type: 'SET_STATUS', id, status: 'done' });
          })
          .catch((err: Error) => {
            dispatch({
              type: 'APPEND_OUTPUT',
              id,
              item: { type: 'error', ename: 'WasmError', evalue: err.message, traceback: [] },
            });
            dispatch({ type: 'SET_STATUS', id, status: 'error' });
          });
      }
    },
    [cells, sessionId, execCount],
  );

  async function handleWheelUpload(file: File) {
    if (!sessionId) return alert('Server not connected');
    try {
      await installWheel(connectionFromConfig(cfg.current), sessionId, file);
      alert(`Installed ${file.name} successfully.`);
    } catch (e) {
      alert(`Wheel install failed: ${e}`);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-6 space-y-3">
      {/* Toolbar — hidden in zen mode */}
      {!zenMode && <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={() => dispatch({ type: 'ADD_CELL' })}
          className="flex items-center gap-1.5 rounded border border-ak-border px-3 py-1.5 text-xs hover:bg-ak-code-bg transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14"/></svg>
          Add code
        </button>

        <button
          onClick={() => dispatch({ type: 'ADD_CELL', cellType: 'markdown' })}
          className="flex items-center gap-1.5 rounded border border-ak-border px-3 py-1.5 text-xs hover:bg-ak-code-bg transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 6h16M4 12h8M4 18h16"/></svg>
          Add markdown
        </button>

        <button
          onClick={() => {
            const promise = cells.reduce((p, c) => p.then(() => {
              return new Promise<void>((res) => {
                runCell(c.id);
                // Small delay between cells
                setTimeout(res, 300);
              });
            }), Promise.resolve());
            void promise;
          }}
          className="flex items-center gap-1.5 rounded border border-ak-border px-3 py-1.5 text-xs hover:bg-ak-code-bg transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Run all
        </button>

        <label className="flex items-center gap-1.5 rounded border border-ak-border px-3 py-1.5 text-xs cursor-pointer hover:bg-ak-code-bg transition-colors">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
          Install wheel
          <input type="file" accept=".whl" className="hidden" onChange={(e) => e.target.files?.[0] && handleWheelUpload(e.target.files[0])} />
        </label>

        <div className="flex-1" />

        <div className="flex items-center gap-1.5 text-xs text-ak-muted">
          <span
            className={`h-2 w-2 rounded-full ${
              serverStatus === 'online' ? 'bg-green-500' : serverStatus === 'offline' ? 'bg-red-400' : 'bg-ak-border'
            }`}
          />
          {serverStatus === 'online' ? 'server ready' : serverStatus === 'offline' ? 'server offline' : 'connecting…'}
        </div>
      </div>}

      {/* Cells */}
      {cells.map((cell, i) => (
        <Cell
          key={cell.id}
          cell={cell}
          index={i}
          onCodeChange={(id, code) => dispatch({ type: 'SET_CODE', id, code })}
          onRun={runCell}
          onDelete={(id) => dispatch({ type: 'REMOVE_CELL', id })}
          onMoveUp={(id) => dispatch({ type: 'MOVE_UP', id })}
          onMoveDown={(id) => dispatch({ type: 'MOVE_DOWN', id })}
          onAddBelow={(id) => dispatch({ type: 'ADD_CELL', afterId: id })}
          onToggleCellType={(id) => dispatch({ type: 'TOGGLE_CELL_TYPE', id })}
          onOutputsChange={(id, outputs) => dispatch({ type: 'SET_OUTPUTS', id, outputs })}
          zenMode={zenMode}
        />
      ))}

      {/* Add cell footer — hidden in zen mode */}
      {!zenMode && (
        <button
          onClick={() => dispatch({ type: 'ADD_CELL' })}
          className="w-full rounded border border-dashed border-ak-border py-2 text-xs text-ak-muted hover:border-ak-muted hover:text-ak-fg transition-colors"
        >
          + add cell
        </button>
      )}
    </div>
  );
}
