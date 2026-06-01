'use client';

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import { v4 as uuid } from 'uuid';
import Cell, { type CellData } from './Cell';
import CommandPalette, { type PaletteCommand } from './CommandPalette';
import RunMenu, { type RunMenuAction } from './RunMenu';
import { writeCellToClipboard } from '@/lib/cell-clipboard';
import { handleNotebookChordKey } from '@/lib/notebook-chords';
import { ADD_CODE_CELL_SHORTCUT_HINT, NOTEBOOK_COMMAND_DEFS, type NotebookCommandId } from '@/lib/notebook-commands';
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
import { useSettings } from '@/components/ui/SettingsContext';
import { connectionFromConfig } from '@/lib/server-connection';
import { isStaticHosting } from '@/lib/hosting';
import { parseNotebookFile } from '@/lib/notebook-import';
import { cellsFromDemoParam, readAutoRunFromUrl, readHideLineNumbersFromUrl } from '@/lib/recording';

// ── State ──────────────────────────────────────────────────────────────────

type Action =
  | { type: 'ADD_CELL'; afterId?: string; cellType?: CellData['cellType'] }
  | { type: 'REMOVE_CELL'; id: string }
  | { type: 'RESTORE_CELL'; cell: CellData; index: number }
  | { type: 'SET_CODE'; id: string; code: string }
  | { type: 'SET_STATUS'; id: string; status: CellData['status'] }
  | { type: 'SET_BACKEND'; id: string; backend: CellData['backend'] }
  | { type: 'APPEND_OUTPUT'; id: string; item: OutputItem }
  | { type: 'CLEAR_OUTPUTS'; id: string }
  | { type: 'CLEAR_ALL_OUTPUTS' }
  | { type: 'SET_OUTPUTS'; id: string; outputs: OutputItem[] }
  | { type: 'POSTPROCESS_OUTPUTS'; id: string }
  | { type: 'SET_EXEC_COUNT'; id: string; count: number }
  | { type: 'MOVE_UP'; id: string }
  | { type: 'MOVE_DOWN'; id: string }
  | { type: 'TOGGLE_CELL_TYPE'; id: string }
  | { type: 'SET_CELLS'; cells: CellData[] };

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
    case 'RESTORE_CELL': {
      const placeholder =
        state.length === 1 &&
        !state[0].code.trim() &&
        state[0].outputs.length === 0 &&
        state[0].status === 'idle';
      if (placeholder) return [{ ...action.cell, id: action.cell.id }];
      const next = [...state];
      const at = Math.min(Math.max(action.index, 0), next.length);
      next.splice(at, 0, { ...action.cell });
      return next;
    }
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
    case 'CLEAR_ALL_OUTPUTS':
      return state.map((c) => ({ ...c, outputs: [] }));
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
    case 'SET_CELLS':
      return action.cells.length > 0 ? action.cells : [newCell()];
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
];

function cellFromDemoSource(code: string): CellData {
  const trimmed = code.trimStart();
  if (trimmed.startsWith('##') || trimmed.startsWith('# Groebner')) {
    return newCell(code, 'markdown');
  }
  return newCell(code, 'code');
}

const NOTEBOOK_STORAGE_KEY = 'alkahest-playground-notebook';

function loadSavedNotebook(): CellData[] | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(NOTEBOOK_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { code: string; cellType: string }[];
    return parsed.map(({ code, cellType }) => newCell(code, cellType as CellData['cellType']));
  } catch { return null; }
}

function saveNotebook(cells: CellData[]) {
  localStorage.setItem(NOTEBOOK_STORAGE_KEY, JSON.stringify(
    cells.map(({ code, cellType }) => ({ code, cellType })),
  ));
}

function toJupyterSource(code: string): string[] {
  if (!code) return [];
  const lines = code.split('\n');
  return lines.map((line, i) => (i < lines.length - 1 ? `${line}\n` : line));
}

function exportAsIpynb(cells: CellData[]) {
  const notebook = {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
      language_info: { name: 'python', version: '3.11.0' },
    },
    cells: cells.map((cell) => {
      const source = toJupyterSource(cell.code);
      if (cell.cellType === 'markdown') {
        return { cell_type: 'markdown', metadata: {}, source };
      }
      return { cell_type: 'code', execution_count: null, metadata: {}, outputs: [], source };
    }),
  };
  const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'alkahest-notebook.ipynb';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function initialCells(demoParam: string): CellData[] {
  const codes = cellsFromDemoParam(demoParam);
  if (codes) return codes.map(cellFromDemoSource);
  const saved = loadSavedNotebook();
  if (saved) return saved;
  return INITIAL_CELLS;
}

// ── Component ─────────────────────────────────────────────────────────────

interface NotebookProps {
  zenMode?: boolean;
  onServerStatusChange?: (status: 'unknown' | 'online' | 'offline') => void;
  /** URL query param for base64 cell injection (default `demo`; compare view uses `left` / `right`). */
  demoParam?: string;
  /** Drop the max-width constraint for side-by-side compare layout. */
  compact?: boolean;
  onReady?: () => void;
  onDirtyChange?: (dirty: boolean) => void;
}

export default function Notebook({
  zenMode,
  onServerStatusChange,
  demoParam = 'demo',
  compact = false,
  onReady,
  onDirtyChange,
}: NotebookProps = {}) {
  const [cells, dispatch] = useReducer(reducer, demoParam, initialCells);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'unknown' | 'online' | 'offline'>('unknown');
  const [execCount, setExecCount] = useState(0);
  const [autoRunPending, setAutoRunPending] = useState(false);
  const [focusedCellId, setFocusedCellId] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [canUndoDelete, setCanUndoDelete] = useState(false);
  const cfg = useRef(loadConfig());
  const cleanupFns = useRef<Map<string, () => void>>(new Map());
  const wasmExecGen = useRef<Map<string, number>>(new Map());
  const deletedStack = useRef<{ cell: CellData; index: number }[]>([]);
  const chordActive = useRef(false);
  const chordTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isDirtyRef = useRef(false);
  const cellsRef = useRef(cells);
  const onDirtyChangeRef = useRef(onDirtyChange);
  const { registerExport, registerImport, closeSettings, openSettings } = useSettings();
  const hideLineNumbers =
    loadConfig().hideLineNumbers || (typeof window !== 'undefined' && readHideLineNumbersFromUrl());

  useEffect(() => { cellsRef.current = cells; }, [cells]);
  useEffect(() => { onDirtyChangeRef.current = onDirtyChange; }, [onDirtyChange]);

  useEffect(() => {
    registerExport(() => exportAsIpynb(cellsRef.current));
    return () => registerExport(null);
  }, [registerExport]);

  const handleImportNotebook = useCallback(
    async (file: File) => {
      try {
        const text = await file.text();
        const imported = parseNotebookFile(text, file.name);
        const next = imported.map((c) => newCell(c.code, c.cellType));
        dispatch({ type: 'SET_CELLS', cells: next });
        saveNotebook(next);
        isDirtyRef.current = false;
        onDirtyChangeRef.current?.(false);
        deletedStack.current = [];
        setCanUndoDelete(false);
        closeSettings();
      } catch (err) {
        alert(`Import failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [closeSettings],
  );

  useEffect(() => {
    registerImport(handleImportNotebook);
    return () => registerImport(null);
  }, [registerImport, handleImportNotebook]);

  function userDispatch(action: Action) {
    dispatch(action);
    if (!isDirtyRef.current) {
      isDirtyRef.current = true;
      onDirtyChangeRef.current?.(true);
    }
  }

  const addCodeCellBelow = useCallback(() => {
    const list = cellsRef.current;
    const afterId = focusedCellId ?? list[list.length - 1]?.id;
    userDispatch({ type: 'ADD_CELL', afterId, cellType: 'code' });
  }, [focusedCellId]);

  const undoDeleteCell = useCallback(() => {
    const entry = deletedStack.current.pop();
    if (!entry) return;
    userDispatch({ type: 'RESTORE_CELL', cell: entry.cell, index: entry.index });
    setCanUndoDelete(deletedStack.current.length > 0);
  }, []);

  const handleDeleteCell = useCallback((id: string) => {
    const list = cellsRef.current;
    const index = list.findIndex((c) => c.id === id);
    if (index < 0) return;
    deletedStack.current.push({ cell: { ...list[index] }, index });
    setCanUndoDelete(true);
    userDispatch({ type: 'REMOVE_CELL', id });
  }, []);

  const handleCopyCell = useCallback(async (id: string) => {
    const cell = cellsRef.current.find((c) => c.id === id);
    if (!cell) return;
    try {
      await writeCellToClipboard(cell);
    } catch {
      alert('Could not copy to clipboard.');
    }
  }, []);

  const handleCutCell = useCallback(
    async (id: string) => {
      await handleCopyCell(id);
      handleDeleteCell(id);
    },
    [handleCopyCell, handleDeleteCell],
  );

  const autoRun = readAutoRunFromUrl();

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

  // Signal headless recorders once CodeMirror has rendered demo cells.
  useEffect(() => {
    if (!onReady) return;
    const editors = document.querySelectorAll('.cm-editor');
    if (editors.length >= cells.filter((c) => c.cellType === 'code').length) {
      onReady();
    }
  }, [cells, onReady]);

  // Auto-run all cells sequentially when triggered by ?autorun=1
  useEffect(() => {
    if (!autoRunPending || !sessionId) return;
    setAutoRunPending(false);
    let cancelled = false;
    (async () => {
      for (const cell of cells) {
        if (cancelled || cell.cellType === 'markdown') continue;
        runCell(cell.id);
        await waitForCellDone(cell.id, () => cancelled);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRunPending, sessionId]);

  const stopCell = useCallback((id: string) => {
    cleanupFns.current.get(id)?.();
    cleanupFns.current.delete(id);
    wasmExecGen.current.set(id, (wasmExecGen.current.get(id) ?? 0) + 1);
    dispatch({ type: 'SET_STATUS', id, status: 'idle' });
  }, []);

  const interruptAll = useCallback(() => {
    for (const id of [...cleanupFns.current.keys()]) {
      stopCell(id);
    }
    cellsRef.current
      .filter((c) => c.status === 'running')
      .forEach((c) => stopCell(c.id));
  }, [stopCell]);

  const runCell = useCallback(
    (id: string) => {
      const cell = cells.find((c) => c.id === id);
      if (!cell || cell.status === 'running' || cell.cellType === 'markdown') return;

      cleanupFns.current.get(id)?.();
      cleanupFns.current.delete(id);

      const gen = (wasmExecGen.current.get(id) ?? 0) + 1;
      wasmExecGen.current.set(id, gen);

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
            if (wasmExecGen.current.get(id) !== gen) return;
            dispatch({ type: 'POSTPROCESS_OUTPUTS', id });
            dispatch({ type: 'SET_EXEC_COUNT', id, count: n });
            dispatch({ type: 'SET_STATUS', id, status: 'done' });
            cleanupFns.current.delete(id);
          },
          (err) => {
            if (wasmExecGen.current.get(id) !== gen) return;
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
            if (wasmExecGen.current.get(id) !== gen) return;
            outputs.forEach((item) => dispatch({ type: 'APPEND_OUTPUT', id, item }));
            const n = execCount + 1;
            setExecCount(n);
            dispatch({ type: 'SET_EXEC_COUNT', id, count: n });
            dispatch({ type: 'SET_STATUS', id, status: 'done' });
          })
          .catch((err: Error) => {
            if (wasmExecGen.current.get(id) !== gen) return;
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

  const runCellsSequential = useCallback(
    async (cellIds: string[]) => {
      for (const id of cellIds) {
        const cell = cellsRef.current.find((c) => c.id === id);
        if (!cell || cell.cellType === 'markdown') continue;
        runCell(id);
        await waitForCellDone(id, () => false);
      }
    },
    [runCell],
  );

  const restartSession = useCallback(async () => {
    interruptAll();
    const conn = connectionFromConfig(cfg.current);
    if (sessionId) {
      await destroySession(conn, sessionId).catch(() => {});
      setSessionId(null);
    }
    if (!conn.httpUrl) {
      setExecCount(0);
      return;
    }
    try {
      const id = await createSession(conn);
      setSessionId(id);
      setServerStatus('online');
      onServerStatusChange?.('online');
    } catch {
      setServerStatus('offline');
      onServerStatusChange?.('offline');
    }
  }, [interruptAll, sessionId, onServerStatusChange]);

  const handleRunMenuAction = useCallback(
    (action: RunMenuAction) => {
      const codeCells = cellsRef.current.filter((c) => c.cellType === 'code');
      switch (action) {
        case 'run-all':
          void runCellsSequential(codeCells.map((c) => c.id));
          break;
        case 'restart':
          void restartSession();
          break;
        case 'restart-run-all':
          void (async () => {
            await restartSession();
            await runCellsSequential(codeCells.map((c) => c.id));
          })();
          break;
        case 'run-below': {
          const focusId = focusedCellId ?? cellsRef.current[0]?.id;
          if (!focusId) break;
          const start = cellsRef.current.findIndex((c) => c.id === focusId);
          if (start < 0) break;
          const below = cellsRef.current.slice(start).filter((c) => c.cellType === 'code');
          void runCellsSequential(below.map((c) => c.id));
          break;
        }
        case 'interrupt':
          interruptAll();
          break;
        case 'clear-outputs':
          dispatch({ type: 'CLEAR_ALL_OUTPUTS' });
          break;
      }
    },
    [focusedCellId, interruptAll, restartSession, runCellsSequential],
  );

  const anyRunning = cells.some((c) => c.status === 'running');

  const executeCommand = useCallback(
    (id: NotebookCommandId) => {
      const focusId = focusedCellId ?? cellsRef.current[0]?.id;
      switch (id) {
        case 'command-palette':
          setPaletteOpen(true);
          break;
        case 'run-cell':
          if (focusId) runCell(focusId);
          break;
        case 'add-code-cell':
          addCodeCellBelow();
          break;
        case 'add-markdown-cell': {
          const afterId = focusedCellId ?? cellsRef.current[cellsRef.current.length - 1]?.id;
          userDispatch({ type: 'ADD_CELL', afterId, cellType: 'markdown' });
          break;
        }
        case 'undo-delete-cell':
          undoDeleteCell();
          break;
        case 'save-notebook':
          saveNotebook(cellsRef.current);
          isDirtyRef.current = false;
          onDirtyChangeRef.current?.(false);
          break;
        case 'open-settings':
          openSettings();
          break;
        case 'delete-focused-cell':
          if (focusId) handleDeleteCell(focusId);
          break;
        case 'move-cell-up':
          if (focusId) userDispatch({ type: 'MOVE_UP', id: focusId });
          break;
        case 'move-cell-down':
          if (focusId) userDispatch({ type: 'MOVE_DOWN', id: focusId });
          break;
        case 'toggle-cell-type':
          if (focusId) userDispatch({ type: 'TOGGLE_CELL_TYPE', id: focusId });
          break;
        default:
          handleRunMenuAction(id);
      }
    },
    [
      focusedCellId,
      addCodeCellBelow,
      undoDeleteCell,
      handleDeleteCell,
      openSettings,
      runCell,
      handleRunMenuAction,
    ],
  );

  const paletteCommands: PaletteCommand[] = useMemo(() => {
    const focusId = focusedCellId;
    const focusCell = focusId ? cells.find((c) => c.id === focusId) : null;
    return NOTEBOOK_COMMAND_DEFS.map((def) => {
      let disabled = false;
      if (def.id === 'undo-delete-cell') disabled = !canUndoDelete;
      if (def.id === 'interrupt') disabled = !anyRunning;
      if (def.id === 'run-cell') disabled = !focusId || focusCell?.cellType === 'markdown';
      if (def.id === 'delete-focused-cell' || def.id === 'move-cell-up' || def.id === 'move-cell-down' || def.id === 'toggle-cell-type') {
        disabled = !focusId;
      }
      return {
        ...def,
        disabled,
        run: () => executeCommand(def.id),
      };
    });
  }, [cells, focusedCellId, anyRunning, canUndoDelete, executeCommand]);

  useEffect(() => {
    if (zenMode) return;
    function onKeyDown(e: KeyboardEvent) {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.shiftKey && e.key.toLowerCase() === 'p') {
        e.preventDefault();
        setPaletteOpen(true);
        return;
      }
      if (handleNotebookChordKey(e, chordActive, chordTimer, {
        onAddCodeCell: addCodeCellBelow,
        onUndoDeleteCell: undoDeleteCell,
      })) {
        return;
      }
      if (mod && e.key === 's') {
        e.preventDefault();
        saveNotebook(cellsRef.current);
        isDirtyRef.current = false;
        onDirtyChangeRef.current?.(false);
      }
    }
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [zenMode, addCodeCellBelow, undoDeleteCell]);

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
    <div className={compact ? 'px-2 py-3 space-y-2' : 'mx-auto max-w-4xl px-4 py-6 space-y-3'}>
      {/* Toolbar — hidden in zen mode */}
      {!zenMode && <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={() => addCodeCellBelow()}
          title={ADD_CODE_CELL_SHORTCUT_HINT}
          className="flex items-center gap-1.5 rounded border border-ak-border px-3 py-1.5 text-xs hover:bg-ak-code-bg transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14"/></svg>
          Add code
        </button>

        <button
          onClick={() => userDispatch({ type: 'ADD_CELL', cellType: 'markdown' })}
          className="flex items-center gap-1.5 rounded border border-ak-border px-3 py-1.5 text-xs hover:bg-ak-code-bg transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 6h16M4 12h8M4 18h16"/></svg>
          Add markdown
        </button>

        <RunMenu onAction={handleRunMenuAction} anyRunning={anyRunning} />

        <label className="flex items-center gap-1.5 rounded border border-ak-border px-3 py-1.5 text-xs cursor-pointer hover:bg-ak-code-bg transition-colors">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
          Install wheel
          <input type="file" accept=".whl" className="hidden" onChange={(e) => e.target.files?.[0] && handleWheelUpload(e.target.files[0])} />
        </label>

        <div className="flex-1" />
      </div>}

      {/* Cells */}
      {cells.map((cell) => (
        <Cell
          key={cell.id}
          cell={cell}
          onCodeChange={(id, code) => userDispatch({ type: 'SET_CODE', id, code })}
          onRun={runCell}
          onStop={stopCell}
          onDelete={handleDeleteCell}
          onMoveUp={(id) => userDispatch({ type: 'MOVE_UP', id })}
          onMoveDown={(id) => userDispatch({ type: 'MOVE_DOWN', id })}
          onAddBelow={(id, cellType) => userDispatch({ type: 'ADD_CELL', afterId: id, cellType })}
          onToggleCellType={(id) => userDispatch({ type: 'TOGGLE_CELL_TYPE', id })}
          onCopyCell={(id) => void handleCopyCell(id)}
          onCutCell={(id) => void handleCutCell(id)}
          onOutputsChange={(id, outputs) => dispatch({ type: 'SET_OUTPUTS', id, outputs })}
          onFocus={setFocusedCellId}
          zenMode={zenMode}
          showLineNumbers={!hideLineNumbers}
        />
      ))}

      {!zenMode && (
        <CommandPalette
          open={paletteOpen}
          onClose={() => setPaletteOpen(false)}
          commands={paletteCommands}
        />
      )}
    </div>
  );
}

function waitForCellDone(cellId: string, cancelled: () => boolean): Promise<void> {
  return new Promise((resolve) => {
    const deadline = Date.now() + 120_000;
    const tick = () => {
      if (cancelled()) return resolve();
      const el = document.querySelector(`[data-cell-id="${cellId}"]`);
      const running = el?.getAttribute('data-cell-status') === 'running';
      if (!running) return resolve();
      if (Date.now() > deadline) return resolve();
      setTimeout(tick, 200);
    };
    setTimeout(tick, 400);
  });
}
