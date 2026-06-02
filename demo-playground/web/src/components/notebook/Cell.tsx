'use client';

import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { EditorView, keymap } from '@codemirror/view';
import { Prec } from '@codemirror/state';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type { OutputItem } from '@/lib/execution';
import { ADD_CODE_CELL_SHORTCUT_HINT } from '@/lib/notebook-commands';
import CellActionsMenu from './CellActionsMenu';
import Output from './Output';

export interface CellData {
  id: string;
  code: string;
  outputs: OutputItem[];
  status: 'idle' | 'running' | 'done' | 'error';
  executionCount: number | null;
  backend: 'wasm' | 'server' | null;
  cellType: 'code' | 'markdown';
}

interface CellProps {
  cell: CellData;
  onCodeChange: (id: string, code: string) => void;
  onRun: (id: string) => void;
  onStop: (id: string) => void;
  onDelete: (id: string) => void;
  onMoveUp: (id: string) => void;
  onMoveDown: (id: string) => void;
  onAddBelow: (id: string, cellType?: CellData['cellType']) => void;
  onToggleCellType: (id: string) => void;
  onCopyCell: (id: string) => void;
  onCutCell: (id: string) => void;
  onOutputsChange?: (id: string, outputs: OutputItem[]) => void;
  onFocus?: (id: string) => void;
  /** Selected cell — shows toolbar and insert bar without hover. */
  isActive?: boolean;
  shouldFocus?: boolean;
  zenMode?: boolean;
  showInsertBar?: boolean;
  showLineNumbers?: boolean;
}

const warmLightTheme = EditorView.theme({
  '&': { backgroundColor: '#eeede7', color: '#111' },
  '.cm-content': { padding: '8px 0' },
  '.cm-line': { padding: '0 12px' },
  '.cm-gutters': { backgroundColor: '#e5e4de', borderRight: '1px solid #e0ded6', color: '#888' },
  '.cm-activeLineGutter': { backgroundColor: '#dddcd6' },
  '.cm-activeLine': { backgroundColor: 'rgba(0,0,0,0.03)' },
  '.cm-selectionBackground': { backgroundColor: '#c41e3a22' },
  '&.cm-focused .cm-selectionBackground': { backgroundColor: '#c41e3a33' },
  '.cm-cursor': { borderLeftColor: '#c41e3a' },
});

function useElapsedTimer(status: CellData['status']) {
  const [elapsed, setElapsed] = useState<number | null>(null);
  const startRef = useRef<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (status === 'running') {
      startRef.current = Date.now();
      setElapsed(0);
      intervalRef.current = setInterval(() => {
        setElapsed(Date.now() - (startRef.current ?? Date.now()));
      }, 80);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (status === 'idle') {
        setElapsed(null);
        startRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [status]);

  return elapsed;
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export default function Cell({
  cell,
  onCodeChange,
  onRun,
  onStop,
  onDelete,
  onMoveUp,
  onMoveDown,
  onAddBelow,
  onToggleCellType,
  onCopyCell,
  onCutCell,
  onOutputsChange,
  onFocus,
  isActive = false,
  shouldFocus,
  zenMode,
  showInsertBar = true,
  showLineNumbers = true,
}: CellProps) {
  const editorRef = useRef<{ view?: { focus: () => void } }>(null);
  const cellBodyRef = useRef<HTMLDivElement>(null);
  const [gutterWidth, setGutterWidth] = useState(0);
  const elapsed = useElapsedTimer(cell.status);
  const onRunRef = useRef(onRun);
  const onFocusRef = useRef(onFocus);
  const zenModeRef = useRef(zenMode);
  const cellIdRef = useRef(cell.id);
  onRunRef.current = onRun;
  onFocusRef.current = onFocus;
  zenModeRef.current = zenMode;
  cellIdRef.current = cell.id;

  const editorExtensions = useMemo(
    () => [
      python(),
      warmLightTheme,
      Prec.highest(
        keymap.of([
          {
            key: 'Mod-Enter',
            run: () => {
              onRunRef.current(cellIdRef.current);
              return true;
            },
          },
        ]),
      ),
      EditorView.domEventHandlers({
        focus: () => {
          if (!zenModeRef.current) onFocusRef.current?.(cellIdRef.current);
          return false;
        },
      }),
    ],
    [],
  );

  const isMarkdown = cell.cellType === 'markdown';

  useEffect(() => {
    if (!shouldFocus || isMarkdown) return;
    const t = setTimeout(() => editorRef.current?.view?.focus(), 0);
    return () => clearTimeout(t);
  }, [shouldFocus, isMarkdown, cell.id]);

  useEffect(() => {
    if (isMarkdown || !showLineNumbers) {
      setGutterWidth(0);
      return;
    }
    const root = cellBodyRef.current;
    if (!root) return;

    const measure = () => {
      const gutters = root.querySelector('.cm-gutters');
      setGutterWidth(gutters ? gutters.getBoundingClientRect().width : 0);
    };

    measure();
    const gutters = root.querySelector('.cm-gutters');
    if (!gutters) return;

    const ro = new ResizeObserver(measure);
    ro.observe(gutters);
    return () => ro.disconnect();
  }, [isMarkdown, showLineNumbers, cell.code]);
  const gutter = isMarkdown
    ? '[M]'
    : cell.executionCount !== null
      ? `[${cell.executionCount}]`
      : cell.status === 'running'
        ? '[*]'
        : '[ ]';

  const hasError = cell.outputs.some((o) => o.type === 'error');
  const isRunning = cell.status === 'running';
  const uiActive = isActive && !zenMode;
  const showChrome = uiActive || isRunning;

  const activateCell = () => {
    if (zenMode) return;
    onFocusRef.current?.(cellIdRef.current);
  };

  const handleCellMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    if (zenMode) return;
    const target = e.target as HTMLElement;
    if (target.closest('button') || target.closest('[role="menu"]')) return;
    activateCell();
  };

  const idleBorder = zenMode
    ? 'border-ak-border'
    : uiActive
      ? 'border-ak-muted/60 shadow-sm'
      : 'border-ak-border hover:border-ak-muted/40';

  return (
    <div className="group/cell">
      <div
        data-cell-id={cell.id}
        data-cell-status={cell.status}
        data-cell-active={uiActive || undefined}
        onMouseDown={handleCellMouseDown}
        className={`relative overflow-hidden rounded-lg border transition-all ${
          isMarkdown
            ? idleBorder
            : isRunning
              ? 'border-ak-brand shadow-sm'
              : hasError
                ? 'border-red-200'
                : idleBorder
        }`}
      >
        {/* Cell header */}
        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-ak-border bg-ak-bg">
          <span className="font-mono text-xs text-ak-muted w-8 shrink-0">{gutter}</span>

          {isMarkdown && (
            <span className="text-xs px-1.5 py-0.5 rounded font-mono bg-purple-100 text-purple-700">
              md
            </span>
          )}

          {!isMarkdown && cell.backend && (
            <span
              className={`text-xs px-1.5 py-0.5 rounded font-mono ${
                cell.backend === 'server'
                  ? 'bg-ak-brand/10 text-ak-brand'
                  : 'bg-green-100 text-green-700'
              }`}
            >
              {cell.backend}
            </span>
          )}

          {!isMarkdown && elapsed !== null && (
            <span
              className={`font-mono text-xs tabular-nums ${
                isRunning ? 'text-ak-brand' : 'text-ak-muted'
              }`}
            >
              {isRunning && (
                <span className="cell-run-pulse inline-block w-1.5 h-1.5 rounded-full bg-ak-brand mr-1 align-middle" />
              )}
              {formatMs(elapsed)}
            </span>
          )}

          <div className="flex-1" />

          {!zenMode && (
            <div
              className={`flex items-center gap-1 transition-opacity ${
                showChrome ? 'opacity-100' : 'opacity-0 group-hover/cell:opacity-100'
              }`}
            >
              <CellActionsMenu
                isMarkdown={isMarkdown}
                onToggleType={() => onToggleCellType(cell.id)}
                onCopy={() => onCopyCell(cell.id)}
                onCut={() => onCutCell(cell.id)}
              />
              <CellBtn title="Move up" onClick={() => onMoveUp(cell.id)}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m18 15-6-6-6 6"/></svg>
              </CellBtn>
              <CellBtn title="Move down" onClick={() => onMoveDown(cell.id)}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m6 9 6 6 6-6"/></svg>
              </CellBtn>
              <CellBtn title="Delete cell" onClick={() => onDelete(cell.id)} danger>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>
              </CellBtn>
              {!isMarkdown && (
                isRunning ? (
                  <button
                    type="button"
                    onClick={() => onStop(cell.id)}
                    title="Stop execution"
                    className="flex items-center gap-1 rounded px-2.5 py-1 text-xs font-medium bg-ak-brand/15 text-ak-brand hover:bg-ak-brand/25 transition-colors"
                  >
                    <span className="relative flex h-3 w-3 items-center justify-center">
                      <span className="cell-run-pulse absolute inset-0 rounded-sm border border-ak-brand/60" />
                      <span className="relative h-2 w-2 rounded-sm bg-ak-brand" />
                    </span>
                    Stop
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => onRun(cell.id)}
                    title="Run cell (Ctrl+Enter)"
                    className="flex items-center gap-1 rounded px-2.5 py-1 text-xs font-medium bg-ak-brand text-white hover:bg-ak-brand-dark transition-colors"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                    Run
                  </button>
                )
              )}
            </div>
          )}

          {!isMarkdown && zenMode && isRunning && (
            <span className="cell-run-pulse flex h-3.5 w-3.5 items-center justify-center rounded-full border-2 border-ak-brand" aria-hidden />
          )}
        </div>

        {isMarkdown ? (
          <MarkdownCell
            source={cell.code}
            onChange={(src) => onCodeChange(cell.id, src)}
            onActivate={activateCell}
            shouldFocus={shouldFocus}
          />
        ) : (
          <div ref={cellBodyRef}>
            <CodeMirror
              ref={editorRef as never}
              value={cell.code}
              onChange={(code) => onCodeChange(cell.id, code)}
              extensions={editorExtensions}
              basicSetup={{
                lineNumbers: showLineNumbers,
                highlightActiveLine: showLineNumbers,
                highlightActiveLineGutter: showLineNumbers,
                autocompletion: true,
                indentOnInput: true,
                bracketMatching: true,
              }}
              style={{ fontSize: '0.875rem' }}
              className="overflow-hidden"
            />
            <Output
              items={cell.outputs}
              gutterWidth={gutterWidth}
              zenMode={zenMode}
              onItemsChange={(outputs) => onOutputsChange?.(cell.id, outputs)}
            />
          </div>
        )}
      </div>

      {showInsertBar && !zenMode && (
        <CellInsertBar
          visible={isActive}
          onAddCode={() => onAddBelow(cell.id, 'code')}
          onAddMarkdown={() => onAddBelow(cell.id, 'markdown')}
        />
      )}
    </div>
  );
}

function CellInsertBar({
  visible,
  onAddCode,
  onAddMarkdown,
}: {
  visible: boolean;
  onAddCode: () => void;
  onAddMarkdown: () => void;
}) {
  return (
    <div
      className={`flex h-5 items-center justify-center gap-2 transition-opacity ${
        visible ? 'opacity-100' : 'opacity-0 group-hover/cell:opacity-100'
      }`}
    >
      <div className="group/addcode relative">
        <button
          type="button"
          onClick={onAddCode}
          title={ADD_CODE_CELL_SHORTCUT_HINT}
          className="flex items-center gap-1 rounded border border-transparent px-2 py-0.5 text-xs text-ak-muted transition-colors hover:border-ak-border hover:bg-ak-code-bg hover:text-ak-fg"
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14"/></svg>
          Add code
        </button>
        <span
          role="tooltip"
          className="pointer-events-none absolute left-1/2 top-full z-10 mt-1 -translate-x-1/2 whitespace-nowrap rounded border border-ak-border bg-ak-bg px-2 py-1 text-[10px] text-ak-muted opacity-0 shadow-sm transition-opacity group-hover/addcode:opacity-100"
        >
          {ADD_CODE_CELL_SHORTCUT_HINT}
        </span>
      </div>
      <button
        type="button"
        onClick={onAddMarkdown}
        className="flex items-center gap-1 rounded border border-transparent px-2 py-0.5 text-xs text-ak-muted transition-colors hover:border-ak-border hover:bg-ak-code-bg hover:text-ak-fg"
      >
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 6h16M4 12h8M4 18h16"/></svg>
        Add markdown
      </button>
    </div>
  );
}

function MarkdownCell({
  source,
  onChange,
  onActivate,
  shouldFocus,
}: {
  source: string;
  onChange: (src: string) => void;
  onActivate?: () => void;
  shouldFocus?: boolean;
}) {
  const [editing, setEditing] = useState(!source.trim());
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (shouldFocus) setEditing(true);
  }, [shouldFocus]);

  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      autoResize(textareaRef.current);
    }
  }, [editing, shouldFocus]);

  function autoResize(el: HTMLTextAreaElement) {
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }

  if (editing) {
    return (
      <textarea
        ref={textareaRef}
        value={source}
        onChange={(e) => { onChange(e.target.value); autoResize(e.target); }}
        onFocus={() => onActivate?.()}
        onBlur={() => { if (source.trim()) setEditing(false); }}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault();
            if (source.trim()) setEditing(false);
          }
        }}
        placeholder="Write markdown here… (Ctrl+Enter to preview)"
        className="w-full resize-none bg-ak-code-bg px-4 py-3 text-sm text-ak-fg outline-none font-mono leading-relaxed placeholder:text-ak-muted/50"
        style={{ minHeight: '80px' }}
      />
    );
  }

  return (
    <div
      onClick={() => {
        onActivate?.();
        setEditing(true);
      }}
      title="Click to edit"
      className="px-5 py-3 cursor-text prose prose-sm max-w-none hover:bg-ak-code-bg/30 transition-colors"
    >
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {source}
      </ReactMarkdown>
    </div>
  );
}

function CellBtn({
  title,
  onClick,
  danger,
  children,
}: {
  title: string;
  onClick: () => void;
  danger?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`rounded p-1 transition-colors ${
        danger
          ? 'text-ak-muted hover:text-red-500 hover:bg-red-50'
          : 'text-ak-muted hover:text-ak-fg hover:bg-ak-code-bg'
      }`}
    >
      {children}
    </button>
  );
}
