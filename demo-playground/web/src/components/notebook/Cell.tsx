'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { EditorView } from '@codemirror/view';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type { OutputItem } from '@/lib/execution';
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
  index: number;
  onCodeChange: (id: string, code: string) => void;
  onRun: (id: string) => void;
  onDelete: (id: string) => void;
  onMoveUp: (id: string) => void;
  onMoveDown: (id: string) => void;
  onAddBelow: (id: string) => void;
  onToggleCellType: (id: string) => void;
  onOutputsChange?: (id: string, outputs: OutputItem[]) => void;
  zenMode?: boolean;
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
      // Keep the final elapsed time when done/error; reset on idle
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
  cell, index, onCodeChange, onRun, onDelete, onMoveUp, onMoveDown, onAddBelow, onToggleCellType, onOutputsChange, zenMode,
}: CellProps) {
  const editorRef = useRef<{ view?: { focus: () => void } }>(null);
  const elapsed = useElapsedTimer(cell.status);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        onRun(cell.id);
      }
    },
    [cell.id, onRun],
  );

  const isMarkdown = cell.cellType === 'markdown';
  const gutter = isMarkdown ? '[M]' : cell.executionCount !== null
    ? `[${cell.executionCount}]`
    : cell.status === 'running' ? '[*]' : '[ ]';

  const hasError = cell.outputs.some((o) => o.type === 'error');

  return (
    <div
      data-cell-id={cell.id}
      className={`group relative overflow-hidden rounded-lg border transition-all ${
        isMarkdown
          ? 'border-ak-border hover:border-ak-muted/40'
          : cell.status === 'running'
          ? 'border-ak-brand shadow-sm'
          : hasError
          ? 'border-red-200'
          : 'border-ak-border hover:border-ak-muted/40'
      }`}
    >
      {/* Cell header */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-ak-border bg-ak-bg">
        {/* Gutter indicator */}
        <span className="font-mono text-xs text-ak-muted w-8 shrink-0">{gutter}</span>

        {/* Markdown badge */}
        {isMarkdown && (
          <span className="text-xs px-1.5 py-0.5 rounded font-mono bg-purple-100 text-purple-700">
            md
          </span>
        )}

        {/* Backend badge (code cells only) */}
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

        {/* Elapsed timer — code cells only */}
        {!isMarkdown && elapsed !== null && (
          <span
            className={`font-mono text-xs tabular-nums ${
              cell.status === 'running' ? 'text-ak-brand' : 'text-ak-muted'
            }`}
          >
            {cell.status === 'running' && (
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-ak-brand animate-pulse mr-1 align-middle" />
            )}
            {formatMs(elapsed)}
          </span>
        )}

        <div className="flex-1" />

        {/* Cell controls — hidden in zen mode */}
        {!zenMode && (
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <CellBtn
              title={isMarkdown ? 'Switch to code' : 'Switch to markdown'}
              onClick={() => onToggleCellType(cell.id)}
            >
              {isMarkdown ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 6h16M4 12h8M4 18h16"/></svg>
              )}
            </CellBtn>
            <CellBtn title="Move up" onClick={() => onMoveUp(cell.id)}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m18 15-6-6-6 6"/></svg>
            </CellBtn>
            <CellBtn title="Move down" onClick={() => onMoveDown(cell.id)}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m6 9 6 6 6-6"/></svg>
            </CellBtn>
            <CellBtn title="Add cell below" onClick={() => onAddBelow(cell.id)}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14"/></svg>
            </CellBtn>
            <CellBtn title="Delete cell" onClick={() => onDelete(cell.id)} danger>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>
            </CellBtn>
          </div>
        )}

        {/* Run button — code cells only, hidden in zen mode */}
        {!isMarkdown && !zenMode && (
          <button
            onClick={() => onRun(cell.id)}
            disabled={cell.status === 'running'}
            title="Run cell (⌘ Enter)"
            className={`flex items-center gap-1 rounded px-2.5 py-1 text-xs font-medium transition-all ${
              cell.status === 'running'
                ? 'bg-ak-brand/20 text-ak-brand cursor-wait'
                : 'bg-ak-brand text-white hover:bg-ak-brand-dark'
            }`}
          >
            {cell.status === 'running' ? (
              <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            )}
            Run
          </button>
        )}

        {/* Zen-mode: show status dot only for code cells */}
        {!isMarkdown && zenMode && cell.status === 'running' && (
          <svg className="animate-spin text-ak-brand" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
          </svg>
        )}
      </div>

      {/* Body */}
      {isMarkdown ? (
        <MarkdownCell
          source={cell.code}
          onChange={(src) => onCodeChange(cell.id, src)}
        />
      ) : (
        <>
          <div onKeyDown={handleKeyDown}>
            <CodeMirror
              ref={editorRef as never}
              value={cell.code}
              onChange={(code) => onCodeChange(cell.id, code)}
              extensions={[python(), warmLightTheme]}
              basicSetup={{
                lineNumbers: true,
                highlightActiveLine: true,
                highlightActiveLineGutter: true,
                autocompletion: true,
                indentOnInput: true,
                bracketMatching: true,
              }}
              style={{ fontSize: '0.875rem' }}
              className="overflow-hidden"
            />
          </div>
          <Output
            items={cell.outputs}
            onItemsChange={(outputs) => onOutputsChange?.(cell.id, outputs)}
          />
        </>
      )}
    </div>
  );
}

function MarkdownCell({ source, onChange }: { source: string; onChange: (src: string) => void }) {
  const [editing, setEditing] = useState(!source.trim());
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      autoResize(textareaRef.current);
    }
  }, [editing]);

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
        onBlur={() => { if (source.trim()) setEditing(false); }}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault();
            if (source.trim()) setEditing(false);
          }
        }}
        placeholder="Write markdown here… (⌘ Enter to preview)"
        className="w-full resize-none bg-ak-code-bg px-4 py-3 text-sm text-ak-fg outline-none font-mono leading-relaxed placeholder:text-ak-muted/50"
        style={{ minHeight: '80px' }}
      />
    );
  }

  return (
    <div
      onClick={() => setEditing(true)}
      title="Click to edit"
      className="px-5 py-3 cursor-text prose prose-sm max-w-none hover:bg-ak-code-bg/30 transition-colors"
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}

function CellBtn({
  title, onClick, danger, children,
}: { title: string; onClick: () => void; danger?: boolean; children: React.ReactNode }) {
  return (
    <button
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
