'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { NotebookCommandDef, NotebookCommandId } from '@/lib/notebook-commands';

export interface PaletteCommand extends NotebookCommandDef {
  disabled?: boolean;
  run: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  commands: PaletteCommand[];
}

export default function CommandPalette({ open, onClose, commands }: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter((cmd) => {
      const hay = `${cmd.label} ${cmd.group} ${cmd.keywords ?? ''} ${cmd.shortcut ?? ''}`.toLowerCase();
      return hay.includes(q);
    });
  }, [commands, query]);

  useEffect(() => {
    if (!open) return;
    setQuery('');
    setActiveIndex(0);
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [open]);

  useEffect(() => {
    setActiveIndex((i) => (filtered.length === 0 ? 0 : Math.min(i, filtered.length - 1)));
  }, [filtered.length]);

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }
      if (filtered.length === 0) return;
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIndex((i) => (i + 1) % filtered.length);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIndex((i) => (i - 1 + filtered.length) % filtered.length);
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        const cmd = filtered[activeIndex];
        if (cmd && !cmd.disabled) {
          cmd.run();
          onClose();
        }
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, filtered, activeIndex, onClose]);

  useEffect(() => {
    const el = listRef.current?.children[activeIndex] as HTMLElement | undefined;
    el?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex, filtered]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/30 px-4 pt-[12vh]"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-label="Command palette"
        className="w-full max-w-lg overflow-hidden rounded-lg border border-ak-border bg-ak-bg shadow-xl"
      >
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search commands…"
          className="w-full border-b border-ak-border bg-transparent px-4 py-3 text-sm outline-none placeholder:text-ak-muted"
          autoComplete="off"
          spellCheck={false}
        />
        <div ref={listRef} className="max-h-72 overflow-y-auto py-1">
          {filtered.length === 0 ? (
            <p className="px-4 py-6 text-center text-xs text-ak-muted">No matching commands</p>
          ) : (
            filtered.map((cmd, i) => (
              <button
                key={cmd.id}
                type="button"
                disabled={cmd.disabled}
                onMouseEnter={() => setActiveIndex(i)}
                onClick={() => {
                  if (!cmd.disabled) {
                    cmd.run();
                    onClose();
                  }
                }}
                className={`flex w-full items-center gap-3 px-4 py-2 text-left text-sm transition-colors ${
                  i === activeIndex ? 'bg-ak-code-bg' : 'hover:bg-ak-code-bg/60'
                } disabled:cursor-not-allowed disabled:opacity-40`}
              >
                <span className="flex-1">{cmd.label}</span>
                {cmd.shortcut && (
                  <kbd className="shrink-0 rounded border border-ak-border bg-ak-bg px-1.5 py-0.5 font-mono text-[10px] text-ak-muted">
                    {cmd.shortcut}
                  </kbd>
                )}
              </button>
            ))
          )}
        </div>
        <div className="flex items-center justify-between border-t border-ak-border px-4 py-2 text-[10px] text-ak-muted">
          <span>↑↓ navigate · Enter run · Esc close</span>
        </div>
      </div>
    </div>
  );
}

export type { NotebookCommandId };
