'use client';

import { useEffect, useRef, useState } from 'react';

export type RunMenuAction =
  | 'run-all'
  | 'restart'
  | 'restart-run-all'
  | 'run-below'
  | 'interrupt'
  | 'clear-outputs';

interface RunMenuProps {
  onAction: (action: RunMenuAction) => void;
  anyRunning: boolean;
}

const ITEMS: { action: RunMenuAction; label: string; dividerBefore?: boolean }[] = [
  { action: 'run-all', label: 'Run all' },
  { action: 'restart', label: 'Restart session', dividerBefore: true },
  { action: 'restart-run-all', label: 'Restart session and run all' },
  { action: 'run-below', label: 'Run focused cell and all below' },
  { action: 'interrupt', label: 'Interrupt execution', dividerBefore: true },
  { action: 'clear-outputs', label: 'Clear outputs' },
];

export default function RunMenu({ onAction, anyRunning }: RunMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: MouseEvent) {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  function pick(action: RunMenuAction) {
    setOpen(false);
    onAction(action);
  }

  return (
    <div ref={rootRef} className="relative">
      <div className="flex rounded border border-ak-border overflow-hidden">
        <button
          type="button"
          onClick={() => onAction('run-all')}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs hover:bg-ak-code-bg transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>
          Run all
        </button>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-haspopup="menu"
          className="flex items-center border-l border-ak-border px-1.5 py-1.5 text-xs text-ak-muted hover:bg-ak-code-bg hover:text-ak-fg transition-colors"
          title="More run options"
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="m6 9 6 6 6-6" />
          </svg>
        </button>
      </div>

      {open && (
        <div
          role="menu"
          className="absolute left-0 top-full z-30 mt-1 min-w-[240px] rounded-lg border border-ak-border bg-ak-bg py-1 shadow-lg"
        >
          {ITEMS.map(({ action, label, dividerBefore }) => (
            <div key={action}>
              {dividerBefore && <div className="my-1 border-t border-ak-border" />}
              <button
                type="button"
                role="menuitem"
                disabled={action === 'interrupt' && !anyRunning}
                onClick={() => pick(action)}
                className="w-full px-3 py-1.5 text-left text-xs transition-colors hover:bg-ak-code-bg disabled:cursor-not-allowed disabled:opacity-40"
              >
                {label}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
