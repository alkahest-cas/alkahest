'use client';

import { useEffect, useRef, useState } from 'react';

interface OutputActionsMenuProps {
  onCopy: () => void;
  onClear: () => void;
}

export default function OutputActionsMenu({ onCopy, onClear }: OutputActionsMenuProps) {
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

  function pick(action: () => void) {
    setOpen(false);
    action();
  }

  return (
    <div ref={rootRef} className="relative flex justify-center">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
        title="Output actions"
        className="rounded p-1 text-ak-muted transition-colors hover:bg-ak-code-bg/80 hover:text-ak-fg"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="5" r="1" fill="currentColor" />
          <circle cx="12" cy="12" r="1" fill="currentColor" />
          <circle cx="12" cy="19" r="1" fill="currentColor" />
        </svg>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute left-0 top-full z-30 mt-1 min-w-[148px] rounded-lg border border-ak-border bg-ak-bg py-1 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => pick(onCopy)}
            className="w-full px-3 py-1.5 text-left text-xs transition-colors hover:bg-ak-code-bg"
          >
            Copy output
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => pick(onClear)}
            className="w-full px-3 py-1.5 text-left text-xs transition-colors hover:bg-ak-code-bg"
          >
            Clear output
          </button>
        </div>
      )}
    </div>
  );
}
