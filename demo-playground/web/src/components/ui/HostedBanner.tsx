'use client';

import { useEffect, useState } from 'react';
import { isStaticHosting } from '@/lib/hosting';

const DISMISS_KEY = 'alkahest-playground-banner-dismissed';

/** Python tab, hosted build only: what runs in the browser and what needs a server. */
export default function HostedBanner() {
  const [dismissed, setDismissed] = useState(true);

  useEffect(() => {
    try {
      setDismissed(localStorage.getItem(DISMISS_KEY) === '1');
    } catch {
      setDismissed(false);
    }
  }, []);

  if (!isStaticHosting || dismissed) return null;

  function dismiss() {
    try {
      localStorage.setItem(DISMISS_KEY, '1');
    } catch {
      /* ignore */
    }
    setDismissed(true);
  }

  return (
    <div className="mx-auto max-w-4xl px-4 pt-5">
      <div className="relative rounded-lg border border-ak-border bg-ak-elev px-4 py-3 pr-10 text-sm">
        <button
          type="button"
          onClick={dismiss}
          aria-label="Dismiss"
          className="absolute right-2 top-2 rounded p-1 text-ak-muted transition-colors hover:bg-ak-sunk hover:text-ak-fg"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
        <p className="font-medium text-ak-fg">Pure Python runs in your browser</p>
        <p className="mt-1 text-ak-fg2">
          Cells without <code className="rounded bg-ak-sunk px-1 font-mono text-[0.85em]">import alkahest</code> run
          in WebAssembly (Pyodide, with SymPy). Alkahest itself needs a backend: open settings (
          <kbd className="rounded border border-ak-border px-1 text-xs">Ctrl+/</kbd>) and add an Alkahest
          server or Jupyter URL and token.
        </p>
      </div>
    </div>
  );
}
