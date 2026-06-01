'use client';

import { useEffect, useState } from 'react';
import { isStaticHosting } from '@/lib/hosting';

const DISMISS_KEY = 'alkahest-playground-banner-dismissed';

export default function HostedBanner({ variant = 'notebook' }: { variant?: 'notebook' | 'agent' }) {
  const [dismissed, setDismissed] = useState(true);

  useEffect(() => {
    setDismissed(localStorage.getItem(DISMISS_KEY) === '1');
  }, []);

  if (!isStaticHosting || dismissed) return null;

  function dismiss() {
    localStorage.setItem(DISMISS_KEY, '1');
    setDismissed(true);
  }

  if (variant === 'agent') {
    return (
      <div className="mx-auto max-w-3xl px-4 pt-4">
        <div className="relative rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 pr-10 text-sm text-amber-950">
          <BannerClose onClose={dismiss} />
          <p className="font-medium">Hosted demo — agent needs a backend</p>
          <p className="mt-1 text-amber-900/90">
            The AI agent calls a server API that is not available on GitHub Pages. Run the playground locally
            with <code className="rounded bg-white/80 px-1">pnpm start</code>, or self-host the full app.
            You can still use the{' '}
            <a href="../" className="underline font-medium">notebook</a> in WASM mode (pure Python).
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-4 pt-4">
      <div className="relative rounded-lg border border-ak-border bg-ak-code-bg/60 px-4 py-3 pr-10 text-sm">
        <BannerClose onClose={dismiss} />
        <p className="font-medium">Try Alkahest in your browser</p>
        <p className="mt-1 text-ak-muted">
          Pure Python cells run in <strong>WASM</strong>. For{' '}
          <code className="rounded bg-ak-bg px-1">import alkahest</code>, open settings (
          <kbd className="rounded border border-ak-border px-1 text-xs">Ctrl+/</kbd>) and add your execution
          server or Jupyter URL + token.
        </p>
      </div>
    </div>
  );
}

function BannerClose({ onClose }: { onClose: () => void }) {
  return (
    <button
      type="button"
      onClick={onClose}
      aria-label="Dismiss banner"
      className="absolute right-2 top-2 rounded p-1 text-ak-muted transition-colors hover:bg-ak-bg hover:text-ak-fg"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 6 6 18M6 6l12 12" />
      </svg>
    </button>
  );
}
