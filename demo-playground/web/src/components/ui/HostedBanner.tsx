'use client';

import { isStaticHosting } from '@/lib/hosting';

export default function HostedBanner({ variant = 'notebook' }: { variant?: 'notebook' | 'agent' }) {
  if (!isStaticHosting) return null;

  if (variant === 'agent') {
    return (
      <div className="mx-auto max-w-3xl px-4 pt-4">
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
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
      <div className="rounded-lg border border-ak-border bg-ak-code-bg/60 px-4 py-3 text-sm">
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
