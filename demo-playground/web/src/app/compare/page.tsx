'use client';

import { useCallback, useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { readZenFromUrl } from '@/lib/recording';

const Notebook = dynamic(() => import('@/components/notebook/Notebook'), { ssr: false });

export default function ComparePage() {
  const [zenMode] = useState(readZenFromUrl);
  const [leftReady, setLeftReady] = useState(false);
  const [rightReady, setRightReady] = useState(false);

  const markLeftReady = useCallback(() => setLeftReady(true), []);
  const markRightReady = useCallback(() => setRightReady(true), []);

  useEffect(() => {
    if (leftReady && rightReady) {
      document.documentElement.setAttribute('data-recording-ready', 'true');
    }
  }, [leftReady, rightReady]);

  return (
    <div className="min-h-screen bg-ak-bg">
      <div className="grid grid-cols-2 divide-x divide-ak-border">
        <section className="min-w-0">
          <header className="border-b border-ak-border bg-ak-code-bg/40 px-4 py-2">
            <h2 className="text-sm font-semibold text-ak-fg">Alkahest</h2>
            <p className="text-xs text-ak-muted">Rust F5B Groebner basis</p>
          </header>
          <Notebook zenMode demoParam="left" compact onReady={markLeftReady} />
        </section>
        <section className="min-w-0">
          <header className="border-b border-ak-border bg-ak-code-bg/40 px-4 py-2">
            <h2 className="text-sm font-semibold text-ak-fg">SymPy</h2>
            <p className="text-xs text-ak-muted">Pure Python reference</p>
          </header>
          <Notebook zenMode demoParam="right" compact onReady={markRightReady} />
        </section>
      </div>
    </div>
  );
}
