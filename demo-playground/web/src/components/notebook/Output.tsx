'use client';

import { useEffect, useRef, useState } from 'react';
import type { OutputItem } from '@/lib/execution';
import MarkdownRender from '@/lib/markdown-render';
import { outputsToPlainText } from '@/lib/output-text';
import LeanCertificate from './LeanCertificate';
import OutputActionsMenu from './OutputActionsMenu';

interface OutputProps {
  items: OutputItem[];
  /** When set, Lean verify state updates propagate back (notebook cells). */
  onItemsChange?: (items: OutputItem[]) => void;
  /** Width of the CodeMirror line-number gutter (px), for alignment with code. */
  gutterWidth?: number;
  /** Zen/recording layout — no output menu or extra chrome. */
  zenMode?: boolean;
}

export default function Output({ items, onItemsChange, gutterWidth = 0, zenMode }: OutputProps) {
  const [localItems, setLocalItems] = useState(items);

  useEffect(() => {
    setLocalItems(items);
  }, [items]);

  if (localItems.length === 0) return null;

  const handleLeanUpdate = (index: number, updated: OutputItem) => {
    const next = localItems.map((it, i) => (i === index ? updated : it));
    setLocalItems(next);
    onItemsChange?.(next);
  };

  const handleClear = () => {
    setLocalItems([]);
    onItemsChange?.([]);
  };

  const handleCopy = async () => {
    const text = outputsToPlainText(localItems);
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      /* ignore */
    }
  };

  const hasGutter = gutterWidth > 0;
  const alignWithCode = hasGutter && zenMode;

  const outputItems = localItems.map((item, i) => (
    <OutputItemView
      key={i}
      item={item}
      onLeanUpdate={item.type === 'lean' ? (u) => handleLeanUpdate(i, u) : undefined}
    />
  ));

  if (zenMode) {
    return (
      <div className="border-t border-ak-border bg-ak-bg">
        <div className="flex min-w-0">
          {alignWithCode && (
            <div className="shrink-0" style={{ width: gutterWidth }} aria-hidden />
          )}
          <div
            className={`min-w-0 flex-1 space-y-2 py-2.5 pr-3 ${
              alignWithCode ? 'cell-output-content' : 'px-3'
            }`}
          >
            {outputItems}
          </div>
        </div>
      </div>
    );
  }

  const menu = (
    <OutputActionsMenu onCopy={() => void handleCopy()} onClear={handleClear} />
  );

  return (
    <div className="border-t border-ak-border bg-ak-bg">
      <div className="flex min-w-0">
        <div
          className={`shrink-0 flex items-start justify-center pt-1.5 ${
            hasGutter ? 'cell-output-gutter' : 'w-7'
          }`}
          style={hasGutter ? { width: gutterWidth } : undefined}
        >
          {menu}
        </div>
        <div className="cell-output-content min-w-0 flex-1 space-y-2 py-2.5 pr-3">
          {outputItems}
        </div>
      </div>
    </div>
  );
}

function OutputItemView({
  item,
  onLeanUpdate,
}: {
  item: OutputItem;
  onLeanUpdate?: (item: OutputItem) => void;
}) {
  if (item.type === 'text') {
    if (item.stream === 'stderr') {
      return (
        <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-red-600">
          {item.text}
        </pre>
      );
    }
    return <MarkdownRender source={item.text} />;
  }

  if (item.type === 'latex') {
    return <MarkdownRender source={`$$\n${item.latex}\n$$`} />;
  }

  if (item.type === 'html') {
    return <SafeHtml html={item.html} />;
  }

  if (item.type === 'image') {
    if (item.format === 'png') {
      return (
        <img
          src={`data:image/png;base64,${item.data}`}
          alt="output"
          className="max-w-full rounded"
          style={{ maxHeight: '480px' }}
        />
      );
    }
    if (item.format === 'svg') {
      return (
        <div
          className="max-w-full"
          dangerouslySetInnerHTML={{ __html: item.data }}
        />
      );
    }
  }

  if (item.type === 'lean') {
    return <LeanCertificate item={item} onUpdate={onLeanUpdate} />;
  }

  if (item.type === 'json') {
    return (
      <pre className="font-mono text-xs bg-ak-code-bg border border-ak-border rounded p-3 overflow-x-auto">
        {JSON.stringify(item.data, null, 2)}
      </pre>
    );
  }

  if (item.type === 'error') {
    return (
      <div className="rounded border border-red-200 bg-red-50 p-3">
        <p className="font-mono text-sm font-semibold text-red-700">
          {item.ename}: {item.evalue}
        </p>
        {item.traceback.length > 0 && (
          <pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-red-600">
            {item.traceback.join('\n').replace(/\x1b\[[0-9;]*m/g, '')}
          </pre>
        )}
      </div>
    );
  }

  return null;
}

function SafeHtml({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const render = async () => {
      if (!ref.current) return;
      const { default: DOMPurify } = await import('dompurify');
      ref.current.innerHTML = DOMPurify.sanitize(html);

      // Post-process any LaTeX in the HTML using KaTeX auto-render
      const katex = (await import('katex')).default;
      const renderMathInElement = (await import('katex/contrib/auto-render')).default;
      renderMathInElement(ref.current, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true },
        ],
        throwOnError: false,
      });
    };
    render();
  }, [html]);

  return <div ref={ref} className="text-sm" />;
}
