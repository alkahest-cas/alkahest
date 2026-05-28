'use client';

import { useEffect, useRef, useState } from 'react';
import type { OutputItem } from '@/lib/execution';
import LeanCertificate from './LeanCertificate';

interface OutputProps {
  items: OutputItem[];
  /** When set, Lean verify state updates propagate back (notebook cells). */
  onItemsChange?: (items: OutputItem[]) => void;
}

export default function Output({ items, onItemsChange }: OutputProps) {
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

  return (
    <div className="border-t border-ak-border bg-ak-code-bg px-4 py-2 space-y-1">
      {localItems.map((item, i) => (
        <OutputItemView
          key={i}
          item={item}
          onLeanUpdate={item.type === 'lean' ? (u) => handleLeanUpdate(i, u) : undefined}
        />
      ))}
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
    return <MixedTextOutput text={item.text} stream={item.stream} />;
  }

  if (item.type === 'latex') {
    return <LatexBlock latex={item.latex} />;
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

type TextSegment = { kind: 'text'; content: string } | { kind: 'latex'; content: string };

function splitMixedText(text: string): TextSegment[] {
  const lines = text.split('\n');
  const segments: TextSegment[] = [];
  let buf: string[] = [];

  const flushBuf = () => {
    if (buf.length > 0) {
      segments.push({ kind: 'text', content: buf.join('\n') });
      buf = [];
    }
  };

  for (const line of lines) {
    const t = line.trim();
    // Match a line that is entirely a $$...$$ block (display math)
    if (t.startsWith('$$') && t.endsWith('$$') && t.length > 4) {
      flushBuf();
      segments.push({ kind: 'latex', content: t.slice(2, -2).trim() });
    } else {
      buf.push(line);
    }
  }
  flushBuf();
  return segments;
}

function MixedTextOutput({ text, stream }: { text: string; stream: 'stdout' | 'stderr' }) {
  const preClass = `whitespace-pre-wrap font-mono text-sm leading-relaxed ${
    stream === 'stderr' ? 'text-red-600' : 'text-ak-fg'
  }`;

  const segments = splitMixedText(text);

  // No LaTeX found — render original <pre> unchanged
  if (segments.length === 1 && segments[0].kind === 'text') {
    return <pre className={preClass}>{text}</pre>;
  }

  return (
    <div>
      {segments.map((seg, i) =>
        seg.kind === 'latex' ? (
          <LatexBlock key={i} latex={seg.content} />
        ) : seg.content !== '' ? (
          <pre key={i} className={preClass}>{seg.content}</pre>
        ) : null,
      )}
    </div>
  );
}

function LatexBlock({ latex }: { latex: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const render = async () => {
      if (!ref.current) return;
      try {
        const katex = (await import('katex')).default;
        const clean = latex.replace(/^\$\$?|\$\$?$/g, '').trim();
        katex.render(clean, ref.current, {
          displayMode: true,
          throwOnError: false,
          output: 'html',
        });
      } catch {
        if (ref.current) ref.current.textContent = latex;
      }
    };
    render();
  }, [latex]);

  return <div ref={ref} className="py-1 overflow-x-auto" />;
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
