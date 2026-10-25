'use client';

import { useMemo } from 'react';
import katex from 'katex';

/** KaTeX-rendered LaTeX. `trust` stays off, so \href / \includegraphics are inert. */
export default function Tex({
  latex,
  display = false,
  className,
}: {
  latex: string;
  display?: boolean;
  className?: string;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(latex, {
        displayMode: display,
        throwOnError: false,
        strict: 'ignore',
        trust: false,
        maxSize: 20,
        maxExpand: 500,
        output: 'html',
      }),
    [latex, display],
  );
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
