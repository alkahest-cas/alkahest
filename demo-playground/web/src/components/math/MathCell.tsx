'use client';

import { useState } from 'react';
import clsx from 'clsx';
import LeanCertificate from '@/components/notebook/LeanCertificate';
import type { MathResult } from '@/lib/math-api';
import { verificationBadge } from '@/lib/math-api';
import Tex from './Tex';

export interface MathCellData {
  id: string;
  input: string;
  status: 'idle' | 'running' | 'done';
  result?: MathResult;
  /** The input the result belongs to (a result goes stale when the input is edited). */
  ranInput?: string;
  preview?: MathResult | null;
  /** The result was computed when the page was built (static site, no backend). */
  precomputed?: boolean;
}

interface MathCellProps {
  cell: MathCellData;
  index: number;
  canRun: boolean;
  /** Alkahest version the precomputed results were made with. */
  precomputedVersion?: string;
  inputRef: (el: HTMLInputElement | null) => void;
  onChange: (value: string) => void;
  onRun: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onFocus: () => void;
  onSelect: (start: number, end: number) => void;
  onDelete: () => void;
}

export default function MathCell({
  cell,
  index,
  canRun,
  precomputedVersion,
  inputRef,
  onChange,
  onRun,
  onKeyDown,
  onFocus,
  onSelect,
  onDelete,
}: MathCellProps) {
  const stale = cell.result != null && cell.ranInput !== cell.input;
  const result = cell.result;
  const interpretation =
    !stale && result?.interpreted_latex
      ? result.interpreted_latex
      : cell.preview?.ok
        ? cell.preview.interpreted_latex
        : null;
  const previewError =
    (!result || stale) && !cell.preview?.ok && cell.preview?.error_message && cell.input.trim()
      ? cell.preview
      : null;

  return (
    <div
      data-math-cell={index + 1}
      data-status={cell.status}
      className={clsx(
        'group/cell rounded-lg border bg-ak-elev transition-colors',
        'border-ak-border focus-within:border-ak-border-strong',
      )}
    >
      <div className="flex items-center gap-2 px-3 py-2 sm:gap-3 sm:px-4">
        <span
          className={clsx(
            'w-5 shrink-0 select-none text-right font-mono text-xs tabular-nums',
            cell.status === 'running' ? 'text-ak-run cell-run-pulse' : 'text-ak-muted',
          )}
          aria-hidden
        >
          {index + 1}
        </span>
        <input
          ref={inputRef}
          type="text"
          value={cell.input}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          onFocus={onFocus}
          onSelect={(e) => {
            const el = e.currentTarget;
            onSelect(el.selectionStart ?? el.value.length, el.selectionEnd ?? el.value.length);
          }}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          autoComplete="off"
          enterKeyHint="go"
          aria-label={`Math input ${index + 1}`}
          placeholder={index === 0 ? 'integrate x^2 dx' : 'Type maths, then Enter'}
          style={{ fontVariantLigatures: 'none' }}
          className="min-w-0 flex-1 bg-transparent py-1.5 font-mono text-[15px] text-ak-fg outline-none placeholder:text-ak-muted/60"
        />
        <button
          type="button"
          onClick={onDelete}
          title="Delete cell"
          aria-label={`Delete cell ${index + 1}`}
          className="rounded p-1 text-ak-muted opacity-0 transition hover:bg-ak-sunk hover:text-ak-fg focus:opacity-100 group-hover/cell:opacity-100 group-focus-within/cell:opacity-100"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
        <button
          type="button"
          onClick={onRun}
          disabled={!canRun || cell.status === 'running' || !cell.input.trim()}
          title={canRun ? 'Run (Enter)' : 'Start the local server (see above) to run this'}
          aria-label={`Run cell ${index + 1}`}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-ak-border text-ak-fg2 transition-colors hover:border-ak-border-strong hover:text-ak-fg disabled:opacity-40"
        >
          {cell.status === 'running' ? (
            <span className="h-3 w-3 animate-spin rounded-full border-2 border-ak-muted border-t-transparent" />
          ) : (
            <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
              <path d="M7 4.5v15l13-7.5z" />
            </svg>
          )}
        </button>
      </div>

      {(interpretation || previewError) && (
        <div className="flex min-h-[28px] items-baseline gap-2 overflow-x-auto px-3 pb-2 pl-10 text-[13px] text-ak-muted no-scrollbar sm:pl-12">
          <span className="shrink-0 text-xs">{previewError ? 'Not understood:' : 'Interpreted as'}</span>
          {previewError ? (
            <span className="text-xs text-ak-fg2">{previewError.error_message}</span>
          ) : (
            <Tex latex={interpretation as string} className="whitespace-nowrap text-ak-fg2" />
          )}
        </div>
      )}

      {result && (
        <div
          className={clsx(
            'border-t border-ak-border px-3 py-2.5 sm:pl-12 sm:pr-4',
            stale && 'opacity-50',
            cell.precomputed && 'flex items-start gap-2',
          )}
        >
          <div className={clsx(cell.precomputed && 'min-w-0 flex-1')}>
            {result.ok ? <ResultView result={result} /> : <RefusalView result={result} />}
          </div>
          {cell.precomputed && (
            <span
              className="mt-0.5 shrink-0 rounded-full border border-ak-border px-2 py-px text-[11px] text-ak-muted"
              data-testid="precomputed-badge"
              title={`Computed in advance with Alkahest ${precomputedVersion ?? ''} when this page was built. Start the local server to run it yourself.`}
            >
              precomputed
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function ResultView({ result }: { result: MathResult }) {
  const [showCert, setShowCert] = useState(false);
  const [copied, setCopied] = useState(false);
  const badge = verificationBadge(result);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <div className="min-w-0 max-w-full overflow-x-auto text-ak-fg">
          <Tex latex={result.latex ?? ''} display />
        </div>
        {result.approx && (
          <span className="font-mono text-sm text-ak-muted" title="Numerical value (rigorous ball arithmetic)">
            ≈ {result.approx}
          </span>
        )}
      </div>

      {result.side_conditions && result.side_conditions.length > 0 && (
        <p className="text-xs text-ak-muted">
          Valid when: <span className="font-mono">{result.side_conditions.join(', ')}</span>
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {badge && (
          <span
            className="rounded-full bg-ak-ok/10 px-2 py-0.5 font-mono text-[12px] text-ak-ok"
            title={badge.title}
          >
            {badge.label}
          </span>
        )}
        {result.certificate && (
          <button
            type="button"
            onClick={() => setShowCert((v) => !v)}
            aria-expanded={showCert}
            className="rounded-full border border-ak-ok/40 px-2 py-0.5 font-mono text-[12px] text-ak-ok transition-colors hover:bg-ak-ok/10"
            title="Alkahest generated a Lean 4 proof of this step. Generated, not yet checked: use Verify in Lean."
          >
            Lean certificate {showCert ? '▴' : '▾'}
          </button>
        )}
        {result.notes?.map((n) => (
          <span key={n} className="text-ak-muted">
            {n}
          </span>
        ))}
        <span className="flex-1" />
        {result.text && (
          <button
            type="button"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(result.text ?? '');
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              } catch {
                /* ignore */
              }
            }}
            className="text-ak-muted transition-colors hover:text-ak-fg"
            title="Copy the result as plain text"
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
        {result.elapsed_ms != null && (
          <span className="font-mono tabular-nums text-ak-muted" title="Time inside Alkahest">
            {formatMs(result.elapsed_ms)}
          </span>
        )}
      </div>

      {showCert && result.certificate && (
        <LeanCertificate
          item={{ type: 'lean', source: result.certificate, operation: result.kind, steps: result.steps }}
          defaultExpanded
        />
      )}
    </div>
  );
}

function RefusalView({ result }: { result: MathResult }) {
  const [open, setOpen] = useState(false);
  const source = result.error_source ?? 'playground';
  const overLimit = result.error_code === 'E-PLAYGROUND-002' || result.error_code === 'E-PLAYGROUND-004';
  const heading =
    source === 'alkahest'
      ? 'Alkahest declined'
      : overLimit
        ? "Over the playground's limits"
        : source === 'input'
          ? 'Could not read this line'
          : 'Stopped';
  const message = result.error_message ?? 'No details.';
  const more = [result.error_detail, result.error_hint].filter(Boolean) as string[];

  return (
    <div className="space-y-1.5" role="status">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={clsx(
            'text-sm font-medium',
            source === 'alkahest' ? 'text-ak-warn' : 'text-ak-fg',
          )}
        >
          {heading}
        </span>
        {result.error_code && (
          <span className="rounded-full bg-ak-sunk px-2 py-0.5 font-mono text-[12px] text-ak-fg2">
            {result.error_code}
          </span>
        )}
      </div>
      <p className="text-sm leading-relaxed text-ak-fg2">
        {source === 'alkahest' ? <>because {lowerFirst(message)}</> : message}
      </p>
      {source === 'alkahest' && (
        <p className="text-xs text-ak-muted">
          A coded refusal is an answer, not a crash: Alkahest returns an error rather than a result it cannot stand behind.
        </p>
      )}
      {more.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="text-xs text-ak-muted underline-offset-2 hover:text-ak-fg hover:underline"
          >
            {open ? 'Hide details' : source === 'alkahest' ? 'Details and what to try' : 'Details'}
          </button>
          {open && (
            <div className="mt-1.5 space-y-1.5 rounded-md bg-ak-sunk px-3 py-2 text-xs leading-relaxed text-ak-fg2">
              {more.map((m) => (
                <p key={m} className="whitespace-pre-wrap break-words">
                  {m}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function lowerFirst(s: string): string {
  if (!s) return s;
  // keep acronyms / symbols as they are
  if (/^[A-Z]{2}/.test(s)) return s;
  return s.charAt(0).toLowerCase() + s.slice(1);
}

function formatMs(ms: number): string {
  if (ms < 1) return '<1 ms';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}
