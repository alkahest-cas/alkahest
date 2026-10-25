'use client';

import { useCallback, useEffect, useState } from 'react';
import { loadConfig } from '@/components/ui/Settings';
import type { OutputItem } from '@/lib/execution';
import { applyVerifyResult, fetchLeanStatus, verifyLeanCertificate } from '@/lib/lean';
import { connectionFromConfig } from '@/lib/server-connection';

interface LeanCertificateProps {
  item: Extract<OutputItem, { type: 'lean' }>;
  onUpdate?: (item: OutputItem) => void;
  /** Show the Lean source open (default) or collapsed behind the title. */
  defaultExpanded?: boolean;
}

export default function LeanCertificate({ item, onUpdate, defaultExpanded = true }: LeanCertificateProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [verifying, setVerifying] = useState(false);
  const [localItem, setLocalItem] = useState(item);
  const [leanAvailable, setLeanAvailable] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setLocalItem(item);
  }, [item]);

  useEffect(() => {
    const conn = connectionFromConfig(loadConfig());
    fetchLeanStatus(conn).then((s) => setLeanAvailable(s?.available ?? false));
  }, []);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(localItem.source);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [localItem.source]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([localItem.source], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alkahest-proof-${Date.now()}.lean`;
    a.click();
    URL.revokeObjectURL(url);
  }, [localItem.source]);

  const handleVerify = useCallback(async () => {
    setVerifying(true);
    try {
      const conn = connectionFromConfig(loadConfig());
      const result = await verifyLeanCertificate(conn, localItem.source);
      const updated = applyVerifyResult(localItem, result);
      if (updated.type === 'lean') {
        setLocalItem(updated);
        onUpdate?.(updated);
      }
    } catch (e) {
      const updated = {
        ...localItem,
        verified: 'fail' as const,
        verifyLog: String(e),
      };
      setLocalItem(updated);
      onUpdate?.(updated);
    } finally {
      setVerifying(false);
    }
  }, [localItem, onUpdate]);

  const title =
    localItem.operation != null
      ? `Lean 4 certificate (${localItem.operation})`
      : 'Lean 4 certificate';

  return (
    <div className="overflow-hidden rounded-lg border border-ak-ok/30 bg-ak-elev">
      <div className="flex flex-wrap items-center gap-2 border-b border-ak-ok/20 bg-ak-ok/[0.06] px-3 py-2">
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="text-xs font-semibold text-ak-ok hover:opacity-80"
        >
          {expanded ? '▼' : '▶'} {title}
          {localItem.steps != null && (
            <span className="ml-1.5 font-normal text-ak-muted">· {localItem.steps} steps</span>
          )}
        </button>
        {localItem.verified === 'ok' && (
          <span className="rounded-full bg-ak-ok/10 px-2 py-0.5 font-mono text-[12px] text-ak-ok">
            verified
          </span>
        )}
        {localItem.verified === 'fail' && (
          <span className="rounded-full bg-ak-brand/10 px-2 py-0.5 font-mono text-[12px] text-ak-brand">
            check failed
          </span>
        )}
        <div className="ml-auto flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={handleCopy}
            className="rounded border border-ak-border px-2 py-0.5 text-xs text-ak-fg2 hover:border-ak-border-strong hover:text-ak-fg"
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button
            type="button"
            onClick={handleDownload}
            className="rounded border border-ak-border px-2 py-0.5 text-xs text-ak-fg2 hover:border-ak-border-strong hover:text-ak-fg"
          >
            Download .lean
          </button>
          <button
            type="button"
            onClick={handleVerify}
            disabled={verifying || leanAvailable === false}
            title={
              leanAvailable === false
                ? 'Install Lean 4 + Mathlib (see demo-playground README)'
                : 'Typecheck with lake env lean'
            }
            className="rounded bg-ak-ok px-2 py-0.5 text-xs text-ak-elev hover:opacity-90 disabled:opacity-40"
          >
            {verifying ? 'Checking…' : 'Verify in Lean'}
          </button>
        </div>
      </div>
      {expanded && (
        <pre className="max-h-80 overflow-auto whitespace-pre bg-ak-sunk/60 px-3 py-2 font-mono text-xs text-ak-fg">
          {localItem.source}
        </pre>
      )}
      {localItem.verifyLog && (
        <pre
          className={`whitespace-pre-wrap border-t border-ak-ok/20 px-3 py-2 font-mono text-xs ${
            localItem.verified === 'ok' ? 'bg-ak-ok/5 text-ak-ok' : 'bg-ak-brand/5 text-ak-brand'
          }`}
        >
          {localItem.verifyLog}
        </pre>
      )}
      {leanAvailable === false && (
        <p className="border-t border-ak-ok/20 px-3 py-1.5 text-xs text-ak-muted">
          Lean verifier not available on this server. Run locally:{' '}
          <code className="font-mono">cd lean && lake env lean proof.lean</code>
        </p>
      )}
    </div>
  );
}
