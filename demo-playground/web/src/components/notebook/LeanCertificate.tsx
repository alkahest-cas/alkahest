'use client';

import { useCallback, useEffect, useState } from 'react';
import { loadConfig } from '@/components/ui/Settings';
import type { OutputItem } from '@/lib/execution';
import { applyVerifyResult, fetchLeanStatus, verifyLeanCertificate } from '@/lib/lean';
import { connectionFromConfig } from '@/lib/server-connection';

interface LeanCertificateProps {
  item: Extract<OutputItem, { type: 'lean' }>;
  onUpdate?: (item: OutputItem) => void;
}

export default function LeanCertificate({ item, onUpdate }: LeanCertificateProps) {
  const [expanded, setExpanded] = useState(true);
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
    <div className="rounded-lg border border-emerald-200/80 bg-emerald-50/40 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-3 py-2 border-b border-emerald-200/60 bg-emerald-50/80">
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="text-xs font-semibold text-emerald-900 hover:text-emerald-700"
        >
          {expanded ? '▼' : '▶'} {title}
          {localItem.steps != null && (
            <span className="ml-1.5 font-normal text-emerald-700">· {localItem.steps} steps</span>
          )}
        </button>
        {localItem.verified === 'ok' && (
          <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-green-100 text-green-800">
            verified
          </span>
        )}
        {localItem.verified === 'fail' && (
          <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-red-100 text-red-800">
            check failed
          </span>
        )}
        <div className="ml-auto flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={handleCopy}
            className="text-xs px-2 py-0.5 rounded border border-emerald-300/80 hover:bg-white/60"
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button
            type="button"
            onClick={handleDownload}
            className="text-xs px-2 py-0.5 rounded border border-emerald-300/80 hover:bg-white/60"
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
            className="text-xs px-2 py-0.5 rounded bg-emerald-700 text-white hover:bg-emerald-800 disabled:opacity-40"
          >
            {verifying ? 'Checking…' : 'Verify in Lean'}
          </button>
        </div>
      </div>
      {expanded && (
        <pre className="max-h-80 overflow-auto px-3 py-2 text-xs font-mono text-emerald-950 whitespace-pre bg-white/50">
          {localItem.source}
        </pre>
      )}
      {localItem.verifyLog && (
        <pre
          className={`px-3 py-2 text-xs font-mono border-t border-emerald-200/60 whitespace-pre-wrap ${
            localItem.verified === 'ok' ? 'text-green-800 bg-green-50/80' : 'text-red-800 bg-red-50/80'
          }`}
        >
          {localItem.verifyLog}
        </pre>
      )}
      {leanAvailable === false && (
        <p className="px-3 py-1.5 text-xs text-ak-muted border-t border-emerald-200/40">
          Lean verifier not available on this server. Run locally:{' '}
          <code className="font-mono">cd lean && lake env lean proof.lean</code>
        </p>
      )}
    </div>
  );
}
