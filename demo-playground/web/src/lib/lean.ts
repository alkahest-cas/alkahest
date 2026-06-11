import type { OutputItem } from '@/lib/execution';
import type { ServerConnection } from '@/lib/server-connection';
import { alkahestAuthHeaders } from '@/lib/server-connection';

export const AK_LEAN_MARKER = '__AK_LEAN_CERT__';
export const AK_LEAN_MIME = 'application/x-alkahest-lean+json';

export interface LeanCertificateMeta {
  operation?: string;
  steps?: number;
}

export interface LeanVerifyResult {
  ok: boolean;
  stdout: string;
  stderr: string;
  duration_ms: number;
  proof_file?: string | null;
}

export interface LeanStatus {
  available: boolean;
  lake?: string | null;
  lean?: string | null;
  elan?: string | null;
  project_dir: string;
  project_exists: boolean;
  toolchain_file?: string | null;
}

/** Normalize raw server/kernel dicts into ``OutputItem`` lean entries. */
export function leanItemFromPayload(payload: Record<string, unknown>): OutputItem | null {
  const source = payload.source;
  if (typeof source !== 'string' || !source.trim()) return null;
  return {
    type: 'lean',
    source,
    operation: typeof payload.operation === 'string' ? payload.operation : undefined,
    steps: typeof payload.steps === 'number' ? payload.steps : undefined,
    verified: undefined,
    verifyLog: undefined,
  };
}

function splitStdoutLean(text: string): { cleaned: string; leanItems: OutputItem[] } {
  const leanItems: OutputItem[] = [];
  const kept: string[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith(AK_LEAN_MARKER)) {
      try {
        const payload = JSON.parse(trimmed.slice(AK_LEAN_MARKER.length)) as Record<string, unknown>;
        const item = leanItemFromPayload(payload);
        if (item) leanItems.push(item);
        continue;
      } catch {
        /* keep line */
      }
    }
    kept.push(line + '\n');
  }
  let cleaned = kept.join('');
  if (cleaned.endsWith('\n\n')) cleaned = cleaned.replace(/\n+$/, '\n');
  return { cleaned, leanItems };
}

/** Extract lean certificates embedded in text stdout; pass through other items. */
export function postprocessOutputItems(items: OutputItem[]): OutputItem[] {
  const out: OutputItem[] = [];
  for (const item of items) {
    if (item.type === 'text' && item.stream === 'stdout') {
      const { cleaned, leanItems } = splitStdoutLean(item.text);
      out.push(...leanItems);
      if (cleaned.trim()) out.push({ ...item, text: cleaned });
    } else if (item.type === 'lean') {
      out.push(item);
    } else {
      out.push(item);
    }
  }
  return out;
}

export function classifyRichMime(data: Record<string, string>): OutputItem | null {
  for (const mime of [AK_LEAN_MIME, 'application/vnd.alkahest.lean+json']) {
    if (mime in data) {
      try {
        const payload = JSON.parse(data[mime]) as Record<string, unknown>;
        return leanItemFromPayload(payload);
      } catch {
        /* fall through */
      }
    }
  }
  if (data['text/latex']) return { type: 'latex', latex: data['text/latex'] };
  if (data['image/png']) return { type: 'image', format: 'png', data: data['image/png'] };
  if (data['image/svg+xml']) return { type: 'image', format: 'svg', data: data['image/svg+xml'] };
  if (data['text/html']) return { type: 'html', html: data['text/html'] };
  if (data['text/markdown']) return { type: 'text', stream: 'stdout', text: data['text/markdown'] };
  if (data['application/json']) {
    try {
      return { type: 'json', data: JSON.parse(data['application/json']) };
    } catch {
      return { type: 'json', data: data['application/json'] };
    }
  }
  if (data['text/plain']) return { type: 'text', stream: 'stdout', text: data['text/plain'] };
  return null;
}

export async function fetchLeanStatus(conn: ServerConnection): Promise<LeanStatus | null> {
  if (!conn.httpUrl || conn.backend !== 'alkahest') return null;
  try {
    const res = await fetch(`${conn.httpUrl}/lean/status`, {
      headers: alkahestAuthHeaders(conn.token),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    return (await res.json()) as LeanStatus;
  } catch {
    return null;
  }
}

export async function verifyLeanCertificate(
  conn: ServerConnection,
  source: string,
  timeoutSec = 120,
): Promise<LeanVerifyResult> {
  if (!conn.httpUrl || conn.backend !== 'alkahest') {
    return {
      ok: false,
      stdout: '',
      stderr: 'Lean verification requires the Alkahest demo server (not Jupyter-only mode).',
      duration_ms: 0,
    };
  }
  const res = await fetch(`${conn.httpUrl}/verify-lean`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...alkahestAuthHeaders(conn.token),
    },
    body: JSON.stringify({ source, timeout_sec: timeoutSec }),
    signal: AbortSignal.timeout((timeoutSec + 15) * 1000),
  });
  if (!res.ok) {
    const text = await res.text();
    return { ok: false, stdout: '', stderr: text || res.statusText, duration_ms: 0 };
  }
  return (await res.json()) as LeanVerifyResult;
}

export function applyVerifyResult(
  item: OutputItem,
  result: LeanVerifyResult,
): OutputItem {
  if (item.type !== 'lean') return item;
  const log = result.ok
    ? `Verified in ${result.duration_ms}ms`
    : result.stderr || result.stdout || 'Verification failed';
  return {
    ...item,
    verified: result.ok ? 'ok' : 'fail',
    verifyLog: log.trim(),
  };
}
