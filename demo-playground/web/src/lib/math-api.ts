/**
 * Client for the Math tab's restricted endpoints on the Alkahest demo server.
 *
 *   POST /math            {input, definitions} → MathResult   (computes)
 *   POST /math/interpret  {input, definitions} → interpretation only (live preview)
 *
 * The server parses a small maths language and calls Alkahest directly; it
 * never runs Python from the request. See demo-playground/server/math_service.py.
 */
import type { ServerConnection } from '@/lib/server-connection';
import { alkahestAuthHeaders } from '@/lib/server-connection';

export interface MathDefinition {
  name: string;
  params: string[];
  value: string;
}

export interface MathResult {
  ok: boolean;
  kind?: string;
  latex?: string;
  text?: string;
  interpreted_latex?: string | null;
  approx?: string;
  notes?: string[];
  side_conditions?: string[];
  verification?: string;
  verified?: boolean;
  certificate?: string;
  steps?: number;
  elapsed_ms?: number;
  definition?: MathDefinition;
  error_code?: string;
  error_message?: string;
  error_detail?: string | null;
  error_hint?: string | null;
  /** alkahest = a coded decline; input = could not read the line; playground = a resource limit or bug */
  error_source?: 'alkahest' | 'input' | 'playground';
}

export type MathBackendState =
  | { status: 'checking' }
  | { status: 'online' }
  | { status: 'none' } // no server URL configured (hosted page default)
  | { status: 'jupyter' } // a Jupyter backend is configured; it has no /math
  | { status: 'offline'; reason: string }
  | { status: 'outdated' }; // reachable, but a server without the Math endpoint

export async function probeMathBackend(conn: ServerConnection): Promise<MathBackendState> {
  if (!conn.httpUrl) return { status: 'none' };
  if (conn.backend === 'jupyter') return { status: 'jupyter' };
  try {
    const res = await fetch(`${conn.httpUrl}/health`, {
      headers: alkahestAuthHeaders(conn.token),
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return { status: 'offline', reason: `HTTP ${res.status}` };
    const data = (await res.json()) as { math?: boolean };
    return data.math ? { status: 'online' } : { status: 'outdated' };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { status: 'offline', reason: msg };
  }
}

async function post(conn: ServerConnection, path: string, body: unknown, timeoutMs: number) {
  const res = await fetch(`${conn.httpUrl}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...alkahestAuthHeaders(conn.token) },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (res.status === 401) {
    return {
      ok: false,
      error_code: 'HTTP-401',
      error_message: 'The server needs an access token. Add it in settings (Ctrl+/).',
      error_source: 'playground',
    } satisfies MathResult;
  }
  if (!res.ok) {
    return {
      ok: false,
      error_code: `HTTP-${res.status}`,
      error_message: `The server answered ${res.status} ${res.statusText}.`,
      error_source: 'playground',
    } satisfies MathResult;
  }
  return (await res.json()) as MathResult;
}

export function runMath(
  conn: ServerConnection,
  input: string,
  definitions: MathDefinition[],
): Promise<MathResult> {
  return post(conn, '/math', { input, definitions }, 30_000);
}

export function interpretMath(
  conn: ServerConnection,
  input: string,
  definitions: MathDefinition[],
): Promise<MathResult> {
  return post(conn, '/math/interpret', { input, definitions }, 4_000);
}

/** Human labels for Alkahest's verification statuses (never overstated). */
export function verificationBadge(r: MathResult): { label: string; title: string } | null {
  switch (r.verification) {
    case 'exactly_verified':
      return {
        label: 'verified',
        title: 'Alkahest checked this result exactly (for example, differentiating the antiderivative gives back the integrand).',
      };
    case 'lean_checked':
      return { label: 'verified in Lean', title: 'A Lean proof of this result was checked.' };
    case 'rigorous_enclosure':
      return {
        label: 'certified digits',
        title: 'Computed with rigorous ball arithmetic: every digit shown is guaranteed.',
      };
    case 'numerically_checked':
      return {
        label: 'numerically checked',
        title: 'Checked at sample points with floating point only; not an exact proof.',
      };
    default:
      return null;
  }
}
