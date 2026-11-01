/**
 * Client for the Math tab's restricted endpoints on the Alkahest demo server.
 *
 *   POST /math            {input, definitions} → MathResult   (computes)
 *   POST /math/interpret  {input, definitions} → interpretation only (live preview)
 *
 * The server parses a small maths language and calls Alkahest directly; it
 * never runs Python from the request. The implementation lives in the Alkahest
 * package (python/alkahest/playground/) and is served either by the demo
 * server (demo-playground/server) or, on a visitor's own machine, by
 * `python -m alkahest.playground` on http://127.0.0.1:8765.
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

/** Where `python -m alkahest.playground` listens (see python/alkahest/playground/server.py). */
export const LOCAL_MATH_URL = 'http://127.0.0.1:8765';

/** The two commands that start a local Math server. */
export const LOCAL_SETUP_COMMANDS = ['pip install -U alkahest', 'python -m alkahest.playground'] as const;

/** Chrome/Edge 142+ and Firefox 153+ gate loopback requests from public pages behind a permission. */
export type LoopbackPermission = 'granted' | 'denied' | 'prompt' | 'unsupported';

export type MathBackendState =
  | { status: 'checking' }
  | { status: 'online'; local: boolean; version?: string }
  // No server URL configured (hosted page default) and no local server in use.
  // `tried`: a local probe was made and nothing answered.
  | { status: 'none'; tried: boolean; permission: LoopbackPermission }
  | { status: 'jupyter' } // a Jupyter backend is configured; it has no /math
  | { status: 'offline'; reason: string }
  | { status: 'outdated' }; // reachable, but a server without the Math endpoint

const LOOPBACK_HOST = /^(localhost|127\.0\.0\.1|\[::1\])$/i;

export function isLoopbackUrl(url: string): boolean {
  try {
    return LOOPBACK_HOST.test(new URL(url).hostname);
  } catch {
    return false;
  }
}

/** True when this page itself is served from this machine (no permission needed). */
export function pageIsLoopback(): boolean {
  return typeof window !== 'undefined' && LOOPBACK_HOST.test(window.location.hostname);
}

/**
 * Fetch options for a request to `url`. For a loopback URL from a public page,
 * Chrome's Local Network Access asks for `targetAddressSpace: 'loopback'`;
 * other browsers ignore the unknown member.
 */
function fetchInit(url: string, init: RequestInit): RequestInit {
  if (isLoopbackUrl(url) && !pageIsLoopback()) {
    return { ...init, targetAddressSpace: 'loopback' } as RequestInit;
  }
  return init;
}

/**
 * The `loopback-network` permission state, without prompting. Never queries the
 * legacy `local-network-access` name (it crashes the renderer in some Chrome
 * versions); browsers that do not know the name throw, which reads as unsupported.
 */
export async function loopbackPermission(): Promise<LoopbackPermission> {
  try {
    if (typeof navigator === 'undefined' || !navigator.permissions?.query) return 'unsupported';
    const status = await navigator.permissions.query({ name: 'loopback-network' as PermissionName });
    return status.state;
  } catch {
    return 'unsupported';
  }
}

/** Safari, and every browser on iOS/iPadOS (all WebKit): blocks https → http://127.0.0.1. */
export function isWebKitOnly(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  const ios = /iPhone|iPad|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const safari = /Safari\//.test(ua) && !/Chrome|Chromium|CriOS|FxiOS|Edg|OPR|Firefox/.test(ua);
  return ios || safari;
}

export function localConnection(): ServerConnection {
  return {
    backend: 'alkahest',
    httpUrl: LOCAL_MATH_URL,
    wsUrl: LOCAL_MATH_URL.replace(/^http/, 'ws'),
    token: '',
  };
}

/** Quietly ask the local server for /health; resolves to its version, or null. */
export async function probeLocalMath(timeoutMs = 1500): Promise<{ version?: string } | null> {
  const url = `${LOCAL_MATH_URL}/health`;
  try {
    const res = await fetch(url, fetchInit(url, { signal: AbortSignal.timeout(timeoutMs), cache: 'no-store' }));
    if (!res.ok) return null;
    const data = (await res.json()) as { math?: boolean; version?: string };
    return data.math ? { version: data.version } : null;
  } catch {
    return null;
  }
}

export async function probeMathBackend(conn: ServerConnection): Promise<MathBackendState> {
  if (!conn.httpUrl) return { status: 'none', tried: false, permission: 'unsupported' };
  if (conn.backend === 'jupyter') return { status: 'jupyter' };
  try {
    const url = `${conn.httpUrl}/health`;
    const res = await fetch(
      url,
      fetchInit(url, {
        headers: alkahestAuthHeaders(conn.token),
        signal: AbortSignal.timeout(4000),
      }),
    );
    if (!res.ok) return { status: 'offline', reason: `HTTP ${res.status}` };
    const data = (await res.json()) as { math?: boolean; version?: string };
    return data.math
      ? { status: 'online', local: isLoopbackUrl(conn.httpUrl), version: data.version }
      : { status: 'outdated' };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { status: 'offline', reason: msg };
  }
}

async function post(conn: ServerConnection, path: string, body: unknown, timeoutMs: number) {
  const url = `${conn.httpUrl}${path}`;
  const res = await fetch(
    url,
    fetchInit(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...alkahestAuthHeaders(conn.token) },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    }),
  );
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
