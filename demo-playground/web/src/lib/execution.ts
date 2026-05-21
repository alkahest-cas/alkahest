import type { ServerConnection } from '@/lib/server-connection';
import { alkahestAuthHeaders } from '@/lib/server-connection';
import { classifyRichMime, postprocessOutputItems } from '@/lib/lean';
import { publicAssetPath } from '@/lib/hosting';
import {
  createJupyterKernel,
  destroyJupyterKernel,
  executeOnJupyter,
  runOnJupyterSync,
} from '@/lib/jupyter-execution';

export type OutputItem =
  | { type: 'text'; stream: 'stdout' | 'stderr'; text: string }
  | { type: 'html'; html: string }
  | { type: 'latex'; latex: string }
  | { type: 'image'; format: 'png' | 'svg'; data: string }
  | { type: 'json'; data: unknown }
  | {
      type: 'lean';
      source: string;
      operation?: string;
      steps?: number;
      verified?: 'ok' | 'fail';
      verifyLog?: string;
    }
  | { type: 'error'; ename: string; evalue: string; traceback: string[] };

export type ExecutionMode = 'auto' | 'wasm' | 'server';

export function needsServer(code: string): boolean {
  return /\bimport\s+alkahest\b|from\s+alkahest\b/.test(code);
}

// ── Server-side execution ─────────────────────────────────────────────────────

export async function createSession(conn: ServerConnection): Promise<string> {
  if (conn.backend === 'jupyter') {
    return createJupyterKernel(conn);
  }
  const res = await fetch(`${conn.httpUrl}/sessions`, {
    method: 'POST',
    headers: alkahestAuthHeaders(conn.token),
  });
  if (!res.ok) throw new Error(`Failed to create session: ${res.statusText}`);
  const data = await res.json();
  return data.session_id as string;
}

export async function destroySession(conn: ServerConnection, sessionId: string): Promise<void> {
  if (conn.backend === 'jupyter') {
    await destroyJupyterKernel(conn, sessionId);
    return;
  }
  await fetch(`${conn.httpUrl}/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: alkahestAuthHeaders(conn.token),
  });
}

export async function installWheel(conn: ServerConnection, sessionId: string, file: File): Promise<void> {
  if (conn.backend === 'jupyter') {
    throw new Error('Wheel install is only supported with the Alkahest execution server');
  }
  const form = new FormData();
  form.append('wheel', file);
  const headers = alkahestAuthHeaders(conn.token) as Record<string, string>;
  const res = await fetch(`${conn.httpUrl}/sessions/${sessionId}/install-wheel`, {
    method: 'POST',
    headers,
    body: form,
  });
  if (!res.ok) throw new Error(`Wheel install failed: ${res.statusText}`);
}

export function executeOnServer(
  conn: ServerConnection,
  sessionId: string,
  code: string,
  onOutput: (item: OutputItem) => void,
  onDone: (executionCount: number) => void,
  onError: (err: string) => void,
): () => void {
  if (conn.backend === 'jupyter') {
    return executeOnJupyter(conn, sessionId, code, onOutput, onDone, onError);
  }

  const wsBase = `${conn.wsUrl}/ws/${sessionId}`;
  const wsUrl = conn.token
    ? `${wsBase}?token=${encodeURIComponent(conn.token)}`
    : wsBase;
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => ws.send(JSON.stringify({ code }));

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data as string) as ServerMessage;

    if (
      msg.type === 'text' ||
      msg.type === 'latex' ||
      msg.type === 'html' ||
      msg.type === 'image' ||
      msg.type === 'json' ||
      msg.type === 'lean'
    ) {
      onOutput(msg as OutputItem);
    } else if (msg.type === 'stream') {
      onOutput({ type: 'text', stream: (msg.name as 'stdout' | 'stderr') ?? 'stdout', text: msg.text ?? '' });
    } else if (msg.type === 'display_data' || msg.type === 'execute_result') {
      const d = msg.data as Record<string, string>;
      const rich = classifyRichMime(d);
      if (rich) onOutput(rich);
    } else if (msg.type === 'error') {
      onOutput({
        type: 'error',
        ename: msg.ename ?? 'Error',
        evalue: msg.evalue ?? '',
        traceback: msg.traceback ?? [],
      });
    } else if (msg.type === 'done') {
      onDone(msg.execution_count ?? 0);
      ws.close();
    }
  };

  ws.onerror = () => onError('WebSocket connection error — is the server running?');
  ws.onclose = (e) => {
    if (e.code !== 1000 && e.code !== 1005) onError(`Connection closed (${e.code})`);
  };

  return () => ws.close();
}

export async function runOnServerSync(
  conn: ServerConnection,
  sessionId: string,
  code: string,
): Promise<OutputItem[]> {
  if (conn.backend === 'jupyter') {
    return runOnJupyterSync(conn, sessionId, code);
  }

  const res = await fetch(`${conn.httpUrl}/sessions/${sessionId}/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...alkahestAuthHeaders(conn.token),
    },
    body: JSON.stringify({ code }),
  });
  if (!res.ok) throw new Error(`Execution failed: ${res.statusText}`);
  const data = await res.json();
  return postprocessOutputItems(data.outputs as OutputItem[]);
}

export { verifyLeanCertificate, fetchLeanStatus, applyVerifyResult } from '@/lib/lean';

// ── WASM execution (Pyodide web worker) ──────────────────────────────────────
let pyodideWorker: Worker | null = null;
let workerReady = false;
const pendingCallbacks = new Map<string, { resolve: (v: OutputItem[]) => void; reject: (e: Error) => void }>();

function getPyodideWorker(): Worker {
  if (!pyodideWorker) {
    pyodideWorker = new Worker(publicAssetPath('/pyodide-worker.js'));
    pyodideWorker.onmessage = (event) => {
      const msg = event.data as WorkerMessage;
      if (msg.type === 'ready') {
        workerReady = true;
        return;
      }
      const cb = pendingCallbacks.get(msg.id);
      if (!cb) return;
      pendingCallbacks.delete(msg.id);
      if (msg.type === 'result') cb.resolve(postprocessOutputItems(msg.outputs ?? []));
      else cb.reject(new Error(msg.error ?? 'Worker error'));
    };
  }
  return pyodideWorker;
}

export function executeInWasm(code: string): Promise<OutputItem[]> {
  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID();
    const worker = getPyodideWorker();

    const waitAndSend = () => {
      if (workerReady) {
        pendingCallbacks.set(id, { resolve, reject });
        worker.postMessage({ type: 'execute', id, code });
      } else {
        setTimeout(waitAndSend, 100);
      }
    };
    waitAndSend();
  });
}

interface ServerMessage {
  type: string;
  name?: string;
  text?: string;
  data?: unknown;
  ename?: string;
  evalue?: string;
  traceback?: string[];
  execution_count?: number;
}

interface WorkerMessage {
  type: 'ready' | 'result' | 'error';
  id: string;
  outputs?: OutputItem[];
  error?: string;
}
