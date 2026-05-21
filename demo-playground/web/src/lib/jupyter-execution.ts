import { v4 as uuid } from 'uuid';
import type { OutputItem } from '@/lib/execution';
import { classifyRichMime, postprocessOutputItems } from '@/lib/lean';
import type { ServerConnection } from '@/lib/server-connection';
import { jupyterUrlWithToken } from '@/lib/server-connection';

interface JupyterKernel {
  id: string;
}

interface JupyterWsMessage {
  channel?: string;
  header?: { msg_type?: string; msg_id?: string };
  parent_header?: { msg_id?: string };
  content?: Record<string, unknown>;
}

async function jupyterProxyFetch(
  conn: ServerConnection,
  path: string,
  method: string,
  body?: unknown,
): Promise<unknown> {
  const res = await fetch('/api/jupyter-proxy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      baseUrl: conn.httpUrl,
      token: conn.token || undefined,
      path,
      method,
      body,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Jupyter request failed: ${res.status}`);
  }
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

export async function createJupyterKernel(conn: ServerConnection): Promise<string> {
  const data = (await jupyterProxyFetch(conn, '/api/kernels', 'POST', {})) as JupyterKernel;
  if (!data?.id) throw new Error('Jupyter did not return a kernel id');
  return data.id;
}

export async function destroyJupyterKernel(conn: ServerConnection, kernelId: string): Promise<void> {
  await jupyterProxyFetch(conn, `/api/kernels/${kernelId}`, 'DELETE');
}

function parseJupyterMessage(raw: string, pendingExecuteId: string | null): {
  item?: OutputItem;
  done?: boolean;
  executionCount?: number;
  pendingExecuteId: string | null;
} {
  let msg: JupyterWsMessage;
  try {
    msg = JSON.parse(raw) as JupyterWsMessage;
  } catch {
    return { pendingExecuteId };
  }

  const channel = msg.channel;
  const msgType = msg.header?.msg_type;
  const content = msg.content ?? {};

  if (channel === 'shell' && msgType === 'execute_reply') {
    const parentId = msg.parent_header?.msg_id;
    if (parentId && parentId === pendingExecuteId) {
      const count = content.execution_count as number | undefined;
      return { done: true, executionCount: count ?? 0, pendingExecuteId: null };
    }
  }

  if (channel === 'iopub') {
    const parentId = msg.parent_header?.msg_id;
    if (pendingExecuteId && parentId && parentId !== pendingExecuteId) {
      return { pendingExecuteId };
    }

    if (msgType === 'stream') {
      return {
        item: {
          type: 'text',
          stream: (content.name as 'stdout' | 'stderr') ?? 'stdout',
          text: String(content.text ?? ''),
        },
        pendingExecuteId,
      };
    }

    if (msgType === 'display_data' || msgType === 'execute_result') {
      const data = content.data as Record<string, string> | undefined;
      if (!data) return { pendingExecuteId };
      const item = classifyRichMime(data);
      if (!item) return { pendingExecuteId };
      const executionCount =
        msgType === 'execute_result' ? (content.execution_count as number | undefined) : undefined;
      return { item, executionCount, pendingExecuteId };
    }

    if (msgType === 'error') {
      return {
        item: {
          type: 'error',
          ename: String(content.ename ?? 'Error'),
          evalue: String(content.evalue ?? ''),
          traceback: (content.traceback as string[]) ?? [],
        },
        pendingExecuteId,
      };
    }
  }

  return { pendingExecuteId };
}

export function executeOnJupyter(
  conn: ServerConnection,
  kernelId: string,
  code: string,
  onOutput: (item: OutputItem) => void,
  onDone: (executionCount: number) => void,
  onError: (err: string) => void,
): () => void {
  const sessionId = uuid();
  const wsBase = `${conn.wsUrl}/api/kernels/${kernelId}/channels?session_id=${encodeURIComponent(sessionId)}`;
  const wsUrl = jupyterUrlWithToken(wsBase, conn.token);
  const ws = new WebSocket(wsUrl);

  let pendingExecuteId: string | null = null;
  let lastExecutionCount = 0;
  let closed = false;

  ws.onopen = () => {
    const msgId = uuid();
    pendingExecuteId = msgId;
    const request = {
      channel: 'shell',
      header: {
        msg_id: msgId,
        username: 'alkahest-playground',
        session: sessionId,
        msg_type: 'execute_request',
        version: '5.3',
      },
      parent_header: {},
      metadata: {},
      content: {
        code,
        silent: false,
        store_history: true,
        user_expressions: {},
        allow_stdin: false,
        stop_on_error: true,
      },
    };
    ws.send(JSON.stringify(request));
  };

  ws.onmessage = (event) => {
    const parsed = parseJupyterMessage(String(event.data), pendingExecuteId);
    pendingExecuteId = parsed.pendingExecuteId;
    if (parsed.item) onOutput(parsed.item);
    if (parsed.executionCount != null) lastExecutionCount = parsed.executionCount;
    if (parsed.done) {
      onDone(lastExecutionCount);
      closed = true;
      ws.close();
    }
  };

  ws.onerror = () => onError('WebSocket connection error — check URL, token, and CORS on the Jupyter server');
  ws.onclose = (e) => {
    if (!closed && e.code !== 1000 && e.code !== 1005) {
      onError(`Connection closed (${e.code})`);
    }
  };

  return () => {
    closed = true;
    ws.close();
  };
}

/** Best-effort sync execute via WebSocket (used by agent tool). */
export function runOnJupyterSync(
  conn: ServerConnection,
  kernelId: string,
  code: string,
): Promise<OutputItem[]> {
  return new Promise((resolve, reject) => {
    const outputs: OutputItem[] = [];
    const cancel = executeOnJupyter(
      conn,
      kernelId,
      code,
      (item) => outputs.push(item),
      () => {
        cancel();
        resolve(postprocessOutputItems(outputs));
      },
      (err) => {
        cancel();
        reject(new Error(err));
      },
    );
  });
}
