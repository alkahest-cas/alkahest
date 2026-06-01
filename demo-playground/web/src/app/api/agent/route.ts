import { streamText, tool, type LanguageModel } from 'ai';
import { z } from 'zod';
import { ALKAHEST_SYSTEM_PROMPT } from '@/lib/alkahest-skill';
import type { OutputItem } from '@/lib/execution';
import { getLanguageModel } from '@/lib/get-language-model';

export const runtime = 'nodejs';
export const maxDuration = 180;

export async function POST(req: Request) {
  const body = await req.json() as {
    messages: unknown[];
    provider?: string;
    model?: string;
    customBaseUrl?: string;
    customApiKey?: string;
    serverHttpUrl?: string;
    serverWsUrl?: string;
    serverBackend?: string;
    serverToken?: string;
    sessionId?: string;
  };

  const provider = body.provider ?? process.env.AI_PROVIDER ?? 'anthropic';
  const model = body.model ?? process.env.AI_MODEL ?? 'claude-sonnet-4-6';
  const serverHttpUrl = body.serverHttpUrl ?? process.env.PYTHON_SERVER_URL ?? 'http://localhost:8000';
  const serverToken = body.serverToken ?? '';
  const sessionId = body.sessionId;

  const serverHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
  if (serverToken) serverHeaders.Authorization = `Bearer ${serverToken}`;

  const languageModel = getLanguageModel(provider, model, {
    customBaseUrl: body.customBaseUrl,
    customApiKey: body.customApiKey,
  });

  const result = streamText({
    model: languageModel as LanguageModel,
    system: ALKAHEST_SYSTEM_PROMPT,
    messages: body.messages as Parameters<typeof streamText>[0]['messages'],
    maxSteps: 15,
    tools: {
      run_python: tool({
        description:
          'Execute Python code on the alkahest server kernel. Use this to run alkahest, SymPy, numpy, matplotlib, or any Python computation. The kernel is stateful — variables persist between calls. ' +
          'For alkahest DerivedResult values, call display_lean_cert(result, operation=...) or print ak.to_lean(result) to surface Lean 4 certificates in the UI.',
        parameters: z.object({
          code: z.string().describe('Python code to execute'),
        }),
        execute: async ({ code }) => {
          if (!sessionId) {
            return { outputs: [{ type: 'error', ename: 'NoSession', evalue: 'No kernel session available.', traceback: [] }] as OutputItem[] };
          }

          try {
            const res = await fetch(`${serverHttpUrl}/sessions/${sessionId}/run`, {
              method: 'POST',
              headers: serverHeaders,
              body: JSON.stringify({ code }),
              signal: AbortSignal.timeout(60_000),
            });

            if (!res.ok) {
              const text = await res.text();
              return {
                outputs: [{ type: 'error', ename: 'ServerError', evalue: text, traceback: [] }] as OutputItem[],
              };
            }

            const data = await res.json() as { outputs: OutputItem[] };
            return data;
          } catch (e) {
            return {
              outputs: [{ type: 'error', ename: 'NetworkError', evalue: String(e), traceback: [] }] as OutputItem[],
            };
          }
        },
      }),
      verify_lean: tool({
        description:
          'Typecheck a Lean 4 certificate (.lean source) using the server Mathlib project. Returns ok, stdout, stderr, and duration_ms.',
        parameters: z.object({
          source: z.string().describe('Complete .lean file contents from alkahest.to_lean or display_lean_cert'),
        }),
        execute: async ({ source }) => {
          try {
            const res = await fetch(`${serverHttpUrl}/verify-lean`, {
              method: 'POST',
              headers: serverHeaders,
              body: JSON.stringify({ source, timeout_sec: 120 }),
              signal: AbortSignal.timeout(135_000),
            });
            if (!res.ok) {
              const text = await res.text();
              return { ok: false, stdout: '', stderr: text, duration_ms: 0 };
            }
            return await res.json();
          } catch (e) {
            return { ok: false, stdout: '', stderr: String(e), duration_ms: 0 };
          }
        },
      }),
    },
  });

  return result.toDataStreamResponse();
}
