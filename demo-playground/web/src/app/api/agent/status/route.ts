import { NextResponse } from 'next/server';
import { agentApiKeyHelp, isAgentApiKeyConfigured } from '@/lib/agent-config';

export const runtime = 'nodejs';

export async function POST(req: Request) {
  const body = (await req.json()) as {
    provider?: string;
    customBaseUrl?: string;
    customApiKey?: string;
  };

  const provider = body.provider ?? process.env.AI_PROVIDER ?? 'anthropic';
  const configured = isAgentApiKeyConfigured({
    provider,
    customBaseUrl: body.customBaseUrl,
    customApiKey: body.customApiKey,
  });

  return NextResponse.json({
    configured,
    message: configured ? null : agentApiKeyHelp(provider),
  });
}
