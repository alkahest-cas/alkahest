import type { ProviderId } from '@/lib/ai-providers';

const OPENAI_COMPATIBLE_ENV_KEYS = ['OPENAI_COMPATIBLE_API_KEY', 'OPENAI_API_KEY'] as const;

const PROVIDER_ENV_KEYS: Record<Exclude<ProviderId, 'openai-compatible'>, string> = {
  anthropic: 'ANTHROPIC_API_KEY',
  openai: 'OPENAI_API_KEY',
  google: 'GOOGLE_GENERATIVE_AI_API_KEY',
  mistral: 'MISTRAL_API_KEY',
  groq: 'GROQ_API_KEY',
  xai: 'XAI_API_KEY',
  deepseek: 'DEEPSEEK_API_KEY',
  together: 'TOGETHER_API_KEY',
  fireworks: 'FIREWORKS_API_KEY',
  cerebras: 'CEREBRAS_API_KEY',
};

export interface AgentConfigCheck {
  provider: string;
  customBaseUrl?: string;
  customApiKey?: string;
}

function envHasKey(name: string): boolean {
  const value = process.env[name]?.trim();
  return Boolean(value);
}

/** Server-side: whether the selected provider has credentials available. */
export function isAgentApiKeyConfigured({
  provider,
  customBaseUrl,
  customApiKey,
}: AgentConfigCheck): boolean {
  if (customApiKey?.trim()) return true;

  if (provider === 'openai-compatible') {
    const baseURL = customBaseUrl?.trim() || process.env.OPENAI_COMPATIBLE_BASE_URL?.trim();
    if (!baseURL) return false;
    return OPENAI_COMPATIBLE_ENV_KEYS.some((key) => envHasKey(key));
  }

  const envKey = PROVIDER_ENV_KEYS[provider as Exclude<ProviderId, 'openai-compatible'>];
  if (!envKey) return false;
  return envHasKey(envKey);
}

export function agentApiKeyHelp(provider: string): string {
  if (provider === 'openai-compatible') {
    return 'Add an API key and base URL in Settings (Ctrl+/) or set OPENAI_COMPATIBLE_API_KEY and OPENAI_COMPATIBLE_BASE_URL in web/.env.local.';
  }
  const envKey = PROVIDER_ENV_KEYS[provider as Exclude<ProviderId, 'openai-compatible'>] ?? 'ANTHROPIC_API_KEY';
  return `Add an API key in web/.env.local (${envKey}) or choose OpenAI-compatible in Settings to enter a key in the browser.`;
}
