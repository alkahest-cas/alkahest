import type { Config } from 'tailwindcss';
import typography from '@tailwindcss/typography';

/** A design token from globals.css, usable with Tailwind's /<alpha> modifier. */
const token = (name: string) => `rgb(var(--${name}-rgb) / <alpha-value>)`;

const config: Config = {
  content: ['./src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            '--tw-prose-body': 'var(--fg-2)',
            '--tw-prose-headings': 'var(--fg)',
            '--tw-prose-code': 'var(--brand)',
            '--tw-prose-pre-bg': 'var(--bg-sunk)',
            '--tw-prose-pre-code': 'var(--fg)',
            '--tw-prose-links': 'var(--brand)',
            '--tw-prose-bold': 'var(--fg)',
            '--tw-prose-counters': 'var(--muted)',
            '--tw-prose-bullets': 'var(--muted)',
            '--tw-prose-hr': 'var(--border)',
            '--tw-prose-quotes': 'var(--muted)',
            '--tw-prose-quote-borders': 'var(--border)',
            '--tw-prose-captions': 'var(--muted)',
            '--tw-prose-th-borders': 'var(--border)',
            '--tw-prose-td-borders': 'var(--border)',
          },
        },
      },
      colors: {
        ak: {
          bg: token('bg'),
          elev: token('bg-elev'),
          'code-bg': token('bg-sunk'),
          sunk: token('bg-sunk'),
          fg: token('fg'),
          fg2: token('fg-2'),
          muted: token('muted'),
          border: token('border'),
          'border-strong': token('border-strong'),
          brand: token('brand'),
          'brand-dark': token('brand-dark'),
          'brand-fg': token('brand-fg'),
          ok: token('ok'),
          warn: token('warn'),
          /** Execution / running state (kept distinct from brand red). */
          run: token('run'),
          'run-dark': token('run'),
        },
      },
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      borderRadius: {
        DEFAULT: '6px',
        lg: '10px',
      },
      maxWidth: {
        site: '1200px',
      },
    },
  },
  plugins: [typography],
};

export default config;
