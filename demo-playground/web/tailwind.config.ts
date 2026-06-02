import type { Config } from 'tailwindcss';
import typography from '@tailwindcss/typography';

const config: Config = {
  content: ['./src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            '--tw-prose-body': '#111111',
            '--tw-prose-headings': '#111111',
            '--tw-prose-code': '#c41e3a',
            '--tw-prose-pre-bg': '#eeede7',
            '--tw-prose-links': '#c41e3a',
            '--tw-prose-bold': '#111111',
            '--tw-prose-counters': '#525252',
            '--tw-prose-bullets': '#525252',
            '--tw-prose-hr': '#e0ded6',
            '--tw-prose-quotes': '#525252',
            '--tw-prose-quote-borders': '#e0ded6',
            '--tw-prose-captions': '#525252',
            '--tw-prose-th-borders': '#e0ded6',
            '--tw-prose-td-borders': '#e0ded6',
          },
        },
      },
      colors: {
        ak: {
          bg: '#f6f5f2',
          fg: '#111111',
          muted: '#525252',
          border: '#e0ded6',
          'code-bg': '#eeede7',
          brand: '#c41e3a',
          'brand-dark': '#9e1830',
          /** Execution / running state (distinct from brand red). */
          run: '#2563eb',
          'run-dark': '#1d4ed8',
        },
      },
      fontFamily: {
        sans: ['ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '6px',
        lg: '10px',
      },
    },
  },
  plugins: [typography],
};

export default config;
