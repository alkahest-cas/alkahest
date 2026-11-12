import type { Metadata, Viewport } from 'next';
import { Providers } from './providers';
import { basePath } from '@/lib/hosting';
import './globals.css';

export const metadata: Metadata = {
  title: 'alkahest – playground',
  description:
    'Type mathematics or Python and run it on the alkahest computer algebra system, with coded refusals and Lean certificates.',
  // The landing page's logo, copied into public/ because the playground is also
  // served on its own (localhost), away from the site root.
  icons: { icon: { url: `${basePath}/logo.svg`, type: 'image/svg+xml' } },
};

export const viewport: Viewport = {
  themeColor: '#f8f7f3',
  colorScheme: 'light',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          crossOrigin="anonymous"
        />
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
          crossOrigin="anonymous"
        />
      </head>
      <body className="min-h-screen bg-ak-bg text-ak-fg">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
