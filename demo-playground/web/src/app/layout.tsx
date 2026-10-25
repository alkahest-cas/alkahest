import type { Metadata } from 'next';
import { Providers } from './providers';
import './globals.css';

const FAVICON =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23c41e3a'/%3E%3Ctext x='16' y='22.5' text-anchor='middle' font-family='Inter,Arial,sans-serif' font-weight='700' font-size='20' fill='white'%3Ea%3C/text%3E%3C/svg%3E";

export const metadata: Metadata = {
  title: 'Alkahest Playground',
  description:
    'Type mathematics or Python and run it on the Alkahest computer algebra system, with coded refusals and Lean certificates.',
  icons: { icon: FAVICON },
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
