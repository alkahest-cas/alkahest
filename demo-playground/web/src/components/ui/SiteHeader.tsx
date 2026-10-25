'use client';

import { useEffect, useState } from 'react';
import clsx from 'clsx';
import { isStaticHosting, basePath } from '@/lib/hosting';

/**
 * The shared top bar (identical on the landing page, the docs and the
 * playground): wordmark on the left, Docs / Playground / Agent / GitHub on the
 * right, collapsing behind a menu button below 720px.
 *
 * On the deployed site the links are root-relative (`/`, `/docs/`, `/agent/`);
 * when the playground runs on its own (localhost:3000) they point at the live
 * site, because those sections are not served by this app.
 */
const SITE = isStaticHosting ? '' : 'https://alkahest-cas.github.io';
const GITHUB = 'https://github.com/alkahest-cas/alkahest';

interface SiteHeaderProps {
  /** Ask before leaving (unsaved Python notebook changes). */
  confirmLeave?: () => boolean;
}

export default function SiteHeader({ confirmLeave }: SiteHeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setMenuOpen(false);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [menuOpen]);

  const guard = (e: React.MouseEvent) => {
    if (confirmLeave && !confirmLeave()) e.preventDefault();
  };

  const links = [
    { href: `${SITE}/docs/`, label: 'Docs', current: false },
    { href: `${basePath}/`, label: 'Playground', current: true },
    { href: `${SITE}/agent/`, label: 'Agent', current: false },
  ];

  return (
    <header className="sticky top-0 z-40 border-b border-ak-border bg-ak-bg/85 backdrop-blur-md">
      <div className="mx-auto flex h-[60px] max-w-site items-center gap-6 px-4 sm:px-6">
        <a href={`${SITE}/`} className="ak-wordmark" onClick={guard} aria-label="Alkahest home">
          alkahest
        </a>
        <div className="flex-1" />
        <nav className="hidden items-center gap-7 min-[720px]:flex" aria-label="Site">
          {links.map((l) => (
            <a
              key={l.label}
              href={l.href}
              onClick={l.current ? undefined : guard}
              aria-current={l.current ? 'page' : undefined}
              className={clsx(
                'text-sm font-medium no-underline transition-colors',
                l.current ? 'text-ak-fg' : 'text-ak-fg2 hover:text-ak-fg',
              )}
            >
              {l.label}
            </a>
          ))}
          <a
            href={GITHUB}
            onClick={guard}
            aria-label="Alkahest on GitHub"
            className="text-ak-fg2 transition-colors hover:text-ak-fg"
          >
            <GitHubMark />
          </a>
        </nav>
        <button
          type="button"
          className="rounded-md p-2 text-ak-fg2 hover:text-ak-fg min-[720px]:hidden"
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((v) => !v)}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
            {menuOpen ? <path d="M6 6l12 12M18 6 6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
          </svg>
        </button>
      </div>
      {menuOpen && (
        <nav
          className="border-t border-ak-border bg-ak-bg px-4 pb-3 pt-1 min-[720px]:hidden"
          aria-label="Site"
        >
          {links.map((l) => (
            <a
              key={l.label}
              href={l.href}
              onClick={l.current ? () => setMenuOpen(false) : guard}
              aria-current={l.current ? 'page' : undefined}
              className={clsx(
                'block py-2.5 text-[15px] font-medium no-underline',
                l.current ? 'text-ak-fg' : 'text-ak-fg2',
              )}
            >
              {l.label}
            </a>
          ))}
          <a href={GITHUB} onClick={guard} className="flex items-center gap-2 py-2.5 text-[15px] font-medium text-ak-fg2 no-underline">
            <GitHubMark /> GitHub
          </a>
        </nav>
      )}
    </header>
  );
}

function GitHubMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}
