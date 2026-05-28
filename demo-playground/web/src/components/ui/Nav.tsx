'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import { useSettings } from './SettingsContext';

interface NavProps {
  isRecording?: boolean;
  onToggleRecording?: () => void;
  serverStatus?: 'unknown' | 'online' | 'offline';
  zenMode?: boolean;
  isDirty?: boolean;
}

export default function Nav({ isRecording, onToggleRecording, serverStatus = 'unknown', zenMode, isDirty }: NavProps) {
  const pathname = usePathname();
  const { toggleSettings } = useSettings();
  const [zenVisible, setZenVisible] = useState(false);
  const zenTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!zenMode) return;
    function handleMouseMove() {
      setZenVisible(true);
      if (zenTimerRef.current) clearTimeout(zenTimerRef.current);
      zenTimerRef.current = setTimeout(() => setZenVisible(false), 2000);
    }
    document.addEventListener('mousemove', handleMouseMove);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      if (zenTimerRef.current) clearTimeout(zenTimerRef.current);
    };
  }, [zenMode]);

  const statusColor =
    serverStatus === 'online' ? 'bg-green-500' :
    serverStatus === 'offline' ? 'bg-red-400' :
    'bg-ak-border';

  // In zen mode, omit the nav entirely until the user moves the mouse (avoids a blank
  // strip at the top in headless recordings before React applies opacity-0).
  if (zenMode && !zenVisible) return null;

  return (
    <>
      <nav
        className="sticky top-0 z-40 border-b border-ak-border bg-ak-bg/95 backdrop-blur-sm transition-opacity duration-300"
      >
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          {/* Logo */}
          <a
            href="https://alkahest-cas.github.io"
            className="flex items-center gap-2 font-semibold text-ak-fg no-underline"
            onClick={(e) => {
              if (isDirty && !window.confirm('Leave the playground? Unsaved changes will be lost.\n\nTip: press Ctrl+S (or ⌘S) to save first.')) {
                e.preventDefault();
              }
            }}
          >
            <span className="text-ak-brand font-bold">alkahest</span>
            <span className="text-ak-muted font-normal text-sm">playground</span>
          </a>

          {/* Separator */}
          <div className="h-4 w-px bg-ak-border" />

          {/* Nav links */}
          <div className="flex items-center gap-1">
            <NavLink href="/" active={pathname === '/'}>Notebook</NavLink>
            <NavLink href="/agent" active={pathname === '/agent'}>Agent</NavLink>
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Server status */}
          <div className="flex items-center gap-1.5 text-xs text-ak-muted">
            <span
              className={clsx('h-2 w-2 rounded-full', statusColor)}
              title={`Server: ${serverStatus}`}
            />
            <span className="hidden sm:inline">server</span>
          </div>

          {/* Record button */}
          {onToggleRecording && (
            <button
              onClick={onToggleRecording}
              className={clsx(
                'flex items-center gap-1.5 rounded px-3 py-1 text-xs font-medium transition-all',
                isRecording
                  ? 'bg-ak-brand text-white'
                  : 'border border-ak-border bg-ak-bg text-ak-muted hover:text-ak-fg',
              )}
              title={isRecording ? 'Stop recording' : 'Start recording'}
            >
              <span
                className={clsx('h-2 w-2 rounded-full bg-current', isRecording && 'rec-indicator')}
              />
              {isRecording ? 'Recording…' : 'Record'}
            </button>
          )}

          {/* Settings */}
          <button
            onClick={toggleSettings}
            className="rounded p-1.5 text-ak-muted hover:bg-ak-code-bg hover:text-ak-fg transition-colors"
            title="Settings (Ctrl+/)"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>
        </div>
      </nav>

    </>
  );
}

function NavLink({ href, active, children }: { href: string; active: boolean; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className={clsx(
        'rounded px-3 py-1.5 text-sm font-medium transition-colors',
        active
          ? 'bg-ak-code-bg text-ak-fg'
          : 'text-ak-muted hover:bg-ak-code-bg hover:text-ak-fg',
      )}
    >
      {children}
    </Link>
  );
}
