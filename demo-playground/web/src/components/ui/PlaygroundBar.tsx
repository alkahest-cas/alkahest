'use client';

import clsx from 'clsx';
import { useSettings } from './SettingsContext';

export type PlaygroundTab = 'math' | 'python';
export type ServerStatus = 'unknown' | 'online' | 'offline';

interface PlaygroundBarProps {
  tab: PlaygroundTab;
  onTabChange: (tab: PlaygroundTab) => void;
  serverStatus?: ServerStatus;
  isRecording?: boolean;
  onToggleRecording?: () => void;
}

const TABS: { id: PlaygroundTab; label: string; hint: string }[] = [
  { id: 'math', label: 'Math', hint: 'Type ordinary mathematics' },
  { id: 'python', label: 'Python', hint: 'A Python notebook with alkahest' },
];

/** Second row under the site header: Math | Python tabs and playground controls. */
export default function PlaygroundBar({
  tab,
  onTabChange,
  serverStatus = 'unknown',
  isRecording,
  onToggleRecording,
}: PlaygroundBarProps) {
  const { toggleSettings } = useSettings();

  const statusText =
    serverStatus === 'online' ? 'Backend connected' : serverStatus === 'offline' ? 'No backend' : 'Checking backend…';

  return (
    <div className="border-b border-ak-border bg-ak-bg">
      <div className="mx-auto flex h-12 max-w-site items-stretch gap-1 px-4 sm:px-6">
        <div role="tablist" aria-label="Playground mode" className="flex items-stretch gap-5">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              type="button"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              title={t.hint}
              onClick={() => onTabChange(t.id)}
              className={clsx(
                '-mb-px border-b-2 px-0.5 text-sm font-medium transition-colors',
                tab === t.id
                  ? 'border-ak-brand text-ak-fg'
                  : 'border-transparent text-ak-muted hover:text-ak-fg',
              )}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-2 sm:gap-3">
          <span className="flex items-center gap-1.5 text-xs text-ak-muted" title={statusText}>
            <span
              className={clsx(
                'h-2 w-2 rounded-full',
                serverStatus === 'online' ? 'bg-ak-ok' : serverStatus === 'offline' ? 'bg-ak-warn' : 'bg-ak-border-strong',
              )}
            />
            <span className="hidden sm:inline">{statusText}</span>
          </span>

          {onToggleRecording && (
            <button
              type="button"
              onClick={onToggleRecording}
              className={clsx(
                'hidden items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors sm:flex',
                isRecording
                  ? 'bg-ak-brand text-ak-brand-fg'
                  : 'border border-ak-border text-ak-muted hover:border-ak-border-strong hover:text-ak-fg',
              )}
              title={isRecording ? 'Stop recording' : 'Record this tab as a video'}
            >
              <span className={clsx('h-2 w-2 rounded-full bg-current', isRecording && 'rec-indicator')} />
              {isRecording ? 'Recording…' : 'Record'}
            </button>
          )}

          <button
            type="button"
            onClick={toggleSettings}
            className="rounded-md p-1.5 text-ak-muted transition-colors hover:bg-ak-sunk hover:text-ak-fg"
            title="Backend settings (Ctrl+/)"
            aria-label="Backend settings"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
