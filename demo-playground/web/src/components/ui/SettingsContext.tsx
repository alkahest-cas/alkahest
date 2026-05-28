'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import Settings from './Settings';

interface SettingsContextValue {
  isOpen: boolean;
  openSettings: () => void;
  closeSettings: () => void;
  toggleSettings: () => void;
  registerExport: (fn: (() => void) | null) => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (target.isContentEditable) return true;
  return Boolean(target.closest('.cm-editor, .cm-content'));
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const [exportFn, setExportFnState] = useState<(() => void) | null>(null);

  const openSettings = useCallback(() => setIsOpen(true), []);
  const closeSettings = useCallback(() => setIsOpen(false), []);
  const toggleSettings = useCallback(() => setIsOpen((v) => !v), []);
  const registerExport = useCallback((fn: (() => void) | null) => {
    setExportFnState(() => fn);
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape' && isOpen) {
        closeSettings();
        return;
      }
      const mod = e.ctrlKey || e.metaKey;
      if (!mod || e.key !== '/') return;
      if (isEditableTarget(e.target)) return;
      e.preventDefault();
      toggleSettings();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [toggleSettings, isOpen, closeSettings]);

  return (
    <SettingsContext.Provider value={{ isOpen, openSettings, closeSettings, toggleSettings, registerExport }}>
      {children}
      {isOpen && <Settings onClose={closeSettings} onExportNotebook={exportFn ?? undefined} />}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
}
