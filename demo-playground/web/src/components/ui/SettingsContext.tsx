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
import { readZenFromUrl } from '@/lib/recording';

interface SettingsContextValue {
  isOpen: boolean;
  openSettings: () => void;
  closeSettings: () => void;
  toggleSettings: () => void;
  registerExport: (fn: (() => void) | null) => void;
  registerImport: (fn: ((file: File) => void | Promise<void>) | null) => void;
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
  const [importFn, setImportFnState] = useState<((file: File) => void | Promise<void>) | null>(null);

  const openSettings = useCallback(() => setIsOpen(true), []);
  const closeSettings = useCallback(() => setIsOpen(false), []);
  const toggleSettings = useCallback(() => setIsOpen((v) => !v), []);
  const registerExport = useCallback((fn: (() => void) | null) => {
    setExportFnState(() => fn);
  }, []);
  const registerImport = useCallback((fn: ((file: File) => void | Promise<void>) | null) => {
    setImportFnState(() => fn);
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape' && isOpen) {
        closeSettings();
        return;
      }
      const mod = e.ctrlKey || e.metaKey;
      if (!mod || e.key !== '/') return;
      // Allow opening settings from the editor during zen/recording (Ctrl+/).
      if (!readZenFromUrl() && isEditableTarget(e.target)) return;
      e.preventDefault();
      toggleSettings();
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [toggleSettings, isOpen, closeSettings]);

  return (
    <SettingsContext.Provider
      value={{ isOpen, openSettings, closeSettings, toggleSettings, registerExport, registerImport }}
    >
      {children}
      {isOpen && (
        <Settings
          onClose={closeSettings}
          onExportNotebook={exportFn ?? undefined}
          onImportNotebook={importFn ?? undefined}
          showNotebookOptions
        />
      )}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
}
