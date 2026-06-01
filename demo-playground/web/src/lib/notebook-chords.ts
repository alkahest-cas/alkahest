import { CHORD_TIMEOUT_MS } from '@/lib/notebook-shortcuts';

export type ChordAction = 'add-code-cell' | 'undo-delete-cell';

export interface ChordHandlers {
  onAddCodeCell: () => void;
  onUndoDeleteCell: () => void;
}

/**
 * Colab-style chords: Mod+M then B (add code cell) or Z (undo delete).
 * Returns true when the event was consumed.
 */
export function handleNotebookChordKey(
  e: KeyboardEvent,
  chordActive: { current: boolean },
  chordTimer: { current: ReturnType<typeof setTimeout> | null },
  handlers: ChordHandlers,
): boolean {
  const mod = e.ctrlKey || e.metaKey;

  if (mod && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'm') {
    e.preventDefault();
    chordActive.current = true;
    if (chordTimer.current) clearTimeout(chordTimer.current);
    chordTimer.current = setTimeout(() => {
      chordActive.current = false;
      chordTimer.current = null;
    }, CHORD_TIMEOUT_MS);
    return true;
  }

  if (!chordActive.current) return false;
  if (mod || e.altKey) return false;

  const key = e.key.toLowerCase();
  if (key === 'b') {
    e.preventDefault();
    chordActive.current = false;
    if (chordTimer.current) clearTimeout(chordTimer.current);
    chordTimer.current = null;
    handlers.onAddCodeCell();
    return true;
  }
  if (key === 'z') {
    e.preventDefault();
    chordActive.current = false;
    if (chordTimer.current) clearTimeout(chordTimer.current);
    chordTimer.current = null;
    handlers.onUndoDeleteCell();
    return true;
  }

  return false;
}
