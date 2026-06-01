/** Display helpers for notebook keyboard shortcuts (Colab-style chords + palette). */

export function isMacPlatform(): boolean {
  if (typeof navigator === 'undefined') return false;
  return /Mac|iPhone|iPad|iPod/.test(navigator.platform);
}

export function modLabel(): string {
  return isMacPlatform() ? '⌘' : 'Ctrl';
}

/** Colab-style chord: Mod+M, then B */
export function addCodeCellChordLabel(): string {
  return `${modLabel()}+M B`;
}

/** Colab-style chord: Mod+M, then Z */
export function undoDeleteCellChordLabel(): string {
  return `${modLabel()}+M Z`;
}

export function runCellShortcutLabel(): string {
  return `${modLabel()}+Enter`;
}

export function commandPaletteShortcutLabel(): string {
  return `${modLabel()}+Shift+P`;
}

export function saveNotebookShortcutLabel(): string {
  return `${modLabel()}+S`;
}

export function openSettingsShortcutLabel(): string {
  return `${modLabel()}+/`;
}

/** Chord timeout after Mod+M (ms). */
export const CHORD_TIMEOUT_MS = 2000;
