import type { RunMenuAction } from '@/components/notebook/RunMenu';
import {
  addCodeCellChordLabel,
  commandPaletteShortcutLabel,
  openSettingsShortcutLabel,
  runCellShortcutLabel,
  saveNotebookShortcutLabel,
  undoDeleteCellChordLabel,
} from '@/lib/notebook-shortcuts';

export type NotebookCommandId =
  | RunMenuAction
  | 'run-cell'
  | 'add-code-cell'
  | 'add-markdown-cell'
  | 'undo-delete-cell'
  | 'save-notebook'
  | 'open-settings'
  | 'command-palette'
  | 'delete-focused-cell'
  | 'move-cell-up'
  | 'move-cell-down'
  | 'toggle-cell-type';

export interface NotebookCommandDef {
  id: NotebookCommandId;
  label: string;
  shortcut?: string;
  group: 'Run' | 'Cells' | 'Notebook' | 'View';
  keywords?: string;
}

export const NOTEBOOK_COMMAND_DEFS: NotebookCommandDef[] = [
  { id: 'command-palette', label: 'Command palette', shortcut: commandPaletteShortcutLabel(), group: 'View' },
  { id: 'run-cell', label: 'Run focused cell', shortcut: runCellShortcutLabel(), group: 'Run', keywords: 'execute' },
  { id: 'run-all', label: 'Run all', group: 'Run' },
  { id: 'restart', label: 'Restart session', group: 'Run', keywords: 'kernel' },
  { id: 'restart-run-all', label: 'Restart session and run all', group: 'Run' },
  { id: 'run-below', label: 'Run focused cell and all below', group: 'Run' },
  { id: 'interrupt', label: 'Interrupt execution', group: 'Run', keywords: 'stop cancel' },
  { id: 'clear-outputs', label: 'Clear outputs', group: 'Run' },
  { id: 'add-code-cell', label: 'Add code cell below', shortcut: addCodeCellChordLabel(), group: 'Cells', keywords: 'insert new' },
  { id: 'add-markdown-cell', label: 'Add markdown cell below', group: 'Cells' },
  { id: 'undo-delete-cell', label: 'Undo delete cell', shortcut: undoDeleteCellChordLabel(), group: 'Cells', keywords: 'restore' },
  { id: 'delete-focused-cell', label: 'Delete focused cell', group: 'Cells' },
  { id: 'move-cell-up', label: 'Move focused cell up', group: 'Cells' },
  { id: 'move-cell-down', label: 'Move focused cell down', group: 'Cells' },
  { id: 'toggle-cell-type', label: 'Toggle code / markdown', group: 'Cells' },
  { id: 'save-notebook', label: 'Save notebook', shortcut: saveNotebookShortcutLabel(), group: 'Notebook' },
  { id: 'open-settings', label: 'Open settings', shortcut: openSettingsShortcutLabel(), group: 'View' },
];

export const ADD_CODE_CELL_SHORTCUT_HINT = `Tip: ${addCodeCellChordLabel()} adds a code cell below`;
