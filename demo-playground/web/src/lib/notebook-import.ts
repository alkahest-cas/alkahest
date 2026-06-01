import type { CellData } from '@/components/notebook/Cell';

export interface ImportedNotebookCell {
  code: string;
  cellType: CellData['cellType'];
}

function sourceToString(source: unknown): string {
  if (Array.isArray(source)) return source.join('');
  if (typeof source === 'string') return source;
  return '';
}

interface IpynbCell {
  cell_type: string;
  source?: unknown;
}

interface IpynbNotebook {
  nbformat?: number;
  cells?: IpynbCell[];
}

export function parseIpynbJson(data: unknown): ImportedNotebookCell[] {
  const nb = data as IpynbNotebook;
  if (!nb || !Array.isArray(nb.cells)) {
    throw new Error('Invalid notebook: missing cells array');
  }
  const cells: ImportedNotebookCell[] = [];
  for (const cell of nb.cells) {
    const code = sourceToString(cell.source);
    if (cell.cell_type === 'markdown') {
      cells.push({ code, cellType: 'markdown' });
    } else if (cell.cell_type === 'code') {
      cells.push({ code, cellType: 'code' });
    }
  }
  if (cells.length === 0) throw new Error('Notebook has no code or markdown cells');
  return cells;
}

export function parsePlaygroundJson(data: unknown): ImportedNotebookCell[] {
  if (!Array.isArray(data)) throw new Error('Invalid saved notebook format');
  const cells: ImportedNotebookCell[] = [];
  for (const item of data) {
    if (!item || typeof item !== 'object') continue;
    const { code, cellType } = item as { code?: unknown; cellType?: unknown };
    if (typeof code !== 'string') continue;
    if (cellType === 'markdown' || cellType === 'code') {
      cells.push({ code, cellType });
    }
  }
  if (cells.length === 0) throw new Error('No valid cells in file');
  return cells;
}

export function parseNotebookFile(text: string, filename: string): ImportedNotebookCell[] {
  const data = JSON.parse(text) as unknown;
  const lower = filename.toLowerCase();
  if (lower.endsWith('.ipynb')) return parseIpynbJson(data);
  if (lower.endsWith('.json')) {
    if (data && typeof data === 'object' && 'cells' in (data as object)) {
      return parseIpynbJson(data);
    }
    return parsePlaygroundJson(data);
  }
  throw new Error('Unsupported file type. Use .ipynb or .json');
}
