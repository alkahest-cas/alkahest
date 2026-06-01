import type { CellData } from '@/components/notebook/Cell';

const CLIPBOARD_MIME = 'application/x-alkahest-cell';

export interface CellClipboardPayload {
  type: 'alkahest-cell';
  cellType: CellData['cellType'];
  code: string;
}

export function cellToClipboardPayload(cell: Pick<CellData, 'cellType' | 'code'>): CellClipboardPayload {
  return { type: 'alkahest-cell', cellType: cell.cellType, code: cell.code };
}

export function serializeCellClipboard(cell: Pick<CellData, 'cellType' | 'code'>): string {
  return JSON.stringify(cellToClipboardPayload(cell));
}

export async function writeCellToClipboard(cell: Pick<CellData, 'cellType' | 'code'>): Promise<void> {
  const json = serializeCellClipboard(cell);
  try {
    await navigator.clipboard.write([
      new ClipboardItem({
        [CLIPBOARD_MIME]: new Blob([json], { type: CLIPBOARD_MIME }),
        'text/plain': new Blob([cell.code], { type: 'text/plain' }),
      }),
    ]);
  } catch {
    await navigator.clipboard.writeText(json);
  }
}
