import type { OutputItem } from '@/lib/execution';

/** Plain-text serialization of cell output for clipboard copy. */
export function outputsToPlainText(items: OutputItem[]): string {
  const parts: string[] = [];

  for (const item of items) {
    switch (item.type) {
      case 'text':
        parts.push(item.text);
        break;
      case 'error': {
        const trace = item.traceback.join('\n').replace(/\x1b\[[0-9;]*m/g, '');
        parts.push(`${item.ename}: ${item.evalue}${trace ? `\n${trace}` : ''}`);
        break;
      }
      case 'latex':
        parts.push(item.latex);
        break;
      case 'json':
        parts.push(JSON.stringify(item.data, null, 2));
        break;
      case 'lean':
        parts.push(item.source);
        break;
      case 'html': {
        if (typeof document !== 'undefined') {
          const el = document.createElement('div');
          el.innerHTML = item.html;
          parts.push(el.textContent ?? '');
        } else {
          parts.push(item.html);
        }
        break;
      }
      case 'image':
        parts.push(`[${item.format} image]`);
        break;
      default:
        break;
    }
  }

  return parts.join('\n').trimEnd();
}
