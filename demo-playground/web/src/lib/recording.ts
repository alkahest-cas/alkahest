/** URL/query helpers shared by notebook and compare recording views. */

export function readZenFromUrl(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('zen') === '1';
}

export function readAutoRunFromUrl(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('autorun') === '1';
}

/** Hide CodeMirror line numbers (e.g. ?hideLineNumbers=1 for clean recordings). */
export function readHideLineNumbersFromUrl(): boolean {
  if (typeof window === 'undefined') return false;
  const params = new URLSearchParams(window.location.search);
  if (params.get('hideLineNumbers') === '1') return true;
  if (params.get('lineNumbers') === '0') return true;
  return false;
}

export function cellsFromDemoParam(param = 'demo'): string[] | null {
  if (typeof window === 'undefined') return null;
  const encoded = new URLSearchParams(window.location.search).get(param);
  if (!encoded) return null;
  try {
    const codes: string[] = JSON.parse(atob(encoded));
    return codes.filter(Boolean);
  } catch {
    return null;
  }
}
