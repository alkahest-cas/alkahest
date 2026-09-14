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
    // Accept base64url (what the CLI now sends) and plain base64 (older links).
    // A plain-base64 payload arrives with its `+` already turned into spaces by
    // `URLSearchParams`, so map those back before decoding.
    const normalised = encoded.replace(/-/g, '+').replace(/_/g, '/').replace(/ /g, '+');
    const padded = normalised + '='.repeat((4 - (normalised.length % 4)) % 4);
    // `atob` yields one character per *byte*, so parsing its output directly
    // reads UTF-8 as Latin-1 and turns every en dash, Greek letter and maths
    // symbol in a demo into mojibake. Decode the bytes properly.
    const bytes = Uint8Array.from(atob(padded), (ch) => ch.charCodeAt(0));
    const codes: string[] = JSON.parse(new TextDecoder().decode(bytes));
    return codes.filter(Boolean);
  } catch (err) {
    // Silence here is how a corrupt payload became "the recording shows the
    // default notebook" instead of an error anybody could see.
    console.warn(`cellsFromDemoParam: could not decode ?${param}=`, err);
    return null;
  }
}
