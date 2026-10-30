/**
 * Results for the Math tab's starter cells and examples, computed ahead of time
 * by the real translator and executor (scripts/precompute-math.py) so the tab
 * never looks empty on the static site. They are always labelled as
 * precomputed, and replaced by live results once a backend answers.
 */
import data from './math-precomputed.json';
import type { MathResult } from './math-api';

interface PrecomputedFile {
  alkahest_version: string;
  generated: string;
  starter: { input: string; result: MathResult }[];
  examples: Record<string, MathResult>;
}

const file = data as unknown as PrecomputedFile;

export const PRECOMPUTED_VERSION = file.alkahest_version;
export const PRECOMPUTED_DATE = file.generated;

function mentions(input: string, name: string): boolean {
  return new RegExp(`(^|[^A-Za-z0-9_])${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}([^A-Za-z0-9_]|$)`).test(input);
}

/**
 * The precomputed result for each input, or null. A starter result applies only
 * while every line above it is still the starter line (later starter lines can
 * use earlier definitions); an example result applies only when no line above
 * defines a name it mentions.
 */
export function precomputedResults(inputs: string[]): (MathResult | null)[] {
  const out: (MathResult | null)[] = [];
  let prefix = true;
  const defined: string[] = [];
  inputs.forEach((input, i) => {
    let res: MathResult | null = null;
    prefix = prefix && file.starter[i]?.input === input;
    if (prefix) {
      res = file.starter[i].result;
    } else {
      const ex = file.examples[input];
      if (ex && !defined.some((name) => mentions(input, name))) res = ex;
    }
    const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*:?=(?!=)/.exec(input);
    if (res?.definition) defined.push(res.definition.name);
    else if (m) defined.push(m[1]);
    out.push(res);
  });
  return out;
}
