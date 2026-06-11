/** Default starter notebook — mirrors demo-playground/demos/default_notebook.py */

export interface DefaultNotebookCell {
  code: string;
  cellType: 'code' | 'markdown';
}

export const DEFAULT_NOTEBOOK_CELLS: DefaultNotebookCell[] = [
  {
    cellType: 'markdown',
    code:
      '# Alkahest playground\n\n' +
      'Symbolic math in Python: simplify, differentiate, integrate, and more.\n' +
      'Run cells with **⌘/Ctrl+Enter**.',
  },
  {
    cellType: 'code',
    code:
      'import alkahest as ak\n' +
      'from alkahest import latex, sin, cos, exp, simplify, simplify_trig, diff, integrate\n\n' +
      'pool = ak.ExprPool()\n' +
      'x = pool.symbol("x")\n' +
      'two = pool.integer(2)\n' +
      'three = pool.integer(3)\n',
  },
  {
    cellType: 'markdown',
    code: '## Simplification',
  },
  {
    cellType: 'code',
    code:
      'r = simplify(x + pool.integer(0))\n' +
      'print("x + 0 = $$" + latex(r.value) + "$$")\n' +
      'print(f"({len(r.steps)} rewrite steps)")\n',
  },
  {
    cellType: 'markdown',
    code: '## Differentiation',
  },
  {
    cellType: 'code',
    code:
      'expr = x**three * sin(x)\n' +
      'r = diff(expr, x)\n' +
      'print("\\\\frac{d}{dx}(x^3 \\\\sin x) = $$" + latex(r.value) + "$$")\n',
  },
  {
    cellType: 'markdown',
    code: '## Integration',
  },
  {
    cellType: 'code',
    code:
      'r = integrate(exp(two * x), x)\n' +
      'print("\\\\int e^{2x}\\,dx = $$" + latex(r.value) + "$$")\n\n' +
      'r2 = integrate(cos(x), x)\n' +
      'print("\\\\int \\\\cos x\\,dx = $$" + latex(r2.value) + "$$")\n',
  },
  {
    cellType: 'markdown',
    code: '## Trigonometric identities',
  },
  {
    cellType: 'code',
    code:
      'r = simplify_trig(sin(x)**2 + cos(x)**2)\n' +
      'print("\\\\sin^2 x + \\\\cos^2 x = $$" + latex(r.value) + "$$")\n',
  },
  {
    cellType: 'markdown',
    code:
      '## Lean 4 certificate\n\n' +
      'Differentiate and emit a Mathlib proof — use **Verify in Lean** in the panel below.',
  },
  {
    cellType: 'code',
    code:
      'from playground_helpers import display_lean_cert\n\n' +
      'result = diff(x**three, x)\n' +
      'print("Symbolic:", result.value)\n' +
      'print("Steps:", len(result.steps))\n' +
      'display_lean_cert(result, operation="diff")\n',
  },
];
