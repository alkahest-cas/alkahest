/** Starter cells for the Math tab: each shows one thing the language can do. */
export const MATH_STARTER_CELLS: string[] = [
  'integrate sin(x)^2 dx',
  'd/dx x^3 sin(x)',
  'solve x^2 - 5x + 6 = 0',
  'f = x^2 sin(x)',
  'd²/dx² f',
  'limit sin(x)/x as x -> 0',
  'sum 1/k^2 for k = 1 to oo',
  'integrate e^(x^2) dx',
];

/** More examples, offered as one-click chips under the palette. */
export const MATH_MORE_EXAMPLES: string[] = [
  '∫ x e^x dx',
  'integrate 1/(1+x^2) from 0 to oo',
  'derivative of tan(x)',
  'solve x^2 + y^2 = 1, x - y = 0 for x, y',
  'simplify (x^2-1)/(x-1)',
  'factor x^4 - 1',
  'expand (x+1)^5',
  'series exp(x) at x = 0 to order 6',
  'sin(pi/6) + sqrt(8)',
  'numeric pi to 50 digits',
];

/** A palette entry inserts `before` + selection + `after`, leaving the caret between. */
export interface PaletteItem {
  label: string;
  title: string;
  before: string;
  after?: string;
  /** Render the label as TeX (KaTeX) rather than text. */
  tex?: string;
}

export interface PaletteGroup {
  name: string;
  items: PaletteItem[];
}

export const MATH_PALETTE: PaletteGroup[] = [
  {
    name: 'Calculus',
    items: [
      { label: '∫', title: 'Integral: ∫ … dx', before: '∫ ', after: ' dx' },
      { label: '∫ₐᵇ', title: 'Definite integral', before: 'integrate ', after: ' from 0 to 1' },
      { label: 'd/dx', title: 'Derivative', before: 'd/dx ' },
      { label: 'd²/dx²', title: 'Second derivative', before: 'd²/dx² ' },
      { label: '∑', title: 'Sum', before: 'sum ', after: ' for k = 1 to oo' },
      { label: 'lim', title: 'Limit', before: 'limit ', after: ' as x -> 0' },
      { label: 'series', title: 'Taylor series', before: 'series ', after: ' at x = 0 to order 6' },
      { label: 'solve', title: 'Solve an equation', before: 'solve ', after: ' = 0' },
    ],
  },
  {
    name: 'Symbols',
    items: [
      { label: 'π', title: 'pi', before: 'π' },
      { label: 'e', title: "Euler's number e", before: 'e' },
      { label: 'i', title: 'Imaginary unit', before: 'i' },
      { label: '∞', title: 'Infinity', before: '∞' },
      { label: '√', title: 'Square root', before: '√(', after: ')' },
      { label: 'xⁿ', title: 'Power', before: '^' },
      { label: 'x²', title: 'Squared', before: '^2' },
      { label: 'a/b', title: 'Fraction', before: '(', after: ')/()' },
      { label: '( )', title: 'Parentheses', before: '(', after: ')' },
      { label: '|x|', title: 'Absolute value', before: '|', after: '|' },
      { label: 'n!', title: 'Factorial', before: '!' },
    ],
  },
  {
    name: 'Functions',
    items: [
      { label: 'sin', title: 'sine', before: 'sin(', after: ')' },
      { label: 'cos', title: 'cosine', before: 'cos(', after: ')' },
      { label: 'tan', title: 'tangent', before: 'tan(', after: ')' },
      { label: 'log', title: 'natural logarithm', before: 'log(', after: ')' },
      { label: 'exp', title: 'exponential', before: 'exp(', after: ')' },
      { label: 'atan', title: 'arctangent', before: 'atan(', after: ')' },
    ],
  },
  {
    name: 'Greek',
    items: ['α', 'β', 'γ', 'θ', 'λ', 'μ', 'σ', 'φ', 'ω'].map((g) => ({
      label: g,
      title: g,
      before: g,
    })),
  },
];
