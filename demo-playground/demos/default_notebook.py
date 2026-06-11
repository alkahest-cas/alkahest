# Default starter notebook for the Alkahest demo playground.
# Cells are delimited by "# ---" for use with the record CLI:
#
#   npx tsx cli/src/index.ts record --code demos/default_notebook.py --output out.webm

## Alkahest playground

Symbolic math in Python: simplify, differentiate, integrate, and more.
Run cells with **⌘/Ctrl+Enter**.

# ---

import alkahest as ak
from alkahest import latex, sin, cos, exp, simplify, simplify_trig, diff, integrate

pool = ak.ExprPool()
x = pool.symbol("x")
two = pool.integer(2)
three = pool.integer(3)

# ---

## Simplification

# ---

r = simplify(x + pool.integer(0))
print("x + 0 = $$" + latex(r.value) + "$$")
print(f"({len(r.steps)} rewrite steps)")

# ---

## Differentiation

# ---

expr = x**three * sin(x)
r = diff(expr, x)
print("\\frac{d}{dx}(x^3 \\sin x) = $$" + latex(r.value) + "$$")

# ---

## Integration

# ---

r = integrate(exp(two * x), x)
print("\\int e^{2x}\\,dx = $$" + latex(r.value) + "$$")

r2 = integrate(cos(x), x)
print("\\int \\cos x\\,dx = $$" + latex(r2.value) + "$$")

# ---

## Trigonometric identities

# ---

r = simplify_trig(sin(x)**2 + cos(x)**2)
print("\\sin^2 x + \\cos^2 x = $$" + latex(r.value) + "$$")

# ---

## Lean 4 certificate

Differentiate and emit a Mathlib proof — use **Verify in Lean** in the panel below.

# ---

from playground_helpers import display_lean_cert

result = diff(x**three, x)
print("Symbolic:", result.value)
print("Steps:", len(result.steps))
display_lean_cert(result, operation="diff")
