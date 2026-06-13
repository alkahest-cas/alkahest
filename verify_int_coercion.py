"""Verification script for fix/expr-int-coercion.

Demonstrates that Python `int` literals are now accepted where previously
only `Expr` was accepted:

1. ak.Matrix.from_rows([[0, 1], [-1, 0]]) — fully-numeric rows via active_pool().
2. ak.Matrix.from_rows([[x, 1], [0, x]]) — mixed Expr/int rows (pool inferred from Expr).
3. ak.series(ak.sin(x), x, 0, 5) — bare-int expansion point.
4. ak.emit_c(expr, x) and ak.emit_c(expr, [x]) — single var and one-element list.
"""

import alkahest as ak

p = ak.ExprPool()
x = p.symbol("x")

print("=== 1. Matrix.from_rows with fully-numeric rows (needs active_pool) ===")
with ak.context(pool=p):
    m = ak.Matrix.from_rows([[0, 1], [-1, 0]])
print(m, "shape:", m.shape())

print("\n=== 1b. Matrix(...) constructor with fully-numeric rows ===")
with ak.context(pool=p):
    m2 = ak.Matrix([[1, 0], [0, 1]])
print(m2, "shape:", m2.shape())

print("\n=== 2. Matrix.from_rows with mixed Expr/int rows (no context needed) ===")
m3 = ak.Matrix.from_rows([[x, 1], [0, x]])
print(m3, "shape:", m3.shape())

print("\n=== 3. series with bare-int expansion point ===")
s = ak.series(ak.sin(x), x, 0, 5)
print(s)

print("\n=== 4. emit_c with single Expr var ===")
poly = x**2 + 2 * x + 1
c_code = ak.emit_c(poly, x)
print(c_code)

print("\n=== 4b. emit_c with one-element list var ===")
c_code2 = ak.emit_c(poly, [x])
print(c_code2)
assert c_code == c_code2

print("\nAll checks passed.")
