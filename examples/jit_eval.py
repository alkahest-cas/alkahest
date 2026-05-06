"""Phase 21 — JIT / Interpreter-based compiled evaluation.

This example demonstrates how to compile symbolic expressions to fast
callable objects using `ak.compile_expr` and `ak.eval_expr`.

When built with `--features jit`, `compile_expr` emits LLVM IR and
JIT-compiles it to native machine code.  Without the feature, a fast
tree-walking interpreter is used (same API, same results).
"""

import math
import alkahest as ak
from alkahest import ExprPool, compile_expr, eval_expr, sin, cos, exp, sqrt

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

print("=" * 60)
print("Phase 21 — JIT / compiled evaluation demo")
print("=" * 60)

# ── 1. Simple arithmetic ──────────────────────────────────────────────────
print("\n1. Arithmetic: f(x) = x² + 2x + 1")
x2 = x ** 2
two_x = pool.integer(2) * x
one = pool.integer(1)
f_expr = x2 + two_x + one
f = compile_expr(f_expr, [x])
for v in [0.0, 1.0, 2.0, 3.0, -1.0]:
    got = f([v])
    expected = (v + 1) ** 2
    print(f"  f({v:4}) = {got:8.4f}  expected {expected:8.4f}  ✓" if abs(got - expected) < 1e-9
          else f"  f({v:4}) = {got:8.4f}  expected {expected:8.4f}  ✗ MISMATCH")

# ── 2. Multivariate ───────────────────────────────────────────────────────
print("\n2. Multivariate: g(x, y) = x² + y²  (Pythagorean triples)")
g_expr = x ** 2 + y ** 2
g = compile_expr(g_expr, [x, y])
for a, b, c in [(3, 4, 5), (5, 12, 13), (8, 15, 17)]:
    got = g([float(a), float(b)])
    print(f"  g({a},{b}) = {got:.1f}  = {c}² = {c**2}  ✓" if abs(got - c**2) < 1e-9
          else f"  MISMATCH at ({a},{b})")

# ── 3. Transcendental functions ───────────────────────────────────────────
print("\n3. Transcendental: h(x) = sin(x)² + cos(x)²  (must equal 1)")
sin_x = sin(x)
cos_x = cos(x)
h_expr = sin_x ** 2 + cos_x ** 2
h = compile_expr(h_expr, [x])
for v in [0.0, 0.5, 1.0, math.pi / 4, math.pi]:
    got = h([v])
    ok = abs(got - 1.0) < 1e-10
    print(f"  h({v:.4f}) = {got:.12f}  diff={abs(got-1.0):.2e}  {'✓' if ok else '✗'}")

# ── 4. eval_expr with dict bindings ──────────────────────────────────────
print("\n4. eval_expr with dict bindings")
e_expr = exp(x) * pool.integer(2)
for v in [0.0, 1.0, 2.0]:
    got = eval_expr(e_expr, {x: v})
    expected = math.exp(v) * 2
    ok = abs(got - expected) < 1e-9
    print(f"  2·e^{v} = {got:.6f}  expected {expected:.6f}  {'✓' if ok else '✗'}")

# ── 5. Performance: large polynomial ─────────────────────────────────────
print("\n5. Large polynomial: sum of x^k for k = 0..19")
terms = [x ** k for k in range(20)]
poly = terms[0]
for t in terms[1:]:
    poly = poly + t
f_poly = compile_expr(poly, [x])
# At x=1: sum = 20
got = f_poly([1.0])
print(f"  f(1) = {got:.1f}  expected 20.0  {'✓' if abs(got-20)<1e-9 else '✗'}")
# At x=2: sum = 2^20 - 1
got = f_poly([2.0])
expected = 2**20 - 1
print(f"  f(2) = {got:.1f}  expected {expected}  {'✓' if abs(got-expected)<1 else '✗'}")

print("\nDone.")
