"""examples/agent_workflow.py — end-to-end workflow from an agent/user perspective.

Covers the full "symbolic → numeric → code" pipeline that a typical agent
would execute: build expressions, differentiate, integrate, solve systems,
JIT-compile for batched evaluation, and emit C code.

Run with:
    python examples/agent_workflow.py
"""

import math
import alkahest as ak
from alkahest import (
    ExprPool,
    trace, grad, jit,
    diff, integrate, symbolic_grad,
    simplify, simplify_trig, simplify_log_exp,
    solve,
    Matrix, jacobian,
    compile_expr, eval_expr,
    horner, emit_c,
    collect_like_terms,
    sin, cos, exp, log, sqrt,
    tan, tanh, erf,
    to_stablehlo,
)

# ---------------------------------------------------------------------------
# 1. Expression construction
# ---------------------------------------------------------------------------

print("=" * 62)
print("1. Expression Construction")
print("=" * 62)

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")
z = pool.symbol("z")

# Integer and rational constants must be interned via the pool.
# Python-native ints cannot be used directly in expressions.
two = pool.integer(2)
half = pool.rational(1, 2)

# Arithmetic operators (+, *, **, /) are all overloaded on Expr.
quadratic = x**2 + two * x + pool.integer(1)
print(f"quadratic   : {quadratic}")
print(f"rational    : {half * x}")
print(f"trig combo  : {sin(x)**2 + cos(x)**2}")

# Expressions are hash-consed: identical trees share storage.
a = x**2 + pool.integer(1)
b = x**2 + pool.integer(1)
print(f"hash-cons   : a is b? {str(a) == str(b)}")

# ---------------------------------------------------------------------------
# 2. Simplification
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("2. Simplification")
print("=" * 62)

r = simplify(x + pool.integer(0))
print(f"x + 0              → {r.value}  ({len(r.steps)} steps)")

r = simplify_trig(sin(x)**2 + cos(x)**2)
print(f"sin²+cos²          → {r.value}")

r = simplify_trig(sin(pool.integer(-1) * x))
print(f"sin(-x)            → {r.value}")

r = simplify_log_exp(log(exp(x)))
print(f"log(exp(x))        → {r.value}")

r = simplify_log_exp(exp(log(x)))
print(f"exp(log(x))        → {r.value}")

r = collect_like_terms(x + x + two * x + y)
print(f"x+x+2x+y           → {r.value}")

# ---------------------------------------------------------------------------
# 3. Differentiation
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("3. Differentiation")
print("=" * 62)

rules = [
    ("x³",               x**3,              "3x²"),
    ("sin(x)",           sin(x),            "cos(x)"),
    ("exp(x)",           exp(x),            "exp(x)"),
    ("sin(x)·exp(x)",    sin(x)*exp(x),     "sin·exp + cos·exp"),
    ("log(x)",           log(x),            "1/x"),
]
for label, expr, _ in rules:
    d = diff(expr, x)
    print(f"d/dx {label:20s} = {d.value}")

# Reverse-mode gradient (all partials at once)
f3 = x**2 + y**2 + z**2
grads = symbolic_grad(f3, [x, y, z])
print(f"\n∇(x²+y²+z²) = {grads}")

# ---------------------------------------------------------------------------
# 4. Integration
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("4. Symbolic Integration")
print("=" * 62)

integrands = [
    ("x²",           x**2),
    ("sin(x)",       sin(x)),
    ("1/x",          x**-1),
    ("exp(2x)",      exp(two * x)),
    ("x·exp(x)",     x * exp(x)),
    ("log(x)",       log(x)),
]
for label, expr in integrands:
    try:
        r = integrate(expr, x)
        print(f"∫ {label:12s} dx = {r.value}")
    except Exception as e:
        print(f"∫ {label:12s} dx = [not elementary: {e}]")

# ---------------------------------------------------------------------------
# 5. Polynomial system solver
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("5. Polynomial System Solver")
print("=" * 62)

# Circle–line intersection: x² + y² = 1, y = x
solutions = solve(
    [x**2 + y**2 + pool.integer(-1), y + pool.integer(-1)*x],
    [x, y]
)
print(f"Circle ∩ line (x²+y²=1, y=x):")
for s in solutions:
    vals = {str(k): round(v, 6) for k, v in s.items()}
    print(f"  {vals}")

# Verify: each solution satisfies both equations numerically
for s in solutions:
    xv = list(s.values())[0]
    yv = list(s.values())[1]
    res1 = abs(xv**2 + yv**2 - 1.0)
    res2 = abs(xv - yv)
    ok = res1 < 1e-9 and res2 < 1e-9
    print(f"  residual check: {res1:.2e}, {res2:.2e}  {'✓' if ok else '✗'}")

# Linear system
linear_soln = solve(
    [x + y + pool.integer(-3), x + pool.integer(-1)*y + pool.integer(-1)],
    [x, y]
)
print(f"\nx+y=3, x-y=1 → {[{str(k): round(v,4) for k,v in s.items()} for s in linear_soln]}")

# ---------------------------------------------------------------------------
# 6. Matrix + Jacobian
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("6. Matrix & Jacobian")
print("=" * 62)

# Rotation matrix
c = cos(x)
s_x = sin(x)
R = Matrix.from_rows([
    [c,                     pool.integer(-1)*s_x],
    [s_x,                   c],
])
print(f"Rotation matrix R:\n  {R.to_list()}")
print(f"det(R) = {R.det()}")

# Jacobian of a 3D → 2D map
f1 = x**2 + y
f2 = sin(x) * y
J = jacobian([f1, f2], [x, y])
print(f"\nJacobian of [x²+y, sin(x)·y] wrt [x,y]:")
for r in range(J.rows):
    row = [str(J.get(r, c)) for c in range(J.cols)]
    print(f"  [{', '.join(row)}]")

# ---------------------------------------------------------------------------
# 7. JAX-style trace / grad / jit
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("7. trace / grad / jit")
print("=" * 62)

@ak.trace(pool)
def energy(x, y):
    return x**2 + ak.sin(y) * ak.exp(x)

print(f"energy expr   : {energy.expr}")
print(f"energy(1, 0)  : {energy(1.0, 0.0):.6f}")

grad_energy = grad(energy)
gs = grad_energy(1.0, 0.0)
print(f"∇energy(1, 0) : [{gs[0]:.6f}, {gs[1]:.6f}]")

fast_energy = jit(energy)
print(f"jit(1, 0)     : {fast_energy(1.0, 0.0):.6f}")
print(f"jit(2, 1)     : {fast_energy(2.0, 1.0):.6f}")

# ---------------------------------------------------------------------------
# 8. Code emission
# ---------------------------------------------------------------------------

print("\n" + "=" * 62)
print("8. Code Emission")
print("=" * 62)

poly = pool.integer(1) + pool.integer(2)*x + pool.integer(3)*x**2 + pool.integer(4)*x**3
print(f"Polynomial    : {poly}")
print(f"Horner form   : {horner(poly, x)}")
c_code = emit_c(poly, x, "x", "poly_eval")
print(f"C code:\n{c_code}")

# StableHLO for XLA/JAX integration
stablehlo = to_stablehlo(sin(x) + exp(y), [x, y], fn_name="my_fn")
print(f"StableHLO:\n{stablehlo}")

# ---------------------------------------------------------------------------
# 9. Error handling (V1-3 structured errors)
# ---------------------------------------------------------------------------

print("=" * 62)
print("9. Error Handling (V1-3 diagnostic codes)")
print("=" * 62)

from alkahest import ConversionError, IntegrationError, poly_normal

try:
    poly_normal(sin(x), [x])
except ConversionError as e:
    print(f"ConversionError  : {e}")
    print(f"  .code          : {getattr(e, 'code', 'N/A')}")
    print(f"  .remediation   : {getattr(e, 'remediation', 'N/A')}")

try:
    # exp(x²) has no elementary antiderivative
    integrate(exp(x**2), x)
except IntegrationError as e:
    print(f"IntegrationError : {e}")
    print(f"  .code          : {getattr(e, 'code', 'N/A')}")
    print(f"  .remediation   : {getattr(e, 'remediation', 'N/A')}")

print("\nDone.")
