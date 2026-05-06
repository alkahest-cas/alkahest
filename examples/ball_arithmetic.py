"""Phase 22 — Ball (interval) arithmetic with rigorous error bounds.

Demonstrates `ArbBall` and `interval_eval` for computing verified enclosures
of symbolic expressions.

The interface mirrors FLINT 3.x / Arb's `arb_t`:
  - `ArbBall(mid, rad, prec)` — real interval [mid ± rad] at MPFR precision `prec`
  - `interval_eval(expr, {var: ball})` — rigorous evaluation

Key property: the output ball is *guaranteed* to contain the true result
for any input in the given input balls.
"""

import math
import alkahest as ak
from alkahest import ExprPool, ArbBall, interval_eval, sin, cos, exp, sqrt

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

print("=" * 60)
print("Phase 22 — Ball arithmetic demo")
print("=" * 60)

# ── 1. Basic ball operations ──────────────────────────────────────────────
print("\n1. Basic ball operations")
a = ArbBall(2.0, 0.5)   # [1.5, 2.5]
b = ArbBall(3.0, 0.5)   # [2.5, 3.5]
print(f"  a = {a}   lo={a.lo:.2f}  hi={a.hi:.2f}")
print(f"  b = {b}   lo={b.lo:.2f}  hi={b.hi:.2f}")
c = a + b
print(f"  a+b = {c}   must contain 5.0: {c.contains(5.0)}")
c = a * b
print(f"  a*b = {c}   must contain 6.0: {c.contains(6.0)}")
c = a - ArbBall(1.0, 0.1)
print(f"  a - [1±0.1] = {c}   must contain 1.0: {c.contains(1.0)}")

# ── 2. Transcendental functions ───────────────────────────────────────────
print("\n2. Transcendental functions")
pi2 = ArbBall(math.pi / 2, 0.01)
s = pi2.sin()
print(f"  sin(π/2 ± 0.01) = {s}   contains 1.0: {s.contains(1.0)}")
z = ArbBall(0.0, 0.1)
e = z.exp()
print(f"  exp([−0.1, 0.1]) = {e}   contains 1.0: {e.contains(1.0)}")
pos = ArbBall(2.0, 0.5)  # [1.5, 2.5]
lv = pos.log()
print(f"  log([1.5, 2.5]) = {lv}   contains ln(2)={math.log(2):.4f}: {lv.contains(math.log(2))}")

# ── 3. interval_eval on symbolic expressions ─────────────────────────────
print("\n3. interval_eval: f(x) = x² + 1  at  x ∈ [2.9, 3.1]")
x2p1 = x ** 2 + pool.integer(1)
x_ball = ArbBall(3.0, 0.1)   # [2.9, 3.1]
result = interval_eval(x2p1, {x: x_ball})
print(f"  result = {result}")
print(f"  lo={result.lo:.4f}  hi={result.hi:.4f}")
# True: [2.9² + 1, 3.1² + 1] = [9.41, 10.61]
print(f"  contains 9.5: {result.contains(9.5)}")
print(f"  contains 10.0: {result.contains(10.0)}")
print(f"  contains 10.5: {result.contains(10.5)}")

# ── 4. Pythagorean identity with ball arithmetic ──────────────────────────
print("\n4. sin²(x) + cos²(x) = 1  (verified with balls)")
sin_x = sin(x)
cos_x = cos(x)
pythagorean = sin_x ** 2 + cos_x ** 2
for angle in [0.0, 0.5, 1.0, math.pi / 4]:
    x_b = ArbBall(angle, 1e-6)
    r = interval_eval(pythagorean, {x: x_b})
    ok = r.contains(1.0)
    print(f"  x={angle:.4f}: result={r}  contains 1: {ok}")

# ── 5. Precision comparison ───────────────────────────────────────────────
print("\n5. Higher precision gives tighter bounds: e^0")
for prec in [32, 64, 128, 256]:
    z = ArbBall(0.0, 1e-3, prec)
    e = z.exp()
    print(f"  prec={prec:3d}: e^[−0.001,0.001] radius = {e.rad:.3e}")

# ── 6. Division and sqrt ──────────────────────────────────────────────────
print("\n6. Division and sqrt")
a = ArbBall(9.0, 0.5)   # [8.5, 9.5]
s = a.sqrt()
print(f"  sqrt([8.5, 9.5]) = {s}   contains 3.0: {s.contains(3.0)}")
b_div = ArbBall(2.0, 0.1)  # [1.9, 2.1]
q = a / b_div
print(f"  [8.5,9.5] / [1.9,2.1] = {q}   contains 4.5: {q.contains(4.5)}")

print("\nDone.")
