"""
Phase 3 demo: Polynomial Representation
========================================
Demonstrates UniPoly, MultiPoly, and RationalFunction from alkahest Phase 3.

Run with:
    .venv/bin/python3 examples/phase3_polynomials.py
"""

import alkahest as ak

print(f"alkahest version: {ak.version()}\n")

# ---------------------------------------------------------------------------
# ExprPool — the symbolic expression builder
# ---------------------------------------------------------------------------
# All symbols live in a pool. The pool hash-conses expressions so identical
# sub-expressions always map to the same id.

pool = ak.ExprPool()

# ---------------------------------------------------------------------------
# UniPoly demo
# ---------------------------------------------------------------------------
print("=== UniPoly (univariate polynomial over ℤ) ===\n")

x = pool.symbol("x")

# Build x^2 + 2x + 1
xsq   = pool.pow(x, pool.integer(2))
two_x = pool.mul([pool.integer(2), x])
one   = pool.integer(1)
expr  = pool.add([xsq, two_x, one])

p = ak.UniPoly.from_symbolic(expr, x, pool)
print(f"p = {p}")                             # x^2+2*x+1
print(f"  degree : {p.degree()}")             # 2
print(f"  coeffs : {p.coefficients()}")       # [1, 2, 1]  (ascending degree)

# q = x + 1
xp1 = ak.UniPoly.from_symbolic(
    pool.add([x, pool.integer(1)]), x, pool
)
# r = x - 1
xm1 = ak.UniPoly.from_symbolic(
    pool.add([x, pool.integer(-1)]), x, pool
)

print(f"\nq = x+1 = {xp1}")
print(f"r = x-1 = {xm1}")

print(f"\np + q = {p + xp1}")     # x^2+3*x+2
print(f"p - q = {p - xp1}")      # x^2+x
print(f"q * r = {xp1 * xm1}")    # x^2-1  (difference of squares)

# Powers
print(f"\n(x+1)^3 = {xp1.pow(3)}")   # x^3+3*x^2+3*x+1
print(f"(x-1)^3 = {xm1.pow(3)}")    # x^3-3*x^2+3*x-1

# GCD: gcd(x^2-1, x-1)  →  x-1  (up to leading coefficient sign)
x2m1 = ak.UniPoly.from_symbolic(
    pool.add([pool.pow(x, pool.integer(2)), pool.integer(-1)]), x, pool
)
g = x2m1.gcd(xm1)
print(f"\ngcd(x^2-1, x-1) = {g}  (degree {g.degree()})")

# ---------------------------------------------------------------------------
# MultiPoly demo
# ---------------------------------------------------------------------------
print("\n=== MultiPoly (multivariate polynomial over ℤ) ===\n")

# MultiPoly uses positional variable notation in display: the first element of
# the vars list is x0, the second is x1, etc.

pool2 = ak.ExprPool()
x2 = pool2.symbol("x")
y2 = pool2.symbol("y")

# x^2 + x*y + y^2
expr_a = pool2.add([
    pool2.pow(x2, pool2.integer(2)),
    pool2.mul([x2, y2]),
    pool2.pow(y2, pool2.integer(2)),
])
a = ak.MultiPoly.from_symbolic(expr_a, [x2, y2], pool2)
print(f"a = x^2 + xy + y^2 = {a}")
print(f"  total_degree    : {a.total_degree()}")    # 2
print(f"  integer_content : {a.integer_content()}")  # 1

# 6x + 4  →  content = 2
expr_b = pool2.add([pool2.mul([pool2.integer(6), x2]), pool2.integer(4)])
b = ak.MultiPoly.from_symbolic(expr_b, [x2, y2], pool2)
print(f"\nb = 6x+4 = {b}")
print(f"  integer_content : {b.integer_content()}")  # 2

# Arithmetic
c = ak.MultiPoly.from_symbolic(pool2.add([x2, y2]), [x2, y2], pool2)
d = ak.MultiPoly.from_symbolic(
    pool2.add([x2, pool2.integer(-1)]), [x2, y2], pool2
)

print(f"\nc = x+y  = {c}")
print(f"d = x-1  = {d}")
print(f"c + d    = {c + d}")   # 2x+y-1
print(f"c * d    = {c * d}")   # x^2 + xy - x - y

# ---------------------------------------------------------------------------
# RationalFunction demo
# ---------------------------------------------------------------------------
print("\n=== RationalFunction ===\n")

pool3 = ak.ExprPool()
x3 = pool3.symbol("x")
y3 = pool3.symbol("y")

# (6x) / 4  →  normalises to (3x) / 2
rf1 = ak.RationalFunction.from_symbolic(
    pool3.mul([pool3.integer(6), x3]),
    pool3.integer(4),
    [x3, y3],
    pool3,
)
print(f"(6x)/(4)   → {rf1}")
print(f"  numer : {rf1.numer()}")   # 3x0
print(f"  denom : {rf1.denom()}")   # 2

# x / 1  →  denominator 1 is elided in display
rf2 = ak.RationalFunction.from_symbolic(
    x3, pool3.integer(1), [x3, y3], pool3
)
print(f"\nx/1        → {rf2}")

# x / (-2)  →  sign normalised so denom has positive leading coefficient
rf3 = ak.RationalFunction.from_symbolic(
    x3, pool3.integer(-2), [x3, y3], pool3
)
print(f"x/(-2)     → {rf3}")

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
print("\n=== Error handling ===\n")

pool4 = ak.ExprPool()
x4 = pool4.symbol("x")
y4 = pool4.symbol("y")

# y is a free symbol when building a UniPoly in x
try:
    ak.UniPoly.from_symbolic(pool4.add([x4, y4]), x4, pool4)
except ValueError as e:
    print(f"Free symbol    : {e}")

# Zero denominator
try:
    ak.RationalFunction.from_symbolic(
        x4, pool4.integer(0), [x4], pool4
    )
except ValueError as e:
    print(f"Zero denom     : {e}")

# Negative exponent
try:
    ak.UniPoly.from_symbolic(
        pool4.pow(x4, pool4.integer(-1)), x4, pool4
    )
except ValueError as e:
    print(f"Negative exp   : {e}")

print("\nDone.")
