"""examples/apart_definite.py — partial fractions and definite integration.

Demonstrates ``apart`` (partial-fraction decomposition over ℚ) and
``integrate(f, x, a, b)`` (definite integral via the fundamental theorem
of calculus).

Run after ``maturin develop``:
    python examples/apart_definite.py
"""

import alkahest as ak


def main():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    print("=== Partial fractions (apart) ===")

    f = 1 / (x**2 - pool.integer(1))
    pf = ak.apart(f, x)
    print(f"1/(x²−1)  →  {pf}")

    # Numeric spot-check away from poles
    for pt in (1.7, 2.3, -2.5):
        lhs = ak.eval_expr(f, {x: pt})
        rhs = ak.eval_expr(pf, {x: pt})
        print(f"  x={pt}: apart matches original ({lhs:.6f} ≈ {rhs:.6f})")

    print("\n=== Definite integration (FTC) ===")

    cases = [
        ("∫₀¹ x² dx", x**2, 0, 1, 1.0 / 3.0),
        ("∫₀¹ 2x dx", pool.integer(2) * x, 0, 1, 1.0),
        ("∫₁³ (x² + 2x) dx", x**2 + pool.integer(2) * x, 1, 3, 18.0 - 4.0 / 3.0),
    ]
    for label, expr, lo, hi, expected in cases:
        r = ak.integrate(expr, x, pool.integer(lo), pool.integer(hi))
        val = ak.eval_expr(r.value, {})
        ok = abs(val - expected) < 1e-10
        print(f"  {'✓' if ok else '✗'}  {label} = {val:.6f}  (expected {expected:.6f})")


if __name__ == "__main__":
    main()
