"""examples/polynomials.py — MultiPoly and RationalFunction workflow.

Run after `maturin develop`:
    PYTHONPATH=python python examples/polynomials.py
"""

from alkahest.alkahest import ExprPool, MultiPoly, RationalFunction, UniPoly, simplify


def main():
    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")

    # ── UniPoly (FLINT-backed dense univariate polynomial) ────────────────

    print("=== UniPoly (univariate, FLINT-backed) ===")

    # Build x^3 - 2x + 1 using Python operators
    expr = x ** 3 + pool.integer(-2) * x + pool.integer(1)
    p = UniPoly.from_symbolic(expr, x)
    print(f"p      = {p}")
    print(f"degree = {p.degree}")
    print(f"coeffs = {p.coefficients()}")

    # GCD: gcd(x^2 - 1, x - 1) = x - 1
    xm1_expr = x + pool.integer(-1)
    x2m1_expr = x ** 2 + pool.integer(-1)
    p_xm1 = UniPoly.from_symbolic(xm1_expr, x)
    p_x2m1 = UniPoly.from_symbolic(x2m1_expr, x)
    g = p_xm1.gcd(p_x2m1)
    print(f"\ngcd(x-1, x^2-1) = {g}")

    # Arithmetic
    f = UniPoly.from_symbolic(x + pool.integer(1), x)
    print(f"\n(x+1) * (x-1) = {f * p_xm1}")
    print(f"(x+1) ^ 3     = {f ** 3}")

    # ── MultiPoly (sparse multivariate polynomial over ℤ) ─────────────────

    print("\n=== MultiPoly (multivariate, sparse) ===")

    # Build x^2*y + x*y^2 - 1
    x2y_expr = x ** 2 * y
    xy2_expr = x * y ** 2
    expr2 = x2y_expr + xy2_expr + pool.integer(-1)
    mp = MultiPoly.from_symbolic(expr2, [x, y])
    print(f"mp            = {mp}")
    print(f"total_degree  = {mp.total_degree}")
    print(f"int_content   = {mp.integer_content()}")

    # Bivariate addition
    xy_expr = x * y
    mp2 = MultiPoly.from_symbolic(xy_expr, [x, y])
    print(f"\n(x^2*y + x*y^2 - 1) + x*y = {mp + mp2}")

    # ── RationalFunction ──────────────────────────────────────────────────

    print("\n=== RationalFunction ===")

    # (x+1)/(x+1) → 1
    xp1_expr = x + pool.integer(1)
    rf = RationalFunction.from_symbolic(xp1_expr, xp1_expr, [x])
    print(f"(x+1)/(x+1) = {rf}")

    # (x^2-1)/(x-1) → x+1
    rf2 = RationalFunction.from_symbolic(x2m1_expr, xm1_expr, [x])
    print(f"(x^2-1)/(x-1) = {rf2}")

    # Arithmetic: (x/1) + (1/x) = (x^2 + 1)/x
    one = pool.integer(1)
    rf_x = RationalFunction.from_symbolic(x, one, [x])
    rf_1x = RationalFunction.from_symbolic(one, x, [x])
    rf_sum = rf_x + rf_1x
    print(f"\nx/1 + 1/x = {rf_sum}")

    # ── Simplification ────────────────────────────────────────────────────

    print("\n=== Simplification ===")

    r = simplify(x + pool.integer(0))
    print(f"simplify(x + 0)       = {r.value}")
    print(f"derivation steps      = {len(r.steps)}")

    nested = (x + pool.integer(0)) * pool.integer(1)
    r2 = simplify(nested)
    print(f"simplify((x+0)*1)     = {r2.value}")


if __name__ == "__main__":
    main()
