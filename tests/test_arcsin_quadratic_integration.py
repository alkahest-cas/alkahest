"""
Arcsin-family algebraic integration tests: ∫ R(x, sqrt(a·x²+b·x+c)) dx with a
negative leading coefficient a < 0.

For a < 0 the conic y² = a·x²+b·x+c has no real point at infinity, so the Euler
substitution used for a > 0 has no real form; the natural normal form is arcsin.
Completing the square, a·x²+b·x+c = |a|·(k² − (x−h)²), and the shift w = x−h
reduces the integrand to poly(w)/√(k²−w²), i.e. asin/√ table integrals.

Each antiderivative is verified numerically by d/dx F == f at sample points
inside the real interval (where the radicand is positive).

Run after `maturin develop`:
    pytest tests/test_arcsin_quadratic_integration.py -v
"""

from alkahest.alkahest import ArbBall, ExprPool, diff, integrate, interval_eval, sqrt


def check_antiderivative(x, f, F, points, label=""):
    """Verify ∫ f dx = F: d/dx F(x) == f(x) at points inside the real interval."""
    dF = diff(F, x).value
    for pt in points:
        bindings = {x: ArbBall(pt)}
        lhs = interval_eval(dF, bindings).mid
        rhs = interval_eval(f, bindings).mid
        assert abs(lhs - rhs) < 1e-9, (
            f"{label}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs} — mismatch\n  F = {F}\n  f = {f}"
        )


def test_one_over_sqrt_2_minus_3x2():
    """∫ 1/√(2−3x²) dx = (1/√3)·asin(x·√(3/2)). Positive on |x| < √(2/3)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = (pool.integer(2) - pool.integer(3) * x**2) ** pool.rational(-1, 2)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, (-0.6, -0.1, 0.3, 0.7), "1/sqrt(2-3x^2)")


def test_one_over_sqrt_5_minus_2x2():
    """∫ 1/√(5−2x²) dx. Positive on |x| < √(5/2) ≈ 1.58."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = (pool.integer(5) - pool.integer(2) * x**2) ** pool.rational(-1, 2)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, (-1.2, -0.4, 0.5, 1.3), "1/sqrt(5-2x^2)")


def test_one_over_sqrt_neg_x2_2x_3():
    """∫ 1/√(−x²+2x+3) dx = asin((x−1)/2). Positive on (−1, 3)."""
    pool = ExprPool()
    x = pool.symbol("x")
    radicand = -(x**2) + pool.integer(2) * x + pool.integer(3)
    f = radicand ** pool.rational(-1, 2)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, (-0.5, 0.7, 1.6, 2.5), "1/sqrt(-x^2+2x+3)")


def test_x_over_sqrt_5_minus_2x2():
    """∫ x/√(5−2x²) dx = −√(5−2x²)/2. Positive on |x| < √(5/2)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * (pool.integer(5) - pool.integer(2) * x**2) ** pool.rational(-1, 2)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, (-1.2, -0.4, 0.5, 1.3), "x/sqrt(5-2x^2)")


def test_sqrt_2_minus_3x2_real_asin_form():
    """∫ √(2−3x²) dx = x√(2−3x²)/2 + (1/√3)·asin(x√(3/2)) — a real asin form."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sqrt(pool.integer(2) - pool.integer(3) * x**2)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, (-0.6, -0.1, 0.3, 0.7), "sqrt(2-3x^2)")


def test_one_over_sqrt_1_minus_x2():
    """Regression: ∫ 1/√(1−x²) dx = asin(x). Positive on (−1, 1)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = (pool.integer(1) - x**2) ** pool.rational(-1, 2)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, (-0.7, -0.2, 0.3, 0.8), "1/sqrt(1-x^2)")


def test_one_over_sqrt_4_minus_x2():
    """Regression: ∫ 1/√(4−x²) dx = asin(x/2). Positive on (−2, 2)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = (pool.integer(4) - x**2) ** pool.rational(-1, 2)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, (-1.5, -0.4, 0.6, 1.7), "1/sqrt(4-x^2)")
