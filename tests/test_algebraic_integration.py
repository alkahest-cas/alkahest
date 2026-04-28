"""
Algebraic Risch integration tests (V1-2).

Tests ∫ f(x, sqrt(P(x))) dx for P of degree 0, 1, and 2 against symbolic
differentiation (verified by d/dx F == f after simplification).

Run after `maturin develop`:
    pytest tests/test_algebraic_integration.py -v
"""

import pytest

from alkahest.alkahest import ArbBall, ExprPool, diff, integrate, interval_eval, sqrt  # noqa: E402

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_TEST_POINTS = (1.5, 3.7, 0.25)


def check_antiderivative(pool, x, f, F, label=""):
    """Verify ∫ f dx = F numerically: d/dx F(x) == f(x) at several test points."""
    dF = diff(F, x).value
    for pt in _TEST_POINTS:
        bindings = {x: ArbBall(pt)}
        lhs = interval_eval(dF, bindings).mid
        rhs = interval_eval(f, bindings).mid
        assert abs(lhs - rhs) < 1e-9, (
            f"{label}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs} — mismatch\n"
            f"  F = {F}\n"
            f"  f = {f}"
        )


# ---------------------------------------------------------------------------
# P = constant
# ---------------------------------------------------------------------------

def test_sqrt_const_times_one():
    """∫ sqrt(5) dx = sqrt(5)·x"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sqrt(pool.integer(5))
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "sqrt(5)")


def test_sqrt_const_times_poly():
    """∫ sqrt(3)·x² dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sqrt(pool.integer(3)) * x ** 2
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "sqrt(3)*x^2")


# ---------------------------------------------------------------------------
# P = linear
# ---------------------------------------------------------------------------

def test_integral_sqrt_x():
    """∫ sqrt(x) dx = (2/3) x^(3/2)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sqrt(x)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "sqrt(x)")


def test_integral_x_times_sqrt_x():
    """∫ x·sqrt(x) dx = (2/5) x^(5/2)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * sqrt(x)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "x*sqrt(x)")


def test_integral_sqrt_2x_plus_1():
    """∫ sqrt(2x+1) dx = (1/3)(2x+1)^(3/2)"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = pool.integer(2) * x + pool.integer(1)
    f = sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "sqrt(2x+1)")


def test_integral_1_over_sqrt_x():
    """∫ 1/sqrt(x) dx = 2·sqrt(x)"""
    pool = ExprPool()
    x = pool.symbol("x")
    sx = sqrt(x)
    f = sx ** -1
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "1/sqrt(x)")


def test_integral_1_over_sqrt_x_plus_1():
    """∫ 1/sqrt(x+1) dx = 2·sqrt(x+1)"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x + pool.integer(1)
    f = (p ** -1) * sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "1/sqrt(x+1)")


def test_integral_x_sqrt_x_plus_1():
    """∫ x·sqrt(x+1) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x + pool.integer(1)
    f = x * sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "x*sqrt(x+1)")


# ---------------------------------------------------------------------------
# P = quadratic
# ---------------------------------------------------------------------------

def test_integral_sqrt_x2_plus_1():
    """∫ sqrt(x²+1) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x ** 2 + pool.integer(1)
    f = sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "sqrt(x^2+1)")


def test_integral_1_over_sqrt_x2_plus_1():
    """∫ 1/sqrt(x²+1) dx = log(x + sqrt(x²+1))"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x ** 2 + pool.integer(1)
    s_p = sqrt(p)
    f = (p ** -1) * s_p
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "1/sqrt(x^2+1)")


def test_integral_sqrt_x2_minus_1():
    """∫ sqrt(x²-1) dx  (evaluated at x > 1 to stay on the real branch)"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x ** 2 + pool.integer(-1)
    f = sqrt(p)
    r = integrate(f, x)
    # Use points where x²-1 > 0
    from alkahest.alkahest import ArbBall, diff, interval_eval
    dF = diff(r.value, x).value
    for pt in (1.5, 2.0, 3.7):
        bindings = {x: ArbBall(pt)}
        lhs = interval_eval(dF, bindings).mid
        rhs = interval_eval(f, bindings).mid
        assert abs(lhs - rhs) < 1e-9, f"sqrt(x²-1): d/dx F({pt})={lhs} vs f({pt})={rhs}"


def test_integral_const_times_sqrt_quadratic():
    """∫ 3·sqrt(x²+1) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x ** 2 + pool.integer(1)
    f = pool.integer(3) * sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "3*sqrt(x^2+1)")


# ---------------------------------------------------------------------------
# Mixed integrand A(x) + B(x)·sqrt(P)
# ---------------------------------------------------------------------------

def test_integral_poly_plus_sqrt():
    """∫ (x² + sqrt(x+1)) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x + pool.integer(1)
    f = x ** 2 + sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "x^2 + sqrt(x+1)")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_elliptic_raises():
    """∫ sqrt(x³+1) dx should raise (elliptic, non-elementary)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x ** 3 + pool.integer(1)
    with pytest.raises(Exception) as exc_info:
        integrate(sqrt(p), x)
    msg = str(exc_info.value)
    assert "E-INT-004" in msg or "elementary" in msg.lower() or "NonElementary" in msg
