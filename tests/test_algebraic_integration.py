"""
Algebraic Risch integration tests (V1-2).

Tests ∫ f(x, sqrt(P(x))) dx for P of degree 0, 1, and 2 against symbolic
differentiation (verified by d/dx F == f after simplification).

Run after `maturin develop`:
    pytest tests/test_algebraic_integration.py -v
"""

import pytest
from alkahest.alkahest import ArbBall, ExprPool, diff, integrate, interval_eval, sqrt

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
            f"{label}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs} — mismatch\n  F = {F}\n  f = {f}"
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
    f = sqrt(pool.integer(3)) * x**2
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
    f = sx**-1
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "1/sqrt(x)")


def test_integral_1_over_sqrt_x_plus_1():
    """∫ 1/sqrt(x+1) dx = 2·sqrt(x+1)"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x + pool.integer(1)
    f = (p**-1) * sqrt(p)
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
    p = x**2 + pool.integer(1)
    f = sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "sqrt(x^2+1)")


def test_integral_1_over_sqrt_x2_plus_1():
    """∫ 1/sqrt(x²+1) dx = log(x + sqrt(x²+1))"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**2 + pool.integer(1)
    s_p = sqrt(p)
    f = (p**-1) * s_p
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "1/sqrt(x^2+1)")


def test_integral_sqrt_x2_minus_1():
    """∫ sqrt(x²-1) dx  (evaluated at x > 1 to stay on the real branch)"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**2 + pool.integer(-1)
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
    p = x**2 + pool.integer(1)
    f = pool.integer(3) * sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "3*sqrt(x^2+1)")


# ---------------------------------------------------------------------------
# General quadratic radicand: leading coeff `a` not a rational square, so the
# completed-square Euler reduction (t = u + √(u²+k), √a factored out) is needed.
# ---------------------------------------------------------------------------


def test_integral_x_over_sqrt_2x2_plus_3():
    """∫ x/√(2x²+3) dx = √(2x²+3)/2  (a=2 not a square; previously declined)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = pool.integer(2) * x**2 + pool.integer(3)
    f = x * (sqrt(p) ** -1)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "x/sqrt(2x^2+3)")


def test_integral_1_over_sqrt_3x2_2x_2():
    """∫ 1/√(3x²+2x+2) dx — asinh/log form (a=3, discriminant < 0)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = pool.integer(3) * x**2 + pool.integer(2) * x + pool.integer(2)
    f = sqrt(p) ** -1
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "1/sqrt(3x^2+2x+2)")


# ---------------------------------------------------------------------------
# Mixed integrand A(x) + B(x)·sqrt(P)
# ---------------------------------------------------------------------------


def test_integral_poly_plus_sqrt():
    """∫ (x² + sqrt(x+1)) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x + pool.integer(1)
    f = x**2 + sqrt(p)
    r = integrate(f, x)
    check_antiderivative(pool, x, f, r.value, "x^2 + sqrt(x+1)")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_elliptic_second_kind_emits_elliptic():
    """∫ sqrt(x³+1) dx now emits a second-kind elliptic form (PR3).

    Previously raised NonElementary; the algebraic engine reduces it to an
    algebraic part plus `EllipticF` (and, for other radicands, `EllipticE`),
    gate-verified by `d/dx F = √(x³+1)`.
    """
    import math

    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 + pool.integer(1)
    res = integrate(sqrt(p), x)
    f = res.value
    s = str(f)
    assert "EllipticF" in s, f"expected EllipticF in {s}"

    # d/dx F = √(x³+1) where x³+1 > 0.  Evaluate the derivative tree directly
    # (asin/acos are not supported by the native eval).
    d = diff(f, x).value
    unary = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sqrt": math.sqrt,
    }

    def _num(expr, xv):
        n = expr.node()
        tag = n[0]
        if tag == "symbol":
            return xv
        if tag == "integer":
            return float(int(n[1]))
        if tag == "rational":
            return float(int(n[1])) / float(int(n[2]))
        if tag == "add":
            return sum(_num(a, xv) for a in n[1])
        if tag == "mul":
            prod = 1.0
            for a in n[1]:
                prod *= _num(a, xv)
            return prod
        if tag == "pow":
            return _num(n[1], xv) ** _num(n[2], xv)
        if tag == "func":
            name, args = n[1], n[2]
            if name in unary and len(args) == 1:
                return unary[name](_num(args[0], xv))
        raise AssertionError(f"cannot evaluate node {n}")

    checked = 0
    for xv in (0.5, 1.0, 2.0, 3.0):
        rhs = math.sqrt(xv**3 + 1.0)
        lhs = _num(d, xv)
        assert abs(lhs - rhs) < 1e-6 * (1.0 + abs(rhs)), f"x={xv}: {lhs} vs {rhs}\n  F={f}"
        checked += 1
    assert checked >= 3


def test_quintic_still_non_elementary():
    """∫ sqrt(x⁵+1) dx is genus-2 — no reduction, still raises NonElementary."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**5 + pool.integer(1)
    with pytest.raises(Exception) as exc_info:
        integrate(sqrt(p), x)
    msg = str(exc_info.value)
    assert "E-INT-004" in msg or "elementary" in msg.lower() or "NonElementary" in msg
