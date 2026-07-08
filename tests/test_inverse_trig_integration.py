"""
Inverse-trigonometric integration by parts (atan / asin / acos).

Covers ∫ rest(x)·f(x) dx for f ∈ {atan, asin, acos} via integration by parts,
verified by symbolic differentiation: d/dx F == f at several test points.

Run after `maturin develop`:
    pytest tests/test_inverse_trig_integration.py -v
"""

import math

import pytest
from alkahest.alkahest import (
    ExprPool,
    acos,
    asin,
    atan,
    diff,
    eval_expr,
    integrate,
)

# atan is finite everywhere; asin/acos need |x| < 1. These points serve both.
_TEST_POINTS = (0.11, 0.37, 0.62, 0.83)


def check_antiderivative(x, f, F, label=""):
    """Verify ∫ f dx = F numerically: d/dx F(x) == f(x) at several test points."""
    dF = diff(F, x).value
    checked = 0
    for pt in _TEST_POINTS:
        lhs = eval_expr(dF, {x: pt})
        rhs = eval_expr(f, {x: pt})
        if not math.isfinite(lhs) or not math.isfinite(rhs):  # outside domain
            continue
        assert abs(lhs - rhs) < 1e-9, (
            f"{label}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs} — mismatch\n  F = {F}\n  f = {f}"
        )
        checked += 1
    assert checked >= 2, f"{label}: not enough usable sample points"


def test_atan():
    """∫ atan(x) dx = x·atan(x) − ½·log(1+x²)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = atan(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "atan(x)")


def test_x_times_atan():
    """∫ x·atan(x) dx = ½(x²+1)·atan(x) − x/2"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * atan(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*atan(x)")


def test_x_squared_times_atan():
    """∫ x²·atan(x) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**2 * atan(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x^2*atan(x)")


def test_atan_over_x_squared():
    """∫ atan(x)/x² dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = atan(x) * x ** (-2)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "atan(x)/x^2")


def test_asin():
    """∫ asin(x) dx = x·asin(x) + √(1−x²)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = asin(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "asin(x)")


def test_x_times_asin():
    """∫ x·asin(x) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * asin(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*asin(x)")


def test_acos():
    """∫ acos(x) dx = x·acos(x) − √(1−x²)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = acos(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "acos(x)")


def test_x_times_acos():
    """∫ x·acos(x) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * acos(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*acos(x)")


def test_atan_squared_declines():
    """∫ atan(x)² dx is out of scope — must raise, not return a wrong answer."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = atan(x) ** 2
    with pytest.raises(Exception):
        integrate(f, x)
