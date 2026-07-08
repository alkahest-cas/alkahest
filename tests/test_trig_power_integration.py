"""
Powers and products of trigonometric functions via Fourier linearization.

Covers ∫ sin^m(x)·cos^n(x) dx (sin², cos², sin³, sin²·cos², …), the
different-frequency products ∫ sin(a·x)·cos(b·x) dx / sin·sin / cos·cos
(product-to-sum), and the small reciprocal-square family
∫ 1/cos²(x) = tan(x), ∫ 1/sin²(x) = −cot(x), ∫ tan²(x) = tan(x) − x.

Each result is verified by symbolic differentiation: d/dx F == f at several
sample points.  A shape outside the supported subset must decline (raise
IntegrationError), never fabricate an answer.

Run after `maturin develop`:
    pytest tests/test_trig_power_integration.py -v
"""

import math

import pytest
from alkahest.alkahest import (
    ExprPool,
    IntegrationError,
    cos,
    diff,
    eval_expr,
    integrate,
    sin,
    tan,
)

_TEST_POINTS = (0.11, 0.37, 0.62, 0.83, 1.29)


def check_antiderivative(x, f, big_f, label=""):
    """Verify ∫ f dx = F numerically: d/dx F(x) == f(x) at several points."""
    d_big_f = diff(big_f, x).value
    checked = 0
    for pt in _TEST_POINTS:
        lhs = eval_expr(d_big_f, {x: pt})
        rhs = eval_expr(f, {x: pt})
        if not math.isfinite(lhs) or not math.isfinite(rhs):
            continue
        assert abs(lhs - rhs) < 1e-9, (
            f"{label}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs} — mismatch\n  F = {big_f}\n  f = {f}"
        )
        checked += 1
    assert checked >= 2, f"{label}: not enough usable sample points"


def test_sin_squared():
    """∫ sin²(x) dx = x/2 − sin(2x)/4"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sin(x)^2")


def test_cos_squared():
    """∫ cos²(x) dx = x/2 + sin(2x)/4"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "cos(x)^2")


def test_sin_cubed():
    """∫ sin³(x) dx = cos³(x)/3 − cos(x)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) ** 3
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sin(x)^3")


def test_cos_cubed():
    """∫ cos³(x) dx = sin(x) − sin³(x)/3"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(x) ** 3
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "cos(x)^3")


def test_sin_squared_cos_squared():
    """∫ sin²(x)·cos²(x) dx = x/8 − sin(4x)/32"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) ** 2 * cos(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sin(x)^2*cos(x)^2")


def test_sin_2x_times_cos_x():
    """∫ sin(2x)·cos(x) dx — product-to-sum of different frequencies."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(2 * x) * cos(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sin(2x)*cos(x)")


def test_sin_x_times_sin_2x():
    """∫ sin(x)·sin(2x) dx — product-to-sum (cos family)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) * sin(2 * x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sin(x)*sin(2x)")


def test_cos_x_times_cos_3x():
    """∫ cos(x)·cos(3x) dx — product-to-sum (cos family)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(x) * cos(3 * x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "cos(x)*cos(3x)")


def test_sec_squared():
    """∫ 1/cos²(x) dx = tan(x)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(x) ** -2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/cos(x)^2")


def test_csc_squared():
    """∫ 1/sin²(x) dx = −cot(x)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) ** -2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/sin(x)^2")


def test_tan_squared():
    """∫ tan²(x) dx = tan(x) − x"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = tan(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "tan(x)^2")


def test_sin_squared_linear_arg():
    """∫ sin²(2x+1) dx — linear (non-unit) trig argument."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(2 * x + 1) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sin(2x+1)^2")


def test_basics_not_regressed():
    """The new fast-path must not disturb the already-working simple cases."""
    pool = ExprPool()
    x = pool.symbol("x")
    for f, label in (
        (sin(x), "sin(x)"),
        (cos(x), "cos(x)"),
        (tan(x), "tan(x)"),
        (sin(x) * cos(x), "sin(x)*cos(x)"),
        (x * sin(x), "x*sin(x)"),
    ):
        r = integrate(f, x)
        check_antiderivative(x, f, r.value, label)


def test_unsupported_trig_shape_declines():
    """∫ sin(x)/x is non-elementary; ∫ 1/cos¹⁰(x) is above the reciprocal-trig
    reduction cap (n ≤ 8) — both decline."""
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(IntegrationError):
        integrate(sin(x) / x, x)
    with pytest.raises(IntegrationError):
        integrate(cos(x) ** -10, x)
