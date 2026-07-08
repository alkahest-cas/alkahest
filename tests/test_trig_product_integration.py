"""
Products of polynomial/exponential with a trigonometric factor.

Covers ∫ p(x)·sin(a·x+b) dx, ∫ p(x)·cos(a·x+b) dx (repeated integration by
parts) and ∫ exp(a·x)·sin(b·x) dx, ∫ exp(a·x)·cos(b·x) dx (cyclic IBP closed
form), verified by symbolic differentiation: d/dx F == f at several points.

Run after `maturin develop`:
    pytest tests/test_trig_product_integration.py -v
"""

import math

from alkahest.alkahest import (
    ExprPool,
    cos,
    diff,
    eval_expr,
    exp,
    integrate,
    sin,
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


def test_x_times_sin():
    """∫ x·sin(x) dx = sin(x) − x·cos(x)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * sin(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*sin(x)")


def test_x_times_cos():
    """∫ x·cos(x) dx = cos(x) + x·sin(x)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * cos(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*cos(x)")


def test_x_squared_times_sin():
    """∫ x²·sin(x) dx (repeated IBP)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**2 * sin(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x^2*sin(x)")


def test_x_squared_times_cos():
    """∫ x²·cos(x) dx (repeated IBP)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**2 * cos(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x^2*cos(x)")


def test_poly_times_sin():
    """∫ (x²+1)·sin(x) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = (x**2 + 1) * sin(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "(x^2+1)*sin(x)")


def test_x_times_sin_linear_arg():
    """∫ x·sin(2x+1) dx — linear (non-unit) trig argument."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * sin(2 * x + 1)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*sin(2x+1)")


def test_exp_times_sin():
    """∫ exp(x)·sin(x) dx = ½·exp(x)·(sin(x) − cos(x))"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = exp(x) * sin(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "exp(x)*sin(x)")


def test_exp_times_cos():
    """∫ exp(x)·cos(x) dx = ½·exp(x)·(sin(x) + cos(x))"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = exp(x) * cos(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "exp(x)*cos(x)")


def test_exp2x_times_cos3x():
    """∫ exp(2x)·cos(3x) dx = exp(2x)·(3·sin(3x) + 2·cos(3x))/13"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = exp(2 * x) * cos(3 * x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "exp(2x)*cos(3x)")
