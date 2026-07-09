"""
Rational functions of sin/cos via the Weierstrass half-angle substitution.

Covers integrands that are a *rational function* of sin(x)/cos(x) — a trig
function inside a denominator, e.g. ∫ 1/(2+cos x), ∫ 1/(1+sin x),
∫ 1/(5+4cos x), ∫ 1/(sin x + cos x), ∫ sin x/(1+sin x) — which the polynomial /
power / product trig fast-paths decline.  The engine substitutes t = tan(x/2)
(sin x = 2t/(1+t²), cos x = (1−t²)/(1+t²), dx = 2/(1+t²) dt), integrates the
resulting rational function in t, and back-substitutes t = tan(x/2).

Each result is verified by symbolic differentiation: d/dx F == f at several
sample points (chosen away from the poles of tan(x/2)).  The nicer closed forms
for ∫sin², ∫sec², ∫tan, etc. must be preserved, and non-rational-trig shapes
must decline (raise IntegrationError), never fabricate an answer.

Run after `maturin develop`:
    pytest tests/test_weierstrass_trig_integration.py -v
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

# Sample points avoiding x = π (pole of tan(x/2)).
_TEST_POINTS = (0.11, 0.37, 0.62, 0.83, 1.29, 2.4)


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


def test_one_over_2_plus_cos():
    """∫ 1/(2+cos x) dx = (2/√3)·atan(tan(x/2)/√3)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = 1 / (2 + cos(x))
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/(2+cos(x))")


def test_one_over_1_plus_sin():
    """∫ 1/(1+sin x) dx = −2/(1+tan(x/2))"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = 1 / (1 + sin(x))
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/(1+sin(x))")


def test_one_over_5_plus_4cos():
    """∫ 1/(5+4cos x) dx = (2/3)·atan(tan(x/2)/3)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = 1 / (5 + 4 * cos(x))
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/(5+4cos(x))")


def test_one_over_sin_plus_cos():
    """∫ 1/(sin x + cos x) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = 1 / (sin(x) + cos(x))
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/(sin(x)+cos(x))")


def test_sin_over_1_plus_sin():
    """∫ sin x/(1+sin x) dx"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) / (1 + sin(x))
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sin(x)/(1+sin(x))")


def test_one_over_2_plus_sin():
    """∫ 1/(2+sin x) dx = (2/√3)·atan((2·tan(x/2)+1)/√3)"""
    pool = ExprPool()
    x = pool.symbol("x")
    f = 1 / (2 + sin(x))
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/(2+sin(x))")


def test_nicer_forms_not_regressed():
    """The Weierstrass path must not disturb the dedicated trig closed forms."""
    pool = ExprPool()
    x = pool.symbol("x")
    for f, label in (
        (sin(x), "sin(x)"),
        (cos(x), "cos(x)"),
        (sin(x) ** 2, "sin(x)^2"),
        (cos(x) ** -2, "1/cos(x)^2"),
        (tan(x), "tan(x)"),
        (sin(2 * x) * cos(x), "sin(2x)*cos(x)"),
    ):
        r = integrate(f, x)
        check_antiderivative(x, f, r.value, label)


def test_non_rational_trig_declines():
    """∫ sin(x)/x is non-elementary — the Weierstrass rewrite hits a bare x and
    declines cleanly (no panic, raises IntegrationError)."""
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(IntegrationError):
        integrate(sin(x) / x, x)
