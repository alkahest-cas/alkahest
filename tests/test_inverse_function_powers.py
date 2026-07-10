"""
Integer powers of inverse (trig / hyperbolic) functions.

Covers ∫ p(x)·f(x)^k dx for f ∈ {asin, acos, asinh, acosh, atan, atanh} and
integer k ≥ 2, via the inverse-function integration-by-parts reduction
(∫ p·f^k = P·f^k − k·∫ P·f^{k−1}·f').

The key correctness property is the algebraic-vs-rational-derivative asymmetry:

  * asin/acos/asinh/acosh have *algebraic* derivatives (1/√(1∓x²), 1/√(x²±1)),
    so their integer powers are ELEMENTARY and must close (verified by d/dx).
  * atan/atanh have *rational* derivatives (1/(1±x²)); for k ≥ 2 the residual
    ∫ log(1∓x²)/(1∓x²) dx is non-elementary, so those integrals must DECLINE
    cleanly (raise) rather than return a wrong closed form.

Run after `maturin develop`:
    pytest tests/test_inverse_function_powers.py -v
"""

import math

import pytest
from alkahest.alkahest import (
    ExprPool,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    diff,
    eval_expr,
    integrate,
)

# asin/acos/atanh need |x| < 1; asinh finite on ℝ; acosh needs x > 1.  Out-of-
# domain points are skipped in check_antiderivative, so one spread covers all.
_TEST_POINTS = (0.11, 0.37, 0.62, 0.83, 1.5, 2.3, 3.1)


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


# --- Elementary: algebraic-derivative inverse functions (must close) ---------


def test_asin_squared():
    """∫ asin(x)² dx = x·asin(x)² + 2√(1−x²)·asin(x) − 2x."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = asin(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "asin(x)^2")


def test_acos_squared():
    """∫ acos(x)² dx — elementary (algebraic derivative)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = acos(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "acos(x)^2")


def test_asinh_squared():
    """∫ asinh(x)² dx — elementary (algebraic derivative)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = asinh(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "asinh(x)^2")


def test_acosh_squared():
    """∫ acosh(x)² dx — elementary (algebraic derivative)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = acosh(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "acosh(x)^2")


def test_x_times_asin_squared():
    """∫ x·asin(x)² dx — elementary (polynomial factor, algebraic derivative)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * asin(x) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*asin(x)^2")


def test_asin_cubed():
    """∫ asin(x)³ dx — elementary (deeper IBP recursion, still algebraic)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = asin(x) ** 3
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "asin(x)^3")


# --- Non-elementary: rational-derivative inverse functions (must decline) ----


def test_atan_squared_declines():
    """∫ atan(x)² dx is NON-elementary — must raise, not return a wrong answer.

    The IBP residual ∫ log(1+x²)/(1+x²) dx is a dilog-type non-elementary
    integral (atan has the rational derivative 1/(1+x²)).
    """
    pool = ExprPool()
    x = pool.symbol("x")
    f = atan(x) ** 2
    with pytest.raises(Exception):
        integrate(f, x)


def test_atanh_squared_declines():
    """∫ atanh(x)² dx is NON-elementary — must raise, not return a wrong answer.

    The residual ∫ log(1−x²)/(1−x²) dx is non-elementary (atanh has the rational
    derivative 1/(1−x²)).
    """
    pool = ExprPool()
    x = pool.symbol("x")
    f = atanh(x) ** 2
    with pytest.raises(Exception):
        integrate(f, x)
