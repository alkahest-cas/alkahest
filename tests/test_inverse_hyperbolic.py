"""
Inverse-hyperbolic functions: asinh / acosh / atanh.

Covers parsing, the public builders, differentiation, and integration by parts
(∫ f dx for f ∈ {asinh, acosh, atanh}), verified by symbolic differentiation:
d/dx F == f at several in-domain sample points.

Run after `maturin develop`:
    pytest tests/test_inverse_hyperbolic.py -v
"""

import math

import alkahest
from alkahest import (
    ExprPool,
    acosh,
    asinh,
    atanh,
    diff,
    eval_expr,
    integrate,
    parse,
)

# asinh is finite on all of ℝ; acosh needs x > 1; atanh needs |x| < 1.  The skip
# logic in check_antiderivative drops out-of-domain points, so a single spread of
# points exercises each function where it is real-valued.
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


# ---------------------------------------------------------------------------
# Parsing and builders
# ---------------------------------------------------------------------------


def test_parse_builds_func_nodes():
    """ak.parse produces the same node as the corresponding builder."""
    pool = ExprPool()
    x = pool.symbol("x")
    assert parse("asinh(x)", pool, {"x": x}) == asinh(x)
    assert parse("acosh(x)", pool, {"x": x}) == acosh(x)
    assert parse("atanh(x)", pool, {"x": x}) == atanh(x)


def test_builders_in_all():
    """The three builders are exported and appear in __all__ (append-only freeze)."""
    for name in ("asinh", "acosh", "atanh"):
        assert name in alkahest.__all__, f"{name} missing from __all__"
        assert hasattr(alkahest, name), f"alkahest.{name} builder missing"


# ---------------------------------------------------------------------------
# Differentiation
# ---------------------------------------------------------------------------


def test_diff_asinh():
    """d/dx asinh(x) = 1/√(x²+1)."""
    pool = ExprPool()
    x = pool.symbol("x")
    d = diff(asinh(x), x).value
    for pt in (0.3, 1.4, -2.1):
        expected = 1.0 / math.sqrt(pt * pt + 1.0)
        assert abs(eval_expr(d, {x: pt}) - expected) < 1e-9


def test_diff_acosh():
    """d/dx acosh(x) = 1/√(x²−1)."""
    pool = ExprPool()
    x = pool.symbol("x")
    d = diff(acosh(x), x).value
    for pt in (1.4, 2.7, 3.9):
        expected = 1.0 / math.sqrt(pt * pt - 1.0)
        assert abs(eval_expr(d, {x: pt}) - expected) < 1e-9


def test_diff_atanh():
    """d/dx atanh(x) = 1/(1−x²)."""
    pool = ExprPool()
    x = pool.symbol("x")
    d = diff(atanh(x), x).value
    for pt in (0.2, 0.55, -0.83):
        expected = 1.0 / (1.0 - pt * pt)
        assert abs(eval_expr(d, {x: pt}) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Integration (bare cases close via IBP)
# ---------------------------------------------------------------------------


def test_integrate_asinh():
    """∫ asinh(x) dx = x·asinh(x) − √(x²+1)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = asinh(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "asinh(x)")


def test_integrate_acosh():
    """∫ acosh(x) dx = x·acosh(x) − √(x²−1)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = acosh(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "acosh(x)")


def test_integrate_atanh():
    """∫ atanh(x) dx = x·atanh(x) + ½·log(1−x²)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = atanh(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "atanh(x)")


def test_integrate_x_times_asinh():
    """∫ x·asinh(x) dx closes via IBP (residual ∫ x²/√(x²+1))."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * asinh(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*asinh(x)")


def test_integrate_x_times_atanh():
    """∫ x·atanh(x) dx closes via IBP (residual ∫ x²/(1−x²))."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * atanh(x)
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "x*atanh(x)")
