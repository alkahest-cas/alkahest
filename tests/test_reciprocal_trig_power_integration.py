"""
Reciprocal powers of trig functions: ∫ secⁿ / ∫ cscⁿ.

Because `sec`/`csc` desugar to reciprocals at parse time, these integrands
arrive as negative integer powers of `cos`/`sin`. This covers:

  - ∫ sec²(x) = tan(x), ∫ csc²(x) = −cot(x)  — both the nested `sec(x)**2`
    spelling (which parses to `(cos(x)^(-1))^2`) and the flattened
    `1/cos(x)**2` = `cos(x)^(-2)` spelling.
  - ∫ sec(x) = log((1+sin x)/cos x), ∫ csc(x) = log((1−cos x)/sin x).
  - ∫ sec(x)³, ∫ csc(x)³, ∫ sec(x)⁴ via the reduction formula.

Each result is verified by symbolic differentiation: d/dx F == f at several
sample points. A shape outside the supported subset (e.g. a power above the
reduction cap) must decline (raise IntegrationError), never fabricate an answer.

Run after `maturin develop`:
    pytest tests/test_reciprocal_trig_power_integration.py -v
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


def test_sec_squared_nested():
    """∫ sec(x)² dx = tan(x) — nested (cos(x)^(-1))^2 spelling."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = (cos(x) ** -1) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sec(x)^2")


def test_sec_squared_flattened():
    """∫ 1/cos(x)² dx = tan(x) — flattened cos(x)^(-2) spelling."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(x) ** -2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/cos(x)^2")


def test_csc_squared_nested():
    """∫ csc(x)² dx = −cot(x) — nested (sin(x)^(-1))^2 spelling."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = (sin(x) ** -1) ** 2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "csc(x)^2")


def test_csc_squared_flattened():
    """∫ 1/sin(x)² dx = −cot(x) — flattened sin(x)^(-2) spelling."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) ** -2
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/sin(x)^2")


def test_sec():
    """∫ sec(x) dx = log((1+sin x)/cos x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(x) ** -1
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/cos(x)")


def test_csc():
    """∫ csc(x) dx = log((1−cos x)/sin x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) ** -1
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "1/sin(x)")


def test_sec_cubed():
    """∫ sec(x)³ dx via the reduction formula."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(x) ** -3
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sec(x)^3")


def test_csc_cubed():
    """∫ csc(x)³ dx via the reduction formula."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = sin(x) ** -3
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "csc(x)^3")


def test_sec_quartic():
    """∫ sec(x)⁴ dx (even power, recurses to the tan base case)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(x) ** -4
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sec(x)^4")


def test_sec_linear_arg():
    """∫ sec(2x+1) dx — the chain-rule 1/a factor must be applied."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = cos(2 * x + 1) ** -1
    r = integrate(f, x)
    check_antiderivative(x, f, r.value, "sec(2x+1)")


def test_power_above_cap_declines():
    """∫ 1/cos(x)¹⁰ is above the reduction cap (n ≤ 8) — must decline cleanly."""
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(IntegrationError):
        integrate(cos(x) ** -10, x)
