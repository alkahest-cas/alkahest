"""
Partial-fraction decomposition (``apart``) over ℚ.

``apart(p/q, x)`` rewrites a univariate rational function as

    poly_part + Σ_i Σ_j  A_ij(x) / f_i(x)**j

where the ``f_i`` are the distinct ℚ-irreducible factors of the denominator and
``deg A_ij < deg f_i``.  Irreducible quadratics are kept intact (decomposition
is over ℚ, not ℚ(α)).

Verification is by numeric sampling: the decomposition must agree with the
original rational function at several non-pole points.

Run after building the extension:
    maturin develop --release
    pytest tests/test_apart.py -v
"""

import math

import alkahest
import pytest
from alkahest import ExprPool, apart, eval_expr

_TEST_POINTS = (1.7, 2.3, 3.9, -2.5, 5.1, 7.3)


def _check_equiv(f, pf, x, points=_TEST_POINTS):
    """Assert ``apart`` result equals the input numerically away from poles."""
    for pt in points:
        lhs = eval_expr(f, {x: pt})
        rhs = eval_expr(pf, {x: pt})
        assert math.isfinite(lhs), f"non-finite input at {pt}"
        assert math.isfinite(rhs), f"non-finite apart result at {pt}"
        assert abs(lhs - rhs) < 1e-7, f"apart mismatch at x={pt}: {lhs} vs {rhs}\n  pf = {pf}"


def test_one_over_x2_minus_1():
    # 1/(x²−1) = 1/(2(x−1)) − 1/(2(x+1)).
    pool = ExprPool()
    x = pool.symbol("x")
    f = 1 / (x**2 - 1)
    pf = apart(f, x)
    _check_equiv(f, pf, x)


def test_improper_x3_over_x2_minus_1():
    # x³/(x²−1) = x + 1/(2(x−1)) + 1/(2(x+1)).
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**3 / (x**2 - 1)
    pf = apart(f, x)
    _check_equiv(f, pf, x)
    # Must carry a polynomial part.
    assert "x" in str(pf)


def test_repeated_and_simple_factor():
    # (x+1)/((x−1)²(x+2)).
    pool = ExprPool()
    x = pool.symbol("x")
    f = (x + 1) / ((x - 1) ** 2 * (x + 2))
    pf = apart(f, x)
    _check_equiv(f, pf, x)


def test_high_multiplicity():
    # x/(x−1)³ = 1/(x−1)² + 1/(x−1)³.
    pool = ExprPool()
    x = pool.symbol("x")
    f = x / (x - 1) ** 3
    pf = apart(f, x)
    _check_equiv(f, pf, x)


def test_irreducible_quadratic_kept():
    # 1/((x−1)(x²+1)) — the quadratic stays intact over ℚ (no sqrt / i).
    pool = ExprPool()
    x = pool.symbol("x")
    f = 1 / ((x - 1) * (x**2 + 1))
    pf = apart(f, x)
    _check_equiv(f, pf, x)
    assert "sqrt" not in str(pf), f"quadratic should not be split: {pf}"


def test_already_polynomial():
    # A polynomial has no proper part; apart returns it unchanged numerically.
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**2 + 1
    pf = apart(f, x)
    _check_equiv(f, pf, x)


def test_recombination_via_diff_consistency():
    # apart preserves the function, so differentiating both forms agrees.
    pool = ExprPool()
    x = pool.symbol("x")
    f = (2 * x + 1) / ((x + 1) ** 2 * (x + 2))
    pf = apart(f, x)
    df = alkahest.diff(f, x).value
    dpf = alkahest.diff(pf, x).value
    for pt in (0.5, 1.3, 3.7, -3.5):
        assert abs(eval_expr(df, {x: pt}) - eval_expr(dpf, {x: pt})) < 1e-6


def test_not_rational_raises():
    # exp(x)/(x²−1) is not a rational function of x.
    pool = ExprPool()
    x = pool.symbol("x")
    f = alkahest.exp(x) / (x**2 - 1)
    with pytest.raises(ValueError):
        apart(f, x)
