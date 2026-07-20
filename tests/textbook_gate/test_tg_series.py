"""Textbook gate — series expansion.

First-course Taylor/Maclaurin series: the standard elementary-function
expansions about 0, the binomial series (integer and rational exponent),
the geometric series, and a couple of expansions about a nonzero center.
See `tests/textbook_gate/README.md` for the verification philosophy.

`ak.series(expr, var, point, order)` returns a `Series` (not a
`DerivedResult` — access via `.expr`, not `.value`). Verification uses
`assert_series_matches_reference` from `_tg_helpers.py`, which drops the
trailing `O(...)` remainder and numerically compares the truncated
polynomial against a plain-`math` reference function near the expansion
point (loose tolerances — this is a truncation-error check, not exactness).
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from _tg_helpers import SMALL_POINTS, assert_series_matches_reference


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


# --- standard Maclaurin series (about 0) -------------------------------------


def test_series_exp_at_0(pool, x):
    s = ak.series(ak.exp(x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, math.exp)


def test_series_sin_at_0(pool, x):
    s = ak.series(ak.sin(x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, math.sin)


def test_series_cos_at_0(pool, x):
    s = ak.series(ak.cos(x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, math.cos)


def test_series_sinh_at_0(pool, x):
    s = ak.series(ak.sinh(x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, math.sinh)


def test_series_cosh_at_0(pool, x):
    s = ak.series(ak.cosh(x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, math.cosh)


def test_series_tanh_at_0(pool, x):
    s = ak.series(ak.tanh(x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, math.tanh)


def test_series_atan_at_0(pool, x):
    s = ak.series(ak.atan(x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, math.atan)


def test_series_log_1_plus_x_at_0(pool, x):
    """log(1+x) — converges slowly, so keep to SMALL_POINTS (default)."""
    s = ak.series(ak.log(1 + x), x, pool.integer(0), 10)
    assert_series_matches_reference(s, x, math.log1p)


# --- geometric series ----------------------------------------------------------


def test_series_geometric_1_over_1_minus_x(pool, x):
    """1/(1-x) = 1 + x + x^2 + ... — geometric series, radius of convergence 1."""
    s = ak.series(1 / (1 - x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, lambda v: 1 / (1 - v))


def test_series_geometric_1_over_1_plus_x(pool, x):
    """1/(1+x) = 1 - x + x^2 - ... — alternating geometric series."""
    s = ak.series(1 / (1 + x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, lambda v: 1 / (1 + v))


# --- binomial series -------------------------------------------------------------


def test_series_binomial_integer_power(pool, x):
    """(1+x)^3 — binomial theorem with an integer exponent, exact after order 3."""
    s = ak.series((1 + x) ** 3, x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, lambda v: (1 + v) ** 3)


def test_series_binomial_sqrt(pool, x):
    """sqrt(1+x) = (1+x)^(1/2) — binomial series with a half-integer exponent."""
    s = ak.series(ak.sqrt(1 + x), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, lambda v: math.sqrt(1 + v))


def test_series_binomial_rational_power(pool, x):
    """(1+x)^(1/3) — binomial series with a non-half-integer rational exponent."""
    s = ak.series((1 + x) ** pool.rational(1, 3), x, pool.integer(0), 8)
    assert_series_matches_reference(s, x, lambda v: (1 + v) ** (1 / 3))


# --- expansion about a nonzero center -------------------------------------------


def test_series_exp_about_1(pool, x):
    """exp(x) expanded about x0=1 — series output is an absolute function of
    `x`, not `x - x0`, so evaluation points are near the center (1), not near 0.
    """
    s = ak.series(ak.exp(x), x, pool.integer(1), 8)
    points_near_1 = tuple(1.0 + p for p in SMALL_POINTS)
    assert_series_matches_reference(s, x, math.exp, points=points_near_1)


def test_series_sin_about_pi_over_4(pool, x):
    """sin(x) expanded about x0=pi/4 (no symbolic pi constant available, so the
    center is passed as a float via `pool.float`).
    """
    center = math.pi / 4
    s = ak.series(ak.sin(x), x, pool.float(center), 8)
    points_near_center = tuple(center + p for p in SMALL_POINTS)
    assert_series_matches_reference(s, x, math.sin, points=points_near_center)
