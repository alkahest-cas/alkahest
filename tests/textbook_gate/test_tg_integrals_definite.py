"""Textbook gate — definite integration.

First-course definite integrals with known closed-form values: power rule,
trig, `1/(1+x^2) -> pi/4`, `1/x -> log`, integration by parts, and a couple
of substitution/asin cases — the standard "evaluate this definite integral"
problems of a calculus 1/2 course. Each case computes
`ak.integrate(f, x, lo, hi).value`, which collapses to a constant expression
with no free symbols, and checks that constant numerically against a known
value via `assert_definite_value` (see `tests/textbook_gate/README.md`).
"""

from __future__ import annotations

import math

import alkahest as ak
from _tg_helpers import assert_definite_value
import pytest


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


# --- power rule ---------------------------------------------------------


def test_definite_x_squared_0_to_1(pool, x):
    r = ak.integrate(x**2, x, pool.integer(0), pool.integer(1)).value
    assert_definite_value(r, 1 / 3)


def test_definite_x_cubed_0_to_2(pool, x):
    r = ak.integrate(x**3, x, pool.integer(0), pool.integer(2)).value
    assert_definite_value(r, 4.0)


def test_definite_x_fourth_0_to_1(pool, x):
    r = ak.integrate(x**4, x, pool.integer(0), pool.integer(1)).value
    assert_definite_value(r, 1 / 5)


def test_definite_sqrt_x_0_to_4(pool, x):
    r = ak.integrate(ak.sqrt(x), x, pool.integer(0), pool.integer(4)).value
    assert_definite_value(r, 16 / 3)


def test_definite_one_over_x_squared_1_to_2(pool, x):
    r = ak.integrate(1 / x**2, x, pool.integer(1), pool.integer(2)).value
    assert_definite_value(r, 0.5)


# --- trig -----------------------------------------------------------------


def test_definite_sin_0_to_pi(pool, x):
    r = ak.integrate(ak.sin(x), x, pool.integer(0), pool.float(math.pi)).value
    assert_definite_value(r, 2.0)


def test_definite_cos_0_to_pi_over_2(pool, x):
    r = ak.integrate(ak.cos(x), x, pool.integer(0), pool.float(math.pi / 2)).value
    assert_definite_value(r, 1.0)


def test_definite_cos_2x_0_to_pi_over_4(pool, x):
    r = ak.integrate(ak.cos(2 * x), x, pool.integer(0), pool.float(math.pi / 4)).value
    assert_definite_value(r, 0.5)


# --- rational / inverse-trig antiderivatives -------------------------------


def test_definite_one_over_one_plus_x_squared_0_to_1(pool, x):
    """∫₀¹ 1/(1+x^2) dx = pi/4 (atan)."""
    r = ak.integrate(1 / (1 + x**2), x, pool.integer(0), pool.integer(1)).value
    assert_definite_value(r, math.pi / 4)


def test_definite_one_over_four_plus_x_squared_0_to_2(pool, x):
    """∫₀² 1/(4+x^2) dx = pi/8."""
    r = ak.integrate(1 / (4 + x**2), x, pool.integer(0), pool.integer(2)).value
    assert_definite_value(r, math.pi / 8)


def test_definite_one_over_sqrt_one_minus_x_squared_0_to_half(pool, x):
    """∫₀^(1/2) 1/sqrt(1-x^2) dx = pi/6 (asin)."""
    r = ak.integrate(1 / ak.sqrt(1 - x**2), x, pool.integer(0), pool.rational(1, 2)).value
    assert_definite_value(r, math.pi / 6)


def test_definite_one_over_x_1_to_e(pool, x):
    """∫₁^e 1/x dx = 1."""
    r = ak.integrate(1 / x, x, pool.integer(1), pool.float(math.e)).value
    assert_definite_value(r, 1.0)


def test_definite_one_over_x_plus_one_0_to_1(pool, x):
    """∫₀¹ 1/(x+1) dx = log(2)."""
    r = ak.integrate(1 / (x + 1), x, pool.integer(0), pool.integer(1)).value
    assert_definite_value(r, math.log(2))


def test_definite_x_over_x_squared_plus_one_0_to_1(pool, x):
    """∫₀¹ x/(x^2+1) dx = log(2)/2, u-substitution."""
    r = ak.integrate(x / (x**2 + 1), x, pool.integer(0), pool.integer(1)).value
    assert_definite_value(r, math.log(2) / 2)


# --- exponential / by parts -------------------------------------------------


def test_definite_exp_0_to_1(pool, x):
    r = ak.integrate(ak.exp(x), x, pool.integer(0), pool.integer(1)).value
    assert_definite_value(r, math.e - 1)


def test_definite_x_exp_x_0_to_1(pool, x):
    """∫₀¹ x*exp(x) dx = 1, by parts."""
    r = ak.integrate(x * ak.exp(x), x, pool.integer(0), pool.integer(1)).value
    assert_definite_value(r, 1.0)


def test_definite_x_squared_exp_x_0_to_1(pool, x):
    """∫₀¹ x^2*exp(x) dx = e - 2, by parts twice."""
    r = ak.integrate(x**2 * ak.exp(x), x, pool.integer(0), pool.integer(1)).value
    assert_definite_value(r, math.e - 2)


def test_definite_x_log_x_1_to_e(pool, x):
    """∫₁^e x*log(x) dx = (e^2+1)/4, by parts."""
    r = ak.integrate(x * ak.log(x), x, pool.integer(1), pool.float(math.e)).value
    assert_definite_value(r, (math.e**2 + 1) / 4)
