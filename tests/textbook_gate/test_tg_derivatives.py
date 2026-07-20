"""Textbook gate — differentiation.

First-course differentiation rules: power/product/quotient/chain rule and the
standard function derivatives. See ``tests/textbook_gate/README.md`` for the
verification philosophy (numeric checks against hand-computed references,
never string-matching alkahest's normal form).
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from _tg_helpers import (
    POSITIVE_POINTS,
    UNIT_INTERVAL_POINTS,
    assert_derivative_matches,
    assert_matches_reference,
)


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


# --- power rule -------------------------------------------------------------


def test_power_rule_x_cubed(x):
    assert_derivative_matches(x**3, x, lambda v: 3 * v**2)


def test_power_rule_x_to_10(x):
    assert_derivative_matches(x**10, x, lambda v: 10 * v**9)


def test_power_rule_negative_exponent(x):
    assert_derivative_matches(x ** (-2), x, lambda v: -2 * v ** (-3))


def test_power_rule_constant(x, pool):
    assert_derivative_matches(pool.integer(7), x, lambda v: 0.0)


def test_linearity_polynomial(x):
    f = 3 * x**3 - 2 * x**2 + 5 * x - 7
    assert_derivative_matches(f, x, lambda v: 9 * v**2 - 4 * v + 5)


# --- product / quotient / chain rule ----------------------------------------


def test_product_rule_poly_times_exp(x):
    f = x**2 * ak.exp(x)
    assert_derivative_matches(f, x, lambda v: (2 * v + v**2) * math.exp(v))


def test_product_rule_sin_times_cos(x):
    f = ak.sin(x) * ak.cos(x)
    assert_derivative_matches(f, x, lambda v: math.cos(v) ** 2 - math.sin(v) ** 2)


def test_quotient_rule_x_over_x_plus_1(x):
    f = x / (x + 1)
    assert_derivative_matches(f, x, lambda v: 1 / (v + 1) ** 2)


def test_quotient_rule_sin_over_x(x):
    f = ak.sin(x) / x
    assert_derivative_matches(f, x, lambda v: (v * math.cos(v) - math.sin(v)) / v**2)


def test_chain_rule_sin_of_square(x):
    f = ak.sin(x**2)
    assert_derivative_matches(f, x, lambda v: 2 * v * math.cos(v**2))


def test_chain_rule_exp_of_negative_square(x):
    f = ak.exp(-(x**2))
    assert_derivative_matches(f, x, lambda v: -2 * v * math.exp(-(v**2)))


def test_chain_rule_nested_trig(x):
    f = ak.sin(ak.cos(x))
    assert_derivative_matches(f, x, lambda v: -math.sin(v) * math.cos(math.cos(v)))


def test_chain_rule_sqrt_of_poly(x):
    f = ak.sqrt(x**2 + 1)
    assert_derivative_matches(f, x, lambda v: v / math.sqrt(v**2 + 1))


# --- standard function derivatives ------------------------------------------


def test_d_sin(x):
    assert_derivative_matches(ak.sin(x), x, math.cos)


def test_d_cos(x):
    assert_derivative_matches(ak.cos(x), x, lambda v: -math.sin(v))


def test_d_tan(x):
    assert_derivative_matches(ak.tan(x), x, lambda v: 1 / math.cos(v) ** 2)


def test_d_exp(x):
    assert_derivative_matches(ak.exp(x), x, math.exp)


def test_d_log(x):
    assert_derivative_matches(ak.log(x), x, lambda v: 1 / v, points=POSITIVE_POINTS)


def test_d_sqrt(x):
    assert_derivative_matches(
        ak.sqrt(x), x, lambda v: 1 / (2 * math.sqrt(v)), points=POSITIVE_POINTS
    )


def test_d_sinh(x):
    assert_derivative_matches(ak.sinh(x), x, math.cosh)


def test_d_cosh(x):
    assert_derivative_matches(ak.cosh(x), x, math.sinh)


def test_d_tanh(x):
    assert_derivative_matches(ak.tanh(x), x, lambda v: 1 / math.cosh(v) ** 2)


def test_d_asin(x):
    assert_derivative_matches(
        ak.asin(x), x, lambda v: 1 / math.sqrt(1 - v**2), points=UNIT_INTERVAL_POINTS
    )


def test_d_atan(x):
    assert_derivative_matches(ak.atan(x), x, lambda v: 1 / (1 + v**2))


def test_d_atan2_wrt_first_arg(x, pool):
    y0 = pool.integer(1)
    # atan2(x, 1): d/dx = 1/(x^2+1)
    assert_derivative_matches(ak.atan2(x, y0), x, lambda v: 1 / (v**2 + 1))


# --- second derivative / repeated differentiation ---------------------------


def test_second_derivative_of_sin(x):
    d1 = ak.diff(ak.sin(x), x).value
    d2 = ak.diff(d1, x).value
    assert_matches_reference(d2, x, lambda v: -math.sin(v))


def test_derivative_of_constant_wrt_unrelated_var(x, pool):
    y = pool.symbol("y")
    r = ak.diff(y**2 + 3, x)
    assert r.value == pool.integer(0)
