"""Textbook gate — indefinite integration.

First-course antidifferentiation: power rule, the standard elementary
antiderivatives, simple u-substitution, integration by parts, and partial
fractions with distinct real linear factors — the material of a calculus 1/2
course. Every case is checked via ``assert_integral_self_consistent``: it
differentiates alkahest's returned antiderivative and compares that
numerically to the original integrand, so it is immune to +C, `log(x-2)` vs
`-log(2-x)`, or any other cosmetic difference in the printed form (see
``tests/textbook_gate/README.md``).

Also includes a regression case for bug B2 (report7-20.md):
`exp(x)*log(x) + exp(x)/x` has elementary antiderivative `exp(x)*log(x)`.
And a handful of "correct refusal" checks confirming genuinely non-elementary
integrands (Gaussian integral, sine integral, exponential integral) still
correctly raise ``IntegrationError`` — those are good behavior, not bugs.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from _tg_helpers import POSITIVE_POINTS, UNIT_INTERVAL_POINTS, assert_integral_self_consistent


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


# --- power rule ---------------------------------------------------------


def test_int_x_cubed(x):
    assert_integral_self_consistent(x**3, x)


def test_int_x_fourth(x):
    assert_integral_self_consistent(x**4, x)


def test_int_x_negative_exponent(x):
    assert_integral_self_consistent(x ** (-2), x)


def test_int_x_fractional_exponent(x, pool):
    assert_integral_self_consistent(x ** pool.rational(1, 2), x, points=POSITIVE_POINTS)


# --- standard elementary antiderivatives ---------------------------------


def test_int_one_over_x(x):
    """1/x -> log(x)."""
    assert_integral_self_consistent(1 / x, x, points=POSITIVE_POINTS)


def test_int_exp(x):
    assert_integral_self_consistent(ak.exp(x), x)


def test_int_sin(x):
    assert_integral_self_consistent(ak.sin(x), x)


def test_int_cos(x):
    assert_integral_self_consistent(ak.cos(x), x)


def test_int_sec_squared(x):
    """1/cos(x)^2 -> tan(x)."""
    assert_integral_self_consistent(1 / ak.cos(x) ** 2, x)


def test_int_one_over_one_plus_x_squared(x):
    """1/(1+x^2) -> atan(x)."""
    assert_integral_self_consistent(1 / (1 + x**2), x)


def test_int_one_over_four_plus_x_squared(x):
    """1/(4+x^2) -> atan(x/2)/2, a scaled atan form."""
    assert_integral_self_consistent(1 / (4 + x**2), x)


def test_int_one_over_sqrt_one_minus_x_squared(x):
    """1/sqrt(1-x^2) -> asin(x)."""
    assert_integral_self_consistent(1 / ak.sqrt(1 - x**2), x, points=UNIT_INTERVAL_POINTS)


# --- simple substitution --------------------------------------------------


def test_int_x_exp_x_squared(x):
    """u = x^2 substitution: x*exp(x^2) -> exp(x^2)/2."""
    assert_integral_self_consistent(x * ak.exp(x**2), x)


def test_int_sin_times_cos(x):
    """u = sin(x) substitution: sin(x)*cos(x) -> sin(x)^2/2."""
    assert_integral_self_consistent(ak.sin(x) * ak.cos(x), x)


def test_int_x_over_x_squared_plus_one(x):
    """u = x^2+1 substitution: x/(x^2+1) -> log(x^2+1)/2."""
    assert_integral_self_consistent(x / (x**2 + 1), x)


# --- trigonometric substitution --------------------------------------------


def test_int_x_over_sqrt_x_squared_plus_one(x):
    """x/sqrt(x^2+1) -> sqrt(x^2+1), a Pythagorean-form substitution."""
    assert_integral_self_consistent(x / ak.sqrt(x**2 + 1), x)


def test_int_sqrt_four_minus_x_squared(x):
    """sqrt(4-x^2): classic x = 2 sin(theta) trig substitution."""
    assert_integral_self_consistent(ak.sqrt(4 - x**2), x, points=(-1.5, -0.5, 0.5, 1.0, 1.5))


def test_int_one_over_x_squared_sqrt_x_squared_plus_one(x):
    """1/(x^2*sqrt(x^2+1)): x = tan(theta) trig substitution."""
    assert_integral_self_consistent(1 / (x**2 * ak.sqrt(x**2 + 1)), x, points=(0.3, 0.7, 1.3, 1.9))


def test_int_one_over_sqrt_x_squared_minus_one(x):
    """1/sqrt(x^2-1): x = sec(theta) trig substitution."""
    assert_integral_self_consistent(1 / ak.sqrt(x**2 - 1), x, points=(1.3, 1.9, 2.4, 3.0))


# --- integration by parts -------------------------------------------------


def test_int_x_exp_x(x):
    assert_integral_self_consistent(x * ak.exp(x), x)


def test_int_x_sin_x(x):
    assert_integral_self_consistent(x * ak.sin(x), x)


def test_int_x_log_x(x):
    assert_integral_self_consistent(x * ak.log(x), x, points=POSITIVE_POINTS)


def test_int_log_x(x):
    assert_integral_self_consistent(ak.log(x), x, points=POSITIVE_POINTS)


def test_int_x_squared_cos_x(x):
    """x^2*cos(x): requires two successive rounds of integration by parts."""
    assert_integral_self_consistent(x**2 * ak.cos(x), x)


def test_int_log_x_squared(x):
    """log(x)^2: integration by parts with u = log(x)^2."""
    assert_integral_self_consistent(ak.log(x) ** 2, x, points=POSITIVE_POINTS)


# --- partial fractions (distinct real linear factors) ---------------------


def test_int_partial_fractions_two_factors(x, pool):
    """1/((x-1)(x-2))."""
    assert_integral_self_consistent(1 / ((x - pool.integer(1)) * (x - pool.integer(2))), x)


def test_int_partial_fractions_three_factors(x, pool):
    """1/((x-1)(x-2)(x-3))."""
    denom = (x - pool.integer(1)) * (x - pool.integer(2)) * (x - pool.integer(3))
    assert_integral_self_consistent(1 / denom, x)


def test_int_partial_fractions_four_factors(x, pool):
    """1/((x-1)(x-2)(x-3)(x-4))."""
    denom = (
        (x - pool.integer(1))
        * (x - pool.integer(2))
        * (x - pool.integer(3))
        * (x - pool.integer(4))
    )
    assert_integral_self_consistent(1 / denom, x)


def test_int_x_over_x_squared_minus_one(x):
    """x/(x^2-1), rational function with a nontrivial numerator."""
    assert_integral_self_consistent(x / (x**2 - 1), x)


def test_int_partial_fractions_repeated_linear_factor(x, pool):
    """1/(x-1)^2 — a repeated linear factor (power rule after a shift, but
    exercises the same repeated-root partial-fraction path).
    """
    assert_integral_self_consistent(1 / (x - pool.integer(1)) ** 2, x, points=(-0.5, 0.3, 1.3, 2.4))


def test_int_partial_fractions_repeated_linear_times_distinct(x, pool):
    """1/((x-1)^2(x+1)): a repeated linear factor combined with a distinct one."""
    denom = (x - pool.integer(1)) ** 2 * (x + pool.integer(1))
    assert_integral_self_consistent(1 / denom, x, points=(-0.5, 0.3, 1.3, 2.4))


# --- tan(x) -----------------------------------------------------------------


def test_int_tan(x):
    """tan(x) -> -log(cos(x))."""
    assert_integral_self_consistent(ak.tan(x), x)


# --- correct-refusal checks: genuinely non-elementary integrands ----------


#: The output basis an antiderivative may name.  An answer carrying one of
#: these is, by construction, not an elementary function.
_NONELEMENTARY_NAMES = ("Ei", "li", "Si", "Ci", "Shi", "Chi", "erf", "fresnels", "fresnelc")


def _assert_non_elementary_closed_form(integrand, x, name):
    """The answer must be right *and* must not be elementary.

    These three used to assert a refusal.  They are now answered over the
    registered special-function basis, and the textbook claim being gated is
    unchanged — "this integral has no elementary antiderivative" — only now it
    is carried by the name in the answer instead of by the absence of one.
    Returning something elementary here would be the false claim.
    """
    result = ak.integrate(integrand, x)
    shown = str(result.value)
    assert name in shown, f"expected {name}, got {shown}"
    assert any(n in shown for n in _NONELEMENTARY_NAMES), shown
    assert_integral_self_consistent(integrand, x, points=POSITIVE_POINTS)


def test_int_exp_neg_x_squared_is_erf(x):
    """Gaussian integral — non-elementary, and equal to (√π/2)·erf(x)."""
    _assert_non_elementary_closed_form(ak.exp(-(x**2)), x, "erf")


def test_int_sin_x_over_x_is_si(x):
    """Sine integral Si(x) — non-elementary, and equal to Si(x)."""
    _assert_non_elementary_closed_form(ak.sin(x) / x, x, "Si")


def test_int_exp_x_over_x_is_ei(x):
    """Exponential integral Ei(x) — non-elementary, and equal to Ei(x)."""
    _assert_non_elementary_closed_form(ak.exp(x) / x, x, "Ei")


def test_int_exp_x_squared_correctly_non_elementary(x):
    """`exp(x²)` is `erfi`, which is not a registered primitive. Should raise."""
    with pytest.raises(ak.IntegrationError):
        ak.integrate(ak.exp(x**2), x)


# --- B2 regression: mixed exp·log elementary ------------------------------


def test_int_exp_log_plus_exp_over_x_elementary(x):
    """exp(x)*log(x) + exp(x)/x has elementary antiderivative exp(x)*log(x)."""
    assert_integral_self_consistent(
        ak.exp(x) * ak.log(x) + ak.exp(x) / x, x, points=POSITIVE_POINTS
    )
