"""Textbook gate — limits.

First-course limits: 0/0 and ∞/∞ indeterminate forms, removable
singularities, rational-function behavior at infinity, one-sided limits,
and classic indeterminate products (`x·log(x)` as `x → 0+`). See
`tests/textbook_gate/README.md` for the verification philosophy.

`ak.limit(expr, var, point, dir=None)` returns a bare `Expr` (not a
`DerivedResult` — no `.value`). Finite results are evaluated numerically
with `ak.eval_expr` and compared to a hand-computed reference. Infinite
results print as `"∞"` / `"(-1 * ∞)"` and cannot be passed to `eval_expr`
(it raises `ValueError`), so those are checked via `str()` instead — this
is fine per the README's philosophy since "is this the infinity marker"
is a behavioral question, not a normal-form one.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


def assert_limit_equals(
    result: ak.Expr, expected: float, *, rtol: float = 1e-9, atol: float = 1e-9
) -> None:
    """Assert a finite `ak.limit` result numerically equals `expected`."""
    got = ak.eval_expr(result, {})
    assert math.isclose(got, expected, rel_tol=rtol, abs_tol=atol), (
        f"alkahest={got!r} expected={expected!r}"
    )


def assert_limit_is_pos_infinity(result: ak.Expr) -> None:
    assert str(result) == "∞", f"expected positive-infinity marker, got {result!r}"


def assert_limit_is_neg_infinity(result: ak.Expr) -> None:
    # Observed printed form for -infinity is a literal `(-1 * infinity)` product,
    # not a dedicated negative-infinity token.
    assert str(result) == "(-1 * ∞)", f"expected negative-infinity marker, got {result!r}"


# --- 0/0 indeterminate forms -------------------------------------------------


def test_limit_sin_x_over_x(pool, x):
    r = ak.limit(ak.sin(x) / x, x, pool.integer(0))
    assert_limit_equals(r, 1.0)


def test_limit_one_minus_cos_over_x_squared(pool, x):
    r = ak.limit((1 - ak.cos(x)) / x**2, x, pool.integer(0))
    assert_limit_equals(r, 0.5)


def test_limit_exp_minus_1_over_x(pool, x):
    """(e^x - 1)/x -> 1 as x -> 0 — derivative of exp at 0, by definition."""
    r = ak.limit((ak.exp(x) - 1) / x, x, pool.integer(0))
    assert_limit_equals(r, 1.0)


def test_limit_sqrt_rationalization(pool, x):
    """(sqrt(x+1) - 1)/x -> 1/2 as x -> 0 — classic rationalize-the-numerator case."""
    r = ak.limit((ak.sqrt(x + 1) - 1) / x, x, pool.integer(0))
    assert_limit_equals(r, 0.5)


def test_limit_exp_minus_1_minus_x_over_x_squared(pool, x):
    """(e^x - 1 - x)/x^2 -> 1/2 as x -> 0 — second-order L'Hopital, needs two
    successive applications since the first derivative ratio is still 0/0."""
    r = ak.limit((ak.exp(x) - 1 - x) / x**2, x, pool.integer(0))
    assert_limit_equals(r, 0.5)


def test_limit_log_over_x_minus_one_at_one(pool, x):
    """log(x)/(x-1) -> 1 as x -> 1 — 0/0 form at a non-zero point."""
    r = ak.limit(ak.log(x) / (x - 1), x, pool.integer(1))
    assert_limit_equals(r, 1.0)


def test_limit_log_one_plus_x_over_x(pool, x):
    """log(1+x)/x -> 1 as x -> 0 — derivative of log at 1, by definition."""
    r = ak.limit(ak.log(1 + x) / x, x, pool.integer(0))
    assert_limit_equals(r, 1.0)


def test_limit_exp_two_x_minus_1_over_x(pool, x):
    """(e^(2x) - 1)/x -> 2 as x -> 0 — chain-rule-scaled version of the
    standard (e^x-1)/x -> 1 limit."""
    r = ak.limit((ak.exp(2 * x) - 1) / x, x, pool.integer(0))
    assert_limit_equals(r, 2.0)


def test_limit_x_to_the_x_at_zero_plus(pool, x):
    """x^x -> 1 as x -> 0+ — indeterminate 0^0 form."""
    r = ak.limit(x**x, x, pool.integer(0), dir="+")
    assert_limit_equals(r, 1.0)


# --- removable singularities (factor-and-cancel) -----------------------------


def test_limit_removable_singularity_difference_of_squares(pool, x):
    """(x^2-1)/(x-1) -> 2 at x=1 — hole at x=1, cancels to x+1."""
    r = ak.limit((x**2 - 1) / (x - 1), x, pool.integer(1))
    assert_limit_equals(r, 2.0)


def test_limit_removable_singularity_difference_of_squares_at_2(pool, x):
    """(x^2-4)/(x-2) -> 4 at x=2."""
    r = ak.limit((x**2 - 4) / (x - 2), x, pool.integer(2))
    assert_limit_equals(r, 4.0)


def test_limit_removable_singularity_factored_quadratic(pool, x):
    """(x^2+3x+2)/(x+1) -> 1 at x=-1 — cancels to x+2."""
    r = ak.limit((x**2 + 3 * x + 2) / (x + 1), x, pool.integer(-1))
    assert_limit_equals(r, 1.0)


# --- rational functions at infinity ------------------------------------------


def test_limit_one_over_x_at_infinity(pool, x):
    r = ak.limit(1 / x, x, pool.pos_infinity())
    assert_limit_equals(r, 0.0)


def test_limit_polynomial_ratio_same_degree_at_infinity(pool, x):
    """Ratio of leading coefficients when numerator/denominator degrees match."""
    r = ak.limit((3 * x**2 + 2 * x) / (x**2 + 1), x, pool.pos_infinity())
    assert_limit_equals(r, 3.0)


def test_limit_polynomial_ratio_lower_degree_at_infinity(pool, x):
    """Numerator degree < denominator degree -> 0."""
    r = ak.limit((2 * x + 1) / (x**2 + 3), x, pool.pos_infinity())
    assert_limit_equals(r, 0.0)


def test_limit_polynomial_ratio_higher_degree_at_infinity(pool, x):
    """Numerator degree > denominator degree -> +infinity."""
    r = ak.limit((x**3) / (x**2 + 1), x, pool.pos_infinity())
    assert_limit_is_pos_infinity(r)


# --- classic transcendental limits at infinity -------------------------------


def test_limit_e_definition(pool, x):
    """(1+1/x)^x -> e as x -> infinity — the limit definition of e."""
    r = ak.limit((1 + 1 / x) ** x, x, pool.pos_infinity())
    assert_limit_equals(r, math.e)


def test_limit_exp_over_x_at_infinity(pool, x):
    """exp(x)/x -> infinity — exponential dominates any polynomial."""
    r = ak.limit(ak.exp(x) / x, x, pool.pos_infinity())
    assert_limit_is_pos_infinity(r)


def test_limit_log_over_x_at_infinity(pool, x):
    """log(x)/x -> 0 — logarithm is dominated by any positive power."""
    r = ak.limit(ak.log(x) / x, x, pool.pos_infinity())
    assert_limit_equals(r, 0.0)


def test_limit_x_log_x_at_zero_plus(pool, x):
    """x*log(x) -> 0 as x -> 0+ — classic 0*(-infinity) indeterminate form."""
    r = ak.limit(x * ak.log(x), x, pool.integer(0), dir="+")
    assert_limit_equals(r, 0.0)


# --- one-sided limits ---------------------------------------------------------


def test_limit_one_over_x_right(pool, x):
    """1/x -> +infinity as x -> 0+."""
    r = ak.limit(1 / x, x, pool.integer(0), dir="+")
    assert_limit_is_pos_infinity(r)


def test_limit_one_over_x_left(pool, x):
    """1/x -> -infinity as x -> 0-."""
    r = ak.limit(1 / x, x, pool.integer(0), dir="-")
    assert_limit_is_neg_infinity(r)


# --- trivial sanity check ------------------------------------------------------


def test_limit_of_constant(pool, x):
    """The limit of a constant is itself, regardless of the approach point."""
    r = ak.limit(pool.integer(5), x, pool.integer(0))
    assert_limit_equals(r, 5.0)


# --- classic constant-defining limits -------------------------------------------


def test_limit_one_plus_x_to_one_over_x_is_e(pool, x):
    """(1+x)^(1/x) -> e as x -> 0 — the limit definition of Euler's number."""
    r = ak.limit((1 + x) ** (1 / x), x, pool.integer(0))
    assert_limit_equals(r, math.e)
