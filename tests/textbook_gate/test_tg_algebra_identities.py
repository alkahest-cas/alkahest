"""Textbook gate — algebraic identities.

Three first-course topics: log/exp inverse identities, rational-function
`cancel`/`together`, and exponent (power) rules. See
`tests/textbook_gate/README.md` for the verification philosophy (numeric
checks against known references, never string-matching alkahest's normal
form).

`log(exp(x)) -> x` and `exp(log(x)) -> x` are checked numerically via
value-preservation (this suite avoids string-matching normal form). The
combined case `log(exp(x)) + exp(log(y))` uses structural equality against
`simplify(x + y)` so a no-op rewrite cannot slip through as a false pass.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from _tg_helpers import POSITIVE_POINTS, assert_definite_value, assert_matches_reference


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


@pytest.fixture
def y(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("y")


def assert_two_var_matches_reference(
    expr: ak.Expr,
    var_a: ak.Expr,
    var_b: ak.Expr,
    reference,
    points=((0.3, 0.7), (1.1, 2.2), (1.9, 0.4)),
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """Two-variable analogue of `assert_matches_reference`: assert `expr`
    numerically equals `reference(a, b)` at each `(var_a, var_b)` sample pair.
    """
    for a, b in points:
        got = ak.eval_expr(expr, {var_a: a, var_b: b})
        want = reference(a, b)
        assert math.isclose(got, want, rel_tol=rtol, abs_tol=atol), (
            f"at {var_a}={a}, {var_b}={b}: alkahest={got!r} reference={want!r} "
            f"(diff={got - want!r})"
        )


# --- log/exp inverse identities ----------------------------------------------


def test_log_exp_inverse_simplify_log_exp(x):
    """log(exp(x)) -> x for all real x, checked via simplify_log_exp."""
    r = ak.simplify_log_exp(ak.log(ak.exp(x))).value
    assert_matches_reference(r, x, lambda v: v)


def test_log_exp_inverse_simplify(x):
    """log(exp(x)) -> x for all real x, checked via plain simplify."""
    r = ak.simplify(ak.log(ak.exp(x))).value
    assert_matches_reference(r, x, lambda v: v)


def test_exp_log_inverse_simplify_log_exp(x):
    """exp(log(x)) -> x for positive x, checked via simplify_log_exp."""
    r = ak.simplify_log_exp(ak.exp(ak.log(x))).value
    assert_matches_reference(r, x, lambda v: v, points=POSITIVE_POINTS)


def test_exp_log_inverse_simplify(x):
    """exp(log(x)) -> x for positive x, checked via plain simplify."""
    r = ak.simplify(ak.exp(ak.log(x))).value
    assert_matches_reference(r, x, lambda v: v, points=POSITIVE_POINTS)


def test_simplify_log_exp_inverse_pair(pool, x, y):
    """log(exp(x)) + exp(log(y)) should collapse to x + y via the two
    independent inverse identities. This is checked via *structural* Expr
    equality against `simplify(x + y)` (the same technique the exemplar
    `test_tg_derivatives.py::test_derivative_of_constant_wrt_unrelated_var`
    uses via `r.value == pool.integer(0)`) rather than numeric evaluation:
    since `log(exp(x)) + exp(log(y))` already evaluates correctly to `x + y`
    even when left completely unsimplified, a numeric value-preservation
    check cannot detect this no-op at all (it trivially passes either way).
    Structural equality is not the same as string-matching the normal form —
    it's the pool's own canonical-node equality, and it's precisely what's
    needed to notice "nothing was folded" as opposed to "something wrong was
    computed".
    """
    combined = ak.log(ak.exp(x)) + ak.exp(ak.log(y))
    r = ak.simplify_log_exp(combined).value
    target = ak.simplify(x + y).value
    assert r == target


# --- log/exp combining rules (product/power/quotient) -------------------------
#
# Inverse cancellations plus the textbook combining/expansion rules:
# `log(x)+log(y)→log(xy)`, `log(x^n)→n·log(x)`, `log(x/y)→log(x)−log(y)`,
# and `exp(x)·exp(y)→exp(x+y)`.  Structural equality is required here for the
# same reason as `test_simplify_log_exp_inverse_pair`: the unsimplified forms
# already evaluate correctly, so numeric checks cannot detect a no-op.


def test_simplify_log_exp_product_rule_folds(pool, x, y):
    """log(x) + log(y) -> log(x*y) for positive x, y."""
    r = ak.simplify_log_exp(ak.log(x) + ak.log(y)).value
    target = ak.simplify(ak.log(x * y)).value
    assert r == target


def test_simplify_log_exp_power_rule_folds(pool, x):
    """log(x**2) -> 2*log(x) for positive x."""
    r = ak.simplify_log_exp(ak.log(x**2)).value
    target = ak.simplify(2 * ak.log(x)).value
    assert r == target


def test_simplify_log_exp_quotient_rule_folds(pool, x, y):
    """log(x/y) -> log(x) - log(y) for positive x, y."""
    r = ak.simplify_log_exp(ak.log(x / y)).value
    target = ak.simplify(ak.log(x) - ak.log(y)).value
    assert r == target


def test_simplify_log_exp_product_of_exps_folds(pool, x, y):
    """exp(x) * exp(y) -> exp(x+y)."""
    r = ak.simplify_log_exp(ak.exp(x) * ak.exp(y)).value
    target = ak.simplify(ak.exp(x + y)).value
    assert r == target


# --- cancel / together --------------------------------------------------------


def test_cancel_difference_of_squares(x):
    """cancel((x^2-4)/(x-2), [x]) -> x+2, checked away from the removed x=2
    singularity.
    """
    e = (x**2 - 4) / (x - 2)
    r = ak.cancel(e, [x])
    assert_matches_reference(r, x, lambda v: v + 2, points=(0.3, 0.7, 1.3, 3.0))


def test_cancel_difference_of_squares_removes_singularity(x):
    """The whole point of cancel is removing the removable singularity: the
    original (x^2-4)/(x-2) is NaN at x=2 (0/0), but the cancelled form
    evaluates cleanly there to the correct limit, 4.
    """
    e = (x**2 - 4) / (x - 2)
    orig_at_2 = ak.eval_expr(e, {x: 2.0})
    assert math.isnan(orig_at_2)
    r = ak.cancel(e, [x])
    assert math.isclose(ak.eval_expr(r, {x: 2.0}), 4.0, rel_tol=1e-9, abs_tol=1e-9)


def test_cancel_another_difference_of_squares(x):
    """cancel((x^2-1)/(x-1), [x]) -> x+1, and the x=1 singularity is removed."""
    e = (x**2 - 1) / (x - 1)
    assert math.isnan(ak.eval_expr(e, {x: 1.0}))
    r = ak.cancel(e, [x])
    assert_matches_reference(r, x, lambda v: v + 1, points=(0.3, 0.7, 1.3, 3.0))
    assert math.isclose(ak.eval_expr(r, {x: 1.0}), 2.0, rel_tol=1e-9, abs_tol=1e-9)


def test_cancel_sum_denominator(x):
    """cancel((x^2-9)/(x+3), [x]) -> x-3 (factor in the denominator, not a
    removable-at-a-positive-point case, just a plain common-factor cancel).
    """
    e = (x**2 - 9) / (x + 3)
    r = ak.cancel(e, [x])
    assert_matches_reference(r, x, lambda v: v - 3, points=(0.3, 0.7, 1.3))


def test_together_two_reciprocals(x, y):
    """together(1/x + 1/y, [x, y]) numerically equals 1/x + 1/y at sample pairs."""
    e = 1 / x + 1 / y
    r = ak.together(e, [x, y])
    assert_two_var_matches_reference(r, x, y, lambda a, b: 1 / a + 1 / b)


def test_together_two_reciprocals_plus_one(x, y):
    """together(1/x + 1/y + 1, [x, y]) preserves value with a third, integer
    term mixed in.
    """
    e = 1 / x + 1 / y + 1
    r = ak.together(e, [x, y])
    assert_two_var_matches_reference(r, x, y, lambda a, b: 1 / a + 1 / b + 1)


def test_together_reciprocal_difference(x, y):
    """together(1/x - 1/y, [x, y]) preserves value."""
    e = 1 / x - 1 / y
    r = ak.together(e, [x, y])
    assert_two_var_matches_reference(r, x, y, lambda a, b: 1 / a - 1 / b)


def test_cancel_requires_vars_argument(x):
    """cancel/together require an explicit vars list — omitting it raises
    TypeError (documented API quirk from report7-20.md's 'API consistency'
    section, kept here as a green canary so the signature isn't silently
    loosened or tightened further without this suite noticing).
    """
    e = (x**2 - 4) / (x - 2)
    with pytest.raises(TypeError):
        ak.cancel(e)


# --- exponent rules ------------------------------------------------------------


def test_exponent_product_rule(x):
    """x^2 * x^3 -> x^5."""
    r = ak.simplify(x**2 * x**3).value
    assert_matches_reference(r, x, lambda v: v**5)


def test_exponent_power_of_power_rule(x):
    """(x^2)^3 -> x^6."""
    r = ak.simplify((x**2) ** 3).value
    assert_matches_reference(r, x, lambda v: v**6)


def test_exponent_zero(x):
    """x^0 -> 1, fully collapsed to a constant with no free symbols."""
    r = ak.simplify(x**0).value
    assert_definite_value(r, 1.0)


def test_exponent_one(x):
    """x^1 -> x."""
    r = ak.simplify(x**1).value
    assert_matches_reference(r, x, lambda v: v)


def test_negative_exponent_reciprocal(x):
    """1/x^3 -> x^-3."""
    r = ak.simplify(1 / (x**3)).value
    assert_matches_reference(r, x, lambda v: v**-3)


def test_exponent_product_cancels_to_constant(x):
    """x^2 * x^-2 -> 1, fully collapsed to a constant."""
    r = ak.simplify(x**2 * x ** (-2)).value
    assert_definite_value(r, 1.0)


def test_exponent_product_mixed_sign(x):
    """x^-2 * x^5 -> x^3."""
    r = ak.simplify(x ** (-2) * x**5).value
    assert_matches_reference(r, x, lambda v: v**3)


def test_negative_power_of_negative_power(x):
    """(x^-1)^-1 -> x."""
    r = ak.simplify((x ** (-1)) ** (-1)).value
    assert_matches_reference(r, x, lambda v: v)


def test_power_of_power_negative_exponent(x):
    """(x^3)^-2 -> x^-6."""
    r = ak.simplify((x**3) ** (-2)).value
    assert_matches_reference(r, x, lambda v: v**-6)


def test_rational_exponent_square_of_sqrt(pool, x):
    """(x^(1/2))^2 -> x for positive x."""
    r = ak.simplify((x ** pool.rational(1, 2)) ** 2).value
    assert_matches_reference(r, x, lambda v: v, points=POSITIVE_POINTS)


def test_rational_exponent_thirds_sum_to_one(pool, x):
    """x^(1/3) * x^(2/3) -> x for positive x."""
    r = ak.simplify(x ** pool.rational(1, 3) * x ** pool.rational(2, 3)).value
    assert_matches_reference(r, x, lambda v: v, points=POSITIVE_POINTS)
