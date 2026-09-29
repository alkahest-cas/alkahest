"""Integers past ``i64`` in the printers, and exact predicates.

Two silent-wrong-answer classes from the 2026-09-29 core audit:

* **A2** — the LaTeX and Unicode printers read coefficients, rationals and
  exponents through ``to_i64().unwrap_or(0 / 1)`` and multiplied them with a
  wrapping ``i64 *=``, so ``latex(x + 10**20)`` printed ``x + 0``,
  ``10**30 * x`` printed ``x`` and ``-2**63 * x`` printed a double minus.
* **A4** — predicates over exact numbers were decided by rounding both sides
  to ``f64``, so ``subs(Eq(x, 10**30), {x: 10**30 + 1})`` folded to ``True``.
"""

import math

import alkahest as ak
import pytest


@pytest.fixture
def pool():
    return ak.ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


def _all_printers(e):
    return {"str": str(e), "latex": ak.latex(e), "unicode": ak.unicode_str(e)}


# ---------------------------------------------------------------------------
# A2: printers
# ---------------------------------------------------------------------------


def test_a_big_integer_term_is_printed(pool, x):
    e = x + pool.integer(10**20)
    for name, out in _all_printers(e).items():
        assert str(10**20) in out, (name, out)


def test_a_big_integer_coefficient_is_printed(pool, x):
    e = pool.integer(10**30) * x
    for name, out in _all_printers(e).items():
        assert str(10**30) in out, (name, out)


def test_i64_min_coefficient_has_a_single_minus(pool, x):
    e = pool.integer(-(2**63)) * x
    for name, out in _all_printers(e).items():
        assert str(2**63) in out, (name, out)
        assert "--" not in out, (name, out)
    assert ak.unicode_str(e) == "-9223372036854775808·x"


def test_a_big_rational_is_printed(pool):
    f21 = math.factorial(21)
    r = pool.rational(1, f21)
    assert ak.unicode_str(r) == f"1/{f21}"
    assert ak.latex(r) == rf"\frac{{1}}{{{f21}}}"
    r = pool.rational(10**30, 7)
    assert ak.unicode_str(r) == f"{10**30}/7"


def test_division_by_a_big_power_of_two_is_printed(pool, x):
    e = pool.rational(1, 2**64) * x
    for name, out in _all_printers(e).items():
        assert str(2**64) in out, (name, out)


def test_a_radical_index_past_i64_is_printed(pool, x):
    e = x ** pool.rational(1, 10**20)
    assert ak.latex(e) == rf"\sqrt[{10**20}]{{x}}"
    assert "1/1" not in ak.unicode_str(e)


def test_the_x21_coefficient_of_the_exp_series(pool, x):
    s = ak.series(ak.exp(x), x, pool.integer(0), 23).expr
    f21 = math.factorial(21)
    assert rf"\frac{{1}}{{{f21}}} x^{{21}}" in ak.latex(s)
    assert f"1/{f21}·x²¹" in ak.unicode_str(s)


# ---------------------------------------------------------------------------
# A4: exact predicates
# ---------------------------------------------------------------------------


def _is(pool, e, value):
    return e == (pool.pred_true() if value else pool.pred_false())


def test_eq_between_integers_past_f64(pool, x):
    e = ak.subs(pool.pred_eq(x, pool.integer(10**30)), {x: pool.integer(10**30 + 1)})
    assert _is(pool, e, False), e


def test_lt_between_integers_past_f64(pool, x):
    e = ak.subs(pool.lt(x, pool.integer(10**30 + 1)), {x: pool.integer(10**30)})
    assert _is(pool, e, True), e


def test_ne_between_close_rationals(pool, x):
    approx = pool.rational(3333333333333333, 10**16)
    e = ak.subs(pool.pred_ne(x, pool.rational(1, 3)), {x: approx})
    assert _is(pool, e, True), e


def test_eq_with_a_rational_that_underflows_f64(pool, x):
    e = ak.subs(pool.pred_eq(x, pool.rational(1, 10**400)), {x: pool.integer(0)})
    assert _is(pool, e, False), e


def test_piecewise_at_two_to_the_53_plus_one(pool, x):
    pw = ak.piecewise([(pool.gt(x, pool.integer(2**53)), pool.integer(1))], pool.integer(0))
    e = ak.subs(pw, {x: pool.integer(2**53 + 1)})
    assert e == pool.integer(1), e


def test_float_against_rational_is_exact(pool, x):
    # The double nearest 1/3 lies strictly below it.
    e = ak.subs(pool.lt(x, pool.rational(1, 3)), {x: pool.float(1 / 3)})
    assert _is(pool, e, True), e


def test_an_undecidable_comparison_is_left_alone(pool, x):
    two = pool.integer(2)
    # (√2)² = 2 exactly, but no finite enclosure proves it: stay unevaluated.
    sq = (two ** pool.rational(1, 2)) ** two
    e = ak.subs(pool.pred_eq(x, two), {x: sq})
    assert not _is(pool, e, True), e
    assert not _is(pool, e, False), e
