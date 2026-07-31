"""Silent-error discipline for numeric evaluation.

``eval_expr`` used to hand back whatever IEEE-754 arithmetic produced for a
substitution, including ``nan``/``inf`` for expressions that are undefined at
the given point — e.g. ``(x**2 - 1) / (x - 1)`` evaluated *as written* at
``x = 1`` is ``0 * (1/0) = nan``, silently returned as a float rather than
raised as an error. An agent that doesn't separately check
``math.isfinite(result)`` would never notice.

These tests lock in the fix: evaluating an expression whose denominator is
zero at the binding point raises ``DomainError`` (``E-EVAL-009``) instead of
returning a number. An explicit ``cancel()`` first is still a legitimate way
to get the removable-singularity limit.
"""

from __future__ import annotations

import alkahest as ak
import pytest


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


def test_removable_singularity_evaluated_as_written_raises(pool, x):
    """f(x) = (x**2-1)/(x-1) is undefined at x=1; evaluating it AS WRITTEN
    must raise, not silently return the limit 2."""
    f = (x**2 - 1) / (x - 1)
    with pytest.raises(ak.DomainError) as excinfo:
        ak.eval_expr(f, {x: 1})
    assert excinfo.value.code == "E-EVAL-009"


def test_removable_singularity_raises_as_value_error_too(pool, x):
    """DomainError must still be catchable as ValueError (backward compat:
    alkahest's structured errors all derive from ValueError)."""
    f = (x**2 - 1) / (x - 1)
    with pytest.raises(ValueError):
        ak.eval_expr(f, {x: 1})


def test_removable_singularity_after_cancel_gives_the_limit(pool, x):
    """cancel() is an intentional, explicit rewrite — after it, evaluating
    the simplified x+1 at x=1 legitimately gives 2."""
    f = (x**2 - 1) / (x - 1)
    cancelled = ak.cancel(f)
    assert ak.eval_expr(cancelled, {x: 1}) == 2.0


def test_removable_singularity_away_from_pole_still_works(pool, x):
    """Away from x=1, f(x) = (x**2-1)/(x-1) evaluates fine (no pole)."""
    f = (x**2 - 1) / (x - 1)
    assert ak.eval_expr(f, {x: 3}) == pytest.approx(4.0)
    assert ak.eval_expr(f, {x: -2}) == pytest.approx(-1.0)


def test_plain_division_by_zero_raises(pool, x):
    """1/x at x=0 is a straightforward pole — must also raise, not return inf."""
    f = 1 / x
    with pytest.raises(ak.DomainError):
        ak.eval_expr(f, {x: 0})


def test_unbound_symbol_still_raises_value_error(pool, x):
    """Pre-existing behavior (unrelated to the non-finite-result fix) is
    unchanged: an unbound free symbol still raises."""
    with pytest.raises(ValueError):
        ak.eval_expr(x, {})


def test_evaluate_stable_api_already_reports_nonfinite(pool, x):
    """`alkahest.evaluate` (the structured, non-raising API) already handled
    this correctly before this fix — kept here as a cross-check that both
    entry points agree."""
    f = (x**2 - 1) / (x - 1)
    result = ak.evaluate(f, {x: 1}, mode="f64")
    assert result.status == "unsupported"
    assert result.reason == "E-EVAL-009"
