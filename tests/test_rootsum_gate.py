"""The Rothstein-Trager route is gated, and its ``RootSum`` answers are checkable.

``integrate``'s rational-function fallback used to return its result directly:
the one route in the integrator whose answer reached the caller with no
verification at all.  Everything it produces for a denominator with algebraic
residues is a ``RootSum``, and nothing could evaluate one, so those answers were
unchecked *and* uncheckable.

These tests pin both halves of the fix from the Python side, where the users
are.  The Rust side has the unit tests; what is asserted here is the end-to-end
behaviour: the answers still come back, ``eval_expr`` can read them, and their
derivative agrees with the integrand at every in-domain sample point.
"""

import math

import alkahest as ak
import pytest

# Two-sided, irrational, and away from the real poles of the integrands below.
_GRID = [0.03, 0.11, 0.23, 0.37, 0.51, 0.71, 0.93, 1.31, 1.87, 2.71, 3.37, 4.13]
_GRID = _GRID + [-v for v in _GRID]

_REL_TOL = 1e-7


def _value(expr, x, xv):
    """``expr`` at ``x = xv`` as a finite float, or ``None``."""
    try:
        v = float(ak.eval_expr(expr, {x: xv}))
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _agreement(integrand, antiderivative, x):
    """(agreeing, disagreeing) sample counts for ``d/dx F`` against ``f``."""
    dF = ak.diff(antiderivative, x)
    dF = getattr(dF, "value", dF)
    agree, disagree = 0, 0
    for xv in _GRID:
        want = _value(integrand, x, xv)
        if want is None:
            continue  # not an in-domain point of the integrand itself
        got = _value(dF, x, xv)
        if got is None or abs(got - want) / max(abs(want), 1.0) >= _REL_TOL:
            disagree += 1
        else:
            agree += 1
    return agree, disagree


@pytest.mark.parametrize(
    "src",
    [
        "1/(x^5 - x - 1)",  # irreducible quintic: five algebraic residues
        "x^12/(x^9 + x + 1)",  # polynomial part + nine algebraic residues
        "1/(x^3 - 2)",  # the smallest case, one real and two complex residues
        "1/(x^4 + x + 1)",
        "1/((x - 1)^2*(x^3 + x + 1))",  # Hermite reduction, then a RootSum
    ],
)
def test_root_sum_answers_are_returned_and_agree_with_the_integrand(src):
    pool = ak.ExprPool()
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=30000)):
        x = ak.symbol("x")
        f = ak.parse(src, pool)
        F = ak.integrate(f, x)
        F = getattr(F, "value", F)
        assert "RootSum" in str(F), f"expected the RootSum arm for {src}, got {F}"
        agree, disagree = _agreement(f, F, x)
    assert disagree == 0, f"{src}: {disagree} of {agree + disagree} samples disagree"
    assert agree >= 8, f"{src}: only {agree} usable samples"


def test_a_root_sum_is_numerically_evaluable():
    """``eval_expr`` used to raise "expression could not be evaluated" here.

    That is what made the answer unverifiable: the gate's numeric arm runs
    through the same interpreter.
    """
    pool = ak.ExprPool()
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=30000)):
        x = ak.symbol("x")
        f = ak.parse("1/(x^3 - 2)", pool)
        F = getattr(ak.integrate(f, x), "value", None)
        assert "RootSum" in str(F)
        # F itself, not just its derivative: Σ_{a³=2} (a/6)·log(x − a).
        value = float(ak.eval_expr(F, {x: 3.0}))
    assert math.isfinite(value)


def test_a_declined_rational_integral_is_never_certified_non_elementary():
    """Every rational function is elementary.

    Above the degree at which the ``f64`` root sum can still be checked to the
    gate's tolerance, ``integrate`` declines — and the decline must stay
    ``E-INT-001``.  Promoting it to ``E-INT-004`` would be a false certificate,
    which is the failure mode the gate exists to prevent, arriving by another
    door.
    """
    pool = ak.ExprPool()
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=30000)):
        x = ak.symbol("x")
        f = ak.parse("1/(x^15 - x - 1)", pool)
        answer, failure = None, None
        try:
            answer = getattr(ak.integrate(f, x), "value", None)
        except Exception as exc:
            # Broad on purpose: the claim is about the error *code*, whatever
            # exception class carries it.
            failure = exc
        if failure is not None:
            code = getattr(failure, "code", None)
            assert code != "E-INT-004", f"a rational function is elementary; got {failure}"
            return
        # If it is answered, it was gated, so it must survive the same check.
        agree, disagree = _agreement(f, answer, x)
    assert disagree == 0
    assert agree >= 8
