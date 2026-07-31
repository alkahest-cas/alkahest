"""Textbook gate — silent-error discipline.

A "silent error" here means a *confident, plausible, wrong* answer: alkahest
returns a normal-looking value with no error, verification flag, or warning
distinguishing it from a correct result, on a problem a first-course student
would recognize as a trap (integrating through a pole, evaluating a function
at its own removable singularity, treating a complex root as a real
solution, or inventing a closed-form antiderivative that doesn't exist).

Unlike the rest of the textbook gate — which checks that alkahest gets
first-course problems *right* — this file checks that alkahest *refuses*
the classic traps rather than confidently getting them wrong. See
``tests/textbook_gate/README.md`` for the general philosophy; the difference
here is that "pass" means "raises", not "returns the correct number".

Background: the four cases below correspond to agent-benchmark trap tasks
in ``agent-benchmark/tasks/catalogue.py`` (``pole_interior_inverse_square``,
``removable_singularity_value``, ``solve_x2_plus_1_real``, and the
non-elementary-antiderivative floor case) — situations where a mainstream
CAS (SymPy, alkahest pre-fix) hands back a clean-looking number that is
either divergent, undefined, or not real.
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


def test_definite_integral_through_interior_pole_raises(pool, x):
    """∫_{-1}^{1} 1/x^2 dx diverges (pole at x=0, strictly inside the bounds).

    Naively applying the power rule gives F(x) = -1/x, so F(1) - F(-1) = -2 —
    a clean, plausible, WRONG number with no indication anything is wrong.
    Fixed pre-3.7.0 (see CHANGELOG); this locks the fix in as a regression
    test rather than relying on it only being exercised in agent-benchmark.
    """
    with pytest.raises(ak.IntegrationError) as excinfo:
        ak.integrate(1 / x**2, x, pool.integer(-1), pool.integer(1))
    assert excinfo.value.code == "E-INT-001"


def test_definite_integral_through_interior_pole_rational_raises(pool, x):
    """∫_0^2 dx/(x^2-1) also diverges (pole at x=1, strictly inside [0, 2]) —
    a second interior-pole shape (pole not at an endpoint or the origin)."""
    with pytest.raises(ak.IntegrationError) as excinfo:
        ak.integrate(1 / (x**2 - 1), x, pool.integer(0), pool.integer(2))
    assert excinfo.value.code == "E-INT-001"


def test_removable_singularity_value_raises(pool, x):
    """f(x) = (x^2-1)/(x-1) is undefined at x=1. Evaluating it AS WRITTEN at
    that point must raise, not return the limit 2 (which is only the value
    of the *cancelled* function x+1, a distinct expression)."""
    f = (x**2 - 1) / (x - 1)
    with pytest.raises(ak.DomainError) as excinfo:
        ak.eval_expr(f, {x: 1})
    assert excinfo.value.code == "E-EVAL-009"


def test_removable_singularity_value_ok_after_explicit_cancel(pool, x):
    """Cancel is an intentional rewrite, not a silent one: once the caller
    explicitly asks for it, evaluating the simplified form is legitimate."""
    f = (x**2 - 1) / (x - 1)
    assert ak.eval_expr(ak.cancel(f), {x: 1}) == 2.0


def test_real_solve_of_x_squared_plus_one_has_no_real_roots(pool, x):
    """x^2 = -1 has no real solutions. `domain="real"` must report zero,
    not hand back the complex roots ±i as though they were real answers."""
    sols = ak.solve([x**2 + 1], [x], domain="real")
    assert sols == []


def test_real_solve_of_x_squared_minus_one_has_two_real_roots(pool, x):
    """Sanity check in the other direction: x^2 = 1 genuinely has two real
    roots, and domain="real" must not over-filter and drop them."""
    sols = ak.solve([x**2 - 1], [x], domain="real")
    values = sorted(ak.eval_expr(sol[x], {}) for sol in sols)
    assert values == pytest.approx([-1.0, 1.0])


def test_non_elementary_antiderivative_raises_not_a_fake_formula(pool, x):
    """∫ exp(x^2) dx has no elementary closed form (it's related to erfi).
    alkahest must refuse (E-INT-004 / NonElementary) rather than invent a
    formula or silently return an unevaluated/incorrect expression."""
    with pytest.raises(ak.IntegrationError) as excinfo:
        ak.integrate(ak.exp(x**2), x)
    assert excinfo.value.code == "E-INT-004"
