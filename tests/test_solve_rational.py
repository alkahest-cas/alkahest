"""``solve`` on rational-function equations — the form nodal analysis produces.

Two things are being tested and they pull in opposite directions.  Accepting
``(Vo - Vin)/R1 + Vo*s*C`` at all is the feature: every admittance term in
circuit analysis is a reciprocal, and requiring the caller to clear denominators
by hand is what blocked symbolic nodal/MNA analysis.  But clearing a denominator
is not an equivalence — ``N/D = 0`` means ``N = 0 and D != 0`` — so the roots it
invents have to come back out again, and a root returned at a pole would be a
wrong answer rather than a rough edge.
"""

from __future__ import annotations

import alkahest as ak
import pytest

pytest.importorskip("alkahest", reason="native extension required")


@pytest.fixture
def pool():
    return ak.ExprPool()


def _values(sols, var):
    """The returned roots for *var*, as floats."""
    return sorted(float(ak.eval_expr(s[var], {})) for s in sols)


# ---------------------------------------------------------------------------
# The gap this closes
# ---------------------------------------------------------------------------


def test_rc_divider_in_admittance_form(pool):
    """``(Vo - Vin)/R1 + Vo*s*C = 0`` → ``Vo = Vin/(1 + s*R1*C)``.

    This raised ``E-SOLVE-001`` ("negative exponent -1 in polynomial") until
    rational equations were accepted, and the workaround was to multiply through
    by ``R1`` by hand.
    """
    Vo, Vin, R1, C, s = (pool.symbol(n) for n in ("Vo", "Vin", "R1", "C", "s"))
    sols = ak.solve([(Vo - Vin) / R1 + Vo * s * C], [Vo])
    assert len(sols) == 1
    env = {Vin: 1.0, s: 7.0, R1: 2.0, C: 5.0}
    assert float(ak.eval_expr(sols[0][Vo], env)) == pytest.approx(1.0 / 71.0)


def test_rc_divider_agrees_with_the_hand_cleared_form(pool):
    """Clearing ``1/R1`` by hand and letting the solver do it give the same answer."""
    Vo, Vin, R1, C, s = (pool.symbol(n) for n in ("Vo", "Vin", "R1", "C", "s"))
    env = {Vin: 3.0, s: 2.0, R1: 11.0, C: 0.5}
    rational = ak.solve([(Vo - Vin) / R1 + Vo * s * C], [Vo])
    cleared = ak.solve([(Vo - Vin) + Vo * s * C * R1], [Vo])
    assert len(rational) == len(cleared) == 1
    assert float(ak.eval_expr(rational[0][Vo], env)) == pytest.approx(
        float(ak.eval_expr(cleared[0][Vo], env))
    )


def test_two_node_mna_system(pool):
    """Two rational equations, two unknowns, five symbolic parameters."""
    V1, V2, Vin = (pool.symbol(n) for n in ("V1", "V2", "Vin"))
    R1, R2, C, s = (pool.symbol(n) for n in ("R1", "R2", "C", "s"))
    node1 = (V1 - Vin) / R1 + (V1 - V2) / R2
    node2 = (V2 - V1) / R2 + V2 * s * C

    sols = ak.solve([node1, node2], [V1, V2])
    assert len(sols) == 1
    # V2 = Vin/(1 + s*C*(R1+R2)) and V1 = V2*(1 + s*C*R2).
    env = {Vin: 1.0, R1: 2.0, R2: 3.0, C: 5.0, s: 7.0}
    assert float(ak.eval_expr(sols[0][V2], env)) == pytest.approx(1.0 / 176.0)
    assert float(ak.eval_expr(sols[0][V1], env)) == pytest.approx(53.0 / 88.0)


# ---------------------------------------------------------------------------
# Spurious roots — the correctness burden
# ---------------------------------------------------------------------------


def test_a_root_at_a_pole_is_not_returned(pool):
    """``x/(x-1) = 1/(x-1)`` clears to ``(x-1)**2 = 0``; ``x = 1`` is not a solution."""
    x = pool.symbol("x")
    assert ak.solve([x / (x - 1) - 1 / (x - 1)], [x]) == []


def test_a_reciprocal_is_never_zero(pool):
    """``1/x = 0`` has no solution — not ``x = 0``, not ``x = inf``."""
    x = pool.symbol("x")
    assert ak.solve([1 / x], [x]) == []


def test_a_removable_singularity_is_not_a_solution(pool):
    """``x**2/x = 0`` is ``0/0`` at ``x = 0``, so it has no solution.

    The case that reducing to lowest terms gets wrong: cancelling gives
    ``x = 0``, a point the expression as written has no value at.
    """
    x = pool.symbol("x")
    assert ak.solve([x * x / x], [x]) == []


def test_a_condition_hidden_by_a_reciprocal_is_not_lost(pool):
    """``1/(1/x - 1) = 0`` has no solution: ``1/x`` is undefined at ``x = 0``.

    A reciprocal swaps numerator and denominator, so the inner denominator
    ``x`` becomes the outer numerator and ``x != 0`` drops out of any
    product-of-denominators bookkeeping. What survives, ``1 - x``, is
    perfectly non-zero at ``x = 0``, so the cleared numerator's only root
    sails straight through unless the *domain* is what gets carried.
    """
    x = pool.symbol("x")
    assert ak.solve([1 / (1 / x - 1)], [x]) == []


def test_a_pole_of_a_sibling_equation_excludes_the_root(pool):
    """The numerator system has the root (1, 1); the first equation has a pole there."""
    x, y = pool.symbol("x"), pool.symbol("y")
    assert ak.solve([(x - 1) / (y - 1), y - 1], [x, y]) == []


def test_genuine_roots_of_a_rational_equation_survive(pool):
    """Excluding poles must not degenerate into excluding roots."""
    x = pool.symbol("x")
    sols = ak.solve([(x * x - 4) / (x - 1)], [x])
    assert _values(sols, x) == pytest.approx([-2.0, 2.0])
    # Nothing parametric is involved, so nothing had to be assumed.
    assert ak.solve_side_conditions() == []


# ---------------------------------------------------------------------------
# Undecidable exclusions are stated, never assumed
# ---------------------------------------------------------------------------


def test_a_cleared_parametric_denominator_is_reported(pool):
    """Whether ``R1`` vanishes is nobody's to decide here, so it is stated."""
    Vo, Vin, R1, C, s = (pool.symbol(n) for n in ("Vo", "Vin", "R1", "C", "s"))
    ak.solve([(Vo - Vin) / R1 + Vo * s * C], [Vo])
    conditions = ak.solve_side_conditions()
    assert any(str(c).startswith("R1 ") for c in conditions), conditions


def test_a_positive_denominator_discharges_its_own_hypothesis(pool):
    """A resistance declared positive cannot be zero, so no hypothesis is reported."""
    Vo, Vin, C, s = (pool.symbol(n) for n in ("Vo", "Vin", "C", "s"))
    R1 = pool.symbol("R1", "positive")
    sols = ak.solve([(Vo - Vin) / R1 + Vo * s * C], [Vo])
    assert len(sols) == 1
    assert not any(str(c).startswith("R1 ") for c in ak.solve_side_conditions())


# ---------------------------------------------------------------------------
# What is still refused
# ---------------------------------------------------------------------------


def test_an_everywhere_undefined_equation_is_refused(pool):
    """``1/(x - x)`` denotes no function, so it has no solution set to report."""
    x = pool.symbol("x")
    with pytest.raises(ak.SolverError) as exc:
        ak.solve([1 / (x - x)], [x])
    assert exc.value.code == "E-SOLVE-005"


def test_a_transcendental_outside_the_closed_form_slice_still_refuses(pool):
    """Widening to rational functions does not widen to transcendental ones."""
    x, y = pool.symbol("x"), pool.symbol("y")
    with pytest.raises(ak.SolverError) as exc:
        ak.solve([ak.exp(x) + ak.sin(y), x - y], [x, y])
    assert exc.value.code == "E-SOLVE-001"


def test_a_rational_transcendental_is_refused_not_cleared(pool):
    """A denominator around a transcendental is still not a rational function."""
    x = pool.symbol("x")
    with pytest.raises(ak.SolverError) as exc:
        ak.solve([1 / (ak.exp(x) - 2) - 1], [x])
    assert exc.value.code == "E-SOLVE-001"


# ---------------------------------------------------------------------------
# Regression: the polynomial path is untouched
# ---------------------------------------------------------------------------


def test_polynomial_input_is_unchanged(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    assert _values(ak.solve([x * x - 4], [x]), x) == pytest.approx([-2.0, 2.0])
    assert ak.solve_side_conditions() == []
    (sol,) = ak.solve([x + y - 1, x - y], [x, y])
    assert float(ak.eval_expr(sol[x], {})) == pytest.approx(0.5)
    assert float(ak.eval_expr(sol[y], {})) == pytest.approx(0.5)
