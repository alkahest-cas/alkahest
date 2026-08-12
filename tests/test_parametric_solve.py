"""Parametric solve: free symbols omitted from ``vars`` stay as parameters."""

from __future__ import annotations

import math

import alkahest as ak
import pytest

pytest.importorskip("alkahest", reason="native extension required")


@pytest.fixture
def pool():
    return ak.ExprPool()


def test_solve_x_squared_equals_y(pool):
    """Headline case from report7-20: ``x² = y`` in ``[x]`` → ``±√y``."""
    x = pool.symbol("x")
    y = pool.symbol("y")
    sols = ak.solve([x**2 - y], [x])
    assert isinstance(sols, list)
    assert len(sols) == 2
    vals = sorted(float(ak.eval_expr(s[x], {y: 4.0})) for s in sols)
    assert vals == pytest.approx([-2.0, 2.0])


def test_solve_linear_parametric_coefficients(pool):
    """``a·x − b = 0`` in ``[x]`` → ``x = b/a``."""
    x = pool.symbol("x")
    a = pool.symbol("a")
    b = pool.symbol("b")
    sols = ak.solve([a * x - b], [x])
    assert len(sols) == 1
    assert float(ak.eval_expr(sols[0][x], {a: 2.0, b: 6.0})) == pytest.approx(3.0)


def test_solve_system_with_parameter(pool):
    """``x + y = c``, ``x − y = 0`` → ``x = y = c/2``."""
    x = pool.symbol("x")
    y = pool.symbol("y")
    c = pool.symbol("c")
    sols = ak.solve([x + y - c, x - y], [x, y])
    assert len(sols) == 1
    env = {c: 10.0}
    assert float(ak.eval_expr(sols[0][x], env)) == pytest.approx(5.0)
    assert float(ak.eval_expr(sols[0][y], env)) == pytest.approx(5.0)


def test_solve_circle_radius_parameter(pool):
    """``x² + y² = r²``, ``y = x`` → ``x = y = ±r/√2``."""
    x = pool.symbol("x")
    y = pool.symbol("y")
    r = pool.symbol("r")
    sols = ak.solve([x**2 + y**2 - r**2, y - x], [x, y])
    assert len(sols) == 2
    env = {r: math.sqrt(2.0)}
    pairs = sorted(
        (
            float(ak.eval_expr(s[x], env)),
            float(ak.eval_expr(s[y], env)),
        )
        for s in sols
    )
    assert pairs[0] == pytest.approx((-1.0, -1.0))
    assert pairs[1] == pytest.approx((1.0, 1.0))


def test_nonparametric_solve_still_works(pool):
    """Regression: fully declared vars unchanged."""
    x = pool.symbol("x")
    y = pool.symbol("y")
    sols = ak.solve([x + y - 1, x - y], [x, y])
    assert len(sols) == 1
    assert float(ak.eval_expr(sols[0][x], {})) == pytest.approx(0.5)
    assert float(ak.eval_expr(sols[0][y], {})) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The hypothesis behind a parametric answer.
#
# `b/a` is the solution of `a·x = b` **for a ≠ 0**. At `a = 0` the equation
# reads `−b = 0`: no solution when `b ≠ 0`, every `x` when `b = 0`. A parametric
# tuple is not a number, so nothing substitutes it back and it is returned
# unverified — which makes the stated hypothesis the only auditable signal.
# ---------------------------------------------------------------------------


def test_parametric_division_reports_its_non_vanishing_hypothesis(pool):
    x = pool.symbol("x")
    a = pool.symbol("a")
    b = pool.symbol("b")
    ak.solve([a * x - b], [x])
    conditions = ak.solve_side_conditions()
    assert conditions == ["a ≠ 0"], conditions
    # Reading it again describes the same call, not an empty list.
    assert ak.solve_side_conditions() == ["a ≠ 0"]


def test_a_provable_divisor_reports_no_hypothesis(pool):
    """The control: `2x − b = 0` divides by the literal 2, so there is nothing
    to assume. A solver that emitted a condition unconditionally would say as
    little as one that never emits any."""
    x = pool.symbol("x")
    b = pool.symbol("b")
    sols = ak.solve([2 * x - b], [x])
    assert float(ak.eval_expr(sols[0][x], {b: 6.0})) == pytest.approx(3.0)
    assert ak.solve_side_conditions() == []


def test_hypotheses_do_not_leak_from_an_earlier_solve(pool):
    """Each call resets the channel, including on paths that never reach the
    symbolic solver."""
    x = pool.symbol("x")
    a = pool.symbol("a")
    b = pool.symbol("b")
    ak.solve([a * x - b], [x])
    assert ak.solve_side_conditions() == ["a ≠ 0"]
    ak.solve([x**2 - 4], [x])
    assert ak.solve_side_conditions() == []
