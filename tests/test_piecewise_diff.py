"""W5 (report10-9): ``diff`` of a Piecewise must keep its conditions and
differentiate every branch.

The report's repro passed the branches in SymPy's ``(value, condition)``
order — the order the API docs documented — while the binding read them as
``(condition, value)`` without checking.  The value landed in the condition
slot, ``diff`` then differentiated the predicate to 0, and ``eval_expr``
refused the result (E-EVAL-008), so the wrong answer could not even be caught
numerically.  Every derivative here is checked against a central finite
difference away from the piece boundaries.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest


@pytest.fixture
def pool():
    return ak.ExprPool()


def _fd(expr, x, at: float, env: dict | None = None, h: float = 1e-6) -> float:
    env = dict(env or {})
    hi = ak.eval_expr(expr, {**env, x: at + h})
    lo = ak.eval_expr(expr, {**env, x: at - h})
    return (hi - lo) / (2 * h)


def _check_derivative(expr, x, points, env=None, tol=1e-5) -> None:
    for diff_fn in (ak.diff, ak.diff_forward):
        d = diff_fn(expr, x).value
        for at in points:
            got = ak.eval_expr(d, {**(env or {}), x: at})
            want = _fd(expr, x, at, env)
            assert got == pytest.approx(want, rel=tol, abs=tol), (diff_fn.__name__, at, str(d))


def test_report_repro_sympy_order(pool):
    """The report's call: ``(value, condition)`` order."""
    x = pool.symbol("x")
    zero = pool.integer(0)
    pw = ak.piecewise([(x**2, pool.gt(x, zero))], pool.integer(-1) * x)
    # Read as (condition, value): the condition is the predicate.
    assert ak.eval_expr(pw, {x: 3.0}) == pytest.approx(9.0)
    assert ak.eval_expr(pw, {x: -3.0}) == pytest.approx(3.0)
    d = ak.diff(pw, x).value
    assert ak.eval_expr(d, {x: 3.0}) == pytest.approx(6.0)
    assert ak.eval_expr(d, {x: -3.0}) == pytest.approx(-1.0)
    # The condition survives intact.
    assert ">" in str(d) or "gt" in str(d).lower()


def test_native_order_unchanged(pool):
    x = pool.symbol("x")
    zero = pool.integer(0)
    a = ak.piecewise([(pool.gt(x, zero), x**2)], pool.integer(-1) * x)
    b = ak.piecewise([(x**2, pool.gt(x, zero))], pool.integer(-1) * x)
    assert a == b


def test_branch_without_condition_raises(pool):
    x = pool.symbol("x")
    with pytest.raises(TypeError, match="no condition"):
        ak.piecewise([(x**2, x + pool.integer(1))], x)


@pytest.mark.parametrize("diff_name", ["diff", "diff_forward"])
def test_diff_keeps_conditions_and_differentiates_branches(pool, diff_name):
    x = pool.symbol("x")
    zero = pool.integer(0)
    pw = ak.piecewise([(pool.gt(x, zero), x**3)], ak.sin(x))
    d = getattr(ak, diff_name)(pw, x).value
    assert ak.eval_expr(d, {x: 2.0}) == pytest.approx(12.0)
    assert ak.eval_expr(d, {x: -2.0}) == pytest.approx(math.cos(-2.0))


def test_finite_difference_multi_branch(pool):
    x = pool.symbol("x")
    one, two = pool.integer(1), pool.integer(2)
    pw = ak.piecewise(
        [
            (pool.lt(x, pool.integer(-1)), ak.exp(x)),
            (pool.lt(x, one), x**3 - two * x),
            (pool.le(x, pool.integer(3)), ak.log(x) * ak.sin(x)),
        ],
        ak.sqrt(x),
    )
    _check_derivative(pw, x, [-2.5, -1.7, -0.4, 0.3, 0.9, 1.5, 2.8, 3.5, 7.0])


def test_finite_difference_nested(pool):
    x = pool.symbol("x")
    zero = pool.integer(0)
    inner = ak.piecewise([(pool.gt(x, pool.integer(2)), x**2)], ak.cos(x))
    outer = ak.piecewise([(pool.gt(x, zero), inner * x)], pool.integer(-3) * x)
    _check_derivative(outer, x, [-1.5, 0.5, 1.5, 2.5, 4.0])


def test_finite_difference_symbolic_condition(pool):
    x, a = pool.symbol("x"), pool.symbol("a")
    pw = ak.piecewise([(pool.gt(x, a), a * x**2)], ak.exp(a * x))
    env = {a: 0.7}
    _check_derivative(pw, x, [-1.0, 0.2, 0.69, 0.71, 2.0], env=env)
    # d/da is also branchwise, and the condition still compares x with a.
    da = ak.diff(pw, a).value
    for xv in (-1.0, 2.0):
        got = ak.eval_expr(da, {x: xv, a: 0.7})
        want = _fd(pw, a, 0.7, {x: xv})
        assert got == pytest.approx(want, rel=1e-5)


def test_piecewise_conditions_survive_simplify_and_subs(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    pw = ak.piecewise([(pool.gt(x, y), x * pool.integer(1) + pool.integer(0))], y)
    s = ak.simplify(pw).value
    assert ak.eval_expr(s, {x: 2.0, y: 1.0}) == pytest.approx(2.0)
    assert ak.eval_expr(s, {x: 0.0, y: 1.0}) == pytest.approx(1.0)
    t = ak.subs(pw, {y: pool.integer(1)})
    assert ak.eval_expr(t, {x: 2.0}) == pytest.approx(2.0)
    assert ak.eval_expr(t, {x: 0.5}) == pytest.approx(1.0)


def test_interval_eval_straddling_boundary_is_not_a_branch_pick(pool):
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(x, pool.integer(0)), x)], pool.integer(-1))
    inside = ak.interval_eval(pw, {x: ak.ArbBall(2.0, 1e-6)})
    assert inside.contains(2.0)
    # A ball straddling x = 0 must not silently pick one branch.
    try:
        ball = ak.interval_eval(pw, {x: ak.ArbBall(0.0, 0.5)})
    except Exception:
        return
    assert ball.contains(-1.0)
    assert ball.contains(0.4)


def test_series_on_piece_boundary_refuses(pool):
    x = pool.symbol("x")
    zero = pool.integer(0)
    step = ak.piecewise([(pool.gt(x, zero), pool.integer(1))], pool.integer(-1))
    with pytest.raises(Exception, match="E-SERIES-004"):
        ak.series(step, x, zero, 3)
    # Strictly inside a piece the expansion still works.
    ak.series(step, x, pool.integer(1), 3)


def test_symbolic_condition_diff_wrt_parameter_keeps_condition(pool):
    x, a = pool.symbol("x"), pool.symbol("a")
    cond = pool.gt(x, a)
    pw = ak.piecewise([(cond, x * a)], a)
    d = ak.diff(pw, a).value
    assert ak.eval_expr(d, {x: 2.0, a: 1.0}) == pytest.approx(2.0)
    assert ak.eval_expr(d, {x: 0.0, a: 1.0}) == pytest.approx(1.0)
