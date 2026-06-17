"""Transcendental (exp/log) solving via ``solve`` (issue #5).

These exercise the scoped transcendental pre-processing layer added on top of
the polynomial (Gröbner) solver.  Skipped when the native module is built
without the ``groebner`` feature (``solve`` itself is gated on it).
"""

import math

import alkahest
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "solve_numerical"),
    reason="native module built without groebner feature",
)


def test_exp_x_eq_a():
    # exp(x) = 3  ->  x = ln 3
    p = alkahest.ExprPool()
    x = p.symbol("x")
    eq = alkahest.exp(x) + p.integer(-3)  # exp(x) - 3 = 0
    sols = alkahest.solve([eq], [x], numeric=True)
    assert len(sols) == 1
    assert abs(sols[0][x] - math.log(3)) < 1e-9


def test_log_x_eq_a():
    # log(x) = 2  ->  x = exp 2
    p = alkahest.ExprPool()
    x = p.symbol("x")
    eq = alkahest.log(x) + p.integer(-2)  # log(x) - 2 = 0
    sols = alkahest.solve([eq], [x], numeric=True)
    assert len(sols) == 1
    assert abs(sols[0][x] - math.exp(2)) < 1e-7


def test_half_life():
    # exp(-k*t) = 1/2  ->  t = ln(2)/k.  k is free; bind it numerically by
    # using a concrete k = 3 so we get a closed numeric answer.
    p = alkahest.ExprPool()
    t = p.symbol("t")
    three = p.integer(3)
    half = p.rational(-1, 2)  # exp(-3t) - 1/2 = 0
    eq = alkahest.exp(p.integer(-1) * three * t) + half
    sols = alkahest.solve([eq], [t], numeric=True)
    assert len(sols) == 1
    assert abs(sols[0][t] - math.log(2) / 3.0) < 1e-9


def test_exp_polynomial():
    # exp(x)^2 - 3 exp(x) + 2 = 0  ->  u in {1, 2}  ->  x in {0, ln 2}
    p = alkahest.ExprPool()
    x = p.symbol("x")
    ex = alkahest.exp(x)
    eq = ex**2 + p.integer(-3) * ex + p.integer(2)
    sols = alkahest.solve([eq], [x], numeric=True)
    vals = sorted(s[x] for s in sols)
    assert len(vals) == 2
    assert abs(vals[0] - 0.0) < 1e-9  # ln 1
    assert abs(vals[1] - math.log(2)) < 1e-9


def test_exp_no_real_solution_returns_empty():
    # exp(x) = -1 has no real solution -> empty list (not a fabricated answer).
    p = alkahest.ExprPool()
    x = p.symbol("x")
    eq = alkahest.exp(x) + p.integer(1)  # exp(x) + 1 = 0
    sols = alkahest.solve([eq], [x], numeric=True)
    assert sols == []


def test_unsupported_transcendental_falls_through_to_error():
    # sin(x) = 0 is not in the supported exp/log slice; the transcendental
    # layer reports Unsupported and the polynomial solver then rejects it with
    # a clean structured error (rather than returning a wrong answer).
    p = alkahest.ExprPool()
    x = p.symbol("x")
    eq = alkahest.sin(x)
    with pytest.raises(Exception):
        alkahest.solve([eq], [x])


def test_symbolic_output_is_ln():
    # Default (symbolic) output of exp(x) = 3 should be a log/ln expression,
    # numerically equal to ln 3.
    p = alkahest.ExprPool()
    x = p.symbol("x")
    eq = alkahest.exp(x) + p.integer(-3)
    sols = alkahest.solve([eq], [x])
    assert len(sols) == 1
    val = sols[0][x]
    # Numeric check via a second (numeric) solve to avoid depending on the
    # exact symbolic spelling.
    num = alkahest.solve([eq], [x], numeric=True)[0][x]
    assert abs(num - math.log(3)) < 1e-9
    # The symbolic value should render with a log/ln.
    s = str(val).lower()
    assert "log" in s or "ln" in s
