"""Pell equations: fundamental units, negative Pell, generalized N, budgets.

Regression for the W8 defect: the continued-fraction step used an exact
division where the recurrence needs a floor, so most D (7, 13, 14, 19, …) were
refused with a false "no fundamental unit", and x² − D·y² = −1 with no solution
ran an unbounded convergent loop that ignored the active budget.
"""

import math
import time

import alkahest as ak
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(ak, "diophantine"),
    reason="native module built without groebner feature",
)


def _cf_sqrt(d):
    """Independent oracle: one period of the continued fraction of √d."""
    a0 = math.isqrt(d)
    m, q, a = 0, 1, a0
    period = []
    while True:
        m = q * a - m
        q = (d - m * m) // q
        a = (a0 + m) // q
        period.append(a)
        if a == 2 * a0:
            return a0, period


def _oracle_units(d):
    """(fundamental +1 solution, fundamental −1 solution or None)."""
    a0, period = _cf_sqrt(d)
    p_prev, q_prev, p, q = 1, 0, a0, 1
    for a in period[:-1]:
        p, p_prev = a * p + p_prev, p
        q, q_prev = a * q + q_prev, q
    if len(period) % 2 == 0:
        assert p * p - d * q * q == 1
        return (p, q), None
    assert p * p - d * q * q == -1
    return (p * p + d * q * q, 2 * p * q), (p, q)


def _brute_plus(d, limit=200_000):
    for y in range(1, limit):
        x = math.isqrt(1 + d * y * y)
        if x * x == 1 + d * y * y:
            return x, y
    return None


NON_SQUARE = [d for d in range(2, 201) if math.isqrt(d) ** 2 != d]


def _pool():
    p = ak.ExprPool()
    return p, p.symbol("x"), p.symbol("y")


def _ints(pair):
    return tuple(int(str(v)) for v in pair)


@pytest.mark.parametrize("d", [7, 13, 14, 19, 21, 22, 23, 46, 61, 94])
def test_report_repro_cases(d):
    p, x, y = _pool()
    sol = ak.diophantine(x**2 - p.integer(d) * y**2 - p.integer(1), [x, y])
    assert sol.kind == "pell_fundamental"
    assert _ints(sol.fundamental) == _oracle_units(d)[0]


def test_x2_minus_13y2_is_649_180():
    _p, x, y = _pool()
    sol = ak.diophantine(x**2 - 13 * y**2 - 1, [x, y])
    assert _ints(sol.fundamental) == (649, 180)


def test_fundamental_unit_every_nonsquare_d_up_to_200():
    p, x, y = _pool()
    for d in NON_SQUARE:
        sol = ak.diophantine(x**2 - p.integer(d) * y**2 - p.integer(1), [x, y])
        assert sol.kind == "pell_fundamental", d
        fx, fy = _ints(sol.fundamental)
        assert (fx, fy) == _oracle_units(d)[0], d
        assert fx * fx - d * fy * fy == 1
        # Independent brute force where the unit is small enough to reach.
        if fy < 200_000:
            assert _brute_plus(d) == (fx, fy), d


def test_negative_pell_solvability_up_to_200():
    p, x, y = _pool()
    for d in NON_SQUARE:
        sol = ak.diophantine(x**2 - p.integer(d) * y**2 + p.integer(1), [x, y])
        expected = _oracle_units(d)[1]
        if expected is None:
            assert sol.kind == "no_solution", d
        else:
            assert sol.kind == "pell_generalized", d
            x0, y0 = _ints(sol.pell_particular)
            assert (x0, y0) == expected, d
            assert x0 * x0 - d * y0 * y0 == -1
            ux, uy = _ints(sol.pell_unit)
            assert ux * ux - d * uy * uy == 1


def test_generalized_pell_against_sweep():
    """x² − D·y² = N decided against a direct sweep (Nagell bounds are tiny here)."""
    p, x, y = _pool()
    for d in [2, 3, 5, 6, 7, 11, 13, 21]:
        for n in [-7, -5, -4, -3, -2, 2, 3, 4, 5, 7, 9]:
            sol = ak.diophantine(x**2 - p.integer(d) * y**2 - p.integer(n), [x, y])
            witness = next(
                (
                    yy
                    for yy in range(0, 5000)
                    if n + d * yy * yy >= 0 and math.isqrt(n + d * yy * yy) ** 2 == n + d * yy * yy
                ),
                None,
            )
            if witness is None:
                assert sol.kind == "no_solution", (d, n)
            else:
                assert sol.kind == "pell_generalized", (d, n)
                x0, y0 = _ints(sol.pell_particular)
                assert x0 * x0 - d * y0 * y0 == n


def test_negated_and_square_d_forms():
    _p, x, y = _pool()
    # −x² − y² + 5 = 0
    sol = ak.diophantine(-(x**2) - y**2 + 5, [x, y])
    assert sol.kind == "finite"
    assert {_ints(pt) for pt in sol.points} == {(1, 2), (2, 1)}
    # −x² + 2y² + 1 = 0  ⇔  x² − 2y² = 1, reported in (x, y) order
    sol = ak.diophantine(-(x**2) + 2 * y**2 + 1, [x, y])
    assert sol.kind == "pell_fundamental"
    assert _ints(sol.fundamental) == (3, 2)
    # x² − 4y² = 5 → only (3, 1)
    sol = ak.diophantine(x**2 - 4 * y**2 - 5, [x, y])
    assert sol.kind == "finite"
    assert {_ints(pt) for pt in sol.points} == {(3, 1)}


def test_unsolvable_negative_pell_respects_budget():
    p, x, y = _pool()
    wall_ms = 2000
    t0 = time.monotonic()
    # D = 10¹⁴ + 31: CF period far beyond what fits in the budget.
    with pytest.raises(ak.AlkahestError), ak.context(budget=ak.Budget(wall_ms=wall_ms)):
        ak.diophantine(x**2 - p.integer(10**14 + 31) * y**2 + 1, [x, y])
    assert time.monotonic() - t0 < 3 * wall_ms / 1000


def test_budget_raises_budget_error():
    p, x, y = _pool()
    # √123456789012347 has a CF period of 737642 terms.
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=3)):
        ak.diophantine(x**2 - p.integer(123456789012347) * y**2 + 1, [x, y])


def test_report_negative_pell_d3_is_fast_no_solution():
    _p, x, y = _pool()
    t0 = time.monotonic()
    with ak.context(budget=ak.Budget(wall_ms=2000)):
        sol = ak.diophantine(x**2 - 3 * y**2 + 1, [x, y])
    assert sol.kind == "no_solution"
    assert time.monotonic() - t0 < 1.0
