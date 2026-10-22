"""diophantine(x^2 + y^2 - n) returns every integer point (all signs and orders).

Regression for N6: only the non-negative representatives were returned
(n = 25 gave 4 of the 12 points).  Oracle: brute-force enumeration.
"""

import math
import random

import alkahest as ak
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(ak, "diophantine"),
    reason="native module built without groebner feature",
)


def _brute(n):
    r = math.isqrt(n) + 1
    return {(x, y) for x in range(-r, r + 1) for y in range(-r, r + 1) if x * x + y * y == n}


def _points(sol):
    if sol.kind == "no_solution":
        return set()
    assert sol.kind == "finite", sol.kind
    pts = [tuple(int(str(v)) for v in pt) for pt in sol.points]
    assert len(pts) == len(set(pts))
    return set(pts)


def _solve(n):
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    return _points(ak.diophantine(x**2 + y**2 - p.integer(n), [x, y]))


NS = [0, 1, 2, 3, 4, 5, 25, 65, 85, 325, 1105, 4225, 406250, 7, 21, 99, 196]


@pytest.mark.parametrize("n", NS)
def test_named_n(n):
    assert _solve(n) == _brute(n)


def test_report_repro_25_has_12_points():
    assert len(_solve(25)) == 12


def test_randomised_against_brute_force():
    rng = random.Random(1021)
    for _ in range(150):
        n = rng.randint(0, 5000)
        assert _solve(n) == _brute(n), n
    for p in (5, 13, 17, 29, 37, 41, 97, 101, 1009, 10009):
        assert len(_solve(p)) == 8
    for n in (3, 7, 11, 15, 19, 23, 27, 31, 35, 4003):
        assert _solve(n) == set()


def test_scaled_form():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sol = ak.diophantine(3 * x**2 + 3 * y**2 - p.integer(75), [x, y])
    assert _points(sol) == _brute(25)
