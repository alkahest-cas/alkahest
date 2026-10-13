"""pell_classes: every solution class of x² − D·y² = N (F17).

`diophantine` reports one class for a generalized Pell equation; N can have
several inequivalent fundamental solutions.  SymPy's `diop_DN` (which returns
all fundamental solutions) is the oracle.
"""

import math

import alkahest as ak
import pytest

sympy_ntheory = pytest.importorskip("sympy.solvers.diophantine.diophantine")

pytestmark = pytest.mark.skipif(
    not hasattr(ak, "pell_classes"),
    reason="native module built without groebner feature",
)


def _sign(s, t, d):
    """Exact sign of s + t*sqrt(d)."""
    if s >= 0 and t >= 0:
        return 0 if s == 0 and t == 0 else 1
    if s <= 0 and t <= 0:
        return -1
    lhs, rhs = s * s, t * t * d
    if s > 0:  # t < 0
        return (lhs > rhs) - (lhs < rhs)
    return (rhs > lhs) - (rhs < lhs)


def _canon(d, n, unit, x, y):
    """Orbit of x+y√d under ±unit^Z as the representative in [√|n|, √|n|·u)."""
    ux, uy = unit
    m = abs(n)
    if _sign(x, y, d) < 0:
        x, y = -x, -y
    ua, ub = ux * ux + d * uy * uy, 2 * ux * uy  # u² = ua + ub√d
    while True:
        aa, bb = x * x + d * y * y, 2 * x * y  # α²
        if _sign(aa - m, bb, d) < 0:
            x, y = x * ux + d * y * uy, x * uy + y * ux
        elif _sign(aa - m * ua, bb - m * ub, d) >= 0:
            x, y = x * ux - d * y * uy, y * ux - x * uy
        else:
            return (x, y)


def _sympy_classes_both_signs(d, n):
    unit = ak.pell_classes(d, 1)["unit"]
    out = set()
    for x, y in sympy_ntheory.diop_DN(d, n):
        for sx in (1, -1):
            out.add(_canon(d, n, unit, sx * int(x), int(y)))
    return out


@pytest.mark.parametrize(
    ("d", "n", "expected"),
    [(10, 9, {(3, 0), (7, 2), (13, 4)}), (2, 49, None), (13, -1, {(18, 5)}), (13, 1, {(1, 0)})],
)
def test_known_cases(d, n, expected):
    r = ak.pell_classes(d, n)
    got = set(map(tuple, r["classes"]))
    for x, y in got:
        assert x * x - d * y * y == n
    if expected is not None:
        assert got == expected
    else:
        assert len(got) >= 2
    assert r["unit"][0] ** 2 - d * r["unit"][1] ** 2 == 1


def test_unsolvable_and_negative_unit():
    assert ak.pell_classes(3, -1)["classes"] == []
    assert ak.pell_classes(3, -1)["negative_unit"] is None
    assert ak.pell_classes(13, 3)["negative_unit"] == (18, 5)


def test_oracle_sweep():
    """|N| <= 50, non-square D <= 100 against SymPy's diop_DN."""
    checked = 0
    for d in range(2, 101):
        if math.isqrt(d) ** 2 == d:
            continue
        for n in range(-50, 51):
            if n == 0:
                continue
            ours = set(map(tuple, ak.pell_classes(d, n)["classes"]))
            # diop_DN returns one orbit rep per class under u^Z with either
            # x-sign normalised; compare after folding in the x -> -x variant,
            # which is the conjugate class and also a solution.
            theirs = _sympy_classes_both_signs(d, n)
            assert ours == theirs, (d, n, ours, theirs)
            checked += 1
    assert checked > 4000


def test_huge_unit_uses_convergents():
    # D = 991 has a fundamental unit with 30 digits: the Nagell sweep is out of
    # reach, so this exercises the convergent branch (|N| < sqrt(D)).
    r = ak.pell_classes(991, 3)
    for n_ in (3, -3, 7, -5):
        r = ak.pell_classes(991, n_)
        got = set(map(tuple, r["classes"]))
        assert got == _sympy_classes_both_signs(991, n_), n_
        for x, y in got:
            assert x * x - 991 * y * y == n_


def test_refusals():
    for d, n in [(9, 2), (-3, 1), (0, 1), (2, 0)]:
        with pytest.raises(ak.DiophantineError):
            ak.pell_classes(d, n)


def test_budget():
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=3)):
        ak.pell_classes(94, -7)
