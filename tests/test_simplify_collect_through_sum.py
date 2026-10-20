"""F15 — like-term collection through a sum factor, and the cheaper zero sweep.

G4 left ``-a**2*(b-1) + a**2*b`` as written under ``simplify`` (only
``simplify_expanded`` reached ``a**2``).  ``simplify`` now collects like terms
that meet only through a sum factor, without expanding anything, and only
when that strictly shortens the sum.  Results are checked numerically.
"""

from __future__ import annotations

import random

import alkahest as ak

P = ak.ExprPool()
a = P.symbol("a")
b = P.symbol("b")
x = P.symbol("x")
y = P.symbol("y")


def _close(e1, e2, syms, trials=5):
    rng = random.Random(15)
    for _ in range(trials):
        env = {s: rng.uniform(0.3, 2.1) for s in syms}
        v1 = ak.eval_expr(e1, env)
        v2 = ak.eval_expr(e2, env)
        assert abs(v1 - v2) <= 1e-9 * (1 + abs(v2))


def test_report_repro_collects_without_expanding():
    e = -(a**2) * (b - 1) + a**2 * b
    r = ak.simplify(e)
    assert r.value == a**2
    _close(r.value, e, [a, b])


def test_rational_coefficients_and_residual_terms():
    e = P.rational(1, 2) * x * (y + 2) - P.rational(1, 2) * x * y + y
    r = ak.simplify(e).value
    assert r == x + y
    _close(r, e, [x, y])


def test_factored_sum_that_does_not_cancel_is_kept():
    e = x * (y + 1) + x
    r = ak.simplify(e).value
    _close(r, e, [x, y])
    # Still factored: no full expansion into x*y + 2*x happened.
    assert r != x * y + 2 * x


def test_nested_zero_still_found():
    inner = x * (y - 1) - x * y + x
    r = ak.simplify(ak.sin(inner))
    assert r.value == P.integer(0)
