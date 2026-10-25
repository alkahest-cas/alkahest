"""``real_roots`` never returns a partial list (launch report 2026-10-24, row 2).

``real_roots(x**20 - 2*(10**6*x - 1)**2, x)`` returned **one** root of four after
about 10 s, with no error. The VAS frame stack was capped at 500 000 frames and
the cap ended the search with a bare ``break``; the roots of every frame still on
the stack were dropped. The cap was reached because the continued-fraction lower
bound searched integer points only up to ``10**6``: a frame whose roots sat beyond
that got no shift and fell back to unit bisection steps. The two roots near
``10**-6`` agree to ~60 digits, so their partial quotients are ~``10**54``.

The oracle is SymPy's ``count_roots`` (Sturm sequences), independent of the
Descartes/VAS search.
"""

from __future__ import annotations

import random
import time

import alkahest as ak
import pytest

sympy = pytest.importorskip("sympy")


def _alkahest_poly(pool, x, coeffs):
    expr = pool.integer(0)
    for k, c in enumerate(coeffs):
        if c:
            expr = expr + pool.integer(c) * x**k
    return expr


def _sympy_count(coeffs):
    t = sympy.Symbol("t")
    return sympy.Poly(list(reversed(coeffs)), t).count_roots()


def _mignotte(n, a):
    """Coefficients (ascending) of ``x**n - 2*(a*x - 1)**2``."""
    c = [0] * (n + 1)
    c[n] += 1
    c[2] -= 2 * a * a
    c[1] += 4 * a
    c[0] -= 2
    return c


def _check(pool, x, coeffs):
    ivs = ak.real_roots(_alkahest_poly(pool, x, coeffs), x)
    assert len(ivs) == _sympy_count(coeffs), coeffs
    ivs = sorted(ivs, key=lambda iv: iv.lo)
    for a, b in zip(ivs, ivs[1:]):
        assert a.hi <= b.lo
    return ivs


@pytest.mark.parametrize("n", [20, 40, 60])
def test_report_repro_mignotte_finds_all_four_roots(n):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    t0 = time.perf_counter()
    ivs = ak.real_roots(x**n - 2 * (pool.integer(10**6) * x - 1) ** 2, x)
    elapsed = time.perf_counter() - t0
    assert len(ivs) == 4, ivs
    assert len(ivs) == _sympy_count(_mignotte(n, 10**6))
    assert elapsed < 5.0, elapsed


def test_tighter_mignotte_cluster():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    for a in [10**12, 10**20]:
        _check(pool, x, _mignotte(20, a))


def test_random_corpus_with_planted_clusters_matches_sturm():
    rng = random.Random(20261024)
    pool = ak.ExprPool()
    x = pool.symbol("x")
    for trial in range(80):
        deg = rng.randint(3, 18)
        c = [rng.randint(-20, 20) for _ in range(deg + 1)]
        if c[-1] == 0:
            c[-1] = 1
        if trial % 2 == 0:
            a = 10 ** rng.randint(2, 9)
            c[2] -= 2 * a * a
            c[1] += 4 * a
            c[0] -= 2
        if all(v == 0 for v in c):
            continue
        _check(pool, x, c)


def test_product_of_close_rational_roots():
    # (x - 1/10**k)(x - 1/10**k - 1/10**(2k)) ... distinct, tightly spaced roots.
    pool = ak.ExprPool()
    x = pool.symbol("x")
    t = sympy.Symbol("t")
    for k in [3, 8, 15]:
        f = sympy.expand((10**k * t - 1) * (10 ** (2 * k) * t - 10**k - 1) * (t + 3) * (t**2 - 2))
        coeffs = [int(v) for v in reversed(sympy.Poly(f, t).all_coeffs())]
        ivs = _check(pool, x, coeffs)
        assert len(ivs) == 5
