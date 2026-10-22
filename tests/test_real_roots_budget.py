"""X12 / report 10-21 C2 — ``real_roots`` honours the active ``Budget``.

A random degree-1500 integer polynomial used to abort the whole process inside
GMP (``GNU MP: Cannot allocate memory``, exit 134) because the isolation loops
never consulted ``Budget(wall_ms=..., max_bytes=...)``. They now poll it and
raise :class:`alkahest.BudgetExceededError` (``E-BUDGET-*``). Separately, the
results of ``real_roots`` are checked against SymPy/mpmath for polynomials that
stress the isolator: Wilkinson, Mignotte-type near-double roots, sparse high
degree, and random polynomials.
"""

from __future__ import annotations

import random
import time

import alkahest as ak
import pytest

try:
    import numpy as np
    import sympy as sp
except ImportError:  # the budget tests below need neither; the oracle ones skip
    np = sp = None

needs_oracle = pytest.mark.skipif(sp is None, reason="numpy and sympy are the oracles")


def _poly(pool, x, coeffs):
    """Sum of ``c * x**i`` for ascending integer ``coeffs``."""
    e = pool.integer(0)
    for i, c in enumerate(coeffs):
        if c:
            e = e + pool.integer(c) * x**i
    return e


def _random_coeffs(deg, seed):
    rng = random.Random(seed)
    c = [rng.randrange(-9, 10) for _ in range(deg + 1)]
    c[-1] = c[-1] or 1
    return c


@pytest.fixture(autouse=True)
def _no_cancel():
    ak.clear_cancel()
    yield
    ak.clear_cancel()


# --------------------------------------------------------------------------
# The repro: coded budget error, quickly, no abort
# --------------------------------------------------------------------------


def test_wall_budget_stops_degree_3000():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    p = _poly(pool, x, _random_coeffs(3000, 2))
    t = time.perf_counter()
    with pytest.raises(ak.BudgetExceededError) as ei, ak.context(budget=ak.Budget(wall_ms=100)):
        ak.real_roots(p, x)
    assert getattr(ei.value, "code", "").startswith("E-BUDGET-")
    assert time.perf_counter() - t < 20


def test_max_bytes_budget_stops_degree_3000():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    p = _poly(pool, x, _random_coeffs(3000, 2))
    t = time.perf_counter()
    budget = ak.Budget(wall_ms=60_000, max_bytes=1 << 18)
    with pytest.raises(ak.BudgetExceededError) as ei, ak.context(budget=budget):
        ak.real_roots(p, x)
    assert getattr(ei.value, "code", "").startswith("E-BUDGET-")
    assert time.perf_counter() - t < 30


def test_budget_does_not_poison_next_call():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    big = _poly(pool, x, _random_coeffs(3000, 2))
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(wall_ms=100)):
        ak.real_roots(big, x)
    assert len(ak.real_roots(x**2 - pool.integer(4), x)) == 2


# --------------------------------------------------------------------------
# Oracle checks under the default budget
# --------------------------------------------------------------------------


def _q(pair):
    return sp.Rational(int(pair[0]), int(pair[1]))


def _eval(coeffs, r):
    """Exact ``p(r)`` for a ``sympy.Rational`` ``r`` (Horner)."""
    acc = sp.Integer(0)
    for c in reversed(coeffs):
        acc = acc * r + c
    return acc


def _check_against_oracle(coeffs, counter="sympy"):
    """Our isolating intervals vs. an independent oracle.

    Every interval must (exactly, in rational arithmetic) bracket a sign change
    of ``p`` or be an exact root, the intervals must be disjoint, and their
    number must equal the oracle's count of distinct real roots: SymPy's Sturm
    ``count_roots`` for the squarefree part, or ``numpy.roots`` where Sturm is
    too slow.
    """
    pool = ak.ExprPool()
    x = pool.symbol("x")
    ours = sorted(ak.real_roots(_poly(pool, x, coeffs), x), key=lambda iv: iv.lo)
    sx = sp.Symbol("x")
    if counter == "sympy":
        pp = sp.Poly(list(reversed(coeffs)), sx)
        ref_n = sp.quo(pp, sp.gcd(pp, pp.diff(sx))).count_roots()
    else:
        roots = np.roots(list(reversed(coeffs)))
        ref_n = int(np.sum(np.abs(roots.imag) < 1e-7 * np.maximum(1.0, np.abs(roots))))
    assert len(ours) == ref_n, (len(ours), ref_n)
    prev_hi = None
    for iv in ours:
        lo, hi = _q(iv.lo_exact()), _q(iv.hi_exact())
        flo, fhi = _eval(coeffs, lo), _eval(coeffs, hi)
        assert flo == 0 or fhi == 0 or (flo > 0) != (fhi > 0), (iv.lo, iv.hi)
        if prev_hi is not None:
            assert prev_hi <= lo
        prev_hi = hi


@needs_oracle
def test_wilkinson_15():
    p = sp.Poly(sp.prod([sp.Symbol("x") - k for k in range(1, 16)]), sp.Symbol("x"))
    coeffs = [int(c) for c in reversed(p.all_coeffs())]
    pool = ak.ExprPool()
    x = pool.symbol("x")
    ivs = sorted(ak.real_roots(_poly(pool, x, coeffs), x), key=lambda iv: iv.lo)
    assert [(iv.lo, iv.hi) for iv in ivs] == [(float(k), float(k)) for k in range(1, 16)]


@needs_oracle
@pytest.mark.parametrize(("n", "a"), [(8, 20), (12, 50), (20, 100)])
def test_mignotte_close_roots(n, a):
    # x^n - 2 (a x - 1)^2 : two roots within ~1/(a^(n/2)) of 1/a.
    sx = sp.Symbol("x")
    p = sp.Poly(sx**n - 2 * (a * sx - 1) ** 2, sx)
    coeffs = [int(c) for c in reversed(p.all_coeffs())]
    _check_against_oracle(coeffs)


@needs_oracle
def test_sparse_high_degree():
    # x^600 - 3 x^301 + x^7 - 5  (sparse, degree 600)
    coeffs = [0] * 601
    coeffs[600], coeffs[301], coeffs[7], coeffs[0] = 1, -3, 1, -5
    _check_against_oracle(coeffs)


@needs_oracle
def test_repeated_roots_collapse():
    sx = sp.Symbol("x")
    p = sp.Poly((sx - 1) ** 3 * (sx + 2) ** 2 * (sx**2 - 3), sx)
    _check_against_oracle([int(c) for c in reversed(p.all_coeffs())])


@needs_oracle
@pytest.mark.parametrize(("deg", "seed"), [(30, 1), (60, 2), (120, 3), (200, 4)])
def test_random_polynomials_match_sympy(deg, seed):
    _check_against_oracle(_random_coeffs(deg, seed), "sympy" if deg <= 60 else "numpy")


@needs_oracle
def test_default_budget_degree_1500_repro_returns():
    """The report's exact repro now finishes instead of aborting (and well inside 5 s)."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    p = _poly(pool, x, _random_coeffs(1500, 2))
    t = time.perf_counter()
    with ak.context(budget=ak.Budget(wall_ms=5000, max_bytes=1 << 30)):
        ivs = ak.real_roots(p, x)
    assert time.perf_counter() - t < 5
    # Independent count from the companion-matrix eigenvalues.
    c = _random_coeffs(1500, 2)
    roots = np.roots(list(reversed(c)))
    assert len(ivs) == int(np.sum(np.abs(roots.imag) < 1e-7 * np.maximum(1.0, np.abs(roots))))
