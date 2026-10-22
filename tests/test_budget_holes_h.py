"""Report 10-21 §2.3 — budget holes H1, H2, H3 and the Leech lattice.

* **H1** ``number_theory.discrete_log`` was a linear sweep over ``0..p-1``
  with no budget checkpoint: ``p = 2**31 - 1`` under ``wall_ms=2000`` ran for
  55 s and ``2**127 - 1`` never finished. It is now Pohlig–Hellman with
  baby-step giant-step / Pollard rho, and polls the budget.
* **H2** ``PermutationGroup.order()`` ignored the budget and cost roughly
  ``n**6`` (``S_70`` under ``wall_ms=1000`` ran 7.5 s, ``symmetric(100)`` 74 s).
  Schreier–Sims now re-checks only new Schreier generators and polls.
* **H3** ``verified_sign`` ran its search on a worker thread that never saw
  the caller's thread-local budget (115 s on a 2-D Black–Scholes box).
* **Leech** ``Lattice.leech().kissing_number()`` ignored ``wall_ms`` (8 s).

Every budget trip is :class:`alkahest.BudgetExceededError` (``E-BUDGET-*``),
never a partial result. Timing ceilings are generous (a termination check on a
loaded CI runner, not a deadline).
"""

from __future__ import annotations

import math
import random
import time

import alkahest as ak
import alkahest.experimental as E
import pytest
from alkahest import number_theory as nt

#: Stopping inside this is "honoured the budget" — before the fix each of
#: these ran for tens of seconds or never finished.
CEILING_S = 10.0


@pytest.fixture(autouse=True)
def _no_cancel():
    ak.clear_cancel()
    yield
    ak.clear_cancel()


def _budget(ms: int):
    return ak.context(budget=ak.Budget(wall_ms=ms))


# ---------------------------------------------------------------------------
# H1 — discrete_log
# ---------------------------------------------------------------------------


def _smooth_prime(rng: random.Random, bits: int, max_factor_bits: int) -> int:
    """A random prime ``p < 2**bits`` whose ``p - 1`` has no prime factor
    above ``2**max_factor_bits`` (rejection sampling with SymPy)."""
    sympy = pytest.importorskip("sympy")
    while True:
        p = sympy.nextprime(rng.randrange(2 ** (bits - 1), 2**bits))
        if p >= 2**bits:
            continue
        if max(sympy.factorint(p - 1)) < 2**max_factor_bits:
            return p


def test_discrete_log_matches_sympy_on_random_primes() -> None:
    sympy = pytest.importorskip("sympy")
    rng = random.Random(20261022)
    for bits in (8, 16, 24, 32, 40, 48, 56, 64):
        for _ in range(3):
            p = _smooth_prime(rng, bits, 36)
            g = rng.randrange(2, p)
            e = rng.randrange(0, p - 1)
            h = pow(g, e, p)
            got = nt.discrete_log(h, g, p)
            assert pow(g, got, p) == h
            # The smallest exponent: below the order of g, and equal to SymPy's.
            assert 0 <= got < sympy.n_order(g, p)
            assert got == sympy.discrete_log(p, h, g)


def test_discrete_log_report_repro_is_fast() -> None:
    # p = 2**31 - 1: 55 s before; 2**127 - 1 (smooth p - 1) never finished.
    for p, g in ((2**31 - 1, 7), (2**127 - 1, 3), (2**61 - 1, 37), (10**12 + 39, 5)):
        h = pow(g, 987654321987654321 % (p - 1), p)
        t = time.perf_counter()
        with _budget(2000):
            e = nt.discrete_log(h, g, p)
        assert pow(g, e, p) == h
        assert time.perf_counter() - t < CEILING_S


def test_discrete_log_refusals() -> None:
    # 4 has order 4 mod 17, and 3 is a generator: 3 is not a power of 4.
    with pytest.raises(nt.NumberTheoryError) as ei:
        nt.discrete_log(3, 4, 17)
    assert ei.value.code == "E-NT-003"
    with pytest.raises(nt.NumberTheoryError) as ei:
        nt.discrete_log(2, 3, 15)
    assert ei.value.code == "E-NT-004"
    # Unchanged small answers (negative inputs are reduced first).
    assert nt.discrete_log(13, 3, 17) == 4
    assert nt.discrete_log(-1, 2, 5) == 2


def test_discrete_log_budget_trips_on_a_hard_order() -> None:
    # Safe prime: the order of 4 is a 61-bit prime, out of reach of any
    # generic algorithm in a test. The budget must stop it promptly.
    p = 4611686018427394499
    h = pow(4, 987654321987654321, p)
    t = time.perf_counter()
    with pytest.raises(ak.BudgetExceededError) as ei, _budget(300):
        nt.discrete_log(h, 4, p)
    assert ei.value.code.startswith("E-BUDGET-")
    assert time.perf_counter() - t < CEILING_S


# ---------------------------------------------------------------------------
# H2 — PermutationGroup.order()
# ---------------------------------------------------------------------------


def _sym(n: int):
    return E.PermutationGroup(
        n,
        [
            E.Permutation.from_cycles(n, [tuple(range(n))]),
            E.Permutation.from_cycles(n, [(0, 1)]),
        ],
    )


def test_symmetric_order_is_exact() -> None:
    # S_70 took 7.5 s before the incremental Schreier–Sims and ~0.4 s after
    # (release, Linux). No wall-clock assertion: CI runners vary by 20x.
    assert _sym(70).order() == math.factorial(70)


def test_permutation_group_orders_unchanged() -> None:
    # Alternating group via 3-cycles, dihedral, a direct product: classic orders.
    n = 12
    alt = E.PermutationGroup(n, [E.Permutation.from_cycles(n, [(0, 1, i)]) for i in range(2, n)])
    assert alt.order() == math.factorial(n) // 2
    d = E.PermutationGroup(
        10,
        [
            E.Permutation.from_cycles(10, [tuple(range(10))]),
            E.Permutation.from_cycles(10, [(1, 9), (2, 8), (3, 7), (4, 6)]),
        ],
    )
    assert d.order() == 20
    prod = E.PermutationGroup(
        9,
        [
            E.Permutation.from_cycles(9, [(0, 1, 2)]),
            E.Permutation.from_cycles(9, [(0, 1)]),
            E.Permutation.from_cycles(9, [(3, 4, 5, 6, 7, 8)]),
            E.Permutation.from_cycles(9, [(3, 4)]),
        ],
    )
    assert prod.order() == 6 * 720


def test_permutation_group_order_honours_budget_and_does_not_cache_the_trip() -> None:
    g = _sym(200)
    t = time.perf_counter()
    with (
        pytest.raises(ak.BudgetExceededError) as ei,
        ak.context(budget=ak.Budget(max_steps=20)),
    ):
        g.order()
    assert ei.value.code.startswith("E-BUDGET-")
    assert time.perf_counter() - t < CEILING_S
    # The stop is not cached: without a budget the same object answers.
    small = _sym(30)
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=2)):
        small.order()
    assert small.order() == math.factorial(30)


def test_permutation_group_order_wall_budget() -> None:
    # S_256 is minutes of work; 100 ms must stop it.
    g = _sym(256)
    t = time.perf_counter()
    with pytest.raises(ak.BudgetExceededError) as ei, _budget(100):
        g.order()
    assert ei.value.code == "E-BUDGET-001"
    assert time.perf_counter() - t < CEILING_S


# ---------------------------------------------------------------------------
# Leech lattice
# ---------------------------------------------------------------------------


def test_leech_kissing_number_honours_wall_budget() -> None:
    leech = E.Lattice.leech()
    t = time.perf_counter()
    with pytest.raises(ak.BudgetExceededError) as ei, _budget(200):
        leech.kissing_number()
    assert ei.value.code == "E-BUDGET-001"
    assert time.perf_counter() - t < CEILING_S


def test_lattice_results_unchanged() -> None:
    assert E.Lattice.e8().kissing_number() == 240
    assert E.Lattice.d_n(4).kissing_number() == 24


# ---------------------------------------------------------------------------
# H3 — verified_sign
# ---------------------------------------------------------------------------


def _bs_delta(pool: ak.ExprPool):
    S = pool.symbol("S", ak.Domain.Positive)
    s = pool.symbol("sigma", ak.Domain.Positive)
    K, r, T = pool.integer(100), pool.rational(1, 50), pool.integer(1)

    def nc(z):
        return pool.rational(1, 2) * (1 + ak.erf(z / ak.sqrt(pool.integer(2))))

    d1 = (ak.log(S / K) + (r + s**2 / 2) * T) / (s * ak.sqrt(T))
    d2 = d1 - s * ak.sqrt(T)
    c = S * nc(d1) - K * ak.exp(-r * T) * nc(d2)
    return ak.diff(c, S).value, S, s


def test_verified_sign_honours_wall_budget() -> None:
    pool = ak.ExprPool()
    delta, S, s = _bs_delta(pool)
    # Unbudgeted this box takes about two minutes; the search runs on a worker
    # thread, which now inherits the caller's budget.
    t = time.perf_counter()
    with pytest.raises(ak.BudgetExceededError) as ei, _budget(500):
        ak.verified_sign(delta, [(S, 80.0, 120.0), (s, 0.1, 0.6)], "nonnegative")
    assert ei.value.code.startswith("E-BUDGET-")
    assert time.perf_counter() - t < CEILING_S


def test_verified_sign_unchanged_without_budget() -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    assert ak.verified_sign(x**2 + 1, [(x, -2.0, 2.0)], "positive") == "true"
    assert ak.verified_sign(x, [(x, -1.0, 1.0)], "positive") == "false"
    # A generous budget does not change a quick verdict.
    with _budget(60_000):
        assert ak.verified_sign(ak.exp(x), [(x, -3.0, 3.0)], "positive") == "true"
