"""Textbook gate — summation.

First-course finite sums: Faulhaber sums (Σk, Σk², Σk³), geometric series,
and telescoping sums. B4 (report7-20.md) fixed Faulhaber/geometric support in
`sum_definite` / `sum_indefinite`.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from _tg_helpers import assert_sum_closed_form


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def k(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("k")


@pytest.fixture
def n(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("n")


# --- green canary: gamma-wrapped factorial shape --------------------------


def test_sum_k_factorial_gamma_wrapped(pool, k, n):
    """Σ_{k=0}^{n} k·k! = (n+1)! - 1, term written via gamma per test_sum_v210.py."""
    import math

    term = ak.simplify(k * ak.gamma(k + pool.integer(1))).value
    assert_sum_closed_form(
        term,
        k,
        n,
        pool.integer(0),
        lambda ni: sum(m * math.factorial(m) for m in range(ni + 1)),
        n_values=(0, 1, 2, 5, 8),
    )


# --- Faulhaber sums -------------------------------------------------------


def test_sum_constant(pool, k, n):
    """Σ_{k=1}^{n} 5 = 5n — the simplest possible Gosper-summable term."""
    assert_sum_closed_form(pool.integer(5), k, n, pool.integer(1), lambda ni: 5 * ni)


def test_sum_k_faulhaber(pool, k, n):
    """Σ_{k=1}^{n} k = n(n+1)/2."""
    assert_sum_closed_form(k, k, n, pool.integer(1), lambda ni: ni * (ni + 1) // 2)


def test_sum_k_squared_faulhaber(pool, k, n):
    """Σ_{k=1}^{n} k² = n(n+1)(2n+1)/6."""
    assert_sum_closed_form(
        k**2, k, n, pool.integer(1), lambda ni: ni * (ni + 1) * (2 * ni + 1) // 6
    )


def test_sum_k_cubed_faulhaber(pool, k, n):
    """Σ_{k=1}^{n} k³ = (n(n+1)/2)² (Nicomachus)."""
    assert_sum_closed_form(k**3, k, n, pool.integer(1), lambda ni: (ni * (ni + 1) // 2) ** 2)


def test_sum_arithmetic_series(pool, k, n):
    """Σ_{k=1}^{n} (2k+1) = n(n+2) — arithmetic series with first term 3, step 2."""
    term = ak.simplify(2 * k + pool.integer(1)).value
    assert_sum_closed_form(term, k, n, pool.integer(1), lambda ni: ni * (ni + 2))


# --- geometric series -----------------------------------------------------


def test_sum_geometric_ratio_2(pool, k, n):
    """Σ_{k=0}^{n} 2^k = 2^(n+1) - 1."""
    term = pool.integer(2) ** k
    assert_sum_closed_form(term, k, n, pool.integer(0), lambda ni: 2 ** (ni + 1) - 1)


def test_sum_geometric_ratio_half(pool, k, n):
    """Σ_{k=0}^{n} (1/2)^k = 2 - (1/2)^n — converges toward 2."""
    term = pool.rational(1, 2) ** k
    assert_sum_closed_form(term, k, n, pool.integer(0), lambda ni: 2 - (0.5) ** ni)


# --- telescoping sums -----------------------------------------------------


def test_sum_telescoping_reciprocal_product(pool, k, n):
    """Σ_{k=1}^{n} 1/(k(k+1)) = 1 - 1/(n+1) = n/(n+1) — classic telescoping sum."""
    term = 1 / (k * (k + pool.integer(1)))
    assert_sum_closed_form(term, k, n, pool.integer(1), lambda ni: ni / (ni + 1))


def test_sum_indefinite_k(k):
    """Σk (antidifference) should be Gosper-summable — it's the textbook example."""
    ak.sum_indefinite(k, k)
