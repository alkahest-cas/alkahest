"""Textbook gate — summation.

First-course finite sums: Faulhaber sums (Σk, Σk², Σk³), geometric series,
and telescoping sums. As of the 2026-07-20 usage eval (report7-20.md, bug
B4), `alkahest.sum_definite`/`sum_indefinite` reject essentially everything
in this category — including the *constant* sum `Σ_{k=1}^{n} 5`, which is
Gosper-summable by definition (constant ratio) and isn't even Faulhaber-tier.
The one demonstrated-working shape is a term pre-wrapped in `gamma(k+1)`
(mirrors `tests/test_sum_v210.py`); it's kept here as a green canary so a
change that breaks summation entirely (vs. just failing to *extend* support)
is still caught.

Every red case below is `xfail(strict=True)`: if a fix lands, the case starts
unexpectedly passing, pytest reports that as a failure, and the marker should
be deleted (see `tests/textbook_gate/README.md`).
"""

from __future__ import annotations

import alkahest as ak
import pytest
from _tg_helpers import assert_sum_closed_form

_SUM_NOT_SUPPORTED = "B4 (report7-20.md): sum_definite/sum_indefinite reject this term as 'not Gosper-summable' / 'not hypergeometric'"


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def k(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("k")


@pytest.fixture
def n(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("n")


# --- green canary: the one shape known to work ------------------------------


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


# --- Faulhaber sums (all currently broken) ----------------------------------


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_constant(pool, k, n):
    """Σ_{k=1}^{n} 5 = 5n — the simplest possible Gosper-summable term."""
    assert_sum_closed_form(pool.integer(5), k, n, pool.integer(1), lambda ni: 5 * ni)


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_k_faulhaber(pool, k, n):
    """Σ_{k=1}^{n} k = n(n+1)/2."""
    assert_sum_closed_form(k, k, n, pool.integer(1), lambda ni: ni * (ni + 1) // 2)


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_k_squared_faulhaber(pool, k, n):
    """Σ_{k=1}^{n} k² = n(n+1)(2n+1)/6."""
    assert_sum_closed_form(
        k**2, k, n, pool.integer(1), lambda ni: ni * (ni + 1) * (2 * ni + 1) // 6
    )


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_k_cubed_faulhaber(pool, k, n):
    """Σ_{k=1}^{n} k³ = (n(n+1)/2)² (Nicomachus)."""
    assert_sum_closed_form(k**3, k, n, pool.integer(1), lambda ni: (ni * (ni + 1) // 2) ** 2)


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_arithmetic_series(pool, k, n):
    """Σ_{k=1}^{n} (2k+1) = n(n+2) — arithmetic series with first term 3, step 2."""
    term = ak.simplify(2 * k + pool.integer(1)).value
    assert_sum_closed_form(term, k, n, pool.integer(1), lambda ni: ni * (ni + 2))


# --- geometric series (all currently broken) --------------------------------


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_geometric_ratio_2(pool, k, n):
    """Σ_{k=0}^{n} 2^k = 2^(n+1) - 1."""
    term = pool.integer(2) ** k
    assert_sum_closed_form(term, k, n, pool.integer(0), lambda ni: 2 ** (ni + 1) - 1)


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_geometric_ratio_half(pool, k, n):
    """Σ_{k=0}^{n} (1/2)^k = 2 - (1/2)^n — converges toward 2."""
    term = pool.rational(1, 2) ** k
    assert_sum_closed_form(term, k, n, pool.integer(0), lambda ni: 2 - (0.5) ** ni)


# --- telescoping sums (all currently broken) --------------------------------


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_telescoping_reciprocal_product(pool, k, n):
    """Σ_{k=1}^{n} 1/(k(k+1)) = 1 - 1/(n+1) = n/(n+1) — classic telescoping sum."""
    term = 1 / (k * (k + pool.integer(1)))
    assert_sum_closed_form(term, k, n, pool.integer(1), lambda ni: ni / (ni + 1))


@pytest.mark.xfail(strict=True, reason=_SUM_NOT_SUPPORTED)
def test_sum_indefinite_k(k):
    """Σk (antidifference) should be Gosper-summable — it's the textbook example."""
    ak.sum_indefinite(k, k)
