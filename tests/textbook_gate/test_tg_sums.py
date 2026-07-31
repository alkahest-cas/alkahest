"""Textbook gate — summation.

First-course finite sums: Faulhaber sums (Σk, Σk², Σk³), geometric series,
and telescoping sums. B4 (report7-20.md) fixed Faulhaber/geometric support in
`sum_definite` / `sum_indefinite`. Also covers the Basel-family infinite
p-series (`Σ 1/n²`, `Σ 1/n⁴`) — the remaining agent-benchmark gap: alkahest
used to refuse every infinite-bound sum with `E-SUM-002` even where a closed
form exists.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from _tg_helpers import assert_infinite_sum_value, assert_sum_closed_form


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


# --- Basel-family infinite sums --------------------------------------------
#
# Gosper's algorithm never applies to Σ 1/k^p (its antidifference is a
# polygamma function, not hypergeometric), so these are recognized via a
# small even-zeta / Bernoulli-number table rather than Gosper — see
# `alkahest_core::sum::special`. Only even powers have a known closed form in
# `pi`; odd ones (Apéry's ζ(3), …) are honestly refused, not guessed.


def test_basel_sum_pi_squared_over_six(pool, k):
    """Σ_{n=1}^∞ 1/n² = π²/6 — the Basel problem."""
    term = ak.simplify(k ** pool.integer(-2)).value
    assert_infinite_sum_value(term, k, pool.integer(1), pool, math.pi**2 / 6)


def test_sum_reciprocal_fourth_power_to_infinity(pool, k):
    """Σ_{n=1}^∞ 1/n⁴ = π⁴/90."""
    term = ak.simplify(k ** pool.integer(-4)).value
    assert_infinite_sum_value(term, k, pool.integer(1), pool, math.pi**4 / 90)


def test_basel_sum_carries_a_scalar_coefficient(pool, k):
    """Σ_{n=1}^∞ 3/n² = 3·π²/6 = π²/2 — the coefficient must ride along."""
    term = ak.simplify(3 / k**2).value
    assert_infinite_sum_value(term, k, pool.integer(1), pool, 3 * math.pi**2 / 6)


def test_sum_reciprocal_cube_to_infinity_refuses(pool, k):
    """Σ 1/n³ = ζ(3) (Apéry's constant) has no known closed form in π — must
    raise E-SUM-002, not silently return a wrong or unevaluated value."""
    term = ak.simplify(k ** pool.integer(-3)).value
    with pytest.raises(ak.SumError) as exc_info:
        ak.sum_definite(term, k, pool.integer(1), pool.pos_infinity())
    assert exc_info.value.code == "E-SUM-002"


def test_sum_k_squared_to_infinity_diverges_and_refuses(pool, k):
    """Σ_{n=1}^∞ n² diverges — must not be mistaken for a p-series."""
    with pytest.raises(ak.SumError):
        ak.sum_definite(k**2, k, pool.integer(1), pool.pos_infinity())


def test_basel_sum_requires_lower_bound_one(pool, k):
    """Σ_{n=2}^∞ 1/n² has no simple closed form here (would need a finite
    correction term with no closed form of its own) — honestly refused
    rather than silently starting from n=1 anyway."""
    term = ak.simplify(k ** pool.integer(-2)).value
    with pytest.raises(ak.SumError):
        ak.sum_definite(term, k, pool.integer(2), pool.pos_infinity())
