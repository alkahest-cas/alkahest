"""M3 — minimal-order certification for Zeilberger's algorithm.

An order-4 recurrence where the literature records a guessed order-5 is a
result; an order-4 recurrence that *might* have been order 3 is a coincidence.
So the certified order is only worth something when something establishes it is
the least one, and in 3.9.0 nothing did: the search visits the
``(order, degree)`` grid cheapest-estimated-cost-first — which is what made
Dixon, Franel and Apéry decidable at the default bounds at all — so it can
reach a cheap order-2 probe long before an expensive order-1 one.

These tests hold two lines. The default search must **not** claim minimality it
did not establish, and ``minimal=True`` must actually establish it.
"""

import time

import alkahest as ak
import pytest


def _binomial(pool, top, bot):
    one = pool.integer(1)
    return ak.gamma(top + one) / (ak.gamma(bot + one) * ak.gamma(top - bot + one))


@pytest.fixture
def nk():
    pool = ak.ExprPool()
    return pool, pool.symbol("n"), pool.symbol("k")


def _dixon(pool, n, k):
    """``Σ_k (−1)^k C(n,k)³`` — minimal order 2, no order-1 relation."""
    c = _binomial(pool, n, k)
    return pool.integer(-1) ** k * c * c * c


def _franel(pool, n, k):
    """``Σ_k C(n,k)³`` — minimal order 2."""
    c = _binomial(pool, n, k)
    return c * c * c


# ---------------------------------------------------------------------------
# The flag is honest in the default (fast) mode
# ---------------------------------------------------------------------------


def test_order_one_is_reported_minimal_without_any_search(nk):
    """``Σ_k C(n,k) = 2ⁿ``: nothing is below order 1, so the flag is free.

    The one case where the cost-ordered plan can claim minimality without
    paying for it, and it should — a flag that is pessimistic where it need not
    be trains callers to ignore it.
    """
    pool, n, k = nk
    cert = ak.zeilberger(_binomial(pool, n, k), n, k)
    assert cert.order == 1
    assert cert.order_is_minimal


def test_default_search_does_not_claim_minimality_it_has_not_established(nk):
    """Dixon at the default bounds: order 2, and ``order_is_minimal`` is False.

    This is the whole point. The plan reaches ``(2, 0)`` after four of the
    seventeen order-1 probes in bounds, so thirteen order-1 degrees were never
    tried and the returned order-2 relation does not rule out an order-1 one.
    Reporting ``True`` here would be the false lemma a research loop inherits.
    """
    pool, n, k = nk
    cert = ak.zeilberger(_dixon(pool, n, k), n, k)
    assert cert.order == 2
    assert not cert.order_is_minimal


def test_the_flag_tracks_the_search_and_not_the_mode(nk):
    """At bounds narrow enough, the *default* plan does establish minimality.

    ``max_degree=4`` puts only five degrees at each order, and the cost-ordered
    plan spends all of order 1's before the order-2 probe that succeeds. The
    flag is computed from the probes that happened, so it says so — which is
    how we know it is not just echoing the ``minimal=`` argument.
    """
    pool, n, k = nk
    cert = ak.zeilberger(_dixon(pool, n, k), n, k, max_degree=4)
    assert cert.order == 2
    assert cert.order_is_minimal


# ---------------------------------------------------------------------------
# minimal=True establishes it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("summand", ["dixon", "franel"])
def test_minimal_mode_returns_the_known_minimal_order_and_sets_the_flag(nk, summand):
    """Both sums have minimal order 2, and ``minimal=True`` says so.

    ``max_degree`` is 6 rather than the default 16 because minimality is
    claimed *against* that bound and the order-1 sweep up to it is what it
    costs; the claim is what is under test, not the wall clock.
    """
    pool, n, k = nk
    term = _dixon(pool, n, k) if summand == "dixon" else _franel(pool, n, k)
    cert = ak.zeilberger(term, n, k, max_degree=6, minimal=True)
    assert cert.order == 2
    assert cert.order_is_minimal
    assert len(cert.coeffs) == 3


def test_minimal_mode_returns_the_same_verified_certificate(nk):
    """Sharper claim, same relation — not a second, weaker code path.

    The certificate is re-checked as an exact ``Q(n)(k)`` identity either way,
    and the side condition on the *sum* is unchanged, so the only difference
    between the two modes must be what was ruled out along the way.
    """
    pool, n, k = nk
    term = _franel(pool, n, k)
    fast = ak.zeilberger(term, n, k, max_degree=6)
    sharp = ak.zeilberger(term, n, k, max_degree=6, minimal=True)
    assert fast.order == sharp.order
    assert [str(c) for c in fast.coeffs] == [str(c) for c in sharp.coeffs]
    assert str(fast.certificate) == str(sharp.certificate)
    assert fast.side_conditions == sharp.side_conditions


def test_minimal_mode_does_not_change_the_default(nk):
    """The default has to stay fast; that is not negotiable.

    Franel at the shipped defaults was seconds-to-never before the cost-ordered
    plan and is well under a second after it. ``minimal=True`` is an opt-in and
    must not have moved the default even slightly — if this bound is breached,
    the traversal has regressed to the order-major sweep for everyone.
    """
    pool, n, k = nk
    start = time.perf_counter()
    cert = ak.zeilberger(_franel(pool, n, k), n, k)
    elapsed = time.perf_counter() - start
    assert cert.order == 2
    assert elapsed < 5.0, f"the default search took {elapsed:.2f}s; it used to take ~0.1s"


def test_minimality_costs_what_it_claims_to_cost(nk):
    """The trade-off is real and is measured, not asserted away.

    At ``max_degree=8`` the order-1 sweep ``minimal=True`` pays for is several
    times the whole default search. The test asserts only the direction — the
    ratio itself is machine-dependent, and the release notes carry the numbers
    — because a strict bound here would be a flaky test rather than a guard.
    """
    pool, n, k = nk
    term = _franel(pool, n, k)

    start = time.perf_counter()
    fast = ak.zeilberger(term, n, k, max_degree=8)
    fast_seconds = time.perf_counter() - start

    start = time.perf_counter()
    sharp = ak.zeilberger(term, n, k, max_degree=8, minimal=True)
    sharp_seconds = time.perf_counter() - start

    assert not fast.order_is_minimal
    assert sharp.order_is_minimal
    assert sharp_seconds > fast_seconds, (
        "minimal=True is supposed to pay for the low-order sweep the default "
        f"plan skips; got {sharp_seconds:.3f}s vs {fast_seconds:.3f}s"
    )


def test_a_refusal_is_still_a_refusal_in_minimal_mode(nk):
    """Non-hypergeometric input is refused before any traversal happens."""
    _pool, n, k = nk
    with pytest.raises(ak.HolonomicError) as excinfo:
        ak.zeilberger(ak.sin(n * k), n, k, minimal=True)
    assert excinfo.value.code == "E-HOLO-001"


def test_the_flag_is_a_property_not_a_method(nk):
    """The accessor convention: a zero-argument O(1) flag is a property.

    ``if cert.order_is_minimal()`` on a bound method is always truthy, which is
    the failure mode the 3.8.0 sweep existed to remove. Reading it as a value
    has to be what works.
    """
    pool, n, k = nk
    cert = ak.zeilberger(_binomial(pool, n, k), n, k)
    assert cert.order_is_minimal is True
    assert not callable(cert.order_is_minimal)


def test_repr_records_the_minimality_claim(nk):
    """A claim worth making is worth showing in the default rendering."""
    pool, n, k = nk
    assert "[minimal]" in repr(ak.zeilberger(_binomial(pool, n, k), n, k))
    assert "[minimal]" not in repr(ak.zeilberger(_dixon(pool, n, k), n, k))
