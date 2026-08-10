"""P1 item 10 — asymptotic expansion at scale.

`series` and Gruntz `limit` handle expansions of a *function*; a loop working on
combinatorics and analysis needs asymptotics of *sums*. This covers the
Euler–Maclaurin route and, just as importantly, that the result object is honest
about how much of itself is proved.
"""

import pytest
from alkahest import ExprPool
from alkahest.experimental import euler_maclaurin


def _harmonic(n):
    return sum(1.0 / i for i in range(1, n + 1))


def _setup():
    pool = ExprPool()
    return pool, pool.symbol("k"), pool.symbol("n")


# ---------------------------------------------------------------------------
# Expansions it must find
# ---------------------------------------------------------------------------


def test_harmonic_numbers():
    """``H_n ~ log n + γ + 1/(2n) − 1/(12n²) + …``"""
    pool, k, n = _setup()

    r = euler_maclaurin(pool.integer(1) / k, k, 1, n, corrections=2)

    assert r.method == "euler-maclaurin"
    assert len(r.terms) >= 2
    assert "log" in str(r.leading) or "ln" in str(r.leading)
    assert r.max_relative_error < 1e-6


def test_harmonic_expansion_matches_the_true_sum():
    import alkahest as ak

    pool, k, n = _setup()
    r = euler_maclaurin(pool.integer(1) / k, k, 1, n, corrections=2)

    total = r.terms[0]
    for t in r.terms[1:]:
        total = total + t

    for ni in (100, 500, 1000):
        approx = float(ak.evaluate(total, {n: float(ni)}).value)
        assert approx == pytest.approx(_harmonic(ni), abs=1e-6)


def test_polynomial_sum_is_reproduced_exactly():
    """``Σ_{k=1}^{n} k = n(n+1)/2``."""
    import alkahest as ak

    _pool, k, n = _setup()
    r = euler_maclaurin(k, k, 1, n, corrections=1)

    total = r.terms[0]
    for t in r.terms[1:]:
        total = total + t

    for ni in (10, 50, 200):
        approx = float(ak.evaluate(total, {n: float(ni)}).value)
        assert approx == pytest.approx(ni * (ni + 1) / 2, rel=1e-9)


# ---------------------------------------------------------------------------
# Honesty of the report
# ---------------------------------------------------------------------------


def test_report_declares_what_is_assumed_rather_than_proved():
    """The additive constant is fitted, and the report must say so.

    Euler–Maclaurin does not produce γ from the `n`-side terms; it is obtained
    numerically. Presenting that as proved would be exactly the kind of quiet
    overclaim this subsystem exists to avoid.
    """
    pool, k, n = _setup()

    r = euler_maclaurin(pool.integer(1) / k, k, 1, n, corrections=2)

    assert r.rigor == "numerically_consistent"
    assert not r.all_hypotheses_checked
    statuses = {status for status, _ in r.hypotheses}
    assert "assumed" in statuses
    assert any("fitted numerically" in stmt for _, stmt in r.hypotheses)


def test_report_carries_verification_evidence():
    pool, k, n = _setup()

    r = euler_maclaurin(pool.integer(1) / k, k, 1, n, corrections=2)

    assert r.verification
    for at, reference, approximation, rel_err in r.verification:
        assert at > 0
        assert rel_err == pytest.approx(abs(reference - approximation) / abs(reference), rel=1e-9)
    assert r.derivation
    assert "AsymptoticReport(" in repr(r)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_refuses_summand_with_no_symbolic_antiderivative():
    pool, k, n = _setup()
    f = pool.func("exp", [pool.integer(-1) * k * k])

    with pytest.raises(Exception):
        euler_maclaurin(f, k, 1, n, corrections=1)


def test_refuses_absurd_correction_count():
    pool, k, n = _setup()

    with pytest.raises(Exception):
        euler_maclaurin(pool.integer(1) / k, k, 1, n, corrections=99)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_exported_surface():
    import alkahest.experimental as exp

    assert "euler_maclaurin" in exp.__all__
    assert "AsymptoticReport" in exp.__all__
