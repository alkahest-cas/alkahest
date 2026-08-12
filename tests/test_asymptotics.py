"""P1 item 10 — asymptotic expansion at scale.

`series` and Gruntz `limit` handle expansions of a *function*; a loop working on
combinatorics and analysis needs asymptotics of *sums*. This covers the
Euler–Maclaurin route and, just as importantly, that the result object is honest
about how much of itself is proved.
"""

import pytest
from alkahest import ExprPool
from alkahest.experimental import coefficient_asymptotics, euler_maclaurin


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
# Singularity analysis of rational generating functions
# ---------------------------------------------------------------------------


def test_fibonacci_growth_from_its_generating_function():
    """``1/(1 - z - z²)`` generates the Fibonacci numbers; ``[zⁿ] ~ φⁿ/√5``."""
    import alkahest as ak

    pool = ak.ExprPool()
    z, n = pool.symbol("z"), pool.symbol("n")
    gf = pool.integer(1) / (pool.integer(1) - z - z * z)

    r = coefficient_asymptotics(gf, z, n)

    assert r.method == "singularity-analysis"
    # [z^n] of this series is F_{n+1}.
    fibs = [0, 1]
    while len(fibs) < 45:
        fibs.append(fibs[-1] + fibs[-2])
    approx = float(ak.evaluate(r.terms[0], {n: 40.0}).value)
    assert approx == pytest.approx(float(fibs[41]), rel=1e-6)


def test_leading_law_error_shrinks_with_n():
    """``1/(1-z)²`` has ``[zⁿ] = n+1``; a leading-order ``C·n`` must converge.

    Fitting the constant at one finite index would absorb the subleading term
    and leave a fixed few-percent bias — an error that stops shrinking is not
    an asymptotic statement at all.
    """
    import alkahest as ak

    pool = ak.ExprPool()
    z, n = pool.symbol("z"), pool.symbol("n")
    gf = pool.integer(1) / ((pool.integer(1) - z) * (pool.integer(1) - z))

    r = coefficient_asymptotics(gf, z, n)

    def rel(ni):
        approx = float(ak.evaluate(r.terms[0], {n: float(ni)}).value)
        return abs(approx - (ni + 1)) / (ni + 1)

    assert rel(100) < 0.05
    assert rel(1000) < rel(100) / 2


def test_refuses_competing_dominant_singularities():
    """``1/(1-z²)`` has poles at ±1: the coefficients oscillate, so there is no
    single power-law term and the routine must decline rather than pick one."""
    import alkahest as ak

    pool = ak.ExprPool()
    z, n = pool.symbol("z"), pool.symbol("n")
    gf = pool.integer(1) / (pool.integer(1) - z * z)

    with pytest.raises(Exception):
        coefficient_asymptotics(gf, z, n)


def test_refuses_non_rational_generating_function():
    import alkahest as ak

    pool = ak.ExprPool()
    z, n = pool.symbol("z"), pool.symbol("n")

    with pytest.raises(Exception):
        coefficient_asymptotics(ak.exp(z), z, n)


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
    assert "coefficient_asymptotics" in exp.__all__
    assert "AsymptoticReport" in exp.__all__
