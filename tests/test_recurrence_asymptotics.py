"""M5 — ``asymptotics_from_recurrence``: growth of a P-recursive sequence.

A certified recurrence already determines how fast the sequence grows, and
after ``zeilberger`` or ``guess_holonomic`` that is always the next question.
These tests check two things, and the second matters more than the first.

1. That the derived quantities are *right*: for every sequence here the
   asymptotic is known independently (Fibonacci, central binomials, Catalan,
   Motzkin, Apéry, and OEIS A359643, whose entry carries
   ``a(n) ~ 283^(n+1/2)/(2^(7/2)·√(πn)·3^(3n+1/2))``), so the growth rate, the
   polynomial exponent and the fitted constant are all compared against the
   truth rather than against each other.

2. That the *fitted* half is never dressed up as the derived half. ``ρ`` and
   ``α`` follow from the recurrence; the connection constant ``C`` does not —
   it depends on the initial conditions. Fibonacci is the control for that:
   there ``C = 1/√5`` is derivable, so a fit that converges to it is evidence
   the machinery is fitting the right thing, and everywhere else the number is
   labelled as fitted whether it is accurate or not.

The hypotheses of Poincaré–Perron are false for plenty of recurrences, and the
last block checks that each failure is *reported* rather than answered with a
confident wrong number.
"""

from __future__ import annotations

import doctest
from fractions import Fraction
from math import comb, exp, log, pi, sqrt

import alkahest as ak
import pytest
from alkahest.experimental import RecurrenceAsymptotics, asymptotics_from_recurrence

# ---------------------------------------------------------------------------
# Sequences and their recurrences, written as `Σ_i p_i(n)·u(n+i) = 0` with each
# `p_i` an ascending tuple of integer coefficients in `n`.
# ---------------------------------------------------------------------------

# F(n+2) − F(n+1) − F(n) = 0.
FIBONACCI = [(-1,), (-1,), (1,)]

# (n+1)·u(n+1) − (4n+2)·u(n) = 0  →  C(2n,n).
CENTRAL_BINOMIAL = [(-2, -4), (1, 1)]

# (n+2)·u(n+1) − (4n+2)·u(n) = 0  →  Catalan.
CATALAN = [(-2, -4), (2, 1)]

# (n+4)·M(n+2) − (2n+5)·M(n+1) − (3n+3)·M(n) = 0.
MOTZKIN = [(-3, -3), (-5, -2), (4, 1)]

# (n+2)³·A(n+2) − (34n³+153n²+231n+117)·A(n+1) + (n+1)³·A(n) = 0  →  A005259.
APERY = [(1, 3, 3, 1), (-117, -231, -153, -34), (8, 12, 6, 1)]

# The order-4 recurrence for A359643, `a(n) = Σ_k C(n,k)·C(4k,k)`.
A359643 = [
    (1698, 3113, 1698, 283),
    (-12978, -16071, -6543, -876),
    (24624, 24705, 8289, 930),
    (-14688, -12833, -3741, -364),
    (1320, 1086, 297, 27),
]


def _n():
    pool = ak.ExprPool()
    return pool.symbol("n")


def _terms(f, count):
    return [f(i) for i in range(count)]


def _rel(a, b):
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# Known asymptotics, checked against the truth
# ---------------------------------------------------------------------------


def test_fibonacci_is_the_control_because_the_constant_is_derivable():
    """``F(n) ~ φⁿ/√5`` — the one case where ``C`` has a closed form.

    Everywhere else the constant is a fitted number nobody can check by eye.
    Here it is exactly ``1/√5``, so a fit that lands on it is evidence the
    extrapolation converges to the right limit and not merely to *a* limit.
    """
    r = asymptotics_from_recurrence(FIBONACCI, _n(), terms=[0, 1])

    assert r.verdict == "single_dominant_root"
    assert _rel(r.growth_rate, (1 + sqrt(5)) / 2) < 1e-12
    assert r.polynomial_exponent == 0.0
    assert r.connection_constant_converged
    assert _rel(r.connection_constant, 1 / sqrt(5)) < 1e-10


def test_central_binomials():
    """``C(2n,n) ~ 4ⁿ/√(πn)`` — both ``ρ`` and ``α`` come out exact."""
    r = asymptotics_from_recurrence(CENTRAL_BINOMIAL, _n(), terms=[1])

    assert str(r.growth_rate_exact) == "4"
    assert str(r.polynomial_exponent_exact) == "-1/2"
    assert _rel(r.connection_constant, 1 / sqrt(pi)) < 1e-8


def test_catalan_shares_the_rate_and_differs_in_the_exponent():
    """``4ⁿ/(√π·n^{3/2})``.

    Same dominant root as the central binomials — the exponent is what tells
    them apart, and it comes from the *subleading* coefficients of the
    recurrence, not from the characteristic polynomial.
    """
    r = asymptotics_from_recurrence(CATALAN, _n(), terms=[1])

    assert str(r.growth_rate_exact) == "4"
    assert str(r.polynomial_exponent_exact) == "-3/2"
    assert _rel(r.connection_constant, 1 / sqrt(pi)) < 1e-6


def test_motzkin():
    """``M(n) ~ 3ⁿ·3√3/(2√π·n^{3/2})``."""
    r = asymptotics_from_recurrence(MOTZKIN, _n(), terms=[1, 1])

    assert str(r.growth_rate_exact) == "3"
    assert str(r.polynomial_exponent_exact) == "-3/2"
    assert _rel(r.connection_constant, 3 * sqrt(3) / (2 * sqrt(pi))) < 1e-6


def test_apery_numbers():
    """A005259: ``ρ = (1+√2)⁴``, ``α = −3/2``, ``C = (1+√2)²/(2^{9/4}π^{3/2})``.

    The rate is irrational, so this is the case with no ``growth_rate_exact``
    and a fit that runs entirely off a numerically located root.
    """
    apery = [sum(comb(n, k) ** 2 * comb(n + k, k) ** 2 for k in range(n + 1)) for n in range(2)]
    r = asymptotics_from_recurrence(APERY, _n(), terms=apery)

    assert _rel(r.growth_rate, (1 + sqrt(2)) ** 4) < 1e-12
    assert r.growth_rate_exact is None
    assert _rel(r.polynomial_exponent, -1.5) < 1e-10
    truth = (1 + sqrt(2)) ** 2 / (2**2.25 * pi**1.5)
    assert _rel(r.connection_constant, truth) < 1e-7


def test_a359643_reproduces_its_oeis_asymptotic():
    """``a(n) ~ 283^(n+1/2) / (2^(7/2)·√(πn)·3^(3n+1/2))``.

    That is ``ρ = 283/27``, ``α = −1/2`` and ``C = √(283/3)/(2^{7/2}√π)``.
    The characteristic polynomial is ``(t−1)³·(27t−283)``: the triple root is
    real and well away from the dominant one, which is why multiplicity has to
    be exact rather than a clustering tolerance.
    """
    terms = _terms(lambda n: sum(comb(n, k) * comb(4 * k, k) for k in range(n + 1)), 4)
    r = asymptotics_from_recurrence(A359643, _n(), terms=terms)

    assert r.verdict == "single_dominant_root"
    assert str(r.growth_rate_exact) == "283/27"
    assert str(r.polynomial_exponent_exact) == "-1/2"
    # `roots()` lists each *distinct* root once, with its exact multiplicity.
    assert sorted(m for _, _, _, m in r.roots()) == [1, 3]

    truth = sqrt(283 / 3) / (2**3.5 * sqrt(pi))
    assert _rel(r.connection_constant, truth) < 1e-8
    assert r.leading_term is not None


def test_the_leading_term_tracks_the_real_terms_at_large_n():
    """The emitted expression, evaluated, against the sequence it describes."""
    terms = _terms(lambda n: sum(comb(n, k) * comb(4 * k, k) for k in range(n + 1)), 4)
    n = _n()
    r = asymptotics_from_recurrence(A359643, n, terms=terms)

    exact = _terms(lambda m: sum(comb(m, k) * comb(4 * k, k) for k in range(m + 1)), 401)
    for index in (200, 400):
        # `a(400)` has 409 digits, so the comparison runs in log space: the
        # claim is `ln a(N) − N·ln ρ − α·ln N → ln C`.
        residual = (
            log(exact[index]) - index * log(r.growth_rate) - r.polynomial_exponent * log(index)
        )
        assert _rel(exp(residual), r.connection_constant) < 1e-2


# ---------------------------------------------------------------------------
# Proved versus fitted
# ---------------------------------------------------------------------------


def test_the_constant_is_reported_as_fitted_and_the_shape_as_derived():
    r = asymptotics_from_recurrence(CENTRAL_BINOMIAL, _n(), terms=[1])
    report = r.report()

    assert report.method == "poincare-perron"
    assert report.rigor == "numerically_consistent"
    assert not report.all_hypotheses_checked

    fitted = [h for s, h in report.hypotheses if s == "assumed" and "fitted numerically" in h]
    derived = [h for s, h in report.hypotheses if s == "checked" and "was fitted" in h]
    assert fitted, "the fitted constant must be declared as assumed"
    assert derived, "ρ and α must be declared as derived"


def test_evidence_separates_the_two_halves():
    r = asymptotics_from_recurrence(MOTZKIN, _n(), terms=[1, 1])
    evidence = r.evidence()

    assert set(evidence["derived"]) == {
        "order",
        "verdict",
        "growth_rate",
        "polynomial_exponent",
        "roots",
        "singular_indices",
    }
    assert set(evidence["fitted"]) == {
        "connection_constant",
        "converged",
        "relative_drift",
        "fitted_at",
        "refit_at",
    }
    # The constant appears only under "fitted"; nothing under "derived" is it.
    assert evidence["fitted"]["connection_constant"] == r.connection_constant
    assert "connection_constant" not in evidence["derived"]


def test_without_terms_the_shape_still_comes_out_and_the_rest_is_assumed():
    """No terms, no constant — but ``ρ`` and ``α`` never needed them."""
    r = asymptotics_from_recurrence(CENTRAL_BINOMIAL, _n())

    assert str(r.growth_rate_exact) == "4"
    assert str(r.polynomial_exponent_exact) == "-1/2"
    assert r.connection_constant is None
    assert r.follows_dominant_root is None
    assert r.leading_term is None
    assert any(s == "assumed" and "tends to *some* root" in h for s, h in r.report().hypotheses)


def test_the_gate_scores_the_constant_at_indices_the_fit_never_saw():
    r = asymptotics_from_recurrence(CATALAN, _n(), terms=[1])
    verification = r.report().verification

    assert verification, "a fitted constant must be corroborated"
    fitted_at = {128, 256, 512, 1024}
    assert not fitted_at.intersection(at for at, *_ in verification)
    # The residual has to *decay*: an asymptotic claim that stops improving is
    # not one.
    relatives = [rel for *_, rel in verification]
    assert relatives[-1] < relatives[0]


# ---------------------------------------------------------------------------
# The hypotheses failing, reported rather than papered over
# ---------------------------------------------------------------------------


def test_equal_modulus_roots_are_reported_not_answered():
    """``u(n+2) = 4·u(n)`` has roots ``±2``; its solutions oscillate.

    Answering ``ρ = 2`` here would be a wrong answer with a confident face on
    it — the class of overclaim this whole result object exists to prevent.
    """
    r = asymptotics_from_recurrence([(-4,), (0,), (1,)], _n(), terms=[1, 2])

    assert r.verdict == "equal_modulus_roots"
    assert r.growth_rate is None
    assert r.polynomial_exponent is None
    assert r.leading_term is None
    assert r.connection_constant is None
    assert "oscillating" in r.verdict_reason
    assert sorted(round(re, 9) for re, _, _, _ in r.roots()) == [-2.0, 2.0]


def test_a_complex_conjugate_pair_is_the_same_failure():
    """``u(n+2) = −u(n)`` has roots ``±i``: equal modulus, period four."""
    r = asymptotics_from_recurrence([(1,), (0,), (1,)], _n(), terms=[1, 1])

    assert r.verdict == "equal_modulus_roots"
    assert r.growth_rate is None


def test_a_repeated_dominant_root_is_reported():
    """``χ = (t−2)²``: the exponent formula would divide by ``χ'(ρ) = 0``."""
    r = asymptotics_from_recurrence([(4,), (-4,), (1,)], _n(), terms=[1, 2])

    assert r.verdict == "repeated_dominant_root"
    assert r.growth_rate is None
    assert [m for _, _, _, m in r.roots()] == [2]


def test_a_degenerate_leading_coefficient_is_reported():
    """``u(n+2) = n·u(n+1)`` grows like ``n!`` — outside Poincaré's theorem."""
    r = asymptotics_from_recurrence([(0,), (0, -1), (1,)], _n(), terms=[1, 1])

    assert r.verdict == "degenerate_leading_coefficient"
    assert r.growth_rate is None
    assert "Birkhoff" in r.verdict_reason


def test_a_leading_coefficient_vanishing_finitely_often_is_a_side_condition():
    """``(n−7)·u(n+1) = 4(n−7)·u(n)``: one bad index, not a bad theorem."""
    r = asymptotics_from_recurrence([(28, -4), (-7, 1)], _n(), terms=[1])

    assert r.singular_indices() == [7]
    assert r.singular_indices_complete
    assert str(r.growth_rate_exact) == "4"
    assert any("n > 7" in h for _, h in r.report().hypotheses)
    # The forward run stops at n = 7, so no constant can be fitted — and that is
    # a different finding from "the sequence does not follow the dominant root".
    assert r.connection_constant is None
    assert r.follows_dominant_root is None


def test_an_eventually_zero_sequence_has_no_growth_rate():
    r = asymptotics_from_recurrence(FIBONACCI, _n(), terms=[0, 0])

    assert r.verdict == "eventually_zero"
    assert r.growth_rate is None
    assert r.leading_term is None


def test_a_sequence_that_does_not_follow_the_dominant_root_is_caught():
    """``u(n+2) = 3u(n+1) − 2u(n)`` with ``u(0) = u(1) = 1`` is constant.

    Poincaré's conclusion is that ``u(n+1)/u(n)`` tends to *some* root. The
    dominant one here is ``2`` and this solution's component along it is zero,
    which is exactly the case where a naive implementation reports exponential
    growth for a constant sequence.
    """
    r = asymptotics_from_recurrence([(2,), (-3,), (1,)], _n(), terms=[1, 1])

    assert str(r.growth_rate_exact) == "2"
    assert r.follows_dominant_root is False
    assert r.connection_constant is None
    assert r.leading_term is None


# ---------------------------------------------------------------------------
# Composition with the rest of the holonomic subsystem
# ---------------------------------------------------------------------------


def test_it_composes_with_guess_holonomic():
    """Guess the recurrence from terms, then ask it how fast the sequence grows.

    This is the loop the capability exists for; before it, the holonomic and
    asymptotics halves of the library did not compose at all.
    """
    motzkin = [
        1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188,
        5798, 15511, 41835, 113634, 310572, 853467,
        2356779, 6536382, 18199284, 50852019,
    ]  # fmt: skip
    guess = ak.guess_holonomic(motzkin)
    assert guess.confirmed

    r = asymptotics_from_recurrence(guess, _n(), terms=motzkin[:2])
    assert str(r.growth_rate_exact) == "3"
    assert _rel(r.connection_constant, 3 * sqrt(3) / (2 * sqrt(pi))) < 1e-6


def test_it_composes_with_zeilberger():
    """A certified recurrence, then its growth rate — ``Σ_k C(n,k) = 2ⁿ``."""
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    binomial = ak.gamma(n + one) / (ak.gamma(k + one) * ak.gamma(n - k + one))
    cert = ak.zeilberger(binomial, n, k)
    assert cert.boundary == "vanishes"

    r = asymptotics_from_recurrence(cert.coeffs, n, terms=[1, 2])
    assert str(r.growth_rate_exact) == "2"
    assert r.polynomial_exponent == 0.0
    assert _rel(r.connection_constant, 1.0) < 1e-9


def test_start_defaults_to_the_guesss_own_start():
    """``start=3`` fits a *shifted* sequence, and the constant shifts with it.

    ``guess_holonomic(motzkin, start=3)`` fits ``u`` with ``u(3+j) = M(j)``, so
    ``u(n) = M(n−3) ~ 3ⁿ·(C_M/27)·n^{−3/2}``. The rate and the exponent are
    unchanged, and the connection constant is divided by ``3³`` — which is the
    check that ``start`` was honoured rather than silently taken as ``0``.
    """
    motzkin = [
        1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188,
        5798, 15511, 41835, 113634, 310572, 853467,
        2356779, 6536382, 18199284, 50852019,
    ]  # fmt: skip
    guess = ak.guess_holonomic(motzkin, start=3)
    assert guess.start == 3

    r = asymptotics_from_recurrence(guess, _n(), terms=motzkin[:2])
    assert str(r.growth_rate_exact) == "3"
    assert str(r.polynomial_exponent_exact) == "-3/2"
    shifted = 3 * sqrt(3) / (2 * sqrt(pi)) / 27
    assert _rel(r.connection_constant, shifted) < 1e-6


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_float_terms_are_refused_not_rounded():
    """A growth law fitted to rounded terms describes a different sequence."""
    with pytest.raises(TypeError, match="exact rational"):
        asymptotics_from_recurrence(CENTRAL_BINOMIAL, _n(), terms=[1.0])


def test_float_coefficients_are_refused_not_rounded():
    """The characteristic polynomial is exact arithmetic all the way down."""
    with pytest.raises(TypeError, match="float"):
        asymptotics_from_recurrence([(-2.0, -4.0), (1, 1)], _n(), terms=[1])


def test_a_string_is_not_a_coefficient_polynomial():
    """``str`` iterates as characters, so it must be rejected on shape."""
    with pytest.raises(TypeError, match="must be an alkahest Expr"):
        asymptotics_from_recurrence(["12", (1, 1)], _n(), terms=[1])


def test_exact_fractions_are_accepted():
    """``u(n+1) = u(n)/2`` from ``u(0) = 1/3``: rational terms, exact all the way."""
    r = asymptotics_from_recurrence([(-1,), (2,)], _n(), terms=[Fraction(1, 3)])

    assert str(r.growth_rate_exact) == "1/2"
    assert _rel(r.connection_constant, 1 / 3) < 1e-9


def test_expression_coefficients_work_too():
    pool = ak.ExprPool()
    n = pool.symbol("n")
    r = asymptotics_from_recurrence([-(4 * n) - 2, n + 1], n, terms=[1])

    assert str(r.growth_rate_exact) == "4"
    assert str(r.polynomial_exponent_exact) == "-1/2"


def test_a_non_polynomial_coefficient_is_refused():
    pool = ak.ExprPool()
    n = pool.symbol("n")
    with pytest.raises(ValueError, match="asymptotic scale"):
        asymptotics_from_recurrence([ak.exp(n), pool.integer(1)], n)


def test_an_order_zero_recurrence_is_refused():
    with pytest.raises(ValueError):
        asymptotics_from_recurrence([(1,)], _n())


def test_big_integer_coefficients_stay_exact():
    """A coefficient past 2⁵³ must not become a float on the way in.

    ``c * n**j`` in Python would silently do that; the binding routes every
    coefficient through the same big-integer path ``pool.integer`` uses.
    """
    big = 2**80
    r = asymptotics_from_recurrence([(-big,), (0,), (1,)], _n(), terms=[1, 1])
    # χ(t) = t² − 2⁸⁰ has roots ±2⁴⁰: equal modulus, and only exact arithmetic
    # gets both of them.
    assert r.verdict == "equal_modulus_roots"
    assert max(mod for _, _, mod, _ in r.roots()) == pytest.approx(2**40)


def test_repr_does_not_leak_rust_option_syntax():
    r = asymptotics_from_recurrence(CENTRAL_BINOMIAL, _n(), terms=[1])
    text = repr(r)
    assert "Some(" not in text
    assert "single_dominant_root" in text
    assert "fitted" in text


# ---------------------------------------------------------------------------
# The index symbol: derived rather than demanded
# ---------------------------------------------------------------------------


def test_n_is_optional_for_a_certificate_that_carries_its_own():
    """``asymptotics_from_recurrence(cert)`` — no pool bookkeeping at all.

    A certificate's coefficients live in the certificate's pool and can be
    combined with nothing else, so requiring the caller to supply a matching
    ``n`` was requiring them to reconstruct something ``cert`` already had.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    binomial = ak.gamma(n + one) / (ak.gamma(k + one) * ak.gamma(n - k + one))
    cert = ak.zeilberger(binomial, n, k)

    assert cert.n == n
    r = asymptotics_from_recurrence(cert, terms=[1, 2])
    assert str(r.growth_rate_exact) == "2"
    assert r.polynomial_exponent == 0.0


def test_n_is_optional_when_the_coefficients_are_plain_integers():
    """A ``GuessedRecurrence`` belongs to no pool, so one is made for it."""
    motzkin = [
        1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188,
        5798, 15511, 41835, 113634, 310572, 853467,
        2356779, 6536382, 18199284, 50852019,
    ]  # fmt: skip
    r = asymptotics_from_recurrence(ak.guess_holonomic(motzkin), terms=motzkin[:2])
    assert str(r.growth_rate_exact) == "3"

    raw = asymptotics_from_recurrence(CENTRAL_BINOMIAL, terms=[1])
    assert raw.growth_rate == 4.0


def test_a_foreign_n_is_a_coded_error_naming_the_argument():
    """The bare pool mismatch said nothing about which argument was wrong.

    It arrived from several frames inside the coefficient walk as an
    uncoded ``PoolError``, with no hint that ``rec`` was carrying the right
    symbol all along.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    binomial = ak.gamma(n + one) / (ak.gamma(k + one) * ak.gamma(n - k + one))
    cert = ak.zeilberger(binomial, n, k)

    with pytest.raises(ak.PoolError) as excinfo:
        asymptotics_from_recurrence(cert, _n(), terms=[1, 2])
    assert excinfo.value.code == "E-POOL-001"
    assert "different ExprPool" in str(excinfo.value)
    assert "rec.n" in excinfo.value.remediation


def test_omitting_n_on_raw_expressions_says_why_it_cannot_be_derived():
    """A bare list of ``Expr`` cannot name its own index variable."""
    pool = ak.ExprPool()
    n = pool.symbol("n")
    with pytest.raises(ak.PoolError) as excinfo:
        asymptotics_from_recurrence([n * pool.integer(-2), n], terms=[1])
    assert excinfo.value.code == "E-POOL-001"
    assert "n must be given" in str(excinfo.value)


def test_docstring_examples():
    import alkahest._recurrence_asymptotics as module

    failures, _ = doctest.testmod(module, verbose=False)
    assert failures == 0


def test_the_class_is_exported_from_experimental():
    from alkahest import experimental

    assert "asymptotics_from_recurrence" in experimental.__all__
    assert "RecurrenceAsymptotics" in experimental.__all__
    assert RecurrenceAsymptotics is experimental.RecurrenceAsymptotics
