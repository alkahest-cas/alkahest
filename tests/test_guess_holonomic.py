"""M2 — ``guess_holonomic``: fit a P-recursive recurrence to finite data.

The point of these tests is not that the linear algebra works. A homogeneous
system with more unknowns than equations *always* has a nontrivial solution, so
an unguarded fitter returns a recurrence for every input ever handed to it and
is worth nothing. What is being tested is the guard: that a fit is only
returned when the terms over-determined it, that a sequence with no such
relation gets ``None`` rather than a confident answer, and that "you did not
give me enough terms" comes back as a refusal instead of being laundered into
"there is no relation".
"""

import decimal
import doctest
import math
from fractions import Fraction

import alkahest as ak
import pytest

# The first 21 Motzkin numbers (OEIS A001006). The recurrence they satisfy,
# (n+4)·M(n+2) = (2n+5)·M(n+1) + (3n+3)·M(n), is order 2 with degree-1
# coefficients: six unknowns, which 21 terms over-determine three times over.
MOTZKIN = [
    1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188,
    5798, 15511, 41835, 113634, 310572, 853467,
    2356779, 6536382, 18199284, 50852019,
]  # fmt: skip

# The first 60 primes: famously not P-recursive, and the standard negative
# control for a recurrence guesser.
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
]  # fmt: skip


def _beatty(count):
    """``floor(n·φ)`` — a lower Wythoff sequence, computed exactly."""
    decimal.getcontext().prec = 120
    phi = (1 + decimal.Decimal(5).sqrt()) / 2
    return [int(decimal.Decimal(n) * phi) for n in range(count)]


# ---------------------------------------------------------------------------
# It recovers a known recurrence
# ---------------------------------------------------------------------------


def test_motzkin_recurrence_is_recovered_from_twenty_one_terms():
    """The headline case: 21 terms, and the answer is Motzkin's recurrence.

    Checked against the closed form rather than against a stored coefficient
    vector, so the test still means something if the normalisation changes.
    """
    guess = ak.guess_holonomic(MOTZKIN)
    assert guess is not None
    assert guess.order == 2
    assert guess.degree == 1
    assert guess.confirmed

    # (3n+3)·M(n) + (2n+5)·M(n+1) − (n+4)·M(n+2) = 0, up to the overall sign
    # and scale a nullspace vector is defined to within.
    p0, p1, p2 = guess.coeffs
    assert p0 == (-3, -3)
    assert p1 == (-5, -2)
    assert p2 == (4, 1)


def test_the_recovered_recurrence_is_exact_on_the_terms_it_was_not_shown():
    """Fitted on a prefix, confirmed on the rest — the loop this exists for."""
    guess = ak.guess_holonomic(MOTZKIN[:16])
    assert guess is not None
    assert guess.confirmed
    assert guess.holds_for(MOTZKIN), "a fit on 16 terms must survive all 21"


def test_surplus_is_reported_and_is_the_number_that_justifies_the_fit():
    """``surplus_terms`` is evidence, so it has to be right, not decorative.

    Six unknowns pinned down by five independent equations out of the nineteen
    the terms provide leaves fourteen that agreed without being asked to.
    """
    guess = ak.guess_holonomic(MOTZKIN)
    assert guess.n_terms == 21
    assert guess.n_equations == 19
    assert guess.equations_used == 5
    assert guess.surplus_terms == 14
    assert guess.dimension == 1
    assert guess.evidence() == {
        "n_terms": 21,
        "n_equations": 19,
        "equations_used": 5,
        "surplus_terms": 14,
        "min_surplus": 6,
        "dimension": 1,
        "singular_indices": [],
        "untested_candidates": 0,
        "status": "confirmed",
        "means": ak.GUESS_STATUS_MEANINGS["confirmed"],
        "confirmed": True,
    }


@pytest.mark.parametrize(
    ("name", "terms", "order", "degree"),
    [
        # F(n+2) = F(n+1) + F(n): order 2, constant coefficients.
        ("fibonacci", [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233,
                       377, 610, 987, 1597, 2584, 4181, 6765], 2, 0),
        # (n+2)·C(n+1) = (4n+2)·C(n): order 1, degree 1.
        ("catalan", [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786,
                     208012, 742900, 2674440, 9694845, 35357670], 1, 1),
    ],
)  # fmt: skip
def test_other_classical_sequences(name, terms, order, degree):
    guess = ak.guess_holonomic(terms)
    assert guess is not None, name
    assert (guess.order, guess.degree) == (order, degree), name
    assert guess.confirmed, name
    assert guess.holds_for(terms), name


def test_fraction_terms_are_accepted_exactly():
    """Harmonic numbers are holonomic and are not integers.

    ``(n+1)·H(n) − (2n+3)·H(n+1) + (n+2)·H(n+2) = 0``. Exact rational input has
    to work, because the sequences a research loop produces are as often ratios
    as integers — and rounding them to floats would fit a different sequence.
    """
    harmonic = [Fraction(0)]
    for i in range(1, 25):
        harmonic.append(harmonic[-1] + Fraction(1, i))

    guess = ak.guess_holonomic(harmonic)
    assert guess is not None
    assert (guess.order, guess.degree) == (2, 1)
    assert guess.coeffs == ((1, 1), (-3, -2), (2, 1))
    assert guess.confirmed


def test_big_integers_are_handled_exactly():
    """``(2n)!`` grows past every machine word; nothing here may overflow."""
    factorial2n = [1]
    for n in range(1, 30):
        factorial2n.append(factorial2n[-1] * (2 * n - 1) * (2 * n))
    assert factorial2n[-1] > 2**128

    guess = ak.guess_holonomic(factorial2n, max_order=2, max_degree=2)
    assert guess is not None
    assert guess.order == 1
    assert guess.confirmed
    assert guess.holds_for(factorial2n)


# ---------------------------------------------------------------------------
# …and refuses the cases where a fit would be interpolation
# ---------------------------------------------------------------------------


def test_just_enough_terms_are_refused_rather_than_fitted():
    """Seven Motzkin terms *determine* the recurrence and confirm nothing.

    Six unknowns need five equations; seven terms supply exactly five. The fit
    would be exact and completely uninformative — the same fit exists for any
    seven numbers. A guesser that returns it has told the caller nothing while
    sounding certain, so this must refuse.
    """
    with pytest.raises(ak.HolonomicError) as excinfo:
        ak.guess_holonomic(MOTZKIN[:7])
    assert excinfo.value.code == "E-HOLO-005"
    assert "not enough" in str(excinfo.value)


def test_the_refusal_says_undecided_and_not_negative():
    """Too few terms must never be reported as "no relation exists".

    This is the failure that matters in an unattended loop: a negative result
    closes a branch, and a branch closed on data that was never able to answer
    is a wrong answer with no symptom. So the short call raises while the same
    sequence with enough terms succeeds.
    """
    with pytest.raises(ak.HolonomicError):
        ak.guess_holonomic(MOTZKIN[:10])
    assert ak.guess_holonomic(MOTZKIN) is not None


def test_an_unconfirmed_fit_is_available_but_only_on_request():
    """``check_evidence=False`` returns the fit *and* says it is worthless.

    The escape hatch has to exist — sometimes the caller wants the candidate to
    feed somewhere else — but it must not lie about what it is: the same seven
    terms that are refused above come back with ``confirmed`` false.
    """
    guess = ak.guess_holonomic(MOTZKIN[:7], check_evidence=False)
    assert guess is not None
    assert not guess.confirmed
    assert guess.surplus_terms < guess.min_surplus


def test_primes_do_not_produce_a_confident_fit():
    """The negative control: primes are not P-recursive, and 60 of them say so.

    ``None`` rather than a refusal is the point — with 60 terms every
    ``(order, degree)`` candidate in bounds is over-determined and was actually
    tested, so this is a genuine negative and a loop may record it as one.
    """
    assert ak.guess_holonomic(PRIMES) is None


def test_beatty_sequence_does_not_produce_a_confident_fit():
    """``floor(n·φ)`` is not P-recursive either, and is the harder control.

    It is *almost* linear, so a fitter with sloppy tolerances finds a spurious
    low-order relation in it. Exact arithmetic and the surplus guard have to
    give ``None``.
    """
    assert ak.guess_holonomic(_beatty(60)) is None


def test_a_negative_is_never_returned_for_a_grid_that_was_not_swept():
    """40 primes cannot test the whole default grid, so ``None`` is not allowed.

    The same input that yields an honest ``None`` at 60 terms must refuse at 40
    rather than quietly answer the smaller question it was able to ask.
    """
    with pytest.raises(ak.HolonomicError) as excinfo:
        ak.guess_holonomic(PRIMES[:40])
    assert excinfo.value.code == "E-HOLO-005"
    assert excinfo.value.remediation


def test_floats_are_refused_rather_than_rounded():
    """Every step downstream is exact; a float would make that exactness a lie."""
    with pytest.raises(TypeError, match="exact rational"):
        ak.guess_holonomic([float(x) for x in MOTZKIN])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_order": 0}, "max_order"),
        ({"max_degree": -1}, "max_degree"),
        ({"min_surplus": -1}, "min_surplus"),
    ],
)
def test_bounds_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ak.guess_holonomic(MOTZKIN, **kwargs)


# ---------------------------------------------------------------------------
# Shape of the result object
# ---------------------------------------------------------------------------


def test_min_surplus_override_is_honoured():
    """The threshold is a knob, and lowering it widens what can be tested.

    Ten Motzkin terms are refused at the default (which demands as many surplus
    equations as the ansatz has unknowns) and accepted at ``min_surplus=2``,
    with the surplus reported either way so the caller can see what they bought.
    """
    with pytest.raises(ak.HolonomicError):
        ak.guess_holonomic(MOTZKIN[:10])

    guess = ak.guess_holonomic(MOTZKIN[:10], min_surplus=2)
    assert guess is not None
    assert guess.order == 2
    assert guess.confirmed
    assert guess.min_surplus == 2
    assert guess.surplus_terms >= 2
    assert guess.holds_for(MOTZKIN), "still the right recurrence, just less evidence"


def test_start_offsets_the_index_the_polynomials_are_written_in():
    """``start`` shifts ``n``, and the coefficients change accordingly.

    Motzkin indexed from 1 satisfies ``(n+3)·M(n+2) = (2n+3)·M(n+1) + 3n·M(n)``
    — the same relation with ``n → n−1``. If ``start`` were ignored this test
    would see the ``start=0`` coefficients.
    """
    shifted = ak.guess_holonomic(MOTZKIN, start=1)
    assert shifted is not None
    assert shifted.start == 1
    assert shifted.coeffs == ((0, -3), (-3, -2), (3, 1))
    assert shifted.holds_for(MOTZKIN)


def test_to_exprs_round_trips_into_the_expression_layer():
    """A guess has to be usable by the rest of the library to be worth having."""
    guess = ak.guess_holonomic(MOTZKIN)
    pool = ak.ExprPool()
    n = pool.symbol("n")
    polys = guess.to_exprs(pool, n)
    assert len(polys) == guess.order + 1
    for index in (0, 3, 7):
        values = [float(ak.evaluate(p, {n: float(index)}).value) for p in polys]
        total = sum(v * MOTZKIN[index + i] for i, v in enumerate(values))
        assert total == pytest.approx(0.0, abs=1e-6)


def test_holds_for_can_fail():
    """A checker that cannot say no is not a checker."""
    guess = ak.guess_holonomic(MOTZKIN)
    assert guess.holds_for(MOTZKIN)
    assert not guess.holds_for([*MOTZKIN[:-1], MOTZKIN[-1] + 1])


def test_repr_shows_the_evidence_not_just_the_answer():
    text = repr(ak.guess_holonomic(MOTZKIN))
    assert "order=2" in text
    assert "surplus_terms=14" in text
    assert "confirmed=True" in text


def test_untested_candidates_qualifies_the_minimality_of_the_order():
    """A skipped candidate is not a refuted one, and the result has to say which.

    At the default threshold 21 Motzkin terms test every candidate below the
    answer, so ``untested_candidates`` is 0 and the order really is the
    smallest in bounds. Ten terms with ``min_surplus=2`` reach the same
    recurrence having skipped some — the order is then only minimal among the
    candidates the data could decide, and the count says so.
    """
    assert ak.guess_holonomic(MOTZKIN).untested_candidates == 0

    thin = ak.guess_holonomic(MOTZKIN[:10], min_surplus=2)
    assert thin.order == 2
    assert thin.untested_candidates > 0


def test_the_new_names_are_on_the_documented_surface():
    """``__all__`` is the public API; a name reachable but unlisted is neither."""
    assert "guess_holonomic" in ak.__all__
    assert "GuessedRecurrence" in ak.__all__
    assert isinstance(ak.guess_holonomic(MOTZKIN), ak.GuessedRecurrence)


def test_the_refusal_is_catchable_as_the_documented_exception():
    """``ak.HolonomicError`` is the *native* class; the refusal must subclass it.

    Subclassing the pure-Python shim in ``exceptions.py`` instead would work in
    every hand-check and slip through every ``except ak.HolonomicError`` a
    caller writes, because ``__init__`` overlays the native classes over the
    module namespace.
    """
    with pytest.raises(ak.HolonomicError):
        ak.guess_holonomic(MOTZKIN[:7])
    with pytest.raises(ak.AlkahestError):
        ak.guess_holonomic(MOTZKIN[:7])


def test_module_docstrings_have_runnable_doctests():
    """The examples in the docstrings are executed, so they cannot rot."""
    from alkahest import _guess_holonomic

    failures, _tests = doctest.testmod(
        _guess_holonomic,
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL,
    )
    assert failures == 0


# ---------------------------------------------------------------------------
# Corrupted data: the fit that holds on the terms and is not the recurrence
# ---------------------------------------------------------------------------


def _at(poly, index):
    """``p(index)`` for an ascending integer coefficient tuple."""
    return sum(c * index**j for j, c in enumerate(poly))


def _motzkin(count):
    """The first *count* Motzkin numbers, from their own recurrence."""
    terms = [1, 1]
    while len(terms) < count:
        i = len(terms) - 1
        terms.append(((2 * i + 3) * terms[-1] + 3 * i * terms[-2]) // (i + 3))
    return terms


def test_a_clean_sequence_has_no_singular_index_and_stays_confirmed():
    """The control the corrupted cases are read against.

    Motzkin's own leading coefficient is ``p_J(n) = n + 4``, whose only root is
    ``−4`` — outside the fitted range, so nothing is reported and the verdict
    is unchanged by any of this.
    """
    guess = ak.guess_holonomic(_motzkin(71))
    assert guess.coeffs[-1] == (4, 1), "p_J(n) = n + 4"
    assert guess.singular_indices == ()
    assert guess.status == "confirmed"
    assert guess.confirmed is True


def test_one_corrupted_term_is_reported_as_singular_and_never_confirmed():
    """A single typo is absorbed into three roots of the leading coefficient.

    At the default ``max_degree`` the fit multiplies the true operator by the
    cubic vanishing at exactly the three indices whose equations the typo
    breaks. Everything a caller reads to judge a fit looked perfect —
    ``dimension`` 1, 55 surplus equations, no untested candidates — and the
    relation really does hold on the terms supplied. It is simply not Motzkin's
    recurrence, and the roots inside the data are the only tell.
    """
    spoiled = _motzkin(71)
    spoiled[30] += 1

    guess = ak.guess_holonomic(spoiled)
    assert guess is not None
    assert guess.singular_indices == (28, 29, 30)
    assert guess.status == "singular"
    assert guess.confirmed is None, "never a bare True on corrupted data"
    # The evidence that used to be the whole story is still exactly as strong.
    assert guess.dimension == 1
    assert guess.surplus_terms == 55
    assert guess.untested_candidates == 0
    assert guess.holds_for(spoiled), "the relation does hold on what it was shown"
    # It is the true operator multiplied by that cubic, so it also holds on the
    # *clean* sequence — which is precisely why no re-check can catch this and
    # why the roots have to be reported. What it does not do is determine the
    # sequence at 28, 29 and 30, where every coefficient vanishes at once.
    assert guess.holds_for(_motzkin(71))
    for index in guess.singular_indices:
        assert all(_at(poly, index) == 0 for poly in guess.coeffs), (
            f"every coefficient vanishes at n = {index}, so the equation there "
            "is 0 = 0 and constrained nothing"
        )


def test_two_corrupted_terms_are_reported_as_six_roots():
    """Two typos need ``max_degree=8``, and produce two triples of roots.

    Same mechanism one degree up: a sextic factor vanishing at the six indices
    the two wrong terms break. The count of roots scales with the corruption,
    which is what makes the field usable as a diagnostic rather than a flag.
    """
    spoiled = _motzkin(71)
    spoiled[30] += 1
    spoiled[50] += 1

    guess = ak.guess_holonomic(spoiled, 4, 8)
    assert guess.singular_indices == (28, 29, 30, 48, 49, 50)
    assert guess.status == "singular"
    assert guess.confirmed is None


def test_the_evidence_dict_carries_the_verdict_and_the_roots():
    """``evidence()`` is what a research loop logs, so it must carry both."""
    spoiled = _motzkin(71)
    spoiled[30] += 1
    evidence = ak.guess_holonomic(spoiled).evidence()

    assert evidence["singular_indices"] == [28, 29, 30]
    assert evidence["status"] == "singular"
    assert evidence["confirmed"] is None
    assert "singular" in evidence["means"]


def test_the_status_vocabulary_is_closed_and_glossed():
    """Every status is nameable and has a meaning, as for ``NoveltyVerdict``."""
    assert set(ak.GUESS_STATUSES) == set(ak.GUESS_STATUS_MEANINGS)
    for name in ("GUESS_STATUSES", "GUESS_STATUS_MEANINGS"):
        assert name in ak.__all__

    spoiled = _motzkin(71)
    spoiled[30] += 1
    for guess in (ak.guess_holonomic(_motzkin(71)), ak.guess_holonomic(spoiled)):
        assert guess.status in ak.GUESS_STATUSES
        assert guess.means == ak.GUESS_STATUS_MEANINGS[guess.status]


def test_a_corrupted_fit_says_so_in_its_repr():
    """The one-line form a loop prints must not read as a clean answer."""
    spoiled = _motzkin(71)
    spoiled[30] += 1
    text = repr(ak.guess_holonomic(spoiled))
    assert "singular_indices=[28, 29, 30]" in text
    assert "status='singular'" in text
    assert "confirmed=None" in text


# ---------------------------------------------------------------------------
# The guard still refuses everything it was built to refuse
# ---------------------------------------------------------------------------


def _partitions(count):
    """``p(n)``, by Euler's pentagonal-number recurrence. Not P-recursive."""
    values = [1]
    for n in range(1, count):
        total = 0
        k = 1
        while True:
            for pentagonal in ((3 * k * k - k) // 2, (3 * k * k + k) // 2):
                if pentagonal > n:
                    break
                total += (-1) ** (k + 1) * values[n - pentagonal]
            if (3 * k * k - k) // 2 > n:
                break
            k += 1
        values.append(total)
    return values


def _bell(count):
    """Bell numbers off the Bell triangle. Not P-recursive."""
    row = [1]
    values = [1]
    for _ in range(count - 1):
        nxt = [row[-1]]
        for value in row:
            nxt.append(nxt[-1] + value)
        row = nxt
        values.append(row[0])
    return values


def _divisor(count, power):
    """``σ(n)`` for *power* 1 and ``τ(n)`` for *power* 0, from ``n = 1``."""
    return [sum(d**power for d in range(1, n + 1) if n % d == 0) for n in range(1, count + 1)]


def _digits_of_pi(count):
    """Decimal digits of ``π``, by the unbounded spigot. Not P-recursive."""
    out = []
    q, r, t, k, m, x = 1, 0, 1, 1, 3, 3
    while len(out) < count:
        if 4 * q + r - t < m * t:
            out.append(m)
            q, r, t, k, m, x = 10 * q, 10 * (r - m * t), t, k, (10 * (3 * q + r)) // t - 10 * m, x
        else:
            q, r, t, k, m, x = (
                q * k,
                (2 * q + r) * x,
                t * x,
                k + 1,
                (q * (7 * k + 2) + r * x) // (t * x),
                x + 2,
            )
    return out


def _pseudorandom(count):
    """A fixed pseudo-random sequence: the control with no structure at all."""
    values = []
    state = 20260820
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        values.append(state % 101 - 50)
    return values


def _beatty_sqrt(radicand, count):
    """``floor(n·√radicand)``, computed exactly. Not P-recursive."""
    decimal.getcontext().prec = 200
    root = decimal.Decimal(radicand).sqrt()
    return [int(decimal.Decimal(n) * root) for n in range(count)]


def _digit_sums(count):
    return [sum(int(c) for c in str(n)) for n in range(count)]


NOT_P_RECURSIVE = {
    "primes": PRIMES,
    "partitions": _partitions(60),
    "bell": _bell(60),
    "sigma": _divisor(60, 1),
    "tau": _divisor(60, 0),
    "pi_digits": _digits_of_pi(60),
    "pseudorandom": _pseudorandom(60),
    "beatty_phi": _beatty(60),
    "beatty_sqrt2": _beatty_sqrt(2, 60),
    "digit_sums": _digit_sums(60),
}


@pytest.mark.parametrize("name", sorted(NOT_P_RECURSIVE))
def test_ten_non_p_recursive_sequences_still_come_back_none(name):
    """The guard is real in the direction it was designed for, and stays real.

    Reporting singular indices and returning an underdetermined fit both loosen
    what comes *back*; neither may loosen what gets returned at all. Ten
    sequences that are not P-recursive — none of which the literature gives a
    P-recursive relation — must still answer ``None`` after a full sweep, not a
    fit with a caveat attached.
    """
    assert ak.guess_holonomic(NOT_P_RECURSIVE[name]) is None


def test_too_few_terms_still_refuses_and_names_the_shortfall():
    """``E-HOLO-005`` is untouched: undecided is still not a negative."""
    with pytest.raises(ak.HolonomicError) as excinfo:
        ak.guess_holonomic(MOTZKIN[:7])
    assert excinfo.value.code == "E-HOLO-005"
    assert "7 terms are not enough" in str(excinfo.value)


# ---------------------------------------------------------------------------
# dimension > 1 returns the solution space instead of refusing
# ---------------------------------------------------------------------------


def _a277060(count):
    """OEIS A277060, ``a(n) = (1/2)·Σ_k (C(n,k)·C(n+k,k+1))²``."""
    return [
        sum((math.comb(n, k) * math.comb(n + k, k + 1)) ** 2 for k in range(n + 1)) // 2
        for n in range(count)
    ]


def test_a_wider_probe_than_the_annihilator_returns_the_basis():
    """A277060: ``dimension`` 2 is information, and used to be a dead end.

    The probe that succeeds first is wider than the sequence's annihilator, so
    the terms admit two independent relations of that shape. Refusing made the
    whole ``(order, degree)`` cell unusable and closed a sequence that
    ``zeilberger`` decides immediately; the space is returned instead, and the
    verdict says the terms did not single a member of it out.
    """
    terms = _a277060(80)
    guess = ak.guess_holonomic(terms, 4, 6)

    assert guess is not None, "used to raise E-HOLO-005"
    assert guess.dimension == 2
    assert len(guess.basis) == 2
    assert guess.basis[0] == guess.coeffs
    assert guess.status == "underdetermined"
    assert guess.confirmed is None

    # Every element of the basis is a relation on the data, which is the whole
    # reason returning it is better than refusing.
    for vector in guess.basis:
        member = ak.GuessedRecurrence(
            order=guess.order,
            degree=guess.degree,
            start=guess.start,
            coeffs=vector,
            n_terms=guess.n_terms,
            n_equations=guess.n_equations,
            rank=guess.equations_used,
            dimension=1,
            min_surplus=guess.min_surplus,
            untested=guess.untested_candidates,
        )
        assert member.holds_for(terms)
