"""M11 — novelty filtering: normal form, hash, lookup, and what a negative means.

Nothing here touches the network. The OEIS half runs against
``tests/data/oeis_novelty_fixture.json``, a cache recorded once from
https://oeis.org (© The OEIS Foundation Inc., CC BY-NC-SA 4.0) and committed;
:class:`~alkahest.experimental.novelty.OeisWeb` is never constructed by a test.
To re-record it::

    from alkahest.experimental.novelty import OeisCache, OeisWeb
    web = OeisWeb(cache=OeisCache(), min_interval=1.5, max_results=8)
    for terms in (...):        # the exact term lists the tests query with
        web.lookup(terms=terms)
    web.cache.save("tests/data/oeis_novelty_fixture.json")

The recorded queries matter as much as the recorded entries: a cache that only
stores hits cannot tell "OEIS was asked and had nothing" from "nobody asked",
and reporting the second as the first is the overclaim this module exists to
prevent.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import alkahest as ak
import pytest
from alkahest.experimental import novelty
from alkahest.experimental.novelty import (
    NoveltyVerdict,
    OeisCache,
    OeisEntry,
    RecurrenceClaim,
    check_novelty,
)

FIXTURE = Path(__file__).resolve().parent / "data" / "oeis_novelty_fixture.json"

# ---------------------------------------------------------------------------
# The sequences this project has already certified recurrences for, computed
# here from their definitions rather than read out of the fixture — the point
# of the end-to-end test is that a claim derived independently lands on the
# same normal form as the one OEIS records.
# ---------------------------------------------------------------------------


def motzkin(count: int) -> list[int]:
    return [
        sum(math.comb(n, 2 * k) * math.comb(2 * k, k) // (k + 1) for k in range(n // 2 + 1))
        for n in range(count)
    ]


def catalan(count: int) -> list[int]:
    return [math.comb(2 * n, n) // (n + 1) for n in range(count)]


def central_binomial(count: int) -> list[int]:
    return [math.comb(2 * n, n) for n in range(count)]


def apery(count: int) -> list[int]:
    return [
        sum((math.comb(n, k) * math.comb(n + k, k)) ** 2 for k in range(n + 1))
        for n in range(count)
    ]


def a359643(count: int) -> list[int]:
    """``Sum_{k=0..n} C(n,k)*C(4k,k)`` — the session's novel result."""
    return [sum(math.comb(n, k) * math.comb(4 * k, k) for k in range(n + 1)) for n in range(count)]


#: The A359643 recurrence OEIS records, verbatim, marked "Conjecture".
A359643_OEIS_LINE = (
    "Conjecture D-finite with recurrence +81*n*(3*n-1)*(3*n-2)*a(n) "
    "+3*(243*n^3-8433*n^2+14984*n-7064)*a(n-1) "
    "+2*(-58607*n^3+297306*n^2-491401*n+269124)*a(n-2) "
    "+6*(n-2)*(56663*n^2-237722*n+252221)*a(n-3) "
    "-3*(n-2)*(n-3)*(111625*n-286402)*a(n-4) "
    "+110653*(n-2)*(n-3)*(n-4)*a(n-5)=0. - _R. J. Mathar_, Jan 09 2023"
)


@pytest.fixture(scope="module")
def cache() -> OeisCache:
    return OeisCache(FIXTURE)


# ---------------------------------------------------------------------------
# 1. Normalisation: presentations of one recurrence hash equal.
# ---------------------------------------------------------------------------

#: (n+1)·u(n+1) − (4n+2)·u(n) = 0, the central binomial recurrence, written six
#: ways.  Every one of them is the same statement about the same sequence.
CENTRAL_BINOMIAL_PRESENTATIONS = {
    "as fitted": ([(-2, -4), (1, 1)], 0),
    "scaled by -2": ([(4, 8), (-2, -2)], 0),
    "scaled by 7": ([(-14, -28), (7, 7)], 0),
    "over a common denominator": (
        [(Fraction(-1), Fraction(-2)), (Fraction(1, 2), Fraction(1, 2))],
        0,
    ),
    "with a padded window": ([(0,), (-2, -4), (1, 1), ()], -1),
    "times the polynomial (n+2)": ([(-4, -10, -4), (2, 3, 1)], 0),
    "stated about u(n+7)": ([(-60, -8), (16, 2)], 7),
}


@pytest.mark.parametrize("label", sorted(CENTRAL_BINOMIAL_PRESENTATIONS))
def test_presentations_of_one_recurrence_hash_equal(label: str) -> None:
    reference = RecurrenceClaim(*CENTRAL_BINOMIAL_PRESENTATIONS["as fitted"][:1])
    coefficients, offset = CENTRAL_BINOMIAL_PRESENTATIONS[label]
    claim = RecurrenceClaim(coefficients, offset=offset)
    assert claim.claim_hash == reference.claim_hash, (
        f"{label!r} normalised to {claim.normal_form!r}, not {reference.normal_form!r}"
    )
    assert claim == reference


def test_an_index_shift_is_not_a_different_claim() -> None:
    """Stating the relation about u(n+7) is re-indexing, not a new fact."""
    base = RecurrenceClaim([(-2, -4), (1, 1)])
    # The same relation applied at index n+7, then scaled by −2.
    shifted = RecurrenceClaim([(-60, -8), (16, 2)], offset=7)
    assert shifted.claim_hash == base.claim_hash
    # ... and stated about u(n−3), the way OEIS usually writes one.
    backwards = RecurrenceClaim([(-10, 4), (2, -1)], offset=-3)
    assert backwards.claim_hash == base.claim_hash


def test_genuinely_different_recurrences_do_not_collide() -> None:
    """The four certified sequences, plus two near misses, are six hashes."""
    claims = {
        "central binomial": RecurrenceClaim([(-2, -4), (1, 1)]),
        "catalan": RecurrenceClaim([(-2, -4), (2, 1)]),
        "one coefficient off": RecurrenceClaim([(-2, -4), (1, 2)]),
        "motzkin": RecurrenceClaim([(3, 3), (5, 2), (-4, -1)]),
        "fibonacci": RecurrenceClaim([(1,), (1,), (-1,)]),
        "apery": RecurrenceClaim.from_text(
            "(n+1)^3*a(n+1) = (34*n^3 + 51*n^2 + 27*n + 5)*a(n) - n^3*a(n-1), n >= 1."
        ),
    }
    hashes = {name: claim.claim_hash for name, claim in claims.items()}
    assert len(set(hashes.values())) == len(hashes), hashes


def test_normal_form_is_versioned_and_readable() -> None:
    claim = RecurrenceClaim([(-2, -4), (1, 1)])
    assert claim.normal_form == "recurrence/1 (4*n + 2)*u(n+0) + (-n - 1)*u(n+1)"
    assert claim.claim_hash.startswith("clm_")
    assert claim.order == 1
    assert claim.degree == 1


def test_a_claim_needs_two_terms() -> None:
    with pytest.raises(ValueError, match="at least two sequence terms"):
        RecurrenceClaim([(1, 1)])
    with pytest.raises(ValueError, match="at least two sequence terms"):
        RecurrenceClaim([(0,), (1, 1), (0,)])


def test_claims_dedupe_in_a_set() -> None:
    """The hash exists so a loop can dedupe its own output cheaply."""
    seen = set()
    for coefficients, offset in CENTRAL_BINOMIAL_PRESENTATIONS.values():
        seen.add(RecurrenceClaim(coefficients, offset=offset).claim_hash)
    assert len(seen) == 1


def test_claims_from_expressions_and_from_a_guess_agree() -> None:
    pool = ak.ExprPool()
    n = pool.symbol("n")
    one = pool.integer(1)
    from_exprs = RecurrenceClaim([-(pool.integer(4) * n + pool.integer(2)), n + one], var=n)
    guess = ak.guess_holonomic(central_binomial(20))
    assert RecurrenceClaim.from_recurrence(guess).claim_hash == from_exprs.claim_hash


def test_expression_coefficients_need_the_variable() -> None:
    pool = ak.ExprPool()
    n = pool.symbol("n")
    with pytest.raises(TypeError, match="index variable"):
        RecurrenceClaim([n, n + pool.integer(1)])


# ---------------------------------------------------------------------------
# 2. Parsing OEIS prose — and refusing to guess at it.
# ---------------------------------------------------------------------------


def test_parses_the_recurrences_oeis_actually_writes() -> None:
    lines = {
        # A000984, homogeneous and equal to zero
        "D-finite with recurrence: n*a(n) + 2*(1-2*n)*a(n-1)=0.": "central binomial",
        # A000108, a division and a trailing initial condition
        "Recurrence: a(n) = 2*(2*n-1)*a(n-1)/(n+1) with a(0) = 1.": "catalan",
        # A001006, prose prefix and an (End) marker
        "D-finite with recurrence: (n+2)*a(n) = (2*n+1)*a(n-1) + (3*n-3)*a(n-2). (End)": "motzkin",
        # A005259, a trailing range condition
        (
            "D-finite with recurrence (n+1)^3*a(n+1) = "
            "(34*n^3 + 51*n^2 + 27*n + 5)*a(n) - n^3*a(n-1), n >= 1."
        ): "apery",
    }
    for line in lines:
        assert RecurrenceClaim.from_text(line) is not None, line

    # A000108 states the same recurrence twice, in two shapes.  They are one
    # claim, which is exactly what the normal form is for.
    first = RecurrenceClaim.from_text("Recurrence: a(n) = 2*(2*n-1)*a(n-1)/(n+1) with a(0) = 1.")
    second = RecurrenceClaim.from_text("a(n) = a(n-1)*(4-6/(n+1)).")
    assert first.claim_hash == second.claim_hash


@pytest.mark.parametrize(
    "line",
    [
        # Another sequence on the right-hand side: truncating at `a(n-1)` would
        # invent a claim nobody made.
        "a(n) = a(n-1) + A002026(n-1). - _R. J. Mathar_, Jul 25 2017",
        # Nonlinear.
        "0 = a(n)*(16*a(n+1) - 10*a(n+2)) + a(n+1)*(2*a(n+1) + a(n+2)) for all n>=0.",
        # Convolution, and an absolute index a(0).
        "a(n+2) - a(n+1) = a(0)*a(n) + a(1)*a(n-1) + ... + a(n)*a(0).",
        # A sum, not a recurrence.
        "a(n) = Sum_{k=0..n} (-1)^(n-k)*binomial(n, k)*A000108(k+1).",
        # Inhomogeneous.
        "a(n) = a(n-1) + 1",
        # Not an equation at all.
        "Limit_{n->infinity} a(n)/a(n-1) = 3. [Aigner]",
        # An index that is not a shift of the running one.
        "a(2*n) = a(n-1)*a(n+1)",
    ],
)
def test_refuses_lines_it_does_not_understand(line: str) -> None:
    assert RecurrenceClaim.from_text(line) is None


def test_a_parsed_recurrence_is_checked_against_the_entrys_own_data() -> None:
    """A line that does not reproduce the entry's terms is not indexed.

    The parser is the weakest link in the chain, so its output is confirmed
    against the data OEIS ships with the entry before it can produce a match.
    """
    honest = OeisEntry(
        "A000984",
        terms=central_binomial(12),
        statements=["D-finite with recurrence: n*a(n) + 2*(1-2*n)*a(n-1)=0."],
    )
    assert len(honest.recurrences()) == 1
    assert honest.unusable_statements() == ()

    mistyped = OeisEntry(
        "A000984",
        terms=central_binomial(12),
        statements=["D-finite with recurrence: n*a(n) + 3*(1-2*n)*a(n-1)=0."],
    )
    assert mistyped.recurrences() == ()
    assert len(mistyped.unusable_statements()) == 1


def test_holds_for_is_exact_and_trailing_confirmations_are_lenient() -> None:
    claim = RecurrenceClaim([(-2, -4), (1, 1)])
    terms = central_binomial(15)
    assert claim.holds_for(terms)
    assert claim.confirmations(terms) == len(terms) - 1
    # A recurrence stated only for large n fails at the front and is still
    # confirmed by the tail — the distinction `confirmations` exists for.
    # `start` names the true index of element 0 of the array *given to this
    # call*, always — prepending `99` shifts every real element one array
    # position to the right, so the array's own index-0 is now where `u(-1)`
    # would be: start=-1, not the default 0. Get that wrong (as `start=0`
    # does here, silently claiming `99 = u(0)`) and every window's `n` is
    # off by one, so *none* of them confirm — not just the one touching `99`
    # — which is exactly why `start` is load-bearing, not cosmetic.
    padded = [99, *terms]
    assert not claim.holds_for(padded, start=-1)
    assert claim.confirmations(padded, start=-1) == len(terms) - 1
    assert claim.confirmations(padded, start=0) == 0, (
        "a caller who forgets to adjust start for the prepended element gets "
        "a meaningless count, not a merely-approximate one"
    )


# ---------------------------------------------------------------------------
# 3. End to end against real OEIS data, offline.
# ---------------------------------------------------------------------------

CERTIFIED = {
    "A005259": (apery, 10),  # Apéry numbers
    "A001006": (motzkin, 12),  # Motzkin numbers
    "A000108": (catalan, 12),  # Catalan numbers
    "A000984": (central_binomial, 12),  # central binomial coefficients
}


@pytest.mark.parametrize("entry_id", sorted(CERTIFIED))
def test_recurrences_this_project_certifies_are_already_in_oeis(
    cache: OeisCache, entry_id: str
) -> None:
    """Guess the recurrence, normalise it, and find it recorded — unhedged."""
    build, query_length = CERTIFIED[entry_id]
    terms = build(30)
    guess = ak.guess_holonomic(terms, max_order=3, max_degree=4)
    claim = RecurrenceClaim.from_recurrence(guess)

    verdict = check_novelty(claim, [cache], terms=terms[:query_length])
    assert verdict.status == "recorded", verdict.report()
    assert verdict.found is True
    assert verdict.hedged is False, "OEIS states these as theorems, not conjectures"
    assert entry_id in {m.entry for m in verdict.matches()}
    assert verdict.entries_examined >= 1
    assert verdict.claim_hash == claim.claim_hash


def test_a359643_is_recorded_but_only_as_a_conjecture(cache: OeisCache) -> None:
    """The distinction the whole filter is for.

    OEIS carries this recurrence marked "Conjecture", i.e. fitted by a guessing
    package and never proved.  A run that certifies it has a result; a run that
    merely restates it does not, and only the hedge tells them apart.
    """
    terms = a359643(40)
    recorded = RecurrenceClaim.from_text(A359643_OEIS_LINE)
    assert recorded.order == 5
    assert recorded.holds_for(terms), "the parse must reproduce the sequence"

    verdict = check_novelty(recorded, [cache], terms=terms[:12])
    assert verdict.status == "recorded_conjecturally", verdict.report()
    assert verdict.found is True
    assert verdict.hedged is True
    match = verdict.matches()[0]
    assert match.entry == "A359643"
    assert "Conjecture" in match.statement


def test_the_order_four_relation_for_a359643_is_not_in_oeis(cache: OeisCache) -> None:
    """A claim OEIS does not have — reported as *not found*, never as novel.

    ``guess_holonomic`` fits an order-4 relation to the same sequence, one
    order below the order-5 recurrence OEIS records as a conjecture.  Both hold
    on every term computed here; they are different claims, and the filter says
    so without saying anything about the literature.
    """
    terms = a359643(80)
    guess = ak.guess_holonomic(terms, max_order=5, max_degree=4)
    claim = RecurrenceClaim.from_recurrence(guess)
    assert claim.order == 4
    assert claim.holds_for(terms)
    assert claim.claim_hash != RecurrenceClaim.from_text(A359643_OEIS_LINE).claim_hash

    verdict = check_novelty(claim, [cache], terms=terms[:12])
    assert verdict.status == "not_found"
    assert verdict.found is False
    assert verdict.entries_examined == 1
    assert verdict.statements_compared >= 1


def test_a_sequence_oeis_does_not_have_is_not_found_not_unavailable(cache: OeisCache) -> None:
    """A recorded empty result is a real negative; an unrecorded query is not."""
    absent = [1, 7, 41, 2393, 168413, 9930001, 700000009, 42000000023, 3100000000007]
    verdict = check_novelty(RecurrenceClaim([(-2, -4), (1, 1)]), [cache], terms=absent)
    assert verdict.status == "not_found"
    assert verdict.entries_examined == 0

    never_asked = check_novelty(
        RecurrenceClaim([(-2, -4), (1, 1)]), [cache], terms=[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    )
    assert never_asked.status == "unavailable"
    assert never_asked.found is None


def test_lookup_by_identifier(cache: OeisCache) -> None:
    claim = RecurrenceClaim.from_text(
        "D-finite with recurrence: n*a(n) + 2*(1-2*n)*a(n-1)=0.",
    )
    verdict = check_novelty(claim, [cache], ids=["A000984"])
    assert verdict.status == "recorded"
    assert check_novelty(claim, [cache], ids=["A999999"]).status == "unavailable"


# ---------------------------------------------------------------------------
# 4. What a negative is allowed to claim.
# ---------------------------------------------------------------------------


def test_a_negative_verdict_never_says_novel(cache: OeisCache) -> None:
    """The one property this module must not have."""
    terms = a359643(80)
    guess = ak.guess_holonomic(terms, max_order=5, max_degree=4)
    verdict = check_novelty(RecurrenceClaim.from_recurrence(guess), [cache], terms=terms[:12])
    assert verdict.status == "not_found"

    for attribute in ("novel", "is_novel", "new", "unpublished", "original"):
        assert not hasattr(verdict, attribute), (
            f"NoveltyVerdict.{attribute} would turn 'not in the one place I "
            f"looked' into a claim about the literature"
        )
    report = verdict.report()
    assert "novel" not in report
    assert not any("novel" in str(v).lower() for v in report.values() if isinstance(v, bool))
    # The gloss that travels with the verdict says what it is worth.
    assert "not evidence of novelty" in verdict.means
    assert verdict.means == novelty.STATUS_MEANINGS["not_found"]


def test_a_verdict_has_no_truth_value(cache: OeisCache) -> None:
    """`if verdict:` is the mistake; it raises instead of reading True."""
    verdict = check_novelty(RecurrenceClaim([(-2, -4), (1, 1)]), [], terms=[1, 2, 6, 20])
    with pytest.raises(TypeError, match="no truth value"):
        bool(verdict)
    with pytest.raises(TypeError, match="no truth value"):  # noqa: PT012 - the `if` itself is the point
        if verdict:  # pragma: no cover - the raise is the assertion
            pass


def test_no_sources_is_unavailable_not_a_pass() -> None:
    verdict = check_novelty(RecurrenceClaim([(-2, -4), (1, 1)]), [], terms=[1, 2, 6, 20])
    assert verdict.status == "unavailable"
    assert verdict.found is None
    assert verdict.hedged is None
    assert verdict.sources_consulted() == ()


def test_a_source_that_cannot_promise_completeness_cannot_produce_a_negative() -> None:
    """A local hit is evidence; a local miss is not, unless the query was asked.

    A cache holding an entry that was never the recorded answer to this query
    can confirm a claim but must not refute one — it does not know what else
    OEIS has.
    """
    cache = OeisCache()
    cache.add(
        OeisEntry(
            "A000984",
            terms=central_binomial(12),
            statements=["D-finite with recurrence: n*a(n) + 2*(1-2*n)*a(n-1)=0."],
        )
    )
    query = central_binomial(12)

    found = check_novelty(RecurrenceClaim([(-2, -4), (1, 1)]), [cache], terms=query)
    assert found.status == "recorded", "a hit is a hit wherever it came from"

    missing = check_novelty(RecurrenceClaim([(3, 3), (5, 2), (-4, -1)]), [cache], terms=query)
    assert missing.status == "unavailable", "an unrecorded query cannot support a negative"
    assert missing.sources_unavailable() == ("oeis-cache",)

    cache.record_query(terms=query, found=["A000984"])
    now_negative = check_novelty(RecurrenceClaim([(3, 3), (5, 2), (-4, -1)]), [cache], terms=query)
    assert now_negative.status == "not_found"


def test_report_carries_the_scope_of_the_search(cache: OeisCache) -> None:
    """A negative comes with how far the search actually reached."""
    terms = motzkin(30)
    guess = ak.guess_holonomic(terms, max_order=3, max_degree=4)
    claim = RecurrenceClaim.from_recurrence(guess)
    report = check_novelty(claim, [cache], terms=terms[:12]).report()
    assert set(report) == {
        "status",
        "found",
        "hedged",
        "means",
        "claim_hash",
        "matches",
        "sources_consulted",
        "sources_unavailable",
        "entries_examined",
        "statements_compared",
        "statements_unusable",
    }
    assert report["sources_consulted"] == ["oeis-cache"]
    # OEIS says a great deal this parser cannot read, and the count of what it
    # could not use is part of the answer rather than a swallowed detail.
    assert report["statements_unusable"] > 0
    assert report["status"] in novelty.NOVELTY_STATUSES


def test_check_novelty_refuses_a_raw_recurrence() -> None:
    with pytest.raises(TypeError, match="must be a RecurrenceClaim"):
        check_novelty([(-2, -4), (1, 1)], [], terms=[1, 2, 6])
    with pytest.raises(ValueError, match="nothing to look up"):
        check_novelty(RecurrenceClaim([(-2, -4), (1, 1)]), [])


# ---------------------------------------------------------------------------
# 5. The offline/online split, and the API surface.
# ---------------------------------------------------------------------------


def test_the_fixture_records_queries_as_well_as_entries(cache: OeisCache) -> None:
    assert cache.n_entries >= 5
    assert cache.n_queries >= 6, "without recorded queries no negative is possible"
    assert cache.name == "oeis-cache"


def test_cache_round_trips_through_a_file(tmp_path: Path) -> None:
    original = OeisCache(FIXTURE)
    target = tmp_path / "nested" / "copy.json"
    original.save(target)
    assert "CC BY-NC-SA" in target.read_text(encoding="utf-8"), "OEIS attribution travels with it"
    reloaded = OeisCache(target)
    assert reloaded.n_entries == original.n_entries
    assert reloaded.n_queries == original.n_queries


def test_web_source_is_opt_in_and_never_default() -> None:
    """No default reaches for the network; the type is not even constructed."""
    import inspect

    signature = inspect.signature(check_novelty)
    assert signature.parameters["sources"].default is inspect.Parameter.empty
    source = inspect.getsource(novelty.check_novelty)
    assert "OeisWeb(" not in source


def test_experimental_exports() -> None:
    from alkahest import experimental

    for name in ("RecurrenceClaim", "NoveltyVerdict", "OeisCache", "OeisWeb", "check_novelty"):
        assert name in experimental.__all__
        assert hasattr(experimental, name)
    assert "novelty" in experimental.__all__
    assert issubclass(NoveltyVerdict, object)


def test_accessor_convention_holds_for_the_new_types() -> None:
    """Zero-argument O(1) scalars are properties; collections are methods."""
    for cls, scalars, collections in (
        (RecurrenceClaim, ("order", "degree", "normal_form", "claim_hash"), ("coefficients",)),
        (
            NoveltyVerdict,
            (
                "status",
                "found",
                "hedged",
                "claim_hash",
                "entries_examined",
                "statements_compared",
                "statements_unusable",
                "means",
            ),
            ("matches", "sources_consulted", "sources_unavailable", "report"),
        ),
        (OeisCache, ("name", "n_entries", "n_queries"), ("lookup", "save", "load", "add")),
        (OeisEntry, (), ("recurrences", "unusable_statements", "to_json")),
    ):
        for name in scalars:
            assert isinstance(inspect_static(cls, name), property), f"{cls.__name__}.{name}"
        for name in collections:
            assert callable(inspect_static(cls, name)), f"{cls.__name__}.{name}"
            assert not isinstance(inspect_static(cls, name), property), f"{cls.__name__}.{name}"


def inspect_static(cls: type, name: str) -> object:
    import inspect

    return inspect.getattr_static(cls, name)
