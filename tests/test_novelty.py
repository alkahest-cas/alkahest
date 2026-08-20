"""M11 — novelty filtering: normal form, hash, lookup, and what a negative means.

Nothing here touches the network. The OEIS half runs against
``tests/data/oeis_novelty_fixture.json``, a cache recorded once from
https://oeis.org (© The OEIS Foundation Inc., CC BY-NC-SA 4.0) and committed.
To re-record it::

    from alkahest.experimental.novelty import OeisCache, OeisWeb
    web = OeisWeb(cache=OeisCache(), min_interval=1.5)
    for terms in (...):        # the exact term lists the tests query with
        web.lookup(terms=terms)
    web.lookup(ids=["A000045"])
    web.cache.save("tests/data/oeis_novelty_fixture.json")

:class:`~alkahest.experimental.novelty.OeisWeb` *is* constructed, by the paging
tests only, and never with a live transport: ``urlopen`` is replaced with one
that serves :data:`PAGING_FIXTURE`, the recorded raw pages, so that what a full
result page means — and does not mean — is covered offline like everything
else.

The recorded queries matter as much as the recorded entries: a cache that only
stores hits cannot tell "OEIS was asked and had nothing" from "nobody asked",
and reporting the second as the first is the overclaim this module exists to
prevent.
"""

from __future__ import annotations

import json
import math
import urllib.parse
from fractions import Fraction
from pathlib import Path

import alkahest as ak
import pytest
from alkahest.experimental import novelty
from alkahest.experimental.novelty import (
    NoveltyVerdict,
    OeisCache,
    OeisEntry,
    OeisWeb,
    QRecurrenceClaim,
    RecurrenceClaim,
    check_novelty,
)

FIXTURE = Path(__file__).resolve().parent / "data" / "oeis_novelty_fixture.json"
#: Raw `search?...&fmt=json` pages, keyed `"query|start"`, recorded once from
#: oeis.org.  `OeisWeb` is exercised against these through a fake transport, so
#: the paging behaviour is tested without the network the module promises never
#: to need.
PAGING_FIXTURE = Path(__file__).resolve().parent / "data" / "oeis_paging_fixture.json"

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
        "terms_check",
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

    for name in (
        "RecurrenceClaim",
        "QRecurrenceClaim",
        "NoveltyVerdict",
        "OeisCache",
        "OeisWeb",
        "check_novelty",
    ):
        assert name in experimental.__all__
        assert hasattr(experimental, name)
    assert "novelty" in experimental.__all__
    assert issubclass(NoveltyVerdict, object)


def test_accessor_convention_holds_for_the_new_types() -> None:
    """Zero-argument O(1) scalars are properties; collections are methods."""
    for cls, scalars, collections in (
        (
            RecurrenceClaim,
            ("order", "degree", "normal_form", "claim_hash", "claim_kind"),
            ("coefficients",),
        ),
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
                "terms_check",
            ),
            ("matches", "sources_consulted", "sources_unavailable", "report"),
        ),
        (
            QRecurrenceClaim,
            ("order", "degree", "q_degree", "normal_form", "claim_hash", "claim_kind"),
            ("coefficients",),
        ),
        (OeisCache, ("name", "n_entries", "n_queries"), ("lookup", "save", "load", "add")),
        (
            OeisEntry,
            (),
            ("recurrences", "unusable_statements", "candidate_lines", "to_json"),
        ),
    ):
        for name in scalars:
            assert isinstance(inspect_static(cls, name), property), f"{cls.__name__}.{name}"
        for name in collections:
            assert callable(inspect_static(cls, name)), f"{cls.__name__}.{name}"
            assert not isinstance(inspect_static(cls, name), property), f"{cls.__name__}.{name}"


def inspect_static(cls: type, name: str) -> object:
    import inspect

    return inspect.getattr_static(cls, name)


# ---------------------------------------------------------------------------
# 6. The coverage the filter actually has (issues #23, #24, #25, #26q).
# ---------------------------------------------------------------------------

#: The Fibonacci recurrence, `u(n+2) = u(n+1) + u(n)`, as a claim.
FIBONACCI = ((-1,), (-1,), (1,))


def test_the_fibonacci_recurrence_is_found_in_the_fibonacci_entry(cache: OeisCache) -> None:
    """#24. A filter that cannot find this clears almost anything.

    A000045 states its recurrence in its **name** — ``Fibonacci numbers: F(n) =
    F(n-1) + F(n-2)`` — and nowhere in the formula lines the parser used to be
    pointed at, so the whole entry came back with zero usable statements and the
    verdict was ``not_found``: "not in the sources searched", read by a loop
    author as novelty.
    """
    claim = RecurrenceClaim(FIBONACCI)
    verdict = check_novelty(claim, [cache], ids=["A000045"])
    assert verdict.status == "recorded", verdict.report()
    assert verdict.found is True
    assert verdict.hedged is False, "the name of an entry is not a conjecture"
    assert verdict.statements_compared > 0
    assert {m.entry for m in verdict.matches()} == {"A000045"}
    assert any(m.statement.startswith("Fibonacci numbers:") for m in verdict.matches())


def test_the_name_of_an_entry_is_a_candidate_line(cache: OeisCache) -> None:
    entry = cache.lookup(ids=["A000045"]).entries[0]
    assert entry.candidate_lines()[0] == entry.name
    assert entry.candidate_lines()[1:] == entry.statements
    assert RecurrenceClaim.from_text(entry.name).claim_hash == RecurrenceClaim(FIBONACCI).claim_hash

    # A name that states nothing is not turned into a candidate.
    plain = OeisEntry("A000027", "The positive integers.", terms=list(range(1, 20)))
    assert plain.candidate_lines() == ()


@pytest.mark.parametrize(
    ("line", "why"),
    [
        (
            "Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.",
            "the sequence under the letter it is named for, in an entry's name",
        ),
        ("a(n) = 2a(n-2) + a(n-3), n > 2.", "implicit multiplication"),
        ("L(n) = L(n-1) + L(n-2).", "another single letter"),
        ("a(n) = a(n-1) + A000045(n-2).", "the entry's own A-number spelled out"),
    ],
)
def test_parses_the_other_notations_oeis_uses(line: str, why: str) -> None:
    assert RecurrenceClaim.from_text(line, names=("A000045",)) is not None, why


@pytest.mark.parametrize(
    "line",
    [
        # Two sequences in one relation is not a recurrence for either of them,
        # whichever way they are spelled.
        "F(n) = L(n-1) + L(n-2).",
        "a(n) = a(n-1) + A002026(n-1). - _R. J. Mathar_, Jul 25 2017",
        "a(n) = 2a(n-1) + 3b(n-2).",
        # A function that is not a sequence must not become one.
        "a(n) = floor(a(n-1)*phi).",
        # Two indices is a triangle, not a sequence; `A(x)` is a generating
        # function, and `x` is not the running index.
        "T(n,k) = T(n-1,k) + T(n-1,k-1).",
        "A(x) = 1 + x*A(x)^2.",
    ],
)
def test_the_widened_parser_still_refuses_what_it_should(line: str) -> None:
    assert RecurrenceClaim.from_text(line, names=("A000045",)) is None, line


@pytest.mark.parametrize(
    "line",
    [
        # Prose after the formula is prose, not a factor: reading `n > 2` as an
        # implicit multiplication would invent `(a(n-1) + a(n-2))*n`, a claim
        # nobody made, and reading `for` as one would invent a different one.
        "a(n) = a(n-1) + a(n-2) for n > 2, with a(0) = 0.",
        "a(n) = a(n-1) + a(n-2), n >= 2.",
        "F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.",
    ],
)
def test_implicit_multiplication_does_not_swallow_the_prose_after_a_formula(line: str) -> None:
    claim = RecurrenceClaim.from_text(line)
    assert claim is not None, line
    assert claim.claim_hash == RecurrenceClaim(FIBONACCI).claim_hash, claim.normal_form


def test_a_single_letter_sequence_is_still_held_to_the_entrys_own_data() -> None:
    """The letter is not trusted; the data is.

    ``b(n) = b(n-1) + b(n-2)`` in a comment may be about an auxiliary sequence,
    so what licenses indexing it is the same thing that licenses an ``a(n)``
    line: it has to reproduce the terms the entry ships with.
    """
    line = "Let b(n) = b(n-1) + b(n-2), with b(0) = 0, b(1) = 1."
    fibonacci = OeisEntry("A000045", terms=[0, 1, 1, 2, 3, 5, 8, 13, 21, 34], statements=[line])
    assert len(fibonacci.recurrences()) == 1

    elsewhere = OeisEntry("A000079", terms=[1, 2, 4, 8, 16, 32, 64, 128], statements=[line])
    assert elsewhere.recurrences() == ()
    assert len(elsewhere.unusable_statements()) == 1


# ---------------------------------------------------------------------------
# Paging (#23), against recorded pages through a fake transport.
# ---------------------------------------------------------------------------


class _RecordedResponse:
    """The two methods :class:`OeisWeb` uses of a ``urlopen`` result."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _RecordedResponse:
        return self

    def __exit__(self, *exception: object) -> bool:
        return False


@pytest.fixture
def oeis_transport(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve :data:`PAGING_FIXTURE`; return the list of ``"query|start"`` asked for."""
    pages = json.loads(PAGING_FIXTURE.read_text(encoding="utf-8"))["pages"]
    asked: list[str] = []

    def urlopen(request: object, timeout: float | None = None) -> _RecordedResponse:
        parameters = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        key = f"{parameters['q'][0]}|{int(parameters.get('start', ['0'])[0])}"
        asked.append(key)
        assert key in pages, f"no page recorded for {key!r}; re-record the fixture"
        return _RecordedResponse(json.dumps(pages[key]).encode("utf-8"))

    monkeypatch.setattr(novelty.urllib.request, "urlopen", urlopen)
    return asked


def test_a_terms_search_is_paged_and_one_full_page_is_not_exhaustive(
    oeis_transport: list[str],
) -> None:
    """#23. ``fmt=json`` sends ten results and no count, so ten is not "all"."""
    web = OeisWeb(cache=OeisCache(), min_interval=0.0, max_results=15)
    answer = web.lookup(terms=[1, 1, 2, 3, 5, 8, 13])
    assert oeis_transport == ["1,1,2,3,5,8,13|0", "1,1,2,3,5,8,13|10"], (
        "a full first page must be followed by `&start=10`"
    )
    assert answer.exhaustive is False, "OEIS never said these were all of them"
    assert len(answer.entries) == 15

    # And the partial answer must not be recorded as a complete one, or every
    # later offline run would read it as a licence to say `not_found`.
    assert web.cache.lookup(terms=[1, 1, 2, 3, 5, 8, 13]).exhaustive is False


def test_an_ids_lookup_stays_exhaustive_after_one_request(oeis_transport: list[str]) -> None:
    """The other direction of #23: ``id:A…`` asks for named entries and gets them."""
    web = OeisWeb(cache=OeisCache(), min_interval=0.0)
    answer = web.lookup(ids=["A000045"])
    assert oeis_transport == ["id:A000045|0"], "an identifier lookup is not paged"
    assert answer.exhaustive is True
    assert [entry.id for entry in answer.entries] == ["A000045"]
    assert web.cache.lookup(ids=["A000045"]).exhaustive is True


def test_a_terms_search_that_runs_out_of_results_is_exhaustive(
    oeis_transport: list[str],
) -> None:
    """A short page *is* the end of the search, and may be recorded as one."""
    web = OeisWeb(cache=OeisCache(), min_interval=0.0)
    terms = apery(10)
    answer = web.lookup(terms=terms)
    assert oeis_transport == [",".join(str(t) for t in terms) + "|0"]
    assert answer.exhaustive is True
    assert [entry.id for entry in answer.entries] == ["A005259"]
    assert web.cache.lookup(terms=terms).exhaustive is True


def test_a_paged_out_search_is_unavailable_and_never_a_negative(
    oeis_transport: list[str],
) -> None:
    """The whole point of #23: it collapsed ``unavailable`` into ``not_found``."""
    web = OeisWeb(cache=OeisCache(), min_interval=0.0, max_results=15)
    absent = RecurrenceClaim([(3, 3), (5, 2), (-4, -1)])
    verdict = check_novelty(absent, [web], terms=[1, 1, 2, 3, 5, 8, 13])
    assert verdict.matches() == ()
    assert verdict.status == "unavailable"
    assert verdict.found is None
    assert verdict.sources_unavailable() == ("oeis",)


# ---------------------------------------------------------------------------
# `q`-recurrences (#25).
# ---------------------------------------------------------------------------

#: `(1 - q^n)·u(n) - u(n+1) = 0`, written five ways.  `(i, j)` is `q^i·(q^n)^j`.
Q_PRESENTATIONS = {
    "as stated": ([{(0, 0): 1, (0, 1): -1}, {(0, 0): -1}], 0),
    "scaled by -2*q^5*(q^n)^2": ([{(5, 2): -2, (5, 3): 2}, {(5, 2): 2}], 0),
    "times the polynomial 1 + q*q^n": (
        [{(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 2): -1}, {(0, 0): -1, (1, 1): -1}],
        0,
    ),
    "stated about u(n+3), scaled by -2q": ([{(1, 0): -2, (4, 1): 2}, {(1, 0): 2}], 3),
    "stated about u(n-3), scaled by 3, padded": (
        [{}, {(0, 0): 3, (-3, 1): -3}, {(0, 0): -3}, {}],
        -4,
    ),
}


@pytest.mark.parametrize("label", sorted(Q_PRESENTATIONS))
def test_presentations_of_one_q_recurrence_hash_equal(label: str) -> None:
    """#25. The claim type M4's `q` half had no route into.

    The index shift is the interesting one: ``n → n+1`` sends ``q^n`` to
    ``q·q^n``, so re-indexing a `q`-recurrence rewrites its coefficients rather
    than leaving them alone.
    """
    reference = QRecurrenceClaim(*Q_PRESENTATIONS["as stated"][:1])
    coefficients, offset = Q_PRESENTATIONS[label]
    claim = QRecurrenceClaim(coefficients, offset=offset)
    assert claim.claim_hash == reference.claim_hash, (
        f"{label!r} normalised to {claim.normal_form!r}, not {reference.normal_form!r}"
    )
    assert claim == reference


def test_a_q_claim_accepts_rational_coefficients_and_expressions() -> None:
    pool = ak.ExprPool()
    one = pool.integer(1)
    n, q = pool.symbol("n"), pool.symbol("q")
    power = q**n
    denominator = one + q * power
    claim = QRecurrenceClaim([(one - power) / denominator, -one / denominator], var=n, q=q)
    assert claim.claim_hash == QRecurrenceClaim(*Q_PRESENTATIONS["as stated"][:1]).claim_hash
    assert claim.normal_form == "q-recurrence/1 (q^n - 1)*u(n+0) + (1)*u(n+1)"
    assert (claim.order, claim.degree, claim.q_degree) == (1, 1, 0)


def test_a_q_certificate_becomes_a_claim() -> None:
    """The exact call that used to raise ``coefficient mentions the symbol 'q'``."""
    from alkahest.experimental import q_zeilberger, qbinomial

    pool = ak.ExprPool()
    n, k, q = pool.symbol("n"), pool.symbol("k"), pool.symbol("q")
    binomial = qbinomial(pool, n, k)
    certificate = q_zeilberger(binomial * binomial * q ** (k * k), q, n, k)

    with pytest.raises(ValueError, match="QRecurrenceClaim"):
        RecurrenceClaim.from_recurrence(certificate, var=n)

    claim = QRecurrenceClaim.from_recurrence(certificate, var=n, q=q)
    assert claim.order == 1
    assert claim.claim_kind == "q-recurrence"
    assert claim.normal_form.startswith("q-recurrence/1 ")
    assert claim.claim_hash.startswith("clm_")


def test_a_q_claim_does_not_collide_with_an_ordinary_one() -> None:
    ordinary = RecurrenceClaim([(1,), (-1,)])  # u(n+1) = u(n)
    q_analogue = QRecurrenceClaim([{(0, 0): 1}, {(0, 0): -1}])
    assert ordinary.normal_form.startswith("recurrence/1 ")
    assert q_analogue.normal_form.startswith("q-recurrence/1 ")
    assert ordinary.claim_hash != q_analogue.claim_hash
    assert len({ordinary, q_analogue}) == 2
    assert ordinary != q_analogue


def test_a_source_that_cannot_state_a_q_recurrence_is_unavailable_for_one(
    cache: OeisCache,
) -> None:
    """Not ``not_found``: a search that could not have matched is not a negative."""
    claim = QRecurrenceClaim([{(0, 0): 1, (0, 1): -1}, {(0, 0): -1}])
    verdict = check_novelty(claim, [cache], terms=[1, 2, 6, 20, 70, 252, 924, 3432])
    assert verdict.status == "unavailable"
    assert verdict.found is None
    assert verdict.sources_consulted() == ()
    assert verdict.sources_unavailable() == ("oeis-cache",)
    assert verdict.terms_check == "not_checked"
    assert verdict.claim_hash == claim.claim_hash


# ---------------------------------------------------------------------------
# The `terms` cross-check (#26q).
# ---------------------------------------------------------------------------


def test_terms_are_checked_against_the_claim_not_only_used_to_search(
    cache: OeisCache,
) -> None:
    """#26q. *terms* said which sequence; the claim has to be about it."""
    claim = RecurrenceClaim([(-2, -4), (1, 1)])  # (n+1)u(n+1) = (4n+2)u(n)

    agrees = check_novelty(claim, [cache], terms=central_binomial(12))
    assert agrees.terms_check == "holds"
    assert agrees.report()["terms_check"] == "holds"
    assert agrees.status == "recorded"

    # The central binomial recurrence, looked up by the Motzkin numbers: the
    # search was about one sequence and the claim about another, and saying so
    # is the difference between a report and a silently misleading `not_found`.
    disagrees = check_novelty(claim, [cache], terms=motzkin(12))
    assert disagrees.terms_check == "fails"
    assert disagrees.report()["terms_check"] == "fails"
    assert "terms_check='fails'" in repr(disagrees)
    assert disagrees.terms_check in novelty.TERMS_CHECKS

    # Nothing to check it against is not a failure.
    assert check_novelty(claim, [cache], ids=["A000984"]).terms_check == "not_checked"
    assert check_novelty(claim, [cache], terms=[1]).terms_check == "not_checked"


def test_the_terms_cross_check_reads_start_the_way_holds_for_does(cache: OeisCache) -> None:
    """The one way an honest caller can trip it, and the knob that fixes it."""
    claim = RecurrenceClaim([(-2, -4), (1, 1)])
    padded = [99, *central_binomial(12)]
    assert check_novelty(claim, [cache], terms=padded, start=0).terms_check == "fails"
    assert check_novelty(claim, [cache], terms=padded, start=-1).terms_check == "holds"
