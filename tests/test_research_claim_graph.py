"""Session-level claim graph (``alkahest.research``).

Covers the four properties the feature has to get right: content-addressed
identity, honest status propagation, lossless/diffable serialisation, and
re-verification that can only lower confidence.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import alkahest as ak
import pytest
from alkahest.research import (
    MACHINE_CHECKED_STATUSES,
    SCHEMA_VERSION,
    Claim,
    ClaimGraph,
    ClaimGraphError,
    CycleError,
    MissingClaimError,
    captured_operations,
    claim_id,
    session,
)
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.fixture
def pool():
    return ak.ExprPool()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_research_is_exported():
    assert "research" in ak.__all__
    assert ak.research.SCHEMA_VERSION == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Content-addressed identity
# ---------------------------------------------------------------------------


def test_claim_id_is_deterministic():
    first = claim_id("(1/2 * log(2))", ["x > 0"], "integrate")
    second = claim_id("(1/2 * log(2))", ["x > 0"], "integrate")
    assert first == second
    assert first.startswith("clm_")
    assert len(first) == len("clm_") + 16


def test_claim_id_ignores_hypothesis_order_and_whitespace():
    assert claim_id("a = b", ["p", "q"], "m") == claim_id("a  =  b", ["q", "p"], "m")


def test_claim_id_depends_on_hypotheses_and_method():
    base = claim_id("a = b", ["x > 0"], "integrate")
    assert base != claim_id("a = b", [], "integrate")
    assert base != claim_id("a = b", ["x > 0"], "simplify")
    assert base != claim_id("a = c", ["x > 0"], "integrate")


def test_same_claim_in_two_sessions_gets_the_same_id(pool):
    def build():
        local_pool = ak.ExprPool()
        x = local_pool.symbol("x")
        with session(pool=local_pool, capture=True) as s:
            ak.integrate(x ** local_pool.integer(2), x)
        return s.graph.ids[0]

    assert build() == build()


# ---------------------------------------------------------------------------
# Honesty: the recording layer never upgrades a status
# ---------------------------------------------------------------------------


def test_record_copies_verification_status_verbatim(pool):
    x = pool.symbol("x")
    definite = ak.integrate(
        x / (x ** pool.integer(2) + pool.integer(1)), x, pool.integer(0), pool.integer(1)
    )
    assert definite.verification["status"] == "unverified"
    with session(pool=pool) as s:
        claim = s.record(definite, method="integrate")
    assert claim.status == "unverified"
    assert claim.machine_checked is False
    assert claim.verification == dict(definite.verification)


def test_verified_result_keeps_its_verified_status(pool):
    x = pool.symbol("x")
    result = ak.integrate(x * ak.sin(x), x)
    assert result.verification["status"] == "exactly_verified"
    with session(pool=pool) as s:
        claim = s.record(result, method="integrate")
    assert claim.status == "exactly_verified"
    assert claim.machine_checked is True
    assert claim.status in MACHINE_CHECKED_STATUSES


def test_conjecture_is_always_unverified(pool):
    x = pool.symbol("x")
    verified = ak.integrate(x * ak.sin(x), x)
    with session(pool=pool) as s:
        claim = s.conjecture(verified, evidence="PSLQ at 60 digits")
    assert claim.status == "unverified"
    assert claim.machine_checked is False
    assert "PSLQ" in claim.evidence


def test_certificate_available_is_not_machine_checked(pool):
    x = pool.symbol("x")
    result = ak.simplify(x - x)
    with session(pool=pool) as s:
        claim = s.record(result, method="simplify")
    if claim.status == "certificate_available":
        assert claim.machine_checked is False
        assert "NOT been machine-checked" in claim.badge


# ---------------------------------------------------------------------------
# Hypotheses travel with the claim
# ---------------------------------------------------------------------------


def test_hypotheses_come_from_the_assumption_context(pool):
    x = pool.symbol("x")
    assumptions = ak.Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    with session(pool=pool, assumptions=assumptions) as s:
        claim = s.record(ak.simplify(x), method="simplify")
    assert claim.hypotheses
    assert any(">" in h for h in claim.hypotheses)


def test_hypotheses_change_the_claim_id(pool):
    x = pool.symbol("x")
    with session(pool=pool) as s:
        bare = s.record(ak.simplify(x), method="simplify")
        conditional = s.record(ak.simplify(x), method="simplify", hypotheses=["x > 0"])
    assert bare.id != conditional.id


# ---------------------------------------------------------------------------
# Automatic capture
# ---------------------------------------------------------------------------


def test_automatic_capture_records_and_infers_edges(pool):
    x = pool.symbol("x")
    with session(title="capture", pool=pool, capture=True) as s:
        integrand = x / (x ** pool.integer(2) + pool.integer(1))
        definite = ak.integrate(integrand, x, pool.integer(0), pool.integer(1))
        ak.simplify(pool.integer(2) * definite.value - ak.log(pool.integer(2)))
    graph = s.graph
    assert len(graph) == 2
    assert captured_operations()
    assert not s.capture_errors
    tail = graph.claims[-1]
    assert tail.depends_on == (graph.claims[0].id,)
    assert graph.impact(graph.claims[0].id) == (tail.id,)


def test_capture_is_off_by_default(pool):
    x = pool.symbol("x")
    with session(pool=pool) as s:
        ak.integrate(x, x)
    assert len(s.graph) == 0


def test_capture_does_not_record_the_recorders_own_calls(pool):
    x = pool.symbol("x")
    with session(pool=pool, capture=True) as s:
        ak.integrate(x ** pool.integer(2), x)
    # Normalisation calls ``simplify`` internally; only one claim must appear.
    assert len(s.graph) == 1


def test_capture_stops_at_the_end_of_the_block(pool):
    x = pool.symbol("x")
    with session(pool=pool, capture=True) as s:
        ak.integrate(x, x)
    before = len(s.graph)
    ak.integrate(x, x)
    assert len(s.graph) == before


def test_concurrent_capture_sessions_do_not_cross_contaminate():
    # The hooks are process-global and permanent; only the thread-local session
    # stack decides what gets recorded, so parallel loops must stay separate.
    results: dict[str, tuple[int, list[str]]] = {}

    def work(tag: str, count: int) -> None:
        local_pool = ak.ExprPool()
        x = local_pool.symbol("x")
        with session(title=tag, pool=local_pool, capture=True) as s:
            for power in range(1, count + 1):
                ak.integrate(x ** local_pool.integer(power), x)
        results[tag] = (len(s.graph), s.capture_errors)

    threads = [threading.Thread(target=work, args=(f"t{i}", i + 2)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {"t0": (2, []), "t1": (3, []), "t2": (4, []), "t3": (5, [])}


def test_run_records_without_capture(pool):
    x = pool.symbol("x")
    with session(pool=pool) as s:
        result = s.run(ak.integrate, x * ak.sin(x), x)
    assert result.verification["status"] == "exactly_verified"
    assert len(s.graph) == 1
    assert s.graph.claims[0].method == "integrate"


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def _claim(name: str, deps=(), status: str = "unverified") -> Claim:
    return Claim(
        id=claim_id(name, (), "test"),
        statement=name,
        kind="text",
        method="test",
        status=status,
        depends_on=tuple(deps),
    )


def test_unknown_dependency_is_rejected():
    graph = ClaimGraph()
    with pytest.raises(MissingClaimError):
        graph.add(_claim("b", deps=("clm_does_not_exist",)))


def test_self_edges_are_dropped():
    graph = ClaimGraph()
    claim = _claim("a")
    graph.add(Claim(**{**claim.__dict__, "depends_on": (claim.id,)}))
    assert graph[claim.id].depends_on == ()


def test_re_recording_merges_edges_instead_of_duplicating():
    graph = ClaimGraph()
    root = graph.add(_claim("root"))
    other = graph.add(_claim("other"))
    graph.add(_claim("leaf", deps=(root.id,)))
    graph.add(_claim("leaf", deps=(other.id,)))
    leaf = graph[claim_id("leaf", (), "test")]
    assert len(graph) == 3
    assert set(leaf.depends_on) == {root.id, other.id}


def test_impact_and_ancestors_are_transitive():
    graph = ClaimGraph()
    a = graph.add(_claim("a"))
    b = graph.add(_claim("b", deps=(a.id,)))
    c = graph.add(_claim("c", deps=(b.id,)))
    assert graph.impact(a.id) == (b.id, c.id)
    assert graph.ancestors(c.id) == (a.id, b.id)
    assert graph.roots() == (a,)
    assert graph.leaves() == (c,)


def test_queries():
    graph = ClaimGraph()
    graph.add(_claim("a", status="exactly_verified"))
    graph.add(_claim("b"))
    assert len(graph.by_status("exactly_verified")) == 1
    assert len(graph.machine_checkable()) == 1
    assert len(graph.unverified()) == 1
    assert graph.summary() == {"exactly_verified": 1, "unverified": 1}
    assert len(graph.by_method("test")) == 2


def test_cycle_in_serialised_graph_is_rejected():
    first = claim_id("a", (), "test")
    second = claim_id("b", (), "test")
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "alkahest.claim_graph",
        "title": None,
        "metadata": {},
        "claims": [
            {"id": first, "statement": "a", "method": "test", "depends_on": [second]},
            {"id": second, "statement": "b", "method": "test", "depends_on": [first]},
        ],
    }
    with pytest.raises(CycleError):
        ClaimGraph.from_dict(document)


def test_future_schema_version_is_refused():
    with pytest.raises(ClaimGraphError):
        ClaimGraph.from_dict(
            {"schema_version": SCHEMA_VERSION + 1, "kind": "alkahest.claim_graph", "claims": []}
        )


def test_foreign_document_is_refused():
    with pytest.raises(ClaimGraphError):
        ClaimGraph.from_dict({"kind": "something.else", "claims": []})


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

_statements = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=24
)
_statuses = st.sampled_from(
    ["exactly_verified", "certificate_available", "numerically_checked", "unverified"]
)


@st.composite
def _graphs(draw):
    names = draw(st.lists(_statements, min_size=1, max_size=6, unique=True))
    graph = ClaimGraph(title=draw(st.one_of(st.none(), _statements)))
    placed: list[str] = []
    for name in names:
        deps = draw(st.lists(st.sampled_from(placed), max_size=2)) if placed else []
        claim = Claim(
            id=claim_id(name, (), "test"),
            statement=name,
            kind="text",
            method="test",
            status=draw(_statuses),
            hypotheses=tuple(draw(st.lists(_statements, max_size=2, unique=True))),
            depends_on=tuple(dict.fromkeys(deps)),
            tags=tuple(draw(st.lists(_statements, max_size=2, unique=True))),
        )
        graph.add(claim)
        placed.append(claim.id)
    return graph


def test_merge_cannot_close_a_cycle():
    """Re-adding a claim must not be able to make the graph cyclic.

    Claim IDs are content-addressed over the *normalised* statement, so "a" and
    " a" are different strings with the same ID. Re-adding one takes the merge
    path, which unions in its dependency edges -- and those can point at claims
    recorded later, including ones that already depend on it.

    That produced a graph which served fine in memory and serialised fine, but
    could never be read back: `from_json` topologically sorts and raised
    CycleError. Found by the round-trip property test below.
    """
    graph = ClaimGraph()
    a = Claim(
        id=claim_id("a", (), "test"),
        statement="a",
        kind="text",
        method="test",
        status="unverified",
    )
    graph.add(a)
    b = Claim(
        id=claim_id("b", (), "test"),
        statement="b",
        kind="text",
        method="test",
        status="unverified",
        depends_on=(a.id,),
    )
    graph.add(b)

    # " a" normalises to "a", so this merges into `a` and would add a -> b.
    colliding = Claim(
        id=claim_id(" a", (), "test"),
        statement=" a",
        kind="text",
        method="test",
        status="unverified",
        depends_on=(b.id,),
    )
    assert colliding.id == a.id, "precondition: normalisation collapses the IDs"

    with pytest.raises(CycleError) as excinfo:
        graph.add(colliding)
    assert a.id in str(excinfo.value)
    assert b.id in str(excinfo.value)

    # The graph is untouched and still round-trips.
    assert graph.ids == (a.id, b.id)
    assert ClaimGraph.from_json(graph.to_json()).to_dict() == graph.to_dict()


def test_merge_still_unions_edges_when_acyclic():
    """The cycle guard must not break legitimate merging."""
    graph = ClaimGraph()
    base = Claim(
        id=claim_id("base", (), "test"),
        statement="base",
        kind="text",
        method="test",
        status="unverified",
    )
    other = Claim(
        id=claim_id("other", (), "test"),
        statement="other",
        kind="text",
        method="test",
        status="unverified",
    )
    graph.add(base)
    graph.add(other)

    dependent = Claim(
        id=claim_id("dependent", (), "test"),
        statement="dependent",
        kind="text",
        method="test",
        status="unverified",
        depends_on=(base.id,),
    )
    graph.add(dependent)
    # Re-derive the same claim, now also citing `other` -- acyclic, so allowed.
    graph.add(replace(dependent, depends_on=(other.id,)))

    assert set(graph[dependent.id].depends_on) == {base.id, other.id}
    assert ClaimGraph.from_json(graph.to_json()).to_dict() == graph.to_dict()


@settings(max_examples=60, deadline=None)
@given(_graphs())
def test_json_round_trip_is_lossless(graph):
    restored = ClaimGraph.from_json(graph.to_json())
    assert restored.to_dict() == graph.to_dict()
    assert restored.digest() == graph.digest()
    assert restored.ids == graph.ids


@settings(max_examples=40, deadline=None)
@given(_graphs())
def test_stable_json_has_no_volatile_fields(graph):
    text = graph.to_json(stable=True)
    document = json.loads(text)
    assert document["schema_version"] == SCHEMA_VERSION
    for entry in document["claims"]:
        assert "recorded_at" not in entry
    # Deterministic key order, so a byte diff is a content diff.
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2, ensure_ascii=False)


def test_two_runs_of_the_same_computation_agree_except_on_metadata():
    def run():
        local_pool = ak.ExprPool()
        x = local_pool.symbol("x")
        with session(title="repeat", pool=local_pool, capture=True) as s:
            ak.integrate(x ** local_pool.integer(2), x)
        return s.graph

    first, second = run(), run()
    assert first.to_json(stable=True) == second.to_json(stable=True)
    assert first.digest() == second.digest()
    assert first.to_json() != second.to_json() or first.metadata == second.metadata


def test_save_and_load_round_trip(tmp_path, pool):
    x = pool.symbol("x")
    with session(title="disk", pool=pool, capture=True) as s:
        ak.integrate(x * ak.sin(x), x)
    path = tmp_path / "graph.json"
    s.graph.save(path)
    restored = ClaimGraph.load(path)
    assert restored.to_dict() == s.graph.to_dict()
    assert restored.claims[0].certificate == s.graph.claims[0].certificate


def test_iteration_n_plus_one_cites_iteration_n(pool):
    x = pool.symbol("x")
    with session(title="run 1", pool=pool, capture=True) as first:
        ak.integrate(x * ak.sin(x), x)
    earlier = first.graph.ids[0]

    carried = ClaimGraph.from_json(first.graph.to_json())
    with session(pool=pool, capture=True, graph=carried) as second:
        second.cite(earlier)
        ak.simplify(x + x)
    assert len(second.graph) == 2
    assert earlier in second.graph.claims[-1].depends_on


# ---------------------------------------------------------------------------
# Re-verification
# ---------------------------------------------------------------------------


def test_verify_rechecks_an_antiderivative(pool):
    x = pool.symbol("x")
    with session(pool=pool, capture=True) as s:
        ak.integrate(x * ak.sin(x), x)
    report = s.graph.verify()
    assert report.ok
    assert report.summary().get("ok") == 1
    assert "Re-verification" in report.to_markdown()


def test_verify_refutes_a_tampered_claim(pool):
    x = pool.symbol("x")
    with session(pool=pool, capture=True) as s:
        ak.integrate(x * ak.sin(x), x)
    graph = ClaimGraph.from_json(s.graph.to_json())
    claim = graph.claims[0]
    tampered = dict(claim.check)
    tampered["antiderivative"] = "x^3"
    graph._replace_claim(type(claim)(**{**claim.__dict__, "check": tampered}))
    report = graph.verify()
    assert not report.ok
    assert report.failed[0].outcome == "failed"
    assert graph[claim.id].status == "refuted"


def test_verify_never_upgrades_a_status(pool):
    x = pool.symbol("x")
    with session(pool=pool, capture=True) as s:
        ak.simplify(pool.integer(2) * x - x - x)
    graph = s.graph
    before = {c.id: c.status for c in graph}
    report = graph.verify()
    after = {c.id: c.status for c in graph}
    assert report.ok
    for cid, status in before.items():
        assert after[cid] == status
    assert all(c.audit for c in graph)


def test_verify_skips_claims_without_a_recipe():
    graph = ClaimGraph()
    graph.add(_claim("no recipe"))
    report = graph.verify()
    assert report.summary() == {"skipped": 1}
    assert report.ok


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_markdown_does_not_dress_an_unverified_claim_as_certified(pool):
    x = pool.symbol("x")
    definite = ak.integrate(
        x / (x ** pool.integer(2) + pool.integer(1)), x, pool.integer(0), pool.integer(1)
    )
    with session(title="honest", pool=pool) as s:
        s.record(definite, method="integrate", label="Definite integral")
    document = s.graph.to_markdown()
    assert "[UNVERIFIED]" in document
    assert "[VERIFIED]" not in document
    assert "Machine-checkable subset: 0 of 1 claims" in document
    assert "Definite integral" in document


def test_markdown_flags_emitted_but_unchecked_certificates(pool):
    x = pool.symbol("x")
    with session(pool=pool, capture=True) as s:
        ak.simplify(pool.integer(2) * x - x - x)
    document = s.graph.to_markdown()
    if s.graph.claims[0].certificate:
        assert "not machine-checked" in document


def test_markdown_links_dependencies(pool):
    x = pool.symbol("x")
    with session(pool=pool, capture=True) as s:
        definite = ak.integrate(
            x / (x ** pool.integer(2) + pool.integer(1)), x, pool.integer(0), pool.integer(1)
        )
        ak.simplify(pool.integer(2) * definite.value - ak.log(pool.integer(2)))
    document = s.graph.to_markdown()
    root = s.graph.ids[0]
    assert f'<a id="{root}"></a>' in document
    assert f"[`{root}`](#{root})" in document


def test_latex_renders_a_standalone_document(pool):
    x = pool.symbol("x")
    with session(title="Report & notes", pool=pool, capture=True) as s:
        ak.integrate(x * ak.sin(x), x)
    document = s.graph.to_latex()
    assert document.startswith("\\documentclass")
    assert "\\end{document}" in document
    assert r"Report \& notes" in document
    assert f"\\label{{clm:{s.graph.ids[0]}}}" in document
    body = s.graph.to_latex(standalone=False)
    assert not body.startswith("\\documentclass")


def test_renderers_handle_an_empty_graph():
    graph = ClaimGraph(title="Nothing yet")
    assert "No claims recorded" in graph.to_markdown()
    assert graph.to_latex().startswith("\\documentclass")
