"""D8: ResearchSession cross-check policy — evidence, never a verdict."""

from __future__ import annotations

import alkahest as ak
import pytest

sympy = pytest.importorskip("sympy")


def _session(**kw):
    pool = ak.ExprPool()
    return pool, ak.research.session(pool=pool, capture=True, **kw)


def test_crosscheck_off_by_default_records_nothing():
    pool, sess = _session()
    x = pool.symbol("x")
    with sess as s:
        ak.integrate(x**2, x)
    claim = s.graph.claims[0]
    assert "crosscheck" not in claim.verification
    assert not any(t.startswith("crosscheck:") for t in claim.tags)


def test_crosscheck_on_attaches_outcome_and_tag():
    pool, sess = _session(crosscheck=True)
    x = pool.symbol("x")
    with sess as s:
        ak.integrate(x**2, x)
    claim = s.graph.claims[0]
    rec = claim.verification["crosscheck"]
    assert rec["outcome"] in ak.crosscheck.OUTCOMES
    assert f"crosscheck:{rec['outcome']}" in claim.tags


def test_agreement_never_upgrades_status():
    """The load-bearing invariant: an oracle agreeing is not a proof."""
    pool, sess = _session(crosscheck=True)
    x = pool.symbol("x")
    with sess as s:
        result = ak.integrate(x**2, x)
    claim = s.graph.claims[0]
    assert claim.status == str((result.verification or {}).get("status", "unverified"))
    assert claim.verification["crosscheck"]["outcome"] == "agree"
    # Agreement must not make the claim machine-checked.
    if claim.status not in ak.research.MACHINE_CHECKED_STATUSES:
        assert claim.machine_checked is False


def test_divergence_does_not_refute():
    """A divergence names two suspects (D5), so it must not set 'refuted'."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    with ak.research.session(pool=pool) as s:
        stub = (
            ak.crosscheck.CrossCheck(operation="integrate", outcome="diverge", oracle="sympy")
            if hasattr(ak.crosscheck, "CrossCheck")
            else None
        )
        if stub is None:
            pytest.skip("CrossCheck not constructible directly")
        claim = s.record(x**2, statement=x**2, method="integrate", crosscheck=stub)
    assert claim.status != "refuted"
    assert claim.verification["crosscheck"]["outcome"] == "diverge"
    assert "crosscheck:diverge" in claim.tags


def test_unavailable_oracle_is_not_agreement():
    """The one thing that must never happen: absent oracle reading as agree."""
    rec = ak.research._crosscheck_record(
        type("S", (), {"outcome": "unavailable", "conclusive": False, "checked": False})()
    )
    assert rec["outcome"] == "unavailable"
    assert rec["checked"] is False


def test_crosscheck_failure_degrades_to_incomparable_not_agree():
    """A cross-check that cannot even be posed is never recorded as agreement."""
    bad = ak.research._run_crosscheck("integrate", ("not-an-expr", "nope"), {})
    assert bad is not None
    assert ak.research._crosscheck_record(bad)["outcome"] == "incomparable"


def test_claim_with_crosscheck_survives_json_round_trip():
    pool, sess = _session(crosscheck=True)
    x = pool.symbol("x")
    with sess as s:
        ak.integrate(x**2, x)
    restored = ak.research.ClaimGraph.from_json(s.graph.to_json())
    assert restored.claims[0].verification["crosscheck"]["outcome"] in ak.crosscheck.OUTCOMES
