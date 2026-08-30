"""Certificate ledger — soundness, drift, and reconciliation tests.

The predicate under test makes a promise an agent will plan against: if
``certifiable(...)`` is truthy, the certificate is really there. These tests are
what makes that promise trustworthy, so they are deliberately stronger than
"the happy path works":

* the **property test** re-runs the whole generating corpus and asserts the
  predicate agrees with reality on every single observation;
* the **drift test** regenerates the checked-in table and fails if it moved;
* the **reconciliation tests** pin ``capabilities()`` to the same ledger, so the
  capability bits cannot advertise a theorem the emitter cannot prove.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import alkahest as ak
import pytest
from alkahest._certificates import (
    OPERATIONS,
    SCHEMA_VERSION,
    _ledger,
    canonical_expression,
    certifiable_primitives,
    classify,
    coverage_summary,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))


@pytest.fixture
def pool():
    return ak.ExprPool()


# ---------------------------------------------------------------------------
# The predicate: positive and negative directions
# ---------------------------------------------------------------------------


def test_certifiable_true_and_the_certificate_really_is_produced(pool):
    """The positive direction: `True` is always backed by an actual certificate."""
    x = pool.symbol("x")
    answer = ak.certifiable("diff", ak.sin(x), x)

    assert bool(answer) is True
    assert answer.reason == "emitted"
    assert answer.checked is True
    # The result is handed back, so establishing certifiability costs nothing extra.
    assert answer.result is not None
    assert answer.result.certificate is not None
    assert "sorry" not in answer.result.certificate

    # ... and running it independently agrees.
    assert ak.diff(ak.sin(x), x).certificate is not None


def test_certifiable_false_and_the_operation_really_yields_nothing(pool):
    """The negative direction: `False` on a shape that genuinely never certifies."""
    x = pool.symbol("x")
    answer = ak.certifiable("integrate", ak.log(x), x)

    assert bool(answer) is False
    assert answer.verdict == "withheld"
    assert answer.reason == "class_withheld"

    # The prediction is correct: the operation really does withhold.
    assert ak.integrate(ak.log(x), x).certificate is None


def test_certifiable_reports_the_blocking_rewrite_rule(pool):
    """A bare `False` is much less useful to an agent than a reason."""
    x = pool.symbol("x")
    answer = ak.certifiable("diff", ak.log(ak.sin(x)), x)

    assert bool(answer) is False
    assert answer.reason == "withheld_uncertifiable_step"
    assert "diff_log" in answer.detail
    assert ak.diff(ak.log(ak.sin(x)), x).certificate is None


def test_certifiable_reports_a_failing_operation_rather_than_raising(pool):
    """Asking "is this certifiable?" must never blow up in the caller's face.

    The probe used to be ``diff(gamma(x))``, which had no derivative rule at
    all; 3.10.0 gave it one (``Γ′ = Γ·ψ``), so the example moved to
    ``trigamma`` — the rung where the polygamma ladder deliberately stops
    (``ψ₁′ = ψ₂``, with no closed-form terminator short of a binary
    ``polygamma(n, x)``).  If that ever gains a derivative, move this probe
    again rather than deleting the test: the property under test is that a
    *failing* operation is reported, not raised.
    """
    x = pool.symbol("x")
    answer = ak.certifiable("diff", ak.trigamma(x), x)

    assert bool(answer) is False
    assert answer.reason == "operation_failed"
    assert isinstance(answer.error, ak.DiffError)


def test_certifiable_accepts_a_callable_as_well_as_a_name(pool):
    x = pool.symbol("x")
    assert bool(ak.certifiable(ak.diff, ak.cos(x), x)) is True
    assert bool(ak.certifiable("diff", ak.cos(x), x)) is True


def test_ledger_mode_runs_nothing(pool, monkeypatch):
    """`mode="ledger"` is the zero-compute planning path: it must not call the op."""
    x = pool.symbol("x")

    def explode(*_args, **_kwargs):  # pragma: no cover — must never be reached
        raise AssertionError("ledger mode must not evaluate the operation")

    monkeypatch.setattr(ak, "integrate", explode)
    answer = ak.certifiable("integrate", ak.log(x), x, mode="ledger")
    assert bool(answer) is False
    assert answer.checked is False
    assert answer.result is None


def test_unknown_shape_under_claims(pool):
    """A shape the corpus never reached answers False, not True."""
    x = pool.symbol("x")
    y = pool.symbol("y")
    answer = ak.certifiable("diff", ak.bessel_j0(x * y), x, mode="ledger")
    assert bool(answer) is False
    assert answer.verdict == "unknown"
    assert answer.reason == "unknown_shape"


def test_certifiable_rejects_untracked_operations_and_modes(pool):
    x = pool.symbol("x")
    with pytest.raises(ValueError, match="unknown operation"):
        ak.certifiable("factor", x)
    with pytest.raises(ValueError, match="mode"):
        ak.certifiable("diff", x, x, mode="guess")


# ---------------------------------------------------------------------------
# The property test — the reason to trust the predicate at all
# ---------------------------------------------------------------------------


def test_predicate_agrees_with_reality_over_the_whole_corpus():
    """`certifiable(op, args) == (result.certificate is not None)`, everywhere.

    Re-runs the generating corpus (the strict Lean corpus plus the textbook
    gate) and checks the ledger-only prediction against what the emitter
    actually did on every observation. Two properties are asserted:

    1. **Soundness** (hard): the predicate never claims `True` for an
       observation that did not emit. This is the promise agents plan against.
    2. **Exactness**: outside `conditional` classes — where emission depends on
       whether the operation rewrote anything, which arguments alone cannot
       predict — prediction and reality agree exactly.
    """
    import certificate_corpus

    index = _ledger()["_index"]
    observations = certificate_corpus.collect()
    assert observations, "corpus produced no observations"

    unsound = []
    inexact = []
    conditional = 0
    for observation in observations:
        row = index.get(observation.shape)
        verdict = row["verdict"] if row else "unknown"
        predicted = verdict == "certified"
        actual = observation.outcome == certificate_corpus.CERTIFIED

        if predicted and not actual:
            unsound.append((observation.shape, observation.expression))
        if verdict == "conditional":
            conditional += 1
        elif predicted != actual:
            inexact.append((observation.shape, observation.expression, verdict))

    assert not unsound, (
        f"certifiable() over-claimed on {len(unsound)} observation(s): {unsound[:5]}"
    )
    assert not inexact, f"prediction disagreed with reality: {inexact[:5]}"
    # `conditional` classes are the honest residue; keep them a rounding error.
    assert conditional < 0.1 * len(observations)


def test_verify_mode_is_exact_on_a_certifiable_and_an_uncertifiable_route(pool):
    """Spot-check the sound mode end to end on both sides of the line."""
    x = pool.symbol("x")
    for op, args in (
        ("diff", (ak.sin(x), x)),
        ("diff", (ak.log(ak.sin(x)), x)),
        ("integrate", (ak.cos(x), x)),
        ("integrate", (ak.log(x), x)),
        ("simplify", (x + pool.integer(0),)),
    ):
        answer = ak.certifiable(op, *args)
        actual = getattr(ak, op)(*args).certificate is not None
        assert bool(answer) is actual, f"{op}{args}: predicted {bool(answer)}, actual {actual}"


# ---------------------------------------------------------------------------
# The coverage table
# ---------------------------------------------------------------------------


def test_coverage_table_is_structured_and_queryable():
    rows = ak.certificate_coverage()
    assert rows, "coverage table is empty"
    assert {row["operation"] for row in rows} <= set(OPERATIONS)
    for row in rows:
        assert {
            "operation",
            "shape",
            "features",
            "verdict",
            "observations",
            "blocking_rules",
            "examples",
        } == row.keys()
        assert row["verdict"] in ("certified", "conditional", "partial", "withheld")
        assert row["shape"].startswith(row["operation"] + "/")

    integrate_rows = ak.certificate_coverage("integrate")
    assert integrate_rows
    assert all(row["operation"] == "integrate" for row in integrate_rows)
    assert len(integrate_rows) < len(rows)


def test_coverage_table_covers_both_sides_of_the_line():
    """A table that only recorded successes would be worthless."""
    verdicts = {row["verdict"] for row in ak.certificate_coverage()}
    assert "certified" in verdicts
    assert "withheld" in verdicts


def test_no_partial_shape_classes():
    """A `partial` class means the feature vector cannot separate certified from
    withheld observations. It is sound (the predicate under-claims) but it is a
    signal the classifier needs another feature — fail loudly if one appears."""
    partial = [row for row in ak.certificate_coverage() if row["verdict"] == "partial"]
    assert not partial, f"shape classes with mixed outcomes: {[r['shape'] for r in partial]}"


def test_ledger_artifact_is_checked_in_and_current_schema():
    path = os.path.join(REPO, "python", "alkahest", "certificate_ledger.json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["corpus"]["sources"] == [
        "tests/lean_corpus.py",
        "tests/lean_tendsto_corpus.py",
        "tests/textbook_gate/",
    ]
    assert data["corpus"]["observations"] > 0


def test_generated_table_does_not_drift_from_a_regeneration():
    """The auditable half: the checked-in table must equal what running the
    corpus produces right now. If the emitter's certifiable surface moves and
    the ledger is not regenerated in the same commit, this fails."""
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "gen_certificate_ledger.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, (
        "certificate ledger is stale — run "
        f"`python scripts/gen_certificate_ledger.py`\n{result.stdout}\n{result.stderr}"
    )


def test_ledger_examples_are_rendered_reproducibly(pool):
    """A drift check is only meaningful if regeneration is reproducible.

    ``str(expr)`` is not: the kernel orders commutative operands using a
    randomly-seeded hasher, so the same sum prints differently between
    processes. The ledger therefore renders examples canonically. This pins
    that — a regression here would turn CI into a coin flip.
    """
    x = pool.symbol("x")
    y = pool.symbol("y")
    expression = ak.sin(x) * y + ak.cos(x) * y
    assert canonical_expression(expression) == canonical_expression(ak.cos(x) * y + ak.sin(x) * y)
    # Commutative operands come out sorted, so the rendering is a property of
    # the expression rather than of the run that built it.
    assert canonical_expression(x + y) == canonical_expression(y + x)

    checked_in = {row["shape"]: row["examples"] for row in ak.certificate_coverage()}
    assert any(checked_in.values()), "no examples recorded in the ledger"


def test_one_plus_sq_pow_base_splits_the_arctan_ftc_atom(pool):
    """∫(1+x²)⁻¹ certifies; ∫(4+x²)⁻¹ does not. Those must not share a class."""
    x = pool.symbol("x")
    _, inv_one = classify("integrate", (1 / (1 + x**2), x))
    _, inv_four = classify("integrate", (1 / (4 + x**2), x))
    assert inv_one["pow_base"] == "one_plus_sq"
    assert inv_four["pow_base"] == "expr"
    assert inv_one != inv_four


def test_generated_markdown_page_exists_and_is_marked_generated():
    path = os.path.join(REPO, "docs", "mdbook", "src", "certificate-coverage.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert "GENERATED FILE" in text
    assert "certifiable" in text


# ---------------------------------------------------------------------------
# Reconciliation with capabilities()
# ---------------------------------------------------------------------------


def test_capabilities_lean_theorem_bits_come_from_the_ledger(pool):
    """One source of truth. Every `lean_theorem: true` primitive must actually
    emit a certificate for `d/dx f(x)` — this is precisely the overclaim that
    once advertised `log`, `tan`, and `gamma` as provable when they were not."""
    claiming = {row["name"] for row in ak.capabilities()["primitives"] if row["lean_theorem"]}
    assert claiming == certifiable_primitives()
    assert claiming, "no primitive claims lean_theorem — the ledger is probably empty"

    x = pool.symbol("x")
    for name in claiming:
        certificate = ak.diff(getattr(ak, name)(x), x).certificate
        assert certificate, f"{name}: lean_theorem=True but no certificate is emitted"
        assert "sorry" not in certificate, f"{name}: certificate contains sorry"
        assert "admit" not in certificate, f"{name}: certificate contains admit"


def test_native_capability_bit_agrees_with_the_ledger():
    """The Rust `Capabilities::LEAN_THEOREM` bit is no longer what `capabilities()`
    reports, but the two must not silently diverge: if the emitter gains or loses
    a primitive, both sides should move together."""
    registry_rows = ak.PrimitiveRegistry.default_registry().coverage_report()
    native = {row["name"] for row in registry_rows if row["lean_theorem"]}
    assert native == certifiable_primitives(), (
        "alkahest-core's Primitive::lean_theorem overrides disagree with the observed "
        f"ledger: native={sorted(native)}, ledger={sorted(certifiable_primitives())}"
    )


def test_capabilities_advertises_only_statuses_it_can_emit(pool):
    """`lean_checked` used to be advertised here and no code path ever produced
    it — checking is out of process by construction. Advertising a status the
    library cannot emit is the same class of overclaim as a false capability bit."""
    verification = ak.capabilities()["verification"]
    assert verification["statuses"] == [
        "certificate_available",
        "exactly_verified",
        # `smt.solve` emits this for an external `unsat`; see
        # tests/test_smt.py::test_unsat_is_externally_asserted_and_never_machine_checked
        "externally_asserted",
        "numerically_checked",
        "unverified",
    ]
    assert "lean_checked" not in verification["statuses"]
    assert verification["checkers"] == {"lean4": "external", "smt": "external"}

    x = pool.symbol("x")
    for result in (
        ak.simplify(x + pool.integer(0)),
        ak.diff(ak.sin(x), x),
        ak.integrate(ak.cos(x), x),
    ):
        assert result.verification["status"] in verification["statuses"]


def test_capabilities_carries_the_coverage_summary():
    coverage = ak.capabilities()["verification"]["coverage"]
    assert coverage == coverage_summary()
    assert coverage["schema_version"] == SCHEMA_VERSION
    assert coverage["shape_classes"]["certified"] > 0
    assert coverage["shape_classes"]["withheld"] > 0
    assert "diff" in coverage["operations"]


# ---------------------------------------------------------------------------
# require_certificate
# ---------------------------------------------------------------------------


def test_require_certificate_per_call_returns_or_raises(pool):
    x = pool.symbol("x")
    derived = ak.diff(ak.sin(x), x)
    assert ak.require_certificate(derived) is derived

    with pytest.raises(ak.CertificateUnavailableError) as excinfo:
        ak.require_certificate(ak.integrate(ak.log(x), x))
    assert excinfo.value.code == "E-CERT-001"
    assert excinfo.value.remediation


def test_require_certificate_names_the_blocking_rule(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.CertificateUnavailableError) as excinfo:
        ak.require_certificate(ak.diff(ak.log(ak.sin(x)), x))
    assert "diff_log" in excinfo.value.remediation


def test_ambient_require_certificate_raises_instead_of_degrading(pool):
    x = pool.symbol("x")
    with ak.context(require_certificate=True):
        assert ak.diff(ak.sin(x), x).value is not None
        with pytest.raises(ak.CertificateUnavailableError):
            ak.integrate(ak.log(x), x)

    # Outside the block, the uncertified result is returned as before.
    assert ak.integrate(ak.log(x), x).certificate is None


def test_ambient_require_certificate_can_be_switched_off_in_an_inner_block(pool):
    x = pool.symbol("x")
    with ak.context(require_certificate=True), ak.context(require_certificate=False):
        assert ak.integrate(ak.log(x), x).certificate is None


def test_certifiable_probe_does_not_raise_under_ambient_requirement(pool):
    """A predicate that raised when the answer is "no" would be useless inside
    the very loop that turns the requirement on."""
    x = pool.symbol("x")
    with ak.context(require_certificate=True):
        answer = ak.certifiable("integrate", ak.log(ak.sin(x)), x)
    assert bool(answer) is False


def test_ambient_requirement_does_not_lose_the_surrounding_context(pool):
    """The probe overlays the active context rather than replacing it."""
    x = pool.symbol("x")
    assumptions = ak.Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    with ak.context(pool=pool, assumptions=assumptions):
        answer = ak.certifiable("simplify_log_exp", ak.exp(ak.log(x)))
        assert answer.result is not None
        assert str(answer.result.value) == str(ak.simplify_log_exp(ak.exp(ak.log(x))).value)


def test_every_gated_entry_point_is_a_tracked_operation():
    """The ambient gate and the ledger must cover the same surface."""
    assert set(ak._DERIVATION_ENTRY_POINTS) == set(OPERATIONS)
    for name in ak._DERIVATION_ENTRY_POINTS:
        assert callable(getattr(ak, name))


# ---------------------------------------------------------------------------
# certificate_status (the PyO3 diagnostic the ledger is built on)
# ---------------------------------------------------------------------------


def test_certificate_status_matches_certificate_presence(pool):
    x = pool.symbol("x")
    for result in (
        ak.diff(ak.sin(x), x),
        ak.diff(ak.log(ak.sin(x)), x),
        ak.integrate(ak.cos(x), x),
        ak.integrate(ak.log(x), x),
        ak.simplify(x + pool.symbol("y")),
    ):
        status = result.certificate_status
        assert status["certifiable"] is (result.certificate is not None)
        assert status["reason"] in (
            "emitted",
            "withheld_no_derivation",
            "withheld_integration_shape",
            "withheld_tendsto_shape",
            "withheld_uncertifiable_step",
        )
        if status["certifiable"]:
            assert status["blocking_steps"] == []


def test_certificate_status_reports_an_empty_derivation(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    status = ak.simplify(x + y).certificate_status
    assert status["certifiable"] is False
    assert status["reason"] == "withheld_no_derivation"


def test_shape_classification_is_stable_and_readable(pool):
    x = pool.symbol("x")
    shape, features = classify("diff", (ak.sin(x), x))
    assert shape == "diff/" + "/".join(f"{k}={features[k]}" for k in features)
    assert features["funcs"] == "sin"
    assert features["fn_arg"] == "var"
    assert features["form"] == "apply"

    # Composite arguments land in a different class than pointwise ones.
    composite, _ = classify("diff", (ak.sin(ak.log(x)), x))
    assert composite != shape


def test_neg_one_pow_splits_from_higher_negative_powers(pool):
    """`x⁻¹` and `x⁻²` must not share a shape class: the former's integral
    certifies via FTC/`log`, the latter withholds on `product_rule`."""
    x = pool.symbol("x")
    _, inv = classify("integrate", (x**-1, x))
    _, neg_two = classify("integrate", (x**-2, x))
    assert inv["pow"] == "neg_one"
    assert neg_two["pow"] == "neg"
    assert inv["pow"] != neg_two["pow"]
