"""Cross-CAS differential testing — the harness's own tests.

A tool whose job is detecting disagreement has an unusual failure mode: it can
be *quietly useless*. Every test here is aimed at one of the three ways that
happens, rather than at "the happy path works":

* **Silent non-coverage.** The translator grows a hole when a new node kind
  lands, and expressions containing it stop being checked at all.
  ``test_dispatch_table_covers_every_node_tag`` re-derives the tag set from the
  Rust binding source, so the table cannot drift.
* **Silent unavailability.** SymPy is not installed, ``check`` answers
  something falsy-but-not-alarming, and a loop believes it is cross-checking.
  ``test_absent_oracle_*`` simulates the missing oracle and pins
  ``outcome="unavailable"``.
* **Silent collapse.** ``incomparable`` gets folded into ``agree`` somewhere in
  the ladder. ``test_incomparable_never_reports_agreement`` and the refusal
  tests pin every route that can produce it.

Oracle-dependent tests use ``pytest.importorskip``, matching ``test_oracle.py``:
without SymPy this file skips cleanly rather than failing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import alkahest as ak
import pytest
from alkahest.crosscheck import (
    FROZEN_CORPUS,
    FUNCTION_MAP,
    NODE_TAGS,
    OPERATIONS,
    OUTCOMES,
    PREDICATE_KINDS,
    REFUSED_FUNCTIONS,
    RUNG_NAMES,
    CrossCheck,
    Divergence,
    FrozenCase,
    Oracle,
    SweepReport,
    SymPyOracle,
    SymPyTranslator,
    Translator,
    check,
    oracles,
    run_frozen_corpus,
    sweep,
    to_sympy,
)
from alkahest.exceptions import CrossCheckError

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def pool():
    return ak.ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


def _sympy():
    return pytest.importorskip("sympy")


# ---------------------------------------------------------------------------
# The translation table must not drift
# ---------------------------------------------------------------------------


def _tags_from_binding_source() -> set[str] | None:
    """Re-derive the node tags from ``alkahest-py/src/lib.rs``.

    The tags are emitted by one exhaustive ``match`` over ``ExprData`` inside
    ``PyExpr::node``; Rust's exhaustiveness check guarantees a new variant
    cannot be added without an arm, and this reads the string literal each arm
    produces. Returns ``None`` when the source is not present (an installed
    wheel rather than a checkout), so the test degrades to a skip.
    """
    source_path = REPO / "alkahest-py" / "src" / "lib.rs"
    if not source_path.is_file():
        return None
    text = source_path.read_text(encoding="utf-8")
    start = text.find("fn node(")
    if start == -1:
        return None
    # Brace-match the function body rather than guessing at an end marker.
    open_index = text.find("{", start)
    depth = 0
    end = open_index
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    body = text[open_index : end + 1]
    return set(re.findall(r'"([a-z_0-9]+)"\.into_py\(py\)', body))


def test_dispatch_table_covers_every_node_tag():
    """The translator's table must be total over what ``node()`` can emit.

    This is the anti-drift ratchet. A partial table does not fail loudly — it
    silently stops cross-checking every expression containing the new node,
    which is the worst possible outcome for a tool whose entire job is
    detecting disagreement.
    """
    emitted = _tags_from_binding_source()
    if emitted is None:
        pytest.skip("alkahest-py/src/lib.rs is not present in this environment")
    assert emitted == set(NODE_TAGS), (
        "alkahest.crosscheck.NODE_TAGS has drifted from the tags PyExpr::node emits.\n"
        f"  only in lib.rs:    {sorted(emitted - set(NODE_TAGS))}\n"
        f"  only in NODE_TAGS: {sorted(set(NODE_TAGS) - emitted)}\n"
        "Add the new tag to Translator._DISPATCH with a mapping you can defend, or "
        "to the refusal path — but it must appear explicitly."
    )
    assert set(Translator._DISPATCH) == set(NODE_TAGS)


def test_every_dispatch_handler_exists_on_the_sympy_translator():
    for tag, handler in Translator._DISPATCH.items():
        assert hasattr(SymPyTranslator, handler), f"{tag} -> missing handler {handler}"


def test_predicate_kinds_match_the_binding_source():
    """``PREDICATE_KINDS`` must track the Rust ``PredicateKind`` match."""
    source_path = REPO / "alkahest-py" / "src" / "lib.rs"
    if not source_path.is_file():
        pytest.skip("alkahest-py/src/lib.rs is not present in this environment")
    text = source_path.read_text(encoding="utf-8")
    emitted = set(re.findall(r'PredicateKind::\w+ => "(\w+)"', text))
    assert emitted == set(PREDICATE_KINDS)


def test_unknown_tag_refuses_rather_than_guessing():
    """A tag absent from the table raises ``E-XCHECK-001``, never a best effort."""
    _sympy()

    class Fake:
        @staticmethod
        def node():
            return ["quaternion", 1]

    with pytest.raises(CrossCheckError) as info:
        SymPyOracle().translate(Fake())
    assert info.value.code == "E-XCHECK-001"
    assert "quaternion" in str(info.value)


def test_function_map_and_refusals_are_disjoint():
    assert not set(FUNCTION_MAP) & set(REFUSED_FUNCTIONS)


def test_every_mapped_primitive_resolves_in_sympy():
    """A name in ``FUNCTION_MAP`` that SymPy does not have is a latent refusal."""
    sympy = _sympy()
    missing = [name for name in FUNCTION_MAP.values() if not hasattr(sympy, name)]
    assert not missing, f"FUNCTION_MAP targets absent from SymPy: {missing}"


def test_registry_primitives_are_classified(pool, x):
    """Every registered primitive is either mapped or explicitly refused.

    "Not in either table" is the state that produces a surprise refusal in the
    middle of a sweep; forcing a decision per primitive is what keeps the
    refusals *documented* rather than accidental.
    """
    _sympy()
    names = {row["name"] for row in ak.capabilities()["primitives"]}
    classified = set(FUNCTION_MAP) | set(REFUSED_FUNCTIONS) | {"bessel_j0", "bessel_j1"}
    unclassified = sorted(names - classified)
    assert not unclassified, (
        f"primitives with no entry in FUNCTION_MAP or REFUSED_FUNCTIONS: {unclassified}"
    )


@pytest.mark.parametrize("name", sorted(REFUSED_FUNCTIONS))
def test_refused_primitives_refuse_with_a_reason(pool, name):
    _sympy()
    x = pool.symbol("x")
    with pytest.raises(CrossCheckError) as info:
        to_sympy(pool.func(name, [x]))
    assert info.value.code == "E-XCHECK-001"
    assert REFUSED_FUNCTIONS[name] in str(info.value)


# ---------------------------------------------------------------------------
# Translation is faithful
# ---------------------------------------------------------------------------


def test_translates_the_arithmetic_core(pool, x):
    sympy = _sympy()
    sx = sympy.Symbol("x")
    cases = [
        (pool.integer(-3), sympy.Integer(-3)),
        (pool.rational(2, 5), sympy.Rational(2, 5)),
        (x + pool.integer(1), sx + 1),
        (x * pool.integer(3), 3 * sx),
        (x ** pool.integer(2), sx**2),
        (ak.sin(x) ** pool.integer(2), sympy.sin(sx) ** 2),
        (ak.log(ak.exp(x)), sympy.log(sympy.exp(sx))),
    ]
    for expr, expected in cases:
        assert sympy.simplify(to_sympy(expr) - expected) == 0, str(expr)


def test_translates_reserved_symbols(pool):
    """``∞`` and ``I`` are interned as ordinary symbols and must not stay free."""
    sympy = _sympy()
    assert to_sympy(pool.pos_infinity()) is sympy.oo
    assert to_sympy(pool.imaginary_unit()) is sympy.I
    assert to_sympy(-pool.pos_infinity()) == -sympy.oo


def test_translates_predicates_and_piecewise(pool, x):
    sympy = _sympy()
    sx = sympy.Symbol("x")
    zero = pool.integer(0)
    assert to_sympy(pool.lt(x, zero)) == sympy.StrictLessThan(sx, 0)
    assert to_sympy(pool.pred_and([pool.gt(x, zero), pool.pred_ne(x, zero)])).func is sympy.And
    assert to_sympy(pool.pred_true()) is sympy.true
    branch = ak.piecewise([(pool.lt(x, zero), pool.integer(1))], pool.integer(2))
    assert to_sympy(branch) == sympy.Piecewise((1, sx < 0), (2, True))


def test_translates_big_o(pool, x):
    sympy = _sympy()
    assert to_sympy(pool.big_o(x ** pool.integer(3))) == sympy.O(sympy.Symbol("x") ** 3)


def test_translates_root_sum(pool, x):
    """``RootSum`` maps onto SymPy's ``RootSum``, which is genuinely the same object."""
    sympy = _sympy()
    antiderivative = ak.integrate(x / (x ** pool.integer(4) + pool.integer(1)), x).value
    assert antiderivative.node()[0] == "root_sum"
    assert isinstance(to_sympy(antiderivative), sympy.RootSum)


def test_quantifiers_refuse(pool, x):
    """SymPy has no term-level quantifier, so translating one would be a fiction."""
    _sympy()
    body = pool.gt(x, pool.integer(0))
    for quantified in (pool.forall(x, body), pool.exists(x, body)):
        with pytest.raises(CrossCheckError) as info:
            to_sympy(quantified)
        assert info.value.code == "E-XCHECK-001"


# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------


def test_mappable_assumptions_travel_with_the_expression(pool, x):
    """``x > 0`` must reach SymPy, or the oracle is asked a weaker question."""
    sympy = _sympy()
    assumptions = ak.Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    translated = to_sympy(ak.sqrt(x ** pool.integer(2)), assumptions=assumptions)
    # Under x > 0 SymPy folds sqrt(x**2) to x; without the assumption it cannot.
    assert translated == sympy.Symbol("x", positive=True)
    assert to_sympy(ak.sqrt(x ** pool.integer(2))) != sympy.Symbol("x")


@pytest.mark.parametrize(
    ("build", "flag"),
    [
        (lambda p, s: p.gt(s, p.integer(0)), "positive"),
        (lambda p, s: p.ge(s, p.integer(0)), "nonnegative"),
        (lambda p, s: p.lt(s, p.integer(0)), "negative"),
        (lambda p, s: p.le(s, p.integer(0)), "nonpositive"),
        (lambda p, s: p.pred_ne(s, p.integer(0)), "nonzero"),
    ],
)
def test_each_sign_assumption_maps_to_its_flag(pool, x, build, flag):
    _sympy()
    assumptions = ak.Assumptions(pool)
    assumptions.refine(build(pool, x))
    flags = SymPyTranslator(pytest.importorskip("sympy")).assumption_flags(assumptions)
    assert flags == {"x": {flag: True}}


def test_unmappable_assumption_refuses(pool, x):
    """A relation between two symbols has no per-symbol flag; refuse, do not drop it."""
    _sympy()
    y = pool.symbol("y")
    assumptions = ak.Assumptions(pool)
    assumptions.refine(pool.gt(x, y))
    with pytest.raises(CrossCheckError) as info:
        to_sympy(x, assumptions=assumptions)
    assert info.value.code == "E-XCHECK-001"


def test_unmappable_assumption_yields_incomparable_not_agree(pool, x):
    _sympy()
    y = pool.symbol("y")
    assumptions = ak.Assumptions(pool)
    assumptions.refine(pool.gt(x, y))
    out = check("diff", ak.sin(x), x, assumptions=assumptions)
    assert out.outcome == "incomparable"
    assert out.reason == "untranslatable"
    assert not out.checked


def test_ambient_assumption_context_is_picked_up(pool, x):
    _sympy()
    assumptions = ak.Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    with ak.context(pool=pool, assumptions=assumptions):
        flags = SymPyTranslator(pytest.importorskip("sympy"))
        assert flags.assumption_flags(ak.active_assumptions()) == {"x": {"positive": True}}


# ---------------------------------------------------------------------------
# The absent oracle must be loud
# ---------------------------------------------------------------------------


class _AbsentOracle(Oracle):
    """An oracle that is never available — stands in for SymPy not installed."""

    name = "absent"

    @classmethod
    def available(cls) -> bool:
        return False

    @property
    def version(self) -> str:  # pragma: no cover - never constructed
        raise AssertionError

    def translate(self, expr, *, assumptions=None):  # pragma: no cover
        raise AssertionError

    def lift(self, obj, pool):  # pragma: no cover
        raise AssertionError

    def render(self, obj):  # pragma: no cover
        raise AssertionError

    def canonical(self, obj):  # pragma: no cover
        raise AssertionError

    def supports(self, operation):  # pragma: no cover
        raise AssertionError

    def run(self, operation, args):  # pragma: no cover
        raise AssertionError

    def is_zero(self, obj):  # pragma: no cover
        raise AssertionError

    def diff(self, obj, var):  # pragma: no cover
        raise AssertionError

    def subs(self, obj, bindings):  # pragma: no cover
        raise AssertionError

    def subs_by_name(self, obj, point):  # pragma: no cover
        raise AssertionError

    def to_float(self, obj):  # pragma: no cover
        raise AssertionError


@pytest.fixture
def no_oracles(monkeypatch):
    """Simulate an environment with no oracle installed at all."""
    monkeypatch.setattr(
        "alkahest.crosscheck._ORACLE_CLASSES", {"absent": _AbsentOracle}, raising=True
    )


def test_absent_oracle_reports_unavailable_never_agree(pool, x, no_oracles):
    """The single easiest way for this feature to become actively harmful."""
    out = check("diff", ak.sin(x), x)
    assert out.outcome == "unavailable"
    assert out.reason == "no_oracle"
    assert out.outcome != "agree"
    assert not out.checked
    assert out.rung is None


def test_absent_oracle_is_visible_in_oracles(no_oracles):
    assert oracles() == {"absent": None}


def test_absent_oracle_makes_the_frozen_corpus_skip_not_pass(no_oracles):
    outcomes = run_frozen_corpus()
    assert len(outcomes) == len(FROZEN_CORPUS)
    assert all(result is None for _case, result in outcomes)


def test_absent_oracle_sweep_does_not_pretend_to_have_run(no_oracles):
    report = sweep(cases=4, seed=1)
    assert report.oracle is None
    assert report.counts()["unavailable"] == 4
    assert report.counts()["agree"] == 0
    assert report.findings == ()


def test_sympy_import_failure_raises_e_xcheck_002(monkeypatch):
    """``E-XCHECK-002`` must name the missing oracle, with a remediation."""

    def boom():
        raise CrossCheckError("no oracle", code="E-XCHECK-002", remediation="pip install sympy")

    monkeypatch.setattr("alkahest.crosscheck._import_sympy", boom)
    assert SymPyOracle.available() is False
    with pytest.raises(CrossCheckError) as info:
        SymPyOracle()
    assert info.value.code == "E-XCHECK-002"
    assert info.value.remediation


def test_oracles_reports_installed_version():
    sympy = _sympy()
    assert oracles()["sympy"] == sympy.__version__


# ---------------------------------------------------------------------------
# Outcome vocabulary
# ---------------------------------------------------------------------------


def test_outcome_vocabulary_is_exactly_four_valued():
    assert OUTCOMES == ("agree", "diverge", "incomparable", "unavailable")


def test_crosscheck_has_no_truthiness():
    """``if check(...)`` must not compile into a silent "it agreed"."""
    assert "__bool__" not in CrossCheck.__dict__


def test_checked_is_false_for_the_two_no_signal_outcomes():
    for outcome in ("incomparable", "unavailable"):
        assert not CrossCheck(operation="diff", outcome=outcome).checked
    for outcome in ("agree", "diverge"):
        assert CrossCheck(operation="diff", outcome=outcome).checked


def test_unknown_operation_raises_e_xcheck_003(pool, x):
    with pytest.raises(CrossCheckError) as info:
        check("factor_the_universe", x)
    assert info.value.code == "E-XCHECK-003"
    assert "integrate" in str(info.value.remediation)


def test_every_operation_names_a_real_alkahest_entry_point():
    for name in OPERATIONS:
        assert hasattr(ak, name), f"OPERATIONS[{name!r}] has no alkahest counterpart"


def test_every_operation_declares_rungs_from_the_ladder():
    for name, spec in OPERATIONS.items():
        assert spec.rungs, name
        assert all(rung in RUNG_NAMES for rung in spec.rungs), name


def test_sweep_avoids_operations_with_a_known_non_terminating_path():
    """``limit`` is excluded deliberately, and the exclusion must not be lost.

    ``limit(sqrt(x**2 + x) - x, x, oo)`` does not terminate at this commit, and
    the kernel holds the GIL throughout, so nothing in Python can bound it. The
    sweep is a nightly job; one wedged candidate is one unread report.
    """
    from alkahest.crosscheck import SWEEP_OPERATIONS

    assert "limit" not in SWEEP_OPERATIONS
    assert set(SWEEP_OPERATIONS) <= set(OPERATIONS)


def test_rung_four_leads_wherever_an_invariant_exists():
    """The stated default: the invariant rung goes first when there is one."""
    for name, spec in OPERATIONS.items():
        if spec.invariant is not None:
            assert spec.rungs[0] == 4, f"{name} has an invariant but does not lead with rung 4"


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_integrate_settles_on_the_invariant_rung(pool, x):
    """``sin·cos`` has three standard antiderivatives differing by constants."""
    _sympy()
    with ak.context(pool=pool):
        out = check("integrate", ak.sin(x) * ak.cos(x), x)
    assert out.outcome == "agree"
    assert out.rung == 4
    assert out.rung_name == "invariant"
    assert out.conclusive


def test_simplify_uses_the_value_preserving_invariant(pool, x):
    """SymPy folds ``sin²+cos²`` and Alkahest may not — a strength gap, not a divergence."""
    _sympy()
    with ak.context(pool=pool):
        out = check("simplify", ak.sin(x) ** pool.integer(2) + ak.cos(x) ** pool.integer(2))
    assert out.outcome == "agree"
    assert out.rung == 4


def test_sum_indefinite_uses_the_telescoping_invariant(pool):
    _sympy()
    k = pool.symbol("k")
    with ak.context(pool=pool):
        out = check("sum_indefinite", k ** pool.integer(2), k)
    assert out.outcome == "agree"
    assert out.rung == 4


def test_solve_substitutes_solutions_back(pool, x):
    _sympy()
    with ak.context(pool=pool):
        out = check("solve", [x ** pool.integer(2) - pool.integer(4)], [x])
    assert out.outcome == "agree"
    assert out.rung == 4
    assert "substituted back" in out.detail


def test_series_strips_the_remainder_before_comparing(pool, x):
    _sympy()
    with ak.context(pool=pool):
        out = check("series", ak.exp(x), x, pool.integer(0), 5)
    assert out.outcome == "agree"
    assert out.rung in (1, 2, 3)


def test_diff_settles_without_an_invariant(pool, x):
    _sympy()
    with ak.context(pool=pool):
        out = check("diff", ak.sin(ak.exp(x)), x)
    assert out.outcome == "agree"
    assert out.rung in (1, 2, 3)


def test_rung_three_agreement_is_reported_as_inconclusive():
    """Sampling can only fail to refute; that must show on the record."""
    settled = CrossCheck(operation="diff", outcome="agree", rung=3, conclusive=False)
    assert settled.rung_name == "rigorous_numeric"
    assert not settled.conclusive


def test_a_plus_c_difference_is_never_reported_as_a_divergence(pool, x, monkeypatch):
    """The classic false alarm: two correct antiderivatives differing by a constant.

    Rung 4 normally settles it. This forces rung 4 to come back inconclusive,
    so rungs 1–3 are reached with two answers that genuinely differ pointwise —
    and they must fall through to ``incomparable``, not fire.
    """
    _sympy()
    real_integrate = ak.integrate

    def shifted(expr, var, *bounds, **kwargs):
        return real_integrate(expr, var, *bounds, **kwargs).value + pool.integer(7)

    monkeypatch.setattr(ak, "integrate", shifted)
    monkeypatch.setattr("alkahest.crosscheck._INVARIANTS", {})  # force rung 4 to be inconclusive
    with ak.context(pool=pool):
        out = check("integrate", x ** pool.integer(2), x)

    assert out.outcome != "diverge", out.detail
    assert out.divergence is None


def test_alkahest_refusal_is_not_a_divergence(pool, x):
    """An honest refusal must never be scored against Alkahest."""
    _sympy()
    with ak.context(pool=pool):
        out = check("integrate", ak.exp(x ** pool.integer(2)), x)
    assert out.outcome == "incomparable"
    assert out.reason == "alkahest_refused"
    assert out.divergence is None


def test_oracle_refusal_is_not_a_divergence(pool, x, monkeypatch):
    """An unevaluated ``Integral`` from SymPy is a refusal, not an answer."""
    sympy = _sympy()
    original = sympy.integrate

    def unevaluated(expr, *args, **kwargs):
        return sympy.Integral(expr, *args)

    monkeypatch.setattr(sympy, "integrate", unevaluated)
    try:
        with ak.context(pool=pool):
            out = check("integrate", x ** pool.integer(2), x)
    finally:
        monkeypatch.setattr(sympy, "integrate", original)
    assert out.outcome == "incomparable"
    assert out.reason == "oracle_refused"
    assert out.divergence is None


def test_incomparable_never_reports_agreement(pool, x):
    """Sweep the routes that produce ``incomparable`` and pin every one."""
    _sympy()
    y = pool.symbol("y")
    unmappable = ak.Assumptions(pool)
    unmappable.refine(pool.gt(x, y))
    routes = [
        check("diff", ak.sin(x), x, assumptions=unmappable),
        check("diff", pool.func("heaviside", [x]), x),
        check("integrate", ak.exp(x ** pool.integer(2)), x),
    ]
    for out in routes:
        assert out.outcome == "incomparable", out
        assert out.outcome != "agree"
        assert not out.checked


# ---------------------------------------------------------------------------
# Divergence records name two suspects
# ---------------------------------------------------------------------------


def test_a_divergence_names_two_suspects():
    divergence = Divergence(
        operation="integrate",
        oracle="sympy",
        oracle_version="1.14.0",
        point={"x": 1.25},
        alkahest_value="x**2",
        oracle_value="x**3",
    )
    statement = divergence.statement()
    assert "alkahest" in statement
    assert "sympy" in statement
    assert "x**2" in statement
    assert "x**3" in statement
    assert divergence.support == "unresolved"
    assert not divergence.silent_error_candidate


def test_divergence_wording_never_declares_alkahest_correct():
    """The API must not editorialise; a divergence is evidence about *both*."""
    forbidden = ("alkahest is right", "alkahest is correct", "sympy is wrong")
    for support in ("unresolved", "alkahest_supported", "oracle_supported"):
        divergence = Divergence(
            operation="integrate",
            oracle="sympy",
            oracle_version="1.14.0",
            support=support,
            detail="the oracle answer fails the invariant",
        )
        text = divergence.statement().lower()
        assert not any(phrase in text for phrase in forbidden)


def test_only_a_rigorously_refuted_alkahest_side_is_a_silent_error_candidate():
    """Routing into ``tests/silent_errors/`` must be earned, not assumed."""
    assert not Divergence("i", "sympy", "1", support="unresolved").silent_error_candidate
    assert not Divergence("i", "sympy", "1", support="alkahest_supported").silent_error_candidate
    assert Divergence("i", "sympy", "1", support="oracle_supported").silent_error_candidate


def test_a_planted_wrong_answer_is_caught_and_attributed(pool, x, monkeypatch):
    """End-to-end: break Alkahest's integrator and the harness must say so.

    Substituting a wrong antiderivative is the cheapest available model of a
    silent error, and it exercises the whole escalation — invariant residual,
    rigorous enclosure, ``support="oracle_supported"``, witness point.
    """
    _sympy()
    real_integrate = ak.integrate

    def wrong(expr, var, *bounds, **kwargs):
        result = real_integrate(expr, var, *bounds, **kwargs)
        return result.value + pool.integer(1) * var  # F + x: derivative is off by one

    monkeypatch.setattr(ak, "integrate", wrong)
    with ak.context(pool=pool):
        out = check("integrate", x ** pool.integer(2), x)

    assert out.outcome == "diverge"
    assert out.rung == 4
    assert out.divergence is not None
    assert out.divergence.support == "oracle_supported"
    assert out.divergence.silent_error_candidate
    assert out.witness is not None
    assert out.witness["point"]
    assert out.alkahest_value
    assert out.oracle_value


def test_a_planted_wrong_oracle_is_attributed_to_the_oracle(pool, x, monkeypatch):
    """The mirror image: the harness must be able to blame the oracle too."""
    sympy = _sympy()
    original = sympy.integrate

    def wrong(expr, *args, **kwargs):
        return original(expr, *args, **kwargs) + sympy.Symbol("x")

    monkeypatch.setattr(sympy, "integrate", wrong)
    try:
        with ak.context(pool=pool):
            out = check("integrate", x ** pool.integer(2), x)
    finally:
        monkeypatch.setattr(sympy, "integrate", original)

    assert out.outcome == "diverge"
    assert out.divergence is not None
    assert out.divergence.support == "alkahest_supported"
    assert not out.divergence.silent_error_candidate


def test_a_planted_missing_solution_is_caught(pool, x, monkeypatch):
    """``solve`` rung 4: every returned solution verifies, but one is missing."""
    _sympy()
    real_solve = ak.solve

    def truncated(equations, variables, **kwargs):
        return real_solve(equations, variables, **kwargs)[:1]

    monkeypatch.setattr(ak, "solve", truncated)
    with ak.context(pool=pool):
        out = check("solve", [x ** pool.integer(2) - pool.integer(4)], [x])

    assert out.outcome == "diverge"
    assert out.reason == "solution_set_size_differs"
    assert out.divergence is not None
    assert out.divergence.support == "unresolved"


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------


def test_checks_are_deterministic_under_a_fixed_seed(pool, x):
    _sympy()
    with ak.context(pool=pool):
        first = check("integrate", ak.sin(x) * ak.cos(x), x, seed=99)
        second = check("integrate", ak.sin(x) * ak.cos(x), x, seed=99)
    assert first.as_dict() == second.as_dict()


def test_sweeps_are_reproducible_from_their_recorded_seed():
    _sympy()
    first = sweep(cases=8, seed=4242)
    second = sweep(cases=8, seed=first.seed)
    assert first.seed == 4242
    assert [c.as_dict() for c in first.checks] == [c.as_dict() for c in second.checks]


def test_sweep_takes_its_seed_from_the_active_budget():
    """D6's nightly sweep is driven by ``budget_seed``, not a private RNG."""
    _sympy()
    with ak.context(budget=ak.Budget(seed=31337)):
        report = sweep(cases=2)
    assert report.seed == 31337


def test_sweep_report_counts_are_total_over_the_vocabulary():
    _sympy()
    report = sweep(cases=6, seed=11)
    counts = report.counts()
    assert set(counts) == set(OUTCOMES)
    assert sum(counts.values()) == len(report.checks)
    assert isinstance(report, SweepReport)
    assert str(report.seed) in report.summary()
    assert report.oracle_version in report.summary()


def test_sweep_findings_and_candidates_are_subsets():
    _sympy()
    report = sweep(cases=10, seed=5)
    assert all(c.outcome == "diverge" for c in report.findings)
    assert set(report.silent_error_candidates) <= set(report.findings)


# ---------------------------------------------------------------------------
# The frozen corpus (tier 2 — the per-PR gate)
# ---------------------------------------------------------------------------


def test_frozen_corpus_ids_are_unique():
    ids = [case.id for case in FROZEN_CORPUS]
    assert len(ids) == len(set(ids))


def test_every_frozen_case_is_well_formed():
    for case in FROZEN_CORPUS:
        assert case.operation in OPERATIONS, case.id
        assert case.expected in OUTCOMES, case.id
        assert case.oracle_versions, case.id
        assert case.found_by, f"{case.id} does not say where it came from"
        assert case.note, f"{case.id} does not say what it protects"


def test_frozen_cases_carry_an_oracle_version_range():
    """Without a version range the corpus rots silently on an oracle upgrade."""
    for case in FROZEN_CORPUS:
        assert case.applies_to("1.14.0"), case.id
        assert not case.applies_to("0.1"), case.id


@pytest.mark.parametrize("case", FROZEN_CORPUS, ids=lambda c: c.id)
def test_frozen_corpus_case(case):
    _sympy()
    results = dict(run_frozen_corpus(cases=[case]))
    outcome = results[case]
    if outcome is None:
        pytest.skip(f"{case.id} does not apply to the installed oracle version")
    assert outcome.outcome == case.expected, (
        f"{case.id}: expected {case.expected}, got {outcome.outcome} "
        f"({outcome.reason}: {outcome.detail})"
    )
    if case.expected_reason is not None:
        assert outcome.reason == case.expected_reason, case.id


def test_version_range_matching():
    build = FrozenCase(id="t", operation="diff", build=lambda p: (), expected="agree")
    for spec, version, expected in [
        (">=1.12", "1.14.0", True),
        (">=1.12", "1.11.1", False),
        (">=1.12,<2", "1.99", True),
        (">=1.12,<2", "2.0.0", False),
        ("==1.14.0", "1.14.0", True),
        ("!=1.14.0", "1.14.0", False),
    ]:
        case = FrozenCase(
            id=build.id,
            operation="diff",
            build=build.build,
            expected="agree",
            oracle_versions=spec,
        )
        assert case.applies_to(version) is expected, (spec, version)


def test_version_range_tolerates_prerelease_suffixes():
    case = FrozenCase(
        id="t", operation="diff", build=lambda p: (), expected="agree", oracle_versions=">=1.12"
    )
    assert case.applies_to("1.14.0rc1")


# ---------------------------------------------------------------------------
# Docs and error-code hygiene
# ---------------------------------------------------------------------------


def test_module_docstrings_have_runnable_doctests():
    """The doctests must pass with *and* without SymPy, so they are guarded."""
    import doctest

    import alkahest.crosscheck as module

    failures, _tests = doctest.testmod(module, verbose=False)
    assert failures == 0


def test_all_exports_exist():
    import alkahest.crosscheck as module

    for name in module.__all__:
        assert hasattr(module, name), name
    assert len(module.__all__) == len(set(module.__all__))


def test_error_codes_used_are_the_registered_three():
    """Only ``E-XCHECK-001/002/003`` may be raised from this module."""
    source = (REPO / "python" / "alkahest" / "crosscheck.py").read_text(encoding="utf-8")
    used = set(re.findall(r'"(E-XCHECK-\d+)"', source))
    assert used <= {"E-XCHECK-001", "E-XCHECK-002", "E-XCHECK-003"}
    assert used == {"E-XCHECK-001", "E-XCHECK-002", "E-XCHECK-003"}


def test_doc_page_exists_and_covers_the_surface():
    page = REPO / "docs" / "mdbook" / "src" / "crosscheck.md"
    assert page.is_file(), "docs/mdbook/src/crosscheck.md is listed in SUMMARY.md"
    text = page.read_text(encoding="utf-8")
    for token in ("unavailable", "incomparable", "E-XCHECK-001", "oracles()", "check("):
        assert token in text, f"crosscheck.md does not mention {token!r}"


def test_module_is_importable_without_touching_sympy():
    """Importing ``alkahest`` must not pull SymPy in — it is an optional dep."""
    source = (REPO / "python" / "alkahest" / "crosscheck.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:  # module level only; lazy imports live inside functions
        assert not isinstance(node, ast.Import) or all(
            alias.name != "sympy" for alias in node.names
        )
        assert not (isinstance(node, ast.ImportFrom) and node.module == "sympy")
