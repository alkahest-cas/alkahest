"""P2 item 3 — the SMT/SAT bridge (``alkahest.to_smtlib`` / ``alkahest.smt``).

Three tiers, deliberately separated:

1. **Emitter goldens.** Pure text, no solver, runnable everywhere. If these
   drift, the artifact Alkahest hands to the outside world changed.
2. **Coverage guards.** The emitter must stay exhaustive over ``Formula`` and
   ``PredicateKind``. Rust's match exhaustiveness enforces that at compile time;
   these assert it from outside so a new variant handled by a wildcard would
   still be caught.
3. **Round-trip against z3**, skipped when no solver is installed — including a
   property test over generated formulas for the one invariant the whole bridge
   rests on: every ``sat`` model survives exact back-substitution.
"""

from __future__ import annotations

import re
import subprocess
from fractions import Fraction
from pathlib import Path

import alkahest as ak
import pytest
from alkahest import research, smt
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parent.parent
EXPR_RS = REPO / "alkahest-core" / "src" / "kernel" / "expr.rs"
LOGIC_RS = REPO / "alkahest-core" / "src" / "logic" / "mod.rs"
SMTLIB_RS = REPO / "alkahest-core" / "src" / "logic" / "smtlib.rs"

has_solver = any(v is not None for v in smt.solvers().values())
requires_solver = pytest.mark.skipif(has_solver is False, reason="no SMT solver installed")

# A wall limit on every solver call so a pathological instance cannot stall the
# suite; the bridge maps a trip onto BudgetExceededError, which the tests that
# expect an answer would surface loudly rather than hang on.
BUDGET = ak.Budget(wall_ms=5_000)


@pytest.fixture
def pool():
    return ak.ExprPool()


# ---------------------------------------------------------------------------
# 1. Emitter goldens — no solver required
# ---------------------------------------------------------------------------


def test_golden_linear_real(pool):
    x = pool.symbol("x")
    f = ak.And(pool.gt(x, pool.integer(0)), pool.lt(x, pool.integer(3)))
    assert ak.to_smtlib(f) == (
        "; alkahest SMT-LIB 2 export\n"
        "(set-logic QF_LRA)\n"
        "(set-option :produce-models true)\n"
        "(declare-fun x () Real)\n"
        "(assert (and (> x 0) (< x 3)))\n"
        "(check-sat)\n"
        "(get-model)\n"
    )


def test_golden_nonlinear_picks_qf_nra(pool):
    x = pool.symbol("x")
    f = pool.gt(x * x, pool.integer(2))
    assert ak.to_smtlib(f) == (
        "; alkahest SMT-LIB 2 export\n"
        "(set-logic QF_NRA)\n"
        "(set-option :produce-models true)\n"
        "(declare-fun x () Real)\n"
        "(assert (> (* x x) 2))\n"
        "(check-sat)\n"
        "(get-model)\n"
    )


def test_golden_mixed_int_real_coerces_with_to_real(pool):
    x = pool.symbol("x")
    n = pool.symbol("n", "integer")
    f = ak.And(pool.gt(x, n), pool.ge(n, pool.integer(2)))
    assert ak.to_smtlib(f) == (
        "; alkahest SMT-LIB 2 export\n"
        "(set-logic QF_LIRA)\n"
        "(set-option :produce-models true)\n"
        "(declare-fun n () Int)\n"
        "(declare-fun x () Real)\n"
        "(assert (and (> x (to_real n)) (>= n 2)))\n"
        "(check-sat)\n"
        "(get-model)\n"
    )


def test_golden_pure_real_never_emits_to_real(pool):
    """``to_real`` lives in Reals_Ints; under QF_LRA it would not even parse."""
    x = pool.symbol("x")
    assert "to_real" not in ak.to_smtlib(pool.gt(x, pool.integer(2)))


def test_golden_quantified_drops_the_qf_prefix(pool):
    x = pool.symbol("x")
    y = pool.symbol("y")
    f = ak.Forall(x, pool.gt(x + y, pool.integer(0)))
    text = ak.to_smtlib(f)
    assert "(set-logic LRA)" in text
    assert "(forall ((x Real))" in text
    # The bound variable is *not* also declared at the top level.
    assert "(declare-fun x () Real)" not in text
    assert "(declare-fun y () Real)" in text


def test_golden_refined_domain_emits_its_side_condition(pool):
    """Dropping the guard would silently widen the question being asked."""
    q = pool.symbol("q", "positive")
    text = ak.to_smtlib(pool.lt(q, pool.integer(1)))
    assert "(assert (> q 0))" in text


@pytest.mark.parametrize(
    ("domain", "guard"),
    [("positive", "(> g 0)"), ("nonneg", "(>= g 0)")],
)
def test_golden_refined_domains(pool, domain, guard):
    g = pool.symbol("g", domain)
    assert f"(assert {guard})" in ak.to_smtlib(pool.lt(g, pool.integer(5)))


def test_golden_rationals_stay_exact(pool):
    x = pool.symbol("x")
    text = ak.to_smtlib(pool.pred_eq(x, pool.rational(-1, 3)))
    assert "(/ (- 1) 3)" in text


def test_golden_powers_expand_into_products(pool):
    x = pool.symbol("x")
    assert "(>= (* x x x) 8)" in ak.to_smtlib(pool.ge(x**3, pool.integer(8)))


def test_golden_negative_power_over_an_int_base_uses_real_division(pool):
    """`/` is real division in SMT-LIB; `div` is integer division.

    `n^-1` for an integer `n` means the real reciprocal, so both operands are
    lifted with `to_real`. Emitting a bare `(/ 1 n)` under a Reals_Ints logic
    would still be real division, but relying on mixed-sort sugar that only some
    logics define; emitting `div` would silently change the question to integer
    division. This pins the coercion so neither can creep in.
    """
    n = pool.symbol("n", "integer")
    text = ak.to_smtlib(pool.pred_eq(n ** pool.integer(-1), pool.rational(1, 2)))
    assert "(/ (to_real 1) (to_real n))" in text
    assert "div" not in text
    # Only one sort appears in the source, but the coercion needs Reals_Ints.
    assert "(set-logic QF_NIRA)" in text


def test_golden_negative_power_over_a_real_base_needs_no_coercion(pool):
    x = pool.symbol("x")
    text = ak.to_smtlib(pool.pred_eq(x ** pool.integer(-1), pool.rational(1, 2)))
    assert "(/ 1 x)" in text
    assert "to_real" not in text


def test_golden_piecewise_becomes_ite(pool):
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(x, pool.integer(0)), pool.integer(1))], pool.integer(-1))
    assert "(ite (> x 0) 1 (- 1))" in ak.to_smtlib(pool.pred_eq(pw, pool.integer(1)))


def test_golden_reserved_names_are_quoted(pool):
    ite = pool.symbol("ite")
    assert "(declare-fun |ite| () Real)" in ak.to_smtlib(pool.gt(ite, pool.integer(0)))


def test_check_sat_and_get_model_are_optional(pool):
    x = pool.symbol("x")
    text = ak.to_smtlib(pool.gt(x, pool.integer(0)), check_sat=False, get_model=False)
    assert "(check-sat)" not in text
    assert "(get-model)" not in text
    assert ":produce-models" not in text


# ---------------------------------------------------------------------------
# Refusals — the emitter is total-or-refuse
# ---------------------------------------------------------------------------


def test_float_literal_is_refused_not_approximated(pool):
    """A float is not the exact question it looks like, so it is not exported."""
    x = pool.symbol("x")
    with pytest.raises(ak.SmtError) as exc:
        ak.to_smtlib(pool.gt(x, pool.float(0.1)))
    assert exc.value.code == "E-SMT-002"
    assert "rational" in (exc.value.remediation or "")


def test_transcendental_head_is_refused(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.SmtError) as exc:
        ak.to_smtlib(pool.gt(ak.sin(x), pool.integer(0)))
    assert exc.value.code == "E-SMT-002"


def test_complex_symbol_is_refused(pool):
    z = pool.symbol("z", "complex")
    with pytest.raises(ak.SmtError) as exc:
        ak.to_smtlib(pool.gt(z, pool.integer(0)))
    assert exc.value.code == "E-SMT-002"


def test_non_predicate_is_refused(pool):
    with pytest.raises(ak.SmtError) as exc:
        ak.to_smtlib(pool.symbol("x"))
    assert exc.value.code == "E-SMT-002"


def test_huge_exponent_is_bounded_not_expanded(pool):
    """`x**100000` must refuse, not emit a megabyte of `x`."""
    x = pool.symbol("x")
    with pytest.raises(ak.SmtError) as exc:
        ak.to_smtlib(pool.ge(x**100_000, pool.integer(1)))
    assert exc.value.code == "E-SMT-002"
    assert "MAX_POW_EXPANSION" in (exc.value.remediation or "")


def test_named_logic_too_weak_is_an_error_not_a_downgrade(pool):
    x = pool.symbol("x")
    f = pool.gt(x * x, pool.integer(2))
    with pytest.raises(ak.SmtError) as exc:
        ak.to_smtlib(f, "QF_LRA")
    assert exc.value.code == "E-SMT-002"
    assert "(set-logic QF_NRA)" in ak.to_smtlib(f, "QF_NRA")


def test_unknown_logic_name_is_refused(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.SmtError):
        ak.to_smtlib(pool.gt(x, pool.integer(0)), "QF_BV")


# ---------------------------------------------------------------------------
# 2. Coverage guards — the emitter cannot silently drift
# ---------------------------------------------------------------------------

#: Every ``PredicateKind`` the emitter has been reviewed against. Adding one to
#: the kernel without adding it here fails, which is the point.
FROZEN_PREDICATE_KINDS = frozenset(
    {"Lt", "Le", "Gt", "Ge", "Eq", "Ne", "And", "Or", "Not", "True", "False"}
)

#: Same, for ``Formula``.
FROZEN_FORMULA_VARIANTS = frozenset(
    {"Atom", "And", "Or", "Not", "True", "False", "Forall", "Exists"}
)


def _enum_variants(source: str, name: str) -> set[str]:
    match = re.search(rf"pub enum {name} \{{(.*?)\n\}}", source, re.DOTALL)
    assert match, f"could not find `pub enum {name}` — did it move?"
    body = match.group(1)
    # Strip nested braces so struct-like variants contribute only their name.
    depth = 0
    flat = []
    for char in body:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0:
            flat.append(char)
    return set(re.findall(r"^\s*([A-Z]\w*)", "".join(flat), re.MULTILINE))


def test_predicate_kinds_have_not_drifted():
    found = _enum_variants(EXPR_RS.read_text(), "PredicateKind")
    assert found == FROZEN_PREDICATE_KINDS, (
        "PredicateKind changed. Handle the new variant in "
        "alkahest-core/src/logic/smtlib.rs (Emitter::formula) and update "
        "FROZEN_PREDICATE_KINDS here."
    )


def test_formula_variants_have_not_drifted():
    found = _enum_variants(LOGIC_RS.read_text(), "Formula")
    assert found == FROZEN_FORMULA_VARIANTS, (
        "Formula changed. Handle the new variant in "
        "alkahest-core/src/logic/smtlib.rs and update FROZEN_FORMULA_VARIANTS here."
    )


def test_emitter_mentions_every_predicate_kind():
    source = SMTLIB_RS.read_text()
    missing = [k for k in FROZEN_PREDICATE_KINDS if f"PredicateKind::{k}" not in source]
    assert not missing, f"smtlib.rs has no arm for PredicateKind::{missing}"


def test_emitter_mentions_every_formula_variant():
    source = SMTLIB_RS.read_text()
    missing = [v for v in FROZEN_FORMULA_VARIANTS if f"Formula::{v}" not in source]
    assert not missing, f"smtlib.rs has no arm for Formula::{missing}"


def test_emitter_has_no_wildcard_match_arm():
    """A `_ =>` arm would let a new node emit *something* instead of failing.

    Match exhaustiveness is the enforcement mechanism for this emitter; a
    catch-all disables it silently, and silently emitting plausible-but-wrong
    SMT-LIB is exactly the failure this subsystem exists to prevent.
    """
    offenders = [
        (i, line)
        for i, line in enumerate(SMTLIB_RS.read_text().splitlines(), 1)
        if re.match(r"^\s*_\s*=>", line)
    ]
    assert not offenders, f"wildcard match arm in smtlib.rs: {offenders}"


def test_every_predicate_kind_actually_emits(pool):
    """The static guards above pin the arms; this pins the behaviour."""
    x = pool.symbol("x")
    z = pool.integer(0)
    leaf = pool.gt(x, z)
    formulas = {
        "Lt": pool.lt(x, z),
        "Le": pool.le(x, z),
        "Gt": pool.gt(x, z),
        "Ge": pool.ge(x, z),
        "Eq": pool.pred_eq(x, z),
        "Ne": pool.pred_ne(x, z),
        "And": ak.And(leaf, leaf),
        "Or": ak.Or(leaf, leaf),
        "Not": ak.Not(leaf),
        "True": pool.pred_true(),
        "False": pool.pred_false(),
    }
    assert set(formulas) == FROZEN_PREDICATE_KINDS
    for kind, formula in formulas.items():
        assert "(assert " in ak.to_smtlib(formula), kind


def test_every_formula_variant_actually_emits(pool):
    x = pool.symbol("x")
    leaf = pool.gt(x, pool.integer(0))
    formulas = {
        "Atom": leaf,
        "And": ak.And(leaf, leaf),
        "Or": ak.Or(leaf, leaf),
        "Not": ak.Not(leaf),
        "True": pool.pred_true(),
        "False": pool.pred_false(),
        "Forall": ak.Forall(x, leaf),
        "Exists": ak.Exists(x, leaf),
    }
    assert set(formulas) == FROZEN_FORMULA_VARIANTS
    for variant, formula in formulas.items():
        assert "(assert " in ak.to_smtlib(formula), variant


# ---------------------------------------------------------------------------
# Model reading — exactness, tested without a solver
# ---------------------------------------------------------------------------


def _sexp(text):
    return smt._parse_sexprs(text)[0]


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("3", Fraction(3)),
        ("(- 3)", Fraction(-3)),
        ("(/ 1 3)", Fraction(1, 3)),
        ("(- (/ 1 3))", Fraction(-1, 3)),
        # SMT-LIB decimals are exact decimal rationals, parsed from the *string*.
        ("(/ 25.0 4.0)", Fraction(25, 4)),
        ("0.1", Fraction(1, 10)),
    ],
)
def test_model_values_lift_exactly(term, expected):
    value = smt._lift_value(_sexp(term) if term.startswith("(") else term, "x")
    assert value == expected
    assert isinstance(value, Fraction)


def test_decimal_lift_is_not_a_binary_float():
    """`0.1` must become 1/10, never the nearest double."""
    assert smt._lift_value("0.1", "x") == Fraction(1, 10)
    assert smt._lift_value("0.1", "x") != Fraction(0.1)


def test_root_obj_is_refused_not_truncated():
    """The whole point of D2: refuse the algebraic number, never round it."""
    term = _sexp("(root-obj (+ (^ x 2) (- 2)) 2)")
    with pytest.raises(ak.SmtError) as exc:
        smt._lift_value(term, "x")
    assert exc.value.code == "E-SMT-003"
    assert "refused, not rounded" in (exc.value.remediation or "")


def test_unknown_model_term_is_refused():
    with pytest.raises(ak.SmtError) as exc:
        smt._lift_value(_sexp("(_ as-array k!0)"), "x")
    assert exc.value.code == "E-SMT-003"


def test_model_parser_handles_both_z3_shapes():
    modern = "(\n  (define-fun x () Real\n    (- 2.0))\n)\n"
    legacy = "(model\n  (define-fun x () Real (- 2.0))\n)\n"
    for text in (modern, legacy):
        model = smt._parse_model(smt._parse_sexprs(text))
        assert smt._lift_value(model["x"], "x") == Fraction(-2)


# ---------------------------------------------------------------------------
# Refusals in the driver, tested without a solver
# ---------------------------------------------------------------------------


def test_missing_solver_is_a_refusal_not_a_fallback(pool, monkeypatch):
    """Never quietly degrade to the weak interval `satisfiable`.

    That would answer ``Unknown`` and look exactly like a solver had run and
    found nothing.
    """
    monkeypatch.setattr(smt, "_find_binary", lambda _binary: None)
    x = pool.symbol("x")
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(pool.gt(x, pool.integer(0)))
    assert exc.value.code == "E-SMT-001"
    assert "refusal, not a fallback" in (exc.value.remediation or "")
    assert smt.solvers() == dict.fromkeys(smt.SOLVERS, None)
    assert smt.supported(pool.gt(x, pool.integer(0))).reason == "no_solver"


def test_unknown_solver_name_is_refused(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(pool.gt(x, pool.integer(0)), solver="nosuchsolver")
    assert exc.value.code == "E-SMT-001"


def test_quantified_formulas_are_refused_by_solve(pool):
    """`to_smtlib` exports them; `solve` will not, because it cannot check them."""
    x = pool.symbol("x")
    f = ak.Exists(x, pool.gt(x, pool.integer(0)))
    assert "(exists ((x Real))" in ak.to_smtlib(f)
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(f)
    assert exc.value.code == "E-SMT-002"
    support = smt.supported(f)
    assert support.exportable
    assert not support.supported
    assert support.reason == "quantified"


def test_existential_prefix_refusal_says_to_drop_the_quantifiers(pool):
    """The refusal is correct; the remediation must name the one-line fix.

    "Does there exist x, y, z such that ..." is the natural way to *write* the
    question and ``Exists`` is exported at top level, so this is the first
    refusal a caller hits. Saying only "quantifier-free formulas only" leaves
    them to guess that deleting the quantifiers is not just allowed but exactly
    equivalent.
    """
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    body = pool.gt(x * y * z, pool.integer(0))
    f = ak.Exists(x, ak.Exists(y, ak.Exists(z, body)))

    with pytest.raises(ak.SmtError) as exc:
        smt.solve(f)
    remediation = exc.value.remediation or ""
    assert "drop the quantifiers" in remediation
    assert "implicitly existentially quantified" in remediation
    # It names the variables it is talking about rather than speaking in general.
    for name in ("x", "y", "z"):
        assert name in remediation
    # And it does not offer to do the rewrite silently.
    assert "will not strip the quantifiers for you" in remediation
    # `supported()` carries the same guidance, so a plan-ahead caller sees it too.
    assert "drop the quantifiers" in smt.supported(f).detail


def test_forall_refusal_does_not_suggest_dropping_the_quantifiers(pool):
    """The rewrite is valid only for an existential prefix; do not over-promise."""
    x, y = pool.symbol("x"), pool.symbol("y")
    f = ak.Forall(x, ak.Exists(y, pool.gt(y, x)))
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(f)
    remediation = exc.value.remediation or ""
    assert "not available under a Forall" in remediation
    # No claim about *this* formula's bound variables, since it is not a prefix.
    assert not remediation.startswith("drop the quantifiers")


# ---------------------------------------------------------------------------
# 2b. The ambient domain reaches the emitted sorts
# ---------------------------------------------------------------------------


def test_pool_symbol_takes_the_ambient_domain(pool):
    """``pool.symbol`` and ``ak.symbol`` must agree inside ``context(domain=…)``.

    They used to disagree, and the disagreement was invisible: ``pool.symbol``
    always produced a ``Domain.Real`` symbol, so an integer feasibility question
    built that way was emitted as ``QF_NRA`` over ``Real`` and silently answered
    as its real relaxation while ``ak.active_domain()`` still said ``integer``.
    """
    with ak.context(pool=pool, domain=ak.Domain.Integer):
        x = pool.symbol("x")
        y = ak.symbol("y")
        script = ak.to_smtlib(pool.pred_eq(x * y, pool.integer(6)))
    assert "(set-logic QF_NIA)" in script
    assert "(declare-fun x () Int)" in script
    assert "(declare-fun y () Int)" in script


def test_pool_symbol_default_outside_a_context_is_still_real(pool):
    """No context, no change: the historical ``Domain.Real`` default stands."""
    x = pool.symbol("x")
    assert "(declare-fun x () Real)" in ak.to_smtlib(pool.gt(x, pool.integer(0)))


def test_explicit_domain_argument_beats_the_ambient_one(pool):
    """The opt-out has to exist, or the context becomes a trap in the other direction."""
    with ak.context(pool=pool, domain=ak.Domain.Integer):
        r = pool.symbol("r", "real")
        script = ak.to_smtlib(pool.gt(r * r, pool.integer(2)))
    assert "(declare-fun r () Real)" in script
    assert "(set-logic QF_NRA)" in script


@requires_solver
def test_integer_context_via_pool_symbol_yields_an_integer_model(pool):
    """The end-to-end shape of the trap: an Erdos-Straus instance, built with
    ``pool.symbol`` only, must come back with integers rather than ``252/13``."""
    with ak.context(pool=pool, domain=ak.Domain.Integer):
        x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
        one = pool.integer(1)
        f = ak.And(
            pool.pred_eq(pool.integer(4) * x * y * z, pool.integer(5) * (y * z + x * z + x * y)),
            ak.And(pool.ge(x, one), ak.And(pool.ge(y, one), pool.ge(z, one))),
        )
        result = smt.solve(f, budget=BUDGET)
    assert result.status == "sat"
    assert result.logic == "QF_NIA"
    assert result.sorts == {"x": "Int", "y": "Int", "z": "Int"}
    assert all(v.denominator == 1 for v in result.model.values())
    assert Fraction(4, 5) == sum(Fraction(1, int(result.model[n])) for n in ("x", "y", "z"))


@requires_solver
def test_verification_carries_the_logic_and_sorts_that_were_sent(pool):
    """``status`` says the answer was checked; ``sorts`` says which question it
    answered, and only the second distinguishes an integer problem from its real
    relaxation. Both statuses are ``exactly_verified``, so ``sorts`` has to be
    somewhere a caller reading ``verification`` will actually see it."""
    n = pool.symbol("n", "integer")
    r = pool.symbol("r", "real")
    result = smt.solve(
        ak.And(pool.gt(n, pool.integer(2)), pool.lt(r * r, pool.integer(2))), budget=BUDGET
    )
    assert result.verification["logic"] == result.logic
    assert result.verification["sorts"] == {"n": "Int", "r": "Real"}
    assert result.sorts == result.verification["sorts"]
    # It survives the trip into a claim graph, which is where a loop reads it.
    with research.session(title="sorts travel") as session:
        claim = session.record(result, statement="feasible", method="smt.solve")
    assert claim.verification["sorts"] == {"n": "Int", "r": "Real"}


@requires_solver
def test_uncheckable_formula_is_refused_before_the_solver_runs(pool, monkeypatch):
    """`abs` exports fine but the kernel cannot evaluate it exactly.

    Guarded on a solver because it is about *ordering*: that the exactness
    probe fires before the solver is invoked. With no solver installed the
    ``E-SMT-001`` refusal legitimately comes first, so there is no ordering
    left to observe.
    """
    x = pool.symbol("x")
    f = pool.gt(pool.func("abs", [x]), pool.integer(1))
    assert "(ite (>= x 0) x (- x))" in ak.to_smtlib(f)

    def _boom(*_args, **_kwargs):
        raise AssertionError("the solver must not be run for an uncheckable formula")

    monkeypatch.setattr(smt, "_run_solver", _boom)
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(f)
    assert exc.value.code == "E-SMT-002"
    assert smt.supported(f).reason == "not_exactly_checkable"


def test_bad_model_raises_loudly_rather_than_warning(pool, monkeypatch):
    """A model that fails back-substitution means something is broken."""
    x = pool.symbol("x")
    monkeypatch.setattr(
        smt,
        "_run_solver",
        lambda *_a, **_k: ("sat\n(\n  (define-fun x () Real\n    (- 5.0))\n)\n", 1.0),
    )
    monkeypatch.setattr(smt, "_resolve_solver", lambda _s: (smt._SPECS["z3"], "/nonexistent/z3"))
    monkeypatch.setattr(smt, "_version_of", lambda *_a: "fake")
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(pool.gt(x, pool.integer(0)))
    assert exc.value.code == "E-SMT-004"
    assert "never warned" in (exc.value.remediation or "")


def test_model_violating_a_refined_domain_is_caught(pool, monkeypatch):
    """The `Positive` guard is a separate assertion, and it is checked too."""
    q = pool.symbol("q", "positive")
    monkeypatch.setattr(
        smt,
        "_run_solver",
        lambda *_a, **_k: ("sat\n(\n  (define-fun q () Real\n    (- 5.0))\n)\n", 1.0),
    )
    monkeypatch.setattr(smt, "_resolve_solver", lambda _s: (smt._SPECS["z3"], "/nonexistent/z3"))
    monkeypatch.setattr(smt, "_version_of", lambda *_a: "fake")
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(pool.lt(q, pool.integer(10)))
    assert exc.value.code == "E-SMT-004"


def test_non_integer_value_for_an_int_symbol_is_caught(pool, monkeypatch):
    n = pool.symbol("n", "integer")
    monkeypatch.setattr(
        smt,
        "_run_solver",
        lambda *_a, **_k: ("sat\n(\n  (define-fun n () Int\n    (/ 1 2))\n)\n", 1.0),
    )
    monkeypatch.setattr(smt, "_resolve_solver", lambda _s: (smt._SPECS["z3"], "/nonexistent/z3"))
    monkeypatch.setattr(smt, "_version_of", lambda *_a: "fake")
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(pool.gt(n, pool.integer(0)))
    assert exc.value.code == "E-SMT-004"


# ---------------------------------------------------------------------------
# supported() — the plan-ahead predicate
# ---------------------------------------------------------------------------


def test_supported_is_truthy_and_carries_the_reason(pool):
    n = pool.symbol("n", "integer")
    support = smt.supported(pool.gt(n * n, pool.integer(10)))
    assert bool(support) is support.supported
    assert support.logic == "QF_NIA"
    assert support.script is not None


def test_supported_prefers_the_in_tree_route_for_real_arithmetic(pool):
    """D4, and it cuts against the usual instinct.

    For QF_NRA the in-tree route yields a certificate that composes with
    ``to_lean``; z3's nlsat yields an answer and no artifact.
    """
    x = pool.symbol("x")
    support = smt.supported(pool.ge(x * x, pool.integer(0)))
    assert support.logic in {"QF_LRA", "QF_NRA"}
    # The recommendation is about which *route* is better, not about what
    # happens to be installed, so it holds with or without a solver.
    assert support.recommendation == "prefer_in_tree"
    if has_solver:
        # `detail` names the in-tree route only when SMT was actually an
        # option; with no solver it explains the missing binary instead.
        assert "prove_nonneg" in support.detail


def test_supported_recommends_smt_for_mixed_arithmetic(pool):
    """The genuinely new capability: mixed integer/real."""
    x = pool.symbol("x")
    n = pool.symbol("n", "integer")
    support = smt.supported(ak.And(pool.gt(x, n), pool.lt(x * x, pool.integer(10))))
    assert support.logic == "QF_NIRA"
    assert support.recommendation == "smt"


def test_supported_never_raises_on_an_unexportable_formula(pool):
    x = pool.symbol("x")
    support = smt.supported(pool.gt(ak.sin(x), pool.integer(0)))
    assert not support
    assert not support.exportable
    assert support.reason == "outside_fragment"
    assert support.error is not None
    assert support.error.code == "E-SMT-002"


# ---------------------------------------------------------------------------
# Verification vocabulary
# ---------------------------------------------------------------------------


def test_externally_asserted_is_not_machine_checked():
    """D3. Widening this set to include "z3 said so" would erode the one
    guarantee ``research.py`` makes."""
    assert smt.EXTERNALLY_ASSERTED not in research.MACHINE_CHECKED_STATUSES
    assert frozenset({"exactly_verified", "lean_checked"}) == research.MACHINE_CHECKED_STATUSES


def test_externally_asserted_badge_is_unflattering():
    # One lowercased value for both checks: the badge says "NO proof", so a
    # case-sensitive second check would let a "Proven" through the guard that
    # exists to reject exactly that word.
    badge = smt.STATUS_BADGES[smt.EXTERNALLY_ASSERTED].lower()
    assert "no proof was checked" in badge
    assert "prov" not in badge.replace("proof", "")


# ---------------------------------------------------------------------------
# 3. Round trip against a real solver
# ---------------------------------------------------------------------------


@requires_solver
def test_sat_model_is_verified_and_reports_its_engine(pool):
    x = pool.symbol("x")
    n = pool.symbol("n", "integer")
    f = ak.And(pool.gt(x, n), ak.And(pool.lt(x * x, pool.integer(10)), pool.gt(n, pool.integer(1))))
    result = smt.solve(f, budget=BUDGET)
    assert result.status == "sat"
    assert result.engine.startswith(("z3", "cvc5"))
    assert result.logic == "QF_NIRA"
    assert set(result.model) == {"x", "n"}
    assert all(isinstance(v, Fraction) for v in result.model.values())
    assert result.model["n"].denominator == 1
    assert result.verification["status"] == "exactly_verified"
    assert result.machine_checked is True
    # Independently re-verify, without going through the bridge's own check.
    symbols = {"x": x, "n": n}
    check = ak.evaluate(f, {symbols[k]: v for k, v in result.model.items()}, mode="exact")
    assert check.status == "ok"
    assert check.value == 1


@requires_solver
def test_unsat_is_externally_asserted_and_never_machine_checked(pool):
    x = pool.symbol("x")
    f = ak.And(pool.gt(x, pool.integer(0)), pool.lt(x, pool.integer(0)))
    result = smt.solve(f, budget=BUDGET)
    assert result.status == "unsat"
    assert result.verification["status"] == smt.EXTERNALLY_ASSERTED
    assert result.machine_checked is False
    assert result.model == {}
    assert "no unsat proof" in result.verification["note"]


@requires_solver
def test_sat_model_lifts_into_the_pool_when_one_is_active(pool):
    x = pool.symbol("x")
    f = ak.And(pool.gt(x, pool.integer(1)), pool.lt(x, pool.integer(2)))
    with ak.context(pool=pool):
        result = smt.solve(f, budget=BUDGET)
    assert result.status == "sat"
    assert set(result.model_exprs) == {"x"}
    # The lifted Expr and the Fraction agree exactly.
    lifted = ak.evaluate(f, {x: result.model["x"]}, mode="exact")
    assert lifted.value == 1
    assert str(result.model_exprs["x"])


@requires_solver
def test_explicit_pool_argument_lifts_without_a_context(pool):
    x = pool.symbol("x")
    result = smt.solve(pool.gt(x, pool.integer(3)), budget=BUDGET, pool=pool)
    assert set(result.model_exprs) == {"x"}


@requires_solver
def test_algebraic_witness_is_refused_rather_than_rounded(pool):
    """`x**2 == 2, x > 0` has no rational witness; z3 answers with `root-obj`."""
    x = pool.symbol("x")
    f = ak.And(pool.pred_eq(x * x, pool.integer(2)), pool.gt(x, pool.integer(0)))
    with pytest.raises(ak.SmtError) as exc:
        smt.solve(f, budget=BUDGET)
    assert exc.value.code == "E-SMT-003"


@requires_solver
def test_piecewise_round_trips(pool):
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(x, pool.integer(0)), pool.integer(1))], pool.integer(-1))
    result = smt.solve(pool.pred_eq(pw, pool.integer(1)), budget=BUDGET)
    assert result.status == "sat"
    assert result.model["x"] > 0


@requires_solver
def test_pure_integer_problem_uses_an_integer_logic(pool):
    a = pool.symbol("a", "integer")
    b = pool.symbol("b", "integer")
    f = ak.And(
        pool.pred_eq(a * a + b * b, pool.integer(25)),
        ak.And(pool.gt(a, pool.integer(0)), pool.gt(b, pool.integer(0))),
    )
    result = smt.solve(f, budget=BUDGET)
    assert result.logic == "QF_NIA"
    assert result.status == "sat"
    assert {result.model["a"], result.model["b"]} == {Fraction(3), Fraction(4)}


@requires_solver
def test_the_mixed_logic_name_is_one_the_installed_solver_accepts(pool):
    """`QF_LIRA` / `QF_NIRA` are absent from the SMT-LIB 2.7 catalog.

    They are emitted anyway because they are what the solvers this bridge
    drives use for mixed Int/Real, and because the catalog alternatives
    (`AUFLIRA` / `AUFNIRA`) are quantified array logics that would discard the
    `QF_` hint. That makes it a contract with the *solver*, so pin it against
    the solver actually installed: the emitted script must be accepted outright,
    with no "unsupported logic" complaint.
    """
    x = pool.symbol("x")
    n = pool.symbol("n", "integer")
    mixed_linear = ak.And(pool.gt(x, n), pool.lt(x, n + pool.integer(1)))
    mixed_nonlinear = ak.And(pool.gt(x, n), pool.lt(x * x, pool.integer(10)))
    assert "(set-logic QF_LIRA)" in ak.to_smtlib(mixed_linear)
    assert "(set-logic QF_NIRA)" in ak.to_smtlib(mixed_nonlinear)

    spec, path = smt._resolve_solver("auto")
    for formula in (mixed_linear, mixed_nonlinear):
        output, _elapsed = smt._run_solver(spec, path, ak.to_smtlib(formula), 5_000)
        assert smt._extract_status(output) == "sat", output
        assert "unsupported" not in output.lower(), output
        assert "error" not in output.lower(), output

    # Non-vacuity: the same runner *does* complain about a name the solver has
    # never heard of, so the assertions above are testing something.
    bogus, _elapsed = smt._run_solver(spec, path, "(set-logic QF_BOGUS)\n(check-sat)\n", 5_000)
    assert "unsupported" in bogus.lower() or "error" in bogus.lower(), bogus

    # And a consumer that will only take catalog names has one available.
    catalog = ak.to_smtlib(mixed_linear, "AUFLIRA")
    assert "(set-logic AUFLIRA)" in catalog
    output, _elapsed = smt._run_solver(spec, path, catalog, 5_000)
    assert smt._extract_status(output) == "sat", output


def test_parent_deadline_maps_onto_budget_exceeded(monkeypatch):
    """D5: a solver timeout is a resource verdict, not a mathematical one.

    Driven directly rather than by racing a real solver against a small wall
    clock: what is under test is the *mapping*, and a faster machine or a
    smarter z3 answering first would fail this for an unrelated reason.
    """

    def never_answers(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout") or 0.0)

    monkeypatch.setattr(smt.subprocess, "run", never_answers)
    with pytest.raises(ak.BudgetExceededError) as exc:
        smt._run_solver(smt._SPECS["z3"], "/nonexistent/z3", "(check-sat)\n", 250)
    assert exc.value.code == "E-BUDGET-001"
    assert "250" in str(exc.value)


def test_solver_reported_timeout_maps_onto_budget_exceeded(pool, monkeypatch):
    """The other half of D5: `unknown` with `:reason-unknown timeout` is a
    resource verdict too, and must not surface as a bare `unknown` result."""
    x = pool.symbol("x")
    f = ak.And(pool.gt(x, pool.integer(0)), pool.lt(x, pool.integer(2)))
    monkeypatch.setattr(smt, "_resolve_solver", lambda name: (smt._SPECS["z3"], "/nonexistent/z3"))
    monkeypatch.setattr(
        smt,
        "_run_solver",
        lambda spec, path, script, wall_ms: ('unknown\n(:reason-unknown "timeout")\n', 250.0),
    )
    with pytest.raises(ak.BudgetExceededError) as exc:
        smt.solve(f, budget=ak.Budget(wall_ms=250))
    assert exc.value.code == "E-BUDGET-001"
    # Without a budget the same output is a result, not a resource refusal.
    result = smt.solve(f)
    assert result.status == "unknown"
    assert result.reason_unknown == "timeout"


@requires_solver
def test_a_real_solver_timeout_produces_a_verdict_or_a_budget_error(pool):
    """The wiring the mocks cannot cover: the solver's own timeout flag.

    Deliberately tolerant about *which* outcome — the point is that a tiny wall
    clock never produces a third thing (a crash, or a bare `unknown` that a
    loop would mistake for a mathematical answer)."""
    a, b, c, d = (pool.symbol(name) for name in "abcd")
    f = ak.And(
        pool.pred_eq(a**4 + b**4 + c**4 + d**4, pool.integer(1)),
        ak.And(
            pool.pred_eq(a**3 + b**3 + c**3 + d**3, pool.integer(0)),
            ak.And(
                pool.gt(a * b * c * d, pool.integer(0)),
                pool.pred_eq(
                    a * a * b * b + c * c * d * d + a * d + b * c, pool.rational(355, 113)
                ),
            ),
        ),
    )
    try:
        outcome = smt.solve(f, budget=ak.Budget(wall_ms=250)).status
    except ak.BudgetExceededError as exc:
        outcome = exc.code
    assert outcome in {"sat", "unsat", "unknown", "E-BUDGET-001"}


@requires_solver
def test_result_records_into_a_claim_graph_with_an_honest_status(pool):
    x = pool.symbol("x")
    session = research.session(title="smt bridge")
    with session:
        sat = smt.solve(
            ak.And(pool.gt(x, pool.integer(0)), pool.lt(x, pool.integer(2))), budget=BUDGET
        )
        claim = session.record(sat, statement="0 < x < 2 is satisfiable", method="smt.solve")
        assert claim.status == "exactly_verified"
        assert claim.machine_checked is True

        unsat = smt.solve(
            ak.And(pool.gt(x, pool.integer(0)), pool.lt(x, pool.integer(0))), budget=BUDGET
        )
        claim2 = session.record(unsat, statement="0 < x < 0 is unsatisfiable", method="smt.solve")
        assert claim2.status == smt.EXTERNALLY_ASSERTED
        # The load-bearing property: an external assertion is never counted as
        # machine-checked, whatever the badge vocabulary says about it.
        assert claim2.machine_checked is False


# ---------------------------------------------------------------------------
# The invariant the whole bridge rests on, as a property test
# ---------------------------------------------------------------------------

_COEFFS = st.integers(min_value=-4, max_value=4)
_RELATIONS = st.sampled_from(["lt", "le", "gt", "ge", "pred_eq", "pred_ne"])


@st.composite
def _atoms(draw, names):
    """A comparison between two small polynomials in the given symbols."""
    name = draw(st.sampled_from(names))
    other = draw(st.sampled_from(names))
    power = draw(st.integers(min_value=1, max_value=2))
    coeff = draw(_COEFFS)
    constant = draw(_COEFFS)
    return (draw(_RELATIONS), name, other, power, coeff, constant)


@st.composite
def _formulas(draw, names):
    atoms = draw(st.lists(_atoms(names), min_size=1, max_size=3))
    connectives = draw(st.lists(st.sampled_from(["and", "or"]), min_size=len(atoms) - 1))
    negate = draw(st.booleans())
    return atoms, connectives, negate


def _build(pool, symbols, spec):
    atoms, connectives, negate = spec
    built = []
    for relation, name, other, power, coeff, constant in atoms:
        lhs = symbols[name] ** power * coeff + symbols[other]
        built.append(getattr(pool, relation)(lhs, pool.integer(constant)))
    formula = built[0]
    for connective, atom in zip(connectives, built[1:]):
        formula = ak.And(formula, atom) if connective == "and" else ak.Or(formula, atom)
    return ak.Not(formula) if negate else formula


@requires_solver
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(spec=_formulas(("x", "n")))
def test_property_every_sat_model_survives_back_substitution(spec):
    """The invariant the bridge rests on, over generated formulas rather than
    a handful of examples.

    ``solve`` enforces this internally and raises ``E-SMT-004`` when it fails;
    the test additionally re-checks the model through ``evaluate`` so a bug in
    the bridge's own checker cannot make the test vacuous.
    """
    pool = ak.ExprPool()
    symbols = {"x": pool.symbol("x"), "n": pool.symbol("n", "integer")}
    formula = _build(pool, symbols, spec)
    try:
        result = smt.solve(formula, budget=ak.Budget(wall_ms=2_000))
    except ak.SmtError as exc:
        # E-SMT-003 (no rational witness) is a legitimate refusal, not a failure.
        refusal = exc
        assert refusal.code == "E-SMT-003", refusal
        return
    except ak.BudgetExceededError:
        return
    assert result.status in {"sat", "unsat", "unknown"}
    if result.status != "sat":
        assert result.model == {}
        assert not result.machine_checked
        return
    bindings = {symbols[name]: value for name, value in result.model.items() if name in symbols}
    check = ak.evaluate(formula, bindings, mode="exact")
    assert check.status == "ok", (ak.to_smtlib(formula), result.model, check.reason)
    assert check.value == 1, (ak.to_smtlib(formula), result.model)
    assert result.verification["status"] == "exactly_verified"


@requires_solver
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(spec=_formulas(("x", "y")))
def test_property_emitted_script_is_accepted_by_the_solver(spec):
    """Anything `to_smtlib` emits must at least *parse*; a script Alkahest
    produces and a solver rejects is an emitter bug, not a user error."""
    pool = ak.ExprPool()
    symbols = {"x": pool.symbol("x"), "y": pool.symbol("y")}
    formula = _build(pool, symbols, spec)
    try:
        result = smt.solve(formula, budget=ak.Budget(wall_ms=2_000))
    except (ak.SmtError, ak.BudgetExceededError) as exc:
        refusal = exc
        assert getattr(refusal, "code", "") in {"E-SMT-003", "E-BUDGET-001"}, refusal
        return
    assert "(error" not in result.raw_output.split(result.status)[0]
