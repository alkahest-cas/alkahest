"""V3-3: First-order logic / FOFormula (satisfiability + quantifiers)."""

from __future__ import annotations

from fractions import Fraction

import alkahest
import pytest
from alkahest import ExprPool


def test_satisfiable_and_contradiction():
    p = ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    f = alkahest.And(p.gt(x, z), p.lt(x, z))
    assert alkahest.satisfiable(f) is False


def test_satisfiable_or_cover():
    p = ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    f = alkahest.Or(p.gt(x, z), p.le(x, z))
    r = alkahest.satisfiable(f)
    assert r is True or (isinstance(r, dict) and "x" in r)


def test_forall_exists_pool_api():
    p = ExprPool()
    x = p.symbol("x")
    body = p.gt(x, p.integer(0))
    ex = p.exists(x, body)
    fa = p.forall(x, body)
    assert alkahest.satisfiable(fa) is None  # unsupported fragment
    n = ex.node()
    assert n[0] == "exists"


def test_top_level_forall_exists():
    p = ExprPool()
    x = p.symbol("x")
    body = p.gt(x, p.integer(0))
    e = alkahest.Exists(x, body)
    assert e.node()[0] == "exists"


# ---------------------------------------------------------------------------
# Propositional satisfiability
#
# `satisfiable` decides the purely propositional fragment completely.  The
# contract that matters most here: `None` means "unsupported fragment" and
# never "unsatisfiable" -- only `False` is a proof of unsatisfiability.
# ---------------------------------------------------------------------------


def _implies(p, a, b):
    """``a -> b``; the kernel has no separate implication node."""
    return p.pred_or([p.pred_not(a), b])


def _iff(p, a, b):
    """``a <-> b``, written as ``(a -> b) and (b -> a)``."""
    return p.pred_and([_implies(p, a, b), _implies(p, b, a)])


def _holds(witness, formula_fn):
    """Evaluate ``formula_fn`` on a witness dict of ``"true"``/``"false"``."""
    return formula_fn({k: v == "true" for k, v in witness.items()})


def test_propositional_unsat():
    # (A or B) and not A and not B
    p = ExprPool()
    a, b = p.symbol("A"), p.symbol("B")
    f = p.pred_and([p.pred_or([a, b]), p.pred_not(a), p.pred_not(b)])
    assert alkahest.satisfiable(f) is False


def test_propositional_sat_returns_a_witness_that_works():
    # (A or B) and not A -- the only model is A = False, B = True.
    p = ExprPool()
    a, b = p.symbol("A"), p.symbol("B")
    f = p.pred_and([p.pred_or([a, b]), p.pred_not(a)])
    r = alkahest.satisfiable(f)
    assert isinstance(r, dict)
    assert r == {"A": "false", "B": "true"}
    assert _holds(r, lambda m: (m["A"] or m["B"]) and not m["A"])


def test_propositional_tautology_and_contradiction():
    p = ExprPool()
    a = p.symbol("A")
    # A or not A is satisfiable; its negation is not.
    tautology = p.pred_or([a, p.pred_not(a)])
    assert alkahest.satisfiable(tautology) not in (False, None)
    assert alkahest.satisfiable(p.pred_not(tautology)) is False
    # A and not A is unsatisfiable.
    assert alkahest.satisfiable(p.pred_and([a, p.pred_not(a)])) is False


def test_propositional_constants():
    p = ExprPool()
    a = p.symbol("A")
    assert alkahest.satisfiable(p.pred_true()) is True
    assert alkahest.satisfiable(p.pred_false()) is False
    assert alkahest.satisfiable(p.pred_and([a, p.pred_false()])) is False
    assert alkahest.satisfiable(p.pred_or([a, p.pred_true()])) is True


def test_propositional_implication():
    p = ExprPool()
    a, b = p.symbol("A"), p.symbol("B")
    # (A -> B) and A and not B denies modus ponens.
    assert alkahest.satisfiable(p.pred_and([_implies(p, a, b), a, p.pred_not(b)])) is False
    # (A -> B) and A forces B.
    r = alkahest.satisfiable(p.pred_and([_implies(p, a, b), a]))
    assert r == {"A": "true", "B": "true"}


def test_propositional_biconditional():
    p = ExprPool()
    a, b = p.symbol("A"), p.symbol("B")
    biconditional = _iff(p, a, b)
    assert alkahest.satisfiable(p.pred_and([biconditional, a, p.pred_not(b)])) is False
    r = alkahest.satisfiable(p.pred_and([biconditional, p.pred_not(a)]))
    assert r == {"A": "false", "B": "false"}


def test_propositional_pigeonhole_is_unsat():
    # Three pigeons into two holes.
    p = ExprPool()
    cell = [[p.symbol(f"p{i}{j}") for j in range(2)] for i in range(3)]
    clauses = [p.pred_or(row) for row in cell]
    for j in range(2):
        for i in range(3):
            for k in range(i + 1, 3):
                clauses.append(p.pred_or([p.pred_not(cell[i][j]), p.pred_not(cell[k][j])]))
    assert alkahest.satisfiable(p.pred_and(clauses)) is False


def test_boolean_witness_values_are_true_false_not_rationals():
    # A witness dict is homogeneous: a propositional one is all "true"/"false",
    # an arithmetic one is all rational literals.  A caller can tell them apart
    # from any single value.
    p = ExprPool()
    a = p.symbol("A")
    x = p.symbol("x")
    prop = alkahest.satisfiable(p.pred_and([a]))
    assert set(prop.values()) <= {"true", "false"}
    arith = alkahest.satisfiable(p.gt(x, p.integer(0)))
    assert isinstance(arith, dict)
    assert set(arith.values()).isdisjoint({"true", "false"})
    Fraction(arith["x"])  # parses as an exact rational


def test_mixed_boolean_and_arithmetic_is_still_none():
    # THE boundary case.  `A and (x > 0)` needs a Boolean-plus-theory
    # combination this function does not implement, so it must answer None --
    # "unsupported" -- and specifically must not answer False.
    p = ExprPool()
    a = p.symbol("A")
    x = p.symbol("x")
    assert alkahest.satisfiable(p.pred_and([a, p.gt(x, p.integer(0))])) is None
    # Even when the arithmetic part alone is contradictory: a False here would
    # be luck, not a proof.
    mixed = p.pred_and([a, p.gt(x, p.integer(0)), p.lt(x, p.integer(0))])
    assert alkahest.satisfiable(mixed) is None


def test_non_formula_term_is_still_none():
    p = ExprPool()
    x = p.symbol("x")
    assert alkahest.satisfiable(x + p.integer(1)) is None


def test_arithmetic_fragment_unchanged():
    p = ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    assert alkahest.satisfiable(alkahest.And(p.gt(x, z), p.lt(x, z))) is False
    r = alkahest.satisfiable(p.gt(x, z))
    assert isinstance(r, dict)
    assert Fraction(r["x"]) > 0


def test_smt_bridge_refuses_rather_than_falling_back_to_satisfiable():
    # The native paths must not depend on an external solver being installed.
    caps = alkahest.capabilities()["verification"]["smt_solvers"]
    p = ExprPool()
    x = p.symbol("x")
    a = p.symbol("A")
    f = p.pred_and([p.gt(p.mul([x, x]), p.integer(2)), p.lt(x, p.integer(5))])
    # to_smtlib is a pure emitter and works with no solver present.
    assert "QF_NRA" in alkahest.smt.to_smtlib(f)
    # ...and satisfiable itself never shells out.
    assert alkahest.satisfiable(p.pred_and([a, p.pred_not(a)])) is False
    if all(v is None for v in caps.values()):
        # With no solver installed, solve() refuses with E-SMT-001 rather than
        # degrading to satisfiable() and reporting its None as the solver's.
        with pytest.raises(alkahest.SmtError) as exc:
            alkahest.smt.solve(f)
        assert exc.value.code == "E-SMT-001"
