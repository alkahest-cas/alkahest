import alkahest
import pytest
from alkahest.experimental import Assumptions


def test_positive_refinement_enables_condition_gated_rewrites():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))

    assert str(assumptions.simplify(alkahest.sqrt(x**2)).value) == "x"
    assert str(assumptions.simplify(alkahest.exp(alkahest.log(x))).value) == "x"


def test_nonzero_refinement_enables_cancellation():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    assumptions = Assumptions(p)
    assumptions.refine(p.pred_ne(x, p.integer(0)))

    # Algebraic cancelation also works without assumptions; under an explicit
    # NonZero fact the colored engine agrees.
    assert str(assumptions.simplify(x**0).value) == "1"
    assert str(assumptions.simplify(x * x**-1).value) == "1"


def test_unproven_branch_cut_rewrites_remain_unchanged():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")

    assert str(alkahest.simplify(alkahest.sqrt(x**2)).value) == "sqrt(x^2)"
    assert str(alkahest.simplify_log_exp(alkahest.exp(alkahest.log(x))).value) == "exp(log(x))"
    assert str(alkahest.simplify_log_exp(alkahest.log(x * y)).value) == "log((x * y))"


def test_contradiction_is_structured_and_context_is_unchanged():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))

    with pytest.raises(alkahest.AssumptionError) as error:
        assumptions.refine(p.le(x, p.integer(0)))

    assert error.value.code == "E-SIMPLIFY-001"
    assert len(assumptions.predicates) == 1


def test_cross_pool_predicate_is_rejected():
    p = alkahest.ExprPool()
    other = alkahest.ExprPool()
    assumptions = Assumptions(p)

    with pytest.raises(alkahest.PoolError):
        assumptions.refine(other.gt(other.symbol("x"), other.integer(0)))
