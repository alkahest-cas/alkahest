import alkahest
import pytest
from alkahest import Assumptions


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
    # Branch-cut log/exp identities stay put without positivity facts.
    assert str(alkahest.simplify_log_exp(alkahest.exp(alkahest.log(x))).value) == "exp(log(x))"
    assert str(alkahest.simplify_log_exp(alkahest.log(x * y)).value) == "log((x * y))"
    assert str(alkahest.simplify_log_exp(alkahest.log(x) + alkahest.log(y)).value) == (
        "(log(x) + log(y))"
    )


def test_simplify_log_exp_folds_under_assumptions():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))
    assumptions.refine(p.gt(y, p.integer(0)))

    assert str(alkahest.simplify_log_exp(alkahest.exp(alkahest.log(x)), assumptions).value) == "x"
    assert str(alkahest.simplify_log_exp(alkahest.log(x) + alkahest.log(y), assumptions).value) == (
        "log((x * y))"
    )


def test_static_positive_domain_enables_exp_of_log():
    p = alkahest.ExprPool()
    x = p.symbol("x", domain=alkahest.Domain.Positive)
    assert str(alkahest.simplify_log_exp(alkahest.exp(alkahest.log(x))).value) == "x"
    assert str(alkahest.simplify(alkahest.sqrt(x**2)).value) == "x"


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


def test_experimental_shim_still_re_exports_assumptions():
    """`Assumptions` graduated to the stable top level; `alkahest.experimental`
    keeps re-exporting it (unchanged) for callers on the old import path."""
    from alkahest.experimental import Assumptions as ExperimentalAssumptions

    assert ExperimentalAssumptions is Assumptions


# ---------------------------------------------------------------------------
# C — abs(x) -> x under a positivity fact
# ---------------------------------------------------------------------------


def test_abs_of_positive_rewrite_under_refinement():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))

    assert str(assumptions.simplify(alkahest.abs(x)).value) == "x"


def test_abs_without_positive_fact_unchanged():
    p = alkahest.ExprPool()
    x = p.symbol("x")

    assert str(alkahest.simplify(alkahest.abs(x)).value) == "abs(x)"
    # A NonZero-only fact doesn't discharge the (stronger) positivity
    # hypothesis `abs(x) -> x` needs — it's sound only for `x > 0`.
    nonzero_assumptions = Assumptions(p)
    nonzero_assumptions.refine(p.pred_ne(x, p.integer(0)))
    assert str(nonzero_assumptions.simplify(alkahest.abs(x)).value) == "abs(x)"


def test_static_positive_domain_enables_abs():
    p = alkahest.ExprPool()
    x = p.symbol("x", domain=alkahest.Domain.Positive)
    assert str(alkahest.simplify(alkahest.abs(x)).value) == "x"


# ---------------------------------------------------------------------------
# A — context manager integration
# ---------------------------------------------------------------------------


def test_context_assumptions_picked_up_by_simplify_and_simplify_log_exp():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))
    assumptions.refine(p.gt(y, p.integer(0)))

    with alkahest.context(pool=p, assumptions=assumptions):
        assert str(alkahest.simplify(alkahest.sqrt(x**2)).value) == "x"
        assert str(alkahest.simplify(alkahest.abs(x)).value) == "x"
        assert str(alkahest.simplify_log_exp(alkahest.exp(alkahest.log(x))).value) == "x"
        assert (
            str(alkahest.simplify_log_exp(alkahest.log(x) + alkahest.log(y)).value)
            == "log((x * y))"
        )

    # Outside the `with` block, the context is gone — branch-cut rewrites
    # revert to requiring their own explicit assumptions.
    assert str(alkahest.simplify(alkahest.sqrt(x**2)).value) == "sqrt(x^2)"
    assert str(alkahest.simplify_log_exp(alkahest.exp(alkahest.log(x))).value) == "exp(log(x))"


def test_explicit_assumptions_argument_overrides_context():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    empty_context_assumptions = Assumptions(p)  # no refinements
    explicit_assumptions = Assumptions(p)
    explicit_assumptions.refine(p.gt(x, p.integer(0)))

    with alkahest.context(pool=p, assumptions=empty_context_assumptions):
        # The context's Assumptions has no facts; the explicit argument wins.
        assert str(alkahest.simplify(alkahest.sqrt(x**2), explicit_assumptions).value) == "x"
        assert (
            str(
                alkahest.simplify_log_exp(alkahest.exp(alkahest.log(x)), explicit_assumptions).value
            )
            == "x"
        )


def test_context_without_assumptions_key_is_unaffected():
    """A plain `context(pool=..., domain=...)` (no `assumptions=`) must not
    change `simplify`'s behavior — regression guard for the context wiring."""
    p = alkahest.ExprPool()
    x = p.symbol("x")

    with alkahest.context(pool=p, domain="real"):
        assert str(alkahest.simplify(alkahest.sqrt(x**2)).value) == "sqrt(x^2)"


# ---------------------------------------------------------------------------
# D — solve respects positivity assumptions
# ---------------------------------------------------------------------------


def test_solve_filters_roots_violating_positive_assumption():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))

    sols = alkahest.solve([x**2 - p.integer(4)], [x], domain="real", assumptions=assumptions)
    assert len(sols) == 1
    assert str(sols[0][x]) == "2"


def test_solve_without_assumptions_keeps_both_roots():
    p = alkahest.ExprPool()
    x = p.symbol("x")

    sols = alkahest.solve([x**2 - p.integer(4)], [x], domain="real")
    assert len(sols) == 2


def test_solve_picks_up_context_assumptions():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))

    with alkahest.context(pool=p, assumptions=assumptions):
        sols = alkahest.solve([x**2 - p.integer(4)], [x], domain="real")
    assert len(sols) == 1
    assert str(sols[0][x]) == "2"


def test_assumptions_is_positive_helper():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    assumptions = Assumptions(p)
    assumptions.refine(p.gt(x, p.integer(0)))

    assert assumptions.is_positive(x) is True
    assert assumptions.is_positive(y) is False


def test_assumptions_is_positive_from_static_domain():
    p = alkahest.ExprPool()
    x = p.symbol("x", domain=alkahest.Domain.Positive)
    assert Assumptions(p).is_positive(x) is True
