"""Deep expressions: refused where recursing over them would be fatal, run where it would not.

Most operations on an expression are a structural recursion over the DAG, and a
native stack overflow is a ``SIGSEGV`` — the process dies with no traceback and
no exception, so an unattended loop's ``except Exception`` never runs and the
whole run is lost.  Past a measured ceiling those operations raise
:class:`~alkahest.DepthLimitError` (``E-DEPTH-001``) instead.

Before that guard the following killed the interpreter outright, at these
depths, on a release build with the usual 8 MiB main-thread stack:

===============================  ============  ==========
operation                        deepest OK    segfaulted
===============================  ============  ==========
``symbolic_grad``                     4 625        4 687
``simplify`` / ``to_lean``            9 216        9 472
``latex``                            13 312       13 824
``unicode_str``                      15 360       15 872
``str`` / ``repr``                   23 552       24 576
===============================  ============  ==========

**The contract is per operation, not one contract for all of them.**  The
``simplify`` row above is history: those traversals no longer recurse without a
net — ``simplify::engine`` and ``simplify::parallel`` run under a
segmented-stack trampoline, ``simplify::redex`` schedules the DAG level by
level and does not recurse at all — so the simplification entry points accept
input of any depth.  A 100 000-level ``sin`` chain simplifies in about 0.1 s.

What they still refuse is an input that would reach the one recursion left on
that path: every default simplification ends in the assumption-gated colored
e-graph pass when the expression carries a ``Domain.Positive`` or
``Domain.NonZero`` symbol, and that pass recurses (SIGSEGV between 60 000 and
100 000 levels) *and* is quadratic (5 000 levels: 4.8 s; 20 000: 100 s).  So
the same shape over a positive symbol is still refused, with the same code.

Three groups of tests below, in that order:

1. entry points that still recurse — the ceiling, unchanged;
2. entry points that no longer do — deep input succeeds;
3. the route that puts a lifted entry point back on a recursion — refused again.

Tests in group 1 stay **just past** the limit rather than out at the old crash
depths: a regression must fail the assertion, not take the test process down
with it.  Tests in group 2 have to go well past it, because succeeding at 2 049
would prove nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import alkahest as ak
import pytest

#: Mirrors ``alkahest_core::kernel::depth::MAX_EXPR_DEPTH``.  Hard-coded rather
#: than imported so that lowering the Rust constant without updating the
#: documented contract shows up here.
MAX_EXPR_DEPTH = 2048

#: Far enough past the ceiling that no fixed limit could be hiding behind a
#: pass: ten times ``MAX_EXPR_DEPTH``, and past every crash depth in the table
#: above.
DEEP = 20_000

#: The headline number.  One test uses it; the rest use :data:`DEEP`, which is
#: already past everything and costs less to build.
VERY_DEEP = 100_000


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x", "real")


def nest(x: ak.Expr, depth: int) -> ak.Expr:
    """``sin(sin(...sin(x)...))`` with ``depth`` applications."""
    e = x
    for _ in range(depth):
        e = ak.sin(e)
    return e


@pytest.fixture
def too_deep(x: ak.Expr) -> ak.Expr:
    """Exactly one level past the ceiling — the cheapest input that must fail."""
    return nest(x, MAX_EXPR_DEPTH)  # depth = MAX_EXPR_DEPTH + 1 counting `x`


@pytest.fixture
def deep(x: ak.Expr) -> ak.Expr:
    """Deep enough that only an unbounded traversal can get through it."""
    return nest(x, DEEP)


@pytest.fixture
def deep_positive(pool: ak.ExprPool) -> ak.Expr:
    """:data:`DEEP`, but over a symbol whose domain routes it to the e-graph."""
    return nest(pool.symbol("p", "positive"), DEEP)


# ---------------------------------------------------------------------------
# 1. The entry points that still recurse
# ---------------------------------------------------------------------------


def test_at_the_limit_is_accepted(x: ak.Expr):
    """The documented number is the deepest that still works, not the first
    that fails — otherwise the limit in the docs is off by one."""
    ok = nest(x, MAX_EXPR_DEPTH - 1)
    assert str(ok).count("sin") == MAX_EXPR_DEPTH - 1


def test_one_past_the_limit_is_refused(too_deep: ak.Expr):
    with pytest.raises(ak.DepthLimitError) as excinfo:
        str(too_deep)
    assert excinfo.value.code == "E-DEPTH-001"
    assert "2048" in str(excinfo.value)


def test_refusal_is_catchable_as_a_plain_exception(too_deep: ak.Expr, x: ak.Expr):
    """The whole point: a loop wrapping work in ``except Exception`` must
    survive.  A ``PanicException`` (a ``BaseException``) or a segfault would
    both slip past the ``except Exception`` a real loop is written with."""
    caught: Exception | None = None
    try:
        ak.symbolic_grad(too_deep, [x])
    except Exception as e:  # deliberately broad: catching it here is the test
        caught = e
    assert isinstance(caught, ak.AlkahestError), (
        f"a too-deep expression must be refused, got {caught!r}"
    )
    assert caught.code == "E-DEPTH-001"


def test_width_is_not_depth(pool: ak.ExprPool, x: ak.Expr):
    """A wide n-ary node is shallow and must still be printable: the guard
    measures nesting, and confusing it with size would refuse ordinary work."""
    wide = pool.add([pool.integer(i) for i in range(50_000)] + [x])
    assert len(str(wide)) > 100_000


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda e, x, p: str(e), id="str"),
        pytest.param(lambda e, x, p: repr(e), id="repr"),
        pytest.param(lambda e, x, p: ak.latex(e), id="latex"),
        pytest.param(lambda e, x, p: ak.unicode_str(e), id="unicode_str"),
        pytest.param(lambda e, x, p: ak.simplify_egraph(e), id="simplify_egraph"),
        pytest.param(lambda e, x, p: ak.simplify_log_exp(e), id="simplify_log_exp"),
        pytest.param(lambda e, x, p: ak.diff(e, x), id="diff"),
        pytest.param(lambda e, x, p: ak.symbolic_grad(e, [x]), id="symbolic_grad"),
        pytest.param(lambda e, x, p: ak.jacobian([e], [x]), id="jacobian"),
        pytest.param(lambda e, x, p: ak.subs(e, {x: p.integer(1)}), id="subs"),
        pytest.param(lambda e, x, p: ak.eval_expr(e, {x: 0.5}), id="eval_expr"),
        pytest.param(lambda e, x, p: ak.evaluate(e, {x: 0.5}), id="evaluate"),
        pytest.param(lambda e, x, p: ak.compile_expr(e, [x]), id="compile_expr"),
        pytest.param(lambda e, x, p: ak.to_lean(e), id="to_lean"),
        pytest.param(lambda e, x, p: ak.to_stablehlo(e, [x]), id="to_stablehlo"),
        pytest.param(lambda e, x, p: ak.plot_dag(e), id="plot_dag"),
        pytest.param(lambda e, x, p: ak.integrate(e, x), id="integrate"),
        pytest.param(lambda e, x, p: ak.limit(e, x, p.integer(0)), id="limit"),
        pytest.param(lambda e, x, p: ak.series(e, x, p.integer(0), 3), id="series"),
        pytest.param(lambda e, x, p: ak.sum_indefinite(e, x), id="sum_indefinite"),
        pytest.param(lambda e, x, p: ak.poly_normal(e, [x]), id="poly_normal"),
        pytest.param(lambda e, x, p: ak.cancel(e), id="cancel"),
        pytest.param(lambda e, x, p: ak.together(e), id="together"),
        pytest.param(lambda e, x, p: ak.apart(e, x), id="apart"),
        pytest.param(lambda e, x, p: ak.horner(e, x), id="horner"),
        pytest.param(lambda e, x, p: ak.emit_c(e, x, "v", "f"), id="emit_c"),
        pytest.param(lambda e, x, p: ak.real_roots(e, x), id="real_roots"),
        pytest.param(lambda e, x, p: ak.resultant(e, e, x), id="resultant"),
        pytest.param(lambda e, x, p: ak.prove_nonneg(e, [x]), id="prove_nonneg"),
        pytest.param(lambda e, x, p: ak.sos_decompose(e, [x]), id="sos_decompose"),
        pytest.param(lambda e, x, p: ak.to_smtlib(p.lt(e, p.integer(1))), id="to_smtlib"),
        pytest.param(lambda e, x, p: ak.satisfiable(p.lt(e, p.integer(1))), id="satisfiable"),
        pytest.param(lambda e, x, p: ak.decide(ak.Forall(x, p.lt(e, p.integer(1)))), id="decide"),
        pytest.param(lambda e, x, p: ak.match_pattern(ak.sin(x), e), id="match_pattern"),
        pytest.param(
            lambda e, x, p: ak.interval_eval(e, {x: ak.ArbBall(0.5, 0.1)}),
            id="interval_eval",
        ),
    ],
)
def test_entry_point_refuses_instead_of_recursing(
    call, too_deep: ak.Expr, x: ak.Expr, pool: ak.ExprPool
):
    with pytest.raises(ak.DepthLimitError) as excinfo:
        call(too_deep, x, pool)
    assert excinfo.value.code == "E-DEPTH-001"


def test_explicit_assumptions_are_refused_at_the_plain_ceiling(
    too_deep: ak.Expr, x: ak.Expr, pool: ak.ExprPool
):
    """An explicit context sends *every* input through the colored e-graph, so
    unlike bare :func:`simplify` there is no depth at which it is safe."""
    ctx = ak.Assumptions(pool)
    ctx.refine(pool.gt(x, pool.integer(0)))

    with pytest.raises(ak.DepthLimitError):
        ctx.simplify(too_deep)
    with pytest.raises(ak.DepthLimitError):
        ak.simplify(too_deep, assumptions=ctx)
    with pytest.raises(ak.DepthLimitError), ak.context(assumptions=ctx):
        ak.simplify(too_deep)


def test_batch_entry_points_report_the_refusal_per_item(too_deep: ak.Expr, x: ak.Expr):
    """``*_many`` collect per-item outcomes rather than raising, so the refusal
    has to show up in the item, not as a crash."""
    (item,) = ak.diff_many([too_deep], x)
    assert item.error is not None
    assert item.error["code"] == "E-DEPTH-001"


# ---------------------------------------------------------------------------
# 2. The entry points that do not
# ---------------------------------------------------------------------------

#: Every simplification entry point whose traversal is stack-safe at any depth.
#: Kept as a list so :func:`test_the_lift_is_exactly_this_list` can check it
#: against the guard the Rust bindings actually call.
LIFTED = [
    pytest.param(lambda e: ak.simplify(e), id="simplify"),
    pytest.param(lambda e: ak.simplify_par(e), id="simplify_par"),
    pytest.param(lambda e: ak.simplify_redex(e), id="simplify_redex"),
    pytest.param(lambda e: ak.simplify_auto(e), id="simplify_auto"),
    pytest.param(lambda e: ak.simplify_expanded(e), id="simplify_expanded"),
    pytest.param(lambda e: ak.simplify_trig(e), id="simplify_trig"),
    pytest.param(lambda e: ak.simplify_trig_normal_form(e), id="simplify_trig_normal_form"),
    pytest.param(lambda e: ak.collect_like_terms(e), id="collect_like_terms"),
    pytest.param(lambda e: ak.simplify_pauli(e), id="simplify_pauli"),
    pytest.param(lambda e: ak.simplify_clifford_orthogonal(e), id="simplify_clifford_orthogonal"),
    pytest.param(lambda e: ak.simplify_with(e, []), id="simplify_with"),
    pytest.param(lambda e: ak.simplify_strategy(e), id="simplify_strategy"),
]


@pytest.mark.parametrize("call", LIFTED)
def test_lifted_entry_point_accepts_a_deep_expression(call, deep: ak.Expr):
    """The capability this file exists to pin: 20 000 levels, ten times the
    ceiling and past every crash depth in the table, and it returns."""
    assert call(deep) is not None


def test_simplify_returns_the_right_answer_at_a_hundred_thousand_levels(pool: ak.ExprPool):
    """Not merely "does not crash": the rewrite still has to happen, and the
    result has to be the expression the rules say it is.

    ``sin`` chains are inert, so the outermost node is a redex the simplifier
    must actually remove — which also puts a 100 000-level expression into the
    derivation log, where the recursive printer lives.
    """
    x = pool.symbol("x", "real")
    chain = nest(x, VERY_DEEP)
    result = ak.simplify(pool.mul([chain, pool.integer(1)]))
    assert result.value == chain


def test_a_deep_derivation_is_reported_rather_than_printed(pool: ak.ExprPool):
    """A log holds whatever subexpressions the rules fired on, and the renderer
    behind it is the same recursion as ``str()``.  Past the printer's reach the
    step is still reported — with its depth in place of its text — instead of
    taking the process out."""
    x = pool.symbol("x", "real")
    result = ak.simplify(pool.mul([nest(x, DEEP), pool.integer(1)]))

    (step,) = result.steps
    assert step["rule"] == "mul_one"
    assert "too deep to render" in step["before"]
    # `x`, `DEEP` levels of `sin`, and the `* 1` the rule removed.
    assert str(DEEP + 2) in step["before"]
    assert "too deep to render" in result.derivation


def test_simplify_many_no_longer_reports_a_refusal_for_a_deep_item(deep: ak.Expr):
    """The batch wrapper reports whatever the single call does, so lifting the
    single call has to show up here too."""
    (item,) = ak.simplify_many([deep])
    assert item.error is None


def test_shallow_results_render_exactly_as_before(pool: ak.ExprPool):
    """The depth-aware renderer must be invisible to every ordinary call: below
    the ceiling the derivation text is produced by the same code that always
    produced it."""
    x = pool.symbol("x", "real")
    result = ak.simplify(pool.mul([x, pool.integer(1)]))
    assert result.value == x
    (step,) = result.steps
    assert step["before"] == "(x * 1)"
    assert step["after"] == "x"
    assert "too deep" not in result.derivation


# ---------------------------------------------------------------------------
# 3. The route that puts a lifted entry point back on a recursion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("call", LIFTED)
def test_lifted_entry_point_still_refuses_what_would_reach_the_e_graph(
    call, deep_positive: ak.Expr
):
    """A ``Domain.Positive`` symbol anywhere in the expression sends the whole
    thing to the colored e-graph pass, which recurses and is quadratic.  Same
    shape, same depth, same entry points as the group above — and refused."""
    with pytest.raises(ak.DepthLimitError) as excinfo:
        call(deep_positive)
    assert excinfo.value.code == "E-DEPTH-001"


def test_a_nonzero_domain_routes_the_same_way(pool: ak.ExprPool):
    """``Domain.NonZero`` authorizes conditional rewrites too, so it takes the
    same route as ``Domain.Positive`` and must be refused with it."""
    with pytest.raises(ak.DepthLimitError):
        ak.simplify(nest(pool.symbol("nz", "nonzero"), DEEP))


def test_the_domain_only_matters_past_the_ceiling(pool: ak.ExprPool):
    """Below the ceiling the e-graph pass recurses at a depth it survives, so a
    positive symbol must not make an ordinary expression unsimplifiable."""
    p = pool.symbol("p", "positive")
    shallow = nest(p, MAX_EXPR_DEPTH - 1)
    assert ak.simplify(shallow) is not None


# ---------------------------------------------------------------------------
# Which guard each binding calls
# ---------------------------------------------------------------------------

_LIB_RS = Path(__file__).resolve().parents[1] / "alkahest-py" / "src" / "lib.rs"

#: The Rust ``#[pyfunction]`` / ``#[pymethods]`` entries that call
#: ``guard_simplify_depth`` rather than ``guard_depth``.  A binding added to
#: this list without a matching entry in :data:`LIFTED` is a lift nobody
#: tested; one removed from it is capability silently taken back.
EXPECTED_SIMPLIFY_GUARDED = {
    "py_collect_like_terms",
    "py_simplify",
    "py_simplify_auto",
    "py_simplify_clifford_orthogonal",
    "py_simplify_expanded",
    "py_simplify_par",
    "py_simplify_pauli",
    "py_simplify_redex",
    "py_simplify_strategy",
    "py_simplify_trig",
    "py_simplify_trig_normal_form",
    "py_simplify_with",
}


def _functions_calling(guard: str) -> set[str]:
    """Names of the Rust functions whose body calls ``guard``."""
    source = _LIB_RS.read_text()
    current = None
    found = set()
    fn_start = re.compile(r"^\s*(?:pub(?:\(crate\))?\s+)?fn\s+([A-Za-z0-9_]+)")
    for line in source.splitlines():
        match = fn_start.match(line)
        if match:
            current = match.group(1)
        if guard + "(" in line and current is not None and not line.lstrip().startswith("fn "):
            found.add(current)
    return found


def test_the_lift_is_exactly_this_list():
    """The lifted set is a decision about which traversals do not recurse, and
    it is not something a reader can infer from a function's name.  Pinning it
    against the source means a new binding has to choose a guard on purpose."""
    assert _functions_calling("guard_simplify_depth") == EXPECTED_SIMPLIFY_GUARDED


def test_every_lifted_binding_has_a_test_here():
    """Both directions of the same fact, so neither list can drift alone."""
    tested = {p.id for p in LIFTED}
    guarded = {name.removeprefix("py_") for name in EXPECTED_SIMPLIFY_GUARDED}
    assert tested == guarded


def test_the_unconditional_guard_is_still_the_common_case():
    """Lifting a dozen entry points must not have quietly lifted the rest: the
    plain ceiling is still what most of the boundary uses."""
    plain = _functions_calling("guard_depth") | _functions_calling("guard_expr_depth")
    assert len(plain) > 40
    assert not plain & EXPECTED_SIMPLIFY_GUARDED


# ---------------------------------------------------------------------------
# Deep polynomials — a different shape that reached different converters
# ---------------------------------------------------------------------------


def test_deep_polynomial_is_refused_by_the_polynomial_converters(pool: ak.ExprPool, x: ak.Expr):
    """``sin`` chains are rejected early by anything polynomial-only, so the
    converters needed a polynomial-shaped deep input to be exercised at all —
    and that shape crashed a different set of entry points."""
    one = pool.integer(1)
    e = x
    for _ in range(MAX_EXPR_DEPTH):
        e = pool.mul([pool.add([e, one]), pool.integer(2)])

    for call in (
        lambda: ak.poly_normal(e, [x]),
        lambda: ak.real_roots(e, x),
        lambda: ak.prove_nonneg(e, [x]),
        lambda: ak.sum_indefinite(e, x),
        lambda: ak.product_indefinite(e, x),
        lambda: ak.solve([pool.pred_eq(e, one)], [x]),
    ):
        with pytest.raises(ak.DepthLimitError):
            call()
