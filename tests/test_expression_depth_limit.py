"""Deeply nested expressions must be refused, not fatal.

Every operation on an expression is a structural recursion over the DAG, and a
native stack overflow is a ``SIGSEGV`` — the process dies with no traceback and
no exception, so an unattended loop's ``except Exception`` never runs and the
whole run is lost.  Past a measured ceiling those operations raise
:class:`~alkahest.DepthLimitError` (``E-DEPTH-001``) instead.

Before this guard the following killed the interpreter outright, at these
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

Every test here stays **just past** the limit rather than out at the old crash
depths: a regression must fail the assertion, not take the test process down
with it.
"""

from __future__ import annotations

import alkahest as ak
import pytest

#: Mirrors ``alkahest_core::kernel::depth::MAX_EXPR_DEPTH``.  Hard-coded rather
#: than imported so that lowering the Rust constant without updating the
#: documented contract shows up here.
MAX_EXPR_DEPTH = 2048


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


# ---------------------------------------------------------------------------
# The boundary itself
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


def test_refusal_is_catchable_as_a_plain_exception(too_deep: ak.Expr):
    """The whole point: a loop wrapping work in ``except Exception`` must
    survive.  A ``PanicException`` (a ``BaseException``) or a segfault would
    both slip past the ``except Exception`` a real loop is written with."""
    caught: Exception | None = None
    try:
        ak.simplify(too_deep)
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


# ---------------------------------------------------------------------------
# Every entry point that used to die
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda e, x, p: str(e), id="str"),
        pytest.param(lambda e, x, p: repr(e), id="repr"),
        pytest.param(lambda e, x, p: ak.latex(e), id="latex"),
        pytest.param(lambda e, x, p: ak.unicode_str(e), id="unicode_str"),
        pytest.param(lambda e, x, p: ak.simplify(e), id="simplify"),
        pytest.param(lambda e, x, p: ak.simplify_par(e), id="simplify_par"),
        pytest.param(lambda e, x, p: ak.simplify_redex(e), id="simplify_redex"),
        pytest.param(lambda e, x, p: ak.simplify_auto(e), id="simplify_auto"),
        pytest.param(lambda e, x, p: ak.simplify_egraph(e), id="simplify_egraph"),
        pytest.param(lambda e, x, p: ak.simplify_expanded(e), id="simplify_expanded"),
        pytest.param(lambda e, x, p: ak.simplify_trig(e), id="simplify_trig"),
        pytest.param(lambda e, x, p: ak.collect_like_terms(e), id="collect_like_terms"),
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


def test_batch_entry_points_report_the_refusal_per_item(too_deep: ak.Expr, x: ak.Expr):
    """``*_many`` collect per-item outcomes rather than raising, so the refusal
    has to show up in the item, not as a crash."""
    (item,) = ak.simplify_many([too_deep])
    assert item.error is not None
    assert item.error["code"] == "E-DEPTH-001"

    (item,) = ak.diff_many([too_deep], x)
    assert item.error is not None
    assert item.error["code"] == "E-DEPTH-001"


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
