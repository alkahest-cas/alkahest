"""Smoke tests: one test per major subsystem so a broken build fails fast."""

import warnings

import alkahest
import pytest
from alkahest import (
    HAS_EGRAPH,
    ArbBall,
    ExprPool,
    MultiPoly,
    RationalFunction,
    UniPoly,
    compile_expr,
    cos,
    diff,
    eval_expr,
    exp,
    integrate,
    interval_eval,
    jit_is_available,
    latex,
    parse,
    simplify,
    sin,
    to_lean,
    unicode_str,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pool():
    return ExprPool()


# ---------------------------------------------------------------------------
# Core import and version
# ---------------------------------------------------------------------------

def test_import():
    assert alkahest is not None


def test_version():
    v = alkahest.version()
    assert isinstance(v, str)
    assert len(v) > 0


# ---------------------------------------------------------------------------
# Differentiation
# ---------------------------------------------------------------------------

def test_diff_sin():
    p = pool()
    x = p.symbol("x")
    r = diff(sin(x), x)
    # d/dx sin(x) = cos(x)
    assert r is not None
    assert r.value is not None
    assert len(r.steps) > 0


def test_diff_polynomial():
    p = pool()
    x = p.symbol("x")
    # d/dx x^3 = 3*x^2
    r = diff(x ** 3, x)
    assert r is not None
    assert r.value is not None


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_integrate_exp():
    p = pool()
    x = p.symbol("x")
    r = integrate(exp(x), x)
    assert r is not None
    assert r.value is not None


def test_integrate_constant():
    p = pool()
    x = p.symbol("x")
    five = p.integer(5)
    r = integrate(five, x)
    assert r is not None
    assert r.value is not None


# ---------------------------------------------------------------------------
# Simplification (rule-based)
# ---------------------------------------------------------------------------

def test_simplify_zero():
    p = pool()
    x = p.symbol("x")
    zero = p.integer(0)
    r = simplify(x + zero)
    assert r.value == x


def test_simplify_trig_identity():
    """sin^2 + cos^2 = 1 via rule-based simplifier (or e-graph if available)."""
    p = pool()
    x = p.symbol("x")
    expr = sin(x) ** 2 + cos(x) ** 2
    if HAS_EGRAPH:
        from alkahest import simplify_egraph
        r = simplify_egraph(expr)
        # The e-graph should collapse this to 1
        assert r.value == p.integer(1)
    else:
        # Without e-graph, at least check that simplify runs without error
        r = simplify(expr)
        assert r.value is not None


# ---------------------------------------------------------------------------
# Expression parsing and pretty-printing
# ---------------------------------------------------------------------------

def test_parse_roundtrip():
    p = pool()
    x = p.symbol("x")
    e = parse("x^2 + 2*x + 1", p, {"x": x})
    assert e is not None


def test_latex_output():
    p = pool()
    x = p.symbol("x")
    out = latex(sin(x))
    assert "sin" in out.lower()


def test_unicode_output():
    p = pool()
    x = p.symbol("x")
    out = unicode_str(x ** 2)
    assert out is not None
    assert len(out) > 0


# ---------------------------------------------------------------------------
# JIT compiled evaluation / interpreter
# ---------------------------------------------------------------------------

def test_jit_is_available_returns_bool():
    assert isinstance(jit_is_available(), bool)


def test_compile_expr_constant():
    p = pool()
    five = p.integer(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = compile_expr(five, [])
    assert abs(f([]) - 5.0) < 1e-10


def test_compile_expr_variable():
    p = pool()
    x = p.symbol("x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = compile_expr(x ** 2, [x])
    assert abs(f([3.0]) - 9.0) < 1e-10


def test_eval_expr_basic():
    p = pool()
    x = p.symbol("x")
    result = eval_expr(x ** 2 + p.integer(1), {x: 3.0})
    assert abs(result - 10.0) < 1e-10


def test_compile_warns_when_jit_unavailable():
    """When JIT is not compiled in, compile_expr should emit a RuntimeWarning."""
    if jit_is_available():
        pytest.skip("JIT is available in this build — no fallback warning expected")
    p = pool()
    x = p.symbol("x")
    with pytest.warns(RuntimeWarning, match="not available"):
        compile_expr(x, [x])


# ---------------------------------------------------------------------------
# Polynomial subsystem
# ---------------------------------------------------------------------------

def test_unipoly_degree():
    p = pool()
    x = p.symbol("x")
    poly = UniPoly.from_symbolic(x ** 3 + p.integer(-2) * x + p.integer(1), x)
    assert poly.degree() == 3


def test_unipoly_gcd():
    p = pool()
    x = p.symbol("x")
    a = UniPoly.from_symbolic(x ** 2 + p.integer(-1), x)
    b = UniPoly.from_symbolic(x + p.integer(-1), x)
    g = a.gcd(b)
    assert g is not None


def test_multipoly_total_degree():
    p = pool()
    x = p.symbol("x")
    y = p.symbol("y")
    mp = MultiPoly.from_symbolic(x ** 2 * y + x * y ** 2, [x, y])
    assert mp.total_degree() == 3


def test_rational_function_normalization():
    p = pool()
    x = p.symbol("x")
    rf = RationalFunction.from_symbolic(
        x ** 2 + p.integer(-1), x + p.integer(-1), [x]
    )
    # (x^2 - 1)/(x - 1) = x + 1
    assert rf is not None


# ---------------------------------------------------------------------------
# Ball arithmetic (Arb)
# ---------------------------------------------------------------------------

def test_arb_ball_construction():
    b = ArbBall(1.0, 1e-10)
    s = str(b)
    assert s is not None


def test_interval_eval_sin():
    p = pool()
    x = p.symbol("x")
    result = interval_eval(sin(x), {x: ArbBall(1.0, 1e-10)})
    assert result is not None


# ---------------------------------------------------------------------------
# Polynomial system solver (Gröbner; optional)
# ---------------------------------------------------------------------------

def test_solver_linear_system():
    try:
        from alkahest import solve
    except ImportError:
        pytest.skip("groebner feature not available")
    p = pool()
    x = p.symbol("x")
    y = p.symbol("y")
    neg_one = p.integer(-1)
    # x + y - 1 = 0,  x - y = 0  →  x = y = 0.5
    r = solve([x + y + neg_one, x + p.integer(-1) * y], [x, y])
    assert r is not None


# ---------------------------------------------------------------------------
# Lean 4 certificate export
# ---------------------------------------------------------------------------

def test_lean_export():
    p = pool()
    x = p.symbol("x")
    # to_lean takes an Expr, not a DerivedResult
    lean = to_lean(x ** 2)
    assert isinstance(lean, str)
    assert len(lean) > 0


# ---------------------------------------------------------------------------
# StableHLO / XLA bridge
# ---------------------------------------------------------------------------

def test_stablehlo_export():
    p = pool()
    x = p.symbol("x")
    mlir = alkahest.to_stablehlo(x ** 2 + p.integer(1), [x])
    assert isinstance(mlir, str)
    assert "stablehlo" in mlir.lower() or "func" in mlir.lower()
