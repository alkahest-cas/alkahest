"""Numeric evaluation: DAG memoization, batch tier upgrade, traced-fn caching.

Regression tests for the numeric-evaluation performance fixes. The timing
bounds are deliberately loose (seconds where the fixed code takes
milliseconds); they exist to catch a return to exponential or per-call
recompilation behaviour, not to benchmark.
"""

import math
import time
import warnings
from fractions import Fraction

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")


def _chebyshev(p, x, n):
    """T_n via T_{k+1} = 2x·T_k − T_{k−1}: ~3n nodes, ~fib(n) paths."""
    a, b = p.integer(1), x
    for _ in range(n - 1):
        a, b = b, 2 * x * b - a
    return b


@pytest.mark.parametrize("mode", ["f64", "complex", "exact"])
def test_evaluate_is_linear_on_a_shared_dag(mode):
    p = ak.ExprPool()
    x = p.symbol("x")
    t40 = _chebyshev(p, x, 40)
    start = time.perf_counter()
    if mode == "exact":
        # T_40(1/2) = cos(40π/3) = −1/2 exactly.
        r = ak.evaluate(t40, {x: Fraction(1, 2)}, mode="exact")
        assert r.status == "ok"
        assert r.value == Fraction(-1, 2)
    else:
        theta = 0.3
        r = ak.evaluate(t40, {x: math.cos(theta)}, mode=mode)
        assert r.status == "ok"
        assert abs(complex(r.value) - math.cos(40 * theta)) < 1e-8
    # The unmemoized walk visits ~1.6e8 paths here (minutes).
    assert time.perf_counter() - start < 2.0


def test_eval_expr_and_evaluate_agree_on_a_dag():
    p = ak.ExprPool()
    x = p.symbol("x")
    t30 = _chebyshev(p, x, 30)
    a = ak.eval_expr(t30, {x: 0.3})
    b = ak.evaluate(t30, {x: 0.3}, mode="f64").value
    assert a == b


def _small(p, x, y):
    return ak.sin(x) * ak.exp(-x * x / 2) + x**3 - 2 * x * y + y**2


def test_compile_expr_reports_tier_and_accepts_expected_evals():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    e = _small(p, x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = ak.compile_expr(e, [x, y])
        g = ak.compile_expr(e, [x, y], expected_evals=1_000_000)
    assert f.tier == "interpreter"
    assert "tier=interpreter" in repr(f)
    if ak.jit_is_available():
        assert g.tier in ("cranelift", "llvm")
    else:
        assert g.tier == "interpreter"
    pt = [0.3, 0.7]
    assert math.isclose(f(pt), g(pt), rel_tol=1e-14, abs_tol=0.0)


def test_large_batch_upgrades_the_interpreter_tier_once_and_agrees():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    e = _small(p, x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = ak.compile_expr(e, [x, y])
    assert f._batch_tier is None

    xs_small = np.linspace(-2, 2, 100)
    ys_small = np.linspace(-1, 3, 100)
    ref_small = ak.numpy_eval(f, xs_small, ys_small)
    assert f._batch_tier is None, "a small batch must not trigger recompilation"
    for i in range(0, 100, 7):
        assert ref_small[i] == f([xs_small[i], ys_small[i]])

    n = 50_000
    xs = np.linspace(-2, 2, n)
    ys = np.linspace(-1, 3, n)
    out = ak.numpy_eval(f, xs, ys)
    expected_tier = "interpreter"
    if ak.jit_is_available():
        expected_tier = f._batch_tier
        assert expected_tier in ("cranelift", "llvm")
    assert f._batch_tier == expected_tier
    # Scalar calls keep the original tier.
    assert f.tier == "interpreter"
    ref = np.array([f([a, b]) for a, b in zip(xs[::97], ys[::97])])
    np.testing.assert_allclose(out[::97], ref, rtol=1e-13, atol=0)
    out2 = ak.numpy_eval(f, xs, ys)
    np.testing.assert_array_equal(out, out2)
    if hasattr(f, "call_batch_buffer_par"):
        out3 = ak.numpy_eval_par(f, xs, ys)
        np.testing.assert_array_equal(out, out3)


def test_interpreter_tier_nan_for_undefined_points_in_batch():
    p = ak.ExprPool()
    x = p.symbol("x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = ak.compile_expr(ak.log(x) + 1, [x])
    out = ak.numpy_eval(f, np.array([-1.0, 1.0, math.e]))
    assert math.isnan(out[0])
    assert out[1] == 1.0
    assert out[2] == pytest.approx(2.0)


def test_traced_functions_compile_once():
    p = ak.ExprPool()

    @ak.trace(p)
    def g(a, b):
        return ak.sin(a) * ak.exp(-a * a / 2) + a**3 - 2 * a * b + b**2

    xs = np.linspace(0, 1, 64)
    ys = np.linspace(1, 2, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r1 = g(xs, ys)
        c1 = g._compiled
        r2 = g(xs, ys)
        assert g._compiled is c1
        np.testing.assert_array_equal(r1, r2)

        gr = ak.grad(g)
        d1 = gr(xs, ys)
        cs = gr._compiled
        d2 = gr(xs, ys)
        assert gr._compiled is cs
        assert len(cs) == 2
        for u, v in zip(d1, d2):
            np.testing.assert_array_equal(u, v)

        gj = ak.jit(gr)
        for u, v in zip(gj(xs, ys), d1):
            np.testing.assert_array_equal(u, v)
        # Scalar path of the compiled gradient uses the compiled partials and
        # agrees with the interpreted gradient.
        s_interp = gr(0.3, 0.7)
        s_comp = gj(0.3, 0.7)
        for u, v in zip(s_interp, s_comp):
            assert math.isclose(u, v, rel_tol=1e-14, abs_tol=1e-300)


def test_no_jit_warning_points_at_cranelift():
    if ak.jit_is_available():
        pytest.skip("a native backend is compiled in; no fallback warning")
    p = ak.ExprPool()
    x = p.symbol("x")
    with pytest.warns(RuntimeWarning, match="not available") as rec:
        ak.compile_expr(x, [x])
    assert any("cranelift" in str(w.message) for w in rec)
