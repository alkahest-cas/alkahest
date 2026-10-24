"""Batch numeric evaluation (report 10-9, G7).

``numpy_eval`` used to be 8-27x slower than NumPy / ``lambdify`` at 10^6
points: the Python 3.9/3.10 wheels copied every array through ``.tolist()``,
every power was a libm ``pow`` call, ``erf`` and friends dropped the whole
expression to the interpreter, and ``numpy_eval_par`` made one scalar call per
point. These tests pin the *values* of the rebuilt path — against the
interpreter (bit for bit), against NumPy (to rounding), and through the
special values — plus the new multi-output ``compile_expr([...])`` and the
``E-JIT-006`` refusal of functions with no numeric implementation, which used
to compile and return NaN at every point.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from alkahest import ExprPool, compile_expr, numpy_eval, numpy_eval_par

np = pytest.importorskip("numpy")


def _same(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.array_equal(a, b, equal_nan=True) and np.array_equal(np.signbit(a), np.signbit(b))


def _relerr(got, want):
    want = np.asarray(want, dtype=float)
    return float(np.max(np.abs(np.asarray(got) - want) / np.maximum(np.abs(want), 1e-300)))


# ---------------------------------------------------------------------------
# The report's workloads, against NumPy
# ---------------------------------------------------------------------------


def _lennard_jones(p):
    r = [p.symbol(f"r{i}") for i in range(3)]
    r2 = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    energy = 4 * (r2**-6 - r2**-3)
    return ak.symbolic_grad(energy, r), r


def test_lennard_jones_gradient_matches_numpy():
    p = ExprPool()
    grad, r = _lennard_jones(p)
    rng = np.random.default_rng(1)
    pts = [rng.uniform(0.9, 2.0, 50_000) for _ in range(3)]
    a, b, c = pts
    rr = a * a + b * b + c * c
    for k, coord in enumerate(pts):
        f = compile_expr(grad[k], r, expected_evals=50_000)
        out = numpy_eval(f, *pts)
        ref = 4 * (-12 * rr**-7 + 6 * rr**-4) * coord  # dU/dr_k
        assert _relerr(out, ref) < 1e-12
        assert _same(numpy_eval_par(f, *pts), out)


def test_degree_ten_polynomial_matches_polyval():
    p = ExprPool()
    x = p.symbol("x")
    poly = sum((p.integer(i + 1) * x**i for i in range(11)), p.integer(0))
    f = compile_expr(poly, [x])
    xs = np.linspace(-1.5, 1.5, 100_001)
    out = numpy_eval(f, xs)
    ref = np.polyval(list(range(11, 0, -1)), xs)
    assert _relerr(out, ref) < 1e-13
    assert _same(numpy_eval_par(f, xs), out)


def test_bateman_and_black_scholes_match_numpy():
    p = ExprPool()
    ka, ke, v, dose, t = (p.symbol(n) for n in ("ka", "ke", "V", "D", "t"))
    bateman = dose * ka / (v * (ka - ke)) * (ak.exp(-ke * t) - ak.exp(-ka * t))
    rng = np.random.default_rng(2)
    n = 40_000
    cols = [
        rng.uniform(0.5, 2, n),
        rng.uniform(0.05, 0.3, n),
        rng.uniform(10, 50, n),
        np.full(n, 100.0),
        rng.uniform(0, 24, n),
    ]
    k_a, k_e, vv, dd, tt = cols
    ref = dd * k_a / (vv * (k_a - k_e)) * (np.exp(-k_e * tt) - np.exp(-k_a * tt))
    f = compile_expr(bateman, [ka, ke, v, dose, t])
    assert _relerr(numpy_eval(f, *cols), ref) < 1e-12

    s, k, tm, r, sg = (p.symbol(n) for n in ("S", "K", "T", "r", "sigma"))
    d1 = (ak.log(s / k) + (r + sg**2 / 2) * tm) / (sg * ak.sqrt(tm))
    d2 = d1 - sg * ak.sqrt(tm)

    def phi(z):
        return (1 + ak.erf(z / ak.sqrt(p.integer(2)))) / 2

    price = s * phi(d1) - k * ak.exp(-r * tm) * phi(d2)
    f = compile_expr(price, [s, k, tm, r, sg], expected_evals=n)
    cols = [
        rng.uniform(50, 150, n),
        rng.uniform(50, 150, n),
        rng.uniform(0.1, 2, n),
        rng.uniform(0, 0.1, n),
        rng.uniform(0.1, 0.5, n),
    ]
    out = numpy_eval(f, *cols)
    S, K, T, R, SG = cols
    D1 = (np.log(S / K) + (R + SG**2 / 2) * T) / (SG * np.sqrt(T))
    D2 = D1 - SG * np.sqrt(T)
    erf = np.vectorize(math.erf)
    ref = (
        S * (1 + erf(D1 / math.sqrt(2))) / 2 - K * np.exp(-R * T) * (1 + erf(D2 / math.sqrt(2))) / 2
    )
    # Absolute: deep out-of-the-money prices are differences of near-equal terms.
    assert float(np.max(np.abs(out - ref))) < 1e-9
    # `erf` used to push the whole expression off the JIT. Only checkable in a
    # build that has a JIT: the nightly hypothesis shard builds without one.
    if getattr(ak, "jit_is_available", lambda: False)():
        assert f._batch_tier in (None, "cranelift", "llvm") or f.tier in ("cranelift", "llvm")


# ---------------------------------------------------------------------------
# Batch ≡ the interpreter, bit for bit; special values unchanged
# ---------------------------------------------------------------------------


def test_batch_equals_eval_expr_bit_for_bit():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    exprs = [
        x**7 - 3 * x**-5 + y**10,
        ak.sqrt(x**2 + y**2),
        (x**2 + y**2) ** p.rational(-1, 2),
        ak.erf(x * y) + ak.atan2(y, x),
        ak.sin(x) * ak.exp(-(x**2)) + x**3 * y,
        ak.abs(x) ** y,
    ]
    rng = np.random.default_rng(3)
    xs = rng.uniform(-3, 3, 30_000)
    ys = rng.uniform(-2, 2, 30_000)
    for e in exprs:
        f = compile_expr(e, [x, y], expected_evals=30_000)
        out = numpy_eval(f, xs, ys)
        for i in range(0, 30_000, 997):
            try:
                want = ak.eval_expr(e, {x: float(xs[i]), y: float(ys[i])})
            except ValueError:
                continue  # eval_expr refuses non-finite results
            assert out[i] == want or (math.isnan(out[i]) and math.isnan(want)), (e, i)


def test_special_values_are_those_of_a_single_call():
    p = ExprPool()
    x = p.symbol("x")
    cases = [1 / x, x**-3, ak.log(x), ak.sqrt(x), x**3, x ** p.rational(1, 2)]
    pts = np.array([0.0, -0.0, -1.0, np.inf, -np.inf, np.nan, 1e-310, 1e200, -1e200])
    for e in cases:
        f = compile_expr(e, [x])
        out = numpy_eval(f, np.tile(pts, 2000))[: pts.size]
        single = [f([float(v)]) for v in pts]
        assert _same(out, single), e
    # pow's IEEE special cases, not sqrt's, for x**(1/2).
    f = compile_expr(x ** p.rational(1, 2), [x])
    assert math.copysign(1.0, f([-0.0])) == 1.0
    assert f([-math.inf]) == math.inf
    # (-0)**3 keeps its sign.
    g = compile_expr(x**3, [x])
    out = numpy_eval(g, np.full(10_000, -0.0))
    assert np.all(np.signbit(out))


def test_in_place_and_non_contiguous_arrays():
    p = ExprPool()
    x = p.symbol("x")
    f = compile_expr(x * x + 1, [x], expected_evals=10_000)
    xs = np.linspace(0, 1, 20_000)
    want = xs * xs + 1
    assert _same(numpy_eval(f, xs[::2]), want[::2])
    buf = xs.copy()
    # Output aliasing an input: the input is read before it is overwritten.
    f._call_batch_arrays([buf], buf)
    assert _same(buf, want)
    with pytest.raises((TypeError, ValueError)):
        f._call_batch_arrays([xs[::2]], np.empty(10_000))
    ro = np.empty(20_000)
    ro.setflags(write=False)
    with pytest.raises(ValueError):
        f._call_batch_arrays([xs], ro)
    with pytest.raises(TypeError):
        f._call_batch_arrays([xs.astype(np.float32)], np.empty(20_000))


def test_shape_is_preserved():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    f = compile_expr(x * y, [x, y])
    a = np.arange(12.0).reshape(3, 4)
    assert numpy_eval(f, a, a).shape == (3, 4)
    assert _same(numpy_eval(f, a, a), a * a)


# ---------------------------------------------------------------------------
# Multi-output compile_expr([...])
# ---------------------------------------------------------------------------


def test_multi_output_matches_each_output_alone():
    p = ExprPool()
    grad, r = _lennard_jones(p)
    g = compile_expr(grad, r)
    assert isinstance(g, ak.CompiledVecFn)
    assert (g.n_inputs, g.n_outputs) == (3, 3)
    assert "CompiledVecFn" in repr(g)
    singles = [compile_expr(e, r) for e in grad]
    point = [1.1, 0.95, 1.3]
    assert g(point) == [f(point) for f in singles]

    rng = np.random.default_rng(4)
    pts = [rng.uniform(0.9, 2.0, 30_000) for _ in range(3)]
    out = numpy_eval(g, *pts)
    assert out.shape == (3, 30_000)
    for k, f in enumerate(singles):
        assert _same(out[k], numpy_eval(f, *pts))
    assert _same(numpy_eval_par(g, *pts), out)
    # 2-D inputs: one leading axis per output.
    sq = [c[:30_000].reshape(150, 200) for c in pts]
    assert numpy_eval(g, *sq).shape == (3, 150, 200)
    # A tuple works like a list.
    assert compile_expr(tuple(grad), r)(point) == g(point)


def test_multi_output_failure_is_per_output():
    p = ExprPool()
    x = p.symbol("x")
    g = compile_expr([ak.lambert_w(x), x + 1], [x])
    w, other = g([-1.0])
    assert math.isnan(w)
    assert other == 0.0


def test_multi_output_refusals():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(ValueError):
        compile_expr([], [x])
    with pytest.raises(ak.JitError) as info:
        compile_expr([x, x + y], [x])
    assert info.value.code == "E-JIT-005"
    q = ExprPool()
    with pytest.raises(ValueError):
        compile_expr([x, q.symbol("z")], [x])


# ---------------------------------------------------------------------------
# E-JIT-006: no numeric implementation
# ---------------------------------------------------------------------------


def test_undefined_function_is_refused_not_nan():
    p = ExprPool()
    x = p.symbol("x")
    fx = p.func("f", [x])
    for expr in (fx, fx + x**2, p.func("Derivative", [fx, p.integer(1)])):
        with pytest.raises(ak.JitError) as info:
            compile_expr(expr, [x])
        assert info.value.code == "E-JIT-006"
    # The derivative `diff` now builds for an undefined function (#459).
    dfx = ak.diff(fx, x).value
    with pytest.raises(ak.JitError) as info:
        compile_expr(dfx * x, [x])
    assert info.value.code == "E-JIT-006"
    with pytest.raises(ak.JitError) as info:
        compile_expr([x, fx], [x])
    assert info.value.code == "E-JIT-006"
    assert "f" in str(info.value)
    # Unchanged: registered functions compile, unbound symbols are E-JIT-005.
    assert compile_expr(ak.erf(x), [x])([0.0]) == 0.0
    with pytest.raises(ak.JitError) as info:
        compile_expr(x + p.symbol("y"), [x])
    assert info.value.code == "E-JIT-005"
