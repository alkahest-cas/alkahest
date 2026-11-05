"""The interpreter tier of ``numpy_eval`` keeps pace with NumPy (benchmark B2).

On the 3.11.0 wheel built without a JIT, ``numpy_eval`` of a ten-operation
expression over 10^6 points took 2.4 s against 0.08 s for
``sympy.lambdify(..., "numpy")`` — about 3 µs a point, because the interpreter
walked the expression once per point. The block interpreter (#509) flattens the
DAG once and runs each operation over a block of points, which brings the same
batch to about 0.05 s, faster than ``lambdify``.

The existing absolute-speed check (``test_numpy_eval_buffer.py``) is skipped in
builds without a JIT, and in builds with one a large batch is recompiled
natively and never reaches the interpreter. The checks here run in every build:
batches below the native-upgrade threshold always run on the interpreter, so
timing them against the same work done by NumPy pins the interpreter itself.

The differential test feeds random expression trees through ``numpy_eval`` and
checks the result bit for bit against single-point calls of the same compiled
function, and against NumPy within a forward-error bound.
"""

from __future__ import annotations

import math
import warnings

import alkahest as ak
import pytest
from alkahest import ExprPool, compile_expr, numpy_eval, numpy_eval_par
from timing_util import best_of, scale

np = pytest.importorskip("numpy")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

#: Below the batch size at which a ``CompiledFn`` is recompiled for a native
#: backend (``BATCH_JIT_MIN_POINTS`` in ``alkahest-py``), so every batch here is
#: evaluated by the interpreter tier whatever features the build has.
CHUNK = 4000

REPRO = "sin(x)*exp(-x**2/3) + log(1 + x**2)*cos(3*x) + sqrt(1 + x**4) - x**5/(1 + x**2)"


def _compile(expr, inputs):
    with warnings.catch_warnings():
        # "Native JIT compilation ... is not available" in a build without one.
        warnings.simplefilter("ignore", RuntimeWarning)
        return compile_expr(expr, inputs)


def _repro_numpy(x):
    return (
        np.sin(x) * np.exp(-(x**2) / 3)
        + np.log(1 + x**2) * np.cos(3 * x)
        + np.sqrt(1 + x**4)
        - x**5 / (1 + x**2)
    )


def _same(a, b):
    """Bit for bit, except that a NaN's sign and payload carry no meaning (which
    NaN operand an IEEE operation propagates depends on operand order)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = ~np.isnan(a)
    return np.array_equal(a, b, equal_nan=True) and np.array_equal(
        np.signbit(a[num]), np.signbit(b[num])
    )


# ---------------------------------------------------------------------------
# The benchmark's expression: values and throughput
# ---------------------------------------------------------------------------


def test_repro_values_match_single_calls_and_numpy():
    p = ExprPool()
    x = p.symbol("x")
    f = _compile(ak.parse(REPRO, p), [x])
    xs = np.linspace(0.1, 5.0, CHUNK)
    out = numpy_eval(f, xs)
    assert f._batch_tier is None, "a sub-threshold batch must stay on the interpreter"
    assert _same(out, [f([v]) for v in xs])
    # `sqrt(1 + x^4) - x^5/(1 + x^2)` cancels, so the bound is on the terms'
    # magnitude, not on the (smaller) result.
    mag = 2 + np.log(1 + xs**2) + np.sqrt(1 + xs**4) + xs**5 / (1 + xs**2)
    assert np.all(np.abs(out - _repro_numpy(xs)) <= 1e-14 * mag)


def test_interpreter_batches_keep_pace_with_numpy():
    """Throughput of the interpreter tier, measured against NumPy doing the
    same work in the same chunks (so per-call overhead is charged to both).

    Before the block interpreter the ratio was about 40; now it is below 1.
    The bound leaves an order of magnitude for noisy runners.
    """
    p = ExprPool()
    x = p.symbol("x")
    f = _compile(ak.parse(REPRO, p), [x])
    chunks = np.array_split(np.linspace(0.1, 5.0, 50 * CHUNK), 50)

    def run_alkahest():
        return [numpy_eval(f, c) for c in chunks]

    def run_numpy():
        return [_repro_numpy(c) for c in chunks]

    run_alkahest(), run_numpy()  # warm-up
    got, t_alkahest = best_of(run_alkahest, runs=5)
    want, t_numpy = best_of(run_numpy, runs=5)
    assert f._batch_tier is None, "the timed batches must have run on the interpreter"
    for g, w, c in zip(got, want, chunks):
        mag = 2 + np.log(1 + c**2) + np.sqrt(1 + c**4) + c**5 / (1 + c**2)
        assert np.all(np.abs(g - w) <= 1e-14 * mag)
    ratio = t_alkahest / t_numpy
    assert ratio < 8 * scale(), (
        f"interpreter numpy_eval took {t_alkahest:.4f}s for {50 * CHUNK} points, "
        f"{ratio:.1f}x NumPy's {t_numpy:.4f}s"
    )


def test_poly_benchmark_values_and_parallel_agree():
    """`benchmarks/tasks.py::poly_jit_eval` at its largest size, through the
    sequential and the parallel entry points."""
    p = ExprPool()
    x = p.symbol("x")
    poly = x**0
    for k in range(1, 21):
        poly = poly + x**k
    f = _compile(poly, [x])
    xs = np.linspace(0.0, 1.0, CHUNK)
    out = numpy_eval(f, xs)
    assert _same(out, numpy_eval_par(f, xs))
    assert _same(out, [f([v]) for v in xs])
    want = sum(xs**k for k in range(21))
    np.testing.assert_allclose(out, want, rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------
# Random expression trees: batch ≡ single point (bitwise) ≈ NumPy (bounded)
# ---------------------------------------------------------------------------

# Each node evaluates to `(alkahest expr, numpy value, magnitude)`, where the
# magnitude bounds the forward error of a straightforward evaluation by a
# small multiple of `eps * magnitude`: it is what the value would be with
# every subtraction turned into an addition, propagated through each
# function by its derivative.

UNARY = ["sin", "cos", "exp", "log", "sqrt", "atan", "tanh", "abs"]


def _unary(name, a, va, ma):
    with np.errstate(all="ignore"):
        if name == "sin":
            return ak.sin(a), np.sin(va), 1 + ma
        if name == "cos":
            return ak.cos(a), np.cos(va), 1 + ma
        if name == "exp":
            v = np.exp(va)
            return ak.exp(a), v, v * (1 + ma)
        if name == "log":
            v = np.log(va)
            return ak.log(a), v, np.abs(v) + ma / np.abs(va)
        if name == "sqrt":
            v = np.sqrt(va)
            return ak.sqrt(a), v, v + ma / (2 * v)
        if name == "atan":
            return ak.atan(a), np.arctan(va), 2 + ma
        if name == "tanh":
            return ak.tanh(a), np.tanh(va), 1 + ma
        if name == "abs":
            return ak.abs(a), np.abs(va), ma
    raise AssertionError(name)


@st.composite
def _trees(draw, pool, x, y, xs, ys, depth):
    if depth == 0 or draw(st.integers(0, 3)) == 0:
        leaf = draw(st.sampled_from(["x", "y", "c"]))
        if leaf == "x":
            return x, xs, np.abs(xs)
        if leaf == "y":
            return y, ys, np.abs(ys)
        c = draw(st.sampled_from([-3, -2, -1, 2, 3, 5]))
        return pool.integer(c), np.full_like(xs, c), np.full_like(xs, abs(c))
    kind = draw(st.sampled_from(["add", "sub", "mul", "div", "pow", "unary"]))
    a, va, ma = draw(_trees(pool, x, y, xs, ys, depth - 1))
    with np.errstate(all="ignore"):
        if kind == "unary":
            return _unary(draw(st.sampled_from(UNARY)), a, va, ma)
        if kind == "pow":
            n = draw(st.sampled_from([-3, -2, -1, 2, 3, 4, 7]))
            v = va ** float(n)
            return a**n, v, np.abs(v) * (1 + abs(n) * ma / np.abs(va))
        b, vb, mb = draw(_trees(pool, x, y, xs, ys, depth - 1))
        if kind == "add":
            return a + b, va + vb, ma + mb
        if kind == "sub":
            return a - b, va - vb, ma + mb
        if kind == "mul":
            return a * b, va * vb, ma * mb + np.abs(va) * mb + ma * np.abs(vb)
        return a / b, va / vb, (ma + np.abs(va / vb) * mb) / np.abs(vb)


_RNG = np.random.default_rng(20261025)
_XS = _RNG.uniform(-2.5, 2.5, 1500)
_YS = _RNG.uniform(-2.5, 2.5, 1500)
# The IEEE specials go through every operation too.
_XS[:6] = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0]
_YS[:6] = [1.0, 0.5, -1.0, 2.0, 0.0, np.nan]


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(data=st.data())
def test_random_trees_batch_equals_single_point_and_numpy(data):
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    expr, want, mag = data.draw(_trees(p, x, y, _XS, _YS, 4))
    f = _compile(expr, [x, y])
    out = numpy_eval(f, _XS, _YS)
    assert f._batch_tier is None

    # Bit for bit what a single-point call of the same function returns.
    single = np.array([f([a, b]) for a, b in zip(_XS, _YS)])
    assert _same(out, single), str(expr)

    # NumPy as an oracle, on the finite points: the two evaluations may round
    # differently (`a/b` is `a·b⁻¹` here; powers are compensated), never by
    # more than a few ulps of the magnitude.
    with np.errstate(all="ignore"):
        both = np.isfinite(out) & np.isfinite(want) & np.isfinite(mag) & (mag < 1e250)
    err = np.abs(out[both] - want[both])
    tol = 1e-12 * mag[both]
    bad = np.flatnonzero(err > tol)
    assert bad.size == 0, (str(expr), _XS[both][bad[:3]], out[both][bad[:3]], want[both][bad[:3]])

    # A domain error is per point: alkahest has a value wherever NumPy has a
    # finite one. (The converse does not hold: a real value reached through a
    # complex intermediate, `sqrt(x)**2` at `x < 0`, is NaN in NumPy and a
    # number here — the complex fallback of `CompiledFn.call`.)
    with np.errstate(all="ignore"):
        has_value = np.isfinite(want) & np.isfinite(mag) & (mag < 1e250)
    lost = np.flatnonzero(has_value & np.isnan(out))
    assert lost.size == 0, (str(expr), _XS[lost[:3]], _YS[lost[:3]], want[lost[:3]])


def test_domain_errors_are_nan_at_exactly_the_failing_points():
    p = ExprPool()
    x = p.symbol("x")
    f = _compile(ak.log(x) + ak.sqrt(x - 1), [x])
    xs = np.array([-1.0, 0.5, 1.0, 2.0, math.e, np.nan, np.inf] * 100)
    out = numpy_eval(f, xs)
    with np.errstate(all="ignore"):
        want = np.log(xs) + np.sqrt(xs - 1)
    assert np.array_equal(np.isnan(out), np.isnan(want))
    np.testing.assert_allclose(out[~np.isnan(out)], want[~np.isnan(want)], rtol=1e-15)
