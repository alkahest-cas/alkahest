"""Regression + perf-sanity tests for the `numpy_eval` buffer fast path.

Guards against the `.tolist()` + `Vec<f64>` round-trip performance cliff
described in `temp-alkahest/planning/report7-20.md` (`numpy_eval` over 1M
points was ~25x slower than `sympy.lambdify`). `CompiledFn.call_batch_buffer`
/ `call_batch_buffer_par` read NumPy buffers directly (no per-element Python
`float` boxing) and release the GIL during the native call; `numpy_eval` /
`numpy_eval_par` use them automatically when available.

NumPy is an optional runtime dependency (not guaranteed present in the lean
dev venv), so every test below imports it lazily via `pytest.importorskip`.
"""

import time

import pytest
from alkahest import ExprPool, compile_expr, numpy_eval, numpy_eval_par


def _quadratic_fn():
    p = ExprPool()
    x = p.symbol("x")
    expr = x**2 + p.integer(1)
    return compile_expr(expr, [x])


class TestBufferFastPathAvailability:
    def test_call_batch_buffer_exists(self):
        f = _quadratic_fn()
        assert hasattr(f, "call_batch_buffer"), (
            "CompiledFn.call_batch_buffer is missing -- numpy_eval would fall "
            "back to the slow .tolist() + call_batch_raw path"
        )

    def test_numpy_eval_uses_buffer_fast_path(self):
        """numpy_eval must prefer call_batch_buffer over call_batch_raw.

        `CompiledFn` is a native PyO3 class without a `__dict__`, so its
        methods cannot be monkeypatched directly; instead we wrap it in a
        thin counting proxy that `numpy_eval` interacts with via duck typing
        (it only ever calls `.n_inputs` and the `call_batch_*` methods).
        """
        np = pytest.importorskip("numpy")
        f = _quadratic_fn()
        calls = {"buffer": 0, "raw": 0}

        class _CountingProxy:
            n_inputs = f.n_inputs

            def call_batch_buffer(self, *a, **kw):
                calls["buffer"] += 1
                return f.call_batch_buffer(*a, **kw)

            def call_batch_raw(self, *a, **kw):
                calls["raw"] += 1
                return f.call_batch_raw(*a, **kw)

        xs = np.linspace(0.0, 1.0, 1000)
        result = numpy_eval(_CountingProxy(), xs)

        assert calls["buffer"] == 1, "numpy_eval did not call the buffer fast path"
        assert calls["raw"] == 0, "numpy_eval fell back to the slow .tolist() path"
        assert np.allclose(result, xs**2 + 1.0)


@pytest.mark.parametrize("n_points", [100_000, 1_000_000])
class TestCorrectnessAtScale:
    def test_numpy_eval_matches_scalar_loop(self, n_points):
        np = pytest.importorskip("numpy")
        f = _quadratic_fn()
        xs = np.linspace(-3.0, 3.0, n_points)

        ys = numpy_eval(f, xs)
        stride = max(1, n_points // 2000)
        scalar_ref = np.array([float(v) ** 2 + 1.0 for v in xs[::stride]])
        vectorized_ref = xs**2 + 1.0

        assert ys.shape == xs.shape
        assert np.allclose(ys, vectorized_ref, atol=1e-9)
        # Cross-check a subsample against a pure Python scalar loop too, to
        # rule out any systematic error introduced by buffer marshaling.
        assert np.allclose(ys[::stride], scalar_ref, atol=1e-9)

    def test_numpy_eval_par_matches_numpy_eval(self, n_points):
        np = pytest.importorskip("numpy")
        f = _quadratic_fn()
        xs = np.linspace(-3.0, 3.0, n_points)

        ys_seq = numpy_eval(f, xs)
        ys_par = numpy_eval_par(f, xs)

        assert np.allclose(ys_seq, ys_par, atol=1e-12)
        assert np.allclose(ys_par, xs**2 + 1.0, atol=1e-9)

    def test_call_batch_buffer_matches_scalar_call(self, n_points):
        """Exercise CompiledFn.call_batch_buffer directly (bypassing numpy_eval)."""
        np = pytest.importorskip("numpy")
        f = _quadratic_fn()
        xs = np.ascontiguousarray(np.linspace(-2.0, 2.0, n_points), dtype=np.float64)
        out = np.empty(n_points, dtype=np.float64)

        f.call_batch_buffer([xs], out)

        assert np.allclose(out, xs**2 + 1.0, atol=1e-9)
        # A handful of scalar calls must agree exactly with the batch result.
        for i in (0, n_points // 3, n_points - 1):
            assert abs(f([float(xs[i])]) - out[i]) < 1e-9


class TestPerfSanity:
    """Soft perf check: the buffer fast path should not regress vs a naive
    pure-Python scalar loop over the same expression, and should comfortably
    beat the legacy `.tolist()` + `call_batch_raw` marshaling path.

    These are wall-clock soft checks (generous margins, no hard SLA) so they
    stay robust on shared/slow CI runners; they exist to catch a reintroduced
    O(N) Python-object-boxing regression, not to enforce a specific speed.
    """

    def test_buffer_path_faster_than_or_comparable_to_python_loop(self):
        np = pytest.importorskip("numpy")
        f = _quadratic_fn()
        n_points = 200_000
        xs = np.linspace(-3.0, 3.0, n_points)

        t0 = time.perf_counter()
        ys = numpy_eval(f, xs)
        native_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        py_ref = np.array([v**2 + 1.0 for v in xs.tolist()])
        python_loop_elapsed = time.perf_counter() - t0

        assert np.allclose(ys, py_ref, atol=1e-9)
        # Generous margin: native path should not be dramatically slower
        # than an equivalent pure-Python loop doing the same math.
        assert native_elapsed < python_loop_elapsed * 3 + 0.5, (
            f"numpy_eval ({native_elapsed * 1000:.1f} ms) is much slower than a "
            f"pure-Python loop ({python_loop_elapsed * 1000:.1f} ms) for "
            f"{n_points} points -- possible regression to per-element boxing"
        )

    def test_buffer_path_not_slower_than_legacy_tolist_path(self):
        """The new call_batch_buffer path must not regress vs the old
        `.tolist()` + `call_batch_raw` marshaling it replaces."""
        np = pytest.importorskip("numpy")
        f = _quadratic_fn()
        n_points = 300_000
        xs = np.linspace(-3.0, 3.0, n_points)

        t0 = time.perf_counter()
        out = np.empty(n_points, dtype=np.float64)
        f.call_batch_buffer([xs], out)
        fast_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        inputs_flat = xs.tolist()
        legacy = np.array(f.call_batch_raw(inputs_flat, 1, n_points), dtype=np.float64)
        legacy_elapsed = time.perf_counter() - t0

        assert np.allclose(out, legacy)
        # 20% slack for noise; the fast path does strictly less work
        # (no Python list of N floats built or unboxed).
        assert fast_elapsed < legacy_elapsed * 1.2 + 0.05, (
            f"call_batch_buffer ({fast_elapsed * 1000:.1f} ms) regressed vs "
            f"legacy call_batch_raw ({legacy_elapsed * 1000:.1f} ms)"
        )

    def test_parallel_path_available_and_correct(self):
        pytest.importorskip("numpy")
        f = _quadratic_fn()
        if not hasattr(f, "call_batch_buffer_par"):
            pytest.skip("parallel feature not compiled in")
        import numpy as np

        n_points = 500_000
        xs = np.linspace(-3.0, 3.0, n_points)
        ys = numpy_eval_par(f, xs)
        assert np.allclose(ys, xs**2 + 1.0, atol=1e-9)
