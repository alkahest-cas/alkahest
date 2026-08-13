"""Thread-safety of the parallel entry points across the PyO3 boundary.

Why this file exists
--------------------
`parallel` now ships in the default PyPI wheel, so the sharded `ExprPool` and
the Rayon fan-out in `simplify_par` / `numpy_eval_par` are on the code path of
every user rather than of the handful who installed a `+full` wheel.

No sanitizer job runs `pytest` (see ``CONTRIBUTING.md`` and ``TESTING.md`` §3),
so a race that only manifests *through* PyO3 — `ExprPool` is a plain, sendable
`#[pyclass]`, and `py_simplify_par` holds a `PyRef` borrow across
`Python::allow_threads` while `&ExprPool` escapes into a Rayon pool — is not
caught by the Rust-side ASan/TSan shards, which never construct a Python
object. The Rust-side counterpart of this file is
``alkahest-core/tests/parallel_stress.rs``, which the nightly `tsan` shard runs
with `--features parallel`; this file is the part of that story that only
exists above the FFI boundary.

These are invariant tests, not timing tests: they assert that concurrent
callers get the *same answers* a lone caller gets, and that nothing raises,
crashes, or corrupts the pool's hash-consing. They deliberately share one
`ExprPool` between threads, because that is the configuration that is unsound
if anything here is wrong.
"""

import threading

import numpy as np
import pytest
from alkahest import (
    ExprPool,
    capabilities,
    compile_expr,
    numpy_eval,
    numpy_eval_par,
    simplify,
    simplify_auto,
    simplify_par,
    simplify_redex,
)

THREADS = 8
JOIN_TIMEOUT = 120.0

HAS_PARALLEL = bool(capabilities()["features"]["parallel"])

# On a build without `--features parallel` every `*_par` entry point is a
# silent alias for its sequential twin, so these tests still pass — they just
# stop testing anything about threads. Say so rather than reporting a green
# tick for coverage that did not happen.
requires_parallel = pytest.mark.skipif(
    not HAS_PARALLEL,
    reason="built without --features parallel; *_par aliases its sequential twin, "
    "so this asserts nothing about concurrency",
)


def run_in_lockstep(work):
    """Run ``work(thread_index)`` on ``THREADS`` threads that start together.

    Without the barrier the first thread routinely finishes before the last is
    scheduled, every intern hits the read-only fast path, and the test passes
    without ever contending a pool shard. Exceptions are collected and
    re-raised on the main thread — a bare ``threading.Thread`` swallows them,
    which would turn a crash on a worker into a silent pass.
    """
    barrier = threading.Barrier(THREADS)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def target(index: int) -> None:
        try:
            barrier.wait()
            work(index)
        except BaseException as exc:  # reported on the main thread, not swallowed
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=target, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=JOIN_TIMEOUT)
    for t in threads:
        assert not t.is_alive(), "worker thread did not finish within the timeout"
    if errors:
        raise AssertionError(f"worker thread raised: {errors[0]!r}") from errors[0]


def build_shape(p, x, k):
    """``((x + k) * (x + k + 1))**2 + k`` — deep enough to fork, cheap enough to loop."""
    return ((x + p.integer(k)) * (x + p.integer(k + 1))) ** p.integer(2) + p.integer(k)


class TestSharedPoolInterning:
    def test_concurrent_symbol_and_expression_building_is_hash_consed(self):
        """Structural identity must still imply id identity under contention.

        `ExprPool.intern` uses a `DashMap` shard write-lock around the node
        push precisely so two threads cannot both insert the same key. If that
        broke, nothing would raise — the pool would just quietly hold two nodes
        for one structure and `==` would stop agreeing with structural
        equality. Comparing rendered forms across threads is the detector.
        """
        p = ExprPool()
        x = p.symbol("x", "real")
        expected = [str(build_shape(p, x, k)) for k in range(40)]

        results: dict[int, list[str]] = {}
        lock = threading.Lock()

        def work(index: int) -> None:
            got = [str(build_shape(p, x, k)) for k in range(40)]
            with lock:
                results[index] = got

        run_in_lockstep(work)

        assert len(results) == THREADS
        for index, got in results.items():
            assert got == expected, f"thread {index} built a different expression"

    def test_distinct_symbols_per_thread_do_not_collide(self):
        p = ExprPool()

        seen: dict[int, list[str]] = {}
        lock = threading.Lock()

        def work(index: int) -> None:
            y = p.symbol(f"y{index}", "real")
            rendered = [str(build_shape(p, y, k)) for k in range(30)]
            with lock:
                seen[index] = rendered

        run_in_lockstep(work)

        flat = [r for rendered in seen.values() for r in rendered]
        assert len(set(flat)) == len(flat), "distinct per-thread structures collided"
        for index, rendered in seen.items():
            assert all(f"y{index}" in r for r in rendered)


class TestParallelSimplifyAcrossThreads:
    """`py_simplify_*` release the GIL and hand `&ExprPool` to Rayon.

    The `PyRef<PyExprPool>` borrow is held across `allow_threads`, so these are
    the calls where several Python threads have live native references into one
    pool at the same time.
    """

    @requires_parallel
    @pytest.mark.parametrize("fn", [simplify_par, simplify_redex, simplify_auto])
    def test_matches_sequential_when_called_concurrently(self, fn):
        p = ExprPool()
        x = p.symbol("x", "real")
        inputs = [build_shape(p, x, k) for k in range(THREADS)]
        expected = [str(simplify(e).value) for e in inputs]

        def work(index: int) -> None:
            for round_ in range(4):
                i = (index + round_) % len(inputs)
                got = str(fn(inputs[i]).value)
                assert got == expected[i], (
                    f"{fn.__name__} on thread {index} round {round_} gave {got!r}, "
                    f"sequential simplify gave {expected[i]!r}"
                )

        run_in_lockstep(work)

    @requires_parallel
    def test_interning_while_simplify_par_runs_gil_free(self):
        """Half the threads simplify, half keep interning into the same pool.

        Nothing in the API stops a caller doing this, and it is the one
        combination neither the sequential tests nor the Rust-internal Rayon
        tests reach: new nodes appear in the pool *while* a GIL-free
        `simplify_par` is walking it.
        """
        p = ExprPool()
        x = p.symbol("x", "real")
        target = build_shape(p, x, 3)
        expected = str(simplify(target).value)

        def work(index: int) -> None:
            if index % 2 == 0:
                for _ in range(4):
                    assert str(simplify_par(target).value) == expected
            else:
                y = p.symbol(f"z{index}", "real")
                for k in range(60):
                    assert str(build_shape(p, y, k))

        run_in_lockstep(work)


class TestNumpyEvalParAcrossThreads:
    @requires_parallel
    def test_concurrent_numpy_eval_par_matches_sequential(self):
        """`numpy_eval_par` copies the buffers before dropping the GIL, then
        Rayon-fans-out over the copy. Several threads each running that fan-out
        at once is the nested case `CompiledFn`'s `unsafe impl Sync` has to hold
        up under.

        Each thread compiles its own `CompiledFn`, because a `CompiledFn` may
        not cross threads at all — see
        ``test_compiled_fn_may_not_be_used_from_another_thread``. The pool *is*
        shared, so this also exercises concurrent interning underneath
        `compile_expr`.
        """
        p = ExprPool()
        x = p.symbol("x", "real")
        expr = build_shape(p, x, 1)
        xs = np.linspace(0.0, 1.0, 4096)
        want = numpy_eval(compile_expr(expr, [x]), xs)

        def work(index: int) -> None:
            f = compile_expr(expr, [x])
            for _ in range(3):
                got = numpy_eval_par(f, xs)
                np.testing.assert_allclose(
                    got,
                    want,
                    rtol=1e-12,
                    atol=0.0,
                    err_msg=f"numpy_eval_par disagreed with numpy_eval on thread {index}",
                )

        run_in_lockstep(work)

    def test_compiled_fn_may_not_be_used_from_another_thread(self):
        """A `CompiledFn` is pinned to the thread that created it.

        `PyCompiledFn` is `#[pyclass(unsendable)]` — it owns JIT code pages and
        a backend handle — so PyO3 checks the owning thread on *every* access
        and panics if it differs. That check is a safety net, not a bug: it
        fires before anything unsound happens. But it is a sharp edge worth
        pinning down, because `parallel` now ships by default and the obvious
        way to use it wrongly is to compile once and fan the handle out over a
        thread pool.

        Two details make it worse than an ordinary error, and are the reason
        this test exists rather than a docs line alone: it applies to plain
        `numpy_eval` just as much as to `numpy_eval_par` — parallelism has
        nothing to do with it — and `pyo3_runtime.PanicException` derives from
        `BaseException`, not `Exception`, so a worker wrapped in a bare
        ``except Exception`` will not catch it.

        Compile per thread (as the test above does); expressions and the
        `ExprPool` itself are shareable.
        """
        p = ExprPool()
        x = p.symbol("x", "real")
        f = compile_expr(build_shape(p, x, 1), [x])
        xs = np.linspace(0.0, 1.0, 64)
        numpy_eval(f, xs)  # fine on the creating thread

        captured: list[BaseException] = []

        def work() -> None:
            try:
                numpy_eval(f, xs)
            except BaseException as exc:  # catching BaseException is the point here
                captured.append(exc)

        t = threading.Thread(target=work)
        t.start()
        t.join(timeout=JOIN_TIMEOUT)

        assert captured, (
            "using a CompiledFn from a foreign thread was expected to be refused; "
            "if this now succeeds, PyCompiledFn's `unsendable` marker changed and "
            "the thread-affinity docs need updating"
        )
        assert "unsendable" in str(captured[0])
        assert not isinstance(captured[0], Exception), (
            "PanicException is expected to derive from BaseException only — if that "
            "changed, the warning about bare `except Exception` can be relaxed"
        )


def test_capabilities_parallel_flag_is_a_bool():
    """`capabilities()["features"]["parallel"]` is the documented way to tell
    whether `*_par` is really parallel; it must not go missing or become a
    truthy string.
    """
    features = capabilities()["features"]
    assert "parallel" in features
    assert isinstance(features["parallel"], bool)
