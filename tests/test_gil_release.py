"""Lazy ``DerivedResult`` rendering and GIL release around the heavy entry points.

Two changes are pinned here.

1. ``DerivedResult.derivation`` / ``.steps`` are rendered from the stored
   rewrite log on first access instead of when the result is built. The text
   must be exactly what eager rendering produced, whatever order the
   attributes are read in, and however many times.

2. ``simplify``, ``diff``, ``solve``, ``evaluate``, ``factor_z`` and the other
   heavy entry points run their core call under ``py.allow_threads``. That is
   only worth having if a thread pool actually overlaps them, and only sound if
   the thread-local channels — the budget frame, explicit assumptions, the
   ``*_side_conditions()`` side channels — still describe the calling thread's
   own call. Both halves are asserted.
"""

import os
import threading
import time

import alkahest as ak
import pytest

THREADS = 8
JOIN_TIMEOUT = 120.0


def run_in_lockstep(work, threads=THREADS):
    """Run ``work(i)`` on ``threads`` threads released together; re-raise errors."""
    barrier = threading.Barrier(threads)
    errors = []
    lock = threading.Lock()

    def target(index):
        try:
            barrier.wait()
            work(index)
        except BaseException as exc:  # reported on the main thread
            with lock:
                errors.append(exc)

    ts = [threading.Thread(target=target, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=JOIN_TIMEOUT)
    for t in ts:
        assert not t.is_alive(), "worker thread did not finish within the timeout"
    if errors:
        raise AssertionError(f"worker thread raised: {errors[0]!r}") from errors[0]


def poly(p, x, y, n):
    return p.add([p.mul([p.integer(i + 1), x**i, y ** (n - i)]) for i in range(n)])


def expected_derivation(steps):
    """The ``derivation`` text, rebuilt from ``steps`` in ``DerivationLog``'s format."""
    lines = []
    for i, s in enumerate(steps):
        line = f"step {i + 1}: {s['rule']} applied to {s['before']} → {s['after']}"
        line += "".join(f"  [{c}]" for c in s["side_conditions"])
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Lazy rendering
# ---------------------------------------------------------------------------


class TestLazyDerivation:
    def test_derivation_matches_steps(self):
        p = ak.ExprPool()
        x, y = p.symbol("x"), p.symbol("y")
        r = ak.diff(poly(p, x, y, 20), x)
        steps = r.steps
        assert len(steps) > 20
        assert r.derivation == expected_derivation(steps)

    def test_read_order_does_not_change_text(self):
        """``derivation`` first, ``steps`` first, or neither yet — same strings."""
        p = ak.ExprPool()
        x, y = p.symbol("x"), p.symbol("y")
        e = poly(p, x, y, 15)
        a = ak.diff(e, x)
        b = ak.diff(e, x)
        a_derivation = a.derivation
        b_steps = b.steps
        assert a.steps == b_steps
        assert b.derivation == a_derivation
        # Repeated reads return the same text.
        assert a.derivation == a_derivation
        assert a.steps == b_steps

    def test_side_conditions_agree_before_and_after_steps_are_rendered(self):
        """``verification`` renders side conditions itself when ``.steps`` has not."""
        p = ak.ExprPool()
        u = p.symbol("u", "positive")
        e = ak.log(u * u)
        fresh = ak.simplify_log_exp(e)
        cold = fresh.verification["side_conditions"]
        warm_result = ak.simplify_log_exp(e)
        steps = warm_result.steps
        warm = warm_result.verification["side_conditions"]
        flat = [c for s in steps for c in s["side_conditions"]]
        assert cold == warm == flat
        assert flat, "log(u·u) → log u + log u should record u > 0"
        assert all(c == "u > 0" for c in flat)

    def test_to_dict_steps_match_steps(self):
        p = ak.ExprPool()
        x = p.symbol("x")
        r = ak.diff(ak.sin(x) * x**3, x)
        d = r.to_dict()
        assert d["steps"] == r.steps
        compact = ak.diff(ak.sin(x) * x**3, x).to_dict(mode="compact")
        assert [s["r"] for s in compact["steps"]] == [s["rule"] for s in r.steps]

    def test_context_simplify_carries_the_log(self):
        p = ak.ExprPool()
        x = p.symbol("x")
        plain = ak.diff(x**3, x)
        with ak.context(simplify=True):
            ctx = ak.diff(x**3, x)
        # Either the value was already simplified (log carried over unchanged)
        # or a `context_simplify` step was appended to it.
        assert ctx.derivation.startswith(plain.derivation)
        assert ctx.derivation == expected_derivation(ctx.steps)

    def test_value_only_callers_do_not_pay_for_rendering(self):
        """Reading only ``.value`` must not be dominated by the log's text.

        Loose on purpose: on a 400-term polynomial eager rendering roughly
        doubled the call, so ``.value`` alone being no slower than the call
        that also renders is a floor, not a benchmark.
        """
        p = ak.ExprPool()
        x, y = p.symbol("x"), p.symbol("y")
        e = poly(p, x, y, 400)
        ak.diff(e, x)  # warm the pool

        # CPU time, not wall time: a loaded runner stretches wall time for
        # reasons that have nothing to do with this call.
        def best(fn, n=7):
            out = float("inf")
            for _ in range(n):
                t0 = time.process_time()
                fn()
                out = min(out, time.process_time() - t0)
            return out

        value_only = best(lambda: ak.diff(e, x).value)
        rendered = best(lambda: (lambda r: (r.derivation, r.steps))(ak.diff(e, x)))
        assert value_only < rendered


# ---------------------------------------------------------------------------
# 2. GIL release
# ---------------------------------------------------------------------------


def _serial_vs_threads(fn, calls=32, threads=THREADS, rounds=3):
    """Best-of-``rounds`` wall time for ``calls`` calls, serially and threaded."""
    fn()
    serial = threaded = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        serial = min(serial, time.perf_counter() - t0)

        per_thread = calls // threads

        def work(_index, per_thread=per_thread):
            for _ in range(per_thread):
                fn()

        t0 = time.perf_counter()
        run_in_lockstep(work, threads)
        threaded = min(threaded, time.perf_counter() - t0)
    return serial, threaded


needs_cores = pytest.mark.skipif(
    (os.cpu_count() or 1) < 2, reason="needs at least two CPUs to overlap anything"
)


@needs_cores
class TestThreadsOverlap:
    """A thread pool must not make these calls slower than running them in a row.

    Holding the GIL for the whole call made eight threads *slower* than one
    for ``diff`` (0.51× measured) — the threads only fought over the lock. The
    bound is deliberately loose (not slower than serial × 1.2) so a busy CI
    runner cannot flake it; on an idle multi-core machine the ratio is well
    above 1.
    """

    def test_diff(self):
        p = ak.ExprPool()
        x, y = p.symbol("x"), p.symbol("y")
        e = poly(p, x, y, 150)
        serial, threaded = _serial_vs_threads(lambda: ak.diff(e, x))
        assert threaded <= serial * 1.2, (serial, threaded)

    def test_simplify(self):
        p = ak.ExprPool()
        x, y = p.symbol("x"), p.symbol("y")
        e = poly(p, x, y, 250) * p.integer(1) + p.integer(0)
        serial, threaded = _serial_vs_threads(lambda: ak.simplify(e))
        assert threaded <= serial * 1.2, (serial, threaded)


class TestContextsSurviveReleasedSections:
    """Everything thread-local is still the caller's inside ``allow_threads``."""

    def test_budget_frame_is_per_thread_inside_simplify(self):
        """An exhausted budget on one thread stops only that thread's simplify.

        ``simplify`` consults the budget once per pass and, having no error
        channel, returns what it has when the budget trips — so under
        ``max_steps=0`` it returns its input untouched. The threads without a
        budget must still simplify fully while their neighbours are tripping.
        """
        p = ak.ExprPool()
        x = p.symbol("x")
        e = x * p.integer(1) + x * p.integer(0) + ak.sin(x) ** 2 * p.integer(0)
        full = ak.simplify(e).value
        assert full != e
        results = {}
        lock = threading.Lock()

        def work(i):
            for _ in range(20):
                if i % 2:
                    with ak.context(budget=ak.Budget(max_steps=0)):
                        got = ak.simplify(e).value
                else:
                    got = ak.simplify(e).value
                with lock:
                    results.setdefault(i, set()).add(got)

        run_in_lockstep(work)
        for i, got in results.items():
            assert got == ({e} if i % 2 else {full}), (i, got)

    def test_explicit_assumptions_apply_on_every_thread(self):
        p = ak.ExprPool()
        x = p.symbol("x")
        assumptions = ak.Assumptions(p)
        assumptions.refine(p.gt(x, p.integer(0)))
        e = ak.sqrt(x**2)
        assert assumptions.simplify(e).value == x
        assert ak.simplify(e).value != x
        bad = []

        def work(i):
            for _ in range(20):
                ok = assumptions.simplify(e).value == x if i % 2 else ak.simplify(e).value != x
                if not ok:
                    bad.append(i)

        run_in_lockstep(work)
        assert not bad

    def test_refine_while_another_thread_simplifies(self):
        """``Assumptions.simplify`` must not hold its borrow across the released call.

        Otherwise a ``refine`` on another thread, which could not run at all
        while the GIL was held, would now fail with ``Already borrowed``.
        """
        p = ak.ExprPool()
        x = p.symbol("x")
        y = p.symbol("y")
        assumptions = ak.Assumptions(p)
        assumptions.refine(p.gt(x, p.integer(0)))
        e = ak.sqrt(x**2) + poly(p, x, y, 40)

        def work(i):
            for _ in range(20):
                if i == 0:
                    assumptions.refine(p.gt(y, p.integer(0)))
                else:
                    assumptions.simplify(e)

        run_in_lockstep(work)

    def test_solve_side_conditions_describe_this_threads_call(self):
        """The side channel is drained on the thread that ran the solve."""
        p = ak.ExprPool()
        x, a, b = p.symbol("x"), p.symbol("a"), p.symbol("b")
        bad = []

        def work(i):
            for _ in range(10):
                if i % 2:
                    ak.solve([a * x - b], [x])
                    want = ["a ≠ 0"]
                else:
                    ak.solve([x**2 - 1], [x])
                    want = []
                got = ak.solve_side_conditions()
                if got != want:
                    bad.append((i, got))

        try:
            ak.solve([x - 1], [x])
        except Exception as exc:  # built without groebner
            pytest.skip(f"solve unavailable: {exc}")
        run_in_lockstep(work)
        assert not bad, bad[:3]

    def test_cancel_request_reaches_a_released_simplify(self):
        """``request_cancel`` stops simplify's pass loop from any thread."""
        p = ak.ExprPool()
        x = p.symbol("x")
        e = x * p.integer(1) + p.integer(0)
        try:
            ak.request_cancel()
            with ak.context(budget=ak.Budget()):
                got = ak.simplify(e).value
        finally:
            ak.clear_cancel()
        assert got == e
        assert ak.simplify(e).value == x

    def test_concurrent_results_equal_serial(self):
        """Released calls on a shared pool give the answers a lone caller gets."""
        p = ak.ExprPool()
        x, y = p.symbol("x"), p.symbol("y")
        exprs = [poly(p, x, y, 10 + k) for k in range(8)]
        want_diff = [ak.diff(e, x).value for e in exprs]
        want_simp = [ak.simplify(e * p.integer(1) + p.integer(0)).value for e in exprs]
        want_cancel = [
            ak.cancel((x**2 - p.integer(k * k)) / (x - p.integer(k))) for k in range(1, 9)
        ]
        bad = []

        def work(i):
            for k, e in enumerate(exprs):
                if ak.diff(e, x).value != want_diff[k]:
                    bad.append(("diff", i, k))
                if ak.simplify(e * p.integer(1) + p.integer(0)).value != want_simp[k]:
                    bad.append(("simplify", i, k))
                c = ak.cancel((x**2 - p.integer((k + 1) ** 2)) / (x - p.integer(k + 1)))
                if c != want_cancel[k]:
                    bad.append(("cancel", i, k))

        run_in_lockstep(work)
        assert not bad, bad[:3]
