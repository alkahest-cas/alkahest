"""Budgets, cooperative cancellation, and determinism (P1 search plumbing item 4).

See ``docs/mdbook/src/budgets.md``. These tests exercise the Python-visible
surface: ``Budget``, ``context(budget=...)``, ``BudgetExceededError``, the
process-wide cancellation flag, the determinism seed, and the
``run_with_wall_fallback`` supplement for calls with no Rust checkpoint.
"""

from __future__ import annotations

import threading
import time

import alkahest as ak
import pytest

#: Far above the few seconds the heavy cases below take when they work; a stuck
#: call is a bug, not a slow machine.
HEAVY_TIMEOUT = 120

#: Ceiling on the **CPU** time a bounded cooperative callee may burn, in
#: milliseconds.  See
#: ``test_run_with_wall_fallback_bounds_a_cooperative_callee`` for why the bound
#: is measured in CPU time rather than wall-clock time.  Measured cost of that
#: case on a 12-core box: 2.7-5.3 s idle, 8.8 s under 2x oversubscription, so
#: this leaves ~7x headroom against a saturated machine.
COOPERATIVE_CALLEE_CPU_BOUND_MS = 60_000


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x", "real")


@pytest.fixture(autouse=True)
def _clear_cancel_before_and_after():
    """Cancellation is a process-wide flag (see the module docs) — never let a
    failing assertion in one test leave it set for the next test to inherit."""
    ak.clear_cancel()
    yield
    ak.clear_cancel()


# ---------------------------------------------------------------------------
# Budget dataclass
# ---------------------------------------------------------------------------


def test_budget_defaults_are_all_none():
    b = ak.Budget()
    assert b.wall_ms is None
    assert b.max_steps is None
    assert b.seed is None


def test_budget_is_immutable():
    b = ak.Budget(wall_ms=1)
    with pytest.raises(AttributeError):
        b.wall_ms = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wall_ms": -1},
        {"wall_ms": float("nan")},
        {"wall_ms": float("inf")},
        {"max_steps": -1},
        {"seed": -1},
    ],
)
def test_budget_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ak.Budget(**kwargs)


# ---------------------------------------------------------------------------
# context(budget=...) nesting
# ---------------------------------------------------------------------------


def test_no_budget_by_default():
    assert not ak.is_budget_active()
    assert ak.budget_seed() is None


def test_context_budget_activates_and_deactivates(pool):
    assert not ak.is_budget_active()
    with ak.context(pool=pool, budget=ak.Budget(max_steps=100)):
        assert ak.is_budget_active()
    assert not ak.is_budget_active()


def test_context_budget_round_trips_active_budget(pool):
    b = ak.Budget(wall_ms=50, max_steps=100, seed=7)
    with ak.context(pool=pool, budget=b):
        assert ak.active_budget() == b
    assert ak.active_budget() is None


def test_seed_round_trips_through_context(pool):
    assert ak.budget_seed() is None
    with ak.context(pool=pool, budget=ak.Budget(seed=42)):
        assert ak.budget_seed() == 42
    assert ak.budget_seed() is None


def test_nested_context_without_budget_kw_keeps_outer_active(pool):
    """A nested context(...) that omits budget= pushes nothing onto the Rust
    stack, so the outer budget (and its seed) stays visible — unlike pool/
    domain, which the inner frame *does* hide (context() doesn't merge)."""
    with ak.context(pool=pool, budget=ak.Budget(seed=1, max_steps=1000)), ak.context(pool=pool):
        assert ak.budget_seed() == 1
        assert ak.is_budget_active()


def test_nested_context_with_budget_kw_shadows_not_merges(pool):
    """A nested budget= replaces the outer one for the block — it does not
    inherit the outer seed."""
    with ak.context(pool=pool, budget=ak.Budget(seed=1, max_steps=1000)):
        with ak.context(pool=pool, budget=ak.Budget(max_steps=2)):
            assert ak.budget_seed() is None
        # Back to the outer frame on exit from the inner `with`.
        assert ak.budget_seed() == 1


def test_context_budget_pops_even_on_exception(pool):
    def _raise_inside_budget():
        assert ak.is_budget_active()
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError), ak.context(pool=pool, budget=ak.Budget(max_steps=100)):
        _raise_inside_budget()
    assert not ak.is_budget_active()


# ---------------------------------------------------------------------------
# Step budget trips integrate()
# ---------------------------------------------------------------------------


def test_step_budget_trips_integrate(pool, x):
    """integrate() checks the cooperative budget at its top-level entry, so a
    max_steps=0 budget must trip on the very first call."""
    with ak.context(pool=pool, budget=ak.Budget(max_steps=0)):
        with pytest.raises(ak.BudgetExceededError) as excinfo:
            ak.integrate(x**2, x)
        assert excinfo.value.code == "E-BUDGET-002"


def test_generous_step_budget_does_not_trip_integrate(pool, x):
    with ak.context(pool=pool, budget=ak.Budget(max_steps=1_000_000)):
        result = ak.integrate(x**2, x)
        assert result.value is not None


def test_budget_exceeded_error_is_also_alkahest_error(pool, x):
    with ak.context(pool=pool, budget=ak.Budget(max_steps=0)), pytest.raises(ak.AlkahestError):
        ak.integrate(x**2, x)


def test_budget_exceeded_error_is_also_value_error(pool, x):
    """Structured alkahest errors all derive from ValueError, same as every
    other exception class in the hierarchy."""
    with ak.context(pool=pool, budget=ak.Budget(max_steps=0)), pytest.raises(ValueError):
        ak.integrate(x**2, x)


def test_no_budget_active_integrate_still_works(pool, x):
    result = ak.integrate(x**2, x)
    assert result.value is not None


# ---------------------------------------------------------------------------
# Wall-clock budget
# ---------------------------------------------------------------------------


def test_wall_budget_trips_after_elapsed(pool, x):
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=1)):
        time.sleep(0.02)
        with pytest.raises(ak.BudgetExceededError) as excinfo:
            ak.integrate(x**2, x)
        assert excinfo.value.code == "E-BUDGET-001"


# ---------------------------------------------------------------------------
# `wall_ms` has to bound the call, not merely be consulted by it
#
# A budget that trips *eventually* is not a budget. These integrands used to
# overshoot `wall_ms=300` by 7-12x, growing with problem size until the last one
# ran for over 90 seconds without ever coming back — the checkpoints existed but
# the seconds were being spent between them, in the Weierstrass half-angle route
# and the rational-function normalisation it feeds.
#
# Deliberately no assertion on elapsed time: the honest claim is "the call comes
# back, and it comes back saying the budget stopped it". A regression puts the
# uninterruptible stretch back and the test hangs until `pytest.mark.timeout`
# kills it, which is the signal we want — a wall-clock assertion would instead
# go red on a loaded CI box for no reason.
#
# The ladder was rebuilt for 3.8. The original rungs — `(12,9) … (40,17)` and
# the pure `1/(sin⁹x + sin x + 1)` — are no longer hard: the LRT `RootSum`
# suppression on the two verify-gated routes and the FLINT-backed `poly_gcd`
# took them from 3.7 s / 14.2 s / ~110 s down to 12 ms / 200 ms / 15 ms, so a
# 300 ms budget has nothing to trip on and the test asserted a decline that no
# longer happens. These tests are about the *budget*, not about those specific
# integrands, so the rungs were moved up to inputs that still cost seconds.
# ---------------------------------------------------------------------------


def _hard_trig_integrand(x: ak.Expr, n: int, d: int) -> ak.Expr:
    """`∫ cos x·sinⁿx/(sin^d x + sin x + 1) dx`.

    Declined by every rule, so it reaches the Weierstrass half-angle
    substitution, which doubles the degree and hands a degree-2n rational
    function to Rothstein–Trager. Hard, and hard in a way that scales.
    """
    s = ak.sin(x)
    return ak.cos(x) * s**n / (s**d + s + 1)


@pytest.mark.timeout(HEAVY_TIMEOUT)
@pytest.mark.parametrize(("n", "d"), [(40, 29), (52, 29), (40, 31), (48, 31), (60, 31)])
def test_wall_budget_stops_a_hard_trig_integral(pool, x, n, d):
    """Every rung of the ladder must trip. Unbudgeted these run 1.6-5.4 s, so a
    300 ms budget that does not trip means an uninterruptible stretch is back.

    Cost is *not* monotone in `n` for fixed `d` — `(72, 31)` declines in 13 ms
    while `(60, 31)` costs 5.4 s — so the rungs are measured choices, not a
    range."""
    with (
        ak.context(pool=pool, budget=ak.Budget(wall_ms=300)),
        pytest.raises(ak.BudgetExceededError) as excinfo,
    ):
        ak.integrate(_hard_trig_integrand(x, n, d), x)
    assert excinfo.value.code == "E-BUDGET-001"


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_wall_budget_stops_the_pure_weierstrass_route(pool, x):
    """No `cos` factor, so u-substitution cannot apply and the Weierstrass route
    is the only thing running. Unbudgeted this integral takes about 2.2 s (the
    `sin⁹` original is now 15 ms — see the rebuilt-ladder note above)."""
    s = ak.sin(x)
    with (
        ak.context(pool=pool, budget=ak.Budget(wall_ms=300)),
        pytest.raises(ak.BudgetExceededError) as excinfo,
    ):
        ak.integrate(1 / (s**25 + s + 1), x)
    assert excinfo.value.code == "E-BUDGET-001"


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_wall_budget_stops_between_summands(pool, x):
    """The sum rule recurses through `integrate_raw`, which had no checkpoint at
    all: a sum of eight hard rational terms ran to completion under a 50 ms
    budget because nothing was consulted between terms."""
    total = None
    for k in range(1, 9):
        term = x ** (k + 12) / (x**11 + k)
        total = term if total is None else total + term
    with (
        ak.context(pool=pool, budget=ak.Budget(wall_ms=50)),
        pytest.raises(ak.BudgetExceededError) as excinfo,
    ):
        ak.integrate(total, x)
    assert excinfo.value.code == "E-BUDGET-001"


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_a_budget_that_is_not_hit_still_returns_the_integral(pool, x):
    """The control. Checkpoints that refuse work they could have finished would
    pass every test above and be a regression, so pin the other direction: an
    integral inside its budget still comes back with a value."""
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=30_000, max_steps=10_000_000)):
        assert ak.integrate(x**2 + 1, x).value is not None
        assert ak.integrate(1 / (x**2 + 1), x).value is not None
        s = ak.sin(x)
        assert ak.integrate(ak.cos(x) * s**3, x).value is not None


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancel_flag_trips_check_and_clears(pool, x):
    assert not ak.is_cancelled()
    ak.request_cancel()
    assert ak.is_cancelled()
    with pytest.raises(ak.BudgetExceededError) as excinfo:
        ak.integrate(x**2, x)
    assert excinfo.value.code == "E-BUDGET-003"
    ak.clear_cancel()
    assert not ak.is_cancelled()
    result = ak.integrate(x**2, x)
    assert result.value is not None


def test_cancel_trips_even_without_a_budget_context(pool, x):
    """Cancellation is process-wide, not scoped to a Budget frame — it trips
    the cooperative checkpoint even with no context(budget=...) active."""
    assert not ak.is_budget_active()
    ak.request_cancel()
    with pytest.raises(ak.BudgetExceededError):
        ak.integrate(x**2, x)


def test_cancel_from_another_thread_trips_check_on_this_thread(pool, x):
    """The whole point of a process-wide flag: an orchestrator thread can
    cancel a heavy call running on a different thread.

    The flag is set *before* the call here — the easy half. See
    ``test_request_cancel_reaches_a_running_*`` below for the half that
    actually matters."""
    barrier = threading.Event()

    def watchdog():
        barrier.wait(timeout=2.0)
        ak.request_cancel()

    t = threading.Thread(target=watchdog)
    t.start()
    barrier.set()
    t.join(timeout=2.0)
    assert ak.is_cancelled()
    with pytest.raises(ak.BudgetExceededError):
        ak.integrate(x**2, x)


# ---------------------------------------------------------------------------
# Cancelling a call that is already running
#
# `integrate` and `limit` used to hold the GIL for their whole run, so a
# watchdog thread could not execute a single bytecode until the call it wanted
# to cancel had already finished: only a flag set *before* the call was ever
# observed. For a fan-out search loop — decide a candidate has had enough time,
# stop it, move on — that is the entire use case, so it is worth its own tests.
#
# Both bindings now release the GIL around the core call, and both workloads
# below are chosen to run for seconds so the flag lands somewhere in the middle
# rather than before the engine has started. Nothing here asserts a wall-clock
# bound: the discriminator is *which exception* comes back. With the GIL held
# the call would run to its own verdict (`LimitError` / `IntegrationError`) and
# the cancellation would arrive too late to matter — a loud failure, not a
# timing flake.
# ---------------------------------------------------------------------------


def _slow_unanswerable_limit(x: ak.Expr) -> ak.Expr:
    """A limit the engine cannot answer, and takes seconds to give up on.

    Triply-nested radicals sit on scales the leading-order route declines, so
    the call falls through to the expansion path and runs until the internal
    work ceiling stops it — a few seconds of uninterrupted Rust.
    """
    return ak.sqrt(ak.sqrt(ak.sqrt(x**2 + x) + x) + x)


def _slow_unanswerable_integrand(x: ak.Expr) -> ak.Expr:
    """An integrand the rules and the rational path both decline.

    That hands it to the derivative-divides u-substitution search, whose
    candidates each run a full recursive `integrate` over a high-degree
    rational function in ``u = sin x`` — seconds of work, with a cooperative
    checkpoint between candidates.
    """
    s = ak.sin(x)
    return ak.cos(x) * s**60 / (s**31 + s + 1)


def _cancelled_mid_flight(call):
    """Run *call* with a watchdog that cancels only once it is already running.

    The watchdog waits for the main thread to say it is about to enter the
    engine and *then* sleeps before setting the flag, so a trip proves the
    engine observed a cancellation raised during its own run. Setting the flag
    beforehand would make this pass with the GIL held, which is the bug.
    """
    entering = threading.Event()
    cancelled = threading.Event()

    def watchdog():
        entering.wait(timeout=HEAVY_TIMEOUT)
        time.sleep(0.05)
        ak.request_cancel()
        cancelled.set()

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    try:
        entering.set()
        return call()
    finally:
        # Always, on every path: the flag is process-wide, and leaving it set
        # would trip every later test in this process.
        cancelled.wait(timeout=HEAVY_TIMEOUT)
        ak.clear_cancel()
        t.join(timeout=HEAVY_TIMEOUT)


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_request_cancel_reaches_a_running_limit(pool, x):
    with pytest.raises(ak.BudgetExceededError) as excinfo:
        _cancelled_mid_flight(lambda: ak.limit(_slow_unanswerable_limit(x), x, pool.pos_infinity()))
    assert excinfo.value.code == "E-BUDGET-003"


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_request_cancel_reaches_a_running_integrate(pool, x):
    with pytest.raises(ak.BudgetExceededError) as excinfo:
        _cancelled_mid_flight(lambda: ak.integrate(_slow_unanswerable_integrand(x), x))
    assert excinfo.value.code == "E-BUDGET-003"


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_engines_still_work_after_a_mid_flight_cancellation(pool, x):
    """A cancelled call must leave the engines usable, not wedged."""
    with pytest.raises(ak.BudgetExceededError):
        _cancelled_mid_flight(lambda: ak.limit(_slow_unanswerable_limit(x), x, pool.pos_infinity()))
    assert not ak.is_cancelled()
    assert ak.limit(ak.sqrt(x**2 + x) - x, x, pool.pos_infinity()) == pool.rational(1, 2)
    assert ak.integrate(x**2, x).value is not None


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_two_threads_can_run_the_engines_on_one_pool(pool):
    """Releasing the GIL is what lets a watchdog run — and also what lets two
    workers genuinely overlap on the same `ExprPool` for the first time.

    `ExprPool` interns through a lock-free index and is `Send + Sync`, so this
    is sound; the test is here because "sound in principle" is what everyone
    says right before a data race. Same answers from every thread, no crash.
    """
    oo = pool.pos_infinity()
    half = pool.rational(1, 2)
    errors: list[BaseException] = []
    answers: list[bool] = []
    lock = threading.Lock()

    def work(k: int) -> None:
        try:
            xk = pool.symbol("x", "real")
            for _ in range(20):
                got = ak.limit(ak.sqrt(xk**2 + xk) - xk, xk, oo)
                anti = ak.integrate(xk**2 + k, xk)
                with lock:
                    answers.append(got == half and anti.value is not None)
        except BaseException as exc:  # reported on the main thread, not swallowed
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in (1, 2, 3, 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=HEAVY_TIMEOUT)
    assert not errors, f"engine raised on a worker thread: {errors[0]!r}"
    assert len(answers) == 80
    assert all(answers)


# ---------------------------------------------------------------------------
# run_with_wall_fallback — Python-layer supplement for calls with no Rust
# checkpoint on every path (documented use case: simplify).
#
# It raises `E-BUDGET-001` when `wall_ms` is overrun, and it enters the budget
# on the worker thread so cooperative call sites there see it. It does *not*
# return control at the deadline: it joins the worker first. Both halves are
# tested below — the second one deliberately, because an unenforced deadline
# that nobody documented is worse than one that is.
# ---------------------------------------------------------------------------


def test_run_with_wall_fallback_passthrough_when_no_wall_ms():
    assert ak.run_with_wall_fallback(lambda: 1 + 1, budget=ak.Budget()) == 2


def test_run_with_wall_fallback_raises_on_timeout():
    def slow():
        time.sleep(0.2)
        return 42

    with pytest.raises(ak.BudgetExceededError) as excinfo:
        ak.run_with_wall_fallback(slow, budget=ak.Budget(wall_ms=10))
    assert excinfo.value.code == "E-BUDGET-001"


def test_run_with_wall_fallback_returns_value_when_fast_enough():
    assert ak.run_with_wall_fallback(lambda: 1 + 1, budget=ak.Budget(wall_ms=5_000)) == 2


def test_run_with_wall_fallback_forwards_args_and_kwargs():
    def add(a, b, *, c=0):
        return a + b + c

    result = ak.run_with_wall_fallback(add, 1, 2, budget=ak.Budget(wall_ms=5_000), c=3)
    assert result == 6


def test_run_with_wall_fallback_propagates_underlying_exception():
    def boom():
        raise KeyError("nope")

    with pytest.raises(KeyError):
        ak.run_with_wall_fallback(boom, budget=ak.Budget(wall_ms=5_000))


def test_run_with_wall_fallback_timeout_does_not_poison_the_process(pool, x):
    """A timeout must not leave the process-wide cancel flag set.

    ``request_cancel`` is global and sticky, so without a restore one expired
    candidate in a long search loop would make *every* later cooperative call
    in the process fail with E-BUDGET-003 for the rest of its lifetime.
    """
    ak.clear_cancel()
    assert not ak.is_cancelled()

    with pytest.raises(ak.BudgetExceededError):
        ak.run_with_wall_fallback(lambda: time.sleep(0.2), budget=ak.Budget(wall_ms=10))

    assert not ak.is_cancelled()
    # The next unrelated call still works.
    assert ak.integrate(x**2, x) is not None


def test_run_with_wall_fallback_preserves_an_existing_cancel_request(pool, x):
    """An orchestrator's own outstanding cancellation survives a timeout."""
    ak.request_cancel()
    try:
        with pytest.raises(ak.BudgetExceededError):
            ak.run_with_wall_fallback(lambda: time.sleep(0.2), budget=ak.Budget(wall_ms=10))
        assert ak.is_cancelled()
    finally:
        ak.clear_cancel()


def test_run_with_wall_fallback_enters_the_budget_on_the_worker_thread():
    """Budget frames are thread-local, so the worker used to run the callee
    with *no* budget active at all — the docstring's "cooperative call sites
    still see it" was false, and the only thing that could stop a runaway call
    was the process-wide cancel flag (which stops everything else too)."""
    seen = ak.run_with_wall_fallback(
        lambda: (ak.is_budget_active(), ak.budget_seed()),
        budget=ak.Budget(wall_ms=30_000, seed=11),
    )
    assert seen == (True, 11)


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_run_with_wall_fallback_bounds_a_cooperative_callee(pool, x):
    """Because the worker now enters the budget, a callee that honours the
    cooperative checkpoint stops on its own budget rather than only on the
    global cancel flag. Unbudgeted this integrand does not come back at all
    (see ``test_wall_budget_stops_a_hard_trig_integral``); the loose bound is
    the property under test.

    The bound is on **CPU** time, not wall-clock time. It used to read
    ``elapsed_ms < 20 * 300`` against a call that measurably costs 2.7-5.3 s
    when it works -- a 13% margin -- so the test went red whenever the machine
    was busy, which is a fact about the box and not about the property. Wall
    time here is ``wall_ms`` (a real-time timer, so it does not stretch) plus
    the join of a worker that is still running, and contention inflates that
    second term without the callee doing any more work. Measured on one box,
    idle vs. 24 spinners on 12 cores: wall 5.3 s -> 24.9 s (4x over the old
    bound, a guaranteed failure), CPU 5.3 s -> 8.8 s. CPU time is not perfectly
    load-free -- contention costs some real cycles -- but it tracks the work
    done rather than the waiting, which is the thing under test, and it leaves
    real headroom under ``COOPERATIVE_CALLEE_CPU_BOUND_MS``.

    The property is still enforced from both ends: a callee that stops seeing
    the budget burns CPU without limit and trips the assertion, and one that
    stops coming back at all trips the ``timeout`` marker above.
    """
    s = ak.sin(x)
    hard = ak.cos(x) * s**60 / (s**31 + s + 1)
    cpu_started = time.process_time()
    with pytest.raises(ak.BudgetExceededError) as excinfo:
        ak.run_with_wall_fallback(ak.integrate, hard, x, budget=ak.Budget(wall_ms=300))
    cpu_ms = (time.process_time() - cpu_started) * 1000.0
    assert excinfo.value.code == "E-BUDGET-001"
    # The call ended on the wall-clock fallback's own join, not on some other
    # budget check that happens to raise the same code.
    assert "returned control after" in str(excinfo.value)
    assert cpu_ms < COOPERATIVE_CALLEE_CPU_BOUND_MS, (
        f"the callee burned {cpu_ms:.0f} ms of CPU against a 300 ms budget: the "
        "wall-clock fallback is no longer bounding a cooperative callee"
    )


@pytest.mark.timeout(HEAVY_TIMEOUT)
def test_run_with_wall_fallback_does_not_bound_an_uncooperative_callee():
    """Pins the documented limitation, so nobody "discovers" it in production.

    ``run_with_wall_fallback`` joins its worker before propagating: Python
    cannot kill a thread, and abandoning one trades a bounded stall for
    unbounded orphan accumulation plus collateral cancellation (the flag is
    process-wide). So for a callee that never reaches a cooperative
    checkpoint, ``wall_ms`` selects the *error*, not the *deadline* — and the
    message has to say how long control was actually withheld, because that is
    the only thing distinguishing this from a real deadline in a log.

    The assertion is deliberately one-sided and far below the callee's own
    duration: it fails if the function ever starts returning early (which
    would mean an orphan thread), not on a slow machine.
    """
    started = time.perf_counter()
    with pytest.raises(ak.BudgetExceededError) as excinfo:
        ak.run_with_wall_fallback(time.sleep, 1.0, budget=ak.Budget(wall_ms=50))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert excinfo.value.code == "E-BUDGET-001"
    assert elapsed_ms > 500, "documented behaviour is to wait for the callee, not to abandon it"
    assert "returned control after" in str(excinfo.value)
    assert not ak.is_cancelled()


# ---------------------------------------------------------------------------
# Error codes present and well-formed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["E-BUDGET-001", "E-BUDGET-002", "E-BUDGET-003"])
def test_budget_error_codes_documented(code):
    """Lock in the three stable codes from the module docstring / mdbook page."""
    assert code.startswith("E-BUDGET-")


def test_budget_exceeded_error_has_remediation(pool, x):
    with ak.context(pool=pool, budget=ak.Budget(max_steps=0)):
        with pytest.raises(ak.BudgetExceededError) as excinfo:
            ak.integrate(x**2, x)
        assert excinfo.value.remediation
        assert isinstance(excinfo.value.remediation, str)
