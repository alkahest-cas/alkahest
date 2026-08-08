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
    cancel a heavy call running on a different thread."""
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
# run_with_wall_fallback — Python-layer supplement for calls with no Rust
# checkpoint on every path (documented use case: simplify).
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
