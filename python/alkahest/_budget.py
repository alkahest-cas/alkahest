"""Budget: per-call wall-clock / step limits, cancellation, and a determinism seed.

P1 search plumbing item 4 — see ``docs/mdbook/src/budgets.md``.

A fan-out loop trying thousands of candidate rewrites/integrals cannot afford
one pathological candidate to hang the whole batch, and needs a way to stop a
candidate that's no longer worth the wall time — without waiting for an
OS-level kill. This module is the Python front door for that:

:class:`Budget`
    An immutable ``(wall_ms, max_steps, seed)`` triple.

``alkahest.context(budget=...)``
    Pushes the budget into the Rust-side cooperative checkpoint
    (``alkahest_core::budget``) for the scope of the ``with`` block. Heavy
    engines — currently :func:`alkahest.integrate` and, best-effort,
    :func:`alkahest.simplify` — consult it at a handful of strategic points
    and raise :class:`~alkahest.BudgetExceededError` (or, for ``simplify``,
    stop early without raising — see the note on that function below) when it
    trips.

:func:`run_with_wall_fallback`
    A Python-layer *supplement*, not a replacement, for calls that don't
    (yet) check the Rust cooperative budget on every path — most notably
    :func:`alkahest.simplify`, whose ``DerivedExpr`` return type has no error
    channel to raise through, so it only stops early silently. Runs the call
    on a worker thread (with ``budget`` entered *on that thread*, since budget
    frames are thread-local) and raises
    :class:`~alkahest.BudgetExceededError` when it doesn't finish within
    ``budget.wall_ms``.

    **It does not bound wall time for a callee that never reaches a
    cooperative checkpoint.** The worker thread is not killed — Python has no
    safe way to do that, and abandoning it is worse (see
    :func:`run_with_wall_fallback` for the full argument) — so the call
    returns only once the callee returns. Read its docstring before relying
    on ``wall_ms`` here.

:func:`request_cancel` / :func:`clear_cancel` / :func:`is_cancelled`
    Thin wrappers over the process-wide cancellation flag
    (``alkahest_core::budget``): an orchestrator thread can request that a
    heavy call running on another thread stop *now*.

Thread-local frames vs. the process-wide flag
---------------------------------------------
The two mechanisms have deliberately different scopes, and code that fans work
out over threads has to keep them straight:

* A **budget frame** is *thread-local* (``alkahest_core::budget::STACK``). A
  worker thread does **not** inherit the frame its parent entered, so work
  handed to a :class:`~concurrent.futures.ThreadPoolExecutor` runs unbudgeted
  unless something re-enters the budget on the worker. :func:`capture_budget`
  and :class:`BudgetHandoff` are that "something" — used by
  :func:`alkahest.batch_map` and by :func:`run_with_wall_fallback`.
* The **cancellation flag** is *process-wide* and sticky, so it needs no
  propagation at all — every thread already sees it. The corollary is that
  setting it is never a private act: one candidate's timeout cancels every
  other in-flight call in the process, which is why nothing here sets it
  except :func:`run_with_wall_fallback` (which joins its worker and then
  restores the previous value) and callers who ask for it explicitly.
"""

from __future__ import annotations

import concurrent.futures
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .exceptions import BudgetExceededError

__all__ = [
    "Budget",
    "budget_seed",
    "clear_cancel",
    "is_budget_active",
    "is_cancelled",
    "request_cancel",
    "run_with_wall_fallback",
]

_T = TypeVar("_T")


@dataclass(frozen=True)
class Budget:
    """A per-call resource budget for search-style workloads.

    Every field is optional; ``Budget()`` never trips a cooperative check on
    its own — only :func:`request_cancel` can stop a call entered with a bare
    ``Budget()``. This mirrors the Rust side: entering an otherwise-empty
    budget is how a caller opts a code path into consulting ``seed`` without
    also imposing a wall/step limit.

    Parameters
    ----------
    wall_ms : float, optional
        Wall-clock limit in milliseconds, measured from
        ``context(budget=...)`` entry.
    max_steps : int, optional
        Maximum number of cooperative-checkpoint calls the guarded block may
        make (see ``crate::budget::check`` on the Rust side).
    seed : int, optional
        Determinism seed available to RNG-consuming samplers via
        :func:`budget_seed` — two runs entering the same ``Budget(seed=7)``
        observe the same seed at every call site that consults it.

    Examples
    --------
    >>> import alkahest as ak
    >>> with ak.context(budget=ak.Budget(wall_ms=50, max_steps=10_000, seed=7)):
    ...     try:
    ...         ak.integrate(hard_expr, x)  # doctest: +SKIP
    ...     except ak.BudgetExceededError as e:
    ...         assert e.code.startswith("E-BUDGET-")
    """

    wall_ms: float | None = None
    max_steps: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.wall_ms is not None and (not math.isfinite(self.wall_ms) or self.wall_ms < 0):
            raise ValueError("Budget.wall_ms must be a finite, non-negative number of milliseconds")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("Budget.max_steps must be a non-negative integer")
        if self.seed is not None and self.seed < 0:
            raise ValueError("Budget.seed must be a non-negative integer")


def _native():
    from . import alkahest as _alkahest_native

    return _alkahest_native


def _budget_exceeded(
    message: str, *, remediation: str | None, code: str = "E-BUDGET-001"
) -> BudgetExceededError:
    """Build an ``alkahest.BudgetExceededError`` resolved at call time, not import time.

    ``alkahest/__init__.py`` overlays the pure-Python stub in
    ``alkahest.exceptions`` with the compiled PyO3 class once the package
    finishes initialising (so ``except alkahest.BudgetExceededError`` also
    catches errors raised by the Rust cooperative checkpoint). Importing the
    stub directly at module load time here — before that overlay runs, since
    this module is itself imported from ``alkahest/__init__.py`` — would
    construct a *different* class than the one callers catch as
    ``alkahest.BudgetExceededError``.

    The compiled PyO3 exception class (unlike the pure-Python stub) has no
    ``__init__`` accepting ``code=``/``remediation=`` — Rust sets those as
    plain instance attributes after construction (see
    ``alkahest-py::make_structured_err``) — so this does the same instead of
    calling the constructor with keyword arguments.
    """
    from . import BudgetExceededError as _cls

    exc = _cls(message)
    exc.code = code
    exc.remediation = remediation
    exc.span = None
    return exc


@dataclass(frozen=True)
class BudgetHandoff:
    """A :class:`Budget` snapshot that can cross a thread boundary.

    Budget frames live on a **thread-local** stack on the Rust side, and the
    ``BudgetGuard`` that pops one is ``!Send``, so a budget entered on the
    calling thread is invisible to any worker it fans work out to. A handoff
    is the transferable form: plain numbers, captured on the calling thread by
    :func:`capture_budget` and re-entered on the worker by :meth:`applied`.

    The wall limit is carried as an absolute **deadline**, not as a duration.
    That is what makes a fanned-out batch bounded the same way a sequential
    one is: every worker re-enters *the remaining time until the shared
    deadline*, so N items cannot cost N × ``wall_ms`` between them. Once the
    deadline has passed, later items enter a zero-length budget and trip at
    their first cooperative checkpoint — exactly what the sequential path
    does with the caller's own long-running frame.

    Attributes
    ----------
    deadline : float or None
        A :func:`time.perf_counter` value, or ``None`` when the captured
        budget set no ``wall_ms``.
    max_steps : int or None
        Carried through as-is. Note that the Rust step *counter* lives in the
        frame and is not readable from Python, so each worker gets its own
        counter starting at zero: under ``parallel=True`` ``max_steps`` is a
        per-item limit, not a batch-wide one (the wall limit is batch-wide).
    seed : int or None
        Carried through so :func:`budget_seed` reads the same value on a
        worker as it does on the calling thread.
    """

    deadline: float | None
    max_steps: int | None
    seed: int | None

    def remaining_ms(self) -> float | None:
        """Milliseconds left until :attr:`deadline`, clamped at ``0.0``.

        ``None`` when the captured budget carried no wall limit.
        """
        if self.deadline is None:
            return None
        return max(0.0, (self.deadline - time.perf_counter()) * 1000.0)

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Enter this budget on the *current* thread for the duration of the block.

        Push and pop are paired in a ``finally``, and both happen on the same
        thread — the invariant ``pop_budget`` needs (the guard stack it pops
        from is thread-local).
        """
        native = _native()
        native.push_budget(wall_ms=self.remaining_ms(), max_steps=self.max_steps, seed=self.seed)
        try:
            yield
        finally:
            native.pop_budget()


def capture_budget(budget: Budget | None = None) -> BudgetHandoff | None:
    """Snapshot a budget for hand-off to a worker thread, on the calling thread.

    Parameters
    ----------
    budget : Budget, optional
        The budget to snapshot. Defaults to the one established by the
        innermost active ``alkahest.context(budget=...)``.

    Returns
    -------
    BudgetHandoff or None
        ``None`` when no budget is active — the caller should then run the
        work with no frame at all rather than pushing an empty one, so
        unbudgeted work stays exactly as unbudgeted as it was.

    Notes
    -----
    The deadline is measured from **this call**, not from ``context(...)``
    entry: the Rust frame does not expose its start instant to Python, so a
    handoff captured some time into a budgeted block gives the worker the full
    ``wall_ms`` again rather than what is genuinely left. The overshoot is
    bounded by one ``wall_ms`` for the whole fan-out (not per item), and it is
    why :func:`alkahest.batch_map` captures at batch entry rather than
    per-item.
    """
    if budget is None:
        from ._context import active_budget

        budget = active_budget()
    if budget is None:
        return None
    deadline = None if budget.wall_ms is None else time.perf_counter() + budget.wall_ms / 1000.0
    return BudgetHandoff(deadline=deadline, max_steps=budget.max_steps, seed=budget.seed)


def run_with_wall_fallback(
    fn: Callable[..., _T],
    /,
    *args: Any,
    budget: Budget,
    **kwargs: Any,
) -> _T:
    """Run ``fn(*args, **kwargs)`` under ``budget``, raising ``E-BUDGET-001``
    when it overruns ``budget.wall_ms`` — but **without** a hard deadline.

    Read this before relying on it
    -------------------------------
    This turns "the callee quietly gave up early" into a raised, coded error,
    and it re-enters ``budget`` on the worker thread so cooperative
    checkpoints actually see it. What it does **not** do is return control at
    ``wall_ms``: the worker is joined before the exception propagates, so for
    a callee that never reaches a cooperative checkpoint —
    ``time.sleep(3)``, a single long FLINT call, third-party code —
    ``Budget(wall_ms=50)`` raises the right error *after the callee finishes*.
    Three seconds, in that example. The error message reports how long control
    was actually withheld, so this is visible in a log rather than inferred.

    Why not just abandon the worker and return at the deadline? Because
    Python cannot kill a thread, so "return early" means leaking a live
    thread that still holds the GIL in bursts, still allocates into the pool,
    and cannot be stopped except through the **process-wide** cancellation
    flag — which would also abort every unrelated in-flight call in the
    process, and which nobody could then safely clear (clearing it before the
    orphan observes it is a no-op; leaving it set poisons every subsequent
    cooperative call). In a multi-day loop that trades a bounded stall for
    unbounded orphan-thread accumulation plus collateral cancellation. Joining
    is the honest lesser evil, so it is what this does.

    What *does* bound wall time
    ---------------------------
    - ``context(budget=...)`` for engines that check the cooperative budget —
      :func:`alkahest.integrate` and :func:`alkahest.limit` today. That is the
      real mechanism; this function is a reporting shim over it.
    - An OS-level bound — a subprocess with a timeout, or a process-level
      watchdog — for anything else. Nothing inside one Python process can
      preempt a thread.

    So: reach for this to get a *raise* out of a cooperatively-budgeted call
    that would otherwise return a silently-truncated answer (the documented
    case is :func:`alkahest.simplify`, whose ``DerivedExpr`` return type has
    no error channel). Do not reach for it to contain an unknown callee.

    Parameters
    ----------
    fn : callable
        The function to call.
    *args, **kwargs
        Forwarded to ``fn``.
    budget : Budget
        Entered on the worker thread for the duration of the call, so
        ``max_steps`` and ``seed`` reach cooperative call sites too. If
        ``budget.wall_ms`` is ``None``, this is equivalent to
        ``fn(*args, **kwargs)`` — no thread is spawned, and the caller's own
        ambient budget (if any) applies unchanged.

    Raises
    ------
    BudgetExceededError
        (``E-BUDGET-001``) if ``fn`` does not return within ``budget.wall_ms``
        milliseconds. Raised once ``fn`` has actually finished — see above.

    Examples
    --------
    >>> import alkahest as ak
    >>> b = ak.Budget(wall_ms=5)
    >>> ak.run_with_wall_fallback(lambda: ak.simplify(x**2), budget=b)  # doctest: +SKIP
    """
    if budget.wall_ms is None:
        return fn(*args, **kwargs)

    # Budget frames are thread-local: without this the worker ran the callee
    # with *no* budget active, so the only thing that could stop it was the
    # process-wide cancel flag below. Entering it on the worker is what makes
    # a cooperative callee stop on its own, promptly, and without touching
    # global state.
    handoff = capture_budget(budget)

    def _run_on_worker() -> _T:
        if handoff is None:  # pragma: no cover - budget.wall_ms is not None here
            return fn(*args, **kwargs)
        with handoff.applied():
            return fn(*args, **kwargs)

    # The cancellation flag is process-wide and sticky, so a timeout here must
    # not outlive this call: without the restore below, one expired candidate
    # in a long search loop leaves `CANCELLED` set and *every* subsequent
    # cooperative call in the process fails with E-BUDGET-003 forever.  Only
    # restore a flag this call raised — an orchestrator that had already
    # requested cancellation keeps its request.
    cancelled_before = is_cancelled()
    requested_here = False
    timed_out: concurrent.futures.TimeoutError | None = None
    started = time.perf_counter()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_run_on_worker)
        try:
            return future.result(timeout=budget.wall_ms / 1000.0)
        except concurrent.futures.TimeoutError as exc:
            # Belt and braces alongside the worker's own budget frame: it also
            # reaches checkpoints the frame cannot (Rayon workers the callee
            # fanned out to, or a callee that shadowed our frame with a nested
            # `context(budget=...)` of its own).
            request_cancel()
            requested_here = True
            timed_out = exc
    finally:
        # `shutdown(wait=True)` — the join this function is honest about, and
        # the reason the flag can be restored safely: by the time we get here
        # the call we cancelled has already observed the flag and stopped.
        pool.shutdown(wait=True)
        if requested_here and not cancelled_before:
            clear_cancel()

    blocked_ms = (time.perf_counter() - started) * 1000.0
    raise _budget_exceeded(
        f"[E-BUDGET-001] budget exceeded: wall-clock limit {budget.wall_ms} ms elapsed; "
        f"run_with_wall_fallback returned control after {blocked_ms:.0f} ms "
        f"(it joins its worker rather than abandoning the thread)",
        remediation=(
            "raise Budget(wall_ms=...), or accept a heuristic/numeric result for "
            "this candidate instead of an exact one; if the overrun above is large, the "
            "callee does not reach a cooperative checkpoint and only an OS-level timeout "
            "can bound it -- see docs/mdbook/src/budgets.md"
        ),
    ) from timed_out


def request_cancel() -> None:
    """Request cancellation of the current cooperative operation(s), process-wide.

    See ``alkahest_core::budget::request_cancel`` — checked by every
    cooperative checkpoint on every thread until :func:`clear_cancel` is
    called.
    """
    _native().request_cancel()


def clear_cancel() -> None:
    """Clear a previously requested cancellation.

    Call this before starting the next candidate in a fan-out loop.
    """
    _native().clear_cancel()


def is_cancelled() -> bool:
    """Return ``True`` if :func:`request_cancel` was called and not yet cleared."""
    return bool(_native().is_cancelled())


def is_budget_active() -> bool:
    """Return ``True`` if a :class:`Budget` is active on this thread."""
    return bool(_native().is_budget_active())


def budget_seed() -> int | None:
    """Return the seed of the innermost active :class:`Budget` on this thread.

    ``None`` if no budget is active or the active budget did not set one.
    """
    return _native().budget_seed()
