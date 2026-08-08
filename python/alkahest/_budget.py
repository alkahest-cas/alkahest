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
    on a worker thread and raises :class:`~alkahest.BudgetExceededError` if it
    doesn't finish within ``budget.wall_ms``. The worker thread is **not**
    killed — Python has no safe way to do that — so on a timeout the call may
    keep running in the background until it hits a cooperative checkpoint or
    finishes. Prefer relying on the Rust cooperative check (via
    ``context(budget=...)`` alone) wherever it's already wired; reach for this
    only when you need a hard deadline on a path it doesn't cover.

:func:`request_cancel` / :func:`clear_cancel` / :func:`is_cancelled`
    Thin wrappers over the process-wide cancellation flag
    (``alkahest_core::budget``): an orchestrator thread can request that a
    heavy call running on another thread stop *now*.
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
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


def run_with_wall_fallback(
    fn: Callable[..., _T],
    /,
    *args: Any,
    budget: Budget,
    **kwargs: Any,
) -> _T:
    """Run ``fn(*args, **kwargs)``, enforcing ``budget.wall_ms`` even if ``fn``
    doesn't check the Rust cooperative budget on every path.

    This is a *supplement* to, not a replacement for, entering the budget via
    ``context(budget=...)`` — call this from inside such a block (or pass a
    budget that also carries ``max_steps``/``seed``) so cooperative call sites
    still see it. See the module docstring for why the worker thread is not
    forcibly stopped on timeout.

    Parameters
    ----------
    fn : callable
        The function to call.
    *args, **kwargs
        Forwarded to ``fn``.
    budget : Budget
        If ``budget.wall_ms`` is ``None``, this is equivalent to
        ``fn(*args, **kwargs)`` — no thread is spawned.

    Raises
    ------
    BudgetExceededError
        (``E-BUDGET-001``) if ``fn`` does not return within ``budget.wall_ms``
        milliseconds.

    Examples
    --------
    >>> import alkahest as ak
    >>> b = ak.Budget(wall_ms=5)
    >>> ak.run_with_wall_fallback(lambda: ak.simplify(x**2), budget=b)  # doctest: +SKIP
    """
    if budget.wall_ms is None:
        return fn(*args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=budget.wall_ms / 1000.0)
        except concurrent.futures.TimeoutError as exc:
            # Best-effort: ask any cooperative Rust checkpoint the call has
            # reached (or will reach) to stop, since we can't stop the
            # Python thread itself.
            request_cancel()
            raise _budget_exceeded(
                f"[E-BUDGET-001] budget exceeded: wall-clock limit {budget.wall_ms} ms elapsed",
                remediation=(
                    "raise Budget(wall_ms=...), or accept a heuristic/numeric result for this "
                    "candidate instead of an exact one"
                ),
            ) from exc


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
