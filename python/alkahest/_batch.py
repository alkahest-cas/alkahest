"""Batch and streaming evaluation — candidate-level fan-out for search loops.

Loops built on Alkahest are embarrassingly parallel *at the candidate level*:
try to integrate a hundred generated integrands, simplify a thousand rewrite
targets, differentiate every entry in a lookup table. Today that fan-out has
to be written by hand at every call site, because :func:`alkahest.integrate`
(and friends) are one-call-one-answer: the first candidate that raises aborts
the whole batch unless the caller remembers ``try/except`` around every
single call.

This module is that plumbing, done once. :func:`batch_map` (and the
``*_many`` convenience wrappers over :func:`alkahest.integrate`,
:func:`alkahest.simplify`, and :func:`alkahest.diff`) call a function once per
item and **never raise** for a single bad element — the exception is caught
and turned into a structured :class:`BatchItem` carrying the failing
exception's stable ``E-*`` diagnostic code (see ``exceptions.py``), so a loop
can tell "this candidate has no elementary antiderivative" (a fine, expected
answer) from "the whole batch process crashed".

Honesty invariant
------------------
:func:`batch_map` always returns exactly one :class:`BatchItem` per input, in
input order — a batch of 100 items yields a list of 100 items, full stop.
Nothing in this module silently drops a failing candidate; a failure is
recorded as ``ok=False`` with its error, never as a missing slot.

Quick start
-----------
>>> import alkahest as ak
>>> pool = ak.ExprPool()
>>> x = pool.symbol("x")
>>> outs = ak.integrate_many([x**2, ak.log(ak.log(x))], x)
>>> outs[0].ok, outs[1].ok
(True, False)
>>> outs[1].error["code"]
'E-INT-001'

Fan out over a thread pool (the Rust kernel releases the GIL for some hot
paths, e.g. the parallel simplifiers and NumPy evaluation, so a thread pool
can overlap those with other Python work; for calls that hold the GIL
throughout, ``parallel=True`` mainly helps when *fn* itself does I/O or
otherwise yields the GIL)::

    outs = ak.batch_map(ak.simplify, candidates, parallel=True)
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable, Iterator

__all__ = [
    "UNEXPECTED_ERROR_CODE",
    "BatchItem",
    "batch_map",
    "batch_map_iter",
    "diff_many",
    "integrate_many",
    "simplify_many",
]

#: Fallback ``error["code"]`` for a failure whose exception carries no ``.code``
#: attribute of its own (i.e. it is not an :class:`~alkahest.exceptions.AlkahestError`
#: or a native PyO3 error exposing the same attribute). See ``exceptions.py``
#: for the registry of codes this module composes with.
UNEXPECTED_ERROR_CODE = "E-BATCH-001"


@dataclass(frozen=True)
class BatchItem:
    """One slot's outcome from :func:`batch_map` or a ``*_many`` helper.

    Exactly one of *value* / *error* is populated: ``ok=True`` implies
    ``error is None`` and ``value`` is whatever *fn* returned (often a
    :class:`~alkahest.DerivedResult`); ``ok=False`` implies ``value is None``
    and ``error`` describes what went wrong.

    Attributes
    ----------
    index : int
        Position of this item in the *original* input sequence. Stable
        under ``parallel=True`` and under :func:`batch_map_iter` streaming in
        completion order — an outcome can always be matched back to its
        input via this field, even when results arrive out of order.
    ok : bool
        ``True`` iff *fn* returned normally for this item.
    value : Any or None
        *fn*'s return value on success; ``None`` on failure.
    error : dict or None
        ``{"code", "message", "remediation", "type"}`` on failure, ``None``
        on success.

        ``code``
            The raised exception's ``.code`` (e.g. ``"E-INT-001"``) when it
            is an :class:`~alkahest.exceptions.AlkahestError`-like exception
            — including the native PyO3 error types, which expose the same
            attribute — otherwise :data:`UNEXPECTED_ERROR_CODE`
            (``"E-BATCH-001"``).
        ``message``
            ``str(exc)``.
        ``remediation``
            The exception's ``.remediation``, or ``None`` when it has none.
        ``type``
            ``type(exc).__name__``.
    elapsed_ms : float or None
        Wall-clock time spent inside *fn* for this item, in milliseconds.
    """

    index: int
    ok: bool
    value: Any | None = None
    error: dict[str, Any] | None = None
    elapsed_ms: float | None = None


def _describe_exception(exc: Exception) -> dict[str, Any]:
    code = getattr(exc, "code", None)
    remediation = getattr(exc, "remediation", None)
    return {
        "code": str(code) if code else UNEXPECTED_ERROR_CODE,
        "message": str(exc),
        "remediation": str(remediation) if remediation is not None else None,
        "type": type(exc).__name__,
    }


def _invoke(fn: Callable[..., Any], item: Any, index: int, kwargs: dict[str, Any]) -> BatchItem:
    """Run ``fn(item, **kwargs)``, turning any :class:`Exception` into a :class:`BatchItem`.

    Deliberately catches ``Exception`` rather than ``BaseException``: a
    ``KeyboardInterrupt`` (or ``SystemExit``) must still propagate and stop
    the batch, since swallowing those would make the process unkillable.
    """
    start = time.perf_counter()
    try:
        value = fn(item, **kwargs)
    except Exception as exc:  # intentional: never abort the batch for one bad element
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        error = _describe_exception(exc)
        return BatchItem(index=index, ok=False, error=error, elapsed_ms=elapsed_ms)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return BatchItem(index=index, ok=True, value=value, elapsed_ms=elapsed_ms)


def batch_map(
    fn: Callable[..., Any],
    items: Iterable[Any],
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    **kwargs: Any,
) -> list[BatchItem]:
    """Call ``fn(item, **kwargs)`` for every item in *items*, never raising.

    Parameters
    ----------
    fn : callable
        Called as ``fn(item, **kwargs)`` for each item. Any :class:`Exception`
        it raises is captured into that item's :class:`BatchItem` rather than
        propagating.
    items : iterable
        The candidates to evaluate. Consumed once, eagerly (so a generator
        works, but is fully materialised before any call happens).
    parallel : bool
        Fan out over a :class:`~concurrent.futures.ThreadPoolExecutor` when
        true. Useful when *fn* releases the GIL for some or all of its work
        (I/O, or a Rust call that calls ``py.allow_threads``); on pure
        Python, GIL-bound work it will not speed anything up, but it also
        will not make anything incorrect — order is preserved either way.
    max_workers : int, optional
        Forwarded to :class:`~concurrent.futures.ThreadPoolExecutor`. Ignored
        when ``parallel=False``.
    **kwargs
        Forwarded to every call to *fn*.

    Returns
    -------
    list of BatchItem
        Exactly ``len(items)`` entries, **in input order** regardless of
        *parallel* — this is the ordering guarantee :func:`batch_map_iter`
        does not make under ``parallel=True``.

    Examples
    --------
    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> outs = ak.batch_map(ak.simplify, [x + 0 * x, ak.log(ak.log(x))])
    >>> [o.ok for o in outs]
    [True, True]
    """
    materialized = list(items)
    if not materialized:
        return []
    if not parallel:
        return [_invoke(fn, item, i, kwargs) for i, item in enumerate(materialized)]

    results: list[BatchItem | None] = [None] * len(materialized)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_invoke, fn, item, i, kwargs): i for i, item in enumerate(materialized)
        }
        for future in futures:
            results[futures[future]] = future.result()
    return results  # type: ignore[return-value]  # every slot was filled above


def batch_map_iter(
    fn: Callable[..., Any],
    items: Iterable[Any],
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    **kwargs: Any,
) -> Iterator[BatchItem]:
    """Streaming counterpart of :func:`batch_map`.

    Order guarantee
    ----------------
    - ``parallel=False``: yields **in input order** — item *i* is fully
      computed and yielded before item *i + 1* starts, so
      ``zip(items, batch_map_iter(fn, items))`` lines up.
    - ``parallel=True``: yields **in completion order**, not input order.
      This is deliberate: the point of streaming under fan-out is that a
      fast failure surfaces immediately instead of waiting behind a slow
      item that happened to be submitted first. Every yielded
      :class:`BatchItem` still carries its original ``index``, so a caller
      that needs input order can sort by it — or just use :func:`batch_map`,
      which always returns in input order.

    Parameters
    ----------
    fn, items, max_workers, **kwargs
        As :func:`batch_map`.
    parallel : bool
        As :func:`batch_map`; also selects the completion-order streaming
        behaviour described above.

    Yields
    ------
    BatchItem

    Examples
    --------
    Sequential streaming preserves input order:

    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> [item.index for item in ak.batch_map_iter(ak.simplify, [x, x + 0 * x])]
    [0, 1]
    """
    materialized = list(items)
    if not materialized:
        return
    if not parallel:
        for i, item in enumerate(materialized):
            yield _invoke(fn, item, i, kwargs)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(_invoke, fn, item, i, kwargs) for i, item in enumerate(materialized)
        }
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()


# ---------------------------------------------------------------------------
# Domain-specific convenience wrappers
# ---------------------------------------------------------------------------
#
# Lazy import of the parent package (mirrors alkahest.research._ak()):
# _batch is imported from __init__.py before integrate/simplify/diff exist in
# its namespace, so these must resolve the module at call time, not at
# import time.


def _ak() -> Any:
    import alkahest

    return alkahest


def integrate_many(
    exprs: Iterable[Any],
    var: Any,
    *bounds: Any,
    parallel: bool = False,
    max_workers: int | None = None,
    **kwargs: Any,
) -> list[BatchItem]:
    """:func:`batch_map` over :func:`alkahest.integrate`, one call per integrand.

    Parameters
    ----------
    exprs : iterable of Expr or DerivedResult
        Integrands, evaluated independently.
    var : Expr
        Integration variable, shared by every call.
    *bounds
        Pass ``a, b`` for a definite integral over every integrand (see
        :func:`alkahest.integrate`); omit for the indefinite integral.
    parallel, max_workers
        As :func:`batch_map`.
    **kwargs
        Forwarded to :func:`alkahest.integrate`.

    Returns
    -------
    list of BatchItem
        ``value`` is the :class:`~alkahest.DerivedResult` antiderivative (or
        definite value) on success; on failure, ``error["code"]`` is the
        integrator's own code (typically ``E-INT-001``) rather than
        :data:`UNEXPECTED_ERROR_CODE`.

    Examples
    --------
    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> outs = ak.integrate_many([x**2, ak.log(ak.log(x)), ak.sin(x)], x)
    >>> [o.ok for o in outs]
    [True, False, True]
    """
    ak = _ak()

    def _one(expr: Any) -> Any:
        return ak.integrate(expr, var, *bounds, **kwargs)

    return batch_map(_one, exprs, parallel=parallel, max_workers=max_workers)


def simplify_many(
    exprs: Iterable[Any],
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    **kwargs: Any,
) -> list[BatchItem]:
    """:func:`batch_map` over :func:`alkahest.simplify`, one call per expression.

    Parameters
    ----------
    exprs : iterable of Expr or DerivedResult
        Expressions to simplify independently.
    parallel, max_workers
        As :func:`batch_map`.
    **kwargs
        Forwarded to :func:`alkahest.simplify` (e.g. ``assumptions=``).

    Returns
    -------
    list of BatchItem

    Examples
    --------
    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> outs = ak.simplify_many([x + 0 * x, x / x])
    >>> [o.value.value for o in outs]
    [x, 1]
    """
    ak = _ak()

    def _one(expr: Any) -> Any:
        return ak.simplify(expr, **kwargs)

    return batch_map(_one, exprs, parallel=parallel, max_workers=max_workers)


def diff_many(
    exprs: Iterable[Any],
    var: Any,
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    **kwargs: Any,
) -> list[BatchItem]:
    """:func:`batch_map` over :func:`alkahest.diff`, one call per expression.

    Parameters
    ----------
    exprs : iterable of Expr or DerivedResult
        Expressions to differentiate independently.
    var : Expr
        Differentiation variable, shared by every call.
    parallel, max_workers
        As :func:`batch_map`.
    **kwargs
        Forwarded to :func:`alkahest.diff`.

    Returns
    -------
    list of BatchItem

    Examples
    --------
    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> outs = ak.diff_many([x**2, ak.sin(x)], x)
    >>> [o.ok for o in outs]
    [True, True]
    """
    ak = _ak()

    def _one(expr: Any) -> Any:
        return ak.diff(expr, var, **kwargs)

    return batch_map(_one, exprs, parallel=parallel, max_workers=max_workers)
