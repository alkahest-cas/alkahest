"""Thread-local context manager for Alkahest.

RW-7 — Provides a ``with alkahest.context(...)`` block that sets thread-local
defaults for domain, pool, and simplification so callers don't have to repeat
common arguments on every call.

The kernel remains stateless; this module is purely a Python-layer convenience.

Example
-------
>>> import alkahest
>>> p = alkahest.ExprPool()
>>> with alkahest.context(pool=p, domain=alkahest.Domain.Real, simplify=True):
...     x = alkahest.symbol("x")          # domain and pool inferred
...     expr = x ** 2
...     d = alkahest.diff(expr, x)        # simplify applied automatically
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from threading import local
from typing import Any

# ---------------------------------------------------------------------------
# Thread-local state
# ---------------------------------------------------------------------------

_state = local()


def _get() -> dict[str, Any]:
    """Return the current context dict (empty if none is active)."""
    if not hasattr(_state, "stack") or not _state.stack:
        return {}
    return _state.stack[-1]


def get_context_value(key: str, default: Any = None) -> Any:
    """Return the value of *key* from the innermost active context, or *default*."""
    return _get().get(key, default)


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------


@contextmanager
def context(
    *,
    pool: Any = None,
    domain: Any = None,
    simplify: bool = False,
    precision: int | None = None,
    **extra: Any,
) -> Generator[None, None, None]:
    """Thread-local context for Alkahest calls.

    Parameters
    ----------
    pool : ExprPool, optional
        Default expression pool used by ``alkahest.symbol`` and other
        pool-aware helpers when called without an explicit ``pool`` argument.
    domain : Domain, optional
        Default domain for ``alkahest.symbol(name)`` calls that omit ``domain``.
    simplify : bool
        When ``True``, Alkahest helper functions that accept a ``simplify``
        keyword will default to simplifying their result.
    precision : int, optional
        Default MPFR precision in bits for ball-arithmetic operations.
    **extra
        Additional key-value pairs stored in the context and accessible via
        :func:`get_context_value`.

    Notes
    -----
    Contexts are thread-local and nest — inner contexts shadow outer ones
    for any keys they define.  The kernel remains fully stateless; this
    module is an ergonomic Python wrapper only.

    Examples
    --------
    >>> import alkahest
    >>> p = alkahest.ExprPool()
    >>> with alkahest.context(pool=p, domain=alkahest.Domain.Real):
    ...     x = alkahest.symbol("x")
    ...     y = alkahest.symbol("y")
    ...     expr = x ** 2 + y ** 2

    Contexts nest::

        with alkahest.context(pool=p):
            with alkahest.context(simplify=True):
                # Both pool and simplify are active here.
                ...

    """
    if not hasattr(_state, "stack"):
        _state.stack = []

    ctx: dict[str, Any] = {}
    if pool is not None:
        ctx["pool"] = pool
    if domain is not None:
        ctx["domain"] = domain
    ctx["simplify"] = simplify
    if precision is not None:
        ctx["precision"] = precision
    ctx.update(extra)

    _state.stack.append(ctx)
    try:
        yield
    finally:
        _state.stack.pop()


# ---------------------------------------------------------------------------
# Context-aware convenience helpers
# ---------------------------------------------------------------------------


def symbol(name: str, *, pool: Any = None, domain: Any = None) -> Any:
    """Create a symbol, inferring *pool* and *domain* from the active context.

    Parameters
    ----------
    name : str
        Symbol name.
    pool : ExprPool, optional
        Explicit pool; overrides the context pool.
    domain : Domain, optional
        Explicit domain; overrides the context domain.

    Returns
    -------
    Expr
        The interned symbol expression.

    Raises
    ------
    RuntimeError
        If no pool is supplied and no pool is set in the active context.
    """
    ctx = _get()
    resolved_pool = pool or ctx.get("pool")
    resolved_domain = domain or ctx.get("domain")

    if resolved_pool is None:
        raise RuntimeError(
            "alkahest.symbol() requires a pool.  Either pass pool= or enter a "
            "alkahest.context(pool=...) block."
        )
    return resolved_pool.symbol(name, resolved_domain)


def active_pool() -> Any | None:
    """Return the pool from the innermost active context, or ``None``."""
    return get_context_value("pool")


def active_domain() -> Any | None:
    """Return the domain from the innermost active context, or ``None``."""
    return get_context_value("domain")


def simplify_enabled() -> bool:
    """Return ``True`` if the active context has ``simplify=True``."""
    return bool(get_context_value("simplify", False))
