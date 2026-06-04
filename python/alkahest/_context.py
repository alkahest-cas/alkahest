"""Thread-local context manager for Alkahest.

RW-7 — Provides a ``with alkahest.context(...)`` block that sets thread-local
defaults for domain, pool, and simplification so callers don't have to repeat
common arguments on every call.

The kernel remains stateless; this module is purely a Python-layer convenience.

Example
-------
>>> import alkahest
>>> p = alkahest.ExprPool()
>>> with alkahest.context(pool=p, domain="real", simplify=True):
...     x = alkahest.symbol("x")          # domain and pool inferred
...     expr = x ** 2
...     d = alkahest.diff(expr, x)        # algebraic simplify applied to .value automatically
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import local
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

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
    domain : str or Domain, optional
        Default domain for ``alkahest.symbol(name)`` calls that omit ``domain``
        (e.g. ``"real"``, ``Domain.Integer``).
    simplify : bool
        When ``True``, :func:`diff`, :func:`integrate`, :func:`sum_indefinite`,
        :func:`sum_definite`, :func:`product_indefinite`, and
        :func:`product_definite` post-process their :class:`DerivedResult` with
        :func:`simplify` (see :func:`simplify_enabled`).  Explicit
        :func:`simplify` / :func:`simplify_trig` calls are unchanged.
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
    >>> with alkahest.context(pool=p, domain="real"):
    ...     x = alkahest.symbol("x")
    ...     y = alkahest.symbol("y")
    ...     expr = x ** 2 + y ** 2

    Contexts nest (inner keys shadow outer ones)::

        with alkahest.context(pool=p):
            with alkahest.context(domain="integer"):
                # pool from outer context; domain overridden here.
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


def symbol(name: str, *, pool: Any = None, domain: Any = None, commutative: bool = True) -> Any:
    """Create a symbol, inferring *pool* and *domain* from the active context.

    Parameters
    ----------
    name : str
        Symbol name.
    pool : ExprPool, optional
        Explicit pool; overrides the context pool.
    domain : str or Domain, optional
        Explicit domain; overrides the context domain.
    commutative : bool
        When ``False``, the symbol does not commute under multiplication (V3-2).

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
    return resolved_pool.symbol(name, resolved_domain, commutative=commutative)


def active_pool() -> Any | None:
    """Return the pool from the innermost active context, or ``None``."""
    return get_context_value("pool")


def active_domain() -> Any | None:
    """Return the domain from the innermost active context, or ``None``."""
    return get_context_value("domain")


def simplify_enabled() -> bool:
    """Return ``True`` if the active context has ``simplify=True``."""
    return bool(get_context_value("simplify", False))
