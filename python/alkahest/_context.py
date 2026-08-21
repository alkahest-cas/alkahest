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

RW-7b — ``context(assumptions=...)`` additionally sets a thread-local
:class:`~alkahest.Assumptions` context. ``alkahest.simplify``,
``alkahest.simplify_log_exp``, and ``alkahest.solve`` pick it up automatically
when the caller omits the corresponding keyword argument, so agents don't have
to thread an ``Assumptions`` object through every call by hand::

    >>> p = alkahest.ExprPool()
    >>> x = p.symbol("x")
    >>> assumptions = alkahest.Assumptions(p)
    >>> assumptions.refine(p.gt(x, p.integer(0)))
    >>> with alkahest.context(pool=p, assumptions=assumptions):
    ...     alkahest.simplify(alkahest.sqrt(x**2)).value   # x, not sqrt(x^2)
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import local
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from ._budget import Budget

# ---------------------------------------------------------------------------
# Thread-local state
# ---------------------------------------------------------------------------

_state = local()


def _note_frame_pushed() -> None:
    """Tell the native layer a context frame went on the stack.

    `pool.symbol()` has to know whether a `context(domain=...)` is open, and
    asking this module through the FFI on every symbol costs ~2 µs — enough to
    make symbol interning 4.6x slower, since almost every call happens with no
    context at all. The native side keeps a count of live frames so it can skip
    the call entirely in that common case; this stack stays the only source of
    truth for what the frames *contain*.

    Best-effort: if the extension is not importable, the native side simply
    never takes its fast path and reads the stack as before.
    """
    try:
        from . import alkahest as _native

        _native._note_context_push()
    except Exception:  # pragma: no cover - extension always present in practice
        pass


def _note_frame_popped() -> None:
    """Counterpart to :func:`_note_frame_pushed`; called from a ``finally``."""
    try:
        from . import alkahest as _native

        _native._note_context_pop()
    except Exception:  # pragma: no cover
        pass


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
    assumptions: Any = None,
    require_certificate: bool | None = None,
    budget: Budget | None = None,
    **extra: Any,
) -> Generator[None, None, None]:
    """Thread-local context for Alkahest calls.

    Parameters
    ----------
    pool : ExprPool, optional
        Default expression pool used by ``alkahest.symbol`` and other
        pool-aware helpers when called without an explicit ``pool`` argument.
    domain : str or Domain, optional
        Default domain for symbol construction that omits ``domain`` — both
        ``alkahest.symbol(name)`` and ``pool.symbol(name)`` (e.g. ``"real"``,
        ``Domain.Integer``).  The two agree deliberately: the domain picks the
        SMT-LIB sort (``Int`` vs ``Real``), so a constructor that ignored it
        would change the question ``alkahest.smt.solve`` answers without
        changing any status a caller reads.  Pass an explicit ``domain=`` to
        either constructor to override the block.
    simplify : bool
        When ``True``, :func:`diff`, :func:`integrate`, :func:`sum_indefinite`,
        :func:`sum_definite`, :func:`product_indefinite`, and
        :func:`product_definite` post-process their :class:`DerivedResult` with
        :func:`simplify` (see :func:`simplify_enabled`).  Explicit
        :func:`simplify` / :func:`simplify_trig` calls are unchanged.
    precision : int, optional
        Default MPFR precision in bits for ball-arithmetic operations.
    assumptions : Assumptions, optional
        Default :class:`~alkahest.Assumptions` context, scoped to *pool*.
        :func:`alkahest.simplify`, :func:`alkahest.simplify_log_exp`, and
        :func:`alkahest.solve` pick this up automatically when called without
        an explicit ``assumptions``/``domain`` argument of their own (see
        :func:`active_assumptions`). Explicit arguments to those functions
        always take precedence over the context.
    require_certificate : bool, optional
        When ``True``, every derivation-producing call in the block
        (:func:`diff`, :func:`integrate`, the :func:`simplify` family, the
        sum/product solvers) must yield a Lean certificate, or it raises
        :class:`~alkahest.CertificateUnavailableError` (``E-CERT-001``) instead
        of returning an uncertified result. This is the ambient form of
        :func:`alkahest.require_certificate`; use it to stop a research loop
        from silently accumulating claims it cannot back up. Pass ``False`` in
        an inner block to opt back out.
    budget : Budget, optional
        P1 search plumbing item 4. When set, pushes a wall-clock / step
        budget and a determinism seed into the Rust-side cooperative
        checkpoint (``alkahest_core::budget``) for the scope of this block.
        :func:`alkahest.integrate` raises :class:`~alkahest.BudgetExceededError`
        (``E-BUDGET-*``) if the budget trips; :func:`alkahest.request_cancel`
        trips it from another thread. Like every other context key, a nested
        ``context(budget=...)`` shadows this one rather than merging with it —
        see :class:`~alkahest.Budget` and ``docs/mdbook/src/budgets.md``.
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

    Assumptions flow through to assumption-aware simplifiers::

        >>> x = p.symbol("x")
        >>> assumptions = alkahest.Assumptions(p)
        >>> assumptions.refine(p.gt(x, p.integer(0)))
        >>> with alkahest.context(pool=p, assumptions=assumptions):
        ...     alkahest.simplify_log_exp(alkahest.exp(alkahest.log(x))).value  # x

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
    if assumptions is not None:
        ctx["assumptions"] = assumptions
    if require_certificate is not None:
        ctx["require_certificate"] = require_certificate
    if budget is not None:
        ctx["budget"] = budget
    ctx.update(extra)

    _state.stack.append(ctx)
    _note_frame_pushed()
    budget_pushed = False
    if budget is not None:
        from . import alkahest as _native

        _native.push_budget(
            wall_ms=budget.wall_ms,
            max_steps=budget.max_steps,
            seed=budget.seed,
            max_bytes=budget.max_bytes,
        )
        budget_pushed = True
    try:
        yield
    finally:
        if budget_pushed:
            from . import alkahest as _native

            _native.pop_budget()
        _state.stack.pop()
        _note_frame_popped()


@contextmanager
def _overlay(**overrides: Any) -> Generator[None, None, None]:
    """Push a context frame that inherits the current one, with *overrides*.

    Unlike :func:`context`, which builds a frame from its own arguments only,
    this copies the active frame first.  Internal helpers use it to change one
    setting (e.g. suppress ``require_certificate``) without silently dropping
    the caller's pool, domain, or assumptions.
    """
    if not hasattr(_state, "stack"):
        _state.stack = []
    ctx = dict(_get())
    ctx.update(overrides)
    _state.stack.append(ctx)
    _note_frame_pushed()
    try:
        yield
    finally:
        _state.stack.pop()
        _note_frame_popped()


# ---------------------------------------------------------------------------
# Context-aware convenience helpers
# ---------------------------------------------------------------------------


def symbol(name: str, *, pool: Any = None, domain: Any = None, commutative: bool = True) -> Any:
    """Create a symbol, inferring *pool* and *domain* from the active context.

    ``pool.symbol(name)`` infers the domain from the same context, so the two
    constructors agree; this one additionally infers the *pool*.

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


def active_budget() -> Any | None:
    """Return the :class:`~alkahest.Budget` from the innermost active context,
    or ``None`` if no context set one.

    The budget is already active in the Rust-side cooperative checkpoint
    while its ``context(budget=...)`` block is open (see :func:`context`);
    this accessor is for introspection, e.g. a fan-out loop wanting to
    log/adjust its own remaining budget.
    """
    return get_context_value("budget")


def active_assumptions() -> Any | None:
    """Return the :class:`~alkahest.Assumptions` from the innermost active
    context, or ``None`` if no context set one.

    Used by :func:`alkahest.simplify`, :func:`alkahest.simplify_log_exp`, and
    :func:`alkahest.solve` to fall back to a caller-established assumption
    context when they're invoked without their own ``assumptions`` argument.
    """
    return get_context_value("assumptions")
