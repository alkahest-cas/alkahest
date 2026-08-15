"""Builders for the ``q``-hypergeometric function heads (M4b).

``q``-Zeilberger consumes ordinary :class:`~alkahest.Expr` trees; the two heads
it recognises, ``qbinomial`` and ``qpochhammer``, are plain named function
nodes.  These helpers are sugar over ``pool.func(...)`` — they exist so a
caller writes the mathematics rather than the spelling, and so that an ``int``
argument does not have to be lifted by hand.

Nothing here validates: the parser in the kernel decides what is in class and
refuses with a coded ``E-HOLO-02x`` error otherwise, which is where that
decision belongs.
"""

from __future__ import annotations

from typing import Any


def _lift(pool: Any, v: Any) -> Any:
    """An ``int`` becomes a pool integer; an ``Expr`` passes through."""
    return pool.integer(v) if isinstance(v, int) else v


def qpochhammer(pool: Any, u: Any, d: Any, v: Any) -> Any:
    """``(q**u; q**d)_v`` — the ``q``-Pochhammer symbol, as an ``Expr``.

    ``u`` and ``v`` must be integer-affine in ``n`` and ``k``; ``d`` is the
    base step, a positive integer literal (``d = 1`` is the usual base ``q``,
    ``d = 2`` gives ``(q**u; q**2)_v``).

    The symbol is defined for **every** integer length by its own recurrence
    ``(a;q**d)_{v+1} = (a;q**d)_v · (1 − a·q**(d·v))``, so a negative ``v`` is
    meaningful and is exactly what makes a ``q``-binomial vanish outside its
    row.
    """
    return pool.func("qpochhammer", [_lift(pool, u), _lift(pool, d), _lift(pool, v)])


def qbinomial(pool: Any, top: Any, bot: Any) -> Any:
    """``[top; bot]_q`` — the Gaussian binomial coefficient, as an ``Expr``.

    Shorthand for ``(q;q)_top / ((q;q)_bot · (q;q)_{top−bot})``, which is how
    the kernel expands it; both arguments must be integer-affine in ``n`` and
    ``k``.
    """
    return pool.func("qbinomial", [_lift(pool, top), _lift(pool, bot)])
