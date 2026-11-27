"""Small expression helpers shared by the FEX modules (no NumPy, so the
Lean certificate emitter can use them in a minimal environment)."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import alkahest as _ak


def _children(e: Any) -> list[Any]:
    nd = e.node()
    out = []
    for a in nd[1:]:
        if isinstance(a, (list, tuple)):
            out.extend(c for c in a if isinstance(c, _ak.Expr))
        elif isinstance(a, _ak.Expr):
            out.append(a)
    return out


def _symbols_in(e: Any, acc: set | None = None) -> set:
    acc = set() if acc is None else acc
    stack = [e]
    seen = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        nd = n.node()
        if nd[0] == "symbol":
            acc.add(n)
        else:
            stack.extend(_children(n))
    return acc


def _number_value(e: Any) -> Fraction | float | None:
    """The value of a numeric literal node (integer, rational, float, or a
    power of one), else ``None``."""
    nd = e.node()
    k = nd[0]
    if k == "integer":
        return Fraction(int(nd[1]))
    if k == "rational":
        return Fraction(int(nd[1]), int(nd[2]))
    if k == "float":
        return float(nd[1])
    if k == "pow":
        base, ex = nd[1], nd[2] if len(nd) > 2 else None
        if isinstance(base, list):
            base, ex = base
        b = _number_value(base)
        x = _number_value(ex) if ex is not None else None
        if (
            isinstance(b, Fraction)
            and isinstance(x, Fraction)
            and x.denominator == 1
            and b != 0
            and abs(x) <= 64
        ):
            return b ** int(x)
    return None


def _rebuild(e: Any, f: Any) -> Any:
    """Rebuild ``e`` bottom-up, applying ``f(node, new_children)`` →
    ``Expr | None`` (``None`` keeps the default reconstruction)."""
    nd = e.node()
    kind = nd[0]
    kids = _children(e)
    if not kids:
        r = f(e, [])
        return e if r is None else r
    new = [_rebuild(c, f) for c in kids]
    r = f(e, new)
    if r is not None:
        return r
    if kind == "add":
        out = new[0]
        for c in new[1:]:
            out = out + c
        return out
    if kind == "mul":
        out = new[0]
        for c in new[1:]:
            out = out * c
        return out
    if kind == "pow":
        return new[0] ** new[1]
    if kind == "func":
        return e._pool.func(nd[1], new)
    return e
