"""PyTree — flatten / unflatten nested Python structures of :class:`Expr`.

PA-10 — PyTree layer in the Python tier.

Follows the JAX ``jax.tree_util`` convention: a *treedef* records the structure
(nested lists, tuples, dicts) while *leaves* are the actual :class:`Expr` nodes.
Transformations like ``alkahest.map_exprs`` then apply a function to every leaf
without the caller having to traverse the structure manually.

Example
-------
>>> import alkahest
>>> p = alkahest.ExprPool()
>>> x = p.symbol("x")
>>> y = p.symbol("y")
>>>
>>> result = alkahest.map_exprs(lambda e: alkahest.diff(e, x).value, [x**2, alkahest.sin(y)])
>>> [str(r) for r in result]
['(2 * x)', 'cos(y)']
>>>
>>> leaves, treedef = alkahest.flatten_exprs([x**2, {"a": alkahest.sin(y)}])
>>> alkahest.unflatten_exprs(leaves, treedef)
[x**2, {"a": sin(y)}]
"""

from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# TreeDef — encodes the nested structure
# ---------------------------------------------------------------------------


class _Leaf:
    """Sentinel: this node is a leaf (an Expr)."""


class _ListNode:
    def __init__(self, length: int):
        self.length = length

    def __repr__(self) -> str:
        return f"_ListNode({self.length})"


class _TupleNode:
    def __init__(self, length: int):
        self.length = length

    def __repr__(self) -> str:
        return f"_TupleNode({self.length})"


class _DictNode:
    def __init__(self, keys: list):
        self.keys = keys

    def __repr__(self) -> str:
        return f"_DictNode({self.keys!r})"


class TreeDef:
    """Encodes the structure of a nested container so it can be reconstructed.

    Internal representation: a flat list of node descriptors obtained by a
    pre-order traversal of the tree.  Leaves are represented by ``_Leaf()``
    instances.
    """

    def __init__(self, nodes: list):
        self._nodes = nodes

    def __repr__(self) -> str:
        return f"TreeDef({self._nodes!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TreeDef):
            return NotImplemented
        return self._nodes == other._nodes


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


def _is_expr(obj: Any) -> bool:
    """Return True if *obj* is a Alkahest ``Expr`` node."""
    try:
        from .alkahest import Expr  # noqa: PLC0415

        return isinstance(obj, Expr)
    except ImportError:
        return False


def _flatten(obj: Any, leaves: list, nodes: list) -> None:
    if _is_expr(obj):
        leaves.append(obj)
        nodes.append(_Leaf())
    elif isinstance(obj, dict):
        keys = sorted(obj.keys(), key=str)
        nodes.append(_DictNode(keys))
        for k in keys:
            _flatten(obj[k], leaves, nodes)
    elif isinstance(obj, tuple):
        nodes.append(_TupleNode(len(obj)))
        for item in obj:
            _flatten(item, leaves, nodes)
    elif isinstance(obj, list):
        nodes.append(_ListNode(len(obj)))
        for item in obj:
            _flatten(item, leaves, nodes)
    else:
        # Non-Expr scalars / opaque objects are treated as leaves too.
        leaves.append(obj)
        nodes.append(_Leaf())


def flatten_exprs(obj: Any) -> tuple[list, TreeDef]:
    """Flatten a nested structure into ``(leaves, treedef)``.

    Parameters
    ----------
    obj : any nested structure of Expr, list, tuple, or dict

    Returns
    -------
    leaves : list
        Flat list of ``Expr`` leaves (in pre-order).
    treedef : TreeDef
        Opaque structure descriptor; pass to :func:`unflatten_exprs` to
        reconstruct the original container.

    Example
    -------
    >>> leaves, td = alkahest.flatten_exprs([x**2, {"a": sin(y)}])
    >>> len(leaves)
    2
    """
    leaves: list = []
    nodes: list = []
    _flatten(obj, leaves, nodes)
    return leaves, TreeDef(nodes)


# ---------------------------------------------------------------------------
# Unflatten
# ---------------------------------------------------------------------------


def _unflatten(nodes: list, pos: list, leaves: list, leaf_pos: list) -> Any:
    """Recursively reconstruct the tree."""
    node = nodes[pos[0]]
    pos[0] += 1

    if isinstance(node, _Leaf):
        val = leaves[leaf_pos[0]]
        leaf_pos[0] += 1
        return val
    elif isinstance(node, _ListNode):
        return [_unflatten(nodes, pos, leaves, leaf_pos) for _ in range(node.length)]
    elif isinstance(node, _TupleNode):
        return tuple(_unflatten(nodes, pos, leaves, leaf_pos) for _ in range(node.length))
    elif isinstance(node, _DictNode):
        return {k: _unflatten(nodes, pos, leaves, leaf_pos) for k in node.keys}
    else:
        raise ValueError(f"Unknown node type: {type(node)}")


def unflatten_exprs(leaves: list, treedef: TreeDef) -> Any:
    """Reconstruct a nested structure from *leaves* and *treedef*.

    Parameters
    ----------
    leaves : list
        Flat list of values (typically transformed ``Expr`` objects).
    treedef : TreeDef
        Structure descriptor returned by :func:`flatten_exprs`.

    Returns
    -------
    The original nested structure with leaves replaced by the provided values.
    """
    pos = [0]
    leaf_pos = [0]
    return _unflatten(treedef._nodes, pos, leaves, leaf_pos)


# ---------------------------------------------------------------------------
# map_exprs — the primary user-facing function
# ---------------------------------------------------------------------------


def map_exprs(fn: Callable, obj: Any) -> Any:
    """Apply *fn* to every ``Expr`` leaf in a nested structure.

    Parameters
    ----------
    fn : callable
        Function ``Expr → Expr`` applied to each leaf.
    obj : any nested structure
        May be a single ``Expr``, a list, tuple, or dict thereof — arbitrarily
        nested.

    Returns
    -------
    The same nested structure with every ``Expr`` leaf replaced by ``fn(leaf)``.

    Example
    -------
    >>> import alkahest
    >>> p = alkahest.ExprPool()
    >>> x = p.symbol("x")
    >>> result = alkahest.map_exprs(lambda e: alkahest.diff(e, x).value, [x**2, alkahest.sin(x)])
    >>> [str(r) for r in result]
    ['(2 * x)', 'cos(x)']
    """
    leaves, treedef = flatten_exprs(obj)
    new_leaves = [fn(leaf) if _is_expr(leaf) else leaf for leaf in leaves]
    return unflatten_exprs(new_leaves, treedef)


__all__ = [
    "TreeDef",
    "flatten_exprs",
    "unflatten_exprs",
    "map_exprs",
]
