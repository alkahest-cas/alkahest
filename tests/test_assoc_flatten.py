"""`Add` and `Mul` are flat at construction, so associativity holds structurally.

The parsers build left-associative binary chains, so ``x*y*z`` arrives as
``(x*y)*z``; the n-ary builder API produces a flat three-child ``Mul``.  Those
used to be *different expressions*, which meant every matcher that scans the
top-level arguments of a product or a sum saw two children where the user wrote
three, and silently declined to fire on parsed input.

``ExprPool.mul``/``ExprPool.add`` now splice nested same-operator children at
construction.  Because both parsers — the Rust one and the separate pure-Python
Pratt parser in ``alkahest/_parse.py`` — and the Python operator overloads all
funnel through those two kernel constructors, the fix reaches every construction
path at once.  This file pins that from Python.

It is associativity and nothing else: no reordering beyond the canonical sort
the pool already applied, no constant folding, no identity elimination.
"""

import alkahest as ak
import pytest
from alkahest.alkahest import ExprPool, simplify

_TRIPLES = ["x*y*z", "(x*y)*z", "x*(y*z)"]
_SUM_TRIPLES = ["x+y+z", "(x+y)+z", "x+(y+z)"]


def _pool_xyz():
    pool = ExprPool()
    return pool, pool.symbol("x"), pool.symbol("y"), pool.symbol("z")


# ---------------------------------------------------------------------------
# 1. The parsed chain and the n-ary builder call are the same expression.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src", _TRIPLES)
def test_parsed_product_chain_equals_flat_mul(src):
    pool, x, y, z = _pool_xyz()
    assert ak.parse(src, pool) == pool.mul([x, y, z]), (
        f"{src} parsed to {ak.parse(src, pool)}, not the flat product"
    )


@pytest.mark.parametrize("src", _SUM_TRIPLES)
def test_parsed_sum_chain_equals_flat_add(src):
    pool, x, y, z = _pool_xyz()
    assert ak.parse(src, pool) == pool.add([x, y, z])


def test_deeper_chains_flatten():
    pool, x, y, z = _pool_xyz()
    assert ak.parse("x*y*z*x*y", pool) == pool.mul([x, y, z, x, y])
    assert ak.parse("x+y+z+x+y", pool) == pool.add([x, y, z, x, y])
    assert ak.parse("x*y*z*x*y*z*x", pool) == pool.mul([x, y, z, x, y, z, x])


def test_node_arity_is_what_the_user_wrote():
    """A three-factor product really has three children, not two."""
    pool, _x, _y, _z = _pool_xyz()
    tag, children = ak.parse("x*y*z", pool).node()
    assert tag == "mul"
    assert len(children) == 3
    tag, children = ak.parse("x+y+z", pool).node()
    assert tag == "add"
    assert len(children) == 3


# ---------------------------------------------------------------------------
# 2. The builder path, pinning that this is not a parser-only fix.
# ---------------------------------------------------------------------------


def test_builder_mul_flattens():
    pool, x, y, z = _pool_xyz()
    flat = pool.mul([x, y, z])
    assert pool.mul([pool.mul([x, y]), z]) == flat
    assert pool.mul([x, pool.mul([y, z])]) == flat


def test_builder_add_flattens():
    pool, x, y, z = _pool_xyz()
    flat = pool.add([x, y, z])
    assert pool.add([pool.add([x, y]), z]) == flat
    assert pool.add([x, pool.add([y, z])]) == flat


def test_operator_overloads_flatten():
    """``x * y * z`` in Python goes through ``__mul__``, not the parser."""
    pool, x, y, z = _pool_xyz()
    assert x * y * z == pool.mul([x, y, z])
    assert x * (y * z) == pool.mul([x, y, z])
    assert x + y + z == pool.add([x, y, z])
    assert x + (y + z) == pool.add([x, y, z])


def test_splicing_does_not_cross_operators():
    """An ``Add`` inside a ``Mul`` is a different operator and must stay put."""
    pool, x, y, z = _pool_xyz()
    assert pool.mul([pool.add([x, y]), z]) != pool.mul([x, y, z])
    assert pool.add([pool.mul([x, y]), z]) != pool.add([x, y, z])
    tag, children = ak.parse("(x+y)*z", pool).node()
    assert tag == "mul"
    assert len(children) == 2


def test_flattening_is_what_simplify_already_did():
    """`simplify` flattened; it just ran too late to help anything upstream."""
    pool, x, y, z = _pool_xyz()
    parsed = ak.parse("x*y*z", pool)
    assert simplify(parsed).value == parsed
    assert simplify(parsed).value == pool.mul([x, y, z])


# ---------------------------------------------------------------------------
# 3. Depth.  Flattening *reduces* depth, so a left-associated chain that used
#    to be refused by the E-DEPTH-001 ceiling is now a depth-2 node.
# ---------------------------------------------------------------------------


def test_a_long_chain_of_binary_operators_is_now_shallow():
    pool = ExprPool()
    x = pool.symbol("x")
    acc = x
    for i in range(3000):
        acc = acc + pool.integer(i)
    # Would have been depth 3001, over the 2048 ceiling, before flattening.
    assert str(acc)  # no E-DEPTH-001, and printing does not overflow
    assert len(acc.node()[1]) == 3001


# ---------------------------------------------------------------------------
# 4. Display round-trips.  `parse → display → parse` must be a fixpoint, in
#    plain text and in the LaTeX / unicode printers.
# ---------------------------------------------------------------------------

_ROUND_TRIP = [
    "x*y*z",
    "x+y+z",
    "x*y+z",
    "(x+y)*z",
    "x*y*z+x*y+z",
    "2*x*y*z",
    "x^2*y*z",
    "x*y*z^2+x+y",
]


@pytest.mark.parametrize("src", _ROUND_TRIP)
def test_display_round_trip_is_stable(src):
    pool, _x, _y, _z = _pool_xyz()
    once = ak.parse(src, pool)
    rendered = str(once)
    twice = ak.parse(rendered, pool)
    assert once == twice, f"{src} -> {rendered} -> {twice}"
    assert str(twice) == rendered


@pytest.mark.parametrize("src", _ROUND_TRIP)
def test_latex_and_unicode_printers_accept_flat_nodes(src):
    pool, _x, _y, _z = _pool_xyz()
    e = ak.parse(src, pool)
    assert ak.latex(e)
    assert ak.unicode_str(e)
    # Spelling the same function three ways gives one node, hence one rendering.
    assert ak.latex(e) == ak.latex(ak.parse(str(e), pool))
    assert ak.unicode_str(e) == ak.unicode_str(ak.parse(str(e), pool))


@pytest.mark.parametrize(("a", "b"), [(_TRIPLES[0], _TRIPLES[1]), (_TRIPLES[0], _TRIPLES[2])])
def test_all_spellings_print_identically(a, b):
    pool, _x, _y, _z = _pool_xyz()
    assert str(ak.parse(a, pool)) == str(ak.parse(b, pool))
    assert ak.latex(ak.parse(a, pool)) == ak.latex(ak.parse(b, pool))
    assert ak.unicode_str(ak.parse(a, pool)) == ak.unicode_str(ak.parse(b, pool))
