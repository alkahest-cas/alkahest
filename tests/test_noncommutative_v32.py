"""V3-2 — non-commutative symbols and Pauli / Clifford simplify helpers."""

from __future__ import annotations

import alkahest


def test_nc_mul_order_preserved():
    pool = alkahest.ExprPool()
    a = pool.symbol("A", "real", commutative=False)
    b = pool.symbol("B", "real", commutative=False)
    assert a * b != b * a


def test_simplify_pauli_sx_sy():
    pool = alkahest.ExprPool()
    sx = pool.symbol("sx", "complex", commutative=False)
    sy = pool.symbol("sy", "complex", commutative=False)
    sz = pool.symbol("sz", "complex", commutative=False)
    imag = pool.symbol("ImagUnit", "complex", commutative=True)
    r = alkahest.simplify_pauli(sx * sy)
    assert r.value == imag * sz


def test_simplify_clifford():
    pool = alkahest.ExprPool()
    e1 = pool.symbol("cliff_e1", "real", commutative=False)
    e2 = pool.symbol("cliff_e2", "real", commutative=False)
    r = alkahest.simplify_clifford_orthogonal(e1 * e2)
    # Match n-aryMul from the kernel (flatten vs nested binary `*`).
    equiv = alkahest.simplify(pool.integer(-1) * e2 * e1).value
    assert r.value == equiv


def test_simplify_egraph_falls_back_for_nc():
    pool = alkahest.ExprPool()
    a = pool.symbol("A", "real", commutative=False)
    expr = a + pool.integer(0)
    r = alkahest.simplify_egraph(expr)
    assert r.value == a
