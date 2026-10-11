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
    # The full Pauli algebra uses the pool's imaginary unit, so i·i = -1 works.
    r = alkahest.simplify_pauli(sx * sy)
    assert r.value == pool.imaginary_unit() * sz


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


def test_g10c_report_repros_without_oracles():
    """The report's G10 noncommutative repros; oracle-checked versions live in
    ``test_noncommutative_g10c.py`` (which needs numpy and SymPy)."""
    pool = alkahest.ExprPool()
    i_ = pool.imaginary_unit()
    X, P = (pool.symbol(n, commutative=False) for n in ("X", "P"))
    hbar = pool.symbol("hbar")
    rule = alkahest.make_rule(X * P, P * X + i_ * hbar)
    out = alkahest.simplify_with(X * P * P - P * P * X, [rule]).value
    assert alkahest.nc_terms(out - 2 * i_ * hbar * P) == []
    A, B, C, D = (pool.symbol(n) for n in "ABCD")
    assert alkahest.simplify_with(A * B * D, [alkahest.make_rule(A * B, C)]).value == C * D
    sx, sy, sz = (pool.symbol(n, commutative=False) for n in ("sx", "sy", "sz"))
    assert alkahest.simplify_pauli(sz**2).value == pool.integer(1)
    assert alkahest.simplify_pauli((sx + sy) ** 2).value == pool.integer(2)
    assert alkahest.nc_terms(alkahest.simplify_pauli(sx * sy * sx).value + sy) == []
    a, ad = (pool.symbol(n, commutative=False) for n in ("a", "ad"))
    r = alkahest.normal_order(a * ad**3, bosons=(a, ad))
    assert alkahest.nc_terms(r - ad**3 * a - 3 * ad**2) == []
    assert alkahest.gamma_trace([5, 0, 1, 2, 3], pool=pool) == -4 * i_
