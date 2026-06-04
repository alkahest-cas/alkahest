"""Symbolic linear algebra coverage (issues #41–#46)."""

from __future__ import annotations

import alkahest


def test_nullspace_rank_column_row_space():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix([[pool.integer(1), pool.integer(2)]])
    assert len(m.nullspace()) == 1
    assert m.rank() == 1
    assert len(m.column_space()) == 1
    assert len(m.row_space()) == 1


def test_lu_rational_2x2():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix([[pool.integer(2), pool.integer(1)], [pool.integer(4), pool.integer(3)]])
    lower, upper, perm = m.lu()
    permuted = alkahest.Matrix([m.to_list()[i] for i in perm])
    assert (lower @ upper).simplify().to_list() == permuted.simplify().to_list()


def test_jordan_block_2x2():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix([[pool.integer(2), pool.integer(1)], [pool.integer(0), pool.integer(2)]])
    p, j = m.jordan_form()
    assert p.rows == 2
    assert j.rows == 2
    inv = p.inverse()
    assert (p @ j @ inv).simplify().to_list() == m.simplify().to_list()


def test_rational_canonical_diagonal():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix(
        [
            [pool.integer(1), pool.integer(0)],
            [pool.integer(0), pool.integer(2)],
        ]
    )
    p, c = m.rational_canonical_form()
    assert p.rows == 2
    assert c.rows == 2


def test_minimal_polynomial_diagonal():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix(
        [
            [pool.integer(1), pool.integer(0)],
            [pool.integer(0), pool.integer(2)],
        ]
    )
    minpoly = m.minimal_polynomial()
    assert minpoly is not None


def test_matrix_exp_diagonal():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix(
        [
            [pool.integer(0), pool.integer(0)],
            [pool.integer(0), pool.integer(1)],
        ]
    )
    expm = m.matrix_exp()
    assert expm.rows == 2
