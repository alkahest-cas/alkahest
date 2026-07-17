"""Symbolic linear algebra coverage (issues #41–#46)."""

from __future__ import annotations

import alkahest
import pytest


def test_matrix_from_rows_mixed_int_expr():
    """from_rows accepts bare ints alongside Expr; pool is inferred from the Expr."""
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    m = alkahest.Matrix.from_rows([[x, 1], [0, x]])
    assert m.shape() == (2, 2)
    assert m.get(0, 1).node() == pool.integer(1).node()


def test_matrix_from_rows_all_int_with_active_pool():
    """from_rows accepts an all-int matrix when an active pool is set via context()."""
    pool = alkahest.ExprPool()
    with alkahest.context(pool=pool):
        m = alkahest.Matrix.from_rows([[0, 1], [-1, 0]])
        m2 = alkahest.Matrix([[1, 0], [0, 1]])
    assert m.shape() == (2, 2)
    assert m2.shape() == (2, 2)
    assert m.get(1, 0).node() == pool.integer(-1).node()


def test_rref_rank_consistency():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix(
        [
            [pool.integer(1), pool.integer(2), pool.integer(3)],
            [pool.integer(2), pool.integer(4), pool.integer(6)],
        ]
    )
    r = m.rref().simplify()
    assert r.shape() == (2, 3)
    assert m.rank() == 1
    assert r.get(1, 0).node() == pool.integer(0).node()
    assert r.get(1, 1).node() == pool.integer(0).node()
    assert r.get(1, 2).node() == pool.integer(0).node()


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


def test_jordan_defective_3x3():
    """Defective matrix with a 3×3 Jordan block (algebraic mult 3, geometric mult 1)."""
    pool = alkahest.ExprPool()
    z = pool.integer(0)
    one = pool.integer(1)
    two = pool.integer(2)
    m = alkahest.Matrix(
        [
            [two, one, z],
            [z, two, one],
            [z, z, two],
        ]
    )
    p, j = m.jordan_form()
    assert p.rows == 3
    assert j.rows == 3
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
    one = pool.integer(1)
    two = pool.integer(2)
    z = pool.integer(0)
    m = alkahest.Matrix([[one, z], [z, two]])
    minpoly = alkahest.simplify(m.minimal_polynomial()).value
    # Distinct eigenvalues {1, 2} ⇒ degree-2 minimal polynomial (λ² - 3λ + 2).
    node = minpoly.node()
    assert node[0] == "add"
    assert len(node[1]) == 3


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


def test_non_square_jordan_form_declines():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix([[pool.integer(1), pool.integer(0), pool.integer(0)]])
    with pytest.raises(alkahest.LinearAlgebraError) as exc_info:
        m.jordan_form()
    assert exc_info.value.code == "E-LINALG-001"


def test_non_square_minimal_polynomial_declines():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix([[pool.integer(1), pool.integer(2)]])
    with pytest.raises(alkahest.LinearAlgebraError) as exc_info:
        m.minimal_polynomial()
    assert exc_info.value.code == "E-LINALG-001"


def test_non_square_matrix_exp_declines():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix([[pool.integer(1), pool.integer(0)]])
    with pytest.raises(alkahest.LinearAlgebraError) as exc_info:
        m.matrix_exp()
    assert exc_info.value.code == "E-LINALG-001"


def test_cholesky_non_spd_declines():
    pool = alkahest.ExprPool()
    m = alkahest.Matrix(
        [
            [pool.integer(1), pool.integer(2)],
            [pool.integer(2), pool.integer(1)],
        ]
    )
    with pytest.raises(alkahest.LinearAlgebraError) as exc_info:
        m.cholesky()
    assert exc_info.value.code == "E-LINALG-003"
