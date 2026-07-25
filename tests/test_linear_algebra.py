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


def _int_matrix(pool, rows):
    """Build a Matrix of Python ints, coercing each entry into `pool`."""
    return alkahest.Matrix([[pool.integer(x) for x in row] for row in rows])


def _entries(m):
    """Simplified node ids for every entry, for structural comparison."""
    return [[e.node() for e in row] for row in m.simplify().to_list()]


def test_matrix_star_matrix_equals_matmul():
    """`A * B` is the matrix product (SymPy convention), identical to `A @ B`."""
    pool = alkahest.ExprPool()
    a = _int_matrix(pool, [[1, 2], [3, 4]])
    b = _int_matrix(pool, [[5, 6], [7, 8]])
    assert _entries(a * b) == _entries(a @ b)
    # Non-square inner-dimension product also matches.
    c = _int_matrix(pool, [[1, 2, 3], [4, 5, 6]])
    assert (a * c).shape() == (2, 3)
    assert _entries(a * c) == _entries(a @ c)


def test_matrix_scalar_multiplication_both_sides():
    """`A * k`, `k * A`, and `A * Expr` scale every entry (int, float, Expr)."""
    pool = alkahest.ExprPool()
    a = _int_matrix(pool, [[1, 2], [3, 4]])
    expected = [[2, 4], [6, 8]]
    right = a * 2
    left = 2 * a
    for prod in (right, left):
        assert _entries(prod) == [[pool.integer(v).node() for v in row] for row in expected]
    # scalar_mul named method agrees with `*`.
    assert _entries(a.scalar_mul(2)) == _entries(a * 2)
    # Expr scalar on both sides.
    x = pool.symbol("x")
    assert _entries(a * x) == _entries(x * a)
    # float scalar is accepted.
    assert (a * 2.0).shape() == (2, 2)


def test_matrix_multiply_named_method():
    """`A.multiply(B)` is an alias for the matrix product."""
    pool = alkahest.ExprPool()
    a = _int_matrix(pool, [[1, 2], [3, 4]])
    b = _int_matrix(pool, [[0, 1], [1, 0]])
    assert _entries(a.multiply(b)) == _entries(a @ b)


def test_matrix_star_dimension_mismatch_raises():
    """Incompatible `*` product raises MatrixError E-MAT-001 with shapes."""
    pool = alkahest.ExprPool()
    c = _int_matrix(pool, [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(alkahest.MatrixError) as exc_info:
        _ = c * c
    assert exc_info.value.code == "E-MAT-001"
    assert "2×3" in str(exc_info.value)


def test_matrix_power_non_negative_integer():
    """`A ** n` is repeated matrix product; `A ** 0` is the identity."""
    pool = alkahest.ExprPool()
    a = _int_matrix(pool, [[1, 1], [0, 1]])
    assert _entries(a**0) == _entries(_int_matrix(pool, [[1, 0], [0, 1]]))
    assert _entries(a**1) == _entries(a)
    assert _entries(a**2) == _entries(a @ a)
    assert _entries(a**3) == _entries(a @ a @ a)
    # [[1,1],[0,1]] ** 3 == [[1,3],[0,1]].
    assert _entries(a**3) == _entries(_int_matrix(pool, [[1, 3], [0, 1]]))


def test_matrix_power_negative_declines():
    """Negative exponents raise TypeError (no inverse via **)."""
    pool = alkahest.ExprPool()
    a = _int_matrix(pool, [[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        _ = a**-1


def test_matrix_power_non_square_declines():
    """Powering a non-square matrix raises MatrixError E-MAT-001."""
    pool = alkahest.ExprPool()
    c = _int_matrix(pool, [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(alkahest.MatrixError) as exc_info:
        _ = c**2
    assert exc_info.value.code == "E-MAT-001"


def test_matrix_hadamard_elementwise():
    """`hadamard` multiplies corresponding entries; mismatched shapes decline."""
    pool = alkahest.ExprPool()
    a = _int_matrix(pool, [[1, 2], [3, 4]])
    b = _int_matrix(pool, [[5, 6], [7, 8]])
    assert _entries(a.hadamard(b)) == [
        [pool.integer(v).node() for v in row] for row in [[5, 12], [21, 32]]
    ]
    c = _int_matrix(pool, [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(alkahest.MatrixError) as exc_info:
        a.hadamard(c)
    assert exc_info.value.code == "E-MAT-001"
