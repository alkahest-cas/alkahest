"""Textbook gate — linear algebra.

First-course linear algebra on `alkahest.Matrix`: determinant, trace, rank,
rref, inverse, transpose, nullspace, LU factorization, and eigenvalues.

**Notable gap found while writing this file:** `Matrix` has no multiplication
operator at all — neither matrix-matrix (`M1 * M2` raises `TypeError`) nor
scalar-matrix (`2 * M` / `M * 2` both raise `TypeError`), despite supporting
far more advanced operations via method calls (`eigenvects`, `jordan_form`,
`qr`, `cholesky`, `rational_canonical_form`). Only `+` and `-` are wired up.
This isn't something to write an `xfail` for — there's no function to call —
so verification below reconstructs matrix multiplication independently in
plain Python (`_matmul`) over `.to_list()`-extracted numeric values, and uses
it to check `alkahest`'s determinant/inverse/nullspace/LU results against
ground truth without ever calling a multiplication operator that doesn't
exist. See `tests/textbook_gate/README.md` for the general verification
philosophy (never assert on alkahest's printed normal form).
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


def to_floats(mat: ak.Matrix) -> list[list[float]]:
    """Extract a Matrix's entries as plain Python floats via eval_expr."""
    rows, cols = mat.shape()
    return [[ak.eval_expr(mat.get(i, j), {}) for j in range(cols)] for i in range(rows)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Plain-Python matrix multiply — alkahest's Matrix has no `*` operator
    (see module docstring), so ground truth is computed independently here.
    """
    rows, inner = len(a), len(a[0])
    inner2, cols = len(b), len(b[0])
    assert inner == inner2, f"shape mismatch: {inner} != {inner2}"
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)
    ]


def _matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(row[k] * v[k] for k in range(len(v))) for row in a]


def assert_matrix_close(
    got: list[list[float]], expected: list[list[float]], *, atol: float = 1e-9
) -> None:
    assert len(got) == len(expected), f"row count mismatch: {len(got)} vs {len(expected)}"
    assert len(got[0]) == len(expected[0]), (
        f"col count mismatch: {len(got[0])} vs {len(expected[0])}"
    )
    for i, row in enumerate(got):
        for j, v in enumerate(row):
            assert math.isclose(v, expected[i][j], abs_tol=atol), (
                f"at ({i},{j}): got={v!r} expected={expected[i][j]!r}"
            )


def identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# --- determinant / trace / rank ---------------------------------------------


def test_det_2x2(pool):
    M = ak.Matrix([[pool.integer(2), pool.integer(1)], [pool.integer(1), pool.integer(1)]])
    assert ak.eval_expr(M.det(), {}) == pytest.approx(1.0)


def test_det_3x3(pool):
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    M = ak.Matrix([[pool.integer(v) for v in row] for row in rows])
    # hand-computed: 1*(5*10-6*8) - 2*(4*10-6*7) + 3*(4*8-5*7) = -3+4-3 = -3
    assert ak.eval_expr(M.det(), {}) == pytest.approx(-3.0)


def test_det_singular_is_zero(pool):
    M = ak.Matrix([[pool.integer(1), pool.integer(2)], [pool.integer(2), pool.integer(4)]])
    assert ak.eval_expr(M.det(), {}) == pytest.approx(0.0)


def test_trace_2x2(pool):
    M = ak.Matrix([[pool.integer(3), pool.integer(0)], [pool.integer(0), pool.integer(5)]])
    assert ak.eval_expr(M.trace(), {}) == pytest.approx(8.0)


def test_rank_full(pool):
    M = ak.Matrix([[pool.integer(1), pool.integer(2)], [pool.integer(3), pool.integer(4)]])
    assert M.rank() == 2


def test_rank_deficient(pool):
    M = ak.Matrix([[pool.integer(1), pool.integer(2)], [pool.integer(2), pool.integer(4)]])
    assert M.rank() == 1


# --- rref ---------------------------------------------------------------------


def test_rref_invertible_goes_to_identity(pool):
    M = ak.Matrix([[pool.integer(2), pool.integer(1)], [pool.integer(1), pool.integer(1)]])
    assert to_floats(M.rref()) == identity(2)


def test_rref_singular_has_zero_row(pool):
    M = ak.Matrix([[pool.integer(1), pool.integer(2)], [pool.integer(2), pool.integer(4)]])
    r = to_floats(M.rref())
    assert any(all(cell == 0.0 for cell in row) for row in r)


# --- transpose ------------------------------------------------------------------


def test_transpose_swaps_entries(pool):
    rows = [[1, 2, 3], [4, 5, 6]]
    M = ak.Matrix([[pool.integer(v) for v in row] for row in rows])
    T = M.transpose()
    for i in range(2):
        for j in range(3):
            assert ak.eval_expr(T.get(j, i), {}) == ak.eval_expr(M.get(i, j), {})


def test_transpose_of_transpose_is_original(pool):
    rows = [[1, 2], [3, 4], [5, 6]]
    M = ak.Matrix([[pool.integer(v) for v in row] for row in rows])
    TT = M.transpose().transpose()
    assert to_floats(TT) == to_floats(M)


# --- addition / subtraction (the only arithmetic operators available) -------


def test_addition_elementwise(pool):
    A = ak.Matrix([[pool.integer(1), pool.integer(2)], [pool.integer(3), pool.integer(4)]])
    B = ak.Matrix([[pool.integer(5), pool.integer(6)], [pool.integer(7), pool.integer(8)]])
    assert to_floats(A + B) == [[6.0, 8.0], [10.0, 12.0]]


def test_subtraction_elementwise(pool):
    A = ak.Matrix([[pool.integer(5), pool.integer(6)], [pool.integer(7), pool.integer(8)]])
    B = ak.Matrix([[pool.integer(1), pool.integer(2)], [pool.integer(3), pool.integer(4)]])
    assert to_floats(A - B) == [[4.0, 4.0], [4.0, 4.0]]


def test_subtraction_from_self_is_zero(pool):
    A = ak.Matrix([[pool.integer(3), pool.integer(1)], [pool.integer(4), pool.integer(1)]])
    assert to_floats(A - A) == [[0.0, 0.0], [0.0, 0.0]]


# --- inverse (verified via independent Python matmul, not alkahest's) ------


def test_inverse_2x2_times_original_is_identity(pool):
    A = ak.Matrix([[pool.integer(2), pool.integer(1)], [pool.integer(1), pool.integer(1)]])
    Ainv = A.inverse()
    assert_matrix_close(_matmul(to_floats(A), to_floats(Ainv)), identity(2))


def test_inverse_3x3_times_original_is_identity(pool):
    rows = [[1, 2, 0], [0, 1, 3], [4, 0, 1]]
    A = ak.Matrix([[pool.integer(v) for v in row] for row in rows])
    Ainv = A.inverse()
    assert_matrix_close(_matmul(to_floats(A), to_floats(Ainv)), identity(3))


def test_inverse_matches_solve_for_linear_system(pool):
    """Ax=b solved via A^-1 b should match alkahest.solve on the same system."""
    x1, x2 = pool.symbol("x1"), pool.symbol("x2")
    A = ak.Matrix([[pool.integer(2), pool.integer(1)], [pool.integer(1), pool.integer(3)]])
    b = [5.0, 10.0]
    x_via_inverse = _matvec(to_floats(A.inverse()), b)

    eqs = [2 * x1 + x2 - pool.integer(5), x1 + 3 * x2 - pool.integer(10)]
    sols = ak.solve(eqs, [x1, x2])
    assert len(sols) == 1
    x_via_solve = [ak.eval_expr(sols[0][x1], {}), ak.eval_expr(sols[0][x2], {})]
    assert x_via_inverse == pytest.approx(x_via_solve)


# --- nullspace ------------------------------------------------------------------


def test_nullspace_vector_maps_to_zero(pool):
    A = ak.Matrix([[pool.integer(1), pool.integer(2)], [pool.integer(2), pool.integer(4)]])
    basis = A.nullspace()
    assert len(basis) == 1
    v = [row[0] for row in to_floats(basis[0])]  # basis[0] is a column-shaped Matrix
    result = _matvec(to_floats(A), v)
    assert result == pytest.approx([0.0, 0.0], abs=1e-9)


@pytest.mark.xfail(
    strict=True,
    reason="new finding: Matrix.nullspace() returns a spurious nonzero vector for "
    "every full-rank (invertible) 2x2 matrix tried (identity, diag(2,3), [[2,1],[1,1]], "
    "[[1,2],[0,1]], ...) — e.g. nullspace(I2) == [[0,1]], but I2 @ [0,1] == [0,1] != 0, "
    "so the returned vector isn't actually in the nullspace. Silently wrong, not a crash. "
    "3x3 full-rank matrices correctly return []; the bug appears specific to the 2x2 path.",
)
def test_nullspace_trivial_for_full_rank(pool):
    A = ak.Matrix([[pool.integer(1), pool.integer(0)], [pool.integer(0), pool.integer(1)]])
    assert A.nullspace() == []


# --- LU factorization (verified via independent Python matmul) -------------


def test_lu_reconstructs_original(pool):
    """L @ U reconstructs A *with rows permuted per `perm`* (partial pivoting) —
    not necessarily A itself."""
    rows = [[4, 3], [6, 3]]
    A = ak.Matrix([[pool.integer(v) for v in row] for row in rows])
    L, U, perm = A.lu()
    reconstructed = _matmul(to_floats(L), to_floats(U))
    expected = [[float(v) for v in rows[perm[i]]] for i in range(len(rows))]
    assert_matrix_close(reconstructed, expected)


# --- eigenvalues: verified as roots of the characteristic polynomial -------


def test_eigenvals_are_roots_of_characteristic_polynomial(pool):
    A = ak.Matrix([[pool.integer(2), pool.integer(0)], [pool.integer(0), pool.integer(3)]])
    eigenvals = A.eigenvals()
    assert {round(ak.eval_expr(ev, {}), 9) for ev in eigenvals} == {2.0, 3.0}
    cp_expr, lam = A.characteristic_polynomial_lambda_minus_m()
    for ev in eigenvals:
        val = ak.eval_expr(cp_expr, {lam: ak.eval_expr(ev, {})})
        assert math.isclose(val, 0.0, abs_tol=1e-6)


def test_eigenvals_symmetric_matrix_sum_equals_trace(pool):
    """Sum of eigenvalues (with multiplicity) equals the trace — a standard
    first-course fact, checked independently of alkahest's own trace()."""
    A = ak.Matrix([[pool.integer(2), pool.integer(1)], [pool.integer(1), pool.integer(2)]])
    eigenvals = A.eigenvals()
    total = sum(ak.eval_expr(ev, {}) * mult for ev, mult in eigenvals.items())
    assert math.isclose(total, 4.0, abs_tol=1e-9)


def test_eigenvals_triangular_matrix_are_diagonal_entries(pool):
    """Eigenvalues of a triangular matrix are exactly its diagonal entries."""
    rows = [[2, 5, 7], [0, 3, 9], [0, 0, 5]]
    A = ak.Matrix([[pool.integer(v) for v in row] for row in rows])
    eigenvals = {round(ak.eval_expr(ev, {}), 9) for ev in A.eigenvals()}
    assert eigenvals == {2.0, 3.0, 5.0}


# --- rational-root-theorem-adjacent: char poly degree matches matrix size --


@pytest.mark.xfail(
    strict=True,
    reason="new finding (parallel to B5): Matrix.eigenvals() raises PyEigenError "
    "'irreducible characteristic factor of degree 3; only degrees 1-2 are supported' "
    "for a plain 3x3 matrix whose characteristic polynomial happens to be an "
    "irreducible cubic — eigenvalues of 3x3+ matrices are a first-course topic",
)
def test_characteristic_polynomial_degree_matches_size(pool):
    rows = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    A = ak.Matrix([[pool.integer(v) for v in row] for row in rows])
    assert sum(A.eigenvals().values()) == 3
