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


def _expm_floats(rows):
    """``exp(M)`` as floats, evaluated in ℂ and reduced to ``Re + |Im|``.

    Complex mode because a matrix with a complex spectrum comes back written
    over ``sqrt(-1)``; ``Re + |Im|`` so a spurious imaginary part is not
    rounded into agreement.
    """
    pool = alkahest.ExprPool()
    m = alkahest.Matrix([[pool.integer(v) for v in row] for row in rows])
    out = []
    for row in m.matrix_exp().to_list():
        vals = []
        for entry in row:
            value = complex(alkahest.evaluate(entry, {}, mode="complex").value)
            vals.append(value.real + abs(value.imag))
        out.append(vals)
    return out


def _assert_close(got, want, tol=1e-9):
    assert len(got) == len(want)
    for grow, wrow in zip(got, want):
        assert len(grow) == len(wrow)
        for g, w in zip(grow, wrow):
            assert abs(g - w) < tol, f"{got} != {want}"


# Expected values are `sympy.Matrix(M).exp()`.  Before 3.10.1 every defective
# (non-diagonalizable) matrix came back wrong with no exception and no flag:
# the Jordan block formula read the nilpotent power `N^k` as `λ^k`, so a
# nilpotent block lost its off-diagonal entirely and every other defective
# block was scaled by `λ^k`.
_E = 2.718281828459045
_E2 = _E * _E


def test_matrix_exp_nilpotent_is_not_the_identity():
    # sympy: exp([[0,1],[0,0]]) == [[1,1],[0,1]].  Returned the identity.
    _assert_close(_expm_floats([[0, 1], [0, 0]]), [[1.0, 1.0], [0.0, 1.0]])


def test_matrix_exp_defective_off_diagonal_is_not_doubled():
    # sympy: exp([[2,1],[0,2]]) == [[e², e²],[0, e²]].  The off-diagonal was 2e².
    _assert_close(_expm_floats([[2, 1], [0, 2]]), [[_E2, _E2], [0.0, _E2]])


def test_matrix_exp_three_by_three_jordan_block():
    # sympy: exp([[2,1,0],[0,2,1],[0,0,2]]) == [[e²,e²,e²/2],[0,e²,e²],[0,0,e²]].
    _assert_close(
        _expm_floats([[2, 1, 0], [0, 2, 1], [0, 0, 2]]),
        [[_E2, _E2, _E2 / 2], [0.0, _E2, _E2], [0.0, 0.0, _E2]],
    )


def test_matrix_exp_two_jordan_blocks_for_one_eigenvalue():
    # sympy: exp(J₂(3) ⊕ J₂(3)) is block diagonal with [[e³,e³],[0,e³]].  This
    # refused before 3.10.1: both Jordan chains came out of the same kernel, so
    # the similarity transform was singular.
    e3 = _E**3
    _assert_close(
        _expm_floats([[3, 1, 0, 0], [0, 3, 0, 0], [0, 0, 3, 1], [0, 0, 0, 3]]),
        [
            [e3, e3, 0.0, 0.0],
            [0.0, e3, 0.0, 0.0],
            [0.0, 0.0, e3, e3],
            [0.0, 0.0, 0.0, e3],
        ],
    )


def test_matrix_exp_defective_without_a_zero_off_diagonal():
    # (λ − 2)² with a rank-1 A − 2I; sympy: exp([[1,1],[-1,3]]) == [[0,e²],[-e²,2e²]].
    _assert_close(_expm_floats([[1, 1], [-1, 3]]), [[0.0, _E2], [-_E2, 2 * _E2]])


def test_matrix_exp_rotation_stays_real():
    # sympy: exp([[0,1],[-1,0]]) == [[cos 1, sin 1],[-sin 1, cos 1]].
    import math

    c, s = math.cos(1.0), math.sin(1.0)
    _assert_close(_expm_floats([[0, 1], [-1, 0]]), [[c, s], [-s, c]])


def test_matrix_exp_diagonalizable_control_does_not_regress():
    # The route that was already right: sympy gives [[51.968956198705, ...], ...].
    _assert_close(
        _expm_floats([[1, 2], [3, 4]]),
        [[51.968956198705, 74.73656456700321], [112.10484685050481, 164.07380304920983]],
        tol=1e-7,
    )


def test_matrix_exp_reports_the_eigenvalue_gap_it_divided_by():
    """``exp([[a,1],[0,b]])`` is ``e^A`` only for ``a != b``, and says so."""
    pool = alkahest.ExprPool()
    a, b = pool.symbol("a"), pool.symbol("b")
    m = alkahest.Matrix([[a, pool.integer(1)], [pool.integer(0), b]])
    m.matrix_exp()
    conds = alkahest.matrix_exp_side_conditions()
    assert len(conds) == 1, conds
    assert "≠ 0" in conds[0]


def test_matrix_exp_reports_nothing_when_every_gap_is_settled():
    """The control: a rational spectrum needs no hypothesis at all."""
    pool = alkahest.ExprPool()
    m = alkahest.Matrix([[pool.integer(2), pool.integer(1)], [pool.integer(0), pool.integer(2)]])
    m.matrix_exp()
    assert alkahest.matrix_exp_side_conditions() == []


def test_jordan_form_transform_is_a_basis():
    """``M = P·J·P⁻¹`` is a claim about ``P``; a rank-deficient one is not a similarity."""
    pool = alkahest.ExprPool()
    rows = [[3, 1, 0, 0], [0, 3, 0, 0], [0, 0, 3, 1], [0, 0, 0, 3]]
    m = alkahest.Matrix([[pool.integer(v) for v in row] for row in rows])
    # Refusing is acceptable — both chains of this matrix are drawn from the
    # same kernel, so a basis may genuinely not be found. Returning a singular
    # P labelled a similarity transform is not.
    try:
        p, _j = m.jordan_form()
    except alkahest.LinearAlgebraError:
        pytest.skip("jordan_form declined this matrix, which is the honest outcome")
    assert p.rank() == 4


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


# ---------------------------------------------------------------------------
# Undecidable zero tests — the refusal has to arrive with its own code
# ---------------------------------------------------------------------------
#
# Both refusals travel on an existing error variant (`UnsupportedField`,
# `SingularMatrix`), because the Rust enums are public and exhaustive and a new
# variant is a major semver break. The specific cause rides alongside and the
# bindings recover it, so these tests are what keeps that recovery honest: if
# the out-of-band channel is ever dropped, the refusal silently degrades to the
# carrier's code and only this asserts otherwise.


def _undecidable(pool):
    """A 1×1 quantity nothing can decide: `mystery` has no rule, no numeric
    kernel and no interval kernel, so it can be neither normalised to zero nor
    rigorously enclosed away from it."""
    return pool.func("mystery", [pool.symbol("a")])


def test_undecidable_pivot_refuses_with_linalg_code():
    """A pivot that can be proven neither zero nor non-zero raises E-LINALG-010."""
    pool = alkahest.ExprPool()
    zero = pool.integer(0)
    m = alkahest.Matrix([[_undecidable(pool), zero], [zero, zero]])
    with pytest.raises(alkahest.LinearAlgebraError) as exc_info:
        m.rank()
    assert exc_info.value.code == "E-LINALG-010"
    assert "mystery" in str(exc_info.value)


def test_undecidable_determinant_refuses_with_matrix_code():
    """An inverse whose determinant cannot be decided raises E-MAT-004.

    Not an inverse divided by a determinant that might be zero, which nothing
    downstream could tell apart from a real one.
    """
    pool = alkahest.ExprPool()
    zero, one = pool.integer(0), pool.integer(1)
    m = alkahest.Matrix([[_undecidable(pool), zero], [zero, one]])
    with pytest.raises(alkahest.MatrixError) as exc_info:
        m.inverse()
    assert exc_info.value.code == "E-MAT-004"


def test_proven_singular_keeps_its_own_code_after_a_refusal():
    """A genuinely singular matrix stays E-MAT-003 even right after a refusal.

    The refusal is recorded out of band, so the failure mode this guards is a
    stale record being picked up by the next unrelated error on the thread.
    """
    pool = alkahest.ExprPool()
    zero = pool.integer(0)
    with pytest.raises(alkahest.LinearAlgebraError):
        alkahest.Matrix([[_undecidable(pool), zero], [zero, zero]]).rank()
    singular = _int_matrix(pool, [[1, 2], [2, 4]])
    with pytest.raises(alkahest.MatrixError) as exc_info:
        singular.inverse()
    assert exc_info.value.code == "E-MAT-003"


@pytest.mark.parametrize("op", ["nullspace", "eigenvects", "jordan_form"])
def test_undecidable_entry_keeps_its_code_through_the_kernel_routines(op):
    """`nullspace`, `eigenvects` and `jordan_form` share one elimination.

    All three used to flatten an undecidable entry into their own generic
    "kernel failed" verdict — `E-LINALG-002` ("could not compute nullspace
    basis") or `E-EIGEN-006` — because the routine they share returned an error
    with no payload, so the reason died at that boundary. A caller could not
    tell "this matrix is hard for the kernel routine" (nothing to be done) from
    "one entry's vanishing is undecidable" (substitute concrete parameters and
    it works).
    """
    pool = alkahest.ExprPool()
    zero = pool.integer(0)
    m = alkahest.Matrix([[_undecidable(pool), zero], [zero, zero]])
    with pytest.raises(alkahest.AlkahestError) as exc_info:
        getattr(m, op)()
    assert exc_info.value.code == "E-LINALG-010"
    assert "mystery" in str(exc_info.value)
    assert exc_info.value.remediation


def test_a_computable_nullspace_is_still_computed():
    """The control: refusing everything would pass the test above and be
    useless. A rank-1 symbolic matrix must still give a 1-dimensional kernel."""
    pool = alkahest.ExprPool()
    a = pool.symbol("a")
    exp_a = alkahest.exp(a)
    m = alkahest.Matrix([[pool.integer(1), exp_a], [exp_a, exp_a * exp_a]])
    assert len(m.nullspace()) == 1


# ---------------------------------------------------------------------------
# 3.10.1 linear-algebra audit
# ---------------------------------------------------------------------------


def _int_rows(pool, rows):
    return alkahest.Matrix([[pool.integer(v) for v in row] for row in rows])


def test_diagonalize_answers_an_irrational_spectrum():
    """`[[1,2],[3,4]]` has distinct eigenvalues, so it is diagonalizable.

    It was refused with `E-EIGEN-005`, *"matrix is not diagonalizable"* — a
    statement that is false about this matrix, and one the library's own
    `jordan_form` contradicted by returning a diagonal `J` for it. The check
    compared two normalised forms structurally, and the `M·v` side carries
    `√33·√33` where the `λ·v` side carries `33`.

    Scored on the identity, not on the shape of `P`: `M·P = P·D`.
    """
    pool = alkahest.ExprPool()
    m = _int_rows(pool, [[1, 2], [3, 4]])
    p, d = m.diagonalize()
    lhs = m.multiply(p).to_list()
    rhs = p.multiply(d).to_list()
    for i in range(2):
        for j in range(2):
            got = float(alkahest.eval_expr(lhs[i][j], {}))
            want = float(alkahest.eval_expr(rhs[i][j], {}))
            assert abs(got - want) < 1e-9
    # `D` is diagonal and its trace is the trace of `M`, which is 5.
    entries = d.to_list()
    assert float(alkahest.eval_expr(entries[0][1], {})) == 0.0
    assert float(alkahest.eval_expr(entries[1][0], {})) == 0.0
    trace = sum(float(alkahest.eval_expr(entries[i][i], {})) for i in range(2))
    assert abs(trace - 5.0) < 1e-9


def test_diagonalize_still_refuses_a_defective_matrix():
    """The control. `[[2,1],[0,2]]` has one eigenvalue of algebraic
    multiplicity 2 and a one-dimensional eigenspace, so no diagonalization
    exists; widening the verification must not have widened the answer set."""
    pool = alkahest.ExprPool()
    m = _int_rows(pool, [[2, 1], [0, 2]])
    with pytest.raises(alkahest.AlkahestError) as exc_info:
        m.diagonalize()
    assert exc_info.value.code == "E-EIGEN-005"


def test_cholesky_answers_an_irrational_pivot():
    """`2I` is symmetric positive definite and its factor is `√2·I`.

    The rational path required every pivot to be a perfect rational square and
    reported `E-LINALG-003`, *"matrix is not symmetric positive definite"*,
    about a matrix that is both.
    """
    pool = alkahest.ExprPool()
    lower = _int_rows(pool, [[2, 0], [0, 2]]).cholesky()
    entries = lower.to_list()
    assert abs(float(alkahest.eval_expr(entries[0][0], {})) - 2**0.5) < 1e-12
    assert float(alkahest.eval_expr(entries[0][1], {})) == 0.0


def test_cholesky_refuses_a_non_symmetric_matrix():
    """`L·Lᵀ` is symmetric for every `L`, so `[[1,5],[0,1]]` has no factor.

    Only the lower triangle was read, so the upper one was discarded and the
    identity came back — a factorisation of `I`, not of the input.
    """
    pool = alkahest.ExprPool()
    with pytest.raises(alkahest.AlkahestError) as exc_info:
        _int_rows(pool, [[1, 5], [0, 1]]).cholesky()
    assert exc_info.value.code == "E-LINALG-003"


def test_lu_permutes_the_multipliers_with_their_rows():
    """`P·A = L·U`, on a 3×3 that pivots twice.

    The multipliers already stored in `L` belong to the rows a pivot swap
    moves. A 2×2 cannot show it: its only swap is at `k = 0`, where `L` has no
    computed column yet to be left behind.
    """
    pool = alkahest.ExprPool()
    rows = [[2, -1, 4], [-1, 1, -1], [3, -4, 0]]
    lower, upper, perm = _int_rows(pool, rows).lu()
    product = lower.multiply(upper).to_list()
    for i, source in enumerate(perm):
        for j in range(3):
            assert abs(float(alkahest.eval_expr(product[i][j], {})) - rows[source][j]) < 1e-9


def test_row_space_basis_comes_from_the_echelon_form():
    """`[[0,0],[1,0]]` has row space `span{(1,0)}`.

    Elimination swaps the rows and marks echelon row 0 as the pivot row;
    reading `m.row(0)` handed back `(0,0)`, the zero vector offered as a basis
    of a one-dimensional space.
    """
    pool = alkahest.ExprPool()
    basis = _int_rows(pool, [[0, 0], [1, 0]]).row_space()
    assert len(basis) == 1
    row = basis[0].to_list()[0]
    assert float(alkahest.eval_expr(row[0], {})) == 1.0
    assert float(alkahest.eval_expr(row[1], {})) == 0.0


def test_minimal_polynomial_of_the_zero_matrix_is_lambda():
    """`p(M) = 0` for `p = λ`, and nothing of lower degree does.

    A vanishing constant term used to leave the matrix power un-advanced, so
    `p(M)` was evaluated as `(p/λ)(M)`: `λ` was rejected and `λ²` returned. A
    zero constant term is exactly `0 ∈ spec(M)`, so every singular matrix was
    exposed. Scored at a point — `λ` gives 3 and `λ²` gives 9.
    """
    import re

    pool = alkahest.ExprPool()
    poly = _int_rows(pool, [[0, 0], [0, 0]]).minimal_polynomial()
    names = set(re.findall(r"__eigen_lambda_\d+", str(poly)))
    assert len(names) == 1
    lam = pool.symbol(names.pop(), "complex")
    assert float(alkahest.eval_expr(poly, {lam: 3.0})) == 3.0


def test_rational_canonical_form_reconstructs_the_matrix():
    """`M·P = P·C` with `P` invertible, and `det C = det M`.

    The companion block wrote its coefficients along the last *row* instead of
    the last *column*, which for `d ≥ 2` also overwrote a subdiagonal 1. On
    `diag(1,2)` that gave `C = [[0,0],[−2,3]]`, determinant 0 against
    `det M = 2` — not similar to `M`, and not even of the same rank.
    """
    pool = alkahest.ExprPool()
    m = _int_rows(pool, [[1, 0], [0, 2]])
    p, c = m.rational_canonical_form()
    lhs = m.multiply(p).to_list()
    rhs = p.multiply(c).to_list()
    for i in range(2):
        for j in range(2):
            got = float(alkahest.eval_expr(lhs[i][j], {}))
            want = float(alkahest.eval_expr(rhs[i][j], {}))
            assert abs(got - want) < 1e-9
    assert p.rank() == 2
    assert abs(float(alkahest.eval_expr(c.det(), {})) - 2.0) < 1e-9


def test_a_dense_symbolic_matrix_is_inverted_and_says_what_it_assumed():
    """A 4×4 of 16 distinct symbols inverts, and reports `det ≠ 0`.

    `det` of a matrix of `n²` distinct symbols is a sum of `n!` distinct
    monomials and is not the zero function, so `adj/det` is the inverse. Sizes
    from 5×5 up refused with `E-MAT-004` only because the non-vanishing probe
    would not bind more than 16 symbols and sampled them along an arithmetic
    progression, which makes the probe matrix rank 2.

    Verified by substitution into ℚ rather than by symbolic cancellation. The
    sample values are deliberately unstructured: an affine function of `(i, j)`
    makes the rows an arithmetic progression, so `det` vanishes there and
    `A⁻¹·A` is `0/0` rather than `I`.
    """
    pool = alkahest.ExprPool()
    n = 4
    syms = [[pool.symbol(f"__pt_{i}_{j}") for j in range(n)] for i in range(n)]
    m = alkahest.Matrix(syms)
    product = m.inverse().multiply(m).to_list()
    conditions = alkahest.matrix_inverse_side_conditions()
    assert len(conditions) == 1, conditions
    assert "≠ 0" in conditions[0]

    sample = [1.7, -2.3, 0.9, 3.1, -0.6, 2.2, 4.7, -1.1, 0.4, 5.3, -3.7, 1.3, 2.9, -0.8, 6.1, 0.3]
    env = {syms[i][j]: sample[(i * n + j) % len(sample)] for i in range(n) for j in range(n)}
    for i in range(n):
        for j in range(n):
            got = float(alkahest.eval_expr(product[i][j], env))
            assert abs(got - (1.0 if i == j else 0.0)) < 1e-9


def test_a_constant_determinant_is_discharged_not_reported():
    """The control for the channel above.

    `det [[1,2],[3,4]] = −2`, a non-zero constant: the hypothesis is settled,
    not assumed, and reporting one anyway is the noise that makes a caller stop
    reading the channel.
    """
    pool = alkahest.ExprPool()
    _int_rows(pool, [[1, 2], [3, 4]]).inverse()
    assert alkahest.matrix_inverse_side_conditions() == []


def test_the_inverse_channel_is_cleared_by_a_refusal():
    """A refused inverse must not leave a hypothesis for the next call."""
    pool = alkahest.ExprPool()
    a = pool.symbol("__pt_q")
    b = pool.symbol("__pt_r")
    two = pool.integer(2)
    dependent = alkahest.Matrix([[a, b], [two * a, two * b]])
    with pytest.raises(alkahest.MatrixError):
        dependent.inverse()
    assert alkahest.matrix_inverse_side_conditions() == []
