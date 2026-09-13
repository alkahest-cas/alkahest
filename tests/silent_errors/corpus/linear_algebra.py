"""Silent-error cases for linear algebra.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Callable

import alkahest as ak
from contracts import Case, Raises, RefusesOr, Returns

from ._shared import _A, _B, POOL, X, _int, _num, _survives_a_panic


def _matrix(rows: list[list[int]]) -> ak.Matrix:
    return ak.Matrix([[_int(v) for v in row] for row in rows])


SINGULAR_2X2 = [[1, 2], [2, 4]]
SINGULAR_3X3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
ZERO_2X2 = [[0, 0], [0, 0]]
NON_SQUARE = [[1, 2, 3], [4, 5, 6]]
#: det = (2^30+1)(2^30-1) - 2^60 = -1 exactly; float64 evaluation cancels to 0.0.
CANCELLING_2X2 = [[2**30 + 1, 2**30], [2**30, 2**30 - 1]]
_EXP_A = ak.exp(_A)
#: Row 2 is exactly ``exp(a)`` × row 1, so the rank is 1 — but only if
#: ``exp(a)·exp(a) − exp(a+a)`` is recognised as zero.
EXP_DEPENDENT_ROWS = ak.Matrix(
    [
        [_int(1), _EXP_A, _EXP_A],
        [_EXP_A, _EXP_A * _EXP_A, ak.exp(_A + _A)],
    ]
)
#: The control: identical except for the last entry, which breaks the
#: proportionality, so the rank really is 2.
EXP_INDEPENDENT_ROWS = ak.Matrix(
    [
        [_int(1), _EXP_A, _EXP_A],
        [_EXP_A, _EXP_A * _EXP_A, _EXP_A],
    ]
)
#: ``mystery`` has no differentiation rule, no numeric kernel and no interval
#: kernel, so ``mystery(a)`` can be neither normalised to zero nor rigorously
#: enclosed away from it.  Whether it is the zero function is not knowable here,
#: and column 1 has no other candidate.
UNDECIDABLE_PIVOT = ak.Matrix(
    [
        [POOL.func("mystery", [_A]), _int(0)],
        [_int(0), _int(0)],
    ]
)
#: ``det = mystery(a)``, so whether this matrix is invertible is exactly as
#: undecidable as whether ``mystery`` is the zero function.  ``rank()`` refuses
#: it; ``nullspace()`` used to return the 1-dimensional basis ``(-1, mystery(a))``
#: — the answer that is right only when ``det = 0``.
UNDECIDABLE_DETERMINANT = ak.Matrix(
    [
        [POOL.func("mystery", [_A]), _int(1)],
        [_int(0), _int(1)],
    ]
)
#: ``det = x``: generically non-zero, so the kernel is trivial.  This needs no
#: uninterpreted function at all — it is an ordinary symbolic matrix, and the
#: cheapest possible trigger for the same defect.
GENERICALLY_INVERTIBLE = ak.Matrix([[X, _int(0)], [_int(0), _int(1)]])
#: ``det = x·x − x·x = 0`` identically: genuinely rank 1, so the kernel really is
#: 1-dimensional.  The control that stops the gate being passed by refusing every
#: symbolic matrix.
GENUINELY_RANK_ONE = ak.Matrix([[X, X], [X, X]])


def _nullspace_dim(m: ak.Matrix) -> Callable[[], int]:
    """Answer = the dimension of ``m.nullspace()``."""
    return lambda: len(m.nullspace())


def _kernel_residual(m: ak.Matrix, at: float = 0.7) -> Callable[[], float]:
    """Answer = max |M·v| over the returned basis, sampled at ``x = at``.

    A basis vector that is not annihilated is the whole failure: the dimension
    can be right while the vector is wrong, so scoring the dimension alone would
    miss it.  Sampled numerically rather than compared structurally so the case
    does not depend on the form the entries come back in.
    """

    def op() -> float:
        worst = 0.0
        for v in m.nullspace():
            for row in (m @ v).to_list():
                for entry in row:
                    worst = max(worst, abs(float(ak.eval_expr(entry, {X: at, _A: at}))))
        return worst

    return op


#: ``exp(A)`` traps.  Every one of these is **defective** — the characteristic
#: polynomial has a repeated root whose eigenspace is too small — which is the
#: only case ``e^A`` needs anything beyond a diagonalisation, and the case that
#: was wrong in 3.10.0.
NILPOTENT_2X2 = [[0, 1], [0, 0]]
DEFECTIVE_2X2 = [[2, 1], [0, 2]]
NILPOTENT_3X3 = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
JORDAN_3X3 = [[2, 1, 0], [0, 2, 1], [0, 0, 2]]
TWO_JORDAN_BLOCKS = [[3, 1, 0, 0], [0, 3, 0, 0], [0, 0, 3, 1], [0, 0, 0, 3]]
#: Defective with nothing on the surface to say so: no zero off-diagonal, and
#: the repeated eigenvalue 2 only appears after the characteristic polynomial
#: is factored as (λ − 2)².
DEFECTIVE_DENSE_2X2 = [[1, 1], [-1, 3]]
ROTATION_2X2 = [[0, 1], [-1, 0]]
SYMBOLIC_GAP_2X2 = ak.Matrix([[_A, _int(1)], [_int(0), _B]])


def _exp_scalar(entry: Any, bindings: dict[Any, float]) -> float:
    """Reduce one ``exp(M)`` entry to ``Re + |Im|``.

    Evaluated in **complex** mode because a matrix with a complex spectrum comes
    back written over ``sqrt(-1)`` — ``exp([[0,1],[-1,0]])[0][1]`` is
    ``(e^{i} − e^{−i})/(2i)``, which is exactly ``sin 1`` and which the real
    evaluator declines rather than mis-evaluating.

    ``Re + |Im|`` rather than ``Re`` so that an answer which is right on the
    real axis and wrong off it is still scored wrong: every matrix in these
    cases is real, so every entry of ``e^M`` is real and a correct answer has
    ``Im = 0`` exactly.
    """
    result = ak.evaluate(entry, bindings, mode="complex")
    if result.value is None:
        # ``evaluate`` reports a decline in the result rather than raising;
        # re-raise it as the ``ValueError`` the gate reads as a weak refusal, so
        # "could not be evaluated" is not scored as a corpus bug.
        raise ValueError(f"{result.status}: {result.reason}")
    value = complex(result.value)
    return value.real + abs(value.imag)


def _exp_entry(rows: list[list[int]], i: int, j: int) -> Callable[[], float]:
    """Answer = entry ``(i, j)`` of ``exp(M)``, as a float.

    One entry rather than the whole matrix because that is where the failure
    lives: every *diagonal* entry of a defective ``e^A`` was already right in
    3.10.0, so a case scored on the diagonal would have passed throughout.
    """
    return lambda: _exp_scalar(_matrix(rows).matrix_exp().to_list()[i][j], {})


def _exp_entry_at(m: ak.Matrix, i: int, j: int, **at: float) -> Callable[[], float]:
    """Answer = entry ``(i, j)`` of ``exp(m)`` evaluated at the given symbols."""
    return lambda: _exp_scalar(
        m.matrix_exp().to_list()[i][j], {POOL.symbol(k): v for k, v in at.items()}
    )


def _jordan_p_rank(rows: list[list[int]]) -> Callable[[], int]:
    """Answer = the rank of the ``P`` returned by ``jordan_form``.

    ``M = P·J·P⁻¹`` is a claim about ``P`` being a *basis*.  A rank-deficient
    ``P`` makes the identity false and ``P⁻¹`` non-existent, and neither matrix
    looks wrong on inspection — so the rank is the thing to score.
    """

    def op() -> int:
        p, _j = _matrix(rows).jordan_form()
        return p.rank()

    return op


def _rref_zero_rows(m: ak.Matrix, at: float = 0.7) -> Callable[[], int]:
    """Answer = how many rows of ``m.rref()`` vanish, sampled at ``a = at``.

    Scored numerically rather than structurally so the case is immune to the
    form the entries come back in; what it pins down is the only thing that
    matters — a row of an rref is either identically zero or it is not.  For a
    rank-deficient matrix the missing zero row reappears as a spurious pivot,
    which for an augmented system reads as ``0 = 1``: the textbook signature of
    an inconsistent system, and a false "no solution" verdict for a search loop.
    """

    def op() -> int:
        count = 0
        for row in m.rref().to_list():
            if all(abs(float(ak.eval_expr(entry, {_A: at}))) < 1e-12 for entry in row):
                count += 1
        return count

    return op


def _lll_rows_stay_in_the_lattice(
    rows: list[list[int]], generator: list[int]
) -> Callable[[], bool]:
    """Answer = does LLL return a basis of the *same* lattice ``ℤ·generator``?

    Every returned row must be an integer multiple of *generator* (nothing left
    the lattice), the generator itself must still be reachable (nothing was
    lost), and the row count must be preserved.  Exact integer arithmetic, no
    reference implementation.
    """

    def op() -> bool:
        reduced = ak.lattice.lll_reduce_rows(rows)
        if len(reduced) != len(rows):
            return False
        multiples = []
        for row in reduced:
            ratios = {Fraction(v, g) for v, g in zip(row, generator) if g != 0}
            leftover = any(v != 0 for v, g in zip(row, generator) if g == 0)
            if leftover or len(ratios) != 1:
                return False
            (r,) = ratios
            if r.denominator != 1:
                return False
            multiples.append(abs(r.numerator))
        return 1 in multiples

    return _survives_a_panic(op)


LU_PIVOTING_3X3 = [[2, -1, 4], [-1, 1, -1], [3, -4, 0]]
QR_2X2 = [[1, 2], [3, 4]]
TWO_I = [[2, 0], [0, 2]]
RANK_ONE_2X2 = [[-4, -4], [-2, -2]]
SPD_IRRATIONAL_PIVOT = [[4, 2], [2, 3]]
NON_SYMMETRIC_2X2 = [[1, 5], [0, 1]]
INDEFINITE_SYMMETRIC = [[1, 2], [2, 1]]
CARDANO_COMPANION = [[0, 0, -1], [1, 0, -2], [0, 1, -3]]
DEFECTIVE_BLOCK_2X2 = [[2, 1], [0, 2]]
DIAG_1_2 = [[1, 0], [0, 2]]
TWO_NILPOTENT_BLOCKS = [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
ZERO_LEADING_ROW = [[0, 0], [1, 0]]
FIBONACCI_2X2 = [[1, 1], [1, 0]]
_INVERSE_SAMPLE = (
    1.7,
    -2.3,
    0.9,
    3.1,
    -0.6,
    2.2,
    4.7,
    -1.1,
    0.4,
    5.3,
    -3.7,
    1.3,
    2.9,
    -0.8,
    6.1,
    0.3,
)


def _permutation_sign(perm: list[int]) -> int:
    """±1 by counting transpositions, on a copy."""
    p = list(perm)
    sign = 1
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            sign = -sign
    return sign


def _lu_signed_pivot_product(rows: list[list[int]]) -> float:
    _l, u, perm = _matrix(rows).lu()
    entries = u.to_list()
    product = 1.0
    for i in range(u.rows):
        product *= _num(entries[i][i])
    return _permutation_sign(list(perm)) * product


def _lu_reconstruction_residual(rows: list[list[int]]) -> float:
    lower, upper, perm = _matrix(rows).lu()
    product = lower.multiply(upper).to_list()
    worst = 0.0
    for i, source in enumerate(perm):
        for j in range(len(rows[0])):
            worst = max(worst, abs(_num(product[i][j]) - float(rows[source][j])))
    return worst


def _qr_entry(rows: list[list[int]], which: int, i: int, j: int) -> float:
    """`which` is 0 for Q and 1 for R."""
    return _num(_matrix(rows).qr()[which].to_list()[i][j])


def _qr_reconstruction_residual(rows: list[list[int]]) -> float:
    q, r = _matrix(rows).qr()
    product = q.multiply(r).to_list()
    return max(
        abs(_num(product[i][j]) - float(rows[i][j]))
        for i in range(len(rows))
        for j in range(len(rows[0]))
    )


def _qr_orthonormality_residual(rows: list[list[int]]) -> float:
    q, _r = _matrix(rows).qr()
    entries = q.to_list()
    worst = 0.0
    for a in range(q.cols):
        for b in range(q.cols):
            dot = sum(_num(entries[i][a]) * _num(entries[i][b]) for i in range(q.rows))
            worst = max(worst, abs(dot - (1.0 if a == b else 0.0)))
    return worst


def _cholesky_entry(rows: list[list[int]], i: int, j: int) -> float:
    return _num(_matrix(rows).cholesky().to_list()[i][j])


def _cholesky_reconstruction_residual(rows: list[list[int]]) -> float:
    lower = _matrix(rows).cholesky()
    product = lower.multiply(lower.transpose()).to_list()
    return max(
        abs(_num(product[i][j]) - float(rows[i][j]))
        for i in range(len(rows))
        for j in range(len(rows[0]))
    )


def _complex_det(a: list[list[complex]]) -> complex:
    """Determinant by partial-pivoted elimination over Python complex.

    Independent of alkahest and of sympy both, so an eigenvalue check using it
    is not comparing the library against itself.
    """
    a = [row[:] for row in a]
    n = len(a)
    det = 1 + 0j
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) == 0.0:
            return 0j
        if p != c:
            a[p], a[c] = a[c], a[p]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            for k in range(c, n):
                a[r][k] -= f * a[c][k]
    return det


def _complex_eigenvalues(rows: list[list[int]]) -> list[complex]:
    out = []
    for lam in _matrix(rows).eigenvals():
        result = ak.evaluate(lam, {}, mode="complex")
        if result.value is None:
            raise ValueError(f"{result.status}: {result.reason}")
        out.append(complex(result.value))
    return out


def _eigenvalue_charpoly_residual(rows: list[list[int]]) -> float:
    n = len(rows)
    worst = 0.0
    for z in _complex_eigenvalues(rows):
        shifted = [
            [complex(rows[i][j]) - (z if i == j else 0j) for j in range(n)] for i in range(n)
        ]
        worst = max(worst, abs(_complex_det(shifted)))
    return worst


def _smallest_real_eigenvalue(rows: list[list[int]]) -> float:
    real = [z.real for z in _complex_eigenvalues(rows) if abs(z.imag) < 1e-9]
    if not real:
        raise ValueError("no real eigenvalue was returned")
    return min(real)


def _geometric_multiplicity(rows: list[list[int]], lam_value: float) -> int:
    for lam, _alg, vecs in _matrix(rows).eigenvects():
        if abs(_num(lam) - lam_value) < 1e-12:
            return len(vecs)
    raise ValueError(f"{lam_value} is not among the reported eigenvalues")


def _similarity_residual(rows: list[list[int]], pair: tuple) -> float:
    p, f = pair
    lhs = _matrix(rows).multiply(p).to_list()
    rhs = p.multiply(f).to_list()
    n = len(rows)
    return max(abs(_num(lhs[i][j]) - _num(rhs[i][j])) for i in range(n) for j in range(n))


def _diagonalize_residual(rows: list[list[int]]) -> float:
    return _similarity_residual(rows, _matrix(rows).diagonalize())


def _rcf_residual(rows: list[list[int]]) -> float:
    return _similarity_residual(rows, _matrix(rows).rational_canonical_form())


def _rcf_determinant(rows: list[list[int]]) -> float:
    _p, c = _matrix(rows).rational_canonical_form()
    return _num(c.det())


def _minimal_polynomial_at(rows: list[list[int]], point: float) -> float:
    """The minimal polynomial evaluated at `point`.

    The indeterminate is a freshly interned `__eigen_lambda_k`, which the
    binding does not hand back, so it is recovered by name from the rendered
    expression and re-interned with the `Complex` domain it was created with.
    """
    poly = _matrix(rows).minimal_polynomial()
    names = set(re.findall(r"__eigen_lambda_\d+", str(poly)))
    if len(names) != 1:
        raise ValueError(f"expected one indeterminate, found {sorted(names)}")
    lam = POOL.symbol(names.pop(), "complex")
    return float(ak.eval_expr(poly, {lam: float(point)}))


def _characteristic_polynomial_at(rows: list[list[int]], point: float) -> float:
    expr, lam = _matrix(rows).characteristic_polynomial_lambda_minus_m()
    return float(ak.eval_expr(expr, {lam: float(point)}))


def _row_space_basis_rank(rows: list[list[int]]) -> int:
    basis = _matrix(rows).row_space()
    if not basis:
        return 0
    return ak.Matrix([b.to_list()[0] for b in basis]).rank()


def _column_space_basis_rank(rows: list[list[int]]) -> int:
    basis = _matrix(rows).column_space()
    if not basis:
        return 0
    return ak.Matrix([[b.to_list()[i][0] for b in basis] for i in range(len(rows))]).rank()


def _entry_value(rows: list[list[int]], power: int, i: int, j: int) -> float:
    return _num((_matrix(rows) ** power).to_list()[i][j])


def _hadamard_entry(a: list[list[int]], b: list[list[int]], i: int, j: int) -> float:
    return _num(_matrix(a).hadamard(_matrix(b)).to_list()[i][j])


def _symbolic_inverse_residual(n: int) -> float:
    syms = [[POOL.symbol(f"__la_audit_{n}_{i}_{j}") for j in range(n)] for i in range(n)]
    m = ak.Matrix(syms)
    product = m.inverse().multiply(m).to_list()
    env = {
        syms[i][j]: _INVERSE_SAMPLE[(i * n + j) % len(_INVERSE_SAMPLE)]
        for i in range(n)
        for j in range(n)
    }
    return max(
        abs(float(ak.eval_expr(product[i][j], env)) - (1.0 if i == j else 0.0))
        for i in range(n)
        for j in range(n)
    )


def _symbolic_2x2() -> ak.Matrix:
    names = ["__la_cond_a", "__la_cond_b", "__la_cond_c", "__la_cond_d"]
    a, b, c, d = (POOL.symbol(n) for n in names)
    return ak.Matrix([[a, b], [c, d]])


def _inverse_condition_count(build: Callable[[], ak.Matrix]) -> int:
    build().inverse()
    return len(ak.matrix_inverse_side_conditions())


CASES: list[Case] = [
    # Linear algebra on singular and ill-conditioned inputs.
    # -----------------------------------------------------------------------
    Case(
        id="matrix_inverse_singular_2x2",
        subsystem="linear_algebra",
        statement="[[1,2],[2,4]] is singular and has no inverse",
        op=lambda: _matrix(SINGULAR_2X2).inverse().to_list(),
        contract=Raises("E-MAT-003"),
        verified_by="det = 1·4 - 2·2 = 0; row 2 is 2× row 1.",
    ),
    Case(
        id="matrix_inverse_zero_2x2",
        subsystem="linear_algebra",
        statement="the zero matrix has no inverse",
        op=lambda: _matrix(ZERO_2X2).inverse().to_list(),
        contract=Raises("E-MAT-003"),
        verified_by="det = 0·0 - 0·0 = 0; the zero matrix has rank 0.",
    ),
    Case(
        id="matrix_inverse_singular_3x3",
        subsystem="linear_algebra",
        statement="[[1,2,3],[4,5,6],[7,8,9]] is singular and has no inverse",
        op=lambda: _matrix(SINGULAR_3X3).inverse().to_list(),
        contract=Raises("E-MAT-003"),
        verified_by="row3 - row2 = row2 - row1 = (3,3,3), so the rows are linearly dependent "
        "and det = 0. Rank 2, not 3.",
    ),
    Case(
        id="matrix_determinant_non_square",
        subsystem="linear_algebra",
        statement="the determinant of a 2×3 matrix is undefined",
        op=lambda: _num(_matrix(NON_SQUARE).det()),
        contract=Raises("E-MAT-002"),
        verified_by="Determinants are defined only for square matrices.",
    ),
    Case(
        id="matrix_inverse_non_square",
        subsystem="linear_algebra",
        statement="a 2×3 matrix has no inverse",
        op=lambda: _matrix(NON_SQUARE).inverse().to_list(),
        contract=Raises("E-MAT-002"),
        verified_by="Inverses are defined only for square matrices.",
    ),
    Case(
        id="matrix_determinant_singular_is_zero",
        subsystem="linear_algebra",
        statement="det[[1,2],[2,4]] = 0 exactly",
        op=lambda: _num(_matrix(SINGULAR_2X2).det()),
        contract=Returns(0.0),
        verified_by="1·4 - 2·2 = 0.",
    ),
    Case(
        id="matrix_determinant_singular_3x3_is_zero",
        subsystem="linear_algebra",
        statement="det[[1,2,3],[4,5,6],[7,8,9]] = 0 exactly",
        op=lambda: _num(_matrix(SINGULAR_3X3).det()),
        contract=Returns(0.0),
        verified_by="Cofactor expansion: 1(45-48) - 2(36-42) + 3(32-35) = -3 + 12 - 9 = 0.",
    ),
    Case(
        id="matrix_rank_singular",
        subsystem="linear_algebra",
        statement="rank[[1,2],[2,4]] = 1",
        op=lambda: _matrix(SINGULAR_2X2).rank(),
        contract=Returns(1),
        verified_by="One independent row.",
    ),
    Case(
        id="matrix_rank_exp_proportional_rows",
        subsystem="linear_algebra",
        statement="rank[[1, e^a, e^a], [e^a, e^a·e^a, e^(a+a)]] = 1",
        op=lambda: EXP_DEPENDENT_ROWS.rank(),
        contract=Returns(1),
        verified_by="Row 2 = e^a · row 1 entry by entry: e^a·1 = e^a, e^a·e^a is the (2,2) "
        "entry verbatim, and e^a·e^a = e^(a+a) by the exponential functional equation "
        "e^u·e^v = e^(u+v), which is the (2,3) entry. Two proportional rows span a "
        "1-dimensional row space, so the rank is 1 for every value of a.",
    ),
    Case(
        id="matrix_rref_exp_proportional_rows_has_zero_row",
        subsystem="linear_algebra",
        statement="the rref of [[1, e^a, e^a], [e^a, e^a·e^a, e^(a+a)]] has exactly one zero row",
        op=_rref_zero_rows(EXP_DEPENDENT_ROWS),
        contract=Returns(1),
        verified_by="A 2×3 matrix of rank 1 has 2 − 1 = 1 zero row in reduced row echelon "
        "form. The wrong answer here is 0 zero rows, i.e. a second pivot in the last "
        "column — read as an augmented system that is the row 0 = 1 of an inconsistent "
        "one, for a system that is consistent.",
    ),
    Case(
        id="matrix_rank_undecidable_pivot_refuses",
        subsystem="linear_algebra",
        statement="rank[[mystery(a), 0], [0, 0]] cannot be stated — mystery(a) may be the "
        "zero function",
        op=lambda: UNDECIDABLE_PIVOT.rank(),
        contract=RefusesOr(),
        verified_by="The rank is 1 if mystery is not identically zero and 0 if it is. "
        "mystery is an uninterpreted function symbol, so both readings are consistent with "
        "everything alkahest knows and neither number is derivable. Deciding whether an "
        "expression over a transcendental extension vanishes is undecidable in general "
        "(Richardson 1968), so this class cannot be normalised away — the only honest "
        "answer is a refusal. Contract is code-agnostic on purpose: what is being pinned "
        "is that no rank is asserted, not which E-LINALG code says so.",
        note="This is the pair to matrix_rank_exp_proportional_rows: that case is 'prove "
        "zero when it is zero', this one is 'do not claim non-zero when you cannot'. A "
        "library that only did the first would pass that case by pivoting on anything it "
        "failed to reduce, which is the bug that motivated both.",
    ),
    Case(
        id="matrix_nullspace_undecidable_determinant_refuses",
        subsystem="linear_algebra",
        statement="the nullspace of [[mystery(a), 1], [0, 1]] cannot be stated — its "
        "dimension is 0 or 1 depending on whether mystery(a) vanishes",
        op=_nullspace_dim(UNDECIDABLE_DETERMINANT),
        contract=RefusesOr(),
        verified_by="det = mystery(a)·1 − 1·0 = mystery(a). If mystery is not identically "
        "zero the matrix is invertible and the kernel is {0}; if it is, the kernel is "
        "1-dimensional. Both are consistent with everything alkahest knows about an "
        "uninterpreted function symbol, so neither dimension is derivable. The wrong "
        "answer alkahest gave was the basis v = (-1, mystery(a)): multiplying back, "
        "M·v = (mystery(a)·(-1) + 1·mystery(a), 0·(-1) + 1·mystery(a)) = (0, mystery(a)), "
        "which is the zero vector only when mystery(a) = 0 — precisely the thing that was "
        "never established. rank() already refuses this matrix, so the two calls also "
        "contradicted each other.",
        note="Shipped in 3.7: the 2x2 fast path's full-rank gate only recognised a "
        "*literal* non-zero determinant, so any symbolic determinant fell through into "
        "the rank-1 branch. That reads 'cannot prove det != 0' as 'det = 0' — the mirror "
        "of the rref defect that motivated the three-valued zero test, which read "
        "'cannot prove zero' as 'non-zero'.",
    ),
    Case(
        id="matrix_nullspace_generic_determinant_is_trivial",
        subsystem="linear_algebra",
        statement="the nullspace of [[x, 0], [0, 1]] is {0} — dimension 0",
        op=_nullspace_dim(GENERICALLY_INVERTIBLE),
        contract=Returns(0),
        verified_by="det = x·1 − 0·0 = x, which is not the zero function, so the matrix is "
        "invertible for all x != 0 and its kernel is trivial — the same generic-rank "
        "reading rank() uses when it reports 2. The wrong answer was the 1-dimensional "
        "basis v = (0, x), for which M·v = (x·0 + 0·x, 0·0 + 1·x) = (0, x) != 0. Needs no "
        "uninterpreted function: an ordinary symbolic matrix was enough, and rank 2 with "
        "nullity 1 makes 3 for a 2-column matrix, violating rank–nullity across two "
        "public calls.",
    ),
    Case(
        id="matrix_nullspace_singular_symbolic_still_answers",
        subsystem="linear_algebra",
        statement="the nullspace of [[x, x], [x, x]] is 1-dimensional",
        op=_nullspace_dim(GENUINELY_RANK_ONE),
        contract=Returns(1),
        verified_by="det = x·x − x·x = 0 identically, and the matrix is not the zero "
        "matrix, so it has rank 1 and by rank–nullity a 1-dimensional kernel, spanned by "
        "(1, -1). The control for the two cases above: a library that fixed them by "
        "refusing every symbolic matrix would pass both and fail this one.",
    ),
    Case(
        id="matrix_nullspace_basis_is_actually_annihilated",
        subsystem="linear_algebra",
        statement="every returned nullspace basis vector v of [[x, x], [x, x]] satisfies M·v = 0",
        op=_kernel_residual(GENUINELY_RANK_ONE),
        contract=Returns(0.0),
        verified_by="M·(1, -1) = (x − x, x − x) = (0, 0) for every x, so the residual is "
        "exactly zero; sampled at x = 0.7. Scoring the dimension alone would miss the "
        "actual failure mode, which was a basis of the right *size* whose vector was not "
        "in the kernel.",
    ),
    Case(
        id="matrix_rank_exp_independent_rows",
        subsystem="linear_algebra",
        statement="rank[[1, e^a, e^a], [e^a, e^a·e^a, e^a]] = 2",
        op=lambda: EXP_INDEPENDENT_ROWS.rank(),
        contract=Returns(2),
        verified_by="The control for matrix_rank_exp_proportional_rows: only the last entry "
        "differs. Row 2 − e^a · row 1 = (0, 0, e^a − e^a·e^a) = (0, 0, e^a(1 − e^a)), which "
        "is not the zero function (it is e·(1−e) ≠ 0 at a = 1), so the rows are independent "
        "and the rank is 2. A gate made only of 'prove this is zero' cases is passed by a "
        "library that calls everything zero.",
    ),
    Case(
        id="matrix_determinant_catastrophic_cancellation",
        subsystem="linear_algebra",
        statement="det[[2³⁰+1, 2³⁰],[2³⁰, 2³⁰-1]] = -1, not 0",
        op=lambda: _num(_matrix(CANCELLING_2X2).det()),
        contract=Returns(-1.0),
        verified_by="(2³⁰+1)(2³⁰-1) - 2³⁰·2³⁰ = (2⁶⁰ - 1) - 2⁶⁰ = -1 exactly. In float64 both "
        "products round to 2⁶⁰ and the difference cancels to 0.0 — a plausible, wrong, "
        "and *sign-flipping* answer (singular vs invertible).",
    ),
    Case(
        id="matrix_inverse_roundtrip",
        subsystem="linear_algebra",
        statement="inverse[[1,2],[3,4]] = [[-2,1],[3/2,-1/2]]",
        op=lambda: [
            [_num(entry) for entry in row] for row in _matrix([[1, 2], [3, 4]]).inverse().to_list()
        ],
        contract=Returns([[-2.0, 1.0], [1.5, -0.5]]),
        verified_by="1/det · adj = (1/-2)·[[4,-2],[-3,1]] = [[-2,1],[1.5,-0.5]], by hand. The "
        "control for the singular-inverse refusals.",
    ),
    # The matrix exponential of a DEFECTIVE matrix.
    #
    # `e^A` for a diagonalisable A is `P·e^D·P⁻¹` and was always right.  For a
    # defective A the Jordan block contributes `e^λ·Σ N^k/k!`, and in 3.10.0 the
    # nilpotent power `N^k` was written `λ^k` — so every nilpotent block lost its
    # off-diagonal entirely (`exp([[0,1],[0,0]])` came back as the identity) and
    # every other defective block was off by a factor of `λ^k`.  Clean, plausible
    # matrices, no exception, no flag.
    #
    # Every expectation below is `sympy.Matrix(M).exp()`, checked independently.
    # The diagonal entries are deliberately *not* scored: they were right
    # throughout, so a case reading one would have passed the whole time.
    # -----------------------------------------------------------------------
    Case(
        id="matrix_exp_nilpotent_2x2_off_diagonal",
        subsystem="linear_algebra",
        statement="exp([[0,1],[0,0]])[0][1] = 1",
        op=_exp_entry(NILPOTENT_2X2, 0, 1),
        contract=Returns(1.0),
        verified_by="N² = 0, so the series terminates: e^N = I + N = [[1,1],[0,1]]. "
        "sympy.Matrix([[0,1],[0,0]]).exp() agrees. alkahest 3.10.0 returned the identity, "
        "i.e. 0 here — the off-diagonal of e^N for a nilpotent N is the one entry that "
        "cannot be zero, since e^N = I would force N = log I = 0.",
    ),
    Case(
        id="matrix_exp_defective_off_diagonal_is_not_doubled",
        subsystem="linear_algebra",
        statement="exp([[2,1],[0,2]])[0][1] = e², not 2e²",
        op=_exp_entry(DEFECTIVE_2X2, 0, 1),
        contract=Returns(7.38905609893065),
        verified_by="A = 2I + N with N² = 0, and 2I commutes with N, so "
        "e^A = e²·(I + N) = [[e², e²], [0, e²]]; e² = 7.38905609893065. "
        "sympy.Matrix([[2,1],[0,2]]).exp() agrees. alkahest 3.10.0 returned 2e² = "
        "14.7781121978613 — exactly the factor λ¹ that the k = 1 term should not carry.",
    ),
    Case(
        id="matrix_exp_nilpotent_3x3_second_superdiagonal",
        subsystem="linear_algebra",
        statement="exp([[0,1,0],[0,0,1],[0,0,0]])[0][2] = 1/2",
        op=_exp_entry(NILPOTENT_3X3, 0, 2),
        contract=Returns(0.5),
        verified_by="N³ = 0, so e^N = I + N + N²/2 and the (0,2) entry is (N²)₀₂/2! = 1/2. "
        "sympy agrees. Two distinct defects met here in 3.10.0: the λ^k factor zeroed it, "
        "and the block-size detector read J[i][i+sz] instead of J[i+sz−1][i+sz], splitting "
        "the 3×3 block into a 2×2 and a 1×1 so the 1/2 had nowhere to come from.",
    ),
    Case(
        id="matrix_exp_full_jordan_block_corner",
        subsystem="linear_algebra",
        statement="exp([[2,1,0],[0,2,1],[0,0,2]])[0][2] = e²/2",
        op=_exp_entry(JORDAN_3X3, 0, 2),
        contract=Returns(3.694528049465325),
        verified_by="e^{2I+N} = e²(I + N + N²/2); the corner is e²/2 = 3.694528049465325. "
        "sympy.Matrix([[2,1,0],[0,2,1],[0,0,2]]).exp() agrees. The 3×3 block is the "
        "smallest matrix on which the block-size misdetection is visible on its own.",
    ),
    Case(
        id="matrix_exp_two_jordan_blocks_one_eigenvalue",
        subsystem="linear_algebra",
        statement="exp(J₂(3) ⊕ J₂(3))[2][3] = e³",
        op=_exp_entry(TWO_JORDAN_BLOCKS, 2, 3),
        contract=Returns(20.085536923187668),
        verified_by="e^A is block diagonal with each block e³(I + N) = [[e³, e³],[0, e³]]; "
        "e³ = 20.085536923187668. sympy agrees. This is the shape that breaks a Jordan-basis "
        "route rather than the block formula: both chains come out of the same kernel, so a "
        "P built by taking the same generator twice is singular.",
    ),
    Case(
        id="matrix_exp_defective_without_a_zero_off_diagonal",
        subsystem="linear_algebra",
        statement="exp([[1,1],[-1,3]])[0][0] = 0",
        op=_exp_entry(DEFECTIVE_DENSE_2X2, 0, 0),
        contract=Returns(0.0),
        verified_by="det(λI − A) = λ² − 4λ + 4 = (λ − 2)², and A − 2I = [[-1,1],[-1,1]] has "
        "rank 1, so A is defective with one 2×2 block. e^A = e²(I + (A − 2I)) = "
        "[[0, e²], [-e², 2e²]], whose (0,0) entry is exactly 0. sympy agrees. Nothing on the "
        "surface of this matrix announces defectiveness — no zero off-diagonal, no repeated "
        "diagonal entry — so it is the case a triangular-only fast path would miss.",
    ),
    Case(
        id="matrix_exp_rotation_off_diagonal_is_sin_one",
        subsystem="linear_algebra",
        statement="exp([[0,1],[-1,0]])[0][1] = sin 1",
        op=_exp_entry(ROTATION_2X2, 0, 1),
        contract=Returns(0.8414709848078965),
        verified_by="A generates rotation: e^{θA} = [[cos θ, sin θ], [-sin θ, cos θ]], so at "
        "θ = 1 the (0,1) entry is sin 1 = 0.8414709848078965 (Euler / the 2×2 rotation "
        "group). sympy.Matrix([[0,1],[-1,0]]).exp() agrees. The complex-spectrum control: "
        "λ = ±i are distinct, so the answer must be real and must not acquire an imaginary "
        "part from the route that produced it.",
    ),
    Case(
        id="matrix_exp_symbolic_defective_gap",
        subsystem="linear_algebra",
        statement="exp([[a,1],[0,b]])[0][1] at a = b = 3/2 is e^{3/2}, not a 0/0 form",
        op=_exp_entry_at(SYMBOLIC_GAP_2X2, 0, 1, a=1.5, b=1.5),
        contract=RefusesOr(4.4816890703380645),
        verified_by="For a != b the entry is (e^a − e^b)/(a − b); its limit as b → a is e^a, "
        "which is also what the defective case gives directly: at a = b the matrix is "
        "a·I + N with N² = 0, so e^A = e^a(I + N) and the entry is e^{3/2} = "
        "4.4816890703380645. sympy: `sp.Matrix([[a,1],[0,b]]).exp().subs(b,a)` and "
        "`sp.limit((sp.exp(a)-sp.exp(b))/(a-b), b, a)` both give exp(a). Refusing is "
        "acceptable — which branch holds is not decidable from the matrix — but a *different* "
        "finite number is not, and neither is the generic form evaluated at the confluence, "
        "which is 0/0.",
        note="Passes today by a refusal, not by producing the confluent value: alkahest "
        "returns the generic (e^a − e^b)/(a − b) — correct everywhere except the confluence "
        "— and evaluating it at a = b raises E-EVAL-004. That is a refusal a caller has to "
        "look at the value to notice, so what carries the real signal is "
        "`alkahest.matrix_exp_side_conditions()`, which lists `a − b ≠ 0`; "
        "test_matrix_exp_reports_the_eigenvalue_gap_it_divided_by in "
        "tests/test_linear_algebra.py pins that. The stronger outcome is the "
        "Returns(e^{3/2}) branch, i.e. taking the confluent limit when the parameters are "
        "bound.",
    ),
    Case(
        id="matrix_exp_control_diagonalizable_2x2",
        subsystem="linear_algebra",
        statement="exp([[1,2],[3,4]])[0][0] = 51.968956198705",
        op=_exp_entry([[1, 2], [3, 4]], 0, 0),
        contract=Returns(51.968956198705),
        verified_by="Eigenvalues (5 ± √33)/2 are distinct, so A = PDP⁻¹ and e^A = Pe^DP⁻¹; "
        "sympy.Matrix([[1,2],[3,4]]).exp().evalf(20) gives 51.968956198705 in the (0,0) "
        "entry. The control for the defective cases above: a library that 'fixed' them by "
        "refusing every matrix with a repeated eigenvalue would still have to answer this "
        "one, and one that broke the diagonalisable route while repairing the Jordan route "
        "would fail here.",
    ),
    Case(
        id="matrix_exp_control_diagonal_2x2",
        subsystem="linear_algebra",
        statement="exp(diag(1,2))[1][1] = e²",
        op=_exp_entry([[1, 0], [0, 2]], 1, 1),
        contract=Returns(7.38905609893065),
        verified_by="exp(diag(d₁,…,dₙ)) = diag(e^{d₁},…,e^{dₙ}); e² = 7.38905609893065. "
        "sympy agrees. The second control, one step simpler than the diagonalisable one: it "
        "pins the entrywise fast path, which is the only route that never consults a "
        "spectrum at all.",
    ),
    Case(
        id="matrix_exp_control_zero_matrix_is_identity",
        subsystem="linear_algebra",
        statement="exp(0) = I, so exp([[0,0],[0,0]])[0][1] = 0",
        op=_exp_entry(ZERO_2X2, 0, 1),
        contract=Returns(0.0),
        verified_by="e^0 = I by the series, whose every term past the first vanishes. The "
        "companion to matrix_exp_nilpotent_2x2_off_diagonal: the identity is the *right* "
        "answer here and the wrong one there, and the two differ in a single entry of the "
        "input — so a gate holding only the nilpotent case could be passed by never "
        "returning the identity at all.",
    ),
    Case(
        id="jordan_form_transform_is_a_basis",
        subsystem="linear_algebra",
        statement="the P of jordan_form(J₂(3) ⊕ J₂(3)) has rank 4",
        op=_jordan_p_rank(TWO_JORDAN_BLOCKS),
        contract=RefusesOr(4),
        verified_by="M = P·J·P⁻¹ is a similarity, so P must be invertible and a 4×4 "
        "invertible matrix has rank 4; sympy.Matrix(M).jordan_form() returns P = I here, "
        "det 1. Both chains of this matrix are drawn from ker(M − 3I)² = ℝ⁴, and alkahest "
        "3.10.0 took the same generator for both, returning a P with two identical columns "
        "— rank 3, det 0, so the identity it claims is false and P⁻¹ does not exist. "
        "Refusing is acceptable; a rank-deficient P silently labelled a similarity transform "
        "is not.",
    ),
    Case(
        id="jordan_form_control_diagonalizable_transform",
        subsystem="linear_algebra",
        statement="the P of jordan_form([[1,2],[3,4]]) has rank 2",
        op=_jordan_p_rank([[1, 2], [3, 4]]),
        contract=Returns(2),
        verified_by="Distinct eigenvalues (5 ± √33)/2 give two independent eigenvectors, so "
        "P is invertible and has rank 2; sympy.Matrix([[1,2],[3,4]]).jordan_form() returns a "
        "P with det ≠ 0. The control for the case above: a library that answered it by "
        "refusing every jordan_form would pass that one and fail this.",
    ),
    Case(
        id="lll_rank_deficient_basis_is_answerable",
        subsystem="linear_algebra",
        statement="LLL on [[1,2],[2,4]] must return a basis of ℤ·(1,2), not panic",
        op=_lll_rows_stay_in_the_lattice([[1, 2], [2, 4]], [1, 2]),
        contract=Returns(True),
        verified_by=(
            "(2,4) = 2·(1,2), so the two rows span the rank-1 lattice ℤ·(1,2). Every row LLL "
            "returns must therefore be an integer multiple of (1,2), and (1,2) itself must still "
            "be reachable — checked in exact Fraction arithmetic on the returned rows, with no "
            "reference implementation involved."
        ),
        note=(
            "Pre-fix any rank-deficient basis divided by a zero Gram–Schmidt norm and panicked. "
            "Scored `no_answer` when that happens, not `silent_error`: the failure mode is a "
            "dead run, not a wrong number."
        ),
    ),
    Case(
        id="matrix_lu_signed_pivot_product_is_the_determinant",
        subsystem="linear_algebra",
        statement="sign(P)·Π u_ii = det A for A = [[2,-1,4],[-1,1,-1],[3,-4,0]], i.e. -1",
        op=lambda: _lu_signed_pivot_product(LU_PIVOTING_3X3),
        contract=Returns(-1.0),
        verified_by="det A = 2(0-4) + 1(0+3) + 4(4-3) = -8 + 3 + 4 = -1 by cofactor expansion; "
        "sympy.Matrix([[2,-1,4],[-1,1,-1],[3,-4,0]]).det() is -1. P·A = L·U with L unit "
        "triangular forces det U = det(P·A) = sign(P)·det A, so the signed product of the "
        "pivots is a number about A that the factorisation has to reproduce. It is read off "
        "U and perm, never off det().",
    ),
    Case(
        id="matrix_lu_reconstructs_the_permuted_matrix",
        subsystem="linear_algebra",
        statement="L·U = P·A entrywise for a 3×3 needing two pivot swaps",
        op=lambda: _lu_reconstruction_residual(LU_PIVOTING_3X3),
        contract=Returns(0.0, tol=1e-9),
        verified_by="P·A = L·U is the definition of the factorisation, so the residual is "
        "exactly 0 for any correct LU; sympy's own LUdecomposition of this matrix satisfies "
        "it. A = [[2,-1,4],[-1,1,-1],[3,-4,0]] pivots to row 2 at k = 0 and again at k = 1 — "
        "the multipliers already stored in L belong to the rows being swapped, and 3.10.0 "
        "left them behind, so L·U came back as the rows of A in the order 2,1,0 while perm "
        "said 2,0,1. Two swaps are needed to see it: the 2×2 case swaps only at k = 0, where "
        "L has no computed column yet.",
    ),
    Case(
        id="matrix_lu_control_needs_no_pivot_swap",
        subsystem="linear_algebra",
        statement="L·U = P·A for [[3,1],[1,2]], where partial pivoting never swaps",
        op=lambda: _lu_reconstruction_residual([[3, 1], [1, 2]]),
        contract=Returns(0.0, tol=1e-9),
        verified_by="|3| > |1| in column 0 and the remaining 1×1 pivot is 5/3, so perm is the "
        "identity and L·U = A with L = [[1,0],[1/3,1]], U = [[3,1],[0,5/3]] — one Gaussian "
        "step by hand, and what sympy's LUdecomposition returns. The control for the case "
        "above: a library that 'fixed' the permutation bug by never permuting would pass "
        "this and fail that.",
    ),
    Case(
        id="matrix_qr_first_diagonal_is_the_column_norm",
        subsystem="linear_algebra",
        statement="R[0][0] = ‖first column of [[1,2],[3,4]]‖ = √10",
        op=lambda: _qr_entry(QR_2X2, 1, 0, 0),
        contract=Returns(3.1622776601683795),
        verified_by="Gram–Schmidt sets q₁ = a₁/‖a₁‖ and r₁₁ = ‖a₁‖; ‖(1,3)‖ = √10 = "
        "3.1622776601683795. That is a number about the input, not about the algorithm, and "
        "it is what sympy.Matrix([[1,2],[3,4]]).QRdecomposition()[1][0,0] returns.",
    ),
    Case(
        id="matrix_qr_reconstructs_the_matrix",
        subsystem="linear_algebra",
        statement="Q·R = A entrywise for [[1,2],[3,4]]",
        op=lambda: _qr_reconstruction_residual(QR_2X2),
        contract=Returns(0.0, tol=1e-9),
        verified_by="Q·R = A is the definition; residual 0 exactly. Paired with the "
        "orthonormality case below, because Q·R = A alone is satisfied by Q = A, R = I.",
    ),
    Case(
        id="matrix_qr_columns_are_orthonormal",
        subsystem="linear_algebra",
        statement="QᵀQ = I for the Q of [[1,2],[3,4]]",
        op=lambda: _qr_orthonormality_residual(QR_2X2),
        contract=Returns(0.0, tol=1e-9),
        verified_by="Orthonormality is the other half of the QR contract and the half a "
        "reconstruction check cannot see. qᵢ·qⱼ = δᵢⱼ by definition of Gram–Schmidt; sympy's "
        "QRdecomposition satisfies it. Scored as the worst |qᵢ·qⱼ − δᵢⱼ| over all pairs.",
    ),
    Case(
        id="matrix_qr_of_a_rank_deficient_matrix_has_an_orthonormal_q",
        subsystem="linear_algebra",
        statement="QᵀQ = I for the Q of the rank-1 matrix [[-4,-4],[-2,-2]]",
        op=lambda: _qr_orthonormality_residual(RANK_ONE_2X2),
        contract=Returns(0.0, tol=1e-9),
        verified_by="Orthonormality is part of the definition of a QR factorisation and "
        "does not lapse when A is rank deficient — sympy returns the rank-sized "
        "Q (2×1 here) rather than a padded one, and LAPACK's full QR returns a 2×2 "
        "orthogonal Q. alkahest padded with the **zero vector**: Q = [[-2/√5,0],[-1/√5,0]], "
        "so Q·R = A held and QᵀQ was diag(1,0). A caller reading Q⁻¹ = Qᵀ, which is the "
        "reason to want a QR at all, was reading a falsehood with nothing to signal it; "
        "62 of 304 randomised matrices came back that way.",
    ),
    Case(
        id="matrix_qr_of_a_rank_deficient_matrix_still_reconstructs",
        subsystem="linear_algebra",
        statement="Q·R = [[-4,-4],[-2,-2]] even though the second column is dependent",
        op=lambda: _qr_reconstruction_residual(RANK_ONE_2X2),
        contract=Returns(0.0, tol=1e-9),
        verified_by="A = QR is the other half of the definition and must survive the "
        "orthonormality repair: filling Q's second column changes nothing in Q·R, because "
        "the R row that multiplies it is zero. The control that pins the repair as a "
        "repair rather than a substitution.",
    ),
    Case(
        id="matrix_cholesky_of_two_i_is_sqrt_two_i",
        subsystem="linear_algebra",
        statement="cholesky(2I)[0][0] = √2",
        op=lambda: _cholesky_entry(TWO_I, 0, 0),
        contract=Returns(1.4142135623730951),
        verified_by="sympy.Matrix([[2,0],[0,2]]).cholesky() is [[sqrt(2),0],[0,sqrt(2)]]; "
        "√2 = 1.4142135623730951. 2I is symmetric with both leading principal minors "
        "positive (2 and 4), so it is positive definite and has a Cholesky factor. alkahest "
        "3.10.0 refused it with E-LINALG-003, *'matrix is not symmetric positive definite'* — "
        "a refusal that states a falsehood about the input, because the rational path "
        "required every pivot to be a perfect rational square.",
    ),
    Case(
        id="matrix_cholesky_irrational_pivot",
        subsystem="linear_algebra",
        statement="cholesky([[4,2],[2,3]])[1][1] = √2",
        op=lambda: _cholesky_entry(SPD_IRRATIONAL_PIVOT, 1, 1),
        contract=Returns(1.4142135623730951),
        verified_by="sympy.Matrix([[4,2],[2,3]]).cholesky() is [[2,0],[1,sqrt(2)]]. By hand: "
        "l₁₁ = √4 = 2, l₂₁ = 2/2 = 1, l₂₂ = √(3 − 1²) = √2 = 1.4142135623730951. The second "
        "pivot is irrational while the first is not, so this separates 'refuses irrational "
        "pivots' from 'refuses irrational matrices'.",
    ),
    Case(
        id="matrix_cholesky_reconstructs_the_matrix",
        subsystem="linear_algebra",
        statement="L·Lᵀ = M entrywise for M = [[4,2],[2,3]]",
        op=lambda: _cholesky_reconstruction_residual(SPD_IRRATIONAL_PIVOT),
        contract=Returns(0.0, tol=1e-9),
        verified_by="L·Lᵀ = M is the definition, so the residual is exactly 0. This is the "
        "check that catches a factor of a *different* matrix: cholesky([[1,5],[0,1]]) used "
        "to return the identity, whose L·Lᵀ is I and not the input.",
    ),
    Case(
        id="matrix_cholesky_refuses_a_non_symmetric_matrix",
        subsystem="linear_algebra",
        statement="[[1,5],[0,1]] has no Cholesky factor",
        op=lambda: _cholesky_entry(NON_SYMMETRIC_2X2, 0, 0),
        contract=Raises("E-LINALG-003"),
        verified_by="L·Lᵀ is symmetric for every L — (L·Lᵀ)ᵀ = L·Lᵀ — so a non-symmetric "
        "matrix has no Cholesky factor at all. sympy.Matrix([[1,5],[0,1]]).cholesky() raises "
        "ValueError('Matrix must be Hermitian'). alkahest read only the lower triangle and "
        "returned the identity, silently factoring I instead.",
    ),
    Case(
        id="matrix_cholesky_refuses_an_indefinite_matrix",
        subsystem="linear_algebra",
        statement="[[1,2],[2,1]] is symmetric but indefinite, so it has no Cholesky factor",
        op=lambda: _cholesky_entry(INDEFINITE_SYMMETRIC, 0, 0),
        contract=Raises("E-LINALG-003"),
        verified_by="Its eigenvalues are 3 and −1 (sympy.Matrix([[1,2],[2,1]]).eigenvals() "
        "gives {3: 1, -1: 1}), and the second leading principal minor is 1 − 4 = −3 < 0, so "
        "Sylvester's criterion fails. The control for the √2 repair above: widening the "
        "rational path to irrational pivots must not have widened it to indefinite pivots.",
    ),
    Case(
        id="matrix_eigenvalues_are_roots_of_the_characteristic_polynomial",
        subsystem="linear_algebra",
        statement="every λ returned for the companion matrix of λ³+3λ²+2λ+1 satisfies "
        "det(λI − M) = 0",
        op=lambda: _eigenvalue_charpoly_residual(CARDANO_COMPANION),
        contract=Returns(0.0, tol=1e-9),
        verified_by="The defining property, evaluated with an independent complex "
        "determinant (Gaussian elimination over Python complex, in this file). λ³+3λ²+2λ+1 "
        "is irreducible over ℚ with negative discriminant, so Cardano's two radicands are "
        "both negative — the casus where two independent Pow nodes pick up different cube "
        "roots of unity and A·B ≠ −p/3. In 3.9 the three returned values gave |p(λ)| = "
        "2.2944788, 1.5048698, 1.5048698.",
    ),
    Case(
        id="matrix_eigenvalue_real_root_of_the_cardano_cubic",
        subsystem="linear_algebra",
        statement="the one real eigenvalue of the companion matrix of λ³+3λ²+2λ+1 is "
        "−2.324717957244746",
        op=lambda: _smallest_real_eigenvalue(CARDANO_COMPANION),
        contract=Returns(-2.324717957244746, tol=1e-9),
        verified_by="sympy.Poly(l**3+3*l**2+2*l+1, l).nroots(n=25) gives "
        "−2.324717957244746025960909 and a conjugate pair; it is the negated supergolden "
        "ratio's companion root and is the unique real root because the discriminant is "
        "negative. A residual case alone would not catch an answer that is a root of the "
        "wrong polynomial, so the value itself is pinned as well.",
    ),
    Case(
        id="matrix_eigenvects_geometric_multiplicity_of_a_defective_block",
        subsystem="linear_algebra",
        statement="λ = 2 of [[2,1],[0,2]] has geometric multiplicity 1, not 2",
        op=lambda: _geometric_multiplicity(DEFECTIVE_BLOCK_2X2, 2.0),
        contract=Returns(1),
        verified_by="A − 2I = [[0,1],[0,0]] has rank 1, so its kernel is 1-dimensional; "
        "sympy.Matrix([[2,1],[0,2]]).eigenvects() returns a single vector (1,0) for λ = 2 "
        "with algebraic multiplicity 2. Reporting 2 here would be reporting a basis that "
        "does not exist, and it is what makes a P singular.",
    ),
    Case(
        id="matrix_eigenvects_control_scalar_matrix_has_a_full_eigenspace",
        subsystem="linear_algebra",
        statement="λ = 2 of 2I has geometric multiplicity 2",
        op=lambda: _geometric_multiplicity(TWO_I, 2.0),
        contract=Returns(2),
        verified_by="2I − 2I = 0, whose kernel is all of ℝ²; sympy.Matrix([[2,0],[0,2]])"
        ".eigenvects() returns two independent vectors for λ = 2. Same spectrum as the "
        "defective case above — one repeated eigenvalue — and the opposite answer, so a "
        "library that reported 1 for every repeated eigenvalue would fail here.",
    ),
    Case(
        id="matrix_diagonalize_reconstructs_an_irrational_spectrum",
        subsystem="linear_algebra",
        statement="M·P = P·D for the diagonalization of [[1,2],[3,4]]",
        op=lambda: _diagonalize_residual(QR_2X2),
        contract=Returns(0.0, tol=1e-9),
        verified_by="Eigenvalues (5 ± √33)/2 are distinct "
        "(sympy.Matrix([[1,2],[3,4]]).eigenvals()), so the matrix is diagonalizable and "
        "M = P·D·P⁻¹ holds exactly; the residual is 0. Before this audit alkahest refused it "
        "with E-EIGEN-005, *'matrix is not diagonalizable'* — false about this matrix, and "
        "contradicted by its own jordan_form, which returned a diagonal J for it. The cause "
        "was a structural comparison of two normalised forms: the M·v side carries √33·√33 "
        "where the λ·v side carries 33.",
    ),
    Case(
        id="matrix_diagonalize_refuses_a_defective_matrix",
        subsystem="linear_algebra",
        statement="[[2,1],[0,2]] is not diagonalizable",
        op=lambda: _diagonalize_residual(DEFECTIVE_BLOCK_2X2),
        contract=Raises("E-EIGEN-005"),
        verified_by="A single eigenvalue 2 of algebraic multiplicity 2 with a 1-dimensional "
        "eigenspace; sympy.Matrix([[2,1],[0,2]]).is_diagonalizable() is False. The control "
        "for the case above: widening the verification to a zero test must not have widened "
        "it to matrices that have no diagonalization.",
    ),
    Case(
        id="matrix_rational_canonical_form_preserves_the_determinant",
        subsystem="linear_algebra",
        statement="det C = det diag(1,2) = 2 for the rational canonical form of diag(1,2)",
        op=lambda: _rcf_determinant(DIAG_1_2),
        contract=Returns(2.0),
        verified_by="Similar matrices have equal determinants (det(PCP⁻¹) = det C), and "
        "det diag(1,2) = 2. The single invariant factor is λ²−3λ+2, whose companion matrix "
        "is [[0,−2],[1,3]] with determinant 2 — the standard Frobenius form, as in Dummit & "
        "Foote §12.2. alkahest wrote the coefficients along the last *row* instead of the "
        "last *column*, producing [[0,0],[−2,3]] with determinant 0: not similar to M, and "
        "not even of the same rank.",
    ),
    Case(
        id="matrix_rational_canonical_form_reconstructs_the_matrix",
        subsystem="linear_algebra",
        statement="M·P = P·C for the rational canonical form of diag(1,2)",
        op=lambda: _rcf_residual(DIAG_1_2),
        contract=Returns(0.0, tol=1e-9),
        verified_by="M = P·C·P⁻¹ rearranges to M·P = P·C, which is the whole content of the "
        "claim and needs no inverse to check. Residual 0 exactly. The two tests this "
        "function had before asserted only that P has 2 rows, which a matrix of the wrong "
        "rank passes.",
    ),
    Case(
        id="matrix_minimal_polynomial_of_the_zero_matrix_is_lambda",
        subsystem="linear_algebra",
        statement="the minimal polynomial of the 2×2 zero matrix, evaluated at λ = 3, is 3",
        op=lambda: _minimal_polynomial_at(ZERO_2X2, 3.0),
        contract=Returns(3.0),
        verified_by="p(M) = 0 with p = λ, since M itself is the zero matrix, and no constant "
        "polynomial annihilates it (p = 1 gives I). So the minimal polynomial is λ and p(3) "
        "= 3; λ² would give 9. Scored at a point rather than structurally so the case is "
        "immune to the form it comes back in. alkahest returned λ², whose defining property "
        "the zero matrix does not need — a correct annihilator, but not the minimal one, "
        "which is the entire content of the function's name.",
    ),
    Case(
        id="matrix_minimal_polynomial_of_two_equal_nilpotent_blocks",
        subsystem="linear_algebra",
        statement="the minimal polynomial of J₂(0) ⊕ J₂(0), evaluated at λ = 3, is 9",
        op=lambda: _minimal_polynomial_at(TWO_NILPOTENT_BLOCKS, 3.0),
        contract=Returns(9.0),
        verified_by="M² = 0 and M ≠ 0, so the minimal polynomial is λ² and p(3) = 9; the "
        "characteristic polynomial is λ⁴ and the largest Jordan block is 2×2, which is the "
        "standard characterisation. alkahest returned λ³ (p(3) = 27) — one degree too high, "
        "because a vanishing *constant* term failed to advance the matrix power, so p(M) was "
        "evaluated as (p/λ)(M). A zero constant term is exactly 0 ∈ spec(M), so every "
        "singular matrix was exposed.",
    ),
    Case(
        id="matrix_minimal_polynomial_control_distinct_eigenvalues",
        subsystem="linear_algebra",
        statement="the minimal polynomial of [[1,2],[3,4]], evaluated at λ = 3, is −8",
        op=lambda: _minimal_polynomial_at(QR_2X2, 3.0),
        contract=Returns(-8.0),
        verified_by="Distinct eigenvalues, so the minimal polynomial equals the "
        "characteristic one, λ²−5λ−2 (sympy.Matrix([[1,2],[3,4]]).charpoly()); at λ = 3 that "
        "is 9 − 15 − 2 = −8. The control for the two above: a library that returned λ for "
        "everything would pass the zero-matrix case and fail this.",
    ),
    Case(
        id="matrix_characteristic_polynomial_value_at_two",
        subsystem="linear_algebra",
        statement="det(2I − [[1,2],[3,4]]) = −8",
        op=lambda: _characteristic_polynomial_at(QR_2X2, 2.0),
        contract=Returns(-8.0),
        verified_by="2I − A = [[1,−2],[−3,−2]], determinant 1·(−2) − (−2)(−3) = −2 − 6 = −8; "
        "sympy.Matrix([[1,2],[3,4]]).charpoly().eval(2) is −8. Scored at a point, so no "
        "assumption is made about which of λI−M or M−λI is returned up to sign — for n = 2 "
        "they agree, which is why n = 2 is used.",
    ),
    Case(
        id="matrix_row_space_basis_is_independent_after_a_row_swap",
        subsystem="linear_algebra",
        statement="the row space basis of [[0,0],[1,0]] has rank 1",
        op=lambda: _row_space_basis_rank(ZERO_LEADING_ROW),
        contract=Returns(1),
        verified_by="The row space is span{(1,0)}, 1-dimensional, so any basis of it has "
        "rank 1 — a basis containing the zero vector has rank 0. Elimination swaps the two "
        "rows and marks echelon row 0 as the pivot row; alkahest then returned m.row(0) = "
        "(0,0), the zero vector offered as a basis of a one-dimensional space. Scored as the "
        "rank of the returned vectors stacked, which is 0 for the old answer and 1 for a "
        "basis.",
    ),
    Case(
        id="matrix_column_space_basis_is_independent",
        subsystem="linear_algebra",
        statement="the column space basis of [[1,2,3],[4,5,6],[7,8,9]] has rank 2",
        op=lambda: _column_space_basis_rank(SINGULAR_3X3),
        contract=Returns(2),
        verified_by="sympy.Matrix([[1,2,3],[4,5,6],[7,8,9]]).rank() is 2 (col3 − col2 = "
        "col2 − col1 = (1,1,1)), so a basis of the column space has exactly 2 independent "
        "vectors. The counterpart to the row-space case: column indices are untouched by row "
        "operations, so this path was never wrong — which is what makes it the control.",
    ),
    Case(
        id="matrix_power_five_is_the_fibonacci_matrix",
        subsystem="linear_algebra",
        statement="[[1,1],[1,0]]^5 = [[8,5],[5,3]], so entry (0,0) is 8",
        op=lambda: _entry_value(FIBONACCI_2X2, 5, 0, 0),
        contract=Returns(8.0),
        verified_by="The Fibonacci matrix identity Qⁿ = [[F_{n+1}, F_n], [F_n, F_{n−1}]] "
        "(Knuth, TAOCP vol. 1 §1.2.8); F₆ = 8, F₅ = 5, F₄ = 3. "
        "sympy.Matrix([[1,1],[1,0]])**5 is [[8,5],[5,3]].",
    ),
    Case(
        id="matrix_power_zero_is_the_identity",
        subsystem="linear_algebra",
        statement="[[1,1],[1,0]]^0 = I, so entry (0,1) is 0",
        op=lambda: _entry_value(FIBONACCI_2X2, 0, 0, 1),
        contract=Returns(0.0),
        verified_by="M⁰ = I by definition of the empty product, for every square M including "
        "a singular one. The control for the case above, and the edge the exponent loop is "
        "most likely to get wrong.",
    ),
    Case(
        id="matrix_trace_is_the_sum_of_the_diagonal",
        subsystem="linear_algebra",
        statement="tr[[1,2],[3,4]] = 5",
        op=lambda: _num(_matrix(QR_2X2).trace()),
        contract=Returns(5.0),
        verified_by="1 + 4 = 5; sympy.Matrix([[1,2],[3,4]]).trace() is 5. Also the sum of "
        "the eigenvalues (5 − √33)/2 + (5 + √33)/2 = 5, which is the identity that ties this "
        "to the spectral cases.",
    ),
    Case(
        id="matrix_hadamard_is_entrywise",
        subsystem="linear_algebra",
        statement="([[1,2],[3,4]] ∘ [[5,6],[7,8]])[1][1] = 32",
        op=lambda: _hadamard_entry(QR_2X2, [[5, 6], [7, 8]], 1, 1),
        contract=Returns(32.0),
        verified_by="4 · 8 = 32 by the definition of the Hadamard product; "
        "sympy.matrix_multiply_elementwise([[1,2],[3,4]], [[5,6],[7,8]]) gives "
        "[[5,12],[21,32]]. The ordinary matrix product of the same operands is "
        "[[19,22],[43,50]], so a confusion of the two shows up here as 50 against 32.",
    ),
    Case(
        id="matrix_transpose_swaps_the_indices",
        subsystem="linear_algebra",
        statement="[[1,2],[3,4]]ᵀ[0][1] = 3",
        op=lambda: _num(_matrix(QR_2X2).transpose().to_list()[0][1]),
        contract=Returns(3.0),
        verified_by="(Aᵀ)ᵢⱼ = Aⱼᵢ, so the (0,1) entry is A₁₀ = 3. Scored off an off-diagonal "
        "entry, the only place a transpose can be wrong.",
    ),
    Case(
        id="matrix_inverse_of_a_dense_symbolic_matrix_round_trips",
        subsystem="linear_algebra",
        statement="A⁻¹·A = I for a 4×4 matrix of 16 distinct free symbols",
        op=lambda: _symbolic_inverse_residual(4),
        contract=Returns(0.0, tol=1e-9),
        verified_by="adj(A)/det(A) is the inverse wherever det A ≠ 0 (Cramer), and det of a "
        "matrix of n² distinct symbols is a sum of n! distinct monomials over ℤ — not the "
        "zero polynomial for any n. Verified by substituting 16 unstructured rationals and "
        "checking the product against I exactly, rather than by symbolic cancellation, which "
        "no normaliser here finishes. The sample values are deliberately *not* an affine "
        "function of (i, j): that makes the probe matrix's rows an arithmetic progression, "
        "so det vanishes at the sample and A⁻¹·A is 0/0 rather than I — a degenerate test "
        "point, not a defect. Before this audit a 5×5 or larger refused with E-MAT-004.",
    ),
    Case(
        id="matrix_inverse_reports_the_determinant_it_divided_by",
        subsystem="linear_algebra",
        statement="inverse([[a,b],[c,d]]) reports exactly one hypothesis, ad − bc ≠ 0",
        op=lambda: _inverse_condition_count(_symbolic_2x2),
        contract=Returns(1),
        verified_by="[[a,b],[c,d]] is invertible iff ad − bc ≠ 0 (Cramer); on the locus "
        "ad = bc it has no inverse at all, and no value of adj/det is correct there. That "
        "the determinant is not the zero *function* is what licenses the formula; that it is "
        "non-zero at a given point is a question about the parameters and belongs to the "
        "caller. Scored as a count so the case does not depend on how the condition prints.",
    ),
    Case(
        id="matrix_inverse_control_constant_determinant_reports_nothing",
        subsystem="linear_algebra",
        statement="inverse([[1,2],[3,4]]) reports no hypothesis",
        op=lambda: _inverse_condition_count(lambda: _matrix(QR_2X2)),
        contract=Returns(0),
        verified_by="det = 1·4 − 2·3 = −2, a non-zero constant, so the hypothesis is "
        "*discharged* rather than skipped and there is nothing to report. The control for "
        "the case above: a channel that reported a condition unconditionally would be noise "
        "a caller learns to ignore — the failure the transform work named when it stopped "
        "reporting π ≠ 0.",
    ),
]
