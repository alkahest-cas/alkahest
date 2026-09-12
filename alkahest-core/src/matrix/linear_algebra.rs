//! Symbolic linear algebra: nullspace, rank, LU/QR/Cholesky, Jordan and rational canonical
//! forms, minimal polynomial, and matrix exponential.

#![allow(clippy::needless_range_loop)]

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::matrix::eigen::{
    self, characteristic_polynomial_lambda_minus_m, concatenate_columns, kernel_column_basis,
    m_minus_lambda_scaled, KernelFailure, KnownSingular,
};
use crate::matrix::normal_form::{smith_form_poly, PolyMatrixQ, RatUniPoly};
use crate::matrix::{exp_gate, putzer, spectrum, zero_test, Matrix, MatrixError};
use crate::poly::unipoly::UniPoly;
use crate::poly::{factor_univariate_z, FactorError};
use crate::simplify::engine::{simplify, simplify_expanded};
use rug::Rational;
use std::fmt;
use std::ops::Mul;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearAlgebraError {
    NonSquare,
    KernelFailed,
    NotPositiveDefinite,
    CharPolyConversion(crate::poly::error::ConversionError),
    Factorization(FactorError),
    UnsupportedIrreducibleDegree {
        degree: usize,
    },
    /// This routine could not answer over the field the entries live in.
    ///
    /// Three ways that happens: a Smith-based decomposition needs rational
    /// constants and got something else; elimination reached an entry whose
    /// vanishing it could not decide — over a transcendental extension that
    /// question is undecidable in general (see the `matrix::zero_test` module),
    /// and pivoting on an entry that might be identically zero is what produced
    /// full-rank verdicts for rank-deficient matrices; or
    /// [`matrix_exponential`] declined (see [`crate::matrix::exp_gate`]).
    ///
    /// Which of the three it was is available out of band, and exactly one of
    /// the two channels answers `Some`:
    /// [`take_matrix_exp_refusal`](crate::matrix::take_matrix_exp_refusal) for
    /// the exponential (code `E-LINALG-011`),
    /// [`take_zero_test_refusal`](crate::matrix::take_zero_test_refusal) for an
    /// undecided entry (code `E-LINALG-010`); neither means non-rational
    /// entries (code `E-LINALG-007`).
    UnsupportedField,
    /// A `(P, canonical form)` pair could not be produced with `M = P·F·P⁻¹`
    /// proven.
    ///
    /// Either `P` is singular — so `P⁻¹` does not exist and the identity is not
    /// even well-formed — or `M·P = P·F` could not be proven entrywise. Both
    /// are one failure from the caller's side: there is no transform to hand
    /// back. Returning the pair anyway is a silent error of the worst kind,
    /// because `P` and `F` each look perfectly ordinary in isolation; see
    /// [`rational_canonical_form`] for the two defects that shipped that way.
    SingularTransform,
    NonRationalEntry,
}

impl fmt::Display for LinearAlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearAlgebraError::NonSquare => write!(f, "operation requires a square matrix"),
            LinearAlgebraError::KernelFailed => write!(f, "could not compute nullspace basis"),
            LinearAlgebraError::NotPositiveDefinite => {
                write!(f, "matrix is not symmetric positive definite")
            }
            LinearAlgebraError::CharPolyConversion(e) => {
                write!(f, "characteristic polynomial: {e}")
            }
            LinearAlgebraError::Factorization(e) => write!(f, "factorization failed: {e}"),
            LinearAlgebraError::UnsupportedIrreducibleDegree { degree } => write!(
                f,
                "irreducible factor of degree {degree} in minimal polynomial"
            ),
            LinearAlgebraError::UnsupportedField => {
                write!(
                    f,
                    "this routine could not answer over the field the entries live in: \
                     a Smith-based decomposition needs rational constants, elimination \
                     needs entries whose vanishing it can decide, and the matrix \
                     exponential needs a confirmed spectrum whose eigenvalue gaps are \
                     settled"
                )
            }
            LinearAlgebraError::SingularTransform => {
                write!(
                    f,
                    "the similarity transform could not be produced: the computed \
                     transform is singular, or it does not conjugate the matrix to \
                     the canonical form that was computed for it"
                )
            }
            LinearAlgebraError::NonRationalEntry => {
                write!(f, "matrix entry is not a rational constant")
            }
        }
    }
}

impl std::error::Error for LinearAlgebraError {}

impl crate::errors::AlkahestError for LinearAlgebraError {
    fn code(&self) -> &'static str {
        match self {
            LinearAlgebraError::NonSquare => "E-LINALG-001",
            LinearAlgebraError::KernelFailed => "E-LINALG-002",
            LinearAlgebraError::NotPositiveDefinite => "E-LINALG-003",
            LinearAlgebraError::CharPolyConversion(_) => "E-LINALG-004",
            LinearAlgebraError::Factorization(_) => "E-LINALG-005",
            LinearAlgebraError::UnsupportedIrreducibleDegree { .. } => "E-LINALG-006",
            LinearAlgebraError::UnsupportedField => "E-LINALG-007",
            LinearAlgebraError::SingularTransform => "E-LINALG-008",
            LinearAlgebraError::NonRationalEntry => "E-LINALG-009",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            LinearAlgebraError::NonSquare => Some("pass a square matrix"),
            LinearAlgebraError::KernelFailed => {
                Some("try a matrix with rational entries or a ℚ-splitting spectrum")
            }
            LinearAlgebraError::NotPositiveDefinite => {
                Some("Cholesky requires a symmetric positive definite matrix")
            }
            LinearAlgebraError::CharPolyConversion(_) => {
                Some("entries must simplify to rationals so det(λI−M) is a polynomial in λ")
            }
            LinearAlgebraError::Factorization(_) => None,
            LinearAlgebraError::UnsupportedIrreducibleDegree { .. } => {
                Some("minimal polynomial has an irreducible factor of degree > 2")
            }
            LinearAlgebraError::UnsupportedField => Some(
                "use entries this routine can work over: rational or integer entries \
                 for Smith-based decompositions, and entries whose vanishing is \
                 decidable for elimination",
            ),
            LinearAlgebraError::SingularTransform => Some(
                "no verified similarity transform is available for this matrix; \
                 substitute concrete values for any parameters",
            ),
            LinearAlgebraError::NonRationalEntry => {
                Some("convert symbolic entries to rationals before calling this routine")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Nullspace, rank, column/row space
// ---------------------------------------------------------------------------

/// Basis of the nullspace (kernel) of `m`, as column vectors.
///
/// # Errors
///
/// [`LinearAlgebraError::UnsupportedField`] when elimination reached an entry
/// whose vanishing it could not decide — the same refusal [`rank`] and [`rref`]
/// make, carrying the specific `E-LINALG-010` through
/// [`take_zero_test_refusal`](crate::matrix::take_zero_test_refusal). It used
/// to be reported as the generic [`LinearAlgebraError::KernelFailed`], which
/// told a caller nothing about the one remediation that works (substitute
/// concrete values for the parameters).
pub fn nullspace_basis(m: &Matrix, pool: &ExprPool) -> Result<Vec<Matrix>, LinearAlgebraError> {
    // An arbitrary matrix: nothing is known about its determinant, so the 2×2
    // fast path has to establish singularity or refuse. See [`KnownSingular`].
    kernel_column_basis(m, pool, KnownSingular::No).map_err(|f| kernel_failure_to_error(f, pool))
}

/// Report a [`KernelFailure`] in this module's error vocabulary.
///
/// The whole point of [`KernelFailure`] carrying a payload: the undecided entry
/// survives the boundary, so the refusal keeps its own `E-LINALG-010` instead of
/// collapsing into [`LinearAlgebraError::KernelFailed`]'s
/// "could not compute nullspace basis".
///
/// [`LinearAlgebraError::KernelFailed`] is *not* a carrier — it has ~30 call
/// sites and no way to tell which one a stale thread-local refusal belongs to,
/// so a genuine kernel failure can never pick up this code by accident.
fn kernel_failure_to_error(f: KernelFailure, pool: &ExprPool) -> LinearAlgebraError {
    match f {
        KernelFailure::Undecidable(e) => inconclusive(pool, e),
    }
}

/// Rank of `m`.
pub fn rank(m: &Matrix, pool: &ExprPool) -> Result<usize, LinearAlgebraError> {
    Ok(row_echelon_pivots(m, pool)?.pivot_cols.len())
}

/// Reduced row echelon form of `m`.
pub fn rref(m: &Matrix, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    Ok(row_echelon_pivots(m, pool)?.echelon)
}

/// Basis of the column space of `m` (original pivot columns).
pub fn column_space_basis(m: &Matrix, pool: &ExprPool) -> Result<Vec<Matrix>, LinearAlgebraError> {
    let rref = row_echelon_pivots(m, pool)?;
    Ok(rref
        .pivot_cols
        .iter()
        .map(|&c| {
            Matrix::new(m.col(c).into_iter().map(|e| vec![e]).collect()).expect("column vector")
        })
        .collect())
}

/// Basis of the row space of `m`: the nonzero pivot rows of the echelon form.
///
/// The rows come from the **echelon form**, not from `m`. `pivot_row_flags[r]`
/// records that row `r` *of the echelon form* is a pivot row, and elimination
/// swaps rows, so reading `m.row(r)` — as this did before 3.10.1 — picks an
/// unrelated row of the input. On `[[0,0],[1,0]]` the pivot lands in echelon
/// row 0 after a swap and the old code returned `m.row(0) = [0, 0]`: the zero
/// vector, offered as a basis of a one-dimensional row space. On a rank-2
/// `3×3` with a duplicated row it returned that row twice. Row operations
/// preserve the row space, so the echelon pivot rows are always a basis of it;
/// the original rows at those indices need not even lie in a basis.
///
/// (The column-space counterpart is *not* affected: column indices are
/// untouched by row operations, so [`column_space_basis`] correctly returns
/// columns of `m`.)
pub fn row_space_basis(m: &Matrix, pool: &ExprPool) -> Result<Vec<Matrix>, LinearAlgebraError> {
    let rref = row_echelon_pivots(m, pool)?;
    Ok(rref
        .pivot_row_flags
        .iter()
        .enumerate()
        .filter_map(|(ri, &is_pivot)| {
            if is_pivot {
                Some(Matrix::new(vec![rref.echelon.row(ri)]).expect("row vector"))
            } else {
                None
            }
        })
        .collect())
}

struct RowEchelonInfo {
    pivot_cols: Vec<usize>,
    pivot_row_flags: Vec<bool>,
    echelon: Matrix,
}

fn row_echelon_pivots(m: &Matrix, pool: &ExprPool) -> Result<RowEchelonInfo, LinearAlgebraError> {
    let rows = m.rows;
    let cols = m.cols;
    if let Some(grid) = matrix_to_rational_grid(m, pool) {
        let (pivot_cols, pivot_row_flags, echelon_grid) =
            rational_row_echelon_pivots(&grid, rows, cols);
        return Ok(RowEchelonInfo {
            pivot_cols,
            pivot_row_flags,
            echelon: rational_grid_to_matrix(&echelon_grid, pool),
        });
    }
    let mut a: Vec<Vec<ExprId>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| simplify(m.get(r, c), pool).value)
                .collect()
        })
        .collect();
    let neg_one = pool.integer(-1_i32);
    let mut pivot_cols = Vec::new();
    let mut pivot_row_flags = vec![false; rows];
    let mut r_at = 0usize;
    for c in 0..cols {
        if r_at >= rows {
            break;
        }
        let Some((pr, piv)) = find_pivot(&mut a, r_at, rows, c, pool)? else {
            continue;
        };
        if pr != r_at {
            a.swap(pr, r_at);
        }
        let inv_p = simplify(pool.pow(piv, pool.integer(-1_i32)), pool).value;
        for cc in 0..cols {
            a[r_at][cc] = simplify(pool.mul(vec![inv_p, a[r_at][cc]]), pool).value;
        }
        for rr in 0..rows {
            if rr == r_at {
                continue;
            }
            let f = simplify(a[rr][c], pool).value;
            // Only *skipping* needs a decision here: subtracting `f · pivot_row`
            // is correct whatever `f` is, so an undecided factor costs work but
            // never correctness. Refusing here would be gratuitous.
            if zero_test::zero_status(pool, f).is_proven_zero() {
                continue;
            }
            for cc in 0..cols {
                let term = simplify(pool.mul(vec![f, a[r_at][cc]]), pool).value;
                let neg_term = simplify(pool.mul(vec![neg_one, term]), pool).value;
                a[rr][cc] = simplify(pool.add(vec![a[rr][cc], neg_term]), pool).value;
            }
        }
        pivot_cols.push(c);
        pivot_row_flags[r_at] = true;
        r_at += 1;
    }
    Ok(RowEchelonInfo {
        pivot_cols,
        pivot_row_flags,
        echelon: Matrix::new(a).expect("row echelon grid"),
    })
}

/// The first row at or below `r_at` whose entry in column `c` is **proven**
/// non-zero, together with that entry.
///
/// Three outcomes, and the middle one is the point of this function:
///
/// * `Ok(Some(..))` — a pivot proven not to vanish identically.
/// * `Ok(None)` — every candidate is proven zero, so the column has no pivot.
/// * `Err(..)` — no candidate could be proven non-zero and at least one could
///   not be proven zero either. Reporting `None` would claim a rank deficiency
///   that has not been established, and picking the undecided entry as a pivot
///   would claim the opposite; both are silent errors, so the caller is told
///   instead. See [`inconclusive`] for how that refusal is coded.
///
/// Entries proven zero are rewritten to the literal `0` in place, so the
/// echelon form that comes out shows a cleared column rather than an
/// unsimplified expression that happens to vanish.
fn find_pivot(
    a: &mut [Vec<ExprId>],
    r_at: usize,
    rows: usize,
    c: usize,
    pool: &ExprPool,
) -> Result<Option<(usize, ExprId)>, LinearAlgebraError> {
    let zero = pool.integer(0_i32);
    let mut undecided: Option<ExprId> = None;
    for rr in r_at..rows {
        let e = simplify(a[rr][c], pool).value;
        match zero_test::zero_status(pool, e) {
            zero_test::ZeroStatus::NonZero => return Ok(Some((rr, e))),
            zero_test::ZeroStatus::Zero => a[rr][c] = zero,
            zero_test::ZeroStatus::Unknown => undecided = undecided.or(Some(e)),
        }
    }
    match undecided {
        None => Ok(None),
        Some(e) => Err(inconclusive(pool, e)),
    }
}

/// Refuse an entry whose vanishing could not be decided.
///
/// [`LinearAlgebraError`] is a public exhaustive enum, so it cannot grow a
/// dedicated variant without a major semver break. The refusal is reported as
/// [`LinearAlgebraError::UnsupportedField`] — true as it stands, since the
/// entry lies in a field this routine cannot decide over — and the entry that
/// caused it is recorded for
/// [`take_zero_test_refusal`](crate::matrix::take_zero_test_refusal), which is
/// how bindings recover the specific `E-LINALG-010`. Same shape as
/// [`crate::calculus::limits::last_budget_trip`].
fn inconclusive(pool: &ExprPool, e: ExprId) -> LinearAlgebraError {
    zero_test::record_refusal(pool, e, zero_test::RefusalSite::Pivot);
    LinearAlgebraError::UnsupportedField
}

/// [`LinearAlgebraError::UnsupportedField`] for its other meaning: entries that
/// are not the rational constants a Smith-based decomposition needs.
///
/// Clears any recorded zero-test refusal, so this error is never re-attributed
/// to an undecided entry left behind by an earlier call on this thread.
fn unsupported_field() -> LinearAlgebraError {
    zero_test::forget_refusal();
    LinearAlgebraError::UnsupportedField
}

fn rational_row_echelon_pivots(
    mat: &[Vec<Rational>],
    rows: usize,
    cols: usize,
) -> (Vec<usize>, Vec<bool>, Vec<Vec<Rational>>) {
    let mut a = mat.to_vec();
    let mut pivot_cols = Vec::new();
    let mut pivot_row_flags = vec![false; rows];
    let mut r = 0usize;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        let mut piv = None;
        for rr in r..rows {
            if a[rr][c] != 0 {
                piv = Some(rr);
                break;
            }
        }
        let Some(pr) = piv else { continue };
        if pr != r {
            a.swap(pr, r);
        }
        let inv = Rational::from(1) / a[r][c].clone();
        for cc in 0..cols {
            a[r][cc] *= inv.clone();
        }
        for rr in 0..rows {
            if rr == r {
                continue;
            }
            let f = a[rr][c].clone();
            if f == 0 {
                continue;
            }
            for cc in 0..cols {
                let pivot_val = a[r][cc].clone();
                a[rr][cc] -= f.clone() * pivot_val;
            }
        }
        pivot_cols.push(c);
        pivot_row_flags[r] = true;
        r += 1;
    }
    (pivot_cols, pivot_row_flags, a)
}

// ---------------------------------------------------------------------------
// LU / QR / Cholesky
// ---------------------------------------------------------------------------

/// `P·A = L·U` with unit-diagonal `L`, upper triangular `U`, and row permutation `perm`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LuDecomposition {
    pub l: Matrix,
    pub u: Matrix,
    pub perm: Vec<usize>,
}

pub fn lu_decomposition(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<LuDecomposition, LinearAlgebraError> {
    let n = m.rows;
    let cols = m.cols;
    if n == 0 {
        return Ok(LuDecomposition {
            l: Matrix::identity(0, pool),
            u: Matrix::zeros(0, cols, pool),
            perm: vec![],
        });
    }
    if let Some(mut a) = matrix_to_rational_grid(m, pool) {
        let mut perm: Vec<usize> = (0..n).collect();
        let mut l = vec![vec![Rational::from(0); n]; n];
        let mut u = vec![vec![Rational::from(0); cols]; n];
        for i in 0..n {
            l[i][i] = Rational::from(1);
        }
        for k in 0..n.min(cols) {
            let mut piv_row = k;
            for r in (k + 1)..n {
                if a[r][k].clone().abs() > a[piv_row][k].clone().abs() {
                    piv_row = r;
                }
            }
            if a[piv_row][k] == 0 {
                for j in k..cols {
                    u[k][j] = a[k][j].clone();
                }
                continue;
            }
            if piv_row != k {
                a.swap(piv_row, k);
                perm.swap(piv_row, k);
                // The multipliers already stored in `L` belong to the rows they
                // were computed from, so a pivot swap has to move them too.
                // Leaving them behind is what made `P·A = L·U` false from the
                // second pivot onwards; see `lu_partial_pivot_permutes_l`.
                // `piv_row > k` here, so the split is always in that order.
                let (head, tail) = l.split_at_mut(piv_row);
                head[k][..k].swap_with_slice(&mut tail[0][..k]);
            }
            let pivot = a[k][k].clone();
            for j in k..cols {
                u[k][j] = a[k][j].clone();
            }
            for i in (k + 1)..n {
                let factor = a[i][k].clone() / pivot.clone();
                l[i][k] = factor.clone();
                let pivot_row: Vec<Rational> = a[k][k..cols].to_vec();
                for j in k..cols {
                    a[i][j] -= factor.clone() * pivot_row[j - k].clone();
                }
            }
        }
        return Ok(LuDecomposition {
            l: rational_grid_to_matrix(&l, pool),
            u: rational_grid_to_matrix(&u, pool),
            perm,
        });
    }
    expr_lu_decomposition(m, pool)
}

fn expr_lu_decomposition(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<LuDecomposition, LinearAlgebraError> {
    let n = m.rows;
    let cols = m.cols;
    let mut a: Vec<Vec<ExprId>> = (0..n)
        .map(|r| {
            (0..cols)
                .map(|c| simplify(m.get(r, c), pool).value)
                .collect()
        })
        .collect();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut l = Matrix::identity(n, pool);
    let mut u = Matrix::zeros(n, cols, pool);
    for k in 0..n.min(cols) {
        // Same contract as `find_pivot`: a column is declared pivot-free only
        // when every candidate is *proven* zero.
        let Some((piv_row, _)) = find_pivot(&mut a, k, n, k, pool)? else {
            for j in k..cols {
                u.set(k, j, a[k][j]);
            }
            continue;
        };
        if piv_row != k {
            a.swap(piv_row, k);
            perm.swap(piv_row, k);
            // Move the multipliers already stored in `L` with their rows — the
            // same defect the rational path had.
            for j in 0..k {
                let t = l.get(k, j);
                l.set(k, j, l.get(piv_row, j));
                l.set(piv_row, j, t);
            }
        }
        let pivot = a[k][k];
        let inv_p = simplify(pool.pow(pivot, pool.integer(-1_i32)), pool).value;
        for j in k..cols {
            u.set(k, j, a[k][j]);
        }
        for i in (k + 1)..n {
            let factor = simplify(pool.mul(vec![a[i][k], inv_p]), pool).value;
            l.set(i, k, factor);
            for j in k..cols {
                let sub = simplify(pool.mul(vec![factor, a[k][j]]), pool).value;
                let neg_sub = simplify(pool.mul(vec![pool.integer(-1_i32), sub]), pool).value;
                a[i][j] = simplify(pool.add(vec![a[i][j], neg_sub]), pool).value;
            }
        }
    }
    Ok(LuDecomposition { l, u, perm })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QrDecomposition {
    pub q: Matrix,
    pub r: Matrix,
}

pub fn qr_decomposition(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<QrDecomposition, LinearAlgebraError> {
    let n = m.rows;
    let k = m.cols;
    if k == 0 {
        return Ok(QrDecomposition {
            q: Matrix::zeros(n, 0, pool),
            r: Matrix::zeros(0, 0, pool),
        });
    }
    let mut q_cols: Vec<Matrix> = Vec::with_capacity(k);
    let mut r = Matrix::zeros(k, k, pool);
    for j in 0..k {
        let mut v = Matrix::new(m.col(j).into_iter().map(|e| vec![e]).collect())
            .map_err(|_| LinearAlgebraError::KernelFailed)?;
        for i in 0..j {
            let qi = &q_cols[i];
            let rij = dot_columns(qi, &v, pool)?;
            r.set(i, j, rij);
            let proj = qi.scale(rij, pool);
            v = v
                .sub(&proj, pool)
                .map_err(|_| LinearAlgebraError::KernelFailed)?;
        }
        let rjj = norm_column(&v, pool)?;
        // `‖v‖ = 0` means column `j` is dependent on the ones before it and Q
        // gets a zero column; `‖v‖ ≠ 0` means we may divide by it. Guessing
        // either way silently changes the rank of Q.
        match zero_test::zero_status(pool, rjj) {
            zero_test::ZeroStatus::Zero => {
                r.set(j, j, pool.integer(0_i32));
                q_cols.push(Matrix::zeros(n, 1, pool));
                continue;
            }
            zero_test::ZeroStatus::NonZero => {}
            zero_test::ZeroStatus::Unknown => return Err(inconclusive(pool, rjj)),
        }
        r.set(j, j, rjj);
        let inv = simplify(pool.pow(rjj, pool.integer(-1_i32)), pool).value;
        v = v.scale(inv, pool);
        q_cols.push(v);
    }
    let q = concatenate_columns(&q_cols, pool).map_err(|_| LinearAlgebraError::KernelFailed)?;
    Ok(QrDecomposition { q, r })
}

/// The lower-triangular `L` with `L·Lᵀ = M`, for symmetric positive definite `M`.
///
/// # Symmetry is checked, not assumed
///
/// The algorithm only ever reads the lower triangle, so before 3.10.1 a
/// non-symmetric matrix got its upper triangle silently discarded and an `L`
/// factoring a *different* matrix came back with no complaint:
/// `cholesky([[1,5],[0,1]])` returned the identity, whose `L·Lᵀ` is `I`, not
/// the input. `[[1,5],[0,1]]` has no Cholesky factor at all — `L·Lᵀ` is
/// symmetric for every `L`. Symmetry is now required, and the refusal is the
/// honest one the error already named.
///
/// # `√2` is a number
///
/// The rational path also refused every matrix whose pivots are not perfect
/// rational squares — `2·I` came back as `E-LINALG-003`, *"matrix is not
/// symmetric positive definite"*, about a matrix that is both. A refusal that
/// states something false is not a safe failure: it tells the caller the input
/// is defective when the implementation is.
///
/// The factorisation is now taken over ℚ **without square roots** — the
/// rational `L·D·Lᵀ` decomposition, whose pivots `dⱼ` decide definiteness
/// exactly — and the square roots are applied to `D` afterwards, symbolically
/// where they are irrational. `dⱼ ≤ 0` is still `E-LINALG-003`, and it now
/// means it. Where every `dⱼ` *is* a perfect square the entries come out as the
/// same rationals as before.
pub fn cholesky(m: &Matrix, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    if m.rows != m.cols {
        return Err(LinearAlgebraError::NonSquare);
    }
    let n = m.rows;
    if !is_symmetric(m, pool) {
        return Err(LinearAlgebraError::NotPositiveDefinite);
    }
    if let Some(a) = matrix_to_rational_grid(m, pool) {
        return rational_cholesky(&a, n, pool);
    }
    let mut l = Matrix::zeros(n, n, pool);
    for i in 0..n {
        for j in 0..=i {
            let mut s = pool.integer(0_i32);
            for t in 0..j {
                s = simplify(
                    pool.add(vec![s, pool.mul(vec![l.get(i, t), l.get(j, t)])]),
                    pool,
                )
                .value;
            }
            if i == j {
                let inner = simplify(
                    pool.add(vec![m.get(i, i), pool.mul(vec![pool.integer(-1_i32), s])]),
                    pool,
                )
                .value;
                l.set(i, j, simplify(pool.func("sqrt", vec![inner]), pool).value);
            } else {
                let num = simplify(
                    pool.add(vec![m.get(i, j), pool.mul(vec![pool.integer(-1_i32), s])]),
                    pool,
                )
                .value;
                l.set(
                    i,
                    j,
                    simplify(
                        pool.mul(vec![num, pool.pow(l.get(j, j), pool.integer(-1_i32))]),
                        pool,
                    )
                    .value,
                );
            }
        }
    }
    Ok(l)
}

/// `M = Mᵀ`, entry by entry, on *proven* equality.
///
/// An undecided pair answers `false`: `L·Lᵀ` is symmetric whatever `L` is, so a
/// matrix that cannot be shown symmetric cannot be shown to have a Cholesky
/// factor either, and the refusal is the correct outcome rather than a
/// conservative one.
fn is_symmetric(m: &Matrix, pool: &ExprPool) -> bool {
    for i in 0..m.rows {
        for j in (i + 1)..m.cols {
            let diff = simplify_expanded(
                pool.add(vec![
                    m.get(i, j),
                    pool.mul(vec![pool.integer(-1_i32), m.get(j, i)]),
                ]),
                pool,
            )
            .value;
            if !zero_test::zero_status(pool, diff).is_proven_zero() {
                return false;
            }
        }
    }
    true
}

/// Cholesky of a symmetric rational matrix, via the square-root-free `L·D·Lᵀ`.
///
/// `dⱼ = a_jj − Σ_{t<j} l_jt²·d_t` and `l_ij = (a_ij − Σ_{t<j} l_it·l_jt·d_t)/dⱼ`
/// are all in ℚ, so positive definiteness is decided exactly: `M` is positive
/// definite iff every `dⱼ > 0` (Sylvester, via the leading principal minors
/// `Π_{t≤j} d_t`). The Cholesky factor is then `L·√D`, column `j` scaled by
/// `√dⱼ` — rational when `dⱼ` is a perfect square, and the exact symbolic
/// `sqrt(dⱼ)` when it is not.
fn rational_cholesky(
    a: &[Vec<Rational>],
    n: usize,
    pool: &ExprPool,
) -> Result<Matrix, LinearAlgebraError> {
    let mut l = vec![vec![Rational::from(0); n]; n];
    let mut d = vec![Rational::from(0); n];
    for j in 0..n {
        let mut s = Rational::from(0);
        for t in 0..j {
            s += l[j][t].clone() * l[j][t].clone() * d[t].clone();
        }
        d[j] = a[j][j].clone() - s;
        if d[j] <= 0 {
            return Err(LinearAlgebraError::NotPositiveDefinite);
        }
        l[j][j] = Rational::from(1);
        for i in (j + 1)..n {
            let mut s = Rational::from(0);
            for t in 0..j {
                s += l[i][t].clone() * l[j][t].clone() * d[t].clone();
            }
            l[i][j] = (a[i][j].clone() - s) / d[j].clone();
        }
    }
    let mut out = Matrix::zeros(n, n, pool);
    for j in 0..n {
        // `√dⱼ` exactly: a rational when `dⱼ` is a perfect square, and the
        // symbolic root otherwise. Never a float, and never a refusal.
        let root = match rational_sqrt(&d[j]) {
            Some(r) => rational_expr(&r, pool),
            None => {
                let dj = rational_expr(&d[j], pool);
                simplify(pool.func("sqrt", vec![dj]), pool).value
            }
        };
        for i in j..n {
            let lij = rational_expr(&l[i][j], pool);
            out.set(i, j, simplify(pool.mul(vec![lij, root]), pool).value);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Jordan form
// ---------------------------------------------------------------------------

/// `(P, J)` with `M = P·J·P⁻¹`.
///
/// # The independence of the chains is checked, not assumed
///
/// When one eigenvalue owns two Jordan blocks of the same size — `J₂(3) ⊕
/// J₂(3)` is the smallest example — both chains are generated from the same
/// `ker (M − λI)^s`, and taking the same basis vector twice produces a `P`
/// whose columns are dependent. `J` is still right; `P` is not a basis, so
/// `M = P·J·P⁻¹` is false and `P⁻¹` does not exist. Until 3.10.1 that pair was
/// returned anyway, with nothing in it to indicate the problem.
///
/// Two things now stop it. Each chain's generator is chosen to be independent
/// of the chains already built (`chain_generator`), and the assembled `P` is
/// refused when its rank can be *proven* deficient — positive evidence only, so
/// a symbolic `P` whose rank is undecidable is still returned rather than newly
/// refused.
pub fn jordan_form(m: &Matrix, pool: &ExprPool) -> Result<(Matrix, Matrix), LinearAlgebraError> {
    if m.rows != m.cols {
        return Err(LinearAlgebraError::NonSquare);
    }
    let n = m.rows;
    let vals = eigen::eigenvalues(m, pool).map_err(map_eigen_err)?;
    let mut j_blocks: Vec<Matrix> = Vec::new();
    let mut p_cols: Vec<Matrix> = Vec::new();
    for (lambda, alg_m) in vals {
        let shifted = m_minus_lambda_scaled(m, lambda, pool);
        let mut ker_dims = vec![0usize];
        let mut pow = Matrix::identity(n, pool);
        for _k in 1..=alg_m {
            pow = pow
                .mul(&shifted, pool)
                .map_err(|_| LinearAlgebraError::KernelFailed)?;
            // `pow` is a power of `A − λI` for an eigenvalue λ, hence singular.
            ker_dims.push(
                kernel_column_basis(&pow, pool, KnownSingular::Yes)
                    .map_err(|f| kernel_failure_to_error(f, pool))?
                    .len(),
            );
        }
        let mut nu = vec![0usize; alg_m + 2];
        for s in 1..=alg_m {
            nu[s] = ker_dims[s] - ker_dims[s - 1];
        }
        let mut block_sizes = Vec::new();
        for s in 1..=alg_m {
            let nb = nu[s] - nu[s + 1];
            for _ in 0..nb {
                block_sizes.push(s);
            }
        }
        block_sizes.sort_by(|a, b| b.cmp(a));
        for &sz in &block_sizes {
            let mut nk = Matrix::identity(n, pool);
            for _ in 0..sz {
                nk = nk
                    .mul(&shifted, pool)
                    .map_err(|_| LinearAlgebraError::KernelFailed)?;
            }
            let bas = kernel_column_basis(&nk, pool, KnownSingular::Yes)
                .map_err(|f| kernel_failure_to_error(f, pool))?;
            let chain = chain_generator(&bas, &shifted, sz, &p_cols, pool)?;
            for col in chain {
                p_cols.push(col);
            }
            j_blocks.push(jordan_block_matrix(lambda, sz, pool));
        }
    }
    if p_cols.len() != n {
        return Err(LinearAlgebraError::KernelFailed);
    }
    let p = concatenate_columns(&p_cols, pool).map_err(|_| LinearAlgebraError::KernelFailed)?;
    // `M = P·J·P⁻¹` needs `P` to be a *basis*. A dependent column set makes the
    // identity false and `P⁻¹` non-existent, and neither `J` nor `P` looks
    // wrong on inspection. Refuse on proof of deficiency; an undecidable rank
    // is no evidence and is left alone.
    if matches!(rank(&p, pool), Ok(r) if r < n) {
        return Err(LinearAlgebraError::SingularTransform);
    }
    let j = block_diagonal(&j_blocks, pool)?;
    Ok((p, j))
}

/// The Jordan chain `[N^{sz−1}v, …, Nv, v]` for a generator `v` drawn from
/// `bas`, chosen so the chain is independent of `built`.
///
/// `bas` spans `ker N^{sz}`, which for a repeated block size contains the
/// previous block's chain as well; taking a fixed element of it — the last one,
/// as this did before 3.10.1 — hands back the same chain twice. Candidates are
/// tried in reverse order, so a matrix with one block per size picks exactly
/// what it picked before.
///
/// This is a greedy search over a spanning set, not a constructive proof: a
/// generator that exists only as a *combination* of `bas` elements is not
/// found, and [`jordan_form`]'s rank check is what keeps that a refusal rather
/// than a wrong answer.
fn chain_generator(
    bas: &[Matrix],
    shifted: &Matrix,
    sz: usize,
    built: &[Matrix],
    pool: &ExprPool,
) -> Result<Vec<Matrix>, LinearAlgebraError> {
    let mut first: Option<Vec<Matrix>> = None;
    for v_top in bas.iter().rev() {
        let mut chain = vec![v_top.clone()];
        let mut cur = v_top.clone();
        for _ in 1..sz {
            cur = shifted
                .mul(&cur, pool)
                .map_err(|_| LinearAlgebraError::KernelFailed)?;
            chain.push(cur.clone());
        }
        chain.reverse();
        // `v` generates a chain of length `sz` only if `N^{sz−1}v ≠ 0`.
        if chain[0]
            .entries()
            .iter()
            .all(|&e| zero_test::zero_status(pool, simplify(e, pool).value).is_proven_zero())
        {
            continue;
        }
        if first.is_none() {
            first = Some(chain.clone());
        }
        if extends_independently(built, &chain, pool) {
            return Ok(chain);
        }
    }
    // Nothing was *proven* independent. Hand back the first well-formed chain
    // and let the rank check decide: over symbolic entries "could not prove
    // independent" is routine and must not become a refusal on its own.
    first
        .or_else(|| bas.last().map(|v| vec![v.clone()]))
        .ok_or(LinearAlgebraError::KernelFailed)
}

/// Is `built ∪ chain` provably a set of `built.len() + chain.len()` independent
/// columns?
fn extends_independently(built: &[Matrix], chain: &[Matrix], pool: &ExprPool) -> bool {
    let cols: Vec<Matrix> = built.iter().chain(chain.iter()).cloned().collect();
    let Ok(m) = concatenate_columns(&cols, pool) else {
        return false;
    };
    matches!(rank(&m, pool), Ok(r) if r == cols.len())
}

fn jordan_block_matrix(lambda: ExprId, size: usize, pool: &ExprPool) -> Matrix {
    let mut m = Matrix::zeros(size, size, pool);
    for i in 0..size {
        m.set(i, i, lambda);
        if i + 1 < size {
            m.set(i, i + 1, pool.integer(1_i32));
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Rational canonical form
// ---------------------------------------------------------------------------

/// `(P, C)` with `M = P·C·P⁻¹` and `C` Frobenius companion block diagonal over ℚ.
///
/// # The pair is checked before it is returned
///
/// `M = P·C·P⁻¹` is a claim about *both* matrices, and until 3.10.1 neither
/// half was verified. Two independent defects shipped behind that:
///
/// * `companion_matrix` wrote the coefficients down the last **row** instead
///   of the last **column**, which for `d ≥ 2` also overwrote a subdiagonal
///   `1`. On `diag(1, 2)` it returned `C = [[0,0],[−2,3]]`, whose determinant
///   is `0` against `det M = 2` — not similar to `M`, not even the same rank.
/// * `P` was assembled from Krylov chains on unit vectors whose only
///   admissibility test was that consecutive chain elements are not
///   *proportional*. That is far weaker than the requirement (each chain
///   independent, and the chains jointly spanning), so `P` was routinely not a
///   basis.
///
/// Both are fixed, and neither can recur silently: `M·P = P·C` is now proven
/// entrywise and `P` is proven invertible before the pair leaves this function.
/// A pair that fails is refused with [`LinearAlgebraError::SingularTransform`]
/// (`E-LINALG-008`) rather than returned. The gate is placed beside the
/// construction rather than inside it — it recomputes nothing the construction
/// chose, it only multiplies out the identity being claimed.
pub fn rational_canonical_form(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<(Matrix, Matrix), LinearAlgebraError> {
    if m.rows != m.cols {
        return Err(LinearAlgebraError::NonSquare);
    }
    let poly_m = lambda_identity_minus_m_poly(m, pool)?;
    let (s, _u, _v) = smith_form_poly(&poly_m);
    let factors = invariant_factors_from_smith(&s)?;
    let c = companion_block_diagonal(&factors, pool)?;
    let p = frobenius_p_from_cyclic_vectors(m, &factors, pool)?;
    confirm_similarity(m, &p, &c, pool)?;
    Ok((p, c))
}

/// Prove `M·P = P·C` entrywise and `P` invertible, or refuse.
///
/// Both halves are needed: `M·P = P·C` with a singular `P` is satisfied by
/// `P = 0`, and an invertible `P` alone says nothing about `C`. Together they
/// are exactly the statement `M = P·C·P⁻¹`.
///
/// Only *proven* equality counts. An entry whose vanishing is undecidable is
/// not evidence for the identity, so it refuses — the inputs that reach here
/// have rational entries (`expr_to_rat_uni_poly` has already rejected anything
/// else), where the zero test is exact and this never triggers spuriously.
fn confirm_similarity(
    m: &Matrix,
    p: &Matrix,
    c: &Matrix,
    pool: &ExprPool,
) -> Result<(), LinearAlgebraError> {
    let n = m.rows;
    if p.rows != n || p.cols != n || c.rows != n || c.cols != n {
        return Err(LinearAlgebraError::SingularTransform);
    }
    let lhs = m
        .mul(p, pool)
        .map_err(|_| LinearAlgebraError::SingularTransform)?;
    let rhs = p
        .mul(c, pool)
        .map_err(|_| LinearAlgebraError::SingularTransform)?;
    for i in 0..n {
        for j in 0..n {
            let diff = simplify_expanded(
                pool.add(vec![
                    lhs.get(i, j),
                    pool.mul(vec![pool.integer(-1_i32), rhs.get(i, j)]),
                ]),
                pool,
            )
            .value;
            if !zero_test::zero_status(pool, diff).is_proven_zero() {
                return Err(LinearAlgebraError::SingularTransform);
            }
        }
    }
    if rank(p, pool)? != n {
        return Err(LinearAlgebraError::SingularTransform);
    }
    Ok(())
}

fn fresh_frobenius_lambda(pool: &ExprPool) -> ExprId {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let k = SEQ.fetch_add(1, Ordering::Relaxed);
    pool.symbol(format!("__frobenius_lambda_{k}"), Domain::Complex)
}

fn lambda_identity_minus_m_poly(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<PolyMatrixQ, LinearAlgebraError> {
    let _lam = fresh_frobenius_lambda(pool);
    let n = m.rows;
    let x = RatUniPoly::x();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let entry = if i == j {
                let c = expr_to_rat_uni_poly(m.get(i, j), pool)?;
                (&x - &c).trim()
            } else {
                let c = expr_to_rat_uni_poly(m.get(i, j), pool)?;
                (-&c).trim()
            };
            row.push(entry);
        }
        rows.push(row);
    }
    PolyMatrixQ::from_nested(rows).map_err(|_| unsupported_field())
}

fn expr_to_rat_uni_poly(e: ExprId, pool: &ExprPool) -> Result<RatUniPoly, LinearAlgebraError> {
    match pool.get(e) {
        ExprData::Integer(n) => Ok(RatUniPoly::constant(Rational::from((n.0.clone(), 1)))),
        ExprData::Rational(r) => Ok(RatUniPoly::constant(r.0.clone())),
        ExprData::Add(args) => {
            let mut acc = RatUniPoly::zero();
            for a in args {
                acc = (&acc + &expr_to_rat_uni_poly(a, pool)?).trim();
            }
            Ok(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = RatUniPoly::one();
            for a in args {
                acc = (&acc * &expr_to_rat_uni_poly(a, pool)?).trim();
            }
            Ok(acc)
        }
        _ => Err(LinearAlgebraError::NonRationalEntry),
    }
}

fn invariant_factors_from_smith(s: &PolyMatrixQ) -> Result<Vec<RatUniPoly>, LinearAlgebraError> {
    let n = s.rows.min(s.cols);
    let mut facs = Vec::new();
    for i in 0..n {
        let p = s.get(i, i).clone().trim();
        if !p.is_zero() && p.degree() > 0 {
            facs.push(p);
        }
    }
    if facs.is_empty() {
        return Err(unsupported_field());
    }
    Ok(facs)
}

/// The Frobenius companion matrix of a monic `f = x^d + a_{d−1}x^{d−1} + … + a₀`:
/// `1`s on the **subdiagonal** and `−a₀ … −a_{d−1}` down the **last column**.
///
/// The pairing with [`frobenius_p_from_cyclic_vectors`] is what fixes the
/// orientation. With `P = [v, Mv, …, M^{d−1}v]`,
/// `M·P = [Mv, …, M^{d−1}v, M^d v]`, and `M^d v = −Σ aⱼ Mʲ v`, so `M·P = P·C`
/// forces `C·e_j = e_{j+1}` for `j < d−1` (the subdiagonal `1`s) and
/// `C·e_{d−1} = (−a₀, …, −a_{d−1})ᵀ` — a *column*.
///
/// Before 3.10.1 the coefficients were written down the last **row**, which for
/// `d ≥ 2` also overwrote the subdiagonal `1` at `(d−1, d−2)`. The result was
/// not similar to `M` and generally not even the same rank: on `diag(1, 2)`,
/// whose single invariant factor is `x²−3x+2`, it produced `[[0,0],[−2,3]]`
/// with `det = 0` against `det M = 2`.
fn companion_matrix(f: &RatUniPoly, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    let f = f.clone().trim();
    if f.degree() <= 0 {
        return Err(unsupported_field());
    }
    let d = f.degree() as usize;
    let coeffs = f.coeffs.clone();
    // `f` comes off a Smith-form diagonal and need not be monic; the companion
    // form is defined for the monic associate, which has the same roots and the
    // same ideal.
    let lead = coeffs[d].clone();
    if lead == 0 {
        return Err(unsupported_field());
    }
    let mut c = Matrix::zeros(d, d, pool);
    for i in 0..d - 1 {
        c.set(i + 1, i, pool.integer(1_i32));
    }
    for j in 0..d {
        let a = coeffs[j].clone() / lead.clone();
        let coeff = rational_expr(&a, pool);
        c.set(
            j,
            d - 1,
            simplify(pool.mul(vec![pool.integer(-1_i32), coeff]), pool).value,
        );
    }
    Ok(c)
}

fn companion_block_diagonal(
    factors: &[RatUniPoly],
    pool: &ExprPool,
) -> Result<Matrix, LinearAlgebraError> {
    let mut blocks = Vec::new();
    for f in factors {
        blocks.push(companion_matrix(f, pool)?);
    }
    block_diagonal(&blocks, pool)
}

fn block_diagonal(blocks: &[Matrix], pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    let total: usize = blocks.iter().map(|b| b.rows).sum();
    let mut out = Matrix::zeros(total, total, pool);
    let mut off = 0usize;
    for b in blocks {
        for i in 0..b.rows {
            for j in 0..b.cols {
                out.set(off + i, off + j, b.get(i, j));
            }
        }
        off += b.rows;
    }
    Ok(out)
}

/// `P = [v₁, Mv₁, …, M^{d₁−1}v₁ | v₂, …, M^{d₂−1}v₂ | …]`, one Krylov chain per
/// invariant factor.
///
/// # Choosing a generator for factor `f`
///
/// `vᵢ` has to satisfy two things, and the old code checked neither: its
/// annihilator must be exactly `fᵢ`, and its chain must be independent of the
/// chains already placed. Both are obtained here rather than hoped for.
///
/// *Annihilator.* With `g` the largest invariant factor — the minimal
/// polynomial, which every other factor divides — set `qᵢ = g / fᵢ`. Then for
/// **any** `w`, `v = qᵢ(M)·w` satisfies `fᵢ(M)v = g(M)w = 0`, so `ann(v)`
/// divides `fᵢ` for free. And if the chain `v, …, M^{dᵢ−1}v` is independent
/// then `deg ann(v) ≥ dᵢ = deg fᵢ`, which forces `ann(v) = fᵢ` exactly. So the
/// independence test does double duty and no separate order computation is
/// needed.
///
/// *Independence.* Tested by rank, against everything already placed, rather
/// than by the old `columns_proportional` — which only asked whether two
/// *consecutive* chain elements are parallel and is satisfied by wildly
/// dependent chains.
///
/// The seed search is a finite heuristic (unit vectors, then pairwise sums), so
/// it can fail to find a generator that exists. That is a refusal, never a
/// wrong answer: [`confirm_similarity`] proves the assembled pair before it is
/// returned.
fn frobenius_p_from_cyclic_vectors(
    m: &Matrix,
    factors: &[RatUniPoly],
    pool: &ExprPool,
) -> Result<Matrix, LinearAlgebraError> {
    let n = m.rows;
    let Some(g) = factors.last() else {
        return Err(LinearAlgebraError::KernelFailed);
    };
    let mut cols: Vec<Matrix> = Vec::with_capacity(n);
    for f in factors {
        let f = f.clone().trim();
        if f.degree() <= 0 {
            return Err(unsupported_field());
        }
        let d = f.degree() as usize;
        let (q, r) = RatUniPoly::div_rem(g, &f);
        if !r.trim().is_zero() {
            // Not a divisibility chain: the Smith diagonal is not a list of
            // invariant factors, so nothing below would mean what it claims.
            return Err(unsupported_field());
        }
        let qm = rat_poly_at_matrix(&q, m, pool)?;
        let chain = cyclic_chain_for_factor(m, &qm, d, n, &cols, pool)?;
        cols.extend(chain);
    }
    if cols.len() != n {
        return Err(LinearAlgebraError::KernelFailed);
    }
    concatenate_columns(&cols, pool).map_err(|_| LinearAlgebraError::KernelFailed)
}

/// `p(M)` for `p ∈ ℚ[x]`, by Horner over matrices.
fn rat_poly_at_matrix(
    p: &RatUniPoly,
    m: &Matrix,
    pool: &ExprPool,
) -> Result<Matrix, LinearAlgebraError> {
    let n = m.rows;
    let p = p.clone().trim();
    let mut acc = Matrix::zeros(n, n, pool);
    for coeff in p.coeffs.iter().rev() {
        let scaled = acc
            .mul(m, pool)
            .map_err(|_| LinearAlgebraError::KernelFailed)?;
        let c = rational_expr(coeff, pool);
        let term = Matrix::identity(n, pool).scale(c, pool);
        acc = scaled
            .add(&term, pool)
            .map_err(|_| LinearAlgebraError::KernelFailed)?
            .simplify_entries(pool);
    }
    Ok(acc)
}

/// The chain `v, Mv, …, M^{d−1}v` for `v = qm·w`, `w` drawn from a finite seed
/// list, chosen so the chain is provably independent of `built`.
fn cyclic_chain_for_factor(
    m: &Matrix,
    qm: &Matrix,
    d: usize,
    n: usize,
    built: &[Matrix],
    pool: &ExprPool,
) -> Result<Vec<Matrix>, LinearAlgebraError> {
    let mut seeds: Vec<Matrix> = Vec::new();
    for c in 0..n {
        seeds.push(unit_column_vector(c, n, pool)?);
    }
    for i in 0..n {
        for j in (i + 1)..n {
            seeds.push(sum_unit_columns(i, j, n, pool)?);
        }
    }
    for w in &seeds {
        let v = qm
            .mul(w, pool)
            .map_err(|_| LinearAlgebraError::KernelFailed)?
            .simplify_entries(pool);
        let mut chain = vec![v.clone()];
        let mut cur = v;
        for _ in 1..d {
            cur = m
                .mul(&cur, pool)
                .map_err(|_| LinearAlgebraError::KernelFailed)?
                .simplify_entries(pool);
            chain.push(cur.clone());
        }
        if extends_independently(built, &chain, pool) {
            return Ok(chain);
        }
    }
    Err(LinearAlgebraError::KernelFailed)
}

fn sum_unit_columns(
    i: usize,
    j: usize,
    n: usize,
    pool: &ExprPool,
) -> Result<Matrix, LinearAlgebraError> {
    let zero = pool.integer(0_i32);
    let one = pool.integer(1_i32);
    let rows: Vec<Vec<ExprId>> = (0..n)
        .map(|r| vec![if r == i || r == j { one } else { zero }])
        .collect();
    Matrix::new(rows).map_err(|_| LinearAlgebraError::KernelFailed)
}

fn unit_column_vector(col: usize, n: usize, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    let zero = pool.integer(0_i32);
    let one = pool.integer(1_i32);
    let rows: Vec<Vec<ExprId>> = (0..n)
        .map(|r| vec![if r == col { one } else { zero }])
        .collect();
    Matrix::new(rows).map_err(|_| LinearAlgebraError::KernelFailed)
}

// ---------------------------------------------------------------------------
// Minimal polynomial
// ---------------------------------------------------------------------------

/// `(minimal_poly(λ), λ)` using the same fresh λ as the characteristic polynomial.
pub fn minimal_polynomial(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<(ExprId, ExprId), LinearAlgebraError> {
    if m.rows != m.cols {
        return Err(LinearAlgebraError::NonSquare);
    }
    let (char_e, lam) = characteristic_polynomial_lambda_minus_m(m, pool).map_err(map_eigen_err)?;
    let uni = UniPoly::from_symbolic_clear_denoms(char_e, lam, pool)
        .map_err(LinearAlgebraError::CharPolyConversion)?;
    let fac = factor_univariate_z(&uni).map_err(LinearAlgebraError::Factorization)?;
    let mut divisors = all_divisors_from_factors(
        &fac.factors
            .iter()
            .map(|(p, e)| (p.clone(), *e as usize))
            .collect::<Vec<_>>(),
    );
    divisors.sort_by_key(|p| p.degree());
    for cand in divisors {
        if matrix_annihilated_by_uni(m, &cand, pool)? {
            return Ok((uni_poly_to_expr(&cand, lam, pool), lam));
        }
    }
    Err(LinearAlgebraError::KernelFailed)
}

fn all_divisors_from_factors(factors: &[(UniPoly, usize)]) -> Vec<UniPoly> {
    let Some((first, _)) = factors.first() else {
        return Vec::new();
    };
    let mut out = vec![UniPoly::constant(first.var, 1)];
    for (base, exp) in factors {
        let mut next = Vec::new();
        for d in &out {
            let mut cur = d.clone();
            for _e in 0..=*exp {
                next.push(cur.clone());
                if _e < *exp {
                    cur = cur.mul(base.clone());
                }
            }
        }
        out = next;
    }
    out
}

fn matrix_annihilated_by_uni(
    m: &Matrix,
    p: &UniPoly,
    pool: &ExprPool,
) -> Result<bool, LinearAlgebraError> {
    let n = m.rows;
    let coeffs = p.coefficients();
    let mut acc = Matrix::zeros(n, n, pool);
    // Invariant: `pow == M^deg` at the top of iteration `deg`. Advancing it is
    // therefore unconditional on the coefficient — a zero coefficient skips the
    // *term*, not the power. Before 3.10.1 a zero coefficient advanced `pow`
    // only `if deg > 0`, so a vanishing **constant** term left `pow` at `M⁰`
    // for the `deg = 1` iteration and every later power was one too low: `p(M)`
    // was evaluated as `(p/λ)(M)`.
    //
    // A zero constant term is exactly the case `λ = 0 ∈ spec(M)`, so this was
    // wrong for every singular matrix that got this far. It rejected the true
    // minimal polynomial and accepted `λ·`(that polynomial) instead:
    // `minimal_polynomial(0₂ₓ₂)` returned `λ²`, whose defining property the
    // zero matrix does not need, and `J₂(0) ⊕ J₂(0)` returned `λ³` for a
    // matrix annihilated by `λ²`.
    let mut pow = Matrix::identity(n, pool);
    for (deg, coeff) in coeffs.iter().enumerate() {
        if !coeff.is_zero() {
            let c = pool.integer(coeff.clone());
            let term = pow.scale(c, pool);
            acc = acc
                .add(&term, pool)
                .map_err(|_| LinearAlgebraError::KernelFailed)?;
        }
        if deg + 1 < coeffs.len() {
            pow = pow
                .mul(m, pool)
                .map_err(|_| LinearAlgebraError::KernelFailed)?;
        }
    }
    for e in acc.entries() {
        let entry = simplify(*e, pool).value;
        match zero_test::zero_status(pool, entry) {
            zero_test::ZeroStatus::Zero => {}
            zero_test::ZeroStatus::NonZero => return Ok(false),
            // "p(M) might be 0" must not be reported as "p does not annihilate
            // M": that would return a non-minimal polynomial as the minimal one.
            zero_test::ZeroStatus::Unknown => return Err(inconclusive(pool, entry)),
        }
    }
    Ok(true)
}

fn uni_poly_to_expr(p: &UniPoly, lam: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for (deg, coeff) in p.coefficients().iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let c = pool.integer(coeff.clone());
        let term = if deg == 0 {
            c
        } else if deg == 1 {
            simplify(pool.mul(vec![c, lam]), pool).value
        } else {
            simplify(
                pool.mul(vec![c, pool.pow(lam, pool.integer(deg as i32))]),
                pool,
            )
            .value
        };
        terms.push(term);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        // `simplify` does not collapse a one-element `Add`, so wrapping here
        // would hand back `Add([λ])` where the answer is `λ`: equal in value,
        // distinct as an `ExprId`, and not equal to the `λ` returned alongside
        // it. `minimal_polynomial` of a zero matrix is exactly that shape.
        1 => simplify(terms[0], pool).value,
        _ => simplify(pool.add(terms), pool).value,
    }
}

// ---------------------------------------------------------------------------
// Matrix exponential
// ---------------------------------------------------------------------------

/// `e^M`.
///
/// # Method
///
/// Three routes, in order:
///
/// 1. a diagonal `M` is exponentiated entrywise;
/// 2. a `M` that [`diagonalize`](crate::matrix::diagonalize) accepts is
///    `P·e^{D}·P⁻¹`;
/// 3. everything else — every **defective** matrix, and every matrix whose
///    eigenbasis the eigenvector machinery could not assemble — goes through
///    **Putzer's algorithm** (`crate::matrix::putzer`), which needs the
///    eigenvalues and nothing else: no eigenvectors, no similarity transform,
///    and no special case for defectiveness.
///
/// Route 3 is why the code changed. Until 3.10.1 a non-diagonalizable
/// `M` went through `jordan_form` and a per-block `e^{J}`, and the block
/// formula was `e^λ·λ^k/k!` where the truth is `e^λ·N^k/k! = e^λ/k!`. It read
/// the nilpotent power as a power of the eigenvalue, so every defective matrix
/// came back wrong, silently:
///
/// ```text
/// exp([[0,1],[0,0]])  →  [[1,0],[0,1]]        truth [[1,1],[0,1]]
/// exp([[2,1],[0,2]])  →  [[e², 2e²],[0, e²]]  truth [[e², e²],[0, e²]]
/// ```
///
/// The same block loop also mis-detected block size — it compared `J[i][i+sz]`
/// against 1 where the superdiagonal is `J[i+sz−1][i+sz]` — so a 3×3 Jordan
/// block was split into a 2×2 and a 1×1. Neither defect can recur: there is no
/// Jordan form on this path any more, and the answer is checked before it is
/// returned.
///
/// # What is guaranteed
///
/// Every answer is checked against an independently computed `e^A` before it is
/// returned ([`crate::matrix::exp_gate`]), and two things are refused rather
/// than guessed, both through [`LinearAlgebraError::UnsupportedField`] with the
/// specific cause available from
/// [`take_matrix_exp_refusal`](crate::matrix::take_matrix_exp_refusal)
/// (`E-LINALG-011`):
///
/// * an eigenvalue list that could not be *positively confirmed* as the
///   spectrum ([`crate::matrix::spectrum`]) — Putzer's expansion is built
///   entirely on it, and a wrong list produces a plausible matrix rather than
///   an error;
/// * a candidate that failed the standing check.
///
/// # What is reported rather than refused
///
/// An eigenvalue gap `λ_i − λ_j` that is neither provably zero nor settled
/// non-zero is *divided by*, because the generic expansion is a correct answer
/// on `λ_i ≠ λ_j` and that is the reading every CAS gives — but the hypothesis
/// is recorded and available from
/// [`take_matrix_exp_side_conditions`](crate::matrix::take_matrix_exp_side_conditions).
/// An open condition is reported; only a condition settled the wrong way is a
/// refusal.
pub fn matrix_exponential(m: &Matrix, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    exp_gate::forget_refusal();
    if m.rows != m.cols {
        return Err(LinearAlgebraError::NonSquare);
    }
    // A diagonal matrix (possibly with free-symbol entries) exponentiates entrywise:
    // exp(diag(d₀, …, dₙ)) = diag(e^{d₀}, …, e^{dₙ}). Short-circuit so symbolic diagonal /
    // decoupled state matrices succeed without invoking the spectrum machinery at all.
    if is_diagonal(m, pool) {
        return diagonal_matrix_exp(m, pool);
    }

    // The diagonalizable route, unchanged: it was never the broken one, it
    // keeps symbolic answers in the `P·e^{D}·P⁻¹` form callers already read,
    // and it settles the `λ_i − λ_j` question by construction — a matrix with a
    // full eigenbasis needs no gap decision at all. It is checked by the same
    // gate as the Putzer route.
    if let Ok((p, d)) = eigen::diagonalize(m, pool) {
        let exp_d = diagonal_matrix_exp(&d, pool)?;
        let inv_p = matrix_inverse(&p, pool).map_err(|_| LinearAlgebraError::SingularTransform)?;
        let candidate = p
            .mul(&exp_d, pool)
            .map_err(|_| LinearAlgebraError::KernelFailed)?
            .mul(&inv_p, pool)
            .map_err(|_| LinearAlgebraError::KernelFailed)?;
        return gated(m, candidate, pool);
    }

    let lambdas = spectrum_with_multiplicity(m, pool)?;
    // Putzer turns a wrong eigenvalue list into a plausible matrix rather than
    // into an error, so the list is confirmed *before* anything is built on it.
    match spectrum::confirm_spectrum(m, &lambdas, pool) {
        Ok(check) if check.is_confirmed() => {}
        Ok(_) => return Err(unconfirmed_spectrum(m.rows)),
        Err(refusal) => {
            spectrum::record_refusal(refusal);
            return Err(LinearAlgebraError::UnsupportedIrreducibleDegree { degree: m.rows });
        }
    }

    let one = pool.integer(1_i32);
    let mut gaps = putzer::RecordedGaps::default();
    // `None` only for `n = 0`, which `NonSquare` above has already excluded for
    // every shape a caller can build.
    let candidate = putzer::matrix_exponential(m, &lambdas, one, &mut gaps, pool)
        .ok_or(LinearAlgebraError::NonSquare)?;
    let out = gated(m, candidate.simplify_entries(pool), pool)?;
    exp_gate::record_side_conditions(gaps.assumed_nonzero);
    Ok(out)
}

/// Return `candidate` only if the standing gate does not refute it.
///
/// `ExpCheck::Unevaluated` is no information rather than a failure, for the
/// same reason it is in [`crate::matrix::spectrum`]: an expression the numeric
/// evaluator does not model is a property of the expression, not evidence
/// against the answer.
fn gated(m: &Matrix, candidate: Matrix, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    match exp_gate::confirm_matrix_exponential(m, &candidate, pool) {
        Ok(_) => Ok(candidate),
        Err(refusal) => {
            exp_gate::record_refusal(refusal.reason(), refusal.to_string());
            Err(LinearAlgebraError::UnsupportedField)
        }
    }
}

/// The `n` eigenvalues of `m`, listed with multiplicity.
///
/// A triangular `m` is read off its diagonal directly. That is not only a
/// shortcut: for a symbolic triangular matrix the general routine would build
/// `det(λI − M)`, fail to clear it to ℤ\[λ\], and fall through to Cardano,
/// returning nested radicals for what the diagonal states in three symbols.
fn spectrum_with_multiplicity(
    m: &Matrix,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, LinearAlgebraError> {
    let n = m.rows;
    if is_triangular(m, pool) {
        return Ok((0..n).map(|i| simplify(m.get(i, i), pool).value).collect());
    }
    let eigs = eigen::eigenvalues(m, pool).map_err(map_eigen_err)?;
    let mut out = Vec::with_capacity(n);
    for (lam, mult) in eigs {
        for _ in 0..mult {
            out.push(simplify(lam, pool).value);
        }
    }
    if out.len() != n {
        // The characteristic polynomial did not split; `eigenvalues` returned a
        // partial list, which Putzer cannot use.
        return Err(LinearAlgebraError::UnsupportedIrreducibleDegree { degree: n });
    }
    Ok(out)
}

/// True iff every entry strictly below — or every entry strictly above — the
/// diagonal is *proven* zero.
fn is_triangular(m: &Matrix, pool: &ExprPool) -> bool {
    let n = m.rows;
    let proven_zero = |i: usize, j: usize| {
        zero_test::zero_status(pool, simplify(m.get(i, j), pool).value).is_proven_zero()
    };
    let lower = (0..n).all(|i| (0..i).all(|j| proven_zero(i, j)));
    let upper = (0..n).all(|i| ((i + 1)..n).all(|j| proven_zero(i, j)));
    lower || upper
}

fn unconfirmed_spectrum(n: usize) -> LinearAlgebraError {
    exp_gate::record_refusal(
        exp_gate::ExpRefusalReason::SpectrumUnconfirmed,
        exp_gate::spectrum_unconfirmed(n),
    );
    LinearAlgebraError::UnsupportedField
}

/// True iff every off-diagonal entry is *proven* zero.
///
/// A fast path, not a claim: an undecided off-diagonal entry answers `false`
/// and `matrix_exponential` falls back to the Jordan route, which is correct
/// for diagonal matrices too.
fn is_diagonal(m: &Matrix, pool: &ExprPool) -> bool {
    if m.rows != m.cols {
        return false;
    }
    for r in 0..m.rows {
        for c in 0..m.cols {
            let entry = simplify(m.get(r, c), pool).value;
            if r != c && !zero_test::zero_status(pool, entry).is_proven_zero() {
                return false;
            }
        }
    }
    true
}

fn diagonal_matrix_exp(d: &Matrix, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    let n = d.rows;
    let mut out = Matrix::zeros(n, n, pool);
    for i in 0..n {
        out.set(
            i,
            i,
            simplify(pool.func("exp", vec![d.get(i, i)]), pool).value,
        );
    }
    Ok(out)
}

#[cfg(test)]
fn apply_row_permutation(m: &Matrix, perm: &[usize]) -> Matrix {
    let rows: Vec<Vec<ExprId>> = perm.iter().map(|&r| m.row(r)).collect();
    Matrix::new(rows).expect("row permutation")
}

// ---------------------------------------------------------------------------
// Matrix inverse (for similarity transforms)
// ---------------------------------------------------------------------------

thread_local! {
    /// The determinant the most recent successful [`matrix_inverse`] divided by
    /// without being able to settle it at a *point*.
    static INVERSE_SIDE_CONDITIONS: std::cell::RefCell<Vec<ExprId>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The hypotheses the matrix returned by the most recent [`matrix_inverse`]
/// call on this thread rests on, as [`crate::deriv::SideCondition::NonZero`].
///
/// Exactly one thing lands here: **`det A ≠ 0`**, for a symbolic `A` whose
/// determinant was proven not to be the zero *function* but cannot be proven
/// non-zero at a particular parameter value — because that is not a question
/// about `A` at all, it is a question about the parameters, and the caller is
/// the only one who can answer it. `Matrix([[a,b],[c,d]]).inverse()` is
/// `adj/det` **for `ad − bc ≠ 0`**; on the locus `ad = bc` the matrix has no
/// inverse and the expression has no value.
///
/// Empty means the determinant is a non-zero rational constant — the
/// hypothesis was *discharged*, not skipped.
///
/// # Why this is auditability and not soundness
///
/// The returned entries are literally `±minor · det⁻¹`, so evaluating one on
/// the singular locus raises rather than returning a number, and no
/// cancellation can hide that: over the local ring at an irreducible factor
/// `p` of `det`, the Smith form `diag(p^{e₁} … p^{eₙ})` gives `det` order
/// `Σeᵢ` and the adjugate order `Σeᵢ − eₙ`, so the *reduced* denominator still
/// vanishes to order `eₙ ≥ 1` on every component of `{det = 0}`. The condition
/// therefore cannot be silently lost. What it can be is *unstated*, which is
/// what this channel fixes — a controls user inverting a symbolic plant needs
/// the genericity assumption in machine-readable form, not implied by the
/// shape of an expression.
///
/// # Why out of band
///
/// [`matrix_inverse`] returns a bare [`Matrix`], a public type; it cannot grow
/// a conditions field without a major semver break. Same treatment as
/// [`crate::matrix::take_matrix_exp_side_conditions`] and
/// [`crate::solver::take_solve_side_conditions`].
///
/// Consuming, so one call's hypotheses cannot be read as a later call's.
pub fn take_matrix_inverse_side_conditions() -> Vec<crate::deriv::SideCondition> {
    INVERSE_SIDE_CONDITIONS.with(|c| {
        std::mem::take(&mut *c.borrow_mut())
            .into_iter()
            .map(crate::deriv::SideCondition::NonZero)
            .collect()
    })
}

pub fn matrix_inverse(m: &Matrix, pool: &ExprPool) -> Result<Matrix, MatrixError> {
    // Cleared unconditionally on entry, including on the early returns below:
    // a stale condition read as this call's would be a hypothesis attached to
    // the wrong answer, which is worse than none.
    INVERSE_SIDE_CONDITIONS.with(|c| c.borrow_mut().clear());
    if m.rows != m.cols {
        return Err(MatrixError::NotSquare);
    }
    let n = m.rows;
    let Some(a) = matrix_to_rational_grid(m, pool) else {
        return symbolic_inverse(m, pool);
    };
    let mut aug: Vec<Vec<Rational>> = a
        .into_iter()
        .map(|mut row| {
            row.resize(2 * n, Rational::from(0));
            row
        })
        .collect();
    for i in 0..n {
        aug[i][n + i] = Rational::from(1);
    }
    for col in 0..n {
        let mut piv = None;
        for r in col..n {
            if aug[r][col] != 0 {
                piv = Some(r);
                break;
            }
        }
        let Some(pr) = piv else {
            // Rational entries: zero-testing is exact here, so this is a proven
            // singularity and never a refusal — see `singular`.
            return Err(singular());
        };
        if pr != col {
            aug.swap(pr, col);
        }
        let inv = Rational::from(1) / aug[col][col].clone();
        for j in 0..2 * n {
            aug[col][j] *= inv.clone();
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = aug[r][col].clone();
            if f == 0 {
                continue;
            }
            for j in 0..2 * n {
                let pivot_val = aug[col][j].clone();
                aug[r][j] -= f.clone() * pivot_val;
            }
        }
    }
    let inv_grid: Vec<Vec<Rational>> = aug.into_iter().map(|row| row[n..].to_vec()).collect();
    Ok(rational_grid_to_matrix(&inv_grid, pool))
}

/// Symbolic matrix inverse for matrices containing non-rational entries.
///
/// Uses the adjugate formula: `inv[i][j] = (-1)^(i+j) · det(minor_ji) / det(A)`,
/// where `minor_ji` removes row `j` and column `i` (note the transpose). The
/// symbolic determinant engine (`Matrix::det`) handles arbitrary entries, so this
/// path supports transfer functions `C(sI−A)⁻¹B+D`, symbolic mass matrices, etc.
///
/// If `det(A)` is proven zero the matrix is genuinely singular; if it can be
/// proven neither zero nor non-zero, no inverse is reported either, because the
/// adjugate formula divides by it. Both are
/// [`MatrixError::SingularMatrix`], whose text states that disjunction; the
/// second case additionally records a
/// [`ZeroTestRefusal`](crate::matrix::ZeroTestRefusal) carrying `E-MAT-004`.
fn symbolic_inverse(m: &Matrix, pool: &ExprPool) -> Result<Matrix, MatrixError> {
    let n = m.rows;
    if n == 0 {
        return Ok(Matrix::zeros(0, 0, pool));
    }
    // Expand the determinant into canonical polynomial form so that the shared
    // `1/det` factor in the resulting entries cancels cleanly against expanded
    // cofactor numerators (e.g. so A·A⁻¹ collapses to the identity on simplify).
    let det = simplify_expanded(m.det(pool)?, pool).value;
    match zero_test::zero_status(pool, det) {
        zero_test::ZeroStatus::Zero => return Err(singular()),
        zero_test::ZeroStatus::NonZero => {
            // Proven not the zero *function*. That licenses `adj/det`, and it
            // is all it licenses: on `{det = 0}` there is no inverse. Record
            // the hypothesis unless the determinant is a non-zero rational
            // constant, where there is no locus and nothing to assume.
            if expr_to_rational_strict(det, pool).is_none() {
                INVERSE_SIDE_CONDITIONS.with(|c| c.borrow_mut().push(det));
            }
        }
        zero_test::ZeroStatus::Unknown => {
            zero_test::record_refusal(pool, det, zero_test::RefusalSite::Determinant);
            return Err(MatrixError::SingularMatrix);
        }
    }
    let inv_det = simplify(pool.pow(det, pool.integer(-1_i32)), pool).value;

    let mut rows: Vec<Vec<ExprId>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<ExprId> = Vec::with_capacity(n);
        for j in 0..n {
            // Transposed cofactor: minor removes row j and column i.
            let minor = m.minor(j, i);
            let minor_det = if n == 1 {
                pool.integer(1_i32)
            } else {
                simplify_expanded(minor.det(pool)?, pool).value
            };
            let sign = if (i + j) % 2 == 0 {
                pool.integer(1_i32)
            } else {
                pool.integer(-1_i32)
            };
            let cofactor = pool.mul(vec![sign, minor_det, inv_det]);
            row.push(simplify(cofactor, pool).value);
        }
        rows.push(row);
    }
    Matrix::new(rows).map_err(|_| singular())
}

/// [`MatrixError::SingularMatrix`] for a *proven* singularity.
///
/// Clears any recorded zero-test refusal so a genuinely singular matrix is
/// never reported with the undecided-determinant code `E-MAT-004`; see
/// [`inconclusive`] for the other half of the arrangement.
fn singular() -> MatrixError {
    zero_test::forget_refusal();
    MatrixError::SingularMatrix
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn map_eigen_err(e: eigen::EigenError) -> LinearAlgebraError {
    match e {
        eigen::EigenError::NonSquare => LinearAlgebraError::NonSquare,
        eigen::EigenError::CharPolyConversion(c) => LinearAlgebraError::CharPolyConversion(c),
        eigen::EigenError::Factorization(f) => LinearAlgebraError::Factorization(f),
        eigen::EigenError::UnsupportedIrreducibleDegree { degree } => {
            LinearAlgebraError::UnsupportedIrreducibleDegree { degree }
        }
        eigen::EigenError::KernelComputationFailed
        | eigen::EigenError::NonDiagonalizable
        | eigen::EigenError::SingularModalMatrix => LinearAlgebraError::KernelFailed,
    }
}

fn matrix_to_rational_grid(m: &Matrix, pool: &ExprPool) -> Option<Vec<Vec<Rational>>> {
    let mut g = Vec::with_capacity(m.rows);
    for r in 0..m.rows {
        let mut row = Vec::with_capacity(m.cols);
        for c in 0..m.cols {
            row.push(expr_to_rational_strict(m.get(r, c), pool)?);
        }
        g.push(row);
    }
    Some(g)
}

fn expr_to_rational_strict(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(Rational::from((n.0.clone(), 1))),
        ExprData::Rational(r) => Some(r.0.clone()),
        ExprData::Add(args) => {
            let mut acc = Rational::from(0);
            for a in args {
                acc += expr_to_rational_strict(a, pool)?;
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = Rational::from(1);
            for a in args {
                acc *= expr_to_rational_strict(a, pool)?;
            }
            Some(acc)
        }
        _ => None,
    }
}

fn rational_grid_to_matrix(grid: &[Vec<Rational>], pool: &ExprPool) -> Matrix {
    let rows: Vec<Vec<ExprId>> = grid
        .iter()
        .map(|row| row.iter().map(|r| rational_expr(r, pool)).collect())
        .collect();
    Matrix::new(rows).expect("rational grid")
}

/// A rational as an expression, as an `Integer` node when its denominator is 1.
///
/// `ExprPool::rational` interns `ExprData::Rational` whatever the denominator
/// is, and `simplify` does not demote it, so `pool.rational(1, 1)` is a node
/// that is *equal in value* to `pool.integer(1)` and *distinct from it* as an
/// `ExprId`. Every structural comparison against a literal then fails: the unit
/// diagonal of `L` out of [`lu_decomposition`] printed as `1/1`, and an
/// identity matrix built this way is not `Matrix::identity`. Nothing wrong is
/// computed, but a caller checking a result structurally is told "no" for the
/// wrong reason. Fixed here, at the one place matrix code turns ℚ into
/// expressions, rather than in the kernel where it would touch every subsystem.
fn rational_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

fn dot_columns(a: &Matrix, b: &Matrix, pool: &ExprPool) -> Result<ExprId, LinearAlgebraError> {
    let mut terms = Vec::new();
    for r in 0..a.rows {
        terms.push(simplify(pool.mul(vec![a.get(r, 0), b.get(r, 0)]), pool).value);
    }
    Ok(simplify(pool.add(terms), pool).value)
}

fn rational_sqrt(r: &Rational) -> Option<Rational> {
    let num = r.numer();
    let den = r.denom();
    let sn = integer_sqrt(num)?;
    let sd = integer_sqrt(den)?;
    Some(Rational::from((sn, sd)))
}

fn integer_sqrt(n: &rug::Integer) -> Option<rug::Integer> {
    if n < &0 {
        return None;
    }
    if n.is_zero() {
        return Some(rug::Integer::from(0));
    }
    let root = n.clone().sqrt();
    let sq = root.clone() * root.clone();
    if sq == *n {
        Some(root)
    } else {
        None
    }
}

fn norm_column(v: &Matrix, pool: &ExprPool) -> Result<ExprId, LinearAlgebraError> {
    let mut terms = Vec::new();
    for r in 0..v.rows {
        let e = v.get(r, 0);
        terms.push(simplify(pool.mul(vec![e, e]), pool).value);
    }
    Ok(simplify(
        pool.func("sqrt", vec![simplify(pool.add(terms), pool).value]),
        pool,
    )
    .value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::eigen;

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn nullspace_line_in_plane() {
        let p = pool();
        let one = p.integer(1_i32);
        let two = p.integer(2_i32);
        let m = Matrix::new(vec![vec![one, two]]).unwrap();
        let bas = nullspace_basis(&m, &p).unwrap();
        assert_eq!(bas.len(), 1);
    }

    #[test]
    fn rank_identity() {
        let p = pool();
        let id = Matrix::identity(3, &p);
        assert_eq!(rank(&id, &p).unwrap(), 3);
    }

    /// `exp(a)²` and `exp(2a)` are the same function written two ways.
    ///
    /// Row 2 is `exp(a)` times row 1, so the rank is 1. Before the zero test
    /// grew a third state this returned 2, and the rref carried a `[0 0 1]`
    /// row: the signature of an inconsistent system, for a consistent one.
    #[test]
    fn rank_sees_through_the_exponential_functional_equation() {
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let exp_a = p.func("exp", vec![a]);
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), exp_a, exp_a],
            vec![
                exp_a,
                p.mul(vec![exp_a, exp_a]),
                p.func("exp", vec![p.add(vec![a, a])]),
            ],
        ])
        .unwrap();
        assert_eq!(rank(&m, &p).unwrap(), 1);

        let echelon = rref(&m, &p).unwrap();
        for c in 0..echelon.cols {
            assert_eq!(
                echelon.get(1, c),
                p.integer(0_i32),
                "rank-1 matrix must have an all-zero second rref row"
            );
        }
    }

    /// The control: the same matrix with one entry changed really has rank 2,
    /// so the fix cannot be "call everything zero".
    #[test]
    fn rank_still_separates_independent_exponential_rows() {
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let exp_a = p.func("exp", vec![a]);
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), exp_a, exp_a],
            vec![exp_a, p.mul(vec![exp_a, exp_a]), exp_a],
        ])
        .unwrap();
        assert_eq!(rank(&m, &p).unwrap(), 2);
    }

    /// An entry nothing can decide must produce a coded refusal, not a rank.
    #[test]
    fn rank_refuses_an_undecidable_pivot() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        // `mystery` has no differentiation rule, no numeric kernel and no ball
        // kernel, so it can be neither normalised to zero nor enclosed away
        // from it.
        let opaque = p.func("mystery", vec![x]);
        let zero = p.integer(0_i32);
        let m = Matrix::new(vec![vec![opaque, zero], vec![zero, zero]]).unwrap();
        let err = rank(&m, &p).expect_err("an undecidable pivot must refuse");
        assert!(
            matches!(err, LinearAlgebraError::UnsupportedField),
            "expected a zero-test refusal, got {err:?}"
        );
        use crate::errors::AlkahestError;
        assert!(err.code().starts_with("E-LINALG-"));
        // The variant is a carrier — the specific cause and its `E-LINALG-010`
        // travel out of band, which is what the bindings raise.
        let refusal = crate::matrix::take_zero_test_refusal()
            .expect("the refusal must be recoverable, or the code is lost");
        assert_eq!(refusal.code(), "E-LINALG-010");
        assert!(
            refusal.entry().contains("mystery"),
            "refusal should name the undecided entry, got {}",
            refusal.entry()
        );
        assert_eq!(
            crate::matrix::take_zero_test_refusal(),
            None,
            "taking must consume, so one refusal cannot be reported twice"
        );
    }

    /// A matrix that is *proven* singular must not borrow the refusal code.
    #[test]
    fn a_proven_singularity_is_not_a_zero_test_refusal() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        // Refuse once so there is something stale to pick up...
        let opaque = p.func("mystery", vec![x]);
        let zero = p.integer(0_i32);
        let undecidable = Matrix::new(vec![vec![opaque, zero], vec![zero, zero]]).unwrap();
        let _ = rank(&undecidable, &p);
        // ...then invert a matrix whose determinant is exactly 0.
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), p.integer(2_i32)],
            vec![p.integer(2_i32), p.integer(4_i32)],
        ])
        .unwrap();
        assert_eq!(matrix_inverse(&m, &p), Err(MatrixError::SingularMatrix));
        assert_eq!(
            crate::matrix::take_zero_test_refusal(),
            None,
            "a proven singularity must not be reported as an undecided determinant"
        );
    }

    /// An undecidable determinant must refuse *and* be recoverable as E-MAT-004.
    #[test]
    fn inverse_refuses_an_undecidable_determinant() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let opaque = p.func("mystery", vec![x]);
        let m = Matrix::new(vec![
            vec![opaque, p.integer(0_i32)],
            vec![p.integer(0_i32), p.integer(1_i32)],
        ])
        .unwrap();
        assert_eq!(matrix_inverse(&m, &p), Err(MatrixError::SingularMatrix));
        use crate::errors::AlkahestError;
        let refusal = crate::matrix::take_zero_test_refusal()
            .expect("an undecided determinant must record its refusal");
        assert_eq!(refusal.code(), "E-MAT-004");
    }

    /// The 2×2 matrix whose only non-zero entry nothing can decide.
    ///
    /// Its nullspace is a real question — it is 1- or 2-dimensional depending
    /// on whether `mystery(x)` vanishes identically — so answering it at all
    /// would be a guess.
    fn undecidable_matrix(p: &ExprPool) -> Matrix {
        let x = p.symbol("x", Domain::Real);
        let opaque = p.func("mystery", vec![x]);
        let zero = p.integer(0_i32);
        Matrix::new(vec![vec![opaque, zero], vec![zero, zero]]).unwrap()
    }

    /// `nullspace` used to flatten this into `KernelFailed` / `E-LINALG-002`
    /// ("could not compute nullspace basis"), which cannot be told apart from a
    /// matrix that is merely hard.
    #[test]
    fn nullspace_reports_the_specific_undecidable_entry() {
        use crate::errors::AlkahestError;
        let p = pool();
        let err = nullspace_basis(&undecidable_matrix(&p), &p)
            .expect_err("an undecidable pivot must refuse");
        assert!(
            matches!(err, LinearAlgebraError::UnsupportedField),
            "expected the zero-test carrier variant, got {err:?}"
        );
        let refusal = crate::matrix::take_zero_test_refusal()
            .expect("the refusal must be recoverable, or the specific code is lost");
        assert_eq!(refusal.code(), "E-LINALG-010");
        assert!(
            refusal.entry().contains("mystery"),
            "refusal should name the undecided entry, got {}",
            refusal.entry()
        );
    }

    /// `jordan_form` reaches the same elimination and must report the same
    /// thing: it is the undecided entry that stops it, not the Jordan search.
    #[test]
    fn jordan_form_reports_the_specific_undecidable_entry() {
        use crate::errors::AlkahestError;
        let p = pool();
        let err =
            jordan_form(&undecidable_matrix(&p), &p).expect_err("an undecidable pivot must refuse");
        assert!(
            matches!(err, LinearAlgebraError::UnsupportedField),
            "expected the zero-test carrier variant, got {err:?}"
        );
        let refusal = crate::matrix::take_zero_test_refusal()
            .expect("the refusal must be recoverable, or the specific code is lost");
        assert_eq!(refusal.code(), "E-LINALG-010");
    }

    /// `eigenvects` shares the same kernel routine; the refusal must survive
    /// that boundary too rather than become the vague `E-EIGEN-006`.
    #[test]
    fn eigenvectors_report_the_specific_undecidable_entry() {
        use crate::errors::AlkahestError;
        let p = pool();
        let err = eigen::eigenvectors(&undecidable_matrix(&p), &p)
            .expect_err("an undecidable pivot must refuse");
        assert_eq!(err, eigen::EigenError::KernelComputationFailed);
        let refusal = crate::matrix::take_zero_test_refusal()
            .expect("the refusal must be recoverable, or the specific code is lost");
        assert_eq!(refusal.code(), "E-LINALG-010");
    }

    /// A refusal recorded by `nullspace` must not be picked up by the next
    /// unrelated error — the reason `KernelFailed` was left alone as a carrier
    /// (~30 call sites, no way to tell which one a stale refusal belongs to).
    #[test]
    fn a_nullspace_refusal_is_not_re_attributed_to_a_later_error() {
        let p = pool();
        // Refuse once and leave the refusal on the thread: a Rust caller that
        // never consults it is exactly how a stale one gets there.
        assert!(nullspace_basis(&undecidable_matrix(&p), &p).is_err());
        // Now an error whose cause is *proven*, not undecided: det = 0 exactly.
        let singular = Matrix::new(vec![
            vec![p.integer(1_i32), p.integer(2_i32)],
            vec![p.integer(2_i32), p.integer(4_i32)],
        ])
        .unwrap();
        assert_eq!(
            matrix_inverse(&singular, &p),
            Err(MatrixError::SingularMatrix)
        );
        assert_eq!(
            crate::matrix::take_zero_test_refusal(),
            None,
            "a proven singularity must not inherit the nullspace refusal's code"
        );
    }

    /// `M·v = 0` for every returned basis vector, checked symbolically.
    fn kernel_vectors_are_annihilated(m: &Matrix, basis: &[Matrix], p: &ExprPool) -> bool {
        basis.iter().all(|v| {
            let prod = m.mul(v, p).expect("M·v");
            (0..prod.rows).all(|r| {
                zero_test::zero_status(p, simplify(prod.get(r, 0), p).value)
                    == zero_test::ZeroStatus::Zero
            })
        })
    }

    /// A symbolic determinant that cannot be decided must not be *assumed* zero.
    ///
    /// The 2×2 fast path returns the perpendicular of a non-vanishing row, which
    /// is the kernel only when `det = 0`. Its full-rank gate only fired for a
    /// literal non-zero constant, so any non-literal determinant fell through
    /// into the rank-1 answer — "could not prove `det ≠ 0`" read as
    /// "`det = 0`", the mirror of the `rref` defect that motivated `zero_test`.
    #[test]
    fn nullspace_refuses_an_undecidable_determinant() {
        use crate::errors::AlkahestError;
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let opaque = p.func("mystery", vec![x]);
        // det = mystery(x): neither provably zero nor provably non-zero.
        let m = Matrix::new(vec![
            vec![opaque, p.integer(1_i32)],
            vec![p.integer(0_i32), p.integer(1_i32)],
        ])
        .unwrap();
        let err = nullspace_basis(&m, &p)
            .expect_err("an undecidable determinant must refuse, not return the det=0 answer");
        assert!(matches!(err, LinearAlgebraError::UnsupportedField));
        let refusal = crate::matrix::take_zero_test_refusal().expect("recoverable refusal");
        assert_eq!(refusal.code(), "E-LINALG-010");
        // And it agrees with `rank`, which already refused this matrix.
        assert!(rank(&m, &p).is_err());
        let _ = crate::matrix::take_zero_test_refusal();
    }

    /// A *decidable* non-zero determinant means a trivial kernel — and `rank`
    /// and `nullspace` must not contradict each other.
    ///
    /// `[[x, 0], [0, 1]]` needs no exotic function: `rank` said 2 while
    /// `nullspace` returned the 1-dimensional `(0, x)`, for which
    /// `M·v = (0, x) ≠ 0`. Two public calls, 2 + 1 = 3 for a 2-column matrix.
    #[test]
    fn a_generically_invertible_symbolic_matrix_has_a_trivial_kernel() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        for m in [
            Matrix::new(vec![
                vec![x, p.integer(0_i32)],
                vec![p.integer(0_i32), p.integer(1_i32)],
            ])
            .unwrap(),
            Matrix::new(vec![
                vec![x, p.integer(1_i32)],
                vec![p.integer(0_i32), p.integer(1_i32)],
            ])
            .unwrap(),
            Matrix::new(vec![vec![x, p.integer(0_i32)], vec![p.integer(0_i32), x]]).unwrap(),
        ] {
            let basis = nullspace_basis(&m, &p).expect("a generic determinant is decidable");
            assert!(
                basis.is_empty(),
                "det is generically non-zero, so the kernel is trivial; got {} vector(s)",
                basis.len()
            );
            // rank + nullity = number of columns, across the two public calls.
            assert_eq!(rank(&m, &p).unwrap() + basis.len(), m.cols);
        }
    }

    /// The control that keeps the fix from being "refuse everything": a matrix
    /// that really is singular must still hand back a kernel, and the vectors
    /// must actually be annihilated.
    #[test]
    fn a_genuinely_singular_symbolic_matrix_still_returns_its_kernel() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        for m in [
            Matrix::new(vec![vec![x, x], vec![x, x]]).unwrap(),
            Matrix::new(vec![
                vec![p.integer(1_i32), p.integer(1_i32)],
                vec![p.integer(1_i32), p.integer(1_i32)],
            ])
            .unwrap(),
            // Rank 1 with a transcendental relation the zero test can prove:
            // row 2 = exp(a)·row 1.
            {
                let a = p.symbol("a", Domain::Real);
                let ea = p.func("exp", vec![a]);
                Matrix::new(vec![
                    vec![p.integer(1_i32), ea],
                    vec![ea, p.mul(vec![ea, ea])],
                ])
                .unwrap()
            },
        ] {
            let basis = nullspace_basis(&m, &p).expect("a provably singular matrix has a kernel");
            assert_eq!(basis.len(), 1, "rank-1 2×2 has a 1-dimensional kernel");
            assert!(
                kernel_vectors_are_annihilated(&m, &basis, &p),
                "returned basis vector is not in the kernel"
            );
            assert_eq!(rank(&m, &p).unwrap() + basis.len(), m.cols);
        }
    }

    /// The control: a nullspace the routine *can* compute must leave nothing
    /// behind for a later error to inherit.
    #[test]
    fn a_computable_nullspace_records_no_refusal() {
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let exp_a = p.func("exp", vec![a]);
        // Rank 1: row 2 is exp(a) times row 1, and the zero test can prove it.
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), exp_a],
            vec![exp_a, p.mul(vec![exp_a, exp_a])],
        ])
        .unwrap();
        assert_eq!(nullspace_basis(&m, &p).unwrap().len(), 1);
        assert_eq!(crate::matrix::take_zero_test_refusal(), None);
    }

    #[test]
    fn rref_2x3_rational() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(1), p.integer(2), p.integer(3)],
            vec![p.integer(2), p.integer(4), p.integer(6)],
        ])
        .unwrap();
        let r = rref(&m, &p).unwrap();
        assert_eq!(r.rows, 2);
        assert_eq!(r.cols, 3);
        let one = p.integer(1_i32);
        let two = p.integer(2_i32);
        let three = p.integer(3_i32);
        let z = p.integer(0_i32);
        assert!(eigen::matrix_eq_simplified(
            &r,
            &Matrix::new(vec![vec![one, two, three], vec![z, z, z]]).unwrap(),
            &p
        ));
    }

    #[test]
    fn lu_2x2_rational() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(2), p.integer(1)],
            vec![p.integer(4), p.integer(3)],
        ])
        .unwrap();
        let lu = lu_decomposition(&m, &p).unwrap();
        let reconstructed = lu.l.mul(&lu.u, &p).unwrap();
        let permuted = apply_row_permutation(&m, &lu.perm);
        assert!(eigen::matrix_eq_simplified(&reconstructed, &permuted, &p));
    }

    #[test]
    fn jordan_block_2x2() {
        let p = pool();
        let two = p.integer(2_i32);
        let one = p.integer(1_i32);
        let z = p.integer(0_i32);
        let m = Matrix::new(vec![vec![two, one], vec![z, two]]).unwrap();
        let (p_mat, j) = jordan_form(&m, &p).unwrap();
        let inv = matrix_inverse(&p_mat, &p).unwrap();
        let check = p_mat
            .mul(&j, &p)
            .unwrap()
            .mul(&inv, &p)
            .unwrap()
            .simplify_entries(&p);
        assert!(eigen::matrix_eq_simplified(&check, &m, &p));
    }

    #[test]
    fn rational_canonical_identity_2() {
        let p = pool();
        let id = Matrix::identity(2, &p);
        let (p_mat, c) = rational_canonical_form(&id, &p).unwrap();
        assert_eq!(p_mat.rows, 2);
        assert_eq!(c.rows, 2);
    }

    #[test]
    fn rational_canonical_diagonal_1_2() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), p.integer(0_i32)],
            vec![p.integer(0_i32), p.integer(2_i32)],
        ])
        .unwrap();
        let poly_m = lambda_identity_minus_m_poly(&m, &p).unwrap();
        let (s, _, _) = smith_form_poly(&poly_m);
        let factors = invariant_factors_from_smith(&s).unwrap();
        frobenius_p_from_cyclic_vectors(&m, &factors, &p).expect("cyclic P");
        let (p_mat, c) = rational_canonical_form(&m, &p).unwrap();
        assert_eq!(p_mat.rows, 2);
        assert_eq!(c.rows, 2);
    }

    #[test]
    fn matrix_exp_diagonal_shape() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(1), p.integer(0)],
            vec![p.integer(0), p.integer(2)],
        ])
        .unwrap();
        let expm = matrix_exponential(&m, &p).unwrap();
        assert_eq!(expm.rows, 2);
        assert_eq!(expm.cols, 2);
        assert!(!zero_test::zero_status(&p, expm.get(0, 0)).is_proven_zero());
        assert!(!zero_test::zero_status(&p, expm.get(1, 1)).is_proven_zero());
    }

    #[test]
    fn matrix_exp_symbolic_diagonal() {
        // exp(diag(a, b)) = diag(e^a, e^b) for free symbols a ≠ b.
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let z = p.integer(0_i32);
        let m = Matrix::new(vec![vec![a, z], vec![z, b]]).unwrap();
        let expm = matrix_exponential(&m, &p).unwrap().simplify_entries(&p);
        let ea = simplify(p.func("exp", vec![a]), &p).value;
        let eb = simplify(p.func("exp", vec![b]), &p).value;
        let expected = Matrix::new(vec![vec![ea, z], vec![z, eb]]).unwrap();
        assert!(
            eigen::matrix_eq_simplified(&expm, &expected, &p),
            "got {}",
            expm.display(&p)
        );
    }

    #[test]
    fn matrix_exp_symbolic_oscillator_has_closed_form() {
        // The headline probe: a state matrix with a FREE SYMBOL now yields e^{A} in closed
        // form (previously errored "entries must simplify to rationals"). A = [[0,1],[-w²,0]].
        let p = pool();
        let w = p.symbol("w", Domain::Real);
        let z = p.integer(0_i32);
        let one = p.integer(1_i32);
        let w2 = p.pow(w, p.integer(2_i32));
        let neg_w2 = p.mul(vec![p.integer(-1_i32), w2]);
        let a = Matrix::new(vec![vec![z, one], vec![neg_w2, z]]).unwrap();
        let expm = matrix_exponential(&a, &p).expect("symbolic e^A closed form");
        assert_eq!(expm.rows, 2);
        assert_eq!(expm.cols, 2);
        // Every entry depends on w (via exp(±√(−4w²)/2)) — i.e. genuinely symbolic.
        let s = expm.display(&p);
        assert!(s.contains("exp"), "expected exponential entries: {s}");
        assert!(s.contains('w'), "expected dependence on free symbol w: {s}");
    }

    // -----------------------------------------------------------------------
    // The defective matrix exponential
    //
    // Every expected value below is `sympy.Matrix(M).exp()`, not alkahest's own
    // output. Before 3.10.1 the first four returned the identity, or an
    // off-diagonal twice too large, or refused — with no exception and no flag.
    // -----------------------------------------------------------------------

    fn int_matrix(rows: &[&[i32]], pool: &ExprPool) -> Matrix {
        Matrix::new(
            rows.iter()
                .map(|r| r.iter().map(|&v| pool.integer(v)).collect())
                .collect(),
        )
        .expect("square integer matrix")
    }

    /// Every entry of `matrix_exponential(rows)` against `expected`, as f64.
    fn assert_exp_entries(rows: &[&[i32]], expected: &[&[f64]]) {
        let p = pool();
        let a = int_matrix(rows, &p);
        let e = matrix_exponential(&a, &p).expect("e^A for an integer matrix");
        let env = std::collections::HashMap::new();
        for (i, row) in expected.iter().enumerate() {
            for (j, &want) in row.iter().enumerate() {
                let got = crate::eval::eval_complex_f64(e.get(i, j), &p, &env)
                    .unwrap_or_else(|err| panic!("entry ({i},{j}) is not a number: {err}"));
                assert!(
                    (got.re - want).abs() < 1e-9 && got.im.abs() < 1e-9,
                    "entry ({i},{j}): got {} + {}i, sympy says {want}\n{}",
                    got.re,
                    got.im,
                    e.display(&p)
                );
            }
        }
    }

    const E1: f64 = std::f64::consts::E;

    #[test]
    fn matrix_exp_nilpotent_2x2_is_not_the_identity() {
        // sympy: exp([[0,1],[0,0]]) == [[1,1],[0,1]]. Returned I before 3.10.1.
        assert_exp_entries(&[&[0, 1], &[0, 0]], &[&[1.0, 1.0], &[0.0, 1.0]]);
    }

    #[test]
    fn matrix_exp_defective_off_diagonal_is_e_lambda_not_lambda_e_lambda() {
        // sympy: exp([[2,1],[0,2]]) == [[e², e²],[0, e²]]. The off-diagonal was
        // 2e² before 3.10.1 — the block formula used λ^k where N^k belongs.
        let e2 = E1 * E1;
        assert_exp_entries(&[&[2, 1], &[0, 2]], &[&[e2, e2], &[0.0, e2]]);
    }

    #[test]
    fn matrix_exp_nilpotent_3x3_keeps_the_half() {
        // sympy: exp([[0,1,0],[0,0,1],[0,0,0]]) == [[1,1,1/2],[0,1,1],[0,0,1]].
        assert_exp_entries(
            &[&[0, 1, 0], &[0, 0, 1], &[0, 0, 0]],
            &[&[1.0, 1.0, 0.5], &[0.0, 1.0, 1.0], &[0.0, 0.0, 1.0]],
        );
    }

    #[test]
    fn matrix_exp_full_3x3_jordan_block() {
        // sympy: exp([[2,1,0],[0,2,1],[0,0,2]]) == [[e²,e²,e²/2],[0,e²,e²],[0,0,e²]].
        // The old block-size detector compared `J[i][i+sz]` against 1 where the
        // superdiagonal is `J[i+sz−1][i+sz]`, so this split into a 2×2 and a
        // 1×1 and lost the e²/2 corner entirely.
        let e2 = E1 * E1;
        assert_exp_entries(
            &[&[2, 1, 0], &[0, 2, 1], &[0, 0, 2]],
            &[&[e2, e2, e2 / 2.0], &[0.0, e2, e2], &[0.0, 0.0, e2]],
        );
    }

    #[test]
    fn matrix_exp_two_jordan_blocks_for_one_eigenvalue() {
        // sympy: exp(J₂(3) ⊕ J₂(3)) is block diagonal with [[e³,e³],[0,e³]].
        // This refused before 3.10.1 (`jordan_form` handed back a singular P);
        // Putzer needs no basis at all.
        let e3 = E1.powi(3);
        assert_exp_entries(
            &[&[3, 1, 0, 0], &[0, 3, 0, 0], &[0, 0, 3, 1], &[0, 0, 0, 3]],
            &[
                &[e3, e3, 0.0, 0.0],
                &[0.0, e3, 0.0, 0.0],
                &[0.0, 0.0, e3, e3],
                &[0.0, 0.0, 0.0, e3],
            ],
        );
    }

    #[test]
    fn matrix_exp_mixed_defective_and_simple_eigenvalue() {
        // sympy: exp([[1,1,0],[0,1,0],[0,0,5]]) == [[e,e,0],[0,e,0],[0,0,e⁵]].
        assert_exp_entries(
            &[&[1, 1, 0], &[0, 1, 0], &[0, 0, 5]],
            &[&[E1, E1, 0.0], &[0.0, E1, 0.0], &[0.0, 0.0, E1.powi(5)]],
        );
    }

    #[test]
    fn matrix_exp_defective_but_not_triangular() {
        // sympy: exp([[1,1],[-1,3]]) == [[0, e²],[-e², 2e²]]. Characteristic
        // polynomial (λ−2)², geometric multiplicity 1, and nothing about the
        // input announces that — there is no zero off-diagonal to read a
        // spectrum off.
        let e2 = E1 * E1;
        assert_exp_entries(&[&[1, 1], &[-1, 3]], &[&[0.0, e2], &[-e2, 2.0 * e2]]);
    }

    #[test]
    fn matrix_exp_defective_3x3_but_not_triangular() {
        // sympy: exp([[4,1,1],[-2,1,-1],[0,1,1]]) ==
        //   [[4e²,2e²,e²],[-3e²,-e²,-e²],[-e²,0,0]]. One eigenvalue λ=2 with a
        //   single 3×3 Jordan block, reached through the general spectrum.
        let e2 = E1 * E1;
        assert_exp_entries(
            &[&[4, 1, 1], &[-2, 1, -1], &[0, 1, 1]],
            &[
                &[4.0 * e2, 2.0 * e2, e2],
                &[-3.0 * e2, -e2, -e2],
                &[-e2, 0.0, 0.0],
            ],
        );
    }

    #[test]
    fn matrix_exp_rotation_is_the_rotation_by_one_radian() {
        // sympy: exp([[0,1],[-1,0]]) == [[cos 1, sin 1],[-sin 1, cos 1]].
        // A complex spectrum {±i}, and the answer must come back real.
        let (c, s) = (1.0_f64.cos(), 1.0_f64.sin());
        assert_exp_entries(&[&[0, 1], &[-1, 0]], &[&[c, s], &[-s, c]]);
    }

    #[test]
    fn matrix_exp_diagonalizable_controls_do_not_regress() {
        // The route that was already right. sympy: exp([[1,0],[0,2]]) =
        // diag(e, e²); exp([[1,2],[3,4]]) = [[51.968956198705, 74.73656456700321],
        // [112.10484685050481, 164.07380304920983]].
        assert_exp_entries(&[&[1, 0], &[0, 2]], &[&[E1, 0.0], &[0.0, E1 * E1]]);
        assert_exp_entries(
            &[&[1, 2], &[3, 4]],
            &[
                &[51.968956198705, 74.73656456700321],
                &[112.10484685050481, 164.07380304920983],
            ],
        );
    }

    #[test]
    fn jordan_form_never_returns_a_singular_p() {
        // J₂(3) ⊕ J₂(3): both chains are drawn from the same kernel, and taking
        // the same vector twice gave a P with two identical columns — so
        // `M = P·J·P⁻¹` was false and `P⁻¹` did not exist, with nothing said.
        // Either an invertible P or a refusal is acceptable; a singular P is not.
        let p = pool();
        let m = int_matrix(
            &[&[3, 1, 0, 0], &[0, 3, 0, 0], &[0, 0, 3, 1], &[0, 0, 0, 3]],
            &p,
        );
        match jordan_form(&m, &p) {
            Ok((pm, j)) => {
                assert_eq!(
                    rank(&pm, &p).expect("rational rank"),
                    4,
                    "P must be a basis"
                );
                let inv = matrix_inverse(&pm, &p).expect("an invertible P");
                let recon = pm
                    .mul(&j, &p)
                    .unwrap()
                    .mul(&inv, &p)
                    .unwrap()
                    .simplify_entries(&p);
                assert!(
                    eigen::matrix_eq_simplified(&recon, &m, &p),
                    "P·J·P⁻¹ must be M, got {}",
                    recon.display(&p)
                );
            }
            Err(e) => assert_eq!(e, LinearAlgebraError::SingularTransform, "{e}"),
        }
    }

    #[test]
    fn jordan_form_handles_two_blocks_of_different_sizes() {
        // [[0,0,1],[0,0,0],[0,0,0]] is J₂(0) ⊕ J₁(0) — one eigenvalue, two
        // blocks, so the same repeated-kernel trap with unequal sizes.
        let p = pool();
        let m = int_matrix(&[&[0, 0, 1], &[0, 0, 0], &[0, 0, 0]], &p);
        let (pm, j) = jordan_form(&m, &p).expect("J₂(0) ⊕ J₁(0)");
        assert_eq!(rank(&pm, &p).expect("rational rank"), 3);
        let inv = matrix_inverse(&pm, &p).expect("an invertible P");
        let recon = pm
            .mul(&j, &p)
            .unwrap()
            .mul(&inv, &p)
            .unwrap()
            .simplify_entries(&p);
        assert!(
            eigen::matrix_eq_simplified(&recon, &m, &p),
            "P·J·P⁻¹ must be M, got {}",
            recon.display(&p)
        );
        // sympy: exp([[0,0,1],[0,0,0],[0,0,0]]) == [[1,0,1],[0,1,0],[0,0,1]].
        assert_exp_entries(
            &[&[0, 0, 1], &[0, 0, 0], &[0, 0, 0]],
            &[&[1.0, 0.0, 1.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
        );
    }

    #[test]
    fn matrix_exp_reports_the_eigenvalue_gap_it_divided_by() {
        // [[a, 1], [0, b]] for free symbols a, b is defective exactly at a = b,
        // where the generic e^A divides by zero and the truth is the confluent
        // e^a(I + N). The generic answer is right everywhere else and is what a
        // caller wants — but only with `a − b ≠ 0` stated, which is the part
        // that was previously invisible.
        let p = pool();
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let z = p.integer(0_i32);
        let one = p.integer(1_i32);
        let m = Matrix::new(vec![vec![a, one], vec![z, b]]).unwrap();
        matrix_exponential(&m, &p).expect("the generic branch");
        let conds = crate::matrix::take_matrix_exp_side_conditions();
        assert_eq!(conds.len(), 1, "a − b must be reported: {conds:?}");
        assert!(matches!(conds[0], crate::deriv::SideCondition::NonZero(_)));
        // Consuming: a second read sees nothing.
        assert!(crate::matrix::take_matrix_exp_side_conditions().is_empty());
    }

    #[test]
    fn matrix_exp_reports_nothing_when_every_gap_is_settled() {
        // The control for the case above: a rational spectrum needs no
        // hypothesis at all, and a channel that always says something is a
        // channel a caller learns to ignore.
        let p = pool();
        let m = int_matrix(&[&[1, 2], &[3, 4]], &p);
        matrix_exponential(&m, &p).expect("distinct rational spectrum");
        assert!(crate::matrix::take_matrix_exp_side_conditions().is_empty());
        let d = int_matrix(&[&[2, 1], &[0, 2]], &p);
        matrix_exponential(&d, &p).expect("repeated rational spectrum");
        assert!(
            crate::matrix::take_matrix_exp_side_conditions().is_empty(),
            "the confluent branch divides by nothing"
        );
    }

    #[test]
    fn matrix_exp_symbolic_state_matrix_t_zero_is_identity() {
        // For a symbolic state matrix A(parameter), exp(0·A) must be the identity.
        // Build A = [[0, 1], [-k, 0]] (oscillator with symbolic stiffness k) scaled by 0.
        let p = pool();
        let k = p.symbol("k", Domain::Real);
        let z = p.integer(0_i32);
        let one = p.integer(1_i32);
        let neg_k = p.mul(vec![p.integer(-1_i32), k]);
        let a = Matrix::new(vec![vec![z, one], vec![neg_k, z]]).unwrap();
        let zero_a = a.scale(z, &p);
        let expm = matrix_exponential(&zero_a, &p)
            .unwrap()
            .simplify_entries(&p);
        assert!(
            eigen::matrix_eq_simplified(&expm, &Matrix::identity(2, &p), &p),
            "exp(0) should be I, got {}",
            expm.display(&p)
        );
    }

    #[test]
    fn symbolic_inverse_diag_s_s() {
        // diag(s, s) has determinant s^2; its inverse is diag(1/s, 1/s).
        let p = pool();
        let s = p.symbol("s", Domain::Real);
        let z = p.integer(0_i32);
        let m = Matrix::new(vec![vec![s, z], vec![z, s]]).unwrap();
        let inv = matrix_inverse(&m, &p).unwrap();
        let inv_s = simplify(p.pow(s, p.integer(-1_i32)), &p).value;
        let expected = Matrix::new(vec![vec![inv_s, z], vec![z, inv_s]]).unwrap();
        assert!(eigen::matrix_eq_simplified(&inv, &expected, &p));
        // And A * A^-1 = I.
        let prod = m.mul(&inv, &p).unwrap().simplify_entries(&p);
        assert!(eigen::matrix_eq_simplified(
            &prod,
            &Matrix::identity(2, &p),
            &p
        ));
    }

    #[test]
    fn symbolic_inverse_2x2_product_is_identity() {
        // [[s, 1], [2, s+3]] inverse, verify A · A⁻¹ = I.
        //
        // The kernel simplifier has no multivariate `together`/`cancel` pass, so a
        // symbolic A·A⁻¹ cannot be coaxed structurally to the literal identity (the
        // shared 1/det factor spread over a *sum* of cofactor terms never collapses
        // — only a bare `Mul([X, X⁻¹])` cancels). We therefore (1) confirm the
        // computed inverse equals adj(A)/det entry-by-entry, and (2) verify the
        // equivalent denominator-cleared identity A · adj(A) = det(A)·I, which is a
        // pure polynomial relation that `simplify_expanded` fully normalizes.
        let p = pool();
        let s = p.symbol("s", Domain::Real);
        let one = p.integer(1_i32);
        let two = p.integer(2_i32);
        let s_plus_3 = simplify(p.add(vec![s, p.integer(3_i32)]), &p).value;
        let m = Matrix::new(vec![vec![s, one], vec![two, s_plus_3]]).unwrap();
        let inv = matrix_inverse(&m, &p).unwrap();
        let det = simplify_expanded(m.det(&p).unwrap(), &p).value;
        let det_inv = simplify(p.pow(det, p.integer(-1_i32)), &p).value;

        // adj(A)[i][j] = (-1)^(i+j) · det(minor_ji)   (transposed cofactor)
        let adj_entry = |i: usize, j: usize, p: &ExprPool| -> ExprId {
            let minor_det = simplify_expanded(m.minor(j, i).det(p).unwrap(), p).value;
            let sign = if (i + j) % 2 == 0 { 1_i32 } else { -1_i32 };
            simplify(p.mul(vec![p.integer(sign), minor_det]), p).value
        };

        // (1) inverse == adj(A) · (1/det), entry-by-entry, after a single cancelling Mul.
        for i in 0..2 {
            for j in 0..2 {
                let expected = simplify(p.mul(vec![adj_entry(i, j, &p), det_inv]), &p).value;
                assert!(
                    eigen::matrix_eq_simplified(
                        &Matrix::new(vec![vec![inv.get(i, j)]]).unwrap(),
                        &Matrix::new(vec![vec![expected]]).unwrap(),
                        &p,
                    ),
                    "inverse entry [{i}][{j}] mismatch"
                );
            }
        }

        // (2) A · adj(A) = det(A) · I  (pure polynomial — no 1/det anywhere).
        let adj = Matrix::new(
            (0..2)
                .map(|i| (0..2).map(|j| adj_entry(i, j, &p)).collect())
                .collect(),
        )
        .unwrap();
        let prod = m.mul(&adj, &p).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { det } else { p.integer(0_i32) };
                let diff = simplify_expanded(
                    p.add(vec![
                        prod.get(i, j),
                        p.mul(vec![p.integer(-1_i32), expected]),
                    ]),
                    &p,
                )
                .value;
                assert!(
                    zero_test::zero_status(&p, diff).is_proven_zero(),
                    "(A·adj)[{i}][{j}] != det·I[{i}][{j}]: {:?}",
                    p.get(diff)
                );
            }
        }
    }

    #[test]
    fn symbolic_inverse_singular_returns_error() {
        // [[s, s], [1, 1]] has determinant s*1 - s*1 = 0 -> genuinely singular.
        let p = pool();
        let s = p.symbol("s", Domain::Real);
        let one = p.integer(1_i32);
        let m = Matrix::new(vec![vec![s, s], vec![one, one]]).unwrap();
        assert_eq!(matrix_inverse(&m, &p), Err(MatrixError::SingularMatrix));
    }

    #[test]
    fn numeric_inverse_still_works() {
        // Rational fast path must remain correct.
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(4), p.integer(7)],
            vec![p.integer(2), p.integer(6)],
        ])
        .unwrap();
        let inv = matrix_inverse(&m, &p).unwrap();
        let prod = m.mul(&inv, &p).unwrap().simplify_entries(&p);
        assert!(eigen::matrix_eq_simplified(
            &prod,
            &Matrix::identity(2, &p),
            &p
        ));
    }

    // =======================================================================
    // 3.10.1 linear-algebra audit — regressions for five silent wrong answers
    //
    // Every expectation below is an *identity* the operation is defined by
    // (`P·A = L·U`, `M·P = P·C`, `p(M) = 0`, `L·Lᵀ = M`, independence of a
    // claimed basis), checked against the input rather than against a value
    // read off alkahest. That is the property the pre-existing tests for these
    // routines did not check: `rational_canonical_form` had two tests and both
    // asserted only `p_mat.rows == 2`, which a matrix of the wrong rank passes.
    // =======================================================================

    /// Random-ish integer matrix, deterministic from a seed.
    fn seeded_matrix(seed: u64, rows: usize, cols: usize, pool: &ExprPool) -> Matrix {
        let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) % 9) as i32 - 4
        };
        let grid: Vec<Vec<ExprId>> = (0..rows)
            .map(|_| (0..cols).map(|_| pool.integer(next())).collect())
            .collect();
        Matrix::new(grid).expect("seeded matrix")
    }

    /// Entrywise equality by *value*: every difference is proven zero.
    ///
    /// Stronger than `matrix_eq_simplified`, which compares normalised forms
    /// and therefore reports `√2·√2` and `2` as different. Here the difference
    /// goes through the full `zero_test` ladder, so only a proven identity
    /// passes and an undecided entry fails rather than sliding through.
    fn entries_equal(a: &Matrix, b: &Matrix, pool: &ExprPool) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        for i in 0..a.rows {
            for j in 0..a.cols {
                let diff = simplify_expanded(
                    pool.add(vec![
                        a.get(i, j),
                        pool.mul(vec![pool.integer(-1_i32), b.get(i, j)]),
                    ]),
                    pool,
                )
                .value;
                if !zero_test::zero_status(pool, diff).is_proven_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// Exact rational value of a constant expression, or `None`.
    ///
    /// A four-line evaluator over ℚ that the simplifier is not involved in, so
    /// a test using it is not checking alkahest against itself. Handles the
    /// integer powers that an adjugate-over-determinant entry is built from.
    fn exact_rational(e: ExprId, pool: &ExprPool) -> Option<Rational> {
        match pool.get(e) {
            ExprData::Integer(n) => Some(Rational::from((n.0.clone(), rug::Integer::from(1)))),
            ExprData::Rational(r) => Some(r.0.clone()),
            ExprData::Add(args) => {
                let mut acc = Rational::from(0);
                for a in args {
                    acc += exact_rational(a, pool)?;
                }
                Some(acc)
            }
            ExprData::Mul(args) => {
                let mut acc = Rational::from(1);
                for a in args {
                    acc *= exact_rational(a, pool)?;
                }
                Some(acc)
            }
            ExprData::Pow { base, exp } => {
                let b = exact_rational(base, pool)?;
                let k = match pool.get(exp) {
                    ExprData::Integer(n) => n.0.to_i32()?,
                    _ => return None,
                };
                if k < 0 && b == 0 {
                    return None;
                }
                let mut acc = Rational::from(1);
                for _ in 0..k.unsigned_abs() {
                    acc *= b.clone();
                }
                Some(if k >= 0 { acc } else { Rational::from(1) / acc })
            }
            _ => None,
        }
    }

    fn permute_rows(m: &Matrix, perm: &[usize]) -> Matrix {
        Matrix::new(perm.iter().map(|&r| m.row(r)).collect()).expect("permuted")
    }

    // ----------------------------------------------------------------- LU

    /// `P·A = L·U` needs the multipliers already in `L` to move when a pivot
    /// swap moves their rows.
    ///
    /// The 2×2 test that stood here before could not see this: its only swap is
    /// at `k = 0`, where `L` has no computed columns yet to be left behind. The
    /// defect needs a swap at `k ≥ 1`, so the smallest witness is 3×3.
    ///
    /// `A = [[2,−1,4],[−1,1,−1],[3,−4,0]]` pivots to row 2 at `k = 0` and to
    /// row 2 again at `k = 1`. Before the fix `L·U` came back as
    /// `[[3,−4,0],[−1,3,4],[2,−3,−1]]` — the rows of `A` in the order
    /// `2, 1, 0`, while `perm` said `2, 0, 1`, so the product was not `P·A` for
    /// the `P` that was handed back.
    #[test]
    fn lu_partial_pivot_permutes_l() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(2_i32), p.integer(-1_i32), p.integer(4_i32)],
            vec![p.integer(-1_i32), p.integer(1_i32), p.integer(-1_i32)],
            vec![p.integer(3_i32), p.integer(-4_i32), p.integer(0_i32)],
        ])
        .unwrap();
        let lu = lu_decomposition(&m, &p).unwrap();
        let reconstructed = lu.l.mul(&lu.u, &p).unwrap();
        let permuted = permute_rows(&m, &lu.perm);
        assert!(
            eigen::matrix_eq_simplified(&reconstructed, &permuted, &p),
            "P·A = L·U must hold; perm = {:?}",
            lu.perm
        );
    }

    /// The same identity over 120 random shapes, square and rectangular.
    ///
    /// A randomised sweep against `P·A = L·U` found 77 of 259 corpus matrices
    /// wrong before the fix; anything past 2×2 with two pivot swaps was liable.
    #[test]
    fn lu_reconstructs_every_random_shape() {
        let p = pool();
        for seed in 0..120u64 {
            let rows = 1 + (seed % 5) as usize;
            let cols = 1 + ((seed / 5) % 5) as usize;
            let m = seeded_matrix(seed, rows, cols, &p);
            let lu = lu_decomposition(&m, &p).unwrap();
            assert_eq!(lu.l.rows, rows);
            assert_eq!(lu.l.cols, rows);
            assert_eq!(lu.u.rows, rows);
            assert_eq!(lu.u.cols, cols);
            let mut sorted = lu.perm.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..rows).collect::<Vec<_>>(),
                "perm must be a permutation, got {:?}",
                lu.perm
            );
            // `L` unit lower triangular.
            for i in 0..rows {
                assert_eq!(lu.l.get(i, i), p.integer(1_i32), "seed {seed}: L diagonal");
                for j in (i + 1)..rows {
                    assert_eq!(
                        simplify(lu.l.get(i, j), &p).value,
                        p.integer(0_i32),
                        "seed {seed}: L must be lower triangular"
                    );
                }
            }
            // `U` upper triangular.
            for i in 0..rows {
                for j in 0..i.min(cols) {
                    assert_eq!(
                        simplify(lu.u.get(i, j), &p).value,
                        p.integer(0_i32),
                        "seed {seed}: U must be upper triangular"
                    );
                }
            }
            let reconstructed = lu.l.mul(&lu.u, &p).unwrap();
            let permuted = permute_rows(&m, &lu.perm);
            assert!(
                eigen::matrix_eq_simplified(&reconstructed, &permuted, &p),
                "seed {seed}: P·A = L·U failed, perm = {:?}",
                lu.perm
            );
        }
    }

    // ---------------------------------------------------------- row space

    /// The pivot rows come from the echelon form, not from `m`.
    ///
    /// `[[0,0],[1,0]]` has row space `span{[1,0]}`. Elimination swaps the two
    /// rows, marking echelon row 0 as the pivot row; reading `m.row(0)` gave
    /// `[0,0]` — the zero vector returned as a basis of a one-dimensional
    /// space.
    #[test]
    fn row_space_of_a_swapped_matrix_is_not_the_zero_vector() {
        let p = pool();
        let z = p.integer(0_i32);
        let one = p.integer(1_i32);
        let m = Matrix::new(vec![vec![z, z], vec![one, z]]).unwrap();
        let basis = row_space_basis(&m, &p).unwrap();
        assert_eq!(basis.len(), 1);
        let row = &basis[0];
        assert_eq!(row.rows, 1);
        assert_eq!(row.cols, 2);
        assert_eq!(simplify(row.get(0, 0), &p).value, one);
        assert_eq!(simplify(row.get(0, 1), &p).value, z);
    }

    /// A duplicated row must not be handed back twice.
    ///
    /// `[[1,2,3],[1,2,3],[0,1,0]]` has rank 2; the old code returned echelon
    /// rows 0 and 1 *of the input*, i.e. the same vector twice, whose span is
    /// one-dimensional.
    #[test]
    fn row_space_basis_of_a_duplicated_row_is_independent() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), p.integer(2_i32), p.integer(3_i32)],
            vec![p.integer(1_i32), p.integer(2_i32), p.integer(3_i32)],
            vec![p.integer(0_i32), p.integer(1_i32), p.integer(0_i32)],
        ])
        .unwrap();
        let basis = row_space_basis(&m, &p).unwrap();
        assert_eq!(basis.len(), 2, "rank is 2");
        let stacked = Matrix::new(basis.iter().map(|b| b.row(0)).collect()).unwrap();
        assert_eq!(
            rank(&stacked, &p).unwrap(),
            2,
            "a basis must be independent"
        );
    }

    /// Over 80 random shapes: the claimed basis is independent, has `rank(m)`
    /// elements, and does not leave the row space of `m`.
    #[test]
    fn row_space_basis_is_a_basis_for_every_random_shape() {
        let p = pool();
        for seed in 200..280u64 {
            let rows = 1 + (seed % 4) as usize;
            let cols = 1 + ((seed / 4) % 4) as usize;
            let mut grid: Vec<Vec<ExprId>> = (0..rows)
                .map(|r| seeded_matrix(seed + r as u64, 1, cols, &p).row(0))
                .collect();
            // Force a dependency half the time so rank-deficient shapes are hit.
            if rows > 1 && seed % 2 == 0 {
                grid[0] = grid[rows - 1].clone();
            }
            let m = Matrix::new(grid).unwrap();
            let r = rank(&m, &p).unwrap();
            let basis = row_space_basis(&m, &p).unwrap();
            assert_eq!(basis.len(), r, "seed {seed}: basis size must equal rank");
            if r == 0 {
                continue;
            }
            let stacked = Matrix::new(basis.iter().map(|b| b.row(0)).collect()).unwrap();
            assert_eq!(
                rank(&stacked, &p).unwrap(),
                r,
                "seed {seed}: basis must be independent"
            );
            // Stacking the basis under `m` adds nothing: the basis lies in row(m).
            let mut all: Vec<Vec<ExprId>> = (0..rows).map(|i| m.row(i)).collect();
            all.extend(basis.iter().map(|b| b.row(0)));
            let combined = Matrix::new(all).unwrap();
            assert_eq!(
                rank(&combined, &p).unwrap(),
                r,
                "seed {seed}: basis must stay inside row(m)"
            );
        }
    }

    // ------------------------------------------------- minimal polynomial

    /// `p(M)` is `Σ cᵢ Mⁱ`, and a vanishing constant term must not shift the
    /// powers.
    ///
    /// The zero matrix is annihilated by `λ`. The old power bookkeeping skipped
    /// advancing `M⁰ → M¹` when `c₀ = 0`, so it evaluated `λ` as `I`, rejected
    /// it, and returned `λ²` — a correct annihilator, but not the minimal one,
    /// which is the entire content of the function's name. A zero constant term
    /// is exactly `0 ∈ spec(M)`, so every singular matrix was exposed.
    #[test]
    fn minimal_polynomial_of_the_zero_matrix_is_lambda() {
        let p = pool();
        for n in 1..=4 {
            let m = Matrix::zeros(n, n, &p);
            let (poly, lam) = minimal_polynomial(&m, &p).unwrap();
            assert_eq!(
                simplify(poly, &p).value,
                simplify(lam, &p).value,
                "minimal polynomial of the {n}×{n} zero matrix is λ, not λ^k"
            );
        }
    }

    /// `J₂(0) ⊕ J₂(0)` is annihilated by `λ²`, not `λ³`.
    ///
    /// Its characteristic polynomial is `λ⁴` and the largest block is 2×2, so
    /// the minimal polynomial is `λ²`. The old code returned `λ³`.
    #[test]
    fn minimal_polynomial_of_two_equal_nilpotent_blocks_is_lambda_squared() {
        let p = pool();
        let z = p.integer(0_i32);
        let one = p.integer(1_i32);
        let m = Matrix::new(vec![
            vec![z, one, z, z],
            vec![z, z, z, z],
            vec![z, z, z, one],
            vec![z, z, z, z],
        ])
        .unwrap();
        let (poly, lam) = minimal_polynomial(&m, &p).unwrap();
        let expected = simplify(p.pow(lam, p.integer(2_i32)), &p).value;
        assert_eq!(simplify(poly, &p).value, expected);
    }

    /// Whatever is returned must annihilate `M`, and nothing of lower degree
    /// that divides the characteristic polynomial may.
    ///
    /// Checked here for the nilpotent family the bug lived in, where the
    /// minimal polynomial `λ^s` is known independently from the block size.
    #[test]
    fn minimal_polynomial_of_a_single_nilpotent_block_is_lambda_to_its_size() {
        let p = pool();
        for size in 1..=4usize {
            let mut m = Matrix::zeros(size, size, &p);
            for i in 0..size.saturating_sub(1) {
                m.set(i, i + 1, p.integer(1_i32));
            }
            let (poly, lam) = minimal_polynomial(&m, &p).unwrap();
            let expected = if size == 1 {
                simplify(lam, &p).value
            } else {
                simplify(p.pow(lam, p.integer(size as i32)), &p).value
            };
            assert_eq!(
                simplify(poly, &p).value,
                expected,
                "J_{size}(0) is annihilated by λ^{size} and nothing smaller"
            );
        }
    }

    // -------------------------------------------- rational canonical form

    /// The companion block carries its coefficients down the last **column**.
    ///
    /// `diag(1,2)` has the single invariant factor `λ²−3λ+2`, whose companion
    /// matrix is `[[0,−2],[1,3]]`. Writing the coefficients along the last row
    /// produced `[[0,0],[−2,3]]`, with determinant `0` where `det M = 2` — not
    /// similar to `M`, and not catchable by the `p_mat.rows == 2` assertions
    /// that were the only coverage this function had.
    #[test]
    fn rational_canonical_form_reconstructs_the_matrix() {
        let p = pool();
        let cases: Vec<Vec<Vec<i32>>> = vec![
            vec![vec![1, 0], vec![0, 2]],
            vec![vec![1, 1], vec![0, 1]],
            vec![vec![4, -1], vec![-2, -2]],
            vec![vec![0, 1], vec![0, 0]],
            vec![vec![2, 0, 0], vec![0, 2, 0], vec![0, 0, 3]],
            vec![vec![1, 2, 3], vec![0, 1, 4], vec![0, 0, 1]],
            vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
            vec![vec![2, 1, 0], vec![0, 2, 1], vec![0, 0, 2]],
        ];
        for case in cases {
            let n = case.len();
            let m = Matrix::new(
                case.iter()
                    .map(|r| r.iter().map(|&v| p.integer(v)).collect())
                    .collect(),
            )
            .unwrap();
            let (pm, c) = match rational_canonical_form(&m, &p) {
                Ok(v) => v,
                // A refusal is always allowed; a wrong pair is not.
                Err(_) => continue,
            };
            let lhs = m.mul(&pm, &p).unwrap();
            let rhs = pm.mul(&c, &p).unwrap();
            assert!(
                eigen::matrix_eq_simplified(&lhs, &rhs, &p),
                "M·P = P·C failed for {case:?}"
            );
            assert_eq!(
                rank(&pm, &p).unwrap(),
                n,
                "P must be invertible for {case:?}"
            );
        }
    }

    /// `det(C) = det(M)` and `tr(C) = tr(M)`, read off the returned `C` alone.
    ///
    /// An independent consequence of similarity that the transposed companion
    /// matrix violated outright: on `diag(1,2)` it gave `det C = 0` against
    /// `det M = 2`.
    #[test]
    fn rational_canonical_form_preserves_determinant_and_trace() {
        let p = pool();
        for seed in 400..460u64 {
            let n = 2 + (seed % 3) as usize;
            let m = seeded_matrix(seed, n, n, &p);
            let Ok((_, c)) = rational_canonical_form(&m, &p) else {
                continue;
            };
            let det_m = simplify_expanded(m.det(&p).unwrap(), &p).value;
            let det_c = simplify_expanded(c.det(&p).unwrap(), &p).value;
            assert_eq!(det_m, det_c, "seed {seed}: det must be preserved");
            let tr = |x: &Matrix| {
                let terms: Vec<ExprId> = (0..n).map(|i| x.get(i, i)).collect();
                simplify_expanded(p.add(terms), &p).value
            };
            assert_eq!(tr(&m), tr(&c), "seed {seed}: trace must be preserved");
        }
    }

    /// 60 random matrices: either a refusal, or a pair that satisfies
    /// `M·P = P·C` with `P` invertible. Never anything else.
    #[test]
    fn rational_canonical_form_is_right_or_refuses() {
        let p = pool();
        let mut answered = 0usize;
        for seed in 500..560u64 {
            let n = 1 + (seed % 4) as usize;
            let m = seeded_matrix(seed, n, n, &p);
            let Ok((pm, c)) = rational_canonical_form(&m, &p) else {
                continue;
            };
            answered += 1;
            let lhs = m.mul(&pm, &p).unwrap();
            let rhs = pm.mul(&c, &p).unwrap();
            assert!(
                eigen::matrix_eq_simplified(&lhs, &rhs, &p),
                "seed {seed}: M·P = P·C failed"
            );
            assert_eq!(rank(&pm, &p).unwrap(), n, "seed {seed}: P singular");
        }
        assert!(
            answered > 30,
            "the gate must not have turned this into a refusal machine; \
             only {answered} of 60 answered"
        );
    }

    // ----------------------------------------------------------- cholesky

    /// `L·Lᵀ` is symmetric for every `L`, so a non-symmetric input has no
    /// Cholesky factor and must be refused.
    ///
    /// The algorithm reads only the lower triangle, so `[[1,5],[0,1]]` used to
    /// come back as the identity — a factorisation of `I`, not of the input,
    /// with nothing to distinguish it from a real answer.
    #[test]
    fn cholesky_refuses_a_non_symmetric_matrix() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), p.integer(5_i32)],
            vec![p.integer(0_i32), p.integer(1_i32)],
        ])
        .unwrap();
        assert_eq!(
            cholesky(&m, &p),
            Err(LinearAlgebraError::NotPositiveDefinite),
            "a non-symmetric matrix has no Cholesky factor"
        );
    }

    /// `2I` is positive definite and its Cholesky factor is `√2·I`.
    ///
    /// The rational path used to require every pivot to be a perfect rational
    /// square and reported `E-LINALG-003` — "matrix is not symmetric positive
    /// definite" — otherwise. A refusal that states a falsehood about the input
    /// is not a safe failure.
    #[test]
    fn cholesky_of_2i_is_sqrt2_i() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(2_i32), p.integer(0_i32)],
            vec![p.integer(0_i32), p.integer(2_i32)],
        ])
        .unwrap();
        let l = cholesky(&m, &p).unwrap();
        let recon = l.mul(&l.transpose(), &p).unwrap();
        assert!(
            entries_equal(&recon, &m, &p),
            "L·Lᵀ must be 2I, got {:?}",
            recon.entries()
        );
        let sqrt2 = simplify(p.func("sqrt", vec![p.integer(2_i32)]), &p).value;
        assert_eq!(simplify(l.get(0, 0), &p).value, sqrt2);
        assert_eq!(simplify(l.get(1, 0), &p).value, p.integer(0_i32));
    }

    /// `L·Lᵀ = M` over a family of symmetric positive definite matrices, with
    /// and without perfect-square pivots.
    #[test]
    fn cholesky_reconstructs_every_spd_matrix() {
        let p = pool();
        let mut answered = 0usize;
        for seed in 600..660u64 {
            let n = 1 + (seed % 4) as usize;
            let b = seeded_matrix(seed, n, n, &p);
            // `BᵀB + n·I` is symmetric and positive definite by construction.
            let m = b
                .transpose()
                .mul(&b, &p)
                .unwrap()
                .add(&Matrix::identity(n, &p).scale(p.integer(n as i32), &p), &p)
                .unwrap()
                .simplify_entries(&p);
            let l = cholesky(&m, &p).expect("BᵀB + nI is positive definite");
            answered += 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    assert_eq!(
                        simplify(l.get(i, j), &p).value,
                        p.integer(0_i32),
                        "seed {seed}: L must be lower triangular"
                    );
                }
            }
            let recon = l.mul(&l.transpose(), &p).unwrap();
            assert!(
                entries_equal(&recon, &m, &p),
                "seed {seed}: L·Lᵀ = M failed"
            );
        }
        assert_eq!(answered, 60);
    }

    /// A symmetric matrix that is *not* positive definite is still refused —
    /// the `√2` repair must not have turned the definiteness test off.
    #[test]
    fn cholesky_still_refuses_an_indefinite_matrix() {
        let p = pool();
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), p.integer(2_i32)],
            vec![p.integer(2_i32), p.integer(1_i32)],
        ])
        .unwrap();
        assert_eq!(
            cholesky(&m, &p),
            Err(LinearAlgebraError::NotPositiveDefinite),
            "[[1,2],[2,1]] has eigenvalues 3 and −1"
        );
        let neg = Matrix::new(vec![
            vec![p.integer(-1_i32), p.integer(0_i32)],
            vec![p.integer(0_i32), p.integer(-1_i32)],
        ])
        .unwrap();
        assert_eq!(
            cholesky(&neg, &p),
            Err(LinearAlgebraError::NotPositiveDefinite)
        );
    }

    // ------------------------------------------------ symbolic inverse 6×6

    /// A fully symbolic `n×n` matrix is invertible and says so, up to `6×6`.
    ///
    /// `det` of a matrix of distinct free symbols is a polynomial with `n!`
    /// monomials and is not the zero function, so `adj/det` is the inverse and
    /// nothing is being assumed. It refused at `n ≥ 5` only because the
    /// non-vanishing probe declined to bind more than 16 symbols and a `5×5`
    /// has 25; see [`crate::matrix::zero_test`]. The verification is the
    /// identity itself: `A⁻¹·A = I`.
    #[test]
    fn symbolic_inverse_handles_a_fully_symbolic_six_by_six() {
        let p = pool();
        for n in 1..=6usize {
            // Names are keyed by `n` so each size interns its symbols fresh,
            // in row-major order. Reusing one name set across the loop made
            // every size after the first inherit a *scrambled* `ExprId` order,
            // which by itself was enough to hide the collinear-sample defect in
            // `zero_test::sample_ball` that this test is here to pin.
            let grid: Vec<Vec<ExprId>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| p.symbol(format!("a{n}_{i}_{j}"), Domain::Complex))
                        .collect()
                })
                .collect();
            let m = Matrix::new(grid).unwrap();
            let inv = matrix_inverse(&m, &p)
                .unwrap_or_else(|e| panic!("{n}×{n} fully symbolic inverse refused: {e}"));
            let prod = inv.mul(&m, &p).unwrap();
            // Verified by substitution into ℚ rather than by symbolic
            // cancellation: `A⁻¹·A − I` for a dense symbolic `A` is a rational
            // function in `n²` variables that no normaliser here reduces in
            // useful time, whereas specialising the symbols to distinct
            // rationals makes every entry an exact rational and the check
            // decisive. Three unrelated points, so an accidental agreement
            // would have to happen three times.
            let id = Matrix::identity(n, &p);
            for round in 0..3usize {
                let mut map = std::collections::HashMap::new();
                // Unstructured sample points. An arithmetic progression in
                // `i·n + j` makes the matrix itself rank-deficient — every row
                // differs from the previous by a constant vector — so `det`
                // vanishes at the sample and the check is vacuous rather than
                // decisive. An LCG has no such structure.
                let mut seed = 0x2545_F491_4F6C_DD1Du64 ^ (round as u64).wrapping_mul(0x9E37_79B9);
                let mut next = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed % 199) as i64 + 1
                };
                for i in 0..n {
                    for j in 0..n {
                        let sym = p.symbol(format!("a{n}_{i}_{j}"), Domain::Complex);
                        map.insert(sym, p.rational(next(), 7 + 2 * round as i64));
                    }
                }
                for i in 0..n {
                    for j in 0..n {
                        let diff = p.add(vec![
                            prod.get(i, j),
                            p.mul(vec![p.integer(-1_i32), id.get(i, j)]),
                        ]);
                        let at = crate::kernel::subs::subs(diff, &map, &p);
                        let v = exact_rational(at, &p).unwrap_or_else(|| {
                            panic!(
                                "{n}×{n} round {round}: entry ({i},{j}) is not a constant: {}",
                                crate::kernel::display::render_unicode(at, &p)
                            )
                        });
                        assert_eq!(
                            v,
                            Rational::from(0),
                            "{n}×{n} round {round}: (A⁻¹·A − I)[{i}][{j}] must vanish"
                        );
                    }
                }
            }
        }
    }

    /// A symbolically singular matrix is still refused, at every size.
    ///
    /// The point of raising the probe cap is to stop declining *decidable*
    /// determinants, not to start inventing inverses. Row 1 is `2×` row 0 here,
    /// so `det` is identically zero and `E-MAT-003` is owed.
    #[test]
    fn symbolic_inverse_still_refuses_a_dependent_row() {
        let p = pool();
        for n in 2..=5usize {
            let grid: Vec<Vec<ExprId>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let s = p.symbol(format!("b_{i}_{j}"), Domain::Complex);
                            // Row 1 is exactly twice row 0, so `det` is the
                            // zero polynomial however the symbols are read.
                            if i == 1 {
                                let s0 = p.symbol(format!("b_0_{j}"), Domain::Complex);
                                p.mul(vec![p.integer(2_i32), s0])
                            } else {
                                s
                            }
                        })
                        .collect()
                })
                .collect();
            let m = Matrix::new(grid).unwrap();
            assert_eq!(
                matrix_inverse(&m, &p),
                Err(MatrixError::SingularMatrix),
                "{n}×{n} with a dependent row must be refused"
            );
        }
    }

    /// The `det ≠ 0` a symbolic inverse rests on is *stated*, not assumed.
    ///
    /// `adj/det` is the inverse of `[[a,b],[c,d]]` exactly where `ad − bc` does
    /// not vanish. That the determinant is not the zero *function* is what
    /// licenses the formula; that it is non-zero at a given point is a question
    /// about the parameters, and the only honest thing to do with it is hand it
    /// back. Asserted against the determinant computed independently here, not
    /// against whatever the channel happened to contain.
    #[test]
    fn a_symbolic_inverse_reports_the_determinant_it_divided_by() {
        let p = pool();
        let a = p.symbol("a", Domain::Complex);
        let b = p.symbol("b", Domain::Complex);
        let c = p.symbol("c", Domain::Complex);
        let d = p.symbol("d", Domain::Complex);
        let m = Matrix::new(vec![vec![a, b], vec![c, d]]).unwrap();
        let _ = crate::matrix::take_matrix_inverse_side_conditions();
        matrix_inverse(&m, &p).expect("[[a,b],[c,d]] is generically invertible");
        let conds = crate::matrix::take_matrix_inverse_side_conditions();
        assert_eq!(conds.len(), 1, "exactly one hypothesis: det ≠ 0");
        let crate::deriv::SideCondition::NonZero(recorded) = conds[0] else {
            panic!("the inverse's hypothesis is a non-vanishing one");
        };
        // `ad − bc`, built here rather than read off the library.
        let want = simplify_expanded(
            p.add(vec![
                p.mul(vec![a, d]),
                p.mul(vec![p.integer(-1_i32), b, c]),
            ]),
            &p,
        )
        .value;
        let diff = simplify_expanded(
            p.add(vec![recorded, p.mul(vec![p.integer(-1_i32), want])]),
            &p,
        )
        .value;
        assert!(
            zero_test::zero_status(&p, diff).is_proven_zero(),
            "the recorded condition must be the determinant itself"
        );
        // Consuming: a second read must not repeat the first call's hypothesis.
        assert!(crate::matrix::take_matrix_inverse_side_conditions().is_empty());
    }

    /// A determinant that is a non-zero *constant* is discharged, not recorded.
    ///
    /// A gate made only of "the condition is reported" cases is passed by a
    /// library that reports one unconditionally, which is noise a caller learns
    /// to ignore — the failure mode the transform work called out by name when
    /// it stopped reporting `π ≠ 0`.
    #[test]
    fn a_constant_determinant_records_no_hypothesis() {
        let p = pool();
        let _ = crate::matrix::take_matrix_inverse_side_conditions();
        // Rational fast path.
        let m = Matrix::new(vec![
            vec![p.integer(1_i32), p.integer(2_i32)],
            vec![p.integer(3_i32), p.integer(4_i32)],
        ])
        .unwrap();
        matrix_inverse(&m, &p).unwrap();
        assert!(crate::matrix::take_matrix_inverse_side_conditions().is_empty());

        // Symbolic entries, unit determinant: `[[cos t, -sin t], [sin t, cos t]]`
        // would need a trig identity, so use the shear `[[1, s], [0, 1]]`, whose
        // determinant is the literal `1` however `s` is read.
        let s = p.symbol("s", Domain::Complex);
        let shear = Matrix::new(vec![
            vec![p.integer(1_i32), s],
            vec![p.integer(0_i32), p.integer(1_i32)],
        ])
        .unwrap();
        matrix_inverse(&shear, &p).unwrap();
        assert!(
            crate::matrix::take_matrix_inverse_side_conditions().is_empty(),
            "det = 1 is not a hypothesis"
        );
    }

    /// A refusal leaves nothing behind for the next call to inherit.
    #[test]
    fn a_refused_inverse_records_no_hypothesis() {
        let p = pool();
        let a = p.symbol("q", Domain::Complex);
        let b = p.symbol("r", Domain::Complex);
        let m = Matrix::new(vec![
            vec![a, b],
            vec![
                p.mul(vec![p.integer(2_i32), a]),
                p.mul(vec![p.integer(2_i32), b]),
            ],
        ])
        .unwrap();
        let _ = crate::matrix::take_matrix_inverse_side_conditions();
        assert_eq!(matrix_inverse(&m, &p), Err(MatrixError::SingularMatrix));
        assert!(crate::matrix::take_matrix_inverse_side_conditions().is_empty());
    }
}
