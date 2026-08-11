//! Symbolic linear algebra: nullspace, rank, LU/QR/Cholesky, Jordan and rational canonical
//! forms, minimal polynomial, and matrix exponential.

#![allow(clippy::needless_range_loop)]

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::matrix::eigen::{
    self, characteristic_polynomial_lambda_minus_m, concatenate_columns, kernel_column_basis,
    m_minus_lambda_scaled, KernelFailure, KnownSingular,
};
use crate::matrix::normal_form::{smith_form_poly, PolyMatrixQ, RatUniPoly};
use crate::matrix::{zero_test, Matrix, MatrixError};
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
    /// The entries lie outside the field this routine can work over.
    ///
    /// Two ways that happens: a Smith-based decomposition needs rational
    /// constants and got something else, or elimination reached an entry whose
    /// vanishing it could not decide — over a transcendental extension that
    /// question is undecidable in general (see the `matrix::zero_test` module),
    /// and pivoting on an entry that might be identically zero is what produced
    /// full-rank verdicts for rank-deficient matrices.
    ///
    /// Which of the two it was is available from
    /// [`take_zero_test_refusal`](crate::matrix::take_zero_test_refusal):
    /// `Some(..)` means an undecided entry (code `E-LINALG-010`), `None` means
    /// non-rational entries (code `E-LINALG-007`).
    UnsupportedField,
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
                    "matrix entries lie outside the field this routine can work over: \
                     a Smith-based decomposition needs rational constants, and \
                     elimination needs entries whose vanishing it can decide"
                )
            }
            LinearAlgebraError::SingularTransform => {
                write!(f, "similarity transform matrix is singular")
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
            LinearAlgebraError::SingularTransform => {
                Some("the computed similarity matrix is not invertible")
            }
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

/// Basis of the row space of `m` (nonzero pivot rows in echelon form).
pub fn row_space_basis(m: &Matrix, pool: &ExprPool) -> Result<Vec<Matrix>, LinearAlgebraError> {
    let rref = row_echelon_pivots(m, pool)?;
    Ok(rref
        .pivot_row_flags
        .iter()
        .enumerate()
        .filter_map(|(ri, &is_pivot)| {
            if is_pivot {
                Some(Matrix::new(vec![m.row(ri)]).expect("row vector"))
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

pub fn cholesky(m: &Matrix, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    if m.rows != m.cols {
        return Err(LinearAlgebraError::NonSquare);
    }
    let n = m.rows;
    if let Some(a) = matrix_to_rational_grid(m, pool) {
        let mut l = vec![vec![Rational::from(0); n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = Rational::from(0);
                for t in 0..j {
                    s += l[i][t].clone() * l[j][t].clone();
                }
                if i == j {
                    let diag = a[i][i].clone() - s;
                    if diag <= 0 {
                        return Err(LinearAlgebraError::NotPositiveDefinite);
                    }
                    l[i][j] =
                        rational_sqrt(&diag).ok_or(LinearAlgebraError::NotPositiveDefinite)?;
                } else {
                    if l[j][j] == 0 {
                        return Err(LinearAlgebraError::NotPositiveDefinite);
                    }
                    l[i][j] = (a[i][j].clone() - s) / l[j][j].clone();
                }
            }
        }
        return Ok(rational_grid_to_matrix(&l, pool));
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

// ---------------------------------------------------------------------------
// Jordan form
// ---------------------------------------------------------------------------

/// `(P, J)` with `M = P·J·P⁻¹`.
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
            let v_top = bas.last().ok_or(LinearAlgebraError::KernelFailed)?.clone();
            let mut chain = vec![v_top.clone()];
            let mut cur = v_top;
            for _ in 1..sz {
                cur = shifted
                    .mul(&cur, pool)
                    .map_err(|_| LinearAlgebraError::KernelFailed)?;
                chain.push(cur.clone());
            }
            chain.reverse();
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
    let j = block_diagonal(&j_blocks, pool)?;
    Ok((p, j))
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
    Ok((p, c))
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
        let p = s.get(i, i).clone();
        if !p.is_zero() && p.degree() > 0 {
            facs.push(p);
        }
    }
    if facs.is_empty() {
        return Err(unsupported_field());
    }
    Ok(facs)
}

fn companion_matrix(f: &RatUniPoly, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    let d = f.degree() as usize;
    if d == 0 {
        return Err(unsupported_field());
    }
    let coeffs = f.coeffs.clone();
    let mut c = Matrix::zeros(d, d, pool);
    for i in 0..d - 1 {
        c.set(i + 1, i, pool.integer(1_i32));
    }
    for j in 0..d {
        let coeff = if j < coeffs.len() {
            pool.rational(coeffs[j].numer().clone(), coeffs[j].denom().clone())
        } else {
            pool.integer(0_i32)
        };
        c.set(
            d - 1,
            j,
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

fn frobenius_p_from_cyclic_vectors(
    m: &Matrix,
    factors: &[RatUniPoly],
    pool: &ExprPool,
) -> Result<Matrix, LinearAlgebraError> {
    let n = m.rows;
    let mut cols: Vec<Matrix> = Vec::with_capacity(n);
    let mut idx = 0usize;
    for f in factors {
        let d = f.degree() as usize;
        let chain = cyclic_column_chain(m, idx, d, n, pool)?;
        cols.extend(chain);
        idx += d;
    }
    if cols.len() != n {
        return Err(LinearAlgebraError::KernelFailed);
    }
    concatenate_columns(&cols, pool).map_err(|_| LinearAlgebraError::KernelFailed)
}

/// Build `v, M v, …, M^{d-1} v` with `v = e_start`, trying later unit vectors if the chain stalls.
fn cyclic_column_chain(
    m: &Matrix,
    start_col: usize,
    d: usize,
    n: usize,
    pool: &ExprPool,
) -> Result<Vec<Matrix>, LinearAlgebraError> {
    let mut seeds: Vec<Matrix> = (start_col..n)
        .filter_map(|c| unit_column_vector(c, n, pool).ok())
        .collect();
    if d > 1 {
        for i in start_col..n {
            for j in (i + 1)..n {
                if let Ok(v) = sum_unit_columns(i, j, n, pool) {
                    seeds.push(v);
                }
            }
        }
    }
    for v in seeds {
        let mut chain = vec![v.clone()];
        let mut cur = v;
        let mut ok = true;
        for _ in 1..d {
            cur = m
                .mul(&cur, pool)
                .map_err(|_| LinearAlgebraError::KernelFailed)?;
            if columns_proportional(&cur, chain.last().unwrap(), pool) {
                ok = false;
                break;
            }
            chain.push(cur.clone());
        }
        if ok && chain.len() == d {
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

fn columns_proportional(a: &Matrix, b: &Matrix, pool: &ExprPool) -> bool {
    if a.rows != b.rows {
        return false;
    }
    let mut ratio: Option<ExprId> = None;
    for r in 0..a.rows {
        let ea = simplify(a.get(r, 0), pool).value;
        let eb = simplify(b.get(r, 0), pool).value;
        // A heuristic search over candidate cyclic vectors: an undecided entry
        // only makes this candidate look unusable, and the caller tries the
        // next one. No mathematical claim rides on `false` here.
        let ea_zero = zero_test::zero_status(pool, ea).is_proven_zero();
        let eb_zero = zero_test::zero_status(pool, eb).is_proven_zero();
        if ea_zero && eb_zero {
            continue;
        }
        if ea_zero || eb_zero {
            return false;
        }
        let cand = simplify(pool.mul(vec![ea, pool.pow(eb, pool.integer(-1_i32))]), pool).value;
        match ratio {
            None => ratio = Some(cand),
            Some(rv) => {
                if simplify(
                    pool.add(vec![cand, pool.mul(vec![pool.integer(-1_i32), rv])]),
                    pool,
                )
                .value
                    != simplify(pool.integer(0_i32), pool).value
                {
                    return false;
                }
            }
        }
    }
    ratio.is_some()
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
    let mut acc = Matrix::zeros(n, n, pool);
    let mut pow = Matrix::identity(n, pool);
    for (deg, coeff) in p.coefficients().iter().enumerate() {
        if coeff.is_zero() {
            if deg > 0 {
                pow = pow
                    .mul(m, pool)
                    .map_err(|_| LinearAlgebraError::KernelFailed)?;
            }
            continue;
        }
        let c = pool.rational(coeff.clone(), rug::Integer::from(1));
        let term = pow.scale(c, pool);
        acc = acc
            .add(&term, pool)
            .map_err(|_| LinearAlgebraError::KernelFailed)?;
        if deg + 1 < p.coefficients().len() {
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
        let c = pool.rational(coeff.clone(), rug::Integer::from(1));
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
    if terms.is_empty() {
        pool.integer(0_i32)
    } else {
        simplify(pool.add(terms), pool).value
    }
}

// ---------------------------------------------------------------------------
// Matrix exponential
// ---------------------------------------------------------------------------

pub fn matrix_exponential(m: &Matrix, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    if m.rows != m.cols {
        return Err(LinearAlgebraError::NonSquare);
    }
    // A diagonal matrix (possibly with free-symbol entries) exponentiates entrywise:
    // exp(diag(d₀, …, dₙ)) = diag(e^{d₀}, …, e^{dₙ}). Short-circuit so symbolic diagonal /
    // decoupled state matrices succeed without invoking the eigenvector machinery, whose
    // radical eigenvalues can collapse the eigenbasis for these cases.
    if is_diagonal(m, pool) {
        return diagonal_matrix_exp(m, pool);
    }
    if let Ok((p, d)) = eigen::diagonalize(m, pool) {
        let exp_d = diagonal_matrix_exp(&d, pool)?;
        let inv_p = matrix_inverse(&p, pool).map_err(|_| LinearAlgebraError::SingularTransform)?;
        return p
            .mul(&exp_d, pool)
            .map_err(|_| LinearAlgebraError::KernelFailed)?
            .mul(&inv_p, pool)
            .map_err(|_| LinearAlgebraError::KernelFailed);
    }
    let (p, j) = jordan_form(m, pool)?;
    let exp_j = jordan_matrix_exp(&j, pool)?;
    let inv_p = matrix_inverse(&p, pool).map_err(|_| LinearAlgebraError::SingularTransform)?;
    p.mul(&exp_j, pool)
        .map_err(|_| LinearAlgebraError::KernelFailed)?
        .mul(&inv_p, pool)
        .map_err(|_| LinearAlgebraError::KernelFailed)
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

fn jordan_matrix_exp(j: &Matrix, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    let n = j.rows;
    let mut out = Matrix::zeros(n, n, pool);
    let mut i = 0usize;
    while i < n {
        let lambda = j.get(i, i);
        let mut sz = 1usize;
        while i + sz < n
            && j.get(i, i + sz) == pool.integer(1_i32)
            && j.get(i + sz, i + sz) == lambda
        {
            sz += 1;
        }
        let block = jordan_block_exp(lambda, sz, pool)?;
        for bi in 0..sz {
            for bj in 0..sz {
                out.set(i + bi, i + bj, block.get(bi, bj));
            }
        }
        i += sz;
    }
    Ok(out)
}

fn jordan_block_exp(
    lambda: ExprId,
    size: usize,
    pool: &ExprPool,
) -> Result<Matrix, LinearAlgebraError> {
    let mut out = Matrix::zeros(size, size, pool);
    let elam = simplify(pool.func("exp", vec![lambda]), pool).value;
    for i in 0..size {
        for j in i..size {
            let k = j - i;
            let fact = pool.integer(factorial_i64(k) as i32);
            let pow = if k == 0 {
                pool.integer(1_i32)
            } else {
                pool.pow(lambda, pool.integer(k as i32))
            };
            out.set(
                i,
                j,
                simplify(
                    pool.mul(vec![elam, pow, pool.pow(fact, pool.integer(-1_i32))]),
                    pool,
                )
                .value,
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
fn apply_row_permutation(m: &Matrix, perm: &[usize]) -> Matrix {
    let rows: Vec<Vec<ExprId>> = perm.iter().map(|&r| m.row(r)).collect();
    Matrix::new(rows).expect("row permutation")
}

fn factorial_i64(k: usize) -> i64 {
    (1..=k).fold(1i64, |a, b| a.saturating_mul(b as i64))
}

// ---------------------------------------------------------------------------
// Matrix inverse (for similarity transforms)
// ---------------------------------------------------------------------------

pub fn matrix_inverse(m: &Matrix, pool: &ExprPool) -> Result<Matrix, MatrixError> {
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
        zero_test::ZeroStatus::NonZero => {}
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
        .map(|row| {
            row.iter()
                .map(|r| pool.rational(r.numer().clone(), r.denom().clone()))
                .collect()
        })
        .collect();
    Matrix::new(rows).expect("rational grid")
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
}
