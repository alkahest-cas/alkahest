//! Symbolic linear algebra: nullspace, rank, LU/QR/Cholesky, Jordan and rational canonical
//! forms, minimal polynomial, and matrix exponential.

#![allow(clippy::needless_range_loop)]

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::matrix::eigen::{
    self, characteristic_polynomial_lambda_minus_m, concatenate_columns, kernel_column_basis,
    m_minus_lambda_scaled,
};
use crate::matrix::normal_form::{smith_form_poly, PolyMatrixQ, RatUniPoly};
use crate::matrix::{Matrix, MatrixError};
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
    UnsupportedIrreducibleDegree { degree: usize },
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
                    "entries must be rational constants for this decomposition"
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
            LinearAlgebraError::UnsupportedField => {
                Some("use rational or integer entries for Smith-based decompositions")
            }
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
pub fn nullspace_basis(m: &Matrix, pool: &ExprPool) -> Result<Vec<Matrix>, LinearAlgebraError> {
    kernel_column_basis(m, pool).map_err(|()| LinearAlgebraError::KernelFailed)
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
        let mut prow = None;
        for rr in r_at..rows {
            let e = simplify(a[rr][c], pool).value;
            if !expr_is_zero(pool, e) {
                prow = Some((rr, e));
                break;
            }
        }
        let Some((pr, piv)) = prow else { continue };
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
            if expr_is_zero(pool, f) {
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
        let mut piv_row = k;
        for r in (k + 1)..n {
            if !expr_is_zero(pool, a[r][k]) && expr_is_zero(pool, a[piv_row][k]) {
                piv_row = r;
            }
        }
        if expr_is_zero(pool, a[piv_row][k]) {
            for j in k..cols {
                u.set(k, j, a[k][j]);
            }
            continue;
        }
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
        if expr_is_zero(pool, rjj) {
            r.set(j, j, pool.integer(0_i32));
            q_cols.push(Matrix::zeros(n, 1, pool));
            continue;
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
            ker_dims.push(
                kernel_column_basis(&pow, pool)
                    .map_err(|_| LinearAlgebraError::KernelFailed)?
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
            let bas =
                kernel_column_basis(&nk, pool).map_err(|_| LinearAlgebraError::KernelFailed)?;
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
    PolyMatrixQ::from_nested(rows).map_err(|_| LinearAlgebraError::UnsupportedField)
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
        return Err(LinearAlgebraError::UnsupportedField);
    }
    Ok(facs)
}

fn companion_matrix(f: &RatUniPoly, pool: &ExprPool) -> Result<Matrix, LinearAlgebraError> {
    let d = f.degree() as usize;
    if d == 0 {
        return Err(LinearAlgebraError::UnsupportedField);
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
        if expr_is_zero(pool, ea) && expr_is_zero(pool, eb) {
            continue;
        }
        if expr_is_zero(pool, ea) || expr_is_zero(pool, eb) {
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
        if !expr_is_zero(pool, simplify(*e, pool).value) {
            return Ok(false);
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

/// True iff every off-diagonal entry simplifies to exactly zero.
fn is_diagonal(m: &Matrix, pool: &ExprPool) -> bool {
    if m.rows != m.cols {
        return false;
    }
    for r in 0..m.rows {
        for c in 0..m.cols {
            if r != c && !expr_is_zero(pool, simplify(m.get(r, c), pool).value) {
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
            return Err(MatrixError::SingularMatrix);
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
/// If `det(A)` simplifies to a literal zero the matrix is genuinely singular and
/// `MatrixError::SingularMatrix` is returned, keeping that error meaningful.
fn symbolic_inverse(m: &Matrix, pool: &ExprPool) -> Result<Matrix, MatrixError> {
    let n = m.rows;
    if n == 0 {
        return Ok(Matrix::zeros(0, 0, pool));
    }
    // Expand the determinant into canonical polynomial form so that the shared
    // `1/det` factor in the resulting entries cancels cleanly against expanded
    // cofactor numerators (e.g. so A·A⁻¹ collapses to the identity on simplify).
    let det = simplify_expanded(m.det(pool)?, pool).value;
    if expr_is_zero(pool, det) {
        return Err(MatrixError::SingularMatrix);
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
    Matrix::new(rows).map_err(|_| MatrixError::SingularMatrix)
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

fn expr_is_zero(pool: &ExprPool, e: ExprId) -> bool {
    match pool.get(e) {
        ExprData::Integer(n) => n.0 == 0,
        ExprData::Rational(r) => r.0 == 0,
        _ => false,
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
        assert!(!expr_is_zero(&p, expm.get(0, 0)));
        assert!(!expr_is_zero(&p, expm.get(1, 1)));
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
                    expr_is_zero(&p, diff),
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
