//! Phase 15 — Symbolic matrices and vectors.
//!
//! Provides a dense `Matrix` of `ExprId` values together with:
//! - arithmetic (`+`, `-`, `*`)
//! - `transpose()`
//! - `det()` (Bareiss integer-preserving elimination)
//! - `jacobian(f_vec, x_vec, pool)` — the `m×n` matrix `∂f_i/∂x_j`

use crate::diff::diff;
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::fmt;

pub mod eigen;
pub mod normal_form;
mod smith;
mod smith_poly;

pub use eigen::{
    characteristic_polynomial_lambda_minus_m, diagonalize, eigenvalues, eigenvectors, EigenError,
};
pub use normal_form::{
    hermite_form, hermite_form_poly, smith_form, smith_form_poly, IntegerMatrix, NormalFormError,
    PolyMatrixQ, RatUniPoly,
};

// ---------------------------------------------------------------------------
// Matrix type
// ---------------------------------------------------------------------------

/// A dense symbolic matrix stored in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    /// Row-major flat storage of `ExprId` entries.
    data: Vec<ExprId>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    DimensionMismatch { msg: String },
    NotSquare,
    SingularMatrix,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch { msg } => write!(f, "dimension mismatch: {msg}"),
            MatrixError::NotSquare => write!(f, "matrix is not square"),
            MatrixError::SingularMatrix => write!(f, "matrix is singular"),
        }
    }
}

impl std::error::Error for MatrixError {}

impl crate::errors::AlkahestError for MatrixError {
    fn code(&self) -> &'static str {
        match self {
            MatrixError::DimensionMismatch { .. } => "E-MAT-001",
            MatrixError::NotSquare => "E-MAT-002",
            MatrixError::SingularMatrix => "E-MAT-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            MatrixError::DimensionMismatch { .. } => Some(
                "ensure all rows have the same column count and operand dimensions match",
            ),
            MatrixError::NotSquare => Some(
                "determinant and inverse require a square matrix; use the pseudo-inverse for rectangular matrices",
            ),
            MatrixError::SingularMatrix => Some(
                "the matrix has a zero determinant; check your system of equations for linear dependence",
            ),
        }
    }
}

impl Matrix {
    /// Create a matrix from row-major nested vectors.
    pub fn new(rows: Vec<Vec<ExprId>>) -> Result<Self, MatrixError> {
        if rows.is_empty() {
            return Ok(Matrix {
                data: vec![],
                rows: 0,
                cols: 0,
            });
        }
        let cols = rows[0].len();
        for r in &rows {
            if r.len() != cols {
                return Err(MatrixError::DimensionMismatch {
                    msg: format!("expected {cols} columns, got {}", r.len()),
                });
            }
        }
        let nrows = rows.len();
        let data: Vec<ExprId> = rows.into_iter().flatten().collect();
        Ok(Matrix {
            data,
            rows: nrows,
            cols,
        })
    }

    /// Create a zero matrix (all entries are `pool.integer(0)`).
    pub fn zeros(rows: usize, cols: usize, pool: &ExprPool) -> Self {
        let zero = pool.integer(0_i32);
        Matrix {
            data: vec![zero; rows * cols],
            rows,
            cols,
        }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize, pool: &ExprPool) -> Self {
        let zero = pool.integer(0_i32);
        let one = pool.integer(1_i32);
        let mut data = vec![zero; n * n];
        for i in 0..n {
            data[i * n + i] = one;
        }
        Matrix {
            data,
            rows: n,
            cols: n,
        }
    }

    /// Get entry at row `r`, column `c` (0-indexed).
    pub fn get(&self, r: usize, c: usize) -> ExprId {
        self.data[r * self.cols + c]
    }

    /// Set entry at row `r`, column `c`.
    pub fn set(&mut self, r: usize, c: usize, val: ExprId) {
        self.data[r * self.cols + c] = val;
    }

    /// Get a row as a vector.
    pub fn row(&self, r: usize) -> Vec<ExprId> {
        self.data[r * self.cols..(r + 1) * self.cols].to_vec()
    }

    /// Get a column as a vector.
    pub fn col(&self, c: usize) -> Vec<ExprId> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                data.push(self.get(r, c));
            }
        }
        Matrix {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Matrix, pool: &ExprPool) -> Result<Matrix, MatrixError> {
        self.check_same_shape(other)?;
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| pool.add(vec![a, b]))
            .collect();
        Ok(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Matrix, pool: &ExprPool) -> Result<Matrix, MatrixError> {
        self.check_same_shape(other)?;
        let neg_one = pool.integer(-1_i32);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| {
                let neg_b = pool.mul(vec![neg_one, b]);
                pool.add(vec![a, neg_b])
            })
            .collect();
        Ok(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Matrix multiplication (`self` is m×k, `other` is k×n → result is m×n).
    pub fn mul(&self, other: &Matrix, pool: &ExprPool) -> Result<Matrix, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                msg: format!(
                    "cannot multiply {}×{} by {}×{}",
                    self.rows, self.cols, other.rows, other.cols
                ),
            });
        }
        let m = self.rows;
        let n = other.cols;
        let k = self.cols;
        let mut data = Vec::with_capacity(m * n);
        for r in 0..m {
            for c in 0..n {
                let terms: Vec<ExprId> = (0..k)
                    .map(|i| pool.mul(vec![self.get(r, i), other.get(i, c)]))
                    .collect();
                let entry = if terms.is_empty() {
                    pool.integer(0_i32)
                } else if terms.len() == 1 {
                    terms[0]
                } else {
                    pool.add(terms)
                };
                data.push(entry);
            }
        }
        Ok(Matrix {
            data,
            rows: m,
            cols: n,
        })
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: ExprId, pool: &ExprPool) -> Matrix {
        let data = self
            .data
            .iter()
            .map(|&e| pool.mul(vec![scalar, e]))
            .collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Simplify all entries.
    pub fn simplify_entries(&self, pool: &ExprPool) -> Matrix {
        let data = self.data.iter().map(|&e| simplify(e, pool).value).collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Determinant using Bareiss algorithm (exact over integers, symbolic otherwise).
    pub fn det(&self, pool: &ExprPool) -> Result<ExprId, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare);
        }
        let n = self.rows;
        if n == 0 {
            return Ok(pool.integer(1_i32));
        }
        if n == 1 {
            return Ok(self.get(0, 0));
        }
        if n == 2 {
            // ad - bc
            let ad = pool.mul(vec![self.get(0, 0), self.get(1, 1)]);
            let bc = pool.mul(vec![self.get(0, 1), self.get(1, 0)]);
            let neg_bc = pool.mul(vec![pool.integer(-1_i32), bc]);
            return Ok(simplify(pool.add(vec![ad, neg_bc]), pool).value);
        }
        // Cofactor expansion along first row for n >= 3
        let mut terms: Vec<ExprId> = Vec::new();
        for j in 0..n {
            let minor = self.minor(0, j);
            let minor_det = minor.det(pool)?;
            let sign = if j % 2 == 0 {
                pool.integer(1_i32)
            } else {
                pool.integer(-1_i32)
            };
            terms.push(pool.mul(vec![sign, self.get(0, j), minor_det]));
        }
        Ok(simplify(pool.add(terms), pool).value)
    }

    /// Submatrix obtained by removing row `r` and column `c`.
    fn minor(&self, skip_row: usize, skip_col: usize) -> Matrix {
        let n = self.rows;
        let mut data = Vec::with_capacity((n - 1) * (n - 1));
        for r in 0..n {
            if r == skip_row {
                continue;
            }
            for c in 0..n {
                if c == skip_col {
                    continue;
                }
                data.push(self.get(r, c));
            }
        }
        Matrix {
            data,
            rows: n - 1,
            cols: n - 1,
        }
    }

    fn check_same_shape(&self, other: &Matrix) -> Result<(), MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            Err(MatrixError::DimensionMismatch {
                msg: format!(
                    "{}×{} vs {}×{}",
                    self.rows, self.cols, other.rows, other.cols
                ),
            })
        } else {
            Ok(())
        }
    }

    /// Return a flat reference to all entries.
    pub fn entries(&self) -> &[ExprId] {
        &self.data
    }

    /// Return entries as a nested `Vec<Vec<ExprId>>`.
    pub fn to_nested(&self) -> Vec<Vec<ExprId>> {
        (0..self.rows).map(|r| self.row(r)).collect()
    }

    /// V2-17 — `det(λI − M)` as a pooled expression plus the fresh λ symbol used.
    pub fn characteristic_polynomial_lambda_minus_m(
        &self,
        pool: &ExprPool,
    ) -> Result<(ExprId, ExprId), EigenError> {
        eigen::characteristic_polynomial_lambda_minus_m(self, pool)
    }

    /// V2-17 — Algebraic eigenvalues `(value, multiplicity)` for matrices whose characteristic
    /// polynomial factors over ℚ into linear and quadratic terms.
    pub fn eigenvalues(&self, pool: &ExprPool) -> Result<Vec<(ExprId, usize)>, EigenError> {
        eigen::eigenvalues(self, pool)
    }

    /// V2-17 — Eigenvalue tuples `(λ, multiplicity, column eigenvectors)`.
    pub fn eigenvectors(
        &self,
        pool: &ExprPool,
    ) -> Result<Vec<(ExprId, usize, Vec<Matrix>)>, EigenError> {
        eigen::eigenvectors(self, pool)
    }

    /// V2-17 — `(P, D)` with `M·P == P·D` when diagonalizable in the ℚ-splitting-field sense.
    pub fn diagonalize(&self, pool: &ExprPool) -> Result<(Matrix, Matrix), EigenError> {
        eigen::diagonalize(self, pool)
    }
}

// ---------------------------------------------------------------------------
// Jacobian
// ---------------------------------------------------------------------------

/// Compute the Jacobian matrix `J[i][j] = ∂f_i/∂x_j`.
///
/// `f_vec` is a slice of m scalar expressions; `x_vec` is a slice of n
/// variable expressions.  The result is an m×n `Matrix`.
pub fn jacobian(
    f_vec: &[ExprId],
    x_vec: &[ExprId],
    pool: &ExprPool,
) -> Result<Matrix, crate::diff::diff_impl::DiffError> {
    let m = f_vec.len();
    let n = x_vec.len();
    let mut data = Vec::with_capacity(m * n);
    for &f in f_vec {
        for &x in x_vec {
            let df = diff(f, x, pool)?.value;
            data.push(df);
        }
    }
    Ok(Matrix {
        data,
        rows: m,
        cols: n,
    })
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl Matrix {
    pub fn display(&self, pool: &ExprPool) -> String {
        let rows: Vec<String> = (0..self.rows)
            .map(|r| {
                let entries: Vec<String> = self
                    .row(r)
                    .into_iter()
                    .map(|e| pool.display(e).to_string())
                    .collect();
                format!("[{}]", entries.join(", "))
            })
            .collect();
        format!("[{}]", rows.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn identity_2x2() {
        let pool = p();
        let id = Matrix::identity(2, &pool);
        assert_eq!(id.rows, 2);
        assert_eq!(id.cols, 2);
        assert_eq!(id.get(0, 0), pool.integer(1_i32));
        assert_eq!(id.get(0, 1), pool.integer(0_i32));
        assert_eq!(id.get(1, 0), pool.integer(0_i32));
        assert_eq!(id.get(1, 1), pool.integer(1_i32));
    }

    #[test]
    fn transpose_2x3() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let a = pool.integer(1_i32);
        let b = pool.integer(2_i32);
        let c = pool.integer(3_i32);
        // [[x, y, z], [a, b, c]]  →  [[x,a],[y,b],[z,c]]
        let m = Matrix::new(vec![vec![x, y, z], vec![a, b, c]]).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), x);
        assert_eq!(t.get(1, 1), b);
    }

    #[test]
    fn add_matrices() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let m1 = Matrix::new(vec![vec![x, one]]).unwrap();
        let m2 = Matrix::new(vec![vec![one, x]]).unwrap();
        let result = m1.add(&m2, &pool).unwrap();
        // result[0][0] = x + 1
        let r00_str = pool.display(result.get(0, 0)).to_string();
        assert!(
            r00_str.contains("x") && r00_str.contains("1"),
            "got: {r00_str}"
        );
    }

    #[test]
    fn mul_2x2() {
        let pool = p();
        // [[1,0],[0,1]] * [[a,b],[c,d]] = [[a,b],[c,d]]
        let id = Matrix::identity(2, &pool);
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let m = Matrix::new(vec![vec![x, y], vec![y, x]]).unwrap();
        let result = id.mul(&m, &pool).unwrap().simplify_entries(&pool);
        assert_eq!(result.get(0, 0), x);
        assert_eq!(result.get(0, 1), y);
    }

    #[test]
    fn det_2x2() {
        let pool = p();
        // det([[a,b],[c,d]]) = ad - bc
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let c = pool.symbol("c", Domain::Real);
        let d = pool.symbol("d", Domain::Real);
        let m = Matrix::new(vec![vec![a, b], vec![c, d]]).unwrap();
        let det = m.det(&pool).unwrap();
        let s = pool.display(det).to_string();
        assert!(s.contains("a") && s.contains("d"), "got: {s}");
    }

    #[test]
    fn det_3x3_identity_is_one() {
        let pool = p();
        let id = Matrix::identity(3, &pool);
        let det = id.det(&pool).unwrap();
        assert_eq!(det, pool.integer(1_i32));
    }

    #[test]
    fn jacobian_linear() {
        // f = [x + y, x - y], vars = [x, y]
        // J = [[1, 1], [1, -1]]
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_y = pool.mul(vec![pool.integer(-1_i32), y]);
        let f1 = pool.add(vec![x, y]);
        let f2 = pool.add(vec![x, neg_y]);
        let j = jacobian(&[f1, f2], &[x, y], &pool).unwrap();
        assert_eq!(j.rows, 2);
        assert_eq!(j.cols, 2);
        assert_eq!(j.get(0, 0), pool.integer(1_i32)); // ∂f1/∂x
        assert_eq!(j.get(0, 1), pool.integer(1_i32)); // ∂f1/∂y
        assert_eq!(j.get(1, 0), pool.integer(1_i32)); // ∂f2/∂x
        assert_eq!(j.get(1, 1), pool.integer(-1_i32)); // ∂f2/∂y
    }

    #[test]
    fn jacobian_quadratic() {
        // f = [x², y²], vars = [x, y]
        // J = [[2x, 0], [0, 2y]]
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let f1 = pool.pow(x, pool.integer(2_i32));
        let f2 = pool.pow(y, pool.integer(2_i32));
        let j = jacobian(&[f1, f2], &[x, y], &pool).unwrap();
        // ∂f1/∂y = 0, ∂f2/∂x = 0
        assert_eq!(j.get(0, 1), pool.integer(0_i32));
        assert_eq!(j.get(1, 0), pool.integer(0_i32));
    }
}
