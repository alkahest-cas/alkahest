//! V2-5 — Hermite and Smith normal forms for dense integer matrices (`IntegerMatrix`)
//! and polynomial matrices over ℚ (`PolyMatrixQ` / `RatUniPoly`).
//!
//! Integer Hermite form uses FLINT `fmpz_mat_hnf_transform` (Storjohann-class implementations
//! inside FLINT). Integer Smith form follows SymPy `smith_normal_decomp` (`U * M * V = S`).
//! Polynomial Hermite / Smith use the same column-elimination pattern over the Euclidean
//! domain `ℚ[x]`.

#![allow(
    clippy::needless_range_loop,
    clippy::cmp_owned,
    clippy::unnecessary_min_or_max
)]

use super::smith;
use super::smith_poly;

use crate::errors::AlkahestError;
use crate::flint::integer::FlintInteger;
use crate::flint::mat::FlintMat;
use rug::{Integer, Rational};
use std::fmt;
use std::ops::Mul;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from constructing or combining normal-form matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalFormError {
    /// A row in a nested initializer had the wrong length.
    DimensionMismatch {
        row: usize,
        expected_cols: usize,
        got: usize,
    },
    /// `A * B` was requested but `A.cols != B.rows`.
    IncompatibleMultiply { left_cols: usize, right_rows: usize },
}

impl fmt::Display for NormalFormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NormalFormError::DimensionMismatch {
                row,
                expected_cols,
                got,
            } => write!(f, "row {row} has {got} columns, expected {expected_cols}",),
            NormalFormError::IncompatibleMultiply {
                left_cols,
                right_rows,
            } => write!(
                f,
                "cannot multiply {left_cols}-wide matrix by matrix with {right_rows} rows",
            ),
        }
    }
}

impl std::error::Error for NormalFormError {}

impl AlkahestError for NormalFormError {
    fn code(&self) -> &'static str {
        match self {
            NormalFormError::DimensionMismatch { .. } => "E-NFM-001",
            NormalFormError::IncompatibleMultiply { .. } => "E-NFM-002",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            NormalFormError::DimensionMismatch { .. } => {
                Some("every row in `IntegerMatrix::from_nested` must have equal width")
            }
            NormalFormError::IncompatibleMultiply { .. } => {
                Some("for `A * B`, use matrices where `A.cols == B.rows`")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Integer matrices
// ---------------------------------------------------------------------------

/// Dense `m × n` matrix over ℤ (row-major).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegerMatrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<Integer>,
}

impl IntegerMatrix {
    /// Build from nested rows of `i64` (must be rectangular).
    pub fn from_nested(rows: Vec<Vec<i64>>) -> Result<Self, NormalFormError> {
        if rows.is_empty() {
            return Ok(Self {
                rows: 0,
                cols: 0,
                data: vec![],
            });
        }
        let cols = rows[0].len();
        let mut data = Vec::with_capacity(rows.len() * cols);
        for (ri, r) in rows.iter().enumerate() {
            if r.len() != cols {
                return Err(NormalFormError::DimensionMismatch {
                    row: ri,
                    expected_cols: cols,
                    got: r.len(),
                });
            }
            for &x in r {
                data.push(Integer::from(x));
            }
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data,
        })
    }

    fn from_rug_rows(rows: Vec<Vec<Integer>>) -> Result<Self, NormalFormError> {
        if rows.is_empty() {
            return Ok(Self {
                rows: 0,
                cols: 0,
                data: vec![],
            });
        }
        let cols = rows[0].len();
        let mut data = Vec::with_capacity(rows.len() * cols);
        for (ri, r) in rows.iter().enumerate() {
            if r.len() != cols {
                return Err(NormalFormError::DimensionMismatch {
                    row: ri,
                    expected_cols: cols,
                    got: r.len(),
                });
            }
            for x in r {
                data.push(x.clone());
            }
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data,
        })
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> &Integer {
        &self.data[r * self.cols + c]
    }

    /// Matrix product `self * other`.
    pub fn mul(&self, other: &IntegerMatrix) -> Result<Self, NormalFormError> {
        if self.cols != other.rows {
            return Err(NormalFormError::IncompatibleMultiply {
                left_cols: self.cols,
                right_rows: other.rows,
            });
        }
        let m = self.rows;
        let n = other.cols;
        let k = self.cols;
        let mut out = vec![Integer::from(0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = Integer::from(0);
                for t in 0..k {
                    acc += self.get(i, t) * other.get(t, j);
                }
                out[i * n + j] = acc;
            }
        }
        Ok(IntegerMatrix {
            rows: m,
            cols: n,
            data: out,
        })
    }

    fn to_flint(&self) -> FlintMat {
        let mut a = FlintMat::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let fi = FlintInteger::from_rug(self.get(i, j));
                a.set_entry(i, j, &fi);
            }
        }
        a
    }

    fn from_flint(m: &FlintMat) -> Self {
        let rows = m.rows();
        let cols = m.cols();
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(m.get_flint(i, j).to_rug());
            }
        }
        Self { rows, cols, data }
    }

    fn to_nested_integer(&self) -> Vec<Vec<Integer>> {
        (0..self.rows)
            .map(|i| (0..self.cols).map(|j| self.get(i, j).clone()).collect())
            .collect()
    }
}

/// Hermite normal form: returns `(H, U)` with `U * M = H`, where `U` is unimodular.
/// Uses FLINT `fmpz_mat_hnf_transform`.
pub fn hermite_form(m: &IntegerMatrix) -> (IntegerMatrix, IntegerMatrix) {
    if m.rows == 0 || m.cols == 0 {
        return (
            IntegerMatrix {
                rows: m.rows,
                cols: m.cols,
                data: vec![],
            },
            IntegerMatrix::identity(m.rows),
        );
    }
    let a = m.to_flint();
    let mut h = FlintMat::new(m.rows, m.cols);
    let mut u = FlintMat::new(m.rows, m.rows);
    a.hnf_transform(&mut h, &mut u);
    (IntegerMatrix::from_flint(&h), IntegerMatrix::from_flint(&u))
}

impl IntegerMatrix {
    fn identity(n: usize) -> Self {
        let mut data = vec![Integer::from(0); n * n];
        for i in 0..n {
            data[i * n + i] = Integer::from(1);
        }
        Self {
            rows: n,
            cols: n,
            data,
        }
    }
}

/// Smith normal form: `(S, U, V)` with `S == U * M * V`, `S` rectangular-diagonal, invariant
/// factors dividing along the diagonal.
pub fn smith_form(
    m: &IntegerMatrix,
) -> Result<(IntegerMatrix, IntegerMatrix, IntegerMatrix), NormalFormError> {
    if m.rows == 0 || m.cols == 0 {
        return Ok((
            IntegerMatrix {
                rows: m.rows,
                cols: m.cols,
                data: vec![],
            },
            IntegerMatrix::identity(m.rows),
            IntegerMatrix::identity(m.cols),
        ));
    }
    let (s, u, v) = smith::smith_normal_decomp(m.to_nested_integer());
    Ok((
        IntegerMatrix::from_rug_rows(s)?,
        IntegerMatrix::from_rug_rows(u)?,
        IntegerMatrix::from_rug_rows(v)?,
    ))
}

// ---------------------------------------------------------------------------
// ℚ[x] polynomials (dense, ascending degree)
// ---------------------------------------------------------------------------

/// Univariate polynomial over ℚ, `∑ cᵢ xⁱ`.
#[derive(Clone, Debug)]
pub struct RatUniPoly {
    /// Ascending coefficients; trailing zeros are stripped.
    pub coeffs: Vec<Rational>,
}

impl PartialEq for RatUniPoly {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl Eq for RatUniPoly {}

impl RatUniPoly {
    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    pub fn one() -> Self {
        Self {
            coeffs: vec![Rational::from(1)],
        }
    }

    pub fn constant(c: Rational) -> Self {
        if c == Rational::from(0) {
            Self::zero()
        } else {
            Self { coeffs: vec![c] }
        }
    }

    /// The polynomial `x`.
    pub fn x() -> Self {
        Self {
            coeffs: vec![Rational::from(0), Rational::from(1)],
        }
    }

    pub(crate) fn trim(mut self) -> Self {
        while self.coeffs.last() == Some(&Rational::from(0)) {
            self.coeffs.pop();
        }
        self
    }

    pub fn degree(&self) -> i32 {
        self.coeffs.len() as i32 - 1
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub(crate) fn leading_coeff(&self) -> Rational {
        self.coeffs
            .last()
            .cloned()
            .unwrap_or_else(|| Rational::from(0))
    }

    /// Euclidean division: `a = q * b + r`, `deg r < deg b` (or `r = 0`).
    pub fn div_rem(a: &Self, b: &Self) -> (Self, Self) {
        assert!(!b.is_zero());
        let mut a = a.clone();
        let mut a_c = std::mem::take(&mut a.coeffs);
        let b = b.clone().trim();
        let b_c = &b.coeffs;
        let db = b_c.len() as i32 - 1;
        let lb = b_c[b_c.len() - 1].clone();

        let mut q = vec![Rational::from(0); (a_c.len().saturating_sub(b_c.len()) + 1).max(0)];

        while a_c.len() as i32 > db && a_c.last().map(|v| v != &Rational::from(0)).unwrap_or(false)
        {
            let da = a_c.len() as i32 - 1;
            let la = a_c.last().unwrap().clone();
            let shift = (da - db) as usize;
            if shift >= q.len() {
                q.resize(shift + 1, Rational::from(0));
            }
            let t = la / &lb;
            q[shift] += &t;
            for j in 0..b_c.len() {
                let i = shift + j;
                let prod = t.clone() * b_c[j].clone();
                a_c[i] -= &prod;
            }
            while a_c.last() == Some(&Rational::from(0)) {
                a_c.pop();
            }
        }

        let q_poly = RatUniPoly { coeffs: q }.trim();
        let r_poly = RatUniPoly { coeffs: a_c }.trim();
        (q_poly, r_poly)
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();
        if a.degree() < b.degree() {
            std::mem::swap(&mut a, &mut b);
        }
        while !b.is_zero() {
            let (_, r) = RatUniPoly::div_rem(&a, &b);
            a = b;
            b = r;
        }
        if a.is_zero() {
            RatUniPoly::zero()
        } else {
            let mut g = a.trim();
            let lc = g.leading_coeff();
            for c in &mut g.coeffs {
                *c /= lc.clone();
            }
            g.trim()
        }
    }

    pub fn gcdex(a: &Self, b: &Self) -> (Self, Self, Self) {
        if b.is_zero() {
            if a.is_zero() {
                return (Self::zero(), Self::one(), Self::zero());
            }
            let mut an = a.clone().trim();
            let lc = an.leading_coeff();
            let inv = Rational::from(1) / lc.clone();
            for c in &mut an.coeffs {
                *c *= inv.clone();
            }
            let an = an.trim();
            return (Self::constant(inv), Self::zero(), an);
        }
        let (q, r) = Self::div_rem(a, b);
        let (s1, t1, g) = Self::gcdex(b, &r);
        let qt = &q * &t1;
        let tt = &s1 - &qt;
        (t1, tt.trim(), g)
    }

    pub(super) fn exquo(&self, g: &Self) -> Self {
        let (q, r) = RatUniPoly::div_rem(self, g);
        if !r.is_zero() {
            panic!("RatUniPoly::exquo: not divisible");
        }
        q
    }
}

impl std::ops::Add for &RatUniPoly {
    type Output = RatUniPoly;
    fn add(self, rhs: &RatUniPoly) -> RatUniPoly {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut c = vec![Rational::from(0); n];
        for i in 0..n {
            if i < self.coeffs.len() {
                c[i] += self.coeffs[i].clone();
            }
            if i < rhs.coeffs.len() {
                c[i] += rhs.coeffs[i].clone();
            }
        }
        RatUniPoly { coeffs: c }.trim()
    }
}

impl std::ops::Sub for &RatUniPoly {
    type Output = RatUniPoly;
    fn sub(self, rhs: &RatUniPoly) -> RatUniPoly {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut c = vec![Rational::from(0); n];
        for i in 0..n {
            if i < self.coeffs.len() {
                c[i] += self.coeffs[i].clone();
            }
            if i < rhs.coeffs.len() {
                c[i] -= rhs.coeffs[i].clone();
            }
        }
        RatUniPoly { coeffs: c }.trim()
    }
}

impl Mul for RatUniPoly {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        (&self).mul(&rhs)
    }
}

impl std::ops::Mul for &RatUniPoly {
    type Output = RatUniPoly;
    fn mul(self, rhs: &RatUniPoly) -> RatUniPoly {
        if self.is_zero() || rhs.is_zero() {
            return RatUniPoly::zero();
        }
        let mut c = vec![Rational::from(0); self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in rhs.coeffs.iter().enumerate() {
                c[i + j] += a.clone() * b;
            }
        }
        RatUniPoly { coeffs: c }.trim()
    }
}

impl std::ops::Neg for &RatUniPoly {
    type Output = RatUniPoly;
    fn neg(self) -> RatUniPoly {
        let coeffs = self.coeffs.iter().map(|c| -c.clone()).collect();
        RatUniPoly { coeffs }.trim()
    }
}

// ---------------------------------------------------------------------------
// Polynomial matrices over ℚ[x]
// ---------------------------------------------------------------------------

/// Rectangular matrix whose entries are univariate polynomials over ℚ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyMatrixQ {
    pub rows: usize,
    pub cols: usize,
    data: Vec<RatUniPoly>,
}

impl PolyMatrixQ {
    pub(super) fn shell(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![],
        }
    }

    pub fn from_nested(rows: Vec<Vec<RatUniPoly>>) -> Result<Self, NormalFormError> {
        if rows.is_empty() {
            return Ok(Self {
                rows: 0,
                cols: 0,
                data: vec![],
            });
        }
        let cols = rows[0].len();
        let mut data = Vec::with_capacity(rows.len() * cols);
        for (ri, r) in rows.iter().enumerate() {
            if r.len() != cols {
                return Err(NormalFormError::DimensionMismatch {
                    row: ri,
                    expected_cols: cols,
                    got: r.len(),
                });
            }
            for p in r {
                data.push(p.clone());
            }
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data,
        })
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> &RatUniPoly {
        &self.data[r * self.cols + c]
    }

    pub fn mul(&self, other: &PolyMatrixQ) -> Result<Self, NormalFormError> {
        if self.cols != other.rows {
            return Err(NormalFormError::IncompatibleMultiply {
                left_cols: self.cols,
                right_rows: other.rows,
            });
        }
        let m = self.rows;
        let n = other.cols;
        let k = self.cols;
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut acc = RatUniPoly::zero();
                for t in 0..k {
                    let prod = self.get(i, t).clone() * other.get(t, j).clone();
                    acc = (&acc + &prod).trim();
                }
                out.push(acc);
            }
        }
        Ok(PolyMatrixQ {
            rows: m,
            cols: n,
            data: out,
        })
    }

    fn transpose(&self) -> PolyMatrixQ {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for j in 0..self.cols {
            for i in 0..self.rows {
                data.push(self.get(i, j).clone());
            }
        }
        PolyMatrixQ {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }
}

/// Hermite column-form on `Mᵀ`, then transpose — yields `(H, U)` with `U * M = H`
/// for the row-style convention used by integer matrices.
pub fn hermite_form_poly(m: &PolyMatrixQ) -> (PolyMatrixQ, PolyMatrixQ) {
    let mt = m.transpose();
    let (ht, v) = smith_poly::hermite_column_poly(&mt);
    (ht.transpose(), v.transpose())
}

/// Smith normal form over `ℚ[x]`: `(S, U, V)` with `S == U * M * V`.
pub fn smith_form_poly(m: &PolyMatrixQ) -> (PolyMatrixQ, PolyMatrixQ, PolyMatrixQ) {
    smith_poly::smith_normal_poly(m)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Complete;

    #[test]
    fn hnf_transform_matches_flint_and_um_equals_h() {
        let m = IntegerMatrix::from_nested(vec![vec![12, 6, 4], vec![3, 9, 6], vec![2, 16, 14]])
            .unwrap();
        let (h, u) = hermite_form(&m);
        let um = u.mul(&m).unwrap();
        assert_eq!(um, h);
        let fh = h.to_flint();
        assert!(fh.is_in_hnf());
    }

    #[test]
    fn snf_sympy_example_3x3() {
        let m = IntegerMatrix::from_nested(vec![vec![12, 6, 4], vec![3, 9, 6], vec![2, 16, 14]])
            .unwrap();
        let (s, u, v) = smith_form(&m).unwrap();
        let umv = u.mul(&m).unwrap().mul(&v).unwrap();
        assert_eq!(umv, s);
        assert!(s.to_flint().is_in_snf());
        // invariant divisibility on diagonal
        let d = m.rows.min(m.cols);
        for i in 0..d.saturating_sub(1) {
            let a = s.get(i, i).clone();
            let b = s.get(i + 1, i + 1).clone();
            if a != Integer::from(0) && b != Integer::from(0) {
                let (_, r) = b.div_rem_floor_ref(&a).complete();
                assert_eq!(r, Integer::from(0));
            }
        }
    }

    #[test]
    fn snf_random_small_matches_flint_diagonal() {
        use rug::rand::RandState;
        let mut rand = RandState::new();
        for _ in 0..30 {
            let mut rows = vec![];
            for _ in 0..4 {
                let mut r = vec![];
                for _ in 0..4 {
                    let x: u32 = rand.bits(6);
                    r.push(x as i64);
                }
                rows.push(r);
            }
            let m = IntegerMatrix::from_nested(rows).unwrap();
            let (s, u, v) = smith_form(&m).unwrap();
            let umv = u.mul(&m).unwrap().mul(&v).unwrap();
            assert_eq!(umv, s);
            let fa = m.to_flint();
            let mut fs = FlintMat::new(m.rows, m.cols);
            fa.snf_diagonal(&mut fs);
            assert!(s.to_flint().equals(&fs));
        }
    }

    #[test]
    fn poly_hermite_and_smith_diag_x() {
        let x = RatUniPoly::x();
        let z = RatUniPoly::zero();
        let m =
            PolyMatrixQ::from_nested(vec![vec![x.clone(), z.clone()], vec![z.clone(), x.clone()]])
                .unwrap();
        let (h, u) = hermite_form_poly(&m);
        let um = u.mul(&m).unwrap();
        assert_eq!(um, h);

        let (s, us, vs) = smith_form_poly(&m);
        let prod = us.mul(&m).unwrap().mul(&vs).unwrap();
        assert_eq!(prod, s);
    }
}
