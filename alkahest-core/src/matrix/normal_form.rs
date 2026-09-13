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

    // =======================================================================
    // 3.10.1 audit — randomised property coverage for the normal forms.
    //
    // What was here before: `U·M·V = S` on 30 random 4×4 matrices of
    // *non-negative* entries, cross-checked against FLINT's own SNF diagonal;
    // one 2×2 `diag(x, x)` for the ℚ[x] pair. What was not checked anywhere:
    // that `U` and `V` are **unimodular** — without that, `S = U·M·V` is a
    // factorisation through singular matrices and says nothing — negative and
    // rectangular and rank-deficient integer input, and the ℚ[x] Smith form on
    // anything but a diagonal matrix, which is what `rational_canonical_form`
    // reads its invariant factors off.
    //
    // The oracle for the invariant factors is the **determinantal divisor**
    // characterisation: `d_k(M)` is the gcd of all `k×k` minors of `M`, and
    // `s_k = d_k / d_{k−1}`. It is a theorem about `M` alone, computed here
    // from the minors directly, so it is independent of the algorithm under
    // test in a way that comparing two implementations is not.
    // =======================================================================

    fn lcg(state: &mut u64) -> i64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state % 19) as i64 - 9
    }

    /// Determinant by cofactor expansion (small `n` only).
    fn int_det(rows: &[Vec<Integer>]) -> Integer {
        let n = rows.len();
        if n == 0 {
            return Integer::from(1);
        }
        if n == 1 {
            return rows[0][0].clone();
        }
        let mut acc = Integer::from(0);
        for j in 0..n {
            let minor: Vec<Vec<Integer>> = rows[1..]
                .iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .filter(|(c, _)| *c != j)
                        .map(|(_, v)| v.clone())
                        .collect()
                })
                .collect();
            let term = rows[0][j].clone() * int_det(&minor);
            if j % 2 == 0 {
                acc += term;
            } else {
                acc -= term;
            }
        }
        acc
    }

    /// `d_k`: the gcd of every `k×k` minor of `m`. `d_0 = 1` by convention.
    fn determinantal_divisor(m: &IntegerMatrix, k: usize) -> Integer {
        if k == 0 {
            return Integer::from(1);
        }
        let mut g = Integer::from(0);
        for rows in combinations(m.rows, k) {
            for cols in combinations(m.cols, k) {
                let sub: Vec<Vec<Integer>> = rows
                    .iter()
                    .map(|&r| cols.iter().map(|&c| m.get(r, c).clone()).collect())
                    .collect();
                g = g.gcd(&int_det(&sub));
            }
        }
        g
    }

    fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut cur = Vec::new();
        fn go(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if cur.len() == k {
                out.push(cur.clone());
                return;
            }
            for i in start..n {
                cur.push(i);
                go(i + 1, n, k, cur, out);
                cur.pop();
            }
        }
        go(0, n, k, &mut cur, &mut out);
        out
    }

    fn as_rows(m: &IntegerMatrix) -> Vec<Vec<Integer>> {
        (0..m.rows)
            .map(|i| (0..m.cols).map(|j| m.get(i, j).clone()).collect())
            .collect()
    }

    /// `U·M = H` with `U` unimodular and `H` in Hermite form, over 90 shapes
    /// including negative entries, rectangular and rank-deficient input.
    ///
    /// Unimodularity of `U` is the half that was never asserted. `U·M = H`
    /// alone is satisfied by `U = 0, H = 0`; it is `det U = ±1` that makes `H`
    /// a *normal form of `M`* rather than an arbitrary product, and that makes
    /// the transform invertible for the caller who wants to undo it.
    #[test]
    fn hermite_form_transform_is_unimodular_over_random_shapes() {
        let mut st = 0x243F_6A88_85A3_08D3u64;
        for case in 0..90u64 {
            let rows = 1 + (case % 4) as usize;
            let cols = 1 + ((case / 4) % 4) as usize;
            let mut grid: Vec<Vec<i64>> = (0..rows)
                .map(|_| (0..cols).map(|_| lcg(&mut st)).collect())
                .collect();
            // A third of the cases get a dependent row.
            if rows > 1 && case % 3 == 0 {
                grid[0] = grid[rows - 1].iter().map(|v| v * 2).collect();
            }
            let m = IntegerMatrix::from_nested(grid).unwrap();
            let (h, u) = hermite_form(&m);
            assert_eq!(u.rows, rows);
            assert_eq!(u.cols, rows);
            assert_eq!(u.mul(&m).unwrap(), h, "case {case}: U·M = H");
            let d = int_det(&as_rows(&u));
            assert!(
                d == 1 || d == -1,
                "case {case}: U must be unimodular, det U = {d}"
            );
            assert!(h.to_flint().is_in_hnf(), "case {case}: H must be in HNF");
        }
    }

    /// `S = U·M·V` with `U`, `V` unimodular, the divisibility chain holding,
    /// and the diagonal equal to the determinantal divisors of `M`.
    #[test]
    fn smith_form_diagonal_matches_the_determinantal_divisors() {
        let mut st = 0xB504_F333_F9DE_6484u64;
        for case in 0..70u64 {
            let rows = 1 + (case % 3) as usize;
            let cols = 1 + ((case / 3) % 3) as usize;
            let mut grid: Vec<Vec<i64>> = (0..rows)
                .map(|_| (0..cols).map(|_| lcg(&mut st)).collect())
                .collect();
            if rows > 1 && case % 4 == 0 {
                grid[0] = grid[rows - 1].iter().map(|v| v * 3).collect();
            }
            if case % 11 == 0 {
                grid = (0..rows).map(|_| vec![0; cols]).collect();
            }
            let m = IntegerMatrix::from_nested(grid).unwrap();
            let (s, u, v) = smith_form(&m).unwrap();

            assert_eq!(
                u.mul(&m).unwrap().mul(&v).unwrap(),
                s,
                "case {case}: S = U·M·V"
            );
            let du = int_det(&as_rows(&u));
            let dv = int_det(&as_rows(&v));
            assert!(du == 1 || du == -1, "case {case}: det U = {du}");
            assert!(dv == 1 || dv == -1, "case {case}: det V = {dv}");

            // Rectangular-diagonal.
            for i in 0..s.rows {
                for j in 0..s.cols {
                    if i != j {
                        assert_eq!(
                            *s.get(i, j),
                            Integer::from(0),
                            "case {case}: S must be diagonal"
                        );
                    }
                }
            }
            // Divisibility chain.
            let d = s.rows.min(s.cols);
            for i in 0..d.saturating_sub(1) {
                let a = s.get(i, i).clone();
                let b = s.get(i + 1, i + 1).clone();
                if a == 0 {
                    assert_eq!(b, 0, "case {case}: a zero invariant factor ends the chain");
                } else {
                    assert_eq!(
                        b.clone() % a.clone(),
                        0,
                        "case {case}: s_{i} must divide s_{}",
                        i + 1
                    );
                }
            }
            // The oracle: s_k = d_k / d_{k−1}.
            let mut prev = Integer::from(1);
            for k in 1..=d {
                let dk = determinantal_divisor(&m, k);
                let sk = s.get(k - 1, k - 1).clone().abs();
                if prev == 0 {
                    assert_eq!(
                        dk,
                        0,
                        "case {case}: d_{k} must vanish once d_{} does",
                        k - 1
                    );
                    assert_eq!(sk, 0, "case {case}: s_{k} must vanish too");
                    continue;
                }
                assert_eq!(
                    sk.clone() * prev.clone(),
                    dk,
                    "case {case}: s_{k}·d_{} must equal d_{k}",
                    k - 1
                );
                prev = dk;
            }
        }
    }

    // ------------------------------------------------------------- ℚ[x]

    fn rand_poly(st: &mut u64, max_deg: usize) -> RatUniPoly {
        let deg = (lcg(st).unsigned_abs() as usize) % (max_deg + 1);
        let coeffs: Vec<Rational> = (0..=deg).map(|_| Rational::from((lcg(st), 1))).collect();
        RatUniPoly { coeffs }.trim()
    }

    fn poly_det(m: &[Vec<RatUniPoly>]) -> RatUniPoly {
        let n = m.len();
        if n == 0 {
            return RatUniPoly::one();
        }
        if n == 1 {
            return m[0][0].clone();
        }
        let mut acc = RatUniPoly::zero();
        for j in 0..n {
            let minor: Vec<Vec<RatUniPoly>> = m[1..]
                .iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .filter(|(c, _)| *c != j)
                        .map(|(_, v)| v.clone())
                        .collect()
                })
                .collect();
            let term = (&m[0][j] * &poly_det(&minor)).trim();
            acc = if j % 2 == 0 {
                (&acc + &term).trim()
            } else {
                (&acc - &term).trim()
            };
        }
        acc
    }

    fn poly_rows(m: &PolyMatrixQ) -> Vec<Vec<RatUniPoly>> {
        (0..m.rows)
            .map(|i| (0..m.cols).map(|j| m.get(i, j).clone()).collect())
            .collect()
    }

    /// `S = U·M·V` over ℚ[x] with `U`, `V` unimodular (non-zero *constant*
    /// determinant, the units of ℚ[x]), a divisibility chain, and
    /// `Π sᵢ = c · det M`.
    ///
    /// This is the routine `rational_canonical_form` reads its invariant
    /// factors off, and its only test was `diag(x, x)`.
    #[test]
    fn smith_form_poly_is_a_unimodular_equivalence() {
        let mut st = 0x1357_9BDF_2468_ACE0u64;
        for case in 0..60u64 {
            let n = 1 + (case % 3) as usize;
            let mut grid: Vec<Vec<RatUniPoly>> = (0..n)
                .map(|_| (0..n).map(|_| rand_poly(&mut st, 2)).collect())
                .collect();
            if n > 1 && case % 5 == 0 {
                grid[0] = grid[n - 1].clone();
            }
            let m = PolyMatrixQ::from_nested(grid).unwrap();
            let (s, u, v) = smith_form_poly(&m);

            let prod = u.mul(&m).unwrap().mul(&v).unwrap();
            assert_eq!(prod, s, "case {case}: S = U·M·V over ℚ[x]");

            for (name, t) in [("U", &u), ("V", &v)] {
                let d = poly_det(&poly_rows(t));
                assert!(
                    !d.is_zero() && d.degree() == 0,
                    "case {case}: det {name} must be a non-zero constant (it is the \
                     unit group of ℚ[x]); got degree {}",
                    d.degree()
                );
            }

            for i in 0..s.rows {
                for j in 0..s.cols {
                    if i != j {
                        assert!(
                            s.get(i, j).clone().trim().is_zero(),
                            "case {case}: S must be diagonal"
                        );
                    }
                }
            }
            let d = s.rows.min(s.cols);
            for i in 0..d.saturating_sub(1) {
                let a = s.get(i, i).clone().trim();
                let b = s.get(i + 1, i + 1).clone().trim();
                if a.is_zero() {
                    assert!(b.is_zero(), "case {case}: chain must stay zero");
                } else {
                    let (_, r) = RatUniPoly::div_rem(&b, &a);
                    assert!(
                        r.trim().is_zero(),
                        "case {case}: s_{i} must divide s_{}",
                        i + 1
                    );
                }
            }

            // `Π sᵢ` and `det M` agree up to a non-zero constant: the product of
            // the invariant factors is the largest determinantal divisor, which
            // for a square matrix is `det M`.
            let mut prod_s = RatUniPoly::one();
            for i in 0..d {
                prod_s = (&prod_s * s.get(i, i)).trim();
            }
            let det_m = poly_det(&poly_rows(&m)).trim();
            if det_m.is_zero() {
                assert!(
                    prod_s.is_zero(),
                    "case {case}: a singular M must have a zero invariant factor"
                );
            } else {
                assert_eq!(
                    prod_s.degree(),
                    det_m.degree(),
                    "case {case}: deg Π sᵢ must equal deg det M"
                );
                let (q, r) = RatUniPoly::div_rem(&det_m, &prod_s);
                assert!(
                    r.trim().is_zero() && q.trim().degree() == 0,
                    "case {case}: det M / Π sᵢ must be a non-zero constant"
                );
            }
        }
    }

    /// `U·M = H` over ℚ[x] with `U` unimodular, on non-diagonal input.
    #[test]
    fn hermite_form_poly_transform_is_unimodular() {
        let mut st = 0x0F0F_0F0F_1234_5678u64;
        for case in 0..40u64 {
            let n = 1 + (case % 3) as usize;
            let grid: Vec<Vec<RatUniPoly>> = (0..n)
                .map(|_| (0..n).map(|_| rand_poly(&mut st, 2)).collect())
                .collect();
            let m = PolyMatrixQ::from_nested(grid).unwrap();
            let (h, u) = hermite_form_poly(&m);
            assert_eq!(u.mul(&m).unwrap(), h, "case {case}: U·M = H over ℚ[x]");
            let d = poly_det(&poly_rows(&u));
            assert!(
                !d.is_zero() && d.degree() == 0,
                "case {case}: det U must be a non-zero constant, got degree {}",
                d.degree()
            );
        }
    }
}
