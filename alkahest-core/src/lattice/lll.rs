//! Lenstra–Lenstra–Lovász lattice basis reduction over ℤ (row basis vectors).
//!
//! Algorithm structure follows Henri Cohen (*A Course in Computational Algebraic Number
//! Theory*, §2.6): Gram–Schmidt orthogonalisation with exact [`rug::Rational`] arithmetic,
//! iterative size reductions and pairwise swaps enforcing the Lovász condition.
//!
//! This is intended for modest dimensions (`n,m ≲ 300`) where squared norms stay
//! representable comfortably in exact rationals — the primary consumers (van-Hoeij knapsacks)
//! rarely exceed that.

use crate::errors::AlkahestError;
use rug::{Assign, Float, Integer, Rational};
use std::fmt;

/// Lattice-basis reductions errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeError {
    /// No basis vectors supplied.
    EmptyBasis,
    /// Row `row` differs in length from the first row (ambient mismatch).
    RaggedBasis {
        row: usize,
        expected_cols: usize,
        got_cols: usize,
    },
    /// `δ ∉ (¼, 1)` as required by the LLL hypotheses.
    InvalidDelta { provided: Rational },
    /// Swap loop exceeded the iteration budget — basis may be degenerate or the implementation buggy.
    IterationLimit { iterations: usize },
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeError::EmptyBasis => write!(f, "LLL expects at least one basis row"),
            LatticeError::RaggedBasis {
                row,
                expected_cols,
                got_cols,
            } => write!(
                f,
                "row {row} has length {got_cols}; expected ambient dimension {expected_cols}"
            ),
            LatticeError::InvalidDelta { .. } => {
                write!(f, "LLL Lovász factor δ must lie strictly between ¼ and 1")
            }
            LatticeError::IterationLimit { iterations } => write!(
                f,
                "LLL reduction aborted after {iterations} swaps (degenerate span or oversized basis)"
            ),
        }
    }
}

impl std::error::Error for LatticeError {}

impl AlkahestError for LatticeError {
    fn code(&self) -> &'static str {
        match self {
            LatticeError::EmptyBasis => "E-LAT-001",
            LatticeError::RaggedBasis { .. } => "E-LAT-002",
            LatticeError::InvalidDelta { .. } => "E-LAT-003",
            LatticeError::IterationLimit { .. } => "E-LAT-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            LatticeError::EmptyBasis => {
                Some("pass a non-empty list of equally long integer coefficient rows")
            }
            LatticeError::RaggedBasis { .. } => {
                Some("pad or trim rows so every basis vector lies in ℤ^m for fixed m")
            }
            LatticeError::InvalidDelta { .. } => {
                Some("use the default δ = ¾, or choose another rational strictly between ¼ and 1")
            }
            LatticeError::IterationLimit { .. } => Some(
                "check for rank-deficient rows, reduce dimension, or report a bug with a minimal basis",
            ),
        }
    }
}

#[inline]
fn dot_int_rat(row: &[Integer], v: &[Rational]) -> Rational {
    let mut acc = Rational::from(0u32);
    for (zi, vv) in row.iter().zip(v.iter()) {
        let mut term = Rational::from(0u32);
        let prod = Rational::from(zi) * vv;
        term.assign(&prod);
        acc += term;
    }
    acc
}

fn dot_rat(a: &[Rational], b: &[Rational]) -> Rational {
    let mut acc = Rational::from(0u32);
    for (x, y) in a.iter().zip(b.iter()) {
        let mut term = Rational::from(0u32);
        let prod = x.clone() * y.clone();
        term.assign(&prod);
        acc += term;
    }
    acc
}

fn int_row_as_rat(row: &[Integer]) -> Vec<Rational> {
    row.iter().map(Rational::from).collect()
}

/// Gram–Schmidt data for rows `basis[0 … n − 1]`.
///
/// * `star[i]` holds the orthogonal residual `b*_i`.
/// * `mu[i][j]` for `j < i` is ⟨b_i,b*_j⟩ / ⟨b*_j,b*_j⟩.
/// * `b_norm_sq[i]` is ⟨b*_i,b*_i⟩ ∈ ℚ.
fn gram_schmidt_rows(
    basis: &[Vec<Integer>],
) -> (Vec<Vec<Rational>>, Vec<Vec<Rational>>, Vec<Rational>) {
    let n = basis.len();
    let ambient = basis[0].len();
    let mut star = vec![vec![Rational::from(0); ambient]; n];
    let mut mu = vec![vec![Rational::from(0); n]; n];
    let mut b_norm_sq = vec![Rational::from(0); n];

    for i in 0..n {
        let mut vip = int_row_as_rat(&basis[i]);
        for j in 0..i {
            mu[i][j].assign(&dot_int_rat(&basis[i], &star[j]) / &b_norm_sq[j]);
            for t in 0..ambient {
                let m = mu[i][j].clone() * star[j][t].clone();
                let vpt = vip[t].clone();
                let sub = vpt - &m;
                vip[t].assign(sub);
            }
        }
        star[i] = vip;
        let ni = dot_rat(&star[i], &star[i]);
        b_norm_sq[i].assign(ni);
    }
    (mu, star, b_norm_sq)
}

fn nearest_integer_rational(x: &Rational) -> Integer {
    Float::with_val(4096u32, x)
        .round()
        .to_integer()
        .unwrap_or_else(|| Integer::from(0))
}

fn validate_rows(basis: &[Vec<Integer>]) -> Result<usize, LatticeError> {
    if basis.is_empty() {
        return Err(LatticeError::EmptyBasis);
    }
    let cols = basis[0].len();
    for (i, row) in basis.iter().enumerate() {
        if row.len() != cols {
            return Err(LatticeError::RaggedBasis {
                row: i,
                expected_cols: cols,
                got_cols: row.len(),
            });
        }
    }
    Ok(cols)
}

fn validate_delta(delta: &Rational) -> Result<(), LatticeError> {
    let low = Rational::from((1i32, 4i32));
    let hi = Rational::from(1u32);
    if *delta <= low || *delta >= hi {
        return Err(LatticeError::InvalidDelta {
            provided: delta.clone(),
        });
    }
    Ok(())
}

fn size_reduce_single(
    basis: &mut [Vec<Integer>],
    mu: &[Vec<Rational>],
    b_norm_sq: &[Rational],
    k: usize,
) -> bool {
    let mut altered = false;
    for j in (0..k).rev() {
        if b_norm_sq[j].is_zero() {
            continue;
        }
        let mij = &mu[k][j];
        let q = nearest_integer_rational(mij);
        if q == 0 {
            continue;
        }
        altered = true;
        for col in 0..basis[k].len() {
            let bjk = basis[j][col].clone();
            basis[k][col] -= &(q.clone() * bjk);
        }
        return altered;
    }
    altered
}

/// Lovász predicate at index `k` (1-based outer loop index = `k` here 0-indexed):
/// \(B*_k ≥ (δ − μ²_{k,k−1}) B*_{k−1}\).
fn lovasz_ok(b_norm_sq: &[Rational], mu: &[Vec<Rational>], delta: &Rational, k: usize) -> bool {
    if k == 0 {
        return true;
    }
    let bk = &b_norm_sq[k];
    let bkm1 = &b_norm_sq[k - 1];
    if bkm1.is_zero() {
        return false;
    }
    let mux = mu[k][k - 1].clone();
    let mux_sq = Rational::from(&mux * &mux);
    let mut slack = delta.clone();
    slack -= &mux_sq;
    let rhs: Rational = slack * bkm1;
    bk.clone() >= rhs
}

fn lll_reduce_once(
    basis_rows: &[Vec<Integer>],
    delta: &Rational,
) -> Result<Vec<Vec<Integer>>, LatticeError> {
    validate_rows(basis_rows)?;
    validate_delta(delta)?;
    let ambient = basis_rows[0].len();
    let n = basis_rows.len();
    let mut basis: Vec<Vec<Integer>> = basis_rows.to_vec();

    let mut k: usize = 1;
    let mut guard: usize = 0;
    const MAX_LLL_SWAPS: usize = 2_000_000;
    loop {
        if k >= n {
            break;
        }
        guard += 1;
        if guard > MAX_LLL_SWAPS {
            return Err(LatticeError::IterationLimit { iterations: guard });
        }
        // Size-reduce row k until stable (against each successive Gram–Schmidt refresh).
        loop {
            let (mu_ref, _, b_norm_sq) = gram_schmidt_rows(&basis);
            if !size_reduce_single(&mut basis, &mu_ref, &b_norm_sq, k) {
                break;
            }
            // Projection numbers changed materially — reorthogonalise implicitly next loop.
        }
        let (mu, _, b_norm_sq) = gram_schmidt_rows(&basis);
        if lovasz_ok(&b_norm_sq, &mu, delta, k) {
            k += 1;
        } else {
            basis.swap(k, k - 1);
            k = k.saturating_sub(1);
            if k < 1 {
                k = 1;
            }
        }
        // Guard against malformed rank-deficient setups that spin forever:
        let _ = ambient;
        if k >= n && n > 8000 {
            break;
        }
    }

    Ok(basis)
}

/// Run LLL on integer row vectors using the conventional Lovász parameter `δ = ¾`.
pub fn lattice_reduce_rows(basis_rows: &[Vec<Integer>]) -> Result<Vec<Vec<Integer>>, LatticeError> {
    let delta = Rational::from((3u32, 4u32));
    lll_reduce_once(basis_rows, &delta)
}

/// Same as [`lattice_reduce_rows`], with an explicit `δ ∈ (¼, 1)`.
pub fn lattice_reduce_rows_with_delta(
    basis_rows: &[Vec<Integer>],
    delta: Rational,
) -> Result<Vec<Vec<Integer>>, LatticeError> {
    lll_reduce_once(basis_rows, &delta)
}

/// Check that `(δ, Lovász residuals, coefficient bounds)` satisfy the textbook LLL
/// inequalities (useful as a regression oracle).
///
/// Uses the **current row order**.
pub fn validate_lll_rows(
    basis_rows: &[Vec<Integer>],
    delta: &Rational,
) -> Result<(), &'static str> {
    validate_rows(basis_rows).map_err(|_| "shape")?;
    validate_delta(delta).map_err(|_| "delta")?;
    let n = basis_rows.len();
    let (mu, _, b_sq) = gram_schmidt_rows(basis_rows);
    if n == 1 {
        return Ok(());
    }
    let half = Rational::from((1u32, 2u32));
    for i in 1..n {
        for mij in mu[i].iter().take(i) {
            let mut absmu = mij.clone();
            absmu.abs_mut();
            if absmu > half {
                return Err("size");
            }
        }
        if !lovasz_ok(&b_sq, &mu, delta, i) {
            return Err("lovasz");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;

    #[test]
    fn planar_two_vectors_lll() {
        let rows: Vec<Vec<Integer>> = vec![
            vec![Integer::from(2), Integer::from(15)],
            vec![Integer::from(1), Integer::from(21)],
        ];
        let reduced = lattice_reduce_rows(&rows).unwrap();
        let delta = Rational::from((3u32, 4u32));
        validate_lll_rows(&reduced, &delta).unwrap();
    }

    #[test]
    fn knapsack_row_weighted_near_origin() {
        let rows: Vec<Vec<Integer>> = vec![
            vec![Integer::from(1), Integer::from(0), Integer::from(5)],
            vec![Integer::from(0), Integer::from(1), Integer::from(6)],
            vec![Integer::from(0), Integer::from(0), Integer::from(33)],
        ];
        let reduced = lattice_reduce_rows(&rows).unwrap();
        validate_lll_rows(&reduced, &Rational::from((3u32, 4u32))).unwrap();
        fn max_row_norm_squared(basis: &[Vec<Integer>]) -> Integer {
            basis
                .iter()
                .map(|row| {
                    row.iter().fold(Integer::from(0), |a, zi| {
                        Integer::from(a.clone() + zi.clone() * zi.clone())
                    })
                })
                .max_by(|x, y| x.cmp(y))
                .unwrap()
        }
        assert!(
            max_row_norm_squared(&reduced) <= max_row_norm_squared(&rows),
            "maximum squared row norm should shrink on this scaffold"
        );
    }
}
