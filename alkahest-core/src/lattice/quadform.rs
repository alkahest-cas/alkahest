//! Exact rational linear algebra for positive-definite quadratic forms.
//!
//! Everything a lattice needs that is *not* enumeration: validating a Gram
//! matrix, Gram–Schmidt from a Gram matrix (no coordinates required),
//! determinant, inverse, and an LLL reduction driven by the Gram matrix alone.
//!
//! All of it is exact [`rug::Rational`] arithmetic. These matrices are `rank ×
//! rank` and the rank is capped at [`super::MAX_ENUM_RANK`], so cubic exact
//! algorithms are the right trade: a float Cholesky would put a rounding error
//! inside the *pruning bound* of the enumeration, where it silently drops
//! lattice vectors instead of reporting a problem.

use super::error::LatticeGeometryError;
use super::lll::LatticeError;
use rug::ops::DivRounding;
use rug::{Integer, Rational};

/// An LLL-reduced Gram matrix `G'` together with the unimodular `U` for which
/// `G' = U G Uᵀ`.
pub(crate) type ReducedGram = (Vec<Vec<Rational>>, Vec<Vec<Integer>>);

/// A square, symmetric rational matrix. Returns the rank (side length).
pub(crate) fn validate_gram(g: &[Vec<Rational>]) -> Result<usize, LatticeGeometryError> {
    if g.is_empty() {
        return Err(LatticeError::EmptyBasis.into());
    }
    let m = g.len();
    for (i, row) in g.iter().enumerate() {
        if row.len() != m {
            return Err(LatticeGeometryError::NonSquareGram {
                rows: m,
                cols: row.len(),
            });
        }
        // Only `i < j` needs checking, but reporting the first offending cell in
        // row-major order makes the message reproducible.
        for (j, _) in row.iter().enumerate().take(i) {
            if g[i][j] != g[j][i] {
                return Err(LatticeGeometryError::AsymmetricGram { row: i, col: j });
            }
        }
    }
    Ok(m)
}

/// Gram–Schmidt data derived from a Gram matrix alone.
///
/// `mu[i][j]` (for `j < i`) is `⟨b_i, b*_j⟩ / ‖b*_j‖²` and `b[i]` is `‖b*_i‖²`.
/// Fails with [`LatticeGeometryError::NotPositiveDefinite`] as soon as a
/// `‖b*_i‖²` is non-positive, which is exactly the statement that the
/// `i`-th leading principal minor is not positive.
pub(crate) fn gram_schmidt(
    g: &[Vec<Rational>],
) -> Result<(Vec<Vec<Rational>>, Vec<Rational>), LatticeGeometryError> {
    let m = g.len();
    let mut mu = vec![vec![Rational::new(); m]; m];
    let mut b = vec![Rational::new(); m];
    for i in 0..m {
        for j in 0..i {
            let mut acc = g[i][j].clone();
            for t in 0..j {
                acc -= Rational::from(&mu[i][t] * &mu[j][t]) * &b[t];
            }
            mu[i][j] = acc / b[j].clone();
        }
        let mut acc = g[i][i].clone();
        for t in 0..i {
            acc -= Rational::from(&mu[i][t] * &mu[i][t]) * &b[t];
        }
        if acc <= 0 {
            return Err(LatticeGeometryError::NotPositiveDefinite { pivot: i + 1 });
        }
        b[i] = acc;
    }
    Ok((mu, b))
}

/// `det G` for a symmetric positive-definite `G`, as `∏ ‖b*_i‖²`.
pub(crate) fn determinant(g: &[Vec<Rational>]) -> Result<Rational, LatticeGeometryError> {
    let (_, b) = gram_schmidt(g)?;
    Ok(b.into_iter().fold(Rational::from(1), |a, x| a * x))
}

/// Exact inverse by Gauss–Jordan. `g` must already be validated and invertible.
pub(crate) fn inverse(g: &[Vec<Rational>]) -> Result<Vec<Vec<Rational>>, LatticeGeometryError> {
    let m = g.len();
    let mut a: Vec<Vec<Rational>> = g.to_vec();
    let mut inv: Vec<Vec<Rational>> = (0..m)
        .map(|i| (0..m).map(|j| Rational::from(i32::from(i == j))).collect())
        .collect();
    for col in 0..m {
        let Some(p) = (col..m).find(|&r| a[r][col] != 0) else {
            return Err(LatticeGeometryError::NotPositiveDefinite { pivot: col + 1 });
        };
        a.swap(col, p);
        inv.swap(col, p);
        let pivot = a[col][col].clone();
        for t in 0..m {
            a[col][t] /= pivot.clone();
            inv[col][t] /= pivot.clone();
        }
        for r in 0..m {
            if r == col {
                continue;
            }
            let f = a[r][col].clone();
            if f == 0 {
                continue;
            }
            for t in 0..m {
                let da = Rational::from(&f * &a[col][t]);
                a[r][t] -= da;
                let di = Rational::from(&f * &inv[col][t]);
                inv[r][t] -= di;
            }
        }
    }
    Ok(inv)
}

/// `rows · rowsᵀ` — the Gram matrix of a rational row basis.
pub(crate) fn gram_of_rows(rows: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
    let m = rows.len();
    let mut g = vec![vec![Rational::new(); m]; m];
    for i in 0..m {
        for j in 0..=i {
            let mut acc = Rational::new();
            for (x, y) in rows[i].iter().zip(rows[j].iter()) {
                acc += Rational::from(x * y);
            }
            g[j][i] = acc.clone();
            g[i][j] = acc;
        }
    }
    g
}

/// Nearest integer to an exact rational, ties toward `+∞`.
///
/// Only ever called where `|x| > 1/2` strictly, so the tie-breaking rule is
/// unobservable; it is pinned by a test anyway so a future caller cannot be
/// surprised by it.
pub(crate) fn round_rational(x: &Rational) -> Integer {
    // round(n/d) = floor((2n + d) / (2d)) for d > 0, which `rug` guarantees.
    let num = Integer::from(x.numer() * 2i32) + x.denom();
    let den = Integer::from(x.denom() * 2i32);
    num.div_floor(den)
}

/// `|x| > 1/2`, exactly.
fn exceeds_half(x: &Rational) -> bool {
    Rational::from(x * Integer::from(2)).abs() > 1
}

/// `b_k ← b_k − q·b_j`, applied to the Gram matrix and to the accumulated
/// transform in step.
fn row_op(g: &mut [Vec<Rational>], u: &mut [Vec<Integer>], k: usize, j: usize, q: &Integer) {
    let qr = Rational::from(q.clone());
    // ⟨b_k − q b_j, b_k − q b_j⟩, computed before the row is disturbed.
    let new_kk = g[k][k].clone() - Rational::from(&qr * &g[k][j]) * Integer::from(2)
        + Rational::from(&qr * &qr) * &g[j][j];
    let row_j = g[j].clone();
    for (t, (dst, src)) in g[k].iter_mut().zip(row_j.iter()).enumerate() {
        if t == k {
            continue;
        }
        *dst -= Rational::from(&qr * src);
    }
    let row_k = g[k].clone();
    // Keep the matrix symmetric: column `k` mirrors the row just written.
    for (t, row) in g.iter_mut().enumerate() {
        if t == k {
            continue;
        }
        row[k] = row_k[t].clone();
    }
    g[k][k] = new_kk;
    let row_ju = u[j].clone();
    for (dst, src) in u[k].iter_mut().zip(row_ju.iter()) {
        *dst -= Integer::from(q * src);
    }
}

fn swap_rows(g: &mut [Vec<Rational>], u: &mut [Vec<Integer>], a: usize, b: usize) {
    let m = g.len();
    g.swap(a, b);
    for row in g.iter_mut().take(m) {
        row.swap(a, b);
    }
    u.swap(a, b);
}

/// LLL, driven by the Gram matrix rather than by coordinates.
///
/// Returns `(G', U)` with `G' = U G Uᵀ` and `U` unimodular. This is what lets a
/// lattice given only by its Gram matrix — `E_8`, the Leech lattice — be
/// enumerated at all: Fincke–Pohst on an unreduced form does not terminate in
/// any useful time.
///
/// `|det U| = 1` is not merely asserted, it is structural: `U` starts as the
/// identity and only ever receives integer row operations and row swaps.
pub(crate) fn gram_lll(
    g0: &[Vec<Rational>],
    delta: &Rational,
) -> Result<ReducedGram, LatticeGeometryError> {
    let m = validate_gram(g0)?;
    let mut g = g0.to_vec();
    let mut u: Vec<Vec<Integer>> = (0..m)
        .map(|i| (0..m).map(|j| Integer::from(i32::from(i == j))).collect())
        .collect();
    // Fails here rather than deep inside the loop if the form is degenerate.
    let _ = gram_schmidt(&g)?;
    if m == 1 {
        return Ok((g, u));
    }

    let mut k = 1usize;
    let mut guard: u64 = 0;
    // Each swap strictly decreases the integer-valued LLL potential
    // `∏_i D_i` by a factor of at least `1/δ`, so the loop terminates; the
    // budget only has to be unreachable for the ranks we allow.
    let max_steps: u64 = 4_000_000;
    while k < m {
        guard += 1;
        if guard > max_steps {
            return Err(LatticeError::IterationLimit {
                iterations: guard as usize,
            }
            .into());
        }
        let (mut mu, b) = gram_schmidt(&g)?;
        for j in (0..k).rev() {
            if !exceeds_half(&mu[k][j]) {
                continue;
            }
            let q = round_rational(&mu[k][j]);
            if q == 0 {
                continue;
            }
            row_op(&mut g, &mut u, k, j, &q);
            let qr = Rational::from(q);
            // `j < k`, so the two rows of `mu` live on opposite sides of the
            // split and can be borrowed at once.
            let (head, tail) = mu.split_at_mut(k);
            for (dst, src) in tail[0].iter_mut().zip(head[j].iter()).take(j) {
                *dst -= Rational::from(&qr * src);
            }
            mu[k][j] -= qr;
        }
        let lhs = b[k].clone();
        let rhs = (delta.clone() - Rational::from(&mu[k][k - 1] * &mu[k][k - 1])) * &b[k - 1];
        if lhs >= rhs {
            k += 1;
        } else {
            swap_rows(&mut g, &mut u, k, k - 1);
            k = k.max(2) - 1;
        }
    }
    Ok((g, u))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat_mat(v: &[&[i64]]) -> Vec<Vec<Rational>> {
        v.iter()
            .map(|r| r.iter().map(|&x| Rational::from(x)).collect())
            .collect()
    }

    #[test]
    fn round_rational_matches_nearest_integer() {
        for (n, d, want) in [
            (1i64, 3i64, 0i64),
            (2, 3, 1),
            (-2, 3, -1),
            (1, 2, 1),
            (-1, 2, 0),
            (7, 2, 4),
            (5, 1, 5),
        ] {
            assert_eq!(
                round_rational(&Rational::from((n, d))),
                Integer::from(want),
                "round({n}/{d})"
            );
        }
    }

    #[test]
    fn determinant_of_identity_and_a2() {
        let id = rat_mat(&[&[1, 0], &[0, 1]]);
        assert_eq!(determinant(&id).unwrap(), Rational::from(1));
        let a2 = rat_mat(&[&[2, -1], &[-1, 2]]);
        assert_eq!(determinant(&a2).unwrap(), Rational::from(3));
    }

    #[test]
    fn inverse_round_trips() {
        let a2 = rat_mat(&[&[2, -1], &[-1, 2]]);
        let inv = inverse(&a2).unwrap();
        for (i, row) in inv.iter().enumerate() {
            for (j, _) in row.iter().enumerate() {
                let mut acc = Rational::new();
                for t in 0..2 {
                    acc += Rational::from(&a2[i][t] * &inv[t][j]);
                }
                assert_eq!(acc, Rational::from(i32::from(i == j)));
            }
        }
    }

    #[test]
    fn non_symmetric_and_non_square_are_refused() {
        let ns = rat_mat(&[&[1, 2], &[3, 1]]);
        assert!(matches!(
            validate_gram(&ns),
            Err(LatticeGeometryError::AsymmetricGram { .. })
        ));
        let nsq = rat_mat(&[&[1, 2, 3], &[3, 1, 0]]);
        assert!(matches!(
            validate_gram(&nsq),
            Err(LatticeGeometryError::NonSquareGram { .. })
        ));
        let indefinite = rat_mat(&[&[1, 2], &[2, 1]]);
        assert!(matches!(
            gram_schmidt(&indefinite),
            Err(LatticeGeometryError::NotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn gram_lll_preserves_the_determinant_and_reduces() {
        let bad = rat_mat(&[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1]]);
        let (g, u) = gram_lll(&bad, &Rational::from((3, 4))).unwrap();
        assert_eq!(determinant(&g).unwrap(), Rational::from(1));
        assert_eq!(u.len(), 3);

        // A deliberately skewed unimodular change of basis of Z^2.
        let skew = rat_mat(&[&[1 + 100 * 100, 100 * 101], &[100 * 101, 1 + 101 * 101]]);
        let det_before = determinant(&skew).unwrap();
        let (g2, _) = gram_lll(&skew, &Rational::from((3, 4))).unwrap();
        assert_eq!(determinant(&g2).unwrap(), det_before);
        assert!(g2[0][0] <= skew[0][0]);
    }
}
