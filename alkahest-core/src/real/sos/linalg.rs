//! Exact rational linear algebra for the Gram-matrix search.
//!
//! Two operations are needed, and both must be exact, because their output is
//! what a certificate is built from:
//!
//! * [`solve_affine`] — the complete solution set of a rational linear system,
//!   as a particular solution plus a null-space basis.  The Gram matrices of a
//!   polynomial form exactly such an affine set (one equation per monomial),
//!   and pinning down that set is what turns "search for a matrix" into
//!   "search over a handful of free parameters".
//! * [`psd_decompose`] — an exact `LDLᵀ` with symmetric pivoting, which both
//!   *decides* positive semidefiniteness over ℚ and, when the answer is yes,
//!   hands back the decomposition `Q = Σ d_k v_k v_kᵀ` with `d_k > 0`.  Each
//!   `(d_k, v_k)` is one weighted square of the certificate, so this is the
//!   step that converts a matrix into an algebraic identity.
//!
//! Neither routine rounds, and neither is a heuristic: `psd_decompose`
//! returning `None` is a proof that the matrix is *not* PSD, and returning
//! `Some` is a proof that it is.

use rug::Rational;

fn zero() -> Rational {
    Rational::from(0)
}

/// The full solution set of a consistent rational linear system:
/// `{ particular + Σ t_k · nullspace[k] : t ∈ ℚ^k }`.
#[derive(Debug, Clone)]
pub struct AffineSolution {
    /// One solution of the system.
    pub particular: Vec<Rational>,
    /// Basis of the homogeneous solution space; empty iff the solution is unique.
    pub nullspace: Vec<Vec<Rational>>,
}

impl AffineSolution {
    /// Number of free parameters (`0` when the solution is unique).
    pub fn dimension(&self) -> usize {
        self.nullspace.len()
    }

    /// The solution at parameter vector `t` (missing entries are taken as 0).
    pub fn at(&self, t: &[Rational]) -> Vec<Rational> {
        let mut x = self.particular.clone();
        for (k, dir) in self.nullspace.iter().enumerate() {
            let Some(tk) = t.get(k) else { break };
            if *tk == 0 {
                continue;
            }
            for (xi, di) in x.iter_mut().zip(dir) {
                *xi += Rational::from(tk * di);
            }
        }
        x
    }
}

/// Solve `rows · x = rhs` exactly by Gauss–Jordan elimination.
///
/// Returns `None` iff the system is inconsistent.  `rows` may be rank
/// deficient and may have more rows than columns; redundant rows are simply
/// eliminated away.
pub fn solve_affine(rows: &[Vec<Rational>], rhs: &[Rational]) -> Option<AffineSolution> {
    let m = rows.len();
    let ncols = rows.first().map_or(0, |r| r.len());
    if m == 0 {
        // No constraints at all: every coordinate is free.
        let nullspace = (0..ncols)
            .map(|k| {
                let mut v = vec![zero(); ncols];
                v[k] = Rational::from(1);
                v
            })
            .collect();
        return Some(AffineSolution {
            particular: vec![zero(); ncols],
            nullspace,
        });
    }

    // Augmented matrix, reduced in place.
    let mut a: Vec<Vec<Rational>> = Vec::with_capacity(m);
    for (row, b) in rows.iter().zip(rhs) {
        debug_assert_eq!(row.len(), ncols);
        let mut r = row.clone();
        r.push(b.clone());
        a.push(r);
    }

    let mut pivots: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for c in 0..ncols {
        let Some(p) = (r..m).find(|&i| a[i][c] != 0) else {
            continue;
        };
        a.swap(r, p);
        let inv = a[r][c].clone();
        for v in a[r].iter_mut() {
            *v /= &inv;
        }
        let prow = a[r].clone();
        for (i, row) in a.iter_mut().enumerate() {
            if i == r || row[c] == 0 {
                continue;
            }
            let f = row[c].clone();
            for (t, pv) in row.iter_mut().zip(prow.iter()) {
                *t -= Rational::from(&f * pv);
            }
        }
        pivots.push(c);
        r += 1;
        if r == m {
            break;
        }
    }

    // A row `0 = nonzero` means the system has no solution at all.
    for row in a.iter().skip(r) {
        if row[ncols] != 0 && row[..ncols].iter().all(|v| *v == 0) {
            return None;
        }
    }

    let mut particular = vec![zero(); ncols];
    for (i, &c) in pivots.iter().enumerate() {
        particular[c] = a[i][ncols].clone();
    }

    let mut nullspace = Vec::new();
    for free in (0..ncols).filter(|c| !pivots.contains(c)) {
        let mut v = vec![zero(); ncols];
        v[free] = Rational::from(1);
        for (i, &c) in pivots.iter().enumerate() {
            v[c] = -a[i][free].clone();
        }
        nullspace.push(v);
    }

    Some(AffineSolution {
        particular,
        nullspace,
    })
}

/// Exact `LDLᵀ` with symmetric (largest-diagonal) pivoting.
///
/// Returns `Some(vec![(d_k, v_k)])` with every `d_k > 0` and
/// `Q = Σ_k d_k · v_k v_kᵀ` *exactly*, or `None` when `Q` is not positive
/// semidefinite.  `Q` must be square and symmetric.
///
/// Symmetric pivoting is what makes this work on the boundary of the PSD cone:
/// the Gram matrices that certify Motzkin-like polynomials are singular by
/// necessity (the polynomial has real zeros), so a plain Cholesky that insists
/// on a non-zero leading pivot would fail on precisely the interesting cases.
pub fn psd_decompose(q: &[Vec<Rational>]) -> Option<Vec<(Rational, Vec<Rational>)>> {
    let n = q.len();
    let mut a: Vec<Vec<Rational>> = q.to_vec();
    let mut done = vec![false; n];
    let mut out: Vec<(Rational, Vec<Rational>)> = Vec::new();

    for _ in 0..n {
        let mut pivot: Option<usize> = None;
        for i in 0..n {
            if done[i] {
                continue;
            }
            // A negative diagonal entry is an immediate refutation: e_iᵀ Q e_i < 0.
            if a[i][i] < 0 {
                return None;
            }
            if a[i][i] > 0 && pivot.map_or(true, |p| a[i][i] > a[p][p]) {
                pivot = Some(i);
            }
        }
        let Some(p) = pivot else {
            // Every remaining diagonal entry is zero; for a PSD matrix that
            // forces the whole remaining block to vanish.
            for i in (0..n).filter(|&i| !done[i]) {
                for j in (0..n).filter(|&j| !done[j]) {
                    if a[i][j] != 0 {
                        return None;
                    }
                }
            }
            break;
        };

        let d = a[p][p].clone();
        let mut v = vec![zero(); n];
        for j in (0..n).filter(|&j| !done[j]) {
            v[j] = Rational::from(&a[p][j] / &d);
        }
        for i in (0..n).filter(|&i| !done[i]) {
            if v[i] == 0 {
                continue;
            }
            let scaled = Rational::from(&d * &v[i]);
            for j in (0..n).filter(|&j| !done[j]) {
                if v[j] == 0 {
                    continue;
                }
                a[i][j] -= Rational::from(&scaled * &v[j]);
            }
        }
        done[p] = true;
        out.push((d, v));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> Rational {
        Rational::from(n)
    }

    fn mat(rows: &[&[i64]]) -> Vec<Vec<Rational>> {
        rows.iter()
            .map(|r| r.iter().map(|&v| Rational::from(v)).collect())
            .collect()
    }

    #[test]
    fn unique_solution_has_no_free_parameters() {
        // x + y = 3, x − y = 1  ⇒  (2, 1)
        let rows = mat(&[&[1, 1], &[1, -1]]);
        let sol = solve_affine(&rows, &[r(3), r(1)]).expect("consistent");
        assert_eq!(sol.dimension(), 0);
        assert_eq!(sol.particular, vec![r(2), r(1)]);
    }

    #[test]
    fn underdetermined_system_reports_its_freedom() {
        // x + y + z = 1
        let rows = mat(&[&[1, 1, 1]]);
        let sol = solve_affine(&rows, &[r(1)]).expect("consistent");
        assert_eq!(sol.dimension(), 2);
        // every point of the reported set really is a solution
        let p = sol.at(&[r(5), r(-2)]);
        let s: Rational = p.iter().fold(r(0), |a, v| a + v.clone());
        assert_eq!(s, 1);
    }

    #[test]
    fn inconsistent_system_is_rejected() {
        let rows = mat(&[&[1, 1], &[2, 2]]);
        assert!(solve_affine(&rows, &[r(1), r(3)]).is_none());
    }

    #[test]
    fn redundant_rows_do_not_break_the_solver() {
        let rows = mat(&[&[1, 1], &[2, 2], &[1, -1]]);
        let sol = solve_affine(&rows, &[r(2), r(4), r(0)]).expect("consistent");
        assert_eq!(sol.dimension(), 0);
        assert_eq!(sol.particular, vec![r(1), r(1)]);
    }

    #[test]
    fn psd_decompose_factors_an_identity_matrix() {
        let q = mat(&[&[1, 0], &[0, 1]]);
        let d = psd_decompose(&q).expect("PSD");
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn psd_decompose_rejects_an_indefinite_matrix() {
        // [[1, 2], [2, 1]] has eigenvalues 3 and −1.
        let q = mat(&[&[1, 2], &[2, 1]]);
        assert!(psd_decompose(&q).is_none());
    }

    #[test]
    fn psd_decompose_accepts_a_singular_psd_matrix() {
        // [[1, 1], [1, 1]] = (e₁ + e₂)(e₁ + e₂)ᵀ, PSD of rank 1.
        let q = mat(&[&[1, 1], &[1, 1]]);
        let d = psd_decompose(&q).expect("PSD");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].0, 1);
    }

    #[test]
    fn psd_decompose_rejects_a_zero_diagonal_with_live_off_diagonal() {
        // [[0, 1], [1, 0]] is indefinite even though its diagonal is fine.
        let q = mat(&[&[0, 1], &[1, 0]]);
        assert!(psd_decompose(&q).is_none());
    }

    #[test]
    fn psd_decomposition_reproduces_the_matrix() {
        let q = mat(&[&[4, 2, 0], &[2, 5, 1], &[0, 1, 3]]);
        let d = psd_decompose(&q).expect("PSD");
        let n = 3;
        let mut acc = vec![vec![r(0); n]; n];
        for (dk, v) in &d {
            for i in 0..n {
                for j in 0..n {
                    acc[i][j] += Rational::from(dk * &v[i]) * &v[j];
                }
            }
        }
        assert_eq!(acc, q);
    }
}
