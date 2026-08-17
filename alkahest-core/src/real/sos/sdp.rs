//! Floating-point *proposal* of a positive semidefinite Gram matrix.
//!
//! # Why floating point is allowed here, and what keeps it sound
//!
//! The rest of this subsystem is exact on principle: a certificate that came
//! out of a floating-point computation is not a proof.  That principle is not
//! weakened by this module, because nothing here is ever believed.
//!
//! The Gram matrices of a polynomial form an affine family
//! `Q(t) = B₀ + Σ t_k·B_k` with **rational** `B_k` ([`super::linalg`] computes
//! it).  When that family has free parameters, some point of it may be
//! positive semidefinite and others not, and finding one is a semidefinite
//! feasibility problem that exact rational arithmetic cannot solve directly.
//! So this module runs a cheap numerical search — alternating projection onto
//! the PSD cone and back onto the affine family — and returns the parameter
//! vector `t` it landed on, as a **suggestion**.
//!
//! The caller then rounds `t` to rationals and rebuilds `Q(t)` in ℚ.  That
//! rebuilt matrix lies in the affine family *exactly*, by construction, whatever
//! the rounding did; and whether it is PSD is then decided *exactly*, by the
//! rational `LDLᵀ` in [`super::linalg::psd_decompose`].  A bad suggestion
//! therefore costs a failed search, never an unsound certificate.  No number
//! computed in this file appears in any certificate.

#![allow(clippy::needless_range_loop)]

/// Symmetric eigendecomposition by the cyclic Jacobi method.
///
/// Returns `(values, vectors)` where `vectors[i]` is the eigenvector for
/// `values[i]`.  Jacobi is used rather than anything faster because the
/// matrices here are small and it is unconditionally stable on symmetric
/// input — accuracy matters more than speed, since a poor eigenbasis just
/// wastes the exact check that follows.
fn jacobi_eigen(input: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = input.len();
    let mut a: Vec<Vec<f64>> = input.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for _ in 0..60 {
        // Off-diagonal mass; stop once it is negligible.
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off <= 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-18 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let values: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    let vectors: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|k| v[k][i]).collect()).collect();
    (values, vectors)
}

/// Projection onto `{Q : Q ⪰ floor·I}` in the eigenvalue sense.
fn project_psd(q: &[Vec<f64>], floor: f64) -> Vec<Vec<f64>> {
    let n = q.len();
    let (vals, vecs) = jacobi_eigen(q);
    let mut out = vec![vec![0.0f64; n]; n];
    for (lambda, vec) in vals.iter().zip(vecs.iter()) {
        let w = lambda.max(floor);
        if w == 0.0 {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                out[i][j] += w * vec[i] * vec[j];
            }
        }
    }
    out
}

/// Smallest eigenvalue of a symmetric matrix (numerically).
pub fn min_eigenvalue(q: &[Vec<f64>]) -> f64 {
    let (vals, _) = jacobi_eigen(q);
    vals.into_iter().fold(f64::INFINITY, f64::min)
}

fn dot(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x * y).sum::<f64>())
        .sum()
}

/// Solve a small symmetric positive definite system by Cholesky.
fn solve_spd(g: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = g.len();
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = g[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 1e-12 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

/// The affine family `Q(t) = base + Σ t_k · dirs[k]`, in floating point.
pub struct Family {
    base: Vec<Vec<f64>>,
    dirs: Vec<Vec<Vec<f64>>>,
    /// Cholesky-ready Gram matrix of `dirs`, for the projection step.
    gram: Vec<Vec<f64>>,
}

impl Family {
    /// `dirs` must be linearly independent (they come from a null-space basis).
    pub fn new(base: Vec<Vec<f64>>, dirs: Vec<Vec<Vec<f64>>>) -> Self {
        let k = dirs.len();
        let mut gram = vec![vec![0.0f64; k]; k];
        for i in 0..k {
            for j in 0..=i {
                let v = dot(&dirs[i], &dirs[j]);
                gram[i][j] = v;
                gram[j][i] = v;
            }
        }
        Family { base, dirs, gram }
    }

    /// `Q(t)`.
    pub fn at(&self, t: &[f64]) -> Vec<Vec<f64>> {
        let n = self.base.len();
        let mut q = self.base.clone();
        for (tk, dir) in t.iter().zip(self.dirs.iter()) {
            if *tk == 0.0 {
                continue;
            }
            for i in 0..n {
                for j in 0..n {
                    q[i][j] += tk * dir[i][j];
                }
            }
        }
        q
    }

    /// Least-squares projection of `q` back onto the family, as parameters.
    fn parameters_of(&self, q: &[Vec<f64>]) -> Option<Vec<f64>> {
        let n = self.base.len();
        let mut delta = q.to_vec();
        for i in 0..n {
            for j in 0..n {
                delta[i][j] -= self.base[i][j];
            }
        }
        let rhs: Vec<f64> = self.dirs.iter().map(|d| dot(d, &delta)).collect();
        solve_spd(&self.gram, &rhs)
    }

    /// Alternating projection between the PSD cone (with an eigenvalue floor,
    /// which pushes the iterate towards the *interior* of the feasible set and
    /// so makes the subsequent rational rounding survive) and this family,
    /// starting from the particular solution (`t = 0`).
    ///
    /// Returns the parameter vector reached, or `None` if the linear algebra
    /// degenerated.  Convergence is not required and not claimed: the result is
    /// a suggestion to be checked exactly.
    pub fn search(&self, floor: f64, iters: usize) -> Option<Vec<f64>> {
        self.search_from(vec![0.0f64; self.dirs.len()], floor, iters)
    }

    /// Same alternating projection as [`Self::search`], from an arbitrary
    /// starting parameter vector.
    ///
    /// Plain (single-start) alternating projection between two convex sets
    /// is only guaranteed to *converge*, not to converge quickly — and when
    /// the only points of intersection are on the relative boundary of the
    /// PSD cone (a **singular** witnessing Gram matrix, which is exactly the
    /// case for a tight/extremal SOS certificate — Motzkin's among them),
    /// it can stall a long way short of the intersection instead. Trying
    /// several starting points and keeping the best result (see
    /// [`super::psd::psd_search`]) is the standard mitigation; this method
    /// is what makes that possible.
    pub fn search_from(&self, start: Vec<f64>, floor: f64, iters: usize) -> Option<Vec<f64>> {
        let mut t = start;
        let mut q = self.at(&t);
        for _ in 0..iters {
            let projected = project_psd(&q, floor);
            let next = self.parameters_of(&projected)?;
            let moved: f64 = next
                .iter()
                .zip(t.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            t = next;
            q = self.at(&t);
            if moved < 1e-13 {
                break;
            }
        }
        if t.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonalises_a_known_matrix() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (mut vals, _) = jacobi_eigen(&a);
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((vals[0] - 1.0).abs() < 1e-10, "{vals:?}");
        assert!((vals[1] - 3.0).abs() < 1e-10, "{vals:?}");
    }

    #[test]
    fn min_eigenvalue_sees_indefiniteness() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(min_eigenvalue(&a) < -0.9);
    }

    #[test]
    fn psd_projection_clips_negative_eigenvalues() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let p = project_psd(&a, 0.0);
        assert!(min_eigenvalue(&p) > -1e-9);
    }

    #[test]
    fn search_finds_a_psd_point_of_a_family() {
        // base = [[1, 1], [1, 0]] (indefinite), direction adds to the (1,1) entry.
        let base = vec![vec![1.0, 1.0], vec![1.0, 0.0]];
        let dirs = vec![vec![vec![0.0, 0.0], vec![0.0, 1.0]]];
        let fam = Family::new(base, dirs);
        let t = fam.search(0.25, 500).expect("search runs");
        assert!(min_eigenvalue(&fam.at(&t)) > -1e-9, "t = {t:?}");
        // The PSD points are exactly t ≥ 1; the interior floor should overshoot it.
        assert!(t[0] >= 1.0, "t = {t:?}");
    }

    #[test]
    fn search_on_an_empty_family_is_a_no_op() {
        let fam = Family::new(vec![vec![1.0]], Vec::new());
        let t = fam.search(0.1, 10).expect("no parameters");
        assert!(t.is_empty());
    }
}
