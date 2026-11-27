//! Sparse iterative solves for the implicit integrators: an incomplete LU
//! factorisation with zero fill-in (ILU(0)) used as a left preconditioner
//! for restarted GMRES (Saad & Schultz 1986; Saad, *Iterative Methods for
//! Sparse Linear Systems*, 2nd ed., Alg. 10.4 and 9.5).
//!
//! The banded direct solver of [`super::banded`] costs `O(N · kl²)` per
//! factorisation and `O(N · kl)` memory. That is ideal in one dimension and
//! fine for small 2-D grids, but an `n × n` grid has `kl ≈ n` (so `O(n⁴)`
//! work) and an `n³` grid `kl ≈ n²` (`O(n⁷)` work, `O(n⁵)` memory). ILU(0)
//! keeps the Jacobian's own sparsity pattern, so its factorisation and each
//! preconditioner application cost `O(nnz)`, and GMRES needs only products
//! with the exact sparse iteration matrix.
//!
//! # Determinism
//!
//! Every reduction (dot products, norms) is computed as a sum of fixed-size
//! chunks combined in chunk order, and every matrix-vector product row by
//! row, so the result is bitwise identical for every thread count, with or
//! without the `parallel` feature. The ILU(0) factorisation and triangular
//! solves are sequential.

#![allow(clippy::needless_range_loop)]

use super::CsrMatrix;

/// Reductions are summed in chunks of this many terms (in chunk order),
/// whatever the thread count.
const RED_CHUNK: usize = 4096;
/// Below this many rows a matrix-vector product or a reduction stays on one
/// thread: GMRES is dominated by the sequential ILU(0) solves, and on smaller
/// systems the fork-join overhead of each Arnoldi step outweighs the gain.
#[cfg(feature = "parallel")]
const PAR_MIN_ROWS: usize = 262_144;

/// Deterministic dot product: fixed chunks, combined in order.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let partial = |c: usize| {
        let lo = c * RED_CHUNK;
        let hi = (lo + RED_CHUNK).min(a.len());
        let mut s = 0.0;
        for i in lo..hi {
            s += a[i] * b[i];
        }
        s
    };
    let nc = a.len().div_ceil(RED_CHUNK);
    #[cfg(feature = "parallel")]
    if a.len() >= PAR_MIN_ROWS {
        use rayon::prelude::*;
        let parts: Vec<f64> = (0..nc).into_par_iter().map(partial).collect();
        let mut s = 0.0;
        for p in parts {
            s += p;
        }
        return s;
    }
    let mut s = 0.0;
    for c in 0..nc {
        s += partial(c);
    }
    s
}

/// `y = A x`, row by row (each row's sum in column order).
pub(crate) fn matvec(a: &CsrMatrix, x: &[f64], y: &mut [f64]) {
    let row = |r: usize| {
        let mut s = 0.0;
        for k in a.indptr[r]..a.indptr[r + 1] {
            s += a.data[k] * x[a.indices[k]];
        }
        s
    };
    #[cfg(feature = "parallel")]
    if a.n_rows >= PAR_MIN_ROWS {
        use rayon::prelude::*;
        y[..a.n_rows]
            .par_chunks_mut(RED_CHUNK)
            .enumerate()
            .for_each(|(c, ys)| {
                for (j, v) in ys.iter_mut().enumerate() {
                    *v = row(c * RED_CHUNK + j);
                }
            });
        return;
    }
    for (r, v) in y.iter_mut().enumerate().take(a.n_rows) {
        *v = row(r);
    }
}

/// ILU(0): `A ≈ L U` on the sparsity pattern of `A` (unit lower `L`).
#[derive(Clone, Debug)]
pub struct Ilu0 {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    /// Position of the diagonal entry of each row.
    diag: Vec<usize>,
    /// `L` strictly below the diagonal, `U` on and above it.
    lu: Vec<f64>,
}

impl Ilu0 {
    /// Factor a square CSR matrix whose rows have sorted column indices and
    /// an explicit diagonal entry. Returns `None` on a zero or non-finite
    /// pivot (the caller falls back to the direct solver).
    pub fn factor(a: &CsrMatrix) -> Option<Ilu0> {
        let n = a.n_rows;
        if a.n_cols != n || a.indptr.len() != n + 1 {
            return None;
        }
        let mut diag = vec![usize::MAX; n];
        for r in 0..n {
            let seg = &a.indices[a.indptr[r]..a.indptr[r + 1]];
            if seg.windows(2).any(|w| w[0] >= w[1]) {
                return None;
            }
            diag[r] = a.indptr[r] + seg.binary_search(&r).ok()?;
        }
        let mut lu = a.data.clone();
        // pos[col] = position of `col` in the current row, or NONE.
        const NONE: usize = usize::MAX;
        let mut pos = vec![NONE; n];
        for i in 0..n {
            let (lo, hi) = (a.indptr[i], a.indptr[i + 1]);
            for p in lo..hi {
                pos[a.indices[p]] = p;
            }
            for p in lo..diag[i] {
                let k = a.indices[p];
                let piv = lu[diag[k]];
                let l = lu[p] / piv;
                lu[p] = l;
                if l != 0.0 {
                    for q in diag[k] + 1..a.indptr[k + 1] {
                        let j = a.indices[q];
                        let at = pos[j];
                        if at != NONE {
                            lu[at] -= l * lu[q];
                        }
                    }
                }
            }
            let d = lu[diag[i]];
            for p in lo..hi {
                pos[a.indices[p]] = NONE;
            }
            if d == 0.0 || !d.is_finite() {
                return None;
            }
        }
        if lu.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(Ilu0 {
            n,
            indptr: a.indptr.clone(),
            indices: a.indices.clone(),
            diag,
            lu,
        })
    }

    /// `x ← U⁻¹ L⁻¹ x` in place.
    pub fn solve(&self, x: &mut [f64]) {
        for i in 0..self.n {
            let mut s = x[i];
            for p in self.indptr[i]..self.diag[i] {
                s -= self.lu[p] * x[self.indices[p]];
            }
            x[i] = s;
        }
        for i in (0..self.n).rev() {
            let mut s = x[i];
            for p in self.diag[i] + 1..self.indptr[i + 1] {
                s -= self.lu[p] * x[self.indices[p]];
            }
            x[i] = s / self.lu[self.diag[i]];
        }
    }
}

/// Outcome of one [`gmres`] solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmresOutcome {
    /// Krylov iterations (matrix-vector products) performed.
    pub iterations: usize,
    /// Whether the weighted preconditioned residual reached the tolerance.
    pub converged: bool,
    /// The final weighted RMS preconditioned residual
    /// `rms(w ⊙ M⁻¹(b - A x))`.
    pub residual: f64,
}

/// Restarted, left-preconditioned GMRES for `A x = b` in the weighted RMS
/// norm `‖v‖ = rms(w ⊙ v)`: the iteration runs on the scaled system
/// `(W M⁻¹ A W⁻¹) (W x) = W M⁻¹ b`. Its residual `M⁻¹(b − A x)` approximates
/// the *error* of `x` (as `M ≈ A`), in the norm of the integrator's Newton
/// corrections — so a badly scaled row (an algebraic constraint multiplied
/// by 10⁻¹²) cannot hide a large error behind a small residual.
///
/// `b` is overwritten with the solution (initial guess zero). Stops when the
/// weighted RMS preconditioned residual is `≤ tol`, after `max_iters`
/// iterations (`0`: only the zero guess is tested), or on breakdown.
pub fn gmres(
    a: &CsrMatrix,
    precond: &Ilu0,
    w: &[f64],
    b: &mut [f64],
    tol: f64,
    restart: usize,
    max_iters: usize,
) -> GmresOutcome {
    let n = a.n_rows;
    // The basis never needs more vectors than the iteration budget allows.
    let m = restart.min(max_iters).clamp(1, n.max(1));
    let sqrt_n = (n.max(1) as f64).sqrt();
    let mut x = vec![0.0; n];
    // r = W M⁻¹ b, the scaled preconditioned residual of the zero guess.
    let mut r = b.to_vec();
    precond.solve(&mut r);
    for i in 0..n {
        r[i] *= w[i];
    }
    let mut beta = dot(&r, &r).sqrt();
    let mut total = 0usize;
    let target = tol * sqrt_n;
    let finish = |x: Vec<f64>, b: &mut [f64], iters, conv, res: f64| {
        b.copy_from_slice(&x);
        GmresOutcome {
            iterations: iters,
            converged: conv,
            residual: res / sqrt_n,
        }
    };
    if !beta.is_finite() {
        return finish(x, b, 0, false, beta);
    }
    if beta <= target {
        return finish(x, b, 0, true, beta);
    }
    if max_iters == 0 {
        return finish(x, b, 0, false, beta);
    }
    let mut v: Vec<Vec<f64>> = vec![vec![0.0; n]; m + 1];
    let mut h = vec![vec![0.0; m]; m + 1];
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];
    let mut tmp = vec![0.0; n];
    let mut av = vec![0.0; n];
    loop {
        for i in 0..n {
            v[0][i] = r[i] / beta;
        }
        g.iter_mut().for_each(|e| *e = 0.0);
        g[0] = beta;
        let mut k_done = 0usize;
        let mut res = beta;
        for k in 0..m {
            // av = W M⁻¹ A W⁻¹ v_k
            for i in 0..n {
                tmp[i] = v[k][i] / w[i];
            }
            matvec(a, &tmp, &mut av);
            precond.solve(&mut av);
            for i in 0..n {
                av[i] *= w[i];
            }
            total += 1;
            // Modified Gram–Schmidt.
            for j in 0..=k {
                let hj = dot(&av, &v[j]);
                h[j][k] = hj;
                for i in 0..n {
                    av[i] -= hj * v[j][i];
                }
            }
            let hn = dot(&av, &av).sqrt();
            h[k + 1][k] = hn;
            if hn > 0.0 && hn.is_finite() {
                for i in 0..n {
                    v[k + 1][i] = av[i] / hn;
                }
            }
            // Apply the previous rotations, then a new one.
            for j in 0..k {
                let t = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
                h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
                h[j][k] = t;
            }
            let (hk, hk1) = (h[k][k], h[k + 1][k]);
            let den = hk.hypot(hk1);
            if den == 0.0 || !den.is_finite() {
                break;
            }
            cs[k] = hk / den;
            sn[k] = hk1 / den;
            h[k][k] = den;
            h[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] *= cs[k];
            res = g[k + 1].abs();
            k_done = k + 1;
            if res <= target || total >= max_iters || !(hn > 0.0 && hn.is_finite()) {
                break;
            }
        }
        if k_done == 0 {
            return finish(x, b, total, false, res);
        }
        // Back-substitute y from the triangular H; x += W⁻¹ V y.
        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev() {
            let mut s = g[i];
            for j in i + 1..k_done {
                s -= h[i][j] * y[j];
            }
            y[i] = s / h[i][i];
        }
        tmp.iter_mut().for_each(|e| *e = 0.0);
        for (j, &yj) in y.iter().enumerate() {
            for i in 0..n {
                tmp[i] += yj * v[j][i];
            }
        }
        for i in 0..n {
            x[i] += tmp[i] / w[i];
        }
        // The true preconditioned residual for the restart (and the report).
        matvec(a, &x, &mut av);
        for i in 0..n {
            av[i] = b[i] - av[i];
        }
        precond.solve(&mut av);
        for i in 0..n {
            r[i] = w[i] * av[i];
        }
        beta = dot(&r, &r).sqrt();
        if !beta.is_finite() {
            return finish(x, b, total, false, beta);
        }
        if beta <= target {
            return finish(x, b, total, true, beta);
        }
        if total >= max_iters {
            return finish(x, b, total, false, beta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 5-point Laplacian on an `m × m` grid (Dirichlet), shifted:
    /// `I - c Δ_h`.
    fn shifted_laplacian(m: usize, c: f64) -> CsrMatrix {
        let n = m * m;
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..m {
            for j in 0..m {
                let r = i * m + j;
                let mut row: Vec<(usize, f64)> = vec![(r, 1.0 + 4.0 * c)];
                if i > 0 {
                    row.push((r - m, -c));
                }
                if i + 1 < m {
                    row.push((r + m, -c));
                }
                if j > 0 {
                    row.push((r - 1, -c));
                }
                if j + 1 < m {
                    row.push((r + 1, -c));
                }
                row.sort_by_key(|e| e.0);
                for (cc, v) in row {
                    indices.push(cc);
                    data.push(v);
                }
                indptr.push(indices.len());
            }
        }
        CsrMatrix {
            n_rows: n,
            n_cols: n,
            indptr,
            indices,
            data,
        }
    }

    #[test]
    fn ilu0_is_exact_for_a_tridiagonal_matrix() {
        // No fill-in: ILU(0) of a tridiagonal matrix is its LU.
        let n = 50;
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in 0..n {
            if r > 0 {
                indices.push(r - 1);
                data.push(-1.0);
            }
            indices.push(r);
            data.push(2.5);
            if r + 1 < n {
                indices.push(r + 1);
                data.push(-1.0);
            }
            indptr.push(indices.len());
        }
        let a = CsrMatrix {
            n_rows: n,
            n_cols: n,
            indptr,
            indices,
            data,
        };
        let ilu = Ilu0::factor(&a).unwrap();
        let x0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut b = vec![0.0; n];
        a.matvec(&x0, &mut b);
        ilu.solve(&mut b);
        for (u, v) in b.iter().zip(&x0) {
            assert!((u - v).abs() < 1e-12, "{u} vs {v}");
        }
    }

    #[test]
    fn gmres_solves_a_shifted_laplacian() {
        let m = 40;
        let a = shifted_laplacian(m, 50.0);
        let n = m * m;
        let x0: Vec<f64> = (0..n).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();
        let mut b = vec![0.0; n];
        a.matvec(&x0, &mut b);
        let ilu = Ilu0::factor(&a).unwrap();
        let w = vec![1.0; n];
        let out = gmres(&a, &ilu, &w, &mut b, 1e-10, 30, 500);
        assert!(out.converged, "{out:?}");
        let err = b
            .iter()
            .zip(&x0)
            .map(|(u, v)| (u - v).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "max error {err}");
    }

    #[test]
    fn gmres_is_bitwise_identical_across_thread_counts() {
        // 520² rows: above the parallel threshold of the products and
        // reductions.
        let m = 520;
        let a = shifted_laplacian(m, 20.0);
        let n = m * m;
        let ilu = Ilu0::factor(&a).unwrap();
        let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.1).collect();
        let b0: Vec<f64> = (0..n).map(|i| ((i * 31 % 97) as f64).cos()).collect();
        let run = |threads: usize| {
            super::super::with_threads(threads, || {
                let mut b = b0.clone();
                let out = gmres(&a, &ilu, &w, &mut b, 1e-8, 30, 60);
                (b, out)
            })
        };
        let (x1, o1) = run(1);
        let (x4, o4) = run(4);
        assert_eq!(o1, o4);
        assert!(x1.iter().zip(&x4).all(|(p, q)| p.to_bits() == q.to_bits()));
    }

    #[test]
    fn ilu0_refuses_a_missing_diagonal() {
        let a = CsrMatrix {
            n_rows: 2,
            n_cols: 2,
            indptr: vec![0, 1, 2],
            indices: vec![1, 0],
            data: vec![1.0, 1.0],
        };
        assert!(Ilu0::factor(&a).is_none());
    }

    #[test]
    fn dot_is_chunked_deterministically() {
        let n = 3 * RED_CHUNK + 17;
        let a: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect();
        let s1 = dot(&a, &a);
        let mut s2 = 0.0;
        for c in 0..n.div_ceil(RED_CHUNK) {
            let mut p = 0.0;
            for i in c * RED_CHUNK..((c + 1) * RED_CHUNK).min(n) {
                p += a[i] * a[i];
            }
            s2 += p;
        }
        assert_eq!(s1.to_bits(), s2.to_bits());
    }
}
