//! Sparse direct solves for the implicit integrators: reverse Cuthill–McKee
//! ordering followed by a banded LU factorisation with partial pivoting
//! (the LAPACK `gbtrf`/`gbtrs` algorithm).
//!
//! Method-of-lines Jacobians are stencil-structured: after RCM a 1-D problem
//! (periodic ones included) has a bandwidth of a few points and a 2-D
//! `n × n` problem a bandwidth of about `n`, so the factorisation costs
//! `O(N · kl · (kl + ku))` and fits in `O(N · (2 kl + ku))` memory.

#![allow(clippy::needless_range_loop)]

use super::CsrMatrix;

/// Reverse Cuthill–McKee ordering of the symmetrised pattern of `a`.
/// Returns `perm` with `perm[new] = old`.
pub fn rcm_order(a: &CsrMatrix) -> Vec<usize> {
    let n = a.n_rows;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for r in 0..n {
        for &c in &a.indices[a.indptr[r]..a.indptr[r + 1]] {
            if c != r {
                adj[r].push(c);
                adj[c].push(r);
            }
        }
    }
    for l in adj.iter_mut() {
        l.sort_unstable();
        l.dedup();
    }
    let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut by_degree: Vec<usize> = (0..n).collect();
    by_degree.sort_by_key(|&v| (deg[v], v));
    for &start in &by_degree {
        if visited[start] {
            continue;
        }
        // A pseudo-peripheral start: repeatedly jump to a farthest,
        // lowest-degree node of the BFS level structure.
        let root = pseudo_peripheral(start, &adj, &deg);
        visited[root] = true;
        let head = order.len();
        order.push(root);
        let mut q = head;
        while q < order.len() {
            let v = order[q];
            q += 1;
            let mut nb: Vec<usize> = adj[v].iter().copied().filter(|&w| !visited[w]).collect();
            nb.sort_by_key(|&w| (deg[w], w));
            for w in nb {
                visited[w] = true;
                order.push(w);
            }
        }
    }
    order.reverse();
    order
}

fn pseudo_peripheral(start: usize, adj: &[Vec<usize>], deg: &[usize]) -> usize {
    let mut root = start;
    let mut ecc = 0usize;
    for _ in 0..8 {
        let (levels, last) = bfs_levels(root, adj);
        let far = last
            .iter()
            .copied()
            .min_by_key(|&v| (deg[v], v))
            .unwrap_or(root);
        if levels <= ecc {
            break;
        }
        ecc = levels;
        root = far;
    }
    root
}

fn bfs_levels(root: usize, adj: &[Vec<usize>]) -> (usize, Vec<usize>) {
    let mut seen = std::collections::HashSet::new();
    seen.insert(root);
    let mut level = vec![root];
    let mut depth = 0;
    loop {
        let mut next = Vec::new();
        for &v in &level {
            for &w in &adj[v] {
                if seen.insert(w) {
                    next.push(w);
                }
            }
        }
        if next.is_empty() {
            return (depth, level);
        }
        depth += 1;
        level = next;
    }
}

/// An LU factorisation `P A = L U` of a permuted banded matrix.
#[derive(Clone, Debug)]
pub struct BandedLu {
    n: usize,
    kl: usize,
    ku: usize,
    ldab: usize,
    ab: Vec<f64>,
    ipiv: Vec<usize>,
    /// `perm[new] = old`.
    perm: Vec<usize>,
}

impl BandedLu {
    /// The bandwidths `(kl, ku)` of `a` under the ordering `perm`.
    pub fn bandwidths(a: &CsrMatrix, perm: &[usize]) -> (usize, usize) {
        let mut inv = vec![0usize; perm.len()];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let (mut kl, mut ku) = (0usize, 0usize);
        for r in 0..a.n_rows {
            let i = inv[r];
            for &c in &a.indices[a.indptr[r]..a.indptr[r + 1]] {
                let j = inv[c];
                if i > j {
                    kl = kl.max(i - j);
                } else {
                    ku = ku.max(j - i);
                }
            }
        }
        (kl, ku)
    }

    /// Bytes of band storage a factorisation with these bandwidths needs.
    pub fn storage_bytes(n: usize, kl: usize, ku: usize) -> usize {
        n.saturating_mul(2 * kl + ku + 1)
            .saturating_mul(std::mem::size_of::<f64>())
    }

    /// Factor `a` (square CSR) in the ordering `perm`, with bandwidths
    /// `(kl, ku)` from [`bandwidths`](Self::bandwidths). Returns `None` for a
    /// singular matrix.
    pub fn factor(a: &CsrMatrix, perm: &[usize], kl: usize, ku: usize) -> Option<BandedLu> {
        let n = a.n_rows;
        let kv = kl + ku;
        let ldab = 2 * kl + ku + 1;
        let mut ab = vec![0.0; ldab * n];
        let mut inv = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let idx = |i: usize, c: usize| c * ldab + (kv + i - c);
        for r in 0..n {
            let i = inv[r];
            for k in a.indptr[r]..a.indptr[r + 1] {
                let j = inv[a.indices[k]];
                ab[idx(i, j)] += a.data[k];
            }
        }
        let mut ipiv = vec![0usize; n];
        let mut ju = 0usize;
        for j in 0..n {
            let km = kl.min(n - 1 - j);
            // Pivot search in column j, rows j..=j+km.
            let mut jp = 0;
            let mut best = ab[idx(j, j)].abs();
            for i in 1..=km {
                let v = ab[idx(j + i, j)].abs();
                if v > best {
                    best = v;
                    jp = i;
                }
            }
            ipiv[j] = j + jp;
            if best == 0.0 || !best.is_finite() {
                return None;
            }
            ju = ju.max((j + ku + jp).min(n - 1));
            if jp != 0 {
                for c in j..=ju {
                    ab.swap(idx(j + jp, c), idx(j, c));
                }
            }
            if km > 0 {
                let piv = ab[idx(j, j)];
                for i in 1..=km {
                    ab[idx(j + i, j)] /= piv;
                }
                for c in j + 1..=ju {
                    let ujc = ab[idx(j, c)];
                    if ujc != 0.0 {
                        for i in 1..=km {
                            let l = ab[idx(j + i, j)];
                            ab[idx(j + i, c)] -= l * ujc;
                        }
                    }
                }
            }
        }
        Some(BandedLu {
            n,
            kl,
            ku,
            ldab,
            ab,
            ipiv,
            perm: perm.to_vec(),
        })
    }

    /// Solve `A x = b` in place (`b` in the original ordering).
    pub fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        let kv = self.kl + self.ku;
        let idx = |i: usize, c: usize| c * self.ldab + (kv + i - c);
        let mut x: Vec<f64> = self.perm.iter().map(|&old| b[old]).collect();
        for j in 0..n {
            let lm = self.kl.min(n - 1 - j);
            let jp = self.ipiv[j];
            if jp != j {
                x.swap(j, jp);
            }
            let xj = x[j];
            if xj != 0.0 {
                for i in 1..=lm {
                    x[j + i] -= self.ab[idx(j + i, j)] * xj;
                }
            }
        }
        for j in (0..n).rev() {
            x[j] /= self.ab[idx(j, j)];
            let xj = x[j];
            if xj != 0.0 {
                for i in j.saturating_sub(kv)..j {
                    x[i] -= self.ab[idx(i, j)] * xj;
                }
            }
        }
        for (new, &old) in self.perm.iter().enumerate() {
            b[old] = x[new];
        }
    }
}
