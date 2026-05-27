//! FGLM algorithm: convert a 0-dimensional Gröbner basis from GRevLex to Lex order.
//!
//! Reference: Faugère, Gianni, Lazard, Mora (1993) "Efficient Computation of
//! Zero-Dimensional Gröbner Bases by Change of Ordering".
//!
//! For 0-dimensional ideals, this is typically orders of magnitude faster than
//! computing a lex basis directly via Buchberger, because GRevLex Buchberger is
//! cheap and the FGLM conversion is O(n * D^3) where D = dim(k\[x\]/I).

use crate::poly::groebner::buchberger::interreduce;
use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::reduce::reduce;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Check whether a GRevLex Gröbner basis represents a 0-dimensional ideal.
///
/// Sufficient condition: for each variable index i, the basis contains a
/// polynomial whose leading monomial is a pure power x_i^d (d ≥ 1).
pub fn is_zero_dimensional(basis: &[GbPoly], n_vars: usize) -> bool {
    (0..n_vars).all(|i| {
        basis.iter().any(|g| {
            g.leading_exp(MonomialOrder::GRevLex)
                .map(|lm| lm[i] > 0 && lm.iter().enumerate().all(|(j, &e)| j == i || e == 0))
                .unwrap_or(false)
        })
    })
}

/// Enumerate the GRevLex staircase of `basis` (monomials not in the leading-term
/// ideal). For 0-dimensional ideals this list is finite; returns `None` if the
/// staircase does not terminate within `MAX_DEGREE` total degree (indicating a
/// positive-dimensional ideal or a basis that hasn't been fully computed).
pub fn grevlex_staircase(basis: &[GbPoly], n_vars: usize) -> Option<Vec<Vec<u32>>> {
    const MAX_DEGREE: u32 = 50;

    let lms: Vec<Vec<u32>> = basis
        .iter()
        .filter_map(|g| g.leading_exp(MonomialOrder::GRevLex))
        .collect();

    let mut staircase: Vec<Vec<u32>> = Vec::new();

    for d in 0..=MAX_DEGREE {
        let new_at_d: Vec<Vec<u32>> = monomials_of_degree(n_vars, d)
            .into_iter()
            .filter(|exp| !lms.iter().any(|lm| divides(lm, exp)))
            .collect();

        if new_at_d.is_empty() {
            staircase.sort_by(|a, b| MonomialOrder::GRevLex.cmp(a, b));
            return Some(staircase);
        }
        staircase.extend(new_at_d);
    }

    None // Staircase did not terminate — not 0-dimensional within cap
}

/// Convert a GRevLex Gröbner basis to a Lex Gröbner basis using the FGLM
/// algorithm. Returns `None` if the ideal is not 0-dimensional (staircase
/// exceeds the cap) or if an internal iteration limit is exceeded.
pub fn fglm(grevlex_basis: &[GbPoly], n_vars: usize) -> Option<Vec<GbPoly>> {
    if n_vars <= 1 {
        // Univariate: GRevLex == Lex, no conversion needed.
        return Some(grevlex_basis.to_vec());
    }

    let staircase = grevlex_staircase(grevlex_basis, n_vars)?;
    let d = staircase.len();

    if d == 0 {
        // 1 ∈ I → trivial ideal.
        let mut terms = std::collections::BTreeMap::new();
        terms.insert(vec![0u32; n_vars], rug::Rational::from(1));
        return Some(vec![GbPoly { terms, n_vars }]);
    }

    // Build a fast index map: exponent vector → position in staircase.
    let staircase_idx: HashMap<Vec<u32>, usize> = staircase
        .iter()
        .enumerate()
        .map(|(i, e)| (e.clone(), i))
        .collect();

    // FGLM state
    let mut gauss = GaussState::new(d);
    let mut lex_standard: Vec<Vec<u32>> = Vec::new();
    let mut lex_basis: Vec<GbPoly> = Vec::new();
    let mut lex_basis_lms: Vec<Vec<u32>> = Vec::new(); // LMs for pruning

    // Visited set and min-heap ordered by ascending Lex.
    let mut visited: HashSet<Vec<u32>> = HashSet::new();
    let mut heap: BinaryHeap<Reverse<LexMonomial>> = BinaryHeap::new();

    // Seed with the constant monomial 1.
    let one = vec![0u32; n_vars];
    visited.insert(one.clone());
    {
        let nf = GbPoly::monomial(one.clone(), rug::Rational::from(1));
        let v = coord_vector(&nf, &staircase_idx, d);
        match gauss.insert_or_express(v) {
            InsertResult::Independent => lex_standard.push(one.clone()),
            InsertResult::Dependent(_) => {
                // 1 ∈ I
                let mut terms = std::collections::BTreeMap::new();
                terms.insert(one, rug::Rational::from(1));
                return Some(vec![GbPoly { terms, n_vars }]);
            }
        }
    }

    // Enqueue each variable.
    for i in 0..n_vars {
        let mut exp = vec![0u32; n_vars];
        exp[i] = 1;
        if visited.insert(exp.clone()) {
            heap.push(Reverse(LexMonomial(exp)));
        }
    }

    // Safety cap: at most d lex-standard + d*n lex-basis elements.
    let safety_cap = d * (n_vars + 2) + 200;
    let mut iters = 0usize;

    while let Some(Reverse(LexMonomial(m_exp))) = heap.pop() {
        iters += 1;
        if iters > safety_cap {
            return None;
        }

        // Compute NF(m_exp) under GRevLex.
        let m_poly = GbPoly::monomial(m_exp.clone(), rug::Rational::from(1));
        let nf = reduce(&m_poly, grevlex_basis, MonomialOrder::GRevLex);
        let v = coord_vector(&nf, &staircase_idx, d);

        match gauss.insert_or_express(v) {
            InsertResult::Independent => {
                lex_standard.push(m_exp.clone());
                // Enqueue children x_i * m_exp.
                for i in 0..n_vars {
                    let mut child = m_exp.clone();
                    child[i] += 1;
                    // Skip if divisible by a known lex-basis LM.
                    if divides_any(&child, &lex_basis_lms) {
                        continue;
                    }
                    if visited.insert(child.clone()) {
                        heap.push(Reverse(LexMonomial(child)));
                    }
                }
            }
            InsertResult::Dependent(coeffs) => {
                // Build lex polynomial: m_exp - Σ c_j * lex_standard[j]
                let mut poly = GbPoly::monomial(m_exp.clone(), rug::Rational::from(1));
                for (j, c) in coeffs.iter().enumerate() {
                    if *c != 0 {
                        let neg_c = rug::Rational::from(-c);
                        let term = GbPoly::monomial(lex_standard[j].clone(), neg_c);
                        poly = poly.add(&term);
                    }
                }
                lex_basis_lms.push(m_exp);
                lex_basis.push(poly);
            }
        }
    }

    if lex_standard.len() < d {
        // Did not find the full basis — fallback.
        return None;
    }

    Some(interreduce(lex_basis, MonomialOrder::Lex))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn divides(lm: &[u32], exp: &[u32]) -> bool {
    lm.len() <= exp.len() && lm.iter().zip(exp.iter()).all(|(a, b)| a <= b)
}

fn divides_any(exp: &[u32], lms: &[Vec<u32>]) -> bool {
    lms.iter().any(|lm| divides(lm, exp))
}

/// Enumerate all monomials of exactly `total_deg` in `n_vars` variables.
fn monomials_of_degree(n_vars: usize, total_deg: u32) -> Vec<Vec<u32>> {
    if n_vars == 0 {
        return if total_deg == 0 { vec![vec![]] } else { vec![] };
    }
    let mut result = Vec::new();
    let mut exp = vec![0u32; n_vars];
    gen_degree(&mut exp, 0, n_vars, total_deg, &mut result);
    result
}

fn gen_degree(exp: &mut Vec<u32>, vi: usize, n: usize, rem: u32, out: &mut Vec<Vec<u32>>) {
    if vi == n - 1 {
        exp[vi] = rem;
        out.push(exp.clone());
        exp[vi] = 0;
        return;
    }
    for k in 0..=rem {
        exp[vi] = k;
        gen_degree(exp, vi + 1, n, rem - k, out);
    }
    exp[vi] = 0;
}

/// Express `p` (already reduced under GRevLex) as a coordinate vector in Q^D,
/// coordinates indexed by the staircase via `staircase_idx`.
fn coord_vector(
    p: &GbPoly,
    staircase_idx: &HashMap<Vec<u32>, usize>,
    d: usize,
) -> Vec<rug::Rational> {
    let mut v = vec![rug::Rational::from(0); d];
    for (exp, coeff) in &p.terms {
        if let Some(&idx) = staircase_idx.get(exp) {
            v[idx] = coeff.clone();
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Lex monomial ordering for BinaryHeap (min-heap via Reverse)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct LexMonomial(Vec<u32>);

impl Ord for LexMonomial {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        MonomialOrder::Lex.cmp(&self.0, &other.0)
    }
}
impl PartialOrd for LexMonomial {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Gaussian elimination state
// ---------------------------------------------------------------------------

enum InsertResult {
    Independent,
    Dependent(Vec<rug::Rational>),
}

/// Maintains a partial row-echelon form for linear independence testing.
/// Also tracks, for each RREF row, its expression as a linear combination
/// of the originally-inserted vectors (needed to recover FGLM coefficients).
struct GaussState {
    d: usize,
    /// Rows in (partial) row echelon form, each normalized so its pivot = 1.
    rref: Vec<Vec<rug::Rational>>,
    /// Pivot column index for each RREF row.
    pivot_col: Vec<usize>,
    /// For column c, which RREF row has its pivot there (if any).
    pivot_col_to_row: Vec<Option<usize>>,
    /// transform[i] = coefficients over original inserted vectors for rref[i]:
    /// rref[i] = Σ_j transform[i][j] * orig[j]
    /// Length of transform[i] is i+1 (only originals 0..=i can contribute).
    transform: Vec<Vec<rug::Rational>>,
    /// Number of rows (= number of independent vectors inserted so far).
    k: usize,
}

impl GaussState {
    fn new(d: usize) -> Self {
        GaussState {
            d,
            rref: Vec::new(),
            pivot_col: Vec::new(),
            pivot_col_to_row: vec![None; d],
            transform: Vec::new(),
            k: 0,
        }
    }

    /// Try to express `v` in terms of already-inserted vectors.
    ///
    /// - `InsertResult::Independent`: `v` is linearly independent; it is inserted
    ///   as the next basis vector and can be referenced in future queries.
    /// - `InsertResult::Dependent(coeffs)`: `v = Σ_j coeffs[j] * orig[j]` where
    ///   `orig[j]` is the j-th vector passed to a prior `insert_or_express` call
    ///   that returned `Independent`.
    fn insert_or_express(&mut self, v: Vec<rug::Rational>) -> InsertResult {
        let (residual, combo) = self.reduce(v);

        if residual.iter().all(|c| *c == 0) {
            return InsertResult::Dependent(self.orig_combo(&combo));
        }

        // Independent: find pivot column and normalize.
        let pc = residual.iter().position(|c| *c != 0).unwrap();
        let pivot_val = residual[pc].clone();
        let normalized: Vec<rug::Rational> = residual
            .iter()
            .map(|c| rug::Rational::from(c / &pivot_val))
            .collect();

        // Build transform row: rref[new] = (orig_k - Σ combo[i]*rref[i]) / pivot_val
        // transform[i] has length i+1, so only access transform[i][j] when j < i+1.
        let k = self.k;
        let mut new_tr = vec![rug::Rational::from(0); k + 1];
        new_tr[k] = rug::Rational::from(1) / pivot_val.clone(); // contribution from orig_k
        for (j, entry) in new_tr.iter_mut().enumerate().take(k) {
            let mut s = rug::Rational::from(0);
            for (i, c) in combo.iter().enumerate() {
                if *c != 0 && j < self.transform[i].len() {
                    s += rug::Rational::from(c * &self.transform[i][j]);
                }
            }
            *entry = rug::Rational::from(&(-s) / &pivot_val);
        }

        let row_idx = self.rref.len();
        self.pivot_col_to_row[pc] = Some(row_idx);
        self.rref.push(normalized);
        self.pivot_col.push(pc);
        self.transform.push(new_tr);
        self.k += 1;

        InsertResult::Independent
    }

    /// Reduce `v` against current RREF rows, accumulating row coefficients.
    /// Returns `(residual, combo)` where `v = Σ combo[i]*rref[i] + residual`.
    fn reduce(&self, mut v: Vec<rug::Rational>) -> (Vec<rug::Rational>, Vec<rug::Rational>) {
        let mut combo = vec![rug::Rational::from(0); self.rref.len()];
        for (i, row) in self.rref.iter().enumerate() {
            let pc = self.pivot_col[i];
            if v[pc] == 0 {
                continue;
            }
            let factor = v[pc].clone(); // pivot of row is 1
            for j in 0..self.d {
                if row[j] != 0 {
                    let sub = rug::Rational::from(&factor * &row[j]);
                    v[j] -= sub;
                }
            }
            combo[i] = factor;
        }
        (v, combo)
    }

    /// Convert combo-over-rref-rows to combo-over-original-vectors.
    fn orig_combo(&self, combo: &[rug::Rational]) -> Vec<rug::Rational> {
        let k = self.k;
        let mut out = vec![rug::Rational::from(0); k];
        for (i, c) in combo.iter().enumerate() {
            if *c == 0 {
                continue;
            }
            for (j, t) in self.transform[i].iter().enumerate() {
                if *t != 0 && j < k {
                    out[j] += rug::Rational::from(c * t);
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::groebner::buchberger::compute_buchberger_basis;
    use crate::poly::groebner::reduce::reduce as poly_reduce;

    fn rat(n: i64) -> rug::Rational {
        rug::Rational::from(n)
    }

    fn poly2(terms: &[(&[u32; 2], i64)]) -> GbPoly {
        GbPoly {
            terms: terms.iter().map(|(e, c)| (e.to_vec(), rat(*c))).collect(),
            n_vars: 2,
        }
    }

    fn bases_equivalent(a: &[GbPoly], b: &[GbPoly], order: MonomialOrder) -> bool {
        a.iter().all(|p| poly_reduce(p, b, order).is_zero())
            && b.iter().all(|p| poly_reduce(p, a, order).is_zero())
    }

    #[test]
    fn staircase_linear_system() {
        // Ideal (x + y - 1, x - y): solution x=1/2, y=1/2 → dim 1.
        let f = poly2(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]);
        let g = poly2(&[(&[1, 0], 1), (&[0, 1], -1)]);
        let gb = compute_buchberger_basis(vec![f, g], MonomialOrder::GRevLex);
        let sc = grevlex_staircase(&gb, 2).expect("should be 0-dim");
        assert_eq!(sc.len(), 1, "staircase should be {{1}}");
    }

    #[test]
    fn fglm_linear_system() {
        let f = poly2(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]);
        let g = poly2(&[(&[1, 0], 1), (&[0, 1], -1)]);
        let grb = compute_buchberger_basis(vec![f.clone(), g.clone()], MonomialOrder::GRevLex);
        let lex_fglm = fglm(&grb, 2).expect("should succeed");
        let lex_direct = compute_buchberger_basis(vec![f, g], MonomialOrder::Lex);
        assert!(
            bases_equivalent(&lex_fglm, &lex_direct, MonomialOrder::Lex),
            "FGLM lex basis differs from direct lex basis"
        );
    }

    #[test]
    fn fglm_circle_parabola() {
        // x^2 + y^2 - 1, y - x^2  (circle ∩ parabola, 2 real solutions)
        let f = poly2(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]);
        let g = poly2(&[(&[0, 1], 1), (&[2, 0], -1)]);
        let grb = compute_buchberger_basis(vec![f.clone(), g.clone()], MonomialOrder::GRevLex);
        let lex_fglm = fglm(&grb, 2).expect("should succeed");
        let lex_direct = compute_buchberger_basis(vec![f, g], MonomialOrder::Lex);
        assert!(
            bases_equivalent(&lex_fglm, &lex_direct, MonomialOrder::Lex),
            "FGLM lex basis differs from direct lex basis"
        );
    }

    #[test]
    fn fglm_katsura3() {
        // Katsura-3 in 3 variables (D = 4 solutions).
        // u0 + 2*u1 + 2*u2 - 1, u1^2 + 2*u0*u2 - u2, u0^2 + 2*u1^2 + 2*u2^2 - u0
        let mut f1 = GbPoly {
            terms: std::collections::BTreeMap::new(),
            n_vars: 3,
        };
        f1.terms.insert(vec![1, 0, 0], rug::Rational::from(1));
        f1.terms.insert(vec![0, 1, 0], rug::Rational::from(2));
        f1.terms.insert(vec![0, 0, 1], rug::Rational::from(2));
        f1.terms.insert(vec![0, 0, 0], rug::Rational::from(-1));

        let mut f2 = GbPoly {
            terms: std::collections::BTreeMap::new(),
            n_vars: 3,
        };
        f2.terms.insert(vec![0, 2, 0], rug::Rational::from(1));
        f2.terms.insert(vec![1, 0, 1], rug::Rational::from(2));
        f2.terms.insert(vec![0, 0, 1], rug::Rational::from(-1));

        let mut f3 = GbPoly {
            terms: std::collections::BTreeMap::new(),
            n_vars: 3,
        };
        f3.terms.insert(vec![2, 0, 0], rug::Rational::from(1));
        f3.terms.insert(vec![0, 2, 0], rug::Rational::from(2));
        f3.terms.insert(vec![0, 0, 2], rug::Rational::from(2));
        f3.terms.insert(vec![1, 0, 0], rug::Rational::from(-1));

        let grb = compute_buchberger_basis(
            vec![f1.clone(), f2.clone(), f3.clone()],
            MonomialOrder::GRevLex,
        );
        let lex_fglm = fglm(&grb, 3).expect("Katsura-3 should be 0-dim");
        let lex_direct = compute_buchberger_basis(vec![f1, f2, f3], MonomialOrder::Lex);
        assert!(
            bases_equivalent(&lex_fglm, &lex_direct, MonomialOrder::Lex),
            "FGLM katsura3 lex basis differs from direct lex"
        );
    }

    #[test]
    fn is_zero_dim_check() {
        // (x - 1, y - 2): 0-dimensional
        let f = GbPoly {
            terms: [
                (vec![1u32, 0], rug::Rational::from(1)),
                (vec![0, 0], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let g = GbPoly {
            terms: [
                (vec![0u32, 1], rug::Rational::from(1)),
                (vec![0, 0], rug::Rational::from(-2)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let gb = compute_buchberger_basis(vec![f, g], MonomialOrder::GRevLex);
        assert!(is_zero_dimensional(&gb, 2));
    }
}
