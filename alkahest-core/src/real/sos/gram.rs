//! Gram-matrix / DSOS (diagonally dominant sum-of-squares) search.
//!
//! For a target polynomial `p` of even total degree `2d`, a classical sum of
//! squares certificate `p = z^T Q z` (with `z` the monomial basis of degree
//! `≤ d`) exists iff there is a **positive semidefinite** Gram matrix `Q`
//! whose quadratic form matches `p`'s coefficients. Deciding *general* PSD
//! feasibility is a semidefinite programme, which this crate does not solve
//! exactly (no floating point is allowed near a certificate).
//!
//! Instead this module searches the **diagonally dominant** (DD) subcone,
//! which is *linear-programming representable* and can be solved with the
//! exact simplex in [`super::lp`]. Every DD matrix is PSD, so any solution
//! found here is a sound certificate. The DD cone is a strict subset of the
//! PSD cone, so failure here means only "no DD certificate at this basis" —
//! never "not SOS" and never "not nonnegative". The DD cone has the
//! well-known explicit generator decomposition used in [`dsos_search`]:
//!
//! ```text
//! Q = Σ_i r_i (e_i e_i^T) + Σ_{i<j} [ p_ij (e_i+e_j)(e_i+e_j)^T
//!                                    + m_ij (e_i-e_j)(e_i-e_j)^T ]
//! ```
//!
//! with `r_i, p_ij, m_ij ≥ 0`. Matching this to `p`'s coefficients is a
//! linear feasibility system in `(r, p, m)`, solved by [`super::lp::Lp`].
//! Every generator here is already a square (`e_i`, `e_i+e_j`, `e_i-e_j`
//! evaluated against the monomial basis), so a feasible point converts
//! directly into an explicit [`super::cert::SosPoly`] with no further work —
//! in particular no eigendecomposition, numeric or otherwise.

use super::cert::SosPoly;
use super::lp::{Lp, LpStatus, Rel};
use super::ratpoly::{Exponents, RatPoly};
use rug::Rational;
use std::collections::{BTreeMap, BTreeSet};

/// All exponent vectors in `nvars` variables with total degree `≤ max_deg`,
/// in graded-then-lexicographic order.
pub fn monomial_basis(nvars: usize, max_deg: u32) -> Vec<Exponents> {
    let mut out = Vec::new();
    if nvars == 0 {
        out.push(Vec::new());
        return out;
    }
    let mut cur = vec![0u32; nvars];
    fn rec(idx: usize, nvars: usize, remaining: u32, cur: &mut Vec<u32>, out: &mut Vec<Exponents>) {
        if idx == nvars {
            out.push(cur.clone());
            return;
        }
        for k in 0..=remaining {
            cur[idx] = k;
            rec(idx + 1, nvars, remaining - k, cur, out);
        }
        cur[idx] = 0;
    }
    rec(0, nvars, max_deg, &mut cur, &mut out);
    out.sort_by(|a, b| {
        let da: u32 = a.iter().sum();
        let db: u32 = b.iter().sum();
        da.cmp(&db).then_with(|| a.cmp(b))
    });
    out
}

fn add_exp(a: &[u32], b: &[u32]) -> Exponents {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Attempt a DSOS certificate for `p` using the monomial basis of degree
/// `≤ basis_deg` (covering target total degree up to `2 * basis_deg`).
///
/// Returns `Some(sos)` with `sos.to_poly(p.nvars()) == *p` exactly on
/// success (the caller should still re-verify — this module never skips
/// that discipline on the caller's behalf), `None` if the DD-Gram LP is
/// infeasible at this basis or `p` has no term reachable by it.
/// Coprime ratios `(a, b)` used to build the weighted generators
/// `(a·e_i ± b·e_j)²`.
///
/// The plain DD generators are the `(1, 1)` case. They alone are not enough:
/// diagonal dominance is not invariant under scaling the basis, so a perfect
/// square as ordinary as `(x/2 + 1/3)²` has a Gram matrix — its *only* Gram
/// matrix — that is PSD but not DD, and a search over `(1, 1)` alone would
/// refuse it. Adding a few fixed ratios enlarges the searched cone while
/// keeping every generator literally a square, so the certificate stays exact
/// and the LP stays an LP.
const RATIOS: &[(i32, i32)] = &[
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (1, 4),
    (4, 1),
];

/// Above this many LP columns the weighted generators are dropped and only the
/// plain DD set is searched, so the exact simplex stays tractable.
const MAX_LP_COLUMNS: usize = 1200;

/// A generator is a linear form over the monomial basis; the cone searched is
/// the set of non-negative combinations of their squares.
fn generators(n: usize) -> Vec<Vec<(usize, Rational)>> {
    let mut gens: Vec<Vec<(usize, Rational)>> = Vec::new();
    for i in 0..n {
        gens.push(vec![(i, Rational::from(1))]);
    }
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let full = n + pairs.len() * RATIOS.len() * 2;
    let ratios: &[(i32, i32)] = if full > MAX_LP_COLUMNS {
        &RATIOS[..1]
    } else {
        RATIOS
    };
    for &(i, j) in &pairs {
        for &(a, b) in ratios {
            for sign in [1, -1] {
                gens.push(vec![(i, Rational::from(a)), (j, Rational::from(sign * b))]);
            }
        }
    }
    gens
}

/// Search the cone spanned by the squares of the generator set for an exact
/// rational sum-of-squares decomposition of `p`.
///
/// Returns `None` when no non-negative combination reproduces `p`; that means
/// "not in this cone at this basis degree", never "not SOS" and never
/// "not non-negative".
pub fn dsos_search(p: &RatPoly, basis_deg: u32) -> Option<SosPoly> {
    let nvars = p.nvars();
    let basis = monomial_basis(nvars, basis_deg);
    let n = basis.len();
    if n == 0 {
        return if p.is_zero() {
            Some(SosPoly::default())
        } else {
            None
        };
    }

    let gens = generators(n);

    // Per target exponent, the (lp_var, coefficient) contributions of each
    // generator's square: (Σ_u c_u z_u)² contributes c_u·c_v to the monomial
    // z_u·z_v for every ordered pair (u, v).
    let mut acc: BTreeMap<Exponents, Vec<(usize, Rational)>> = BTreeMap::new();
    for (g_idx, g) in gens.iter().enumerate() {
        let mut local: BTreeMap<Exponents, Rational> = BTreeMap::new();
        for (u, cu) in g {
            for (v, cv) in g {
                let e = add_exp(&basis[*u], &basis[*v]);
                *local.entry(e).or_insert_with(|| Rational::from(0)) += cu.clone() * cv.clone();
            }
        }
        for (e, c) in local {
            if c != 0 {
                acc.entry(e).or_default().push((g_idx, c));
            }
        }
    }

    let mut all_exps: BTreeSet<Exponents> = acc.keys().cloned().collect();
    all_exps.extend(p.terms().keys().cloned());

    let mut lp = Lp::new(gens.len());
    for exp in &all_exps {
        let mut row = vec![Rational::from(0); gens.len()];
        if let Some(contribs) = acc.get(exp) {
            for (idx, c) in contribs {
                row[*idx] += c.clone();
            }
        }
        lp.constrain(row, Rel::Eq, p.coeff(exp));
    }
    // Minimise the total weight so the returned certificate is reasonably
    // sparse; correctness does not depend on this (any feasible point works).
    for k in 0..gens.len() {
        lp.set_objective(k, Rational::from(1));
    }

    let x = match lp.solve() {
        LpStatus::Optimal(x) => x,
        _ => return None,
    };

    let mut sos = SosPoly::default();
    for (g_idx, g) in gens.iter().enumerate() {
        let w = x[g_idx].clone();
        if w <= 0 {
            continue;
        }
        let mut square = RatPoly::zero(nvars);
        for (u, c) in g {
            square = square.add(&RatPoly::monomial(nvars, basis[*u].clone(), c.clone()));
        }
        sos.push(w, square);
    }
    Some(sos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    #[test]
    fn monomial_basis_univariate() {
        let b = monomial_basis(1, 2);
        assert_eq!(b, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn monomial_basis_bivariate_degree1() {
        let b = monomial_basis(2, 1);
        let mut expect = vec![vec![0, 0], vec![0, 1], vec![1, 0]];
        expect.sort();
        let mut got = b.clone();
        got.sort();
        assert_eq!(got, expect);
    }

    #[test]
    fn dsos_finds_perfect_square() {
        // p = x^2 + 2x + 1 = (x+1)^2, diagonally dominant trivially.
        let mut p = RatPoly::monomial(1, vec![2], Rational::from(1));
        p = p.add(&RatPoly::monomial(1, vec![1], Rational::from(2)));
        p = p.add(&RatPoly::constant(1, Rational::from(1)));
        let sos = dsos_search(&p, 1).expect("DSOS should find a certificate");
        assert_eq!(sos.to_poly(1), p);
    }

    #[test]
    fn dsos_finds_diagonal_sum() {
        // p = x^2 + y^2 (already diagonal).
        let p = RatPoly::monomial(2, vec![2, 0], Rational::from(1)).add(&RatPoly::monomial(
            2,
            vec![0, 2],
            Rational::from(1),
        ));
        let sos = dsos_search(&p, 1).expect("DSOS should find a certificate");
        assert_eq!(sos.to_poly(2), p);
    }

    #[test]
    fn dsos_refuses_unreachable_degree() {
        // p = x^4, but basis only goes up to degree 1 (so max reachable
        // quadratic-form degree is 2) — must refuse, not silently drop terms.
        let p = RatPoly::monomial(1, vec![4], Rational::from(1));
        assert!(dsos_search(&p, 1).is_none());
    }

    #[test]
    fn dsos_handles_off_diagonal_quadratic() {
        // p = x^2 - 2xy + 2y^2 = (x-y)^2 + y^2, DD-representable directly.
        let mut p = RatPoly::monomial(2, vec![2, 0], Rational::from(1));
        p = p.add(&RatPoly::monomial(2, vec![1, 1], Rational::from(-2)));
        p = p.add(&RatPoly::monomial(2, vec![0, 2], Rational::from(2)));
        let sos = dsos_search(&p, 1).expect("DSOS should find a certificate");
        assert_eq!(sos.to_poly(2), p);
    }

    #[test]
    fn dsos_rational_coefficients() {
        // p = (1/2 x + 1/3)^2 = 1/4 x^2 + 1/3 x + 1/9
        let mut p = RatPoly::monomial(1, vec![2], r(1, 4));
        p = p.add(&RatPoly::monomial(1, vec![1], r(1, 3)));
        p = p.add(&RatPoly::constant(1, r(1, 9)));
        let sos = dsos_search(&p, 1).expect("DSOS should find a certificate");
        assert_eq!(sos.to_poly(1), p);
    }
}
