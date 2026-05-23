//! Buchberger's algorithm for Gröbner basis computation over ℚ.
//!
//! Implements the sequential Buchberger algorithm with:
//! - Gebauer-Möller criteria M and F to prune S-pairs
//! - Sugar selection strategy: process pair with minimum sugar degree first, break ties by lcm degree
//! - Incremental basis update: each new element is added before selecting the next pair
//!
//! Reference: Becker & Weispfenning (1993) "Gröbner Bases", Algorithm 6.5 (GROEBNERNEWS2),
//! Gebauer & Möller (1988) "On an Installation of Buchberger's Algorithm",
//! and Giovini et al. (1991) "One Sugar Cube, Please" for the sugar selection strategy.

use std::collections::BinaryHeap;

use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::reduce::{reduce, s_polynomial};

// ---------------------------------------------------------------------------
// Monomial helpers
// ---------------------------------------------------------------------------

#[inline]
fn lcm_exp(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect()
}

/// True if every component of `a` ≤ corresponding component of `b`.
#[inline]
fn monomial_divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b.iter()).all(|(ai, bi)| ai <= bi)
}

/// Total degree of an exponent vector.
#[inline]
fn total_deg(e: &[u32]) -> u32 {
    e.iter().sum()
}

// ---------------------------------------------------------------------------
// Critical pair with sugar-ordered comparison (min-heap)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
struct CriticalPair {
    /// Sugar degree of the pair: lcm_deg + max(ecart_i, ecart_j).
    /// Primary sort key — the "sugar" selection strategy (Giovini et al. 1991).
    /// For homogeneous systems this equals lcm_deg; for inhomogeneous ones it
    /// avoids the late-sugar blowup that the normal strategy suffers.
    sugar_deg: u32,
    /// Total degree of lcm(LM(basis[i]), LM(basis[j])) — secondary sort key.
    lcm_deg: u32,
    lcm_exp: Vec<u32>,
    i: usize,
    j: usize,
}

impl Ord for CriticalPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering so BinaryHeap (max-heap) acts as a min-heap.
        other
            .sugar_deg
            .cmp(&self.sugar_deg)
            .then_with(|| other.lcm_deg.cmp(&self.lcm_deg))
            .then_with(|| self.i.cmp(&other.i))
            .then_with(|| self.j.cmp(&other.j))
    }
}
impl PartialOrd for CriticalPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Gebauer-Möller pair update
// ---------------------------------------------------------------------------

/// Update the critical pair list when `basis[new_idx]` is added to the basis.
///
/// Applies:
/// - **Criterion M**: Among new pairs (g, h), keep only those whose lcm is
///   not strictly divisible by the lcm of another candidate pair.
/// - **Criterion F**: Discard old pairs (g1, g2) where lm(h) | lcm(lm(g1), lm(g2))
///   and the pair is truly covered (the two equality conditions from B&W §6.5).
///
/// `basis_sugar[k]` = max total degree of any term in `basis[k]` (the sugar).
fn update_pairs(
    basis: &[GbPoly],
    basis_sugar: &[u32],
    pairs: &mut Vec<CriticalPair>,
    new_idx: usize,
    order: MonomialOrder,
) {
    let lh = match basis[new_idx].leading_exp(order) {
        Some(e) => e,
        None => return,
    };
    let lh_deg = total_deg(&lh);
    let ecart_h = basis_sugar[new_idx].saturating_sub(lh_deg);

    // -----------------------------------------------------------------------
    // Step 1: build candidate pairs (g, h), filtered by product criterion.
    // -----------------------------------------------------------------------
    struct Cand {
        g_idx: usize,
        lcm: Vec<u32>,
        ecart_g: u32,
    }

    let candidates: Vec<Cand> = (0..new_idx)
        .filter_map(|g_idx| {
            let lg = basis[g_idx].leading_exp(order)?;
            // Product criterion: coprime LMs ⟹ S-poly = 0, skip.
            if lh.iter().zip(lg.iter()).all(|(&a, &b)| a == 0 || b == 0) {
                return None;
            }
            let ecart_g = basis_sugar[g_idx].saturating_sub(total_deg(&lg));
            Some(Cand {
                g_idx,
                lcm: lcm_exp(&lh, &lg),
                ecart_g,
            })
        })
        .collect();

    // -----------------------------------------------------------------------
    // Step 2: Criterion M — keep only minimal candidates.
    // Discard (g, h) if ∃ (g', h) ∈ candidates with g' ≠ g and
    //   lcm(g', h) strictly divides lcm(g, h).
    // -----------------------------------------------------------------------
    let c_min: Vec<&Cand> = candidates
        .iter()
        .filter(|ci| {
            !candidates.iter().any(|cj| {
                cj.g_idx != ci.g_idx && monomial_divides(&cj.lcm, &ci.lcm) && cj.lcm != ci.lcm
            })
        })
        .collect();

    // -----------------------------------------------------------------------
    // Step 3: Criterion F — remove old pairs subsumed by h.
    // Discard (g1, g2) ∈ pairs if:
    //   lm(h) | lcm(g1, g2)
    //   AND lcm(g1, h) ≠ lcm(g1, g2)    [g1 is not the "cover witness"]
    //   AND lcm(g2, h) ≠ lcm(g1, g2)    [g2 is not the "cover witness"]
    // The equality conditions prevent incorrectly discarding pairs whose
    // chain-criterion witness is itself degenerate (B&W §6.5).
    // -----------------------------------------------------------------------
    pairs.retain(|p| {
        let lg1 = match basis[p.i].leading_exp(order) {
            Some(e) => e,
            None => return false,
        };
        let lg2 = match basis[p.j].leading_exp(order) {
            Some(e) => e,
            None => return false,
        };
        let lcm_12 = lcm_exp(&lg1, &lg2);

        if !monomial_divides(&lh, &lcm_12) {
            return true; // lm(h) doesn't divide — keep
        }
        if lcm_exp(&lg1, &lh) == lcm_12 {
            return true; // g1 is the witness — keep (pair is not truly covered)
        }
        if lcm_exp(&lg2, &lh) == lcm_12 {
            return true; // g2 is the witness — keep
        }
        false // discard: h truly subverts this pair
    });

    // -----------------------------------------------------------------------
    // Step 4: add minimal candidates to the pair list with sugar degrees.
    // Sugar of pair (g, h) with lcm L = deg(L) + max(ecart(g), ecart(h)).
    // -----------------------------------------------------------------------
    for c in c_min {
        let lcm_deg = total_deg(&c.lcm);
        let sugar_deg = lcm_deg + c.ecart_g.max(ecart_h);
        pairs.push(CriticalPair {
            sugar_deg,
            lcm_deg,
            lcm_exp: c.lcm.clone(),
            i: c.g_idx,
            j: new_idx,
        });
    }
}

// ---------------------------------------------------------------------------
// Main algorithm
// ---------------------------------------------------------------------------

/// Compute a Gröbner basis for the ideal generated by `generators` under `order`.
///
/// Uses sequential Buchberger with Gebauer-Möller pair management and
/// normal selection (process the pair with minimum lcm degree first).
pub fn compute_buchberger_basis(generators: Vec<GbPoly>, order: MonomialOrder) -> Vec<GbPoly> {
    let initial: Vec<GbPoly> = generators
        .into_iter()
        .filter(|g| !g.is_zero())
        .map(|g| g.make_monic(order))
        .collect();

    if initial.is_empty() {
        return initial;
    }

    let mut basis: Vec<GbPoly> = Vec::with_capacity(initial.len() * 2);
    let mut basis_sugar: Vec<u32> = Vec::with_capacity(initial.len() * 2);
    let mut pair_vec: Vec<CriticalPair> = Vec::new();

    // Add initial generators one by one, applying GM update after each.
    for gen in initial {
        let sugar = gen.sugar();
        let new_idx = basis.len();
        basis.push(gen);
        basis_sugar.push(sugar);
        update_pairs(&basis, &basis_sugar, &mut pair_vec, new_idx, order);
    }

    // Build min-heap (CriticalPair::Ord is reversed for min-heap behaviour).
    let mut heap: BinaryHeap<CriticalPair> = BinaryHeap::from(pair_vec);

    while let Some(pair) = heap.pop() {
        let sp = s_polynomial(&basis[pair.i], &basis[pair.j], order);
        let r = reduce(&sp, &basis, order);

        if !r.is_zero() {
            let r = r.make_monic(order);
            let sugar = r.sugar();
            let new_idx = basis.len();
            basis.push(r);
            basis_sugar.push(sugar);

            // Flatten heap → apply GM update → rebuild heap.
            let mut pv: Vec<CriticalPair> = heap.into_vec();
            update_pairs(&basis, &basis_sugar, &mut pv, new_idx, order);
            heap = BinaryHeap::from(pv);
        }
    }

    interreduce(basis, order)
}

// ---------------------------------------------------------------------------
// Interreduction
// ---------------------------------------------------------------------------

/// Interreduce a Gröbner basis: reduce each element by all others and remove
/// elements whose leading term is divisible by another's.
pub(crate) fn interreduce(mut basis: Vec<GbPoly>, order: MonomialOrder) -> Vec<GbPoly> {
    let mut i = 0;
    while i < basis.len() {
        let others: Vec<GbPoly> = basis
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, g)| g.clone())
            .collect();
        let reduced = reduce(&basis[i], &others, order);
        if reduced.is_zero() {
            basis.remove(i);
        } else {
            basis[i] = reduced.make_monic(order);
            i += 1;
        }
    }
    basis
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64, d: i64) -> rug::Rational {
        rug::Rational::from((n, d))
    }

    fn poly(terms: &[(&[u32], i64)]) -> GbPoly {
        let n_vars = terms.first().map(|(e, _)| e.len()).unwrap_or(1);
        GbPoly {
            terms: terms
                .iter()
                .map(|(e, c)| (e.to_vec(), rat(*c, 1)))
                .collect(),
            n_vars,
        }
    }

    #[test]
    fn groebner_x_squared_minus_1_and_x_minus_1() {
        let f = poly(&[(&[2], 1), (&[0], -1)]);
        let g = poly(&[(&[1], 1), (&[0], -1)]);
        let basis = compute_buchberger_basis(vec![f, g], MonomialOrder::Lex);
        assert_eq!(basis.len(), 1);
        assert!(basis[0].terms.contains_key(&vec![1]));
    }

    #[test]
    fn groebner_trivial_ideal() {
        let f = poly(&[(&[1, 0], 1)]);
        let g = poly(&[(&[0, 1], 1)]);
        let basis = compute_buchberger_basis(vec![f, g], MonomialOrder::GRevLex);
        assert_eq!(basis.len(), 2);
    }

    #[test]
    fn groebner_linear_system() {
        let f = poly(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]);
        let g = poly(&[(&[1, 0], 1), (&[0, 1], -1)]);
        let basis = compute_buchberger_basis(vec![f, g], MonomialOrder::Lex);
        assert!(!basis.is_empty());
        let orig_f = poly(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]);
        let orig_g = poly(&[(&[1, 0], 1), (&[0, 1], -1)]);
        let r_f = reduce(&orig_f, &basis, MonomialOrder::Lex);
        let r_g = reduce(&orig_g, &basis, MonomialOrder::Lex);
        assert!(r_f.is_zero(), "f does not reduce to 0 mod basis: {:?}", r_f);
        assert!(r_g.is_zero(), "g does not reduce to 0 mod basis: {:?}", r_g);
    }

    #[test]
    fn contains_check() {
        let f = poly(&[(&[1, 0], 1)]);
        let basis = compute_buchberger_basis(vec![f], MonomialOrder::Lex);
        let zero = GbPoly::zero(2);
        let r = reduce(&zero, &basis, MonomialOrder::Lex);
        assert!(r.is_zero());
    }

    #[test]
    fn circle_parabola_grevlex() {
        // x^2 + y^2 - 4, y - x^2 + 1
        let x2_y2_m4 = poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -4)]);
        let y_mx2_p1 = poly(&[(&[0, 1], 1), (&[2, 0], -1), (&[0, 0], 1)]);
        let basis =
            compute_buchberger_basis(vec![x2_y2_m4, y_mx2_p1.clone()], MonomialOrder::GRevLex);
        // Verify generators reduce to 0
        assert!(!basis.is_empty());
        let r = reduce(&y_mx2_p1, &basis, MonomialOrder::GRevLex);
        assert!(r.is_zero());
    }
}
