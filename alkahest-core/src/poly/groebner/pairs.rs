//! Critical-pair management for Buchberger's algorithm.
//!
//! Everything here depends on *leading exponents only*, never on coefficients,
//! which is why the same code drives the basis over ℚ
//! ([`super::buchberger`]) and the one over `Q(params)`
//! ([`super::parametric`]).  That is not a refactoring convenience: it is the
//! reason a `Q(params)` basis specialises so cleanly.  Pair selection, the
//! Gebauer–Möller criteria and the product criterion all read the same
//! monomial data before and after a parameter substitution, so a specialisation
//! that keeps every leading monomial keeps the whole pair schedule too.
//!
//! Reference: Becker & Weispfenning (1993) "Gröbner Bases", Algorithm 6.5
//! (GROEBNERNEWS2), Gebauer & Möller (1988), and Giovini et al. (1991)
//! "One Sugar Cube, Please" for the sugar selection strategy.

#[inline]
pub(crate) fn lcm_exp(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect()
}

/// True if every component of `a` ≤ the corresponding component of `b`.
#[inline]
pub(crate) fn monomial_divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b.iter()).all(|(ai, bi)| ai <= bi)
}

/// Total degree of an exponent vector.
#[inline]
pub(crate) fn total_deg(e: &[u32]) -> u32 {
    e.iter().sum()
}

/// A critical pair, ordered for a min-heap by sugar degree then lcm degree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CriticalPair {
    /// Sugar degree of the pair: `lcm_deg + max(ecart_i, ecart_j)`.
    /// Primary sort key — the "sugar" selection strategy (Giovini et al. 1991).
    /// For homogeneous systems this equals `lcm_deg`; for inhomogeneous ones it
    /// avoids the late-sugar blowup that the normal strategy suffers.
    pub(crate) sugar_deg: u32,
    /// Total degree of `lcm(LM(basis[i]), LM(basis[j]))` — secondary sort key.
    pub(crate) lcm_deg: u32,
    pub(crate) lcm_exp: Vec<u32>,
    pub(crate) i: usize,
    pub(crate) j: usize,
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

/// Update the critical-pair list when basis element `new_idx` is added.
///
/// `lead[k]` is the leading exponent vector of basis element `k` and
/// `sugar[k]` its sugar (max total degree of any term).  Both are indexed in
/// parallel with the basis and must cover `new_idx`.
///
/// Applies:
/// - **Product criterion**: coprime leading monomials ⟹ the S-polynomial
///   reduces to zero, so the pair is never formed.
/// - **Criterion M**: among the new pairs `(g, h)`, keep only those whose lcm
///   is not strictly divisible by another candidate's lcm.
/// - **Criterion F**: discard old pairs `(g₁, g₂)` where `lm(h) | lcm(g₁, g₂)`
///   and the pair is truly covered (the two equality conditions from B&W §6.5).
pub(crate) fn update_pairs(
    lead: &[Vec<u32>],
    sugar: &[u32],
    pairs: &mut Vec<CriticalPair>,
    new_idx: usize,
) {
    let lh = &lead[new_idx];
    let lh_deg = total_deg(lh);
    let ecart_h = sugar[new_idx].saturating_sub(lh_deg);

    // -----------------------------------------------------------------------
    // Step 1: build candidate pairs (g, h), filtered by the product criterion.
    // -----------------------------------------------------------------------
    struct Cand {
        g_idx: usize,
        lcm: Vec<u32>,
        ecart_g: u32,
    }

    let candidates: Vec<Cand> = (0..new_idx)
        .filter_map(|g_idx| {
            let lg = &lead[g_idx];
            if lh.iter().zip(lg.iter()).all(|(&a, &b)| a == 0 || b == 0) {
                return None;
            }
            Some(Cand {
                g_idx,
                lcm: lcm_exp(lh, lg),
                ecart_g: sugar[g_idx].saturating_sub(total_deg(lg)),
            })
        })
        .collect();

    // -----------------------------------------------------------------------
    // Step 2: Criterion M — keep only minimal candidates.
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
    // The equality conditions prevent incorrectly discarding pairs whose
    // chain-criterion witness is itself degenerate (B&W §6.5).
    // -----------------------------------------------------------------------
    pairs.retain(|p| {
        let lg1 = &lead[p.i];
        let lg2 = &lead[p.j];
        let lcm_12 = lcm_exp(lg1, lg2);

        if !monomial_divides(lh, &lcm_12) {
            return true; // lm(h) doesn't divide — keep
        }
        if lcm_exp(lg1, lh) == lcm_12 {
            return true; // g1 is the witness — keep (pair is not truly covered)
        }
        if lcm_exp(lg2, lh) == lcm_12 {
            return true; // g2 is the witness — keep
        }
        false // discard: h truly subverts this pair
    });

    // -----------------------------------------------------------------------
    // Step 4: add the minimal candidates, with sugar degrees.
    // -----------------------------------------------------------------------
    for c in c_min {
        let lcm_deg = total_deg(&c.lcm);
        pairs.push(CriticalPair {
            sugar_deg: lcm_deg + c.ecart_g.max(ecart_h),
            lcm_deg,
            lcm_exp: c.lcm.clone(),
            i: c.g_idx,
            j: new_idx,
        });
    }
}
