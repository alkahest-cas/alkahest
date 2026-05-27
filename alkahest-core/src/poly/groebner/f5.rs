//! Faugère F5 — signature-based Gröbner basis over ℚ.
//!
//! Each basis element carries a module signature `m · e_i` (monomial `m`, `i`
//! the index of an original generator). Signatures are ordered by
//! **lexicographic order on `m`** then by generator index, as in V2-8.
//!
//! **Signature-bounded reduction**: a reducer `g` is only applied at shift `t`
//! if `t·sig(g) < sig_bound` (strict). This is the core F5 invariant — it
//! ensures a zero reduction genuinely witnesses a module syzygy and prevents
//! spurious cancellations from elements with higher signatures.
//!
//! **Criterion applied**: Buchberger product criterion (coprime leading
//! monomials → skip).  The divisibility-lifting syzygy criterion requires the
//! full LM-compatibility check (valid only for Koszul syzygies of the original
//! generators, not for basis-state-dependent zeros) and is intentionally
//! omitted; the sig-bounded reduction alone eliminates the bulk of redundant
//! zero reductions.

use crate::poly::groebner::buchberger::interreduce;
use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::reduce::s_polynomial;
use std::cmp::Ordering;

/// Module signature: monomial part (exponent vector) × original generator index.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Signature {
    exp: Vec<u32>,
    index: usize,
}

impl Signature {
    fn new(n_vars: usize, index: usize) -> Self {
        Signature {
            exp: vec![0u32; n_vars],
            index,
        }
    }

    fn mul_monomial(&self, shift: &[u32]) -> Signature {
        Signature {
            exp: self
                .exp
                .iter()
                .zip(shift.iter())
                .map(|(a, b)| a + b)
                .collect(),
            index: self.index,
        }
    }
}

/// Compare signatures: `Lex(exp) × index`.
fn cmp_signature(a: &Signature, b: &Signature) -> Ordering {
    match MonomialOrder::Lex.cmp(&a.exp, &b.exp) {
        Ordering::Equal => a.index.cmp(&b.index),
        o => o,
    }
}

fn lcm_exp(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(x, y)| (*x).max(*y)).collect()
}

fn exp_sub(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn divides_exp(u: &[u32], v: &[u32]) -> bool {
    u.len() == v.len() && u.iter().zip(v.iter()).all(|(a, b)| a <= b)
}

fn total_deg(exp: &[u32]) -> u32 {
    exp.iter().sum()
}

/// Labelled basis element used during F5.
#[derive(Debug, Clone)]
struct Labelled {
    sig: Signature,
    poly: GbPoly,
}

/// Buchberger product criterion on the polynomial parts.
fn product_criterion(f: &GbPoly, g: &GbPoly, order: MonomialOrder) -> bool {
    let lf = match f.leading_exp(order) {
        Some(e) => e,
        None => return true,
    };
    let lg = match g.leading_exp(order) {
        Some(e) => e,
        None => return true,
    };
    lf.iter().zip(lg.iter()).all(|(&a, &b)| a == 0 || b == 0)
}

/// Signature of the S-polynomial of two labelled parents.
fn s_pair_signature(f: &Labelled, g: &Labelled, order: MonomialOrder) -> Option<Signature> {
    let lf = f.poly.leading_exp(order)?;
    let lg = g.poly.leading_exp(order)?;
    let lcm = lcm_exp(&lf, &lg);
    let t_f = exp_sub(&lcm, &lf);
    let t_g = exp_sub(&lcm, &lg);
    let sf = f.sig.mul_monomial(&t_f);
    let sg = g.sig.mul_monomial(&t_g);
    Some(if cmp_signature(&sf, &sg) >= Ordering::Equal {
        sf
    } else {
        sg
    })
}

/// Signature-bounded polynomial reduction (F5 style).
///
/// Reduces `poly` using only basis elements `lb` where `shift·sig(lb) < sig_bound`
/// (strict). Stops immediately when no sig-compatible reducer covers the leading
/// term. Returns zero iff the S-polynomial is a module syzygy at `sig_bound`.
fn reduce_f5(
    poly: &GbPoly,
    sig_bound: &Signature,
    basis: &[Labelled],
    order: MonomialOrder,
) -> GbPoly {
    let mut h = poly.clone();
    'outer: loop {
        if h.is_zero() {
            break;
        }
        let lt_exp = match h.leading_exp(order) {
            Some(e) => e,
            None => break,
        };
        let lt_coeff = h.leading_coeff(order).unwrap();
        for lb in basis {
            let lb_lm = match lb.poly.leading_exp(order) {
                Some(e) => e,
                None => continue,
            };
            if !divides_exp(&lb_lm, &lt_exp) {
                continue;
            }
            let shift = exp_sub(&lt_exp, &lb_lm);
            let scaled_sig = lb.sig.mul_monomial(&shift);
            if cmp_signature(&scaled_sig, sig_bound) != Ordering::Less {
                continue;
            }
            let lb_lc = lb.poly.leading_coeff(order).unwrap();
            let factor = rug::Rational::from(&lt_coeff / &lb_lc);
            let subtrahend = lb.poly.mul_monomial(&shift, &factor);
            h = h.sub(&subtrahend);
            continue 'outer;
        }
        // No sig-compatible reducer for the leading term — stop.
        break;
    }
    h
}

/// S-pair heap element ordered by **signature degree then signature**
/// (smallest first, via inverted `BinaryHeap`).
#[derive(Debug, Clone)]
struct Pair {
    sig_s: Signature,
    /// Total degree of `sig_s.exp` — primary sort key.
    sig_deg: u32,
    i: usize,
    j: usize,
}

impl Pair {
    fn new(f: &Labelled, g: &Labelled, i: usize, j: usize, order: MonomialOrder) -> Option<Self> {
        let sig_s = s_pair_signature(f, g, order)?;
        let sig_deg = total_deg(&sig_s.exp);
        Some(Pair {
            sig_s,
            sig_deg,
            i,
            j,
        })
    }
}

impl Eq for Pair {}
impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.j == other.j
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .sig_deg
            .cmp(&self.sig_deg)
            .then_with(|| cmp_signature(&other.sig_s, &self.sig_s))
    }
}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute a Gröbner basis using Faugère F5 with signature-bounded reduction.
///
/// The output is interreduced and monic in the same sense as
/// [`super::buchberger::compute_buchberger_basis`].
pub fn compute_groebner_basis_f5(
    generators: Vec<GbPoly>,
    poly_order: MonomialOrder,
) -> Vec<GbPoly> {
    let mut basis: Vec<Labelled> = generators
        .into_iter()
        .enumerate()
        .filter(|(_, g)| !g.is_zero())
        .map(|(idx, g)| {
            let p = g.make_monic(poly_order);
            let n_vars = p.n_vars;
            Labelled {
                sig: Signature::new(n_vars, idx),
                poly: p,
            }
        })
        .collect();

    if basis.is_empty() {
        return vec![];
    }

    let mut heap: std::collections::BinaryHeap<Pair> = std::collections::BinaryHeap::new();

    for i in 0..basis.len() {
        for j in (i + 1)..basis.len() {
            if product_criterion(&basis[i].poly, &basis[j].poly, poly_order) {
                continue;
            }
            if let Some(pair) = Pair::new(&basis[i], &basis[j], i, j, poly_order) {
                heap.push(pair);
            }
        }
    }

    while let Some(pair) = heap.pop() {
        // Stale-pair check: signature may shift as basis grows.
        let sig_s = match s_pair_signature(&basis[pair.i], &basis[pair.j], poly_order) {
            Some(s) => s,
            None => continue,
        };
        if cmp_signature(&sig_s, &pair.sig_s) != Ordering::Equal {
            continue;
        }

        let s = s_polynomial(&basis[pair.i].poly, &basis[pair.j].poly, poly_order);
        let h = reduce_f5(&s, &sig_s, &basis, poly_order);
        if h.is_zero() {
            continue;
        }

        let new_entry = Labelled {
            sig: sig_s,
            poly: h.make_monic(poly_order),
        };
        let new_idx = basis.len();
        basis.push(new_entry);

        for k in 0..new_idx {
            if product_criterion(&basis[k].poly, &basis[new_idx].poly, poly_order) {
                continue;
            }
            if let Some(p) = Pair::new(&basis[k], &basis[new_idx], k, new_idx, poly_order) {
                heap.push(p);
            }
        }
    }

    let polys: Vec<GbPoly> = basis.into_iter().map(|lp| lp.poly).collect();
    interreduce(polys, poly_order)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::groebner::buchberger::compute_buchberger_basis;
    use crate::poly::groebner::reduce::reduce;
    use std::collections::BTreeMap;

    fn rat(n: i64) -> rug::Rational {
        rug::Rational::from(n)
    }

    fn poly(terms: &[(&[u32], i64)], n_vars: usize) -> GbPoly {
        GbPoly {
            terms: terms.iter().map(|(e, c)| (e.to_vec(), rat(*c))).collect(),
            n_vars,
        }
    }

    fn bases_equivalent(a: &[GbPoly], b: &[GbPoly], order: MonomialOrder) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for p in a {
            if !reduce(p, b, order).is_zero() {
                return false;
            }
        }
        for p in b {
            if !reduce(p, a, order).is_zero() {
                return false;
            }
        }
        true
    }

    /// Build the Cyclic-n benchmark system in `n` variables.
    pub(crate) fn cyclic_system(n: usize) -> Vec<GbPoly> {
        let mut polys = Vec::with_capacity(n);
        for k in 1..=n {
            let mut terms: BTreeMap<Vec<u32>, rug::Rational> = BTreeMap::new();
            for start in 0..n {
                let mut exp = vec![0u32; n];
                for d in 0..k {
                    exp[(start + d) % n] += 1;
                }
                let c = terms.entry(exp).or_insert_with(|| rug::Rational::from(0));
                *c += rug::Rational::from(1);
            }
            if k == n {
                let zero_exp = vec![0u32; n];
                let c = terms
                    .entry(zero_exp)
                    .or_insert_with(|| rug::Rational::from(0));
                *c -= rug::Rational::from(1);
            }
            terms.retain(|_, v| *v != 0);
            polys.push(GbPoly { terms, n_vars: n });
        }
        polys
    }

    #[test]
    fn f5_agrees_with_buchberger_small() {
        let systems: Vec<Vec<GbPoly>> = vec![
            vec![
                poly(&[(&[2], 1), (&[0], -1)], 1),
                poly(&[(&[1], 1), (&[0], -1)], 1),
            ],
            vec![poly(&[(&[1, 0], 1)], 2), poly(&[(&[0, 1], 1)], 2)],
            vec![
                poly(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)], 2),
                poly(&[(&[1, 0], 1), (&[0, 1], -1)], 2),
            ],
        ];

        for order in [MonomialOrder::Lex, MonomialOrder::GRevLex] {
            for sys in &systems {
                let b1 = compute_buchberger_basis(sys.clone(), order);
                let b2 = compute_groebner_basis_f5(sys.clone(), order);
                assert!(
                    bases_equivalent(&b1, &b2, order),
                    "F5 vs Buchberger mismatch for order {:?}",
                    order
                );
            }
        }
    }

    #[test]
    fn f5_member_checks_match() {
        let f = poly(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)], 2);
        let g = poly(&[(&[1, 0], 1), (&[0, 1], -1)], 2);
        let orig_f = f.clone();
        let orig_g = g.clone();
        let order = MonomialOrder::Lex;
        let b = compute_groebner_basis_f5(vec![f, g], order);
        assert!(reduce(&orig_f, &b, order).is_zero());
        assert!(reduce(&orig_g, &b, order).is_zero());
    }

    #[test]
    fn f5_agrees_cyclic4() {
        let order = MonomialOrder::GRevLex;
        let sys = cyclic_system(4);
        let b4 = compute_buchberger_basis(sys.clone(), order);
        let b5 = compute_groebner_basis_f5(sys, order);
        if !bases_equivalent(&b4, &b5, order) {
            eprintln!("Buchberger basis ({} elements):", b4.len());
            for p in &b4 {
                eprintln!("  LM={:?}", p.leading_exp(order));
            }
            eprintln!("F5 basis ({} elements):", b5.len());
            for p in &b5 {
                eprintln!("  LM={:?}", p.leading_exp(order));
            }
            panic!("F5 vs Buchberger mismatch on Cyclic-4");
        }
    }

    #[test]
    fn f5_circle_line_intersection() {
        let order = MonomialOrder::Lex;
        let sys = vec![
            poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)], 2),
            poly(&[(&[0, 1], 1), (&[1, 0], -1)], 2),
        ];
        let orig = sys.clone();
        let b = compute_groebner_basis_f5(sys, order);
        for p in &orig {
            assert!(reduce(p, &b, order).is_zero(), "generator not in F5 basis");
        }
    }
}
