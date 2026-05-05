//! Faugère's F5 — signature-based Gröbner basis computation over ℚ.
//!
//! This follows the labeled-polynomial presentation: each basis element carries
//! a module signature `m · e_i` (monomial `m`, `i` the index of an original
//! generator). Signatures are ordered by **lexicographic order on `m`** and
//! then by **generator index** (`i`), as required by V2-8.
//!
//! Polynomial leading terms use the caller-selected [`MonomialOrder`]; signature
//! monomials always compare under [`MonomialOrder::Lex`].

use crate::poly::groebner::f4::interreduce;
use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::reduce::s_polynomial;
use std::cmp::Ordering;

/// Module signature: monomial part (exponent vector) × original generator index.
#[derive(Debug, Clone)]
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
        let exp: Vec<u32> = self
            .exp
            .iter()
            .zip(shift.iter())
            .map(|(a, b)| a + b)
            .collect();
        Signature {
            exp,
            index: self.index,
        }
    }
}

/// Compare signatures: `Lex(exp) × index` (roadmap V2-8).
fn cmp_signature(a: &Signature, b: &Signature) -> Ordering {
    match MonomialOrder::Lex.cmp(&a.exp, &b.exp) {
        Ordering::Equal => a.index.cmp(&b.index),
        o => o,
    }
}

fn lcm_exp(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(x, y)| (*x).max(*y)).collect()
}

/// `a - b` when `a_i >= b_i` for all `i`.
fn exp_sub(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn divides_exp(u: &[u32], v: &[u32]) -> bool {
    u.len() == v.len() && u.iter().zip(v.iter()).all(|(a, b)| a <= b)
}

/// Labelled basis element used during F5.
#[derive(Debug, Clone)]
struct Labelled {
    sig: Signature,
    poly: GbPoly,
}

/// Buchberger product criterion on the polynomial parts.
fn product_criterion(f: &GbPoly, g: &GbPoly, poly_order: MonomialOrder) -> bool {
    let lf = match f.leading_exp(poly_order) {
        Some(e) => e,
        None => return true,
    };
    let lg = match g.leading_exp(poly_order) {
        Some(e) => e,
        None => return true,
    };
    lf.iter().zip(lg.iter()).all(|(&a, &b)| a == 0 || b == 0)
}

/// F5-style normal form: reduce `p` w.r.t. `basis`, only using reducers whose
/// multiplied signature does not exceed `bound` (signature-based Buchberger
/// reduction).
fn reduce_f5(
    bound: &Signature,
    mut p: GbPoly,
    basis: &[Labelled],
    poly_order: MonomialOrder,
) -> GbPoly {
    let n_vars = p.n_vars;
    let mut r = GbPoly::zero(n_vars);

    'outer: while !p.is_zero() {
        let (lt_exp, lt_coeff) = match p.leading_term(poly_order) {
            Some((e, c)) => (e.clone(), c.clone()),
            None => break,
        };

        for lp in basis {
            if let Some((lg_exp, lg_coeff)) = lp.poly.leading_term(poly_order) {
                if lt_exp.len() == lg_exp.len() && divides_exp(lg_exp, &lt_exp) {
                    let u = exp_sub(&lt_exp, lg_exp);
                    let sig_uh = lp.sig.mul_monomial(&u);
                    if cmp_signature(&sig_uh, bound) != Ordering::Greater {
                        let coeff = rug::Rational::from(&lt_coeff / lg_coeff);
                        let subtrahend = lp.poly.mul_monomial(&u, &coeff);
                        p = p.sub(&subtrahend);
                        continue 'outer;
                    }
                }
            }
        }

        // No admissible reducer — strip leading term into the remainder.
        let lt = GbPoly::monomial(lt_exp.clone(), lt_coeff);
        r = r.add(&lt);
        let mut p_terms = p.terms.clone();
        p_terms.remove(&lt_exp);
        p.terms = p_terms;
    }

    r
}

/// Signature of the combined head of an S-polynomial from two labeled parents.
fn s_pair_signature(f: &Labelled, g: &Labelled, poly_order: MonomialOrder) -> Option<Signature> {
    let lf = f.poly.leading_exp(poly_order)?;
    let lg = g.poly.leading_exp(poly_order)?;
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

fn compute_s_poly(f: &Labelled, g: &Labelled, poly_order: MonomialOrder) -> GbPoly {
    s_polynomial(&f.poly, &g.poly, poly_order)
}

/// A candidate S-pair, ordered for a min-heap on S-pair signature.
#[derive(Debug, Clone)]
struct Pair {
    sig_s: Signature,
    i: usize,
    j: usize,
}

impl Pair {
    fn new(
        f: &Labelled,
        g: &Labelled,
        i: usize,
        j: usize,
        poly_order: MonomialOrder,
    ) -> Option<Self> {
        let sig_s = s_pair_signature(f, g, poly_order)?;
        Some(Pair { sig_s, i, j })
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
        // Max-heap by signature → pop_smallest when using BinaryHeap::pop
        cmp_signature(&other.sig_s, &self.sig_s)
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute a Gröbner basis with the F5 labeled-polynomial strategy (signature
/// filtering during reduction). The output is interreduced and monic in the same
/// sense as [`super::f4::compute_groebner_basis`].
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
            if !product_criterion(&basis[i].poly, &basis[j].poly, poly_order) {
                if let Some(pair) = Pair::new(&basis[i], &basis[j], i, j, poly_order) {
                    heap.push(pair);
                }
            }
        }
    }

    while let Some(pair) = heap.pop() {
        let f = &basis[pair.i];
        let g = &basis[pair.j];
        if product_criterion(&f.poly, &g.poly, poly_order) {
            continue;
        }
        let sig_s = match s_pair_signature(f, g, poly_order) {
            Some(s) => s,
            None => continue,
        };
        // Skip if signature drifted from queue (stale pair) — rare; cheap check.
        if cmp_signature(&sig_s, &pair.sig_s) != Ordering::Equal {
            continue;
        }

        let s = compute_s_poly(f, g, poly_order);
        let h = reduce_f5(&sig_s, s, &basis, poly_order);
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
            if !product_criterion(&basis[k].poly, &basis[new_idx].poly, poly_order) {
                if let Some(p) = Pair::new(&basis[k], &basis[new_idx], k, new_idx, poly_order) {
                    heap.push(p);
                }
            }
        }
    }

    let polys: Vec<GbPoly> = basis.into_iter().map(|lp| lp.poly).collect();
    interreduce(polys, poly_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::groebner::f4::compute_groebner_basis;
    use crate::poly::groebner::reduce::reduce;

    fn rat(n: i64, d: i64) -> rug::Rational {
        rug::Rational::from((n, d))
    }

    fn poly(terms: &[(&[u32], i64)], n_vars: usize) -> GbPoly {
        GbPoly {
            terms: terms
                .iter()
                .map(|(e, c)| (e.to_vec(), rat(*c, 1)))
                .collect(),
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
                let b1 = compute_groebner_basis(sys.clone(), order);
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
}
