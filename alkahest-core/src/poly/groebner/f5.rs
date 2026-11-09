//! Signature-based Gröbner basis over ℚ, in the F5 family.
//!
//! Each basis element `g` carries a signature `sig(g) = t·e_i`: the leading
//! term of some module representation `g = Σ h_k f_k` (input `f_k`).
//! Signatures are compared **position over term** — first by the index `i`,
//! then by the monomial `t` under the *same* order as the polynomials — so the
//! inputs are processed incrementally, as in Faugère's F5 (2002).  The
//! algorithm is the "rewrite basis" formulation of Eder & Faugère, *A survey
//! on signature-based algorithms for computing Gröbner bases*, JSC 80 (2017),
//! §7, with:
//!
//! * **Regular reduction** — a term is reduced by `u·g` only when
//!   `u·sig(g) < T`, the signature of the element being reduced, so the
//!   result keeps signature `T`.  A result whose leading term is still
//!   reducible by some `u·g` with `u·sig(g) = T` (singular top-reducible) is
//!   redundant and dropped.
//! * **Syzygy criterion** — an S-pair whose signature is divisible by the
//!   signature of a known syzygy is skipped.  Known syzygies are the principal
//!   (Koszul) ones, `lm(g)·e_i` for every `g` of index below `i`, and every
//!   signature at which a reduction reached zero.
//! * **Rewrite criterion** — of the elements `g` whose signature divides an
//!   S-pair signature `T`, the one with the smallest `lm((T / sig(g))·g)`
//!   (the signature-ratio order of Arri–Perry / Gao–Volny–Wang / Roune–Stillman)
//!   is the only one allowed to generate `T`.  This must be that order, not
//!   "the most recently added": an S-pair whose reduction is singular
//!   top-reducible is dropped, and the argument for that drop is that the
//!   element which singularly reduces it, having a smaller `lm` at `T`, wins
//!   `T` instead.  Under add order a later element keeps `T`, its multiple can
//!   stay regular top-reducible, and every S-pair at `T` is discarded with
//!   nothing left to represent it — which lost basis elements on inhomogeneous
//!   input (`⟨1 + y² + x·y²·z + x²·z², 1 + x²·y², y + y·z⟩` under grevlex).
//!
//! Every S-pair the criteria do not discard is a genuine reduction step of
//! the signature-based theory.  The final basis is interreduced and monic,
//! like [`super::buchberger::compute_buchberger_basis`]'s, and **verified
//! exactly** before it is returned: every generator and every critical pair
//! left by Gebauer–Möller reduces to zero modulo it.  A basis that fails is
//! not returned; Buchberger's algorithm answers instead.  Every element is
//! built from the input by S-polynomials and reductions, so it lies in the
//! input ideal, and the verification gives the other inclusion: the result is
//! a proof, independently of the correctness of the criteria above.
//!
//! This engine works in exact rational arithmetic.  It is correct and avoids
//! most zero reductions, but it is **not** the fast path:
//! [`super::GroebnerBasis::compute`] under grevlex is multimodular (F4 modulo
//! word-size primes, then lifted and verified) and is far faster on anything
//! beyond toy systems.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use rug::Rational;

use crate::poly::groebner::buchberger::{compute_buchberger_basis, interreduce};
use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::multimodular::verify_over_q;

/// `t·e_index`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Sig {
    index: usize,
    mon: Vec<u32>,
}

fn mul_exp(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| crate::poly::exponent::add_or_panic(x, y))
        .collect()
}

fn divides(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

fn quo(b: &[u32], a: &[u32]) -> Vec<u32> {
    b.iter().zip(a).map(|(x, y)| x - y).collect()
}

/// An order-preserving key for an exponent vector: comparing keys
/// lexicographically is comparing monomials under `order`.
fn key(order: MonomialOrder, e: &[u32]) -> Vec<u32> {
    match order {
        MonomialOrder::Lex => e.to_vec(),
        MonomialOrder::GrLex => {
            let mut k = Vec::with_capacity(e.len() + 1);
            k.push(crate::poly::exponent::total_degree_or_panic(e));
            k.extend_from_slice(e);
            k
        }
        MonomialOrder::GRevLex => {
            let mut k = Vec::with_capacity(e.len() + 1);
            k.push(crate::poly::exponent::total_degree_or_panic(e));
            k.extend(e.iter().rev().map(|&x| u32::MAX - x));
            k
        }
    }
}

fn unkey(order: MonomialOrder, k: &[u32]) -> Vec<u32> {
    match order {
        MonomialOrder::Lex => k.to_vec(),
        MonomialOrder::GrLex => k[1..].to_vec(),
        MonomialOrder::GRevLex => k[1..].iter().rev().map(|&x| u32::MAX - x).collect(),
    }
}

/// A basis element: monic polynomial (terms in decreasing order) and its
/// signature.
struct Elem {
    sig: Sig,
    lead: Vec<u32>,
    /// Terms in decreasing order; `terms[0]` is the leading term, coefficient 1.
    terms: Vec<(Vec<u32>, Rational)>,
}

/// A critical pair, keyed by its signature `T = t·sig(g_k)`, the larger of
/// the two halves' signatures; `g_k` is the half that carries it.
struct SPair {
    t_key: Vec<u32>,
    t: Vec<u32>,
    index: usize,
    k: usize,
    j: usize,
}

impl PartialEq for SPair {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for SPair {}
impl Ord for SPair {
    /// Reversed (a min-heap on the signature), ties broken by the newest
    /// generator first so the rewrite criterion sees it first.
    fn cmp(&self, o: &Self) -> Ordering {
        o.index
            .cmp(&self.index)
            .then_with(|| o.t_key.cmp(&self.t_key))
            .then_with(|| self.k.cmp(&o.k))
            .then_with(|| self.j.cmp(&o.j))
    }
}
impl PartialOrd for SPair {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

struct State {
    order: MonomialOrder,
    basis: Vec<Elem>,
    /// Known syzygy signatures, per index.
    syz: Vec<Vec<Vec<u32>>>,
    pairs: BinaryHeap<SPair>,
}

impl State {
    fn sig_lt(&self, a: &Sig, b: &Sig) -> bool {
        match a.index.cmp(&b.index) {
            Ordering::Equal => self.order.cmp(&a.mon, &b.mon) == Ordering::Less,
            o => o == Ordering::Less,
        }
    }

    fn syzygy_covers(&self, index: usize, t: &[u32]) -> bool {
        self.syz[index].iter().any(|s| divides(s, t))
    }

    /// The element allowed to generate the signature `t·e_index`: among those
    /// whose signature divides it, the one whose multiple at `t` has the
    /// smallest leading monomial (the newest of equals).  See the module
    /// documentation for why it has to be this order.
    fn rewriter(&self, index: usize, t: &[u32]) -> Option<usize> {
        let mut best: Option<(Vec<u32>, usize)> = None;
        for (m, g) in self.basis.iter().enumerate() {
            if g.sig.index != index || !divides(&g.sig.mon, t) {
                continue;
            }
            let lm = key(self.order, &mul_exp(&quo(t, &g.sig.mon), &g.lead));
            // `<=`: the newest of equals, as `m` is increasing.
            if best.as_ref().is_none_or(|(b, _)| lm <= *b) {
                best = Some((lm, m));
            }
        }
        best.map(|(_, m)| m)
    }

    /// Regular reduction of `p` (signature `sig`), top and tail.  `None` when
    /// the result is singular top-reducible (redundant); `Some(empty)` for zero.
    fn regular_reduce(
        &self,
        p: BTreeMap<Vec<u32>, Rational>,
        sig: &Sig,
    ) -> Option<Vec<(Vec<u32>, Rational)>> {
        let order = self.order;
        let mut p = p;
        let mut out: Vec<(Vec<u32>, Rational)> = Vec::new();
        let mut ticks = 0u32;
        while let Some((k, c)) = p.pop_last() {
            if crate::poly::groebner::reduce::poll_due(&mut ticks) && crate::budget::interrupted() {
                return Some(Vec::new());
            }
            let m = unkey(order, &k);
            let mut reducer = None;
            let mut singular = false;
            for g in &self.basis {
                if !divides(&g.lead, &m) {
                    continue;
                }
                let u = quo(&m, &g.lead);
                let s = Sig {
                    index: g.sig.index,
                    mon: mul_exp(&u, &g.sig.mon),
                };
                if self.sig_lt(&s, sig) {
                    reducer = Some((g, u));
                    break;
                }
                if s == *sig {
                    singular = true;
                }
            }
            match reducer {
                Some((g, u)) => {
                    for (e, gc) in &g.terms[1..] {
                        let kk = key(order, &mul_exp(e, &u));
                        let d = Rational::from(&c * gc);
                        match p.get_mut(&kk) {
                            Some(v) => {
                                *v -= d;
                                if *v == 0 {
                                    p.remove(&kk);
                                }
                            }
                            None => {
                                p.insert(kk, -d);
                            }
                        }
                    }
                }
                None => {
                    if out.is_empty() && singular {
                        return None;
                    }
                    out.push((m, c));
                }
            }
        }
        Some(out)
    }

    /// Add a reduced, non-zero element with signature `sig`.
    fn add(&mut self, mut terms: Vec<(Vec<u32>, Rational)>, sig: Sig) {
        let lc = terms[0].1.clone();
        if lc != 1 {
            for (_, c) in &mut terms {
                *c /= &lc;
            }
        }
        let lead = terms[0].0.clone();
        let new = self.basis.len();
        // Principal syzygies with elements of the same index.
        for g in &self.basis {
            if g.sig.index != sig.index {
                continue;
            }
            let a = mul_exp(&g.lead, &sig.mon);
            let b = mul_exp(&lead, &g.sig.mon);
            match self.order.cmp(&a, &b) {
                Ordering::Greater => self.syz[sig.index].push(a),
                Ordering::Less => self.syz[sig.index].push(b),
                Ordering::Equal => {}
            }
        }
        // S-pairs with every element.
        for (j, g) in self.basis.iter().enumerate() {
            let l: Vec<u32> = lead.iter().zip(&g.lead).map(|(a, b)| *a.max(b)).collect();
            let sn = Sig {
                index: sig.index,
                mon: mul_exp(&quo(&l, &lead), &sig.mon),
            };
            let sg = Sig {
                index: g.sig.index,
                mon: mul_exp(&quo(&l, &g.lead), &g.sig.mon),
            };
            let (t, k, other) = if self.sig_lt(&sg, &sn) {
                (sn, new, j)
            } else if self.sig_lt(&sn, &sg) {
                (sg, j, new)
            } else {
                continue; // singular pair
            };
            self.pairs.push(SPair {
                t_key: key(self.order, &t.mon),
                t: t.mon,
                index: t.index,
                k,
                j: other,
            });
        }
        self.basis.push(Elem { sig, lead, terms });
    }

    fn s_poly(&self, t: &[u32], k: usize, j: usize) -> BTreeMap<Vec<u32>, Rational> {
        let (gk, gj) = (&self.basis[k], &self.basis[j]);
        let u_k = quo(t, &gk.sig.mon);
        let l = mul_exp(&u_k, &gk.lead);
        let u_j = quo(&l, &gj.lead);
        let mut p = BTreeMap::new();
        for (e, c) in &gk.terms[1..] {
            p.insert(key(self.order, &mul_exp(e, &u_k)), c.clone());
        }
        for (e, c) in &gj.terms[1..] {
            let kk = key(self.order, &mul_exp(e, &u_j));
            match p.get_mut(&kk) {
                Some(v) => {
                    *v -= c;
                    if *v == 0 {
                        p.remove(&kk);
                    }
                }
                None => {
                    p.insert(kk, Rational::from(-c));
                }
            }
        }
        p
    }
}

/// Compute a Gröbner basis with the signature-based algorithm described in
/// the module documentation.  The output is interreduced and monic in the
/// same sense as [`super::buchberger::compute_buchberger_basis`], and has been
/// verified to be a reduced Gröbner basis of the input ideal.
pub fn compute_groebner_basis_f5(
    generators: Vec<GbPoly>,
    poly_order: MonomialOrder,
) -> Vec<GbPoly> {
    let gens: Vec<GbPoly> = generators.into_iter().filter(|g| !g.is_zero()).collect();
    if gens.is_empty() {
        return Vec::new();
    }
    let basis = signature_basis(&gens, poly_order);
    if crate::budget::interrupt_pending() {
        return basis;
    }
    match verify_over_q(&gens, &basis, poly_order) {
        Some(true) if is_reduced(&basis, poly_order) => basis,
        // Stopped by an armed budget: the caller discards this.
        None => basis,
        // The criteria dropped something they should not have.  Never return
        // it: Buchberger needs no criteria beyond Gebauer–Möller's.
        Some(_) => compute_buchberger_basis(gens, poly_order),
    }
}

/// The signature-based computation proper, unverified.  `gens` is non-empty
/// and has no zero element.
fn signature_basis(gens: &[GbPoly], poly_order: MonomialOrder) -> Vec<GbPoly> {
    let n_vars = gens[0].n_vars;
    let mut st = State {
        order: poly_order,
        basis: Vec::new(),
        syz: vec![Vec::new(); gens.len()],
        pairs: BinaryHeap::new(),
    };
    for (i, f) in gens.iter().enumerate() {
        if crate::budget::interrupted() {
            break;
        }
        // Principal syzygies lm(g)·e_i for every element of lower index.
        let koszul: Vec<Vec<u32>> = st.basis.iter().map(|g| g.lead.clone()).collect();
        st.syz[i].extend(koszul);
        let sig = Sig {
            index: i,
            mon: vec![0; n_vars],
        };
        let p: BTreeMap<Vec<u32>, Rational> = f
            .terms
            .iter()
            .filter(|(_, c)| **c != 0)
            .map(|(e, c)| (key(poly_order, e), c.clone()))
            .collect();
        match st.regular_reduce(p, &sig) {
            Some(r) if r.is_empty() => st.syz[i].push(sig.mon),
            Some(r) => st.add(r, sig),
            None => {}
        }
        while let Some(first) = st.pairs.pop() {
            if crate::budget::interrupted() {
                break;
            }
            // Every pair of this signature, of which at most one is reduced.
            let mut pairs_at_t = vec![first];
            while st
                .pairs
                .peek()
                .is_some_and(|p| p.index == pairs_at_t[0].index && p.t == pairs_at_t[0].t)
            {
                pairs_at_t.extend(st.pairs.pop());
            }
            let (index, t) = (pairs_at_t[0].index, pairs_at_t[0].t.clone());
            if st.syzygy_covers(index, &t) {
                continue;
            }
            let Some(rewriter) = st.rewriter(index, &t) else {
                continue;
            };
            let Some(pair) = pairs_at_t.into_iter().find(|p| p.k == rewriter) else {
                continue;
            };
            let sig = Sig {
                index: pair.index,
                mon: pair.t.clone(),
            };
            let p = st.s_poly(&pair.t, pair.k, pair.j);
            match st.regular_reduce(p, &sig) {
                Some(r) if r.is_empty() => st.syz[pair.index].push(pair.t),
                Some(r) => st.add(r, sig),
                None => {}
            }
        }
    }
    let polys: Vec<GbPoly> = st
        .basis
        .into_iter()
        .map(|g| GbPoly {
            terms: g.terms.into_iter().collect(),
            n_vars,
        })
        .collect();
    if crate::budget::interrupt_pending() {
        return polys;
    }
    interreduce(polys, poly_order)
}

/// No term of any element is divisible by the leading monomial of another.
fn is_reduced(basis: &[GbPoly], order: MonomialOrder) -> bool {
    let leads: Vec<Vec<u32>> = basis.iter().filter_map(|g| g.leading_exp(order)).collect();
    basis.iter().enumerate().all(|(i, g)| {
        g.terms.keys().all(|e| {
            leads
                .iter()
                .enumerate()
                .all(|(j, l)| j == i || !divides(l, e))
        })
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
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

    fn canon(b: &[GbPoly]) -> Vec<Vec<(Vec<u32>, rug::Rational)>> {
        let mut v: Vec<Vec<(Vec<u32>, rug::Rational)>> = b
            .iter()
            .map(|g| {
                g.terms
                    .iter()
                    .map(|(e, c)| (e.clone(), c.clone()))
                    .collect()
            })
            .collect();
        v.sort();
        v
    }

    /// Same reduced basis as Buchberger, term for term, under every order.
    #[test]
    fn f5_reduced_basis_equals_buchberger() {
        let ks = crate::poly::groebner::multimodular::tests::katsura_system;
        let mut cases: Vec<(String, Vec<GbPoly>, Vec<MonomialOrder>)> = vec![];
        let all = vec![
            MonomialOrder::Lex,
            MonomialOrder::GrLex,
            MonomialOrder::GRevLex,
        ];
        for n in 2..=4 {
            cases.push((format!("cyclic-{n}"), cyclic_system(n), all.clone()));
        }
        for n in 1..=3 {
            cases.push((format!("katsura-{n}"), ks(n), all.clone()));
        }
        cases.push((
            "cyclic-5".into(),
            cyclic_system(5),
            vec![MonomialOrder::GRevLex],
        ));
        cases.push(("katsura-5".into(), ks(5), vec![MonomialOrder::GRevLex]));
        for (name, sys, orders) in cases {
            for order in orders {
                let b1 = compute_buchberger_basis(sys.clone(), order);
                let b2 = compute_groebner_basis_f5(sys.clone(), order);
                assert_eq!(canon(&b1), canon(&b2), "{name} {order:?}");
            }
        }
    }

    /// `⟨1 + y² + x·y²·z + x²·z², 1 + x²·y², y + y·z⟩`: the input that exposed
    /// the rewrite criterion's order.  Under grevlex the S-pair of `z + 1` and
    /// `x³z² + …` was discarded, in favour of a newer element whose multiple
    /// was itself still regular top-reducible; the basis lost `x³ + …` and
    /// disagreed with Buchberger, SymPy and Wolfram.
    #[test]
    fn f5_keeps_element_the_add_order_rewrite_lost() {
        let sys = vec![
            poly(
                &[
                    (&[0, 2, 0], 1),
                    (&[1, 2, 1], 1),
                    (&[2, 0, 2], 1),
                    (&[0, 0, 0], 1),
                ],
                3,
            ),
            poly(&[(&[2, 2, 0], 1), (&[0, 0, 0], 1)], 3),
            poly(&[(&[0, 1, 0], 1), (&[0, 1, 1], 1)], 3),
        ];
        for order in [
            MonomialOrder::Lex,
            MonomialOrder::GrLex,
            MonomialOrder::GRevLex,
        ] {
            let b1 = compute_buchberger_basis(sys.clone(), order);
            // The engine itself, so the verification cannot hide a criterion
            // that loses something: it would only cost the Buchberger fallback.
            let raw = signature_basis(&sys, order);
            assert_eq!(canon(&b1), canon(&raw), "{order:?}, unverified");
            let b2 = compute_groebner_basis_f5(sys.clone(), order);
            assert_eq!(canon(&b1), canon(&b2), "{order:?}");
        }
        // The hand-checked grevlex basis: z + 1 and three polynomials in x, y.
        let b = compute_groebner_basis_f5(sys, MonomialOrder::GRevLex);
        assert_eq!(b.len(), 4);
        let leads: Vec<Vec<u32>> = b
            .iter()
            .filter_map(|g| g.leading_exp(MonomialOrder::GRevLex))
            .collect();
        for l in [vec![0, 4, 0], vec![3, 0, 0], vec![1, 2, 0], vec![0, 0, 1]] {
            assert!(leads.contains(&l), "missing leading monomial {l:?}");
        }
    }

    /// The unverified engine against Buchberger on many random ideals, far
    /// harder than the proptest below: `F5_FUZZ_CASES` (default 5000) ideals
    /// from the seed `F5_FUZZ_SEED`, each under a wall budget.  Run with
    /// `cargo test --release --features groebner -p alkahest-cas fuzz_f5 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn fuzz_f5_unverified_against_buchberger() {
        use crate::budget::{enter, Budget};
        use crate::poly::groebner::armed;
        use std::time::Duration;

        let env = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let cases = env("F5_FUZZ_CASES", 5000);
        let mut x = env("F5_FUZZ_SEED", 1).max(1);
        let mut next = move |m: u64| {
            // xorshift64
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        let orders = [
            MonomialOrder::Lex,
            MonomialOrder::GrLex,
            MonomialOrder::GRevLex,
        ];
        let mut stats = [0u64; 3]; // agreed, a side over budget, zero input
        for _ in 0..cases {
            let n = 2 + next(3) as usize;
            let m = 2 + next(3) as usize;
            let max_e = 1 + next(3) as u32;
            let mut gens = Vec::with_capacity(m);
            for _ in 0..m {
                let mut p = GbPoly::zero(n);
                for _ in 0..1 + next(5) {
                    let e: Vec<u32> = (0..n).map(|_| next(u64::from(max_e) + 1) as u32).collect();
                    let c = rug::Rational::from((next(13) as i64 - 6, 1 + next(3) as i64));
                    p = p.add(&GbPoly::monomial(e, c));
                }
                if !p.is_zero() {
                    gens.push(p);
                }
            }
            if gens.len() < 2 {
                stats[2] += 1;
                continue;
            }
            let order = orders[next(3) as usize];
            let limit = Budget::new().with_wall(Duration::from_secs(3));
            let run = |g: Vec<GbPoly>, raw: bool| {
                let _b = enter(limit);
                armed(|| {
                    if raw {
                        signature_basis(&g, order)
                    } else {
                        compute_buchberger_basis(g, order)
                    }
                })
                .ok()
            };
            let (Some(a), Some(b)) = (run(gens.clone(), true), run(gens.clone(), false)) else {
                stats[1] += 1;
                continue;
            };
            assert_eq!(canon(&a), canon(&b), "{order:?} {gens:?}");
            stats[0] += 1;
        }
        eprintln!(
            "f5 fuzz: {} agreed, {} over budget, {} degenerate",
            stats[0], stats[1], stats[2]
        );
    }

    fn small_system() -> impl proptest::strategy::Strategy<Value = (Vec<GbPoly>, usize)> {
        use proptest::prelude::*;
        (2usize..=3, 2usize..=4, 0usize..3).prop_flat_map(|(n, m, ord)| {
            let term = (proptest::collection::vec(0u32..=3, n), -5i64..=5);
            let gen = proptest::collection::vec(term, 1..=4).prop_map(move |ts| {
                let mut p = GbPoly::zero(n);
                for (e, c) in ts {
                    if e.iter().sum::<u32>() <= 3 {
                        p = p.add(&GbPoly::monomial(e, rat(c)));
                    }
                }
                p
            });
            (proptest::collection::vec(gen, m), Just(ord))
        })
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

        #[test]
        fn f5_random_ideals_equal_buchberger((gens, ord) in small_system()) {
            let order = [MonomialOrder::Lex, MonomialOrder::GrLex, MonomialOrder::GRevLex][ord];
            let b1 = compute_buchberger_basis(gens.clone(), order);
            // Against the unverified engine as well: verification only makes a
            // wrong basis cost a fallback, and that must not hide a wrong basis.
            let nonzero: Vec<GbPoly> = gens.iter().filter(|g| !g.is_zero()).cloned().collect();
            if !nonzero.is_empty() {
                let raw = signature_basis(&nonzero, order);
                proptest::prop_assert_eq!(canon(&b1), canon(&raw));
            }
            let b2 = compute_groebner_basis_f5(gens, order);
            proptest::prop_assert_eq!(canon(&b1), canon(&b2));
        }
    }
}
