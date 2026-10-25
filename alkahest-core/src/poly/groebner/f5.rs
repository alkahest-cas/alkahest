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
//! * **Rewrite criterion** — among the elements whose signature divides an
//!   S-pair signature `T`, only the most recently added may generate it; pairs
//!   from any other element are skipped.
//!
//! Every S-pair the criteria do not discard is a genuine reduction step of
//! the signature-based theory, so the result is a Gröbner basis (Eder–Faugère
//! Thm 7.1 / Roune–Stillman 2012).  The final basis is interreduced and monic,
//! like [`super::buchberger::compute_buchberger_basis`]'s.
//!
//! This engine works in exact rational arithmetic.  It is correct and avoids
//! most zero reductions, but it is **not** the fast path:
//! [`super::GroebnerBasis::compute`] under grevlex is multimodular (F4 modulo
//! word-size primes, then lifted and verified) and is far faster on anything
//! beyond toy systems.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use rug::Rational;

use crate::poly::groebner::buchberger::interreduce;
use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;

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
/// the two halves' signatures.
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

    /// Some element newer than `k` has a signature dividing `index, t`.
    fn rewritable(&self, index: usize, t: &[u32], k: usize) -> bool {
        self.basis[k + 1..]
            .iter()
            .any(|g| g.sig.index == index && divides(&g.sig.mon, t))
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
/// same sense as [`super::buchberger::compute_buchberger_basis`].
pub fn compute_groebner_basis_f5(
    generators: Vec<GbPoly>,
    poly_order: MonomialOrder,
) -> Vec<GbPoly> {
    let gens: Vec<GbPoly> = generators.into_iter().filter(|g| !g.is_zero()).collect();
    let Some(n_vars) = gens.first().map(|g| g.n_vars) else {
        return Vec::new();
    };
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
        let mut last: Option<Vec<u32>> = None;
        while let Some(pair) = st.pairs.pop() {
            if crate::budget::interrupted() {
                break;
            }
            // One reduction per signature: the first pair popped for `T` has
            // the newest generator; the rest are rewritten by it.
            if last.as_ref() == Some(&pair.t) {
                continue;
            }
            last = Some(pair.t.clone());
            if st.syzygy_covers(pair.index, &pair.t) || st.rewritable(pair.index, &pair.t, pair.k) {
                continue;
            }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
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
            let b2 = compute_groebner_basis_f5(gens, order);
            proptest::prop_assert_eq!(canon(&b1), canon(&b2));
        }
    }
}
