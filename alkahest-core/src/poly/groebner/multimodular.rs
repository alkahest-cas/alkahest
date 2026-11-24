//! Multimodular Gröbner bases over ℚ (the modStd / msolve design).
//!
//! The reduced Gröbner basis `G` of an ideal `I ⊂ ℚ[x]` is unique, and for all
//! but finitely many primes `p` its image modulo `p` is the reduced basis of
//! the image ideal, which [`super::modp`] computes in machine words.  So:
//!
//! 1. compute the reduced basis modulo random primes `p < 2³¹` by F4
//!    ([`super::modp`]), recording each run's *trace* (which rows it built and
//!    which of them reduced to something new);
//! 2. once two full runs at independent primes give the same *shape* (the
//!    support of every element), replay one of their traces modulo each
//!    further prime instead of running F4 again: no pair handling, no symbolic
//!    preprocessing, and none of the rows that reduced to zero ("learn and
//!    apply", as in msolve and Groebner.jl).  A replay that diverges from the
//!    recorded run is discarded, and repeated divergence sends the loop back
//!    to full runs;
//! 3. combine the images of the agreed shape coefficient-wise by the Chinese
//!    remainder theorem and lift by rational reconstruction; accept a lift
//!    once one more prime agrees with it;
//! 4. **verify the lift over ℚ** before returning it: every input reduces to
//!    zero modulo the candidate `B` (so `I ⊆ ⟨B⟩`) and every critical pair of
//!    `B` that survives the Gebauer–Möller criteria reduces to zero (so `B` is
//!    a Gröbner basis), by the F4-style integer reduction of
//!    [`super::qverify`].  A candidate that fails is not returned; the loop
//!    learns again from fresh primes, and after a few failures the caller
//!    falls back to the ℚ algorithm.
//!
//! What the verification proves: `B` is the reduced Gröbner basis of an ideal
//! containing `I`.  For a **homogeneous** input that ideal is `I` itself
//! (Arnold, *Modular algorithms for computing Gröbner bases*, JSC 35 (2003),
//! Thm 7.1: a basis with the leading terms of the modular image of the input
//! ideal cannot generate a strictly larger homogeneous ideal).
//!
//! For an inhomogeneous input that equality is not proved, and
//! [`super::GroebnerBasis::proof_status`] says so.  Equal leading terms would
//! still force it, so `⟨B⟩ ⊋ I` needs the two primes whose full runs fixed the
//! shape, each drawn independently from the ≈5·10⁷ primes in `[2³⁰, 2³¹)`, to
//! give the same leading terms, ones `I` does not have.  Those primes divide a fixed
//! non-zero integer that depends only on the input; if there are `k` of them,
//! the chance is of order `(k / 5·10⁷)²`.  The basis `{1}` is the case to worry
//! about, since it passes the verification vacuously: `⟨y, y + N·x − 1⟩` has
//! the zero `(1/N, 0)`, yet modulo any `p | N` it is the unit ideal.
//!
//! Everything here is crate-private; [`super::GroebnerBasis::compute`] and
//! [`super::fglm`] use it behind their unchanged signatures.

use std::collections::{BTreeMap, HashMap};

use rug::{Assign, Integer, Rational};

use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::modp::{
    groebner_mod_p_learn, is_prime_u32, ApplyFail, ModPoly, MonoTable, Prime, Stop, Trace,
};
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::pairs::{update_pairs, CriticalPair};
use crate::poly::groebner::reduce::Divisors;

/// Primes taken before giving up on a reconstruction (each adds ~30 bits, so
/// this allows coefficients of about a million bits).  Past it the ℚ
/// algorithm would be hopeless too; an armed budget stops either earlier.
const MAX_PRIMES: usize = 1 << 16;

/// Candidates that fail verification over ℚ before giving up.
const MAX_FAILED_VERIFICATIONS: usize = 3;

// ---------------------------------------------------------------------------
// Primes
// ---------------------------------------------------------------------------

/// Distinct primes drawn independently and uniformly from `[2³⁰, 2³¹)`, so the
/// chance that several are all unlucky for an input is the product of their
/// chances.  (A run of consecutive primes from one random start would not do:
/// an input can make a whole run unlucky by having their product as a
/// coefficient, and then one random draw decides all of them.)
pub(crate) struct RandomPrimes {
    state: u64,
    seen: std::collections::HashSet<u64>,
}

impl RandomPrimes {
    pub(crate) fn new() -> Self {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(0x9e37_79b9_7f4a_7c15);
        RandomPrimes {
            state: h.finish(),
            seen: std::collections::HashSet::new(),
        }
    }

    /// SplitMix64: the seed is drawn after the input is fixed, so a statistical
    /// generator is enough; nothing here has to withstand a later observer.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

impl Iterator for RandomPrimes {
    type Item = Prime;
    fn next(&mut self) -> Option<Prime> {
        let lo = 1u64 << 30;
        loop {
            let n = (lo + self.next_u64() % lo) | 1;
            if is_prime_u32(n) && self.seen.insert(n) {
                return Some(Prime::new(n));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inputs over ℤ
// ---------------------------------------------------------------------------

/// A polynomial with integer coefficients, monomials interned and decreasing.
pub(crate) struct IntPoly {
    pub(crate) mons: Vec<u32>,
    pub(crate) coefs: Vec<Integer>,
}

/// The primitive integer multiple of `g` (same ideal), interned in `table`.
/// `None` if an exponent does not fit the table.
pub(crate) fn int_poly(table: &mut MonoTable, g: &GbPoly) -> Option<IntPoly> {
    let mut den = Integer::from(1);
    for c in g.terms.values() {
        den.lcm_mut(c.denom());
    }
    let mut terms: Vec<(u32, Integer)> = Vec::with_capacity(g.terms.len());
    for (e, c) in &g.terms {
        if *c == 0 {
            continue;
        }
        if e.len() != table.n_vars() {
            return None;
        }
        let m = table.intern_u32(e)?;
        let mut v = Integer::from(den.div_exact_ref(c.denom()));
        v *= c.numer();
        terms.push((m, v));
    }
    let mut content = Integer::new();
    for (_, c) in &terms {
        content.gcd_mut(c);
    }
    if content > 1 {
        for (_, c) in &mut terms {
            c.div_exact_mut(&content);
        }
    }
    terms.sort_by(|a, b| table.cmp(b.0, a.0));
    Some(IntPoly {
        mons: terms.iter().map(|t| t.0).collect(),
        coefs: terms.into_iter().map(|t| t.1).collect(),
    })
}

/// The coefficients of `gens` modulo `p`, on their full integer supports
/// (zeros kept, so every prime sees the same supports); `None` when `p`
/// divides a leading coefficient (the image would have a different leading
/// term).
pub(crate) fn coefs_mod_p(gens: &[IntPoly], pr: Prime) -> Option<Vec<Vec<u64>>> {
    let mut out = Vec::with_capacity(gens.len());
    for g in gens {
        let c: Vec<u64> = g
            .coefs
            .iter()
            .map(|c| u64::from(c.mod_u(pr.p as u32)))
            .collect();
        if c.first().is_some_and(|&lc| lc == 0) {
            return None;
        }
        out.push(c);
    }
    Some(out)
}

/// The images of `gens` given by [`coefs_mod_p`], as polynomials over 𝔽_p
/// with the full integer supports.
fn full_support_images(gens: &[IntPoly], coefs: Vec<Vec<u64>>) -> Vec<ModPoly> {
    gens.iter()
        .zip(coefs)
        .map(|(g, coefs)| ModPoly {
            mons: g.mons.clone(),
            coefs,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Chinese remaindering and rational reconstruction
// ---------------------------------------------------------------------------

/// The supports of a list of polynomials (monomial ids).
pub(crate) type Shape = Vec<Vec<u32>>;

/// A list of polynomials modulo one prime: its shape and its coefficients,
/// flattened in shape order.
pub(crate) struct Image {
    pub(crate) shape: Shape,
    pub(crate) coefs: Vec<u64>,
}

impl Image {
    pub(crate) fn from_polys(polys: Vec<ModPoly>) -> Self {
        let mut shape = Vec::with_capacity(polys.len());
        let mut coefs = Vec::new();
        for g in polys {
            coefs.extend_from_slice(&g.coefs);
            shape.push(g.mons);
        }
        Image { shape, coefs }
    }
}

/// `n/d` with `|n|, d ≤ ⌊√(m/2)⌋` and `n ≡ d·r (mod m)`, if one exists
/// (Wang's algorithm; the bound makes it unique).
fn rational_reconstruction(r: &Integer, m: &Integer, bound: &Integer) -> Option<Rational> {
    let (mut r0, mut r1) = (m.clone(), r.clone());
    let (mut t0, mut t1) = (Integer::new(), Integer::from(1));
    while r1 > *bound {
        let (q, r2) = std::mem::take(&mut r0).div_rem_floor(r1.clone());
        r0 = std::mem::replace(&mut r1, r2);
        let t2: Integer = &t0 - Integer::from(&q * &t1);
        t0 = std::mem::replace(&mut t1, t2);
    }
    if t1 == 0 || t1.cmp_abs(bound) == std::cmp::Ordering::Greater {
        return None;
    }
    if Integer::from(r1.gcd_ref(&t1)) != 1 {
        return None;
    }
    Some(Rational::from((r1, t1)))
}

/// `x mod p` for a rational with denominator prime to `p`.
pub(crate) fn rational_mod_p(x: &Rational, pr: Prime) -> Option<u64> {
    let d = u64::from(x.denom().mod_u(pr.p as u32));
    if d == 0 {
        return None;
    }
    let n = u64::from(x.numer().mod_u(pr.p as u32));
    Some(pr.mul(n, pr.inv(d)))
}

/// The images that share one shape, combined by CRT.
struct Group {
    shape: Shape,
    n_primes: usize,
    modulus: Integer,
    residues: Vec<Integer>,
    /// Primes are first combined in small batches (cheap, word-sized
    /// moduli) and merged into `residues` a batch at a time, so each big
    /// residue is touched once per batch rather than once per prime.
    batch_modulus: Integer,
    batch_residues: Vec<Integer>,
    batch_len: usize,
    next_try: usize,
}

/// Primes combined before a merge into the full-size residues.
const CRT_BATCH: usize = 16;

impl Group {
    /// Merge the pending batch into the full residues.
    fn flush(&mut self) {
        if self.batch_len == 0 {
            return;
        }
        if self.modulus == 1 {
            std::mem::swap(&mut self.modulus, &mut self.batch_modulus);
            std::mem::swap(&mut self.residues, &mut self.batch_residues);
        } else {
            let mb = &self.batch_modulus;
            let inv = Integer::from(&self.modulus % mb)
                .invert(mb)
                .expect("distinct primes");
            let mut t = Integer::new();
            for (r, b) in self.residues.iter_mut().zip(&self.batch_residues) {
                // r += M·((b − r)·M⁻¹ mod Mb)
                t.assign(b - &*r);
                t *= &inv;
                t.modulo_mut(mb);
                if t != 0 {
                    t *= &self.modulus;
                    *r += &t;
                }
            }
            self.modulus *= mb;
        }
        self.batch_modulus = Integer::from(1);
        for r in &mut self.batch_residues {
            r.assign(0);
        }
        self.batch_len = 0;
    }
}

/// Collects images modulo successive primes and lifts the majority shape.
pub(crate) struct Lifter {
    groups: Vec<Group>,
    /// A lift of `groups[.0]` waiting for one more agreeing prime.
    candidate: Option<(usize, Vec<Rational>)>,
    /// The group of the lift last returned by [`Self::add`], pending
    /// verification.
    proposed: Option<usize>,
    /// The coefficient that stopped the last reconstruction.
    last_fail: usize,
    /// The denominator lcm carried up to that coefficient.
    last_den: Integer,
}

impl Lifter {
    pub(crate) fn new() -> Self {
        Lifter {
            groups: Vec::new(),
            candidate: None,
            proposed: None,
            last_fail: 0,
            last_den: Integer::from(1),
        }
    }

    /// Discard the pending candidate (it failed verification): it will only
    /// be proposed again with more primes behind it.
    pub(crate) fn reject_candidate(&mut self) {
        if let Some(g) = self.proposed.take() {
            let grp = &mut self.groups[g];
            grp.next_try = grp.n_primes * 2;
        }
    }

    /// Add the image modulo `pr`.  Returns a lift that this image agrees with —
    /// the shape and the coefficients reconstructed from the earlier primes.
    pub(crate) fn add(&mut self, img: Image, pr: Prime) -> Option<(Shape, Vec<Rational>)> {
        let gi = match self.groups.iter().position(|g| g.shape == img.shape) {
            Some(i) => i,
            None => {
                self.groups.push(Group {
                    residues: vec![Integer::new(); img.coefs.len()],
                    batch_residues: vec![Integer::new(); img.coefs.len()],
                    batch_modulus: Integer::from(1),
                    batch_len: 0,
                    shape: img.shape.clone(),
                    n_primes: 0,
                    modulus: Integer::from(1),
                    next_try: 1,
                });
                self.groups.len() - 1
            }
        };
        if let Some((cg, lift)) = &self.candidate {
            if *cg == gi {
                let agrees = lift
                    .iter()
                    .zip(&img.coefs)
                    .all(|(x, &c)| rational_mod_p(x, pr) == Some(c));
                // A minority shape does not get to lift, however well its own
                // primes agree: that is what unlucky primes look like, and the
                // verification over ℚ cannot catch one whose ideal is larger.
                let with_this_prime = self.groups[gi].n_primes + 1;
                let majority = self
                    .groups
                    .iter()
                    .enumerate()
                    .all(|(i, g)| i == gi || g.n_primes < with_this_prime);
                if agrees && majority {
                    let (_, lift) = self.candidate.take().expect("candidate");
                    self.proposed = Some(gi);
                    let shape = self.groups[gi].shape.clone();
                    // Keep the image too, in case the lift is rejected.
                    self.crt(gi, &img, pr);
                    return Some((shape, lift));
                }
                self.candidate = None;
            }
        }
        self.crt(gi, &img, pr);
        let largest = (0..self.groups.len())
            .max_by_key(|&i| self.groups[i].n_primes)
            .expect("a group");
        let g = &self.groups[gi];
        if largest == gi && g.n_primes >= g.next_try {
            let n = g.n_primes;
            // The coefficient that stopped the last attempt is probed alone
            // first, without merging the pending batch into every residue:
            // that keeps an attempt per prime cheap, so the lift is found
            // within a prime or so of the precision it needs.
            if !self.probe(gi) {
                self.groups[gi].next_try = n + 1;
            } else {
                self.groups[gi].next_try = n + (n / 4).max(1);
                if let Some(lift) = self.reconstruct(gi) {
                    self.candidate = Some((gi, lift));
                }
            }
        }
        None
    }

    /// Whether the coefficient that failed the last reconstruction of group
    /// `gi` lifts now (`true` when there is none to probe).  The residue is
    /// combined from the merged residues and the pending batch for this one
    /// coefficient only.
    fn probe(&self, gi: usize) -> bool {
        let g = &self.groups[gi];
        let k = self.last_fail;
        if k >= g.residues.len() {
            return true;
        }
        let (r, m) = if g.modulus == 1 {
            (g.batch_residues[k].clone(), g.batch_modulus.clone())
        } else if g.batch_len == 0 {
            (g.residues[k].clone(), g.modulus.clone())
        } else {
            let mb = &g.batch_modulus;
            let Ok(inv) = Integer::from(&g.modulus % mb).invert(mb) else {
                return true;
            };
            let mut t = Integer::from(&g.batch_residues[k] - &g.residues[k]);
            t *= &inv;
            t.modulo_mut(mb);
            t *= &g.modulus;
            t += &g.residues[k];
            (t, Integer::from(&g.modulus * mb))
        };
        lift_coefficient(&r, &m, &mut self.last_den.clone()).is_some()
    }

    fn crt(&mut self, gi: usize, img: &Image, pr: Prime) {
        let g = &mut self.groups[gi];
        let p = pr.p;
        let m_mod_p = u64::from(g.batch_modulus.mod_u(p as u32));
        let m_inv = pr.inv(m_mod_p);
        let mut t = Integer::new();
        for (r, &a) in g.batch_residues.iter_mut().zip(&img.coefs) {
            let rp = u64::from(r.mod_u(p as u32));
            let delta = pr.mul((a + p - rp) % p, m_inv);
            if delta != 0 {
                t.assign(&g.batch_modulus * delta);
                *r += &t;
            }
        }
        g.batch_modulus *= p;
        g.batch_len += 1;
        g.n_primes += 1;
        if g.batch_len == CRT_BATCH {
            g.flush();
        }
    }

    /// Lift every coefficient of group `gi`, or `None` if one does not lift
    /// yet.  Coefficients of a Gröbner basis tend to share their denominators,
    /// so the lcm `den` of the denominators found so far is carried along: a
    /// coefficient `c` with `c·den` integral is recovered as the symmetric
    /// residue of `r·den` as soon as the modulus exceeds `|c·den|` by 64 bits —
    /// about half the primes balanced rational reconstruction needs.  (A lift
    /// is only ever accepted after another prime agrees and the result is
    /// verified over ℚ, so the 64-bit margin is a heuristic, not a proof.)
    fn reconstruct(&mut self, gi: usize) -> Option<Vec<Rational>> {
        self.groups[gi].flush();
        let g = &self.groups[gi];
        let m = &g.modulus;
        let lift = |r: &Integer, den: &mut Integer| lift_coefficient(r, m, den);
        // The coefficient that failed last time usually fails again: try it
        // first, with the denominator carried up to it then.  Then start from
        // scratch: a lift that went wrong at a lower precision must not leave
        // a spurious factor in `den`.
        if let Some(r) = g.residues.get(self.last_fail) {
            lift(r, &mut self.last_den.clone())?;
        }
        let mut den = Integer::from(1);
        let mut out = Vec::with_capacity(g.residues.len());
        for (k, r) in g.residues.iter().enumerate() {
            // The modulus of a hard input runs to millions of bits, and one
            // reconstruction is a half-gcd on it.  `None` is "not liftable
            // yet", and the budget stops the caller at its next poll.
            if crate::budget::interrupted() {
                return None;
            }
            match lift(r, &mut den) {
                Some(x) => out.push(x),
                None => {
                    self.last_fail = k;
                    self.last_den = den;
                    return None;
                }
            }
        }
        Some(out)
    }
}

/// Lift one residue `r` modulo `m`.  Coefficients of a Gröbner basis tend
/// to share their denominators, so `den`, the lcm of the denominators found so
/// far, is carried along: a coefficient `c` with `c·den` integral is
/// recovered as the symmetric residue of `r·den` as soon as `m` exceeds
/// `|c·den|` by 64 bits, about half the precision balanced rational
/// reconstruction needs; otherwise rational reconstruction of `r·den`, whose
/// denominator then joins `den`.
pub(crate) fn lift_coefficient(r: &Integer, m: &Integer, den: &mut Integer) -> Option<Rational> {
    let half = Integer::from(m >> 1u32);
    let mut t = Integer::from(r * &*den);
    t %= m;
    if t > half {
        t -= m;
    }
    if t.significant_bits() + 64 < m.significant_bits() {
        return Some(Rational::from((t, den.clone())));
    }
    let t = if t < 0 { t + m } else { t };
    let mut bound = half;
    bound.sqrt_mut();
    let x = rational_reconstruction(&t, m, &bound)?;
    let d = x.denom().clone();
    let out = Rational::from((x.numer().clone(), Integer::from(&d * &*den)));
    *den *= d;
    Some(out)
}

/// Rebuild `GbPoly`s from a shape and its coefficients.
pub(crate) fn polys_from_lift(
    table: &MonoTable,
    shape: &Shape,
    coefs: Vec<Rational>,
) -> Vec<GbPoly> {
    let mut it = coefs.into_iter();
    shape
        .iter()
        .map(|mons| {
            let terms: BTreeMap<Vec<u32>, Rational> = mons
                .iter()
                .map(|&m| (table.exps_u32(m), it.next().expect("coefficient")))
                .collect();
            GbPoly {
                terms,
                n_vars: table.n_vars(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Verification over ℚ
// ---------------------------------------------------------------------------

/// `true` when every input reduces to zero modulo `cand` and every critical
/// pair of `cand` left by the Gebauer–Möller criteria reduces to zero — i.e.
/// `cand` is a Gröbner basis of an ideal containing the inputs.  `None` when
/// an armed budget fired.
pub(crate) fn verify_over_q(
    inputs: &[GbPoly],
    cand: &[GbPoly],
    order: MonomialOrder,
) -> Option<bool> {
    let divs = Divisors::new_fraction_free(cand, order);
    for f in inputs {
        let r = divs.reduce_scaled(f, None);
        if crate::budget::interrupted() {
            return None;
        }
        if !r.is_zero() {
            return Some(false);
        }
    }
    let mut lead = Vec::with_capacity(cand.len());
    let mut sugar = Vec::with_capacity(cand.len());
    let mut pairs: Vec<CriticalPair> = Vec::new();
    for (k, g) in cand.iter().enumerate() {
        let Some(l) = g.leading_exp(order) else {
            return Some(false);
        };
        lead.push(l);
        sugar.push(g.sugar());
        update_pairs(&lead, &sugar, &mut pairs, k);
    }
    // In chunks, each spread over the worker threads; the budget is polled
    // between chunks here and inside every reduction by the workers.
    let chunk = 8 * batch_width();
    for c in pairs.chunks(chunk) {
        if crate::budget::interrupted() {
            return None;
        }
        let zero = run_jobs(c.iter().collect(), |p| {
            divs.reduce_s_pair_scaled(p.i, p.j).is_zero()
        })
        .ok()?;
        if crate::budget::interrupt_pending() {
            return None;
        }
        if zero.iter().any(|z| !z) {
            return Some(false);
        }
    }
    Some(true)
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Why the multimodular path produced no basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unavailable {
    /// Not applicable or not conclusive: use the ℚ algorithm.
    Fallback,
    /// An armed budget fired; the caller discards whatever it returns.
    Interrupted,
}

/// The reduced Gröbner basis of `⟨gens⟩` over ℚ under `order`, monic and
/// sorted by *decreasing* leading monomial (SymPy's convention, and the order
/// Buchberger's insertion order most often produced for these callers),
/// computed modularly and verified.
pub(crate) fn groebner_basis(
    gens: &[GbPoly],
    order: MonomialOrder,
) -> Result<Vec<GbPoly>, Unavailable> {
    groebner_basis_checked(gens, order, false)
}

/// [`groebner_basis`], with the lift checked at further primes instead of
/// over ℚ when `modular` (see [`super::GroebnerVerification::Modular`]).
pub(crate) fn groebner_basis_checked(
    gens: &[GbPoly],
    order: MonomialOrder,
    modular: bool,
) -> Result<Vec<GbPoly>, Unavailable> {
    let gens: Vec<GbPoly> = gens.iter().filter(|g| !g.is_zero()).cloned().collect();
    let Some(n) = gens.first().map(|g| g.n_vars) else {
        return Err(Unavailable::Fallback);
    };
    if gens.iter().any(|g| g.n_vars != n) {
        return Err(Unavailable::Fallback);
    }
    let mut table = MonoTable::new(n, order);
    let mut ints = Vec::with_capacity(gens.len());
    for g in &gens {
        ints.push(int_poly(&mut table, g).ok_or(Unavailable::Fallback)?);
    }

    traced_lift_loop(&mut table, &ints, modular, |cand| {
        verify_over_q(&gens, cand, order)
    })
    .map(|mut b| {
        b.reverse();
        b
    })
}

// ---------------------------------------------------------------------------
// Parallel prime loop
// ---------------------------------------------------------------------------

/// Run `f` on every job, in parallel when the `parallel` feature is on, under
/// the caller's budget: each worker enters what is left of the caller's
/// [`crate::budget`] and arms interrupt points if the caller had, so a long
/// job stops on time; a trip in any worker is recorded on the calling thread.
/// `Err(())` after such a trip.
pub(crate) fn run_jobs<T: Send, R: Send>(
    jobs: Vec<T>,
    f: impl Fn(T) -> R + Sync,
) -> Result<Vec<R>, ()> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        if jobs.len() > 1 {
            let armed = crate::budget::is_armed();
            let rem = if armed {
                crate::budget::remaining()
            } else {
                None
            };
            let out: Vec<Result<R, crate::budget::BudgetTrip>> = jobs
                .into_par_iter()
                .map(|job| {
                    let _budget = rem.map(|(b, m)| crate::budget::enter_with_memory(b, m));
                    let guard = armed.then(crate::budget::arm);
                    let r = f(job);
                    match guard.as_ref().and_then(|g| g.trip()) {
                        Some(trip) => Err(trip),
                        None => Ok(r),
                    }
                })
                .collect();
            let mut res = Vec::with_capacity(out.len());
            for r in out {
                match r {
                    Ok(x) => res.push(x),
                    Err(trip) => {
                        crate::budget::record_trip(trip);
                        return Err(());
                    }
                }
            }
            return Ok(res);
        }
    }
    let mut res = Vec::with_capacity(jobs.len());
    for job in jobs {
        res.push(f(job));
        if crate::budget::interrupt_pending() {
            return Err(());
        }
    }
    Ok(res)
}

/// Primes computed side by side after the first.
pub(crate) fn batch_width() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().clamp(1, 16)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Further primes a lift must agree with in the modular mode, after the one
/// every lift needs before it is proposed.
const MODULAR_CHECKS: usize = 2;

/// A proposed lift under the modular check
/// ([`super::GroebnerVerification::Modular`]): it is accepted once
/// [`MODULAR_CHECKS`] images at primes not used to reconstruct it agree with
/// it, and rejected at the first that does not.
struct ModularCheck {
    shape: Shape,
    lift: Vec<Rational>,
    agreed: usize,
}

impl ModularCheck {
    fn new(shape: Shape, lift: Vec<Rational>) -> Self {
        ModularCheck {
            shape,
            lift,
            agreed: 0,
        }
    }

    /// `Some(true)` once enough images agree, `Some(false)` at one that does
    /// not (another shape, or a coefficient that differs), `None` while
    /// undecided.  An image modulo a prime dividing a denominator of the lift
    /// says nothing and is skipped.
    fn feed(&mut self, img: &Image, pr: Prime) -> Option<bool> {
        if img.shape != self.shape {
            return Some(false);
        }
        let mut informative = true;
        for (x, &c) in self.lift.iter().zip(&img.coefs) {
            match rational_mod_p(x, pr) {
                Some(r) if r == c => {}
                Some(_) => return Some(false),
                None => informative = false,
            }
        }
        if informative {
            self.agreed += 1;
        }
        (self.agreed >= MODULAR_CHECKS).then_some(true)
    }
}

/// What one prime gave: a basis image (monomials in the table it was computed
/// with), nothing usable (a bad prime), or a reason to stop.
pub(crate) type PrimeResult = Result<Option<Vec<ModPoly>>, Unavailable>;

/// The multimodular loop shared by the grevlex and FGLM paths: compute images
/// modulo successive random primes with `compute` (the first alone, then in
/// parallel batches on clones of `table`), lift the majority shape, and hand
/// each lift to `verify` until one passes.
///
/// With `modular`, a lift is accepted on [`ModularCheck`] instead and
/// `verify` is only called on the accepted one, to finish it: its answer is
/// then whether the candidate could be built at all (`Some(false)` rejects
/// it like a disagreeing prime).
pub(crate) fn lift_loop(
    table: &mut MonoTable,
    modular: bool,
    compute: impl Fn(&mut MonoTable, Prime) -> PrimeResult + Sync,
    mut verify: impl FnMut(&[GbPoly]) -> Option<bool>,
) -> Result<Vec<GbPoly>, Unavailable> {
    let mut checking: Option<ModularCheck> = None;
    let mut lifter = Lifter::new();
    let mut failed = 0usize;
    let mut primes = RandomPrimes::new();
    let mut used = 0usize;
    let mut width = 1usize;
    loop {
        if used >= MAX_PRIMES {
            return Err(Unavailable::Fallback);
        }
        if crate::budget::interrupted() {
            return Err(Unavailable::Interrupted);
        }
        let batch: Vec<Prime> = primes.by_ref().take(width).collect();
        if batch.is_empty() {
            return Err(Unavailable::Fallback);
        }
        used += batch.len();
        width = batch_width();
        let results: Vec<(Prime, PrimeResult)> = if batch.len() == 1 {
            let p = batch[0];
            vec![(p, compute(table, p))]
        } else {
            // Each worker interns into its own copy of the table; ids below
            // `known` mean the same monomial everywhere, the rest are mapped
            // back into `table` by exponent vector.
            let known = table.len();
            let base: &MonoTable = table;
            let out = run_jobs(batch, |p| {
                let mut t = base.clone();
                let r = compute(&mut t, p);
                (p, r, t)
            })
            .map_err(|()| Unavailable::Interrupted)?;
            out.into_iter()
                .map(|(p, r, t)| {
                    let r = r.map(|o| {
                        o.map(|mut polys| {
                            for g in &mut polys {
                                for m in &mut g.mons {
                                    if (*m as usize) >= known {
                                        *m = table.intern(t.exps(*m));
                                    }
                                }
                            }
                            polys
                        })
                    });
                    (p, r)
                })
                .collect()
        };
        for (p, r) in results {
            let Some(gb) = r? else {
                continue;
            };
            let img = Image::from_polys(gb);
            if let Some(chk) = &mut checking {
                let verdict = chk.feed(&img, p);
                if verdict.is_none() {
                    continue;
                }
                let chk = checking.take().expect("checking");
                if verdict == Some(true) {
                    let cand = polys_from_lift(table, &chk.shape, chk.lift);
                    match verify(&cand) {
                        None => return Err(Unavailable::Interrupted),
                        Some(true) => return Ok(cand),
                        Some(false) => {}
                    }
                }
                failed += 1;
                if failed >= MAX_FAILED_VERIFICATIONS {
                    return Err(Unavailable::Fallback);
                }
                lifter.reject_candidate();
            }
            let lifted = lifter.add(img, p);
            let Some((shape, lift)) = lifted else {
                continue;
            };
            if modular {
                checking = Some(ModularCheck::new(shape, lift));
                continue;
            }
            let cand = polys_from_lift(table, &shape, lift);
            let vr = verify(&cand);
            match vr {
                None => return Err(Unavailable::Interrupted),
                Some(true) => return Ok(cand),
                Some(false) => {
                    failed += 1;
                    if failed >= MAX_FAILED_VERIFICATIONS {
                        return Err(Unavailable::Fallback);
                    }
                    lifter.reject_candidate();
                }
            }
        }
    }
}

/// A prime and the inputs' coefficients modulo it.
type PrimeImage = (Prime, Vec<Vec<u64>>);

/// What replaying a trace modulo one prime gave.
type Replay = Result<Vec<ModPoly>, ApplyFail>;

/// Primes replayed side by side in one pass over a trace.
const LANES: usize = 4;

/// Full F4 runs kept while looking for two whose shapes agree.
const MAX_LEARN_RUNS: usize = 4;

/// Consecutive primes at which a trace fails to replay before it is
/// relearned, and relearnings before every prime gets a full run.
const MAX_APPLY_MISMATCHES: usize = 2;
const MAX_RELEARNS: usize = 3;

/// The multimodular loop of the grevlex path, with tracing ("learn and
/// apply", as in msolve and Groebner.jl).
///
/// A full F4 run modulo a prime records which rows it built and which of them
/// reduced to something new (a [`Trace`]).  Once **two** full runs at
/// independent random primes give the same shape, one of their traces is
/// replayed modulo each further prime: only the rows that mattered are
/// rebuilt and reduced, with no pair handling or symbolic preprocessing.  A
/// replay that diverges from the learned run (a pivot vanishes, or a column
/// the learning prime saw as zero is not) reports a mismatch; repeated
/// mismatches, or a candidate that fails verification, send the loop back to
/// full runs.
///
/// Correctness does not rest on the trace.  The images are lifted and
/// verified over ℚ exactly as in [`lift_loop`], and the shape of every lift
/// is one that two independent full F4 runs produced, as before.  A replay
/// modulo `q` is the image modulo `q` of one fixed sequence of row operations
/// over ℚ (it only divides by pivots that are non-zero modulo `q`, hence over
/// ℚ), so each replayed element is the image of an element of the ideal.
///
/// With `modular`, a lift is accepted on [`ModularCheck`] instead of being
/// verified over ℚ.
fn traced_lift_loop(
    table: &mut MonoTable,
    ints: &[IntPoly],
    modular: bool,
    mut verify: impl FnMut(&[GbPoly]) -> Option<bool>,
) -> Result<Vec<GbPoly>, Unavailable> {
    let mut checking: Option<ModularCheck> = None;
    let mut lifter = Lifter::new();
    let mut failed = 0usize;
    let mut primes = RandomPrimes::new();
    let mut used = 0usize;
    let mut learned: Vec<(Shape, Trace)> = Vec::new();
    let mut trace: Option<Trace> = None;
    let mut mismatches = 0usize;
    let mut relearns = 0usize;
    let mut replayed = 0usize;
    loop {
        if used >= MAX_PRIMES {
            return Err(Unavailable::Fallback);
        }
        if crate::budget::interrupted() {
            return Err(Unavailable::Interrupted);
        }
        let mut results: Vec<(Prime, Vec<ModPoly>)> = Vec::new();
        match &trace {
            None => {
                let p = primes.next().ok_or(Unavailable::Fallback)?;
                used += 1;
                let Some(c) = coefs_mod_p(ints, p) else {
                    continue;
                };
                let (gb, tr) = match groebner_mod_p_learn(table, &full_support_images(ints, c), p) {
                    Ok(x) => x,
                    Err(Stop::Interrupted) => return Err(Unavailable::Interrupted),
                    Err(Stop::Overflow) => return Err(Unavailable::Fallback),
                };
                if relearns <= MAX_RELEARNS {
                    let shape: Shape = gb.iter().map(|g| g.mons.clone()).collect();
                    if let Some(i) = learned.iter().position(|(s, _)| *s == shape) {
                        trace = Some(learned.swap_remove(i).1);
                        learned.clear();
                        mismatches = 0;
                    } else if learned.len() < MAX_LEARN_RUNS {
                        learned.push((shape, tr));
                    } else {
                        // Shapes keep disagreeing: no trace; every prime
                        // gets a full run and the majority shape lifts.
                        relearns = MAX_RELEARNS + 1;
                        learned.clear();
                    }
                }
                results.push((p, gb));
            }
            Some(tr) => {
                // The first pass over a trace takes two primes: a small input
                // may need no more, and four would be wasted on it.
                let lanes = if replayed == 0 { 2 } else { LANES };
                let batch: Vec<Prime> = primes.by_ref().take(lanes * batch_width()).collect();
                replayed += batch.len();
                if batch.is_empty() {
                    return Err(Unavailable::Fallback);
                }
                used += batch.len();
                // Several primes per pass over the trace (see `Trace::apply4`).
                let good: Vec<PrimeImage> = batch
                    .into_iter()
                    .filter_map(|p| coefs_mod_p(ints, p).map(|c| (p, c)))
                    .collect();
                let jobs: Vec<&[PrimeImage]> = good.chunks(LANES).collect();
                let out: Vec<Option<(Prime, Replay)>> = run_jobs(jobs, |job| match job {
                    [(p, a), (q, b), (r, c), (s, e)] => {
                        let [ra, rb, rc, re] = tr.apply4(
                            [a.as_slice(), b.as_slice(), c.as_slice(), e.as_slice()],
                            [*p, *q, *r, *s],
                        );
                        vec![
                            Some((*p, ra)),
                            Some((*q, rb)),
                            Some((*r, rc)),
                            Some((*s, re)),
                        ]
                    }
                    [(p, a), (q, b), rest @ ..] => {
                        let [ra, rb] = tr.apply2([a.as_slice(), b.as_slice()], [*p, *q]);
                        let mut v = vec![Some((*p, ra)), Some((*q, rb))];
                        for (r, c) in rest {
                            v.push(Some((*r, tr.apply(c, *r))));
                        }
                        v
                    }
                    [(p, a)] => vec![Some((*p, tr.apply(a, *p)))],
                    _ => Vec::new(),
                })
                .map_err(|()| Unavailable::Interrupted)?
                .into_iter()
                .flatten()
                .collect();
                for r in out.into_iter().flatten() {
                    match r {
                        (p, Ok(gb)) => {
                            mismatches = 0;
                            results.push((p, gb));
                        }
                        (_, Err(ApplyFail::Interrupted)) => return Err(Unavailable::Interrupted),
                        (_, Err(ApplyFail::Mismatch)) => mismatches += 1,
                    }
                }
                if mismatches >= MAX_APPLY_MISMATCHES {
                    trace = None;
                    relearns += 1;
                }
            }
        }
        for (p, gb) in results {
            let img = Image::from_polys(gb);
            if let Some(chk) = &mut checking {
                match chk.feed(&img, p) {
                    None => continue,
                    Some(true) => {
                        let chk = checking.take().expect("checking");
                        return Ok(polys_from_lift(table, &chk.shape, chk.lift));
                    }
                    Some(false) => {
                        checking = None;
                        failed += 1;
                        if failed >= MAX_FAILED_VERIFICATIONS {
                            return Err(Unavailable::Fallback);
                        }
                        lifter.reject_candidate();
                        if trace.take().is_some() {
                            relearns += 1;
                        }
                        learned.clear();
                    }
                }
            }
            let lifted = lifter.add(img, p);
            let Some((shape, lift)) = lifted else {
                continue;
            };
            if modular {
                checking = Some(ModularCheck::new(shape, lift));
                continue;
            }
            let fast = crate::poly::groebner::qverify::verify_lift(table, ints, &shape, &lift);
            let cand = polys_from_lift(table, &shape, lift);
            let vr = match fast {
                Ok(v) => Some(v),
                Err(Stop::Interrupted) => None,
                // An exponent past the table's range: the term-by-term check.
                Err(Stop::Overflow) => verify(&cand),
            };
            match vr {
                None => return Err(Unavailable::Interrupted),
                Some(true) => return Ok(cand),
                Some(false) => {
                    failed += 1;
                    if failed >= MAX_FAILED_VERIFICATIONS {
                        return Err(Unavailable::Fallback);
                    }
                    lifter.reject_candidate();
                    // The trace may come from an unlucky pair of primes:
                    // learn again.
                    if trace.take().is_some() {
                        relearns += 1;
                    }
                    learned.clear();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FGLM: grevlex → lex for a zero-dimensional ideal
// ---------------------------------------------------------------------------

/// Where `x_i · s` lands, for a staircase monomial `s`: another staircase
/// monomial, or (on the border) the normal form, a sparse rational vector.
pub(crate) enum MulCol {
    Std(usize),
    Nf(Vec<(usize, Rational)>),
}

/// A multiplication matrix modulo `p`, column by column (see [`MulCol`]).
pub(crate) type ModMulMatrix = Vec<Result<usize, Vec<(usize, u64)>>>;

/// The multiplication matrices of `ℚ[x]/I` on the grevlex staircase basis:
/// `mul[i][j]` is the normal form of `x_i · staircase[j]`.
///
/// With `partial`, only the matrix of the last variable is complete; the
/// others hold their first column alone (the coordinates of `x_i · 1`), which
/// is all the shape-position path ([`super::shape`]) and its verification
/// read.
pub(crate) struct MulMatrices {
    pub(crate) mul: Vec<Vec<MulCol>>,
    pub(crate) d: usize,
    pub(crate) partial: bool,
}

impl MulMatrices {
    pub(crate) fn new(
        g: &[GbPoly],
        n: usize,
        staircase: &[Vec<u32>],
        idx: &HashMap<Vec<u32>, usize>,
        partial: bool,
    ) -> Option<Self> {
        let divs = Divisors::new(g, MonomialOrder::GRevLex);
        let d = staircase.len();
        let mut mul = Vec::with_capacity(n);
        for i in 0..n {
            let cols_wanted = if partial && i + 1 < n { 1 } else { d };
            let mut cols = Vec::with_capacity(cols_wanted);
            for s in &staircase[..cols_wanted] {
                if crate::budget::interrupted() {
                    return None;
                }
                let mut e = s.clone();
                e[i] += 1;
                if let Some(&k) = idx.get(&e) {
                    cols.push(MulCol::Std(k));
                    continue;
                }
                let nf = divs.reduce(&GbPoly::monomial(e, Rational::from(1)), None);
                let mut v = Vec::with_capacity(nf.terms.len());
                for (m, c) in nf.terms {
                    // A normal form is supported on the staircase.
                    v.push((*idx.get(&m)?, c));
                }
                cols.push(MulCol::Nf(v));
            }
            mul.push(cols);
        }
        Some(MulMatrices { mul, d, partial })
    }

    /// The matrices modulo `p`; `None` if `p` divides a denominator.
    pub(crate) fn mod_p(&self, pr: Prime) -> Option<Vec<ModMulMatrix>> {
        let mut out = Vec::with_capacity(self.mul.len());
        for cols in &self.mul {
            let mut oc = Vec::with_capacity(cols.len());
            for c in cols {
                oc.push(match c {
                    MulCol::Std(k) => Ok(*k),
                    MulCol::Nf(v) => {
                        let mut w = Vec::with_capacity(v.len());
                        for (k, x) in v {
                            let r = rational_mod_p(x, pr)?;
                            if r != 0 {
                                w.push((*k, r));
                            }
                        }
                        Err(w)
                    }
                });
            }
            out.push(oc);
        }
        Some(out)
    }
}

/// `*a += x` for `a, x < p²`, folded back below `p²` (`p < 2³¹`).
#[inline(always)]
fn add_folded(a: &mut u64, x: u64, p2: u64) {
    *a += x;
    if *a >= p2 {
        *a -= p2;
    }
}

/// A candidate monomial of the lex traversal: `x_var · std[parent]`.
#[derive(PartialEq, Eq)]
struct Cand {
    exps: Vec<u32>,
    parent: usize,
    var: usize,
}

impl Ord for Cand {
    /// Lex on the exponents (vectors of one length compare lexicographically).
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.exps.cmp(&o.exps).then(self.parent.cmp(&o.parent))
    }
}

impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

/// One FGLM run modulo `p`: the reduced lex basis, monomials in `table`
/// (a lex table), sorted by increasing leading monomial.  `Err(Overflow)`
/// also reports a rank defect (more than `d` independent vectors, or fewer):
/// impossible for a Gröbner basis input and a good prime.
fn fglm_mod_p(
    table: &mut MonoTable,
    mul: &[ModMulMatrix],
    d: usize,
    n: usize,
    pr: Prime,
) -> Result<Vec<ModPoly>, Stop> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let p = pr.p;
    let p2 = p * p;
    // Standard (lex) monomials and their coordinate vectors, and the echelon
    // form of those vectors: row k has pivot column `piv[k]`, entries
    // `rows[k]`, and `rows[k] = Σ_j tr[k][j]·vec(std_j)`.
    let mut std_exps: Vec<Vec<u32>> = Vec::new();
    let mut std_vecs: Vec<Vec<u64>> = Vec::new();
    let mut rows: Vec<Vec<u64>> = Vec::new();
    let mut piv: Vec<usize> = Vec::new();
    let mut tr: Vec<Vec<u64>> = Vec::new();
    let mut leads: Vec<Vec<u32>> = Vec::new();
    let mut out: Vec<ModPoly> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut heap: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();

    let mut unit = vec![0u64; d];
    // The staircase is sorted increasingly, so `1` is its first element.
    unit[0] = 1;
    let mut pending: Option<(Vec<u32>, Vec<u64>)> = Some((vec![0u32; n], unit));
    let mut ticks = 0u32;
    loop {
        let (m, v) = match pending.take() {
            Some(x) => x,
            None => {
                let Some(Reverse(c)) = heap.pop() else {
                    break;
                };
                if leads
                    .iter()
                    .any(|l| l.iter().zip(&c.exps).all(|(a, b)| a <= b))
                {
                    continue;
                }
                // vec(x_var · parent) = M_var · vec(parent), accumulated
                // below p² and reduced once at the end.
                let mut v = vec![0u64; d];
                for (j, &x) in std_vecs[c.parent].iter().enumerate() {
                    if x == 0 {
                        continue;
                    }
                    match &mul[c.var][j] {
                        Ok(k) => add_folded(&mut v[*k], x, p2),
                        Err(col) => {
                            for &(k, y) in col {
                                add_folded(&mut v[k], x * y, p2);
                            }
                        }
                    }
                }
                for x in &mut v {
                    *x %= p;
                }
                (c.exps, v)
            }
        };
        if crate::poly::groebner::reduce::poll_due(&mut ticks) && crate::budget::interrupted() {
            return Err(Stop::Interrupted);
        }
        // Reduce against the echelon rows, tracking the combination.
        // Entries stay below p² (folded) and are reduced when read.
        let mut w = v.clone();
        let mut comb = vec![0u64; std_exps.len()];
        for k in 0..rows.len() {
            let f = w[piv[k]] % p;
            if f == 0 {
                continue;
            }
            let nf = p - f;
            for (a, &b) in w.iter_mut().zip(&rows[k]) {
                add_folded(a, nf * b, p2);
            }
            for (a, &b) in comb.iter_mut().zip(&tr[k]) {
                add_folded(a, f * b, p2);
            }
        }
        for x in w.iter_mut().chain(comb.iter_mut()) {
            *x %= p;
        }
        match w.iter().position(|&x| x != 0) {
            None => {
                // m ≡ Σ comb_j·std_j : the lex element m − Σ comb_j·std_j.
                let mut terms: Vec<(u32, u64)> = Vec::with_capacity(comb.len() + 1);
                terms.push((table.intern_u32(&m).ok_or(Stop::Overflow)?, 1));
                for (j, &c) in comb.iter().enumerate() {
                    if c != 0 {
                        let id = table.intern_u32(&std_exps[j]).ok_or(Stop::Overflow)?;
                        terms.push((id, p - c));
                    }
                }
                terms[1..].sort_by(|a, b| table.cmp(b.0, a.0));
                out.push(ModPoly {
                    mons: terms.iter().map(|t| t.0).collect(),
                    coefs: terms.iter().map(|t| t.1).collect(),
                });
                leads.push(m);
            }
            Some(c0) => {
                if std_exps.len() == d {
                    return Err(Stop::Overflow);
                }
                let inv = pr.inv(w[c0]);
                for a in w.iter_mut() {
                    *a = pr.mul(*a, inv);
                }
                // row_new = (v − Σ f_k·row_k)/w[c0], so
                // tr_new = (e_new − comb)/w[c0].
                let k_new = std_exps.len();
                let mut t = vec![0u64; k_new + 1];
                for (j, &c) in comb.iter().enumerate() {
                    t[j] = pr.mul((p - c) % p, inv);
                }
                t[k_new] = inv;
                for row in &mut tr {
                    row.push(0);
                }
                tr.push(t);
                rows.push(w);
                piv.push(c0);
                for var in 0..n {
                    let mut e = m.clone();
                    e[var] += 1;
                    if seen.insert(e.clone()) {
                        heap.push(Reverse(Cand {
                            exps: e,
                            parent: k_new,
                            var,
                        }));
                    }
                }
                std_exps.push(m);
                std_vecs.push(v);
            }
        }
    }
    if std_exps.len() != d {
        return Err(Stop::Overflow);
    }
    Ok(out)
}

/// The number of monomials outside `⟨leads⟩`, or `None` if it exceeds `cap`.
fn staircase_size(leads: &[Vec<u32>], n: usize, cap: usize) -> Option<usize> {
    let divisible = |m: &[u32]| leads.iter().any(|l| l.iter().zip(m).all(|(a, b)| a <= b));
    let one = vec![0u32; n];
    if divisible(&one) {
        return Some(0);
    }
    let mut seen = std::collections::HashSet::new();
    seen.insert(one.clone());
    let mut stack = vec![one];
    while let Some(m) = stack.pop() {
        if seen.len() > cap {
            return None;
        }
        for i in 0..n {
            let mut e = m.clone();
            e[i] += 1;
            if !divisible(&e) && seen.insert(e.clone()) {
                stack.push(e);
            }
        }
    }
    (seen.len() <= cap).then_some(seen.len())
}

/// A multiplication matrix scaled to integers: sparse columns and the scale.
type IntMatrix = (Vec<Vec<(usize, Integer)>>, Integer);

/// Exact coordinate vectors over ℚ (integer numerators over one
/// denominator), for verifying a lifted lex basis.
struct ExactVectors {
    d: usize,
    /// Per variable: its multiplication matrix scaled to integers, column by
    /// column, and the scale.
    int_cols: Vec<IntMatrix>,
    memo: HashMap<Vec<u32>, (Vec<Integer>, Integer)>,
}

impl ExactVectors {
    fn new(mats: &MulMatrices) -> Self {
        let mut int_cols = Vec::with_capacity(mats.mul.len());
        for cols in &mats.mul {
            let mut den = Integer::from(1);
            for c in cols {
                if let MulCol::Nf(v) = c {
                    for (_, x) in v {
                        den.lcm_mut(x.denom());
                    }
                }
            }
            let ic = cols
                .iter()
                .map(|c| match c {
                    MulCol::Std(k) => vec![(*k, den.clone())],
                    MulCol::Nf(v) => v
                        .iter()
                        .map(|(k, x)| {
                            let mut a = Integer::from(den.div_exact_ref(x.denom()));
                            a *= x.numer();
                            (*k, a)
                        })
                        .collect(),
                })
                .collect();
            int_cols.push((ic, den));
        }
        let mut memo = HashMap::new();
        let mut one = vec![Integer::new(); mats.d];
        one[0] = Integer::from(1);
        memo.insert(vec![0u32; mats.mul.len()], (one, Integer::from(1)));
        ExactVectors {
            d: mats.d,
            int_cols,
            memo,
        }
    }

    /// Make sure the coordinate vector of `m` is in the memo; `false` when an
    /// armed budget fired.
    fn ensure(&mut self, m: &[u32]) -> bool {
        if self.memo.contains_key(m) {
            return true;
        }
        // Walk down to a known divisor, then multiply back up.
        let mut chain = vec![m.to_vec()];
        loop {
            let last = chain.last().expect("non-empty");
            let i = last.iter().position(|&e| e > 0).expect("1 is memoised");
            let mut parent = last.clone();
            parent[i] -= 1;
            if self.memo.contains_key(&parent) {
                break;
            }
            chain.push(parent);
        }
        while let Some(e) = chain.pop() {
            if crate::budget::interrupted() {
                return false;
            }
            let i = e.iter().position(|&x| x > 0).expect("non-constant");
            let mut parent = e.clone();
            parent[i] -= 1;
            let (pv, pden) = &self.memo[&parent];
            let (cols, den) = &self.int_cols[i];
            let mut v = vec![Integer::new(); self.d];
            let mut t = Integer::new();
            for (j, x) in pv.iter().enumerate() {
                if *x == 0 {
                    continue;
                }
                for (k, a) in &cols[j] {
                    t.assign(x * a);
                    v[*k] += &t;
                }
            }
            let mut dn = Integer::from(pden * den);
            let mut g = dn.clone();
            for x in &v {
                if g == 1 {
                    break;
                }
                g.gcd_mut(x);
            }
            if g != 1 {
                for x in &mut v {
                    x.div_exact_mut(&g);
                }
                dn.div_exact_mut(&g);
            }
            self.memo.insert(e, (v, dn));
        }
        true
    }
}

/// Quotient dimension from which [`verify_lex`] checks one element per job.
const PARALLEL_VERIFY_MIN_DIM: usize = 32;

/// `Some(true)` when every polynomial of `cand` lies in `I` (its normal form,
/// assembled from exact coordinate vectors, vanishes) and the lex staircase of
/// `cand` has exactly `dim ℚ[x]/I` monomials.  Together these make `cand` a
/// lex Gröbner basis of `I`: `⟨lt(cand)⟩ ⊆ lt(I)`, and both have the same
/// finite codimension.  `None` when an armed budget fired.
pub(crate) fn verify_lex(cand: &[GbPoly], mats: &MulMatrices, n: usize) -> Option<bool> {
    if mats.partial {
        // Only `x_last^k` and the single variables have coordinates here.
        let readable = |e: &Vec<u32>| {
            e[..n - 1].iter().all(|&x| x == 0) || (e[n - 1] == 0 && e.iter().sum::<u32>() == 1)
        };
        if !cand.iter().all(|g| g.terms.keys().all(readable)) {
            return Some(false);
        }
    }
    let mut leads = Vec::with_capacity(cand.len());
    for g in cand {
        let Some(l) = g.leading_exp(MonomialOrder::Lex) else {
            return Some(false);
        };
        leads.push(l);
    }
    if staircase_size(&leads, n, mats.d) != Some(mats.d) {
        return Some(false);
    }
    let mut ev = ExactVectors::new(mats);
    // The coordinate vectors first (each builds on earlier ones), then one
    // element per job.
    for g in cand {
        for m in g.terms.keys() {
            if !ev.ensure(m) {
                return None;
            }
        }
    }
    let ev = &ev;
    let check = |g: &GbPoly| {
        // Σ_m c_m·vec(m) with c_m = a_m/b_m and vec(m) = w_m/e_m: multiply
        // through by B·E, B and E the lcms of the b_m and of the e_m, and
        // accumulate (B/b_m)·a_m·(E/e_m)·w_m over ℤ.  The b_m of one basis
        // element are mostly equal, so the lcms are taken only when a
        // divisibility test fails (no gcd per term).
        let mut big_b = Integer::from(1);
        let mut big_e = Integer::from(1);
        for (m, c) in &g.terms {
            if !big_b.is_divisible(c.denom()) {
                big_b.lcm_mut(c.denom());
            }
            let e = &ev.memo[m].1;
            if !big_e.is_divisible(e) {
                big_e.lcm_mut(e);
            }
        }
        let mut acc = vec![Integer::new(); mats.d];
        for (m, c) in &g.terms {
            let mut s = Integer::from(big_b.div_exact_ref(c.denom()));
            s *= c.numer();
            s *= Integer::from(big_e.div_exact_ref(&ev.memo[m].1));
            for (a, x) in acc.iter_mut().zip(&ev.memo[m].0) {
                if *x != 0 {
                    *a += &s * x;
                }
            }
        }
        acc.iter().all(|x| *x == 0)
    };
    // Small bases are checked in place: a parallel job costs more than they do.
    if mats.d < PARALLEL_VERIFY_MIN_DIM {
        return Some(cand.iter().all(check));
    }
    let in_ideal = run_jobs(cand.iter().collect(), check).ok()?;
    Some(in_ideal.into_iter().all(|z| z))
}

/// The reduced lex basis of the zero-dimensional ideal whose reduced grevlex
/// basis is `g` (with `staircase`, sorted increasingly, as its standard
/// monomials): FGLM modulo primes, lifted, and verified exactly against `g`.
/// Sorted by increasing leading monomial.
///
/// With `modular`, the lift is checked at further primes instead
/// ([`super::GroebnerVerification::Modular`]).
pub(crate) fn fglm_lex(
    g: &[GbPoly],
    n: usize,
    staircase: &[Vec<u32>],
    modular: bool,
) -> Result<Vec<GbPoly>, Unavailable> {
    let d = staircase.len();
    if d == 0 || staircase[0].iter().any(|&e| e != 0) {
        return Err(Unavailable::Fallback);
    }
    let idx: HashMap<Vec<u32>, usize> = staircase
        .iter()
        .enumerate()
        .map(|(i, e)| (e.clone(), i))
        .collect();
    match super::shape::shape_lex(g, n, staircase, &idx, modular) {
        Ok(Some(lex)) => return Ok(lex),
        Ok(None) => {}
        Err(Unavailable::Interrupted) => return Err(Unavailable::Interrupted),
        // Not conclusive in shape position: the general conversion below.
        Err(Unavailable::Fallback) => {}
    }
    fglm_lex_general(g, n, staircase, &idx, modular)
}

/// [`fglm_lex`] without the shape-position path: the general FGLM traversal
/// modulo each prime.
pub(crate) fn fglm_lex_general(
    g: &[GbPoly],
    n: usize,
    staircase: &[Vec<u32>],
    idx: &HashMap<Vec<u32>, usize>,
    modular: bool,
) -> Result<Vec<GbPoly>, Unavailable> {
    let d = staircase.len();
    let Some(mats) = MulMatrices::new(g, n, staircase, idx, false) else {
        return Err(if crate::budget::interrupt_pending() {
            Unavailable::Interrupted
        } else {
            Unavailable::Fallback
        });
    };
    let mut table = MonoTable::new(n, MonomialOrder::Lex);
    lift_loop(
        &mut table,
        modular,
        |t, pr| {
            let Some(mul) = mats.mod_p(pr) else {
                return Ok(None);
            };
            match fglm_mod_p(t, &mul, d, n, pr) {
                Ok(b) => Ok(Some(b)),
                Err(Stop::Interrupted) => Err(Unavailable::Interrupted),
                // A rank defect: a bad prime (or an input that is not a
                // Gröbner basis, which no prime will get past).
                Err(Stop::Overflow) => Ok(None),
            }
        },
        |cand| {
            if modular {
                Some(true)
            } else {
                verify_lex(cand, &mats, n)
            }
        },
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn rational_reconstruction_recovers_small_fractions() {
        let m = Integer::from(2147483647u64) * Integer::from(2147483629u64);
        let mut bound = Integer::from(&m >> 1u32);
        bound.sqrt_mut();
        for (n, d) in [(1i64, 3i64), (-7, 11), (123456, 7891), (0, 1), (-1, 1)] {
            let x = Rational::from((n, d));
            // r ≡ n·d⁻¹ (mod m)
            let dinv = Integer::from(d).invert(&m).unwrap();
            let r = (Integer::from(n) * dinv).modulo(&m);
            assert_eq!(rational_reconstruction(&r, &m, &bound), Some(x));
        }
    }

    /// A lift that fails verification is not proposed again until the
    /// group has twice the primes.
    #[test]
    fn rejected_lift_waits_for_more_primes() {
        let third = Rational::from((1, 3));
        let img = |pr: Prime| Image {
            shape: vec![vec![0]],
            coefs: vec![rational_mod_p(&third, pr).unwrap()],
        };
        let mut lifter = Lifter::new();
        let mut ps = RandomPrimes::new();
        let (p1, p2) = (ps.next().unwrap(), ps.next().unwrap());
        assert!(lifter.add(img(p1), p1).is_none());
        let (_, lift) = lifter.add(img(p2), p2).expect("second prime agrees");
        assert_eq!(lift, vec![third]);
        lifter.reject_candidate();
        assert_eq!(lifter.groups[0].next_try, 2 * lifter.groups[0].n_primes);
        assert!(lifter.proposed.is_none());
    }

    #[test]
    fn random_primes_are_distinct_primes_in_range() {
        let ps: Vec<u64> = RandomPrimes::new().take(50).map(|p| p.p).collect();
        let mut s = ps.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), ps.len());
        assert!(ps.iter().all(|&p| (1 << 30..1 << 31).contains(&p)));
    }

    /// Primes are drawn, not walked: a run of consecutive primes from one random
    /// start could be made unlucky all at once by a single large coefficient.
    #[test]
    fn random_primes_are_not_a_consecutive_run() {
        let mut ps: Vec<u64> = RandomPrimes::new().take(50).map(|p| p.p).collect();
        ps.sort_unstable();
        let spread = ps[49] - ps[0];
        // 50 uniform draws from an interval of length 2³⁰ span most of it, where
        // 50 consecutive primes would span about 50·ln(2³¹) ≈ 1100.
        assert!(spread > 1 << 28, "spread {spread}");
    }

    /// A shape that has been outvoted does not lift, even when one of its own
    /// primes agrees with the candidate it left pending.
    #[test]
    fn a_minority_shape_does_not_lift() {
        let bad = |pr: Prime| Image {
            shape: vec![vec![0]],
            coefs: vec![rational_mod_p(&Rational::from(1), pr).unwrap()],
        };
        // A coefficient too large to reconstruct from a few primes, so the good
        // group gets votes without replacing the bad group's pending candidate.
        let huge = Rational::from((Integer::from(Integer::u_pow_u(10, 200)) + 1, 3));
        let good = |pr: Prime| Image {
            shape: vec![vec![1, 0]],
            coefs: vec![1, rational_mod_p(&huge, pr).unwrap()],
        };
        let mut lifter = Lifter::new();
        let mut ps = RandomPrimes::new();
        let mut next = || ps.next().unwrap();
        let p = next();
        assert!(lifter.add(bad(p), p).is_none());
        assert!(lifter.candidate.is_some(), "the bad shape proposes first");
        for _ in 0..3 {
            let p = next();
            assert!(lifter.add(good(p), p).is_none());
        }
        let p = next();
        assert!(lifter.add(bad(p), p).is_none(), "outvoted 3 to 2");
    }

    use crate::poly::groebner::buchberger::compute_buchberger_basis;
    use crate::poly::groebner::f5::tests::cyclic_system;
    use proptest::prelude::*;

    pub(crate) fn katsura_system(n: usize) -> Vec<GbPoly> {
        let nv = n + 1;
        let var = |l: i64| -> GbPoly {
            let l = l.unsigned_abs() as usize;
            if l <= n {
                let mut e = vec![0u32; nv];
                e[l] = 1;
                GbPoly::monomial(e, Rational::from(1))
            } else {
                GbPoly::zero(nv)
            }
        };
        let mut out = vec![];
        for m in 0..n as i64 {
            let mut s = GbPoly::zero(nv);
            for l in -(n as i64)..=(n as i64) {
                s = s.add(&var(l).mul(&var(m - l)));
            }
            out.push(s.sub(&var(m)));
        }
        let mut s = GbPoly::zero(nv);
        for l in -(n as i64)..=(n as i64) {
            s = s.add(&var(l));
        }
        out.push(s.sub(&GbPoly::constant(Rational::from(1), nv)));
        out
    }

    /// The bases as sorted lists of term maps (order of generators ignored).
    fn canon(b: &[GbPoly]) -> Vec<Vec<(Vec<u32>, Rational)>> {
        let mut v: Vec<Vec<(Vec<u32>, Rational)>> = b
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

    fn agrees_with_buchberger(gens: Vec<GbPoly>) {
        let order = MonomialOrder::GRevLex;
        let reference = compute_buchberger_basis(gens.clone(), order);
        match groebner_basis(&gens, order) {
            Ok(b) => {
                assert_eq!(canon(&b), canon(&reference));
                // Sorted by decreasing leading monomial.
                let leads: Vec<Vec<u32>> =
                    b.iter().map(|g| g.leading_exp(order).unwrap()).collect();
                assert!(leads
                    .windows(2)
                    .all(|w| order.cmp(&w[0], &w[1]) == std::cmp::Ordering::Greater));
            }
            Err(e) => panic!("multimodular path gave up: {e:?}"),
        }
    }

    /// Replaying a learned trace modulo another prime gives exactly the
    /// basis a full F4 run there gives.
    #[test]
    fn trace_replay_matches_full_runs() {
        use crate::poly::groebner::modp::groebner_mod_p;
        for gens in [cyclic_system(5), katsura_system(4), katsura_system(5)] {
            let mut table = MonoTable::new(gens[0].n_vars, MonomialOrder::GRevLex);
            let ints: Vec<IntPoly> = gens
                .iter()
                .map(|g| int_poly(&mut table, g).unwrap())
                .collect();
            let mut ps = RandomPrimes::new();
            let p1 = ps.next().unwrap();
            let c1 = coefs_mod_p(&ints, p1).unwrap();
            let (gb1, trace) =
                groebner_mod_p_learn(&mut table, &full_support_images(&ints, c1.clone()), p1)
                    .unwrap();
            assert_eq!(trace.apply(&c1, p1).unwrap(), gb1);
            for _ in 0..4 {
                let q = ps.next().unwrap();
                let c = coefs_mod_p(&ints, q).unwrap();
                let full =
                    groebner_mod_p(&mut table, &full_support_images(&ints, c.clone()), q).unwrap();
                assert_eq!(trace.apply(&c, q).unwrap(), full);
                // Two primes in one pass give what each gives alone.
                let [a, b] = trace.apply2([c.as_slice(), c1.as_slice()], [q, p1]);
                assert_eq!(a.unwrap(), full);
                assert_eq!(b.unwrap(), gb1);
            }
        }
    }

    /// A trace learned at a prime where a coefficient vanishes does not
    /// replay elsewhere: the divergence is reported, not papered over.
    #[test]
    fn trace_from_an_unlucky_prime_reports_a_mismatch() {
        let p = 2147483647u64;
        // ⟨x² + p·y, x·y − 1⟩: modulo p it is the unit ideal, over ℚ it is not.
        let f = GbPoly {
            terms: [
                (vec![2, 0], Rational::from(1)),
                (vec![0, 1], Rational::from(p)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let g = GbPoly {
            terms: [
                (vec![1, 1], Rational::from(1)),
                (vec![0, 0], Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let mut table = MonoTable::new(2, MonomialOrder::GRevLex);
        let ints: Vec<IntPoly> = [f.clone(), g.clone()]
            .iter()
            .map(|g| int_poly(&mut table, g).unwrap())
            .collect();
        let bad = Prime::new(p);
        let cb = coefs_mod_p(&ints, bad).unwrap();
        let (gb_bad, trace) =
            groebner_mod_p_learn(&mut table, &full_support_images(&ints, cb.clone()), bad).unwrap();
        let q = RandomPrimes::new().find(|q| q.p != p).unwrap();
        let c = coefs_mod_p(&ints, q).unwrap();
        assert_eq!(trace.apply(&c, q), Err(ApplyFail::Mismatch));
        // In a two-prime pass the diverging lane fails alone.
        let [a, b] = trace.apply2([cb.as_slice(), c.as_slice()], [bad, q]);
        assert_eq!(a, Ok(gb_bad));
        assert_eq!(b, Err(ApplyFail::Mismatch));
        // The driver still finds the basis over ℚ.
        agrees_with_buchberger(vec![f, g]);
    }

    #[test]
    fn matches_buchberger_on_benchmark_systems() {
        for n in 3..=5 {
            agrees_with_buchberger(cyclic_system(n));
        }
        for n in 1..=5 {
            agrees_with_buchberger(katsura_system(n));
        }
    }

    #[test]
    fn rational_coefficients_and_the_unit_ideal() {
        let p = |t: &[(&[u32], i64, i64)]| GbPoly {
            terms: t
                .iter()
                .map(|(e, n, d)| (e.to_vec(), Rational::from((*n, *d))))
                .collect(),
            n_vars: 2,
        };
        // (x²/3 + 5y/7 − 1/2, x·y − 2/9)
        agrees_with_buchberger(vec![
            p(&[(&[2, 0], 1, 3), (&[0, 1], 5, 7), (&[0, 0], -1, 2)]),
            p(&[(&[1, 1], 1, 1), (&[0, 0], -2, 9)]),
        ]);
        // x + y, x + y − 1  ⟹  ⟨1⟩
        agrees_with_buchberger(vec![
            p(&[(&[1, 0], 1, 1), (&[0, 1], 1, 1)]),
            p(&[(&[1, 0], 1, 1), (&[0, 1], 1, 1), (&[0, 0], -1, 1)]),
        ]);
        // Large coefficients: (x − 10³⁰/7, y² − x)
        let big = Rational::from((Integer::from(Integer::u_pow_u(10, 30)), Integer::from(7)));
        let mut f = p(&[(&[1, 0], 1, 1)]);
        f.terms.insert(vec![0, 0], -big);
        agrees_with_buchberger(vec![f, p(&[(&[0, 2], 1, 1), (&[1, 0], -1, 1)])]);
    }

    fn lex_via_fglm(gens: &[GbPoly]) -> Result<Vec<GbPoly>, Unavailable> {
        let n = gens[0].n_vars;
        let g = groebner_basis(gens, MonomialOrder::GRevLex)?;
        let st = crate::poly::groebner::fglm::grevlex_staircase(&g, n).expect("zero-dimensional");
        fglm_lex(&g, n, &st, false)
    }

    #[test]
    fn fglm_matches_direct_lex_buchberger() {
        // Direct lex Buchberger is too slow beyond katsura-3; there the exact
        // rational FGLM is the reference.
        for n in 1..=4 {
            let gens = katsura_system(n);
            let lex = lex_via_fglm(&gens).expect("modular FGLM");
            let reference = if n <= 3 {
                compute_buchberger_basis(gens, MonomialOrder::Lex)
            } else {
                let g = groebner_basis(&gens, MonomialOrder::GRevLex).unwrap();
                let st = crate::poly::groebner::fglm::grevlex_staircase(&g, n + 1).unwrap();
                crate::poly::groebner::fglm::fglm_exact(&g, n + 1, st).unwrap()
            };
            assert_eq!(canon(&lex), canon(&reference), "katsura-{n}");
            let leads: Vec<Vec<u32>> = lex
                .iter()
                .map(|g| g.leading_exp(MonomialOrder::Lex).unwrap())
                .collect();
            assert!(leads.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn verify_lex_rejects_a_wrong_candidate() {
        let gens = katsura_system(3);
        let g = groebner_basis(&gens, MonomialOrder::GRevLex).unwrap();
        let st = crate::poly::groebner::fglm::grevlex_staircase(&g, 4).unwrap();
        let idx: HashMap<Vec<u32>, usize> =
            st.iter().enumerate().map(|(i, e)| (e.clone(), i)).collect();
        let mats = MulMatrices::new(&g, 4, &st, &idx, false).unwrap();
        let mut lex = fglm_lex(&g, 4, &st, false).unwrap();
        assert_eq!(verify_lex(&lex, &mats, 4), Some(true));
        // Perturb one coefficient: no longer in the ideal.
        let last = lex.last_mut().unwrap();
        let k = last.terms.keys().next().unwrap().clone();
        *last.terms.get_mut(&k).unwrap() += 1;
        assert_eq!(verify_lex(&lex, &mats, 4), Some(false));
        // Drop an element: the staircase is too big.
        let mut lex = fglm_lex(&g, 4, &st, false).unwrap();
        lex.pop();
        assert_eq!(verify_lex(&lex, &mats, 4), Some(false));
    }

    #[test]
    fn verify_over_q_rejects_a_non_basis() {
        let gens = cyclic_system(4);
        let g = groebner_basis(&gens, MonomialOrder::GRevLex).unwrap();
        assert_eq!(verify_over_q(&gens, &g, MonomialOrder::GRevLex), Some(true));
        let mut short = g.clone();
        short.pop();
        assert_eq!(
            verify_over_q(&gens, &short, MonomialOrder::GRevLex),
            Some(false)
        );
        let mut bad = g.clone();
        let last = bad.last_mut().unwrap();
        let k = last.terms.keys().next().unwrap().clone();
        *last.terms.get_mut(&k).unwrap() += 1;
        assert_eq!(
            verify_over_q(&gens, &bad, MonomialOrder::GRevLex),
            Some(false)
        );
    }

    /// The table-based check agrees with the term-by-term one, both ways, on
    /// a true basis, a basis missing an element, and a perturbed one.
    #[test]
    fn verify_lift_agrees_with_verify_over_q() {
        use crate::poly::groebner::qverify::verify_lift;
        for gens in [cyclic_system(4), cyclic_system(5), katsura_system(4)] {
            let order = MonomialOrder::GRevLex;
            let g = groebner_basis(&gens, order).unwrap();
            let mut short = g.clone();
            short.remove(short.len() / 2);
            let mut bad = g.clone();
            let last = bad.last_mut().unwrap();
            let k = last.terms.keys().next().unwrap().clone();
            *last.terms.get_mut(&k).unwrap() += 1;
            for (cand, want) in [(g, true), (short, false), (bad, false)] {
                assert_eq!(verify_over_q(&gens, &cand, order), Some(want));
                let mut table = MonoTable::new(gens[0].n_vars, order);
                let ints: Vec<IntPoly> = gens
                    .iter()
                    .map(|p| int_poly(&mut table, p).unwrap())
                    .collect();
                let mut shape: Shape = Vec::new();
                let mut lift: Vec<Rational> = Vec::new();
                for p in &cand {
                    let mut terms: Vec<(u32, Rational)> = p
                        .terms
                        .iter()
                        .map(|(e, c)| (table.intern_u32(e).unwrap(), c.clone()))
                        .collect();
                    terms.sort_by(|a, b| table.cmp(b.0, a.0));
                    shape.push(terms.iter().map(|t| t.0).collect());
                    lift.extend(terms.into_iter().map(|t| t.1));
                }
                assert_eq!(verify_lift(&mut table, &ints, &shape, &lift), Ok(want));
            }
        }
    }

    /// Every exponent vector in `n` variables of total degree at most 3.
    fn monomials_up_to_degree_3(n: usize) -> Vec<Vec<u32>> {
        let mut out = vec![vec![0u32; n]];
        for _ in 0..3 {
            let mut next = out.clone();
            for e in &out {
                for i in 0..n {
                    let mut f = e.clone();
                    f[i] += 1;
                    if !next.contains(&f) {
                        next.push(f);
                    }
                }
            }
            out = next;
        }
        out
    }

    /// Random ideals: 2–4 generators in 2–3 variables, total degree at most
    /// 3, each the sum of up to five terms `c/d · x^e` (`|c| <= 9`,
    /// `1 <= d <= 4`, exponents drawn per variable from `0..=3` and the term
    /// dropped when its total degree exceeds 3).
    ///
    /// That generator makes many zero polynomials, and now and then a whole
    /// ideal of them. The zero ideal used to be rejected with `prop_assume!`;
    /// under the nightly's `PROPTEST_CASES=50000` it turned up more often than
    /// proptest's 1024 global rejects allow ("Too many global rejects"). It is
    /// now repaired instead of rejected: an all-zero draw has one generator
    /// (chosen by the draw) replaced with a non-zero monomial of degree at
    /// most 3. Every ideal the old strategy accepted is still drawn with the
    /// same probability and nothing is filtered, so the property is exactly
    /// as strong as before — and also covers the one-monomial ideals the
    /// repair adds. 50 000 cases take about 4 s in a release build. (Drawing
    /// non-zero generators directly, on distinct monomials, was tried: the
    /// denser ideals made 2 000 cases take over 8 minutes.)
    fn small_system() -> impl Strategy<Value = Vec<GbPoly>> {
        (2usize..=3, 2usize..=4).prop_flat_map(|(n, m)| {
            let term = (proptest::collection::vec(0u32..=3, n), -9i64..=9, 1i64..=4);
            let gen = proptest::collection::vec(term, 1..=5).prop_map(move |ts| {
                let mut p = GbPoly::zero(n);
                for (e, c, d) in ts {
                    if e.iter().sum::<u32>() <= 3 {
                        p = p.add(&GbPoly::monomial(e, Rational::from((c, d))));
                    }
                }
                p
            });
            let witness = (
                0..m,
                proptest::sample::select(monomials_up_to_degree_3(n)),
                prop_oneof![-9i64..=-1, 1i64..=9],
                1i64..=4,
            );
            (proptest::collection::vec(gen, m), witness).prop_map(|(mut gens, (i, e, c, d))| {
                if gens.iter().all(|g| g.is_zero()) {
                    gens[i] = GbPoly::monomial(e, Rational::from((c, d)));
                }
                gens
            })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn random_ideals_match_buchberger(gens in small_system()) {
            // Repaired, not rejected: see `small_system`.
            prop_assert!(gens.iter().any(|g| !g.is_zero()));
            agrees_with_buchberger(gens);
        }
    }
}
