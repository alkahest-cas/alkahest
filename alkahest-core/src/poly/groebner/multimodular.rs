//! Multimodular Gröbner bases over ℚ (the modStd / msolve design).
//!
//! The reduced Gröbner basis `G` of an ideal `I ⊂ ℚ[x]` is unique, and for all
//! but finitely many primes `p` its image modulo `p` is the reduced basis of
//! the image ideal, which [`super::modp`] computes in machine words.  So:
//!
//! 1. compute the reduced basis modulo random primes `p < 2³¹`;
//! 2. group the images by *shape* (the support of every element) — an unlucky
//!    prime shows up as a shape the others do not share, and the majority
//!    shape wins;
//! 3. combine the majority images coefficient-wise by the Chinese remainder
//!    theorem and lift by rational reconstruction; accept a lift once one more
//!    prime agrees with it;
//! 4. **verify the lift over ℚ** before returning it: every input reduces to
//!    zero modulo the candidate `B` (so `I ⊆ ⟨B⟩`) and every critical pair of
//!    `B` that survives the Gebauer–Möller criteria reduces to zero (so `B` is
//!    a Gröbner basis).  A candidate that fails is not returned; more primes are
//!    taken, and after a few failures the caller falls back to the ℚ
//!    algorithm.
//!
//! What the verification proves: `B` is the reduced Gröbner basis of an ideal
//! containing `I`.  For a **homogeneous** input that ideal is `I` itself
//! (Arnold, *Modular algorithms for computing Gröbner bases*, JSC 35 (2003),
//! Thm 7.1: a basis with the leading terms of the modular image of the input
//! ideal cannot generate a strictly larger homogeneous ideal).  For an
//! inhomogeneous input, `⟨B⟩ ⊋ I` would need *every* prime used — drawn at
//! random from the ~10⁸ primes in `[2³⁰, 2³¹)` — to be unlucky in the same way;
//! the unlucky primes of an ideal are the divisors of a fixed non-zero integer,
//! so this has negligible probability for every input, adversarial or not.
//!
//! Everything here is crate-private; [`super::GroebnerBasis::compute`] and
//! [`super::fglm`] use it behind their unchanged signatures.

use std::collections::{BTreeMap, HashMap};

use rug::{Assign, Integer, Rational};

use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::modp::{groebner_mod_p, is_prime_u32, ModPoly, MonoTable, Prime, Stop};
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

/// Primes in `[2³⁰, 2³¹)`, walking down from a random starting point (and
/// wrapping), so no fixed input can make every prime unlucky.
pub(crate) struct RandomPrimes {
    next: u64,
    wrapped: bool,
}

impl RandomPrimes {
    pub(crate) fn new() -> Self {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(0x9e37_79b9_7f4a_7c15);
        let r = h.finish();
        let lo = 1u64 << 30;
        let next = (lo + r % lo) | 1;
        RandomPrimes {
            next,
            wrapped: false,
        }
    }
}

impl Iterator for RandomPrimes {
    type Item = Prime;
    fn next(&mut self) -> Option<Prime> {
        let lo = 1u64 << 30;
        loop {
            if self.next < lo {
                if self.wrapped {
                    return None;
                }
                self.wrapped = true;
                self.next = (1u64 << 31) - 1;
            }
            let n = self.next;
            self.next -= 2;
            if is_prime_u32(n) {
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

/// The images of `gens` modulo `p`; `None` when `p` divides a leading
/// coefficient (the image would have a different leading term).
pub(crate) fn image_mod_p(gens: &[IntPoly], pr: Prime) -> Option<Vec<ModPoly>> {
    let mut out = Vec::with_capacity(gens.len());
    for g in gens {
        let mut mons = Vec::with_capacity(g.mons.len());
        let mut coefs = Vec::with_capacity(g.mons.len());
        for (k, (&m, c)) in g.mons.iter().zip(&g.coefs).enumerate() {
            let r = u64::from(c.mod_u(pr.p as u32));
            if r == 0 {
                if k == 0 {
                    return None;
                }
                continue;
            }
            mons.push(m);
            coefs.push(r);
        }
        out.push(ModPoly { mons, coefs });
    }
    Some(out)
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
fn rational_mod_p(x: &Rational, pr: Prime) -> Option<u64> {
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
}

impl Lifter {
    pub(crate) fn new() -> Self {
        Lifter {
            groups: Vec::new(),
            candidate: None,
            proposed: None,
            last_fail: 0,
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
                if agrees {
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
            let next = (g.n_primes * 3).div_ceil(2).max(g.n_primes + 1);
            self.groups[gi].next_try = next;
            if let Some(lift) = self.reconstruct(gi) {
                self.candidate = Some((gi, lift));
            }
        }
        None
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
        let mut bound = Integer::from(m >> 1u32);
        bound.sqrt_mut();
        let half = Integer::from(m >> 1u32);
        let m_bits = m.significant_bits();
        let lift = |r: &Integer, den: &mut Integer| -> Option<Rational> {
            let mut t = Integer::from(r * &*den);
            t %= m;
            if t > half {
                t -= m;
            }
            if t.significant_bits() + 64 < m_bits {
                return Some(Rational::from((t, den.clone())));
            }
            let t = if t < 0 { t + m } else { t };
            let x = rational_reconstruction(&t, m, &bound)?;
            let d = x.denom().clone();
            let out = Rational::from((x.numer().clone(), Integer::from(&d * &*den)));
            *den *= d;
            Some(out)
        };
        // The coefficient that failed last time usually fails again: try it
        // first.  Then start from scratch: a lift that went wrong at a lower
        // precision must not leave a spurious factor in `den`.
        if let Some(r) = g.residues.get(self.last_fail) {
            lift(r, &mut Integer::from(1))?;
        }
        let mut den = Integer::from(1);
        let mut out = Vec::with_capacity(g.residues.len());
        for (k, r) in g.residues.iter().enumerate() {
            match lift(r, &mut den) {
                Some(x) => out.push(x),
                None => {
                    self.last_fail = k;
                    return None;
                }
            }
        }
        Some(out)
    }
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

    lift_loop(
        &mut table,
        |t, pr| {
            let Some(img) = image_mod_p(&ints, pr) else {
                return Ok(None);
            };
            match groebner_mod_p(t, &img, pr) {
                Ok(gb) => Ok(Some(gb)),
                Err(Stop::Interrupted) => Err(Unavailable::Interrupted),
                Err(Stop::Overflow) => Err(Unavailable::Fallback),
            }
        },
        |cand| verify_over_q(&gens, cand, order),
    )
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
fn run_jobs<T: Send, R: Send>(jobs: Vec<T>, f: impl Fn(T) -> R + Sync) -> Result<Vec<R>, ()> {
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
fn batch_width() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().clamp(1, 16)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// What one prime gave: a basis image (monomials in the table it was computed
/// with), nothing usable (a bad prime), or a reason to stop.
type PrimeResult = Result<Option<Vec<ModPoly>>, Unavailable>;

/// The multimodular loop shared by the grevlex and FGLM paths: compute images
/// modulo successive random primes with `compute` (the first alone, then in
/// parallel batches on clones of `table`), lift the majority shape, and hand
/// each lift to `verify` until one passes.
fn lift_loop(
    table: &mut MonoTable,
    compute: impl Fn(&mut MonoTable, Prime) -> PrimeResult + Sync,
    mut verify: impl FnMut(&[GbPoly]) -> Option<bool>,
) -> Result<Vec<GbPoly>, Unavailable> {
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
            let Some((shape, lift)) = lifter.add(Image::from_polys(gb), p) else {
                continue;
            };
            let cand = polys_from_lift(table, &shape, lift);
            match verify(&cand) {
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

// ---------------------------------------------------------------------------
// FGLM: grevlex → lex for a zero-dimensional ideal
// ---------------------------------------------------------------------------

/// Where `x_i · s` lands, for a staircase monomial `s`: another staircase
/// monomial, or (on the border) the normal form, a sparse rational vector.
enum MulCol {
    Std(usize),
    Nf(Vec<(usize, Rational)>),
}

/// A multiplication matrix modulo `p`, column by column (see [`MulCol`]).
type ModMulMatrix = Vec<Result<usize, Vec<(usize, u64)>>>;

/// The multiplication matrices of `ℚ[x]/I` on the grevlex staircase basis:
/// `mul[i][j]` is the normal form of `x_i · staircase[j]`.
struct MulMatrices {
    mul: Vec<Vec<MulCol>>,
    d: usize,
}

impl MulMatrices {
    fn new(
        g: &[GbPoly],
        n: usize,
        staircase: &[Vec<u32>],
        idx: &HashMap<Vec<u32>, usize>,
    ) -> Option<Self> {
        let divs = Divisors::new(g, MonomialOrder::GRevLex);
        let d = staircase.len();
        let mut mul = Vec::with_capacity(n);
        for i in 0..n {
            let mut cols = Vec::with_capacity(d);
            for s in staircase {
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
        Some(MulMatrices { mul, d })
    }

    /// The matrices modulo `p`; `None` if `p` divides a denominator.
    fn mod_p(&self, pr: Prime) -> Option<Vec<ModMulMatrix>> {
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

/// `Some(true)` when every polynomial of `cand` lies in `I` (its normal form,
/// assembled from exact coordinate vectors, vanishes) and the lex staircase of
/// `cand` has exactly `dim ℚ[x]/I` monomials.  Together these make `cand` a
/// lex Gröbner basis of `I`: `⟨lt(cand)⟩ ⊆ lt(I)`, and both have the same
/// finite codimension.  `None` when an armed budget fired.
fn verify_lex(cand: &[GbPoly], mats: &MulMatrices, n: usize) -> Option<bool> {
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
    for g in cand {
        // Σ_m c_m·vec(m) with vec(m) = w_m/den_m: scale every c_m/den_m to
        // the common denominator and accumulate over ℤ (no gcds).
        let mut factors = Vec::with_capacity(g.terms.len());
        let mut common = Integer::from(1);
        for (m, c) in &g.terms {
            if !ev.ensure(m) {
                return None;
            }
            let f = Rational::from(c / &ev.memo[m].1);
            common.lcm_mut(f.denom());
            factors.push((m, f));
        }
        let mut acc = vec![Integer::new(); mats.d];
        let mut t = Integer::new();
        for (m, f) in factors {
            let mut s = Integer::from(common.div_exact_ref(f.denom()));
            s *= f.numer();
            for (a, x) in acc.iter_mut().zip(&ev.memo[m].0) {
                if *x != 0 {
                    t.assign(&s * x);
                    *a += &t;
                }
            }
        }
        if acc.iter().any(|x| *x != 0) {
            return Some(false);
        }
    }
    Some(true)
}

/// The reduced lex basis of the zero-dimensional ideal whose reduced grevlex
/// basis is `g` (with `staircase`, sorted increasingly, as its standard
/// monomials): FGLM modulo primes, lifted, and verified exactly against `g`.
/// Sorted by increasing leading monomial.
pub(crate) fn fglm_lex(
    g: &[GbPoly],
    n: usize,
    staircase: &[Vec<u32>],
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
    let Some(mats) = MulMatrices::new(g, n, staircase, &idx) else {
        return Err(if crate::budget::interrupt_pending() {
            Unavailable::Interrupted
        } else {
            Unavailable::Fallback
        });
    };
    let mut table = MonoTable::new(n, MonomialOrder::Lex);
    lift_loop(
        &mut table,
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
        |cand| verify_lex(cand, &mats, n),
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
        fglm_lex(&g, n, &st)
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
        let mats = MulMatrices::new(&g, 4, &st, &idx).unwrap();
        let mut lex = fglm_lex(&g, 4, &st).unwrap();
        assert_eq!(verify_lex(&lex, &mats, 4), Some(true));
        // Perturb one coefficient: no longer in the ideal.
        let last = lex.last_mut().unwrap();
        let k = last.terms.keys().next().unwrap().clone();
        *last.terms.get_mut(&k).unwrap() += 1;
        assert_eq!(verify_lex(&lex, &mats, 4), Some(false));
        // Drop an element: the staircase is too big.
        let mut lex = fglm_lex(&g, 4, &st).unwrap();
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
            proptest::collection::vec(gen, m)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn random_ideals_match_buchberger(gens in small_system()) {
            prop_assume!(gens.iter().any(|g| !g.is_zero()));
            agrees_with_buchberger(gens);
        }
    }
}
