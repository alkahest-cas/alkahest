//! Gröbner bases over a word-size prime field 𝔽_p, by a matrix F4.
//!
//! This is the engine behind the multimodular strategy in
//! [`super::multimodular`]: the reduced Gröbner basis of an ideal over ℚ is
//! recovered from its images modulo several primes `p < 2³¹`, so all the work
//! that dominates a Gröbner computation — the S-pair reductions, most of which
//! reduce to zero — runs in machine words instead of growing rationals.
//!
//! The algorithm is Faugère's F4 (1999) in its plain form:
//!
//! * **Pair selection** — every critical pair of minimal lcm degree (the
//!   normal strategy) is taken at once; the pair list is kept by the
//!   Gebauer–Möller code of the ℚ Buchberger engine
//!   ([`super::pairs::update_pairs_live`]).
//! * **Symbolic preprocessing** — both halves of each selected pair become rows
//!   of a matrix; every monomial of the matrix divisible by a leading monomial
//!   of the basis gets a reducer row `t·g` (one per monomial).
//! * **Linear algebra** — columns sorted by the monomial order; the reducer
//!   rows are pivots, the remaining rows are reduced left to right against
//!   them (and against each other) in a dense accumulator.  Entries are kept
//!   below `p²` and folded with one comparison, so the inner loop has no
//!   division.  Rows that survive have leading monomials outside the current
//!   leading-term ideal and join the basis.
//!
//! The output is the reduced basis (minimal, tail-reduced, monic).  Everything
//! here is crate-private: it is an implementation strategy, not API.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::pairs::{update_pairs_live, CriticalPair};

type Hasher = foldhash::fast::RandomState;

// ---------------------------------------------------------------------------
// Prime field arithmetic
// ---------------------------------------------------------------------------

/// A prime `2 < p < 2³¹`, so a product of two residues fits in 62 bits and a
/// sum of such a product with a value below `p²` fits in a `u64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Prime {
    pub(crate) p: u64,
    p2: u64,
    /// `⌊(2⁶⁴ − 1)/p⌋`, for [`Self::reduce`].
    bar: u64,
}

impl Prime {
    pub(crate) fn new(p: u64) -> Self {
        debug_assert!(p > 2 && p < (1 << 31) && is_prime_u32(p));
        Prime {
            p,
            p2: p * p,
            bar: u64::MAX / p,
        }
    }

    /// `v mod p` for `v < 2⁶²` (so any value kept below `p²`), by Barrett
    /// reduction: the quotient estimate `⌊v·bar/2⁶⁴⌋` is at most one short
    /// (`v·bar/2⁶⁴ > v/p − v/2⁶⁴ − 1`, and `v/2⁶⁴ < ¼`), so one correction
    /// suffices.  A multiplication instead of a 64-bit division.
    #[inline(always)]
    pub(crate) fn reduce(self, v: u64) -> u64 {
        debug_assert!(v < 1 << 62);
        let q = ((v as u128 * self.bar as u128) >> 64) as u64;
        let r = v - q * self.p;
        if r >= self.p {
            r - self.p
        } else {
            r
        }
    }

    #[inline]
    pub(crate) fn mul(self, a: u64, b: u64) -> u64 {
        a * b % self.p
    }

    /// Inverse of a non-zero residue.
    pub(crate) fn inv(self, a: u64) -> u64 {
        let (mut r0, mut r1) = (self.p as i64, (a % self.p) as i64);
        let (mut t0, mut t1) = (0i64, 1i64);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0 - q * t1);
        }
        debug_assert_eq!(r0, 1, "inverse of a non-unit");
        t0.rem_euclid(self.p as i64) as u64
    }
}

fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1u64;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    r
}

/// Deterministic Miller–Rabin for `n < 2³²` (witnesses 2, 7, 61).
pub(crate) fn is_prime_u32(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for q in [2u64, 3, 5, 7, 11, 13, 61] {
        if n % q == 0 {
            return n == q;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'w: for a in [2u64, 7, 61] {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = x * x % n;
            if x == n - 1 {
                continue 'w;
            }
        }
        return false;
    }
    true
}

/// The primes below `2³¹`, largest first.
#[cfg(test)]
pub(crate) fn primes() -> impl Iterator<Item = Prime> {
    (3..(1u64 << 31))
        .rev()
        .step_by(2)
        .filter(|&n| is_prime_u32(n))
        .map(Prime::new)
}

// ---------------------------------------------------------------------------
// Monomials
// ---------------------------------------------------------------------------

/// Why a modular computation stopped without a basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    /// An exponent left the `u16` range of the monomial table.
    Overflow,
    /// An armed budget fired (see [`crate::budget::arm`]).
    Interrupted,
}

/// Interned monomials in `n` variables: each distinct exponent vector gets a
/// dense `u32` id, so polynomials are vectors of ids and products are one hash
/// lookup.
///
/// The hash of a monomial is linear in its exponents (`Σ wᵢ·eᵢ` for fixed
/// random weights), as in Groebner.jl and msolve: the hash of a product or
/// quotient is the sum or difference of two stored hashes, so `mul` and
/// `quo` find an existing monomial without forming its exponent vector.
#[derive(Clone)]
pub(crate) struct MonoTable {
    n: usize,
    order: MonomialOrder,
    exps: Vec<u16>,
    deg: Vec<u32>,
    /// Short exponent vector: a bit mask with `a | b ⟹ sev(a) ⊆ sev(b)`.
    sev: Vec<u64>,
    /// Linear hash of each monomial.
    hv: Vec<u64>,
    /// Hash weight of each variable.
    weights: Vec<u64>,
    /// Open addressing: monomial ids, [`EMPTY`] for a free slot; the length
    /// is a power of two, at least twice the number of monomials.
    slots: Vec<u32>,
    scratch: Vec<u16>,
    /// Scratch space for symbolic preprocessing: the column of a monomial id,
    /// and a per-pass mark (`stamp[m] == epoch`: seen in this pass).
    col_scratch: Vec<u32>,
    stamp: Vec<u32>,
    epoch: u32,
}

/// A free slot of [`MonoTable::slots`].
const EMPTY: u32 = u32::MAX;

/// Spread a linear hash over the slot index bits.
#[inline]
fn mix(h: u64) -> u64 {
    let h = (h ^ (h >> 32)).wrapping_mul(0xd6e8_feb8_6659_fd93);
    h ^ (h >> 32)
}

impl MonoTable {
    pub(crate) fn new(n: usize, order: MonomialOrder) -> Self {
        // Fixed weights (SplitMix64 from a constant): ids, and so everything
        // computed from them, are the same from run to run.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let weights = (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^ (z >> 31)
            })
            .collect();
        MonoTable {
            n,
            order,
            exps: Vec::new(),
            deg: Vec::new(),
            sev: Vec::new(),
            hv: Vec::new(),
            weights,
            slots: vec![EMPTY; 1024],
            scratch: vec![0; n],
            col_scratch: Vec::new(),
            stamp: Vec::new(),
            epoch: 0,
        }
    }

    /// Number of monomials interned so far (ids are `0..len`).
    pub(crate) fn len(&self) -> usize {
        self.deg.len()
    }

    pub(crate) fn n_vars(&self) -> usize {
        self.n
    }

    #[inline]
    pub(crate) fn exps(&self, m: u32) -> &[u16] {
        let i = m as usize * self.n;
        &self.exps[i..i + self.n]
    }

    #[inline]
    pub(crate) fn deg(&self, m: u32) -> u32 {
        self.deg[m as usize]
    }

    fn sev_of(&self, e: &[u16]) -> u64 {
        if self.n == 0 {
            return 0;
        }
        let bits = (64 / self.n).max(1);
        let mut s = 0u64;
        for (i, &x) in e.iter().enumerate() {
            let base = i * bits;
            if base >= 64 {
                break;
            }
            for k in 0..bits.min(64 - base) {
                if u32::from(x) > k as u32 {
                    s |= 1 << (base + k);
                }
            }
        }
        s
    }

    #[inline]
    fn hash_of(&self, e: &[u16]) -> u64 {
        e.iter().zip(&self.weights).fold(0u64, |h, (&x, &w)| {
            h.wrapping_add(w.wrapping_mul(u64::from(x)))
        })
    }

    /// The id of the monomial with hash `h` for which `eq(id)` holds, or the
    /// free slot where it would go.
    #[inline]
    fn probe(&self, h: u64, eq: impl Fn(u32) -> bool) -> Result<u32, usize> {
        let mask = self.slots.len() - 1;
        let mut i = mix(h) as usize & mask;
        loop {
            let id = self.slots[i];
            if id == EMPTY {
                return Err(i);
            }
            if self.hv[id as usize] == h && eq(id) {
                return Ok(id);
            }
            i = (i + 1) & mask;
        }
    }

    /// Add a new monomial (not present) with hash `h`, whose free slot is
    /// `slot`.
    fn insert_new(&mut self, e: &[u16], h: u64, slot: usize) -> u32 {
        let id = self.deg.len() as u32;
        self.exps.extend_from_slice(e);
        self.deg.push(e.iter().map(|&x| u32::from(x)).sum());
        let s = self.sev_of(e);
        self.sev.push(s);
        self.hv.push(h);
        self.slots[slot] = id;
        if 2 * self.deg.len() > self.slots.len() {
            self.grow();
        }
        id
    }

    fn grow(&mut self) {
        let cap = self.slots.len() * 2;
        let mut slots = vec![EMPTY; cap];
        let mask = cap - 1;
        for (id, &h) in self.hv.iter().enumerate() {
            let mut i = mix(h) as usize & mask;
            while slots[i] != EMPTY {
                i = (i + 1) & mask;
            }
            slots[i] = id as u32;
        }
        self.slots = slots;
    }

    pub(crate) fn intern(&mut self, e: &[u16]) -> u32 {
        let h = self.hash_of(e);
        let n = self.n;
        let exps = &self.exps;
        match self.probe(h, |id| {
            let i = id as usize * n;
            &exps[i..i + n] == e
        }) {
            Ok(id) => id,
            Err(slot) => self.insert_new(e, h, slot),
        }
    }

    /// Intern a `u32` exponent vector; `None` if an exponent exceeds `u16`.
    pub(crate) fn intern_u32(&mut self, e: &[u32]) -> Option<u32> {
        let mut v = std::mem::take(&mut self.scratch);
        v.clear();
        for &x in e {
            v.push(u16::try_from(x).ok()?);
        }
        let id = self.intern(&v);
        self.scratch = v;
        Some(id)
    }

    #[inline]
    pub(crate) fn mul(&mut self, a: u32, b: u32) -> Result<u32, Stop> {
        let h = self.hv[a as usize].wrapping_add(self.hv[b as usize]);
        let n = self.n;
        let (ia, ib) = (a as usize * n, b as usize * n);
        let d = self.deg[a as usize] + self.deg[b as usize];
        let (exps, degs) = (&self.exps, &self.deg);
        let found = self.probe(h, |id| {
            let i = id as usize * n;
            degs[id as usize] == d
                && (0..n).all(|k| {
                    u32::from(exps[i + k]) == u32::from(exps[ia + k]) + u32::from(exps[ib + k])
                })
        });
        match found {
            Ok(id) => Ok(id),
            Err(slot) => {
                let mut v = std::mem::take(&mut self.scratch);
                v.clear();
                for k in 0..n {
                    match self.exps[ia + k].checked_add(self.exps[ib + k]) {
                        Some(s) => v.push(s),
                        None => {
                            self.scratch = v;
                            return Err(Stop::Overflow);
                        }
                    }
                }
                let id = self.insert_new(&v, h, slot);
                self.scratch = v;
                Ok(id)
            }
        }
    }

    /// Start a new marking pass (see [`Self::mark`]).
    fn new_pass(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
    }

    /// Mark `m` in the current pass; `true` if it was not marked yet.
    #[inline]
    fn mark(&mut self, m: u32) -> bool {
        let i = m as usize;
        if i >= self.stamp.len() {
            self.stamp.resize((i + 1).max(self.deg.len()), 0);
        }
        if self.stamp[i] == self.epoch {
            return false;
        }
        self.stamp[i] = self.epoch;
        true
    }

    /// `b / a`, for `a | b`.
    pub(crate) fn quo(&mut self, b: u32, a: u32) -> u32 {
        let h = self.hv[b as usize].wrapping_sub(self.hv[a as usize]);
        let n = self.n;
        let (ia, ib) = (a as usize * n, b as usize * n);
        let d = self.deg[b as usize] - self.deg[a as usize];
        let (exps, degs) = (&self.exps, &self.deg);
        let found = self.probe(h, |id| {
            let i = id as usize * n;
            degs[id as usize] == d
                && (0..n).all(|k| {
                    u32::from(exps[i + k]) + u32::from(exps[ia + k]) == u32::from(exps[ib + k])
                })
        });
        match found {
            Ok(id) => id,
            Err(slot) => {
                let mut v = std::mem::take(&mut self.scratch);
                v.clear();
                for k in 0..n {
                    v.push(self.exps[ib + k] - self.exps[ia + k]);
                }
                let id = self.insert_new(&v, h, slot);
                self.scratch = v;
                id
            }
        }
    }

    #[inline]
    pub(crate) fn divides(&self, a: u32, b: u32) -> bool {
        if self.sev[a as usize] & !self.sev[b as usize] != 0 {
            return false;
        }
        self.exps(a).iter().zip(self.exps(b)).all(|(x, y)| x <= y)
    }

    /// The divisor mask of `m`: `a | b` implies `sev(a) ⊆ sev(b)`.
    #[inline]
    pub(crate) fn sev(&self, m: u32) -> u64 {
        self.sev[m as usize]
    }

    /// `a | b`, by exponents only (the caller has checked the masks).
    #[inline]
    fn divides_exps(&self, a: u32, b: u32) -> bool {
        self.exps(a).iter().zip(self.exps(b)).all(|(x, y)| x <= y)
    }

    /// Compare under the table's order.
    #[inline]
    pub(crate) fn cmp(&self, a: u32, b: u32) -> Ordering {
        if a == b {
            return Ordering::Equal;
        }
        let (ea, eb) = (self.exps(a), self.exps(b));
        match self.order {
            MonomialOrder::Lex => ea.cmp(eb),
            MonomialOrder::GrLex => self.deg(a).cmp(&self.deg(b)).then_with(|| ea.cmp(eb)),
            MonomialOrder::GRevLex => self.deg(a).cmp(&self.deg(b)).then_with(|| {
                for (x, y) in ea.iter().rev().zip(eb.iter().rev()) {
                    match y.cmp(x) {
                        Ordering::Equal => {}
                        c => return c,
                    }
                }
                Ordering::Equal
            }),
        }
    }

    pub(crate) fn exps_u32(&self, m: u32) -> Vec<u32> {
        self.exps(m).iter().map(|&x| u32::from(x)).collect()
    }
}

// ---------------------------------------------------------------------------
// Polynomials over 𝔽_p
// ---------------------------------------------------------------------------

/// A polynomial over 𝔽_p: monomial ids in strictly decreasing order, with
/// non-zero coefficients below `p`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModPoly {
    pub(crate) mons: Vec<u32>,
    pub(crate) coefs: Vec<u64>,
}

impl ModPoly {
    pub(crate) fn lead(&self) -> u32 {
        self.mons[0]
    }

    fn make_monic(&mut self, pr: Prime) {
        let lc = self.coefs[0];
        if lc != 1 {
            let inv = pr.inv(lc);
            for c in &mut self.coefs {
                *c = pr.mul(*c, inv);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// F4
// ---------------------------------------------------------------------------

struct Basis {
    polys: Vec<ModPoly>,
    lead: Vec<Vec<u32>>,
    sugar: Vec<u32>,
    /// Leading monomial divisible by a later element's: never a reducer, not
    /// part of the minimal basis.
    redundant: Vec<bool>,
    /// Leading monomial divisible by a *later* element's: out of the basis in
    /// Gebauer and Möller's sense, so no new pairs are formed with it.
    dropped: Vec<bool>,
    pairs: Vec<CriticalPair>,
    /// The non-redundant elements, with their leading monomials and divisor
    /// masks side by side, so the reducer search in symbolic preprocessing
    /// scans flat arrays instead of the whole basis.
    live: Vec<u32>,
    live_lead: Vec<u32>,
    live_sev: Vec<u64>,
}

impl Basis {
    fn new() -> Self {
        Basis {
            polys: Vec::new(),
            lead: Vec::new(),
            sugar: Vec::new(),
            redundant: Vec::new(),
            dropped: Vec::new(),
            pairs: Vec::new(),
            live: Vec::new(),
            live_lead: Vec::new(),
            live_sev: Vec::new(),
        }
    }

    fn add(&mut self, table: &MonoTable, g: ModPoly, sugar: u32) {
        let lm = g.lead();
        let mut live_changed = false;
        let mut newly_dropped = Vec::new();
        for (k, h) in self.polys.iter().enumerate() {
            if !self.dropped[k] && table.divides(lm, h.lead()) {
                newly_dropped.push(k);
                if !self.redundant[k] {
                    self.redundant[k] = true;
                    live_changed = true;
                }
            }
        }
        // An input generator can have a leading monomial divisible by an
        // earlier one's (elements found by F4 never do): it still takes part
        // in pairs, but it is not a reducer and not in the minimal basis.
        let covered = self
            .polys
            .iter()
            .enumerate()
            .any(|(k, h)| !self.redundant[k] && table.divides(h.lead(), lm));
        let idx = self.polys.len();
        self.lead.push(table.exps_u32(lm));
        self.sugar.push(sugar.max(table.deg(lm)));
        self.polys.push(g);
        self.redundant.push(covered);
        self.dropped.push(false);
        if live_changed {
            let mut w = 0;
            for r in 0..self.live.len() {
                if !self.redundant[self.live[r] as usize] {
                    self.live[w] = self.live[r];
                    self.live_lead[w] = self.live_lead[r];
                    self.live_sev[w] = self.live_sev[r];
                    w += 1;
                }
            }
            self.live.truncate(w);
            self.live_lead.truncate(w);
            self.live_sev.truncate(w);
        }
        if !covered {
            self.live.push(idx as u32);
            self.live_lead.push(lm);
            self.live_sev.push(table.sev[lm as usize]);
        }
        // No new pairs with an element an earlier addition made redundant:
        // such a pair is covered through that element.  (The elements `g`
        // makes redundant still pair with `g` itself, as in Gebauer and
        // Möller's UPDATE, which removes them only afterwards.)
        let dropped = &self.dropped;
        update_pairs_live(&self.lead, &self.sugar, &mut self.pairs, idx, |g| {
            !dropped[g]
        });
        for k in newly_dropped {
            self.dropped[k] = true;
        }
    }

    /// A non-redundant element whose leading monomial divides `m`, preferring
    /// the sparsest.
    fn reducer(&self, table: &MonoTable, m: u32) -> Option<usize> {
        let not_m = !table.sev[m as usize];
        let mut best: Option<usize> = None;
        for (i, &s) in self.live_sev.iter().enumerate() {
            if s & not_m != 0 || !table.divides_exps(self.live_lead[i], m) {
                continue;
            }
            let k = self.live[i] as usize;
            if best.map_or(true, |b| {
                self.polys[k].mons.len() < self.polys[b].mons.len()
            }) {
                best = Some(k);
            }
        }
        best
    }
}

/// No pivot in this column.
const NONE: u32 = u32::MAX;
/// `piv_of_col` entries at or above this are rows of the lower block (new
/// pivots found in the same step), the rest rows of the upper block.
const LOWER: u32 = 1 << 31;

/// A row of the symbolic matrix before columns are assigned:
/// `mons[i] = t·basis[k].mons[skip + i]`, coefficients `basis[k].coefs[skip..]`.
struct SymRow {
    k: u32,
    skip: u32,
    mons: Vec<u32>,
}

/// A matrix row as column indices (ascending, i.e. monomials descending); its
/// coefficients are `basis[k].coefs[skip..]`, never copied.
struct MRow {
    k: u32,
    skip: u32,
    cols: Vec<u32>,
}

/// The F4 matrix of one step: reducer rows (at most one per pivot column) and
/// the rows to reduce, over the columns of every monomial they mention.
struct Matrix {
    /// Monomial of each column, decreasing.
    col_mon: Vec<u32>,
    upper: Vec<MRow>,
    lower: Vec<MRow>,
    /// The upper row whose leading column this is, or [`NONE`].
    piv_of_col: Vec<u32>,
}

/// Symbolic preprocessing: give every monomial of `todo` (and, recursively, of
/// the reducers added) that is divisible by a basis leading monomial a reducer
/// row `t·g`; `pivots` seeds reducers already chosen by the caller (keyed by
/// leading monomial).  Then lay the rows out over sorted columns.
fn build_matrix(
    table: &mut MonoTable,
    basis: &Basis,
    mut pivots: HashMap<u32, SymRow, Hasher>,
    todo: Vec<SymRow>,
) -> Result<Matrix, Stop> {
    table.new_pass();
    let mut col_mon: Vec<u32> = Vec::new();
    for r in pivots.values().chain(todo.iter()) {
        for &m in &r.mons {
            if table.mark(m) {
                col_mon.push(m);
            }
        }
    }
    let mut queue: Vec<u32> = col_mon.clone();
    let mut ticks = 0u32;
    while let Some(m) = queue.pop() {
        if crate::poly::groebner::reduce::poll_due(&mut ticks) && crate::budget::interrupted() {
            return Err(Stop::Interrupted);
        }
        if pivots.contains_key(&m) {
            continue;
        }
        let Some(k) = basis.reducer(table, m) else {
            continue;
        };
        let g = &basis.polys[k];
        let t = table.quo(m, g.lead());
        let mut mons = Vec::with_capacity(g.mons.len());
        for &gm in &g.mons {
            let tm = table.mul(t, gm)?;
            mons.push(tm);
            if table.mark(tm) {
                col_mon.push(tm);
                queue.push(tm);
            }
        }
        pivots.insert(
            m,
            SymRow {
                k: k as u32,
                skip: 0,
                mons,
            },
        );
    }

    col_mon.sort_unstable_by(|&a, &b| table.cmp(b, a));
    // Column of each monomial id, through a scratch array kept in the table.
    let mut col_of = std::mem::take(&mut table.col_scratch);
    if col_of.len() < table.len() {
        col_of.resize(table.len(), 0);
    }
    for (c, &m) in col_mon.iter().enumerate() {
        col_of[m as usize] = c as u32;
    }
    let to_row = |r: SymRow| MRow {
        k: r.k,
        skip: r.skip,
        cols: r.mons.iter().map(|&m| col_of[m as usize]).collect(),
    };
    let mut piv_of_col = vec![NONE; col_mon.len()];
    let mut upper = Vec::with_capacity(pivots.len());
    for (_, r) in pivots {
        let r = to_row(r);
        piv_of_col[r.cols[0] as usize] = upper.len() as u32;
        upper.push(r);
    }
    let lower = todo.into_iter().map(to_row).collect();
    table.col_scratch = col_of;
    Ok(Matrix {
        col_mon,
        upper,
        lower,
        piv_of_col,
    })
}

/// `dense += f · row`, keeping every entry below `p²` with one comparison
/// (an entry is below `p²` and `f·x < p²`, so the sum fits a `u64`).
#[inline]
fn axpy(dense: &mut [u64], cols: &[u32], coefs: &[u64], f: u64, p2: u64) {
    for (&c, &x) in cols.iter().zip(coefs) {
        let d = &mut dense[c as usize];
        *d += f * x;
        if *d >= p2 {
            *d -= p2;
        }
    }
}

/// A reduced row: its non-zero columns and coefficients (below `p`).
struct OutRow {
    cols: Vec<u32>,
    coefs: Vec<u64>,
}

/// The result of reducing a matrix: the reduced rows that are kept, which
/// lower rows they came from, and the upper rows any of them used.
struct Reduction {
    rows: Vec<OutRow>,
    survivors: Vec<u32>,
    used_upper: Vec<bool>,
}

/// Reduce the lower rows of `mat` left to right against the upper rows.
///
/// With `echelon`, also against each other: a row that survives becomes a
/// pivot for the rows after it and is returned, monic; rows that reduce to
/// zero are dropped.  Without, each row is reduced independently against the
/// upper rows and returned as is (possibly empty), one per lower row.
fn reduce_matrix(mat: &Matrix, basis: &Basis, pr: Prime, echelon: bool) -> Result<Reduction, Stop> {
    let ncols = mat.col_mon.len();
    let mut dense = vec![0u64; ncols];
    let mut piv_of_col = mat.piv_of_col.clone();
    let mut rows: Vec<OutRow> = Vec::new();
    let mut survivors = Vec::new();
    let mut used_upper = vec![false; mat.upper.len()];
    let mut used: Vec<u32> = Vec::new();
    for (ri, row) in mat.lower.iter().enumerate() {
        if ri % 16 == 15 && crate::budget::interrupted() {
            return Err(Stop::Interrupted);
        }
        let Some(&start) = row.cols.first() else {
            if !echelon {
                survivors.push(ri as u32);
                rows.push(OutRow {
                    cols: Vec::new(),
                    coefs: Vec::new(),
                });
            }
            continue;
        };
        let coefs = &basis.polys[row.k as usize].coefs[row.skip as usize..];
        for (&c, &v) in row.cols.iter().zip(coefs) {
            dense[c as usize] = v;
        }
        used.clear();
        let mut cols = Vec::new();
        let mut out = Vec::new();
        for c in start as usize..ncols {
            let v = dense[c];
            if v == 0 {
                continue;
            }
            dense[c] = 0;
            let v = pr.reduce(v);
            if v == 0 {
                continue;
            }
            let pv = piv_of_col[c];
            if pv == NONE {
                cols.push(c as u32);
                out.push(v);
                continue;
            }
            // Pivot rows are monic: subtract v·piv, i.e. add (p − v)·piv.
            let f = pr.p - v;
            if pv >= LOWER {
                let r = &rows[(pv - LOWER) as usize];
                axpy(&mut dense, &r.cols[1..], &r.coefs[1..], f, pr.p2);
            } else {
                let u = &mat.upper[pv as usize];
                let uc = &basis.polys[u.k as usize].coefs[u.skip as usize..];
                axpy(&mut dense, &u.cols[1..], &uc[1..], f, pr.p2);
                used.push(pv);
            }
        }
        if echelon {
            if cols.is_empty() {
                continue;
            }
            let inv = pr.inv(out[0]);
            if inv != 1 {
                for c in &mut out {
                    *c = pr.mul(*c, inv);
                }
            }
            piv_of_col[cols[0] as usize] = LOWER + rows.len() as u32;
        }
        for &u in &used {
            used_upper[u as usize] = true;
        }
        survivors.push(ri as u32);
        rows.push(OutRow { cols, coefs: out });
    }
    Ok(Reduction {
        rows,
        survivors,
        used_upper,
    })
}

fn out_to_poly(mat: &Matrix, r: OutRow) -> ModPoly {
    ModPoly {
        mons: r.cols.iter().map(|&c| mat.col_mon[c as usize]).collect(),
        coefs: r.coefs,
    }
}

// ---------------------------------------------------------------------------
// Tracing: learn the computation once, replay it modulo further primes
// ---------------------------------------------------------------------------

/// A row of a traced matrix: `basis[k]` from coefficient `skip` on, laid out
/// over the matrix's columns.
struct TRow {
    k: u32,
    skip: u32,
    cols: Vec<u32>,
}

/// One traced matrix: the upper rows some kept row used, the lower rows that
/// were kept (in reduction order), and the support each had once reduced.
struct TMatrix {
    ncols: u32,
    upper: Vec<TRow>,
    lower: Vec<TRow>,
    /// Upper row index, `LOWER + i` for the `i`-th kept lower row (echelon
    /// matrices only), or [`NONE`].
    piv_of_col: Vec<u32>,
    /// Support of each reduced lower row at the learning prime, lead first.
    out_cols: Vec<Vec<u32>>,
}

impl TMatrix {
    fn from_learn(mat: &Matrix, red: &Reduction, echelon: bool) -> Self {
        // Keep only the upper rows a kept row used; renumber.
        let mut new_index = vec![NONE; mat.upper.len()];
        let mut upper = Vec::new();
        for (u, row) in mat.upper.iter().enumerate() {
            if red.used_upper[u] {
                new_index[u] = upper.len() as u32;
                upper.push(TRow {
                    k: row.k,
                    skip: row.skip,
                    cols: row.cols.clone(),
                });
            }
        }
        let mut piv_of_col: Vec<u32> = mat
            .piv_of_col
            .iter()
            .map(|&pv| {
                if pv == NONE {
                    NONE
                } else {
                    new_index[pv as usize]
                }
            })
            .collect();
        if echelon {
            for (i, r) in red.rows.iter().enumerate() {
                piv_of_col[r.cols[0] as usize] = LOWER + i as u32;
            }
        }
        let lower = red
            .survivors
            .iter()
            .map(|&i| {
                let row = &mat.lower[i as usize];
                TRow {
                    k: row.k,
                    skip: row.skip,
                    cols: row.cols.clone(),
                }
            })
            .collect();
        TMatrix {
            ncols: mat.col_mon.len() as u32,
            upper,
            lower,
            piv_of_col,
            out_cols: red.rows.iter().map(|r| r.cols.clone()).collect(),
        }
    }
}

/// The record of one F4 run, which a computation modulo another prime replays
/// without pair handling, symbolic preprocessing, or the rows that reduced to
/// zero — the "Gröbner trace" of Traverso (1988), as msolve and Groebner.jl
/// use it ("learn and apply").
pub(crate) struct Trace {
    /// Number of input polynomials (in the caller's order).
    n_inputs: usize,
    /// Basis element `i < input_perm.len()` is input `input_perm[i]`, monic.
    input_perm: Vec<u32>,
    /// One per F4 step: the matrix, and the order its kept rows join the basis.
    steps: Vec<(TMatrix, Vec<u32>)>,
    /// The final tail reduction: one lower row per element of the reduced
    /// basis, and the monomial of each of its columns.
    inter: TMatrix,
    inter_col_mon: Vec<u32>,
    /// Leading monomial of each reduced basis element, and the order
    /// (increasing leading monomial) the result is returned in.
    leads: Vec<u32>,
    final_order: Vec<u32>,
}

/// Why replaying a trace modulo a prime gave no basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ApplyFail {
    /// The computation differs from the learned one at this prime (a pivot
    /// vanished, or a coefficient that was zero at the learning prime is
    /// not): one of the two primes is unlucky for this trace.
    Mismatch,
    /// An armed budget fired.
    Interrupted,
}

/// `dense += f · row` in each of `N` lanes (one prime per lane), keeping every
/// entry below that lane's `p²` as [`axpy`] does.
///
/// Every factor is below `p < 2³¹`; the casts through `u32` say so, which lets
/// the compiler multiply all lanes with one 32×32→64-bit vector instruction
/// where the target has one (see [`apply_matrix`]).
#[inline(always)]
fn axpy_n<const N: usize>(
    dense: &mut [[u64; N]],
    cols: &[u32],
    coefs: &[[u64; N]],
    f: &[u64; N],
    p2: &[u64; N],
) {
    for (&c, x) in cols.iter().zip(coefs) {
        let d = &mut dense[c as usize];
        for l in 0..N {
            let s = d[l] + u64::from(f[l] as u32) * u64::from(x[l] as u32);
            d[l] = if s >= p2[l] { s - p2[l] } else { s };
        }
    }
}

/// Replay one traced matrix in `N` lanes at once, one prime per lane: reduce
/// each lower row as the learning run did and return its coefficients on the
/// learned support (zeros allowed after the lead).  With `echelon`, each
/// result is made monic and becomes a pivot for the rows after it.  A lane
/// whose computation diverges from the learned one is marked in `failed` and
/// carries zeros from then on; the other lanes are unaffected.
fn apply_matrix<const N: usize>(
    tm: &TMatrix,
    coefs: &[Vec<[u64; N]>],
    prs: &[Prime; N],
    echelon: bool,
    dense: &mut Vec<[u64; N]>,
    failed: &mut [bool; N],
) -> Result<Vec<Vec<[u64; N]>>, ApplyFail> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked just above).
            return unsafe { apply_matrix_avx2(tm, coefs, prs, echelon, dense, failed) };
        }
    }
    apply_matrix_body(tm, coefs, prs, echelon, dense, failed)
}

/// [`apply_matrix_body`] compiled for AVX2, where a lane group of four
/// accumulators is one vector register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_matrix_avx2<const N: usize>(
    tm: &TMatrix,
    coefs: &[Vec<[u64; N]>],
    prs: &[Prime; N],
    echelon: bool,
    dense: &mut Vec<[u64; N]>,
    failed: &mut [bool; N],
) -> Result<Vec<Vec<[u64; N]>>, ApplyFail> {
    apply_matrix_body(tm, coefs, prs, echelon, dense, failed)
}

#[inline(always)]
fn apply_matrix_body<const N: usize>(
    tm: &TMatrix,
    coefs: &[Vec<[u64; N]>],
    prs: &[Prime; N],
    echelon: bool,
    dense: &mut Vec<[u64; N]>,
    failed: &mut [bool; N],
) -> Result<Vec<Vec<[u64; N]>>, ApplyFail> {
    let ncols = tm.ncols as usize;
    dense.clear();
    dense.resize(ncols, [0; N]);
    let p2: [u64; N] = std::array::from_fn(|l| prs[l].p2);
    let mut out_rows: Vec<Vec<[u64; N]>> = Vec::with_capacity(tm.lower.len());
    let mut found: Vec<(u32, [u64; N])> = Vec::new();
    for (j, row) in tm.lower.iter().enumerate() {
        if j % 16 == 15 && crate::budget::interrupted() {
            return Err(ApplyFail::Interrupted);
        }
        let want = &tm.out_cols[j];
        let Some(&start) = row.cols.first() else {
            out_rows.push(Vec::new());
            continue;
        };
        let rc = &coefs[row.k as usize][row.skip as usize..];
        for (&c, v) in row.cols.iter().zip(rc) {
            dense[c as usize] = *v;
        }
        found.clear();
        for c in start as usize..ncols {
            let v = dense[c];
            if v == [0; N] {
                continue;
            }
            dense[c] = [0; N];
            let mut r = [0u64; N];
            let mut any = false;
            for l in 0..N {
                r[l] = if failed[l] { 0 } else { prs[l].reduce(v[l]) };
                any |= r[l] != 0;
            }
            if !any {
                continue;
            }
            let pv = tm.piv_of_col[c];
            let f: [u64; N] = std::array::from_fn(|l| if r[l] == 0 { 0 } else { prs[l].p - r[l] });
            if pv == NONE {
                found.push((c as u32, r));
            } else if pv >= LOWER {
                let i = (pv - LOWER) as usize;
                if i >= j {
                    found.push((c as u32, r));
                } else {
                    axpy_n(dense, &tm.out_cols[i][1..], &out_rows[i][1..], &f, &p2);
                }
            } else {
                let u = &tm.upper[pv as usize];
                let uc = &coefs[u.k as usize][u.skip as usize..];
                axpy_n(dense, &u.cols[1..], &uc[1..], &f, &p2);
            }
        }
        // Lay the result over the learned support.  (`dense` is all zero
        // again here: the scan cleared every entry it passed.)
        let mut res = vec![[0u64; N]; want.len()];
        let mut w = 0usize;
        for &(c, v) in &found {
            while w < want.len() && want[w] < c {
                w += 1;
            }
            if w == want.len() || want[w] != c {
                for l in 0..N {
                    if v[l] != 0 {
                        failed[l] = true;
                    }
                }
                continue;
            }
            res[w] = v;
        }
        for l in 0..N {
            if failed[l] {
                for x in res.iter_mut() {
                    x[l] = 0;
                }
                continue;
            }
            if echelon {
                let lc = res.first().map_or(0, |x| x[l]);
                if lc == 0 {
                    failed[l] = true;
                    for x in res.iter_mut() {
                        x[l] = 0;
                    }
                    continue;
                }
                let inv = prs[l].inv(lc);
                if inv != 1 {
                    for x in res.iter_mut() {
                        x[l] = prs[l].mul(x[l], inv);
                    }
                }
            }
        }
        if failed.iter().all(|&f| f) {
            return Err(ApplyFail::Mismatch);
        }
        out_rows.push(res);
    }
    Ok(out_rows)
}

impl Trace {
    /// Replay the trace modulo `pr`.  `inputs` are the coefficients of the
    /// inputs on the supports they had when the trace was learned (the full
    /// integer supports, in the caller's order; zeros are allowed except in
    /// the leading position).  The result has the shape of the learned basis
    /// except where a coefficient vanishes modulo `pr`.
    pub(crate) fn apply(&self, inputs: &[Vec<u64>], pr: Prime) -> Result<Vec<ModPoly>, ApplyFail> {
        let [r] = self.apply_n::<1>([inputs], [pr]);
        r
    }

    /// [`Self::apply`] modulo two primes in one pass over the trace, so the
    /// index work is shared between them.
    pub(crate) fn apply2(
        &self,
        inputs: [&[Vec<u64>]; 2],
        prs: [Prime; 2],
    ) -> [Result<Vec<ModPoly>, ApplyFail>; 2] {
        self.apply_n::<2>(inputs, prs)
    }

    /// [`Self::apply`] modulo four primes in one pass over the trace.
    pub(crate) fn apply4(
        &self,
        inputs: [&[Vec<u64>]; 4],
        prs: [Prime; 4],
    ) -> [Result<Vec<ModPoly>, ApplyFail>; 4] {
        self.apply_n::<4>(inputs, prs)
    }

    fn apply_n<const N: usize>(
        &self,
        inputs: [&[Vec<u64>]; N],
        prs: [Prime; N],
    ) -> [Result<Vec<ModPoly>, ApplyFail>; N] {
        let mut failed = [false; N];
        for l in 0..N {
            if inputs[l].len() != self.n_inputs {
                failed[l] = true;
            }
        }
        let fail_all = |e: ApplyFail| std::array::from_fn(|_| Err(e));
        if failed.iter().all(|&f| f) {
            return fail_all(ApplyFail::Mismatch);
        }
        let mut coefs: Vec<Vec<[u64; N]>> = Vec::new();
        for &i in &self.input_perm {
            let len = (0..N)
                .find(|&l| !failed[l])
                .map_or(0, |l| inputs[l][i as usize].len());
            let mut c = vec![[0u64; N]; len];
            for l in 0..N {
                if failed[l] {
                    continue;
                }
                let src = &inputs[l][i as usize];
                let lc = src.first().copied().unwrap_or(0);
                if lc == 0 || src.len() != len {
                    failed[l] = true;
                    continue;
                }
                let inv = prs[l].inv(lc);
                for (x, &y) in c.iter_mut().zip(src) {
                    x[l] = if inv == 1 { y } else { prs[l].mul(y, inv) };
                }
            }
            coefs.push(c);
        }
        if failed.iter().all(|&f| f) {
            return fail_all(ApplyFail::Mismatch);
        }
        let mut dense = Vec::new();
        for (tm, order) in &self.steps {
            if crate::budget::interrupted() {
                return fail_all(ApplyFail::Interrupted);
            }
            let mut rows = match apply_matrix(tm, &coefs, &prs, true, &mut dense, &mut failed) {
                Ok(r) => r,
                Err(e) => return fail_all(e),
            };
            for &i in order {
                coefs.push(std::mem::take(&mut rows[i as usize]));
            }
        }
        let tails = match apply_matrix(&self.inter, &coefs, &prs, false, &mut dense, &mut failed) {
            Ok(r) => r,
            Err(e) => return fail_all(e),
        };
        std::array::from_fn(|l| {
            if failed[l] {
                return Err(ApplyFail::Mismatch);
            }
            let mut out = Vec::with_capacity(self.final_order.len());
            for &i in &self.final_order {
                let i = i as usize;
                let mut mons = vec![self.leads[i]];
                let mut cs = vec![1u64];
                for (&c, v) in self.inter.out_cols[i].iter().zip(&tails[i]) {
                    if v[l] != 0 {
                        mons.push(self.inter_col_mon[c as usize]);
                        cs.push(v[l]);
                    }
                }
                out.push(ModPoly { mons, coefs: cs });
            }
            Ok(out)
        })
    }
}

/// One F4 step on the selected pairs: the new basis elements (monic, in the
/// order found) and the step's trace.
fn f4_step(
    table: &mut MonoTable,
    basis: &Basis,
    sel: &[CriticalPair],
    pr: Prime,
) -> Result<(Vec<ModPoly>, TMatrix), Stop> {
    let mut pivots: HashMap<u32, SymRow, Hasher> = HashMap::with_hasher(Hasher::default());
    let mut todo: Vec<SymRow> = Vec::new();
    let mut have: HashMap<(u32, usize), (), Hasher> = HashMap::with_hasher(Hasher::default());
    for pair in sel {
        let l = table.intern_u32(&pair.lcm_exp).ok_or(Stop::Overflow)?;
        for k in [pair.i, pair.j] {
            let g = &basis.polys[k];
            let t = table.quo(l, g.lead());
            if have.insert((t, k), ()).is_some() {
                continue;
            }
            let mut mons = Vec::with_capacity(g.mons.len());
            for &gm in &g.mons {
                mons.push(table.mul(t, gm)?);
            }
            let row = SymRow {
                k: k as u32,
                skip: 0,
                mons,
            };
            if let std::collections::hash_map::Entry::Vacant(e) = pivots.entry(l) {
                e.insert(row);
            } else {
                todo.push(row);
            }
        }
    }
    // Sparse rows first: they make cheaper pivots for the rows after them.
    todo.sort_by_key(|r| r.mons.len());
    let mat = build_matrix(table, basis, pivots, todo)?;
    let red = reduce_matrix(&mat, basis, pr, true)?;
    let tm = TMatrix::from_learn(&mat, &red, true);
    let polys = red.rows.into_iter().map(|r| out_to_poly(&mat, r)).collect();
    Ok((polys, tm))
}

/// The reduced Gröbner basis of the ideal generated by `gens` over 𝔽_p.
///
/// `gens` must be non-zero and sorted (decreasing monomials); they need not be
/// monic.  The result is sorted by increasing leading monomial.
#[cfg(test)]
pub(crate) fn groebner_mod_p(
    table: &mut MonoTable,
    gens: &[ModPoly],
    pr: Prime,
) -> Result<Vec<ModPoly>, Stop> {
    groebner_mod_p_learn(table, gens, pr).map(|(gb, _)| gb)
}

/// The reduced Gröbner basis of `⟨gens⟩` over 𝔽_p, as [`groebner_mod_p`], and
/// the trace of the run.  `gens` may hold zero coefficients after the leading
/// one (the inputs' full integer supports, so that every prime sees the same
/// supports); the leading coefficients must be non-zero.
pub(crate) fn groebner_mod_p_learn(
    table: &mut MonoTable,
    gens: &[ModPoly],
    pr: Prime,
) -> Result<(Vec<ModPoly>, Trace), Stop> {
    let mut basis = Basis::new();
    let mut init: Vec<(usize, &ModPoly)> = gens
        .iter()
        .enumerate()
        .filter(|(_, g)| !g.mons.is_empty())
        .collect();
    init.sort_by(|a, b| table.cmp(a.1.lead(), b.1.lead()));
    let input_perm: Vec<u32> = init.iter().map(|(i, _)| *i as u32).collect();
    for (_, g) in init {
        let mut g = g.clone();
        g.make_monic(pr);
        let s = g.mons.iter().map(|&m| table.deg(m)).max().unwrap_or(0);
        basis.add(table, g, s);
    }
    let mut steps = Vec::new();
    while !basis.pairs.is_empty() {
        if crate::budget::interrupted() {
            return Err(Stop::Interrupted);
        }
        // The normal strategy (every pair of least lcm degree), as msolve and
        // Groebner.jl use: on cyclic-7 it replays about a third faster than
        // selecting by sugar.
        let d = basis.pairs.iter().map(|p| p.lcm_deg).min().unwrap_or(0);
        let (sel, rest): (Vec<CriticalPair>, Vec<CriticalPair>) =
            basis.pairs.drain(..).partition(|p| p.lcm_deg == d);
        let d = sel.iter().map(|p| p.sugar_deg).min().unwrap_or(d);
        basis.pairs = rest;
        let (new, tm) = f4_step(table, &basis, &sel, pr)?;
        let mut order: Vec<u32> = (0..new.len() as u32).collect();
        order.sort_by(|&a, &b| table.cmp(new[a as usize].lead(), new[b as usize].lead()));
        let mut new: Vec<Option<ModPoly>> = new.into_iter().map(Some).collect();
        for &i in &order {
            basis.add(table, new[i as usize].take().expect("each once"), d);
        }
        steps.push((tm, order));
    }
    interreduce(table, &basis, pr, gens.len(), input_perm, steps)
}

/// The reduced basis from a Gröbner basis: drop redundant elements, then
/// tail-reduce the rest against each other.  Completes the trace.
fn interreduce(
    table: &mut MonoTable,
    basis: &Basis,
    pr: Prime,
    n_inputs: usize,
    input_perm: Vec<u32>,
    steps: Vec<(TMatrix, Vec<u32>)>,
) -> Result<(Vec<ModPoly>, Trace), Stop> {
    let keep: Vec<u32> = (0..basis.polys.len())
        .filter(|&k| !basis.redundant[k])
        .map(|k| k as u32)
        .collect();
    let tails: Vec<SymRow> = keep
        .iter()
        .map(|&k| SymRow {
            k,
            skip: 1,
            mons: basis.polys[k as usize].mons[1..].to_vec(),
        })
        .collect();
    let mat = build_matrix(table, basis, HashMap::with_hasher(Hasher::default()), tails)?;
    let red = reduce_matrix(&mat, basis, pr, false)?;
    let inter = TMatrix::from_learn(&mat, &red, false);
    let leads: Vec<u32> = keep
        .iter()
        .map(|&k| basis.polys[k as usize].lead())
        .collect();
    let mut final_order: Vec<u32> = (0..keep.len() as u32).collect();
    final_order.sort_by(|&a, &b| table.cmp(leads[a as usize], leads[b as usize]));
    let mut polys: Vec<Option<ModPoly>> = red
        .rows
        .into_iter()
        .zip(&leads)
        .map(|(r, &lead)| {
            let t = out_to_poly(&mat, r);
            let mut mons = vec![lead];
            let mut coefs = vec![1u64];
            mons.extend(t.mons);
            coefs.extend(t.coefs);
            Some(ModPoly { mons, coefs })
        })
        .collect();
    let out: Vec<ModPoly> = final_order
        .iter()
        .map(|&i| polys[i as usize].take().expect("each once"))
        .collect();
    let trace = Trace {
        n_inputs,
        input_perm,
        steps,
        inter,
        inter_col_mon: mat.col_mon,
        leads,
        final_order,
    };
    Ok((out, trace))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes_are_prime_and_descending() {
        let ps: Vec<u64> = primes().take(5).map(|p| p.p).collect();
        assert_eq!(ps[0], 2147483647);
        assert!(ps.windows(2).all(|w| w[0] > w[1]));
        for &p in &ps {
            assert!((2..50_000u64).all(|d| p % d != 0));
        }
        assert!(!is_prime_u32(2147483649)); // 3 · 715827883
        assert!(!is_prime_u32(3215031751)); // strong pseudoprime to 2, 3, 5, 7
    }

    /// Products and quotients found through the linear hash are the
    /// monomials interning their exponent vectors gives, across growth of the
    /// slot array; an exponent past `u16` is refused.
    #[test]
    fn monomial_table_hashing() {
        let mut t = MonoTable::new(3, MonomialOrder::GRevLex);
        let mut ids = Vec::new();
        for a in 0..12u16 {
            for b in 0..12u16 {
                for c in 0..12u16 {
                    ids.push((t.intern(&[a, b, c]), [a, b, c]));
                }
            }
        }
        assert_eq!(t.len(), 12 * 12 * 12);
        for (i, &(x, ex)) in ids.iter().enumerate().step_by(7) {
            assert_eq!(t.exps(x), &ex);
            assert_eq!(t.intern(&ex), x);
            let (y, ey) = ids[(i * 31) % ids.len()];
            let p = t.mul(x, y).unwrap();
            let sum: Vec<u16> = ex.iter().zip(&ey).map(|(a, b)| a + b).collect();
            assert_eq!(t.exps(p), &sum[..]);
            assert_eq!(t.intern(&sum), p);
            assert_eq!(t.quo(p, y), x);
            assert!(t.divides(y, p) && t.divides(x, p));
        }
        let big = t.intern(&[u16::MAX, 0, 0]);
        let one = t.intern(&[1, 0, 0]);
        assert_eq!(t.mul(big, one), Err(Stop::Overflow));
    }

    #[test]
    fn inverse_mod_p() {
        let pr = Prime::new(2147483647);
        for a in [1u64, 2, 3, 12345, 2147483646] {
            assert_eq!(pr.mul(a, pr.inv(a)), 1);
        }
    }

    #[test]
    fn barrett_reduction_matches_the_remainder() {
        for p in [3u64, 65537, 1_073_741_827, 2_147_483_629, 2_147_483_647] {
            let pr = Prime::new(p);
            let p2 = p * p;
            let mut x = 0x9e37_79b9_7f4a_7c15u64;
            let mut samples = vec![0, 1, p - 1, p, p + 1, p2 - 1, p2 - p, 2 * p - 1];
            for _ in 0..10_000 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                samples.push(x % p2);
            }
            for v in samples {
                assert_eq!(pr.reduce(v), v % p, "p = {p}, v = {v}");
            }
        }
    }
}
