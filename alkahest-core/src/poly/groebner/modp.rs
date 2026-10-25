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
//! * **Pair selection** — every critical pair of minimal sugar degree is taken
//!   at once; the pair list is kept by the same Gebauer–Möller code as the ℚ
//!   Buchberger engine ([`super::pairs::update_pairs`]).
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
use crate::poly::groebner::pairs::{update_pairs, CriticalPair};

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
}

impl Prime {
    pub(crate) fn new(p: u64) -> Self {
        debug_assert!(p > 2 && p < (1 << 31) && is_prime_u32(p));
        Prime { p, p2: p * p }
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
#[derive(Clone)]
pub(crate) struct MonoTable {
    n: usize,
    order: MonomialOrder,
    exps: Vec<u16>,
    deg: Vec<u32>,
    /// Short exponent vector: a bit mask with `a | b ⟹ sev(a) ⊆ sev(b)`.
    sev: Vec<u64>,
    map: HashMap<Box<[u16]>, u32, Hasher>,
    scratch: Vec<u16>,
}

impl MonoTable {
    pub(crate) fn new(n: usize, order: MonomialOrder) -> Self {
        MonoTable {
            n,
            order,
            exps: Vec::new(),
            deg: Vec::new(),
            sev: Vec::new(),
            map: HashMap::with_hasher(Hasher::default()),
            scratch: vec![0; n],
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

    pub(crate) fn intern(&mut self, e: &[u16]) -> u32 {
        if let Some(&id) = self.map.get(e) {
            return id;
        }
        let id = self.deg.len() as u32;
        self.exps.extend_from_slice(e);
        self.deg.push(e.iter().map(|&x| u32::from(x)).sum());
        let s = self.sev_of(e);
        self.sev.push(s);
        self.map.insert(e.into(), id);
        id
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
        let mut v = std::mem::take(&mut self.scratch);
        v.clear();
        let (ia, ib) = (a as usize * self.n, b as usize * self.n);
        for k in 0..self.n {
            let s = self.exps[ia + k]
                .checked_add(self.exps[ib + k])
                .ok_or(Stop::Overflow)?;
            v.push(s);
        }
        let id = self.intern(&v);
        self.scratch = v;
        Ok(id)
    }

    /// `b / a`, for `a | b`.
    fn quo(&mut self, b: u32, a: u32) -> u32 {
        let mut v = std::mem::take(&mut self.scratch);
        v.clear();
        let (ia, ib) = (a as usize * self.n, b as usize * self.n);
        for k in 0..self.n {
            v.push(self.exps[ib + k] - self.exps[ia + k]);
        }
        let id = self.intern(&v);
        self.scratch = v;
        id
    }

    #[inline]
    pub(crate) fn divides(&self, a: u32, b: u32) -> bool {
        if self.sev[a as usize] & !self.sev[b as usize] != 0 {
            return false;
        }
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
    pairs: Vec<CriticalPair>,
}

impl Basis {
    fn add(&mut self, table: &MonoTable, g: ModPoly, sugar: u32) {
        let lm = g.lead();
        for (k, h) in self.polys.iter().enumerate() {
            if !self.redundant[k] && table.divides(lm, h.lead()) {
                self.redundant[k] = true;
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
        update_pairs(&self.lead, &self.sugar, &mut self.pairs, idx);
    }

    /// A non-redundant element whose leading monomial divides `m`, preferring
    /// the sparsest.
    fn reducer(&self, table: &MonoTable, m: u32) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (k, g) in self.polys.iter().enumerate() {
            if self.redundant[k] || !table.divides(g.lead(), m) {
                continue;
            }
            if best.map_or(true, |b| g.mons.len() < self.polys[b].mons.len()) {
                best = Some(k);
            }
        }
        best
    }
}

/// A matrix row: column indices ascending (monomials descending) and
/// coefficients below `p`.
struct Row {
    cols: Vec<u32>,
    coefs: Vec<u64>,
}

/// The F4 matrix of one step: reducer rows (one per pivot column) and the rows
/// to reduce, over the columns of every monomial they mention.
struct Matrix {
    /// Monomial of each column, decreasing.
    col_mon: Vec<u32>,
    pivots: Vec<Option<Row>>,
    todo: Vec<Row>,
}

/// Rows given by monomials (decreasing) and coefficients.
type MonRow = (Vec<u32>, Vec<u64>);

/// Symbolic preprocessing: give every monomial of `todo` (and, recursively, of
/// the reducers added) that is divisible by a basis leading monomial a reducer
/// row `t·g`; `pivots` seeds reducers already chosen by the caller (keyed by
/// leading monomial).  Then lay the rows out over sorted columns.
fn build_matrix(
    table: &mut MonoTable,
    basis: &Basis,
    mut pivots: HashMap<u32, MonRow, Hasher>,
    todo: Vec<MonRow>,
) -> Result<Matrix, Stop> {
    let mut seen: HashMap<u32, (), Hasher> = HashMap::with_hasher(Hasher::default());
    let mut queue: Vec<u32> = Vec::new();
    for (mons, _) in pivots.values().chain(todo.iter()) {
        for &m in mons {
            if seen.insert(m, ()).is_none() {
                queue.push(m);
            }
        }
    }
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
            if seen.insert(tm, ()).is_none() {
                queue.push(tm);
            }
        }
        pivots.insert(m, (mons, g.coefs.clone()));
    }

    let mut col_mon: Vec<u32> = seen.into_keys().collect();
    col_mon.sort_unstable_by(|&a, &b| table.cmp(b, a));
    let mut col_of: HashMap<u32, u32, Hasher> =
        HashMap::with_capacity_and_hasher(col_mon.len(), Hasher::default());
    for (c, &m) in col_mon.iter().enumerate() {
        col_of.insert(m, c as u32);
    }
    let to_row = |(mons, coefs): MonRow| Row {
        cols: mons.iter().map(|m| col_of[m]).collect(),
        coefs,
    };
    let mut piv_rows: Vec<Option<Row>> = (0..col_mon.len()).map(|_| None).collect();
    for (_, r) in pivots {
        let r = to_row(r);
        let c = r.cols[0] as usize;
        piv_rows[c] = Some(r);
    }
    let todo = todo.into_iter().map(to_row).collect();
    Ok(Matrix {
        col_mon,
        pivots: piv_rows,
        todo,
    })
}

/// Reduce every `todo` row of `mat` left to right against the pivots.  With
/// `echelon`, a row that survives becomes a pivot for the rows after it (and is
/// returned, monic); without, each row is reduced independently and returned
/// as is (possibly zero).
fn reduce_matrix(mat: &mut Matrix, pr: Prime, echelon: bool) -> Result<Vec<Row>, Stop> {
    let ncols = mat.col_mon.len();
    let mut dense = vec![0u64; ncols];
    let mut out = Vec::new();
    let todo = std::mem::take(&mut mat.todo);
    for (ri, row) in todo.into_iter().enumerate() {
        if ri % 16 == 15 && crate::budget::interrupted() {
            return Err(Stop::Interrupted);
        }
        let Some(&start) = row.cols.first() else {
            if !echelon {
                out.push(row);
            }
            continue;
        };
        for (&c, &v) in row.cols.iter().zip(&row.coefs) {
            dense[c as usize] = v;
        }
        let mut cols = Vec::new();
        let mut coefs = Vec::new();
        for c in start as usize..ncols {
            let v = dense[c];
            if v == 0 {
                continue;
            }
            dense[c] = 0;
            let v = v % pr.p;
            if v == 0 {
                continue;
            }
            if let Some(piv) = &mat.pivots[c] {
                // Pivot rows are monic: subtract v·piv, i.e. add (p − v)·piv.
                let f = pr.p - v;
                for (&cc, &cf) in piv.cols[1..].iter().zip(&piv.coefs[1..]) {
                    let d = &mut dense[cc as usize];
                    *d += f * cf;
                    if *d >= pr.p2 {
                        *d -= pr.p2;
                    }
                }
            } else {
                cols.push(c as u32);
                coefs.push(v);
            }
        }
        let mut r = Row { cols, coefs };
        if echelon {
            if r.cols.is_empty() {
                continue;
            }
            let inv = pr.inv(r.coefs[0]);
            for c in &mut r.coefs {
                *c = pr.mul(*c, inv);
            }
            let c0 = r.cols[0] as usize;
            mat.pivots[c0] = Some(Row {
                cols: r.cols.clone(),
                coefs: r.coefs.clone(),
            });
        }
        out.push(r);
    }
    Ok(out)
}

fn row_to_poly(mat: &Matrix, r: Row) -> ModPoly {
    ModPoly {
        mons: r.cols.iter().map(|&c| mat.col_mon[c as usize]).collect(),
        coefs: r.coefs,
    }
}

/// One F4 step on the selected pairs: the new basis elements (monic).
fn f4_step(
    table: &mut MonoTable,
    basis: &Basis,
    sel: &[CriticalPair],
    pr: Prime,
) -> Result<Vec<ModPoly>, Stop> {
    let mut pivots: HashMap<u32, MonRow, Hasher> = HashMap::with_hasher(Hasher::default());
    let mut todo: Vec<MonRow> = Vec::new();
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
            let row = (mons, g.coefs.clone());
            if let std::collections::hash_map::Entry::Vacant(e) = pivots.entry(l) {
                e.insert(row);
            } else {
                todo.push(row);
            }
        }
    }
    // Sparse rows first: they make cheaper pivots for the rows after them.
    todo.sort_by_key(|(m, _)| m.len());
    let mut mat = build_matrix(table, basis, pivots, todo)?;
    let rows = reduce_matrix(&mut mat, pr, true)?;
    Ok(rows.into_iter().map(|r| row_to_poly(&mat, r)).collect())
}

/// The reduced Gröbner basis of the ideal generated by `gens` over 𝔽_p.
///
/// `gens` must be non-zero and sorted (decreasing monomials); they need not be
/// monic.  The result is sorted by increasing leading monomial.
pub(crate) fn groebner_mod_p(
    table: &mut MonoTable,
    gens: &[ModPoly],
    pr: Prime,
) -> Result<Vec<ModPoly>, Stop> {
    let mut basis = Basis {
        polys: Vec::new(),
        lead: Vec::new(),
        sugar: Vec::new(),
        redundant: Vec::new(),
        pairs: Vec::new(),
    };
    let mut init: Vec<ModPoly> = gens
        .iter()
        .filter(|g| !g.mons.is_empty())
        .cloned()
        .collect();
    init.sort_by(|a, b| table.cmp(a.lead(), b.lead()));
    for mut g in init {
        g.make_monic(pr);
        let s = g.mons.iter().map(|&m| table.deg(m)).max().unwrap_or(0);
        basis.add(table, g, s);
    }
    while !basis.pairs.is_empty() {
        if crate::budget::interrupted() {
            return Err(Stop::Interrupted);
        }
        let d = basis.pairs.iter().map(|p| p.sugar_deg).min().unwrap_or(0);
        let (sel, rest): (Vec<CriticalPair>, Vec<CriticalPair>) =
            basis.pairs.drain(..).partition(|p| p.sugar_deg == d);
        basis.pairs = rest;
        let mut new = f4_step(table, &basis, &sel, pr)?;
        new.sort_by(|a, b| table.cmp(a.lead(), b.lead()));
        for g in new {
            basis.add(table, g, d);
        }
    }
    interreduce(table, &basis, pr)
}

/// The reduced basis from a Gröbner basis: drop redundant elements, then
/// tail-reduce the rest against each other.
fn interreduce(table: &mut MonoTable, basis: &Basis, pr: Prime) -> Result<Vec<ModPoly>, Stop> {
    let keep: Vec<usize> = (0..basis.polys.len())
        .filter(|&k| !basis.redundant[k])
        .collect();
    let tails: Vec<MonRow> = keep
        .iter()
        .map(|&k| {
            let g = &basis.polys[k];
            (g.mons[1..].to_vec(), g.coefs[1..].to_vec())
        })
        .collect();
    let mut mat = build_matrix(table, basis, HashMap::with_hasher(Hasher::default()), tails)?;
    let rows = reduce_matrix(&mut mat, pr, false)?;
    let mut out: Vec<ModPoly> = keep
        .iter()
        .zip(rows)
        .map(|(&k, r)| {
            let t = row_to_poly(&mat, r);
            let mut mons = vec![basis.polys[k].lead()];
            let mut coefs = vec![1u64];
            mons.extend(t.mons);
            coefs.extend(t.coefs);
            ModPoly { mons, coefs }
        })
        .collect();
    out.sort_by(|a, b| table.cmp(a.lead(), b.lead()));
    Ok(out)
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

    #[test]
    fn inverse_mod_p() {
        let pr = Prime::new(2147483647);
        for a in [1u64, 2, 3, 12345, 2147483646] {
            assert_eq!(pr.mul(a, pr.inv(a)), 1);
        }
    }
}
