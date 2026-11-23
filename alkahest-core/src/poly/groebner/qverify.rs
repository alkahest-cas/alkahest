//! Exact verification over ℚ of a multimodular lift, in the monomial table of
//! the modular computation.
//!
//! [`super::multimodular`] accepts a lifted basis `B` only once it has checked,
//! exactly, that every input reduces to zero modulo `B` (so `I ⊆ ⟨B⟩`) and
//! that every critical pair of `B` left by the Gebauer–Möller criteria
//! reduces to zero (so `B` is a Gröbner basis).  This module does those
//! reductions the way F4 does its own: the rows to reduce (inputs and
//! S-polynomials, a batch at a time) go through symbolic preprocessing, so
//! that every monomial they can reach gets one reducer row `t·b` and a column;
//! each row is then reduced over ℤ in a dense accumulator indexed by column.
//!
//! The arithmetic is fraction-free: every element of `B` is replaced by its
//! primitive integer multiple, and a row `R` is reduced by
//! `R ← (β/h)·R − (c/h)·t·b`, with `β` the leading coefficient of `b`, `c` the
//! entry being eliminated and `h = gcd(c, β)`; the row stays a non-zero
//! rational multiple of the rational reduction, so it ends at zero exactly
//! when that does.  A row is abandoned, and the candidate rejected, at the
//! first column it cannot reduce: that is a non-zero remainder.
//!
//! Choosing one reducer per monomial is a legitimate reduction strategy, and
//! reduction to zero under any strategy gives the standard representation
//! Buchberger's criterion asks for; if `B` is a Gröbner basis, every strategy
//! reduces every S-polynomial to zero.  So the check is exact in both
//! directions, as the term-by-term division it replaces was.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use rug::{Assign, Integer, Rational};

use crate::poly::groebner::modp::{MonoTable, Stop};
use crate::poly::groebner::multimodular::{IntPoly, Shape};
use crate::poly::groebner::pairs::{update_pairs, CriticalPair};

type Hasher = foldhash::fast::RandomState;

/// No reducer for this column.
const NONE: u32 = u32::MAX;

/// Rows of S-polynomials prepared and reduced together.
const PAIR_BATCH: usize = 512;

/// A polynomial over ℤ in the table: monomials decreasing, with coefficients.
struct ZPoly {
    mons: Vec<u32>,
    coefs: Vec<Integer>,
}

/// A row to reduce, before columns are assigned: terms (monomial, coefficient),
/// in any order, repeats allowed (they add up).
type ZTerms = Vec<(u32, Integer)>;

/// The primitive integer multiple of a polynomial with rational coefficients,
/// with a positive leading coefficient.
fn primitive(mons: &[u32], coefs: &[Rational]) -> ZPoly {
    let mut den = Integer::from(1);
    for c in coefs {
        den.lcm_mut(c.denom());
    }
    let mut ints: Vec<Integer> = coefs
        .iter()
        .map(|c| {
            let mut m = Integer::from(den.div_exact_ref(c.denom()));
            m *= c.numer();
            m
        })
        .collect();
    let mut g = Integer::new();
    for c in &ints {
        g.gcd_mut(c);
    }
    if ints.first().is_some_and(|c| *c < 0) {
        g = -g;
    }
    if g != 1 && g != 0 {
        for c in &mut ints {
            c.div_exact_mut(&g);
        }
    }
    ZPoly {
        mons: mons.to_vec(),
        coefs: ints,
    }
}

/// The leading monomials of the candidate, for reducer search.
struct Reducers {
    lead: Vec<u32>,
    sev: Vec<u64>,
    len: Vec<usize>,
}

impl Reducers {
    /// The sparsest candidate element whose leading monomial divides `m`.
    fn find(&self, table: &MonoTable, m: u32) -> Option<usize> {
        let not_m = !table.sev(m);
        let mut best: Option<usize> = None;
        for (i, &s) in self.sev.iter().enumerate() {
            if s & not_m != 0 || !table.divides(self.lead[i], m) {
                continue;
            }
            if best.map_or(true, |b| self.len[i] < self.len[b]) {
                best = Some(i);
            }
        }
        best
    }
}

/// A batch of rows laid out over columns, with one reducer per reducible
/// column.
struct ZMatrix {
    ncols: usize,
    /// For each column, the reducer row index or [`NONE`].
    piv_of_col: Vec<u32>,
    /// Reducer rows: candidate element and its columns (leading first).
    piv: Vec<(u32, Vec<u32>)>,
    /// Rows to reduce: (column, coefficient), repeats allowed.
    rows: Vec<Vec<(u32, Integer)>>,
}

/// Symbolic preprocessing for `rows` against the candidate.
fn preprocess(
    table: &mut MonoTable,
    cand: &[ZPoly],
    reducers: &Reducers,
    rows: Vec<ZTerms>,
    cache: &mut HashMap<u32, u32, Hasher>,
) -> Result<ZMatrix, Stop> {
    let mut seen: HashMap<u32, (), Hasher> = HashMap::with_hasher(Hasher::default());
    let mut queue: Vec<u32> = Vec::new();
    for r in &rows {
        for (m, _) in r {
            if seen.insert(*m, ()).is_none() {
                queue.push(*m);
            }
        }
    }
    // Reducer of each reducible monomial: (candidate element, row monomials).
    let mut red_of: HashMap<u32, (u32, Vec<u32>), Hasher> = HashMap::with_hasher(Hasher::default());
    let mut ticks = 0u32;
    while let Some(m) = queue.pop() {
        if crate::poly::groebner::reduce::poll_due(&mut ticks) && crate::budget::interrupted() {
            return Err(Stop::Interrupted);
        }
        let k = match cache.get(&m) {
            Some(&k) => k,
            None => {
                let k = reducers.find(table, m).map_or(NONE, |k| k as u32);
                cache.insert(m, k);
                k
            }
        };
        if k == NONE {
            continue;
        }
        let b = &cand[k as usize];
        let t = table.quo(m, b.mons[0]);
        let mut mons = Vec::with_capacity(b.mons.len());
        for &bm in &b.mons {
            let tm = table.mul(t, bm)?;
            mons.push(tm);
            if seen.insert(tm, ()).is_none() {
                queue.push(tm);
            }
        }
        red_of.insert(m, (k, mons));
    }
    let mut col_mon: Vec<u32> = seen.into_keys().collect();
    col_mon.sort_unstable_by(|&a, &b| table.cmp(b, a));
    let mut col_of: HashMap<u32, u32, Hasher> =
        HashMap::with_capacity_and_hasher(col_mon.len(), Hasher::default());
    for (c, &m) in col_mon.iter().enumerate() {
        col_of.insert(m, c as u32);
    }
    let mut piv_of_col = vec![NONE; col_mon.len()];
    let mut piv = Vec::with_capacity(red_of.len());
    for (_, (k, mons)) in red_of {
        let cols: Vec<u32> = mons.iter().map(|m| col_of[m]).collect();
        piv_of_col[cols[0] as usize] = piv.len() as u32;
        piv.push((k, cols));
    }
    let rows = rows
        .into_iter()
        .map(|r| r.into_iter().map(|(m, c)| (col_of[&m], c)).collect())
        .collect();
    Ok(ZMatrix {
        ncols: col_mon.len(),
        piv_of_col,
        piv,
        rows,
    })
}

/// Content removals are done after this many rescalings of a row.
const CONTENT_EVERY: u32 = 16;

/// Does every row of `mat` reduce to zero?  `None` if an armed budget fired.
fn rows_reduce_to_zero(mat: &ZMatrix, cand: &[ZPoly], rows: &[usize]) -> Option<bool> {
    let mut dense: Vec<Integer> = vec![Integer::new(); mat.ncols];
    let mut pending: Vec<bool> = vec![false; mat.ncols];
    let mut heap: BinaryHeap<Reverse<u32>> = BinaryHeap::new();
    let mut h = Integer::new();
    let mut scale = Integer::new();
    let mut mult = Integer::new();
    let mut ticks = 0u32;
    for &ri in rows {
        for (c, x) in &mat.rows[ri] {
            dense[*c as usize] += x;
            if !std::mem::replace(&mut pending[*c as usize], true) {
                heap.push(Reverse(*c));
            }
        }
        let mut scalings = 0u32;
        let mut zero = true;
        while let Some(Reverse(c)) = heap.pop() {
            let c = c as usize;
            pending[c] = false;
            if dense[c].is_zero() {
                continue;
            }
            if crate::poly::groebner::reduce::poll_due(&mut ticks) && crate::budget::interrupted() {
                return None;
            }
            let pv = mat.piv_of_col[c];
            if pv == NONE {
                zero = false;
                dense[c] = Integer::new();
                break;
            }
            let (k, cols) = &mat.piv[pv as usize];
            let b = &cand[*k as usize];
            let x = std::mem::take(&mut dense[c]);
            let beta = &b.coefs[0];
            h.assign(x.gcd_ref(beta));
            scale.assign(beta.div_exact_ref(&h));
            mult.assign(x.div_exact_ref(&h));
            if scale != 1 {
                for Reverse(c2) in heap.iter() {
                    dense[*c2 as usize] *= &scale;
                }
                scalings += 1;
            }
            for (&c2, y) in cols[1..].iter().zip(&b.coefs[1..]) {
                let d = &mut dense[c2 as usize];
                *d -= &mult * y;
                if !std::mem::replace(&mut pending[c2 as usize], true) {
                    heap.push(Reverse(c2));
                }
            }
            if scalings >= CONTENT_EVERY {
                scalings = 0;
                h.assign(0);
                for Reverse(c2) in heap.iter() {
                    h.gcd_mut(&dense[*c2 as usize]);
                    if h == 1 {
                        break;
                    }
                }
                if h > 1 {
                    for Reverse(c2) in heap.iter() {
                        dense[*c2 as usize].div_exact_mut(&h);
                    }
                }
            }
        }
        // Clear whatever is left (a row abandoned at a non-zero remainder).
        for Reverse(c2) in heap.drain() {
            pending[c2 as usize] = false;
            dense[c2 as usize] = Integer::new();
        }
        if !zero {
            return Some(false);
        }
    }
    Some(true)
}

/// Reduce the rows of one batch, over the worker threads.
fn batch_reduces_to_zero(
    table: &mut MonoTable,
    cand: &[ZPoly],
    reducers: &Reducers,
    rows: Vec<ZTerms>,
    cache: &mut HashMap<u32, u32, Hasher>,
) -> Result<bool, Stop> {
    if rows.is_empty() {
        return Ok(true);
    }
    let mat = preprocess(table, cand, reducers, rows, cache)?;
    let n = mat.rows.len();
    let width = super::multimodular::batch_width();
    let chunk = n.div_ceil(width).max(1);
    let jobs: Vec<Vec<usize>> = (0..n)
        .collect::<Vec<_>>()
        .chunks(chunk)
        .map(|c| c.to_vec())
        .collect();
    let res = super::multimodular::run_jobs(jobs, |rows| rows_reduce_to_zero(&mat, cand, &rows))
        .map_err(|()| Stop::Interrupted)?;
    let mut all = true;
    for r in res {
        match r {
            None => return Err(Stop::Interrupted),
            Some(false) => all = false,
            Some(true) => {}
        }
    }
    Ok(all)
}

/// A set of critical pairs of a basis with leading exponents `lead` whose
/// S-polynomials reducing to zero proves the basis a Gröbner basis: the pairs
/// `(i, j)` with leading monomials not coprime (Buchberger's first criterion)
/// and with no `k` such that `lm(k) | lcm(i, j)` while `lcm(i, k)` and
/// `lcm(j, k)` are both proper divisors of `lcm(i, j)` (his second criterion).
/// With every leading monomial known in advance the second criterion can use
/// any `k`: its witnesses have strictly smaller lcms, so the eliminations are
/// well founded and need no insertion order.  Among the pairs left with one
/// and the same lcm `L`, only a spanning forest is kept: along a path
/// `i = k₀, k₁, …, k_r = j` of pairs with lcm `L`, the S-polynomials of the
/// monic elements telescope to `S(i, j)`, all at `L`.
fn chain_pairs(lead: &[Vec<u32>]) -> Vec<(Vec<u32>, usize, usize)> {
    let n = lead.len();
    let lcm =
        |a: &[u32], b: &[u32]| -> Vec<u32> { a.iter().zip(b).map(|(x, y)| *x.max(y)).collect() };
    let divides = |a: &[u32], b: &[u32]| a.iter().zip(b).all(|(x, y)| x <= y);
    let mut kept: Vec<(Vec<u32>, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&lead[i], &lead[j]);
            if a.iter().zip(b.iter()).all(|(&x, &y)| x == 0 || y == 0) {
                continue;
            }
            let l = lcm(a, b);
            let covered = (0..n).any(|k| {
                k != i
                    && k != j
                    && divides(&lead[k], &l)
                    && lcm(a, &lead[k]) != l
                    && lcm(b, &lead[k]) != l
            });
            if !covered {
                kept.push((l, i, j));
            }
        }
    }
    // Spanning forest per lcm.
    kept.sort();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut out = Vec::with_capacity(kept.len());
    let mut start = 0;
    while start < kept.len() {
        let mut end = start;
        while end < kept.len() && kept[end].0 == kept[start].0 {
            end += 1;
        }
        let mut touched = Vec::new();
        for (l, i, j) in &kept[start..end] {
            let (ri, rj) = (find(&mut parent, *i), find(&mut parent, *j));
            if ri != rj {
                parent[ri] = rj;
                touched.push(*i);
                touched.push(*j);
                out.push((l.clone(), *i, *j));
            }
        }
        for t in touched {
            parent[t] = t;
        }
        start = end;
    }
    out
}

/// `true` when every input reduces to zero modulo the candidate given by
/// `shape` and `lift` (monomials in `table`, rational coefficients), and so
/// does every critical pair of the candidate left by the Gebauer–Möller
/// criteria: the candidate is a Gröbner basis of an ideal containing the
/// inputs.  Exact.  `Err(Stop::Interrupted)` when an armed budget fired,
/// `Err(Stop::Overflow)` when an exponent left the table's range (the caller
/// then falls back to the term-by-term check).
pub(crate) fn verify_lift(
    table: &mut MonoTable,
    inputs: &[IntPoly],
    shape: &Shape,
    lift: &[Rational],
) -> Result<bool, Stop> {
    let mut cand = Vec::with_capacity(shape.len());
    let mut at = 0usize;
    for mons in shape {
        if mons.is_empty() {
            return Ok(false);
        }
        cand.push(primitive(mons, &lift[at..at + mons.len()]));
        at += mons.len();
    }
    let reducers = Reducers {
        lead: cand.iter().map(|b| b.mons[0]).collect(),
        sev: cand.iter().map(|b| table.sev(b.mons[0])).collect(),
        len: cand.iter().map(|b| b.mons.len()).collect(),
    };
    let mut cache: HashMap<u32, u32, Hasher> = HashMap::with_hasher(Hasher::default());

    // I ⊆ ⟨B⟩: every input reduces to zero.
    let rows: Vec<ZTerms> = inputs
        .iter()
        .map(|g| {
            g.mons
                .iter()
                .copied()
                .zip(g.coefs.iter().cloned())
                .collect()
        })
        .collect();
    if !batch_reduces_to_zero(table, &cand, &reducers, rows, &mut cache)? {
        return Ok(false);
    }

    // B is a Gröbner basis: the critical pairs reduce to zero.
    let mut lead = Vec::with_capacity(cand.len());
    let mut sugar = Vec::with_capacity(cand.len());
    let mut pairs: Vec<CriticalPair> = Vec::new();
    for (k, b) in cand.iter().enumerate() {
        lead.push(table.exps_u32(b.mons[0]));
        sugar.push(b.mons.iter().map(|&m| table.deg(m)).max().unwrap_or(0));
        update_pairs(&lead, &sugar, &mut pairs, k);
    }
    // Two sufficient sets of pairs; use the smaller (each is valid on its
    // own, so they must not be mixed).
    let chains = chain_pairs(&lead);
    let mut pairs: Vec<(Vec<u32>, usize, usize)> = if chains.len() < pairs.len() {
        chains
    } else {
        pairs.into_iter().map(|p| (p.lcm_exp, p.i, p.j)).collect()
    };
    pairs.sort_by_key(|p| p.0.iter().sum::<u32>());
    for batch in pairs.chunks(PAIR_BATCH) {
        if crate::budget::interrupted() {
            return Err(Stop::Interrupted);
        }
        let mut rows: Vec<ZTerms> = Vec::with_capacity(batch.len());
        for (lcm, i, j) in batch {
            let l = table.intern_u32(lcm).ok_or(Stop::Overflow)?;
            let (bi, bj) = (&cand[*i], &cand[*j]);
            let ti = table.quo(l, bi.mons[0]);
            let tj = table.quo(l, bj.mons[0]);
            // (βj/h)·ti·bi − (βi/h)·tj·bj, leading terms cancelled.
            let h = Integer::from(bi.coefs[0].gcd_ref(&bj.coefs[0]));
            let ci = Integer::from(bj.coefs[0].div_exact_ref(&h));
            let cj = Integer::from(bi.coefs[0].div_exact_ref(&h));
            let mut row: ZTerms = Vec::with_capacity(bi.mons.len() + bj.mons.len());
            for (&m, c) in bi.mons[1..].iter().zip(&bi.coefs[1..]) {
                row.push((table.mul(ti, m)?, Integer::from(&ci * c)));
            }
            for (&m, c) in bj.mons[1..].iter().zip(&bj.coefs[1..]) {
                row.push((table.mul(tj, m)?, -Integer::from(&cj * c)));
            }
            rows.push(row);
        }
        if !batch_reduces_to_zero(table, &cand, &reducers, rows, &mut cache)? {
            return Ok(false);
        }
    }
    Ok(true)
}
