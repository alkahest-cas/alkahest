//! Putinar (quadratic-module) positivity certificates on semialgebraic sets.
//!
//! [`super::prove_nonneg`] with constraints searches Handelman certificates —
//! non-negative *constant* combinations of constraint products — which are an
//! exact LP but cannot express the ordinary textbook certificate for a
//! constrained inequality: `1 − x² ≥ 0` on the unit disk is
//! `y² + 1·(1 − x² − y²)`, and the `y²` is a sum of squares, not a constant.
//! This module searches the **quadratic module** instead:
//!
//! ```text
//! f = σ₀ + Σᵢ σᵢ·gᵢ + Σⱼ λⱼ·hⱼ       σ's sums of squares, λ's arbitrary polynomials
//! ```
//!
//! which proves `f ≥ 0` on `{gᵢ ≥ 0, hⱼ = 0}` because every summand is
//! non-negative there. At *level* `k` (Lasserre's relaxation order) every
//! summand has degree at most `2k`: `σ₀` is a Gram form over the monomials of
//! degree `≤ k`, `σᵢ` over those of degree `≤ ⌊(2k − deg gᵢ)/2⌋`, and `λⱼ`
//! ranges over all polynomials of degree `≤ 2k − deg hⱼ`.
//!
//! # Soundness
//!
//! The same discipline as the rest of [`super`]: floating point only ever
//! *proposes*. Each level is searched first by an exact rational LP over
//! scaled diagonally dominant Gram blocks (no floating point at all), then by
//! the numeric block-diagonal Gram search shared with [`super::psd`]
//! ([`super::psd::search_family_with`]), whose every candidate is rounded to
//! rationals, split into its blocks, decided PSD *exactly* by
//! [`super::linalg::psd_decompose`] (an exact `LDLᵀ`), and only accepted
//! when the full identity re-expands to `f` exactly in ℚ. The returned
//! [`PositivityCertificate`] goes through [`PositivityCertificate::verify`]
//! like every other certificate.
//!
//! An equality `h = 0` is recorded in the certificate as the two constraints
//! `h ≥ 0` and `−h ≥ 0`, with `λ = σ⁺ − σ⁻` split into two explicit sums of
//! squares (every polynomial is such a difference — see [`split_free`]). That
//! keeps the certificate's shape — every multiplier a weighted sum of squares
//! of rational polynomials — uniform with the other kinds, so the existing
//! verifier and Lean export apply unchanged.
//!
//! # The Archimedean condition
//!
//! Putinar's theorem guarantees a certificate at *some* level for every `f`
//! strictly positive on the set when the quadratic module is **Archimedean**,
//! i.e. contains `N − Σxᵢ²` for some `N`. When the plain search at a level
//! fails, this module looks for exactly that (with `N` left free, see
//! [`archimedean_bound`]), records the outcome in the trace, and — when a
//! bound is found — retries with `N − Σxᵢ²` as an additional generator. A
//! certificate found that way is composed back into one over the caller's
//! constraints only (products of sums of squares are sums of squares), so
//! the claim proved never mentions the auxiliary ball.

#![allow(clippy::needless_range_loop)]

use super::cert::{CertificateKind, Multiplier, PositivityCertificate, SosPoly};
use super::gram::{generators, monomial_basis};
use super::linalg::{psd_decompose, solve_affine};
use super::lp::{Lp, LpStatus, Rel};
use super::psd::{facial_reduction_with, search_family_with};
use super::ratpoly::{rational_expr, Exponents, RatPoly};
use super::{budget_ok, finish, format_point, var_names, SosError, SosOpts, SEARCH_TRACE_MARKER};
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Rational;
use std::collections::BTreeMap;

/// Search bounds for [`prove_nonneg_on`].
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct PutinarOpts {
    /// Relaxation order `k`: every summand of the certificate has degree at
    /// most `2k`. `None` starts at the smallest level that can represent
    /// the target and the constraints and tries [`AUTO_LEVEL_STEPS`] more.
    pub level: Option<u32>,
}

impl PutinarOpts {
    /// Options searching exactly `level` (when `Some`).
    pub fn with_level(level: Option<u32>) -> Self {
        PutinarOpts { level }
    }
}

/// Highest level accepted (explicitly or by automatic escalation).
pub const MAX_PUTINAR_LEVEL: u32 = 8;

/// Levels tried above the minimum one when `level` is `None`.
const AUTO_LEVEL_STEPS: u32 = 2;

/// Above this many packed Gram unknowns (summed over all blocks) the exact
/// coefficient-matching solve is skipped and the level reported as `NOT
/// SEARCHED`; the numeric search has its own free-parameter ceiling below
/// this ([`super::psd`]'s `MAX_FREE_PARAMETERS`).
const MAX_PACKED_UNKNOWNS: usize = 2500;

/// Above this many LP columns the exact DD stage is skipped (the numeric
/// stage still runs).
const MAX_LP_COLUMNS: usize = 4000;

// ---------------------------------------------------------------------------
// Problem statement
// ---------------------------------------------------------------------------

/// One `weight · σ` summand with `σ` a Gram form over `basis`.
#[derive(Clone, Debug)]
struct PsdBlock {
    weight: RatPoly,
    basis: Vec<Exponents>,
}

/// One `weight · λ` summand with `λ` an arbitrary polynomial supported on
/// `monomials`.
#[derive(Clone, Debug)]
struct FreeBlock {
    weight: RatPoly,
    monomials: Vec<Exponents>,
}

#[derive(Clone, Debug)]
struct QmProblem {
    nvars: usize,
    target: RatPoly,
    psd: Vec<PsdBlock>,
    free: Vec<FreeBlock>,
}

#[derive(Clone, Debug)]
struct QmSolution {
    psd: Vec<SosPoly>,
    free: Vec<RatPoly>,
}

impl QmProblem {
    fn rhs(&self, sol: &QmSolution) -> RatPoly {
        let mut acc = RatPoly::zero(self.nvars);
        for (blk, s) in self.psd.iter().zip(&sol.psd) {
            acc = acc.add(&blk.weight.mul(&s.to_poly(self.nvars)));
        }
        for (blk, l) in self.free.iter().zip(&sol.free) {
            acc = acc.add(&blk.weight.mul(l));
        }
        acc
    }

    /// The exact acceptance test every candidate passes through.
    fn accepts(&self, sol: &QmSolution) -> bool {
        sol.psd.len() == self.psd.len()
            && sol.free.len() == self.free.len()
            && sol.psd.iter().all(|s| s.weights_nonnegative())
            && self.rhs(sol) == self.target
    }
}

fn add_exp(a: &[u32], b: &[u32]) -> Exponents {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn pack_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Monomial basis of a Gram block whose weight has degree `wdeg`, at level
/// `k`: degree `≤ ⌊(2k − wdeg)/2⌋`, or `None` when the weight alone already
/// exceeds degree `2k`.
fn block_basis(nvars: usize, k: u32, wdeg: u32) -> Option<Vec<Exponents>> {
    if wdeg > 2 * k {
        return None;
    }
    Some(monomial_basis(nvars, (2 * k - wdeg) / 2))
}

/// The truncated quadratic module at level `k` for `target` over
/// `gs` (inequalities) and `hs` (equalities). Generators whose own degree
/// exceeds `2k` get no multiplier; each such omission is noted in `log`.
fn build_problem(
    target: &RatPoly,
    gs: &[RatPoly],
    hs: &[RatPoly],
    k: u32,
    log: &mut Vec<String>,
) -> QmProblem {
    let nvars = target.nvars();
    let mut psd = vec![PsdBlock {
        weight: RatPoly::one(nvars),
        basis: monomial_basis(nvars, k),
    }];
    for (i, g) in gs.iter().enumerate() {
        match block_basis(nvars, k, g.total_degree()) {
            Some(basis) => psd.push(PsdBlock {
                weight: g.clone(),
                basis,
            }),
            None => {
                log.push(format!(
                    "level {k}: constraint #{i} has degree {} > {}, so it gets no multiplier \
                     at this level",
                    g.total_degree(),
                    2 * k
                ));
                psd.push(PsdBlock {
                    weight: g.clone(),
                    basis: Vec::new(),
                });
            }
        }
    }
    let mut free = Vec::new();
    for (j, h) in hs.iter().enumerate() {
        let hd = h.total_degree();
        let monomials = if hd > 2 * k {
            log.push(format!(
                "level {k}: equality #{j} has degree {hd} > {}, so it gets no multiplier at \
                 this level",
                2 * k
            ));
            Vec::new()
        } else {
            monomial_basis(nvars, 2 * k - hd)
        };
        free.push(FreeBlock {
            weight: h.clone(),
            monomials,
        });
    }
    QmProblem {
        nvars,
        target: target.clone(),
        psd,
        free,
    }
}

// ---------------------------------------------------------------------------
// Stage 1: exact LP over scaled-diagonally-dominant Gram blocks
// ---------------------------------------------------------------------------

enum LpCol {
    /// A square `(Σ c_u z_u)²` of block `block`, as a basis-index linear form.
    Square {
        block: usize,
        form: Vec<(usize, Rational)>,
    },
    /// `±monomial` in the free multiplier of `block`.
    Free {
        block: usize,
        monomial: usize,
        sign: i32,
    },
}

/// Exact LP search: every Gram block restricted to the cone spanned by the
/// squares of [`generators`] (scaled DD), every free multiplier split into
/// positive and negative parts. No floating point anywhere.
fn lp_search(prob: &QmProblem, log: &mut Vec<String>) -> Option<QmSolution> {
    let nvars = prob.nvars;
    let mut cols: Vec<(LpCol, RatPoly)> = Vec::new();
    for (b, blk) in prob.psd.iter().enumerate() {
        let n = blk.basis.len();
        if n == 0 {
            continue;
        }
        for form in generators(n) {
            let mut lin = RatPoly::zero(nvars);
            for (u, c) in &form {
                lin = lin.add(&RatPoly::monomial(nvars, blk.basis[*u].clone(), c.clone()));
            }
            let poly = blk.weight.mul(&lin.square());
            cols.push((LpCol::Square { block: b, form }, poly));
        }
        if cols.len() > MAX_LP_COLUMNS {
            break;
        }
    }
    for (b, blk) in prob.free.iter().enumerate() {
        for (mi, m) in blk.monomials.iter().enumerate() {
            let poly = blk
                .weight
                .mul(&RatPoly::monomial(nvars, m.clone(), Rational::from(1)));
            for sign in [1, -1] {
                let p = if sign == 1 { poly.clone() } else { poly.neg() };
                cols.push((
                    LpCol::Free {
                        block: b,
                        monomial: mi,
                        sign,
                    },
                    p,
                ));
            }
        }
    }
    if cols.len() > MAX_LP_COLUMNS {
        log.push(format!(
            "exact DD stage: NOT SEARCHED — {} LP columns, above the ceiling of {MAX_LP_COLUMNS}",
            cols.len()
        ));
        return None;
    }
    let mut row_of: BTreeMap<Exponents, usize> = BTreeMap::new();
    for (_, p) in &cols {
        for e in p.terms().keys() {
            let next = row_of.len();
            row_of.entry(e.clone()).or_insert(next);
        }
    }
    for e in prob.target.terms().keys() {
        let next = row_of.len();
        row_of.entry(e.clone()).or_insert(next);
    }
    let nrows = row_of.len();
    let mut rows = vec![vec![Rational::from(0); cols.len()]; nrows];
    for (c, (_, p)) in cols.iter().enumerate() {
        for (e, coeff) in p.terms() {
            rows[row_of[e]][c] += coeff;
        }
    }
    let mut lp = Lp::new(cols.len());
    let mut rhs = vec![Rational::from(0); nrows];
    for (e, &r) in &row_of {
        rhs[r] = prob.target.coeff(e);
    }
    for (row, b) in rows.into_iter().zip(rhs) {
        lp.constrain(row, Rel::Eq, b);
    }
    for c in 0..cols.len() {
        lp.set_objective(c, Rational::from(1));
    }
    log.push(format!(
        "exact DD stage: LP with {} columns over {nrows} monomials",
        cols.len()
    ));
    let x = match lp.solve() {
        LpStatus::Optimal(x) => x,
        _ => {
            log.push("exact DD stage: infeasible".to_string());
            return None;
        }
    };
    let mut sol = QmSolution {
        psd: vec![SosPoly::default(); prob.psd.len()],
        free: vec![RatPoly::zero(nvars); prob.free.len()],
    };
    for ((col, _), w) in cols.iter().zip(x) {
        if w == 0 {
            continue;
        }
        match col {
            LpCol::Square { block, form } => {
                let basis = &prob.psd[*block].basis;
                let mut lin = RatPoly::zero(nvars);
                for (u, c) in form {
                    lin = lin.add(&RatPoly::monomial(nvars, basis[*u].clone(), c.clone()));
                }
                sol.psd[*block].push(w, lin);
            }
            LpCol::Free {
                block,
                monomial,
                sign,
            } => {
                let m = prob.free[*block].monomials[*monomial].clone();
                let coeff = if *sign == 1 { w } else { -w };
                sol.free[*block] = sol.free[*block].add(&RatPoly::monomial(nvars, m, coeff));
            }
        }
    }
    if prob.accepts(&sol) {
        log.push("exact DD stage: certificate found".to_string());
        Some(sol)
    } else {
        log.push("exact DD stage: LP point failed the exact identity check".to_string());
        None
    }
}

// ---------------------------------------------------------------------------
// Stage 2: numeric block-diagonal PSD Gram search, exactly rounded
// ---------------------------------------------------------------------------

/// Extract block `b` (rows/cols `off..off+n`) of a block-diagonal matrix.
fn sub_block(q: &[Vec<Rational>], off: usize, n: usize) -> Vec<Vec<Rational>> {
    (0..n).map(|i| q[off + i][off..off + n].to_vec()).collect()
}

fn sdp_search(prob: &QmProblem, log: &mut Vec<String>) -> Option<QmSolution> {
    let nvars = prob.nvars;
    let sizes: Vec<usize> = prob.psd.iter().map(|b| b.basis.len()).collect();
    let packed: usize = sizes.iter().map(|&n| pack_len(n)).sum();
    if packed > MAX_PACKED_UNKNOWNS {
        log.push(format!(
            "PSD Gram search: NOT SEARCHED — {packed} Gram unknowns across all blocks, above \
             the ceiling of {MAX_PACKED_UNKNOWNS}"
        ));
        return None;
    }
    let dim: usize = sizes.iter().sum();
    if dim == 0 {
        return None;
    }

    // Column bookkeeping: packed (i ≤ j) entry of block b → global column,
    // and the block-diagonal matrix position it occupies.
    let mut col_pos: Vec<(usize, usize)> = Vec::with_capacity(packed);
    let mut col_poly: Vec<RatPoly> = Vec::with_capacity(packed);
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut off = 0usize;
    for blk in &prob.psd {
        offsets.push(off);
        let n = blk.basis.len();
        for i in 0..n {
            for j in i..n {
                let mult = if i == j { 1 } else { 2 };
                let mono = RatPoly::monomial(
                    nvars,
                    add_exp(&blk.basis[i], &blk.basis[j]),
                    Rational::from(mult),
                );
                col_pos.push((off + i, off + j));
                col_poly.push(blk.weight.mul(&mono));
            }
        }
        off += n;
    }
    // Free-multiplier columns.
    let mut free_cols: Vec<(usize, usize, RatPoly)> = Vec::new();
    for (b, blk) in prob.free.iter().enumerate() {
        for (mi, m) in blk.monomials.iter().enumerate() {
            let p = blk
                .weight
                .mul(&RatPoly::monomial(nvars, m.clone(), Rational::from(1)));
            free_cols.push((b, mi, p));
        }
    }

    let mut row_of: BTreeMap<Exponents, usize> = BTreeMap::new();
    let polys = col_poly
        .iter()
        .chain(free_cols.iter().map(|(_, _, p)| p))
        .chain(std::iter::once(&prob.target));
    for p in polys {
        for e in p.terms().keys() {
            let next = row_of.len();
            row_of.entry(e.clone()).or_insert(next);
        }
    }
    let nrows = row_of.len();
    let mut a_gram = vec![vec![Rational::from(0); packed]; nrows];
    for (c, p) in col_poly.iter().enumerate() {
        for (e, coeff) in p.terms() {
            a_gram[row_of[e]][c] += coeff;
        }
    }
    let mut a_free = vec![vec![Rational::from(0); free_cols.len()]; nrows];
    for (c, (_, _, p)) in free_cols.iter().enumerate() {
        for (e, coeff) in p.terms() {
            a_free[row_of[e]][c] += coeff;
        }
    }
    let mut rhs = vec![Rational::from(0); nrows];
    for (e, &r) in &row_of {
        rhs[r] = prob.target.coeff(e);
    }

    // Eliminate the free multipliers: the Gram entries must satisfy
    // `yᵀ·(A_gram·x − f) = 0` for every `y` with `yᵀ·A_free = 0`; the free
    // coefficients are then recovered exactly for each candidate.
    let (rows, rhs_red): (Vec<Vec<Rational>>, Vec<Rational>) = if free_cols.is_empty() {
        (a_gram, rhs.clone())
    } else {
        let at: Vec<Vec<Rational>> = (0..free_cols.len())
            .map(|c| (0..nrows).map(|r| a_free[r][c].clone()).collect())
            .collect();
        let left = solve_affine(&at, &vec![Rational::from(0); free_cols.len()])?;
        let mut rows = Vec::with_capacity(left.nullspace.len());
        let mut rr = Vec::with_capacity(left.nullspace.len());
        for y in &left.nullspace {
            let mut row = vec![Rational::from(0); packed];
            let mut b = Rational::from(0);
            for r in 0..nrows {
                if y[r] == 0 {
                    continue;
                }
                for c in 0..packed {
                    if a_gram[r][c] != 0 {
                        row[c] += Rational::from(&y[r] * &a_gram[r][c]);
                    }
                }
                b += Rational::from(&y[r] * &rhs[r]);
            }
            rows.push(row);
            rr.push(b);
        }
        (rows, rr)
    };

    let sol = if rows.is_empty() {
        super::linalg::AffineSolution {
            particular: vec![Rational::from(0); packed],
            nullspace: (0..packed)
                .map(|k| {
                    let mut v = vec![Rational::from(0); packed];
                    v[k] = Rational::from(1);
                    v
                })
                .collect(),
        }
    } else {
        match solve_affine(&rows, &rhs_red) {
            Some(s) => s,
            None => {
                log.push(
                    "PSD Gram search: the coefficient-matching system is inconsistent, so no \
                     identity of this shape exists at this level, PSD or not"
                        .to_string(),
                );
                return None;
            }
        }
    };

    let unpack = |v: &[Rational]| -> Vec<Vec<Rational>> {
        let mut q = vec![vec![Rational::from(0); dim]; dim];
        for (c, &(i, j)) in col_pos.iter().enumerate() {
            q[i][j] = v[c].clone();
            q[j][i] = v[c].clone();
        }
        q
    };
    let base = unpack(&sol.particular);
    let dirs: Vec<Vec<Vec<Rational>>> = sol.nullspace.iter().map(|d| unpack(d)).collect();

    let decode = |q: &[Vec<Rational>]| -> Option<QmSolution> {
        let mut psd = Vec::with_capacity(prob.psd.len());
        for (b, blk) in prob.psd.iter().enumerate() {
            let n = blk.basis.len();
            let mut sos = SosPoly::default();
            if n > 0 {
                let decomp = psd_decompose(&sub_block(q, offsets[b], n))?;
                for (d, v) in decomp {
                    if d <= 0 {
                        continue;
                    }
                    let mut square = RatPoly::zero(nvars);
                    for (u, c) in v.iter().enumerate() {
                        if *c != 0 {
                            square = square.add(&RatPoly::monomial(
                                nvars,
                                blk.basis[u].clone(),
                                c.clone(),
                            ));
                        }
                    }
                    sos.push(d, square);
                }
            }
            psd.push(sos);
        }
        let mut free: Vec<RatPoly> = vec![RatPoly::zero(nvars); prob.free.len()];
        if !free_cols.is_empty() {
            let mut partial = RatPoly::zero(nvars);
            for (blk, s) in prob.psd.iter().zip(&psd) {
                partial = partial.add(&blk.weight.mul(&s.to_poly(nvars)));
            }
            let residual = prob.target.sub(&partial);
            let mut r = vec![Rational::from(0); nrows];
            for (e, c) in residual.terms() {
                r[*row_of.get(e)?] = c.clone();
            }
            let u = solve_affine(&a_free, &r)?;
            for ((b, mi, _), coeff) in free_cols.iter().zip(&u.particular) {
                if *coeff != 0 {
                    let m = prob.free[*b].monomials[*mi].clone();
                    free[*b] = free[*b].add(&RatPoly::monomial(nvars, m, coeff.clone()));
                }
            }
        }
        let cand = QmSolution { psd, free };
        if prob.accepts(&cand) {
            Some(cand)
        } else {
            None
        }
    };

    let (found, candidates) = search_family_with(&base, &dirs, &decode, log);
    if found.is_some() {
        return found;
    }
    if candidates.is_empty() || !budget_ok() {
        return None;
    }
    let search = |b: &[Vec<Rational>], d: &[Vec<Vec<Rational>>], log: &mut Vec<String>| {
        search_family_with(b, d, &decode, log).0
    };
    let found = facial_reduction_with(&base, &dirs, &candidates, &search, log);
    if found.is_some() {
        log.push("PSD Gram search: facial reduction found the certificate".to_string());
    }
    found
}

/// Both stages, in order.
fn search_level(prob: &QmProblem, log: &mut Vec<String>) -> Option<QmSolution> {
    if let Some(sol) = lp_search(prob, log) {
        return Some(sol);
    }
    if !budget_ok() {
        return None;
    }
    sdp_search(prob, log)
}

// ---------------------------------------------------------------------------
// The Archimedean bound
// ---------------------------------------------------------------------------

/// Look for `N − Σxᵢ² = σ₀ + Σσᵢgᵢ + Σλⱼhⱼ` at level `k` with `N` free (a
/// `(−1)·c` block with `c ≥ 0` absorbs it), returning `N` and the identity's
/// multipliers for [`build_problem`]'s block layout over `gs`/`hs`.
fn archimedean_bound(
    gs: &[RatPoly],
    hs: &[RatPoly],
    nvars: usize,
    k: u32,
    log: &mut Vec<String>,
) -> Option<(Rational, QmSolution)> {
    if k == 0 {
        return None;
    }
    let target = RatPoly::sum_of_squares(nvars).neg();
    let mut scratch = Vec::new();
    let mut prob = build_problem(&target, gs, hs, k, &mut scratch);
    prob.psd.push(PsdBlock {
        weight: RatPoly::constant(nvars, Rational::from(-1)),
        basis: vec![vec![0; nvars]],
    });
    let mut inner = Vec::new();
    let mut sol = search_level(&prob, &mut inner)?;
    let slack = sol.psd.pop()?;
    let n = slack.to_poly(nvars).as_constant()?;
    if n <= 0 {
        log.push(
            "Archimedean check: the constraints certify a set inside the origin (or empty); \
             no ball generator is used"
                .to_string(),
        );
        return None;
    }
    Some((n, sol))
}

/// `(Σ aⱼ qⱼ²)·(Σ b_k r_k²) = Σ aⱼb_k (qⱼr_k)²`, exactly.
fn sos_product(a: &SosPoly, b: &SosPoly) -> SosPoly {
    let mut out = SosPoly::default();
    for s in &a.terms {
        for t in &b.terms {
            out.push(Rational::from(&s.coeff * &t.coeff), s.square.mul(&t.square));
        }
    }
    out
}

fn sos_concat(a: &mut SosPoly, b: SosPoly) {
    for t in b.terms {
        a.push(t.coeff, t.square);
    }
}

// ---------------------------------------------------------------------------
// Equality multipliers as differences of sums of squares
// ---------------------------------------------------------------------------

/// Write any polynomial `λ` as `σ⁺ − σ⁻` with `σ±` explicit sums of squares:
/// `c·x^a` with `a` even is `±c·(x^{a/2})²`; otherwise `a = b + d` with
/// `b ≠ d` and `x^b·x^d = ¼(x^b + x^d)² − ¼(x^b − x^d)²`. Exact.
fn split_free(lambda: &RatPoly) -> (SosPoly, SosPoly) {
    let nvars = lambda.nvars();
    let mut pos = SosPoly::default();
    let mut neg = SosPoly::default();
    let quarter = Rational::from((1, 4));
    for (a, c) in lambda.terms() {
        if a.iter().all(|e| e % 2 == 0) {
            let half: Exponents = a.iter().map(|e| e / 2).collect();
            let sq = RatPoly::monomial(nvars, half, Rational::from(1));
            if *c > 0 {
                pos.push(c.clone(), sq);
            } else {
                neg.push(Rational::from(-c), sq);
            }
            continue;
        }
        let b: Exponents = a.iter().map(|e| e / 2).collect();
        let d: Exponents = a.iter().zip(&b).map(|(e, h)| e - h).collect();
        let xb = RatPoly::monomial(nvars, b, Rational::from(1));
        let xd = RatPoly::monomial(nvars, d, Rational::from(1));
        let plus = xb.add(&xd);
        let minus = xb.sub(&xd);
        let w = Rational::from(c.clone().abs() * &quarter);
        if *c > 0 {
            pos.push(w.clone(), plus);
            neg.push(w, minus);
        } else {
            pos.push(w.clone(), minus);
            neg.push(w, plus);
        }
    }
    (pos, neg)
}

// ---------------------------------------------------------------------------
// Refutation
// ---------------------------------------------------------------------------

/// Candidate coordinate values for the witness grid, simplest first.
fn grid_values(nvars: usize) -> Vec<Rational> {
    let dens: &[i64] = match nvars {
        0..=2 => &[1, 2, 3, 4, 5, 8],
        3 => &[1, 2, 4],
        4 => &[1, 2],
        5..=6 => &[1],
        _ => &[],
    };
    let mut vals: Vec<Rational> = Vec::new();
    if dens.is_empty() {
        // 3^n points: -1, 0, 1.
        return vec![Rational::from(0), Rational::from(1), Rational::from(-1)];
    }
    for &d in dens {
        for k in -(3 * d)..=(3 * d) {
            let v = Rational::from((k, d));
            if !vals.contains(&v) {
                vals.push(v);
            }
        }
    }
    vals.sort_by(|a, b| {
        let ka = (a.denom().clone(), a.clone().abs());
        let kb = (b.denom().clone(), b.clone().abs());
        ka.cmp(&kb).then_with(|| b.cmp(a))
    });
    if nvars == 6 {
        vals.truncate(5);
    } else if nvars == 5 {
        vals.truncate(7);
    }
    vals
}

/// An exact rational point satisfying every `g ≥ 0` and `h = 0` at which
/// `target < 0`, found on a bounded grid; `None` when the grid has none
/// (which proves nothing).
fn violation_point(target: &RatPoly, gs: &[RatPoly], hs: &[RatPoly]) -> Option<Vec<Rational>> {
    let nvars = target.nvars();
    if nvars == 0 || nvars > 9 {
        return None;
    }
    let vals = grid_values(nvars);
    let per = vals.len();
    let total = per.checked_pow(nvars as u32)?;
    if total > 40_000 {
        return None;
    }
    let mut best: Option<(Rational, Vec<Rational>)> = None;
    for idx in 0..total {
        let mut rest = idx;
        let mut point = Vec::with_capacity(nvars);
        for _ in 0..nvars {
            point.push(vals[rest % per].clone());
            rest /= per;
        }
        if hs.iter().any(|h| h.eval(&point) != 0) || gs.iter().any(|g| g.eval(&point) < 0) {
            continue;
        }
        let v = target.eval(&point);
        if v < 0 && best.as_ref().map_or(true, |(bv, _)| v < *bv) {
            best = Some((v, point));
        }
    }
    best.map(|(_, p)| p)
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// A parsed constraint: `g ≥ 0`, or `h = 0`.
enum Parsed {
    Ineq(RatPoly, bool),
    Eq(RatPoly),
}

fn parse_constraint(c: ExprId, vars: &[ExprId], pool: &ExprPool) -> Result<Parsed, SosError> {
    let pred = pool.with(c, |data| match data {
        ExprData::Predicate { kind, args } => Some((kind.clone(), args.clone())),
        _ => None,
    });
    let poly = |e: ExprId| RatPoly::from_expr(e, vars, pool).map_err(SosError::NotPolynomial);
    match pred {
        None => Ok(Parsed::Ineq(poly(c)?, false)),
        Some((kind, args)) => {
            if args.len() != 2 {
                return Err(SosError::InvalidInput(format!(
                    "constraint `{}` is not a binary comparison; pass g (meaning g ≥ 0), \
                     a ≥ b, a ≤ b or a = b",
                    pool.display(c)
                )));
            }
            let (a, b) = (poly(args[0])?, poly(args[1])?);
            match kind {
                PredicateKind::Ge => Ok(Parsed::Ineq(a.sub(&b), false)),
                PredicateKind::Le => Ok(Parsed::Ineq(b.sub(&a), false)),
                PredicateKind::Gt => Ok(Parsed::Ineq(a.sub(&b), true)),
                PredicateKind::Lt => Ok(Parsed::Ineq(b.sub(&a), true)),
                PredicateKind::Eq => Ok(Parsed::Eq(a.sub(&b))),
                _ => Err(SosError::InvalidInput(format!(
                    "constraint `{}` is not ≥, ≤, >, < or =",
                    pool.display(c)
                ))),
            }
        }
    }
}

struct Parsed3 {
    target: RatPoly,
    gs: Vec<RatPoly>,
    hs: Vec<RatPoly>,
    strict: usize,
}

fn parse_all(
    expr: ExprId,
    constraints: &[ExprId],
    equalities: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<Parsed3, SosError> {
    if vars.is_empty() {
        return Err(SosError::InvalidInput(
            "at least one variable is required".into(),
        ));
    }
    let target = RatPoly::from_expr(expr, vars, pool).map_err(SosError::NotPolynomial)?;
    let mut gs = Vec::new();
    let mut hs = Vec::new();
    let mut strict = 0;
    for &c in constraints {
        match parse_constraint(c, vars, pool)? {
            Parsed::Ineq(g, s) => {
                strict += usize::from(s);
                gs.push(g);
            }
            Parsed::Eq(h) => hs.push(h),
        }
    }
    for &e in equalities {
        match parse_constraint(e, vars, pool)? {
            Parsed::Ineq(h, false) | Parsed::Eq(h) => hs.push(h),
            Parsed::Ineq(_, true) => {
                return Err(SosError::InvalidInput(format!(
                    "`{}` was passed as an equality but is a strict inequality",
                    pool.display(e)
                )))
            }
        }
    }
    // A zero equality is vacuous and a zero inequality is `0 ≥ 0`.
    gs.retain(|g| !g.is_zero());
    hs.retain(|h| !h.is_zero());
    Ok(Parsed3 {
        target,
        gs,
        hs,
        strict,
    })
}

/// A point of `{g ≥ 0, h = 0}` at which `expr < 0`, as rational constants in
/// `pool` (one per variable), when a bounded rational grid has one.
///
/// `Ok(None)` proves nothing. Constraints are read exactly as
/// [`prove_nonneg_on`] reads them.
pub fn nonneg_on_witness(
    expr: ExprId,
    constraints: &[ExprId],
    equalities: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<Option<Vec<ExprId>>, SosError> {
    let p = parse_all(expr, constraints, equalities, vars, pool)?;
    Ok(violation_point(&p.target, &p.gs, &p.hs)
        .map(|pt| pt.iter().map(|c| rational_expr(c, pool)).collect()))
}

/// Prove `expr ≥ 0` on `{x : gᵢ(x) ≥ 0, hⱼ(x) = 0}` with a **Putinar**
/// certificate `expr = σ₀ + Σ σᵢ·gᵢ + Σ λⱼ·hⱼ` (σ's sums of squares of
/// rational polynomials, λ's arbitrary rational polynomials).
///
/// `constraints` holds either bare polynomials `g` (meaning `g ≥ 0`) or
/// comparison predicates `a ≥ b`, `a ≤ b`, `a = b` (strict `>`/`<` are
/// relaxed to their closure, which is sound: a bound on the closure holds on
/// the set). `equalities` holds polynomials `h` (meaning `h = 0`) or `a = b`.
/// With neither, this is [`super::sos_decompose`].
///
/// Outcomes, kept distinct as everywhere in [`super`]:
///
/// * `Ok(cert)` with `cert.kind == CertificateKind::Putinar` — proved, and
///   already exactly re-verified. Equalities appear in `cert.constraints` as
///   the pair `h ≥ 0`, `−h ≥ 0`.
/// * `Err(SosError::Negative)` (`E-SOS-003`) — a rational point of the set
///   with `expr < 0` (see [`nonneg_on_witness`] for it as values).
/// * `Err(SosError::NoCertificate)` (`E-SOS-002`) — nothing found at the
///   level(s) searched; undecided, never a refutation. The trace says what
///   ran, and whether the module was found to be Archimedean.
pub fn prove_nonneg_on(
    expr: ExprId,
    constraints: &[ExprId],
    equalities: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
    opts: &PutinarOpts,
) -> Result<PositivityCertificate, SosError> {
    crate::budget::clear_trip();
    let Parsed3 {
        target,
        gs,
        hs,
        strict,
    } = parse_all(expr, constraints, equalities, vars, pool)?;
    if let Some(level) = opts.level {
        if level == 0 || level > MAX_PUTINAR_LEVEL {
            return Err(SosError::InvalidInput(format!(
                "level must be between 1 and {MAX_PUTINAR_LEVEL}"
            )));
        }
    }
    if gs.is_empty() && hs.is_empty() {
        let o = SosOpts {
            basis_degree: opts.level,
            ..SosOpts::default()
        };
        return super::sos_decompose(expr, vars, pool, &o);
    }
    let names = var_names(vars, pool);
    let nvars = vars.len();

    if let Some(point) = violation_point(&target, &gs, &hs) {
        return Err(SosError::Negative(format!(
            "p({}) = {} < 0 at a point satisfying every constraint",
            format_point(&names, &point),
            target.eval(&point)
        )));
    }

    let mut log: Vec<String> = Vec::new();
    if strict > 0 {
        log.push(format!(
            "{strict} strict inequalit{} relaxed to non-strict (a certificate on the closure \
             holds on the set)",
            if strict == 1 { "y" } else { "ies" }
        ));
    }
    let target_min = target.total_degree().div_ceil(2).max(1);
    let gen_min = gs
        .iter()
        .chain(&hs)
        .map(|g| g.total_degree().div_ceil(2))
        .max()
        .unwrap_or(1)
        .max(1);
    let (levels, explicit): (Vec<u32>, bool) = match opts.level {
        Some(k) => (vec![k], true),
        None => {
            let lo = target_min.max(gen_min);
            if lo > MAX_PUTINAR_LEVEL {
                return Err(SosError::InvalidInput(format!(
                    "the target or a constraint needs level {lo}, above the maximum of \
                     {MAX_PUTINAR_LEVEL}"
                )));
            }
            (
                (lo..=(lo + AUTO_LEVEL_STEPS).min(MAX_PUTINAR_LEVEL)).collect(),
                false,
            )
        }
    };

    let mut arch_note: Option<String> = None;
    for &k in &levels {
        if !budget_ok() {
            break;
        }
        if 2 * k < target.total_degree() {
            log.push(format!(
                "level {k}: NOT SEARCHED — every summand has degree ≤ {}, but the target has \
                 degree {}; the smallest level that can represent it is {target_min}",
                2 * k,
                target.total_degree()
            ));
            continue;
        }
        log.push(format!(
            "level {k}: searching f = σ₀ + Σσᵢgᵢ + Σλⱼhⱼ with every summand of degree ≤ {}",
            2 * k
        ));
        let prob = build_problem(&target, &gs, &hs, k, &mut log);
        if let Some(sol) = search_level(&prob, &mut log) {
            log.push(format!("level {k}: certificate found"));
            return finish(assemble(
                vars, &names, &target, &gs, &hs, k, &sol.psd, &sol.free, log,
            ));
        }
        if !budget_ok() {
            break;
        }

        // Archimedean check, then a retry with the ball as an extra generator.
        let mut arch_log = Vec::new();
        match archimedean_bound(&gs, &hs, nvars, k, &mut arch_log) {
            Some((n_bound, arch)) => {
                arch_note = Some(format!(
                    "the quadratic module is Archimedean: {n_bound} − Σxᵢ² was certified at \
                     level {k}, so by Putinar's theorem every f strictly positive on the set \
                     has a certificate at some level (raising level is a meaningful retry)"
                ));
                log.push(format!(
                    "level {k}: Archimedean check: {n_bound} − Σxᵢ² ∈ QM"
                ));
                log.extend(arch_log);
                let ball =
                    RatPoly::constant(nvars, n_bound.clone()).sub(&RatPoly::sum_of_squares(nvars));
                let redundant = gs
                    .iter()
                    .any(|g| ball.sub(g).as_constant().is_some_and(|c| c >= 0));
                if redundant {
                    log.push(format!(
                        "level {k}: a constraint already is a ball; no extra generator added"
                    ));
                    continue;
                }
                if !budget_ok() {
                    break;
                }
                let mut gs_aug = gs.clone();
                gs_aug.push(ball);
                let mut aug_log = Vec::new();
                let prob_aug = build_problem(&target, &gs_aug, &hs, k, &mut aug_log);
                log.push(format!(
                    "level {k}: retrying with {n_bound} − Σxᵢ² as an additional generator"
                ));
                if let Some(mut sol) = search_level(&prob_aug, &mut log) {
                    // Compose σ*·(N − Σx²) = σ*·(τ₀ + Σ τᵢgᵢ + Σ μⱼhⱼ) back in.
                    let star = sol.psd.pop().unwrap_or_default();
                    let star_poly = star.to_poly(nvars);
                    let mut psd = sol.psd;
                    for (b, tau) in arch.psd.iter().enumerate() {
                        sos_concat(&mut psd[b], sos_product(&star, tau));
                    }
                    for (j, mu) in arch.free.iter().enumerate() {
                        sol.free[j] = sol.free[j].add(&star_poly.mul(mu));
                    }
                    log.push(format!(
                        "level {k}: certificate found with the ball generator; composed with the \
                         ball's own certificate so it uses only the given constraints (its \
                         summands may exceed degree {})",
                        2 * k
                    ));
                    return finish(assemble(
                        vars, &names, &target, &gs, &hs, k, &psd, &sol.free, log,
                    ));
                }
            }
            None => {
                log.push(format!(
                    "level {k}: Archimedean check: no bound N − Σxᵢ² was certified at this level"
                ));
                if arch_note.is_none() {
                    arch_note = Some(
                        "no bound N − Σxᵢ² could be certified from the constraints at the levels \
                         searched, so the quadratic module is not known to be Archimedean: the \
                         set may be unbounded, and Putinar's theorem then gives no guarantee \
                         that any level suffices"
                            .to_string(),
                    );
                }
            }
        }
    }

    let tried = if explicit {
        format!("level {}", levels[0])
    } else {
        format!(
            "levels {}..={}",
            levels.first().copied().unwrap_or(1),
            levels.last().copied().unwrap_or(1)
        )
    };
    let next = levels.last().copied().unwrap_or(1) + 1;
    let trace = log
        .iter()
        .map(|line| format!("\n  - {line}"))
        .collect::<String>();
    Err(SosError::NoCertificate(format!(
        "undecided, not a refutation — no Putinar certificate f = σ₀ + Σσᵢgᵢ + Σλⱼhⱼ was found \
         at {tried}. {} Raise the level (try level={next}); this is not a proof that f is \
         negative anywhere on the set.\n{SEARCH_TRACE_MARKER}{trace}",
        arch_note.unwrap_or_default()
    )))
}

/// Build the certificate object from a solution over [`build_problem`]'s
/// layout (σ₀, then one block per `g`; one free multiplier per `h`).
#[allow(clippy::too_many_arguments)]
fn assemble(
    vars: &[ExprId],
    names: &[String],
    target: &RatPoly,
    gs: &[RatPoly],
    hs: &[RatPoly],
    k: u32,
    psd: &[SosPoly],
    free: &[RatPoly],
    mut log: Vec<String>,
) -> PositivityCertificate {
    let m = gs.len();
    let mut constraints: Vec<RatPoly> = gs.to_vec();
    for h in hs {
        constraints.push(h.clone());
        constraints.push(h.neg());
    }
    let mut terms = Vec::new();
    for (b, sos) in psd.iter().enumerate() {
        if sos.is_empty() {
            continue;
        }
        terms.push(Multiplier {
            constraints: if b == 0 { Vec::new() } else { vec![b - 1] },
            sos: sos.clone(),
        });
    }
    for (j, lambda) in free.iter().enumerate() {
        if lambda.is_zero() {
            continue;
        }
        let (pos, neg) = split_free(lambda);
        log.push(format!(
            "equality #{j} (constraints #{} and #{} are h ≥ 0 and −h ≥ 0): λ = {}",
            m + 2 * j,
            m + 2 * j + 1,
            lambda.display(names)
        ));
        if !pos.is_empty() {
            terms.push(Multiplier {
                constraints: vec![m + 2 * j],
                sos: pos,
            });
        }
        if !neg.is_empty() {
            terms.push(Multiplier {
                constraints: vec![m + 2 * j + 1],
                sos: neg,
            });
        }
    }
    PositivityCertificate {
        vars: vars.to_vec(),
        var_names: names.to_vec(),
        target: target.clone(),
        constraints,
        kind: CertificateKind::Putinar,
        degree: k,
        terms,
        log,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::kernel::Domain;

    struct Ctx {
        pool: ExprPool,
        x: ExprId,
        y: ExprId,
    }

    fn ctx() -> Ctx {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        Ctx { pool, x, y }
    }

    impl Ctx {
        fn int(&self, n: i64) -> ExprId {
            self.pool.integer(n)
        }
        fn pw(&self, e: ExprId, n: i64) -> ExprId {
            self.pool.pow(e, self.pool.integer(n))
        }
        fn neg(&self, e: ExprId) -> ExprId {
            self.pool.mul(vec![self.int(-1), e])
        }
        /// 1 − x² − y²
        fn disk(&self) -> ExprId {
            self.pool.add(vec![
                self.int(1),
                self.neg(self.pw(self.x, 2)),
                self.neg(self.pw(self.y, 2)),
            ])
        }
    }

    fn assert_putinar(cert: &PositivityCertificate) {
        assert_eq!(cert.kind, CertificateKind::Putinar);
        cert.verify().expect("re-expands exactly");
        assert_eq!(
            cert.expand(),
            cert.target,
            "a direct identity, no multiplier"
        );
        assert!(cert.multiplier().is_none());
    }

    #[test]
    fn one_minus_x_squared_on_the_disk() {
        // The report's repro: 1 − x² = y² + 1·(1 − x² − y²). Handelman
        // cannot prove this at any level.
        let c = ctx();
        let f = c.pool.add(vec![c.int(1), c.neg(c.pw(c.x, 2))]);
        let cert = prove_nonneg_on(
            f,
            &[c.disk()],
            &[],
            &[c.x, c.y],
            &c.pool,
            &PutinarOpts::with_level(Some(1)),
        )
        .expect("certificate");
        assert_putinar(&cert);
        let handelman = super::super::prove_nonneg(
            f,
            &[c.disk()],
            &[c.x, c.y],
            &c.pool,
            &SosOpts {
                basis_degree: None,
                level: 4,
            },
        );
        assert!(handelman.is_err(), "Handelman alone cannot do this");
    }

    #[test]
    fn one_minus_quartics_on_the_disk() {
        let c = ctx();
        let f = c
            .pool
            .add(vec![c.int(1), c.neg(c.pw(c.x, 4)), c.neg(c.pw(c.y, 4))]);
        let cert = prove_nonneg_on(
            f,
            &[c.disk()],
            &[],
            &[c.x, c.y],
            &c.pool,
            &PutinarOpts::default(),
        )
        .expect("certificate");
        assert_putinar(&cert);
        assert_eq!(cert.degree, 2);
        let lean = cert.to_lean().expect("lean renders");
        assert!(lean.contains("hg0") && !lean.contains("sorry"));
    }

    #[test]
    fn x_on_the_unit_interval_from_one_quadratic_constraint() {
        // x ≥ 0 on {x(1 − x) ≥ 0} = [0, 1]: x = x² + 1·x(1 − x).
        let c = ctx();
        let g = c
            .pool
            .mul(vec![c.x, c.pool.add(vec![c.int(1), c.neg(c.x)])]);
        let cert = prove_nonneg_on(c.x, &[g], &[], &[c.x], &c.pool, &PutinarOpts::default())
            .expect("certificate");
        assert_putinar(&cert);
    }

    #[test]
    fn a_case_that_needs_level_two() {
        // 1 − x² on {1 − x ≥ 0, 1 + x ≥ 0}: at level 1 the multipliers of the
        // linear constraints are constants and the x² coefficient cannot be
        // matched; level 2 has ½(1+x)²(1−x) + ½(1−x)²(1+x).
        let c = ctx();
        let f = c.pool.add(vec![c.int(1), c.neg(c.pw(c.x, 2))]);
        let g1 = c.pool.add(vec![c.int(1), c.neg(c.x)]);
        let g2 = c.pool.add(vec![c.int(1), c.x]);
        let vars = [c.x];
        let err = prove_nonneg_on(
            f,
            &[g1, g2],
            &[],
            &vars,
            &c.pool,
            &PutinarOpts::with_level(Some(1)),
        )
        .expect_err("level 1 is too low");
        assert_eq!(err.code(), "E-SOS-002");
        assert!(err.to_string().contains("level=2"), "{err}");
        let cert = prove_nonneg_on(
            f,
            &[g1, g2],
            &[],
            &vars,
            &c.pool,
            &PutinarOpts::with_level(Some(2)),
        )
        .expect("level 2 certifies");
        assert_putinar(&cert);
        assert_eq!(cert.degree, 2);
    }

    #[test]
    fn equality_constraints_get_free_multipliers() {
        // x·y ≤ 1/2 on the circle x² + y² = 1: 1/2 − xy = ½(x − y)² + ½·(1 − x² − y²).
        let c = ctx();
        let f = c.pool.add(vec![
            c.pool.rational(1, 2),
            c.neg(c.pool.mul(vec![c.x, c.y])),
        ]);
        let cert = prove_nonneg_on(
            f,
            &[],
            &[c.disk()],
            &[c.x, c.y],
            &c.pool,
            &PutinarOpts::default(),
        )
        .expect("certificate");
        assert_putinar(&cert);
        assert_eq!(cert.constraints.len(), 2, "h and −h");
        assert_eq!(cert.constraints[0], cert.constraints[1].neg());
        // Only one side of the pair carries a multiplier here; the other
        // hypothesis must be `_`-named or Mathlib's unused-variable linter
        // rejects the file under -DwarningAsError=true.
        let lean = cert.to_lean().expect("lean renders");
        assert!(lean.contains("(_hg"), "{lean}");
    }

    #[test]
    fn false_claims_are_refuted_with_a_point_of_the_set() {
        // 1/2 − x² − y² on the disk fails at (0, 1).
        let c = ctx();
        let f = c.pool.add(vec![
            c.pool.rational(1, 2),
            c.neg(c.pw(c.x, 2)),
            c.neg(c.pw(c.y, 2)),
        ]);
        let err = prove_nonneg_on(
            f,
            &[c.disk()],
            &[],
            &[c.x, c.y],
            &c.pool,
            &PutinarOpts::default(),
        )
        .expect_err("false");
        assert_eq!(err.code(), "E-SOS-003");
        let pt = nonneg_on_witness(f, &[c.disk()], &[], &[c.x, c.y], &c.pool)
            .unwrap()
            .expect("a witness");
        let tp = RatPoly::from_expr(f, &[c.x, c.y], &c.pool).unwrap();
        let g = RatPoly::from_expr(c.disk(), &[c.x, c.y], &c.pool).unwrap();
        let vals: Vec<Rational> = pt
            .iter()
            .map(|&e| {
                RatPoly::from_expr(e, &[c.x, c.y], &c.pool)
                    .unwrap()
                    .as_constant()
                    .unwrap()
            })
            .collect();
        assert!(tp.eval(&vals) < 0);
        assert!(g.eval(&vals) >= 0);
    }

    #[test]
    fn predicates_are_read_as_constraints() {
        let c = ctx();
        let f = c.pool.add(vec![c.int(1), c.neg(c.pw(c.x, 2))]);
        let zero = c.int(0);
        let ge = c.pool.pred_ge(c.disk(), zero);
        let cert = prove_nonneg_on(f, &[ge], &[], &[c.x, c.y], &c.pool, &PutinarOpts::default())
            .expect("certificate");
        assert_putinar(&cert);
        let ne = c.pool.pred_ne(c.x, zero);
        let err = prove_nonneg_on(f, &[ne], &[], &[c.x, c.y], &c.pool, &PutinarOpts::default())
            .expect_err("≠ is not a constraint");
        assert_eq!(err.code(), "E-SOS-004");
    }

    #[test]
    fn split_free_is_exact() {
        let mut l = RatPoly::monomial(2, vec![1, 0], Rational::from(3));
        l = l.add(&RatPoly::monomial(2, vec![2, 1], Rational::from((-5, 7))));
        l = l.add(&RatPoly::monomial(2, vec![2, 2], Rational::from(-2)));
        l = l.add(&RatPoly::constant(2, Rational::from(4)));
        let (p, n) = split_free(&l);
        assert_eq!(p.to_poly(2).sub(&n.to_poly(2)), l);
    }

    #[test]
    fn unbounded_sets_are_reported_as_non_archimedean() {
        // x·y ≥ 0 on {x ≥ 0, y ≥ 0} is true but its only certificate (x·y as
        // a product) is not in the quadratic module; the set is unbounded.
        let c = ctx();
        let f = c.pool.mul(vec![c.x, c.y]);
        let err = prove_nonneg_on(
            f,
            &[c.x, c.y],
            &[],
            &[c.x, c.y],
            &c.pool,
            &PutinarOpts::with_level(Some(1)),
        )
        .expect_err("undecided");
        assert_eq!(err.code(), "E-SOS-002");
        assert!(
            err.to_string().contains("not known to be Archimedean"),
            "{err}"
        );
    }
}
