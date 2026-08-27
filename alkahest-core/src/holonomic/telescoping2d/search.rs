//! The Apagodu–Zeilberger ansatz search: given a proper hypergeometric
//! `F(n, x_1, …, x_m)` with `m ≥ 1` bound indices, find `a_0(n), …, a_J(n)`
//! (not all zero) and `m` rational certificates `c_1, …, c_m ∈ Q(n,x)` such
//! that
//!
//! ```text
//! Σ_i a_i(n)·F(n+i,x) = Σ_t Δ_t G_t,   G_t = c_t·F
//! ```
//!
//! # Method — undetermined coefficients, not a Gosper normal form
//!
//! The single-sum engine ([`super::super::zeilberger`]) puts the shift ratio
//! of `F` into *Gosper normal form* before solving, which is what lets it
//! search a smaller ansatz efficiently. There is no standard multivariate
//! analogue of that normal form for a general proper hypergeometric
//! `F(n,x_1,…,x_m)` — this is exactly why Apagodu–Zeilberger's method (unlike
//! single-index Zeilberger) is usually presented as an undetermined-
//! coefficients search: posit a certificate of bounded degree over a *fixed*
//! denominator, clear it, and solve the resulting linear system.
//!
//! Concretely: divide the target identity by `F(n,x)`. Writing
//! `ρ_t(n,x) = F(…,x_t+1,…)/F(…,x_t,…) = N_t/D_t` (a known rational function,
//! computed exactly by [`super::term::ProperTermM`]) for each bound axis `t`,
//! and taking the certificate ansatz `c_t = P_t(n,x)/E_t(n,x)` with `P_t` a
//! polynomial of bounded *box* degree and `E_t := D_t · (∏_i D_{n,i})` (see
//! below for why the denominator is not just `D_t` alone), the identity
//! becomes, after multiplying through by a common denominator built from the
//! `E_t` and their axis-`t` shifts, a **polynomial** identity in
//! `Q[n,x_1,…,x_m]`, linear in the unknown coefficients of
//! `a_i(n) = Σ_p α_{i,p}·n^p` and of every `P_t`. Matching coefficients of
//! every monomial gives one linear equation per monomial;
//! [`solve_ansatz_md`] assembles that system and takes its nullspace over `Q`
//! by plain Gaussian elimination (see [`rational_nullspace`]).
//!
//! `E_t`'s `D_t` factor (the *raw*, un-reduced denominator of `ρ_t`) is not
//! the minimal possible certificate denominator in general — a genuine
//! multivariate Gosper reduction would sometimes need a smaller one after
//! cancelling a shift-equivalent factor between `N_t` and a shifted `D_t`
//! (exactly what the single-sum engine's `C(k)` factor exists to supply).
//! This module does **not** compute that reduction. For the ordinary
//! "binomial-type" sums this targets, the shift ratios are already close to
//! normal form, so the raw denominator is already sufficient — but this is a
//! property of the *examples*, not a theorem the code establishes. When it is
//! not sufficient, the bounded search below simply finds nothing and reports
//! [`Telescoping2dError::SearchExhausted`]; it never claims a false
//! certificate, because every candidate is re-verified from scratch (see
//! [`verify_certificate_md`]) before it is returned.
//!
//! # From two bound indices to `m`
//!
//! The `m = 2` case (`telescope2d_search`) is now a thin wrapper around the
//! general `telescope_md_search`: it converts `Telescoping2dOpts` to
//! [`TelescopingMdOpts`], calls the general search with `indices = [j, k]`,
//! and repackages the two-certificate result as [`Telescoping2dResult`]. The
//! two-index public API's behavior (including its error variants and search
//! order) is unchanged by this — it is now derived from, rather than
//! duplicating, the general path.

use super::poly::{Axis, PolyM, RatM, AXIS_N};
use super::term::ProperTermM;
use super::Telescoping2dError;
use crate::kernel::{ExprId, ExprPool};
use rug::{Integer, Rational};

/// Upper bound on the total unknown count (`a_i(n)` coefficients plus every
/// certificate numerator's box coefficients, summed) a single
/// `(order, a_degree, cert_degree)` probe is allowed to build a linear system
/// for, checked *before* any polynomial arithmetic for that probe begins.
///
/// This exists because [`rational_nullspace`]'s plain dense Gaussian
/// elimination is `O(rows · cols²)` over exact (unbounded) `Rational`
/// coefficients, and both `rows` (one equation per distinct monomial in
/// `n, x_1, …, x_m` appearing anywhere in the cleared identity) and `cols`
/// (`total`, this bound's subject) grow with `m` and `cert_degree` far
/// faster than the box-degree numbers themselves suggest — see this crate's
/// own measurements: at `m = 3`, `cert_degree = 2` already means `rows ≈
/// 10 000`, `cols = 245`, and a single probe's elimination step alone took
/// **≈ 47 seconds**; `cert_degree = 3` (`cols = 770`) was still running
/// after several minutes when profiled. This is genuine `O(rows·cols²)`
/// arithmetic cost on a real, correctly-posed linear system — not a bug, an
/// infinite loop, or unbounded coefficient blowup — but it is exactly the
/// kind of input-dependent resource cliff a caller must be protected from by
/// a fast, honest refusal rather than an unbounded hang. `400` is
/// calibrated to comfortably admit every worked example this module ships
/// with (the largest, the `m = 3` multinomial-coefficient example, needs
/// `cols = 245`) while excluding the next box-degree step up at `m = 3`
/// (`cols = 770`), which is the one that was actually observed to run
/// unacceptably long. A probe whose unknown count would exceed this is
/// skipped — reported as no candidate at that `(a_degree, cert_degree)`,
/// exactly like a probe the linear algebra genuinely found nothing for — and
/// [`telescope_md_search`]'s final [`Telescoping2dError::SearchExhausted`]
/// message says so explicitly when at least one probe was skipped for this
/// reason, so a caller sees a fast, clearly-explained refusal instead of a
/// silent guess about whether raising the bounds would even help.
const MAX_ANSATZ_UNKNOWNS: usize = 400;

/// A probe's own unknown count must reach this before it counts against
/// [`MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS`] at all. Below this, a probe is
/// "cheap" by construction (this crate's own `m = 2` default search never
/// exceeds ~140 unknowns for any probe it tries — see the constant's own
/// docs) and is exempted from the cumulative accounting entirely, so this
/// budget cannot regress the existing two-index search in any way.
const LARGE_PROBE_THRESHOLD: usize = 150;

/// A single probe under [`MAX_ANSATZ_UNKNOWNS`] can still be individually
/// slow (the `m = 3` multinomial-coefficient worked example's `cols = 245`
/// probe takes ≈ 30 seconds) — tolerable *once*, but
/// [`telescope_md_search`]'s outer loop tries every `(order, a_degree,
/// cert_degree)` combination, and nothing about `total`'s formula depends
/// much on `a_degree` or `order`, so a caller whose input has no certificate
/// at all would otherwise pay that same ≈ 30–50 second cost again for *every*
/// `a_degree` and `order` value tried — six repeats of the exact scenario
/// that motivated [`MAX_ANSATZ_UNKNOWNS`] in the first place, for the
/// triple-binomial-chain example that was this ceiling's original motivating
/// case (see `mod.rs`'s
/// `chained_product_at_original_bounds_refuses_fast_via_resource_ceiling`
/// regression test). This is a running budget across the *whole* search
/// call: every probe with `total >= LARGE_PROBE_THRESHOLD` that is actually
/// attempted adds its `total` to a running sum, and once that sum would
/// exceed this bound, every further large probe is skipped for the rest of
/// the search — capping the number of genuinely expensive elimination
/// attempts to about one or two, regardless of how large the caller's
/// `max_order` / `max_a_degree` / `max_cert_degree` are. `300` admits
/// exactly one probe the size of the multinomial example (`245`) before
/// refusing further ones of that size.
pub(crate) const MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS: usize = 300;

/// The three numbers above, as data rather than as constants read directly
/// by the search loop.
///
/// The search always runs on [`Ceilings::PRODUCTION`]; nothing outside this
/// module's own tests can supply anything else, and the *logic* that consults
/// them — which probe is "large", when the per-probe ceiling refuses, when
/// the cumulative budget refuses — is the same code either way.
///
/// It exists because the ceilings are calibrated in units of *unknowns*, and
/// the cost of the one probe they deliberately let through is quadratic in
/// that number: the regression test in `mod.rs` used to spend ~450 s
/// uninstrumented and 3550 s (~59 min) under AddressSanitizer inside a single
/// 245-unknown exact-rational Gaussian elimination, purely to arrive at
/// counters that the skip logic had already decided. Scaling the ceilings
/// down by the same factor as the probe sizes lets that test assert exactly
/// the same `SearchStats` shape — one large probe attempted, per-probe
/// ceiling fired, cumulative ceiling fired — on the identical input and the
/// identical code path, for a 25th of the cost: 143 s under ASan and 20 s
/// uninstrumented, measured, against 3550 s and ≈450 s. (Not free, because
/// every probe assembles a system of the same ~10 000 rows whatever its
/// column count; what scales away is the `cols²` in `rows · cols²`.) What
/// the scaled run does not re-measure is how long 245 unknowns takes, which
/// was never something the assertions depended on;
/// `production_ceilings_classify_the_chained_product_probe_ladder` pins the
/// shipped numbers against that ladder separately, by arithmetic.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ceilings {
    /// See [`MAX_ANSATZ_UNKNOWNS`].
    pub max_ansatz_unknowns: usize,
    /// See [`LARGE_PROBE_THRESHOLD`].
    pub large_probe_threshold: usize,
    /// See [`MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS`].
    pub max_cumulative_large_probe_unknowns: usize,
}

impl Ceilings {
    /// The shipped calibration. Every non-test entry point uses this.
    pub(crate) const PRODUCTION: Ceilings = Ceilings {
        max_ansatz_unknowns: MAX_ANSATZ_UNKNOWNS,
        large_probe_threshold: LARGE_PROBE_THRESHOLD,
        max_cumulative_large_probe_unknowns: MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS,
    };
}

/// What one [`telescope_md_search`] call actually *attempted*, and which of
/// the two resource ceilings above stopped it attempting more.
///
/// This is the deterministic statement of the property
/// [`MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS`] exists to guarantee — "roughly one
/// genuinely expensive elimination per search call, not one per `(order,
/// a_degree)` combination". `mod.rs`'s
/// `chained_product_at_original_bounds_refuses_fast_via_resource_ceiling`
/// asserts on these counts. It used to assert on elapsed wall-clock instead,
/// which measures the same property only through a proxy that varies ~15x
/// across platforms (76 s on Linux vs 480 s on Windows for the identical
/// elimination), ~30x under AddressSanitizer, and more than 2x with nothing
/// but machine load — and so had to be relaxed or carved out three times
/// before this. Probes attempted is none of those things: it is fixed by the
/// input and the ceilings alone.
///
/// Keeping it costs a few `usize` increments per probe, against a probe body
/// that assembles and Gaussian-eliminates a dense exact-rational system of
/// thousands of rows, so it is not gated; only the entry point that hands it
/// back to a caller is `#[cfg(test)]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SearchStats {
    /// Probes whose linear system was actually assembled and solved.
    #[cfg_attr(not(test), allow(dead_code))]
    pub attempted: usize,
    /// Of [`Self::attempted`], those at or above [`LARGE_PROBE_THRESHOLD`]
    /// unknowns — this module's unit of "genuinely expensive elimination".
    #[cfg_attr(not(test), allow(dead_code))]
    pub large_attempted: usize,
    /// Sum of the unknown counts of the [`Self::large_attempted`] probes:
    /// exactly the running quantity
    /// [`MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS`] bounds.
    pub large_unknowns_spent: usize,
    /// Probes skipped by the per-probe ceiling [`MAX_ANSATZ_UNKNOWNS`]
    /// (including the `checked_pow` overflow guard, which is that same
    /// ceiling reached the only other way it can be).
    pub skipped_per_probe_ceiling: usize,
    /// Probes skipped because attempting them would have pushed
    /// [`Self::large_unknowns_spent`] past
    /// [`MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS`].
    pub skipped_cumulative_ceiling: usize,
}

impl SearchStats {
    /// Whether either ceiling refused at least one probe — the condition the
    /// `SearchExhausted` message's ceiling note is attached under.
    fn any_ceiling_fired(&self) -> bool {
        self.skipped_per_probe_ceiling > 0 || self.skipped_cumulative_ceiling > 0
    }
}

use std::collections::BTreeMap;

/// Search bounds for [`telescope2d`](super::telescope2d) and
/// [`telescope_md`](super::telescope_md). All three are genuine upper
/// bounds — the search tries every combination
/// `1..=max_order × 0..=max_a_degree × 0..=max_cert_degree` in ascending
/// order (cheapest first in each axis), so raising them only admits harder
/// inputs.
///
/// Unlike [`super::super::zeilberger::ZeilbergerOpts`] this is **not**
/// cost-ordered across the three axes jointly (see the module docs): the
/// three loops are simply nested, ascending. That is a real scope
/// simplification, not an oversight — the point of the cost-ordered plan in
/// the single-sum engine is to make expensive high-degree probes at low
/// order not block a cheap high-order solution; the ansatz here does not yet
/// have that tuning.
#[derive(Debug, Clone, Copy)]
pub struct Telescoping2dOpts {
    /// Largest recurrence order `J`; orders are tried from 1 upward.
    pub max_order: usize,
    /// Largest degree (in `n`) tried for each `a_i(n)`.
    pub max_a_degree: usize,
    /// Largest *box* degree (in each of `n`, `j`, `k` independently) tried
    /// for the certificate numerators `P_1(n,j,k)`, `P_2(n,j,k)`.
    pub max_cert_degree: usize,
}

impl Default for Telescoping2dOpts {
    fn default() -> Self {
        Telescoping2dOpts {
            max_order: 2,
            max_a_degree: 2,
            max_cert_degree: 3,
        }
    }
}

/// Search bounds for [`telescope_md_search`], the `m`-bound-index
/// generalization of [`Telescoping2dOpts`]. Same fields, same search
/// discipline (see that struct's docs); `max_cert_degree` bounds the box
/// degree of each `P_t` in **every** one of the `m + 1` variables
/// independently, so the per-certificate unknown count is
/// `(max_cert_degree + 1)^(m + 1)` — raising `max_cert_degree` gets
/// expensive fast as `m` grows, more so than in the two-index case.
#[derive(Debug, Clone, Copy)]
pub struct TelescopingMdOpts {
    pub max_order: usize,
    pub max_a_degree: usize,
    pub max_cert_degree: usize,
}

impl Default for TelescopingMdOpts {
    fn default() -> Self {
        TelescopingMdOpts {
            max_order: 2,
            max_a_degree: 2,
            max_cert_degree: 2,
        }
    }
}

impl From<Telescoping2dOpts> for TelescopingMdOpts {
    fn from(o: Telescoping2dOpts) -> Self {
        TelescopingMdOpts {
            max_order: o.max_order,
            max_a_degree: o.max_a_degree,
            max_cert_degree: o.max_cert_degree,
        }
    }
}

/// A verified double-sum creative-telescoping certificate (`m = 2`).
///
/// The verified content is the identity
/// `Σ_i a_i(n)·F(n+i,j,k) = Δ_j G_1 + Δ_k G_2` with `G_1 = cert1·F`,
/// `G_2 = cert2·F` — checked exactly in `Q(n,j,k)` by `verify_certificate_md`
/// before this is ever constructed. Turning it into a recurrence for
/// `S(n) = Σ_j Σ_k F(n,j,k)` over a stated range is a separate step; see
/// [`super::boundary`].
#[derive(Debug, Clone)]
pub struct Telescoping2dResult {
    pub order: usize,
    /// `a_0(n), …, a_J(n)`, integer-content-primitive as a family (the same
    /// discipline as the single-sum engine's `clear_denominators`, applied to
    /// the flat vector of every unknown in this certificate — see
    /// `primitive_scale_rationals`).
    pub coeffs: Vec<ExprId>,
    /// `c_1(n,j,k)`, with `G_1 = c_1·F`.
    pub cert1: ExprId,
    /// `c_2(n,j,k)`, with `G_2 = c_2·F`.
    pub cert2: ExprId,
}

/// A verified `m`-bound-index creative-telescoping certificate, the general
/// form of [`Telescoping2dResult`] (which is `certs.len() == 2` repackaged).
///
/// The verified content is `Σ_i a_i(n)·F(n+i,x) = Σ_t Δ_t(c_t·F)` — checked
/// exactly in `Q(n,x_1,…,x_m)` before this is ever constructed. See
/// [`super::boundary::boundary_status_md`] for turning it into a recurrence
/// for the `m`-fold sum over a stated box.
#[derive(Debug, Clone)]
pub struct TelescopingMdResult {
    pub order: usize,
    /// `a_0(n), …, a_J(n)`.
    pub coeffs: Vec<ExprId>,
    /// `c_1(n,x), …, c_m(n,x)`, one per bound index, in the order the caller
    /// supplied `indices`. `G_t = certs[t-1]·F`.
    pub certs: Vec<ExprId>,
}

/// Internal (pre-`ExprId`) form of a candidate, kept in algebraic form so
/// [`verify_certificate_md`] can re-check it without any expression-pool
/// round-trip.
struct CandidateMd {
    order: usize,
    a: Vec<PolyM>,    // a_i(n), i = 0..=order, degree only in the N axis
    certs: Vec<RatM>, // one per bound index
}

/// Apagodu–Zeilberger search for the `m = 2` case: find and verify a
/// double-sum certificate for `term`, a proper hypergeometric `F(n,j,k)`.
///
/// A thin wrapper around [`telescope_md_search`] with `indices = [j, k]` —
/// see the module docs.
pub fn telescope2d_search(
    term: ExprId,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    pool: &ExprPool,
    opts: &Telescoping2dOpts,
) -> Result<Telescoping2dResult, Telescoping2dError> {
    let md_opts: TelescopingMdOpts = (*opts).into();
    let r = telescope_md_search(term, n, &[j, k], pool, &md_opts)?;
    debug_assert_eq!(r.certs.len(), 2);
    Ok(Telescoping2dResult {
        order: r.order,
        coeffs: r.coeffs,
        cert1: r.certs[0],
        cert2: r.certs[1],
    })
}

/// Apagodu–Zeilberger search for general `m ≥ 1`: find and verify a
/// creative-telescoping certificate for `term`, a proper hypergeometric
/// `F(n, x_1, …, x_m)` with `indices = [x_1, …, x_m]`.
///
/// See the module docs (`telescoping2d::search`) for the method, and the honest limitations
/// list in `mod.rs`: this covers the same proper-hypergeometric-only,
/// fixed-denominator-ansatz, box-degree-search scope the `m = 2` engine has,
/// generalized to arbitrary `m` — it is **not** a broader class of summand.
pub fn telescope_md_search(
    term: ExprId,
    n: ExprId,
    indices: &[ExprId],
    pool: &ExprPool,
    opts: &TelescopingMdOpts,
) -> Result<TelescopingMdResult, Telescoping2dError> {
    telescope_md_search_impl(
        term,
        n,
        indices,
        pool,
        opts,
        &Ceilings::PRODUCTION,
        &mut SearchStats::default(),
    )
}

/// [`telescope_md_search`], additionally reporting the [`SearchStats`] for
/// the call.
///
/// Crate-internal and test-only: this module is private and only the plain
/// entry point is re-exported, so this adds nothing to the public API. Its
/// one caller is the resource-ceiling regression test in `mod.rs`, which
/// needs to assert on *work attempted* rather than on elapsed time.
#[cfg(test)]
pub(crate) fn telescope_md_search_instrumented(
    term: ExprId,
    n: ExprId,
    indices: &[ExprId],
    pool: &ExprPool,
    opts: &TelescopingMdOpts,
    ceilings: &Ceilings,
) -> (Result<TelescopingMdResult, Telescoping2dError>, SearchStats) {
    let mut stats = SearchStats::default();
    let result = telescope_md_search_impl(term, n, indices, pool, opts, ceilings, &mut stats);
    (result, stats)
}

fn telescope_md_search_impl(
    term: ExprId,
    n: ExprId,
    indices: &[ExprId],
    pool: &ExprPool,
    opts: &TelescopingMdOpts,
    ceilings: &Ceilings,
    stats: &mut SearchStats,
) -> Result<TelescopingMdResult, Telescoping2dError> {
    let m = indices.len();
    if m == 0 {
        return Err(Telescoping2dError::InvalidInput(
            "at least one bound index is required".into(),
        ));
    }
    if indices.contains(&n) || has_duplicate(indices) {
        return Err(Telescoping2dError::InvalidInput(
            "n and every bound index must be pairwise distinct symbols".into(),
        ));
    }
    if opts.max_order == 0 {
        return Err(Telescoping2dError::InvalidInput(
            "max_order must be at least 1".into(),
        ));
    }
    let num_axes = m + 1;

    let f = ProperTermM::parse(term, n, indices, pool)?;
    let mut rhos: Vec<RatM> = Vec::with_capacity(m);
    for t in 0..m {
        let r = f.ratio_axis(t + 1, 1)?;
        if r.den.is_zero() {
            return Err(Telescoping2dError::NotProperHypergeometric(
                "shift ratio has a zero denominator".into(),
            ));
        }
        rhos.push(r);
    }

    for order in 1..=opts.max_order {
        let mut nn = Vec::with_capacity(order + 1);
        let mut dn = Vec::with_capacity(order + 1);
        for i in 0..=order as i64 {
            let r = f.ratio_axis(AXIS_N, i)?;
            if r.den.is_zero() {
                return Err(Telescoping2dError::NotProperHypergeometric(
                    "shift ratio in n has a zero denominator".into(),
                ));
            }
            nn.push(r.num);
            dn.push(r.den);
        }

        for a_degree in 0..=opts.max_a_degree {
            for cert_degree in 0..=opts.max_cert_degree {
                // Cheap, purely arithmetic pre-check — see MAX_ANSATZ_UNKNOWNS'
                // docs — before any polynomial construction or linear-system
                // assembly for this probe begins.
                let box_len = cert_degree + 1;
                let Some(cert_box_count) = box_len.checked_pow(num_axes as u32) else {
                    stats.skipped_per_probe_ceiling += 1;
                    continue;
                };
                let a_count = (order + 1) * (a_degree + 1);
                let total = a_count.saturating_add(m.saturating_mul(cert_box_count));
                if total > ceilings.max_ansatz_unknowns {
                    stats.skipped_per_probe_ceiling += 1;
                    continue;
                }
                // See MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS' docs: a probe
                // large enough to be individually slow also counts against a
                // running total-search budget, so a caller cannot pay that
                // same cost over and over across every (order, a_degree)
                // combination when no certificate exists at all.
                if total >= ceilings.large_probe_threshold {
                    if stats.large_unknowns_spent.saturating_add(total)
                        > ceilings.max_cumulative_large_probe_unknowns
                    {
                        stats.skipped_cumulative_ceiling += 1;
                        continue;
                    }
                    stats.large_unknowns_spent += total;
                    stats.large_attempted += 1;
                }

                stats.attempted += 1;
                if let Some(cand) =
                    solve_ansatz_md(order, m, a_degree, cert_degree, &nn, &dn, &rhos, num_axes)?
                {
                    if verify_certificate_md(&f, &cand) {
                        return Ok(finish_md(cand, n, indices, pool));
                    }
                    // A genuine implementation bug, not a user-facing error:
                    // never happens for a correct construction, but refusing
                    // silently and continuing keeps the "never return an
                    // unverified certificate" discipline absolute.
                }
            }
        }
    }

    let budget_note = if stats.any_ceiling_fired() {
        format!(
            " (at least one (order, a_degree, cert_degree) combination within these bounds was \
             skipped without being attempted, refused by this module's resource ceilings \
             (MAX_ANSATZ_UNKNOWNS = {} unknowns for any single probe; \
             MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS = {} total \
             across every probe at or above {} unknowns in one search call) \
             rather than attempted and left to run arbitrarily long — raising max_cert_degree or \
             the bound-index count further will only make this worse, not better)",
            ceilings.max_ansatz_unknowns,
            ceilings.max_cumulative_large_probe_unknowns,
            ceilings.large_probe_threshold,
        )
    } else {
        String::new()
    };
    Err(Telescoping2dError::SearchExhausted(format!(
        "no verified {}-index certificate of order <= {} was found for {} \
         within a_degree <= {} and certificate degree <= {}{budget_note}",
        m,
        opts.max_order,
        pool.display(term),
        opts.max_a_degree,
        opts.max_cert_degree
    )))
}

fn has_duplicate(xs: &[ExprId]) -> bool {
    for i in 0..xs.len() {
        for j in (i + 1)..xs.len() {
            if xs[i] == xs[j] {
                return true;
            }
        }
    }
    false
}

fn flatten(exps: &[usize], box_len: usize) -> usize {
    exps.iter().fold(0, |acc, &x| acc * box_len + x)
}

fn unflatten(mut combo: usize, num_axes: usize, box_len: usize) -> Vec<usize> {
    let mut exps = vec![0usize; num_axes];
    for idx in (0..num_axes).rev() {
        exps[idx] = combo % box_len;
        combo /= box_len;
    }
    exps
}

fn monomial_m(exps: &[usize], c: Rational, num_axes: usize) -> PolyM {
    let mut p = PolyM::constant(c, num_axes);
    for (axis, &e) in exps.iter().enumerate() {
        if e > 0 {
            p = p.mul(&PolyM::var(axis, num_axes).pow_u32(e as u32));
        }
    }
    p
}

/// Build and solve the linear system for one `(order, a_degree, cert_degree)`
/// probe. `Ok(None)` means the system has no solution with a non-trivial
/// leading coefficient — not an error, just this probe failing.
#[allow(clippy::too_many_arguments)]
fn solve_ansatz_md(
    order: usize,
    m: usize,
    a_degree: usize,
    cert_degree: usize,
    nn: &[PolyM],
    dn: &[PolyM],
    rhos: &[RatM],
    num_axes: usize,
) -> Result<Option<CandidateMd>, Telescoping2dError> {
    let ds: Vec<&PolyM> = rhos.iter().map(|r| &r.den).collect();
    let ns_: Vec<&PolyM> = rhos.iter().map(|r| &r.num).collect();

    let mut dnfull = PolyM::one(num_axes);
    for d in dn {
        dnfull = dnfull.mul(d);
    }
    let mut dnfull_excl: Vec<PolyM> = Vec::with_capacity(order + 1);
    for i in 0..=order {
        let mut p = PolyM::one(num_axes);
        for (idx, d) in dn.iter().enumerate() {
            if idx != i {
                p = p.mul(d);
            }
        }
        dnfull_excl.push(p);
    }

    // e[t] = D_t * dnfull ; es[t] = shift(e[t], axis = t+1, 1). This is
    // exactly the m=2 module's `e1`/`e2`/`e1s`/`e2s`, generalized.
    let mut e: Vec<PolyM> = Vec::with_capacity(m);
    let mut es: Vec<PolyM> = Vec::with_capacity(m);
    for (t, d) in ds.iter().enumerate().take(m) {
        let et = d.mul(&dnfull);
        let ets = et.shift(t + 1, 1);
        e.push(et);
        es.push(ets);
    }
    // block[t] = e[t] * es[t] * D_t — the m=2 module's `e1*e1s*dj`/`e2*e2s*dk`.
    let block: Vec<PolyM> = (0..m).map(|t| e[t].mul(&es[t]).mul(ds[t])).collect();

    // mn[i]: the multiplier for a_i(n)'s column, built as `D_total / D_{n,i}`
    // by replacing exactly one of the two `dnfull` copies inside `block[0]`
    // with `dnfull_excl[i]` (any one of `D_total`'s `2m` redundant copies of
    // `D_{n,i}` may be the one "removed" — always using axis 0's block for
    // this, mirroring the m=2 module's asymmetric-but-valid choice of always
    // using `e1`/`e1s`/`dj`).
    let mut mn: Vec<PolyM> = Vec::with_capacity(order + 1);
    for excl in &dnfull_excl {
        let mut p = ds[0].mul(excl).mul(&es[0]).mul(ds[0]);
        for blk in block.iter().skip(1) {
            p = p.mul(blk);
        }
        mn.push(p);
    }
    // mt[t]: the multiplier for P_t's columns, `D_total / block[t]` — the
    // product of every *other* axis's block.
    let mut mt: Vec<PolyM> = Vec::with_capacity(m);
    for t in 0..m {
        let mut p = PolyM::one(num_axes);
        for (tp, blk) in block.iter().enumerate() {
            if tp != t {
                p = p.mul(blk);
            }
        }
        mt.push(p);
    }

    let na = a_degree + 1;
    let box_len = cert_degree + 1;
    let cert_box_count = box_len.pow(num_axes as u32);
    let a_count = (order + 1) * na;
    let total = a_count + m * cert_box_count;

    let idx_a = |i: usize, p: usize| i * na + p;
    let idx_cert = |t: usize, exps: &[usize]| a_count + t * cert_box_count + flatten(exps, box_len);

    let mut rows: BTreeMap<Vec<u32>, Vec<Rational>> = BTreeMap::new();
    let mut add_contribution = |mono: &BTreeMap<Vec<u32>, Rational>, col: usize| {
        for (exp, c) in mono {
            let row = rows
                .entry(exp.clone())
                .or_insert_with(|| vec![Rational::from(0); total]);
            row[col] += c.clone();
        }
    };

    for i in 0..=order {
        for p in 0..=a_degree {
            let basis = PolyM::var(AXIS_N, num_axes)
                .pow_u32(p as u32)
                .mul(&nn[i])
                .mul(&mn[i]);
            add_contribution(&basis.terms, idx_a(i, p));
        }
    }

    // Memoized plain (`x_t^q`) and shifted (`(x_t+1)^q`) powers, per bound
    // axis, for q = 0..=cert_degree; and plain `n^p` powers.
    let mut plain_pow: Vec<Vec<PolyM>> = Vec::with_capacity(m);
    let mut shift_pow: Vec<Vec<PolyM>> = Vec::with_capacity(m);
    for t in 0..m {
        let axis = t + 1;
        let xv = PolyM::var(axis, num_axes);
        let xv1 = xv.add(&PolyM::one(num_axes));
        let mut pp = Vec::with_capacity(box_len);
        let mut sp = Vec::with_capacity(box_len);
        let mut accp = PolyM::one(num_axes);
        let mut accs = PolyM::one(num_axes);
        for _ in 0..box_len {
            pp.push(accp.clone());
            sp.push(accs.clone());
            accp = accp.mul(&xv);
            accs = accs.mul(&xv1);
        }
        plain_pow.push(pp);
        shift_pow.push(sp);
    }
    let n_pow: Vec<PolyM> = {
        let nv = PolyM::var(AXIS_N, num_axes);
        let mut acc = PolyM::one(num_axes);
        let mut v = Vec::with_capacity(box_len);
        for _ in 0..box_len {
            v.push(acc.clone());
            acc = acc.mul(&nv);
        }
        v
    };

    let total_combos = box_len.pow(num_axes as u32);
    for combo in 0..total_combos {
        let exps = unflatten(combo, num_axes, box_len);
        let p = exps[0];
        let np = n_pow[p].clone();
        for t in 0..m {
            let mut mono_shift = np.clone();
            let mut mono_plain = np.clone();
            for tp in 0..m {
                let q = exps[tp + 1];
                mono_plain = mono_plain.mul(&plain_pow[tp][q]);
                mono_shift = mono_shift.mul(if tp == t {
                    &shift_pow[tp][q]
                } else {
                    &plain_pow[tp][q]
                });
            }
            // [c_t(x_t+1)*rho_t - c_t] * (D_total/block[t]) — see the module
            // docs derivation, generalizing the m=2 module's `basis1`/`basis2`.
            let basis = mono_shift
                .mul(ns_[t])
                .mul(&e[t])
                .sub(&mono_plain.mul(&es[t]).mul(ds[t]))
                .mul(&mt[t])
                .neg();
            add_contribution(&basis.terms, idx_cert(t, &exps));
        }
    }

    let matrix: Vec<Vec<Rational>> = rows.into_values().collect();
    let basis_vecs = rational_nullspace(matrix, total);
    if basis_vecs.is_empty() {
        return Ok(None);
    }

    for vec in &basis_vecs {
        let top_nonzero = (0..na).any(|p| vec[idx_a(order, p)] != 0);
        if !top_nonzero {
            continue;
        }
        let mut scaled = vec.clone();
        primitive_scale_rationals(&mut scaled);

        let mut a = Vec::with_capacity(order + 1);
        for i in 0..=order {
            let mut poly = PolyM::zero();
            for pw in 0..=a_degree {
                let c = scaled[idx_a(i, pw)].clone();
                if c != 0 {
                    poly = poly.add(&PolyM::var(AXIS_N, num_axes).pow_u32(pw as u32).scale(&c));
                }
            }
            a.push(poly);
        }

        let mut certs_num: Vec<PolyM> = vec![PolyM::zero(); m];
        for combo in 0..total_combos {
            let exps = unflatten(combo, num_axes, box_len);
            for (t, cn) in certs_num.iter_mut().enumerate() {
                let c = scaled[idx_cert(t, &exps)].clone();
                if c != 0 {
                    *cn = cn.add(&monomial_m(&exps, c, num_axes));
                }
            }
        }
        let certs: Vec<RatM> = (0..m)
            .map(|t| RatM {
                num: certs_num[t].clone(),
                den: e[t].clone(),
            })
            .collect();
        return Ok(Some(CandidateMd { order, a, certs }));
    }
    Ok(None)
}

/// Scale a flat family of rationals to integer, content-1 (up to sign) —
/// the same principle as [`super::super::qfield::make_primitive`], applied
/// to one combined vector spanning `a_i(n)` *and* every certificate
/// numerator, because a homogeneous linear relation stays a solution under
/// any single overall rescaling. Kept local rather than reusing
/// `make_primitive` because that helper is specialized to a family of
/// single-variable `RatUniPoly`s; here the family spans `1 + m` different
/// polynomial rings (one univariate, `m` multivariate) sharing only their
/// scalar coefficients.
fn primitive_scale_rationals(v: &mut [Rational]) {
    let mut den_lcm = Integer::from(1);
    for c in v.iter() {
        den_lcm = den_lcm.lcm(c.denom());
    }
    for c in v.iter_mut() {
        *c *= den_lcm.clone();
    }
    let mut content = Integer::from(0);
    for c in v.iter() {
        content = content.gcd(c.numer());
    }
    if content != 0 && content != 1 {
        let inv = Rational::from((1, content.clone()));
        for c in v.iter_mut() {
            *c *= inv.clone();
        }
    }
    let sign_ref = v.iter().rev().find(|c| **c != 0).cloned();
    if let Some(s) = sign_ref {
        if s < 0 {
            for c in v.iter_mut() {
                *c *= Rational::from(-1);
            }
        }
    }
}

/// Nullspace basis of `mat` (rows = equations, `ncols` unknowns) over `Q`, by
/// plain Gaussian elimination to row-echelon form.
#[allow(clippy::needless_range_loop)]
fn rational_nullspace(mut mat: Vec<Vec<Rational>>, ncols: usize) -> Vec<Vec<Rational>> {
    let nrows = mat.len();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows {
            break;
        }
        let Some(pr) = (row..nrows).find(|&r| mat[r][col] != 0) else {
            continue;
        };
        mat.swap(row, pr);
        let inv = Rational::from(1) / mat[row][col].clone();
        for c in col..ncols {
            mat[row][c] *= inv.clone();
        }
        for r in 0..nrows {
            if r == row {
                continue;
            }
            let factor = mat[r][col].clone();
            if factor == 0 {
                continue;
            }
            for c in col..ncols {
                let sub = factor.clone() * mat[row][c].clone();
                mat[r][c] -= sub;
            }
        }
        pivot_cols.push(col);
        row += 1;
    }
    let is_pivot: Vec<bool> = {
        let mut v = vec![false; ncols];
        for &c in &pivot_cols {
            v[c] = true;
        }
        v
    };
    let mut basis = Vec::new();
    for free_col in 0..ncols {
        if is_pivot[free_col] {
            continue;
        }
        let mut v = vec![Rational::from(0); ncols];
        v[free_col] = Rational::from(1);
        for (r, &pc) in pivot_cols.iter().enumerate() {
            v[pc] = -mat[r][free_col].clone();
        }
        basis.push(v);
    }
    basis
}

/// Re-derive and check the telescoping identity **from scratch** — never
/// trusting the linear-algebra search that produced `cand`. This mirrors
/// `zeilberger()`'s own final check and `PositivityCertificate::verify`'s
/// discipline: a candidate only becomes a result after this passes.
fn verify_certificate_md(f: &ProperTermM, cand: &CandidateMd) -> bool {
    let num_axes = f.m + 1;
    let mut lhs = RatM::from_rational(Rational::from(0), num_axes);
    for (i, ai) in cand.a.iter().enumerate() {
        let Ok(ratio) = f.ratio_axis(AXIS_N, i as i64) else {
            return false;
        };
        lhs = lhs.add(&RatM::from_poly(ai.clone(), num_axes).mul(&ratio));
    }
    let mut rhs = RatM::from_rational(Rational::from(0), num_axes);
    for (t, cert_t) in cand.certs.iter().enumerate() {
        let axis: Axis = t + 1;
        let Ok(rho_t) = f.ratio_axis(axis, 1) else {
            return false;
        };
        let g_delta = cert_t.shift(axis, 1).mul(&rho_t).sub(cert_t);
        rhs = rhs.add(&g_delta);
    }
    lhs.eq_rat(&rhs)
}

fn finish_md(
    cand: CandidateMd,
    n: ExprId,
    indices: &[ExprId],
    pool: &ExprPool,
) -> TelescopingMdResult {
    let coeffs = cand.a.iter().map(|p| p.to_expr(pool, n, indices)).collect();
    let certs = cand
        .certs
        .iter()
        .map(|c| c.to_expr(pool, n, indices))
        .collect();
    TelescopingMdResult {
        order: cand.order,
        coeffs,
        certs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn njk(pool: &ExprPool) -> (ExprId, ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("j", Domain::Real),
            pool.symbol("k", Domain::Real),
        )
    }

    fn binom(pool: &ExprPool, top: ExprId, bot: ExprId) -> ExprId {
        pool.func("binomial", vec![top, bot])
    }

    /// `F(n,j,k) = C(n,j)*C(j,k)`, `Σ_j Σ_k F = Σ_j C(n,j)*2^j`. Not itself
    /// the closed-form check (that lives in `mod.rs`'s worked-example test);
    /// this is the narrower claim that the ansatz search finds *some*
    /// verified telescoping certificate for a genuinely non-separable proper
    /// hypergeometric term in three indices.
    #[test]
    fn finds_a_certificate_for_double_binomial_product() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let f = pool.mul(vec![binom(&pool, n, j), binom(&pool, j, k)]);
        let opts = Telescoping2dOpts::default();
        let result = telescope2d_search(f, n, j, k, &pool, &opts)
            .expect("a double-sum certificate should be found for C(n,j)*C(j,k)");
        assert!(result.order >= 1);
        assert_eq!(result.coeffs.len(), result.order + 1);
    }

    #[test]
    fn refuses_coincident_indices() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let opts = Telescoping2dOpts::default();
        let err =
            telescope2d_search(n, n, n, n, &pool, &opts).expect_err("n, j, k must be distinct");
        assert!(matches!(err, Telescoping2dError::InvalidInput(_)));
    }

    #[test]
    fn md_refuses_zero_indices() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let opts = TelescopingMdOpts::default();
        let err = telescope_md_search(n, n, &[], &pool, &opts)
            .expect_err("at least one bound index required");
        assert!(matches!(err, Telescoping2dError::InvalidInput(_)));
    }

    /// `m = 1` should degenerate cleanly to the same shape as the classical
    /// single-sum engine's easiest examples: `F(n,k) = C(n,k)`, order-1
    /// certificate, `Σ_k C(n,k) = 2^n`.
    #[test]
    fn md_single_index_matches_classical_binomial_sum() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let f = binom(&pool, n, k);
        let opts = TelescopingMdOpts::default();
        let result = telescope_md_search(f, n, &[k], &pool, &opts)
            .expect("a certificate should be found for C(n,k)");
        assert_eq!(result.certs.len(), 1);
        assert!(result.order >= 1);
    }
}
