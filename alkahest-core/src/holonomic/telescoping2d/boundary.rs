//! Deciding the *`m`-dimensional* boundary hypothesis a creative-telescoping
//! certificate rests on, for `m ≥ 1` bound indices.
//!
//! [`super::search::telescope_md_search`] proves an identity about the
//! **summand**:
//!
//! ```text
//! Σ_i a_i(n)·F(n+i,x) = Σ_t Δ_t G_t,   G_t = c_t·F
//! ```
//!
//! Turning that into a statement about `S(n) = Σ_{x_1} … Σ_{x_m} F(n,x)` over
//! a rectangular *box* `x_t = lo_t..hi_t` is a second, separate step —
//! exactly as in the single-sum engine ([`super::super::boundary`]) — and it
//! is genuinely *more* subtle here, not just a repeat of the 1-D argument in
//! `m` coordinates.
//!
//! # The boundary is `2m` face sums, not `2^m` corner evaluations
//!
//! Summing the identity over the box and telescoping each difference operator
//! over its own index (Fubini — the order of summation does not matter for a
//! finite multiple sum) gives, for each axis `t`, a term
//!
//! ```text
//! Σ_{x_1} … (no x_t) … Σ_{x_m} [G_t(…, x_t = hi_t+1, …) − G_t(…, x_t = lo_t, …)]
//! ```
//!
//! summed over the remaining `m - 1` free bound indices. The right-hand side
//! is **`2m` sums, each over the `(m-1)`-dimensional face where one axis has
//! been fixed to a boundary value** — not `2^m` point evaluations at the
//! box's corners. The `m = 2` case ([`super::super::telescoping2d::boundary`]
//! before this generalization) already spelled out why the naive corner-
//! evaluation generalization is the wrong formula; the trap recurs, unchanged
//! in kind, at every `m`: a face is a sum over an `(m-1)`-dimensional slab,
//! and collapsing it to `2^m` corner evaluations both fails to compute what
//! is above and (for `m ≥ 2`) miscounts every face that is not literally a
//! single point.
//!
//! # What this module actually proves, and what it refuses to
//!
//! Summing a face in closed form is, in general, itself a creative-
//! telescoping problem — potentially as hard as the original one. This
//! module does not attempt that. Instead it establishes the **sufficient**
//! (not necessary) condition that each of the `2m` faces is *identically the
//! zero function* of its remaining free variables, in which case the sum over
//! it is trivially `0` term by term. Concretely, for the face
//! `G_t(…, x_t = v, …)` (as a function of `n` and the `m - 1` other bound
//! indices): if `F`'s parsed `Γ(a·n + Σ_s b_s·x_s + d)^e` factor list contains
//! an `e < 0` factor whose argument, after substituting `x_t ↦ v`, no longer
//! depends on `n` or on any *other* bound index (i.e. every coefficient
//! except `x_t`'s own vanishes) and evaluates to a non-positive integer, then
//! `1/Γ(·)^{|e|}` is exactly `0` there — a genuine identity, not a numeric
//! coincidence — and as long as no *other* factor in `F`, its rational
//! prefactor, or the certificate's own denominator has a matching pole there,
//! the whole face is `0`.
//!
//! This is the same *tool* the single-sum engine's order counting uses (a
//! `1/Γ` factor at a non-positive integer argument); the `m`-dimensional
//! generalization is mechanical once one axis is fixed to a constant — the
//! same algebraic check as the `m = 2` case, just against "every *other*
//! bound index" instead of "the one other bound index". It is **strictly
//! weaker** than the single-sum module's analysis: it never resolves a
//! vanishing that needs cancellation between multiple factors, and it proves
//! nothing about a face that is merely non-constant-but-summing-to-zero. Both
//! of those report [`BoundaryStatusMd::Unknown`] rather than a guess.
//!
//! # Scope: only ranges that do not depend on `n`
//!
//! Exactly as in the `m = 2` case: **every `lo_t, hi_t` must be an integer
//! constant**, not an expression in `n`. A caller with an `n`-dependent
//! natural range (e.g. `x_1 = 0..n`) can still use this module whenever the
//! summand vanishes combinatorially outside the true range — pick a fixed
//! bound safely larger than any `n` of interest and let `F`'s own vanishing
//! do the rest. Passing an `n`-dependent limit expression is refused as
//! [`BoundaryStatusMd::Unknown`], not silently misinterpreted.

use super::poly::{Axis, PolyM};
use super::search::{Telescoping2dResult, TelescopingMdResult};
use super::term::{affine_partsm, as_ratm, GammaFactorM, ProperTermM};
use crate::kernel::{ExprId, ExprPool};
use rug::Rational;

/// The verdict on the boundary of a rectangular double sum (`m = 2`). See the
/// [module docs](self) for exactly what each variant is allowed to mean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundaryStatus2d {
    /// Every one of the four boundary strips was proved identically zero, so
    /// the homogeneous recurrence `Σ_i a_i(n)·S(n+i) = 0` holds for the sum.
    Vanishes,
    /// Reserved for a future extension that computes an explicit non-zero
    /// boundary term `b(n)` by summing a boundary strip in closed form. The
    /// current analysis never constructs one — a strip that does not meet
    /// the [`Vanishes`](BoundaryStatus2d::Vanishes) criterion is reported
    /// [`Unknown`](BoundaryStatus2d::Unknown), not resolved as nonzero — so
    /// this variant is not produced by this version. It is kept in the
    /// public enum so the three-valued discipline the rest of this codebase
    /// uses (see [`super::super::boundary::BoundaryStatus`]) is visible in
    /// the type even where the implementation is not yet complete.
    Nonzero { rhs: ExprId, witness_n: i64 },
    /// Neither was established. **No** recurrence for the sum follows.
    Unknown { reason: String },
}

impl BoundaryStatus2d {
    pub fn tag(&self) -> &'static str {
        match self {
            BoundaryStatus2d::Vanishes => "vanishes",
            BoundaryStatus2d::Nonzero { .. } => "nonzero",
            BoundaryStatus2d::Unknown { .. } => "unknown",
        }
    }

    pub fn implies_sum_recurrence(&self) -> bool {
        !matches!(self, BoundaryStatus2d::Unknown { .. })
    }

    /// `j_range`/`k_range` are full range descriptions such as `"j = 0..40"`
    /// (the same convention [`super::super::boundary::BoundaryStatus::side_conditions`]
    /// uses) — not bare bounds, since the variable name is not otherwise
    /// repeated in the message.
    pub fn side_conditions(&self, j_range: &str, k_range: &str) -> Vec<String> {
        match self {
            BoundaryStatus2d::Vanishes => vec![format!(
                "all four boundary strips for {j_range}, {k_range} were proved to \
                 vanish identically (pointwise, which is sufficient but not necessary), so the \
                 homogeneous recurrence sum_i a_i(n)*S(n+i) = 0 holds for the double sum"
            )],
            BoundaryStatus2d::Nonzero { witness_n, .. } => vec![format!(
                "the boundary does not vanish; b({witness_n}) != 0 was checked exactly"
            )],
            BoundaryStatus2d::Unknown { reason } => vec![format!(
                "the 2-D boundary for {j_range}, {k_range} could not be decided \
                 ({reason}); the certificate proves the telescoping identity in (j,k) and \
                 NOTHING follows about sum_j sum_k F(n,j,k) until this is discharged \
                 independently"
            )],
        }
    }
}

/// The verdict on the boundary of an `m`-dimensional box sum, the general
/// form of [`BoundaryStatus2d`] (`m = 2`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundaryStatusMd {
    /// Every one of the `2m` boundary faces was proved identically zero, so
    /// the homogeneous recurrence `Σ_i a_i(n)·S(n+i) = 0` holds for the sum.
    Vanishes,
    /// Reserved for a future extension, exactly as
    /// [`BoundaryStatus2d::Nonzero`] — never produced by this version.
    Nonzero { rhs: ExprId, witness_n: i64 },
    /// Neither was established. **No** recurrence for the sum follows.
    Unknown { reason: String },
}

impl BoundaryStatusMd {
    pub fn tag(&self) -> &'static str {
        match self {
            BoundaryStatusMd::Vanishes => "vanishes",
            BoundaryStatusMd::Nonzero { .. } => "nonzero",
            BoundaryStatusMd::Unknown { .. } => "unknown",
        }
    }

    pub fn implies_sum_recurrence(&self) -> bool {
        !matches!(self, BoundaryStatusMd::Unknown { .. })
    }

    /// `ranges[t]` is a full range description for bound index `t`, such as
    /// `"x1 = 0..40"` — one entry per bound index, in the same order
    /// `indices` was supplied to [`boundary_status_md`].
    pub fn side_conditions(&self, ranges: &[String]) -> Vec<String> {
        let joined = ranges.join(", ");
        match self {
            BoundaryStatusMd::Vanishes => vec![format!(
                "all {} boundary faces for {joined} were proved to vanish identically \
                 (pointwise, which is sufficient but not necessary), so the homogeneous \
                 recurrence sum_i a_i(n)*S(n+i) = 0 holds for the {}-fold sum",
                2 * ranges.len(),
                ranges.len()
            )],
            BoundaryStatusMd::Nonzero { witness_n, .. } => vec![format!(
                "the boundary does not vanish; b({witness_n}) != 0 was checked exactly"
            )],
            BoundaryStatusMd::Unknown { reason } => vec![format!(
                "the boundary for {joined} could not be decided ({reason}); the certificate \
                 proves the telescoping identity in the bound indices and NOTHING follows \
                 about the sum until this is discharged independently"
            )],
        }
    }
}

/// Decide the boundary hypothesis for `result` over the rectangle
/// `j = j_limits.0 .. j_limits.1`, `k = k_limits.0 .. k_limits.1`.
///
/// `term` must be the same `F(n,j,k)` that produced `result`. Both limit
/// pairs must be integer constants (see the [module docs](self)); anything
/// else — an `n`-dependent bound, a second free symbol, a non-integer — is
/// reported [`BoundaryStatus2d::Unknown`] rather than guessed.
///
/// A thin wrapper around [`boundary_status_md`] with `indices = [j, k]` — see
/// the module docs.
#[allow(clippy::too_many_arguments)]
pub fn boundary_status_2d(
    result: &Telescoping2dResult,
    term: ExprId,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    j_limits: (ExprId, ExprId),
    k_limits: (ExprId, ExprId),
    pool: &ExprPool,
) -> BoundaryStatus2d {
    let md_result = TelescopingMdResult {
        order: result.order,
        coeffs: result.coeffs.clone(),
        certs: vec![result.cert1, result.cert2],
    };
    match boundary_status_md(&md_result, term, n, &[j, k], &[j_limits, k_limits], pool) {
        BoundaryStatusMd::Vanishes => BoundaryStatus2d::Vanishes,
        BoundaryStatusMd::Nonzero { rhs, witness_n } => {
            BoundaryStatus2d::Nonzero { rhs, witness_n }
        }
        BoundaryStatusMd::Unknown { reason } => BoundaryStatus2d::Unknown { reason },
    }
}

/// Decide the boundary hypothesis for a general `m`-bound-index `result`
/// over the box `indices[t] = limits[t].0 .. limits[t].1`.
///
/// `term` must be the same `F(n,x)` that produced `result`, and `indices` the
/// same order used to produce it. Every limit must be an integer constant
/// (see the [module docs](self)); anything else is reported
/// [`BoundaryStatusMd::Unknown`] rather than guessed.
pub fn boundary_status_md(
    result: &TelescopingMdResult,
    term: ExprId,
    n: ExprId,
    indices: &[ExprId],
    limits: &[(ExprId, ExprId)],
    pool: &ExprPool,
) -> BoundaryStatusMd {
    match analyze_md(result, term, n, indices, limits, pool) {
        Ok(()) => BoundaryStatusMd::Vanishes,
        Err(reason) => BoundaryStatusMd::Unknown { reason },
    }
}

fn const_limit(
    expr: ExprId,
    n: ExprId,
    indices: &[ExprId],
    pool: &ExprPool,
) -> Result<Rational, String> {
    let (coeffs, d) = affine_partsm(expr, n, indices, pool).ok_or_else(|| {
        format!(
            "limit {} is not an integer-affine expression in n and the bound indices",
            pool.display(expr)
        )
    })?;
    if coeffs.iter().any(|&c| c != 0) {
        return Err(format!(
            "limit {} depends on n or on a bound index; this module only supports constant \
             (n-independent) rectangular ranges — see the module docs",
            pool.display(expr)
        ));
    }
    if *d.clone().denom() != 1 {
        return Err(format!("limit {} is not an integer", pool.display(expr)));
    }
    Ok(d)
}

fn analyze_md(
    result: &TelescopingMdResult,
    term: ExprId,
    n: ExprId,
    indices: &[ExprId],
    limits: &[(ExprId, ExprId)],
    pool: &ExprPool,
) -> Result<(), String> {
    let m = indices.len();
    if limits.len() != m {
        return Err(format!(
            "{m} bound indices were supplied but {} limit pairs",
            limits.len()
        ));
    }
    if result.certs.len() != m {
        return Err(format!(
            "result carries {} certificates but {m} bound indices were supplied",
            result.certs.len()
        ));
    }
    let f = ProperTermM::parse(term, n, indices, pool)
        .map_err(|e| format!("term does not re-parse as proper hypergeometric: {e}"))?;

    let mut certs = Vec::with_capacity(m);
    for (t, &c) in result.certs.iter().enumerate() {
        let parsed = as_ratm(c, n, indices, pool, 0)
            .ok_or_else(|| format!("certificate {} does not parse back into Q(n,x)", t + 1))?;
        certs.push(parsed);
    }

    let mut los = Vec::with_capacity(m);
    let mut his_p1 = Vec::with_capacity(m);
    for &(lo_e, hi_e) in limits {
        let lo = const_limit(lo_e, n, indices, pool)?;
        let hi = const_limit(hi_e, n, indices, pool)?;
        his_p1.push(hi + Rational::from(1));
        los.push(lo);
    }

    for t in 0..m {
        let axis: Axis = t + 1;
        for (label, value) in [
            (format!("x{} = lo", t + 1), &los[t]),
            (format!("x{} = hi + 1", t + 1), &his_p1[t]),
        ] {
            if !face_vanishes(
                &f.gammas,
                &f.rat.den,
                &certs[t].num,
                &certs[t].den,
                axis,
                value,
                m + 1,
            ) {
                return Err(format!(
                    "boundary face at {label} was not provably zero by the (deliberately \
                     conservative) pointwise criterion this module implements"
                ));
            }
        }
    }
    Ok(())
}

/// See the [module docs](self) for the exact soundness argument. Returns
/// `true` only when the corresponding `G = c·F` is *provably* the zero
/// function of its remaining free variables — never a guess, and never a
/// claim about the face *summing* to zero without every term being zero.
///
/// Two independent sufficient routes are checked, either being enough:
///
/// 1. `F` itself vanishes identically along the face (a dominant `1/Γ` zero
///    among `F`'s own gamma factors — the natural-boundary case, e.g.
///    `C(n,k)` vanishing at `k = n+1`).
/// 2. `F` is finite (no unresolved pole) along the face, and the
///    certificate's own numerator is the identically zero polynomial there.
///
/// Both need `F` to have no pole along the face (checked once, up front) —
/// otherwise `0 · ∞` is not resolved by either route, and this function
/// correctly refuses rather than guessing.
#[allow(clippy::too_many_arguments)]
fn face_vanishes(
    gammas: &[GammaFactorM],
    rat_den: &PolyM,
    cert_num: &PolyM,
    cert_den: &PolyM,
    axis: Axis,
    value: &Rational,
    num_axes: usize,
) -> bool {
    let mut zero_order: i64 = 0;
    let mut pole_order: i64 = 0;
    for g in gammas {
        let elim_coeff = g.coeffs[axis];
        let other_nonzero = (0..num_axes).any(|a| a != axis && g.coeffs[a] != 0);
        if other_nonzero {
            // Argument still depends on n or on another free bound index:
            // not usable as a *pointwise-in-everything-remaining* pole/zero.
            continue;
        }
        let new_const = g.d.clone() + Rational::from(elim_coeff) * value.clone();
        if *new_const.clone().denom() != 1 {
            continue;
        }
        let mnum = new_const.numer().clone();
        if mnum > 0 {
            continue; // Gamma(positive integer): finite, non-zero, no information.
        }
        if g.e < 0 {
            zero_order += (-g.e) as i64;
        } else if g.e > 0 {
            pole_order += g.e as i64;
        }
    }
    if pole_order > 0 {
        return false; // F itself has an unresolved pole along this face.
    }
    let const_poly = PolyM::constant(value.clone(), num_axes);
    if rat_den.eliminate_axis(axis, &const_poly).is_zero() {
        return false; // F's rational prefactor has an unresolved pole.
    }
    if cert_den.eliminate_axis(axis, &const_poly).is_zero() {
        return false; // the certificate's own denominator vanishes here.
    }
    if zero_order > 0 {
        return true; // route 1: F itself is identically zero along the face.
    }
    // route 2: F is finite here (checked above); zero if the certificate's
    // own numerator is identically zero along the face.
    cert_num.eliminate_axis(axis, &const_poly).is_zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holonomic::telescoping2d::search::{telescope2d_search, Telescoping2dOpts};
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

    /// `Σ_j Σ_k 2^n·C(10,j)*C(10,k)` over a constant rectangle covering the
    /// true (n-*independent*) support: the natural boundary should be
    /// provably `Vanishes`.
    #[test]
    fn separable_fixed_support_boundary_vanishes() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let two_n = pool.pow(pool.integer(2_i32), n);
        let f = pool.mul(vec![
            two_n,
            binom(&pool, pool.integer(10_i32), j),
            binom(&pool, pool.integer(10_i32), k),
        ]);
        let opts = Telescoping2dOpts::default();
        let result = telescope2d_search(f, n, j, k, &pool, &opts).expect("certificate");
        let lo = pool.integer(0_i32);
        let hi = pool.integer(15_i32);
        let status = boundary_status_2d(&result, f, n, j, k, (lo, hi), (lo, hi), &pool);
        assert_eq!(
            status.tag(),
            "vanishes",
            "expected the natural boundary to vanish, got {status:?}"
        );
    }

    /// `Σ_j Σ_k C(n,j)*C(n,k)`: the true support of `C(n,j)` grows with `n`,
    /// so **no constant rectangle can soundly dominate it for every symbolic
    /// `n`** — a rectangle sized for `n <= 30` misses genuinely non-zero
    /// terms for `n > 30`. The module's constant-range boundary analysis
    /// correctly refuses `Vanishes` here rather than making a false claim;
    /// this is a regression test for that refusal, not a limitation of the
    /// example. See `separable_fixed_support_boundary_vanishes` above for
    /// the corresponding case where the support genuinely is
    /// `n`-independent and `Vanishes` is provided.
    #[test]
    fn growing_support_correctly_refuses_vanishes() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let f = pool.mul(vec![binom(&pool, n, j), binom(&pool, n, k)]);
        let opts = Telescoping2dOpts::default();
        let result = telescope2d_search(f, n, j, k, &pool, &opts).expect("certificate");
        let lo = pool.integer(0_i32);
        let hi = pool.integer(30_i32);
        let status = boundary_status_2d(&result, f, n, j, k, (lo, hi), (lo, hi), &pool);
        assert_eq!(
            status.tag(),
            "unknown",
            "a constant rectangle cannot soundly dominate n-growing support; got {status:?}"
        );
    }

    /// An `n`-dependent limit must be refused, not silently misread.
    #[test]
    fn n_dependent_limit_is_unknown() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let f = pool.mul(vec![binom(&pool, n, j), binom(&pool, n, k)]);
        let opts = Telescoping2dOpts::default();
        let result = telescope2d_search(f, n, j, k, &pool, &opts).expect("certificate");
        let lo = pool.integer(0_i32);
        let status = boundary_status_2d(
            &result,
            f,
            n,
            j,
            k,
            (lo, n),
            (lo, pool.integer(30_i32)),
            &pool,
        );
        assert_eq!(status.tag(), "unknown");
    }
}
