//! Deciding the *two-dimensional* boundary hypothesis a double-sum
//! certificate rests on.
//!
//! [`super::search::telescope2d_search`] proves an identity about the
//! **summand**:
//!
//! ```text
//! Σ_i a_i(n)·F(n+i,j,k) = Δ_j G_1(n,j,k) + Δ_k G_2(n,j,k)
//! ```
//!
//! Turning that into a statement about `S(n) = Σ_j Σ_k F(n,j,k)` over a
//! rectangular range `j = j_lo..j_hi`, `k = k_lo..k_hi` is a second, separate
//! step — exactly as in the single-sum engine
//! ([`super::super::boundary`]) — and it is genuinely *more* subtle here,
//! not just a repeat of the 1-D argument in two coordinates.
//!
//! # The boundary is four strip sums, not four corner evaluations
//!
//! Summing the identity over the rectangle and telescoping each difference
//! operator over its own index (Fubini — the order of summation does not
//! matter for a finite double sum) gives:
//!
//! ```text
//! Σ_i a_i(n)·Σ_j Σ_k F(n+i,j,k)
//!   = Σ_{k=k_lo}^{k_hi} [G_1(n, j_hi+1, k) − G_1(n, j_lo, k)]
//!   + Σ_{j=j_lo}^{j_hi} [G_2(n, j, k_hi+1) − G_2(n, j, k_lo)]
//! ```
//!
//! The right-hand side is **four one-dimensional sums along the rectangle's
//! edges**, not four point evaluations at its corners. Point-evaluating `G_1`
//! and `G_2` at the four corners `(j_lo, k_lo)`, `(j_lo, k_hi+1)`,
//! `(j_hi+1, k_lo)`, `(j_hi+1, k_hi+1)` and combining them with `±` signs — the
//! naive generalisation of the 1-D endpoint story — is simply the wrong
//! formula: it neither computes what is above and would double count in
//! exactly the case where it happened to (this module does **not** do that,
//! and its docs above are the reason).
//!
//! # What this module actually proves, and what it refuses to
//!
//! Summing a strip in closed form is, in general, itself a creative-
//! telescoping problem — potentially as hard as the original one. This
//! module does not attempt that. Instead it establishes the **sufficient**
//! (not necessary) condition that each of the four strips is *identically
//! the zero function* of its remaining free variables, in which case the sum
//! along it is trivially `0` term by term. Concretely, for the strip
//! `G_1(n, j_lo, k)` (as a function of `n` and `k`): if `F`'s parsed
//! `Γ(a·n + b·j + c·k + d)^e` factor list contains an `e < 0` factor whose
//! argument, after substituting `j ↦ j_lo`, no longer depends on `n` or `k`
//! (i.e. its `n`- and `k`-coefficients vanish) and evaluates to a
//! non-positive integer, then `1/Γ(·)^{|e|}` is exactly `0` there — a
//! genuine identity (`1/Γ` at a non-positive integer is `0`, not merely a
//! limit), not a numeric coincidence — and as long as no *other* factor in
//! `F`, its rational prefactor, or the certificate's own denominator has a
//! matching pole there, the whole strip is `0`.
//!
//! This is the same *tool* the single-sum engine's order counting uses (a
//! `1/Γ` factor at a non-positive integer argument), but a much simpler use
//! of it: [`super::super::boundary`] locates a pole *approaching an endpoint
//! along the summed variable* and counts orders there, because in 1-D the
//! endpoint is generally the only place `k` is fixed to a specific affine
//! function of `n`. Here the swept boundary line already fixes one entire
//! index (`j` or `k`) to a constant, so the question collapses to "is this
//! `Γ` argument *identically* a non-positive integer over the remaining
//! variables" — a purely algebraic check on `F`'s known integer coefficients,
//! no order-counting needed. It is **strictly weaker** than the single-sum
//! module's analysis: it never resolves a vanishing that needs cancellation
//! between multiple factors, and it proves nothing about a strip that is
//! merely non-constant-but-summing-to-zero. Both of those report
//! [`BoundaryStatus2d::Unknown`] rather than a guess.
//!
//! # Scope: only ranges that do not depend on `n`
//!
//! The single-sum module's `b(n)` formula has an extra term `D_i(n)`
//! (`boundary.rs`, module docs) precisely because summing
//! `F(n+i, k)` over `k`'s range *at n*, rather than at `n+i`, is not what
//! `S(n+i)` means when the range itself moves with `n`. That correction
//! carries over unchanged in spirit to two dimensions, but doubles the
//! bookkeeping (both `j` and `k` ranges could each move, independently, and
//! with each other). This module does not implement it: **`j_lo, j_hi, k_lo,
//! k_hi` must be integer constants**, not expressions in `n`. A caller with
//! an `n`-dependent natural range (e.g. `j = 0..n`) can still use this module
//! whenever the summand vanishes combinatorially outside the true range (the
//! ordinary situation for binomial-type double sums): pick a fixed bound
//! safely larger than any `n` of interest and let `F`'s own vanishing do the
//! rest — exactly how the worked example in `mod.rs` is set up. Passing an
//! `n`-dependent limit expression is refused as
//! [`BoundaryStatus2d::Unknown`], not silently misinterpreted.

use super::poly::{Axis, Poly3};
use super::search::Telescoping2dResult;
use super::term::{affine_parts3, as_rat3, GammaFactor3, ProperTerm3};
use crate::kernel::{ExprId, ExprPool};
use rug::Rational;

/// The verdict on the boundary of a rectangular double sum. See the
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

/// Decide the boundary hypothesis for `result` over the rectangle
/// `j = j_limits.0 .. j_limits.1`, `k = k_limits.0 .. k_limits.1`.
///
/// `term` must be the same `F(n,j,k)` that produced `result`. Both limit
/// pairs must be integer constants (see the [module docs](self)); anything
/// else — an `n`-dependent bound, a second free symbol, a non-integer — is
/// reported [`BoundaryStatus2d::Unknown`] rather than guessed.
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
    match analyze(result, term, n, j, k, j_limits, k_limits, pool) {
        Ok(()) => BoundaryStatus2d::Vanishes,
        Err(reason) => BoundaryStatus2d::Unknown { reason },
    }
}

fn const_limit(
    expr: ExprId,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Result<Rational, String> {
    let (a, b, c, d) = affine_parts3(expr, n, j, k, pool).ok_or_else(|| {
        format!(
            "limit {} is not an integer-affine expression in n, j, k",
            pool.display(expr)
        )
    })?;
    if a != 0 || b != 0 || c != 0 {
        return Err(format!(
            "limit {} depends on n (or on j/k); this module only supports constant \
             (n-independent) rectangular ranges — see the module docs",
            pool.display(expr)
        ));
    }
    if *d.clone().denom() != 1 {
        return Err(format!("limit {} is not an integer", pool.display(expr)));
    }
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
fn analyze(
    result: &Telescoping2dResult,
    term: ExprId,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    j_limits: (ExprId, ExprId),
    k_limits: (ExprId, ExprId),
    pool: &ExprPool,
) -> Result<(), String> {
    let f = ProperTerm3::parse(term, n, j, k, pool)
        .map_err(|e| format!("term does not re-parse as proper hypergeometric: {e}"))?;
    let cert1 = as_rat3(result.cert1, n, j, k, pool, 0)
        .ok_or_else(|| "certificate 1 does not parse back into Q(n,j,k)".to_string())?;
    let cert2 = as_rat3(result.cert2, n, j, k, pool, 0)
        .ok_or_else(|| "certificate 2 does not parse back into Q(n,j,k)".to_string())?;

    let j_lo = const_limit(j_limits.0, n, j, k, pool)?;
    let j_hi = const_limit(j_limits.1, n, j, k, pool)?;
    let k_lo = const_limit(k_limits.0, n, j, k, pool)?;
    let k_hi = const_limit(k_limits.1, n, j, k, pool)?;
    let j_hi_p1 = j_hi + Rational::from(1);
    let k_hi_p1 = k_hi + Rational::from(1);

    let lines: [(Axis, &Rational, &str, &Poly3, &Poly3); 4] = [
        (Axis::J, &j_lo, "j = j_lo", &cert1.num, &cert1.den),
        (Axis::J, &j_hi_p1, "j = j_hi + 1", &cert1.num, &cert1.den),
        (Axis::K, &k_lo, "k = k_lo", &cert2.num, &cert2.den),
        (Axis::K, &k_hi_p1, "k = k_hi + 1", &cert2.num, &cert2.den),
    ];
    for (axis, value, label, cert_num, cert_den) in lines {
        if !line_vanishes(&f.gammas, &f.rat.den, cert_num, cert_den, axis, value) {
            return Err(format!(
                "boundary strip at {label} was not provably zero by the (deliberately \
                 conservative) pointwise criterion this module implements"
            ));
        }
    }
    Ok(())
}

/// See the [module docs](self) for the exact soundness argument. Returns
/// `true` only when the corresponding `G = c·F` is *provably* the zero
/// function of its remaining free variables — never a guess, and never a
/// claim about the strip *summing* to zero without every term being zero.
///
/// Two independent sufficient routes are checked, either being enough:
///
/// 1. `F` itself vanishes identically along the line (a dominant `1/Γ` zero
///    among `F`'s own gamma factors — the natural-boundary case, e.g.
///    `C(n,k)` vanishing at `k = n+1`).
/// 2. `F` is finite (no unresolved pole) along the line, and the
///    certificate's own numerator (`P_1` or `P_2`) is the identically zero
///    polynomial there — the case a classical WZ certificate's own extra
///    factor (e.g. `R(n,k) ∝ k`, zero at `k = 0`) supplies, independent of
///    whether `F` itself vanishes at that endpoint.
///
/// Both need `F` to have no pole along the line (checked once, up front) —
/// otherwise `0 · ∞` is not resolved by either route, and this function
/// correctly refuses rather than guessing.
fn line_vanishes(
    gammas: &[GammaFactor3],
    rat_den: &Poly3,
    cert_num: &Poly3,
    cert_den: &Poly3,
    axis: Axis,
    value: &Rational,
) -> bool {
    let mut zero_order: i64 = 0;
    let mut pole_order: i64 = 0;
    for g in gammas {
        let (elim_coeff, other_coeff) = match axis {
            Axis::J => (g.b, g.c), // eliminate j; k is the other free (non-n) axis
            Axis::K => (g.c, g.b), // eliminate k; j is the other free (non-n) axis
            Axis::N => unreachable!("boundary lines are only ever j or k"),
        };
        if g.a != 0 || other_coeff != 0 {
            // Argument still depends on n or on the other free variable:
            // not usable as a *pointwise-in-everything-remaining* pole/zero.
            continue;
        }
        let new_const = g.d.clone() + Rational::from(elim_coeff) * value.clone();
        if *new_const.clone().denom() != 1 {
            continue;
        }
        let m = new_const.numer().clone();
        if m > 0 {
            continue; // Gamma(positive integer): finite, non-zero, no information.
        }
        if g.e < 0 {
            zero_order += (-g.e) as i64;
        } else if g.e > 0 {
            pole_order += g.e as i64;
        }
    }
    if pole_order > 0 {
        return false; // F itself has an unresolved pole along this line.
    }
    let const_poly = Poly3::constant(value.clone());
    if rat_den.eliminate_axis(axis, &const_poly).is_zero() {
        return false; // F's rational prefactor has an unresolved pole.
    }
    if cert_den.eliminate_axis(axis, &const_poly).is_zero() {
        return false; // the certificate's own denominator vanishes here.
    }
    if zero_order > 0 {
        return true; // route 1: F itself is identically zero along the line.
    }
    // route 2: F is finite here (checked above); zero if the certificate's
    // own numerator is identically zero along the line.
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
