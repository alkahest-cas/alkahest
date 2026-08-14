//! Adaptive branch-and-bound range enclosure, verified 1-D integration, and
//! verified sign / root-absence predicates, built on [`super::taylor`].
//!
//! # Soundness contract (recap, see [`super`] for the full statement)
//!
//! * Every enclosure returned here is a **true outer bound** — [`bound_on_box`]
//!   and [`verified_integral`] never shrink an enclosure to hit a tolerance;
//!   they only *stop refining* it. When the work budget
//!   (`max_subdivisions`) runs out before the target tolerance is reached,
//!   the accumulated — possibly wide — enclosure is returned with
//!   `budget_exhausted = true` rather than silently reporting a tighter but
//!   unjustified bound.
//! * [`verified_no_roots`] and [`verified_sign`] return a three-valued
//!   [`Verdict`]. `True` and `False` are both *certified*: `False` is only
//!   returned when a proof is in hand (for root-absence, an intermediate
//!   value theorem argument on a box already proven to be free of poles and
//!   branch cuts). `Undecided` means the enclosure was not informative
//!   enough to prove either case — it is never conflated with `False`.
//! * [`verified_integral`] integrates the **continuous extension** of the
//!   integrand across a *removable* singularity `N(x)/D(x)` with
//!   `N(p) = D(p) = 0` and `D'` non-vanishing — the enclosure there comes
//!   from Cauchy's mean value theorem, not from ignoring the singular point.
//!   Genuine (non-removable) singularities are still refused.
//! * A sub-box that cannot be bounded rigorously (branch cut, pole, or
//!   domain violation persisting after the box has been bisected far below
//!   the scale of the original box) causes the whole call to **refuse**
//!   with the underlying [`super::ValidatedError`], rather than silently
//!   omitting that piece of the domain from the answer.

use super::taylor::{taylor_range, TaylorContext, MAX_ORDER};
use super::{contains_zero, from_bounds, from_float, is_finite, lb, ub, width, ValidatedError};
use crate::ball::ArbBall;
use crate::diff::diff;
use crate::kernel::subs::subs;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::simplify;
use rug::float::Round;
use rug::Float;
use std::collections::{HashMap, VecDeque};

type Result<T> = std::result::Result<T, ValidatedError>;

/// Number of halvings of the *original* box/interval permitted while trying
/// to bisect away from a domain violation (pole, branch cut) before giving
/// up and refusing. 60 halvings shrinks any dimension by a factor of 2^60 —
/// far more than enough to isolate a non-degenerate singularity, and far
/// short of exhausting `f64` exponent range.
const SINGULARITY_BISECTION_LIMIT: i32 = 60;

// ---------------------------------------------------------------------------
// bound_on_box — verified range enclosure
// ---------------------------------------------------------------------------

/// Options controlling the branch-and-bound search in [`bound_on_box`],
/// [`verified_no_roots`], and [`verified_sign`].
#[derive(Clone, Debug)]
pub struct BoundOptions {
    /// Taylor expansion order used on every sub-box (`1..=taylor::MAX_ORDER`).
    pub order: usize,
    /// Working precision in bits.
    pub prec: u32,
    /// Absolute width at which a sub-box's enclosure is accepted without
    /// further refinement.
    pub tol: f64,
    /// Work budget: maximum number of branch-and-bound subdivisions to
    /// perform. When exhausted, the search stops and returns whatever
    /// true-but-possibly-wide enclosure it has accumulated so far — see
    /// [`BoundResult::budget_exhausted`].
    pub max_subdivisions: usize,
}

impl Default for BoundOptions {
    fn default() -> Self {
        BoundOptions {
            order: 6,
            prec: 128,
            tol: 1e-9,
            max_subdivisions: 2048,
        }
    }
}

/// Result of [`bound_on_box`]: a rigorous enclosure of the range of `f` over
/// the box, plus whether the search converged to `tol` or ran out of budget.
#[derive(Clone, Debug)]
pub struct BoundResult {
    enclosure: ArbBall,
    /// `true` if the work budget ran out before every sub-box's enclosure
    /// reached `tol`. The enclosure is still a **sound** outer bound in
    /// either case — this flag only says whether it is as tight as
    /// requested.
    pub budget_exhausted: bool,
    /// Number of branch-and-bound subdivisions actually performed.
    pub subdivisions: usize,
}

impl BoundResult {
    /// The rigorous enclosure as an [`ArbBall`].
    pub fn enclosure(&self) -> &ArbBall {
        &self.enclosure
    }

    /// Rigorous lower bound, rounded **down** to `f64` (never rounds the
    /// true lower bound upward, so this can only widen, never narrow, the
    /// enclosure relative to the arbitrary-precision value).
    pub fn lower(&self) -> f64 {
        self.enclosure.lo().to_f64_round(Round::Down)
    }

    /// Rigorous upper bound, rounded **up** to `f64`.
    pub fn upper(&self) -> f64 {
        self.enclosure.hi().to_f64_round(Round::Up)
    }
}

type FBox = (ExprId, Float, Float);

/// Widest dimension's width (rounded up to `f64`) — used only as a search
/// heuristic and as the floor test for giving up on an unresolvable domain
/// violation; never used in a way that could unsoundly narrow a result.
fn max_dim_width(boxes: &[FBox], prec: u32) -> f64 {
    boxes
        .iter()
        .map(|(_, lo, hi)| Float::with_val(prec, hi - lo).to_f64_round(Round::Up))
        .fold(0.0_f64, f64::max)
}

/// Midpoint of `[lo, hi]` at working precision.
///
/// Round-to-nearest of a value that lies between two `prec`-representable
/// numbers cannot leave `[lo, hi]`, so the result is always a point *of the
/// interval* — which is what every use here relies on (it is evaluated as a
/// witness point, or used as a split boundary).
fn midpoint(lo: &Float, hi: &Float, prec: u32) -> Float {
    Float::with_val(prec, Float::with_val(prec, lo + hi) / 2u32)
}

/// Split the widest dimension of `boxes` at its midpoint into two boxes that
/// exactly cover the original (they share the midpoint as a boundary, so
/// their union is the original box with no gap).
fn split_widest(boxes: &[FBox], prec: u32) -> (Vec<FBox>, Vec<FBox>) {
    let mut best = 0usize;
    let mut best_w = Float::with_val(prec, 0.0);
    for (i, (_, lo, hi)) in boxes.iter().enumerate() {
        let w = Float::with_val(prec, hi - lo);
        if i == 0 || w > best_w {
            best_w = w;
            best = i;
        }
    }
    let (v, lo, hi) = &boxes[best];
    let mid = midpoint(lo, hi, prec);
    let mut b1 = boxes.to_vec();
    let mut b2 = boxes.to_vec();
    b1[best] = (*v, lo.clone(), mid.clone());
    b2[best] = (*v, mid, hi.clone());
    (b1, b2)
}

fn is_recoverable_domain_issue(e: &ValidatedError) -> bool {
    matches!(
        e,
        ValidatedError::DomainViolation { .. } | ValidatedError::NotFinite { .. }
    )
}

/// Rigorous enclosure of the range of `expr` over an axis-aligned box, via
/// adaptive Taylor-model branch-and-bound subdivision.
///
/// `boxes` is a list of `(variable, lo, hi)`; every free symbol in `expr`
/// must appear exactly once.
///
/// The search is Moore–Skelboe branch-and-bound, run once for the minimum
/// and once for the maximum. Each pass keeps a rigorous bound on the
/// extremum and **prunes** any sub-box whose enclosure proves it cannot
/// contain that extremum, so effort concentrates where the extremum
/// actually is. This matters: a stopping rule that instead demanded every
/// sub-box's enclosure be narrower than `tol` would be asking for a
/// pointwise-tight model everywhere, which no finite budget achieves for a
/// function with a wide range — `exp` on `[-5, 5]` would exhaust the budget
/// and return an enclosure loose enough to still straddle zero.
///
/// A pass stops when the remaining uncertainty in the extremum is at most
/// `tol`, or when `opts.max_subdivisions` is exhausted — at which point the
/// enclosure is returned, still sound and possibly wide, with
/// `budget_exhausted = true`.
///
/// If a sub-box hits an unsupported primitive or an out-of-domain symbol,
/// the error is returned immediately (subdividing cannot help). If a
/// sub-box hits a domain violation that *might* be a boundary effect (a
/// pole or branch cut inside the box), the search first tries bisecting
/// away from it; only once the box has been bisected
/// a fixed number of times relative to its original size (or
/// the subdivision budget runs out) does it give up and refuse — this is
/// the correct behaviour for a genuine interior singularity, where the true
/// range is not a bounded real interval.
///
/// # Example
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::validated::bounds::{bound_on_box, BoundOptions};
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let one = pool.integer(1_i32);
/// // x(1-x); true range on [0,1] is [0, 1/4], but naive intervals give [0,1]
/// let minus_x = pool.mul(vec![pool.integer(-1_i32), x]);
/// let e = pool.mul(vec![x, pool.add(vec![one, minus_x])]);
/// let r = bound_on_box(e, &pool, &[(x, 0.0, 1.0)], &BoundOptions::default()).unwrap();
/// assert!(r.lower() <= 0.0 && r.upper() >= 0.25);
/// ```
pub fn bound_on_box(
    expr: ExprId,
    pool: &ExprPool,
    boxes: &[(ExprId, f64, f64)],
    opts: &BoundOptions,
) -> Result<BoundResult> {
    let prec = opts.prec;
    let boxes0: Vec<FBox> = boxes
        .iter()
        .map(|(v, lo, hi)| (*v, Float::with_val(prec, lo), Float::with_val(prec, hi)))
        .collect();
    bound_on_fboxes(expr, pool, &boxes0, opts)
}

/// [`bound_on_box`] over a box whose endpoints are already exact working-precision
/// `Float`s.
///
/// The public entry point rounds `f64` endpoints into `Float`s; sub-interval
/// endpoints produced by repeated bisection need more than 53 bits, so the
/// internal callers use this form rather than round-tripping through `f64`
/// (which would silently bound a *different* box).
fn bound_on_fboxes(
    expr: ExprId,
    pool: &ExprPool,
    boxes0: &[FBox],
    opts: &BoundOptions,
) -> Result<BoundResult> {
    if boxes0.is_empty() {
        return Err(ValidatedError::InvalidInput {
            what: "the box must constrain at least one variable".into(),
        });
    }
    if opts.order == 0 || opts.order > MAX_ORDER {
        return Err(ValidatedError::InvalidInput {
            what: format!("Taylor order must be in 1..={MAX_ORDER}"),
        });
    }
    let prec = opts.prec;

    let (lo_bound, used_lo, exhausted_lo) =
        extremum_search(expr, pool, boxes0, opts, Extremum::Min)?;
    let (hi_bound, used_hi, exhausted_hi) =
        extremum_search(expr, pool, boxes0, opts, Extremum::Max)?;

    let enclosure = from_bounds(&lo_bound, &hi_bound, prec);
    if !is_finite(&enclosure) {
        return Err(ValidatedError::NotFinite {
            what: "range enclosure".into(),
        });
    }
    Ok(BoundResult {
        enclosure,
        budget_exhausted: exhausted_lo || exhausted_hi,
        subdivisions: used_lo + used_hi,
    })
}

/// Which end of the range an [`extremum_search`] pass is bounding.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Extremum {
    Min,
    Max,
}

/// Moore–Skelboe branch-and-bound for one end of the range.
///
/// Returns `(bound, subdivisions, budget_exhausted)` where `bound` is a
/// **rigorous** lower bound on the global minimum (`Extremum::Min`) or upper
/// bound on the global maximum (`Extremum::Max`) — outward in both cases, so
/// the enclosure built from the pair can only be too wide, never too narrow.
fn extremum_search(
    expr: ExprId,
    pool: &ExprPool,
    boxes0: &[FBox],
    opts: &BoundOptions,
    which: Extremum,
) -> Result<(Float, usize, bool)> {
    let prec = opts.prec;
    let floor = max_dim_width(boxes0, prec) * 2f64.powi(-SINGULARITY_BISECTION_LIMIT);
    let tol_f = Float::with_val(prec, opts.tol);

    // Signed view: work with `f` for Min and `-f` for Max, so one code path
    // handles both. `key(b)` is the rigorous lower bound of the signed range.
    let sign_lo = |r: &ArbBall| -> Float {
        match which {
            Extremum::Min => lb(r),
            Extremum::Max => -ub(r),
        }
    };
    let sign_hi = |r: &ArbBall| -> Float {
        match which {
            Extremum::Min => ub(r),
            Extremum::Max => -lb(r),
        }
    };

    // Active list of (lower bound of signed range, box). `best_ub` is the best
    // proven upper bound on the signed extremum, from any box seen so far.
    let mut active: Vec<(Float, Vec<FBox>)> = Vec::new();
    let mut best_ub: Option<Float> = None;
    let mut subdivisions = 0usize;
    let mut exhausted = false;

    let seed = evaluate_box(expr, pool, boxes0, opts, floor, prec)?;
    match seed {
        BoxOutcome::Range(r) => {
            best_ub = Some(sign_hi(&r));
            active.push((sign_lo(&r), boxes0.to_vec()));
        }
        BoxOutcome::Refine => {
            active.push((Float::with_val(prec, f64::NEG_INFINITY), boxes0.to_vec()));
        }
    }

    // Pick the box with the smallest signed lower bound each round: it is the
    // only one that can still improve the extremum.
    while let Some(idx) = argmin_key(&active) {
        let (key, b) = active.swap_remove(idx);

        // Prune: this box cannot contain the extremum.
        if let Some(ub_best) = &best_ub {
            if &key > ub_best {
                continue;
            }
            // Converged: the uncertainty in the extremum is within tol.
            let gap = Float::with_val(prec, ub_best - &key);
            if gap <= tol_f {
                active.push((key, b));
                break;
            }
        }

        if subdivisions >= opts.max_subdivisions {
            exhausted = true;
            active.push((key, b));
            break;
        }
        if max_dim_width(&b, prec) <= floor {
            // Cannot refine further; keep it as-is so its bound still counts.
            active.push((key, b));
            // Everything else is either pruned or equally unrefinable.
            let stuck = active
                .iter()
                .all(|(_, bx)| max_dim_width(bx, prec) <= floor);
            if stuck {
                exhausted = true;
                break;
            }
            continue;
        }

        subdivisions += 1;
        let (b1, b2) = split_widest(&b, prec);
        for child in [b1, b2] {
            match evaluate_box(expr, pool, &child, opts, floor, prec)? {
                BoxOutcome::Range(r) => {
                    let child_ub = sign_hi(&r);
                    best_ub = Some(match best_ub {
                        Some(cur) if cur <= child_ub => cur,
                        _ => child_ub,
                    });
                    active.push((sign_lo(&r), child));
                }
                BoxOutcome::Refine => {
                    active.push((Float::with_val(prec, f64::NEG_INFINITY), child));
                }
            }
        }
    }

    // The rigorous bound on the signed extremum is the smallest lower bound
    // still active (pruned boxes provably cannot beat `best_ub`).
    let mut bound = active
        .iter()
        .map(|(k, _)| k.clone())
        .fold(None::<Float>, |acc, k| {
            Some(match acc {
                Some(cur) if cur <= k => cur,
                _ => k,
            })
        })
        .or_else(|| best_ub.clone())
        .ok_or_else(|| ValidatedError::InvalidInput {
            what: "no enclosure was produced".into(),
        })?;

    if let Some(ub_best) = &best_ub {
        if &bound > ub_best {
            bound = ub_best.clone();
        }
    }
    if !bound.is_finite() {
        return Err(ValidatedError::NotFinite {
            what: "range enclosure".into(),
        });
    }
    // Undo the sign flip for the Max pass.
    Ok((
        match which {
            Extremum::Min => bound,
            Extremum::Max => -bound,
        },
        subdivisions,
        exhausted,
    ))
}

/// What to do with a sub-box after trying to evaluate it.
enum BoxOutcome {
    /// A rigorous enclosure of `f` on the box.
    Range(ArbBall),
    /// A recoverable domain issue (possible pole/branch cut inside): bisecting
    /// away from it may still work.
    Refine,
}

fn evaluate_box(
    expr: ExprId,
    pool: &ExprPool,
    b: &[FBox],
    opts: &BoundOptions,
    floor: f64,
    prec: u32,
) -> Result<BoxOutcome> {
    match taylor_range(expr, pool, b, opts.order, prec) {
        Ok(r) => Ok(BoxOutcome::Range(r)),
        Err(e) if is_recoverable_domain_issue(&e) => {
            if max_dim_width(b, prec) <= floor {
                // Bisecting has stopped helping: this is a genuine interior
                // singularity, and the true range is not a bounded interval.
                Err(e)
            } else {
                Ok(BoxOutcome::Refine)
            }
        }
        Err(e) => Err(e),
    }
}

/// Index of the entry with the smallest key.
fn argmin_key(active: &[(Float, Vec<FBox>)]) -> Option<usize> {
    let mut best: Option<(usize, &Float)> = None;
    for (i, (k, _)) in active.iter().enumerate() {
        match best {
            Some((_, bk)) if bk <= k => {}
            _ => best = Some((i, k)),
        }
    }
    best.map(|(i, _)| i)
}

// ---------------------------------------------------------------------------
// verified_integral — verified 1-D definite integral
// ---------------------------------------------------------------------------

/// Options controlling the adaptive quadrature in [`verified_integral`].
#[derive(Clone, Debug)]
pub struct IntegralOptions {
    /// Taylor expansion order used on every sub-interval.
    pub order: usize,
    /// Working precision in bits.
    pub prec: u32,
    /// Target absolute width of the total integral enclosure.
    pub tol: f64,
    /// Work budget: maximum number of adaptive subdivisions.
    pub max_subdivisions: usize,
}

impl Default for IntegralOptions {
    fn default() -> Self {
        IntegralOptions {
            order: 6,
            prec: 128,
            tol: 1e-9,
            max_subdivisions: 2048,
        }
    }
}

/// Result of [`verified_integral`].
#[derive(Clone, Debug)]
pub struct IntegralResult {
    enclosure: ArbBall,
    /// `true` if the work budget ran out before the accumulated enclosure
    /// reached `tol` in width. Still a sound outer bound either way.
    pub budget_exhausted: bool,
    /// Number of adaptive subdivisions actually performed.
    pub subdivisions: usize,
}

impl IntegralResult {
    /// The rigorous enclosure of `∫_a^b f dx` as an [`ArbBall`].
    pub fn enclosure(&self) -> &ArbBall {
        &self.enclosure
    }

    /// Rigorous lower bound, rounded down to `f64`.
    pub fn lower(&self) -> f64 {
        self.enclosure.lo().to_f64_round(Round::Down)
    }

    /// Rigorous upper bound, rounded up to `f64`.
    pub fn upper(&self) -> f64 {
        self.enclosure.hi().to_f64_round(Round::Up)
    }
}

/// Rigorous enclosure of `∫ f du` over one sub-interval `[lo, hi]`, using
/// the Taylor model's exact polynomial antiderivative
/// ([`super::taylor::TaylorModel::integrate_normalized_1d`]) rather than a
/// crude `width * range` bound.
fn local_integral(
    expr: ExprId,
    pool: &ExprPool,
    var: ExprId,
    lo: &Float,
    hi: &Float,
    order: usize,
    prec: u32,
) -> Result<ArbBall> {
    let boxes = vec![(var, lo.clone(), hi.clone())];
    let mut ctx = TaylorContext::new(pool, &boxes, order, prec)?;
    let tm = ctx.eval(expr)?;
    let poly_integral = tm.integrate_normalized_1d()?;
    // x = c + r*u  =>  dx = r*du; r = (hi-lo)/2.
    let half_width = Float::with_val(prec, Float::with_val(prec, hi - lo) / 2u32);
    let r_ball = from_float(&half_width, prec);
    let piece = r_ball * poly_integral;
    if !is_finite(&piece) {
        return Err(ValidatedError::NotFinite {
            what: "integral piece".into(),
        });
    }
    Ok(piece)
}

// ---------------------------------------------------------------------------
// Removable singularities
// ---------------------------------------------------------------------------

/// Split a product into `(numerator, denominator)` at its negative integer
/// powers, so that `expr = numerator / denominator` as real functions wherever
/// the denominator does not vanish.
///
/// Returns `None` when there is no negative power to divide by — such an
/// expression cannot have a `0/0` removable singularity in this shape.
fn split_quotient(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let factors: Vec<ExprId> = match pool.get(expr) {
        ExprData::Mul(args) => args.to_vec(),
        ExprData::Pow { .. } => vec![expr],
        _ => return None,
    };
    let mut num: Vec<ExprId> = Vec::new();
    let mut den: Vec<ExprId> = Vec::new();
    for f in factors {
        match pool.get(f) {
            ExprData::Pow { base, exp } => match pool.get(exp) {
                ExprData::Integer(n) if n.0 < 0 => {
                    let k: rug::Integer = -n.0.clone();
                    den.push(pool.pow(base, pool.integer(k)));
                }
                _ => num.push(f),
            },
            _ => num.push(f),
        }
    }
    if den.is_empty() {
        return None;
    }
    let n = match num.len() {
        0 => pool.integer(1_i32),
        1 => num[0],
        _ => pool.mul(num),
    };
    let d = if den.len() == 1 {
        den[0]
    } else {
        pool.mul(den)
    };
    Some((n, d))
}

/// `expr` with `var` replaced by the *exact* rational `at`, simplified.
///
/// Returns `true` only when the result is the literal integer `0`. A numeric
/// enclosure can never prove a value is exactly zero, so the removable-
/// singularity test has to go through the symbolic path; anything short of an
/// exact zero is treated as "not removable", which is the safe direction.
fn vanishes_exactly(expr: ExprId, pool: &ExprPool, var: ExprId, at: &Float) -> bool {
    let Some(rational) = at.to_rational() else {
        return false;
    };
    let (n, d) = rational.into_numer_denom();
    let point = if d == 1 {
        pool.integer(n)
    } else {
        pool.rational(n, d)
    };
    let mut mapping = HashMap::new();
    mapping.insert(var, point);
    let substituted = subs(expr, &mapping, pool);
    let reduced = simplify(substituted, pool).value;
    match pool.get(reduced) {
        ExprData::Integer(n) => n.0.cmp0() == std::cmp::Ordering::Equal,
        ExprData::Rational(r) => r.0.cmp0() == std::cmp::Ordering::Equal,
        _ => false,
    }
}

/// Whether the rigorous enclosure of `expr` at the single point `at` contains
/// zero.
///
/// This is *not* a proof that the value is zero — no numeric enclosure can be
/// one — and it is not used as such. It is a cross-check on
/// [`vanishes_exactly`]: `simplify` reporting an exact zero that outward-rounded
/// ball arithmetic contradicts would mean a bug in the simplifier, and the safe
/// response is to decline the removable branch rather than certify anything on
/// top of it.
fn enclosure_admits_zero(
    expr: ExprId,
    pool: &ExprPool,
    var: ExprId,
    at: &Float,
    order: usize,
    prec: u32,
) -> bool {
    let point = vec![(var, at.clone(), at.clone())];
    match taylor_range(expr, pool, &point, order, prec) {
        Ok(r) => contains_zero(&r),
        Err(_) => false,
    }
}

/// Midpoint of the enclosure of `expr` at the point `at`, or `None` if it
/// could not be evaluated.
fn point_value(
    expr: ExprId,
    pool: &ExprPool,
    var: ExprId,
    at: &Float,
    order: usize,
    prec: u32,
) -> Option<Float> {
    let point = vec![(var, at.clone(), at.clone())];
    let r = taylor_range(expr, pool, &point, order, prec).ok()?;
    r.mid.is_finite().then(|| r.mid.clone())
}

/// The ingredients of the L'Hôpital enclosure for an integrand written as a
/// quotient: `N`, `D`, `D'`, and the mean-value quotient `N'/D'`.
struct RemovableQuotient {
    num: ExprId,
    den: ExprId,
    dden: ExprId,
    /// `N' · (D')⁻¹`.
    ratio: ExprId,
}

impl RemovableQuotient {
    /// Structural analysis of the integrand, done once per
    /// [`verified_integral`] call. This is only *shape* recognition — nothing
    /// here asserts that a removable singularity exists.
    fn detect(expr: ExprId, pool: &ExprPool, var: ExprId) -> Option<Self> {
        let (num, den) = split_quotient(expr, pool)?;
        let dnum = diff(num, var, pool).ok()?.value;
        let dden = diff(den, var, pool).ok()?.value;
        let ratio = pool.mul(vec![dnum, pool.pow(dden, pool.integer(-1_i32))]);
        Some(RemovableQuotient {
            num,
            den,
            dden,
            ratio,
        })
    }

    /// Newton iterates of `D` inside `[lo, hi]`, as *candidate* locations for
    /// the singular point.
    ///
    /// This is pure heuristic search and carries no soundness weight: whatever
    /// points come out are still put through the exact symbolic vanishing test
    /// before anything is certified. Its only job is to name the singular point
    /// exactly when it is a working-precision number that the dyadic bisection
    /// grid would never land on — for a denominator like `x - 1/4` a single
    /// Newton step from anywhere produces exactly `1/4`.
    fn newton_candidates(
        &self,
        pool: &ExprPool,
        var: ExprId,
        lo: &Float,
        hi: &Float,
        opts: &IntegralOptions,
    ) -> Vec<Float> {
        const STEPS: usize = 6;
        let (order, prec) = (opts.order, opts.prec);
        let mut out = Vec::new();
        let mut z = midpoint(lo, hi, prec);
        for _ in 0..STEPS {
            let Some(f) = point_value(self.den, pool, var, &z, order, prec) else {
                break;
            };
            let Some(df) = point_value(self.dden, pool, var, &z, order, prec) else {
                break;
            };
            if df.is_zero() || !df.is_finite() {
                break;
            }
            let next = Float::with_val(prec, &z - Float::with_val(prec, &f / &df));
            if !next.is_finite() || next < *lo || next > *hi || next == z {
                break;
            }
            z = next;
            out.push(z.clone());
        }
        out
    }

    /// Rigorous enclosure of `∫_lo^hi N/D dx` when `N/D` has a **removable**
    /// singularity somewhere in `[lo, hi]`; `None` when that cannot be
    /// established, in which case the caller must keep refusing.
    ///
    /// # Why this is an enclosure
    ///
    /// Write `J = [lo, hi]` and let `p ∈ J` be a point at which `N` and `D`
    /// both vanish *exactly* (checked symbolically, not numerically). The
    /// checks below establish, in order:
    ///
    /// 1. `N` and `D` are analytic on the whole of `J` — a successful
    ///    [`bound_on_fboxes`] covers `J` by sub-boxes on each of which the
    ///    Taylor model was built without a domain violation, and every rule in
    ///    [`super::taylor`] refuses unless its argument stays strictly inside
    ///    the analytic interior of the primitive's domain. Analytic implies
    ///    differentiable, so the symbolic derivatives `N'`, `D'` really are the
    ///    derivatives of `N`, `D` on `J`.
    /// 2. `D'` has no zero on `J` (its enclosure excludes zero). Hence `D` is
    ///    strictly monotone on `J`, so `p` is its *only* zero there and
    ///    `D(x) ≠ 0` for every other `x ∈ J`.
    /// 3. `R` is a rigorous enclosure of the range of `N'/D'` over `J`.
    ///
    /// Cauchy's mean value theorem then gives, for every `x ∈ J \ {p}`, some
    /// `ξ` strictly between `p` and `x` with
    /// `(N(x) − N(p))·D'(ξ) = (D(x) − D(p))·N'(ξ)`; since `N(p) = D(p) = 0`,
    /// `D(x) ≠ 0` and `D'(ξ) ≠ 0`, this is `N(x)/D(x) = N'(ξ)/D'(ξ) ∈ R`.
    /// So the integrand is bounded by `R` on `J` minus a single point, and
    /// `∫_J N/D dx ∈ (hi − lo)·R`.
    ///
    /// Note what is being integrated: the integrand is *undefined* at `p`, and
    /// the value returned is the integral of its continuous extension (which
    /// is the same number for every extension, `{p}` being a null set).
    fn piece(
        &self,
        pool: &ExprPool,
        var: ExprId,
        lo: &Float,
        hi: &Float,
        opts: &IntegralOptions,
    ) -> Option<ArbBall> {
        let prec = opts.prec;
        if lo >= hi {
            return None;
        }
        let bopts = BoundOptions {
            order: opts.order,
            prec,
            tol: opts.tol,
            max_subdivisions: opts.max_subdivisions,
        };
        // The singular point has to be *named* exactly, since only a symbolic
        // test can prove a value is zero. Candidates are the two endpoints, the
        // midpoint, and whatever Newton's method on the denominator turns up —
        // the last of those is what reaches a singularity sitting at a point
        // the dyadic bisection grid never visits. A singularity at a point that
        // no candidate names exactly is simply refused.
        let mut candidates = vec![lo.clone(), hi.clone(), midpoint(lo, hi, prec)];
        candidates.extend(self.newton_candidates(pool, var, lo, hi, opts));
        candidates.iter().find(|p| {
            vanishes_exactly(self.den, pool, var, p)
                && vanishes_exactly(self.num, pool, var, p)
                && enclosure_admits_zero(self.den, pool, var, p, opts.order, prec)
                && enclosure_admits_zero(self.num, pool, var, p, opts.order, prec)
        })?;

        let j = vec![(var, lo.clone(), hi.clone())];
        // (1) N and D analytic on J.
        bound_on_fboxes(self.num, pool, &j, &bopts).ok()?;
        bound_on_fboxes(self.den, pool, &j, &bopts).ok()?;
        // (2) D' non-vanishing on J.
        let dd = bound_on_fboxes(self.dden, pool, &j, &bopts).ok()?;
        if contains_zero(dd.enclosure()) {
            return None;
        }
        // (3) the mean-value quotient.
        let r = bound_on_fboxes(self.ratio, pool, &j, &bopts).ok()?;

        let w = Float::with_val(prec, hi - lo);
        let piece = from_float(&w, prec) * r.enclosure().clone();
        if !is_finite(&piece) {
            return None;
        }
        Some(piece)
    }
}

/// Turn a refusal that survived bisection into a message that says *what* is
/// singular and *where*, and that distinguishes "no rigorous enclosure of the
/// integrand exists here" from "the integral does not exist".
fn describe_singularity(
    cause: ValidatedError,
    lo: &Float,
    hi: &Float,
    a: &Float,
    b: &Float,
) -> ValidatedError {
    let ValidatedError::DomainViolation { what } = &cause else {
        return cause;
    };
    let where_ = if lo <= a {
        "the left endpoint"
    } else if hi >= b {
        "the right endpoint"
    } else {
        "an interior point"
    };
    let at = midpoint(lo, hi, lo.prec()).to_f64();
    ValidatedError::DomainViolation {
        what: format!(
            "the integrand is singular at {where_} x ≈ {at:e} ({what}). \
             This is a statement about the *integrand*, not about the integral: \
             an integrable singularity still has a finite integral, which this \
             routine cannot certify. Removable singularities written as N(x)/D(x) \
             with N(p) = D(p) = 0 exactly and D'(p) ≠ 0 are handled automatically \
             (their continuous extension is integrated); this one is not of that form"
        ),
    }
}

/// Rigorous enclosure of `∫_a^b f(x) dx`, via adaptive Taylor-model
/// quadrature.
///
/// Each accepted sub-interval contributes a rigorous enclosure computed by
/// exactly integrating the Taylor polynomial in normalised coordinates and
/// folding the remainder in as `2·I·r` (see
/// [`super::taylor::TaylorModel::integrate_normalized_1d`]) — this is exact
/// on polynomials up to the truncation order, unlike naive
/// `width × range(f)` quadrature. Sub-interval enclosures are summed
/// (interval addition is exact for disjoint, adjacent pieces), so the total
/// is always a sound outer bound of the true integral.
///
/// # Removable singularities
///
/// An integrand written as `N(x)/D(x)` with `N(p) = D(p) = 0` — `ln(1+x)/x` on
/// `[0, 1]`, `sin(x)/x` on `[-1, 1]` — has no rigorous Taylor model at `p`
/// (the reciprocal's enclosure contains zero) even though nothing about the
/// integral is singular. Such a sub-interval is enclosed through Cauchy's mean
/// value theorem instead, from an enclosure of `N'/D'`; see
/// [`RemovableQuotient::piece`] for the argument. The vanishing of `N` and `D`
/// at `p` is checked *symbolically* — a numeric enclosure cannot prove a value
/// is exactly zero — and `D'` must be certified non-vanishing on the
/// sub-interval, so a genuine pole is never mistaken for a removable one. The
/// number returned is the integral of the continuous extension.
///
/// Refuses (does not guess) when:
/// - `a` or `b` is non-finite (infinite-limit improper integrals are not
///   supported — there is no box to Taylor-expand over),
/// - `a > b`,
/// - the integrand has a singularity in `[a, b]` that is not removable in the
///   above sense (e.g. `1/sqrt(x)` on `[0, 1]`, or the *integrable* endpoint
///   singularity of `-log(x)` on `[0, 1]`) — subdivision is tried first in
///   case the domain violation is only a boundary artefact of a coarse box,
///   but a persistent one refuses with a [`ValidatedError`] that names the
///   location and distinguishes "the integrand is singular here" from "the
///   integral does not exist", rather than silently skipping the offending
///   piece.
pub fn verified_integral(
    expr: ExprId,
    pool: &ExprPool,
    var: ExprId,
    a: f64,
    b: f64,
    opts: &IntegralOptions,
) -> Result<IntegralResult> {
    if !(a.is_finite() && b.is_finite()) {
        return Err(ValidatedError::InvalidInput {
            what: "integration bounds must be finite; infinite-limit improper integrals are not supported".into(),
        });
    }
    if a > b {
        return Err(ValidatedError::InvalidInput {
            what: "verified_integral requires a <= b".into(),
        });
    }
    if opts.order == 0 || opts.order > MAX_ORDER {
        return Err(ValidatedError::InvalidInput {
            what: format!("Taylor order must be in 1..={MAX_ORDER}"),
        });
    }
    let prec = opts.prec;
    if a == b {
        return Ok(IntegralResult {
            enclosure: ArbBall::from_f64(0.0, prec),
            budget_exhausted: false,
            subdivisions: 0,
        });
    }

    let a_f = Float::with_val(prec, a);
    let b_f = Float::with_val(prec, b);
    let total_width = Float::with_val(prec, &b_f - &a_f);
    let floor = total_width.to_f64_round(Round::Up) * 2f64.powi(-SINGULARITY_BISECTION_LIMIT);
    let tol_total = Float::with_val(prec, opts.tol);

    let mut stack: Vec<(Float, Float)> = vec![(a_f.clone(), b_f.clone())];
    let mut total: Option<ArbBall> = None;
    let mut subdivisions = 0usize;
    let mut exhausted = false;
    // Structural `N/D` analysis of the integrand, built at most once and only
    // if a sub-interval actually refuses.
    let mut removable: Option<Option<RemovableQuotient>> = None;

    while let Some((lo, hi)) = stack.pop() {
        let piece_w = Float::with_val(prec, &hi - &lo);
        let piece_tol = Float::with_val(
            prec,
            &tol_total * Float::with_val(prec, &piece_w / &total_width),
        );
        let outcome = match local_integral(expr, pool, var, &lo, &hi, opts.order, prec) {
            Ok(piece) => Ok(piece),
            Err(e) if is_recoverable_domain_issue(&e) => {
                // The Taylor model refuses here. Before bisecting (and before
                // eventually giving up), see whether the refusal is only the
                // `0/0` shape of a removable singularity, which has a rigorous
                // enclosure of its own.
                let q = removable
                    .get_or_insert_with(|| RemovableQuotient::detect(expr, pool, var))
                    .as_ref();
                match q.and_then(|q| q.piece(pool, var, &lo, &hi, opts)) {
                    Some(piece) => Ok(piece),
                    None => Err(e),
                }
            }
            Err(e) => return Err(e),
        };

        match outcome {
            Ok(piece) => {
                let w = width(&piece);
                if w <= piece_tol || subdivisions >= opts.max_subdivisions {
                    if w > piece_tol {
                        exhausted = true;
                    }
                    total = Some(match total {
                        Some(t) => t + piece,
                        None => piece,
                    });
                } else {
                    subdivisions += 1;
                    let mid = midpoint(&lo, &hi, prec);
                    stack.push((mid.clone(), hi));
                    stack.push((lo, mid));
                }
            }
            Err(e) => {
                let w = piece_w.to_f64_round(Round::Up);
                if subdivisions >= opts.max_subdivisions || w <= floor {
                    return Err(describe_singularity(e, &lo, &hi, &a_f, &b_f));
                }
                subdivisions += 1;
                let mid = midpoint(&lo, &hi, prec);
                stack.push((mid.clone(), hi));
                stack.push((lo, mid));
            }
        }
    }

    let enclosure = total.ok_or_else(|| ValidatedError::InvalidInput {
        what: "no enclosure was produced".into(),
    })?;
    if !is_finite(&enclosure) {
        return Err(ValidatedError::NotFinite {
            what: "integral enclosure".into(),
        });
    }
    Ok(IntegralResult {
        enclosure,
        budget_exhausted: exhausted,
        subdivisions,
    })
}

// ---------------------------------------------------------------------------
// Verified predicates: roots and sign
// ---------------------------------------------------------------------------

/// Three-valued verdict for a rigorously-checked predicate.
///
/// `True` and `False` are both certified — the underlying proof is sound
/// under the [module soundness contract](super). `Undecided` means the
/// available enclosure (possibly after exhausting the work budget) was not
/// informative enough to establish either case; it is never collapsed into
/// `True` or `False`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The predicate is certified true everywhere on the box.
    True,
    /// The predicate is certified false — a counterexample is proven to
    /// exist (or, for sign predicates, the range is proven to violate it
    /// somewhere).
    False,
    /// Neither could be established from the computed enclosure.
    Undecided,
}

fn determined_sign(b: &ArbBall) -> Option<bool> {
    if lb(b) > 0 {
        Some(true)
    } else if ub(b) < 0 {
        Some(false)
    } else {
        None
    }
}

/// Proven sign witnesses collected anywhere in one box.
///
/// Only *proven* signs are recorded: a value whose enclosure straddles zero
/// contributes nothing. Once both a positive and a negative witness exist the
/// intermediate value theorem applies along the segment joining them, which
/// stays inside the (convex) box.
#[derive(Default)]
struct SignWitnesses {
    positive: bool,
    negative: bool,
}

impl SignWitnesses {
    fn record(&mut self, s: Option<bool>) {
        match s {
            Some(true) => self.positive = true,
            Some(false) => self.negative = true,
            None => {}
        }
    }

    fn both(&self) -> bool {
        self.positive && self.negative
    }
}

/// Rigorously evaluate `expr` at one point of the box and report its sign, or
/// `None` when the enclosure straddles zero or the evaluation refused.
///
/// The point is passed as a degenerate box, so the Taylor model collapses to a
/// ball evaluation and the sign test goes through [`lb`]/[`ub`], which round
/// outward. A reported sign is therefore a proof, never a rounding artefact.
fn point_sign(
    expr: ExprId,
    pool: &ExprPool,
    point: &[FBox],
    order: usize,
    prec: u32,
) -> Option<bool> {
    match taylor_range(expr, pool, point, order, prec) {
        Ok(r) => determined_sign(&r),
        Err(_) => None,
    }
}

/// Degenerate boxes for the centre, the per-axis endpoints and (in low
/// dimension) the corners of `boxes0` — the cheap witnesses that settle most
/// boxes before any subdivision happens.
fn seed_points(boxes0: &[FBox], prec: u32) -> Vec<Vec<FBox>> {
    let n = boxes0.len();
    let centre: Vec<Float> = boxes0
        .iter()
        .map(|(_, lo, hi)| midpoint(lo, hi, prec))
        .collect();
    let mut points: Vec<Vec<Float>> = vec![centre.clone()];
    for (i, (_, lo, hi)) in boxes0.iter().enumerate() {
        for end in [lo, hi] {
            let mut p = centre.clone();
            p[i] = end.clone();
            points.push(p);
        }
    }
    // 2ⁿ corners is only affordable in low dimension.
    if n <= 3 {
        for mask in 0..(1usize << n) {
            points.push(
                boxes0
                    .iter()
                    .enumerate()
                    .map(|(i, (_, lo, hi))| {
                        if mask & (1 << i) == 0 {
                            lo.clone()
                        } else {
                            hi.clone()
                        }
                    })
                    .collect(),
            );
        }
    }
    points
        .into_iter()
        .map(|p| {
            boxes0
                .iter()
                .zip(p)
                .map(|((v, _, _), c)| (*v, c.clone(), c))
                .collect()
        })
        .collect()
}

/// Search for a proof that `expr` has a root in the box, by finding one point
/// where it is provably positive and one where it is provably negative.
///
/// # Why this is a proof
///
/// The caller must already have obtained a successful [`bound_on_box`] over
/// the same box. That is the continuity certificate: the branch-and-bound
/// search only returns `Ok` once every part of the box lies in some sub-box on
/// which a Taylor model was built, and [`super::taylor`] refuses to build one
/// unless every elementary step stayed strictly inside the analytic interior
/// of its domain (`log`/`sqrt` of a strictly positive enclosure, reciprocals of
/// an enclosure bounded away from zero, `tan` where `cos` cannot vanish, …).
/// A box on which a `Refine` outcome is never resolved keeps a `-∞` key, is
/// never pruned and never satisfies the convergence test, so it can only end in
/// a refusal — `Ok` really does mean "analytic on a finite closed cover of the
/// box", hence continuous on the box.
///
/// The box is a product of intervals and therefore convex, so for any two of
/// its points `p`, `q` the segment `[p, q]` stays inside it. If `f(p) > 0` and
/// `f(q) < 0` — each proven by an outward-rounded enclosure at a degenerate box
/// — then `t ↦ f(p + t(q − p))` is continuous on `[0, 1]` and changes sign, so
/// it vanishes somewhere: a root of `f` inside the box. This is exactly the
/// intermediate value theorem, and it needs no relationship between `p` and `q`
/// beyond both lying in the box — which is what lets an even number of roots
/// (`x² − 2` on `[-2, 2]`, whose *endpoints* are both positive) be settled.
///
/// The search itself is the same Moore–Skelboe subdivision used elsewhere:
/// sub-boxes whose enclosure has a determined sign are recorded as witnesses
/// and dropped (they cannot contain a root); the rest are sampled at their
/// centre and split. It is a *search*, so failing to find a witness pair proves
/// nothing — the caller must answer `Undecided`, never `True`.
fn root_exists_witness(
    expr: ExprId,
    pool: &ExprPool,
    boxes0: &[FBox],
    opts: &BoundOptions,
) -> bool {
    let (prec, order) = (opts.prec, opts.order);
    let floor = max_dim_width(boxes0, prec) * 2f64.powi(-SINGULARITY_BISECTION_LIMIT);
    let mut w = SignWitnesses::default();

    for point in seed_points(boxes0, prec) {
        w.record(point_sign(expr, pool, &point, order, prec));
        if w.both() {
            return true;
        }
    }

    let mut queue: VecDeque<Vec<FBox>> = VecDeque::new();
    queue.push_back(boxes0.to_vec());
    let mut budget = opts.max_subdivisions;

    while let Some(b) = queue.pop_front() {
        if budget == 0 {
            break;
        }
        budget -= 1;

        // A sub-box whose enclosure has a determined sign is non-empty, so it
        // contains a point of that sign — a witness — and it cannot contain a
        // root, so there is nothing left to look for inside it.
        if let Ok(r) = taylor_range(expr, pool, &b, order, prec) {
            if let Some(s) = determined_sign(&r) {
                w.record(Some(s));
                if w.both() {
                    return true;
                }
                continue;
            }
        }

        let centre: Vec<FBox> = b
            .iter()
            .map(|(v, lo, hi)| {
                let m = midpoint(lo, hi, prec);
                (*v, m.clone(), m)
            })
            .collect();
        w.record(point_sign(expr, pool, &centre, order, prec));
        if w.both() {
            return true;
        }

        if max_dim_width(&b, prec) <= floor {
            continue;
        }
        let (b1, b2) = split_widest(&b, prec);
        queue.push_back(b1);
        queue.push_back(b2);
    }

    false
}

/// Verified check for the absence of roots of `expr` on `boxes`.
///
/// Returns:
/// - [`Verdict::True`] when the rigorous range enclosure of `expr` over the
///   whole box does not contain zero — `expr` is certified to have no root
///   anywhere in the box.
/// - [`Verdict::False`] when the box is proven free of poles/branch cuts (the
///   full-box enclosure succeeded, so `expr` is continuous on the box) *and*
///   two points of the box are found at which `expr` is rigorously proven to
///   have opposite signs — a root is then certified to exist by the
///   intermediate value theorem along the segment joining them, which stays in
///   the box because a box is convex. The points are looked for by subdividing
///   the box, so an even number of roots no longer defeats the test: `x² − 2`
///   on `[-2, 2]` has two roots and two positive endpoints, and is settled at
///   the first bisection.
/// - [`Verdict::Undecided`] otherwise: the enclosure straddles zero and the
///   search found no pair of opposite-signed points within the budget. This is
///   the honest answer for a root that never produces a sign change at all —
///   a double root such as `(x − 1)²` — and it is never collapsed into either
///   of the other two verdicts.
///
/// Propagates a [`ValidatedError`] (refuses) exactly when
/// [`bound_on_box`] would: unsupported primitives, unbound symbols, or a
/// singularity that resists bisection.
pub fn verified_no_roots(
    expr: ExprId,
    pool: &ExprPool,
    boxes: &[(ExprId, f64, f64)],
    opts: &BoundOptions,
) -> Result<Verdict> {
    let full = bound_on_box(expr, pool, boxes, opts)?;
    if !contains_zero(full.enclosure()) {
        return Ok(Verdict::True);
    }

    // The call above is the continuity certificate the IVT argument needs; see
    // `root_exists_witness` for why, and for why a witness pair anywhere in the
    // box (not just at its endpoints) is enough.
    let prec = opts.prec;
    let boxes0: Vec<FBox> = boxes
        .iter()
        .map(|(v, lo, hi)| (*v, Float::with_val(prec, lo), Float::with_val(prec, hi)))
        .collect();
    if root_exists_witness(expr, pool, &boxes0, opts) {
        return Ok(Verdict::False);
    }

    Ok(Verdict::Undecided)
}

/// Which sign condition [`verified_sign`] should check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignPredicate {
    /// `f(x) > 0` for every `x` in the box.
    Positive,
    /// `f(x) < 0` for every `x` in the box.
    Negative,
    /// `f(x) >= 0` for every `x` in the box.
    NonNegative,
    /// `f(x) <= 0` for every `x` in the box.
    NonPositive,
}

/// Verified sign check for `expr` over `boxes`, built on [`bound_on_box`].
///
/// Returns [`Verdict::True`] when the rigorous range enclosure proves the
/// predicate holds everywhere on the box, [`Verdict::False`] when the
/// enclosure proves it is violated somewhere, and [`Verdict::Undecided`]
/// when the enclosure straddles the boundary needed to decide either way.
pub fn verified_sign(
    expr: ExprId,
    pool: &ExprPool,
    boxes: &[(ExprId, f64, f64)],
    predicate: SignPredicate,
    opts: &BoundOptions,
) -> Result<Verdict> {
    let r = bound_on_box(expr, pool, boxes, opts)?;
    let lo = lb(r.enclosure());
    let hi = ub(r.enclosure());

    let holds_everywhere = match predicate {
        SignPredicate::Positive => lo > 0,
        SignPredicate::Negative => hi < 0,
        SignPredicate::NonNegative => lo >= 0,
        SignPredicate::NonPositive => hi <= 0,
    };
    if holds_everywhere {
        return Ok(Verdict::True);
    }

    // The enclosure over the whole box proves the predicate fails everywhere.
    let fails_everywhere = match predicate {
        SignPredicate::Positive => hi <= 0,
        SignPredicate::Negative => lo >= 0,
        SignPredicate::NonNegative => hi < 0,
        SignPredicate::NonPositive => lo > 0,
    };
    if fails_everywhere {
        return Ok(Verdict::False);
    }

    // These are universally quantified claims, so a single point where the
    // predicate provably fails disproves them. Without this, `x > 0` on
    // [-1, 1] — plainly false — could only ever come back `Undecided`, since
    // the range enclosure straddles zero by construction.
    if violates_at_some_sample(expr, pool, boxes, predicate, opts)? {
        return Ok(Verdict::False);
    }

    Ok(Verdict::Undecided)
}

/// Rigorously evaluate `expr` at a handful of points of the box (corners and
/// centre) and report whether any of them *proves* the predicate false there.
///
/// Each sample is a degenerate box, so the Taylor model reduces to a point
/// enclosure; the verdict is only `true` when that enclosure lies strictly on
/// the wrong side, which is a proof. A domain violation at a sample is not
/// evidence either way and is skipped.
fn violates_at_some_sample(
    expr: ExprId,
    pool: &ExprPool,
    boxes: &[(ExprId, f64, f64)],
    predicate: SignPredicate,
    opts: &BoundOptions,
) -> Result<bool> {
    let n = boxes.len();
    // Corners get exponential in `n`, so sample them only for small boxes;
    // the centre and the per-axis endpoints are always cheap.
    let mut samples: Vec<Vec<f64>> = Vec::new();
    samples.push(boxes.iter().map(|(_, lo, hi)| 0.5 * (lo + hi)).collect());
    for i in 0..n {
        for pick_lo in [true, false] {
            let mut p: Vec<f64> = boxes.iter().map(|(_, lo, hi)| 0.5 * (lo + hi)).collect();
            p[i] = if pick_lo { boxes[i].1 } else { boxes[i].2 };
            samples.push(p);
        }
    }
    if n <= 3 {
        for mask in 0..(1usize << n) {
            let p: Vec<f64> = boxes
                .iter()
                .enumerate()
                .map(|(i, (_, lo, hi))| if mask & (1 << i) == 0 { *lo } else { *hi })
                .collect();
            samples.push(p);
        }
    }

    for point in samples {
        let degenerate: Vec<(ExprId, f64, f64)> = boxes
            .iter()
            .zip(point.iter())
            .map(|((v, _, _), &c)| (*v, c, c))
            .collect();
        let r = match bound_on_box(expr, pool, &degenerate, opts) {
            Ok(r) => r,
            // A sample landing on a pole or outside the domain says nothing
            // about the predicate; try the next one.
            Err(_) => continue,
        };
        let lo = lb(r.enclosure());
        let hi = ub(r.enclosure());
        let proven_violation = match predicate {
            SignPredicate::Positive => hi <= 0,
            SignPredicate::Negative => lo >= 0,
            SignPredicate::NonNegative => hi < 0,
            SignPredicate::NonPositive => lo > 0,
        };
        if proven_violation {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `a - b` (the pool has no `sub`; subtraction is `a + (-1)·b`).
    fn sub(pool: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
        pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])])
    }

    /// `a / b` (the pool has no `div`; division is `a · b^(-1)`).
    fn div(pool: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
        pool.mul(vec![a, pool.pow(b, pool.integer(-1_i32))])
    }
    use crate::kernel::Domain;

    fn opts() -> BoundOptions {
        BoundOptions {
            order: 6,
            prec: 128,
            tol: 1e-9,
            max_subdivisions: 4096,
        }
    }

    fn iopts() -> IntegralOptions {
        IntegralOptions {
            order: 8,
            prec: 128,
            tol: 1e-9,
            max_subdivisions: 4096,
        }
    }

    // ── bound_on_box ────────────────────────────────────────────────────

    #[test]
    fn x_minus_x_is_tight_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = sub(&pool, x, x);
        let r = bound_on_box(e, &pool, &[(x, -5.0, 5.0)], &opts()).unwrap();
        assert!(r.lower() >= -1e-9 && r.upper() <= 1e-9, "{:?}", r);
    }

    #[test]
    fn x_times_one_minus_x_needs_subdivision_and_converges() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let e = pool.mul(vec![x, sub(&pool, one, x)]);
        let r = bound_on_box(e, &pool, &[(x, 0.0, 1.0)], &opts()).unwrap();
        assert!(r.lower() <= 1e-6, "{:?}", r);
        assert!(
            r.upper() >= 0.25 - 1e-6 && r.upper() <= 0.25 + 1e-6,
            "{:?}",
            r
        );
        assert!(!r.budget_exhausted);
    }

    #[test]
    fn sin_squared_plus_cos_squared_is_one() {
        // A strong dependency-effect case: without correlation tracking a
        // sum of two independently-widened intervals could report a range
        // far from the true constant {1}.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let s = pool.func("sin", vec![x]);
        let c = pool.func("cos", vec![x]);
        let e = pool.add(vec![
            pool.pow(s, pool.integer(2_i32)),
            pool.pow(c, pool.integer(2_i32)),
        ]);
        let r = bound_on_box(e, &pool, &[(x, -3.0, 3.0)], &opts()).unwrap();
        assert!(r.lower() <= 1.0 && r.upper() >= 1.0, "{:?}", r);
        // Should be much tighter than the trivial [-2, 2] naive bound.
        assert!(r.upper() - r.lower() < 0.5, "too wide: {:?}", r);
    }

    #[test]
    fn taylor_bound_beats_plain_interval_eval() {
        use crate::ball::IntervalEval;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let e = pool.mul(vec![x, sub(&pool, one, x)]);
        let r = bound_on_box(e, &pool, &[(x, 0.0, 1.0)], &opts()).unwrap();

        let mut ev = IntervalEval::new(128);
        ev.bind(x, ArbBall::from_midpoint_radius(0.5, 0.5, 128));
        let iv = ev.eval(e, &pool).unwrap();
        let iv_width = iv.rad_f64() * 2.0;

        assert!(
            r.upper() - r.lower() < iv_width,
            "{:?} vs iv width {}",
            r,
            iv_width
        );
    }

    #[test]
    fn budget_exhaustion_is_reported_and_still_sound() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let e = pool.mul(vec![x, sub(&pool, one, x)]);
        let tight = BoundOptions {
            order: 1,
            prec: 64,
            tol: 1e-12,
            max_subdivisions: 1, // deliberately far too small
        };
        let r = bound_on_box(e, &pool, &[(x, 0.0, 1.0)], &tight).unwrap();
        assert!(r.budget_exhausted);
        // Still sound: true range [0, 0.25] must be contained.
        assert!(r.lower() <= 0.0 && r.upper() >= 0.25, "{:?}", r);
    }

    #[test]
    fn two_d_box_with_near_tangential_extremum() {
        // f(x,y) = (x - y)^2 + 1e-4, box straddling the near-tangential
        // minimum along x = y. True range is [1e-4, ~4 + 1e-4].
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let diff = sub(&pool, x, y);
        let sq = pool.pow(diff, pool.integer(2_i32));
        let eps = pool.rational(1_i32, 10000_i32);
        let e = pool.add(vec![sq, eps]);
        let r = bound_on_box(e, &pool, &[(x, -1.0, 1.0), (y, -1.0, 1.0)], &opts()).unwrap();
        assert!(r.lower() >= 0.0, "unsound: {:?}", r);
        assert!(r.lower() <= 1e-3, "{:?}", r);
        assert!(r.upper() >= 4.0 - 1e-3, "{:?}", r);
    }

    #[test]
    fn refuses_on_pole_that_resists_bisection() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(&pool, pool.integer(1_i32), x);
        let err = bound_on_box(e, &pool, &[(x, -1.0, 1.0)], &opts()).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-003");
    }

    #[test]
    fn bisects_away_from_a_boundary_domain_issue() {
        // log(x) on a box that only touches x=0 as an artefact of a coarse
        // initial box: [-0.1, 2] would refuse without subdivision help, but
        // here we start entirely inside the domain, so this is a control
        // case confirming ordinary boxes are unaffected by the pole logic.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("log", vec![x]);
        let r = bound_on_box(e, &pool, &[(x, 0.5, 2.0)], &opts()).unwrap();
        assert!(r.lower() <= 0.5_f64.ln() + 1e-6);
        assert!(r.upper() >= 2.0_f64.ln() - 1e-6);
    }

    // ── verified_integral ──────────────────────────────────────────────

    #[test]
    fn integral_of_x_squared() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.pow(x, pool.integer(2_i32));
        // ∫_0^1 x^2 dx = 1/3.
        let r = verified_integral(e, &pool, x, 0.0, 1.0, &iopts()).unwrap();
        assert!(r.lower() <= 1.0 / 3.0 && r.upper() >= 1.0 / 3.0, "{:?}", r);
        assert!(r.upper() - r.lower() < 1e-6, "{:?}", r);
    }

    #[test]
    fn integral_of_sin_over_full_period() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("sin", vec![x]);
        // ∫_0^{2π} sin(x) dx = 0.
        let two_pi = std::f64::consts::PI * 2.0;
        let r = verified_integral(e, &pool, x, 0.0, two_pi, &iopts()).unwrap();
        assert!(r.lower() <= 1e-4 && r.upper() >= -1e-4, "{:?}", r);
    }

    #[test]
    fn integral_of_exp() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("exp", vec![x]);
        // ∫_0^1 exp(x) dx = e - 1.
        let expected = std::f64::consts::E - 1.0;
        let r = verified_integral(e, &pool, x, 0.0, 1.0, &iopts()).unwrap();
        assert!(r.lower() <= expected && r.upper() >= expected, "{:?}", r);
        assert!(r.upper() - r.lower() < 1e-6, "{:?}", r);
    }

    #[test]
    fn integral_degenerate_interval_is_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("exp", vec![x]);
        let r = verified_integral(e, &pool, x, 1.0, 1.0, &iopts()).unwrap();
        assert_eq!(r.lower(), 0.0);
        assert_eq!(r.upper(), 0.0);
        assert_eq!(r.subdivisions, 0);
    }

    #[test]
    fn integral_refuses_on_singular_integrand() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(&pool, pool.integer(1_i32), pool.func("sqrt", vec![x]));
        let err = verified_integral(e, &pool, x, 0.0, 1.0, &iopts()).unwrap_err();
        // Domain violation (sqrt/recip touching zero) rather than a guess.
        assert!(matches!(
            crate::errors::AlkahestError::code(&err),
            "E-VALIDATED-003" | "E-VALIDATED-004"
        ));
    }

    // ── verified_integral: removable singularities ─────────────────────

    /// Assert that `r` really brackets `exact` and is no wider than `tol`.
    /// An enclosure that is merely *returned* is worth nothing.
    fn assert_brackets(r: &IntegralResult, exact: f64, tol: f64, label: &str) {
        assert!(
            r.lower() <= exact && exact <= r.upper(),
            "{label}: {exact} not in [{}, {}]",
            r.lower(),
            r.upper()
        );
        assert!(
            r.upper() - r.lower() <= tol,
            "{label}: enclosure [{}, {}] is wider than {tol}",
            r.lower(),
            r.upper()
        );
    }

    #[test]
    fn integral_of_log1p_over_x_is_pi_squared_over_twelve() {
        // ∫₀¹ ln(1+x)/x dx = π²/12.  The reciprocal's enclosure contains zero
        // at x = 0, but the singularity is removable, so the continuous
        // extension is integrated instead of refusing.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(
            &pool,
            pool.func("log", vec![pool.add(vec![pool.integer(1_i32), x])]),
            x,
        );
        let r = verified_integral(e, &pool, x, 0.0, 1.0, &iopts()).unwrap();
        let exact = std::f64::consts::PI * std::f64::consts::PI / 12.0;
        assert_brackets(&r, exact, 1e-6, "ln(1+x)/x");
    }

    #[test]
    fn integral_of_sinc_is_twice_si_of_one() {
        // ∫_{-1}^{1} sin(x)/x dx = 2·Si(1) = 1.892166140734366…, and the
        // removable point x = 0 is *interior*: it only becomes a sub-interval
        // endpoint after the first bisection.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(&pool, pool.func("sin", vec![x]), x);
        let r = verified_integral(e, &pool, x, -1.0, 1.0, &iopts()).unwrap();
        assert_brackets(&r, 1.892_166_140_734_366_4, 1e-6, "sin(x)/x");
    }

    #[test]
    fn integral_of_expm1_over_x() {
        // ∫₀¹ (exp(x)-1)/x dx = Σ_{n≥1} 1/(n·n!) = 1.317902151454403…
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let num = sub(&pool, pool.func("exp", vec![x]), pool.integer(1_i32));
        let e = div(&pool, num, x);
        let r = verified_integral(e, &pool, x, 0.0, 1.0, &iopts()).unwrap();
        assert_brackets(&r, 1.317_902_151_454_403_9, 1e-6, "(exp(x)-1)/x");
    }

    #[test]
    fn removable_extension_agrees_with_the_regular_quadrature_away_from_zero() {
        // Cross-check: ∫₀¹ ln(1+x)/x must equal ∫₀^{1/2} + ∫_{1/2}^1, and the
        // second piece needs no special handling at all. If the removable
        // branch were biased, the two would disagree.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(
            &pool,
            pool.func("log", vec![pool.add(vec![pool.integer(1_i32), x])]),
            x,
        );
        let whole = verified_integral(e, &pool, x, 0.0, 1.0, &iopts()).unwrap();
        let left = verified_integral(e, &pool, x, 0.0, 0.5, &iopts()).unwrap();
        let right = verified_integral(e, &pool, x, 0.5, 1.0, &iopts()).unwrap();
        assert!(
            whole.lower() <= left.upper() + right.upper()
                && whole.upper() >= left.lower() + right.lower(),
            "split disagrees: whole [{}, {}] vs pieces [{}, {}] + [{}, {}]",
            whole.lower(),
            whole.upper(),
            left.lower(),
            left.upper(),
            right.lower(),
            right.upper()
        );
    }

    #[test]
    fn genuine_pole_is_not_mistaken_for_a_removable_one() {
        // 1/x on [-1,1]: the numerator does not vanish, so the removable
        // branch must decline and the call must still refuse.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(&pool, pool.integer(1_i32), x);
        let err = verified_integral(e, &pool, x, -1.0, 1.0, &iopts()).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-003");
    }

    #[test]
    fn double_pole_with_a_simple_numerator_zero_is_refused() {
        // sin(x)/x² ~ 1/x near 0: the numerator vanishes to order 1 but the
        // denominator to order 2, so the singularity is *not* removable and
        // the integral does not converge. D' = 2x vanishes at 0, which is
        // exactly the check that stops the L'Hôpital argument here.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(
            &pool,
            pool.func("sin", vec![x]),
            pool.pow(x, pool.integer(2_i32)),
        );
        let err = verified_integral(e, &pool, x, -1.0, 1.0, &iopts()).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-003");
    }

    #[test]
    fn integrable_endpoint_singularity_is_refused_with_a_message_that_says_so() {
        // ∫₀¹ -log x dx = 1 exists, but no rigorous enclosure of the
        // *integrand* does. Option (c) of the issue: the refusal must not read
        // as "the integral does not exist".
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.mul(vec![pool.integer(-1_i32), pool.func("log", vec![x])]);
        let err = verified_integral(e, &pool, x, 0.0, 1.0, &iopts()).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-003");
        let msg = err.to_string();
        assert!(msg.contains("the left endpoint"), "{msg}");
        assert!(msg.contains("integrable singularity"), "{msg}");
    }

    #[test]
    fn interior_singularity_is_reported_as_interior() {
        // 1/(x - 1/2) on [0,1]: the refusal should localise the pole away from
        // both endpoints.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(
            &pool,
            pool.integer(1_i32),
            sub(&pool, x, pool.rational(1_i32, 2_i32)),
        );
        let err = verified_integral(e, &pool, x, 0.0, 1.0, &iopts()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("an interior point"), "{msg}");
    }

    #[test]
    fn integral_refuses_on_infinite_bounds() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = x;
        let err = verified_integral(e, &pool, x, 0.0, f64::INFINITY, &iopts()).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-005");
    }

    #[test]
    fn integral_refuses_when_a_greater_than_b() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let err = verified_integral(x, &pool, x, 1.0, 0.0, &iopts()).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-005");
    }

    // ── verified_no_roots / verified_sign ─────────────────────────────

    #[test]
    fn no_roots_verified_true_for_shifted_square() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        // (x - 5)^2 + 1: never zero.
        let sq = pool.pow(sub(&pool, x, pool.integer(5_i32)), pool.integer(2_i32));
        let e = pool.add(vec![sq, one]);
        let v = verified_no_roots(e, &pool, &[(x, -10.0, 10.0)], &opts()).unwrap();
        assert_eq!(v, Verdict::True);
    }

    #[test]
    fn no_roots_verified_false_via_sign_change() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // x - 0.5 changes sign on [0,1]: root at x=0.5 certified.
        let e = sub(&pool, x, pool.rational(1_i32, 2_i32));
        let v = verified_no_roots(e, &pool, &[(x, 0.0, 1.0)], &opts()).unwrap();
        assert_eq!(v, Verdict::False);
    }

    #[test]
    fn no_roots_false_across_a_convex_multivariate_box() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // x - y on [-1,1]²: it is +2 at (1,-1) and -2 at (-1,1), and the
        // segment between them stays in the box, so a root is certified. This
        // is the honest verdict — the whole plane x = y of roots is inside the
        // box — and it is why the witness search does not need the two points
        // to be endpoints of a 1-D interval.
        let e = sub(&pool, x, y);
        let v = verified_no_roots(e, &pool, &[(x, -1.0, 1.0), (y, -1.0, 1.0)], &opts()).unwrap();
        assert_eq!(v, Verdict::False);
    }

    #[test]
    fn no_roots_undecided_for_a_multivariate_tangential_zero() {
        // (x-1/2)² + (y-1/2)² is zero at exactly one point of [0,1]² and
        // positive everywhere else, so no sign-change witness can exist. The
        // enclosure straddles zero, so `True` is unavailable too: `Undecided`
        // is the only honest answer and must not collapse either way.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let half = pool.rational(1_i32, 2_i32);
        let dx = sub(&pool, x, half);
        let dy = sub(&pool, y, half);
        let e = pool.add(vec![pool.mul(vec![dx, dx]), pool.mul(vec![dy, dy])]);
        let cheap = BoundOptions {
            max_subdivisions: 64,
            ..opts()
        };
        let v = verified_no_roots(e, &pool, &[(x, 0.0, 1.0), (y, 0.0, 1.0)], &cheap).unwrap();
        assert_eq!(v, Verdict::Undecided);
    }

    /// `x² - 2` on every box from the issue-13 table, plus the product that
    /// hides the same two roots behind a strictly positive factor. Before the
    /// witness search only the boxes with a *top-level* endpoint sign change
    /// could be settled; an even number of roots defeated the rest.
    #[test]
    fn no_roots_false_regardless_of_the_root_count_in_the_box() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = sub(&pool, x2, pool.integer(2_i32));
        // (x²-2)(x²+1) — same two roots, second factor never vanishes.
        let g = pool.mul(vec![f, pool.add(vec![x2, pool.integer(1_i32)])]);

        for (expr, lo, hi, label) in [
            (f, -2.0, 0.0, "x^2-2 on [-2,0] (1 root)"),
            (f, 0.0, 2.0, "x^2-2 on [0,2] (1 root)"),
            (f, 1.3, 1.5, "x^2-2 on [1.3,1.5] (1 root)"),
            (f, -2.0, 2.0, "x^2-2 on [-2,2] (2 roots)"),
            (f, -10.0, 10.0, "x^2-2 on [-10,10] (2 roots)"),
            (g, -2.0, 2.0, "(x^2-2)(x^2+1) on [-2,2] (2 roots)"),
        ] {
            let v = verified_no_roots(expr, &pool, &[(x, lo, hi)], &opts()).unwrap();
            assert_eq!(v, Verdict::False, "{label}");
        }
    }

    #[test]
    fn no_roots_stays_true_where_it_was_true() {
        // The witness search must never be reached when the enclosure already
        // proves absence — these are the `True` rows that have to stay `True`.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        for (expr, lo, hi) in [
            (pool.add(vec![x2, pool.integer(1_i32)]), -10.0, 10.0),
            (pool.add(vec![x2, pool.integer(2_i32)]), -2.0, 2.0),
            (pool.func("exp", vec![x]), -5.0, 5.0),
        ] {
            let v = verified_no_roots(expr, &pool, &[(x, lo, hi)], &opts()).unwrap();
            assert_eq!(v, Verdict::True);
        }
    }

    #[test]
    fn no_roots_undecided_for_a_double_root_that_cannot_be_witnessed() {
        // (x-1)² has a genuine root at x = 1 inside [0,2], but it never
        // changes sign, so no IVT witness exists and none may be invented:
        // turning this into `False` would be a lucky guess, not a proof.
        // `True` is also unavailable (the enclosure contains zero).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let d = sub(&pool, x, pool.integer(1_i32));
        let e = pool.mul(vec![d, d]);
        let v = verified_no_roots(e, &pool, &[(x, 0.0, 2.0)], &opts()).unwrap();
        assert_eq!(v, Verdict::Undecided);
    }

    #[test]
    fn no_roots_undecided_for_a_quartic_double_root() {
        // (x²-1)² — two double roots at ±1 in [-2,2], again with no sign
        // change anywhere.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let d = sub(&pool, pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32));
        let e = pool.mul(vec![d, d]);
        let v = verified_no_roots(e, &pool, &[(x, -2.0, 2.0)], &opts()).unwrap();
        assert_eq!(v, Verdict::Undecided);
    }

    #[test]
    fn no_roots_near_tangential_case_stays_true() {
        // (x-1)^2 + 1e-6 touches close to zero but never reaches it; a
        // narrowing bug would likely misreport this.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sq = pool.pow(sub(&pool, x, pool.integer(1_i32)), pool.integer(2_i32));
        let eps = pool.rational(1_i32, 1_000_000_i32);
        let e = pool.add(vec![sq, eps]);
        let tight = BoundOptions {
            order: 8,
            prec: 128,
            tol: 1e-10,
            max_subdivisions: 8192,
        };
        let v = verified_no_roots(e, &pool, &[(x, 0.0, 2.0)], &tight).unwrap();
        assert_eq!(v, Verdict::True);
    }

    #[test]
    fn sign_positive_verified_true() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("exp", vec![x]);
        let v = verified_sign(
            e,
            &pool,
            &[(x, -5.0, 5.0)],
            SignPredicate::Positive,
            &opts(),
        )
        .unwrap();
        assert_eq!(v, Verdict::True);
    }

    #[test]
    fn sign_positive_verified_false() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = x; // takes negative values on [-1,1]
        let v = verified_sign(
            e,
            &pool,
            &[(x, -1.0, 1.0)],
            SignPredicate::Positive,
            &opts(),
        )
        .unwrap();
        assert_eq!(v, Verdict::False);
    }

    #[test]
    fn sign_false_is_certified_by_a_point_witness() {
        // `x - 1/2 > 0` on [0,1] is plainly false (x = 0 gives -1/2), even
        // though the range enclosure straddles zero and so cannot settle it.
        // A universally quantified claim is disproved by one point.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = sub(&pool, x, pool.rational(1_i32, 2_i32));
        let v =
            verified_sign(e, &pool, &[(x, 0.0, 1.0)], SignPredicate::Positive, &opts()).unwrap();
        assert_eq!(v, Verdict::False);
    }

    #[test]
    fn sign_undecided_when_neither_can_be_established() {
        // `(x - 1/3)^2 - 1e-12 >= 0` on [-1,1] fails only in a sliver around
        // x = 1/3, which is neither an endpoint nor the centre, so no sample
        // hits it; with a tight budget the enclosure still straddles zero.
        // Neither a proof nor a witness is available, so the verdict must stay
        // Undecided rather than collapse either way.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let shifted = sub(&pool, x, pool.rational(1_i32, 3_i32));
        let tiny = pool.rational(1_i32, 1_000_000_000_000_i64);
        let e = sub(&pool, pool.mul(vec![shifted, shifted]), tiny);
        let cheap = BoundOptions {
            order: 4,
            prec: 128,
            tol: 1e-9,
            max_subdivisions: 2,
        };
        let v = verified_sign(
            e,
            &pool,
            &[(x, -1.0, 1.0)],
            SignPredicate::NonNegative,
            &cheap,
        )
        .unwrap();
        assert_eq!(v, Verdict::Undecided);
    }

    #[test]
    fn dense_sampling_cross_check_polynomial() {
        // Cross-check bound_on_box against dense sampling: every sampled
        // value must be contained in the returned enclosure.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = sub(
            &pool,
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(3_i32), x]),
        ); // x^3 - 3x
        let r = bound_on_box(e, &pool, &[(x, -2.5, 2.5)], &opts()).unwrap();
        for i in 0..=500 {
            let t = -2.5 + 5.0 * (i as f64) / 500.0;
            let v = t.powi(3) - 3.0 * t;
            assert!(
                v >= r.lower() - 1e-6 && v <= r.upper() + 1e-6,
                "x={t} f={v} escaped {:?}",
                r
            );
        }
    }
}
