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
use super::{
    contains_zero, from_bounds, from_float, is_finite, lb, mag, ub, width, ValidatedError,
};
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

    let (lo_bound, used_lo, exhausted_lo) = extremum_search(
        expr,
        pool,
        boxes0,
        opts,
        Extremum::Min,
        SearchGoal::Tolerance,
    )?;
    let (hi_bound, used_hi, exhausted_hi) = extremum_search(
        expr,
        pool,
        boxes0,
        opts,
        Extremum::Max,
        SearchGoal::Tolerance,
    )?;

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

/// What makes an [`extremum_search`] pass stop early.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchGoal {
    /// Stop once the extremum is pinned down to within `opts.tol`.
    Tolerance,
    /// Stop once the running rigorous bound is proven strictly positive in the
    /// signed view — i.e. `min f > 0` for [`Extremum::Min`], `max f < 0` for
    /// [`Extremum::Max`].
    ///
    /// `tol` is the wrong stopping rule for a sign question, because it is an
    /// **absolute** width. On an expression whose extremum is closer to zero
    /// than `tol`, the tolerance test fires while the enclosure still straddles
    /// zero, and the predicate can only answer `Undecided` — the search stopped
    /// on a criterion unrelated to the question being asked. Under this goal
    /// the pass instead refines until the sign is settled, or until the work
    /// budget or the width floor stops it.
    DecideSign,
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
    goal: SearchGoal,
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
    // Smallest key among boxes that have been bisected down to `floor` and so
    // can never be refined again. Such a box must leave `active`: pushing it
    // back would make it the argmin again on the very next iteration with
    // nothing changed, and the loop would spin forever without consuming any
    // budget. Only the key is kept, which is all the final bound needs.
    let mut retired: Option<Float> = None;
    let mut best_ub: Option<Float> = None;
    let mut subdivisions = 0usize;
    let mut exhausted = false;

    // Running minimum of two optional keys.
    let keep_min = |cur: Option<Float>, k: Float| -> Option<Float> {
        Some(match cur {
            Some(c) if c <= k => c,
            _ => k,
        })
    };

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

        // Rigorous lower bound on the signed extremum as things stand: `key` is
        // the smallest key still active, and no retired box holds anything
        // smaller than `retired`.
        let overall = match &retired {
            Some(r) if r < &key => r.clone(),
            _ => key.clone(),
        };

        if goal == SearchGoal::DecideSign && overall > 0 {
            // `min f > 0` (or, in the signed view of a Max pass, `max f < 0`).
            // The sign question is settled and no amount of further refinement
            // can unsettle it.
            active.push((key, b));
            break;
        }

        // Prune: this box cannot contain the extremum.
        if let Some(ub_best) = &best_ub {
            if &key > ub_best {
                continue;
            }
            // Converged: the uncertainty in the extremum is within tol.
            if goal == SearchGoal::Tolerance {
                let gap = Float::with_val(prec, ub_best - &overall);
                if gap <= tol_f {
                    active.push((key, b));
                    break;
                }
            }
        }

        if subdivisions >= opts.max_subdivisions {
            exhausted = true;
            active.push((key, b));
            break;
        }
        if max_dim_width(&b, prec) <= floor {
            // Cannot refine further: retire it, so the loop makes progress.
            retired = keep_min(retired, key);
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

    // Every point of the original box lies in an active box, a retired box, or
    // a pruned one — and a pruned box provably cannot beat `best_ub`. So the
    // smallest key over active ∪ retired is a rigorous bound on the signed
    // extremum, whenever the loop happened to stop.
    let mut bound = active
        .iter()
        .map(|(k, _)| k.clone())
        .chain(retired.clone())
        .fold(None::<Float>, keep_min)
        .or_else(|| best_ub.clone())
        .ok_or_else(|| ValidatedError::InvalidInput {
            what: "no enclosure was produced".into(),
        })?;

    // Retiring every remaining box means the search ran out of width, not that
    // it converged — the old `stuck` case, reported the same way.
    if active.is_empty() && retired.is_some() {
        exhausted = true;
    }

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
    vanishes_exactly_at(expr, pool, &[(var, at.clone(), at.clone())])
}

/// `expr` with **every** variable of the degenerate box `point` replaced by its
/// exact rational coordinate, simplified.
///
/// Returns `true` only when the result is the literal integer (or rational)
/// zero — a symbolic proof that `expr` vanishes at that point, of the same kind
/// [`vanishes_exactly`] provides in one dimension. A `Float` coordinate is a
/// binary rational and converts exactly, so nothing is rounded on the way in;
/// a coordinate whose interval is not degenerate names a set rather than a
/// point and is declined.
fn vanishes_exactly_at(expr: ExprId, pool: &ExprPool, point: &[FBox]) -> bool {
    let mut mapping = HashMap::new();
    for (var, lo, hi) in point {
        if lo != hi {
            return false;
        }
        let Some(rational) = lo.to_rational() else {
            return false;
        };
        let (n, d) = rational.into_numer_denom();
        let value = if d == 1 {
            pool.integer(n)
        } else {
            pool.rational(n, d)
        };
        mapping.insert(*var, value);
    }
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
/// value theorem instead: `N(x)/D(x) = N'(ξ)/D'(ξ)` for some `ξ` in the
/// sub-interval, so an enclosure `R` of `N'/D'` there — which is perfectly
/// regular — gives `∫_J N/D dx ∈ |J| · R`. The vanishing of `N` and `D`
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

/// True only for the degenerate enclosure `[0, 0]`.
///
/// An enclosure is a superset of the value it describes, so `[0, 0]` — and
/// *only* `[0, 0]` — pins that value to zero. An enclosure that merely
/// *contains* zero proves nothing about whether the value is zero, which is why
/// this is a two-sided test on the outward-rounded [`lb`]/[`ub`] rather than
/// [`contains_zero`].
fn is_exact_zero(b: &ArbBall) -> bool {
    lb(b) == 0 && ub(b) == 0
}

/// What a rigorous evaluation at a single point of the box established there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PointOutcome {
    /// The value at that point is **proven** to be exactly zero, so the point
    /// is a root — and being a point of the (closed) box, it is a root *in* the
    /// box.
    Zero,
    /// The value is proven strictly positive (`true`) or strictly negative
    /// (`false`).
    Sign(bool),
    /// Nothing was proven: the enclosure straddles zero without pinning it, or
    /// the evaluation refused.
    Unknown,
}

/// Rigorously evaluate `expr` at one point of the box and report what that
/// proves.
///
/// The point is passed as a degenerate box, so the Taylor model collapses to a
/// ball evaluation and the sign test goes through [`lb`]/[`ub`], which round
/// outward. A reported sign is therefore a proof, never a rounding artefact.
///
/// [`PointOutcome::Zero`] is likewise only ever a proof, by one of two routes:
///
/// * the enclosure is the degenerate interval `[0, 0]`, which forces the value
///   it encloses to be zero — this settles the exactly-representable cases
///   (`x` at `0`, `x − 1` at `1`, `(x − 1)²` at `1`, `sin x` at `0`), where the
///   arithmetic is exact and no rounding term is ever added because every
///   intermediate midpoint is zero; or
/// * substituting the point's exact rational coordinates and simplifying lands
///   on the literal `0` — the same symbolic argument the removable-singularity
///   path already relies on, and the only kind of argument that can prove a
///   transcendental combination such as `exp(x) − 1` vanishes at `0`, where the
///   enclosure is `[-ε, ε]` and can never be tightened to a point.
///
/// The symbolic route is cross-checked against the enclosure exactly as
/// [`enclosure_admits_zero`] does elsewhere: a `simplify` that claims an exact
/// zero which outward-rounded ball arithmetic contradicts would be a simplifier
/// bug, and the safe response is to prove nothing rather than to certify on top
/// of it. That check is free here — reaching it means [`determined_sign`]
/// already declined, which is precisely `contains_zero`.
///
/// `symbolic` selects whether the second route is attempted. It substitutes and
/// simplifies, which interns a fresh copy of `expr` in the pool, so callers
/// that run this once per bisection leave it off; the box's own distinguished
/// points — endpoints, corners, centre — are where a root "sitting on the box"
/// can be, and they are a fixed, small set.
fn point_outcome(
    expr: ExprId,
    pool: &ExprPool,
    point: &[FBox],
    order: usize,
    prec: u32,
    symbolic: bool,
) -> PointOutcome {
    let Ok(r) = taylor_range(expr, pool, point, order, prec) else {
        return PointOutcome::Unknown;
    };
    if let Some(s) = determined_sign(&r) {
        return PointOutcome::Sign(s);
    }
    if is_exact_zero(&r) {
        return PointOutcome::Zero;
    }
    if symbolic && contains_zero(&r) && vanishes_exactly_at(expr, pool, point) {
        return PointOutcome::Zero;
    }
    PointOutcome::Unknown
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

/// Search for a proof that `expr` has a root in the box.
///
/// Two independent proofs are looked for, and either one suffices:
///
/// 1. **A point proven to be a root.** The box is closed, so a point of it at
///    which `expr` is *proven* zero — by a degenerate `[0, 0]` enclosure or by
///    exact symbolic substitution, see [`point_outcome`] — is a root in the
///    box, full stop. No continuity argument, no sign change, and no relation
///    between that point and any other is needed. This is what settles a root
///    sitting exactly on an endpoint (`x` on `[0, 1]`), where the function has
///    one sign throughout the interior and the search below provably cannot
///    find a witness pair, and what settles a root of even multiplicity that a
///    sign change cannot see either (`(x − 1)²` on `[0, 1]`).
/// 2. **A sign change**, described next.
///
/// # Why the sign-change route is a proof
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
        match point_outcome(expr, pool, &point, order, prec, true) {
            PointOutcome::Zero => return true,
            PointOutcome::Sign(s) => {
                w.record(Some(s));
                if w.both() {
                    return true;
                }
            }
            PointOutcome::Unknown => {}
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
            // An enclosure of the *range* over a whole sub-box that is the
            // degenerate `[0, 0]` proves `expr` vanishes identically there, and
            // the sub-box is non-empty — so every one of its points is a root.
            if is_exact_zero(&r) {
                return true;
            }
        }

        let centre: Vec<FBox> = b
            .iter()
            .map(|(v, lo, hi)| {
                let m = midpoint(lo, hi, prec);
                (*v, m.clone(), m)
            })
            .collect();
        match point_outcome(expr, pool, &centre, order, prec, false) {
            PointOutcome::Zero => return true,
            PointOutcome::Sign(s) => {
                w.record(Some(s));
                if w.both() {
                    return true;
                }
            }
            PointOutcome::Unknown => {}
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
/// - [`Verdict::False`] when a root is *certified to exist* in the box, by
///   either of two independent arguments (see [`root_exists_witness`]):
///   - **a point of the box proven to be a root** — its value pinned to zero by
///     a degenerate `[0, 0]` enclosure or by exact symbolic substitution. The
///     box is closed, so this covers a root sitting exactly on an endpoint
///     (`x` on `[0, 1]`, `x − 1` on `[0, 1]`) and a root of even multiplicity
///     that produces no sign change at all (`(x − 1)²` on `[0, 1]`); or
///   - **a sign change**: the box is proven free of poles/branch cuts (the
///     full-box enclosure succeeded, so `expr` is continuous on the box) *and*
///     two points of the box are found at which `expr` is rigorously proven to
///     have opposite signs — a root then exists by the intermediate value
///     theorem along the segment joining them, which stays in the box because
///     a box is convex. The points are looked for by subdividing the box, so
///     an even number of roots does not defeat the test: `x² − 2` on `[-2, 2]`
///     has two roots and two positive endpoints, and is settled at the first
///     bisection.
/// - [`Verdict::Undecided`] otherwise: the enclosure straddles zero, no point
///   of the box was *proven* to be a root, and the search found no pair of
///   opposite-signed points within the budget. An enclosure that merely
///   *contains* zero is never enough — a value known only to lie in `[-ε, ε]`
///   is not a proven root — so a function that grazes zero to within the
///   working precision without provably touching it stays `Undecided`, and it
///   is never collapsed into either of the other two verdicts.
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

    // The search above stopped on `tol`, which is an absolute width and so says
    // nothing about the sign. Re-run it with the sign as the stopping rule: it
    // keeps refining while the running bound still straddles zero, which is
    // what decides an inequality whose margin is narrower than `tol`.
    let strictly_signed = match predicate {
        // `min f > 0` proves `f > 0` everywhere, hence also `f >= 0`.
        SignPredicate::Positive | SignPredicate::NonNegative => {
            sign_targeted_bound(expr, pool, boxes, opts, Extremum::Min)? > 0
        }
        SignPredicate::Negative | SignPredicate::NonPositive => {
            sign_targeted_bound(expr, pool, boxes, opts, Extremum::Max)? < 0
        }
    };
    if strictly_signed {
        return Ok(Verdict::True);
    }

    // Still undecided, which is what a margin that *vanishes* at an endpoint
    // always looks like to a subdivision search. Try the series argument, which
    // is the only one of the two that can reach a tight endpoint at all.
    if let Some(v) = endpoint_series_verdict(expr, pool, boxes, predicate, opts) {
        return Ok(v);
    }

    Ok(Verdict::Undecided)
}

/// Rigorous one-sided bound computed with [`SearchGoal::DecideSign`]: a lower
/// bound on `min f` for [`Extremum::Min`], an upper bound on `max f` for
/// [`Extremum::Max`].
fn sign_targeted_bound(
    expr: ExprId,
    pool: &ExprPool,
    boxes: &[(ExprId, f64, f64)],
    opts: &BoundOptions,
    which: Extremum,
) -> Result<Float> {
    let prec = opts.prec;
    let boxes0: Vec<FBox> = boxes
        .iter()
        .map(|(v, lo, hi)| (*v, Float::with_val(prec, lo), Float::with_val(prec, hi)))
        .collect();
    let (bound, _, _) = extremum_search(expr, pool, &boxes0, opts, which, SearchGoal::DecideSign)?;
    Ok(bound)
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

// ---------------------------------------------------------------------------
// Endpoint series proof — inequalities that are tight where the box ends
// ---------------------------------------------------------------------------

/// Highest derivative order the endpoint expansion will search for the first
/// coefficient that is not proven to vanish.
const MAX_VANISHING_ORDER: usize = 16;

/// Taylor terms kept past the leading one before the Lagrange remainder takes
/// over. Three gives the halving loop room to work while keeping the number of
/// symbolic derivatives — and so their size — small.
const SERIES_TAIL_TERMS: usize = 3;

/// Halvings of the candidate sub-interval tried while looking for one on which
/// the series argument closes.
const SERIES_HALVINGS: u32 = 240;

/// A rigorously bounded Taylor expansion of `g` at one endpoint of the box.
///
/// Write `p` for the endpoint and `s = ±1` for the direction pointing into the
/// box. This describes `h(t) = g(p + s·t)` on `t ∈ [0, span]`:
///
/// ```text
/// h(t) = Σ_{k<m} c_k t^k + R(t),          |R(t)| ≤ rem · t^m
/// ```
///
/// where `c_0 … c_{j-1}` are **proven** to be exactly zero, `cj` encloses
/// `c_j`, and `tail[k-j-1]` encloses `|c_k|` for `j < k < m`.
struct EndpointSeries {
    /// Index of the first coefficient not proven to be exactly zero.
    j: usize,
    /// Truncation order.
    m: usize,
    /// Enclosure of the leading coefficient `c_j`.
    cj: ArbBall,
    /// Enclosures of `|c_{j+1}| … |c_{m-1}|`.
    tail: Vec<ArbBall>,
    /// Enclosure of the Lagrange constant `sup|g^{(m)}| / m!` over the whole
    /// candidate interval.
    rem: ArbBall,
    /// Width of the box, rounded **down**: the largest collar the certificate
    /// covers, and the starting point of the halving sequence.
    span: Float,
}

/// Build the endpoint expansion of `g` at `p`, looking a distance `span` into
/// the box in direction `inward` (`+1` for a left endpoint, `-1` for a right
/// one).
///
/// # Why the result is rigorous
///
/// Two separate facts are established, by two different mechanisms:
///
/// * **The vanishing coefficients really are zero.** `c_k = s^k g^{(k)}(p)/k!`
///   is accepted as zero only when [`vanishes_exactly`] — substitution followed
///   by `simplify`, checked to land on the literal integer `0` — says so. A
///   numeric enclosure can never prove a value is zero, and none is used for
///   that here. As elsewhere in this module the symbolic verdict is
///   cross-checked against ball arithmetic, and a disagreement (which would
///   mean a simplifier bug) abandons the expansion rather than building on it.
/// * **The remainder really is bounded.** `bound_on_fboxes` is run on every
///   derivative `g, g', …, g^{(m)}` over the closed candidate interval. Those
///   calls succeed only if each is analytic throughout — [`super::taylor`]
///   refuses otherwise — which is exactly the hypothesis of Taylor's theorem
///   with Lagrange remainder on that interval. `rem` is then an outward-rounded
///   bound on `sup|g^{(m)}|/m!` there, so `|R(t)| ≤ rem·t^m` holds for every
///   `t` in the interval, and a fortiori on any sub-interval `[0, δ]` of it.
///   This is a *proven* remainder, not a truncation assumed to be small.
///
/// Returns `None` — never an error — whenever any step declines. The series
/// argument is an extra attempt layered on top of the subdivision search, so
/// its failure must leave the caller's verdict at `Undecided`, not turn it into
/// a refusal.
fn expand_at_endpoint(
    g: ExprId,
    pool: &ExprPool,
    var: ExprId,
    ilo: &Float,
    ihi: &Float,
    inward: i32,
    opts: &BoundOptions,
) -> Option<EndpointSeries> {
    let prec = opts.prec;
    // Expand at whichever end `inward` points away from.
    let p = if inward > 0 { ilo } else { ihi };
    // Rounded **down**, so `p ± span` can never leave the box and every `δ` the
    // halving loop considers names a collar the analyticity certificate below
    // actually covers.
    let span = Float::with_val_round(prec, ihi - ilo, Round::Down).0;
    if span <= 0 {
        return None;
    }

    // Successive symbolic derivatives, simplified at each step to keep them
    // from growing.
    let mut derivs: Vec<ExprId> = vec![g];
    let extend = |derivs: &mut Vec<ExprId>| -> bool {
        let last = *derivs.last().expect("derivs is never empty");
        match diff(last, var, pool) {
            Ok(d) => {
                derivs.push(simplify(d.value, pool).value);
                true
            }
            Err(_) => false,
        }
    };

    // Locate `j`, the first coefficient not proven to vanish at `p`.
    let mut found = None;
    for k in 0..=MAX_VANISHING_ORDER {
        if k > 0 && !extend(&mut derivs) {
            return None;
        }
        if vanishes_exactly(derivs[k], pool, var, p) {
            // `simplify` claims an exact zero; ball arithmetic must agree.
            if !enclosure_admits_zero(derivs[k], pool, var, p, opts.order, prec) {
                return None;
            }
            continue;
        }
        found = Some(k);
        break;
    }
    let j = found?;

    let m = j + 1 + SERIES_TAIL_TERMS;
    while derivs.len() <= m {
        if !extend(&mut derivs) {
            return None;
        }
    }

    // Analyticity certificate for Taylor's theorem, plus the sup bound feeding
    // the Lagrange remainder. Both are taken over the *whole* box, so they hold
    // on every collar the halving loop can propose.
    let interval = vec![(var, ilo.clone(), ihi.clone())];
    let mut sup_m = None;
    for (k, &d) in derivs.iter().enumerate().take(m + 1) {
        let r = bound_on_fboxes(d, pool, &interval, opts).ok()?;
        if !is_finite(r.enclosure()) {
            return None;
        }
        if k == m {
            sup_m = Some(mag(r.enclosure()));
        }
    }

    // Exact factorials (20! < 2^62, so these are exact at working precision).
    let mut fact = vec![Float::with_val(prec, 1)];
    for k in 1..=m {
        let next = Float::with_val(prec, &fact[k - 1] * k as u32);
        fact.push(next);
    }

    let point = vec![(var, p.clone(), p.clone())];
    let at_point = |k: usize| taylor_range(derivs[k], pool, &point, opts.order, prec).ok();

    // c_j = s^j · g^{(j)}(p) / j!
    let mut cj = (at_point(j)? / from_float(&fact[j], prec))?;
    if inward < 0 && j % 2 == 1 {
        cj = -cj;
    }
    if !is_finite(&cj) {
        return None;
    }

    // |c_k| for j < k < m — the sign is irrelevant, so `s` does not enter.
    let mut tail = Vec::with_capacity(m - j - 1);
    for (k, fk) in fact.iter().enumerate().take(m).skip(j + 1) {
        let b = (at_point(k)?.abs_ball() / from_float(fk, prec))?;
        if !is_finite(&b) {
            return None;
        }
        tail.push(b);
    }

    let rem = (from_float(&sup_m?, prec) / from_float(&fact[m], prec))?;
    if !is_finite(&rem) {
        return None;
    }

    Some(EndpointSeries {
        j,
        m,
        cj,
        tail,
        rem,
        span,
    })
}

impl EndpointSeries {
    /// Upper bound on the tail `Σ_{j<k<m} |c_k| δ^{k-j} + rem · δ^{m-j}`.
    ///
    /// Every term carries a non-negative power of `δ ≥ 0`, and the whole sum is
    /// accumulated in ball arithmetic, so `ub` of the result dominates the true
    /// tail for every `t ∈ [0, δ]`.
    fn tail_bound(&self, delta: &Float, prec: u32) -> Float {
        let d = from_float(delta, prec);
        let mut acc = ArbBall::from_f64(0.0, prec);
        for (i, c) in self.tail.iter().enumerate() {
            acc = acc + c.clone() * d.powi((i + 1) as i64);
        }
        acc = acc + self.rem.clone() * d.powi((self.m - self.j) as i64);
        ub(&acc)
    }

    /// The largest `δ` of the halving sequence starting at `span` on which
    /// `h(t)` is proven to keep the sign of `c_j` throughout `t ∈ [0, δ]`,
    /// together with that sign (`true` for positive).
    ///
    /// # The argument
    ///
    /// With `c_0 … c_{j-1}` proven zero,
    ///
    /// ```text
    /// h(t) = c_j t^j + Σ_{j<k<m} c_k t^k + R(t)
    ///      ≥ t^j · [ c_j − Σ_{j<k<m} |c_k| t^{k-j} − rem · t^{m-j} ]
    ///      ≥ t^j · [ c_j − tail_bound(δ) ]                for t ∈ [0, δ]
    /// ```
    ///
    /// using `t ≥ 0` twice: once to factor `t^j ≥ 0` out without flipping the
    /// inequality, and once for `t^{k-j} ≤ δ^{k-j}` (every exponent there is
    /// `≥ 1`). So a proven `c_j > tail_bound(δ)` gives `h ≥ 0` on `[0, δ]`, and
    /// symmetrically a proven `c_j < −tail_bound(δ)` gives `h ≤ 0` there, with
    /// `h < 0` for `t > 0`.
    fn reach(&self, prec: u32) -> Option<(Float, bool)> {
        let (lo, hi) = (lb(&self.cj), ub(&self.cj));
        // Only a *proven* sign for the leading coefficient is usable.
        let (positive, margin) = if lo > 0 {
            (true, lo)
        } else if hi < 0 {
            (false, Float::with_val(prec, -hi))
        } else {
            return None;
        };

        let mut delta = self.span.clone();
        for _ in 0..SERIES_HALVINGS {
            if margin > self.tail_bound(&delta, prec) {
                return Some((delta, positive));
            }
            delta = Float::with_val(prec, &delta / 2u32);
            if delta <= 0 {
                break;
            }
        }
        None
    }
}

/// Try to settle a sign predicate whose margin vanishes at an endpoint of the
/// box, by splitting it into series-proved collars at the ends and an ordinary
/// branch-and-bound in the middle.
///
/// Subdivision alone provably cannot do this: where the margin goes to zero,
/// every enclosure of the range straddles zero no matter how fine the boxes
/// get, so the honest verdict from that machinery is always `Undecided`. The
/// standard remedy, and the one used here, is a truncated Taylor expansion at
/// the endpoint with a rigorous remainder — see [`expand_at_endpoint`] for why
/// the remainder is proven rather than assumed, and [`EndpointSeries::reach`]
/// for the positivity argument.
///
/// # Composition
///
/// The three pieces are `[a, a+δₗ]`, `[a+δₗ, b−δᵣ]` and `[b−δᵣ, b]`. They are
/// closed and share their endpoints, so their union is exactly `[a, b]` with no
/// gap — in particular the join points themselves are covered twice, never
/// zero times. A collar is only claimed when its own expansion closed, and the
/// middle is only claimed when the subdivision search proves a strict sign
/// there; `True` is returned only if every piece that exists is proven. If the
/// collars already meet or overlap there is no middle piece to prove.
///
/// Restricted to one dimension, since the expansion is in a single variable.
fn endpoint_series_verdict(
    expr: ExprId,
    pool: &ExprPool,
    boxes: &[(ExprId, f64, f64)],
    predicate: SignPredicate,
    opts: &BoundOptions,
) -> Option<Verdict> {
    if boxes.len() != 1 {
        return None;
    }
    let (var, lo, hi) = boxes[0];
    if lo >= hi || !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let prec = opts.prec;

    // Reduce every predicate to a statement about `g ≥ 0` / `g > 0`.
    let g = match predicate {
        SignPredicate::NonNegative | SignPredicate::Positive => expr,
        SignPredicate::NonPositive | SignPredicate::Negative => {
            pool.mul(vec![pool.integer(-1_i32), expr])
        }
    };
    let strict = matches!(predicate, SignPredicate::Positive | SignPredicate::Negative);

    let a = Float::with_val(prec, lo);
    let b = Float::with_val(prec, hi);

    let left = expand_at_endpoint(g, pool, var, &a, &b, 1, opts);
    let right = expand_at_endpoint(g, pool, var, &a, &b, -1, opts);

    // A leading coefficient proven *negative* at an endpoint disproves the
    // predicate outright: `h(t) < 0` for every `t ∈ (0, δ]`, and those points
    // are in the box.
    for s in [&left, &right].into_iter().flatten() {
        if let Some((_, false)) = s.reach(prec) {
            return Some(Verdict::False);
        }
    }

    // `j ≥ 1` means `g` was *proven* to be exactly zero at that endpoint, which
    // is a point of the box — so a strict inequality fails there.
    if strict
        && [&left, &right]
            .iter()
            .any(|e| e.as_ref().is_some_and(|s| s.j >= 1))
    {
        return Some(Verdict::False);
    }

    let collar = |e: &Option<EndpointSeries>| -> Option<Float> {
        e.as_ref()
            .and_then(|s| s.reach(prec))
            .filter(|(_, positive)| *positive)
            .map(|(d, _)| d)
    };
    let dl = collar(&left);
    let dr = collar(&right);
    if dl.is_none() && dr.is_none() {
        return None;
    }

    // The join points are rounded **into** the collars — `a+δₗ` down, `b−δᵣ` up
    // — so the middle piece can only overlap what the series already proved,
    // never leave a sliver between them uncovered. Rounding the other way would
    // open a gap narrower than an ulp, and a gap is a hole in the proof however
    // narrow it is.
    let mlo = dl.map_or_else(
        || a.clone(),
        |d| Float::with_val_round(prec, &a + &d, Round::Down).0,
    );
    let mhi = dr.map_or_else(
        || b.clone(),
        |d| Float::with_val_round(prec, &b - &d, Round::Up).0,
    );

    // The collars already cover the box; nothing left to prove.
    if mlo >= mhi {
        return Some(Verdict::True);
    }

    let middle = vec![(var, mlo, mhi)];
    let bound = extremum_search(
        g,
        pool,
        &middle,
        opts,
        Extremum::Min,
        SearchGoal::DecideSign,
    )
    .ok()?;
    (bound.0 > 0).then_some(Verdict::True)
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
        // (x-1/3)² + (y-1/3)² is zero at exactly one point of [0,1]² and
        // positive everywhere else, so no sign-change witness can exist. The
        // enclosure straddles zero, so `True` is unavailable too: `Undecided`
        // is the only honest answer and must not collapse either way.
        //
        // The zero sits at a point no search this module performs can name —
        // seed points and bisection midpoints are all dyadic — so the
        // point-root proof cannot reach it either. That is deliberate: it is
        // what keeps this test about the *absence* of a proof rather than
        // about arithmetic luck. When the tangential zero does land on a point
        // the search visits, the honest verdict changes; see
        // `no_roots_false_for_a_tangential_zero_at_a_point_the_search_visits`.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let third = pool.rational(1_i32, 3_i32);
        let dx = sub(&pool, x, third);
        let dy = sub(&pool, y, third);
        let e = pool.add(vec![pool.mul(vec![dx, dx]), pool.mul(vec![dy, dy])]);
        let cheap = BoundOptions {
            max_subdivisions: 64,
            ..opts()
        };
        let v = verified_no_roots(e, &pool, &[(x, 0.0, 1.0), (y, 0.0, 1.0)], &cheap).unwrap();
        assert_eq!(v, Verdict::Undecided);
    }

    /// The same shape, with its single zero at the centre of the box — a point
    /// the seed sweep evaluates. Substituting `x = y = 1/2` and simplifying
    /// lands on the literal `0`, which *proves* the value there is zero, and
    /// the centre is a point of the box. `False` is then a certificate, not the
    /// guess the sign-change search would have had to make.
    #[test]
    fn no_roots_false_for_a_tangential_zero_at_a_point_the_search_visits() {
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
        assert_eq!(v, Verdict::False);
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

    /// A root sitting exactly *on* an endpoint of the box.
    ///
    /// Subdivision provably cannot settle these: `x` on `[0, 1]` is
    /// non-negative throughout, so no negative witness exists anywhere in the
    /// box and the IVT search must come back empty however long it runs. The
    /// proof is not a search at all — the box is closed, `x = 0` is a point of
    /// it, and `0` is exactly zero there.
    #[test]
    fn no_roots_false_for_a_root_on_a_box_endpoint() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for (expr, lo, hi, label) in [
            (x, 0.0, 1.0, "x on [0,1] (root at the left endpoint)"),
            (
                sub(&pool, x, pool.integer(1_i32)),
                0.0,
                1.0,
                "x-1 on [0,1] (root at the right endpoint)",
            ),
            (
                pool.func("sin", vec![x]),
                0.0,
                1.0,
                "sin on [0,1] (root at the left endpoint)",
            ),
            (
                pool.func("log", vec![x]),
                1.0,
                2.0,
                "log on [1,2] (root at the left endpoint)",
            ),
            (
                sub(&pool, pool.func("exp", vec![x]), pool.integer(1_i32)),
                0.0,
                1.0,
                "exp-1 on [0,1] (root at the left endpoint)",
            ),
        ] {
            let v = verified_no_roots(expr, &pool, &[(x, lo, hi)], &opts()).unwrap();
            assert_eq!(v, Verdict::False, "{label}");
        }
    }

    /// The other direction, and the one that keeps `False` a certificate: an
    /// enclosure at an endpoint that merely *straddles* zero proves nothing.
    ///
    /// `exp(x) − 1 + 10⁻⁴⁰` is strictly positive on `[0, 1]` — it has no root
    /// at all — but its value at `x = 0` is `10⁻⁴⁰`, far below the `2⁻¹²⁸`-scale
    /// width of the enclosure there, so the enclosure contains zero. `True` is
    /// therefore out of reach, and `False` must **not** be claimed: containing
    /// zero is not being zero. `Undecided` is the only sound answer.
    #[test]
    fn no_roots_undecided_when_the_endpoint_enclosure_only_straddles_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 10^40, written out so the test does not depend on an integer-power
        // helper.
        let ten_to_40 =
            rug::Integer::from_str_radix("10000000000000000000000000000000000000000", 10).unwrap();
        let tiny = pool.rational(rug::Integer::from(1), ten_to_40);
        let e = pool.add(vec![pool.func("exp", vec![x]), pool.integer(-1_i32), tiny]);
        let v = verified_no_roots(e, &pool, &[(x, 0.0, 1.0)], &opts()).unwrap();
        assert_eq!(v, Verdict::Undecided);
    }

    /// Randomised sweep over expressions whose roots are known exactly.
    ///
    /// Every box endpoint and every root is a multiple of 1/4, so roots land on
    /// endpoints, on bisection midpoints and strictly between them — the three
    /// cases the two proof routes divide up. The assertion is one-sided in each
    /// direction and is the whole contract: `True` may only be returned when
    /// there is genuinely no root in the closed box, `False` only when there
    /// genuinely is one. `Undecided` is always permitted.
    #[test]
    fn no_roots_verdict_is_never_wrong_on_a_randomised_sweep() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move |n: i64| -> i64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as i64 % n
        };
        let cheap = BoundOptions {
            max_subdivisions: 64,
            tol: 1e-6,
            ..opts()
        };

        for _ in 0..120 {
            // Roots and box endpoints on the same quarter-integer grid.
            let (r1, r2) = (next(17) - 8, next(17) - 8);
            let lo_q = next(17) - 8;
            let hi_q = lo_q + 1 + next(8);
            let (lo, hi) = (lo_q as f64 / 4.0, hi_q as f64 / 4.0);

            let d1 = sub(&pool, x, pool.rational(r1, 4_i64));
            let d2 = sub(&pool, x, pool.rational(r2, 4_i64));
            for (expr, roots) in [(d1, vec![r1]), (pool.mul(vec![d1, d2]), vec![r1, r2])] {
                let has_root = roots.iter().any(|r| *r >= lo_q && *r <= hi_q);
                let v = verified_no_roots(expr, &pool, &[(x, lo, hi)], &cheap).unwrap();
                match v {
                    Verdict::True => assert!(
                        !has_root,
                        "certified root-free but roots {roots:?}/4 are in [{lo}, {hi}]"
                    ),
                    Verdict::False => assert!(
                        has_root,
                        "certified a root but roots {roots:?}/4 are outside [{lo}, {hi}]"
                    ),
                    Verdict::Undecided => {}
                }
            }
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
        // (x-1/3)² has a genuine root at x = 1/3 inside [0,2], but it never
        // changes sign, so no IVT witness exists and none may be invented:
        // turning this into `False` on the strength of the search having got
        // *close* would be a lucky guess, not a proof. `True` is also
        // unavailable (the enclosure contains zero). The root is at a
        // non-dyadic point, so no seed point or bisection midpoint ever lands
        // on it and the point-root proof cannot fire either.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let d = sub(&pool, x, pool.rational(1_i32, 3_i32));
        let e = pool.mul(vec![d, d]);
        let v = verified_no_roots(e, &pool, &[(x, 0.0, 2.0)], &opts()).unwrap();
        assert_eq!(v, Verdict::Undecided);
    }

    /// The same double root, this time at a point the seed sweep visits.
    /// A root of even multiplicity produces no sign change anywhere, so the
    /// IVT search provably cannot settle it — but `(1-1)² = 0` is an exact
    /// symbolic identity at a point of the box, which settles it outright.
    #[test]
    fn no_roots_false_for_a_double_root_proven_at_a_point() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let d = sub(&pool, x, pool.integer(1_i32));
        let e = pool.mul(vec![d, d]);
        // x = 1 is the centre of [0, 2] and the right endpoint of [0, 1]:
        // interior and boundary alike are points of the closed box.
        for (lo, hi) in [(0.0, 2.0), (0.0, 1.0), (1.0, 3.0)] {
            let v = verified_no_roots(e, &pool, &[(x, lo, hi)], &opts()).unwrap();
            assert_eq!(v, Verdict::False, "(x-1)^2 on [{lo}, {hi}]");
        }
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

    // ── endpoint-tight inequalities ─────────────────────────────────────

    /// `x·(2 + cos x) − 3·sin x` — Cusa–Huygens with the denominator cleared.
    /// Non-negative on `[0, π/2)` and tight as `x → 0`, where it vanishes to
    /// fifth order.
    fn cusa_huygens(pool: &ExprPool, x: ExprId) -> ExprId {
        let two_plus_cos = pool.add(vec![pool.integer(2_i32), pool.func("cos", vec![x])]);
        sub(
            pool,
            pool.mul(vec![x, two_plus_cos]),
            pool.mul(vec![pool.integer(3_i32), pool.func("sin", vec![x])]),
        )
    }

    #[test]
    fn tight_at_the_endpoint_is_certified_by_the_series_split() {
        // Subdivision alone can never do this: at x = 0 the margin is zero, so
        // every enclosure of the range straddles zero however fine the boxes.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let v = verified_sign(
            cusa_huygens(&pool, x),
            &pool,
            &[(x, 0.0, 1.5)],
            SignPredicate::NonNegative,
            &opts(),
        )
        .unwrap();
        assert_eq!(v, Verdict::True);
    }

    #[test]
    fn reversing_a_tight_inequality_is_refuted_not_certified() {
        // The control for the above: a sign error in the series argument would
        // surface here as a `True`.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let reversed = pool.mul(vec![pool.integer(-1_i32), cusa_huygens(&pool, x)]);
        let v = verified_sign(
            reversed,
            &pool,
            &[(x, 0.0, 1.5)],
            SignPredicate::NonNegative,
            &opts(),
        )
        .unwrap();
        assert_eq!(v, Verdict::False);
    }

    #[test]
    fn a_violation_only_next_to_the_endpoint_is_never_certified() {
        // `x^3 − x^2/1000` is negative exactly on (0, 1/1000): invisible to
        // endpoint and centre sampling, and far narrower than `tol`.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = sub(
            &pool,
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![
                pool.rational(1_i32, 1000_i32),
                pool.pow(x, pool.integer(2_i32)),
            ]),
        );
        let v = verified_sign(
            e,
            &pool,
            &[(x, 0.0, 1.5)],
            SignPredicate::NonNegative,
            &opts(),
        )
        .unwrap();
        assert_ne!(v, Verdict::True);
    }

    #[test]
    fn tightness_in_the_interior_is_still_out_of_reach() {
        // `(x − 7/10)^2 · (x + 1)` is non-negative and touches zero at an
        // interior point. The endpoint expansion does not apply, and the
        // honest answer stays Undecided rather than becoming a wrong True.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let shifted = sub(&pool, x, pool.rational(7_i32, 10_i32));
        let e = pool.mul(vec![
            shifted,
            shifted,
            pool.add(vec![x, pool.integer(1_i32)]),
        ]);
        let v = verified_sign(
            e,
            &pool,
            &[(x, 0.0, 1.5)],
            SignPredicate::NonNegative,
            &opts(),
        )
        .unwrap();
        assert_eq!(v, Verdict::Undecided);
    }

    #[test]
    fn strict_positivity_is_false_where_the_function_vanishes_exactly() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.mul(vec![x, x]);
        assert_eq!(
            verified_sign(e, &pool, &[(x, 0.0, 1.0)], SignPredicate::Positive, &opts()).unwrap(),
            Verdict::False
        );
        assert_eq!(
            verified_sign(
                e,
                &pool,
                &[(x, 0.0, 1.0)],
                SignPredicate::NonNegative,
                &opts()
            )
            .unwrap(),
            Verdict::True
        );
    }

    // ── termination ─────────────────────────────────────────────────────

    #[test]
    fn an_unreachable_tolerance_terminates_instead_of_spinning() {
        // Regression: a sub-box bisected down to the width floor used to be
        // pushed back onto the active list and immediately re-selected, so the
        // loop spun without ever consuming its subdivision budget. It only
        // showed up once `tol` was out of reach — which a large rational
        // coefficient arranges, because `tol` is an *absolute* width.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = sub(
            &pool,
            pool.mul(vec![
                pool.integer(1_000_000_000_000_i64),
                pool.func("sin", vec![x]),
            ]),
            pool.mul(vec![pool.integer(636_619_772_368_i64), x]),
        );
        let tight = BoundOptions {
            tol: 1e-40,
            ..opts()
        };
        let r = bound_on_box(e, &pool, &[(x, 0.0, 1.5)], &tight).unwrap();
        assert!(r.lower() <= 0.0 && r.upper() >= 0.0, "{r:?}");
    }

    #[test]
    fn a_sharp_rational_constant_is_certified_rather_than_timing_out() {
        // `10^12·sin x − 636619772368·x >= 0` on [0, 3/2]: Jordan's inequality
        // with 2/π rationalised to twelve digits, tight at x = 0.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = sub(
            &pool,
            pool.mul(vec![
                pool.integer(1_000_000_000_000_i64),
                pool.func("sin", vec![x]),
            ]),
            pool.mul(vec![pool.integer(636_619_772_368_i64), x]),
        );
        let v = verified_sign(
            e,
            &pool,
            &[(x, 0.0, 1.5)],
            SignPredicate::NonNegative,
            &opts(),
        )
        .unwrap();
        assert_eq!(v, Verdict::True);
    }
}
