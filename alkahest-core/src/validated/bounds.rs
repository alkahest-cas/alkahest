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
//! * A sub-box that cannot be bounded rigorously (branch cut, pole, or
//!   domain violation persisting after the box has been bisected far below
//!   the scale of the original box) causes the whole call to **refuse**
//!   with the underlying [`super::ValidatedError`], rather than silently
//!   omitting that piece of the domain from the answer.

use super::taylor::{taylor_range, TaylorContext, MAX_ORDER};
use super::{contains_zero, from_bounds, from_float, is_finite, lb, ub, width, ValidatedError};
use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use rug::float::Round;
use rug::Float;

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
    let mid = Float::with_val(prec, Float::with_val(prec, lo + hi) / 2u32);
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
    if boxes.is_empty() {
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
    let boxes0: Vec<FBox> = boxes
        .iter()
        .map(|(v, lo, hi)| (*v, Float::with_val(prec, lo), Float::with_val(prec, hi)))
        .collect();

    let (lo_bound, used_lo, exhausted_lo) =
        extremum_search(expr, pool, &boxes0, opts, Extremum::Min)?;
    let (hi_bound, used_hi, exhausted_hi) =
        extremum_search(expr, pool, &boxes0, opts, Extremum::Max)?;

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
/// Refuses (does not guess) when:
/// - `a` or `b` is non-finite (infinite-limit improper integrals are not
///   supported — there is no box to Taylor-expand over),
/// - `a > b`,
/// - the integrand has a genuine singularity in `[a, b]` (e.g. `1/sqrt(x)`
///   on `[0, 1]`) — subdivision is tried first in case the domain violation
///   is only a boundary artefact of a coarse box, but a true interior
///   singularity persists down to the bisection floor and the call refuses
///   with the underlying [`ValidatedError`] rather than silently skipping
///   the offending piece.
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

    let mut stack: Vec<(Float, Float)> = vec![(a_f, b_f)];
    let mut total: Option<ArbBall> = None;
    let mut subdivisions = 0usize;
    let mut exhausted = false;

    while let Some((lo, hi)) = stack.pop() {
        match local_integral(expr, pool, var, &lo, &hi, opts.order, prec) {
            Ok(piece) => {
                let piece_w = Float::with_val(prec, &hi - &lo);
                let piece_tol = Float::with_val(
                    prec,
                    &tol_total * Float::with_val(prec, &piece_w / &total_width),
                );
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
                    let mid = Float::with_val(prec, Float::with_val(prec, &lo + &hi) / 2u32);
                    stack.push((mid.clone(), hi));
                    stack.push((lo, mid));
                }
            }
            Err(e) if is_recoverable_domain_issue(&e) => {
                let piece_w = Float::with_val(prec, &hi - &lo).to_f64_round(Round::Up);
                if subdivisions >= opts.max_subdivisions || piece_w <= floor {
                    return Err(e);
                }
                subdivisions += 1;
                let mid = Float::with_val(prec, Float::with_val(prec, &lo + &hi) / 2u32);
                stack.push((mid.clone(), hi));
                stack.push((lo, mid));
            }
            Err(e) => return Err(e),
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

/// Verified check for the absence of roots of `expr` on `boxes`.
///
/// Returns:
/// - [`Verdict::True`] when the rigorous range enclosure of `expr` over the
///   whole box does not contain zero — `expr` is certified to have no root
///   anywhere in the box.
/// - [`Verdict::False`] in the univariate case, when the box is proven free
///   of poles/branch cuts (the full-box enclosure succeeded) *and* the
///   rigorously evaluated endpoint values have determined, opposite signs —
///   a root is certified to exist by the intermediate value theorem.
/// - [`Verdict::Undecided`] otherwise: the enclosure straddles zero and no
///   sign-change proof was available (multivariate, or same-signed /
///   indeterminate endpoints).
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

    // Root-existence (Verdict::False) via IVT: only sound in 1-D, and only
    // once we already know (from the successful full-box call above) that
    // `expr` is finite and pole/branch-cut free across the whole box, so it
    // is continuous on it.
    if boxes.len() == 1 {
        let (v, lo, hi) = boxes[0];
        if lo < hi {
            let flo = bound_on_box(expr, pool, &[(v, lo, lo)], opts);
            let fhi = bound_on_box(expr, pool, &[(v, hi, hi)], opts);
            if let (Ok(flo), Ok(fhi)) = (flo, fhi) {
                if let (Some(slo), Some(shi)) = (
                    determined_sign(flo.enclosure()),
                    determined_sign(fhi.enclosure()),
                ) {
                    if slo != shi {
                        return Ok(Verdict::False);
                    }
                }
            }
        }
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
    fn no_roots_undecided_when_enclosure_straddles_zero_without_sign_change() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // x - y over a 2-D box that straddles zero: no 1-D IVT argument
        // applies, so this must not be misreported as True or False.
        let e = sub(&pool, x, y);
        let v = verified_no_roots(e, &pool, &[(x, -1.0, 1.0), (y, -1.0, 1.0)], &opts()).unwrap();
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
