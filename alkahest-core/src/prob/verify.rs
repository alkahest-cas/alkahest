//! The gate. Nothing leaves this module's siblings without passing through
//! here first.
//!
//! Every closed form this module can produce — a mean, a variance, a raw
//! moment, a CDF value, a quantile, a general `E[f(X)]` — is a claim about the
//! value of a specific integral or sum. [`check`] takes the claim and the
//! **defining** integral, in the original variable, and asks whether they
//! agree numerically at several admissible parameter points.
//!
//! Three properties make this a real check rather than a ritual:
//!
//! * It integrates `f(x)·p(x)` over the support **in `x`**, from
//!   [`super::dists::pdf`], never in the reduction variable the closed form
//!   was derived in. A wrong Jacobian, a wrong inverse map, a dropped
//!   normalising constant and a mis-mapped integration bound all live in the
//!   reduction, and all of them show up here as a disagreement.
//! * It runs at several parameter points, so a closed form that happens to be
//!   right at `σ = 1` and wrong elsewhere — the classic "forgot to square the
//!   scale" defect — fails.
//! * Every way of *not* concluding is a refusal. A claim that cannot be
//!   evaluated, a quadrature that will not converge, and a parameter space
//!   with no admissible point are all [`ProbError::Unverified`]. There is no
//!   path through this file that returns success without a comparison having
//!   actually happened.

use rug::Float;

use crate::ball::IntervalEval;
use crate::deriv::{RewriteStep, SideCondition};
use crate::kernel::expr::PredicateKind;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

use super::dists;
use super::quad::{
    eval_ball, fl, locate, pi_ball, quadrature, quadrature_at, QuadOutcome, Region, Sample,
    VERIFY_PREC,
};
use super::{Distribution, ProbError, Support, UnverifiedReason};

/// What the numeric gate established about a returned closed form.
///
/// Attached to nothing by default — it is returned alongside the value by the
/// internal routines and surfaced through the derivation log — but a caller
/// that wants to know *how hard* the check was can read the margin.
#[derive(Clone, Debug, PartialEq)]
pub struct Evidence {
    /// How many admissible parameter points the claim was compared at.
    pub parameter_points: usize,
    /// The largest relative disagreement seen, across those points. A correct
    /// closed form sits at the level of the quadrature's own error.
    pub worst_relative_error: f64,
    /// Working precision of the quadrature, in bits.
    pub precision_bits: u32,
}

/// Relative agreement required between the closed form and the quadrature.
///
/// Not set at the quadrature's own accuracy (~`1e-30` here) on purpose. Closed
/// forms reaching this gate can carry `f64` literals — the integrator emits
/// `√(π/2)` as `1.2533141373155001` — so an honest answer is only good to
/// about `1e-16`. A *wrong* answer is wrong by an `O(1)` factor or an `O(1)`
/// additive term, never by `1e-13`: there is no defect that produces a moment
/// off in the thirteenth digit. So the threshold sits in the empty region
/// between the two, twelve orders clear of the noise and twelve orders clear
/// of any real error.
pub(crate) const REL_TOL: f64 = 1e-12;

/// Most free symbols the sampler will try to bind.
const MAX_FREE: usize = 4;

/// Fewest agreeing parameter points that count as verified **when there is
/// more than one to have**. One point is a coincidence: the
/// `σ`-versus-`σ²` family of mistakes agrees at exactly one.
const MIN_POINTS: usize = 2;

/// How many agreeing points this particular query needs.
///
/// A distribution whose parameters are all literals has exactly one point in
/// its parameter space, and demanding two of them would refuse
/// `Normal(0, 1).mean()` — a correct answer — for want of a second `σ` to try.
/// Checking one point *is* weaker, and it is weaker in a way the caller chose
/// by pinning the parameters; what must not happen is the count being met by
/// evaluating the same point twice, which is why `sample_points` returns one
/// entry rather than `MAX_POINTS` copies of it.
fn required_points(available: usize) -> usize {
    available.clamp(1, MIN_POINTS)
}

/// Most parameter points tried, to bound the cost of a check.
const MAX_POINTS: usize = 3;

/// Terms summed before a discrete tail is called inconclusive.
const MAX_DISCRETE_TERMS: u32 = 2000;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Check `claim` against `∫ integrand dx` (or `Σ integrand`) over the support
/// of `dist`.
///
/// `integrand` must be `f(x)·p(x)` written in `x`, with `p` taken from
/// [`super::dists::pdf`]. Passing anything else defeats the point of the file.
pub(crate) fn check(
    claim: ExprId,
    integrand: ExprId,
    x: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Result<Evidence, ProbError> {
    check_with_breaks(claim, integrand, x, dist, &[], pool)
}

/// As [`check`], but told where the integrand stops being smooth.
///
/// A double-exponential rule assumes the integrand is analytic on the
/// interval. `max(x - K, 0)·p(x)` is not — it has a corner at `K` — and the
/// rule's convergence collapses from doubly exponential to first order there,
/// which shows up as [`UnverifiedReason::QuadratureInconclusive`] and would
/// leave every option payoff unverifiable. Splitting the integral at the corner
/// restores analyticity on each piece. The breakpoints come from the same
/// analysis that split the payoff symbolically, so the verifier is cutting at
/// the same places the derivation did — but it is still integrating the
/// original `f(x)·p(x)`, kink and all, not the derivation's rewritten pieces.
pub(crate) fn check_with_breaks(
    claim: ExprId,
    integrand: ExprId,
    x: ExprId,
    dist: &Distribution,
    breaks: &[ExprId],
    pool: &ExprPool,
) -> Result<Evidence, ProbError> {
    let points = sample_points(claim, integrand, x, dist, pool);
    if points.is_empty() {
        return Err(ProbError::Unverified(UnverifiedReason::NoAdmissiblePoint));
    }
    let required = required_points(points.len());

    let integrand = ballable(integrand, pool);
    let density = ballable(dists::pdf(dist, x, pool), pool);
    let support = dist.support(pool);

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut claim_ever_evaluable = false;
    let mut divergences = 0usize;

    for binding in &points {
        // The quadrature runs **before** the claim is evaluated, deliberately.
        // A claim that cannot be evaluated is usually a symptom rather than the
        // disease: the reduction produced `∞` or `log 0` because the integral
        // it was built from has no value. Establishing that the integral
        // diverges is the more useful half of the answer, and it is only
        // reachable if the quadrature is not skipped on the way to reporting
        // the claim unevaluable.
        let rule = if support.is_discrete() {
            ValidatedRule {
                region: Region::Real,
                density_window: None,
            }
        } else {
            match prepare_rule(density, x, &support, binding, pool) {
                Some(r) => r,
                None => continue,
            }
        };
        let cuts = evaluate_breaks(breaks, binding, pool);
        let outcome = integrate_definition(integrand, x, &support, &rule, &cuts, binding, pool);
        let (value, est_err) = match outcome {
            QuadOutcome::Value { value, est_err } => (value, est_err),
            QuadOutcome::Divergent => {
                divergences += 1;
                continue;
            }
            QuadOutcome::Inconclusive => continue,
        };

        let Some(claim_ball) = eval_ball(claim, binding, pool, VERIFY_PREC) else {
            continue;
        };
        claim_ever_evaluable = true;

        let diff = Float::with_val(VERIFY_PREC, &claim_ball.mid - &value).abs();
        let scale = {
            let mut s = value.clone();
            s.abs_mut();
            if s < 1 {
                s = fl(VERIFY_PREC, 1.0);
            }
            s
        };
        let allowed = {
            let base = Float::with_val(VERIFY_PREC, &scale * fl(VERIFY_PREC, REL_TOL));
            let slack = Float::with_val(
                VERIFY_PREC,
                Float::with_val(VERIFY_PREC, &est_err + &claim_ball.rad) * 16u32,
            );
            if slack > base {
                slack
            } else {
                base
            }
        };
        if diff > allowed {
            return Err(ProbError::Unverified(UnverifiedReason::Disagreement(
                format!(
                    "closed form {} vs quadrature {} (allowed {})",
                    fmt(&claim_ball.mid),
                    fmt(&value),
                    fmt(&allowed)
                ),
            )));
        }
        let rel = Float::with_val(VERIFY_PREC, &diff / &scale).to_f64();
        if rel > worst {
            worst = rel;
        }
        compared += 1;
    }

    if divergences > 0 && compared == 0 {
        return Err(ProbError::Divergent(
            "the defining integral does not converge: the transformed integrand fails to \
             decay at the ends of the integration range at every admissible parameter point"
                .to_string(),
        ));
    }
    if !claim_ever_evaluable {
        return Err(ProbError::Unverified(UnverifiedReason::ClaimNotEvaluable));
    }
    if compared < required {
        return Err(ProbError::Unverified(
            UnverifiedReason::QuadratureInconclusive,
        ));
    }

    Ok(Evidence {
        parameter_points: compared,
        worst_relative_error: worst,
        precision_bits: VERIFY_PREC,
    })
}

/// Is the expectation of `integrand` over `dist` convergent?
///
/// Used ahead of a symbolic route so a divergent expectation is reported as
/// divergent rather than as "the integrator declined": `∫ e^{z²/2} dz` has no
/// antiderivative *and* no value, and only one of those is the interesting
/// half of the answer.
pub(crate) fn convergence_probe(
    integrand: ExprId,
    x: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Option<ProbError> {
    let points = sample_points(integrand, integrand, x, dist, pool);
    if points.is_empty() {
        return None;
    }
    let integrand_b = ballable(integrand, pool);
    let density = ballable(dists::pdf(dist, x, pool), pool);
    let support = dist.support(pool);
    let mut divergent = 0usize;
    let mut decided = 0usize;
    for binding in &points {
        let rule = if support.is_discrete() {
            ValidatedRule {
                region: Region::Real,
                density_window: None,
            }
        } else {
            match prepare_rule(density, x, &support, binding, pool) {
                Some(r) => r,
                None => continue,
            }
        };
        match integrate_definition(integrand_b, x, &support, &rule, &[], binding, pool) {
            QuadOutcome::Divergent => {
                divergent += 1;
                decided += 1;
            }
            QuadOutcome::Value { .. } => decided += 1,
            QuadOutcome::Inconclusive => {}
        }
    }
    if decided > 0 && divergent == decided {
        return Some(ProbError::Divergent(format!(
            "E[f(X)] over {} does not converge: the integrand fails to decay at every \
             admissible parameter point",
            dist.kind().name()
        )));
    }
    None
}

fn fmt(v: &Float) -> String {
    format!("{:.17e}", v.to_f64())
}

// ---------------------------------------------------------------------------
// Integrating the definition
// ---------------------------------------------------------------------------

/// The rule, *validated*, for one parameter point.
///
/// A double-exponential rule shifted onto the wrong part of the line does not
/// fail loudly — it converges, quickly and cleanly, to the integral of the
/// empty region, and every claim it is then compared against is refused for
/// the wrong reason or (worse, if the claim is small) confirmed. So the rule is
/// not used until it has reproduced the one number about this integrand that
/// is known in advance: **`∫p = 1`**. A window that cannot do that is
/// discarded, and the parameter point with it.
struct ValidatedRule {
    region: Region,
    /// Where the *density's* mass is. `None` on a compact support, where the
    /// rule needs no window.
    density_window: Option<crate::prob::quad::Window>,
}

fn prepare_rule(
    density: ExprId,
    x: ExprId,
    support: &Support,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> Option<ValidatedRule> {
    let region = match support {
        Support::Real => Region::Real,
        Support::Positive => Region::Positive,
        Support::Interval(lo, hi) => Region::Interval(
            eval_ball(*lo, binding, pool, VERIFY_PREC)?.mid,
            eval_ball(*hi, binding, pool, VERIFY_PREC)?.mid,
        ),
        _ => return None,
    };
    let window = match region {
        Region::Interval(_, _) => None,
        Region::Real => {
            let mut g = integrand_closure(density, x, binding, pool);
            Some(locate(&mut g, VERIFY_PREC, false)?)
        }
        Region::Positive => {
            let mut g = integrand_closure(density, x, binding, pool);
            Some(locate(&mut g, VERIFY_PREC, true)?)
        }
    };
    let mut g = integrand_closure(density, x, binding, pool);
    let QuadOutcome::Value { value, .. } =
        quadrature_at(&mut g, region.clone(), window.as_ref(), VERIFY_PREC)
    else {
        return None;
    };
    if rel(&value, &fl(VERIFY_PREC, 1.0)) > NORMALISATION_TOL {
        return None;
    }
    Some(ValidatedRule {
        region,
        density_window: window,
    })
}

/// How far `∫p` may sit from `1` before the rule is discarded.
///
/// Looser than [`REL_TOL`] because it is a *soundness* screen, not the
/// comparison itself: it exists to catch a rule that missed the mass
/// altogether — off by a factor, not by a digit.
const NORMALISATION_TOL: f64 = 1e-10;

/// The finite, in-support, numeric cut points of a break list at one
/// parameter point, in increasing order.
fn evaluate_breaks(breaks: &[ExprId], binding: &[(ExprId, Float)], pool: &ExprPool) -> Vec<Float> {
    let mut cuts: Vec<Float> = breaks
        .iter()
        .filter_map(|&b| eval_ball(b, binding, pool, VERIFY_PREC))
        .filter(|b| b.mid.is_finite())
        .map(|b| b.mid)
        .collect();
    cuts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    cuts.dedup();
    cuts
}

fn integrate_definition(
    integrand: ExprId,
    x: ExprId,
    support: &Support,
    rule: &ValidatedRule,
    cuts: &[Float],
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> QuadOutcome {
    if support.is_discrete() {
        return discrete_sum(integrand, x, support, binding, pool);
    }
    if !cuts.is_empty() {
        return integrate_pieces(integrand, x, &rule.region, cuts, binding, pool);
    }
    // `f·p` can put its mass somewhere the density's own window does not
    // reach — `E[X¹²]` under a log-normal lives twelve scale-lengths out. Take
    // the union: the density's window is validated, this one is not, and
    // covering both is the only combination that cannot lose mass either way.
    let own = match rule.region {
        Region::Interval(_, _) => None,
        Region::Real => {
            let mut g = integrand_closure(integrand, x, binding, pool);
            locate(&mut g, VERIFY_PREC, false)
        }
        Region::Positive => {
            let mut g = integrand_closure(integrand, x, binding, pool);
            locate(&mut g, VERIFY_PREC, true)
        }
    };
    let window = match (&rule.density_window, &own) {
        (Some(a), Some(b)) => Some(a.union(b, VERIFY_PREC)),
        (Some(a), None) => Some(a.clone()),
        (None, w) => w.clone(),
    };
    let mut g = integrand_closure(integrand, x, binding, pool);
    quadrature_at(&mut g, rule.region.clone(), window.as_ref(), VERIFY_PREC)
}

/// Integrate over `(lo, c₁), (c₁, c₂), …, (c_k, hi)`, each piece with the rule
/// that suits its shape: tanh–sinh where both ends are finite, and a
/// shift-or-reflect onto `(0, ∞)` where one is not.
fn integrate_pieces(
    integrand: ExprId,
    x: ExprId,
    region: &Region,
    cuts: &[Float],
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> QuadOutcome {
    // Endpoints of the support, `None` meaning unbounded on that side.
    let (lo, hi): (Option<Float>, Option<Float>) = match region {
        Region::Real => (None, None),
        Region::Positive => (Some(fl(VERIFY_PREC, 0.0)), None),
        Region::Interval(a, b) => (Some(a.clone()), Some(b.clone())),
    };
    // A cut outside the support is not a cut: drop it rather than producing an
    // empty or inverted piece.
    let inside: Vec<Float> = cuts
        .iter()
        .filter(|c| lo.as_ref().map_or(true, |l| *c > l) && hi.as_ref().map_or(true, |h| *c < h))
        .cloned()
        .collect();
    let mut edges: Vec<Option<Float>> = Vec::with_capacity(inside.len() + 2);
    edges.push(lo);
    edges.extend(inside.into_iter().map(Some));
    edges.push(hi);

    let mut total = fl(VERIFY_PREC, 0.0);
    let mut err = fl(VERIFY_PREC, 0.0);
    for w in edges.windows(2) {
        let outcome = match (&w[0], &w[1]) {
            (Some(a), Some(b)) => {
                let mut g = integrand_closure(integrand, x, binding, pool);
                quadrature(&mut g, Region::Interval(a.clone(), b.clone()), VERIFY_PREC)
            }
            // `∫_a^∞ g = ∫_0^∞ g(a + y) dy`
            (Some(a), None) => {
                let a = a.clone();
                let mut inner = integrand_closure(integrand, x, binding, pool);
                let mut g = |y: &Float| inner(&Float::with_val(VERIFY_PREC, &a + y));
                quadrature(&mut g, Region::Positive, VERIFY_PREC)
            }
            // `∫_{-∞}^b g = ∫_0^∞ g(b - y) dy`
            (None, Some(b)) => {
                let b = b.clone();
                let mut inner = integrand_closure(integrand, x, binding, pool);
                let mut g = |y: &Float| inner(&Float::with_val(VERIFY_PREC, &b - y));
                quadrature(&mut g, Region::Positive, VERIFY_PREC)
            }
            (None, None) => {
                let mut g = integrand_closure(integrand, x, binding, pool);
                quadrature(&mut g, Region::Real, VERIFY_PREC)
            }
        };
        match outcome {
            QuadOutcome::Value { value, est_err } => {
                total += value;
                err += est_err;
            }
            other => return other,
        }
    }
    QuadOutcome::Value {
        value: total,
        est_err: err,
    }
}

/// Sum `integrand` over a discrete support.
///
/// The infinite case is truncated, and the truncation is *justified* rather
/// than assumed: the sum runs until the ratio of successive terms has been
/// below ½ for long enough that the geometric bound on the remainder is below
/// the working precision. A run that does not get there is inconclusive, not
/// "close enough".
fn discrete_sum(
    integrand: ExprId,
    k: ExprId,
    support: &Support,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> QuadOutcome {
    let upper = match support {
        Support::IntegersUpTo(n) => Some(*n),
        Support::NonNegativeIntegers => None,
        _ => unreachable!("only discrete supports reach here"),
    };
    let limit = upper.unwrap_or(MAX_DISCRETE_TERMS);
    if limit > MAX_DISCRETE_TERMS {
        return QuadOutcome::Inconclusive;
    }

    let mut env: Vec<(ExprId, Float)> = binding.to_vec();
    env.push((k, fl(VERIFY_PREC, 0.0)));
    let last = env.len() - 1;

    let mut sum = fl(VERIFY_PREC, 0.0);
    let mut max_term = fl(VERIFY_PREC, 0.0);
    let mut small_run = 0u32;
    let mut last_term = fl(VERIFY_PREC, 0.0);

    for i in 0..=limit {
        env[last].1 = fl(VERIFY_PREC, f64::from(i));
        let Some(b) = eval_ball(integrand, &env, pool, VERIFY_PREC) else {
            return QuadOutcome::Inconclusive;
        };
        if !b.mid.is_finite() {
            return QuadOutcome::Divergent;
        }
        let mut mag = b.mid.clone();
        mag.abs_mut();
        if mag > max_term {
            max_term = mag.clone();
        }
        sum += &b.mid;
        // Below the working precision relative to the largest term seen, and
        // shrinking: the remaining tail is bounded by twice this term.
        let tiny = Float::with_val(VERIFY_PREC, &max_term * fl(VERIFY_PREC, 1e-40));
        if mag < tiny && mag <= last_term {
            small_run += 1;
        } else {
            small_run = 0;
        }
        last_term = mag;
        if upper.is_none() && small_run >= 8 {
            return QuadOutcome::Value {
                value: sum,
                est_err: Float::with_val(VERIFY_PREC, last_term.clone() * 2u32),
            };
        }
    }
    match upper {
        Some(_) => QuadOutcome::Value {
            value: sum,
            est_err: fl(VERIFY_PREC, 0.0),
        },
        // Ran out of terms without the tail going quiet.
        None => QuadOutcome::Inconclusive,
    }
}

// ---------------------------------------------------------------------------
// max / min → Piecewise, so the ball evaluator can see them
// ---------------------------------------------------------------------------

/// Rewrite `max`/`min` into `Piecewise` for the benefit of the interval
/// evaluator, which dispatches `Func` nodes through the primitive registry and
/// finds no ball kernel for either.
///
/// Exact for real arguments, and the only rewrite applied to an integrand
/// before it is quadratured — the verifier must integrate what the user asked
/// about, not a convenient neighbour of it.
fn ballable(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Func { name, args } if args.len() == 2 && (name == "max" || name == "min") => {
            let a = ballable(args[0], pool);
            let b = ballable(args[1], pool);
            let (first, second) = if name == "max" { (a, b) } else { (b, a) };
            let cond = pool.pred_ge(first, second);
            pool.piecewise(vec![(cond, first)], second)
        }
        ExprData::Add(args) => pool.add(args.iter().map(|&a| ballable(a, pool)).collect()),
        ExprData::Mul(args) => pool.mul(args.iter().map(|&a| ballable(a, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(ballable(base, pool), ballable(exp, pool)),
        ExprData::Func { name, args } => {
            pool.func(name, args.iter().map(|&a| ballable(a, pool)).collect())
        }
        _ => expr,
    }
}

// ---------------------------------------------------------------------------
// Choosing parameter points
// ---------------------------------------------------------------------------

/// Role-appropriate values for a distribution's parameters, tried in order.
///
/// Deliberately not all "nice": a table checked only at `Normal(0, 1)`,
/// `Uniform(0, 1)` and `Exponential(1)` passes for a formula that has dropped
/// every `σ`, every `b - a` and every `λ` — at those points they are all `1`.
/// Each row below moves the location and the scale away from the identity, and
/// at least one row per distribution sits near the awkward edge of its
/// parameter space (`Beta(½, ½)` is unbounded at both endpoints;
/// `Gamma(½, ·)` is unbounded at zero).
fn role_rows(dist: &Distribution) -> &'static [&'static [f64]] {
    use super::DistKind::*;
    match dist.kind() {
        Normal => &[&[0.0, 1.0], &[2.0, 0.5], &[-1.5, 2.0], &[0.25, 3.0]],
        LogNormal => &[&[0.0, 0.3], &[0.5, 1.0], &[-1.0, 0.75], &[0.2, 0.6]],
        Uniform => &[&[0.0, 1.0], &[-2.0, 3.0], &[1.5, 4.25], &[-0.5, 0.25]],
        Exponential | Poisson => &[&[1.0], &[0.4], &[2.5], &[1.75]],
        Gamma => &[&[2.0, 1.0], &[0.5, 2.0], &[3.5, 0.7], &[1.0, 3.0]],
        Beta => &[&[2.0, 3.0], &[0.5, 0.5], &[1.0, 4.0], &[3.2, 1.4]],
        Bernoulli => &[&[0.3], &[0.5], &[0.8], &[0.125]],
        Binomial => &[&[0.0, 0.3], &[0.0, 0.5], &[0.0, 0.8], &[0.0, 0.125]],
    }
}

/// Values tried for a free symbol that is not one of the distribution's own
/// parameters — a strike, a threshold, a dose.
const EXTRA_ROWS: [f64; 4] = [0.6, 1.7, 1.2, 0.35];

/// Build up to [`MAX_POINTS`] parameter assignments at which the distribution's
/// constraints hold.
///
/// A parameter that is already a literal is left alone; a parameter that is a
/// bare symbol is bound to its role's value; a parameter that is a compound
/// expression has its own free symbols bound from the extra ladder and is then
/// **checked** against the constraints like everything else. That last case is
/// why the constraint check runs on every candidate row rather than only on
/// the fallback path: `Normal(0, s - 2)` must not be verified at `s = 1.7`.
fn sample_points(
    claim: ExprId,
    integrand: ExprId,
    x: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Vec<Vec<(ExprId, Float)>> {
    use crate::eval::symbols::collect_free_symbols;

    let mut free: Vec<ExprId> = Vec::new();
    collect_free_symbols(claim, pool, &mut free);
    collect_free_symbols(integrand, pool, &mut free);
    for &p in dist.params() {
        collect_free_symbols(p, pool, &mut free);
    }
    free.retain(|&s| s != x && !is_infinity(s, pool));
    if free.len() > MAX_FREE {
        return Vec::new();
    }

    // Which free symbols are exactly the distribution's own bare-symbol
    // parameters, and in which slot.
    let mut role: Vec<Option<usize>> = vec![None; free.len()];
    for (slot, &p) in dist.params().iter().enumerate() {
        if matches!(pool.get(p), ExprData::Symbol { .. }) {
            if let Some(i) = free.iter().position(|&s| s == p) {
                role[i] = Some(slot);
            }
        }
    }

    let rows = role_rows(dist);
    let constraints = dist.constraints(pool);
    let mut out = Vec::new();
    if free.is_empty() {
        // Every parameter is a literal. There is exactly one point to check at,
        // and repeating it would inflate the evidence count without adding any
        // evidence — see `required_points`.
        return if admissible(&constraints, &[], pool) {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
    }
    for r in 0..EXTRA_ROWS.len().max(rows.len()) {
        let mut binding: Vec<(ExprId, Float)> = Vec::with_capacity(free.len());
        for (i, &sym) in free.iter().enumerate() {
            let v = match role[i] {
                Some(slot) => match rows.get(r).and_then(|row| row.get(slot)) {
                    Some(v) => *v,
                    None => EXTRA_ROWS[r % EXTRA_ROWS.len()],
                },
                None => EXTRA_ROWS[(r + i) % EXTRA_ROWS.len()],
            };
            binding.push((sym, fl(VERIFY_PREC, v)));
        }
        if admissible(&constraints, &binding, pool) {
            out.push(binding);
        }
        if out.len() == MAX_POINTS {
            break;
        }
    }
    out
}

fn is_infinity(sym: ExprId, pool: &ExprPool) -> bool {
    sym == pool.pos_infinity()
}

/// Do every one of the distribution's constraints hold at this binding?
///
/// A constraint that cannot be evaluated counts as *not* holding: a point the
/// gate cannot confirm is admissible is a point the gate must not verify at,
/// or the whole exercise reduces to checking a formula outside its own domain.
fn admissible(constraints: &[ExprId], binding: &[(ExprId, Float)], pool: &ExprPool) -> bool {
    constraints.iter().all(|&c| {
        eval_ball(c, binding, pool, VERIFY_PREC)
            .map(|b| b.mid > 0.5)
            .unwrap_or(false)
    })
}

/// Re-exported for the moment/CDF routines, which need the density in `x`.
pub(crate) fn definition_integrand(
    f: ExprId,
    x: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> ExprId {
    pool.mul(vec![f, dists::pdf(dist, x, pool)])
}

// ---------------------------------------------------------------------------
// The derivation-log record
// ---------------------------------------------------------------------------

/// The log entry every verified value carries, with the distribution's
/// *undischarged* constraints attached as side conditions.
///
/// A numeric parameter's constraint was decided at construction and is not
/// repeated here. A symbolic one was not decided by anybody, and travels with
/// the answer — that is the whole difference between "checked" and "assumed",
/// and putting it in the log is what lets a caller see which it got.
pub(crate) fn evidence_step(
    evidence: &Evidence,
    claim: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> RewriteStep {
    let mut conditions = Vec::new();
    for c in dist.constraints(pool) {
        if let Some(side) = undischarged(c, pool) {
            conditions.push(side);
        }
    }
    let _ = evidence;
    RewriteStep::with_conditions("prob_verified_against_quadrature", claim, claim, conditions)
}

/// A constraint that a numeric check already settled contributes nothing; one
/// that is still open becomes a [`SideCondition`].
fn undischarged(pred: ExprId, pool: &ExprPool) -> Option<SideCondition> {
    let ExprData::Predicate { kind, args } = pool.get(pred) else {
        return None;
    };
    let [lhs, rhs] = args.as_slice() else {
        return None;
    };
    let (lhs, rhs) = (*lhs, *rhs);
    // Simplified, because this string is the whole of what the caller sees:
    // `sigma - 0 > 0` and `sigma > 0` are the same hypothesis, and only one of
    // them reads like one.
    let gap = crate::simplify::simplify(
        match kind {
            PredicateKind::Gt | PredicateKind::Ge => super::sub(lhs, rhs, pool),
            PredicateKind::Lt | PredicateKind::Le => super::sub(rhs, lhs, pool),
            _ => return None,
        },
        pool,
    )
    .value;
    // Decided already: nothing for the caller to discharge.
    if !matches!(dists::sign_of(gap, pool), dists::Sign::Unknown) {
        return None;
    }
    match kind {
        PredicateKind::Gt | PredicateKind::Lt => Some(SideCondition::Positive(gap)),
        _ => Some(SideCondition::InDomain(gap, Domain::NonNegative)),
    }
}

// ---------------------------------------------------------------------------
// CDF and quantile: checked against the density, not against a table
// ---------------------------------------------------------------------------

/// Fractions of the located scale at which a CDF is sampled.
const CDF_OFFSETS: [f64; 5] = [-2.0, -0.7, 0.0, 0.8, 2.5];

/// Fractions of a compact support at which a CDF is sampled.
const CDF_FRACTIONS: [f64; 5] = [0.05, 0.3, 0.5, 0.75, 0.95];

/// Probabilities a quantile is checked at.
const QUANTILE_LADDER: [f64; 4] = [0.25, 0.5, 0.75, 0.9];

/// Check a CDF claim against the density it is supposed to accumulate.
///
/// Two things have to hold, and both are checked:
///
/// * **increments** — `F(t_{i+1}) - F(t_i)` equals `∫_{t_i}^{t_{i+1}} p`, on a
///   compact interval where tanh–sinh is essentially exact;
/// * **level** — `F(t₀)` equals the *tail* `∫_{lo}^{t₀} p`. Without this an
///   `F` off by a constant passes every increment, which is the mistake a
///   CDF check is most likely to be asked to catch (`erf` versus `erfc`,
///   `1 - F` versus `F`).
///
/// Sampling `F` at a point and integrating a jump discontinuity is deliberately
/// *not* how this works: `1_{x ≤ t}·p(x)` is not smooth, a
/// double-exponential rule across the jump converges at first order, and the
/// resulting "agreement to 1e-2" would be indistinguishable from a pass.
pub(crate) fn check_cdf(
    claim: ExprId,
    x: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Result<Evidence, ProbError> {
    let density = dists::pdf(dist, x, pool);
    let points = sample_points(claim, density, x, dist, pool);
    if points.is_empty() {
        return Err(ProbError::Unverified(UnverifiedReason::NoAdmissiblePoint));
    }
    let support = dist.support(pool);
    if support.is_discrete() {
        return Err(ProbError::Unsupported(
            "a discrete CDF has no symbolic form here to check".to_string(),
        ));
    }
    let density = ballable(density, pool);

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut claim_ever_evaluable = false;

    for binding in &points {
        let Some(abscissae) = cdf_abscissae(density, x, &support, binding, pool) else {
            continue;
        };
        let mut levels = Vec::with_capacity(abscissae.len());
        let mut ok = true;
        for t in &abscissae {
            match eval_at(claim, x, t, binding, pool) {
                Some(v) => levels.push(v),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        claim_ever_evaluable = true;

        // Level.
        let QuadOutcome::Value { value: tail, .. } =
            tail_below(density, x, &support, &abscissae[0], binding, pool)
        else {
            continue;
        };
        compare(&levels[0], &tail, "CDF level at the left sample")?;
        worst = worst.max(rel(&levels[0], &tail));
        compared += 1;

        // Increments.
        for i in 0..abscissae.len() - 1 {
            let mut g = integrand_closure(density, x, binding, pool);
            let QuadOutcome::Value { value: mass, .. } = quadrature(
                &mut g,
                Region::Interval(abscissae[i].clone(), abscissae[i + 1].clone()),
                VERIFY_PREC,
            ) else {
                continue;
            };
            let step = Float::with_val(VERIFY_PREC, &levels[i + 1] - &levels[i]);
            compare(&step, &mass, "CDF increment")?;
            worst = worst.max(rel(&step, &mass));
            compared += 1;
        }
    }

    finish(
        compared,
        worst,
        claim_ever_evaluable,
        required_points(points.len()),
    )
}

/// Check a quantile claim by pushing it back through the density:
/// `∫_{lo}^{Q(p)} density` must equal `p`.
///
/// Inverting the *claim* rather than comparing it to a second closed form is
/// the point — there is no second closed form, and a table checked against
/// itself measures nothing.
pub(crate) fn check_quantile(
    claim: ExprId,
    p_arg: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Result<Evidence, ProbError> {
    let x = dists::fresh_var(&[claim, p_arg], pool);
    let density = dists::pdf(dist, x, pool);
    let points = sample_points(claim, density, x, dist, pool);
    if points.is_empty() {
        return Err(ProbError::Unverified(UnverifiedReason::NoAdmissiblePoint));
    }
    let support = dist.support(pool);
    if support.is_discrete() {
        return Err(ProbError::Unsupported(
            "a discrete quantile has no symbolic form here to check".to_string(),
        ));
    }
    let density = ballable(density, pool);

    // `p` is a free symbol of the claim, and the generic sampler would have
    // bound it to whatever its ladder says — including values outside `(0,1)`,
    // where a quantile means nothing. Override it here.
    let p_sym = match pool.get(p_arg) {
        ExprData::Symbol { .. } => Some(p_arg),
        _ => None,
    };

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut claim_ever_evaluable = false;

    for (i, binding) in points.iter().enumerate() {
        for (j, &pv) in QUANTILE_LADDER.iter().enumerate() {
            if (i + j) % 2 == 1 {
                continue; // two probabilities per parameter point is enough
            }
            let mut env = binding.clone();
            let target = match p_sym {
                Some(sym) => {
                    match env.iter_mut().find(|(s, _)| *s == sym) {
                        Some(slot) => slot.1 = fl(VERIFY_PREC, pv),
                        None => env.push((sym, fl(VERIFY_PREC, pv))),
                    }
                    fl(VERIFY_PREC, pv)
                }
                // A literal probability: use it as written, once.
                None => match eval_ball(p_arg, &env, pool, VERIFY_PREC) {
                    Some(b) if b.mid > 0 && b.mid < 1 => b.mid,
                    _ => continue,
                },
            };
            let Some(q) = eval_ball(claim, &env, pool, VERIFY_PREC) else {
                continue;
            };
            claim_ever_evaluable = true;
            let QuadOutcome::Value { value: mass, .. } =
                tail_below(density, x, &support, &q.mid, &env, pool)
            else {
                continue;
            };
            compare(&mass, &target, "quantile round-trip F(Q(p)) = p")?;
            worst = worst.max(rel(&mass, &target));
            compared += 1;
            if p_sym.is_none() {
                break;
            }
        }
    }

    finish(
        compared,
        worst,
        claim_ever_evaluable,
        required_points(points.len()),
    )
}

fn finish(
    compared: usize,
    worst: f64,
    claim_ever_evaluable: bool,
    minimum: usize,
) -> Result<Evidence, ProbError> {
    if !claim_ever_evaluable {
        return Err(ProbError::Unverified(UnverifiedReason::ClaimNotEvaluable));
    }
    if compared < minimum {
        return Err(ProbError::Unverified(
            UnverifiedReason::QuadratureInconclusive,
        ));
    }
    Ok(Evidence {
        parameter_points: compared,
        worst_relative_error: worst,
        precision_bits: VERIFY_PREC,
    })
}

fn rel(a: &Float, b: &Float) -> f64 {
    let mut scale = b.clone();
    scale.abs_mut();
    if scale < 1 {
        scale = fl(VERIFY_PREC, 1.0);
    }
    Float::with_val(
        VERIFY_PREC,
        Float::with_val(VERIFY_PREC, a - b).abs() / scale,
    )
    .to_f64()
}

fn compare(claim: &Float, truth: &Float, what: &str) -> Result<(), ProbError> {
    if rel(claim, truth) > REL_TOL {
        return Err(ProbError::Unverified(UnverifiedReason::Disagreement(
            format!(
                "{what}: closed form {} vs quadrature {}",
                fmt(claim),
                fmt(truth)
            ),
        )));
    }
    Ok(())
}

fn eval_at(
    expr: ExprId,
    x: ExprId,
    at: &Float,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> Option<Float> {
    let mut env = binding.to_vec();
    env.push((x, at.clone()));
    eval_ball(expr, &env, pool, VERIFY_PREC).map(|b| b.mid)
}

/// A closure that evaluates `density` at a point, with everything that does
/// not depend on `x` computed **once**.
///
/// A quadrature calls this up to ten thousand times. Without the fold, every
/// one of those calls recomputes `Γ(k)`, `θ^k` and `1/√(2π)` at
/// [`VERIFY_PREC`] bits — `Γ` in particular is the single most expensive thing
/// in the loop, and it does not vary with the integration variable. Folding
/// them into pre-evaluated balls is what takes the gate from minutes to
/// seconds, and it is exact: the constants are *balls*, not `f64` literals, so
/// nothing is rounded on the way in.
fn integrand_closure<'a>(
    density: ExprId,
    x: ExprId,
    binding: &'a [(ExprId, Float)],
    pool: &'a ExprPool,
) -> impl FnMut(&Float) -> Sample + 'a {
    let (folded, consts) = fold_constants(density, x, binding, pool);
    let mut ev = IntervalEval::new(VERIFY_PREC);
    ev.bind(super::pi(pool), pi_ball(VERIFY_PREC));
    for (sym, val) in binding {
        ev.bind(*sym, point_ball(val));
    }
    for (sym, ball) in consts {
        ev.bind(sym, ball);
    }
    move |v: &Float| {
        ev.bind(x, point_ball(v));
        match ev.eval(folded, pool) {
            Some(b) if b.mid.is_finite() && b.rad.is_finite() => Sample::Value(b.mid),
            // A non-finite ball at an abscissa is *not* evidence of divergence:
            // the outermost nodes of a double-exponential rule sit at `e^{10⁴³}`,
            // where any density overflows. The weight there is below the working
            // precision, so dropping the node is exact — and a genuinely
            // divergent integrand is still caught, by the `w·f` product
            // overflowing at nodes that *are* inside the representable range.
            _ => Sample::Unevaluable,
        }
    }
}

fn point_ball(v: &Float) -> crate::ball::ArbBall {
    crate::ball::ArbBall {
        mid: v.clone(),
        rad: fl(VERIFY_PREC, 0.0),
        prec: VERIFY_PREC,
    }
}

/// Replace every maximal `x`-free compound subexpression by a fresh symbol
/// bound to its value.
fn fold_constants(
    expr: ExprId,
    x: ExprId,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> (ExprId, Vec<(ExprId, crate::ball::ArbBall)>) {
    let mut consts = Vec::new();
    let folded = fold_walk(expr, x, binding, pool, &mut consts, 0);
    (folded, consts)
}

fn fold_walk(
    expr: ExprId,
    x: ExprId,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
    consts: &mut Vec<(ExprId, crate::ball::ArbBall)>,
    depth: u32,
) -> ExprId {
    if depth > 32 {
        return expr;
    }
    if !dists::contains(expr, x, pool) {
        // Atoms are already free; folding them would only add a lookup.
        if matches!(
            pool.get(expr),
            ExprData::Integer(_)
                | ExprData::Rational(_)
                | ExprData::Float(_)
                | ExprData::Symbol { .. }
        ) {
            return expr;
        }
        if let Some(ball) = eval_ball(expr, binding, pool, VERIFY_PREC) {
            let sym = pool.symbol(format!("_prob_k{}", consts.len()), Domain::Real);
            consts.push((sym, ball));
            return sym;
        }
        return expr;
    }
    match pool.get(expr) {
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| fold_walk(a, x, binding, pool, consts, depth + 1))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| fold_walk(a, x, binding, pool, consts, depth + 1))
                .collect(),
        ),
        ExprData::Pow { base, exp } => pool.pow(
            fold_walk(base, x, binding, pool, consts, depth + 1),
            fold_walk(exp, x, binding, pool, consts, depth + 1),
        ),
        ExprData::Func { name, args } => pool.func(
            name,
            args.iter()
                .map(|&a| fold_walk(a, x, binding, pool, consts, depth + 1))
                .collect(),
        ),
        ExprData::Piecewise { branches, default } => pool.piecewise(
            branches
                .iter()
                .map(|(c, v)| {
                    (
                        fold_walk(*c, x, binding, pool, consts, depth + 1),
                        fold_walk(*v, x, binding, pool, consts, depth + 1),
                    )
                })
                .collect(),
            fold_walk(default, x, binding, pool, consts, depth + 1),
        ),
        ExprData::Predicate { kind, args } => pool.predicate(
            kind,
            args.iter()
                .map(|&a| fold_walk(a, x, binding, pool, consts, depth + 1))
                .collect(),
        ),
        _ => expr,
    }
}

/// Sample points inside the support, placed where the density actually is.
fn cdf_abscissae(
    density: ExprId,
    x: ExprId,
    support: &Support,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> Option<Vec<Float>> {
    match support {
        Support::Interval(lo, hi) => {
            let lo = eval_ball(*lo, binding, pool, VERIFY_PREC)?.mid;
            let hi = eval_ball(*hi, binding, pool, VERIFY_PREC)?.mid;
            let width = Float::with_val(VERIFY_PREC, &hi - &lo);
            Some(
                CDF_FRACTIONS
                    .iter()
                    .map(|&f| Float::with_val(VERIFY_PREC, &lo + &width * fl(VERIFY_PREC, f)))
                    .collect(),
            )
        }
        Support::Real => {
            let mut g = integrand_closure(density, x, binding, pool);
            let w = locate(&mut g, VERIFY_PREC, false)?;
            Some(
                CDF_OFFSETS
                    .iter()
                    .map(|&o| {
                        Float::with_val(VERIFY_PREC, &w.centre + &w.scale * fl(VERIFY_PREC, o))
                    })
                    .collect(),
            )
        }
        Support::Positive => {
            let mut g = integrand_closure(density, x, binding, pool);
            let w = locate(&mut g, VERIFY_PREC, true)?;
            Some(
                CDF_OFFSETS
                    .iter()
                    .map(|&o| {
                        Float::with_val(
                            VERIFY_PREC,
                            Float::with_val(VERIFY_PREC, &w.centre + &w.scale * fl(VERIFY_PREC, o))
                                .exp(),
                        )
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// `∫_{lo}^{t} density`, where `lo` is the bottom of the support (possibly
/// `-∞`, in which case the tail is reflected onto `(0, ∞)` and integrated
/// there).
fn tail_below(
    density: ExprId,
    x: ExprId,
    support: &Support,
    t: &Float,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> QuadOutcome {
    match support {
        Support::Real => {
            let t = t.clone();
            let mut inner = integrand_closure(density, x, binding, pool);
            let mut g = |y: &Float| inner(&Float::with_val(VERIFY_PREC, &t - y));
            quadrature(&mut g, Region::Positive, VERIFY_PREC)
        }
        Support::Positive => {
            let mut g = integrand_closure(density, x, binding, pool);
            quadrature(
                &mut g,
                Region::Interval(fl(VERIFY_PREC, 0.0), t.clone()),
                VERIFY_PREC,
            )
        }
        Support::Interval(lo, _) => {
            let Some(lo) = eval_ball(*lo, binding, pool, VERIFY_PREC) else {
                return QuadOutcome::Inconclusive;
            };
            let mut g = integrand_closure(density, x, binding, pool);
            quadrature(&mut g, Region::Interval(lo.mid, t.clone()), VERIFY_PREC)
        }
        _ => QuadOutcome::Inconclusive,
    }
}

// ---------------------------------------------------------------------------
// Characteristic functions
// ---------------------------------------------------------------------------

/// Frequencies a characteristic function is checked at.
///
/// Small and of both signs. `t = 0` is deliberately absent: `φ(0) = 1` for
/// every law and every mistake, so it is the one point that distinguishes
/// nothing. A negative `t` is present because the sign half of the
/// `φ ↔ F` convention only shows up there — `φ` of a symmetric law is even,
/// and `Normal(0, σ)` would pass a same-sign-only ladder with the sign wrong.
const T_LADDER: [f64; 3] = [0.4, -0.9, 1.3];

/// Check `φ` against the definition, and — where the transform has a rule —
/// against [`crate::transform::fourier_transform`] as well.
///
/// Returns the evidence and whether the Fourier cross-check actually ran.
///
/// The definition check splits `E[e^{itX}]` into its real and imaginary parts,
/// `∫p(x)cos(tx)dx` and `∫p(x)sin(tx)dx`, both of which are ordinary real
/// integrals the existing rule handles. The claim itself is evaluated in ℂ (see
/// [`super::cplx`]) — reading a complex expression through the real evaluator
/// would come back as a refusal, and mistaking that for a failure is how this
/// check would end up never running.
pub(crate) fn check_characteristic(
    claim: ExprId,
    t: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Result<(Evidence, bool), ProbError> {
    use super::cplx::eval_complex;

    let x = dists::fresh_var(&[claim, t], pool);
    let density_expr = dists::pdf(dist, x, pool);
    let points = sample_points(claim, density_expr, x, dist, pool);
    if points.is_empty() {
        return Err(ProbError::Unverified(UnverifiedReason::NoAdmissiblePoint));
    }
    let support = dist.support(pool);
    let density = ballable(density_expr, pool);

    // `φ(t) = F{p}(-t/2π)`, through the actual transform. Absent for the laws
    // whose density is not in the transform's table — a missing cross-check,
    // not a failing one.
    let fourier_phi = fourier_route(dist, x, t, pool);

    let t_sym = match pool.get(t) {
        ExprData::Symbol { .. } => Some(t),
        _ => None,
    };

    let tx = pool.mul(vec![t, x]);
    let cos_part = ballable(
        pool.mul(vec![density_expr, pool.func("cos", vec![tx])]),
        pool,
    );
    let sin_part = ballable(
        pool.mul(vec![density_expr, pool.func("sin", vec![tx])]),
        pool,
    );

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut claim_ever_evaluable = false;
    let mut fourier_agreed = false;

    for (i, binding) in points.iter().enumerate() {
        let rule = if support.is_discrete() {
            ValidatedRule {
                region: Region::Real,
                density_window: None,
            }
        } else {
            match prepare_rule(density, x, &support, binding, pool) {
                Some(r) => r,
                None => continue,
            }
        };
        for (j, &tv) in T_LADDER.iter().enumerate() {
            if (i + j) % 2 == 1 {
                continue; // two frequencies per parameter point
            }
            let mut env = binding.clone();
            match t_sym {
                Some(sym) => match env.iter_mut().find(|(s, _)| *s == sym) {
                    Some(slot) => slot.1 = fl(VERIFY_PREC, tv),
                    None => env.push((sym, fl(VERIFY_PREC, tv))),
                },
                // A literal frequency: check it as written, once.
                None => {
                    if j > 0 {
                        continue;
                    }
                }
            }
            let Some(value) = eval_complex(claim, &env, pool, VERIFY_PREC) else {
                continue;
            };
            claim_ever_evaluable = true;

            // A discrete law sums; a continuous one integrates over a compact
            // core with the truncated probability mass carried as the error
            // bar. See `oscillatory_core` for why `φ` does not go through the
            // same route as everything else.
            let (re, im, tail) = if support.is_discrete() {
                let (QuadOutcome::Value { value: re, .. }, QuadOutcome::Value { value: im, .. }) = (
                    integrate_definition(cos_part, x, &support, &rule, &[], &env, pool),
                    integrate_definition(sin_part, x, &support, &rule, &[], &env, pool),
                ) else {
                    continue;
                };
                (re, im, fl(VERIFY_PREC, 0.0))
            } else {
                let Some((core, tail)) = oscillatory_core(density, x, &rule, &env, pool) else {
                    continue;
                };
                let Some(re) = integrate_core(cos_part, x, &core, &env, pool) else {
                    continue;
                };
                let Some(im) = integrate_core(sin_part, x, &core, &env, pool) else {
                    continue;
                };
                (re, im, tail)
            };

            compare_within(&value.re, &re, &tail, "Re φ(t) against ∫p(x)cos(tx)dx")?;
            compare_within(&value.im, &im, &tail, "Im φ(t) against ∫p(x)sin(tx)dx")?;
            worst = worst.max(rel(&value.re, &re)).max(rel(&value.im, &im));
            compared += 2;

            if let Some(phi_f) = fourier_phi {
                if let Some(vf) = eval_complex(phi_f, &env, pool, VERIFY_PREC) {
                    compare(&value.re, &vf.re, "Re φ(t) against F{p}(-t/2π)")?;
                    compare(&value.im, &vf.im, "Im φ(t) against F{p}(-t/2π)")?;
                    fourier_agreed = true;
                    compared += 2;
                }
            }
        }
    }

    // Two comparisons per point — the real and the imaginary part.
    let evidence = finish(
        compared,
        worst,
        claim_ever_evaluable,
        2 * required_points(points.len()),
    )?;
    Ok((evidence, fourier_agreed))
}

/// `F{p}(-t/2π)` as an expression, or `None` when the transform has no rule for
/// this density.
fn fourier_route(dist: &Distribution, x: ExprId, t: ExprId, pool: &ExprPool) -> Option<ExprId> {
    use std::collections::HashMap;

    let p = super::charfun::fourier_ready_density(dist, x, pool)?;
    let xi = dists::fresh_var(&[p, t, x], pool);
    let transformed = crate::transform::fourier_transform(p, x, xi, pool).ok()?;
    // ξ = -t/2π is the whole of the convention bridge; see `charfun`'s docs.
    let two_pi = pool.mul(vec![pool.integer(2), super::pi(pool)]);
    let sub_xi = super::div(pool.mul(vec![pool.integer(-1), t]), two_pi, pool);
    let mut m = HashMap::new();
    m.insert(xi, sub_xi);
    Some(crate::kernel::subs(transformed, &m, pool))
}

/// Widest the oscillatory core is allowed to grow before the point is
/// abandoned. Each step doubles the half-width.
const CORE_WIDENINGS: u32 = 8;

/// Mass allowed to sit outside the core before it counts as not-a-bound.
const CORE_TAIL_TOL: f64 = 1e-22;

/// A compact interval holding all but `tail` of the distribution's mass.
///
/// # Why `φ` is not integrated the way everything else here is
///
/// `∫p(x)cos(tx)dx` over `(0, ∞)` on the logarithmic scale the rest of this
/// file uses is a bad pairing: the substitution `x = e^s` turns a fixed
/// oscillation period in `x` into one that shrinks like `e^{-s}`, so the rule
/// needs its finest level to resolve the middle of the range and takes
/// thousands of evaluations to get there. Measured at a minute per
/// characteristic function, for an integral whose answer is four digits of
/// arithmetic.
///
/// The compact core sidesteps it, and *rigorously* rather than by hoping:
/// `|cos| ≤ 1` and `|sin| ≤ 1`, so whatever the oscillatory integral does
/// outside `[A, B]` is bounded by the probability mass out there — and that
/// mass is `1 - ∫_A^B p`, which this function measures with the same rule.
/// The bound is then carried into the comparison rather than assumed
/// negligible. The rule has already been shown to reproduce `∫p = 1` over the
/// whole support (see [`prepare_rule`]), which is what makes `1 - ∫_A^B p` the
/// tail rather than an artefact.
fn oscillatory_core(
    density: ExprId,
    x: ExprId,
    rule: &ValidatedRule,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> Option<(Region, Float)> {
    if let Region::Interval(a, b) = &rule.region {
        // Already compact: nothing is outside it.
        return Some((Region::Interval(a.clone(), b.clone()), fl(VERIFY_PREC, 0.0)));
    }
    let w = rule.density_window.as_ref()?;
    let log_scale = matches!(rule.region, Region::Positive);
    let mut reach = fl(VERIFY_PREC, 1.0);
    for _ in 0..CORE_WIDENINGS {
        let span = Float::with_val(VERIFY_PREC, &w.scale * &reach);
        let (a, b) = if log_scale {
            // The support starts at 0 and tanh–sinh resolves an endpoint
            // singularity there, so there is no reason to clip the left end.
            (
                fl(VERIFY_PREC, 0.0),
                Float::with_val(
                    VERIFY_PREC,
                    Float::with_val(VERIFY_PREC, &w.centre + &span).exp(),
                ),
            )
        } else {
            (
                Float::with_val(VERIFY_PREC, &w.centre - &span),
                Float::with_val(VERIFY_PREC, &w.centre + &span),
            )
        };
        if a.is_finite() && b.is_finite() && a < b {
            let region = Region::Interval(a, b);
            let mut g = integrand_closure(density, x, binding, pool);
            if let QuadOutcome::Value { value, .. } =
                quadrature(&mut g, region.clone(), VERIFY_PREC)
            {
                let tail = Float::with_val(VERIFY_PREC, fl(VERIFY_PREC, 1.0) - &value).abs();
                if tail <= fl(VERIFY_PREC, CORE_TAIL_TOL) {
                    return Some((region, tail));
                }
            }
        }
        reach *= 2u32;
    }
    None
}

fn integrate_core(
    integrand: ExprId,
    x: ExprId,
    core: &Region,
    binding: &[(ExprId, Float)],
    pool: &ExprPool,
) -> Option<Float> {
    let mut g = integrand_closure(integrand, x, binding, pool);
    match quadrature(&mut g, core.clone(), VERIFY_PREC) {
        QuadOutcome::Value { value, .. } => Some(value),
        _ => None,
    }
}

/// [`compare`] with an extra absolute allowance — the truncated tail.
fn compare_within(
    claim: &Float,
    truth: &Float,
    slack: &Float,
    what: &str,
) -> Result<(), ProbError> {
    let diff = Float::with_val(VERIFY_PREC, claim - truth).abs();
    let allowed = {
        let base = fl(VERIFY_PREC, REL_TOL);
        let s = Float::with_val(VERIFY_PREC, slack * 4u32);
        if s > base {
            s
        } else {
            base
        }
    };
    if diff > allowed {
        return Err(ProbError::Unverified(UnverifiedReason::Disagreement(
            format!(
                "{what}: closed form {} vs quadrature {}",
                fmt(claim),
                fmt(truth)
            ),
        )));
    }
    Ok(())
}
