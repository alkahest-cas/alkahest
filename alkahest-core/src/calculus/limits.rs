//! Symbolic limits towards finite points or ±∞ via local expansions (`Series`),
//! L'Hôpital iterations, algebraic transforms, and the Gruntz comparability-graph
//! algorithm for exp-log combinations (V2-16/V2-17).

use crate::budget::BudgetError;
use crate::calculus::asymptotic::regularize_at_zero;
use crate::calculus::gruntz::try_gruntz;
use crate::calculus::series::{enter_coeff_ceiling, local_expansion, LocalExpansion};
use crate::diff::{diff, DiffError};
use crate::kernel::pool::POS_INFINITY_SYMBOL;
use crate::kernel::{subs, ExprData, ExprId, ExprPool};
use crate::poly::{poly_normal, RationalFunction};
use crate::simplify::{simplify, simplify_expanded};
use crate::SeriesError;
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;

/// Approach direction toward `point` (real-axis ordering).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LimitDirection {
    /// Ordinary two-sided limit.
    Bidirectional,
    /// Limits with `var > point` (approach from the right on the usual number line picture).
    Plus,
    /// Limits with `var < point`.
    Minus,
}

#[derive(Debug)]
pub enum LimitError {
    /// Sub-problem rejected by [`mod@crate::calculus::series`].
    Series(SeriesError),
    /// Derivative unavailable for L'Hôpital.
    Diff(DiffError),
    /// Odd-order pole requires a one-sided direction.
    NeedsOneSided,
    /// The search ran out of room: the L'Hôpital / recursion depth cap, the
    /// internal work ceiling ([`limit`]'s termination guard), or the ambient
    /// [`crate::budget`] — see [`last_budget_trip`] to tell a budget trip from
    /// the engine's own limits.
    DepthExceeded,
    /// No implemented rule applies (non-comparable growth, oscillation, …).
    Unsupported,
}

impl fmt::Display for LimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LimitError::Series(e) => write!(f, "{e}"),
            LimitError::Diff(e) => write!(f, "{e}"),
            LimitError::NeedsOneSided => {
                write!(
                    f,
                    "two-sided limit undefined at this pole; pass direction Plus or Minus"
                )
            }
            LimitError::DepthExceeded => write!(f, "limit refinement depth exceeded"),
            LimitError::Unsupported => write!(f, "limit could not be computed with current rules"),
        }
    }
}

impl std::error::Error for LimitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LimitError::Series(e) => Some(e),
            LimitError::Diff(e) => Some(e),
            _ => None,
        }
    }
}

impl crate::errors::AlkahestError for LimitError {
    fn code(&self) -> &'static str {
        match self {
            LimitError::Series(_) => "E-LIMIT-001",
            LimitError::Diff(_) => "E-LIMIT-002",
            LimitError::NeedsOneSided => "E-LIMIT-003",
            LimitError::DepthExceeded => "E-LIMIT-004",
            LimitError::Unsupported => "E-LIMIT-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            LimitError::Series(_) => {
                "increase truncation order indirectly by simplifying the expression, or rewrite using standard limits"
            }
            LimitError::Diff(_) => {
                "ensure primitives have differentiation rules, or simplify before taking the limit"
            }
            LimitError::NeedsOneSided => "use LimitDirection::Plus or Minus matching the desired one-sided approach",
            LimitError::DepthExceeded => {
                "try manual algebra (quotient form, cancellations) or split into simpler sub-expressions"
            }
            LimitError::Unsupported => {
                "limit could not be computed — try manual algebra, or the expression may involve oscillation or non-comparable growth not yet handled"
            }
        })
    }
}

impl From<SeriesError> for LimitError {
    fn from(e: SeriesError) -> Self {
        LimitError::Series(e)
    }
}

impl From<DiffError> for LimitError {
    fn from(e: DiffError) -> Self {
        LimitError::Diff(e)
    }
}

// ---------------------------------------------------------------------------
// Termination guard — cooperative budget checkpoints plus an internal work
// ceiling, so no search path in this engine can run unboundedly.
// ---------------------------------------------------------------------------

/// How many *new* expression nodes one top-level [`limit`] call may intern
/// before the engine gives up with [`LimitError::DepthExceeded`].
///
/// The pathological shape this bounds is repeated symbolic differentiation:
/// [`crate::calculus::series`] builds Taylor coefficients by differentiating
/// without re-simplifying, so an expression whose derivatives do not close
/// (nested radicals, in particular) grows by a constant factor per
/// coefficient. Thirty-two coefficients of `√(x²+x)` rewritten at `t → 0⁺`
/// is not a slow computation, it is an unfinishable one — the loop ran for
/// hours with no output. Counting interned nodes rather than iterations
/// catches that directly, is `O(1)` per checkpoint ([`ExprPool::len`] is a
/// lock-free counter), and is monotone, so no path can evade it.
///
/// Sized with an order of magnitude of headroom: the heaviest limit in the
/// Rust and Python suites interns ~9k nodes (`x·sin(1/x)` at 0; most are under
/// 300), while a runaway radical expansion reaches this ceiling in a few
/// hundred milliseconds.
const MAX_LIMIT_POOL_GROWTH: usize = 100_000;

thread_local! {
    /// `pool.len()` when the outermost [`limit`] call on this thread started.
    /// `None` outside any `limit` call.
    static WORK_BASELINE: Cell<Option<usize>> = const { Cell::new(None) };
    /// The [`BudgetError`] that tripped the most recent outermost [`limit`]
    /// call, if any — see [`last_budget_trip`].
    static BUDGET_TRIP: Cell<Option<BudgetError>> = const { Cell::new(None) };
}

/// RAII marker for the outermost [`limit`] frame on this thread.
///
/// [`limit`] re-enters itself (Gruntz sub-limits, growth comparisons), and the
/// work ceiling must bound the *whole* call rather than restart at every
/// re-entry, so only the outermost frame installs and clears the baseline.
struct WorkFrame {
    outermost: bool,
}

impl Drop for WorkFrame {
    fn drop(&mut self) {
        if self.outermost {
            WORK_BASELINE.with(|c| c.set(None));
        }
    }
}

fn enter_work_frame(pool: &ExprPool) -> WorkFrame {
    WORK_BASELINE.with(|c| {
        if c.get().is_some() {
            return WorkFrame { outermost: false };
        }
        c.set(Some(pool.len()));
        WorkFrame { outermost: true }
    })
}

/// `true` once this `limit` call has interned more than
/// [`MAX_LIMIT_POOL_GROWTH`] nodes.
fn work_exhausted(pool: &ExprPool) -> bool {
    WORK_BASELINE.with(|c| match c.get() {
        Some(base) => pool.len().saturating_sub(base) > MAX_LIMIT_POOL_GROWTH,
        None => false,
    })
}

/// The absolute `pool.len()` ceiling for the current `limit` call, for handing
/// to [`enter_coeff_ceiling`] so the Taylor-coefficient loop stops at the same
/// place this engine's own checkpoints do.
fn coeff_ceiling(pool: &ExprPool) -> usize {
    WORK_BASELINE.with(|c| {
        c.get()
            .unwrap_or_else(|| pool.len())
            .saturating_add(MAX_LIMIT_POOL_GROWTH)
    })
}

/// Cooperative checkpoint for the limit engine: honours [`crate::budget`] and
/// the internal work ceiling.
///
/// Placed on every path that can iterate or recurse — see [`limit_inner`],
/// [`canonical_polynomial_quotient_in_var`], [`try_expansion_limit`],
/// [`try_regularized_infinity_limit`] and [`crate::calculus::gruntz`].
pub(crate) fn checkpoint(pool: &ExprPool) -> Result<(), LimitError> {
    if let Err(e) = crate::budget::check() {
        BUDGET_TRIP.with(|c| c.set(Some(e)));
        return Err(LimitError::DepthExceeded);
    }
    if work_exhausted(pool) {
        return Err(LimitError::DepthExceeded);
    }
    Ok(())
}

/// The [`BudgetError`] that stopped the most recent outermost [`limit`] call on
/// this thread, or `None` if that call was not stopped by a budget.
///
/// [`LimitError`] is an exhaustive public enum, so it cannot grow a `Budget`
/// variant without a major semver break (the same constraint
/// [`mod@crate::integrate`] works around by encoding budget trips inside
/// `NotImplemented`). Limit budget trips are reported as
/// [`LimitError::DepthExceeded`] — an honest "gave up" — and this function
/// tells a caller *why* it gave up, so bindings can raise a dedicated
/// budget-exceeded error carrying the `E-BUDGET-*` code.
///
/// Cleared at the start of every outermost `limit` call, so it only ever
/// describes the call that just returned.
pub fn last_budget_trip() -> Option<BudgetError> {
    BUDGET_TRIP.with(|c| c.get())
}

/// `limit(expr, var, point, dir)` — see [`LimitDirection`].
///
/// `point` may be finite or [`ExprPool::pos_infinity`]. Limits at `-∞` use
/// `pool.mul(pool.integer(-1), pool.pos_infinity())`.
///
/// # Termination
///
/// The search is bounded: it honours [`crate::budget`] (wall clock, steps,
/// [`crate::budget::request_cancel`]) and, with no budget active, an internal
/// work ceiling. Either way an unsolvable case returns
/// [`LimitError::DepthExceeded`] rather than running unboundedly; use
/// [`last_budget_trip`] to tell the two apart.
pub fn limit(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
) -> Result<ExprId, LimitError> {
    let frame = enter_work_frame(pool);
    if frame.outermost {
        BUDGET_TRIP.with(|c| c.set(None));
    }
    // Bound the Taylor-coefficient loop in `series` for the whole call, not
    // just at the boundaries this module can see: a single `local_expansion`
    // at order 32 is one uninterruptible call from here.
    let _ceiling = enter_coeff_ceiling(coeff_ceiling(pool));

    limit_body(expr, var, point, direction, pool).map_err(|e| attribute_failure(e, pool))
}

/// Re-attribute a failed [`limit`] to the resource that actually stopped it.
///
/// Every rule in this engine turns a failed sub-problem into "this rule does
/// not apply" (`Err(_) => Ok(None)`), and the coefficient loop in
/// [`crate::calculus::series`] simply stops producing terms. So a call that ran
/// out of budget usually surfaces as `Unsupported` — "no rule worked" — which
/// is true but useless: it tells the caller to rewrite the problem when what
/// they need to do is raise the budget. Any failure that coincides with an
/// exhausted budget, a cancellation, or a blown work ceiling is reported as
/// [`LimitError::DepthExceeded`], with [`last_budget_trip`] carrying the
/// `E-BUDGET-*` cause when there was one.
fn attribute_failure(e: LimitError, pool: &ExprPool) -> LimitError {
    if BUDGET_TRIP.with(|c| c.get()).is_some() || work_exhausted(pool) {
        return LimitError::DepthExceeded;
    }
    if let Err(b) = crate::budget::check() {
        BUDGET_TRIP.with(|c| c.set(Some(b)));
        return LimitError::DepthExceeded;
    }
    e
}

fn limit_body(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
) -> Result<ExprId, LimitError> {
    let r = limit_inner(expr, var, point, direction, pool, 0)?;
    let r_simp = simplify(r, pool).value;
    let r_fold = fold_known_reals(r_simp, pool);
    let result = simplify(r_fold, pool).value;
    // A residual `0^{negative}` is not a value: it is what is left over when the
    // substitution `x ↦ 1/t`, `t → 0` never resolved.  Returning it produces
    // confident nonsense for limits that do not exist — `lim_{x→∞} sin x` came
    // back as `sin(0^{-1})` and `lim_{x→0} exp(1/x)` as `exp(0^{-1})`, neither
    // flagged as an error.  Report the honest failure instead.  Genuine
    // infinities use the canonical `∞` symbol and are unaffected.
    if contains_zero_to_negative_power(result, pool) {
        return Err(LimitError::Unsupported);
    }
    if approach_side_is_outside_the_domain(expr, var, point, direction, pool) {
        return Err(LimitError::Unsupported);
    }
    if numeric_evidence_contradicts(expr, var, point, direction, result, pool) {
        return Err(LimitError::Unsupported);
    }
    Ok(result)
}

/// True when `expr` takes no real value anywhere on the side the caller asked
/// about, so the one-sided limit does not exist over ℝ.
///
/// `lim_{x→0⁻} √x` came back as `0`. It is not that the value is hard to pin
/// down: `√x` is undefined at *every* point of every left neighbourhood of `0`,
/// so there is no sequence to take a limit along and the question has no answer
/// over the reals. Same for `lim_{x→1⁺} arccos x`. A `0` there is the kind of
/// answer a loop reasoning about domains of definition inherits and cannot
/// audit — it looks exactly like the (correct) `lim_{x→0⁺} √x = 0`.
///
/// The evidence required is positive and cheap:
///
/// * every sampled offset on the approach side evaluates to `NaN` — the
///   interpreter *ran* and the result was not a real number, as opposed to
///   returning `None` because it did not recognise the expression; and
/// * the mirror point on the opposite side evaluates to a finite real, which
///   witnesses that this expression is within the interpreter's vocabulary and
///   that the `NaN`s are therefore facts about the function's domain.
///
/// `±inf` deliberately does **not** count: `lim_{x→0⁻} 1/x = −∞` is a pole, not
/// a domain boundary, and the answer `−∞` is correct.
///
/// Two-sided limits are left alone. There the usual convention takes the limit
/// relative to the domain, under which `lim_{x→0} √x = 0` is defensible; a
/// caller who writes `dir="-"` has asked a question that convention does not
/// cover.
fn approach_side_is_outside_the_domain(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
) -> bool {
    let sign = match direction {
        LimitDirection::Plus => 1.0,
        LimitDirection::Minus => -1.0,
        LimitDirection::Bidirectional => return false,
    };
    // A polynomial is defined on the whole line; so is anything whose samples
    // would be meaningless because a second symbol is unbound.
    if is_polynomial_in(expr, var, pool) || has_free_symbol_besides(expr, var, pool) {
        return false;
    }
    let Some(at) = constant_f64(point, pool) else {
        return false;
    };

    let mut env: HashMap<ExprId, f64> = HashMap::with_capacity(1);
    let mut sample = |offset: f64| -> Option<f64> {
        env.insert(var, at + offset);
        crate::jit::eval_interp(expr, &env, pool)
    };

    for offset in APPROACH_OFFSETS {
        match sample(sign * offset) {
            Some(v) if v.is_nan() => {}
            // Evaluable and real, or not evaluable at all: no verdict.
            _ => return false,
        }
    }
    // The witness that the expression itself is evaluable.
    APPROACH_OFFSETS
        .iter()
        .any(|&offset| sample(-sign * offset).is_some_and(|v| v.is_finite()))
}

/// Offsets used to sample a function as it approaches a finite point.
///
/// Deliberately stops at `1e-4`: closer in, catastrophic cancellation in
/// expressions like `(cos x - 1)/x²` dominates the signal and the sampler
/// would start manufacturing disagreements that are artifacts of binary
/// floating point rather than facts about the function.
const APPROACH_OFFSETS: [f64; 4] = [1e-1, 1e-2, 1e-3, 1e-4];

/// A one-sided numeric estimate, kept only when the samples have settled.
struct SideEstimate {
    /// Value at the closest offset.
    value: f64,
    /// How far the estimate still moved over the last refinement — the scale
    /// below which a disagreement is not yet meaningful.
    movement: f64,
}

/// Sample `expr` approaching `at` from one side, returning an estimate only
/// when the samples converge.
///
/// `sign` is `+1.0` to approach from above, `-1.0` from below. Returns `None`
/// when the function cannot be evaluated, or when the samples are still moving
/// enough that no honest verdict can be drawn from them — an oscillating
/// integrand such as `x·sin(1/x)` must fall in the second bucket, so that this
/// check stays silent rather than guessing.
fn side_estimate(
    expr: ExprId,
    var: ExprId,
    at: f64,
    sign: f64,
    pool: &ExprPool,
) -> Option<SideEstimate> {
    let mut samples = Vec::with_capacity(APPROACH_OFFSETS.len());
    let mut env: HashMap<ExprId, f64> = HashMap::with_capacity(1);
    for offset in APPROACH_OFFSETS {
        env.insert(var, at + sign * offset);
        match crate::jit::eval_interp(expr, &env, pool) {
            Some(v) if v.is_finite() => samples.push(v),
            // A single unevaluable or non-finite sample is not evidence of
            // anything; it just means this offset landed on a hole.
            _ => continue,
        }
    }
    if samples.len() < 3 {
        return None;
    }
    let last = samples[samples.len() - 1];
    let prev = samples[samples.len() - 2];
    let movement = (last - prev).abs();
    let scale = 1.0 + last.abs();
    // Still moving by more than 1% of its own magnitude: not converged.
    if movement > 0.01 * scale {
        return None;
    }
    Some(SideEstimate {
        value: last,
        movement,
    })
}

/// True when numeric sampling clearly contradicts the symbolic `result`.
///
/// The symbolic machinery can return a confident value that the function never
/// approaches. `lim_{x→0} x/|x|` came back as `0` in all three directions, when
/// the one-sided limits are `∓1` and the two-sided limit does not exist —
/// a plausible finite number with nothing to distinguish it from a correct one.
/// (The algebraically identical `|x|/x` was refused, so argument order alone
/// decided whether the caller got a refusal or a wrong answer.)
///
/// This is a *refutation* check, not a verification one: it fires only on a
/// clear contradiction and stays silent whenever the evidence is weak, so it
/// can turn a wrong answer into a refusal but never a right answer into one.
/// Every guard below is a reason to say nothing.
fn numeric_evidence_contradicts(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    result: ExprId,
    pool: &ExprPool,
) -> bool {
    // Checked first, and in this order, because both are whole-expression
    // walks and polynomials are the common case in hot paths.
    //
    // A polynomial is continuous on the whole line: its limit at a finite point
    // is just its value there, and none of the failure modes this check hunts —
    // poles, branch cuts, sign discontinuities, one-sided divergence — can
    // occur. Sampling it can only confirm what substitution already settled.
    // `is_polynomial_in` also rejects any symbol other than `var`, so passing it
    // subsumes the free-parameter check below.
    if is_polynomial_in(expr, var, pool) {
        return false;
    }
    // A free parameter besides `var` makes the samples meaningless.
    if has_free_symbol_besides(expr, var, pool) {
        return false;
    }
    // Only finite approach points; `∞` is not a place to sample around.
    let Some(at) = constant_f64(point, pool) else {
        return false;
    };

    let claimed = constant_f64(result, pool);

    // Cheap probe before the full analysis. The convergence test below costs up
    // to eight evaluations, and the overwhelming majority of calls are limits
    // that are simply correct — one sample per relevant side is enough to see
    // that and leave. Escalating only on a whiff of disagreement keeps the
    // common path at two evaluations instead of eight.
    //
    // Skipping here can only make the check stay *silent*, never fire wrongly,
    // which is the direction a refutation check is allowed to be wrong in.
    if !probe_looks_suspicious(expr, var, at, direction, claimed, pool) {
        return false;
    }

    let left = side_estimate(expr, var, at, -1.0, pool);
    let right = side_estimate(expr, var, at, 1.0, pool);

    // Two-sided: settled but disagreeing sides mean the limit does not exist,
    // whatever value the symbolic route produced.
    if direction == LimitDirection::Bidirectional {
        if let (Some(l), Some(r)) = (&left, &right) {
            let tol = 1e-6 + 20.0 * (l.movement + r.movement);
            if (l.value - r.value).abs() > tol {
                return true;
            }
        }
    }

    // Any direction: compare the symbolic answer against the side(s) it claims
    // to describe. Only meaningful when the answer is itself a finite number —
    // `∞` and symbolic results are left alone.
    let Some(claimed) = claimed else {
        return false;
    };
    let sides: [&Option<SideEstimate>; 2] = match direction {
        LimitDirection::Plus => [&right, &None],
        LimitDirection::Minus => [&left, &None],
        LimitDirection::Bidirectional => [&left, &right],
    };
    for side in sides.into_iter().flatten() {
        let tol = 1e-6 * (1.0 + claimed.abs()) + 20.0 * side.movement;
        if (side.value - claimed).abs() > tol {
            return true;
        }
    }
    false
}

/// True when `expr` is a polynomial in `var` — sums and products of `var`,
/// constants, and non-negative integer powers thereof.
///
/// Conservative: anything it does not recognise (a `Func`, a negative or
/// non-integer exponent, a symbolic exponent) returns `false`, which merely
/// costs the caller the sampling it was trying to avoid.
fn is_polynomial_in(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    match pool.get(expr) {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
        ExprData::Symbol { .. } => false,
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().all(|&x| is_polynomial_in(x, var, pool)),
        ExprData::Pow { base, exp } => {
            matches!(pool.get(exp), ExprData::Integer(n) if n.0 >= 0)
                && is_polynomial_in(base, var, pool)
        }
        _ => false,
    }
}

/// One sample per relevant side, to decide whether the full convergence
/// analysis is worth running.
///
/// Returns `true` when the closest sample already disagrees with `claimed`
/// (or, for a two-sided limit with no numeric `claimed`, when the two sides
/// disagree with each other). A `false` here ends the check, so this is
/// deliberately biased toward escalating: a needless escalation costs six more
/// evaluations, while a missed one costs a silent error.
fn probe_looks_suspicious(
    expr: ExprId,
    var: ExprId,
    at: f64,
    direction: LimitDirection,
    claimed: Option<f64>,
    pool: &ExprPool,
) -> bool {
    // Two offsets a hundredfold apart, so the *trend* is visible.
    //
    // Comparing a single sample against `claimed` does not work: a function
    // approaching its limit is legitimately still some distance away at a
    // finite offset — `(x⁸−1)/(x−1)` is 8.0028 at `x = 1.0001`, not 8 — so an
    // absolute-tolerance probe escalates for essentially every non-constant
    // function and saves nothing. What distinguishes a correct limit is that
    // the samples *close in on* the claimed value as the offset shrinks.
    let far = APPROACH_OFFSETS[1];
    let near = APPROACH_OFFSETS[APPROACH_OFFSETS.len() - 1];
    // One map, rewritten per sample. Allocating a fresh `HashMap` per
    // evaluation costs more than the evaluation does on small expressions.
    let mut env: HashMap<ExprId, f64> = HashMap::with_capacity(1);
    let mut sample = |sign: f64, offset: f64| -> Option<f64> {
        env.insert(var, at + sign * offset);
        crate::jit::eval_interp(expr, &env, pool).filter(|v| v.is_finite())
    };

    let mut side_is_suspicious = |sign: f64| -> bool {
        let (Some(f_far), Some(f_near)) = (sample(sign, far), sample(sign, near)) else {
            // Cannot see the trend here. The full analysis needs three samples
            // on a side, so it would not reach a verdict either — stay silent.
            return false;
        };
        match claimed {
            // Converging toward the claimed value: nothing to investigate. The
            // 0.75 factor is deliberately lenient — linear convergence shrinks
            // the gap a hundredfold over this range, so anything genuinely
            // heading for `claimed` clears it easily, while `x/|x|` (gap 1 at
            // both offsets) does not.
            Some(c) => (f_near - c).abs() > 0.75 * (f_far - c).abs(),
            None => false,
        }
    };

    let left_bad = direction != LimitDirection::Plus && side_is_suspicious(-1.0);
    let right_bad = direction != LimitDirection::Minus && side_is_suspicious(1.0);
    if left_bad || right_bad {
        return true;
    }

    // No numeric answer to compare against: only the two-sided
    // does-not-exist check can fire, and it needs both sides.
    if claimed.is_none() && direction == LimitDirection::Bidirectional {
        if let (Some(l), Some(r)) = (sample(-1.0, near), sample(1.0, near)) {
            return (l - r).abs() > 1e-6 * (1.0 + l.abs().max(r.abs()));
        }
    }
    false
}

/// Evaluate a closed-form expression to `f64`, or `None` if it is not a
/// constant this interpreter can reduce to a finite number.
fn constant_f64(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    let env = HashMap::new();
    crate::jit::eval_interp(expr, &env, pool).filter(|v| v.is_finite())
}

/// True when `expr` mentions a symbol other than `var`.
fn has_free_symbol_besides(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return false;
    }
    match pool.get(expr) {
        ExprData::Symbol { .. } => true,
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            xs.iter().any(|&x| has_free_symbol_besides(x, var, pool))
        }
        ExprData::Pow { base, exp } => {
            has_free_symbol_besides(base, var, pool) || has_free_symbol_besides(exp, var, pool)
        }
        ExprData::Func { args, .. } => args.iter().any(|&a| has_free_symbol_besides(a, var, pool)),
        _ => false,
    }
}

/// True when `expr` contains a `0^n` node with `n` a negative integer — the
/// unresolved-pole artifact described in [`limit`].
fn contains_zero_to_negative_power(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let zero_base = matches!(pool.get(base), ExprData::Integer(n) if n.0 == 0);
            let negative_exp = match pool.get(exp) {
                ExprData::Integer(n) => n.0 < 0,
                ExprData::Rational(r) => r.0 < 0,
                _ => false,
            };
            (zero_base && negative_exp)
                || contains_zero_to_negative_power(base, pool)
                || contains_zero_to_negative_power(exp, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            xs.iter().any(|&x| contains_zero_to_negative_power(x, pool))
        }
        ExprData::Func { args, .. } => args
            .iter()
            .any(|&a| contains_zero_to_negative_power(a, pool)),
        _ => false,
    }
}

/// `(g^m)^n ↦ g^{m n}` when `m,n ∈ ℤ`, so substitutions like `(1/t)^k` become `t^{-k}` Laurent heads.
fn flatten_nested_integer_pow(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let base = flatten_nested_integer_pow(base, pool);
            let exp_fl = flatten_nested_integer_pow(exp, pool);
            if let (
                ExprData::Pow {
                    base: b2,
                    exp: inner_exp,
                },
                ExprData::Integer(outer_e),
            ) = (pool.get(base), pool.get(exp_fl))
            {
                if let ExprData::Integer(inner_e) = pool.get(inner_exp) {
                    let prod = inner_e.0.clone() * outer_e.0.clone();
                    return pool.pow(flatten_nested_integer_pow(b2, pool), pool.integer(prod));
                }
            }
            pool.pow(base, exp_fl)
        }
        ExprData::Mul(xs) => pool.mul(
            xs.iter()
                .map(|x| flatten_nested_integer_pow(*x, pool))
                .collect(),
        ),
        ExprData::Add(xs) => pool.add(
            xs.iter()
                .map(|x| flatten_nested_integer_pow(*x, pool))
                .collect(),
        ),
        ExprData::Func { name, args } => {
            let na: Vec<ExprId> = args
                .iter()
                .map(|a| flatten_nested_integer_pow(*a, pool))
                .collect();
            pool.func(name.clone(), na)
        }
        _ => expr,
    }
}

/// After ``x ↦ 1/t``, common forms are ``Mul(numer, denom^{-1})`` with ``Pow(t,-1)``
/// sprinkled through both.  Clear those poles by multiplying by ``t^k`` until
/// numerator and denominator describe an honest polynomial quotient in ``t``.
fn canonical_polynomial_quotient_in_var(
    expr: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LimitError> {
    let (n_raw, d_raw) = numerator_denominator(expr, pool);
    let has_trivial_denom = d_raw == pool.integer(1_i32);
    // When d_raw == 1 the expr might still have negative powers of t in a sum (e.g. 1 + t^{-1}).
    // Skip k=0 in that case to avoid an infinite loop, but still try higher k values.
    for k in 0_i64..=40 {
        if has_trivial_denom && k == 0 {
            continue;
        }
        // Each pass runs `simplify_expanded` twice on the whole expression, so
        // a 41-iteration sweep over a large input is long enough to need to be
        // interruptible even though the loop count itself is bounded.
        checkpoint(pool)?;
        let tk = pool.pow(t, pool.integer(k));
        let n = simplify_expanded(pool.mul(vec![tk, n_raw]), pool).value;
        let d = simplify_expanded(pool.mul(vec![tk, d_raw]), pool).value;
        let (n, d) = match (poly_normal(n, vec![t], pool), poly_normal(d, vec![t], pool)) {
            (Ok(nn), Ok(dd)) => (nn, dd),
            _ => continue,
        };
        if let Ok(rf) = RationalFunction::from_symbolic(n, d, vec![t], pool) {
            let nx = rf.numer.to_expr(pool);
            let dx = rf.denom.to_expr(pool);
            return Ok(
                simplify(pool.mul(vec![nx, pool.pow(dx, pool.integer(-1_i32))]), pool).value,
            );
        }
    }
    Ok(expr)
}

fn limit_inner(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<ExprId, LimitError> {
    const MAX_DEPTH: u32 = 48;
    const SERIES_ORDER: u32 = 32;
    if depth > MAX_DEPTH {
        return Err(LimitError::DepthExceeded);
    }
    checkpoint(pool)?;

    if !depends_on(expr, var, pool) {
        if substitution_is_singular(expr, pool) {
            return Err(LimitError::Unsupported);
        }
        return Ok(fold_known_reals(simplify(expr, pool).value, pool));
    }

    if let Some(r) = try_special_function_limits(expr, var, point, direction, pool)? {
        return Ok(r);
    }

    // The special-function basis the integrator emits over. Placed with the
    // other table lookups and ahead of the `x ↦ 1/t` substitution, which would
    // hand `erf(1/t)` to a Taylor expansion that cannot see through it.
    if let Some(r) = try_special_basis_limit(expr, var, point, direction, pool, depth)? {
        return Ok(r);
    }
    if let Some(r) = try_special_basis_algebra(expr, var, point, direction, pool, depth)? {
        return Ok(r);
    }

    // Indeterminate power f^g (1^∞, 0^0, ∞^0): rewrite to exp(g·log f).
    // Runs for finite points as well as ±∞ so textbook forms like
    // `(1+x)^(1/x) → e` as `x → 0` are not lost to the `1^anything → 1` fold.
    if let Some(r) = try_indeterminate_power(expr, var, point, direction, pool, depth)? {
        return Ok(r);
    }

    // Gruntz algorithm — best for exp/log expressions at +∞ (runs before the 1/t substitution
    // so the exp structure is still visible in the original variable).
    if is_pos_infinity(point, pool) {
        if let Some(r) = try_gruntz(expr, var, pool)? {
            return Ok(r);
        }
    }

    // Leading-order route at ±∞ for algebraic/analytic scales, tried before the
    // plain `x ↦ 1/t` substitution below: that substitution hands the result to
    // a Taylor expansion, which cannot see through a radical and instead
    // differentiates it thirty-two times.
    if is_pos_infinity(point, pool) || is_neg_infinity(point, pool) {
        let toward_pos = is_pos_infinity(point, pool);
        if let Some(r) = try_regularized_infinity_limit(expr, var, toward_pos, pool)? {
            return Ok(r);
        }
    }

    if is_pos_infinity(point, pool) {
        let t = pool.symbol("__lt_inf", crate::kernel::Domain::Real);
        let inv_t = pool.pow(t, pool.integer(-1_i32));
        let mut m = HashMap::new();
        m.insert(var, inv_t);
        let after_subs = subs(expr, &m, pool);
        let after_flatten = flatten_nested_integer_pow(after_subs, pool);
        let after_canon = canonical_polynomial_quotient_in_var(after_flatten, t, pool)?;
        let e2 = simplify(after_canon, pool).value;
        return limit_inner(
            e2,
            t,
            pool.integer(0_i32),
            LimitDirection::Plus,
            pool,
            depth + 1,
        );
    }

    if is_neg_infinity(point, pool) {
        let t = pool.symbol("__lt_ninf", crate::kernel::Domain::Real);
        let rep = pool.mul(vec![
            pool.integer(-1_i32),
            pool.pow(t, pool.integer(-1_i32)),
        ]);
        let mut m = HashMap::new();
        m.insert(var, rep);
        let canon = canonical_polynomial_quotient_in_var(
            flatten_nested_integer_pow(subs(expr, &m, pool), pool),
            t,
            pool,
        )?;
        let e2 = simplify(canon, pool).value;
        let substituted = limit_inner(
            e2,
            t,
            pool.integer(0_i32),
            LimitDirection::Plus,
            pool,
            depth + 1,
        );
        // **Reflection fallback**, on the unresolved path only.
        //
        // `lim_{x→−∞} f(x) = lim_{y→+∞} f(−y)` is an identity, and the `+∞`
        // side of this function is strictly better equipped: the Gruntz
        // algorithm runs there and nowhere else. The `x ↦ −1/t` substitution
        // above turns `x·exp(−x²)` into `−exp(−t⁻²)/t`, which has no Taylor
        // expansion at `0` — the recursive call comes back with a residual
        // `0^{negative}` rather than an error, and `limit_body` is what
        // eventually rejects it. Gruntz answers `lim_{y→+∞} −y·exp(−y²) = 0`
        // immediately, and that limit is exactly what
        // `∫_{-∞}^{∞} x²·exp(−x²) dx` needs at its lower bound.
        //
        // "Unresolved" therefore has to mean the same thing `limit_body` means
        // by it: an error, a residual `0^{negative}`, or a value that still
        // mentions the substitution variable. In every one of those cases the
        // original outcome is returned unchanged if the reflection also fails,
        // so nothing that resolves today can resolve differently.
        let unresolved = match &substituted {
            Err(_) => true,
            Ok(v) => contains_zero_to_negative_power(*v, pool) || depends_on(*v, t, pool),
        };
        if unresolved {
            if let Ok(v) = reflected_limit(expr, var, direction, pool, depth) {
                return Ok(v);
            }
        }
        return substituted;
    }

    if let Some(r) = try_direct_substitution(expr, var, point, pool) {
        return Ok(r);
    }

    if let Some(r) = try_x_log_x_at_zero(expr, var, point, direction, pool, depth)? {
        return Ok(r);
    }

    if let Some(r) = try_lhopital(expr, var, point, direction, pool, depth)? {
        return Ok(r);
    }

    if let Some(r) = try_expansion_limit(expr, var, point, direction, pool, SERIES_ORDER)? {
        return Ok(r);
    }

    Err(LimitError::Unsupported)
}

/// True when `expr` contains an algebraic (non-integer-power) head — `sqrt`,
/// `cbrt`, or a `Pow` with a fractional exponent.
fn contains_radical(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            name == "sqrt" || name == "cbrt" || args.iter().any(|&a| contains_radical(a, pool))
        }
        ExprData::Pow { base, exp } => {
            matches!(pool.get(exp), ExprData::Rational(_))
                || contains_radical(base, pool)
                || contains_radical(exp, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| contains_radical(x, pool)),
        _ => false,
    }
}

/// Leading-order limit of an *algebraic* expression at `±∞`.
///
/// Substitutes `x ↦ ±1/t` (`t → 0⁺`), regularizes the result structurally as
/// `f(±1/t) = t^v · u(t)` with `u` analytic and non-vanishing at `t = 0`
/// ([`regularize_at_zero`], the same valuation calculus
/// [`crate::calculus::asymptotic`] expands with), and reads the limit off the
/// leading Taylor coefficient of `u`.
///
/// The generic route below — substitute, clear poles, Taylor-expand the whole
/// thing — cannot see through a radical: for `√(x²+x) − x` it hands
/// `√(t⁻² + t⁻¹) − t⁻¹` to a 32-term Taylor expansion, and each successive
/// derivative of a nested radical is a constant factor larger than the last,
/// so the call never returns. Pulling the pole out of the radical first
/// (`√(t⁻²(1+t)) = t⁻¹√(1+t)`) leaves `t⁻¹·(√(1+t) − 1)`, whose analytic part
/// has bounded derivatives — the expansion is then immediate and exact, and
/// the ∞−∞ cancellation resolves to `1/2` instead of hanging.
///
/// Restricted to expressions that actually contain a radical: everything else
/// is already served by the existing routes, and this keeps their answers
/// untouched.
fn try_regularized_infinity_limit(
    expr: ExprId,
    var: ExprId,
    toward_pos: bool,
    pool: &ExprPool,
) -> Result<Option<ExprId>, LimitError> {
    // Escalated rather than fixed: only the first nonzero coefficient of `u`
    // decides the limit, and every further coefficient costs one more symbolic
    // derivative. Stopping at the first order that resolves keeps the common
    // case at three derivatives instead of thirty-two.
    const ORDERS: [u32; 3] = [4, 10, 24];

    if !contains_radical(expr, pool) {
        return Ok(None);
    }

    // `Domain::Positive`: the substituted variable approaches 0 from above, and
    // `regularize_at_zero`'s `(t^v·u)^e = t^{v·e}·u^e` step is only valid for
    // `t > 0`.
    let t = pool.symbol("__lt_reg", crate::kernel::Domain::Positive);
    let inv_t = pool.pow(t, pool.integer(-1_i32));
    let rep = if toward_pos {
        inv_t
    } else {
        pool.mul(vec![pool.integer(-1_i32), inv_t])
    };
    let mut m = HashMap::new();
    m.insert(var, rep);
    let f_of_t = simplify(subs(expr, &m, pool), pool).value;

    let Some((val, analytic)) = regularize_at_zero(f_of_t, t, pool) else {
        return Ok(None);
    };
    let Ok(val) = i32::try_from(val) else {
        return Ok(None);
    };

    let zero = pool.integer(0_i32);
    for order in ORDERS {
        checkpoint(pool)?;
        let Ok(exp) = local_expansion(analytic, t, zero, order, pool) else {
            return Ok(None);
        };
        let LocalExpansion {
            valuation,
            coeffs,
            h_expr,
        } = exp;
        let Some(total) = val.checked_add(valuation) else {
            return Ok(None);
        };
        let shifted = LocalExpansion {
            valuation: total,
            coeffs,
            h_expr,
        };
        // `t → 0⁺`, so even an odd-order pole has a determinate sign.
        if let Some(r) = expansion_to_limit(shifted, pool, LimitDirection::Plus)? {
            return Ok(Some(r));
        }
    }
    Ok(None)
}

fn try_x_log_x_at_zero(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<Option<ExprId>, LimitError> {
    if direction == LimitDirection::Minus {
        return Ok(None);
    }
    if !matches!(pool.get(point), ExprData::Integer(n) if n.0 == 0) {
        return Ok(None);
    }
    let ExprData::Mul(args) = pool.get(expr) else {
        return Ok(None);
    };
    if args.len() != 2 {
        return Ok(None);
    }
    let (a, b) = (args[0], args[1]);
    let log_of_var = |u: ExprId| {
        matches!(
            pool.get(u),
            ExprData::Func { name, args: av } if name == "log" && av.len() == 1 && av[0] == var
        )
    };
    let is_var = |u: ExprId| u == var;
    let ok = (is_var(a) && log_of_var(b)) || (is_var(b) && log_of_var(a));
    if !ok {
        return Ok(None);
    }
    // L'Hôpital on log(x) / x^{-1}: (1/x) / (-1/x^2) = -x  → 0 as x→0+.
    let f = pool.func("log", vec![var]);
    let g = pool.pow(var, pool.integer(-1_i32));
    let fp = diff(f, var, pool)?.value;
    let gp = diff(g, var, pool)?.value;
    let ratio = rational_quotient(fp, gp, pool);
    Ok(Some(limit_inner(
        ratio,
        var,
        point,
        LimitDirection::Plus,
        pool,
        depth + 1,
    )?))
}

/// Detect an indeterminate power `base^exp` as `var → ±∞` and rewrite it to
/// `exp(exp · log(base))`, feeding that through the recursive limit machinery
/// (Gruntz collects the resulting `exp(h)` and gets the right answer).
///
/// Only the genuinely indeterminate exponential forms are rewritten:
///   * `1^∞`  (base → 1, exp → ±∞)
///   * `∞^0`  (base → ±∞, exp → 0)
///   * `0^0`  (base → 0, exp → 0)  — only when `base` is structurally positive
///
/// Non-indeterminate powers (e.g. `2^x → ∞`, `x^2 → ∞`, or `base → c ≠ 1` with
/// `exp → ∞`) are left untouched so the existing fast paths still apply and no
/// new silent-wrong answers are introduced.  `log(base)` is only formed when we
/// can establish `base > 0` near the limit.
fn try_indeterminate_power(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<Option<ExprId>, LimitError> {
    let ExprData::Pow { base, exp } = pool.get(expr) else {
        return Ok(None);
    };
    // A constant base (e.g. exp(...) form is already handled, and 2^x is not
    // indeterminate) — only proceed when the base genuinely varies with `var`.
    if !depends_on(base, var, pool) {
        return Ok(None);
    }

    // Limits of the base and exponent (independently).
    let base_lim = match limit_inner(base, var, point, direction, pool, depth + 1) {
        Ok(b) => b,
        Err(_) => return Ok(None),
    };

    // When `base → 1`, the classic `1^∞` rewrite `exp(g·log f)` is licensed even
    // if `lim g` itself needs a one-sided approach (e.g. `(1+x)^(1/x)` as
    // `x → 0`): `lim (g·log f) = lim log(1+x)/x = 1` exists bidirectionally.
    if is_one_like(base_lim, pool) {
        let log_base = pool.func("log", vec![base]);
        let inner = simplify(pool.mul(vec![exp, log_base]), pool).value;
        if let Ok(inner_lim) = limit_inner(inner, var, point, direction, pool, depth + 1) {
            if is_pos_infinity(inner_lim, pool) {
                return Ok(Some(pool.pos_infinity()));
            }
            if is_neg_infinity(inner_lim, pool) {
                return Ok(Some(pool.integer(0_i32)));
            }
            let result = simplify(pool.func("exp", vec![inner_lim]), pool).value;
            return Ok(Some(result));
        }
    }

    let exp_lim = match limit_inner(exp, var, point, direction, pool, depth + 1) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };

    let base_is_one = is_one_like(base_lim, pool);
    let base_is_zero = is_zero_like(base_lim, pool);
    let base_is_inf = is_pos_infinity(base_lim, pool) || is_neg_infinity(base_lim, pool);
    let exp_is_zero = is_zero_like(exp_lim, pool);
    let exp_is_inf = is_pos_infinity(exp_lim, pool) || is_neg_infinity(exp_lim, pool);

    // Classify the indeterminate exponential forms.
    let indeterminate = (base_is_one && exp_is_inf)            // 1^∞
        || (base_is_inf && exp_is_zero)                       // ∞^0
        || (base_is_zero && exp_is_zero); // 0^0
    if !indeterminate {
        return Ok(None);
    }

    // `log(base)` must be valid: require base > 0 near the limit.  base → 1 or
    // base → +∞ are positive; base → 0 only qualifies if structurally positive.
    let base_positive = base_is_one
        || is_pos_infinity(base_lim, pool)
        || (base_is_zero && structurally_positive(base, pool));
    if !base_positive {
        return Ok(None);
    }

    // Rewrite f^g → exp(g · log f).  Compute the inner limit `L = lim(g · log f)`
    // via the existing (correct) machinery, then map it through exp:
    //   L finite → exp(L),   L = +∞ → +∞,   L = -∞ → 0.
    // Computing L directly (rather than recursing on `exp(g·log f)`) avoids the
    // Gruntz `exp(finite)` path, which only retains the leading order and would
    // give e.g. exp(1) for (1+2/x)^x instead of exp(2).
    let log_base = pool.func("log", vec![base]);
    let inner = simplify(pool.mul(vec![exp, log_base]), pool).value;
    let inner_lim = match limit_inner(inner, var, point, direction, pool, depth + 1) {
        Ok(l) => l,
        Err(_) => return Ok(None),
    };
    if is_pos_infinity(inner_lim, pool) {
        return Ok(Some(pool.pos_infinity()));
    }
    if is_neg_infinity(inner_lim, pool) {
        return Ok(Some(pool.integer(0_i32)));
    }
    // Finite inner limit ⇒ exp(L).
    let result = simplify(pool.func("exp", vec![inner_lim]), pool).value;
    Ok(Some(result))
}

/// Conservative structural test that `e > 0` everywhere it is defined — used to
/// license `log(e)` for `0^0` rewrites.  `1 + h` with positive constant part,
/// positive constants, even powers, and products/sums of positives qualify.
fn structurally_positive(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Integer(n) => n.0 > 0,
        ExprData::Rational(r) => r.0 > 0,
        ExprData::Func { name, .. } if name == "exp" || name == "cosh" => true,
        ExprData::Pow { base, exp } => {
            if let ExprData::Integer(n) = pool.get(exp) {
                if n.0.clone() % 2 == 0 {
                    return true;
                }
            }
            structurally_positive(base, pool)
        }
        ExprData::Mul(xs) => xs.iter().all(|x| structurally_positive(*x, pool)),
        _ => false,
    }
}

fn try_special_function_limits(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
) -> Result<Option<ExprId>, LimitError> {
    let ExprData::Func { name, args } = pool.get(expr) else {
        return Ok(None);
    };
    if args.len() != 1 || args[0] != var {
        return Ok(None);
    }
    match name.as_str() {
        "exp" => {
            if is_pos_infinity(point, pool) {
                return Ok(Some(pool.pos_infinity()));
            }
            if is_neg_infinity(point, pool) {
                return Ok(Some(pool.integer(0_i32)));
            }
            if matches!(pool.get(point), ExprData::Integer(n) if n.0 == 0) {
                return Ok(Some(pool.integer(1_i32)));
            }
        }
        "log" => {
            if is_pos_infinity(point, pool) {
                return Ok(Some(pool.pos_infinity()));
            }
            if matches!(pool.get(point), ExprData::Integer(n) if n.0 == 0) {
                if direction == LimitDirection::Plus {
                    return Ok(Some(neg_infinity(pool)));
                }
                return Err(LimitError::NeedsOneSided);
            }
        }
        _ => {}
    }
    Ok(None)
}

/// Limits at `±∞` of the special functions the integrator now **emits**.
///
/// # Why this exists
///
/// `∫exp(−x²/2) dx` has been answered as `1.2533…·erf(0.7071…·x)` since the
/// special-function emitter landed, but `∫_{-∞}^{∞} exp(−x²/2) dx` still
/// refused: the fundamental theorem needs `lim_{x→±∞} F`, `limit` had no rule
/// for `erf`, and `eval_bound` (correctly) will not substitute an unevaluated
/// limit into `F(b) − F(a)`. The indefinite answer was useless to the definite
/// question. Every entry below is the standard value of a function this
/// codebase already defines to a pinned convention — DLMF §6.2 for the
/// exponential-integral family, DLMF §7.2(iii) for the normalised Fresnel
/// integrals — not a new claim.
///
/// # The argument is taken to its own limit first
///
/// [`try_special_function_limits`] only fires on `f(var)` literally. That is
/// too narrow here, because the emitter's answers carry a *scaled* argument
/// (`erf(0.7071·x)`, `Si(2·x)`): the limit of the argument is computed
/// recursively, and only a `±∞` there is dispatched on. A finite inner limit is
/// left to [`try_direct_substitution`], which is continuity and needs no table.
///
/// # `Ci`, `Chi` and `li` are `+∞` only
///
/// All three have a branch cut along the negative reals — `Ci(−x) = Ci(x) ± iπ`
/// — so they have **no real value** for a negative argument, which is exactly
/// why [`crate::primitive`]'s kernels refuse there. There is therefore no real
/// limit at `−∞` to report, and returning one would be inventing a real part of
/// a complex number. `None` here surfaces as a refusal upstream, which is the
/// honest answer.
fn try_special_basis_limit(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<Option<ExprId>, LimitError> {
    let ExprData::Func { name, args } = pool.get(expr) else {
        return Ok(None);
    };
    if args.len() != 1 {
        return Ok(None);
    }
    if !matches!(
        name.as_str(),
        "erf" | "erfc" | "Si" | "Ci" | "Shi" | "Chi" | "Ei" | "li" | "fresnels" | "fresnelc"
    ) {
        return Ok(None);
    }
    let Some(toward_pos) = argument_runs_to_infinity(args[0], var, point, direction, pool, depth)
    else {
        return Ok(None);
    };

    let half_pi = || {
        pool.mul(vec![
            pool.symbol("pi", crate::kernel::Domain::Real),
            pool.rational(1, 2),
        ])
    };
    let value = match (name.as_str(), toward_pos) {
        // erf(±∞) = ±1, erfc = 1 − erf.
        ("erf", true) => pool.integer(1_i32),
        ("erf", false) => pool.integer(-1_i32),
        ("erfc", true) => pool.integer(0_i32),
        ("erfc", false) => pool.integer(2_i32),
        // Si is odd with Si(∞) = π/2 (DLMF 6.2.9); Ci(∞) = 0 (DLMF 6.2.11).
        ("Si", true) => half_pi(),
        ("Si", false) => pool.mul(vec![pool.integer(-1_i32), half_pi()]),
        ("Ci", true) => pool.integer(0_i32),
        // Shi is odd and grows like eˣ/x; Chi and Ei diverge at +∞ and Ei
        // decays to 0 at −∞ (DLMF 6.2.5: Ei(x) = ⨍_{-∞}^{x} eᵗ/t dt).
        ("Shi" | "Chi" | "Ei" | "li", true) => pool.pos_infinity(),
        ("Shi", false) => neg_infinity(pool),
        ("Ei", false) => pool.integer(0_i32),
        // Normalised Fresnel integrals: odd, with S(∞) = C(∞) = 1/2.
        ("fresnels" | "fresnelc", true) => pool.rational(1, 2),
        ("fresnels" | "fresnelc", false) => pool.rational(-1, 2),
        // `Ci`, `Chi`, `li` at −∞: no real value — see the doc comment.
        _ => return Ok(None),
    };
    Ok(Some(value))
}

/// Which infinity the argument of a special function runs to, or `None`.
///
/// The recursive [`limit_inner`] call is the general answer and is tried first.
/// The **linear fallback** behind it is not redundant: `limit` cannot currently
/// resolve `lim_{x→∞} 0.7071·x` at all (`lim_{x→∞} 2·x` is fine — the
/// difference is that a `Float` coefficient makes `poly_normal` decline, so the
/// `x ↦ 1/t` route hands `0.7071·t⁻¹` to a series expansion with nothing to say
/// about it). That is a pre-existing gap in the engine, unrelated to special
/// functions, but it lands squarely on this path because the emitted
/// antiderivatives carry exactly such coefficients: `∫exp(−x²/2) dx` is
/// `1.2533…·erf(0.7071…·x)`. Reading `c·x + d` off structurally, with `c` a
/// non-zero finite `f64` and `d` a finite constant, covers it without touching
/// the general engine.
///
/// A symbolic coefficient (`erf(√p·x)`) is **not** covered: its sign is a
/// question about `p`, and this function does not answer questions about `p`.
fn argument_runs_to_infinity(
    arg: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Option<bool> {
    if let Ok(inner) = limit_inner(arg, var, point, direction, pool, depth + 1) {
        if is_pos_infinity(inner, pool) {
            return Some(true);
        }
        if is_neg_infinity(inner, pool) {
            return Some(false);
        }
    }
    let point_pos = if is_pos_infinity(point, pool) {
        true
    } else if is_neg_infinity(point, pool) {
        false
    } else {
        return None;
    };
    let c = linear_coefficient_f64(arg, var, pool)?;
    Some((c > 0.0) == point_pos)
}

/// `Some(c)` when `arg` is `c·var + d` with `c` a non-zero finite `f64` and `d`
/// a finite `var`-free constant.
fn linear_coefficient_f64(arg: ExprId, var: ExprId, pool: &ExprPool) -> Option<f64> {
    if arg == var {
        return Some(1.0);
    }
    if !depends_on(arg, var, pool) {
        return None;
    }
    match pool.get(arg) {
        ExprData::Mul(xs) => {
            let mut coeff = 1.0_f64;
            let mut seen_var = false;
            for x in xs {
                if x == var {
                    if seen_var {
                        return None;
                    }
                    seen_var = true;
                    continue;
                }
                if depends_on(x, var, pool) {
                    return None;
                }
                coeff *= constant_f64(x, pool).filter(|v| v.is_finite())?;
            }
            (seen_var && coeff != 0.0 && coeff.is_finite()).then_some(coeff)
        }
        ExprData::Add(xs) => {
            let mut coeff: Option<f64> = None;
            for x in xs {
                if depends_on(x, var, pool) {
                    if coeff.is_some() {
                        return None;
                    }
                    coeff = Some(linear_coefficient_f64(x, var, pool)?);
                } else {
                    constant_f64(x, pool).filter(|v| v.is_finite())?;
                }
            }
            coeff
        }
        _ => None,
    }
}

/// Name prefix of every variable this module substitutes in for the original
/// one (`__lt_inf`, `__lt_ninf`, `__lt_refl`).  Seeing one in a *result* means
/// the sub-problem it was introduced for was never solved.
const LIMIT_SUBSTITUTION_PREFIX: &str = "__lt_";

/// The heads [`try_special_basis_limit`] has a table for.
const SPECIAL_BASIS_HEADS: [&str; 10] = [
    "erf", "erfc", "Si", "Ci", "Shi", "Chi", "Ei", "li", "fresnels", "fresnelc",
];

/// `true` when `expr` mentions one of [`SPECIAL_BASIS_HEADS`].
fn mentions_special_basis(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            SPECIAL_BASIS_HEADS.contains(&name.as_str())
                || args.iter().any(|&a| mentions_special_basis(a, pool))
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| mentions_special_basis(a, pool))
        }
        ExprData::Pow { base, exp } => {
            mentions_special_basis(base, pool) || mentions_special_basis(exp, pool)
        }
        _ => false,
    }
}

/// `true` when a computed limit is an ordinary finite value — no `∞`, and none
/// of this module's internal substitution variables, anywhere.
///
/// The second half is not hypothetical.  [`limit_inner`] hands a `±∞` point to
/// a `x ↦ ±1/t` substitution and returns whatever the recursive call on `t`
/// produced; when no rule fires, that is the substituted expression itself,
/// still written in `t`.  `limit_body` rejects the one shape of that it knows
/// (`0^{negative}`) and `eval_bound` rejects a value still mentioning the
/// *original* variable, but neither looks for `__lt_inf`.  A rule that
/// *combines* child limits — as [`try_special_basis_algebra`] does — must not
/// be the thing that launders an unsolved sub-problem into a finite-looking
/// product.
fn is_finite_limit_value(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Symbol { name, .. } => {
            name != POS_INFINITY_SYMBOL && !name.starts_with(LIMIT_SUBSTITUTION_PREFIX)
        }
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().all(|&a| is_finite_limit_value(a, pool))
        }
        ExprData::Pow { base, exp } => {
            is_finite_limit_value(base, pool) && is_finite_limit_value(exp, pool)
        }
        _ => true,
    }
}

/// The algebra of limits, applied to sums and products that mention a
/// special-function head.
///
/// `∫exp(−x²/2) dx` is returned as `1.2533…·erf(0.7071…·x)`, a **product**, so
/// the table in [`try_special_basis_limit`] never sees it: the top node is a
/// `Mul`, and every general route below fails on it (`x ↦ 1/t` hands
/// `erf(0.7071/t)` to a Taylor expansion with no rule for `erf`). The limit of
/// a product whose factors all have finite limits is the product of those
/// limits — elementary, and the one step needed to make the emitted
/// antiderivatives usable under the fundamental theorem.
///
/// Two restrictions keep this from being a general-purpose rule that could
/// change unrelated answers:
///
/// * it runs **only** on expressions mentioning [`SPECIAL_BASIS_HEADS`], which
///   before this commit had no limit rule at all and so can have no behaviour
///   to regress; and
/// * every child limit must be **finite**. `∞ · 0` and `∞ − ∞` are exactly the
///   indeterminate forms this identity does not cover, and a child that came
///   back infinite (or that still mentions `var`, which is `limit`'s way of
///   saying it did not solve the sub-problem) makes the whole rule decline
///   rather than guess.
fn try_special_basis_algebra(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<Option<ExprId>, LimitError> {
    let args = match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => args,
        _ => return Ok(None),
    };
    if !mentions_special_basis(expr, pool) {
        return Ok(None);
    }
    let is_sum = matches!(pool.get(expr), ExprData::Add(_));

    let mut parts = Vec::with_capacity(args.len());
    for a in args {
        let Ok(v) = limit_inner(a, var, point, direction, pool, depth + 1) else {
            return Ok(None);
        };
        if !is_finite_limit_value(v, pool) || depends_on(v, var, pool) {
            return Ok(None);
        }
        parts.push(v);
    }
    let combined = if is_sum {
        pool.add(parts)
    } else {
        pool.mul(parts)
    };
    Ok(Some(fold_known_reals(simplify(combined, pool).value, pool)))
}

/// `lim_{x→−∞} f(x)` computed as `lim_{y→+∞} f(−y)`.
///
/// A one-sided approach flips with the reflection — coming at `−∞` "from the
/// right" (larger `x`) is coming at `+∞` "from the left" (smaller `y`) — but at
/// an infinite point every rule in this engine treats the three directions
/// alike, so the flip is recorded for correctness rather than because anything
/// downstream reads it.
fn reflected_limit(
    expr: ExprId,
    var: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<ExprId, LimitError> {
    let y = pool.symbol("__lt_refl", crate::kernel::Domain::Real);
    let mut m = HashMap::new();
    m.insert(var, pool.mul(vec![pool.integer(-1_i32), y]));
    let reflected = simplify(subs(expr, &m, pool), pool).value;
    let flipped = match direction {
        LimitDirection::Plus => LimitDirection::Minus,
        LimitDirection::Minus => LimitDirection::Plus,
        LimitDirection::Bidirectional => LimitDirection::Bidirectional,
    };
    limit_inner(reflected, y, pool.pos_infinity(), flipped, pool, depth + 1)
}

fn neg_infinity(pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), pool.pos_infinity()])
}

fn is_pos_infinity(e: ExprId, pool: &ExprPool) -> bool {
    matches!(
        pool.get(e),
        ExprData::Symbol {
            name,
            domain: crate::kernel::Domain::Positive,
            ..
        } if name == POS_INFINITY_SYMBOL
    ) || matches!(
        pool.get(e),
        ExprData::Symbol {
            name,
            domain: crate::kernel::Domain::Real,
            ..
        } if name == POS_INFINITY_SYMBOL
    )
}

fn is_neg_infinity(e: ExprId, pool: &ExprPool) -> bool {
    let ExprData::Mul(args) = pool.get(e) else {
        return false;
    };
    if args.len() != 2 {
        return false;
    }
    let (a, b) = (args[0], args[1]);
    let m_one = pool.integer(-1_i32);
    (a == m_one && is_pos_infinity(b, pool)) || (b == m_one && is_pos_infinity(a, pool))
}

fn depends_on(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|a| depends_on(*a, var, pool)),
        ExprData::Pow { base, exp } => depends_on(base, var, pool) || depends_on(exp, var, pool),
        ExprData::Func { args, .. } => args.iter().any(|a| depends_on(*a, var, pool)),
        ExprData::Piecewise { branches, default } => {
            branches
                .iter()
                .any(|(c, v)| depends_on(*c, var, pool) || depends_on(*v, var, pool))
                || depends_on(default, var, pool)
        }
        ExprData::Predicate { args, .. } => args.iter().any(|a| depends_on(*a, var, pool)),
        ExprData::Forall { var: bv, body } | ExprData::Exists { var: bv, body } => {
            bv != var && depends_on(body, var, pool)
        }
        ExprData::RootSum {
            poly,
            var: bv,
            body,
        } => depends_on(poly, var, pool) || (bv != var && depends_on(body, var, pool)),
        ExprData::BigO(a) => depends_on(a, var, pool),
        ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_)
        | ExprData::Symbol { .. } => false,
    }
}

fn try_direct_substitution(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if quotient_is_zero_over_zero(expr, var, point, pool) {
        return None;
    }
    let mut m = HashMap::new();
    m.insert(var, point);
    let raw = subs(expr, &m, pool);
    if is_zero_times_pole_indeterminate(raw, pool) {
        return None;
    }
    let sub = fold_known_reals(simplify(raw, pool).value, pool);
    let dep = depends_on(sub, var, pool);
    let sing = substitution_is_singular(sub, pool);
    if dep || sing {
        None
    } else {
        Some(sub)
    }
}

/// True when ``expr`` is a product quotient `n/d` with `n,d → 0` at substitution (classic `0/0`).
fn quotient_is_zero_over_zero(expr: ExprId, var: ExprId, point: ExprId, pool: &ExprPool) -> bool {
    let (n, d) = numerator_denominator(expr, pool);
    if d == pool.integer(1_i32) {
        return false;
    }
    let n0 = substitute_fully(n, var, point, pool);
    let d0 = substitute_fully(d, var, point, pool);
    is_zero_like(n0, pool) && is_zero_like(d0, pool)
}

/// `0 · (pole at 0)` style indeterminate — must not simplify to misleading `0`.
fn is_zero_times_pole_indeterminate(expr: ExprId, pool: &ExprPool) -> bool {
    let factors: Vec<ExprId> = if matches!(pool.get(expr), ExprData::Mul(_)) {
        flatten_mul(expr, pool)
    } else {
        vec![expr]
    };
    let mut any_zero_factor = false;
    let mut any_pole = false;
    for f in factors {
        if substitution_is_singular(f, pool) {
            any_pole = true;
        }
        if matches!(pool.get(f), ExprData::Integer(z) if z.0 == 0) {
            any_zero_factor = true;
        }
        if let ExprData::Func { name, args } = pool.get(f) {
            if args.len() == 1
                && matches!(name.as_str(), "sin" | "sinh" | "tan")
                && matches!(pool.get(args[0]), ExprData::Integer(z) if z.0 == 0)
            {
                any_zero_factor = true;
            }
        }
    }
    any_zero_factor && any_pole
}

/// `true` after substitution if some sub-expression is ``0^{-n}`` (possibly nested via ``(0^{-1})^e``).
fn substitution_is_singular(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            if let ExprData::Integer(nn) = pool.get(exp) {
                if nn.0 < 0 {
                    let b = simplify(base, pool).value;
                    if matches!(pool.get(b), ExprData::Integer(z) if z.0 == 0) {
                        return true;
                    }
                }
            }
            substitution_is_singular(base, pool) || substitution_is_singular(exp, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            xs.iter().any(|a| substitution_is_singular(*a, pool))
        }
        ExprData::Func { args, .. } => args.iter().any(|a| substitution_is_singular(*a, pool)),
        _ => false,
    }
}

fn try_lhopital(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<Option<ExprId>, LimitError> {
    let (nume, deno) = numerator_denominator(expr, pool);
    if simplify(nume, pool).value == simplify(deno, pool).value {
        return Ok(None);
    }
    let n0 = substitute_fully(nume, var, point, pool);
    let d0 = substitute_fully(deno, var, point, pool);

    if !is_zero_like(n0, pool) || !is_zero_like(d0, pool) {
        return Ok(None);
    }

    let dn = diff(nume, var, pool)?.value;
    let dd = diff(deno, var, pool)?.value;
    if dn == nume && dd == deno {
        return Ok(None);
    }
    let quot = rational_quotient(dn, dd, pool);
    Ok(Some(limit_inner(
        quot,
        var,
        point,
        direction,
        pool,
        depth + 1,
    )?))
}

fn substitute_fully(expr: ExprId, var: ExprId, point: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(var, point);
    let s = simplify(subs(expr, &m, pool), pool).value;
    fold_known_reals(s, pool)
}

fn rational_quotient(n: ExprId, d: ExprId, pool: &ExprPool) -> ExprId {
    simplify(pool.mul(vec![n, pool.pow(d, pool.integer(-1_i32))]), pool).value
}

fn is_zero_like(e: ExprId, pool: &ExprPool) -> bool {
    let e = simplify(e, pool).value;
    if matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0) {
        return true;
    }
    if let ExprData::Rational(r) = pool.get(e) {
        if r.0 == 0 {
            return true;
        }
    }
    if let ExprData::Func { name, args } = pool.get(e) {
        if args.len() == 1 && matches!(name.as_str(), "sin" | "tan" | "sinh") {
            return is_zero_like(args[0], pool);
        }
    }
    false
}

fn is_one_like(e: ExprId, pool: &ExprPool) -> bool {
    let e = simplify(e, pool).value;
    if matches!(pool.get(e), ExprData::Integer(n) if n.0 == 1) {
        return true;
    }
    if let ExprData::Rational(r) = pool.get(e) {
        return r.0 == 1;
    }
    false
}

/// Constant-fold `sin`, `cos`, `exp`, … after limits (`sin(0) → 0`, `cos(0) → 1`).
fn fold_known_reals(expr: ExprId, pool: &ExprPool) -> ExprId {
    let e = simplify(expr, pool).value;
    match pool.get(e) {
        ExprData::Add(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| fold_known_reals(*x, pool)).collect();
            simplify(pool.add(ys), pool).value
        }
        ExprData::Mul(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| fold_known_reals(*x, pool)).collect();
            simplify(pool.mul(ys), pool).value
        }
        ExprData::Pow { base, exp } => {
            let b = fold_known_reals(base, pool);
            let xp = fold_known_reals(exp, pool);
            // `1^∞` / `1^(singular)` must not collapse to 1 — that silently
            // turns `(1+x)^(1/x)|_{x=0}` into 1 before the indeterminate-power
            // rewrite can recover `e`.
            if is_one_like(b, pool) {
                if substitution_is_singular(xp, pool)
                    || is_pos_infinity(xp, pool)
                    || is_neg_infinity(xp, pool)
                {
                    return simplify(pool.pow(b, xp), pool).value;
                }
                return pool.integer(1_i32);
            }
            simplify(pool.pow(b, xp), pool).value
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let inner = fold_known_reals(args[0], pool);
            if is_zero_like(inner, pool) {
                match name.as_str() {
                    "sin" | "tan" | "sinh" => return pool.integer(0_i32),
                    "cos" | "cosh" => return pool.integer(1_i32),
                    "exp" => return pool.integer(1_i32),
                    _ => {}
                }
            }
            simplify(pool.func(name, vec![inner]), pool).value
        }
        ExprData::Func { name, args } => {
            let ys: Vec<ExprId> = args.iter().map(|x| fold_known_reals(*x, pool)).collect();
            simplify(pool.func(name, ys), pool).value
        }
        _ => e,
    }
}

fn flatten_mul(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(expr) {
        ExprData::Mul(xs) => xs.iter().flat_map(|a| flatten_mul(*a, pool)).collect(),
        _ => vec![expr],
    }
}

fn numerator_denominator(expr: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let fac = flatten_mul(expr, pool);
    let mut nums = Vec::new();
    let mut dens = Vec::new();
    for f in fac {
        match pool.get(f) {
            ExprData::Pow { base, exp } => {
                if let ExprData::Integer(n) = pool.get(exp) {
                    let nn = &n.0;
                    if *nn == 0 {
                        nums.push(pool.integer(1_i32));
                    } else if *nn > 0 {
                        nums.push(f);
                    } else {
                        let m = nn
                            .clone()
                            .abs()
                            .to_u64()
                            .and_then(|u| u32::try_from(u).ok())
                            .map(|mag| pool.pow(base, pool.integer(mag as i64)));
                        if let Some(p) = m {
                            dens.push(p);
                        } else {
                            nums.push(f);
                        }
                    }
                } else {
                    nums.push(f);
                }
            }
            _ => nums.push(f),
        }
    }
    let n = if nums.is_empty() {
        pool.integer(1_i32)
    } else if nums.len() == 1 {
        nums[0]
    } else {
        pool.mul(nums)
    };
    let d = if dens.is_empty() {
        pool.integer(1_i32)
    } else if dens.len() == 1 {
        dens[0]
    } else {
        pool.mul(dens)
    };
    (n, d)
}

fn try_expansion_limit(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    order: u32,
) -> Result<Option<ExprId>, LimitError> {
    let exp = match local_expansion(expr, var, point, order, pool) {
        Ok(e) => e,
        Err(_) => {
            checkpoint(pool)?;
            return Ok(None);
        }
    };
    let r = expansion_to_limit(exp, pool, direction)?;
    if r.is_none() {
        // The expansion resolved nothing. If that is because the coefficient
        // loop hit the work ceiling (or a budget) part-way, say so rather than
        // letting the caller report `Unsupported`.
        checkpoint(pool)?;
    }
    Ok(r)
}

fn expansion_to_limit(
    exp: LocalExpansion,
    pool: &ExprPool,
    direction: LimitDirection,
) -> Result<Option<ExprId>, LimitError> {
    let LocalExpansion {
        valuation,
        coeffs,
        h_expr: _,
    } = exp;

    let mut idx = 0usize;
    while idx < coeffs.len() && is_zero_like(coeffs[idx], pool) {
        idx += 1;
    }
    if idx >= coeffs.len() {
        // Truncation hit all zeros — indeterminate within this order.
        return Ok(None);
    }
    let power = valuation + idx as i32;
    let coeff = coeffs[idx];

    if power > 0 {
        return Ok(Some(pool.integer(0_i32)));
    }
    if power == 0 {
        return Ok(Some(coeff));
    }

    // Polar — power < 0
    let pole_order = (-power) as u32;
    let sgn_c = structural_sign(coeff, pool).unwrap_or(1);
    if pole_order % 2 == 0 {
        return Ok(Some(signed_infinity(pool, sgn_c)));
    }
    let Some(hdir) = sign_from_h(direction, power) else {
        return Err(LimitError::NeedsOneSided);
    };
    Ok(Some(signed_infinity(pool, sgn_c * hdir)))
}

/// For odd pole: sign of `h^power` with `power < 0` as `h → 0` from one side.
fn sign_from_h(direction: LimitDirection, power: i32) -> Option<i8> {
    if power >= 0 {
        return Some(1);
    }
    let odd = (-power) % 2 != 0;
    if !odd {
        return Some(1);
    }
    match direction {
        LimitDirection::Plus => Some(1),
        LimitDirection::Minus => Some(-1),
        LimitDirection::Bidirectional => None,
    }
}

fn signed_infinity(pool: &ExprPool, sign: i8) -> ExprId {
    if sign < 0 {
        neg_infinity(pool)
    } else {
        pool.pos_infinity()
    }
}

fn structural_sign(e: ExprId, pool: &ExprPool) -> Option<i8> {
    match pool.get(e) {
        ExprData::Integer(n) => {
            if n.0 > 0 {
                Some(1)
            } else if n.0 < 0 {
                Some(-1)
            } else {
                None
            }
        }
        ExprData::Rational(r) => {
            if r.0 == 0 {
                None
            } else if r.0 > 0 {
                Some(1)
            } else {
                Some(-1)
            }
        }
        ExprData::Mul(xs) => {
            let mut s = 1i8;
            for a in xs {
                let sa = structural_sign(a, pool)?;
                s *= sa;
            }
            Some(s)
        }
        ExprData::Pow { base: _, exp } if matches!(pool.get(exp), ExprData::Integer(n) if n.0.clone() % 2 == 0) => {
            Some(1)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn limit_sin_over_x_zero() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(
            p.mul(vec![p.func("sin", vec![x]), p.pow(x, p.integer(-1_i32))]),
            &p,
        )
        .value;
        let r = limit(ex, x, p.integer(0_i32), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(1_i32));
    }

    #[test]
    fn one_sided_limit_off_the_domain_is_refused() {
        // √x is undefined at every point of every left neighbourhood of 0, so
        // there is no sequence along which to take `lim_{x→0⁻} √x` and the
        // question has no answer over ℝ. It used to return `√0 = 0` —
        // indistinguishable from the correct `lim_{x→0⁺} √x = 0`.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(p.func("sqrt", vec![x]), &p).value;
        assert!(
            limit(ex, x, p.integer(0_i32), LimitDirection::Minus, &p).is_err(),
            "√x has no left-hand limit at 0 over ℝ"
        );
        // The control: from the right the limit exists and is 0.
        let r = limit(ex, x, p.integer(0_i32), LimitDirection::Plus, &p).unwrap();
        assert_eq!(constant_f64(r, &p), Some(0.0), "got {}", p.display(r));

        // arccos is undefined to the right of 1, for the same reason.
        let ac = simplify(p.func("acos", vec![x]), &p).value;
        assert!(
            limit(ac, x, p.integer(1_i32), LimitDirection::Plus, &p).is_err(),
            "arccos has no right-hand limit at 1 over ℝ"
        );

        // …and a pole is *not* a domain boundary: 1/x is perfectly well
        // defined to the left of 0 and the one-sided limit is −∞.
        let inv = simplify(p.pow(x, p.integer(-1_i32)), &p).value;
        assert!(
            limit(inv, x, p.integer(0_i32), LimitDirection::Minus, &p).is_ok(),
            "lim_{{x→0⁻}} 1/x = −∞ must survive"
        );
        // √(x²) is defined on both sides; nothing to refuse.
        let sq = simplify(p.func("sqrt", vec![p.pow(x, p.integer(2_i32))]), &p).value;
        let r = limit(sq, x, p.integer(0_i32), LimitDirection::Minus, &p).unwrap();
        assert_eq!(constant_f64(r, &p), Some(0.0), "got {}", p.display(r));
    }

    #[test]
    fn limit_x_log_x_zero_plus() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(p.mul(vec![x, p.func("log", vec![x])]), &p).value;
        let r = limit(ex, x, p.integer(0_i32), LimitDirection::Plus, &p).unwrap();
        assert_eq!(r, p.integer(0_i32));
    }

    #[test]
    fn limit_exp_inf() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = p.func("exp", vec![x]);
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.pos_infinity());
    }

    #[test]
    fn limit_x_squared_at_positive_infinity() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(p.pow(x, p.integer(2_i32)), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.pos_infinity(), "{}", p.display(r));
    }

    /// `lim_{x→∞} (1 + 1/x)^x = e = exp(1)`  (the silent-wrong-answer regression).
    #[test]
    fn limit_compound_interest_is_e() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // (1 + 1/x)^x
        let base = p.add(vec![p.integer(1), p.pow(x, p.integer(-1))]);
        let ex = simplify(p.pow(base, x), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        let expected = simplify(p.func("exp", vec![p.integer(1)]), &p).value;
        assert_eq!(r, expected, "got {}", p.display(r));
    }

    /// `lim_{x→∞} (1 + a/x)^x = exp(a)` for a concrete integer `a = 2`.
    #[test]
    fn limit_one_plus_a_over_x_pow_x_is_exp_a() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // (1 + 2/x)^x
        let two_over_x = p.mul(vec![p.integer(2), p.pow(x, p.integer(-1))]);
        let base = p.add(vec![p.integer(1), two_over_x]);
        let ex = simplify(p.pow(base, x), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        let expected = simplify(p.func("exp", vec![p.integer(2)]), &p).value;
        assert_eq!(r, expected, "got {}", p.display(r));
    }

    /// Non-regression: `2^x` has a constant base (not the indeterminate `1^∞`
    /// form), so the new rewrite must NOT fire and must NOT fabricate a finite
    /// value.  The engine declines it (as it did before this fix); the key point
    /// is that it never returns a wrong finite limit.
    #[test]
    fn limit_two_pow_x_not_rewritten_to_finite() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(p.pow(p.integer(2), x), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p);
        // Either it stays unsupported or returns +∞ — but never a finite number.
        if let Ok(v) = r {
            assert_eq!(
                v,
                p.pos_infinity(),
                "2^x must not be a finite value: {}",
                p.display(v)
            );
        }
    }

    /// `lim_{x→0} (1 + x)^(1/x) = e` — the finite-point twin of the compound-interest form.
    #[test]
    fn limit_one_plus_x_to_one_over_x_is_e() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let base = p.add(vec![p.integer(1), x]);
        let ex = simplify(p.pow(base, p.pow(x, p.integer(-1))), &p).value;
        let r = limit(ex, x, p.integer(0_i32), LimitDirection::Bidirectional, &p).unwrap();
        let expected = simplify(p.func("exp", vec![p.integer(1)]), &p).value;
        assert_eq!(r, expected, "got {}", p.display(r));
    }

    /// Non-regression: `lim_{x→∞} (1 + 1/x) = 1` (not a power; sanity that the
    /// helper does not perturb the simple base limit).
    #[test]
    fn limit_one_plus_one_over_x_is_one() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(p.add(vec![p.integer(1), p.pow(x, p.integer(-1))]), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(1), "got {}", p.display(r));
    }

    #[test]
    fn rational_x_over_x_plus_one_after_inf_subst() {
        let p = ExprPool::new();
        let t = p.symbol("__lt_inf", Domain::Real);
        let inv = p.pow(t, p.integer(-1));
        let ex = p.mul(vec![
            inv,
            p.pow(p.add(vec![p.integer(1), inv]), p.integer(-1)),
        ]);
        let folded = flatten_nested_integer_pow(ex, &p);
        let canon = canonical_polynomial_quotient_in_var(folded, t, &p).unwrap();
        let r = simplify(canon, &p).value;
        let mut m = HashMap::new();
        m.insert(t, p.integer(0));
        let sub = fold_known_reals(simplify(subs(r, &m, &p), &p).value, &p);
        assert_eq!(sub, p.integer(1), "canonical={}", p.display(canon));
    }
}

/// Termination: the engine must always come back, with a value or a coded
/// refusal, and must stop when the ambient budget says so.
#[cfg(test)]
mod termination_tests {
    use super::*;
    use crate::budget::{self, Budget, BudgetError};
    use crate::errors::AlkahestError;
    use crate::kernel::Domain;

    /// `√(x²+x) − x` at `+∞`: an `∞−∞` cancellation whose conjugate is
    /// `x/(√(x²+x)+x) → 1/2`.
    ///
    /// This call did not return at all — the `x ↦ 1/t` substitution handed
    /// `√(t⁻²+t⁻¹)` to a 32-term Taylor expansion, and each derivative of a
    /// nested radical is a constant factor larger than the last.
    #[test]
    fn sqrt_x_squared_plus_x_minus_x_at_infinity_is_one_half() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let root = p.func("sqrt", vec![p.add(vec![p.pow(x, p.integer(2)), x])]);
        let ex = simplify(p.add(vec![root, p.mul(vec![p.integer(-1), x])]), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.rational(1, 2), "got {}", p.display(r));
    }

    /// The same cancellation with other coefficients, and its mirror at `−∞`.
    #[test]
    fn algebraic_cancellations_at_infinity() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let neg_inf = p.mul(vec![p.integer(-1), p.pos_infinity()]);

        // √(x²+3x) − x → 3/2
        let root = p.func(
            "sqrt",
            vec![p.add(vec![p.pow(x, p.integer(2)), p.mul(vec![p.integer(3), x])])],
        );
        let ex = simplify(p.add(vec![root, p.mul(vec![p.integer(-1), x])]), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.rational(3, 2), "√(x²+3x)−x: {}", p.display(r));

        // √(x²+1) − x → 0
        let root = p.func(
            "sqrt",
            vec![p.add(vec![p.pow(x, p.integer(2)), p.integer(1)])],
        );
        let ex = simplify(p.add(vec![root, p.mul(vec![p.integer(-1), x])]), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(0), "√(x²+1)−x: {}", p.display(r));

        // As x → −∞ there is no cancellation: √(x²+x) ~ |x| = −x, so the sum
        // is ~ −2x → +∞.
        let root = p.func("sqrt", vec![p.add(vec![p.pow(x, p.integer(2)), x])]);
        let ex = simplify(p.add(vec![root, p.mul(vec![p.integer(-1), x])]), &p).value;
        let r = limit(ex, x, neg_inf, LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.pos_infinity(), "√(x²+x)−x at −∞: {}", p.display(r));
    }

    /// A radical limit the engine cannot solve must still come back — with
    /// `E-LIMIT-004`, not by running forever — when no budget is active.
    #[test]
    fn unsolvable_radical_limit_refuses_within_the_work_ceiling() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // √(√(x²+x) + x): a half-integer scale the regularizer declines, so
        // this falls through to the expansion route that used to run away.
        let inner = p.func("sqrt", vec![p.add(vec![p.pow(x, p.integer(2)), x])]);
        let ex = p.func("sqrt", vec![p.add(vec![inner, x])]);
        // No wall-clock assertion here on purpose. Termination *is* the property
        // under test, and this call returning at all already proves it: before
        // the work ceiling existed this expression ran effectively forever, so a
        // regression hangs the test rather than failing a timing bound. An
        // elapsed() budget would only add flakiness — the AddressSanitizer job
        // builds with -Z build-std and runs many times slower than a normal
        // build, and a 30s bound that held locally failed there while the
        // refusal itself worked correctly.
        let err = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap_err();
        assert!(
            matches!(err, LimitError::DepthExceeded),
            "expected a bounded refusal, got {err:?}"
        );
        assert_eq!(err.code(), "E-LIMIT-004");
        // Not a budget trip — the internal ceiling stopped it.
        assert_eq!(last_budget_trip(), None);
    }

    /// A step budget stops the search and is reported as a budget trip, so a
    /// binding can raise `E-BUDGET-002` rather than "this limit is too hard".
    ///
    /// Steps and wall clock live on the thread-local budget stack, so this is
    /// safe to run in parallel with the rest of the suite (unlike the
    /// process-wide cancellation flag, which `budget`'s own tests serialize).
    #[test]
    fn step_budget_stops_a_hard_limit_and_is_attributed() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let inner = p.func("sqrt", vec![p.add(vec![p.pow(x, p.integer(2)), x])]);
        let ex = p.func("sqrt", vec![p.add(vec![inner, x])]);

        let _guard = budget::enter(Budget::new().with_max_steps(3));
        let err = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap_err();
        assert!(matches!(err, LimitError::DepthExceeded), "{err:?}");
        assert!(
            matches!(last_budget_trip(), Some(BudgetError::Steps { .. })),
            "budget trip not recorded: {:?}",
            last_budget_trip()
        );
    }

    /// A limit that *succeeds* under a generous budget must not be reported as
    /// a budget trip, and must not leave a stale trip behind for the next call.
    #[test]
    fn a_solved_limit_leaves_no_budget_trip() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        {
            let _guard = budget::enter(Budget::new().with_max_steps(3));
            let inner = p.func("sqrt", vec![p.add(vec![p.pow(x, p.integer(2)), x])]);
            let ex = p.func("sqrt", vec![p.add(vec![inner, x])]);
            assert!(limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).is_err());
            assert!(last_budget_trip().is_some());
        }
        let root = p.func("sqrt", vec![p.add(vec![p.pow(x, p.integer(2)), x])]);
        let ex = simplify(p.add(vec![root, p.mul(vec![p.integer(-1), x])]), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.rational(1, 2));
        assert_eq!(last_budget_trip(), None, "stale trip left behind");
    }

    /// The work ceiling bounds a whole `limit` call, not each re-entry: Gruntz
    /// sub-limits call back into `limit`, and a per-call baseline would reset
    /// the ceiling every time and never trip.
    #[test]
    fn work_baseline_is_installed_once_and_cleared_on_exit() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        assert!(WORK_BASELINE.with(|c| c.get()).is_none());
        let ex = simplify(p.pow(x, p.integer(2)), &p).value;
        let _ = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p);
        assert!(
            WORK_BASELINE.with(|c| c.get()).is_none(),
            "baseline leaked past the outermost call"
        );
    }
}

#[cfg(test)]
mod numeric_refutation_tests {
    use super::*;
    use crate::kernel::Domain;

    /// `x/|x|` is `sign(x)`: it never takes the value 0, yet the symbolic
    /// route returned 0 in all three directions.
    ///
    /// The algebraically identical `|x|/x` was already refused, so before this
    /// guard the *order of the operands* decided whether a caller got an honest
    /// refusal or a confident wrong answer.
    #[test]
    fn sign_function_limit_is_refused_in_every_direction() {
        for direction in [
            LimitDirection::Bidirectional,
            LimitDirection::Plus,
            LimitDirection::Minus,
        ] {
            let p = ExprPool::new();
            let x = p.symbol("x", Domain::Real);
            let ex = p.mul(vec![x, p.pow(p.func("abs", vec![x]), p.integer(-1_i32))]);
            let got = limit(ex, x, p.integer(0_i32), direction, &p);
            assert!(
                got.is_err(),
                "x/|x| at 0 ({direction:?}) should refuse, got {}",
                p.display(got.unwrap())
            );
        }
    }

    /// The guard must not fire on limits that genuinely exist, including ones
    /// that need cancellation to evaluate.
    #[test]
    fn ordinary_limits_survive_the_guard() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1_i32);

        let sinc = simplify(
            p.mul(vec![p.func("sin", vec![x]), p.pow(x, p.integer(-1_i32))]),
            &p,
        )
        .value;
        assert_eq!(
            limit(sinc, x, p.integer(0_i32), LimitDirection::Bidirectional, &p).unwrap(),
            one
        );

        // (1 - cos x)/x² = 1/2 — the case that would break if the sampler
        // pushed closer than 1e-4 and hit catastrophic cancellation.
        let half = p.mul(vec![
            p.add(vec![
                one,
                p.mul(vec![p.integer(-1_i32), p.func("cos", vec![x])]),
            ]),
            p.pow(x, p.integer(-2_i32)),
        ]);
        let got = limit(
            simplify(half, &p).value,
            x,
            p.integer(0_i32),
            LimitDirection::Bidirectional,
            &p,
        )
        .unwrap();
        assert_eq!(got, p.rational(1, 2));
    }

    /// An oscillating factor has no settled one-sided estimate, so the guard
    /// must stay silent rather than refuse a correct answer.
    ///
    /// `x·sin(1/x) → 0` at 0 by squeeze, and the samples swing wildly, so this
    /// pins that "no verdict" is distinct from "contradiction".
    #[test]
    fn oscillation_does_not_trigger_a_false_refusal() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = p.mul(vec![x, p.func("sin", vec![p.pow(x, p.integer(-1_i32))])]);
        let got = limit(ex, x, p.integer(0_i32), LimitDirection::Bidirectional, &p);
        assert_eq!(got.unwrap(), p.integer(0_i32));
    }

    /// A free parameter makes sampling meaningless, so the guard abstains.
    #[test]
    fn symbolic_parameter_abstains() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        assert!(has_free_symbol_besides(p.mul(vec![a, x]), x, &p));
        assert!(!has_free_symbol_besides(
            p.mul(vec![x, p.func("sin", vec![x])]),
            x,
            &p
        ));
    }

    // ── the special-function basis at ±∞ ────────────────────────────────────

    fn at_infinity(src_fn: &str, sign: i32) -> (ExprPool, Result<ExprId, LimitError>) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.func(src_fn, vec![x]);
        let point = if sign > 0 {
            p.pos_infinity()
        } else {
            neg_infinity(&p)
        };
        let r = limit(e, x, point, LimitDirection::Bidirectional, &p);
        (p, r)
    }

    /// The table itself, at both ends where both ends are real.
    #[test]
    fn special_basis_limits_match_dlmf() {
        for (name, sign, want) in [
            ("erf", 1, 1.0),
            ("erf", -1, -1.0),
            ("erfc", 1, 0.0),
            ("erfc", -1, 2.0),
            ("Si", 1, std::f64::consts::FRAC_PI_2),
            ("Si", -1, -std::f64::consts::FRAC_PI_2),
            ("Ci", 1, 0.0),
            ("Ei", -1, 0.0),
            ("fresnels", 1, 0.5),
            ("fresnels", -1, -0.5),
            ("fresnelc", 1, 0.5),
            ("fresnelc", -1, -0.5),
        ] {
            let (p, r) = at_infinity(name, sign);
            let v = r.unwrap_or_else(|e| panic!("lim {name} at {sign}∞: {e}"));
            let mut binds = HashMap::new();
            binds.insert(p.symbol("pi", Domain::Real), std::f64::consts::PI);
            let got = crate::eval::eval_f64(v, &p, &binds)
                .unwrap_or_else(|e| panic!("{} did not evaluate: {e}", p.display(v)));
            assert!(
                (got - want).abs() < 1e-12,
                "lim_{{x→{sign}∞}} {name}(x) = {want}, got {got}"
            );
        }
    }

    /// `Ci`, `Chi` and `li` have a branch cut on the negative reals and no real
    /// value there, so there is no real limit at `−∞` to report.  Inventing one
    /// would be reporting the real part of a complex number as if it were the
    /// answer.
    #[test]
    fn the_cut_functions_have_no_limit_at_minus_infinity() {
        for name in ["Ci", "Chi", "li"] {
            let (_, r) = at_infinity(name, -1);
            assert!(
                r.is_err(),
                "{name} has no real value on the negative reals; a limit at −∞ must not be invented"
            );
        }
    }

    /// The emitted antiderivatives are *products* — `1.2533…·erf(0.7071…·x)` —
    /// so the table alone is not enough; the algebra of limits has to reach
    /// through the `Mul`, and the `f64` coefficient inside the `erf` has to be
    /// read structurally (`lim 0.7071·x` is a case the general engine still
    /// cannot do).
    #[test]
    fn a_scaled_erf_product_reaches_its_limit() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let inner = p.mul(vec![p.float(std::f64::consts::FRAC_1_SQRT_2, 53), x]);
        let outer = (std::f64::consts::PI / 2.0).sqrt();
        let e = p.mul(vec![p.float(outer, 53), p.func("erf", vec![inner])]);
        let v = limit(e, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).expect("limit");
        let got = crate::eval::eval_f64(v, &p, &HashMap::new()).expect("value");
        assert!((got - outer).abs() < 1e-12, "got {got}");
    }

    /// `lim_{x→−∞} x·exp(−x²) = 0`.  The `x ↦ −1/t` substitution cannot do
    /// this — it produces `−exp(−t⁻²)/t`, which has no Taylor expansion at `0`
    /// — and it is the limit `∫_{-∞}^{∞} x²·exp(−x²) dx` needs at its lower
    /// bound.  The reflection to `+∞`, where Gruntz runs, settles it.
    #[test]
    fn the_reflection_fallback_reaches_a_gruntz_only_limit_at_minus_infinity() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.mul(vec![
            x,
            p.func(
                "exp",
                vec![p.mul(vec![p.integer(-1_i32), p.pow(x, p.integer(2_i32))])],
            ),
        ]);
        let v = limit(e, x, neg_infinity(&p), LimitDirection::Bidirectional, &p).expect("0");
        assert_eq!(v, p.integer(0_i32));
    }

    /// The fallback must not turn a limit that does not exist into one that
    /// does: `lim_{x→−∞} sin x` has no value, from either direction of
    /// approach, and the reflection is the identity on it.
    #[test]
    fn the_reflection_fallback_does_not_invent_a_missing_limit() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.func("sin", vec![x]);
        assert!(limit(e, x, neg_infinity(&p), LimitDirection::Bidirectional, &p).is_err());
    }

    /// `∞ − ∞` and `∞ · 0` are exactly what the finiteness requirement in
    /// `try_special_basis_algebra` exists to refuse: `Ei(x) − Shi(x)` has both
    /// terms diverging at `+∞` and the rule must decline rather than subtract
    /// two infinities.
    #[test]
    fn the_algebra_rule_declines_an_indeterminate_difference() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.add(vec![
            p.func("Ei", vec![x]),
            p.mul(vec![p.integer(-1_i32), p.func("Shi", vec![x])]),
        ]);
        assert!(try_special_basis_algebra(
            e,
            x,
            p.pos_infinity(),
            LimitDirection::Bidirectional,
            &p,
            0
        )
        .unwrap()
        .is_none());
    }
}
