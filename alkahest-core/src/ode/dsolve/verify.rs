//! Substitution-based verification gate for [`super::dsolve`].
//!
//! Given a candidate solution `y(x)`, build the residual of the original
//! equation with `y`, `y'`, `y''`, … replaced by the candidate and its
//! derivatives, then require the residual to be the symbolic zero, or — when
//! `simplify` cannot close it — numerically `≈ 0` at several `x` samples over
//! several random assignments of the integration constants.
//!
//! # Why the residual is not evaluated alone
//!
//! The substituted residual is a *conflation*: the equation's own coefficients
//! and forcing term and the candidate's contribution are mixed into one
//! expression, so when it fails to evaluate at a sample there is, on the face
//! of it, nothing to compare against.  Skipping every such sample is what makes
//! the gate unsound — a candidate that blows up at a point where the ODE is
//! perfectly regular is *evidence that the candidate is wrong*, and blanket
//! skipping throws that evidence away.  A wrong candidate accepted exactly that
//! way is pinned in
//! `tests::wrong_candidate_blowing_up_at_a_regular_point_is_rejected`.
//!
//! So a non-finite sample is **classified** rather than skipped, by asking the
//! two questions the conflated residual cannot answer separately:
//!
//! 1. *Is the ODE itself evaluable at this `x`?*  Probe `input.equation` with
//!    finite dummy values bound to `y, y', …` ([`ode_is_regular_at`]).  If no
//!    probe is finite the equation is singular there (a `√(a − x)` coefficient
//!    past its branch point, a pole in the forcing term) and the sample really
//!    does carry no information — skip it.
//! 2. *Does the candidate stay finite at this `x`?*  Evaluate `y(x)` and its
//!    derivatives on their own.  If the ODE is regular here and the candidate
//!    is not, that is a disagreement, not a skip.
//!
//! If both sides are finite and only the *simplified residual* was not, the
//! non-finiteness was an artefact of the residual's algebraic form (an
//! `∞ − ∞` produced by a rewriting).  Every quantity involved is then a real
//! number, so the verdict is taken from the original equation evaluated at the
//! candidate's own values — which recovers a sample the old code discarded and
//! catches a disagreement it used to hide.
//!
//! The conflated residual remains the *primary* numeric check and is not
//! replaced.  It has to be: `simplify` frequently cancels the non-elementary
//! part of a candidate (`Ei`, `Si`, `Ci`) out of the residual, leaving an
//! elementary expression this module can evaluate, while the candidate itself
//! cannot be evaluated at all.  Four corpus ODEs (`y''−y=1/x`, `y''−y=eˣ/x`,
//! `y''−4y=1/x`, `y'''−y'=1/x`) certify only because of that cancellation, so
//! evaluating the split form *instead* would lose them.  The split evaluation
//! is a discriminator layered on top, reached only for samples the conflated
//! residual could not resolve.
//!
//! **Known conservatism.** For a *nonlinear* ODE a correct solution may have a
//! movable singularity at a regular point (`y' = 1 + y²` has `y = tan(x + C)`).
//! Landing a sample on one is now a decline rather than a skip.  That is the
//! intended direction of the trade: a decline is acceptable, a wrong `y(x)` is
//! not.  In practice `powf` and division produce a finite (merely huge) value
//! near a pole, so the case is not reached on the corpus — where the classifier
//! is never entered at all, every sample of every numerically-certified
//! solution being finite.

//! # Free parameters
//!
//! An equation whose coefficients contain symbols other than `x` — `y'' +
//! 2ζω y' + ω² y = 0` — has a candidate whose residual mentions those symbols,
//! and the real sampler above cannot evaluate it at all: every sample comes
//! back `None`, the report is `unevaluable`, and the gate declines.  The
//! parametric path binds each free parameter to a value as well, over several
//! deterministic assignments.
//!
//! It evaluates in **ℂ**, not ℝ.  The uniform two-exponential form of a
//! symbolic-coefficient equation is `e^{(−ζω ± ω√(ζ²−1))t}`, whose exponent is
//! complex for `|ζ| < 1` — the underdamped branch, the one the caller most
//! often means.  A real evaluator returns `NaN` there, and a gate that reads
//! `NaN` as disagreement would refuse the correct answer on exactly the
//! parameter range it matters for; one that skipped it would only ever check
//! the overdamped side.  Evaluating on the principal complex branch makes both
//! sides of the residual ordinary complex numbers and the check meaningful on
//! the whole parameter space.  The classification of a sample is otherwise the
//! same three-way split as in the real path, and as in `integrate::gate`:
//! finite-and-zero agrees, finite-and-non-zero disagrees, and non-finite or
//! unevaluable is *no information* rather than evidence either way.
//!
//! Tolerance is relative there.  `e^{λt}` with a sampled `λ ≈ 3` is `O(10)`
//! before the equation's own coefficients multiply it, so a fixed `1e-6`
//! absolute band is not the same test at both ends of the parameter grid; the
//! band is scaled by the magnitude of the candidate and its derivatives at the
//! sample.
//!
//! The complex pass may end the verification only by *disagreeing*.  Its
//! evaluator implements a fixed list of heads, and a candidate written with one
//! it does not have — `lambert_w`, which the separable class produces when it
//! inverts `k·log y + m·y = T` — makes every sample unevaluable.  That is an
//! absence of information, not evidence, so the real sampler is then asked as a
//! second opinion, over the *same* parameters bound to the *same* values
//! ([`parameter_env`] is the single table both draw from).  It knows a few
//! heads ℂ does not; where it also cannot conclude, the candidate is refused.
//!
//! [`PARAM_SETS`] samples **both signs**.  It did not always: every row used to
//! be positive, so a candidate right for `Kₘ > 0` and wrong for `Kₘ < 0` was
//! certified on the half of the parameter space it happened to be right on,
//! while `OdeInput` carries no assumption that would justify the restriction.
//! Negative rows are safe here only because the pass they feed evaluates over
//! ℂ — where `log(−1.3)` and `√(−1.3)` are ordinary numbers and a correct
//! candidate's analytic continuation is still correct — and because a row that
//! cannot be resolved is *no information* rather than a refusal.  See
//! [`PARAM_SETS`] for the measurement.
//!
//! # Implicit solutions
//!
//! A first-order class that can only answer with a relation `G(x, y) = 0` is
//! gated by [`implicit_relation_is_zero`] instead, which substitutes the slope
//! field `y′ = −Gₓ/G_y` the implicit function theorem gives and requires the
//! result to vanish identically in **two** free variables.  It draws its
//! parameter values from the same [`parameter_env`].

use super::{contains, ddx, simp, subs1, DsolveError, OdeInput};
use crate::eval::symbols::{collect_free_symbols, is_pi, walk_symbols};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;
use std::fmt;

/// Absolute tolerance for "this sample of the residual is zero".
const ZERO_TOL: f64 = 1e-6;

/// Minimum number of resolved, agreeing samples before a numeric certificate is
/// issued.  Samples the classifier could not resolve do not count towards it.
const MIN_AGREEING_SAMPLES: usize = 6;

// Per-thread `(candidates offered, candidates refused)` tally, so the corpus
// harness can split a decline into "no method produced a candidate" and "the
// gate refused the candidate a method produced".  Test-only; nothing outside
// the measurement harness reads it.
#[cfg(test)]
thread_local! {
    pub(crate) static GATE_TALLY: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

/// Reset [`GATE_TALLY`] and return the tally accumulated since the last reset.
#[cfg(test)]
pub(crate) fn take_gate_tally() -> (usize, usize) {
    GATE_TALLY.with(|t| t.replace((0, 0)))
}

/// Would this candidate be certified by the *symbolic* branch alone?
///
/// Exposed so tests can pin which half of the gate a case depends on; the two
/// halves have very different reach and a comment claiming one of them is stale
/// as soon as `simplify` changes.
#[cfg(test)]
pub(crate) fn certifies_symbolically(input: &OdeInput, y_of_x: ExprId, pool: &ExprPool) -> bool {
    match build_residual(input, y_of_x, pool) {
        Ok((residual, _)) => {
            is_symbolic_zero(residual, pool)
                || is_symbolic_zero(super::simp_plain(residual, pool), pool)
        }
        Err(_) => false,
    }
}

/// Verify a candidate `y(x)` against `input.equation = 0`.
///
/// Returns `Ok(())` if the residual is symbolically or numerically zero.
pub(crate) fn residual_is_zero(
    input: &OdeInput,
    y_of_x: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveError> {
    let outcome = verify_inner(input, y_of_x, constants, pool);
    #[cfg(test)]
    GATE_TALLY.with(|t| {
        let (offered, refused) = t.get();
        t.set((offered + 1, refused + usize::from(outcome.is_err())));
    });
    outcome
}

fn verify_inner(
    input: &OdeInput,
    y_of_x: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveError> {
    let (residual, candidate_derivs) = build_residual(input, y_of_x, pool)?;

    // Symbolic zero?  Try both the expanded and the plain (non-expanding)
    // normal forms — expansion flattens polynomial cancellations, while plain
    // simplify is better at collapsing products such as `√D·√D⁻¹ → 1`.
    if is_symbolic_zero(residual, pool) || is_symbolic_zero(super::simp_plain(residual, pool), pool)
    {
        return Ok(());
    }

    // Free parameters (symbols that are neither `x`, the unknown, a derivative
    // symbol, nor an integration constant) make the real sampler useless: every
    // sample is `None` and the report says only "unevaluable".  Bind them too,
    // and evaluate over ℂ so the complex branch of the answer is reachable.
    //
    // A residual that mentions the imaginary unit takes the same route with no
    // parameters at all: `eval` has no `f64` for `i` and reports the whole
    // residual unevaluable, so the real sampler can only ever decline it.  The
    // complex evaluator knows `i` natively and is the right instrument; the
    // classification it applies is the same one.
    let params = free_parameters(input, &[residual], constants, pool);
    if !params.is_empty() || mentions_imaginary_unit(residual, pool) {
        let report =
            parametric_report(input, residual, &candidate_derivs, constants, &params, pool);
        if report.certifies() {
            return Ok(());
        }
        // A *disagreement* over ℂ is evidence against the candidate and is
        // final.  "I could not evaluate this at all" is not evidence of
        // anything, and it is the ordinary outcome for a candidate written
        // with a head the complex evaluator does not implement — the Lambert-W
        // inversion of a separable equation is the case that reaches it.  The
        // real sampler below binds the *same* parameters to the *same* values
        // and knows a few heads ℂ does not, so it is asked as a second opinion
        // rather than the answer being refused unheard.
        if report.has_counterevidence() {
            return Err(DsolveError::VerificationFailed(format!(
                "residual did not reduce to zero over the parameters {} ({report}): {}",
                param_names(&params, pool),
                pool.display(residual)
            )));
        }
        let real = numeric_report(input, residual, &candidate_derivs, constants, &params, pool);
        if real.certifies() {
            return Ok(());
        }
        return Err(DsolveError::VerificationFailed(format!(
            "residual did not reduce to zero over the parameters {} (over ℂ: {report}; \
             over ℝ: {real}): {}",
            param_names(&params, pool),
            pool.display(residual)
        )));
    }

    // Numeric fallback: sample x over several constant assignments.
    let report = numeric_report(input, residual, &candidate_derivs, constants, &[], pool);
    if report.certifies() {
        return Ok(());
    }

    Err(DsolveError::VerificationFailed(format!(
        "residual did not reduce to zero ({report}): {}",
        pool.display(residual)
    )))
}

/// Symbols in `residual` that the numeric sampler would otherwise leave unbound.
///
/// `x` and the integration constants are bound by the sampler already; `y` and
/// the derivative symbols cannot survive [`build_residual`]'s substitution, but
/// are excluded defensively so a stray one becomes a decline rather than a
/// parameter that gets a random value bound to it.
///
/// `pi` and the imaginary unit are excluded by
/// [`collect_free_symbols`](crate::eval::symbols::collect_free_symbols): they
/// are ordinary [`ExprData::Symbol`]s in this crate but they already denote a
/// number, and sampling them is a false-refusal machine — a candidate written
/// in the casus irreducibilis form `2√(−p/3)·cos((acos c + 2πk)/3)` evaluated
/// at `π = 1.7` disagrees at every sample.  [`eval`] and [`eval_complex`] give
/// them their real values instead.
fn free_parameters(
    input: &OdeInput,
    exprs: &[ExprId],
    constants: &[ExprId],
    pool: &ExprPool,
) -> Vec<ExprId> {
    let mut bound: Vec<ExprId> = vec![input.x, input.y];
    bound.extend_from_slice(&input.derivs);
    bound.extend_from_slice(constants);
    let mut out: Vec<ExprId> = Vec::new();
    for &e in exprs {
        collect_free_symbols(e, pool, &mut out);
    }
    out.retain(|s| !bound.contains(s));
    // Deterministic order: the sampled value of a parameter must not depend on
    // the traversal order of a pool shared with other work.
    out.sort_by_key(|&s| pool.display(s).to_string());
    out.dedup();
    out
}

/// Comma-separated parameter names, for a refusal message.
fn param_names(params: &[ExprId], pool: &ExprPool) -> String {
    if params.is_empty() {
        // The complex path was entered for the residual's own sake, not for a
        // parameter; saying "the parameters " with nothing after it reads as a
        // formatting bug.
        return "(none: a complex-valued residual)".to_string();
    }
    params
        .iter()
        .map(|&p| pool.display(p).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Does `expr` mention the imaginary unit anywhere?
///
/// The real [`eval`] has no value for it, so such a residual is unevaluable to
/// the real sampler however its parameters are bound.
fn mentions_imaginary_unit(expr: ExprId, pool: &ExprPool) -> bool {
    let mut found = false;
    walk_symbols(expr, pool, &mut |s, pool| {
        found |= pool.is_imaginary_unit(s);
    });
    found
}

/// Substitute the candidate into the equation.
///
/// Returns the simplified residual **and** the candidate's contribution kept on
/// its own — `candidate_derivs[k]` is `dᵏ/dxᵏ y(x)`, with `[0] = y(x)`.  Keeping
/// the second half is the whole point: it is what lets [`classify_nonfinite`]
/// ask about the candidate without the equation's coefficients mixed in.
fn build_residual(
    input: &OdeInput,
    y_of_x: ExprId,
    pool: &ExprPool,
) -> Result<(ExprId, Vec<ExprId>), DsolveError> {
    let mut candidate_derivs = Vec::with_capacity(input.derivs.len() + 1);
    candidate_derivs.push(y_of_x);
    let mut cur = y_of_x;
    for _ in &input.derivs {
        cur = ddx(cur, input.x, pool)?;
        candidate_derivs.push(cur);
    }

    // Substitute y → y(x), y^(k) → d^k/dx^k y(x).  They are distinct symbols,
    // so substitution order does not matter.
    let mut residual = subs1(input.equation, input.y, y_of_x, pool);
    for (k, &dsym) in input.derivs.iter().enumerate() {
        residual = subs1(residual, dsym, candidate_derivs[k + 1], pool);
    }
    Ok((simp(residual, pool), candidate_derivs))
}

fn is_symbolic_zero(expr: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(expr), ExprData::Integer(n) if n.0 == 0)
}

/// Per-sample tally produced by [`numeric_report`].
///
/// Every sample of the `x × constants` grid lands in exactly one bucket, unless
/// `unevaluable` is set, in which case sampling stopped early.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct NumericReport {
    /// Resolved to a finite value `≈ 0`.
    agree: usize,
    /// Resolved to a finite value that is **not** `≈ 0` — the candidate is wrong.
    disagree: usize,
    /// The ODE is regular at this `x` but the candidate is not finite there —
    /// also evidence the candidate is wrong (see the module docs for the
    /// nonlinear movable-singularity caveat).
    blowup_at_regular_point: usize,
    /// The equation itself is not evaluable at this sample — no information.
    skipped_singular_ode: usize,
    /// The candidate contains a construct [`eval`] does not know, so the sample
    /// could not be classified either way — no information.
    skipped_unknown_construct: usize,
    /// The residual itself contains a construct [`eval`] does not know; the gate
    /// refuses to certify numerically at all.
    unevaluable: bool,
}

impl NumericReport {
    fn certifies(&self) -> bool {
        !self.unevaluable
            && self.disagree == 0
            && self.blowup_at_regular_point == 0
            && self.agree >= MIN_AGREEING_SAMPLES
    }
}

impl fmt::Display for NumericReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.unevaluable {
            return write!(
                f,
                "residual contains a construct the sampler cannot evaluate"
            );
        }
        write!(
            f,
            "samples: {} agreeing, {} disagreeing, {} candidate blow-ups at a regular point, \
             {} skipped (ODE singular), {} skipped (unknown construct)",
            self.agree,
            self.disagree,
            self.blowup_at_regular_point,
            self.skipped_singular_ode,
            self.skipped_unknown_construct
        )
    }
}

// ---------------------------------------------------------------------------
// Sample grids
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random constant assignments (no rng dependency).
/// Constants are kept positive and reasonably large so that radicands such as
/// `sqrt(4·C − 3x²)` arising from quadratic-implicit solutions stay real over
/// the (small) x-sample range.
pub(crate) const CONST_SETS: [&[f64]; 3] = [
    &[5.7, 4.3, 6.4, 5.1, 4.9],
    &[8.5, 7.8, 6.6, 9.2, 7.1],
    &[12.3, 10.0, 11.7, 10.5, 9.4],
];

/// `x` sample points, shared by the explicit, parametric and implicit gates.
pub(crate) const X_SAMPLES: [f64; 5] = [0.11, 0.27, 0.43, 0.61, 0.79];

/// Numerically check residual ≈ 0 at several `x` over random constants.
///
/// When there are parameters the grid is the *product* of [`PARAM_SETS`] and
/// [`CONST_SETS`], not a pairing of the two: pairing them by index would leave
/// the negative rows of `PARAM_SETS` unsampled here, so the real second opinion
/// would be reasoning about a different parameter region than the complex pass
/// that asked for it.  With no parameters every row gives the same environment,
/// so one pass is the whole grid.
fn numeric_report(
    input: &OdeInput,
    residual: ExprId,
    candidate_derivs: &[ExprId],
    constants: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
) -> NumericReport {
    let mut report = NumericReport::default();
    let rows = if params.is_empty() {
        1
    } else {
        PARAM_SETS.len()
    };
    for (set, cs) in (0..rows).flat_map(|s| CONST_SETS.iter().map(move |cs| (s, cs))) {
        let param_env = parameter_env(params, set);
        let mut env: HashMap<ExprId, f64> = param_env.clone();
        for (i, &c) in constants.iter().enumerate() {
            env.insert(c, cs[i % cs.len()]);
        }
        for &xv in &X_SAMPLES {
            env.insert(input.x, xv);
            match eval(residual, &env, pool) {
                Some(v) if v.is_finite() => record(&mut report, v),
                // Non-finite: the conflated residual cannot say whose fault it
                // is.  Ask the equation and the candidate separately.
                Some(_) => classify_nonfinite(
                    input,
                    candidate_derivs,
                    &env,
                    &param_env,
                    xv,
                    pool,
                    &mut report,
                ),
                // Unknown construct → refuse to certify numerically.
                None => {
                    report.unevaluable = true;
                    return report;
                }
            }
        }
    }
    report
}

/// Bucket a finite residual value as agreement or disagreement.
fn record(report: &mut NumericReport, v: f64) {
    if v.abs() < ZERO_TOL {
        report.agree += 1;
    } else {
        report.disagree += 1;
    }
}

/// Decide what a non-finite sample of the conflated residual means.
///
/// See the module docs for the three outcomes.  The ordering matters: "the ODE
/// is singular here" dominates, because nothing can be concluded about a
/// candidate at a point the equation itself does not reach.
#[allow(clippy::too_many_arguments)]
fn classify_nonfinite(
    input: &OdeInput,
    candidate_derivs: &[ExprId],
    env: &HashMap<ExprId, f64>,
    params: &HashMap<ExprId, f64>,
    xv: f64,
    pool: &ExprPool,
    report: &mut NumericReport,
) {
    // 1. Is the equation itself well-defined at this `x`, candidate aside?
    if !ode_is_regular_at(input, xv, params, pool) {
        report.skipped_singular_ode += 1;
        return;
    }

    // 2. It is.  Does the candidate stay finite here?
    let mut vals = Vec::with_capacity(candidate_derivs.len());
    for &d in candidate_derivs {
        match eval(d, env, pool) {
            Some(v) if v.is_finite() => vals.push(v),
            // The ODE is regular here and the candidate is not: evidence of a
            // wrong answer, which is exactly what the old blanket skip lost.
            Some(_) => {
                report.blowup_at_regular_point += 1;
                return;
            }
            // Cannot evaluate the candidate (e.g. it contains `Ei`), so no
            // conclusion is available either way.
            None => {
                report.skipped_unknown_construct += 1;
                return;
            }
        }
    }

    // 3. Both sides are finite, so the non-finiteness came from the residual's
    //    algebraic form.  Re-ask the original equation at the candidate's own
    //    values — a real verdict where the old code had none.
    let mut eq_env: HashMap<ExprId, f64> = params.clone();
    eq_env.insert(input.x, xv);
    // `build_residual` always pushes `y(x)` first and the loop above either
    // filled `vals` completely or returned, so index 0 exists.
    eq_env.insert(input.y, vals[0]);
    for (k, &dsym) in input.derivs.iter().enumerate() {
        eq_env.insert(dsym, vals[k + 1]);
    }
    match eval(input.equation, &eq_env, pool) {
        Some(v) if v.is_finite() => record(report, v),
        // The equation is singular at *this state*, not merely at this `x` (a
        // `1/(y − 3)` reached exactly at `y = 3`) — no information.
        _ => report.skipped_singular_ode += 1,
    }
}

/// Is `input.equation` evaluable to a finite value at `x = xv`, independently of
/// the candidate?
///
/// Probes several finite states `(y, y', y'', …)`; one finite result is enough,
/// since the question is whether the *equation* has a singularity at this `x`,
/// not whether some particular state is admissible.  Distinct values per
/// derivative order stop a probe from cancelling the equation by accident.
fn ode_is_regular_at(
    input: &OdeInput,
    xv: f64,
    params: &HashMap<ExprId, f64>,
    pool: &ExprPool,
) -> bool {
    const PROBES: [f64; 4] = [1.0, 2.5, 0.5, -1.5];
    PROBES.iter().any(|&p| {
        let mut env: HashMap<ExprId, f64> = params.clone();
        env.insert(input.x, xv);
        env.insert(input.y, p);
        for (k, &dsym) in input.derivs.iter().enumerate() {
            env.insert(dsym, p + 0.25 * (k as f64 + 1.0));
        }
        matches!(eval(input.equation, &env, pool), Some(v) if v.is_finite())
    })
}

/// Evaluate `expr` to an `f64` given a symbol→value environment.
/// Returns `None` for constructs the evaluator does not understand (so the
/// caller refuses to certify rather than guessing).
///
/// `pi` resolves to π without being in `env` — it is a plain symbol in this
/// crate, and [`free_parameters`] deliberately does not sample it, so nothing
/// else would bind it.  The imaginary unit stays `None` here: it has no `f64`
/// value, and "unknown construct" (→ no information) is the honest reading.
pub(crate) fn eval(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (num, den) = r.0.clone().into_numer_denom();
            Some(num.to_f64() / den.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { .. } => env
            .get(&expr)
            .copied()
            .or_else(|| is_pi(expr, pool).then_some(std::f64::consts::PI)),
        ExprData::Add(args) => {
            let mut s = 0.0;
            for a in args {
                s += eval(a, env, pool)?;
            }
            Some(s)
        }
        ExprData::Mul(args) => {
            let mut p = 1.0;
            for a in args {
                p *= eval(a, env, pool)?;
            }
            Some(p)
        }
        ExprData::Pow { base, exp } => {
            let b = eval(base, env, pool)?;
            let e = eval(exp, env, pool)?;
            Some(b.powf(e))
        }
        ExprData::Func { name, args } => {
            let v: Vec<f64> = args
                .iter()
                .map(|&a| eval(a, env, pool))
                .collect::<Option<_>>()?;
            eval_func(&name, &v)
        }
        _ => None,
    }
}

fn eval_func(name: &str, a: &[f64]) -> Option<f64> {
    let x = *a.first()?;
    Some(match name {
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "exp" => x.exp(),
        "log" | "ln" => x.ln(),
        "sqrt" => x.sqrt(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "abs" => x.abs(),
        // Principal branch only, and `None` (→ "unknown construct", → skip)
        // rather than `NaN` below `−1/e`, so a sample outside `W₀`'s domain is
        // no information instead of a fake disagreement.  Reachable because
        // the separable class inverts `k·log y + m·y = T` through `W`.
        "lambert_w" => return crate::special::lambert_w0(x),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Parametric verification: sampling over free parameters, evaluated in C
// ---------------------------------------------------------------------------

/// Deterministic parameter assignments.  Chosen so that a two-parameter
/// equation such as `y'' + 2ζω y' + ω² y = 0` is sampled on *both* sides of its
/// discriminant — `ζ < 1` (complex roots) and `ζ > 1` (real roots) — rather
/// than only on the side the real evaluator happens to reach.
///
/// # Both signs, and why that is safe here
///
/// The first three rows are positive; the last three carry the sign patterns a
/// two-parameter equation needs to be sampled in all four quadrants —
/// `(−, +)`, `(+, −)`, `(−, −)`.  Sampling positive values only certified a
/// candidate that is right for `Kₘ > 0` and wrong for `Kₘ < 0`, and `OdeInput`
/// carries no assumption that would justify the restriction.
///
/// "Add negative values" is not on its own a safe change: many correct answers
/// are legitimately domain-limited, and a `log k` or `√k` in a candidate is
/// genuinely undefined at `k < 0`.  Three things make it safe:
///
/// * The parametric gate evaluates over **ℂ**, where `log(−1.3)` and `√(−1.3)`
///   are ordinary finite numbers on the principal branch, and where a correct
///   candidate's analytic continuation is still correct.  The negative rows
///   therefore mostly *resolve* rather than dropping out.
/// * Where a sample does not resolve, the three-way classification the whole
///   gate is built on books it as **no information** — never as agreement and
///   never as disagreement.  A row that is entirely unevaluable raises
///   `sets_unresolved`, which cannot refuse anything on its own; only
///   `disagree` and `blowup_at_regular_point` can.
/// * The bar is `sets_agreeing ≥ 2` out of six rows rather than out of three,
///   so a candidate that resolves only on the positive side is still certified
///   on the evidence it does produce.
///
/// Measured on the dsolve corpus (`corpus::corpus_report`), the six rows leave
/// all 121 of 125 solved entries solved, with no entry changing status in
/// either direction.  Nor are the new rows inert: running the corpus with the
/// three **negative** rows *alone* also solves 121 of 125, so every
/// parameterised entry resolves and agrees there rather than dropping out of
/// the evidence.
pub(crate) const PARAM_SETS: [&[f64]; 6] = [
    &[1.7, 0.6, 2.3, 1.1, 0.4],
    &[0.37, 1.9, 0.83, 2.7, 1.3],
    &[2.9, 0.45, 1.15, 0.71, 3.3],
    &[-1.3, 0.9, -2.1, 1.6, -0.55],
    &[0.62, -1.45, 2.05, -0.78, 1.1],
    &[-2.4, -0.83, -1.35, -3.1, -0.6],
];

/// Bind `params` to the values of `PARAM_SETS[set]`.
///
/// The one sampler both parameter-aware gates draw from — the complex
/// [`parametric_report`] builds its own `C64` environment from the same rows,
/// and the real [`implicit_numeric_report`] uses this one directly.  Keeping a
/// single table means a parameter is given the same value whichever gate asks,
/// and there is one place to look when a sampled value has to change.
fn parameter_env(params: &[ExprId], set: usize) -> HashMap<ExprId, f64> {
    let values = PARAM_SETS[set % PARAM_SETS.len()];
    params
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, values[i % values.len()]))
        .collect()
}

/// Agreeing samples required before a *parametric* numeric certificate issues.
///
/// Higher than [`MIN_AGREEING_SAMPLES`] because the grid is far larger
/// and because an identity in the parameters is a stronger claim than an
/// identity at fixed coefficients: it has to hold on an open set, not at a
/// point.
const PARAM_MIN_AGREEING: usize = 12;

/// Relative band for "this sample of the residual is zero".
///
/// The residual's natural magnitude varies by orders across the parameter grid
/// (`e^{λ x}` with a sampled `λ`), so the absolute [`ZERO_TOL`] would be a
/// different test at each corner of it.
const PARAM_REL_TOL: f64 = 1e-7;

/// [`NumericReport`] plus the parameter-set bookkeeping.
#[derive(Default, Debug, Clone, Copy)]
struct ParametricReport {
    inner: NumericReport,
    /// Parameter sets that produced at least one agreeing sample.
    sets_agreeing: usize,
    /// Parameter sets that produced no resolved sample at all.
    sets_unresolved: usize,
}

impl ParametricReport {
    /// Two agreeing parameter sets are required, not one: a candidate can be
    /// right on one branch of the discriminant and wrong on the other, and a
    /// single set cannot tell those apart.
    fn certifies(&self) -> bool {
        !self.inner.unevaluable
            && self.inner.disagree == 0
            && self.inner.blowup_at_regular_point == 0
            && self.inner.agree >= PARAM_MIN_AGREEING
            && self.sets_agreeing >= 2
    }

    /// Did the complex pass see something that counts *against* the candidate?
    ///
    /// Only this may end the verification in a refusal.  Everything else the
    /// pass can report — nothing evaluated, too few resolved samples, only one
    /// parameter set resolving — is an absence of information, not evidence.
    fn has_counterevidence(&self) -> bool {
        self.inner.disagree > 0 || self.inner.blowup_at_regular_point > 0
    }
}

impl fmt::Display for ParametricReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}; {} parameter sets agreeing, {} with no resolved sample",
            self.inner, self.sets_agreeing, self.sets_unresolved
        )
    }
}

/// Sample the residual over `x` x integration constants x parameter values.
fn parametric_report(
    input: &OdeInput,
    residual: ExprId,
    candidate_derivs: &[ExprId],
    constants: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
) -> ParametricReport {
    parametric_report_over(
        &PARAM_SETS,
        input,
        residual,
        candidate_derivs,
        constants,
        params,
        pool,
    )
}

/// [`parametric_report`] restricted to `rows` of the parameter table.
///
/// Split out so a test can ask what a *subset* of [`PARAM_SETS`] would have
/// concluded — which is the only way to pin that the negative rows are the
/// thing catching a candidate the positive rows certify.
fn parametric_report_over(
    rows: &[&[f64]],
    input: &OdeInput,
    residual: ExprId,
    candidate_derivs: &[ExprId],
    constants: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
) -> ParametricReport {
    let mut report = ParametricReport::default();
    for &ps in rows {
        let mut env: HashMap<ExprId, C64> = HashMap::new();
        for (i, &p) in params.iter().enumerate() {
            env.insert(p, C64::real(ps[i % ps.len()]));
        }
        let before = report.inner.agree;
        for cs in CONST_SETS {
            for (i, &c) in constants.iter().enumerate() {
                env.insert(c, C64::real(cs[i % cs.len()]));
            }
            for &xv in &X_SAMPLES {
                env.insert(input.x, C64::real(xv));
                match eval_complex(residual, &env, pool) {
                    Some(v) if v.is_finite() => {
                        record_parametric(&mut report.inner, v, candidate_derivs, &env, pool)
                    }
                    Some(_) => classify_nonfinite_complex(
                        input,
                        candidate_derivs,
                        &env,
                        xv,
                        pool,
                        &mut report.inner,
                    ),
                    None => {
                        report.inner.unevaluable = true;
                        return report;
                    }
                }
            }
        }
        if report.inner.agree > before {
            report.sets_agreeing += 1;
        } else if report.inner.disagree == 0 && report.inner.blowup_at_regular_point == 0 {
            report.sets_unresolved += 1;
        }
    }
    report
}

/// Bucket a finite complex residual, with the zero band scaled by how large the
/// candidate itself is at this sample.
fn record_parametric(
    report: &mut NumericReport,
    v: C64,
    candidate_derivs: &[ExprId],
    env: &HashMap<ExprId, C64>,
    pool: &ExprPool,
) {
    let mut scale = 1.0_f64;
    for &d in candidate_derivs {
        if let Some(dv) = eval_complex(d, env, pool) {
            if dv.is_finite() {
                scale = scale.max(dv.abs());
            }
        }
    }
    if v.abs() <= PARAM_REL_TOL * scale {
        report.agree += 1;
    } else {
        report.disagree += 1;
    }
}

/// The complex analogue of [`classify_nonfinite`]: same three outcomes, same
/// precedence, with the equation and the candidate probed over C.
fn classify_nonfinite_complex(
    input: &OdeInput,
    candidate_derivs: &[ExprId],
    env: &HashMap<ExprId, C64>,
    xv: f64,
    pool: &ExprPool,
    report: &mut NumericReport,
) {
    if !ode_is_regular_at_complex(input, env, xv, pool) {
        report.skipped_singular_ode += 1;
        return;
    }
    let mut vals = Vec::with_capacity(candidate_derivs.len());
    for &d in candidate_derivs {
        match eval_complex(d, env, pool) {
            Some(v) if v.is_finite() => vals.push(v),
            Some(_) => {
                report.blowup_at_regular_point += 1;
                return;
            }
            None => {
                report.skipped_unknown_construct += 1;
                return;
            }
        }
    }
    let mut eq_env = env.clone();
    eq_env.insert(input.x, C64::real(xv));
    eq_env.insert(input.y, vals[0]);
    for (k, &dsym) in input.derivs.iter().enumerate() {
        eq_env.insert(dsym, vals[k + 1]);
    }
    match eval_complex(input.equation, &eq_env, pool) {
        Some(v) if v.is_finite() => record_parametric(report, v, candidate_derivs, env, pool),
        _ => report.skipped_singular_ode += 1,
    }
}

/// Is the equation itself finite at this `x` and this parameter assignment,
/// candidate aside?  Probes several finite states, as [`ode_is_regular_at`]
/// does.
fn ode_is_regular_at_complex(
    input: &OdeInput,
    env: &HashMap<ExprId, C64>,
    xv: f64,
    pool: &ExprPool,
) -> bool {
    const PROBES: [f64; 4] = [1.0, 2.5, 0.5, -1.5];
    PROBES.iter().any(|&p| {
        let mut e = env.clone();
        e.insert(input.x, C64::real(xv));
        e.insert(input.y, C64::real(p));
        for (k, &dsym) in input.derivs.iter().enumerate() {
            e.insert(dsym, C64::real(p + 0.25 * (k as f64 + 1.0)));
        }
        matches!(eval_complex(input.equation, &e, pool), Some(v) if v.is_finite())
    })
}

// ---------------------------------------------------------------------------
// A minimal complex double
// ---------------------------------------------------------------------------

/// `re + i*im`, with principal branches for `log`, `sqrt` and `pow`.
///
/// Deliberately small: the only expressions it has to evaluate are the ones
/// `dsolve` itself manufactures (exponentials, powers, the elementary
/// functions the real [`eval`] already handles) plus whatever the caller wrote
/// in the equation.  Anything else returns `None`, which the gate reads as *no
/// information* rather than as agreement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    pub(crate) fn real(re: f64) -> Self {
        C64 { re, im: 0.0 }
    }
    pub(crate) fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    pub(crate) fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub(crate) fn add(self, o: C64) -> C64 {
        C64 {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    pub(crate) fn sub(self, o: C64) -> C64 {
        C64 {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
    pub(crate) fn mul(self, o: C64) -> C64 {
        C64 {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    pub(crate) fn div(self, o: C64) -> C64 {
        let d = o.re * o.re + o.im * o.im;
        C64 {
            re: (self.re * o.re + self.im * o.im) / d,
            im: (self.im * o.re - self.re * o.im) / d,
        }
    }
    fn neg(self) -> C64 {
        C64 {
            re: -self.re,
            im: -self.im,
        }
    }
    fn exp(self) -> C64 {
        let m = self.re.exp();
        C64 {
            re: m * self.im.cos(),
            im: m * self.im.sin(),
        }
    }
    fn ln(self) -> C64 {
        C64 {
            re: self.abs().ln(),
            im: self.im.atan2(self.re),
        }
    }
    fn powi(self, n: i64) -> C64 {
        if n < 0 {
            return C64::real(1.0).div(self.powi(-n));
        }
        let mut acc = C64::real(1.0);
        for _ in 0..n {
            acc = acc.mul(self);
        }
        acc
    }
    /// `self^w` on the principal branch.  Integer exponents go through repeated
    /// multiplication, which is exact at `0` (where `exp(w*log 0)` is not) and
    /// avoids a branch choice the caller did not ask for.
    fn pow(self, w: C64) -> C64 {
        if w.im == 0.0 && w.re.fract() == 0.0 && w.re.abs() <= 64.0 {
            return self.powi(w.re as i64);
        }
        if self.re == 0.0 && self.im == 0.0 {
            return if w.re > 0.0 {
                C64::real(0.0)
            } else {
                C64::real(f64::INFINITY)
            };
        }
        w.mul(self.ln()).exp()
    }
    fn sin(self) -> C64 {
        C64 {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }
    fn cos(self) -> C64 {
        C64 {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }
    fn sinh(self) -> C64 {
        C64 {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }
    fn cosh(self) -> C64 {
        C64 {
            re: self.re.cosh() * self.im.cos(),
            im: self.re.sinh() * self.im.sin(),
        }
    }
}

const I: C64 = C64 { re: 0.0, im: 1.0 };

/// Evaluate `expr` over C.  `None` for constructs this evaluator does not know,
/// which the gate treats as no information (never as agreement).
///
/// `pi` and the imaginary unit resolve to their own values without being in
/// `env`; see [`crate::eval::symbols`] for why binding them to samples instead
/// is a false-refusal machine.
pub(crate) fn eval_complex(
    expr: ExprId,
    env: &HashMap<ExprId, C64>,
    pool: &ExprPool,
) -> Option<C64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(C64::real(n.0.to_f64())),
        ExprData::Rational(r) => {
            let (num, den) = r.0.clone().into_numer_denom();
            Some(C64::real(num.to_f64() / den.to_f64()))
        }
        ExprData::Float(f) => Some(C64::real(f.inner.to_f64())),
        ExprData::Symbol { .. } => env.get(&expr).copied().or_else(|| {
            if is_pi(expr, pool) {
                Some(C64::real(std::f64::consts::PI))
            } else if pool.is_imaginary_unit(expr) {
                Some(I)
            } else {
                None
            }
        }),
        ExprData::Add(args) => {
            let mut s = C64::real(0.0);
            for a in args {
                s = s.add(eval_complex(a, env, pool)?);
            }
            Some(s)
        }
        ExprData::Mul(args) => {
            let mut p = C64::real(1.0);
            for a in args {
                p = p.mul(eval_complex(a, env, pool)?);
            }
            Some(p)
        }
        ExprData::Pow { base, exp } => {
            let b = eval_complex(base, env, pool)?;
            let e = eval_complex(exp, env, pool)?;
            Some(b.pow(e))
        }
        ExprData::Func { name, args } => {
            let v: Vec<C64> = args
                .iter()
                .map(|&a| eval_complex(a, env, pool))
                .collect::<Option<_>>()?;
            eval_func_complex(&name, &v)
        }
        _ => None,
    }
}

fn eval_func_complex(name: &str, a: &[C64]) -> Option<C64> {
    let z = *a.first()?;
    let one = C64::real(1.0);
    let half = C64::real(0.5);
    Some(match name {
        "sin" => z.sin(),
        "cos" => z.cos(),
        "tan" => z.sin().div(z.cos()),
        "exp" => z.exp(),
        "log" | "ln" => z.ln(),
        "sqrt" => z.pow(half),
        "sinh" => z.sinh(),
        "cosh" => z.cosh(),
        "tanh" => z.sinh().div(z.cosh()),
        // atan z = (i/2)*(log(1 - i z) - log(1 + i z)); asin/acos follow.
        "atan" => I.div(C64::real(2.0)).mul(
            one.add(I.mul(z).neg())
                .ln()
                .add(one.add(I.mul(z)).ln().neg()),
        ),
        "asin" => I
            .neg()
            .mul(I.mul(z).add(one.add(z.mul(z).neg()).pow(half)).ln()),
        "acos" => C64::real(std::f64::consts::FRAC_PI_2)
            .add(I.mul(I.mul(z).add(one.add(z.mul(z).neg()).pow(half)).ln())),
        "abs" => C64::real(z.abs()),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Implicit solutions
// ---------------------------------------------------------------------------

/// Re-verify a returned [`DsolveBranch`](super::DsolveBranch) in whichever form it came in.
///
/// The gate inside each class already ran; this is the entry point for callers
/// that want to check a returned answer *independently* of it — which the
/// corpus harness and every solving test do, on the principle that a gate is
/// not allowed to be its own witness.
#[cfg(test)]
pub(crate) fn solution_is_verified(
    input: &OdeInput,
    sol: &super::DsolveBranch,
    pool: &ExprPool,
) -> Result<(), DsolveError> {
    match sol.form {
        super::SolutionForm::Explicit(y) => residual_is_zero(input, y, &sol.constants, pool),
        super::SolutionForm::Implicit(g) => {
            implicit_relation_is_zero(input, g, &sol.constants, pool)
        }
    }
}

/// Verify an implicit general solution `relation(x, y) = 0` of a *first-order*
/// ODE.
///
/// # What is checked, and why it is the right thing
///
/// The implicit function theorem turns the relation into a slope field:
/// wherever `G_y ≠ 0`, the level set through a point has slope
/// `y' = −Gₓ/G_y`.  The relation is a general solution exactly when
///
/// ```text
///     F(x, y, −Gₓ(x,y)/G_y(x,y)) ≡ 0     for all (x, y) in the region,
/// ```
///
/// which says every level set of `G` — i.e. every member of the one-parameter
/// family — solves the ODE.  Note that this is an identity in **two** free
/// variables, so it is a *stronger* statement than the explicit gate's
/// one-variable identity, and it needs no root-finding: the point `(x, y)`
/// carries its own constant.
///
/// # The precondition that makes free `(x, y)` sampling legitimate
///
/// Sampling `y` independently of `x` is only sound when the slope field does
/// not depend on the integration constant.  If it does — `relation = y − C·eˣ`
/// has `Gₓ/G_y = −C·eˣ`, a slope that is only correct *on* the curve `y = C eˣ`
/// — then the identity holds on each curve and nowhere else, and sampling off
/// the curve would reject a perfectly good answer.  So `Gₓ` and `G_y` are
/// required to be free of every constant, which is the case for the
/// `G(x, y) − C` shape every class here produces, and the verification is
/// declined otherwise rather than being run in a form that cannot conclude.
///
/// A relation must also mention at least one constant (a *general* solution is
/// a family, not one curve) and must genuinely depend on `y`.
pub(crate) fn implicit_relation_is_zero(
    input: &OdeInput,
    relation: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveError> {
    let outcome = implicit_inner(input, relation, constants, pool);
    #[cfg(test)]
    GATE_TALLY.with(|t| {
        let (offered, refused) = t.get();
        t.set((offered + 1, refused + usize::from(outcome.is_err())));
    });
    outcome
}

fn implicit_inner(
    input: &OdeInput,
    relation: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveError> {
    if input.derivs.len() != 1 {
        return Err(DsolveError::VerificationFailed(
            "implicit solutions are only verified for first-order equations".to_string(),
        ));
    }
    let yp = input.derivs[0];
    if !contains(relation, input.y, pool) {
        return Err(DsolveError::VerificationFailed(
            "implicit relation does not depend on y".to_string(),
        ));
    }
    if !constants.iter().any(|&c| contains(relation, c, pool)) {
        return Err(DsolveError::VerificationFailed(
            "implicit relation carries no integration constant, so it is not a \
             general solution"
                .to_string(),
        ));
    }

    let gx = ddx(relation, input.x, pool)?;
    let gy = ddx(relation, input.y, pool)?;
    if super::is_zero(gy, pool) {
        return Err(DsolveError::VerificationFailed(
            "∂G/∂y is identically zero: the relation defines no slope field".to_string(),
        ));
    }
    for &c in constants {
        if contains(gx, c, pool) || contains(gy, c, pool) {
            return Err(DsolveError::VerificationFailed(
                "the slope field −Gx/Gy still mentions an integration constant, so \
                 the relation cannot be checked off its own level sets"
                    .to_string(),
            ));
        }
    }

    // y' = −Gx/Gy, substituted into the equation.
    let slope = super::div(
        simp(pool.mul(vec![pool.integer(-1_i32), gx]), pool),
        gy,
        pool,
    );
    let residual = simp(subs1(input.equation, yp, slope, pool), pool);
    if is_symbolic_zero(residual, pool) || is_symbolic_zero(super::simp_plain(residual, pool), pool)
    {
        return Ok(());
    }

    let report = implicit_numeric_report(input, residual, slope, gy, constants, pool);
    if report.certifies() {
        return Ok(());
    }
    Err(DsolveError::VerificationFailed(format!(
        "implicit relation did not reduce to zero ({report}): {}",
        pool.display(residual)
    )))
}

/// `y` sample points for the implicit gate.  Positive and bounded away from
/// zero: `log y` and `1/y` are the two commonest things a separable
/// antiderivative produces, and both are real and finite here.
const Y_SAMPLES: [f64; 5] = [0.37, 0.83, 1.4, 2.1, 3.3];

/// Sample `F(x, y, −Gₓ/G_y) ≈ 0` over an `(x, y)` grid.
///
/// The classification of a non-finite sample follows [`classify_nonfinite`]'s
/// discipline, with one deliberate difference: a point where the *slope* blows
/// up is a **skip**, not a disagreement.  A vertical tangent is ordinary
/// behaviour for a level curve — `x² + y² = C` has one at `y = 0` — and says
/// nothing about whether the relation solves the ODE, whereas an explicit
/// candidate blowing up where the ODE is regular is evidence that it is wrong.
fn implicit_numeric_report(
    input: &OdeInput,
    residual: ExprId,
    slope: ExprId,
    gy: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> NumericReport {
    let sources = [input.equation, residual, slope];
    let free = free_parameters(input, &sources, constants, pool);
    let mut report = NumericReport::default();
    for set in 0..PARAM_SETS.len() {
        let params = parameter_env(&free, set);
        let mut env = params.clone();
        for &xv in &X_SAMPLES {
            env.insert(input.x, xv);
            for &yv in &Y_SAMPLES {
                env.insert(input.y, yv);
                match eval(residual, &env, pool) {
                    Some(v) if v.is_finite() => {
                        record_scaled(&mut report, v, input, &env, slope, pool)
                    }
                    Some(_) => classify_implicit_nonfinite(
                        input,
                        slope,
                        gy,
                        &env,
                        &params,
                        xv,
                        yv,
                        pool,
                        &mut report,
                    ),
                    None => {
                        report.unevaluable = true;
                        return report;
                    }
                }
            }
        }
    }
    report
}

/// Bucket a finite residual against the *scale of the terms that produced it*.
///
/// The implicit residual is the equation evaluated at a slope, and the slope
/// can be large: `(Kₘ + y)·y' + Vₘ·y` at `y' = −40` has terms of size 100, and
/// a cancellation between them is only meaningful to about `100·ε`.  An
/// absolute `1e-6` would be simultaneously too strict there and too lax for an
/// equation whose terms are all `1e-9`.  The scale is the largest magnitude
/// among the equation's own top-level additive terms at this sample.
fn record_scaled(
    report: &mut NumericReport,
    v: f64,
    input: &OdeInput,
    env: &HashMap<ExprId, f64>,
    slope: ExprId,
    pool: &ExprPool,
) {
    let scale = equation_term_scale(input, env, slope, pool).unwrap_or(0.0);
    if v.abs() <= ZERO_TOL * (1.0 + scale) {
        report.agree += 1;
    } else {
        report.disagree += 1;
    }
}

/// Largest magnitude among the top-level additive terms of `input.equation`
/// with `y' = slope`, at the sample in `env`.  `None` if any term is not a
/// finite real there.
fn equation_term_scale(
    input: &OdeInput,
    env: &HashMap<ExprId, f64>,
    slope: ExprId,
    pool: &ExprPool,
) -> Option<f64> {
    let eq = subs1(input.equation, input.derivs[0], slope, pool);
    let terms: Vec<ExprId> = match pool.get(eq) {
        ExprData::Add(args) => args,
        _ => vec![eq],
    };
    let mut scale: f64 = 0.0;
    for t in terms {
        let v = eval(t, env, pool)?;
        if !v.is_finite() {
            return None;
        }
        scale = scale.max(v.abs());
    }
    Some(scale)
}

#[allow(clippy::too_many_arguments)]
fn classify_implicit_nonfinite(
    input: &OdeInput,
    slope: ExprId,
    gy: ExprId,
    env: &HashMap<ExprId, f64>,
    params: &HashMap<ExprId, f64>,
    xv: f64,
    yv: f64,
    pool: &ExprPool,
    report: &mut NumericReport,
) {
    // 1. Is the equation itself well-defined at this `x`, the relation aside?
    if !ode_is_regular_at(input, xv, params, pool) {
        report.skipped_singular_ode += 1;
        return;
    }
    // 2. Does the relation define a finite slope here?  A vanishing `G_y` (a
    //    vertical tangent) or an unevaluable slope carries no information.
    match (eval(gy, env, pool), eval(slope, env, pool)) {
        (Some(g), Some(s)) if g.is_finite() && g.abs() > 1e-12 && s.is_finite() => {
            // 3. Both sides are finite, so the non-finiteness was an artefact of
            //    the residual's algebraic form.  Re-ask the original equation.
            let mut eq_env: HashMap<ExprId, f64> = params.clone();
            eq_env.insert(input.x, xv);
            eq_env.insert(input.y, yv);
            eq_env.insert(input.derivs[0], s);
            match eval(input.equation, &eq_env, pool) {
                Some(v) if v.is_finite() => record_scaled(report, v, input, env, slope, pool),
                // Singular at this state rather than at this `x`.
                _ => report.skipped_singular_ode += 1,
            }
        }
        (None, _) | (_, None) => report.skipped_unknown_construct += 1,
        _ => report.skipped_singular_ode += 1,
    }
}

#[cfg(test)]
mod tests;
