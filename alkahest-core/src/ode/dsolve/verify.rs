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

use super::{contains, ddx, simp, subs1, DsolveError, OdeInput};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::{HashMap, HashSet};
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

    // Numeric fallback: sample x over several constant assignments.
    let report = numeric_report(input, residual, &candidate_derivs, constants, pool);
    if report.certifies() {
        return Ok(());
    }

    Err(DsolveError::VerificationFailed(format!(
        "residual did not reduce to zero ({report}): {}",
        pool.display(residual)
    )))
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

/// Deterministic pseudo-random constant assignments (no rng dependency).
/// Constants are kept positive and reasonably large so that radicands such as
/// `sqrt(4·C − 3x²)` arising from quadratic-implicit solutions stay real over
/// the (small) x-sample range.
const CONST_SETS: [&[f64]; 3] = [
    &[5.7, 4.3, 6.4, 5.1, 4.9],
    &[8.5, 7.8, 6.6, 9.2, 7.1],
    &[12.3, 10.0, 11.7, 10.5, 9.4],
];

/// Values bound to the equation's *symbolic parameters* (see
/// [`parameter_env`]).  Kept `O(1)` and positive: a parameter is typically a
/// rate or a half-saturation constant, appears under a `log` or in an exponent
/// as often as not, and a value of `12` in an exponent overflows an `f64`
/// where `1.3` does not.  One row per row of [`CONST_SETS`].
///
/// Positive-only is a deliberate limit, and it is the same convention the rest
/// of this module already commits to by writing `log u` and `exp_of`'s `u^c`:
/// a candidate that is right for `Kₘ > 0` and wrong for `Kₘ < 0` is certified
/// here.  Sampling negative values instead would reject correct answers to
/// equations whose parameters are physically positive far more often than it
/// would catch anything, and the sign condition is not represented in the
/// input — `OdeInput` carries no assumptions.
const PARAM_SETS: [&[f64]; 3] = [
    &[1.3, 2.1, 0.7, 1.9, 3.1],
    &[0.9, 1.7, 2.3, 1.1, 0.6],
    &[2.7, 0.8, 1.5, 3.3, 2.2],
];

/// `x` sample points, shared by the explicit and implicit gates.
const X_SAMPLES: [f64; 5] = [0.11, 0.27, 0.43, 0.61, 0.79];

/// Bind every symbol of `exprs` that is neither an ODE variable nor an
/// integration constant to a value from `PARAM_SETS[set]`.
///
/// Without this a residual mentioning a symbolic coefficient (`y' + kₑ·y = 0`,
/// Michaelis–Menten's `Kₘ`, `Vₘ`) is *unevaluable* — [`eval`] has nothing to
/// put in its place — so the numeric half of the gate could not run at all and
/// every parameterised ODE had to certify symbolically or be declined.  The
/// binding is sound in the direction that matters: a candidate correct for all
/// parameter values disagrees at none of them, and a disagreement at one
/// value is still a disagreement.
fn parameter_env(
    exprs: &[ExprId],
    input: &OdeInput,
    constants: &[ExprId],
    set: usize,
    pool: &ExprPool,
) -> HashMap<ExprId, f64> {
    let mut bound: HashSet<ExprId> = HashSet::new();
    bound.insert(input.x);
    bound.insert(input.y);
    bound.extend(input.derivs.iter().copied());
    bound.extend(constants.iter().copied());

    let mut params: Vec<ExprId> = Vec::new();
    for &e in exprs {
        collect_symbols(e, pool, &mut |s| {
            if !bound.contains(&s) && !params.contains(&s) {
                params.push(s);
            }
        });
    }
    // Sort by `ExprId` so the assignment does not depend on traversal order.
    params.sort();
    let values = PARAM_SETS[set % PARAM_SETS.len()];
    params
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, values[i % values.len()]))
        .collect()
}

fn collect_symbols(expr: ExprId, pool: &ExprPool, out: &mut impl FnMut(ExprId)) {
    match pool.get(expr) {
        ExprData::Symbol { .. } => out(expr),
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            for a in args {
                collect_symbols(a, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_symbols(base, pool, out);
            collect_symbols(exp, pool, out);
        }
        _ => {}
    }
}

/// Numerically check residual ≈ 0 at several `x` over random constants.
fn numeric_report(
    input: &OdeInput,
    residual: ExprId,
    candidate_derivs: &[ExprId],
    constants: &[ExprId],
    pool: &ExprPool,
) -> NumericReport {
    let mut sources: Vec<ExprId> = vec![input.equation, residual];
    sources.extend_from_slice(candidate_derivs);

    let mut report = NumericReport::default();
    for (set, cs) in CONST_SETS.iter().enumerate() {
        let params = parameter_env(&sources, input, constants, set, pool);
        let mut env: HashMap<ExprId, f64> = params.clone();
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
                    &params,
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
pub(crate) fn eval(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (num, den) = r.0.clone().into_numer_denom();
            Some(num.to_f64() / den.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { .. } => env.get(&expr).copied(),
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
// Implicit solutions
// ---------------------------------------------------------------------------

/// Re-verify a returned [`DsolveSolution`] in whichever form it came in.
///
/// The gate inside each class already ran; this is the entry point for callers
/// that want to check a returned answer *independently* of it — which the
/// corpus harness and every solving test do, on the principle that a gate is
/// not allowed to be its own witness.
#[cfg(test)]
pub(crate) fn solution_is_verified(
    input: &OdeInput,
    sol: &super::DsolveSolution,
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
    let mut report = NumericReport::default();
    for set in 0..CONST_SETS.len() {
        let params = parameter_env(&sources, input, constants, set, pool);
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
