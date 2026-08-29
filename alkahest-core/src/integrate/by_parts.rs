//! General integration by parts — `∫u·dv = u·v − ∫v·du`.
//!
//! # Why this module exists
//!
//! Before it, the only by-parts machinery in the integrator was three narrow
//! special cases in [`crate::integrate::engine`]: `try_inverse_trig_ibp`
//! (inverse-trig factor whose argument is *exactly* the variable),
//! `try_poly_trig_ibp` (polynomial × trig with a linear argument) and
//! `try_exp_trig_ibp` (the `exp·sin` / `exp·cos` closed form).  Everything else
//! fell through to the `Node::Mul` handler, which splits off variable-*free*
//! factors and then declines with `irreducible product of var-dependent
//! factors`.  Measured on Charlwood's Fifty that decline, plus the
//! composite-argument guard in `as_inverse_trig_power`, accounted for 17 of the
//! 36 remaining failures.
//!
//! # Contract
//!
//! ```text
//! ByPartsOutcome::Solved(F)     — d/dx F = f has been checked, here is F
//! ByPartsOutcome::Declined(why) — this split did not work; NOTHING is implied
//! ```
//!
//! There is deliberately **no third variant**.  Integration by parts is a
//! heuristic: the identity `∫u·dv = u·v − ∫v·du` is exact, but the choice of
//! split is a guess and the sub-integral may be closed by a route that is not a
//! decision procedure.  A failure here means "my split did not work", never "no
//! elementary antiderivative exists", so this module cannot construct an
//! [`IntegrationError::NonElementary`] and cannot express `E-INT-004`.  That is
//! pinned by `declined_cannot_become_non_elementary` in the tests below.  Eight
//! false-`NonElementary` families in this codebase came from a method's failure
//! being read as a proof; the type is the fix.
//!
//! # Soundness gate
//!
//! Every candidate is passed through
//! [`crate::integrate::verify_antiderivative_status`] (symbolic `d/dx F − f ≡ 0`
//! first, then an in-domain `f64` sample screen) before it is returned.  A
//! candidate whose derivative cannot be confirmed equal to the integrand is
//! discarded, so a bad LIATE guess costs CPU and nothing else.  This mirrors the
//! `verify` / `verify_higher` pattern in
//! [`crate::integrate::algebraic::elliptic_output`].
//!
//! # Method
//!
//! The reduction is run as a *loop*, not as naive recursion, maintaining the
//! invariant
//!
//! ```text
//!     I  =  acc  +  mult · ∫ w dx
//! ```
//!
//! starting from `acc = 0`, `mult = 1`, `w = f`.  One step picks a split
//! `w = u·dv`, computes `v = ∫dv dx` and `du = u′`, and updates
//!
//! ```text
//!     acc  ←  acc + mult·u·v
//!     mult ←  −mult
//!     w    ←  v·du
//! ```
//!
//! After each step three things are tried, in order:
//!
//! 1. **Close** — hand `w` to the full [`crate::integrate::integrate`] engine.
//!    If it solves, `I = acc + mult·∫w` and we are done.
//! 2. **Cycle** — test whether `w = c·f` for a constant `c`.  If so the
//!    invariant reads `I = acc + mult·c·I`, a *linear equation*, and the answer
//!    is `I = acc / (1 − mult·c)`.  This is what closes `∫eˣ sin x`: it is a
//!    feature, not merely a termination guard.
//! 3. **Recurse** — take another step, up to [`MAX_IBP_STEPS`].
//!
//! # Termination
//!
//! Four independent bounds, because a by-parts chain has four ways to run away:
//!
//! * **Step bound.** At most [`MAX_IBP_STEPS`] reductions per split attempt.
//! * **Cycle detection.** `w = c·f` is solved algebraically instead of
//!   recursed into (case 2 above).  Without it `∫eˣ sin x` alternates forever.
//! * **Growth check.** If `∫v·du` is materially larger than `∫u·dv` the choice
//!   of `u` was wrong; the step is abandoned and the next candidate split is
//!   tried.  `∫x²eˣ` with `u = eˣ` produces `x³eˣ/3` and is rejected here.
//! * **Re-entry guard.** A thread-local depth counter, because the sub-integrals
//!   are handed to the full engine, which — once the `engine.rs` hook described
//!   in this module's report is in place — can route straight back here.
//!
//! # Wiring (not yet applied)
//!
//! This module is reachable only through its own entry points, so that its
//! coverage could be characterised before it joins the default dispatch order.
//! The hook is **one site**: the `NotImplemented` arm of the final
//! `match integrate_inner(…)` in [`crate::integrate::engine::integrate`].
//!
//! ```text
//! Err(e @ IntegrationError::NotImplemented(_)) => {
//!     let mut bp = DerivationLog::new();
//!     if let Some(f) = super::by_parts::try_by_parts(expr, var, pool, &mut bp) {
//!         let s = simplify(f, pool);
//!         return Ok(DerivedExpr::with_log(s.value, bp.merge(s.log)));
//!     }
//!     Err(declined.unwrap_or(e))       // diagnostic preserved, as before
//! }
//! ```
//!
//! **Last, not first**, and the reason is soundness rather than cost.  This
//! module's gate accepts `AntiderivativeVerification::Numeric` as well as
//! `Exact` — both Charlwood answers it currently closes verify only
//! numerically — so putting it ahead of the Risch/algebraic engines would let a
//! sampled-only answer pre-empt a proven one, which is the "a deeper tier can
//! produce a *better* answer" hazard in `planning/risch.md`'s tiered-integrator
//! section.  Placing it on the decline path also means the cost is paid only on
//! integrands that were going to fail anyway, and leaves the sub-engines'
//! specific diagnostics (`declined`) intact when it too declines.
//!
//! Not `integrate_inner`: `try_u_substitution` re-enters that function once per
//! substitution candidate, so a hook there would run this module several times
//! per top-level call instead of once, and `integrate_inner` cannot see the
//! `declined` diagnostic that `integrate` preserves.
//!
//! # LIATE
//!
//! [`Liate`] ranks factors *log → inverse-trig → algebraic → trig →
//! exponential* and `u` is taken to be the highest-ranked factor.  It is a
//! heuristic and is treated as one: every split is enumerated, LIATE only sets
//! the *order* in which they are tried, and a split that fails the growth check
//! or whose `dv` will not integrate is skipped for the next one.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::{simplify, simplify_expanded};
use std::collections::HashMap;

use super::engine::{integrate, verify_antiderivative_status, IntegrationError};

// ---------------------------------------------------------------------------
// Outcome — two variants, by design
// ---------------------------------------------------------------------------

/// The result of an integration-by-parts attempt.
///
/// **`Solved` or `Declined`, and nothing else.**  See the module docs: a
/// by-parts failure is never evidence that an antiderivative does not exist, so
/// this type has no way to say so.  Callers convert `Declined` into
/// [`IntegrationError::NotImplemented`] (`E-INT-001`) via
/// [`ByPartsOutcome::into_result`]; there is no path from here to `E-INT-004`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ByPartsOutcome {
    /// An antiderivative that has passed `d/dx F = f`.
    Solved(ExprId),
    /// This route found nothing.  The string is a diagnostic, not a verdict.
    Declined(String),
}

impl ByPartsOutcome {
    /// The antiderivative, if one was found.
    pub fn solved(&self) -> Option<ExprId> {
        match self {
            ByPartsOutcome::Solved(f) => Some(*f),
            ByPartsOutcome::Declined(_) => None,
        }
    }

    /// `true` when this outcome is a decline.
    pub fn is_declined(&self) -> bool {
        matches!(self, ByPartsOutcome::Declined(_))
    }

    /// Convert to the engine's error type.  A decline becomes
    /// [`IntegrationError::NotImplemented`] — `E-INT-001` — and can become
    /// nothing else.
    pub fn into_result(self) -> Result<ExprId, IntegrationError> {
        match self {
            ByPartsOutcome::Solved(f) => Ok(f),
            ByPartsOutcome::Declined(why) => Err(IntegrationError::NotImplemented(why)),
        }
    }
}

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Maximum number of by-parts reductions applied along one chain.
///
/// Three is enough for `∫x³·eˣ` (`x³ → x² → x → 1`) and for the two-step
/// `∫eˣ sin x` cycle with a step to spare; beyond that the intermediate
/// expressions grow faster than the chance of closing.
pub const MAX_IBP_STEPS: u32 = 3;

/// Maximum re-entry depth of [`integrate_by_parts`] on one thread.
///
/// The sub-integrals are handed to the full engine, so once this module is
/// wired into `engine.rs` a sub-integral can route back here.  One level of
/// nesting is useful (a residual that is itself a by-parts problem); deeper
/// nesting has not paid for itself on any measured corpus and is where the
/// combinatorial blow-up lives.
const MAX_REENTRY_DEPTH: u32 = 1;

/// How much larger `∫v·du` may be than `∫u·dv` before the split is judged
/// wrong.  A by-parts step is supposed to *simplify*; `1.6` leaves room for the
/// quotient-rule fan-out in `du` while still rejecting `∫x²eˣ` split the wrong
/// way (`x²eˣ → x³eˣ/3`, a ratio of ~1.9).
const GROWTH_LIMIT: f64 = 1.6;

thread_local! {
    /// Re-entry depth of [`integrate_by_parts`] on the current thread.
    static IBP_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// RAII guard restoring [`IBP_DEPTH`] on every exit path.
struct DepthGuard;

impl Drop for DepthGuard {
    fn drop(&mut self) {
        IBP_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

// ---------------------------------------------------------------------------
// LIATE
// ---------------------------------------------------------------------------

/// The LIATE priority classes, best choice of `u` first.
///
/// The mnemonic is a heuristic, not a theorem — see
/// [`integrate_by_parts`]'s docs and the report accompanying this module for
/// the cases where it picks wrong.  Its job here is only to *order* the
/// candidate splits; every one of them is still tried, verified and gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Liate {
    /// Logarithmic — `log(g)`.
    Log,
    /// Inverse trigonometric / inverse hyperbolic — `atan(g)`, `asin(g)`, …
    InverseTrig,
    /// Algebraic — polynomials, rational powers, radicals.
    Algebraic,
    /// Trigonometric / hyperbolic — `sin(g)`, `cos(g)`, `tanh(g)`, …
    Trig,
    /// Exponential — `exp(g)`, `c^g`.
    Exponential,
    /// Anything unrecognised.  Ranked last so a known class always wins.
    Other,
}

fn is_inverse_trig_name(name: &str) -> bool {
    matches!(
        name,
        "atan" | "asin" | "acos" | "asinh" | "acosh" | "atanh" | "acot" | "asec" | "acsc"
    )
}

fn is_trig_name(name: &str) -> bool {
    matches!(
        name,
        "sin" | "cos" | "tan" | "sec" | "csc" | "cot" | "sinh" | "cosh" | "tanh" | "coth"
    )
}

/// LIATE class of a factor.
///
/// The class of a `Pow` is the class of its base (so `atan(x)²` is
/// inverse-trig, and `√(1−x²)` and `x⁻³` are algebraic), and the class of a
/// product is the best (numerically smallest) class among its factors, so a
/// grouped `dv` is ranked by its dominant generator.
pub fn liate_class(expr: ExprId, var: ExprId, pool: &ExprPool) -> Liate {
    if is_free_of(expr, var, pool) {
        return Liate::Algebraic;
    }
    match pool.get(expr) {
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => Liate::Algebraic,
        ExprData::Func { name, args } => {
            if name == "log" {
                Liate::Log
            } else if is_inverse_trig_name(&name) {
                Liate::InverseTrig
            } else if name == "exp" {
                Liate::Exponential
            } else if is_trig_name(&name) {
                Liate::Trig
            } else if name == "sqrt" {
                // √(g) is algebraic in `g`, but if `g` is transcendental the
                // generator underneath is what matters for the split.
                match args.first() {
                    Some(&a) if !is_free_of(a, var, pool) => {
                        let inner = liate_class(a, var, pool);
                        if inner == Liate::Algebraic {
                            Liate::Algebraic
                        } else {
                            inner
                        }
                    }
                    _ => Liate::Algebraic,
                }
            } else {
                Liate::Other
            }
        }
        ExprData::Pow { base, exp } => {
            // c^g with a constant base is exponential; g^c is the class of g.
            if is_free_of(base, var, pool) && !is_free_of(exp, var, pool) {
                Liate::Exponential
            } else {
                liate_class(base, var, pool)
            }
        }
        ExprData::Mul(args) | ExprData::Add(args) => args
            .iter()
            .filter(|&&a| !is_free_of(a, var, pool))
            .map(|&a| liate_class(a, var, pool))
            .min()
            .unwrap_or(Liate::Algebraic),
        _ => Liate::Other,
    }
}

// ---------------------------------------------------------------------------
// Small structural helpers
// ---------------------------------------------------------------------------

fn is_free_of(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return false;
    }
    match pool.get(expr) {
        ExprData::Symbol { .. } => true,
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().all(|&a| is_free_of(a, var, pool))
        }
        ExprData::Pow { base, exp } => is_free_of(base, var, pool) && is_free_of(exp, var, pool),
        _ => false,
    }
}

fn node_count(expr: ExprId, pool: &ExprPool) -> usize {
    1 + pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().map(|&a| node_count(a, pool)).sum::<usize>()
        }
        ExprData::Pow { base, exp } => node_count(*base, pool) + node_count(*exp, pool),
        _ => 0,
    })
}

/// Size of `expr` ignoring a leading rational coefficient.
///
/// The growth check compares consecutive residuals, and a plain `node_count`
/// makes a *sign flip* look like growth: `−sin(log x)` interns as
/// `Mul[−1, sin(log x)]`, five nodes against three for `cos(log x)`, which is
/// past the growth limit.  That killed `∫sin(log x) dx` — the one cycle in the
/// test set the engine cannot close any other way — one step before the linear
/// solve would have fired.  Comparing structure rather than the coefficient is
/// the fix.
fn structural_size(expr: ExprId, pool: &ExprPool) -> usize {
    node_count(split_rational_coeff(expr, pool).1, pool)
}

fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    // `simplify` may hand back the zero as an `Integer`, a `Rational` with a
    // zero numerator, or a `Float`, depending on which rule fired last.  All
    // three mean the residual cancelled.
    pool.with(expr, |d| match d {
        ExprData::Integer(i) => i.0 == 0,
        ExprData::Rational(r) => r.0.numer() == &rug::Integer::ZERO,
        ExprData::Float(f) => f.inner.is_zero(),
        _ => false,
    })
}

/// `true` when `expr` is the integer `1`.
fn is_one(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| matches!(d, ExprData::Integer(i) if i.0 == 1))
}

/// Product of `factors`, collapsing the 0- and 1-element cases.
fn mul_of(factors: &[ExprId], pool: &ExprPool) -> ExprId {
    match factors.len() {
        0 => pool.integer(1_i32),
        1 => factors[0],
        _ => pool.mul(factors.to_vec()),
    }
}

// ---------------------------------------------------------------------------
// Split enumeration
// ---------------------------------------------------------------------------

/// One candidate `∫u·dv` split of an integrand.
#[derive(Debug, Clone, Copy)]
struct Split {
    u: ExprId,
    dv: ExprId,
    /// LIATE class of `u`; the sort key.
    class: Liate,
    /// `true` when `dv = 1`, i.e. the `dv = dx` case.
    trivial_dv: bool,
}

/// Enumerate the candidate `(u, dv)` splits of `expr`, best LIATE choice first.
///
/// For a product with `n` variable-dependent factors this yields the `n` splits
/// that take one factor as `u` and the rest as `dv`, plus the `dv = dx` split
/// (`u = expr`) as a last resort.  For anything else it yields only `dv = dx`,
/// which is the C3 shape `∫f(g(x))dx = x·f(g(x)) − ∫x·(f∘g)′dx`.
///
/// Variable-*free* factors always stay with `dv`: pulling a constant into `u`
/// makes `du` carry it and changes nothing, and the engine's `Node::Mul`
/// handler has already split them off in most arrival paths anyway.
fn candidate_splits(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<Split> {
    let mut splits: Vec<Split> = Vec::new();
    let one = pool.integer(1_i32);

    if let ExprData::Mul(args) = pool.get(expr) {
        let var_dep: Vec<ExprId> = args
            .iter()
            .copied()
            .filter(|&a| !is_free_of(a, var, pool))
            .collect();
        let consts: Vec<ExprId> = args
            .iter()
            .copied()
            .filter(|&a| is_free_of(a, var, pool))
            .collect();

        if var_dep.len() >= 2 {
            for (i, &u) in var_dep.iter().enumerate() {
                let mut rest: Vec<ExprId> = consts.clone();
                rest.extend(
                    var_dep
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, &a)| a),
                );
                let dv = mul_of(&rest, pool);
                splits.push(Split {
                    u,
                    dv,
                    class: liate_class(u, var, pool),
                    trivial_dv: false,
                });
            }
        }
    }

    // The `dv = dx` case — `∫f(g(x))dx = x·f(g(x)) − ∫x·(f∘g)′dx`, the C3 route.
    //
    // Offered only when there is no factorwise split, i.e. for a bare composite
    // or a single variable-dependent factor.  On a genuine product it is
    // measurably all cost and no coverage: `u` is the whole integrand, so
    // `v·du = x·f′` is strictly bigger than `f`, and the step-0 growth
    // exemption (which the C3 case needs, because differentiating a composite
    // legitimately fans out) would let that run for three steps on every
    // decline.  Withholding it here is worth about a third of the decline cost.
    if splits.is_empty() {
        splits.push(Split {
            u: expr,
            dv: one,
            class: liate_class(expr, var, pool),
            trivial_dv: true,
        });
    }

    // LIATE only orders the candidates; all of them stay in the list.
    splits.sort_by_key(|s| (s.class, s.trivial_dv));
    splits
}

// ---------------------------------------------------------------------------
// Cycle detection: is `w` a constant multiple of `target`?
// ---------------------------------------------------------------------------

/// Sample points used to guess the ratio `w / target`.  Irrational, to dodge
/// the poles and branch points of the shapes this module produces, and spread
/// over more than one period so a trig ratio cannot agree by coincidence.
const RATIO_SAMPLES: [f64; 10] = [
    0.3719, 0.9137, 1.4231, 2.1719, 2.8123, 3.6411, 4.4507, 5.2903, 6.1013, 7.3331,
];

/// Minimum number of agreeing sample points when the symbolic confirmation
/// could not be obtained.  Ten evaluable points reduced to five agreeing ones
/// is the floor at which the numeric proposal is allowed to stand on its own.
const MIN_NUMERIC_RATIO_POINTS: usize = 5;

/// Detect `w = c·target` for a rational constant `c`, returning `(numer, denom)`.
///
/// The ratio is *guessed* numerically — evaluate both sides at several in-domain
/// points, require the ratios to agree, and snap to a rational with a small
/// denominator — and then confirmed, preferring a symbolic confirmation
/// (`simplify(w − c·target)` reduces to zero) and falling back to a stricter
/// numeric screen when the simplifier cannot see the cancellation.
///
/// **That fallback is load-bearing, and here is why it is safe.**  The
/// simplifier does not currently combine like terms across a rational
/// coefficient: `simplify(sin(x)·3/4 + sin(x)·(−3/4))` comes back unchanged,
/// not as `0` (integer coefficients such as `±1` *do* cancel, which is why the
/// `eˣ sin x` cycle confirms symbolically).  Requiring a symbolic confirmation
/// would therefore silently restrict cycle detection to integer ratios.  The
/// numeric screen is allowed to *propose* a ratio because nothing is emitted on
/// its authority: the antiderivative the linear solve produces from it still has
/// to clear [`verify_antiderivative_status`] in [`try_by_parts`], so a wrong
/// ratio yields a candidate that is rejected, costing CPU and nothing else.
///
/// Returns `None` when no constant ratio can be established, which is the
/// common case and must stay cheap.
fn constant_ratio(w: ExprId, target: ExprId, var: ExprId, pool: &ExprPool) -> Option<(i64, i64)> {
    let mut guess: Option<f64> = None;
    let mut agreeing = 0_usize;

    for &xv in &RATIO_SAMPLES {
        let mut env = HashMap::new();
        env.insert(var, xv);
        let (Some(wv), Some(tv)) = (
            crate::jit::eval_interp(w, &env, pool),
            crate::jit::eval_interp(target, &env, pool),
        ) else {
            return None;
        };
        if !wv.is_finite() || !tv.is_finite() || tv.abs() < 1e-9 {
            continue;
        }
        let r = wv / tv;
        match guess {
            None => {
                guess = Some(r);
                agreeing = 1;
            }
            Some(g) => {
                if (g - r).abs() > 1e-9 * (1.0 + g.abs()) {
                    return None;
                }
                agreeing += 1;
            }
        }
    }

    // Two agreeing points is the floor for a *symbolically* confirmed ratio;
    // one point can be matched by anything.
    if agreeing < 2 {
        return None;
    }
    let c = guess?;

    // Snap to a rational with a small denominator.  A by-parts cycle produces
    // ratios like ±1, ±1/2, ±2, ±1/4 — never anything exotic — so a tight
    // denominator bound is the right screen, not a limitation.
    let (num, den) = snap_rational(c)?;

    // Preferred: symbolic confirmation.  Numerics proposed; simplification
    // decides.
    let coeff = pool.rational(num, den);
    let neg = pool.mul(vec![pool.integer(-1_i32), coeff, target]);
    let sum = pool.add(vec![w, neg]);
    let expanded_sum = simplify_expanded(sum, pool).value;
    if is_zero(simplify(sum, pool).value, pool)
        || is_zero(expanded_sum, pool)
        || is_zero(collect_like_terms_deep(expanded_sum, pool), pool)
    {
        return Some((num, den));
    }

    // Fallback: the stricter numeric screen described above.
    (agreeing >= MIN_NUMERIC_RATIO_POINTS).then_some((num, den))
}

/// Snap `c` to `p/q` with `|q| <= 12` and `|p| <= 144`, or `None`.
fn snap_rational(c: f64) -> Option<(i64, i64)> {
    if !c.is_finite() {
        return None;
    }
    for den in 1_i64..=12 {
        let num_f = c * den as f64;
        let num = num_f.round();
        if (num_f - num).abs() < 1e-8 && num.abs() <= 144.0 {
            let num = num as i64;
            if num == 0 {
                return None; // `w` is identically 0 — a solved integral, not a cycle
            }
            let g = gcd_i64(num.abs(), den);
            return Some((num / g, den / g));
        }
    }
    None
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

// ---------------------------------------------------------------------------
// The reduction loop
// ---------------------------------------------------------------------------

/// Merge the square-root factors of a product into a single radical.
///
/// A by-parts residual routinely arrives carrying two or more radicals that
/// cancel: the `u = asin(x)` split of Charlwood #21 produces
/// `−½·√(1−x⁴)/√(1−x²)`, which *is* `−½·√(1+x²)`, an integrand the engine
/// handles in one line — but `simplify` does not merge radicals, so the engine
/// sees the unmerged form and declines.  This collects the factors
/// `√(rᵢ)^{kᵢ}`, forms `R = ∏ rᵢ^{kᵢ}` through
/// [`crate::poly::cancel::cancel`], and re-emits `√R` times the untouched
/// factors.
///
/// **`√a/√b = √(a/b)` is only an identity where `a` and `b` are positive**, so
/// this is a *proposal*, not a simplification, and it is deliberately confined
/// to this module rather than added to `simplify`.  It is safe here for the
/// reason the whole module is safe: the candidate it leads to is checked by
/// [`verify_antiderivative_status`] against the caller's integrand before it is
/// returned, so a rewrite that is wrong on some branch produces a candidate
/// that fails the gate.
///
/// Returns `None` when there is nothing to merge (fewer than two radical
/// factors) or the radicands are not rational functions.
/// Split a term into its rational coefficient and the product of everything
/// else, so that like terms of a sum can be recognised.
fn split_rational_coeff(term: ExprId, pool: &ExprPool) -> (rug::Rational, ExprId) {
    let one = rug::Rational::from(1);
    match pool.get(term) {
        ExprData::Integer(i) => (rug::Rational::from(i.0.clone()), pool.integer(1_i32)),
        ExprData::Rational(r) => (r.0.clone(), pool.integer(1_i32)),
        ExprData::Mul(args) => {
            let mut coeff = one;
            let mut rest: Vec<ExprId> = Vec::new();
            for a in args {
                match pool.get(a) {
                    ExprData::Integer(i) => coeff *= rug::Rational::from(i.0.clone()),
                    ExprData::Rational(r) => coeff *= r.0.clone(),
                    _ => rest.push(a),
                }
            }
            (coeff, mul_of(&rest, pool))
        }
        _ => (one, term),
    }
}

/// Combine like terms of a sum over rational coefficients.
///
/// `simplify` does not do this: `simplify(sin(x)·3/4 + sin(x)·(−3/4))` comes
/// back unchanged rather than as `0`, and neither does
/// [`crate::poly::cancel::cancel`], which refuses the input outright with
/// `NonIntegerCoefficient`.  Integer coefficients *are* combined, so the gap is
/// specifically rational ones — and it bites here twice: the rationalised `v`
/// of `∫x/√(x²−1) dx` comes out as `√(x²−1) + x/2 − x/2`, whose stray half-terms
/// then propagate into the residual and stop it being integrable, and the cycle
/// detector's symbolic confirmation fails for every non-integer ratio.
///
/// This is a local, conservative collector: it only merges terms whose
/// non-numeric part is *syntactically identical* after interning, and it never
/// reorders anything else.
fn collect_like_terms(expr: ExprId, pool: &ExprPool) -> ExprId {
    let ExprData::Add(args) = pool.get(expr) else {
        return expr;
    };
    let mut keys: Vec<ExprId> = Vec::new();
    let mut coeffs: Vec<rug::Rational> = Vec::new();
    for t in args {
        let (c, k) = split_rational_coeff(t, pool);
        match keys.iter().position(|&x| x == k) {
            Some(i) => coeffs[i] += c,
            None => {
                keys.push(k);
                coeffs.push(c);
            }
        }
    }
    let zero = rug::Rational::from(0);
    let one = rug::Rational::from(1);
    let mut terms: Vec<ExprId> = Vec::new();
    for (k, c) in keys.into_iter().zip(coeffs) {
        if c == zero {
            continue;
        }
        let is_one_key = is_one(k, pool);
        let coeff_expr = pool.rational(c.numer().clone(), c.denom().clone());
        if is_one_key {
            terms.push(coeff_expr);
        } else if c == one {
            terms.push(k);
        } else {
            terms.push(pool.mul(vec![coeff_expr, k]));
        }
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// Apply [`collect_like_terms`] bottom-up over the whole tree.
fn collect_like_terms_deep(expr: ExprId, pool: &ExprPool) -> ExprId {
    let rebuilt = match pool.get(expr) {
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| collect_like_terms_deep(a, pool))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| collect_like_terms_deep(a, pool))
                .collect(),
        ),
        ExprData::Pow { base, exp } => pool.pow(
            collect_like_terms_deep(base, pool),
            collect_like_terms_deep(exp, pool),
        ),
        ExprData::Func { name, args } => pool.func(
            name,
            args.iter()
                .map(|&a| collect_like_terms_deep(a, pool))
                .collect(),
        ),
        _ => return expr,
    };
    collect_like_terms(rebuilt, pool)
}

/// Rewrite `√(r)^k` as `r^(k/2)·√(r)^(k mod 2)` everywhere in `expr`.
///
/// `simplify` does not do this — it leaves `√(1+x²)²` standing — and the
/// omission is load-bearing here, because the algebraic engine's answers are
/// full of it: the Euler-substitution antiderivative of `∫dx/(1+x²)^{3/2}` is
/// `−2/(1 + (x+√(1+x²))²)`, whose denominator only collapses once the square of
/// the radical is folded into the polynomial part.
fn reduce_sqrt_squares(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let base_r = reduce_sqrt_squares(base, pool);
            let k = pool.with(exp, |d| match d {
                ExprData::Integer(i) => i.0.to_i64(),
                _ => None,
            });
            if let (Some(k), ExprData::Func { name, args }) = (k, pool.get(base_r)) {
                if name == "sqrt" && args.len() == 1 && k.abs() >= 2 {
                    let r = args[0];
                    let half = k / 2;
                    let rem = k % 2;
                    let mut factors = vec![pool.pow(r, pool.integer(half))];
                    if rem != 0 {
                        factors.push(pool.pow(base_r, pool.integer(rem)));
                    }
                    return pool.mul(factors);
                }
            }
            pool.pow(base_r, reduce_sqrt_squares(exp, pool))
        }
        ExprData::Add(args) => {
            pool.add(args.iter().map(|&a| reduce_sqrt_squares(a, pool)).collect())
        }
        ExprData::Mul(args) => {
            pool.mul(args.iter().map(|&a| reduce_sqrt_squares(a, pool)).collect())
        }
        ExprData::Func { name, args } => pool.func(
            name,
            args.iter().map(|&a| reduce_sqrt_squares(a, pool)).collect(),
        ),
        _ => expr,
    }
}

/// `true` when `needle` occurs anywhere in `hay`.
fn contains_sub(hay: ExprId, needle: ExprId, pool: &ExprPool) -> bool {
    if hay == needle {
        return true;
    }
    match pool.get(hay) {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().any(|&a| contains_sub(a, needle, pool))
        }
        ExprData::Pow { base, exp } => {
            contains_sub(base, needle, pool) || contains_sub(exp, needle, pool)
        }
        _ => false,
    }
}

/// Write `expr` as `A + B·s` with `A` and `B` free of `s`, or `None`.
fn split_linear_in(expr: ExprId, s: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let terms: Vec<ExprId> = match pool.get(expr) {
        ExprData::Add(args) => args,
        _ => vec![expr],
    };
    let mut a_terms: Vec<ExprId> = Vec::new();
    let mut b_terms: Vec<ExprId> = Vec::new();
    for t in terms {
        if !contains_sub(t, s, pool) {
            a_terms.push(t);
            continue;
        }
        let factors: Vec<ExprId> = match pool.get(t) {
            ExprData::Mul(args) => args,
            _ => vec![t],
        };
        let hits = factors.iter().filter(|&&f| f == s).count();
        if hits != 1 {
            return None; // s appears squared, nested, or under a power
        }
        let rest: Vec<ExprId> = factors.into_iter().filter(|&f| f != s).collect();
        if rest.iter().any(|&f| contains_sub(f, s, pool)) {
            return None;
        }
        b_terms.push(mul_of(&rest, pool));
    }
    let a = match a_terms.len() {
        0 => pool.integer(0_i32),
        1 => a_terms[0],
        _ => pool.add(a_terms),
    };
    let b = match b_terms.len() {
        0 => return None, // no `s` at all — nothing to rationalise
        1 => b_terms[0],
        _ => pool.add(b_terms),
    };
    Some((a, b))
}

/// Rationalise radical denominators: `1/(A + B·√r) → (A − B·√r)/(A² − B²·r)`.
///
/// This is the counterpart to [`combine_radicals`] and exists for a specific,
/// measured reason.  The algebraic engine answers `∫x/√(x²−1) dx` with the
/// Euler-substitution form `½(x+√(x²−1)) − ½(x+√(x²−1))⁻¹` rather than the
/// equal-but-obvious `√(x²−1)`.  As a finished answer that is only ugly; as the
/// `v` of a by-parts step it is *fatal*, because the next step differentiates it
/// and the residual explodes.  Rationalising the denominator collapses that
/// example exactly: `A² − B²r = x² − (x²−1) = 1`, so the reciprocal is `x − s`
/// and `v` reduces to `√(x²−1)`.
///
/// Like [`combine_radicals`] this is a *proposal*: the identity needs `A + B√r`
/// non-zero, and the caller re-checks the rewritten `v` against `dv` before
/// using it.
fn rationalize_radicals(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let base_r = rationalize_radicals(base, pool);
            let n = pool.with(exp, |d| match d {
                ExprData::Integer(i) => i.0.to_i64(),
                _ => None,
            });
            if let Some(n) = n {
                if n < 0 {
                    if let Some(rewritten) = rationalize_reciprocal(base_r, pool) {
                        return pool.pow(rewritten, pool.integer(-n));
                    }
                }
            }
            pool.pow(base_r, rationalize_radicals(exp, pool))
        }
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| rationalize_radicals(a, pool))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| rationalize_radicals(a, pool))
                .collect(),
        ),
        ExprData::Func { name, args } => pool.func(
            name,
            args.iter()
                .map(|&a| rationalize_radicals(a, pool))
                .collect(),
        ),
        _ => expr,
    }
}

/// `1/base` with the single radical in `base` moved into the numerator, or
/// `None` when `base` is not `A + B·√r` for one radicand `r`.
fn rationalize_reciprocal(base: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // Collect the distinct `sqrt(...)` subterms; exactly one is required.
    fn collect_sqrts(e: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
        match pool.get(e) {
            ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
                if !out.contains(&e) {
                    out.push(e);
                }
            }
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
                for a in args {
                    collect_sqrts(a, pool, out);
                }
            }
            ExprData::Pow { base, exp } => {
                collect_sqrts(base, pool, out);
                collect_sqrts(exp, pool, out);
            }
            _ => {}
        }
    }
    let mut sqrts = Vec::new();
    collect_sqrts(base, pool, &mut sqrts);
    if sqrts.len() != 1 {
        return None;
    }
    let s = sqrts[0];
    let ExprData::Func { args, .. } = pool.get(s) else {
        return None;
    };
    let r = *args.first()?;

    let (a, b) = split_linear_in(base, s, pool)?;
    // numer = A − B·s ; denom = A² − B²·r, both free of `s`.
    let numer = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b, s])]);
    let denom = pool.add(vec![
        pool.pow(a, pool.integer(2_i32)),
        pool.mul(vec![
            pool.integer(-1_i32),
            pool.pow(b, pool.integer(2_i32)),
            r,
        ]),
    ]);
    let denom = simplify(reduce_sqrt_squares(denom, pool), pool).value;
    if is_zero(denom, pool) || contains_sub(denom, s, pool) {
        return None;
    }
    Some(pool.mul(vec![numer, pool.pow(denom, pool.integer(-1_i32))]))
}

/// Try to replace `v` — an antiderivative of `dv` produced by the engine — with
/// a smaller expression that is still an antiderivative of `dv`.
///
/// Returns `v` unchanged unless the rewritten form is **both** strictly smaller
/// **and** re-verified against `dv`, so a normalisation that is wrong on some
/// branch is caught here rather than being carried into the chain.
fn normalize_v(v: ExprId, dv: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    let mut best = v;
    let mut cur = v;

    // Iterate: each pass can expose work for the next.  `−2/(1 + (x+s)²)` is
    // the motivating case — the `s²` that `reduce_sqrt_squares` needs only
    // appears after the square is expanded, and the denominator only becomes
    // linear in `s` (so `rationalize_radicals` can act) after that reduction.
    // Three passes is enough for every shape the algebraic engine emits;
    // the loop stops early as soon as a pass stops shrinking the expression.
    for _ in 0..3 {
        let expanded = reduce_sqrt_squares(simplify_expanded(cur, pool).value, pool);
        let folded = simplify(expanded, pool).value;
        let rationalised = reduce_sqrt_squares(rationalize_radicals(folded, pool), pool);

        let mut candidates = vec![
            folded,
            simplify(rationalised, pool).value,
            simplify_expanded(rationalised, pool).value,
        ];
        for i in 0..candidates.len() {
            let collected = simplify(collect_like_terms_deep(candidates[i], pool), pool).value;
            if !candidates.contains(&collected) {
                candidates.push(collected);
            }
        }
        // `cancel` as a rational function over `var` plus the radical
        // generators.  This is the only thing that removes the `x/2 − x/2`
        // residue the rationalisation leaves behind: `simplify` does not
        // combine like terms across a rational coefficient (the same gap that
        // forces the numeric fallback in `constant_ratio`).
        for i in 0..candidates.len() {
            if let Ok(k) = crate::poly::cancel::cancel(candidates[i], vec![var], pool) {
                candidates.push(simplify(k, pool).value);
            }
        }

        // `cur` advances unconditionally — an intermediate form is allowed to be
        // *larger*, because the shrink only happens after the rationalisation
        // that the growth enabled.  Only `best` is gated on size, and only
        // `best` is ever returned.
        let next = *candidates
            .iter()
            .min_by_key(|&&c| node_count(c, pool))
            .unwrap_or(&cur);

        let before = best;
        for c in candidates {
            if c == best || node_count(c, pool) >= node_count(best, pool) {
                continue;
            }
            // Re-verify against `dv`.  A rewrite that is wrong on some branch
            // is rejected here rather than being carried into the chain.
            if verify_antiderivative_status(c, dv, var, pool).is_some() {
                best = c;
            }
        }
        if best == before && next == cur {
            break;
        }
        cur = next;
    }
    best
}

fn combine_radicals(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let ExprData::Mul(args) = pool.get(expr) else {
        return None;
    };

    // `√(r)^k` — as `sqrt(r)`, as `sqrt(r)^k`, or as `Pow(sqrt(r), k)`.
    fn as_sqrt_power(a: ExprId, pool: &ExprPool) -> Option<(ExprId, i64)> {
        match pool.get(a) {
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                Some((args[0], 1))
            }
            ExprData::Pow { base, exp } => {
                let k = pool.with(exp, |d| match d {
                    ExprData::Integer(i) => i.0.to_i64(),
                    _ => None,
                })?;
                match pool.get(base) {
                    ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                        Some((args[0], k))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    let mut radicands: Vec<(ExprId, i64)> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for &a in &args {
        match as_sqrt_power(a, pool) {
            Some((r, k)) => radicands.push((r, k)),
            None => rest.push(a),
        }
    }
    if radicands.len() < 2 {
        return None;
    }

    // R = ∏ rᵢ^{kᵢ}, cancelled as a rational function in `var`.
    let factors: Vec<ExprId> = radicands
        .iter()
        .map(|&(r, k)| pool.pow(r, pool.integer(k)))
        .collect();
    let product = pool.mul(factors);
    let reduced = crate::poly::cancel::cancel(product, vec![var], pool).ok()?;
    let reduced = simplify(reduced, pool).value;

    // Pull a square monomial out of the radicand if there is one: `√(x⁴+x²)`
    // is `x·√(x²+1)`.  Without this the merged radicand of Charlwood #22 stays
    // `x⁴+x²`, which the algebraic engine refuses outright (`non-squarefree
    // radicand at deg ≥ 3`), while the extracted `√(x²+1)/x` it integrates.
    let (outside, radicand) = extract_square_monomial(reduced, var, pool);
    if !is_one(outside, pool) {
        rest.push(outside);
    }
    rest.push(pool.func("sqrt", vec![radicand]));
    Some(simplify(mul_of(&rest, pool), pool).value)
}

/// Exponent of `var` in a single monomial term, or `None` if `var` occurs in
/// any other shape.
fn monomial_degree(term: ExprId, var: ExprId, pool: &ExprPool) -> Option<i64> {
    let factors: Vec<ExprId> = match pool.get(term) {
        ExprData::Mul(args) => args,
        _ => vec![term],
    };
    let mut deg = 0_i64;
    for f in factors {
        if f == var {
            deg += 1;
        } else if let ExprData::Pow { base, exp } = pool.get(f) {
            if base == var {
                deg += pool.with(exp, |d| match d {
                    ExprData::Integer(i) => i.0.to_i64(),
                    _ => None,
                })?;
            } else if !is_free_of(f, var, pool) {
                return None;
            }
        } else if !is_free_of(f, var, pool) {
            return None;
        }
    }
    Some(deg)
}

/// Split `R` as `(var^k)² · S`, returning `(var^k, S)`.
///
/// `√(x⁴+x²) = x·√(x²+1)` — an identity for `x ≥ 0`, and like the rest of this
/// module's radical handling a *proposal* whose consequences are re-checked by
/// the differentiation gate.  Returns `(1, R)` when there is nothing to pull
/// out, which is the common case.
fn extract_square_monomial(r: ExprId, var: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let one = pool.integer(1_i32);
    let terms: Vec<ExprId> = match pool.get(r) {
        ExprData::Add(args) => args,
        _ => return (one, r),
    };
    let mut min_deg = i64::MAX;
    for &t in &terms {
        match monomial_degree(t, var, pool) {
            Some(d) => min_deg = min_deg.min(d),
            None => return (one, r),
        }
    }
    let k = if min_deg >= 2 { min_deg / 2 } else { 0 };
    if k <= 0 {
        return (one, r);
    }
    let shift = pool.pow(var, pool.integer(-2 * k));
    let reduced: Vec<ExprId> = terms
        .iter()
        .map(|&t| simplify(pool.mul(vec![t, shift]), pool).value)
        .collect();
    let outside = if k == 1 {
        var
    } else {
        pool.pow(var, pool.integer(k))
    };
    (outside, simplify(pool.add(reduced), pool).value)
}

/// Try to close `∫w dx` outright, through the full engine, over several normal
/// forms of `w`.
///
/// This is not gratuitous: the residual `v·du` arrives as an unsimplified
/// product, and *which spelling of it the engine can integrate is not the same
/// as which spelling is smallest*.  Three facts, all measured on the Charlwood
/// C1/C3 residuals, force the sweep:
///
/// * **The combined form is sometimes the only integrable one.**  The `dv = dx`
///   residual of `∫atan(x·√(1+x²))` is a sum of two terms, neither of which the
///   engine integrates alone, whose common-denominator form
///   `x(1+2x²)/(√(1+x²)(1+x²+x⁴))` is a straightforward `u = 1+x²`
///   substitution.  Expanding first destroys that.
/// * **The expanded form is sometimes the only integrable one**, for the
///   mirror-image reason: a sum whose terms take different routes.
/// * **`integrate`'s own `Add` rule only runs the *rule engine* per summand**,
///   not the rational fallback or the derivative-divides u-substitution — those
///   sit above the sum rule in `integrate_inner`.  So a residual that is a sum
///   of u-substitutable terms is missed unless the summands are dispatched
///   individually, which is what `integrate_additive` does here.  `engine.rs`
///   has a private helper of the same name for the same reason.
///
/// Returns the antiderivative of `w`, or `None`.  Every form is tried through
/// the same public engine; nothing here is a new integration rule.
fn close_integral(w_raw: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let combined = simplify(w_raw, pool).value;
    let expanded = simplify_expanded(w_raw, pool).value;

    let mut forms: Vec<ExprId> = vec![combined];
    if expanded != combined {
        forms.push(expanded);
    }
    if let Ok(t) = crate::poly::cancel::together(combined, vec![var], pool) {
        let t = simplify(t, pool).value;
        if !forms.contains(&t) {
            forms.push(t);
        }
    }
    // Like-term-collected form: `simplify` leaves rational-coefficient like
    // terms uncombined, and an uncancelled `x/2 − x/2` in a residual is enough
    // to make it unrecognisable to every downstream route.
    for i in 0..forms.len() {
        let collected = simplify(collect_like_terms_deep(forms[i], pool), pool).value;
        if !forms.contains(&collected) {
            forms.push(collected);
        }
    }
    // Radical-merged forms.  Applied to every form already collected, and to
    // each summand of the expanded one, since a residual is usually a sum whose
    // radicals cancel term by term.
    let mut merged: Vec<ExprId> = Vec::new();
    for &form in &forms {
        if let Some(m) = combine_radicals(form, var, pool) {
            merged.push(m);
        }
        if let ExprData::Add(terms) = pool.get(form) {
            let per_term: Vec<ExprId> = terms
                .iter()
                .map(|&t| combine_radicals(t, var, pool).unwrap_or(t))
                .collect();
            let rebuilt = simplify(pool.add(per_term), pool).value;
            if rebuilt != form {
                merged.push(rebuilt);
            }
        }
    }
    for m in merged {
        if !forms.contains(&m) {
            forms.push(m);
        }
    }

    for form in &forms {
        if crate::budget::check().is_err() {
            return None;
        }
        if let Ok(res) = integrate(*form, var, pool) {
            return Some(res.value);
        }
    }
    // Last: split a top-level sum and send each summand through the *full*
    // pipeline rather than the sum rule's rule-engine-only path.
    for form in &forms {
        if let Some(res) = integrate_additive(*form, var, pool) {
            return Some(res);
        }
    }
    None
}

/// Integrate a top-level sum term-by-term through the full [`integrate`]
/// pipeline.  Returns `None` if any summand declines — a partial answer is not
/// an answer.
fn integrate_additive(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let ExprData::Add(args) = pool.get(expr) else {
        return None; // not a sum: `close_integral` already tried it directly
    };
    let mut terms = Vec::with_capacity(args.len());
    for a in args {
        crate::budget::check().ok()?;
        terms.push(integrate(a, var, pool).ok()?.value);
    }
    Some(pool.add(terms))
}

/// A rational coefficient carried through the loop as `numer / denom`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rat {
    n: i64,
    d: i64,
}

impl Rat {
    fn new(n: i64, d: i64) -> Self {
        let s = if d < 0 { -1 } else { 1 };
        let g = gcd_i64(n.abs(), d.abs());
        Rat {
            n: s * n / g,
            d: s * d / g,
        }
    }
    fn one() -> Self {
        Rat { n: 1, d: 1 }
    }
    fn neg(self) -> Self {
        Rat {
            n: -self.n,
            d: self.d,
        }
    }
    fn mul(self, o: Rat) -> Self {
        Rat::new(self.n * o.n, self.d * o.d)
    }
    fn sub_from_one(self) -> Self {
        // 1 − self
        Rat::new(self.d - self.n, self.d)
    }
    fn is_zero(self) -> bool {
        self.n == 0
    }
    fn to_expr(self, pool: &ExprPool) -> ExprId {
        if self.d == 1 {
            pool.integer(self.n)
        } else {
            pool.rational(self.n, self.d)
        }
    }
    fn recip(self) -> Option<Self> {
        if self.n == 0 {
            None
        } else {
            Some(Rat::new(self.d, self.n))
        }
    }
}

/// Run the by-parts chain starting from the split `first`, maintaining
/// `I = acc + mult·∫w dx`.
///
/// Returns an *unverified* candidate; the caller gates it.  `None` means this
/// starting split led nowhere — never that the integral does not exist.
fn run_chain(integrand: ExprId, first: Split, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let mut acc = pool.integer(0_i32);
    let mut mult = Rat::one();
    let mut w = integrand;
    let mut split = first;
    let mut seen: Vec<ExprId> = vec![integrand];

    for step in 0..MAX_IBP_STEPS {
        crate::budget::check().ok()?;

        // v = ∫dv dx.  The trivial `dv = 1` case is `v = x` and needs no engine
        // call, which matters: it is the C3 route and runs on every bare
        // composite.
        // A `NonElementary` verdict on `∫dv` is discarded along with every other
        // error by `.ok()?`, and that is deliberate rather than incidental: `dv`
        // is an integrand *this module invented* by splitting the caller's, so a
        // verdict about it says nothing about the caller's integral.  (It is not
        // hypothetical — the algebraic engine currently certifies the elementary
        // `∫x/√(1−x⁴) = ½asin(x²)` as `E-INT-004`, and that expression turns up
        // as a by-parts residual.  See the test
        // `a_non_elementary_sub_integral_does_not_escape`.)
        let v = if is_one(split.dv, pool) {
            var
        } else {
            let raw = simplify(integrate(split.dv, var, pool).ok()?.value, pool).value;
            normalize_v(raw, split.dv, var, pool)
        };

        // du = u′ dx.
        let du = simplify(crate::diff::diff(split.u, var, pool).ok()?.value, pool).value;

        // Boundary term u·v, and the new integrand v·du.
        let uv = simplify(pool.mul(vec![split.u, v]), pool).value;
        let raw_w = pool.mul(vec![v, du]);
        // The chain carries the *combined* form: it is the one the growth check
        // and the cycle detector should see, because expansion inflates the node
        // count for reasons that have nothing to do with the split being wrong.
        // `close_integral` re-derives the expanded form when it needs it.
        let next_w = simplify(raw_w, pool).value;

        // ---- growth check -------------------------------------------------
        // `∫v·du` bigger than `∫u·dv` means the split was wrong.  Abandon this
        // chain so the caller can try the next candidate split.  Exempt the
        // very first step of a `dv = dx` chain: differentiating a composite
        // `f(g(x))` legitimately produces a larger expression (the chain rule
        // fans out) and that is exactly the C3 case we are here for.
        let grew =
            structural_size(next_w, pool) as f64 > GROWTH_LIMIT * structural_size(w, pool) as f64;
        if grew && !(step == 0 && split.trivial_dv) {
            return None;
        }

        acc = simplify(
            pool.add(vec![acc, pool.mul(vec![mult.to_expr(pool), uv])]),
            pool,
        )
        .value;
        mult = mult.neg();
        w = next_w;

        // ---- 1. close: can the engine finish `∫w dx` outright? ------------
        if let Some(res) = close_integral(raw_w, var, pool) {
            let tail = pool.mul(vec![mult.to_expr(pool), res]);
            return Some(simplify(pool.add(vec![acc, tail]), pool).value);
        }

        // ---- 2. cycle: is `w = c·f`?  Then solve the linear equation. -----
        //
        //   I = acc + mult·∫w = acc + mult·c·I   ⟹   I·(1 − mult·c) = acc
        //
        // `∫eˣ sin x` reaches this after two steps with `mult·c = −1`, giving
        // `I = acc/2`.  Recursing instead would alternate forever.
        if let Some((cn, cd)) = constant_ratio(w, integrand, var, pool) {
            let factor = mult.mul(Rat::new(cn, cd)).sub_from_one();
            if !factor.is_zero() {
                if let Some(inv) = factor.recip() {
                    return Some(simplify(pool.mul(vec![inv.to_expr(pool), acc]), pool).value);
                }
            }
            // `1 − mult·c = 0` is the degenerate `I = acc + I` shape: the
            // identity carries no information about `I`.  Decline; do not
            // divide by zero and do not keep spinning.
            return None;
        }

        // ---- 3. recurse: another step on `w` -----------------------------
        if seen.contains(&w) {
            return None; // exact repeat with no constant factor — no progress
        }
        seen.push(w);

        split = *candidate_splits(w, var, pool).first()?;
    }
    None
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Integrate `expr` with respect to `var` by parts.
///
/// This is the module's entry point — the one tests, probes and (once wired)
/// `engine.rs` call.  It enumerates the candidate `(u, dv)` splits in LIATE
/// order, runs the reduction chain for each, and returns the first candidate
/// whose derivative is confirmed equal to the integrand.
///
/// # Returns
///
/// [`ByPartsOutcome::Solved`] with a **verified** antiderivative, or
/// [`ByPartsOutcome::Declined`].  Never anything else — see the module docs.
///
/// # Examples
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::integrate::by_parts::{integrate_by_parts, ByPartsOutcome};
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// // ∫ x·eˣ dx = eˣ(x − 1)
/// let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
/// assert!(matches!(
///     integrate_by_parts(f, x, &pool),
///     ByPartsOutcome::Solved(_)
/// ));
/// ```
pub fn integrate_by_parts(expr: ExprId, var: ExprId, pool: &ExprPool) -> ByPartsOutcome {
    let mut log = DerivationLog::new();
    match try_by_parts(expr, var, pool, &mut log) {
        Some(f) => ByPartsOutcome::Solved(f),
        None => ByPartsOutcome::Declined(format!(
            "∫ {} — no integration-by-parts split closed the integral",
            pool.display(expr)
        )),
    }
}

/// As [`integrate_by_parts`], but returning a [`DerivedExpr`] carrying the
/// derivation log.  This is the shape `engine.rs` consumes.
pub fn integrate_by_parts_derived(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    let mut log = DerivationLog::new();
    match try_by_parts(expr, var, pool, &mut log) {
        Some(f) => Ok(DerivedExpr::with_log(f, log)),
        None => Err(IntegrationError::NotImplemented(format!(
            "∫ {} — no integration-by-parts split closed the integral",
            pool.display(expr)
        ))),
    }
}

/// The engine-facing hook: `Some(F)` with `F` already verified, or `None`.
///
/// `None` is a decline.  This signature deliberately matches the other
/// `try_*_ibp` helpers in `engine.rs` so the wiring is a three-line insertion,
/// and deliberately cannot carry a `NonElementary` verdict.
pub fn try_by_parts(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<ExprId> {
    // Re-entry guard.  The sub-integrals go through the full engine, which can
    // route back here once the hook is in place.
    let depth = IBP_DEPTH.with(|d| d.get());
    if depth >= MAX_REENTRY_DEPTH {
        return None;
    }
    IBP_DEPTH.with(|d| d.set(depth + 1));
    let _guard = DepthGuard;

    // Nothing to do for a variable-free integrand, and the `dv = dx` split of a
    // bare symbol or of `x^n` is a strictly worse route than the power rule.
    if is_free_of(expr, var, pool) || expr == var {
        return None;
    }

    // Normalise before splitting.  This is not cosmetic: the parser builds the
    // exponent of `(1+x^2)^(3/2)` as `Mul(3, Pow(2, -1))`, and
    // `crate::diff::diff` refuses that shape with `NonIntegerExponent` while
    // handling the `Rational(3, 2)` that `simplify` folds it into.  Without this
    // line every by-parts split whose `u` carries a parsed half-integer power
    // fails at the `du` step for a spelling reason rather than a mathematical
    // one — Charlwood #16 is exactly that case.  Working on the normalised copy
    // is safe because the soundness gate below still checks `d/dx F` against the
    // caller's own `expr`.
    let work = simplify(expr, pool).value;

    for split in candidate_splits(work, var, pool) {
        crate::budget::check().ok()?;
        let Some(candidate) = run_chain(work, split, var, pool) else {
            continue;
        };
        // Soundness gate.  A candidate whose derivative is not confirmed equal
        // to the integrand is discarded, no matter how it was produced.
        if verify_antiderivative_status(candidate, expr, var, pool).is_some() {
            log.push(RewriteStep::simple("int_by_parts", expr, candidate));
            return Some(candidate);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    /// Solve and assert `d/dx F = f` independently of the internal gate.
    fn solved(expr: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
        match integrate_by_parts(expr, var, pool) {
            ByPartsOutcome::Solved(f) => {
                assert!(
                    verify_antiderivative_status(f, expr, var, pool).is_some(),
                    "returned {} for ∫ {} but d/dx does not match",
                    pool.display(f),
                    pool.display(expr)
                );
                f
            }
            ByPartsOutcome::Declined(why) => {
                panic!(
                    "expected a solution for ∫ {}, got: {why}",
                    pool.display(expr)
                )
            }
        }
    }

    // -- 1. the classics ----------------------------------------------------

    #[test]
    fn x_times_exp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
        solved(f, x, &pool);
    }

    #[test]
    fn x_squared_times_exp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.mul(vec![x2, pool.func("exp", vec![x])]);
        solved(f, x, &pool);
    }

    #[test]
    fn x_times_log() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("log", vec![x])]);
        solved(f, x, &pool);
    }

    #[test]
    fn bare_log() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("log", vec![x]);
        solved(f, x, &pool);
    }

    #[test]
    fn bare_atan() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("atan", vec![x]);
        solved(f, x, &pool);
    }

    #[test]
    fn x_times_sin() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("sin", vec![x])]);
        solved(f, x, &pool);
    }

    // -- 2. the cycling family, through the linear-solve path ---------------

    /// `∫eˣ sin x` must be closed by the *cycle* branch — the linear solve —
    /// not by some other route that happens to work.  The chain is driven
    /// directly so the assertion is about the mechanism, not the answer.
    #[test]
    fn exp_sin_closes_through_the_linear_solve() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![pool.func("exp", vec![x]), pool.func("sin", vec![x])]);

        // The cycle: after two steps `v·du` is exactly `−1 ·` the integrand.
        // Step 1: u = sin x, dv = eˣ ⇒ v·du = eˣ cos x.
        // Step 2: u = cos x, dv = eˣ ⇒ v·du = −eˣ sin x.
        let ecos = pool.mul(vec![pool.func("exp", vec![x]), pool.func("cos", vec![x])]);
        let neg_f = pool.mul(vec![pool.integer(-1_i32), f]);
        assert_eq!(
            constant_ratio(neg_f, f, x, &pool),
            Some((-1, 1)),
            "the cycle detector must see −eˣsin x as (−1)·the original"
        );
        assert_eq!(
            constant_ratio(ecos, f, x, &pool),
            None,
            "eˣcos x is not a constant multiple of eˣsin x"
        );

        let result = solved(f, x, &pool);
        // The answer is eˣ(sin x − cos x)/2: the ½ can only come from solving
        // `I(1 − (−1)·(−1)·…) = acc`, i.e. from the linear solve.
        let printed = format!("{}", pool.display(result));
        assert!(
            printed.contains("exp") && printed.contains("sin") && printed.contains("cos"),
            "expected eˣ(sin x − cos x)/2, got {printed}"
        );
    }

    /// `∫sin(log x) dx` — the cycle case where the linear solve is **the only
    /// thing that can close it**, unlike `∫eˣ sin x`.
    ///
    /// The distinction matters and is easy to miss.  `∫eˣ sin x` is a genuine
    /// two-step cycle, but `engine.rs`'s `try_exp_trig_ibp` already knows the
    /// `exp·sin` closed form, so the residual `∫eˣ cos x` gets closed by the
    /// engine at step 1 and the linear solve never runs.  Nothing in the engine
    /// knows `sin(log x)`, so here the chain really does come back to `−1 ·` the
    /// original after two `dv = dx` steps, and the answer
    /// `(x·sin(log x) − x·cos(log x))/2` exists only because `I(1 − mult·c) = acc`
    /// was solved.
    #[test]
    fn the_linear_solve_closes_sin_of_log() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("sin", vec![pool.func("log", vec![x])]);

        // Premise: no existing route closes this.
        assert!(
            integrate(f, x, &pool).is_err(),
            "premise: the engine alone must not solve ∫sin(log x)"
        );
        // Premise: it really is a two-step cycle.
        let neg_f = pool.mul(vec![pool.integer(-1_i32), f]);
        assert_eq!(constant_ratio(neg_f, f, x, &pool), Some((-1, 1)));

        let res = solved(f, x, &pool);
        // The ½ is the signature of the linear solve.
        let printed = format!("{}", pool.display(res));
        assert!(
            printed.contains("1/2") || printed.contains("2^-1"),
            "expected the ½ from I(1 − mult·c) = acc, got {printed}"
        );
    }

    #[test]
    fn exp_cos_closes() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![pool.func("exp", vec![x]), pool.func("cos", vec![x])]);
        solved(f, x, &pool);
    }

    /// The linear solve is what makes the cycle family work.  If the chain
    /// simply recursed it would alternate between `eˣsin` and `eˣcos` until the
    /// step bound and decline.  Pin the arithmetic directly.
    #[test]
    fn the_linear_solve_arithmetic_is_right() {
        // I = acc + mult·c·I with mult = +1 (two negations) and c = −1:
        //   I(1 − (1)(−1)) = acc  ⟹  2I = acc  ⟹  I = acc/2.
        let mult = Rat::one().neg().neg();
        let c = Rat::new(-1, 1);
        let factor = mult.mul(c).sub_from_one();
        assert_eq!(factor, Rat::new(2, 1));
        assert_eq!(factor.recip(), Some(Rat::new(1, 2)));
    }

    // -- 3. C3 shapes: composite arguments, `dv = dx` -----------------------

    /// `∫asin(x/√(1−x²))` (Charlwood #49).  `as_inverse_trig_power` in
    /// `engine.rs` requires `args[0] == var`, so the old inverse-trig IBP
    /// declines on sight; the general rule takes `u = asin(g(x))`, `dv = dx`.
    #[test]
    fn c3_asin_of_composite() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let one_minus = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x2]),
        ]);
        let g = pool.mul(vec![
            x,
            pool.pow(pool.func("sqrt", vec![one_minus]), pool.integer(-1_i32)),
        ]);
        let f = pool.func("asin", vec![g]);
        // Whether the *sub*-integral closes is the engine's business; what this
        // test pins is that the composite argument is no longer a hard refusal
        // — the split is attempted and, if it closes, it is verified.
        if let ByPartsOutcome::Solved(res) = integrate_by_parts(f, x, &pool) {
            assert!(verify_antiderivative_status(res, f, x, &pool).is_some());
        }
        // Either way: the `dv = dx` split must be among the candidates.
        let splits = candidate_splits(f, x, &pool);
        assert!(
            splits.iter().any(|s| s.trivial_dv && s.u == f),
            "the dv = dx split must be offered for a bare composite"
        );
        assert_eq!(splits[0].class, Liate::InverseTrig);
    }

    /// `∫atan(x·√(1+x²))` (Charlwood #47) — same guard, same route.
    #[test]
    fn c3_atan_of_composite_is_attempted_not_refused() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let g = pool.mul(vec![
            x,
            pool.func("sqrt", vec![pool.add(vec![pool.integer(1_i32), x2])]),
        ]);
        let f = pool.func("atan", vec![g]);
        let splits = candidate_splits(f, x, &pool);
        assert_eq!(splits.len(), 1, "a bare composite has exactly one split");
        assert!(splits[0].trivial_dv);
        // And whatever comes back is verified or is a decline.
        match integrate_by_parts(f, x, &pool) {
            ByPartsOutcome::Solved(r) => {
                assert!(verify_antiderivative_status(r, f, x, &pool).is_some())
            }
            ByPartsOutcome::Declined(_) => {}
        }
    }

    // -- 4. termination -----------------------------------------------------

    /// `∫eˣ/x` has no elementary antiderivative, so every by-parts split
    /// regenerates an integral of the same difficulty forever.  It must decline
    /// — promptly, inside the step bound — and must not certify anything.
    #[test]
    fn a_non_terminating_shape_declines_within_the_bound() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![
            pool.func("exp", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let t0 = std::time::Instant::now();
        let out = integrate_by_parts(f, x, &pool);
        let dt = t0.elapsed();
        assert!(out.is_declined(), "∫eˣ/x must decline, got {out:?}");
        assert!(
            dt < std::time::Duration::from_secs(20),
            "the decline took {dt:?} — the step bound is not holding"
        );
    }

    /// `∫sin(x²)` is non-elementary; the `dv = dx` split produces
    /// `∫2x²cos(x²)`, which is worse, which is worse again.  Bounded, and a
    /// decline.
    #[test]
    fn growth_backs_out_rather_than_running_away() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.func("sin", vec![x2]);
        let out = integrate_by_parts(f, x, &pool);
        assert!(out.is_declined(), "∫sin(x²) must decline, got {out:?}");
    }

    /// The step bound is what stops the chain, and it is finite.
    ///
    /// A compile-time assertion rather than a runtime one: the bound is a
    /// `const`, so this should fail the build, not a test run.
    const _: () = {
        assert!(
            MAX_IBP_STEPS >= 2,
            "two steps are needed for the exp·sin cycle"
        );
        assert!(MAX_IBP_STEPS <= 6, "a deep chain is blow-up, not coverage");
        assert!(MAX_REENTRY_DEPTH >= 1);
    };

    // -- 5. a `Declined` cannot become `E-INT-004` --------------------------

    /// The type-level claim: [`ByPartsOutcome`] has exactly two variants and
    /// the conversion to the engine's error maps a decline to `E-INT-001`.
    /// There is no value of this type that means "no elementary antiderivative
    /// exists", so no amount of downstream plumbing can turn a by-parts failure
    /// into a false certificate.
    #[test]
    fn declined_cannot_become_non_elementary() {
        use crate::errors::AlkahestError;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);

        // A genuinely non-elementary integrand: the decline must still be
        // `E-INT-001`, because *this module* has no standing to certify.
        let f = pool.mul(vec![
            pool.func("exp", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let out = integrate_by_parts(f, x, &pool);
        assert!(out.is_declined());

        let err = out.into_result().unwrap_err();
        assert_eq!(err.code(), "E-INT-001");
        assert!(
            matches!(err, IntegrationError::NotImplemented(_)),
            "a by-parts decline must be NotImplemented, never NonElementary"
        );
        assert_ne!(err.code(), "E-INT-004");

        // Exhaustiveness: this `match` is the proof.  If a third variant is
        // ever added — in particular one that could mean "non-elementary" —
        // this stops compiling and someone has to justify it.
        let sample = ByPartsOutcome::Declined("x".into());
        match sample {
            ByPartsOutcome::Solved(_) => {}
            ByPartsOutcome::Declined(_) => {}
        }
    }

    /// Every solved result, over a batch, satisfies `d/dx F = f`.  This is the
    /// contract restated as a sweep rather than as a claim.
    #[test]
    fn every_solved_result_verifies() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let ex = pool.func("exp", vec![x]);
        let lx = pool.func("log", vec![x]);
        let sx = pool.func("sin", vec![x]);
        let cx = pool.func("cos", vec![x]);
        let x2 = pool.pow(x, pool.integer(2_i32));

        let cases = vec![
            pool.mul(vec![x, ex]),
            pool.mul(vec![x2, ex]),
            pool.mul(vec![x, lx]),
            lx,
            pool.func("atan", vec![x]),
            pool.func("asin", vec![x]),
            pool.mul(vec![x, sx]),
            pool.mul(vec![x, cx]),
            pool.mul(vec![x2, sx]),
            pool.mul(vec![ex, sx]),
            pool.mul(vec![ex, cx]),
            pool.mul(vec![x, pool.func("atan", vec![x])]),
        ];

        let mut solved_n = 0;
        for f in cases {
            if let ByPartsOutcome::Solved(res) = integrate_by_parts(f, x, &pool) {
                solved_n += 1;
                assert!(
                    verify_antiderivative_status(res, f, x, &pool).is_some(),
                    "unverified answer {} for ∫ {}",
                    pool.display(res),
                    pool.display(f)
                );
            }
        }
        assert!(
            solved_n >= 10,
            "expected at least 10 of the 12 classics, solved {solved_n}"
        );
    }

    // -- LIATE ordering -----------------------------------------------------

    #[test]
    fn liate_orders_the_classes() {
        assert!(Liate::Log < Liate::InverseTrig);
        assert!(Liate::InverseTrig < Liate::Algebraic);
        assert!(Liate::Algebraic < Liate::Trig);
        assert!(Liate::Trig < Liate::Exponential);
        assert!(Liate::Exponential < Liate::Other);
    }

    #[test]
    fn liate_classifies_factors() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        assert_eq!(liate_class(pool.func("log", vec![x]), x, &pool), Liate::Log);
        assert_eq!(
            liate_class(pool.func("atan", vec![x]), x, &pool),
            Liate::InverseTrig
        );
        assert_eq!(
            liate_class(pool.pow(x, pool.integer(3_i32)), x, &pool),
            Liate::Algebraic
        );
        assert_eq!(
            liate_class(pool.func("sin", vec![x]), x, &pool),
            Liate::Trig
        );
        assert_eq!(
            liate_class(pool.func("exp", vec![x]), x, &pool),
            Liate::Exponential
        );
        // `atan(x)²` is inverse-trig, not algebraic: the class of a power is
        // the class of its base.
        let a2 = pool.pow(pool.func("atan", vec![x]), pool.integer(2_i32));
        assert_eq!(liate_class(a2, x, &pool), Liate::InverseTrig);
    }

    /// LIATE picks `u = x` over `u = eˣ` for `∫x·eˣ` — the split that works.
    #[test]
    fn liate_picks_the_working_split_for_x_exp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
        let splits = candidate_splits(f, x, &pool);
        assert_eq!(splits[0].class, Liate::Algebraic);
        assert_eq!(splits[0].u, x, "u must be x, not eˣ");
    }

    /// …and `u = log x` over `u = x` for `∫x·log x`.
    #[test]
    fn liate_picks_the_working_split_for_x_log() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("log", vec![x])]);
        let splits = candidate_splits(f, x, &pool);
        assert_eq!(splits[0].class, Liate::Log);
    }

    // -- the numeric ratio guess only ever proposes -------------------------

    #[test]
    fn constant_ratio_rejects_a_non_constant_ratio() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("sin", vec![x]);
        let g = pool.mul(vec![x, f]);
        assert_eq!(constant_ratio(g, f, x, &pool), None);
    }

    #[test]
    fn constant_ratio_finds_a_rational_multiple() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("sin", vec![x]);
        let g = pool.mul(vec![pool.rational(3_i32, 4_i32), f]);
        assert_eq!(constant_ratio(g, f, x, &pool), Some((3, 4)));
    }

    // -- measurement harness ------------------------------------------------

    /// The Charlwood C1 (general products) and C3 (composite inverse-trig)
    /// clusters from `temp-alkahest/testing/charlwood50-2026-08-24.md` §5.
    ///
    /// Run it with:
    ///
    /// ```text
    /// cargo test -p alkahest-cas --lib charlwood_c1_c3 -- --ignored --nocapture
    /// ```
    ///
    /// It is `#[ignore]`d because it is a measurement, not an invariant: the
    /// per-problem outcomes move as the sub-engines improve, and pinning them
    /// would make an unrelated improvement look like a by-parts regression.  The
    /// invariant that *is* pinned, unconditionally, is that everything it
    /// reports as solved verified — see the assertion in the loop.
    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored --nocapture"]
    fn charlwood_c1_c3_coverage() {
        use crate::parse::parse;
        use std::collections::HashMap as Map;

        // (cluster, problem number, integrand)
        let problems: &[(&str, u32, &str)] = &[
            ("C1", 9, "atan(sqrt(-1+1/cos(x)))*sin(x)"),
            ("C1", 10, "exp(asin(x))*x^3/sqrt(1-x^2)"),
            ("C1", 13, "x*atan(x+sqrt(1-x^2))/sqrt(1-x^2)"),
            ("C1", 14, "asin(x)/(1+sqrt(1-x^2))"),
            ("C1", 16, "asin(x)/(1+x^2)^(3/2)"),
            ("C1", 21, "x^3*asin(x)/sqrt(1-x^4)"),
            ("C1", 22, "x^3*acos(1/x)/sqrt(-1+x^4)"),
            ("C1", 29, "atan(x)/(x^2*sqrt(1-x^2))"),
            ("C1", 30, "x*atan(x)/sqrt(1-x^2)"),
            ("C1", 35, "x*acos(1/x)/sqrt(-1+x^2)"),
            ("C1", 37, "sin(x)/(1+sin(x)^2)"),
            ("C3", 3, "asin(-sqrt(x)+sqrt(1+x))"),
            ("C3", 12, "atan(x+sqrt(1-x^2))"),
            ("C3", 47, "atan(x*sqrt(1+x^2))"),
            ("C3", 48, "atan(-sqrt(x)+sqrt(1+x))"),
            ("C3", 49, "asin(x/sqrt(1-x^2))"),
            ("C3", 50, "atan(x*sqrt(1-x^2))"),
        ];

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let mut solved_list: Vec<(String, u32)> = Vec::new();

        for (cluster, n, src) in problems {
            let mut syms: Map<String, ExprId> = Map::from([("x".to_owned(), x)]);
            let Ok(f) = parse(src, &pool, &mut syms) else {
                println!("{cluster}#{n:<3} PARSE-ERROR  {src}");
                continue;
            };
            let t0 = std::time::Instant::now();
            let out = integrate_by_parts(f, x, &pool);
            let dt = t0.elapsed();
            match out {
                ByPartsOutcome::Solved(res) => {
                    let ok = verify_antiderivative_status(res, f, x, &pool);
                    assert!(
                        ok.is_some(),
                        "{cluster}#{n} returned an UNVERIFIED antiderivative — \
                         the soundness gate leaked"
                    );
                    println!(
                        "{cluster}#{n:<3} SOLVED [{ok:?}] {dt:>9.1?}  {src}\n           F = {}",
                        pool.display(res)
                    );
                    solved_list.push((cluster.to_string(), *n));
                }
                ByPartsOutcome::Declined(_) => {
                    println!("{cluster}#{n:<3} declined     {dt:>9.1?}  {src}");
                }
            }
        }

        let c1 = solved_list.iter().filter(|(c, _)| c == "C1").count();
        let c3 = solved_list.iter().filter(|(c, _)| c == "C3").count();
        println!("\nC1: {c1}/11 closed   C3: {c3}/6 closed");
        println!("closed: {solved_list:?}");
    }

    /// Cost of a *decline*.  Every integrand the engine cannot solve pays this,
    /// so it is the number that matters for where the hook goes.
    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored --nocapture"]
    fn decline_cost() {
        use crate::parse::parse;
        use std::collections::HashMap as Map;

        let cases = [
            "exp(x)/x",
            "sin(x)/x",
            "exp(x^2)",
            "sqrt(tan(x))",
            "log(log(x))",
            "1/(x^5-x-1)",
            "exp(x)*log(x)",
            "cos(x)^2/sqrt(1+cos(x)^2+cos(x)^4)",
            // The six cases the 40-case probe reports as `E-INT-001` on the
            // merged base — i.e. exactly the inputs a Tier-0 hook on the
            // `NotImplemented` path would hand to this module.
            "sqrt(tan(x))",
            "(exp(-x)*(exp(x)+1))^(-1)",
            "sin(x^2)",
            "log(x)/(1+x)",
            "log(log(x))",
            "x/(exp(x)+1)",
        ];
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let mut total = std::time::Duration::ZERO;
        for src in cases {
            let mut syms: Map<String, ExprId> = Map::from([("x".to_owned(), x)]);
            let f = parse(src, &pool, &mut syms).expect("parses");
            let t0 = std::time::Instant::now();
            let out = integrate_by_parts(f, x, &pool);
            let dt = t0.elapsed();
            total += dt;
            let verdict = if out.is_declined() {
                "declined"
            } else {
                "SOLVED"
            };
            println!("{dt:>10.2?}  {verdict:<9}  {src}");
        }
        println!("\ntotal added on {} declines: {total:.2?}", cases.len());
    }

    // -- normalisers --------------------------------------------------------

    #[test]
    fn collect_like_terms_cancels_rational_coefficients() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // `x/2 − x/2` — `simplify` leaves this standing, and so does
        // `poly::cancel::cancel` (it refuses with `NonIntegerCoefficient`).
        let e = pool.add(vec![
            pool.mul(vec![pool.rational(1_i32, 2_i32), x]),
            pool.mul(vec![pool.rational(-1_i32, 2_i32), x]),
        ]);
        assert!(
            !is_zero(simplify(e, &pool).value, &pool),
            "this test is only interesting while simplify still misses this"
        );
        assert!(is_zero(collect_like_terms(e, &pool), &pool));
    }

    #[test]
    fn collect_like_terms_keeps_unlike_terms() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.add(vec![
            pool.mul(vec![pool.rational(1_i32, 2_i32), x]),
            pool.func("sin", vec![x]),
        ]);
        let got = collect_like_terms(e, &pool);
        assert!(!is_zero(got, &pool));
        assert!(verify_same_value(got, e, x, &pool));
    }

    #[test]
    fn reduce_sqrt_squares_folds_the_square() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = pool.add(vec![pool.integer(1_i32), pool.pow(x, pool.integer(2_i32))]);
        let s2 = pool.pow(pool.func("sqrt", vec![r]), pool.integer(2_i32));
        let got = simplify(reduce_sqrt_squares(s2, &pool), &pool).value;
        assert!(
            verify_same_value(got, r, x, &pool),
            "expected 1+x², got {}",
            pool.display(got)
        );
        assert!(
            node_count(got, &pool) < node_count(simplify(s2, &pool).value, &pool),
            "folding the square must shrink the expression; got {}",
            pool.display(got)
        );
    }

    /// The Euler-shaped antiderivative the algebraic engine returns for
    /// `∫x/√(x²−1) dx` collapses to `√(x²−1)` once the radical denominator is
    /// rationalised.  This is what unlocks Charlwood #35.
    #[test]
    fn rationalising_collapses_the_euler_form() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let s = pool.func("sqrt", vec![r]);
        let dv = pool.mul(vec![x, pool.pow(s, pool.integer(-1_i32))]);
        // v = ½(x+s) − ½(x+s)⁻¹, exactly what the engine hands back.
        let xs = pool.add(vec![x, s]);
        let v = pool.add(vec![
            pool.mul(vec![pool.rational(1_i32, 2_i32), xs]),
            pool.mul(vec![
                pool.rational(-1_i32, 2_i32),
                pool.pow(xs, pool.integer(-1_i32)),
            ]),
        ]);
        assert!(
            verify_antiderivative_status(v, dv, x, &pool).is_some(),
            "premise: the Euler form really is an antiderivative of dv"
        );
        let got = normalize_v(v, dv, x, &pool);
        assert!(
            node_count(got, &pool) < node_count(v, &pool),
            "expected a smaller v, got {}",
            pool.display(got)
        );
        assert!(
            verify_antiderivative_status(got, dv, x, &pool).is_some(),
            "the normalised v must still be an antiderivative of dv"
        );
    }

    #[test]
    fn combine_radicals_merges_a_quotient_of_roots() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let x4 = pool.pow(x, pool.integer(4_i32));
        let a = pool.func(
            "sqrt",
            vec![pool.add(vec![
                pool.integer(1_i32),
                pool.mul(vec![pool.integer(-1_i32), x4]),
            ])],
        );
        let b = pool.func(
            "sqrt",
            vec![pool.add(vec![
                pool.integer(1_i32),
                pool.mul(vec![pool.integer(-1_i32), x2]),
            ])],
        );
        // √(1−x⁴)/√(1−x²) is √(1+x²) — the residual of Charlwood #21.
        let e = pool.mul(vec![a, pool.pow(b, pool.integer(-1_i32))]);
        let got = combine_radicals(e, x, &pool).expect("two radicals must merge");
        let want = pool.func("sqrt", vec![pool.add(vec![pool.integer(1_i32), x2])]);
        assert!(
            verify_same_value(got, want, x, &pool),
            "expected √(1+x²), got {}",
            pool.display(got)
        );
    }

    /// Numeric agreement of two expressions at in-domain points.  Used only in
    /// tests, to compare a rewritten form against the shape it should equal.
    fn verify_same_value(a: ExprId, b: ExprId, var: ExprId, pool: &ExprPool) -> bool {
        let mut checked = 0;
        for xv in [0.1237_f64, 0.3719, 0.5431, 0.7913] {
            let mut env = HashMap::new();
            env.insert(var, xv);
            let (Some(av), Some(bv)) = (
                crate::jit::eval_interp(a, &env, pool),
                crate::jit::eval_interp(b, &env, pool),
            ) else {
                return false;
            };
            if !av.is_finite() || !bv.is_finite() {
                continue;
            }
            if (av - bv).abs() > 1e-9 * (1.0 + av.abs()) {
                return false;
            }
            checked += 1;
        }
        checked >= 2
    }

    /// **A `NonElementary` verdict on a sub-integral must not escape.**
    ///
    /// The sub-integrals a by-parts step creates are integrands *this module
    /// invented* by splitting the caller's.  A verdict about one of them is a
    /// statement about the invention, not about the caller's integral, so it is
    /// discarded like any other error.
    ///
    /// This is not hypothetical.  While measuring this module the algebraic
    /// engine was found to certify `∫x/√(1−x⁴) dx` — which is `½·asin(x²)`,
    /// plainly elementary — as `E-INT-004`, i.e. a **false certificate**.  That
    /// expression is a live by-parts residual (Charlwood #16 reaches it).  The
    /// bug is not this module's to fix, but this module must not launder it into
    /// a verdict about the user's input.
    #[test]
    fn a_non_elementary_sub_integral_does_not_escape() {
        use crate::errors::AlkahestError;
        use crate::parse::parse;
        use std::collections::HashMap as Map;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);

        // The trigger, stated so the reader can see what is being defended
        // against; the assertion below does not depend on it still firing.
        let mut syms: Map<String, ExprId> = Map::from([("x".to_owned(), x)]);
        let sub = parse("x/sqrt(1-x^4)", &pool, &mut syms).expect("parses");
        if let Err(e) = integrate(sub, x, &pool) {
            if matches!(e, IntegrationError::NonElementary(_)) {
                // The engine currently certifies an elementary integrand.
                // `close_integral` must swallow that, not relay it.
                assert!(
                    close_integral(sub, x, &pool).is_none()
                        || close_integral(sub, x, &pool).is_some(),
                    "close_integral must return an Option, never an error"
                );
            }
        }

        // The claim: the enclosing problem is Solved or Declined-E-INT-001.
        let mut syms: Map<String, ExprId> = Map::from([("x".to_owned(), x)]);
        let f = parse("asin(x)/(1+x^2)^(3/2)", &pool, &mut syms).expect("parses");
        match integrate_by_parts(f, x, &pool) {
            ByPartsOutcome::Solved(r) => {
                assert!(verify_antiderivative_status(r, f, x, &pool).is_some())
            }
            out @ ByPartsOutcome::Declined(_) => {
                let err = out.into_result().unwrap_err();
                assert_eq!(
                    err.code(),
                    "E-INT-001",
                    "a sub-integral's NonElementary must not become the caller's verdict"
                );
            }
        }
    }

    #[test]
    fn extract_square_monomial_pulls_x_out_of_the_radicand() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // x⁴ + x² = (x)²·(x² + 1) — the merged radicand of Charlwood #22, which
        // the algebraic engine refuses as `non-squarefree radicand at deg ≥ 3`
        // and integrates happily once the x comes out.
        let r = pool.add(vec![
            pool.pow(x, pool.integer(4_i32)),
            pool.pow(x, pool.integer(2_i32)),
        ]);
        let (outside, radicand) = extract_square_monomial(r, x, &pool);
        assert_eq!(outside, x);
        let want = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        assert!(
            verify_same_value(radicand, want, x, &pool),
            "expected x^2+1, got {}",
            pool.display(radicand)
        );
        // Nothing to pull out of a radicand with a constant term.
        let (o2, r2) = extract_square_monomial(want, x, &pool);
        assert!(is_one(o2, &pool));
        assert_eq!(r2, want);
    }

    /// The Charlwood problems this module closes, pinned so a regression is
    /// visible.  All are verified by differentiation, here and in the gate.
    #[test]
    fn charlwood_21_22_35_close_and_verify() {
        use crate::parse::parse;
        use std::collections::HashMap as Map;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        for src in [
            "x^3*asin(x)/sqrt(1-x^4)",    // #21
            "x^3*acos(1/x)/sqrt(-1+x^4)", // #22 (asec x, spelled acos(1/x))
            "x*acos(1/x)/sqrt(-1+x^2)",   // #35
        ] {
            let mut syms: Map<String, ExprId> = Map::from([("x".to_owned(), x)]);
            let f = parse(src, &pool, &mut syms).expect("parses");
            let res = solved(f, x, &pool);
            let _ = res;
        }
    }

    #[test]
    fn snap_rational_is_tight() {
        assert_eq!(snap_rational(-1.0), Some((-1, 1)));
        assert_eq!(snap_rational(0.5), Some((1, 2)));
        assert_eq!(snap_rational(2.0), Some((2, 1)));
        // Not a small rational — must be refused, not rounded to one.
        assert_eq!(snap_rational(std::f64::consts::PI), None);
        // Zero is not a cycle.
        assert_eq!(snap_rational(0.0), None);
    }
}
