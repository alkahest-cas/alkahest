//! Generator substitution `t = g(x)` for **algebraic-over-transcendental**
//! integrands — a radical whose radicand is a polynomial in a transcendental
//! generator rather than in `x`.
//!
//! # The gap
//!
//! `genus_zero::integrate_b_sqrt_high_degree` asks for the radicand as a
//! polynomial in `x` and declines with *"radicand P is not a polynomial in the
//! variable"* otherwise.  That decline is the second-largest single cause of
//! misses on Charlwood's fifty (8 of 50): `∫tan x·√(1+tan⁴x)`,
//! `∫tan x/√(1+sec³x)`, `∫sin x/√(1−sin⁶x)`, `∫sec x/√(sec⁴x−1)`, …
//!
//! [`subst`](super::subst) already closes the sub-family in which the
//! substitution makes the integrand **rational** (`∫√(tan x) dx`).  It closes
//! none of the eight, because those are the harder `R(g, √P(g))` shape with
//! `deg P` 4–6: rationalizing `√` is not on the table, and what comes out is
//! an *algebraic* function of `t`, not a rational one.
//!
//! # The reduction
//!
//! This module is the mirror image of the mixed-tower work: instead of
//! transcendental-over-algebraic it does algebraic-over-transcendental, by
//! substituting the transcendental away and handing what is left to the
//! genus-0/1 engine that already exists.
//!
//! Pick a generator `g(x)` — `tan`, `cot`, `sin`, `csc`, `cos`, `sec`, their
//! hyperbolic analogues, or `exp` — of one linear argument `θ = a·x + b`.
//! Divide the integrand by `g′` **as an expression in `x`** and ask whether
//! what remains is a function of `g` alone:
//!
//! ```text
//!     ∫f(x) dx = ∫ h(t) dt,     t = g(x),   h(g(x)) = f(x)/g′(x).
//! ```
//!
//! Dividing *first* and rewriting *second* is what makes this sound.  The
//! textbook form of the substitution needs `dx = dt/g′(t)` with `g′` rational
//! in `g`, which holds for `tan`/`cot`/`tanh`/`coth`/`exp` and **fails** for
//! `sin`/`cos`/`sec`: `cos x` is only `±√(1−sin²x)`, and picking a sign there
//! is exactly how this class of algorithm classically goes wrong.  Here no
//! sign is ever picked — `g′` is written exactly (`a·cos θ`, `a·sin θ·cos⁻²θ`,
//! …), and the rewrite succeeds only when the odd part cancels identically.
//!
//! The rewrite itself is the parity argument, in two passes.  Writing `P` for
//! the generator's *primary* function and `S` for its *secondary*
//! (`Tan`: `sin = t·cos`, secondary `cos`; `Cos`: `cos = t`, secondary `sin`;
//! `Sec`: `cos = 1/t`, secondary `sin`; …), pass 1 substitutes `P` and pass 2
//! resolves the leftover `S`, which is expressible **only in even powers**
//! (`cos²θ = 1/(1+t²)` for `Tan`, `sin²θ = 1−t²` for `Cos`, …).  An odd
//! leftover power of `S` is refused.  Both identities are exact, so the whole
//! rewrite is exact.
//!
//! # Two normalizations, and the one place a sign is lost
//!
//! What the rewrite produces is typically a radical over a *rational function*
//! of `t` — `√(−1+(1−t²)⁻²)` for `∫sec x/√(sec⁴x−1)` — which the algebraic
//! engine declines for the very reason we came here.  So the radicand is
//! normalized: with `A = N/D` in lowest terms and `N·D = E²·Q`,
//!
//! ```text
//!     √A = |E|·√Q / |D|        (Q squarefree, E the repeated part)
//! ```
//!
//! and the reduction uses `E·√Q/D`.  That drops `|·|`, and it is the **only**
//! step in this module that can be off by a sign — a real effect, not an
//! artefact: `∫sin x/√(1−sin⁶x)` genuinely has `sign(cos x)` in its
//! antiderivative, which is why Rubi writes the optimal form with a bare
//! `cos x` in the numerator of an `atanh` argument.  Rather than reason about
//! it, three candidates are proposed — `F`, `F·E√Q/(D√A)` (Rubi's shape: the
//! ratio *is* `sign(E·D)`), and `F·E·D/√((E·D)²)` — and the gate decides.
//!
//! # Soundness
//!
//! * Every emission is checked by [`crate::integrate::gate`] as
//!   `d/dx F = f` against the **original** integrand in `x`, over the original
//!   integrand's own domain — not the substituted one.  A candidate that is
//!   only right on one branch of `cos x > 0` fails on the samples where
//!   `cos x < 0` and is rejected.
//! * Before any of that, the reduction is checked numerically:
//!   `f(x) ≈ ±h(g(x))·g′(x)` at real samples.  A rewriting bug declines here
//!   instead of costing a recursive integration.
//! * The route reports **`Solved` or `Declined`, never `NonElementary`**.  A
//!   `NonElementary` verdict from the recursive call on `∫h dt` is discarded,
//!   not imported: `t = g(x)` is not invertible as an elementary map on the
//!   whole line, so a verdict does not transfer backwards, and a route that
//!   could manufacture a certificate from a failed substitution is exactly the
//!   mistake that produced eight false certificate families in this codebase.
//!
//! # Honest limitations
//!
//! * One generator argument `θ = a·x+b`, linear with a rational slope, and one
//!   radical.  `∫√(√(sec x+1) − √(sec x−1))` (Charlwood #45) has nested,
//!   distinct radicals and is refused at the first step.
//! * A bare `x` outside the generator is refused — `x = ψ(t)` is not algebraic.
//! * The reduction succeeding does not mean the result integrates: the reduced
//!   curve for `∫cos²x/√(1+cos²x+cos⁴x)` is genus 1 with an elementary
//!   answer, and the genus-1 logarithmic part declines it downstream.  That is
//!   reported as a decline of *this* route, with the inner message attached.

use std::collections::{HashMap, HashSet};

use rug::Rational;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::integrate::gate;
use crate::integrate::risch::poly_rde::{
    degree, expr_to_qpoly, is_free_of_var, poly_mul, qpoly_to_expr, trim, QPoly,
};
use crate::integrate::risch::rational_rde::{expr_to_qrational, poly_div_exact, poly_gcd};
use crate::integrate::risch::tower::literal_integer;
use crate::kernel::subs::subs;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::integral_basis::squarefree_factors;

/// Symbol standing for the substituted generator.  `$…$` fencing matches the
/// convention in `parametrize` (`$param_s$`) and `subst` (`$subst_u$`) so it
/// cannot collide with a user symbol, and reusing one name makes the recursion
/// guard a single equality test.
const T_NAME: &str = "$gensubst_t$";

/// How many distinct (generator, argument) pairs may reach the recursive
/// integration.  Each surviving pair costs a full `integrate` call; the rewrite
/// itself is cheap and rejects almost everything before this bound bites.
const MAX_ATTEMPTS: usize = 8;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Try `∫f dx = ∫h(t) dt` with `t = g(x)` for a transcendental generator `g`.
///
/// `None` means the shape is not one this route recognises, so the caller
/// keeps its own verdict.  `Some(Err(..))` means a reduction was found but
/// could not be closed — never a wrong answer, and never a non-elementarity
/// claim.
pub(super) fn try_generator_substitution(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let t = pool.symbol(T_NAME, Domain::Real);
    if var == t {
        return None; // already inside a generator substitution: do not nest
    }
    // This route exists for radicands that are *not* polynomials in `x`.  With
    // a polynomial radicand the genus-0/1 machinery owns the problem and emits
    // better forms, so leave it alone.
    if !has_transcendental_radicand(expr, var, pool) {
        return None;
    }

    // `tan`/`tanh` are expanded so the parity passes below only ever face
    // `sin`/`cos` (resp. `sinh`/`cosh`); `sec`/`csc`/`cot` never enter the pool
    // at all — the parser desugars them to `cos(x)^-1` and friends.
    let normalized = simplify(expand_tangents(expr, var, pool), pool).value;

    let mut attempts = 0usize;
    let mut last_err: Option<String> = None;
    // Best-so-far, ranked by how much of the original integrand's domain the
    // answer is *real* on.  A gate-verified answer is correct wherever it is
    // real; one that is real on the whole domain is simply a better answer, and
    // which generator produces it is not something ordering can predict.
    let mut best: Option<(usize, ExprId)> = None;
    let full = in_domain_samples(expr, var, pool).len();

    'search: for theta in collect_arguments(normalized, var, pool) {
        let Some(a) = linear_slope(theta, var, pool) else {
            continue;
        };
        for kind in kinds_for(normalized, theta, pool) {
            if attempts >= MAX_ATTEMPTS {
                break 'search;
            }
            let Some(red) = reduce(normalized, expr, var, theta, &a, kind, t, pool) else {
                continue;
            };
            attempts += 1;
            match integrate_reduced(&red, var, t, expr, pool) {
                Ok((cand, real)) => {
                    if best.map_or(true, |(seen, _)| real > seen) {
                        best = Some((real, cand));
                    }
                    if real >= full {
                        break 'search; // real everywhere: nothing left to beat
                    }
                }
                Err(msg) => last_err = Some(msg),
            }
        }
    }

    if let Some((_, antiderivative)) = best {
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple(
            "generator_substitution",
            expr,
            antiderivative,
        ));
        return Some(Ok(DerivedExpr {
            value: antiderivative,
            log,
        }));
    }

    last_err.map(|msg| {
        Err(IntegrationError::NotImplemented(format!(
            "generator substitution t = g(x) reduced the integrand, but {msg}"
        )))
    })
}

/// The gate's sample grid, restricted to where the original integrand is a
/// finite real.  This — not the substituted integrand's domain — is what the
/// answer has to be checked against.
fn in_domain_samples(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<f64> {
    super::subst::x_samples()
        .into_iter()
        .filter(|&x| gate::eval_at(expr, var, x, pool).is_some_and(f64::is_finite))
        .collect()
}

/// Integrate a reduction, back-substitute, and gate it against the original.
///
/// On success returns the antiderivative together with the number of in-domain
/// sample points at which it is a finite real — the ranking key for
/// [`try_generator_substitution`]'s search.
fn integrate_reduced(
    red: &Reduction,
    var: ExprId,
    t: ExprId,
    original: ExprId,
    pool: &ExprPool,
) -> Result<(ExprId, usize), String> {
    // A `RootSum` is unevaluable by the gate's numeric tier and opaque to
    // `simplify`, so it could never clear *this* route's gate; suppressing it
    // makes the inner engine decline early rather than build one.  The
    // `contains_root_sum` check below is what enforces that, not this guard —
    // the guard only saves the work.
    //
    // The suppression is per-frame, not per-subtree: if the reduced integral is
    // itself algebraic and reaches `parametrize`, that frame re-declares itself
    // with `RootSumExpandedByCaller`, because it turns a `RootSum` into explicit
    // real `log`/`atan` before returning and so hands back something this gate
    // can read.  Suppressing through it instead is what used to make
    // `∫√(2+2·tan x+tan²x) dx` decline on a reduced integral that solves in
    // 0.01 s at top level.
    let f_t = {
        let _guard = crate::integrate::risch::rational_integrate::RootSumSuppressed::enter();
        // Only `Ok` is consumed.  A `NonElementary` verdict on `∫h dt` is
        // deliberately *not* imported: it is a statement about `t`, and this
        // route is not allowed to turn a failed substitution into a
        // certificate about `x`.
        crate::integrate::integrate(red.integrand, t, pool)
            .map_err(|e| match e {
                // Spelled out because the inner text reads as a verdict and the
                // outer one must not: `t = g(x)` is not an elementary
                // isomorphism, so "∫h dt is not elementary" is not "∫f dx is
                // not elementary".
                IntegrationError::NonElementary(msg) => format!(
                    "the reduced integral ∫h(t) dt was reported non-elementary \
                     ({msg}); that is a statement about t and is not imported"
                ),
                other => format!("the reduced integral ∫h(t) dt declined: {other}"),
            })?
            .value
    };
    if contains_root_sum(f_t, pool) {
        return Err("the reduced integral needs an unevaluable RootSum".to_string());
    }

    let back = |e: ExprId| -> ExprId {
        let mut map = HashMap::new();
        map.insert(t, red.generator);
        simplify(subs(e, &map, pool), pool).value
    };
    let f_x = back(f_t);

    // `F`, then the two sign-repaired forms (see the module docs).  Only the
    // gate decides which — if the radicand normalization lost no sign, the
    // first clears and the others are never built into an answer.
    let mut candidates = vec![f_x];
    for fix in red.sign_fixes(pool) {
        let fixed = simplify(pool.mul(vec![f_x, back(fix)]), pool).value;
        candidates.push(fixed);
    }

    let samples = super::subst::x_samples();
    let in_domain =
        |x: f64| -> bool { gate::eval_at(original, var, x, pool).is_some_and(f64::is_finite) };
    let domain = gate::Domain::from_samples(samples.clone())
        .with_predicate(in_domain)
        .with_boxes(super::subst::domain_boxes(&samples, &|x| {
            gate::eval_at(original, var, x, pool).is_some_and(f64::is_finite)
        }));
    let target = gate::Target::symbolic(original);
    let opts = super::subst::gate_options();
    let live = in_domain_samples(original, var, pool);

    // Among the candidates the gate accepts, the one that is real over the most
    // of the original integrand's domain wins.
    //
    // The gate *skips* a sample where a side does not evaluate to a real, which
    // is how a form real on only one component still clears it (`log` written
    // without absolute values — the convention the rest of the integrator
    // uses).  Correctness is the gate's business either way; this only decides
    // which correct form to hand back, and the sign-repaired candidates are
    // exactly the ones that tend to widen it.
    let mut best: Option<(usize, ExprId)> = None;
    for candidate in candidates {
        if !gate::verify(candidate, &target, var, &domain, &opts, pool).is_verified() {
            continue;
        }
        let real = live
            .iter()
            .filter(|&&x| gate::eval_at(candidate, var, x, pool).is_some_and(f64::is_finite))
            .count();
        if best.map_or(true, |(seen, _)| real > seen) {
            best = Some((real, candidate));
        }
    }
    best.ok_or_else(|| {
        format!(
            "no back-substituted candidate for t = {} passed the gate",
            pool.display(red.generator)
        )
    })
    .map(|(real, cand)| (cand, real))
}

fn contains_root_sum(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::RootSum { .. } => true,
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_root_sum(a, pool))
        }
        ExprData::Func { args, .. } => args.iter().any(|&a| contains_root_sum(a, pool)),
        ExprData::Pow { base, exp } => {
            contains_root_sum(base, pool) || contains_root_sum(exp, pool)
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// A reduction
// ---------------------------------------------------------------------------

/// One successful reduction `∫f dx = ∫h(t) dt`.
struct Reduction {
    /// `h(t)`, ready for the recursive integrator.
    integrand: ExprId,
    /// `g(x)`, for back-substitution.
    generator: ExprId,
    /// `E·√Q/(D·√A)` in `t`, which evaluates to `sign(E·D)`; present only when
    /// the radicand normalization could have dropped a sign.
    witness: Option<ExprId>,
    /// `E·D` in `t`, the other spelling of the same sign.
    w: Option<ExprId>,
}

impl Reduction {
    /// Multiplicative repairs to try, in order of how readable the answer is.
    fn sign_fixes(&self, pool: &ExprPool) -> Vec<ExprId> {
        let mut out = Vec::new();
        if let Some(wit) = self.witness {
            out.push(wit);
        }
        if let Some(w) = self.w {
            let sq = pool.pow(w, pool.integer(2_i32));
            let root = pool.func("sqrt", vec![sq]);
            out.push(pool.mul(vec![w, pool.pow(root, pool.integer(-1_i32))]));
        }
        out
    }
}

/// Build the reduction for one (generator kind, argument) pair, or refuse.
#[allow(clippy::too_many_arguments)]
fn reduce(
    normalized: ExprId,
    original: ExprId,
    var: ExprId,
    theta: ExprId,
    a: &Rational,
    kind: Kind,
    t: ExprId,
    pool: &ExprPool,
) -> Option<Reduction> {
    let generator = kind.generator(theta, pool);
    let gp = kind.derivative(theta, a, pool);
    // Put every exponential onto this generator first — see
    // `normalize_exponential_arguments`.  A no-op for the trigonometric and
    // hyperbolic kinds and for an integrand that already has one exponential
    // argument.
    let normalized = normalize_exponential_arguments(normalized, var, theta, a, kind, pool);
    let h_x = kind.divide_by_derivative(normalized, theta, a, pool);

    let h_t = rewrite_in_t(h_x, var, theta, kind, t, pool)?;
    let norm = normalize_radicals(h_t, t, pool)?;

    let red = Reduction {
        integrand: norm.integrand,
        generator,
        witness: norm.witness,
        w: norm.w,
    };
    // Numeric guard: the reduction must reproduce the original integrand up to
    // the one sign the radicand normalization may have dropped.  A rewriting
    // bug declines here rather than costing a recursive integration.
    check_reduction(&red, original, var, t, gp, pool).then_some(red)
}

/// Put every exponential in `expr` onto the generator `exp(θ)`.
///
/// `exp(c·x + d)` with `c = k·a` for an integer `k` is exactly
/// `exp(d − k·b)·exp(θ)^k`, where `θ = a·x + b`.  Real exponentials only, so
/// there is no branch to choose and the identity is unconditional.
///
/// **This is the normalization the route cannot do without.** `∫eˣ·√(1+e⁴ˣ)dx`
/// names `exp(x)` and `exp(4x)`, which are two unrelated nodes to every matcher
/// in this module even though the second is the fourth power of the first.
/// Without it the `θ = x` attempt trips over `exp(4x)` in
/// [`replace_secondary`], the `θ = 4x` attempt trips over `exp(x)`, and an
/// integrand that reduces to `∫√(1+t⁴)dt` in one step has no single-generator
/// spelling at all.  That is the "tower normalised onto one generator" gap
/// `planning/risch.md` records against the `exp(k·x)` family.
///
/// A non-integer ratio (`exp(x)` against `exp(x·√2)`, or against `exp(x²)`) is
/// left alone: those really are independent generators, and rewriting them
/// would need a branch choice.  `k = 1` with a zero shift rewrites to the node
/// it started from, which is why the identity case is skipped rather than
/// rebuilt.
fn normalize_exponential_arguments(
    expr: ExprId,
    var: ExprId,
    theta: ExprId,
    a: &Rational,
    kind: Kind,
    pool: &ExprPool,
) -> ExprId {
    if kind != Kind::Exp {
        return expr;
    }
    let Some(tp) = expr_to_qpoly(theta, var, pool) else {
        return expr;
    };
    let b = tp.first().cloned().unwrap_or_else(|| Rational::from(0));
    rewrite_exponentials(expr, var, theta, a, &b, pool)
}

fn rewrite_exponentials(
    expr: ExprId,
    var: ExprId,
    theta: ExprId,
    a: &Rational,
    b: &Rational,
    pool: &ExprPool,
) -> ExprId {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "exp" && args.len() == 1 => {
            let u = args[0];
            if u == theta || is_free_of_var(u, var, pool) {
                return expr;
            }
            // `u = c·x + d`, and `k = c/a` must be an integer.
            let Some(up) = expr_to_qpoly(u, var, pool) else {
                return expr;
            };
            if up.len() > 2 {
                return expr;
            }
            let Some(c) = up.get(1) else {
                return expr;
            };
            let k_q = c / a.clone();
            if *k_q.denom() != 1 {
                return expr;
            }
            let Ok(k) = i32::try_from(k_q.numer().clone()) else {
                return expr;
            };
            let d = up.first().cloned().unwrap_or_else(|| Rational::from(0));
            let shift = d - Rational::from(k) * b.clone();
            let powered = pool.pow(pool.func("exp", vec![theta]), pool.integer(k));
            if shift == 0 {
                powered
            } else {
                let s = pool.rational(shift.numer().clone(), shift.denom().clone());
                pool.mul(vec![pool.func("exp", vec![s]), powered])
            }
        }
        ExprData::Func { ref name, ref args } => pool.func(
            name.clone(),
            args.iter()
                .map(|&x| rewrite_exponentials(x, var, theta, a, b, pool))
                .collect(),
        ),
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&x| rewrite_exponentials(x, var, theta, a, b, pool))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&x| rewrite_exponentials(x, var, theta, a, b, pool))
                .collect(),
        ),
        ExprData::Pow { base, exp } => pool.pow(
            rewrite_exponentials(base, var, theta, a, b, pool),
            rewrite_exponentials(exp, var, theta, a, b, pool),
        ),
        _ => expr,
    }
}

/// `f(x) ≈ ±h(g(x))·g′(x)` at real samples, with the sign consistent enough to
/// be a sign and not noise.
fn check_reduction(
    red: &Reduction,
    original: ExprId,
    var: ExprId,
    t: ExprId,
    gp: ExprId,
    pool: &ExprPool,
) -> bool {
    let mut checked = 0usize;
    for x in super::subst::x_samples() {
        let (Some(f), Some(tv), Some(d)) = (
            gate::eval_at(original, var, x, pool),
            gate::eval_at(red.generator, var, x, pool),
            gate::eval_at(gp, var, x, pool),
        ) else {
            continue;
        };
        let Some(h) = gate::eval_at(red.integrand, t, tv, pool) else {
            continue;
        };
        let rhs = h * d;
        if !f.is_finite() || !rhs.is_finite() || f.abs() < 1e-9 {
            continue;
        }
        if (f.abs() - rhs.abs()).abs() > 1e-6 * (1.0 + f.abs()) {
            return false;
        }
        checked += 1;
    }
    checked >= 6
}

// ---------------------------------------------------------------------------
// Generator kinds
// ---------------------------------------------------------------------------

/// A transcendental generator `g = h(θ)` of one linear argument `θ = a·x+b`.
///
/// Each kind names a *primary* function, which is exactly expressible in
/// `t` (possibly times the secondary), and a *secondary*, expressible only in
/// even powers.  That asymmetry is the parity condition for
/// `f/g′ ∈ ℚ(g)` written as a rewrite rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Tan,
    Cot,
    Sin,
    Csc,
    Cos,
    Sec,
    Tanh,
    Coth,
    Sinh,
    Csch,
    Cosh,
    Sech,
    Exp,
}

/// Circular kinds, most productive first.
const TRIG_KINDS: [Kind; 6] = [
    Kind::Tan,
    Kind::Cot,
    Kind::Cos,
    Kind::Sin,
    Kind::Sec,
    Kind::Csc,
];
/// Hyperbolic kinds.
const HYP_KINDS: [Kind; 6] = [
    Kind::Tanh,
    Kind::Coth,
    Kind::Cosh,
    Kind::Sinh,
    Kind::Sech,
    Kind::Csch,
];

impl Kind {
    /// `g` as an expression in `x`.
    fn generator(self, theta: ExprId, pool: &ExprPool) -> ExprId {
        let f = |n: &str| pool.func(n, vec![theta]);
        let inv = |e: ExprId| pool.pow(e, pool.integer(-1_i32));
        match self {
            Kind::Tan => f("tan"),
            Kind::Cot => inv(f("tan")),
            Kind::Sin => f("sin"),
            Kind::Csc => inv(f("sin")),
            Kind::Cos => f("cos"),
            Kind::Sec => inv(f("cos")),
            Kind::Tanh => f("tanh"),
            Kind::Coth => inv(f("tanh")),
            Kind::Sinh => f("sinh"),
            Kind::Csch => inv(f("sinh")),
            Kind::Cosh => f("cosh"),
            Kind::Sech => inv(f("cosh")),
            Kind::Exp => f("exp"),
        }
    }

    /// `g′` as an exact **factored** expression — `±a · ∏ fᵢ(θ)^{eᵢ}`, with no
    /// `±√` anywhere.
    ///
    /// Kept factored because `1/g′` has to cancel against the integrand's own
    /// factors, and `simplify` does not push a reciprocal through a product:
    /// `sin x·cos⁻¹x·(sin x·cos⁻²x)⁻¹` stays exactly as written and the rewrite
    /// then sees an odd `sin` that is not really there.  Negating the exponents
    /// one at a time in [`Kind::divide_by_derivative`] makes the cancellation a
    /// plain `simplify` job.
    fn derivative_factors(self) -> (i32, &'static [(&'static str, i32)]) {
        match self {
            Kind::Tan => (1, &[("cos", -2)]),
            Kind::Cot => (-1, &[("sin", -2)]),
            Kind::Sin => (1, &[("cos", 1)]),
            Kind::Csc => (-1, &[("cos", 1), ("sin", -2)]),
            Kind::Cos => (-1, &[("sin", 1)]),
            Kind::Sec => (1, &[("sin", 1), ("cos", -2)]),
            Kind::Tanh => (1, &[("cosh", -2)]),
            Kind::Coth => (-1, &[("sinh", -2)]),
            Kind::Sinh => (1, &[("cosh", 1)]),
            Kind::Csch => (-1, &[("cosh", 1), ("sinh", -2)]),
            Kind::Cosh => (1, &[("sinh", 1)]),
            Kind::Sech => (-1, &[("sinh", 1), ("cosh", -2)]),
            Kind::Exp => (1, &[("exp", 1)]),
        }
    }

    /// `g′` as an exact expression in `x`.
    fn derivative(self, theta: ExprId, a: &Rational, pool: &ExprPool) -> ExprId {
        self.derivative_power(theta, a, 1, pool)
    }

    /// `f/g′`, built by negating `g′`'s exponents rather than wrapping it in a
    /// `^-1` the simplifier will not distribute.
    fn divide_by_derivative(
        self,
        f: ExprId,
        theta: ExprId,
        a: &Rational,
        pool: &ExprPool,
    ) -> ExprId {
        let inv = self.derivative_power(theta, a, -1, pool);
        simplify(pool.mul(vec![f, inv]), pool).value
    }

    /// `(g′)^s` for `s ∈ {1, −1}`.
    fn derivative_power(self, theta: ExprId, a: &Rational, s: i32, pool: &ExprPool) -> ExprId {
        let (sign, factors) = self.derivative_factors();
        let coeff = if s > 0 {
            Rational::from(sign) * a.clone()
        } else {
            (Rational::from(sign) * a.clone()).recip()
        };
        let mut parts = vec![pool.rational(coeff.numer().clone(), coeff.denom().clone())];
        for &(name, e) in factors {
            let base = pool.func(name, vec![theta]);
            parts.push(pool.pow(base, pool.integer(e * s)));
        }
        pool.mul(parts)
    }

    /// The function that is expressible exactly.
    fn primary(self) -> &'static str {
        match self {
            Kind::Tan | Kind::Sin | Kind::Csc => "sin",
            Kind::Cot | Kind::Cos | Kind::Sec => "cos",
            Kind::Tanh | Kind::Sinh | Kind::Csch => "sinh",
            Kind::Coth | Kind::Cosh | Kind::Sech => "cosh",
            Kind::Exp => "exp",
        }
    }

    /// `primary(θ)` in terms of `t` — exact, and possibly still mentioning the
    /// secondary (`sin θ = tan θ · cos θ`).
    fn primary_repl(self, theta: ExprId, t: ExprId, pool: &ExprPool) -> ExprId {
        let f = |n: &str| pool.func(n, vec![theta]);
        let inv = |e: ExprId| pool.pow(e, pool.integer(-1_i32));
        match self {
            Kind::Tan => pool.mul(vec![t, f("cos")]),
            Kind::Cot => pool.mul(vec![t, f("sin")]),
            Kind::Tanh => pool.mul(vec![t, f("cosh")]),
            Kind::Coth => pool.mul(vec![t, f("sinh")]),
            Kind::Sin | Kind::Cos | Kind::Sinh | Kind::Cosh | Kind::Exp => t,
            Kind::Csc | Kind::Sec | Kind::Csch | Kind::Sech => inv(t),
        }
    }

    /// The function admitted only in even powers.
    fn secondary(self) -> Option<&'static str> {
        match self {
            Kind::Tan | Kind::Sin | Kind::Csc => Some("cos"),
            Kind::Cot | Kind::Cos | Kind::Sec => Some("sin"),
            Kind::Tanh | Kind::Sinh | Kind::Csch => Some("cosh"),
            Kind::Coth | Kind::Cosh | Kind::Sech => Some("sinh"),
            Kind::Exp => None,
        }
    }

    /// `secondary(θ)²` in terms of `t`.
    fn secondary_sq(self, t: ExprId, pool: &ExprPool) -> Option<ExprId> {
        let one = pool.integer(1_i32);
        let t2 = pool.pow(t, pool.integer(2_i32));
        let t_m2 = pool.pow(t, pool.integer(-2_i32));
        let neg = |e: ExprId| pool.mul(vec![pool.integer(-1_i32), e]);
        let inv = |e: ExprId| pool.pow(e, pool.integer(-1_i32));
        Some(match self {
            // cos²θ = 1/(1+tan²θ);  sin²θ = 1/(1+cot²θ)
            Kind::Tan | Kind::Cot => inv(pool.add(vec![one, t2])),
            // cos²θ = 1−sin²θ;  sin²θ = 1−cos²θ
            Kind::Sin | Kind::Cos => pool.add(vec![one, neg(t2)]),
            // cos²θ = 1−1/csc²θ;  sin²θ = 1−1/sec²θ
            Kind::Csc | Kind::Sec => pool.add(vec![one, neg(t_m2)]),
            // cosh²θ = 1/(1−tanh²θ)
            Kind::Tanh => inv(pool.add(vec![one, neg(t2)])),
            // sinh²θ = 1/(coth²θ−1)
            Kind::Coth => inv(pool.add(vec![t2, neg(one)])),
            // cosh²θ = 1+sinh²θ
            Kind::Sinh => pool.add(vec![one, t2]),
            // sinh²θ = cosh²θ−1
            Kind::Cosh => pool.add(vec![t2, neg(one)]),
            // cosh²θ = 1+1/csch²θ
            Kind::Csch => pool.add(vec![one, t_m2]),
            // sinh²θ = 1/sech²θ−1
            Kind::Sech => pool.add(vec![t_m2, neg(one)]),
            Kind::Exp => return None,
        })
    }
}

/// Which generators are worth trying for `θ`, given which functions of `θ` the
/// integrand actually mentions.
fn kinds_for(expr: ExprId, theta: ExprId, pool: &ExprPool) -> Vec<Kind> {
    let mut names = HashSet::new();
    collect_func_names(expr, theta, pool, &mut names);
    let mut out = Vec::new();
    if names.contains("sin") || names.contains("cos") {
        out.extend_from_slice(&TRIG_KINDS);
    }
    if names.contains("sinh") || names.contains("cosh") {
        out.extend_from_slice(&HYP_KINDS);
    }
    if names.contains("exp") {
        out.push(Kind::Exp);
    }
    out
}

fn collect_func_names(
    expr: ExprId,
    theta: ExprId,
    pool: &ExprPool,
    out: &mut HashSet<&'static str>,
) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } => {
            if args.len() == 1 && args[0] == theta {
                for known in ["sin", "cos", "sinh", "cosh", "exp"] {
                    if name == known {
                        out.insert(known);
                    }
                }
            }
            for &a in args.iter() {
                collect_func_names(a, theta, pool, out);
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                collect_func_names(a, theta, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_func_names(base, theta, pool, out);
            collect_func_names(exp, theta, pool, out);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Shape probes
// ---------------------------------------------------------------------------

/// Is there a radical whose radicand moves with `var` but is not a polynomial
/// in it?  That is exactly the condition `genus_zero` declines on.
fn has_transcendental_radicand(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if matches!(name.as_str(), "sqrt" | "cbrt") && args.len() == 1 =>
        {
            (!is_free_of_var(args[0], var, pool) && expr_to_qpoly(args[0], var, pool).is_none())
                || has_transcendental_radicand(args[0], var, pool)
        }
        ExprData::Func { ref args, .. } => args
            .iter()
            .any(|&a| has_transcendental_radicand(a, var, pool)),
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.denom() != 1
                    && !is_free_of_var(base, var, pool)
                    && expr_to_qpoly(base, var, pool).is_none()
                {
                    return true;
                }
            }
            has_transcendental_radicand(base, var, pool)
        }
        ExprData::Add(args) | ExprData::Mul(args) => args
            .iter()
            .any(|&a| has_transcendental_radicand(a, var, pool)),
        _ => false,
    }
}

/// Arguments of `sin`/`cos`/`sinh`/`cosh`/`exp` that move with `var`.
fn collect_arguments(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut seen = Vec::new();
    fn walk(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
        match pool.get(expr) {
            ExprData::Func { ref name, ref args } => {
                if args.len() == 1
                    && matches!(name.as_str(), "sin" | "cos" | "sinh" | "cosh" | "exp")
                    && !is_free_of_var(args[0], var, pool)
                    && !out.contains(&args[0])
                {
                    out.push(args[0]);
                }
                for &a in args.iter() {
                    walk(a, var, pool, out);
                }
            }
            ExprData::Add(args) | ExprData::Mul(args) => {
                for &a in &args {
                    walk(a, var, pool, out);
                }
            }
            ExprData::Pow { base, exp } => {
                walk(base, var, pool, out);
                walk(exp, var, pool, out);
            }
            _ => {}
        }
    }
    walk(expr, var, pool, &mut seen);
    seen
}

/// `dθ/dx` when `θ` is linear in `var` with a nonzero rational slope.
fn linear_slope(theta: ExprId, var: ExprId, pool: &ExprPool) -> Option<Rational> {
    let p = expr_to_qpoly(theta, var, pool)?;
    if p.len() > 2 {
        return None;
    }
    let a = p.get(1)?.clone();
    (a != 0).then_some(a)
}

/// Rewrite `tan(u) → sin(u)·cos(u)⁻¹` and `tanh(u) → sinh(u)·cosh(u)⁻¹`.
///
/// `sec`/`csc`/`cot` need no counterpart: the parser desugars them to
/// `cos(u)^-1`/`sin(u)^-1`/`tan(u)^-1` and no such node ever exists.
fn expand_tangents(expr: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let inner = expand_tangents(args[0], var, pool);
            let pair = match name.as_str() {
                "tan" => Some(("sin", "cos")),
                "tanh" => Some(("sinh", "cosh")),
                _ => None,
            };
            match pair {
                Some((num, den)) if !is_free_of_var(inner, var, pool) => {
                    let n = pool.func(num, vec![inner]);
                    let d = pool.func(den, vec![inner]);
                    pool.mul(vec![n, pool.pow(d, pool.integer(-1_i32))])
                }
                _ => pool.func(name.clone(), vec![inner]),
            }
        }
        ExprData::Func { ref name, ref args } => pool.func(
            name.clone(),
            args.iter()
                .map(|&a| expand_tangents(a, var, pool))
                .collect(),
        ),
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| expand_tangents(a, var, pool))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| expand_tangents(a, var, pool))
                .collect(),
        ),
        ExprData::Pow { base, exp } => pool.pow(expand_tangents(base, var, pool), exp),
        _ => expr,
    }
}

// ---------------------------------------------------------------------------
// The two rewrite passes
// ---------------------------------------------------------------------------

/// Write `h_x` — already divided by `g′` — as a function of `t` alone.
fn rewrite_in_t(
    h_x: ExprId,
    var: ExprId,
    theta: ExprId,
    kind: Kind,
    t: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    // Pass 1: the primary, exactly.  Simplifying afterwards is what collects
    // the leftover secondary into a single power for pass 2 to read.
    let prim = pool.func(kind.primary(), vec![theta]);
    let mut map = HashMap::new();
    map.insert(prim, kind.primary_repl(theta, t, pool));
    let stage = simplify(subs(h_x, &map, pool), pool).value;

    // Pass 2: the secondary, in even powers only.
    let out = replace_secondary(stage, var, theta, kind, t, pool)?;
    let out = simplify(out, pool).value;
    is_free_of_var(out, var, pool).then_some(out)
}

fn replace_secondary(
    expr: ExprId,
    var: ExprId,
    theta: ExprId,
    kind: Kind,
    t: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if is_free_of_var(expr, var, pool) {
        return Some(expr);
    }
    if expr == var {
        return None; // a bare `x` cannot be written in `t`
    }
    let recur = |e: ExprId| replace_secondary(e, var, theta, kind, t, pool);
    match pool.get(expr) {
        ExprData::Add(args) => {
            let mapped: Option<Vec<ExprId>> = args.iter().map(|&a| recur(a)).collect();
            Some(pool.add(mapped?))
        }
        ExprData::Mul(args) => {
            let mapped: Option<Vec<ExprId>> = args.iter().map(|&a| recur(a)).collect();
            Some(pool.mul(mapped?))
        }
        ExprData::Pow { base, exp } => {
            if !is_free_of_var(exp, var, pool) {
                return None;
            }
            if is_secondary(base, kind, theta, pool) {
                // `secondary^n` is expressible exactly when `n` is even.
                let n = literal_integer(exp, pool)?;
                if n % 2 != 0 {
                    return None;
                }
                let sq = kind.secondary_sq(t, pool)?;
                return Some(pool.pow(sq, pool.integer(i32::try_from(n / 2).ok()?)));
            }
            // `(t·cos θ)^k` that `simplify` left folded: distribute so the
            // even-power test above can see each factor's real exponent.
            if let (ExprData::Mul(args), Some(n)) = (pool.get(base), literal_integer(exp, pool)) {
                let k = i32::try_from(n).ok()?;
                let mapped: Option<Vec<ExprId>> = args
                    .iter()
                    .map(|&a| recur(pool.pow(a, pool.integer(k))))
                    .collect();
                return Some(pool.mul(mapped?));
            }
            Some(pool.pow(recur(base)?, exp))
        }
        ExprData::Func { ref name, ref args } => {
            // A bare secondary, or any other transcendental of `var`, is not a
            // function of `t`.
            if args.len() == 1 && is_secondary(expr, kind, theta, pool) {
                return None;
            }
            if !matches!(name.as_str(), "sqrt" | "cbrt" | "abs") {
                return None;
            }
            let mapped: Option<Vec<ExprId>> = args.iter().map(|&a| recur(a)).collect();
            Some(pool.func(name.clone(), mapped?))
        }
        _ => None,
    }
}

fn is_secondary(expr: ExprId, kind: Kind, theta: ExprId, pool: &ExprPool) -> bool {
    let Some(sec) = kind.secondary() else {
        return false;
    };
    matches!(pool.get(expr), ExprData::Func { ref name, ref args }
        if name == sec && args.len() == 1 && args[0] == theta)
}

// ---------------------------------------------------------------------------
// Radicand normalization
// ---------------------------------------------------------------------------

/// The reduced integrand with its radicand turned into a squarefree polynomial.
struct Normalized {
    integrand: ExprId,
    witness: Option<ExprId>,
    w: Option<ExprId>,
}

/// Turn `√(N/D)` into `E·√Q/D` with `N·D = E²·Q` and `Q` squarefree.
///
/// The algebraic engine wants a polynomial, squarefree radicand: it declines
/// `√(−1+(1−t²)⁻²)` with the same message that sent us here, and
/// `√(2t²−t⁴)` with *"non-squarefree radicand at deg ≥ 3"*.  Both are
/// exactly this shape.
///
/// Returns the integrand unchanged when there is nothing to do, and `None`
/// when the radicand cannot be read as a rational function of `t` (so the
/// caller moves on rather than guessing).
fn normalize_radicals(expr: ExprId, t: ExprId, pool: &ExprPool) -> Option<Normalized> {
    let mut radicands = Vec::new();
    collect_radicands(expr, t, pool, &mut radicands);
    let Some(&a_expr) = radicands.first() else {
        return Some(Normalized {
            integrand: expr,
            witness: None,
            w: None,
        });
    };
    if radicands.len() > 1 {
        return None; // more than one algebraic generator: not this route
    }

    let (num, den) = expr_to_qrational(a_expr, t, pool)?;
    let (num, den) = {
        let g = poly_gcd(&trim(num.clone()), &trim(den.clone()));
        if degree(&g) >= 1 {
            (poly_div_exact(&num, &g), poly_div_exact(&den, &g))
        } else {
            (trim(num), trim(den))
        }
    };
    if num.is_empty() || den.is_empty() {
        return None;
    }
    let s = trim(poly_mul(&num, &den));
    if s.is_empty() {
        return None;
    }

    // `s = ∏ⱼ fⱼ^{j+1}` ⟹ `E = ∏ⱼ fⱼ^{⌊(j+1)/2⌋}`, `Q = ∏_{j+1 odd} fⱼ`.
    let mut e_poly: QPoly = vec![Rational::from(1)];
    let mut q_poly: QPoly = vec![Rational::from(1)];
    for (j, f) in squarefree_factors(&s).iter().enumerate() {
        let mult = j + 1;
        for _ in 0..(mult / 2) {
            e_poly = poly_mul(&e_poly, f);
        }
        if mult % 2 == 1 {
            q_poly = poly_mul(&q_poly, f);
        }
    }
    // `squarefree_factors` drops the content, so restore it on `Q` (the odd
    // part) and check the identity exactly rather than trusting it.
    let built = poly_mul(&poly_mul(&e_poly, &e_poly), &q_poly);
    let (Some(lc_s), Some(lc_b)) = (trim(s.clone()).last().cloned(), trim(built).last().cloned())
    else {
        return None;
    };
    if lc_b == 0 {
        return None;
    }
    let c = lc_s / lc_b;
    for coeff in q_poly.iter_mut() {
        *coeff *= c.clone();
    }
    if trim(poly_mul(&poly_mul(&e_poly, &e_poly), &q_poly)) != trim(s) {
        return None;
    }

    let trivial = degree(&e_poly) < 1
        && e_poly.first().is_some_and(|v| *v == 1)
        && degree(&den) < 1
        && den.first().is_some_and(|v| *v == 1);
    if trivial {
        return Some(Normalized {
            integrand: expr,
            witness: None,
            w: None,
        });
    }

    let e_expr = qpoly_to_expr(&e_poly, t, pool);
    let q_expr = qpoly_to_expr(&q_poly, t, pool);
    let d_expr = qpoly_to_expr(&den, t, pool);
    let sqrt_q = pool.func("sqrt", vec![q_expr]);
    let new_sqrt = pool.mul(vec![e_expr, sqrt_q, pool.pow(d_expr, pool.integer(-1_i32))]);

    let integrand = simplify(replace_radical(expr, a_expr, new_sqrt, pool), pool).value;
    // `E·√Q / (D·√A) = sign(E)·sign(D)` — the sign the `|·|` dropped, spelled
    // with the integrand's own radical, which is how Rubi writes these.
    let sqrt_a = pool.func("sqrt", vec![a_expr]);
    let witness = simplify(
        pool.mul(vec![new_sqrt, pool.pow(sqrt_a, pool.integer(-1_i32))]),
        pool,
    )
    .value;
    let w = simplify(
        pool.mul(vec![
            qpoly_to_expr(&e_poly, t, pool),
            qpoly_to_expr(&den, t, pool),
        ]),
        pool,
    )
    .value;

    Some(Normalized {
        integrand,
        witness: Some(witness),
        w: Some(w),
    })
}

/// Distinct radicands under a square root that move with `t`.
fn collect_radicands(expr: ExprId, t: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } => {
            if name == "sqrt"
                && args.len() == 1
                && !is_free_of_var(args[0], t, pool)
                && !out.contains(&args[0])
            {
                out.push(args[0]);
            }
            for &a in args.iter() {
                collect_radicands(a, t, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.denom() == 2 && !is_free_of_var(base, t, pool) && !out.contains(&base) {
                    out.push(base);
                }
            }
            collect_radicands(base, t, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                collect_radicands(a, t, pool, out);
            }
        }
        _ => {}
    }
}

/// Replace every `√A` (however spelled) by `new_sqrt`.
fn replace_radical(expr: ExprId, a: ExprId, new_sqrt: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && args[0] == a =>
        {
            new_sqrt
        }
        ExprData::Func { ref name, ref args } => pool.func(
            name.clone(),
            args.iter()
                .map(|&x| replace_radical(x, a, new_sqrt, pool))
                .collect(),
        ),
        ExprData::Pow { base, exp } if base == a => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.denom() == 2 {
                    if let Some(m) = r.0.numer().to_i32() {
                        return pool.pow(new_sqrt, pool.integer(m));
                    }
                }
            }
            pool.pow(replace_radical(base, a, new_sqrt, pool), exp)
        }
        ExprData::Pow { base, exp } => pool.pow(replace_radical(base, a, new_sqrt, pool), exp),
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&x| replace_radical(x, a, new_sqrt, pool))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&x| replace_radical(x, a, new_sqrt, pool))
                .collect(),
        ),
        _ => expr,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        (pool, x)
    }

    /// `d/dx F = f` at every sample of `[-6, 6]` where `f` is a finite real.
    ///
    /// Deliberately not a string comparison against Rubi's optimal form: the
    /// answers here legitimately differ from it (`∫tan x·√(1+tan⁴x)` comes out
    /// through an Euler substitution, not as Rubi's `asinh` + `atanh` pair) and
    /// the only thing that makes an antiderivative right is its derivative.
    ///
    /// Returns `(points checked, points where F itself is a finite real)`.
    fn check_everywhere(
        f: ExprId,
        integrand: ExprId,
        x: ExprId,
        pool: &ExprPool,
    ) -> (usize, usize) {
        let df = simplify(crate::diff::diff(f, x, pool).unwrap().value, pool).value;
        let (mut checked, mut real) = (0usize, 0usize);
        for k in -820..820 {
            let xv = f64::from(k) * 0.0073;
            let Some(rhs) = gate::eval_at(integrand, x, xv, pool).filter(|v| v.is_finite()) else {
                continue;
            };
            if gate::eval_at(f, x, xv, pool).is_some_and(f64::is_finite) {
                real += 1;
            }
            let Some(lhs) = gate::eval_at(df, x, xv, pool).filter(|v| v.is_finite()) else {
                continue;
            };
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "d/dx F ≠ f at x = {xv}: {lhs} vs {rhs}"
            );
            checked += 1;
        }
        (checked, real)
    }

    fn tan(x: ExprId, pool: &ExprPool) -> ExprId {
        pool.func("tan", vec![x])
    }
    fn sec_pow(x: ExprId, k: i32, pool: &ExprPool) -> ExprId {
        pool.pow(pool.func("cos", vec![x]), pool.integer(-k))
    }
    fn sqrt(e: ExprId, pool: &ExprPool) -> ExprId {
        pool.func("sqrt", vec![e])
    }
    fn inv(e: ExprId, pool: &ExprPool) -> ExprId {
        pool.pow(e, pool.integer(-1_i32))
    }

    /// Charlwood #6, `∫tan x·√(1+tan⁴x) dx` — the `tan` family, `deg P = 4`.
    #[test]
    fn charlwood_6_tan_times_sqrt_one_plus_tan_fourth() {
        let (pool, x) = setup();
        let t4 = pool.pow(tan(x, &pool), pool.integer(4_i32));
        let e = pool.mul(vec![
            tan(x, &pool),
            sqrt(pool.add(vec![pool.integer(1_i32), t4]), &pool),
        ]);
        let got = crate::integrate::integrate(e, x, &pool).expect("Charlwood #6 should close");
        let (checked, real) = check_everywhere(got.value, e, x, &pool);
        assert!(checked > 1500, "only {checked} points checked");
        assert_eq!(
            real, checked,
            "the answer should be real on the whole domain"
        );
    }

    /// Charlwood #43, `∫tan x/√(1+tan⁴x) dx` — same curve, reciprocal radical.
    #[test]
    fn charlwood_43_tan_over_sqrt_one_plus_tan_fourth() {
        let (pool, x) = setup();
        let t4 = pool.pow(tan(x, &pool), pool.integer(4_i32));
        let rad = sqrt(pool.add(vec![pool.integer(1_i32), t4]), &pool);
        let e = pool.mul(vec![tan(x, &pool), inv(rad, &pool)]);
        let got = crate::integrate::integrate(e, x, &pool).expect("Charlwood #43 should close");
        let (checked, real) = check_everywhere(got.value, e, x, &pool);
        assert!(checked > 1500, "only {checked} points checked");
        assert_eq!(real, checked);
    }

    /// Charlwood #7, `∫tan x/√(1+sec³x) dx`.
    ///
    /// The generator here is `sec`, not `tan`: dividing by `(sec x)′` cancels
    /// the `tan` identically and leaves `1/(t√(1+t³))`.  Its domain is only
    /// `cos x > 0` — for `cos x < 0` the radicand `1 + sec³x` is negative — so
    /// the check covers about half the grid by construction.
    #[test]
    fn charlwood_7_tan_over_sqrt_one_plus_sec_cubed() {
        let (pool, x) = setup();
        let rad = sqrt(
            pool.add(vec![pool.integer(1_i32), sec_pow(x, 3, &pool)]),
            &pool,
        );
        let e = pool.mul(vec![tan(x, &pool), inv(rad, &pool)]);
        let got = crate::integrate::integrate(e, x, &pool).expect("Charlwood #7 should close");
        let (checked, real) = check_everywhere(got.value, e, x, &pool);
        assert!(checked > 700, "only {checked} points checked");
        assert_eq!(real, checked);
    }

    /// Charlwood #44, `∫sin x/√(1−sin⁶x) dx` — the `sin` family.
    ///
    /// This is the case that pins the sign repair down.  `t = cos x` leaves
    /// `−1/√(1−(1−t²)³)`, whose radicand normalizes to `t²·(t⁴−3t²+3)`; taking
    /// `t` out of the radical costs a `sign(cos x)`, and the answer is only
    /// right on both halves of the domain because a repaired candidate was
    /// proposed and preferred.  Rubi's optimal form carries the same factor.
    #[test]
    fn charlwood_44_sin_over_sqrt_one_minus_sin_sixth() {
        let (pool, x) = setup();
        let s6 = pool.pow(pool.func("sin", vec![x]), pool.integer(6_i32));
        let rad = sqrt(
            pool.add(vec![
                pool.integer(1_i32),
                pool.mul(vec![pool.integer(-1_i32), s6]),
            ]),
            &pool,
        );
        let e = pool.mul(vec![pool.func("sin", vec![x]), inv(rad, &pool)]);
        let got = crate::integrate::integrate(e, x, &pool).expect("Charlwood #44 should close");
        let (checked, real) = check_everywhere(got.value, e, x, &pool);
        assert!(checked > 1500, "only {checked} points checked");
        assert_eq!(
            real, checked,
            "the sign repair is what makes this real on cos x < 0 too"
        );
    }

    /// **The rule that keeps this route out of the false-certificate business.**
    ///
    /// `∫sinh x·√(1+sinh⁴x) dx` reduces under `t = cosh x` to
    /// `∫√(t⁴−2t²+2) dt`, which the engine certifies non-elementary — and that
    /// certificate is about `t`.  `t = g(x)` is not an elementary isomorphism,
    /// so the verdict does not transfer backwards; the route must report a
    /// decline, never `NonElementary`.
    #[test]
    fn a_non_elementary_reduced_integral_declines_rather_than_certifying() {
        let (pool, x) = setup();
        let s4 = pool.pow(pool.func("sinh", vec![x]), pool.integer(4_i32));
        let e = pool.mul(vec![
            pool.func("sinh", vec![x]),
            sqrt(pool.add(vec![pool.integer(1_i32), s4]), &pool),
        ]);
        match crate::integrate::integrate(e, x, &pool) {
            Ok(f) => {
                check_everywhere(f.value, e, x, &pool);
            }
            Err(err) => assert!(
                matches!(err, IntegrationError::NotImplemented(_)),
                "a non-elementary verdict on ∫h dt must not become one on ∫f dx: {err}"
            ),
        }
    }

    /// Charlwood #42, `∫sec x/√(sec⁴x−1) dx` — reached through `t = csc x`.
    #[test]
    fn charlwood_42_sec_over_sqrt_sec_fourth_minus_one() {
        let (pool, x) = setup();
        let rad = sqrt(
            pool.add(vec![pool.integer(-1_i32), sec_pow(x, 4, &pool)]),
            &pool,
        );
        let e = pool.mul(vec![sec_pow(x, 1, &pool), inv(rad, &pool)]);
        let got = crate::integrate::integrate(e, x, &pool).expect("Charlwood #42 should close");
        let (checked, _real) = check_everywhere(got.value, e, x, &pool);
        assert!(checked > 1500, "only {checked} points checked");
    }

    /// The hyperbolic half of the table is a family, not a special case for
    /// `tan`: `∫tanh x·√(1+tanh⁴x) dx` goes through `t = tanh x`.
    #[test]
    fn tanh_times_sqrt_one_plus_tanh_fourth() {
        let (pool, x) = setup();
        let t4 = pool.pow(pool.func("tanh", vec![x]), pool.integer(4_i32));
        let e = pool.mul(vec![
            pool.func("tanh", vec![x]),
            sqrt(pool.add(vec![pool.integer(1_i32), t4]), &pool),
        ]);
        let got = crate::integrate::integrate(e, x, &pool).expect("∫tanh x·√(1+tanh⁴x) dx");
        let (checked, _) = check_everywhere(got.value, e, x, &pool);
        assert!(checked > 1000, "only {checked} points checked");
    }

    /// A bare `x` outside the generator has no image under `t = g(x)`, so the
    /// rewrite must refuse rather than guess.
    #[test]
    fn a_bare_x_outside_the_generator_is_refused() {
        let (pool, x) = setup();
        let e = pool.mul(vec![x, sqrt(tan(x, &pool), &pool)]);
        let t = pool.symbol(T_NAME, Domain::Real);
        assert!(
            rewrite_in_t(e, x, x, Kind::Tan, t, &pool).is_none(),
            "a bare x must not be rewritten in t"
        );
    }

    /// Charlwood #5, `∫cos²x/√(1+cos²x+cos⁴x) dx`: the reduction succeeds and
    /// the *reduced* integral is the one that declines (genus-1 logarithmic
    /// part).  What matters is that the decline stays a decline — the route
    /// must never turn a failed substitution into a non-elementarity claim.
    #[test]
    fn charlwood_5_declines_without_certifying_non_elementary() {
        let (pool, x) = setup();
        let c2 = pool.pow(pool.func("cos", vec![x]), pool.integer(2_i32));
        let c4 = pool.pow(pool.func("cos", vec![x]), pool.integer(4_i32));
        let rad = sqrt(pool.add(vec![pool.integer(1_i32), c2, c4]), &pool);
        let e = pool.mul(vec![c2, inv(rad, &pool)]);
        match crate::integrate::integrate(e, x, &pool) {
            Ok(f) => {
                // If a future genus-1 improvement closes it, it must still be right.
                check_everywhere(f.value, e, x, &pool);
            }
            Err(err) => assert!(
                matches!(err, IntegrationError::NotImplemented(_)),
                "must decline, never certify: {err}"
            ),
        }
    }

    /// Charlwood #45, `∫√(√(sec x+1) − √(sec x−1)) dx`: two distinct nested
    /// radicals, refused at the first step without panicking.
    #[test]
    fn charlwood_45_nested_radicals_declines_cleanly() {
        let (pool, x) = setup();
        let sec = sec_pow(x, 1, &pool);
        let a = sqrt(pool.add(vec![pool.integer(1_i32), sec]), &pool);
        let b = sqrt(pool.add(vec![pool.integer(-1_i32), sec]), &pool);
        let e = sqrt(
            pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]),
            &pool,
        );
        match crate::integrate::integrate(e, x, &pool) {
            Ok(f) => {
                check_everywhere(f.value, e, x, &pool);
            }
            Err(err) => assert!(matches!(err, IntegrationError::NotImplemented(_)), "{err}"),
        }
    }

    /// A polynomial radicand belongs to the genus-0/1 machinery, which emits
    /// better forms; this route must not claim it.
    #[test]
    fn a_polynomial_radicand_is_not_this_routes_business() {
        let (pool, x) = setup();
        let x2 = pool.pow(x, pool.integer(2_i32));
        let e = sqrt(pool.add(vec![pool.integer(1_i32), x2]), &pool);
        assert!(!has_transcendental_radicand(e, x, &pool));
        assert!(try_generator_substitution(e, x, &pool).is_none());
    }

    /// The radicand normalization is an identity up to one sign, and the
    /// witness it hands back *is* that sign.
    #[test]
    fn the_sign_witness_evaluates_to_the_sign_it_claims() {
        let pool = ExprPool::new();
        let t = pool.symbol(T_NAME, Domain::Real);
        // √(1 − (1 − t²)³) = |t|·√(t⁴ − 3t² + 3)
        let one = pool.integer(1_i32);
        let t2 = pool.pow(t, pool.integer(2_i32));
        let inner = pool.add(vec![one, pool.mul(vec![pool.integer(-1_i32), t2])]);
        let cube = pool.pow(inner, pool.integer(3_i32));
        let a = pool.add(vec![one, pool.mul(vec![pool.integer(-1_i32), cube])]);
        let e = pool.pow(pool.func("sqrt", vec![a]), pool.integer(-1_i32));

        let norm = normalize_radicals(e, t, &pool).expect("radicand is rational in t");
        let witness = norm.witness.expect("a square factor was taken out");
        for k in [-0.9_f64, -0.4, 0.3, 0.8] {
            let w = gate::eval_at(witness, t, k, &pool).expect("witness evaluates");
            assert!(
                (w - k.signum()).abs() < 1e-9,
                "witness at t={k} is {w}, not sign(t)"
            );
            // The reduced form times that sign reproduces the original.
            let lhs = gate::eval_at(norm.integrand, t, k, &pool).unwrap() * w;
            let rhs = gate::eval_at(e, t, k, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-9 * (1.0 + rhs.abs()),
                "{lhs} vs {rhs}"
            );
        }
    }
}
