//! `E[f(X)]` — the general route, and the two safe combination rules.
//!
//! # The pipeline
//!
//! 1. **Convergence first.** `E[f(X)]` is quadratured numerically before any
//!    symbolic work starts. `∫ e^{z²/2} dz` has no antiderivative *and* no
//!    value, and "the integrator declined" is the less useful half of that
//!    answer: a caller told the integral did not close will go looking for a
//!    cleverer integrator, and there is nothing to find.
//! 2. **Split the payoff.** `max(x - K, 0)` is not differentiable and no
//!    integrator will touch it, but it is *exactly* `(x - K)·1_{x>K}`, so the
//!    support is cut at the kink and each side carries its own linear piece.
//!    The cut point is computed from the expression, not pattern-matched
//!    against a payoff catalogue.
//! 3. **Reduce.** Each piece is pushed through the distribution's change of
//!    variables (see [`super::dists::Reduction`]) so the density becomes a
//!    standard kernel.
//! 4. **Prepare.** The reduced integrand is expanded into a sum, each term's
//!    `z`-free factor is pulled out front, and the `z`-dependent exponentials
//!    in a term are fused into one. That last step is not cosmetic: alkahest's
//!    integrator closes `∫e^{-z²/2+σz}` and declines the identical
//!    `∫e^{σz}·e^{-z²/2}`, so fusing is the difference between a Black–Scholes
//!    derivation and a refusal.
//! 5. **Integrate** each `z`-kernel with
//!    [`crate::integrate::integrate_definite`], and refuse — naming the
//!    integral — if any of them declines.
//! 6. **Verify** the assembled answer against quadrature of `f(x)·p(x)` in the
//!    *original* variable. Steps 2–5 are exactly where a wrong answer would
//!    come from, and this is the step that catches one.
//!
//! # Black–Scholes
//!
//! `E[max(S - K, 0)]` for `S ~ LogNormal(μ, σ)` runs the pipeline above with
//! no special case anywhere in it: the kink splits the support at `K`, the
//! log-normal reduction turns `∫_K^∞ (s-K)p(s)ds` into
//! `∫_{d}^{∞} (e^{μ+σz} - K)φ(z)dz` with `d = (log K - μ)/σ`, the fusion step
//! turns the first term into `∫e^{-z²/2+σz}`, and the integrator returns `erf`
//! for both halves. The result is the Black–Scholes formula, derived.
//!
//! The split at `K` needs `K` to be inside the support, which for a positive
//! random variable means `K > 0`. When `K` is symbolic that is not decidable,
//! so it is **carried** as a [`SideCondition`] on the derivation rather than
//! assumed silently.

use std::collections::HashMap;

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::integrate::integrate_definite;
use crate::kernel::{subs, ExprData, ExprId, ExprPool};
use crate::simplify::simplify;

use super::dists::{self, Sign};
use super::verify;
use super::{sub, Distribution, ProbError, Support};

/// Most pieces a payoff may be split into. Two kinks is a collar; more than
/// that and the split loop, not the mathematics, is what would be under test.
const MAX_PIECES: usize = 4;

/// Cap on the expansion of a product of sums, so a pathological integrand
/// cannot make this loop the expensive part of the library.
const MAX_TERMS: usize = 64;

// ---------------------------------------------------------------------------
// E[f(X)]
// ---------------------------------------------------------------------------

/// `E[f(X)]` where `f` is written in the variable `x`.
///
/// # Errors
///
/// * [`ProbError::Divergent`] when the defining integral does not converge.
/// * [`ProbError::IntegralDidNotClose`] naming the integral the symbolic
///   integrator declined.
/// * [`ProbError::Unverified`] when the assembled closed form did not survive
///   the numeric gate — the value is discarded, not returned with a caveat.
/// * [`ProbError::Unsupported`] for a shape outside the pipeline: a
///   non-polynomial `f` over a countably infinite discrete support, or a
///   payoff with more than [`MAX_PIECES`] pieces.
pub fn expectation(
    f: ExprId,
    x: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    let definition = verify::definition_integrand(f, x, dist, pool);
    if let Some(divergent) = verify::convergence_probe(definition, x, dist, pool) {
        return Err(divergent);
    }

    let support = dist.support(pool);
    let (claim, mut log, breaks) = if support.is_discrete() {
        let (c, l) = discrete_expectation(f, x, dist, &support, pool)?;
        (c, l, Vec::new())
    } else {
        continuous_expectation(f, x, dist, pool)?
    };

    let claim = simplify(claim, pool).value;
    let evidence = verify::check_with_breaks(claim, definition, x, dist, &breaks, pool)?;
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    Ok(DerivedExpr::with_log(claim, log))
}

// ---------------------------------------------------------------------------
// Discrete
// ---------------------------------------------------------------------------

fn discrete_expectation(
    f: ExprId,
    x: ExprId,
    dist: &Distribution,
    support: &Support,
    pool: &ExprPool,
) -> Result<(ExprId, DerivationLog), ProbError> {
    let mut log = DerivationLog::new();
    match support {
        Support::IntegersUpTo(n) => {
            // The sum is finite, so it is the definition, evaluated. No
            // closed form is being claimed beyond `Σ` itself.
            let mut terms = Vec::with_capacity(*n as usize + 1);
            for k in 0..=*n {
                let kk = pool.integer(k);
                let mut m = HashMap::new();
                m.insert(x, kk);
                let fk = subs(f, &m, pool);
                terms.push(pool.mul(vec![fk, dists::pdf(dist, kk, pool)]));
            }
            let total = simplify(pool.add(terms), pool).value;
            log.push(RewriteStep::simple("prob_finite_expectation_sum", f, total));
            Ok((total, log))
        }
        Support::NonNegativeIntegers => {
            // Countably infinite: only a polynomial `f` closes, through the
            // factorial-moment expansion. Anything else is refused rather than
            // truncated — a truncated infinite sum returned as an expectation
            // is a wrong answer with a plausible magnitude.
            let Some(coeffs) = polynomial_coefficients(f, x, pool) else {
                return Err(ProbError::Unsupported(format!(
                    "E[f(X)] over {} needs f to be a polynomial in the variate: the support is \
                     countably infinite and a general f has no closed-form sum here",
                    dist.kind().name()
                )));
            };
            if coeffs.len() > super::MAX_MOMENT_ORDER as usize + 1 {
                return Err(ProbError::Unsupported(format!(
                    "polynomial degree {} exceeds MAX_MOMENT_ORDER = {}",
                    coeffs.len() - 1,
                    super::MAX_MOMENT_ORDER
                )));
            }
            let mut terms = Vec::new();
            for (i, c) in coeffs.iter().enumerate() {
                if is_zero(*c, pool) {
                    continue;
                }
                let m = super::moments::raw_moment_unchecked(dist, i as u32, pool)?;
                terms.push(pool.mul(vec![*c, m]));
            }
            let total = simplify(pool.add(terms), pool).value;
            log.push(RewriteStep::simple("prob_moment_expansion", f, total));
            Ok((total, log))
        }
        _ => unreachable!("only discrete supports reach here"),
    }
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0)
}

/// Coefficients of `f` as a polynomial in `x`, lowest degree first, or `None`
/// when `f` is not one.
fn polynomial_coefficients(f: ExprId, x: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    fn go(e: ExprId, x: ExprId, pool: &ExprPool, depth: u32) -> Option<Vec<ExprId>> {
        if depth > 16 {
            return None;
        }
        if !dists::contains(e, x, pool) {
            return Some(vec![e]);
        }
        if e == x {
            return Some(vec![pool.integer(0), pool.integer(1)]);
        }
        match pool.get(e) {
            ExprData::Add(args) => {
                let mut acc: Vec<ExprId> = vec![pool.integer(0)];
                for a in args {
                    let c = go(a, x, pool, depth + 1)?;
                    if c.len() > acc.len() {
                        acc.resize(c.len(), pool.integer(0));
                    }
                    for (i, v) in c.iter().enumerate() {
                        acc[i] = pool.add(vec![acc[i], *v]);
                    }
                }
                Some(acc)
            }
            ExprData::Mul(args) => {
                let mut acc: Vec<ExprId> = vec![pool.integer(1)];
                for a in args {
                    let c = go(a, x, pool, depth + 1)?;
                    if acc.len() + c.len() - 1 > super::MAX_MOMENT_ORDER as usize + 1 {
                        return None;
                    }
                    let mut next = vec![pool.integer(0); acc.len() + c.len() - 1];
                    for (i, u) in acc.iter().enumerate() {
                        for (j, v) in c.iter().enumerate() {
                            next[i + j] = pool.add(vec![next[i + j], pool.mul(vec![*u, *v])]);
                        }
                    }
                    acc = next;
                }
                Some(acc)
            }
            ExprData::Pow { base, exp } => {
                let ExprData::Integer(n) = pool.get(exp) else {
                    return None;
                };
                let n = n.0.to_u32()?;
                if n > super::MAX_MOMENT_ORDER {
                    return None;
                }
                let b = go(base, x, pool, depth + 1)?;
                let mut acc = vec![pool.integer(1)];
                for _ in 0..n {
                    let mut next = vec![pool.integer(0); acc.len() + b.len() - 1];
                    for (i, u) in acc.iter().enumerate() {
                        for (j, v) in b.iter().enumerate() {
                            next[i + j] = pool.add(vec![next[i + j], pool.mul(vec![*u, *v])]);
                        }
                    }
                    acc = next;
                }
                Some(acc)
            }
            _ => None,
        }
    }
    let c = go(f, x, pool, 0)?;
    Some(c.into_iter().map(|e| simplify(e, pool).value).collect())
}

// ---------------------------------------------------------------------------
// Continuous
// ---------------------------------------------------------------------------

/// One piece of a split payoff: `f` restricted to `[lo, hi]` in `x`.
struct Piece {
    lo: ExprId,
    hi: ExprId,
    f: ExprId,
    conditions: Vec<SideCondition>,
}

type ContinuousResult = (ExprId, DerivationLog, Vec<ExprId>);

fn continuous_expectation(
    f: ExprId,
    x: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Result<ContinuousResult, ProbError> {
    let mut log = DerivationLog::new();
    let Some(red) = dists::reduction(dist, x, pool) else {
        return Err(ProbError::Unsupported(
            "no reduction for this distribution".to_string(),
        ));
    };
    let (support_lo, support_hi) = support_bounds(dist, pool);
    let pieces = split_payoff(f, x, support_lo, support_hi, pool)?;
    if pieces.len() > 1 {
        log.push(RewriteStep::simple("prob_split_payoff_at_kink", f, f));
    }

    let mut total: Vec<ExprId> = Vec::new();
    for piece in &pieces {
        // Map the x-bounds through ζ = ξ⁻¹. Monotone increasing (see
        // `Reduction`), so the orientation is preserved and no swap is needed.
        let z_lo = map_bound(piece.lo, x, &red, support_lo, pool);
        let z_hi = map_bound(piece.hi, x, &red, support_hi, pool);

        let mut m = HashMap::new();
        m.insert(x, red.x_of_z);
        let f_of_z = subs(piece.f, &m, pool);
        let integrand = simplify(pool.mul(vec![f_of_z, red.density]), pool).value;

        for (constant, kernel) in prepare_terms(integrand, red.z, pool)? {
            if is_zero(kernel, pool) || is_zero(constant, pool) {
                continue;
            }
            // Complete the square before handing over. `∫e^{-z²/2+σz}` is
            // within alkahest's integrator's reach as an *antiderivative*, but
            // its limit engine cannot evaluate the resulting shifted `erf` at
            // `±∞` and the definite integral is declined. Substituting
            // `w = z - σ` moves the shift out of the `erf` and into a constant,
            // and `∫e^{-w²/2}` between the same (shifted) bounds is a limit it
            // does evaluate. Exact, and general for any quadratic exponent.
            let (kernel, constant, z_lo, z_hi) =
                match complete_the_square(kernel, red.z, z_lo, z_hi, pool) {
                    Some((k, extra, lo, hi)) => {
                        log.push(RewriteStep::simple("prob_complete_the_square", kernel, k));
                        (k, pool.mul(vec![constant, extra]), lo, hi)
                    }
                    None => (kernel, constant, z_lo, z_hi),
                };
            let piece_value = integrate_definite(kernel, red.z, z_lo, z_hi, pool).map_err(|e| {
                ProbError::IntegralDidNotClose(format!(
                    "∫_{{{}}}^{{{}}} {} d{} — {e}",
                    pool.display(z_lo),
                    pool.display(z_hi),
                    pool.display(kernel),
                    pool.display(red.z)
                ))
            })?;
            log = log.merge(piece_value.log);
            total.push(pool.mul(vec![constant, piece_value.value]));
        }
        for c in &piece.conditions {
            log.push(RewriteStep::with_conditions(
                "prob_kink_inside_support",
                piece.f,
                piece.f,
                vec![c.clone()],
            ));
        }
    }

    let value = if total.is_empty() {
        pool.integer(0)
    } else {
        pool.add(total)
    };
    // Hand the kinks on: the verifier needs to cut its quadrature at the same
    // places, or a corner in the integrand costs it the analyticity a
    // double-exponential rule is built on.
    let breaks = pieces.iter().skip(1).map(|p| p.lo).collect::<Vec<ExprId>>();
    Ok((value, log, breaks))
}

fn support_bounds(dist: &Distribution, pool: &ExprPool) -> (ExprId, ExprId) {
    let inf = pool.pos_infinity();
    let neg_inf = pool.mul(vec![pool.integer(-1), inf]);
    match dist.support(pool) {
        Support::Real => (neg_inf, inf),
        Support::Positive => (pool.integer(0), inf),
        Support::Interval(lo, hi) => (lo, hi),
        _ => (neg_inf, inf),
    }
}

/// `ζ(bound)`, or the reduction's own endpoint when the bound *is* the
/// endpoint of the support (where `ζ` would be `log 0` or similar).
fn map_bound(
    bound: ExprId,
    x: ExprId,
    red: &dists::Reduction,
    support_end: ExprId,
    pool: &ExprPool,
) -> ExprId {
    if bound == support_end {
        return if support_end == pool.pos_infinity() {
            red.hi
        } else {
            red.lo
        };
    }
    let mut m = HashMap::new();
    m.insert(x, bound);
    simplify(subs(red.z_of_x, &m, pool), pool).value
}

/// Cut the support at every kink `f` has, and return `f` with each `max`/`min`
/// replaced by the branch that is active on that piece.
///
/// `max(u, v)` is handled when `u - v` is affine in `x` with a coefficient
/// whose sign can be decided numerically — which is every payoff of the form
/// `max(x - K, 0)`, `max(K - x, 0)`, `min(x, C)`. A kink this cannot locate is
/// left in place, and the integrator then declines it, which surfaces as
/// [`ProbError::IntegralDidNotClose`] rather than as a quietly wrong split.
fn split_payoff(
    f: ExprId,
    x: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Result<Vec<Piece>, ProbError> {
    let mut pieces = vec![Piece {
        lo,
        hi,
        f,
        conditions: Vec::new(),
    }];
    for _ in 0..MAX_PIECES {
        let mut split_index = None;
        for (i, p) in pieces.iter().enumerate() {
            if find_kink(p.f, x, pool).is_some() {
                split_index = Some(i);
                break;
            }
        }
        let Some(i) = split_index else {
            return Ok(pieces);
        };
        let piece = pieces.remove(i);
        let (node, lower_branch, upper_branch, root) =
            find_kink(piece.f, x, pool).expect("just found");
        let mut conditions = piece.conditions.clone();
        // The cut point has to lie inside the support for the split to be the
        // partition it claims to be. When that is not decidable it travels
        // with the answer instead of being assumed.
        if let Some(c) = kink_inside_condition(root, piece.lo, piece.hi, pool) {
            conditions.push(c);
        }
        let below = replace(piece.f, node, lower_branch, pool);
        let above = replace(piece.f, node, upper_branch, pool);
        pieces.insert(
            i,
            Piece {
                lo: root,
                hi: piece.hi,
                f: above,
                conditions: conditions.clone(),
            },
        );
        pieces.insert(
            i,
            Piece {
                lo: piece.lo,
                hi: root,
                f: below,
                conditions,
            },
        );
    }
    Err(ProbError::Unsupported(format!(
        "the payoff has more than {MAX_PIECES} kinks in the integration variable"
    )))
}

/// `(node, branch below the root, branch above the root, root)` for the first
/// locatable `max`/`min` in `e`.
fn find_kink(e: ExprId, x: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    match pool.get(e) {
        ExprData::Func { name, args } if args.len() == 2 && (name == "max" || name == "min") => {
            let (u, v) = (args[0], args[1]);
            let d = simplify(sub(u, v, pool), pool).value;
            let (slope, intercept) = affine_in(d, x, pool)?;
            let slope_sign = dists::sign_of(slope, pool);
            let root = simplify(
                super::div(pool.mul(vec![pool.integer(-1), intercept]), slope, pool),
                pool,
            )
            .value;
            // Below the root `u - v` has the opposite sign to its slope.
            let (below, above) = match (name.as_str(), slope_sign) {
                ("max", Sign::Positive) => (v, u),
                ("max", Sign::Negative) => (u, v),
                ("min", Sign::Positive) => (u, v),
                ("min", Sign::Negative) => (v, u),
                _ => return None,
            };
            Some((e, below, above, root))
        }
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().find_map(|&a| find_kink(a, x, pool))
        }
        ExprData::Pow { base, exp } => find_kink(base, x, pool).or_else(|| find_kink(exp, x, pool)),
        _ => None,
    }
}

/// `(slope, intercept)` when `e = slope·x + intercept` with both free of `x`.
fn affine_in(e: ExprId, x: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let coeffs = polynomial_coefficients(e, x, pool)?;
    match coeffs.len() {
        0 => None,
        1 => None, // no `x` at all: not a kink in this variable
        2 => Some((coeffs[1], coeffs[0])),
        _ => {
            if coeffs[2..].iter().all(|&c| is_zero(c, pool)) {
                Some((coeffs[1], coeffs[0]))
            } else {
                None
            }
        }
    }
}

/// A side condition asserting the kink is strictly inside `(lo, hi)`, or
/// `None` when that is already decided.
fn kink_inside_condition(
    root: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Option<SideCondition> {
    let inf = pool.pos_infinity();
    let mut gaps = Vec::new();
    if lo != pool.mul(vec![pool.integer(-1), inf]) {
        gaps.push(sub(root, lo, pool));
    }
    if hi != inf {
        gaps.push(sub(hi, root, pool));
    }
    for g in gaps {
        if matches!(dists::sign_of(g, pool), Sign::Unknown) {
            return Some(SideCondition::Positive(simplify(g, pool).value));
        }
    }
    None
}

fn replace(e: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(from, to);
    subs(e, &m, pool)
}

// ---------------------------------------------------------------------------
// Preparing an integrand for the integrator
// ---------------------------------------------------------------------------

/// Split the reduced integrand into `(z-free constant, z-kernel)` terms.
///
/// The two rewrites here are the ones alkahest's integrator needs and does not
/// do for itself:
///
/// * **pull the constant out** — `∫c·g = c∫g` is linearity, and it keeps a
///   symbolic parameter out of the integrand where it would otherwise be
///   mistaken for a second generator;
/// * **fuse the exponentials** — `e^a·e^b = e^{a+b}` for real `a`, `b`. The
///   integrator completes the square inside a single `exp` and refuses a
///   product of two, so this rewrite is load-bearing rather than tidy.
fn prepare_terms(
    integrand: ExprId,
    z: ExprId,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, ExprId)>, ProbError> {
    let expanded = expand(integrand, pool, 0);
    let terms: Vec<ExprId> = match pool.get(expanded) {
        ExprData::Add(args) => args.to_vec(),
        _ => vec![expanded],
    };
    if terms.len() > MAX_TERMS {
        return Err(ProbError::Unsupported(format!(
            "the reduced integrand expanded to {} terms, past the ceiling of {MAX_TERMS}",
            terms.len()
        )));
    }
    Ok(terms
        .into_iter()
        .map(|t| split_constant(t, z, pool))
        .collect())
}

fn split_constant(term: ExprId, z: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let factors: Vec<ExprId> = match pool.get(term) {
        ExprData::Mul(args) => args.to_vec(),
        _ => vec![term],
    };
    let mut constants = Vec::new();
    let mut kernels = Vec::new();
    let mut exp_args = Vec::new();
    for fct in factors {
        if !dists::contains(fct, z, pool) {
            constants.push(fct);
            continue;
        }
        match pool.get(fct) {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                exp_args.push(args[0])
            }
            _ => kernels.push(fct),
        }
    }
    if !exp_args.is_empty() {
        kernels.push(pool.func("exp", vec![pool.add(exp_args)]));
    }
    let constant = if constants.is_empty() {
        pool.integer(1)
    } else {
        simplify(pool.mul(constants), pool).value
    };
    let kernel = if kernels.is_empty() {
        pool.integer(1)
    } else {
        pool.mul(kernels)
    };
    (constant, kernel)
}

/// Rewrite `P(z)·e^{az²+bz+c}` as `e^{c-b²/4a}·P(z - b/2a)·e^{az²}` with the
/// bounds shifted by `b/2a`, when the exponent is quadratic with a nonzero
/// linear part.
///
/// Returns `(new kernel, constant factor, new lower bound, new upper bound)`.
/// An infinite bound stays infinite — shifting `∞` is still `∞`, and writing
/// `∞ + σ` would give the limit engine an expression it cannot reduce.
fn complete_the_square(
    kernel: ExprId,
    z: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    let factors: Vec<ExprId> = match pool.get(kernel) {
        ExprData::Mul(args) => args.to_vec(),
        _ => vec![kernel],
    };
    let mut exp_index = None;
    for (i, &fct) in factors.iter().enumerate() {
        if let ExprData::Func { name, args } = pool.get(fct) {
            if name == "exp" && args.len() == 1 {
                exp_index = Some((i, args[0]));
                break;
            }
        }
    }
    let (idx, arg) = exp_index?;
    let coeffs = polynomial_coefficients(arg, z, pool)?;
    if coeffs.len() != 3 {
        return None;
    }
    let (c0, c1, c2) = (coeffs[0], coeffs[1], coeffs[2]);
    if is_zero(c1, pool) {
        return None;
    }
    // A positive leading coefficient is a divergent Gaussian; leave it alone
    // rather than producing a tidy expression for something with no value.
    if !matches!(dists::sign_of(c2, pool), Sign::Negative) {
        return None;
    }
    let two = pool.integer(2);
    let four = pool.integer(4);
    let shift = super::div(c1, pool.mul(vec![two, c2]), pool);
    let constant = pool.func(
        "exp",
        vec![sub(
            c0,
            super::div(pool.pow(c1, two), pool.mul(vec![four, c2]), pool),
            pool,
        )],
    );
    let new_exp = pool.func("exp", vec![pool.mul(vec![c2, pool.pow(z, two)])]);
    let mut rest: Vec<ExprId> = Vec::with_capacity(factors.len());
    let mut m = HashMap::new();
    m.insert(z, sub(z, shift, pool));
    for (i, &fct) in factors.iter().enumerate() {
        if i == idx {
            continue;
        }
        rest.push(subs(fct, &m, pool));
    }
    rest.push(new_exp);
    let inf = pool.pos_infinity();
    let neg_inf = pool.mul(vec![pool.integer(-1), inf]);
    let shift_bound = |b: ExprId| -> ExprId {
        if b == inf || b == neg_inf {
            b
        } else {
            simplify(pool.add(vec![b, shift]), pool).value
        }
    };
    Some((
        simplify(pool.mul(rest), pool).value,
        simplify(constant, pool).value,
        shift_bound(lo),
        shift_bound(hi),
    ))
}

/// Distribute products over sums so the caller sees a flat `Add`.
fn expand(e: ExprId, pool: &ExprPool, depth: u32) -> ExprId {
    if depth > 8 {
        return e;
    }
    match pool.get(e) {
        ExprData::Add(args) => pool.add(args.iter().map(|&a| expand(a, pool, depth + 1)).collect()),
        ExprData::Mul(args) => {
            let parts: Vec<Vec<ExprId>> = args
                .iter()
                .map(|&a| match pool.get(expand(a, pool, depth + 1)) {
                    ExprData::Add(inner) => inner.to_vec(),
                    other => vec![pool.intern(other)],
                })
                .collect();
            let mut acc: Vec<Vec<ExprId>> = vec![Vec::new()];
            for part in &parts {
                if acc.len() * part.len() > MAX_TERMS {
                    return e;
                }
                let mut next = Vec::with_capacity(acc.len() * part.len());
                for a in &acc {
                    for &p in part {
                        let mut row = a.clone();
                        row.push(p);
                        next.push(row);
                    }
                }
                acc = next;
            }
            pool.add(acc.into_iter().map(|row| pool.mul(row)).collect())
        }
        _ => e,
    }
}

// ---------------------------------------------------------------------------
// Linearity, and the one rule that needs independence
// ---------------------------------------------------------------------------

/// `E[Σ aᵢXᵢ + c]  =  Σ aᵢE[Xᵢ] + c`.
///
/// Linearity of expectation holds for **any** joint law, dependent or not —
/// which is exactly why this is the combination rule the module offers and
/// covariance is not. `expr` must be affine in the listed variates, with
/// coefficients free of all of them; anything else is refused, because the
/// moment a product `XY` appears the answer depends on a joint distribution
/// this module does not model.
///
/// # Errors
///
/// [`ProbError::Unsupported`] when `expr` is not affine in the variates, and
/// whatever [`Distribution::mean`] returns for a component whose mean does not
/// verify.
pub fn expectation_affine(
    expr: ExprId,
    variates: &[(ExprId, Distribution)],
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    let (coeffs, constant) = affine_decomposition(expr, variates, pool)?;
    let mut log = DerivationLog::new();
    let mut terms = vec![constant];
    for ((_, dist), a) in variates.iter().zip(coeffs.iter()) {
        if is_zero(*a, pool) {
            continue;
        }
        let m = dist.mean(pool)?;
        log = log.merge(m.log);
        terms.push(pool.mul(vec![*a, m.value]));
    }
    let value = simplify(pool.add(terms), pool).value;
    log.push(RewriteStep::simple(
        "prob_linearity_of_expectation",
        expr,
        value,
    ));
    Ok(DerivedExpr::with_log(value, log))
}

/// `Var[Σ aᵢXᵢ + c] = Σ aᵢ²Var[Xᵢ]`, **assuming the variates are independent**.
///
/// The assumption is in the name because it cannot be checked: nothing in a
/// list of marginal distributions records whether they are independent, and
/// under dependence the answer is wrong by `2Σ_{i<j} aᵢaⱼCov(Xᵢ,Xⱼ)` with no
/// indication that anything is missing. A caller who cannot assert
/// independence should not call this.
///
/// # Errors
///
/// As [`expectation_affine`].
pub fn variance_affine_independent(
    expr: ExprId,
    variates: &[(ExprId, Distribution)],
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    let (coeffs, _) = affine_decomposition(expr, variates, pool)?;
    let mut log = DerivationLog::new();
    let mut terms = Vec::new();
    for ((_, dist), a) in variates.iter().zip(coeffs.iter()) {
        if is_zero(*a, pool) {
            continue;
        }
        let v = dist.variance(pool)?;
        log = log.merge(v.log);
        terms.push(pool.mul(vec![pool.pow(*a, pool.integer(2)), v.value]));
    }
    let value = if terms.is_empty() {
        pool.integer(0)
    } else {
        simplify(pool.add(terms), pool).value
    };
    log.push(RewriteStep::simple(
        "prob_variance_of_independent_sum",
        expr,
        value,
    ));
    Ok(DerivedExpr::with_log(value, log))
}

/// `expr = Σ aᵢ·varᵢ + c`, with every `aᵢ` and `c` free of every variate.
fn affine_decomposition(
    expr: ExprId,
    variates: &[(ExprId, Distribution)],
    pool: &ExprPool,
) -> Result<(Vec<ExprId>, ExprId), ProbError> {
    let vars: Vec<ExprId> = variates.iter().map(|(v, _)| *v).collect();
    let mut coeffs = vec![pool.integer(0); vars.len()];
    let mut constant = pool.integer(0);

    let terms: Vec<ExprId> = match pool.get(simplify(expr, pool).value) {
        ExprData::Add(args) => args.to_vec(),
        other => vec![pool.intern(other)],
    };
    for term in terms {
        let present: Vec<usize> = vars
            .iter()
            .enumerate()
            .filter(|(_, &v)| dists::contains(term, v, pool))
            .map(|(i, _)| i)
            .collect();
        match present.as_slice() {
            [] => constant = pool.add(vec![constant, term]),
            [i] => {
                let v = vars[*i];
                let Some(c) = linear_coefficient(term, v, pool) else {
                    return Err(ProbError::Unsupported(format!(
                        "E[·] of `{}` is not affine in `{}`; a non-linear function of a variate \
                         needs `expectation`, and a product of two variates needs a joint law \
                         this module does not model",
                        pool.display(term),
                        pool.display(v)
                    )));
                };
                coeffs[*i] = pool.add(vec![coeffs[*i], c]);
            }
            _ => {
                return Err(ProbError::Unsupported(format!(
                    "`{}` involves more than one variate at once: its expectation depends on \
                     their joint law, which this module does not model",
                    pool.display(term)
                )))
            }
        }
    }
    Ok((
        coeffs
            .into_iter()
            .map(|c| simplify(c, pool).value)
            .collect(),
        simplify(constant, pool).value,
    ))
}

/// `c` when `term = c·v` with `c` free of `v`.
fn linear_coefficient(term: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let coeffs = polynomial_coefficients(term, v, pool)?;
    if coeffs.len() != 2 || !is_zero(coeffs[0], pool) {
        return None;
    }
    Some(coeffs[1])
}
