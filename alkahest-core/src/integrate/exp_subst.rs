//! Integrands that are rational functions of **one** exponential `u = e^{g·x}`,
//! including hyperbolic functions and constant-base powers rewritten into that
//! form.
//!
//! # Why this module exists
//!
//! The Risch exponential tower takes one generator per distinct exponential,
//! so `eˣ/(3e^{2x} + 2)` reaches it with the coefficient `1/(3e^{2x} + 2)` of
//! `eˣ`, which it does not treat as a rational function of `eˣ` and declines
//! (`E-INT-001 coefficient … is not a polynomial or rational function`).  The
//! hyperbolic functions never reach it at all: `∫ dx/cosh x` came back as
//! `∫ (expr)^(exp) where base or exp is non-trivial`, and `∫ 10ˣ·eˣ dx` as an
//! irreducible product.
//!
//! All three are the same integral in disguise.  Write every exponential of
//! a linear argument as a power of a single generator `u = e^{g·x}` —
//!
//! ```text
//! sinh a = (eᵃ − e⁻ᵃ)/2      cosh a = (eᵃ + e⁻ᵃ)/2      tanh a = (e^{2a} − 1)/(e^{2a} + 1)
//! bˣ     = e^{x·log b}       e^{c·x + d} = e^d · u^{c/g}
//! ```
//!
//! — and the integrand becomes `R(u)`, `dx = du/(g·u)`, a rational function of
//! `u` that the rational integrator decides completely.  Back-substitution
//! `u ↦ e^{g·x}` gives an antiderivative that is continuous on ℝ (`u > 0`, so
//! none of the rational integrator's logs or arctangents meets a pole that the
//! integrand does not have).
//!
//! When the hyperbolic rewrite does not give a single-generator rational
//! function (`x·sech²x`, `cosh(x²)`), the rewritten integrand is handed to the
//! full integrator instead, which reaches the Risch tower with plain
//! exponentials.
//!
//! # Soundness
//!
//! Each candidate is checked against the **original** integrand with
//! [`verify_antiderivative_status_parametric`] before it is returned; a candidate
//! that cannot be confirmed is a decline (`None`), never an answer, and this
//! module has no path to a non-elementarity verdict.
//!
//! # Where it runs
//!
//! Only on the decline cascade of `engine::integrate_dispatch`, after every
//! existing route has had its turn, so no answer the integrator already gave
//! changes.

use std::cell::Cell;
use std::collections::HashMap;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::engine::{
    integrate, is_free_of, verify_antiderivative_status_parametric, IntegrationError,
};

thread_local! {
    /// Nesting depth of this pass.  The inner integrals it starts are rational
    /// in `u` (or free of hyperbolic functions), so re-entry is bounded anyway;
    /// the cap keeps a pathological nesting from multiplying the cost.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

const MAX_DEPTH: u32 = 2;
/// Largest `|n|` in `u^n` the substitution produces before giving up: the
/// rational integrator's cost grows with the degree.
const MAX_EXPONENT: i64 = 24;

struct DepthGuard;

impl DepthGuard {
    fn enter() -> Option<Self> {
        DEPTH.with(|d| {
            if d.get() >= MAX_DEPTH {
                None
            } else {
                d.set(d.get() + 1);
                Some(DepthGuard)
            }
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get() - 1));
    }
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

fn recip(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(e, pool.integer(-1_i32))
}

/// `true` when `expr` mentions a hyperbolic function or a constant-base power
/// with a `var`-dependent exponent — the shapes [`to_exponential`] rewrites.
fn has_rewritable(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            (matches!(name.as_str(), "sinh" | "cosh" | "tanh")
                && args.iter().any(|&a| !is_free_of(a, var, pool)))
                || args.iter().any(|&a| has_rewritable(a, var, pool))
        }
        ExprData::Pow { base, exp } => {
            (is_free_of(base, var, pool)
                && !is_free_of(exp, var, pool)
                && positive_constant(base, pool))
                || has_rewritable(base, var, pool)
                || has_rewritable(exp, var, pool)
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| has_rewritable(a, var, pool))
        }
        _ => false,
    }
}

/// A numeric constant `> 0` other than `1` (`10` in `10ˣ`).  `log b` is then a
/// real constant, so `bˣ = e^{x·log b}` holds on all of ℝ.
fn positive_constant(base: ExprId, pool: &ExprPool) -> bool {
    match pool.get(base) {
        ExprData::Integer(n) => n.0 > 1,
        ExprData::Rational(r) => r.0 > 0 && r.0 != 1,
        _ => false,
    }
}

/// `(r, k)` with `base = r^k` and `k` maximal, for an integer `base > 1`;
/// `(base, 1)` otherwise.
fn perfect_power(base: ExprId, pool: &ExprPool) -> (ExprId, u32) {
    let ExprData::Integer(n) = pool.get(base) else {
        return (base, 1);
    };
    let n = n.0;
    let bits = n.significant_bits();
    for k in (2..=bits).rev() {
        let r = rug::Integer::from(n.root_ref(k));
        if rug::Integer::from(rug::ops::Pow::pow(&r, k)) == n {
            return (pool.integer(r), k);
        }
    }
    (base, 1)
}

/// Rewrite hyperbolic functions of `var` and `bˣ` into exponentials.  Every
/// rewrite is an identity on ℝ.
fn to_exponential(expr: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    if is_free_of(expr, var, pool) {
        return expr;
    }
    let half = pool.rational(1_i32, 2_i32);
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            let args: Vec<ExprId> = args.iter().map(|&a| to_exponential(a, var, pool)).collect();
            if args.len() == 1 {
                let a = args[0];
                let ep = pool.func("exp", vec![a]);
                let em = pool.func("exp", vec![neg(a, pool)]);
                match name.as_str() {
                    "sinh" => return pool.mul(vec![half, pool.add(vec![ep, neg(em, pool)])]),
                    "cosh" => return pool.mul(vec![half, pool.add(vec![ep, em])]),
                    "tanh" => {
                        let e2 = pool.func("exp", vec![pool.mul(vec![pool.integer(2_i32), a])]);
                        let one = pool.integer(1_i32);
                        return pool.mul(vec![
                            pool.add(vec![e2, pool.integer(-1_i32)]),
                            recip(pool.add(vec![e2, one]), pool),
                        ]);
                    }
                    _ => {}
                }
            }
            pool.func(name, args)
        }
        ExprData::Pow { base, exp } => {
            if is_free_of(base, var, pool) && positive_constant(base, pool) {
                let e = to_exponential(exp, var, pool);
                // `4ˣ = e^{2x·log 2}`: reduce a perfect-power base to its root so
                // that `2ˣ` and `4ˣ` share a generator.
                let (root, k) = perfect_power(base, pool);
                return pool.func(
                    "exp",
                    vec![pool.mul(vec![pool.integer(k), e, pool.func("log", vec![root])])],
                );
            }
            pool.pow(
                to_exponential(base, var, pool),
                to_exponential(exp, var, pool),
            )
        }
        ExprData::Add(args) => {
            pool.add(args.iter().map(|&a| to_exponential(a, var, pool)).collect())
        }
        ExprData::Mul(args) => merge_exponentials(
            args.iter().map(|&a| to_exponential(a, var, pool)).collect(),
            pool,
        ),
        _ => expr,
    }
}

/// `∏ factors`, with every `exp(a)` / `exp(a)^n` (integer `n`) factor merged
/// into one `exp(Σ n·a)` — an identity on ℝ.  This is what turns
/// `10ˣ·eˣ = e^{x·log 10}·eˣ` into the single exponential `e^{x(1 + log 10)}`.
fn merge_exponentials(factors: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    let mut exponent = Vec::new();
    let mut rest = Vec::new();
    for f in factors {
        match pool.get(f) {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                exponent.push(args[0]);
            }
            ExprData::Pow { base, exp } => match (pool.get(base), pool.get(exp)) {
                (ExprData::Func { name, args }, ExprData::Integer(_))
                    if name == "exp" && args.len() == 1 =>
                {
                    exponent.push(pool.mul(vec![exp, args[0]]));
                }
                _ => rest.push(f),
            },
            _ => rest.push(f),
        }
    }
    if exponent.len() >= 2 {
        rest.push(pool.func("exp", vec![pool.add(exponent)]));
    } else {
        rest.extend(exponent.into_iter().map(|a| pool.func("exp", vec![a])));
    }
    if rest.len() == 1 {
        rest[0]
    } else {
        pool.mul(rest)
    }
}

/// Collect the distinct `exp(arg)` nodes whose argument mentions `var`.
fn collect_exps(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    if is_free_of(expr, var, pool) {
        return;
    }
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            if name == "exp" && args.len() == 1 {
                if !out.contains(&expr) {
                    out.push(expr);
                }
                return;
            }
            for a in args {
                collect_exps(a, var, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_exps(base, var, pool, out);
            collect_exps(exp, var, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect_exps(a, var, pool, out);
            }
        }
        _ => {}
    }
}

/// `arg = a·var + b` with `a`, `b` free of `var`, read through the derivative
/// so that `x·(1 + log 10)` and `2·(x + 1)` are recognised as well.
fn linear_parts(arg: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let a = simplify(crate::diff::diff(arg, var, pool).ok()?.value, pool).value;
    if !is_free_of(a, var, pool) {
        return None;
    }
    let b = simplify(pool.add(vec![arg, neg(pool.mul(vec![a, var]), pool)]), pool).value;
    if !is_free_of(b, var, pool) {
        return None;
    }
    Some((a, b))
}

fn as_rational(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

fn is_zero_expr(e: ExprId, pool: &ExprPool) -> bool {
    as_rational(e, pool).is_some_and(|r| r == 0)
}

/// Express `expr` as `R(u)` with `u = e^{g·var}`.  Returns `(R(u), g)` or `None`
/// when some exponential has a non-linear argument, two frequencies are not
/// rationally related, or `var` survives outside the exponentials.
fn single_generator_form(
    expr: ExprId,
    var: ExprId,
    u: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let mut exps = Vec::new();
    collect_exps(expr, var, pool, &mut exps);
    if exps.is_empty() {
        return None;
    }
    let mut parts = Vec::with_capacity(exps.len());
    for &e in &exps {
        let ExprData::Func { args, .. } = pool.get(e) else {
            return None;
        };
        let (a, b) = linear_parts(args[0], var, pool)?;
        if is_zero_expr(a, pool) {
            return None;
        }
        parts.push((e, a, b));
    }
    // Ratios to the first frequency must be rational: a_i = r_i · a_0.
    let a0 = parts[0].1;
    let mut ratios = Vec::with_capacity(parts.len());
    for &(_, a, _) in &parts {
        let r = simplify(pool.mul(vec![a, recip(a0, pool)]), pool).value;
        ratios.push(as_rational(r, pool)?);
    }
    // g = a_0 · N/D with D = lcm of denominators, N = gcd of the scaled numerators.
    let mut den = rug::Integer::from(1);
    for r in &ratios {
        den = den.lcm(r.denom());
    }
    let mut num = rug::Integer::from(0);
    for r in &ratios {
        let scaled = rug::Integer::from(r.numer() * &den) / r.denom();
        num = num.gcd(&scaled);
    }
    if num == 0 {
        return None;
    }
    let step = rug::Rational::from((num.clone(), den.clone()));
    let g = simplify(
        pool.mul(vec![
            a0,
            pool.rational(step.numer().clone(), step.denom().clone()),
        ]),
        pool,
    )
    .value;
    let mut map = HashMap::new();
    for (&(e, _, b), r) in parts.iter().zip(&ratios) {
        let n = rug::Rational::from(r / &step);
        if *n.denom() != 1 {
            return None;
        }
        let n = n.numer().to_i64()?;
        if n.abs() > MAX_EXPONENT {
            return None;
        }
        let mut rep = pool.pow(u, pool.integer(n));
        if !is_zero_expr(b, pool) {
            rep = pool.mul(vec![pool.func("exp", vec![b]), rep]);
        }
        map.insert(e, rep);
    }
    let replaced = crate::kernel::subs(expr, &map, pool);
    if !is_free_of(replaced, var, pool) {
        return None;
    }
    Some((replaced, g))
}

fn derived(rule: &'static str, expr: ExprId, f: ExprId) -> DerivedExpr<ExprId> {
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(rule, expr, f));
    DerivedExpr::with_log(f, log)
}

/// Which half of [`try_exp_substitution`] to run.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpStage {
    /// Only the rational-in-one-exponential substitution: cheap, decisive,
    /// and its answers are the natural closed forms, so it runs before
    /// integration by parts.
    Rational,
    /// Only the fallback that hands hyperbolic integrands, rewritten into
    /// exponentials, to the full integrator.
    Rewrite,
}

/// Integrate `expr` as a rational function of one exponential, after rewriting
/// hyperbolic functions and constant-base powers into exponentials.
///
/// `Ok(None)` is a decline.  A budget error propagates.
pub(crate) fn try_exp_substitution(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    stage: ExpStage,
) -> Result<Option<DerivedExpr<ExprId>>, IntegrationError> {
    crate::budget::check()?;
    let rewritable = has_rewritable(expr, var, pool);
    let mut probe = Vec::new();
    collect_exps(expr, var, pool, &mut probe);
    if !rewritable && probe.is_empty() {
        return Ok(None);
    }
    let Some(_guard) = DepthGuard::enter() else {
        return Ok(None);
    };
    let rewritten = if rewritable {
        simplify(to_exponential(expr, var, pool), pool).value
    } else {
        expr
    };

    // 1. A rational function of a single exponential.
    let u = pool.symbol("__exp_subst_u", Domain::Real);
    let single = if stage == ExpStage::Rational {
        single_generator_form(rewritten, var, u, pool)
    } else {
        None
    };
    if let Some((r, g)) = single {
        // `1/g` is kept outside the inner integral: a symbolic `g` (`log 2`
        // for `2ˣ`) would otherwise stop the rational integrator.
        let h = simplify(pool.mul(vec![r, recip(u, pool)]), pool).value;
        let inner = {
            let _no_root_sum = super::risch::rational_integrate::RootSumSuppressed::enter();
            match integrate(h, u, pool) {
                Ok(v) => Some(pool.mul(vec![recip(g, pool), v.value])),
                Err(e) if e.is_budget() => return Err(e),
                Err(_) => None,
            }
        };
        if let Some(inner) = inner {
            // `log u = g·x` exactly (`u > 0`); substitute that first so the
            // answer does not carry `log(exp(g·x))`.
            let gx = pool.mul(vec![g, var]);
            let mut logs = HashMap::new();
            logs.insert(pool.func("log", vec![u]), gx);
            let inner = crate::kernel::subs(inner, &logs, pool);
            let mut back = HashMap::new();
            back.insert(u, pool.func("exp", vec![gx]));
            let f = simplify(crate::kernel::subs(inner, &back, pool), pool).value;
            if is_free_of(f, u, pool)
                && verify_antiderivative_status_parametric(f, expr, var, pool).is_some()
            {
                let f = super::real_log::realify_logs(f, expr, var, pool);
                return Ok(Some(derived("int_exp_rational_substitution", expr, f)));
            }
        }
    }

    // 2. Hyperbolic functions rewritten into exponentials, handed to the full
    //    integrator (the Risch tower sees plain exponentials).
    if stage == ExpStage::Rewrite && rewritable && rewritten != expr {
        crate::budget::check()?;
        match integrate(rewritten, var, pool) {
            Ok(v) => {
                let f = v.value;
                if verify_antiderivative_status_parametric(f, expr, var, pool).is_some() {
                    let f = super::real_log::realify_logs(f, expr, var, pool);
                    return Ok(Some(derived("int_hyperbolic_as_exponential", expr, f)));
                }
            }
            Err(e) if e.is_budget() => return Err(e),
            Err(_) => {}
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::engine::verify_antiderivative_status;

    fn parsed(src: &str) -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let x = pool.symbol("x", Domain::Real);
        syms.insert("x".to_string(), x);
        let e = crate::parse::parse(src, &pool, &mut syms).expect("parses");
        (pool, x, e)
    }

    #[test]
    fn exponentials_in_a_product_merge() {
        let (pool, x, e) = parsed("10^x*exp(x)");
        let r = simplify(to_exponential(e, x, &pool), &pool).value;
        let mut exps = Vec::new();
        collect_exps(r, x, &pool, &mut exps);
        assert_eq!(exps.len(), 1, "{}", pool.display(r));
    }

    fn solves(src: &str) {
        let (pool, x, e) = parsed(src);
        let f = integrate(e, x, &pool)
            .unwrap_or_else(|err| panic!("∫ {src} dx failed: {err}"))
            .value;
        assert!(
            verify_antiderivative_status(f, e, x, &pool).is_some(),
            "∫ {src} dx = {} does not verify",
            pool.display(f)
        );
    }

    #[test]
    fn sech_and_friends() {
        for src in [
            "1/cosh(x)",
            "1/sinh(x)",
            "1/tanh(x)",
            "1/cosh(x)^2",
            "1/cosh(x)^3",
            "1/sinh(x)^2",
            "tanh(x)^3",
            "1/(cosh(x)*sinh(x))",
            "sinh(x)/cosh(x)^3",
            "1/(1 + cosh(x))",
            "1/(2 + sinh(x))",
            "1/cosh(2*x)",
        ] {
            solves(src);
        }
    }

    #[test]
    fn rational_in_one_exponential() {
        for src in [
            "exp(x)/(3*exp(2*x) + 2)",
            "2*exp(x)/(3*exp(2*x) + 2)",
            "exp(x + 1)/(exp(x) + 1)",
            "exp(6*x)/(exp(4*x) + 1)",
            "10^x*exp(x)",
            "1/(exp(x/2) + exp(x))",
        ] {
            solves(src);
        }
    }

    #[test]
    fn hyperbolic_with_polynomial_coefficient() {
        solves("x/cosh(x)^2");
    }

    #[test]
    fn unrelated_frequencies_decline() {
        let (pool, x, e) = parsed("exp(x)/(exp(sqrt(2)*x) + 1)");
        let u = pool.symbol("u", Domain::Real);
        assert!(single_generator_form(e, x, u, &pool).is_none());
    }
}
