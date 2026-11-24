//! Rational functions of `sin`, `cos`, `tan` at **several commensurable
//! angles**, or with only monomial denominators, reduced to a rational
//! function of one angle.
//!
//! The half-angle and Bioche routes read `R(sin x, cos x)` with every
//! argument exactly `x`, and only fire when a trigonometric sum sits in a
//! denominator.  Two common shapes fall outside both:
//!
//! * mixed angles — `cos x/(sin x·(1 + tan(x/2)))`, which is what integration
//!   by parts hands over after a half-angle antiderivative;
//! * monomial denominators that hide a sum — `cot x·(tan x − sec x)`, which is
//!   `1 − csc x`.
//!
//! Every angle `aᵢ·x` with rational ratios is written as `kᵢ·θ`, `θ = g·x`, and
//! `sin(kθ)`, `cos(kθ)` are expanded into `s = sin θ`, `c = cos θ` (de Moivre).
//! The result `R(s, c)` is put over a common denominator; if that denominator
//! is a monomial `sⁱcʲ` the quotient is expanded into monomials.  The
//! integrand `R(sin y, cos y)/g` in a fresh `y = θ` then goes through the full
//! integrator — which now sees a single angle and, where there is one, a
//! trigonometric sum in the denominator — and the answer is back-substituted
//! `y ↦ g·x` and gated against the original integrand.

use std::cell::Cell;
use std::collections::HashMap;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::{simplify, simplify_expanded};

use super::engine::{integrate, is_free_of, verify_antiderivative_status, IntegrationError};

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

struct ActiveGuard;

impl ActiveGuard {
    fn enter() -> Option<Self> {
        ACTIVE.with(|a| {
            if a.get() {
                None
            } else {
                a.set(true);
                Some(ActiveGuard)
            }
        })
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.with(|a| a.set(false));
    }
}

/// Largest multiple `|k|` of the base angle that is expanded.
const MAX_MULTIPLE: i64 = 8;

fn as_rational(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

/// The rational slope `a` of a trig argument `a·var`, or `None`.
fn slope(arg: ExprId, var: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    if arg == var {
        return Some(rug::Rational::from(1));
    }
    let a = simplify(crate::diff::diff(arg, var, pool).ok()?.value, pool).value;
    let a = as_rational(a, pool)?;
    let rest = simplify(
        pool.add(vec![
            arg,
            pool.mul(vec![
                pool.rational(-a.numer().clone(), a.denom().clone()),
                var,
            ]),
        ]),
        pool,
    )
    .value;
    (as_rational(rest, pool)? == 0 && a != 0).then_some(a)
}

/// Collect the slopes of every `sin`/`cos`/`tan` argument; `None` when `var`
/// appears anywhere else (or under another function).
fn collect(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<rug::Rational>) -> Option<()> {
    if is_free_of(expr, var, pool) {
        return Some(());
    }
    match pool.get(expr) {
        ExprData::Func { name, args } if args.len() == 1 => {
            if !matches!(name.as_str(), "sin" | "cos" | "tan") {
                return None;
            }
            let a = slope(args[0], var, pool)?;
            if !out.contains(&a) {
                out.push(a);
            }
            Some(())
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect(a, var, pool, out)?;
            }
            Some(())
        }
        ExprData::Pow { base, exp } => {
            if !is_free_of(exp, var, pool) {
                return None;
            }
            as_rational(exp, pool).filter(|r| *r.denom() == 1)?;
            collect(base, var, pool, out)
        }
        _ => None,
    }
}

/// `(cos kθ, sin kθ)` as polynomials in `c`, `s` (de Moivre), `k ≥ 0`.
fn multiple_angle(k: i64, s: ExprId, c: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    // (c + i s)^k = Σ C(k, j) c^{k−j} (i s)^j.
    let mut re = Vec::new();
    let mut im = Vec::new();
    let mut binom = rug::Integer::from(1);
    for j in 0..=k {
        let term = pool.mul(vec![
            pool.integer(binom.clone()),
            pool.pow(c, pool.integer(k - j)),
            pool.pow(s, pool.integer(j)),
        ]);
        let sign = if (j / 2) % 2 == 0 { 1 } else { -1 };
        let signed = pool.mul(vec![pool.integer(sign), term]);
        if j % 2 == 0 {
            re.push(signed);
        } else {
            im.push(signed);
        }
        binom = binom * (k - j) / (j + 1);
    }
    let sum = |v: Vec<ExprId>| {
        if v.is_empty() {
            pool.integer(0_i32)
        } else {
            pool.add(v)
        }
    };
    (sum(re), sum(im))
}

fn rewrite(
    expr: ExprId,
    var: ExprId,
    g: &rug::Rational,
    s: ExprId,
    c: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if is_free_of(expr, var, pool) {
        return Some(expr);
    }
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            let a = slope(args[0], var, pool)?;
            let k = rug::Rational::from(&a / g);
            if *k.denom() != 1 {
                return None;
            }
            let k = k.numer().to_i64()?;
            if k.abs() > MAX_MULTIPLE {
                return None;
            }
            let (cos_k, sin_abs) = multiple_angle(k.abs(), s, c, pool);
            let sin_k = if k < 0 {
                pool.mul(vec![pool.integer(-1_i32), sin_abs])
            } else {
                sin_abs
            };
            Some(match name.as_str() {
                "sin" => sin_k,
                "cos" => cos_k,
                _ => pool.mul(vec![sin_k, pool.pow(cos_k, pool.integer(-1_i32))]),
            })
        }
        ExprData::Add(args) => Some(
            pool.add(
                args.iter()
                    .map(|&a| rewrite(a, var, g, s, c, pool))
                    .collect::<Option<Vec<_>>>()?,
            ),
        ),
        ExprData::Mul(args) => Some(
            pool.mul(
                args.iter()
                    .map(|&a| rewrite(a, var, g, s, c, pool))
                    .collect::<Option<Vec<_>>>()?,
            ),
        ),
        ExprData::Pow { base, exp } => Some(pool.pow(rewrite(base, var, g, s, c, pool)?, exp)),
        _ => None,
    }
}

/// `true` when `d` is a product of powers of `s`, `c` and constants.
fn is_monomial(d: ExprId, s: ExprId, c: ExprId, pool: &ExprPool) -> bool {
    if is_free_of(d, s, pool) && is_free_of(d, c, pool) {
        return true;
    }
    match pool.get(d) {
        _ if d == s || d == c => true,
        ExprData::Mul(args) => args.iter().all(|&a| is_monomial(a, s, c, pool)),
        ExprData::Pow { base, .. } => base == s || base == c,
        _ => false,
    }
}

/// See the module docs.  `Ok(None)` is a decline.
pub(crate) fn try_trig_angle_reduction(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Option<DerivedExpr<ExprId>>, IntegrationError> {
    let mut slopes = Vec::new();
    if collect(expr, var, pool, &mut slopes).is_none() || slopes.is_empty() {
        return Ok(None);
    }
    let Some(_guard) = ActiveGuard::enter() else {
        return Ok(None);
    };
    crate::budget::check()?;
    // θ = g·x with g = gcd of the slopes (as rationals).
    let mut den = rug::Integer::from(1);
    for a in &slopes {
        den = den.lcm(a.denom());
    }
    let mut num = rug::Integer::from(0);
    for a in &slopes {
        num = num.gcd(&(rug::Integer::from(a.numer() * &den) / a.denom()));
    }
    let g = rug::Rational::from((num, den));
    let s = pool.symbol("__trig_angle_s", Domain::Real);
    let c = pool.symbol("__trig_angle_c", Domain::Real);
    let Some(r) = rewrite(expr, var, &g, s, c, pool) else {
        return Ok(None);
    };
    let Ok((n, d)) = crate::poly::cancel::together_parts(r, vec![s, c], pool) else {
        return Ok(None);
    };
    let body = if is_monomial(d, s, c, pool) {
        simplify_expanded(pool.mul(vec![n, pool.pow(d, pool.integer(-1_i32))]), pool).value
    } else {
        pool.mul(vec![n, pool.pow(d, pool.integer(-1_i32))])
    };
    let y = pool.symbol("__trig_angle_y", Domain::Real);
    let mut to_y = HashMap::new();
    to_y.insert(s, pool.func("sin", vec![y]));
    to_y.insert(c, pool.func("cos", vec![y]));
    let g_expr = pool.rational(g.numer().clone(), g.denom().clone());
    let h = simplify(
        pool.mul(vec![
            crate::kernel::subs(body, &to_y, pool),
            pool.pow(g_expr, pool.integer(-1_i32)),
        ]),
        pool,
    )
    .value;
    // Nothing to gain when the reduction reproduces the integrand.
    let mut same = HashMap::new();
    same.insert(y, var);
    if g == 1
        && simplify(crate::kernel::subs(h, &same, pool), pool).value == simplify(expr, pool).value
    {
        return Ok(None);
    }
    let inner = match integrate(h, y, pool) {
        Ok(v) => v.value,
        Err(e) if e.is_budget() => return Err(e),
        Err(_) => return Ok(None),
    };
    let mut back = HashMap::new();
    back.insert(y, pool.mul(vec![g_expr, var]));
    let f = simplify(crate::kernel::subs(inner, &back, pool), pool).value;
    if verify_antiderivative_status(f, expr, var, pool).is_none() {
        return Ok(None);
    }
    let mut log = DerivationLog::new();
    // The inner half-angle logarithms (`log(tan(x/8) − 1)`) are negative on
    // part of the real line: give the real branch.
    let f = super::real_log::realify_logs(f, expr, var, pool);
    log.push(RewriteStep::simple("int_trig_single_angle", expr, f));
    Ok(Some(DerivedExpr::with_log(f, log)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solves(src: &str) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        let e = crate::parse::parse(src, &pool, &mut syms).unwrap();
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
    fn mixed_angles_and_hidden_sums() {
        solves("cos(x)/(sin(x)*(1 + tan(x/2)))");
        solves("cos(x)/sin(x)*(tan(x) - 1/cos(x))");
        solves("sin(2*x)/(1 + cos(x))");
    }

    #[test]
    fn de_moivre() {
        let pool = ExprPool::new();
        let s = pool.symbol("s", Domain::Real);
        let c = pool.symbol("c", Domain::Real);
        let (c3, s3) = multiple_angle(3, s, c, &pool);
        // cos 3θ = c³ − 3cs², sin 3θ = 3c²s − s³ at θ = 0.4.
        let th = 0.4_f64;
        let env = HashMap::from([(s, th.sin()), (c, th.cos())]);
        let v = |e| crate::jit::eval_interp(e, &env, &pool).unwrap();
        assert!((v(c3) - (3.0 * th).cos()).abs() < 1e-14);
        assert!((v(s3) - (3.0 * th).sin()).abs() < 1e-14);
    }
}
