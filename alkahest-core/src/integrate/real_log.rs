//! Real-valued logarithms in antiderivatives built by substitution.
//!
//! A substitution route integrates in `u` and back-substitutes, and the
//! rational integrator's logarithms come out on the principal branch:
//! `∫ sin x/(sin²x + 1) dx` by `u = cos x` gives
//! `log(2cos x − 2√2)/(2√2) − …`, whose argument is negative for **every**
//! real `x`.  Its derivative is still the integrand (so the gate accepts it),
//! but the value is complex — off by the constant `iπ/(2√2)` — on the whole
//! real line, which is a defect for anyone evaluating it.
//!
//! [`realify_logs`] rewrites each `log(a)` whose argument is negative somewhere
//! on the integrand's real domain: to `log(−a)` when `a` is never positive
//! there (smooth, and the same derivative), else to `log(|a|)` (the same
//! derivative wherever `a ≠ 0`) — or to `½·log(a²)` when the integrand has a
//! complex branch on the real line, where `log|a|` would not differentiate
//! back to it.  The rewritten candidate is re-gated against the integrand and
//! must evaluate to a finite real number at every sample where the integrand
//! does; otherwise the original is kept.
//!
//! [`realify_answer`] runs this as the last step of every outermost
//! `integrate` call, whichever route produced the answer.

use std::collections::HashMap;

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::engine::{is_free_of, verify_antiderivative_status_parametric};

/// Real sample points: dense enough to cross a period of the trigonometric
/// integrands these routes produce, and irrational-looking to dodge poles.
fn samples() -> Vec<f64> {
    (0..64).map(|i| -7.31 + 0.2317 * i as f64).collect()
}

fn eval_at(e: ExprId, var: ExprId, at: f64, pool: &ExprPool) -> Option<f64> {
    let mut env = HashMap::new();
    env.insert(var, at);
    crate::jit::eval_interp(e, &env, pool).filter(|v| v.is_finite())
}

fn collect_logs(e: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    if is_free_of(e, var, pool) {
        return;
    }
    match pool.get(e) {
        ExprData::Func { name, args } => {
            if name == "log" && args.len() == 1 && !out.contains(&e) {
                out.push(e);
            }
            for a in args {
                collect_logs(a, var, pool, out);
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect_logs(a, var, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_logs(base, var, pool, out);
            collect_logs(exp, var, pool, out);
        }
        _ => {}
    }
}

/// `true` when `f` evaluates to a finite real number at every sample where
/// `integrand` does (and at two at least).
fn real_where_integrand_is(f: ExprId, integrand: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let mut checked = 0;
    for at in samples() {
        if eval_at(integrand, var, at, pool).is_none() {
            continue;
        }
        if eval_at(f, var, at, pool).is_none() {
            return false;
        }
        checked += 1;
    }
    checked >= 2
}

/// Derivation-log rule name of the final real-log pass in `integrate`.
pub(crate) const REAL_LOG_RULE: &str = "int_real_valued_logs";

thread_local! {
    /// Depth of [`PrincipalLogs`] scopes on this thread.
    static PRINCIPAL_LOGS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// While alive, the outermost `integrate` keeps principal-branch logarithms
/// (skips [`realify_answer`]).  For callers whose antiderivatives only ever
/// feed an expression with a free constant that absorbs the difference —
/// `dsolve`'s `exp(∫p dx)` is `x`, not `|x|`, for `p = 1/x`, and `C·x` is the
/// same family as `C·|x|`.
pub(crate) struct PrincipalLogs;

impl PrincipalLogs {
    pub(crate) fn enter() -> Self {
        PRINCIPAL_LOGS.with(|d| d.set(d.get() + 1));
        PrincipalLogs
    }
}

impl Drop for PrincipalLogs {
    fn drop(&mut self) {
        PRINCIPAL_LOGS.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// The final pass of the outermost `integrate`: [`realify_logs`] on any
/// answer that carries a `log` of the variable, `None` when nothing changed.
/// Answers without one (most of them) cost a single tree walk.
pub(crate) fn realify_answer(
    f: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if PRINCIPAL_LOGS.with(|d| d.get()) > 0 {
        return None;
    }
    let mut logs = Vec::new();
    collect_logs(f, var, pool, &mut logs);
    if logs.is_empty() {
        return None;
    }
    let g = realify_logs(f, integrand, var, pool);
    (g != f).then_some(g)
}

/// See the module docs.  Returns `f` unchanged when nothing needs rewriting
/// or the rewrite cannot be confirmed.
pub(crate) fn realify_logs(f: ExprId, integrand: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    if real_where_integrand_is(f, integrand, var, pool) {
        return f;
    }
    let mut logs = Vec::new();
    collect_logs(f, var, pool, &mut logs);
    if logs.is_empty() {
        return f;
    }
    // `log|a|` differentiates to `a′/a` only where `a` is real: on a complex
    // branch of the integrand it is not an antiderivative where `log(a)` was,
    // and a symbolic `d/dx` (which simplifies `sign(a)/|a|` to `1/a`) cannot
    // see that.  There the analytic `½·log(a²)` is used instead: the same
    // real value on the real domain, derivative `a′/a` everywhere.
    let complex_branch = super::branch_check::has_complex_branch(integrand, var, pool);
    let mut map = HashMap::new();
    for l in logs {
        let ExprData::Func { args, .. } = pool.get(l) else {
            continue;
        };
        let a = args[0];
        let (mut neg, mut pos) = (false, false);
        for at in samples() {
            if eval_at(integrand, var, at, pool).is_none() {
                continue;
            }
            match eval_at(a, var, at, pool) {
                Some(v) if v < 0.0 => neg = true,
                Some(v) if v > 0.0 => pos = true,
                _ => {}
            }
        }
        if !neg {
            continue;
        }
        let replacement = if pos && complex_branch {
            pool.mul(vec![
                pool.rational(1, 2),
                pool.func("log", vec![pool.pow(a, pool.integer(2_i32))]),
            ])
        } else if pos {
            pool.func("log", vec![pool.func("abs", vec![a])])
        } else {
            let minus_a = simplify(pool.mul(vec![pool.integer(-1_i32), a]), pool).value;
            pool.func("log", vec![minus_a])
        };
        map.insert(l, replacement);
    }
    if map.is_empty() {
        return f;
    }
    let g = simplify(crate::kernel::subs(f, &map, pool), pool).value;
    if real_where_integrand_is(g, integrand, var, pool)
        && verify_antiderivative_status_parametric(g, integrand, var, pool).is_some()
    {
        g
    } else {
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// `∫ sin x/(sin²x + 1) dx` by `u = cos x` used to carry
    /// `log(2cos x − 2√2)`, negative on all of ℝ: the answer must now be real
    /// at every real point, and still differentiate back to the integrand.
    #[test]
    fn bioche_answer_is_real_on_the_line() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        let f = crate::parse::parse("sin(x)/(sin(x)^2 + 1)", &pool, &mut syms).unwrap();
        let r = crate::integrate::engine::integrate(f, x, &pool)
            .unwrap()
            .value;
        for at in samples() {
            assert!(
                eval_at(r, x, at, &pool).is_some(),
                "F({at}) not real: {}",
                pool.display(r)
            );
        }
        assert!(verify_antiderivative_status_parametric(r, f, x, &pool).is_some());
    }
}
