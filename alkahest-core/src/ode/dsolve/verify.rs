//! Substitution-based verification gate for [`super::dsolve`].
//!
//! Given a candidate solution `y(x)`, build the residual of the original
//! equation with `y`, `y'`, `y''`, … replaced by the candidate and its
//! derivatives, then require the residual to be the symbolic zero, or — when
//! `simplify` cannot close it — numerically `≈ 0` at several `x` samples over
//! several random assignments of the integration constants.

use super::{ddx, simp, subs1, DsolveError, OdeInput};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;

/// Verify a candidate `y(x)` against `input.equation = 0`.
///
/// Returns `Ok(())` if the residual is symbolically or numerically zero.
pub(crate) fn residual_is_zero(
    input: &OdeInput,
    y_of_x: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveError> {
    // Build the residual: substitute y → y(x), y^(k) → d^k/dx^k y(x).
    let mut residual = input.equation;
    // highest derivative first so we don't clobber lower-order symbols that
    // also appear inside higher-derivative expressions (they are distinct
    // symbols, so order does not actually matter, but do y then derivs).
    residual = subs1(residual, input.y, y_of_x, pool);

    let mut cur = y_of_x;
    for &dsym in &input.derivs {
        cur = ddx(cur, input.x, pool)?;
        residual = subs1(residual, dsym, cur, pool);
    }

    let residual = simp(residual, pool);

    // Symbolic zero?  Try both the expanded and the plain (non-expanding)
    // normal forms — expansion flattens polynomial cancellations, while plain
    // simplify is better at collapsing products such as `√D·√D⁻¹ → 1`.
    if is_symbolic_zero(residual, pool) || is_symbolic_zero(super::simp_plain(residual, pool), pool)
    {
        return Ok(());
    }

    // Numeric fallback: sample x over several constant assignments.
    if numeric_zero(residual, input.x, constants, pool) {
        return Ok(());
    }

    Err(DsolveError::VerificationFailed(format!(
        "residual did not reduce to zero: {}",
        pool.display(residual)
    )))
}

fn is_symbolic_zero(expr: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(expr), ExprData::Integer(n) if n.0 == 0)
}

/// Numerically check residual ≈ 0 at several `x` over random constants.
fn numeric_zero(residual: ExprId, x: ExprId, constants: &[ExprId], pool: &ExprPool) -> bool {
    // Deterministic pseudo-random constant assignments (no rng dependency).
    // Constants are kept positive and reasonably large so that radicands such as
    // `sqrt(4·C − 3x²)` arising from quadratic-implicit solutions stay real over
    // the (small) x-sample range; samples that still hit a pole/branch-cut are
    // skipped rather than failing.
    let const_sets: [&[f64]; 3] = [
        &[5.7, 4.3, 6.4, 5.1, 4.9],
        &[8.5, 7.8, 6.6, 9.2, 7.1],
        &[12.3, 10.0, 11.7, 10.5, 9.4],
    ];
    let x_samples = [0.11, 0.27, 0.43, 0.61, 0.79];

    let mut checked = 0usize;
    let mut ok = 0usize;
    for cs in const_sets {
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        for (i, &c) in constants.iter().enumerate() {
            env.insert(c, cs[i % cs.len()]);
        }
        for &xv in &x_samples {
            env.insert(x, xv);
            match eval(residual, &env, pool) {
                Some(v) if v.is_finite() => {
                    checked += 1;
                    if v.abs() < 1e-6 {
                        ok += 1;
                    }
                }
                Some(_) => { /* non-finite (pole at this sample) — skip */ }
                None => return false, // unknown construct → cannot certify numerically
            }
        }
    }
    // Require a healthy number of finite samples and all of them ≈ 0.
    checked >= 6 && ok == checked
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
        _ => return None,
    })
}
