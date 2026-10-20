//! Derivatives of the non-smooth functions `abs`, `sign`, `floor`, `ceil`,
//! `round`, `min`, `max` (and the distributional `heaviside`).
//!
//! # Convention (the one used by `diff`, `diff_forward` and `symbolic_grad`)
//!
//! The derivative returned is the **almost-everywhere derivative**, with the
//! value *at a kink* chosen as the midpoint of the one-sided derivatives, and
//! the kink set is **stated as a side condition** on the derivation log
//! (`DerivedResult.side_conditions` in Python) — never an unstated zero:
//!
//! | function          | derivative returned                                   | side condition (derivative holds where) |
//! |-------------------|-------------------------------------------------------|-----------------------------------------|
//! | `abs(g)`          | `sign(g)·g'`  (`0` at `g = 0`)                        | `g ≠ 0`                                 |
//! | `sign(g)`         | `0`                                                   | `g ≠ 0`                                 |
//! | `floor(g)`/`ceil(g)` | `0`                                                | `g − floor(g) ≠ 0` (g not an integer)   |
//! | `round(g)`        | `0`                                                   | `g + 1/2 − floor(g + 1/2) ≠ 0`          |
//! | `min(a, b)`       | `Piecewise((a<b, a'), (a>b, b'); (a'+b')/2)`          | `a − b ≠ 0`                             |
//! | `max(a, b)`       | `Piecewise((a>b, a'), (a<b, b'); (a'+b')/2)`          | `a − b ≠ 0`                             |
//! | `heaviside(g)`    | `diracdelta(g)·g'` (distributional)                   | —                                       |
//!
//! `min`/`max` of three or more arguments are folded pairwise, so the rule is
//! the two-argument one applied to `min(a₁, min(a₂, …))`.
//!
//! Reverse mode takes its local partials from the forward rules above, so it
//! returns the same almost-everywhere gradient; the conditions are available
//! from [`nonsmooth_side_conditions`] on the differentiated expression.

use crate::deriv::log::SideCondition;
use crate::kernel::{ExprData, ExprId, ExprPool, IdSet};

/// Is `name` one of the functions whose derivative is the a.e. rule above?
pub(crate) fn is_nonsmooth_name(name: &str) -> bool {
    matches!(
        name,
        "abs" | "sign" | "floor" | "ceil" | "round" | "min" | "max"
    )
}

fn children(data: &ExprData) -> Vec<ExprId> {
    match data {
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => vec![],
        ExprData::Add(a) | ExprData::Mul(a) => a.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => args.clone(),
        ExprData::Piecewise { branches, default } => {
            let mut v: Vec<ExprId> = branches.iter().flat_map(|&(c, x)| [c, x]).collect();
            v.push(*default);
            v
        }
        ExprData::Forall { body, .. } | ExprData::Exists { body, .. } => vec![*body],
        ExprData::BigO(a) => vec![*a],
        ExprData::RootSum { poly, body, .. } => vec![*poly, *body],
    }
}

fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])])
}

/// `g − floor(g)`: zero exactly when `g` is an integer.
fn frac_part(g: ExprId, pool: &ExprPool) -> ExprId {
    sub(g, pool.func("floor", vec![g]), pool)
}

/// Points at which the almost-everywhere derivative of `expr` (with respect to
/// `var`) is not the derivative: one [`SideCondition::NonZero`] per distinct
/// non-smooth node whose argument depends on `var`.
///
/// Empty when `expr` contains no `abs`/`sign`/`floor`/`ceil`/`round`/`min`/`max`
/// over `var`.
pub fn nonsmooth_side_conditions(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<SideCondition> {
    let mut out: Vec<SideCondition> = Vec::new();
    let mut seen = IdSet::default();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        let data = pool.get(e);
        if let ExprData::Func { name, args } = &data {
            if is_nonsmooth_name(name) && args.iter().any(|&a| mentions(a, var, pool)) {
                let conds: Vec<ExprId> = match name.as_str() {
                    "abs" | "sign" => vec![args[0]],
                    "floor" | "ceil" => vec![frac_part(args[0], pool)],
                    "round" => {
                        let shifted = pool.add(vec![args[0], pool.rational(1, 2)]);
                        vec![frac_part(shifted, pool)]
                    }
                    _ => {
                        // min / max: every pair of arguments must differ.
                        let mut v = Vec::new();
                        for i in 0..args.len() {
                            for j in (i + 1)..args.len() {
                                v.push(sub(args[i], args[j], pool));
                            }
                        }
                        v
                    }
                };
                for c in conds {
                    let c = crate::simplify::engine::simplify(c, pool).value;
                    let sc = SideCondition::NonZero(c);
                    if !out.contains(&sc) {
                        out.push(sc);
                    }
                }
            }
        }
        stack.extend(children(&data));
    }
    out
}

fn mentions(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    crate::kernel::subs::mentions_var(e, var, pool)
}

fn half(pool: &ExprPool) -> ExprId {
    pool.rational(1, 2)
}

/// `d/dx abs(g) = sign(g)·g'`.
pub(crate) fn abs_tangent(g: ExprId, dg: ExprId, pool: &ExprPool) -> ExprId {
    let s = pool.func("sign", vec![g]);
    if dg == pool.integer(1_i32) {
        s
    } else {
        pool.mul(vec![s, dg])
    }
}

/// `d/dx min(a, b)` (`is_min`) or `max(a, b)` given the arguments' derivatives.
pub(crate) fn minmax_tangent2(
    is_min: bool,
    a: ExprId,
    b: ExprId,
    da: ExprId,
    db: ExprId,
    pool: &ExprPool,
) -> ExprId {
    if da == db {
        return da;
    }
    let (first, second) = if is_min {
        (pool.pred_lt(a, b), pool.pred_gt(a, b))
    } else {
        (pool.pred_gt(a, b), pool.pred_lt(a, b))
    };
    let mid = pool.mul(vec![half(pool), pool.add(vec![da, db])]);
    pool.piecewise(vec![(first, da), (second, db)], mid)
}

/// `d/dx min(a₁, …, aₙ)` / `max`, folded right-to-left.
pub(crate) fn minmax_tangent(
    is_min: bool,
    args: &[ExprId],
    dargs: &[ExprId],
    pool: &ExprPool,
) -> ExprId {
    let name = if is_min { "min" } else { "max" };
    let n = args.len();
    debug_assert!(n >= 2 && dargs.len() == n);
    // acc_val / acc_der = min(a_{k..}) and its derivative.
    let mut acc_val = args[n - 1];
    let mut acc_der = dargs[n - 1];
    for k in (0..n - 1).rev() {
        let t = minmax_tangent2(is_min, args[k], acc_val, dargs[k], acc_der, pool);
        acc_val = pool.func(name, vec![args[k], acc_val]);
        acc_der = t;
    }
    acc_der
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{diff, diff_forward, try_grad};
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn abs_derivative_is_sign_with_side_condition() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("abs", vec![x]);
        let d = diff(e, x, &pool).unwrap();
        assert_eq!(d.value, pool.func("sign", vec![x]));
        let steps = d.log.steps();
        assert!(steps
            .iter()
            .any(|s| s.side_conditions.contains(&SideCondition::NonZero(x))));
        let f = diff_forward(e, x, &pool).unwrap();
        assert_eq!(f.value, d.value);
        let g = try_grad(e, &[x], &pool).unwrap();
        assert_eq!(g[0], d.value);
    }

    #[test]
    fn floor_family_is_zero_with_condition() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for f in ["floor", "ceil", "round", "sign"] {
            let e = pool.func(f, vec![pool.mul(vec![pool.integer(3_i32), x])]);
            let d = diff(e, x, &pool).unwrap();
            assert_eq!(d.value, pool.integer(0_i32), "{f}");
            assert!(
                d.log.steps().iter().any(|s| !s.side_conditions.is_empty()),
                "{f}: side condition missing"
            );
            // A constant argument is smooth: no condition.
            let c = pool.func(f, vec![pool.integer(3_i32)]);
            let dc = diff(c, x, &pool).unwrap();
            assert!(dc.log.steps().iter().all(|s| s.side_conditions.is_empty()));
        }
    }

    #[test]
    fn min_max_are_piecewise() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        for name in ["min", "max"] {
            let e = pool.func(name, vec![x, x2]);
            let d = diff(e, x, &pool).unwrap();
            assert!(
                matches!(pool.get(d.value), ExprData::Piecewise { .. }),
                "{name}: {}",
                pool.display(d.value)
            );
            let f = diff_forward(e, x, &pool).unwrap();
            assert!(matches!(pool.get(f.value), ExprData::Piecewise { .. }));
        }
    }
}
