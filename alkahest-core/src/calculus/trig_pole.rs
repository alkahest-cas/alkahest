//! One-sided values of an expression at a pole of `tan`, `cot`, `sec` or `csc`.
//!
//! Substitution-based antiderivatives carry these functions: the Weierstrass
//! substitution leaves a `tan(x/2)`, the `t = tan x` reduction a `tan x`.  Plain
//! substitution of a bound at which such an argument lands on a pole produces
//! `tan(π/2)` — not a number, but a `Func` node the `f64` interpreter happily
//! evaluates to `1.6e16` (the tangent of the nearest double), so the closed form
//! *looks* finite and is ill-defined.  `∫_0^{π/2} dx/√(1 − sin²x/4)` came back
//! as an expression in `tan(π/2)` and evaluated 3e-9 away from `K(1/4)`.
//!
//! [`contains_trig_pole`] recognises the situation exactly (the argument is a
//! rational multiple of `π` sitting on a pole), and [`one_sided_value`] replaces
//! the substitution there by a one-sided limit, computed structurally: the pole
//! becomes `±∞` with the sign fixed by the direction of approach, and that
//! infinity is propagated through sums, products, powers and `atan` only where
//! the result is determinate.  Anything indeterminate (`∞ − ∞`, `0·∞`, `tan(∞)`,
//! a head it does not model) gives `None`, and the caller must refuse.

use crate::kernel::{subs, Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::collections::HashMap;

/// Which side of the point the variable approaches from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Side {
    /// `var → point⁻` (from below).
    Below,
    /// `var → point⁺` (from above).
    Above,
}

impl Side {
    fn sign(self) -> i32 {
        match self {
            Side::Below => -1,
            Side::Above => 1,
        }
    }
}

/// `q` with `expr = q·π` exactly, when `expr` is a rational multiple of `π`.
pub(crate) fn pi_multiple(expr: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(expr) {
        ExprData::Integer(n) if n.0 == 0 => return Some(rug::Rational::new()),
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => return None,
        _ => {}
    }
    let pi = pool.symbol("pi", Domain::Real);
    let q = simplify(
        pool.mul(vec![expr, pool.pow(pi, pool.integer(-1_i32))]),
        pool,
    )
    .value;
    match pool.get(q) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

/// When `name(arg)` is a pole, the integer `k` locating it: `arg = π/2 + kπ` for
/// `tan`/`sec`, `arg = kπ` for `cot`/`csc`.
fn pole_index(name: &str, arg: ExprId, pool: &ExprPool) -> Option<i64> {
    if !matches!(name, "tan" | "cot" | "sec" | "csc") {
        return None;
    }
    let q = pi_multiple(arg, pool)?;
    match name {
        "tan" | "sec" => {
            // q = 1/2 + k  ⇔  2q − 1 = 2k.
            let t = rug::Rational::from(&q * 2u32) - 1u32;
            if t.is_integer() && t.numer().is_even() {
                (rug::Integer::from(t.numer() / 2u32)).to_i64()
            } else {
                None
            }
        }
        _ => {
            if q.is_integer() {
                q.numer().to_i64()
            } else {
                None
            }
        }
    }
}

/// True when the var-free `expr` contains `tan`, `cot`, `sec` or `csc` applied
/// exactly at one of its poles (`tan(π/2)`, `cot(0)`, `sec(3π/2)`, …).
pub(crate) fn contains_trig_pole(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            (args.len() == 1 && pole_index(&name, args[0], pool).is_some())
                || args.iter().any(|&a| contains_trig_pole(a, pool))
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&a| contains_trig_pole(a, pool)),
        ExprData::Pow { base, exp } => {
            contains_trig_pole(base, pool) || contains_trig_pole(exp, pool)
        }
        _ => false,
    }
}

/// A value in the extended reals, as far as this module tracks it.
#[derive(Clone, Copy)]
enum Ext {
    Fin(ExprId),
    PosInf,
    NegInf,
}

impl Ext {
    fn inf(sign: i32) -> Ext {
        if sign > 0 {
            Ext::PosInf
        } else {
            Ext::NegInf
        }
    }
}

fn numeric(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    crate::jit::eval_interp(expr, &HashMap::new(), pool).filter(|v| v.is_finite())
}

fn mentions(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    crate::kernel::subs::mentions_var(expr, var, pool)
}

/// `lim_{var → point^side} expr`, for an `expr` whose plain substitution at
/// `point` hits a pole of `tan`/`cot`/`sec`/`csc` (see [`contains_trig_pole`]).
///
/// Returns the finite limit, simplified, or `None` when the limit is infinite
/// or could not be established — never a value containing the pole.
pub(crate) fn one_sided_value(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: Side,
    pool: &ExprPool,
) -> Option<ExprId> {
    match eval(expr, var, point, side, pool, 0)? {
        Ext::Fin(v) => {
            let v = simplify(v, pool).value;
            (!contains_trig_pole(v, pool)).then_some(v)
        }
        _ => None,
    }
}

const MAX_DEPTH: u32 = 64;

fn eval(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: Side,
    pool: &ExprPool,
    depth: u32,
) -> Option<Ext> {
    if depth > MAX_DEPTH {
        return None;
    }
    if !mentions(expr, var, pool) {
        return (!contains_trig_pole(expr, pool)).then_some(Ext::Fin(expr));
    }
    // Plain substitution is the limit wherever it does not land on a pole.
    let mut map = HashMap::new();
    map.insert(var, point);
    let plain = subs(expr, &map, pool);
    if !contains_trig_pole(plain, pool) {
        return Some(Ext::Fin(plain));
    }
    let rec = |e: ExprId| eval(e, var, point, side, pool, depth + 1);
    match pool.get(expr) {
        ExprData::Add(xs) => {
            let mut fin = Vec::with_capacity(xs.len());
            let mut inf: Option<i32> = None;
            for &a in &xs {
                let s = match rec(a)? {
                    Ext::Fin(v) => {
                        fin.push(v);
                        continue;
                    }
                    Ext::PosInf => 1,
                    Ext::NegInf => -1,
                };
                if inf.is_some_and(|t| t != s) {
                    return None; // ∞ − ∞
                }
                inf = Some(s);
            }
            Some(match inf {
                Some(s) => Ext::inf(s),
                None => Ext::Fin(pool.add(fin)),
            })
        }
        ExprData::Mul(xs) => {
            let mut fin = Vec::with_capacity(xs.len());
            let mut inf_sign: Option<i32> = None;
            for &a in &xs {
                match rec(a)? {
                    Ext::Fin(v) => fin.push(v),
                    Ext::PosInf => inf_sign = Some(inf_sign.unwrap_or(1)),
                    Ext::NegInf => inf_sign = Some(-inf_sign.unwrap_or(1)),
                }
            }
            match inf_sign {
                None => Some(Ext::Fin(pool.mul(fin))),
                Some(s) => {
                    // ∞ times a finite factor: determinate only for a numeric,
                    // non-zero factor.
                    let c = if fin.is_empty() {
                        1.0
                    } else {
                        numeric(pool.mul(fin), pool)?
                    };
                    if c == 0.0 {
                        return None;
                    }
                    Some(Ext::inf(if c > 0.0 { s } else { -s }))
                }
            }
        }
        ExprData::Pow { base, exp } => {
            if mentions(exp, var, pool) {
                return None;
            }
            match rec(base)? {
                Ext::Fin(b) => Some(Ext::Fin(pool.pow(b, exp))),
                inf => {
                    let e = numeric(exp, pool)?;
                    if e < 0.0 {
                        Some(Ext::Fin(pool.integer(0_i32)))
                    } else if e == 0.0 {
                        None
                    } else if matches!(inf, Ext::PosInf) {
                        Some(Ext::PosInf)
                    } else if e.fract() == 0.0 {
                        Some(Ext::inf(if (e as i64) % 2 == 0 { 1 } else { -1 }))
                    } else {
                        None
                    }
                }
            }
        }
        ExprData::Func { name, args } => {
            if args.len() == 1 && matches!(name.as_str(), "tan" | "cot" | "sec" | "csc") {
                let Ext::Fin(a) = rec(args[0])? else {
                    return None;
                };
                let Some(k) = pole_index(&name, a, pool) else {
                    return Some(Ext::Fin(pool.func(name.clone(), vec![a])));
                };
                // Direction in which the argument crosses the pole.
                let d = crate::diff::diff(args[0], var, pool).ok()?.value;
                let mut m = HashMap::new();
                m.insert(var, point);
                let slope = numeric(simplify(subs(d, &m, pool), pool).value, pool)?;
                if slope == 0.0 {
                    return None;
                }
                let eps = side.sign() * if slope > 0.0 { 1 } else { -1 };
                let parity = if k.rem_euclid(2) == 0 { 1 } else { -1 };
                // Near the pole, with arg = pole + ε:
                //   tan ≈ −1/ε,  cot ≈ 1/ε,  sec ≈ −(−1)^k/ε,  csc ≈ (−1)^k/ε.
                let s = match name.as_str() {
                    "tan" => -eps,
                    "cot" => eps,
                    "sec" => -parity * eps,
                    _ => parity * eps,
                };
                return Some(Ext::inf(s));
            }
            let mut vals = Vec::with_capacity(args.len());
            for &a in &args {
                vals.push(rec(a)?);
            }
            if name == "atan" && vals.len() == 1 {
                let half_pi = pool.mul(vec![pool.rational(1, 2), pool.symbol("pi", Domain::Real)]);
                return Some(match vals[0] {
                    Ext::Fin(v) => Ext::Fin(pool.func("atan", vec![v])),
                    Ext::PosInf => Ext::Fin(half_pi),
                    Ext::NegInf => Ext::Fin(pool.mul(vec![pool.integer(-1_i32), half_pi])),
                });
            }
            let mut fin = Vec::with_capacity(vals.len());
            for v in vals {
                match v {
                    Ext::Fin(e) => fin.push(e),
                    _ => return None,
                }
            }
            Some(Ext::Fin(pool.func(name.clone(), fin)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let pi = pool.symbol("pi", Domain::Real);
        (pool, x, pi)
    }

    fn half(pool: &ExprPool, e: ExprId) -> ExprId {
        pool.mul(vec![pool.rational(1, 2), e])
    }

    fn val(e: ExprId, pool: &ExprPool) -> f64 {
        numeric(e, pool).expect("numeric value")
    }

    #[test]
    fn detects_poles_exactly() {
        let (pool, _x, pi) = setup();
        let tan_half_pi = pool.func("tan", vec![half(&pool, pi)]);
        assert!(contains_trig_pole(tan_half_pi, &pool));
        let three_half = pool.mul(vec![pool.rational(3, 2), pi]);
        assert!(contains_trig_pole(
            pool.func("sec", vec![three_half]),
            &pool
        ));
        assert!(contains_trig_pole(
            pool.func("cot", vec![pool.integer(0_i32)]),
            &pool
        ));
        assert!(contains_trig_pole(pool.func("csc", vec![pi]), &pool));
        // Not poles.
        assert!(!contains_trig_pole(pool.func("tan", vec![pi]), &pool));
        assert!(!contains_trig_pole(
            pool.func("cot", vec![half(&pool, pi)]),
            &pool
        ));
        assert!(!contains_trig_pole(
            pool.func("tan", vec![pool.integer(1_i32)]),
            &pool
        ));
    }

    #[test]
    fn atan_of_tan_takes_the_side_of_approach() {
        // lim_{x→π/2∓} atan(2·tan x) = ±π/2.
        let (pool, x, pi) = setup();
        let f = pool.func(
            "atan",
            vec![pool.mul(vec![pool.integer(2_i32), pool.func("tan", vec![x])])],
        );
        let p = half(&pool, pi);
        let below = one_sided_value(f, x, p, Side::Below, &pool).unwrap();
        let above = one_sided_value(f, x, p, Side::Above, &pool).unwrap();
        assert!((val(below, &pool) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert!((val(above, &pool) + std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn a_negative_slope_flips_the_side() {
        // atan(tan(−x)) as x → −π/2 from below: the argument −x approaches π/2
        // from above, so tan → −∞.
        let (pool, x, pi) = setup();
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let f = pool.func("atan", vec![pool.func("tan", vec![neg_x])]);
        let p = pool.mul(vec![pool.integer(-1_i32), half(&pool, pi)]);
        let v = one_sided_value(f, x, p, Side::Below, &pool).unwrap();
        assert!((val(v, &pool) + std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn indeterminate_and_infinite_limits_are_refused() {
        let (pool, x, pi) = setup();
        let p = half(&pool, pi);
        let t = pool.func("tan", vec![x]);
        // tan x itself diverges.
        assert!(one_sided_value(t, x, p, Side::Below, &pool).is_none());
        // tan x − tan(2x)/… style ∞ − ∞: tan x + tan(−x) is +∞ + −∞.
        let neg = pool.func("tan", vec![pool.mul(vec![pool.integer(-1_i32), x])]);
        let diff = pool.add(vec![t, neg]);
        assert!(one_sided_value(diff, x, p, Side::Below, &pool).is_none());
        // 0·∞ is indeterminate.
        let zero_inf = pool.mul(vec![
            pool.add(vec![x, pool.mul(vec![pool.integer(-1_i32), p])]),
            t,
        ]);
        assert!(one_sided_value(zero_inf, x, p, Side::Below, &pool).is_none());
        // 1/tan x → 0.
        let inv = pool.pow(t, pool.integer(-1_i32));
        let v = one_sided_value(inv, x, p, Side::Below, &pool).unwrap();
        assert_eq!(val(v, &pool), 0.0);
    }
}
