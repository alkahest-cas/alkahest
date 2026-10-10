//! Exact Legendre forms for trigonometric elliptic integrands.
//!
//! `∫ (A + B·sin²u)^{∓1/2} dx` is, by definition of the incomplete elliptic
//! integrals (parameter convention `m = k²`, as everywhere in Alkahest),
//!
//! ```text
//!   ∫ dx / √(A + B·sin²u) = A^{-1/2} · F(u | m) / u′,
//!   ∫ √(A + B·sin²u) dx   = A^{1/2}  · E(u | m) / u′,      m = −B/A,  A > 0,
//! ```
//!
//! for `u` linear in `x`.  Before this module the integrator only reached these
//! integrands through the `t = tan x` generator substitution, whose elliptic
//! reduction computes its constants in `f64` and lifts them to exact-looking
//! `…/2⁵²` rationals: `∫_0^{π/2} dx/√(1 − sin²x/4)` came back as an
//! expression in `4503599618403511/4503599627370496` and `tan(π/2)`, 3e-9 away
//! from `K(1/4)`.  The forms here are exact, so the answer is `K(1/4)`.
//!
//! Also covered, by the identities they reduce through:
//!
//! * `A + B·cos²u` — `cos²u = sin²(u − π/2)`, giving `F(u − π/2 | m)`, so the
//!   complete integral over `[0, π/2]` is again `K(m)`;
//! * a mix of `sin²u` and `cos²u` — `cos²u = 1 − sin²u`;
//! * `A + B·cos u` and `A + B·sin u` — the half-angle identity
//!   `cos u = 1 − 2·sin²(u/2)`, giving `2·(A+B)^{∓1/2}·F(u/2 | 2B/(A+B))`.
//!
//! Only `m < 1` is accepted when `m` is numeric: there the integrand is real
//! and the Legendre form is real-analytic on the whole line (no branch, no
//! jump), so it is a valid antiderivative on every interval.

use crate::calculus::trig_pole::pi_multiple;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::collections::HashMap;

fn mentions(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    crate::kernel::subs::mentions_var(expr, var, pool)
}

fn numeric(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    crate::jit::eval_interp(expr, &HashMap::new(), pool).filter(|v| v.is_finite())
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

fn half_pi(pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.rational(1, 2), pool.symbol("pi", Domain::Real)])
}

/// The integrand as `c · R^{s/2}` with `c` var-free and `s = ±1`.
fn split_radical(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, i32)> {
    let factors = match pool.get(expr) {
        ExprData::Mul(xs) => xs,
        _ => vec![expr],
    };
    let mut consts = Vec::new();
    let mut radical: Option<(ExprId, i32)> = None;
    for f in factors {
        if !mentions(f, var, pool) {
            consts.push(f);
            continue;
        }
        if radical.is_some() {
            return None;
        }
        radical = Some(match pool.get(f) {
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => (args[0], 1),
            ExprData::Pow { base, exp } => match (pool.get(base), pool.get(exp)) {
                (ExprData::Func { name, args }, ExprData::Integer(n))
                    if name == "sqrt" && args.len() == 1 && n.0 == -1 =>
                {
                    (args[0], -1)
                }
                (_, ExprData::Rational(r))
                    if *r.0.denom() == 2 && (*r.0.numer() == 1 || *r.0.numer() == -1) =>
                {
                    (base, if *r.0.numer() > 0 { 1 } else { -1 })
                }
                _ => return None,
            },
            _ => return None,
        });
    }
    let (radicand, s) = radical?;
    Some((pool.mul(consts), radicand, s))
}

/// Coefficients of a radicand that is affine in `sin²u`/`cos²u` or in
/// `sin u`/`cos u`, for a single argument `u`.
#[derive(Default)]
struct TrigAffine {
    constant: Vec<ExprId>,
    sin2: Vec<ExprId>,
    cos2: Vec<ExprId>,
    sin1: Vec<ExprId>,
    cos1: Vec<ExprId>,
    arg: Option<ExprId>,
}

impl TrigAffine {
    fn absorb(&mut self, expr: ExprId, scale: ExprId, var: ExprId, pool: &ExprPool) -> Option<()> {
        if !mentions(expr, var, pool) {
            self.constant.push(pool.mul(vec![scale, expr]));
            return Some(());
        }
        match pool.get(expr) {
            ExprData::Add(xs) => {
                for t in xs {
                    self.absorb(t, scale, var, pool)?;
                }
                Some(())
            }
            ExprData::Mul(xs) => {
                let mut c = vec![scale];
                let mut rest = None;
                for f in xs {
                    if mentions(f, var, pool) {
                        if rest.is_some() {
                            return None;
                        }
                        rest = Some(f);
                    } else {
                        c.push(f);
                    }
                }
                self.absorb(rest?, pool.mul(c), var, pool)
            }
            ExprData::Pow { base, exp } => {
                if !matches!(pool.get(exp), ExprData::Integer(n) if n.0 == 2) {
                    return None;
                }
                let (name, u) = self.trig(base, pool)?;
                self.set_arg(u)?;
                if name == "sin" {
                    self.sin2.push(scale);
                } else {
                    self.cos2.push(scale);
                }
                Some(())
            }
            ExprData::Func { .. } => {
                let (name, u) = self.trig(expr, pool)?;
                self.set_arg(u)?;
                if name == "sin" {
                    self.sin1.push(scale);
                } else {
                    self.cos1.push(scale);
                }
                Some(())
            }
            _ => None,
        }
    }

    fn trig(&self, e: ExprId, pool: &ExprPool) -> Option<(String, ExprId)> {
        match pool.get(e) {
            ExprData::Func { name, args }
                if args.len() == 1 && (name == "sin" || name == "cos") =>
            {
                Some((name, args[0]))
            }
            _ => None,
        }
    }

    fn set_arg(&mut self, u: ExprId) -> Option<()> {
        match self.arg {
            Some(a) if a != u => None,
            _ => {
                self.arg = Some(u);
                Some(())
            }
        }
    }
}

/// `m` must be a number below 1 (and not 0), or carry a free parameter.
fn acceptable_parameter(m: ExprId, pool: &ExprPool) -> bool {
    match numeric(m, pool) {
        Some(v) => v < 1.0 && v != 0.0,
        None => !matches!(
            pool.get(m),
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
        ),
    }
}

/// Exact antiderivative of a trigonometric elliptic integrand, or `None`.
pub(super) fn try_elliptic_trig(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (c, radicand, s) = split_radical(expr, var, pool)?;
    let mut t = TrigAffine::default();
    t.absorb(radicand, pool.integer(1_i32), var, pool)?;
    let u = t.arg?;
    // `u` must be linear in `var`, with a numeric non-zero slope.
    let alpha = simplify(crate::diff::diff(u, var, pool).ok()?.value, pool).value;
    if mentions(alpha, var, pool) || numeric(alpha, pool)? == 0.0 {
        return None;
    }
    let sum = |v: &Vec<ExprId>| -> ExprId {
        if v.is_empty() {
            pool.integer(0_i32)
        } else {
            pool.add(v.clone())
        }
    };
    let quadratic = !t.sin2.is_empty() || !t.cos2.is_empty();
    let linear = !t.sin1.is_empty() || !t.cos1.is_empty();
    // (amplitude φ, scale A, parameter m, extra factor from dφ/du)
    let (phi, a, m, chain) = if quadratic && !linear {
        let (a, b, phi) = if t.sin2.is_empty() {
            // A + B·cos²u = A + B·sin²(u − π/2).
            (
                sum(&t.constant),
                sum(&t.cos2),
                pool.add(vec![u, neg(half_pi(pool), pool)]),
            )
        } else {
            // A + S·sin²u + C·cos²u = (A + C) + (S − C)·sin²u.
            let c2 = sum(&t.cos2);
            (
                pool.add(vec![sum(&t.constant), c2]),
                pool.add(vec![sum(&t.sin2), neg(c2, pool)]),
                u,
            )
        };
        let a = simplify(a, pool).value;
        let m = simplify(
            pool.mul(vec![neg(b, pool), pool.pow(a, pool.integer(-1_i32))]),
            pool,
        )
        .value;
        (phi, a, m, pool.integer(1_i32))
    } else if linear && !quadratic && (t.sin1.is_empty() || t.cos1.is_empty()) {
        // A + B·cos v = (A + B)·(1 − m·sin²(v/2)),  m = 2B/(A + B);
        // A + B·sin u = A + B·cos(u − π/2).
        let (b, v) = if t.sin1.is_empty() {
            (sum(&t.cos1), u)
        } else {
            (sum(&t.sin1), pool.add(vec![u, neg(half_pi(pool), pool)]))
        };
        let a = simplify(pool.add(vec![sum(&t.constant), b]), pool).value;
        let m = simplify(
            pool.mul(vec![
                pool.integer(2_i32),
                b,
                pool.pow(a, pool.integer(-1_i32)),
            ]),
            pool,
        )
        .value;
        let phi = pool.mul(vec![pool.rational(1, 2), v]);
        (phi, a, m, pool.integer(2_i32))
    } else {
        return None;
    };
    if !(numeric(a, pool)? > 0.0) || !acceptable_parameter(m, pool) {
        return None;
    }
    let (head, a_pow) = if s < 0 {
        ("EllipticF", pool.rational(-1, 2))
    } else {
        ("EllipticE", pool.rational(1, 2))
    };
    let result = pool.mul(vec![
        c,
        chain,
        pool.pow(a, a_pow),
        pool.pow(alpha, pool.integer(-1_i32)),
        pool.func(head, vec![phi, m]),
    ]);
    let result = simplify(result, pool).value;
    // Defence in depth: the identities are exact, so the derivative must match
    // the integrand wherever both evaluate.
    derivative_agrees(result, expr, var, pool).then_some(result)
}

fn derivative_agrees(f: ExprId, integrand: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let Ok(d) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let mut checked = 0;
    for &t in &[0.3_f64, 0.71, 1.37, 2.9, -0.55] {
        let env = HashMap::from([(var, t)]);
        let (Some(a), Some(b)) = (
            crate::jit::eval_interp(d.value, &env, pool),
            crate::jit::eval_interp(integrand, &env, pool),
        ) else {
            continue;
        };
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        if (a - b).abs() > 1e-9 * (1.0 + b.abs()) {
            return false;
        }
        checked += 1;
    }
    // With a free parameter nothing evaluates; the identities stand on their own.
    checked > 0 || !free_of_symbols_but(integrand, var, pool)
}

fn free_of_symbols_but(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Symbol { name, .. } => expr == var || name == "pi",
        ExprData::Add(xs) | ExprData::Mul(xs) | ExprData::Func { args: xs, .. } => {
            xs.iter().all(|&a| free_of_symbols_but(a, var, pool))
        }
        ExprData::Pow { base, exp } => {
            free_of_symbols_but(base, var, pool) && free_of_symbols_but(exp, var, pool)
        }
        _ => true,
    }
}

/// Rewrite `F(nπ/2 | m) = n·K(m)` and `E(nπ/2 | m) = n·E(m)` (`n ∈ ℤ`) — and
/// `tan(kπ) = atan(0) = 0` — in a var-free endpoint value.  `n = 0, ±1` hold by definition; `|n| ≥ 2` uses the
/// quasi-periodicity `F(φ + π) = F(φ) + 2K`, which needs `m < 1`.
pub(super) fn reduce_elliptic_amplitudes(expr: ExprId, pool: &ExprPool) -> ExprId {
    let mut memo = HashMap::new();
    let out = reduce(expr, pool, &mut memo);
    if out == expr {
        expr
    } else {
        simplify(out, pool).value
    }
}

fn reduce(expr: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, ExprId>) -> ExprId {
    if let Some(&r) = memo.get(&expr) {
        return r;
    }
    let out = match pool.get(expr) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&a| reduce(a, pool, memo)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&a| reduce(a, pool, memo)).collect()),
        ExprData::Pow { base, exp } => pool.pow(reduce(base, pool, memo), reduce(exp, pool, memo)),
        ExprData::Func { name, args } => {
            let args: Vec<ExprId> = args.iter().map(|&a| reduce(a, pool, memo)).collect();
            quarter_period(&name, &args, pool).unwrap_or_else(|| pool.func(name, args))
        }
        _ => expr,
    };
    memo.insert(expr, out);
    out
}

fn quarter_period(name: &str, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    // `tan(kπ) = 0` and `atan(0) = 0`: what a one-sided `tan(x/2)` endpoint
    // value reduces to on the far side of the pole (`atan(c·tan π)`).
    if args.len() == 1 && name == "tan" && pi_multiple(args[0], pool)?.is_integer() {
        return Some(pool.integer(0_i32));
    }
    if args.len() == 1 && name == "atan" {
        let a = simplify(args[0], pool).value;
        return matches!(pool.get(a), ExprData::Integer(n) if n.0 == 0)
            .then(|| pool.integer(0_i32));
    }
    if args.len() != 2 || !(name == "EllipticF" || name == "EllipticE") {
        return None;
    }
    let (phi, m) = (args[0], args[1]);
    let q = pi_multiple(phi, pool)?;
    let n2 = rug::Rational::from(&q * 2u32);
    if !n2.is_integer() {
        return None;
    }
    let n = n2.numer().to_i64()?;
    if n == 0 {
        return Some(pool.integer(0_i32));
    }
    if n.abs() >= 2 && !numeric(m, pool).is_some_and(|v| v < 1.0) {
        return None;
    }
    let complete = if name == "EllipticF" {
        "EllipticK"
    } else {
        "EllipticE"
    };
    Some(pool.mul(vec![pool.integer(n), pool.func(complete, vec![m])]))
}

/// Substitute and reduce: the endpoint value of an elliptic antiderivative.
#[cfg(test)]
fn at(f: ExprId, var: ExprId, v: ExprId, pool: &ExprPool) -> ExprId {
    let m = HashMap::from([(var, v)]);
    reduce_elliptic_amplitudes(simplify(crate::kernel::subs(f, &m, pool), pool).value, pool)
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

    fn sin2(pool: &ExprPool, u: ExprId) -> ExprId {
        pool.pow(pool.func("sin", vec![u]), pool.integer(2_i32))
    }

    fn val(e: ExprId, pool: &ExprPool) -> f64 {
        numeric(e, pool).expect("evaluates")
    }

    /// K(m) by the AGM, independently of the primitive under test.
    fn agm_k(m: f64) -> f64 {
        let (mut a, mut b) = (1.0_f64, (1.0 - m).sqrt());
        for _ in 0..40 {
            let (an, bn) = (0.5 * (a + b), (a * b).sqrt());
            a = an;
            b = bn;
        }
        std::f64::consts::PI / (2.0 * a)
    }

    #[test]
    fn complete_first_kind_is_exactly_k() {
        // ∫_0^{π/2} dx/√(1 − sin²x/4) = K(1/4), exactly.
        let (pool, x, pi) = setup();
        let rad = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.rational(-1, 4), sin2(&pool, x)]),
        ]);
        let f = pool.pow(pool.func("sqrt", vec![rad]), pool.integer(-1_i32));
        let big_f = try_elliptic_trig(f, x, &pool).expect("recognised");
        let hi = at(big_f, x, pool.mul(vec![pool.rational(1, 2), pi]), &pool);
        let lo = at(big_f, x, pool.integer(0_i32), &pool);
        let v = simplify(pool.add(vec![hi, neg(lo, &pool)]), &pool).value;
        assert_eq!(v, pool.func("EllipticK", vec![pool.rational(1, 4)]));
        assert!((val(v, &pool) - 1.685_750_354_812_596).abs() < 1e-14);
    }

    #[test]
    fn cos_squared_form_reflects_to_k() {
        let (pool, x, pi) = setup();
        let cos2 = pool.pow(pool.func("cos", vec![x]), pool.integer(2_i32));
        let rad = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.rational(-1, 2), cos2]),
        ]);
        let f = pool.pow(rad, pool.rational(-1, 2));
        let big_f = try_elliptic_trig(f, x, &pool).expect("recognised");
        let hi = at(big_f, x, pool.mul(vec![pool.rational(1, 2), pi]), &pool);
        let lo = at(big_f, x, pool.integer(0_i32), &pool);
        let v = simplify(pool.add(vec![hi, neg(lo, &pool)]), &pool).value;
        assert_eq!(v, pool.func("EllipticK", vec![pool.rational(1, 2)]));
    }

    #[test]
    fn linear_cos_form_uses_the_half_angle() {
        // ∫_0^π dx/√(2 + cos x) = 2/√3 · K(2/3).
        let (pool, x, pi) = setup();
        let rad = pool.add(vec![pool.integer(2_i32), pool.func("cos", vec![x])]);
        let f = pool.pow(pool.func("sqrt", vec![rad]), pool.integer(-1_i32));
        let big_f = try_elliptic_trig(f, x, &pool).expect("recognised");
        let v = simplify(
            pool.add(vec![
                at(big_f, x, pi, &pool),
                neg(at(big_f, x, pool.integer(0_i32), &pool), &pool),
            ]),
            &pool,
        )
        .value;
        let want = 2.0 / 3f64.sqrt() * agm_k(2.0 / 3.0);
        assert!((val(v, &pool) - want).abs() < 1e-13, "{}", pool.display(v));
        assert!(pool.display(v).to_string().contains("EllipticK"));
    }

    #[test]
    fn second_kind_and_scaled_argument() {
        // ∫ √(3 − sin²(2x)) dx: A = 3, m = 1/3, u′ = 2.
        let (pool, x, _pi) = setup();
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let rad = pool.add(vec![
            pool.integer(3_i32),
            pool.mul(vec![pool.integer(-1_i32), sin2(&pool, two_x)]),
        ]);
        let f = pool.func("sqrt", vec![rad]);
        let big_f = try_elliptic_trig(f, x, &pool).expect("recognised");
        let s = pool.display(big_f).to_string();
        assert!(s.contains("EllipticE") && s.contains("1/3"), "{s}");
    }

    #[test]
    fn declines_outside_the_real_range() {
        let (pool, x, _pi) = setup();
        // m = 4 > 1: the integrand is not real on the whole line.
        let rad = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-4_i32), sin2(&pool, x)]),
        ]);
        let f = pool.pow(rad, pool.rational(-1, 2));
        assert!(try_elliptic_trig(f, x, &pool).is_none());
        // A ≤ 0.
        let rad = pool.add(vec![pool.integer(-1_i32), sin2(&pool, x)]);
        assert!(try_elliptic_trig(pool.pow(rad, pool.rational(-1, 2)), x, &pool).is_none());
        // Not a radical.
        let rad = pool.add(vec![pool.integer(2_i32), sin2(&pool, x)]);
        assert!(try_elliptic_trig(pool.pow(rad, pool.integer(-1_i32)), x, &pool).is_none());
    }

    #[test]
    fn quarter_periods_beyond_one_need_m_below_one() {
        let (pool, _x, pi) = setup();
        let k = pool.symbol("k", Domain::Real);
        let three_half_pi = pool.mul(vec![pool.rational(3, 2), pi]);
        let f = pool.func("EllipticF", vec![three_half_pi, pool.rational(1, 3)]);
        assert_eq!(
            reduce_elliptic_amplitudes(f, &pool),
            pool.mul(vec![
                pool.integer(3_i32),
                pool.func("EllipticK", vec![pool.rational(1, 3)])
            ])
        );
        let g = pool.func("EllipticF", vec![three_half_pi, k]);
        assert_eq!(reduce_elliptic_amplitudes(g, &pool), g);
    }
}
