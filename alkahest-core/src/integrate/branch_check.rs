//! The **complex branch** of an antiderivative: `d/dx F = f` where the
//! integrand is not real.
//!
//! # Why the real gates are not enough
//!
//! Every numeric gate in this crate compares `d/dx F` with `f` at real sample
//! points where `f` is a finite *real* number, and skips the rest as "outside
//! the integrand's domain". For `∫√(tan³x) dx` that skips every `x` with
//! `tan x < 0`, where the integrand is perfectly well defined — it is
//! `i·|tan x|^{3/2}` on the principal branch — and the answer
//! `(√(tan x))³` is **wrong** there by a sign: `(√t)³ = −i|t|^{3/2}` for
//! `t < 0`. The rewrites that lose that sign (`(a³)^{1/2} ↦ (a^{1/2})³`,
//! `(1/a)^{1/2} ↦ 1/a^{1/2}`, `√a·√b ↦ √(ab)`) are only valid for `a > 0`, and
//! the substitution routes use them freely because their own gates cannot see
//! the other half of the line (audit A10: `∫cot^{3/2}` gave `0.108 + 2.668i`
//! on `[1, 5/2]` where the value is `0.108 − 0.446i`).
//!
//! # What is checked
//!
//! At the real points of [`SAMPLES`] where `f` evaluates, on the principal
//! branch, to a finite value with a non-zero imaginary part, `d/dx F` must
//! agree with it. The evaluator here is deliberately small (arithmetic,
//! integer and fractional powers, `exp`/`log`/`sqrt`, the circular and
//! hyperbolic functions and their principal inverses); anything else makes a
//! sample *no information*, never agreement.
//!
//! Principal values **on** the negative real axis are taken as the limit from
//! above (`arg = π`), which is the convention Mathematica, SymPy and the
//! evaluators of this crate use, and it is the right one here: along a real
//! path on which a radicand stays negative, the principal `√` is analytic in
//! `x`, so `d/dx` of it is what the chain rule says. A value only *nearly* on
//! the axis — one rounding error off it — is a coin toss between two branches,
//! so such a sample is skipped rather than decided.

use std::collections::HashMap;

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// Real sample points. Irrational-looking and two-sided, like
/// `engine::GATE_X_SAMPLES`, and spread over several periods so a trig
/// integrand is seen on both signs of every factor.
const SAMPLES: [f64; 16] = [
    0.3719, 0.9137, 1.4231, 1.8813, 2.1719, 2.8123, 3.6411, 4.4172, -0.3719, -0.9137, -1.4231,
    -1.8813, -2.1719, -2.8123, -3.6411, -4.4172,
];

/// Relative tolerance for a disagreement. Loose on purpose: this check only
/// ever *refuses*, and a false refusal of a correct answer is the cost of a
/// tight one.
const TOL: f64 = 1e-6;

/// `Some(x)` when `d/dx candidate` and `integrand` disagree at the real point
/// `x`, at which the integrand is a finite non-real number; `None` when every
/// such point agrees or none could be evaluated.
pub(crate) fn complex_branch_mismatch(
    candidate: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<f64> {
    // Cheapest exit, and the common case: an integrand built only from
    // operations that map reals to reals (polynomials, `sin`, `exp`, …) has no
    // complex branch at all. This runs on every `integrate` call, so it must
    // not cost an evaluation.
    if !may_be_non_real(integrand, pool) {
        return None;
    }
    // Next: an integrand that is real at every sample (or cannot be evaluated
    // at all) has no complex branch on the grid to check.
    let complex_points: Vec<(f64, C)> = SAMPLES
        .iter()
        .filter_map(|&x| {
            let f = eval(integrand, var, x, pool)?;
            (f.im != 0.0 && f.is_finite()).then_some((x, f))
        })
        .collect();
    if complex_points.is_empty() {
        return None;
    }
    let d = crate::diff::diff(candidate, var, pool).ok()?;
    let d = simplify(d.value, pool).value;
    for (x, f) in complex_points {
        let Some(g) = eval(d, var, x, pool) else {
            continue;
        };
        if !g.is_finite() {
            continue;
        }
        let err = (g.re - f.re).hypot(g.im - f.im);
        if err > TOL * (1.0 + f.abs()) {
            return Some(x);
        }
    }
    None
}

/// Grid resolution for [`complex_antiderivative_jump`].
const JUMP_SCAN_POINTS: usize = 512;

/// For a definite integral whose integrand is **not real** somewhere on
/// `[lo, hi]`: the reason `F(hi) − F(lo)` cannot be trusted, or `None`.
///
/// The fundamental theorem needs `F` continuous on the interval, and the jump
/// and domain-hole checks in `engine` establish that on the real line only —
/// they sample `F` where it is a finite *real* number. A principal-branch
/// antiderivative built by a substitution through a pole of the substituted
/// function (`u = √(tan x)` across `x = π/2`) is continuous on each side and
/// jumps at the pole, and on the non-real part of the line nothing looked.
/// So walk a fine grid and compare every step `|F(xₖ₊₁) − F(xₖ)|` with what
/// the integrand allows (`h · max|f|`, with generous slack); a step far
/// beyond it is a jump. A grid on which `F` or `f` cannot be evaluated at
/// enough points is also a refusal: there is no evidence of continuity.
///
/// `None` means either the integrand is real at every grid point (the real
/// checks own that case) or the walk found `F` continuous.
pub(crate) fn complex_antiderivative_jump(
    antiderivative: ExprId,
    integrand: ExprId,
    var: ExprId,
    lo: f64,
    hi: f64,
    pool: &ExprPool,
) -> Option<String> {
    let width = hi - lo;
    if !(width > 0.0 && width.is_finite() && may_be_non_real(integrand, pool)) {
        return None;
    }
    let h = width / JUMP_SCAN_POINTS as f64;
    let xs: Vec<f64> = (0..=JUMP_SCAN_POINTS).map(|k| lo + h * k as f64).collect();
    let fs: Vec<Option<C>> = xs.iter().map(|&x| eval(integrand, var, x, pool)).collect();
    if !fs.iter().flatten().any(|f| f.im != 0.0 && f.is_finite()) {
        return None;
    }
    let big: Vec<Option<C>> = xs
        .iter()
        .map(|&x| eval(antiderivative, var, x, pool).filter(|v| v.is_finite()))
        .collect();
    let evaluable = big.iter().filter(|v| v.is_some()).count();
    if evaluable * 4 < xs.len() * 3 {
        return Some(format!(
            "the integrand is not real on part of [{lo}, {hi}], and the antiderivative could \
             not be evaluated there on the principal branch at enough points to establish \
             that it is continuous, so the fundamental-theorem difference F(b) - F(a) is \
             not known to be the integral"
        ));
    }
    for k in 0..JUMP_SCAN_POINTS {
        let (Some(a), Some(b)) = (big[k], big[k + 1]) else {
            continue;
        };
        let step = (b.re - a.re).hypot(b.im - a.im);
        let slope = [fs[k], fs[k + 1]]
            .iter()
            .flatten()
            .filter(|f| f.is_finite())
            .map(|f| f.abs())
            .fold(0.0_f64, f64::max);
        if step > 1e-6 + 50.0 * h * (1.0 + slope) {
            return Some(format!(
                "the antiderivative jumps by {step:.3e} between {} = {:.6} and {:.6} on the \
                 non-real branch of the integrand, so it is not continuous on [{lo}, {hi}] and \
                 the fundamental-theorem difference F(b) - F(a) is not the integral",
                pool.display(var),
                xs[k],
                xs[k + 1]
            ));
        }
    }
    None
}

/// `false` when `expr` is built only from operations that take real inputs to
/// real outputs — `+`, `·`, integer powers, `exp`, `sin`, `cos`, `tan`,
/// `atan`, the hyperbolic functions, `asinh` — so it is real wherever it is
/// finite and there is no complex branch to check. Anything else (a
/// fractional power, `sqrt`, `log`, `asin`, the imaginary unit, a head this
/// list does not know) answers `true`, which only means "look". Syntactic,
/// allocation-free, and linear in the tree.
fn may_be_non_real(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => false,
        ExprData::Symbol { .. } => pool.is_imaginary_unit(expr),
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| may_be_non_real(x, pool)),
        ExprData::Pow { base, exp } => {
            !matches!(pool.get(exp), ExprData::Integer(_)) || may_be_non_real(base, pool)
        }
        ExprData::Func { name, args } => {
            !matches!(
                name.as_str(),
                "exp" | "sin" | "cos" | "tan" | "atan" | "sinh" | "cosh" | "tanh" | "asinh"
            ) || args.iter().any(|&a| may_be_non_real(a, pool))
        }
        _ => true,
    }
}

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const fn real(re: f64) -> Self {
        C { re, im: 0.0 }
    }
    fn is_real(self) -> bool {
        self.im == 0.0
    }
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn add(self, o: C) -> C {
        C {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    fn mul(self, o: C) -> C {
        if self.is_real() && o.is_real() {
            return C::real(self.re * o.re);
        }
        C {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn recip(self) -> C {
        if self.is_real() {
            return C::real(1.0 / self.re);
        }
        let d = self.re * self.re + self.im * self.im;
        C {
            re: self.re / d,
            im: -self.im / d,
        }
    }
    fn exp(self) -> C {
        if self.is_real() {
            return C::real(self.re.exp());
        }
        let m = self.re.exp();
        C {
            re: m * self.im.cos(),
            im: m * self.im.sin(),
        }
    }
    /// Principal argument, `π` on the negative real axis; `None` when the value
    /// is within rounding of that axis without being on it.
    fn arg(self) -> Option<f64> {
        if self.im == 0.0 {
            return Some(if self.re < 0.0 {
                std::f64::consts::PI
            } else {
                0.0
            });
        }
        if self.re < 0.0 && self.im.abs() <= 1e-12 * self.re.abs() {
            return None;
        }
        Some(self.im.atan2(self.re))
    }
    fn ln(self) -> Option<C> {
        if self.re == 0.0 && self.im == 0.0 {
            return None;
        }
        if self.is_real() && self.re > 0.0 {
            return Some(C::real(self.re.ln()));
        }
        Some(C {
            re: self.abs().ln(),
            im: self.arg()?,
        })
    }
    fn powi(self, n: i64) -> C {
        if self.is_real() {
            return C::real(
                self.re
                    .powi(n.clamp(i32::MIN as i64, i32::MAX as i64) as i32),
            );
        }
        let mut acc = C::real(1.0);
        let mut base = if n < 0 { self.recip() } else { self };
        let mut k = n.unsigned_abs();
        while k > 0 {
            if k & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            k >>= 1;
        }
        acc
    }
    /// `self^w` on the principal branch.
    fn pow(self, w: C) -> Option<C> {
        if w.is_real() && w.re.fract() == 0.0 && w.re.abs() <= 1024.0 {
            return Some(self.powi(w.re as i64));
        }
        if self.re == 0.0 && self.im == 0.0 {
            return (w.is_real() && w.re > 0.0).then_some(C::real(0.0));
        }
        if self.is_real() && self.re > 0.0 && w.is_real() {
            return Some(C::real(self.re.powf(w.re)));
        }
        Some(w.mul(self.ln()?).exp())
    }
    fn sqrt(self) -> Option<C> {
        if self.is_real() {
            return Some(if self.re >= 0.0 {
                C::real(self.re.sqrt())
            } else {
                C {
                    re: 0.0,
                    im: (-self.re).sqrt(),
                }
            });
        }
        self.pow(C::real(0.5))
    }
    fn sin(self) -> C {
        if self.is_real() {
            return C::real(self.re.sin());
        }
        C {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }
    fn cos(self) -> C {
        if self.is_real() {
            return C::real(self.re.cos());
        }
        C {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }
    fn sinh(self) -> C {
        if self.is_real() {
            return C::real(self.re.sinh());
        }
        C {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }
    fn cosh(self) -> C {
        if self.is_real() {
            return C::real(self.re.cosh());
        }
        C {
            re: self.re.cosh() * self.im.cos(),
            im: self.re.sinh() * self.im.sin(),
        }
    }
}

const I: C = C { re: 0.0, im: 1.0 };

fn neg(z: C) -> C {
    C {
        re: -z.re,
        im: -z.im,
    }
}

/// Evaluate `expr` at `var = x` on the principal branch. `None` for anything
/// outside this evaluator's grammar, a free symbol, or a value too close to a
/// branch cut to place.
fn eval(expr: ExprId, var: ExprId, x: f64, pool: &ExprPool) -> Option<C> {
    let mut memo = HashMap::new();
    walk(expr, var, x, pool, &mut memo, 0)
}

fn walk(
    expr: ExprId,
    var: ExprId,
    x: f64,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, Option<C>>,
    depth: u32,
) -> Option<C> {
    if depth > 256 {
        return None;
    }
    if let Some(v) = memo.get(&expr) {
        return *v;
    }
    let v = walk_uncached(expr, var, x, pool, memo, depth);
    memo.insert(expr, v);
    v
}

fn walk_uncached(
    expr: ExprId,
    var: ExprId,
    x: f64,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, Option<C>>,
    depth: u32,
) -> Option<C> {
    if expr == var {
        return Some(C::real(x));
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(C::real(n.0.to_f64())),
        ExprData::Rational(r) => Some(C::real(r.0.to_f64())),
        ExprData::Float(f) => Some(C::real(f.inner.to_f64())),
        ExprData::Symbol { .. } => {
            if pool.is_imaginary_unit(expr) {
                Some(I)
            } else if crate::eval::symbols::is_pi(expr, pool) {
                Some(C::real(std::f64::consts::PI))
            } else {
                None
            }
        }
        ExprData::Add(args) => {
            let mut acc = C::real(0.0);
            for a in args {
                acc = acc.add(walk(a, var, x, pool, memo, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = C::real(1.0);
            for a in args {
                acc = acc.mul(walk(a, var, x, pool, memo, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let b = walk(base, var, x, pool, memo, depth + 1)?;
            let e = walk(exp, var, x, pool, memo, depth + 1)?;
            b.pow(e)
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let z = walk(args[0], var, x, pool, memo, depth + 1)?;
            let one = C::real(1.0);
            match name.as_str() {
                "exp" => Some(z.exp()),
                "log" => z.ln(),
                "sqrt" => z.sqrt(),
                "sin" => Some(z.sin()),
                "cos" => Some(z.cos()),
                "tan" => Some(z.sin().mul(z.cos().recip())),
                "sinh" => Some(z.sinh()),
                "cosh" => Some(z.cosh()),
                "tanh" => Some(z.sinh().mul(z.cosh().recip())),
                "atan" if z.is_real() => Some(C::real(z.re.atan())),
                // atan z = (i/2)·(log(1 − iz) − log(1 + iz))
                "atan" => {
                    let a = one.add(neg(I.mul(z))).ln()?;
                    let b = one.add(I.mul(z)).ln()?;
                    Some(C { re: 0.0, im: 0.5 }.mul(a.add(neg(b))))
                }
                "asin" if z.is_real() && z.re.abs() <= 1.0 => Some(C::real(z.re.asin())),
                // asin z = −i·log(iz + √(1 − z²))
                "asin" => {
                    let r = one.add(neg(z.mul(z))).sqrt()?;
                    Some(neg(I).mul(I.mul(z).add(r).ln()?))
                }
                "acos" if z.is_real() && z.re.abs() <= 1.0 => Some(C::real(z.re.acos())),
                "acos" => {
                    let r = one.add(neg(z.mul(z))).sqrt()?;
                    let asin = neg(I).mul(I.mul(z).add(r).ln()?);
                    Some(C::real(std::f64::consts::FRAC_PI_2).add(neg(asin)))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// `(√(tan x))³` and `√(tan³x)` agree where `tan x > 0` and differ by a sign
    /// where it is negative — the audit A10 answer.
    #[test]
    fn a_lost_sign_on_the_complex_branch_is_seen() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let t = p.func("tan", vec![x]);
        let half = p.rational(1, 2);
        let right = p.pow(p.pow(t, p.integer(3)), half);
        let wrong = p.pow(p.pow(t, half), p.integer(3));
        let integrand = crate::diff::diff(right, x, &p).unwrap().value;
        assert!(complex_branch_mismatch(wrong, integrand, x, &p).is_some());
        assert!(complex_branch_mismatch(right, integrand, x, &p).is_none());
    }

    /// An integrand that is real everywhere has nothing on the complex branch
    /// to check, whatever the candidate.
    #[test]
    fn a_real_integrand_is_left_to_the_real_gates() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.func("cos", vec![x]);
        let not_even_close = p.func("exp", vec![x]);
        assert!(complex_branch_mismatch(not_even_close, f, x, &p).is_none());
    }

    /// Principal values on the negative real axis: `√(−4) = 2i`, and `(−8)^{1/3}`
    /// is `1 + i√3`, not `−2`.
    #[test]
    fn principal_values_on_the_cut() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let s = eval(p.func("sqrt", vec![x]), x, -4.0, &p).unwrap();
        assert!((s.re).abs() < 1e-15 && (s.im - 2.0).abs() < 1e-15);
        let c = eval(p.pow(x, p.rational(1, 3)), x, -8.0, &p).unwrap();
        assert!((c.re - 1.0).abs() < 1e-12 && (c.im - 3f64.sqrt()).abs() < 1e-12);
    }
}
