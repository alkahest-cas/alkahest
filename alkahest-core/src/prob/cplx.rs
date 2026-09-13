//! Just enough complex arithmetic, at [`super::quad::VERIFY_PREC`] bits, to
//! evaluate a characteristic function.
//!
//! # Why this exists rather than a call to something already in the crate
//!
//! `eval_complex_f64` is `f64`, which leaves three digits between "agrees" and
//! "agrees to the limit of the instrument" once the gate's `1e-12` threshold is
//! accounted for — not enough for the comparison to mean much. `AcbBall` has
//! the precision but no transcendental kernels, so it cannot evaluate `e^{it}`.
//!
//! What is here is deliberately **narrow**: the operations a characteristic
//! function in this module's own table is built from, and nothing else. An
//! expression outside that grammar returns `None` and the verification refuses,
//! which is the right failure — a half-supported complex evaluator that guesses
//! at an unknown head is how a wrong branch cut becomes a wrong probability.

use rug::Float;

use crate::kernel::{ExprData, ExprId, ExprPool};

use super::quad::{fl, pi_ball};

/// A complex number as a pair of [`rug::Float`]s.
#[derive(Clone, Debug)]
pub(crate) struct C {
    pub re: Float,
    pub im: Float,
}

impl C {
    pub fn new(re: Float, im: Float) -> Self {
        C { re, im }
    }

    pub fn real(v: Float, prec: u32) -> Self {
        C::new(v, fl(prec, 0.0))
    }

    fn prec(&self) -> u32 {
        self.re.prec()
    }

    pub fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn add(&self, o: &C) -> C {
        let p = self.prec();
        C::new(
            Float::with_val(p, &self.re + &o.re),
            Float::with_val(p, &self.im + &o.im),
        )
    }

    fn mul(&self, o: &C) -> C {
        let p = self.prec();
        let ac = Float::with_val(p, &self.re * &o.re);
        let bd = Float::with_val(p, &self.im * &o.im);
        let ad = Float::with_val(p, &self.re * &o.im);
        let bc = Float::with_val(p, &self.im * &o.re);
        C::new(Float::with_val(p, ac - bd), Float::with_val(p, ad + bc))
    }

    fn inv(&self) -> Option<C> {
        let p = self.prec();
        let d = Float::with_val(
            p,
            Float::with_val(p, &self.re * &self.re) + Float::with_val(p, &self.im * &self.im),
        );
        if d == 0 {
            return None;
        }
        Some(C::new(
            Float::with_val(p, &self.re / &d),
            Float::with_val(p, -Float::with_val(p, &self.im / &d)),
        ))
    }

    fn exp(&self) -> C {
        let p = self.prec();
        let r = Float::with_val(p, self.re.clone().exp());
        C::new(
            Float::with_val(p, &r * Float::with_val(p, self.im.clone().cos())),
            Float::with_val(p, &r * Float::with_val(p, self.im.clone().sin())),
        )
    }

    /// Principal logarithm: `ln|z| + i·arg z`, with the branch cut on the
    /// negative real axis.
    ///
    /// Every base this module raises to a power — `1 - iθt` for the `Gamma`
    /// characteristic function — has a positive real part, so the principal
    /// branch is the continuous one there and the cut is never approached. A
    /// base *on* the cut returns `None` rather than a choice of side.
    fn log(&self) -> Option<C> {
        let p = self.prec();
        if self.re == 0 && self.im == 0 {
            return None;
        }
        if self.im == 0 && self.re < 0 {
            return None;
        }
        let modulus = Float::with_val(
            p,
            Float::with_val(
                p,
                Float::with_val(p, &self.re * &self.re) + Float::with_val(p, &self.im * &self.im),
            )
            .sqrt(),
        );
        let arg = Float::with_val(p, self.im.clone().atan2(&self.re));
        Some(C::new(Float::with_val(p, modulus.ln()), arg))
    }

    fn powi(&self, n: i64) -> Option<C> {
        let p = self.prec();
        if n == 0 {
            return Some(C::real(fl(p, 1.0), p));
        }
        let mut acc = C::real(fl(p, 1.0), p);
        for _ in 0..n.unsigned_abs() {
            acc = acc.mul(self);
        }
        if n < 0 {
            acc.inv()
        } else {
            Some(acc)
        }
    }

    fn pow(&self, e: &C) -> Option<C> {
        if e.im == 0 {
            if let Some(n) = float_to_i64(&e.re) {
                return self.powi(n);
            }
        }
        Some(e.mul(&self.log()?).exp())
    }
}

fn float_to_i64(v: &Float) -> Option<i64> {
    if !v.is_integer() {
        return None;
    }
    let n = v.to_integer()?;
    if n.significant_bits() > 32 {
        return None;
    }
    n.to_i64()
}

/// Evaluate `expr` in ℂ at `bindings` (all real), with `π` and the imaginary
/// unit bound.
///
/// `None` for anything outside the small grammar this module generates.
pub(crate) fn eval_complex(
    expr: ExprId,
    bindings: &[(ExprId, Float)],
    pool: &ExprPool,
    prec: u32,
) -> Option<C> {
    let v = walk(expr, bindings, pool, prec, 0)?;
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

fn walk(
    expr: ExprId,
    bindings: &[(ExprId, Float)],
    pool: &ExprPool,
    prec: u32,
    depth: u32,
) -> Option<C> {
    if depth > 64 {
        return None;
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(C::real(Float::with_val(prec, &n.0), prec)),
        ExprData::Rational(r) => Some(C::real(Float::with_val(prec, &r.0), prec)),
        ExprData::Float(f) => Some(C::real(Float::with_val(prec, f.inner.to_f64()), prec)),
        ExprData::Symbol { .. } => {
            if pool.is_imaginary_unit(expr) {
                return Some(C::new(fl(prec, 0.0), fl(prec, 1.0)));
            }
            if expr == super::pi(pool) {
                return Some(C::real(pi_ball(prec).mid, prec));
            }
            bindings
                .iter()
                .find(|(s, _)| *s == expr)
                .map(|(_, v)| C::real(v.clone(), prec))
        }
        ExprData::Add(args) => {
            let mut acc = C::real(fl(prec, 0.0), prec);
            for a in args {
                acc = acc.add(&walk(a, bindings, pool, prec, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = C::real(fl(prec, 1.0), prec);
            for a in args {
                acc = acc.mul(&walk(a, bindings, pool, prec, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let b = walk(base, bindings, pool, prec, depth + 1)?;
            let e = walk(exp, bindings, pool, prec, depth + 1)?;
            b.pow(&e)
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let a = walk(args[0], bindings, pool, prec, depth + 1)?;
            match name.as_str() {
                "exp" => Some(a.exp()),
                "log" => a.log(),
                "sqrt" => a.pow(&C::real(fl(prec, 0.5), prec)),
                "cos" => {
                    // cos z = (e^{iz} + e^{-iz})/2
                    let i = C::new(fl(prec, 0.0), fl(prec, 1.0));
                    let p1 = i.mul(&a).exp();
                    let p2 = i.mul(&a).mul(&C::real(fl(prec, -1.0), prec)).exp();
                    Some(p1.add(&p2).mul(&C::real(fl(prec, 0.5), prec)))
                }
                "sin" => {
                    let i = C::new(fl(prec, 0.0), fl(prec, 1.0));
                    let p1 = i.mul(&a).exp();
                    let p2 = i.mul(&a).mul(&C::real(fl(prec, -1.0), prec)).exp();
                    let diff = p1.add(&p2.mul(&C::real(fl(prec, -1.0), prec)));
                    // divide by 2i
                    diff.mul(&C::new(fl(prec, 0.0), fl(prec, -0.5))).into()
                }
                _ => None,
            }
        }
        _ => None,
    }
}
