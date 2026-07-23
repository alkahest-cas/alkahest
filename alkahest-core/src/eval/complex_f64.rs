//! IEEE-754 complex evaluation for `re` / `im` / `conjugate` / `arg`.

use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;

use super::{error, EvalError, UnsupportedReason};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexF64 {
    pub re: f64,
    pub im: f64,
}

impl ComplexF64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn powi(self, n: i64) -> Result<Self, EvalError> {
        if n == 0 {
            return Ok(Self::ONE);
        }
        if n < 0 {
            if self.re == 0.0 && self.im == 0.0 {
                return Err(error(UnsupportedReason::ZeroToNegativePower));
            }
            let p = self.powi(-n)?;
            let d = p.re * p.re + p.im * p.im;
            if d == 0.0 || !d.is_finite() {
                return Err(error(UnsupportedReason::NonFiniteResult));
            }
            return Ok(Self::new(p.re / d, -p.im / d));
        }
        let mut acc = Self::ONE;
        let mut base = self;
        let mut e = n;
        while e != 0 {
            if e & 1 == 1 {
                acc = acc.mul(base);
            }
            e >>= 1;
            if e != 0 {
                base = base.mul(base);
            }
        }
        Ok(acc)
    }

    /// Principal-branch power `z^w = exp(w · Log z)` for `z ≠ 0`.
    fn powc(self, exp: Self) -> Result<Self, EvalError> {
        if self.re == 0.0 && self.im == 0.0 {
            // 0^w: only non-negative real exponents are defined in the
            // principal sense we support here.
            if exp.im == 0.0 && exp.re > 0.0 {
                return Ok(Self::ZERO);
            }
            if exp.re == 0.0 && exp.im == 0.0 {
                return Err(error(UnsupportedReason::UnsupportedExpression {
                    kind: "branch_cut",
                }));
            }
            return Err(error(UnsupportedReason::ZeroToNegativePower));
        }
        let ln = self.ln()?;
        Ok(exp.mul(ln).exp())
    }

    fn sqrt(self) -> Result<Self, EvalError> {
        // Use the principal logarithm rather than the textbook geometric
        // formula: `((r±re)/2)^½` suffers catastrophic cancellation for
        // arguments near the negative-real cut (e.g. -100 + 1e-6·i).
        if self.re == 0.0 && self.im == 0.0 {
            return Ok(Self::ZERO);
        }
        self.powc(Self::new(0.5, 0.0))
    }
    fn exp(self) -> Self {
        let s = self.re.exp();
        Self::new(s * self.im.cos(), s * self.im.sin())
    }
    fn ln(self) -> Result<Self, EvalError> {
        if self.re == 0.0 && self.im == 0.0 {
            return Err(error(UnsupportedReason::UnsupportedExpression {
                kind: "branch_cut",
            }));
        }
        let r = (self.re * self.re + self.im * self.im).sqrt();
        Ok(Self::new(r.ln(), self.im.atan2(self.re)))
    }
    fn sin(self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }
    fn cos(self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }
    fn principal_arg(self) -> Result<f64, EvalError> {
        // Principal arg is undefined at 0 and discontinuous on the negative real axis.
        if self.im == 0.0 && self.re <= 0.0 {
            return Err(error(UnsupportedReason::UnsupportedExpression {
                kind: "branch_cut",
            }));
        }
        Ok(self.im.atan2(self.re))
    }
}

pub fn eval_complex_f64(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, ComplexF64>,
) -> Result<ComplexF64, EvalError> {
    let v = eval_node(expr, pool, bindings)?;
    if v.re.is_finite() && v.im.is_finite() {
        Ok(v)
    } else {
        Err(error(UnsupportedReason::NonFiniteResult))
    }
}

fn eval_node(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, ComplexF64>,
) -> Result<ComplexF64, EvalError> {
    match pool.get(expr) {
        ExprData::Integer(n) => Ok(ComplexF64::new(n.0.to_f64(), 0.0)),
        ExprData::Rational(r) => Ok(ComplexF64::new(r.0.to_f64(), 0.0)),
        ExprData::Float(f) => Ok(ComplexF64::new(f.inner.to_f64(), 0.0)),
        ExprData::Symbol { .. } => {
            if let Some(&v) = bindings.get(&expr) {
                Ok(v)
            } else if pool.is_imaginary_unit(expr) {
                // Canonical I evaluates to 0+1j in complex mode without an
                // explicit binding (matches symbolic `i² → −1` folding).
                Ok(ComplexF64::new(0.0, 1.0))
            } else {
                Err(error(UnsupportedReason::UnboundSymbol { symbol: expr }))
            }
        }
        ExprData::Add(args) => args.iter().try_fold(ComplexF64::ZERO, |a, &x| {
            Ok(a.add(eval_node(x, pool, bindings)?))
        }),
        ExprData::Mul(args) => args.iter().try_fold(ComplexF64::ONE, |a, &x| {
            Ok(a.mul(eval_node(x, pool, bindings)?))
        }),
        ExprData::Pow { base, exp } => {
            let b = eval_node(base, pool, bindings)?;
            match pool.get(exp) {
                ExprData::Integer(n) => b.powi(n.0.to_i64().unwrap_or(0)),
                ExprData::Rational(r) if *r.0.denom() == 1 => {
                    b.powi(r.0.numer().to_i64().unwrap_or(0))
                }
                // Principal branch: z^w = exp(w · Log z). Covers float and
                // non-integer rational exponents (e.g. (-1)^(1/2) → i).
                _ => {
                    let e = eval_node(exp, pool, bindings)?;
                    // Fast path: pure integer-valued real exponent.
                    if e.im == 0.0 && e.re.fract() == 0.0 && e.re.abs() < (i64::MAX as f64) {
                        b.powi(e.re as i64)
                    } else {
                        b.powc(e)
                    }
                }
            }
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let x = eval_node(args[0], pool, bindings)?;
            match name.as_str() {
                "sin" => Ok(x.sin()),
                "cos" => Ok(x.cos()),
                "exp" => Ok(x.exp()),
                "log" => x.ln(),
                "sqrt" => x.sqrt(),
                "re" => Ok(ComplexF64::new(x.re, 0.0)),
                "im" => Ok(ComplexF64::new(x.im, 0.0)),
                "conjugate" => Ok(ComplexF64::new(x.re, -x.im)),
                "arg" => Ok(ComplexF64::new(x.principal_arg()?, 0.0)),
                _ => Err(error(UnsupportedReason::UnsupportedFunction {
                    name: name.clone(),
                })),
            }
        }
        ExprData::Func { name, .. } => Err(error(UnsupportedReason::UnsupportedFunction {
            name: name.clone(),
        })),
        other => Err(error(UnsupportedReason::UnsupportedExpression {
            kind: super::expr_kind(&other),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_declines_on_branch_cut() {
        let pool = crate::kernel::ExprPool::new();
        let expr = pool.func("arg", vec![pool.integer(-1_i32)]);
        assert_eq!(
            eval_complex_f64(expr, &pool, &HashMap::new())
                .unwrap_err()
                .reason,
            UnsupportedReason::UnsupportedExpression { kind: "branch_cut" }
        );
    }

    #[test]
    fn imaginary_unit_auto_binds() {
        let pool = crate::kernel::ExprPool::new();
        let i = pool.imaginary_unit();
        let v = eval_complex_f64(i, &pool, &HashMap::new()).unwrap();
        assert_eq!(v, ComplexF64::new(0.0, 1.0));
        let i2 = pool.mul(vec![i, i]);
        let v2 = eval_complex_f64(i2, &pool, &HashMap::new()).unwrap();
        assert!((v2.re + 1.0).abs() < 1e-12 && v2.im.abs() < 1e-12);
    }

    #[test]
    fn principal_sqrt_of_negative_one() {
        let pool = crate::kernel::ExprPool::new();
        let expr = pool.func("sqrt", vec![pool.integer(-1_i32)]);
        let v = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap();
        assert!((v.re).abs() < 1e-12 && (v.im - 1.0).abs() < 1e-12);
    }

    #[test]
    fn principal_half_power_of_negative_one() {
        let pool = crate::kernel::ExprPool::new();
        let expr = pool.pow(pool.integer(-1_i32), pool.rational(1, 2));
        let v = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap();
        assert!((v.re).abs() < 1e-12 && (v.im - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_of_negative_one_is_i_pi() {
        let pool = crate::kernel::ExprPool::new();
        let expr = pool.func("log", vec![pool.integer(-1_i32)]);
        let v = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap();
        assert!(v.re.abs() < 1e-12 && (v.im - std::f64::consts::PI).abs() < 1e-12);
    }
}
