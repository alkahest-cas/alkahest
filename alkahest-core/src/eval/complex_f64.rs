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
    fn powi(self, n: i64) -> Self {
        if n == 0 {
            return Self::ONE;
        }
        if n < 0 {
            let p = self.powi(-n);
            let d = p.re * p.re + p.im * p.im;
            return Self::new(p.re / d, -p.im / d);
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
        acc
    }
    fn sqrt(self) -> Self {
        let r = (self.re * self.re + self.im * self.im).sqrt();
        let re = ((r + self.re) / 2.0).max(0.0).sqrt();
        let mut im = ((r - self.re) / 2.0).max(0.0).sqrt();
        if self.im < 0.0 {
            im = -im;
        }
        Self::new(re, im)
    }
    fn exp(self) -> Self {
        let s = self.re.exp();
        Self::new(s * self.im.cos(), s * self.im.sin())
    }
    fn ln(self) -> Result<Self, EvalError> {
        if self.re == 0.0 && self.im == 0.0 {
            return Err(error(UnsupportedReason::BranchCutIndeterminate));
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
        if (self.re == 0.0 && self.im == 0.0) || (self.im == 0.0 && self.re <= 0.0) {
            return Err(error(UnsupportedReason::BranchCutIndeterminate));
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
        ExprData::Symbol { .. } => bindings
            .get(&expr)
            .copied()
            .ok_or(error(UnsupportedReason::UnboundSymbol { symbol: expr })),
        ExprData::Add(args) => args.iter().try_fold(ComplexF64::ZERO, |a, &x| {
            Ok(a.add(eval_node(x, pool, bindings)?))
        }),
        ExprData::Mul(args) => args.iter().try_fold(ComplexF64::ONE, |a, &x| {
            Ok(a.mul(eval_node(x, pool, bindings)?))
        }),
        ExprData::Pow { base, exp } => {
            let b = eval_node(base, pool, bindings)?;
            match pool.get(exp) {
                ExprData::Integer(n) => Ok(b.powi(n.0.to_i64().unwrap_or(0))),
                ExprData::Rational(r) if *r.0.denom() == 1 => {
                    Ok(b.powi(r.0.numer().to_i64().unwrap_or(0)))
                }
                _ => Err(error(UnsupportedReason::NonIntegerExponent)),
            }
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let x = eval_node(args[0], pool, bindings)?;
            match name.as_str() {
                "sin" => Ok(x.sin()),
                "cos" => Ok(x.cos()),
                "exp" => Ok(x.exp()),
                "log" => x.ln(),
                "sqrt" => Ok(x.sqrt()),
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
            UnsupportedReason::BranchCutIndeterminate
        );
    }
}
