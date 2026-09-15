//! IEEE-754 complex evaluation for `re` / `im` / `conjugate` / `arg`.

use crate::kernel::{
    integer_to_f64, pow_f64_integer_exponent, rational_to_f64, ExprData, ExprId, ExprPool,
};
use rug::Integer;
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
    /// `self ^ n` for an exact integer `n` of **any** width.
    ///
    /// The `i64` arms this replaces read `n.to_i64().unwrap_or(0)`, so an
    /// exponent wider than `i64` — `x ** (10**30 + 1)`, which the pool holds
    /// exactly — silently became the exponent **0**, and every such power
    /// evaluated to `1 + 0i`.
    ///
    /// Beyond `i64` the answer splits in two:
    ///
    /// * a **real** base still has one: `z^n = ±|z|^n`, and the sign is the
    ///   parity of `n`, which the exact integer knows (see
    ///   [`pow_f64_integer_exponent`]);
    /// * a base off the real axis does not. `z^n = |z|^n · e^{i·n·θ}`, and
    ///   with `n` past `2^63` the reduction of `n·θ` modulo `2π` has no
    ///   correct digits left in double precision. That is a refusal
    ///   (`E-EVAL-012`), not a number.
    fn powi_big(self, n: &Integer) -> Result<Self, EvalError> {
        if let Some(k) = n.to_i64() {
            return self.powi(k);
        }
        if self.im == 0.0 {
            return Ok(Self::new(pow_f64_integer_exponent(self.re, n), 0.0));
        }
        Err(error(UnsupportedReason::UnsupportedExpression {
            kind: "unrepresentable_exponent",
        }))
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
    /// `acos z = π/2 + i·Log(iz + √(1 − z²))`, principal branch.
    ///
    /// Present because `matrix::eigen`'s casus-irreducibilis form —
    /// `2√(−p/3)·cos((acos c + 2πk)/3)`, the *correct* closed form for a cubic
    /// with three real roots — is otherwise a spectrum
    /// [`crate::matrix::spectrum`]'s check cannot evaluate, and an unevaluable
    /// spectrum is no information rather than a confirmation.
    fn acos(self) -> Result<Self, EvalError> {
        let one = Self::new(1.0, 0.0);
        let i = Self::new(0.0, 1.0);
        let root = one.add(Self::new(-1.0, 0.0).mul(self.mul(self))).sqrt()?;
        let ln = i.mul(self).add(root).ln()?;
        Ok(Self::new(std::f64::consts::FRAC_PI_2, 0.0).add(i.mul(ln)))
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
        ExprData::Integer(n) => Ok(ComplexF64::new(integer_to_f64(&n.0), 0.0)),
        ExprData::Rational(r) => Ok(ComplexF64::new(rational_to_f64(&r.0), 0.0)),
        ExprData::Float(f) => Ok(ComplexF64::new(f.inner.to_f64(), 0.0)),
        ExprData::Symbol { .. } => {
            if let Some(&v) = bindings.get(&expr) {
                Ok(v)
            } else if pool.is_imaginary_unit(expr) {
                // Canonical I evaluates to 0+1j in complex mode without an
                // explicit binding (matches symbolic `i² → −1` folding).
                Ok(ComplexF64::new(0.0, 1.0))
            } else if super::symbols::is_pi(expr, pool) {
                // The other named constant, for the same reason: `pi` is an
                // ordinary symbol in this crate and binding it to a sample is
                // a false-refusal machine.  An explicit binding still wins.
                Ok(ComplexF64::new(std::f64::consts::PI, 0.0))
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
                ExprData::Integer(n) => b.powi_big(&n.0),
                ExprData::Rational(r) if *r.0.denom() == 1 => b.powi_big(r.0.numer()),
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
                "acos" => x.acos(),
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
    fn acos_matches_the_real_branch_inside_the_unit_interval() {
        let pool = crate::kernel::ExprPool::new();
        let expr = pool.func("acos", vec![pool.rational(1, 2)]);
        let v = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap();
        assert!((v.re - std::f64::consts::FRAC_PI_3).abs() < 1e-12, "{v:?}");
        assert!(v.im.abs() < 1e-12, "{v:?}");
    }

    #[test]
    fn acos_continues_off_the_real_interval() {
        let pool = crate::kernel::ExprPool::new();
        // cos(acos 2) must be 2 again, whatever branch the continuation picks.
        let expr = pool.func("cos", vec![pool.func("acos", vec![pool.integer(2_i32)])]);
        let v = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap();
        assert!((v.re - 2.0).abs() < 1e-12 && v.im.abs() < 1e-12, "{v:?}");
    }

    #[test]
    fn complex_mode_keeps_the_parity_of_an_exponent_wider_than_i64() {
        use rug::ops::Pow;
        let pool = crate::kernel::ExprPool::new();
        let n = Integer::from(10).pow(30) + 1u32;
        let expr = pool.pow(pool.integer(-1_i32), pool.integer(n.clone()));

        // `n.to_i64()` is `None`, and the arm this replaces read
        // `.unwrap_or(0)`: the exponent silently became 0 and the answer 1+0i.
        let v = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap();
        assert_eq!((v.re, v.im), (-1.0, 0.0));

        // Control: the even neighbour, and a small exponent.
        let even = pool.pow(pool.integer(-1_i32), pool.integer(n - 1u32));
        let v = eval_complex_f64(even, &pool, &HashMap::new()).unwrap();
        assert_eq!((v.re, v.im), (1.0, 0.0));
        let three = pool.pow(pool.integer(-1_i32), pool.integer(3_i32));
        let v = eval_complex_f64(three, &pool, &HashMap::new()).unwrap();
        assert_eq!((v.re, v.im), (-1.0, 0.0));
    }

    #[test]
    fn complex_mode_refuses_a_wide_exponent_off_the_real_axis() {
        use rug::ops::Pow;
        let pool = crate::kernel::ExprPool::new();
        let i = pool.imaginary_unit();
        let expr = pool.pow(i, pool.integer(Integer::from(10).pow(30) + 1u32));

        // i^n cycles with period 4, but `n·θ mod 2π` past 2^63 has no correct
        // digits in double precision — so this is a refusal, not a number.
        let err = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap_err();
        assert_eq!(err.reason.agent_code(), "E-EVAL-012");

        // Control: i^3 = -i is still answered.
        let expr = pool.pow(i, pool.integer(3_i32));
        let v = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap();
        assert!(v.re.abs() < 1e-12 && (v.im + 1.0).abs() < 1e-12, "{v:?}");
    }

    #[test]
    fn log_of_negative_one_is_i_pi() {
        let pool = crate::kernel::ExprPool::new();
        let expr = pool.func("log", vec![pool.integer(-1_i32)]);
        let v = eval_complex_f64(expr, &pool, &HashMap::new()).unwrap();
        assert!(v.re.abs() < 1e-12 && (v.im - std::f64::consts::PI).abs() < 1e-12);
    }
}
