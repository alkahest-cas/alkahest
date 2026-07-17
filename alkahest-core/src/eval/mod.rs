//! Unified expression evaluation facade.
//!
//! The underlying evaluators deliberately retain their native representations:
//! exact rationals stay exact, `f64` remains a fast approximate mode, and
//! [`IntervalEval`] provides rigorous enclosures.  This module gives callers a
//! single dispatch point and reports unsupported constructs structurally.

mod complex_f64;

use crate::ball::{ArbBall, IntervalEval};
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Rational;
use std::collections::HashMap;
use std::fmt;

pub use complex_f64::{eval_complex_f64, ComplexF64};

/// Input bindings and representation selected for an evaluation.
///
/// Complex evaluation is intentionally a separate entry point
/// ([`eval_complex_f64`]) so this enum stays semver-compatible without a
/// major bump when the complex path lands.
pub enum EvalMode<'a> {
    /// Exact evaluation over rational numbers.  Float literals and
    /// transcendental functions are rejected.
    ExactRational(&'a HashMap<ExprId, Rational>),
    /// Fast approximate evaluation using IEEE-754 double precision.
    F64(&'a HashMap<ExprId, f64>),
    /// Rigorous ball evaluation through the existing [`IntervalEval`] engine.
    Interval(&'a IntervalEval),
}

/// Value returned by [`evaluate`].
#[derive(Clone, Debug, PartialEq)]
pub enum EvalValue {
    Rational(Rational),
    F64(f64),
    Interval(ArbBall),
}

/// A structured reason why an expression cannot be evaluated in a mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnsupportedReason {
    UnboundSymbol {
        symbol: ExprId,
    },
    FloatLiteralInExactMode,
    NonIntegerExponent,
    ZeroToNegativePower,
    UnsupportedFunction {
        name: String,
    },
    UnsupportedExpression {
        kind: &'static str,
    },
    InvalidPredicateArity {
        kind: PredicateKind,
        expected: usize,
        actual: usize,
    },
    IndeterminatePredicate,
    NonFiniteResult,
    IntervalEvaluationFailed,
}

impl UnsupportedReason {
    /// Stable machine-readable code for an unsupported evaluation outcome.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::UnboundSymbol { .. } => "E-EVAL-001",
            Self::FloatLiteralInExactMode => "E-EVAL-002",
            Self::NonIntegerExponent => "E-EVAL-003",
            Self::ZeroToNegativePower => "E-EVAL-004",
            Self::UnsupportedFunction { .. } => "E-EVAL-005",
            Self::UnsupportedExpression { .. } => "E-EVAL-006",
            Self::InvalidPredicateArity { .. } => "E-EVAL-007",
            Self::IndeterminatePredicate => "E-EVAL-008",
            Self::NonFiniteResult => "E-EVAL-009",
            Self::IntervalEvaluationFailed => "E-EVAL-010",
        }
    }

    /// Agent-facing error code, including complex branch-cut declines that
    /// reuse [`UnsupportedExpression`] without a breaking enum variant.
    pub fn agent_code(&self) -> &'static str {
        match self {
            Self::UnsupportedExpression { kind: "branch_cut" } => "E-EVAL-011",
            other => other.code(),
        }
    }
}

/// Evaluation failed because the requested mode cannot represent an operation
/// or establish a required precondition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalError {
    pub reason: UnsupportedReason,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "evaluation unsupported: {:?}", self.reason)
    }
}

impl std::error::Error for EvalError {}

/// Evaluate an expression in the representation selected by `mode`.
pub fn evaluate(expr: ExprId, pool: &ExprPool, mode: EvalMode<'_>) -> Result<EvalValue, EvalError> {
    match mode {
        EvalMode::ExactRational(bindings) => {
            eval_exact_rational(expr, pool, bindings).map(EvalValue::Rational)
        }
        EvalMode::F64(bindings) => eval_f64(expr, pool, bindings).map(EvalValue::F64),
        EvalMode::Interval(eval) => eval
            .eval(expr, pool)
            .map(EvalValue::Interval)
            .ok_or(error(UnsupportedReason::IntervalEvaluationFailed)),
    }
}

/// Evaluate using exact rational arithmetic.
pub fn eval_exact_rational(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, Rational>,
) -> Result<Rational, EvalError> {
    eval_rational_node(expr, pool, bindings)
}

/// Evaluate using IEEE-754 double precision.
pub fn eval_f64(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, f64>,
) -> Result<f64, EvalError> {
    let result = eval_f64_node(expr, pool, bindings)?;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(error(UnsupportedReason::NonFiniteResult))
    }
}

/// Evaluate using the existing rigorous interval evaluator.
pub fn eval_interval(
    expr: ExprId,
    pool: &ExprPool,
    eval: &IntervalEval,
) -> Result<ArbBall, EvalError> {
    evaluate(expr, pool, EvalMode::Interval(eval)).map(|value| match value {
        EvalValue::Interval(ball) => ball,
        _ => unreachable!("interval mode always returns an interval"),
    })
}

fn error(reason: UnsupportedReason) -> EvalError {
    EvalError { reason }
}

fn eval_rational_node(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, Rational>,
) -> Result<Rational, EvalError> {
    match pool.get(expr) {
        ExprData::Integer(n) => Ok(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Ok(r.0.clone()),
        ExprData::Float(_) => Err(error(UnsupportedReason::FloatLiteralInExactMode)),
        ExprData::Symbol { .. } => bindings
            .get(&expr)
            .cloned()
            .ok_or(error(UnsupportedReason::UnboundSymbol { symbol: expr })),
        ExprData::Add(args) => {
            let mut sum = Rational::from(0);
            for arg in args {
                sum += eval_rational_node(arg, pool, bindings)?;
            }
            Ok(sum)
        }
        ExprData::Mul(args) => {
            let mut product = Rational::from(1);
            for arg in args {
                product *= eval_rational_node(arg, pool, bindings)?;
            }
            Ok(product)
        }
        ExprData::Pow { base, exp } => {
            let base = eval_rational_node(base, pool, bindings)?;
            let exponent = integer_exponent(exp, pool)?;
            rational_pow(base, exponent)
        }
        ExprData::Piecewise { branches, default } => {
            for (condition, value) in branches {
                if eval_rational_predicate(condition, pool, bindings)? {
                    return eval_rational_node(value, pool, bindings);
                }
            }
            eval_rational_node(default, pool, bindings)
        }
        ExprData::Predicate { .. } => Ok(Rational::from(eval_rational_predicate(
            expr, pool, bindings,
        )? as i32)),
        ExprData::Func { name, .. } => Err(error(UnsupportedReason::UnsupportedFunction {
            name: name.clone(),
        })),
        other => Err(error(UnsupportedReason::UnsupportedExpression {
            kind: expr_kind(&other),
        })),
    }
}

fn integer_exponent(expr: ExprId, pool: &ExprPool) -> Result<i64, EvalError> {
    match pool.get(expr) {
        ExprData::Integer(n) => {
            n.0.to_i64()
                .ok_or(error(UnsupportedReason::NonIntegerExponent))
        }
        ExprData::Rational(r) if *r.0.denom() == 1 => {
            r.0.numer()
                .to_i64()
                .ok_or(error(UnsupportedReason::NonIntegerExponent))
        }
        _ => Err(error(UnsupportedReason::NonIntegerExponent)),
    }
}

fn rational_pow(mut base: Rational, exponent: i64) -> Result<Rational, EvalError> {
    if exponent < 0 && base == 0 {
        return Err(error(UnsupportedReason::ZeroToNegativePower));
    }
    let mut result = Rational::from(1);
    let mut power = exponent.unsigned_abs();
    while power != 0 {
        if power & 1 == 1 {
            result *= &base;
        }
        power >>= 1;
        if power != 0 {
            base *= base.clone();
        }
    }
    if exponent < 0 {
        Ok(Rational::from(1) / result)
    } else {
        Ok(result)
    }
}

fn eval_rational_predicate(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, Rational>,
) -> Result<bool, EvalError> {
    let ExprData::Predicate { kind, args } = pool.get(expr) else {
        return Err(error(UnsupportedReason::IndeterminatePredicate));
    };
    match kind {
        PredicateKind::True => check_arity(&kind, &args, 0).map(|_| true),
        PredicateKind::False => check_arity(&kind, &args, 0).map(|_| false),
        PredicateKind::Not => Ok(!eval_rational_predicate(
            predicate_arg(&kind, &args, 0)?,
            pool,
            bindings,
        )?),
        PredicateKind::And => {
            for &arg in &args {
                if !eval_rational_predicate(arg, pool, bindings)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateKind::Or => {
            for &arg in &args {
                if eval_rational_predicate(arg, pool, bindings)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        PredicateKind::Lt
        | PredicateKind::Le
        | PredicateKind::Gt
        | PredicateKind::Ge
        | PredicateKind::Eq
        | PredicateKind::Ne => {
            check_arity(&kind, &args, 2)?;
            let lhs = eval_rational_node(args[0], pool, bindings)?;
            let rhs = eval_rational_node(args[1], pool, bindings)?;
            Ok(match kind {
                PredicateKind::Lt => lhs < rhs,
                PredicateKind::Le => lhs <= rhs,
                PredicateKind::Gt => lhs > rhs,
                PredicateKind::Ge => lhs >= rhs,
                PredicateKind::Eq => lhs == rhs,
                PredicateKind::Ne => lhs != rhs,
                _ => unreachable!(),
            })
        }
    }
}

fn eval_f64_node(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, f64>,
) -> Result<f64, EvalError> {
    match pool.get(expr) {
        ExprData::Integer(n) => Ok(n.0.to_f64()),
        ExprData::Rational(r) => Ok(r.0.to_f64()),
        ExprData::Float(f) => Ok(f.inner.to_f64()),
        ExprData::Symbol { .. } => bindings
            .get(&expr)
            .copied()
            .ok_or(error(UnsupportedReason::UnboundSymbol { symbol: expr })),
        ExprData::Add(args) => {
            let mut sum = 0.0;
            for arg in args {
                sum += eval_f64_node(arg, pool, bindings)?;
            }
            Ok(sum)
        }
        ExprData::Mul(args) => {
            let mut product = 1.0;
            for arg in args {
                product *= eval_f64_node(arg, pool, bindings)?;
            }
            Ok(product)
        }
        ExprData::Pow { base, exp } => {
            Ok(eval_f64_node(base, pool, bindings)?.powf(eval_f64_node(exp, pool, bindings)?))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let arg = eval_f64_node(args[0], pool, bindings)?;
            match name.as_str() {
                "sin" => Ok(arg.sin()),
                "cos" => Ok(arg.cos()),
                "exp" => Ok(arg.exp()),
                "log" => Ok(arg.ln()),
                "sqrt" => Ok(arg.sqrt()),
                _ => Err(error(UnsupportedReason::UnsupportedFunction {
                    name: name.clone(),
                })),
            }
        }
        ExprData::Func { name, .. } => Err(error(UnsupportedReason::UnsupportedFunction {
            name: name.clone(),
        })),
        ExprData::Piecewise { branches, default } => {
            for (condition, value) in branches {
                if eval_f64_predicate(condition, pool, bindings)? {
                    return eval_f64_node(value, pool, bindings);
                }
            }
            eval_f64_node(default, pool, bindings)
        }
        ExprData::Predicate { .. } => Ok(eval_f64_predicate(expr, pool, bindings)? as i32 as f64),
        other => Err(error(UnsupportedReason::UnsupportedExpression {
            kind: expr_kind(&other),
        })),
    }
}

fn eval_f64_predicate(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, f64>,
) -> Result<bool, EvalError> {
    let ExprData::Predicate { kind, args } = pool.get(expr) else {
        return Err(error(UnsupportedReason::IndeterminatePredicate));
    };
    match kind {
        PredicateKind::True => check_arity(&kind, &args, 0).map(|_| true),
        PredicateKind::False => check_arity(&kind, &args, 0).map(|_| false),
        PredicateKind::Not => Ok(!eval_f64_predicate(
            predicate_arg(&kind, &args, 0)?,
            pool,
            bindings,
        )?),
        PredicateKind::And => {
            for &arg in &args {
                if !eval_f64_predicate(arg, pool, bindings)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateKind::Or => {
            for &arg in &args {
                if eval_f64_predicate(arg, pool, bindings)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        PredicateKind::Lt
        | PredicateKind::Le
        | PredicateKind::Gt
        | PredicateKind::Ge
        | PredicateKind::Eq
        | PredicateKind::Ne => {
            check_arity(&kind, &args, 2)?;
            let lhs = eval_f64_node(args[0], pool, bindings)?;
            let rhs = eval_f64_node(args[1], pool, bindings)?;
            Ok(match kind {
                PredicateKind::Lt => lhs < rhs,
                PredicateKind::Le => lhs <= rhs,
                PredicateKind::Gt => lhs > rhs,
                PredicateKind::Ge => lhs >= rhs,
                PredicateKind::Eq => lhs == rhs,
                PredicateKind::Ne => lhs != rhs,
                _ => unreachable!(),
            })
        }
    }
}

pub(crate) fn check_arity(
    kind: &PredicateKind,
    args: &[ExprId],
    expected: usize,
) -> Result<(), EvalError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(error(UnsupportedReason::InvalidPredicateArity {
            kind: kind.clone(),
            expected,
            actual: args.len(),
        }))
    }
}

pub(crate) fn predicate_arg(
    kind: &PredicateKind,
    args: &[ExprId],
    index: usize,
) -> Result<ExprId, EvalError> {
    check_arity(kind, args, 1)?;
    Ok(args[index])
}

pub(crate) fn expr_kind(expr: &ExprData) -> &'static str {
    match expr {
        ExprData::Forall { .. } => "Forall",
        ExprData::Exists { .. } => "Exists",
        ExprData::BigO(_) => "BigO",
        ExprData::RootSum { .. } => "RootSum",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ball::ArbBall;
    use crate::kernel::Domain;

    #[test]
    fn exact_rational_mode_preserves_fractional_result() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![pool.rational(1, 3), x]);
        let bindings = HashMap::from([(x, Rational::from((1, 6)))]);

        assert_eq!(
            eval_exact_rational(expr, &pool, &bindings).unwrap(),
            Rational::from((1, 2))
        );
    }

    #[test]
    fn f64_mode_evaluates_transcendental_function() {
        let pool = ExprPool::new();
        let expr = pool.func("sqrt", vec![pool.integer(9_i32)]);

        let result = eval_f64(expr, &pool, &HashMap::new()).unwrap();
        assert_eq!(result, 3.0);
    }

    #[test]
    fn exact_mode_rejects_float_literal_structurally() {
        let pool = ExprPool::new();
        let expr = pool.float(0.5, 53);

        assert_eq!(
            eval_exact_rational(expr, &pool, &HashMap::new())
                .unwrap_err()
                .reason,
            UnsupportedReason::FloatLiteralInExactMode
        );
    }

    #[test]
    fn facade_interval_mode_refuses_threshold_spanning_piecewise() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.piecewise(
            vec![(pool.pred_ge(x, pool.integer(0_i32)), pool.integer(1_i32))],
            pool.integer(-1_i32),
        );
        let mut interval = IntervalEval::new(128);
        interval.bind(x, ArbBall::from_midpoint_radius(0.0, 1.0, 128));

        assert_eq!(
            eval_interval(expr, &pool, &interval).unwrap_err().reason,
            UnsupportedReason::IntervalEvaluationFailed
        );
    }
}
