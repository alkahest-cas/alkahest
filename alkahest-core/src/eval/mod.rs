//! Unified expression evaluation facade.
//!
//! The underlying evaluators deliberately retain their native representations:
//! exact rationals stay exact, `f64` remains a fast approximate mode, and
//! [`IntervalEval`] provides rigorous enclosures.  This module gives callers a
//! single dispatch point and reports unsupported constructs structurally.

mod complex_f64;
mod id_map;
pub(crate) mod program;
pub(crate) mod root_sum;
pub(crate) mod symbols;

use crate::ball::{ArbBall, IntervalEval};
use crate::kernel::expr::PredicateKind;
use crate::kernel::{integer_to_f64, pow_f64, rational_to_f64, ExprData, ExprId, ExprPool};
use rug::{Integer, Rational};
use std::collections::HashMap;
use std::fmt;

pub use complex_f64::{eval_complex_f64, ComplexF64};
pub(crate) use id_map::IdMap;
pub(crate) use root_sum::eval_root_sum_f64;

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

    /// Agent-facing error code, including the declines that reuse
    /// [`UnsupportedReason::UnsupportedExpression`] with a distinguishing
    /// `kind` rather than adding a variant to this public exhaustive enum.
    pub fn agent_code(&self) -> &'static str {
        match self {
            Self::UnsupportedExpression { kind: "branch_cut" } => "E-EVAL-011",
            // An exact integer exponent whose *answer* the requested
            // representation cannot hold: an exact power that would need
            // megabytes of digits, or a complex power whose phase is past the
            // last correct bit of a double. Both are refusals rather than the
            // plausible wrong number the callers used to return.
            Self::UnsupportedExpression {
                kind: "exact_pow_overflow" | "unrepresentable_exponent",
            } => "E-EVAL-012",
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
    // One memo per call: pool expressions are DAGs, and a node shared by two
    // parents used to be evaluated once per *path* to it — exponential on a
    // recurrence such as Chebyshev's `T_{n+1} = 2x·T_n − T_{n−1}`. The memo
    // holds only successes; the first failure still aborts the whole call, so
    // the result (value or error) is the one the tree walk returned.
    let mut memo = RationalMemo::default();
    eval_rational_node(expr, pool, bindings, &mut memo)
}

/// Evaluate using IEEE-754 double precision.
pub fn eval_f64(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, f64>,
) -> Result<f64, EvalError> {
    // Memoized per call for the same reason as [`eval_exact_rational`].
    let mut memo = F64Memo::default();
    let result = eval_f64_node(expr, pool, bindings, &mut memo)?;
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

type RationalMemo = IdMap<Rational>;

fn eval_rational_node(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, Rational>,
    memo: &mut RationalMemo,
) -> Result<Rational, EvalError> {
    if let Some(v) = memo.get(&expr) {
        return Ok(v.clone());
    }
    let (value, interior) = pool.with(expr, |data| {
        let interior = !matches!(
            data,
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Symbol { .. }
        );
        eval_rational_data(expr, data, pool, bindings, memo).map(|v| (v, interior))
    })?;
    // A leaf is as cheap to rebuild as to look up; memoize interior nodes.
    if interior {
        memo.insert(expr, value.clone());
    }
    Ok(value)
}

fn eval_rational_data(
    expr: ExprId,
    data: &ExprData,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, Rational>,
    memo: &mut RationalMemo,
) -> Result<Rational, EvalError> {
    match data {
        ExprData::Integer(n) => Ok(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Ok(r.0.clone()),
        ExprData::Float(_) => Err(error(UnsupportedReason::FloatLiteralInExactMode)),
        // `π` is **not** resolved here, unlike in every other mode: this
        // evaluator returns exact rationals, and π is not one.  Handing back a
        // 53-bit approximation under the name "exact" is the failure this mode
        // exists to prevent, so an unbound `pi` stays `E-EVAL-001`.
        ExprData::Symbol { .. } => bindings
            .get(&expr)
            .cloned()
            .ok_or(error(UnsupportedReason::UnboundSymbol { symbol: expr })),
        ExprData::Add(args) => {
            let mut sum = Rational::from(0);
            for &arg in args {
                sum += eval_rational_node(arg, pool, bindings, memo)?;
            }
            Ok(sum)
        }
        ExprData::Mul(args) => {
            let mut product = Rational::from(1);
            for &arg in args {
                product *= eval_rational_node(arg, pool, bindings, memo)?;
            }
            Ok(product)
        }
        ExprData::Pow { base, exp } => {
            let base = eval_rational_node(*base, pool, bindings, memo)?;
            let exponent = integer_exponent(*exp, pool)?;
            rational_pow(base, &exponent)
        }
        ExprData::Piecewise { branches, default } => {
            for &(condition, value) in branches {
                if eval_rational_predicate(condition, pool, bindings, memo)? {
                    return eval_rational_node(value, pool, bindings, memo);
                }
            }
            eval_rational_node(*default, pool, bindings, memo)
        }
        ExprData::Predicate { .. } => Ok(Rational::from(eval_rational_predicate(
            expr, pool, bindings, memo,
        )? as i32)),
        ExprData::Func { name, .. } => Err(error(UnsupportedReason::UnsupportedFunction {
            name: name.clone(),
        })),
        other => Err(error(UnsupportedReason::UnsupportedExpression {
            kind: expr_kind(other),
        })),
    }
}

/// The largest exact power [`eval_exact_rational`] will build, in bits.
///
/// `b^n` occupies about `n · bits(b)` bits, and GMP **aborts the process**
/// when an allocation that size fails — `evaluate(x**(10**12), {x: 2},
/// mode="exact")` used to die with `GNU MP: Cannot allocate memory` and a core
/// dump, which no `except` can catch.  So the size has to be refused before the
/// multiplication rather than recovered from after it.
///
/// `2^24` bits is 2 MiB, a little over five million decimal digits: past
/// anything `evaluate` can even hand back (`fractions.Fraction` is built from a
/// decimal string, and CPython refuses to parse one over 4300 digits by
/// default) and still under a millisecond of work.
const MAX_EXACT_POW_BITS: u64 = 1 << 24;

/// The exact integer an exponent node holds.
///
/// Unbounded on purpose: this used to return `i64` and reported an exponent of
/// `10**30` — an ordinary integer the pool holds exactly — as
/// [`UnsupportedReason::NonIntegerExponent`], which it is not.  Whether the
/// *result* is affordable is [`rational_pow`]'s question, and it is a different
/// one: `(-1)^(10**30 + 1)` costs nothing.
fn integer_exponent(expr: ExprId, pool: &ExprPool) -> Result<Integer, EvalError> {
    match pool.get(expr) {
        ExprData::Integer(n) => Ok(n.0.clone()),
        ExprData::Rational(r) if *r.0.denom() == 1 => Ok(r.0.numer().clone()),
        _ => Err(error(UnsupportedReason::NonIntegerExponent)),
    }
}

fn rational_pow(mut base: Rational, exponent: &Integer) -> Result<Rational, EvalError> {
    if *exponent == 0 {
        return Ok(Rational::from(1));
    }
    let negative = *exponent < 0;
    // The three bases whose powers cost nothing at any exponent at all. They
    // are also the three the `f64` evaluator gets wrong by rounding, so exact
    // mode should not be the one that gives up on them.
    if base == 0 {
        return if negative {
            Err(error(UnsupportedReason::ZeroToNegativePower))
        } else {
            Ok(Rational::from(0))
        };
    }
    if base == 1 {
        return Ok(Rational::from(1));
    }
    if base == -1 {
        return Ok(Rational::from(if exponent.is_odd() { -1 } else { 1 }));
    }
    // Every other base grows geometrically. `significant_bits() >= 1` here
    // (both parts are non-zero), so a surviving exponent is at most
    // `MAX_EXACT_POW_BITS` and fits `u32` comfortably.
    let bits = base
        .numer()
        .significant_bits()
        .max(base.denom().significant_bits()) as u64;
    let mut power = exponent
        .clone()
        .abs()
        .to_u64()
        .and_then(|n| n.checked_mul(bits))
        .filter(|&cost| cost <= MAX_EXACT_POW_BITS)
        .and_then(|_| exponent.clone().abs().to_u32())
        .ok_or(error(UnsupportedReason::UnsupportedExpression {
            kind: "exact_pow_overflow",
        }))?;

    let mut result = Rational::from(1);
    while power != 0 {
        if power & 1 == 1 {
            result *= &base;
        }
        power >>= 1;
        if power != 0 {
            base *= base.clone();
        }
    }
    if negative {
        Ok(Rational::from(1) / result)
    } else {
        Ok(result)
    }
}

fn eval_rational_predicate(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, Rational>,
    memo: &mut RationalMemo,
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
            memo,
        )?),
        PredicateKind::And => {
            for &arg in &args {
                if !eval_rational_predicate(arg, pool, bindings, memo)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateKind::Or => {
            for &arg in &args {
                if eval_rational_predicate(arg, pool, bindings, memo)? {
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
            let lhs = eval_rational_node(args[0], pool, bindings, memo)?;
            let rhs = eval_rational_node(args[1], pool, bindings, memo)?;
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

/// Per-call memo for [`eval_f64`]: numeric values and predicate truth values
/// of the nodes already evaluated successfully.
#[derive(Default)]
struct F64Memo {
    values: IdMap<f64>,
    predicates: IdMap<bool>,
}

fn eval_f64_node(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, f64>,
    memo: &mut F64Memo,
) -> Result<f64, EvalError> {
    if let Some(&v) = memo.values.get(&expr) {
        return Ok(v);
    }
    // Borrow the node instead of cloning it: `pool.get` copies the argument
    // vector (and a `Func`'s name) on every visit.
    let value = pool.with(expr, |data| eval_f64_data(expr, data, pool, bindings, memo))?;
    memo.values.insert(expr, value);
    Ok(value)
}

fn eval_f64_data(
    expr: ExprId,
    data: &ExprData,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, f64>,
    memo: &mut F64Memo,
) -> Result<f64, EvalError> {
    match data {
        ExprData::Integer(n) => Ok(integer_to_f64(&n.0)),
        ExprData::Rational(r) => Ok(rational_to_f64(&r.0)),
        ExprData::Float(f) => Ok(f.inner.to_f64()),
        // `π` resolves to its own value without a binding; see the module docs
        // of [`symbols`] for why it must never be sampled instead.  An explicit
        // binding still wins, so a caller that deliberately wants `pi` to be a
        // free parameter can still say so.  The imaginary unit is the other
        // named constant and is deliberately *not* resolved here: it has no
        // `f64` value, and `UnboundSymbol` is the honest refusal.
        ExprData::Symbol { name, .. } => bindings
            .get(&expr)
            .copied()
            .or_else(|| (name == symbols::PI_NAME).then_some(std::f64::consts::PI))
            .ok_or(error(UnsupportedReason::UnboundSymbol { symbol: expr })),
        ExprData::Add(args) => {
            let mut sum = 0.0;
            for &arg in args {
                sum += eval_f64_node(arg, pool, bindings, memo)?;
            }
            Ok(sum)
        }
        ExprData::Mul(args) => {
            let mut product = 1.0;
            for &arg in args {
                product *= eval_f64_node(arg, pool, bindings, memo)?;
            }
            Ok(product)
        }
        ExprData::Pow { base, exp } => {
            let b = eval_f64_node(*base, pool, bindings, memo)?;
            let e = eval_f64_node(*exp, pool, bindings, memo)?;
            Ok(pow_f64(b, e, || Some(pool.get(*exp))))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let arg = eval_f64_node(args[0], pool, bindings, memo)?;
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
            for &(condition, value) in branches {
                if eval_f64_predicate(condition, pool, bindings, memo)? {
                    return eval_f64_node(value, pool, bindings, memo);
                }
            }
            eval_f64_node(*default, pool, bindings, memo)
        }
        ExprData::Predicate { .. } => {
            Ok(eval_f64_predicate(expr, pool, bindings, memo)? as i32 as f64)
        }
        other => Err(error(UnsupportedReason::UnsupportedExpression {
            kind: expr_kind(other),
        })),
    }
}

fn eval_f64_predicate(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, f64>,
    memo: &mut F64Memo,
) -> Result<bool, EvalError> {
    if let Some(&b) = memo.predicates.get(&expr) {
        return Ok(b);
    }
    let value = eval_f64_predicate_uncached(expr, pool, bindings, memo)?;
    memo.predicates.insert(expr, value);
    Ok(value)
}

fn eval_f64_predicate_uncached(
    expr: ExprId,
    pool: &ExprPool,
    bindings: &HashMap<ExprId, f64>,
    memo: &mut F64Memo,
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
            memo,
        )?),
        PredicateKind::And => {
            for &arg in &args {
                if !eval_f64_predicate(arg, pool, bindings, memo)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateKind::Or => {
            for &arg in &args {
                if eval_f64_predicate(arg, pool, bindings, memo)? {
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
            let lhs = eval_f64_node(args[0], pool, bindings, memo)?;
            let rhs = eval_f64_node(args[1], pool, bindings, memo)?;
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
    use rug::ops::Pow;

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

    /// `10^30 + 1` is odd; its `f64` image `1e30` is even.
    fn odd_wide_exponent(pool: &ExprPool) -> ExprId {
        pool.integer(Integer::from(10).pow(30) + 1u32)
    }

    #[test]
    fn f64_mode_keeps_the_parity_of_an_exponent_wider_than_f64() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, odd_wide_exponent(&pool));
        let bindings = HashMap::from([(x, -1.0)]);

        // A product of an odd number of factors of −1 is −1. Was `1.0`.
        assert_eq!(eval_f64(expr, &pool, &bindings).unwrap(), -1.0);

        // Control: the neighbouring even exponent, and a small odd one.
        let even = pool.pow(x, pool.integer(Integer::from(10).pow(30)));
        assert_eq!(eval_f64(even, &pool, &bindings).unwrap(), 1.0);
        let three = pool.pow(x, pool.integer(3_i32));
        assert_eq!(eval_f64(three, &pool, &bindings).unwrap(), -1.0);
    }

    #[test]
    fn f64_mode_refuses_a_wide_power_it_cannot_represent() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, odd_wide_exponent(&pool));
        // 2^(10^30) is not a number `f64` has; the sign fix does not pretend
        // otherwise, it just signs the overflow.
        for base in [2.0, -2.0] {
            assert_eq!(
                eval_f64(expr, &pool, &HashMap::from([(x, base)]))
                    .unwrap_err()
                    .reason,
                UnsupportedReason::NonFiniteResult,
                "base {base}"
            );
        }
    }

    #[test]
    fn exact_mode_answers_a_wide_exponent_over_a_base_that_costs_nothing() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, odd_wide_exponent(&pool));

        // Used to be `E-EVAL-003` ("only integer exponents"), for an exponent
        // that is an integer.
        for (base, want) in [(-1, -1), (1, 1), (0, 0)] {
            assert_eq!(
                eval_exact_rational(expr, &pool, &HashMap::from([(x, Rational::from(base))]))
                    .unwrap(),
                Rational::from(want),
                "base {base}"
            );
        }
        let even = pool.pow(x, pool.integer(Integer::from(10).pow(30)));
        assert_eq!(
            eval_exact_rational(even, &pool, &HashMap::from([(x, Rational::from(-1))])).unwrap(),
            Rational::from(1)
        );
    }

    #[test]
    fn exact_mode_refuses_a_power_too_large_to_build_instead_of_aborting() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 2^(10^12) needs 10^12 bits — 125 GB. `rational_pow` used to try, and
        // GMP aborts the *process* on a failed allocation ("GNU MP: Cannot
        // allocate memory"), which no caller can catch.
        for exp in [Integer::from(10).pow(12), Integer::from(10).pow(30) + 1u32] {
            let expr = pool.pow(x, pool.integer(exp.clone()));
            let err = eval_exact_rational(expr, &pool, &HashMap::from([(x, Rational::from(2))]))
                .unwrap_err();
            assert_eq!(err.reason.agent_code(), "E-EVAL-012", "{exp}");
        }

        // Control: a big-but-affordable power still evaluates exactly.
        let expr = pool.pow(x, pool.integer(1000_i32));
        let got =
            eval_exact_rational(expr, &pool, &HashMap::from([(x, Rational::from(2))])).unwrap();
        assert_eq!(*got.numer(), Integer::from(1u32) << 1000);
        assert_eq!(*got.denom(), 1);

        // And a rational base: (2/3)^-3 is 27/8.
        let expr = pool.pow(x, pool.integer(-3_i32));
        assert_eq!(
            eval_exact_rational(expr, &pool, &HashMap::from([(x, Rational::from((2, 3)))]))
                .unwrap(),
            Rational::from((27, 8))
        );
    }

    #[test]
    fn exact_mode_still_rejects_a_genuinely_non_integer_exponent() {
        let pool = ExprPool::new();
        let expr = pool.pow(pool.integer(4_i32), pool.rational(1, 2));
        assert_eq!(
            eval_exact_rational(expr, &pool, &HashMap::new())
                .unwrap_err()
                .reason,
            UnsupportedReason::NonIntegerExponent
        );
    }

    /// Chebyshev recurrence `T_{k+1} = 2x·T_k − T_{k−1}`: `~3n` distinct nodes
    /// but `~fib(n)` root-to-leaf paths, so an unmemoized walk is exponential.
    fn chebyshev(pool: &ExprPool, x: ExprId, n: usize) -> ExprId {
        let (two, minus_one) = (pool.integer(2_i32), pool.integer(-1_i32));
        let (mut a, mut b) = (pool.integer(1_i32), x);
        for _ in 1..n {
            let next = pool.add(vec![
                pool.mul(vec![two, x, b]),
                pool.mul(vec![minus_one, a]),
            ]);
            a = b;
            b = next;
        }
        b
    }

    #[test]
    fn every_mode_evaluates_a_shared_dag_in_linear_time() {
        // T_40 has ~1.6e8 paths: the tree walk this replaces took minutes.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t40 = chebyshev(&pool, x, 40);
        let start = std::time::Instant::now();

        // T_n(cos θ) = cos(nθ).
        let theta = 0.3f64;
        let v = eval_f64(t40, &pool, &HashMap::from([(x, theta.cos())])).unwrap();
        assert!((v - (40.0 * theta).cos()).abs() < 1e-8, "{v}");

        let c = eval_complex_f64(
            t40,
            &pool,
            &HashMap::from([(x, ComplexF64::new(theta.cos(), 0.0))]),
        )
        .unwrap();
        assert!((c.re - (40.0 * theta).cos()).abs() < 1e-8, "{c:?}");

        // T_n(1) = 1 exactly; T_n(1/2) = cos(nπ/3) = −1/2 for n = 40.
        let one = eval_exact_rational(t40, &pool, &HashMap::from([(x, Rational::from(1))]));
        assert_eq!(one.unwrap(), Rational::from(1));
        let half = eval_exact_rational(t40, &pool, &HashMap::from([(x, Rational::from((1, 2)))]));
        assert_eq!(half.unwrap(), Rational::from((-1, 2)));

        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "took {:?}",
            start.elapsed()
        );
    }

    /// Timing probe for the numeric evaluators (not a CI test):
    /// `cargo test --release -p alkahest-cas --lib numeric_eval_timings -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn numeric_eval_timings() {
        use std::time::Instant;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // sin(x)·exp(−x²/2) + x³ − 2xy + y²
        let e = pool.add(vec![
            pool.mul(vec![
                pool.func("sin", vec![x]),
                pool.func(
                    "exp",
                    vec![pool.mul(vec![pool.rational(-1, 2), pool.pow(x, pool.integer(2_i32))])],
                ),
            ]),
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-2_i32), x, y]),
            pool.pow(y, pool.integer(2_i32)),
        ]);
        let n = 200_000;
        let per = |label: &str, f: &mut dyn FnMut(f64) -> f64| {
            let mut best = f64::INFINITY;
            let mut sink = 0.0;
            for _ in 0..5 {
                let t = Instant::now();
                for i in 0..n {
                    sink += f(i as f64 * 1e-6);
                }
                best = best.min(t.elapsed().as_secs_f64() / n as f64);
            }
            println!("{label:44} {:8.1} ns/point  (sink {sink:.3})", best * 1e9);
        };
        per("eval::eval_f64 (small)", &mut |v| {
            eval_f64(e, &pool, &HashMap::from([(x, v), (y, 0.7)])).unwrap()
        });
        per("jit::eval_interp (small)", &mut |v| {
            crate::jit::eval_interp(e, &HashMap::from([(x, v), (y, 0.7)]), &pool).unwrap()
        });
        let f = crate::jit::compile(e, &[x, y], &pool).unwrap();
        per("CompiledFn::call, interpreter tier (small)", &mut |v| {
            f.call(&[v, 0.7])
        });
        let prog = program::NumericProgram::compile(e, &[x, y], &pool).unwrap();
        let mut scratch = prog.scratch();
        per("NumericProgram::eval_in (small)", &mut |v| {
            prog.eval_in(&[v, 0.7], &mut scratch).unwrap()
        });
        let s = crate::jit::Sampler::new(e, x, &HashMap::from([(y, 0.7)]), &pool);
        per("jit::Sampler::eval (small)", &mut |v| s.eval(v).unwrap());
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 1e-6).collect();
        let ys = vec![0.7; n];
        let mut out = vec![0.0; n];
        let t = Instant::now();
        f.call_batch(&[&xs, &ys], &mut out);
        println!(
            "{:44} {:8.1} ns/point",
            "CompiledFn::call_batch, interpreter tier",
            t.elapsed().as_secs_f64() / n as f64 * 1e9
        );

        for k in [16usize, 20, 24, 28, 40] {
            let t_k = chebyshev(&pool, x, k);
            let t = Instant::now();
            let v = eval_f64(t_k, &pool, &HashMap::from([(x, 0.3)])).unwrap();
            println!("eval_f64(T_{k}) {:?} ({v:.6})", t.elapsed());
        }
    }

    /// The memo must not change *which* error a failing evaluation reports:
    /// the first failure in evaluation order still wins.
    #[test]
    fn memoized_walk_reports_the_same_first_failure() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        // `Pow` evaluates base before exponent and its operands are not
        // reordered by the pool, so the first failure is the base's.
        let shared = pool.func("sin", vec![x]);
        let base = pool.func("zeta_unknown", vec![shared]);
        let expr = pool.pow(pool.pow(base, pool.add(vec![shared, z])), shared);
        let err = eval_f64(expr, &pool, &HashMap::from([(x, 0.5)])).unwrap_err();
        assert_eq!(
            err.reason,
            UnsupportedReason::UnsupportedFunction {
                name: "zeta_unknown".into()
            }
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
