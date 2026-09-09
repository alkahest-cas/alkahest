//! Constant folding helpers for predicates and numeric evaluation.

use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::float::Round;
use rug::{Float, Integer, Rational};

/// The exact integer `n` rounded to the nearest `f64` (ties to even).
///
/// `rug::Integer::to_f64` rounds **towards zero** (documented, and its own
/// doctest asserts it). Every evaluator in the crate used it directly, so
/// `10^30` — an integer any exact computation can produce — evaluated to
/// `9.999999999999999e29` rather than `1e30`: one ulp low, and *biased*, since
/// truncation never rounds up. `emit_expr_c` writes the exact decimal and lets
/// the C compiler round it correctly, so the interpreter and the emitted C
/// disagreed on the same literal.
///
/// Values with at most 53 significant bits are exact in `f64` and take the
/// cheap path; everything else goes through MPFR at 53 bits, subnormalised so
/// that a result below `2^-1022` is rounded once rather than twice.
pub fn integer_to_f64(n: &Integer) -> f64 {
    if n.significant_bits() <= f64::MANTISSA_DIGITS {
        // Exactly representable: `to_f64` cannot round at all.
        return n.to_f64();
    }
    let (mut f, dir) = Float::with_val_round(f64::MANTISSA_DIGITS, n, Round::Nearest);
    f.subnormalize_ieee_round(dir, Round::Nearest);
    f.to_f64()
}

/// The exact rational `r` rounded to the nearest `f64` (ties to even).
///
/// Two defects this replaces, both reachable from ordinary use:
///
/// * `rug::Rational::to_f64` rounds **towards zero**, so `evaluate(mode="f64")`
///   returned `0.39999999999999997` for `2/5` where the nearest double is `0.4`.
/// * The interpreter, the Cranelift backend and the C emitter instead computed
///   `numer.to_f64() / denom.to_f64()`, which is correctly rounded only while
///   both parts are finite. A rational whose numerator *and* denominator
///   overflow `f64` — coprime, so no cancellation saves it — gave `inf/inf`,
///   i.e. `NaN`, for a value as ordinary as `3/2`.
pub fn rational_to_f64(r: &Rational) -> f64 {
    let (numer, denom) = (r.numer(), r.denom());
    if numer.significant_bits() <= f64::MANTISSA_DIGITS
        && denom.significant_bits() <= f64::MANTISSA_DIGITS
    {
        // Both operands are exact in `f64`, so IEEE-754 division returns the
        // correctly rounded quotient — the same value MPFR would produce.
        return numer.to_f64() / denom.to_f64();
    }
    // `Float::to_f64` rounds to nearest, and `subnormalize_ieee_round` carries
    // the direction of the first rounding into the subnormal range, so the two
    // roundings compose into the single correct one.
    let (mut f, dir) = Float::with_val_round(f64::MANTISSA_DIGITS, r, Round::Nearest);
    f.subnormalize_ieee_round(dir, Round::Nearest);
    f.to_f64()
}

/// If *expr* is a numeric constant, return its `f64` value.
pub fn try_expr_f64(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(integer_to_f64(&n.0)),
        ExprData::Rational(r) => Some(rational_to_f64(&r.0)),
        ExprData::Float(f) => Some(f.inner.to_f64()),
        _ => None,
    }
}

/// Evaluate a predicate when all arguments are numeric constants.
pub fn try_predicate_bool(kind: &PredicateKind, args: &[ExprId], pool: &ExprPool) -> Option<bool> {
    match kind {
        PredicateKind::True => Some(true),
        PredicateKind::False => Some(false),
        PredicateKind::Not => {
            let inner = try_predicate_bool_from_expr(args[0], pool)?;
            Some(!inner)
        }
        PredicateKind::And => {
            for &a in args {
                if !try_predicate_bool_from_expr(a, pool)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        PredicateKind::Or => {
            for &a in args {
                if try_predicate_bool_from_expr(a, pool)? {
                    return Some(true);
                }
            }
            Some(false)
        }
        PredicateKind::Lt => Some(try_expr_f64(args[0], pool)? < try_expr_f64(args[1], pool)?),
        PredicateKind::Le => Some(try_expr_f64(args[0], pool)? <= try_expr_f64(args[1], pool)?),
        PredicateKind::Gt => Some(try_expr_f64(args[0], pool)? > try_expr_f64(args[1], pool)?),
        PredicateKind::Ge => Some(try_expr_f64(args[0], pool)? >= try_expr_f64(args[1], pool)?),
        PredicateKind::Eq => Some(try_expr_f64(args[0], pool)? == try_expr_f64(args[1], pool)?),
        PredicateKind::Ne => Some(try_expr_f64(args[0], pool)? != try_expr_f64(args[1], pool)?),
    }
}

/// Evaluate a predicate expression node (may be nested `And`/`Or` trees).
pub fn try_predicate_bool_from_expr(expr: ExprId, pool: &ExprPool) -> Option<bool> {
    match pool.get(expr) {
        ExprData::Predicate { kind, args } => try_predicate_bool(&kind, &args, pool),
        _ => None,
    }
}

#[cfg(test)]
mod conversion_tests {
    use super::*;
    use rug::ops::Pow;

    /// Reference values computed independently: Python's `float(int)` and
    /// `float(Fraction(n, d))` are correctly rounded (round-half-even) by
    /// construction, so their `repr`s below are the exact nearest doubles.
    #[test]
    fn integer_to_f64_rounds_to_nearest_not_towards_zero() {
        // `Integer::to_f64` truncates. 10^30 needs 70 significant bits, so it
        // is not exact in `f64`; truncation lands one ulp below the nearest.
        let n = Integer::from(10).pow(30);
        assert_eq!(n.to_f64(), 9.999999999999999e29, "rug still truncates");
        assert_eq!(integer_to_f64(&n), 1e30);

        let n = Integer::from(2u64).pow(70) - 1u32;
        assert_eq!(integer_to_f64(&n), 1.1805916207174113e21);

        let n: Integer = "12345678901234567890123456789".parse().unwrap();
        assert_eq!(integer_to_f64(&n), 1.2345678901234568e28);
        assert_eq!(integer_to_f64(&-n.clone()), -1.2345678901234568e28);

        // Exact cases must be untouched by the slow path.
        for k in [0i64, 1, -1, 42, -7, 1 << 52, -(1 << 52)] {
            assert_eq!(integer_to_f64(&Integer::from(k)), k as f64, "k = {k}");
        }
        // 2^53 + 1 is the classic tie: nearest is 2^53 either way.
        let n = Integer::from(1u64 << 53) + 1u32;
        assert_eq!(integer_to_f64(&n), 9007199254740992.0);
    }

    #[test]
    fn integer_to_f64_saturates_past_dbl_max() {
        let huge = Integer::from(10).pow(400);
        assert_eq!(integer_to_f64(&huge), f64::INFINITY);
        assert_eq!(integer_to_f64(&-huge.clone()), f64::NEG_INFINITY);
    }

    #[test]
    fn rational_to_f64_rounds_to_nearest_not_towards_zero() {
        // 2/5 is the shortest counterexample: 0.4 is the nearest double and
        // `Rational::to_f64` returns the one below it.
        let r = Rational::from((2, 5));
        assert_eq!(r.to_f64(), 0.39999999999999997, "rug still truncates");
        assert_eq!(rational_to_f64(&r), 0.4);

        for (n, d, want) in [
            (-7i64, 3i64, -2.3333333333333335f64),
            (7, 3, 2.3333333333333335),
            (-5, 3, -1.6666666666666667),
            (355, 113, 3.1415929203539825),
            (1, 3, 0.3333333333333333),
            (22, 7, 3.142857142857143),
        ] {
            assert_eq!(rational_to_f64(&Rational::from((n, d))), want, "{n}/{d}");
        }
    }

    #[test]
    fn rational_to_f64_survives_numerator_and_denominator_both_overflowing() {
        // (3·10^400 + 1) / (2·10^400) is 1.5 to well past `f64` precision, and
        // the parts are coprime so nothing cancels. `numer/denom` in `f64` is
        // `inf/inf` — `NaN`, which `eval_expr` then reports as "undefined here".
        let numer = Integer::from(3) * Integer::from(10).pow(400) + 1u32;
        let denom = Integer::from(2) * Integer::from(10).pow(400);
        assert!(numer.to_f64().is_infinite() && denom.to_f64().is_infinite());
        assert!(
            (numer.to_f64() / denom.to_f64()).is_nan(),
            "the old formula"
        );

        let r = Rational::from((numer, denom));
        assert_eq!(rational_to_f64(&r), 1.5);
    }

    #[test]
    fn rational_to_f64_handles_subnormals_and_overflow() {
        // 1 / 10^400 underflows to zero; 10^400 overflows to +inf.
        let big = Integer::from(10).pow(400);
        assert_eq!(rational_to_f64(&Rational::from((1, big.clone()))), 0.0);
        assert_eq!(rational_to_f64(&Rational::from((big, 1))), f64::INFINITY);

        // 2^-1074 is the smallest positive subnormal and must round to itself
        // rather than to zero.
        let denom = Integer::from(2u32).pow(1074);
        assert_eq!(rational_to_f64(&Rational::from((1, denom))), 5e-324);
    }
}
