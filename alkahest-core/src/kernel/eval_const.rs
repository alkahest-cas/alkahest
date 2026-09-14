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

/// `2^53` — the smallest magnitude at which an `f64` can fail to be the exact
/// image of the integer it was rounded from.
///
/// Below it, every integer is its own `f64`, so the `f64` carries the
/// integer's parity and `powf` needs no help.
const EXACT_INTEGER_F64_LIMIT: f64 = 9_007_199_254_740_992.0;

/// The exact integer an exponent node holds, if it holds one.
fn exact_integer_node(data: &ExprData) -> Option<&Integer> {
    match data {
        ExprData::Integer(n) => Some(&n.0),
        // `ExprPool::rational` interns whole values as `Integer`, so this arm
        // is unreachable through the pool — it is here because every other
        // evaluator in the crate accepts a `Rational` exponent with a unit
        // denominator and disagreeing about that would be a new bug.
        ExprData::Rational(r) if *r.0.denom() == 1 => Some(r.0.numer()),
        _ => None,
    }
}

/// True when `n` is its own `f64` — nothing is lost converting it, and every
/// consumer that reduces it to a double still has the whole integer.
///
/// `significant_bits() <= 53` is sufficient but not necessary: `2^60` has 61
/// bits and is still exact.  Round-trip it rather than counting.
pub fn integer_is_exact_f64(n: &Integer) -> bool {
    n.significant_bits() <= f64::MANTISSA_DIGITS
        || Integer::from_f64(integer_to_f64(n)).as_ref() == Some(n)
}

/// `base ^ exp` where `exp` is the `f64` image of an exponent whose *exact*
/// value the node still holds.
///
/// # The defect this exists to close
///
/// `ExprPool::integer` is `rug`-backed and unbounded, so `x ** (10**30 + 1)`
/// interns the exponent exactly.  Every `f64` evaluator then reduced it to
/// `1e30` before computing, and `1e30` is **even**: `eval_expr(x ** (10**30 +
/// 1), {x: -1})` returned `1.0` where the answer is `-1`.  A clean, plausible,
/// confidently-returned number with the wrong sign.
///
/// What rounding an integer exponent to `f64` destroys is exactly one thing
/// that changes the *answer* rather than its precision: the parity, and only
/// for a negative base.  So that is the only thing this corrects — the
/// magnitude is still `|base|.powf(exp)`, whose relative error is bounded by
/// `|ln result| · 2⁻⁵³` (below `8e-14` anywhere in the `f64` range, and exactly
/// zero when `|base|` is 1, which is the case the sign actually matters in).
///
/// # Boundary
///
/// Where the true magnitude is not representable this returns `±inf` or `±0`
/// just as IEEE `pow` does, correctly signed.  Every *checked* entry point
/// ([`crate::eval::eval_f64`], [`crate::jit::eval_interp_checked`], and so
/// `alkahest.eval_expr`) turns the infinity into `E-EVAL-009` rather than a
/// number: `(-1) ** (10**30 + 1)` is `-1`, and `2 ** (10**30)` is a refusal.
///
/// `exp_node` is fetched lazily: on the hot path — any exponent below `2^53`,
/// which is every exponent anyone writes — this costs one compare and one
/// predictable branch, and the node is never looked at.
#[inline]
pub fn pow_f64(base: f64, exp: f64, exp_node: impl FnOnce() -> Option<ExprData>) -> f64 {
    if exp.abs() < EXACT_INTEGER_F64_LIMIT {
        return base.powf(exp);
    }
    pow_f64_wide_exponent(base, exp, exp_node())
}

#[cold]
fn pow_f64_wide_exponent(base: f64, exp: f64, exp_node: Option<ExprData>) -> f64 {
    match exp_node.as_ref().and_then(exact_integer_node) {
        Some(n) => pow_f64_integer_exponent(base, n),
        // A genuine float exponent (`x ** 1e300`), or a symbol bound to one.
        // Nothing was lost rounding it, because it was never exact.
        None => base.powf(exp),
    }
}

/// `base ^ n` for an exact integer `n` of any width, in `f64`.
///
/// See [`pow_f64`] for what is and is not established by the result.
pub fn pow_f64_integer_exponent(base: f64, n: &Integer) -> f64 {
    let exp = integer_to_f64(n);
    // `is_sign_negative` rather than `< 0.0` so that `-0.0` keeps its sign:
    // `(-0.0) ** odd` is `-0.0`, and `(-0.0).powf(1e30)` is `+0.0`.
    if !base.is_finite() || !base.is_sign_negative() {
        return base.powf(exp);
    }
    let magnitude = (-base).powf(exp);
    if n.is_odd() {
        -magnitude
    } else {
        magnitude
    }
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

    /// The exponent whose `f64` image is even while the exponent is odd.
    fn odd_wide_exponent() -> Integer {
        Integer::from(10).pow(30) + 1u32
    }

    #[test]
    fn pow_f64_keeps_the_parity_of_an_exponent_f64_cannot_hold() {
        let n = odd_wide_exponent();
        // The mechanism: the `f64` image is even, so `powf` says `+1`.
        assert_eq!(integer_to_f64(&n), 1e30);
        assert_eq!((-1.0f64).powf(1e30), 1.0, "the defect");

        // Hand derivation, no library needed: a product of an odd number of
        // factors of −1 is −1.
        assert_eq!(pow_f64_integer_exponent(-1.0, &n), -1.0);
        assert_eq!(pow_f64_integer_exponent(-1.0, &(n.clone() - 1u32)), 1.0);
        assert_eq!(pow_f64_integer_exponent(1.0, &n), 1.0);

        // Through the lazy entry point, with the node supplied.
        let node = ExprData::Integer(crate::kernel::BigInt(n.clone()));
        assert_eq!(pow_f64(-1.0, 1e30, || Some(node.clone())), -1.0);
    }

    #[test]
    fn pow_f64_leaves_every_representable_exponent_bit_for_bit_alone() {
        // The hot path must not merely agree with `powf` — it must *be* it.
        for &base in &[-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 7.25] {
            for &e in &[0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, -0.5, 53.0, 1e15, -1e15] {
                let n = Integer::from(e as i64);
                let node = ExprData::Integer(crate::kernel::BigInt(n));
                let got = pow_f64(base, e, || Some(node));
                let want = base.powf(e);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{base} ^ {e}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn pow_f64_leaves_an_exactly_representable_wide_exponent_alone() {
        // 2^60 has 61 significant bits and is still its own `f64`, so nothing
        // was lost and the answer is `powf`'s.
        let n = Integer::from(1u64 << 60);
        assert_eq!(pow_f64_integer_exponent(-1.0, &n), 1.0);
        assert_eq!((-1.0f64).powf(integer_to_f64(&n)), 1.0);
        // 2^60 + 1 is odd and is *not* representable — it rounds to 2^60.
        let odd = n + 1u32;
        assert_eq!(integer_to_f64(&odd), 1.152921504606847e18);
        assert_eq!(pow_f64_integer_exponent(-1.0, &odd), -1.0);
    }

    #[test]
    fn pow_f64_signs_the_overflow_and_the_underflow() {
        let n = odd_wide_exponent();
        // Magnitude beyond `f64`: an infinity, correctly signed. Checked
        // entry points turn it into `E-EVAL-009` rather than a number.
        assert_eq!(pow_f64_integer_exponent(-2.0, &n), f64::NEG_INFINITY);
        assert_eq!(pow_f64_integer_exponent(2.0, &n), f64::INFINITY);
        // Magnitude below `f64`: a zero, correctly signed.
        assert!(pow_f64_integer_exponent(-0.5, &n).is_sign_negative());
        assert_eq!(pow_f64_integer_exponent(-0.5, &n), 0.0);
        assert!(pow_f64_integer_exponent(-0.0, &n).is_sign_negative());
    }

    #[test]
    fn pow_f64_falls_back_for_an_exponent_that_was_never_exact() {
        // A float exponent lost nothing by being a float, so `powf` is right.
        let node = ExprData::Float(crate::kernel::BigFloat {
            inner: rug::Float::with_val(53, 1e30),
            prec: 53,
        });
        assert_eq!(pow_f64(-1.0, 1e30, || Some(node)), (-1.0f64).powf(1e30));
        // And with no node at all (a bound symbol, say) there is nothing to
        // recover.
        assert_eq!(pow_f64(-1.0, 1e30, || None), (-1.0f64).powf(1e30));
        assert!(pow_f64(-2.0, f64::NAN, || None).is_nan());
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
