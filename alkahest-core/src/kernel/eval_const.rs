//! Constant folding helpers for predicates and numeric evaluation.

use crate::ball::IntervalEval;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::float::Round;
use rug::{Float, Integer, Rational};
use std::cmp::Ordering;

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
///
/// Comparisons are decided **exactly** or not at all.  They used to be decided
/// by rounding both sides to `f64`, so `10^30 + 1 = 10^30` folded to `True`,
/// `3333333333333333/10^16 ≠ 1/3` to `False`, `1/10^400 = 0` to `True`, and a
/// `Piecewise` guarded by `x > 2^53` took its default branch at `2^53 + 1`.
///
/// * Two numeric literals (`Integer`, `Rational`, `Float`) are compared as the
///   exact numbers they are — a `Float` is a dyadic rational, so this is exact
///   too.  A NaN keeps its IEEE meaning: every ordering and `=` are false,
///   `≠` is true.
/// * Otherwise both sides are enclosed with rigorous ball arithmetic
///   ([`crate::ball::IntervalEval`]).  The comparison is decided only when the
///   enclosures prove it — separated balls for an ordering or `≠`, and
///   identical *exact* balls for `=` — and is left unevaluated (`None`) when
///   they overlap, rather than guessed.
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
        PredicateKind::Lt
        | PredicateKind::Le
        | PredicateKind::Gt
        | PredicateKind::Ge
        | PredicateKind::Eq
        | PredicateKind::Ne => {
            let [lhs, rhs] = args else {
                return None;
            };
            match exact_numeric_cmp(*lhs, *rhs, pool) {
                Some(ord) => Some(decide_ordering(kind, ord)),
                None => ball_decide(kind, *lhs, *rhs, pool),
            }
        }
    }
}

/// The truth of `kind` given the exact ordering of its two sides (`None` when
/// they are unordered, i.e. a NaN is involved).
fn decide_ordering(kind: &PredicateKind, ord: Option<Ordering>) -> bool {
    match (kind, ord) {
        (PredicateKind::Ne, None) => true,
        (_, None) => false,
        (PredicateKind::Lt, Some(o)) => o == Ordering::Less,
        (PredicateKind::Le, Some(o)) => o != Ordering::Greater,
        (PredicateKind::Gt, Some(o)) => o == Ordering::Greater,
        (PredicateKind::Ge, Some(o)) => o != Ordering::Less,
        (PredicateKind::Eq, Some(o)) => o == Ordering::Equal,
        (PredicateKind::Ne, Some(o)) => o != Ordering::Equal,
        _ => unreachable!("only comparisons reach decide_ordering"),
    }
}

/// An exact numeric literal.
enum ExactNumber {
    Rational(Rational),
    Float(Float),
}

fn exact_number(expr: ExprId, pool: &ExprPool) -> Option<ExactNumber> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(ExactNumber::Rational(Rational::from(n.0))),
        ExprData::Rational(r) => Some(ExactNumber::Rational(r.0)),
        ExprData::Float(f) => Some(ExactNumber::Float(f.inner)),
        _ => None,
    }
}

/// The exact ordering of two numeric literals: `None` when either is not a
/// literal, `Some(None)` when they are unordered (a NaN).
fn exact_numeric_cmp(a: ExprId, b: ExprId, pool: &ExprPool) -> Option<Option<Ordering>> {
    let (a, b) = (exact_number(a, pool)?, exact_number(b, pool)?);
    Some(match (&a, &b) {
        (ExactNumber::Rational(x), ExactNumber::Rational(y)) => Some(x.cmp(y)),
        (ExactNumber::Rational(x), ExactNumber::Float(y)) => {
            y.partial_cmp(x).map(Ordering::reverse)
        }
        (ExactNumber::Float(x), ExactNumber::Rational(y)) => x.partial_cmp(y),
        (ExactNumber::Float(x), ExactNumber::Float(y)) => x.partial_cmp(y),
    })
}

/// Working precisions for [`ball_decide`]: a cheap attempt, then one retry for
/// operands that agree to more than `128` bits.
const BALL_PRECISIONS: [u32; 2] = [128, 512];

/// Decide a comparison between two closed constant expressions (`√2`, `π/2`,
/// `exp(1)`) by rigorous enclosure, or return `None`.
fn ball_decide(kind: &PredicateKind, a: ExprId, b: ExprId, pool: &ExprPool) -> Option<bool> {
    for prec in BALL_PRECISIONS {
        let ev = IntervalEval::new(prec);
        // `None`: a free symbol, a complex or undefined value, an unsupported
        // function — nothing to decide, at any precision.
        let (x, y) = (ev.eval(a, pool)?, ev.eval(b, pool)?);
        let (x_lo, x_hi, y_lo, y_hi) = (x.lo(), x.hi(), y.lo(), y.hi());
        // The endpoints are rounded outward, and an indeterminate ball is
        // `(-∞, ∞)`, so a NaN never reaches these comparisons and a strict
        // separation is a proof.
        let below = x_hi < y_lo;
        let above = x_lo > y_hi;
        let same_point = x.is_exact() && y.is_exact() && x.mid == y.mid;
        let decided = match kind {
            PredicateKind::Lt if below => Some(true),
            PredicateKind::Lt if above || same_point => Some(false),
            PredicateKind::Le if below || same_point => Some(true),
            PredicateKind::Le if above => Some(false),
            PredicateKind::Gt if above => Some(true),
            PredicateKind::Gt if below || same_point => Some(false),
            PredicateKind::Ge if above || same_point => Some(true),
            PredicateKind::Ge if below => Some(false),
            PredicateKind::Eq if same_point => Some(true),
            PredicateKind::Eq if below || above => Some(false),
            PredicateKind::Ne if below || above => Some(true),
            PredicateKind::Ne if same_point => Some(false),
            _ => None,
        };
        if decided.is_some() {
            return decided;
        }
    }
    None
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

/// Audit A4: predicates over exact numbers were decided in `f64`.
#[cfg(test)]
mod exact_predicate_tests {
    use super::*;
    use crate::kernel::subs::{fold_predicates, subs};
    use crate::kernel::Domain;
    use rug::ops::Pow;
    use std::collections::HashMap;

    fn big(e: u32) -> Integer {
        Integer::from(10).pow(e)
    }

    fn decide(p: &ExprPool, pred: ExprId) -> Option<bool> {
        try_predicate_bool_from_expr(pred, p)
    }

    #[test]
    fn integers_past_f64_compare_exactly() {
        let p = ExprPool::new();
        let a = p.integer(big(30));
        let b = p.integer(big(30) + 1u32);
        assert_eq!(decide(&p, p.pred_eq(b, a)), Some(false));
        assert_eq!(decide(&p, p.pred_ne(b, a)), Some(true));
        assert_eq!(decide(&p, p.pred_lt(a, b)), Some(true));
        assert_eq!(decide(&p, p.pred_le(b, a)), Some(false));
        assert_eq!(decide(&p, p.pred_gt(b, a)), Some(true));
        assert_eq!(decide(&p, p.pred_ge(a, b)), Some(false));
        let two53 = p.integer(Integer::from(1u64 << 53));
        let two53p1 = p.integer(Integer::from((1u64 << 53) + 1));
        assert_eq!(decide(&p, p.pred_gt(two53p1, two53)), Some(true));
    }

    #[test]
    fn rationals_compare_exactly() {
        let p = ExprPool::new();
        let third = p.rational(1, 3);
        let approx = p.rational(3_333_333_333_333_333_i64, big(16));
        assert_eq!(decide(&p, p.pred_ne(approx, third)), Some(true));
        assert_eq!(decide(&p, p.pred_eq(approx, third)), Some(false));
        assert_eq!(decide(&p, p.pred_lt(approx, third)), Some(true));
        // Underflows `f64` to 0, and is not 0.
        let tiny = p.rational(1, big(400));
        let zero = p.integer(0);
        assert_eq!(decide(&p, p.pred_eq(zero, tiny)), Some(false));
        assert_eq!(decide(&p, p.pred_lt(zero, tiny)), Some(true));
        // Overflows `f64` to ∞ on both sides, and they differ.
        let h1 = p.rational(big(400) + 1u32, 3);
        let h2 = p.rational(big(400) + 2u32, 3);
        assert_eq!(decide(&p, p.pred_lt(h1, h2)), Some(true));
    }

    #[test]
    fn floats_compare_as_the_dyadic_rationals_they_are() {
        let p = ExprPool::new();
        // The double nearest 1/3 is strictly below it.
        let f = p.float(1.0 / 3.0, 53);
        let third = p.rational(1, 3);
        assert_eq!(decide(&p, p.pred_lt(f, third)), Some(true));
        assert_eq!(decide(&p, p.pred_eq(f, third)), Some(false));
        assert_eq!(decide(&p, p.pred_gt(third, f)), Some(true));
        let half = p.float(0.5, 53);
        assert_eq!(decide(&p, p.pred_eq(half, p.rational(1, 2))), Some(true));
        // NaN keeps its IEEE meaning.
        let nan = p.float(f64::NAN, 53);
        let one = p.integer(1);
        assert_eq!(decide(&p, p.pred_eq(nan, one)), Some(false));
        assert_eq!(decide(&p, p.pred_ne(nan, one)), Some(true));
        assert_eq!(decide(&p, p.pred_lt(one, nan)), Some(false));
    }

    #[test]
    fn subs_folds_exact_comparisons_and_picks_the_right_branch() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let at = |e: ExprId, v: Integer| {
            let mut m = HashMap::new();
            m.insert(x, p.integer(v));
            fold_predicates(subs(e, &m, &p), &p)
        };
        let t = p.pred_true();
        let f = p.pred_false();
        assert_eq!(at(p.pred_eq(x, p.integer(big(30))), big(30) + 1u32), f);
        assert_eq!(at(p.pred_lt(x, p.integer(big(30) + 1u32)), big(30)), t);
        let two53 = Integer::from(1u64 << 53);
        let pw = p.piecewise(
            vec![(p.pred_gt(x, p.integer(two53.clone())), p.integer(1))],
            p.integer(0),
        );
        assert_eq!(at(pw, two53 + 1u32), p.integer(1));
    }

    #[test]
    fn closed_constants_are_decided_by_enclosure() {
        let p = ExprPool::new();
        let sqrt2 = p.pow(p.integer(2), p.rational(1, 2));
        let one = p.integer(1);
        assert_eq!(decide(&p, p.pred_gt(sqrt2, one)), Some(true));
        assert_eq!(decide(&p, p.pred_eq(sqrt2, one)), Some(false));
        let pi = p.symbol("pi", Domain::Real);
        let approx = p.rational(355, 113);
        // 355/113 − π ≈ 2.7e-7.
        assert_eq!(decide(&p, p.pred_lt(pi, approx)), Some(true));
        assert_eq!(decide(&p, p.pred_ne(pi, approx)), Some(true));
        // Agrees with its argument past 128 bits: needs the 512-bit retry.
        let close = p.add(vec![one, p.rational(1, Integer::from(2).pow(200))]);
        assert_eq!(decide(&p, p.pred_gt(close, one)), Some(true));
    }

    #[test]
    fn an_overlap_is_left_undecided_rather_than_guessed() {
        let p = ExprPool::new();
        let two = p.integer(2);
        // (√2)² is exactly 2, but no finite enclosure proves it.
        let sq = p.pow(p.pow(two, p.rational(1, 2)), two);
        assert_eq!(decide(&p, p.pred_eq(sq, two)), None);
        assert_eq!(decide(&p, p.pred_lt(sq, two)), None);
        assert_eq!(decide(&p, p.pred_ge(sq, two)), None);
        // A free symbol decides nothing.
        let x = p.symbol("x", Domain::Real);
        assert_eq!(decide(&p, p.pred_gt(x, two)), None);
    }

    #[test]
    fn a_sum_past_the_working_precision_is_decided_on_retry() {
        // `10^50` needs 167 bits: at 128 the `+ 1` is inside the enclosure's
        // rounding and nothing is proved; the 512-bit retry proves it.
        let p = ExprPool::new();
        let a = p.integer(big(50));
        let b = p.add(vec![p.integer(big(50)), p.integer(1)]);
        assert_eq!(decide(&p, p.pred_eq(a, b)), Some(false));
        assert_eq!(decide(&p, p.pred_lt(a, b)), Some(true));
    }
}
