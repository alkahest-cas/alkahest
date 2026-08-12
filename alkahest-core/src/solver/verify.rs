//! Rigorous post-condition checking for the polynomial system solver.
//!
//! [`solve_polynomial_system`](super::solve_polynomial_system) builds candidate
//! solutions by back-substituting through a Lex Gröbner basis.  Substituting a
//! finished tuple back into the *original* equations is far cheaper than
//! producing it, so the solver does exactly that before returning: a tuple
//! whose residual is **provably** non-zero is dropped rather than reported.
//!
//! "Provably" is the operative word.  The check runs in complex ball
//! arithmetic ([`CBall`], built on [`ArbBall`]), where every operation is
//! outward-rounded, so the true residual is always inside the returned ball.
//! A candidate is discarded only when its residual ball is *separated from
//! zero* — a rigorous certificate that the tuple is not a solution.  A ball
//! that straddles zero proves nothing and the candidate survives, so the check
//! can never remove a genuine solution (which would trade one silent error for
//! a worse one).
//!
//! The same separation test drives de-duplication: two candidates collapse
//! only when no coordinate can be proved distinct.
//!
//! Values the solver can build are rationals combined with `+`, `*`, integer
//! powers and `sqrt`.  Anything else — a free parameter, say — makes the tuple
//! *unverifiable* rather than *refuted*, and it is returned untouched.

use crate::ball::ArbBall;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::groebner::GbPoly;
use rug::ops::Pow;
use rug::Rational;

/// Working precision, in bits, for the solver's post-condition check.
///
/// Well above `f64`: the residual of a true solution shrinks towards zero as
/// precision grows, while the residual of a spurious one does not, so the
/// separation test only sharpens with more bits.  192 keeps the whole check in
/// the microsecond range for the systems the symbolic solver accepts (degree
/// ≤ 2 per variable).
const VERIFY_PREC: u32 = 192;

/// Why a candidate could not be evaluated to a complex ball.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerifyGap {
    /// The expression is a well-formed value the evaluator does not model
    /// (a free parameter, an unsupported function).  Nothing can be concluded.
    Unsupported,
    /// The expression denotes no complex number at all — `0/0`, `0^-1`.
    /// A tuple with such a coordinate is not a point of ℂⁿ.
    Undefined,
}

/// A rigorous enclosure of a complex number: `re ± re.rad` and `im ± im.rad`.
#[derive(Clone, Debug)]
pub(crate) struct CBall {
    re: ArbBall,
    im: ArbBall,
}

fn zero_ball() -> ArbBall {
    ArbBall::from_f64(0.0, VERIFY_PREC)
}

impl CBall {
    fn real(re: ArbBall) -> Self {
        CBall {
            re,
            im: zero_ball(),
        }
    }

    fn from_rational(r: &Rational) -> Self {
        CBall::real(ArbBall::from_rational(r, VERIFY_PREC))
    }

    fn from_integer(n: &rug::Integer) -> Self {
        CBall::real(ArbBall::from_integer(n, VERIFY_PREC))
    }

    fn from_f64(v: f64) -> Self {
        CBall::real(ArbBall::from_f64(v, VERIFY_PREC))
    }

    fn one() -> Self {
        CBall::from_f64(1.0)
    }

    fn zero() -> Self {
        CBall::from_f64(0.0)
    }

    fn add(&self, other: &CBall) -> CBall {
        CBall {
            re: self.re.clone() + other.re.clone(),
            im: self.im.clone() + other.im.clone(),
        }
    }

    fn mul(&self, other: &CBall) -> CBall {
        let ac = self.re.clone() * other.re.clone();
        let bd = self.im.clone() * other.im.clone();
        let ad = self.re.clone() * other.im.clone();
        let bc = self.im.clone() * other.re.clone();
        CBall {
            re: ac - bd,
            im: ad + bc,
        }
    }

    /// `1/z = conj(z) / |z|²`.  `None` when `|z|²` cannot be separated from
    /// zero, which is exactly when the reciprocal may not exist.
    fn recip(&self) -> Option<CBall> {
        let d = self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone();
        let re = (self.re.clone() / d.clone())?;
        let im = (-self.im.clone() / d)?;
        Some(CBall { re, im })
    }

    fn powi(&self, n: i64) -> Option<CBall> {
        if n == 0 {
            return Some(CBall::one());
        }
        if n < 0 {
            let pos = self.powi(-n)?;
            return pos.recip();
        }
        let mut acc = CBall::one();
        let mut base = self.clone();
        let mut e = n as u64;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base);
            }
        }
        Some(acc)
    }

    /// Principal-branch complex square root, matching `eval_complex_f64`
    /// (`sqrt(-1) = +i`).
    ///
    /// `sqrt(a + bi) = u + sign(b)·v·i` with `u = √((|z| + a)/2)` and
    /// `v = √((|z| − a)/2)`, and `sign(0) = +1`.  When the sign of `b` cannot
    /// be decided from its ball, the imaginary part is widened to `[−v, v]`,
    /// which encloses both branches: weaker, never wrong.
    fn sqrt(&self) -> Option<CBall> {
        // A real argument keeps the result on one axis *exactly*, which is what
        // preserves the distinction between the two roots further up.  Going
        // through the general formula instead would leave `√1` with a spurious
        // imaginary width of about 2^-95 (the square root of the discarded
        // rounding term), and one more level of nesting then widens `√−4` to
        // "±2i", at which point `+i` and `−i` are no longer provably different
        // and de-duplication silently merges two genuine solutions.
        let im_is_exact_zero = self.im.is_exact() && self.im.mid_f64() == 0.0;
        if im_is_exact_zero {
            if self.re.lo() >= 0 {
                return Some(CBall::real(self.re.sqrt()?));
            }
            if self.re.hi() <= 0 {
                return Some(CBall {
                    re: zero_ball(),
                    im: (-self.re.clone()).sqrt()?,
                });
            }
        }
        let norm2 = self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone();
        let modulus = clamp_nonneg(norm2).sqrt()?;
        let half = ArbBall::from_f64(0.5, VERIFY_PREC);
        let u = clamp_nonneg((modulus.clone() + self.re.clone()) * half.clone()).sqrt()?;
        let v = clamp_nonneg((modulus - self.re.clone()) * half).sqrt()?;
        // `im` is exactly zero for every discriminant built from rationals,
        // which is the case that has to stay sharp.
        let im_is_exact_zero = self.im.is_exact() && self.im.mid_f64() == 0.0;
        let im = if im_is_exact_zero || self.im.lo() > 0 {
            v
        } else if self.im.hi() < 0 {
            -v
        } else {
            widen_around_zero(&v)
        };
        Some(CBall { re: u, im })
    }

    /// True when the ball is separated from the origin, i.e. **no** complex
    /// number it encloses is zero.  This is the only direction the check may
    /// act on: a ball containing zero proves nothing either way.
    pub(crate) fn excludes_zero(&self) -> bool {
        !self.re.contains(0.0) || !self.im.contains(0.0)
    }

    /// Is every point of this ball within `2^exp` of the origin?
    fn is_within_scale(&self, exp: i32) -> bool {
        let bound = rug::Float::with_val(VERIFY_PREC, 2.0_f64).pow(exp);
        let reach =
            |b: &ArbBall| rug::Float::with_val(VERIFY_PREC, b.mid.clone().abs()) + b.rad.clone();
        reach(&self.re) < bound && reach(&self.im) < bound
    }

    #[cfg(test)]
    fn neg_ball(&self) -> CBall {
        CBall {
            re: -self.re.clone(),
            im: -self.im.clone(),
        }
    }

    fn sub(&self, other: &CBall) -> CBall {
        CBall {
            re: self.re.clone() - other.re.clone(),
            im: self.im.clone() - other.im.clone(),
        }
    }
}

/// Replace a ball whose lower end has drifted below zero by rounding with one
/// clamped at zero.  Sound only for quantities that are non-negative by
/// construction (`|z|`, `|z| ± Re z`), which is where it is used.
fn clamp_nonneg(b: ArbBall) -> ArbBall {
    if b.lo() >= 0 {
        return b;
    }
    let hi = b.hi();
    let mid = rug::Float::with_val(VERIFY_PREC, &hi / 2u32);
    ArbBall {
        mid: mid.clone(),
        rad: mid,
        prec: VERIFY_PREC,
    }
}

/// `[−|b|, +|b|]` — encloses both `+b` and `−b`.
fn widen_around_zero(b: &ArbBall) -> ArbBall {
    let hi = rug::Float::with_val(VERIFY_PREC, b.hi().abs());
    let lo = rug::Float::with_val(VERIFY_PREC, b.lo().abs());
    let bound = if hi > lo { hi } else { lo };
    ArbBall {
        mid: rug::Float::with_val(VERIFY_PREC, 0.0),
        rad: bound,
        prec: VERIFY_PREC,
    }
}

/// Evaluate a solver-produced value to a rigorous complex enclosure, memoising
/// on [`ExprId`].
///
/// Sibling roots `(−b ± √D)/2a` and the successive back-substitution levels
/// share almost all of their structure, so the expression is a DAG whose
/// tree expansion grows with the variable count.  The memo keeps the check
/// linear in the number of distinct nodes.
#[derive(Default)]
pub(crate) struct CBallEval {
    memo: std::collections::HashMap<ExprId, Result<CBall, VerifyGap>>,
}

impl CBallEval {
    pub(crate) fn eval(&mut self, expr: ExprId, pool: &ExprPool) -> Result<CBall, VerifyGap> {
        if let Some(hit) = self.memo.get(&expr) {
            return hit.clone();
        }
        let out = self.eval_uncached(expr, pool);
        self.memo.insert(expr, out.clone());
        out
    }

    fn eval_uncached(&mut self, expr: ExprId, pool: &ExprPool) -> Result<CBall, VerifyGap> {
        match pool.get(expr) {
            ExprData::Integer(n) => Ok(CBall::from_integer(&n.0)),
            ExprData::Rational(r) => Ok(CBall::from_rational(&r.0)),
            ExprData::Float(f) => Ok(CBall::from_f64(f.inner.to_f64())),
            ExprData::Add(args) => {
                let mut acc = CBall::zero();
                for a in args {
                    acc = acc.add(&self.eval(a, pool)?);
                }
                Ok(acc)
            }
            ExprData::Mul(args) => {
                let mut acc = CBall::one();
                for a in args {
                    acc = acc.mul(&self.eval(a, pool)?);
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp } => {
                let ExprData::Integer(n) = pool.get(exp) else {
                    return Err(VerifyGap::Unsupported);
                };
                let n = n.0.to_i64().ok_or(VerifyGap::Unsupported)?;
                let b = self.eval(base, pool)?;
                b.powi(n).ok_or(VerifyGap::Undefined)
            }
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                let x = self.eval(args[0], pool)?;
                x.sqrt().ok_or(VerifyGap::Undefined)
            }
            _ => Err(VerifyGap::Unsupported),
        }
    }
}

/// Evaluate `poly` at the given complex enclosures (one per indeterminate).
pub(crate) fn poly_residual(poly: &GbPoly, values: &[CBall]) -> Option<CBall> {
    let mut acc = CBall::zero();
    for (exp, coeff) in &poly.terms {
        let mut term = CBall::from_rational(coeff);
        for (i, &e) in exp.iter().enumerate() {
            if e == 0 {
                continue;
            }
            term = term.mul(&values.get(i)?.powi(e as i64)?);
        }
        acc = acc.add(&term);
    }
    Some(acc)
}

/// Residual at a *partial* assignment: `None` unless every indeterminate the
/// polynomial actually uses has a value.
pub(crate) fn poly_residual_partial(poly: &GbPoly, values: &[Option<CBall>]) -> Option<CBall> {
    let mut acc = CBall::zero();
    for (exp, coeff) in &poly.terms {
        let mut term = CBall::from_rational(coeff);
        for (i, &e) in exp.iter().enumerate() {
            if e == 0 {
                continue;
            }
            term = term.mul(&values.get(i)?.as_ref()?.powi(e as i64)?);
        }
        acc = acc.add(&term);
    }
    Some(acc)
}

/// True when some equation's residual at `values` is provably non-zero, i.e.
/// the tuple is certainly **not** a solution of the system.
pub(crate) fn is_refuted(polys: &[GbPoly], values: &[CBall]) -> bool {
    polys.iter().any(|p| match poly_residual(p, values) {
        Some(r) => r.excludes_zero(),
        // An undefined residual means the tuple does not lie in the domain of
        // the polynomial map — it is not a point of ℂⁿ, so it is not a
        // solution either.
        None => true,
    })
}

/// Binary exponent below which two candidates are treated as the same point.
///
/// Chosen well above the widest enclosure the solver produces: a discriminant
/// that is zero but only numerically so (`4y − 4` reached through `√1`) leaves
/// its two roots enclosed to about 2^-94, so anything tighter than that would
/// report a double root twice.  Chosen well below any separation the
/// *representable* fragment can produce: two distinct roots of a quadratic
/// over ℚ are further apart than 2^-64 unless its coefficients exceed 2^32.
const SAME_POINT_EXP: i32 = -64;

/// True when every coordinate of `a` is enclosed within 2^`SAME_POINT_EXP` of
/// the matching coordinate of `b`.
///
/// This is deliberately *positive* evidence rather than "could not be proved
/// different".  De-duplication removes a solution, so basing it on ignorance
/// would let an imprecise enclosure delete a genuine root — the exact failure
/// mode this module exists to prevent.  When the enclosures are too wide to
/// decide, both candidates are kept: a duplicate entry is a cosmetic fault, a
/// dropped solution is a silent error.
pub(crate) fn same_point(a: &[CBall], b: &[CBall]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.sub(y).is_within_scale(SAME_POINT_EXP))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ExprPool;

    fn cball(expr: ExprId, pool: &ExprPool) -> Result<CBall, VerifyGap> {
        CBallEval::default().eval(expr, pool)
    }

    fn sqrt_of(pool: &ExprPool, n: i32) -> ExprId {
        pool.func("sqrt", vec![pool.integer(n)])
    }

    #[test]
    fn rational_arithmetic_is_exact_enough_to_separate() {
        let pool = ExprPool::new();
        let two = pool.integer(2_i32);
        let three = pool.integer(3_i32);
        let sum = pool.add(vec![two, three]);
        let v = cball(sum, &pool).expect("evaluable");
        assert!(v.excludes_zero());
        let diff = pool.add(vec![two, pool.integer(-2_i32)]);
        let z = cball(diff, &pool).expect("evaluable");
        assert!(!z.excludes_zero(), "2 + (-2) must not be separated from 0");
    }

    #[test]
    fn zero_over_zero_is_undefined_not_unsupported() {
        let pool = ExprPool::new();
        let zero = pool.integer(0_i32);
        let inv = pool.pow(zero, pool.integer(-1_i32));
        assert!(matches!(cball(inv, &pool), Err(VerifyGap::Undefined)));
    }

    #[test]
    fn free_symbol_is_unsupported() {
        let pool = ExprPool::new();
        let a = pool.symbol("a", crate::kernel::Domain::Real);
        assert!(matches!(cball(a, &pool), Err(VerifyGap::Unsupported)));
    }

    #[test]
    fn principal_sqrt_of_negative_is_positive_imaginary() {
        let pool = ExprPool::new();
        let s = sqrt_of(&pool, -4);
        let v = cball(s, &pool).expect("evaluable");
        assert!(v.re.contains(0.0), "real part of √-4 is 0");
        assert!(v.im.contains(2.0), "√-4 = +2i, got im = {}", v.im);
        assert!(!v.im.contains(-2.0));
    }

    #[test]
    fn sqrt_of_zero_is_zero() {
        let pool = ExprPool::new();
        let s = sqrt_of(&pool, 0);
        let v = cball(s, &pool).expect("evaluable");
        assert!(!v.excludes_zero());
        assert!(v.re.contains(0.0) && v.im.contains(0.0));
    }

    #[test]
    fn spurious_root_is_refuted_and_true_root_is_not() {
        // x² − x·y and x·y − y at (−1, 1): the first residual is 1 + 1 = 2.
        let pool = ExprPool::new();
        let mut f1 = GbPoly::zero(2);
        f1 = f1.add(&GbPoly::monomial(vec![2, 0], Rational::from(1)));
        f1 = f1.add(&GbPoly::monomial(vec![1, 1], Rational::from(-1)));
        let minus_one = cball(pool.integer(-1_i32), &pool).unwrap();
        let one = cball(pool.integer(1_i32), &pool).unwrap();
        assert!(is_refuted(&[f1.clone()], &[minus_one, one.clone()]));
        assert!(!is_refuted(&[f1], &[one.clone(), one]));
    }

    #[test]
    fn irrational_root_survives_and_its_negation_is_separated() {
        // x² − 2 at x = √2 must not be refuted; at x = √2 + 1 it must be.
        let pool = ExprPool::new();
        let mut f = GbPoly::zero(1);
        f = f.add(&GbPoly::monomial(vec![2], Rational::from(1)));
        f = f.add(&GbPoly::monomial(vec![0], Rational::from(-2)));
        let root = cball(sqrt_of(&pool, 2), &pool).unwrap();
        assert!(!is_refuted(&[f.clone()], std::slice::from_ref(&root)));
        let shifted = cball(
            pool.add(vec![sqrt_of(&pool, 2), pool.integer(1_i32)]),
            &pool,
        )
        .unwrap();
        assert!(is_refuted(&[f], std::slice::from_ref(&shifted)));
    }

    #[test]
    fn real_argument_keeps_the_root_on_one_axis_exactly() {
        // √1 must come back with an imaginary part that is *exactly* zero, and
        // √−4 with a real part that is exactly zero.  Anything wider survives
        // one more nesting level and stops `+i` and `−i` being distinguishable.
        let pool = ExprPool::new();
        let one = cball(sqrt_of(&pool, 1), &pool).unwrap();
        assert!(one.im.is_exact() && one.im.mid_f64() == 0.0);
        let neg = cball(sqrt_of(&pool, -4), &pool).unwrap();
        assert!(neg.re.is_exact() && neg.re.mid_f64() == 0.0);
        // …and the two square roots of −4 stay provably apart after halving.
        let half = CBall::from_rational(&Rational::from((1, 2)));
        let plus = neg.mul(&half);
        let minus = plus.neg_ball();
        assert!(!same_point(
            std::slice::from_ref(&plus),
            std::slice::from_ref(&minus)
        ));
    }

    #[test]
    fn distinctness_separates_plus_and_minus_root() {
        let pool = ExprPool::new();
        let plus = cball(sqrt_of(&pool, 2), &pool).unwrap();
        let minus = cball(
            pool.mul(vec![pool.integer(-1_i32), sqrt_of(&pool, 2)]),
            &pool,
        )
        .unwrap();
        assert!(!same_point(
            std::slice::from_ref(&plus),
            std::slice::from_ref(&minus)
        ));
        assert!(same_point(
            std::slice::from_ref(&plus),
            std::slice::from_ref(&plus)
        ));
        // ±√0 is one point, and the enclosure has to say so.
        let z = cball(sqrt_of(&pool, 0), &pool).unwrap();
        let neg_z = z.neg_ball();
        assert!(same_point(
            std::slice::from_ref(&z),
            std::slice::from_ref(&neg_z)
        ));
    }
}
