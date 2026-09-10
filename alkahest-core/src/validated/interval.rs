//! Directed-rounding interval arithmetic over the **extended** reals, used as
//! the last-resort range bound for one panel of [`super::bounds::verified_integral`].
//!
//! # Why this exists next to the Taylor models
//!
//! [`super::taylor`] is the right tool almost everywhere: it tracks
//! correlations between subexpressions, so it does not lose `x - x` to the
//! dependency problem, and its remainder shrinks like `O(h^{p+1})` under
//! subdivision. What it cannot do is stand on the *boundary* of a primitive's
//! domain. Every rule there needs a derivative bound over the argument
//! enclosure, and `asin`, `acos`, `sqrt`, `log` and the reciprocal all have an
//! unbounded derivative where their domain ends. So a panel that touches
//! `x = 1` refuses for `asin`, and one that touches `x = 0` refuses for
//! `sqrt` — even though `asin` is *bounded* there (`asin(1) = π/2`) and
//! `sqrt(x) ∈ [0, √h]`.
//!
//! A bounded integrand on a panel of width `h` contributes at most `h ·
//! range` to the integral, which is all that panel needs to be closed. This
//! module computes that `range` with the one tool that does not need a
//! derivative: plain interval arithmetic, evaluated with **directed rounding
//! on the endpoints** rather than midpoint/radius balls, and extended to
//! `±∞` so that a subexpression which really is unbounded (`log x` as
//! `x → 0⁺`) can still be composed with one that tames it (`exp(x·log x) ∈
//! [0, 1]`).
//!
//! Two properties matter and are the reason this is not simply
//! [`crate::ball::IntervalEval`]:
//!
//! * **Endpoints stay exact.** The range of the integration variable over the
//!   panel `[lo, hi]` is *exactly* `[lo, hi]`; there is no ball radius to
//!   round outward. That is what makes `sqrt(x)` on `[0, h]` work at all — a
//!   ball whose lower endpoint has been inflated to `-ε` is out of `sqrt`'s
//!   domain and is (correctly) refused. Likewise `1 - x²` on `[1-h, 1]`
//!   evaluates to `[0, …]` on the nose, because `1 - 1·1` is exact.
//! * **Unboundedness is representable, not an error.** `log([0, h])` is
//!   `[-∞, log h]`, and `[0, h] · [-∞, log h]` is `[-∞, 0]`. Only the *final*
//!   range has to be finite; refusing at the first infinity would lose every
//!   integrand whose boundedness comes from a cancellation between an
//!   unbounded factor and a vanishing one.
//!
//! # Soundness
//!
//! Every rule below returns a superset of `{ f(x) : x ∈ box }`:
//!
//! * Endpoint arithmetic uses [`rug::float::Round::Down`] for lower bounds and
//!   [`rug::float::Round::Up`] for upper bounds, so no rounding step can ever
//!   narrow an interval.
//! * A NaN endpoint (`∞ - ∞`, `0/0`) makes the interval meaningless and
//!   returns `None`, which propagates to a refusal.
//! * `0 · ±∞` is taken to be `0` in the multiplication corners. This is the
//!   IEEE 1788 convention and it is exactly right here: the product set of a
//!   bounded interval containing `0` with an unbounded one has `0` among its
//!   attainable values (take the finite factor to be `0`), and `±∞` as its
//!   limit (take the finite factor away from `0`) — which is what the corner
//!   rule produces.
//! * A primitive whose domain the argument interval genuinely leaves — `log`
//!   or `sqrt` of an interval with a *strictly negative* lower endpoint,
//!   `asin` outside `[-1, 1]` — returns `None`. A domain *endpoint* (`log` of
//!   `[0, h]`, `sqrt` of `[0, h]`) is not a violation: the value set over the
//!   half-open panel is enclosed, and the single point where the integrand is
//!   undefined is a null set, exactly as for the removable singularities
//!   [`super::bounds::verified_integral`] already integrates through.
//! * Anything with no rule here returns `None`. There is no fallback that
//!   guesses.

use super::from_bounds;
use crate::ball::ArbBall;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::float::Round;
use rug::ops::Pow;
use rug::Float;

/// Recursion ceiling, so a pathological expression cannot blow the stack in
/// what is only ever a best-effort fallback.
const MAX_DEPTH: usize = 256;

/// A closed interval of the extended reals `[-∞, +∞]`.
///
/// Invariant: `lo <= hi`, neither endpoint is NaN. Both endpoints are held at
/// the working precision and every operation rounds them outward.
#[derive(Clone, Debug)]
pub(super) struct XInterval {
    lo: Float,
    hi: Float,
    prec: u32,
}

/// `a op b` rounded in `dir`, or `None` when the result is NaN.
fn rounded(v: Float, dir: Round, prec: u32) -> Option<Float> {
    let out = Float::with_val_round(prec, v, dir).0;
    (!out.is_nan()).then_some(out)
}

/// `a · b` with the IEEE 1788 convention `0 · ±∞ = 0`; `None` on NaN.
fn xmul(a: &Float, b: &Float, dir: Round, prec: u32) -> Option<Float> {
    if (a.is_zero() && b.is_infinite()) || (b.is_zero() && a.is_infinite()) {
        return Some(Float::new(prec));
    }
    rounded(Float::with_val(prec + 32, a * b), dir, prec)
}

fn fmin(a: Float, b: Float) -> Float {
    if a <= b {
        a
    } else {
        b
    }
}

fn fmax(a: Float, b: Float) -> Float {
    if a >= b {
        a
    } else {
        b
    }
}

impl XInterval {
    fn new(lo: Float, hi: Float, prec: u32) -> Option<Self> {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return None;
        }
        Some(XInterval { lo, hi, prec })
    }

    fn constant(v: f64, prec: u32) -> Option<Self> {
        XInterval::new(Float::with_val(prec, v), Float::with_val(prec, v), prec)
    }

    fn is_finite(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    /// Midpoint and an upward-rounded radius, both finite. `None` for an
    /// unbounded interval.
    fn mid_rad(&self) -> Option<(Float, Float)> {
        if !self.is_finite() {
            return None;
        }
        let p = self.prec;
        let mid = Float::with_val(p, Float::with_val(p + 32, &self.lo + &self.hi) / 2u32);
        let a = Float::with_val_round(p, Float::with_val(p + 32, &mid - &self.lo), Round::Up).0;
        let b = Float::with_val_round(p, Float::with_val(p + 32, &self.hi - &mid), Round::Up).0;
        Some((mid, fmax(a, b)))
    }

    fn add(&self, other: &Self) -> Option<Self> {
        let p = self.prec;
        let lo = rounded(
            Float::with_val(p + 32, &self.lo + &other.lo),
            Round::Down,
            p,
        )?;
        let hi = rounded(Float::with_val(p + 32, &self.hi + &other.hi), Round::Up, p)?;
        XInterval::new(lo, hi, p)
    }

    fn mul(&self, other: &Self) -> Option<Self> {
        let p = self.prec;
        let ends = [
            (&self.lo, &other.lo),
            (&self.lo, &other.hi),
            (&self.hi, &other.lo),
            (&self.hi, &other.hi),
        ];
        let mut lo: Option<Float> = None;
        let mut hi: Option<Float> = None;
        for (a, b) in ends {
            let l = xmul(a, b, Round::Down, p)?;
            let h = xmul(a, b, Round::Up, p)?;
            lo = Some(match lo {
                Some(c) => fmin(c, l),
                None => l,
            });
            hi = Some(match hi {
                Some(c) => fmax(c, h),
                None => h,
            });
        }
        XInterval::new(lo?, hi?, p)
    }

    /// `1/self`. An interval with `0` in its *interior* becomes `[-∞, +∞]`;
    /// one whose closed end is `0` keeps the one-sided bound the values on the
    /// other side of it actually have.
    fn recip(&self) -> Option<Self> {
        let p = self.prec;
        let one = Float::with_val(p, 1);
        let inv = |v: &Float, dir: Round| -> Option<Float> {
            if v.is_zero() {
                return None;
            }
            rounded(Float::with_val(p + 32, &one / v), dir, p)
        };
        let zlo = self.lo.is_zero();
        let zhi = self.hi.is_zero();
        if zlo && zhi {
            return None;
        }
        if self.lo < 0 && self.hi > 0 {
            return XInterval::new(
                Float::with_val(p, f64::NEG_INFINITY),
                Float::with_val(p, f64::INFINITY),
                p,
            );
        }
        if zlo {
            // values in (0, hi] ⇒ [1/hi, +∞]
            return XInterval::new(
                inv(&self.hi, Round::Down)?,
                Float::with_val(p, f64::INFINITY),
                p,
            );
        }
        if zhi {
            // values in [lo, 0) ⇒ [-∞, 1/lo]
            return XInterval::new(
                Float::with_val(p, f64::NEG_INFINITY),
                inv(&self.lo, Round::Up)?,
                p,
            );
        }
        // 0 is outside the closed interval, so `1/·` is monotone decreasing on it.
        XInterval::new(inv(&self.hi, Round::Down)?, inv(&self.lo, Round::Up)?, p)
    }

    fn neg(&self) -> Option<Self> {
        let p = self.prec;
        XInterval::new(
            Float::with_val(p, -&self.hi),
            Float::with_val(p, -&self.lo),
            p,
        )
    }

    fn abs(&self) -> Option<Self> {
        let p = self.prec;
        if self.lo >= 0 {
            return Some(self.clone());
        }
        if self.hi <= 0 {
            return self.neg();
        }
        let m = fmax(
            Float::with_val(p, self.lo.abs_ref()),
            Float::with_val(p, self.hi.abs_ref()),
        );
        XInterval::new(Float::new(p), m, p)
    }

    fn powi(&self, n: i64) -> Option<Self> {
        let p = self.prec;
        if n == 0 {
            return XInterval::constant(1.0, p);
        }
        if n < 0 {
            return self.powi(-n)?.recip();
        }
        let pw = |v: &Float, dir: Round| -> Option<Float> {
            let e = u32::try_from(n).ok()?;
            rounded(Float::with_val(p + 32, v.pow(e)), dir, p)
        };
        if n % 2 == 1 {
            // Odd powers are increasing on the whole line.
            return XInterval::new(pw(&self.lo, Round::Down)?, pw(&self.hi, Round::Up)?, p);
        }
        if self.lo >= 0 {
            return XInterval::new(pw(&self.lo, Round::Down)?, pw(&self.hi, Round::Up)?, p);
        }
        if self.hi <= 0 {
            return XInterval::new(pw(&self.hi, Round::Down)?, pw(&self.lo, Round::Up)?, p);
        }
        // Straddles zero: the minimum of an even power is 0, the maximum is at
        // whichever endpoint is farther from it.
        let a = pw(&self.lo, Round::Up)?;
        let b = pw(&self.hi, Round::Up)?;
        XInterval::new(Float::new(p), fmax(a, b), p)
    }

    /// Image under a function that is non-decreasing on the whole of `self`.
    fn increasing(&self, f: impl Fn(&Float, Round) -> Option<Float>) -> Option<Self> {
        XInterval::new(
            f(&self.lo, Round::Down)?,
            f(&self.hi, Round::Up)?,
            self.prec,
        )
    }

    /// Image under a function that is non-increasing on the whole of `self`.
    fn decreasing(&self, f: impl Fn(&Float, Round) -> Option<Float>) -> Option<Self> {
        XInterval::new(
            f(&self.hi, Round::Down)?,
            f(&self.lo, Round::Up)?,
            self.prec,
        )
    }

    fn exp(&self) -> Option<Self> {
        let p = self.prec;
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.exp_ref()), d, p))
    }

    /// `log`, defined on `(0, ∞)`. A *closed* left end at `0` is not a
    /// violation — it contributes `-∞`, which the caller may still tame.
    fn log(&self) -> Option<Self> {
        let p = self.prec;
        if self.lo < 0 {
            return None;
        }
        if self.lo.is_zero() {
            let hi = if self.hi.is_zero() {
                Float::with_val(p, f64::NEG_INFINITY)
            } else {
                rounded(Float::with_val(p + 32, self.hi.ln_ref()), Round::Up, p)?
            };
            return XInterval::new(Float::with_val(p, f64::NEG_INFINITY), hi, p);
        }
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.ln_ref()), d, p))
    }

    fn sqrt(&self) -> Option<Self> {
        let p = self.prec;
        if self.lo < 0 {
            return None;
        }
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.sqrt_ref()), d, p))
    }

    fn asin(&self) -> Option<Self> {
        let p = self.prec;
        if self.lo < -1 || self.hi > 1 {
            return None;
        }
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.asin_ref()), d, p))
    }

    fn acos(&self) -> Option<Self> {
        let p = self.prec;
        if self.lo < -1 || self.hi > 1 {
            return None;
        }
        self.decreasing(|v, d| rounded(Float::with_val(p + 32, v.acos_ref()), d, p))
    }

    fn atan(&self) -> Option<Self> {
        let p = self.prec;
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.atan_ref()), d, p))
    }

    fn atanh(&self) -> Option<Self> {
        let p = self.prec;
        if self.lo < -1 || self.hi > 1 {
            return None;
        }
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.atanh_ref()), d, p))
    }

    fn asinh(&self) -> Option<Self> {
        let p = self.prec;
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.asinh_ref()), d, p))
    }

    fn acosh(&self) -> Option<Self> {
        let p = self.prec;
        if self.lo < 1 {
            return None;
        }
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.acosh_ref()), d, p))
    }

    fn sinh(&self) -> Option<Self> {
        let p = self.prec;
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.sinh_ref()), d, p))
    }

    fn tanh(&self) -> Option<Self> {
        let p = self.prec;
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.tanh_ref()), d, p))
    }

    fn erf(&self) -> Option<Self> {
        let p = self.prec;
        self.increasing(|v, d| rounded(Float::with_val(p + 32, v.erf_ref()), d, p))
    }

    fn erfc(&self) -> Option<Self> {
        let p = self.prec;
        self.decreasing(|v, d| rounded(Float::with_val(p + 32, v.erfc_ref()), d, p))
    }

    fn cosh(&self) -> Option<Self> {
        let p = self.prec;
        let f = |v: &Float, d: Round| rounded(Float::with_val(p + 32, v.cosh_ref()), d, p);
        if self.lo >= 0 {
            return self.increasing(f);
        }
        if self.hi <= 0 {
            return self.decreasing(f);
        }
        let a = f(&self.lo, Round::Up)?;
        let b = f(&self.hi, Round::Up)?;
        XInterval::new(Float::with_val(p, 1), fmax(a, b), p)
    }

    /// `sin`/`cos` through the Lipschitz bound `|f'| ≤ 1` around the midpoint,
    /// intersected with the global range `[-1, 1]`.
    ///
    /// This is deliberately not a monotonicity analysis: locating the extrema
    /// means deciding whether a multiple of `π/2` lies in the interval, and
    /// getting that wrong by an ulp would be unsound. `|f(m + t) - f(m)| ≤ |t|`
    /// needs nothing but the mean value theorem, and on the floor-width panels
    /// this fallback runs on it is tight to the last bit.
    fn trig(&self, cosine: bool) -> Option<Self> {
        let p = self.prec;
        let (mid, rad) = self.mid_rad()?;
        let centre = if cosine {
            Float::with_val(p + 32, mid.cos_ref())
        } else {
            Float::with_val(p + 32, mid.sin_ref())
        };
        let lo = rounded(Float::with_val(p + 32, &centre - &rad), Round::Down, p)?;
        let hi = rounded(Float::with_val(p + 32, &centre + &rad), Round::Up, p)?;
        XInterval::new(
            fmax(lo, Float::with_val(p, -1)),
            fmin(hi, Float::with_val(p, 1)),
            p,
        )
    }
}

/// Rigorous enclosure of the range of `expr` over the panel `[lo, hi]`, or
/// `None` when no rule applies or the range is not bounded.
///
/// The returned ball is a superset of `{ f(x) : x ∈ [lo, hi], f defined }`.
pub(super) fn panel_range(
    expr: ExprId,
    pool: &ExprPool,
    var: ExprId,
    lo: &Float,
    hi: &Float,
    prec: u32,
) -> Option<ArbBall> {
    let x = XInterval::new(
        Float::with_val_round(prec, lo, Round::Down).0,
        Float::with_val_round(prec, hi, Round::Up).0,
        prec,
    )?;
    let r = eval(expr, pool, var, &x, prec, 0)?;
    r.is_finite()
        .then(|| from_bounds(&r.lo, &r.hi, prec))
        .filter(super::is_finite)
}

/// Same as [`panel_range`], phrased as the `width × range` contribution of one
/// panel to an integral: `∫_lo^hi f dx ∈ (hi − lo) · range(f)`.
///
/// The width factor is the interval `[⌊hi − lo⌋, ⌈hi − lo⌉]`, i.e. the exact
/// width bracketed by directed rounding — *not* `[0, hi − lo]`. The difference
/// is not cosmetic: for a range that does not contain zero (`asin` near
/// `x = 1`, whose values are all close to `π/2`) the loose factor multiplies
/// the panel's contribution by the whole of `π/2` instead of by the tiny
/// variation of `asin` across the panel, and the quadrature loop then bisects
/// the panel until it runs out of budget trying to recover.
pub(super) fn panel_integral(
    expr: ExprId,
    pool: &ExprPool,
    var: ExprId,
    lo: &Float,
    hi: &Float,
    prec: u32,
) -> Option<ArbBall> {
    let range = panel_range(expr, pool, var, lo, hi, prec)?;
    let wide = Float::with_val(prec + 32, hi - lo);
    let w_lo = Float::with_val_round(prec, &wide, Round::Down).0;
    let w_hi = Float::with_val_round(prec, &wide, Round::Up).0;
    let piece = from_bounds(&w_lo, &w_hi, prec) * range;
    super::is_finite(&piece).then_some(piece)
}

fn eval(
    expr: ExprId,
    pool: &ExprPool,
    var: ExprId,
    x: &XInterval,
    prec: u32,
    depth: usize,
) -> Option<XInterval> {
    if depth > MAX_DEPTH {
        return None;
    }
    if expr == var {
        return Some(x.clone());
    }
    match pool.get(expr) {
        // Bracketed straight from the exact value, at the working precision.
        // Rounding to an intermediate `Float` (or to `f64`) first and bracketing
        // *that* would enclose the rounded number rather than the literal, which
        // for a constant wider than `prec` bits is not an enclosure at all.
        ExprData::Integer(n) => XInterval::new(
            Float::with_val_round(prec, &n.0, Round::Down).0,
            Float::with_val_round(prec, &n.0, Round::Up).0,
            prec,
        ),
        ExprData::Rational(r) => XInterval::new(
            Float::with_val_round(prec, &r.0, Round::Down).0,
            Float::with_val_round(prec, &r.0, Round::Up).0,
            prec,
        ),
        ExprData::Float(f) => XInterval::new(
            Float::with_val_round(prec, &f.inner, Round::Down).0,
            Float::with_val_round(prec, &f.inner, Round::Up).0,
            prec,
        ),
        // A free symbol other than the integration variable has no interval.
        ExprData::Symbol { .. } => None,
        ExprData::Add(args) => {
            let mut acc = XInterval::constant(0.0, prec)?;
            for a in args {
                acc = acc.add(&eval(a, pool, var, x, prec, depth + 1)?)?;
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = XInterval::constant(1.0, prec)?;
            for a in args {
                acc = acc.mul(&eval(a, pool, var, x, prec, depth + 1)?)?;
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let b = eval(base, pool, var, x, prec, depth + 1)?;
            if let ExprData::Integer(n) = pool.get(exp) {
                return b.powi(n.0.to_i64()?);
            }
            // Anything else goes through `exp(e · log b)`, which is the real
            // branch only for a non-negative base — and `log` refuses when the
            // base interval reaches below zero.
            let e = eval(exp, pool, var, x, prec, depth + 1)?;
            b.log()?.mul(&e)?.exp()
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let a = eval(args[0], pool, var, x, prec, depth + 1)?;
            match name.as_str() {
                "exp" => a.exp(),
                "log" | "ln" => a.log(),
                "sqrt" => a.sqrt(),
                "sin" => a.trig(false),
                "cos" => a.trig(true),
                "asin" => a.asin(),
                "acos" => a.acos(),
                "atan" => a.atan(),
                "sinh" => a.sinh(),
                "cosh" => a.cosh(),
                "tanh" => a.tanh(),
                "asinh" => a.asinh(),
                "acosh" => a.acosh(),
                "atanh" => a.atanh(),
                "erf" => a.erf(),
                "erfc" => a.erfc(),
                "abs" => a.abs(),
                // `tan`, the Bessel functions, `gamma`, `digamma` and
                // `lambert_w` have no interval rule here. They are not refused
                // outright anywhere else — the Taylor models handle them — and
                // guessing a monotone rule for a function with poles is exactly
                // the mistake this module exists to avoid.
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    const P: u32 = 128;

    fn f(v: f64) -> Float {
        Float::with_val(P, v)
    }

    #[test]
    fn variable_range_is_exact_at_the_panel_endpoints() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let r = panel_range(x, &pool, x, &f(0.0), &f(0.25), P).unwrap();
        assert!(r.lo() <= 0.0 && r.hi() >= 0.25);
        // The panel's own endpoints must not be inflated past the domain of a
        // `sqrt` sitting on top of them.
        let s = pool.func("sqrt", vec![x]);
        let rs = panel_range(s, &pool, x, &f(0.0), &f(0.25), P).unwrap();
        assert!(rs.lo() <= 0.0);
        assert!(rs.hi() >= 0.5 - 1e-30);
    }

    #[test]
    fn asin_is_bounded_at_the_end_of_its_domain() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.func("asin", vec![x]);
        let r = panel_range(a, &pool, x, &f(1.0 - 1e-9), &f(1.0), P).unwrap();
        let half_pi = std::f64::consts::FRAC_PI_2;
        assert!(r.hi() >= half_pi);
        assert!(r.lo() <= half_pi);
    }

    #[test]
    fn one_minus_x_squared_reaches_exactly_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let inner = pool.add(vec![
            one,
            pool.mul(vec![pool.integer(-1_i32), pool.mul(vec![x, x])]),
        ]);
        let s = pool.func("sqrt", vec![inner]);
        // The Taylor model refuses here; interval arithmetic with exact
        // endpoints does not, because `1 - 1·1` is exact.
        let r = panel_range(s, &pool, x, &f(1.0 - 1e-9), &f(1.0), P).unwrap();
        assert!(r.lo() <= 0.0);
        assert!(r.hi() >= 4.4e-5);
    }

    #[test]
    fn x_to_the_x_is_bounded_at_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // exp(x · log x) — unbounded factor times a vanishing one.
        let e = pool.func("exp", vec![pool.mul(vec![x, pool.func("log", vec![x])])]);
        let r = panel_range(e, &pool, x, &f(0.0), &f(1e-9), P).unwrap();
        assert!(r.lo() <= 0.0);
        assert!(r.hi() >= 1.0);
    }

    #[test]
    fn genuine_poles_stay_unbounded() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 1/x on [0, h] — the integral does not exist, and no range does either.
        let recip = pool.pow(x, pool.integer(-1_i32));
        assert!(panel_range(recip, &pool, x, &f(0.0), &f(1e-9), P).is_none());
        // -log(x) on [0, h] — integrable but unbounded; still refused.
        let nlog = pool.mul(vec![pool.integer(-1_i32), pool.func("log", vec![x])]);
        assert!(panel_range(nlog, &pool, x, &f(0.0), &f(1e-9), P).is_none());
        // 1/sqrt(x) on [0, h].
        let isq = pool.pow(pool.func("sqrt", vec![x]), pool.integer(-1_i32));
        assert!(panel_range(isq, &pool, x, &f(0.0), &f(1e-9), P).is_none());
    }

    #[test]
    fn out_of_domain_on_a_set_of_positive_measure_is_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let s = pool.func("sqrt", vec![x]);
        assert!(panel_range(s, &pool, x, &f(-1.0), &f(1.0), P).is_none());
        let l = pool.func("log", vec![x]);
        assert!(panel_range(l, &pool, x, &f(-1.0), &f(1.0), P).is_none());
        let a = pool.func("asin", vec![x]);
        assert!(panel_range(a, &pool, x, &f(0.5), &f(2.0), P).is_none());
    }

    #[test]
    fn trig_bound_contains_the_true_range() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let s = pool.func("sin", vec![x]);
        let r = panel_range(s, &pool, x, &f(0.0), &f(3.2), P).unwrap();
        // sin reaches 1 on [0, 3.2]; the Lipschitz bound must not miss it.
        assert!(r.hi() >= 1.0);
        assert!(r.lo() <= 0.0);
        let c = pool.func("cos", vec![x]);
        let rc = panel_range(c, &pool, x, &f(0.0), &f(6.5), P).unwrap();
        assert!(rc.lo() <= -1.0 && rc.hi() >= 1.0);
    }

    #[test]
    fn a_constant_wider_than_the_working_precision_is_still_bracketed() {
        // 2^200 + 1 is not representable in 128 bits: rounding it to a `Float`
        // and bracketing *that* would enclose the rounded value, not the
        // literal. The interval has to straddle the true integer.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let big: rug::Integer = (rug::Integer::from(1) << 200) + 1;
        let e = pool.add(vec![pool.integer(big.clone()), pool.mul(vec![x, x])]);
        let r = panel_range(e, &pool, x, &f(0.0), &f(0.0), P).unwrap();
        let truth = Float::with_val(400, &big);
        assert!(r.lo() <= truth, "lo {} > {}", r.lo(), truth);
        assert!(r.hi() >= truth, "hi {} < {}", r.hi(), truth);
        assert!(
            r.lo() < r.hi(),
            "a 201-bit integer cannot be exact at 128 bits"
        );
    }

    #[test]
    fn no_rule_means_no_bound() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.func("tan", vec![x]);
        assert!(panel_range(t, &pool, x, &f(0.0), &f(0.1), P).is_none());
    }
}
