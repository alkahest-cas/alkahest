//! Certified numeric root-finding for one real equation in one unknown.
//!
//! [`super::solve_transcendental`] inverts the equations it recognises in
//! closed form; everything else — implied volatility `BS(σ) = price`,
//! `sin x = x/10`, `cos x = x` — has no closed form and used to be a refusal
//! even with `numeric=True`. This module finds such roots numerically and
//! says how much of the answer is *proved*:
//!
//! * [`nsolve`] — one root, from a bracket `[a, b]` with a sign change
//!   (Brent's method) or from a starting point (damped Newton);
//! * [`nsolve_all`] — every real root in `[a, b]` (subdivision with an
//!   interval-derivative test), or an explicit list of the sub-intervals it
//!   could not settle.
//!
//! # What "verified" means
//!
//! The search runs in `f64`; the *certificate* is computed afterwards in
//! 128-bit outward-rounded ball arithmetic ([`crate::ball::IntervalEval`]),
//! so nothing about how the candidate was found enters the proof:
//!
//! * **existence** — `f(lo)` and `f(hi)` are enclosed by balls of strictly
//!   opposite sign and `f` is continuous on `[lo, hi]` (it is built from
//!   smooth functions only and has a finite enclosure over the whole box), so
//!   a root lies in `[lo, hi]` by the intermediate value theorem;
//! * **uniqueness** — an enclosure of `f′` over `[lo, hi]` excludes zero, so
//!   `f` is strictly monotone there.
//!
//! Both hold → [`RootStatus::Verified`]; existence only →
//! [`RootStatus::Exists`]; neither (a double root, a non-smooth function, an
//! enclosure that could not be tightened) → [`RootStatus::Heuristic`], which
//! is the `f64` iterate and nothing more.
//!
//! [`nsolve_all`] discards a sub-interval only when an enclosure of `f` over
//! it excludes zero, and accepts one only with the monotonicity test above,
//! so its root list is complete whenever [`RootIsolation::unresolved`] is
//! empty. A double root, or a stretch where `f` cannot be enclosed (outside
//! its domain), lands in `unresolved` instead of being guessed at.
//!
//! Every loop consults the active [`crate::budget`]; a stop is recorded with
//! [`crate::budget::record_trip`] and reported as `E-SOLVE-012`.

use std::collections::HashMap;
use std::fmt;

use rug::{Float, Rational};

use crate::ball::{ArbBall, IntervalEval};
use crate::errors::AlkahestError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::collect_free_vars;

/// Working precision of the certificates, in bits.
const PREC: u32 = 128;

/// How [`nsolve`] looks for its root.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NsolveStart {
    /// A bracket `[a, b]` on which `f` changes sign (Brent's method).
    Bracket(f64, f64),
    /// A starting point (damped Newton).
    Guess(f64),
}

/// How much of a [`NumericRoot`] is proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RootStatus {
    /// Exactly one root lies in `[lo, hi]` (sign change + monotonicity).
    Verified,
    /// At least one root lies in `[lo, hi]` (sign change); uniqueness was not
    /// proved.
    Exists,
    /// Numerical evidence only: `[lo, hi]` is the bare iterate.
    Heuristic,
}

impl RootStatus {
    /// `"verified"`, `"exists"` or `"heuristic"`.
    pub fn as_str(self) -> &'static str {
        match self {
            RootStatus::Verified => "verified",
            RootStatus::Exists => "exists",
            RootStatus::Heuristic => "heuristic",
        }
    }
}

/// A numeric root with its enclosure and how much of it is proved.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct NumericRoot {
    /// The `f64` approximation (always inside `[lo, hi]`).
    pub value: f64,
    /// Lower end of the enclosure.
    pub lo: f64,
    /// Upper end of the enclosure.
    pub hi: f64,
    /// What `[lo, hi]` is a proof of.
    pub status: RootStatus,
}

/// Every real root of `f` in an interval, as far as it could be settled.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RootIsolation {
    /// Verified roots, in increasing order, with disjoint enclosures.
    pub roots: Vec<NumericRoot>,
    /// Sub-intervals that could neither be excluded nor shown to hold exactly
    /// one root (a multiple root, a stretch outside `f`'s domain), merged and
    /// in increasing order.  Empty when `roots` is the complete list.
    pub unresolved: Vec<(f64, f64)>,
}

impl RootIsolation {
    /// `true` when every root in the interval is in [`RootIsolation::roots`].
    pub fn is_complete(&self) -> bool {
        self.unresolved.is_empty()
    }
}

/// Why [`nsolve`] / [`nsolve_all`] declined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsolveError {
    code: &'static str,
    detail: String,
}

impl NsolveError {
    /// The input cannot be searched (`E-SOLVE-009`): a free parameter, an
    /// empty or non-finite interval, a bracket without a sign change.
    fn input(detail: impl Into<String>) -> Self {
        NsolveError {
            code: "E-SOLVE-009",
            detail: detail.into(),
        }
    }

    /// The search itself failed (`E-SOLVE-012`): no convergence, `f` not
    /// evaluable where it had to be, a budget stop.
    fn failed(detail: impl Into<String>) -> Self {
        NsolveError {
            code: "E-SOLVE-012",
            detail: detail.into(),
        }
    }

    /// A [`RootIsolation`] that is not complete, where a complete root list is
    /// required (`E-SOLVE-012`) — `solve(numeric=True, interval=…)` promises a
    /// whole solution set.
    pub fn incomplete(iso: &RootIsolation) -> Self {
        let gaps: Vec<String> = iso
            .unresolved
            .iter()
            .map(|(lo, hi)| format!("[{lo}, {hi}]"))
            .collect();
        NsolveError::failed(format!(
            "could not account for every root: {} verified, but {} could not be settled \
             (a multiple root, or a stretch where the equation has no real value)",
            iso.roots.len(),
            gaps.join(", ")
        ))
    }

    /// What went wrong, in words.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for NsolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for NsolveError {}

impl AlkahestError for NsolveError {
    fn code(&self) -> &'static str {
        self.code
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self.code {
            "E-SOLVE-009" => {
                "substitute values for every symbol except the unknown, and pass a finite \
                 bracket (a, b) on which the equation changes sign, or a starting point x0"
            }
            _ => {
                "try a tighter bracket around the root, a different starting point, or \
                 all_in=(a, b) to isolate every root in an interval"
            }
        })
    }
}

/// Heads that are not continuously differentiable in their argument; an
/// equation containing one (applied to the unknown) is solved heuristically.
const NONSMOOTH: &[&str] = &[
    "abs",
    "sign",
    "floor",
    "ceil",
    "round",
    "frac",
    "heaviside",
    "diracdelta",
    "min",
    "max",
    "re",
    "im",
    "arg",
    "conjugate",
];

/// The equation, its derivative and the evaluators shared by both entry points.
struct Problem<'a> {
    f: ExprId,
    df: Option<ExprId>,
    var: ExprId,
    pool: &'a ExprPool,
    /// Built from smooth functions of `var` only, so the certificates apply.
    smooth: bool,
}

impl<'a> Problem<'a> {
    fn new(equation: ExprId, var: ExprId, pool: &'a ExprPool) -> Result<Self, NsolveError> {
        if !matches!(pool.get(var), ExprData::Symbol { .. }) {
            return Err(NsolveError::input(format!(
                "the unknown must be a symbol, got {}",
                pool.display(var)
            )));
        }
        let free = collect_free_vars(equation, pool);
        let params: Vec<String> = free
            .iter()
            .filter(|&&s| s != var && !crate::eval::symbols::is_named_constant(s, pool))
            .map(|&s| pool.display(s).to_string())
            .collect();
        if !params.is_empty() {
            return Err(NsolveError::input(format!(
                "a numeric root needs every symbol but {} to have a value; the equation also \
                 depends on {}",
                pool.display(var),
                params.join(", ")
            )));
        }
        if !free.contains(&var) {
            return Err(NsolveError::input(format!(
                "the equation does not depend on {}",
                pool.display(var)
            )));
        }
        let df = crate::diff::diff(equation, var, pool).ok().map(|d| d.value);
        Ok(Problem {
            f: equation,
            df,
            var,
            pool,
            smooth: is_smooth(equation, var, pool),
        })
    }

    fn f64_at(&self, e: ExprId, x: f64) -> Option<f64> {
        let env: HashMap<ExprId, f64> = [(self.var, x)].into_iter().collect();
        crate::jit::eval_interp(e, &env, self.pool).filter(|v| v.is_finite())
    }

    fn f(&self, x: f64) -> Option<f64> {
        self.f64_at(self.f, x)
    }

    fn ball(&self, e: ExprId, b: ArbBall) -> Option<ArbBall> {
        let mut ev = IntervalEval::new(PREC);
        ev.bind(self.var, b);
        ev.eval(e, self.pool)
            .filter(|r| r.lo().is_finite() && r.hi().is_finite())
    }

    /// `f` at the exact point `x`.
    fn f_point(&self, x: f64) -> Option<ArbBall> {
        self.ball(self.f, ArbBall::from_f64(x, PREC))
    }

    fn f_box(&self, lo: f64, hi: f64) -> Option<ArbBall> {
        self.ball(self.f, box_ball(lo, hi))
    }

    fn df_box(&self, lo: f64, hi: f64) -> Option<ArbBall> {
        self.ball(self.df?, box_ball(lo, hi))
    }

    /// `f(x) = 0` exactly, proved: the ball at `x` contains zero and the
    /// equation with `x` substituted as an exact rational simplifies to `0`
    /// (`sin(0)`, `x·eˣ` at `0`).  The ball alone cannot say it — a radius
    /// from rounding never shrinks to nothing.
    fn exact_zero_at(&self, x: f64) -> bool {
        let Some(b) = self.f_point(x) else {
            return false;
        };
        if sign(&b).is_some() {
            return false;
        }
        if b.lo() == 0 && b.hi() == 0 {
            return true;
        }
        let Some(q) = Rational::from_f64(x) else {
            return false;
        };
        let (n, d) = q.into_numer_denom();
        let xv = if d == 1 {
            self.pool.integer(n)
        } else {
            self.pool.rational(n, d)
        };
        let map: HashMap<ExprId, ExprId> = [(self.var, xv)].into_iter().collect();
        let at = crate::kernel::subs(self.f, &map, self.pool);
        let s = crate::simplify::engine::simplify(at, self.pool).value;
        matches!(self.pool.get(s), ExprData::Integer(n) if n.0 == 0)
    }

    /// `f′` excludes zero on `[lo, hi]`.
    fn monotone(&self, lo: f64, hi: f64) -> bool {
        self.df_box(lo, hi).is_some_and(|d| sign(&d).is_some())
    }

    /// Prove what can be proved about a root near `r`, trying successively
    /// wider boxes `r ± δ` clipped to `[lo_lim, hi_lim]`, then `extra`.
    fn certify(&self, r: f64, lo_lim: f64, hi_lim: f64, extra: &[(f64, f64)]) -> NumericRoot {
        let heuristic = NumericRoot {
            value: r,
            lo: r,
            hi: r,
            status: RootStatus::Heuristic,
        };
        if !self.smooth {
            return heuristic;
        }
        // An exact zero at r.
        if self.exact_zero_at(r) {
            let mut delta = 4.0 * f64::EPSILON * r.abs().max(1e-300);
            for _ in 0..24 {
                let (lo, hi) = ((r - delta).max(lo_lim), (r + delta).min(hi_lim));
                if self.monotone(lo, hi) {
                    return NumericRoot {
                        value: r,
                        lo: r,
                        hi: r,
                        status: RootStatus::Verified,
                    };
                }
                delta *= 16.0;
            }
            return NumericRoot {
                value: r,
                lo: r,
                hi: r,
                status: RootStatus::Exists,
            };
        }
        let mut boxes: Vec<(f64, f64)> = Vec::new();
        let mut delta = 2.0 * f64::EPSILON * r.abs().max(1e-300);
        for _ in 0..24 {
            boxes.push(((r - delta).max(lo_lim), (r + delta).min(hi_lim)));
            delta *= 16.0;
        }
        boxes.extend_from_slice(extra);
        for (lo, hi) in boxes {
            let usable = lo <= r && r <= hi && lo < hi;
            if !usable {
                continue;
            }
            let (Some(fl), Some(fh)) = (self.f_point(lo), self.f_point(hi)) else {
                continue;
            };
            match (sign(&fl), sign(&fh)) {
                (Some(a), Some(b)) if a != b => {}
                _ => continue,
            }
            // Continuity on the box: a finite enclosure of a smooth expression.
            if self.f_box(lo, hi).is_none() {
                continue;
            }
            let status = if self.monotone(lo, hi) {
                RootStatus::Verified
            } else {
                RootStatus::Exists
            };
            return NumericRoot {
                value: r,
                lo,
                hi,
                status,
            };
        }
        heuristic
    }
}

/// A ball holding exactly `[lo, hi]` (outward).
fn box_ball(lo: f64, hi: f64) -> ArbBall {
    crate::validated::from_bounds(&Float::with_val(PREC, lo), &Float::with_val(PREC, hi), PREC)
}

/// `Some(1)` / `Some(-1)` for a ball strictly above / below zero.
fn sign(b: &ArbBall) -> Option<i8> {
    if b.lo() > 0 {
        Some(1)
    } else if b.hi() < 0 {
        Some(-1)
    } else {
        None
    }
}

fn is_smooth(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    fn walk(
        e: ExprId,
        var: ExprId,
        pool: &ExprPool,
        seen: &mut std::collections::HashSet<ExprId>,
    ) -> bool {
        if !seen.insert(e) {
            return true;
        }
        match pool.get(e) {
            ExprData::Func { name, args } => {
                let has_var = args
                    .iter()
                    .any(|&a| collect_free_vars(a, pool).contains(&var));
                if has_var && NONSMOOTH.contains(&name.as_str()) {
                    return false;
                }
                args.iter().all(|&a| walk(a, var, pool, seen))
            }
            ExprData::Piecewise { .. } | ExprData::Predicate { .. } => {
                !collect_free_vars(e, pool).contains(&var)
            }
            ExprData::Add(args) | ExprData::Mul(args) => {
                args.iter().all(|&a| walk(a, var, pool, seen))
            }
            ExprData::Pow { base, exp } => {
                walk(base, var, pool, seen) && walk(exp, var, pool, seen)
            }
            _ => true,
        }
    }
    walk(e, var, pool, &mut std::collections::HashSet::new())
}

fn budget() -> Result<(), NsolveError> {
    crate::budget::check().map_err(|e| {
        crate::budget::record_trip(crate::budget::BudgetTrip::Budget(e));
        NsolveError::failed(format!("stopped by the active budget: {e}"))
    })
}

/// One real root of `equation = 0` in `var`.
///
/// * [`NsolveStart::Bracket`]`(a, b)` — `f(a)` and `f(b)` must have opposite
///   signs (or one be zero); Brent's method then always converges, to a root
///   inside `[a, b]`.
/// * [`NsolveStart::Guess`]`(x0)` — damped Newton from `x0`; which root it
///   reaches (if any) depends on `x0`.
///
/// The returned [`NumericRoot`] carries an enclosure and its
/// [`RootStatus`] — see the module documentation for what is proved.
///
/// # Errors
///
/// `E-SOLVE-009` for a free parameter, a non-finite or empty bracket, or a
/// bracket without a sign change; `E-SOLVE-012` when the iteration does not
/// converge, `f` cannot be evaluated where the search needs it, or the active
/// budget stops it.
pub fn nsolve(
    equation: ExprId,
    var: ExprId,
    start: NsolveStart,
    pool: &ExprPool,
) -> Result<NumericRoot, NsolveError> {
    crate::budget::clear_trip();
    let p = Problem::new(equation, var, pool)?;
    match start {
        NsolveStart::Bracket(a, b) => bracketed(&p, a, b),
        NsolveStart::Guess(x0) => newton(&p, x0),
    }
}

fn bracketed(p: &Problem<'_>, a: f64, b: f64) -> Result<NumericRoot, NsolveError> {
    if !(a.is_finite() && b.is_finite()) || a >= b {
        return Err(NsolveError::input(format!(
            "the bracket must be two finite numbers a < b, got ({a}, {b})"
        )));
    }
    let not_evaluable =
        |x: f64| NsolveError::failed(format!("the equation has no finite real value at {x}"));
    let fa = p.f(a).ok_or_else(|| not_evaluable(a))?;
    let fb = p.f(b).ok_or_else(|| not_evaluable(b))?;
    if fa == 0.0 {
        return Ok(p.certify(a, a, b, &[]));
    }
    if fb == 0.0 {
        return Ok(p.certify(b, a, b, &[]));
    }
    if fa.signum() == fb.signum() {
        return Err(NsolveError::input(format!(
            "the bracket ({a}, {b}) has no sign change: f({a}) = {fa:e} and f({b}) = {fb:e} \
             have the same sign"
        )));
    }
    let (r, lo, hi) = brent(p, a, fa, b, fb)?;
    Ok(p.certify(r, a, b, &[(lo, hi), (a, b)]))
}

/// Brent's method on a sign-changing bracket.  Returns the root estimate and
/// the final bracket `[lo, hi]` (which still changes sign in `f64`).
fn brent(
    p: &Problem<'_>,
    a0: f64,
    fa0: f64,
    b0: f64,
    fb0: f64,
) -> Result<(f64, f64, f64), NsolveError> {
    let (mut a, mut fa, mut b, mut fb) = (a0, fa0, b0, fb0);
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = d;
    for _ in 0..400 {
        budget()?;
        if (fb > 0.0) == (fc > 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * f64::EPSILON * b.abs() + f64::MIN_POSITIVE;
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return Ok((b, b.min(c), b.max(c)));
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut pp, mut q);
            if a == c {
                pp = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                let qq = fa / fc;
                let r = fb / fc;
                pp = s * (2.0 * m * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if pp > 0.0 {
                q = -q;
            } else {
                pp = -pp;
            }
            if 2.0 * pp < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = pp / q;
            } else {
                d = m;
                e = d;
            }
        } else {
            d = m;
            e = d;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = p.f(b).ok_or_else(|| {
            NsolveError::failed(format!(
                "the equation has no finite real value at {b}, inside the bracket"
            ))
        })?;
    }
    Err(NsolveError::failed(
        "Brent's method did not converge in 400 iterations",
    ))
}

fn newton(p: &Problem<'_>, x0: f64) -> Result<NumericRoot, NsolveError> {
    if !x0.is_finite() {
        return Err(NsolveError::input(format!(
            "the starting point must be finite, got {x0}"
        )));
    }
    let mut x = x0;
    let mut fx = p.f(x).ok_or_else(|| {
        NsolveError::failed(format!(
            "the equation has no finite real value at x0 = {x0}"
        ))
    })?;
    for _ in 0..200 {
        budget()?;
        if fx == 0.0 {
            return Ok(p.certify(x, f64::NEG_INFINITY, f64::INFINITY, &[]));
        }
        let slope = match p.df.and_then(|d| p.f64_at(d, x)) {
            Some(s) if s != 0.0 => s,
            _ => {
                // Secant slope when f′ is unavailable or zero.
                let h = 1e-7 * x.abs().max(1.0);
                match p.f(x + h) {
                    Some(f2) if f2 != fx => (f2 - fx) / h,
                    _ => {
                        return Err(NsolveError::failed(format!(
                            "Newton's method stalled at {x}: the derivative vanishes"
                        )))
                    }
                }
            }
        };
        let step = fx / slope;
        // Damping: halve the step until |f| decreases.
        let mut t = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let xn = x - t * step;
            if let Some(fxn) = p.f(xn) {
                if fxn.abs() < fx.abs() || (t * step).abs() <= 4.0 * f64::EPSILON * x.abs() {
                    accepted = Some((xn, fxn));
                    break;
                }
            }
            t *= 0.5;
        }
        let Some((xn, fxn)) = accepted else {
            // No decrease anywhere along the step: converged as far as f64
            // resolves, or stuck.  Certification decides which.
            let root = p.certify(x, f64::NEG_INFINITY, f64::INFINITY, &[]);
            if root.status != RootStatus::Heuristic {
                return Ok(root);
            }
            return Err(NsolveError::failed(format!(
                "Newton's method from {x0} stalled at {x} with f = {fx:e}"
            )));
        };
        let moved = (xn - x).abs();
        x = xn;
        fx = fxn;
        if moved <= 4.0 * f64::EPSILON * x.abs().max(f64::MIN_POSITIVE) {
            return Ok(p.certify(x, f64::NEG_INFINITY, f64::INFINITY, &[]));
        }
    }
    let root = p.certify(x, f64::NEG_INFINITY, f64::INFINITY, &[]);
    if root.status != RootStatus::Heuristic {
        return Ok(root);
    }
    Err(NsolveError::failed(format!(
        "Newton's method from {x0} did not converge in 200 iterations (last iterate {x})"
    )))
}

/// Every real root of `equation = 0` in `[a, b]`.
///
/// The interval is subdivided; a piece is **discarded** when a ball
/// enclosure of `f` over it excludes zero, and **settled** when an enclosure
/// of `f′` excludes zero (so `f` is monotone there) — it then holds exactly
/// one root if `f` changes sign across it (proved in ball arithmetic) and
/// none otherwise. Every root returned is [`RootStatus::Verified`]. A piece
/// that reaches the resolution limit without being settled (a multiple root,
/// a tangency, a stretch outside `f`'s domain) is reported in
/// [`RootIsolation::unresolved`]; the root list is complete exactly when that
/// is empty.
///
/// # Errors
///
/// `E-SOLVE-009` for a free parameter or an empty/non-finite interval, and
/// for an equation that is not smooth in `var` (`abs`, `floor`, `Piecewise`,
/// …), where the monotonicity certificate does not apply; `E-SOLVE-012` when
/// the active budget stops the search.
pub fn nsolve_all(
    equation: ExprId,
    var: ExprId,
    a: f64,
    b: f64,
    pool: &ExprPool,
) -> Result<RootIsolation, NsolveError> {
    crate::budget::clear_trip();
    if !(a.is_finite() && b.is_finite()) || a >= b {
        return Err(NsolveError::input(format!(
            "the interval must be two finite numbers a < b, got ({a}, {b})"
        )));
    }
    let p = Problem::new(equation, var, pool)?;
    if !p.smooth {
        return Err(NsolveError::input(
            "all_in isolation needs an equation built from smooth functions of the unknown \
             (no abs, sign, floor, min/max or Piecewise); use a bracket instead",
        ));
    }
    if p.df.is_none() {
        return Err(NsolveError::input(
            "all_in isolation needs the derivative of the equation, which could not be formed",
        ));
    }

    let span = b - a;
    // Resolution limits: a piece where f′ may vanish is split down to
    // `min_width`; one where f cannot be enclosed at all, to `undefined_width`
    // (so a stretch outside f's domain costs thousands of pieces, not 2⁴⁰).
    let min_width = (span * 1e-12).max(f64::MIN_POSITIVE);
    let undefined_width = span / 4096.0;
    const MAX_PIECES: usize = 200_000;

    let mut roots: Vec<NumericRoot> = Vec::new();
    let mut unresolved: Vec<(f64, f64)> = Vec::new();
    let mut stack: Vec<(f64, f64)> = vec![(a, b)];
    let mut processed = 0usize;
    while let Some((lo, hi)) = stack.pop() {
        budget()?;
        processed += 1;
        if processed > MAX_PIECES {
            unresolved.push((lo, hi));
            unresolved.append(&mut stack);
            break;
        }
        let width = hi - lo;
        // Split slightly off-centre so a "round" root (0, an integer) is
        // unlikely to sit exactly on a cut.
        let mid = lo + width * (255.0 / 512.0);
        let can_split = mid > lo && mid < hi;
        let fx = match p.f_box(lo, hi) {
            Some(fx) => fx,
            None => {
                if width > undefined_width && can_split {
                    stack.push((mid, hi));
                    stack.push((lo, mid));
                } else {
                    unresolved.push((lo, hi));
                }
                continue;
            }
        };
        if sign(&fx).is_some() {
            continue; // no root here
        }
        if p.monotone(lo, hi) {
            match settle_monotone(&p, lo, hi)? {
                Settled::None => continue,
                Settled::Root(r) => {
                    roots.push(r);
                    continue;
                }
                Settled::Unsettled => {}
            }
        }
        if width > min_width && can_split {
            stack.push((mid, hi));
            stack.push((lo, mid));
        } else {
            unresolved.push((lo, hi));
        }
    }

    // A root on a shared cut is found from both sides; enclosures from
    // adjacent monotone pieces that overlap hold the same root (f′ has one
    // sign across the cut), so they merge.
    roots.sort_by(|x, y| x.lo.total_cmp(&y.lo));
    let mut merged: Vec<NumericRoot> = Vec::new();
    for r in roots {
        match merged.last_mut() {
            Some(last) if r.lo <= last.hi => {
                last.lo = last.lo.min(r.lo);
                last.hi = last.hi.max(r.hi);
                if last.value < last.lo || last.value > last.hi {
                    last.value = r.value;
                }
            }
            _ => merged.push(r),
        }
    }
    unresolved.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut gaps: Vec<(f64, f64)> = Vec::new();
    for (lo, hi) in unresolved {
        match gaps.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => gaps.push((lo, hi)),
        }
    }
    Ok(RootIsolation {
        roots: merged,
        unresolved: gaps,
    })
}

enum Settled {
    None,
    Root(NumericRoot),
    Unsettled,
}

/// `f` is strictly monotone on `[lo, hi]`: decide whether it has a root there.
fn settle_monotone(p: &Problem<'_>, lo: f64, hi: f64) -> Result<Settled, NsolveError> {
    let (Some(fl), Some(fh)) = (p.f_point(lo), p.f_point(hi)) else {
        return Ok(Settled::Unsettled);
    };
    // Monotone, so an exact zero at an end is the piece's only root.
    for end in [lo, hi] {
        if p.exact_zero_at(end) {
            return Ok(Settled::Root(NumericRoot {
                value: end,
                lo: end,
                hi: end,
                status: RootStatus::Verified,
            }));
        }
    }
    match (sign(&fl), sign(&fh)) {
        (Some(s), Some(t)) if s == t => Ok(Settled::None),
        (Some(_), Some(_)) => {
            // Exactly one root; locate it and tighten the enclosure.
            let r = match (p.f(lo), p.f(hi)) {
                (Some(fa), Some(fb)) if fa != 0.0 && fb != 0.0 && fa.signum() != fb.signum() => {
                    brent(p, lo, fa, hi, fb)?.0
                }
                (Some(0.0), _) => lo,
                (_, Some(0.0)) => hi,
                _ => lo + 0.5 * (hi - lo),
            };
            let mut root = p.certify(r, lo, hi, &[]);
            if root.status != RootStatus::Verified {
                // The whole piece is a valid (if wide) certificate.
                root = NumericRoot {
                    value: r,
                    lo,
                    hi,
                    status: RootStatus::Verified,
                };
            }
            Ok(Settled::Root(root))
        }
        _ => Ok(Settled::Unsettled),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn x_and_pool() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        (pool, x)
    }

    #[test]
    fn bracket_cos_x_minus_x_is_verified() {
        let (pool, x) = x_and_pool();
        let eq = pool.add(vec![
            pool.func("cos", vec![x]),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let r = nsolve(eq, x, NsolveStart::Bracket(0.0, 1.0), &pool).unwrap();
        assert_eq!(r.status, RootStatus::Verified);
        let dottie = 0.739_085_133_215_160_6;
        assert!(r.lo <= dottie && dottie <= r.hi, "{r:?}");
        assert!(r.hi - r.lo < 1e-14, "{r:?}");
        assert!((r.value - dottie).abs() < 1e-15);
    }

    #[test]
    fn newton_from_a_guess() {
        let (pool, x) = x_and_pool();
        // x² − 2 from x0 = 1 → √2.
        let eq = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-2_i32)]);
        let r = nsolve(eq, x, NsolveStart::Guess(1.0), &pool).unwrap();
        assert_eq!(r.status, RootStatus::Verified);
        assert!(r.lo <= std::f64::consts::SQRT_2 && std::f64::consts::SQRT_2 <= r.hi);
    }

    #[test]
    fn bracket_without_sign_change_is_refused() {
        let (pool, x) = x_and_pool();
        let eq = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let err = nsolve(eq, x, NsolveStart::Bracket(-1.0, 1.0), &pool).unwrap_err();
        assert_eq!(err.code(), "E-SOLVE-009");
    }

    #[test]
    fn free_parameter_is_refused() {
        let (pool, x) = x_and_pool();
        let y = pool.symbol("y", Domain::Real);
        let eq = pool.add(vec![x, y]);
        let err = nsolve(eq, x, NsolveStart::Guess(0.0), &pool).unwrap_err();
        assert_eq!(err.code(), "E-SOLVE-009");
        assert!(err.detail().contains('y'));
    }

    #[test]
    fn all_in_sin_x_equals_x_over_10() {
        let (pool, x) = x_and_pool();
        // sin x − x/10 on [−12, 12]: 7 roots (0, ±2.85, ±7.07, ±8.42).
        let eq = pool.add(vec![
            pool.func("sin", vec![x]),
            pool.mul(vec![pool.rational(-1, 10), x]),
        ]);
        let iso = nsolve_all(eq, x, -12.0, 12.0, &pool).unwrap();
        assert!(iso.is_complete(), "{iso:?}");
        assert_eq!(iso.roots.len(), 7, "{iso:?}");
        for r in &iso.roots {
            assert_eq!(r.status, RootStatus::Verified);
            assert!((r.value.sin() - r.value / 10.0).abs() < 1e-14);
        }
        assert!(iso.roots.windows(2).all(|w| w[0].hi < w[1].lo));
    }

    #[test]
    fn all_in_reports_a_double_root_as_unresolved() {
        let (pool, x) = x_and_pool();
        // (x − 1)² has a double root: no sign change, nothing to certify.
        let eq = pool.pow(pool.add(vec![x, pool.integer(-1_i32)]), pool.integer(2_i32));
        let iso = nsolve_all(eq, x, -3.0, 3.0, &pool).unwrap();
        assert!(iso.roots.is_empty());
        assert_eq!(iso.unresolved.len(), 1, "{iso:?}");
        let (lo, hi) = iso.unresolved[0];
        assert!(lo <= 1.0 && 1.0 <= hi && hi - lo < 1e-6, "{iso:?}");
    }

    #[test]
    fn all_in_root_on_the_interval_end() {
        let (pool, x) = x_and_pool();
        let eq = pool.func("sin", vec![x]);
        let iso = nsolve_all(eq, x, 0.0, 7.0, &pool).unwrap();
        assert!(iso.is_complete(), "{iso:?}");
        let vals: Vec<f64> = iso.roots.iter().map(|r| r.value).collect();
        assert_eq!(vals.len(), 3, "{vals:?}");
        assert_eq!(vals[0], 0.0);
        assert!((vals[1] - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn all_in_honours_the_budget() {
        let (pool, x) = x_and_pool();
        let eq = pool.add(vec![
            pool.func("sin", vec![pool.mul(vec![pool.integer(50_i32), x])]),
            pool.rational(-1, 3),
        ]);
        let _g = crate::budget::enter(crate::budget::Budget::new().with_max_steps(5));
        let err = nsolve_all(eq, x, -10.0, 10.0, &pool).unwrap_err();
        assert_eq!(err.code(), "E-SOLVE-012");
        assert!(crate::budget::take_trip().is_some());
    }

    #[test]
    fn lambert_wm1_root_is_found_and_verified() {
        // x·eˣ = −1/4 on [−5, 0]: the W₋₁ and W₀ roots.
        let (pool, x) = x_and_pool();
        let eq = pool.add(vec![
            pool.mul(vec![x, pool.func("exp", vec![x])]),
            pool.rational(1, 4),
        ]);
        let iso = nsolve_all(eq, x, -5.0, 0.0, &pool).unwrap();
        assert!(iso.is_complete());
        assert_eq!(iso.roots.len(), 2);
        let wm1 = crate::special::lambert_wm1(-0.25).unwrap();
        let w0 = crate::special::lambert_w0(-0.25).unwrap();
        assert!(iso.roots[0].lo <= wm1 + 1e-15 && wm1 - 1e-15 <= iso.roots[0].hi);
        assert!(iso.roots[1].lo <= w0 + 1e-15 && w0 - 1e-15 <= iso.roots[1].hi);
    }
}
