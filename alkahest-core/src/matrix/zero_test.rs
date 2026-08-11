//! Three-valued zero testing for symbolic matrix entries.
//!
//! # Why a third value
//!
//! Gaussian elimination has to answer one question about every candidate
//! pivot: *is this entry zero?*  Over ℚ that question is decidable and the
//! answer is a `bool`.  Over a transcendental extension it is not: deciding
//! whether an expression built from `exp`, `log` and the field operations
//! vanishes identically is undecidable in general (Richardson's theorem), and
//! even the decidable fragments are only as strong as the normaliser in front
//! of them.
//!
//! The elimination code used to ask `expr_is_zero`, a `bool` that answered
//! "the simplified entry is not the literal `0`".  Collapsing *unknown* into
//! *non-zero* is what made `Matrix::rank` report 2 for
//!
//! ```text
//! [ 1       exp(a)   exp(a)   ]
//! [ exp(a)  exp(a)²  exp(2a)  ]
//! ```
//!
//! whose second row is exactly `exp(a)` times its first, so the true rank is 1.
//! The entry that should have been recognised as zero was
//! `exp(2a) − exp(a)·exp(a)`; because it was not, elimination "cleared" a
//! column it had not cleared and produced the `[0 0 1]` row of an inconsistent
//! system for a consistent one.  No exception, no flag — the worst failure
//! class there is.
//!
//! [`ZeroStatus`] therefore has three values and the callers decide what to do
//! with [`ZeroStatus::Unknown`].  The contract every pivot-selecting routine in
//! this crate now follows is:
//!
//! * pivot **only** on [`ZeroStatus::NonZero`] — an entry proven not to vanish
//!   identically;
//! * skip **only** on [`ZeroStatus::Zero`] — an entry proven to vanish;
//! * refuse with a coded error on [`ZeroStatus::Unknown`].
//!
//! A refusal closes one branch of a search; a wrong rank poisons every
//! derivation downstream of it.
//!
//! # How each verdict is established
//!
//! `Zero` is symbolic: a ladder of increasingly aggressive normalisers, each
//! sound (they only rewrite to equal expressions), and the verdict is taken
//! only when one of them reaches the literal `0`.
//!
//! `NonZero` is numeric but **rigorous**: the entry is evaluated in ball
//! arithmetic ([`crate::ball`], outward-rounded so the true value is always
//! enclosed) at a fixed set of sample points.  If some enclosure excludes `0`,
//! the entry provably takes a non-zero value somewhere and therefore is not the
//! zero function.  That is exactly the condition a *generic* pivot needs.
//!
//! Note what `NonZero` does and does not say: `x − 1` is `NonZero` because it
//! is not identically zero, so pivoting on it is the usual generic-rank
//! semantics shared with every other CAS (the answer is the rank for all but a
//! measure-zero set of parameter values).  What is now impossible is pivoting
//! on an entry that is identically zero, which is the case that turns a
//! consistent system into an inconsistent-looking one.

use crate::ball::{ArbBall, IntervalEval};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::{
    simplify, simplify_expanded, simplify_log_exp, simplify_trig_normal_form,
};
use rug::{Float, Rational};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;

/// Working precision (bits) for the non-vanishing certificate.
const PROBE_PREC: u32 = 128;

/// Number of distinct sample points tried before giving up on a certificate.
///
/// More than one because a non-zero entry can still vanish *at* a point
/// (`x − c` at `x = c`); the samples are chosen to make that unlikely, and a
/// second and third point make it unlikely twice more.
const PROBE_ROUNDS: usize = 3;

/// Give up rather than probe a wide parameter space.
const MAX_PROBE_SYMBOLS: usize = 16;

/// Whether an expression is identically zero, where "I cannot tell" is a
/// first-class answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZeroStatus {
    /// Proven to be the zero expression by a sound normalisation.
    Zero,
    /// Proven **not** to be the zero expression: a rigorous enclosure at some
    /// sample point excludes `0`, or the entry is a non-zero literal.
    NonZero,
    /// Neither could be established. Callers must not treat this as either.
    Unknown,
}

impl ZeroStatus {
    /// True only for [`ZeroStatus::Zero`].
    ///
    /// For call sites where an inconclusive answer is *safe* to treat as
    /// "not zero" — a conservative test that only ever gives up an
    /// optimisation, never a mathematical claim.
    pub(crate) fn is_proven_zero(self) -> bool {
        matches!(self, ZeroStatus::Zero)
    }
}

// ---------------------------------------------------------------------------
// Refusals, reported out of band
// ---------------------------------------------------------------------------

/// Which routine refused, which fixes the stable code the refusal carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefusalSite {
    /// A quantity an elimination-family routine has to decide before it can go
    /// on: a candidate pivot, a Gram–Schmidt norm, an entry of `p(M)` in the
    /// minimal-polynomial search — `E-LINALG-010`.
    Pivot,
    /// The determinant of a matrix being inverted — `E-MAT-004`.
    Determinant,
}

/// A zero test that could be settled neither way, with the code it carries.
///
/// # Why this is not an error variant
///
/// [`MatrixError`](crate::matrix::MatrixError) and
/// [`LinearAlgebraError`](crate::matrix::LinearAlgebraError) are public
/// *exhaustive* enums, so growing either of them a `ZeroTestInconclusive`
/// variant is a major semver break — and so is marking them `#[non_exhaustive]`
/// to allow it later. A correctness fix inside a patch release cannot spend a
/// major version, so the refusal travels out of band instead: the refusing
/// routine returns the existing variant whose meaning covers the case
/// (`LinearAlgebraError::UnsupportedField` for a pivot — the entries lie in a
/// field this routine cannot decide over — and `MatrixError::SingularMatrix`
/// for a determinant, whose reworded text states exactly the disjunction that
/// is known), and the real cause is recorded here for
/// [`take_zero_test_refusal`] to hand to the bindings.
///
/// This is the pattern
/// [`crate::calculus::limits::last_budget_trip`] already uses for budget trips
/// inside `LimitError::DepthExceeded`, and `integrate` for budget trips inside
/// `IntegrationError::NotImplemented`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroTestRefusal {
    entry: String,
    site: RefusalSite,
}

impl ZeroTestRefusal {
    /// The expression whose vanishing could not be decided, rendered.
    pub fn entry(&self) -> &str {
        &self.entry
    }
}

impl fmt::Display for ZeroTestRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.site {
            RefusalSite::Pivot => write!(
                f,
                "cannot decide whether the entry `{}` is zero; refusing rather \
                 than report a rank, a factorisation or a minimal polynomial \
                 that silently assumes an answer",
                self.entry
            ),
            RefusalSite::Determinant => write!(
                f,
                "cannot decide whether the determinant `{}` is zero; refusing to \
                 report an inverse that assumes it is not",
                self.entry
            ),
        }
    }
}

impl std::error::Error for ZeroTestRefusal {}

impl crate::errors::AlkahestError for ZeroTestRefusal {
    fn code(&self) -> &'static str {
        match self.site {
            RefusalSite::Pivot => "E-LINALG-010",
            RefusalSite::Determinant => "E-MAT-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self.site {
            RefusalSite::Pivot => Some(
                "rewrite the entry into a form whose vanishing is decidable, or \
                 substitute concrete values for the parameters",
            ),
            RefusalSite::Determinant => Some(
                "rewrite the entries into a form whose determinant's vanishing is \
                 decidable, or substitute concrete values",
            ),
        }
    }
}

thread_local! {
    /// The refusal behind the error the current thread is about to return, if
    /// that error is a zero-test refusal rather than the thing its variant
    /// usually means.
    static LAST_REFUSAL: RefCell<Option<ZeroTestRefusal>> = const { RefCell::new(None) };
}

/// Record `e` as undecided and hand the refusal to the caller to attach to its
/// own error variant.
pub(crate) fn record_refusal(pool: &ExprPool, e: ExprId, site: RefusalSite) {
    let refusal = ZeroTestRefusal {
        entry: pool.display(e).to_string(),
        site,
    };
    LAST_REFUSAL.with(|c| *c.borrow_mut() = Some(refusal));
}

/// Drop any recorded refusal.
///
/// Called wherever one of the carrier variants is returned for its *original*
/// meaning — a genuinely singular matrix, entries that are genuinely not
/// rational — so that error can never be re-attributed to an undecided zero
/// test left behind by an earlier call on this thread.
pub(crate) fn forget_refusal() {
    LAST_REFUSAL.with(|c| *c.borrow_mut() = None);
}

/// Take the refusal behind the error that just came back, if there was one.
///
/// Bindings call this when they see one of the carrier variants
/// (`LinearAlgebraError::UnsupportedField`, `MatrixError::SingularMatrix`) and
/// raise the refusal's own `E-LINALG-010` / `E-MAT-004` when it is present, so
/// a caller still gets the specific code. `Some` means *this* error is a
/// refusal; `None` means the variant means what it usually means.
///
/// Consuming, so one refusal is reported once and cannot leak into a later
/// unrelated error. Thread-local, like the zero test itself.
pub fn take_zero_test_refusal() -> Option<ZeroTestRefusal> {
    LAST_REFUSAL.with(|c| c.borrow_mut().take())
}

/// How deep the structural tier looks before handing over to the probe.
const MAX_STRUCTURAL_DEPTH: u32 = 8;

/// Decide whether `e` is identically zero.
pub(crate) fn zero_status(pool: &ExprPool, e: ExprId) -> ZeroStatus {
    status_of(pool, simplify(e, pool).value, 0)
}

fn status_of(pool: &ExprPool, e: ExprId, depth: u32) -> ZeroStatus {
    if let Some(status) = literal_status(pool, e) {
        return status;
    }
    if depth < MAX_STRUCTURAL_DEPTH {
        if let Some(status) = structural_status(pool, e, depth) {
            return status;
        }
    }
    // A non-vanishing certificate needs no normalisation at all, and in
    // elimination most surviving entries really are non-zero.
    if probe_nonzero(pool, e) {
        return ZeroStatus::NonZero;
    }
    if normalises_to_zero(pool, e) {
        return ZeroStatus::Zero;
    }
    ZeroStatus::Unknown
}

/// Verdicts that follow from the shape of `e` alone.
///
/// These are the cases where zero-ness of a compound expression is determined
/// by zero-ness of its parts, which lets the test succeed on values the ball
/// evaluator cannot reach at all. The one that matters in practice is a product
/// of radicals of negative quantities — `−2·√(−w²)`, the determinant of the
/// modal matrix of an undamped oscillator — which is complex for every real `w`
/// and so has no real enclosure, yet is obviously non-zero once you look at it
/// as a product.
///
/// Every rule here is an identity over ℂ:
/// * a product vanishes iff one of its factors does (ℂ is an integral domain);
/// * `zⁿ = 0 ⟺ z = 0` for `n > 0`, and `z⁻ⁿ` never vanishes for `z ≠ 0`;
/// * `√z = 0 ⟺ z = 0`, on either branch;
/// * `exp` has no zero anywhere in ℂ.
fn structural_status(pool: &ExprPool, e: ExprId, depth: u32) -> Option<ZeroStatus> {
    let next = depth + 1;
    match pool.get(e) {
        ExprData::Mul(args) => {
            let statuses: Vec<ZeroStatus> = args
                .iter()
                .map(|&a| status_of(pool, a, next))
                .collect::<Vec<_>>();
            if statuses.contains(&ZeroStatus::Zero) {
                Some(ZeroStatus::Zero)
            } else if statuses.iter().all(|s| *s == ZeroStatus::NonZero) {
                Some(ZeroStatus::NonZero)
            } else {
                None
            }
        }
        ExprData::Pow { base, exp } => match integer_exponent(pool, exp) {
            Some(n) if n > 0 => decisive(status_of(pool, base, next)),
            // `z⁻ⁿ` is non-zero when `z` is; when `z` is zero it is undefined
            // rather than zero, so that case gets no verdict here.
            Some(n) if n < 0 && status_of(pool, base, next) == ZeroStatus::NonZero => {
                Some(ZeroStatus::NonZero)
            }
            _ => None,
        },
        ExprData::Func { ref name, ref args } if args.len() == 1 => match name.as_str() {
            "sqrt" => decisive(status_of(pool, args[0], next)),
            "exp" if !has_opaque_constant(pool, args[0], 0) => Some(ZeroStatus::NonZero),
            _ => None,
        },
        _ => None,
    }
}

/// Forget an inconclusive verdict.
///
/// An `Unknown` from a sub-expression must not short-circuit the
/// whole-expression probe and normalisation ladder, which may still decide the
/// compound even when a part of it is undecided.
fn decisive(status: ZeroStatus) -> Option<ZeroStatus> {
    (status != ZeroStatus::Unknown).then_some(status)
}

fn integer_exponent(pool: &ExprPool, e: ExprId) -> Option<i64> {
    pool.with(e, |data| match data {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    })
}

/// Whether `e` mentions a symbol standing for a value the probe does not model
/// (`oo`, `nan`, …), or a node whose contents cannot be inspected.
///
/// Conservative in the safe direction: an answer of `true` only ever withholds
/// a verdict.
fn has_opaque_constant(pool: &ExprPool, e: ExprId, depth: u32) -> bool {
    if depth >= MAX_STRUCTURAL_DEPTH {
        return true;
    }
    let next = depth + 1;
    match pool.get(e) {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => false,
        ExprData::Symbol { ref name, .. } => OPAQUE_CONSTANTS.contains(&name.as_str()),
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| has_opaque_constant(pool, a, next))
        }
        ExprData::Pow { base, exp } => {
            has_opaque_constant(pool, base, next) || has_opaque_constant(pool, exp, next)
        }
        ExprData::Func { args, .. } => args.iter().any(|&a| has_opaque_constant(pool, a, next)),
        _ => true,
    }
}

/// A verdict read straight off a numeric literal, if `e` is one.
fn literal_status(pool: &ExprPool, e: ExprId) -> Option<ZeroStatus> {
    pool.with(e, |data| match data {
        ExprData::Integer(n) => Some(if n.0 == 0 {
            ZeroStatus::Zero
        } else {
            ZeroStatus::NonZero
        }),
        ExprData::Rational(r) => Some(if r.0 == 0 {
            ZeroStatus::Zero
        } else {
            ZeroStatus::NonZero
        }),
        ExprData::Float(f) => Some(if f.inner.is_zero() {
            ZeroStatus::Zero
        } else {
            ZeroStatus::NonZero
        }),
        _ => None,
    })
}

/// True when some sound normaliser drives `e` to the literal `0`.
///
/// The ladder is ordered by cost. Every rung is a semantics-preserving rewrite,
/// so a `0` from any of them is a proof; failing all of them proves nothing.
fn normalises_to_zero(pool: &ExprPool, e: ExprId) -> bool {
    let expanded = simplify_expanded(e, pool).value;
    if is_literal_zero(pool, expanded) {
        return true;
    }
    // `simplify_log_exp` merges `exp(x)·exp(y)` and `exp(x)^n` into a single
    // `exp` of a sum, which is what makes `exp(a)² − exp(2a)` collectible.
    for start in [e, expanded] {
        if is_literal_zero(pool, simplify_log_exp(start, pool, &[]).value) {
            return true;
        }
    }
    is_literal_zero(pool, simplify_trig_normal_form(e, pool).value)
}

fn is_literal_zero(pool: &ExprPool, e: ExprId) -> bool {
    literal_status(pool, e) == Some(ZeroStatus::Zero)
}

/// Rigorously certify that `e` is not the zero expression.
///
/// Evaluates `e` in ball arithmetic at [`PROBE_ROUNDS`] sample points. A ball
/// that excludes `0` is a proof that the true value there is non-zero, hence
/// that `e` is not identically zero. Anything else — an unsupported node, a
/// domain error, an enclosure straddling `0` — returns `false`, which the
/// caller reads as "no certificate", never as "zero".
fn probe_nonzero(pool: &ExprPool, e: ExprId) -> bool {
    let Some(symbols) = probe_symbols(pool, e) else {
        return false;
    };
    if symbols.len() > MAX_PROBE_SYMBOLS {
        return false;
    }
    for round in 0..PROBE_ROUNDS {
        let mut eval = IntervalEval::new(PROBE_PREC);
        for (index, &sym) in symbols.iter().enumerate() {
            eval.bind(sym, sample_ball(pool, sym, index, round));
        }
        if let Some(ball) = eval.eval(e, pool) {
            if ball_excludes_zero(&ball) {
                return true;
            }
        }
    }
    false
}

/// A ball that provably does not contain `0`.
fn ball_excludes_zero(ball: &ArbBall) -> bool {
    if !ball.rad.is_finite() || !ball.mid.is_finite() {
        return false;
    }
    let lo = ball.lo();
    let hi = ball.hi();
    if lo.is_nan() || hi.is_nan() {
        return false;
    }
    lo > 0.0 || hi < 0.0
}

/// The free symbols of `e`, or `None` when `e` must not be probed at all.
///
/// Returns `None` when the expression mentions a symbol whose *value* is fixed
/// by mathematics rather than free to vary. Binding `pi` to an arbitrary sample
/// would report `sin(pi)` as non-zero; binding the imaginary unit `I` to a real
/// sample would report `I² + 1` as non-zero. Both would be exactly the silent
/// error this module exists to prevent, so those expressions get no certificate.
/// `pi` itself is the one constant handled properly, in [`sample_ball`].
fn probe_symbols(pool: &ExprPool, e: ExprId) -> Option<Vec<ExprId>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    collect_symbols(pool, e, &mut seen, &mut out)?;
    out.sort_unstable();
    Some(out)
}

/// Symbol names that denote a specific number and are not handled rigorously.
const OPAQUE_CONSTANTS: &[&str] = &["oo", "inf", "infinity", "zoo", "nan", "NaN"];

fn collect_symbols(
    pool: &ExprPool,
    e: ExprId,
    seen: &mut HashSet<ExprId>,
    out: &mut Vec<ExprId>,
) -> Option<()> {
    if !seen.insert(e) {
        return Some(());
    }
    match pool.get(e) {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => Some(()),
        ExprData::Symbol { ref name, .. } => {
            if pool.is_imaginary_unit(e) || OPAQUE_CONSTANTS.contains(&name.as_str()) {
                return None;
            }
            out.push(e);
            Some(())
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect_symbols(pool, a, seen, out)?;
            }
            Some(())
        }
        ExprData::Pow { base, exp } => {
            collect_symbols(pool, base, seen, out)?;
            collect_symbols(pool, exp, seen, out)
        }
        ExprData::Func { args, .. } => {
            for a in args {
                collect_symbols(pool, a, seen, out)?;
            }
            Some(())
        }
        // Anything else (`Piecewise`, `BigO`, `RootSum`, quantifiers, …) is
        // either unsupported by the ball evaluator or has binding structure
        // that would make a free-symbol list meaningless.
        _ => None,
    }
}

/// The sample value bound to `sym` in probe round `round`.
///
/// Deterministic — the same matrix always produces the same verdict, and a
/// refusal is reproducible rather than flaky. The points are ratios with a
/// large prime-ish denominator so that no small algebraic relation between two
/// symbols (`x − y`, `x − 2`, `2x − y`) holds accidentally.
fn sample_ball(pool: &ExprPool, sym: ExprId, index: usize, round: usize) -> ArbBall {
    if pool.with(
        sym,
        |data| matches!(data, ExprData::Symbol { name, .. } if name.as_str() == "pi"),
    ) {
        return pi_ball();
    }
    let integral = pool.with(sym, |data| {
        matches!(
            data,
            ExprData::Symbol {
                domain: Domain::Integer,
                ..
            }
        )
    });
    if integral {
        // An integer-domain symbol must be sampled at an integer: identities
        // such as `sin(pi·n) = 0` hold only there, and a fractional sample
        // would "certify" them non-zero.
        let n = 7 + 11 * index as i64 + 101 * round as i64;
        return ArbBall::from_integer(&rug::Integer::from(n), PROBE_PREC);
    }
    let numer = 733 + 269 * index as i64 + 1123 * round as i64;
    let denom = 1021 + 7 * round as i64;
    ArbBall::from_rational(&Rational::from((numer, denom)), PROBE_PREC)
}

/// An enclosure of π that is honest about its own error.
///
/// `ArbBall::from_f64` would claim radius `0`, i.e. that the sample *is* π, and
/// the enclosure would then no longer be rigorous.
fn pi_ball() -> ArbBall {
    // Same construction as `ArbBall::from_rational`: round at working
    // precision, then take the distance to a much more accurate value as the
    // radius.
    let mid = Float::with_val(PROBE_PREC, rug::float::Constant::Pi);
    let accurate = Float::with_val(PROBE_PREC * 2, rug::float::Constant::Pi);
    let rad = Float::with_val(PROBE_PREC, &accurate - &mid).abs();
    ArbBall {
        mid,
        rad,
        prec: PROBE_PREC,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn literal_zero_is_zero() {
        let pool = p();
        assert_eq!(
            zero_status(&pool, pool.integer(0_i32)),
            ZeroStatus::Zero,
            "the literal 0"
        );
        assert_eq!(zero_status(&pool, pool.integer(3_i32)), ZeroStatus::NonZero);
    }

    #[test]
    fn symbol_is_generically_nonzero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        assert_eq!(zero_status(&pool, x), ZeroStatus::NonZero);
    }

    #[test]
    fn exp_square_minus_exp_double_is_zero() {
        // The regression: exp(a)·exp(a) − exp(a+a) is the zero function.
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let ea = pool.func("exp", vec![a]);
        let lhs = pool.mul(vec![ea, ea]);
        let rhs = pool.func("exp", vec![pool.add(vec![a, a])]);
        let diff = pool.add(vec![lhs, pool.mul(vec![pool.integer(-1_i32), rhs])]);
        assert_eq!(zero_status(&pool, diff), ZeroStatus::Zero);
    }

    #[test]
    fn non_identity_difference_of_exps_is_nonzero() {
        // The control: exp(a)·exp(a) − exp(a) is *not* the zero function.
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let ea = pool.func("exp", vec![a]);
        let lhs = pool.mul(vec![ea, ea]);
        let diff = pool.add(vec![lhs, pool.mul(vec![pool.integer(-1_i32), ea])]);
        assert_eq!(zero_status(&pool, diff), ZeroStatus::NonZero);
    }

    #[test]
    fn sin_of_pi_is_not_certified_nonzero() {
        // Binding `pi` to an arbitrary sample would "prove" sin(pi) ≠ 0.
        let pool = p();
        let pi = pool.symbol("pi", Domain::Real);
        let e = pool.func("sin", vec![pi]);
        assert_ne!(zero_status(&pool, e), ZeroStatus::NonZero);
    }

    #[test]
    fn imaginary_unit_is_not_probed() {
        // I² + 1 = 0; a real sample for `I` would certify it non-zero.
        let pool = p();
        let i = pool.imaginary_unit();
        let e = pool.add(vec![pool.mul(vec![i, i]), pool.integer(1_i32)]);
        assert_ne!(zero_status(&pool, e), ZeroStatus::NonZero);
    }

    #[test]
    fn unknown_function_yields_unknown() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("mystery", vec![x]);
        let g = pool.func("mystery", vec![pool.mul(vec![pool.integer(1_i32), x])]);
        // f(x) − g(x) is zero, but nothing here can prove it either way once
        // the two arguments are written differently; what matters is that the
        // answer is not a confident `NonZero`.
        let diff = pool.add(vec![f, pool.mul(vec![pool.integer(-1_i32), g])]);
        assert_ne!(zero_status(&pool, diff), ZeroStatus::NonZero);
    }
}
