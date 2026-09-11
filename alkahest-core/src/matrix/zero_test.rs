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
///
/// 144 is `12×12` distinct entries: enough that a fully symbolic matrix of the
/// size controls work actually uses gets a verdict. At 16 a `5×5` matrix of
/// distinct symbols — 25 of them — was already over the line, which is why
/// `inverse()` on a symbolic `5×5` or `6×6` refused with `E-MAT-004`
/// ("cannot decide whether the determinant is zero") about a determinant that
/// is a 720-term polynomial and manifestly not the zero function. A `4×4`
/// (16 symbols) answered.
///
/// Raising it cannot make a verdict wrong. `NonZero` is only ever returned on
/// a **rigorous** enclosure that excludes `0` at a sample point, which is a
/// proof that the expression is not identically zero however many symbols it
/// has; the cap never contributed to soundness, only to cost. A sample that
/// happens to land on a root yields no certificate and the answer stays
/// `Unknown`, which is safe.
const MAX_PROBE_SYMBOLS: usize = 144;

/// Give up rather than probe a very large expression.
///
/// This is the bound that actually tracks cost: each round is one interval
/// evaluation, whose work is proportional to the size of the expression DAG and
/// not to the number of symbols in it. Bounding the symbol count was bounding
/// the wrong quantity — it let a huge single-variable expression through and
/// stopped a small many-variable one.
const MAX_PROBE_NODES: usize = 250_000;

/// Generator budget for the `cancel` rung of [`normalises_to_zero`].
///
/// Deliberately far below [`MAX_PROBE_SYMBOLS`]: the non-vanishing probe costs
/// one interval evaluation per round however many symbols there are, while
/// `cancel` runs multivariate GCDs whose cost explodes with the generator
/// count. 12 covers the eliminated-entry case this rung exists for.
const CANCEL_MAX_SYMBOLS: usize = 12;

/// Expression-size budget for the `cancel` rung.
const CANCEL_MAX_NODES: usize = 4_000;

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

/// Is `e` non-zero as a **number**, rather than merely not identically zero?
///
/// The distinction is the whole point, and it is not pedantry. For an
/// expression mentioning a free symbol, [`ZeroStatus::NonZero`] means only "not
/// identically zero as a function of it" — `ζ² − 1` qualifies, and still
/// vanishes at `ζ = 1`. A caller about to *divide* by `e` needs the stronger
/// claim, because dividing by something that vanishes somewhere produces an
/// answer that is wrong exactly on the set the caller was never told about.
///
/// Three tiers, cheapest first: a literal non-zero value; then a bail-out on
/// any free symbol; then the zero test, which by that point is being asked
/// about a closed-form constant (`−2√(−1)`, the eigenvalue gap of a rotation)
/// where `NonZero` really does mean non-zero.
pub(crate) fn settled_nonzero(e: ExprId, pool: &ExprPool) -> bool {
    if matches!(crate::kernel::eval_const::try_expr_f64(e, pool), Some(v) if v != 0.0) {
        return true;
    }
    if has_free_symbol(e, pool) {
        return false;
    }
    matches!(zero_status(pool, e), ZeroStatus::NonZero)
}

fn has_free_symbol(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| match d {
        ExprData::Symbol { .. } => true,
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().any(|&a| has_free_symbol(a, pool))
        }
        ExprData::Pow { base, exp } => has_free_symbol(*base, pool) || has_free_symbol(*exp, pool),
        _ => false,
    })
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
    if is_literal_zero(pool, simplify_trig_normal_form(e, pool).value) {
        return true;
    }
    let deradicalised = fold_rational_radicals(pool, e);
    if deradicalised != e {
        if is_literal_zero(pool, simplify_expanded(deradicalised, pool).value) {
            return true;
        }
        if cancels_to_zero(pool, deradicalised) {
            return true;
        }
    }
    cancels_to_zero(pool, e)
}

/// Rewrite `(√c)^k` as `c^{k/2}` wherever `c` is a **non-negative rational
/// literal**, leaving everything else untouched.
///
/// `simplify` does not do this: `√2·√2` comes back as `√2²` and `1/√10²` as
/// `√10⁻²`, so `√2·√2 − 2` — the residual of `L·Lᵀ − M` for the Cholesky factor
/// of `2I`, and the shape every Gram–Schmidt norm in [`qr_decomposition`]
/// produces — is not recognised as zero. Over a *non-negative rational* base
/// the rewrite needs no branch convention at all: `√c` is the non-negative real
/// root, so `(√c)^k = c^{k/2}` holds outright, for negative `k` as well.
///
/// Restricting to rational literals is what keeps it sound. `(√x)² = x` is
/// false for `x < 0` under the principal branch (it is `|x|` on the reals and
/// picks up a sign in ℂ), so a symbolic base is left alone rather than
/// rewritten under an assumption nobody stated.
///
/// [`qr_decomposition`]: crate::matrix::qr_decomposition
fn fold_rational_radicals(pool: &ExprPool, e: ExprId) -> ExprId {
    fold_rational_radicals_at(pool, e, 0)
}

fn fold_rational_radicals_at(pool: &ExprPool, e: ExprId, depth: u32) -> ExprId {
    if depth > MAX_STRUCTURAL_DEPTH {
        return e;
    }
    let d = depth + 1;
    match pool.get(e) {
        ExprData::Add(args) => {
            let next: Vec<ExprId> = args
                .iter()
                .map(|&a| fold_rational_radicals_at(pool, a, d))
                .collect();
            pool.add(next)
        }
        ExprData::Mul(args) => {
            let next: Vec<ExprId> = args
                .iter()
                .map(|&a| fold_rational_radicals_at(pool, a, d))
                .collect();
            pool.mul(next)
        }
        ExprData::Func { name, args } => {
            let next: Vec<ExprId> = args
                .iter()
                .map(|&a| fold_rational_radicals_at(pool, a, d))
                .collect();
            pool.func(name.as_str(), next)
        }
        ExprData::Pow { base, exp } => {
            let (base, exp) = (base, exp);
            let folded_base = fold_rational_radicals_at(pool, base, d);
            let folded_exp = fold_rational_radicals_at(pool, exp, d);
            if let (Some(radicand), Some(k)) = (
                sqrt_of_nonneg_rational(pool, folded_base),
                integer_of(pool, folded_exp),
            ) {
                let half = Rational::from((k, rug::Integer::from(2)));
                let new_exp = if *half.denom() == 1 {
                    pool.integer(half.numer().clone())
                } else {
                    pool.rational(half.numer().clone(), half.denom().clone())
                };
                let rad = pool.rational(radicand.numer().clone(), radicand.denom().clone());
                return pool.pow(rad, new_exp);
            }
            pool.pow(folded_base, folded_exp)
        }
        _ => e,
    }
}

/// `Some(c)` when `e` is `sqrt(c)` or `c^(1/2)` for a non-negative rational `c`.
fn sqrt_of_nonneg_rational(pool: &ExprPool, e: ExprId) -> Option<Rational> {
    let inner = pool.with(e, |d| match d {
        ExprData::Func { name, args } if name.as_str() == "sqrt" && args.len() == 1 => {
            Some(args[0])
        }
        ExprData::Pow { base, exp } => {
            let half = pool.with(*exp, |x| match x {
                ExprData::Rational(r) => *r.0.numer() == 1 && *r.0.denom() == 2,
                _ => false,
            });
            if half {
                Some(*base)
            } else {
                None
            }
        }
        _ => None,
    })?;
    let r = rational_of(pool, inner)?;
    if r < 0 {
        return None;
    }
    Some(r)
}

fn rational_of(pool: &ExprPool, e: ExprId) -> Option<Rational> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(Rational::from((n.0.clone(), rug::Integer::from(1)))),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    })
}

fn integer_of(pool: &ExprPool, e: ExprId) -> Option<rug::Integer> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(n.0.clone()),
        _ => None,
    })
}

/// Last rung: put `e` over a common denominator and cancel the GCD.
///
/// Elimination does not produce polynomials, it produces **rational
/// functions** — every entry after a pivot step carries a `1/pivot` factor —
/// and `expand` alone cannot see that a sum of them vanishes. `A⁻¹·A` for a
/// symbolic `A` is the standard example: entry `(0,0)` of the `2×2` case is
/// `a·d·(ad−bc)⁻¹ − b·c·(ad−bc)⁻¹ − 1`, which is `0` and which every rung above
/// this one answers `Unknown` for, because none of them combines the two
/// quotients.
///
/// Sound in the only direction it is used. [`crate::poly::cancel::cancel`]
/// works over a generator list in which anything it does not recognise —
/// `sin(x)`, `√2`, `x^n` — is an *opaque* generator, i.e. treated as an
/// independent transcendental. A zero numerator in that free ring is therefore
/// zero for every substitution, so `true` is a proof. The converse does not
/// hold and is not claimed: `√2·√2 − 2` is `g² − 2` over an opaque `g` and
/// stays `Unknown`, which is a missed `Zero`, never a wrong verdict.
///
/// Placed last because it is by far the most expensive rung, and budgeted
/// separately and much more tightly than [`probe_nonzero`].
///
/// The cost is not the node count. `cancel` runs multivariate polynomial GCDs
/// over the generator list, which is super-linear in the *number of
/// generators*: the residual of `A⁻¹·A` for a fully symbolic `6×6` has 36 of
/// them over a 720-term determinant, and the reduction does not finish in any
/// useful time. [`CANCEL_MAX_SYMBOLS`] and [`CANCEL_MAX_NODES`] are therefore
/// sized for what actually motivates this rung — a handful of parameters in an
/// eliminated matrix entry — and not for a dense symbolic determinant, which
/// is left `Unknown` and hence a refusal.
fn cancels_to_zero(pool: &ExprPool, e: ExprId) -> bool {
    let Some((symbols, nodes)) = probe_symbols(pool, e) else {
        return false;
    };
    if symbols.len() > CANCEL_MAX_SYMBOLS || nodes > CANCEL_MAX_NODES {
        return false;
    }
    match crate::poly::cancel::cancel(e, Vec::new(), pool) {
        Ok(c) => is_literal_zero(pool, simplify(c, pool).value),
        Err(_) => false,
    }
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
    let Some((symbols, nodes)) = probe_symbols(pool, e) else {
        return false;
    };
    if symbols.len() > MAX_PROBE_SYMBOLS || nodes > MAX_PROBE_NODES {
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
/// Returns the symbols together with the number of distinct DAG nodes walked,
/// which is what [`MAX_PROBE_NODES`] bounds.
fn probe_symbols(pool: &ExprPool, e: ExprId) -> Option<(Vec<ExprId>, usize)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    collect_symbols(pool, e, &mut seen, &mut out)?;
    out.sort_unstable();
    Some((out, seen.len()))
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
///
/// # Why the numerator is scrambled rather than counted up
///
/// It used to be `733 + 269·index`, which is *linear in the index* — and that
/// is itself an algebraic relation, just one between three symbols rather than
/// two: `x_{k+1} − 2x_k + x_{k−1} = 0` holds identically for every consecutive
/// triple. Any expression that vanishes on collinear inputs therefore vanished
/// at every sample, in every round, since the round only shifted all the
/// numerators by the same `1123·round` and scaled the denominator.
///
/// The expression that vanishes on collinear inputs is **the determinant of a
/// dense matrix**. Binding the `n²` entries of a symbolic matrix in index order
/// produced a probe matrix whose rows are in arithmetic progression, i.e. of
/// rank 2, so `det` evaluated to exactly `0` for every `n ≥ 3` — verified
/// directly: the probe matrices for `n = 3..6` and rounds `0..2` all have
/// rank 2 and determinant 0. The probe could not certify a single dense
/// symbolic determinant as non-vanishing, and `inverse`, `rank`, `rref` and
/// `nullspace` refused the whole class with `E-MAT-004`/`E-LINALG-010`.
///
/// The scrambled sequence below is a xorshift keyed by `(index, round)`: still
/// deterministic and still reproducible, but with no low-degree relation among
/// the values. This is the substantive half of supporting the generic symbolic
/// matrix; raising [`MAX_PROBE_SYMBOLS`] alone only got as far as *trying*.
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
        // Scrambled for the same reason as the rational branch: `7 + 11·index`
        // is an arithmetic progression, and a determinant vanishes on one.
        let n = 7 + scrambled(index, round) % 977;
        return ArbBall::from_integer(&rug::Integer::from(n), PROBE_PREC);
    }
    let numer = 733 + scrambled(index, round);
    let denom = 1021 + 7 * round as i64;
    ArbBall::from_rational(&Rational::from((numer, denom)), PROBE_PREC)
}

/// A deterministic, low-structure offset for `(index, round)`.
///
/// xorshift64 over a seed mixing both, reduced to a few thousand. Deterministic
/// by construction — no RNG state, no clock — so a verdict is reproducible; and
/// unlike an arithmetic progression it satisfies no low-degree relation across
/// consecutive indices, which is the property the probe actually needs.
fn scrambled(index: usize, round: usize) -> i64 {
    let mut x = (index as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((round as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x % 7919) as i64
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

    /// The determinant of a dense matrix of distinct symbols is certified
    /// non-vanishing, at every size the probe budget covers.
    ///
    /// This is the test the collinear samples failed. `733 + 269·index` puts
    /// the sample values in arithmetic progression, so binding the `n²` entries
    /// of a symbolic matrix in index order produced a probe matrix of **rank 2**
    /// — verified directly with exact rational arithmetic for `n = 3..6` and
    /// every round — whose determinant is exactly `0`. The enclosure therefore
    /// contained `0` at every sample, no certificate was ever issued, and the
    /// verdict was `Unknown`: `inverse` refused a dense symbolic `3×3` with
    /// `E-MAT-004`, and `rank`/`rref`/`nullspace` refused the same class with
    /// `E-LINALG-010`.
    ///
    /// `det` of a matrix of `n²` distinct symbols is a sum of `n!` distinct
    /// monomials over ℤ and is not the zero polynomial for any `n ≥ 1`, so
    /// `NonZero` is the only correct verdict here and `Unknown` is a pure loss.
    #[test]
    fn dense_symbolic_determinant_is_certified_non_vanishing() {
        for n in 1..=6usize {
            let pool = p();
            let grid: Vec<Vec<crate::kernel::ExprId>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| pool.symbol(format!("s_{i}_{j}"), Domain::Complex))
                        .collect()
                })
                .collect();
            let m = crate::matrix::Matrix::new(grid).unwrap();
            let det =
                crate::simplify::engine::simplify_expanded(m.det(&pool).unwrap(), &pool).value;
            assert_eq!(
                zero_status(&pool, det),
                ZeroStatus::NonZero,
                "det of a dense symbolic {n}×{n} is a sum of {n}! distinct monomials \
                 and cannot be the zero function"
            );
        }
    }

    /// The samples themselves are in general position.
    ///
    /// Stated on the generator rather than on a consequence of it, so a future
    /// change to the sequence is checked against the property that matters
    /// rather than against one matrix that happens to exercise it: no three
    /// consecutive numerators may be collinear, which is exactly the relation
    /// `x_{k+1} − 2x_k + x_{k−1} = 0` that an arithmetic progression satisfies
    /// and that a determinant vanishes on.
    #[test]
    fn probe_samples_are_not_collinear_in_the_index() {
        for round in 0..PROBE_ROUNDS {
            let vals: Vec<i64> = (0..40).map(|i| scrambled(i, round)).collect();
            let mut collinear = 0usize;
            for w in vals.windows(3) {
                if w[2] - 2 * w[1] + w[0] == 0 {
                    collinear += 1;
                }
            }
            assert!(
                collinear < 3,
                "round {round}: {collinear} of 38 consecutive triples are collinear; \
                 an arithmetic progression would give 38"
            );
            // And they must stay distinct, or two symbols get the same value and
            // `x − y` looks like zero.
            let mut sorted = vals.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), vals.len(), "round {round}: duplicate samples");
        }
    }

    /// A rational function that vanishes identically is proven zero.
    ///
    /// `a/(a+b) + b/(a+b) − 1` is `0` for every `(a, b)` with `a + b ≠ 0`, and
    /// no rung above the `cancel` one sees it: `expand` cannot combine two
    /// quotients, and the log/exp and trig normalisers have nothing to do here.
    /// This is the shape every entry of an eliminated symbolic matrix has.
    #[test]
    fn a_vanishing_rational_function_is_proven_zero() {
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let sum = pool.add(vec![a, b]);
        let inv = pool.pow(sum, pool.integer(-1_i32));
        let e = pool.add(vec![
            pool.mul(vec![a, inv]),
            pool.mul(vec![b, inv]),
            pool.integer(-1_i32),
        ]);
        assert_eq!(zero_status(&pool, e), ZeroStatus::Zero);
    }

    /// `√c·√c − c` is proven zero for a non-negative rational `c`.
    ///
    /// `simplify` leaves `√2·√2` as `√2²`, so without the radical fold the
    /// residual of `L·Lᵀ − M` for the Cholesky factor of `2I` — and every
    /// Gram–Schmidt norm in `qr_decomposition` — is `Unknown`.
    #[test]
    fn a_squared_rational_radical_is_proven_zero() {
        let pool = p();
        for c in [2_i32, 3, 10, 12] {
            let lit = pool.integer(c);
            for root in [
                pool.func("sqrt", vec![lit]),
                pool.pow(lit, pool.rational(1, 2)),
            ] {
                let e = pool.add(vec![
                    pool.mul(vec![root, root]),
                    pool.mul(vec![pool.integer(-1_i32), lit]),
                ]);
                assert_eq!(
                    zero_status(&pool, e),
                    ZeroStatus::Zero,
                    "√{c}·√{c} − {c} = 0"
                );
            }
        }
    }

    /// The radical fold does not fire on a base whose sign is unknown.
    ///
    /// `(√x)² = x` is false for `x < 0` under the principal branch, so a
    /// symbolic base must be left alone. Asserting the *absence* of a `Zero`
    /// verdict is the point: a rewrite that is sound only on a region, applied
    /// unconditionally, is how a normaliser turns into a silent error.
    #[test]
    fn the_radical_fold_does_not_touch_a_symbolic_base() {
        let pool = p();
        let x = pool.symbol("x", Domain::Complex);
        let root = pool.func("sqrt", vec![x]);
        let e = pool.add(vec![
            pool.mul(vec![root, root]),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        // Whatever the verdict is, it must not come from folding `(√x)² → x`.
        let folded = fold_rational_radicals(&pool, e);
        assert_eq!(folded, e, "a symbolic radicand must not be folded");
    }

    /// A negative rational radicand is left alone too: `(√−4)² = −4` holds in ℂ
    /// but `√−4` is not on the branch the fold assumes, so it is out of scope.
    #[test]
    fn the_radical_fold_does_not_touch_a_negative_radicand() {
        let pool = p();
        let neg = pool.integer(-4_i32);
        let root = pool.func("sqrt", vec![neg]);
        let e = pool.pow(root, pool.integer(2_i32));
        assert_eq!(fold_rational_radicals(&pool, e), e);
    }
}
