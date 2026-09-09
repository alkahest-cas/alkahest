//! Puiseux expansion — truncated series in **fractional** powers of `h = var − point`.
//!
//! [`crate::calculus::series::series`] expands in integer powers of `h`. That
//! covers Taylor and Laurent, and refuses everything else with
//! [`crate::calculus::series::SeriesRefusalCause::IndeterminateCoefficient`]
//! (`E-SERIES-004`) — because a coefficient of `√x` at `0` is the literal
//! `sqrt(0)⁻¹`, which is not a number. The refusal is honest but the expansion
//! exists: `√x = x^{1/2}`, `√(sin x) = x^{1/2}(1 − x²/12 + x⁴/1440 − ⋯)`.
//!
//! A **Puiseux series** at `h = 0` is a finite-tail sum `Σ_k c_k h^{e_k}` whose
//! exponents `e_k` are rationals with a common denominator `e` — the
//! **ramification index**. `e = 1` recovers Laurent. This module computes one,
//! and — this is the whole point — **checks it before returning it**.
//!
//! # Why a separate type rather than a wider [`crate::calculus::series::Series`]
//!
//! `Series` is `pub struct Series(pub ExprId)`: a publicly-constructible tuple
//! struct. Giving it a ramification field is a major semver break
//! (`cargo semver-checks` blocks it), and `#[non_exhaustive]` cannot be added
//! retroactively either. The *internal* `series::LocalExpansion`
//! carries `valuation: i32`; widening that to a rational is a one-line change
//! and a change to the meaning of the value every consumer of `local_expansion`
//! reads — `limit`, `gruntz`, `asymptotic`, `fps`. `limit` in particular scans
//! `LocalExpansion` for the first nonzero coefficient, so a shifted valuation
//! convention there is a silently different *limit*, which is exactly the class
//! of defect this library refuses to ship.
//!
//! So: [`PuiseuxExpansion`] is a sibling. `series` keeps refusing what it
//! always refused, every existing consumer is untouched, and the new capability
//! is reached through the new entry point. The cost is that the two share only
//! `local_expansion` (which this module calls, unmodified, for every analytic
//! sub-part) rather than one representation.
//!
//! # Algorithm
//!
//! Recursive Puiseux arithmetic over the expression tree, with the *existing*
//! integer-exponent engine used as the leaf oracle wherever it succeeds:
//!
//! * a node whose ordinary Laurent expansion has usable coefficients is taken
//!   from `local_expansion` verbatim — that is how `sin`, `exp`, `tan`,
//!   `1/(1+x)`, `√(1+x)` and every other analytic head enter;
//! * `Add` / `Mul` are series addition and convolution, with the truncation
//!   exponent propagated exactly (`prec(ab) = min(prec_a + val_b, prec_b + val_a)`);
//! * `f^{p/q}` is the branch that makes the exponents fractional. Write the
//!   base as `f = c₀ h^v (1 + u)` with `val(u) > 0`; then
//!   `f^{p/q} = c₀^{p/q} h^{v·p/q} (1 + u)^{p/q}` and the last factor is the
//!   binomial series `Σ_k C(p/q, k) u^k`, which terminates against the
//!   truncation exponent because `val(u^k) = k·val(u)` grows.
//!   `v` is whatever the base's expansion says it is, so the valuation
//!   `v·p/q` — and hence the ramification — is **computed, not assumed**;
//! * `g(f)` for any other head is the Taylor series of `g` at `f`'s constant
//!   term, composed with `f − c` by Horner. A base with a *negative* exponent
//!   inside a function head is refused, which is what makes `e^{1/x}` and
//!   `sin(1/x)` errors rather than truncated fiction.
//!
//! # What is refused, and why that is right
//!
//! | input | reason |
//! |---|---|
//! | `log x`, `√x·log x` | a logarithm is **not** a Puiseux series. `√x·log x` needs a Puiseux–log (transseries) representation this module does not have; truncating it to `x^{1/2}` would be a wrong answer, not a coarse one |
//! | `e^{1/x}`, `sin(1/x)` | essential singularity: no exponent is a lower bound |
//! | ramification `> `[`MAX_RAMIFICATION`] | the exponent lattice is too fine for the returned expansion to be checked apart from a neighbouring wrong one; see `verify` |
//! | anything the verifier could not conclude on | [`PuiseuxError::Unverified`] — an expansion that was computed but not confirmed is withheld |
//!
//! # Verification
//!
//! Nothing leaves this module unchecked. See `verify`.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use rug::{Integer, Rational};

use crate::budget::BudgetError;
use crate::calculus::series::{
    enter_coeff_ceiling, expansion_increment, first_indeterminate, is_structural_zero,
    local_expansion, SeriesError, MAX_SERIES_POOL_GROWTH,
};
use crate::kernel::{subs, Domain, ExprData, ExprId, ExprPool};
use crate::simplify::simplify;

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/// Largest ramification index `e` this module will return an expansion for.
///
/// Not a performance knob — a **verification** knob. The numeric check
/// distinguishes a correct expansion from one whose first wrong coefficient
/// sits at exponent `order − 1/e`, so the decay exponents it must tell apart
/// differ by `1/e`. Past this the two are inside the noise of an `f64`
/// residual and the check stops being able to fail, which would turn it into
/// decoration. A larger `e` is refused with
/// [`NotPuiseuxReason::UnboundedRamification`] rather than returned unchecked.
pub const MAX_RAMIFICATION: u64 = 8;

/// Recursion depth ceiling for the expander.
const MAX_DEPTH: u32 = 24;

/// Most binomial / composition terms one node may accumulate.
const MAX_SERIES_TERMS: u32 = 96;

/// Most `local_expansion` re-tries when the leaf oracle came back short.
const MAX_LEAF_RETRIES: u32 = 4;

/// Largest `order` the leaf oracle is ever asked for.
const MAX_LEAF_ORDER: u32 = 256;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a [`puiseux_series`] call did not return an expansion.
///
/// `#[non_exhaustive]` from birth: a future refusal reason must be an additive
/// change, which is the mistake [`crate::calculus::series::SeriesError`] had to
/// work around by carrying its refusals out of band.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PuiseuxError {
    /// `order` was `0`. Reported as `E-SERIES-002`, the same user error
    /// [`crate::calculus::series::series`] reports.
    InvalidOrder,
    /// Differentiation failed while forming coefficients — `E-SERIES-001`.
    Diff(String),
    /// The expression has no Puiseux expansion at this point — `E-SERIES-005`.
    NotPuiseux(NotPuiseuxReason),
    /// The expansion could not be carried to the requested `order` within the
    /// work ceiling, or the [`crate::budget`] carried here ran out —
    /// `E-SERIES-003`, or the budget's own `E-BUDGET-*` when it was a budget.
    ///
    /// Attributing the two apart matters to a caller deciding what to do next:
    /// "raise your budget" and "this expression's expansion does not close" are
    /// different problems.
    Exhausted(Option<BudgetError>),
    /// An expansion **was** computed and then **withheld**, because the
    /// verifier could not confirm it — `E-SERIES-006`.
    ///
    /// This is not the same as "no expansion exists". It is the prime
    /// directive: an unconfirmed expansion is indistinguishable, to the
    /// caller, from a confirmed one, so it must not be returned.
    Unverified(UnverifiedReason),
}

/// Which non-Puiseux shape was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NotPuiseuxReason {
    /// A logarithm of something vanishing at the point. `log x` is not a
    /// Puiseux series at `0` and neither is `√x·log x`: representing them needs
    /// a Puiseux–log / transseries type, which this module is not.
    Logarithmic,
    /// A function head applied to something with a **pole** at the point:
    /// `e^{1/x}`, `sin(1/x)`. There is no rational `e` with
    /// `f = O(h^{−N/e})`, so no truncation exponent is honest.
    EssentialSingularity,
    /// The exponent lattice is finer than [`MAX_RAMIFICATION`].
    UnboundedRamification,
    /// A shape the expander does not implement — a variable in an exponent
    /// (`x^x`), a `Piecewise`, a `RootSum`, an unregistered head whose Taylor
    /// coefficients could not be formed.
    UnsupportedForm,
    /// A coefficient came out as an indeterminate form (`0/0`, `0⁻¹`, `log 0`)
    /// that neither the Laurent engine nor the fractional route could clear.
    IndeterminateCoefficient,
}

/// Why `verify` could not confirm an expansion it was handed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnverifiedReason {
    /// The residual did not decay at the claimed rate: the expansion is
    /// **wrong**, or right in a way this check cannot see. Either way it is
    /// not returned.
    DecayTooSlow,
    /// Neither the original function nor the series could be evaluated on
    /// either side of the expansion point — a head with no numeric kernel, a
    /// symbolic expansion point, a branch that is not real. Nothing was
    /// measured, so nothing is claimed.
    NotEvaluable,
    /// The exact `S^e` versus `f^e` comparison ran and **disagreed**.
    PowerCheckDisagrees,
}

impl fmt::Display for PuiseuxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PuiseuxError::InvalidOrder => {
                write!(
                    f,
                    "puiseux order must be >= 1 (exclusive truncation exponent in h)"
                )
            }
            PuiseuxError::Diff(e) => write!(f, "{e}"),
            PuiseuxError::NotPuiseux(r) => write!(
                f,
                "no Puiseux expansion at this point: {}",
                match r {
                    NotPuiseuxReason::Logarithmic =>
                        "the expansion contains a logarithm of a quantity vanishing at the \
                         point (log x, sqrt(x)*log x). That is a Puiseux-log / transseries \
                         term, not a Puiseux series, and this engine has no representation \
                         for one",
                    NotPuiseuxReason::EssentialSingularity =>
                        "a function head is applied to something with a pole at the point \
                         (exp(1/x), sin(1/x)); no truncation exponent bounds the remainder",
                    NotPuiseuxReason::UnboundedRamification =>
                        "the ramification index exceeds the largest this engine can verify \
                         an expansion at",
                    NotPuiseuxReason::UnsupportedForm =>
                        "the expression contains a shape the expander does not implement \
                         (a variable exponent, a piecewise, an unexpandable head)",
                    NotPuiseuxReason::IndeterminateCoefficient =>
                        "a coefficient is an indeterminate form (0/0, 0^-1, log 0) rather \
                         than a number",
                }
            ),
            PuiseuxError::Exhausted(_) => write!(
                f,
                "the Puiseux expansion did not reach the requested order within the work \
                 ceiling; refusing to return a shorter series under the requested O(.) label"
            ),
            PuiseuxError::Unverified(r) => write!(
                f,
                "a Puiseux expansion was computed and withheld because it could not be \
                 verified: {}",
                match r {
                    UnverifiedReason::DecayTooSlow =>
                        "the truncation residual does not decay at the claimed rate",
                    UnverifiedReason::NotEvaluable =>
                        "neither the function nor the series could be evaluated near the \
                         expansion point, so nothing was measured",
                    UnverifiedReason::PowerCheckDisagrees =>
                        "raising the series to the ramification index disagreed with the \
                         independently computed expansion of the same power",
                }
            ),
        }
    }
}

impl std::error::Error for PuiseuxError {}

impl crate::errors::AlkahestError for PuiseuxError {
    fn code(&self) -> &'static str {
        match self {
            PuiseuxError::Diff(_) => "E-SERIES-001",
            PuiseuxError::InvalidOrder => "E-SERIES-002",
            PuiseuxError::Exhausted(_) => "E-SERIES-003",
            PuiseuxError::NotPuiseux(_) => "E-SERIES-005",
            PuiseuxError::Unverified(_) => "E-SERIES-006",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            PuiseuxError::Diff(_) => {
                Some("ensure all functions are registered primitives with differentiation rules")
            }
            PuiseuxError::InvalidOrder => {
                Some("pass order >= 1 (exclusive truncation exponent in h)")
            }
            PuiseuxError::Exhausted(_) => Some(
                "ask for a lower order, raise the budget, or rewrite the expression so its \
                 expansion closes",
            ),
            PuiseuxError::NotPuiseux(_) => Some(
                "a logarithm of a vanishing quantity (`sqrt(x)*log x`) needs a \
                 Puiseux-log/transseries type this engine does not have, and an essential \
                 singularity has no truncation exponent at all: factor the log term out by \
                 hand, or expand about a nearby regular point. A ramification index past \
                 the verifiable range is the third reading — a Python `Fraction` exponent \
                 reaches the kernel through `f64` and is not the fraction you wrote; build \
                 it with `pow_expr(pool.rational(p, q))`",
            ),
            PuiseuxError::Unverified(_) => Some(
                "ask for a lower order, or expand about a point where the function is real \
                 and its heads have numeric kernels; an unverified expansion is withheld \
                 rather than returned",
            ),
        }
    }
}

impl From<SeriesError> for PuiseuxError {
    fn from(e: SeriesError) -> Self {
        match e {
            SeriesError::Diff(d) => PuiseuxError::Diff(d.to_string()),
            SeriesError::InvalidOrder => PuiseuxError::Exhausted(crate::budget::check().err()),
        }
    }
}

// ---------------------------------------------------------------------------
// Public result type
// ---------------------------------------------------------------------------

/// A verified truncated Puiseux expansion.
///
/// Every instance of this type has passed `verify`. Constructing one outside
/// this module is not possible, which is what makes "you are holding a
/// `PuiseuxExpansion`" mean "this expansion was checked".
#[derive(Clone, Debug)]
pub struct PuiseuxExpansion {
    terms: Vec<(Rational, ExprId)>,
    remainder: Rational,
    ramification: u64,
    h_expr: ExprId,
    expr: ExprId,
    evidence: Evidence,
}

/// What was actually checked before the expansion was released.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Evidence {
    /// Rungs of the prefix ladder that were run: one per truncation of the
    /// series, per parameter assignment. See `verify`.
    pub rungs: usize,
    /// Rungs that produced an actual fitted decay exponent, as opposed to a
    /// residual that sat at the noise floor throughout. **At least one**, or
    /// the expansion is not returned: a sum that matches to machine precision
    /// is not a measurement of a rate.
    pub conclusive_rungs: usize,
    /// The tightest `observed − claimed` decay margin over all fitted rungs.
    /// Negative but within `decay_tolerance` is a pass; more negative than
    /// that is [`UnverifiedReason::DecayTooSlow`] and the expansion is
    /// withheld, so this is always above `−tolerance`.
    pub worst_margin: f64,
    /// Whether the exact `Sᵉ` versus `f^e` comparison was applicable and
    /// agreed. `false` means it did not apply, never that it disagreed —
    /// a disagreement is [`UnverifiedReason::PowerCheckDisagrees`].
    pub power_check_passed: bool,
}

impl PuiseuxExpansion {
    /// The expansion as one expression: `Σ cₖ h^{eₖ} + O(h^order)`.
    pub fn expr(&self) -> ExprId {
        self.expr
    }

    /// `h`, i.e. `var − point` (or bare `var` when `point` is the integer `0`).
    pub fn increment(&self) -> ExprId {
        self.h_expr
    }

    /// The ramification index `e`: every exponent has denominator dividing `e`.
    /// `1` means the expansion is an ordinary Laurent series.
    ///
    /// Computed from the exponents that came out, never assumed from the shape
    /// of the input.
    pub fn ramification(&self) -> u64 {
        self.ramification
    }

    /// `(exponent, coefficient)` pairs, strictly ascending in exponent, with
    /// no structurally-zero coefficients.
    pub fn terms(&self) -> &[(Rational, ExprId)] {
        &self.terms
    }

    /// Leading exponent, or `None` when the expansion is zero to the requested
    /// order (in which case nothing is claimed beyond `O(h^order)`).
    pub fn valuation(&self) -> Option<&Rational> {
        self.terms.first().map(|(e, _)| e)
    }

    /// The exclusive truncation exponent: every omitted term has exponent
    /// `>=` this, so the remainder really is `O(h^{this})`.
    pub fn remainder_order(&self) -> &Rational {
        &self.remainder
    }

    /// What was checked before this expansion was released.
    pub fn evidence(&self) -> Evidence {
        self.evidence
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Puiseux expansion of `expr` in `var` about `point`, to exclusive truncation
/// exponent `order`.
///
/// Let `h = var − point`. Every term with exponent `< order` is present; the
/// remainder is `O(h^order)`. That is the same reading `order` has in
/// [`crate::calculus::series::series`] for an analytic expansion, and a
/// *sharper* one for a polar expansion: `series(1/sin x, x, 0, 4)` labels its
/// result `O(x¹)` (its Laurent convention), while this returns
/// `x⁻¹ + x/6 + 7x³/360 + O(x⁴)`.
///
/// # Guarantee
///
/// A returned [`PuiseuxExpansion`] has been checked — numerically at several
/// points approaching `point`, and, where the shape allows it, exactly. An
/// expansion that could not be checked is an `Err`, never a value. See
/// `verify`.
///
/// # Examples
///
/// ```
/// use alkahest_cas::calculus::puiseux::puiseux_series;
/// use alkahest_cas::kernel::{Domain, ExprPool};
///
/// let p = ExprPool::new();
/// let x = p.symbol("x", Domain::Real);
/// let s = p.func("sin", vec![x]);
/// let e = p.func("sqrt", vec![s]);
///
/// // √(sin x) = x^{1/2} − x^{5/2}/12 + x^{9/2}/1440 + O(x⁵)
/// let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
/// assert_eq!(px.ramification(), 2);
/// assert_eq!(px.terms().len(), 3);
/// assert_eq!(*px.valuation().unwrap(), rug::Rational::from((1, 2)));
/// ```
pub fn puiseux_series(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<PuiseuxExpansion, PuiseuxError> {
    if order == 0 {
        return Err(PuiseuxError::InvalidOrder);
    }
    let ceiling = pool.len().saturating_add(MAX_SERIES_POOL_GROWTH);
    let _coeff_ceiling = enter_coeff_ceiling(ceiling);
    let _pool_ceiling = enter_pool_ceiling(ceiling);

    let xi = pool.symbol("__pxi", Domain::Real);
    let mut map = HashMap::new();
    map.insert(var, pool.add(vec![point, xi]));
    let shifted = subs(expr, &map, pool);
    let h_expr = expansion_increment(pool, var, point);

    let prec = Rational::from(order);
    let mut s = expand(shifted, xi, &prec, pool, 0)?;
    if s.prec < prec {
        return Err(PuiseuxError::Exhausted(None));
    }
    s.truncate(&prec);

    let ramification = ramification_of(&s.terms);
    if ramification > MAX_RAMIFICATION {
        return Err(PuiseuxError::NotPuiseux(
            NotPuiseuxReason::UnboundedRamification,
        ));
    }

    let terms: Vec<(Rational, ExprId)> = s.terms.iter().map(|(e, c)| (e.clone(), *c)).collect();

    // ---- the load-bearing step -------------------------------------------
    let evidence = verify(expr, var, point, &terms, &prec, ramification, pool)?;

    let out = assemble(&terms, &prec, h_expr, pool);
    Ok(PuiseuxExpansion {
        terms,
        remainder: prec,
        ramification,
        h_expr,
        expr: out,
        evidence,
    })
}

// ---------------------------------------------------------------------------
// The truncated-series type used inside the expander
// ---------------------------------------------------------------------------

/// `Σ terms[e] · h^e`, with everything at exponent `>= prec` unknown.
///
/// The invariant that makes the whole module honest: **`prec` is never
/// overstated.** Every operation computes the truncation exponent it can
/// actually justify and takes the minimum, so a short leaf expansion degrades
/// the final `prec` rather than being papered over.
#[derive(Clone, Debug)]
struct PSeries {
    terms: BTreeMap<Rational, ExprId>,
    prec: Rational,
}

impl PSeries {
    fn empty(prec: Rational) -> Self {
        PSeries {
            terms: BTreeMap::new(),
            prec,
        }
    }

    fn constant(c: ExprId, prec: Rational, pool: &ExprPool) -> Self {
        let mut terms = BTreeMap::new();
        if !is_structural_zero(c, pool) {
            terms.insert(Rational::from(0), c);
        }
        PSeries { terms, prec }
    }

    /// Leading exponent, or — when nothing is known below `prec` — `prec`
    /// itself, which is a valid *lower bound* on the valuation and therefore
    /// safe everywhere a valuation is used to justify a truncation exponent.
    fn val(&self) -> Rational {
        self.terms
            .keys()
            .next()
            .cloned()
            .unwrap_or_else(|| self.prec.clone())
    }

    fn lead(&self) -> Option<(Rational, ExprId)> {
        self.terms.iter().next().map(|(e, c)| (e.clone(), *c))
    }

    fn truncate(&mut self, prec: &Rational) {
        if *prec < self.prec {
            self.prec = prec.clone();
        }
        let cut = self.prec.clone();
        self.terms.retain(|e, _| *e < cut);
    }
}

fn qi(n: i32) -> Rational {
    Rational::from(n)
}

fn qadd(a: &Rational, b: &Rational) -> Rational {
    Rational::from(a + b)
}

fn qsub(a: &Rational, b: &Rational) -> Rational {
    Rational::from(a - b)
}

fn qmul(a: &Rational, b: &Rational) -> Rational {
    Rational::from(a * b)
}

fn qmin(a: Rational, b: Rational) -> Rational {
    if a < b {
        a
    } else {
        b
    }
}

/// Smallest integer `>= r`.
fn qceil(r: &Rational) -> Integer {
    let (q, rem) = Integer::from(r.numer()).div_rem_floor(Integer::from(r.denom()));
    if rem == 0 {
        q
    } else {
        q + 1
    }
}

fn ramification_of(terms: &BTreeMap<Rational, ExprId>) -> u64 {
    let mut e: u64 = 1;
    for k in terms.keys() {
        let d = k.denom().to_u64().unwrap_or(u64::MAX);
        e = lcm_u64(e, d);
        if e >= u64::MAX / 2 {
            return u64::MAX;
        }
    }
    e
}

fn lcm_u64(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    let g = gcd_u64(a, b);
    (a / g).saturating_mul(b)
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

// ---------------------------------------------------------------------------
// Series arithmetic
// ---------------------------------------------------------------------------

fn simp(id: ExprId, pool: &ExprPool) -> ExprId {
    simplify(id, pool).value
}

fn insert_term(terms: &mut BTreeMap<Rational, ExprId>, e: Rational, c: ExprId, pool: &ExprPool) {
    if is_structural_zero(c, pool) {
        return;
    }
    match terms.entry(e) {
        std::collections::btree_map::Entry::Occupied(mut o) => {
            let merged = simp(pool.add(vec![*o.get(), c]), pool);
            if is_structural_zero(merged, pool) {
                o.remove();
            } else {
                o.insert(merged);
            }
        }
        std::collections::btree_map::Entry::Vacant(v) => {
            v.insert(c);
        }
    }
}

fn p_add(a: &PSeries, b: &PSeries, pool: &ExprPool) -> PSeries {
    let prec = qmin(a.prec.clone(), b.prec.clone());
    let mut terms = BTreeMap::new();
    for (e, c) in a.terms.iter().chain(b.terms.iter()) {
        if *e < prec {
            insert_term(&mut terms, e.clone(), *c, pool);
        }
    }
    PSeries { terms, prec }
}

/// `prec(ab) = min(prec_a + val_b, prec_b + val_a)` — the exact statement of
/// "the first term either side could be hiding, times the other's leading one".
fn p_mul(a: &PSeries, b: &PSeries, pool: &ExprPool) -> PSeries {
    let prec = qmin(qadd(&a.prec, &b.val()), qadd(&b.prec, &a.val()));
    let mut terms = BTreeMap::new();
    for (ea, ca) in &a.terms {
        for (eb, cb) in &b.terms {
            let e = qadd(ea, eb);
            if e < prec {
                let c = simp(pool.mul(vec![*ca, *cb]), pool);
                insert_term(&mut terms, e, c, pool);
            }
        }
    }
    PSeries { terms, prec }
}

/// Multiply by `k · h^shift` (an exact monomial: the truncation exponent moves
/// with it).
fn p_shift_scale(a: &PSeries, shift: &Rational, k: &Rational, pool: &ExprPool) -> PSeries {
    let kk = pool.rational(k.numer().clone(), k.denom().clone());
    let mut terms = BTreeMap::new();
    for (e, c) in &a.terms {
        let c2 = simp(pool.mul(vec![*c, kk]), pool);
        insert_term(&mut terms, qadd(e, shift), c2, pool);
    }
    PSeries {
        terms,
        prec: qadd(&a.prec, shift),
    }
}

fn binomial_rational(alpha: &Rational, k: u32) -> Rational {
    let mut acc = Rational::from(1);
    for i in 0..k {
        acc *= qsub(alpha, &Rational::from(i));
    }
    let mut f = Integer::from(1);
    for i in 2..=k {
        f *= i;
    }
    acc /= Rational::from(f);
    acc
}

/// `a^alpha` for a rational `alpha`, by `a = c₀h^v(1+u)` and the binomial
/// series in `u`.
///
/// The leading term must exist: `alpha` may be negative or fractional, and
/// `c₀^alpha` and `h^{v·alpha}` are both meaningless without it. When `a` is
/// zero to its own precision and `alpha` is a positive integer the answer is
/// still known (`O(h^{alpha·prec})`), and that case is taken first.
fn p_pow_rational(
    a: &PSeries,
    alpha: &Rational,
    prec: &Rational,
    pool: &ExprPool,
) -> Result<PSeries, PuiseuxError> {
    let Some((v, c0)) = a.lead() else {
        // Nothing known below `a.prec`.
        if *alpha.denom() == 1 && *alpha.numer() > 0 {
            let n = alpha.numer().to_u32().unwrap_or(1);
            return Ok(PSeries::empty(qmul(&a.prec, &Rational::from(n))));
        }
        return Err(PuiseuxError::Exhausted(None));
    };

    let lead_exp = qmul(&v, alpha);
    // Precision the `(1+u)^alpha` factor must reach for the product to reach `prec`.
    let target = qsub(prec, &lead_exp);
    if target <= qi(0) {
        return Ok(PSeries::empty(lead_exp));
    }

    // u = a/(c₀h^v) − 1, built by dropping the leading term outright rather
    // than trusting `simplify` to reduce `c₀·c₀⁻¹` to `1`.
    let inv_c0 = simp(pool.pow(c0, pool.integer(-1_i32)), pool);
    let mut u = PSeries::empty(qsub(&a.prec, &v));
    for (e, c) in a.terms.iter().skip(1) {
        let c2 = simp(pool.mul(vec![*c, inv_c0]), pool);
        insert_term(&mut u.terms, qsub(e, &v), c2, pool);
    }

    let wprec = qmin(target, u.prec.clone());
    if wprec <= qi(0) {
        return Ok(PSeries::empty(lead_exp));
    }
    let vu = u.val();
    if vu <= qi(0) {
        // Cannot happen: every exponent in `u` is strictly above `v`. Refusing
        // rather than asserting keeps a future change from turning this into a
        // divide-by-zero in the term count.
        return Err(PuiseuxError::NotPuiseux(NotPuiseuxReason::UnsupportedForm));
    }
    let kmax_i = qceil(&Rational::from(&wprec / &vu));
    let kmax = kmax_i.to_u32().unwrap_or(u32::MAX);
    if kmax > MAX_SERIES_TERMS {
        return Err(PuiseuxError::Exhausted(None));
    }

    let one = pool.integer(1_i32);
    let mut w = PSeries::constant(one, wprec.clone(), pool);
    let mut upow = PSeries::constant(one, wprec.clone(), pool);
    for k in 1..kmax.max(1) {
        check_work(pool)?;
        upow = p_mul(&upow, &u, pool);
        upow.truncate(&wprec);
        if upow.terms.is_empty() {
            break;
        }
        let bk = binomial_rational(alpha, k);
        if bk != 0 {
            let scaled = p_shift_scale(&upow, &qi(0), &bk, pool);
            w = p_add(&w, &scaled, pool);
        }
    }
    w.truncate(&wprec);

    // c₀^alpha. `simplify` folds the easy cases (`1^α = 1`, `(x²)^{1/2}` is not
    // one of them); anything it leaves symbolic is fine — the verifier decides.
    let alpha_e = pool.rational(alpha.numer().clone(), alpha.denom().clone());
    let scale = simp(pool.pow(c0, alpha_e), pool);

    let mut terms = BTreeMap::new();
    for (e, c) in &w.terms {
        let c2 = simp(pool.mul(vec![*c, scale]), pool);
        insert_term(&mut terms, qadd(e, &lead_exp), c2, pool);
    }
    Ok(PSeries {
        terms,
        prec: qadd(&lead_exp, &w.prec),
    })
}

// ---------------------------------------------------------------------------
// The expander
// ---------------------------------------------------------------------------

fn check_work(pool: &ExprPool) -> Result<(), PuiseuxError> {
    if let Err(b) = crate::budget::check() {
        return Err(PuiseuxError::Exhausted(Some(b)));
    }
    if pool.len() > pool_ceiling() {
        return Err(PuiseuxError::Exhausted(None));
    }
    Ok(())
}

thread_local! {
    /// Absolute `pool.len()` ceiling for the Puiseux expander, mirroring
    /// [`crate::calculus::series::MAX_SERIES_POOL_GROWTH`]'s role in the
    /// integer engine. Without it the binomial/composition loops are bounded
    /// only by the term counts, and it is *size* rather than iteration count
    /// that runs away on a nested radical.
    static POOL_CEILING: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

fn pool_ceiling() -> usize {
    POOL_CEILING.with(|c| c.get())
}

/// RAII installer for [`POOL_CEILING`]; restores the previous value on drop,
/// including on panic-unwind.
struct PoolCeiling(usize);

impl Drop for PoolCeiling {
    fn drop(&mut self) {
        POOL_CEILING.with(|c| c.set(self.0));
    }
}

fn enter_pool_ceiling(ceiling: usize) -> PoolCeiling {
    POOL_CEILING.with(|c| {
        let prev = c.get();
        c.set(ceiling);
        PoolCeiling(prev)
    })
}

fn depends_on(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| depends_on(x, var, pool)),
        ExprData::Pow { base, exp } => depends_on(base, var, pool) || depends_on(exp, var, pool),
        ExprData::Func { args, .. } => args.iter().any(|&a| depends_on(a, var, pool)),
        ExprData::BigO(a) => depends_on(a, var, pool),
        _ => false,
    }
}

/// A numeric exponent, if this node is one.
fn as_rational(id: ExprId, pool: &ExprPool) -> Option<Rational> {
    pool.with(id, |d| match d {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        // A binary float **is** a rational, exactly, and reading it as one is
        // the difference between `x**1.5` expanding and refusing: Python's
        // `Expr.__pow__` coerces a `Fraction` exponent through `f64`, so
        // `x ** Fraction(3, 2)` arrives here as a `Float`. The conversion is
        // exact in both directions — no rounding is introduced and nothing is
        // assumed about what the user "meant".
        //
        // The safety of that is entirely on the exactness: `Fraction(1, 3)`
        // arrives as `6004799503160661/18014398509481984`, whose denominator
        // makes the ramification astronomical, and the expansion is refused as
        // `UnboundedRamification` rather than quietly rounded to `x^{1/3}` —
        // which it is not.
        ExprData::Float(f) => f.inner.to_rational(),
        _ => None,
    })
}

/// Take the node's expansion from the **existing** integer-exponent engine.
///
/// `Ok(None)` means "that engine cannot expand this node here" — an
/// indeterminate coefficient, i.e. exactly the case the fractional route exists
/// for. It is not an error, because the caller has somewhere else to go.
///
/// The precision claimed is derived from the number of coefficients that came
/// *back*, not from the order that was asked for, so a coefficient loop that
/// stopped early degrades the truncation exponent instead of overstating it.
fn leaf_expand(
    node: ExprId,
    xi: ExprId,
    target: &Rational,
    pool: &ExprPool,
) -> Result<Option<PSeries>, PuiseuxError> {
    let zero = pool.integer(0_i32);
    let mut order = qceil(target).to_u32().unwrap_or(MAX_LEAF_ORDER).max(1);
    for _ in 0..MAX_LEAF_RETRIES {
        check_work(pool)?;
        if order > MAX_LEAF_ORDER {
            return Err(PuiseuxError::Exhausted(None));
        }
        let le = match local_expansion(node, xi, zero, order, pool) {
            Ok(le) => le,
            // A differentiation failure is the integer engine saying it has no
            // route to this node — a non-integer power, an unregistered head.
            // That is the signal to try the fractional route, not a reason to
            // give up: propagating it would report `x^{3/2}` as "cannot
            // differentiate" when the expansion is a single exact term.
            Err(SeriesError::Diff(_)) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        if first_indeterminate(&le.coeffs, pool).is_some() {
            return Ok(None);
        }
        let achieved = Rational::from(le.valuation) + Rational::from(le.coeffs.len() as u32);
        let mut terms = BTreeMap::new();
        for (k, c) in le.coeffs.iter().enumerate() {
            let e = Rational::from(le.valuation) + Rational::from(k as u32);
            insert_term(&mut terms, e, *c, pool);
        }
        if achieved >= *target || le.coeffs.is_empty() {
            return Ok(Some(PSeries {
                terms,
                prec: achieved,
            }));
        }
        let deficit = qceil(&qsub(target, &achieved)).to_u32().unwrap_or(1).max(1);
        order = order.saturating_add(deficit);
    }
    Err(PuiseuxError::Exhausted(None))
}

fn expand(
    e: ExprId,
    xi: ExprId,
    prec: &Rational,
    pool: &ExprPool,
    depth: u32,
) -> Result<PSeries, PuiseuxError> {
    check_work(pool)?;
    if depth > MAX_DEPTH {
        return Err(PuiseuxError::Exhausted(None));
    }
    if !depends_on(e, xi, pool) {
        return Ok(PSeries::constant(simp(e, pool), prec.clone(), pool));
    }
    if e == xi {
        let mut terms = BTreeMap::new();
        terms.insert(qi(1), pool.integer(1_i32));
        return Ok(PSeries {
            terms,
            prec: prec.clone(),
        });
    }

    match pool.get(e) {
        ExprData::Add(args) => {
            let mut acc = PSeries::constant(pool.integer(0_i32), prec.clone(), pool);
            for a in args {
                let s = expand(a, xi, prec, pool, depth + 1)?;
                acc = p_add(&acc, &s, pool);
            }
            Ok(acc)
        }
        ExprData::Mul(args) => expand_mul(&args, xi, prec, pool, depth),
        ExprData::Pow { base, exp } => expand_pow(e, base, exp, xi, prec, pool, depth),
        ExprData::Func { ref name, ref args } => expand_func(e, name, args, xi, prec, pool, depth),
        ExprData::Symbol { .. } => {
            // A symbol that depends on `xi` and is not `xi`: impossible.
            Err(PuiseuxError::NotPuiseux(NotPuiseuxReason::UnsupportedForm))
        }
        _ => Err(PuiseuxError::NotPuiseux(NotPuiseuxReason::UnsupportedForm)),
    }
}

/// Expand a product, raising each factor's target by what the *other* factors'
/// valuations eat.
///
/// The pass structure is what makes the truncation exponent right: a factor
/// with valuation `2` consumes two orders of every other factor, and asking for
/// `prec` from each independently would silently return a series two orders
/// short of its label.
fn expand_mul(
    args: &[ExprId],
    xi: ExprId,
    prec: &Rational,
    pool: &ExprPool,
    depth: u32,
) -> Result<PSeries, PuiseuxError> {
    let mut parts: Vec<PSeries> = Vec::with_capacity(args.len());
    for &a in args {
        parts.push(expand(a, xi, prec, pool, depth + 1)?);
    }
    for _ in 0..3 {
        let vals: Vec<Rational> = parts.iter().map(|p| p.val()).collect();
        let mut total = qi(0);
        for v in &vals {
            total = qadd(&total, v);
        }
        let mut changed = false;
        for i in 0..parts.len() {
            let others = qsub(&total, &vals[i]);
            let needed = qsub(prec, &others);
            if needed > parts[i].prec {
                parts[i] = expand(args[i], xi, &needed, pool, depth + 1)?;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut acc = PSeries::constant(pool.integer(1_i32), prec.clone(), pool);
    // A product of exact monomials must not be capped by the requested `prec`
    // before the factors are in: start from the first factor instead.
    let mut iter = parts.into_iter();
    if let Some(first) = iter.next() {
        acc = first;
    }
    for p in iter {
        check_work(pool)?;
        acc = p_mul(&acc, &p, pool);
    }
    Ok(acc)
}

fn expand_pow(
    node: ExprId,
    base: ExprId,
    exp: ExprId,
    xi: ExprId,
    prec: &Rational,
    pool: &ExprPool,
    depth: u32,
) -> Result<PSeries, PuiseuxError> {
    if depends_on(exp, xi, pool) {
        // `x^x`, `2^x`: only the integer engine has a route, and only when the
        // base is constant.
        return match leaf_expand(node, xi, prec, pool)? {
            Some(s) => Ok(s),
            None => Err(PuiseuxError::NotPuiseux(NotPuiseuxReason::UnsupportedForm)),
        };
    }
    let Some(alpha) = as_rational(exp, pool) else {
        return match leaf_expand(node, xi, prec, pool)? {
            Some(s) => Ok(s),
            None => Err(PuiseuxError::NotPuiseux(NotPuiseuxReason::UnsupportedForm)),
        };
    };
    // The integer engine first: `√(1+x)`, `(1+x)^-1`, `x^3` all come back from
    // it exactly as `series` would return them.
    if let Some(s) = leaf_expand(node, xi, prec, pool)? {
        return Ok(s);
    }
    expand_rational_power(base, &alpha, xi, prec, pool, depth)
}

fn expand_rational_power(
    base: ExprId,
    alpha: &Rational,
    xi: ExprId,
    prec: &Rational,
    pool: &ExprPool,
    depth: u32,
) -> Result<PSeries, PuiseuxError> {
    let mut b = expand(base, xi, prec, pool, depth + 1)?;
    // `f^α = c₀^α h^{vα}(1+u)^α` needs `u` to precision `prec − vα`, i.e. `f`
    // to precision `prec + v(1 − α)`.
    let v = b.val();
    let needed = qadd(prec, &qmul(&v, &qsub(&qi(1), alpha)));
    if needed > b.prec {
        b = expand(base, xi, &needed, pool, depth + 1)?;
    }
    p_pow_rational(&b, alpha, prec, pool)
}

fn expand_func(
    node: ExprId,
    name: &str,
    args: &[ExprId],
    xi: ExprId,
    prec: &Rational,
    pool: &ExprPool,
    depth: u32,
) -> Result<PSeries, PuiseuxError> {
    // The integer engine first, so every analytic head keeps the coefficients
    // `series` would have produced.
    if let Some(s) = leaf_expand(node, xi, prec, pool)? {
        return Ok(s);
    }
    if args.len() != 1 {
        return Err(PuiseuxError::NotPuiseux(NotPuiseuxReason::UnsupportedForm));
    }
    if name == "sqrt" {
        return expand_rational_power(args[0], &Rational::from((1, 2)), xi, prec, pool, depth);
    }
    if name == "cbrt" {
        return expand_rational_power(args[0], &Rational::from((1, 3)), xi, prec, pool, depth);
    }
    compose_head(name, args[0], xi, prec, pool, depth)
}

/// `g(f)` where `f` is a Puiseux series: Taylor of `g` at `f`'s constant term,
/// composed with `f − c` by Horner.
fn compose_head(
    name: &str,
    arg: ExprId,
    xi: ExprId,
    prec: &Rational,
    pool: &ExprPool,
    depth: u32,
) -> Result<PSeries, PuiseuxError> {
    let inner = expand(arg, xi, prec, pool, depth + 1)?;

    // A pole inside a function head is an essential singularity, not a Puiseux
    // series: `exp(1/x)` is `O(h^{-N})` for no `N` at all.
    if inner.terms.keys().next().is_some_and(|e| *e < qi(0)) {
        return Err(PuiseuxError::NotPuiseux(
            NotPuiseuxReason::EssentialSingularity,
        ));
    }

    let zero = pool.integer(0_i32);
    let c = inner.terms.get(&qi(0)).copied().unwrap_or(zero);
    let mut rest = inner.clone();
    rest.terms.remove(&qi(0));

    if rest.terms.is_empty() {
        // `g(c) + O(rest)`; `rest` is `O(h^{inner.prec})` and `g` is Lipschitz
        // near `c` wherever it is differentiable, so that is the honest
        // truncation exponent.
        let val = simp(pool.func(name, vec![c]), pool);
        let p = qmin(prec.clone(), inner.prec.clone());
        return Ok(PSeries::constant(val, p, pool));
    }
    let vr = rest.val();
    if vr <= qi(0) {
        return Err(PuiseuxError::NotPuiseux(NotPuiseuxReason::UnsupportedForm));
    }

    // Taylor coefficients of `g(c + s)` about `s = 0`, from the integer engine.
    let s = pool.symbol("__pcomp", Domain::Real);
    let shifted_arg = simp(pool.add(vec![c, s]), pool);
    let g = pool.func(name, vec![shifted_arg]);

    let kmax_r = Rational::from(&qmin(prec.clone(), rest.prec.clone()) / &vr);
    let kmax = qceil(&kmax_r).to_u32().unwrap_or(u32::MAX).max(1);
    if kmax > MAX_SERIES_TERMS {
        return Err(PuiseuxError::Exhausted(None));
    }

    let le = local_expansion(g, s, zero, kmax, pool)?;
    if let Some(idx) = first_indeterminate(&le.coeffs, pool) {
        // `log(0 + s)` lands here, and so does every other head singular at the
        // constant term. Naming the logarithmic case separately matters: it is
        // the one a caller might otherwise expect to work.
        let reason = if name == "log" && idx == 0 {
            NotPuiseuxReason::Logarithmic
        } else {
            NotPuiseuxReason::IndeterminateCoefficient
        };
        return Err(PuiseuxError::NotPuiseux(reason));
    }
    if le.coeffs.is_empty() {
        return Err(PuiseuxError::Exhausted(None));
    }

    // Horner in `rest`: `((t_{n-1}·r + t_{n-2})·r + …)·r^{valuation}`.
    let head_prec = qmul(&vr, &Rational::from(le.coeffs.len() as u32));
    let mut acc = PSeries::empty(head_prec);
    for &t in le.coeffs.iter().rev() {
        check_work(pool)?;
        acc = p_mul(&acc, &rest, pool);
        acc = p_add(&acc, &PSeries::constant(t, acc.prec.clone(), pool), pool);
    }
    if le.valuation != 0 {
        let shift = Rational::from(le.valuation);
        let pw = p_pow_rational(&rest, &shift, prec, pool)?;
        acc = p_mul(&acc, &pw, pool);
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

fn pow_term(h: ExprId, e: &Rational, pool: &ExprPool) -> ExprId {
    if *e == 0 {
        pool.integer(1_i32)
    } else if *e == 1 {
        h
    } else if *e.denom() == 1 {
        pool.pow(h, pool.integer(e.numer().clone()))
    } else {
        pool.pow(h, pool.rational(e.numer().clone(), e.denom().clone()))
    }
}

fn assemble(terms: &[(Rational, ExprId)], order: &Rational, h: ExprId, pool: &ExprPool) -> ExprId {
    let mut out = Vec::with_capacity(terms.len() + 1);
    for (e, c) in terms {
        out.push(pool.mul(vec![*c, pow_term(h, e, pool)]));
    }
    out.push(pool.big_o(pow_term(h, order, pool)));
    pool.add(out)
}

// ---------------------------------------------------------------------------
// Verification — nothing leaves this module unchecked
// ---------------------------------------------------------------------------

/// Predicted residual magnitudes the sample points are chosen to produce.
///
/// The sample `h` are derived *from* the claimed decay exponent rather than
/// fixed: `h = target^{1/γ}`. That is what keeps the measurement out of the
/// `f64` noise floor whatever `γ` is — the residual comes out near `10⁻³` at
/// the widest point and near `10⁻¹¹` at the narrowest whether the exponent
/// being checked is `1/2` or `9`. A fixed grid would put every low-order check
/// at a residual of order one and every high-order one under the noise.
const RESIDUAL_TARGETS: [f64; 6] = [1e-3, 1e-5, 1e-7, 1e-9, 1e-10, 1e-11];

/// Fallback grid for a claimed exponent too small (or negative) for
/// [`RESIDUAL_TARGETS`] to invert — the leading-term check of a polar
/// expansion, where the "residual" is the function itself and grows as `h → 0`.
const FALLBACK_HS: [f64; 5] = [0.25, 0.0625, 0.015625, 0.00390625, 0.0009765625];

/// Largest `h` any sample may use, so "approaching the expansion point" means
/// it even when the claimed exponent is large.
const MAX_SAMPLE_H: f64 = 0.25;

/// Relative floor below which a residual is indistinguishable from `f64`
/// rounding of the terms that produced it.
const NOISE_REL: f64 = 1e-11;

/// How far above the floor a residual must sit to be *fitted* rather than
/// merely counted as resolved. A residual at 3× the floor carries 30% noise,
/// which at these lever arms is worth several tenths of an exponent; at 100×
/// it carries 1%, which is worth a few hundredths.
const FIT_MARGIN: f64 = 100.0;

/// Generic values substituted for free parameters. Deliberately irrational:
/// a parameter value that happened to cancel a coefficient would check the
/// expansion on a degenerate slice and report a pass for the wrong reason.
const PARAM_SAMPLES: [[f64; 3]; 2] = [
    [0.6180339887498949, 1.3195079107728942, 2.23606797749979],
    [1.7320508075688772, 0.4142135623730951, 2.6457513110645907],
];

/// Most free parameters a verified expansion may carry.
const MAX_FREE_PARAMS: usize = 3;

/// Check a computed expansion, and refuse it if the check does not conclude.
///
/// # The prefix ladder
///
/// A single check of the whole series against `O(h^order)` is a weak test, and
/// for a good series it is a *vacuous* one: when the first omitted term sits
/// well above `order` — `√(sin x)` truncated at `5` next has an `x^{13/2}` term
/// — the residual drops under the `f64` noise floor before it can be measured,
/// and "no measurable residual" is not evidence about a rate.
///
/// So every **prefix** is checked instead. For each `k`, the truncation
/// `Σ_{i<k} cᵢ h^{eᵢ}` is compared against `f`, and its residual must decay
/// like `h^{e_k}` — the exponent of the *first omitted term*, which is a rate
/// the measurement can actually resolve because the term is right there. The
/// ladder runs from `k = 0`, whose residual is `f` itself and whose claimed
/// rate is the **valuation** (so the valuation is checked, not assumed),
/// through `k = n`, which is the `O(h^order)` claim.
///
/// Each rung isolates one coefficient: `c_{k}` is wrong if and only if rung
/// `k+1` decays at `e_k` instead of `e_{k+1}`, low by at least `1/e`. That is
/// the reason [`MAX_RAMIFICATION`] exists — past it `1/e` is smaller than the
/// systematic error of the fit, the rungs stop being able to fail, and a check
/// that cannot fail is worse than no check because it reads as evidence.
///
/// # Reading a rung
///
/// Residuals are measured at six `h` approaching the point, chosen from the
/// claimed exponent (see [`RESIDUAL_TARGETS`]). Then:
///
/// * a residual under the noise floor at *every* sample is a **pass** — the
///   truncation reproduces `f` to machine precision, which is stronger than any
///   rate, not weaker;
/// * a sample under the floor followed by a wider one over it means the
///   residual **grows** as `h → 0`, which no positive claimed exponent
///   survives: refused;
/// * otherwise the decay exponent is fitted on the narrowest pair that is at
///   least [`FIT_MARGIN`] above the floor, where the next-order contamination
///   is smallest, and must come within `decay_tolerance` of the claim.
///
/// Two rules make the ladder's silence mean something. At least one rung must
/// produce a fitted exponent - a verification in which every rung sat at the
/// noise floor has measured a sum, not a rate. And the **last** rung, the only
/// one that sees every returned coefficient, must not come back
/// `Unevaluable`: a coefficient the interpreter cannot value silences every
/// rung at or above it, and the rungs below would otherwise add up to a pass
/// with the tail unchecked. Either way the result is
/// [`UnverifiedReason::NotEvaluable`] and the expansion is withheld.
///
/// Only `h > 0` is sampled: a fractional power has no real value on the other
/// side, so a two-sided check would fail every ramified expansion by
/// construction. A function that is not real just above the point is refused,
/// never verified against `NaN`.
///
/// # Sensitivity, stated plainly
///
/// A coefficient wrong by `δ` shows up as a residual `δ·h^β` at the widest
/// sample, where `h^β ≥ 10⁻³`; against a floor near `10⁻¹¹` that resolves for
/// `δ ≳ 10⁻⁸`. Coefficients here are exact rationals produced by symbolic
/// arithmetic, so a defect makes them wrong by an `O(1)` amount, not by `10⁻⁹`.
/// This check is a screen, and a sharp one, but it is not a proof — which is
/// why the exact route below runs whenever it applies.
///
/// # Exact, where the shape allows it
///
/// With ramification `e`, `Sᵉ` has integer exponents, so it can be compared
/// against the expansion of `f^e` computed by the *independent* integer-exponent
/// engine. Where `f^e` is something that engine can expand — `√x` ⇒ `x`,
/// `√(sin x)` ⇒ `sin x` — this is exact, with no tolerance in it at all. Where
/// it is not, the numeric ladder stands alone. Where it applies and
/// *disagrees*, the expansion is refused.
///
/// # On failure
///
/// The expansion is discarded and [`PuiseuxError::Unverified`] returned. It is
/// never returned with a flag or a weakened `O(·)`: a caller cannot tell a
/// checked expansion from an unchecked one once it is in their hands, so the
/// unchecked one must not reach them.
fn verify(
    orig: ExprId,
    var: ExprId,
    point: ExprId,
    terms: &[(Rational, ExprId)],
    order: &Rational,
    ramification: u64,
    pool: &ExprPool,
) -> Result<Evidence, PuiseuxError> {
    let power_check_passed = exact_power_check(orig, var, point, terms, order, ramification, pool)?;
    let (rungs, conclusive, worst_margin) =
        numeric_ladder(orig, var, point, terms, order, ramification, pool)?;
    Ok(Evidence {
        rungs,
        conclusive_rungs: conclusive,
        worst_margin,
        power_check_passed,
    })
}

/// Free symbols of `expr` other than `var`. `false` means the walk met a node
/// whose symbol content it cannot account for, so the list is not trustworthy.
fn free_params(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) -> bool {
    match pool.get(expr) {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
        ExprData::Symbol { .. } => {
            if expr != var && !out.contains(&expr) {
                out.push(expr);
            }
            true
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().all(|&x| free_params(x, var, pool, out)),
        ExprData::Pow { base, exp } => {
            free_params(base, var, pool, out) && free_params(exp, var, pool, out)
        }
        ExprData::Func { args, .. } => args.iter().all(|&a| free_params(a, var, pool, out)),
        _ => false,
    }
}

/// What one rung of the ladder concluded.
enum Rung {
    /// Every residual sat at the noise floor: the truncation reproduces `f` to
    /// machine precision over the whole sample range.
    AtFloor,
    /// A decay exponent was fitted.
    Fitted(f64),
    /// Nothing could be evaluated (an unregistered head, a non-real branch).
    Unevaluable,
    /// The residual grows as `h → 0`, or the fit came out below the claim.
    Fails,
}

fn numeric_ladder(
    orig: ExprId,
    var: ExprId,
    point: ExprId,
    terms: &[(Rational, ExprId)],
    order: &Rational,
    ramification: u64,
    pool: &ExprPool,
) -> Result<(usize, usize, f64), PuiseuxError> {
    let empty = HashMap::new();
    let Some(a) = crate::jit::eval_interp(point, &empty, pool) else {
        return Err(PuiseuxError::Unverified(UnverifiedReason::NotEvaluable));
    };
    if !a.is_finite() {
        return Err(PuiseuxError::Unverified(UnverifiedReason::NotEvaluable));
    }

    let mut params = Vec::new();
    if !free_params(orig, var, pool, &mut params) || params.len() > MAX_FREE_PARAMS {
        return Err(PuiseuxError::Unverified(UnverifiedReason::NotEvaluable));
    }

    let assignments: Vec<Vec<f64>> = if params.is_empty() {
        vec![Vec::new()]
    } else {
        PARAM_SAMPLES
            .iter()
            .map(|row| (0..params.len()).map(|i| row[i]).collect())
            .collect()
    };

    let tol = decay_tolerance(ramification);
    let mut rungs = 0usize;
    let mut conclusive = 0usize;
    let mut worst_margin = f64::INFINITY;

    for assign in &assignments {
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        for (p, v) in params.iter().zip(assign.iter()) {
            env.insert(*p, *v);
        }
        for k in 0..=terms.len() {
            let claimed = if k < terms.len() {
                terms[k].0.to_f64()
            } else {
                order.to_f64()
            };
            if !claimed.is_finite() {
                return Err(PuiseuxError::Unverified(UnverifiedReason::NotEvaluable));
            }
            rungs += 1;
            let outcome = measure_rung(orig, var, a, &terms[..k], claimed, &env, pool);
            // The **last** rung is the only one that sees every returned
            // coefficient, so it is the only one whose silence is a hole: a
            // coefficient the interpreter cannot value makes every rung at or
            // above it `Unevaluable`, and without this the rungs *below* it
            // would still add up to a pass while the tail went unchecked.
            if k == terms.len() && matches!(outcome, Rung::Unevaluable) {
                return Err(PuiseuxError::Unverified(UnverifiedReason::NotEvaluable));
            }
            match outcome {
                Rung::Fails => {
                    return Err(PuiseuxError::Unverified(UnverifiedReason::DecayTooSlow))
                }
                Rung::Fitted(obs) => {
                    let margin = obs - claimed;
                    if margin < -tol {
                        return Err(PuiseuxError::Unverified(UnverifiedReason::DecayTooSlow));
                    }
                    worst_margin = worst_margin.min(margin);
                    conclusive += 1;
                }
                Rung::AtFloor | Rung::Unevaluable => {}
            }
        }
    }

    if conclusive == 0 {
        // Nothing was measured. A sum that matches to machine precision at
        // every sample is still no evidence about a *rate*, and the rate is
        // what the `O(·)` label asserts.
        return Err(PuiseuxError::Unverified(UnverifiedReason::NotEvaluable));
    }
    if !worst_margin.is_finite() {
        worst_margin = 0.0;
    }
    Ok((rungs, conclusive, worst_margin))
}

/// Sample offsets for a rung claiming decay `gamma`.
fn sample_hs(gamma: f64) -> Vec<f64> {
    if gamma >= 0.5 {
        let mut hs: Vec<f64> = RESIDUAL_TARGETS
            .iter()
            .map(|t| t.powf(1.0 / gamma).min(MAX_SAMPLE_H))
            .collect();
        hs.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON * a.abs().max(1.0));
        if hs.len() >= 2 {
            return hs;
        }
    }
    FALLBACK_HS.to_vec()
}

fn measure_rung(
    orig: ExprId,
    var: ExprId,
    a: f64,
    prefix: &[(Rational, ExprId)],
    claimed: f64,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
) -> Rung {
    let hs = sample_hs(claimed);
    // (h, residual, floor)
    let mut pts: Vec<(f64, f64, f64)> = Vec::with_capacity(hs.len());
    for &h in &hs {
        if !(h > 0.0 && h.is_finite()) {
            continue;
        }
        let mut e2 = env.clone();
        e2.insert(var, a + h);
        let Some(fv) = crate::jit::eval_interp(orig, &e2, pool) else {
            continue;
        };
        if !fv.is_finite() {
            continue;
        }
        let Some((sv, scale)) = eval_terms(prefix, h, env, pool) else {
            continue;
        };
        let r = (fv - sv).abs();
        if !r.is_finite() {
            continue;
        }
        let floor = NOISE_REL * scale.max(fv.abs()).max(1.0);
        pts.push((h, r, floor));
    }
    if pts.len() < 2 {
        return Rung::Unevaluable;
    }

    // Resolved samples must form a prefix: `h` descends, so a resolved sample
    // *after* an unresolved one means the residual grew on the way in, which no
    // positive claimed exponent survives.
    let resolved: Vec<bool> = pts.iter().map(|(_, r, f)| r > f).collect();
    let first_unresolved = resolved.iter().position(|b| !b).unwrap_or(pts.len());
    if claimed > 0.0 && resolved[first_unresolved..].iter().any(|b| *b) {
        return Rung::Fails;
    }
    let usable = if claimed > 0.0 {
        &pts[..first_unresolved]
    } else {
        &pts[..]
    };
    if usable.len() < 2 {
        return Rung::AtFloor;
    }

    // Fit on the narrowest pair that is comfortably above the floor: smallest
    // `h` means smallest contamination from the next term, and the margin keeps
    // rounding noise from eating the exponent.
    let good: Vec<&(f64, f64, f64)> = usable
        .iter()
        .filter(|(_, r, f)| *r > FIT_MARGIN * f)
        .collect();
    let (h0, r0, h1, r1) = if good.len() >= 2 {
        let n = good.len();
        (good[n - 2].0, good[n - 2].1, good[n - 1].0, good[n - 1].1)
    } else {
        let n = usable.len();
        (usable[0].0, usable[0].1, usable[n - 1].0, usable[n - 1].1)
    };
    if !(r0 > 0.0 && r1 > 0.0 && h0 > h1) {
        return Rung::AtFloor;
    }
    let observed = (r0 / r1).ln() / (h0 / h1).ln();
    if !observed.is_finite() {
        return Rung::AtFloor;
    }
    Rung::Fitted(observed)
}

/// How far below the claimed exponent an observed decay may sit.
///
/// A wrong coefficient moves the observed exponent down by at least `1/e`, so
/// the tolerance has to stay below `1/e` or the rung cannot tell the two apart.
/// It also has to stay above the systematic error of the fit — the next term
/// contaminates the residual, and what survives the ratio of a narrow pair is
/// the *change* in that contamination across the pair, a few hundredths.
/// `0.55/e` sits between the two, and [`MAX_RAMIFICATION`] is what keeps that
/// interval from closing.
fn decay_tolerance(ramification: u64) -> f64 {
    let e = ramification.max(1) as f64;
    (0.55 / e).min(0.4)
}

/// Evaluate `Σ cₖ h^{eₖ}` at `h`, returning the value and the largest term
/// magnitude — the scale that sets the floating-point noise floor, because a
/// polar expansion cancels large terms to reach a small one.
fn eval_terms(
    terms: &[(Rational, ExprId)],
    h: f64,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
) -> Option<(f64, f64)> {
    let mut sum = 0.0f64;
    let mut scale = 0.0f64;
    for (e, c) in terms {
        let cv = crate::jit::eval_interp(*c, env, pool)?;
        if !cv.is_finite() {
            return None;
        }
        let t = cv * h.powf(e.to_f64());
        if !t.is_finite() {
            return None;
        }
        sum += t;
        scale = scale.max(t.abs());
    }
    Some((sum, scale))
}

/// Push an exponent inside a product / power / `sqrt`, so `f^e` reaches the
/// integer engine as something it can expand.
///
/// Without this, `(x^{1/2}·sin x)²` arrives as an opaque square of a branch
/// point and the exact check simply never applies. Every rewrite here is an
/// identity on the domain where the original is real: `(∏aᵢ)^e = ∏aᵢ^e` and
/// `(a^p)^e = a^{pe}` for the integer `e` this is called with, and
/// `√a^e = a^{e/2}` wherever `√a` is defined at all. A rewrite that leaves a
/// fractional exponent behind simply makes the check inapplicable — the leaf
/// oracle declines it — which is the safe direction for a verifier.
fn raise_to(expr: ExprId, e: &Rational, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Mul(args) => pool.mul(args.iter().map(|&a| raise_to(a, e, pool)).collect()),
        ExprData::Pow { base, exp } => match as_rational(exp, pool) {
            Some(p) => {
                let ne = qmul(&p, e);
                raise_to_leaf(base, &ne, pool)
            }
            None => raise_to_leaf(expr, e, pool),
        },
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            let half = Rational::from((1, 2));
            raise_to_leaf(args[0], &qmul(e, &half), pool)
        }
        _ => raise_to_leaf(expr, e, pool),
    }
}

fn raise_to_leaf(base: ExprId, e: &Rational, pool: &ExprPool) -> ExprId {
    if *e == 1 {
        return base;
    }
    if *e.denom() == 1 {
        pool.pow(base, pool.integer(e.numer().clone()))
    } else {
        pool.pow(base, pool.rational(e.numer().clone(), e.denom().clone()))
    }
}

/// `Sᵉ` versus the independently computed expansion of `f^e`.
///
/// `Ok(true)`: the check ran and agreed. `Ok(false)`: it does not apply — `f^e`
/// is not something the integer engine can expand, which is the usual case for
/// a sum of ramified pieces. `Err`: it ran and **disagreed**.
fn exact_power_check(
    orig: ExprId,
    var: ExprId,
    point: ExprId,
    terms: &[(Rational, ExprId)],
    order: &Rational,
    ramification: u64,
    pool: &ExprPool,
) -> Result<bool, PuiseuxError> {
    if ramification <= 1 || ramification > MAX_RAMIFICATION {
        return Ok(false);
    }
    let e = ramification as i64;
    let Some(lead) = terms.first().map(|(x, _)| x.clone()) else {
        return Ok(false);
    };

    let mut s = PSeries::empty(order.clone());
    for (ex, c) in terms {
        s.terms.insert(ex.clone(), *c);
    }
    let e_q = Rational::from(e);
    let powered = match p_pow_rational(&s, &e_q, &qmul(order, &e_q), pool) {
        Ok(p) => p,
        Err(_) => return Ok(false),
    };

    let xi = pool.symbol("__pxi", Domain::Real);
    let mut map = HashMap::new();
    map.insert(var, pool.add(vec![point, xi]));
    let shifted = subs(orig, &map, pool);
    let fe = simp(raise_to(shifted, &e_q, pool), pool);

    let mut cmp_prec = qmin(powered.prec.clone(), qmul(order, &e_q));
    let Ok(Some(reference)) = leaf_expand(fe, xi, &cmp_prec, pool) else {
        return Ok(false);
    };
    cmp_prec = qmin(cmp_prec, reference.prec.clone());
    if cmp_prec <= qmul(&lead, &e_q) {
        return Ok(false);
    }

    let mut keys: Vec<Rational> = powered
        .terms
        .keys()
        .chain(reference.terms.keys())
        .filter(|k| **k < cmp_prec)
        .cloned()
        .collect();
    keys.sort();
    keys.dedup();
    let zero = pool.integer(0_i32);
    let empty = HashMap::new();
    for k in keys {
        let x = powered.terms.get(&k).copied().unwrap_or(zero);
        let y = reference.terms.get(&k).copied().unwrap_or(zero);
        let diff = simp(
            pool.add(vec![x, pool.mul(vec![pool.integer(-1_i32), y])]),
            pool,
        );
        if is_structural_zero(diff, pool) {
            continue;
        }
        // A symbolic non-zero difference may still *be* zero, so only a
        // numerically non-zero one is evidence of disagreement; anything the
        // interpreter cannot value makes the check inapplicable rather than
        // failed.
        match crate::jit::eval_interp(diff, &empty, pool) {
            Some(v) if v.abs() > 1e-9 => {
                return Err(PuiseuxError::Unverified(
                    UnverifiedReason::PowerCheckDisagrees,
                ));
            }
            Some(_) => {}
            None => return Ok(false),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculus::series::series;
    use crate::errors::AlkahestError;

    fn pool_with_x() -> (ExprPool, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        (p, x)
    }

    /// Coefficients as `f64`, paired with their exponents, for comparison
    /// against a value worked out by hand.
    fn coeffs(px: &PuiseuxExpansion, pool: &ExprPool) -> Vec<(Rational, f64)> {
        let env = HashMap::new();
        px.terms()
            .iter()
            .map(|(e, c)| {
                (
                    e.clone(),
                    crate::jit::eval_interp(*c, &env, pool).unwrap_or(f64::NAN),
                )
            })
            .collect()
    }

    fn assert_terms(got: &[(Rational, f64)], want: &[((i32, u32), f64)]) {
        assert_eq!(got.len(), want.len(), "term count: {got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(
                g.0,
                Rational::from((w.0 .0, w.0 .1)),
                "exponent: {got:?} vs {want:?}"
            );
            assert!(
                (g.1 - w.1).abs() <= 1e-12 * w.1.abs().max(1.0),
                "coefficient: {got:?} vs {want:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // The gap: expansions `series` refuses, and their known values
    // -----------------------------------------------------------------------

    /// `√x = x^{1/2}`, exactly. The valuation is a half-integer, which is the
    /// whole reason `series` cannot represent it.
    #[test]
    fn sqrt_x_is_a_single_half_integer_term() {
        let (p, x) = pool_with_x();
        let e = p.func("sqrt", vec![x]);
        let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_terms(&coeffs(&px, &p), &[((1, 2), 1.0)]);
        assert_eq!(*px.remainder_order(), Rational::from(5));
        // `series` still refuses it — this module is a sibling, not a rewrite.
        assert!(series(e, x, p.integer(0), 5, &p).is_err());
    }

    #[test]
    fn x_to_the_three_halves() {
        let (p, x) = pool_with_x();
        let e = p.pow(x, p.rational(3, 2));
        let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_terms(&coeffs(&px, &p), &[((3, 2), 1.0)]);
    }

    /// `√(sin x) = x^{1/2}(1 − x²/12 + x⁴/1440 − ⋯)`.
    ///
    /// Derived independently: `sin x = x(1 − x²/6 + x⁴/120)`, and
    /// `(1+u)^{1/2} = 1 + u/2 − u²/8` gives
    /// `1 − x²/12 + x⁴/240 − x⁴/288 = 1 − x²/12 + x⁴/1440`.
    /// SymPy's `series(sqrt(sin(x)), x, 0, 5)` agrees.
    #[test]
    fn sqrt_of_a_simple_zero() {
        let (p, x) = pool_with_x();
        let s = p.func("sin", vec![x]);
        let e = p.func("sqrt", vec![s]);
        let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_terms(
            &coeffs(&px, &p),
            &[((1, 2), 1.0), ((5, 2), -1.0 / 12.0), ((9, 2), 1.0 / 1440.0)],
        );
        // The exact `S² = sin x` route applied and agreed, on top of the ladder.
        assert!(px.evidence().power_check_passed);
    }

    /// `x^{1/2}·sin x = x^{3/2} − x^{7/2}/6 + ⋯`: a fractional power times an
    /// analytic factor, where the valuation is the *sum* of the two.
    #[test]
    fn half_power_times_an_analytic_factor() {
        let (p, x) = pool_with_x();
        let h = p.pow(x, p.rational(1, 2));
        let s = p.func("sin", vec![x]);
        let e = p.mul(vec![h, s]);
        let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_terms(&coeffs(&px, &p), &[((3, 2), 1.0), ((7, 2), -1.0 / 6.0)]);
    }

    /// `(sin x)^{1/3} = x^{1/3}(1 − x²/18 − x⁴/3240 − ⋯)`, ramification 3.
    ///
    /// By hand: `(1+u)^{1/3} = 1 + u/3 − u²/9` with `u = −x²/6 + x⁴/120`, so
    /// the `x⁴` coefficient is `1/360 − 1/324 = −1/3240`.
    #[test]
    fn cube_root_of_a_simple_zero_has_ramification_three() {
        let (p, x) = pool_with_x();
        let s = p.func("sin", vec![x]);
        let e = p.pow(s, p.rational(1, 3));
        let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
        assert_eq!(px.ramification(), 3);
        assert_terms(
            &coeffs(&px, &p),
            &[
                ((1, 3), 1.0),
                ((7, 3), -1.0 / 18.0),
                ((13, 3), -1.0 / 3240.0),
            ],
        );
    }

    /// `sin(√x) = x^{1/2} − x^{3/2}/6 + x^{5/2}/120 − x^{7/2}/5040`: an
    /// analytic head *composed with* a ramified argument.
    #[test]
    fn an_analytic_head_composed_with_a_ramified_argument() {
        let (p, x) = pool_with_x();
        let s = p.func("sqrt", vec![x]);
        let e = p.func("sin", vec![s]);
        let px = puiseux_series(e, x, p.integer(0), 4, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_terms(
            &coeffs(&px, &p),
            &[
                ((1, 2), 1.0),
                ((3, 2), -1.0 / 6.0),
                ((5, 2), 1.0 / 120.0),
                ((7, 2), -1.0 / 5040.0),
            ],
        );
    }

    /// `x^{-1/2}` — a *negative* fractional valuation, which is neither Taylor
    /// nor Laurent.
    #[test]
    fn a_negative_fractional_valuation() {
        let (p, x) = pool_with_x();
        let e = p.pow(x, p.rational(-1, 2));
        let px = puiseux_series(e, x, p.integer(0), 3, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_terms(&coeffs(&px, &p), &[((-1, 2), 1.0)]);
        assert_eq!(*px.valuation().unwrap(), Rational::from((-1, 2)));
    }

    /// Expansion about a point other than `0`: `√(x−1)` at `1` is `h^{1/2}`.
    #[test]
    fn expansion_about_a_nonzero_point() {
        let (p, x) = pool_with_x();
        let inner = p.add(vec![x, p.integer(-1)]);
        let e = p.func("sqrt", vec![inner]);
        let px = puiseux_series(e, x, p.integer(1), 3, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_terms(&coeffs(&px, &p), &[((1, 2), 1.0)]);
    }

    /// `√(t⁻² + t⁻¹)` — the expression `series`'s own documentation names as
    /// unfinishable — expands here in milliseconds.
    ///
    /// `series` forms coefficients by differentiating without re-simplifying,
    /// so a nested radical's derivatives grow by a constant factor each time
    /// and order 8 is out of reach; it refuses (`E-SERIES-004`/`003`), which is
    /// the honest answer for that method. Factoring the pole out first —
    /// `t⁻¹(1+t)^{1/2}` — turns the same expansion into a binomial series with
    /// one term per order, and the coefficients are the ordinary
    /// `C(1/2, k)`. Nothing is ramified about it: `e = 1`.
    #[test]
    fn the_radical_series_cannot_finish_expands_through_the_pole() {
        let p = ExprPool::new();
        let t = p.symbol("t", Domain::Real);
        let inner = p.add(vec![p.pow(t, p.integer(-2)), p.pow(t, p.integer(-1))]);
        let e = p.func("sqrt", vec![inner]);

        assert!(series(e, t, p.integer(0), 8, &p).is_err());

        let px = puiseux_series(e, t, p.integer(0), 8, &p).unwrap();
        assert_eq!(px.ramification(), 1);
        assert_terms(
            &coeffs(&px, &p),
            &[
                ((-1, 1), 1.0),
                ((0, 1), 0.5),
                ((1, 1), -0.125),
                ((2, 1), 1.0 / 16.0),
                ((3, 1), -5.0 / 128.0),
                ((4, 1), 7.0 / 256.0),
                ((5, 1), -21.0 / 1024.0),
                ((6, 1), 33.0 / 2048.0),
                ((7, 1), -429.0 / 32768.0),
            ],
        );
    }

    // -----------------------------------------------------------------------
    // Ramification is computed, never guessed from the shape
    // -----------------------------------------------------------------------

    /// A `√` over a zero of **even** order is not ramified at all:
    /// `√(x²+x³) = x·√(1+x)`, every exponent an integer, `e = 1`.
    ///
    /// Reading the ramification off the `1/2` in the input would say `2` here
    /// and be wrong; it comes from the exponents that actually came out.
    #[test]
    fn a_square_root_of_an_even_order_zero_is_unramified() {
        let (p, x) = pool_with_x();
        let a = p.add(vec![p.pow(x, p.integer(2)), p.pow(x, p.integer(3))]);
        let e = p.func("sqrt", vec![a]);
        let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
        assert_eq!(px.ramification(), 1);
        assert_terms(
            &coeffs(&px, &p),
            &[
                ((1, 1), 1.0),
                ((2, 1), 0.5),
                ((3, 1), -0.125),
                ((4, 1), 1.0 / 16.0),
            ],
        );
    }

    /// `(x²)^{1/4} = x^{1/2}`: the `4` in the input, the `2` in the answer.
    #[test]
    fn ramification_reduces_when_the_valuation_shares_a_factor() {
        let (p, x) = pool_with_x();
        let a = p.pow(x, p.integer(2));
        let e = p.pow(a, p.rational(1, 4));
        let px = puiseux_series(e, x, p.integer(0), 3, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_terms(&coeffs(&px, &p), &[((1, 2), 1.0)]);
    }

    /// `√(1 − cos x)` has a zero of order 2 under the radical, so it is
    /// unramified — and its coefficients are irrational (`1/√2`), which the
    /// symbolic route carries and the numeric ladder still checks.
    #[test]
    fn an_unramified_radical_with_irrational_coefficients() {
        let (p, x) = pool_with_x();
        let c = p.func("cos", vec![x]);
        let a = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), c])]);
        let e = p.func("sqrt", vec![a]);
        let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
        assert_eq!(px.ramification(), 1);
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        assert_terms(
            &coeffs(&px, &p),
            &[((1, 1), inv_sqrt2), ((3, 1), -inv_sqrt2 / 24.0)],
        );
    }

    // -----------------------------------------------------------------------
    // Refusals — the shapes that are not Puiseux
    // -----------------------------------------------------------------------

    fn refuses(expr: ExprId, x: ExprId, order: u32, p: &ExprPool) -> PuiseuxError {
        puiseux_series(expr, x, p.integer(0), order, p)
            .err()
            .unwrap_or_else(|| panic!("expected a refusal, got an expansion"))
    }

    /// `√x·log x` is the case that matters most: it *has* a leading behaviour
    /// (`x^{1/2}log x`), which is exactly what makes truncating it to `x^{1/2}`
    /// a wrong answer rather than a coarse one. A Puiseux–log representation is
    /// what it needs, and this engine does not have one, so it refuses.
    #[test]
    fn sqrt_times_log_is_refused_as_logarithmic() {
        let (p, x) = pool_with_x();
        let e = p.mul(vec![p.func("sqrt", vec![x]), p.func("log", vec![x])]);
        let err = refuses(e, x, 5, &p);
        assert_eq!(
            err,
            PuiseuxError::NotPuiseux(NotPuiseuxReason::Logarithmic),
            "{err}"
        );
        assert_eq!(err.code(), "E-SERIES-005");
    }

    #[test]
    fn a_bare_logarithm_is_refused() {
        let (p, x) = pool_with_x();
        let e = p.func("log", vec![x]);
        assert_eq!(
            refuses(e, x, 4, &p),
            PuiseuxError::NotPuiseux(NotPuiseuxReason::Logarithmic)
        );
    }

    #[test]
    fn an_essential_singularity_is_refused() {
        let (p, x) = pool_with_x();
        let inv = p.pow(x, p.integer(-1));
        for head in ["exp", "sin", "cos"] {
            let e = p.func(head, vec![inv]);
            assert_eq!(
                refuses(e, x, 4, &p),
                PuiseuxError::NotPuiseux(NotPuiseuxReason::EssentialSingularity),
                "{head}(1/x)"
            );
        }
    }

    /// `√x·e^{1/x}` — the essential singularity has to survive being wrapped in
    /// a product with something this engine *can* expand.
    #[test]
    fn an_essential_singularity_inside_a_product_is_still_refused() {
        let (p, x) = pool_with_x();
        let inv = p.pow(x, p.integer(-1));
        let e = p.mul(vec![p.func("sqrt", vec![x]), p.func("exp", vec![inv])]);
        assert_eq!(
            refuses(e, x, 4, &p),
            PuiseuxError::NotPuiseux(NotPuiseuxReason::EssentialSingularity)
        );
    }

    /// `x^{1/12}` is a perfectly good Puiseux series; it is refused because the
    /// verifier cannot tell it apart from a wrong one at that ramification, and
    /// an unverifiable expansion is not returned. The refusal names the reason.
    #[test]
    fn ramification_past_the_verifiable_range_is_refused() {
        let (p, x) = pool_with_x();
        let e = p.pow(x, p.rational(1, 12));
        let err = refuses(e, x, 2, &p);
        assert_eq!(
            err,
            PuiseuxError::NotPuiseux(NotPuiseuxReason::UnboundedRamification),
            "{err}"
        );
        // …and one just inside the range is not.
        let ok = p.pow(x, p.rational(1, 8));
        let px = puiseux_series(ok, x, p.integer(0), 2, &p).unwrap();
        assert_eq!(px.ramification(), 8);
        assert_eq!(px.ramification(), MAX_RAMIFICATION);
    }

    #[test]
    fn a_variable_exponent_is_refused() {
        let (p, x) = pool_with_x();
        let log_x = p.func("log", vec![x]);
        let x_to_x = p.func("exp", vec![p.mul(vec![x, log_x])]);
        let err = refuses(x_to_x, x, 3, &p);
        assert!(matches!(err, PuiseuxError::NotPuiseux(_)), "{err}");
    }

    #[test]
    fn order_zero_is_a_user_error() {
        let (p, x) = pool_with_x();
        let e = p.func("sqrt", vec![x]);
        let err = puiseux_series(e, x, p.integer(0), 0, &p).unwrap_err();
        assert_eq!(err, PuiseuxError::InvalidOrder);
        assert_eq!(err.code(), "E-SERIES-002");
    }

    /// A branch that is not real just above the point cannot be checked, and an
    /// expansion that cannot be checked is not returned. `√(−x)` at `0` is the
    /// shape: the coefficient is `(−1)^{1/2}`.
    #[test]
    fn a_non_real_branch_is_withheld_rather_than_returned() {
        let (p, x) = pool_with_x();
        let e = p.func("sqrt", vec![p.mul(vec![p.integer(-1), x])]);
        let err = puiseux_series(e, x, p.integer(0), 3, &p);
        assert!(
            matches!(err, Err(PuiseuxError::Unverified(_))),
            "{:?}",
            err.map(|s| s.ramification())
        );
    }

    // -----------------------------------------------------------------------
    // The verifier has to be able to fail
    // -----------------------------------------------------------------------

    /// The check that makes every other test mean something: a **deliberately
    /// corrupted** expansion must be rejected.
    ///
    /// The correct `√(sin x)` expansion is perturbed one coefficient at a time
    /// and handed straight to `verify`. Each perturbation moves the residual
    /// from `O(x^5)` to `O(x^{1/2})`…`O(x^{9/2})`, and the ladder has to see
    /// it. A verifier that passes these is decoration, and every "verified"
    /// claim in this module would be worthless.
    #[test]
    fn the_verifier_rejects_a_corrupted_expansion() {
        let (p, x) = pool_with_x();
        let s = p.func("sin", vec![x]);
        let e = p.func("sqrt", vec![s]);
        let order = Rational::from(5);
        let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
        let good: Vec<(Rational, ExprId)> = px.terms().to_vec();

        // Sanity: the untouched expansion passes.
        verify(e, x, p.integer(0), &good, &order, 2, &p).expect("the true expansion verifies");

        for i in 0..good.len() {
            for factor in [2_i32, -1, 3] {
                let mut bad = good.clone();
                bad[i].1 = simp(p.mul(vec![bad[i].1, p.integer(factor)]), &p);
                let r = verify(e, x, p.integer(0), &bad, &order, 2, &p);
                assert!(
                    matches!(r, Err(PuiseuxError::Unverified(_))),
                    "coefficient {i} scaled by {factor} was accepted"
                );
            }
        }

        // A wrong *exponent* is caught too: shifting the leading term by 1
        // makes the valuation wrong.
        let mut shifted = good.clone();
        shifted[0].0 = Rational::from((3, 2));
        let r = verify(e, x, p.integer(0), &shifted, &order, 2, &p);
        assert!(
            matches!(r, Err(PuiseuxError::Unverified(_))),
            "a shifted valuation was accepted"
        );

        // …and so is a *dropped* term, which is the failure mode that looks
        // most like an honest truncation.
        let dropped: Vec<(Rational, ExprId)> = good[..good.len() - 1].to_vec();
        let r = verify(e, x, p.integer(0), &dropped, &order, 2, &p);
        assert!(
            matches!(r, Err(PuiseuxError::Unverified(_))),
            "a dropped term was accepted under the full O(.) label"
        );
    }

    /// A coefficient the interpreter cannot put a number on silences the last
    /// rung of the ladder — the only rung that sees the whole series — and that
    /// has to be a refusal rather than a pass on the rungs below it.
    ///
    /// The rungs below a mute coefficient still measure fine, so without the
    /// last-rung rule this shape produced `conclusive_rungs >= 1` and sailed
    /// through with its tail unchecked. Constructed by hand because it is hard
    /// to reach from an expression: every head the expander accepts has a
    /// numeric kernel, which is why this is a guard rather than a hot path.
    #[test]
    fn a_coefficient_that_cannot_be_valued_is_a_refusal_not_a_pass() {
        let (p, x) = pool_with_x();
        let e = p.func("sqrt", vec![x]);
        let order = Rational::from(4);
        let px = puiseux_series(e, x, p.integer(0), 4, &p).unwrap();
        let good: Vec<(Rational, ExprId)> = px.terms().to_vec();
        verify(e, x, p.integer(0), &good, &order, 2, &p).expect("the true expansion verifies");

        // An unregistered head has no `numeric_f64` kernel, so `eval_interp`
        // declines the coefficient and every rung that includes it goes mute.
        let mute = p.func("__no_such_primitive", vec![p.integer(1_i32)]);
        let mut opaque = good.clone();
        opaque.push((Rational::from((7, 2)), mute));
        let r = verify(e, x, p.integer(0), &opaque, &order, 2, &p);
        assert!(
            matches!(
                r,
                Err(PuiseuxError::Unverified(UnverifiedReason::NotEvaluable))
            ),
            "an unvaluable trailing coefficient was accepted: {r:?}"
        );
    }

    /// Every returned expansion carries a *measured* rate, not just a computed
    /// one: at least one rung of the ladder must have resolved above the noise
    /// floor, and its margin against the claim is recorded.
    #[test]
    fn every_returned_expansion_has_a_measured_decay_rate() {
        let (p, x) = pool_with_x();
        let s = p.func("sin", vec![x]);
        let cases = [
            p.func("sqrt", vec![x]),
            p.func("sqrt", vec![s]),
            p.pow(s, p.rational(1, 3)),
            p.mul(vec![p.pow(x, p.rational(1, 2)), s]),
            p.func("sin", vec![p.func("sqrt", vec![x])]),
        ];
        for e in cases {
            let px = puiseux_series(e, x, p.integer(0), 5, &p).unwrap();
            let ev = px.evidence();
            assert!(ev.conclusive_rungs >= 1, "{ev:?} for {}", p.display(e));
            assert!(
                ev.worst_margin >= -decay_tolerance(px.ramification()),
                "{ev:?} for {}",
                p.display(e)
            );
        }
    }

    // -----------------------------------------------------------------------
    // Integer-exponent expansions must come back unchanged
    // -----------------------------------------------------------------------

    /// For an **analytic** expansion the two engines agree node for node, not
    /// just numerically: `puiseux_series` routes through `local_expansion` for
    /// every sub-part the integer engine can handle, so an input with no
    /// fractional exponent produces literally the expression `series` produces.
    ///
    /// This is the regression that matters for the "existing behaviour is
    /// untouched" claim — checked, not assumed.
    #[test]
    fn an_analytic_expansion_is_identical_to_the_one_series_returns() {
        let (p, x) = pool_with_x();
        let cases: Vec<(ExprId, u32)> = vec![
            (p.func("sin", vec![x]), 6),
            (p.func("cos", vec![x]), 6),
            (p.func("exp", vec![x]), 5),
            (p.func("sqrt", vec![p.add(vec![p.integer(1), x])]), 5),
            (p.func("tan", vec![x]), 6),
            (p.pow(p.add(vec![p.integer(1), x]), p.integer(-1)), 5),
            (
                p.mul(vec![p.func("sin", vec![x]), p.pow(x, p.integer(-1))]),
                4,
            ),
        ];
        for (e, order) in cases {
            let s = series(e, x, p.integer(0), order, &p).unwrap();
            let px = puiseux_series(e, x, p.integer(0), order, &p).unwrap();
            assert_eq!(px.ramification(), 1, "{}", p.display(e));
            assert_eq!(
                px.expr(),
                s.expr(),
                "{} : {} vs {}",
                p.display(e),
                p.display(px.expr()),
                p.display(s.expr())
            );
        }
    }

    /// A **polar** expansion is the one place the two conventions differ, and
    /// the difference is that this one is *sharper*: `series` labels a Laurent
    /// result `O(h¹)` whatever the order, while this returns every term below
    /// `order` and labels it `O(h^order)`. The coefficients agree.
    #[test]
    fn a_polar_expansion_keeps_more_terms_than_series_labels() {
        let (p, x) = pool_with_x();
        let e = p.pow(p.func("sin", vec![x]), p.integer(-1));
        let px = puiseux_series(e, x, p.integer(0), 4, &p).unwrap();
        assert_eq!(px.ramification(), 1);
        assert_terms(
            &coeffs(&px, &p),
            &[((-1, 1), 1.0), ((1, 1), 1.0 / 6.0), ((3, 1), 7.0 / 360.0)],
        );
        assert_eq!(*px.remainder_order(), Rational::from(4));
    }

    // -----------------------------------------------------------------------
    // Free parameters, budget, invariants
    // -----------------------------------------------------------------------

    /// A symbolic parameter survives — and is checked at two generic values, so
    /// a coefficient that is only right for one of them cannot pass.
    #[test]
    fn a_free_parameter_survives_and_is_checked_at_two_values() {
        let (p, x) = pool_with_x();
        let k = p.symbol("k", Domain::Real);
        let e = p.mul(vec![k, p.func("sqrt", vec![x])]);
        let px = puiseux_series(e, x, p.integer(0), 4, &p).unwrap();
        assert_eq!(px.ramification(), 2);
        assert_eq!(px.terms().len(), 1);
        assert_eq!(px.terms()[0].0, Rational::from((1, 2)));
        assert_eq!(px.terms()[0].1, k);
        // Two parameter assignments × two ladder rungs.
        assert_eq!(px.evidence().rungs, 4);
    }

    /// A budget stops the expander rather than letting it run, and the stop is
    /// a refusal, never a shorter series under the requested `O(·)` label.
    #[test]
    fn a_budget_stops_the_expansion() {
        use crate::budget::{self, Budget, BudgetError};
        let (p, x) = pool_with_x();
        let s = p.func("sin", vec![x]);
        let e = p.func("sqrt", vec![s]);
        let _guard = budget::enter(Budget::new().with_max_steps(3));
        let err = puiseux_series(e, x, p.integer(0), 12, &p).unwrap_err();
        assert!(
            matches!(
                err,
                PuiseuxError::Exhausted(Some(BudgetError::Steps { .. }))
            ),
            "{err:?}"
        );
        assert_eq!(err.code(), "E-SERIES-003");
    }

    /// Every truncation exponent the module reports is one it can justify: the
    /// terms it returns really are every term below `remainder_order`, strictly
    /// ascending, and every one of them lies on the `(1/e)ℤ` lattice the
    /// reported ramification claims.
    #[test]
    fn the_reported_exponents_match_the_reported_lattice_and_order() {
        let (p, x) = pool_with_x();
        let s = p.func("sin", vec![x]);
        for order in 2..=8u32 {
            let e = p.func("sqrt", vec![s]);
            let px = puiseux_series(e, x, p.integer(0), order, &p).unwrap();
            let order_q = Rational::from(order);
            assert_eq!(*px.remainder_order(), order_q);
            let e_r = px.ramification();
            let mut last: Option<Rational> = None;
            for (ex, _) in px.terms() {
                assert!(*ex < order_q, "order {order}: {ex}");
                if let Some(l) = &last {
                    assert!(ex > l, "order {order}: {ex} after {l}");
                }
                assert_eq!(e_r % ex.denom().to_u64().unwrap(), 0, "order {order}: {ex}");
                last = Some(ex.clone());
            }
        }
    }
}
