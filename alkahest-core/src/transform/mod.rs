//! Symbolic integral and discrete transforms.
//!
//! - The [`laplace`] submodule provides the **Laplace transform** `L{f(t)}(s)`
//!   and its **inverse** `L⁻¹{F(s)}(t)`.
//! - The [`fourier`] submodule provides the **Fourier transform** `F{f(x)}(ξ)`
//!   and its **inverse** `F⁻¹{g(ξ)}(x)` (unitary, ordinary-frequency convention).
//! - The [`ztransform`] submodule provides the **(unilateral) Z-transform**
//!   `Z{a[n]}(z)` and its **inverse** `Z⁻¹{A(z)}(n)`.
//!
//! These are *formal* transforms: no region-of-convergence is tracked, and the
//! abscissa of convergence of an answer is not reported (matching SymPy's
//! `noconds=True` default).  See each submodule for its rule coverage,
//! fallbacks, and declines.
//!
//! # Symbolic parameters and genericity — `Genericity`
//!
//! The transforms accept expressions whose coefficients, shifts and rates
//! mention symbols other than the transform variable (`ω`, `ζ`, `K`, `ka`,
//! `ke`, `a`, …).  Such an answer can rest on hypotheses about those symbols
//! that are *not* consequences of the input:
//!
//! ```text
//!   L⁻¹{1/((s+ka)(s+ke))}  =  (e^{−ka t} − e^{−ke t}) / (ke − ka)    for ka ≠ ke
//!                          =  t·e^{−ka t}                            at ka = ke
//!
//!   L{θ(t−a)}              =  e^{−a s}/s                             for a ≥ 0
//!                          =  1/s                                    at a < 0
//!
//!   F{2a/(a² + 4π²x²)}     =  e^{−a|ξ|}                              for a > 0
//!                          =  −e^{a|ξ|}                              at a < 0
//! ```
//!
//! In each pair both lines are right and neither is right in general.  Every
//! such hypothesis is recorded as a [`SideCondition`] and returned by the
//! `*_with_conditions` entry points.  The historical signatures
//! ([`laplace_transform`], [`inverse_laplace_transform`], [`fourier_transform`],
//! [`inverse_fourier_transform`], [`inverse_z_transform`]) cannot express it in
//! their return type, so for them the hypotheses travel on the consuming
//! thread-local [`take_transform_side_conditions`] — the same out-of-band
//! channel as [`crate::solver::take_solve_side_conditions`].
//!
//! An **empty** condition list means every branch taken was forced, not that
//! nothing was examined.
//!
//! ## A hypothesis and a refutation are different things
//!
//! A condition is recorded only where the input leaves it *open*.  Where the
//! input **settles it the wrong way** — a literal negative shift in
//! `L{θ(t−a)}`, a literal non-positive rate in `F{e^{−a|x|}}`, an advance
//! `e^{+a s}` handed to `L⁻¹` — there is no branch to report, only a wrong
//! answer to decline, and the call returns `NoRule` / `NotInvertible` instead.
//! Recording an unsatisfiable side condition would be the worst of both: an
//! answer that is wrong, carrying a hypothesis that can never be discharged.

use crate::deriv::SideCondition;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::logic::{satisfiable, Satisfiability};
use crate::simplify::assumptions::AssumptionContext;

pub mod fourier;
pub mod laplace;
pub mod ztransform;

pub use fourier::{
    fourier_transform, fourier_transform_with_conditions, inverse_fourier_transform,
    inverse_fourier_transform_with_conditions, FourierError,
};
pub use laplace::{
    inverse_laplace_transform, inverse_laplace_transform_with_assumptions,
    inverse_laplace_transform_with_conditions, laplace_transform,
    laplace_transform_with_conditions, LaplaceError,
};
pub use ztransform::{
    inverse_z_transform, inverse_z_transform_with_assumptions, inverse_z_transform_with_conditions,
    z_shift_advance, z_shift_delay, z_transform, ZTransformError,
};

// ===========================================================================
// Genericity bookkeeping
// ===========================================================================

std::thread_local! {
    static TRANSFORM_CONDITIONS: std::cell::RefCell<Vec<SideCondition>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The hypotheses the most recent [`inverse_laplace_transform`] /
/// [`inverse_z_transform`] call on this thread rests on.
///
/// Consuming, so one call's hypotheses cannot be read as a later call's.  Empty
/// means every division and every branch was forced by the input, not that
/// nothing was checked.  Prefer the `*_with_conditions` entry points, which
/// return the same list in band.
pub fn take_transform_side_conditions() -> Vec<SideCondition> {
    TRANSFORM_CONDITIONS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

pub(crate) fn stash_transform_side_conditions(conds: Vec<SideCondition>) {
    TRANSFORM_CONDITIONS.with(|c| *c.borrow_mut() = conds);
}

/// What an inverse transform had to assume, and what the caller already knows.
///
/// Every place the tables divide by a parametric quantity, or pick one of two
/// genuinely different closed forms, goes through [`Genericity::need_nonzero`]
/// or [`Genericity::need_positive`].  Each first tries to *discharge* the
/// hypothesis — from a literal value, from the caller's
/// [`AssumptionContext`], from static symbol domains
/// ([`crate::kernel::Domain::Positive`] / `NonZero`), or from
/// [`crate::logic::satisfiable`] — and only records it when it cannot.
///
/// Discharging is deliberately incomplete: `satisfiable` answering `Unknown`
/// leaves the hypothesis recorded, which over-reports but never under-reports.
#[derive(Default)]
pub(crate) struct Genericity<'a> {
    assumptions: Option<&'a AssumptionContext>,
    conditions: Vec<SideCondition>,
}

impl<'a> Genericity<'a> {
    pub(crate) fn new(assumptions: Option<&'a AssumptionContext>) -> Self {
        Genericity {
            assumptions,
            conditions: Vec::new(),
        }
    }

    pub(crate) fn into_conditions(self) -> Vec<SideCondition> {
        self.conditions
    }

    /// Record `e ≠ 0` unless it can be discharged.
    ///
    /// `e` is first split into irreducible factors, so a single division by
    /// `ω²(1 − ζ²)` reports the three separate facts `ω ≠ 0`, `ζ ≠ 1`,
    /// `ζ ≠ −1` rather than one opaque product — and so `ω² ≠ 0` reports as
    /// `ω ≠ 0`, which is the same statement in a form a caller can act on.
    pub(crate) fn need_nonzero(&mut self, e: ExprId, pool: &ExprPool) {
        for f in irreducible_factors(e, pool) {
            let cond = SideCondition::NonZero(f);
            if self.holds(&cond, pool) {
                continue;
            }
            self.push(cond);
        }
    }

    /// Record `e > 0` unless it can be discharged.
    ///
    /// An even power of a real base is non-negative for free, so `Positive(ω²)`
    /// degrades to the strictly weaker-looking but equivalent `ω ≠ 0` instead
    /// of asking the caller to accept a positivity claim they cannot check.
    pub(crate) fn need_positive(&mut self, e: ExprId, pool: &ExprPool) {
        if let Some(base) = even_power_base(e, pool) {
            self.need_nonzero(base, pool);
            return;
        }
        let cond = SideCondition::Positive(e);
        if self.holds(&cond, pool) {
            return;
        }
        self.push(cond);
    }

    /// Require `e ∈ d` (used for the `a ≥ 0` hypothesis of the unilateral
    /// time-shift rule, where `Positive` would wrongly exclude `a = 0`).
    pub(crate) fn need_in_domain(&mut self, e: ExprId, d: crate::kernel::Domain, pool: &ExprPool) {
        let cond = SideCondition::InDomain(e, d);
        if self.holds(&cond, pool) {
            return;
        }
        self.push(cond);
    }

    /// Take on a hypothesis raised elsewhere (`apart`'s decomposition), giving
    /// it the same discharge attempt and the same factoring as a locally
    /// raised one.
    pub(crate) fn adopt(&mut self, cond: SideCondition, pool: &ExprPool) {
        match cond {
            SideCondition::NonZero(e) => self.need_nonzero(e, pool),
            SideCondition::Positive(e) => self.need_positive(e, pool),
            SideCondition::InDomain(e, d) => self.need_in_domain(e, d, pool),
        }
    }

    /// Is `e` known to be strictly positive?  Used to *choose* a branch, so it
    /// must only answer `true` when the fact is established.
    pub(crate) fn known_positive(&self, e: ExprId, pool: &ExprPool) -> bool {
        self.holds(&SideCondition::Positive(e), pool)
    }

    /// Is `e` known to be strictly negative?
    pub(crate) fn known_negative(&self, e: ExprId, pool: &ExprPool) -> bool {
        let neg = pool.mul(vec![pool.integer(-1_i32), e]);
        let neg = crate::simplify::simplify(neg, pool).value;
        self.holds(&SideCondition::Positive(neg), pool)
    }

    fn push(&mut self, cond: SideCondition) {
        if !self.conditions.contains(&cond) {
            self.conditions.push(cond);
        }
    }

    /// Can `cond` be established without assuming it?
    fn holds(&self, cond: &SideCondition, pool: &ExprPool) -> bool {
        if let SideCondition::InDomain(e, d) = cond {
            return self.in_domain_holds(*e, *d, pool);
        }
        let (target, want_positive) = match cond {
            SideCondition::NonZero(e) => (*e, false),
            SideCondition::Positive(e) => (*e, true),
            SideCondition::InDomain(..) => unreachable!("handled above"),
        };

        // 1. A literal decides itself, and so does a named mathematical
        //    constant — see [`evidently_positive_constant`].
        if let Some(r) = literal_rational(target, pool) {
            return if want_positive { r > 0 } else { r != 0 };
        }
        if evidently_positive_constant(target, pool) {
            return true;
        }

        // 2. Facts: the caller's context plus static symbol domains.
        let mut facts: Vec<SideCondition> = self
            .assumptions
            .map(|a| a.facts().to_vec())
            .unwrap_or_default();
        crate::simplify::assumptions::collect_static_domain_facts(target, pool, &mut facts);
        if fact_entails(&facts, target, want_positive, pool) {
            return true;
        }

        // 3. Ask the solver whether the assumptions rule out the negation.
        //    `Unknown` (the common answer for anything nonlinear) is not a
        //    proof, so it leaves the hypothesis standing.
        self.entailed_by_predicates(target, want_positive, pool)
    }

    /// Only `NonNegative` is decidable here — it is the one domain the tables
    /// actually need, and the only one the fact vocabulary can establish.
    fn in_domain_holds(&self, e: ExprId, d: crate::kernel::Domain, pool: &ExprPool) -> bool {
        use crate::kernel::Domain;
        if d != Domain::NonNegative {
            return false;
        }
        if let Some(r) = literal_rational(e, pool) {
            return r >= 0;
        }
        if let ExprData::Symbol { domain, .. } = pool.get(e) {
            if matches!(domain, Domain::Positive | Domain::NonNegative) {
                return true;
            }
        }
        let mut facts: Vec<SideCondition> = self
            .assumptions
            .map(|a| a.facts().to_vec())
            .unwrap_or_default();
        crate::simplify::assumptions::collect_static_domain_facts(e, pool, &mut facts);
        if facts
            .iter()
            .any(|f| matches!(f, SideCondition::Positive(id) if *id == e))
        {
            return true;
        }
        let Some(ctx) = self.assumptions else {
            return false;
        };
        if ctx.predicates().is_empty() {
            return false;
        }
        let zero = pool.integer(0_i32);
        let negated = pool.predicate(PredicateKind::Lt, vec![e, zero]);
        let mut conj = ctx.predicates().to_vec();
        conj.push(negated);
        let formula = pool.predicate(PredicateKind::And, conj);
        matches!(satisfiable(formula, pool), Satisfiability::Unsat)
    }

    fn entailed_by_predicates(&self, target: ExprId, want_positive: bool, pool: &ExprPool) -> bool {
        let Some(ctx) = self.assumptions else {
            return false;
        };
        if ctx.predicates().is_empty() {
            return false;
        }
        let zero = pool.integer(0_i32);
        // ¬goal: `target ≤ 0` for positivity, `target = 0` for non-vanishing.
        let negated = if want_positive {
            pool.predicate(PredicateKind::Le, vec![target, zero])
        } else {
            pool.predicate(PredicateKind::Eq, vec![target, zero])
        };
        let mut conj = ctx.predicates().to_vec();
        conj.push(negated);
        let formula = pool.predicate(PredicateKind::And, conj);
        matches!(satisfiable(formula, pool), Satisfiability::Unsat)
    }
}

/// Does one of `facts` establish `target ≠ 0` (or `target > 0`)?
///
/// Structural and deliberately shallow: a product is non-zero when every
/// factor is, an integer power is non-zero when its base is, and a positive
/// quantity is non-zero.  Anything subtler is left to
/// [`Genericity::entailed_by_predicates`].
fn fact_entails(
    facts: &[SideCondition],
    target: ExprId,
    want_positive: bool,
    pool: &ExprPool,
) -> bool {
    if facts.iter().any(|f| match f {
        SideCondition::Positive(id) => *id == target,
        SideCondition::NonZero(id) => !want_positive && *id == target,
        SideCondition::InDomain(..) => false,
    }) {
        return true;
    }
    if want_positive {
        return false;
    }
    // `−e ≠ 0` is `e ≠ 0`.
    let neg = crate::simplify::simplify(pool.mul(vec![pool.integer(-1_i32), target]), pool).value;
    if facts.iter().any(|f| match f {
        SideCondition::NonZero(id) | SideCondition::Positive(id) => *id == neg,
        SideCondition::InDomain(..) => false,
    }) {
        return true;
    }
    match pool.get(target) {
        ExprData::Mul(args) => args
            .iter()
            .all(|&a| fact_entails(facts, a, false, pool) || literal_nonzero(a, pool)),
        ExprData::Pow { base, exp } => {
            matches!(pool.get(exp), ExprData::Integer(_))
                && (fact_entails(facts, base, false, pool) || literal_nonzero(base, pool))
        }
        _ => false,
    }
}

fn literal_nonzero(e: ExprId, pool: &ExprPool) -> bool {
    literal_rational(e, pool).is_some_and(|r| r != 0)
}

fn literal_rational(expr: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

/// Is `e` built only from things that are positive *as a matter of fact*?
///
/// `π` is interned as an ordinary [`ExprData::Symbol`] whose `Domain` depends
/// on whichever call created it first, so without this
/// [`Genericity::need_positive`] would hand the caller `π > 0` as a hypothesis
/// to discharge — and a hypothesis a caller cannot decline is not a hypothesis,
/// it is noise that hides the real ones.  `L⁻¹{π/(s² + π²)}` reported `π ≠ 0`
/// and `F{e^{−πx²}}` reported `π > 0` on exactly this route.
///
/// Deliberately short: only positive literals, `π`, and products/powers of
/// them.  Anything else stays a hypothesis, which over-reports and never
/// under-reports.
fn evidently_positive_constant(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Integer(_) | ExprData::Rational(_) => {
            literal_rational(e, pool).is_some_and(|r| r > 0)
        }
        ExprData::Float(f) => f.inner.to_f64() > 0.0,
        ExprData::Symbol { name, .. } => &*name == "pi",
        ExprData::Mul(args) => {
            !args.is_empty() && args.iter().all(|&a| evidently_positive_constant(a, pool))
        }
        // A positive base raised to a real power is positive; a literal
        // rational exponent is the only case the tables produce (`π²`, `√π`).
        ExprData::Pow { base, exp } => {
            evidently_positive_constant(base, pool) && literal_rational(exp, pool).is_some()
        }
        _ => false,
    }
}

/// `b` when `e` is `b^{2k}` for a real `b` and `k ≥ 1`, else `None`.
///
/// Used to turn a positivity hypothesis into the non-vanishing hypothesis it is
/// equivalent to over ℝ.
fn even_power_base(e: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let ExprData::Pow { base, exp } = pool.get(e) else {
        return None;
    };
    let ExprData::Integer(n) = pool.get(exp) else {
        return None;
    };
    let k = n.0.to_i64()?;
    if k >= 2 && k % 2 == 0 && evidently_real(base, pool) {
        Some(base)
    } else {
        None
    }
}

/// Conservative real-valuedness test: literals and arithmetic over symbols
/// whose declared [`crate::kernel::Domain`] is not `Complex`.
///
/// Only used to justify `b^{2k} ≥ 0`, so `false` costs a slightly clumsier
/// side condition and never costs correctness.
fn evidently_real(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
        ExprData::Symbol { domain, .. } => domain != crate::kernel::Domain::Complex,
        ExprData::Add(args) | ExprData::Mul(args) => args.iter().all(|&a| evidently_real(a, pool)),
        ExprData::Pow { base, exp } => {
            matches!(pool.get(exp), ExprData::Integer(_)) && evidently_real(base, pool)
        }
        _ => false,
    }
}

/// Split `e` into the irreducible factors a `≠ 0` claim should be reported on.
///
/// A ℤ-polynomial in `e`'s free symbols is factored (`ω²(1 − ζ²)` → `ω`,
/// `ζ − 1`, `ζ + 1`, dropping multiplicities, which do not change vanishing);
/// anything that is not such a polynomial is reported whole.
fn irreducible_factors(e: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    use crate::poly::{collect_free_vars, MultiPoly};
    let vars = collect_free_vars(e, pool);
    if vars.is_empty() {
        return vec![e];
    }
    let Ok(mp) = MultiPoly::from_symbolic(e, vars, pool) else {
        return vec![e];
    };
    let Some((_unit, factors)) = mp.factor_irreducible() else {
        return vec![e];
    };
    let mut out = Vec::new();
    for (f, _mult) in factors {
        if f.total_degree() == 0 {
            continue;
        }
        let fe = f.to_expr(pool);
        if !out.contains(&fe) {
            out.push(fe);
        }
    }
    if out.is_empty() {
        vec![e]
    } else {
        out
    }
}
