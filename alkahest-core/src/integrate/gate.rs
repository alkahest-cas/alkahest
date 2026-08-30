//! **Propose → fit → verify → emit-or-decline**: a reusable soundness gate for
//! integration routes whose *candidate* antiderivatives are produced by an
//! untrusted search.
//!
//! # Scope
//!
//! Several integration routes can cheaply *guess* the shape of an
//! antiderivative but cannot cheaply *derive* it: a Byrd–Friedman elliptic
//! block set, a real partial-fraction log/arctan basis, a rationalizing
//! substitution `u = φ(x)`.  For those routes this module supplies the missing
//! half of the pattern:
//!
//! ```text
//!   propose   a candidate `F` (or an ansatz basis `B₁…Bₙ`)
//!   fit       the block coefficients numerically (least squares on in-domain
//!             samples, then snapped to exact rationals)
//!   verify    `d/dx F` against the integrand `f` through a graded gate
//!   emit      only when the verdict clears the caller's floor, else decline
//! ```
//!
//! The inversion of the usual CAS trade-off is the whole point.  Numeric
//! coefficient fitting is normally unacceptable because its output cannot be
//! trusted; here it is only ever allowed to **propose**.  Nothing is emitted
//! that the verification tier did not independently check, so a bad fit costs
//! CPU and nothing else.  Numeric fitting is never the reason a result is
//! emitted.
//!
//! # Method — a tiered gate
//!
//! [`verify`] runs up to three tiers, cheapest first, and reports the
//! strongest one that succeeded.
//!
//! 1. **Symbolic.**  `simplify(d/dx F − f)`, optionally followed by the
//!    e-graph simplifier.  If the residual reduces to a syntactic zero the
//!    verdict is [`Verdict::Proven`] and no numerics run at all.  This is the
//!    cheapest tier *and* the strongest outcome, which is why it goes first.
//! 2. **Sampled (`f64`).**  Evaluate `d/dx F` and `f` at the caller's in-domain
//!    sample points in IEEE-754 double precision and compare with a relative
//!    tolerance.  A single in-domain disagreement is a **refutation**
//!    ([`Verdict::Failed`]); too few evaluable points is a **decline**
//!    ([`Verdict::Declined`]), never a pass.
//! 3. **Rigorous enclosure.**  Bound `d/dx F − f` over the caller's closed
//!    boxes with [`crate::validated::bounds::bound_on_box`] — Taylor models in
//!    outward-rounded [`crate::ball`] arithmetic.  When the bound clears the
//!    budget's tolerance the verdict is upgraded to
//!    [`Verdict::EnclosureVerified`], which states a property of the *whole
//!    interval* rather than of finitely many points.
//!
//! # What each verdict does and does not prove
//!
//! | Verdict | Proves | Does **not** prove |
//! |---|---|---|
//! | [`Verdict::Proven`] | `d/dx F − f` is the zero expression after simplification: an identity wherever both sides are defined | nothing about domains — `F` may still be the wrong branch on part of the real line |
//! | [`Verdict::EnclosureVerified`] | `sup{ \|d/dx F(x) − f(x)\| : x ∈ ⋃ boxes } ≤ residual_bound`, rigorously (outward rounding, remainder terms folded in, never dropped) | that the residual is *exactly* zero; that anything holds **outside** the listed boxes — in particular near branch points, poles and at infinity |
//! | [`Verdict::SampledOnly`] | `\|d/dx F − f\| ≤ tol·(1+\|f\|)` at the reported number of in-domain `f64` sample points | any statement between the sample points — agreement at finitely many points is evidence, not an identity |
//! | [`Verdict::Failed`] | the candidate disagrees with the integrand at a specific in-domain point, beyond tolerance and beyond plausible `f64` error | — |
//! | [`Verdict::Declined`] | nothing at all — the gate could not run | — |
//!
//! [`Verdict::Failed`] and [`Verdict::Declined`] both mean *do not emit*.  They
//! are kept distinct because only the first is evidence about the candidate.
//!
//! # Honest limitations
//!
//! * **No verdict here is a proof of the integral.**  Even [`Verdict::Proven`]
//!   is a statement about `simplify`, i.e. about a rewrite system that is
//!   sound but not complete and that carries side conditions
//!   (branch cuts, non-vanishing denominators) it does not always re-check.
//!   The gate certifies *`d/dx F = f`*, never *`F` is the antiderivative you
//!   wanted on the branch you meant*.
//! * **The enclosure tier is domain-limited by construction.**  A residual of
//!   the form `(…)/√P` is unbounded at every root of `P`, so no rigorous
//!   bound over a box containing a branch point can exist.  Callers must hand
//!   in boxes that stay strictly inside the domain, and the verdict records
//!   exactly which boxes were covered.  Neighbourhoods of the branch points,
//!   and the unbounded tails, are **never** covered.
//! * **The enclosure tier is not cheap.**  A single Taylor-model
//!   branch-and-bound pass costs orders of magnitude more than the `f64`
//!   screen (measured: hundreds of milliseconds versus microseconds for an
//!   elliptic candidate).  That is why it is budgeted, and why it runs only on
//!   candidates that already survived the screen.  When the budget runs out
//!   before the tolerance is met, the verdict stays [`Verdict::SampledOnly`] —
//!   a wide-but-true bound is never quietly reported as a tight one.
//! * **The enclosure tier refuses expressions it cannot Taylor-model.**
//!   `EllipticF`/`EllipticE`/`EllipticPi`, `RootSum`, and any primitive with
//!   no Taylor rule make [`crate::validated::bounds::bound_on_box`] refuse.
//!   For the elliptic route this happens not to bite — differentiating an
//!   elliptic block with a *constant* modulus produces an elementary residual
//!   — but a candidate that keeps a special function in its derivative gets
//!   [`Verdict::SampledOnly`], explicitly, never a silent pass.
//! * **A candidate with an interior removable singularity is only covered
//!   around it.**  The enclosure tier bounds the residual *as written*.  If
//!   the candidate contains a `1/S(x)` whose `S` vanishes somewhere inside a
//!   box — which happens even when the residual's true value there is
//!   perfectly finite — `bound_on_box` refuses on that box.  The gate halves
//!   and retries, so the certified coverage is the box *minus a
//!   neighbourhood of the written singularity*, and the gap is visible in the
//!   `boxes` list.  It is not filled in silently.  (The `∫dx/√(x⁴+1)`
//!   reduction is a live example: it certifies `[0, 2.2]`, `[−2.2, −1.1]` and
//!   `[−0.55, 0]`, and leaves `(−1.1, −0.55)` uncovered.)
//! * **The `f64` tier can be defeated by catastrophic cancellation.**  A
//!   residual whose true value is `1e-9` and whose evaluation error is `1e-8`
//!   is indistinguishable from zero at these tolerances.  This is exactly the
//!   weakness the enclosure tier exists to remove, and exactly why callers
//!   that can afford it should ask for [`Strength::Enclosure`].
//! * **The numeric evaluator is registry-backed, not universal.**  Function
//!   nodes are dispatched through [`crate::primitive::PrimitiveRegistry`], so
//!   the gate sees exactly the function set the registry advertises — but
//!   `Piecewise` with an indeterminate predicate and unregistered heads are
//!   not evaluable and their sample points are skipped.  If that leaves too
//!   few points the gate declines.  A `RootSum` *is* evaluable as of the
//!   `crate::eval` `root_sum` module — numerically, by root-finding — which is
//!   what makes a Rothstein–Trager answer with an algebraic residue something
//!   this gate can have an opinion about at all.
//! * **A candidate undefined where the integrand is finite is *skipped* here,
//!   not refuted — unlike at the engine-level gate.**  That asymmetry is
//!   deliberate and it is load-bearing, so it is worth stating plainly.
//!   [`crate::integrate::verify_antiderivative_status`] treats a sample where
//!   `f` is an ordinary finite real and `d/dx F` is not as a **disagreement**,
//!   because there `F` is not an antiderivative of `f` (see
//!   `engine::classify_sample`, and Charlwood #35, which is exactly that).
//!   Applying the same rule *here* was tried and reverted: this gate's callers
//!   deliberately sample beyond the region their reduction claims.
//!   [`crate::integrate::algebraic::elliptic_output`]'s `gate_samples` puts
//!   points in **every** `P > 0` interval and says so — "points where the
//!   substitution is invalid simply evaluate non-finite and are skipped" —
//!   while a cubic-three-real reduction is valid only beyond the largest root.
//!   For `∫dx/√(x³−x)` the integrand is finite on `(−1, 0)` and the elliptic
//!   candidate's derivative is not, so the stricter rule refuses four correct,
//!   branch-limited elliptic answers (`tests/test_elliptic_integrate.py`).
//!   Adopting it needs the sample set narrowed to the region each reduction
//!   actually claims — or a [`Verdict`] that can name that region — not a
//!   one-line change to the loop.
//!
//! # Relationship to `verify_antiderivative_status`
//!
//! [`crate::integrate::verify_antiderivative_status`] is the *engine-level*
//! post-check applied to a finished answer.  This module is the
//! *route-level* gate applied to a candidate that a search proposed, before it
//! is allowed to become an answer.  They are complementary: this gate is
//! domain-aware (it is handed the region where the reduction is valid) and
//! graded, and it is what makes an untrusted proposer safe to run at all.

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::primitive::PrimitiveRegistry;
use crate::simplify::engine::simplify;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------

/// How strong a passing [`Verdict`] is.  Ordered: a caller asking for
/// [`Strength::Enclosure`] also accepts [`Strength::Symbolic`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Strength {
    /// `f64` agreement at finitely many in-domain points.
    Sampled,
    /// Rigorous enclosure of the residual over stated closed boxes.
    Enclosure,
    /// The simplified residual is syntactically zero.
    Symbolic,
}

/// Why the gate could not reach a verdict about the candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeclineReason {
    /// `d/dx candidate` could not be formed.
    Differentiation,
    /// Fewer than `min_points` sample points were both in-domain and
    /// numerically evaluable on **both** sides.
    NotEnoughPoints {
        /// How many usable points were found.
        found: usize,
        /// How many were required.
        required: usize,
    },
    /// A rigorous enclosure was *required* and could not be established.
    EnclosureUnavailable {
        /// The refusal reported by the validated-numerics subsystem, or a
        /// description of why no box could be covered.
        what: String,
    },
    /// The verdict reached was weaker than the caller's floor.
    BelowRequiredStrength {
        /// What the gate managed to establish.
        reached: Strength,
        /// What the caller asked for.
        required: Strength,
    },
}

impl std::fmt::Display for DeclineReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeclineReason::Differentiation => {
                write!(f, "the candidate could not be differentiated")
            }
            DeclineReason::NotEnoughPoints { found, required } => write!(
                f,
                "only {found} in-domain sample points were evaluable, {required} required"
            ),
            DeclineReason::EnclosureUnavailable { what } => {
                write!(f, "no rigorous enclosure could be established: {what}")
            }
            DeclineReason::BelowRequiredStrength { reached, required } => write!(
                f,
                "verification reached {reached:?} but {required:?} was required"
            ),
        }
    }
}

/// One closed box on which the residual was rigorously bounded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerifiedBox {
    /// Left endpoint (inclusive).
    pub lo: f64,
    /// Right endpoint (inclusive).
    pub hi: f64,
    /// Rigorous upper bound for `|d/dx F − f|` everywhere on `[lo, hi]`.
    pub residual_bound: f64,
}

/// The graded outcome of [`verify`].
///
/// See the module docs for the precise statement each variant does and does
/// not support.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// `simplify(d/dx F − f)` is a syntactic zero.
    Proven,
    /// The residual was rigorously bounded over every listed box.
    ///
    /// This **subsumes** the sampled tier: the candidate also passed the
    /// `f64` screen at `points` points before the enclosure ran.
    EnclosureVerified {
        /// The boxes actually covered, each with its own rigorous bound.
        boxes: Vec<VerifiedBox>,
        /// `max` of the per-box bounds — a rigorous bound for `|d/dx F − f|`
        /// on the union of the boxes.
        residual_bound: f64,
        /// Sample points that passed the `f64` screen first.
        points: usize,
        /// Relative tolerance used by that screen.
        tolerance: f64,
    },
    /// Only the `f64` screen passed (or was attempted).
    SampledOnly {
        /// In-domain points at which both sides evaluated and agreed.
        points: usize,
        /// Relative tolerance used.
        tolerance: f64,
    },
    /// Refuted: the sides disagree at a specific in-domain point.
    Failed {
        /// Where.
        at: f64,
        /// `|d/dx F − f|` there.
        residual: f64,
        /// The relative tolerance that was exceeded.
        tolerance: f64,
    },
    /// The gate could not run.  Says nothing about the candidate.
    Declined {
        /// Why.
        reason: DeclineReason,
    },
}

impl Verdict {
    /// The strength of a passing verdict, or `None` for
    /// [`Verdict::Failed`] / [`Verdict::Declined`].
    pub fn strength(&self) -> Option<Strength> {
        match self {
            Verdict::Proven => Some(Strength::Symbolic),
            Verdict::EnclosureVerified { .. } => Some(Strength::Enclosure),
            Verdict::SampledOnly { .. } => Some(Strength::Sampled),
            Verdict::Failed { .. } | Verdict::Declined { .. } => None,
        }
    }

    /// `true` when the verdict is at least `min`.
    pub fn meets(&self, min: Strength) -> bool {
        self.strength().is_some_and(|s| s >= min)
    }

    /// `true` for any passing verdict (the historical boolean gate).
    pub fn is_verified(&self) -> bool {
        self.strength().is_some()
    }
}

// ---------------------------------------------------------------------------
// Domain
// ---------------------------------------------------------------------------

/// Where the candidate is claimed to be an antiderivative.
///
/// Domain-awareness is not decoration.  A Byrd–Friedman reduction is only
/// valid on the real region where the radicand is positive; sampling outside
/// it compares two things that are both `NaN` and proves nothing.  The caller
/// owns that knowledge, so the caller supplies it here.
///
/// * `samples` — candidate abscissae for the `f64` screen.  Points failing
///   `predicate` are skipped, not counted, and never cause a failure.
/// * `predicate` — the in-domain test (default: everything).
/// * `boxes` — closed intervals, **strictly inside** the domain, on which a
///   rigorous enclosure may be attempted.  Empty means "no enclosure tier".
pub struct Domain<'a> {
    samples: Vec<f64>,
    predicate: Option<Box<dyn Fn(f64) -> bool + 'a>>,
    boxes: Vec<(f64, f64)>,
}

impl std::fmt::Debug for Domain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Domain")
            .field("samples", &self.samples.len())
            .field("predicate", &self.predicate.is_some())
            .field("boxes", &self.boxes)
            .finish()
    }
}

impl<'a> Domain<'a> {
    /// A domain given by an explicit list of sample abscissae, with no
    /// restriction and no enclosure boxes.
    pub fn from_samples(samples: Vec<f64>) -> Self {
        Domain {
            samples,
            predicate: None,
            boxes: Vec::new(),
        }
    }

    /// Restrict to the points where `p` holds.
    pub fn with_predicate(mut self, p: impl Fn(f64) -> bool + 'a) -> Self {
        self.predicate = Some(Box::new(p));
        self
    }

    /// Attach the closed boxes the enclosure tier may use.
    ///
    /// Degenerate or non-finite boxes are dropped.
    pub fn with_boxes(mut self, boxes: Vec<(f64, f64)>) -> Self {
        self.boxes = boxes
            .into_iter()
            .filter(|(lo, hi)| lo.is_finite() && hi.is_finite() && hi > lo)
            .collect();
        self
    }

    /// Is `x` inside the domain?
    pub fn contains(&self, x: f64) -> bool {
        x.is_finite() && self.predicate.as_ref().map_or(true, |p| p(x))
    }

    /// The sample abscissae (unfiltered).
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }

    /// The enclosure boxes.
    pub fn boxes(&self) -> &[(f64, f64)] {
        &self.boxes
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Work budget and accuracy target for the rigorous enclosure tier.
///
/// `tol` is an **absolute** bound on `|d/dx F − f|`.  It is meaningful only
/// because the caller chose boxes on which the integrand itself is `O(1)`;
/// there is deliberately no normalisation by a *sampled* integrand magnitude,
/// because a non-rigorous scale factor would let a loose enclosure clear a
/// threshold it had not earned.
#[derive(Clone, Copy, Debug)]
pub struct EnclosureBudget {
    /// Taylor expansion order per sub-box.
    pub order: usize,
    /// Working precision in bits.
    pub prec: u32,
    /// Absolute residual bound that counts as verified.
    pub tol: f64,
    /// Branch-and-bound subdivision budget, per box.
    pub max_subdivisions: usize,
    /// At most this many `bound_on_box` calls are made (cost control).  A box
    /// that refuses is halved and its halves re-queued, so this counts
    /// *attempts*, not distinct caller-supplied boxes.
    pub max_boxes: usize,
}

impl Default for EnclosureBudget {
    fn default() -> Self {
        EnclosureBudget {
            order: 6,
            prec: 128,
            tol: 1e-7,
            max_subdivisions: 64,
            max_boxes: 4,
        }
    }
}

impl EnclosureBudget {
    /// A deliberately small budget for hot paths: enough to certify an easy
    /// candidate, cheap enough to pay on every emission.
    pub fn cheap() -> Self {
        EnclosureBudget {
            order: 6,
            prec: 96,
            tol: 1e-7,
            max_subdivisions: 16,
            max_boxes: 2,
        }
    }

    /// A generous budget for offline / test use.
    pub fn thorough() -> Self {
        EnclosureBudget {
            order: 8,
            prec: 128,
            tol: 1e-8,
            max_subdivisions: 256,
            max_boxes: 12,
        }
    }
}

/// Whether, and how hard, to attempt the rigorous enclosure tier.
#[derive(Clone, Copy, Debug)]
pub enum EnclosurePolicy {
    /// Do not attempt it.  The gate tops out at [`Verdict::SampledOnly`].
    Skip,
    /// Attempt it; on failure keep the sampled verdict.  Purely additive.
    BestEffort(EnclosureBudget),
    /// Attempt it; on failure **decline** the candidate.
    Required(EnclosureBudget),
}

/// Tier configuration for [`verify`].
#[derive(Clone, Copy, Debug)]
pub struct GateOptions {
    /// Relative tolerance for the `f64` screen: a point passes when
    /// `|lhs − rhs| ≤ tolerance · (1 + |rhs|)`.
    pub tolerance: f64,
    /// Minimum number of in-domain, evaluable sample points.  Fewer is a
    /// decline.
    pub min_points: usize,
    /// Try `simplify(d/dx F − f) == 0` first.
    pub symbolic: bool,
    /// If the plain simplifier does not close the residual, also try the
    /// e-graph simplifier.  No-op without the `egraph` feature.
    pub egraph: bool,
    /// The rigorous tier's policy.
    pub enclosure: EnclosurePolicy,
    /// The caller's acceptance floor.  A verdict below this becomes
    /// [`Verdict::Declined`].
    pub min_strength: Strength,
}

impl Default for GateOptions {
    fn default() -> Self {
        GateOptions {
            tolerance: 1e-7,
            min_points: 3,
            symbolic: true,
            egraph: false,
            enclosure: EnclosurePolicy::BestEffort(EnclosureBudget::cheap()),
            min_strength: Strength::Sampled,
        }
    }
}

impl GateOptions {
    /// The historical elliptic-route gate: symbolic check, `f64` screen at
    /// `1e-7` relative over at least three points, no enclosure.
    pub fn sampled_only() -> Self {
        GateOptions {
            enclosure: EnclosurePolicy::Skip,
            ..GateOptions::default()
        }
    }

    /// Demand a rigorous enclosure; decline anything weaker.
    pub fn rigorous(budget: EnclosureBudget) -> Self {
        GateOptions {
            enclosure: EnclosurePolicy::Required(budget),
            min_strength: Strength::Enclosure,
            ..GateOptions::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Target
// ---------------------------------------------------------------------------

/// The integrand the candidate is checked against.
///
/// `integrand` is the symbolic form and is what the symbolic and enclosure
/// tiers use.  `numeric` is an optional fast `f64` evaluator for the sampled
/// tier; supplying one lets a caller that already holds the integrand in a
/// numerically stable coefficient form (e.g. ascending polynomial
/// coefficients) avoid re-evaluating a tree that may have removable
/// singularities at the sample points.
pub struct Target<'a> {
    /// Symbolic integrand `f`.
    pub integrand: ExprId,
    /// Optional `f64` evaluator for `f`.
    pub numeric: Option<&'a dyn Fn(f64) -> Option<f64>>,
}

impl<'a> Target<'a> {
    /// A target evaluated purely from its symbolic form.
    pub fn symbolic(integrand: ExprId) -> Self {
        Target {
            integrand,
            numeric: None,
        }
    }

    /// Attach a fast `f64` evaluator for the sampled tier.
    pub fn with_numeric(mut self, f: &'a dyn Fn(f64) -> Option<f64>) -> Self {
        self.numeric = Some(f);
        self
    }

    fn eval(&self, x: f64, var: ExprId, pool: &ExprPool) -> Option<f64> {
        match self.numeric {
            Some(f) => f(x),
            None => eval_at(self.integrand, var, x, pool),
        }
    }
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// Verify that `d/dx candidate = f` on `domain`, returning a graded verdict.
///
/// Never panics on user input: every failure path is a [`Verdict`].
pub fn verify(
    candidate: ExprId,
    target: &Target<'_>,
    var: ExprId,
    domain: &Domain<'_>,
    opts: &GateOptions,
    pool: &ExprPool,
) -> Verdict {
    let Ok(df) = crate::diff::diff(candidate, var, pool) else {
        return Verdict::Declined {
            reason: DeclineReason::Differentiation,
        };
    };
    let dfs = simplify(df.value, pool).value;

    // ── Tier 1: symbolic ───────────────────────────────────────────────────
    let residual = pool.add(vec![dfs, negate(target.integrand, pool)]);
    if opts.symbolic {
        let r = simplify(residual, pool).value;
        if is_syntactic_zero(r, pool) {
            return finalize(Verdict::Proven, opts);
        }
        #[cfg(feature = "egraph")]
        if opts.egraph {
            let r2 = crate::simplify::simplify_egraph(r, pool).value;
            if is_syntactic_zero(r2, pool) {
                return finalize(Verdict::Proven, opts);
            }
        }
    }

    // ── Tier 2: f64 screen ─────────────────────────────────────────────────
    let mut checked = 0usize;
    for &x in domain.samples() {
        if !domain.contains(x) {
            continue;
        }
        let Some(rhs) = target.eval(x, var, pool) else {
            continue;
        };
        let Some(lhs) = eval_at(dfs, var, x, pool) else {
            continue;
        };
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        let err = (lhs - rhs).abs();
        if err > opts.tolerance * (1.0 + rhs.abs()) {
            return Verdict::Failed {
                at: x,
                residual: err,
                tolerance: opts.tolerance,
            };
        }
        checked += 1;
    }
    if checked < opts.min_points {
        return Verdict::Declined {
            reason: DeclineReason::NotEnoughPoints {
                found: checked,
                required: opts.min_points,
            },
        };
    }
    let sampled = Verdict::SampledOnly {
        points: checked,
        tolerance: opts.tolerance,
    };

    // ── Tier 3: rigorous enclosure ─────────────────────────────────────────
    let budget = match opts.enclosure {
        EnclosurePolicy::Skip => return finalize(sampled, opts),
        EnclosurePolicy::BestEffort(b) | EnclosurePolicy::Required(b) => b,
    };
    let required = matches!(opts.enclosure, EnclosurePolicy::Required(_));
    // Re-simplify the residual for the bounding pass: `bound_on_box` walks the
    // expression tree, so a smaller tree is a cheaper (and usually tighter)
    // Taylor model.
    let resid = simplify(residual, pool).value;
    match enclose(resid, var, domain, &budget, pool) {
        Ok((boxes, residual_bound)) => finalize(
            Verdict::EnclosureVerified {
                boxes,
                residual_bound,
                points: checked,
                tolerance: opts.tolerance,
            },
            opts,
        ),
        Err(what) => {
            if required {
                Verdict::Declined {
                    reason: DeclineReason::EnclosureUnavailable { what },
                }
            } else {
                finalize(sampled, opts)
            }
        }
    }
}

/// Downgrade a passing verdict to a decline when it is below the caller's
/// floor.  Keeps the "never silently accept something weaker" rule in one
/// place.
fn finalize(v: Verdict, opts: &GateOptions) -> Verdict {
    match v.strength() {
        Some(s) if s >= opts.min_strength => v,
        Some(s) => Verdict::Declined {
            reason: DeclineReason::BelowRequiredStrength {
                reached: s,
                required: opts.min_strength,
            },
        },
        None => v,
    }
}

/// Rigorously bound `resid` on each of the domain's boxes.
///
/// Returns the covered boxes and the worst bound, or a description of why no
/// box could be covered / why a covered box exceeded the tolerance.
fn enclose(
    resid: ExprId,
    var: ExprId,
    domain: &Domain<'_>,
    budget: &EnclosureBudget,
    pool: &ExprPool,
) -> Result<(Vec<VerifiedBox>, f64), String> {
    use crate::validated::bounds::{bound_on_box, BoundOptions};

    if domain.boxes().is_empty() {
        return Err("the caller supplied no in-domain box".to_string());
    }
    let opts = BoundOptions {
        order: budget.order,
        prec: budget.prec,
        tol: budget.tol,
        max_subdivisions: budget.max_subdivisions,
    };
    // Cheap `f64` pre-screen.  A reduction is typically valid on *one*
    // component of its domain — outside it a `cos φ` substitution goes complex
    // and the residual is `NaN`.  Taylor-modelling such a box can only refuse,
    // after burning the whole subdivision budget getting there.  Skipping boxes
    // where the residual does not even evaluate is a pure cost optimisation: a
    // skipped box is simply not among the boxes the verdict claims.
    let screened: Vec<(f64, f64)> = domain
        .boxes()
        .iter()
        .copied()
        .filter(|&(lo, hi)| {
            [0.02_f64, 0.25, 0.5, 0.75, 0.98].iter().all(|f| {
                eval_at(resid, var, lo + (hi - lo) * f, pool).is_some_and(|v| v.is_finite())
            })
        })
        .collect();

    /// How many times a refused box may be halved before it is abandoned.
    const MAX_SPLIT_DEPTH: u32 = 3;
    /// A sub-box narrower than this is not worth splitting further.
    const MIN_SPLIT_WIDTH: f64 = 0.1;

    let mut queue: std::collections::VecDeque<(f64, f64, u32)> =
        screened.iter().map(|&(lo, hi)| (lo, hi, 0)).collect();
    let mut covered: Vec<VerifiedBox> = Vec::new();
    let mut worst = 0.0_f64;
    let mut attempts = 0usize;
    let mut last_refusal = String::from("no box survived the f64 pre-screen");

    while let Some((lo, hi, depth)) = queue.pop_front() {
        if attempts >= budget.max_boxes {
            break;
        }
        attempts += 1;
        // A refused or too-wide box is halved and retried.  Two things make
        // this worth doing rather than giving up: a Taylor remainder shrinks
        // like `O(h^{p+1})`, so a narrower box may clear a tolerance the whole
        // box cannot; and a candidate written with an interior *removable*
        // singularity (a `1/S(x)` whose `S` vanishes inside) can only ever be
        // bounded on boxes that avoid it.  Halves that still refuse are simply
        // not part of the coverage the verdict claims.
        let retry = |q: &mut std::collections::VecDeque<(f64, f64, u32)>| {
            if depth >= MAX_SPLIT_DEPTH || (hi - lo) < 2.0 * MIN_SPLIT_WIDTH {
                return;
            }
            let mid = 0.5 * (lo + hi);
            for (a, b) in [(lo, mid), (mid, hi)] {
                let evaluable = [0.02_f64, 0.5, 0.98].iter().all(|f| {
                    eval_at(resid, var, a + (b - a) * f, pool).is_some_and(|v| v.is_finite())
                });
                if evaluable {
                    q.push_back((a, b, depth + 1));
                }
            }
        };

        match bound_on_box(resid, pool, &[(var, lo, hi)], &opts) {
            Ok(r) => {
                let bound = r.lower().abs().max(r.upper().abs());
                if !bound.is_finite() {
                    last_refusal = format!("enclosure on [{lo}, {hi}] is not finite");
                    retry(&mut queue);
                    continue;
                }
                if bound > budget.tol {
                    // A bound that is real but too wide is *not* a refutation —
                    // Taylor models over-estimate, and a wide box over-estimates
                    // more.  Record it and try narrower; never report a failure.
                    last_refusal = format!(
                        "residual bound {bound:.3e} on [{lo}, {hi}] exceeds the \
                         tolerance {:.3e} (budget exhausted: {})",
                        budget.tol, r.budget_exhausted
                    );
                    retry(&mut queue);
                    continue;
                }
                worst = worst.max(bound);
                covered.push(VerifiedBox {
                    lo,
                    hi,
                    residual_bound: bound,
                });
            }
            Err(e) => {
                last_refusal = format!("[{lo}, {hi}]: {e}");
                retry(&mut queue);
            }
        }
    }
    if covered.is_empty() {
        return Err(last_refusal);
    }
    Ok((covered, worst))
}

/// `−expr`, pushed through a top-level sum.
///
/// `simplify` does not distribute a leading `−1` over an `Add`, so building the
/// residual as `d/dx F + (−1)·(a + b)` leaves `3x² + 2x + (−1)·(3x² + 2x)`
/// standing — a syntactic non-zero for an exact identity.  Negating term-wise
/// costs nothing and lets the symbolic tier close the common case.
fn negate(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Add(args) => pool.add(args.iter().map(|&a| negate(a, pool)).collect()),
        _ => pool.mul(vec![pool.integer(-1_i32), expr]),
    }
}

/// Is `expr` a syntactic zero (integer `0` or rational `0/1`)?
fn is_syntactic_zero(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Integer(n) => n.0 == 0,
        ExprData::Rational(r) => *r.0.numer() == 0,
        ExprData::Float(f) => f.inner.to_f64() == 0.0,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Numeric evaluation (registry-backed)
// ---------------------------------------------------------------------------

fn registry() -> &'static PrimitiveRegistry {
    static REGISTRY: OnceLock<PrimitiveRegistry> = OnceLock::new();
    REGISTRY.get_or_init(PrimitiveRegistry::dispatch_registry)
}

/// Evaluate `expr` at `var = x` in `f64`.
///
/// Function nodes are dispatched through the shared
/// [`crate::primitive::PrimitiveRegistry`], so the set of heads this accepts
/// is exactly the set the registry advertises — it is not a private list that
/// can silently drift away from what `diff` can produce.  Nodes with no
/// numeric rule (unregistered heads, unbound symbols) return `None`, which
/// makes the gate *skip* that sample point rather than pass it.
///
/// A `RootSum` **is** evaluated, by finding the roots of its minimal polynomial
/// numerically and summing the body over them in complex arithmetic — see
/// [`crate::eval`]'s `root_sum` module for the conditions under which that
/// declines.  It is an `f64` screen like the rest of this function, not a proof.
pub fn eval_at(expr: ExprId, var: ExprId, x: f64, pool: &ExprPool) -> Option<f64> {
    if expr == var {
        return Some(x);
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => Some(r.0.to_f64()),
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { .. } => None,
        ExprData::Add(args) => args
            .iter()
            .try_fold(0.0, |s, &a| Some(s + eval_at(a, var, x, pool)?)),
        ExprData::Mul(args) => args
            .iter()
            .try_fold(1.0, |s, &a| Some(s * eval_at(a, var, x, pool)?)),
        ExprData::Pow { base, exp } => {
            Some(eval_at(base, var, x, pool)?.powf(eval_at(exp, var, x, pool)?))
        }
        ExprData::Func { ref name, ref args } if !args.is_empty() => {
            let mut vals = Vec::with_capacity(args.len());
            for &a in args.iter() {
                vals.push(eval_at(a, var, x, pool)?);
            }
            registry().numeric_f64(name, &vals)
        }
        ExprData::RootSum {
            poly,
            var: rvar,
            body,
        } => {
            let env: std::collections::HashMap<ExprId, f64> = std::iter::once((var, x)).collect();
            crate::eval::eval_root_sum_f64(poly, rvar, body, &env, pool)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Fitting — the *propose* half.  Never a reason to emit.
// ---------------------------------------------------------------------------

/// Knobs for [`fit_blocks`].
#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    /// Reject the fit outright when the design cannot reproduce the target to
    /// this relative accuracy at every sample.  This is a *cost* filter, not a
    /// soundness one — the gate would reject a bad fit anyway; rejecting early
    /// keeps a cheap ansatz from masking a richer one.
    pub max_residual: f64,
    /// A fitted coefficient is replaced by a nearby simple rational only when
    /// the move is smaller than this (relative).
    pub snap_window: f64,
    /// Coefficients smaller than this in magnitude are dropped.
    pub drop_below: f64,
    /// Largest denominator considered when snapping.
    pub max_denominator: i64,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            max_residual: 1e-7,
            snap_window: 1e-10,
            drop_below: 1e-12,
            max_denominator: 60,
        }
    }
}

/// Least-squares fit of `c` so that `d/dx Σ cᵢ·blocksᵢ ≈ target` at `samples`.
///
/// Returns the (snapped) coefficients, or `None` when the design is
/// rank-deficient, not evaluable, or cannot reproduce the target to
/// `opts.max_residual`.
///
/// **This function can only propose.**  Its output is not evidence of
/// anything; run [`verify`] on the assembled candidate.
pub fn fit_blocks(
    blocks: &[ExprId],
    var: ExprId,
    samples: &[f64],
    target: &dyn Fn(f64) -> Option<f64>,
    opts: &FitOptions,
    pool: &ExprPool,
) -> Option<Vec<f64>> {
    let block_dx: Vec<ExprId> = blocks
        .iter()
        .map(|&blk| {
            crate::diff::diff(blk, var, pool)
                .ok()
                .map(|d| simplify(d.value, pool).value)
        })
        .collect::<Option<Vec<_>>>()?;
    let nblk = blocks.len();
    if nblk == 0 {
        return None;
    }

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for &xv in samples {
        let Some(yv) = target(xv) else { continue };
        if !yv.is_finite() {
            continue;
        }
        let mut row = Vec::with_capacity(nblk);
        let mut ok = true;
        for &dxi in &block_dx {
            match eval_at(dxi, var, xv, pool) {
                Some(v) if v.is_finite() => row.push(v),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            rows.push(row);
            ys.push(yv);
        }
    }
    if rows.len() < nblk + 1 {
        return None;
    }

    let coeffs = lstsq(&rows, &ys, nblk)?;

    let mut maxr = 0.0_f64;
    for (row, &y) in rows.iter().zip(&ys) {
        let pred: f64 = row.iter().zip(&coeffs).map(|(a, b)| a * b).sum();
        maxr = maxr.max((pred - y).abs() / (1.0 + y.abs()));
    }
    if !maxr.is_finite() || maxr > opts.max_residual {
        return None;
    }

    Some(
        coeffs
            .into_iter()
            .map(|c| {
                let snapped = snap_rational(c, opts.max_denominator);
                if (snapped - c).abs() < opts.snap_window * (1.0 + c.abs()) {
                    snapped
                } else {
                    c
                }
            })
            .collect(),
    )
}

/// Assemble `Σ cᵢ·blocksᵢ`, dropping coefficients below `opts.drop_below` and
/// rendering each surviving coefficient with `coeff_to_expr`.
pub fn assemble(
    blocks: &[ExprId],
    coeffs: &[f64],
    opts: &FitOptions,
    coeff_to_expr: &dyn Fn(f64, &ExprPool) -> ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let mut terms: Vec<ExprId> = Vec::new();
    for (&c, &blk) in coeffs.iter().zip(blocks) {
        if c.abs() < opts.drop_below {
            continue;
        }
        terms.push(pool.mul(vec![coeff_to_expr(c, pool), blk]));
    }
    if terms.is_empty() {
        return None;
    }
    Some(simplify(pool.add(terms), pool).value)
}

/// Minimal least-squares solver: normal equations `AᵀA c = Aᵀy` with Gaussian
/// elimination (partial pivoting).  `n` = number of unknowns.
pub fn lstsq(rows: &[Vec<f64>], ys: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return None;
    }
    let mut ata = vec![vec![0.0_f64; n]; n];
    let mut aty = vec![0.0_f64; n];
    for (row, &y) in rows.iter().zip(ys) {
        if row.len() != n {
            return None;
        }
        for i in 0..n {
            aty[i] += row[i] * y;
            for j in 0..n {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    for col in 0..n {
        let mut piv = col;
        let mut best = ata[col][col].abs();
        for (r, arow) in ata.iter().enumerate().take(n).skip(col + 1) {
            if arow[col].abs() > best {
                best = arow[col].abs();
                piv = r;
            }
        }
        if best < 1e-12 {
            return None; // singular / rank-deficient design
        }
        ata.swap(col, piv);
        aty.swap(col, piv);
        let d = ata[col][col];
        let pivot_row = ata[col].clone();
        let pivot_y = aty[col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = ata[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for (c, prc) in pivot_row.iter().enumerate().take(n).skip(col) {
                ata[r][c] -= f * prc;
            }
            aty[r] -= f * pivot_y;
        }
    }
    let mut out = vec![0.0; n];
    for (i, oi) in out.iter_mut().enumerate() {
        *oi = aty[i] / ata[i][i];
        if !oi.is_finite() {
            return None;
        }
    }
    Some(out)
}

/// Snap a fitted float to a nearby simple rational (denominators up to
/// `max_den`) and zero out numerical noise.  Cosmetic only: the gate guards
/// correctness either way.
pub fn snap_rational(v: f64, max_den: i64) -> f64 {
    if v.abs() < 1e-9 {
        return 0.0;
    }
    for den in 1..=max_den.max(1) {
        let num = (v * den as f64).round();
        let cand = num / den as f64;
        if (cand - v).abs() < 1e-9 * (1.0 + v.abs()) {
            return cand;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// A candidate that cleared the gate, with the verdict that cleared it.
#[derive(Clone, Debug)]
pub struct Accepted {
    /// The antiderivative to emit.
    pub antiderivative: ExprId,
    /// What was actually established about it.
    pub verdict: Verdict,
}

/// Run the gate over proposals in order and return the first that clears it.
///
/// This is the "several progressively richer ansätze, first gate-pass wins"
/// search, factored out.  A proposal that fails or declines costs CPU only.
pub fn verify_first<I>(
    proposals: I,
    target: &Target<'_>,
    var: ExprId,
    domain: &Domain<'_>,
    opts: &GateOptions,
    pool: &ExprPool,
) -> Option<Accepted>
where
    I: IntoIterator<Item = ExprId>,
{
    for candidate in proposals {
        let verdict = verify(candidate, target, var, domain, opts, pool);
        if verdict.is_verified() {
            return Some(Accepted {
                antiderivative: candidate,
                verdict,
            });
        }
    }
    None
}

/// The full pattern in one call: for each ansatz in `recipes`, fit the block
/// coefficients on `fit_samples`, assemble, and gate-verify; return the first
/// that clears the gate.
#[allow(clippy::too_many_arguments)]
pub fn propose_fit_verify(
    recipes: &[Vec<ExprId>],
    fit_samples: &[f64],
    fit_target: &dyn Fn(f64) -> Option<f64>,
    coeff_to_expr: &dyn Fn(f64, &ExprPool) -> ExprId,
    target: &Target<'_>,
    var: ExprId,
    domain: &Domain<'_>,
    fit_opts: &FitOptions,
    gate_opts: &GateOptions,
    pool: &ExprPool,
) -> Option<Accepted> {
    for blocks in recipes {
        let Some(coeffs) = fit_blocks(blocks, var, fit_samples, fit_target, fit_opts, pool) else {
            continue;
        };
        let Some(cand) = assemble(blocks, &coeffs, fit_opts, coeff_to_expr, pool) else {
            continue;
        };
        let verdict = verify(cand, target, var, domain, gate_opts, pool);
        if verdict.is_verified() {
            return Some(Accepted {
                antiderivative: cand,
                verdict,
            });
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain as SymDomain;

    fn pool_and_x() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", SymDomain::Real);
        (pool, x)
    }

    fn grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        (0..=n)
            .map(|i| lo + (hi - lo) * (i as f64) / (n as f64))
            .collect()
    }

    #[test]
    fn exact_identity_is_proven() {
        // F = x³/3, f = x².  simplify(d/dx F − f) = 0.
        let (pool, x) = pool_and_x();
        let third = pool.rational(rug::Integer::from(1), rug::Integer::from(3));
        let f_cand = pool.mul(vec![third, pool.pow(x, pool.integer(3_i32))]);
        let integrand = pool.pow(x, pool.integer(2_i32));
        let dom = Domain::from_samples(grid(-2.0, 2.0, 20));
        let v = verify(
            f_cand,
            &Target::symbolic(integrand),
            x,
            &dom,
            &GateOptions::default(),
            &pool,
        );
        assert_eq!(v, Verdict::Proven, "expected a symbolic proof, got {v:?}");
        assert_eq!(v.strength(), Some(Strength::Symbolic));
        assert!(v.meets(Strength::Enclosure));
    }

    #[test]
    fn a_wrong_candidate_is_refuted_not_declined() {
        let (pool, x) = pool_and_x();
        // F = x², f = x²  →  d/dx F = 2x ≠ x².
        let f_cand = pool.pow(x, pool.integer(2_i32));
        let integrand = pool.pow(x, pool.integer(2_i32));
        let dom = Domain::from_samples(grid(1.0, 3.0, 20));
        let v = verify(
            f_cand,
            &Target::symbolic(integrand),
            x,
            &dom,
            &GateOptions::sampled_only(),
            &pool,
        );
        assert!(matches!(v, Verdict::Failed { .. }), "got {v:?}");
        assert!(!v.is_verified());
    }

    #[test]
    fn enclosure_tier_upgrades_a_float_coefficient_candidate() {
        // F = 0.5·x² + 1e-12·x  — *not* a syntactic identity for f = x, but the
        // residual is a rigorous 1e-12 everywhere.  The enclosure tier must say
        // so over the whole box, which point sampling cannot.
        let (pool, x) = pool_and_x();
        let half = pool.rational(rug::Integer::from(1), rug::Integer::from(2));
        let tiny = pool.rational(
            rug::Integer::from(1),
            rug::Integer::from(1_000_000_000_000_i64),
        );
        let f_cand = pool.add(vec![
            pool.mul(vec![half, pool.pow(x, pool.integer(2_i32))]),
            pool.mul(vec![tiny, x]),
        ]);
        let dom = Domain::from_samples(grid(-2.0, 2.0, 20)).with_boxes(vec![(-2.0, 2.0)]);
        let opts = GateOptions::rigorous(EnclosureBudget::thorough());
        let v = verify(f_cand, &Target::symbolic(x), x, &dom, &opts, &pool);
        match &v {
            Verdict::EnclosureVerified {
                boxes,
                residual_bound,
                ..
            } => {
                assert_eq!(boxes.len(), 1);
                assert_eq!((boxes[0].lo, boxes[0].hi), (-2.0, 2.0));
                assert!(
                    *residual_bound < 1e-8 && *residual_bound > 0.0,
                    "bound {residual_bound:e}"
                );
            }
            other => panic!("expected an enclosure verdict, got {other:?}"),
        }
        assert_eq!(v.strength(), Some(Strength::Enclosure));
    }

    #[test]
    fn required_enclosure_declines_when_no_box_is_given() {
        let (pool, x) = pool_and_x();
        let half = pool.rational(rug::Integer::from(1), rug::Integer::from(2));
        let tiny = pool.rational(
            rug::Integer::from(1),
            rug::Integer::from(1_000_000_000_000_i64),
        );
        let f_cand = pool.add(vec![
            pool.mul(vec![half, pool.pow(x, pool.integer(2_i32))]),
            pool.mul(vec![tiny, x]),
        ]);
        let dom = Domain::from_samples(grid(-2.0, 2.0, 20)); // no boxes
        let opts = GateOptions::rigorous(EnclosureBudget::cheap());
        let v = verify(f_cand, &Target::symbolic(x), x, &dom, &opts, &pool);
        assert!(
            matches!(
                v,
                Verdict::Declined {
                    reason: DeclineReason::EnclosureUnavailable { .. }
                }
            ),
            "got {v:?}"
        );
    }

    #[test]
    fn domain_predicate_skips_out_of_domain_points() {
        // F = 2·√x, f = 1/√x.  Both are NaN for x < 0; a domain predicate keeps
        // those points from being counted, and the gate still finds enough.
        let (pool, x) = pool_and_x();
        let sqrt_x = pool.func("sqrt", vec![x]);
        let f_cand = pool.mul(vec![pool.integer(2_i32), sqrt_x]);
        let integrand = pool.pow(sqrt_x, pool.integer(-1_i32));
        let dom = Domain::from_samples(grid(-3.0, 3.0, 24)).with_predicate(|v: f64| v > 1e-6);
        let v = verify(
            f_cand,
            &Target::symbolic(integrand),
            x,
            &dom,
            &GateOptions::sampled_only(),
            &pool,
        );
        assert!(v.is_verified(), "got {v:?}");
    }

    #[test]
    fn too_few_evaluable_points_declines() {
        let (pool, x) = pool_and_x();
        let sqrt_x = pool.func("sqrt", vec![x]);
        let f_cand = pool.mul(vec![pool.integer(2_i32), sqrt_x]);
        let integrand = pool.pow(sqrt_x, pool.integer(-1_i32));
        // Only two in-domain points; min_points is 3.
        let dom = Domain::from_samples(vec![-1.0, 1.0, 2.0]).with_predicate(|v: f64| v > 1e-6);
        let v = verify(
            f_cand,
            &Target::symbolic(integrand),
            x,
            &dom,
            &GateOptions {
                symbolic: false,
                ..GateOptions::sampled_only()
            },
            &pool,
        );
        assert!(
            matches!(
                v,
                Verdict::Declined {
                    reason: DeclineReason::NotEnoughPoints { found: 2, .. }
                }
            ),
            "got {v:?}"
        );
    }

    #[test]
    fn min_strength_downgrades_a_sampled_pass_to_a_decline() {
        let (pool, x) = pool_and_x();
        let sqrt_x = pool.func("sqrt", vec![x]);
        let f_cand = pool.mul(vec![pool.integer(2_i32), sqrt_x]);
        let integrand = pool.pow(sqrt_x, pool.integer(-1_i32));
        let dom = Domain::from_samples(grid(0.5, 3.0, 20)).with_predicate(|v: f64| v > 1e-6);
        let opts = GateOptions {
            symbolic: false,
            enclosure: EnclosurePolicy::Skip,
            min_strength: Strength::Enclosure,
            ..GateOptions::default()
        };
        let v = verify(f_cand, &Target::symbolic(integrand), x, &dom, &opts, &pool);
        assert!(
            matches!(
                v,
                Verdict::Declined {
                    reason: DeclineReason::BelowRequiredStrength {
                        reached: Strength::Sampled,
                        required: Strength::Enclosure,
                    }
                }
            ),
            "got {v:?}"
        );
    }

    #[test]
    fn eval_at_dispatches_through_the_primitive_registry() {
        // `asin` and the two-argument `EllipticF` are both registry primitives
        // and both evaluate; an unregistered head does not.
        let (pool, x) = pool_and_x();
        let asin = pool.func("asin", vec![x]);
        assert!((eval_at(asin, x, 0.5, &pool).unwrap() - 0.5_f64.asin()).abs() < 1e-15);
        let m = pool.rational(rug::Integer::from(1), rug::Integer::from(2));
        let ef = pool.func("EllipticF", vec![x, m]);
        let v = eval_at(ef, x, std::f64::consts::FRAC_PI_4, &pool).expect("EllipticF evaluates");
        assert!((v - 0.826_017_876).abs() < 1e-6, "{v}");
        let bogus = pool.func("no_such_primitive", vec![x]);
        assert_eq!(eval_at(bogus, x, 1.0, &pool), None);
    }

    #[test]
    fn fit_then_verify_recovers_an_exact_ansatz() {
        // Target f = 3x² + 2x.  Ansatz blocks {x³, x², x}.  The fit must find
        // (1, 1, 0) and the gate must prove the assembled candidate.
        let (pool, x) = pool_and_x();
        let blocks = vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.pow(x, pool.integer(2_i32)),
            x,
        ];
        let integrand = pool.add(vec![
            pool.mul(vec![pool.integer(3_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.mul(vec![pool.integer(2_i32), x]),
        ]);
        let samples = grid(-2.0, 2.0, 40);
        let target_num = |v: f64| Some(3.0 * v * v + 2.0 * v);
        let to_expr = |c: f64, p: &ExprPool| -> ExprId {
            match rug::Rational::from_f64(c) {
                Some(r) => {
                    let (n, d) = r.into_numer_denom();
                    p.rational(n, d)
                }
                None => p.integer(0_i32),
            }
        };
        let got = propose_fit_verify(
            &[blocks],
            &samples,
            &target_num,
            &to_expr,
            &Target::symbolic(integrand),
            x,
            &Domain::from_samples(samples.clone()),
            &FitOptions::default(),
            &GateOptions::default(),
            &pool,
        )
        .expect("the ansatz spans the antiderivative");
        assert_eq!(got.verdict, Verdict::Proven);
        let s = pool.display(got.antiderivative).to_string();
        assert!(s.contains("x^3") && s.contains("x^2"), "{s}");
    }

    #[test]
    fn a_deficient_ansatz_declines_rather_than_emitting() {
        // Target f = 1/x, whose antiderivative is log x — not in the span of
        // {x, x²}.  The fit is poor, so nothing is proposed at all.
        let (pool, x) = pool_and_x();
        let blocks = vec![x, pool.pow(x, pool.integer(2_i32))];
        let integrand = pool.pow(x, pool.integer(-1_i32));
        let samples = grid(0.5, 3.0, 40);
        let target_num = |v: f64| Some(1.0 / v);
        let to_expr = |c: f64, p: &ExprPool| -> ExprId {
            match rug::Rational::from_f64(c) {
                Some(r) => {
                    let (n, d) = r.into_numer_denom();
                    p.rational(n, d)
                }
                None => p.integer(0_i32),
            }
        };
        let got = propose_fit_verify(
            &[blocks],
            &samples,
            &target_num,
            &to_expr,
            &Target::symbolic(integrand),
            x,
            &Domain::from_samples(samples.clone()),
            &FitOptions::default(),
            &GateOptions::default(),
            &pool,
        );
        assert!(got.is_none(), "a deficient ansatz must decline");
    }

    #[test]
    fn snap_rational_is_cosmetic_and_bounded() {
        assert_eq!(snap_rational(0.5000000000001, 60), 0.5);
        assert_eq!(snap_rational(1e-12, 60), 0.0);
        // √2 has no simple rational nearby, so it is left alone.
        let s = snap_rational(std::f64::consts::SQRT_2, 60);
        assert_eq!(s, std::f64::consts::SQRT_2);
    }
}
