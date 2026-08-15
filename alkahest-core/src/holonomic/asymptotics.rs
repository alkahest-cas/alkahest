//! Asymptotics of a P-recursive sequence, read off its recurrence (M5).
//!
//! A certified recurrence
//!
//! ```text
//! Σ_{i=0}^{J} p_i(n) · u(n+i) = 0
//! ```
//!
//! already determines how fast `u` grows. Poincaré's theorem says so: write
//! `D = max_i deg p_i`, let `a_i` be the coefficient of `n^D` in `p_i`, and call
//!
//! ```text
//! χ(t) = Σ_i a_i tⁱ
//! ```
//!
//! the **characteristic polynomial**. If `p_J(n) ≠ 0` for all large `n` and the
//! roots of `χ` have pairwise distinct moduli, then every solution is either
//! eventually zero or satisfies `u(n+1)/u(n) → ρ` for one of those roots.
//! Perron's refinement pins the polynomial correction: with `b_i` the
//! coefficient of `n^{D-1}` in `p_i` and `χ₁(t) = Σ_i b_i tⁱ`, substituting the
//! ansatz `u(n) ≈ C·ρⁿ·n^α` and killing the `1/n` term gives
//!
//! ```text
//! α = −χ₁(ρ) / (ρ · χ'(ρ)).
//! ```
//!
//! Both `ρ` and `α` are **derived** — they are functions of the recurrence and
//! of nothing else. `C` is not.
//!
//! # What is proved and what is fitted
//!
//! This module keeps the two apart structurally, because the constant is the
//! part that is usually hard and is exactly the part a research loop is tempted
//! to overclaim:
//!
//! * [`RecurrenceAsymptotics::characteristic`] — the growth rate, the
//!   polynomial exponent, the full root list with exact multiplicities, and the
//!   verdict on Poincaré's hypotheses. Derived from the recurrence.
//! * [`RecurrenceAsymptotics::connection`] — the connection constant `C`, and
//!   only that. It is determined by the initial conditions, not by the
//!   recurrence, so it is obtained **numerically** from the terms and reported
//!   as [`ConnectionConstant`], with the point it was fitted at, the point it
//!   was refit at, and how far it moved between them. This is the discipline
//!   [`crate::calculus::euler_maclaurin`] uses for its additive constant, for
//!   the same reason: γ does not come out of the boundary algebra, and `1/√π`
//!   does not come out of the recurrence.
//!
//! `C` is a *limit*, so it is only ever known to the accuracy the terms
//! support: it is fitted by extrapolating `u(N)/(ρᴺ·N^α)` to `N = ∞`, refit on
//! a second, smaller triple of points, and emitted only if the two agree.
//! [`AsymptoticReport::rigor`] is therefore always
//! [`Rigor::NumericallyConsistent`], never `ProvedUnderHypotheses`.
//!
//! # The hypotheses are stated, not assumed
//!
//! Poincaré–Perron is false without its hypotheses, and the interesting
//! sequences are the ones that break them. Each failure has its own verdict
//! ([`PerronVerdict`]) and is *reported* rather than silently papered over:
//!
//! * [`PerronVerdict::EqualModulusRoots`] — two or more roots of the same
//!   largest modulus. `u(n+2) = 4·u(n)` has roots `±2`; its solutions oscillate
//!   and no single `C·ρⁿ·n^α` describes them. Returning `ρ = 2` here would be a
//!   wrong answer with a confident face on it.
//! * [`PerronVerdict::RepeatedDominantRoot`] — the dominant root is repeated,
//!   so `χ'(ρ) = 0` and the exponent formula divides by zero. The true
//!   behaviour carries extra powers of `n` that this module does not compute.
//!   Multiplicity is **exact**: it is read off the squarefree decomposition of
//!   `χ` over `ℚ`, not from clustering numeric roots. It has to be — A359643's
//!   `χ = (t−1)³·(27t−283)` has a triple root that is *not* the dominant one,
//!   and a tolerance-based test that confused the two would refuse a case that
//!   works perfectly.
//! * [`PerronVerdict::DegenerateLeadingCoefficient`] — `deg p_J < D`, so
//!   `deg χ < J` and a characteristic root has escaped to infinity. This is
//!   outside Poincaré's theorem entirely (the sequence typically grows like
//!   `ρⁿ·n^{cn}`, which Birkhoff–Trjitzinsky handles and this module does not).
//! * [`PerronVerdict::EventuallyZero`] — the terms are zero from some index on.
//!   Every root is vacuously consistent with that and none of them is the
//!   answer.
//!
//! A leading coefficient that vanishes at *finitely many* `n` is not a verdict:
//! `p_J` is a polynomial, so its integer zeros are enumerated exactly and
//! reported in [`CharacteristicAnalysis::singular_indices`], and the theorem is
//! applied beyond the largest of them.
//!
//! # What the terms are used for
//!
//! Nothing in the characteristic analysis needs the sequence — pass no terms
//! and you still get `ρ`, `α`, the roots and the verdict. The terms buy two
//! things: the constant, and the *check* that the sequence really does follow
//! the dominant root. Poincaré's conclusion is that `u(n+1)/u(n)` tends to
//! *some* root; a sequence whose dominant component happens to vanish follows a
//! smaller one. `u(n+2) = 3u(n+1) − 2u(n)` with `u(0) = u(1) = 1` is the
//! constant sequence: dominant root `2`, actual growth `1ⁿ`. When terms are
//! supplied that is caught by the fit failing to converge and reported through
//! [`RecurrenceAsymptotics::follows_dominant_root`]; when they are not, it is
//! an explicitly *assumed* hypothesis.

use crate::calculus::asymptotic::AsymptoticError;
use crate::calculus::asymptotic_common::{
    as_rational_function, complex_roots, gate_accept, qp_add, qp_degree, qp_eval, qp_is_zero,
    qp_neg, qp_trim, rational_to_expr, verification_points, AsymptoticReport, Hypothesis, QPoly,
    Rigor, VerificationPoint, C64, DEFAULT_SLACK,
};
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::simplify;
use rug::{Float, Integer, Rational};

/// Relative separation two moduli must show before the larger counts as
/// *strictly* dominant.
///
/// The roots are located numerically, so this is a tolerance and not a proof —
/// which is why the corresponding hypothesis is reported as assumed. It matches
/// [`crate::calculus::singularity`]'s margin, which decides the same question
/// for generating-function poles.
const DOMINANCE_MARGIN: f64 = 1e-6;

/// Working precision for the logarithms used to fit the connection constant.
///
/// `u(1024)` for a growth rate around `34` is a 5000-bit integer, so the fit
/// runs in `MPFR` rather than `f64`: the quantity wanted is
/// `ln|u(N)| − N·ln ρ − α·ln N`, a difference of two numbers near 5000 that has
/// to be accurate to `1e-13`.
const FIT_PRECISION: u32 = 192;

/// Indices (relative to the first supplied term) the numeric gate scores at.
const GATE_OFFSETS: [i64; 4] = [80, 160, 320, 640];

/// Indices the connection constant is fitted at — deliberately disjoint from
/// [`GATE_OFFSETS`], for the reason spelled out in
/// [`crate::calculus::euler_maclaurin`]: a constant fitted at a point the gate
/// then scores makes the residual there zero by construction and the gate
/// vacuous.
const FIT_OFFSETS: [i64; 3] = [256, 512, 1024];

/// Where the constant is *refit* to check it is a constant at all. A limit is
/// the same number wherever it is extrapolated from; a mis-modelled shape is
/// not.
const REFIT_OFFSETS: [i64; 3] = [128, 256, 512];

/// How far the refit may move, relative to the fit.
///
/// Across the whole clean battery — Fibonacci, central binomials, Catalan,
/// Motzkin, Apéry, Franel, A359643 — the observed drift never exceeded
/// `5.5e-7`, and Fibonacci (where `C = 1/√5` exactly) came in at `2.6e-14`. A
/// sequence that does *not* follow the dominant root drifts by `1.0`: the two
/// extrapolations are not even the same order of magnitude. Six orders of
/// margin either way; `1e-3` sits in the middle.
const CONSTANT_DRIFT_TOL: f64 = 1e-3;

/// How much `ln|u(N)| − N·ln ρ − α·ln N` may move across the fit points before
/// the sequence is declared not to follow the root being tested.
///
/// This is the coarse test, and it runs in log space on purpose: a sequence
/// following a subdominant root sends the ratio to zero fast enough to
/// underflow `f64` (`2^-1024`), so the value that would diagnose the problem is
/// the value that cannot be computed. The observed spread is `≤ 7.2e-3` for
/// every sequence in the battery and `532` for the subdominant one.
const LOG_SPREAD_TOL: f64 = 0.25;

/// Cap on the size of an extended term, in bits. Extension stops here rather
/// than letting a pathological recurrence run the process out of memory.
const MAX_TERM_BITS: u32 = 1 << 22;

/// Largest integer whose divisors are enumerated when looking for exact
/// rational roots or for integer zeros of the leading coefficient.
const DIVISOR_SEARCH_CAP: i64 = 1 << 34;

// ---------------------------------------------------------------------------
// Result vocabulary
// ---------------------------------------------------------------------------

/// One root of the characteristic polynomial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacteristicRoot {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
    /// `|root|` — what decides dominance.
    pub modulus: f64,
    /// Multiplicity as a root of `χ`.
    ///
    /// **Exact.** It comes from the squarefree decomposition of `χ` over `ℚ`,
    /// not from clustering numeric roots, so a triple root reports `3` and not
    /// "three roots that happen to be close".
    pub multiplicity: usize,
}

/// Whether Poincaré–Perron delivers a single growth law, and if not, why not.
#[derive(Clone, Debug, PartialEq)]
pub enum PerronVerdict {
    /// One characteristic root of strictly largest modulus, simple and real.
    /// The growth rate and the polynomial exponent both follow.
    SingleDominantRoot,
    /// Two or more distinct roots share the largest modulus — the solutions
    /// oscillate and no single `C·ρⁿ·n^α` describes them.
    EqualModulusRoots {
        /// The shared modulus.
        modulus: f64,
        /// How many distinct roots share it.
        count: usize,
    },
    /// The dominant root is repeated, so `χ'(ρ) = 0` and the exponent formula
    /// does not apply.
    RepeatedDominantRoot {
        /// Its exact multiplicity.
        multiplicity: usize,
    },
    /// `deg χ < J`: the top-degree part of the leading coefficient vanishes, so
    /// a characteristic root has run off to infinity and Poincaré's theorem
    /// does not cover the recurrence.
    DegenerateLeadingCoefficient {
        /// `deg χ`.
        characteristic_degree: usize,
        /// The recurrence order `J`.
        order: usize,
    },
    /// The supplied terms are zero from this index on, so there is no growth
    /// law to state.
    EventuallyZero {
        /// First index of the all-zero tail.
        from: i64,
    },
}

impl PerronVerdict {
    /// Stable lower-case tag used by the Python bindings.
    pub fn tag(&self) -> &'static str {
        match self {
            PerronVerdict::SingleDominantRoot => "single_dominant_root",
            PerronVerdict::EqualModulusRoots { .. } => "equal_modulus_roots",
            PerronVerdict::RepeatedDominantRoot { .. } => "repeated_dominant_root",
            PerronVerdict::DegenerateLeadingCoefficient { .. } => "degenerate_leading_coefficient",
            PerronVerdict::EventuallyZero { .. } => "eventually_zero",
        }
    }

    /// Whether a single `C·ρⁿ·n^α` law is available at all.
    pub fn is_single_law(&self) -> bool {
        matches!(self, PerronVerdict::SingleDominantRoot)
    }

    /// One sentence saying what the verdict means for the caller.
    pub fn explanation(&self) -> String {
        match self {
            PerronVerdict::SingleDominantRoot => {
                "the characteristic polynomial has a unique root of largest modulus and it is \
                 simple and real, so Poincaré–Perron gives a single growth law"
                    .to_string()
            }
            PerronVerdict::EqualModulusRoots { modulus, count } => format!(
                "{count} distinct characteristic roots share the largest modulus {modulus}; the \
                 solutions carry an oscillating factor and no single power law describes them, \
                 so no growth rate is claimed"
            ),
            PerronVerdict::RepeatedDominantRoot { multiplicity } => format!(
                "the dominant characteristic root has multiplicity {multiplicity}, so χ'(ρ) = 0 \
                 and the polynomial exponent is not given by the simple formula; the true \
                 behaviour carries extra powers of n that are not computed here"
            ),
            PerronVerdict::DegenerateLeadingCoefficient {
                characteristic_degree,
                order,
            } => format!(
                "the leading coefficient p_{order}(n) has degree below the maximum over the \
                 coefficients, so the characteristic polynomial has degree \
                 {characteristic_degree} < {order} and a root has escaped to infinity; this is \
                 outside Poincaré's theorem and needs the full Birkhoff–Trjitzinsky theory"
            ),
            PerronVerdict::EventuallyZero { from } => format!(
                "the sequence is zero for every index from {from} on, so it has no growth rate"
            ),
        }
    }
}

/// The part of the answer that follows from the recurrence alone.
///
/// Everything here is a function of the coefficient polynomials and of nothing
/// else — no initial condition, no fitted number. The one qualification is that
/// the roots are located *numerically*, so the strict-modulus-separation test
/// that produced [`CharacteristicAnalysis::verdict`] is decided against a
/// relative tolerance; that is listed as an assumed hypothesis on the result.
#[derive(Clone, Debug)]
pub struct CharacteristicAnalysis {
    /// Recurrence order `J`.
    pub order: usize,
    /// `D = max_i deg p_i`.
    pub coefficient_degree: usize,
    /// `χ`, ascending: the coefficient of `n^D` in each `p_i`.
    pub characteristic: Vec<Rational>,
    /// `χ₁`, ascending: the coefficient of `n^{D−1}` in each `p_i`, empty when
    /// `D = 0`.
    pub subleading: Vec<Rational>,
    /// Every root of `χ`, modulus-descending, with exact multiplicities.
    pub roots: Vec<CharacteristicRoot>,
    /// Whether a single growth law is available, and if not, why not.
    pub verdict: PerronVerdict,
    /// `ρ` — the dominant characteristic root. `Some` exactly when the verdict
    /// is [`PerronVerdict::SingleDominantRoot`].
    pub growth_rate: Option<f64>,
    /// `ρ` exactly, when it is a rational number.
    pub growth_rate_exact: Option<Rational>,
    /// `α = −χ₁(ρ)/(ρ·χ'(ρ))`. `Some` exactly when `growth_rate` is.
    pub polynomial_exponent: Option<f64>,
    /// `α` exactly, available when `ρ` is rational (then so is `α`).
    pub polynomial_exponent_exact: Option<Rational>,
    /// Integer `n ≥ start` at which the leading coefficient `p_J(n)` vanishes.
    ///
    /// Poincaré's hypothesis is that this does not happen for large `n`; since
    /// `p_J` is a polynomial there are finitely many such `n` and the theorem
    /// applies beyond the largest.
    pub singular_indices: Vec<i64>,
    /// Whether that enumeration was exhaustive. `false` means the constant term
    /// of `p_J` was too large to factor within the search cap, so the list is a
    /// lower bound rather than the complete set.
    pub singular_indices_complete: bool,
}

/// The part of the answer that does **not** follow from the recurrence.
///
/// `C = lim_{n→∞} u(n)/(ρⁿ·n^α)` depends on the initial conditions, and for the
/// sequences this exists for it is usually the hard half of the result:
/// `1/√π` for the central binomial coefficients, `3√3/(2√π)` for Motzkin,
/// `(1+√2)²/(2^{9/4}π^{3/2})` for Apéry. None of those is recoverable from the
/// coefficient polynomials. This struct is what a caller checks before quoting
/// a constant as though it had been derived.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConnectionConstant {
    /// The fitted value.
    pub value: f64,
    /// Largest index used by the fit.
    pub fitted_at: i64,
    /// The value obtained by repeating the extrapolation on a smaller triple.
    pub refit_value: f64,
    /// Largest index used by that second fit.
    pub refit_at: i64,
    /// `|value − refit_value| / max(|value|, |refit_value|)`.
    pub relative_drift: f64,
    /// Whether the drift is inside the `1e-3` this module requires.
    ///
    /// A constant that did not converge is reported, not emitted: `false` here
    /// means the number in [`ConnectionConstant::value`] is evidence about the
    /// fit, not a result.
    pub converged: bool,
}

/// Asymptotics of a P-recursive sequence: what the recurrence proves, what the
/// terms fitted, and which hypotheses hold.
#[derive(Clone, Debug)]
pub struct RecurrenceAsymptotics {
    /// The asymptotic variable the result is written in.
    pub var: ExprId,
    /// Derived from the recurrence: roots, growth rate, polynomial exponent.
    pub characteristic: CharacteristicAnalysis,
    /// Fitted from the terms: the connection constant, or `None` when no terms
    /// were supplied, too few were supplied to run the recurrence forward, or
    /// the verdict left nothing to fit against.
    pub connection: Option<ConnectionConstant>,
    /// Whether the supplied terms were observed to follow the *dominant*
    /// characteristic root.
    ///
    /// `None` when no terms were supplied — the conclusion of Poincaré's
    /// theorem is that `u(n+1)/u(n)` tends to *some* root, and without terms
    /// there is no way to tell which. `Some(false)` is a real answer: the
    /// sequence's dominant component vanishes and it grows more slowly than the
    /// recurrence's generic solution.
    pub follows_dominant_root: Option<bool>,
    /// `C·ρⁿ·n^α` as an expression in [`RecurrenceAsymptotics::var`].
    ///
    /// `Some` only when the verdict is [`PerronVerdict::SingleDominantRoot`],
    /// the sequence was seen to follow the dominant root, the connection
    /// constant converged, **and** the result passed the numeric gate. The
    /// constant in it is fitted; see [`RecurrenceAsymptotics::connection`].
    pub leading_term: Option<ExprId>,
    /// Hypotheses of the method, each marked checked or assumed.
    pub hypotheses: Vec<Hypothesis>,
    /// Numeric corroboration of the fitted constant.
    ///
    /// `reference` is `u(N)/(ρᴺ·N^α)` and `approximation` is the fitted `C`,
    /// not the raw term against the raw expansion: `u(640)` overflows `f64` for
    /// every sequence this is interesting for. Dividing by the derived shape
    /// first is a *stronger* check, not a weaker one — a wrong `ρ` or `α` makes
    /// the reference diverge instead of settling.
    pub verification: Vec<VerificationPoint>,
    /// Ordered, human-readable derivation log.
    pub derivation: Vec<String>,
}

impl RecurrenceAsymptotics {
    /// The result as the asymptotics family's shared [`AsymptoticReport`].
    ///
    /// `terms` is empty exactly when [`RecurrenceAsymptotics::leading_term`] is
    /// `None`. `rigor` is always [`Rigor::NumericallyConsistent`]: the root
    /// separation is decided numerically and the connection constant is fitted,
    /// so nothing this module returns is proved outright.
    pub fn report(&self) -> AsymptoticReport {
        AsymptoticReport {
            method: "poincare-perron",
            var: self.var,
            terms: self.leading_term.into_iter().collect(),
            rigor: Rigor::NumericallyConsistent,
            hypotheses: self.hypotheses.clone(),
            verification: self.verification.clone(),
            derivation: self.derivation.clone(),
        }
    }

    /// The worst relative error observed by the numeric gate.
    pub fn max_relative_error(&self) -> Option<f64> {
        self.verification
            .iter()
            .map(|v| v.relative_error)
            .fold(None, |acc, e| Some(acc.map_or(e, |a: f64| a.max(e))))
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Asymptotic growth of the sequence satisfying `Σ_i coeffs[i](n)·u(n+i) = 0`.
///
/// `coeffs` are the coefficient polynomials `p_0 … p_J` in `n`, in that order;
/// they must be polynomials in `n` with rational coefficients. `terms` are the
/// exact leading terms of the sequence, `terms[0] = u(start)`; pass an empty
/// slice to get the characteristic analysis without a connection constant.
///
/// The result separates what the recurrence proves
/// ([`RecurrenceAsymptotics::characteristic`]) from what the terms fitted
/// ([`RecurrenceAsymptotics::connection`]). A degenerate case — equal-modulus
/// roots, a repeated dominant root, a degenerate leading coefficient, an
/// eventually-zero sequence — is *reported* through
/// [`CharacteristicAnalysis::verdict`], not refused and not papered over.
///
/// Refuses ([`AsymptoticError`]) only for malformed input: fewer than two
/// coefficient polynomials, a coefficient that is not a polynomial in `n` over
/// `ℚ`, or a characteristic polynomial whose every root is zero (there is no
/// growth law to state and nothing useful to report about it).
pub fn asymptotics_from_recurrence(
    coeffs: &[ExprId],
    n: ExprId,
    terms: &[Rational],
    start: i64,
    pool: &ExprPool,
) -> Result<RecurrenceAsymptotics, AsymptoticError> {
    // --- 1. The recurrence as exact polynomials over ℚ ---
    let mut polys = coefficient_polynomials(coeffs, n, pool)?;
    while polys.len() > 2 && qp_is_zero(polys.last().unwrap()) {
        polys.pop();
    }
    if polys.len() < 2 || qp_is_zero(polys.last().unwrap()) {
        return Err(AsymptoticError::InvalidTermCount);
    }
    let order = polys.len() - 1;
    let coefficient_degree = polys.iter().map(qp_degree).max().unwrap_or(0);

    let mut derivation = vec![format!(
        "recurrence of order {order} with coefficient polynomials of degree \
         at most {coefficient_degree}"
    )];

    // --- 2. χ and χ₁ ---
    let chi: QPoly = polys
        .iter()
        .map(|p| nth_coefficient(p, coefficient_degree))
        .collect();
    let subleading: Vec<Rational> = if coefficient_degree == 0 {
        Vec::new()
    } else {
        polys
            .iter()
            .map(|p| nth_coefficient(p, coefficient_degree - 1))
            .collect()
    };
    derivation.push(format!(
        "characteristic polynomial χ(t) = {}",
        display_poly(&chi)
    ));

    // --- 3. Integer zeros of the leading coefficient ---
    let (singular_indices, singular_indices_complete) = integer_zeros_from(&polys[order], start);
    if !singular_indices.is_empty() {
        derivation.push(format!(
            "the leading coefficient p_{order}(n) vanishes at n ∈ {singular_indices:?}; \
             Poincaré–Perron is applied beyond the largest of them"
        ));
    }

    // A degenerate leading coefficient is a verdict, not an error: the roots of
    // the (lower-degree) χ are still worth reporting.
    if chi[order] == 0 {
        let characteristic_degree = qp_degree(&qp_trim(chi.clone()));
        let roots = characteristic_roots(&chi).unwrap_or_default();
        derivation.push(format!(
            "deg p_{order} = {} < {coefficient_degree}, so deg χ = {characteristic_degree} < \
             {order}",
            qp_degree(&polys[order])
        ));
        let analysis = CharacteristicAnalysis {
            order,
            coefficient_degree,
            characteristic: chi,
            subleading,
            roots,
            verdict: PerronVerdict::DegenerateLeadingCoefficient {
                characteristic_degree,
                order,
            },
            growth_rate: None,
            growth_rate_exact: None,
            polynomial_exponent: None,
            polynomial_exponent_exact: None,
            singular_indices,
            singular_indices_complete,
        };
        return Ok(degenerate_result(n, analysis, start, derivation));
    }

    // --- 4. Roots, with exact multiplicities ---
    let roots = characteristic_roots(&chi).ok_or(AsymptoticError::UnsupportedScale)?;
    if roots.is_empty() || roots[0].modulus <= 0.0 {
        // Every root is zero: `χ = a_J·t^J`. There is no growth law and nothing
        // to hedge about, so this is a refusal rather than a verdict.
        return Err(AsymptoticError::UnsupportedScale);
    }
    derivation.push(format!(
        "roots of χ, modulus-descending: {}",
        display_roots(&roots)
    ));

    let dominant = roots[0];
    let shared = roots
        .iter()
        .filter(|r| (r.modulus - dominant.modulus).abs() <= DOMINANCE_MARGIN * dominant.modulus)
        .count();

    let mut verdict = PerronVerdict::SingleDominantRoot;
    if shared > 1 {
        verdict = PerronVerdict::EqualModulusRoots {
            modulus: dominant.modulus,
            count: shared,
        };
    } else if dominant.multiplicity > 1 {
        verdict = PerronVerdict::RepeatedDominantRoot {
            multiplicity: dominant.multiplicity,
        };
    } else if dominant.im.abs() > DOMINANCE_MARGIN * dominant.modulus {
        // Unreachable for a real χ — a complex root comes with its conjugate,
        // which the equal-modulus test above catches first — but a wrong answer
        // here would be a confident one, so it is guarded rather than argued.
        verdict = PerronVerdict::EqualModulusRoots {
            modulus: dominant.modulus,
            count: 2,
        };
    }

    if !verdict.is_single_law() {
        derivation.push(verdict.explanation());
        let analysis = CharacteristicAnalysis {
            order,
            coefficient_degree,
            characteristic: chi,
            subleading,
            roots,
            verdict,
            growth_rate: None,
            growth_rate_exact: None,
            polynomial_exponent: None,
            polynomial_exponent_exact: None,
            singular_indices,
            singular_indices_complete,
        };
        return Ok(degenerate_result(n, analysis, start, derivation));
    }

    // --- 5. Growth rate and polynomial exponent ---
    let rho = dominant.re;
    let rho_exact = exact_rational_root_near(&chi, rho);
    let (alpha, alpha_exact) = polynomial_exponent(&chi, &subleading, rho, rho_exact.as_ref())
        .ok_or(AsymptoticError::UnsupportedScale)?;
    derivation.push(match &rho_exact {
        Some(q) => format!("dominant root ρ = {q} (exact), simple"),
        None => format!("dominant root ρ ≈ {rho:.15} , simple"),
    });
    derivation.push(match &alpha_exact {
        Some(q) => format!("polynomial exponent α = −χ₁(ρ)/(ρ·χ'(ρ)) = {q} (exact)"),
        None => format!("polynomial exponent α = −χ₁(ρ)/(ρ·χ'(ρ)) ≈ {alpha:.15}"),
    });

    let mut analysis = CharacteristicAnalysis {
        order,
        coefficient_degree,
        characteristic: chi,
        subleading,
        roots,
        verdict,
        growth_rate: Some(rho),
        growth_rate_exact: rho_exact.clone(),
        polynomial_exponent: Some(alpha),
        polynomial_exponent_exact: alpha_exact.clone(),
        singular_indices,
        singular_indices_complete,
    };

    // --- 6. The connection constant, from the terms ---
    let base = start.max(0);
    let needed = base + FIT_OFFSETS[FIT_OFFSETS.len() - 1];
    let extended = extend_sequence(&polys, terms, start, needed);

    if let Some(from) = eventually_zero_from(&extended, start, order) {
        analysis.verdict = PerronVerdict::EventuallyZero { from };
        analysis.growth_rate = None;
        analysis.growth_rate_exact = None;
        analysis.polynomial_exponent = None;
        analysis.polynomial_exponent_exact = None;
        derivation.push(analysis.verdict.explanation());
        return Ok(degenerate_result(n, analysis, start, derivation));
    }

    // Whether the recurrence could actually be run out to the fitting index.
    // It can stop short: the leading coefficient vanishes at some integer, or
    // the terms outgrow the size cap. That is a different answer from "the
    // sequence does not follow the dominant root", and conflating the two would
    // put a false verdict on `follows_dominant_root`.
    let reached = extended
        .as_ref()
        .is_some_and(|u| start + u.len() as i64 > needed);

    let fit = extended.as_ref().filter(|_| reached).and_then(|u| {
        fit_connection_constant(
            u,
            start,
            base,
            rho,
            rho_exact.as_ref(),
            alpha,
            &mut derivation,
        )
    });

    let (connection, follows_dominant_root, gate) = match (&extended, reached, fit) {
        (None, _, _) => {
            derivation.push(format!(
                "no connection constant: {} initial terms were supplied and {order} are needed \
                 to run the recurrence forward",
                terms.len()
            ));
            (None, None, None)
        }
        (Some(u), false, _) => {
            derivation.push(format!(
                "no connection constant: the recurrence could only be run forward to n = {}, \
                 short of the n = {needed} the fit needs — the leading coefficient vanishes at \
                 an integer in range, or the terms outgrew the size cap",
                start + u.len() as i64 - 1
            ));
            (None, None, None)
        }
        (Some(_), true, None) => {
            derivation.push(
                "the sequence does not follow the dominant characteristic root: \
                 u(N)/(ρ^N·N^α) does not settle, so its dominant component is zero and no \
                 connection constant is claimed"
                    .to_string(),
            );
            (None, Some(false), None)
        }
        (Some(u), true, Some(fit)) => {
            let gate = fit
                .converged
                .then(|| gate_constant(u, start, base, rho, rho_exact.as_ref(), alpha, fit.value))
                .flatten();
            (Some(fit), Some(true), gate)
        }
    };

    let leading_term = match (&connection, &gate) {
        (Some(c), Some(_)) if c.converged => Some(build_leading_term(
            c.value,
            rho,
            rho_exact.as_ref(),
            alpha,
            alpha_exact.as_ref(),
            n,
            pool,
        )),
        _ => None,
    };
    if leading_term.is_some() {
        derivation.push(format!(
            "u(n) ~ C·ρⁿ·n^α with the derived ρ, α and the fitted C = {}",
            connection.as_ref().map_or(f64::NAN, |c| c.value)
        ));
    } else if connection.as_ref().is_some_and(|c| c.converged) {
        derivation.push(
            "the fitted constant did not survive the numeric gate, so no leading term is emitted"
                .to_string(),
        );
    }

    let hypotheses = hypotheses_for(&analysis, connection.as_ref(), follows_dominant_root, start);

    Ok(RecurrenceAsymptotics {
        var: n,
        characteristic: analysis,
        connection,
        follows_dominant_root,
        leading_term,
        hypotheses,
        verification: gate.unwrap_or_default(),
        derivation,
    })
}

/// A result carrying only the characteristic analysis — no growth law was
/// available, and the verdict says why.
fn degenerate_result(
    n: ExprId,
    analysis: CharacteristicAnalysis,
    start: i64,
    derivation: Vec<String>,
) -> RecurrenceAsymptotics {
    let hypotheses = hypotheses_for(&analysis, None, None, start);
    RecurrenceAsymptotics {
        var: n,
        characteristic: analysis,
        connection: None,
        follows_dominant_root: None,
        leading_term: None,
        hypotheses,
        verification: Vec::new(),
        derivation,
    }
}

/// The hypothesis list — the honest half of the result.
fn hypotheses_for(
    analysis: &CharacteristicAnalysis,
    connection: Option<&ConnectionConstant>,
    follows: Option<bool>,
    start: i64,
) -> Vec<Hypothesis> {
    let mut out = vec![Hypothesis::checked(
        "every coefficient of the recurrence is a polynomial in n with rational coefficients, \
         so the characteristic polynomial is exact",
    )];

    out.push(if !analysis.singular_indices_complete {
        Hypothesis::assumed(format!(
            "the leading coefficient p_{}(n) was not proved free of integer zeros: its \
             coefficients were too large to factor within the search cap",
            analysis.order
        ))
    } else if let Some(&last) = analysis.singular_indices.last() {
        Hypothesis::checked(format!(
            "the leading coefficient p_{}(n) is non-zero for every integer n > {last}; its \
             integer zeros were enumerated exactly and are {:?}",
            analysis.order, analysis.singular_indices
        ))
    } else {
        Hypothesis::checked(format!(
            "the leading coefficient p_{}(n) is non-zero for every integer n ≥ {start}; its \
             integer zeros were enumerated exactly and there are none",
            analysis.order
        ))
    });

    out.push(Hypothesis::checked(
        "the multiplicity of every characteristic root is exact — it is read off the squarefree \
         decomposition of χ over ℚ, not from clustering numeric roots",
    ));
    out.push(Hypothesis::assumed(format!(
        "the roots of χ were located numerically, so the separation of their moduli — which is \
         what decides whether Poincaré–Perron applies — is judged against a relative tolerance \
         of {DOMINANCE_MARGIN:e} rather than proved"
    )));

    if analysis.verdict.is_single_law() {
        out.push(Hypothesis::checked(
            "the growth rate ρ and the polynomial exponent α are derived from the recurrence by \
             Poincaré–Perron; neither of them was fitted",
        ));
    } else {
        out.push(Hypothesis::checked(analysis.verdict.explanation()));
    }

    match follows {
        Some(true) => out.push(Hypothesis::checked(
            "the sequence's component along the dominant characteristic root is non-zero: \
             u(N)/(ρ^N·N^α) was computed from the exact terms and settles",
        )),
        Some(false) => out.push(Hypothesis::checked(
            "the sequence's component along the dominant characteristic root is zero — it grows \
             more slowly than the recurrence's generic solution, so ρ is not its growth rate",
        )),
        None => out.push(Hypothesis::assumed(
            "the sequence's component along the dominant characteristic root is non-zero; \
             Poincaré's conclusion is only that u(n+1)/u(n) tends to *some* root, and this was \
             not checked — no terms were supplied, or the verdict left no dominant root to \
             check against",
        )),
    }

    match connection {
        Some(c) if c.converged => out.push(Hypothesis::assumed(format!(
            "the connection constant was fitted numerically from the exact terms, not derived; \
             extrapolating again from a smaller triple of indices moved it by {:.3e} relative, \
             within the {CONSTANT_DRIFT_TOL:e} required",
            c.relative_drift
        ))),
        Some(c) => out.push(Hypothesis::checked(format!(
            "no connection constant is claimed: the two extrapolations disagree by {:.3e} \
             relative, so the fit has not converged",
            c.relative_drift
        ))),
        None => out.push(Hypothesis::checked(
            "no numerically fitted connection constant is part of this result",
        )),
    }

    out
}

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

/// Each coefficient expression as an exact polynomial in `n` over `ℚ`.
fn coefficient_polynomials(
    coeffs: &[ExprId],
    n: ExprId,
    pool: &ExprPool,
) -> Result<Vec<QPoly>, AsymptoticError> {
    if coeffs.len() < 2 {
        return Err(AsymptoticError::InvalidTermCount);
    }
    let mut out = Vec::with_capacity(coeffs.len());
    for &c in coeffs {
        let rf = as_rational_function(c, n, pool).ok_or(AsymptoticError::UnsupportedScale)?;
        let den = qp_trim(rf.den.clone());
        if qp_degree(&den) != 0 || den[0] == 0 {
            // A genuinely rational coefficient can be cleared by multiplying
            // the whole recurrence through, but doing that silently would
            // change the object the caller handed over.
            return Err(AsymptoticError::UnsupportedScale);
        }
        let inv = Rational::from(1) / den[0].clone();
        out.push(qp_trim(
            rf.num.iter().map(|c| Rational::from(c * &inv)).collect(),
        ));
    }
    Ok(out)
}

/// Coefficient of `x^d` in `p`, zero past the end.
fn nth_coefficient(p: &QPoly, d: usize) -> Rational {
    p.get(d).cloned().unwrap_or_else(|| Rational::from(0))
}

// ---------------------------------------------------------------------------
// Exact polynomial helpers (the ones `asymptotic_common` does not already have)
// ---------------------------------------------------------------------------

fn qp_sub(a: &QPoly, b: &QPoly) -> QPoly {
    qp_add(a, &qp_neg(b))
}

fn qp_deriv(p: &QPoly) -> QPoly {
    if p.len() <= 1 {
        return vec![Rational::from(0)];
    }
    qp_trim(
        (1..p.len())
            .map(|i| Rational::from(&p[i] * &Rational::from(i as i64)))
            .collect(),
    )
}

fn qp_is_one(p: &QPoly) -> bool {
    let t = qp_trim(p.clone());
    t.len() == 1 && t[0] == 1
}

fn qp_monic(p: &QPoly) -> QPoly {
    let t = qp_trim(p.clone());
    if qp_is_zero(&t) {
        return t;
    }
    let lc = t.last().unwrap().clone();
    t.iter().map(|c| Rational::from(c / &lc)).collect()
}

/// Exact division with remainder over `ℚ`. `None` when the divisor is zero.
fn qp_divmod(a: &QPoly, b: &QPoly) -> Option<(QPoly, QPoly)> {
    let b = qp_trim(b.clone());
    if qp_is_zero(&b) {
        return None;
    }
    let mut r = qp_trim(a.clone());
    let bd = b.len() - 1;
    let blc = b.last().unwrap().clone();
    if qp_is_zero(&r) || r.len() <= bd {
        return Some((vec![Rational::from(0)], r));
    }
    let mut q = vec![Rational::from(0); r.len() - bd];
    while !qp_is_zero(&r) && r.len() > bd {
        let shift = r.len() - 1 - bd;
        let factor = Rational::from(r.last().unwrap() / &blc);
        q[shift] = factor.clone();
        for (i, bc) in b.iter().enumerate() {
            r[shift + i] -= Rational::from(&factor * bc);
        }
        r = qp_trim(r);
    }
    Some((qp_trim(q), r))
}

/// Monic gcd over `ℚ`.
fn qp_gcd(a: &QPoly, b: &QPoly) -> QPoly {
    let mut x = qp_trim(a.clone());
    let mut y = qp_trim(b.clone());
    while !qp_is_zero(&y) {
        let Some((_, r)) = qp_divmod(&x, &y) else {
            break;
        };
        x = y;
        y = qp_trim(r);
    }
    qp_monic(&x)
}

/// Yun's squarefree decomposition: returns `[f_1, f_2, …]` with
/// `f = lc · Π_i f_iⁱ`, each `f_i` squarefree and monic.
///
/// Multiplicity comes from here rather than from clustering numeric roots
/// because it has to be exact: A359643's characteristic polynomial is
/// `(t−1)³·(27t−283)`, whose triple root sits well away from the dominant one,
/// and a tolerance that merged them would refuse a case the theory handles.
fn squarefree_decomposition(f: &QPoly) -> Vec<QPoly> {
    let f = qp_monic(f);
    if qp_degree(&f) == 0 {
        return Vec::new();
    }
    let fp = qp_deriv(&f);
    let a0 = qp_gcd(&f, &fp);
    let Some((mut b, _)) = qp_divmod(&f, &a0) else {
        return Vec::new();
    };
    let Some((c, _)) = qp_divmod(&fp, &a0) else {
        return Vec::new();
    };
    let mut d = qp_sub(&c, &qp_deriv(&b));

    let mut out = Vec::new();
    // The loop peels one multiplicity level per pass, so `deg f` passes is a
    // hard bound; the counter is an invariant guard, not a heuristic cutoff.
    for _ in 0..=qp_degree(&f) {
        if qp_is_one(&b) {
            break;
        }
        let ai = qp_gcd(&b, &d);
        let Some((next_b, _)) = qp_divmod(&b, &ai) else {
            break;
        };
        let Some((next_c, _)) = qp_divmod(&d, &ai) else {
            break;
        };
        out.push(ai);
        b = next_b;
        d = qp_sub(&next_c, &qp_deriv(&b));
    }
    out
}

// ---------------------------------------------------------------------------
// Roots
// ---------------------------------------------------------------------------

/// Every root of `χ`, modulus-descending, with exact multiplicities.
fn characteristic_roots(chi: &QPoly) -> Option<Vec<CharacteristicRoot>> {
    let factors = squarefree_decomposition(chi);
    let mut out: Vec<CharacteristicRoot> = Vec::new();
    for (idx, f) in factors.iter().enumerate() {
        let multiplicity = idx + 1;
        if qp_degree(f) == 0 {
            continue;
        }
        let as_f64: Vec<f64> = f.iter().map(|c| c.to_f64()).collect();
        let roots = complex_roots(&as_f64)?;
        for z in roots {
            let z = polish(&as_f64, z);
            out.push(CharacteristicRoot {
                re: z.re,
                im: z.im,
                modulus: z.abs(),
                multiplicity,
            });
        }
    }
    out.sort_by(|a, b| {
        b.modulus
            .partial_cmp(&a.modulus)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(out)
}

/// A few Newton steps against the squarefree factor the root came from.
///
/// Durand–Kerner stops at a fixed step size; on a squarefree factor Newton is
/// quadratically convergent and costs nothing, and the fitted constant reads
/// `ρ` through `N·ln ρ` with `N = 1024`, so the last few digits of `ρ` are the
/// first few digits of `C`.
fn polish(coeffs: &[f64], mut z: C64) -> C64 {
    for _ in 0..8 {
        let mut val = C64::new(0.0, 0.0);
        let mut der = C64::new(0.0, 0.0);
        for &c in coeffs.iter().rev() {
            der = der.mul(z).add(val);
            val = val.mul(z).add(C64::new(c, 0.0));
        }
        if der.abs() < 1e-300 {
            break;
        }
        let step = val.div(der);
        z = z.sub(step);
        if step.abs() < 1e-18 * (1.0 + z.abs()) {
            break;
        }
    }
    z
}

/// `α = −χ₁(ρ)/(ρ·χ'(ρ))`, exactly when `ρ` is rational.
fn polynomial_exponent(
    chi: &QPoly,
    subleading: &QPoly,
    rho: f64,
    rho_exact: Option<&Rational>,
) -> Option<(f64, Option<Rational>)> {
    if let Some(q) = rho_exact {
        let dchi = qp_deriv(chi);
        let denom = q * qp_eval(&dchi, q);
        if denom != 0 {
            let num = if subleading.is_empty() {
                Rational::from(0)
            } else {
                qp_eval(subleading, q)
            };
            let alpha = -num / denom;
            let as_f64 = alpha.to_f64();
            return Some((as_f64, Some(alpha)));
        }
    }
    let chi_f: Vec<f64> = chi.iter().map(|c| c.to_f64()).collect();
    let dchi_f: Vec<f64> = qp_deriv(chi).iter().map(|c| c.to_f64()).collect();
    let denom = rho * horner(&dchi_f, rho);
    if denom == 0.0 || !denom.is_finite() {
        return None;
    }
    let num = if subleading.is_empty() {
        0.0
    } else {
        horner(
            &subleading.iter().map(|c| c.to_f64()).collect::<Vec<_>>(),
            rho,
        )
    };
    let _ = chi_f;
    let alpha = -num / denom;
    alpha.is_finite().then_some((alpha, None))
}

fn horner(p: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in p.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// The exact rational root of `chi` nearest `target`, if one exists there.
///
/// Rational-root theorem plus a divisor enumeration, verified by exact
/// evaluation — so a `Some` here is a fact about `χ`, not a rounding of the
/// numeric root. `None` means "not established": either there is no rational
/// root near `target` or the coefficients were too large to factor.
fn exact_rational_root_near(chi: &QPoly, target: f64) -> Option<Rational> {
    let chi = qp_trim(chi.clone());
    if qp_degree(&chi) == 0 {
        return None;
    }
    // Clear denominators to an integer polynomial with the same roots.
    let mut lcm = Integer::from(1);
    for c in &chi {
        lcm = lcm.lcm(c.denom());
    }
    let ints: Vec<Integer> = chi
        .iter()
        .map(|c| Integer::from(&lcm / c.denom()) * c.numer())
        .collect();
    // Strip the root at zero; it is never the dominant one.
    let first_nonzero = ints.iter().position(|c| *c != 0)?;
    let c0 = ints[first_nonzero].clone().abs().to_i64()?;
    let cd = ints.last()?.clone().abs().to_i64()?;
    if c0 == 0 || cd == 0 || c0 > DIVISOR_SEARCH_CAP || cd > DIVISOR_SEARCH_CAP {
        return None;
    }
    let scale = target.abs().max(1.0);
    let mut best: Option<(f64, Rational)> = None;
    for p in divisors(c0) {
        for q in divisors(cd) {
            for sign in [1i64, -1] {
                let cand = Rational::from((sign * p, q));
                let approx = cand.to_f64();
                let err = (approx - target).abs();
                if err > 1e-6 * scale {
                    continue;
                }
                if qp_eval(&chi, &cand) != 0 {
                    continue;
                }
                if best.as_ref().map_or(true, |(e, _)| err < *e) {
                    best = Some((err, cand));
                }
            }
        }
    }
    best.map(|(_, q)| q)
}

/// Integer `n ≥ start` at which `p` vanishes, and whether the search was
/// exhaustive.
fn integer_zeros_from(p: &QPoly, start: i64) -> (Vec<i64>, bool) {
    let p = qp_trim(p.clone());
    if qp_is_zero(&p) {
        return (Vec::new(), false);
    }
    if qp_degree(&p) == 0 {
        return (Vec::new(), true);
    }
    let mut lcm = Integer::from(1);
    for c in &p {
        lcm = lcm.lcm(c.denom());
    }
    let ints: Vec<Integer> = p
        .iter()
        .map(|c| Integer::from(&lcm / c.denom()) * c.numer())
        .collect();
    let mut out = Vec::new();
    // A zero constant term means `n = 0` is a root; divide it out before the
    // divisor enumeration, which needs a non-zero constant term.
    let first_nonzero = match ints.iter().position(|c| *c != 0) {
        Some(i) => i,
        None => return (Vec::new(), false),
    };
    if first_nonzero > 0 && start <= 0 {
        out.push(0);
    }
    let c0 = match ints[first_nonzero].clone().abs().to_i64() {
        Some(v) => v,
        None => return (out, false),
    };
    if c0 == 0 || c0 > DIVISOR_SEARCH_CAP {
        return (out, false);
    }
    for d in divisors(c0) {
        for sign in [1i64, -1] {
            let cand = sign * d;
            if cand < start {
                continue;
            }
            if qp_eval(&p, &Rational::from(cand)) == 0 && !out.contains(&cand) {
                out.push(cand);
            }
        }
    }
    out.sort_unstable();
    (out, true)
}

fn divisors(n: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let mut d = 1i64;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            out.push(d);
            if d != n / d {
                out.push(n / d);
            }
        }
        d += 1;
    }
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------
// The sequence itself
// ---------------------------------------------------------------------------

/// Run the recurrence forward, exactly, from the supplied terms.
///
/// Exact rational arithmetic and not floating point, deliberately: forward
/// iteration in `f64` is *attracted* to the dominant solution, so a sequence
/// whose dominant component is zero would acquire one from the rounding error
/// and the fit would then converge on a growth rate the sequence does not have.
/// That is the silent wrong answer this whole module is arranged to avoid.
///
/// `None` when there are fewer terms than the order. The returned vector may be
/// shorter than asked for — the leading coefficient can vanish, or the terms can
/// outgrow [`MAX_TERM_BITS`] — and the caller checks the length it got.
fn extend_sequence(
    polys: &[QPoly],
    terms: &[Rational],
    start: i64,
    upto: i64,
) -> Option<Vec<Rational>> {
    let order = polys.len() - 1;
    if terms.len() < order || order == 0 {
        return None;
    }
    let mut u: Vec<Rational> = terms.to_vec();
    while (start + u.len() as i64) <= upto {
        let idx = start + u.len() as i64 - order as i64;
        let at = Rational::from(idx);
        let lead = qp_eval(&polys[order], &at);
        if lead == 0 {
            break;
        }
        let mut acc = Rational::from(0);
        for (i, p) in polys.iter().enumerate().take(order) {
            acc += qp_eval(p, &at) * &u[u.len() - order + i];
        }
        let next = -acc / lead;
        if next.numer().significant_bits() > MAX_TERM_BITS
            || next.denom().significant_bits() > MAX_TERM_BITS
        {
            break;
        }
        u.push(next);
    }
    Some(u)
}

/// The index the sequence becomes identically zero at, if it does.
///
/// A tail of `order + 1` consecutive zeros pins every later term to zero
/// through the recurrence, so this is a statement about the whole sequence and
/// not only about the terms that were computed.
fn eventually_zero_from(u: &Option<Vec<Rational>>, start: i64, order: usize) -> Option<i64> {
    let u = u.as_ref()?;
    if u.len() < order + 1 {
        return None;
    }
    if !u[u.len() - order - 1..].iter().all(|v| *v == 0) {
        return None;
    }
    let first_zero_tail = u.iter().rposition(|v| *v != 0).map_or(0, |i| i + 1);
    Some(start + first_zero_tail as i64)
}

/// `ln|u(N)| − N·ln ρ − α·ln N`, and the sign of `u(N)`.
///
/// In log space because `u(640)` overflows `f64` for every sequence worth
/// asking about, and because a sequence following a *subdominant* root sends
/// the ratio below `f64`'s smallest denormal — so the number that diagnoses the
/// problem is precisely the one that cannot be represented.
fn log_ratio(
    u: &[Rational],
    start: i64,
    index: i64,
    ln_rho: &Float,
    alpha: f64,
) -> Option<(f64, i32)> {
    let offset = usize::try_from(index - start).ok()?;
    let v = u.get(offset)?;
    if *v == 0 {
        return None;
    }
    let sign = if *v < 0 { -1 } else { 1 };
    let magnitude = Float::with_val(FIT_PRECISION, v).abs();
    let mut r = magnitude.ln();
    r -= Float::with_val(FIT_PRECISION, index) * ln_rho;
    r -= Float::with_val(FIT_PRECISION, index).ln() * Float::with_val(FIT_PRECISION, alpha);
    let out = r.to_f64();
    out.is_finite().then_some((out, sign))
}

fn ln_of(rho: f64, rho_exact: Option<&Rational>) -> Float {
    match rho_exact {
        Some(q) => Float::with_val(FIT_PRECISION, q).abs().ln(),
        None => Float::with_val(FIT_PRECISION, rho).abs().ln(),
    }
}

/// `u(N)/(ρᴺ·N^α)` at the given indices, or `None` when the sequence does not
/// follow this root at all.
fn ratios_at(
    u: &[Rational],
    start: i64,
    base: i64,
    offsets: &[i64],
    ln_rho: &Float,
    alpha: f64,
) -> Option<Vec<(f64, f64)>> {
    let mut logs = Vec::with_capacity(offsets.len());
    for &o in offsets {
        logs.push(log_ratio(u, start, base + o, ln_rho, alpha)?);
    }
    let first_sign = logs[0].1;
    if logs.iter().any(|(_, s)| *s != first_sign) {
        return None;
    }
    let lo = logs.iter().map(|(l, _)| *l).fold(f64::INFINITY, f64::min);
    let hi = logs
        .iter()
        .map(|(l, _)| *l)
        .fold(f64::NEG_INFINITY, f64::max);
    if (hi - lo).abs() > LOG_SPREAD_TOL {
        return None;
    }
    Some(
        offsets
            .iter()
            .zip(logs)
            .map(|(&o, (l, s))| ((base + o) as f64, f64::from(s) * l.exp()))
            .collect(),
    )
}

/// Extrapolate `C(N) = u(N)/(ρᴺ·N^α)` to `N = ∞`.
///
/// `C(N) = C + d₁/N + d₂/N² + …`, so three indices determine `C` up to
/// `O(N⁻³)`. One Richardson step (two indices) is what
/// [`crate::calculus::singularity`] uses; two steps were measured to be worth
/// it here — the error against the known constants drops from `2e-6` to `8e-9`
/// for Catalan and from `1.7e-7` to `5.5e-11` for Apéry — because the terms are
/// exact, so there is no noise for the extra step to amplify.
fn extrapolate(points: &[(f64, f64)]) -> Option<f64> {
    let mut m = [[0.0f64; 4]; 3];
    for (i, &(n, c)) in points.iter().enumerate().take(3) {
        m[i] = [1.0, 1.0 / n, 1.0 / (n * n), c];
    }
    for i in 0..3 {
        let pivot = (i..3).max_by(|&a, &b| {
            m[a][i]
                .abs()
                .partial_cmp(&m[b][i].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        m.swap(i, pivot);
        if m[i][i].abs() < 1e-300 {
            return None;
        }
        let pivot_row = m[i];
        for (r, row) in m.iter_mut().enumerate() {
            if r == i {
                continue;
            }
            let f = row[i] / pivot_row[i];
            for (c, v) in row.iter_mut().enumerate() {
                *v -= f * pivot_row[c];
            }
        }
    }
    let c = m[0][3] / m[0][0];
    c.is_finite().then_some(c)
}

/// Fit the connection constant, and refit it to check it is one.
fn fit_connection_constant(
    u: &[Rational],
    start: i64,
    base: i64,
    rho: f64,
    rho_exact: Option<&Rational>,
    alpha: f64,
    derivation: &mut Vec<String>,
) -> Option<ConnectionConstant> {
    let ln_rho = ln_of(rho, rho_exact);
    let fit_points = ratios_at(u, start, base, &FIT_OFFSETS, &ln_rho, alpha)?;
    let refit_points = ratios_at(u, start, base, &REFIT_OFFSETS, &ln_rho, alpha)?;
    let value = extrapolate(&fit_points)?;
    let refit_value = extrapolate(&refit_points)?;
    let scale = value.abs().max(refit_value.abs());
    let relative_drift = if scale > 0.0 {
        (value - refit_value).abs() / scale
    } else {
        0.0
    };
    let converged = relative_drift <= CONSTANT_DRIFT_TOL;
    let fitted_at = base + FIT_OFFSETS[FIT_OFFSETS.len() - 1];
    let refit_at = base + REFIT_OFFSETS[REFIT_OFFSETS.len() - 1];
    derivation.push(format!(
        "connection constant fitted from the exact terms at N = {:?}: C = {value} \
         (refit at N = {:?} gave {refit_value}, a relative move of {relative_drift:.3e})",
        FIT_OFFSETS.map(|o| base + o),
        REFIT_OFFSETS.map(|o| base + o),
    ));
    Some(ConnectionConstant {
        value,
        fitted_at,
        refit_value,
        refit_at,
        relative_drift,
        converged,
    })
}

/// Score `C` against `u(N)/(ρᴺ·N^α)` at indices the fit never saw.
#[allow(clippy::too_many_arguments)]
fn gate_constant(
    u: &[Rational],
    start: i64,
    base: i64,
    rho: f64,
    rho_exact: Option<&Rational>,
    alpha: f64,
    constant: f64,
) -> Option<Vec<VerificationPoint>> {
    let ln_rho = ln_of(rho, rho_exact);
    let points = ratios_at(u, start, base, &GATE_OFFSETS, &ln_rho, alpha)?;
    let at: Vec<f64> = points.iter().map(|(n, _)| *n).collect();
    let oracle: Vec<f64> = points.iter().map(|(_, c)| *c).collect();
    let term_vals = vec![vec![constant; oracle.len()]];
    let accepted = gate_accept(&oracle, &term_vals, DEFAULT_SLACK);
    if accepted == 0 {
        return None;
    }
    Some(verification_points(&at, &oracle, &term_vals, accepted))
}

/// `C·ρⁿ·n^α` as an expression, exact wherever the quantity is exact.
fn build_leading_term(
    constant: f64,
    rho: f64,
    rho_exact: Option<&Rational>,
    alpha: f64,
    alpha_exact: Option<&Rational>,
    n: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let mut factors = vec![float_to_expr(constant, pool)];
    let rho_expr = match rho_exact {
        Some(q) => rational_to_expr(q, pool),
        None => float_to_expr(rho, pool),
    };
    factors.push(pool.pow(rho_expr, n));
    let alpha_is_zero = alpha_exact.map_or(alpha == 0.0, |q| *q == 0);
    if !alpha_is_zero {
        let alpha_expr = match alpha_exact {
            Some(q) => rational_to_expr(q, pool),
            None => float_to_expr(alpha, pool),
        };
        factors.push(pool.pow(n, alpha_expr));
    }
    simplify(pool.mul(factors), pool).value
}

/// A float as a rational literal, rounded to a manageable denominator.
///
/// The connection constant is empirical and meaningful to at most a dozen
/// digits, so an exact binary fraction with a `2^52` denominator would be
/// noise dressed as precision.
fn float_to_expr(v: f64, pool: &ExprPool) -> ExprId {
    match Rational::from_f64(v) {
        Some(q) => {
            let scale = Integer::from(1_000_000_000_000_i64);
            let scaled = (q * Rational::from(scale.clone())).round();
            rational_to_expr(&Rational::from((scaled.numer().clone(), scale)), pool)
        }
        None => pool.integer(0_i32),
    }
}

// ---------------------------------------------------------------------------
// Display helpers for the derivation log
// ---------------------------------------------------------------------------

fn display_poly(p: &QPoly) -> String {
    let mut out = String::new();
    for (i, c) in p.iter().enumerate().rev() {
        if *c == 0 {
            continue;
        }
        let negative = *c < 0;
        let magnitude = c.clone().abs();
        if out.is_empty() {
            if negative {
                out.push('-');
            }
        } else {
            out.push_str(if negative { " - " } else { " + " });
        }
        out.push_str(&match i {
            0 => format!("{magnitude}"),
            1 if magnitude == 1 => "t".to_string(),
            1 => format!("{magnitude}·t"),
            _ if magnitude == 1 => format!("t^{i}"),
            _ => format!("{magnitude}·t^{i}"),
        });
    }
    if out.is_empty() {
        "0".to_string()
    } else {
        out
    }
}

fn display_roots(roots: &[CharacteristicRoot]) -> String {
    roots
        .iter()
        .map(|r| {
            let base = if r.im.abs() < 1e-12 {
                format!("{:.12}", r.re)
            } else {
                format!("{:.12}{:+.12}i", r.re, r.im)
            };
            if r.multiplicity > 1 {
                format!("{base} (multiplicity {})", r.multiplicity)
            } else {
                base
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculus::asymptotic_common::HypothesisStatus;
    use crate::kernel::Domain;

    /// Build `[p_0, …, p_J]` from ascending integer coefficient lists.
    fn recurrence(pool: &ExprPool, n: ExprId, polys: &[&[i64]]) -> Vec<ExprId> {
        polys
            .iter()
            .map(|coeffs| {
                let mut terms = Vec::new();
                for (i, &c) in coeffs.iter().enumerate() {
                    if c == 0 {
                        continue;
                    }
                    let lit = pool.integer(c);
                    terms.push(if i == 0 {
                        lit
                    } else {
                        pool.mul(vec![lit, pool.pow(n, pool.integer(i as i32))])
                    });
                }
                if terms.is_empty() {
                    pool.integer(0_i32)
                } else {
                    simplify(pool.add(terms), pool).value
                }
            })
            .collect()
    }

    fn env() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        (pool, n)
    }

    fn ints(vs: &[i64]) -> Vec<Rational> {
        vs.iter().map(|&v| Rational::from(v)).collect()
    }

    fn relative(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// `F(n+2) = F(n+1) + F(n)` — the control case, because the connection
    /// constant is *derivable* here (`1/√5`) and so the fit has a known target.
    #[test]
    fn fibonacci_growth_and_constant() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[-1], &[-1], &[1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[0, 1]), 0, &pool).expect("analysis");

        assert_eq!(r.characteristic.verdict, PerronVerdict::SingleDominantRoot);
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!(relative(r.characteristic.growth_rate.unwrap(), phi) < 1e-12);
        assert_eq!(r.characteristic.polynomial_exponent.unwrap(), 0.0);
        assert_eq!(r.follows_dominant_root, Some(true));

        let c = r.connection.expect("constant");
        assert!(c.converged);
        assert!(
            relative(c.value, 1.0 / 5.0_f64.sqrt()) < 1e-10,
            "fitted C = {} should be 1/sqrt(5) = {}",
            c.value,
            1.0 / 5.0_f64.sqrt()
        );
        assert!(r.leading_term.is_some());
    }

    /// `(n+1)·u(n+1) = (4n+2)·u(n)` — the central binomial coefficients,
    /// `C(2n,n) ~ 4ⁿ/√(πn)`. Both `ρ = 4` and `α = −1/2` come out exact.
    #[test]
    fn central_binomials_are_four_to_the_n_over_root_pi_n() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[-2, -4], &[1, 1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1]), 0, &pool).expect("analysis");

        assert_eq!(r.characteristic.growth_rate_exact, Some(Rational::from(4)));
        assert_eq!(
            r.characteristic.polynomial_exponent_exact,
            Some(Rational::from((-1, 2)))
        );
        let c = r.connection.expect("constant");
        assert!(c.converged);
        assert!(
            relative(c.value, 1.0 / std::f64::consts::PI.sqrt()) < 1e-8,
            "fitted C = {}",
            c.value
        );
        assert!(r.max_relative_error().unwrap() < 1e-2);
    }

    /// Catalan: same `ρ = 4`, but `α = −3/2` — the exponent, not the rate, is
    /// what separates them, and it comes from `χ₁` rather than from `χ`.
    #[test]
    fn catalan_has_the_same_rate_but_a_different_exponent() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[-2, -4], &[2, 1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1]), 0, &pool).expect("analysis");

        assert_eq!(r.characteristic.growth_rate_exact, Some(Rational::from(4)));
        assert_eq!(
            r.characteristic.polynomial_exponent_exact,
            Some(Rational::from((-3, 2)))
        );
        let c = r.connection.expect("constant");
        assert!(relative(c.value, 1.0 / std::f64::consts::PI.sqrt()) < 1e-6);
    }

    /// Motzkin: `M(n) ~ 3ⁿ·3√3/(2√π·n^{3/2})`.
    #[test]
    fn motzkin() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[-3, -3], &[-5, -2], &[4, 1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1, 1]), 0, &pool).expect("analysis");

        assert_eq!(r.characteristic.growth_rate_exact, Some(Rational::from(3)));
        assert_eq!(
            r.characteristic.polynomial_exponent_exact,
            Some(Rational::from((-3, 2)))
        );
        let truth = 3.0 * 3.0_f64.sqrt() / (2.0 * std::f64::consts::PI.sqrt());
        assert!(relative(r.connection.unwrap().value, truth) < 1e-6);
    }

    /// Apéry numbers A005259: `ρ = (1+√2)⁴ = 17 + 12√2`, `α = −3/2`, and the
    /// constant `(1+√2)²/(2^{9/4}·π^{3/2})`. The rate is irrational, so this is
    /// the case where `growth_rate_exact` is `None` and the whole fit runs off
    /// a numerically located root.
    #[test]
    fn apery_numbers() {
        let (pool, n) = env();
        let rec = recurrence(
            &pool,
            n,
            &[&[1, 3, 3, 1], &[-117, -231, -153, -34], &[8, 12, 6, 1]],
        );
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1, 5]), 0, &pool).expect("analysis");

        let rho = 17.0 + 12.0 * 2.0_f64.sqrt();
        assert!(relative(r.characteristic.growth_rate.unwrap(), rho) < 1e-12);
        assert!(r.characteristic.growth_rate_exact.is_none());
        assert!(relative(r.characteristic.polynomial_exponent.unwrap(), -1.5) < 1e-10);

        let s2 = 2.0_f64.sqrt();
        let truth = (1.0 + s2).powi(2) / (2.0_f64.powf(2.25) * std::f64::consts::PI.powf(1.5));
        assert!(
            relative(r.connection.unwrap().value, truth) < 1e-7,
            "Apéry constant"
        );
    }

    /// OEIS A359643, `a(n) = Σ_k C(n,k)·C(4k,k)`, whose entry records
    /// `a(n) ~ 283^(n+1/2) / (2^{7/2}·√(πn)·3^{3n+1/2})`.
    ///
    /// That is `ρ = 283/27`, `α = −1/2` and `C = √(283/3)/(2^{7/2}√π)`; the
    /// order-4 recurrence below is the one this project certified. Note that
    /// `χ = (t−1)³·(27t−283)` — the triple root is real and is *not* the
    /// dominant one, which is exactly why multiplicity has to be exact rather
    /// than a tolerance.
    #[test]
    fn a359643_matches_its_oeis_asymptotic() {
        let (pool, n) = env();
        let rec = recurrence(
            &pool,
            n,
            &[
                &[1698, 3113, 1698, 283],
                &[-12978, -16071, -6543, -876],
                &[24624, 24705, 8289, 930],
                &[-14688, -12833, -3741, -364],
                &[1320, 1086, 297, 27],
            ],
        );
        // a(0..3).
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1, 5, 37, 317]), 0, &pool)
            .expect("analysis");

        assert_eq!(r.characteristic.verdict, PerronVerdict::SingleDominantRoot);
        assert_eq!(
            r.characteristic.growth_rate_exact,
            Some(Rational::from((283, 27)))
        );
        assert_eq!(
            r.characteristic.polynomial_exponent_exact,
            Some(Rational::from((-1, 2)))
        );
        // The triple root at t = 1 is present and correctly identified.
        assert!(r
            .characteristic
            .roots
            .iter()
            .any(|x| (x.re - 1.0).abs() < 1e-9 && x.multiplicity == 3));

        let truth = (283.0f64 / 3.0).sqrt() / (2.0_f64.powf(3.5) * std::f64::consts::PI.sqrt());
        let c = r.connection.expect("constant");
        assert!(
            relative(c.value, truth) < 1e-8,
            "fitted C = {} vs OEIS {truth}",
            c.value
        );
        assert!(r.leading_term.is_some());
    }

    /// `u(n+2) = 4·u(n)` has characteristic roots `±2`. Reporting `ρ = 2` would
    /// be a wrong answer with a confident face on it, so the verdict says so
    /// and no growth rate is offered.
    #[test]
    fn equal_modulus_roots_are_reported_not_guessed() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[-4], &[0], &[1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1, 2]), 0, &pool).expect("analysis");

        match r.characteristic.verdict {
            PerronVerdict::EqualModulusRoots { modulus, count } => {
                assert!((modulus - 2.0).abs() < 1e-9);
                assert_eq!(count, 2);
            }
            other => panic!("expected equal-modulus verdict, got {other:?}"),
        }
        assert!(r.characteristic.growth_rate.is_none());
        assert!(r.characteristic.polynomial_exponent.is_none());
        assert!(r.leading_term.is_none());
        assert!(r.connection.is_none());
    }

    /// A complex conjugate pair of largest modulus is the same failure wearing
    /// a different hat: `u(n+2) = −u(n)` has roots `±i`.
    #[test]
    fn complex_dominant_pair_is_equal_modulus() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[1], &[0], &[1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1, 1]), 0, &pool).expect("analysis");
        assert!(matches!(
            r.characteristic.verdict,
            PerronVerdict::EqualModulusRoots { .. }
        ));
    }

    /// `χ = (t−2)²` — the exponent formula would divide by `χ'(ρ) = 0`.
    #[test]
    fn repeated_dominant_root_is_reported() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[4], &[-4], &[1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1, 2]), 0, &pool).expect("analysis");

        assert_eq!(
            r.characteristic.verdict,
            PerronVerdict::RepeatedDominantRoot { multiplicity: 2 }
        );
        assert!(r.characteristic.growth_rate.is_none());
        assert!(r.leading_term.is_none());
    }

    /// `deg p_J < D` puts a characteristic root at infinity and the recurrence
    /// outside Poincaré's theorem: `u(n+2) = n·u(n+1)` grows like `n!`.
    #[test]
    fn degenerate_leading_coefficient_is_reported() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[0], &[0, -1], &[1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1, 1]), 0, &pool).expect("analysis");

        assert!(matches!(
            r.characteristic.verdict,
            PerronVerdict::DegenerateLeadingCoefficient {
                characteristic_degree: 1,
                order: 2
            }
        ));
        assert!(r.characteristic.growth_rate.is_none());
    }

    /// The leading coefficient vanishing at finitely many `n` is a *reported*
    /// side condition, not a refusal: `(n−7)·u(n+1) = 4·(n−7)·u(n)` still has
    /// `ρ = 4`.
    #[test]
    fn finitely_many_singular_indices_are_enumerated() {
        let (pool, n) = env();
        // p_1(n) = n − 7, p_0(n) = −4n + 28.
        let rec = recurrence(&pool, n, &[&[28, -4], &[-7, 1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1]), 0, &pool).expect("analysis");

        assert_eq!(r.characteristic.singular_indices, vec![7]);
        assert!(r.characteristic.singular_indices_complete);
        assert_eq!(r.characteristic.growth_rate_exact, Some(Rational::from(4)));
        assert!(r
            .hypotheses
            .iter()
            .any(|h| h.statement.contains("non-zero for every integer n > 7")));

        // The forward run stops dead at n = 7, so there is nothing to fit. That
        // is *not* the same finding as "the sequence does not follow the
        // dominant root", and must not be reported as one.
        assert!(r.connection.is_none());
        assert_eq!(r.follows_dominant_root, None);
        assert!(r
            .derivation
            .iter()
            .any(|d| d.contains("could only be run forward")));
    }

    /// An eventually-zero sequence has no growth rate, and every root is
    /// vacuously consistent with it.
    #[test]
    fn eventually_zero_sequence_is_reported() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[-1], &[-1], &[1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[0, 0]), 0, &pool).expect("analysis");

        assert!(matches!(
            r.characteristic.verdict,
            PerronVerdict::EventuallyZero { .. }
        ));
        assert!(r.characteristic.growth_rate.is_none());
        assert!(r.leading_term.is_none());
    }

    /// Poincaré's conclusion is that `u(n+1)/u(n)` tends to *some* root.
    /// `u(n+2) = 3u(n+1) − 2u(n)` with `u(0) = u(1) = 1` is the constant
    /// sequence: the dominant root is `2` and the sequence's component along it
    /// is zero.
    #[test]
    fn a_sequence_that_does_not_follow_the_dominant_root_is_caught() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[2], &[-3], &[1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1, 1]), 0, &pool).expect("analysis");

        assert_eq!(r.characteristic.growth_rate_exact, Some(Rational::from(2)));
        assert_eq!(r.follows_dominant_root, Some(false));
        assert!(r.connection.is_none());
        assert!(r.leading_term.is_none());
    }

    /// With no terms there is no constant and no way to check which root the
    /// sequence follows — and the report says both, rather than assuming.
    #[test]
    fn no_terms_gives_the_shape_and_an_assumed_hypothesis() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[-2, -4], &[1, 1]]);
        let r = asymptotics_from_recurrence(&rec, n, &[], 0, &pool).expect("analysis");

        assert_eq!(r.characteristic.growth_rate_exact, Some(Rational::from(4)));
        assert_eq!(
            r.characteristic.polynomial_exponent_exact,
            Some(Rational::from((-1, 2)))
        );
        assert!(r.connection.is_none());
        assert_eq!(r.follows_dominant_root, None);
        assert!(r.leading_term.is_none());
        assert!(r
            .hypotheses
            .iter()
            .any(|h| h.status == HypothesisStatus::Assumed
                && h.statement.contains("tends to *some* root")));
    }

    /// The fitted constant must never be presented as a derived one.
    #[test]
    fn the_constant_is_labelled_fitted_and_the_exponent_derived() {
        let (pool, n) = env();
        let rec = recurrence(&pool, n, &[&[-2, -4], &[1, 1]]);
        let r = asymptotics_from_recurrence(&rec, n, &ints(&[1]), 0, &pool).expect("analysis");

        let report = r.report();
        assert_eq!(report.method, "poincare-perron");
        assert_eq!(report.rigor, Rigor::NumericallyConsistent);
        assert!(!report.all_hypotheses_checked());
        assert!(r.hypotheses.iter().any(|h| {
            h.status == HypothesisStatus::Assumed && h.statement.contains("fitted numerically")
        }));
        assert!(r.hypotheses.iter().any(|h| {
            h.status == HypothesisStatus::Checked
                && h.statement.contains("neither of them was fitted")
        }));
        assert_eq!(report.terms.len(), 1);
    }

    /// A coefficient that is not a polynomial in `n` is refused rather than
    /// approximated.
    #[test]
    fn refuses_a_non_polynomial_coefficient() {
        let (pool, n) = env();
        let bad = pool.func("exp", vec![n]);
        let one = pool.integer(1_i32);
        let err =
            asymptotics_from_recurrence(&[bad, one], n, &[], 0, &pool).expect_err("must refuse");
        assert!(matches!(err, AsymptoticError::UnsupportedScale));
    }

    #[test]
    fn refuses_a_recurrence_of_order_zero() {
        let (pool, n) = env();
        let one = pool.integer(1_i32);
        assert!(asymptotics_from_recurrence(&[one], n, &[], 0, &pool).is_err());
    }

    #[test]
    fn squarefree_decomposition_recovers_multiplicities() {
        // (t − 1)³·(27t − 283), A359643's characteristic polynomial.
        let chi: QPoly = [283, -876, 930, -364, 27]
            .iter()
            .map(|&c| Rational::from(c))
            .collect();
        let factors = squarefree_decomposition(&chi);
        assert_eq!(factors.len(), 3);
        assert_eq!(qp_degree(&factors[0]), 1); // the simple root
        assert!(qp_is_one(&factors[1]));
        assert_eq!(qp_degree(&factors[2]), 1); // the triple root
        assert_eq!(qp_eval(&factors[2], &Rational::from(1)), 0);
    }

    #[test]
    fn exact_rational_root_is_found_or_declined() {
        let chi: QPoly = [283, -876, 930, -364, 27]
            .iter()
            .map(|&c| Rational::from(c))
            .collect();
        assert_eq!(
            exact_rational_root_near(&chi, 283.0 / 27.0),
            Some(Rational::from((283, 27)))
        );
        // t² − 2 has no rational root; "not established" rather than a rounding.
        let irrational: QPoly = [-2, 0, 1].iter().map(|&c| Rational::from(c)).collect();
        assert_eq!(exact_rational_root_near(&irrational, 2.0_f64.sqrt()), None);
    }

    #[test]
    fn polynomial_division_is_exact() {
        // (t² − 1) = (t − 1)(t + 1)
        let a: QPoly = [-1, 0, 1].iter().map(|&c| Rational::from(c)).collect();
        let b: QPoly = [-1, 1].iter().map(|&c| Rational::from(c)).collect();
        let (q, r) = qp_divmod(&a, &b).unwrap();
        assert!(qp_is_zero(&r));
        assert_eq!(q, vec![Rational::from(1), Rational::from(1)]);
        assert_eq!(
            crate::calculus::asymptotic_common::qp_mul(&q, &b),
            qp_trim(a)
        );
    }
}
