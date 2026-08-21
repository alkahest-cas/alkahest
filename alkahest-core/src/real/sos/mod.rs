//! Positivity certificates: sum-of-squares and Positivstellensatz-lite.
//!
//! [`crate::real::cad`] decides real-algebraic questions completely, and pays
//! doubly-exponential cost for that completeness. This module answers the
//! narrower question that actually arises in applied autoresearch — *is this
//! polynomial non-negative (here)?* — by **searching for a certificate** rather
//! than deciding, which is fast when it succeeds and honest when it does not.
//!
//! The certificate is a short algebraic identity:
//!
//! ```text
//! unconstrained:  p = Σ_j σ_j·q_j²                        (CertificateKind::Sos)
//! constrained:    p = Σ_α c_α·Π_i g_i^{α_i},  c_α ≥ 0     (CertificateKind::Handelman)
//! ```
//!
//! Anyone can check it by expanding — no trust in this implementation is
//! required, which is exactly what makes it useful to a proof assistant
//! (`PositivityCertificate::to_lean`) and to a referee.
//!
//! # Soundness
//!
//! The DSOS search ([`mod@gram`]) runs through the rational simplex in
//! [`lp`], with no floating point anywhere near it. The general PSD search
//! ([`mod@psd`], on top of [`mod@sdp`] and [`mod@linalg`]) and the Reznick
//! multiplier search built on it (below) *do* use a floating-point numeric
//! search — but only ever to *propose* a Gram matrix; every proposal is
//! rounded to nearby rationals and re-expanded to check it equals the target
//! **exactly**, in ℚ, before it is returned ([`PositivityCertificate::verify`]
//! runs the identical check on demand). A certificate that fails that check
//! is a bug in the search and is refused, never returned — no floating-point
//! result is ever trusted as a certificate on its own.
//!
//! # What a failure means
//!
//! The search covers, in order: the linear-programming-representable DSOS
//! subcone (solvable exactly); the full PSD Gram cone, when DSOS fails (a
//! strict superset, but only reachable via the sound-but-incomplete numeric
//! search above); and a Reznick multiplier search `(Σxᵢ²)^N·p`, when even
//! that fails on `p` itself. None of these three is complete — a certificate
//! of a given shape may exist at a higher degree/budget than was searched, or
//! not exist in that shape at all, and `sos_decompose` cannot tell those
//! apart. The multiplier search specifically had to add Douglas–Rachford
//! splitting and facial reduction alongside its original annealed
//! alternating-projection search (see `real::sos::sdp::Family::douglas_rachford_from`
//! and `real::sos::psd`'s `facial_reduction_search`) because certificates
//! whose witnessing Gram matrix is *singular* — sitting exactly on the PSD
//! cone's boundary, as for the textbook examples Motzkin and Robinson — are a
//! well-known hard case for plain alternating projection (see
//! `real::sos::tests::motzkin_certifies_via_a_reznick_multiplier` for the
//! worked diagnosis); both of those examples are covered by the current
//! search, but a boundary-only certificate at some other degree is not
//! guaranteed to be. So [`SosError::NoCertificate`] means precisely *"no
//! certificate of this shape was found at this degree/budget"*. It does
//! **not** mean "not a sum of squares", and it does **not** mean "not
//! non-negative". The three answers are kept distinct in the API on purpose —
//! a loop that conflates them will discard true results:
//!
//! | Outcome | Meaning |
//! |---|---|
//! | `Ok(cert)` | Proved non-negative, with a checkable witness |
//! | `Err(Negative { witness })` | Proved *not* non-negative — a point where `p < 0` |
//! | `Err(NoCertificate { .. })` | Undecided at this degree — raise it, or use `decide` |

pub mod cert;
pub mod gram;
pub mod linalg;
pub mod lp;
pub mod psd;
pub mod ratpoly;
pub mod sdp;

pub use cert::{CertificateKind, Multiplier, PositivityCertificate, SosPoly, SosTerm};
pub use ratpoly::RatPoly;

use crate::kernel::{ExprId, ExprPool};
use lp::{Lp, LpStatus, Rel};
use ratpoly::Exponents;
use rug::Rational;
use std::fmt;

/// Search bounds for the certificate search.
#[derive(Debug, Clone, Copy)]
pub struct SosOpts {
    /// Highest total degree of the monomial basis for SOS multipliers. `None`
    /// derives it from the target (`⌈deg p / 2⌉`), which is the smallest basis
    /// that can possibly work.
    pub basis_degree: Option<u32>,
    /// Handelman level: the largest total power of the constraint products
    /// `Π g_i^{α_i}` considered (`Σ α_i ≤ level`).
    pub level: u32,
}

impl Default for SosOpts {
    fn default() -> Self {
        SosOpts {
            basis_degree: None,
            level: 2,
        }
    }
}

/// Why a positivity question was not answered with a certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SosError {
    /// The input (or a constraint) is not a polynomial in the given variables.
    NotPolynomial(String),
    /// No certificate of the searched shape exists at this degree/level. This
    /// is *not* a claim that `p` is negative, nor that it is not SOS.
    NoCertificate(String),
    /// `p` is definitely **not** non-negative: a witness point is included.
    Negative(String),
    /// Malformed call (no variables, degree bounds out of range, …).
    InvalidInput(String),
    /// A candidate certificate failed exact re-expansion. Refused rather than
    /// returned; this indicates a bug in the search.
    VerificationFailed(String),
}

/// Separates the human-readable half of a [`SosError::NoCertificate`]
/// message from the trace of what the search actually did. Stable: callers
/// split on it via [`SosError::search_trace`].
const SEARCH_TRACE_MARKER: &str = "what the search actually did:";

impl SosError {
    /// The trace of what the certificate search actually did, one step per
    /// line, or `None` for errors that carry no trace.
    ///
    /// `E-SOS-002` covers three materially different situations — a search
    /// that ran and was exhausted, a search that ran up to an iteration or
    /// rounding budget, and a basis or multiplier power that was **never
    /// searched at all** because it was over a size ceiling — and the code
    /// alone cannot distinguish them. Reading the trace can; so can
    /// [`Self::hit_a_search_ceiling`], which is the single question most
    /// callers actually have.
    pub fn search_trace(&self) -> Option<&str> {
        match self {
            SosError::NoCertificate(msg) => msg
                .split_once(SEARCH_TRACE_MARKER)
                .map(|(_, trace)| trace.trim_start_matches('\n')),
            _ => None,
        }
    }

    /// Did a size ceiling stop some part of the search from running at all?
    ///
    /// `true` means the refusal is *weaker* than an exhausted search: at
    /// least one basis or multiplier power was skipped without being looked
    /// at, so "no certificate was found" does not even mean "everything in
    /// scope was tried". Never a reason to record a positive verdict either
    /// way — `E-SOS-002` is `unknown` in all cases — but a caller escalating
    /// to a more expensive method should know which kind of refusal it got.
    pub fn hit_a_search_ceiling(&self) -> bool {
        self.search_trace()
            .is_some_and(|trace| trace.contains("NOT SEARCHED"))
    }
}

impl fmt::Display for SosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SosError::NotPolynomial(s) => write!(f, "sos: not a polynomial: {s}"),
            SosError::NoCertificate(s) => write!(f, "sos: no certificate found: {s}"),
            SosError::Negative(s) => write!(f, "sos: the target is negative somewhere: {s}"),
            SosError::InvalidInput(s) => write!(f, "sos: invalid input: {s}"),
            SosError::VerificationFailed(s) => {
                write!(f, "sos: certificate failed exact verification: {s}")
            }
        }
    }
}

impl std::error::Error for SosError {}

impl crate::errors::AlkahestError for SosError {
    fn code(&self) -> &'static str {
        match self {
            SosError::NotPolynomial(_) => "E-SOS-001",
            SosError::NoCertificate(_) => "E-SOS-002",
            SosError::Negative(_) => "E-SOS-003",
            SosError::InvalidInput(_) => "E-SOS-004",
            SosError::VerificationFailed(_) => "E-SOS-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            SosError::NotPolynomial(_) => {
                "positivity certificates are for polynomials in the listed variables; expand or \
                 clear denominators first, and pass every symbol that occurs as a variable"
            }
            SosError::NoCertificate(_) => {
                "record this as unknown, not as a closed branch: read the 'what the search \
                 actually did' trace in the message first — a line marked NOT SEARCHED is a \
                 size ceiling that fired, not a search that came up empty — then raise \
                 basis_degree (unconstrained) or level (constrained); this is not a proof \
                 that no SOS decomposition exists, and still less that the inequality is \
                 false — alkahest.decide is the complete (and far more expensive) fallback"
            }
            SosError::Negative(_) => {
                "the witness point in the message satisfies the constraints and makes the target \
                 negative; the claim is false as stated"
            }
            SosError::InvalidInput(_) => {
                "pass at least one variable, and keep basis_degree/level within the supported range"
            }
            SosError::VerificationFailed(_) => {
                "internal: report the target and constraints as a minimal failing example"
            }
        })
    }
}

/// Small rational grid used to look for a point where the claim already fails.
/// Finding one turns "no certificate" into the *definite* answer "negative",
/// which is far more useful to a loop than an inconclusive refusal.
fn negativity_witness(
    target: &RatPoly,
    constraints: &[RatPoly],
    nvars: usize,
) -> Option<Vec<Rational>> {
    const GRID: [(i32, i32); 9] = [
        (0, 1),
        (1, 1),
        (-1, 1),
        (1, 2),
        (-1, 2),
        (2, 1),
        (-2, 1),
        (3, 1),
        (-3, 1),
    ];
    // Cartesian product over a small grid; bounded so this stays cheap.
    let per_var = if nvars <= 2 {
        GRID.len()
    } else if nvars <= 4 {
        5
    } else {
        3
    };
    let total = per_var.checked_pow(nvars as u32)?;
    if total > 20_000 {
        return None;
    }
    for idx in 0..total {
        let mut rest = idx;
        let mut point = Vec::with_capacity(nvars);
        for _ in 0..nvars {
            let (num, den) = GRID[rest % per_var];
            rest /= per_var;
            point.push(Rational::from((num, den)));
        }
        if constraints.iter().any(|g| g.eval(&point) < 0) {
            continue;
        }
        if target.eval(&point) < 0 {
            return Some(point);
        }
    }
    None
}

fn format_point(names: &[String], point: &[Rational]) -> String {
    names
        .iter()
        .zip(point.iter())
        .map(|(n, v)| format!("{n} = {v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn var_names(vars: &[ExprId], pool: &ExprPool) -> Vec<String> {
    vars.iter().map(|&v| pool.display(v).to_string()).collect()
}

/// Finish a candidate: verify exactly, then return it. A candidate that does
/// not re-expand to the target is refused — this is the single choke point that
/// every success path goes through.
fn finish(cert: PositivityCertificate) -> Result<PositivityCertificate, SosError> {
    match cert.verify() {
        Ok(()) => Ok(cert),
        Err(why) => Err(SosError::VerificationFailed(why)),
    }
}

/// Highest Reznick multiplier power `N` [`sos_decompose`] tries, once both
/// the diagonally dominant and the general PSD Gram searches on `p` itself
/// have failed. Reznick's theorem promises that `(x_1²+…+x_n²)^N·p` is SOS
/// for *some* `N` when `p` is a positive definite form; there is no bound on
/// how large that `N` needs to be in general, so this is a search budget,
/// not a completeness guarantee — running past it is `NoCertificate`
/// (undecided), never a claim that no such `N` exists.
pub(crate) const MAX_MULTIPLIER_POWER: u32 = 4;

/// Above this many monomials in `σ·p`'s monomial basis, a multiplier power
/// is skipped rather than searched, so the (no longer LP-cheap) PSD Gram
/// search stays bounded.
const MAX_MULTIPLIER_BASIS_LEN: usize = 90;

/// Cooperative checkpoint for the positivity search: `false` means stop.
///
/// The search layer is `Option`-shaped throughout (`multiplier_search`,
/// `psd_search`, `dsos_search` all return `Option`), deliberately — it is
/// structurally unable to report anything but "found" or "not found". That
/// makes a budget stop indistinguishable from an exhausted search at this
/// level, and `E-SOS-002` already says "record this as unknown, not as a
/// closed branch". So the trip is recorded out of band and the *bindings*
/// raise `BudgetExceededError` instead of `SosError`, which is the one thing
/// that keeps an unattended loop from filing a timeout as a false negative.
pub(crate) fn budget_ok() -> bool {
    match crate::budget::check_all() {
        Ok(()) => true,
        Err(trip) => {
            crate::budget::record_trip(trip);
            false
        }
    }
}

/// Try `p·(x_1²+…+x_n²)^N` for `N = 1, 2, …, max_power` (Reznick
/// multipliers): a positive semidefinite form can fail to be SOS itself
/// (Motzkin, Choi–Lam, Robinson's form) yet become SOS after multiplying by
/// a high enough power of the sum of squares. Each `N` is tried with the
/// full (non-diagonally-dominant) PSD Gram search in [`psd`], which is
/// itself only ever a sound suggestion mechanism — see that module's
/// soundness note.
///
/// `max_power` and `max_basis_len` are explicit parameters rather than the
/// module constants above so tests can exercise the "budget exhausted"
/// path with a small, fast budget instead of the production one.
///
/// `basis_degree` is the caller's `SosOpts::basis_degree`. `None` uses the
/// smallest basis that can possibly work for `σ·p` (`⌈deg(σ·p)/2⌉`); a
/// larger value genuinely widens the basis searched here. It used to be
/// ignored outright on this path, which made `E-SOS-002`'s advice to raise
/// `basis_degree` inert in exactly the case where that error fires most.
fn multiplier_search(
    target: &RatPoly,
    nvars: usize,
    max_power: u32,
    max_basis_len: usize,
    basis_degree: Option<u32>,
    log: &mut Vec<String>,
) -> Option<(RatPoly, SosPoly, u32)> {
    for n in 1..=max_power {
        if !budget_ok() {
            return None;
        }
        let sigma = RatPoly::sum_of_squares(nvars).pow(n);
        let q = target.mul(&sigma);
        let qdeg = q.total_degree();
        if qdeg % 2 != 0 {
            continue;
        }
        let min_basis_deg = qdeg.div_ceil(2);
        // A basis below ⌈deg(σ·p)/2⌉ cannot reproduce σ·p's top-degree
        // terms at all, so the caller's request is a floor to raise to, not
        // a ceiling to clamp at.
        let basis_deg = basis_degree.unwrap_or(min_basis_deg).max(min_basis_deg);
        // Budget against the basis actually searched, not the raw
        // `monomial_basis` count: `psd_search` applies both the homogeneity
        // restriction and the half-Newton-polytope reduction before it
        // searches anything, and comparing the *unreduced* count against the
        // budget rejects powers whose real basis is well inside it (C₇'s
        // N=1, for instance: 120 raw, 84 real, against a budget of 90).
        let raw_len = gram::monomial_basis(nvars, basis_deg).len();
        let basis_len = psd::searched_basis_len(&q, basis_deg);
        if basis_len > max_basis_len {
            log.push(format!(
                "multiplier search: N={n} NOT SEARCHED — σ·p would need a degree-{basis_deg} \
                 basis of {basis_len} monomials ({raw_len} before the homogeneity and \
                 half-Newton reductions), over the search budget of {max_basis_len}; stopping \
                 here, so no multiplier power from N={n} up was searched at all"
            ));
            break;
        }
        log.push(format!(
            "multiplier search: trying σ = (Σxᵢ²)^{n}, searching the full PSD Gram cone for \
             σ·p over the degree-{basis_deg} monomial basis ({basis_len} monomials, {raw_len} \
             before reduction)"
        ));
        if let Some(sos) = psd::psd_search_logged(&q, basis_deg, log) {
            log.push(format!("multiplier search succeeded at N={n}"));
            return Some((sigma, sos, basis_deg));
        }
        log.push(format!(
            "multiplier search: N={n} produced no certificate (see the lines above for what \
             ran at this power and what was skipped)"
        ));
    }
    None
}

/// `p = Σ_j σ_j·q_j²` — an exact rational sum-of-squares decomposition.
///
/// Refuses with [`SosError`] rather than guessing: `E-SOS-003` when `p` is
/// negative somewhere (with a witness point), `E-SOS-002` when no certificate
/// of the searched shape exists at this basis degree.
pub fn sos_decompose(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
    opts: &SosOpts,
) -> Result<PositivityCertificate, SosError> {
    if vars.is_empty() {
        return Err(SosError::InvalidInput(
            "at least one variable is required".into(),
        ));
    }
    // Only this call's trip may be attributed to this call.
    crate::budget::clear_trip();
    let names = var_names(vars, pool);
    let target = RatPoly::from_expr(expr, vars, pool).map_err(SosError::NotPolynomial)?;
    let nvars = vars.len();
    let mut log = Vec::new();

    // A constant is trivially decided, either way.
    if let Some(c) = target.as_constant() {
        if c < 0 {
            return Err(SosError::Negative(format!(
                "the target is the negative constant {c}"
            )));
        }
        let mut sos = SosPoly::default();
        sos.push(c.clone(), RatPoly::one(nvars));
        log.push(format!("target is the non-negative constant {c}"));
        return finish(PositivityCertificate {
            vars: vars.to_vec(),
            var_names: names,
            target,
            constraints: Vec::new(),
            kind: CertificateKind::Sos,
            degree: 0,
            terms: vec![Multiplier {
                constraints: Vec::new(),
                sos,
            }],
            log,
        });
    }

    // Odd total degree cannot be globally non-negative (the leading behaviour
    // flips sign), and the witness search below usually finds the point.
    let deg = target.total_degree();
    if let Some(point) = negativity_witness(&target, &[], nvars) {
        return Err(SosError::Negative(format!(
            "p({}) = {} < 0",
            format_point(&names, &point),
            target.eval(&point)
        )));
    }
    if deg % 2 == 1 {
        return Err(SosError::NoCertificate(format!(
            "total degree {deg} is odd, so p cannot be a sum of squares (no witness point was \
             found on the sampling grid, so this is a statement about the SOS question, not a \
             claim that p is negative)"
        )));
    }

    let basis_deg = opts.basis_degree.unwrap_or(deg.div_ceil(2));
    if basis_deg > 12 {
        return Err(SosError::InvalidInput(
            "basis_degree above 12 is refused: the monomial basis (and the exact LP over it) \
             grows too fast to be useful"
                .into(),
        ));
    }
    log.push(format!(
        "searching the diagonally dominant cone over the degree-{basis_deg} monomial basis"
    ));

    let Some(sos) = gram::dsos_search(&target, basis_deg) else {
        log.push(
            "diagonally dominant search failed; trying the full PSD Gram cone directly".to_string(),
        );
        if let Some(sos) = psd::psd_search_logged(&target, basis_deg, &mut log) {
            log.push(
                "full PSD Gram search succeeded (p is SOS but its Gram matrix is not \
                 diagonally dominant)"
                    .to_string(),
            );
            return finish(PositivityCertificate {
                vars: vars.to_vec(),
                var_names: names,
                target,
                constraints: Vec::new(),
                kind: CertificateKind::Sos,
                degree: basis_deg,
                terms: vec![Multiplier {
                    constraints: Vec::new(),
                    sos,
                }],
                log,
            });
        }
        if let Some((_sigma, sos, mult_basis_deg)) = multiplier_search(
            &target,
            nvars,
            MAX_MULTIPLIER_POWER,
            MAX_MULTIPLIER_BASIS_LEN,
            opts.basis_degree,
            &mut log,
        ) {
            return finish(PositivityCertificate {
                vars: vars.to_vec(),
                var_names: names,
                target,
                constraints: Vec::new(),
                kind: CertificateKind::Sos,
                degree: mult_basis_deg,
                terms: vec![Multiplier {
                    constraints: Vec::new(),
                    sos,
                }],
                log,
            });
        }
        // The trace is not decoration: `None` out of the searches above
        // covers "searched and exhausted", "searched up to a budget" and
        // "never searched at all" (a multiplier power skipped for basis
        // size, or an affine family over `psd`'s free-parameter ceiling),
        // and the bare error code cannot tell those apart. Reporting which
        // multiplier powers actually ran — and which were skipped, with the
        // reason — is the difference between "we looked and found nothing"
        // and "we did not look".
        let trace = log
            .iter()
            .map(|line| format!("\n  - {line}"))
            .collect::<String>();
        return Err(SosError::NoCertificate(format!(
            "undecided, not a refutation — no diagonally dominant or general PSD Gram matrix \
             over the degree-{basis_deg} monomial basis reproduces p, and no Reznick multiplier \
             (Σxᵢ²)^N up to N={MAX_MULTIPLIER_POWER} that was actually searched made σ·p SOS \
             either. None of this is a proof that p is not SOS (with or without a multiplier), \
             and still less that p is not non-negative — only that no certificate of these \
             shapes was found at this size. Read the trace below before treating this as \
             exhaustive: a line marked NOT SEARCHED is a budget that fired, not a search that \
             came up empty. Raise basis_degree, or fall back to alkahest.decide.\
             \n{SEARCH_TRACE_MARKER}{trace}"
        )));
    };

    finish(PositivityCertificate {
        vars: vars.to_vec(),
        var_names: names,
        target,
        constraints: Vec::new(),
        kind: CertificateKind::Sos,
        degree: basis_deg,
        terms: vec![Multiplier {
            constraints: Vec::new(),
            sos,
        }],
        log,
    })
}

/// All exponent multisets `α` over `k` constraints with `1 ≤ Σ α_i ≤ level`,
/// plus the empty product (the constant term).
fn constraint_products(k: usize, level: u32) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    let mut frontier: Vec<Vec<usize>> = vec![Vec::new()];
    for _ in 0..level {
        let mut next = Vec::new();
        for base in &frontier {
            let start = base.last().copied().unwrap_or(0);
            for i in start..k {
                let mut v = base.clone();
                v.push(i);
                next.push(v);
            }
        }
        out.extend(next.iter().cloned());
        frontier = next;
    }
    out
}

/// Prove `p ≥ 0` on `{x : g_i(x) ≥ 0}` with a Handelman-style certificate
/// `p = Σ_α c_α·Π g_i^{α_i}`, `c_α ≥ 0` rational.
///
/// With no constraints this is [`sos_decompose`].
pub fn prove_nonneg(
    expr: ExprId,
    constraints: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
    opts: &SosOpts,
) -> Result<PositivityCertificate, SosError> {
    if constraints.is_empty() {
        return sos_decompose(expr, vars, pool, opts);
    }
    crate::budget::clear_trip();
    if vars.is_empty() {
        return Err(SosError::InvalidInput(
            "at least one variable is required".into(),
        ));
    }
    if opts.level == 0 || opts.level > 8 {
        return Err(SosError::InvalidInput(
            "level must be between 1 and 8".into(),
        ));
    }
    let names = var_names(vars, pool);
    let nvars = vars.len();
    let target = RatPoly::from_expr(expr, vars, pool).map_err(SosError::NotPolynomial)?;
    let gs: Vec<RatPoly> = constraints
        .iter()
        .map(|&g| RatPoly::from_expr(g, vars, pool).map_err(SosError::NotPolynomial))
        .collect::<Result<_, _>>()?;

    if let Some(point) = negativity_witness(&target, &gs, nvars) {
        return Err(SosError::Negative(format!(
            "p({}) = {} < 0 at a point satisfying every constraint",
            format_point(&names, &point),
            target.eval(&point)
        )));
    }

    let products = constraint_products(gs.len(), opts.level);
    // Expand each product Π g_i^{α_i} once; the LP unknowns are its weights.
    let expanded: Vec<RatPoly> = products
        .iter()
        .map(|idxs| {
            let mut acc = RatPoly::one(nvars);
            for &i in idxs {
                acc = acc.mul(&gs[i]);
            }
            acc
        })
        .collect();

    // Match coefficients: Σ_α c_α·(Π g)_α = p, one equation per monomial that
    // occurs anywhere, with c_α ≥ 0 (the simplex's implicit variable bound).
    let mut monomials: std::collections::BTreeSet<Exponents> = Default::default();
    for e in &expanded {
        monomials.extend(e.terms().keys().cloned());
    }
    monomials.extend(target.terms().keys().cloned());

    let mut prog = Lp::new(expanded.len());
    for m in &monomials {
        let row: Vec<Rational> = expanded.iter().map(|e| e.coeff(m)).collect();
        prog.constrain(row, Rel::Eq, target.coeff(m));
    }

    let weights = match prog.solve() {
        LpStatus::Optimal(w) => w,
        _ => {
            return Err(SosError::NoCertificate(format!(
                "undecided, not a refutation — no non-negative combination of constraint \
                 products up to level {} reproduces the target. Raise level, or the claim may \
                 need a Putinar-style certificate with SOS (not merely non-negative constant) \
                 multipliers. Record this as unknown, not as a closed branch",
                opts.level
            )));
        }
    };

    let mut terms = Vec::new();
    for (idxs, w) in products.iter().zip(weights.iter()) {
        if *w == 0 {
            continue;
        }
        let mut sos = SosPoly::default();
        sos.push(w.clone(), RatPoly::one(nvars));
        terms.push(Multiplier {
            constraints: idxs.clone(),
            sos,
        });
    }

    finish(PositivityCertificate {
        vars: vars.to_vec(),
        var_names: names,
        target,
        constraints: gs,
        kind: CertificateKind::Handelman,
        degree: opts.level,
        terms,
        log: vec![format!(
            "Handelman search over {} constraint products up to level {}",
            products.len(),
            opts.level
        )],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::kernel::Domain;

    fn setup() -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        (pool, x, y)
    }

    /// The Motzkin form `x⁴y² + x²y⁴ − 3x²y²z² + z⁶`: non-negative, not SOS,
    /// and the case 2026-08-19 item 26d measured at **418.5 s inside a
    /// `Budget(wall_ms=3000)`** — 140x over, ending in `E-SOS-002`.
    ///
    /// `E-SOS-002` says "record this as unknown, not as a closed branch", but
    /// an unattended loop that reads a *timeout* as "not SOS" files a false
    /// negative. So the budget stop must stay distinguishable: the search still
    /// returns `NoCertificate`, and the trip is recorded out of band for the
    /// bindings to raise as `BudgetExceededError`.
    fn motzkin(pool: &ExprPool, x: ExprId, y: ExprId, z: ExprId) -> ExprId {
        let sq = |e: ExprId| pool.mul(vec![e, e]);
        pool.add(vec![
            pool.mul(vec![sq(sq(x)), sq(y)]),
            pool.mul(vec![sq(x), sq(sq(y))]),
            pool.mul(vec![pool.integer(-3_i32), sq(x), sq(y), sq(z)]),
            sq(sq(z)),
        ])
    }

    #[test]
    fn sos_decompose_honours_a_wall_budget() {
        let (pool, x, y) = setup();
        let z = pool.symbol("z", Domain::Real);
        let p = motzkin(&pool, x, y, z);
        let _guard = crate::budget::enter(
            crate::budget::Budget::new().with_wall(std::time::Duration::from_millis(200)),
        );
        let start = std::time::Instant::now();
        let err = sos_decompose(p, &[x, y, z], &pool, &SosOpts::default())
            .expect_err("no SOS certificate exists for the Motzkin form");
        // Loose by two orders of magnitude against the 418.5 s that a 3 s
        // budget used to buy: this asserts the budget is consulted at all.
        assert!(start.elapsed().as_secs() < 60, "budget was not consulted");
        assert!(matches!(err, SosError::NoCertificate(_)), "{err:?}");
        let trip = crate::budget::take_trip().expect("the budget trip must be recorded");
        assert_eq!(trip.code(), "E-BUDGET-001");
    }

    #[test]
    fn perfect_square_is_certified() {
        let (pool, x, y) = setup();
        // (x - y)^2 = x^2 - 2xy + y^2
        let p = pool.add(vec![
            pool.mul(vec![x, x]),
            pool.mul(vec![pool.integer(-2_i32), x, y]),
            pool.mul(vec![y, y]),
        ]);
        let cert = sos_decompose(p, &[x, y], &pool, &SosOpts::default()).expect("certificate");
        assert_eq!(cert.kind, CertificateKind::Sos);
        cert.verify().expect("re-expands exactly");
        assert!(cert.num_squares() >= 1);
    }

    #[test]
    fn sum_of_even_powers_is_certified() {
        let (pool, x, y) = setup();
        let p = pool.add(vec![
            pool.mul(vec![x, x, x, x]),
            pool.mul(vec![y, y, y, y]),
            pool.integer(1_i32),
        ]);
        let cert = sos_decompose(p, &[x, y], &pool, &SosOpts::default()).expect("certificate");
        cert.verify().expect("re-expands exactly");
    }

    #[test]
    fn negative_polynomial_returns_a_witness_not_a_refusal() {
        let (pool, x, y) = setup();
        // x^2 - 1 is negative at x = 0.
        let p = pool.add(vec![pool.mul(vec![x, x]), pool.integer(-1_i32)]);
        let err = sos_decompose(p, &[x, y], &pool, &SosOpts::default()).expect_err("negative");
        assert!(matches!(err, SosError::Negative(_)));
        assert_eq!(err.code(), "E-SOS-003");
    }

    #[test]
    fn motzkin_certifies_via_a_reznick_multiplier() {
        let (pool, x, y) = setup();
        // Motzkin: x^4·y^2 + x^2·y^4 − 3·x^2·y^2 + 1 is non-negative but is
        // the textbook example of a polynomial that is *not* itself a sum of
        // squares — Hilbert's 1888 theorem allows non-SOS PSD forms outside
        // ternary quartics, and Motzkin (1967) is the standard witness.
        // Multiplying by (x²+y²) is classically known to fix this (it is
        // exactly the kind of case Reznick's theorem covers). The witnessing
        // Gram matrix for that fact is *singular* — it sits exactly on the
        // boundary of the PSD cone, not in its interior — which is why a
        // plain annealed alternating-projection search (the first version of
        // this feature) could get arbitrarily close (min eigenvalue from
        // roughly −1.6 down to roughly −0.0018 as its floor annealed to 0)
        // without ever closing the gap: the textbook symptom of a tangential
        // (non-transversal) set intersection. `real::sos::psd`'s search now
        // also tries Douglas–Rachford splitting and facial reduction, either
        // of which is known to escape exactly this kind of stall, and it
        // finds Motzkin's certificate here.
        let p = pool.add(vec![
            pool.mul(vec![x, x, x, x, y, y]),
            pool.mul(vec![x, x, y, y, y, y]),
            pool.mul(vec![pool.integer(-3_i32), x, x, y, y]),
            pool.integer(1_i32),
        ]);
        let cert =
            sos_decompose(p, &[x, y], &pool, &SosOpts::default()).expect("Motzkin now certifies");
        assert_eq!(cert.kind, CertificateKind::Sos);
        let sigma = cert
            .multiplier()
            .expect("Motzkin is not itself SOS, so this must be a multiplier certificate");
        // The exact identity actually checked: σ·p = Σ c_i q_i², in ℚ — the
        // real soundness argument, independent of whatever the numeric
        // search that proposed it actually converged to.
        assert_eq!(sigma.mul(&cert.target), cert.expand());
        cert.verify().expect("re-verifies exactly end to end");

        // Composes with to_lean: a self-contained, sorry-free Lean sketch.
        let lean = cert
            .to_lean()
            .expect("multiplier certificates emit Lean too");
        assert!(!lean.contains("sorry"));
        assert!(!lean.contains("admit"));
        assert!(lean.contains("alkahest_multiplier_factor"));
        assert!(lean.contains("ring"));
    }

    #[test]
    fn multiplier_search_reports_undecided_not_not_sos_when_out_of_budget() {
        let (pool, x, y) = setup();
        // Same Motzkin target as the previous test, which now certifies at
        // the production budget. Here the internal search is instead driven
        // with a budget of *zero* multiplier powers directly — i.e. exactly
        // the "search legitimately runs out of budget" case — and it must
        // come back empty-handed rather than fabricate a certificate. This
        // test's point is narrower than the previous one: independent of
        // whether the production budget finds a given target's certificate,
        // a caller-supplied budget of zero must never manufacture one.
        let p = pool.add(vec![
            pool.mul(vec![x, x, x, x, y, y]),
            pool.mul(vec![x, x, y, y, y, y]),
            pool.mul(vec![pool.integer(-3_i32), x, x, y, y]),
            pool.integer(1_i32),
        ]);
        let target = RatPoly::from_expr(p, &[x, y], &pool).unwrap();
        let mut log = Vec::new();
        let out = multiplier_search(&target, 2, /* max_power */ 0, 90, None, &mut log);
        assert!(
            out.is_none(),
            "a zero-power budget must not manufacture a certificate"
        );

        // `multiplier_search`'s signature (`Option`, not a `Result` with a
        // "negative" branch) makes it structurally unable to report anything
        // but "found" or "not found within budget" — it cannot claim
        // negativity even by construction. The public `sos_decompose` wires
        // exactly this `None` into `SosError::NoCertificate` (see the
        // `multiplier_search(...)` call a few lines above the final `Err` in
        // `sos_decompose`), never `SosError::Negative`. That distinction
        // matters here specifically because Motzkin genuinely is
        // non-negative everywhere (confirmed independently: the grid search
        // below finds no witness), so "not SOS within budget" and "negative"
        // are not just differently coded, they are different facts, and only
        // one of them is true.
        assert!(
            negativity_witness(&target, &[], 2).is_none(),
            "Motzkin is non-negative, so a witness must not exist"
        );
    }

    /// The C₇ copositivity form: `Σ_ij M_ij x_i² x_j²` with
    /// `M = 3·(I + A(C₇)) − J`, `A(C₇)` the adjacency matrix of the 7-cycle.
    /// A 7-variable quartic that is non-negative but not SOS.
    fn c7_copositivity_form() -> RatPoly {
        let n = 7usize;
        let adj = |i: usize, j: usize| -> i64 {
            let d = (i + n - j) % n;
            i64::from(d == 1 || d == n - 1)
        };
        let mut p = RatPoly::zero(n);
        for i in 0..n {
            for j in 0..n {
                let m = 3 * (i64::from(i == j) + adj(i, j)) - 1;
                if m == 0 {
                    continue;
                }
                let mut e = vec![0u32; n];
                e[i] += 2;
                e[j] += 2;
                p = p.add(&RatPoly::monomial(n, e, Rational::from(m)));
            }
        }
        p
    }

    #[test]
    fn the_multiplier_budget_uses_the_basis_actually_searched() {
        // `MAX_MULTIPLIER_BASIS_LEN` used to be compared against the *raw*
        // `monomial_basis` count, which is not the basis `psd_search`
        // searches: it applies a homogeneity restriction and a
        // half-Newton-polytope reduction first. For C₇'s N=1 multiplier that
        // is the difference between 120 (rejected, "over the search budget")
        // and the 84 monomials really at stake (inside the budget of 90) —
        // so the power was skipped on a number that never described the
        // search, and the log reported a size the search never used.
        let p = c7_copositivity_form();
        let q = p.mul(&RatPoly::sum_of_squares(7));
        assert_eq!(gram::monomial_basis(7, 3).len(), 120);
        assert!(gram::monomial_basis(7, 3).len() > MAX_MULTIPLIER_BASIS_LEN);
        assert_eq!(psd::searched_basis_len(&q, 3), 84);
        assert!(psd::searched_basis_len(&q, 3) <= MAX_MULTIPLIER_BASIS_LEN);

        let mut log = Vec::new();
        let out = multiplier_search(&p, 7, 1, MAX_MULTIPLIER_BASIS_LEN, None, &mut log);
        assert!(out.is_none(), "C₇'s N=1 certificate is still out of reach");
        let tried = log
            .iter()
            .find(|l| l.contains("trying σ"))
            .unwrap_or_else(|| panic!("N=1 must no longer be rejected on basis size: {log:?}"));
        assert!(
            tried.contains("84 monomials") && tried.contains("120 before reduction"),
            "the log must report the searched size, with the raw one for context: {tried}"
        );
    }

    #[test]
    fn a_refusal_reports_which_multiplier_powers_were_actually_searched() {
        // The Horn form (copositivity of the Horn matrix, `Σ_ij H_ij x_i²
        // x_j²`): non-negative, not SOS, and — as `E-SOS-002` used to present
        // it — "no Reznick multiplier up to N=4 made σ·p SOS", which reads as
        // if four multiplier powers had been tried. In fact *no* multiplier
        // power is searched here at all: N=1's affine family has 420 free
        // parameters, over `psd`'s ceiling, and the higher powers are over
        // the basis-length budget. The refusal is the same either way; what
        // must not be the same is what it claims to have done.
        let (pool, _x, _y) = setup();
        let vars: Vec<ExprId> = (0..5)
            .map(|i| pool.symbol(format!("h{i}"), Domain::Real))
            .collect();
        let h = [
            [1, -1, 1, 1, -1],
            [-1, 1, -1, 1, 1],
            [1, -1, 1, -1, 1],
            [1, 1, -1, 1, -1],
            [-1, 1, 1, -1, 1],
        ];
        let mut terms = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                terms.push(pool.mul(vec![
                    pool.integer(h[i][j]),
                    vars[i],
                    vars[i],
                    vars[j],
                    vars[j],
                ]));
            }
        }
        let p = pool.add(terms);

        let err = sos_decompose(p, &vars, &pool, &SosOpts::default()).expect_err("still refused");
        assert_eq!(err.code(), "E-SOS-002");
        let msg = err.to_string();
        assert!(
            msg.contains("what the search actually did:"),
            "the refusal must carry a trace of what ran: {msg}"
        );
        assert!(
            msg.contains("NOT SEARCHED"),
            "a budget that fired must be reported as such, not folded into the same \
             undifferentiated refusal an exhausted search produces: {msg}"
        );
        assert!(
            msg.contains("that was actually searched"),
            "the message must not imply multiplier powers were tried when they were not: {msg}"
        );

        // The same distinction, reachable without string-scraping.
        let trace = err.search_trace().expect("NoCertificate carries a trace");
        assert!(trace.contains("NOT SEARCHED"));
        assert!(
            err.hit_a_search_ceiling(),
            "this refusal is weaker than an exhausted search and must say so"
        );

        // …and the counterpart: an error with no trace does not pretend to
        // have one, and a refusal is never silently upgraded to a refutation.
        let neg = SosError::Negative("witness".into());
        assert!(neg.search_trace().is_none());
        assert!(!neg.hit_a_search_ceiling());
    }

    #[test]
    fn basis_degree_is_not_ignored_on_the_multiplier_path() {
        // `multiplier_search` derived its basis degree from `deg(σ·p)` alone
        // and never read `opts.basis_degree`, so the multiplier path was
        // bit-identical at every setting — while `E-SOS-002`'s remediation
        // told callers to raise `basis_degree`. Raising it now genuinely
        // widens the basis searched on this path.
        let (pool, x, y) = setup();
        let p = pool.add(vec![
            pool.mul(vec![x, x, x, x, y, y]),
            pool.mul(vec![x, x, y, y, y, y]),
            pool.mul(vec![pool.integer(-3_i32), x, x, y, y]),
            pool.integer(1_i32),
        ]);
        let target = RatPoly::from_expr(p, &[x, y], &pool).unwrap();

        let mut default_log = Vec::new();
        multiplier_search(
            &target,
            2,
            1,
            MAX_MULTIPLIER_BASIS_LEN,
            None,
            &mut default_log,
        );
        let mut raised_log = Vec::new();
        multiplier_search(
            &target,
            2,
            1,
            MAX_MULTIPLIER_BASIS_LEN,
            Some(5),
            &mut raised_log,
        );
        let degree_of = |log: &[String]| -> String {
            log.iter()
                .find(|l| l.contains("trying σ"))
                .expect("N=1 is searched either way")
                .split("degree-")
                .nth(1)
                .and_then(|t| t.split(' ').next())
                .expect("the log names the basis degree")
                .to_string()
        };
        assert_eq!(degree_of(&default_log), "4");
        assert_eq!(
            degree_of(&raised_log),
            "5",
            "basis_degree must reach the multiplier path, or E-SOS-002 must stop recommending it"
        );
    }

    #[test]
    fn non_polynomial_is_refused() {
        let (pool, x, y) = setup();
        let p = pool.func("sin", vec![x]);
        let err = sos_decompose(p, &[x, y], &pool, &SosOpts::default()).expect_err("not a poly");
        assert_eq!(err.code(), "E-SOS-001");
    }

    #[test]
    fn handelman_certifies_on_a_box() {
        let (pool, x, _y) = setup();
        // On 0 ≤ x ≤ 1 (as x ≥ 0 and 1 − x ≥ 0): x − x² = x·(1 − x) ≥ 0.
        let g1 = x;
        let g2 = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let p = pool.add(vec![x, pool.mul(vec![pool.integer(-1_i32), x, x])]);
        let cert =
            prove_nonneg(p, &[g1, g2], &[x], &pool, &SosOpts::default()).expect("certificate");
        assert_eq!(cert.kind, CertificateKind::Handelman);
        cert.verify().expect("re-expands exactly");
    }

    #[test]
    fn handelman_finds_the_witness_when_the_claim_is_false() {
        let (pool, x, _y) = setup();
        // x − 1/2 is negative at x = 0, which satisfies x ≥ 0.
        let g1 = x;
        let p = pool.add(vec![x, pool.rational(-1_i32, 2_i32)]);
        let err = prove_nonneg(p, &[g1], &[x], &pool, &SosOpts::default()).expect_err("negative");
        assert!(matches!(err, SosError::Negative(_)));
    }
}
