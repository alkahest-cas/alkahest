//! General coupled twisted-derivation Risch DE over `ℚ(x)(α)` — Risch milestone
//! **M1-step-2**.
//!
//! Solves `D(y) + f·y = g` for `y ∈ ℚ(x)(α)`, where `g ∈ ℚ(x)(α)`, `α` is
//! algebraic of degree `d` over `ℚ(x)` given by an [`AlgExtension`] (M0), and `f`
//! is **either** a base scalar `∈ ℚ(x)` (the historical *diagonal* entry point
//! `solve_alg_rde`) **or** a general extension element `∈ ℚ(x)(α)` (the
//! *non-diagonal* coupled entry point `solve_alg_rde_general`).  This is the
//! "no new logarithm" mixed integral part: an integrand `a(x,α)·exp(kη)` whose
//! antiderivative is `v(x,α)·exp(kη)` with `v ∈ ℚ(x)(α)` and `f = kη'` — when `η`
//! is itself algebraic, `f` is non-base.
//!
//! Writing `y = Σⱼ bⱼ(x) αʲ` and substituting into `D(y) + f·y = g` collects, over
//! the power basis `{1, α, …, α^{d−1}}`, into a **coupled** first-order linear ODE
//! system
//!
//! ```text
//!   b′ + M(x)·b = c,        M[k][j] = the αᵏ-component of D(αʲ) + f·αʲ
//! ```
//!
//! over `ℚ(x)`.  The pure-radical case (`αⁿ = a`) with a *base* `f` is cyclic and
//! `M` is diagonal — solved component-wise by
//! `super::exp_case::try_radical_poly_rde` through the scalar solvers.  This
//! module handles the **general (non-cyclic)** `α` — nested / compositum radicals
//! such as `√x + √(x+1)` — and non-base `f`, where `M` genuinely couples the
//! components.
//!
//! ## Method
//!
//! An undetermined-coefficient ansatz `bⱼ = pⱼ(x)/Den(x)` over candidate
//! denominators `Den` and bounded numerator degree.  Because the operator
//! `L(y) = D(y) + f·y` is `ℚ`-linear in the unknown coefficients of the `pⱼ`, we
//! evaluate `L` on each basis element `αʲ·xᵐ/Den`, clear the common
//! `x`-denominator of every power-basis component (then match each `xᵐ`), and
//! assemble an exact `ℚ`-linear system `A·u = c`.  We Gauss-solve it and **verify
//! `D(y) + f·y = g` exactly in the field** before returning.
//!
//! The *positive* direction is sound by construction: the linear system is
//! faithful to `L(y) = g` (denominator clearing is exact and every `x`-power is
//! matched), and the final field equality is an independent check — a wrong
//! antiderivative can never be emitted.
//!
//! ## Why the answer is three-valued
//!
//! The *negative* direction is a different claim.  A heuristic ansatz that finds
//! nothing has proved nothing: a denominator or degree bound too small to contain
//! the true solution also yields "no solution".  Callers that turned that into
//! `IntegrationError::NonElementary` were publishing a theorem they had not
//! proved.  So the solvers return the three-valued `AlgRdeOutcome`:
//!
//! * `AlgRdeOutcome::Solved` — an exactly-verified `y`.
//! * `AlgRdeOutcome::NoRationalSolution` — **proved**: the denominator bound
//!   *and* the degree bound used were both established complete, and the
//!   resulting `ℚ`-linear system is inconsistent.  Only this may license a
//!   non-elementarity certificate.
//! * `AlgRdeOutcome::Declined` — nothing may be concluded; see [`RdeDecline`].
//!
//! The `Option`-returning entry points are retained as shims and documented as
//! unsafe to conclude from.
//!
//! ## Which branch can honestly prove non-existence
//!
//! Exactly one: **an inconsistent linear system built at a pair of bounds that
//! were both proved complete.**  Everything else declines.  Establishing those
//! bounds for a coupled system is the real work, and it needs three separate
//! things to hold.
//!
//! ### 1. The coefficient ring must be the field
//!
//! `AlgExtension` is `ℚ(x)[y]/(q)`.  If `q` factors, that ring is a *product* of
//! fields and the integrand lives in only one factor; "no solution in the ring"
//! then does **not** imply "no solution in the field".  So the caller must hand
//! over a `MinPolyStatus::ProvedIrreducible`, for which this module supplies
//! the three shape-specific tests actually reachable from the integrator:
//! `radical_minpoly_status`, `compositum_minpoly_status` and
//! `nested_radical_minpoly_status`.  Without it we decline.
//!
//! ### 2. A complete denominator bound (finite poles)
//!
//! Let `Dm` be the lcm of the reduced denominators of `M` and `Dc` that of `c`.
//! At a point `x₀` where both are regular, a pole of `b` of order `m` makes
//! `ord(b′) = m+1` strictly worse than `ord(M·b) ≤ m`, so `b′+M·b` would have a
//! pole where `c` does not — hence **every pole of `b` is a pole of `M` or of
//! `c`**.
//!
//! At a pole `x₀`, write the Laurent expansions.  When `M` has *at most a simple*
//! pole there (residue matrix `R`), the coefficient of `t^{−m−1}` in `b′+M·b` is
//! `(R − m·I)·B₋ₘ`, and `c` has no `t^{−m−1}` term once `m+1 > μ` (`μ` = pole
//! order of `c`).  So
//!
//! ```text
//!   m ≤ max( μ − 1 ,  the largest positive integer eigenvalue of R ).
//! ```
//!
//! This is the matrix form of the scalar *resonance* in
//! [`super::rational_rde::solve_rational_rde_generalized_checked`], and it is
//! computed the same way — over `ℚ`, without ever naming an algebraic `x₀`:
//! `k` is an eigenvalue of `R` at some root of `Dm` iff
//! `gcd(Dm, det(Dm·M − k·Dm′·I)) ≠ 1`, and the search over `k` terminates because
//! the eigenvalues of `R` across *all* roots of `Dm` are exactly the eigenvalues
//! of the `d·deg Dm` square rational matrix representing `R` on `ℚ[x]/(Dm)`,
//! whose spectral radius is bounded by its maximum absolute row sum.
//!
//! If `Dm` is **not squarefree** — `M` has a double or worse finite pole — the
//! leading term of `M·b` can vanish against `ker M₋ᵥ` and the argument gives no
//! bound.  We decline.
//!
//! ### 3. A complete degree bound (the place(s) at infinity)
//!
//! Two independent arguments, whichever applies (both when both do):
//!
//! * **Matrix.** With `ρ = max deg_∞ M[k][j]`, `γ = max deg_∞ c[k]` and
//!   `n = maxⱼ deg_∞ bⱼ`: `deg_∞ b′ ≤ n−1` always.  If `ρ ≥ 0` **and the leading
//!   matrix `M_ρ` is invertible**, `M·b` strictly dominates and `n = γ − ρ`
//!   exactly.  If `ρ = −1`, the two balance and `−n` must be an eigenvalue of
//!   `M₋₁` unless `n ≤ γ+1`.  If `ρ ≤ −2`, `b′` dominates and `n ≤ γ+1`.
//!   A singular `M_ρ` with `ρ ≥ 0` is an irregular singularity — declined.
//! * **Ramified radical.** For `αⁿ = p` with `gcd(n, deg p) = 1` there is a
//!   *single, totally ramified* place above `x = ∞`, so distinct power-basis
//!   components have distinct valuations there and **cannot cancel**.  Grading by
//!   `V(Σ hⱼαʲ) = maxⱼ (n·deg_∞ hⱼ + j·deg p)` gives `V(D u) = V(u) − n`, and the
//!   same three-case dominance argument bounds `V(y)`, hence every `deg_∞ bⱼ`.
//!   This is what decides `∫ exp(√x)/x dx`, whose matrix leading term is singular
//!   precisely *because* `α` has half-integral degree.
//!
//! References: Bronstein (2005) *Symbolic Integration I* §5.4, §6.1, §6.5;
//! Barkatou, "On rational solutions of systems of linear differential equations"
//! (J. Symbolic Comput. 28, 1999) for the residue-matrix denominator bound.

use rug::Rational;

use super::alg_field::{AlgElem, AlgExtension, RatFn};
use super::poly_rde::{
    degree, poly_add, poly_deriv, poly_mul, poly_one, poly_scale, poly_zero, trim, QPoly,
};
use super::rational_rde::{
    poly_div_exact, poly_divrem, poly_gcd, poly_monic, poly_pow, poly_sub, RdeDecline,
};

/// Heuristic floor on the numerator degree tried in the ansatz `bⱼ = pⱼ(x)/Den`.
/// The effective cap is `max(DEG_CAP, analytic_bound)` (see [`alg_x_degree_bound`]),
/// so the search ceiling is never below this floor — existing solves never regress.
const DEG_CAP: usize = 6;

/// Hard clamp on the numerator-degree ansatz ceiling, guarding against a
/// pathological analytic bound that would blow up the `ℚ`-linear system.  Larger
/// solutions are rare and the exact in-field verification keeps soundness anyway.
const X_DEG_SANITY_CAP: usize = 48;

/// Cap on the positive-integer resonance search inside the *proved* denominator
/// and degree bounds.  Past this the solver declines rather than guessing — the
/// bound would no longer be established, so no certificate could follow.
const MAX_ALG_RESONANCE_SEARCH: i64 = 256;

/// Cap on the number of unknowns (`d · (ncap+1)`) in the proved-bound linear
/// system.  Gaussian elimination over `rug::Rational` is cubic with bignum
/// coefficients, so this bounds worst-case latency; past it we decline.
const MAX_ALG_RDE_UNKNOWNS: usize = 512;

/// Largest per-component shearing weight `w` tried in [`system_infinity_bound`]
/// (`sⱼ = j·w`).  Every weight is independently sound, so this only bounds how
/// hard we look; `w` needs to reach `deg_∞ α`, which for the reachable radical /
/// compositum / nested shapes is small.
const MAX_SHEAR_WEIGHT: i64 = 4;

/// Cap on the extension degree for which the proof path runs at all.  The
/// denominator bound needs `d × d` polynomial determinants once per candidate
/// resonance; past this the cost is not worth the rare verdict.
const MAX_PROVED_EXT_DEGREE: usize = 8;

// ---------------------------------------------------------------------------
// The three-valued contract
// ---------------------------------------------------------------------------

/// Three-valued outcome of an algebraic Risch DE solve — the `ℚ(x)(α)` analogue
/// of [`super::rational_rde::RdeOutcome`], sharing its [`RdeDecline`] vocabulary.
///
/// The two-valued `Option` this replaces conflated *"proved there is no solution
/// in `ℚ(x)(α)`"* with *"my ansatz did not find one"*.  Only the first may
/// license a `NonElementary` certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AlgRdeOutcome {
    /// A solution `y`, verified by exact substitution in the field.
    Solved(AlgElem),
    /// **Proved**: no `y ∈ ℚ(x)(α)` satisfies the equation.  The denominator and
    /// degree bounds used were both established complete and the resulting
    /// `ℚ`-linear system is inconsistent.  This — and only this — may be
    /// reported as `NonElementary`.
    NoRationalSolution,
    /// Nothing may be concluded; see [`RdeDecline`].
    Declined(RdeDecline),
}

impl AlgRdeOutcome {
    /// The solution, if one was found and verified.
    pub(crate) fn solution(self) -> Option<AlgElem> {
        match self {
            AlgRdeOutcome::Solved(y) => Some(y),
            _ => None,
        }
    }

    /// Whether this outcome is a *proof* that no solution exists — the only
    /// predicate that may license a non-elementarity certificate.
    ///
    /// Call sites match on the variants directly (so the compiler forces them to
    /// handle `Declined`); this predicate exists for assertions.
    #[allow(dead_code)]
    pub(crate) fn proves_no_solution(&self) -> bool {
        matches!(self, AlgRdeOutcome::NoRationalSolution)
    }

    /// Whether this outcome is a decline (nothing may be concluded).
    #[allow(dead_code)]
    pub(crate) fn is_declined(&self) -> bool {
        matches!(self, AlgRdeOutcome::Declined(_))
    }
}

/// What the caller has **proved** about the minimal polynomial of `α`.
///
/// `AlgExtension` is the quotient *ring* `ℚ(x)[y]/(q)`.  Only when `q` is
/// irreducible is that ring the field `ℚ(x)(α)` the integrand lives in, and only
/// then does "the ansatz system is inconsistent" say anything about the field.
/// `Unknown` therefore forces a decline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MinPolyStatus {
    /// The caller has established `q` irreducible over `ℚ(x)`.
    ProvedIrreducible,
    /// Not established — no non-existence claim may be made.
    Unknown,
}

// ---------------------------------------------------------------------------
// Heuristic search ceiling
// ---------------------------------------------------------------------------

/// A sound upper bound (Bronstein §6.5, algebraic level) on the `x`-degree of the
/// per-component numerator `pⱼ(x)` in the ansatz `y = Σⱼ (pⱼ/Den) αʲ` for a
/// solution of `D(y) + f·y = g` over `ℚ(x)(α)`.  Driven by the largest `x`-degree
/// occurring across the components of `f`, `g`, and the basis derivatives
/// `D(αʲ)`, plus a small slack.  It is used only as a *search ceiling* (the
/// caller takes `max(DEG_CAP, this)`), so over-estimating merely widens the
/// search; verification gates correctness.
///
/// **This is a heuristic ceiling, not a proved bound** — its failure to find a
/// solution proves nothing.  The proved bounds live in [`system_infinity_bound`]
/// and [`radical_infinity_bound`].
pub(crate) fn alg_x_degree_bound(e: &AlgExtension, f: &AlgElem, g: &AlgElem) -> usize {
    let comp_xdeg = |el: &AlgElem| -> i64 {
        el.iter()
            .map(|c| degree(c.numer()).max(degree(c.denom())))
            .max()
            .unwrap_or(0)
    };
    let mut base = comp_xdeg(f).max(comp_xdeg(g));
    let d = e.degree() as usize;
    let gen = e.generator();
    for j in 0..d {
        if let Some(aj) = e.pow(&gen, j as i64) {
            base = base.max(comp_xdeg(&e.derivation(&aj)));
        }
    }
    (base.max(0) + 2) as usize
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Solve `D(y) + f·y = g` for `y ∈ ℚ(x)(α)` with a **base scalar** `f ∈ ℚ(x)`,
/// returning the three-valued [`AlgRdeOutcome`].
///
/// Thin wrapper over [`solve_alg_rde_general_checked`] that lifts the base scalar
/// `f` to the constant extension element `e.constant(f)`.
pub(crate) fn solve_alg_rde_checked(
    e: &AlgExtension,
    f: &RatFn,
    g: &AlgElem,
    minpoly: MinPolyStatus,
) -> AlgRdeOutcome {
    solve_alg_rde_general_checked(e, &e.constant(f.clone()), g, minpoly)
}

/// Solve `D(y) + f·y = g` for `y ∈ ℚ(x)(α)` with a **general** `f ∈ ℚ(x)(α)`
/// (possibly carrying `α`-powers — the *non-diagonal* coupled case), returning
/// the three-valued [`AlgRdeOutcome`].
///
/// Two phases, in order:
///
/// 1. The historical heuristic ansatz search (candidate denominators × a degree
///    ceiling).  A hit is exactly verified in the field and returned as
///    [`AlgRdeOutcome::Solved`] — this phase is behaviourally identical to the
///    old solver, so nothing that used to solve stops solving.
/// 2. If it found nothing, an attempt to **prove** non-existence: build the
///    coupled system `b′ + M·b = c`, establish a complete denominator bound and
///    a complete degree bound (module docs §2 and §3), and solve *that* system.
///    Inconsistent ⇒ [`AlgRdeOutcome::NoRationalSolution`].  Any step of the
///    proof that cannot be discharged ⇒ [`AlgRdeOutcome::Declined`].
pub(crate) fn solve_alg_rde_general_checked(
    e: &AlgExtension,
    f: &AlgElem,
    g: &AlgElem,
    minpoly: MinPolyStatus,
) -> AlgRdeOutcome {
    let d = e.degree();
    if d <= 0 {
        return AlgRdeOutcome::Declined(RdeDecline::MalformedInput);
    }
    let d = d as usize;

    // Phase 1 — heuristic search (unchanged positive direction).
    let cap = alg_x_degree_bound(e, f, g).clamp(DEG_CAP, X_DEG_SANITY_CAP);
    let dens = candidate_denominators(e, f, g, d);
    for den in &dens {
        for ncap in 0..=cap {
            if let DenSolve::Solved(y) = solve_with_denominator_outcome(e, f, g, den, ncap, d) {
                return AlgRdeOutcome::Solved(y);
            }
        }
    }

    // Phase 2 — try to prove there is none.
    prove_no_solution(e, f, g, d, minpoly)
}

/// Solve `D(y) + f·y = g` for `y ∈ ℚ(x)(α)` with a **base scalar** `f ∈ ℚ(x)`,
/// or `None`.
///
/// # `None` is not a proof
///
/// This two-valued shim cannot distinguish *"no solution exists in `ℚ(x)(α)`"*
/// from *"the ansatz search came up empty"*.  Turning it into a non-elementarity
/// certificate is a soundness bug; use [`solve_alg_rde_checked`] and match on
/// [`AlgRdeOutcome`] whenever the answer feeds a verdict.
///
/// Retained for API compatibility and for the module's own tests; every
/// production call site now takes the checked form.
#[allow(dead_code)]
pub(crate) fn solve_alg_rde(e: &AlgExtension, f: &RatFn, g: &AlgElem) -> Option<AlgElem> {
    solve_alg_rde_checked(e, f, g, MinPolyStatus::Unknown).solution()
}

/// Solve `D(y) + f·y = g` for `y ∈ ℚ(x)(α)` with a **general** `f ∈ ℚ(x)(α)`, or
/// `None`.
///
/// # `None` is not a proof
///
/// Same caveat as [`solve_alg_rde`]: use [`solve_alg_rde_general_checked`] when
/// the answer feeds a verdict.
pub(crate) fn solve_alg_rde_general(e: &AlgExtension, f: &AlgElem, g: &AlgElem) -> Option<AlgElem> {
    solve_alg_rde_general_checked(e, f, g, MinPolyStatus::Unknown).solution()
}

// ---------------------------------------------------------------------------
// Phase 2: the non-existence proof
// ---------------------------------------------------------------------------

/// Try to establish that `D(y) + f·y = g` has **no** solution in `ℚ(x)(α)`.
///
/// Returns [`AlgRdeOutcome::NoRationalSolution`] only when every premise below is
/// discharged; otherwise a [`RdeDecline`] naming the premise that failed.
fn prove_no_solution(
    e: &AlgExtension,
    f: &AlgElem,
    g: &AlgElem,
    d: usize,
    minpoly: MinPolyStatus,
) -> AlgRdeOutcome {
    // Premise 1: the coefficient ring is the field the integrand lives in.
    if minpoly != MinPolyStatus::ProvedIrreducible {
        return AlgRdeOutcome::Declined(RdeDecline::AlgebraicFieldNotProved);
    }
    if d > MAX_PROVED_EXT_DEGREE {
        return AlgRdeOutcome::Declined(RdeDecline::AlgebraicBoundNotProved);
    }
    let Some(m) = system_matrix(e, f, d) else {
        return AlgRdeOutcome::Declined(RdeDecline::AlgebraicBoundNotProved);
    };
    let c: Vec<RatFn> = (0..d)
        .map(|k| g.get(k).cloned().unwrap_or_else(|| RatFn::int(0)))
        .collect();

    // Premise 2: a complete denominator bound.
    let den_star = match system_denominator_bound(&m, &c, d) {
        Ok(q) => q,
        Err(reason) => return AlgRdeOutcome::Declined(reason),
    };

    // Premise 3: a complete degree bound.  Two independent arguments; take the
    // tighter when both apply, and decline only when neither does.
    let via_matrix = system_infinity_bound(&m, &c, d);
    let via_radical = radical_infinity_bound(e, f, g);
    let n_bound = match (via_matrix, via_radical) {
        (Ok(a), Some(b)) => a.min(b),
        (Ok(a), None) => a,
        (Err(_), Some(b)) => b,
        (Err(reason), None) => return AlgRdeOutcome::Declined(reason),
    };

    // deg pⱼ = deg_∞ bⱼ + deg Den*.
    let ncap = (n_bound + degree(&den_star).max(0)).max(0) as usize;
    if ncap.saturating_add(1).saturating_mul(d) > MAX_ALG_RDE_UNKNOWNS {
        return AlgRdeOutcome::Declined(RdeDecline::AnsatzTooLarge);
    }

    match solve_with_denominator_outcome(e, f, g, &den_star, ncap, d) {
        DenSolve::Solved(y) => AlgRdeOutcome::Solved(y),
        // Both bounds are complete, so this ansatz contains *every* solution: an
        // inconsistent system is a genuine non-existence proof.
        DenSolve::Inconsistent => AlgRdeOutcome::NoRationalSolution,
        // Unreachable barring a bug: a consistent faithful system must yield a
        // solution.  Surfaced as a decline so it can never become a certificate.
        DenSolve::VerificationFailed => AlgRdeOutcome::Declined(RdeDecline::VerificationFailed),
    }
}

/// The coupled system matrix `M[k][j]` = the `αᵏ`-component of `D(αʲ) + f·αʲ`.
///
/// With `y = Σⱼ bⱼαʲ`, `D(y) + f·y = Σⱼ bⱼ′αʲ + Σⱼ bⱼ·(D(αʲ) + f·αʲ)`, so the
/// `αᵏ`-component of the equation is `bₖ′ + Σⱼ M[k][j]·bⱼ = cₖ`.
fn system_matrix(e: &AlgExtension, f: &AlgElem, d: usize) -> Option<Vec<Vec<RatFn>>> {
    let gen = e.generator();
    let mut m = vec![vec![RatFn::int(0); d]; d];
    for j in 0..d {
        let aj = e.pow(&gen, j as i64)?;
        let col = e.add(&e.derivation(&aj), &e.mul(f, &aj));
        for (k, row) in m.iter_mut().enumerate() {
            if let Some(r) = col.get(k) {
                row[j] = r.clone();
            }
        }
    }
    Some(m)
}

/// A **proved-complete** denominator bound for `b` in `b′ + M·b = c`.
///
/// See the module docs §2.  Returns a polynomial `Den*` such that every rational
/// solution can be written `b = p/Den*` with `p` a polynomial vector, or a
/// decline naming the premise that failed.
fn system_denominator_bound(m: &[Vec<RatFn>], c: &[RatFn], d: usize) -> Result<QPoly, RdeDecline> {
    // lcm of the reduced denominators of M and of c.
    let mut dm = poly_one();
    for row in m {
        for r in row {
            dm = poly_lcm(&dm, r.denom());
        }
    }
    let dm = poly_monic(&dm);
    let mut dc = poly_one();
    for r in c {
        dc = poly_lcm(&dc, r.denom());
    }
    let dc = poly_monic(&dc);

    // Poles forced by c alone: order ≤ μ−1, i.e. Dc/rad(Dc) = gcd(Dc, Dc′).
    let e_part = poly_gcd(&dc, &poly_deriv(&dc));

    if degree(&dm) < 1 {
        return Ok(poly_monic(&e_part)); // M is pole-free: no resonance
    }
    // The residue argument needs *at most simple* finite poles of M.
    if degree(&poly_gcd(&dm, &poly_deriv(&dm))) >= 1 {
        return Err(RdeDecline::AlgebraicBoundNotProved);
    }

    let dmp = poly_deriv(&dm);
    // Mnum = Dm·M is polynomial because Dm is the lcm of every entry denominator.
    let mut mnum = vec![vec![poly_zero(); d]; d];
    for (k, row) in mnum.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = poly_mul(m[k][j].numer(), &poly_div_exact(&dm, m[k][j].denom()));
        }
    }

    // R = Mnum·(Dm′)⁻¹ over A = ℚ[x]/(Dm); Dm squarefree ⇒ Dm′ is a unit there.
    let Some(inv) = poly_inverse_mod(&dmp, &dm) else {
        return Err(RdeDecline::VerificationFailed);
    };
    let mut rmat = vec![vec![poly_zero(); d]; d];
    for (k, row) in rmat.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = poly_divrem(&poly_mul(&mnum[k][j], &inv), &dm).1;
        }
    }

    // Every eigenvalue of R at every root of Dm is an eigenvalue of the rational
    // matrix representing R on ℚ[x]/(Dm); bound its spectral radius.
    let bound = residue_spectral_bound(&rmat, &dm, d);
    let k_max = bound.floor().numer().to_i64().unwrap_or(i64::MAX);
    if k_max > MAX_ALG_RESONANCE_SEARCH {
        return Err(RdeDecline::ResonanceSearchTooLarge);
    }

    let mut q = poly_one();
    for k in 1..=k_max.max(0) {
        // k is an eigenvalue of R at a root of Dm iff Dm shares a root with
        // det(Mnum − k·Dm′·I).
        let kd = poly_scale(&dmp, &Rational::from(k));
        let mut probe = mnum.clone();
        for (i, row) in probe.iter_mut().enumerate() {
            row[i] = poly_sub(&row[i], &kd);
        }
        let theta = poly_matrix_det(&probe, d);
        let gk = poly_gcd(&dm, &theta);
        if degree(&gk) >= 1 {
            let Ok(kk) = u32::try_from(k) else {
                return Err(RdeDecline::ResonanceSearchTooLarge);
            };
            q = poly_mul(&q, &poly_pow(&gk, kk));
            // A degenerate `Θ_k ≡ 0` makes every root resonate at every `k`, and
            // `q` would grow quadratically in `k_max`.  The ansatz is capped
            // anyway, so stop before paying for a system we would refuse.
            if degree(&q) > MAX_ALG_RDE_UNKNOWNS as i64 {
                return Err(RdeDecline::AnsatzTooLarge);
            }
        }
    }
    Ok(poly_monic(&poly_mul(&e_part, &q)))
}

/// Certified ceiling on `|λ|` over every eigenvalue `λ` of the residue matrix `R`
/// at every root of `Dm`.
///
/// With `Dm` squarefree, `ℚ[x]/(Dm) ⊗ ℚ̄ ≅ ℚ̄^{deg Dm}` by evaluation at the
/// roots, under which the `ℚ`-matrix representing `R` on `(ℚ[x]/(Dm))^d` becomes
/// block-diagonal with the `R(x₀)` as blocks.  So its spectrum is exactly the
/// union of theirs, and `ρ(M) ≤ min(‖M‖_∞, ‖M‖_1)` bounds all of them at once.
fn residue_spectral_bound(rmat: &[Vec<QPoly>], dm: &QPoly, d: usize) -> Rational {
    let dd = degree(dm).max(0) as usize;
    if dd == 0 {
        return Rational::from(0);
    }
    let n = d * dd;
    let mut rows = vec![Rational::from(0); n];
    let mut cols = vec![Rational::from(0); n];
    for (k, row) in rmat.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            // Columns of the multiplication-by-`entry` matrix on ℚ[x]/(Dm).
            let mut col = poly_divrem(entry, dm).1;
            for l in 0..dd {
                for (i, cell) in col.iter().enumerate() {
                    if i < dd {
                        let a = cell.clone().abs();
                        rows[k * dd + i] += a.clone();
                        cols[j * dd + l] += a;
                    }
                }
                let mut shifted = vec![Rational::from(0)];
                shifted.extend(col.iter().cloned());
                col = poly_divrem(&trim(shifted), dm).1;
            }
        }
    }
    let max = |v: Vec<Rational>| {
        v.into_iter()
            .fold(Rational::from(0), |m, r| if r > m { r } else { m })
    };
    let (r, c) = (max(rows), max(cols));
    if r < c {
        r
    } else {
        c
    }
}

/// A **proved-complete** bound on `n = maxⱼ deg_∞ bⱼ` from the matrix argument at
/// infinity (module docs §3, first bullet), searched over integer *shearings*.
///
/// The bare argument needs the leading matrix `M_ρ` to be invertible, and it is
/// routinely not: the power-basis components `bⱼαʲ` carry the *weights*
/// `j·deg_∞ α`, so the entries of `M` are staggered and only one anti-diagonal
/// survives at `x^ρ`.  Substituting `bⱼ = x^{−sⱼ}·b̃ⱼ` for integer weights `sⱼ`
/// (a gauge transformation, exact for **any** `sⱼ`) restaggers them:
///
/// ```text
///   M̃[k][j] = M[k][j]·x^{sₖ−sⱼ} − δ_{kj}·sₖ/x,    c̃ₖ = cₖ·x^{sₖ}.
/// ```
///
/// Every weight vector gives an independently valid bound
/// `deg_∞ bⱼ ≤ ñ − sⱼ ≤ ñ` (as `s₀ = 0` and the `sⱼ` are non-decreasing), so
/// trying several and keeping the tightest is sound.  `sⱼ = j·w` for small `w`
/// is exactly what an `α` of `∞`-degree `w` needs — e.g. `∫ exp(√(x²+1)) dx`,
/// where `α` has degree 1 and `w = 1` turns a singular `M_ρ` into `[[0,1],[1,0]]`.
fn system_infinity_bound(m: &[Vec<RatFn>], c: &[RatFn], d: usize) -> Result<i64, RdeDecline> {
    let mut best: Option<i64> = None;
    let mut last_err = RdeDecline::AlgebraicBoundNotProved;
    for w in 0..=MAX_SHEAR_WEIGHT {
        let s: Vec<i64> = (0..d).map(|j| j as i64 * w).collect();
        let (ms, cs) = shear_system(m, c, d, &s);
        match unsheared_infinity_bound(&ms, &cs, d) {
            // deg_∞ bⱼ ≤ ñ − sⱼ ≤ ñ because s₀ = 0 ≤ sⱼ.
            Ok(nt) => best = Some(best.map_or(nt, |b: i64| b.min(nt))),
            Err(reason) => last_err = reason,
        }
    }
    best.ok_or(last_err)
}

/// Apply the gauge transformation `bⱼ = x^{−sⱼ}·b̃ⱼ` to `b′ + M·b = c`.
fn shear_system(
    m: &[Vec<RatFn>],
    c: &[RatFn],
    d: usize,
    s: &[i64],
) -> (Vec<Vec<RatFn>>, Vec<RatFn>) {
    // r · x^k for any integer k.
    let shift = |r: &RatFn, k: i64| -> RatFn {
        if k >= 0 {
            RatFn::new(poly_mul(r.numer(), &x_pow(k as usize)), r.denom().clone())
        } else {
            RatFn::new(
                r.numer().clone(),
                poly_mul(r.denom(), &x_pow((-k) as usize)),
            )
        }
    };
    let mut ms = vec![vec![RatFn::int(0); d]; d];
    for (k, row) in ms.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = shift(&m[k][j], s[k] - s[j]);
            if k == j && s[k] != 0 {
                // subtract sₖ/x : (num·x − sₖ·den) / (den·x)
                let den_x = poly_mul(cell.denom(), &x_pow(1));
                let num = poly_sub(
                    &poly_mul(cell.numer(), &x_pow(1)),
                    &poly_scale(cell.denom(), &Rational::from(s[k])),
                );
                *cell = RatFn::new(num, den_x);
            }
        }
    }
    let cs: Vec<RatFn> = (0..d).map(|k| shift(&c[k], s[k])).collect();
    (ms, cs)
}

/// The un-sheared matrix argument at infinity — see [`system_infinity_bound`].
fn unsheared_infinity_bound(m: &[Vec<RatFn>], c: &[RatFn], d: usize) -> Result<i64, RdeDecline> {
    let Some(gamma) = c.iter().filter_map(ratfn_deg_inf).max() else {
        // c = 0.  Then y = 0 solves and phase 1 already returned; be conservative.
        return Ok(0);
    };
    let Some(rho) = m.iter().flatten().filter_map(ratfn_deg_inf).max() else {
        return Ok((gamma + 1).max(0)); // M = 0: b′ = c
    };
    if rho <= -2 {
        return Ok((gamma + 1).max(0)); // b′ strictly dominates M·b
    }

    // Leading coefficient matrix at x^ρ.
    let mut lead = vec![vec![Rational::from(0); d]; d];
    for (k, row) in lead.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if ratfn_deg_inf(&m[k][j]) == Some(rho) {
                *cell = ratfn_lead(&m[k][j]);
            }
        }
    }

    if rho >= 0 {
        // M·b strictly dominates b′ *provided* M_ρ·B ≠ 0 for every B ≠ 0.
        if qmat_det(&lead, d) == 0 {
            return Err(RdeDecline::AlgebraicBoundNotProved);
        }
        return Ok(gamma - rho);
    }

    // ρ = −1: b′ and M·b balance; the non-resonant case gives n ≤ γ+1 and the
    // resonance needs −n ∈ spec(M₋₁).
    let bound = qmat_norm_bound(&lead, d);
    let k_max = bound.floor().numer().to_i64().unwrap_or(i64::MAX);
    if k_max > MAX_ALG_RESONANCE_SEARCH {
        return Err(RdeDecline::ResonanceSearchTooLarge);
    }
    let mut best = (gamma + 1).max(0);
    for k in 1..=k_max.max(0) {
        let mut a = lead.clone();
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += Rational::from(k);
        }
        if qmat_det(&a, d) == 0 {
            best = best.max(k);
        }
    }
    Ok(best)
}

/// A **proved-complete** bound on `n = maxⱼ deg_∞ bⱼ` for a *pure radical*
/// `αⁿ = p` that is **totally ramified at infinity** (`gcd(n, deg p) = 1`), from
/// the valuation argument (module docs §3, second bullet).
///
/// `None` when the shape does not apply or the resonant case `V(f) = −n` is hit.
fn radical_infinity_bound(e: &AlgExtension, f: &AlgElem, g: &AlgElem) -> Option<i64> {
    let (n, p) = radical_shape(e)?;
    let deg_p = degree(&p);
    if deg_p < 1 {
        return None;
    }
    let n_i = n as i64;
    if gcd_i64(n_i, deg_p) != 1 {
        return None; // more than one place above ∞: components can cancel
    }
    // V(Σ hⱼαʲ) = maxⱼ (n·deg_∞ hⱼ + j·deg p), attained at a unique j.
    let vval = |el: &AlgElem| -> Option<i64> {
        (0..n)
            .filter_map(|j| {
                el.get(j)
                    .and_then(ratfn_deg_inf)
                    .map(|dg| n_i * dg + j as i64 * deg_p)
            })
            .max()
    };
    let gamma = vval(g)?; // g = 0 was already solved in phase 1
    let y_bound = match vval(f) {
        // `D(y)` strictly dominates, giving `V(y) = Γ + n` — *unless* `V(y) = 0`,
        // where `y` is a unit at the place and `V(D y)` is only bounded above.
        // `0` must therefore stay in the bound.
        None => (gamma + n_i).max(0), // f = 0: D(y) = g
        // `f·y` strictly dominates for *every* `V(y)` including 0, so this one is
        // an equality and needs no floor.
        Some(phi) if phi > -n_i => gamma - phi,
        Some(phi) if phi < -n_i => (gamma + n_i).max(0),
        Some(_) => return None, // V(f) = −n: resonance, unhandled
    };
    // n·deg_∞ bⱼ ≤ V(y) − j·deg p ≤ V(y).
    Some(y_bound.div_euclid(n_i))
}

/// Recognise `ℚ(x)[y]/(yⁿ − p)`: returns `(n, p)`.
fn radical_shape(e: &AlgExtension) -> Option<(usize, QPoly)> {
    let n = e.degree();
    if n < 2 {
        return None;
    }
    let n = n as usize;
    let modulus = e.quotient().modulus();
    if modulus.len() != n + 1 || modulus[n] != RatFn::int(1) {
        return None;
    }
    for m in modulus.iter().take(n).skip(1) {
        if !m.numer().is_empty() {
            return None;
        }
    }
    if degree(modulus[0].denom()) != 0 {
        return None;
    }
    Some((n, poly_scale(modulus[0].numer(), &Rational::from(-1))))
}

// ---------------------------------------------------------------------------
// Irreducibility witnesses for the shapes the integrator actually builds
// ---------------------------------------------------------------------------

/// Is `yⁿ − p` irreducible over `ℚ(x)`?
///
/// `yⁿ − a` is irreducible over a field `F` iff `a ∉ Fˡ` for every prime `l | n`
/// and, when `4 | n`, `a ∉ −4·F⁴` (Lang, *Algebra* VI §9).  Two cheap sufficient
/// conditions cover every radicand the integrator builds:
///
/// * `p` **squarefree** with `deg p ≥ 1` — then `p = c·hˡ` forces `deg h = 0`,
///   i.e. `p` constant; and `−4g⁴` is never squarefree.
/// * `gcd(n, deg p) = 1` with `deg p ≥ 1` — then no prime `l | n` divides
///   `deg p = l·deg h`, and `4 | n` would force `2 | deg p`.
pub(crate) fn radical_minpoly_status(n: usize, p: &QPoly) -> MinPolyStatus {
    let deg_p = degree(p);
    if n < 2 || deg_p < 1 {
        return MinPolyStatus::Unknown;
    }
    let squarefree = degree(&poly_gcd(p, &poly_deriv(p))) < 1;
    if squarefree || gcd_i64(n as i64, deg_p) == 1 {
        MinPolyStatus::ProvedIrreducible
    } else {
        MinPolyStatus::Unknown
    }
}

/// Is the minimal polynomial `α⁴ − 2(p+q)α² + (p−q)²` of `α = √p + √q`
/// irreducible over `ℚ(x)`?
///
/// `[ℚ(x)(√p, √q) : ℚ(x)] = 4` exactly when none of `p`, `q`, `p·q` is a square in
/// `ℚ(x)`, and in characteristic ≠ 2 the sum `√p + √q` is then a primitive
/// element — so the degree-4 polynomial above *is* its minimal polynomial.
pub(crate) fn compositum_minpoly_status(p: &QPoly, q: &QPoly) -> MinPolyStatus {
    if !is_square_in_qx(p) && !is_square_in_qx(q) && !is_square_in_qx(&poly_mul(p, q)) {
        MinPolyStatus::ProvedIrreducible
    } else {
        MinPolyStatus::Unknown
    }
}

/// Is the minimal polynomial `α⁴ − 2a·α² + (a² − b)` of `α = √(a + √b)`
/// irreducible over `ℚ(x)`?
///
/// `a + √b` is a square `(u + v√b)²` in `ℚ(x)(√b)` iff `2uv = 1` and
/// `u² + v²b = a`, i.e. `4u⁴ − 4a·u² + b = 0`, i.e. `u² = (a ± √(a²−b))/2` — which
/// needs `a² − b` to be a square in `ℚ(x)`.  So `b` not a square (making
/// `ℚ(x)(√b)` quadratic) together with `a² − b` not a square is sufficient for
/// `[ℚ(x)(α) : ℚ(x)] = 4`.
pub(crate) fn nested_radical_minpoly_status(a: &QPoly, b: &QPoly) -> MinPolyStatus {
    let disc = poly_sub(&poly_mul(a, a), b);
    if !is_square_in_qx(b) && !is_square_in_qx(&disc) {
        MinPolyStatus::ProvedIrreducible
    } else {
        MinPolyStatus::Unknown
    }
}

/// Is `p` the square of an element of `ℚ(x)`?
///
/// `ℚ[x]` is integrally closed in `ℚ(x)`, so a square root of a polynomial is
/// itself a polynomial: `p = (e·h)²` with `h` monic and `e ∈ ℚ`.  Hence `lc(p)`
/// must be a rational square and `p/lc(p)` a monic polynomial square.
fn is_square_in_qx(p: &QPoly) -> bool {
    let dp = degree(p);
    if dp < 0 {
        return true; // 0 = 0²
    }
    let lc = p[dp as usize].clone();
    if !is_rational_square(&lc) {
        return false;
    }
    if dp == 0 {
        return true;
    }
    let inv = Rational::from(1) / lc;
    poly_sqrt_monic(&poly_scale(p, &inv)).is_some()
}

/// Is `r` the square of a rational?
fn is_rational_square(r: &Rational) -> bool {
    *r >= 0 && r.numer().is_perfect_square() && r.denom().is_perfect_square()
}

/// The monic `h` with `h² = p` for a **monic** `p`, or `None`.
fn poly_sqrt_monic(p: &QPoly) -> Option<QPoly> {
    let dp = degree(p);
    if dp < 0 || dp % 2 != 0 {
        return None;
    }
    let m = (dp / 2) as usize;
    let mut h = vec![Rational::from(0); m + 1];
    h[m] = Rational::from(1);
    // Coefficient of x^{m+i} in h² is 2·h_m·h_i + Σ_{j+k=m+i, i<j,k<m} h_j·h_k.
    for i in (0..m).rev() {
        let mut s = Rational::from(0);
        for j in (i + 1)..m {
            let k = m + i - j;
            if k > i && k < m {
                s += h[j].clone() * h[k].clone();
            }
        }
        let target = p.get(m + i).cloned().unwrap_or_else(|| Rational::from(0));
        h[i] = (target - s) / Rational::from(2);
    }
    if trim(poly_mul(&h, &h)) == trim(p.clone()) {
        Some(h)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Small linear-algebra / polynomial helpers
// ---------------------------------------------------------------------------

/// `deg_∞ (num/den) = deg num − deg den`; `None` for the zero function.
fn ratfn_deg_inf(r: &RatFn) -> Option<i64> {
    if r.numer().is_empty() {
        None
    } else {
        Some(degree(r.numer()) - degree(r.denom()))
    }
}

/// The coefficient of `x^{deg_∞ r}` in the expansion of `r` at infinity.
fn ratfn_lead(r: &RatFn) -> Rational {
    let dn = degree(r.numer());
    let dd = degree(r.denom());
    if dn < 0 || dd < 0 {
        return Rational::from(0);
    }
    r.numer()[dn as usize].clone() / r.denom()[dd as usize].clone()
}

/// Determinant of a `d × d` rational matrix by Gaussian elimination.
fn qmat_det(m: &[Vec<Rational>], d: usize) -> Rational {
    let mut a: Vec<Vec<Rational>> = m.to_vec();
    let mut det = Rational::from(1);
    for col in 0..d {
        let Some(piv) = (col..d).find(|&r| a[r][col] != 0) else {
            return Rational::from(0);
        };
        if piv != col {
            a.swap(piv, col);
            det = -det;
        }
        det *= a[col][col].clone();
        let inv = Rational::from(1) / a[col][col].clone();
        for r in (col + 1)..d {
            if a[r][col] == 0 {
                continue;
            }
            let factor = a[r][col].clone() * inv.clone();
            let pivot_row: Vec<Rational> = a[col][col..d].to_vec();
            for (dst, pv) in a[r][col..d].iter_mut().zip(pivot_row.iter()) {
                *dst -= pv.clone() * factor.clone();
            }
        }
    }
    det
}

/// `min(‖m‖_∞, ‖m‖_1)` — a certified ceiling on the spectral radius of `m`.
fn qmat_norm_bound(m: &[Vec<Rational>], d: usize) -> Rational {
    let mut rows = vec![Rational::from(0); d];
    let mut cols = vec![Rational::from(0); d];
    for (i, row) in m.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            let a = cell.clone().abs();
            rows[i] += a.clone();
            cols[j] += a;
        }
    }
    let max = |v: Vec<Rational>| {
        v.into_iter()
            .fold(Rational::from(0), |m, r| if r > m { r } else { m })
    };
    let (r, c) = (max(rows), max(cols));
    if r < c {
        r
    } else {
        c
    }
}

/// Determinant of a `d × d` matrix over `ℚ[x]`, by exact evaluation and
/// Lagrange interpolation.
///
/// `deg det ≤ Σᵢ maxⱼ deg m[i][j]` (every Leibniz term is one entry per row), so
/// evaluating at that many `+1` distinct rationals and interpolating recovers it
/// exactly.  Chosen over fraction-free elimination because the result feeds a
/// **denominator bound**: a determinant that came out too small would shrink
/// `Den*` and could turn a decline into a false certificate, and interpolation
/// has no exact-division step to get wrong.
fn poly_matrix_det(m: &[Vec<QPoly>], d: usize) -> QPoly {
    if d == 0 {
        return poly_one();
    }
    let mut bound = 0i64;
    for row in m.iter().take(d) {
        let r = row.iter().take(d).map(degree).max().unwrap_or(-1);
        if r < 0 {
            return poly_zero(); // an all-zero row
        }
        bound += r;
    }
    let npts = bound.max(0) as usize + 1;
    let xs: Vec<Rational> = (0..npts).map(|i| Rational::from(i as i64)).collect();
    let ys: Vec<Rational> = xs
        .iter()
        .map(|t| {
            let ev: Vec<Vec<Rational>> = m
                .iter()
                .take(d)
                .map(|row| row.iter().take(d).map(|p| poly_eval(p, t)).collect())
                .collect();
            qmat_det(&ev, d)
        })
        .collect();
    lagrange_interpolate(&xs, &ys)
}

/// `p(t)` by Horner.
fn poly_eval(p: &QPoly, t: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in p.iter().rev() {
        acc = acc * t.clone() + c.clone();
    }
    acc
}

/// The unique polynomial of degree `< xs.len()` through `(xs, ys)`.
fn lagrange_interpolate(xs: &[Rational], ys: &[Rational]) -> QPoly {
    let n = xs.len();
    let mut acc = poly_zero();
    for i in 0..n {
        // basis_i = Π_{j≠i} (x − xⱼ) / (xᵢ − xⱼ)
        let mut num = poly_one();
        let mut den = Rational::from(1);
        for j in 0..n {
            if i == j {
                continue;
            }
            num = poly_mul(&num, &vec![-xs[j].clone(), Rational::from(1)]);
            den *= xs[i].clone() - xs[j].clone();
        }
        let scale = ys[i].clone() / den;
        acc = poly_add(&acc, &poly_scale(&num, &scale));
    }
    trim(acc)
}

/// `a⁻¹ mod m` over `ℚ[x]`, or `None` when `gcd(a, m)` is not a unit.
fn poly_inverse_mod(a: &QPoly, m: &QPoly) -> Option<QPoly> {
    let mut r0 = trim(m.clone());
    let mut r1 = poly_divrem(a, m).1;
    let mut t0 = poly_zero();
    let mut t1 = poly_one();
    while !r1.is_empty() {
        let (q, r) = poly_divrem(&r0, &r1);
        let t2 = poly_sub(&t0, &poly_mul(&q, &t1));
        r0 = r1;
        r1 = trim(r);
        t0 = t1;
        t1 = t2;
    }
    if degree(&r0) != 0 {
        return None;
    }
    let inv = Rational::from(1) / r0[0].clone();
    Some(poly_divrem(&poly_scale(&t0, &inv), m).1)
}

/// `gcd` of two `i64`s (by absolute value).
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

// ---------------------------------------------------------------------------
// The ansatz system
// ---------------------------------------------------------------------------

/// Candidate `x`-denominators for `y`, increasing in complexity: `1`, then the
/// LCM `B` of every `x`-denominator that appears in `D(αʲ)`, `f`, and `g`, then
/// `B²`, `B³`.  Over-clearing is harmless (the numerator ansatz just needs more
/// terms); verification guards correctness.
///
/// **Heuristic** — this list is not proved to contain the true denominator; that
/// is what [`system_denominator_bound`] is for.
fn candidate_denominators(e: &AlgExtension, f: &AlgElem, g: &AlgElem, d: usize) -> Vec<QPoly> {
    let mut base = poly_one();
    let gen = e.generator();
    for j in 0..d {
        if let Some(aj) = e.pow(&gen, j as i64) {
            for c in e.derivation(&aj) {
                base = poly_lcm(&base, c.denom());
            }
        }
    }
    // f may be a general extension element — LCM every component's denominator.
    for c in f {
        base = poly_lcm(&base, c.denom());
    }
    for c in g {
        base = poly_lcm(&base, c.denom());
    }
    let base2 = poly_mul(&base, &base);
    let base3 = poly_mul(&base2, &base);
    let mut out = vec![poly_one(), base, base2, base3];
    out.dedup_by(|a, b| trim(a.clone()) == trim(b.clone()));
    out
}

/// Outcome of one fixed-`(Den, ncap)` ansatz solve.
enum DenSolve {
    /// A solution, verified exactly in the field.
    Solved(AlgElem),
    /// The `ℚ`-linear system has no solution — no `y` of *this shape* exists.
    Inconsistent,
    /// The system was consistent but the reconstruction failed the exact field
    /// check.  Never a mathematical statement; see
    /// [`RdeDecline::VerificationFailed`].
    VerificationFailed,
}

/// Solve seeking `y = Σⱼ (pⱼ(x)/Den) αʲ` with `deg pⱼ ≤ ncap`, for the fixed `Den`.
fn solve_with_denominator_outcome(
    e: &AlgExtension,
    f: &AlgElem,
    g: &AlgElem,
    den: &QPoly,
    ncap: usize,
    d: usize,
) -> DenSolve {
    // Ansatz basis: component j carries numerator xᵐ over the common Den.
    let basis: Vec<(usize, usize)> = (0..d)
        .flat_map(|j| (0..=ncap).map(move |m| (j, m)))
        .collect();
    let elems: Vec<AlgElem> = basis
        .iter()
        .map(|&(j, m)| {
            let coeff = RatFn::new(x_pow(m), den.clone()); // xᵐ / Den
            let mut v = vec![RatFn::int(0); d];
            v[j] = coeff;
            e.reduce(&v)
        })
        .collect();

    // L(·) = D(·) + f·(·) applied to each basis element.  `f` is a general
    // extension element, so `e.mul(f, m)` mixes the power basis (the coupling).
    let cols: Vec<AlgElem> = elems
        .iter()
        .map(|m| e.add(&e.derivation(m), &e.mul(f, m)))
        .collect();

    let (matrix, rhs) = extract_linear_system(&cols, g, d);
    let Some(sol) = gauss_solve(matrix, rhs, basis.len()) else {
        return DenSolve::Inconsistent;
    };

    // Reconstruct y = Σ solᵢ · elemᵢ.
    let mut y = e.from_int(0);
    for (idx, elem) in elems.iter().enumerate() {
        if sol[idx] != 0 {
            let s = e.constant(RatFn::from_poly(&vec![sol[idx].clone()]));
            y = e.add(&y, &e.mul(&s, elem));
        }
    }

    // Exact verification: D(y) + f·y == g.
    let lhs = e.add(&e.derivation(&y), &e.mul(f, &y));
    if e.elem_eq(&lhs, g) {
        DenSolve::Solved(y)
    } else {
        DenSolve::VerificationFailed
    }
}

/// Solve seeking `y = Σⱼ (pⱼ(x)/Den) αʲ` with `deg pⱼ ≤ ncap`, or `None`.
///
/// `None` conflates "inconsistent system" with "verification failed"; it is kept
/// for the module's own tests.  Production paths use
/// [`solve_with_denominator_outcome`].
#[cfg(test)]
fn solve_with_denominator(
    e: &AlgExtension,
    f: &AlgElem,
    g: &AlgElem,
    den: &QPoly,
    ncap: usize,
    d: usize,
) -> Option<AlgElem> {
    match solve_with_denominator_outcome(e, f, g, den, ncap, d) {
        DenSolve::Solved(y) => Some(y),
        _ => None,
    }
}

/// Build the exact `ℚ`-linear system `Σᵢ uᵢ·colᵢ = target` by, for each power-basis
/// component `k`, clearing the common `x`-denominator of the `ℚ(x)` entries and
/// matching every `xᵐ` coefficient.
fn extract_linear_system(
    cols: &[AlgElem],
    target: &AlgElem,
    d: usize,
) -> (Vec<Vec<Rational>>, Vec<Rational>) {
    let comp =
        |a: &AlgElem, k: usize| -> RatFn { a.get(k).cloned().unwrap_or_else(|| RatFn::int(0)) };
    let mut matrix: Vec<Vec<Rational>> = Vec::new();
    let mut rhs: Vec<Rational> = Vec::new();

    for k in 0..d {
        let col_rf: Vec<RatFn> = cols.iter().map(|c| comp(c, k)).collect();
        let tgt_rf = comp(target, k);

        // Common x-denominator of this component across all columns and target.
        let mut d_x = poly_one();
        for r in &col_rf {
            d_x = poly_lcm(&d_x, r.denom());
        }
        d_x = poly_lcm(&d_x, tgt_rf.denom());

        let s_cols: Vec<QPoly> = col_rf
            .iter()
            .map(|r| poly_mul(r.numer(), &poly_div_exact(&d_x, r.denom())))
            .collect();
        let s_tgt = poly_mul(tgt_rf.numer(), &poly_div_exact(&d_x, tgt_rf.denom()));

        let max_m = s_cols
            .iter()
            .map(|s| s.len())
            .chain(std::iter::once(s_tgt.len()))
            .max()
            .unwrap_or(0);
        for m in 0..max_m {
            matrix.push(
                s_cols
                    .iter()
                    .map(|s| s.get(m).cloned().unwrap_or_else(|| Rational::from(0)))
                    .collect(),
            );
            rhs.push(s_tgt.get(m).cloned().unwrap_or_else(|| Rational::from(0)));
        }
    }
    (matrix, rhs)
}

/// Solve `M·x = b` over `ℚ` by Gauss–Jordan, returning a particular solution
/// (free variables set to 0) or `None` if inconsistent.
fn gauss_solve(
    mut m: Vec<Vec<Rational>>,
    mut b: Vec<Rational>,
    ncols: usize,
) -> Option<Vec<Rational>> {
    let nrows = m.len();
    let mut pivot_row_of_col: Vec<Option<usize>> = vec![None; ncols];
    let mut row = 0usize;
    for col in 0..ncols {
        if row >= nrows {
            break;
        }
        let Some(sel) = (row..nrows).find(|&r| m[r][col] != 0) else {
            continue;
        };
        m.swap(row, sel);
        b.swap(row, sel);
        let piv = m[row][col].clone();
        for v in m[row].iter_mut() {
            *v = v.clone() / piv.clone();
        }
        b[row] = b[row].clone() / piv.clone();
        let pivot_row = m[row].clone();
        let pivot_b = b[row].clone();
        for r in 0..nrows {
            if r != row && m[r][col] != 0 {
                let factor = m[r][col].clone();
                for (dst, pv) in m[r].iter_mut().zip(pivot_row.iter()) {
                    *dst -= factor.clone() * pv.clone();
                }
                b[r] -= factor * pivot_b.clone();
            }
        }
        pivot_row_of_col[col] = Some(row);
        row += 1;
    }
    for r in 0..nrows {
        if m[r].iter().all(|v| *v == 0) && b[r] != 0 {
            return None;
        }
    }
    let mut x = vec![Rational::from(0); ncols];
    for (col, pr) in pivot_row_of_col.iter().enumerate() {
        if let Some(r) = pr {
            x[col] = b[*r].clone();
        }
    }
    Some(x)
}

/// `lcm(a, b)` over `ℚ[x]` (non-monic is fine — used only as a clearing factor).
fn poly_lcm(a: &QPoly, b: &QPoly) -> QPoly {
    let g = poly_gcd(a, b);
    poly_div_exact(&poly_mul(a, b), &g)
}

/// The monomial `xᵐ` as a `ℚ[x]` polynomial.
fn x_pow(m: usize) -> QPoly {
    let mut p = vec![Rational::from(0); m + 1];
    p[m] = Rational::from(1);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::poly_rde::poly_scale;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    /// Cyclic sanity check: `α = √x` (`α² = x`), solve `D(y) = (3/2)·α`.  The
    /// antiderivative is `y = x·α = x^{3/2}` since `D(x^{3/2}) = (3/2)x^{1/2}`.
    #[test]
    fn cyclic_sqrt_recovers_solution() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
                                                                 // g = (3/2)·α  (= component vector [0, 3/2]).
        let g: AlgElem = vec![
            RatFn::int(0),
            RatFn::from_poly(&vec![Rational::from((3, 2))]),
        ];
        let f = RatFn::int(0);
        let y = solve_alg_rde(&e, &f, &g).expect("should solve");
        // D(y) must equal g.
        assert!(e.elem_eq(&e.derivation(&y), &g));
        // y = x·α.
        let expected: AlgElem = vec![RatFn::int(0), RatFn::from_poly(&vec![rat(0), rat(1)])];
        assert!(e.elem_eq(&y, &expected), "y = {:?}", y);
    }

    /// **Non-cyclic** (the M1-step-2 case): `α = √x + √(x+1)`, a degree-4
    /// extension whose minimal polynomial `α⁴ − 2(2x+1)α² + 1 = 0` is *not* a
    /// pure radical, so `D(α)` mixes the power basis and the system is coupled.
    /// We construct `g = D(α)` and confirm the solver recovers a `y` with
    /// `D(y) = g` (namely `y = α`).
    #[test]
    fn noncyclic_compositum_pure_antiderivative() {
        // q = α⁴ − 2(2x+1)α² + 1 : coeffs ascending [1, 0, −2(2x+1), 0, 1].
        let q: Vec<QPoly> = vec![
            poly_one(),                                  // 1
            Vec::new(),                                  // 0·α
            poly_scale(&vec![rat(1), rat(2)], &rat(-2)), // −2(2x+1) = −4x−2
            Vec::new(),                                  // 0·α³
            poly_one(),                                  // α⁴
        ];
        let e = AlgExtension::new(&q);
        assert_eq!(e.degree(), 4);

        let alpha = e.generator();
        let g = e.derivation(&alpha); // genuinely coupled element
        let f = RatFn::int(0);
        let y = solve_alg_rde(&e, &f, &g).expect("coupled RDE should solve");
        assert!(
            e.elem_eq(&e.derivation(&y), &g),
            "D(y) must equal g; y = {y:?}"
        );
    }

    /// Non-cyclic with a nonzero `f`: with `α = √x + √(x+1)` and `f = 1/x`,
    /// `g = D(α) + (1/x)·α` is solved by `y = α`.
    #[test]
    fn noncyclic_compositum_with_f() {
        let q: Vec<QPoly> = vec![
            poly_one(),
            Vec::new(),
            poly_scale(&vec![rat(1), rat(2)], &rat(-2)),
            Vec::new(),
            poly_one(),
        ];
        let e = AlgExtension::new(&q);
        let alpha = e.generator();
        let f = RatFn::new(poly_one(), vec![rat(0), rat(1)]); // 1/x
        let f_elem = e.constant(f.clone());
        let g = e.add(&e.derivation(&alpha), &e.mul(&f_elem, &alpha));
        let y = solve_alg_rde(&e, &f, &g).expect("coupled RDE with f should solve");
        let lhs = e.add(&e.derivation(&y), &e.mul(&f_elem, &y));
        assert!(e.elem_eq(&lhs, &g), "D(y)+f·y must equal g; y = {y:?}");
    }

    /// A target with **no** rational solution must return `None` (never a wrong
    /// antiderivative).  `g = 1/x` (embedded from `ℚ(x)`) has antiderivative
    /// `log x ∉ ℚ(x)(α)`, so no rational `y` solves `D(y) = 1/x`.
    #[test]
    fn unsolvable_log_returns_none() {
        let q: Vec<QPoly> = vec![
            poly_one(),
            Vec::new(),
            poly_scale(&vec![rat(1), rat(2)], &rat(-2)),
            Vec::new(),
            poly_one(),
        ];
        let e = AlgExtension::new(&q);
        // g = 1/x (constant element of ℚ(x)).
        let g = e.constant(RatFn::new(poly_one(), vec![rat(0), rat(1)]));
        let f = RatFn::int(0);
        assert!(solve_alg_rde(&e, &f, &g).is_none());
    }

    // -- Non-base (non-diagonal) f: the genuine coupled case --------------

    /// **Non-base `f` over `α = √x`.**  Take `f = 1/(2√x) = (1/(2x))·α` (a
    /// non-base extension element, the `∫exp(√x)` twist `D(√x)`) and `y = √x`.
    /// Then `g = D(y) + f·y = (1/(2x))·α + (1/(2x))·α·α = (1/(2x))·α + 1/2`.
    /// The generalized solver must recover a `y'` with `D(y')+f·y' = g`.
    #[test]
    fn nonbase_f_sqrt_coupled() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
        let alpha = e.generator();
        // f = (1/(2x))·α   (= 1/(2√x))
        let f: AlgElem = vec![RatFn::int(0), RatFn::new(poly_one(), vec![rat(0), rat(2)])];
        // y = √x = α
        let y_true = alpha.clone();
        let g = e.add(&e.derivation(&y_true), &e.mul(&f, &y_true));
        let y = solve_alg_rde_general(&e, &f, &g).expect("non-base f coupled solve");
        let lhs = e.add(&e.derivation(&y), &e.mul(&f, &y));
        assert!(e.elem_eq(&lhs, &g), "D(y)+f·y must equal g; y = {y:?}");
        // And the headline antiderivative is recovered.
        assert!(e.elem_eq(&y, &y_true), "expected y = √x; got {y:?}");
    }

    /// **Non-base `f` over `α = ∛x`.**  `f = (1/x)·α²` (a non-base element) and
    /// `y = x` (a base function, but the equation couples via `f`).
    /// `g = D(y) + f·y = 1 + (1/x)·α²·x = 1 + α²`.  Solver must recover it.
    #[test]
    fn nonbase_f_cbrt_coupled() {
        let e = AlgExtension::radical(3, &vec![rat(0), rat(1)]); // α³ = x
                                                                 // f = (1/x)·α²
        let f: AlgElem = vec![
            RatFn::int(0),
            RatFn::int(0),
            RatFn::new(poly_one(), vec![rat(0), rat(1)]),
        ];
        // y = x
        let y_true = e.constant(RatFn::from_poly(&vec![rat(0), rat(1)]));
        let g = e.add(&e.derivation(&y_true), &e.mul(&f, &y_true));
        let y = solve_alg_rde_general(&e, &f, &g).expect("non-base cbrt f coupled solve");
        let lhs = e.add(&e.derivation(&y), &e.mul(&f, &y));
        assert!(e.elem_eq(&lhs, &g), "D(y)+f·y must equal g; y = {y:?}");
        assert!(e.elem_eq(&y, &y_true), "expected y = x; got {y:?}");
    }

    /// **Wrapper equivalence / regression.**  For a base scalar `f`, the thin
    /// `solve_alg_rde(e, &f, g)` wrapper must return exactly what
    /// `solve_alg_rde_general(e, &e.constant(f), g)` returns.
    #[test]
    fn base_f_wrapper_matches_general() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
        let alpha = e.generator();
        let f = RatFn::new(poly_one(), vec![rat(0), rat(1)]); // 1/x ∈ base
        let f_elem = e.constant(f.clone());
        let g = e.add(&e.derivation(&alpha), &e.mul(&f_elem, &alpha));
        let via_wrapper = solve_alg_rde(&e, &f, &g);
        let via_general = solve_alg_rde_general(&e, &f_elem, &g);
        match (&via_wrapper, &via_general) {
            (Some(a), Some(b)) => assert!(e.elem_eq(a, b), "wrapper vs general differ"),
            (None, None) => {}
            _ => panic!("wrapper vs general disagree on solvability"),
        }
        assert!(via_wrapper.is_some(), "base-f case should solve");
    }

    /// **Unsolvable non-base `f`.**  With `f = α` (non-base) and `g = 1/x`
    /// (whose antiderivative would be `log x ∉ ℚ(x)(α)`), there is no rational
    /// `y` — the generalized solver must return `None`, never a wrong answer.
    #[test]
    fn nonbase_f_unsolvable_returns_none() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
        let f = e.generator(); // α (non-base)
        let g = e.constant(RatFn::new(poly_one(), vec![rat(0), rat(1)])); // 1/x
        assert!(solve_alg_rde_general(&e, &f, &g).is_none());
    }

    /// **Degree-bound polymorphism demonstration.**  The true solution has a
    /// numerator of `x`-degree 8 — *beyond* the historical fixed `DEG_CAP = 6`,
    /// so the old fixed-cap search would miss it.  The analytic Bronstein §6.5
    /// bound ([`alg_x_degree_bound`]) raises the search ceiling to ≥ 8, so the
    /// generalized solver now recovers it (still verified exactly in-field).
    #[test]
    fn high_degree_solution_recovered_by_analytic_bound() {
        let e = AlgExtension::radical(2, &vec![rat(0), rat(1)]); // α² = x
                                                                 // y = x⁸·α  (numerator x-degree 8 > DEG_CAP)
        let mut p8 = vec![rat(0); 9];
        p8[8] = rat(1);
        let y_true: AlgElem = vec![RatFn::int(0), RatFn::from_poly(&p8)];
        let f_elem = e.constant(RatFn::int(1)); // f = 1 (base scalar)
        let g = e.add(&e.derivation(&y_true), &e.mul(&f_elem, &y_true));

        // The analytic bound sees the degree-8 data and exceeds the old cap.
        let bound = alg_x_degree_bound(&e, &f_elem, &g);
        assert!(
            bound >= 8,
            "analytic bound {bound} must cover the degree-8 solution"
        );
        assert!(
            bound > DEG_CAP,
            "bound {bound} must exceed the old fixed DEG_CAP {DEG_CAP}"
        );

        // The OLD behavior (search only up to DEG_CAP at Den = 1) misses it.
        assert!(
            solve_with_denominator(&e, &f_elem, &g, &poly_one(), DEG_CAP, 2).is_none(),
            "degree-{DEG_CAP} ansatz must NOT contain the degree-8 solution"
        );

        // The bounded solver now recovers it, verified in-field.
        let y = solve_alg_rde_general(&e, &f_elem, &g)
            .expect("analytic bound must recover the high-degree solution");
        let lhs = e.add(&e.derivation(&y), &e.mul(&f_elem, &y));
        assert!(e.elem_eq(&lhs, &g), "D(y)+f·y must equal g; y = {y:?}");
        assert!(e.elem_eq(&y, &y_true), "expected y = x⁸·α; got {y:?}");
    }
}

// ---------------------------------------------------------------------------
// Three-valued contract: the proved non-existence path
// ---------------------------------------------------------------------------

#[cfg(test)]
mod proof_tests {
    use super::*;
    use crate::integrate::risch::poly_rde::{poly_add, poly_scale};

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    /// `α = √x + √(x+1)`: the compositum minimal polynomial.
    fn compositum(p: &QPoly, q: &QPoly) -> AlgExtension {
        let pq_sum = poly_add(p, q);
        let pq_diff = poly_sub(p, q);
        AlgExtension::new(&[
            poly_mul(&pq_diff, &pq_diff),
            poly_zero(),
            poly_scale(&pq_sum, &rat(-2)),
            poly_zero(),
            poly_one(),
        ])
    }

    /// `α = √(a + √b)`: the nested-radical minimal polynomial.
    fn nested(a: &QPoly, b: &QPoly) -> AlgExtension {
        AlgExtension::new(&[
            poly_sub(&poly_mul(a, a), b),
            poly_zero(),
            poly_scale(a, &rat(-2)),
            poly_zero(),
            poly_one(),
        ])
    }

    /// The `∫ exp(√x)/x dx` premise, in the solver's own terms: `α² = x`,
    /// `f = D(α) = (1/(2x))·α`, `g = 1/x`.
    ///
    /// The matrix leading term at infinity is **singular** here (`α` has
    /// half-integral degree, so the power-basis components cannot balance), so
    /// this verdict rests entirely on the ramified-place valuation bound —
    /// `V(f) = −1 > −n = −2`, hence `V(y) = V(g) − V(f) = −1` and every `bⱼ`
    /// must vanish.  Without that argument the honest answer would be a decline.
    #[test]
    fn exp_sqrt_over_x_non_existence_is_proved() {
        let p = vec![rat(0), rat(1)];
        let e = AlgExtension::radical(2, &p);
        let f: AlgElem = vec![RatFn::int(0), RatFn::new(poly_one(), vec![rat(0), rat(2)])];
        let g = e.constant(RatFn::new(poly_one(), vec![rat(0), rat(1)]));

        // The matrix route really does decline; the radical route really does decide.
        let m = system_matrix(&e, &f, 2).unwrap();
        let c: Vec<RatFn> = (0..2)
            .map(|k| g.get(k).cloned().unwrap_or_else(|| RatFn::int(0)))
            .collect();
        assert!(
            system_infinity_bound(&m, &c, 2).is_err(),
            "the matrix bound is expected to be unavailable (singular leading term)"
        );
        assert_eq!(radical_infinity_bound(&e, &f, &g), Some(-1));

        let out = solve_alg_rde_general_checked(&e, &f, &g, radical_minpoly_status(2, &p));
        assert_eq!(out, AlgRdeOutcome::NoRationalSolution, "got {out:?}");
    }

    /// The `∫ (√x + √(x+1))·eˣ dx` premise: `f = 1`, `g = α`.  Here the matrix
    /// route applies (`ρ = 0` with leading matrix `I`), so non-existence is
    /// proved outright.
    #[test]
    fn compositum_non_existence_is_proved() {
        let p = vec![rat(0), rat(1)];
        let q = vec![rat(1), rat(1)];
        let e = compositum(&p, &q);
        let g = e.generator();
        let out = solve_alg_rde_checked(&e, &RatFn::int(1), &g, compositum_minpoly_status(&p, &q));
        assert_eq!(out, AlgRdeOutcome::NoRationalSolution, "got {out:?}");
    }

    /// The `∫ √(x + √x)·eˣ dx` premise: `f = 1`, `g = α`.
    #[test]
    fn nested_radical_non_existence_is_proved() {
        let a = vec![rat(0), rat(1)];
        let b = vec![rat(0), rat(1)];
        let e = nested(&a, &b);
        let g = e.generator();
        let out = solve_alg_rde_checked(
            &e,
            &RatFn::int(1),
            &g,
            nested_radical_minpoly_status(&a, &b),
        );
        assert_eq!(out, AlgRdeOutcome::NoRationalSolution, "got {out:?}");
    }

    /// Without the caller's irreducibility witness nothing may be concluded —
    /// the same equation that is *proved* unsolvable above must **decline**.
    ///
    /// This is the structural guarantee: `MinPolyStatus::Unknown` can never
    /// reach `NoRationalSolution`, so the `Option` shims can never certify.
    #[test]
    fn unknown_minpoly_can_never_prove_non_existence() {
        let p = vec![rat(0), rat(1)];
        let q = vec![rat(1), rat(1)];
        let e = compositum(&p, &q);
        let g = e.generator();
        let out = solve_alg_rde_checked(&e, &RatFn::int(1), &g, MinPolyStatus::Unknown);
        assert_eq!(
            out,
            AlgRdeOutcome::Declined(RdeDecline::AlgebraicFieldNotProved),
            "got {out:?}"
        );
        assert!(!out.proves_no_solution());
    }

    /// A *solvable* equation must still solve through the checked entry point,
    /// and must never be reported as proved-unsolvable.
    #[test]
    fn solvable_compositum_still_solves() {
        let p = vec![rat(0), rat(1)];
        let q = vec![rat(1), rat(1)];
        let e = compositum(&p, &q);
        let alpha = e.generator();
        let f_elem = e.constant(RatFn::int(1));
        let g = e.add(&e.derivation(&alpha), &e.mul(&f_elem, &alpha));
        let out = solve_alg_rde_checked(&e, &RatFn::int(1), &g, compositum_minpoly_status(&p, &q));
        let y = out.solution().expect("D(α)+α is solved by y = α");
        let lhs = e.add(&e.derivation(&y), &e.mul(&f_elem, &y));
        assert!(e.elem_eq(&lhs, &g));
    }

    /// The `n = 0` guard: `g = 0` always solves with `y = 0`, so the proof path
    /// is never entered for it.
    #[test]
    fn zero_rhs_solves_rather_than_certifying() {
        let p = vec![rat(0), rat(1)];
        let e = AlgExtension::radical(2, &p);
        let g = e.from_int(0);
        let out = solve_alg_rde_checked(&e, &RatFn::int(1), &g, radical_minpoly_status(2, &p));
        assert!(matches!(out, AlgRdeOutcome::Solved(_)), "got {out:?}");
    }

    // -- The irreducibility witnesses -------------------------------------

    #[test]
    fn minpoly_status_recognises_the_reachable_shapes() {
        // y² − x: x is squarefree of degree 1.
        assert_eq!(
            radical_minpoly_status(2, &vec![rat(0), rat(1)]),
            MinPolyStatus::ProvedIrreducible
        );
        // y² − x²: x² *is* a square in ℚ(x), so the "extension" is trivial and
        // the status must not be claimed.
        assert_eq!(
            radical_minpoly_status(2, &vec![rat(0), rat(0), rat(1)]),
            MinPolyStatus::Unknown
        );
        // y³ − x²: gcd(3, 2) = 1 ⇒ irreducible even though x² is not squarefree.
        assert_eq!(
            radical_minpoly_status(3, &vec![rat(0), rat(0), rat(1)]),
            MinPolyStatus::ProvedIrreducible
        );
        // √x, √(x+1): neither, nor their product, is a square.
        assert_eq!(
            compositum_minpoly_status(&vec![rat(0), rat(1)], &vec![rat(1), rat(1)]),
            MinPolyStatus::ProvedIrreducible
        );
        // √x and √(x³): the product x⁴ *is* a square, so ℚ(x)(√x, √(x³)) is only
        // quadratic and the degree-4 "minimal polynomial" is reducible.
        assert_eq!(
            compositum_minpoly_status(&vec![rat(0), rat(1)], &vec![rat(0), rat(0), rat(0), rat(1)]),
            MinPolyStatus::Unknown
        );
        // √(x + √x): b = x is not a square and a²−b = x²−x is not a square.
        assert_eq!(
            nested_radical_minpoly_status(&vec![rat(0), rat(1)], &vec![rat(0), rat(1)]),
            MinPolyStatus::ProvedIrreducible
        );
        // √(x + √(x²)) — b = x² is a square, so the nesting is spurious.
        assert_eq!(
            nested_radical_minpoly_status(&vec![rat(0), rat(1)], &vec![rat(0), rat(0), rat(1)]),
            MinPolyStatus::Unknown
        );
    }

    #[test]
    fn square_detection_over_qx() {
        assert!(is_square_in_qx(&vec![rat(1), rat(2), rat(1)])); // (x+1)²
        assert!(is_square_in_qx(&vec![rat(4), rat(8), rat(4)])); // (2x+2)²
        assert!(!is_square_in_qx(&vec![rat(2), rat(4), rat(2)])); // 2(x+1)²
        assert!(!is_square_in_qx(&vec![rat(0), rat(1)])); // x
        assert!(!is_square_in_qx(&vec![rat(-1), rat(0), rat(1)])); // x²−1
        assert!(is_square_in_qx(&vec![rat(9)])); // 9
        assert!(!is_square_in_qx(&vec![rat(3)])); // 3
    }

    // -- The linear-algebra helpers ---------------------------------------

    #[test]
    fn polynomial_determinant_matches_hand_computation() {
        // | x   1 |
        // | x²  x |  =  x² − x²  =  0
        let m = vec![
            vec![vec![rat(0), rat(1)], poly_one()],
            vec![vec![rat(0), rat(0), rat(1)], vec![rat(0), rat(1)]],
        ];
        assert_eq!(trim(poly_matrix_det(&m, 2)), poly_zero());
        // | x+1  2 |
        // | 3    x |  =  x² + x − 6
        let m = vec![
            vec![vec![rat(1), rat(1)], vec![rat(2)]],
            vec![vec![rat(3)], vec![rat(0), rat(1)]],
        ];
        assert_eq!(trim(poly_matrix_det(&m, 2)), vec![rat(-6), rat(1), rat(1)]);
    }

    #[test]
    fn polynomial_inverse_mod_round_trips() {
        let m = vec![rat(-2), rat(0), rat(1)]; // x² − 2
        let a = vec![rat(1), rat(1)]; // x + 1
        let inv = poly_inverse_mod(&a, &m).expect("x+1 is a unit mod x²−2");
        let prod = poly_divrem(&poly_mul(&a, &inv), &m).1;
        assert_eq!(trim(prod), poly_one());
        // x²−2 ≡ 0 is not invertible.
        assert!(poly_inverse_mod(&m, &m).is_none());
    }
}

#[cfg(test)]
mod engine_verdict_tests {
    use crate::errors::AlkahestError;
    use crate::integrate::engine::{integrate, IntegrationError};
    use crate::kernel::{Domain, ExprId, ExprPool};
    use std::collections::HashMap;

    fn run(src: &str) -> (ExprPool, ExprId, Result<ExprId, IntegrationError>) {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let e = crate::parse::parse(src, &pool, &mut syms).expect("parse");
        let x = pool.symbol("x", Domain::Real);
        let r = integrate(e, x, &pool).map(|d| d.value);
        (pool, e, r)
    }

    fn assert_certified(src: &str) {
        let (_pool, _e, r) = run(src);
        assert!(
            matches!(r, Err(IntegrationError::NonElementary(_))),
            "∫ {src} dx must stay certified E-INT-004; got {r:?}"
        );
    }

    fn assert_not_certified(src: &str) {
        let (_pool, _e, r) = run(src);
        assert!(
            !matches!(r, Err(IntegrationError::NonElementary(_))),
            "∫ {src} dx must not be certified E-INT-004; got {r:?}"
        );
    }

    // -- The verdicts that must survive the three-valued rewrite -------------
    //
    // Each of these reaches one of the three call sites that used to read the
    // solver's `None` as a certificate, and each is still `E-INT-004` — but now
    // because non-existence was *proved*, not because the ansatz came up empty.

    /// `exp_algebraic.rs`, ramified place at infinity (`α² = x`, `deg p = 1`).
    /// The matrix bound is unavailable here; the radical valuation bound decides.
    #[test]
    fn verdict_exp_sqrt_over_x_stays_certified() {
        assert_certified("exp(sqrt(x))/x");
    }

    /// `exp_algebraic.rs`, *unramified* at infinity (`α² = x²+1`, `deg p = 2`).
    /// Neither the radical bound (two places above `∞`) nor the bare matrix
    /// bound (singular leading term) applies; the shearing `sⱼ = j` does.
    #[test]
    fn verdict_exp_sqrt_x2_plus_1_stays_certified() {
        assert_certified("exp(sqrt(x^2+1))");
    }

    /// `exp_case.rs::try_compositum_poly_rde` — `α = √x + √(x+1)`, `f = 1`.
    #[test]
    fn verdict_compositum_times_exp_stays_certified() {
        assert_certified("(sqrt(x)+sqrt(x+1))*exp(x)");
    }

    /// `exp_case.rs::try_nested_radical_poly_rde` — `α = √(x + √x)`, `f = 1`.
    #[test]
    fn verdict_nested_radical_times_exp_stays_certified() {
        assert_certified("sqrt(x+sqrt(x))*exp(x)");
    }

    /// A wider band of the same three families, all still certified.
    #[test]
    fn neighbouring_verdicts_stay_certified() {
        for src in [
            "exp(sqrt(x))/x^2",
            "exp(sqrt(x))/(x+1)",
            "exp(sqrt(x))/(x^2+1)",
            "exp(sqrt(x))/(x-1)",
            "exp(sqrt(x+1))/x",
            "exp(sqrt(x^2+1))/x",
            "(sqrt(x)-sqrt(x+1))*exp(x)",
            "sqrt(x)*sqrt(x+1)*exp(x)",
            "(sqrt(x)+sqrt(x^2+1))*exp(x)",
            "(sqrt(x+1)+sqrt(x+2))*exp(x)",
            "(sqrt(x)+sqrt(x+1))*exp(2*x)",
            "(sqrt(x)+sqrt(x+1))*exp(x^2)",
            "sqrt(x+sqrt(x^2+1))*exp(x)",
            "sqrt(x+1+sqrt(x))*exp(x)",
            "sqrt(x+sqrt(x))*exp(2*x)",
            "sqrt(x^2+sqrt(x))*exp(x)",
            "exp(x)/sqrt(x+sqrt(x))",
        ] {
            assert_certified(src);
        }
    }

    /// Nothing that used to solve stopped solving.
    #[test]
    fn solved_integrals_stay_solved() {
        for src in [
            "exp(sqrt(x))",
            "exp(sqrt(x))*(1/(2*sqrt(x))+1/2)",
            "x*exp(sqrt(x))",
            "sqrt(x)*exp(sqrt(x))",
            "exp(sqrt(x))/sqrt(x)",
            "(sqrt(x)+1/(2*sqrt(x)))*exp(x)",
            "(sqrt(x)+sqrt(x+1)+1/(2*sqrt(x))+1/(2*sqrt(x+1)))*exp(x)",
        ] {
            let (_pool, _e, r) = run(src);
            assert!(r.is_ok(), "∫ {src} dx should still solve; got {r:?}");
        }
    }

    // -- The false certificate this rewrite removes -------------------------

    /// **A false `E-INT-004` on the pre-rewrite tree.**
    ///
    /// `∫ (√x + 1/√(4x))·eˣ dx = √x·eˣ` — elementary, because `1/√(4x)` is just
    /// `1/(2√x)` and `d/dx(√x·eˣ) = eˣ·(√x + 1/(2√x))`.  The old code fed it to
    /// `try_compositum_poly_rde`, which treated `√x` and `√(4x)` as two
    /// *independent* square roots and built the degree-4 minimal polynomial
    /// `α⁴ − 2(x+4x)α² + (x−4x)²`.  That polynomial is **reducible** (`α = 3√x`
    /// generates only a quadratic extension), so `ℚ(x)[y]/(q)` is a product of
    /// two fields; the true `v` lives in one factor and does not lift to the
    /// product.  The ansatz found nothing, and the site published that as a
    /// theorem.
    ///
    /// It is now an `E-INT-001` decline naming the unproved premise.  Declining
    /// is the honest verdict; *solving* it needs radicand normalisation in
    /// `exp_case::detect_two_sqrt_compositum` (out of this module's scope).
    #[test]
    fn degenerate_compositum_no_longer_falsely_certifies() {
        assert_not_certified("sqrt(x)*exp(x) + exp(x)/sqrt(4*x)");
        assert_not_certified("sqrt(9*x)*exp(x) + 3*exp(x)/sqrt(4*x)");
    }

    /// And the antiderivative really does exist: `d/dx(√x·eˣ)` equals the
    /// integrand above at every sample point, so the old certificate was false.
    #[test]
    fn degenerate_compositum_witness_is_elementary() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let x = pool.symbol("x", Domain::Real);
        let integrand = crate::parse::parse("sqrt(x)*exp(x) + exp(x)/sqrt(4*x)", &pool, &mut syms)
            .expect("parse");
        let antideriv = crate::parse::parse("sqrt(x)*exp(x)", &pool, &mut syms).expect("parse");
        let d = crate::diff::diff(antideriv, x, &pool).expect("diff");
        let ds = crate::simplify::engine::simplify(d.value, &pool).value;
        for &xv in &[0.4_f64, 1.1, 2.3, 5.7] {
            let bindings = HashMap::from([(x, xv)]);
            let lhs = crate::eval::eval_f64(ds, &pool, &bindings).expect("eval d/dx F");
            let rhs = crate::eval::eval_f64(integrand, &pool, &bindings).expect("eval f");
            assert!(
                (lhs - rhs).abs() <= 1e-9 * (1.0 + rhs.abs()),
                "x={xv}: d/dx(√x·eˣ) = {lhs} but the integrand is {rhs}"
            );
        }
    }

    // -- A decline can never reach E-INT-004 --------------------------------

    /// The structural guarantee, end to end: an integrand that reaches
    /// `try_compositum_poly_rde` and makes the solver **decline** must come back
    /// as `E-INT-001` carrying the reason — never as `E-INT-004`.
    #[test]
    fn a_declined_algebraic_rde_reports_e_int_001_with_a_reason() {
        for src in [
            "(sqrt(x)+sqrt(4*x))*exp(x)",
            "(sqrt(x)+sqrt(x^3))*exp(x)",
            "sqrt(x)*exp(x) + exp(x)/sqrt(4*x)",
        ] {
            let (_pool, _e, r) = run(src);
            let Err(err) = r else {
                continue; // solving it is fine too — just never a false theorem
            };
            assert_eq!(
                err.code(),
                "E-INT-001",
                "∫ {src} dx: a decline must be E-INT-001; got {err}"
            );
            assert!(
                format!("{err}").contains("could not be decided"),
                "∫ {src} dx: the decline must name its reason; got {err}"
            );
        }
    }
}
