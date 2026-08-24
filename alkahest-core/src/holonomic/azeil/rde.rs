//! The shared inner solver: the **parametric Risch differential equation**
//!
//! ```text
//! R'(x) + θ(x)·R(x) = Σ_{i=0}^{J} a_i·r_i(x)
//! ```
//!
//! over `Q(n)(x)`, solved by undetermined coefficients.
//!
//! # Why one solver serves both stages
//!
//! Dividing the certificate identity `Σ_i a_i(n)·F(n+i,x) = D_x(R·F)` through
//! by `F` turns it into exactly the equation above, with `θ = ∂_x F/F` and
//! `r_i = F(n+i,x)/F(n,x)` — both supplied exactly by
//! [`super::hyperexp::HyperExpTerm`]:
//!
//! ```text
//! D_x(R·F) = (R' + θ·R)·F
//! ```
//!
//! The indefinite case ([`mod@super::dgosper`]) is the same equation with `J = 0`
//! and `a_0` normalised to `1`, i.e. `R' + θ·R = 1`; creative telescoping
//! (`super::search`) is the same equation with `a_0 … a_{J−1}` unknown and
//! `a_J = 1`. Writing two solvers would mean two ansatz shapes, two
//! completeness stories and two places for a certificate to be wrong.
//!
//! # Relationship to `integrate::risch::rational_rde`
//!
//! `integrate::risch::rational_rde::solve_rational_rde_generalized` already
//! solves `v' + f·v = c` over `ℚ(x)` for rational `f`, which is the `J = 0`,
//! `n`-free case of this. It is **not** reused, for two independent reasons,
//! and the first one is a correctness reason:
//!
//! 1. Its denominator bound for `v` is `E = gcd(D, D')` with `D` the
//!    denominator of the right-hand side `c` — Bronstein §6.1, which is exact
//!    when `f` is a *polynomial*. When `f` has poles, `v` may have poles that
//!    `c` knows nothing about. Concretely, `∫ x²·eˣ dx = (x²−2x+2)·eˣ` is the
//!    equation `R' + (2/x + 1)·R = 1`, whose solution `R = (x²−2x+2)/x²` has a
//!    double pole at `0`, while `c = 1` gives `D = E = 1` and admits only
//!    polynomial `v`. That solver returns `None` there — a false refusal for
//!    this module's purposes. (Its `f` is `k·η'` in the exponential tower,
//!    where the polynomial hypothesis does hold; the gap is in the
//!    *generalized* entry point, not in the tower that uses it.)
//! 2. It is over `ℚ`, not `Q(n)`, and the `a_i` unknowns of stage 3 are
//!    coupled to the unknown numerator of `R` in one linear system. A
//!    standalone RDE solve cannot express that coupling at all.
//!
//! Reusing it as a fast path for the `n`-free case and falling back here would
//! give the same question two answers with two different completeness
//! profiles. One solver, one ansatz, one honest limitation is better.
//!
//! # The ansatz, and exactly how much of it is a guess
//!
//! Write `θ = A/D` (reduced, `D` monic) and let `B` be a common denominator of
//! the `r_i`, so `r_i = ρ_i/B` with `ρ_i ∈ Q(n)[x]`. The ansatz is
//!
//! ```text
//! R = P(x) / (D(x)^κ · B(x)),    deg P ≤ d.
//! ```
//!
//! The *support* of that denominator is forced, not guessed. Let `α` be a pole
//! of `R` of order `m ≥ 1` and look at the orders on both sides of the
//! equation at `α`:
//!
//! - `θ` regular at `α`: `R'` has a pole of order `m+1` and `θ·R` at most `m`,
//!   so no cancellation is possible and the right-hand side must have a pole of
//!   order `m+1` there. So `α` is a pole of some `r_i`, i.e. a root of `B`, and
//!   `m ≤ ord_α(B) − 1`.
//! - `θ` with a pole of order `e ≥ 2`: `θ·R` has order `m+e > m+1`, so again no
//!   cancellation, `m + e = ord_α(RHS) ≤ ord_α(B)`.
//! - `θ` with a *simple* pole of residue `ρ`: `R'` and `θ·R` both have order
//!   `m+1`, with leading coefficients summing to `(ρ − m)·c`. Unless `ρ = m`
//!   the same argument applies; when `ρ = m` they cancel and `R` may have a
//!   pole of order `m` at a point the right-hand side is regular at.
//!
//! So a pole of `R` lies over a root of `D` or of `B` and nowhere else — which
//! is what `D^κ·B` encodes. What is *not* forced is `κ`: it is exactly the
//! resonance `ρ = m` of the third bullet, and `ρ ∈ Q(n)` is symbolic in the
//! cases this module exists for (`θ = n/x + …` for `F = xⁿ·…`), so whether it
//! occurs is not decidable. `κ` is therefore a bounded search, `d` likewise,
//! and running out of either is [`super::DiffTelescopingError::SearchExhausted`] —
//! never a certificate that was not verified.
//!
//! # The linear system
//!
//! Clearing `Q²·D` out of `Σ_i a_i·r_i = R' + θ·R` with `R = P/Q`,
//! `Q = D^κ·B`, and using `Q²D/B = D^{2κ+1}·B`, gives a *polynomial* identity
//! in `Q(n)[x]` that is linear in the unknowns:
//!
//! ```text
//! Σ_i a_i · (ρ_i · D^{2κ+1} · B)  =  D·Q·P′  −  (D·Q′ − A·Q)·P
//! ```
//!
//! Equating coefficients of each power of `x` is a linear system over the field
//! `Q(n)`, solved by the same Gaussian elimination the discrete engine uses
//! ([`mod@super::super::zeilberger`]'s `field_gaussian_solve`).

use crate::holonomic::qfield::{
    polyk_deriv_k, ratk_deriv_k, rn_int, rn_is_zero, rn_neg, rn_one, rn_zero, PolyK, RatK, Rn,
};
use crate::holonomic::zeilberger::field_gaussian_solve;

/// Refuse a probe whose linear system would be larger than this.
///
/// The same protection [`super::super::telescoping2d`]'s `MAX_ANSATZ_UNKNOWNS`
/// provides: the elimination is `O(rows · cols²)` over unbounded-precision
/// rational functions, so a caller who raises the bounds far enough must still
/// not be able to wedge the process. Every worked example in this module stays
/// under 40.
pub(super) const MAX_ANSATZ_UNKNOWNS: usize = 256;

/// The degree-independent part of one order's setup: the shift ratios put over
/// a common denominator, and `θ` split into numerator and denominator.
#[derive(Clone, Debug)]
pub(super) struct RdeSetup {
    /// `A`, the numerator of `θ`.
    pub a_num: PolyK,
    /// `D`, the (monic) denominator of `θ`.
    pub d_den: PolyK,
    /// `B`, a common denominator of the shift ratios.
    pub b_den: PolyK,
    /// `ρ_i = r_i · B ∈ Q(n)[x]`, for `i = 0..=J`.
    pub rho: Vec<PolyK>,
    /// The shift ratios themselves, kept for exact verification.
    pub ratios: Vec<RatK>,
    /// `θ`, kept for exact verification.
    pub theta: RatK,
}

impl RdeSetup {
    /// Build the setup for the right-hand side `Σ_i a_i·r_i`.
    ///
    /// `None` (rather than `Err`) when the shift ratios have no common
    /// denominator this construction can use — a structural dead end for this
    /// order, not a statement about the integrand.
    pub(super) fn new(theta: &RatK, ratios: Vec<RatK>) -> Option<RdeSetup> {
        let mut b_den = PolyK::one();
        for r in &ratios {
            b_den = PolyK::lcm(&b_den, &r.den);
        }
        if b_den.is_zero() {
            return None;
        }
        let rho: Option<Vec<PolyK>> = ratios
            .iter()
            .map(|r| PolyK::exact_div(&b_den.mul(&r.num), &r.den))
            .collect();
        Some(RdeSetup {
            a_num: theta.num.clone(),
            d_den: theta.den.clone(),
            b_den,
            rho: rho?,
            ratios,
            theta: theta.clone(),
        })
    }

    /// Recurrence order `J` (`ratios.len() - 1`).
    pub(super) fn order(&self) -> usize {
        self.ratios.len() - 1
    }
}

/// One solved candidate: the recurrence coefficients (with `a_J = 1`) and the
/// certificate. **Not** yet verified — [`verify`] is a separate step on
/// purpose, so that what makes a candidate a result is independent of how the
/// search found it.
#[derive(Clone, Debug)]
pub(super) struct RdeCandidate {
    pub a: Vec<Rn>,
    pub r: RatK,
}

/// `x^j` as an element of `Q(n)[x]`.
fn x_mono(j: usize) -> PolyK {
    let mut coeffs = vec![rn_zero(); j + 1];
    coeffs[j] = rn_one();
    PolyK::from_coeffs(coeffs)
}

/// Try one `(κ, d)` probe: solve the linear system for `P` and `a_0..a_{J−1}`
/// with `a_J = 1`.
///
/// `None` means this probe's system has no solution (or was refused as too
/// large), which says nothing at all about neighbouring probes.
pub(super) fn solve_probe(setup: &RdeSetup, kappa: usize, d: usize) -> Option<RdeCandidate> {
    let order = setup.order();
    let n_var = (d + 1) + order;
    if n_var > MAX_ANSATZ_UNKNOWNS {
        return None;
    }

    // Q = D^κ · B, and the multipliers of the polynomial identity
    //     Σ_i a_i·(ρ_i·D^{2κ+1}·B) = D·Q·P′ − (D·Q′ − A·Q)·P.
    let mut d_pow = PolyK::one();
    for _ in 0..kappa {
        d_pow = d_pow.mul(&setup.d_den);
    }
    let q = d_pow.mul(&setup.b_den);
    if q.is_zero() {
        return None;
    }
    let q_prime = polyk_deriv_k(&q);
    let dq = setup.d_den.mul(&q);
    let vv = setup.d_den.mul(&q_prime).sub(&setup.a_num.mul(&q));

    // `D^{2κ+1}·B`, the factor the right-hand side picks up.
    let mut d_pow2 = setup.d_den.clone();
    for _ in 0..(2 * kappa) {
        d_pow2 = d_pow2.mul(&setup.d_den);
    }
    let rhs_scale = d_pow2.mul(&setup.b_den);
    let l: Vec<PolyK> = setup
        .rho
        .iter()
        .map(|rho| rho.mul(&rhs_scale))
        .collect::<Vec<_>>();

    // M_j = D·Q·(j·x^{j−1}) − (D·Q′ − A·Q)·x^j.
    let mut m: Vec<PolyK> = Vec::with_capacity(d + 1);
    for j in 0..=d {
        let dp = if j == 0 {
            PolyK::zero()
        } else {
            dq.mul(&x_mono(j - 1)).scale(&rn_int(j as i64))
        };
        m.push(dp.sub(&vv.mul(&x_mono(j))));
    }

    let mut max_deg = 0i32;
    for p in m.iter().chain(l.iter()) {
        max_deg = max_deg.max(p.degree());
    }
    if max_deg < 0 {
        max_deg = 0;
    }
    let n_eq = (max_deg as usize) + 1;

    let mut mat = vec![vec![rn_zero(); n_var]; n_eq];
    let mut rhs = vec![rn_zero(); n_eq];
    for (row_idx, row) in mat.iter_mut().enumerate() {
        for (j, mj) in m.iter().enumerate() {
            row[j] = mj.coeff(row_idx);
        }
        for i in 0..order {
            row[(d + 1) + i] = rn_neg(&l[i].coeff(row_idx));
        }
        rhs[row_idx] = l[order].coeff(row_idx);
    }

    let sol = field_gaussian_solve(mat, rhs)?;
    let p_poly = PolyK::from_coeffs(sol[..=d].to_vec());
    let mut a: Vec<Rn> = sol[(d + 1)..].to_vec();
    a.push(rn_one());

    let r = RatK {
        num: p_poly,
        den: q,
    }
    .normalize();
    Some(RdeCandidate { a, r })
}

/// The **only** thing that makes a candidate a result: check
/// `Σ_i a_i·r_i = R′ + θ·R` as an exact identity in `Q(n)(x)`.
///
/// This re-derives `R′` from `R` and never consults the linear system that
/// produced it, so a bug in the ansatz bookkeeping cannot produce a passing
/// certificate.
pub(super) fn verify(setup: &RdeSetup, a: &[Rn], r: &RatK) -> bool {
    if a.len() != setup.ratios.len() {
        return false;
    }
    if a.iter().all(rn_is_zero) {
        return false;
    }
    let mut lhs = RatK::zero();
    for (ai, ri) in a.iter().zip(setup.ratios.iter()) {
        lhs = lhs.add(&RatK::from_rn(ai.clone()).mul(ri));
    }
    let rhs = ratk_deriv_k(r).add(&setup.theta.mul(r));
    lhs.sub(&rhs).is_zero()
}
