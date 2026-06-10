//! van Hoeij integral-basis **enlargement loop** (MB) — builds a global integral
//! basis of `ℚ(x)(y)`, `F(x,y)=0`, from the power basis by repeatedly dividing
//! out singular places.
//!
//! Following van Hoeij (1994 §2): start `b₀ = 1`; for `d = 1 … n−1`, set the
//! initial `b_d = y·b_{d−1}` (degree `d`) and enlarge it as long as some
//! `a = (a₀b₀ + ⋯ + a_{d−1}b_{d−1} + b_d)/(x−α)` is integral, where `α` is a
//! singular place (`(x−α)² | disc`).  The coefficients `aᵢ ∈ ℚ` are found by
//! substituting the Puiseux expansions at `α` into `a` and requiring every
//! **negative-power** coefficient (in the place's uniformizer) to vanish — a
//! `ℚ`-linear system.  Each proposed enlargement is then **gated by the exact
//! [`is_integral`] test**, so the result is sound regardless of Puiseux
//! truncation: a wrong proposal is rejected, never accepted.
//!
//! Efficiency (van Hoeij's remark): a **simple radical** `yⁿ = p(x)` skips the
//! Puiseux enlargement loop entirely — [`integral_basis`] dispatches to Trager's
//! explicit basis `wᵢ = yⁱ/dᵢ` ([`super::integral_basis::radical_integral_basis`]),
//! whose only cost is a squarefree factorization of `p`.
//!
//! Scope: enlargements are proposed at **rational** singular places (rational
//! `α`, `try_enlarge`) and at **algebraic** singular places (irreducible
//! discriminant factors `q` of degree ≥ 2 with `q² | disc`, `try_enlarge_algebraic`).
//! At an algebraic place the proposal has van Hoeij's q-adic form
//! `(Σ aᵢ(x) bᵢ + bd)/q(x)` with `aᵢ ∈ ℚ[x]` of degree `< deg q`; the
//! coefficients are found from the **`K`-valued** Puiseux sheets over a root `α`
//! of `q` (`K = ℚ(α)`, via [`crate::poly::puiseux::puiseux_at_algebraic`]) —
//! each negative-power `K`-coefficient is `deg q` rational equations, so the
//! linear system is still over ℚ.  Sheets whose coefficients escape `K` (skipped
//! by `puiseux_at_algebraic`) merely *under*-constrain the system, so a proposal
//! there is rejected by the exact [`super::integral_basis::is_integral`] gate —
//! the basis may then be non-maximal at such a place, but is **never incorrect**.
//! Produces correct integral bases for radical curves, rational-branch curves
//! such as the nodal cubic `y² = x³ + x²` (`{1, y/x}`), and curves singular only
//! at an irrational place such as `y² = (x²−2)²·(x+3)` (`{1, y/(x²−2)}`).

use rug::{Integer, Rational};
use std::collections::BTreeMap;

use super::super::risch::alg_field::{AlgElem, AlgExtension, RatFn, RationalFunctionField};
use super::super::risch::number_field::{CoeffField, KElem, NumberField};
use super::super::risch::poly_rde::{degree, poly_deriv, QPoly};
use super::super::risch::rational_rde::poly_gcd;
use super::integral_basis::{discriminant, is_integral, rational_singularities};
use crate::poly::puiseux::{
    factor_over_q, puiseux_at, puiseux_at_algebraic, AlgBasePuiseuxSeries, PuiseuxSeries,
};

/// A Laurent series in the place uniformizer `t = (x−α)^{1/e}` (integer
/// `t`-exponents), truncated.  Shared with [`super::residues`].
pub(super) type TS = BTreeMap<i64, Rational>;

/// Compute an integral basis `b₀, …, b_{n−1}` of `ℚ(x)(y)` over `ℚ[x]` for the
/// monic minimal polynomial `F = Σ f_coeffs[j] yʲ`.  Returns the basis as
/// `AlgElem`s (each verified integral), or `None` on failure.
///
/// Maximal when every singular place is rational with rational branches;
/// otherwise an integral (possibly non-maximal) basis — see the module docs.
pub fn integral_basis(f_coeffs: &[QPoly]) -> Option<Vec<AlgElem>> {
    let ext = AlgExtension::new(f_coeffs);
    let n = ext.degree() as usize;
    if n < 1 {
        return None;
    }
    // Efficiency fast-path (van Hoeij's remark): for a **simple radical**
    // `yⁿ = p(x)` the integral basis is Trager's explicit `wᵢ = yⁱ/dᵢ`, needing
    // only a squarefree factorization of `p` — no Puiseux expansions or
    // enlargement loop.  Each `wᵢ` is `is_integral`-gated inside, so this is
    // sound; we only take it when it yields a full basis.
    if let Some(p) = as_simple_radical(f_coeffs, n) {
        if let Some(basis) = super::integral_basis::radical_integral_basis(n, &p) {
            return Some(basis);
        }
    }
    let monos = to_monomials(f_coeffs);
    let disc = discriminant(f_coeffs);
    let sing = rational_singularities(&disc);
    // Algebraic singular places: irreducible factors `q` of the discriminant of
    // degree ≥ 2 with `q² | disc`.  Enlargements there carry coefficients in
    // `K = ℚ(α)`, `q(α)=0` (see `try_enlarge_algebraic`).
    let alg_sing = algebraic_singularities(&disc);

    let mut b: Vec<AlgElem> = vec![ext.from_int(1)]; // b₀ = 1
    for d in 1..n {
        let mut bd = ext.mul(&b[d - 1], &ext.generator()); // y·b_{d−1}
                                                           // Each real enlargement drops the discriminant by (x−α)² (rational) or q²
                                                           // (algebraic); bound the work.
        let cap = (degree(&disc).max(0) as usize + 2) * (sing.len() + alg_sing.len()).max(1) + 4;
        let mut iters = 0;
        'enlarge: loop {
            if iters >= cap {
                break;
            }
            for alpha in &sing {
                if let Some(cand) = try_enlarge(&ext, &b, &bd, d, alpha, &monos, n) {
                    if !ext.elem_eq(&cand, &bd) && is_integral(f_coeffs, &cand) {
                        bd = cand;
                        iters += 1;
                        continue 'enlarge;
                    }
                }
            }
            for q in &alg_sing {
                if let Some(cand) = try_enlarge_algebraic(&ext, &b, &bd, d, q, &monos, n) {
                    if !ext.elem_eq(&cand, &bd) && is_integral(f_coeffs, &cand) {
                        bd = cand;
                        iters += 1;
                        continue 'enlarge;
                    }
                }
            }
            break;
        }
        b.push(bd);
    }

    for bi in &b {
        if !is_integral(f_coeffs, bi) {
            return None;
        }
    }
    Some(b)
}

/// Try to enlarge `bd` at the place `x = α`: find `a₀…a_{d−1} ∈ ℚ` so that
/// `(Σ aᵢ bᵢ + bd)/(x−α)` is integral, returning that element (unverified —
/// caller gates with [`is_integral`]).  `None` if the place has algebraic
/// branches or no enlargement exists.
fn try_enlarge(
    ext: &AlgExtension,
    b: &[AlgElem],
    bd: &AlgElem,
    d: usize,
    alpha: &Rational,
    monos: &[(u32, u32, Rational)],
    n: usize,
) -> Option<AlgElem> {
    // Precision: enough Puiseux terms to resolve the coordinate denominators.
    let dmax = max_denom_degree(b, bd);
    let prec = (dmax + n + 3) as u32;
    // Substitute every (rational) sheet at α.  Missing algebraic sheets can only
    // *under*-constrain the system → an unverified proposal that `is_integral`
    // (the caller's gate) then rejects; soundness does not depend on completeness.
    let branches = puiseux_at(monos, alpha, prec);
    if branches.is_empty() {
        return None;
    }
    let emax = branches.iter().map(|s| s.ramification).max().unwrap_or(1) as i64;
    let u_bound = (prec as i64 + 3) * emax + 4;

    // Build the ℚ-linear system: for each branch and each negative-power
    // coefficient of (Σ aᵢ bᵢ + bd)/(x−α), the linear form in (a₀…a_{d−1}) is 0.
    let mut matrix: Vec<Vec<Rational>> = Vec::new();
    let mut rhs: Vec<Rational> = Vec::new();
    for s in &branches {
        let e = s.ramification as i64;
        let bts = branch_ts(s);
        // Series of each bᵢ (i<d) and of bd along this branch.
        let bi_ts: Vec<TS> = (0..d)
            .map(|i| elem_ts(&b[i], alpha, e, u_bound, &bts))
            .collect();
        let bd_ts = elem_ts(bd, alpha, e, u_bound, &bts);
        // Negative-power range: keys k < e (after dividing by (x−α)=t^e).
        let low = bi_ts
            .iter()
            .chain(std::iter::once(&bd_ts))
            .filter_map(|s| s.keys().next().copied())
            .min()
            .unwrap_or(0);
        for k in low..e {
            let row: Vec<Rational> = bi_ts
                .iter()
                .map(|s| s.get(&k).cloned().unwrap_or_else(|| Rational::from(0)))
                .collect();
            let r = -bd_ts.get(&k).cloned().unwrap_or_else(|| Rational::from(0));
            matrix.push(row);
            rhs.push(r);
        }
    }
    let a = gauss_solve(matrix, rhs, d)?;

    // candidate = (Σ aᵢ bᵢ + bd) · 1/(x−α).
    let mut num = bd.clone();
    for (i, ai) in a.iter().enumerate() {
        if *ai != 0 {
            num = ext.add(&num, &scale(&b[i], &RatFn::from_poly(&vec![ai.clone()])));
        }
    }
    let inv_lin = RatFn::new(
        vec![Rational::from(1)],
        vec![-alpha.clone(), Rational::from(1)],
    );
    Some(scale(&num, &inv_lin))
}

// ---------------------------------------------------------------------------
// Algebraic singular places  (irreducible q, deg ≥ 2, q² | disc)
// ---------------------------------------------------------------------------

/// The **algebraic** singular places of the curve: monic irreducible factors `q`
/// of the discriminant of degree `≥ 2` with `q² | disc` (equivalently `q` divides
/// the repeated part `gcd(D, D′)`).  A root `α` of such a `q` is an irrational
/// singular base point; the place(s) over it are where the power basis can fail
/// integrality but the existing **rational** loop never enlarges.
fn algebraic_singularities(disc: &QPoly) -> Vec<QPoly> {
    if degree(disc) < 2 {
        return Vec::new();
    }
    let repeated = poly_gcd(disc, &poly_deriv(disc)); // = ∏ qⱼ^{mⱼ−1}
    if degree(&repeated) < 2 {
        return Vec::new();
    }
    factor_over_q(disc)
        .into_iter()
        .filter_map(|(q, deg)| {
            // Only genuinely-algebraic places (deg ≥ 2) with q² | disc, i.e.
            // q | gcd(D, D′).  (The deg-1 / rational case is the existing loop.)
            if deg < 2 {
                return None;
            }
            let g = poly_gcd(&repeated, &q);
            if degree(&g) == deg as i64 {
                Some(q)
            } else {
                None
            }
        })
        .collect()
}

/// Try to enlarge `bd` at the **algebraic** place over a root `α` of the monic
/// irreducible `q` (`deg q = m ≥ 2`, `q² | disc`): find polynomials
/// `a₀, …, a_{d−1} ∈ ℚ[x]` of degree `< m` so that `(Σ aᵢ bᵢ + bd)/q(x)` is
/// integral, returning that element (unverified — the caller gates it with the
/// exact [`is_integral`]).  `None` if no enlargement exists or the place's
/// branches could not be expanded over `K = ℚ(α)`.
///
/// van Hoeij's q-adic form (1994 §2/§4 non-rational singularities): the
/// numerator `Σ aᵢ(x) bᵢ + bd` is expanded along each Puiseux sheet over `α`
/// (coefficients in `K`, via [`puiseux_at_algebraic`]); requiring the
/// negative-power coefficients in the place uniformizer to vanish gives a
/// **ℚ-linear** system in the unknown rationals `aᵢₗ` (`aᵢ = Σ_{l<m} aᵢₗ xˡ`),
/// since each `K`-coefficient is `m` ℚ-components.
fn try_enlarge_algebraic(
    ext: &AlgExtension,
    b: &[AlgElem],
    bd: &AlgElem,
    d: usize,
    q: &QPoly,
    monos: &[(u32, u32, Rational)],
    n: usize,
) -> Option<AlgElem> {
    let m = degree(q) as usize; // = [K:ℚ]
    if m < 2 {
        return None;
    }
    let nf = NumberField::new(q.clone());
    let alpha = nf.reduce(&vec![Rational::from(0), Rational::from(1)]); // α = t

    let dmax = max_denom_degree(b, bd);
    let prec = (dmax + n + 3) as u32;
    let (branches, _skipped) = puiseux_at_algebraic(monos, q, prec);
    // A skipped branch can only *under*-constrain the system → an unverified
    // proposal that `is_integral` (the caller's gate) then rejects; soundness
    // never depends on completeness.
    if branches.is_empty() {
        return None;
    }
    let emax = branches.iter().map(|s| s.ramification).max().unwrap_or(1) as i64;
    let u_bound = (prec as i64 + 3) * emax + 4;

    // Unknown layout: column (i·m + l) ↔ coefficient aᵢₗ of xˡ in aᵢ (i<d, l<m).
    let ncols = d * m;
    // x = α + t^e as a K-series: powers xˡ are reused per branch.
    let mut matrix: Vec<Vec<Rational>> = Vec::new();
    let mut rhs: Vec<Rational> = Vec::new();

    for s in &branches {
        let e = s.ramification as i64;
        let bts = branch_kts(&nf, s);
        let xseries = x_alpha_kts(&nf, &alpha, e); // x along the branch
                                                   // Series of `xˡ` (l = 0..m−1).
        let mut xpows: Vec<Kts> = Vec::with_capacity(m);
        xpows.push(kts_one(&nf));
        for l in 1..m {
            xpows.push(kts_mul(&nf, &xpows[l - 1], &xseries, u_bound));
        }
        // Series of each bᵢ and of bd along this branch (K-coefficients).
        let bi_ts: Vec<Kts> = (0..d)
            .map(|i| elem_kts(&nf, &b[i], &alpha, e, u_bound, &bts))
            .collect();
        let bd_ts = elem_kts(&nf, bd, &alpha, e, u_bound, &bts);
        // For column (i,l): the series of `xˡ·bᵢ` (the linear contribution of
        // aᵢₗ to the numerator).
        let mut col_ts: Vec<Kts> = Vec::with_capacity(ncols);
        for bi in &bi_ts {
            for xl in &xpows {
                col_ts.push(kts_mul(&nf, xl, bi, u_bound));
            }
        }
        // Negative-power range: keys k < e (after dividing by q ~ t^e at this
        // simple, ramification-`e` place — same valuation drop as `(x−α)`).
        let low = col_ts
            .iter()
            .chain(std::iter::once(&bd_ts))
            .filter_map(|s| s.keys().next().copied())
            .min()
            .unwrap_or(0);
        for k in low..e {
            // Each K-coefficient (length m) is m ℚ-equations.
            for comp in 0..m {
                let row: Vec<Rational> = col_ts.iter().map(|s| kts_comp(s, k, comp)).collect();
                let r = -kts_comp(&bd_ts, k, comp);
                matrix.push(row);
                rhs.push(r);
            }
        }
    }
    let a = gauss_solve(matrix, rhs, ncols)?;

    // candidate = (Σ aᵢ(x) bᵢ + bd) · 1/q(x), with aᵢ = Σ_{l<m} a[i·m+l] xˡ.
    let mut num = bd.clone();
    for i in 0..d {
        let ai: QPoly = (0..m).map(|l| a[i * m + l].clone()).collect();
        if degree(&ai) >= 0 {
            num = ext.add(&num, &scale(&b[i], &RatFn::from_poly(&ai)));
        }
    }
    let inv_q = RatFn::new(vec![Rational::from(1)], q.clone());
    Some(scale(&num, &inv_q))
}

// ---------------------------------------------------------------------------
// Laurent-series-in-t arithmetic over K = ℚ(α)  (for algebraic places)
// ---------------------------------------------------------------------------

/// A Laurent series in the place uniformizer `t`, coefficients in `K = ℚ(α)`.
type Kts = BTreeMap<i64, KElem>;

fn kts_one(nf: &NumberField) -> Kts {
    let mut s = Kts::new();
    s.insert(0, nf.from_int(1));
    s
}

/// `α + t^e` as a `K`-series in `t`.
fn x_alpha_kts(nf: &NumberField, alpha: &KElem, e: i64) -> Kts {
    let mut s = Kts::new();
    if !NumberField::is_zero(alpha) {
        s.insert(0, alpha.clone());
    }
    s.insert(e, nf.from_int(1));
    s
}

/// The Puiseux branch `y(t) = Σ c_k t^{exp·e}` (`c_k ∈ K`) as a `Kts`.
fn branch_kts(nf: &NumberField, s: &AlgBasePuiseuxSeries) -> Kts {
    let e = s.ramification as i64;
    let mut ts = Kts::new();
    for (exp, c) in &s.terms {
        let k = (exp.clone() * Rational::from(e))
            .numer()
            .to_i64()
            .unwrap_or(0);
        let slot = ts.entry(k).or_insert_with(NumberField::k_zero);
        *slot = nf.add(slot, c);
    }
    ts.retain(|_, c| !NumberField::is_zero(c));
    ts
}

/// Series of `b = Σⱼ bⱼ(x) yʲ` (`bⱼ ∈ ℚ(x)`) along a `K`-branch `y = bts(t)`,
/// `x = α + t^e`.
fn elem_kts(nf: &NumberField, b: &AlgElem, alpha: &KElem, e: i64, u: i64, bts: &Kts) -> Kts {
    let mut acc = Kts::new();
    for (j, coeff) in b.iter().enumerate() {
        if coeff.numer().is_empty() {
            continue;
        }
        let cj = ratfn_kts(nf, coeff, alpha, e, u);
        let yj = kts_pow(nf, bts, j as u32, u);
        acc = kts_add(nf, &acc, &kts_mul(nf, &cj, &yj, u));
    }
    acc
}

fn ratfn_kts(nf: &NumberField, r: &RatFn, alpha: &KElem, e: i64, u: i64) -> Kts {
    let num = poly_kts(nf, r.numer(), alpha, e, u);
    let den = poly_kts(nf, r.denom(), alpha, e, u);
    match kts_inv(nf, &den, u) {
        Some(inv) => kts_mul(nf, &num, &inv, u),
        None => Kts::new(),
    }
}

/// `p(α + t^e)` (`p ∈ ℚ[x]`) as a `K`-series truncated to `t`-exponents `< u`.
fn poly_kts(nf: &NumberField, p: &QPoly, alpha: &KElem, e: i64, u: i64) -> Kts {
    let mut ts = Kts::new();
    for (mexp, pm) in p.iter().enumerate() {
        if *pm == 0 {
            continue;
        }
        let pmk = nf.from_rational(pm);
        // (α + t^e)^m = Σ_l C(m,l) α^{m−l} t^{e l}.
        for l in 0..=mexp {
            let exp = e * l as i64;
            if exp >= u {
                break;
            }
            let binom = nf.from_rational(&Rational::from(binom(mexp as u32, l as u32)));
            let apow = k_pow(nf, alpha, (mexp - l) as u32);
            let coeff = nf.mul(&nf.mul(&pmk, &binom), &apow);
            if !NumberField::is_zero(&coeff) {
                let slot = ts.entry(exp).or_insert_with(NumberField::k_zero);
                *slot = nf.add(slot, &coeff);
            }
        }
    }
    ts.retain(|_, c| !NumberField::is_zero(c));
    ts
}

fn kts_add(nf: &NumberField, a: &Kts, b: &Kts) -> Kts {
    let mut r = a.clone();
    for (k, c) in b {
        let slot = r.entry(*k).or_insert_with(NumberField::k_zero);
        *slot = nf.add(slot, c);
    }
    r.retain(|_, c| !NumberField::is_zero(c));
    r
}

fn kts_mul(nf: &NumberField, a: &Kts, b: &Kts, u: i64) -> Kts {
    let mut r = Kts::new();
    for (ka, ca) in a {
        for (kb, cb) in b {
            let k = ka + kb;
            if k < u {
                let slot = r.entry(k).or_insert_with(NumberField::k_zero);
                *slot = nf.add(slot, &nf.mul(ca, cb));
            }
        }
    }
    r.retain(|_, c| !NumberField::is_zero(c));
    r
}

fn kts_pow(nf: &NumberField, a: &Kts, m: u32, u: i64) -> Kts {
    let mut acc = kts_one(nf);
    for _ in 0..m {
        acc = kts_mul(nf, &acc, a, u);
    }
    acc
}

/// Inverse of a `K`-Laurent series, truncated to exponents `< u`.  `None` if the
/// leading coefficient is not invertible in `K` (e.g. a zero divisor mod a
/// reducible `q` — which cannot arise for an irreducible discriminant factor).
fn kts_inv(nf: &NumberField, s: &Kts, u: i64) -> Option<Kts> {
    let (&v0, c0) = s.iter().next()?; // lowest exponent
    let inv_c0 = nf.inv(c0)?;
    let kmax = (u + v0).max(1);
    // Normalized series u[k] = s[v0+k]·c0⁻¹  (u[0] = 1).
    let mut us = vec![NumberField::k_zero(); kmax as usize + 1];
    for (exp, c) in s {
        let k = exp - v0;
        if (0..=kmax).contains(&k) {
            us[k as usize] = nf.mul(c, &inv_c0);
        }
    }
    let mut iu = vec![NumberField::k_zero(); kmax as usize + 1];
    iu[0] = nf.from_int(1);
    for k in 1..=kmax as usize {
        let mut acc = NumberField::k_zero();
        for i in 1..=k {
            acc = nf.add(&acc, &nf.mul(&us[i], &iu[k - i]));
        }
        iu[k] = nf.neg(&acc);
    }
    let mut res = Kts::new();
    for (k, iuk) in iu.iter().enumerate() {
        let exp = -v0 + k as i64;
        if exp < u && !NumberField::is_zero(iuk) {
            res.insert(exp, nf.mul(&inv_c0, iuk));
        }
    }
    Some(res)
}

/// The ℚ-component `comp` (coefficient of `αᶜᵒᵐᵖ`) of `s[t^k]`.
fn kts_comp(s: &Kts, k: i64, comp: usize) -> Rational {
    s.get(&k)
        .and_then(|c| c.get(comp))
        .cloned()
        .unwrap_or_else(|| Rational::from(0))
}

/// `c^e` in `K`.
fn k_pow(nf: &NumberField, c: &KElem, e: u32) -> KElem {
    let mut acc = nf.from_int(1);
    for _ in 0..e {
        acc = nf.mul(&acc, c);
    }
    acc
}

// ---------------------------------------------------------------------------
// Laurent-series-in-t arithmetic (over ℚ)
// ---------------------------------------------------------------------------

pub(super) fn branch_ts(s: &PuiseuxSeries) -> TS {
    let e = s.ramification as i64;
    let mut ts = TS::new();
    for (exp, c) in &s.terms {
        // exp has denominator dividing e ⇒ exp·e is an integer t-exponent.
        let k = (exp.clone() * Rational::from(e))
            .numer()
            .to_i64()
            .unwrap_or(0);
        ts.insert(k, c.clone());
    }
    ts
}

/// Series of `b = Σⱼ bⱼ(x) yʲ` along a branch `y = bts(t)`, `x = α + t^e`.
pub(super) fn elem_ts(b: &AlgElem, alpha: &Rational, e: i64, u: i64, bts: &TS) -> TS {
    let mut acc = TS::new();
    for (j, coeff) in b.iter().enumerate() {
        if coeff.numer().is_empty() {
            continue;
        }
        let cj = ratfn_ts(coeff, alpha, e, u);
        let yj = ts_pow(bts, j as u32, u);
        acc = ts_add(&acc, &ts_mul(&cj, &yj, u));
    }
    acc
}

fn ratfn_ts(r: &RatFn, alpha: &Rational, e: i64, u: i64) -> TS {
    let num = poly_ts(r.numer(), alpha, e, u);
    let den = poly_ts(r.denom(), alpha, e, u);
    match ts_inv(&den, u) {
        Some(inv) => ts_mul(&num, &inv, u),
        None => TS::new(),
    }
}

/// `p(α + t^e)` truncated to `t`-exponents `< u`.
fn poly_ts(p: &QPoly, alpha: &Rational, e: i64, u: i64) -> TS {
    let mut ts = TS::new();
    for (m, pm) in p.iter().enumerate() {
        if *pm == 0 {
            continue;
        }
        // (α + t^e)^m = Σ_l C(m,l) α^{m−l} t^{e l}.
        for l in 0..=m {
            let exp = e * l as i64;
            if exp >= u {
                break;
            }
            let coeff = pm.clone()
                * Rational::from(binom(m as u32, l as u32))
                * rat_pow(alpha, (m - l) as u32);
            if coeff != 0 {
                *ts.entry(exp).or_insert_with(|| Rational::from(0)) += &coeff;
            }
        }
    }
    ts.retain(|_, c| *c != 0);
    ts
}

pub(super) fn ts_add(a: &TS, b: &TS) -> TS {
    let mut r = a.clone();
    for (k, c) in b {
        *r.entry(*k).or_insert_with(|| Rational::from(0)) += c;
    }
    r.retain(|_, c| *c != 0);
    r
}

pub(super) fn ts_mul(a: &TS, b: &TS, u: i64) -> TS {
    let mut r = TS::new();
    for (ka, ca) in a {
        for (kb, cb) in b {
            let k = ka + kb;
            if k < u {
                *r.entry(k).or_insert_with(|| Rational::from(0)) += ca.clone() * cb;
            }
        }
    }
    r.retain(|_, c| *c != 0);
    r
}

pub(super) fn ts_pow(a: &TS, m: u32, u: i64) -> TS {
    let mut acc = TS::new();
    acc.insert(0, Rational::from(1));
    for _ in 0..m {
        acc = ts_mul(&acc, a, u);
    }
    acc
}

/// Inverse of a Laurent series, truncated to exponents `< u`.
pub(super) fn ts_inv(s: &TS, u: i64) -> Option<TS> {
    let (&v0, c0) = s.iter().next()?; // lowest exponent
    let inv_c0 = Rational::from(1) / c0.clone();
    let kmax = (u + v0).max(1);
    // Normalized u-series coefficients: u[k] = s[v0+k]/c0  (u[0] = 1).
    let mut us = vec![Rational::from(0); kmax as usize + 1];
    for (exp, c) in s {
        let k = exp - v0;
        if (0..=kmax).contains(&k) {
            us[k as usize] = c.clone() * &inv_c0;
        }
    }
    let mut iu = vec![Rational::from(0); kmax as usize + 1];
    iu[0] = Rational::from(1);
    for k in 1..=kmax as usize {
        let mut acc = Rational::from(0);
        for i in 1..=k {
            acc += us[i].clone() * &iu[k - i];
        }
        iu[k] = -acc;
    }
    let mut res = TS::new();
    for (k, iuk) in iu.iter().enumerate() {
        let exp = -v0 + k as i64;
        if exp < u && *iuk != 0 {
            res.insert(exp, inv_c0.clone() * iuk);
        }
    }
    Some(res)
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// If `F = yⁿ − p(x)` is a **simple radical** (monic in `y`, only the top and
/// constant `y`-coefficients nonzero), return `p`.  `None` otherwise.
fn as_simple_radical(f_coeffs: &[QPoly], n: usize) -> Option<QPoly> {
    if f_coeffs.len() != n + 1 {
        return None;
    }
    // Monic leading coefficient.
    if f_coeffs[n].len() != 1 || f_coeffs[n][0] != 1 {
        return None;
    }
    // All middle coefficients (1..n) must vanish.
    if f_coeffs[1..n].iter().any(|c| degree(c) >= 0) {
        return None;
    }
    // p = −(constant coefficient); must be nonzero.
    let p: QPoly = f_coeffs[0].iter().map(|c| -c.clone()).collect();
    if degree(&p) < 0 {
        return None;
    }
    Some(p)
}

fn to_monomials(f_coeffs: &[QPoly]) -> Vec<(u32, u32, Rational)> {
    let mut out = Vec::new();
    for (j, cj) in f_coeffs.iter().enumerate() {
        for (i, c) in cj.iter().enumerate() {
            if *c != 0 {
                out.push((i as u32, j as u32, c.clone()));
            }
        }
    }
    out
}

fn max_denom_degree(b: &[AlgElem], bd: &AlgElem) -> usize {
    let one = b
        .iter()
        .chain(std::iter::once(bd))
        .flat_map(|e| e.iter())
        .map(|c| degree(c.denom()).max(0) as usize)
        .max()
        .unwrap_or(0);
    one
}

fn scale(b: &AlgElem, s: &RatFn) -> AlgElem {
    let f = RationalFunctionField;
    b.iter().map(|c| f.mul(s, c)).collect()
}

fn binom(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::from(0);
    }
    let mut num = Integer::from(1);
    for t in 0..k {
        num *= Integer::from(n - t);
    }
    let mut den = Integer::from(1);
    for t in 1..=k {
        den *= Integer::from(t);
    }
    num / den
}

fn rat_pow(c: &Rational, e: u32) -> Rational {
    let mut acc = Rational::from(1);
    for _ in 0..e {
        acc *= c;
    }
    acc
}

/// Solve `M·a = rhs` over `ℚ` for `ncols` unknowns; particular solution (free
/// vars 0) or `None` if inconsistent.
fn gauss_solve(
    mut m: Vec<Vec<Rational>>,
    mut b: Vec<Rational>,
    ncols: usize,
) -> Option<Vec<Rational>> {
    let nrows = m.len();
    let mut pivot_of_col = vec![None; ncols];
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
            *v /= &piv;
        }
        b[row] /= &piv;
        let pr = m[row].clone();
        let pb = b[row].clone();
        for r in 0..nrows {
            if r != row && m[r][col] != 0 {
                let f = m[r][col].clone();
                for (dst, pv) in m[r].iter_mut().zip(pr.iter()) {
                    *dst -= f.clone() * pv;
                }
                b[r] -= f * &pb;
            }
        }
        pivot_of_col[col] = Some(row);
        row += 1;
    }
    for r in 0..nrows {
        if m[r].iter().all(|v| *v == 0) && b[r] != 0 {
            return None; // inconsistent
        }
    }
    let mut x = vec![Rational::from(0); ncols];
    for (col, pr) in pivot_of_col.iter().enumerate() {
        if let Some(r) = pr {
            x[col] = b[*r].clone();
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    /// Power basis for a non-singular curve: y² − x − 1 (disc = −4, no repeated
    /// roots) ⇒ {1, y}.
    #[test]
    fn nonsingular_power_basis() {
        let basis = integral_basis(&[qp(&[-1, -1]), qp(&[]), qp(&[1])]).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], vec![RatFn::int(1)]);
        assert_eq!(basis[1], vec![RatFn::int(0), RatFn::int(1)]);
    }

    /// Cusp y² = x³ ⇒ {1, y/x}: a genuine enlargement at the singular x=0.
    #[test]
    fn cusp_basis() {
        let basis = integral_basis(&[qp(&[0, 0, 0, -1]), qp(&[]), qp(&[1])]).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], vec![RatFn::int(1)]);
        // b₁ = y/x.
        assert_eq!(
            basis[1],
            vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, 1]))]
        );
        // All integral.
        let f = [qp(&[0, 0, 0, -1]), qp(&[]), qp(&[1])];
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }

    /// Nodal cubic y² = x³ + x² (non-radical) ⇒ {1, y/x}; (y/x)² = x + 1.
    #[test]
    fn nodal_cubic_basis() {
        // F = y² − x² − x³.
        let f = [qp(&[0, 0, -1, -1]), qp(&[]), qp(&[1])];
        let basis = integral_basis(&f).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(
            basis[1],
            vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, 1]))]
        );
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }

    /// Radical fast-path: `integral_basis` on a simple radical short-circuits to
    /// the explicit Trager basis (no Puiseux loop) and matches it.  Degree-4
    /// radical y⁴ = x³ ⇒ {1, y, y²/x, y³/x²}.
    #[test]
    fn radical_fast_path_matches_explicit() {
        let f = [qp(&[0, 0, 0, -1]), qp(&[]), qp(&[]), qp(&[]), qp(&[1])]; // y⁴ − x³
        let via_loop = integral_basis(&f).expect("basis");
        let explicit =
            super::super::integral_basis::radical_integral_basis(4, &qp(&[0, 0, 0, 1])).unwrap();
        assert_eq!(via_loop, explicit);
        assert_eq!(via_loop.len(), 4);
        assert_eq!(
            via_loop[2],
            vec![
                RatFn::int(0),
                RatFn::int(0),
                RatFn::new(qp(&[1]), qp(&[0, 1]))
            ]
        ); // y²/x
        assert!(via_loop.iter().all(|bi| is_integral(&f, bi)));
    }

    /// Algebraic singular place: `y² = (x²−2)²·(x+3)` is singular only at the
    /// **irrational** points `x = ±√2` (disc `= 4(x+3)(x²−2)²`, the only repeated
    /// factor `q = x²−2` is irreducible of degree 2).  The rational loop never
    /// enlarges; the algebraic loop must produce `w = y/(x²−2)` (integral, since
    /// `w² = x+3`).
    #[test]
    fn algebraic_singular_place_sqrt2() {
        // F = y² − (x²−2)²(x+3) = y² − (x⁵ + 3x⁴ − 4x³ − 12x² + 4x + 12).
        let f = [qp(&[-12, -4, 12, 4, -3, -1]), qp(&[]), qp(&[1])];
        let basis = integral_basis(&f).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], vec![RatFn::int(1)]);
        // b₁ = y/(x²−2): component 1 is 1/(x²−2).
        assert_eq!(
            basis[1],
            vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[-2, 0, 1]))]
        );
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }

    /// The same `q = x²−2` factor appears in the discriminant, but only to the
    /// **first** power (`y² = (x²−2)(x+3)`, disc `= 4(x²−2)(x+3)`): no algebraic
    /// singular place, so the basis stays the power basis `{1, y}` (control — no
    /// enlargement should be attempted/accepted).
    #[test]
    fn algebraic_factor_simple_no_enlargement() {
        // F = y² − (x²−2)(x+3) = y² − (x³ + 3x² − 2x − 6).
        let f = [qp(&[-6, -2, 3, 1]), qp(&[]), qp(&[1])];
        // Sanity: q=x²−2 is not a repeated discriminant factor here.
        let disc = discriminant(&f);
        assert!(algebraic_singularities(&disc).is_empty());
        let basis = integral_basis(&f).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], vec![RatFn::int(1)]);
        assert_eq!(basis[1], vec![RatFn::int(0), RatFn::int(1)]);
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }

    /// Agreement with the radical fast-path where both apply: a simple radical
    /// `y² = x²−2` whose (irrational) branch points `±√2` are **simple** disc
    /// roots is non-singular; both paths give the power basis `{1, y}`.  (The
    /// fast-path short-circuits; this checks the general loop would agree.)
    #[test]
    fn radical_with_irrational_branch_points() {
        let f = [qp(&[2, 0, -1]), qp(&[]), qp(&[1])]; // y² − x² + 2 = y² − (x²−2)
        let basis = integral_basis(&f).expect("basis");
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], vec![RatFn::int(1)]);
        assert_eq!(basis[1], vec![RatFn::int(0), RatFn::int(1)]);
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }

    /// Degree-3 radical y³ = x² ⇒ {1, y, y²/x}.
    #[test]
    fn cubic_radical_basis() {
        let f = [qp(&[0, 0, 1]), qp(&[]), qp(&[]), qp(&[1])]; // y³ − x²
        let basis = integral_basis(&f).expect("basis");
        assert_eq!(basis.len(), 3);
        assert_eq!(basis[1], vec![RatFn::int(0), RatFn::int(1)]); // y
        assert_eq!(
            basis[2],
            vec![
                RatFn::int(0),
                RatFn::int(0),
                RatFn::new(qp(&[1]), qp(&[0, 1]))
            ]
        ); // y²/x
        assert!(basis.iter().all(|bi| is_integral(&f, bi)));
    }
}
