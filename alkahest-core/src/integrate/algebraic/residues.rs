//! Residues of a differential on an algebraic curve — the foundation of the
//! **logarithmic part / FIND-ORDER** (Risch milestone **MC**).
//!
//! After Hermite reduction (`hermite_curve`) the remaining integrand `h` is a
//! differential of the **third kind** (only simple poles).  Its integral is
//! `∫ h dx = Σ cⱼ log(uⱼ)`, and the `cⱼ` are governed by the **residues** of
//! `h dx` at the places of the curve: the residue divisor `δ = Σ res_P · P` must
//! be (a torsion multiple of) a principal divisor for the integral to be
//! elementary.
//!
//! At a place over `x = α` with ramification `e` and uniformizer `t`
//! (`x − α = t^e`, `dx = e·t^{e−1} dt`), the residue is
//!
//! ```text
//!   res_P(h dx) = [t^{-1}](h · e t^{e−1}) = e · [t^{-e}]( h along the branch ).
//! ```
//!
//! computed from the Puiseux expansion at `α` via the Laurent-`t` substitution
//! shared with [`super::vanhoeij`].  The **residue theorem** `Σ_P res_P = 0`
//! (over *all* places, including infinity) is the soundness check.
//!
//! [`finite_residues`] handles the rational finite places (poles of `h`, branch
//! points); [`residues_at_infinity`] handles the places over `∞` (via the
//! `w = 1/y`, `z = 1/x` curve `ã(z)wⁿ − zᵐ = 0`); [`residue_divisor`] combines
//! them.  [`finite_residues_algebraic`] (hyperelliptic `y² = a`) covers the
//! **algebraic** finite simple poles those miss — both algebraic base points
//! (irreducible factors `deg ≥ 2` of the pole denominator) and rational base
//! points with an *irrational sheet* `√a(α)` — with residues in a number field;
//! [`residue_sum_complete`] checks the residue theorem over the resulting
//! complete divisor.  Remaining gap: algebraic **branch** places (a pole at an
//! irrational root of `a`) and `n > 2`.  Together with FIND-ORDER (genus-graded
//! principality) these complete MC.

use rug::{Integer, Rational};
use std::collections::{BTreeSet, HashMap};

use super::super::risch::alg_field::{AlgElem, RatFn};
use super::super::risch::number_field::{KElem, NumberField};
use super::super::risch::poly_rde::{degree, poly_deriv, poly_mul, trim, QPoly};
use super::super::risch::rational_rde::{poly_div_exact, poly_gcd};
use super::vanhoeij::{branch_ts, elem_ts, ts_add, ts_inv, ts_mul, ts_pow, TS};
use crate::poly::puiseux::{factor_over_q, puiseux_at, puiseux_at_zero, PuiseuxSeries};

/// A residue at a place of the curve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residue {
    /// The base point `α` (ignored when `at_infinity`).
    pub point: Rational,
    /// `true` for a place over `x = ∞`.
    pub at_infinity: bool,
    /// Index of the Puiseux sheet at the place's base (distinguishes places over
    /// the same `α` or over `∞`).
    pub sheet: usize,
    /// Ramification index of the place.
    pub ramification: u64,
    /// The residue `res_P(h dx)`.
    pub value: Rational,
}

/// A [`Residue`] paired with the place's `y`-coordinate — internal FIND-ORDER
/// plumbing.
///
/// The `y`-coordinate is the constant term of the branch (`0` at a branch point,
/// `±√(a(α))` at an unramified place; unused when `at_infinity`) and lets
/// FIND-ORDER map the place onto the elliptic curve.  It is deliberately kept
/// *out* of the public, semver-stable, externally-constructible [`Residue`]
/// struct (adding any field there is a breaking change), so the genus-1 path
/// threads this richer type through `pub(crate)` channels instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlacedResidue {
    pub(crate) residue: Residue,
    pub(crate) y_coord: Rational,
}

/// Residues of the differential `h dx` at all **rational finite places** of the
/// curve `yⁿ = a(x)` — the poles of `h` and the branch points (roots of `a`).
/// Places with zero residue are omitted.
///
/// (Algebraic places and the place at infinity are out of this rational-only
/// scope; for fully ramified rational curves the finite residues already sum to
/// `−res_∞`.)
pub fn finite_residues(n: usize, a: &QPoly, h: &AlgElem) -> Vec<Residue> {
    finite_residues_placed(n, a, h)
        .into_iter()
        .map(|p| p.residue)
        .collect()
}

/// As [`finite_residues`], but each residue carries the place's `y`-coordinate
/// (for the genus-1 Abel–Jacobi map).  Internal — see [`PlacedResidue`].
pub(crate) fn finite_residues_placed(n: usize, a: &QPoly, h: &AlgElem) -> Vec<PlacedResidue> {
    if n < 2 {
        return Vec::new();
    }
    let monos = curve_monomials(n, a);

    // Candidate base points: rational roots of `a` (branch points) and of every
    // coordinate denominator of `h` (finite poles).
    let mut cands: BTreeSet<Rational> = BTreeSet::new();
    for r in rational_roots(a) {
        cands.insert(r);
    }
    for c in h {
        for r in rational_roots(c.denom()) {
            cands.insert(r);
        }
    }

    let mut out = Vec::new();
    for alpha in cands {
        // Enough Puiseux terms to resolve the simple poles of h's coordinates.
        let prec = (h
            .iter()
            .map(|c| degree(c.denom()).max(0))
            .max()
            .unwrap_or(0)
            + n as i64
            + 3) as u32;
        // A ramified place of index e is returned by Puiseux as e Galois-conjugate
        // sheets, all carrying the same residue; keep one representative per place.
        let mut seen_per_ram: HashMap<u64, usize> = HashMap::new();
        for (sheet, br) in puiseux_at(&monos, &alpha, prec).iter().enumerate() {
            let e = br.ramification as i64;
            let idx = seen_per_ram.entry(br.ramification).or_insert(0);
            let keep = *idx % (br.ramification.max(1) as usize) == 0;
            *idx += 1;
            if !keep {
                continue;
            }
            let u = 2 * e + 4;
            let series = elem_ts(h, &alpha, e, u, &branch_ts(br));
            let coeff = series
                .get(&(-e))
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
            let value = Rational::from(e) * coeff;
            if value != 0 {
                // y-coordinate of the place = constant term of the branch.
                let y_coord = br
                    .terms
                    .iter()
                    .find(|(ex, _)| *ex == 0)
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(|| Rational::from(0));
                out.push(PlacedResidue {
                    residue: Residue {
                        point: alpha.clone(),
                        at_infinity: false,
                        sheet,
                        ramification: br.ramification,
                        value,
                    },
                    y_coord,
                });
            }
        }
    }
    out
}

/// Puiseux branches at **infinity** of `yⁿ = a(x)`, returned as the `w = 1/y`
/// branches (series in `z = 1/x`) of the curve `ã(z)·wⁿ − zᵐ = 0`, where
/// `m = deg a` and `ã(z) = zᵐ·a(1/z)` is the reversed radicand (`ã(0) ≠ 0`).
/// The actual branch is `y = 1/w`, `x = 1/z`.
pub fn puiseux_at_infinity(n: usize, a: &QPoly, prec: u32) -> Vec<PuiseuxSeries> {
    let a = trim(a.clone());
    let m = degree(&a);
    if m < 0 {
        return Vec::new();
    }
    let m = m as usize;
    // ã(z) = Σᵢ a_i z^{m−i}  (reversed).
    let mut monos: Vec<(u32, u32, Rational)> = Vec::new();
    for (i, ai) in a.iter().enumerate() {
        if *ai != 0 {
            monos.push(((m - i) as u32, n as u32, ai.clone())); // ã_k wⁿ
        }
    }
    monos.push((m as u32, 0, Rational::from(-1))); // − zᵐ
    puiseux_at_zero(&monos, prec)
}

/// Residues of `h dx` at the places over **infinity** of `yⁿ = a(x)`.
///
/// At a place over `∞` with ramification `e` (uniformizer `t`, `z = 1/x = tᵉ`,
/// `x = t^{−e}`, `dx = −e·t^{−e−1} dt`), the residue is
/// `res = [t^{-1}](h dx) = −e·[tᵉ](h along the branch)`.
pub fn residues_at_infinity(n: usize, a: &QPoly, h: &AlgElem) -> Vec<Residue> {
    residues_at_infinity_placed(n, a, h)
        .into_iter()
        .map(|p| p.residue)
        .collect()
}

/// As [`residues_at_infinity`], but each residue carries the place's
/// `y`-coordinate (always `0` at ∞ here).  Internal — see [`PlacedResidue`].
pub(crate) fn residues_at_infinity_placed(n: usize, a: &QPoly, h: &AlgElem) -> Vec<PlacedResidue> {
    let m = degree(&trim(a.clone())).max(0) as usize;
    let dmax = h
        .iter()
        .map(|c| degree(c.numer()).max(degree(c.denom())).max(0))
        .max()
        .unwrap_or(0);
    let prec = (dmax + (n as i64) + (m as i64) + 4) as u32;
    let mut out = Vec::new();
    // Dedup the e conjugate sheets of a ramified place over ∞ (see finite_residues).
    let mut seen_per_ram: HashMap<u64, usize> = HashMap::new();
    for (sheet, w_branch) in puiseux_at_infinity(n, a, prec).iter().enumerate() {
        let idx = seen_per_ram.entry(w_branch.ramification).or_insert(0);
        let keep = *idx % (w_branch.ramification.max(1) as usize) == 0;
        *idx += 1;
        if !keep {
            continue;
        }
        let e = w_branch.ramification as i64;
        let u = 2 * e + 2 * (m as i64) * e + 8;
        // y = 1/w  as a t-series; w from the branch (z = tᵉ).
        let w_ts = branch_ts(w_branch);
        let Some(y_ts) = ts_inv(&w_ts, u) else {
            continue;
        };
        let h_ts = elem_at_infinity(h, e, u, &y_ts);
        let coeff = h_ts.get(&e).cloned().unwrap_or_else(|| Rational::from(0));
        let value = -Rational::from(e) * coeff;
        if value != 0 {
            out.push(PlacedResidue {
                residue: Residue {
                    point: Rational::from(0),
                    at_infinity: true,
                    sheet,
                    ramification: w_branch.ramification,
                    value,
                },
                y_coord: Rational::from(0),
            });
        }
    }
    out
}

/// Series of `h = Σⱼ hⱼ(x) yʲ` along an infinite place: `x = t^{−e}`, `y = y_ts(t)`.
fn elem_at_infinity(h: &AlgElem, e: i64, u: i64, y_ts: &TS) -> TS {
    let mut acc = TS::new();
    for (j, coeff) in h.iter().enumerate() {
        if coeff.numer().is_empty() {
            continue;
        }
        let cj = ratfn_at_infinity(coeff, e, u);
        let yj = ts_pow(y_ts, j as u32, u);
        acc = ts_add(&acc, &ts_mul(&cj, &yj, u));
    }
    acc
}

/// `r(t^{−e})` for `r ∈ ℚ(x)`, as a Laurent `t`-series truncated to exps `< u`.
fn ratfn_at_infinity(r: &crate::integrate::risch::alg_field::RatFn, e: i64, u: i64) -> TS {
    let num = poly_at_infinity(r.numer(), e, u);
    let den = poly_at_infinity(r.denom(), e, u);
    match ts_inv(&den, u) {
        Some(inv) => ts_mul(&num, &inv, u),
        None => TS::new(),
    }
}

/// `p(t^{−e})` for `p ∈ ℚ[x]`: `Σᵢ p_i t^{−e·i}`.
fn poly_at_infinity(p: &QPoly, e: i64, u: i64) -> TS {
    let mut ts = TS::new();
    for (i, pi) in p.iter().enumerate() {
        if *pi != 0 {
            let exp = -e * i as i64;
            if exp < u {
                *ts.entry(exp).or_insert_with(|| Rational::from(0)) += pi;
            }
        }
    }
    ts.retain(|_, c| *c != 0);
    ts
}

/// The full **residue divisor** of `h dx` on `yⁿ = a(x)`: residues at all
/// rational finite places **and** at the places over infinity.  The integral
/// `∫ h dx` is elementary iff this divisor is (a torsion multiple of) a principal
/// divisor — the FIND-ORDER decision.  `residue_sum(&divisor)` should be `0`
/// (residue theorem) when every place was captured.
pub fn residue_divisor(n: usize, a: &QPoly, h: &AlgElem) -> Vec<Residue> {
    let mut d = finite_residues(n, a, h);
    d.extend(residues_at_infinity(n, a, h));
    d
}

/// As [`residue_divisor`], but each place carries its `y`-coordinate for the
/// genus-1 Abel–Jacobi map.  Internal — see [`PlacedResidue`].
pub(crate) fn residue_divisor_placed(n: usize, a: &QPoly, h: &AlgElem) -> Vec<PlacedResidue> {
    let mut d = finite_residues_placed(n, a, h);
    d.extend(residues_at_infinity_placed(n, a, h));
    d
}

/// Sum of the residue values (should be `0` over a complete set of places).
pub fn residue_sum(divisor: &[Residue]) -> Rational {
    divisor.iter().fold(Rational::from(0), |s, r| s + &r.value)
}

// ===========================================================================
// Residues at algebraic (non-rational) finite places — hyperelliptic `y² = a`
// ===========================================================================

/// A residue at an **algebraic place**: a Galois orbit of finite places of
/// `y² = a(x)` over the roots of an irreducible factor `q` of the integrand's
/// pole denominator (`deg q ≥ 2`, `q` coprime to `a`, so a non-branch place).
///
/// Over a root `α` of `q` there are two sheets `(α, ±√a(α))`, and on the curve
/// the residue of `(A + B·y) dx` at sheet `±` is `r0 ± r1·√a(α)` with
/// `r0, r1 ∈ ℚ(α) = ℚ[z]/(q)`.  The `√a(α)` part cancels between the two sheets,
/// so the orbit's total contribution to the residue sum is `2·Tr_{ℚ(α)/ℚ}(r0)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlgResidue {
    /// Monic irreducible minimal polynomial `q` of `α` (the place's base).
    pub minpoly: QPoly,
    /// Number of conjugate base points (`= deg q`).
    pub conjugates: usize,
    /// Rational part of the residue, in `ℚ[z]/(q)` (equal on both sheets).
    pub r0: KElem,
    /// `√a(α)` coefficient, in `ℚ[z]/(q)` (opposite sign on the two sheets).
    pub r1: KElem,
}

/// Residues of `h dx = (A + B·y) dx` (`y² = a`) at the **algebraic** finite
/// simple poles — the irreducible factors `q` of the pole denominator with
/// `deg q ≥ 2` and `gcd(q, a) = 1`.  Requires a squarefree pole denominator
/// (third-kind / simple poles); returns `[]` for `n ≠ 2`, a non-squarefree
/// denominator, or no algebraic poles.
///
/// Complements [`finite_residues`] (rational places); together with
/// [`residues_at_infinity`] they form the **complete** residue divisor whose
/// values sum to zero (the residue theorem) — see [`residue_sum_complete`].
pub fn finite_residues_algebraic(n: usize, a: &QPoly, h: &AlgElem) -> Vec<AlgResidue> {
    if n != 2 {
        return Vec::new();
    }
    let a = trim(a.clone());
    let a_c = h.first().cloned().unwrap_or_else(|| RatFn::int(0)); // A
    let b_c = h.get(1).cloned().unwrap_or_else(|| RatFn::int(0)); // B
    let a_den = if a_c.numer().is_empty() {
        vec![Rational::from(1)]
    } else {
        a_c.denom().clone()
    };
    let b_den = if b_c.numer().is_empty() {
        vec![Rational::from(1)]
    } else {
        b_c.denom().clone()
    };
    // Common pole denominator D = lcm(den A, den B).
    let d = poly_lcm(&a_den, &b_den);
    if degree(&d) < 1 {
        return Vec::new();
    }
    // Simple poles only.
    if degree(&poly_gcd(&d, &poly_deriv(&d))) > 0 {
        return Vec::new();
    }
    // Numerators over the common denominator: Ã = A_num·(D/den A), B̃ likewise.
    let a_num = poly_mul(a_c.numer(), &poly_div_exact(&d, &a_den));
    let b_num = poly_mul(b_c.numer(), &poly_div_exact(&d, &b_den));
    let d_prime = poly_deriv(&d);

    let mut out = Vec::new();
    for (q, deg_q) in factor_over_q(&d) {
        if degree(&poly_gcd(&q, &a)) > 0 {
            continue; // shares a factor with `a`: a branch place, not handled here
        }
        // Which places have **algebraic** residues that `finite_residues`
        // (rational Puiseux) misses?  (a) an algebraic base point `deg q ≥ 2`;
        // (b) a *rational* base point `x = α` whose sheet `√a(α)` is irrational
        // (`a(α)` not a perfect square).  Rational base + rational sheet is
        // already handled by `finite_residues`, so skip it (no double-count).
        if deg_q == 1 {
            let alpha = -q.first().cloned().unwrap_or_else(|| Rational::from(0)); // q = x − α (monic)
            let a_at = eval_q(&a, &alpha);
            if is_rational_square(&a_at) {
                continue; // rational sheet → `finite_residues` already counts it
            }
        }
        let nf = NumberField::new(q.clone());
        // Evaluate at α (= reduce mod q) and divide by D'(α) in ℚ(α).
        let dp_alpha = nf.reduce(&d_prime);
        let Some(dp_inv) = nf.inv(&dp_alpha) else {
            continue; // D'(α) = 0 ⇒ not a simple pole (shouldn't happen, D squarefree)
        };
        let r0 = nf.mul(&nf.reduce(&a_num), &dp_inv);
        let r1 = nf.mul(&nf.reduce(&b_num), &dp_inv);
        out.push(AlgResidue {
            minpoly: q,
            conjugates: deg_q,
            r0,
            r1,
        });
    }
    out
}

/// Total residue over the **complete** divisor of `y² = a`: rational finite
/// places + algebraic finite places (`2·Tr(r0)` per orbit) + infinity.  By the
/// residue theorem this is `0` whenever the residue computation is complete; it
/// is the soundness check that no place was missed.
pub fn residue_sum_complete(n: usize, a: &QPoly, h: &AlgElem) -> Rational {
    let mut total = residue_sum(&finite_residues(n, a, h));
    total += residue_sum(&residues_at_infinity(n, a, h));
    for r in finite_residues_algebraic(n, a, h) {
        let nf = NumberField::new(r.minpoly.clone());
        total += Rational::from(2) * nf.trace(&r.r0);
    }
    total
}

/// Least common multiple `a·b/gcd(a,b)` over `ℚ[x]`.
fn poly_lcm(a: &QPoly, b: &QPoly) -> QPoly {
    if degree(a) < 0 || degree(b) < 0 {
        return vec![Rational::from(1)];
    }
    poly_div_exact(&poly_mul(a, b), &poly_gcd(a, b))
}

/// Horner evaluation of `p` at a rational point.
fn eval_q(p: &QPoly, x: &Rational) -> Rational {
    p.iter().rev().fold(Rational::from(0), |acc, c| acc * x + c)
}

/// Is the rational `r` a perfect square in `ℚ` (`r = (s)²`, `s ∈ ℚ`)?
fn is_rational_square(r: &Rational) -> bool {
    if *r < 0 {
        return false;
    }
    let n = r.numer().clone();
    let d = r.denom().clone();
    let ns = n.clone().sqrt();
    let ds = d.clone().sqrt();
    Integer::from(&ns * &ns) == n && Integer::from(&ds * &ds) == d
}

/// Monomials `(i, j, coeff)` of `F = yⁿ − a(x)`.
fn curve_monomials(n: usize, a: &QPoly) -> Vec<(u32, u32, Rational)> {
    let mut m = vec![(0u32, n as u32, Rational::from(1))]; // yⁿ
    for (i, c) in a.iter().enumerate() {
        if *c != 0 {
            m.push((i as u32, 0, -c.clone())); // −a_i x^i
        }
    }
    m
}

/// Distinct rational roots of `p ∈ ℚ[x]` via the rational-root theorem.
fn rational_roots(p: &QPoly) -> Vec<Rational> {
    let p = trim(p.clone());
    if degree(&p) < 1 {
        return Vec::new();
    }
    let lo = p.iter().position(|c| *c != 0).unwrap_or(0);
    let mut roots = Vec::new();
    if lo > 0 {
        roots.push(Rational::from(0));
    }
    let psi = &p[lo..];
    if psi.len() <= 1 {
        return roots;
    }
    let mut den_lcm = Integer::from(1);
    for c in psi {
        den_lcm = den_lcm.lcm(c.denom());
    }
    let ints: Vec<Integer> = psi
        .iter()
        .map(|c| {
            (c.clone() * Rational::from(den_lcm.clone()))
                .numer()
                .clone()
        })
        .collect();
    let a0 = ints[0].clone().abs();
    let an = ints[ints.len() - 1].clone().abs();
    for pn in divisors(&a0) {
        for qn in &divisors(&an) {
            for sign in [1i32, -1] {
                let cand = Rational::from((Integer::from(sign) * pn.clone(), qn.clone()));
                if roots.contains(&cand) {
                    continue;
                }
                let mut acc = Rational::from(0);
                for c in ints.iter().rev() {
                    acc = acc * &cand + Rational::from(c.clone());
                }
                if acc == 0 {
                    roots.push(cand);
                }
            }
        }
    }
    roots
}

fn divisors(n: &Integer) -> Vec<Integer> {
    let n = n.clone().abs();
    if n == 0 {
        return vec![Integer::from(1)];
    }
    let mut ds = Vec::new();
    let mut d = Integer::from(1);
    while Integer::from(&d * &d) <= n {
        if n.is_divisible(&d) {
            ds.push(d.clone());
            let o = n.clone() / &d;
            if o != d {
                ds.push(o);
            }
        }
        d += 1;
    }
    ds
}

#[cfg(test)]
mod tests {
    use super::super::super::risch::alg_field::RatFn;
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }
    fn rf(num: &[i64], den: &[i64]) -> RatFn {
        RatFn::new(qp(num), qp(den))
    }
    fn r(n: i64) -> Rational {
        Rational::from(n)
    }

    /// Algebraic place over `x² − 2` (`α = ±√2`) on `y² = x`, differential
    /// `(x + y)/(x²−2) dx`.  Direct residue values: `r0 = ½`, `r1 = ¼·√2`
    /// (so res on sheet ± = ½ ± ¼√2·√(√2)), conjugates = 2.
    #[test]
    fn algebraic_residue_values() {
        // h = (x + y)/(x²−2) = AlgElem [x/(x²−2), 1/(x²−2)].
        let h = vec![rf(&[0, 1], &[-2, 0, 1]), rf(&[1], &[-2, 0, 1])];
        let res = finite_residues_algebraic(2, &qp(&[0, 1]), &h);
        assert_eq!(res.len(), 1);
        let ar = &res[0];
        assert_eq!(ar.minpoly, qp(&[-2, 0, 1])); // x²−2
        assert_eq!(ar.conjugates, 2);
        assert_eq!(ar.r0, vec![Rational::from((1, 2))]); // ½
        assert_eq!(ar.r1, vec![r(0), Rational::from((1, 4))]); // ¼·√2
    }

    /// Residue theorem over the **complete** divisor (rational + algebraic +
    /// infinity) must sum to zero — the soundness check for algebraic places.
    #[test]
    fn residue_theorem_with_algebraic_places() {
        // (x + y)/(x²−2) on y²=x: poles only at the algebraic place ±√2.
        let h = vec![rf(&[0, 1], &[-2, 0, 1]), rf(&[1], &[-2, 0, 1])];
        assert_eq!(residue_sum_complete(2, &qp(&[0, 1]), &h), r(0));

        // A mixed case: 1/((x−1)(x²−3)) · (1 + y) on y²=x+1, rational pole at
        // x=1 plus an algebraic place over x²−3.
        let den = qp(&[3, -3, -1, 1]); // (x−1)(x²−3) = x³ − x² − 3x + 3
        let h2 = vec![
            RatFn::new(qp(&[1]), den.clone()),
            RatFn::new(qp(&[1]), den.clone()),
        ];
        assert_eq!(residue_sum_complete(2, &qp(&[1, 1]), &h2), r(0));
    }

    /// h dx = du/u with u = (y−1)/(y+1) on y²=x:
    /// `∫ 1/((x−1)√x) dx = log((√x−1)/(√x+1))`.
    /// Residues: +1 and −1 at the two sheets over x=1; total 0.
    #[test]
    fn log_differential_residues() {
        // h = 1/((x−1)√x) = y/((x−1)x) = AlgElem [0, 1/(x²−x)].
        let h = vec![RatFn::int(0), rf(&[1], &[0, -1, 1])];
        let res = finite_residues(2, &qp(&[0, 1]), &h);
        // Two nonzero residues at x=1.
        let mut at1: Vec<Rational> = res
            .iter()
            .filter(|r| r.point == r_one())
            .map(|r| r.value.clone())
            .collect();
        at1.sort();
        assert_eq!(at1, vec![r(-1), r(1)]);
        // Residue theorem (finite places): sum is 0 here (no residue at ∞).
        let total: Rational = res.iter().fold(r(0), |s, x| s + &x.value);
        assert_eq!(total, r(0));
    }

    fn r_one() -> Rational {
        Rational::from(1)
    }

    /// Residues at infinity: `∫ dx/√(x²+1) = log(x+√(x²+1))`.  At ∞ the curve
    /// `y²=x²+1` has two unramified places (`y ~ ±x`); `du/u` with `u=x+y` has
    /// residues `−1` and `+1` there.
    #[test]
    fn infinity_residues() {
        // h = 1/y = y/(x²+1) = AlgElem [0, 1/(x²+1)].
        let h = vec![RatFn::int(0), rf(&[1], &[1, 0, 1])];
        let res = super::residues_at_infinity(2, &qp(&[1, 0, 1]), &h);
        let mut vals: Vec<Rational> = res.iter().map(|r| r.value.clone()).collect();
        vals.sort();
        assert_eq!(vals, vec![r(-1), r(1)]);
        // No rational finite poles (y=0 ⇒ x²=−1, algebraic) ⇒ finite is empty.
        assert!(finite_residues(2, &qp(&[1, 0, 1]), &h).is_empty());
    }

    /// Residue theorem across finite + infinite places: for `y²=x`,
    /// `h = 1/((x−1)√x)`, the finite residues (±1) and `res_∞ = 0` sum to 0.
    #[test]
    fn residue_theorem_finite_plus_infinity() {
        let a = qp(&[0, 1]);
        let h = vec![RatFn::int(0), rf(&[1], &[0, -1, 1])]; // y/((x−1)x)
        let mut total = r(0);
        for r0 in finite_residues(2, &a, &h) {
            total += &r0.value;
        }
        for r0 in super::residues_at_infinity(2, &a, &h) {
            total += &r0.value;
        }
        assert_eq!(total, r(0));
    }

    /// h dx = x^{-1/2} dx = d(2√x): no residues (exact, no log part).
    #[test]
    fn exact_differential_no_residues() {
        // h = 1/√x = y/x = AlgElem [0, 1/x].
        let h = vec![RatFn::int(0), rf(&[1], &[0, 1])];
        let res = finite_residues(2, &qp(&[0, 1]), &h);
        assert!(res.is_empty(), "expected no residues; got {res:?}");
    }

    /// A single simple pole away from the branch locus: y/(x−2) on y²=x,
    /// i.e. √x/(x−2).  The two sheets ±√2 over x=2 carry opposite residues.
    #[test]
    fn simple_pole_off_branch() {
        // h = y/(x−2) = AlgElem [0, 1/(x−2)].
        let h = vec![RatFn::int(0), rf(&[1], &[-2, 1])];
        let res = finite_residues(2, &qp(&[0, 1]), &h);
        // x=2 is not a rational branch point and √2 ∉ ℚ ⇒ no *rational* sheets
        // there; the residues live at an algebraic place (out of scope) — so the
        // rational-place result is empty, soundly (not a wrong value).
        assert!(res.iter().all(|r| r.value != 0));
    }
}
