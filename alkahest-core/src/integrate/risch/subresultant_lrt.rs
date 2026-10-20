//! Lazard–Rioboo–Trager log arguments from the subresultant chain.
//!
//! For `∫ A/D dx` with `D` squarefree and `deg A < deg D`, the residues are the
//! roots of `R(t) = res_x(D, A − t·D′)`, and the log argument belonging to a
//! residue `α` of multiplicity `i` in `R` is `gcd_x(D, A − α·D′)`, of degree
//! exactly `i`.  Lazard & Rioboo (1990) showed that this gcd is the degree-`i`
//! member `S_i(t, x)` of the subresultant chain of `D` and `A − t·D′` (computed
//! once, over `ℚ[t]`), specialised at `t = α` — no arithmetic in `ℚ(α)[x]` beyond
//! reducing `S_i`'s coefficients modulo the minimal polynomial of `α`.
//!
//! The alternative, a Euclidean gcd over the number field `ℚ[t]/Q(t)`, is what
//! [`super::number_field::NumberField::kpoly_gcd`] does, and its ℚ-coefficients
//! grow through the remainder sequence: on the degree-15 field of
//! `∫ (x⁷+3x²−1)/(x¹⁵+x⁴+2x+1)² dx` it ran for minutes.  The subresultant chain
//! keeps every coefficient a polynomial in `t` whose size is bounded by
//! Hadamard's inequality (Bronstein, *Symbolic Integration I*, §1.5 and §2.5).
//!
//! Every log argument produced here is **checked** before it is used: it must
//! divide both `D` and `A − α·D′` in `ℚ(α)[x]` (one division each, by a monic
//! divisor), and its degree must be the multiplicity of the residue — which is
//! the degree of the gcd.  A candidate that fails the check is reported as
//! `None`, and the caller falls back to the Euclidean gcd, so this module can
//! only make the log part faster, never different.

use rug::Rational;

use super::number_field::{KPoly, NumberField};
use super::poly_rde::{degree, poly_mul, poly_scale, poly_zero, trim, QPoly};
use super::rational_rde::{poly_div_exact, poly_divrem, poly_sub};

/// A polynomial in `x` whose coefficients are polynomials in `t` (ascending in
/// `x`; each coefficient ascending in `t`).
pub(super) type BPoly = Vec<QPoly>;

fn bdeg(p: &BPoly) -> i64 {
    let mut d = p.len() as i64 - 1;
    while d >= 0 && trim(p[d as usize].clone()).is_empty() {
        d -= 1;
    }
    d
}

fn btrim(mut p: BPoly) -> BPoly {
    while p.last().is_some_and(|c| trim(c.clone()).is_empty()) {
        p.pop();
    }
    p
}

fn bscale(p: &BPoly, c: &QPoly) -> BPoly {
    p.iter().map(|a| trim(poly_mul(a, c))).collect()
}

fn bdiv_exact(p: &BPoly, c: &QPoly) -> BPoly {
    p.iter()
        .map(|a| {
            if trim(a.clone()).is_empty() {
                poly_zero()
            } else {
                trim(poly_div_exact(a, c))
            }
        })
        .collect()
}

fn qpow(p: &QPoly, n: i64) -> QPoly {
    let mut acc: QPoly = vec![Rational::from(1)];
    for _ in 0..n.max(0) {
        acc = poly_mul(&acc, p);
    }
    acc
}

/// Pseudo-remainder `lc(b)^{deg a − deg b + 1}·a mod b` over `ℚ[t]`.
fn bprem(a: &BPoly, b: &BPoly) -> Option<BPoly> {
    let db = bdeg(b);
    if db < 0 {
        return None;
    }
    let lcb = b[db as usize].clone();
    let mut r = btrim(a.clone());
    let mut e = bdeg(&r) - db + 1;
    if e <= 0 {
        return Some(r);
    }
    while bdeg(&r) >= db {
        crate::budget::check().ok()?;
        let dr = bdeg(&r);
        let c = r[dr as usize].clone();
        let shift = (dr - db) as usize;
        let mut next = bscale(&r, &lcb);
        for (i, bi) in b.iter().enumerate() {
            let prod = poly_mul(&c, bi);
            next[shift + i] = trim(poly_sub(&next[shift + i], &prod));
        }
        r = btrim(next);
        e -= 1;
    }
    let f = qpow(&lcb, e);
    Some(btrim(bscale(&r, &f)))
}

/// The subresultant PRS `R₀ = a, R₁ = b, R₂, …, R_k` of `a`, `b` in `ℚ[t][x]`
/// (`deg_x a ≥ deg_x b`), Bronstein's `SubResultant` (Algorithm 1.5.?, *Symbolic
/// Integration I*).  Every member is, up to a nonzero factor in `ℚ[t]` that the
/// caller's check makes irrelevant, the subresultant of its degree.
pub(super) fn subresultant_prs(a: &BPoly, b: &BPoly) -> Option<Vec<BPoly>> {
    let a = btrim(a.clone());
    let b = btrim(b.clone());
    if bdeg(&a) < bdeg(&b) || bdeg(&b) < 0 {
        return None;
    }
    let mut rs: Vec<BPoly> = vec![a, b];
    // γ₁ = −1, δ₁ = deg a − deg b, β₁ = (−1)^{δ₁+1}.
    let mut gamma: QPoly = vec![Rational::from(-1)];
    let mut delta = bdeg(&rs[0]) - bdeg(&rs[1]);
    let mut beta: QPoly = vec![Rational::from(if (delta + 1) % 2 == 0 { 1 } else { -1 })];
    loop {
        crate::budget::check().ok()?;
        let n = rs.len();
        let ri = &rs[n - 1];
        if bdeg(ri) < 0 {
            rs.pop();
            break;
        }
        let r_lc = ri[bdeg(ri) as usize].clone();
        let prem = bprem(&rs[n - 2], ri)?;
        let next = btrim(bdiv_exact(&prem, &beta));
        let deg_prev = bdeg(ri);
        let is_zero = bdeg(&next) < 0;
        rs.push(next);
        if is_zero {
            rs.pop();
            break;
        }
        // γ_{i} = (−r_{i−1})^{δ_{i−1}} · γ_{i−1}^{1−δ_{i−1}}
        let neg_r = poly_scale(&r_lc, &Rational::from(-1));
        gamma = if delta >= 1 {
            let num = qpow(&neg_r, delta);
            let den = qpow(&gamma, delta - 1);
            trim(poly_div_exact(&num, &den))
        } else {
            // δ = 0 only on the first step (equal degrees): γ·1.
            trim(poly_mul(&qpow(&neg_r, 0), &gamma))
        };
        delta = deg_prev - bdeg(rs.last().unwrap());
        beta = trim(poly_mul(&neg_r, &qpow(&gamma, delta)));
    }
    Some(rs)
}

/// The log argument `gcd_x(d, a − α·d′)` over `K = ℚ[t]/qf` taken from the
/// precomputed subresultant chain `prs` of `d` and `a − t·d′`, made monic in `x`.
///
/// `mult` is the multiplicity of `qf` in the Rothstein–Trager resultant; the gcd
/// has exactly that degree.  `d_k` and `num_k` are `d` and `a − t·d′` as
/// `K`-polynomials.  Returns `None` whenever the chain member does not verify
/// (see the module docs) — never a wrong divisor.
pub(super) fn log_argument_from_prs(
    k: &NumberField,
    prs: &[BPoly],
    mult: i64,
    d_k: &KPoly,
    num_k: &KPoly,
) -> Option<KPoly> {
    let cand: KPoly = if mult == NumberField::kdeg(d_k) {
        d_k.clone()
    } else {
        let s = prs.iter().skip(1).find(|r| bdeg(r) == mult)?;
        s.iter().map(|c| k.reduce(c)).collect()
    };
    let cand = NumberField::kpoly_trim(cand);
    if NumberField::kdeg(&cand) != mult || mult < 1 {
        return None;
    }
    let lc_inv = k.inv(&cand[mult as usize])?;
    let monic: KPoly = cand.iter().map(|c| k.mul(c, &lc_inv)).collect();
    // Check: the candidate divides both inputs.
    let (_, r1) = k.kpoly_divrem(d_k, &monic)?;
    if NumberField::kdeg(&r1) >= 0 {
        return None;
    }
    let (_, r2) = k.kpoly_divrem(num_k, &monic)?;
    if NumberField::kdeg(&r2) >= 0 {
        return None;
    }
    Some(monic)
}

/// Multiplicity of the (nonconstant) factor `f` in `p`.
pub(super) fn multiplicity(p: &QPoly, f: &QPoly) -> i64 {
    let mut p = trim(p.clone());
    let mut m = 0;
    if degree(f) < 1 {
        return 0;
    }
    loop {
        let (q, r) = poly_divrem(&p, f);
        if !trim(r).is_empty() {
            return m;
        }
        m += 1;
        p = q;
        if degree(&p) < degree(f) {
            return m;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::poly_rde::poly_deriv;
    use crate::integrate::risch::rational_rde::poly_monic;

    fn q(v: &[i64]) -> QPoly {
        v.iter().map(|&c| Rational::from(c)).collect()
    }

    fn qr(v: &[(i64, i64)]) -> QPoly {
        v.iter().map(|&c| Rational::from(c)).collect()
    }

    fn at(p: &QPoly, i: usize) -> Rational {
        p.get(i).cloned().unwrap_or_else(|| Rational::from(0))
    }

    /// `(d, a − t·d′)` as bivariate polynomials, and as `K`-polynomials.
    fn setup(k: &NumberField, a: &QPoly, d: &QPoly) -> (BPoly, BPoly, KPoly, KPoly) {
        let dp = poly_deriv(d);
        let num_b: BPoly = (0..d.len())
            .map(|i| trim(vec![at(a, i), -at(&dp, i)]))
            .collect();
        let d_b: BPoly = d.iter().map(|c| trim(vec![c.clone()])).collect();
        let num_k = NumberField::kpoly_trim(num_b.iter().map(|c| k.reduce(c)).collect());
        let d_k = NumberField::kpoly_trim(d_b.iter().map(|c| k.reduce(c)).collect());
        (d_b, num_b, d_k, num_k)
    }

    fn same_as_euclid(a: &QPoly, d: &QPoly, qf: &QPoly, mult: i64) {
        let k = NumberField::new(qf.clone());
        let (d_b, num_b, d_k, num_k) = setup(&k, a, d);
        let prs = subresultant_prs(&d_b, &num_b).expect("chain");
        let fast =
            log_argument_from_prs(&k, &prs, mult, &d_k, &num_k).expect("chain member verifies");
        let slow = k.kpoly_gcd(&d_k, &num_k).expect("euclid");
        assert_eq!(NumberField::kdeg(&fast), mult);
        assert!(
            NumberField::kpoly_eq(&fast, &slow),
            "subresultant log argument differs from the Euclidean gcd"
        );
    }

    /// The last chain member is the resultant (degree 0 in `x`); its monic
    /// form is the Rothstein–Trager polynomial.
    fn rt_poly(a: &QPoly, d: &QPoly) -> QPoly {
        let k = NumberField::new(q(&[0, 1]));
        let (d_b, num_b, _, _) = setup(&k, a, d);
        let prs = subresultant_prs(&d_b, &num_b).unwrap();
        let res = prs.last().unwrap();
        assert_eq!(bdeg(res), 0, "the last member is the resultant");
        poly_monic(&res[0])
    }

    #[test]
    fn matches_euclidean_gcd_simple_residues() {
        // ∫ 1/(x⁵ − x − 1): five distinct residues, R irreducible of degree 5.
        let d = q(&[-1, -1, 0, 0, 0, 1]);
        let a = q(&[1]);
        let r = rt_poly(&a, &d);
        assert_eq!(degree(&r), 5);
        same_as_euclid(&a, &d, &r, 1);
        // ∫ (x²+1)/(x⁷ − 3x + 1).
        let d = q(&[1, -3, 0, 0, 0, 0, 0, 1]);
        let a = q(&[1, 0, 1]);
        let r = rt_poly(&a, &d);
        same_as_euclid(&a, &d, &r, 1);
    }

    #[test]
    fn matches_euclidean_gcd_multiplicity_two() {
        // ∫ x/(x⁴+1): the residues ±i/4 each belong to two roots, R = (16t²+1)²,
        // and the log arguments are quadratics in x.
        let d = q(&[1, 0, 0, 0, 1]);
        let a = q(&[0, 1]);
        let r = rt_poly(&a, &d);
        let qf = qr(&[(1, 16), (0, 1), (1, 1)]);
        assert_eq!(multiplicity(&r, &qf), 2);
        same_as_euclid(&a, &d, &qf, 2);
    }

    #[test]
    fn a_wrong_multiplicity_is_refused_not_used() {
        let d = q(&[1, 0, 0, 0, 1]);
        let a = q(&[0, 1]);
        let k = NumberField::new(qr(&[(1, 16), (0, 1), (1, 1)]));
        let (d_b, num_b, d_k, num_k) = setup(&k, &a, &d);
        let prs = subresultant_prs(&d_b, &num_b).unwrap();
        // Degree 1 is not the gcd's degree, so no chain member may be used.
        assert!(log_argument_from_prs(&k, &prs, 1, &d_k, &num_k).is_none());
    }

    #[test]
    fn multiplicity_counts_repeated_factors() {
        let f = q(&[1, 0, 1]);
        let p = poly_mul(&poly_mul(&f, &f), &q(&[2, 1]));
        assert_eq!(multiplicity(&p, &f), 2);
        assert_eq!(multiplicity(&p, &q(&[3, 1])), 0);
    }
}
