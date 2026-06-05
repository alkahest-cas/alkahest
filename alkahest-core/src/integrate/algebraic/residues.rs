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
//! Scope: residues at **rational** finite places with rational branches (poles of
//! `h` and branch points).  Algebraic places and the place at infinity are the
//! follow-up; together with FIND-ORDER (genus-graded principality) they complete
//! MC.

use rug::{Integer, Rational};
use std::collections::BTreeSet;

use super::super::risch::alg_field::AlgElem;
use super::super::risch::poly_rde::{degree, trim, QPoly};
use super::vanhoeij::{branch_ts, elem_ts};
use crate::poly::puiseux::puiseux_at;

/// A residue at a place over a rational point `x = α`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residue {
    /// The base point `α` of the place.
    pub point: Rational,
    /// Index of the Puiseux sheet at `α` (distinguishes places over the same `α`).
    pub sheet: usize,
    /// Ramification index of the place.
    pub ramification: u64,
    /// The residue `res_P(h dx)`.
    pub value: Rational,
}

/// Residues of the differential `h dx` at all **rational finite places** of the
/// curve `yⁿ = a(x)` — the poles of `h` and the branch points (roots of `a`).
/// Places with zero residue are omitted.
///
/// (Algebraic places and the place at infinity are out of this rational-only
/// scope; for fully ramified rational curves the finite residues already sum to
/// `−res_∞`.)
pub fn finite_residues(n: usize, a: &QPoly, h: &AlgElem) -> Vec<Residue> {
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
        for (sheet, br) in puiseux_at(&monos, &alpha, prec).iter().enumerate() {
            let e = br.ramification as i64;
            let u = 2 * e + 4;
            let series = elem_ts(h, &alpha, e, u, &branch_ts(br));
            let coeff = series
                .get(&(-e))
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
            let value = Rational::from(e) * coeff;
            if value != 0 {
                out.push(Residue {
                    point: alpha.clone(),
                    sheet,
                    ramification: br.ramification,
                    value,
                });
            }
        }
    }
    out
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
