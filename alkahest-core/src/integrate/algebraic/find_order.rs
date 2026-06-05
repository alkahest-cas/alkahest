//! FIND-ORDER — genus-graded principality of the residue divisor (Risch **MC**).
//!
//! The logarithmic part of `∫ R(x,y) dx` is `Σ cⱼ log(uⱼ)`, and it exists (the
//! integral is elementary) iff the **residue divisor** `δ = Σ res_P · P`
//! ([`super::residues`]) is, after scaling residues to integers, a **torsion**
//! element of the divisor class group — i.e. some `N·δ` is **principal**, with
//! `div(uⱼ) = N·δⱼ` and the log term `(1/N) log(uⱼ)`.  FIND-ORDER decides that
//! `N` (the *order* of the class), and is genus-graded:
//!
//! * **genus 0** — every degree-0 divisor is principal, so `N = 1` always
//!   (the rational-parametrization win, seen from the divisor side).
//! * **genus 1** — torsion over ℚ has order ≤ 12 (**Mazur**), so `N ∈ 1..=12` is
//!   a *complete* test; the genuine elliptic frontier (Weierstrass form + point
//!   arithmetic) — currently `NotDecided`.
//! * **genus ≥ 2** — no uniform bound; the Weil-bound reduction-mod-good-prime
//!   search is best-effort — currently `NotDecided`.
//!
//! This module provides the **genus** computation (superelliptic curves) and the
//! genus-graded decision with **genus 0 complete** and the necessary conditions
//! (degree-0 residue divisor; rational residues).  Sound: only `Principal`/
//! `NonElementary` verdicts are asserted; everything uncertain is `NotDecided`.

use super::super::risch::poly_rde::{degree, poly_deriv, trim, QPoly};
use super::super::risch::rational_rde::poly_gcd;
use super::residues::{residue_sum, Residue};

/// Result of FIND-ORDER on a residue divisor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FindOrder {
    /// `N·δ` is principal with this minimal `N` (the log part is `(1/N)·Σ log`).
    Principal { order: u32 },
    /// The integral is provably non-elementary (a necessary condition failed).
    NonElementary,
    /// Undecided within the implemented scope (e.g. genus ≥ 1 torsion, or an
    /// incomplete divisor with missing algebraic places).
    NotDecided,
}

/// Geometric genus of the **superelliptic** curve `yⁿ = a(x)` for **squarefree**
/// `a` of degree `m`:
///
/// ```text
///   g = ( m·n − m − n + 2 − gcd(n, m) ) / 2.
/// ```
///
/// `None` when `a` is not squarefree (the singular/normal-form case is out of
/// this bounded scope).
pub fn genus(n: usize, a: &QPoly) -> Option<usize> {
    let a = trim(a.clone());
    let m = degree(&a);
    if m < 1 || n < 2 {
        return None;
    }
    // Squarefree ⟺ gcd(a, a') is constant.
    if degree(&poly_gcd(&a, &poly_deriv(&a))) > 0 {
        return None;
    }
    let (m, n) = (m as i64, n as i64);
    let g2 = m * n - m - n + 2 - gcd_i64(n, m);
    if g2 < 0 || g2 % 2 != 0 {
        return None;
    }
    Some((g2 / 2) as usize)
}

/// Decide the order of the residue divisor's class — the FIND-ORDER step.
///
/// `divisor` is the residue divisor from [`super::residues::residue_divisor`].
pub fn find_order(n: usize, a: &QPoly, divisor: &[Residue]) -> FindOrder {
    // Necessary: a complete residue divisor has degree 0 (Σ res = 0).  A nonzero
    // sum means places are missing (algebraic) — we cannot conclude.
    if residue_sum(divisor) != 0 {
        return FindOrder::NotDecided;
    }
    // No residues ⇒ no logarithmic part: the empty divisor is principal (N = 1).
    if divisor.iter().all(|r| r.value == 0) {
        return FindOrder::Principal { order: 1 };
    }
    match genus(n, a) {
        // Genus 0: every degree-0 divisor is principal.
        Some(0) => FindOrder::Principal { order: 1 },
        // Genus ≥ 1: the torsion decision (Mazur for genus 1, Weil for ≥ 2) is
        // the elliptic/Jacobian frontier — not yet implemented.
        Some(_) => FindOrder::NotDecided,
        None => FindOrder::NotDecided,
    }
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::super::super::risch::alg_field::RatFn;
    use super::super::residues::residue_divisor;
    use super::*;
    use rug::Rational;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    #[test]
    fn genus_values() {
        assert_eq!(genus(2, &qp(&[0, 1])), Some(0)); // y²=x         (rational)
        assert_eq!(genus(2, &qp(&[1, 0, 0, 1])), Some(1)); // y²=x³+1 (elliptic)
        assert_eq!(genus(2, &qp(&[0, 1, 0, 0, 1])), Some(1)); // y²=x⁴+x (genus 1)
        assert_eq!(genus(2, &qp(&[1, 0, 0, 0, 0, 1])), Some(2)); // y²=x⁵+1 (genus 2)
        assert_eq!(genus(3, &qp(&[0, 1])), Some(0)); // y³=x         (rational)
        assert_eq!(genus(3, &qp(&[1, 0, 1])), Some(1)); // y³=x²+1   (genus 1)
                                                        // Non-squarefree radicand ⇒ unknown (out of scope).
        assert_eq!(genus(2, &qp(&[0, 0, 1])), None); // y²=x²
    }

    /// Genus-0 curve `y²=x`: the residue divisor of `∫1/((x−1)√x) dx` (residues
    /// ±1 at x=1, res_∞=0, sum 0) is principal with order 1.
    #[test]
    fn genus0_principal_order_one() {
        let a = qp(&[0, 1]);
        let h = vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, -1, 1]))]; // y/((x−1)x)
        let div = residue_divisor(2, &a, &h);
        assert_eq!(find_order(2, &a, &div), FindOrder::Principal { order: 1 });
    }

    /// Exact differential (no residues) ⇒ principal order 1.
    #[test]
    fn empty_divisor_principal() {
        let a = qp(&[0, 1]);
        let h = vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[0, 1]))]; // y/x = x^{-1/2}
        let div = residue_divisor(2, &a, &h);
        assert!(matches!(
            find_order(2, &a, &div),
            FindOrder::Principal { order: 1 }
        ));
    }

    /// Genus-1 curve `y²=x³+1`: a nonzero residue divisor is the elliptic
    /// frontier — `NotDecided` (sound, not a wrong verdict).
    #[test]
    fn genus1_not_decided() {
        let a = qp(&[1, 0, 0, 1]); // x³+1
                                   // h = 1/((x−2)y): a simple pole at x=2 (and conjugate sheets).
        let h = vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[-2, 1]))];
        let div = residue_divisor(2, &a, &h);
        // Either the divisor is incomplete (algebraic places) or genus ≥ 1:
        // both yield NotDecided.
        assert_eq!(find_order(2, &a, &div), FindOrder::NotDecided);
    }
}
