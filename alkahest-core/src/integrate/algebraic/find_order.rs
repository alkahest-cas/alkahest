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
//! * **genus 1** (`y²=cubic`, or `y²=quartic` with a rational point) — **implemented**:
//!   map the residue divisor to `E(ℚ)` (Abel–Jacobi, [`super::elliptic`]) and read
//!   off the order of its class; by **Mazur** a rational torsion point has order
//!   ≤ 12, so this is a *complete* decision — `Principal{N}` if torsion,
//!   `NonElementary` if not.
//! * **genus ≥ 2** — no uniform bound; the Weil-bound reduction-mod-good-prime
//!   search is best-effort — currently `NotDecided`.
//!
//! Sound: only `Principal`/`NonElementary` verdicts are asserted; anything
//! uncertain (incomplete divisor with missing algebraic places, a quartic with
//! no usable rational point, genus ≥ 2) is `NotDecided`.

use rug::{Integer, Rational};

use super::super::risch::poly_rde::{degree, poly_deriv, trim, QPoly};
use super::super::risch::rational_rde::poly_gcd;
use super::elliptic::{
    quartic_point_model, short_weierstrass, weierstrass_from_quartic, EllipticCurve, Point,
};
use super::residues::{residue_sum, PlacedResidue, Residue};

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
/// This entry decides **genus 0** (and the degree-0 / empty-divisor cases); the
/// **genus-1** Abel–Jacobi torsion decision needs each place's `y`-coordinate,
/// which the public [`Residue`] does not carry, so it is reached through the
/// internal `find_order_placed` (used by the genus-1 integrator).  Genus ≥ 1
/// here therefore returns [`FindOrder::NotDecided`].
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
        // Genus ≥ 1: the torsion decision needs `y`-coordinates — see
        // [`find_order_placed`].
        _ => FindOrder::NotDecided,
    }
}

/// FIND-ORDER on a divisor whose places carry `y`-coordinates ([`PlacedResidue`]),
/// enabling the **genus-1** Abel–Jacobi torsion decision (Mazur ⇒ complete for
/// `y² = cubic`/`quartic`).  Internal entry for the genus-1 integrator.
pub(crate) fn find_order_placed(n: usize, a: &QPoly, divisor: &[PlacedResidue]) -> FindOrder {
    if divisor
        .iter()
        .fold(Rational::from(0), |s, r| s + &r.residue.value)
        != 0
    {
        return FindOrder::NotDecided;
    }
    if divisor.iter().all(|r| r.residue.value == 0) {
        return FindOrder::Principal { order: 1 };
    }
    match genus(n, a) {
        Some(0) => FindOrder::Principal { order: 1 },
        Some(1) => genus1(n, a, divisor).unwrap_or(FindOrder::NotDecided),
        _ => FindOrder::NotDecided,
    }
}

/// Genus-1 FIND-ORDER for `y² = a(x)` (cubic or quartic): map the residue divisor
/// to `E(ℚ)` (Abel–Jacobi), then read off the order of its class.  `None` when
/// outside scope (degree ≠ 3,4; a quartic with a nonzero residue at infinity, a
/// residue on the base-point fibre, or no rational point found; or a place not on
/// `E(ℚ)`).
fn genus1(n: usize, a: &QPoly, divisor: &[PlacedResidue]) -> Option<FindOrder> {
    if n != 2 {
        return None;
    }
    let a = trim(a.clone());
    // Build the curve `E` and a place-mapper `φ`: place ↦ Some(point on E), or
    // None when the place maps to the origin O (and drops out of the sum).
    #[allow(clippy::type_complexity)]
    let (e, mapper): (EllipticCurve, Box<dyn Fn(&PlacedResidue) -> Option<Point>>) =
        match degree(&a) {
            3 => {
                let (e, map) = short_weierstrass(&a)?;
                // Cubic: ∞ is the origin O; finite places map directly.
                let f = move |r: &PlacedResidue| -> Option<Point> {
                    if r.residue.at_infinity {
                        None
                    } else {
                        let (x, y) = map(&r.residue.point, &r.y_coord);
                        Some(Point::Affine(x, y))
                    }
                };
                (e, Box::new(f))
            }
            4 => {
                // Quartic: handled only with no residue at infinity.
                if divisor
                    .iter()
                    .any(|r| r.residue.at_infinity && r.residue.value != 0)
                {
                    return None;
                }
                if let Some(root) = first_rational_root(&a) {
                    // Rational root: the place at x=root maps to O.
                    let (e, map) = weierstrass_from_quartic(&a, &root)?;
                    let f = move |r: &PlacedResidue| -> Option<Point> {
                        if r.residue.at_infinity || r.residue.point == root {
                            None
                        } else {
                            let (x, y) = map(&r.residue.point, &r.y_coord);
                            Some(Point::Affine(x, y))
                        }
                    };
                    (e, Box::new(f))
                } else {
                    // No rational root: reduce via a finite rational point (Nagell).
                    // Residues on the base-point fibre `x=x₀` aren't placeable here
                    // (one sheet ↦ O, the other ↦ a finite point) — bail for safety.
                    let (x0, y0) = first_rational_point(&a)?;
                    if divisor
                        .iter()
                        .any(|r| !r.residue.at_infinity && r.residue.point == x0)
                    {
                        return None;
                    }
                    let m = quartic_point_model(&a, &x0, &y0)?;
                    let (e, _) = short_weierstrass(&m.c)?;
                    let c3 = m.c[3].clone();
                    let c2 = m.c.get(2).cloned().unwrap_or_else(|| Rational::from(0));
                    let f = move |r: &PlacedResidue| -> Option<Point> {
                        if r.residue.at_infinity {
                            return None; // infinity residues are zero here
                        }
                        let (z, w) = m.zw(&r.residue.point, &r.y_coord)?;
                        let big_x = c3.clone() * &z + c2.clone() / Rational::from(3);
                        let big_y = c3.clone() * &w;
                        Some(Point::Affine(big_x, big_y))
                    };
                    (e, Box::new(f))
                }
            }
            _ => return None,
        };

    // Scale residues to a primitive integer divisor: coeffs = value·L / g.
    let mut l = Integer::from(1);
    for r in divisor {
        l = l.lcm(r.residue.value.denom());
    }
    let int_coeffs: Vec<Integer> = divisor
        .iter()
        .map(|r| {
            (r.residue.value.clone() * Rational::from(l.clone()))
                .numer()
                .clone()
        })
        .collect();
    let mut g = Integer::from(0);
    for c in &int_coeffs {
        g = g.gcd(c);
    }
    if g == 0 {
        return Some(FindOrder::Principal { order: 1 }); // all residues zero
    }

    // S = Σ (coeffₚ/g) · φ(P)  in E(ℚ).
    let mut s = Point::Infinity;
    for (r, c) in divisor.iter().zip(&int_coeffs) {
        let Some(p) = mapper(r) else {
            continue; // place maps to O
        };
        if !e.contains(&p) {
            return None; // place not on E(ℚ): outside the rational-image scope
        }
        let coeff = (c.clone() / &g).to_i64()?;
        let term = if coeff >= 0 {
            e.mul(coeff as u64, &p)
        } else {
            e.mul((-coeff) as u64, &e.neg(&p))
        };
        s = e.add(&s, &term);
    }

    Some(match e.order(&s) {
        // Class order N: N·δ is principal ⇒ the log part is (1/N)·log(u).
        Some(order) => FindOrder::Principal { order },
        // Non-torsion class ⇒ no multiple is principal ⇒ no elementary log part.
        None => FindOrder::NonElementary,
    })
}

/// A finite **rational point** `(x₀, y₀)` on `y² = q(x)` with `y₀ ≠ 0`, searched
/// over small rationals.  Used to reduce a rational-root-free quartic to
/// Weierstrass form (Nagell).  Best-effort: `None` if none is found in range.
pub(super) fn first_rational_point(q: &QPoly) -> Option<(Rational, Rational)> {
    let q = trim(q.clone());
    if degree(&q) != 4 {
        return None;
    }
    let evalq =
        |x: &Rational| -> Rational { q.iter().rev().fold(Rational::from(0), |acc, c| acc * x + c) };
    let sqrt_rat = |v: &Rational| -> Option<Rational> {
        if *v < 0 {
            return None;
        }
        let n = v.numer().clone();
        let d = v.denom().clone();
        let ns = n.clone().sqrt();
        let ds = d.clone().sqrt();
        if Integer::from(&ns * &ns) == n && Integer::from(&ds * &ds) == d {
            Some(Rational::from((ns, ds)))
        } else {
            None
        }
    };
    for dn in 1..=4i64 {
        for nn in -8..=8i64 {
            let x0 = Rational::from((nn, dn));
            let v = evalq(&x0);
            if v == 0 {
                continue; // a root → handled by first_rational_root (want y₀ ≠ 0)
            }
            if let Some(y0) = sqrt_rat(&v) {
                return Some((x0, y0));
            }
        }
    }
    None
}

/// A rational root of `p ∈ ℚ[x]` (rational-root theorem), if any.
pub(super) fn first_rational_root(p: &QPoly) -> Option<Rational> {
    let p = trim(p.clone());
    if degree(&p) < 1 {
        return None;
    }
    if p[0] == 0 {
        return Some(Rational::from(0));
    }
    let mut den_lcm = Integer::from(1);
    for c in &p {
        den_lcm = den_lcm.lcm(c.denom());
    }
    let ints: Vec<Integer> = p
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
                let mut acc = Rational::from(0);
                for c in ints.iter().rev() {
                    acc = acc * &cand + Rational::from(c.clone());
                }
                if acc == 0 {
                    return Some(cand);
                }
            }
        }
    }
    None
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

    fn place(point: i64, y: i64, value: i64, inf: bool, ram: u64) -> PlacedResidue {
        PlacedResidue {
            residue: Residue {
                point: Rational::from(point),
                at_infinity: inf,
                sheet: 0,
                ramification: ram,
                value: Rational::from(value),
            },
            y_coord: Rational::from(y),
        }
    }

    /// Genus-1 `y²=x³+1` (torsion ℤ/6).  Residue divisor `(−1,0) − O`: the
    /// Abel–Jacobi image is the 2-torsion point `(−1,0)` ⇒ `Principal{2}`.
    #[test]
    fn genus1_order_two() {
        let a = qp(&[1, 0, 0, 1]);
        let div = [place(-1, 0, 1, false, 2), place(0, 0, -1, true, 1)];
        assert_eq!(
            find_order_placed(2, &a, &div),
            FindOrder::Principal { order: 2 }
        );
    }

    /// Same curve, divisor `(2,3) − O`: `(2,3)` has order 6 ⇒ `Principal{6}`.
    #[test]
    fn genus1_order_six() {
        let a = qp(&[1, 0, 0, 1]);
        let div = [place(2, 3, 1, false, 1), place(0, 0, -1, true, 1)];
        assert_eq!(
            find_order_placed(2, &a, &div),
            FindOrder::Principal { order: 6 }
        );
    }

    /// Mordell curve `y²=x³−2` (rank 1).  Divisor `(3,5) − O` maps to the
    /// infinite-order point `(3,5)` ⇒ the integral is non-elementary.
    #[test]
    fn genus1_non_elementary() {
        let a = qp(&[-2, 0, 0, 1]);
        assert_eq!(genus(2, &a), Some(1));
        let div = [place(3, 5, 1, false, 1), place(0, 0, -1, true, 1)];
        assert_eq!(find_order_placed(2, &a, &div), FindOrder::NonElementary);
    }

    /// An incomplete divisor (Σ res ≠ 0, missing algebraic places) ⇒ NotDecided.
    #[test]
    fn genus1_incomplete_not_decided() {
        let a = qp(&[1, 0, 0, 1]);
        let div = [place(-1, 0, 1, false, 2)]; // sum = 1 ≠ 0
        assert_eq!(find_order_placed(2, &a, &div), FindOrder::NotDecided);
    }

    /// Genus-1 **quartic** `y²=(x²−1)(x²−4)=x⁴−5x²+4` (rational roots ±1,±2).
    /// Divisor `(1,0) − (2,0)` of two branch points (each 2-torsion) ⇒ the class
    /// is 2-torsion ⇒ `Principal{2}`.
    #[test]
    fn genus1_quartic_order_two() {
        let a = qp(&[4, 0, -5, 0, 1]);
        assert_eq!(genus(2, &a), Some(1));
        let div = [place(1, 0, 1, false, 2), place(2, 0, -1, false, 2)];
        assert_eq!(
            find_order_placed(2, &a, &div),
            FindOrder::Principal { order: 2 }
        );
    }
}
