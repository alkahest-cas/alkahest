//! Elliptic-curve arithmetic over `ℚ` — the genus-1 engine for FIND-ORDER
//! (Risch **MC1**).
//!
//! For a genus-1 curve `y² = a(x)` (`a` a squarefree cubic/quartic), the
//! logarithmic part's residue divisor is elementary iff its class in `Pic⁰` is
//! **torsion**; by **Mazur's theorem** the torsion of `E(ℚ)` has order ≤ 12, so
//! testing `m·S = O` for `m ∈ 1..=12` is a *complete* decision.  This module
//! provides the short-Weierstrass model `y² = x³ + a·x + b`, the group law over
//! `ℚ`, and the torsion-order test that genus-1 FIND-ORDER calls once the residue
//! divisor's Abel–Jacobi image `S ∈ E(ℚ)` is known.
//!
//! Everything here is exact rational arithmetic; the curve is required smooth
//! (nonzero discriminant).

use rug::Rational;

use super::super::risch::poly_rde::{degree, trim, QPoly};

/// The short-Weierstrass elliptic curve `y² = x³ + a·x + b` over `ℚ`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EllipticCurve {
    pub a: Rational,
    pub b: Rational,
}

/// A point of [`EllipticCurve`] over `ℚ`: the identity `O` or an affine `(x, y)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Point {
    Infinity,
    Affine(Rational, Rational),
}

impl EllipticCurve {
    pub fn new(a: Rational, b: Rational) -> Self {
        EllipticCurve { a, b }
    }

    /// Discriminant `−16·(4a³ + 27b²)`; the curve is smooth iff this is nonzero.
    pub fn discriminant(&self) -> Rational {
        let a3 = self.a.clone() * &self.a * &self.a;
        let b2 = self.b.clone() * &self.b;
        Rational::from(-16) * (Rational::from(4) * a3 + Rational::from(27) * b2)
    }

    pub fn is_smooth(&self) -> bool {
        self.discriminant() != 0
    }

    /// Is `P` on the curve?
    pub fn contains(&self, p: &Point) -> bool {
        match p {
            Point::Infinity => true,
            Point::Affine(x, y) => {
                let lhs = y.clone() * y;
                let rhs = x.clone() * x * x + self.a.clone() * x + &self.b;
                lhs == rhs
            }
        }
    }

    /// `−P`.
    pub fn neg(&self, p: &Point) -> Point {
        match p {
            Point::Infinity => Point::Infinity,
            Point::Affine(x, y) => Point::Affine(x.clone(), -y.clone()),
        }
    }

    /// The group law `P + Q`.
    pub fn add(&self, p: &Point, q: &Point) -> Point {
        match (p, q) {
            (Point::Infinity, _) => q.clone(),
            (_, Point::Infinity) => p.clone(),
            (Point::Affine(x1, y1), Point::Affine(x2, y2)) => {
                if x1 == x2 && (y1.clone() + y2) == 0 {
                    return Point::Infinity; // P = −Q
                }
                let lambda = if x1 == x2 {
                    // Doubling: λ = (3x₁² + a) / (2y₁).
                    let num = Rational::from(3) * x1.clone() * x1 + &self.a;
                    let den = Rational::from(2) * y1.clone();
                    num / den
                } else {
                    (y2.clone() - y1) / (x2.clone() - x1)
                };
                let x3 = lambda.clone() * &lambda - x1 - x2;
                let y3 = lambda * (x1.clone() - &x3) - y1;
                Point::Affine(x3, y3)
            }
        }
    }

    /// `m·P` for `m ≥ 0` (double-and-add).
    pub fn mul(&self, mut m: u64, p: &Point) -> Point {
        let mut acc = Point::Infinity;
        let mut base = p.clone();
        while m > 0 {
            if m & 1 == 1 {
                acc = self.add(&acc, &base);
            }
            base = self.add(&base, &base);
            m >>= 1;
        }
        acc
    }

    /// The order of `P` in `E(ℚ)`, or `None` if `P` has **infinite** order.
    /// Sound by Mazur: a rational torsion point has order in `{1,…,10,12}`, so
    /// `m·P ≠ O` for all `m ≤ 12` proves infinite order.
    pub fn order(&self, p: &Point) -> Option<u32> {
        let mut cur = p.clone();
        for m in 1..=12u32 {
            if cur == Point::Infinity {
                return Some(m);
            }
            cur = self.add(&cur, p);
        }
        None
    }
}

/// Reduce a smooth cubic `c(x) = c₃x³ + c₂x² + c₁x + c₀` (`c₃ ≠ 0`) to a short
/// Weierstrass curve `y² = x³ + A·x + B`, returning the curve and the forward
/// point map `(x, y) ↦ (X, Y)` from `y² = c(x)`:
///
/// ```text
///   X = c₃·x + c₂/3,   Y = c₃·y.
/// ```
#[allow(clippy::type_complexity)] // (curve, forward point map) — a closure return
pub fn short_weierstrass(
    c: &QPoly,
) -> Option<(
    EllipticCurve,
    impl Fn(&Rational, &Rational) -> (Rational, Rational),
)> {
    let c = trim(c.clone());
    if degree(&c) != 3 {
        return None;
    }
    let c0 = c[0].clone();
    let c1 = c[1].clone();
    let c2 = c[2].clone();
    let c3 = c[3].clone();
    if c3 == 0 {
        return None;
    }
    // After X = c₃x, Y = c₃y:  Y² = X³ + c₂X² + (c₃c₁)X + c₃²c₀.
    let b2 = c2.clone();
    let b1 = c3.clone() * &c1;
    let b0 = c3.clone() * &c3 * &c0;
    // Depress X = t − b₂/3:  y² = t³ + p·t + q.
    let p = b1.clone() - b2.clone() * &b2 / Rational::from(3);
    let q = b0 - b1 * &b2 / Rational::from(3)
        + Rational::from(2) * b2.clone() * &b2 * &b2 / Rational::from(27);
    let curve = EllipticCurve::new(p, q);
    if !curve.is_smooth() {
        return None;
    }
    let c3m = c3.clone();
    let c2m = c2;
    let map = move |x: &Rational, y: &Rational| -> (Rational, Rational) {
        let big_x = c3m.clone() * x + c2m.clone() / Rational::from(3);
        let big_y = c3m.clone() * y;
        (big_x, big_y)
    };
    Some((curve, map))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> Rational {
        Rational::from(n)
    }
    fn pt(x: i64, y: i64) -> Point {
        Point::Affine(r(x), r(y))
    }

    /// y² = x³ + 1 has rational torsion ℤ/6: (2,3) order 6, (0,1) order 3,
    /// (−1,0) order 2, O order 1.
    #[test]
    fn torsion_z6() {
        let e = EllipticCurve::new(r(0), r(1));
        assert!(e.is_smooth());
        assert!(e.contains(&pt(2, 3)) && e.contains(&pt(0, 1)) && e.contains(&pt(-1, 0)));
        assert_eq!(e.order(&Point::Infinity), Some(1));
        assert_eq!(e.order(&pt(-1, 0)), Some(2));
        assert_eq!(e.order(&pt(0, 1)), Some(3));
        assert_eq!(e.order(&pt(2, 3)), Some(6));
        // 6·(2,3) = O.
        assert_eq!(e.mul(6, &pt(2, 3)), Point::Infinity);
    }

    /// y² = x³ − x has full 2-torsion ℤ/2×ℤ/2: (0,0),(1,0),(−1,0) order 2.
    #[test]
    fn full_two_torsion() {
        let e = EllipticCurve::new(r(-1), r(0));
        for p in [pt(0, 0), pt(1, 0), pt(-1, 0)] {
            assert!(e.contains(&p));
            assert_eq!(e.order(&p), Some(2));
        }
        // (0,0)+(1,0) = (−1,0).
        assert_eq!(e.add(&pt(0, 0), &pt(1, 0)), pt(-1, 0));
    }

    /// Infinite-order point: (3,5) on the Mordell curve y² = x³ − 2 (rank 1) —
    /// `order` returns `None` (no `m·P = O` for `m ≤ 12`).
    #[test]
    fn infinite_order() {
        let e = EllipticCurve::new(r(0), r(-2));
        assert!(e.contains(&pt(3, 5))); // 27 − 2 = 25 = 5²
        assert_eq!(e.order(&pt(3, 5)), None);
    }

    /// Group law sanity: P + (−P) = O, and P + O = P.
    #[test]
    fn group_axioms() {
        let e = EllipticCurve::new(r(-1), r(0));
        let p = pt(0, 0);
        assert_eq!(e.add(&p, &e.neg(&p)), Point::Infinity);
        assert_eq!(e.add(&p, &Point::Infinity), p);
    }

    /// short_weierstrass maps points of y²=c(x) onto the reduced curve.
    #[test]
    fn weierstrass_reduction() {
        // c(x) = x³ + 1 (already short): map is identity-ish (c₃=1, c₂=0).
        let c = vec![r(1), r(0), r(0), r(1)];
        let (e, map) = short_weierstrass(&c).expect("cubic");
        assert_eq!(e, EllipticCurve::new(r(0), r(1)));
        let (xx, yy) = map(&r(2), &r(3));
        assert!(e.contains(&Point::Affine(xx, yy)));

        // Non-monic / shifted cubic: 2x³ + 3x² + 1, check a point maps onto E.
        let c2 = vec![r(1), r(0), r(3), r(2)];
        let (e2, map2) = short_weierstrass(&c2).expect("cubic");
        assert!(e2.is_smooth());
        // x=0 ⇒ y²=1 ⇒ (0,1) on y²=c2(x); its image lies on E2.
        let (xx, yy) = map2(&r(0), &r(1));
        assert!(e2.contains(&Point::Affine(xx, yy)));
    }
}
