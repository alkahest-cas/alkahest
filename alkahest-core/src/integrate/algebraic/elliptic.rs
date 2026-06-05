//! Elliptic-curve arithmetic over `ℚ` — the genus-1 engine for FIND-ORDER
//! (Risch **MC1**).
//!
//! For a genus-1 curve `y² = a(x)` (`a` a squarefree cubic/quartic), the
//! logarithmic part's residue divisor is elementary iff its class in `Pic⁰` is
//! **torsion**; by **Mazur's theorem** the torsion of `E(ℚ)` has order ≤ 12, so
//! testing `m·S = O` for `m ∈ 1..=12` is a *complete* decision.  This module
//! provides the short-Weierstrass model `y² = x³ + a·x + b` (incl. reduction of a
//! cubic via [`short_weierstrass`] or a quartic via [`weierstrass_from_quartic`]),
//! the group law over `ℚ`, the torsion-order test ([`EllipticCurve::order`]) that
//! genus-1 FIND-ORDER calls, and — once the order `m` is known — **Miller's
//! algorithm** ([`EllipticCurve::miller_function`]) that *constructs* the actual
//! log argument `u` with `div(u) = m·(P) − m·(O)` for the term `(1/m)·log(u)`.
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

/// Reduce a genus-1 quartic `y² = q(x)` (deg `q = 4`) with a **rational root**
/// `r` to a short-Weierstrass cubic, returning the curve and the birational
/// point map for places with `x ≠ r`:
///
/// ```text
///   q(x) = (x − r)·c(x),   u = 1/(x − r),   Y = y/(x − r)²,
///   Y² = C(u) := u³·c(r + 1/u)   (a cubic in u),
/// ```
/// then composed with [`short_weierstrass`] of `C`.  The place at `x = r`
/// (`(r,0)`) maps to `u = ∞` = the origin `O` and must be handled by the caller.
#[allow(clippy::type_complexity)]
pub fn weierstrass_from_quartic(
    q: &QPoly,
    r: &Rational,
) -> Option<(
    EllipticCurve,
    impl Fn(&Rational, &Rational) -> (Rational, Rational),
)> {
    let q = trim(q.clone());
    if degree(&q) != 4 {
        return None;
    }
    // c = q / (x − r)  (synthetic division; remainder must be 0).
    let mut c = vec![Rational::from(0); 4];
    c[3] = q[4].clone();
    c[2] = q[3].clone() + r.clone() * &c[3];
    c[1] = q[2].clone() + r.clone() * &c[2];
    c[0] = q[1].clone() + r.clone() * &c[1];
    if q[0].clone() + r.clone() * &c[0] != 0 {
        return None; // r is not a root of q
    }
    // C(u) = Σᵢ cᵢ·(r·u + 1)^i·u^{3−i}.
    let lin = vec![Rational::from(1), r.clone()]; // r·u + 1
    let mut pw = vec![Rational::from(1)]; // (r·u+1)^0
    let mut big_c = vec![Rational::from(0); 4];
    for (i, ci) in c.iter().enumerate() {
        // term = cᵢ · pw · u^{3−i}.
        for (j, pj) in pw.iter().enumerate() {
            let k = j + (3 - i);
            if k < big_c.len() {
                big_c[k] += ci.clone() * pj;
            }
        }
        if i < 3 {
            pw = poly_mul_small(&pw, &lin);
        }
    }
    let (e, map_c) = short_weierstrass(&big_c)?;
    let rr = r.clone();
    let map = move |x: &Rational, y: &Rational| -> (Rational, Rational) {
        let d = x.clone() - &rr;
        let u = Rational::from(1) / d.clone();
        let yy = y.clone() / (d.clone() * &d);
        map_c(&u, &yy)
    };
    Some((e, map))
}

/// A factor of an elliptic function: the **vertical** line `x − x₀`, or the
/// **chord/tangent** line `y − (λ·x + ν)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EllFactor {
    Vertical(Rational),
    Line(Rational, Rational),
}

/// A function on an elliptic curve as a quotient of line factors `∏num / ∏den`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct EllipticFunction {
    pub num: Vec<EllFactor>,
    pub den: Vec<EllFactor>,
}

impl EllipticCurve {
    /// **Miller's algorithm**: the function `f_{m,P}` with divisor
    /// `m·(P) − ([m]P) − (m−1)·(O)`.  When `P` has order `m` (so `[m]P = O`) this
    /// is exactly the **log argument** `u` with `div(u) = m·(P) − m·(O)` — the
    /// `u` in the logarithmic term `(1/m)·log(u)`.  `None` if a step degenerates
    /// unexpectedly.
    pub fn miller_function(&self, p: &Point, m: u32) -> Option<EllipticFunction> {
        if m == 0 {
            return Some(EllipticFunction::default());
        }
        let bits: Vec<bool> = (0..32).rev().map(|i| (m >> i) & 1 == 1).collect();
        let first = bits.iter().position(|&b| b)?; // MSB
        let mut f = EllipticFunction::default();
        let mut t = p.clone();
        for &bit in &bits[first + 1..] {
            // f ← f² · g_{T,T},  T ← 2T.
            f = f.squared();
            let (g, two_t) = self.double_factor(&t);
            f.compose(g);
            t = two_t;
            if bit {
                // f ← f · g_{T,P},  T ← T + P.
                let (g, tp) = self.add_factor(&t, p);
                f.compose(g);
                t = tp;
            }
        }
        f.cancel();
        Some(f)
    }

    /// **General Miller**: the function with divisor `D = Σ nₚ·(P)` for a
    /// *principal* `D` (degree 0 and `Σ nₚ·P = O`).  Built by folding the points
    /// into a running Abel–Jacobi accumulator, multiplying by the chord/vertical
    /// factor at each step (and its inverse for poles).  `None` if `D` is not
    /// principal (the accumulator does not return to `O`).
    pub fn general_miller(&self, divisor: &[(Point, i64)]) -> Option<EllipticFunction> {
        let mut f = EllipticFunction::default();
        let mut acc = Point::Infinity;
        for (p, n) in divisor {
            for _ in 0..n.unsigned_abs() {
                if *n > 0 {
                    let (g, new_acc) = self.add_factor(&acc, p);
                    f.compose(g);
                    acc = new_acc;
                } else {
                    // Incorporate −(P): (acc)−(O)−(P) ~ (acc−P)−(O) via 1/g_{acc−P,P}.
                    let am = self.add(&acc, &self.neg(p));
                    let (g, _back) = self.add_factor(&am, p);
                    f.compose_inverse(g);
                    acc = am;
                }
            }
        }
        f.cancel();
        if acc != Point::Infinity {
            return None; // divisor not principal
        }
        Some(f)
    }

    /// `g_{T,T} = ℓ_tangent(T) / v_{2T}` and `2T`.
    fn double_factor(&self, t: &Point) -> (EllipticFunction, Point) {
        let Point::Affine(x, y) = t else {
            return (EllipticFunction::default(), Point::Infinity);
        };
        if *y == 0 {
            // 2-torsion: the tangent is vertical, 2T = O (no v factor).
            return (
                EllipticFunction::num1(EllFactor::Vertical(x.clone())),
                Point::Infinity,
            );
        }
        let lambda =
            (Rational::from(3) * x.clone() * x + &self.a) / (Rational::from(2) * y.clone());
        let nu = y.clone() - lambda.clone() * x;
        let two_t = self.add(t, t);
        let mut g = EllipticFunction::num1(EllFactor::Line(lambda, nu));
        if let Point::Affine(x2, _) = &two_t {
            g.den.push(EllFactor::Vertical(x2.clone()));
        }
        (g, two_t)
    }

    /// `g_{T,P} = ℓ_{T,P} / v_{T+P}` and `T + P`.
    fn add_factor(&self, t: &Point, p: &Point) -> (EllipticFunction, Point) {
        let (Point::Affine(x1, y1), Point::Affine(x2, y2)) = (t, p) else {
            // One is O: g = 1.
            let sum = self.add(t, p);
            return (EllipticFunction::default(), sum);
        };
        if x1 == x2 {
            if (y1.clone() + y2) == 0 {
                // P = −T ⇒ T+P = O, line is vertical, no v factor.
                return (
                    EllipticFunction::num1(EllFactor::Vertical(x1.clone())),
                    Point::Infinity,
                );
            }
            return self.double_factor(t); // T = P
        }
        let lambda = (y2.clone() - y1) / (x2.clone() - x1);
        let nu = y1.clone() - lambda.clone() * x1;
        let tp = self.add(t, p);
        let mut g = EllipticFunction::num1(EllFactor::Line(lambda, nu));
        if let Point::Affine(x3, _) = &tp {
            g.den.push(EllFactor::Vertical(x3.clone()));
        }
        (g, tp)
    }
}

impl EllipticFunction {
    fn num1(f: EllFactor) -> Self {
        EllipticFunction {
            num: vec![f],
            den: Vec::new(),
        }
    }
    fn squared(&self) -> Self {
        EllipticFunction {
            num: [self.num.clone(), self.num.clone()].concat(),
            den: [self.den.clone(), self.den.clone()].concat(),
        }
    }
    fn compose(&mut self, g: EllipticFunction) {
        self.num.extend(g.num);
        self.den.extend(g.den);
    }
    fn compose_inverse(&mut self, g: EllipticFunction) {
        self.num.extend(g.den);
        self.den.extend(g.num);
    }
    /// Cancel matching factors between numerator and denominator.
    fn cancel(&mut self) {
        let mut i = 0;
        while i < self.num.len() {
            if let Some(j) = self.den.iter().position(|d| *d == self.num[i]) {
                self.num.remove(i);
                self.den.remove(j);
            } else {
                i += 1;
            }
        }
    }
}

fn poly_mul_small(a: &QPoly, b: &QPoly) -> QPoly {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![Rational::from(0); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            r[i + j] += ai.clone() * bj;
        }
    }
    r
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

    /// Quartic reduction: y² = (x²−1)(x²−4) = x⁴ − 5x² + 4, rational root r=1.
    /// The point (0,2) (2² = 4 = q(0)) maps onto the reduced cubic.
    #[test]
    fn quartic_reduction() {
        let q = vec![r(4), r(0), r(-5), r(0), r(1)];
        let (e, map) = weierstrass_from_quartic(&q, &r(1)).expect("quartic with root");
        assert!(e.is_smooth());
        let (xx, yy) = map(&r(0), &r(2));
        assert!(e.contains(&Point::Affine(xx, yy)));
        // The branch point (2,0) (a root ≠ r) maps to 2-torsion (Y=0).
        let (_, y2) = map(&r(2), &r(0));
        assert_eq!(y2, r(0));
    }

    /// Miller log-argument construction on y²=x³+1:
    /// `(−1,0)` is 2-torsion ⇒ `f_{2,P} = x + 1` (div = 2(−1,0) − 2(O));
    /// `(0,1)` has order 3 ⇒ `f_{3,P} = y − 1` (div = 3(0,1) − 3(O)).
    #[test]
    fn miller_log_arguments() {
        let e = EllipticCurve::new(r(0), r(1));
        // f_{2,(−1,0)} = (x − (−1)) = x + 1.
        let f2 = e.miller_function(&pt(-1, 0), 2).expect("miller");
        assert_eq!(f2.num, vec![EllFactor::Vertical(r(-1))]);
        assert!(f2.den.is_empty());
        // f_{3,(0,1)} = (y − (0·x + 1)) = y − 1.
        let f3 = e.miller_function(&pt(0, 1), 3).expect("miller");
        assert_eq!(f3.num, vec![EllFactor::Line(r(0), r(1))]);
        assert!(f3.den.is_empty());
    }

    /// Miller on a higher-order point: `(2,3)` has order 6 on y²=x³+1; the
    /// running point ends at `[6]P = O`, and `f_{6,P}` is a well-formed quotient.
    #[test]
    fn miller_order_six_terminates() {
        let e = EllipticCurve::new(r(0), r(1));
        assert_eq!(e.mul(6, &pt(2, 3)), Point::Infinity); // [6]P = O
        let f = e.miller_function(&pt(2, 3), 6).expect("miller");
        // After cancellation the function is nonempty (a genuine 6-torsion log
        // argument) and shares no factor between num and den.
        assert!(!f.num.is_empty());
        assert!(f.num.iter().all(|n| !f.den.contains(n)));
    }

    /// General Miller on the principal divisor `3(0,1) − 3(O)` of y²=x³+1
    /// reproduces `y − 1` (= `f_{3,(0,1)}`).
    #[test]
    fn general_miller_multipoint() {
        let e = EllipticCurve::new(r(0), r(1));
        let div = [(pt(0, 1), 3), (Point::Infinity, -3)];
        let f = e.general_miller(&div).expect("principal");
        assert_eq!(f.num, vec![EllFactor::Line(r(0), r(1))]); // y − 1
        assert!(f.den.is_empty());
        // A non-principal divisor returns None.
        assert!(e.general_miller(&[(pt(0, 1), 1), (Point::Infinity, -1)]).is_none());
    }
}
