//! Genus-1 quartics `y² = s²·x⁴ + …` with **residues at the places over ∞**.
//!
//! When the leading coefficient of a quartic radicand is a rational square
//! `s²`, the two places over `x = ∞` (`y ≈ ±s·x²`) are rational points of the
//! curve, and one of them can serve as the origin of an elliptic model.  This
//! is the case the other genus-1 models ([`super::find_order`],
//! [`super::genus1_log`]) cannot place: they send *both* places over ∞ to `O`,
//! so a residue divisor carrying `∞₊ − ∞₋` — Chebyshev's
//! `∫ x dx/√(x⁴+10x²−96x−71)` (Trager 1984, Bronstein §10) is the classic —
//! had no image on `E` and FIND-ORDER returned `NotDecided`.
//!
//! # The model
//!
//! Write `a = s²·(x⁴ + b x³ + c x² + d x + e)`, `ỹ = y/s`, `Q = x² + (b/2)x`,
//! `P = c − b²/4`, so `ỹ² = Q² + P x² + d x + e`.  With
//!
//! ```text
//!   T = ỹ + Q,      V = 2(2T + P)·x + b·T + d
//! ```
//!
//! one checks `T² = (2T+P)x² + (bT+d)x + e`, and multiplying by `4(2T+P)`
//!
//! ```text
//!   V² = C(T) = 8T³ + (4P + b²)T² + (2bd − 8e)T + (d² − 4Pe).
//! ```
//!
//! The map `(x, ỹ) ↦ (T, V)` is birational (`x = (V − bT − d)/(2(2T+P))`,
//! `ỹ = T − Q`), hence an isomorphism of the smooth projective models.  `T` has
//! its only pole — a double one — at `∞₊` (`ỹ ≈ +x²`), so `∞₊ ↦ O`; along `∞₋`
//! (`ỹ = −Q − P/2 − γ/x − …`, `γ = (d − bP/2)/2`) one gets
//! `∞₋ ↦ (T, V) = (−P/2, −(d − bP/2))`.  [`short_weierstrass`] then takes `C`
//! to `E: Y² = X³ + AX + B` with `X = 8T + c₂/3`, `Y = 8V`.
//!
//! Everything downstream is the existing genus-1 machinery: the Abel–Jacobi
//! sum on `E(ℚ)`, its order (Mazur: ≤ 12, so a complete decision), Miller's
//! function with the scaled divisor, pulled back to `ℚ(x)[y]/(y² − a)` as an
//! exact [`AlgElem`].  The emitted antiderivative passes the same
//! `d/dx F = integrand` gate as [`super::genus1_log`].

use rug::{Integer, Rational};

use super::super::risch::alg_field::{AlgElem, AlgExtension, RatFn};
use super::super::risch::poly_rde::{degree, rational_to_expr, trim, QPoly};
use super::elliptic::{short_weierstrass, EllFactor, EllipticCurve, Point};
use super::find_order::FindOrder;
use super::residues::PlacedResidue;
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// The `(T, V)` model of a quartic with square leading coefficient.
#[derive(Clone, Debug)]
pub(super) struct QuarticInfinityModel {
    /// `s > 0` with `s² = lc(a)`.
    s: Rational,
    b: Rational,
    p: Rational,
    d: Rational,
    /// Leading two coefficients of `C(T)` — the short-Weierstrass shift.
    c3: Rational,
    c2: Rational,
    curve: EllipticCurve,
}

fn rat_sqrt(r: &Rational) -> Option<Rational> {
    if *r <= 0 {
        return None;
    }
    let (n, d) = (r.numer().clone(), r.denom().clone());
    let (ns, ds) = (n.clone().sqrt(), d.clone().sqrt());
    if Integer::from(&ns * &ns) == n && Integer::from(&ds * &ds) == d {
        Some(Rational::from((ns, ds)))
    } else {
        None
    }
}

impl QuarticInfinityModel {
    /// The model for `y² = a`, when `deg a = 4` and `lc(a)` is a positive
    /// rational square (and the resulting cubic is smooth, i.e. `a` squarefree).
    pub(super) fn new(a: &QPoly) -> Option<Self> {
        let a = trim(a.clone());
        if degree(&a) != 4 {
            return None;
        }
        let s = rat_sqrt(&a[4])?;
        let lc = a[4].clone();
        let b = a[3].clone() / &lc;
        let c = a[2].clone() / &lc;
        let d = a[1].clone() / &lc;
        let e = a[0].clone() / &lc;
        let p = c - b.clone() * &b / Rational::from(4);
        let cubic: QPoly = vec![
            d.clone() * &d - Rational::from(4) * p.clone() * &e,
            Rational::from(2) * b.clone() * &d - Rational::from(8) * e.clone(),
            Rational::from(4) * p.clone() + b.clone() * &b,
            Rational::from(8),
        ];
        let (curve, _) = short_weierstrass(&cubic)?;
        Some(Self {
            s,
            b,
            p,
            d,
            c3: cubic[3].clone(),
            c2: cubic[2].clone(),
            curve,
        })
    }

    pub(super) fn curve(&self) -> &EllipticCurve {
        &self.curve
    }

    /// `(T, V) ↦ (X, Y)` on `E`.
    fn to_e(&self, t: Rational, v: Rational) -> Point {
        Point::Affine(
            self.c3.clone() * t + self.c2.clone() / Rational::from(3),
            self.c3.clone() * v,
        )
    }

    /// The image on `E` of a place of the residue divisor, or `None` when the
    /// place cannot be identified (a place over ∞ with no sheet label, or a
    /// ramified place over ∞ — impossible for an even-degree radicand).
    pub(super) fn place_point(&self, r: &PlacedResidue) -> Option<Point> {
        if r.residue.at_infinity {
            if r.residue.ramification != 1 {
                return None;
            }
            if r.y_coord == self.s {
                Some(Point::Infinity) // ∞₊ ↦ O
            } else if r.y_coord == -self.s.clone() {
                let t = -self.p.clone() / Rational::from(2);
                let v = -(self.d.clone() - self.b.clone() * &self.p / Rational::from(2));
                Some(self.to_e(t, v))
            } else {
                None // unlabelled sheet: cannot tell ∞₊ from ∞₋
            }
        } else {
            let x0 = &r.residue.point;
            let t = r.y_coord.clone() / &self.s
                + x0.clone() * x0
                + self.b.clone() * x0 / Rational::from(2);
            let v = Rational::from(2) * (Rational::from(2) * t.clone() + &self.p) * x0
                + self.b.clone() * &t
                + &self.d;
            Some(self.to_e(t, v))
        }
    }

    /// `T` and `V` as elements `A(x) + B(x)·y` of `ℚ(x)[y]/(y² − a)`.
    fn t_v(&self) -> (AlgElem, AlgElem) {
        let half_b = self.b.clone() / Rational::from(2);
        let inv_s = Rational::from(1) / self.s.clone();
        // T = x² + (b/2)x + y/s
        let q: QPoly = vec![Rational::from(0), half_b.clone(), Rational::from(1)];
        let t = vec![RatFn::from_poly(&q), RatFn::from_poly(&vec![inv_s.clone()])];
        // V = 2(2T+P)x + bT + d
        //   = [4x³ + 2b x² + 2P x + b x² + (b²/2) x + d] + [(4x + b)/s]·y
        let v0: QPoly = vec![
            self.d.clone(),
            Rational::from(2) * self.p.clone() + self.b.clone() * &half_b,
            Rational::from(3) * self.b.clone(),
            Rational::from(4),
        ];
        let v1: QPoly = vec![self.b.clone() * &inv_s, Rational::from(4) * inv_s];
        let v = vec![RatFn::from_poly(&v0), RatFn::from_poly(&v1)];
        (t, v)
    }

    /// Pull a line factor on `E` back to `ℚ(x)[y]/(y² − a)`.
    fn factor_to_alg(&self, ext: &AlgExtension, f: &EllFactor) -> AlgElem {
        let (t, v) = self.t_v();
        let c3 = RatFn::from_poly(&vec![self.c3.clone()]);
        let x_e = ext.add(
            &ext.mul(&ext.constant(c3.clone()), &t),
            &ext.constant(RatFn::from_poly(&vec![self.c2.clone() / Rational::from(3)])),
        );
        match f {
            EllFactor::Vertical(x0) => {
                ext.sub(&x_e, &ext.constant(RatFn::from_poly(&vec![x0.clone()])))
            }
            EllFactor::Line(lam, nu) => {
                let y_e = ext.mul(&ext.constant(c3), &v);
                let lx = ext.mul(&ext.constant(RatFn::from_poly(&vec![lam.clone()])), &x_e);
                ext.sub(
                    &ext.sub(&y_e, &lx),
                    &ext.constant(RatFn::from_poly(&vec![nu.clone()])),
                )
            }
        }
    }
}

/// Does the divisor carry a nonzero residue at a place over ∞?
pub(super) fn has_infinite_residue(divisor: &[PlacedResidue]) -> bool {
    divisor
        .iter()
        .any(|r| r.residue.at_infinity && r.residue.value != 0)
}

/// Scale the residues to integers (`·L`, `L` = lcm of denominators) and map
/// each place to `E`.  `None` if a place is unplaceable or off `E(ℚ)`.
fn scaled_points(
    m: &QuarticInfinityModel,
    divisor: &[PlacedResidue],
) -> Option<(Integer, Vec<(Point, i64)>)> {
    // An incomplete divisor (missing algebraic places) has Σ res ≠ 0; its
    // Abel–Jacobi sum means nothing.
    if divisor
        .iter()
        .fold(Rational::from(0), |acc, r| acc + &r.residue.value)
        != 0
    {
        return None;
    }
    let mut l = Integer::from(1);
    for r in divisor {
        l = l.lcm(r.residue.value.denom());
    }
    let mut pairs = Vec::new();
    for r in divisor {
        if r.residue.value == 0 {
            continue;
        }
        let p = m.place_point(r)?;
        if !m.curve.contains(&p) {
            return None;
        }
        let c = (r.residue.value.clone() * Rational::from(l.clone()))
            .numer()
            .to_i64()?;
        pairs.push((p, c));
    }
    Some((l, pairs))
}

fn abel_jacobi(e: &EllipticCurve, pairs: &[(Point, i64)]) -> Point {
    let mut s = Point::Infinity;
    for (p, c) in pairs {
        let t = if *c >= 0 {
            e.mul(*c as u64, p)
        } else {
            e.mul(c.unsigned_abs(), &e.neg(p))
        };
        s = e.add(&s, &t);
    }
    s
}

/// FIND-ORDER on `y² = a` (quartic, square leading coefficient) for a residue
/// divisor whose places over ∞ carry sheet labels.  `None` when outside scope.
pub(super) fn find_order_infinity(a: &QPoly, divisor: &[PlacedResidue]) -> Option<FindOrder> {
    let m = QuarticInfinityModel::new(a)?;
    let (_, pairs) = scaled_points(&m, divisor)?;
    // Use the primitive integer divisor (divide by the content) so the order
    // is that of the class of δ itself.
    let mut g = Integer::from(0);
    for (_, c) in &pairs {
        g = g.gcd(&Integer::from(*c));
    }
    if g == 0 {
        return Some(FindOrder::Principal { order: 1 });
    }
    let g = g.to_i64()?;
    let prim: Vec<(Point, i64)> = pairs.into_iter().map(|(p, c)| (p, c / g)).collect();
    let s = abel_jacobi(m.curve(), &prim);
    Some(match m.curve().order(&s) {
        Some(order) => FindOrder::Principal { order },
        None => FindOrder::NonElementary,
    })
}

/// The log part `(1/(N·L))·log(u)` for a torsion residue divisor on the
/// `(T, V)` model, `u` returned as an exact `A(x) + B(x)·y`.  Returns
/// `(u, N·L)`, or `None` if the class is not torsion or a step degenerates.
pub(super) fn log_argument(a: &QPoly, divisor: &[PlacedResidue]) -> Option<(AlgElem, Integer)> {
    let m = QuarticInfinityModel::new(a)?;
    let (l, pairs) = scaled_points(&m, divisor)?;
    let s = abel_jacobi(m.curve(), &pairs);
    let n = m.curve().order(&s)?;
    let scaled: Vec<(Point, i64)> = pairs
        .iter()
        .map(|(p, c)| Some((p.clone(), c.checked_mul(i64::from(n))?)))
        .collect::<Option<_>>()?;
    let u_e = m.curve().general_miller(&scaled)?;
    let ext = AlgExtension::radical(2, &trim(a.clone()));
    let mut u = ext.from_int(1);
    for f in &u_e.num {
        u = ext.mul(&u, &m.factor_to_alg(&ext, f));
    }
    for f in &u_e.den {
        u = ext.mul(&u, &ext.inv(&m.factor_to_alg(&ext, f))?);
    }
    // Normalise the content: divide by a rational constant so the leading
    // coefficient of the rational part is ±1-ish (a constant factor only
    // shifts the log by a constant).
    let u = normalise(&ext, u);
    Some((u, l * Integer::from(n)))
}

/// Divide `u` by the leading coefficient of its first nonzero coordinate's
/// numerator — a rational constant, harmless inside `log`.
fn normalise(ext: &AlgExtension, u: AlgElem) -> AlgElem {
    let Some(lead) = u
        .iter()
        .find(|c| !c.numer().is_empty())
        .and_then(|c| c.numer().last().cloned())
    else {
        return u;
    };
    if lead == 0 {
        return u;
    }
    let inv = RatFn::from_poly(&vec![Rational::from(1) / lead]);
    ext.mul(&u, &ext.constant(inv))
}

/// `∫ integrand dx = g + (1/(N·L))·log(u)` on the `(T, V)` model; `None` when
/// the log argument cannot be built.  The caller verifies.
pub(super) fn antiderivative(
    a: &QPoly,
    g_expr: ExprId,
    divisor: &[PlacedResidue],
    y_sym: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let (u, nl) = log_argument(a, divisor)?;
    let u_expr = super::genus1_log::alg_to_expr(&u, y_sym, var, pool);
    let log_u = pool.func("log", vec![u_expr]);
    let coeff = rational_to_expr(&Rational::from((Integer::from(1), nl)), pool);
    Some(simplify(pool.add(vec![g_expr, pool.mul(vec![coeff, log_u])]), pool).value)
}

#[cfg(test)]
mod tests {
    use super::super::residues::Residue;
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    fn inf(sign: i64, value: i64) -> PlacedResidue {
        PlacedResidue {
            residue: Residue {
                point: Rational::from(0),
                at_infinity: true,
                sheet: if sign > 0 { 0 } else { 1 },
                ramification: 1,
                value: Rational::from(value),
            },
            y_coord: Rational::from(sign),
        }
    }

    /// The model is consistent: finite rational points of the quartic land on
    /// `E`, and so does `∞₋`.
    #[test]
    fn model_maps_points_onto_curve() {
        // y² = x⁴ + 10x² − 96x − 71 has the rational point (−1, ±6).
        let a = qp(&[-71, -96, 10, 0, 1]);
        let m = QuarticInfinityModel::new(&a).expect("square lc");
        for y in [6, -6] {
            let r = PlacedResidue {
                residue: Residue {
                    point: Rational::from(-1),
                    at_infinity: false,
                    sheet: 0,
                    ramification: 1,
                    value: Rational::from(1),
                },
                y_coord: Rational::from(y),
            };
            let p = m.place_point(&r).unwrap();
            assert!(m.curve().contains(&p), "{p:?}");
        }
        let pm = m.place_point(&inf(-1, 1)).unwrap();
        assert!(m.curve().contains(&pm));
        assert_eq!(m.place_point(&inf(1, 1)), Some(Point::Infinity));
    }

    /// Chebyshev's example: `∞₊ − ∞₋` is torsion of order 8 on
    /// `y² = x⁴ + 10x² − 96x − 71`.
    #[test]
    fn chebyshev_divisor_is_torsion() {
        let a = qp(&[-71, -96, 10, 0, 1]);
        let div = [inf(1, -1), inf(-1, 1)];
        assert!(matches!(
            find_order_infinity(&a, &div),
            Some(FindOrder::Principal { order: 8 })
        ));
    }

    /// Perturbing the constant term destroys the torsion:
    /// `∫ x dx/√(x⁴+10x²−96x−72)` is the classic non-elementary twin.
    #[test]
    fn perturbed_chebyshev_is_not_torsion() {
        let a = qp(&[-72, -96, 10, 0, 1]);
        let div = [inf(1, -1), inf(-1, 1)];
        assert_eq!(
            find_order_infinity(&a, &div),
            Some(FindOrder::NonElementary)
        );
    }

    /// An unlabelled place over ∞ is refused, not guessed.
    #[test]
    fn unlabelled_infinity_refused() {
        let a = qp(&[-71, -96, 10, 0, 1]);
        let mut p = inf(1, -1);
        p.y_coord = Rational::from(0);
        let div = [p, inf(-1, 1)];
        assert_eq!(find_order_infinity(&a, &div), None);
    }

    /// End to end through the engine: `∫ x dx/√(x⁴+10x²−96x−71)` is
    /// `⅛·log(u)` with `u = A(x) + B(x)·√a`, and differentiates back exactly
    /// (sampled on both real components of `a > 0`).
    #[test]
    fn chebyshev_integral_end_to_end() {
        use crate::kernel::{Domain, ExprPool};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = std::collections::HashMap::from([("x".to_owned(), x)]);
        let f = crate::parse::parse("x/sqrt(x^4+10*x^2-96*x-71)", &pool, &mut syms).unwrap();
        let big_f = crate::integrate::engine::integrate(f, x, &pool)
            .expect("elementary")
            .value;
        let df = crate::diff::diff(big_f, x, &pool).unwrap().value;
        for xv in [5.0_f64, 5.5, 9.0, -2.5, -6.0] {
            let d = crate::integrate::gate::eval_at(df, x, xv, &pool).unwrap();
            let want = crate::integrate::gate::eval_at(f, x, xv, &pool).unwrap();
            assert!(
                (d - want).abs() < 1e-9 * (1.0 + want.abs()),
                "x = {xv}: {d} vs {want}"
            );
        }
    }
}
