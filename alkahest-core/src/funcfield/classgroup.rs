//! The divisor class group `Pic⁰` of an imaginary hyperelliptic function field.
//!
//! This is a thin, typed layer over the Cantor arithmetic that
//! [`crate::integrate::algebraic::jacobian_torsion`] already implements for
//! Risch integration — the same `HypQ`/`MumQ` (exact, over ℚ) and
//! `HypFp`/`Mumford` (over `F_p`) code, reached through `pub(crate)`
//! visibility rather than reimplemented.
//!
//! # Representation
//!
//! A class is stored as a **reduced Mumford pair** `(u, v)` on the monicised
//! model `Y² = F(X)`, reached from `y² = a(x)` by `X = lc·x`, `Y = lc^g·y` —
//! the same transform `jacobian_torsion` and `coates` use, so the
//! representations agree bit for bit.  `u` is monic with `deg u ≤ g`,
//! `deg v < deg u`, and `v² ≡ F (mod u)`; the pair stands for
//!
//! ```text
//!     Σ_{roots X₀ of u} (X₀, v(X₀))  −  (deg u)·∞.
//! ```
//!
//! On the imaginary model the reduced representative is **unique**, so class
//! equality is a comparison of `(u, v)` and needs no search.
//!
//! # Torsion order
//!
//! [`DivisorClass::order`] runs the classical reduction-modulo-good-primes
//! argument (Bronstein 1990, Prop. 1.16–1.17) that `jacobian_torsion`
//! documents: reduce the class at several good primes, take the order in each
//! finite Jacobian, reconstruct the candidate `N` from prime-to-`p`
//! injectivity, and then **confirm it exactly over ℚ**.  The three outcomes are
//! kept apart on purpose:
//!
//! * a divisor `d | N` with `d·δ = 0` over ℚ — the order, certified exactly;
//! * `N·δ ≠ 0` over ℚ, or orders that disagree across good primes —
//!   [`FunctionFieldError::NonTorsion`], a **verdict**;
//! * too few good primes, or `N` past the work cap —
//!   [`FunctionFieldError::UndecidedOrder`], which is neither.

use rug::{Integer, Rational};

use super::divisor::{Divisor, Place};
use super::error::FunctionFieldError;
use super::field::FunctionField;
use super::util::{eval, rational_roots};
use crate::integrate::algebraic::jacobian_torsion::{
    fp_deg, fp_deriv, fp_gcd, fp_mul, fp_rem, fp_sub, fp_trim, rat_to_fp,
    reconstruct_candidate_order, FpPoly, HypFp, HypQ, MumQ, Mumford,
};
use crate::integrate::risch::poly_rde::{degree, trim, QPoly};

/// Largest prime tried when reducing a class to a finite field.
const MAX_PRIME: u64 = 200;
/// How many good primes to gather before reconstructing the candidate order.
const WANT_PRIMES: usize = 4;
/// Work cap on the candidate order confirmed exactly over ℚ.  Beyond this the
/// exact Cantor multiples grow faster than they are worth, and the answer is
/// reported undecided rather than guessed.
const MAX_EXACT_ORDER: u64 = 10_000;

/// `r^e` for a rational base and a possibly negative integer exponent.
fn pow_rat(r: &Rational, e: i64) -> Rational {
    let mut acc = Rational::from(1);
    for _ in 0..e.abs() {
        acc *= r;
    }
    if e >= 0 {
        acc
    } else {
        Rational::from(1) / acc
    }
}

/// The monicised model `Y² = F(X)` of `y² = a(x)`, plus the scalings
/// `X = lc·x`, `Y = lc^g·y`.
pub(crate) struct MonicModel {
    pub(crate) f: QPoly,
    pub(crate) g: usize,
    pub(crate) lc: Rational,
    pub(crate) lc_g: Rational,
}

impl MonicModel {
    pub(crate) fn of(field: &FunctionField) -> Self {
        let a = field.curve();
        let d = field.curve_degree();
        let lc = a[d].clone();
        let g = field.genus();
        let mut f = vec![Rational::from(0); d + 1];
        for (k, slot) in f.iter_mut().enumerate() {
            let e = d as i64 - 1 - k as i64;
            *slot = a[k].clone() * pow_rat(&lc, e);
        }
        MonicModel {
            f: trim(f),
            g,
            lc_g: pow_rat(&lc, g as i64),
            lc,
        }
    }

    fn curve(&self) -> HypQ {
        HypQ {
            f: self.f.clone(),
            g: self.g,
        }
    }
}

/// A class in `Pic⁰(K)`, as a reduced Mumford pair on the monicised model.
#[derive(Clone, Debug)]
pub struct DivisorClass {
    field: FunctionField,
    u: QPoly,
    v: QPoly,
}

impl DivisorClass {
    /// The identity class.
    pub fn identity(field: FunctionField) -> Result<Self, FunctionFieldError> {
        field.require_imaginary("DivisorClass")?;
        Ok(DivisorClass {
            field,
            u: vec![Rational::from(1)],
            v: Vec::new(),
        })
    }

    /// The class of a **degree-zero** divisor.
    ///
    /// `D = Σ cᵢPᵢ + c∞·∞` of degree zero is `Σ cᵢ(Pᵢ − ∞)`, which is exactly
    /// what Cantor's composition accumulates.  A divisor of non-zero degree has
    /// no class in `Pic⁰` and is refused ([`FunctionFieldError::NotDegreeZero`]).
    pub fn of(divisor: &Divisor) -> Result<Self, FunctionFieldError> {
        let field = divisor.field().clone();
        field.require_imaginary("DivisorClass")?;
        let deg = divisor.degree();
        if deg != 0 {
            return Err(FunctionFieldError::NotDegreeZero {
                degree: deg.to_string(),
            });
        }
        let model = MonicModel::of(&field);
        let curve = model.curve();
        let mut acc = curve.identity();
        for (x, y, c) in divisor.finite_terms() {
            let big_x = model.lc.clone() * &x;
            let big_y = model.lc_g.clone() * &y;
            if Rational::from(&big_y * &big_y) != eval(&model.f, &big_x) {
                return Err(FunctionFieldError::SelfCheckFailed {
                    detail: format!(
                        "({x}, {y}) failed the on-curve check after monicisation; the \
                         normalised model and the place disagree"
                    ),
                });
            }
            let k = c
                .clone()
                .abs()
                .to_u64()
                .ok_or_else(|| FunctionFieldError::TooLarge {
                    reason: format!("multiplicity {c} at ({x}, {y}) exceeds a machine word"),
                })?;
            if k == 0 {
                continue;
            }
            let cls = curve.point_class(&big_x, &big_y);
            let term = curve.mul(k, &cls);
            let term = if c < 0 { curve.neg(&term) } else { term };
            acc = curve.add(&acc, &term);
        }
        Ok(DivisorClass {
            field,
            u: acc.u,
            v: acc.v,
        })
    }

    /// The field this class lives on.
    pub fn field(&self) -> &FunctionField {
        &self.field
    }

    /// The Mumford `u(X)` of the reduced representative, on the monicised model.
    pub fn mumford_u(&self) -> &QPoly {
        &self.u
    }

    /// The Mumford `v(X)` of the reduced representative, on the monicised model.
    pub fn mumford_v(&self) -> &QPoly {
        &self.v
    }

    /// `deg u`, the weight of the reduced representative.  Between `0` and `g`.
    pub fn weight(&self) -> usize {
        degree(&self.u).max(0) as usize
    }

    /// `true` for the identity class — equivalently, the divisor is principal.
    pub fn is_identity(&self) -> bool {
        degree(&self.u) == 0
    }

    fn mum(&self) -> MumQ {
        MumQ {
            u: self.u.clone(),
            v: self.v.clone(),
        }
    }

    fn wrap(&self, m: MumQ) -> DivisorClass {
        DivisorClass {
            field: self.field.clone(),
            u: m.u,
            v: m.v,
        }
    }

    /// Class addition, by Cantor composition followed by reduction.
    pub fn add(&self, other: &DivisorClass) -> Result<DivisorClass, FunctionFieldError> {
        self.field.require_same(&other.field)?;
        let curve = MonicModel::of(&self.field).curve();
        Ok(self.wrap(curve.add(&self.mum(), &other.mum())))
    }

    /// The inverse class, `(u, −v mod u)` — the image under the hyperelliptic
    /// involution.
    pub fn neg(&self) -> DivisorClass {
        let curve = MonicModel::of(&self.field).curve();
        self.wrap(curve.neg(&self.mum()))
    }

    /// `k·[D]` for any integer `k`.
    pub fn scalar_mul(&self, k: &Integer) -> Result<DivisorClass, FunctionFieldError> {
        let mag = k
            .clone()
            .abs()
            .to_u64()
            .ok_or_else(|| FunctionFieldError::TooLarge {
                reason: format!("the multiplier {k} exceeds a machine word"),
            })?;
        let curve = MonicModel::of(&self.field).curve();
        let out = curve.mul(mag, &self.mum());
        let out = if *k < 0 { curve.neg(&out) } else { out };
        Ok(self.wrap(out))
    }

    /// The reduced representative as an explicit divisor,
    /// `Σ (X₀, v(X₀)) − (deg u)·∞` pulled back to the caller's coordinates.
    ///
    /// Refuses with [`FunctionFieldError::NonRationalSupport`] when `u` does not
    /// split over ℚ: the representative is then supported at a place of degree
    /// ≥ 2, which [`Place`] cannot hold.  The class itself is still perfectly
    /// well defined — [`DivisorClass::mumford_u`] always works.
    pub fn reduced_divisor(&self) -> Result<Divisor, FunctionFieldError> {
        let model = MonicModel::of(&self.field);
        let roots =
            rational_roots(&self.u).ok_or_else(|| FunctionFieldError::NonRationalSupport {
                context: format!(
                    "the reduced representative's Mumford u(X) (degree {}) does not split over ℚ",
                    self.weight()
                ),
            })?;
        let mut terms: Vec<(Place, Integer)> = Vec::new();
        let mut total = 0i64;
        for (big_x, mult) in roots {
            let big_y = eval(&self.v, &big_x);
            let x = big_x.clone() / model.lc.clone();
            let y = big_y / model.lc_g.clone();
            terms.push((Place::finite(x, y), Integer::from(mult as u64)));
            total += mult as i64;
        }
        terms.push((Place::Infinity, Integer::from(-total)));
        Divisor::from_terms(self.field.clone(), terms)
    }

    /// The order of the class in `Pic⁰`.
    ///
    /// See the module documentation.  `Ok(n)` is an exact certificate;
    /// [`FunctionFieldError::NonTorsion`] is a verdict of infinite order;
    /// [`FunctionFieldError::UndecidedOrder`] is neither.
    pub fn order(&self) -> Result<u64, FunctionFieldError> {
        if self.is_identity() {
            return Ok(1);
        }
        let model = MonicModel::of(&self.field);
        let mut data: Vec<(u64, u64)> = Vec::new();
        let mut p = 3u64;
        while p <= MAX_PRIME && data.len() < WANT_PRIMES {
            if crate::modular::is_prime(p) {
                if let Some(m) = self.order_mod(&model, p) {
                    data.push((p, m));
                }
            }
            p += 2;
        }
        if data.len() < 2 {
            return Err(FunctionFieldError::UndecidedOrder {
                reason: format!(
                    "only {} good prime(s) below {MAX_PRIME} reduced this class faithfully; at \
                     least two are needed to pin the prime-to-p part of the order",
                    data.len()
                ),
            });
        }
        // `None` here is a prime-to-ℓ inconsistency, which is a theorem-backed
        // non-torsion certificate.
        let Some(n) = reconstruct_candidate_order(&data) else {
            return Err(FunctionFieldError::NonTorsion);
        };
        if n == 0 || n > MAX_EXACT_ORDER {
            return Err(FunctionFieldError::UndecidedOrder {
                reason: format!(
                    "the reconstructed candidate order {n} is past the exact-confirmation cap \
                     of {MAX_EXACT_ORDER}"
                ),
            });
        }
        let curve = model.curve();
        let me = self.mum();
        if !HypQ::is_identity(&curve.mul(n, &me)) {
            // N is forced if the class is torsion at all, so this refutes it.
            return Err(FunctionFieldError::NonTorsion);
        }
        // The true order is the least divisor of N that kills the class.
        let mut best = n;
        for d in divisors_of(n) {
            if d < best && HypQ::is_identity(&curve.mul(d, &me)) {
                best = d;
            }
        }
        Ok(best)
    }

    /// `true` when the class is torsion.  Distinguishes a verdict from a
    /// refusal by propagating the latter.
    pub fn is_torsion(&self) -> Result<bool, FunctionFieldError> {
        match self.order() {
            Ok(_) => Ok(true),
            Err(FunctionFieldError::NonTorsion) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Order of the reduction of this class in `Jac(F_p)`, or `None` when `p`
    /// is a bad prime for this data (the caller then tries the next one).
    ///
    /// Bad means any of: `p = 2`; a coefficient of `F`, `u` or `v` whose
    /// denominator vanishes mod `p`; `F` not staying monic of full degree, or
    /// losing squarefreeness (bad reduction of the curve); `u` dropping degree
    /// or failing to stay monic (the class not reducing faithfully); or
    /// `v² ≢ F (mod u)` after reduction.
    fn order_mod(&self, model: &MonicModel, p: u64) -> Option<u64> {
        if p == 2 {
            return None;
        }
        let d = degree(&model.f) as usize;
        let f = reduce_poly(&model.f, p)?;
        if f.len() != d + 1 || f[d] != 1 {
            return None;
        }
        if fp_deg(&fp_gcd(&f, &fp_deriv(&f, p), p)) != Some(0) {
            return None; // bad reduction of the curve
        }
        let du = degree(&self.u);
        let u = reduce_poly(&self.u, p)?;
        if u.len() as i64 != du + 1 {
            return None; // the class did not reduce faithfully
        }
        if u[du.max(0) as usize] != 1 {
            return None;
        }
        let v = reduce_poly(&self.v, p)?;
        if fp_deg(&v).map(|dv| dv as i64 >= du).unwrap_or(false) {
            return None;
        }
        // v² ≡ F (mod u) must survive reduction, else the pair is not a class.
        let residue = fp_rem(&fp_sub(&fp_mul(&v, &v, p), &f, p), &u, p);
        if fp_deg(&residue).is_some() {
            return None;
        }
        let curve = HypFp { p, f, g: model.g };
        curve.order(&Mumford { u, v })
    }
}

impl PartialEq for DivisorClass {
    /// Reduced Mumford representatives are **unique** on the imaginary model,
    /// so this is exact class equality, not a heuristic.
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field && self.u == other.u && self.v == other.v
    }
}

impl Eq for DivisorClass {}

impl std::fmt::Display for DivisorClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_identity() {
            return write!(f, "[0]");
        }
        write!(f, "[u = {:?}, v = {:?}]", self.u, self.v)
    }
}

/// Reduce a `QPoly` mod `p`; `None` if a denominator vanishes.
fn reduce_poly(p_q: &QPoly, p: u64) -> Option<FpPoly> {
    let mut out = Vec::with_capacity(p_q.len());
    for c in p_q.iter() {
        out.push(rat_to_fp(c, p)?);
    }
    Some(fp_trim(out))
}

/// All divisors of `n`, unsorted.
fn divisors_of(n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut i = 1u64;
    while i * i <= n {
        if n % i == 0 {
            out.push(i);
            if i != n / i {
                out.push(n / i);
            }
        }
        i += 1;
    }
    out
}

impl FunctionField {
    /// `true` when `D` is a principal divisor, i.e. `D = div(u)` for some `u`.
    ///
    /// Exact: the class is computed by Cantor arithmetic over ℚ and compared to
    /// the identity.  A divisor of non-zero degree is never principal, and is
    /// reported as [`FunctionFieldError::NotDegreeZero`] rather than `false`,
    /// because it is a malformed question rather than a negative answer.
    pub fn is_principal(&self, divisor: &Divisor) -> Result<bool, FunctionFieldError> {
        self.require_same(divisor.field())?;
        Ok(DivisorClass::of(divisor)?.is_identity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::funcfield::element::FunctionFieldElement;

    fn e1() -> FunctionField {
        // y² = x³ − x, g = 1.  Torsion: (0,0), (±1,0) are the 2-torsion points.
        FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap()
    }

    fn c2() -> FunctionField {
        // y² = x⁵ + 1, g = 2.
        FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 0, 1]).unwrap()
    }

    fn pl(x: i64, y: i64) -> Place {
        Place::finite(Rational::from(x), Rational::from(y))
    }

    fn deg0(f: &FunctionField, terms: &[(Place, i64)]) -> Divisor {
        let total: i64 = terms.iter().map(|(_, c)| *c).sum();
        let mut t: Vec<(Place, Integer)> = terms
            .iter()
            .map(|(p, c)| (p.clone(), Integer::from(*c)))
            .collect();
        t.push((Place::Infinity, Integer::from(-total)));
        Divisor::from_terms(f.clone(), t).unwrap()
    }

    #[test]
    fn the_zero_divisor_is_the_identity_class() {
        let f = e1();
        let z = Divisor::zero(f.clone()).unwrap();
        let c = DivisorClass::of(&z).unwrap();
        assert!(c.is_identity());
        assert_eq!(c, DivisorClass::identity(f).unwrap());
    }

    #[test]
    fn a_class_plus_its_inverse_is_the_identity() {
        for f in [e1(), c2()] {
            let d = deg0(&f, &[(pl(0, if f.genus() == 1 { 0 } else { 1 }), 1)]);
            let c = DivisorClass::of(&d).unwrap();
            let sum = c.add(&c.neg()).unwrap();
            assert!(sum.is_identity(), "c + (−c) must be 0 on {f}");
        }
    }

    #[test]
    fn two_torsion_on_the_elliptic_curve() {
        // On y² = x³ − x the branch points (0,0), (1,0), (−1,0) give classes of
        // order exactly 2: 2·((α,0) − ∞) = div(x − α).
        let f = e1();
        for a in [-1i64, 0, 1] {
            let d = deg0(&f, &[(pl(a, 0), 1)]);
            let c = DivisorClass::of(&d).unwrap();
            assert!(!c.is_identity(), "(({a},0) − ∞) is not principal");
            assert!(
                c.scalar_mul(&Integer::from(2)).unwrap().is_identity(),
                "2·(({a},0) − ∞) must be principal"
            );
            assert_eq!(c.order().unwrap(), 2);
        }
    }

    #[test]
    fn two_torsion_is_certified_by_an_explicit_function() {
        // The principal divisor is div(x − α), and div() reproduces it.
        let f = e1();
        let d = deg0(&f, &[(pl(0, 0), 2)]);
        assert!(f.is_principal(&d).unwrap());
        let u = FunctionFieldElement::x(f.clone());
        assert_eq!(u.divisor().unwrap(), d);
    }

    #[test]
    fn the_sum_of_the_three_two_torsion_classes_is_principal() {
        // div(y) = (−1,0) + (0,0) + (1,0) − 3∞ on y² = x³ − x.
        let f = e1();
        let d = deg0(&f, &[(pl(-1, 0), 1), (pl(0, 0), 1), (pl(1, 0), 1)]);
        assert!(f.is_principal(&d).unwrap());
        assert_eq!(FunctionFieldElement::y(f).divisor().unwrap(), d);
    }

    #[test]
    fn n_delta_is_principal_exactly_when_the_order_divides_n() {
        let f = e1();
        let d = deg0(&f, &[(pl(0, 0), 1)]);
        let c = DivisorClass::of(&d).unwrap();
        let order = c.order().unwrap();
        assert_eq!(order, 2);
        for n in 1i64..=8 {
            let scaled = d.scale(&Integer::from(n));
            let principal = f.is_principal(&scaled).unwrap();
            assert_eq!(
                principal,
                n % order as i64 == 0,
                "{n}·δ principal should hold iff {order} | {n}"
            );
        }
    }

    #[test]
    fn torsion_orders_on_y2_equals_x3_plus_1() {
        // E(ℚ)_tors ≅ ℤ/6 for y² = x³ + 1, and the orders are checkable by
        // hand from the duplication formula:
        //   P = (0, 1):  λ = 3x²/2y = 0, so 2P = (0, −1) = −P  ⇒  order 3.
        //   Q = (2, 3):  order 6 (Q + P has order 2).
        //   T = (−1, 0): a branch point  ⇒  order 2.
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 1]).unwrap();
        let cases = [((0i64, 1i64), 3u64), ((2, 3), 6), ((-1, 0), 2)];
        for ((x, y), want) in cases {
            let c = DivisorClass::of(&deg0(&f, &[(pl(x, y), 1)])).unwrap();
            assert_eq!(c.order().unwrap(), want, "order of (({x},{y}) − ∞)");
            // Minimality: no smaller multiple kills the class.
            for k in 1..want {
                assert!(
                    !c.scalar_mul(&Integer::from(k as i64))
                        .unwrap()
                        .is_identity(),
                    "({x},{y}): {k} already kills a class claimed to have order {want}"
                );
            }
            assert!(c
                .scalar_mul(&Integer::from(want as i64))
                .unwrap()
                .is_identity());
        }
    }

    #[test]
    fn a_non_torsion_class_is_a_verdict_not_a_refusal() {
        // y² = x³ − 2 has rank 1 with generator (3, 5); that class is of
        // infinite order.
        let f = FunctionField::hyperelliptic_from_i64(&[-2, 0, 0, 1]).unwrap();
        assert!(f.contains_point(&Rational::from(3), &Rational::from(5)));
        let d = deg0(&f, &[(pl(3, 5), 1)]);
        let c = DivisorClass::of(&d).unwrap();
        let err = c.order().unwrap_err();
        assert_eq!(err.code(), "E-FFLD-007");
        assert!(!c.is_torsion().unwrap());
    }

    #[test]
    fn class_addition_agrees_with_divisor_addition() {
        let f = e1();
        let d1 = deg0(&f, &[(pl(0, 0), 1)]);
        let d2 = deg0(&f, &[(pl(1, 0), 1)]);
        let sum = d1.add(&d2).unwrap();
        assert_eq!(
            DivisorClass::of(&d1)
                .unwrap()
                .add(&DivisorClass::of(&d2).unwrap())
                .unwrap(),
            DivisorClass::of(&sum).unwrap()
        );
    }

    #[test]
    fn reduction_is_idempotent_and_preserves_the_class() {
        let f = e1();
        let d = deg0(&f, &[(pl(0, 0), 3), (pl(1, 0), 5)]);
        let c = DivisorClass::of(&d).unwrap();
        let rep = c.reduced_divisor().unwrap();
        assert_eq!(rep.degree(), Integer::from(0));
        let c2 = DivisorClass::of(&rep).unwrap();
        assert_eq!(
            c, c2,
            "re-reducing a reduced representative must not move it"
        );
        assert_eq!(c2.reduced_divisor().unwrap(), rep);
    }

    #[test]
    fn the_reduced_representative_has_weight_at_most_g() {
        for f in [e1(), c2()] {
            let g = f.genus();
            let pt = if g == 1 { pl(0, 0) } else { pl(0, 1) };
            let d = deg0(&f, &[(pt, 7)]);
            let c = DivisorClass::of(&d).unwrap();
            assert!(c.weight() <= g, "weight {} exceeds g = {g}", c.weight());
        }
    }

    #[test]
    fn a_non_zero_degree_divisor_has_no_class() {
        let f = e1();
        let d = Divisor::from_terms(f, [(pl(0, 0), Integer::from(1))]).unwrap();
        let err = DivisorClass::of(&d).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-005");
    }

    #[test]
    fn classes_on_different_curves_do_not_mix() {
        let a = DivisorClass::identity(e1()).unwrap();
        let b = DivisorClass::identity(c2()).unwrap();
        assert_eq!(a.add(&b).unwrap_err().code(), "E-FFLD-009");
        assert_ne!(a, b);
    }

    #[test]
    fn scalar_mul_is_a_homomorphism() {
        let f = e1();
        let d = deg0(&f, &[(pl(1, 0), 1)]);
        let c = DivisorClass::of(&d).unwrap();
        for k in [-3i64, -1, 0, 1, 2, 5] {
            let by_scalar = c.scalar_mul(&Integer::from(k)).unwrap();
            let by_divisor = DivisorClass::of(&d.scale(&Integer::from(k))).unwrap();
            assert_eq!(by_scalar, by_divisor, "k = {k}");
        }
    }

    #[test]
    fn divisors_of_small_numbers() {
        let mut d = divisors_of(12);
        d.sort_unstable();
        assert_eq!(d, vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(divisors_of(1), vec![1]);
    }

    #[test]
    fn a_genus_two_torsion_class() {
        // y² = x⁵ + 1: the class of (0,1) − ∞.  (0,1) is a point of finite
        // order on this Jacobian; the test pins whatever the exact machinery
        // certifies and checks it is genuinely minimal.
        let f = c2();
        let d = deg0(&f, &[(pl(0, 1), 1)]);
        let c = DivisorClass::of(&d).unwrap();
        let n = c.order().unwrap();
        assert!(n > 1);
        assert!(c
            .scalar_mul(&Integer::from(n as i64))
            .unwrap()
            .is_identity());
        for k in 1..n {
            assert!(
                !c.scalar_mul(&Integer::from(k as i64))
                    .unwrap()
                    .is_identity(),
                "the reported order {n} is not minimal: {k} already kills the class"
            );
        }
    }
}
