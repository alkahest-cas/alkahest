//! Places and divisors on the normalised hyperelliptic model.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use rug::{Integer, Rational};

use super::error::FunctionFieldError;
use super::field::FunctionField;

/// A **degree-one** place of the function field.
///
/// # What is representable, and what is not
///
/// Only places of degree 1 over ℚ live here:
///
/// * [`Place::Finite`] — a rational point `(α, β)` with `α, β ∈ ℚ` and
///   `β² = a(α)`.  `β = 0` marks a **branch** (ramified) place, where `y` is a
///   uniformiser and `v_P(x − α) = 2`.
/// * [`Place::Infinity`] — the single rational place above `x = ∞` on the
///   **imaginary** (odd-degree) model, where `v_∞(x) = −2` and
///   `v_∞(y) = −deg a`.
///
/// A conjugate pair `(α, ±√c)` with `c ∈ ℚ` a non-square is one place of
/// degree 2, and a place over an irrational `α` has degree ≥ 2; neither has a
/// representation here.  Operations that would need one refuse with
/// [`FunctionFieldError::NonRationalSupport`] rather than drop it — dropping it
/// would silently change the degree of the divisor.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Place {
    /// A finite rational place `(x, y)` with `y² = a(x)`.
    Finite {
        /// The `x`-coordinate.
        x: Rational,
        /// The `y`-coordinate; `0` at a branch point.
        y: Rational,
    },
    /// The place at infinity of the imaginary model.
    Infinity,
}

impl Place {
    /// A finite place `(x, y)`.  Not validated against any curve here — see
    /// [`Divisor::from_terms`], which is.
    pub fn finite(x: Rational, y: Rational) -> Self {
        Place::Finite { x, y }
    }

    /// The place at infinity.
    pub fn infinity() -> Self {
        Place::Infinity
    }

    /// `true` for [`Place::Infinity`].
    pub fn is_infinite(&self) -> bool {
        matches!(self, Place::Infinity)
    }

    /// The `x`-coordinate, or `None` at infinity.
    pub fn x(&self) -> Option<&Rational> {
        match self {
            Place::Finite { x, .. } => Some(x),
            Place::Infinity => None,
        }
    }

    /// The `y`-coordinate, or `None` at infinity.
    pub fn y(&self) -> Option<&Rational> {
        match self {
            Place::Finite { y, .. } => Some(y),
            Place::Infinity => None,
        }
    }

    /// The residue degree.  Always `1`: nothing else is representable.
    pub fn degree(&self) -> usize {
        1
    }

    /// `true` at a branch point `(α, 0)`, where the place is ramified over
    /// `ℚ(x)` and `v_P(x − α) = 2`.
    ///
    /// The place at infinity of the imaginary model is also ramified, and
    /// reports `true`.
    pub fn is_ramified(&self) -> bool {
        match self {
            Place::Finite { y, .. } => *y == 0,
            Place::Infinity => true,
        }
    }

    /// The hyperelliptic involution `(α, β) ↦ (α, −β)`; `∞` is fixed.
    pub fn involution(&self) -> Place {
        match self {
            Place::Finite { x, y } => Place::Finite {
                x: x.clone(),
                y: -y.clone(),
            },
            Place::Infinity => Place::Infinity,
        }
    }

    /// `v_P(x − α₀)` for a finite `α₀`, and `v_∞(x − α₀) = −2`.
    pub fn valuation_of_x_minus(&self, alpha: &Rational) -> i64 {
        match self {
            Place::Infinity => -2,
            Place::Finite { x, y } => {
                if x != alpha {
                    0
                } else if *y == 0 {
                    2
                } else {
                    1
                }
            }
        }
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Place::Finite { x, y } => write!(f, "({x}, {y})"),
            Place::Infinity => write!(f, "∞"),
        }
    }
}

/// A divisor: a finite formal ℤ-combination of degree-one places.
///
/// Carries the [`FunctionField`] it lives on, so that mixing divisors from two
/// curves is a typed refusal rather than nonsense.  Divisors exist only on the
/// **imaginary** model — the real model's two places at infinity have no
/// representation, so [`Divisor::from_terms`] refuses there.
#[derive(Clone, Debug)]
pub struct Divisor {
    field: FunctionField,
    /// Canonical: sorted by place, never holding a zero coefficient.
    terms: BTreeMap<Place, Integer>,
}

impl Divisor {
    /// The zero divisor on `field`.
    pub fn zero(field: FunctionField) -> Result<Self, FunctionFieldError> {
        field.require_imaginary("Divisor")?;
        Ok(Divisor {
            field,
            terms: BTreeMap::new(),
        })
    }

    /// Build a divisor from `(place, multiplicity)` pairs, summing repeats.
    ///
    /// Every finite place is checked against the curve: `β² = a(α)`, else
    /// [`FunctionFieldError::PlaceNotOnCurve`].  Coordinates are in the
    /// **normalised** model (see [`FunctionField::normalisation`]).
    pub fn from_terms<I>(field: FunctionField, terms: I) -> Result<Self, FunctionFieldError>
    where
        I: IntoIterator<Item = (Place, Integer)>,
    {
        field.require_imaginary("Divisor")?;
        let mut map: BTreeMap<Place, Integer> = BTreeMap::new();
        for (place, coeff) in terms {
            if let Place::Finite { x, y } = &place {
                if !field.contains_point(x, y) {
                    return Err(FunctionFieldError::PlaceNotOnCurve {
                        x: x.to_string(),
                        y: y.to_string(),
                    });
                }
            }
            *map.entry(place).or_insert_with(|| Integer::from(0)) += coeff;
        }
        map.retain(|_, v| *v != 0);
        Ok(Divisor { field, terms: map })
    }

    /// The field this divisor lives on.
    pub fn field(&self) -> &FunctionField {
        &self.field
    }

    /// The multiplicity of `place`; zero when it is not in the support.
    pub fn coefficient(&self, place: &Place) -> Integer {
        self.terms
            .get(place)
            .cloned()
            .unwrap_or_else(|| Integer::from(0))
    }

    /// `deg D = Σ nᵢ·deg Pᵢ`.  Every place here has degree 1, so this is the
    /// sum of the multiplicities.
    pub fn degree(&self) -> Integer {
        self.terms
            .values()
            .fold(Integer::from(0), |acc, v| acc + v.clone())
    }

    /// The places with non-zero multiplicity, in canonical order.
    pub fn support(&self) -> Vec<Place> {
        self.terms.keys().cloned().collect()
    }

    /// The `(place, multiplicity)` pairs, in canonical order.
    pub fn terms(&self) -> Vec<(Place, Integer)> {
        self.terms
            .iter()
            .map(|(p, c)| (p.clone(), c.clone()))
            .collect()
    }

    /// The number of places in the support.
    pub fn support_len(&self) -> usize {
        self.terms.len()
    }

    /// `true` for the zero divisor.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// `true` when every multiplicity is ≥ 0.
    pub fn is_effective(&self) -> bool {
        self.terms.values().all(|c| *c >= 0)
    }

    /// `D + E`.  Refuses when the two live on different curves.
    pub fn add(&self, other: &Divisor) -> Result<Divisor, FunctionFieldError> {
        self.field.require_same(&other.field)?;
        let mut terms = self.terms.clone();
        for (p, c) in &other.terms {
            *terms.entry(p.clone()).or_insert_with(|| Integer::from(0)) += c.clone();
        }
        terms.retain(|_, v| *v != 0);
        Ok(Divisor {
            field: self.field.clone(),
            terms,
        })
    }

    /// `−D`.
    pub fn neg(&self) -> Divisor {
        Divisor {
            field: self.field.clone(),
            terms: self
                .terms
                .iter()
                .map(|(p, c)| (p.clone(), -c.clone()))
                .collect(),
        }
    }

    /// `D − E`.
    pub fn sub(&self, other: &Divisor) -> Result<Divisor, FunctionFieldError> {
        self.add(&other.neg())
    }

    /// `k·D`.
    pub fn scale(&self, k: &Integer) -> Divisor {
        if *k == 0 {
            return Divisor {
                field: self.field.clone(),
                terms: BTreeMap::new(),
            };
        }
        Divisor {
            field: self.field.clone(),
            terms: self
                .terms
                .iter()
                .map(|(p, c)| (p.clone(), c.clone() * k.clone()))
                .collect(),
        }
    }

    /// The image of `D` under the hyperelliptic involution.
    pub fn involution(&self) -> Divisor {
        let mut terms: BTreeMap<Place, Integer> = BTreeMap::new();
        for (p, c) in &self.terms {
            *terms
                .entry(p.involution())
                .or_insert_with(|| Integer::from(0)) += c.clone();
        }
        terms.retain(|_, v| *v != 0);
        Divisor {
            field: self.field.clone(),
            terms,
        }
    }

    /// `true` when `self ≤ other` in the divisor partial order, i.e. `other −
    /// self` is effective.
    pub fn leq(&self, other: &Divisor) -> Result<bool, FunctionFieldError> {
        Ok(other.sub(self)?.is_effective())
    }

    /// The finite places of the support, as `(x, y, multiplicity)`.
    pub(crate) fn finite_terms(&self) -> Vec<(Rational, Rational, Integer)> {
        self.terms
            .iter()
            .filter_map(|(p, c)| match p {
                Place::Finite { x, y } => Some((x.clone(), y.clone(), c.clone())),
                Place::Infinity => None,
            })
            .collect()
    }

    /// The multiplicity at infinity.
    pub(crate) fn infinity_coefficient(&self) -> Integer {
        self.coefficient(&Place::Infinity)
    }
}

impl PartialEq for Divisor {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field && self.terms == other.terms
    }
}

impl Eq for Divisor {}

/// The **divisor partial order**: `D ≤ E` iff `E − D` is effective.
///
/// Genuinely partial — `P` and `Q` at distinct places are incomparable and
/// `partial_cmp` returns `None`, which is the whole reason this is not `Ord`.
impl PartialOrd for Divisor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.field != other.field {
            return None;
        }
        let mut saw_less = false;
        let mut saw_greater = false;
        let mut places: Vec<&Place> = self.terms.keys().collect();
        places.extend(other.terms.keys());
        places.sort();
        places.dedup();
        for p in places {
            match self.coefficient(p).cmp(&other.coefficient(p)) {
                Ordering::Less => saw_less = true,
                Ordering::Greater => saw_greater = true,
                Ordering::Equal => {}
            }
        }
        match (saw_less, saw_greater) {
            (false, false) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (true, true) => None,
        }
    }
}

impl fmt::Display for Divisor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (p, c) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            if *c == 1 {
                write!(f, "{p}")?;
            } else {
                write!(f, "{c}·{p}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;

    fn genus1() -> FunctionField {
        // y² = x³ − x, branch points at x ∈ {−1, 0, 1}.
        FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap()
    }

    fn p(x: i64, y: i64) -> Place {
        Place::finite(Rational::from(x), Rational::from(y))
    }

    #[test]
    fn degree_is_additive() {
        let f = genus1();
        let d1 = Divisor::from_terms(
            f.clone(),
            [
                (p(0, 0), Integer::from(3)),
                (Place::Infinity, Integer::from(-1)),
            ],
        )
        .unwrap();
        let d2 = Divisor::from_terms(f.clone(), [(p(1, 0), Integer::from(2))]).unwrap();
        assert_eq!(d1.degree(), Integer::from(2));
        assert_eq!(d2.degree(), Integer::from(2));
        assert_eq!(d1.add(&d2).unwrap().degree(), Integer::from(4));
    }

    #[test]
    fn repeated_places_are_summed_and_zeroes_dropped() {
        let f = genus1();
        let d = Divisor::from_terms(
            f,
            [
                (p(0, 0), Integer::from(2)),
                (p(0, 0), Integer::from(-2)),
                (p(1, 0), Integer::from(5)),
            ],
        )
        .unwrap();
        assert_eq!(d.support_len(), 1);
        assert_eq!(d.coefficient(&p(0, 0)), Integer::from(0));
        assert_eq!(d.coefficient(&p(1, 0)), Integer::from(5));
    }

    #[test]
    fn a_place_off_the_curve_is_refused() {
        let f = genus1();
        // (2, 0): 2³ − 2 = 6 ≠ 0.
        let err = Divisor::from_terms(f, [(p(2, 0), Integer::from(1))]).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-004");
    }

    #[test]
    fn negation_and_scaling() {
        let f = genus1();
        let d = Divisor::from_terms(
            f,
            [
                (p(0, 0), Integer::from(3)),
                (Place::Infinity, Integer::from(-3)),
            ],
        )
        .unwrap();
        assert_eq!(d.neg().coefficient(&p(0, 0)), Integer::from(-3));
        assert_eq!(d.scale(&Integer::from(4)).degree(), Integer::from(0));
        assert!(d.scale(&Integer::from(0)).is_zero());
        assert!(d.add(&d.neg()).unwrap().is_zero());
    }

    #[test]
    fn the_partial_order_is_partial() {
        let f = genus1();
        let a = Divisor::from_terms(f.clone(), [(p(0, 0), Integer::from(1))]).unwrap();
        let b = Divisor::from_terms(f.clone(), [(p(1, 0), Integer::from(1))]).unwrap();
        let ab = a.add(&b).unwrap();
        assert!(a < ab);
        assert!(ab > b);
        // Distinct single places are incomparable — not "equal", not "less".
        assert_eq!(a.partial_cmp(&b), None);
        assert!(a.leq(&ab).unwrap());
        assert!(!a.leq(&b).unwrap());
    }

    #[test]
    fn effectivity() {
        let f = genus1();
        let d = Divisor::from_terms(
            f.clone(),
            [
                (p(0, 0), Integer::from(1)),
                (Place::Infinity, Integer::from(-1)),
            ],
        )
        .unwrap();
        assert!(!d.is_effective());
        assert!(Divisor::zero(f).unwrap().is_effective());
    }

    #[test]
    fn involution_flips_the_sheet_and_fixes_branch_points() {
        let f = genus1();
        // (−1, 0) is a branch point, so it is fixed.
        let d = Divisor::from_terms(f.clone(), [(p(-1, 0), Integer::from(1))]).unwrap();
        assert_eq!(d.involution(), d);
        // y² = x³ − x at x = 2 gives y² = 6, not rational; use x = 9/4:
        //   (9/4)³ − 9/4 = 729/64 − 144/64 = 585/64, not a square either.
        // Use the genus-2 style check on a curve with a rational non-branch point.
        let g = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 1]).unwrap(); // y² = x³ + 1
        let pt = Place::finite(Rational::from(2), Rational::from(3)); // 8+1 = 9
        let d = Divisor::from_terms(g, [(pt.clone(), Integer::from(1))]).unwrap();
        assert_eq!(
            d.involution().support(),
            vec![Place::finite(Rational::from(2), Rational::from(-3))]
        );
    }

    #[test]
    fn divisors_on_different_curves_do_not_mix() {
        let a = Divisor::zero(genus1()).unwrap();
        let b =
            Divisor::zero(FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 1]).unwrap()).unwrap();
        let err = a.add(&b).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-009");
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    fn a_real_model_has_no_divisors_at_all() {
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 1]).unwrap();
        let err = Divisor::zero(f).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-002");
    }

    #[test]
    fn valuation_of_x_minus_alpha() {
        // Ramified: v_P(x − α) = 2; unramified: 1; at ∞: −2.
        assert_eq!(p(0, 0).valuation_of_x_minus(&Rational::from(0)), 2);
        let q = Place::finite(Rational::from(2), Rational::from(3));
        assert_eq!(q.valuation_of_x_minus(&Rational::from(2)), 1);
        assert_eq!(q.valuation_of_x_minus(&Rational::from(5)), 0);
        assert_eq!(Place::Infinity.valuation_of_x_minus(&Rational::from(0)), -2);
    }

    #[test]
    fn display_is_readable() {
        let f = genus1();
        let d = Divisor::from_terms(
            f,
            [
                (p(0, 0), Integer::from(2)),
                (Place::Infinity, Integer::from(-2)),
            ],
        )
        .unwrap();
        assert_eq!(d.to_string(), "2·(0, 0) + -2·∞");
    }
}
