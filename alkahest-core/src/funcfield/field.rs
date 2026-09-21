//! The function field `ℚ(x)[y]/(f(x, y))` and its normalised curve model.

use std::fmt;

use rug::Rational;

use super::divisor::{Divisor, Place};
use super::error::FunctionFieldError;
use super::util::{constant, content_normalise, eval, is_zero, squarefree_split};
use crate::integrate::risch::poly_rde::{degree, poly_mul, poly_scale, qpoly_to_expr, trim, QPoly};
use crate::integrate::risch::rational_rde::poly_sub;
use crate::kernel::{ExprId, ExprPool};

/// How the caller's `y` was rewritten to reach the normalised model.
///
/// The normalisation is a ℚ(x)-isomorphism of function fields, so genus,
/// divisor class group and Riemann–Roch are unaffected by it — but the
/// **coordinates of places are not**.  A [`Place`] is always given in the
/// normalised model, and this record says how to translate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Normalisation {
    /// The constant `c₂` (the `y²` coefficient of the input).
    pub leading: Rational,
    /// The polynomial `c₁(x)` (the `y¹` coefficient of the input).
    pub linear_term: QPoly,
    /// The monic `b(x)` with `c₁² − 4c₂c₀ = a(x)·b(x)²` before the constant is
    /// canonicalised.
    pub square_factor: QPoly,
    /// The final rational scaling `c` of `y`, absorbing square constants.
    pub scale: rug::Rational,
}

impl Normalisation {
    /// `true` when the normalisation is the identity — the input was already
    /// `y² − a(x)` with `a` squarefree, and places mean what the caller wrote.
    pub fn is_identity(&self) -> bool {
        self.leading == 1
            && is_zero(&self.linear_term)
            && degree(&self.square_factor) == 0
            && self.scale == 1
    }
}

impl fmt::Display for Normalisation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "Y = y (identity)");
        }
        write!(
            f,
            "Y = {}·(2·{}·y + c₁(x)) / b(x), with deg c₁ = {} and deg b = {}",
            self.scale,
            self.leading,
            degree(&self.linear_term),
            degree(&self.square_factor)
        )
    }
}

/// An algebraic function field `ℚ(x)[y]/(f(x, y))`, normalised to a
/// hyperelliptic model `y² = a(x)` with `a` squarefree of degree ≥ 1.
///
/// # Which models are supported
///
/// Construction accepts exactly
///
/// ```text
///     f(x, y) = c₂·y² + c₁(x)·y + c₀(x),      c₂ ∈ ℚ∖{0} a constant,
/// ```
///
/// and normalises it by completing the square and peeling square factors:
///
/// ```text
///     Y = 2c₂y + c₁(x)   ⇒   Y² = c₁² − 4c₂c₀ =: A(x) = a(x)·b(x)²
///     Z = Y / b(x)       ⇒   Z² = a(x),     a squarefree.
/// ```
///
/// Both steps are ℚ(x)-isomorphisms, so nothing about the field changes; the
/// **coordinates of places do**, and [`FunctionField::normalisation`] records
/// the map.  When the input is already `y² − a(x)` with `a` squarefree the
/// normalisation is the identity.
///
/// Refused at construction ([`FunctionFieldError::UnsupportedModel`]):
///
/// * `deg_y f ≠ 2` — trigonal and higher plane models, superelliptic `yⁿ = a`
///   with `n > 2`.  The Mumford/Cantor machinery this layer reuses is
///   `n = 2` only.
/// * a non-constant `c₂(x)`.  Clearing it is a further birational change this
///   module does not perform, so it declines rather than guess.
/// * `A(x)` identically zero, or a non-zero constant: `f` is then reducible
///   over `ℚ̄(x)` and defines no function field of positive transcendence
///   degree over ℚ.
#[derive(Clone, Debug)]
pub struct FunctionField {
    /// `y² = a(x)`, `a` squarefree, `deg a ≥ 1`.
    a: QPoly,
    genus: usize,
    normalisation: Normalisation,
}

impl PartialEq for FunctionField {
    /// Two function fields are the same iff their **normalised** models agree.
    ///
    /// The route taken to the model is not part of the field's identity: `2y² =
    /// 2x³ − 2x` and `y² = x³ − x` are the same object here.
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a
    }
}

impl Eq for FunctionField {}

impl FunctionField {
    /// Build the field from the `y`-coefficients of `f`: `f = Σ coeffs[i]·yⁱ`.
    ///
    /// See the type documentation for the accepted models.
    pub fn new(coeffs: &[QPoly]) -> Result<Self, FunctionFieldError> {
        let coeffs: Vec<QPoly> = coeffs.iter().map(|c| trim(c.clone())).collect();
        // Strip trailing zero coefficients to find deg_y.
        let mut top = coeffs.len();
        while top > 0 && is_zero(&coeffs[top - 1]) {
            top -= 1;
        }
        if top != 3 {
            return Err(FunctionFieldError::UnsupportedModel {
                reason: format!(
                    "deg_y f = {}, but only quadratic models (deg_y f = 2) are implemented",
                    top as i64 - 1
                ),
            });
        }
        let c2 = &coeffs[2];
        if degree(c2) != 0 {
            return Err(FunctionFieldError::UnsupportedModel {
                reason: format!(
                    "the y² coefficient has degree {} in x; only a non-zero rational constant is accepted",
                    degree(c2)
                ),
            });
        }
        let gamma = c2[0].clone();
        let c1 = coeffs[1].clone();
        let c0 = coeffs[0].clone();

        // Complete the square: (2c₂y + c₁)² = c₁² − 4c₂c₀.
        let four_c2_c0 = poly_scale(&c0, &(Rational::from(4) * &gamma));
        let big_a = trim(poly_sub(&poly_mul(&c1, &c1), &four_c2_c0));
        Self::from_discriminant(big_a, gamma, c1)
    }

    /// Build `y² = a(x)` directly.  `a` is normalised by peeling square factors.
    pub fn hyperelliptic(a: &QPoly) -> Result<Self, FunctionFieldError> {
        Self::from_discriminant(trim(a.clone()), Rational::from(1), Vec::new())
    }

    /// Build `y² = a(x)` from integer coefficients, lowest degree first.
    pub fn hyperelliptic_from_i64(a: &[i64]) -> Result<Self, FunctionFieldError> {
        let q: QPoly = a.iter().map(|&c| Rational::from(c)).collect();
        Self::hyperelliptic(&trim(q))
    }

    fn from_discriminant(
        big_a: QPoly,
        gamma: Rational,
        c1: QPoly,
    ) -> Result<Self, FunctionFieldError> {
        if is_zero(&big_a) {
            return Err(FunctionFieldError::UnsupportedModel {
                reason: "the discriminant c₁² − 4c₂c₀ is identically zero, so f is a perfect \
                         square and defines a double line, not a curve"
                    .into(),
            });
        }
        let (a, b) = squarefree_split(&big_a);
        // Absorb square constants into `y`; a squarefree constant is a genuine
        // quadratic twist and stays on the curve.
        let (a, scale) = content_normalise(&a);
        if degree(&a) < 1 {
            return Err(FunctionFieldError::UnsupportedModel {
                reason: format!(
                    "after removing square factors the discriminant is the constant {}, so \
                     y is algebraic over ℚ and f factors over a quadratic number field; \
                     this is not a function field of a curve",
                    a.first().cloned().unwrap_or_else(|| Rational::from(0))
                ),
            });
        }
        let genus = crate::integrate::algebraic::find_order::genus(2, &a).ok_or_else(|| {
            FunctionFieldError::UnsupportedModel {
                reason: "the genus formula declined the normalised model (a is not squarefree \
                         of positive degree)"
                    .into(),
            }
        })?;
        Ok(FunctionField {
            a,
            genus,
            normalisation: Normalisation {
                leading: gamma,
                linear_term: c1,
                square_factor: b,
                scale,
            },
        })
    }

    /// The normalised curve `a(x)` in `y² = a(x)`, squarefree of degree ≥ 1.
    pub fn curve(&self) -> &QPoly {
        &self.a
    }

    /// `deg a`, i.e. `2g+1` on the imaginary model and `2g+2` on the real one.
    pub fn curve_degree(&self) -> usize {
        degree(&self.a) as usize
    }

    /// The geometric genus.
    ///
    /// Model-independent, and available for **every** accepted model —
    /// including the even-degree ones whose divisor arithmetic is refused.
    /// For `y² = a(x)` with `a` squarefree of degree `2g+1` or `2g+2` this is
    /// `g`.
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// `true` for the **imaginary** model, `deg a` odd: one rational place
    /// above `x = ∞`, and the base point every divisor class is measured from.
    ///
    /// Divisors, the class group and Riemann–Roch are implemented only here.
    pub fn is_imaginary(&self) -> bool {
        self.curve_degree() % 2 == 1
    }

    /// How the caller's `y` maps to the normalised one.
    pub fn normalisation(&self) -> &Normalisation {
        &self.normalisation
    }

    /// Refuse `operation` when the model is the even-degree ("real") one.
    pub(crate) fn require_imaginary(
        &self,
        operation: &'static str,
    ) -> Result<(), FunctionFieldError> {
        if self.is_imaginary() {
            Ok(())
        } else {
            Err(FunctionFieldError::RealModel {
                degree: self.curve_degree(),
                genus: self.genus,
                operation,
            })
        }
    }

    /// `true` when `(α, β)` satisfies `β² = a(α)`.
    pub fn contains_point(&self, x: &Rational, y: &Rational) -> bool {
        Rational::from(y * y) == eval(&self.a, x)
    }

    /// `true` when the finite place above `α` is ramified, i.e. `a(α) = 0`.
    pub fn is_branch_point(&self, x: &Rational) -> bool {
        eval(&self.a, x) == 0
    }

    /// Refuse when two operands live on different curves.
    pub(crate) fn require_same(&self, other: &Self) -> Result<(), FunctionFieldError> {
        if self == other {
            Ok(())
        } else {
            Err(FunctionFieldError::CurveMismatch {
                left: self.to_string(),
                right: other.to_string(),
            })
        }
    }

    /// A canonical divisor.
    ///
    /// On the imaginary model `K = (2g − 2)·∞` is canonical: `div(dx/y) =
    /// (2g−2)·∞`.  Its degree is `2g − 2` and `dim L(K) = g`, both of which
    /// this module's tests check directly.  For `g = 0` the coefficient is
    /// `−2`, a perfectly good non-effective divisor.
    ///
    /// Refused on the real model — there `K = (g−1)·(∞₊ + ∞₋)`, and `∞₊`, `∞₋`
    /// are not representable here.
    pub fn canonical_divisor(&self) -> Result<Divisor, FunctionFieldError> {
        self.require_imaginary("canonical_divisor")?;
        let k = 2 * self.genus as i64 - 2;
        Divisor::from_terms(self.clone(), [(Place::Infinity, rug::Integer::from(k))])
    }

    /// `a(x)` as a symbolic expression in `var`.
    pub fn curve_expr(&self, var: ExprId, pool: &ExprPool) -> ExprId {
        qpoly_to_expr(&self.a, var, pool)
    }

    /// The constant polynomial `1`, handy for callers building elements.
    pub(crate) fn one(&self) -> QPoly {
        constant(Rational::from(1))
    }
}

impl fmt::Display for FunctionField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y² = ")?;
        let d = degree(&self.a);
        if d < 0 {
            return write!(f, "0");
        }
        let mut first = true;
        for k in (0..=d as usize).rev() {
            let c = &self.a[k];
            if *c == 0 {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            match k {
                0 => write!(f, "{c}")?,
                1 if *c == 1 => write!(f, "x")?,
                1 => write!(f, "{c}·x")?,
                _ if *c == 1 => write!(f, "x^{k}")?,
                _ => write!(f, "{c}·x^{k}")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        trim(cs.iter().map(|&c| Rational::from(c)).collect())
    }

    #[test]
    fn genus_of_the_standard_models() {
        // y² = x³ − x : deg 3 = 2·1+1 ⇒ g = 1.
        let e = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
        assert_eq!(e.genus(), 1);
        assert!(e.is_imaginary());

        // y² = x⁵ + 1 : deg 5 = 2·2+1 ⇒ g = 2.
        let c2 = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 0, 1]).unwrap();
        assert_eq!(c2.genus(), 2);
        assert!(c2.is_imaginary());

        // y² = x : deg 1 ⇒ g = 0, still imaginary.
        let r = FunctionField::hyperelliptic_from_i64(&[0, 1]).unwrap();
        assert_eq!(r.genus(), 0);
        assert!(r.is_imaginary());
    }

    #[test]
    fn even_degree_models_report_genus_but_are_real() {
        // y² = x⁴ + 1 : deg 4 = 2·1+2 ⇒ g = 1, real model.
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 1]).unwrap();
        assert_eq!(f.genus(), 1);
        assert!(!f.is_imaginary());

        // y² = x⁶ + x + 1 : deg 6 = 2·2+2 ⇒ g = 2.
        let f = FunctionField::hyperelliptic_from_i64(&[1, 1, 0, 0, 0, 0, 1]).unwrap();
        assert_eq!(f.genus(), 2);
        assert!(!f.is_imaginary());
    }

    #[test]
    fn a_real_model_refuses_divisor_operations_but_not_genus() {
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 1]).unwrap();
        assert_eq!(f.genus(), 1);
        let err = f.canonical_divisor().unwrap_err();
        assert!(matches!(err, FunctionFieldError::RealModel { .. }));
        use crate::errors::AlkahestError;
        assert_eq!(err.code(), "E-FFLD-002");
    }

    #[test]
    fn completing_the_square_normalises_a_general_quadratic() {
        // f = y² + 2x·y + (x² − x³)  ⇒  (2y + 2x)² = 4x³.
        // 4x³ is not squarefree: peeling x² leaves 4x, and 4 is a square
        // constant, so the normal form is Z² = x with g = 0.
        let f = FunctionField::new(&[qp(&[0, 0, 1, -1]), qp(&[0, 2]), qp(&[1])]).unwrap();
        assert_eq!(f.curve(), &qp(&[0, 1]));
        assert_eq!(f.genus(), 0);
        assert!(!f.normalisation().is_identity());
    }

    #[test]
    fn a_scalar_multiple_of_a_model_is_the_same_field() {
        let a = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
        // 2y² = 2x³ − 2x  ⇒  same field.
        let b = FunctionField::new(&[qp(&[0, 2, 0, -2]), Vec::new(), qp(&[2])]).unwrap();
        assert_eq!(a, b, "the normalised models must coincide");
        assert_eq!(b.genus(), 1);
    }

    #[test]
    fn identity_normalisation_is_reported_as_such() {
        let f = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
        assert!(f.normalisation().is_identity());
    }

    #[test]
    fn higher_degree_models_are_refused_not_guessed() {
        use crate::errors::AlkahestError;
        // y³ = x⁴ − 1 — a genuine superelliptic curve, and out of scope.
        let err = FunctionField::new(&[qp(&[1, 0, 0, 0, -1]), Vec::new(), Vec::new(), qp(&[1])])
            .unwrap_err();
        assert_eq!(err.code(), "E-FFLD-001");
        assert!(format!("{err}").contains("deg_y f = 3"));
    }

    #[test]
    fn a_non_constant_leading_coefficient_is_refused() {
        use crate::errors::AlkahestError;
        // x·y² = x³ − 1.
        let err = FunctionField::new(&[qp(&[1, 0, 0, -1]), Vec::new(), qp(&[0, 1])]).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-001");
    }

    #[test]
    fn a_perfect_square_is_refused() {
        use crate::errors::AlkahestError;
        // (y + x)² = y² + 2xy + x².
        let err = FunctionField::new(&[qp(&[0, 0, 1]), qp(&[0, 2]), qp(&[1])]).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-001");
        assert!(format!("{err}").contains("identically zero"));
    }

    #[test]
    fn a_constant_discriminant_is_refused() {
        use crate::errors::AlkahestError;
        // y² = 2 — a number field, not a function field.
        let err = FunctionField::hyperelliptic_from_i64(&[2]).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-001");
    }

    #[test]
    fn contains_point_checks_the_normalised_model() {
        let f = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
        // (−1, 0), (0, 0), (1, 0) are the branch points of y² = x³ − x.
        for x in [-1i64, 0, 1] {
            assert!(f.contains_point(&Rational::from(x), &Rational::from(0)));
            assert!(f.is_branch_point(&Rational::from(x)));
        }
        // (2, 0) is not on the curve: 2³ − 2 = 6 ≠ 0.
        assert!(!f.contains_point(&Rational::from(2), &Rational::from(0)));
        // but (2, ±√6) would be — and √6 ∉ ℚ, so no rational place lies over 2.
    }

    #[test]
    fn display_renders_the_normalised_curve() {
        let f = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
        assert_eq!(f.to_string(), "y² = x^3 + -1·x");
    }
}
