//! Elements of the function field, and the divisor of a function.
//!
//! Every element of `ℚ(x)[y]/(y² − a(x))` is `(p(x) + q(x)·y) / d(x)` with
//! `p, q, d ∈ ℚ[x]`, `d ≠ 0`, and that is exactly how [`FunctionFieldElement`]
//! stores it — no expression parsing, no ambiguity about which square root is
//! meant.
//!
//! # How `div(u)` is computed, and where it refuses
//!
//! Write `N(u) = p² − q²·a`, the norm down to `ℚ(x)`.  For a finite `α ∈ ℚ`,
//!
//! ```text
//!     v_α(N(u)) = Σ_{P | α}  f_P · v_P(u),
//! ```
//!
//! so on the imaginary model the multiplicity of `(x − α)` in the norm splits
//! the local contribution between the (at most two) places above `α`:
//!
//! * `a(α) = 0` — one ramified place `(α, 0)`, `f = 1`, so `v_P(u) = v_α(N)`.
//! * `a(α) = β²` with `β ∈ ℚ∖{0}` — two rational places `(α, ±β)`.  Let `k` be
//!   the largest power of `(x − α)` dividing both `p` and `q`.  After dividing
//!   it out, `p₁(α) + βq₁(α)` and `p₁(α) − βq₁(α)` cannot both vanish, so
//!   whichever is non-zero pins that place's valuation at `k` and degree
//!   balance gives the other.
//! * `a(α)` a non-zero **non-square** — the place above `α` has degree 2 and is
//!   not representable: [`FunctionFieldError::NonRationalSupport`].
//!
//! The multiplicity at `∞` is then fixed by `deg div(u) = 0`, and **checked**
//! against `v_∞(p + q·y) = min(−2·deg p, −deg a − 2·deg q)` — the two are never
//! equal on the odd-degree model, one being even and the other odd, so the
//! minimum is exact and the cross-check is sharp.  A disagreement is withheld
//! as [`FunctionFieldError::SelfCheckFailed`] rather than returned.
//!
//! A norm (or denominator) that does not split into rational linear factors
//! means the divisor has a place of degree ≥ 2 in its support, and is refused.
//! That refusal is *conservative*: the rational-root search is capped, so a
//! `NonRationalSupport` never proves irreducibility.

use rug::{Integer, Rational};

use super::divisor::{Divisor, Place};
use super::error::FunctionFieldError;
use super::field::FunctionField;
use super::util::{constant, deflate, eval, is_zero, rational_roots, rational_sqrt, valuation_at};
use crate::integrate::risch::poly_rde::{degree, poly_add, poly_mul, qpoly_to_expr, trim, QPoly};
use crate::integrate::risch::rational_rde::poly_sub;
use crate::kernel::{ExprId, ExprPool};

/// `(p(x) + q(x)·y) / d(x)` in `ℚ(x)[y]/(y² − a(x))`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionFieldElement {
    field: FunctionField,
    p: QPoly,
    q: QPoly,
    d: QPoly,
}

impl FunctionFieldElement {
    /// `(p + q·y)/d`.  Refuses `d = 0`.
    pub fn new(
        field: FunctionField,
        p: QPoly,
        q: QPoly,
        d: QPoly,
    ) -> Result<Self, FunctionFieldError> {
        let d = trim(d);
        if is_zero(&d) {
            return Err(FunctionFieldError::UnsupportedModel {
                reason: "a function-field element cannot have a zero denominator".into(),
            });
        }
        Ok(FunctionFieldElement {
            field,
            p: trim(p),
            q: trim(q),
            d,
        })
    }

    /// `p(x) + q(x)·y`.
    pub fn polynomial(field: FunctionField, p: QPoly, q: QPoly) -> Self {
        let d = field.one();
        FunctionFieldElement {
            field,
            p: trim(p),
            q: trim(q),
            d,
        }
    }

    /// The constant function `c`.
    pub fn constant(field: FunctionField, c: Rational) -> Self {
        Self::polynomial(field, constant(c), Vec::new())
    }

    /// The coordinate function `x`.
    pub fn x(field: FunctionField) -> Self {
        Self::polynomial(
            field,
            vec![Rational::from(0), Rational::from(1)],
            Vec::new(),
        )
    }

    /// The coordinate function `y`.
    pub fn y(field: FunctionField) -> Self {
        let one = field.one();
        Self::polynomial(field, Vec::new(), one)
    }

    /// The field this element lives in.
    pub fn field(&self) -> &FunctionField {
        &self.field
    }

    /// The `y⁰` part of the numerator.
    pub fn numerator_rational_part(&self) -> &QPoly {
        &self.p
    }

    /// The `y¹` part of the numerator.
    pub fn numerator_algebraic_part(&self) -> &QPoly {
        &self.q
    }

    /// The denominator.
    pub fn denominator(&self) -> &QPoly {
        &self.d
    }

    /// `true` for the zero function.
    pub fn is_zero(&self) -> bool {
        is_zero(&self.p) && is_zero(&self.q)
    }

    /// The norm `N(u) = p² − q²·a` of the numerator, down to `ℚ(x)`.
    pub fn numerator_norm(&self) -> QPoly {
        let p2 = poly_mul(&self.p, &self.p);
        let q2a = poly_mul(&poly_mul(&self.q, &self.q), self.field.curve());
        trim(poly_sub(&p2, &q2a))
    }

    /// Product of two elements of the same field.
    pub fn mul(&self, other: &Self) -> Result<Self, FunctionFieldError> {
        self.field.require_same(&other.field)?;
        let a = self.field.curve();
        let p = poly_add(
            &poly_mul(&self.p, &other.p),
            &poly_mul(&poly_mul(&self.q, &other.q), a),
        );
        let q = poly_add(&poly_mul(&self.p, &other.q), &poly_mul(&self.q, &other.p));
        Ok(FunctionFieldElement {
            field: self.field.clone(),
            p: trim(p),
            q: trim(q),
            d: trim(poly_mul(&self.d, &other.d)),
        })
    }

    /// `div(u)`, the divisor of this function.
    ///
    /// See the module documentation for the algorithm and its refusals.
    pub fn divisor(&self) -> Result<Divisor, FunctionFieldError> {
        self.field.require_imaginary("divisor_of_function")?;
        if self.is_zero() {
            return Err(FunctionFieldError::ZeroFunction);
        }
        let num = numerator_divisor(&self.field, &self.p, &self.q, "the numerator")?;
        let den = numerator_divisor(&self.field, &self.d, &Vec::new(), "the denominator")?;
        num.sub(&den)
    }

    /// `(p(x) + q(x)·√a(x)) / d(x)` as a symbolic expression in `var`.
    pub fn to_expr(&self, var: ExprId, pool: &ExprPool) -> ExprId {
        let p_e = qpoly_to_expr(&self.p, var, pool);
        let num = if is_zero(&self.q) {
            p_e
        } else {
            let root = pool.func("sqrt", vec![self.field.curve_expr(var, pool)]);
            let q_e = qpoly_to_expr(&self.q, var, pool);
            let term = pool.mul(vec![q_e, root]);
            if is_zero(&self.p) {
                term
            } else {
                pool.add(vec![p_e, term])
            }
        };
        if degree(&self.d) == 0 && self.d[0] == 1 {
            num
        } else {
            let d_e = qpoly_to_expr(&self.d, var, pool);
            pool.mul(vec![num, pool.pow(d_e, pool.integer(-1_i32))])
        }
    }
}

/// `div(p + q·y)` for `p, q ∈ ℚ[x]` not both zero.
fn numerator_divisor(
    field: &FunctionField,
    p: &QPoly,
    q: &QPoly,
    what: &'static str,
) -> Result<Divisor, FunctionFieldError> {
    let a = field.curve();
    let p = trim(p.clone());
    let q = trim(q.clone());
    if is_zero(&p) && is_zero(&q) {
        return Err(FunctionFieldError::ZeroFunction);
    }
    // N = p² − q²·a.  Zero only when a = (p/q)², impossible for a squarefree
    // of degree ≥ 1, so a zero here would mean the model invariant broke.
    let norm = trim(poly_sub(&poly_mul(&p, &p), &poly_mul(&poly_mul(&q, &q), a)));
    if is_zero(&norm) {
        return Err(FunctionFieldError::SelfCheckFailed {
            detail: format!(
                "the norm of {what} vanished, which cannot happen for a squarefree curve"
            ),
        });
    }

    let roots = rational_roots(&norm).ok_or_else(|| FunctionFieldError::NonRationalSupport {
        context: format!("{what} has a norm that does not split into rational linear factors"),
    })?;

    let mut terms: Vec<(Place, Integer)> = Vec::new();
    let mut total = Integer::from(0);

    for (alpha, mult) in roots {
        let m = Integer::from(mult as u64);
        let a_alpha = eval(a, &alpha);
        if a_alpha == 0 {
            // Ramified: one place, f = 1, so v_P = v_α(N) directly.
            terms.push((Place::finite(alpha, Rational::from(0)), m.clone()));
            total += m;
            continue;
        }
        let Some(beta) = rational_sqrt(&a_alpha) else {
            return Err(FunctionFieldError::NonRationalSupport {
                context: format!(
                    "{what} vanishes over x = {alpha}, where a(x) = {a_alpha} is not a rational \
                     square, so the place there has degree 2"
                ),
            });
        };
        // k = v_α(gcd(p, q)), with v_α(0) = +∞.
        let k = match (valuation_at(&p, &alpha), valuation_at(&q, &alpha)) {
            (Some(kp), Some(kq)) => kp.min(kq),
            (Some(kp), None) => kp,
            (None, Some(kq)) => kq,
            (None, None) => unreachable!("p and q are not both zero"),
        };
        let p1 = deflate(&p, &alpha, k);
        let q1 = deflate(&q, &alpha, k);
        let at_plus = eval(&p1, &alpha) + Rational::from(&beta * &eval(&q1, &alpha));
        let kk = Integer::from(k as u64);
        let (v_plus, v_minus) = if at_plus != 0 {
            (kk.clone(), m.clone() - kk)
        } else {
            (m.clone() - kk.clone(), kk)
        };
        if v_plus < 0 || v_minus < 0 {
            return Err(FunctionFieldError::SelfCheckFailed {
                detail: format!(
                    "splitting the norm multiplicity {m} at x = {alpha} produced a negative \
                     valuation ({v_plus}, {v_minus})"
                ),
            });
        }
        terms.push((Place::finite(alpha.clone(), beta.clone()), v_plus));
        terms.push((Place::finite(alpha, -beta), v_minus));
        total += m;
    }

    // Degree balance fixes the multiplicity at infinity.
    let at_infinity = -total.clone();

    // Cross-check: on the odd model, v_∞(p + q·y) = min(−2·deg p, −deg a − 2·deg q),
    // and the two candidates always differ in parity, so the minimum is exact.
    let expected = infinity_valuation(field, &p, &q);
    if at_infinity != expected {
        return Err(FunctionFieldError::SelfCheckFailed {
            detail: format!(
                "v_∞ from degree balance is {at_infinity}, but the degrees of {what} give \
                 {expected}"
            ),
        });
    }

    terms.push((Place::Infinity, at_infinity));
    Divisor::from_terms(field.clone(), terms)
}

/// `v_∞(p + q·y)` on the imaginary model, from the degrees alone.
fn infinity_valuation(field: &FunctionField, p: &QPoly, q: &QPoly) -> i64 {
    let da = field.curve_degree() as i64;
    let from_p = if is_zero(p) { i64::MAX } else { -2 * degree(p) };
    let from_q = if is_zero(q) {
        i64::MAX
    } else {
        -da - 2 * degree(q)
    };
    from_p.min(from_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;

    fn qp(cs: &[i64]) -> QPoly {
        trim(cs.iter().map(|&c| Rational::from(c)).collect())
    }

    /// y² = x³ − x, genus 1, branch points (−1,0), (0,0), (1,0).
    fn e1() -> FunctionField {
        FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap()
    }

    /// y² = x⁵ + 1, genus 2.
    fn c2() -> FunctionField {
        FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 0, 1]).unwrap()
    }

    fn pl(x: i64, y: i64) -> Place {
        Place::finite(Rational::from(x), Rational::from(y))
    }

    #[test]
    fn divisor_of_x_on_a_branch_point() {
        // On y² = x³ − x, x = 0 is a branch point, so div(x) = 2·(0,0) − 2·∞.
        let f = e1();
        let d = FunctionFieldElement::x(f).divisor().unwrap();
        assert_eq!(d.coefficient(&pl(0, 0)), Integer::from(2));
        assert_eq!(d.coefficient(&Place::Infinity), Integer::from(-2));
        assert_eq!(d.degree(), Integer::from(0));
    }

    #[test]
    fn divisor_of_y_is_the_branch_locus() {
        // div(y) = (−1,0) + (0,0) + (1,0) − 3·∞.
        let f = e1();
        let d = FunctionFieldElement::y(f).divisor().unwrap();
        for x in [-1i64, 0, 1] {
            assert_eq!(d.coefficient(&pl(x, 0)), Integer::from(1));
        }
        assert_eq!(d.coefficient(&Place::Infinity), Integer::from(-3));
        assert_eq!(d.degree(), Integer::from(0));
    }

    #[test]
    fn divisor_of_x_at_a_non_branch_rational_fibre() {
        // y² = x³ + 1 has (2, ±3).  div(x − 2) = (2,3) + (2,−3) − 2·∞.
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 1]).unwrap();
        let u = FunctionFieldElement::polynomial(f, qp(&[-2, 1]), Vec::new());
        let d = u.divisor().unwrap();
        assert_eq!(d.coefficient(&pl(2, 3)), Integer::from(1));
        assert_eq!(d.coefficient(&pl(2, -3)), Integer::from(1));
        assert_eq!(d.coefficient(&Place::Infinity), Integer::from(-2));
    }

    #[test]
    fn divisor_of_y_minus_beta_lands_on_one_sheet_only() {
        // y² = x³ + 1, place (2, 3).  y − 3 vanishes at (2,3) and also at the
        // other points with y = 3: x³ + 1 = 9 ⇒ x³ = 8 ⇒ x = 2 (triple in the
        // norm sense).  N(y − 3) = 9 − (x³+1) = 8 − x³ = −(x − 2)(x² + 2x + 4),
        // and x² + 2x + 4 has no rational root, so this must refuse.
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 1]).unwrap();
        let u = FunctionFieldElement::polynomial(f, qp(&[-3]), qp(&[1]));
        let err = u.divisor().unwrap_err();
        assert_eq!(err.code(), "E-FFLD-003");
    }

    #[test]
    fn divisor_of_y_minus_beta_when_the_norm_splits() {
        // y² = x³ − x.  Take y − 0 handled above; instead use the curve
        // y² = x²(x+1) — not squarefree — so use y² = (x−1)(x−4)(x−9) and the
        // function y, whose divisor is the branch locus.  Simpler sharp case:
        // on y² = x³ + 1, the function y − 1: N = 1 − x³ − 1 = −x³.
        // Only place above x = 0 with y = 1 is (0,1); (0,−1) is the other.
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 1]).unwrap();
        let u = FunctionFieldElement::polynomial(f, qp(&[-1]), qp(&[1]));
        let d = u.divisor().unwrap();
        // v at (0,1) must be 3 and at (0,−1) must be 0.
        assert_eq!(d.coefficient(&pl(0, 1)), Integer::from(3));
        assert_eq!(d.coefficient(&pl(0, -1)), Integer::from(0));
        assert_eq!(d.coefficient(&Place::Infinity), Integer::from(-3));
        assert_eq!(d.degree(), Integer::from(0));
    }

    #[test]
    fn divisor_of_a_quotient_subtracts() {
        // u = y/x on y² = x³ − x.
        // div(y) = (−1,0)+(0,0)+(1,0) − 3∞ ; div(x) = 2(0,0) − 2∞.
        // div(u) = (−1,0) − (0,0) + (1,0) − ∞.
        let f = e1();
        let u = FunctionFieldElement::new(f, Vec::new(), qp(&[1]), qp(&[0, 1])).unwrap();
        let d = u.divisor().unwrap();
        assert_eq!(d.coefficient(&pl(-1, 0)), Integer::from(1));
        assert_eq!(d.coefficient(&pl(0, 0)), Integer::from(-1));
        assert_eq!(d.coefficient(&pl(1, 0)), Integer::from(1));
        assert_eq!(d.coefficient(&Place::Infinity), Integer::from(-1));
        assert_eq!(d.degree(), Integer::from(0));
    }

    #[test]
    fn divisor_is_additive_under_multiplication() {
        let f = e1();
        let u = FunctionFieldElement::x(f.clone());
        let v = FunctionFieldElement::y(f);
        let du = u.divisor().unwrap();
        let dv = v.divisor().unwrap();
        let duv = u.mul(&v).unwrap().divisor().unwrap();
        assert_eq!(duv, du.add(&dv).unwrap());
    }

    #[test]
    fn a_constant_has_the_zero_divisor() {
        let f = e1();
        let d = FunctionFieldElement::constant(f, Rational::from((7, 3)))
            .divisor()
            .unwrap();
        assert!(d.is_zero());
    }

    #[test]
    fn genus_two_curve_divisor_of_y() {
        // y² = x⁵ + 1.  x⁵ + 1 = (x+1)(x⁴ − x³ + x² − x + 1), and the quartic
        // has no rational root, so div(y) cannot be written over rational
        // places and must refuse rather than report a degree-2 divisor.
        let err = FunctionFieldElement::y(c2()).divisor().unwrap_err();
        assert_eq!(err.code(), "E-FFLD-003");
    }

    #[test]
    fn genus_two_curve_divisor_of_x() {
        // div(x) on y² = x⁵ + 1: a(0) = 1 = 1², so the fibre is (0, ±1).
        let f = c2();
        let d = FunctionFieldElement::x(f).divisor().unwrap();
        assert_eq!(d.coefficient(&pl(0, 1)), Integer::from(1));
        assert_eq!(d.coefficient(&pl(0, -1)), Integer::from(1));
        assert_eq!(d.coefficient(&Place::Infinity), Integer::from(-2));
    }

    #[test]
    fn the_zero_function_has_no_divisor() {
        let f = e1();
        let z = FunctionFieldElement::constant(f, Rational::from(0));
        assert_eq!(z.divisor().unwrap_err().code(), "E-FFLD-008");
    }

    #[test]
    fn a_zero_denominator_is_refused() {
        let f = e1();
        let err = FunctionFieldElement::new(f, qp(&[1]), Vec::new(), Vec::new()).unwrap_err();
        assert_eq!(err.code(), "E-FFLD-001");
    }

    #[test]
    fn divisor_on_a_real_model_is_refused() {
        let f = FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 1]).unwrap();
        let err = FunctionFieldElement::x(f).divisor().unwrap_err();
        assert_eq!(err.code(), "E-FFLD-002");
    }

    #[test]
    fn every_divisor_of_a_function_has_degree_zero() {
        let f = e1();
        let cases = [
            (qp(&[0, 1]), Vec::new()),
            (Vec::new(), qp(&[1])),
            (qp(&[-1, 1]), Vec::new()),
            (qp(&[0, 1]), qp(&[1])),
        ];
        for (p, q) in cases {
            let u = FunctionFieldElement::polynomial(f.clone(), p, q);
            if let Ok(d) = u.divisor() {
                assert_eq!(
                    d.degree(),
                    Integer::from(0),
                    "div({u:?}) has non-zero degree"
                );
            }
        }
    }

    #[test]
    fn to_expr_round_trips_through_the_pool() {
        use crate::kernel::{Domain, ExprPool};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = e1();
        let u = FunctionFieldElement::new(f, qp(&[1]), qp(&[1]), qp(&[0, 1])).unwrap();
        let e = u.to_expr(x, &pool);
        let s = pool.display(e).to_string();
        assert!(
            s.contains("sqrt"),
            "rendered element should show the radical: {s}"
        );
    }
}
