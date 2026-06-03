use super::error::ConversionError;
use super::multipoly::MultiPoly;
use crate::flint::{FlintInteger, FlintPoly};
use crate::kernel::{ExprId, ExprPool};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Rational function over ℤ: numer / denom where both are `MultiPoly`.
///
/// Invariant (maintained by `new`):
/// 1. `denom` is not the zero polynomial.
/// 2. Both numerator and denominator are divided by their combined integer
///    content GCD so that no common integer factor remains.
/// 3. The leading term (lexicographically last key in `denom.terms`) has a
///    positive leading coefficient.
#[derive(Clone, PartialEq, Eq)]
pub struct RationalFunction {
    pub numer: MultiPoly,
    pub denom: MultiPoly,
}

impl RationalFunction {
    /// Construct and normalise a rational function.
    ///
    /// Returns `Err(ZeroDenominator)` if `denom` is the zero polynomial.
    pub fn new(numer: MultiPoly, denom: MultiPoly) -> Result<Self, ConversionError> {
        if denom.is_zero() {
            return Err(ConversionError::ZeroDenominator);
        }

        // Divide both by the GCD of all coefficients across numerator and denominator.
        let gcd_n = numer.integer_content();
        let gcd_d = denom.integer_content();
        let combined_gcd = if gcd_n == 0 {
            gcd_d.clone()
        } else if gcd_d == 0 {
            gcd_n.clone()
        } else {
            rug::Integer::from(gcd_n.gcd_ref(&gcd_d))
        };

        let (mut n, mut d) = if combined_gcd > 1 {
            (
                numer.div_integer(&combined_gcd),
                denom.div_integer(&combined_gcd),
            )
        } else {
            (numer, denom)
        };

        // Polynomial GCD reduction.
        // Try multivariate GCD first (works for both univariate and multivariate).
        // Fall back to the univariate FLINT path for robustness.
        let reduced = n.gcd(&d).and_then(|g| {
            // Only reduce if GCD is non-trivial (degree ≥ 1 or non-unit constant)
            let is_unit =
                g.terms.len() == 1 && g.terms.get(&vec![]).is_some_and(|c| *c == 1 || *c == -1);
            if is_unit {
                return None;
            }
            // Exact division: n / g and d / g
            // We use the FLINT mpoly divides path via a helper
            Some(g)
        });
        if let Some(ref g) = reduced {
            if let (Some(qn), Some(qd)) = (mpoly_exact_div(&n, g), mpoly_exact_div(&d, g)) {
                n = qn;
                d = qd;
            } else {
                // GCD divide failed; fall through to univariate path
                if let (Some(fp_n), Some(fp_d)) = (to_flintpoly(&n), to_flintpoly(&d)) {
                    let g = fp_n.gcd(&fp_d);
                    if g.degree() > 0 {
                        let q_n = fp_n.div_exact(&g);
                        let q_d = fp_d.div_exact(&g);
                        n = from_flintpoly(&q_n, n.vars.clone());
                        d = from_flintpoly(&q_d, d.vars.clone());
                    }
                }
            }
        } else if let (Some(fp_n), Some(fp_d)) = (to_flintpoly(&n), to_flintpoly(&d)) {
            let g = fp_n.gcd(&fp_d);
            if g.degree() > 0 {
                let q_n = fp_n.div_exact(&g);
                let q_d = fp_d.div_exact(&g);
                n = from_flintpoly(&q_n, n.vars.clone());
                d = from_flintpoly(&q_d, d.vars.clone());
            }
        }

        // Ensure leading coefficient of denominator is positive.
        // "Leading" = lexicographically last exponent key (highest total degree,
        // then highest variable index).
        if let Some((_, lc)) = d.terms.iter().next_back() {
            if *lc < 0 {
                n = -n;
                d = -d;
            }
        }

        Ok(RationalFunction { numer: n, denom: d })
    }

    /// Construct from two symbolic expressions.
    pub fn from_symbolic(
        numer_expr: ExprId,
        denom_expr: ExprId,
        vars: Vec<ExprId>,
        pool: &ExprPool,
    ) -> Result<Self, ConversionError> {
        let n = MultiPoly::from_symbolic(numer_expr, vars.clone(), pool)?;
        let d = MultiPoly::from_symbolic(denom_expr, vars, pool)?;
        Self::new(n, d)
    }

    pub fn is_zero(&self) -> bool {
        self.numer.is_zero()
    }
}

// ---------------------------------------------------------------------------
// Arithmetic operators
// ---------------------------------------------------------------------------

impl Neg for RationalFunction {
    type Output = Self;
    fn neg(self) -> Self {
        RationalFunction {
            numer: -self.numer,
            denom: self.denom,
        }
    }
}

impl Add for RationalFunction {
    type Output = Result<Self, ConversionError>;
    fn add(self, rhs: Self) -> Result<Self, ConversionError> {
        // a/b + c/d = (a*d + c*b) / (b*d)
        let ad = self.numer.clone() * rhs.denom.clone();
        let cb = rhs.numer.clone() * self.denom.clone();
        let numer = ad + cb;
        let denom = self.denom * rhs.denom;
        RationalFunction::new(numer, denom)
    }
}

impl Sub for RationalFunction {
    type Output = Result<Self, ConversionError>;
    fn sub(self, rhs: Self) -> Result<Self, ConversionError> {
        self.add(-rhs)
    }
}

impl Mul for RationalFunction {
    type Output = Result<Self, ConversionError>;
    fn mul(self, rhs: Self) -> Result<Self, ConversionError> {
        // (a/b) * (c/d) = (a*c) / (b*d)
        let numer = self.numer * rhs.numer;
        let denom = self.denom * rhs.denom;
        RationalFunction::new(numer, denom)
    }
}

impl Div for RationalFunction {
    type Output = Result<Self, ConversionError>;
    fn div(self, rhs: Self) -> Result<Self, ConversionError> {
        if rhs.is_zero() {
            return Err(ConversionError::ZeroDenominator);
        }
        // (a/b) / (c/d) = (a*d) / (b*c)
        let numer = self.numer * rhs.denom;
        let denom = self.denom * rhs.numer;
        RationalFunction::new(numer, denom)
    }
}

// ---------------------------------------------------------------------------
// Univariate conversion helpers for GCD reduction
// ---------------------------------------------------------------------------

/// Convert a MultiPoly to FlintPoly if it is effectively univariate
/// (all exponent vectors have length ≤ 1, i.e. only the first variable appears).
/// Returns `None` for multivariate or zero polynomials.
fn to_flintpoly(p: &MultiPoly) -> Option<FlintPoly> {
    if p.terms.keys().any(|exp| exp.len() > 1) {
        return None;
    }
    let mut fp = FlintPoly::new();
    for (exp, coeff) in &p.terms {
        let deg = exp.first().copied().unwrap_or(0) as usize;
        let fi = FlintInteger::from_rug(coeff);
        fp.set_coeff_flint(deg, &fi);
    }
    Some(fp)
}

/// Exact division of multivariate polynomials: `a / b` assuming `b | a`.
/// Returns `None` if the division is not exact or if FLINT fails.
fn mpoly_exact_div(a: &MultiPoly, b: &MultiPoly) -> Option<MultiPoly> {
    use crate::flint::mpoly::FlintMPolyCtx;
    use std::sync::Arc;
    let nvars = a.vars.len().max(1);
    let ctx = FlintMPolyCtx::new(nvars);

    let fa = super::multipoly::multi_to_flint_pub(a, Arc::clone(&ctx));
    let fb = super::multipoly::multi_to_flint_pub(b, Arc::clone(&ctx));

    // FlintMPoly::divides calls fmpz_mpoly_divides internally where ctx.as_ptr() is accessible.
    let q = fa.divides(&fb)?;
    let terms = q.terms();
    Some(MultiPoly {
        vars: a.vars.clone(),
        terms,
    })
}

/// Convert a univariate FlintPoly back to MultiPoly using the given variable list.
fn from_flintpoly(fp: &FlintPoly, vars: Vec<ExprId>) -> MultiPoly {
    let mut terms: BTreeMap<Vec<u32>, rug::Integer> = BTreeMap::new();
    for i in 0..fp.length() {
        let fi = fp.get_coeff_flint(i);
        let r = fi.to_rug();
        if r != 0 {
            let exp = if i == 0 { vec![] } else { vec![i as u32] };
            terms.insert(exp, r);
        }
    }
    MultiPoly { vars, terms }
}

impl RationalFunction {
    /// Pretty-print using symbol names from *pool*.
    pub fn display_with(&self, pool: &ExprPool) -> String {
        let d_is_one =
            self.denom.terms.len() == 1 && self.denom.terms.get(&vec![]).is_some_and(|c| *c == 1);
        if d_is_one {
            self.numer.display_with(pool)
        } else {
            format!(
                "({}) / ({})",
                self.numer.display_with(pool),
                self.denom.display_with(pool)
            )
        }
    }
}

impl fmt::Display for RationalFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d_is_one =
            self.denom.terms.len() == 1 && self.denom.terms.get(&vec![]).is_some_and(|c| *c == 1);
        if d_is_one {
            write!(f, "{}", self.numer)
        } else {
            write!(f, "({}) / ({})", self.numer, self.denom)
        }
    }
}

impl fmt::Debug for RationalFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RationalFunction({:?} / {:?})", self.numer, self.denom)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn pool_xy() -> (ExprPool, ExprId, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        (p, x, y)
    }

    #[test]
    fn zero_denominator_error() {
        let (p, x, y) = pool_xy();
        let n = MultiPoly::from_symbolic(x, vec![x, y], &p).unwrap();
        let z = MultiPoly::zero(vec![x, y]);
        assert!(matches!(
            RationalFunction::new(n, z),
            Err(ConversionError::ZeroDenominator)
        ));
    }

    #[test]
    fn integer_content_normalisation() {
        // (6x) / (4) → (3x) / (2)
        let (p, x, y) = pool_xy();
        let n_expr = p.mul(vec![p.integer(6_i32), x]);
        let d_expr = p.integer(4_i32);
        let rf = RationalFunction::from_symbolic(n_expr, d_expr, vec![x, y], &p).unwrap();
        assert_eq!(
            rf.numer.terms[&vec![1]],
            rug::Integer::from(3),
            "numerator coefficient should be 3"
        );
        assert_eq!(
            rf.denom.terms[&vec![]],
            rug::Integer::from(2),
            "denominator constant should be 2"
        );
    }

    #[test]
    fn positive_leading_coeff_normalisation() {
        // x / (-2) → (-x) / 2
        let (p, x, y) = pool_xy();
        let n_expr = x;
        let d_expr = p.integer(-2_i32);
        let rf = RationalFunction::from_symbolic(n_expr, d_expr, vec![x, y], &p).unwrap();
        // denominator leading coefficient must be positive
        let lc = rf.denom.terms.values().next_back().unwrap();
        assert!(*lc > 0, "leading coeff of denominator should be positive");
    }

    #[test]
    fn polynomial_gcd_reduces_common_factor() {
        // (x+1) / (x+1) → 1 / 1
        let (p, x, y) = pool_xy();
        let xp1 = p.add(vec![x, p.integer(1_i32)]);
        let rf = RationalFunction::from_symbolic(xp1, xp1, vec![x, y], &p).unwrap();
        assert!(
            rf.numer.terms.len() == 1 && rf.numer.terms.get(&vec![]).is_some_and(|c| *c == 1),
            "numerator should be 1, got {:?}",
            rf.numer
        );
        assert!(
            rf.denom.terms.len() == 1 && rf.denom.terms.get(&vec![]).is_some_and(|c| *c == 1),
            "denominator should be 1, got {:?}",
            rf.denom
        );
    }

    #[test]
    fn polynomial_gcd_partial_factor() {
        // (x^2 - 1) / (x - 1) = (x+1) / 1
        let (p, x, y) = pool_xy();
        let xsq_m1 = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(-1_i32)]);
        let x_m1 = p.add(vec![x, p.integer(-1_i32)]);
        let rf = RationalFunction::from_symbolic(xsq_m1, x_m1, vec![x, y], &p).unwrap();
        // denominator should reduce to 1
        assert!(
            rf.denom.terms.len() == 1 && rf.denom.terms.get(&vec![]).is_some_and(|c| *c == 1),
            "denominator should be 1, got {:?}",
            rf.denom
        );
        // numerator should be (x + 1)
        assert_eq!(
            rf.numer.terms.get(&vec![1]).cloned(),
            Some(rug::Integer::from(1))
        );
        assert_eq!(
            rf.numer.terms.get(&vec![]).cloned(),
            Some(rug::Integer::from(1))
        );
    }

    #[test]
    fn trivial_rational() {
        // x / 1 → displayed without denominator
        let (p, x, y) = pool_xy();
        let n = MultiPoly::from_symbolic(x, vec![x, y], &p).unwrap();
        let d = MultiPoly::constant(vec![x, y], 1);
        let rf = RationalFunction::new(n, d).unwrap();
        let s = rf.to_string();
        assert!(
            !s.contains('/'),
            "should not show '/' for denominator 1: {s}"
        );
    }
}
