//! Terms, monomials and polynomials for Gröbner basis computation over ℚ.
//!
//! We use a sparse representation: a polynomial is a `BTreeMap` from exponent
//! vector to rational coefficient.

use crate::poly::groebner::monomial_order::MonomialOrder;
use std::collections::BTreeMap;

/// A term: (exponent vector, rational coefficient).
pub type Term = (Vec<u32>, rug::Rational);

/// A polynomial suitable for Gröbner basis computation.
///
/// Stored as a map from exponent vector to coefficient.
/// The number of variables is fixed at construction.
#[derive(Debug, Clone)]
pub struct GbPoly {
    /// Coefficients keyed by exponent vector.
    pub terms: BTreeMap<Vec<u32>, rug::Rational>,
    /// Number of variables.
    pub n_vars: usize,
}

impl GbPoly {
    /// The zero polynomial in `n_vars` variables.
    pub fn zero(n_vars: usize) -> Self {
        GbPoly {
            terms: BTreeMap::new(),
            n_vars,
        }
    }

    /// A constant polynomial.
    pub fn constant(val: rug::Rational, n_vars: usize) -> Self {
        let mut terms = BTreeMap::new();
        if val != 0 {
            terms.insert(vec![0u32; n_vars], val);
        }
        GbPoly { terms, n_vars }
    }

    /// A monomial with given exponent vector and coefficient.
    pub fn monomial(exp: Vec<u32>, coeff: rug::Rational) -> Self {
        let n_vars = exp.len();
        let mut terms = BTreeMap::new();
        if coeff != 0 {
            terms.insert(exp, coeff);
        }
        GbPoly { terms, n_vars }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Leading term under the given order (highest monomial).
    pub fn leading_term(&self, order: MonomialOrder) -> Option<(&Vec<u32>, &rug::Rational)> {
        self.terms
            .iter()
            .max_by(|(ea, _), (eb, _)| order.cmp(ea, eb))
    }

    /// Leading monomial exponent.
    pub fn leading_exp(&self, order: MonomialOrder) -> Option<Vec<u32>> {
        self.leading_term(order).map(|(e, _)| e.clone())
    }

    /// Leading coefficient.
    pub fn leading_coeff(&self, order: MonomialOrder) -> Option<rug::Rational> {
        self.leading_term(order).map(|(_, c)| c.clone())
    }

    /// Multiply by a scalar.
    pub fn scale(&self, s: &rug::Rational) -> Self {
        if *s == 0 {
            return GbPoly::zero(self.n_vars);
        }
        GbPoly {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), rug::Rational::from(c * s)))
                .collect(),
            n_vars: self.n_vars,
        }
    }

    /// Negate.
    pub fn neg(&self) -> Self {
        self.scale(&rug::Rational::from(-1))
    }

    /// Add two polynomials.
    pub fn add(&self, other: &GbPoly) -> GbPoly {
        let mut terms = self.terms.clone();
        for (e, c) in &other.terms {
            let entry = terms
                .entry(e.clone())
                .or_insert_with(|| rug::Rational::from(0));
            *entry += c;
            if *entry == 0 {
                terms.remove(e);
            }
        }
        GbPoly {
            terms,
            n_vars: self.n_vars,
        }
    }

    /// Subtract.
    pub fn sub(&self, other: &GbPoly) -> GbPoly {
        self.add(&other.neg())
    }

    /// Multiply two polynomials.
    pub fn mul(&self, other: &GbPoly) -> GbPoly {
        let mut result = GbPoly::zero(self.n_vars);
        for (ea, ca) in &self.terms {
            for (eb, cb) in &other.terms {
                let e: Vec<u32> = ea.iter().zip(eb.iter()).map(|(a, b)| a + b).collect();
                let c = rug::Rational::from(ca * cb);
                let entry = result
                    .terms
                    .entry(e)
                    .or_insert_with(|| rug::Rational::from(0));
                *entry += &c;
            }
        }
        result.terms.retain(|_, v| *v != 0);
        result
    }

    /// Multiply by a monomial (shift exponents by `exp_shift`, scale by `coeff`).
    pub fn mul_monomial(&self, exp_shift: &[u32], coeff: &rug::Rational) -> GbPoly {
        if *coeff == 0 {
            return GbPoly::zero(self.n_vars);
        }
        GbPoly {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| {
                    let new_e: Vec<u32> =
                        e.iter().zip(exp_shift.iter()).map(|(a, b)| a + b).collect();
                    (new_e, rug::Rational::from(c * coeff))
                })
                .collect(),
            n_vars: self.n_vars,
        }
    }

    /// Sugar of the polynomial: max total degree over all terms.
    ///
    /// Used for the sugar selection strategy in Buchberger's algorithm.
    /// For a zero polynomial returns 0.
    #[inline]
    pub fn sugar(&self) -> u32 {
        self.terms
            .keys()
            .map(|e| e.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// Make monic under the given order (leading coeff → 1).
    pub fn make_monic(&self, order: MonomialOrder) -> Self {
        if let Some(lc) = self.leading_coeff(order) {
            let inv_lc = rug::Rational::from(1) / lc;
            self.scale(&inv_lc)
        } else {
            self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::groebner::monomial_order::MonomialOrder;

    fn rat(n: i64, d: i64) -> rug::Rational {
        rug::Rational::from((n, d))
    }

    #[test]
    fn zero_poly() {
        let z = GbPoly::zero(2);
        assert!(z.is_zero());
        assert!(z.leading_term(MonomialOrder::Lex).is_none());
    }

    #[test]
    fn add_polys() {
        // x + (-x) = 0
        let p = GbPoly::monomial(vec![1, 0], rat(1, 1));
        let q = GbPoly::monomial(vec![1, 0], rat(-1, 1));
        let r = p.add(&q);
        assert!(r.is_zero());
    }

    #[test]
    fn mul_polys() {
        // (x + 1) * (x - 1) = x² - 1
        let p = GbPoly {
            terms: [(vec![1, 0], rat(1, 1)), (vec![0, 0], rat(1, 1))]
                .into_iter()
                .collect(),
            n_vars: 2,
        };
        let q = GbPoly {
            terms: [(vec![1, 0], rat(1, 1)), (vec![0, 0], rat(-1, 1))]
                .into_iter()
                .collect(),
            n_vars: 2,
        };
        let r = p.mul(&q);
        assert_eq!(r.terms.get(&vec![2, 0]), Some(&rat(1, 1)));
        assert_eq!(r.terms.get(&vec![0, 0]), Some(&rat(-1, 1)));
        assert!(r.terms.get(&vec![1, 0]).is_none());
    }
}
