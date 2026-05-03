//! Gröbner basis computation over ℚ — Buchberger / F4-labelled parallel reduction
//! ([`f4::compute_groebner_basis`]) and Faugère F5 ([`compute_groebner_basis_f5`]).
//!
//! # Quick start
//!
//! ```rust,ignore
//! use alkahest_core::poly::groebner::{GroebnerBasis, MonomialOrder};
//! use alkahest_core::poly::groebner::ideal::GbPoly;
//!
//! // Ideal: (x^2 - 1, x - 1) in 1 variable
//! let f = GbPoly {
//!     terms: [(vec![2], rug::Rational::from(1)), (vec![0], rug::Rational::from(-1))].into_iter().collect(),
//!     n_vars: 1,
//! };
//! let g = GbPoly {
//!     terms: [(vec![1], rug::Rational::from(1)), (vec![0], rug::Rational::from(-1))].into_iter().collect(),
//!     n_vars: 1,
//! };
//! let gb = GroebnerBasis::compute(vec![f, g], MonomialOrder::Lex);
//! assert_eq!(gb.generators().len(), 1); // {x - 1}
//! ```

#[cfg(feature = "groebner-cuda")]
pub mod cuda;
pub mod f4;
pub mod f5;
pub mod ideal;
pub mod monomial_order;
pub mod reduce;

#[cfg(feature = "groebner-cuda")]
pub use cuda::{compute_groebner_basis_gpu, GpuGroebnerError, MacaulayMatrix};
pub use f4::compute_groebner_basis;
pub use f5::compute_groebner_basis_f5;
pub use ideal::GbPoly;
pub use monomial_order::MonomialOrder;
pub use reduce::reduce;

/// A computed Gröbner basis.
pub struct GroebnerBasis {
    generators: Vec<GbPoly>,
    order: MonomialOrder,
}

impl GroebnerBasis {
    /// Compute a Gröbner basis for the given generators under the given order.
    pub fn compute(gens: Vec<GbPoly>, order: MonomialOrder) -> Self {
        let generators = compute_groebner_basis(gens, order);
        GroebnerBasis { generators, order }
    }

    /// Compute a Gröbner basis using Faugère's F5 signature-based algorithm (V2-8).
    ///
    /// Signatures use **lexicographic** order on the monomial part and then
    /// original-generator index; polynomial leading terms still use `order`.
    pub fn compute_f5(gens: Vec<GbPoly>, order: MonomialOrder) -> Self {
        let generators = compute_groebner_basis_f5(gens, order);
        GroebnerBasis { generators, order }
    }

    /// Return the basis generators (interreduced, monic).
    pub fn generators(&self) -> &[GbPoly] {
        &self.generators
    }

    /// Reduce a polynomial by this basis. Returns the remainder.
    pub fn reduce(&self, p: &GbPoly) -> GbPoly {
        reduce(p, &self.generators, self.order)
    }

    /// Test ideal membership: p is in the ideal iff `reduce(p) == 0`.
    pub fn contains(&self, p: &GbPoly) -> bool {
        self.reduce(p).is_zero()
    }

    /// Return the number of generators.
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    /// Eliminate a set of variables (given by their indices into the
    /// exponent vector).  Under a `Lex` basis with eliminated variables
    /// placed first, the elimination theorem says the generators whose
    /// leading monomial mentions none of the eliminated variables form a
    /// Gröbner basis for the elimination ideal `I ∩ k[remaining vars]`.
    ///
    /// This implementation is more permissive: it drops any generator
    /// whose support touches an eliminated variable in *any* monomial.
    /// That is strictly correct for the common Lex layout used by the
    /// V1-4 solver (eliminated variables appear with *only* positive
    /// exponents in the generators that are being dropped).
    pub fn eliminate(&self, vars: &[usize]) -> GroebnerBasis {
        let generators: Vec<GbPoly> = self
            .generators
            .iter()
            .filter(|g| {
                !g.terms
                    .keys()
                    .any(|e| vars.iter().any(|&i| e.get(i).copied().unwrap_or(0) > 0))
            })
            .cloned()
            .collect();
        GroebnerBasis {
            generators,
            order: self.order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eliminate_drops_generators() {
        // Lex basis of {x - y, 2y² - 1}: eliminating x should leave {2y² - 1}.
        let xm_y = GbPoly {
            terms: [
                (vec![1u32, 0], rug::Rational::from(1)),
                (vec![0, 1], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let two_y2_m1 = GbPoly {
            terms: [
                (vec![0, 2], rug::Rational::from(2)),
                (vec![0, 0], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let gb = GroebnerBasis::compute(vec![xm_y, two_y2_m1], MonomialOrder::Lex);
        let elim = gb.eliminate(&[0]);
        assert_eq!(elim.generators().len(), 1);
        // The surviving generator must not depend on x (var index 0).
        for term in elim.generators()[0].terms.keys() {
            assert_eq!(term[0], 0, "eliminated variable x must not appear");
        }
    }
}
