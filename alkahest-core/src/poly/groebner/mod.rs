//! Gröbner basis computation over ℚ — Buchberger algorithm
//! ([`buchberger::compute_buchberger_basis`]), Faugère F5
//! ([`compute_groebner_basis_f5`]), and the grevlex-then-FGLM strategy
//! ([`GroebnerBasis::compute_lex`]).
//!
//! # Quick start
//!
//! ```rust,ignore
//! use alkahest_cas::poly::groebner::{GroebnerBasis, MonomialOrder};
//! use alkahest_cas::poly::groebner::ideal::GbPoly;
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
pub mod buchberger;
pub mod f5;
pub mod fglm;
pub mod ideal;
pub mod monomial_order;
pub mod reduce;

// Keep f4 as a thin re-export so any external crate that depended on the old
// module path still compiles (semver-minor compat shim).
pub mod f4 {
    pub use super::buchberger::compute_buchberger_basis as compute_groebner_basis;
}

#[cfg(feature = "groebner-cuda")]
pub use cuda::{compute_groebner_basis_gpu, GpuGroebnerError, MacaulayMatrix};
pub use buchberger::compute_buchberger_basis;
pub use f5::compute_groebner_basis_f5;
pub use fglm::{fglm, grevlex_staircase, is_zero_dimensional};
pub use ideal::GbPoly;
pub use monomial_order::MonomialOrder;
pub use reduce::reduce;

/// A computed Gröbner basis.
#[derive(Clone, Debug)]
pub struct GroebnerBasis {
    generators: Vec<GbPoly>,
    order: MonomialOrder,
}

impl GroebnerBasis {
    /// Compute a Gröbner basis for the given generators under the given order.
    pub fn compute(gens: Vec<GbPoly>, order: MonomialOrder) -> Self {
        let generators = compute_buchberger_basis(gens, order);
        GroebnerBasis { generators, order }
    }

    /// Compute a lex Gröbner basis using the grevlex-then-FGLM strategy.
    ///
    /// For 0-dimensional ideals this is typically orders of magnitude faster
    /// than direct lex Buchberger. Falls back to direct lex Buchberger when
    /// the ideal is positive-dimensional or FGLM fails.
    pub fn compute_lex(gens: Vec<GbPoly>) -> Self {
        let n_vars = gens.first().map(|g| g.n_vars).unwrap_or(0);

        if n_vars <= 1 {
            return Self::compute(gens, MonomialOrder::Lex);
        }

        // Step 1: GRevLex basis (fast).
        let grb = compute_buchberger_basis(gens.clone(), MonomialOrder::GRevLex);

        // Step 2: Try FGLM.
        if is_zero_dimensional(&grb, n_vars) {
            if let Some(generators) = fglm(&grb, n_vars) {
                return GroebnerBasis {
                    generators,
                    order: MonomialOrder::Lex,
                };
            }
        }

        // Fallback: direct lex Buchberger.
        Self::compute(gens, MonomialOrder::Lex)
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
        for term in elim.generators()[0].terms.keys() {
            assert_eq!(term[0], 0, "eliminated variable x must not appear");
        }
    }

    #[test]
    fn compute_lex_circle_parabola() {
        // Same system as fglm.rs test — verify the public API path.
        let f = GbPoly {
            terms: [
                (vec![2u32, 0], rug::Rational::from(1)),
                (vec![0, 2], rug::Rational::from(1)),
                (vec![0, 0], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let g = GbPoly {
            terms: [
                (vec![0u32, 1], rug::Rational::from(1)),
                (vec![2, 0], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let gb_lex = GroebnerBasis::compute_lex(vec![f.clone(), g.clone()]);
        let gb_direct = GroebnerBasis::compute(vec![f, g], MonomialOrder::Lex);
        for p in gb_direct.generators() {
            assert!(gb_lex.contains(p), "FGLM basis missing generator");
        }
        for p in gb_lex.generators() {
            assert!(gb_direct.contains(p), "direct basis missing FGLM generator");
        }
    }
}
