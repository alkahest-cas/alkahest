//! V2-7 — Polynomial factorization over ℤ, 𝔽_p, and multivariate ℤ[𝑥₁,…].
//!
//! Univariate ℤ\[x\] uses FLINT `fmpz_poly_factor` (modular Berlekamp, Zassenhaus
//! recombination, van Hoeij’s knapsack–LLL).  Multivariate ℤ[x₁,…] uses
//! `fmpz_mpoly_factor` (Bernardin–Monagan EEZ pipeline).  Word-sized primes
//! use `nmod_poly_factor` (Berlekamp / Cantor–Zassenhaus / Kaltofen–Shoup per
//! FLINT’s internal choice).

use super::error::FactorError;
use super::multipoly::{multi_to_flint_pub, MultiPoly};
use super::unipoly::UniPoly;
use crate::flint::mpoly::{FlintMPolyCtx, FlintMPolyFactor};
use crate::flint::nmod::{FlintNmodPoly, FlintNmodPolyFactor};
use crate::flint::FlintPoly;
use crate::kernel::ExprId;
use std::sync::Arc;

/// Factors of a non-zero `UniPoly`: `polynomial = unit · ∏ baseᵢ^expᵢ`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniPolyFactorization {
    pub unit: rug::Integer,
    pub factors: Vec<(UniPoly, u32)>,
}

/// Factors of a non-zero multivariate integer polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiPolyFactorization {
    pub unit: rug::Integer,
    pub factors: Vec<(MultiPoly, u32)>,
}

/// Factors of a univariate polynomial over ℤ/pℤ (coefficients ascending, reduced mod p).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniPolyFactorModP {
    pub modulus: u64,
    pub factors: Vec<(Vec<u64>, u32)>,
}

impl UniPolyFactorization {
    /// Expand the factorization back to a single `UniPoly` (verification helper).
    pub fn expand_with_var(&self, var: ExprId) -> UniPoly {
        let mut acc = FlintPoly::from_rug_coefficients(std::slice::from_ref(&self.unit));
        for (f, e) in &self.factors {
            let powed = f.coeffs.pow(*e);
            acc = &acc * &powed;
        }
        UniPoly { var, coeffs: acc }
    }

    /// Check exactly that this factorization reconstructs `original`.
    ///
    /// This verifies the represented product over the integer coefficient
    /// ring. It does not establish irreducibility of the returned factors.
    pub fn verifies_product(&self, original: &UniPoly) -> bool {
        self.expand_with_var(original.var) == *original
    }
}

impl MultiPolyFactorization {
    /// Expand the factorization (verification helper).
    pub fn expand_clone_vars(&self) -> MultiPoly {
        let vars = self
            .factors
            .first()
            .map(|(m, _)| m.vars.clone())
            .unwrap_or_default();
        let mut terms = std::collections::BTreeMap::new();
        terms.insert(vec![], self.unit.clone());
        let mut acc = MultiPoly { vars, terms };
        for (f, e) in &self.factors {
            let mut powered = MultiPoly::constant(f.vars.clone(), 1);
            for _ in 0..*e {
                powered = powered * f.clone();
            }
            acc = acc * powered;
        }
        acc
    }

    /// Check exactly that this factorization reconstructs `original`.
    ///
    /// The variable list comes from `original` so constant multivariate
    /// factorizations retain their ambient polynomial ring.
    pub fn verifies_product(&self, original: &MultiPoly) -> bool {
        let mut terms = std::collections::BTreeMap::new();
        terms.insert(vec![], self.unit.clone());
        let mut acc = MultiPoly {
            vars: original.vars.clone(),
            terms,
        };
        for (factor, exponent) in &self.factors {
            if factor.vars != original.vars {
                return false;
            }
            let mut powered = MultiPoly::constant(original.vars.clone(), 1);
            for _ in 0..*exponent {
                powered = powered * factor.clone();
            }
            acc = acc * powered;
        }
        acc == *original
    }
}

/// [`UniPoly::factor_z`](UniPoly::factor_z).
pub fn factor_univariate_z(p: &UniPoly) -> Result<UniPolyFactorization, FactorError> {
    p.factor_z()
}

/// [`MultiPoly::factor_z`](MultiPoly::factor_z).
pub fn factor_multivariate_z(p: &MultiPoly) -> Result<MultiPolyFactorization, FactorError> {
    p.factor_z()
}

impl UniPoly {
    /// Factor over ℤ using FLINT (`fmpz_poly_factor`).
    pub fn factor_z(&self) -> Result<UniPolyFactorization, FactorError> {
        if self.is_zero() {
            return Err(FactorError::ZeroPolynomial);
        }
        let (unit, facs) = self
            .coeffs
            .factor_over_z()
            .map_err(|()| FactorError::FlintFailure)?;
        let factors: Vec<_> = facs
            .into_iter()
            .map(|(c, e)| {
                (
                    UniPoly {
                        var: self.var,
                        coeffs: c,
                    },
                    e,
                )
            })
            .collect();
        Ok(UniPolyFactorization {
            unit: unit.to_rug(),
            factors,
        })
    }
}

impl MultiPoly {
    /// Factor over ℤ[𝑥₁,…] using FLINT `fmpz_mpoly_factor`.
    pub fn factor_z(&self) -> Result<MultiPolyFactorization, FactorError> {
        if self.is_zero() {
            return Err(FactorError::ZeroPolynomial);
        }
        let nvars = self.vars.len().max(1);
        // Arc-shared context: FlintMPoly and FlintMPolyFactor both clone it,
        // so their Drop impls can call the matching FLINT clear functions.
        let ctx = FlintMPolyCtx::new(nvars);
        let a = multi_to_flint_pub(self, Arc::clone(&ctx));

        let mut fac = FlintMPolyFactor::new(Arc::clone(&ctx));
        if !fac.factor(&a) {
            return Err(FactorError::FlintFailure);
        }
        if !fac.constant_den_is_one() {
            return Err(FactorError::FlintFailure);
        }

        let unit = fac.unit().to_rug();
        let mut factors = Vec::with_capacity(fac.len());
        for i in 0..fac.len() {
            let base = fac.base_at(i);
            let terms = base.terms();
            let mp = MultiPoly {
                vars: self.vars.clone(),
                terms,
            };
            let exp = fac.exp_at(i);
            factors.push((mp, exp));
        }
        Ok(MultiPolyFactorization { unit, factors })
    }
}

/// Reduce coefficients mod `p` (must satisfy 2 ≤ p ≤ 2⁶³) and factor over 𝔽_p.
pub fn factor_univariate_mod_p(
    coeffs: &[i64],
    modulus: u64,
) -> Result<UniPolyFactorModP, FactorError> {
    if modulus < 2 {
        return Err(FactorError::InvalidModulus);
    }
    let p = modulus as i128;

    // FlintNmodPoly and FlintNmodPolyFactor are drop-safe: no manual
    // nmod_poly_clear / nmod_poly_factor_clear needed.
    let mut poly = FlintNmodPoly::new(modulus);
    for (i, &c) in coeffs.iter().enumerate() {
        let r = ((c as i128 % p) + p) % p;
        poly.set_coeff(i, r as u64);
    }

    let mut fac = FlintNmodPolyFactor::new();
    fac.factor(&poly);

    let factors = (0..fac.len())
        .map(|i| {
            let z = fac.poly_at(modulus, i);
            let deg = z.degree();
            let vc = (0..=deg).map(|j| z.get_coeff(j)).collect::<Vec<_>>();
            (vc, fac.exp_at(i))
        })
        .collect();

    Ok(UniPolyFactorModP { modulus, factors })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn univariate_x_squared_minus_one() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let p = UniPoly::from_symbolic(e, x, &pool).unwrap();
        let fac = p.factor_z().unwrap();
        assert_eq!(fac.factors.len(), 2);
        let prod = fac.expand_with_var(x);
        assert_eq!(prod, p);
        assert!(fac.verifies_product(&p));
    }

    #[test]
    fn swinnerton_dyer_irreducible_degree() {
        let fp = FlintPoly::swinnerton_dyer(5);
        assert_eq!(fp.degree(), 32);
        let fac = fp.factor_over_z().unwrap();
        assert_eq!(fac.1.len(), 1);
        assert_eq!(fac.1[0].1, 1);
    }

    #[test]
    fn cyclotomic_105_mod2_splits() {
        let phi = FlintPoly::cyclotomic(105);
        let coeffs: Vec<i64> = (0..phi.length())
            .map(|i| {
                (phi.get_coeff_flint(i).to_rug() % 2i32)
                    .to_i64()
                    .expect("coeff mod 2")
            })
            .collect();
        let fac = factor_univariate_mod_p(&coeffs, 2).unwrap();
        let deg_total: i64 = fac
            .factors
            .iter()
            .map(|(f, e)| (f.len() as i64 - 1) * (*e as i64))
            .sum();
        assert_eq!(deg_total, phi.degree());
        assert!(
            fac.factors.len() >= 2,
            "Φ_105 should have multiple factors over GF(2)"
        );
    }

    #[test]
    fn multivariate_product_recovered() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let vars = vec![x, y];
        let f1 = MultiPoly::from_symbolic(
            pool.add(vec![
                pool.pow(x, pool.integer(2_i32)),
                pool.pow(y, pool.integer(2_i32)),
                pool.integer(-1_i32),
            ]),
            vars.clone(),
            &pool,
        )
        .unwrap();
        let x_minus_y = pool.add(vec![x, pool.mul(vec![pool.integer(-1i32), y])]);
        let f2 = MultiPoly::from_symbolic(x_minus_y, vars.clone(), &pool).unwrap();
        let product = f1.clone() * f2.clone();
        let fac = product.factor_z().unwrap();
        let expanded = fac.expand_clone_vars();
        assert_eq!(expanded, product);
        assert!(fac.verifies_product(&product));
    }
}
