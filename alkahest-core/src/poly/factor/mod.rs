//! V2-7 — Polynomial factorization over ℤ, 𝔽_p, and multivariate ℤ[𝑥₁,…].
//!
//! Univariate ℤ[x] uses FLINT `fmpz_poly_factor` (modular Berlekamp, Zassenhaus
//! recombination, van Hoeij’s knapsack–LLL).  Multivariate ℤ[x₁,…] uses
//! `fmpz_mpoly_factor` (Bernardin–Monagan EEZ pipeline).  Word-sized primes
//! use `nmod_poly_factor` (Berlekamp / Cantor–Zassenhaus / Kaltofen–Shoup per
//! FLINT’s internal choice).

use super::error::FactorError;
use super::multipoly::{multi_to_flint_pub, MultiPoly};
use super::unipoly::UniPoly;
use crate::flint::ffi::{
    self, FmpzMPolyFactorStruct, NmodPolyFactorStruct, NmodPolyStruct,
};
use crate::flint::integer::FlintInteger;
use crate::flint::mpoly::{FlintMPoly, FlintMPolyCtx};
use crate::flint::FlintPoly;
use crate::kernel::ExprId;

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
        let ctx = FlintMPolyCtx::new(nvars);
        let mut a = multi_to_flint_pub(self, &ctx);
        unsafe {
            let mut fac = std::mem::MaybeUninit::<FmpzMPolyFactorStruct>::uninit();
            ffi::fmpz_mpoly_factor_init(fac.as_mut_ptr(), ctx.as_ptr());
            let mut fac = fac.assume_init();
            let ok = ffi::fmpz_mpoly_factor(&mut fac, a.as_ptr(), ctx.as_ptr());
            if ok == 0 {
                ffi::fmpz_mpoly_factor_clear(&mut fac, ctx.as_ptr());
                a.clear_with_ctx(&ctx);
                return Err(FactorError::FlintFailure);
            }
            if ffi::fmpz_cmp_ui(std::ptr::addr_of!(fac.constant_den), 1) != 0 {
                ffi::fmpz_mpoly_factor_clear(&mut fac, ctx.as_ptr());
                a.clear_with_ctx(&ctx);
                return Err(FactorError::FlintFailure);
            }
            let mut unit = FlintInteger::new();
            ffi::fmpz_mpoly_factor_get_constant_fmpz(
                unit.inner_mut_ptr(),
                &fac,
                ctx.as_ptr(),
            );
            let n = ffi::fmpz_mpoly_factor_length(&fac, ctx.as_ptr());
            let mut factors = Vec::with_capacity(n as usize);
            for i in 0..n {
                let mut base = FlintMPoly::new(&ctx);
                ffi::fmpz_mpoly_factor_get_base(base.as_mut_ptr(), &fac, i, ctx.as_ptr());
                let terms = base.terms(nvars, &ctx);
                base.clear_with_ctx(&ctx);
                let mp = MultiPoly {
                    vars: self.vars.clone(),
                    terms,
                };
                let exp =
                    ffi::fmpz_mpoly_factor_get_exp_si(&mut fac, i, ctx.as_ptr()) as u32;
                factors.push((mp, exp));
            }
            ffi::fmpz_mpoly_factor_clear(&mut fac, ctx.as_ptr());
            a.clear_with_ctx(&ctx);
            Ok(MultiPolyFactorization {
                unit: unit.to_rug(),
                factors,
            })
        }
    }
}

/// Reduce coefficients mod `p` (must satisfy 2 ≤ p ≤ 2⁶³) and factor over 𝔽_p.
pub fn factor_univariate_mod_p(coeffs: &[i64], modulus: u64) -> Result<UniPolyFactorModP, FactorError> {
    if modulus < 2 {
        return Err(FactorError::InvalidModulus);
    }
    let p = modulus as i128;
    let mut poly: NmodPolyStruct = unsafe { std::mem::zeroed() };
    let factors = unsafe {
        ffi::nmod_poly_init(&mut poly, modulus);
        for (i, &c) in coeffs.iter().enumerate() {
            let r = (c as i128 % p + p) % p;
            ffi::nmod_poly_set_coeff_ui(&mut poly, i as ffi::slong, r as ffi::ulong);
        }
        let mut fac = std::mem::MaybeUninit::<NmodPolyFactorStruct>::uninit();
        ffi::nmod_poly_factor_init(fac.as_mut_ptr());
        let mut fac = fac.assume_init();
        ffi::nmod_poly_factor(&mut fac, &poly);
        let mut factors = Vec::with_capacity(fac.num as usize);
        for i in 0..fac.num {
            let mut z: NmodPolyStruct = std::mem::zeroed();
            ffi::nmod_poly_init(&mut z, modulus);
            ffi::nmod_poly_factor_get_nmod_poly(&mut z, &mut fac, i);
            let deg = ffi::nmod_poly_degree(&z);
            let mut vc = Vec::with_capacity(deg as usize + 1);
            for j in 0..=deg {
                vc.push(ffi::nmod_poly_get_coeff_ui(&z, j));
            }
            ffi::nmod_poly_clear(&mut z);
            let exp = *fac.exp.add(i as usize) as u32;
            factors.push((vc, exp));
        }
        ffi::nmod_poly_factor_clear(&mut fac);
        ffi::nmod_poly_clear(&mut poly);
        factors
    };
    Ok(UniPolyFactorModP {
        modulus,
        factors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn univariate_x_squared_minus_one() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.integer(-1_i32),
        ]);
        let p = UniPoly::from_symbolic(e, x, &pool).unwrap();
        let fac = p.factor_z().unwrap();
        assert_eq!(fac.factors.len(), 2);
        let prod = fac.expand_with_var(x);
        assert_eq!(prod, p);
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
    }
}
