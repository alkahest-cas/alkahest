//! V2-7 — Polynomial factorization over ℤ, 𝔽_p, and multivariate ℤ[𝑥₁,…].
//!
//! Univariate ℤ\[x\] uses FLINT `fmpz_poly_factor` (modular Berlekamp, Zassenhaus
//! recombination, van Hoeij’s knapsack–LLL).  Multivariate ℤ[x₁,…] uses
//! `fmpz_mpoly_factor` (Bernardin–Monagan EEZ pipeline).  Word-sized primes
//! use `nmod_poly_factor` (Berlekamp / Cantor–Zassenhaus / Kaltofen–Shoup per
//! FLINT’s internal choice).

pub(crate) mod binomial;

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

// ---------------------------------------------------------------------------
// Canonical factor order
// ---------------------------------------------------------------------------
//
// FLINT lists factors in whatever order its recombination produced them: for
// `fmpz_poly_factor` that is decided by the row order of an LLL-reduced
// lattice, and it differs between FLINT 3.5 and 3.6 (`x^156 − 1`, `x^240 − 1`,
// …). Every factorisation this crate returns is therefore sorted into the
// order below, which depends on the factors alone.

/// The canonical order of univariate factors over ℤ: ascending degree, then
/// the coefficients compared as signed integers from the leading term down,
/// then ascending multiplicity.
///
/// `x^12 − 1` factors as `x − 1, x + 1, x² − x + 1, x² + 1, x² + x + 1,
/// x⁴ − x² + 1` (the order SymPy's `factor_list` uses for squarefree input).
/// A factor `x` sits between `x − 1` and `x + 1`.
pub fn cmp_univariate_factors(a: &(FlintPoly, u32), b: &(FlintPoly, u32)) -> std::cmp::Ordering {
    a.0.cmp_canonical(&b.0).then(a.1.cmp(&b.1))
}

/// The canonical order of factors over 𝔽_p (ascending coefficient vectors):
/// ascending degree, then the residues compared from the leading term down,
/// then ascending multiplicity.
pub fn cmp_mod_p_factors(a: &(Vec<u64>, u32), b: &(Vec<u64>, u32)) -> std::cmp::Ordering {
    a.0.len()
        .cmp(&b.0.len())
        .then_with(|| a.0.iter().rev().cmp(b.0.iter().rev()))
        .then(a.1.cmp(&b.1))
}

/// Compare exponent vectors lexicographically, a missing trailing entry
/// counting as 0 (`[1]` and `[1, 0]` are the same monomial).
fn cmp_exponents<T: Copy + Into<u64>>(a: &[T], b: &[T]) -> std::cmp::Ordering {
    let n = a.len().max(b.len());
    for i in 0..n {
        let x: u64 = a.get(i).map_or(0, |&v| v.into());
        let y: u64 = b.get(i).map_or(0, |&v| v.into());
        if x != y {
            return x.cmp(&y);
        }
    }
    std::cmp::Ordering::Equal
}

/// The terms of `m` with their monomials in descending lexicographic order.
fn terms_descending<T: Copy + Into<u64>>(
    m: &std::collections::BTreeMap<Vec<T>, rug::Integer>,
) -> Vec<(&[T], &rug::Integer)> {
    let mut v: Vec<(&[T], &rug::Integer)> = m.iter().map(|(e, c)| (e.as_slice(), c)).collect();
    v.sort_by(|x, y| cmp_exponents(y.0, x.0));
    v
}

/// The canonical order of multivariate factors (terms as exponent vector →
/// coefficient): ascending total degree, then the coefficients compared as
/// signed integers over the monomials in descending lexicographic order
/// (variables in ring order; a monomial absent from one side counts as a
/// zero coefficient there). Multiplicity is the caller's final tiebreak.
///
/// `x⁶ − y⁶` factors as `x − y, x + y, x² − x·y + y², x² + x·y + y²`.
pub fn cmp_multivariate_terms<T: Copy + Into<u64>>(
    a: &std::collections::BTreeMap<Vec<T>, rug::Integer>,
    b: &std::collections::BTreeMap<Vec<T>, rug::Integer>,
) -> std::cmp::Ordering {
    let total = |m: &std::collections::BTreeMap<Vec<T>, rug::Integer>| {
        m.keys()
            .map(|e| e.iter().map(|&v| u128::from(v.into())).sum::<u128>())
            .max()
            .unwrap_or(0)
    };
    let by_degree = total(a).cmp(&total(b));
    if by_degree.is_ne() {
        return by_degree;
    }
    let (ta, tb) = (terms_descending(a), terms_descending(b));
    let zero = rug::Integer::new();
    let (mut i, mut j) = (0, 0);
    while i < ta.len() || j < tb.len() {
        // The larger of the two current monomials; its coefficient on the
        // side that lacks it is zero.
        let (ca, cb) = match (ta.get(i), tb.get(j)) {
            (Some(x), Some(y)) => match cmp_exponents(x.0, y.0) {
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                    (x.1, y.1)
                }
                std::cmp::Ordering::Greater => {
                    i += 1;
                    (x.1, &zero)
                }
                std::cmp::Ordering::Less => {
                    j += 1;
                    (&zero, y.1)
                }
            },
            (Some(x), None) => {
                i += 1;
                (x.1, &zero)
            }
            (None, Some(y)) => {
                j += 1;
                (&zero, y.1)
            }
            (None, None) => unreachable!(),
        };
        let c = ca.cmp(cb);
        if c.is_ne() {
            return c;
        }
    }
    std::cmp::Ordering::Equal
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
    ///
    /// Factors are primitive with positive leading coefficients, the content
    /// and sign are in `unit`, and the list is in the canonical order of
    /// [`cmp_univariate_factors`] (ascending degree, then coefficients from
    /// the leading term down) — independent of the FLINT version. `x^n ± 1`
    /// and the binomials `u^n·x^n ± v^n` are factored through cyclotomic
    /// polynomials without calling FLINT's factoriser.
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
    ///
    /// The factors are in the canonical order of [`cmp_multivariate_terms`]
    /// (ascending total degree, then coefficients over descending lex
    /// monomials), then ascending multiplicity — independent of the FLINT
    /// version.
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
        for i in fac.canonical_order() {
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

/// Reduce coefficients mod `p` and factor over 𝔽_p.
///
/// The factors are listed in the canonical order of [`cmp_mod_p_factors`]
/// (ascending degree, then residues from the leading term down).
///
/// The factors are **monic**; the leading coefficient that makes their
/// product equal the input is dropped here and returned by
/// [`factor_univariate_mod_p_with_unit`].
///
/// # Errors
///
/// `E-POLY-009` unless `modulus` is a prime (a composite modulus is not a
/// field, and FLINT's factoriser aborts the process on the first
/// non-invertible leading coefficient it meets); `E-POLY-008` when the
/// polynomial is zero modulo `p`.
pub fn factor_univariate_mod_p(
    coeffs: &[i64],
    modulus: u64,
) -> Result<UniPolyFactorModP, FactorError> {
    factor_univariate_mod_p_with_unit(coeffs, modulus).map(|(_, fac)| fac)
}

/// [`factor_univariate_mod_p`], also returning the unit: the leading
/// coefficient `u` of the input reduced mod `p`, so that
/// `polynomial ≡ u · ∏ fᵢ^eᵢ (mod p)` with every `fᵢ` monic.
///
/// A non-zero constant factors as `(u, [])`, and `2x` mod 7 as
/// `(2, [([0, 1], 1)])`.
///
/// # Errors
///
/// As [`factor_univariate_mod_p`].
pub fn factor_univariate_mod_p_with_unit(
    coeffs: &[i64],
    modulus: u64,
) -> Result<(u64, UniPolyFactorModP), FactorError> {
    // `nmod_poly_factor` assumes a field: over ℤ/nℤ with n composite it meets
    // a leading coefficient with no inverse and calls `flint_throw`, which
    // aborts the process. Primality is the precondition, not a nicety.
    crate::budget::clear_trip();
    // SAFETY: `n_is_prime` is a pure function of a machine word.
    if modulus < 2 || unsafe { crate::flint::ffi::n_is_prime(modulus) } == 0 {
        return Err(FactorError::InvalidModulus);
    }
    let p = modulus as i128;
    let reduced: Vec<u64> = coeffs
        .iter()
        .map(|&c| (c as i128).rem_euclid(p) as u64)
        .collect();
    let Some(last) = reduced.iter().rposition(|&c| c != 0) else {
        // ≡ 0 mod p: no factorisation, exactly as over ℤ.
        return Err(FactorError::ZeroPolynomial);
    };
    let unit = reduced[last];

    // The factoriser works in several copies of the input; refuse up front a
    // size that would abort inside FLINT instead. The cause is left for the
    // bindings in `budget::take_trip` (FactorError is exhaustive).
    let words = (last as u64 + 1).saturating_mul(16);
    if let Err(trip) = crate::budget::preflight_bytes(words.saturating_mul(8)) {
        crate::budget::record_trip(trip);
        return Err(FactorError::FlintFailure);
    }

    // FlintNmodPoly and FlintNmodPolyFactor are drop-safe: no manual
    // nmod_poly_clear / nmod_poly_factor_clear needed.
    let mut poly = FlintNmodPoly::new(modulus);
    for (i, &r) in reduced[..=last].iter().enumerate() {
        poly.set_coeff(i, r);
    }

    let mut fac = FlintNmodPolyFactor::new();
    fac.factor(&poly);

    let mut factors: Vec<(Vec<u64>, u32)> = (0..fac.len())
        .map(|i| {
            let z = fac.poly_at(modulus, i);
            let deg = z.degree();
            let vc = (0..=deg).map(|j| z.get_coeff(j)).collect::<Vec<_>>();
            (vc, fac.exp_at(i))
        })
        .collect();
    // FLINT's order comes out of randomised equal-degree splitting.
    factors.sort_by(cmp_mod_p_factors);

    Ok((unit, UniPolyFactorModP { modulus, factors }))
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

    fn coeffs_of(f: &FlintPoly) -> Vec<i64> {
        f.coefficients()
    }

    /// `x^12 − 1` in the documented order (SymPy's `factor_list` order).
    #[test]
    fn canonical_order_x12_minus_1() {
        let f = FlintPoly::from_coefficients(&[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        let (unit, fac) = f.factor_over_z().unwrap();
        assert_eq!(unit.to_rug(), 1);
        let got: Vec<Vec<i64>> = fac.iter().map(|(p, _)| coeffs_of(p)).collect();
        assert_eq!(
            got,
            vec![
                vec![-1, 1],
                vec![1, 1],
                vec![1, -1, 1],
                vec![1, 0, 1],
                vec![1, 1, 1],
                vec![1, 0, -1, 0, 1],
            ]
        );
    }

    /// The order is a function of the factors alone: strictly increasing in
    /// the canonical comparison, and the same whatever order the input was
    /// multiplied together in — including through FLINT's van Hoeij path,
    /// whose own order differs between FLINT 3.5 and 3.6 on these inputs.
    #[test]
    fn canonical_order_is_independent_of_flint() {
        let x = FlintPoly::from_coefficients(&[0, 1]);
        let xp2 = FlintPoly::from_coefficients(&[2, 1]);
        for n in [156usize, 168, 240] {
            let mut c = vec![0i64; n + 1];
            c[0] = -1;
            c[n] = 1;
            // Not a binomial, so FLINT factors it (van Hoeij on x^n − 1).
            let f = &(&FlintPoly::from_coefficients(&c) * &xp2) * &x.pow(2);
            let (unit, fac) = f.factor_over_z().unwrap();
            for w in fac.windows(2) {
                assert_eq!(
                    cmp_univariate_factors(&w[0], &w[1]),
                    std::cmp::Ordering::Less,
                    "n = {n}: not strictly sorted"
                );
            }
            let (unit_ref, mut raw) = f.factor_over_z_flint_order().unwrap();
            assert_eq!(unit, unit_ref);
            raw.sort_by(cmp_univariate_factors);
            assert!(raw == fac, "n = {n}");
            // Degree-1 factors: x − 1, x (squared), x + 1, x + 2.
            let lin: Vec<(Vec<i64>, u32)> = fac
                .iter()
                .take_while(|(p, _)| p.degree() == 1)
                .map(|(p, e)| (coeffs_of(p), *e))
                .collect();
            assert_eq!(
                lin,
                vec![
                    (vec![-1, 1], 1),
                    (vec![0, 1], 2),
                    (vec![1, 1], 1),
                    (vec![2, 1], 1)
                ]
            );
        }
    }

    /// Random products: the factorisation (unit, factors, multiplicities) is
    /// FLINT's, and its order does not depend on the order of the input.
    #[test]
    fn random_products_canonical_and_unchanged() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |m: i64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % (2 * m as u64 + 1)) as i64 - m
        };
        for trial in 0..200 {
            let k = 1 + (next(2) + 2) as usize;
            let mut parts = Vec::new();
            for _ in 0..k {
                let deg = 1 + (next(1) + 1) as usize;
                let mut c: Vec<i64> = (0..deg).map(|_| next(4)).collect();
                c.push(1 + next(1).abs());
                parts.push(FlintPoly::from_coefficients(&c));
            }
            let scale = FlintPoly::from_coefficients(&[
                next(3).max(1) * if trial % 2 == 0 { 1 } else { -1 }
            ]);
            let mut f = scale.clone();
            for p in &parts {
                f = &f * p;
            }
            let mut g = scale;
            for p in parts.iter().rev() {
                g = &g * p;
            }
            assert!(f == g);
            let (uf, ff) = f.factor_over_z().unwrap();
            let (ur, mut fr) = f.factor_over_z_flint_order().unwrap();
            fr.sort_by(cmp_univariate_factors);
            assert!(uf == ur && ff == fr, "trial {trial}");
            for w in ff.windows(2) {
                assert_eq!(
                    cmp_univariate_factors(&w[0], &w[1]),
                    std::cmp::Ordering::Less
                );
            }
            let up = UniPoly {
                var: crate::kernel::ExprPool::new().symbol("x", Domain::Real),
                coeffs: f.clone(),
            };
            assert!(
                up.factor_z().unwrap().verifies_product(&up),
                "trial {trial}"
            );
        }
    }

    #[test]
    fn multivariate_canonical_order() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let vars = vec![x, y];
        let e = pool.add(vec![
            pool.pow(x, pool.integer(6_i32)),
            pool.mul(vec![pool.integer(-1_i32), pool.pow(y, pool.integer(6_i32))]),
        ]);
        let p = MultiPoly::from_symbolic(e, vars, &pool).unwrap();
        let fac = p.factor_z().unwrap();
        assert!(fac.verifies_product(&p));
        let mut want = Vec::new();
        for terms in [
            vec![(vec![1u32], 1i64), (vec![0, 1], -1)],
            vec![(vec![1], 1), (vec![0, 1], 1)],
            vec![(vec![2], 1), (vec![1, 1], -1), (vec![0, 2], 1)],
            vec![(vec![2], 1), (vec![1, 1], 1), (vec![0, 2], 1)],
        ] {
            let t: std::collections::BTreeMap<Vec<u32>, rug::Integer> = terms
                .into_iter()
                .map(|(e, c)| (e, rug::Integer::from(c)))
                .collect();
            want.push(t);
        }
        let got: Vec<_> = fac.factors.iter().map(|(m, _)| m.terms.clone()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn mod_p_factors_are_sorted() {
        // x^8 − 1 over 𝔽_17 splits into eight linear factors.
        let fac = factor_univariate_mod_p(&[-1, 0, 0, 0, 0, 0, 0, 0, 1], 17).unwrap();
        let roots: Vec<u64> = fac.factors.iter().map(|(f, _)| f[0]).collect();
        let mut sorted = roots.clone();
        sorted.sort_unstable();
        assert_eq!(roots, sorted);
        assert_eq!(roots.len(), 8);
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

    /// A composite modulus used to reach `nmod_poly_factor`, which aborts the
    /// process ("Cannot invert modulo 3*5") on the first non-invertible leading
    /// coefficient. It must be refused before FLINT sees it.
    #[test]
    fn composite_modulus_is_refused_not_aborted() {
        for m in [4u64, 9, 12, 15, 21, 1 << 32, u64::MAX] {
            for coeffs in [&[6i64, 5, 1][..], &[1, 0, 1], &[0, 0, 1], &[1, 2, 1]] {
                assert_eq!(
                    factor_univariate_mod_p(coeffs, m),
                    Err(FactorError::InvalidModulus),
                    "{coeffs:?} mod {m}"
                );
            }
        }
        assert_eq!(
            factor_univariate_mod_p(&[1, 0, 1], 1),
            Err(FactorError::InvalidModulus)
        );
        // Primes up to the top of the machine word are still accepted.
        let big = factor_univariate_mod_p(&[1, 0, 1], 18_446_744_073_709_551_557).unwrap();
        assert!(!big.factors.is_empty());
    }

    /// The unit used to be dropped silently: `5` mod 7 factored as the empty
    /// product (i.e. 1) and `2x` as `x`.
    #[test]
    fn mod_p_factorisation_keeps_the_unit() {
        let (u, f) = factor_univariate_mod_p_with_unit(&[5], 7).unwrap();
        assert_eq!((u, f.factors), (5, vec![]));
        let (u, f) = factor_univariate_mod_p_with_unit(&[0, 2], 7).unwrap();
        assert_eq!((u, f.factors), (2, vec![(vec![0, 1], 1)]));
        // Negative coefficients reduce to their least non-negative residue.
        let (u, f) = factor_univariate_mod_p_with_unit(&[2, -5], 7).unwrap();
        assert_eq!(u, 2);
        assert_eq!(f.factors, vec![(vec![1, 1], 1)]);
        // Reconstruct u·∏fᵢ^eᵢ and compare with the reduced input.
        let input = [3i64, -4, 0, 6, 2];
        let p = 11u64;
        let (u, f) = factor_univariate_mod_p_with_unit(&input, p).unwrap();
        let mul = |a: &[u64], b: &[u64]| {
            let mut out = vec![0u64; a.len() + b.len() - 1];
            for (i, &x) in a.iter().enumerate() {
                for (j, &y) in b.iter().enumerate() {
                    out[i + j] = (out[i + j] + x * y) % p;
                }
            }
            out
        };
        let mut prod = vec![u];
        for (g, e) in &f.factors {
            for _ in 0..*e {
                prod = mul(&prod, g);
            }
        }
        let want: Vec<u64> = input
            .iter()
            .map(|&c| c.rem_euclid(p as i64) as u64)
            .collect();
        assert_eq!(prod, want);
    }

    #[test]
    fn zero_mod_p_is_the_zero_polynomial() {
        for coeffs in [&[][..], &[7i64], &[0, 14, -21]] {
            assert_eq!(
                factor_univariate_mod_p(coeffs, 7),
                Err(FactorError::ZeroPolynomial),
                "{coeffs:?}"
            );
        }
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
