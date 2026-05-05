//! Regular chains and Lex-basis triangular decomposition (V2-11).
//!
//! A [`RegularChain`] packages a triangular subset of a lexicographic
//! Gröbner basis, with an optional split of the bottom univariate factor via
//! [`crate::poly::factor_univariate_z`] (V2-7) when that eliminant factors over ℤ.
//! Lazard–Kalkbrener splitting on general initials may be added later; the
//! subresultant PRS toolkit ([`crate::poly::subresultant_prs`], V2-2) is the
//! intended extension point for initial-based refinement.

use crate::kernel::{ExprId, ExprPool};
use crate::poly::factor::factor_univariate_z;
use crate::poly::groebner::{GbPoly, GroebnerBasis, MonomialOrder};
use crate::poly::unipoly::UniPoly;
use rug::Rational;
use std::collections::BTreeMap;

use super::{expr_to_gbpoly, SolverError};

/// A triangular set extracted from a lex Gröbner basis: polynomials ordered by
/// increasing recursive main variable (see [`main_variable_recursive`]).
#[derive(Debug, Clone)]
pub struct RegularChain {
    pub n_vars: usize,
    pub polys: Vec<GbPoly>,
}

impl RegularChain {
    /// Number of polynomials in the chain.
    pub fn len(&self) -> usize {
        self.polys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.polys.is_empty()
    }
}

/// Largest variable index that appears with positive total degree (recursive main variable).
pub fn main_variable_recursive(poly: &GbPoly) -> Option<usize> {
    let mut best: Option<usize> = None;
    for exp in poly.terms.keys() {
        for (i, &e) in exp.iter().enumerate() {
            if e > 0 {
                best = Some(best.map_or(i, |b| b.max(i)));
            }
        }
    }
    best
}

fn degree_in_var(poly: &GbPoly, var: usize) -> u32 {
    poly.terms
        .keys()
        .map(|e| e.get(var).copied().unwrap_or(0))
        .max()
        .unwrap_or(0)
}

/// True iff every monomial is constant except possibly in `var`.
fn is_univariate_in(poly: &GbPoly, var: usize) -> bool {
    !poly.is_zero()
        && poly
            .terms
            .keys()
            .all(|e| e.iter().enumerate().all(|(i, &exp)| i == var || exp == 0))
}

fn is_unit_ideal(gens: &[GbPoly], n_vars: usize) -> bool {
    gens.len() == 1
        && gens[0].terms.len() == 1
        && gens[0].leading_exp(MonomialOrder::Lex) == Some(vec![0u32; n_vars])
}

/// From a Gröbner basis, pick one polynomial per recursive main variable — the
/// one of minimal degree in that variable among candidates.
pub fn extract_regular_chain_from_basis(
    gens: &[GbPoly],
    n_vars: usize,
    order: MonomialOrder,
) -> RegularChain {
    let mut best: Vec<Option<(GbPoly, u32)>> = vec![None; n_vars];
    for g in gens {
        if let Some(mv) = main_variable_recursive(g) {
            let d = degree_in_var(g, mv);
            let replace = match &best[mv] {
                None => true,
                Some((_, deg)) => d < *deg,
            };
            if replace {
                best[mv] = Some((g.clone().make_monic(order), d));
            }
        }
    }
    let polys: Vec<GbPoly> = best.into_iter().flatten().map(|(p, _)| p).collect();
    RegularChain { n_vars, polys }
}

fn lcm_rational_denoms(coeffs: &[Rational]) -> rug::Integer {
    let mut m = rug::Integer::from(1);
    for c in coeffs {
        m = m.lcm(c.denom());
    }
    m
}

/// Convert a univariate (in `var_idx`) `GbPoly` over ℚ to an integer `UniPoly` in `var_expr`.
fn gbpoly_to_unipoly_z(
    p: &GbPoly,
    var_idx: usize,
    var_expr: ExprId,
) -> Result<UniPoly, SolverError> {
    let mut coeffs_map: BTreeMap<u32, Rational> = BTreeMap::new();
    for (exp, c) in &p.terms {
        let e = exp.get(var_idx).copied().unwrap_or(0);
        if exp.iter().enumerate().any(|(i, &x)| i != var_idx && x > 0) {
            return Err(SolverError::NotPolynomial(
                "expected univariate polynomial for factor split".into(),
            ));
        }
        coeffs_map.insert(e, c.clone());
    }
    let coeffs_rat: Vec<Rational> = (0..=*coeffs_map.keys().max().unwrap_or(&0))
        .map(|d| {
            coeffs_map
                .get(&d)
                .cloned()
                .unwrap_or_else(|| Rational::from(0))
        })
        .collect();
    let lcm = lcm_rational_denoms(&coeffs_rat);
    let mut coeff_ints = Vec::new();
    for r in coeffs_rat {
        let t = r * Rational::from((lcm.clone(), 1));
        let (n, d) = t.into_numer_denom();
        debug_assert_eq!(d, 1);
        coeff_ints.push(n);
    }
    // trim trailing zeros for FlintPoly
    while coeff_ints.len() > 1 && coeff_ints.last() == Some(&rug::Integer::from(0)) {
        coeff_ints.pop();
    }
    let flint = crate::flint::FlintPoly::from_rug_coefficients(&coeff_ints);
    Ok(UniPoly {
        var: var_expr,
        coeffs: flint,
    })
}

/// Embed integer univariate `UniPoly` (single var `var_idx`) into `GbPoly` over ℚ.
fn unipoly_z_to_gbpoly_last(u: &UniPoly, n_vars: usize, var_idx: usize) -> GbPoly {
    let mut terms = BTreeMap::new();
    let deg = u.degree().max(0) as usize;
    for d in 0..=deg {
        let zi = u.coeffs.get_coeff_flint(d).to_rug();
        if zi == 0 {
            continue;
        }
        let mut exp = vec![0u32; n_vars];
        exp[var_idx] = d as u32;
        terms.insert(exp, Rational::from((zi, 1)));
    }
    GbPoly { terms, n_vars }
}

/// After extracting a chain, split on square-free factors of the bottom univariate
/// (in the lex-smallest / highest-index variable) when it factors nontrivially over ℤ.
fn split_chain_at_bottom_univariate(
    chain: RegularChain,
    last_var: ExprId,
) -> Result<Vec<RegularChain>, SolverError> {
    let n = chain.n_vars;
    if n == 0 {
        return Ok(vec![chain]);
    }
    let last = n - 1;
    // Prefer the highest-degree univariate in `last` among chain polynomials.
    let uni_entry = chain
        .polys
        .iter()
        .enumerate()
        .filter(|(_, p)| is_univariate_in(p, last))
        .max_by_key(|(_, p)| degree_in_var(p, last));

    let Some((idx, uni_poly)) = uni_entry else {
        return Ok(vec![chain]);
    };

    let u_z = gbpoly_to_unipoly_z(uni_poly, last, last_var)?;
    let sqf = u_z.squarefree_part();
    if sqf.degree() <= 1 {
        return Ok(vec![chain]);
    }

    let fac = factor_univariate_z(&sqf)
        .map_err(|e| SolverError::NotPolynomial(format!("triangularize factorization: {e}")))?;

    let nontrivial = fac.factors.iter().filter(|(f, _)| f.degree() >= 1).count();
    if nontrivial <= 1 {
        return Ok(vec![chain]);
    }

    let mut out = Vec::new();
    for (factor, _) in fac.factors {
        if factor.degree() < 1 {
            continue;
        }
        let f_gbp = unipoly_z_to_gbpoly_last(&factor, n, last).make_monic(MonomialOrder::Lex);
        let mut polys = chain.polys.clone();
        polys[idx] = f_gbp;
        out.push(RegularChain {
            n_vars: chain.n_vars,
            polys,
        });
    }

    if out.is_empty() {
        Ok(vec![chain])
    } else {
        Ok(out)
    }
}

/// Kalkbrener / Lazard style triangular decomposition: compute a lex Gröbner basis,
/// extract a recursive main-variable chain, then split along square-free factors of
/// the bottom univariate when possible (V2-7).
///
/// Returns an empty list when the ideal is the whole ring (`⟨1⟩`).
pub fn triangularize(
    equations: Vec<ExprId>,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<Vec<RegularChain>, SolverError> {
    let n_vars = vars.len();
    if n_vars == 0 {
        return Ok(vec![]);
    }
    let last_var = *vars.last().expect("n_vars > 0");

    let mut polys: Vec<GbPoly> = Vec::with_capacity(equations.len());
    for eq in &equations {
        polys.push(expr_to_gbpoly(*eq, &vars, pool)?);
    }

    let gb = GroebnerBasis::compute(polys, MonomialOrder::Lex);
    let gens = gb.generators();

    if is_unit_ideal(gens, n_vars) {
        return Ok(vec![]);
    }

    let chain = extract_regular_chain_from_basis(gens, n_vars, MonomialOrder::Lex);
    split_chain_at_bottom_univariate(chain, last_var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn extract_chain_linear_system() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let eq1 = pool.add(vec![x, y, neg_one]);
        let eq2 = pool.add(vec![x, pool.mul(vec![neg_one, y])]);
        let chains = triangularize(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        assert_eq!(chains.len(), 1);
        assert!(!chains[0].is_empty());
    }

    #[test]
    fn split_univariate_square() {
        // (x^2 - 1) = 0  →  two chains after bottom split: x-1 and x+1
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let x2 = pool.pow(x, pool.integer(2));
        let eq = pool.add(vec![x2, pool.mul(vec![pool.integer(-1), one])]);
        let chains = triangularize(vec![eq], vec![x], &pool).unwrap();
        assert_eq!(chains.len(), 2);
        for c in &chains {
            assert_eq!(c.len(), 1);
            assert_eq!(degree_in_var(&c.polys[0], 0), 1);
        }
    }
}
