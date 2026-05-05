//! Primary decomposition over ℚ\[x₁,…,xₙ\] via saturation splits and Lex
//! factorization (Gianni–Trager–Zacharias fragment).
//!
//! Splits use `I = (I : x_i^∞) ∩ (I + (x_i))` when the intersection checks out,
//! and a zero-dimensional split from factoring the univariate generator in the
//! first Lex variable when it has multiple distinct irreducible factors over ℚ.

use crate::flint::FlintPoly;
use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::GroebnerBasis;
use std::collections::BTreeMap;
use std::fmt;

const MAX_SPLIT_DEPTH: usize = 48;

/// One primary component together with its associated prime (√Q).
#[derive(Clone, Debug)]
pub struct PrimaryComponent {
    pub primary: GroebnerBasis,
    pub associated_prime: GroebnerBasis,
}

/// Primary-decomposition failures (inconsistent input, depth limit, FLINT).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimaryDecompositionError {
    EmptyGenerators,
    InconsistentNvars,
    RecursionDepth,
    Factorization(&'static str),
}

impl fmt::Display for PrimaryDecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimaryDecompositionError::EmptyGenerators => {
                write!(f, "ideal generators must be non-empty")
            }
            PrimaryDecompositionError::InconsistentNvars => {
                write!(f, "inconsistent n_vars across generators")
            }
            PrimaryDecompositionError::RecursionDepth => {
                write!(f, "primary decomposition exceeded recursion depth")
            }
            PrimaryDecompositionError::Factorization(msg) => {
                write!(f, "univariate factorization failed: {msg}")
            }
        }
    }
}

impl std::error::Error for PrimaryDecompositionError {}

fn lcm_rational_denoms(coeffs: &[rug::Rational]) -> rug::Integer {
    let mut m = rug::Integer::from(1);
    for c in coeffs {
        m = m.lcm(c.denom());
    }
    m
}

/// Clear denominators and strip integer content → primitive `FlintPoly`.
fn primitive_flint_from_rational_asc(coeffs: &[rug::Rational]) -> Option<FlintPoly> {
    if coeffs.is_empty() {
        return None;
    }
    let mut hi = coeffs.len();
    while hi > 0 && coeffs[hi - 1] == 0 {
        hi -= 1;
    }
    if hi == 0 {
        return None;
    }
    let coeffs = &coeffs[..hi];
    let lcm = lcm_rational_denoms(coeffs);
    let mut ints: Vec<rug::Integer> = Vec::with_capacity(coeffs.len());
    for c in coeffs {
        let t = c * rug::Rational::from((lcm.clone(), 1));
        let (n, d) = t.into_numer_denom();
        debug_assert_eq!(d, rug::Integer::from(1));
        ints.push(n);
    }
    let mut g = ints[0].clone();
    for a in ints.iter().skip(1) {
        g = g.gcd(a);
    }
    if g != 0 {
        for a in &mut ints {
            let (q, r) = a.clone().div_rem(g.clone());
            debug_assert_eq!(r, 0);
            *a = q;
        }
    }
    Some(FlintPoly::from_rug_coefficients(&ints))
}

fn gbpoly_eq(a: &GbPoly, b: &GbPoly) -> bool {
    a.n_vars == b.n_vars && a.terms == b.terms
}

/// Radical √I via repeated augmentation with squarefree parts of univariate
/// polynomials in each variable (characteristic 0), then a Gröbner basis pass.
pub fn radical(
    gens: Vec<GbPoly>,
    order: MonomialOrder,
) -> Result<GroebnerBasis, PrimaryDecompositionError> {
    validate_gens(&gens)?;
    let gb = GroebnerBasis::compute(gens, order);
    Ok(radical_from_basis(&gb, order))
}

/// Irredundant primary decomposition (partial — see module docs).
pub fn primary_decomposition(
    gens: Vec<GbPoly>,
    order: MonomialOrder,
) -> Result<Vec<PrimaryComponent>, PrimaryDecompositionError> {
    validate_gens(&gens)?;
    let gb = GroebnerBasis::compute(gens, order);
    if is_unit_ideal(&gb) {
        return Ok(vec![]);
    }
    let mut raw = decompose_recursive(gb, order, 0)?;
    dedup_components(&mut raw);
    for c in &mut raw {
        c.associated_prime = radical_from_basis(&c.primary, order);
    }
    Ok(raw)
}

fn validate_gens(gens: &[GbPoly]) -> Result<(), PrimaryDecompositionError> {
    if gens.is_empty() {
        return Err(PrimaryDecompositionError::EmptyGenerators);
    }
    let n = gens[0].n_vars;
    if gens.iter().any(|g| g.n_vars != n) {
        return Err(PrimaryDecompositionError::InconsistentNvars);
    }
    Ok(())
}

fn is_unit_ideal(gb: &GroebnerBasis) -> bool {
    gb.generators().iter().any(|g| {
        g.terms.len() == 1
            && g.terms
                .keys()
                .next()
                .is_some_and(|e| e.iter().all(|&x| x == 0))
            && g.terms.values().next().is_some_and(|c| *c != 0)
    })
}

fn ideals_equal(a: &GroebnerBasis, b: &GroebnerBasis) -> bool {
    for g in a.generators() {
        if !b.contains(g) {
            return false;
        }
    }
    for g in b.generators() {
        if !a.contains(g) {
            return false;
        }
    }
    true
}

/// Embed `p ∈ R` into `R[t]` with `t` the first exponent (`t^0 · …`).
fn embed_add_t_front(p: &GbPoly) -> GbPoly {
    let n = p.n_vars + 1;
    let mut terms = BTreeMap::new();
    for (e, c) in &p.terms {
        let mut ne = Vec::with_capacity(n);
        ne.push(0u32);
        ne.extend_from_slice(e);
        terms.insert(ne, c.clone());
    }
    GbPoly { terms, n_vars: n }
}

/// `1 - t·f` in `R[t,…]` (first variable is `t`).
fn one_minus_t_times_f(f: &GbPoly) -> GbPoly {
    let fe = embed_add_t_front(f);
    let mut terms: BTreeMap<Vec<u32>, rug::Rational> = BTreeMap::new();
    for (e, c) in &fe.terms {
        let mut ne = e.clone();
        ne[0] += 1;
        let entry = terms.entry(ne).or_insert_with(|| rug::Rational::from(0));
        *entry -= c;
    }
    let zero = vec![0u32; fe.n_vars];
    let one = terms.entry(zero).or_insert_with(|| rug::Rational::from(0));
    *one += 1;
    terms.retain(|_, v| *v != 0);
    GbPoly {
        terms,
        n_vars: fe.n_vars,
    }
}

/// Saturation `I : ⟨f⟩^∞` as `(I, 1 - t·f) ∩ R` (new variable `t` first in Lex).
fn saturate_ideal(generators: &[GbPoly], f: &GbPoly, order: MonomialOrder) -> GroebnerBasis {
    let mut ext: Vec<GbPoly> = Vec::with_capacity(generators.len() + 1);
    for g in generators {
        ext.push(embed_add_t_front(g));
    }
    ext.push(one_minus_t_times_f(f));
    let gb_ext = GroebnerBasis::compute(ext, order);
    let elim = gb_ext.eliminate(&[0]);
    let stripped: Vec<GbPoly> = elim.generators().iter().map(strip_first_var).collect();
    GroebnerBasis::compute(stripped, order)
}

fn strip_first_var(p: &GbPoly) -> GbPoly {
    let old_n = p.n_vars;
    if old_n == 0 {
        return GbPoly::zero(0);
    }
    let n = old_n - 1;
    let mut terms = BTreeMap::new();
    for (e, c) in &p.terms {
        if e.len() != old_n || e[0] != 0 {
            continue;
        }
        if n == 0 {
            terms.insert(vec![], c.clone());
        } else {
            terms.insert(e[1..].to_vec(), c.clone());
        }
    }
    GbPoly { terms, n_vars: n }
}

/// `I ∩ J` from `t·I + (t-1)·J` (first variable is `t`).
fn ideal_intersection(i: &[GbPoly], j: &[GbPoly], order: MonomialOrder) -> GroebnerBasis {
    let mut ext = Vec::with_capacity(i.len() + j.len());
    for g in i {
        let ge = embed_add_t_front(g);
        ext.push(mul_t(&ge));
    }
    for h in j {
        let he = embed_add_t_front(h);
        let th = mul_t(&he);
        ext.push(th.sub(&he));
    }
    let gb_ext = GroebnerBasis::compute(ext, order);
    let elim = gb_ext.eliminate(&[0]);
    let stripped: Vec<GbPoly> = elim.generators().iter().map(strip_first_var).collect();
    GroebnerBasis::compute(stripped, order)
}

fn mul_t(p: &GbPoly) -> GbPoly {
    let mut terms = BTreeMap::new();
    for (e, c) in &p.terms {
        let mut ne = e.clone();
        if ne.is_empty() {
            continue;
        }
        ne[0] += 1;
        terms.insert(ne, c.clone());
    }
    GbPoly {
        terms,
        n_vars: p.n_vars,
    }
}

fn var_monomial(n_vars: usize, idx: usize) -> GbPoly {
    let mut exp = vec![0u32; n_vars];
    exp[idx] = 1;
    GbPoly::monomial(exp, rug::Rational::from(1))
}

fn decompose_recursive(
    gb: GroebnerBasis,
    order: MonomialOrder,
    depth: usize,
) -> Result<Vec<PrimaryComponent>, PrimaryDecompositionError> {
    if depth > MAX_SPLIT_DEPTH {
        return Err(PrimaryDecompositionError::RecursionDepth);
    }
    if is_unit_ideal(&gb) {
        return Ok(vec![]);
    }

    let n_vars = gb.generators()[0].n_vars;

    for i in 0..n_vars {
        let f = var_monomial(n_vars, i);
        let sat_gb = saturate_ideal(gb.generators(), &f, order);
        let mut sum_gens = gb.generators().to_vec();
        sum_gens.push(var_monomial(n_vars, i));
        let sum_gb = GroebnerBasis::compute(sum_gens, order);

        if is_unit_ideal(&sat_gb) || is_unit_ideal(&sum_gb) {
            continue;
        }
        if ideals_equal(&sat_gb, &gb) || ideals_equal(&sum_gb, &gb) {
            continue;
        }
        let inter = ideal_intersection(sat_gb.generators(), sum_gb.generators(), order);
        if !ideals_equal(&inter, &gb) {
            continue;
        }

        let left = decompose_recursive(sat_gb, order, depth + 1)?;
        let right = decompose_recursive(sum_gb, order, depth + 1)?;
        let mut out = left;
        out.extend(right);
        return Ok(out);
    }

    if order == MonomialOrder::Lex {
        if let Some(pieces) = try_lex_factor_split(gb.generators(), order, n_vars)? {
            let mut acc = Vec::new();
            for piece_gens in pieces {
                let piece = GroebnerBasis::compute(piece_gens, order);
                acc.extend(decompose_recursive(piece, order, depth + 1)?);
            }
            return Ok(acc);
        }
    }

    let ass = radical_from_basis(&gb, order);
    Ok(vec![PrimaryComponent {
        primary: gb,
        associated_prime: ass,
    }])
}

/// If the basis contains a univariate polynomial only in variable 0, split along
/// coprime factors `p_j^{e_j}` as `⟨I, p_j^{e_j}⟩`.
fn try_lex_factor_split(
    gens: &[GbPoly],
    _order: MonomialOrder,
    n_vars: usize,
) -> Result<Option<Vec<Vec<GbPoly>>>, PrimaryDecompositionError> {
    let u = match find_univariate_in_var0(gens) {
        Some(u) => u,
        None => return Ok(None),
    };
    let facs = factor_univariate_q_monic(&u, n_vars)?;
    if facs.len() <= 1 {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(facs.len());
    for (p, e) in facs {
        let mut g = gens.to_vec();
        g.push(gbpoly_pow(&p, e));
        out.push(g);
    }
    Ok(Some(out))
}

fn find_univariate_in_var0(gens: &[GbPoly]) -> Option<GbPoly> {
    for g in gens {
        if g.is_zero() {
            continue;
        }
        let mut ok = true;
        for e in g.terms.keys() {
            if e.is_empty() {
                ok = false;
                break;
            }
            if e[0..].iter().skip(1).any(|&x| x != 0) {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(g.clone());
        }
    }
    None
}

fn gbpoly_pow(p: &GbPoly, e: u32) -> GbPoly {
    let mut acc = GbPoly::constant(rug::Rational::from(1), p.n_vars);
    for _ in 0..e {
        acc = acc.mul(p);
    }
    acc
}

fn flint_monic_to_gbpoly_var0(fz: &FlintPoly, n_vars: usize) -> GbPoly {
    let deg = fz.degree();
    if deg < 0 {
        return GbPoly::zero(n_vars);
    }
    let lc = fz.get_coeff_flint(deg as usize).to_rug();
    let mut terms = BTreeMap::new();
    for d in 0..=deg as usize {
        let cz = fz.get_coeff_flint(d).to_rug();
        if cz == 0 {
            continue;
        }
        let rq = rug::Rational::from((cz.clone(), lc.clone()));
        let mut expv = vec![0u32; n_vars];
        expv[0] = d as u32;
        terms.insert(expv, rq);
    }
    GbPoly { terms, n_vars }
}

/// Factor a univariate `p(x₀)`; returns **monic** irreducible factors over ℚ.
fn factor_univariate_q_monic(
    p: &GbPoly,
    n_vars: usize,
) -> Result<Vec<(GbPoly, u32)>, PrimaryDecompositionError> {
    if n_vars < 1 || !is_univariate_in_var(p, 0) {
        return Err(PrimaryDecompositionError::Factorization(
            "internal: expected univariate in var 0",
        ));
    }
    let mut coeff_map: BTreeMap<u32, rug::Rational> = BTreeMap::new();
    for (e, c) in &p.terms {
        coeff_map.insert(e[0], c.clone());
    }
    let deg = *coeff_map.keys().max().unwrap_or(&0);
    let coeffs_r: Vec<rug::Rational> = (0..=deg)
        .map(|d| {
            coeff_map
                .get(&d)
                .cloned()
                .unwrap_or_else(|| rug::Rational::from(0))
        })
        .collect();
    let fp = primitive_flint_from_rational_asc(&coeffs_r).ok_or(
        PrimaryDecompositionError::Factorization("could not build integer model"),
    )?;
    let (_unit, facs) = fp
        .factor_over_z()
        .map_err(|_| PrimaryDecompositionError::Factorization("FLINT factor_over_z"))?;
    let mut pairs = Vec::new();
    for (fz, exp) in facs {
        let g = flint_monic_to_gbpoly_var0(&fz, n_vars);
        pairs.push((g, exp));
    }
    Ok(pairs)
}

fn is_univariate_in_var(p: &GbPoly, var: usize) -> bool {
    p.terms
        .keys()
        .all(|e| e.len() == p.n_vars && e.iter().enumerate().all(|(i, &v)| i == var || v == 0))
}

fn radical_from_basis(gb: &GroebnerBasis, order: MonomialOrder) -> GroebnerBasis {
    let n = gb.generators().first().map(|p| p.n_vars).unwrap_or(0);
    let mut gens = gb.generators().to_vec();
    for _ in 0..(n + 4) {
        let mut appended = false;
        for i in 0..n {
            if let Some(u) = find_any_univariate(&gens, i) {
                let sf = univariate_squarefree_part(&u, i, order);
                if !gbpoly_eq(&sf, &u) {
                    gens.push(sf);
                    appended = true;
                }
            }
        }
        if !appended {
            break;
        }
        gens = GroebnerBasis::compute(gens, order).generators().to_vec();
    }
    GroebnerBasis::compute(gens, order)
}

fn find_any_univariate(gens: &[GbPoly], var: usize) -> Option<GbPoly> {
    for g in gens {
        if g.is_zero() || !is_univariate_in_var(g, var) {
            continue;
        }
        return Some(g.clone());
    }
    None
}

fn univariate_squarefree_part(u: &GbPoly, var: usize, order: MonomialOrder) -> GbPoly {
    let n = u.n_vars;
    let mut coeff_map: BTreeMap<u32, rug::Rational> = BTreeMap::new();
    for (e, c) in &u.terms {
        coeff_map.insert(e[var], c.clone());
    }
    let deg = *coeff_map.keys().max().unwrap_or(&0);
    let coeffs_r: Vec<rug::Rational> = (0..=deg)
        .map(|d| {
            coeff_map
                .get(&d)
                .cloned()
                .unwrap_or_else(|| rug::Rational::from(0))
        })
        .collect();
    let fp = match primitive_flint_from_rational_asc(&coeffs_r) {
        Some(p) => p,
        None => return GbPoly::zero(n),
    };
    let der = fp.derivative();
    let g = fp.gcd(&der);
    let sf = fp.div_exact(&g);
    let mut terms = BTreeMap::new();
    for d in 0..=sf.degree() {
        let cz = sf.get_coeff_flint(d as usize).to_rug();
        if cz == 0 {
            continue;
        }
        let mut expv = vec![0u32; n];
        expv[var] = d as u32;
        terms.insert(expv, rug::Rational::from((cz, 1)));
    }
    GbPoly { terms, n_vars: n }.make_monic(order)
}

fn dedup_components(v: &mut Vec<PrimaryComponent>) {
    let mut i = 0;
    while i < v.len() {
        let mut dup = false;
        for j in 0..i {
            if ideals_equal(&v[i].primary, &v[j].primary) {
                dup = true;
                break;
            }
        }
        if dup {
            v.remove(i);
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64, d: i64) -> rug::Rational {
        rug::Rational::from((n, d))
    }

    #[test]
    fn intersection_xy_xz() {
        let n = 3usize;
        let xy = GbPoly {
            terms: [(vec![1, 1, 0], rat(1, 1))].into_iter().collect(),
            n_vars: n,
        };
        let xz = GbPoly {
            terms: [(vec![1, 0, 1], rat(1, 1))].into_iter().collect(),
            n_vars: n,
        };
        let gb_i = GroebnerBasis::compute(vec![xy, xz], MonomialOrder::Lex);
        let f = var_monomial(n, 0);
        let a = saturate_ideal(gb_i.generators(), &f, MonomialOrder::Lex);
        let mut sg = gb_i.generators().to_vec();
        sg.push(f);
        let b = GroebnerBasis::compute(sg, MonomialOrder::Lex);
        let inter = ideal_intersection(a.generators(), b.generators(), MonomialOrder::Lex);
        assert!(ideals_equal(&inter, &gb_i));
    }

    #[test]
    fn primary_xy_xz() {
        let n = 3usize;
        let xy = GbPoly {
            terms: [(vec![1, 1, 0], rat(1, 1))].into_iter().collect(),
            n_vars: n,
        };
        let xz = GbPoly {
            terms: [(vec![1, 0, 1], rat(1, 1))].into_iter().collect(),
            n_vars: n,
        };
        let dec = primary_decomposition(vec![xy, xz], MonomialOrder::Lex).unwrap();
        assert_eq!(dec.len(), 2);
    }

    #[test]
    fn primary_x2_xy_embedded() {
        let n = 2usize;
        let x2 = GbPoly {
            terms: [(vec![2, 0], rat(1, 1))].into_iter().collect(),
            n_vars: n,
        };
        let xy_ = GbPoly {
            terms: [(vec![1, 1], rat(1, 1))].into_iter().collect(),
            n_vars: n,
        };
        let gens = vec![x2.clone(), xy_.clone()];
        let dec = primary_decomposition(gens.clone(), MonomialOrder::Lex).unwrap();
        assert_eq!(dec.len(), 2);
        let r = radical(gens, MonomialOrder::Lex).unwrap();
        let one_x = GbPoly {
            terms: [(vec![1, 0], rat(1, 1))].into_iter().collect(),
            n_vars: n,
        };
        assert!(r.contains(&one_x));
    }

    #[test]
    fn factor_split_x2_minus_one() {
        let n = 2usize;
        let xm1 = GbPoly {
            terms: [(vec![2, 0], rat(1, 1)), (vec![0, 0], rat(-1, 1))]
                .into_iter()
                .collect(),
            n_vars: n,
        };
        let y = GbPoly {
            terms: [(vec![0, 1], rat(1, 1))].into_iter().collect(),
            n_vars: n,
        };
        let dec = primary_decomposition(vec![xm1, y], MonomialOrder::Lex).unwrap();
        assert_eq!(dec.len(), 2);
    }
}
