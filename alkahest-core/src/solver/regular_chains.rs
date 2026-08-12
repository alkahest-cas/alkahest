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

/// A triangular set extracted from a lex Gröbner basis: at most one polynomial
/// per recursive main variable (see [`main_variable_recursive`]), stored in
/// increasing variable index — that is, from the lex-greatest variable down.
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

/// The **recursive main variable** of `poly`: the greatest variable, in the
/// ambient ordering, that occurs with positive degree.
///
/// Variables are indexed in *decreasing* rank — index `0` is the lex-greatest
/// variable, index `n − 1` the lex-least, which is the layout
/// [`crate::solver::expr_to_gbpoly`] builds and the one the bottom-univariate
/// split assumes when it calls `n − 1` "the bottom". The main variable is
/// therefore the **smallest** occurring index.
///
/// This returned the *largest* index until 3.8. Under that reading every
/// generator mentioning a low-ranked variable was filed in the same slot of
/// [`extract_regular_chain_from_basis`], and the min-degree tie-break dropped
/// the rest: `[x − y − 1, y² − 2]` — already a reduced lex basis of a two-point
/// ideal — came back as the single chain `[x − y − 1]`, which cuts out a curve.
pub fn main_variable_recursive(poly: &GbPoly) -> Option<usize> {
    let mut best: Option<usize> = None;
    for exp in poly.terms.keys() {
        for (i, &e) in exp.iter().enumerate() {
            if e > 0 {
                best = Some(best.map_or(i, |b| b.min(i)));
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
///
/// This is a *selection*, not a decomposition: when two basis elements share a
/// main variable only one survives, so `⟨chain⟩` can be strictly smaller than
/// the ideal and `V(chain)` strictly larger than `V(I)`. Every polynomial kept
/// is a basis element, so `⟨chain⟩ ⊆ I` always holds — which is what
/// [`crate::solver::solve_polynomial_system`] relies on when it uses the chain
/// as a source of *candidate* solutions that its own post-condition filter then
/// checks. [`triangularize`], whose output is the answer rather than a
/// candidate list, verifies the reverse containment and refuses without it.
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

/// True iff `⟨chain⟩ ⊇ I`, i.e. every generator of the input basis reduces to
/// zero modulo the chain.
///
/// This is the soundness direction that matters: with it, `V(chain) ⊆ V(I)`, so
/// the chain cannot describe points the input system does not have. Without it
/// the chain cuts out a *larger* set than the system — a curve where the answer
/// is two points — which is exactly the failure `main_variable_recursive`'s
/// reversed ordering used to produce.
fn chain_contains_ideal(chain: &RegularChain, gb_gens: &[GbPoly]) -> bool {
    if chain.polys.is_empty() {
        return gb_gens.iter().all(|g| g.is_zero());
    }
    let chain_gb = GroebnerBasis::compute(chain.polys.clone(), MonomialOrder::Lex);
    gb_gens.iter().all(|g| chain_gb.contains(g))
}

/// Kalkbrener / Lazard style triangular decomposition: compute a lex Gröbner basis,
/// extract a recursive main-variable chain, then split along square-free factors of
/// the bottom univariate when possible (V2-7).
///
/// Returns an empty list when the ideal is the whole ring (`⟨1⟩`).
///
/// # Refusals
///
/// Chain extraction keeps one polynomial per main variable, so a basis with two
/// generators sharing a main variable — `⟨xy, xz⟩` is the smallest example —
/// loses one of them and the surviving chain describes a larger variety than
/// the input system. Splitting on general initials (Lazard–Kalkbrener) is what
/// would decompose those ideals properly, and is not implemented. Rather than
/// return the under-determined chain, this function checks `⟨chain⟩ ⊇ I` for
/// every chain it is about to return and refuses when the check fails; see
/// [`TriangularizeRefusal`] for the code that refusal carries.
pub fn triangularize(
    equations: Vec<ExprId>,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<Vec<RegularChain>, SolverError> {
    forget_triangularize_refusal();
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
    let chains = split_chain_at_bottom_univariate(chain, last_var)?;

    for c in &chains {
        if !chain_contains_ideal(c, gens) {
            return Err(refuse_triangularize(gens.len(), c.polys.len()));
        }
    }
    Ok(chains)
}

// ---------------------------------------------------------------------------
// Refusals, reported out of band
// ---------------------------------------------------------------------------

/// `triangularize` declined because the chain it extracted is not a triangular
/// decomposition of the input ideal.
///
/// # Why this is not an error variant
///
/// [`SolverError`] is a public *exhaustive* enum, so growing it a
/// `NotTriangularizable` variant is a major semver break — and so is marking it
/// `#[non_exhaustive]` to allow one later. A correctness fix inside a patch
/// release cannot spend a major version, so the refusal travels out of band:
/// [`triangularize`] returns `SolverError::NotPolynomial`, which this module
/// already uses as its generic carrier (the FLINT failure path of the
/// bottom-univariate split reports through it too), with a message that names
/// the real reason, and the stable `E-SOLVE-004` code is recorded here for
/// [`take_triangularize_refusal`] to hand to the bindings.
///
/// This is the pattern
/// [`crate::matrix::take_zero_test_refusal`] uses for undecided zero tests
/// inside `LinearAlgebraError::UnsupportedField`, and
/// [`crate::calculus::limits::last_budget_trip`] for budget trips inside
/// `LimitError::DepthExceeded`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TriangularizeRefusal {
    basis_len: usize,
    chain_len: usize,
}

impl TriangularizeRefusal {
    /// How many generators the lex basis had.
    pub fn basis_len(&self) -> usize {
        self.basis_len
    }

    /// How many polynomials the extracted chain kept.
    pub fn chain_len(&self) -> usize {
        self.chain_len
    }
}

impl std::fmt::Display for TriangularizeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "triangularize: the {} lex basis generators do not extract to a \
             triangular set — the {}-polynomial chain does not generate an ideal \
             containing them, so it cuts out a larger variety than the input \
             system; refusing rather than returning an under-determined chain",
            self.basis_len, self.chain_len
        )
    }
}

impl std::error::Error for TriangularizeRefusal {}

impl crate::errors::AlkahestError for TriangularizeRefusal {
    fn code(&self) -> &'static str {
        "E-SOLVE-004"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "this ideal needs a splitting triangular decomposition (Lazard–Kalkbrener \
             on the initials), which is not implemented; use GroebnerBasis::compute or \
             primary_decomposition instead",
        )
    }
}

thread_local! {
    /// The refusal behind the `SolverError::NotPolynomial` the current thread is
    /// about to return, when that variant is a carrier rather than what it
    /// usually means.
    static LAST_TRIANGULARIZE_REFUSAL: std::cell::RefCell<Option<TriangularizeRefusal>> =
        const { std::cell::RefCell::new(None) };
}

/// Drop any recorded refusal, so a later unrelated `NotPolynomial` — a genuinely
/// non-polynomial equation, say — can never be re-attributed to it.
fn forget_triangularize_refusal() {
    LAST_TRIANGULARIZE_REFUSAL.with(|c| *c.borrow_mut() = None);
}

fn refuse_triangularize(basis_len: usize, chain_len: usize) -> SolverError {
    let refusal = TriangularizeRefusal {
        basis_len,
        chain_len,
    };
    let message = refusal.to_string();
    LAST_TRIANGULARIZE_REFUSAL.with(|c| *c.borrow_mut() = Some(refusal));
    SolverError::NotPolynomial(message)
}

/// Take the refusal behind the error that just came back, if there was one.
///
/// Bindings call this when [`triangularize`] returns
/// `SolverError::NotPolynomial` and raise the refusal's own `E-SOLVE-004` when
/// it is present, so the caller still gets the specific code. `Some` means
/// *this* error is a refusal; `None` means the variant means what it usually
/// means. Consuming, so one refusal is reported once; thread-local.
pub fn take_triangularize_refusal() -> Option<TriangularizeRefusal> {
    LAST_TRIANGULARIZE_REFUSAL.with(|c| c.borrow_mut().take())
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
    fn main_variable_is_the_lex_greatest_occurring() {
        // x - y in vars [x, y]: x is lex-greatest, so the main variable is 0.
        let p = GbPoly {
            terms: [
                (vec![1u32, 0], Rational::from(1)),
                (vec![0, 1], Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        assert_eq!(main_variable_recursive(&p), Some(0));
    }

    #[test]
    fn triangularize_keeps_every_generator_of_a_two_point_ideal() {
        // {x - y - 1, y² - 2} is already a reduced lex basis; its variety is the
        // two points (1 ± √2, ±√2).  A one-polynomial chain in two variables
        // cuts out a curve, so it cannot be the answer whichever poly is kept.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let eq1 = pool.add(vec![
            x,
            pool.mul(vec![pool.integer(-1), y]),
            pool.integer(-1),
        ]);
        let eq2 = pool.add(vec![pool.pow(y, pool.integer(2)), pool.integer(-2)]);
        let chains = triangularize(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].len(), 2, "both generators must survive");
        let mains: Vec<Option<usize>> = chains[0]
            .polys
            .iter()
            .map(main_variable_recursive)
            .collect();
        assert_eq!(mains, vec![Some(0), Some(1)], "one poly per main variable");
    }

    #[test]
    fn triangularize_refuses_rather_than_drop_a_generator() {
        // ⟨xy, xz⟩ = ⟨x⟩ ∩ ⟨y, z⟩ needs a *split*; both generators have main
        // variable x, so extraction can only keep one and ⟨xy⟩ ⊉ ⟨xy, xz⟩.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let err = triangularize(
            vec![pool.mul(vec![x, y]), pool.mul(vec![x, z])],
            vec![x, y, z],
            &pool,
        )
        .expect_err("must refuse rather than return a chain missing xz");
        assert!(matches!(err, SolverError::NotPolynomial(_)));
        let refusal = take_triangularize_refusal().expect("refusal recorded out of band");
        assert_eq!(crate::errors::AlkahestError::code(&refusal), "E-SOLVE-004");
        assert_eq!(refusal.basis_len(), 2);
        assert_eq!(refusal.chain_len(), 1);
        // Consuming: a second take must not re-report it.
        assert_eq!(take_triangularize_refusal(), None);
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
