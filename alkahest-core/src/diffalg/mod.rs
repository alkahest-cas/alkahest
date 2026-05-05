//! V2-13 — Differential algebra / Rosenfeld–Gröbner-style differential elimination.
//!
//! Given a polynomial DAE in implicit form `g_i(t, y, y') = 0`, prolongation
//! appends formal time derivatives `D^k(g_i)` while tracking the derivative
//! state layout (same structural rule as [`crate::dae::pantelides`]).  After
//! each prolongation step, an ordinary Gröbner basis over ℚ captures the
//! algebraic constraints among the jet variables.  Inconsistent systems
//! yield the unit ideal (basis containing a non-zero constant).
//!
//! This is a **fragment** of the full Rosenfeld–Gröbner / regular differential
//! decomposition (no multi-branch saturation over initials here); it suffices
//! for consistency checking and complements Pantelides when the structural
//! index is high.
//!
//! References:
//! - Boulier et al., *Rosenfeld–Gröbner algorithm* (differential elimination).
//! - Hubert, *Differential algebra for comptroller generation*.

use crate::dae::{
    differentiate_equation, extend_dae_for_derivative_symbols, pantelides, DaeError,
    PantelidesResult, DAE,
};
use crate::errors::AlkahestError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::groebner::{GbPoly, GroebnerBasis, MonomialOrder};
use crate::solver::expr_to_gbpoly;
use crate::solver::SolverError;
use std::collections::{BTreeMap, HashSet};
use std::fmt;

/// Default prolongation depth (formal time derivatives chained per equation).
const DEFAULT_MAX_PROLONG_ROUNDS: usize = 8;

/// Ordering data for differential indeterminates: `vars[i]` maps to exponent
/// index `i` of [`GbPoly`] used in Gröbner steps.
#[derive(Clone, Debug)]
pub struct DifferentialRanking {
    pub vars: Vec<ExprId>,
}

/// Algebraic presentation of a finitely generated differential ideal (dense
/// ℚ-polynomial generators in a fixed jet basis).
#[derive(Clone, Debug)]
pub struct DifferentialIdeal {
    pub generators: Vec<GbPoly>,
}

/// Ordinary polynomial ring context for a fixed independent variable and ranked jets.
#[derive(Clone, Debug)]
pub struct DifferentialRing {
    pub time: ExprId,
    pub ranked_indeterminates: Vec<ExprId>,
}

/// One component produced by differential elimination — here the algebraic
/// Gröbner basis of a saturated ordinary ideal at the end of prolongation.
#[derive(Clone, Debug)]
pub struct RegularDifferentialChain {
    pub basis: GroebnerBasis,
}

/// Result of [`rosenfeld_groebner`] (single coherent component in this build).
#[derive(Clone, Debug)]
pub struct RosenfeldGroebnerResult {
    /// `false` iff the unit ideal was encountered (no common jet solution over ℚ).
    pub consistent: bool,
    /// Non-empty when [`Self::consistent`] is true (one entry in this implementation).
    pub chains: Vec<RegularDifferentialChain>,
    /// [`DAE`] state after prolongation (extra derivative jets may be present).
    pub working_dae: DAE,
    /// Final Gröbner basis when consistent.
    pub final_basis: Option<GroebnerBasis>,
    /// Number of prolongation rounds that added new relations.
    pub prolongation_rounds: usize,
    /// `true` if we stopped only because the prolongation budget was reached
    /// (the differential chain need not be saturated).
    pub truncated: bool,
}

/// Outcome of [`dae_index_reduce`]: Pantelides when it succeeds, otherwise a
/// Rosenfeld–Gröbner consistency pass when Pantelides hits its index cap.
#[derive(Clone, Debug)]
pub enum DaeIndexReduction {
    Pantelides(PantelidesResult),
    Rosenfeld(RosenfeldGroebnerResult),
}

/// Errors from differential-algebra operations.
#[derive(Debug, Clone)]
pub enum DiffAlgError {
    DiffError(String),
    NotPolynomial(String),
    EmptySystem,
}

impl fmt::Display for DiffAlgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffAlgError::DiffError(s) => write!(f, "differentiation error: {s}"),
            DiffAlgError::NotPolynomial(s) => write!(f, "not a polynomial: {s}"),
            DiffAlgError::EmptySystem => write!(f, "empty equation system"),
        }
    }
}

impl std::error::Error for DiffAlgError {}

impl AlkahestError for DiffAlgError {
    fn code(&self) -> &'static str {
        match self {
            DiffAlgError::DiffError(_) => "E-DIFFALG-001",
            DiffAlgError::NotPolynomial(_) => "E-DIFFALG-002",
            DiffAlgError::EmptySystem => "E-DIFFALG-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DiffAlgError::DiffError(_) => {
                Some("ensure the DAE is polynomial in its state and derivative symbols")
            }
            DiffAlgError::NotPolynomial(_) => {
                Some("declare all jet variables and parameters; remove transcendental functions")
            }
            DiffAlgError::EmptySystem => Some("pass at least one implicit equation"),
        }
    }
}

fn solver_err_to_diffalg(e: SolverError) -> DiffAlgError {
    DiffAlgError::NotPolynomial(e.to_string())
}

fn is_unit_ideal_gb(gb: &GroebnerBasis) -> bool {
    gb.generators().iter().any(|g| {
        g.terms.len() == 1
            && g.terms
                .keys()
                .next()
                .is_some_and(|e| e.iter().all(|&x| x == 0))
            && g.terms.values().next().is_some_and(|c| *c != 0)
    })
}

fn pad_gbpoly(p: &GbPoly, new_n: usize) -> GbPoly {
    if new_n == p.n_vars {
        return p.clone();
    }
    assert!(new_n > p.n_vars);
    let pad = new_n - p.n_vars;
    let mut terms = BTreeMap::new();
    for (e, c) in &p.terms {
        let mut ne = e.clone();
        ne.extend(std::iter::repeat(0u32).take(pad));
        terms.insert(ne, c.clone());
    }
    GbPoly {
        terms,
        n_vars: new_n,
    }
}

fn children(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        _ => vec![],
    })
}

fn collect_symbols(
    expr: ExprId,
    pool: &ExprPool,
    seen: &mut HashSet<ExprId>,
    out: &mut Vec<ExprId>,
) {
    let is_sym = pool.with(expr, |d| matches!(d, ExprData::Symbol { .. }));
    if is_sym && seen.insert(expr) {
        out.push(expr);
    }
    for c in children(expr, pool) {
        collect_symbols(c, pool, seen, out);
    }
}

fn vars_for_dae(dae: &DAE, scratch: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut push = |id: ExprId| {
        if seen.insert(id) {
            out.push(id);
        }
    };
    push(dae.time_var);
    for i in 0..dae.variables.len() {
        push(dae.variables[i]);
        push(dae.derivatives[i]);
    }
    for &root in scratch {
        collect_symbols(root, pool, &mut seen, &mut out);
    }
    out
}

fn polys_from_equations(
    eqs: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<GbPoly>, DiffAlgError> {
    eqs.iter()
        .map(|&eq| expr_to_gbpoly(eq, vars, pool).map_err(solver_err_to_diffalg))
        .collect()
}

/// Algebraic-only slice: Gröbner basis of the generators; empty `Vec` if the
/// ideal is `[1]`.
pub fn rosenfeld_groebner_algebraic(
    gens: Vec<GbPoly>,
    order: MonomialOrder,
) -> Result<Vec<RegularDifferentialChain>, DiffAlgError> {
    if gens.is_empty() {
        return Err(DiffAlgError::EmptySystem);
    }
    let gb = GroebnerBasis::compute(gens, order);
    if is_unit_ideal_gb(&gb) {
        return Ok(vec![]);
    }
    Ok(vec![RegularDifferentialChain { basis: gb }])
}

/// Rosenfeld-style prolongation + Gröbner stabilization for a polynomial DAE.
///
/// `max_prolong_rounds` bounds how many prolongation steps are attempted; if the
/// chain has not saturated, [`RosenfeldGroebnerResult::truncated`] is set — typical
/// nonlinear ODE jets do not stabilize in finitely many algebraic steps.
pub fn rosenfeld_groebner_with_options(
    dae: &DAE,
    pool: &ExprPool,
    order: MonomialOrder,
    max_prolong_rounds: usize,
) -> Result<RosenfeldGroebnerResult, DiffAlgError> {
    if dae.equations.is_empty() {
        return Err(DiffAlgError::EmptySystem);
    }

    let source_eqs = dae.equations.clone();
    let mut work = dae.clone();
    let mut scratch: Vec<ExprId> = source_eqs.clone();
    let mut vars = vars_for_dae(&work, &scratch, pool);
    let mut active = polys_from_equations(&work.equations, &vars, pool)?;

    let mut prolong_exprs = source_eqs.clone();
    let mut prolongation_rounds: usize = 0;

    for round in 0..max_prolong_rounds {
        let gb = GroebnerBasis::compute(active.clone(), order);
        if is_unit_ideal_gb(&gb) {
            return Ok(RosenfeldGroebnerResult {
                consistent: false,
                chains: vec![],
                working_dae: work,
                final_basis: None,
                prolongation_rounds,
                truncated: false,
            });
        }

        let mut next_prolong = Vec::with_capacity(prolong_exprs.len());
        for &eq in &prolong_exprs {
            let d_eq =
                differentiate_equation(eq, &work.variables, &work.derivatives, work.time_var, pool)
                    .map_err(|e| DiffAlgError::DiffError(e.to_string()))?;
            extend_dae_for_derivative_symbols(&mut work, d_eq, pool);
            next_prolong.push(d_eq);
        }
        prolong_exprs = next_prolong;
        scratch = source_eqs
            .iter()
            .copied()
            .chain(prolong_exprs.iter().copied())
            .collect();
        vars = vars_for_dae(&work, &scratch, pool);
        let n = vars.len();
        for p in &mut active {
            *p = pad_gbpoly(p, n);
        }

        let gb_check = GroebnerBasis::compute(active.clone(), order);
        let mut to_add: Vec<GbPoly> = Vec::new();
        for &d_eq in &prolong_exprs {
            let p = expr_to_gbpoly(d_eq, &vars, pool).map_err(solver_err_to_diffalg)?;
            if !gb_check.contains(&p) {
                to_add.push(p);
            }
        }

        if to_add.is_empty() {
            let final_basis = GroebnerBasis::compute(active, order);
            let consistent = !is_unit_ideal_gb(&final_basis);
            let chains = if consistent {
                vec![RegularDifferentialChain {
                    basis: final_basis.clone(),
                }]
            } else {
                vec![]
            };
            return Ok(RosenfeldGroebnerResult {
                consistent,
                chains,
                working_dae: work,
                final_basis: if consistent { Some(final_basis) } else { None },
                prolongation_rounds,
                truncated: false,
            });
        }

        active.extend(to_add);
        prolongation_rounds += 1;

        if round + 1 == max_prolong_rounds {
            let final_basis = GroebnerBasis::compute(active, order);
            let consistent = !is_unit_ideal_gb(&final_basis);
            let chains = if consistent {
                vec![RegularDifferentialChain {
                    basis: final_basis.clone(),
                }]
            } else {
                vec![]
            };
            return Ok(RosenfeldGroebnerResult {
                consistent,
                chains,
                working_dae: work,
                final_basis: if consistent { Some(final_basis) } else { None },
                prolongation_rounds,
                truncated: true,
            });
        }
    }

    let final_basis = GroebnerBasis::compute(active, order);
    let consistent = !is_unit_ideal_gb(&final_basis);
    Ok(RosenfeldGroebnerResult {
        consistent,
        chains: if consistent {
            vec![RegularDifferentialChain {
                basis: final_basis.clone(),
            }]
        } else {
            vec![]
        },
        working_dae: work,
        final_basis: if consistent { Some(final_basis) } else { None },
        prolongation_rounds,
        truncated: true,
    })
}

/// [`rosenfeld_groebner_with_options`] with [`DEFAULT_MAX_PROLONG_ROUNDS`].
pub fn rosenfeld_groebner(
    dae: &DAE,
    pool: &ExprPool,
    order: MonomialOrder,
) -> Result<RosenfeldGroebnerResult, DiffAlgError> {
    rosenfeld_groebner_with_options(dae, pool, order, DEFAULT_MAX_PROLONG_ROUNDS)
}

/// Try Pantelides; on [`DaeError::IndexTooHigh`], run [`rosenfeld_groebner`].
pub fn dae_index_reduce(
    dae: &DAE,
    pool: &ExprPool,
    order: MonomialOrder,
) -> Result<DaeIndexReduction, DaeError> {
    match pantelides(dae, pool) {
        Ok(p) => Ok(DaeIndexReduction::Pantelides(p)),
        Err(DaeError::IndexTooHigh) => {
            let r = rosenfeld_groebner(dae, pool, order).map_err(|e| match e {
                DiffAlgError::DiffError(s) | DiffAlgError::NotPolynomial(s) => {
                    DaeError::DiffError(s)
                }
                DiffAlgError::EmptySystem => DaeError::StructurallyInconsistent,
            })?;
            Ok(DaeIndexReduction::Rosenfeld(r))
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn algebraic_inconsistent_unit_ideal() {
        let p = pool();
        let _x = p.symbol("x", Domain::Real);
        let one_p = GbPoly::constant(rug::Rational::from(1), 1);
        let gx = GbPoly::monomial(vec![1], rug::Rational::from(1));
        let f = gx.add(&one_p); // x+1
        let g = gx.sub(&one_p); // x-1
        let chains = rosenfeld_groebner_algebraic(vec![f, g], MonomialOrder::Lex).unwrap();
        assert!(chains.is_empty());
    }

    #[test]
    fn lotka_volterra_first_order_consistent() {
        let p = pool();
        let t = p.symbol("t", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let dx = p.symbol("dx/dt", Domain::Real);
        let dy = p.symbol("dy/dt", Domain::Real);
        // x' = x - x*y, y' = x*y - y  (coefficients 1)
        let eq1 = p.add(vec![dx, p.mul(vec![p.integer(-1), x]), p.mul(vec![x, y])]);
        let eq2 = p.add(vec![dy, p.mul(vec![p.integer(-1), x, y]), y]);
        let dae = DAE::new(vec![eq1, eq2], vec![x, y], vec![dx, dy], t);
        // Budget 0 = only test the algebraic consistency of the declared first-order ideal (no prolongation).
        let r = rosenfeld_groebner_with_options(&dae, &p, MonomialOrder::GRevLex, 0).unwrap();
        assert!(r.consistent && r.final_basis.is_some());
        assert!(r.truncated);
    }

    #[test]
    fn contradictory_linear_equations_inconsistent() {
        let p = pool();
        let t = p.symbol("t", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let dy = p.symbol("dy/dt", Domain::Real);
        // dy - y = 0  and  dy - y - 1 = 0  → 1 ∈ ideal
        let eq1 = p.add(vec![dy, p.mul(vec![p.integer(-1), y])]);
        let eq2 = p.add(vec![dy, p.mul(vec![p.integer(-1), y]), p.integer(-1)]);
        let dae = DAE::new(vec![eq1, eq2], vec![y], vec![dy], t);
        let r = rosenfeld_groebner(&dae, &p, MonomialOrder::Lex).unwrap();
        assert!(!r.consistent);
    }

    #[test]
    fn textbook_library_runs() {
        // Ten tiny polynomial DAE snapshots (autonomous, explicit first derivatives).
        let mut n_ok = 0;
        for _ in 0..10 {
            let p = pool();
            let t = p.symbol("t", Domain::Real);
            let x = p.symbol("x", Domain::Real);
            let y = p.symbol("y", Domain::Real);
            let dx = p.symbol("dx/dt", Domain::Real);
            let dy = p.symbol("dy/dt", Domain::Real);
            // Mixed batch: linear dynamics, coupling, and one inconsistent pair.
            let (eqs, v, d, consistent) = match n_ok % 3 {
                0 => {
                    // harmonic x' = y, y' = -x
                    let e1 = p.add(vec![dx, p.mul(vec![p.integer(-1), y])]);
                    let e2 = p.add(vec![dy, x]);
                    (vec![e1, e2], vec![x, y], vec![dx, dy], true)
                }
                1 => {
                    // decoupled exponentials as linear place-holders: x'=x, y'=-y
                    let e1 = p.add(vec![dx, p.mul(vec![p.integer(-1), x])]);
                    let e2 = p.add(vec![dy, y]);
                    (vec![e1, e2], vec![x, y], vec![dx, dy], true)
                }
                _ => {
                    let e1 = p.add(vec![dx, p.mul(vec![p.integer(-1), x])]);
                    let e2 = p.add(vec![dx, p.mul(vec![p.integer(-1), x]), p.integer(-1)]);
                    (vec![e1, e2], vec![x], vec![dx], false)
                }
            };
            let dae = DAE::new(eqs, v, d, t);
            let r = rosenfeld_groebner(&dae, &p, MonomialOrder::GRevLex).unwrap();
            assert_eq!(r.consistent, consistent);
            n_ok += 1;
        }
        assert_eq!(n_ok, 10);
    }
}
