//! Polynomial system solver via Gröbner bases.
//!
//! # V2-11 — Regular chains
//!
//! [`regular_chains::triangularize`] exposes a Lex-basis triangular decomposition
//! (with optional bottom-variable factor splitting).  On a triangular back-sub
//! stall, [`solve_polynomial_system`] retries using an extracted regular chain
//! from the same Gröbner basis.
//!
//! # V2-14 — Homotopy continuation (numerical algebraic geometry)
//!
//! [`homotopy::solve_numerical`] runs a total-degree homotopy in `ℂⁿ` (Bézout
//! start system) and yields real roots with Smale-style checks and `ArbBall`
//! enclosures — see module documentation for limitations on **deficient**
//! systems.
//!
//! # V1-4 — Symbolic triangular solving (`solve_polynomial_system`)
//!
//! Inputs are polynomial equations (`lhs - rhs = 0`), variables, and an
//! `ExprPool`; outputs are symbolic `ExprId` values (may include `sqrt`),
//! or `SolutionSet::Parametric` / `SolutionSet::NoSolution`.

pub mod diophantine;
pub mod homotopy;
pub mod regular_chains;

pub use regular_chains::{
    extract_regular_chain_from_basis, main_variable_recursive, triangularize, RegularChain,
};

pub use homotopy::{solve_numerical, CertifiedPoint, HomotopyError, HomotopyOpts};

pub use diophantine::{diophantine, DiophantineError, DiophantineSolution};

use crate::errors::AlkahestError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::groebner::{GbPoly, GroebnerBasis, MonomialOrder};
use rug::{ops::NegAssign, Rational};
use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single solution point: `ExprId`s parallel to the `vars` passed to the
/// solver.  Values may be rationals (for linear systems) or symbolic
/// expressions involving `sqrt` (for quadratic elimination tails).
pub type Solution = Vec<ExprId>;

/// The result of `solve_polynomial_system`.
pub enum SolutionSet {
    /// Finitely many solutions (each is a `Vec<ExprId>` parallel to `vars`).
    Finite(Vec<Solution>),
    /// Infinitely many solutions; the Gröbner basis is returned for downstream use.
    Parametric(GroebnerBasis),
    /// No solution (ideal = ⟨1⟩).
    NoSolution,
}

/// Errors from the polynomial system solver.
#[derive(Debug, Clone)]
pub enum SolverError {
    /// An equation is not a polynomial in the given variables.
    NotPolynomial(String),
    /// Back-substitution would require solving a degree > 2 univariate — not yet
    /// implemented for general algebraic numbers.
    HighDegree(usize),
    /// Number of equations doesn't match number of variables (for zero-dim check).
    ShapeMismatch,
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::NotPolynomial(s) => write!(f, "not a polynomial: {s}"),
            SolverError::HighDegree(d) => write!(
                f,
                "back-substitution requires solving a degree-{d} univariate polynomial \
                 (only degree ≤ 2 is currently supported)"
            ),
            SolverError::ShapeMismatch => write!(
                f,
                "number of equations must equal number of variables for zero-dimensional solving"
            ),
        }
    }
}

impl std::error::Error for SolverError {}

impl AlkahestError for SolverError {
    fn code(&self) -> &'static str {
        match self {
            SolverError::NotPolynomial(_) => "E-SOLVE-001",
            SolverError::HighDegree(_) => "E-SOLVE-002",
            SolverError::ShapeMismatch => "E-SOLVE-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            SolverError::NotPolynomial(_) => Some(
                "ensure all equations are polynomial in the declared variables; \
                 transcendental functions are not supported",
            ),
            SolverError::HighDegree(_) => Some(
                "degree > 2 univariate solving is not yet implemented; \
                 the Gröbner basis is still returned for manual inspection",
            ),
            SolverError::ShapeMismatch => {
                Some("provide one equation per variable for zero-dimensional system solving")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Expr → GbPoly conversion
// ---------------------------------------------------------------------------

/// Convert an `Expr` (which must be a polynomial in `vars`) to a `GbPoly`
/// with rational coefficients.  The variable order in the exponent vector
/// follows the order of `vars`.
pub fn expr_to_gbpoly(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<GbPoly, SolverError> {
    let n = vars.len();
    expr_to_gbpoly_rec(expr, vars, n, pool)
}

fn expr_to_gbpoly_rec(
    expr: ExprId,
    vars: &[ExprId],
    n_vars: usize,
    pool: &ExprPool,
) -> Result<GbPoly, SolverError> {
    if let Some(idx) = vars.iter().position(|&v| v == expr) {
        let mut exp = vec![0u32; n_vars];
        exp[idx] = 1;
        let mut terms = BTreeMap::new();
        terms.insert(exp, rug::Rational::from(1));
        return Ok(GbPoly { terms, n_vars });
    }

    enum Node {
        Var(usize),
        IntConst(rug::Integer),
        RatConst(Rational),
        FloatConst(f64),
        FreeSymbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String),
        Other,
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Integer(n) => Node::IntConst(n.0.clone()),
        ExprData::Rational(r) => Node::RatConst(r.0.clone()),
        ExprData::Float(f) => Node::FloatConst(f.inner.to_f64()),
        ExprData::Symbol { name, .. } => {
            if let Some(idx) = vars.iter().position(|&v| v == expr) {
                Node::Var(idx)
            } else {
                Node::FreeSymbol(name.clone())
            }
        }
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
        ExprData::Func { name, .. } => Node::Func(name.clone()),
        _ => Node::Other,
    });

    match node {
        Node::Var(idx) => {
            let mut exp = vec![0u32; n_vars];
            exp[idx] = 1;
            let mut terms = BTreeMap::new();
            terms.insert(exp, Rational::from(1));
            Ok(GbPoly { terms, n_vars })
        }
        Node::IntConst(n) => Ok(GbPoly::constant(Rational::from(n), n_vars)),
        Node::RatConst(r) => Ok(GbPoly::constant(r, n_vars)),
        Node::FloatConst(v) => {
            let r = Rational::from_f64(v).unwrap_or_else(|| Rational::from(0));
            Ok(GbPoly::constant(r, n_vars))
        }
        Node::FreeSymbol(name) => Err(SolverError::NotPolynomial(format!(
            "free symbol '{name}' not in variable list"
        ))),
        Node::Add(args) => {
            let mut result = GbPoly::zero(n_vars);
            for a in args {
                let p = expr_to_gbpoly_rec(a, vars, n_vars, pool)?;
                result = result.add(&p);
            }
            Ok(result)
        }
        Node::Mul(args) => {
            let mut result = GbPoly::constant(Rational::from(1), n_vars);
            for a in args {
                let p = expr_to_gbpoly_rec(a, vars, n_vars, pool)?;
                result = result.mul(&p);
            }
            Ok(result)
        }
        Node::Pow(base, exp_id) => {
            let exp_node = pool.with(exp_id, |d| match d {
                ExprData::Integer(n) => Some(n.0.clone()),
                _ => None,
            });
            match exp_node {
                Some(n) => {
                    let n_val = n.to_i64().unwrap_or(-1);
                    if n_val < 0 {
                        return Err(SolverError::NotPolynomial(format!(
                            "negative exponent {n_val} in polynomial"
                        )));
                    }
                    let base_poly = expr_to_gbpoly_rec(base, vars, n_vars, pool)?;
                    let mut result = GbPoly::constant(Rational::from(1), n_vars);
                    let mut cur = base_poly;
                    let mut rem = n_val as u64;
                    while rem > 0 {
                        if rem & 1 == 1 {
                            result = result.mul(&cur);
                        }
                        let cur2 = cur.clone();
                        cur = cur.mul(&cur2);
                        rem >>= 1;
                    }
                    Ok(result)
                }
                None => Err(SolverError::NotPolynomial(
                    "symbolic or non-integer exponent".to_string(),
                )),
            }
        }
        Node::Func(name) => Err(SolverError::NotPolynomial(format!(
            "function '{name}' is not a polynomial"
        ))),
        Node::Other => Err(SolverError::NotPolynomial(
            "unsupported expression node".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// ExprId builders
// ---------------------------------------------------------------------------

fn rational_to_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    let (num, den) = r.clone().into_numer_denom();
    if den == 1 {
        pool.integer(num)
    } else {
        pool.rational(num, den)
    }
}

fn neg_expr(e: ExprId, pool: &ExprPool) -> ExprId {
    let neg_one = pool.integer(rug::Integer::from(-1));
    pool.mul(vec![neg_one, e])
}

fn div_expr(num: ExprId, den: ExprId, pool: &ExprPool) -> ExprId {
    // num / den = num * den^(-1)
    let neg_one = pool.integer(rug::Integer::from(-1));
    let inv_den = pool.pow(den, neg_one);
    pool.mul(vec![num, inv_den])
}

/// Is this ExprId structurally the integer zero?
fn is_syntactic_zero(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Integer(n) if n.0 == 0))
}

/// Extract the coefficient of `var_idx^k` in `poly`, substituting
/// already-solved vars (`assigned[i] = Some(ExprId)`) into the remaining
/// factors.  Unsolved vars that happen to appear (other than `var_idx`)
/// are left as their original `ExprId` variable — callers should only
/// invoke this when the generator involves exactly one unsolved variable
/// at `var_idx`.
fn extract_coeff_in_var(
    poly: &GbPoly,
    var_idx: usize,
    k: u32,
    vars: &[ExprId],
    assigned: &[Option<ExprId>],
    pool: &ExprPool,
) -> ExprId {
    let mut sum_terms: Vec<ExprId> = Vec::new();
    for (exp, coeff) in &poly.terms {
        let e_k = exp.get(var_idx).copied().unwrap_or(0);
        if e_k != k {
            continue;
        }
        let mut factors: Vec<ExprId> = Vec::new();
        if *coeff != 1 {
            factors.push(rational_to_expr(coeff, pool));
        }
        for (i, &e) in exp.iter().enumerate() {
            if i == var_idx || e == 0 {
                continue;
            }
            let base = assigned
                .get(i)
                .and_then(|o| o.as_ref())
                .copied()
                .unwrap_or(vars[i]);
            if e == 1 {
                factors.push(base);
            } else {
                let exp_id = pool.integer(rug::Integer::from(e));
                factors.push(pool.pow(base, exp_id));
            }
        }
        let term = match factors.len() {
            0 => pool.integer(rug::Integer::from(1)),
            1 => factors[0],
            _ => pool.mul(factors),
        };
        // Re-apply the rational coefficient sign if it wasn't a 1 above
        let signed = if *coeff == 1 {
            term
        } else {
            // Already included in factors
            term
        };
        sum_terms.push(signed);
    }
    match sum_terms.len() {
        0 => pool.integer(rug::Integer::from(0)),
        1 => sum_terms[0],
        _ => pool.add(sum_terms),
    }
}

// ---------------------------------------------------------------------------
// Univariate solver (symbolic output, ℚ-only and symbolic paths)
// ---------------------------------------------------------------------------

/// Solve `a₀ + a₁·x + a₂·x² = 0` where each `aᵢ` is an already-substituted
/// `ExprId`.  Returns a `Vec<ExprId>` of roots (symbolic).  Degree is
/// inferred from `coeffs.len()`; higher-degree terms must be syntactic-zero
/// (the caller trims first).  A degree-2 equation always yields two roots
/// (symbolically distinct even if discriminant = 0; the caller can dedupe
/// numerically if desired).
fn solve_univariate_symbolic(
    coeffs: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<ExprId>, SolverError> {
    let mut degree = 0usize;
    for (i, &c) in coeffs.iter().enumerate() {
        if !is_syntactic_zero(c, pool) {
            degree = i;
        }
    }
    match degree {
        0 => {
            // Constant equation.  If coefficient is zero it's trivially
            // satisfied (0 = 0) — shouldn't happen for a proper generator.
            // Otherwise it's 0 = nonzero → no solution, but we signal that
            // by returning empty (the caller treats this as contradiction).
            if coeffs.is_empty() || is_syntactic_zero(coeffs[0], pool) {
                Ok(vec![])
            } else {
                Ok(vec![])
            }
        }
        1 => {
            let a = coeffs[1];
            let b = coeffs[0];
            let neg_b = neg_expr(b, pool);
            Ok(vec![div_expr(neg_b, a, pool)])
        }
        2 => {
            let a = coeffs[2];
            let b = coeffs[1];
            let c = coeffs[0];
            let two = pool.integer(rug::Integer::from(2));
            let four = pool.integer(rug::Integer::from(4));
            let b2 = pool.pow(b, two);
            let four_ac = pool.mul(vec![four, a, c]);
            let neg_four_ac = neg_expr(four_ac, pool);
            let disc = pool.add(vec![b2, neg_four_ac]);
            let sqrt_disc = pool.func("sqrt", vec![disc]);
            let two_b = pool.integer(rug::Integer::from(2));
            let two_a = pool.mul(vec![two_b, a]);
            let neg_b = neg_expr(b, pool);
            let root_plus = div_expr(pool.add(vec![neg_b, sqrt_disc]), two_a, pool);
            let neg_sqrt = neg_expr(sqrt_disc, pool);
            let root_minus = div_expr(pool.add(vec![neg_b, neg_sqrt]), two_a, pool);
            Ok(vec![root_plus, root_minus])
        }
        d => Err(SolverError::HighDegree(d)),
    }
}

/// Retained Rational-only univariate solver (used for the trivial-ideal
/// check and for rational-root shortcuts).  Returns `None` if the poly
/// has any irrational root, `Some(Vec<Rational>)` when every root is in ℚ.
#[allow(dead_code)]
fn try_solve_univariate_rational(p: &GbPoly, var_idx: usize) -> Option<Vec<Rational>> {
    let mut coeffs: BTreeMap<u32, Rational> = BTreeMap::new();
    for (exp, coeff) in &p.terms {
        let deg = exp.get(var_idx).copied().unwrap_or(0);
        let entry = coeffs.entry(deg).or_insert_with(|| Rational::from(0));
        *entry += coeff.clone();
    }
    coeffs.retain(|_, v| *v != 0);
    let degree = coeffs.keys().max().copied().unwrap_or(0);
    match degree {
        0 => Some(vec![]),
        1 => {
            let a = coeffs.get(&1).cloned().unwrap_or_else(|| Rational::from(0));
            let b = coeffs.get(&0).cloned().unwrap_or_else(|| Rational::from(0));
            let mut neg_b = b;
            neg_b.neg_assign();
            Some(vec![Rational::from(neg_b / a)])
        }
        2 => {
            let a = coeffs.get(&2).cloned().unwrap_or_else(|| Rational::from(0));
            let b = coeffs.get(&1).cloned().unwrap_or_else(|| Rational::from(0));
            let c = coeffs.get(&0).cloned().unwrap_or_else(|| Rational::from(0));
            let b2 = Rational::from(&b * &b);
            let four_ac = Rational::from(Rational::from(4) * &a * &c);
            let disc = Rational::from(b2 - four_ac);
            if disc < 0 {
                return Some(vec![]);
            }
            let disc_numer = disc.numer().clone();
            let disc_denom = disc.denom().clone();
            let (sn, rem_n) = disc_numer.sqrt_rem(rug::Integer::new());
            let (sd, rem_d) = disc_denom.sqrt_rem(rug::Integer::new());
            if rem_n == 0 && rem_d == 0 {
                let sqrt_disc = Rational::from((sn, sd));
                let two_a = Rational::from(Rational::from(2) * &a);
                let mut neg_b = b;
                neg_b.neg_assign();
                let root1 = Rational::from((Rational::from(&neg_b + &sqrt_disc)) / &two_a);
                let root2 = Rational::from((Rational::from(neg_b - sqrt_disc)) / &two_a);
                if root1 == root2 {
                    Some(vec![root1])
                } else {
                    Some(vec![root1, root2])
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------

/// Return the set of variable indices that actually appear (with positive
/// exponent in any term) in `poly`.
fn active_vars(poly: &GbPoly, n_vars: usize) -> Vec<usize> {
    (0..n_vars)
        .filter(|&i| {
            poly.terms
                .keys()
                .any(|e| e.get(i).copied().unwrap_or(0) > 0)
        })
        .collect()
}

fn max_degree_in_var(poly: &GbPoly, var_idx: usize) -> u32 {
    poly.terms
        .keys()
        .map(|e| e.get(var_idx).copied().unwrap_or(0))
        .max()
        .unwrap_or(0)
}

/// Given the current partial assignment, find a generator solvable in
/// exactly one unsolved variable.  Returns `(var_idx, gen, max_deg)`.
fn find_solvable<'a>(
    gens: &'a [GbPoly],
    assigned: &[Option<ExprId>],
    n_vars: usize,
) -> Option<(usize, &'a GbPoly, u32)> {
    for g in gens {
        let active = active_vars(g, n_vars);
        let unsolved: Vec<usize> = active
            .iter()
            .copied()
            .filter(|&i| assigned[i].is_none())
            .collect();
        if unsolved.len() == 1 {
            let var_idx = unsolved[0];
            let max_deg = max_degree_in_var(g, var_idx);
            if max_deg > 0 {
                return Some((var_idx, g, max_deg));
            }
        }
    }
    None
}

/// Lex-order backsolve over a fixed generator list (full Gröbner basis or a
/// triangular subset).
enum BacksolveOutcome {
    Finite(Vec<Solution>),
    /// No triangular step applied (`find_solvable` stuck) — caller may retry a smaller set.
    Stuck,
    NoSolution,
}

fn try_backsolve_generators(
    gens: &[GbPoly],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<BacksolveOutcome, SolverError> {
    let n_vars = vars.len();
    let mut partials: Vec<Vec<Option<ExprId>>> = vec![vec![None; n_vars]];

    for _ in 0..n_vars {
        let mut new_partials = Vec::new();
        for partial in &partials {
            let solvable = find_solvable(gens, partial, n_vars);
            let (var_idx, gen, max_deg) = match solvable {
                Some(t) => t,
                None => return Ok(BacksolveOutcome::Stuck),
            };
            if max_deg > 2 {
                return Err(SolverError::HighDegree(max_deg as usize));
            }
            let coeffs: Vec<ExprId> = (0..=max_deg)
                .map(|k| extract_coeff_in_var(gen, var_idx, k, vars, partial, pool))
                .collect();
            let roots = solve_univariate_symbolic(&coeffs, pool)?;
            if roots.is_empty() {
                continue;
            }
            for root in roots {
                let mut np = partial.clone();
                np[var_idx] = Some(root);
                new_partials.push(np);
            }
        }
        partials = new_partials;
        if partials.is_empty() {
            return Ok(BacksolveOutcome::NoSolution);
        }
    }

    let solutions: Vec<Solution> = partials
        .into_iter()
        .map(|p| {
            p.into_iter()
                .map(|o| o.expect("all vars assigned"))
                .collect()
        })
        .collect();

    Ok(BacksolveOutcome::Finite(solutions))
}

/// Solve a zero-dimensional polynomial system.
///
/// `equations` — list of `ExprId` each representing `p(vars) = 0`.
/// `vars` — list of symbolic variables in the order used for `GbPoly` exponent vectors.
///
/// Returns a [`SolutionSet`] with symbolic `ExprId` values for each solution.
pub fn solve_polynomial_system(
    equations: Vec<ExprId>,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<SolutionSet, SolverError> {
    let n_vars = vars.len();

    let mut polys: Vec<GbPoly> = Vec::with_capacity(equations.len());
    for eq in &equations {
        polys.push(expr_to_gbpoly(*eq, &vars, pool)?);
    }

    let gb = GroebnerBasis::compute(polys, MonomialOrder::Lex);
    let gens = gb.generators();

    // Trivial ideal ⟨1⟩ → no solution.
    if gens.len() == 1
        && gens[0].terms.len() == 1
        && gens[0].leading_exp(MonomialOrder::Lex) == Some(vec![0u32; n_vars])
    {
        return Ok(SolutionSet::NoSolution);
    }

    match try_backsolve_generators(gens, &vars, pool)? {
        BacksolveOutcome::Finite(solutions) => Ok(SolutionSet::Finite(solutions)),
        BacksolveOutcome::NoSolution => Ok(SolutionSet::NoSolution),
        BacksolveOutcome::Stuck => {
            let chain = extract_regular_chain_from_basis(gens, n_vars, MonomialOrder::Lex);
            if chain.polys.is_empty() {
                return Ok(SolutionSet::Parametric(gb));
            }
            match try_backsolve_generators(&chain.polys, &vars, pool)? {
                BacksolveOutcome::Finite(solutions) => Ok(SolutionSet::Finite(solutions)),
                _ => Ok(SolutionSet::Parametric(gb)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::{Domain, ExprPool};
    use std::collections::HashMap;

    fn eval_no_env(e: ExprId, pool: &ExprPool) -> f64 {
        eval_interp(e, &HashMap::new(), pool).expect("numeric eval")
    }

    fn has_numeric_pair(sols: &[Solution], pool: &ExprPool, expected: &[(f64, f64)]) -> bool {
        let tol = 1e-10;
        expected.iter().all(|(ex, ey)| {
            sols.iter().any(|s| {
                let x = eval_no_env(s[0], pool);
                let y = eval_no_env(s[1], pool);
                (x - ex).abs() < tol && (y - ey).abs() < tol
            })
        })
    }

    #[test]
    fn linear_system() {
        // x + y - 1 = 0, x - y = 0  →  x = 1/2, y = 1/2
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let eq1 = pool.add(vec![x, y, neg_one]);
        let eq2 = pool.add(vec![x, pool.mul(vec![neg_one, y])]);
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        if let SolutionSet::Finite(sols) = result {
            assert!(has_numeric_pair(&sols, &pool, &[(0.5, 0.5)]));
        } else {
            panic!("expected finite solution set");
        }
    }

    #[test]
    fn univariate_quadratic() {
        // x² - 1 = 0  →  x = ±1
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let eq = pool.add(vec![x2, neg_one]);
        let result = solve_polynomial_system(vec![eq], vec![x], &pool).unwrap();
        if let SolutionSet::Finite(sols) = result {
            let vals: Vec<f64> = sols.iter().map(|s| eval_no_env(s[0], &pool)).collect();
            assert!(vals.iter().any(|v| (v - 1.0).abs() < 1e-10));
            assert!(vals.iter().any(|v| (v + 1.0).abs() < 1e-10));
        } else {
            panic!("expected finite solution set");
        }
    }

    #[test]
    fn circle_line_intersection() {
        // x² + y² - 1 = 0,  y - x = 0  →  x = y = ±√2/2
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let two = pool.integer(2_i32);
        let x2 = pool.pow(x, two);
        let y2 = pool.pow(y, two);
        // x² + y² - 1
        let eq1 = pool.add(vec![x2, y2, neg_one]);
        // y - x
        let eq2 = pool.add(vec![y, pool.mul(vec![neg_one, x])]);
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        if let SolutionSet::Finite(sols) = result {
            assert_eq!(
                sols.len(),
                2,
                "expected exactly 2 solutions, got {}",
                sols.len()
            );
            let root = (0.5_f64).sqrt(); // √2/2
            assert!(has_numeric_pair(
                &sols,
                &pool,
                &[(root, root), (-root, -root)]
            ));
        } else {
            panic!("expected finite solution set");
        }
    }

    #[test]
    fn no_solution_inconsistent() {
        // x = 0 and x = 1 simultaneously → ⟨1⟩ ideal
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let eq1 = x; // x = 0
        let eq2 = pool.add(vec![x, neg_one]); // x - 1 = 0
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x], &pool).unwrap();
        assert!(matches!(result, SolutionSet::NoSolution));
    }

    #[test]
    fn parabola_and_line() {
        // y - x² = 0,  y - x = 0  →  x(x-1)=0 → (0,0) and (1,1)
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let two = pool.integer(2_i32);
        let x2 = pool.pow(x, two);
        // y - x²
        let eq1 = pool.add(vec![y, pool.mul(vec![neg_one, x2])]);
        // y - x
        let eq2 = pool.add(vec![y, pool.mul(vec![neg_one, x])]);
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        if let SolutionSet::Finite(sols) = result {
            assert_eq!(sols.len(), 2);
            assert!(has_numeric_pair(&sols, &pool, &[(0.0, 0.0), (1.0, 1.0)]));
        } else {
            panic!("expected finite solution set");
        }
    }
}
