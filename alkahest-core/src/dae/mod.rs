//! Phase 17 — DAE structural analysis.
//!
//! Represents a Differential-Algebraic Equation (DAE) system and implements
//! the Pantelides algorithm for structural index reduction.
//!
//! A DAE is a system `g_i(t, y, y') = 0` where some equations may be purely
//! algebraic (not involving any derivative).  The *structural index* measures
//! how many times the system must be differentiated to convert it to an ODE.
//!
//! # Pantelides Algorithm
//!
//! The algorithm finds which equations need to be differentiated and creates
//! new equations by differentiating them, until a perfect matching between
//! equations and variables exists.
//!
//! References:
//! - Pantelides (1988) "The consistent initialization of differential-algebraic systems"
//! - Mattsson & Söderlind (1993) "Index reduction in differential-algebraic equations"

use crate::diff::diff;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::collections::HashSet;
use std::fmt;

/// Extend `(variables, derivatives)` when an equation references a derivative
/// symbol that has not yet been promoted to the *state* vector — same rule as
/// the Pantelides inner loop (for higher-order derivative algebra).
pub fn extend_derivative_state_vectors(
    variables: &mut Vec<ExprId>,
    derivatives: &mut Vec<ExprId>,
    new_eq: ExprId,
    pool: &ExprPool,
) {
    for (j, _) in variables.clone().iter().enumerate() {
        let deriv = derivatives[j];
        if structurally_depends(new_eq, deriv, pool) && !variables.contains(&deriv) {
            let d2_name = pool.with(deriv, |d| match d {
                ExprData::Symbol { name, .. } => format!("d{name}/dt"),
                _ => "d?/dt".to_string(),
            });
            let d2 = pool.symbol(&d2_name, Domain::Real);
            variables.push(deriv);
            derivatives.push(d2);
        }
    }
}

/// [`extend_derivative_state_vectors`] on [`DAE::variables`] / [`DAE::derivatives`].
pub fn extend_dae_for_derivative_symbols(dae: &mut DAE, new_eq: ExprId, pool: &ExprPool) {
    extend_derivative_state_vectors(&mut dae.variables, &mut dae.derivatives, new_eq, pool);
}

// ---------------------------------------------------------------------------
// DAE type
// ---------------------------------------------------------------------------

/// A DAE system `g_i(t, y, y') = 0`.
///
/// Equations are in implicit form: `g_i = 0`.
/// Variables are split into:
/// - `alg_vars`: purely algebraic variables (not differentiated anywhere)
/// - `diff_vars`: differential variables with corresponding `derivatives`
#[derive(Clone, Debug)]
pub struct DAE {
    /// Implicit equations `g_i(t, y, y') = 0`
    pub equations: Vec<ExprId>,
    /// Algebraic + differential variables
    pub variables: Vec<ExprId>,
    /// Derivative symbols `dy_i/dt` (for `diff_vars[i]`)
    pub derivatives: Vec<ExprId>,
    /// The independent variable
    pub time_var: ExprId,
    /// Differentiation index (None = not yet computed)
    pub index: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum DaeError {
    DiffError(String),
    IndexTooHigh,
    StructurallyInconsistent,
}

impl fmt::Display for DaeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DaeError::DiffError(s) => write!(f, "differentiation error: {s}"),
            DaeError::IndexTooHigh => write!(f, "DAE structural index too high (> 10)"),
            DaeError::StructurallyInconsistent => write!(f, "DAE is structurally inconsistent"),
        }
    }
}

impl std::error::Error for DaeError {}

impl crate::errors::AlkahestError for DaeError {
    fn code(&self) -> &'static str {
        match self {
            DaeError::DiffError(_) => "E-DAE-001",
            DaeError::IndexTooHigh => "E-DAE-002",
            DaeError::StructurallyInconsistent => "E-DAE-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DaeError::DiffError(_) => Some(
                "ensure all functions in the DAE are differentiable before calling pantelides()",
            ),
            DaeError::IndexTooHigh => {
                Some("DAE index exceeds 10; reformulate the model or use an index-reduction tool")
            }
            DaeError::StructurallyInconsistent => Some(
                "the DAE is structurally inconsistent; check constraint count vs. variable count",
            ),
        }
    }
}

impl DAE {
    /// Create a new DAE.
    ///
    /// `equations` — implicit equations `g_i = 0`
    /// `variables` — all variables (algebraic + differential)
    /// `derivatives` — derivative symbols for each variable
    pub fn new(
        equations: Vec<ExprId>,
        variables: Vec<ExprId>,
        derivatives: Vec<ExprId>,
        time_var: ExprId,
    ) -> Self {
        DAE {
            equations,
            variables,
            derivatives,
            time_var,
            index: None,
        }
    }

    /// Number of equations.
    pub fn n_equations(&self) -> usize {
        self.equations.len()
    }

    /// Number of variables.
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }

    /// Build the structural incidence matrix.
    ///
    /// `incidence[i][j]` is `true` if equation `i` structurally depends on
    /// variable `j` or its derivative.
    pub fn incidence_matrix(&self, pool: &ExprPool) -> Vec<Vec<bool>> {
        let m = self.equations.len();
        let n = self.variables.len();
        let mut inc = vec![vec![false; n]; m];
        for (i, &eq) in self.equations.iter().enumerate() {
            for (j, &var) in self.variables.iter().enumerate() {
                let deriv = self.derivatives[j];
                if structurally_depends(eq, var, pool) || structurally_depends(eq, deriv, pool) {
                    inc[i][j] = true;
                }
            }
        }
        inc
    }

    /// Display the DAE.
    pub fn display(&self, pool: &ExprPool) -> String {
        self.equations
            .iter()
            .map(|&eq| format!("  {} = 0", pool.display(eq)))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ---------------------------------------------------------------------------
// Pantelides algorithm for structural index reduction
// ---------------------------------------------------------------------------

/// Result of applying the Pantelides algorithm.
#[derive(Clone, Debug)]
pub struct PantelidesResult {
    /// The index-reduced DAE (index ≤ 1)
    pub reduced_dae: DAE,
    /// Number of differentiation steps applied
    pub differentiation_steps: usize,
    /// Which original equations were differentiated (and how many times)
    pub sigma: Vec<usize>, // sigma[i] = number of times equation i was differentiated
}

/// Apply the Pantelides algorithm to reduce a DAE to differentiation index ≤ 1.
///
/// Returns the reduced DAE together with bookkeeping information.
pub fn pantelides(dae: &DAE, pool: &ExprPool) -> Result<PantelidesResult, DaeError> {
    let max_iter = 10;

    let mut equations = dae.equations.clone();
    let mut variables = dae.variables.clone();
    let mut derivatives = dae.derivatives.clone();
    let mut sigma = vec![0usize; equations.len()];
    let mut total_steps = 0;

    for iteration in 0..max_iter {
        // Build incidence structure
        let n_eq = equations.len();
        let n_var = variables.len();
        let inc = incidence(&equations, &variables, &derivatives, pool);

        // Find maximum matching using Hopcroft-Karp
        let matching = maximum_matching(&inc, n_eq, n_var);

        // Check if perfect matching exists
        let unmatched_eqs: Vec<usize> = (0..n_eq)
            .filter(|&i| matching.eq_to_var[i].is_none())
            .collect();

        if unmatched_eqs.is_empty() {
            // Perfect matching found → index ≤ 1
            let mut reduced = DAE::new(equations, variables, derivatives, dae.time_var);
            reduced.index = Some(iteration);
            return Ok(PantelidesResult {
                reduced_dae: reduced,
                differentiation_steps: total_steps,
                sigma,
            });
        }

        // Differentiate unmatched equations
        for &eq_idx in &unmatched_eqs {
            let new_eq = differentiate_equation(
                equations[eq_idx],
                &variables,
                &derivatives,
                dae.time_var,
                pool,
            )
            .map_err(|e| DaeError::DiffError(e.to_string()))?;
            equations.push(new_eq);
            sigma.push(sigma[eq_idx] + 1);
            total_steps += 1;

            extend_derivative_state_vectors(&mut variables, &mut derivatives, new_eq, pool);
        }
    }

    Err(DaeError::IndexTooHigh)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct Matching {
    eq_to_var: Vec<Option<usize>>,
    #[allow(dead_code)]
    var_to_eq: Vec<Option<usize>>,
}

/// Build an incidence list: `result[i]` = set of variable indices that equation `i` depends on.
fn incidence(
    equations: &[ExprId],
    variables: &[ExprId],
    derivatives: &[ExprId],
    pool: &ExprPool,
) -> Vec<Vec<usize>> {
    equations
        .iter()
        .map(|&eq| {
            variables
                .iter()
                .zip(derivatives.iter())
                .enumerate()
                .filter(|(_, (&var, &deriv))| {
                    structurally_depends(eq, var, pool) || structurally_depends(eq, deriv, pool)
                })
                .map(|(j, _)| j)
                .collect()
        })
        .collect()
}

/// Augmenting path search for maximum bipartite matching (DFS).
fn augment(
    eq: usize,
    adj: &[Vec<usize>],
    var_to_eq: &mut Vec<Option<usize>>,
    visited: &mut HashSet<usize>,
) -> bool {
    for &var in &adj[eq] {
        if visited.contains(&var) {
            continue;
        }
        visited.insert(var);
        if var_to_eq[var].is_none() || augment(var_to_eq[var].unwrap(), adj, var_to_eq, visited) {
            var_to_eq[var] = Some(eq);
            return true;
        }
    }
    false
}

fn maximum_matching(adj: &[Vec<usize>], n_eq: usize, n_var: usize) -> Matching {
    let mut var_to_eq: Vec<Option<usize>> = vec![None; n_var];
    for eq in 0..n_eq {
        let mut visited = HashSet::new();
        augment(eq, adj, &mut var_to_eq, &mut visited);
    }
    let mut eq_to_var = vec![None; n_eq];
    for (var, &opt_eq) in var_to_eq.iter().enumerate() {
        if let Some(eq) = opt_eq {
            eq_to_var[eq] = Some(var);
        }
    }
    Matching {
        eq_to_var,
        var_to_eq,
    }
}

/// Differentiate `equation` with respect to time, using `d(var)/dt = deriv`.
pub(crate) fn differentiate_equation(
    equation: ExprId,
    variables: &[ExprId],
    derivatives: &[ExprId],
    time_var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, crate::diff::diff_impl::DiffError> {
    // d(g)/dt = Σ_i (∂g/∂y_i) * (dy_i/dt)  +  ∂g/∂t
    // Use chain rule symbolically
    let mut terms: Vec<ExprId> = Vec::new();

    // ∂g/∂t
    let dg_dt = diff(equation, time_var, pool)?.value;
    if dg_dt != pool.integer(0_i32) {
        terms.push(dg_dt);
    }

    // For each variable y_i: (∂g/∂y_i) * (dy_i/dt)
    for (&var, &deriv) in variables.iter().zip(derivatives.iter()) {
        let dg_dyi = diff(equation, var, pool)?.value;
        if dg_dyi != pool.integer(0_i32) {
            let term = pool.mul(vec![dg_dyi, deriv]);
            terms.push(term);
        }
        // Also differentiate w.r.t. the derivative (for higher-index terms)
        let dg_ddyi = diff(equation, deriv, pool)?.value;
        if dg_ddyi != pool.integer(0_i32) {
            // d(dy_i/dt)/dt is a new symbol — use the naming convention
            let d2_name = pool.with(deriv, |d| match d {
                ExprData::Symbol { name, .. } => format!("d{name}/dt"),
                _ => "d?/dt".to_string(),
            });
            let d2 = pool.symbol(&d2_name, Domain::Real);
            let term = pool.mul(vec![dg_ddyi, d2]);
            terms.push(term);
        }
    }

    let result = match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    };
    Ok(simplify(result, pool).value)
}

/// True if `expr` structurally contains `var` as a sub-expression.
pub fn structurally_depends(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    let children = pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::BigO(inner) => vec![*inner],
        _ => vec![],
    });
    children
        .into_iter()
        .any(|c| structurally_depends(c, var, pool))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn ode_is_index_0() {
        // An explicit ODE y' - f(y) = 0  has differentiation index 0 (or 1 in some conventions)
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let dy = pool.symbol("dy/dt", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let neg_y = pool.mul(vec![pool.integer(-1_i32), y]);
        // Equation: dy/dt - y = 0  →  dy - y
        let eq = pool.add(vec![dy, neg_y]);
        let dae = DAE::new(vec![eq], vec![y], vec![dy], t);
        let result = pantelides(&dae, &pool).unwrap();
        assert_eq!(result.differentiation_steps, 0);
    }

    #[test]
    fn incidence_matrix_correct() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let dx = pool.symbol("dx/dt", Domain::Real);
        let dy = pool.symbol("dy/dt", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        // g1 = x + y,  g2 = dx + y
        let g1 = pool.add(vec![x, y]);
        let g2 = pool.add(vec![dx, y]);
        let dae = DAE::new(vec![g1, g2], vec![x, y], vec![dx, dy], t);
        let inc = dae.incidence_matrix(&pool);
        // g1 depends on x (j=0) and y (j=1)
        assert!(inc[0][0]);
        assert!(inc[0][1]);
        // g2 depends on dx (structurally related to j=0) and y (j=1)
        assert!(inc[1][0]); // dx is deriv of x
        assert!(inc[1][1]); // y
    }

    #[test]
    fn structurally_depends_nested() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let expr = pool.add(vec![sin_x, y]);
        assert!(structurally_depends(expr, x, &pool));
        assert!(structurally_depends(expr, y, &pool));
    }

    #[test]
    fn differentiate_equation_linear() {
        // g(x, y) = x + y,  variables = [x, y], derivatives = [dx, dy]
        // dg/dt = dx + dy
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let dx = pool.symbol("dx/dt", Domain::Real);
        let dy = pool.symbol("dy/dt", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let eq = pool.add(vec![x, y]);
        let result = differentiate_equation(eq, &[x, y], &[dx, dy], t, &pool).unwrap();
        // Should give dx + dy (in some order)
        let s = pool.display(result).to_string();
        assert!(s.contains("dx") || s.contains("dy"), "got: {s}");
    }
}
