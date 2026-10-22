//! Phase 17 — DAE structural analysis.
//!
//! Represents a Differential-Algebraic Equation (DAE) system and implements
//! the Pantelides algorithm for structural index reduction.
//!
//! A DAE is a system `g_i(t, y, y') = 0` where some equations may be purely
//! algebraic (not involving any derivative).  The *structural index* measures
//! how many times the system must be differentiated to convert it to an ODE.
//!
//! # Pantelides / Σ-method
//!
//! [`pantelides`] decides how many times each equation must be differentiated
//! so that the system determines the highest derivative of every unknown,
//! returns the system with those derivatives appended, and reports the
//! differentiation index.  The analysis matches equations to unknowns on
//! their *highest* derivatives (Pryce's Σ-method, equivalent to Pantelides'
//! algorithm — see the `sigma` module) and then checks that the Jacobian
//! with respect to those derivatives is non-singular, refusing otherwise.
//!
//! References:
//! - Pantelides (1988) "The consistent initialization of differential-algebraic systems"
//! - Pryce (2001) "A simple structural analysis method for DAEs", BIT 41
//! - Mattsson & Söderlind (1993) "Index reduction in differential-algebraic equations"

use crate::diff::diff;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::fmt;

mod sigma;
pub(crate) use sigma::{next_jet_symbol, normalized_jets};

/// Extend `(variables, derivatives)` when an equation references a derivative
/// symbol that has not yet been promoted to the *state* vector — same rule as
/// the Pantelides inner loop (for higher-order derivative algebra).
pub fn extend_derivative_state_vectors(
    variables: &mut Vec<ExprId>,
    derivatives: &mut Vec<ExprId>,
    new_eq: ExprId,
    pool: &ExprPool,
) {
    // A trailing variable with no listed derivative is algebraic: give it the
    // synthesised `d{name}/dt` so the two vectors stay parallel.
    while derivatives.len() < variables.len() {
        let v = variables[derivatives.len()];
        derivatives.push(next_jet_symbol(v, pool));
    }
    for j in 0..variables.len() {
        let deriv = derivatives[j];
        if variables.contains(&deriv) {
            continue;
        }
        let d2_name = pool.with(deriv, |d| match d {
            ExprData::Symbol { name, .. } => format!("d{name}/dt"),
            _ => "d?/dt".to_string(),
        });
        let d2 = pool.symbol(&d2_name, Domain::Real);
        // `deriv` becomes a state when the equation mentions it — or when it
        // mentions `d(deriv)/dt`, which means `deriv` has *already* been
        // differentiated once.  Without the second test the chain stops: the
        // jet is in the equation but has no successor, so the next
        // differentiation treats it as a constant and drops its contribution,
        // asserting a relation the system does not imply.
        if structurally_depends(new_eq, deriv, pool) || structurally_depends(new_eq, d2, pool) {
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

/// Refusals from the checked DAE entry points ([`pantelides_checked`],
/// [`DAE::try_new`]).
///
/// [`DaeError`] is exhaustive and cannot grow, so the refusals added with the
/// Σ-method live here; [`DaeStructureError::Dae`] carries the old ones with
/// their old codes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DaeStructureError {
    /// A [`DaeError`] (E-DAE-001 … E-DAE-003).
    Dae(DaeError),
    /// The DAE itself is malformed: more derivative symbols than variables, a
    /// variable or derivative that is not a symbol, or duplicates (E-DAE-004).
    MalformedDae(String),
    /// The structural analysis succeeded but the Jacobian of the reduced
    /// system with respect to the highest derivatives is singular, so the
    /// structural index under-estimates the true one (E-DAE-005).
    SingularJacobian(String),
}

impl From<DaeError> for DaeStructureError {
    fn from(e: DaeError) -> Self {
        DaeStructureError::Dae(e)
    }
}

impl fmt::Display for DaeStructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DaeStructureError::Dae(DaeError::StructurallyInconsistent) => write!(
                f,
                "DAE is structurally singular: it needs one equation per unknown function \
                 (a variable listed as another's derivative is not a new unknown) and a \
                 matching of equations to unknowns"
            ),
            DaeStructureError::Dae(e) => e.fmt(f),
            DaeStructureError::MalformedDae(s) => write!(f, "malformed DAE: {s}"),
            DaeStructureError::SingularJacobian(s) => {
                write!(f, "DAE index reduction failed: {s}")
            }
        }
    }
}

impl std::error::Error for DaeStructureError {}

impl crate::errors::AlkahestError for DaeStructureError {
    fn code(&self) -> &'static str {
        match self {
            DaeStructureError::Dae(e) => e.code(),
            DaeStructureError::MalformedDae(_) => "E-DAE-004",
            DaeStructureError::SingularJacobian(_) => "E-DAE-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DaeStructureError::Dae(e) => e.remediation(),
            DaeStructureError::MalformedDae(_) => Some(
                "pass distinct symbols; derivatives[i] is the derivative of variables[i], and \
                 variables after the last derivative are algebraic",
            ),
            DaeStructureError::SingularJacobian(_) => Some(
                "the model has hidden constraints the structural analysis cannot see; \
                 reformulate it (e.g. eliminate the dependent equations) or use \
                 rosenfeld_groebner for a consistency analysis",
            ),
        }
    }
}

impl DAE {
    /// Create a new DAE.
    ///
    /// `equations` — implicit equations `g_i = 0`
    /// `variables` — all variables (algebraic + differential)
    /// `derivatives` — derivative symbols for `variables[i]`; it may be
    /// shorter than `variables`, in which case the trailing variables are
    /// algebraic.  Nothing is checked here — see [`DAE::try_new`].
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

    /// [`DAE::new`], refusing a malformed system with
    /// [`DaeStructureError::MalformedDae`] (E-DAE-004): more derivatives than
    /// variables, a variable or derivative that is not a symbol, a duplicate, or
    /// the time variable among the unknowns.
    pub fn try_new(
        equations: Vec<ExprId>,
        variables: Vec<ExprId>,
        derivatives: Vec<ExprId>,
        time_var: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, DaeStructureError> {
        let dae = DAE::new(equations, variables, derivatives, time_var);
        sigma::validate_dae_shape(&dae, pool)?;
        Ok(dae)
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
        let (_, derivatives) = normalized_jets(self, pool);
        let mut inc = vec![vec![false; n]; m];
        for (i, &eq) in self.equations.iter().enumerate() {
            for (j, &var) in self.variables.iter().enumerate() {
                let deriv = derivatives[j];
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

/// Reduce a DAE so that it determines the highest derivative of every unknown.
///
/// Equation `i` is differentiated `sigma` times (Pryce's Σ-method, equivalent
/// to Pantelides' algorithm); the result holds the original equations followed
/// by their derivatives, and `reduced_dae.index` is the differentiation index
/// (`0` for an explicit ODE, `1` for a semi-explicit index-1 system, `3` for
/// the Cartesian pendulum).
///
/// Refusals map onto the exhaustive [`DaeError`]: a malformed DAE or a singular
/// reduced Jacobian is reported as [`DaeError::StructurallyInconsistent`].  Use
/// [`pantelides_checked`] to tell them apart.
pub fn pantelides(dae: &DAE, pool: &ExprPool) -> Result<PantelidesResult, DaeError> {
    pantelides_checked(dae, pool).map_err(|e| match e {
        DaeStructureError::Dae(e) => e,
        DaeStructureError::MalformedDae(_) | DaeStructureError::SingularJacobian(_) => {
            DaeError::StructurallyInconsistent
        }
    })
}

/// [`pantelides`] with the full refusal set of [`DaeStructureError`].
///
/// * E-DAE-001 — an equation could not be differentiated.
/// * E-DAE-002 — some equation would need more than 10 differentiations.
/// * E-DAE-003 — not one equation per unknown function, or no matching of
///   equations to unknowns (structurally singular).
/// * E-DAE-004 — malformed DAE (see [`DAE::try_new`]).
/// * E-DAE-005 — the Jacobian of the reduced system with respect to the
///   highest derivatives is singular at generic points, so the structural
///   result would be wrong.
///
/// The Jacobian test evaluates exactly (rational arithmetic, `f64` for
/// transcendental entries) at deterministic sample points; it refuses only when
/// every point it could evaluate gives a zero determinant.
pub fn pantelides_checked(
    dae: &DAE,
    pool: &ExprPool,
) -> Result<PantelidesResult, DaeStructureError> {
    sigma::sigma_reduce(dae, pool)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
        // Also differentiate w.r.t. the derivative (for higher-index terms) —
        // but only when `deriv` has not itself been promoted to the state
        // vector.  Once it has, its own `(deriv, d²y_i)` pair contributes
        // exactly this term via the branch above, and emitting it here too
        // doubles it.  Prolongation promotes derivatives as it goes, so this
        // fires from the very first round on any system where one equation
        // mentions another's derivative.
        let already_a_state = variables.contains(&deriv);
        let dg_ddyi = if already_a_state {
            pool.integer(0_i32)
        } else {
            diff(equation, deriv, pool)?.value
        };
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

    // -----------------------------------------------------------------------
    // Σ-method / Pantelides: textbook indices (Hairer–Wanner, Pryce)
    // -----------------------------------------------------------------------

    fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
        pool.mul(vec![pool.integer(-1_i32), e])
    }

    fn sub(pool: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
        pool.add(vec![a, neg(pool, b)])
    }

    fn sym(pool: &ExprPool, n: &str) -> ExprId {
        pool.symbol(n, Domain::Real)
    }

    /// The Cartesian pendulum `x' = u, y' = w, u' = -λx, w' = -λy - 1,
    /// x² + y² = 1`, with λ algebraic (four derivatives for five variables).
    fn pendulum(pool: &ExprPool) -> DAE {
        let t = sym(pool, "t");
        let [x, y, u, w, lam] = ["x", "y", "u", "w", "lam"].map(|n| sym(pool, n));
        let [dx, dy, du, dw] = ["dx/dt", "dy/dt", "du/dt", "dw/dt"].map(|n| sym(pool, n));
        let two = pool.integer(2_i32);
        let one = pool.integer(1_i32);
        let eqs = vec![
            sub(pool, dx, u),
            sub(pool, dy, w),
            pool.add(vec![du, pool.mul(vec![lam, x])]),
            pool.add(vec![dw, pool.mul(vec![lam, y]), one]),
            pool.add(vec![
                pool.pow(x, two),
                pool.pow(y, two),
                pool.integer(-1_i32),
            ]),
        ];
        DAE::new(eqs, vec![x, y, u, w, lam], vec![dx, dy, du, dw], t)
    }

    #[test]
    fn pendulum_is_index_3() {
        let pool = p();
        let r = pantelides_checked(&pendulum(&pool), &pool).unwrap();
        assert_eq!(r.reduced_dae.index, Some(3));
        // The constraint is differentiated twice, the velocity equations once.
        assert_eq!(r.sigma, vec![0, 0, 0, 0, 0, 1, 1, 1, 2]);
        assert_eq!(r.differentiation_steps, 4);
        assert_eq!(r.reduced_dae.equations.len(), 9);
        assert!(r
            .reduced_dae
            .derivatives
            .iter()
            .any(|&d| pool.display(d).to_string() == "ddx/dt/dt"));
    }

    #[test]
    fn x_prime_eq_y_x_eq_sin_t_is_index_2() {
        let pool = p();
        let t = sym(&pool, "t");
        let [x, y, dx, dy] = ["x", "y", "dx/dt", "dy/dt"].map(|n| sym(&pool, n));
        let dae = DAE::new(
            vec![sub(&pool, dx, y), sub(&pool, x, pool.func("sin", vec![t]))],
            vec![x, y],
            vec![dx, dy],
            t,
        );
        let r = pantelides_checked(&dae, &pool).unwrap();
        assert_eq!(r.reduced_dae.index, Some(2));
        assert_eq!(r.sigma, vec![0, 0, 1]);
    }

    #[test]
    fn hessenberg_index_2_and_3() {
        // Index 2: x1' = x2 + y, x2' = -x1, 0 = x1 - sin t   (g_x f_y = 1).
        let pool = p();
        let t = sym(&pool, "t");
        let [x1, x2, y, dx1, dx2] = ["x1", "x2", "y", "dx1/dt", "dx2/dt"].map(|n| sym(&pool, n));
        let dae = DAE::new(
            vec![
                sub(&pool, dx1, pool.add(vec![x2, y])),
                pool.add(vec![dx2, x1]),
                sub(&pool, x1, pool.func("sin", vec![t])),
            ],
            vec![x1, x2, y],
            vec![dx1, dx2],
            t,
        );
        assert_eq!(
            pantelides_checked(&dae, &pool).unwrap().reduced_dae.index,
            Some(2)
        );

        // Index 3: x' = v, v' = y, 0 = x - t³.
        let [x, v, dx, dv] = ["x", "v", "dx/dt", "dv/dt"].map(|n| sym(&pool, n));
        let dae = DAE::new(
            vec![
                sub(&pool, dx, v),
                sub(&pool, dv, y),
                sub(&pool, x, pool.pow(t, pool.integer(3_i32))),
            ],
            vec![x, v, y],
            vec![dx, dv],
            t,
        );
        assert_eq!(
            pantelides_checked(&dae, &pool).unwrap().reduced_dae.index,
            Some(3)
        );
    }

    #[test]
    fn semi_explicit_index_1_and_ode() {
        // x' = -x + z, 0 = z - x² : index 1.
        let pool = p();
        let t = sym(&pool, "t");
        let [x, z, dx] = ["x", "z", "dx/dt"].map(|n| sym(&pool, n));
        let dae = DAE::new(
            vec![
                pool.add(vec![dx, x, neg(&pool, z)]),
                sub(&pool, z, pool.pow(x, pool.integer(2_i32))),
            ],
            vec![x, z],
            vec![dx],
            t,
        );
        let r = pantelides_checked(&dae, &pool).unwrap();
        assert_eq!(r.reduced_dae.index, Some(1));
        assert_eq!(r.differentiation_steps, 0);

        let dae = DAE::new(vec![sub(&pool, dx, x)], vec![x], vec![dx], t);
        assert_eq!(pantelides(&dae, &pool).unwrap().reduced_dae.index, Some(0));
    }

    #[test]
    fn second_order_declared_as_a_chain() {
        // variables [x, v], derivatives [v, dv]: v *is* x', so this is the
        // single unknown x with x'' + x = 0 — an ODE, index 0.
        let pool = p();
        let t = sym(&pool, "t");
        let [x, v, dv] = ["x", "v", "dv/dt"].map(|n| sym(&pool, n));
        let dae = DAE::new(vec![pool.add(vec![dv, x])], vec![x, v], vec![v, dv], t);
        assert_eq!(pantelides(&dae, &pool).unwrap().reduced_dae.index, Some(0));
    }

    #[test]
    fn singular_reduced_jacobian_is_refused() {
        // x' + y' = 1, x + y = t: structurally fine, but the equations are
        // dependent and never determine x', y' separately.
        let pool = p();
        let t = sym(&pool, "t");
        let [x, y, dx, dy] = ["x", "y", "dx/dt", "dy/dt"].map(|n| sym(&pool, n));
        let dae = DAE::new(
            vec![
                pool.add(vec![dx, dy, pool.integer(-1_i32)]),
                sub(&pool, pool.add(vec![x, y]), t),
            ],
            vec![x, y],
            vec![dx, dy],
            t,
        );
        let e = pantelides_checked(&dae, &pool).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&e), "E-DAE-005");
        assert!(matches!(
            pantelides(&dae, &pool),
            Err(DaeError::StructurallyInconsistent)
        ));
    }

    #[test]
    fn non_square_and_malformed_are_refused() {
        let pool = p();
        let t = sym(&pool, "t");
        let [x, y, dx, dy] = ["x", "y", "dx/dt", "dy/dt"].map(|n| sym(&pool, n));
        // Two equations, one unknown.
        let dae = DAE::new(vec![sub(&pool, dx, x), x], vec![x], vec![dx], t);
        let e = pantelides_checked(&dae, &pool).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&e), "E-DAE-003");
        // More derivatives than variables.
        let e = DAE::try_new(vec![sub(&pool, dx, x)], vec![x], vec![dx, dy], t, &pool).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&e), "E-DAE-004");
        let dae = DAE::new(vec![sub(&pool, dx, x)], vec![x], vec![dx, dy], t);
        assert_eq!(
            crate::errors::AlkahestError::code(&pantelides_checked(&dae, &pool).unwrap_err()),
            "E-DAE-004"
        );
        // Non-symbol derivative.
        let e = DAE::try_new(vec![dx], vec![x], vec![pool.add(vec![x, y])], t, &pool).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&e), "E-DAE-004");
        // Fewer derivatives is fine: trailing variables are algebraic.
        assert!(DAE::try_new(vec![dx, y], vec![x, y], vec![dx], t, &pool).is_ok());
    }

    #[test]
    fn incidence_matrix_with_algebraic_trailing_variable() {
        let pool = p();
        let dae = pendulum(&pool);
        let inc = dae.incidence_matrix(&pool);
        assert_eq!(inc.len(), 5);
        assert!(inc[2][4]); // u' + λx mentions λ
    }
}
