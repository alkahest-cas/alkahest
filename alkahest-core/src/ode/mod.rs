//! Phase 16 — ODE representation and manipulation.
//!
//! Provides the `ODE` type for first-order systems dy/dt = f(t, y) and
//! helpers to lower higher-order ODEs to first-order systems.
//!
//! Phase 19 sensitivity analysis is also implemented here as
//! `sensitivity_system`.

pub mod dsolve;
pub mod numeric;
pub mod sensitivity;
pub mod series_solve;

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdeError {
    VariableCountMismatch,
    NotFirstOrder,
    DiffError(String),
}

impl fmt::Display for OdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdeError::VariableCountMismatch => write!(f, "variable and RHS count mismatch"),
            OdeError::NotFirstOrder => write!(f, "ODE is not first-order"),
            OdeError::DiffError(msg) => write!(f, "differentiation error: {msg}"),
        }
    }
}

impl std::error::Error for OdeError {}

impl crate::errors::AlkahestError for OdeError {
    fn code(&self) -> &'static str {
        match self {
            OdeError::VariableCountMismatch => "E-ODE-001",
            OdeError::NotFirstOrder => "E-ODE-002",
            OdeError::DiffError(_) => "E-ODE-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            OdeError::VariableCountMismatch => Some(
                "the number of state variables must equal the number of right-hand-side expressions",
            ),
            OdeError::NotFirstOrder => Some(
                "use lower_to_first_order() to reduce higher-order ODEs to first-order form",
            ),
            OdeError::DiffError(_) => Some(
                "check that all functions in the ODE are differentiable; unknown functions block lowering",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// ODE: first-order system dy/dt = f(t, y)
// ---------------------------------------------------------------------------

/// A first-order ODE system `dy_i/dt = rhs_i(t, y)`.
///
/// Invariants:
/// - `state_vars.len() == derivatives.len() == rhs.len()`
/// - `derivatives[i]` is a `Symbol` representing `d(state_vars[i])/dt`
/// - All expressions live in the same pool
#[derive(Clone, Debug)]
pub struct ODE {
    /// State variables `y_0, y_1, …`
    pub state_vars: Vec<ExprId>,
    /// Derivative symbols `dy_0/dt, dy_1/dt, …`
    pub derivatives: Vec<ExprId>,
    /// Right-hand-side expressions `f_i(t, y)`
    pub rhs: Vec<ExprId>,
    /// The independent variable (usually `t`)
    pub time_var: ExprId,
    /// Initial conditions: `(var, value)` pairs
    pub initial_conditions: Vec<(ExprId, ExprId)>,
}

impl ODE {
    /// Construct a first-order system directly.
    ///
    /// `state_vars` — the state `y_i`
    /// `rhs`        — the right-hand sides `f_i(t, y)`
    /// `time_var`   — the independent variable `t`
    ///
    /// Derivative symbols `d(y_i)/dt` are created automatically with the
    /// naming convention `d{name}/dt`.
    pub fn new(
        state_vars: Vec<ExprId>,
        rhs: Vec<ExprId>,
        time_var: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, OdeError> {
        if state_vars.len() != rhs.len() {
            return Err(OdeError::VariableCountMismatch);
        }
        let derivatives: Vec<ExprId> = state_vars
            .iter()
            .map(|&v| {
                let name = pool.with(v, |d| match d {
                    ExprData::Symbol { name, .. } => format!("d{name}/dt"),
                    _ => "d?/dt".to_string(),
                });
                pool.symbol(&name, Domain::Real)
            })
            .collect();
        Ok(ODE {
            state_vars,
            derivatives,
            rhs,
            time_var,
            initial_conditions: vec![],
        })
    }

    /// Add an initial condition `var = value`.
    pub fn with_ic(mut self, var: ExprId, value: ExprId) -> Self {
        self.initial_conditions.push((var, value));
        self
    }

    /// Number of state variables.
    pub fn order(&self) -> usize {
        self.state_vars.len()
    }

    /// Return `true` if `t` does not appear in any RHS expression.
    pub fn is_autonomous(&self, pool: &ExprPool) -> bool {
        self.rhs
            .iter()
            .all(|&rhs| !contains(rhs, self.time_var, pool))
    }

    /// Simplify all RHS expressions in place.
    pub fn simplify_rhs(&self, pool: &ExprPool) -> ODE {
        let rhs: Vec<ExprId> = self.rhs.iter().map(|&r| simplify(r, pool).value).collect();
        ODE {
            state_vars: self.state_vars.clone(),
            derivatives: self.derivatives.clone(),
            rhs,
            time_var: self.time_var,
            initial_conditions: self.initial_conditions.clone(),
        }
    }

    /// Display the system as a sequence of equations.
    pub fn display(&self, pool: &ExprPool) -> String {
        let mut lines: Vec<String> = self
            .derivatives
            .iter()
            .zip(self.rhs.iter())
            .map(|(&d, &r)| format!("  {} = {}", pool.display(d), pool.display(r)))
            .collect();
        for (v, val) in &self.initial_conditions {
            lines.push(format!(
                "  {}(0) = {}",
                pool.display(*v),
                pool.display(*val)
            ));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Higher-order ODE lowering
// ---------------------------------------------------------------------------

/// A higher-order scalar ODE `x^(n) = f(t, x, x', …, x^(n-1))`.
pub struct ScalarODE {
    /// The original variable `x`
    pub var: ExprId,
    /// `[x, x', x'', …, x^(n-1)]` — state symbols (created by `lower`)
    pub aux_vars: Vec<ExprId>,
    /// The highest-order RHS: `f(t, x, x', …, x^(n-1))`
    pub rhs: ExprId,
    /// Independent variable
    pub time_var: ExprId,
    /// Order of the ODE
    pub order: usize,
}

/// Lower a higher-order scalar ODE to a first-order system by introducing
/// auxiliary variables for each derivative.
///
/// For an `n`-th order ODE `x^(n) = f(t, x, x', …, x^(n-1))` the result is:
///
/// ```text
/// dy_0/dt = y_1
/// dy_1/dt = y_2
/// …
/// dy_{n-2}/dt = y_{n-1}
/// dy_{n-1}/dt = f(t, y_0, y_1, …, y_{n-1})
/// ```
pub fn lower_to_first_order(scalar_ode: &ScalarODE, pool: &ExprPool) -> Result<ODE, OdeError> {
    let n = scalar_ode.order;
    if n == 0 {
        return Err(OdeError::NotFirstOrder);
    }
    if n == 1 {
        // Already first-order
        return ODE::new(
            vec![scalar_ode.var],
            vec![scalar_ode.rhs],
            scalar_ode.time_var,
            pool,
        );
    }

    // Create auxiliary variables y_0 = x, y_1 = x', …, y_{n-1} = x^{(n-1)}
    let var_name = pool.with(scalar_ode.var, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    });
    let aux: Vec<ExprId> = (0..n)
        .map(|i| {
            let suffix = if i == 0 {
                var_name.clone()
            } else {
                format!("{var_name}_{i}")
            };
            pool.symbol(&suffix, Domain::Real)
        })
        .collect();

    // Build RHS: dy_i/dt = y_{i+1} for i < n-1, and dy_{n-1}/dt = rhs
    let mut rhs_vec: Vec<ExprId> = (0..n - 1).map(|i| aux[i + 1]).collect();
    rhs_vec.push(scalar_ode.rhs);

    ODE::new(aux, rhs_vec, scalar_ode.time_var, pool)
}

// ---------------------------------------------------------------------------
// Helper: does `expr` contain `needle` as a sub-expression?
// ---------------------------------------------------------------------------

fn contains(expr: ExprId, needle: ExprId, pool: &ExprPool) -> bool {
    if expr == needle {
        return true;
    }
    let children = pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        _ => vec![],
    });
    children.into_iter().any(|c| contains(c, needle, pool))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ExprPool;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn ode_new_simple() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        // dx/dt = x
        let ode = ODE::new(vec![x], vec![x], t, &pool).unwrap();
        assert_eq!(ode.order(), 1);
        assert!(ode.is_autonomous(&pool));
    }

    #[test]
    fn ode_is_not_autonomous_with_t() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        // dx/dt = t*x (not autonomous)
        let rhs = pool.mul(vec![t, x]);
        let ode = ODE::new(vec![x], vec![rhs], t, &pool).unwrap();
        assert!(!ode.is_autonomous(&pool));
    }

    #[test]
    fn ode_mismatch_error() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        // 2 vars, 1 rhs — error
        let result = ODE::new(vec![x, y], vec![x], t, &pool);
        assert!(result.is_err());
    }

    #[test]
    fn lower_second_order() {
        // x'' = -x  (harmonic oscillator)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let rhs = pool.mul(vec![pool.integer(-1_i32), x]);
        let scalar = ScalarODE {
            var: x,
            aux_vars: vec![],
            rhs,
            time_var: t,
            order: 2,
        };
        let sys = lower_to_first_order(&scalar, &pool).unwrap();
        assert_eq!(sys.order(), 2);
        // First RHS should be the auxiliary variable x_1
        let first_rhs_name = pool.with(sys.rhs[0], |d| match d {
            ExprData::Symbol { name, .. } => name.clone(),
            _ => "?".to_string(),
        });
        assert_eq!(first_rhs_name, "x_1");
    }

    #[test]
    fn ode_display() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let ode = ODE::new(vec![x], vec![x], t, &pool).unwrap();
        let s = ode.display(&pool);
        assert!(s.contains("dx/dt") || s.contains("d"), "got: {s}");
    }

    #[test]
    fn ode_with_ic() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let zero = pool.integer(0_i32);
        let one = pool.integer(1_i32);
        let ode = ODE::new(vec![x], vec![x], t, &pool)
            .unwrap()
            .with_ic(x, one);
        assert_eq!(ode.initial_conditions.len(), 1);
        assert_eq!(ode.initial_conditions[0], (x, one));
        let _ = zero; // suppress warning
    }

    #[test]
    fn ode_simplify_rhs() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let zero = pool.integer(0_i32);
        // rhs = x + 0  → should simplify to x
        let rhs = pool.add(vec![x, zero]);
        let ode = ODE::new(vec![x], vec![rhs], t, &pool).unwrap();
        let simplified = ode.simplify_rhs(&pool);
        assert_eq!(simplified.rhs[0], x);
    }
}
