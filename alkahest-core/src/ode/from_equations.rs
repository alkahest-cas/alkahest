//! Build a first-order [`ODE`] from equations written in applied undefined
//! functions — `y''(t) + 3 y'(t) + 2 y(t) = 0`, `x'(t) = −a x(t)` — the form
//! the Euler–Lagrange pipeline and a textbook produce.
//!
//! [`ode_from_equations`] maps every `y(t)` and `Derivative(y(t), t…)` to a
//! symbol with [`functions_to_symbols`], takes the state of a function whose
//! highest derivative is of order `m` to be `[y, y', …, y^(m−1)]`, and solves
//! the equations (affine in the `n` highest derivatives, `M·h + r = 0`) for
//! those derivatives.  A leading coefficient matrix `M` that is singular makes
//! the system differential-algebraic and is refused; a symbolic `M` is
//! inverted and `det M ≠ 0` is reported as a side condition.

use crate::deriv::SideCondition;
use crate::diff::diff;
use crate::diff::undefined::{
    derivative_of, functions_to_symbols, FunctionSymbol, UndefinedFunctionError,
};
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::matrix::{take_matrix_inverse_side_conditions, Matrix};
use crate::ode::dsolve::contains;
use crate::ode::dsolve::system::DsolveSystemError;
use crate::ode::ODE;
use crate::simplify::simplify;
use std::fmt;

/// The first-order system equivalent to a set of equations in `y(t)` form.
#[derive(Clone, Debug)]
pub struct EquationSystem {
    /// `d(state)/dt = rhs`, ready for [`crate::ode::dsolve::system::dsolve_system`].
    pub ode: ODE,
    /// Every `y⁽ᵏ⁾(t)` node (orders `0 … m`) with the symbol standing for it;
    /// the states are the entries with `order < m` of their function.
    pub table: Vec<FunctionSymbol>,
    /// Hypotheses the reduction needed (`det M ≠ 0` for a symbolic leading
    /// coefficient matrix).
    pub side_conditions: Vec<SideCondition>,
}

/// Failure of [`ode_from_equations`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquationSystemError {
    /// An unknown that is not an undefined function of `t` alone.
    Function(UndefinedFunctionError),
    /// The equations do not reduce to an explicit first-order system (wrong
    /// number of equations, an unknown that never appears differentiated,
    /// nonlinear in the highest derivatives, singular leading matrix).
    System(DsolveSystemError),
}

impl fmt::Display for EquationSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquationSystemError::Function(e) => write!(f, "{e}"),
            EquationSystemError::System(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EquationSystemError {}

impl AlkahestError for EquationSystemError {
    fn code(&self) -> &'static str {
        match self {
            EquationSystemError::Function(e) => e.code(),
            EquationSystemError::System(e) => e.code(),
        }
    }
    fn remediation(&self) -> Option<&'static str> {
        match self {
            EquationSystemError::Function(e) => e.remediation(),
            EquationSystemError::System(e) => e.remediation(),
        }
    }
}

fn not_reducible(msg: String) -> EquationSystemError {
    EquationSystemError::System(DsolveSystemError::NotLinear(msg))
}

/// Reduce `equations` (each read as `= 0`) in the unknowns `funcs = [y₁(t), …]`
/// to a first-order [`ODE`].
///
/// Requires one equation per unknown, every unknown to occur differentiated,
/// and the equations to be affine in the highest derivative of each unknown
/// with an invertible coefficient matrix.  Refusals carry `E-DIFF-005/007`
/// (bad unknown) or `E-ODE-030` (does not reduce).
pub fn ode_from_equations(
    equations: &[ExprId],
    funcs: &[ExprId],
    t: ExprId,
    pool: &ExprPool,
) -> Result<EquationSystem, EquationSystemError> {
    let n = funcs.len();
    if equations.len() != n || n == 0 {
        return Err(not_reducible(format!(
            "{} equation(s) for {n} unknown function(s): a system needs one equation per unknown",
            equations.len()
        )));
    }
    // Pass 1 finds the highest order of each unknown; pass 2 adds a dummy
    // expression mentioning every order below it so the table is gap-free.
    let (_, table1) = functions_to_symbols(equations, funcs, t, Domain::Real, pool)
        .map_err(EquationSystemError::Function)?;
    let mut top = vec![0u32; n];
    for e in &table1 {
        let i = funcs.iter().position(|&f| f == e.func).unwrap();
        top[i] = top[i].max(e.order);
    }
    for (i, &m) in top.iter().enumerate() {
        if m == 0 {
            return Err(not_reducible(format!(
                "{} never appears differentiated, so the system is algebraic in it \
                 (differential-algebraic)",
                pool.display(funcs[i])
            )));
        }
    }
    let mut all = equations.to_vec();
    let mut dummy = Vec::new();
    for (i, &f) in funcs.iter().enumerate() {
        for k in 0..=top[i] {
            dummy.push(derivative_of(f, &[k], pool).map_err(EquationSystemError::Function)?);
        }
    }
    all.push(pool.add(dummy));
    let (mapped, table) = functions_to_symbols(&all, funcs, t, Domain::Real, pool)
        .map_err(EquationSystemError::Function)?;
    let sym = |i: usize, k: u32| -> ExprId {
        table
            .iter()
            .find(|e| e.func == funcs[i] && e.order == k)
            .map(|e| e.symbol)
            .expect("gap-free table")
    };

    let highest: Vec<ExprId> = (0..n).map(|i| sym(i, top[i])).collect();
    let mut m_rows = Vec::with_capacity(n);
    let mut resid = Vec::with_capacity(n);
    for &eq in mapped.iter().take(n) {
        let mut row = Vec::with_capacity(n);
        let mut rest = eq;
        for &h in &highest {
            let c = diff(eq, h, pool)
                .map_err(|e| not_reducible(e.to_string()))?
                .value;
            let c = simplify(c, pool).value;
            for &h2 in &highest {
                if contains(c, h2, pool) {
                    return Err(not_reducible(format!(
                        "the equation is not affine in the highest derivatives: the \
                         coefficient of {} still contains {}",
                        pool.display(h),
                        pool.display(h2)
                    )));
                }
            }
            rest = pool.add(vec![rest, pool.mul(vec![pool.integer(-1), c, h])]);
            row.push(c);
        }
        m_rows.push(row);
        let rest = simplify(rest, pool).value;
        for &h in &highest {
            if contains(rest, h, pool) {
                return Err(not_reducible(format!(
                    "the equation is not affine in {}",
                    pool.display(h)
                )));
            }
        }
        resid.push(rest);
    }
    let m = Matrix::new(m_rows).map_err(|e| not_reducible(e.to_string()))?;
    let inv = m.inverse(pool).map_err(|_| {
        not_reducible(
            "the matrix of highest-derivative coefficients is singular: the system is \
             differential-algebraic, not an explicit ODE system"
                .to_string(),
        )
    })?;
    let side_conditions = take_matrix_inverse_side_conditions();

    // h_i = − Σ_j inv[i][j]·resid[j]
    let mut state_vars = Vec::new();
    let mut rhs = Vec::new();
    for i in 0..n {
        for k in 0..top[i] {
            state_vars.push(sym(i, k));
            if k + 1 < top[i] {
                rhs.push(sym(i, k + 1));
            } else {
                let mut terms = Vec::with_capacity(n);
                for (j, &r) in resid.iter().enumerate() {
                    terms.push(pool.mul(vec![pool.integer(-1), inv.get(i, j), r]));
                }
                rhs.push(simplify(pool.add(terms), pool).value);
            }
        }
    }
    let ode = ODE::new(state_vars, rhs, t, pool).map_err(|e| not_reducible(e.to_string()))?;
    Ok(EquationSystem {
        ode,
        table,
        side_conditions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::dsolve::system::dsolve_system;

    fn setup() -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let y = pool.func("y", vec![t]);
        (pool, t, y)
    }

    #[test]
    fn second_order_scalar_reduces_and_solves() {
        let (pool, t, y) = setup();
        let yd = derivative_of(y, &[1], &pool).unwrap();
        let ydd = derivative_of(y, &[2], &pool).unwrap();
        // y'' + 3y' + 2y = 0
        let eq = pool.add(vec![
            ydd,
            pool.mul(vec![pool.integer(3), yd]),
            pool.mul(vec![pool.integer(2), y]),
        ]);
        let sys = ode_from_equations(&[eq], &[y], t, &pool).unwrap();
        assert_eq!(sys.ode.state_vars.len(), 2);
        let sol = dsolve_system(&sys.ode, &pool).unwrap();
        assert_eq!(sol.y_of_t.len(), 2);
    }

    #[test]
    fn singular_leading_matrix_is_refused() {
        let (pool, t, y) = setup();
        let z = pool.func("z", vec![t]);
        let yd = derivative_of(y, &[1], &pool).unwrap();
        let zd = derivative_of(z, &[1], &pool).unwrap();
        // y' + z' = 0 twice
        let e = pool.add(vec![yd, zd]);
        let r = ode_from_equations(&[e, e], &[y, z], t, &pool);
        assert!(matches!(
            r,
            Err(EquationSystemError::System(DsolveSystemError::NotLinear(_)))
        ));
    }

    #[test]
    fn unknown_without_derivative_is_refused() {
        let (pool, t, y) = setup();
        let r = ode_from_equations(&[pool.add(vec![y, pool.integer(1)])], &[y], t, &pool);
        assert!(r.is_err());
    }
}
