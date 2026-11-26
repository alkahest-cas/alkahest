//! PDE problems for FEX scoring: an operator on the jet `(u, ∇u, Δu, uₜ, x)`,
//! a right-hand side, boundary data, and the loss that turns residuals into a
//! score.
//!
//! The operator is either a **linear-plus-cubic** form with constant
//! coefficients and an optional potential,
//!
//! ```text
//! F = c_Δ·Δu + Σᵢ cᵢ·∂ᵢu + c_u·u + c₃·u³ + V(x)·u,
//! ```
//!
//! which covers the four FEX benchmark operators (Poisson `−Δu`, the
//! conservation law `c·uₜ − Σ u_{xᵢ}`, the cubic Schrödinger operator
//! `−Δu + u³ + V u`, and the Rayleigh-quotient integrands), or an arbitrary
//! **expression** in jet symbols ([`JetSymbols`]) compiled once, together with
//! its partial derivatives, by the JIT. The residual is `F − f(x)`.
//!
//! Every quantity that depends only on the points — `f(x)`, `V(x)`, the
//! boundary data `g(x)` — is evaluated once when an
//! [`super::Evaluator`] is built, never per candidate.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::diff::diff;
use crate::jit::{compile, compile_many, CompiledFn, CompiledVecFn};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::simplify;

use super::error::FexError;

/// A Rust closure of the coordinates (see [`ScalarField::Func`]).
pub type CoordFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// A scalar function of the coordinates, evaluated once per point.
#[derive(Clone, Default)]
#[non_exhaustive]
pub enum ScalarField {
    /// Identically zero.
    #[default]
    Zero,
    /// A constant.
    Const(f64),
    /// A compiled expression in the coordinates.
    Compiled(Arc<CompiledFn>),
    /// A Rust closure of the coordinates.
    Func(CoordFn),
}

impl fmt::Debug for ScalarField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarField::Zero => write!(f, "Zero"),
            ScalarField::Const(c) => write!(f, "Const({c})"),
            ScalarField::Compiled(_) => write!(f, "Compiled(..)"),
            ScalarField::Func(_) => write!(f, "Func(..)"),
        }
    }
}

impl ScalarField {
    /// Compile `expr`, an expression in `coords` only.
    pub fn from_expr(
        expr: ExprId,
        coords: &[ExprId],
        pool: &ExprPool,
    ) -> Result<ScalarField, FexError> {
        if let Some(c) = literal(expr, pool) {
            return Ok(if c == 0.0 {
                ScalarField::Zero
            } else {
                ScalarField::Const(c)
            });
        }
        let f = compile(expr, coords, pool).map_err(|e| FexError::Operator {
            detail: format!("source term {}: {e}", pool.display(expr)),
        })?;
        Ok(ScalarField::Compiled(Arc::new(f)))
    }

    /// Evaluate at one point.
    pub fn eval(&self, x: &[f64]) -> f64 {
        match self {
            ScalarField::Zero => 0.0,
            ScalarField::Const(c) => *c,
            ScalarField::Compiled(f) => f.call(x),
            ScalarField::Func(f) => f(x),
        }
    }

    pub(crate) fn is_zero(&self) -> bool {
        matches!(self, ScalarField::Zero)
    }
}

/// The symbols an expression-based operator is written in.
///
/// `ut`, when present, is shorthand for `grad[time]`.
#[derive(Clone, Debug)]
pub struct JetSymbols {
    /// The unknown `u`.
    pub u: ExprId,
    /// The Laplacian `Δu` over the spatial coordinates.
    pub lap: ExprId,
    /// The gradient components `∂u/∂xᵢ`, one per coordinate.
    pub grad: Vec<ExprId>,
    /// The coordinates `xᵢ`.
    pub coords: Vec<ExprId>,
    /// `∂u/∂t` (shorthand for `grad[time]`), if a time coordinate is used.
    pub ut: Option<ExprId>,
}

impl JetSymbols {
    /// Fresh real symbols `u`, `lap_u`, `u_x0 …`, `x0 …` (and `u_t`) in
    /// `pool`.
    pub fn new(pool: &ExprPool, dim: usize, with_time: bool) -> JetSymbols {
        use crate::kernel::Domain;
        JetSymbols {
            u: pool.symbol("u", Domain::Real),
            lap: pool.symbol("lap_u", Domain::Real),
            grad: (0..dim)
                .map(|i| pool.symbol(format!("u_x{i}"), Domain::Real))
                .collect(),
            coords: (0..dim)
                .map(|i| pool.symbol(format!("x{i}"), Domain::Real))
                .collect(),
            ut: with_time.then(|| pool.symbol("u_t", Domain::Real)),
        }
    }
}

/// A linear-plus-cubic operator with constant coefficients:
/// `c_Δ·Δu + Σᵢ cᵢ·∂ᵢu + c_u·u + c₃·u³ (+ V(x)·u when `potential`)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearOperator {
    /// Coefficient of `Δu`.
    pub lap: f64,
    /// Coefficients of `∂u/∂xᵢ` (empty means all zero).
    pub grad: Vec<f64>,
    /// Coefficient of `u`.
    pub u: f64,
    /// Coefficient of `u³`.
    pub cubic: f64,
    /// Whether the problem's potential `V(x)` multiplies `u`.
    pub potential: bool,
}

#[derive(Clone)]
pub(crate) struct CompiledOperator {
    /// Outputs: `[F, F_u, F_lap, F_g(active[0]), …]`; inputs
    /// `[u, lap, g_0 … g_{d−1}, x_0 … x_{d−1}]`.
    pub(crate) f: Arc<CompiledVecFn>,
    pub(crate) active_grad: Vec<usize>,
    pub(crate) uses_lap: bool,
    pub(crate) uses_grad: bool,
}

/// The PDE operator `F(u, ∇u, Δu, uₜ, x)`.
#[derive(Clone)]
#[non_exhaustive]
pub enum PdeOperator {
    /// Constant-coefficient linear-plus-cubic form (fast path).
    Linear(LinearOperator),
    /// A compiled expression and its jet partials.
    #[doc(hidden)]
    Compiled(CompiledOperatorHandle),
}

/// Opaque handle to a compiled expression operator.
#[derive(Clone)]
pub struct CompiledOperatorHandle(pub(crate) CompiledOperator);

impl fmt::Debug for PdeOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PdeOperator::Linear(l) => write!(f, "Linear({l:?})"),
            PdeOperator::Compiled(c) => write!(
                f,
                "Compiled(uses_lap={}, active_grad={:?})",
                c.0.uses_lap, c.0.active_grad
            ),
        }
    }
}

fn literal(e: ExprId, pool: &ExprPool) -> Option<f64> {
    match pool.get(e) {
        ExprData::Integer(z) => Some(z.0.to_f64()),
        ExprData::Rational(r) => Some(r.0.to_f64()),
        ExprData::Float(f) => Some(f.inner.to_f64()),
        _ => None,
    }
}

impl PdeOperator {
    /// `−Δu` (Poisson).
    pub fn poisson() -> PdeOperator {
        PdeOperator::Linear(LinearOperator {
            lap: -1.0,
            grad: vec![],
            u: 0.0,
            cubic: 0.0,
            potential: false,
        })
    }

    /// `Σᵢ cᵢ ∂u/∂xᵢ` (first-order transport / linear conservation law).
    pub fn transport(coeffs: Vec<f64>) -> PdeOperator {
        PdeOperator::Linear(LinearOperator {
            lap: 0.0,
            grad: coeffs,
            u: 0.0,
            cubic: 0.0,
            potential: false,
        })
    }

    /// `−Δu + u³ + V(x)·u` (cubic nonlinear Schrödinger, real stationary
    /// form); `V` is the problem's potential.
    pub fn cubic_schrodinger() -> PdeOperator {
        PdeOperator::Linear(LinearOperator {
            lap: -1.0,
            grad: vec![],
            u: 0.0,
            cubic: 1.0,
            potential: true,
        })
    }

    /// An operator written as an expression in the jet symbols and the
    /// coordinates. Its partial derivatives in `u`, `Δu` and each `∂ᵢu` are
    /// taken symbolically and compiled with it. When every partial is a
    /// numeric constant the operator is recognised as linear and the fast
    /// path is used.
    pub fn from_expr(
        expr: ExprId,
        syms: &JetSymbols,
        time: Option<usize>,
        pool: &ExprPool,
    ) -> Result<PdeOperator, FexError> {
        let dim = syms.coords.len();
        if syms.grad.len() != dim {
            return Err(FexError::Dimension {
                detail: format!("{} gradient symbols for {dim} coordinates", syms.grad.len()),
            });
        }
        // u_t is shorthand for the time component of the gradient.
        let mut expr = expr;
        if let Some(ut) = syms.ut {
            let mut map = HashMap::new();
            match time {
                Some(t) if t < dim => {
                    map.insert(ut, syms.grad[t]);
                }
                _ => {
                    let mut seen = Vec::new();
                    crate::eval::symbols::collect_free_symbols(expr, pool, &mut seen);
                    if seen.contains(&ut) {
                        return Err(FexError::Dimension {
                            detail: "the operator uses u_t but no time coordinate was given".into(),
                        });
                    }
                }
            }
            if !map.is_empty() {
                expr = crate::kernel::subs::subs(expr, &map, pool);
            }
        }
        let mut inputs = vec![syms.u, syms.lap];
        inputs.extend(&syms.grad);
        inputs.extend(&syms.coords);
        let mut free = Vec::new();
        crate::eval::symbols::collect_free_symbols(expr, pool, &mut free);
        for s in &free {
            if !inputs.contains(s) {
                return Err(FexError::Operator {
                    detail: format!(
                        "symbol {} is not a jet symbol or a coordinate",
                        pool.display(*s)
                    ),
                });
            }
        }
        let d = |v: ExprId| -> Result<ExprId, FexError> {
            let r = diff(expr, v, pool).map_err(|e| FexError::Operator {
                detail: format!("cannot differentiate the operator: {e}"),
            })?;
            Ok(simplify(r.value, pool).value)
        };
        let fu = d(syms.u)?;
        let flap = d(syms.lap)?;
        let mut fg = Vec::with_capacity(dim);
        for &g in &syms.grad {
            fg.push(d(g)?);
        }
        // Fast path: every partial is a constant and nothing depends on x
        // beyond a source term (which belongs in `rhs`, not here).
        let consts: Option<Vec<f64>> = std::iter::once(fu)
            .chain(std::iter::once(flap))
            .chain(fg.iter().copied())
            .map(|e| literal(e, pool))
            .collect();
        if let Some(c) = consts {
            // F is affine in the jet; its jet-free part must vanish (it would
            // be a source term).
            let mut zero = HashMap::new();
            let z = pool.integer(0);
            zero.insert(syms.u, z);
            zero.insert(syms.lap, z);
            for &g in &syms.grad {
                zero.insert(g, z);
            }
            let rest = simplify(crate::kernel::subs::subs(expr, &zero, pool), pool).value;
            if literal(rest, pool) == Some(0.0) {
                return Ok(PdeOperator::Linear(LinearOperator {
                    lap: c[1],
                    grad: c[2..].to_vec(),
                    u: c[0],
                    cubic: 0.0,
                    potential: false,
                }));
            }
        }
        let active_grad: Vec<usize> = (0..dim)
            .filter(|&i| literal(fg[i], pool) != Some(0.0))
            .collect();
        let uses_lap = literal(flap, pool) != Some(0.0);
        let mut outs = vec![expr, fu, flap];
        outs.extend(active_grad.iter().map(|&i| fg[i]));
        let f = compile_many(&outs, &inputs, pool).map_err(|e| FexError::Operator {
            detail: format!("cannot compile the operator: {e}"),
        })?;
        Ok(PdeOperator::Compiled(CompiledOperatorHandle(
            CompiledOperator {
                f: Arc::new(f),
                uses_grad: !active_grad.is_empty(),
                active_grad,
                uses_lap,
            },
        )))
    }

    /// Highest jet order the operator reads: 0 (`u`), 1 (`∇u`) or 2 (`Δu`).
    pub fn order(&self) -> u8 {
        match self {
            PdeOperator::Linear(l) => {
                if l.lap != 0.0 {
                    2
                } else if l.grad.iter().any(|&c| c != 0.0) {
                    1
                } else {
                    0
                }
            }
            PdeOperator::Compiled(c) => {
                if c.0.uses_lap {
                    2
                } else if c.0.uses_grad {
                    1
                } else {
                    0
                }
            }
        }
    }
}

/// How residuals become a scalar loss.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum LossKind {
    /// `mean(r_int²) + λ·mean(r_bd²)`: a least-squares problem with a
    /// residual vector and Jacobian (Levenberg–Marquardt applies).
    LeastSquares,
    /// `mean(r_int²) / mean(|u|^p) + λ·mean(r_bd²)`, the FEX reference's
    /// normalisation for homogeneous problems such as the cubic Schrödinger
    /// equation (it excludes the trivial solution `u = 0`).
    Normalized {
        /// The exponent `p` (the reference uses 3).
        power: f64,
    },
    /// The Rayleigh quotient `mean(|∇u|² + V u²) / mean(u²) + λ·mean(r_bd²)`
    /// for the eigenproblem `−Δu + V u = λ u`.
    Rayleigh,
    /// `mean((r_int − mean r_int)²) + λ·mean(r_bd²)`: least squares with the
    /// interior residual's mean removed, i.e. minimised over an unknown
    /// additive constant in the equation. This is the loss for `F(u) = c`
    /// with `c` unknown — after the substitution `u = e^v`, the eigenproblem
    /// `−Δu + V u = λ u` becomes `−Δv − |∇v|² + V = λ`, and the fitted mean
    /// of the residual is the eigenvalue. Levenberg–Marquardt applies.
    CenteredLeastSquares,
}

impl LossKind {
    /// Whether the loss is a sum of squared residuals (so a residual vector
    /// and Jacobian exist and Levenberg–Marquardt applies).
    pub fn is_least_squares(self) -> bool {
        matches!(
            self,
            LossKind::LeastSquares | LossKind::CenteredLeastSquares
        )
    }
}

/// A PDE problem: operator, source, boundary data, loss.
#[derive(Clone, Debug)]
pub struct Problem {
    pub(crate) dim: usize,
    pub(crate) time: Option<usize>,
    pub(crate) operator: PdeOperator,
    pub(crate) rhs: ScalarField,
    pub(crate) bc: ScalarField,
    pub(crate) potential: ScalarField,
    pub(crate) bc_weight: f64,
    pub(crate) loss: LossKind,
}

impl Problem {
    /// A least-squares problem `F(u) = rhs` in the interior, `u = bc` on the
    /// boundary points, with boundary weight `λ = 100` (the FEX paper's).
    pub fn new(
        dim: usize,
        operator: PdeOperator,
        rhs: ScalarField,
        bc: ScalarField,
    ) -> Result<Problem, FexError> {
        if dim == 0 {
            return Err(FexError::Dimension {
                detail: "dim must be at least 1".into(),
            });
        }
        if let PdeOperator::Linear(l) = &operator {
            if !l.grad.is_empty() && l.grad.len() != dim {
                return Err(FexError::Dimension {
                    detail: format!("{} gradient coefficients for dimension {dim}", l.grad.len()),
                });
            }
        }
        Ok(Problem {
            dim,
            time: None,
            operator,
            rhs,
            bc,
            potential: ScalarField::Zero,
            bc_weight: 100.0,
            loss: LossKind::LeastSquares,
        })
    }

    /// Designate coordinate `t` as time: it is left out of `Δu`, and `u_t`
    /// refers to `∂u/∂x_t`.
    pub fn with_time(mut self, t: usize) -> Result<Problem, FexError> {
        if t >= self.dim {
            return Err(FexError::Dimension {
                detail: format!("time coordinate {t} is not below dim {}", self.dim),
            });
        }
        self.time = Some(t);
        Ok(self)
    }

    /// Set the potential `V(x)` (used by `V·u` terms and the Rayleigh
    /// quotient).
    pub fn with_potential(mut self, v: ScalarField) -> Problem {
        self.potential = v;
        self
    }

    /// Set the boundary weight `λ`.
    pub fn with_bc_weight(mut self, w: f64) -> Problem {
        self.bc_weight = w;
        self
    }

    /// Set the loss kind.
    pub fn with_loss(mut self, loss: LossKind) -> Problem {
        self.loss = loss;
        self
    }

    /// Dimension (number of coordinates, time included).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The time coordinate, if any.
    pub fn time(&self) -> Option<usize> {
        self.time
    }

    /// The loss kind.
    pub fn loss(&self) -> LossKind {
        self.loss
    }

    /// The operator.
    pub fn operator(&self) -> &PdeOperator {
        &self.operator
    }

    /// Highest jet order needed at interior points.
    pub(crate) fn interior_order(&self) -> u8 {
        match self.loss {
            LossKind::Rayleigh => 1,
            _ => self.operator.order(),
        }
    }
}
