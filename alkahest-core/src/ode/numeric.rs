//! Built-in numeric ODE integrators (Phase 16b).
//!
//! Provides a fixed-step **RK4** integrator and an adaptive **RK45**
//! (Dormand–Prince) integrator for first-order ODE systems
//! `dy/dt = f(t, y)` expressed as Alkahest symbolic expressions.
//!
//! # Expression evaluation
//!
//! The vector field `f_i(t, y)` is evaluated by substituting concrete `f64`
//! values for the time variable and each state variable into the symbolic
//! expression tree, then walking the tree numerically.  This interpreter
//! understands the same node set as the `dsolve` verification gate
//! (`add`, `mul`, `pow`, `exp`, `sin`, `cos`, …).
//!
//! # Integrators
//!
//! | Function | Order | Step control |
//! |----------|-------|-------------|
//! | [`integrate_rk4`] | 4 | fixed step `h` |
//! | [`integrate_rk45`] | 4(5) | adaptive, Dormand–Prince |
//!
//! # Output
//!
//! Both functions return [`OdeTrajectory`]: a vector of time points and a
//! matching matrix of state values (row = time step, column = component).

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::ode::ODE;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during numeric ODE integration.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericOdeError {
    /// The ODE system has no state variables.
    EmptySystem,
    /// A step size was too small (step collapsed to `h < h_min`).
    StepSizeTooSmall,
    /// Maximum number of steps was exceeded without reaching `t_end`.
    MaxStepsExceeded,
    /// An expression in the vector field evaluated to `NaN` or `±∞`.
    NonFiniteValue(String),
    /// A symbol in the vector field could not be evaluated (unknown symbol).
    UnknownSymbol(String),
    /// The initial-condition vector length does not match `ode.order()`.
    IcLengthMismatch { got: usize, expected: usize },
    /// `t_start == t_end` or `t_end < t_start` (for positive `h`).
    InvalidTimeInterval,
}

impl fmt::Display for NumericOdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericOdeError::EmptySystem => write!(f, "ODE system has no state variables"),
            NumericOdeError::StepSizeTooSmall => write!(
                f,
                "adaptive step size collapsed below minimum — stiff problem or singularity?"
            ),
            NumericOdeError::MaxStepsExceeded => {
                write!(f, "maximum step count exceeded before reaching t_end")
            }
            NumericOdeError::NonFiniteValue(s) => {
                write!(f, "vector field produced a non-finite value: {s}")
            }
            NumericOdeError::UnknownSymbol(s) => {
                write!(
                    f,
                    "unknown symbol in vector field (not t or a state var): {s}"
                )
            }
            NumericOdeError::IcLengthMismatch { got, expected } => write!(
                f,
                "initial condition vector has length {got} but ODE has {expected} state variables"
            ),
            NumericOdeError::InvalidTimeInterval => {
                write!(f, "t_start must be strictly less than t_end")
            }
        }
    }
}

impl std::error::Error for NumericOdeError {}

impl crate::errors::AlkahestError for NumericOdeError {
    fn code(&self) -> &'static str {
        match self {
            NumericOdeError::EmptySystem => "E-ODE-020",
            NumericOdeError::StepSizeTooSmall => "E-ODE-021",
            NumericOdeError::MaxStepsExceeded => "E-ODE-022",
            NumericOdeError::NonFiniteValue(_) => "E-ODE-023",
            NumericOdeError::UnknownSymbol(_) => "E-ODE-024",
            NumericOdeError::IcLengthMismatch { .. } => "E-ODE-025",
            NumericOdeError::InvalidTimeInterval => "E-ODE-026",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            NumericOdeError::EmptySystem => None,
            NumericOdeError::StepSizeTooSmall => {
                Some("try a larger rtol/atol or check for singularities in the vector field")
            }
            NumericOdeError::MaxStepsExceeded => {
                Some("increase max_steps or use a larger step size")
            }
            NumericOdeError::NonFiniteValue(_) => {
                Some("check that the vector field is well-defined on [t_start, t_end]")
            }
            NumericOdeError::UnknownSymbol(_) => {
                Some("all symbols in the RHS must be either the time variable or a state variable")
            }
            NumericOdeError::IcLengthMismatch { .. } => {
                Some("supply one initial value per state variable")
            }
            NumericOdeError::InvalidTimeInterval => {
                Some("set t_end > t_start for forward integration")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Output: OdeTrajectory
// ---------------------------------------------------------------------------

/// The sampled trajectory returned by the numeric integrators.
///
/// `t[i]` is the time at step `i`; `y[i][j]` is the value of state variable
/// `j` at step `i`.
#[derive(Clone, Debug)]
pub struct OdeTrajectory {
    /// Time points of each recorded step (length = number of accepted steps + 1).
    pub t: Vec<f64>,
    /// State values at each time point.  Row `i` = step `i`, column `j` = component `j`.
    pub y: Vec<Vec<f64>>,
}

impl OdeTrajectory {
    /// Number of time points in the trajectory (including the initial condition).
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// `true` when the trajectory has no points (only possible if `t_start == t_end`).
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// The final time point.
    pub fn t_final(&self) -> Option<f64> {
        self.t.last().copied()
    }

    /// The final state vector.
    pub fn y_final(&self) -> Option<&[f64]> {
        self.y.last().map(|v| v.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Parameters for RK4 and RK45
// ---------------------------------------------------------------------------

/// Configuration for the fixed-step RK4 integrator.
#[derive(Clone, Debug)]
pub struct Rk4Options {
    /// Fixed step size `h`.  Must be positive.
    pub h: f64,
    /// Maximum number of steps (prevents infinite loops).
    pub max_steps: usize,
}

impl Default for Rk4Options {
    fn default() -> Self {
        Rk4Options {
            h: 0.01,
            max_steps: 1_000_000,
        }
    }
}

/// Configuration for the adaptive RK45 (Dormand–Prince) integrator.
#[derive(Clone, Debug)]
pub struct Rk45Options {
    /// Initial step size.
    pub h_init: f64,
    /// Minimum allowable step size (returns [`NumericOdeError::StepSizeTooSmall`]).
    pub h_min: f64,
    /// Maximum allowable step size.
    pub h_max: f64,
    /// Relative tolerance.
    pub rtol: f64,
    /// Absolute tolerance.
    pub atol: f64,
    /// Maximum total number of function evaluations.
    pub max_steps: usize,
}

impl Default for Rk45Options {
    fn default() -> Self {
        Rk45Options {
            h_init: 0.01,
            h_min: 1e-12,
            h_max: 1.0,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 1_000_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Expression evaluator (symbol→f64 environment)
// ---------------------------------------------------------------------------

/// Evaluate `expr` given a `{ExprId → f64}` environment.
///
/// Returns `None` for symbols not present in `env`, or for constructs that
/// the evaluator does not understand.
fn eval_expr(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (num, den) = r.0.clone().into_numer_denom();
            Some(num.to_f64() / den.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { .. } => env.get(&expr).copied(),
        ExprData::Add(args) => {
            let mut s = 0.0_f64;
            for a in args {
                s += eval_expr(a, env, pool)?;
            }
            Some(s)
        }
        ExprData::Mul(args) => {
            let mut p = 1.0_f64;
            for a in args {
                p *= eval_expr(a, env, pool)?;
            }
            Some(p)
        }
        ExprData::Pow { base, exp } => {
            let b = eval_expr(base, env, pool)?;
            let e = eval_expr(exp, env, pool)?;
            Some(b.powf(e))
        }
        ExprData::Func { name, args } => {
            let vals: Vec<f64> = args
                .iter()
                .map(|&a| eval_expr(a, env, pool))
                .collect::<Option<_>>()?;
            eval_named_func(&name, &vals)
        }
        _ => None,
    }
}

fn eval_named_func(name: &str, a: &[f64]) -> Option<f64> {
    let x = *a.first()?;
    Some(match name {
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "exp" => x.exp(),
        "log" | "ln" => x.ln(),
        "sqrt" => x.sqrt(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "abs" => x.abs(),
        "sign" | "signum" => x.signum(),
        _ if a.len() == 2 => {
            let y = a[1];
            match name {
                "atan2" => x.atan2(y),
                _ => return None,
            }
        }
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Shared: build environment and evaluate the vector field
// ---------------------------------------------------------------------------

/// Build a `{symbol → value}` map from `(t_val, y_val)` and evaluate all RHS
/// expressions.  Returns `Err` with a descriptive message on unknown symbol or
/// non-finite result.
fn eval_rhs(
    ode: &ODE,
    t_val: f64,
    y_val: &[f64],
    pool: &ExprPool,
) -> Result<Vec<f64>, NumericOdeError> {
    let mut env: HashMap<ExprId, f64> = HashMap::with_capacity(y_val.len() + 1);
    env.insert(ode.time_var, t_val);
    for (i, &sv) in ode.state_vars.iter().enumerate() {
        env.insert(sv, y_val[i]);
    }

    let mut out = Vec::with_capacity(ode.rhs.len());
    for &rhs_expr in &ode.rhs {
        let v = eval_expr(rhs_expr, &env, pool).ok_or_else(|| {
            // Try to identify which symbol was unknown.
            NumericOdeError::UnknownSymbol(pool.display(rhs_expr).to_string())
        })?;
        if !v.is_finite() {
            return Err(NumericOdeError::NonFiniteValue(format!("f({t_val}) = {v}")));
        }
        out.push(v);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Fixed-step RK4
// ---------------------------------------------------------------------------

/// Integrate `ode` from `t_start` to `t_end` using the classical 4th-order
/// Runge–Kutta method with fixed step size `opts.h`.
///
/// # Arguments
///
/// * `ode`        — First-order ODE system `dy/dt = f(t, y)`.
/// * `y0`         — Initial conditions; must have length `ode.order()`.
/// * `t_start`    — Start of the integration interval.
/// * `t_end`      — End of the integration interval; must satisfy `t_end > t_start`.
/// * `opts`       — Integrator parameters (step size, max steps).
/// * `pool`       — Expression pool that owns the expression IDs in `ode`.
///
/// # Errors
///
/// Returns [`NumericOdeError`] when the system is empty, the initial condition
/// length mismatches, the time interval is invalid, the maximum step count is
/// exceeded, or the vector field produces a non-finite value.
pub fn integrate_rk4(
    ode: &ODE,
    y0: &[f64],
    t_start: f64,
    t_end: f64,
    opts: &Rk4Options,
    pool: &ExprPool,
) -> Result<OdeTrajectory, NumericOdeError> {
    let n = ode.order();
    if n == 0 {
        return Err(NumericOdeError::EmptySystem);
    }
    if y0.len() != n {
        return Err(NumericOdeError::IcLengthMismatch {
            got: y0.len(),
            expected: n,
        });
    }
    if t_end <= t_start {
        return Err(NumericOdeError::InvalidTimeInterval);
    }

    let h = opts.h;
    let mut t = t_start;
    let mut y: Vec<f64> = y0.to_vec();

    let mut t_out = vec![t];
    let mut y_out = vec![y.clone()];

    let mut steps = 0usize;
    while t < t_end {
        if steps >= opts.max_steps {
            return Err(NumericOdeError::MaxStepsExceeded);
        }
        let h_step = h.min(t_end - t);

        // k1 = f(t, y)
        let k1 = eval_rhs(ode, t, &y, pool)?;
        // k2 = f(t + h/2, y + h/2·k1)
        let y_k2: Vec<f64> = y
            .iter()
            .zip(&k1)
            .map(|(yi, ki)| yi + 0.5 * h_step * ki)
            .collect();
        let k2 = eval_rhs(ode, t + 0.5 * h_step, &y_k2, pool)?;
        // k3 = f(t + h/2, y + h/2·k2)
        let y_k3: Vec<f64> = y
            .iter()
            .zip(&k2)
            .map(|(yi, ki)| yi + 0.5 * h_step * ki)
            .collect();
        let k3 = eval_rhs(ode, t + 0.5 * h_step, &y_k3, pool)?;
        // k4 = f(t + h, y + h·k3)
        let y_k4: Vec<f64> = y.iter().zip(&k3).map(|(yi, ki)| yi + h_step * ki).collect();
        let k4 = eval_rhs(ode, t + h_step, &y_k4, pool)?;

        // y_{n+1} = y_n + h/6·(k1 + 2k2 + 2k3 + k4)
        for i in 0..n {
            y[i] += h_step / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h_step;
        t_out.push(t);
        y_out.push(y.clone());
        steps += 1;
    }

    Ok(OdeTrajectory { t: t_out, y: y_out })
}

// ---------------------------------------------------------------------------
// Adaptive RK45 (Dormand–Prince)
// ---------------------------------------------------------------------------

// Dormand–Prince Butcher tableau constants.
// c:  nodes
// a:  stage coefficients
// b:  4th-order weights
// e:  error coefficients (b5 - b4)
const DP_A21: f64 = 1.0 / 5.0;
const DP_A31: f64 = 3.0 / 40.0;
const DP_A32: f64 = 9.0 / 40.0;
const DP_A41: f64 = 44.0 / 45.0;
const DP_A42: f64 = -56.0 / 15.0;
const DP_A43: f64 = 32.0 / 9.0;
const DP_A51: f64 = 19372.0 / 6561.0;
const DP_A52: f64 = -25360.0 / 2187.0;
const DP_A53: f64 = 64448.0 / 6561.0;
const DP_A54: f64 = -212.0 / 729.0;
const DP_A61: f64 = 9017.0 / 3168.0;
const DP_A62: f64 = -355.0 / 33.0;
const DP_A63: f64 = 46732.0 / 5247.0;
const DP_A64: f64 = 49.0 / 176.0;
const DP_A65: f64 = -5103.0 / 18656.0;

// 5th-order solution weights (b5)
const DP_B51: f64 = 35.0 / 384.0;
const DP_B53: f64 = 500.0 / 1113.0;
const DP_B54: f64 = 125.0 / 192.0;
const DP_B55: f64 = -2187.0 / 6784.0;
const DP_B56: f64 = 11.0 / 84.0;

// Error coefficients (b5 - b4) from D-P tableau
const DP_E1: f64 = 71.0 / 57600.0;
const DP_E3: f64 = -71.0 / 16695.0;
const DP_E4: f64 = 71.0 / 1920.0;
const DP_E5: f64 = -17253.0 / 339200.0;
const DP_E6: f64 = 22.0 / 525.0;
const DP_E7: f64 = -1.0 / 40.0;

/// Integrate `ode` adaptively from `t_start` to `t_end` using the
/// Dormand–Prince RK4(5) method with step-size control.
///
/// The step-size controller uses the standard PI formula:
///
/// ```text
/// err   = ‖ e_i / (atol + rtol · max(|y_n|, |y_{n+1}|)) ‖₂ / √n
/// h_new = h · min(10, max(0.1, 0.9 · err^{-1/5}))
/// ```
///
/// Steps are rejected and retried when `err > 1`.
///
/// # Arguments
///
/// * `ode`     — First-order ODE system.
/// * `y0`      — Initial conditions.
/// * `t_start` — Start of integration.
/// * `t_end`   — End of integration.
/// * `opts`    — Tolerance and step-size parameters.
/// * `pool`    — Expression pool.
///
/// # Errors
///
/// Same as [`integrate_rk4`]; additionally returns
/// [`NumericOdeError::StepSizeTooSmall`] if the adaptive controller requires
/// `h < opts.h_min`.
pub fn integrate_rk45(
    ode: &ODE,
    y0: &[f64],
    t_start: f64,
    t_end: f64,
    opts: &Rk45Options,
    pool: &ExprPool,
) -> Result<OdeTrajectory, NumericOdeError> {
    let n = ode.order();
    if n == 0 {
        return Err(NumericOdeError::EmptySystem);
    }
    if y0.len() != n {
        return Err(NumericOdeError::IcLengthMismatch {
            got: y0.len(),
            expected: n,
        });
    }
    if t_end <= t_start {
        return Err(NumericOdeError::InvalidTimeInterval);
    }

    let mut t = t_start;
    let mut y: Vec<f64> = y0.to_vec();
    let mut h = opts.h_init.min(t_end - t_start);

    let mut t_out = vec![t];
    let mut y_out = vec![y.clone()];

    let mut steps = 0usize;

    while t < t_end {
        if steps >= opts.max_steps {
            return Err(NumericOdeError::MaxStepsExceeded);
        }
        if h < opts.h_min {
            return Err(NumericOdeError::StepSizeTooSmall);
        }
        let h_step = h.min(t_end - t);

        // Stage evaluations (6 + 1 = 7, using FSAL property).
        let k1 = eval_rhs(ode, t, &y, pool)?;

        let y2: Vec<f64> = (0..n).map(|i| y[i] + h_step * DP_A21 * k1[i]).collect();
        let k2 = eval_rhs(ode, t + h_step / 5.0, &y2, pool)?;

        let y3: Vec<f64> = (0..n)
            .map(|i| y[i] + h_step * (DP_A31 * k1[i] + DP_A32 * k2[i]))
            .collect();
        let k3 = eval_rhs(ode, t + 3.0 * h_step / 10.0, &y3, pool)?;

        let y4: Vec<f64> = (0..n)
            .map(|i| y[i] + h_step * (DP_A41 * k1[i] + DP_A42 * k2[i] + DP_A43 * k3[i]))
            .collect();
        let k4 = eval_rhs(ode, t + 4.0 * h_step / 5.0, &y4, pool)?;

        let y5: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h_step * (DP_A51 * k1[i] + DP_A52 * k2[i] + DP_A53 * k3[i] + DP_A54 * k4[i])
            })
            .collect();
        let k5 = eval_rhs(ode, t + 8.0 * h_step / 9.0, &y5, pool)?;

        let y6: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h_step
                    * (DP_A61 * k1[i]
                        + DP_A62 * k2[i]
                        + DP_A63 * k3[i]
                        + DP_A64 * k4[i]
                        + DP_A65 * k5[i])
            })
            .collect();
        let k6 = eval_rhs(ode, t + h_step, &y6, pool)?;

        // 5th-order solution (accepted solution)
        let y_new: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h_step
                    * (DP_B51 * k1[i]
                        + DP_B53 * k3[i]
                        + DP_B54 * k4[i]
                        + DP_B55 * k5[i]
                        + DP_B56 * k6[i])
            })
            .collect();

        // FSAL: 7th stage at y_new (= k1 for next step)
        let k7 = eval_rhs(ode, t + h_step, &y_new, pool)?;

        // Error estimate via difference of 5th- and 4th-order solutions
        let mut err_sq_sum = 0.0_f64;
        for i in 0..n {
            let e_i = h_step
                * (DP_E1 * k1[i]
                    + DP_E3 * k3[i]
                    + DP_E4 * k4[i]
                    + DP_E5 * k5[i]
                    + DP_E6 * k6[i]
                    + DP_E7 * k7[i]);
            let sc = opts.atol + opts.rtol * y[i].abs().max(y_new[i].abs());
            err_sq_sum += (e_i / sc).powi(2);
        }
        let err = (err_sq_sum / n as f64).sqrt();

        if err <= 1.0 || h_step <= opts.h_min * 2.0 {
            // Accept the step
            t += h_step;
            y = y_new;
            t_out.push(t);
            y_out.push(y.clone());
            steps += 1;
        }

        // Update step size
        let factor = if err == 0.0 {
            10.0_f64
        } else {
            (0.9 * err.powf(-0.2)).clamp(0.1, 10.0)
        };
        h = (h_step * factor).clamp(opts.h_min, opts.h_max);
    }

    Ok(OdeTrajectory { t: t_out, y: y_out })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::ode::ODE;

    fn make_exp_ode() -> (ExprPool, ODE) {
        // y' = y  (solution: y(t) = y0 * e^t)
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let ode = ODE::new(vec![y], vec![y], t, &pool).unwrap();
        (pool, ode)
    }

    fn make_harmonic_ode() -> (ExprPool, ODE) {
        // y'' = -y  →  y0' = y1, y1' = -y0
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let y0 = pool.symbol("y0", Domain::Real);
        let y1 = pool.symbol("y1", Domain::Real);
        let neg_y0 = pool.mul(vec![pool.integer(-1_i32), y0]);
        let ode = ODE::new(vec![y0, y1], vec![y1, neg_y0], t, &pool).unwrap();
        (pool, ode)
    }

    // -----------------------------------------------------------------------
    // RK4 tests
    // -----------------------------------------------------------------------

    #[test]
    fn rk4_exp_growth_unit_interval() {
        // y' = y, y(0) = 1  →  y(1) ≈ e ≈ 2.71828...
        let (pool, ode) = make_exp_ode();
        let opts = Rk4Options {
            h: 0.001,
            max_steps: 10_000,
        };
        let traj = integrate_rk4(&ode, &[1.0], 0.0, 1.0, &opts, &pool).unwrap();
        let y1 = traj.y_final().unwrap()[0];
        let e = std::f64::consts::E;
        assert!(
            (y1 - e).abs() < 1e-9,
            "RK4 y(1) = {y1:.10}, expected e = {e:.10}, diff = {}",
            (y1 - e).abs()
        );
    }

    #[test]
    fn rk4_harmonic_oscillator() {
        // y'' + y = 0, y(0)=1, y'(0)=0  →  y(t)=cos(t)
        let (pool, ode) = make_harmonic_ode();
        let opts = Rk4Options {
            h: 0.001,
            max_steps: 100_000,
        };
        let t_end = std::f64::consts::PI; // y(π) = cos(π) = -1
        let traj = integrate_rk4(&ode, &[1.0, 0.0], 0.0, t_end, &opts, &pool).unwrap();
        let y_end = traj.y_final().unwrap()[0];
        assert!(
            (y_end - (-1.0)).abs() < 1e-7,
            "harmonic osc y(π) = {y_end}, expected -1"
        );
    }

    #[test]
    fn rk4_trajectory_length() {
        let (pool, ode) = make_exp_ode();
        let opts = Rk4Options {
            h: 0.1,
            max_steps: 100,
        };
        let traj = integrate_rk4(&ode, &[1.0], 0.0, 1.0, &opts, &pool).unwrap();
        // 10 or 11 steps + initial point, depending on floating-point rounding
        // (h=0.1 repeated 10 times may accumulate to slightly less than 1.0).
        assert!(
            traj.len() >= 11 && traj.len() <= 12,
            "expected 11 or 12 points, got {}",
            traj.len()
        );
        // Regardless, the final time should be ≈ 1.0
        let t_fin = traj.t_final().unwrap();
        assert!((t_fin - 1.0).abs() < 1e-10, "t_final = {t_fin}");
    }

    #[test]
    fn rk4_ic_mismatch_error() {
        let (pool, ode) = make_exp_ode();
        let opts = Rk4Options::default();
        let err = integrate_rk4(&ode, &[1.0, 2.0], 0.0, 1.0, &opts, &pool).unwrap_err();
        assert!(matches!(
            err,
            NumericOdeError::IcLengthMismatch {
                got: 2,
                expected: 1
            }
        ));
    }

    #[test]
    fn rk4_invalid_interval_error() {
        let (pool, ode) = make_exp_ode();
        let opts = Rk4Options::default();
        let err = integrate_rk4(&ode, &[1.0], 1.0, 0.0, &opts, &pool).unwrap_err();
        assert!(matches!(err, NumericOdeError::InvalidTimeInterval));
    }

    // -----------------------------------------------------------------------
    // RK45 tests
    // -----------------------------------------------------------------------

    #[test]
    fn rk45_exp_growth_unit_interval() {
        // y' = y, y(0) = 1  →  y(1) ≈ e
        // Use tighter tolerances for better accuracy.
        let (pool, ode) = make_exp_ode();
        let opts = Rk45Options {
            rtol: 1e-9,
            atol: 1e-12,
            ..Rk45Options::default()
        };
        let traj = integrate_rk45(&ode, &[1.0], 0.0, 1.0, &opts, &pool).unwrap();
        let y1 = traj.y_final().unwrap()[0];
        let e = std::f64::consts::E;
        assert!(
            (y1 - e).abs() < 1e-8,
            "RK45 y(1) = {y1:.10}, expected e = {e:.10}"
        );
    }

    #[test]
    fn rk45_harmonic_oscillator() {
        // y'' + y = 0, y(0)=1, y'(0)=0  →  y(π) = -1
        let (pool, ode) = make_harmonic_ode();
        let opts = Rk45Options {
            rtol: 1e-9,
            atol: 1e-12,
            ..Rk45Options::default()
        };
        let t_end = std::f64::consts::PI;
        let traj = integrate_rk45(&ode, &[1.0, 0.0], 0.0, t_end, &opts, &pool).unwrap();
        let y_end = traj.y_final().unwrap()[0];
        assert!(
            (y_end - (-1.0)).abs() < 1e-8,
            "harmonic osc y(π) = {y_end:.10}, expected -1"
        );
    }

    #[test]
    fn rk45_two_state_linear_system() {
        // dy1/dt = -2*y1 + y2,  dy2/dt = y1 - 2*y2
        // y0 = [1, 0]
        // Exact: y1(t) = (e^{-t} + e^{-3t})/2,  y2(t) = (e^{-t} - e^{-3t})/2
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let y1 = pool.symbol("y1", Domain::Real);
        let y2 = pool.symbol("y2", Domain::Real);
        let neg2 = pool.integer(-2_i32);
        let rhs1 = pool.add(vec![pool.mul(vec![neg2, y1]), y2]);
        let rhs2 = pool.add(vec![y1, pool.mul(vec![neg2, y2])]);
        let ode = ODE::new(vec![y1, y2], vec![rhs1, rhs2], t, &pool).unwrap();
        let opts = Rk45Options {
            rtol: 1e-9,
            atol: 1e-12,
            ..Rk45Options::default()
        };
        let t_end = 1.0;
        let traj = integrate_rk45(&ode, &[1.0, 0.0], 0.0, t_end, &opts, &pool).unwrap();
        let y_fin = traj.y_final().unwrap();
        let exact_y1 = ((-1.0_f64).exp() + (-3.0_f64).exp()) / 2.0;
        let exact_y2 = ((-1.0_f64).exp() - (-3.0_f64).exp()) / 2.0;
        assert!(
            (y_fin[0] - exact_y1).abs() < 1e-8,
            "y1(1) = {:.10}, exact = {:.10}",
            y_fin[0],
            exact_y1
        );
        assert!(
            (y_fin[1] - exact_y2).abs() < 1e-8,
            "y2(1) = {:.10}, exact = {:.10}",
            y_fin[1],
            exact_y2
        );
    }

    #[test]
    fn rk45_ic_mismatch_error() {
        let (pool, ode) = make_exp_ode();
        let opts = Rk45Options::default();
        let err = integrate_rk45(&ode, &[1.0, 2.0], 0.0, 1.0, &opts, &pool).unwrap_err();
        assert!(matches!(
            err,
            NumericOdeError::IcLengthMismatch {
                got: 2,
                expected: 1
            }
        ));
    }

    #[test]
    fn rk45_invalid_interval_error() {
        let (pool, ode) = make_exp_ode();
        let opts = Rk45Options::default();
        let err = integrate_rk45(&ode, &[1.0], 1.0, 0.0, &opts, &pool).unwrap_err();
        assert!(matches!(err, NumericOdeError::InvalidTimeInterval));
    }
}
