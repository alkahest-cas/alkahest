//! Numeric read-out of a symbolic solution set (`solve(..., numeric=True)`).
//!
//! `solve` works over ℂ by default, so a symbolic root such as `√(-1)` is a
//! perfectly good answer and its numeric value is the complex number `i`. The
//! read-out used to go through the *real* interpreter and report every value it
//! could not evaluate as `NaN` — which turned `±i` into `[nan, nan]` and the
//! parametric answer `x = y` into `x = nan`. A `NaN` cannot be told apart from
//! a number, so it was a wrong answer rather than a refusal.
//!
//! [`numeric_solution_values`] evaluates each value over ℂ instead, and refuses
//! with a coded error when a value is not a number at all:
//!
//! * `E-SOLVE-006` — the value depends on a free parameter (a symbol that was
//!   not solved for), so it has no numeric value;
//! * `E-SOLVE-007` — the value is parameter-free but could not be evaluated to
//!   a finite complex number, or a numeric solver could not account for every
//!   root.
//!
//! [`SolverError`](super::SolverError) is an exhaustive public enum, so these
//! refusals are a separate type rather than new variants.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use crate::errors::AlkahestError;
use crate::eval::{eval_complex_f64, ComplexF64};
use crate::kernel::{ExprId, ExprPool};
use crate::poly::collect_free_vars;

/// Why a numeric read-out of a solution set was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumericSolveRefusal {
    code: &'static str,
    detail: String,
}

impl NumericSolveRefusal {
    /// A solution value still mentions a free parameter (`E-SOLVE-006`).
    pub fn free_parameter(names: &str) -> Self {
        NumericSolveRefusal {
            code: "E-SOLVE-006",
            detail: format!(
                "numeric=True needs every solution to be a number, but a solution depends on \
                 the free parameter(s) {names}"
            ),
        }
    }

    /// A parameter-free value could not be evaluated, or roots could not be
    /// accounted for (`E-SOLVE-007`).
    pub fn not_evaluable(detail: impl Into<String>) -> Self {
        NumericSolveRefusal {
            code: "E-SOLVE-007",
            detail: detail.into(),
        }
    }

    /// What went wrong, in words.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for NumericSolveRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for NumericSolveRefusal {}

impl AlkahestError for NumericSolveRefusal {
    fn code(&self) -> &'static str {
        self.code
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self.code {
            "E-SOLVE-006" => {
                "solve with numeric=False to get the solution in terms of the parameter, or \
                 substitute a value for the parameter first"
            }
            _ => "solve with numeric=False to get the exact solution set",
        })
    }
}

/// Numeric value of one symbolic solution component, over ℂ.
///
/// The real interpreter is tried first so a real value keeps exactly the
/// number it always had; anything it cannot produce (a negative radicand, an
/// unsupported node) is evaluated in complex double precision.
pub fn numeric_value(value: ExprId, pool: &ExprPool) -> Result<ComplexF64, NumericSolveRefusal> {
    let params: BTreeSet<ExprId> = collect_free_vars(value, pool).into_iter().collect();
    if !params.is_empty() {
        let names: Vec<String> = params
            .iter()
            .map(|&p| pool.display(p).to_string())
            .collect();
        return Err(NumericSolveRefusal::free_parameter(&names.join(", ")));
    }
    let env: HashMap<ExprId, f64> = HashMap::new();
    if let Some(r) = crate::jit::eval_interp(value, &env, pool) {
        if r.is_finite() {
            return Ok(ComplexF64::new(r, 0.0));
        }
    }
    match eval_complex_f64(value, pool, &HashMap::new()) {
        Ok(c) if c.re.is_finite() && c.im.is_finite() => Ok(clean_rounding(c)),
        _ => Err(NumericSolveRefusal::not_evaluable(format!(
            "numeric=True could not evaluate the solution value {} to a finite complex number",
            pool.display(value)
        ))),
    }
}

/// Drop a component that is pure rounding noise next to the other one.
///
/// The complex evaluator reaches `√(-1)` through `exp(½·log(-1))`, which lands
/// at `6.1e-17 + 1i` rather than `i`.  A part below a few ulps of the modulus
/// carries no information, so it is reported as the zero it stands for.
fn clean_rounding(c: ComplexF64) -> ComplexF64 {
    let tiny = 8.0 * f64::EPSILON * c.re.abs().max(c.im.abs());
    ComplexF64::new(
        if c.re.abs() <= tiny { 0.0 } else { c.re },
        if c.im.abs() <= tiny { 0.0 } else { c.im },
    )
}

/// Numeric values of a whole finite solution set, over ℂ.
///
/// Every component of every tuple is evaluated with [`numeric_value`]; the
/// first one that is not a number refuses the whole call — dropping it, or
/// reporting `NaN` in its place, would present an incomplete or wrong solution
/// set as a complete one.
pub fn numeric_solution_values(
    solutions: &[Vec<ExprId>],
    pool: &ExprPool,
) -> Result<Vec<Vec<ComplexF64>>, NumericSolveRefusal> {
    solutions
        .iter()
        .map(|sol| sol.iter().map(|&v| numeric_value(v, pool)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn sqrt_of_minus_one_is_i_not_nan() {
        let pool = ExprPool::new();
        let m1 = pool.integer(-1);
        let half = pool.rational(1, 2);
        let v = pool.pow(m1, half);
        let c = numeric_value(v, &pool).unwrap();
        assert_eq!(c.re, 0.0, "{c:?}");
        assert!((c.im - 1.0).abs() < 1e-15, "{c:?}");
    }

    #[test]
    fn a_real_value_stays_real() {
        let pool = ExprPool::new();
        let two = pool.integer(2);
        let half = pool.rational(1, 2);
        let v = pool.pow(two, half);
        let c = numeric_value(v, &pool).unwrap();
        assert_eq!(c.im, 0.0);
        assert!((c.re - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn a_free_parameter_is_refused_with_e_solve_006() {
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Complex);
        let err = numeric_solution_values(&[vec![y]], &pool).unwrap_err();
        assert_eq!(err.code(), "E-SOLVE-006");
        assert!(err.detail().contains('y'));
    }
}
