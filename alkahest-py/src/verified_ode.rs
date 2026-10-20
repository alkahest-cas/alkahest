//! PyO3 bindings for `alkahest_core::validated::ode` — validated (rigorous)
//! ODE integration (F28).
//!
//! Refusals arrive as `alkahest.ValidatedError` with a stable
//! `E-VALIDATED-NNN` `.code` (a budget trip as `BudgetExceededError`).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use alkahest_core::validated::ode::{
    verified_ode as core_verified_ode, VerifiedOdeError, VerifiedOdeOptions, VerifiedOdeSolution,
};

use crate::{PyExpr, PyExprPool};

fn ode_err(e: VerifiedOdeError) -> PyErr {
    Python::with_gil(|py| {
        if let VerifiedOdeError::Budget(b) = &e {
            let exc_type = py.get_type_bound::<crate::PyBudgetExceededError>();
            return crate::make_structured_err(py, &exc_type, b);
        }
        let exc_type = py.get_type_bound::<crate::PyValidatedError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// Rigorous enclosures of an ODE solution, returned by
/// :func:`alkahest.verified_ode`.
///
/// ``times[i]`` is an exact grid point and ``enclosures[i]`` a list of
/// ``(lo, hi)`` intervals, one per state component, that is **guaranteed** to
/// contain ``y(times[i])``. Bounds are rounded outward to ``float``.
/// :meth:`enclosure` encloses the solution at any ``t`` in the span.
#[pyclass(name = "VerifiedOdeSolution", module = "alkahest")]
struct PyVerifiedOdeSolution {
    inner: VerifiedOdeSolution,
}

#[pymethods]
impl PyVerifiedOdeSolution {
    /// The adaptive time grid ``t0 < t1 < … < tN`` (exact floats).
    #[getter]
    fn times(&self) -> Vec<f64> {
        self.inner.times().to_vec()
    }

    /// Grid enclosures: ``enclosures[i][c] = (lo, hi)`` contains component
    /// ``c`` of the solution at ``times[i]``.
    #[getter]
    fn enclosures(&self) -> Vec<Vec<(f64, f64)>> {
        self.inner.bounds()
    }

    /// Widest component width at each grid point.
    #[getter]
    fn widths(&self) -> Vec<f64> {
        self.inner
            .bounds()
            .iter()
            .map(|b| b.iter().map(|(lo, hi)| hi - lo).fold(0.0, f64::max))
            .collect()
    }

    /// Widest component width over the whole grid.
    #[getter]
    fn max_width(&self) -> f64 {
        self.inner.max_width()
    }

    /// Number of accepted integration steps.
    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps()
    }

    /// State dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// A rigorous enclosure ``[(lo, hi), …]`` of ``y(t)`` for any ``t`` in the
    /// integrated span (not only grid points).
    fn enclosure(&self, t: f64) -> PyResult<Vec<(f64, f64)>> {
        self.inner.bounds_at(t).map_err(ode_err)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let times = self.inner.times();
        format!(
            "VerifiedOdeSolution(dim={}, t=[{}, {}], steps={}, max_width={:e})",
            self.inner.dim(),
            times[0],
            times[times.len() - 1],
            self.inner.steps(),
            self.inner.max_width()
        )
    }
}

fn extract_expr<'py>(
    ob: &Bound<'py, PyAny>,
    pool: &Option<Py<PyExprPool>>,
    what: &str,
) -> PyResult<(alkahest_core::ExprId, Py<PyExprPool>)> {
    let e: PyRef<'py, PyExpr> = ob.extract().map_err(|_| {
        PyTypeError::new_err(format!("verified_ode: {what} must be an alkahest Expr"))
    })?;
    if let Some(p) = pool {
        if !e.pool.is(p) {
            return Err(crate::cross_pool_err(format!(
                "[E-POOL-001] verified_ode(): {what} belongs to a different ExprPool"
            )));
        }
    }
    Ok((e.id, e.pool.clone_ref(ob.py())))
}

/// `alkahest.verified_ode(rhs, vars, t, y0, t_span, *, order=20, prec=128, tol=1e-16, max_steps=20000, h_min=None, max_rel_width=1.0)`
///
/// **Validated** integration of ``y' = f(t, y)``, ``y(t0) ∈ y0``: returns
/// boxes that are *guaranteed* to contain the true solution (every solution,
/// when ``y0`` is a box), or refuses with a coded :exc:`ValidatedError`.
///
/// Method: interval Taylor series of order ``order`` (coefficients by
/// automatic differentiation of ``rhs`` in ball arithmetic), an a-priori
/// enclosure per step validated by the Picard–Lindelöf operator, a Lagrange
/// remainder bound, and Lohner's QR coordinate frame against the wrapping
/// effect.
///
/// Args:
///     rhs: list of Expr — ``f_i(t, y)``.
///     vars: list of state symbols ``y_i`` (same length as ``rhs``).
///     t: the time symbol, or ``None`` for an autonomous system.
///     y0: initial values — each a float (a point) or a ``(lo, hi)`` pair.
///     t_span: ``(t0, t1)`` with ``t0 < t1``.
///     order: Taylor order per step (1..60).
///     prec: working precision in bits.
///     tol: local error target (relative to ``max(1, |y|)``); only steers the
///         step size, never soundness.
///     max_steps: refuse with ``E-VALIDATED-008`` beyond this many steps.
///     h_min: smallest step tried (default ``1e-12 * (t1 - t0)``).
///     max_rel_width: refuse with ``E-VALIDATED-007`` when a component's width
///         exceeds this times ``max(1, |y|)``.
///
/// Returns:
///     :class:`VerifiedOdeSolution`.
///
/// Raises:
///     ValidatedError: ``E-VALIDATED-001`` unsupported primitive in ``rhs``;
///         ``-002`` unbound symbol; ``-003`` ``rhs`` leaves its domain;
///         ``-004`` overflow; ``-005`` malformed request; ``-006`` step size
///         underflow (finite-time blow-up such as ``y' = y**2`` near
///         ``t = 1``, a singularity, or a non-Lipschitz point); ``-007``
///         enclosure too wide; ``-008`` step limit (stiff problems).
///     BudgetExceededError: the active budget ran out.
///
/// Example::
///
///     y = pool.symbol("y")
///     sol = ak.verified_ode([-y], [y], None, [1.0], (0.0, 1.0))
///     lo, hi = sol.enclosure(1.0)[0]      # contains exp(-1), width ~1e-25
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "verified_ode", signature = (rhs, vars, t, y0, t_span, *, order = 20, prec = 128, tol = 1e-16, max_steps = 20000, h_min = None, max_rel_width = 1.0))]
fn py_verified_ode(
    py: Python<'_>,
    rhs: Vec<Bound<'_, PyAny>>,
    vars: Vec<Bound<'_, PyAny>>,
    t: Option<Bound<'_, PyAny>>,
    y0: Vec<Bound<'_, PyAny>>,
    t_span: (f64, f64),
    order: usize,
    prec: u32,
    tol: f64,
    max_steps: usize,
    h_min: Option<f64>,
    max_rel_width: f64,
) -> PyResult<PyVerifiedOdeSolution> {
    let prec = crate::checked_prec(prec)?;
    if vars.is_empty() {
        return Err(PyValueError::new_err(
            "verified_ode: at least one state variable is required",
        ));
    }
    let mut pool: Option<Py<PyExprPool>> = None;
    let mut var_ids = Vec::with_capacity(vars.len());
    for v in &vars {
        let (id, p) = extract_expr(v, &pool, "a state variable")?;
        pool.get_or_insert(p);
        var_ids.push(id);
    }
    let mut rhs_ids = Vec::with_capacity(rhs.len());
    for e in &rhs {
        let (id, _) = extract_expr(e, &pool, "a right-hand side")?;
        rhs_ids.push(id);
    }
    let time = match &t {
        Some(ob) if !ob.is_none() => Some(extract_expr(ob, &pool, "the time variable")?.0),
        _ => None,
    };
    let mut y0v = Vec::with_capacity(y0.len());
    for (i, ob) in y0.iter().enumerate() {
        if let Ok(v) = ob.extract::<f64>() {
            y0v.push((v, v));
        } else if let Ok((lo, hi)) = ob.extract::<(f64, f64)>() {
            y0v.push((lo, hi));
        } else {
            return Err(PyTypeError::new_err(format!(
                "verified_ode: y0[{i}] must be a float or a (lo, hi) pair"
            )));
        }
    }
    let pool_py = pool.expect("vars is non-empty");
    let pool_ref = pool_py.borrow(py);
    for &id in &rhs_ids {
        crate::guard_depth(&pool_ref.inner, id)?;
    }
    let mut opts = VerifiedOdeOptions::default();
    opts.order = order;
    opts.prec = prec;
    opts.tol = tol;
    opts.max_steps = max_steps;
    opts.h_min = h_min;
    opts.max_rel_width = max_rel_width;
    let inner = &pool_ref.inner;
    // Run on this thread so an active `alkahest.context(budget=...)` (a
    // thread-local stack) is honoured.
    let result = py
        .allow_threads(|| core_verified_ode(&rhs_ids, &var_ids, time, &y0v, t_span, inner, &opts));
    result
        .map(|inner| PyVerifiedOdeSolution { inner })
        .map_err(ode_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVerifiedOdeSolution>()?;
    m.add_function(wrap_pyfunction!(py_verified_ode, m)?)?;
    Ok(())
}
