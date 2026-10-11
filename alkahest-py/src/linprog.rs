//! PyO3 bindings for `alkahest_core::linprog` — exact linear and mixed-integer
//! programming with certificates.
//!
//! Python surface: `alkahest.linprog`, `alkahest.milp`, `alkahest.linprog_expr`,
//! the result classes `LPResult` / `MILPResult`, and `alkahest.LinprogError`
//! (codes `E-LP-001` … `E-LP-006`). A budget trip raises
//! `BudgetExceededError` (`E-BUDGET-*`).
//!
//! # Exactness across the boundary
//!
//! Inputs are `int`, `fractions.Fraction` (anything with integral
//! `numerator`/`denominator`), `"p/q"` strings, or rational `Expr`s. A float is
//! refused with `E-LP-002` — a certificate for the float's nearest rational is
//! a certificate for a different programme. The only floats accepted are
//! `±inf` in `bounds`, meaning "no bound". Every value returned is an `int` or a
//! `fractions.Fraction`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};
use rug::Rational;

use alkahest_core::kernel::ExprData;
use alkahest_core::linprog::{
    linear_form, linprog as core_linprog, milp as core_milp, program_from_exprs, LinearProgram,
    LinprogError, LpOutcome, MilpOptions, MilpOutcome, Sense, VarBound,
};

use crate::{PyExpr, PyExprPool};

pyo3::create_exception!(alkahest, PyLinprogError, crate::PyAlkahestError);

fn rat(py: Python<'_>, r: &Rational) -> PyResult<PyObject> {
    crate::bigint::rational_to_py(py, r)
}

fn rat_list(py: Python<'_>, v: &[Rational]) -> PyResult<PyObject> {
    let items = v.iter().map(|r| rat(py, r)).collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new_bound(py, items).into_py(py))
}

fn lp_err(e: LinprogError) -> PyErr {
    Python::with_gil(|py| {
        let (incumbent, bound) = match &e {
            LinprogError::NodeLimit {
                incumbent, bound, ..
            }
            | LinprogError::Budget {
                incumbent, bound, ..
            } => (incumbent.clone(), bound.clone()),
            _ => (None, None),
        };
        let err = if matches!(e, LinprogError::Budget { .. }) {
            let exc = py.get_type_bound::<crate::PyBudgetExceededError>();
            crate::make_structured_err(py, &exc, &e)
        } else {
            let exc = py.get_type_bound::<PyLinprogError>();
            crate::make_structured_err(py, &exc, &e)
        };
        // Attach what a stopped branch-and-bound had, so a caller can use the
        // incumbent knowing it is unproven.
        let value = err.value_bound(py);
        let set = |name: &str, v: PyObject| {
            let _ = value.setattr(name, v);
        };
        let none = || py.None();
        match &incumbent {
            Some(inc) => {
                set("incumbent", rat_list(py, &inc.x).unwrap_or_else(|_| none()));
                set(
                    "incumbent_fun",
                    rat(py, &inc.objective).unwrap_or_else(|_| none()),
                );
            }
            None => {
                set("incumbent", none());
                set("incumbent_fun", none());
            }
        }
        set(
            "bound",
            bound
                .as_ref()
                .map(|b| rat(py, b).unwrap_or_else(|_| none()))
                .unwrap_or_else(none),
        );
        if let LinprogError::NodeLimit { nodes, .. } = &e {
            set("nodes", nodes.into_py(py));
        }
        err
    })
}

fn inexact(what: &str, ob: &Bound<'_, PyAny>) -> PyErr {
    let shown = ob
        .repr()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "?".into());
    lp_err(LinprogError::Inexact {
        detail: format!("{what} = {shown}"),
    })
}

/// An exact rational from an int, Fraction, "p/q" string or rational Expr.
fn to_rational(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<Rational> {
    if let Some(r) = crate::bigint::rational_from_py(ob)? {
        return Ok(r);
    }
    if let Ok(s) = ob.downcast::<PyString>() {
        let text = s.to_string_lossy();
        return Rational::from_str_radix(text.trim(), 10).map_err(|_| inexact(what, ob));
    }
    if let Ok(e) = ob.extract::<PyRef<PyExpr>>() {
        let pool = e.pool.borrow(ob.py());
        match pool.inner.get(e.id) {
            ExprData::Integer(z) => return Ok(Rational::from(z.0)),
            ExprData::Rational(q) => return Ok(q.0),
            _ => return Err(inexact(what, ob)),
        }
    }
    Err(inexact(what, ob))
}

fn shape(detail: String) -> PyErr {
    lp_err(LinprogError::Shape { detail })
}

fn to_vector(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<Rational>> {
    let items: Vec<Bound<'_, PyAny>> = ob
        .iter()
        .map_err(|_| shape(format!("{what} must be a sequence")))?
        .collect::<PyResult<_>>()?;
    items
        .iter()
        .enumerate()
        .map(|(i, x)| to_rational(x, &format!("{what}[{i}]")))
        .collect()
}

fn to_matrix(ob: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<Vec<Vec<Rational>>> {
    let Some(ob) = ob.filter(|o| !o.is_none()) else {
        return Ok(Vec::new());
    };
    let rows: Vec<Bound<'_, PyAny>> = ob
        .iter()
        .map_err(|_| shape(format!("{what} must be a sequence of rows")))?
        .collect::<PyResult<_>>()?;
    rows.iter()
        .enumerate()
        .map(|(i, r)| to_vector(r, &format!("{what}[{i}]")))
        .collect()
}

fn opt_vector(ob: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<Vec<Rational>> {
    match ob.filter(|o| !o.is_none()) {
        None => Ok(Vec::new()),
        Some(o) => to_vector(o, what),
    }
}

/// One side of a bound: `None` / `±inf` is infinite.
fn bound_side(ob: &Bound<'_, PyAny>, what: &str, upper: bool) -> PyResult<Option<Rational>> {
    if ob.is_none() {
        return Ok(None);
    }
    if let Ok(f) = ob.downcast::<PyFloat>() {
        let v = f.value();
        if v.is_infinite() && (v > 0.0) == upper {
            return Ok(None);
        }
        return Err(inexact(what, ob));
    }
    to_rational(ob, what).map(Some)
}

fn is_pair_of_scalars(ob: &Bound<'_, PyAny>) -> bool {
    let Ok(t) = ob.extract::<Vec<Bound<'_, PyAny>>>() else {
        return false;
    };
    t.len() == 2
        && t.iter()
            .all(|x| x.is_none() || x.iter().is_err() || x.downcast::<PyString>().is_ok())
}

fn to_bounds(ob: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<Vec<VarBound>> {
    let Some(ob) = ob.filter(|o| !o.is_none()) else {
        return Ok(vec![VarBound::nonnegative(); n]);
    };
    let pair = |p: &Bound<'_, PyAny>, i: Option<usize>| -> PyResult<VarBound> {
        let what = match i {
            Some(i) => format!("bounds[{i}]"),
            None => "bounds".into(),
        };
        let t: Vec<Bound<'_, PyAny>> = p
            .extract()
            .map_err(|_| shape(format!("{what} must be a (lower, upper) pair")))?;
        if t.len() != 2 {
            return Err(shape(format!("{what} must be a (lower, upper) pair")));
        }
        Ok(VarBound::new(
            bound_side(&t[0], &format!("{what}.lower"), false)?,
            bound_side(&t[1], &format!("{what}.upper"), true)?,
        ))
    };
    if is_pair_of_scalars(ob) {
        // One (lower, upper) pair for every variable, as in SciPy.
        return Ok(vec![pair(ob, None)?; n]);
    }
    let items: Vec<Bound<'_, PyAny>> = ob
        .iter()
        .map_err(|_| shape("bounds must be a pair or a sequence of pairs".into()))?
        .collect::<PyResult<_>>()?;
    items
        .iter()
        .enumerate()
        .map(|(i, p)| pair(p, Some(i)))
        .collect()
}

fn to_sense(sense: &str) -> PyResult<Sense> {
    match sense {
        "min" | "minimize" | "minimise" => Ok(Sense::Minimize),
        "max" | "maximize" | "maximise" => Ok(Sense::Maximize),
        other => Err(shape(format!(
            "sense must be 'min' or 'max', not {other:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn build(
    c: &Bound<'_, PyAny>,
    a_ub: Option<&Bound<'_, PyAny>>,
    b_ub: Option<&Bound<'_, PyAny>>,
    a_eq: Option<&Bound<'_, PyAny>>,
    b_eq: Option<&Bound<'_, PyAny>>,
    bounds: Option<&Bound<'_, PyAny>>,
    sense: &str,
) -> PyResult<LinearProgram> {
    let c = to_vector(c, "c")?;
    let n = c.len();
    let mut lp = LinearProgram::new(c);
    lp.a_ub = to_matrix(a_ub, "A_ub")?;
    lp.b_ub = opt_vector(b_ub, "b_ub")?;
    lp.a_eq = to_matrix(a_eq, "A_eq")?;
    lp.b_eq = opt_vector(b_eq, "b_eq")?;
    lp.bounds = to_bounds(bounds, n)?;
    lp.sense = to_sense(sense)?;
    lp.validate().map_err(lp_err)?;
    Ok(lp)
}

fn status_str(o: &LpOutcome) -> &'static str {
    match o {
        LpOutcome::Optimal(_) => "optimal",
        LpOutcome::Infeasible(_) => "infeasible",
        LpOutcome::Unbounded(_) => "unbounded",
    }
}

fn solution_dict(
    py: Python<'_>,
    vars: &Option<Vec<Py<PyExpr>>>,
    x: Option<&[Rational]>,
) -> PyResult<PyObject> {
    let (Some(vars), Some(x)) = (vars, x) else {
        return Ok(py.None());
    };
    let d = PyDict::new_bound(py);
    for (v, xi) in vars.iter().zip(x) {
        d.set_item(v.clone_ref(py), rat(py, xi)?)?;
    }
    Ok(d.into_py(py))
}

// ---------------------------------------------------------------------------
// LPResult
// ---------------------------------------------------------------------------

/// Certified result of :func:`linprog` / :func:`linprog_expr`.
///
/// ``status`` is ``"optimal"``, ``"infeasible"`` or ``"unbounded"``; every
/// number is an ``int`` or ``fractions.Fraction``. The certificate for the
/// status is attached and :meth:`verify` re-checks it exactly:
///
/// * optimal — ``x``, ``fun`` and the dual multipliers ``ineqlin``,
///   ``eqlin``, ``lower``, ``upper`` (SciPy's ``marginals`` convention) with
///   ``c = A_ub.T @ ineqlin + A_eq.T @ eqlin + lower + upper``, complementary
///   slackness, and dual objective equal to ``fun``;
/// * infeasible — ``farkas``: a dict of non-negative weights (``ineqlin``,
///   ``lower``, ``upper``) and free weights (``eqlin``) combining the
///   constraints into ``0 <= negative``;
/// * unbounded — ``x`` a feasible point and ``ray`` an improving direction.
#[pyclass(name = "LPResult", module = "alkahest")]
pub struct PyLpResult {
    lp: LinearProgram,
    outcome: LpOutcome,
    vars: Option<Vec<Py<PyExpr>>>,
}

#[pymethods]
impl PyLpResult {
    /// ``"optimal"``, ``"infeasible"`` or ``"unbounded"``.
    #[getter]
    fn status(&self) -> &'static str {
        status_str(&self.outcome)
    }

    /// ``True`` iff an optimum was found.
    #[getter]
    fn success(&self) -> bool {
        matches!(self.outcome, LpOutcome::Optimal(_))
    }

    /// The optimal point, or the feasible base point of the ray when
    /// unbounded; ``None`` when infeasible.
    #[getter]
    fn x(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            LpOutcome::Optimal(c) => rat_list(py, &c.x),
            LpOutcome::Unbounded(u) => rat_list(py, &u.point),
            LpOutcome::Infeasible(_) => Ok(py.None()),
        }
    }

    /// The optimal objective value, or ``None``.
    #[getter]
    fn fun(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            LpOutcome::Optimal(c) => rat(py, &c.objective),
            _ => Ok(py.None()),
        }
    }

    /// Multipliers of the ``A_ub`` rows (optimal only).
    #[getter]
    fn ineqlin(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            LpOutcome::Optimal(c) => rat_list(py, &c.y_ub),
            _ => Ok(py.None()),
        }
    }

    /// Multipliers of the ``A_eq`` rows (optimal only).
    #[getter]
    fn eqlin(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            LpOutcome::Optimal(c) => rat_list(py, &c.y_eq),
            _ => Ok(py.None()),
        }
    }

    /// Multipliers of the lower bounds (optimal only).
    #[getter]
    fn lower(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            LpOutcome::Optimal(c) => rat_list(py, &c.z_lower),
            _ => Ok(py.None()),
        }
    }

    /// Multipliers of the upper bounds (optimal only).
    #[getter]
    fn upper(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            LpOutcome::Optimal(c) => rat_list(py, &c.z_upper),
            _ => Ok(py.None()),
        }
    }

    /// The Farkas certificate (infeasible only): a dict with keys
    /// ``ineqlin``, ``eqlin``, ``lower``, ``upper``.
    #[getter]
    fn farkas(&self, py: Python<'_>) -> PyResult<PyObject> {
        let LpOutcome::Infeasible(f) = &self.outcome else {
            return Ok(py.None());
        };
        let d = PyDict::new_bound(py);
        d.set_item("ineqlin", rat_list(py, &f.y_ub)?)?;
        d.set_item("eqlin", rat_list(py, &f.y_eq)?)?;
        d.set_item("lower", rat_list(py, &f.w_lower)?)?;
        d.set_item("upper", rat_list(py, &f.w_upper)?)?;
        Ok(d.into_py(py))
    }

    /// The improving direction (unbounded only).
    #[getter]
    fn ray(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            LpOutcome::Unbounded(u) => rat_list(py, &u.direction),
            _ => Ok(py.None()),
        }
    }

    /// ``{variable: value}`` for :func:`linprog_expr` results (``None``
    /// otherwise, or when there is no point).
    #[getter]
    fn solution(&self, py: Python<'_>) -> PyResult<PyObject> {
        let x = match &self.outcome {
            LpOutcome::Optimal(c) => Some(c.x.as_slice()),
            LpOutcome::Unbounded(u) => Some(u.point.as_slice()),
            LpOutcome::Infeasible(_) => None,
        };
        solution_dict(py, &self.vars, x)
    }

    /// The programme in matrix form, as a dict (``c``, ``A_ub``, ``b_ub``,
    /// ``A_eq``, ``b_eq``, ``bounds``, ``sense``) — for :func:`linprog_expr`
    /// this shows how the relations were laid out as rows.
    fn program(&self, py: Python<'_>) -> PyResult<PyObject> {
        program_dict(py, &self.lp)
    }

    /// Re-check the certificate exactly against the programme; ``True`` when
    /// it holds. (It was already checked before this result was returned.)
    fn verify(&self) -> bool {
        self.outcome.verify(&self.lp).is_ok()
    }

    fn __repr__(&self) -> String {
        match &self.outcome {
            LpOutcome::Optimal(c) => format!(
                "LPResult(status='optimal', fun={}, x=[{}])",
                c.objective,
                join(&c.x)
            ),
            LpOutcome::Infeasible(_) => "LPResult(status='infeasible')".into(),
            LpOutcome::Unbounded(u) => format!(
                "LPResult(status='unbounded', x=[{}], ray=[{}])",
                join(&u.point),
                join(&u.direction)
            ),
        }
    }
}

fn join(v: &[Rational]) -> String {
    v.iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn program_dict(py: Python<'_>, lp: &LinearProgram) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    let mat = |m: &[Vec<Rational>]| -> PyResult<PyObject> {
        let rows = m
            .iter()
            .map(|r| rat_list(py, r))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new_bound(py, rows).into_py(py))
    };
    d.set_item("c", rat_list(py, &lp.c)?)?;
    d.set_item("A_ub", mat(&lp.a_ub)?)?;
    d.set_item("b_ub", rat_list(py, &lp.b_ub)?)?;
    d.set_item("A_eq", mat(&lp.a_eq)?)?;
    d.set_item("b_eq", rat_list(py, &lp.b_eq)?)?;
    let side = |s: &Option<Rational>| -> PyResult<PyObject> {
        match s {
            None => Ok(py.None()),
            Some(r) => rat(py, r),
        }
    };
    let bounds = lp
        .bounds
        .iter()
        .map(|b| -> PyResult<PyObject> {
            Ok(PyTuple::new_bound(py, [side(&b.lower)?, side(&b.upper)?]).into_py(py))
        })
        .collect::<PyResult<Vec<_>>>()?;
    d.set_item("bounds", PyList::new_bound(py, bounds))?;
    d.set_item(
        "sense",
        match lp.sense {
            Sense::Minimize => "min",
            Sense::Maximize => "max",
        },
    )?;
    Ok(d.into_py(py))
}

// ---------------------------------------------------------------------------
// MILPResult
// ---------------------------------------------------------------------------

/// Result of :func:`milp`.
///
/// ``status`` is ``"optimal"`` (the branch-and-bound tree closed: ``gap`` is
/// exactly 0 and ``bound == fun``), ``"infeasible"`` (every leaf was
/// LP-infeasible with a verified Farkas vector), or ``"unbounded"`` (``x`` is
/// an integer-feasible point and ``ray`` a direction, integral on the integer
/// variables, along which the objective improves without bound). A search that
/// hit ``max_nodes`` or the active budget raises instead, carrying the
/// incumbent.
#[pyclass(name = "MILPResult", module = "alkahest")]
pub struct PyMilpResult {
    lp: LinearProgram,
    integrality: Vec<bool>,
    outcome: MilpOutcome,
    vars: Option<Vec<Py<PyExpr>>>,
}

#[pymethods]
impl PyMilpResult {
    /// ``"optimal"``, ``"infeasible"`` or ``"unbounded"``.
    #[getter]
    fn status(&self) -> &'static str {
        match self.outcome {
            MilpOutcome::Optimal(_) => "optimal",
            MilpOutcome::Infeasible { .. } => "infeasible",
            MilpOutcome::Unbounded(_) => "unbounded",
        }
    }

    /// ``True`` iff an optimum was found.
    #[getter]
    fn success(&self) -> bool {
        matches!(self.outcome, MilpOutcome::Optimal(_))
    }

    /// The optimal point (or the integer base point when unbounded).
    #[getter]
    fn x(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            MilpOutcome::Optimal(s) => rat_list(py, &s.x),
            MilpOutcome::Unbounded(u) => rat_list(py, &u.point),
            MilpOutcome::Infeasible { .. } => Ok(py.None()),
        }
    }

    /// The optimal objective value, or ``None``.
    #[getter]
    fn fun(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            MilpOutcome::Optimal(s) => rat(py, &s.objective),
            _ => Ok(py.None()),
        }
    }

    /// The proven bound; equal to ``fun`` for an optimal result.
    #[getter]
    fn bound(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.fun(py)
    }

    /// ``bound − fun``: exactly ``0`` for an optimal result, else ``None``.
    #[getter]
    fn gap(&self, py: Python<'_>) -> PyObject {
        match self.outcome {
            MilpOutcome::Optimal(_) => 0i64.into_py(py),
            _ => py.None(),
        }
    }

    /// Branch-and-bound nodes whose LP was solved.
    #[getter]
    fn nodes(&self, py: Python<'_>) -> PyObject {
        match self.outcome {
            MilpOutcome::Optimal(ref s) => s.nodes.into_py(py),
            MilpOutcome::Infeasible { nodes } => nodes.into_py(py),
            MilpOutcome::Unbounded(_) => py.None(),
        }
    }

    /// The integral improving direction (unbounded only).
    #[getter]
    fn ray(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.outcome {
            MilpOutcome::Unbounded(u) => rat_list(py, &u.direction),
            _ => Ok(py.None()),
        }
    }

    /// ``{variable: value}`` for :func:`linprog_expr` results.
    #[getter]
    fn solution(&self, py: Python<'_>) -> PyResult<PyObject> {
        let x = match &self.outcome {
            MilpOutcome::Optimal(s) => Some(s.x.as_slice()),
            MilpOutcome::Unbounded(u) => Some(u.point.as_slice()),
            MilpOutcome::Infeasible { .. } => None,
        };
        solution_dict(py, &self.vars, x)
    }

    /// The programme in matrix form, as a dict (see :meth:`LPResult.program`),
    /// plus ``integrality``.
    fn program(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = program_dict(py, &self.lp)?;
        d.bind(py)
            .set_item("integrality", self.integrality.clone())?;
        Ok(d)
    }

    /// Exact re-check of what can be checked locally: the point is feasible
    /// and integral where required (and, when unbounded, the ray is a valid
    /// integral improving direction). Optimality itself rests on the closed
    /// search tree, whose every node certificate was checked during the
    /// search.
    fn verify(&self) -> bool {
        let integral = |x: &[Rational]| {
            x.iter()
                .zip(&self.integrality)
                .all(|(v, &int)| !int || *v.denom() == 1)
        };
        match &self.outcome {
            MilpOutcome::Optimal(s) => {
                self.lp.is_feasible(&s.x)
                    && integral(&s.x)
                    && self.lp.objective_at(&s.x) == s.objective
            }
            MilpOutcome::Infeasible { .. } => true,
            MilpOutcome::Unbounded(u) => {
                u.verify(&self.lp).is_ok() && integral(&u.point) && integral(&u.direction)
            }
        }
    }

    fn __repr__(&self) -> String {
        match &self.outcome {
            MilpOutcome::Optimal(s) => format!(
                "MILPResult(status='optimal', fun={}, x=[{}], nodes={})",
                s.objective,
                join(&s.x),
                s.nodes
            ),
            MilpOutcome::Infeasible { nodes } => {
                format!("MILPResult(status='infeasible', nodes={nodes})")
            }
            MilpOutcome::Unbounded(u) => format!(
                "MILPResult(status='unbounded', x=[{}], ray=[{}])",
                join(&u.point),
                join(&u.direction)
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// Solve a linear programme exactly over the rationals, with a certificate.
///
/// ``minimise`` (or, with ``sense="max"``, maximise) ``c @ x`` subject to
/// ``A_ub @ x <= b_ub``, ``A_eq @ x == b_eq`` and ``bounds`` — the same
/// calling convention as ``scipy.optimize.linprog``, but every coefficient
/// must be exact (``int``, ``Fraction``, ``"p/q"`` or a rational ``Expr``) and
/// every result is exact. ``bounds`` is one ``(lower, upper)`` pair for all
/// variables or one per variable; ``None`` (or ``±inf``) is no bound; the
/// default is ``(0, None)``.
///
/// The solver is a two-phase simplex with Bland's rule (cannot cycle) that
/// honours the active ``alkahest.context(budget=...)`` at every pivot.
///
/// Returns:
///     :class:`LPResult` — ``status`` with its certificate (dual multipliers,
///     Farkas vector, or ray), already verified exactly.
///
/// Raises:
///     LinprogError: ``E-LP-001`` malformed shapes, ``E-LP-002`` a float or
///         other inexact coefficient, ``E-LP-006`` a certificate that failed
///         its own check (a bug; the result is withheld).
///     BudgetExceededError: the active budget tripped (``E-BUDGET-*``).
///
/// Example:
///     >>> r = alkahest.linprog([1, 1], A_ub=[[1, 2], [3, 1]], b_ub=[4, 6], sense="max")
///     >>> r.fun, r.x
///     (Fraction(14, 5), [Fraction(8, 5), Fraction(6, 5)])
#[pyfunction]
#[pyo3(name = "linprog")]
#[pyo3(signature = (c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, sense="min"))]
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn py_linprog(
    c: &Bound<'_, PyAny>,
    A_ub: Option<&Bound<'_, PyAny>>,
    b_ub: Option<&Bound<'_, PyAny>>,
    A_eq: Option<&Bound<'_, PyAny>>,
    b_eq: Option<&Bound<'_, PyAny>>,
    bounds: Option<&Bound<'_, PyAny>>,
    sense: &str,
) -> PyResult<PyLpResult> {
    let lp = build(c, A_ub, b_ub, A_eq, b_eq, bounds, sense)?;
    let outcome = core_linprog(&lp).map_err(lp_err)?;
    Ok(PyLpResult {
        lp,
        outcome,
        vars: None,
    })
}

fn to_integrality(ob: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<Vec<bool>> {
    let Some(ob) = ob.filter(|o| !o.is_none()) else {
        return Ok(vec![true; n]);
    };
    if let Ok(b) = ob.extract::<bool>() {
        return Ok(vec![b; n]);
    }
    let items: Vec<Bound<'_, PyAny>> = ob
        .iter()
        .map_err(|_| shape("integrality must be a bool or a sequence".into()))?
        .collect::<PyResult<_>>()?;
    items
        .iter()
        .map(|v| {
            v.is_truthy()
                .map_err(|_| shape("integrality entries must be 0/1 or bools".into()))
        })
        .collect()
}

/// Solve a mixed-integer linear programme exactly by branch and bound.
///
/// Arguments as for :func:`linprog`, plus ``integrality`` — one flag per
/// variable (``1``/``True`` integer, ``0``/``False`` continuous, SciPy's
/// convention), or a single bool for all; the default makes every variable an
/// integer. Binary variables are integers with ``bounds=(0, 1)``.
///
/// The search is depth-first until a first integer point is found, then
/// best-bound; every node LP is solved exactly and its certificate checked.
/// ``max_nodes`` caps the number of node LPs, and the active budget is
/// honoured at every node and pivot.
///
/// Returns:
///     :class:`MILPResult` — ``"optimal"`` with ``gap == 0``,
///     ``"infeasible"``, or ``"unbounded"`` (integer point plus integral ray).
///
/// Raises:
///     LinprogError: ``E-LP-005`` when ``max_nodes`` is reached — the error
///         carries ``incumbent`` (best integer point, unproven, or ``None``),
///         ``incumbent_fun``, ``bound`` (best open bound) and ``nodes``; and
///         the codes of :func:`linprog`.
///     BudgetExceededError: the budget tripped; carries ``incumbent``,
///         ``incumbent_fun`` and ``bound`` likewise.
///
/// Example:
///     >>> r = alkahest.milp([10, 13, 7, 8], A_ub=[[4, 6, 3, 5]], b_ub=[12],
///     ...                   bounds=(0, 1), sense="max")
///     >>> r.fun, r.x
///     (25, [1, 0, 1, 1])
#[pyfunction]
#[pyo3(name = "milp")]
#[pyo3(signature = (c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, integrality=None, sense="min", max_nodes=MilpOptions::DEFAULT_MAX_NODES))]
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn py_milp(
    c: &Bound<'_, PyAny>,
    A_ub: Option<&Bound<'_, PyAny>>,
    b_ub: Option<&Bound<'_, PyAny>>,
    A_eq: Option<&Bound<'_, PyAny>>,
    b_eq: Option<&Bound<'_, PyAny>>,
    bounds: Option<&Bound<'_, PyAny>>,
    integrality: Option<&Bound<'_, PyAny>>,
    sense: &str,
    max_nodes: usize,
) -> PyResult<PyMilpResult> {
    let lp = build(c, A_ub, b_ub, A_eq, b_eq, bounds, sense)?;
    let integrality = to_integrality(integrality, lp.nvars())?;
    let opts = MilpOptions {
        integrality: integrality.clone(),
        max_nodes,
    };
    let outcome = core_milp(&lp, &opts).map_err(lp_err)?;
    Ok(PyMilpResult {
        lp,
        integrality,
        outcome,
        vars: None,
    })
}

/// Solve a linear (or mixed-integer) programme stated with expressions.
///
/// ``objective`` is an affine ``Expr`` in ``vars``; ``constraints`` is a list
/// of relations built with ``pool.le``, ``pool.ge`` or ``pool.pred_eq`` (or an
/// ``And`` of them) between affine expressions with rational coefficients.
/// The variables are **free** unless a constraint bounds them (write
/// ``pool.ge(x, 0)`` for ``x ≥ 0``). Each ``<=``/``>=`` becomes one ``A_ub`` row
/// and each ``==`` one ``A_eq`` row, in the order given, so ``ineqlin`` and
/// ``eqlin`` line up with ``constraints`` (``result.program()`` shows the rows).
///
/// ``integer`` — ``None`` for an LP; ``True`` to make every variable an
/// integer; or a list of the variables that must be integers (then the result
/// is a :class:`MILPResult`, and ``max_nodes`` applies).
///
/// Returns:
///     :class:`LPResult` or :class:`MILPResult`; ``result.solution`` maps each
///     variable to its value.
///
/// Raises:
///     LinprogError: ``E-LP-003`` non-affine objective/constraint or a symbol
///         that is not a decision variable; ``E-LP-002`` a float or irrational
///         coefficient; ``E-LP-004`` a strict inequality or a non-relation; and
///         the codes of :func:`linprog` / :func:`milp`.
///
/// Example:
///     >>> p = alkahest.ExprPool(); x, y = p.symbol("x"), p.symbol("y")
///     >>> r = alkahest.linprog_expr(3*x + 2*y,
///     ...     [p.le(x + y, p.integer(4)), p.le(x + 3*y, p.integer(6)),
///     ...      p.ge(x, p.integer(0)), p.ge(y, p.integer(0)), p.le(x, p.integer(3))],
///     ...     [x, y], sense="max")
///     >>> r.fun
///     11
#[pyfunction]
#[pyo3(name = "linprog_expr")]
#[pyo3(signature = (objective, constraints, vars, sense="min", integer=None, max_nodes=MilpOptions::DEFAULT_MAX_NODES))]
fn py_linprog_expr(
    py: Python<'_>,
    objective: PyRef<'_, PyExpr>,
    constraints: &Bound<'_, PyAny>,
    vars: &Bound<'_, PyAny>,
    sense: &str,
    integer: Option<&Bound<'_, PyAny>>,
    max_nodes: usize,
) -> PyResult<PyObject> {
    let pool_py: Py<PyExprPool> = objective.pool.clone_ref(py);
    let var_objs: Vec<Bound<'_, PyAny>> = vars.iter()?.collect::<PyResult<_>>()?;
    let mut var_ids = Vec::new();
    let mut var_py = Vec::new();
    for v in &var_objs {
        let e = crate::extract_expr_in(v, &pool_py, "linprog_expr", "a variable")?;
        var_ids.push(e.id);
        var_py.push(Py::new(py, (*e).clone())?);
    }
    let mut con_ids = Vec::new();
    for c in constraints.iter()? {
        let c = c?;
        let e = crate::extract_expr_in(&c, &pool_py, "linprog_expr", "a constraint")?;
        con_ids.push(e.id);
    }
    let sense = to_sense(sense)?;
    let lp = {
        let pool = pool_py.borrow(py);
        program_from_exprs(&pool.inner, objective.id, &con_ids, &var_ids, sense).map_err(lp_err)?
    };
    let Some(integer) = integer.filter(|o| !o.is_none()) else {
        let outcome = core_linprog(&lp).map_err(lp_err)?;
        return Ok(PyLpResult {
            lp,
            outcome,
            vars: Some(var_py),
        }
        .into_py(py));
    };
    let integrality = if let Ok(all) = integer.extract::<bool>() {
        vec![all; var_ids.len()]
    } else {
        let mut flags = vec![false; var_ids.len()];
        for v in integer.iter()? {
            let v = v?;
            let e = crate::extract_expr_in(&v, &pool_py, "linprog_expr", "an integer variable")?;
            let Some(i) = var_ids.iter().position(|&id| id == e.id) else {
                return Err(shape(
                    "every variable in `integer` must also be in `vars`".into(),
                ));
            };
            flags[i] = true;
        }
        flags
    };
    let opts = MilpOptions {
        integrality: integrality.clone(),
        max_nodes,
    };
    let outcome = core_milp(&lp, &opts).map_err(lp_err)?;
    Ok(PyMilpResult {
        lp,
        integrality,
        outcome,
        vars: Some(var_py),
    }
    .into_py(py))
}

/// ``_linear_form(expr, vars)`` — ``(coefficients, constant)`` of an affine
/// expression; refuses with ``E-LP-003`` / ``E-LP-002``. Internal helper.
#[pyfunction]
#[pyo3(name = "_linear_form")]
fn py_linear_form(
    py: Python<'_>,
    expr: PyRef<'_, PyExpr>,
    vars: &Bound<'_, PyAny>,
) -> PyResult<(PyObject, PyObject)> {
    let pool_py: Py<PyExprPool> = expr.pool.clone_ref(py);
    let mut ids = Vec::new();
    for v in vars.iter()? {
        let v = v?;
        ids.push(crate::extract_expr_in(&v, &pool_py, "_linear_form", "a variable")?.id);
    }
    let pool = pool_py.borrow(py);
    let (coeffs, k) = linear_form(&pool.inner, expr.id, &ids).map_err(lp_err)?;
    Ok((rat_list(py, &coeffs)?, rat(py, &k)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLpResult>()?;
    m.add_class::<PyMilpResult>()?;
    m.add("LinprogError", m.py().get_type_bound::<PyLinprogError>())?;
    m.add_function(wrap_pyfunction!(py_linprog, m)?)?;
    m.add_function(wrap_pyfunction!(py_milp, m)?)?;
    m.add_function(wrap_pyfunction!(py_linprog_expr, m)?)?;
    m.add_function(wrap_pyfunction!(py_linear_form, m)?)?;
    Ok(())
}
