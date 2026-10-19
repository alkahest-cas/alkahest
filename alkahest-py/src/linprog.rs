//! PyO3 bindings for `alkahest_core::linprog` — exact linear and mixed-integer
//! programming with certificates.
//!
//! Python surface: `alkahest.linprog`, `alkahest.milp`, `alkahest.linprog_expr`,
//! `alkahest.linprog_parametric`, the result classes `LPResult` /
//! `MILPResult` / `ParametricLPResult`, and `alkahest.LinprogError` (codes
//! `E-LP-001` … `E-LP-007`). A budget trip raises
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
    linear_form, linprog_parametric as core_parametric, linprog_with, milp_with,
    parametric_from_exprs, program_from_exprs, BasisEntry, CutOptions, LinearProgram, LinprogError,
    LpBasis, LpMethod, LpOptions, LpOutcome, LpSolution, MilpOptions, MilpOutcome, ParametricDelta,
    ParametricRegion, ParametricSolution, RegionStatus, Sense, VarBound,
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
    basis: LpBasis,
    iterations: usize,
    method: LpMethod,
    warm_started: bool,
}

impl PyLpResult {
    fn new(lp: LinearProgram, sol: LpSolution, vars: Option<Vec<Py<PyExpr>>>) -> Self {
        PyLpResult {
            lp,
            outcome: sol.outcome,
            vars,
            basis: sol.basis,
            iterations: sol.iterations,
            method: sol.method,
            warm_started: sol.warm_started,
        }
    }
}

fn basis_to_py(py: Python<'_>, b: &LpBasis) -> PyObject {
    let items: Vec<PyObject> = b
        .entries
        .iter()
        .filter_map(|e| {
            let (tag, k) = match *e {
                BasisEntry::Var(j) => ("x", j),
                BasisEntry::VarNeg(j) => ("x-", j),
                BasisEntry::Slack(i) => ("slack", i),
                BasisEntry::UpperSlack(j) => ("upper", j),
                _ => return None,
            };
            Some((tag, k).into_py(py))
        })
        .collect();
    PyList::new_bound(py, items).into_py(py)
}

/// A warm-start basis: a list of ``(kind, index)`` pairs as returned in
/// ``LPResult.basis``, or an ``LPResult`` itself.
fn basis_from_py(ob: &Bound<'_, PyAny>) -> PyResult<LpBasis> {
    if let Ok(r) = ob.extract::<PyRef<PyLpResult>>() {
        return Ok(r.basis.clone());
    }
    let bad = || {
        shape(
            "basis must be a list of (kind, index) pairs with kind 'x', 'x-', 'slack' or \
             'upper', or an LPResult"
                .into(),
        )
    };
    let items: Vec<Bound<'_, PyAny>> = ob.iter().map_err(|_| bad())?.collect::<PyResult<_>>()?;
    let mut entries = Vec::with_capacity(items.len());
    for it in &items {
        let (tag, k): (String, usize) = it.extract().map_err(|_| bad())?;
        entries.push(match tag.as_str() {
            "x" => BasisEntry::Var(k),
            "x-" => BasisEntry::VarNeg(k),
            "slack" => BasisEntry::Slack(k),
            "upper" => BasisEntry::UpperSlack(k),
            _ => return Err(bad()),
        });
    }
    Ok(LpBasis::new(entries))
}

fn to_method(method: &str) -> PyResult<LpMethod> {
    match method {
        "auto" => Ok(LpMethod::Auto),
        "dense" => Ok(LpMethod::Dense),
        "revised" => Ok(LpMethod::Revised),
        other => Err(shape(format!(
            "method must be 'auto', 'dense' or 'revised', not {other:?}"
        ))),
    }
}

fn method_str(m: LpMethod) -> &'static str {
    match m {
        LpMethod::Dense => "dense",
        LpMethod::Revised => "revised",
        _ => "auto",
    }
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

    /// The final simplex basis, a list of ``(kind, index)`` pairs — ``("x", j)``
    /// variable ``j`` (measured from its finite bound, or the positive part of
    /// a free variable), ``("x-", j)`` the negative part of free variable
    /// ``j``, ``("slack", i)`` the slack of ``A_ub`` row ``i``, ``("upper", j)``
    /// the slack of variable ``j``'s upper bound. Pass it (or this result) as
    /// ``linprog(..., basis=...)`` to warm-start a related programme.
    #[getter]
    fn basis(&self, py: Python<'_>) -> PyObject {
        basis_to_py(py, &self.basis)
    }

    /// Simplex pivots performed.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// The method that ran: ``"dense"`` or ``"revised"``.
    #[getter]
    fn method(&self) -> &'static str {
        method_str(self.method)
    }

    /// ``True`` when the supplied warm-start basis was used.
    #[getter]
    fn warm_started(&self) -> bool {
        self.warm_started
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
    cuts: Vec<(Vec<Rational>, Rational)>,
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

    /// The Gomory mixed-integer cuts added at the root, as ``(row, rhs)``
    /// pairs meaning ``row @ x <= rhs``; each holds at every integer-feasible
    /// point (optimal results only; ``[]`` when none were added).
    #[getter]
    fn cuts(&self, py: Python<'_>) -> PyResult<PyObject> {
        if !matches!(self.outcome, MilpOutcome::Optimal(_)) {
            return Ok(PyList::empty_bound(py).into_py(py));
        }
        let items = self
            .cuts
            .iter()
            .map(|(row, rhs)| -> PyResult<PyObject> {
                Ok(PyTuple::new_bound(py, [rat_list(py, row)?, rat(py, rhs)?]).into_py(py))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new_bound(py, items).into_py(py))
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
/// ``method`` picks the simplex: ``"dense"`` — a two-phase tableau with
/// Bland's rule; ``"revised"`` — a revised simplex keeping ``A`` sparse with an
/// exact product-form factorisation of the basis (Dantzig pricing with a
/// Bland fallback, so it cannot cycle either), much faster on medium sparse
/// programmes; ``"auto"`` (default) — dense for at most 12 rows, revised
/// otherwise. ``basis`` warm-starts the revised method from a previous
/// result's :attr:`LPResult.basis` (or the result itself): a basis still
/// primal feasible goes straight to phase 2, one still dual feasible (after
/// a right-hand-side change or added rows) runs the dual simplex, anything
/// else falls back to a cold start (``warm_started`` says which). Both honour
/// the active ``alkahest.context(budget=...)`` at every pivot.
///
/// Returns:
///     :class:`LPResult` — ``status`` with its certificate (dual multipliers,
///     Farkas vector, or ray), already verified exactly, plus ``basis``,
///     ``iterations``, ``method`` and ``warm_started``.
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
#[pyo3(signature = (c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, sense="min", method="auto", basis=None))]
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
    method: &str,
    basis: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyLpResult> {
    let lp = build(c, A_ub, b_ub, A_eq, b_eq, bounds, sense)?;
    let mut opts = LpOptions::default().with_method(to_method(method)?);
    if let Some(b) = basis.filter(|o| !o.is_none()) {
        opts = opts.with_basis(basis_from_py(b)?);
    }
    let sol = linprog_with(&lp, &opts).map_err(lp_err)?;
    Ok(PyLpResult::new(lp, sol, None))
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
/// Before branching, up to ``cut_rounds`` rounds of Gomory mixed-integer
/// cuts (at most ``max_cuts`` in all) strengthen the root relaxation
/// (cut-and-branch). The cuts are read off the exact optimal tableau, so each
/// one is valid: it never removes an integer-feasible point (they are listed
/// in :attr:`MILPResult.cuts`). Pass ``cut_rounds=0`` to branch on the plain
/// relaxation. The search is then depth-first until a first integer point is
/// found, then best-bound; every node LP is solved exactly, warm-started from
/// its parent's basis (``warm_start``), and its certificate checked.
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
#[pyo3(signature = (c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, integrality=None, sense="min", max_nodes=MilpOptions::DEFAULT_MAX_NODES, cut_rounds=CutOptions::DEFAULT_CUT_ROUNDS, max_cuts=CutOptions::DEFAULT_MAX_CUTS, warm_start=true))]
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
    cut_rounds: usize,
    max_cuts: usize,
    warm_start: bool,
) -> PyResult<PyMilpResult> {
    let lp = build(c, A_ub, b_ub, A_eq, b_eq, bounds, sense)?;
    let integrality = to_integrality(integrality, lp.nvars())?;
    let mut opts = MilpOptions::new(integrality.clone());
    opts.max_nodes = max_nodes;
    let mut cut_opts = CutOptions::default();
    cut_opts.cut_rounds = cut_rounds;
    cut_opts.max_cuts = max_cuts;
    cut_opts.warm_start = warm_start;
    let report = milp_with(&lp, &opts, &cut_opts).map_err(lp_err)?;
    Ok(PyMilpResult {
        lp,
        integrality,
        outcome: report.outcome,
        vars: None,
        cuts: report.cuts,
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
/// ``parameter`` — a symbol ``θ`` that may appear affinely **either** in the
/// objective's coefficients (``(2 + θ)*x``) **or** in the constraints'
/// constant terms (``x + y <= 4 + θ``); the result is then a
/// :class:`ParametricLPResult` over ``parameter_range`` (a ``(lo, hi)`` pair,
/// ``None`` for an infinite end), as from :func:`linprog_parametric`.
///
/// Returns:
///     :class:`LPResult`, :class:`MILPResult` or :class:`ParametricLPResult`;
///     ``result.solution`` (``solution_at(θ)`` for a parametric result) maps
///     each variable to its value.
///
/// Raises:
///     LinprogError: ``E-LP-003`` non-affine objective/constraint or a symbol
///         that is not a decision variable; ``E-LP-002`` a float or irrational
///         coefficient; ``E-LP-004`` a strict inequality or a non-relation;
///         ``E-LP-007`` an unsupported parametric form (the parameter
///         multiplying a variable inside a constraint, entering non-linearly,
///         moving the objective and the right-hand side at once, or combined
///         with ``integer``); and the codes of :func:`linprog` / :func:`milp`.
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
#[pyo3(signature = (objective, constraints, vars, sense="min", integer=None, max_nodes=MilpOptions::DEFAULT_MAX_NODES, parameter=None, parameter_range=None))]
#[allow(clippy::too_many_arguments)]
fn py_linprog_expr(
    py: Python<'_>,
    objective: PyRef<'_, PyExpr>,
    constraints: &Bound<'_, PyAny>,
    vars: &Bound<'_, PyAny>,
    sense: &str,
    integer: Option<&Bound<'_, PyAny>>,
    max_nodes: usize,
    parameter: Option<&Bound<'_, PyAny>>,
    parameter_range: Option<&Bound<'_, PyAny>>,
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
    if let Some(param) = parameter.filter(|o| !o.is_none()) {
        if integer.is_some_and(|o| !o.is_none()) {
            return Err(lp_err(LinprogError::UnsupportedParametric {
                detail: "a parameter cannot be combined with integer variables".into(),
            }));
        }
        let pe = crate::extract_expr_in(param, &pool_py, "linprog_expr", "the parameter")?;
        let (lp, delta) = {
            let pool = pool_py.borrow(py);
            parametric_from_exprs(&pool.inner, objective.id, &con_ids, &var_ids, pe.id, sense)
                .map_err(lp_err)?
        };
        let (lo, hi) = parse_theta_range(parameter_range)?;
        let sol = core_parametric(&lp, &delta, lo, hi).map_err(lp_err)?;
        return Ok(PyParametricResult {
            lp,
            delta,
            sol,
            vars: Some(var_py),
        }
        .into_py(py));
    }
    let lp = {
        let pool = pool_py.borrow(py);
        program_from_exprs(&pool.inner, objective.id, &con_ids, &var_ids, sense).map_err(lp_err)?
    };
    let Some(integer) = integer.filter(|o| !o.is_none()) else {
        let sol = linprog_with(&lp, &LpOptions::default()).map_err(lp_err)?;
        return Ok(PyLpResult::new(lp, sol, Some(var_py)).into_py(py));
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
    let mut opts = MilpOptions::new(integrality.clone());
    opts.max_nodes = max_nodes;
    let report = milp_with(&lp, &opts, &CutOptions::default()).map_err(lp_err)?;
    Ok(PyMilpResult {
        lp,
        integrality,
        outcome: report.outcome,
        vars: Some(var_py),
        cuts: report.cuts,
    }
    .into_py(py))
}

// ---------------------------------------------------------------------------
// Parametric LP
// ---------------------------------------------------------------------------

fn parse_theta_range(
    ob: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<Rational>, Option<Rational>)> {
    let Some(ob) = ob.filter(|o| !o.is_none()) else {
        return Ok((None, None));
    };
    let t: Vec<Bound<'_, PyAny>> = ob
        .extract()
        .map_err(|_| shape("theta_range must be a (lower, upper) pair".into()))?;
    if t.len() != 2 {
        return Err(shape("theta_range must be a (lower, upper) pair".into()));
    }
    Ok((
        bound_side(&t[0], "theta_range.lower", false)?,
        bound_side(&t[1], "theta_range.upper", true)?,
    ))
}

fn status_name(s: RegionStatus) -> &'static str {
    match s {
        RegionStatus::Optimal => "optimal",
        RegionStatus::Infeasible => "infeasible",
        RegionStatus::Unbounded => "unbounded",
        _ => "unknown",
    }
}

fn opt_rat(py: Python<'_>, v: &Option<Rational>) -> PyResult<PyObject> {
    match v {
        None => Ok(py.None()),
        Some(r) => rat(py, r),
    }
}

fn affine_pair(py: Python<'_>, p: &Option<(Vec<Rational>, Vec<Rational>)>) -> PyResult<PyObject> {
    match p {
        None => Ok(py.None()),
        Some((a, b)) => {
            Ok(PyTuple::new_bound(py, [rat_list(py, a)?, rat_list(py, b)?]).into_py(py))
        }
    }
}

fn region_dict(py: Python<'_>, r: &ParametricRegion) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item(
        "theta",
        PyTuple::new_bound(py, [opt_rat(py, &r.lower)?, opt_rat(py, &r.upper)?]),
    )?;
    d.set_item("status", status_name(r.status))?;
    match &r.value {
        Some((v0, v1)) => d.set_item(
            "value",
            PyTuple::new_bound(py, [rat(py, v0)?, rat(py, v1)?]),
        )?,
        None => d.set_item("value", py.None())?,
    }
    d.set_item("x", affine_pair(py, &r.x)?)?;
    d.set_item("ineqlin", affine_pair(py, &r.y_ub)?)?;
    d.set_item("eqlin", affine_pair(py, &r.y_eq)?)?;
    d.set_item("lower", affine_pair(py, &r.z_lower)?)?;
    d.set_item("upper", affine_pair(py, &r.z_upper)?)?;
    match &r.basis {
        Some(b) => d.set_item("basis", basis_to_py(py, b))?,
        None => d.set_item("basis", py.None())?,
    }
    Ok(d.into_py(py))
}

/// Result of :func:`linprog_parametric` (or :func:`linprog_expr` with a
/// ``parameter``): the critical regions of ``θ``, left to right.
///
/// Each region is a dict: ``theta`` — ``(lo, hi)`` (``None`` for an infinite
/// end); ``status`` — ``"optimal"`` (closed region, one basis optimal
/// throughout), ``"infeasible"`` or ``"unbounded"`` (open at the end shared
/// with an optimal region); and, when optimal, ``value``, ``x``,
/// ``ineqlin``, ``eqlin``, ``lower``, ``upper`` as affine pairs ``(a, b)``
/// meaning ``a + θ·b`` (``value`` a pair of numbers, the rest pairs of
/// lists), and the region's ``basis``. Every optimal region's certificate was
/// verified exactly for **all** ``θ`` in it (at its ends, and through the
/// certificate's slope for an infinite end).
#[pyclass(name = "ParametricLPResult", module = "alkahest")]
pub struct PyParametricResult {
    lp: LinearProgram,
    delta: ParametricDelta,
    sol: ParametricSolution,
    vars: Option<Vec<Py<PyExpr>>>,
}

#[pymethods]
impl PyParametricResult {
    /// ``"rhs"`` or ``"objective"`` — what moves with ``θ``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.delta {
            ParametricDelta::Rhs { .. } => "rhs",
            _ => "objective",
        }
    }

    /// The critical regions (see the class docs).
    #[getter]
    fn regions(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items = self
            .sol
            .regions
            .iter()
            .map(|r| region_dict(py, r))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new_bound(py, items).into_py(py))
    }

    /// The exact ``θ`` where the optimal basis or the status changes.
    #[getter]
    fn breakpoints(&self, py: Python<'_>) -> PyResult<PyObject> {
        rat_list(py, &self.sol.breakpoints)
    }

    /// The optimal value at ``theta``; ``None`` where the programme is
    /// infeasible or unbounded (see :meth:`status_at`) or outside the range.
    fn value_at(&self, py: Python<'_>, theta: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let t = to_rational(theta, "theta")?;
        opt_rat(py, &self.sol.value_at(&t))
    }

    /// ``"optimal"``, ``"infeasible"`` or ``"unbounded"`` at ``theta``
    /// (``None`` outside the range).
    fn status_at(&self, py: Python<'_>, theta: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let t = to_rational(theta, "theta")?;
        Ok(match self.sol.status_at(&t) {
            Some(s) => status_name(s).into_py(py),
            None => py.None(),
        })
    }

    /// An optimal point at ``theta`` (``None`` where there is none).
    fn x_at(&self, py: Python<'_>, theta: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let t = to_rational(theta, "theta")?;
        match self.sol.region_at(&t).and_then(|r| r.certificate_at(&t)) {
            Some(c) => rat_list(py, &c.x),
            None => Ok(py.None()),
        }
    }

    /// ``{variable: value}`` at ``theta`` for :func:`linprog_expr` results.
    fn solution_at(&self, py: Python<'_>, theta: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let t = to_rational(theta, "theta")?;
        let cert = self.sol.region_at(&t).and_then(|r| r.certificate_at(&t));
        solution_dict(py, &self.vars, cert.as_ref().map(|c| c.x.as_slice()))
    }

    /// The programme at ``θ = 0`` in matrix form (see
    /// :meth:`LPResult.program`), plus its ``θ``-direction under
    /// ``c_theta`` or ``b_ub_theta`` / ``b_eq_theta``.
    fn program(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = program_dict(py, &self.lp)?;
        let b = d.bind(py);
        match &self.delta {
            ParametricDelta::Rhs { b_ub, b_eq } => {
                b.set_item("b_ub_theta", rat_list(py, b_ub)?)?;
                b.set_item("b_eq_theta", rat_list(py, b_eq)?)?;
            }
            ParametricDelta::Objective { c } => b.set_item("c_theta", rat_list(py, c)?)?,
            _ => {}
        }
        Ok(d)
    }

    /// Re-verify every optimal region's certificate exactly, for all ``θ`` in
    /// it; ``True`` when they all hold.
    fn verify(&self) -> bool {
        self.sol.verify(&self.lp, &self.delta).is_ok()
    }

    fn __repr__(&self) -> String {
        let parts: Vec<String> = self
            .sol
            .regions
            .iter()
            .map(|r| {
                let lo = r.lower.as_ref().map_or("-oo".into(), |v| v.to_string());
                let hi = r.upper.as_ref().map_or("oo".into(), |v| v.to_string());
                match &r.value {
                    Some((a, b)) => format!("[{lo}, {hi}]: {a} + {b}*theta"),
                    None => format!("({lo}, {hi}): {}", status_name(r.status)),
                }
            })
            .collect();
        format!("ParametricLPResult({})", parts.join("; "))
    }
}

/// Solve a linear programme whose right-hand side **or** objective is
/// affine in one parameter ``θ``, exactly, for every ``θ`` at once.
///
/// Arguments as for :func:`linprog`, plus the ``θ``-direction: either
/// ``c_theta`` (the objective is ``(c + θ·c_theta) @ x``) or ``b_ub_theta`` /
/// ``b_eq_theta`` (the right-hand sides are ``b_ub + θ·b_ub_theta`` and
/// ``b_eq + θ·b_eq_theta``; a missing one is zero) — not both. ``theta_range``
/// is a ``(lo, hi)`` pair, ``None`` (or ``±inf``) for an infinite end; the
/// default is the whole line.
///
/// The optimal value is piecewise affine in ``θ`` (convex or concave); the
/// result lists the exact breakpoints and, for each critical region, the
/// optimal basis, value, point and multipliers as affine functions of ``θ``,
/// each region's certificate verified exactly for **every** ``θ`` in it.
/// Regions where the programme is infeasible (RHS case) or unbounded
/// (objective case) are reported as such.
///
/// Returns:
///     :class:`ParametricLPResult`.
///
/// Raises:
///     LinprogError: ``E-LP-007`` when both ``c_theta`` and a right-hand-side
///         direction are given (a parameter in the constraint matrix cannot be
///         stated here; :func:`linprog_expr` refuses it with the same code);
///         ``E-LP-001`` a direction of the wrong length or an empty
///         ``theta_range``; and the codes of :func:`linprog`.
///     BudgetExceededError: the active budget tripped.
///
/// Example:
///     >>> r = alkahest.linprog_parametric([1, 1], A_ub=[[1, 2], [3, 1]], b_ub=[4, 6],
///     ...                                 b_ub_theta=[1, 0], sense="max")
///     >>> r.breakpoints
///     [-4, -2, 8]
///     >>> r.value_at(0), r.value_at(10)
///     (Fraction(14, 5), 6)
#[pyfunction]
#[pyo3(name = "linprog_parametric")]
#[pyo3(signature = (c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, sense="min", *, c_theta=None, b_ub_theta=None, b_eq_theta=None, theta_range=None))]
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn py_linprog_parametric(
    c: &Bound<'_, PyAny>,
    A_ub: Option<&Bound<'_, PyAny>>,
    b_ub: Option<&Bound<'_, PyAny>>,
    A_eq: Option<&Bound<'_, PyAny>>,
    b_eq: Option<&Bound<'_, PyAny>>,
    bounds: Option<&Bound<'_, PyAny>>,
    sense: &str,
    c_theta: Option<&Bound<'_, PyAny>>,
    b_ub_theta: Option<&Bound<'_, PyAny>>,
    b_eq_theta: Option<&Bound<'_, PyAny>>,
    theta_range: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyParametricResult> {
    let lp = build(c, A_ub, b_ub, A_eq, b_eq, bounds, sense)?;
    let given = |o: Option<&Bound<'_, PyAny>>| o.is_some_and(|o| !o.is_none());
    let rhs_given = given(b_ub_theta) || given(b_eq_theta);
    let delta = if let Some(ct) = c_theta.filter(|o| !o.is_none()) {
        if rhs_given {
            return Err(lp_err(LinprogError::UnsupportedParametric {
                detail: "c_theta together with b_ub_theta / b_eq_theta: only one of the \
                         objective and the right-hand side may move"
                    .into(),
            }));
        }
        ParametricDelta::Objective {
            c: to_vector(ct, "c_theta")?,
        }
    } else {
        let or_zero =
            |o: Option<&Bound<'_, PyAny>>, n: usize, what: &str| -> PyResult<Vec<Rational>> {
                match o.filter(|o| !o.is_none()) {
                    Some(v) => to_vector(v, what),
                    None => Ok(vec![Rational::new(); n]),
                }
            };
        ParametricDelta::Rhs {
            b_ub: or_zero(b_ub_theta, lp.b_ub.len(), "b_ub_theta")?,
            b_eq: or_zero(b_eq_theta, lp.b_eq.len(), "b_eq_theta")?,
        }
    };
    let (lo, hi) = parse_theta_range(theta_range)?;
    let sol = core_parametric(&lp, &delta, lo, hi).map_err(lp_err)?;
    Ok(PyParametricResult {
        lp,
        delta,
        sol,
        vars: None,
    })
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
    m.add_class::<PyParametricResult>()?;
    m.add("LinprogError", m.py().get_type_bound::<PyLinprogError>())?;
    m.add_function(wrap_pyfunction!(py_linprog, m)?)?;
    m.add_function(wrap_pyfunction!(py_milp, m)?)?;
    m.add_function(wrap_pyfunction!(py_linprog_expr, m)?)?;
    m.add_function(wrap_pyfunction!(py_linprog_parametric, m)?)?;
    m.add_function(wrap_pyfunction!(py_linear_form, m)?)?;
    Ok(())
}
