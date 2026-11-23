//! PyO3 bindings for `alkahest_core::pde::second_order` — milestone M2:
//! canonical forms, closed-form general solutions of linear second-order
//! equations in two variables, and the Laplace cascade.
//!
//! The Python surface is `alkahest.experimental` (`canonical_form`,
//! `canonical_forms`, `solve_second_order`, `laplace_cascade`,
//! `laplace_invariants`), implemented in
//! `python/alkahest/experimental/_pde_second_order.py` over the private
//! `_pde2_*` functions below.

use std::collections::HashMap;

use alkahest_core::kernel::ExprId;
use alkahest_core::pde::{
    canonical_form, canonical_forms, laplace_cascade, laplace_invariants, solve_second_order_with,
    CanonicalForm, ConditionKind, PdeCondition, PdeProblem, PdeVerification, SecondOrderOptions,
    SecondOrderSolution,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::IntoPyObjectExt;

use crate::pde::pde_err;
use crate::{PyExpr, PyExprPool, ToPyAny};

fn wrap(py: Python<'_>, id: ExprId, pool: &Py<PyExprPool>) -> Py<PyAny> {
    PyExpr {
        id,
        pool: pool.clone_ref(py),
    }
    .to_py_any(py)
}

fn wrap_list(py: Python<'_>, ids: &[ExprId], pool: &Py<PyExprPool>) -> PyResult<Py<PyAny>> {
    PyList::new(py, ids.iter().map(|&i| wrap(py, i, pool)))?.into_py_any(py)
}

fn same_pool(a: &PyExpr, b: &PyExpr, op: &str) -> PyResult<()> {
    if !a.pool.is(&b.pool) {
        return Err(crate::cross_pool_err(format!(
            "[E-POOL-001] {op}(): arguments belong to different ExprPools"
        )));
    }
    Ok(())
}

fn verification_dict<'py>(
    py: Python<'py>,
    v: &PdeVerification,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("status", v.status.as_str())?;
    d.set_item("verified", v.status.is_verified())?;
    d.set_item(
        "evidence",
        match v.status.as_str() {
            "exactly_verified" => "symbolic_residual",
            "refuted" => "nonzero_residual",
            "numerically_checked" => "ball_arithmetic_samples",
            _ => "none",
        },
    )?;
    d.set_item("externally_verified", false)?;
    d.set_item("arbitrary_functions", v.arbitrary_functions.clone())?;
    let mut by_kind: HashMap<&str, Bound<'_, PyList>> = HashMap::new();
    for k in ["pde", "bc", "ic"] {
        by_kind.insert(k, PyList::empty(py));
    }
    for c in &v.checks {
        let cd = PyDict::new(py);
        cd.set_item("index", c.index)?;
        cd.set_item("residual", wrap(py, c.residual, pool))?;
        cd.set_item("status", c.status.as_str())?;
        cd.set_item("method", c.method)?;
        cd.set_item("evidence", c.evidence.clone())?;
        if let Some(l) = by_kind.get(c.kind) {
            l.append(cd)?;
        }
    }
    for (k, l) in by_kind {
        d.set_item(k, l)?;
    }
    Ok(d)
}

fn canonical_dict<'py>(
    py: Python<'py>,
    f: &CanonicalForm,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("type", f.pde_type.as_str())?;
    match &f.region {
        Some(r) => {
            let rd = PyDict::new(py);
            rd.set_item("type", r.pde_type.as_str())?;
            rd.set_item("condition", wrap(py, r.condition, pool))?;
            rd.set_item("var", r.var.map(|e| wrap(py, e, pool)))?;
            rd.set_item("lower", r.lower.map(|e| wrap(py, e, pool)))?;
            rd.set_item("upper", r.upper.map(|e| wrap(py, e, pool)))?;
            d.set_item("region", rd)?;
        }
        None => d.set_item("region", py.None())?,
    }
    d.set_item(
        "old_variables",
        wrap_list(py, &[f.old_variables.0, f.old_variables.1], pool)?,
    )?;
    d.set_item(
        "new_variables",
        wrap_list(py, &[f.new_variables.0, f.new_variables.1], pool)?,
    )?;
    d.set_item(
        "transformation",
        wrap_list(py, &[f.transformation.0, f.transformation.1], pool)?,
    )?;
    match f.inverse {
        Some((a, b)) => d.set_item("inverse", wrap_list(py, &[a, b], pool)?)?,
        None => d.set_item("inverse", py.None())?,
    }
    d.set_item("unknown", wrap(py, f.unknown, pool))?;
    d.set_item("equation", wrap(py, f.equation, pool))?;
    d.set_item("coefficients", wrap_list(py, &f.coefficients, pool)?)?;
    d.set_item("in_new_variables", f.in_new_variables)?;
    d.set_item("multiplier", wrap(py, f.multiplier, pool))?;
    d.set_item("conditions", wrap_list(py, &f.conditions, pool)?)?;
    Ok(d)
}

fn solution_dict<'py>(
    py: Python<'py>,
    s: &SecondOrderSolution,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("solution", wrap(py, s.solution, pool))?;
    d.set_item("kind", s.kind)?;
    d.set_item("method", s.method.clone())?;
    d.set_item("arbitrary_functions", s.arbitrary_functions.clone())?;
    d.set_item("parameters", wrap_list(py, &s.parameters, pool)?)?;
    d.set_item("conditions", wrap_list(py, &s.conditions, pool)?)?;
    d.set_item("steps", s.steps.clone())?;
    match &s.canonical {
        Some(c) => d.set_item("canonical", canonical_dict(py, c, pool)?)?,
        None => d.set_item("canonical", py.None())?,
    }
    d.set_item(
        "verification",
        verification_dict(py, &s.verification, pool)?,
    )?;
    Ok(d)
}

/// Private: ``canonical_form`` (``all_regions=False``) or ``canonical_forms``
/// as a list of dicts.
#[pyfunction]
#[pyo3(signature = (equation, unknown, all_regions=false))]
fn _pde2_canonical_form<'py>(
    py: Python<'py>,
    equation: PyRef<'_, PyExpr>,
    unknown: PyRef<'_, PyExpr>,
    all_regions: bool,
) -> PyResult<Bound<'py, PyList>> {
    same_pool(&equation, &unknown, "canonical_form")?;
    let pool = equation.pool.clone_ref(py);
    let forms = {
        let p = pool.borrow(py);
        crate::guard_depth(&p.inner, equation.id)?;
        let pl = &p.inner;
        let (e, u) = (equation.id, unknown.id);
        if all_regions {
            py.detach(|| canonical_forms(e, u, pl)).map_err(pde_err)?
        } else {
            vec![py.detach(|| canonical_form(e, u, pl)).map_err(pde_err)?]
        }
    };
    let out = PyList::empty(py);
    for f in &forms {
        out.append(canonical_dict(py, f, &pool)?)?;
    }
    Ok(out)
}

/// Private: ``solve_second_order`` on loose parts; ``conditions`` is a list of
/// ``(kind, lhs, rhs)`` with kind ``"ic"`` / ``"bc"``.
#[pyfunction]
#[pyo3(signature = (equation, unknown, conditions=None, max_cascade_steps=8, prefix="F", allow_particular=true))]
fn _pde2_solve<'py>(
    py: Python<'py>,
    equation: PyRef<'_, PyExpr>,
    unknown: PyRef<'_, PyExpr>,
    conditions: Option<&Bound<'_, PyAny>>,
    max_cascade_steps: usize,
    prefix: &str,
    allow_particular: bool,
) -> PyResult<Bound<'py, PyDict>> {
    same_pool(&equation, &unknown, "solve_second_order")?;
    let pool = equation.pool.clone_ref(py);
    let mut conds = Vec::new();
    if let Some(cs) = conditions.filter(|o| !o.is_none()) {
        for c in cs.try_iter()? {
            let c = c?;
            let kind: String = c.get_item(0)?.extract()?;
            let lhs: PyRef<'_, PyExpr> = c.get_item(1)?.extract()?;
            let rhs: PyRef<'_, PyExpr> = c.get_item(2)?.extract()?;
            same_pool(&equation, &lhs, "solve_second_order")?;
            same_pool(&equation, &rhs, "solve_second_order")?;
            conds.push((kind, lhs.id, rhs.id));
        }
    }
    let opts = SecondOrderOptions::new()
        .with_max_cascade_steps(max_cascade_steps)
        .with_prefix(prefix)
        .with_allow_particular(allow_particular);
    let sol = {
        let p = pool.borrow(py);
        crate::guard_depth(&p.inner, equation.id)?;
        let pl = &p.inner;
        let (e, u) = (equation.id, unknown.id);
        let mut prob = PdeProblem::new(&[e], &[u], pl).map_err(pde_err)?;
        for (kind, lhs, rhs) in conds {
            let c = PdeCondition {
                kind: if kind == "ic" {
                    ConditionKind::Initial
                } else {
                    ConditionKind::Boundary
                },
                lhs,
                rhs,
            };
            prob = match c.kind {
                ConditionKind::Initial => prob.ic(c.lhs, c.rhs, pl),
                _ => prob.bc(c.lhs, c.rhs, pl),
            }
            .map_err(pde_err)?;
        }
        py.detach(|| solve_second_order_with(&prob, &opts, pl))
            .map_err(pde_err)?
    };
    solution_dict(py, &sol, &pool)
}

/// Private: ``laplace_cascade`` as a dict.
#[pyfunction]
#[pyo3(signature = (equation, unknown, max_steps=8))]
fn _pde2_laplace_cascade<'py>(
    py: Python<'py>,
    equation: PyRef<'_, PyExpr>,
    unknown: PyRef<'_, PyExpr>,
    max_steps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    same_pool(&equation, &unknown, "laplace_cascade")?;
    let pool = equation.pool.clone_ref(py);
    let sol = {
        let p = pool.borrow(py);
        crate::guard_depth(&p.inner, equation.id)?;
        let pl = &p.inner;
        let (e, u) = (equation.id, unknown.id);
        py.detach(|| laplace_cascade(e, u, max_steps, pl))
            .map_err(pde_err)?
    };
    solution_dict(py, &sol, &pool)
}

/// Private: the Laplace invariants ``(h, k)``.
#[pyfunction]
fn _pde2_laplace_invariants(
    py: Python<'_>,
    equation: PyRef<'_, PyExpr>,
    unknown: PyRef<'_, PyExpr>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    same_pool(&equation, &unknown, "laplace_invariants")?;
    let pool = equation.pool.clone_ref(py);
    let (h, k) = {
        let p = pool.borrow(py);
        laplace_invariants(equation.id, unknown.id, &p.inner).map_err(pde_err)?
    };
    Ok((wrap(py, h, &pool), wrap(py, k, &pool)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_pde2_canonical_form, m)?)?;
    m.add_function(wrap_pyfunction!(_pde2_solve, m)?)?;
    m.add_function(wrap_pyfunction!(_pde2_laplace_cascade, m)?)?;
    m.add_function(wrap_pyfunction!(_pde2_laplace_invariants, m)?)?;
    Ok(())
}
