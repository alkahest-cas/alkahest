//! PyO3 bindings for PDE milestone M4: separation of variables and
//! eigenfunction-series solutions (`alkahest_core::pde::separation`).
//!
//! The private `_pde_*` functions return plain dicts; the Python classes in
//! `python/alkahest/experimental/_pde_m4.py` wrap them.

use std::collections::HashMap;

use alkahest_core::kernel::ExprId;
use alkahest_core::pde::{
    curvilinear_laplacian, numeric_truncation, separate_with, separation_solve_with,
    CoordinateSystem, PdeError, PdeProblem, SeparateOptions, Separation, SeparationKind,
    SeparationOptions, SeriesSolution,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::pde::{
    conditions_from, domain_from_spec, expr_in, exprs_in, first_pool, pde_err, wrap, wrap_list,
};
use crate::PyExprPool;

fn kind_from(s: &str) -> PyResult<SeparationKind> {
    match s.to_ascii_lowercase().as_str() {
        "product" | "multiplicative" => Ok(SeparationKind::Product),
        "additive" | "sum" => Ok(SeparationKind::Additive),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "kind must be 'product' or 'additive', not {other:?}"
        ))),
    }
}

fn separation_dict<'py>(
    py: Python<'py>,
    s: &Separation,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", s.kind.as_str())?;
    d.set_item("unknown", wrap(py, s.unknown, pool))?;
    d.set_item("ansatz", wrap(py, s.ansatz, pool))?;
    d.set_item("factors", wrap_list(py, &s.factors, pool)?)?;
    d.set_item("constants", wrap_list(py, &s.constants, pool)?)?;
    let eqs = PyList::empty(py);
    for e in &s.equations {
        let ed = PyDict::new(py);
        ed.set_item("var", wrap(py, e.var, pool))?;
        ed.set_item("factor", wrap(py, e.factor, pool))?;
        ed.set_item("equation", wrap(py, e.equation, pool))?;
        ed.set_item(
            "constant",
            match e.constant {
                Some(c) => wrap(py, c, pool),
                None => py.None(),
            },
        )?;
        ed.set_item("multiplier", wrap(py, e.multiplier, pool))?;
        eqs.append(ed)?;
    }
    d.set_item("equations", eqs)?;
    Ok(d)
}

/// Private: separate `equation` in `unknown`.
#[pyfunction]
#[pyo3(signature = (equation, unknown, kind="product", prefer=None, last=None))]
fn _pde_separate<'py>(
    py: Python<'py>,
    equation: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    kind: &str,
    prefer: Option<&Bound<'_, PyAny>>,
    last: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknown)?;
    let eq = expr_in(equation, &pool, "separate")?;
    let u = expr_in(unknown, &pool, "separate")?;
    let k = kind_from(kind)?;
    let mut opts = SeparateOptions::new();
    if let Some(p) = prefer {
        opts = opts.prefer(&exprs_in(p, &pool, "separate")?);
    }
    if let Some(l) = last {
        opts = opts.last(expr_in(l, &pool, "separate")?);
    }
    let sep = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| separate_with(eq, u, k, &opts, pl))
            .map_err(pde_err)?
    };
    separation_dict(py, &sep, &pool)
}

fn solution_dict<'py>(
    py: Python<'py>,
    s: &SeriesSolution,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let opt = |e: Option<ExprId>| match e {
        Some(id) => wrap(py, id, pool),
        None => py.None(),
    };
    let d = PyDict::new(py);
    d.set_item("value", wrap(py, s.value, pool))?;
    d.set_item("kind", s.kind)?;
    d.set_item("method", s.method.clone())?;
    d.set_item("steady_state", opt(s.steady_state))?;
    let modes = PyList::empty(py);
    for g in &s.modes {
        let gd = PyDict::new(py);
        let idx = PyList::empty(py);
        for &(i, f) in &g.indices {
            idx.append((wrap(py, i, pool), wrap(py, f, pool)))?;
        }
        gd.set_item("indices", idx)?;
        gd.set_item("upper", wrap_list(py, &g.upper, pool)?)?;
        let ev = PyList::empty(py);
        for &(c, v) in &g.eigenvalues {
            ev.append((wrap(py, c, pool), wrap(py, v, pool)))?;
        }
        gd.set_item("eigenvalues", ev)?;
        gd.set_item("eigenfunctions", wrap_list(py, &g.eigenfunctions, pool)?)?;
        gd.set_item("weight", wrap(py, g.weight, pool))?;
        gd.set_item("norms_squared", wrap_list(py, &g.norms_squared, pool)?)?;
        let cs = PyList::empty(py);
        for per in &g.coefficients {
            let row = PyDict::new(py);
            for (n, c) in per {
                row.set_item(n.clone(), wrap(py, *c, pool))?;
            }
            cs.append(row)?;
        }
        gd.set_item("coefficients", cs)?;
        gd.set_item("amplitudes", wrap_list(py, &g.amplitudes, pool)?)?;
        gd.set_item("term", wrap(py, g.term, pool))?;
        gd.set_item("value", wrap(py, g.value, pool))?;
        gd.set_item("closed", g.closed)?;
        gd.set_item("status", g.status.as_str())?;
        modes.append(gd)?;
    }
    d.set_item("modes", modes)?;
    d.set_item(
        "separation",
        match &s.separation {
            Some(sep) => separation_dict(py, sep, pool)?.into_any().unbind(),
            None => py.None(),
        },
    )?;
    let v = PyDict::new(py);
    v.set_item("status", s.status.as_str())?;
    v.set_item("verified", s.status.is_verified())?;
    let checks = PyList::empty(py);
    for c in &s.checks {
        let group: Option<usize> = (c.group != usize::MAX).then_some(c.group);
        checks.append((c.what, group, c.status.as_str(), c.method, c.detail.clone()))?;
    }
    v.set_item("checks", checks)?;
    d.set_item("verification", v)?;
    let proj = PyList::empty(py);
    for p in &s.projection {
        let pd = PyDict::new(py);
        pd.set_item("data", p.data)?;
        pd.set_item("status", p.status.as_str())?;
        pd.set_item("terms", p.terms)?;
        pd.set_item("max_error", p.max_error)?;
        pd.set_item("evidence", p.evidence.clone())?;
        proj.append(pd)?;
    }
    d.set_item("projection", proj)?;
    d.set_item("convergence", s.convergence)?;
    d.set_item("side_conditions", wrap_list(py, &s.side_conditions, pool)?)?;
    d.set_item("notes", s.notes.clone())?;
    let alts = PyList::empty(py);
    for (n, e) in &s.alternatives {
        alts.append((n.clone(), wrap(py, *e, pool)))?;
    }
    d.set_item("alternatives", alts)?;
    d.set_item(
        "coordinates",
        match &s.coordinates {
            Some((sys, cs)) => (sys.as_str(), wrap_list(py, cs, pool)?)
                .into_pyobject(py)?
                .into_any()
                .unbind(),
            None => py.None(),
        },
    )?;
    d.set_item("curvilinear_value", opt(s.curvilinear_value))?;
    d.set_item("constants", wrap_list(py, &s.constants, pool)?)?;
    Ok(d)
}

/// Private: separation of variables on a validated problem.
#[pyfunction]
#[pyo3(signature = (equations, unknowns, domain, conditions, check_members=4, steady_state=true, index_names=None))]
#[allow(clippy::too_many_arguments)]
fn _pde_separation_solve<'py>(
    py: Python<'py>,
    equations: &Bound<'_, PyAny>,
    unknowns: &Bound<'_, PyAny>,
    domain: &Bound<'_, PyAny>,
    conditions: &Bound<'_, PyAny>,
    check_members: usize,
    steady_state: bool,
    index_names: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknowns)?;
    let eqs = exprs_in(equations, &pool, "separation_solve")?;
    let us = exprs_in(unknowns, &pool, "separation_solve")?;
    let dom = domain_from_spec(domain, &pool)?;
    let conds = conditions_from(py, conditions, &pool)?;
    let mut opts = SeparationOptions::default();
    opts.check_members = check_members;
    opts.steady_state = steady_state;
    if let Some(n) = index_names {
        opts.index_names = n;
    }
    let sol = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| -> Result<SeriesSolution, PdeError> {
            let mut prob = PdeProblem::new(&eqs, &us, pl)?.on(dom, pl)?;
            for c in conds {
                prob = match c.kind {
                    alkahest_core::pde::ConditionKind::Initial => prob.ic(c.lhs, c.rhs, pl)?,
                    _ => prob.bc(c.lhs, c.rhs, pl)?,
                };
            }
            separation_solve_with(&prob, &opts, pl)
        })
        .map_err(pde_err)?
    };
    solution_dict(py, &sol, &pool)
}

/// Private: the Laplacian of `f` in polar / cylindrical / spherical
/// coordinates (`None` for a coordinate `f` does not depend on).
#[pyfunction]
fn _pde_curvilinear_laplacian(
    py: Python<'_>,
    f: &Bound<'_, PyAny>,
    system: &str,
    coords: Vec<Option<Bound<'_, PyAny>>>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(f)?;
    let ff = expr_in(f, &pool, "laplacian")?;
    let sys = CoordinateSystem::from_name(system).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "system must be 'polar', 'cylindrical' or 'spherical', not {system:?}"
        ))
    })?;
    let mut cs = Vec::new();
    for c in &coords {
        cs.push(match c {
            Some(c) => Some(expr_in(c, &pool, "laplacian")?),
            None => None,
        });
    }
    let r = {
        let g = pool.borrow(py);
        curvilinear_laplacian(ff, sys, &cs, &g.inner).map_err(pde_err)?
    };
    Ok(wrap(py, r, &pool))
}

/// Private: a numeric truncation (indices ≤ n, parameters substituted,
/// RootOfFamily members and unevaluated integrals evaluated).
#[pyfunction]
#[pyo3(signature = (expr, n, params=None))]
fn _pde_numeric_truncation(
    py: Python<'_>,
    expr: &Bound<'_, PyAny>,
    n: i64,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(expr)?;
    let e = expr_in(expr, &pool, "truncate")?;
    let mut map = HashMap::new();
    if let Some(d) = params {
        let g = pool.borrow(py);
        for (k, v) in d.iter() {
            let kk = expr_in(&k, &pool, "truncate")?;
            let vv = match expr_in(&v, &pool, "truncate") {
                Ok(id) => id,
                Err(_) => {
                    if let Ok(i) = v.extract::<i64>() {
                        g.inner.integer(i)
                    } else {
                        g.inner.float(v.extract::<f64>()?, 53)
                    }
                }
            };
            map.insert(kk, vv);
        }
    }
    let r = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| numeric_truncation(e, n, &map, pl))
            .map_err(pde_err)?
    };
    Ok(wrap(py, r, &pool))
}

/// Register the M4 surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_pde_separate, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_separation_solve, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_curvilinear_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_numeric_truncation, m)?)?;
    Ok(())
}
