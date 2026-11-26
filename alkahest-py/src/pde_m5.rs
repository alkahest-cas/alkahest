//! PyO3 bindings for PDE milestone M5: Green's functions and fundamental
//! solutions (`alkahest_core::pde::green`).
//!
//! The private `_pde_*` functions below return plain dicts; the Python
//! classes in `python/alkahest/experimental/_pde_m5.py` wrap them.

use alkahest_core::kernel::ExprId;
use alkahest_core::pde::{
    eigen_green_function, fundamental_solution, fundamental_solution_nd, green_solve,
    image_green_function, ode_green_function, FundamentalSolution, GreenCheck, GreenFunction,
    GreenOperator, GreenSolution, ImageRegion, PdeError, PdeProblem, SeriesKind,
    SturmLiouvilleProblem, WallCondition,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

use crate::pde::{
    conditions_from, domain_from_spec, expr_in, exprs_in, first_pool, pde_err, wrap, wrap_list,
};
use crate::pde_m3::boundary_from;
use crate::PyExprPool;

fn value_err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

/// `(name, {param: expr})` → [`GreenOperator`].
fn operator_from(
    name: &str,
    params: Option<&Bound<'_, PyDict>>,
    pool: &Py<PyExprPool>,
) -> PyResult<GreenOperator> {
    let get = |key: &str| -> PyResult<ExprId> {
        let d = params.ok_or_else(|| value_err(format!("the {name} operator needs {key}=")))?;
        let v = d
            .get_item(key)?
            .ok_or_else(|| value_err(format!("the {name} operator needs {key}=")))?;
        expr_in(&v, pool, "fundamental_solution")
    };
    Ok(match name.to_ascii_lowercase().as_str() {
        "laplace" | "laplacian" => GreenOperator::Laplace,
        "biharmonic" => GreenOperator::Biharmonic,
        "helmholtz" => GreenOperator::Helmholtz { k: get("k")? },
        "modified_helmholtz" => GreenOperator::ModifiedHelmholtz { k: get("k")? },
        "heat" | "diffusion" => GreenOperator::Heat {
            diffusivity: get("diffusivity")?,
        },
        "wave" => GreenOperator::Wave {
            speed: get("speed")?,
        },
        "klein_gordon" => GreenOperator::KleinGordon {
            speed: get("speed")?,
            mass: get("mass")?,
        },
        "telegraph" => GreenOperator::Telegraph {
            speed: get("speed")?,
            damping: get("damping")?,
        },
        other => {
            return Err(value_err(format!(
                "unknown operator {other:?}: use 'laplace', 'biharmonic', 'helmholtz', \
                 'modified_helmholtz', 'heat', 'wave', 'klein_gordon' or 'telegraph'"
            )))
        }
    })
}

fn checks_list<'py>(py: Python<'py>, checks: &[GreenCheck]) -> PyResult<Bound<'py, PyList>> {
    let l = PyList::empty(py);
    for c in checks {
        let d = PyDict::new(py);
        d.set_item("what", c.what)?;
        d.set_item("status", c.status.as_str())?;
        d.set_item("method", c.method.clone())?;
        d.set_item("detail", c.detail.clone())?;
        l.append(d)?;
    }
    Ok(l)
}

fn fundamental_dict<'py>(
    py: Python<'py>,
    f: &FundamentalSolution,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("operator", f.operator.name())?;
    d.set_item("dimension", wrap(py, f.dimension, pool))?;
    d.set_item("space", wrap_list(py, &f.space, pool)?)?;
    d.set_item("time", f.time.map(|t| wrap(py, t, pool)))?;
    d.set_item("radius", wrap(py, f.radius, pool))?;
    d.set_item("kernel", wrap(py, f.kernel, pool))?;
    d.set_item("radial", wrap(py, f.radial, pool))?;
    d.set_item("smooth", wrap(py, f.smooth, pool))?;
    d.set_item("support", f.support.map(|s| wrap(py, s, pool)))?;
    d.set_item("identity", f.identity.clone())?;
    d.set_item("identity_proof", f.identity_proof.clone())?;
    d.set_item("derivation", f.derivation.clone())?;
    d.set_item("checks", checks_list(py, &f.checks)?)?;
    d.set_item("status", f.status.as_str())?;
    d.set_item("verified", f.status.is_verified())?;
    d.set_item("side_conditions", wrap_list(py, &f.side_conditions, pool)?)?;
    d.set_item("references", f.references.clone())?;
    Ok(d)
}

fn green_dict<'py>(
    py: Python<'py>,
    g: &GreenFunction,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("operator", g.operator.clone())?;
    d.set_item("method", g.method.clone())?;
    d.set_item("space", wrap_list(py, &g.space, pool)?)?;
    d.set_item("source", wrap_list(py, &g.source, pool)?)?;
    d.set_item("time", g.time.map(|t| wrap(py, t, pool)))?;
    d.set_item("value", wrap(py, g.value, pool))?;
    d.set_item("series", g.series.map(|s| wrap(py, s, pool)))?;
    d.set_item(
        "branches",
        g.branches
            .map(|(a, b)| (wrap(py, a, pool), wrap(py, b, pool))),
    )?;
    d.set_item(
        "poisson_kernel",
        g.poisson_kernel.map(|s| wrap(py, s, pool)),
    )?;
    d.set_item("boundary_vars", wrap_list(py, &g.boundary_vars, pool)?)?;
    d.set_item("checks", checks_list(py, &g.checks)?)?;
    d.set_item("status", g.status.as_str())?;
    d.set_item("verified", g.status.is_verified())?;
    d.set_item("side_conditions", wrap_list(py, &g.side_conditions, pool)?)?;
    d.set_item("notes", g.notes.clone())?;
    d.set_item("references", g.references.clone())?;
    Ok(d)
}

fn solution_dict<'py>(
    py: Python<'py>,
    s: &GreenSolution,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("unknown", wrap(py, s.unknown, pool))?;
    d.set_item("kind", s.kind.as_str())?;
    d.set_item("method", s.method.clone())?;
    d.set_item("value", wrap(py, s.value, pool))?;
    d.set_item("closed_part", wrap(py, s.closed_part, pool))?;
    let terms = PyList::empty(py);
    for t in &s.terms {
        let td = PyDict::new(py);
        td.set_item("role", t.role)?;
        td.set_item("kernel", wrap(py, t.kernel, pool))?;
        td.set_item("density", wrap(py, t.density, pool))?;
        td.set_item("vars", wrap_list(py, &t.vars, pool)?)?;
        let bounds = PyList::empty(py);
        for &(lo, hi) in &t.bounds {
            bounds.append((wrap(py, lo, pool), wrap(py, hi, pool)))?;
        }
        td.set_item("bounds", bounds)?;
        td.set_item("closed", t.closed.map(|c| wrap(py, c, pool)))?;
        terms.append(td)?;
    }
    d.set_item("terms", terms)?;
    d.set_item("green", wrap(py, s.green, pool))?;
    d.set_item("source", wrap_list(py, &s.source, pool)?)?;
    d.set_item("checks", checks_list(py, &s.checks)?)?;
    d.set_item("status", s.status.as_str())?;
    d.set_item("verified", s.status.is_verified())?;
    d.set_item("side_conditions", wrap_list(py, &s.side_conditions, pool)?)?;
    d.set_item("notes", s.notes.clone())?;
    d.set_item("steps", s.steps.clone())?;
    Ok(d)
}

/// Private: an entry of the fundamental-solution table.
#[pyfunction]
#[pyo3(signature = (operator, space, time=None, params=None, dimension=None, radius=None))]
fn _pde_fundamental_solution<'py>(
    py: Python<'py>,
    operator: &str,
    space: &Bound<'_, PyAny>,
    time: Option<&Bound<'_, PyAny>>,
    params: Option<&Bound<'_, PyDict>>,
    dimension: Option<&Bound<'_, PyAny>>,
    radius: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = match dimension {
        Some(n) => first_pool(n)?,
        None => first_pool(space)?,
    };
    let op = operator_from(operator, params, &pool)?;
    let tt = time
        .map(|t| expr_in(t, &pool, "fundamental_solution"))
        .transpose()?;
    let fs = if let Some(n) = dimension {
        let n = expr_in(n, &pool, "fundamental_solution")?;
        let r = radius
            .ok_or_else(|| value_err("a symbolic dimension needs radius="))
            .and_then(|r| expr_in(r, &pool, "fundamental_solution"))?;
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| fundamental_solution_nd(op, n, r, tt, pl))
            .map_err(pde_err)?
    } else {
        let sp = exprs_in(space, &pool, "fundamental_solution")?;
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| fundamental_solution(op, &sp, tt, pl))
            .map_err(pde_err)?
    };
    fundamental_dict(py, &fs, &pool)
}

fn wall_from(s: &str) -> PyResult<WallCondition> {
    match s.to_ascii_lowercase().as_str() {
        "dirichlet" => Ok(WallCondition::Dirichlet),
        "neumann" => Ok(WallCondition::Neumann),
        other => Err(value_err(format!(
            "a wall condition is 'dirichlet' or 'neumann', not {other:?}"
        ))),
    }
}

/// Private: the method of images.  `region` is `("half_space", axis)`,
/// `("quarter_space", a, b)`, `("strip", axis, width)`, `("disk", R)` or
/// `("ball", R)`.
#[pyfunction]
#[pyo3(signature = (operator, space, source, region, walls, time=None, params=None))]
#[allow(clippy::too_many_arguments)]
fn _pde_image_green<'py>(
    py: Python<'py>,
    operator: &str,
    space: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    region: &Bound<'_, PyTuple>,
    walls: Vec<String>,
    time: Option<&Bound<'_, PyAny>>,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(space)?;
    let op = operator_from(operator, params, &pool)?;
    let sp = exprs_in(space, &pool, "image_green_function")?;
    let so = exprs_in(source, &pool, "image_green_function")?;
    let tt = time
        .map(|t| expr_in(t, &pool, "image_green_function"))
        .transpose()?;
    let kind: String = region.get_item(0)?.extract()?;
    let reg = match kind.to_ascii_lowercase().as_str() {
        "half_space" | "half_plane" | "half_line" => ImageRegion::HalfSpace {
            axis: region.get_item(1)?.extract()?,
        },
        "quarter_space" | "quarter_plane" => ImageRegion::QuarterSpace {
            axes: (
                region.get_item(1)?.extract()?,
                region.get_item(2)?.extract()?,
            ),
        },
        "strip" | "slab" | "interval" => ImageRegion::Strip {
            axis: region.get_item(1)?.extract()?,
            width: expr_in(&region.get_item(2)?, &pool, "image_green_function")?,
        },
        "disk" => ImageRegion::Disk {
            radius: expr_in(&region.get_item(1)?, &pool, "image_green_function")?,
        },
        "ball" => ImageRegion::Ball {
            radius: expr_in(&region.get_item(1)?, &pool, "image_green_function")?,
        },
        other => {
            return Err(value_err(format!(
                "unknown region {other:?}: use half_space, quarter_space, strip, disk or ball"
            )))
        }
    };
    let ws: Vec<WallCondition> = walls
        .iter()
        .map(|w| wall_from(w))
        .collect::<PyResult<_>>()?;
    let gf = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| image_green_function(op, &sp, tt, &so, reg, &ws, pl))
            .map_err(pde_err)?
    };
    green_dict(py, &gf, &pool)
}

/// Private: eigenfunction-series (`kind` = "static" / "heat" / "wave") or
/// variation-of-parameters (`kind` = "closed") Green's function of a
/// Sturm–Liouville problem.
#[pyfunction]
#[pyo3(signature = (var, p, q, w, a, b, left, right, periodic, xi, kind="static", param=None, time=None))]
#[allow(clippy::too_many_arguments)]
fn _pde_sl_green<'py>(
    py: Python<'py>,
    var: &Bound<'_, PyAny>,
    p: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    periodic: bool,
    xi: &Bound<'_, PyAny>,
    kind: &str,
    param: Option<&Bound<'_, PyAny>>,
    time: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(var)?;
    let e = |o: &Bound<'_, PyAny>| expr_in(o, &pool, "green_function");
    let (v, pp, qq, ww, aa, bb, xx) = (e(var)?, e(p)?, e(q)?, e(w)?, e(a)?, e(b)?, e(xi)?);
    let boundary = boundary_from(left, right, periodic, &pool)?;
    let need = |o: Option<&Bound<'_, PyAny>>, what: &str| -> PyResult<ExprId> {
        o.ok_or_else(|| value_err(format!("the {kind} Green's function needs {what}")))
            .and_then(e)
    };
    enum K {
        Closed,
        Series(SeriesKind),
    }
    let k = match kind {
        "closed" => K::Closed,
        "static" => K::Series(SeriesKind::Static),
        "heat" => K::Series(SeriesKind::Heat {
            diffusivity: need(param, "a diffusivity")?,
            time: need(time, "a time variable")?,
        }),
        "wave" => K::Series(SeriesKind::Wave {
            speed: need(param, "a speed")?,
            time: need(time, "a time variable")?,
        }),
        other => {
            return Err(value_err(format!(
                "kind must be 'closed', 'static', 'heat' or 'wave', not {other:?}"
            )))
        }
    };
    let gf = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| -> Result<GreenFunction, PdeError> {
            let prob = SturmLiouvilleProblem::new(v, pp, qq, ww, aa, bb, boundary, pl)?;
            match k {
                K::Closed => ode_green_function(&prob, xx, pl),
                K::Series(s) => eigen_green_function(&prob, xx, s, pl),
            }
        })
        .map_err(pde_err)?
    };
    green_dict(py, &gf, &pool)
}

/// Private: `u = ∫ G f + boundary terms` for a validated problem.
#[pyfunction]
fn _pde_green_solve<'py>(
    py: Python<'py>,
    equations: &Bound<'_, PyAny>,
    unknowns: &Bound<'_, PyAny>,
    domain: &Bound<'_, PyAny>,
    conditions: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknowns)?;
    let eqs = exprs_in(equations, &pool, "green_solve")?;
    let us = exprs_in(unknowns, &pool, "green_solve")?;
    let dom = domain_from_spec(domain, &pool)?;
    let conds = conditions_from(py, conditions, &pool)?;
    let sol = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| -> Result<GreenSolution, PdeError> {
            let mut prob = PdeProblem::new(&eqs, &us, pl)?.on(dom, pl)?;
            for c in conds {
                prob = match c.kind {
                    alkahest_core::pde::ConditionKind::Initial => prob.ic(c.lhs, c.rhs, pl)?,
                    _ => prob.bc(c.lhs, c.rhs, pl)?,
                };
            }
            green_solve(&prob, pl)
        })
        .map_err(pde_err)?
    };
    solution_dict(py, &sol, &pool)
}

/// Register the M5 surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_pde_fundamental_solution, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_image_green, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_sl_green, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_green_solve, m)?)?;
    Ok(())
}
