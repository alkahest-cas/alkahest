//! PyO3 bindings for the Green's-function gap round: the Neumann problem on a
//! disk (`alkahest_core::pde::neumann_disk_solve`) and the certified count of
//! eigenvalues below the threshold of a Sturm–Liouville problem
//! (`alkahest_core::pde::count_lower_eigenvalues`).
//!
//! The private `_pde_*` functions return plain dicts; the Python wrappers are
//! in `python/alkahest/experimental/_pde_m5.py`.

use std::collections::HashMap;

use alkahest_core::pde::{
    count_lower_eigenvalues, neumann_disk_solve, GreenCheck, LowerCount, NeumannDiskSolution,
    PdeError, SturmLiouvilleProblem,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::pde::{expr_in, first_pool, pde_err, wrap, wrap_list};
use crate::pde_m3::boundary_from;
use crate::PyExprPool;

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

fn neumann_dict<'py>(
    py: Python<'py>,
    s: &NeumannDiskSolution,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("value", wrap(py, s.value, pool))?;
    d.set_item("constant", wrap(py, s.constant, pool))?;
    d.set_item("polar", wrap(py, s.polar, pool))?;
    d.set_item(
        "polar_vars",
        (
            wrap(py, s.polar_vars.0, pool),
            wrap(py, s.polar_vars.1, pool),
        ),
    )?;
    d.set_item("compatibility", wrap(py, s.compatibility, pool))?;
    let co = PyList::empty(py);
    for &(j, a, b) in &s.coefficients {
        co.append((j, wrap(py, a, pool), wrap(py, b, pool)))?;
    }
    d.set_item("coefficients", co)?;
    d.set_item("particular", wrap(py, s.particular, pool))?;
    d.set_item("green", wrap(py, s.green.value, pool))?;
    d.set_item("green_source", wrap_list(py, &s.green.source, pool)?)?;
    d.set_item(
        "boundary_kernel",
        s.green.poisson_kernel.map(|k| wrap(py, k, pool)),
    )?;
    d.set_item("green_verified", s.green.status.is_verified())?;
    d.set_item("checks", checks_list(py, &s.checks)?)?;
    d.set_item("status", s.status.as_str())?;
    d.set_item("verified", s.status.is_verified())?;
    d.set_item("side_conditions", wrap_list(py, &s.side_conditions, pool)?)?;
    d.set_item("notes", s.notes.clone())?;
    d.set_item("references", s.references.clone())?;
    Ok(d)
}

/// Private: the Neumann problem `Δu = f` in `x² + y² < R²`, `∂u/∂r = g(θ)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _pde_neumann_disk<'py>(
    py: Python<'py>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    radius: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    theta: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(x)?;
    let e = |o: &Bound<'_, PyAny>| expr_in(o, &pool, "neumann_disk_solve");
    let (xx, yy, rr, ff, gg, tt) = (e(x)?, e(y)?, e(radius)?, e(f)?, e(g)?, e(theta)?);
    let sol = {
        let gp = pool.borrow(py);
        let pl = &gp.inner;
        py.detach(|| neumann_disk_solve(xx, yy, rr, ff, gg, tt, pl))
            .map_err(pde_err)?
    };
    neumann_dict(py, &sol, &pool)
}

fn count_dict<'py>(
    py: Python<'py>,
    c: &LowerCount,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("count", c.count)?;
    d.set_item("threshold", wrap(py, c.threshold, pool))?;
    d.set_item("threshold_is_eigenvalue", c.threshold_is_eigenvalue)?;
    d.set_item("negative", c.negative)?;
    let enc: Vec<(i64, f64, f64)> = c
        .enclosures
        .iter()
        .map(|e| {
            (
                e.index,
                e.lo.to_f64_round(rug::float::Round::Down),
                e.hi.to_f64_round(rug::float::Round::Up),
            )
        })
        .collect();
    d.set_item("enclosures", enc)?;
    d.set_item("kappa", wrap(py, c.kappa, pool))?;
    d.set_item("characteristic", wrap(py, c.characteristic, pool))?;
    d.set_item("eigenfunction", wrap(py, c.eigenfunction, pool))?;
    d.set_item("method", c.method)?;
    d.set_item(
        "tail_bound",
        c.tail_bound
            .as_ref()
            .map(|t| t.to_f64_round(rug::float::Round::Up)),
    )?;
    d.set_item("cells", c.cells)?;
    let checks = PyList::empty(py);
    for (what, st, detail) in &c.checks {
        let cd = PyDict::new(py);
        cd.set_item("what", *what)?;
        cd.set_item("status", st.as_str())?;
        cd.set_item("detail", detail.clone())?;
        checks.append(cd)?;
    }
    d.set_item("checks", checks)?;
    d.set_item("status", c.status.as_str())?;
    d.set_item("verified", c.status.is_verified())?;
    d.set_item("certificate", c.certificate.clone())?;
    Ok(d)
}

/// Private: the certified count of eigenvalues below the threshold.
#[pyfunction]
#[pyo3(signature = (var, p, q, w, a, b, left, right, params=None))]
#[allow(clippy::too_many_arguments)]
fn _pde_count_lower<'py>(
    py: Python<'py>,
    var: &Bound<'_, PyAny>,
    p: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(var)?;
    let e = |o: &Bound<'_, PyAny>| expr_in(o, &pool, "count_lower_eigenvalues");
    let (v, pp, qq, ww, aa, bb) = (e(var)?, e(p)?, e(q)?, e(w)?, e(a)?, e(b)?);
    let boundary = boundary_from(left, right, false, &pool)?;
    let mut map = HashMap::new();
    if let Some(d) = params {
        for (k, val) in d.iter() {
            map.insert(e(&k)?, e(&val)?);
        }
    }
    let c = {
        let gp = pool.borrow(py);
        let pl = &gp.inner;
        py.detach(|| -> Result<LowerCount, PdeError> {
            let prob = SturmLiouvilleProblem::new(v, pp, qq, ww, aa, bb, boundary, pl)?;
            count_lower_eigenvalues(&prob, &map, pl)
        })
        .map_err(pde_err)?
    };
    count_dict(py, &c, &pool)
}

/// Register the gap-round surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_pde_neumann_disk, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_count_lower, m)?)?;
    Ok(())
}
