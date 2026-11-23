//! PyO3 bindings for PDE milestone M3: the Sturm–Liouville eigenproblem
//! solver (`alkahest_core::pde::sturm_liouville`), the transform methods
//! (`alkahest_core::pde::transform_methods`) and the Fourier sine / cosine
//! transforms (`alkahest_core::transform::sincos`).
//!
//! The private `_pde_*` functions below return plain dicts; the Python
//! classes in `python/alkahest/experimental/_pde_m3.py` wrap them.

use std::collections::HashMap;

use alkahest_core::kernel::ExprId;
use alkahest_core::pde::{
    family_enclosures, sturm_liouville_with, EigenEnclosure, EigenFamily, EigenMode, Eigensystem,
    EndCondition, FamilyRef, LowerSpectrum, PdeError, PdeProblem, SlBoundary, SlOptions,
    SturmLiouvilleProblem, TransformSolution,
};
use alkahest_core::transform::{SinCosError, SinCosKind};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

use crate::pde::{
    conditions_from, domain_from_spec, expr_in, exprs_in, first_pool, pde_err, wrap, wrap_list,
};
use crate::PyExprPool;

/// `"dirichlet"`, `"neumann"`, `"bounded"`, `"decay"` or
/// `("robin", alpha, beta)`.
fn end_from(ob: &Bound<'_, PyAny>, pool: &Py<PyExprPool>) -> PyResult<EndCondition> {
    if let Ok(s) = ob.extract::<String>() {
        return Ok(match s.to_ascii_lowercase().as_str() {
            "dirichlet" => EndCondition::Dirichlet,
            "neumann" => EndCondition::Neumann,
            "bounded" => EndCondition::Bounded,
            "decay" => EndCondition::Decay,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "end condition must be 'dirichlet', 'neumann', 'bounded', 'decay' or \
                     ('robin', alpha, beta), not {other:?}"
                )))
            }
        });
    }
    let t: Bound<'_, PyTuple> = ob.extract()?;
    let tag: String = t.get_item(0)?.extract()?;
    if tag.to_ascii_lowercase() != "robin" || t.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a tuple end condition is ('robin', alpha, beta)",
        ));
    }
    Ok(EndCondition::Robin {
        alpha: expr_in(&t.get_item(1)?, pool, "sturm_liouville")?,
        beta: expr_in(&t.get_item(2)?, pool, "sturm_liouville")?,
    })
}

fn boundary_from(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    periodic: bool,
    pool: &Py<PyExprPool>,
) -> PyResult<SlBoundary> {
    if periodic {
        return Ok(SlBoundary::Periodic);
    }
    Ok(SlBoundary::Separated {
        left: end_from(left, pool)?,
        right: end_from(right, pool)?,
    })
}

fn opt_expr(py: Python<'_>, e: Option<ExprId>, pool: &Py<PyExprPool>) -> Py<PyAny> {
    match e {
        Some(id) => wrap(py, id, pool),
        None => py.None(),
    }
}

fn enclosure_tuples(encs: &[EigenEnclosure]) -> Vec<(i64, f64, f64)> {
    encs.iter()
        .map(|e| {
            (
                e.index,
                e.lo.to_f64_round(rug::float::Round::Down),
                e.hi.to_f64_round(rug::float::Round::Up),
            )
        })
        .collect()
}

fn family_dict<'py>(
    py: Python<'py>,
    f: &EigenFamily,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("index", wrap(py, f.index, pool))?;
    d.set_item("first", f.first)?;
    d.set_item("frequency_var", wrap(py, f.frequency_var, pool))?;
    d.set_item("frequency", wrap(py, f.frequency, pool))?;
    d.set_item("eigenvalue", wrap(py, f.eigenvalue, pool))?;
    d.set_item("eigenvalue_of_frequency", wrap(py, f.eigenvalue_of_frequency, pool))?;
    d.set_item("eigenfunctions", wrap_list(py, &f.eigenfunctions, pool)?)?;
    d.set_item(
        "eigenfunctions_of_frequency",
        wrap_list(py, &f.eigenfunctions_of_frequency, pool)?,
    )?;
    let norms = PyList::empty(py);
    for n in &f.norms_squared {
        norms.append(opt_expr(py, *n, pool))?;
    }
    d.set_item("norms_squared", norms)?;
    d.set_item("closed_form", f.closed_form)?;
    d.set_item("defining_equation", opt_expr(py, f.defining_equation, pool))?;
    d.set_item("enclosures", enclosure_tuples(&f.enclosures))?;
    let v = PyDict::new(py);
    v.set_item("status", f.verification.status.as_str())?;
    v.set_item("verified", f.verification.status.is_verified())?;
    v.set_item("ode", f.verification.ode.as_str())?;
    v.set_item("left", f.verification.left.as_str())?;
    v.set_item("right", f.verification.right.as_str())?;
    v.set_item("method", f.verification.method)?;
    v.set_item("members", f.verification.members.clone())?;
    d.set_item("verification", v)?;
    Ok(d)
}

fn mode_dict<'py>(
    py: Python<'py>,
    m: &EigenMode,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("eigenvalue", wrap(py, m.eigenvalue, pool))?;
    d.set_item("eigenfunction", wrap(py, m.eigenfunction, pool))?;
    d.set_item("norm_squared", opt_expr(py, m.norm_squared, pool))?;
    d.set_item("condition", opt_expr(py, m.condition, pool))?;
    d.set_item("verification", m.verification.as_str())?;
    Ok(d)
}

fn system_dict<'py>(
    py: Python<'py>,
    s: &Eigensystem,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", s.kind)?;
    d.set_item("var", wrap(py, s.var, pool))?;
    d.set_item("weight", wrap(py, s.weight, pool))?;
    d.set_item(
        "interval",
        (wrap(py, s.interval.0, pool), wrap(py, s.interval.1, pool)),
    )?;
    d.set_item("threshold", wrap(py, s.threshold, pool))?;
    let fams = PyList::empty(py);
    for f in &s.families {
        fams.append(family_dict(py, f, pool)?)?;
    }
    d.set_item("families", fams)?;
    let modes = PyList::empty(py);
    for m in &s.modes {
        modes.append(mode_dict(py, m, pool)?)?;
    }
    d.set_item("modes", modes)?;
    let lower = PyDict::new(py);
    match &s.lower {
        LowerSpectrum::None => {
            lower.set_item("status", "none")?;
        }
        LowerSpectrum::Conditional { condition, family } => {
            lower.set_item("status", "conditional")?;
            lower.set_item("condition", wrap(py, *condition, pool))?;
            if let Some(f) = family {
                lower.set_item("family", family_dict(py, f, pool)?)?;
            }
        }
        LowerSpectrum::Present { family, count } => {
            lower.set_item("status", "present")?;
            lower.set_item("family", family_dict(py, family, pool)?)?;
            lower.set_item("count", *count)?;
        }
        _ => {
            lower.set_item("status", "unknown")?;
        }
    }
    d.set_item("lower", lower)?;
    d.set_item("side_conditions", wrap_list(py, &s.side_conditions, pool)?)?;
    d.set_item("notes", s.notes.clone())?;
    Ok(d)
}

fn options(enclosures: usize, bits: u32, index: &str) -> SlOptions {
    SlOptions::default()
        .with_enclosures(enclosures)
        .with_index_name(index)
        .with_bits(bits)
}

/// Private: solve `(p X')' + q X + λ w X = 0` on `(a, b)`.
#[pyfunction]
#[pyo3(signature = (var, p, q, w, a, b, left, right, periodic=false, enclosures=5, bits=64, index="n"))]
#[allow(clippy::too_many_arguments)]
fn _pde_sturm_liouville<'py>(
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
    enclosures: usize,
    bits: u32,
    index: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(var)?;
    let e = |o: &Bound<'_, PyAny>| expr_in(o, &pool, "sturm_liouville");
    let (v, pp, qq, ww, aa, bb) = (e(var)?, e(p)?, e(q)?, e(w)?, e(a)?, e(b)?);
    let boundary = boundary_from(left, right, periodic, &pool)?;
    let opts = options(enclosures, bits, index);
    let sys = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| {
            let prob = SturmLiouvilleProblem::new(v, pp, qq, ww, aa, bb, boundary, pl)?;
            sturm_liouville_with(&prob, &opts, pl)
        })
        .map_err(pde_err)?
    };
    system_dict(py, &sys, &pool)
}

/// Private: read the eigenproblem off an ODE in `X(x)` and `λ`, then solve.
#[pyfunction]
#[pyo3(signature = (equation, unknown, lam, a, b, left, right, periodic=false, enclosures=5, bits=64, index="n"))]
#[allow(clippy::too_many_arguments)]
fn _pde_sl_from_ode<'py>(
    py: Python<'py>,
    equation: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    lam: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    periodic: bool,
    enclosures: usize,
    bits: u32,
    index: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknown)?;
    let e = |o: &Bound<'_, PyAny>| expr_in(o, &pool, "sturm_liouville");
    let (eq, u, l, aa, bb) = (e(equation)?, e(unknown)?, e(lam)?, e(a)?, e(b)?);
    let boundary = boundary_from(left, right, periodic, &pool)?;
    let opts = options(enclosures, bits, index);
    let sys = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| {
            let prob = SturmLiouvilleProblem::from_ode(eq, u, l, aa, bb, boundary, pl)?;
            sturm_liouville_with(&prob, &opts, pl)
        })
        .map_err(pde_err)?
    };
    system_dict(py, &sys, &pool)
}

/// Private: certified enclosures of a family's first eigenvalues after
/// substituting numbers for its parameters.
#[pyfunction]
#[pyo3(signature = (index, first, frequency_var, frequency, eigenvalue_of_frequency, closed_form, count, params=None, bits=64))]
#[allow(clippy::too_many_arguments)]
fn _pde_sl_enclosures(
    py: Python<'_>,
    index: &Bound<'_, PyAny>,
    first: i64,
    frequency_var: &Bound<'_, PyAny>,
    frequency: &Bound<'_, PyAny>,
    eigenvalue_of_frequency: &Bound<'_, PyAny>,
    closed_form: bool,
    count: usize,
    params: Option<&Bound<'_, PyDict>>,
    bits: u32,
) -> PyResult<Vec<(i64, f64, f64)>> {
    let pool = first_pool(index)?;
    let e = |o: &Bound<'_, PyAny>| expr_in(o, &pool, "EigenFamily.enclosures");
    let fam = FamilyRef {
        index: e(index)?,
        first,
        frequency_var: e(frequency_var)?,
        frequency: e(frequency)?,
        eigenvalue_of_frequency: e(eigenvalue_of_frequency)?,
        closed_form,
    };
    let mut map = HashMap::new();
    if let Some(d) = params {
        let g = pool.borrow(py);
        for (k, v) in d.iter() {
            let kk = e(&k)?;
            let vv = match expr_in(&v, &pool, "EigenFamily.enclosures") {
                Ok(id) => id,
                Err(_) => {
                    if let Ok(i) = v.extract::<i64>() {
                        g.inner.integer(i)
                    } else {
                        let f: f64 = v.extract()?;
                        g.inner.float(f, 53)
                    }
                }
            };
            map.insert(kk, vv);
        }
    }
    let encs = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| family_enclosures(fam, count, &map, bits, pl))
            .map_err(pde_err)?
    };
    Ok(enclosure_tuples(&encs))
}

fn solution_dict<'py>(
    py: Python<'py>,
    s: &TransformSolution,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", s.kind.as_str())?;
    d.set_item("method", s.method.clone())?;
    d.set_item("value", wrap(py, s.value, pool))?;
    d.set_item("closed_part", wrap(py, s.closed_part, pool))?;
    let terms = PyList::empty(py);
    for t in &s.terms {
        let td = PyDict::new(py);
        td.set_item("kernel", wrap(py, t.kernel, pool))?;
        td.set_item("density", wrap(py, t.density, pool))?;
        td.set_item("var", wrap(py, t.var, pool))?;
        td.set_item("lower", wrap(py, t.lower, pool))?;
        td.set_item("upper", wrap(py, t.upper, pool))?;
        terms.append(td)?;
    }
    d.set_item("terms", terms)?;
    d.set_item("transformed", wrap(py, s.transformed, pool))?;
    d.set_item("transform_var", wrap(py, s.transform_var, pool))?;
    let v = PyDict::new(py);
    v.set_item("status", s.status.as_str())?;
    v.set_item("verified", s.status.is_verified())?;
    let checks = PyList::empty(py);
    for c in &s.checks {
        checks.append((c.what, c.index, c.status.as_str()))?;
    }
    v.set_item("checks", checks)?;
    d.set_item("verification", v)?;
    d.set_item("side_conditions", wrap_list(py, &s.side_conditions, pool)?)?;
    d.set_item("notes", s.notes.clone())?;
    Ok(d)
}

/// Private: a transform method on a validated problem.
#[pyfunction]
#[pyo3(signature = (equations, unknowns, domain, conditions, method="auto"))]
fn _pde_transform_solve<'py>(
    py: Python<'py>,
    equations: &Bound<'_, PyAny>,
    unknowns: &Bound<'_, PyAny>,
    domain: &Bound<'_, PyAny>,
    conditions: &Bound<'_, PyAny>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknowns)?;
    let eqs = exprs_in(equations, &pool, "transform_solve")?;
    let us = exprs_in(unknowns, &pool, "transform_solve")?;
    let dom = domain_from_spec(domain, &pool)?;
    let conds = conditions_from(py, conditions, &pool)?;
    let method = method.to_string();
    let sol = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        py.detach(|| -> Result<TransformSolution, PdeError> {
            let mut prob = PdeProblem::new(&eqs, &us, pl)?.on(dom, pl)?;
            for c in conds {
                prob = match c.kind {
                    alkahest_core::pde::ConditionKind::Initial => prob.ic(c.lhs, c.rhs, pl)?,
                    _ => prob.bc(c.lhs, c.rhs, pl)?,
                };
            }
            match method.as_str() {
                "fourier" => alkahest_core::pde::fourier_method(&prob, pl),
                "laplace" => alkahest_core::pde::laplace_method(&prob, pl),
                "image" | "sine" | "cosine" => alkahest_core::pde::image_method(&prob, pl),
                "auto" => alkahest_core::pde::transform_solve(&prob, pl),
                other => Err(PdeError::TransformMethodUnsupported(format!(
                    "unknown method {other:?}: use 'auto', 'fourier', 'laplace' or 'image'"
                ))),
            }
        })
        .map_err(pde_err)?
    };
    solution_dict(py, &sol, &pool)
}

fn sincos_err(e: SinCosError) -> PyErr {
    Python::attach(|py| {
        let exc_type = py.get_type::<crate::PyTransformError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// Private: the Fourier sine / cosine transform (`inverse` for the
/// inverse).
#[pyfunction]
#[pyo3(signature = (f, x, w, kind, inverse=false))]
fn _sincos_transform(
    py: Python<'_>,
    f: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    w: &Bound<'_, PyAny>,
    kind: &str,
    inverse: bool,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(x)?;
    let (ff, xx, ww) = (
        expr_in(f, &pool, "sine_transform")?,
        expr_in(x, &pool, "sine_transform")?,
        expr_in(w, &pool, "sine_transform")?,
    );
    let k = match kind {
        "sine" => SinCosKind::Sine,
        "cosine" => SinCosKind::Cosine,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "kind must be 'sine' or 'cosine', not {other:?}"
            )))
        }
    };
    let r = {
        let g = pool.borrow(py);
        let pl = &g.inner;
        match (k, inverse) {
            (SinCosKind::Sine, false) => alkahest_core::transform::fourier_sine_transform(ff, xx, ww, pl),
            (SinCosKind::Sine, true) => alkahest_core::transform::inverse_fourier_sine_transform(ff, xx, ww, pl),
            (_, false) => alkahest_core::transform::fourier_cosine_transform(ff, xx, ww, pl),
            (_, true) => alkahest_core::transform::inverse_fourier_cosine_transform(ff, xx, ww, pl),
        }
        .map_err(sincos_err)?
    };
    Ok(wrap(py, r, &pool))
}

/// Register the M3 surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_pde_sturm_liouville, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_sl_from_ode, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_sl_enclosures, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_transform_solve, m)?)?;
    m.add_function(wrap_pyfunction!(_sincos_transform, m)?)?;
    Ok(())
}
