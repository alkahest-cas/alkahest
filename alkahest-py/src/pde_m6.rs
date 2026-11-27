//! Python bindings of the PDE milestone M6: Lie point symmetries,
//! conservation laws, reductions and linearising transformations.  The
//! private `_pde_*` functions return plain dicts; the public wrappers live in
//! `alkahest.experimental._pde_nonlinear`.

use alkahest_core::kernel::{ExprId, ExprPool};
use alkahest_core::pde::{self as core_pde, PdeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::pde::{expr_in, exprs_in, first_pool, pde_err, verification_dict, wrap, wrap_list};
use crate::PyExprPool;

fn with_pool<T, F>(py: Python<'_>, pool: &Py<PyExprPool>, f: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce(&ExprPool) -> Result<T, PdeError> + Send,
{
    let p = pool.borrow(py);
    let pl = &p.inner;
    py.detach(|| f(pl)).map_err(pde_err)
}

fn generator_dict<'py>(
    py: Python<'py>,
    g: &core_pde::SymmetryGenerator,
    c: &core_pde::PointSymmetries,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("xi", wrap_list(py, &g.xi, pool)?)?;
    d.set_item("eta", wrap(py, g.eta, pool))?;
    d.set_item("verified", g.verified)?;
    let text = {
        let p = pool.borrow(py);
        core_pde::generator_display(g, &c.independent, c.dependent, &p.inner)
    };
    d.set_item("operator", text)?;
    Ok(d)
}

fn pairs<'py>(
    py: Python<'py>,
    ps: &[(ExprId, ExprId)],
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyList>> {
    let l = PyList::empty(py);
    for &(a, b) in ps {
        l.append((wrap(py, a, pool), wrap(py, b, pool)))?;
    }
    Ok(l)
}

fn symmetry_dict<'py>(
    py: Python<'py>,
    c: &core_pde::PointSymmetries,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("independent", wrap_list(py, &c.independent, pool)?)?;
    d.set_item("unknown", wrap(py, c.unknown, pool))?;
    d.set_item("dependent", wrap(py, c.dependent, pool))?;
    d.set_item("equation", wrap(py, c.equation, pool))?;
    let gens = PyList::empty(py);
    for g in &c.generators {
        gens.append(generator_dict(py, g, c, pool)?)?;
    }
    d.set_item("generators", gens)?;
    match &c.structure_constants {
        Some(sc) => {
            let l = PyList::empty(py);
            for &(a, b, k, v) in sc {
                l.append((a, b, k, wrap(py, v, pool)))?;
            }
            d.set_item("structure_constants", l)?;
        }
        None => d.set_item("structure_constants", py.None())?,
    }
    match &c.infinite {
        Some(inf) => {
            let i = PyDict::new(py);
            i.set_item("components", wrap_list(py, &inf.components, pool)?)?;
            i.set_item("functions", wrap_list(py, &inf.functions, pool)?)?;
            i.set_item("constraints", wrap_list(py, &inf.constraints, pool)?)?;
            i.set_item("kind", inf.kind)?;
            d.set_item("infinite", i)?;
        }
        None => d.set_item("infinite", py.None())?,
    }
    d.set_item("complete", c.complete)?;
    d.set_item("case", pairs(py, &c.case, pool)?)?;
    d.set_item("nonzero", wrap_list(py, &c.nonzero, pool)?)?;
    d.set_item("excluded", pairs(py, &c.excluded, pool)?)?;
    d.set_item(
        "determining_equations",
        wrap_list(py, &c.determining_equations, pool)?,
    )?;
    d.set_item("notes", c.notes.clone())?;
    Ok(d)
}

/// Private: the point symmetries of ``eq = 0``, one dict per parameter case.
#[pyfunction]
fn _pde_symmetries<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "pde_symmetries")?;
    let u = expr_in(unknown, &pool, "pde_symmetries")?;
    let cases = with_pool(py, &pool, |p| core_pde::point_symmetries(e, u, p))?;
    let out = PyList::empty(py);
    for c in &cases {
        out.append(symmetry_dict(py, c, &pool)?)?;
    }
    Ok(out)
}

/// Private: the determining equations.
#[pyfunction]
fn _pde_determining_equations(
    py: Python<'_>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "determining_equations")?;
    let u = expr_in(unknown, &pool, "determining_equations")?;
    let eqs = with_pool(py, &pool, |p| core_pde::determining_equations(e, u, p))?;
    wrap_list(py, &eqs, &pool)
}

/// Private: zero status of ``pr X(F)`` on solutions.
#[pyfunction]
fn _pde_verify_symmetry(
    py: Python<'_>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    components: &Bound<'_, PyAny>,
) -> PyResult<&'static str> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "verify_symmetry")?;
    let u = expr_in(unknown, &pool, "verify_symmetry")?;
    let cs = exprs_in(components, &pool, "verify_symmetry")?;
    let st = with_pool(py, &pool, |p| core_pde::verify_symmetry(e, u, &cs, p))?;
    Ok(st.as_str())
}

/// Private: the commutator of two vector fields.
#[pyfunction]
fn _pde_commutator(
    py: Python<'_>,
    vars: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(vars)?;
    let vs = exprs_in(vars, &pool, "commutator")?;
    let xa = exprs_in(a, &pool, "commutator")?;
    let xb = exprs_in(b, &pool, "commutator")?;
    if xa.len() != vs.len() || xb.len() != vs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "each vector field needs one component per variable",
        ));
    }
    let c = with_pool(py, &pool, |p| Ok(core_pde::commutator(&vs, &xa, &xb, p)))?;
    wrap_list(py, &c, &pool)
}

/// Private: multipliers of order ``order``, one dict per case.
#[pyfunction]
#[pyo3(signature = (eq, unknown, order=2))]
fn _pde_multipliers<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    order: u32,
) -> PyResult<Bound<'py, PyList>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "conservation_multipliers")?;
    let u = expr_in(unknown, &pool, "conservation_multipliers")?;
    let ms = with_pool(py, &pool, |p| {
        core_pde::conservation_multipliers(e, u, order, p)
    })?;
    let out = PyList::empty(py);
    for m in &ms {
        let d = PyDict::new(py);
        d.set_item("independent", wrap_list(py, &m.independent, &pool)?)?;
        d.set_item("coordinates", wrap_list(py, &m.coordinates, &pool)?)?;
        d.set_item("multipliers", wrap_list(py, &m.multipliers, &pool)?)?;
        match &m.infinite {
            Some((x, cons)) => d.set_item(
                "infinite",
                (wrap(py, *x, &pool), wrap_list(py, cons, &pool)?),
            )?,
            None => d.set_item("infinite", py.None())?,
        }
        d.set_item("complete", m.complete)?;
        d.set_item("case", pairs(py, &m.case, &pool)?)?;
        d.set_item("notes", m.notes.clone())?;
        out.append(d)?;
    }
    Ok(out)
}

fn law_dict<'py>(
    py: Python<'py>,
    l: &core_pde::ConservationLaw,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("multiplier", wrap(py, l.multiplier, pool))?;
    d.set_item("components", wrap_list(py, &l.components, pool)?)?;
    d.set_item("verified", l.verified)?;
    Ok(d)
}

/// Private: conservation laws with multipliers of order ``≤ order``.
#[pyfunction]
#[pyo3(signature = (eq, unknown, order=2))]
fn _pde_conservation_laws<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    order: u32,
) -> PyResult<Bound<'py, PyList>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "conservation_laws")?;
    let u = expr_in(unknown, &pool, "conservation_laws")?;
    let laws = with_pool(py, &pool, |p| core_pde::conservation_laws(e, u, order, p))?;
    let out = PyList::empty(py);
    for l in &laws {
        out.append(law_dict(py, l, &pool)?)?;
    }
    Ok(out)
}

/// Private: the verified density/flux of one multiplier.
#[pyfunction]
fn _pde_conserved_flux<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    multiplier: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "conserved_flux")?;
    let u = expr_in(unknown, &pool, "conserved_flux")?;
    let m = expr_in(multiplier, &pool, "conserved_flux")?;
    let l = with_pool(py, &pool, |p| core_pde::conserved_flux(e, u, m, p))?;
    law_dict(py, &l, &pool)
}

/// Private: the Euler operator.
#[pyfunction]
fn _pde_euler_operator(
    py: Python<'_>,
    lagrangian: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(unknown)?;
    let l = expr_in(lagrangian, &pool, "euler_operator")?;
    let u = expr_in(unknown, &pool, "euler_operator")?;
    let e = with_pool(py, &pool, |p| core_pde::euler_operator(l, u, p))?;
    Ok(wrap(py, e, &pool))
}

/// Private: zero status of `Σ D_i Φ^i − Λ F`.
#[pyfunction]
fn _pde_verify_conservation_law(
    py: Python<'_>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    multiplier: &Bound<'_, PyAny>,
    components: &Bound<'_, PyAny>,
) -> PyResult<&'static str> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "verify_conservation_law")?;
    let u = expr_in(unknown, &pool, "verify_conservation_law")?;
    let m = expr_in(multiplier, &pool, "verify_conservation_law")?;
    let cs = exprs_in(components, &pool, "verify_conservation_law")?;
    let st = with_pool(py, &pool, |p| {
        core_pde::verify_conservation_law(e, u, m, &cs, p)
    })?;
    Ok(st.as_str())
}

fn reduced_list<'py>(
    py: Python<'py>,
    sols: &[core_pde::ReducedSolution],
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for r in sols {
        let d = PyDict::new(py);
        d.set_item("value", wrap(py, r.value, pool))?;
        d.set_item("method", r.method)?;
        d.set_item("constants", wrap_list(py, &r.constants, pool)?)?;
        d.set_item("assumptions", r.assumptions.clone())?;
        d.set_item("steps", r.steps.clone())?;
        d.set_item(
            "verification",
            verification_dict(py, &r.verification, pool)?,
        )?;
        out.append(d)?;
    }
    Ok(out)
}

/// Private: verified reduced solutions.  ``kind`` is ``"traveling_wave"``,
/// ``"similarity"``, ``"invariant"`` (needs ``generator``) or an expansion
/// method (``"tanh"``, ``"sech"``, ``"tan"``).
#[pyfunction]
#[pyo3(signature = (eq, unknown, kind, generator=None))]
fn _pde_reduced_solutions<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    kind: &str,
    generator: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "reduced solutions")?;
    let u = expr_in(unknown, &pool, "reduced solutions")?;
    let g = match generator {
        Some(g) => exprs_in(g, &pool, "invariant_solutions")?,
        None => Vec::new(),
    };
    let kind = kind.to_string();
    let sols = with_pool(py, &pool, move |p| match kind.as_str() {
        "traveling_wave" => core_pde::traveling_wave_solutions(e, u, p),
        "similarity" => core_pde::similarity_solutions(e, u, p),
        "invariant" => core_pde::invariant_solutions(e, u, &g, p),
        m => core_pde::ansatz_solutions_for(e, u, m, p),
    })?;
    reduced_list(py, &sols, &pool)
}

/// Private: the reduction by a generator.
#[pyfunction]
fn _pde_reduce_by_generator<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    generator: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "reduce_by_generator")?;
    let u = expr_in(unknown, &pool, "reduce_by_generator")?;
    let g = exprs_in(generator, &pool, "reduce_by_generator")?;
    let r = with_pool(py, &pool, |p| core_pde::reduce_by_generator(e, u, &g, p))?;
    let d = PyDict::new(py);
    d.set_item("variable", wrap(py, r.variable, &pool))?;
    d.set_item("symbol", wrap(py, r.symbol, &pool))?;
    d.set_item("form", wrap(py, r.form, &pool))?;
    d.set_item("ode", wrap(py, r.ode, &pool))?;
    d.set_item("function", wrap(py, r.function, &pool))?;
    Ok(d)
}

/// Private: Cole–Hopf.
#[pyfunction]
fn _pde_cole_hopf<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "cole_hopf")?;
    let u = expr_in(unknown, &pool, "cole_hopf")?;
    let c = with_pool(py, &pool, |p| core_pde::cole_hopf(e, u, p))?;
    let d = PyDict::new(py);
    d.set_item("substitution", wrap(py, c.substitution, &pool))?;
    d.set_item("heat_equation", wrap(py, c.heat_equation, &pool))?;
    d.set_item("theta", wrap(py, c.theta, &pool))?;
    d.set_item("viscosity", wrap(py, c.viscosity, &pool))?;
    d.set_item("coefficient", wrap(py, c.coefficient, &pool))?;
    d.set_item("verified", c.verified)?;
    Ok(d)
}

/// Private: Miura.
#[pyfunction]
fn _pde_miura<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
    kdv_unknown: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "miura")?;
    let v = expr_in(unknown, &pool, "miura")?;
    let w = expr_in(kdv_unknown, &pool, "miura")?;
    let m = with_pool(py, &pool, |p| core_pde::miura(e, v, w, p))?;
    let d = PyDict::new(py);
    d.set_item("substitution", wrap(py, m.substitution, &pool))?;
    d.set_item("kdv", wrap(py, m.kdv, &pool))?;
    d.set_item("kdv_unknown", wrap(py, m.kdv_unknown, &pool))?;
    d.set_item("verified", m.verified)?;
    Ok(d)
}

/// Private: hodograph.
#[pyfunction]
fn _pde_hodograph<'py>(
    py: Python<'py>,
    equations: &Bound<'_, PyAny>,
    unknowns: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknowns)?;
    let es = exprs_in(equations, &pool, "hodograph")?;
    let us = exprs_in(unknowns, &pool, "hodograph")?;
    let h = with_pool(py, &pool, |p| core_pde::hodograph(&es, &us, p))?;
    let d = PyDict::new(py);
    d.set_item("equations", wrap_list(py, &h.equations, &pool)?)?;
    d.set_item("new_unknowns", wrap_list(py, &h.new_unknowns, &pool)?)?;
    d.set_item("jacobian", wrap(py, h.jacobian, &pool))?;
    d.set_item("verified", h.verified)?;
    Ok(d)
}

/// Private: Hopf–Lax.
#[pyfunction]
fn _pde_hopf_lax<'py>(
    py: Python<'py>,
    eq: &Bound<'_, PyAny>,
    unknown: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknown)?;
    let e = expr_in(eq, &pool, "hopf_lax")?;
    let u = expr_in(unknown, &pool, "hopf_lax")?;
    let h = with_pool(py, &pool, |p| core_pde::hopf_lax(e, u, p))?;
    let d = PyDict::new(py);
    d.set_item("hamiltonian", wrap(py, h.hamiltonian, &pool))?;
    d.set_item("p", wrap(py, h.p, &pool))?;
    d.set_item("lagrangian", wrap(py, h.lagrangian, &pool))?;
    d.set_item("q", wrap(py, h.q, &pool))?;
    d.set_item("objective", wrap(py, h.objective, &pool))?;
    d.set_item("y", wrap(py, h.y, &pool))?;
    d.set_item("verified", h.verified)?;
    d.set_item("convexity", wrap(py, h.convexity, &pool))?;
    Ok(d)
}

/// Register the M6 surface.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_pde_symmetries, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_determining_equations, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_verify_symmetry, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_commutator, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_multipliers, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_conservation_laws, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_conserved_flux, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_euler_operator, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_verify_conservation_law, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_reduced_solutions, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_reduce_by_generator, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_cole_hopf, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_miura, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_hodograph, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_hopf_lax, m)?)?;
    Ok(())
}
