//! PyO3 bindings for `alkahest_core::pde` — the PDE foundation layer
//! (jets, problem validation, classification, verification).
//!
//! The Python surface is `alkahest.experimental` (`JetSpace`, `Function`, `D`,
//! `PDE`, `classify_pde`, `verify_pde_solution`, the domain classes); the
//! problem objects themselves are thin Python classes in
//! `python/alkahest/experimental/_pde.py` that call the private `_pde_*`
//! functions below, so Rust and Python validate with the same code.

use std::collections::HashMap;

use alkahest_core::kernel::{Domain, ExprId};
use alkahest_core::pde::{
    classify_pde, functions_to_jets, pdsolve, pdsolve_all, verify_solution, ConditionKind,
    JetNaming, JetSpace, PdeCondition, PdeDomain, PdeError, PdeProblem, PdeSolution,
    PdsolveOptions,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::{PyExpr, PyExprPool, ToPyAny};

pyo3::create_exception!(alkahest, PyPdeError, crate::PyAlkahestError);

pub(crate) fn pde_err(e: PdeError) -> PyErr {
    Python::attach(|py| {
        let exc_type = py.get_type::<PyPdeError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

pub(crate) fn wrap(py: Python<'_>, id: ExprId, pool: &Py<PyExprPool>) -> Py<PyAny> {
    PyExpr {
        id,
        pool: pool.clone_ref(py),
    }
    .to_py_any(py)
}

pub(crate) fn wrap_list(
    py: Python<'_>,
    ids: &[ExprId],
    pool: &Py<PyExprPool>,
) -> PyResult<Py<PyAny>> {
    PyList::new(py, ids.iter().map(|&i| wrap(py, i, pool)))?.into_py_any(py)
}

/// Extract an `Expr` and check it lives in `pool` (`E-POOL-001` otherwise).
pub(crate) fn expr_in(ob: &Bound<'_, PyAny>, pool: &Py<PyExprPool>, op: &str) -> PyResult<ExprId> {
    let e: PyRef<'_, PyExpr> = ob.extract()?;
    if !e.pool.is(pool) {
        return Err(crate::cross_pool_err(format!(
            "[E-POOL-001] {op}(): arguments belong to different ExprPools"
        )));
    }
    Ok(e.id)
}

pub(crate) fn exprs_in(
    ob: &Bound<'_, PyAny>,
    pool: &Py<PyExprPool>,
    op: &str,
) -> PyResult<Vec<ExprId>> {
    let mut out = Vec::new();
    for item in ob.try_iter()? {
        out.push(expr_in(&item?, pool, op)?);
    }
    Ok(out)
}

/// The pool of the first `Expr` found in `ob` (an `Expr` or an iterable of
/// them).
pub(crate) fn first_pool(ob: &Bound<'_, PyAny>) -> PyResult<Py<PyExprPool>> {
    if let Ok(e) = ob.extract::<PyRef<'_, PyExpr>>() {
        return Ok(e.pool.clone_ref(ob.py()));
    }
    // A string iterates to one-character strings forever: never descend.
    if ob.is_instance_of::<pyo3::types::PyString>() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "expected an Expr or a list of Exprs",
        ));
    }
    for item in ob.try_iter()? {
        if let Ok(p) = first_pool(&item?) {
            return Ok(p);
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected an Expr or a list of Exprs",
    ))
}

fn parse_naming(s: &str) -> PyResult<JetNaming> {
    Ok(match s {
        "subscript" => JetNaming::Subscript,
        "prime" => JetNaming::Prime,
        "dot" => JetNaming::Dot,
        "dae" | "dae_dt" => JetNaming::DaeDt,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "naming must be 'subscript', 'prime', 'dot' or 'dae', not {other:?}"
            )))
        }
    })
}

fn naming_str(n: JetNaming) -> &'static str {
    match n {
        JetNaming::Subscript => "subscript",
        JetNaming::Prime => "prime",
        JetNaming::Dot => "dot",
        JetNaming::DaeDt => "dae",
        _ => "other",
    }
}

/// `(unknown, alpha, symbol)` as `JetSpace.jets_in` returns it.
type JetTriple = (Py<PyAny>, Vec<u32>, Py<PyAny>);

/// A multi-index jet space `J^k`: independent variables, unknowns and a
/// maximum order, with the bijection ``Derivative(u(x, t), …) ↔ u_xt``.
///
/// ``JetSpace(independent, unknowns, max_order, naming="subscript",
/// domain=None)``; ``independent`` may be ``None`` to read the variables off
/// the unknowns.  See :mod:`alkahest.experimental` for the full description.
#[pyclass(name = "JetSpace", module = "alkahest")]
pub(crate) struct PyJetSpace {
    inner: JetSpace,
    pool: Py<PyExprPool>,
}

impl PyJetSpace {
    fn var_index(&self, py: Python<'_>, var: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = var.extract::<usize>() {
            return Ok(i);
        }
        let id = expr_in(var, &self.pool, "JetSpace")?;
        self.inner
            .independent()
            .iter()
            .position(|&x| x == id)
            .ok_or_else(|| {
                let pool = self.pool.borrow(py);
                pde_err(PdeError::InconsistentVariables(format!(
                    "{} is not an independent variable of the jet space",
                    pool.inner.display(id)
                )))
            })
    }

    fn dep_index(&self, py: Python<'_>, dep: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = dep.extract::<usize>() {
            return Ok(i);
        }
        let id = expr_in(dep, &self.pool, "JetSpace")?;
        self.inner
            .unknowns()
            .iter()
            .position(|&u| u == id)
            .ok_or_else(|| {
                let pool = self.pool.borrow(py);
                pde_err(PdeError::NotAnUnknown(format!(
                    "{} is not an unknown of the jet space",
                    pool.inner.display(id)
                )))
            })
    }

    fn with(&self, py: Python<'_>, inner: JetSpace) -> Self {
        PyJetSpace {
            inner,
            pool: self.pool.clone_ref(py),
        }
    }
}

#[pymethods]
impl PyJetSpace {
    #[new]
    #[pyo3(signature = (independent, unknowns, max_order, naming="subscript", domain=None))]
    fn new(
        py: Python<'_>,
        independent: Option<&Bound<'_, PyAny>>,
        unknowns: &Bound<'_, PyAny>,
        max_order: u32,
        naming: &str,
        domain: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let pool = first_pool(unknowns)?;
        let us = exprs_in(unknowns, &pool, "JetSpace")?;
        let dom = match domain.filter(|o| !o.is_none()) {
            Some(d) => crate::parse_domain_arg(Some(d))?,
            None => Domain::Real,
        };
        let inner = {
            let p = pool.borrow(py);
            let base = match independent.filter(|o| !o.is_none()) {
                Some(ind) => {
                    let xs = exprs_in(ind, &pool, "JetSpace")?;
                    JetSpace::new(&xs, &us, max_order, &p.inner)
                }
                None => JetSpace::from_unknowns(&us, max_order, &p.inner),
            }
            .map_err(pde_err)?;
            base.with_naming(parse_naming(naming)?)
                .map_err(pde_err)?
                .with_domain(dom)
        };
        Ok(PyJetSpace { inner, pool })
    }

    /// The independent variables.
    #[getter]
    fn independent(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        wrap_list(py, self.inner.independent(), &self.pool)
    }

    /// The unknowns, as applied functions.
    #[getter]
    fn unknowns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        wrap_list(py, &self.inner.unknowns(), &self.pool)
    }

    /// The maximum order.
    #[getter]
    fn max_order(&self) -> u32 {
        self.inner.max_order()
    }

    /// ``"subscript"``, ``"prime"``, ``"dot"`` or ``"dae"``.
    #[getter]
    fn naming(&self) -> &'static str {
        naming_str(self.inner.naming())
    }

    /// The same space with maximum order raised by ``k``.
    fn prolonged(&self, py: Python<'_>, k: u32) -> Self {
        self.with(py, self.inner.prolonged(k))
    }

    /// The same space with jets named by ``naming``.
    fn with_naming(&self, py: Python<'_>, naming: &str) -> PyResult<Self> {
        let inner = self
            .inner
            .clone()
            .with_naming(parse_naming(naming)?)
            .map_err(pde_err)?;
        Ok(self.with(py, inner))
    }

    /// The jet symbol ``u_α``: ``dep`` is an unknown (or its index), ``alpha``
    /// one order per independent variable.
    fn jet(&self, py: Python<'_>, dep: &Bound<'_, PyAny>, alpha: Vec<u32>) -> PyResult<Py<PyAny>> {
        let d = self.dep_index(py, dep)?;
        let id = {
            let p = self.pool.borrow(py);
            self.inner.jet(d, &alpha, &p.inner).map_err(pde_err)?
        };
        Ok(wrap(py, id, &self.pool))
    }

    /// The ``Derivative(u(x, …), …)`` node the jet ``u_α`` stands for.
    fn derivative(
        &self,
        py: Python<'_>,
        dep: &Bound<'_, PyAny>,
        alpha: Vec<u32>,
    ) -> PyResult<Py<PyAny>> {
        let d = self.dep_index(py, dep)?;
        let id = {
            let p = self.pool.borrow(py);
            self.inner
                .derivative(d, &alpha, &p.inner)
                .map_err(pde_err)?
        };
        Ok(wrap(py, id, &self.pool))
    }

    /// ``(unknown, alpha)`` when ``symbol`` is a jet coordinate of this
    /// space, else ``None``.
    fn decode(
        &self,
        py: Python<'_>,
        symbol: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(Py<PyAny>, Vec<u32>)>> {
        let id = expr_in(symbol, &self.pool, "JetSpace.decode")?;
        let p = self.pool.borrow(py);
        Ok(self
            .inner
            .decode_jet(id, &p.inner)
            .map(|(d, a)| (wrap(py, self.inner.unknowns()[d], &self.pool), a)))
    }

    /// Replace every unknown and derivative in ``expr`` by its jet symbol.
    fn to_jets(&self, py: Python<'_>, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let id = expr_in(expr, &self.pool, "JetSpace.to_jets")?;
        let out = {
            let p = self.pool.borrow(py);
            self.inner.to_jets(id, &p.inner).map_err(pde_err)?
        };
        Ok(wrap(py, out, &self.pool))
    }

    /// Replace every jet symbol in ``expr`` by its derivative (inverse of
    /// :meth:`to_jets`).
    // The Python name mirrors `JetSpace::from_jets`; it is not a constructor.
    #[allow(clippy::wrong_self_convention)]
    fn from_jets(&self, py: Python<'_>, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let id = expr_in(expr, &self.pool, "JetSpace.from_jets")?;
        let out = {
            let p = self.pool.borrow(py);
            self.inner.from_jets(id, &p.inner)
        };
        Ok(wrap(py, out, &self.pool))
    }

    /// Every jet coordinate up to :attr:`max_order`.
    fn coordinates(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ids = {
            let p = self.pool.borrow(py);
            self.inner.coordinates(&p.inner)
        };
        wrap_list(py, &ids, &self.pool)
    }

    /// The jet coordinates occurring in ``expr`` as ``(unknown, alpha,
    /// symbol)`` triples, highest-ranked first.
    fn jets_in(&self, py: Python<'_>, expr: &Bound<'_, PyAny>) -> PyResult<Vec<JetTriple>> {
        let id = expr_in(expr, &self.pool, "JetSpace.jets_in")?;
        let cs = {
            let p = self.pool.borrow(py);
            self.inner.jets_in(id, &p.inner).map_err(pde_err)?
        };
        let us = self.inner.unknowns();
        Ok(cs
            .into_iter()
            .map(|c| {
                (
                    wrap(py, us[c.dep], &self.pool),
                    c.alpha,
                    wrap(py, c.symbol, &self.pool),
                )
            })
            .collect())
    }

    /// The highest jet order in ``expr`` (``None`` when it has none).
    fn order_of(&self, py: Python<'_>, expr: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
        let id = expr_in(expr, &self.pool, "JetSpace.order_of")?;
        let p = self.pool.borrow(py);
        self.inner.order_of(id, &p.inner).map_err(pde_err)
    }

    /// The total derivative ``D_var expr`` (``var`` an independent variable
    /// or its index), in jets.
    fn total_derivative(
        &self,
        py: Python<'_>,
        expr: &Bound<'_, PyAny>,
        var: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let id = expr_in(expr, &self.pool, "JetSpace.total_derivative")?;
        let i = self.var_index(py, var)?;
        let out = {
            let p = self.pool.borrow(py);
            self.inner
                .total_derivative(id, i, &p.inner)
                .map_err(pde_err)?
        };
        Ok(wrap(py, out, &self.pool))
    }

    /// The iterated total derivative ``D^alpha expr``.
    fn total_derivative_multi(
        &self,
        py: Python<'_>,
        expr: &Bound<'_, PyAny>,
        alpha: Vec<u32>,
    ) -> PyResult<Py<PyAny>> {
        let id = expr_in(expr, &self.pool, "JetSpace.total_derivative_multi")?;
        let out = {
            let p = self.pool.borrow(py);
            self.inner
                .total_derivative_multi(id, &alpha, &p.inner)
                .map_err(pde_err)?
        };
        Ok(wrap(py, out, &self.pool))
    }

    /// The ``k``-th prolongation of ``equations``: every ``D^β F`` with
    /// ``|β| ≤ k``.
    fn prolong(&self, py: Python<'_>, equations: &Bound<'_, PyAny>, k: u32) -> PyResult<Py<PyAny>> {
        let ids = exprs_in(equations, &self.pool, "JetSpace.prolong")?;
        let out = {
            let p = self.pool.borrow(py);
            self.inner.prolong(&ids, k, &p.inner).map_err(pde_err)?
        };
        wrap_list(py, &out, &self.pool)
    }

    /// The system as polynomials in the jets: a dict with ``equations``
    /// (numerators, ``= 0``), ``nonzero`` (cleared denominators),
    /// ``variables`` (jets, orderly ranking), ``independent`` and
    /// ``parameters``.
    fn to_rational_system<'py>(
        &self,
        py: Python<'py>,
        equations: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ids = exprs_in(equations, &self.pool, "JetSpace.to_rational_system")?;
        let sys = {
            let p = self.pool.borrow(py);
            self.inner
                .to_rational_system(&ids, &p.inner)
                .map_err(pde_err)?
        };
        let d = PyDict::new(py);
        d.set_item("equations", wrap_list(py, &sys.equations, &self.pool)?)?;
        d.set_item("nonzero", wrap_list(py, &sys.nonzero, &self.pool)?)?;
        d.set_item("variables", wrap_list(py, &sys.variables, &self.pool)?)?;
        d.set_item("independent", wrap_list(py, &sys.independent, &self.pool)?)?;
        d.set_item("parameters", wrap_list(py, &sys.parameters, &self.pool)?)?;
        Ok(d)
    }

    /// ``dsolve``'s input convention for an ODE ``equation = 0``: a dict with
    /// ``x``, ``y``, ``derivs`` (``[y', y'', …]``) and ``equation`` in them.
    fn to_ode_input<'py>(
        &self,
        py: Python<'py>,
        equation: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let id = expr_in(equation, &self.pool, "JetSpace.to_ode_input")?;
        let inp = {
            let p = self.pool.borrow(py);
            self.inner.to_ode_input(id, &p.inner).map_err(pde_err)?
        };
        let d = PyDict::new(py);
        d.set_item("x", wrap(py, inp.x, &self.pool))?;
        d.set_item("y", wrap(py, inp.y, &self.pool))?;
        d.set_item("derivs", wrap_list(py, &inp.derivs, &self.pool)?)?;
        d.set_item("equation", wrap(py, inp.equation, &self.pool))?;
        Ok(d)
    }

    /// The DAE convention: ``[(y, dy/dt), (dy/dt, ddy/dt/dt), …]``.
    fn dae_pairs(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        let pairs = {
            let p = self.pool.borrow(py);
            self.inner.dae_pairs(&p.inner).map_err(pde_err)?
        };
        Ok(pairs
            .into_iter()
            .map(|(a, b)| (wrap(py, a, &self.pool), wrap(py, b, &self.pool)))
            .collect())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let p = self.pool.borrow(py);
        let show = |ids: &[ExprId]| {
            ids.iter()
                .map(|&i| p.inner.display(i).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "JetSpace(independent=[{}], unknowns=[{}], max_order={})",
            show(self.inner.independent()),
            show(&self.inner.unknowns()),
            self.inner.max_order()
        )
    }
}

/// ``functions_to_jets(exprs, funcs, domain=None)`` — the multi-variable
/// :func:`functions_to_symbols`: returns ``(new_exprs, mapping)`` with
/// ``mapping = {node: symbol}`` and jet names ``u``, ``u_x``, ``u_xt``.
#[pyfunction]
#[pyo3(name = "functions_to_jets", signature = (exprs, funcs, domain=None))]
fn py_functions_to_jets(
    py: Python<'_>,
    exprs: &Bound<'_, PyAny>,
    funcs: &Bound<'_, PyAny>,
    domain: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let single = exprs.extract::<PyRef<'_, PyExpr>>().is_ok();
    let pool = first_pool(funcs)?;
    let ids = if single {
        vec![expr_in(exprs, &pool, "functions_to_jets")?]
    } else {
        exprs_in(exprs, &pool, "functions_to_jets")?
    };
    let fids = exprs_in(funcs, &pool, "functions_to_jets")?;
    let dom = match domain.filter(|o| !o.is_none()) {
        Some(d) => crate::parse_domain_arg(Some(d))?,
        None => Domain::Real,
    };
    let (out, table) = {
        let p = pool.borrow(py);
        functions_to_jets(&ids, &fids, dom, &p.inner).map_err(pde_err)?
    };
    let mapping = PyDict::new(py);
    for e in table {
        mapping.set_item(wrap(py, e.node, &pool), wrap(py, e.symbol, &pool))?;
    }
    let out_obj = if single {
        wrap(py, out[0], &pool)
    } else {
        wrap_list(py, &out, &pool)?
    };
    Ok((out_obj, mapping.into_py_any(py)?))
}

/// Private: ``classify_pde`` as a dict (``alkahest.experimental`` wraps it in
/// ``PdeInfo``).
#[pyfunction]
fn _pde_classify<'py>(
    py: Python<'py>,
    equation: PyRef<'_, PyExpr>,
    unknown: PyRef<'_, PyExpr>,
) -> PyResult<Bound<'py, PyDict>> {
    if !equation.pool.is(&unknown.pool) {
        return Err(crate::cross_pool_err(
            "[E-POOL-001] classify_pde(): arguments belong to different ExprPools".into(),
        ));
    }
    let pool = equation.pool.clone_ref(py);
    let info = {
        let p = pool.borrow(py);
        crate::guard_depth(&p.inner, equation.id)?;
        classify_pde(equation.id, unknown.id, &p.inner).map_err(pde_err)?
    };
    let d = PyDict::new(py);
    d.set_item("unknown", wrap(py, info.unknown, &pool))?;
    d.set_item("independent", wrap_list(py, &info.independent, &pool)?)?;
    d.set_item("equation_in_jets", wrap(py, info.equation_in_jets, &pool))?;
    d.set_item("order", info.order)?;
    d.set_item("linearity", info.linearity.as_str())?;
    d.set_item("homogeneous", info.homogeneous)?;
    d.set_item("constant_coefficients", info.constant_coefficients)?;
    d.set_item("type", info.pde_type.map(|t| t.as_str()))?;
    d.set_item(
        "type_conditions",
        wrap_list(py, &info.type_conditions, &pool)?,
    )?;
    match &info.principal_symbol {
        Some(rows) => {
            let l = PyList::empty(py);
            for r in rows {
                l.append(wrap_list(py, r, &pool)?)?;
            }
            d.set_item("principal_symbol", l)?;
        }
        None => d.set_item("principal_symbol", py.None())?,
    }
    d.set_item(
        "discriminant",
        info.discriminant.map(|e| wrap(py, e, &pool)),
    )?;
    let regions = PyList::empty(py);
    for r in &info.regions {
        let rd = PyDict::new(py);
        rd.set_item("type", r.pde_type.as_str())?;
        rd.set_item("condition", wrap(py, r.condition, &pool))?;
        rd.set_item("var", r.var.map(|e| wrap(py, e, &pool)))?;
        rd.set_item("lower", r.lower.map(|e| wrap(py, e, &pool)))?;
        rd.set_item("upper", r.upper.map(|e| wrap(py, e, &pool)))?;
        regions.append(rd)?;
    }
    d.set_item("regions", regions)?;
    let named = PyList::empty(py);
    for m in &info.named {
        let md = PyDict::new(py);
        md.set_item("name", &m.name)?;
        let params = PyDict::new(py);
        for (k, v) in &m.parameters {
            params.set_item(k, wrap(py, *v, &pool))?;
        }
        md.set_item("parameters", params)?;
        md.set_item("conditions", wrap_list(py, &m.conditions, &pool)?)?;
        md.set_item("time", m.time.map(|e| wrap(py, e, &pool)))?;
        named.append(md)?;
    }
    d.set_item("named", named)?;
    Ok(d)
}

/// Decode a domain spec tuple built by `_pde.py`.
pub(crate) fn domain_from_spec(
    spec: &Bound<'_, PyAny>,
    pool: &Py<PyExprPool>,
) -> PyResult<PdeDomain> {
    let py = spec.py();
    let t: Bound<'_, PyTuple> = spec.extract()?;
    let kind: String = t.get_item(0)?.extract()?;
    let e = |i: usize| -> PyResult<ExprId> { expr_in(&t.get_item(i)?, pool, "PDE.on") };
    let p = pool.borrow(py);
    let pl = &p.inner;
    let r = match kind.as_str() {
        "interval" => PdeDomain::interval(e(1)?, e(2)?, e(3)?, pl),
        "half_line" => PdeDomain::half_line(e(1)?, e(2)?, t.get_item(3)?.extract()?, pl),
        "line" => PdeDomain::line(e(1)?, pl),
        "box" => {
            let vars = exprs_in(&t.get_item(1)?, pool, "PDE.on")?;
            let mut bounds = Vec::new();
            for b in t.get_item(2)?.try_iter()? {
                let b = b?;
                let lo = expr_in(&b.get_item(0)?, pool, "PDE.on")?;
                let hi = expr_in(&b.get_item(1)?, pool, "PDE.on")?;
                bounds.push((lo, hi));
            }
            PdeDomain::box_(&vars, &bounds, pl)
        }
        "disk" => {
            let vars = exprs_in(&t.get_item(1)?, pool, "PDE.on")?;
            let center = exprs_in(&t.get_item(2)?, pool, "PDE.on")?;
            if vars.len() != 2 || center.len() != 2 {
                return Err(pde_err(PdeError::InvalidDomain(
                    "a disk needs two coordinates and a two-component centre".into(),
                )));
            }
            PdeDomain::disk([vars[0], vars[1]], [center[0], center[1]], e(3)?, pl)
        }
        "ball" => {
            let vars = exprs_in(&t.get_item(1)?, pool, "PDE.on")?;
            let center = exprs_in(&t.get_item(2)?, pool, "PDE.on")?;
            PdeDomain::ball(&vars, &center, e(3)?, pl)
        }
        "curvilinear" => {
            let system: String = t.get_item(1)?.extract()?;
            let sys =
                alkahest_core::pde::CoordinateSystem::from_name(&system).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown coordinate system {system:?}"
                    ))
                })?;
            let mut coords = Vec::new();
            for c in t.get_item(2)?.try_iter()? {
                let c = c?;
                coords.push(if c.is_none() {
                    None
                } else {
                    Some(expr_in(&c, pool, "PDE.on")?)
                });
            }
            let mut bounds = Vec::new();
            for b in t.get_item(3)?.try_iter()? {
                let b = b?;
                let lo = expr_in(&b.get_item(0)?, pool, "PDE.on")?;
                let hi = expr_in(&b.get_item(1)?, pool, "PDE.on")?;
                bounds.push((lo, hi));
            }
            PdeDomain::curvilinear(sys, &coords, &bounds, pl)
        }
        "product" => {
            drop(p);
            let mut parts = Vec::new();
            for s in t.get_item(1)?.try_iter()? {
                parts.push(domain_from_spec(&s?, pool)?);
            }
            let p = pool.borrow(py);
            return PdeDomain::product(parts, &p.inner).map_err(pde_err);
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown domain kind {other:?}"
            )))
        }
    };
    r.map_err(pde_err)
}

/// Private: the predicate of a domain spec (`(0 < x) ∧ (x < L)`).
#[pyfunction]
fn _pde_domain_condition(py: Python<'_>, spec: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let pool = first_pool(&spec.extract::<Bound<'_, PyTuple>>()?.get_item(1)?)?;
    let d = domain_from_spec(spec, &pool)?;
    let id = {
        let p = pool.borrow(py);
        d.condition(&p.inner)
    };
    Ok(wrap(py, id, &pool))
}

pub(crate) fn conditions_from(
    py: Python<'_>,
    conds: &Bound<'_, PyAny>,
    pool: &Py<PyExprPool>,
) -> PyResult<Vec<PdeCondition>> {
    let _ = py;
    let mut out = Vec::new();
    for c in conds.try_iter()? {
        let c = c?;
        let kind: String = c.get_item(0)?.extract()?;
        let lhs = expr_in(&c.get_item(1)?, pool, "PDE")?;
        let rhs = expr_in(&c.get_item(2)?, pool, "PDE")?;
        out.push(PdeCondition {
            kind: if kind == "ic" {
                ConditionKind::Initial
            } else {
                ConditionKind::Boundary
            },
            lhs,
            rhs,
        });
    }
    Ok(out)
}

/// Private: validate a problem (equations, unknowns, optional domain spec,
/// conditions) with `PdeProblem`; returns ``{"order", "independent"}``.
#[pyfunction]
#[pyo3(signature = (equations, unknowns, domain=None, conditions=None))]
fn _pde_check_problem<'py>(
    py: Python<'py>,
    equations: &Bound<'_, PyAny>,
    unknowns: &Bound<'_, PyAny>,
    domain: Option<&Bound<'_, PyAny>>,
    conditions: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknowns)?;
    let eqs = exprs_in(equations, &pool, "PDE")?;
    let us = exprs_in(unknowns, &pool, "PDE")?;
    let dom = match domain.filter(|o| !o.is_none()) {
        Some(s) => Some(domain_from_spec(s, &pool)?),
        None => None,
    };
    let conds = match conditions.filter(|o| !o.is_none()) {
        Some(c) => conditions_from(py, c, &pool)?,
        None => Vec::new(),
    };
    let problem = {
        let p = pool.borrow(py);
        let mut prob = PdeProblem::new(&eqs, &us, &p.inner).map_err(pde_err)?;
        if let Some(d) = dom {
            prob = prob.on(d, &p.inner).map_err(pde_err)?;
        }
        for c in conds {
            prob = match c.kind {
                ConditionKind::Initial => prob.ic(c.lhs, c.rhs, &p.inner),
                _ => prob.bc(c.lhs, c.rhs, &p.inner),
            }
            .map_err(pde_err)?;
        }
        prob
    };
    let d = PyDict::new(py);
    d.set_item("order", problem.order())?;
    d.set_item("independent", wrap_list(py, problem.independent(), &pool)?)?;
    Ok(d)
}

/// Private: `verify_solution` as a dict mirroring `integrate`'s
/// `verification`.
#[pyfunction]
#[pyo3(signature = (equations, unknowns, solution, conditions=None))]
fn _pde_verify<'py>(
    py: Python<'py>,
    equations: &Bound<'_, PyAny>,
    unknowns: &Bound<'_, PyAny>,
    solution: &Bound<'_, PyAny>,
    conditions: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = first_pool(unknowns)?;
    let eqs = exprs_in(equations, &pool, "verify_pde_solution")?;
    let us = exprs_in(unknowns, &pool, "verify_pde_solution")?;
    let sol = exprs_in(solution, &pool, "verify_pde_solution")?;
    let conds = match conditions.filter(|o| !o.is_none()) {
        Some(c) => conditions_from(py, c, &pool)?,
        None => Vec::new(),
    };
    let v = {
        let p = pool.borrow(py);
        let pl = &p.inner;
        py.detach(|| verify_solution(&eqs, &us, &sol, &conds, pl))
            .map_err(pde_err)?
    };
    verification_dict(py, &v, &pool)
}

pub(crate) fn verification_dict<'py>(
    py: Python<'py>,
    v: &alkahest_core::pde::PdeVerification,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let pool = pool.clone_ref(py);
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
        cd.set_item("residual", wrap(py, c.residual, &pool))?;
        cd.set_item("status", c.status.as_str())?;
        cd.set_item("method", c.method)?;
        cd.set_item("evidence", c.evidence.clone())?;
        by_kind[c.kind].append(cd)?;
    }
    for (k, l) in by_kind {
        d.set_item(k, l)?;
    }
    Ok(d)
}

// ---------------------------------------------------------------------------
// Reserved heads: Sum / Indexed / KroneckerDelta / RootOfFamily
// ---------------------------------------------------------------------------

fn one_expr<F>(py: Python<'_>, pool: &Py<PyExprPool>, f: F) -> PyResult<Py<PyAny>>
where
    F: FnOnce(&alkahest_core::kernel::ExprPool) -> Result<ExprId, PdeError>,
{
    let id = {
        let p = pool.borrow(py);
        f(&p.inner).map_err(pde_err)?
    };
    Ok(wrap(py, id, pool))
}

/// Private: `Sum(body, k, lo, hi)`.
#[pyfunction]
fn _pde_sum(
    py: Python<'_>,
    body: &Bound<'_, PyAny>,
    k: &Bound<'_, PyAny>,
    lo: &Bound<'_, PyAny>,
    hi: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(k)?;
    let (b, kk, l, h) = (
        expr_in(body, &pool, "Sum")?,
        expr_in(k, &pool, "Sum")?,
        expr_in(lo, &pool, "Sum")?,
        expr_in(hi, &pool, "Sum")?,
    );
    one_expr(py, &pool, |p| alkahest_core::pde::sum(b, kk, l, h, p))
}

/// Private: `Integral(body, t, lo, hi)` (definite) or `Integral(body, x)`
/// (indefinite) when `lo` / `hi` are omitted.
#[pyfunction]
#[pyo3(signature = (body, t, lo=None, hi=None))]
fn _pde_integral(
    py: Python<'_>,
    body: &Bound<'_, PyAny>,
    t: &Bound<'_, PyAny>,
    lo: Option<&Bound<'_, PyAny>>,
    hi: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(t)?;
    let b = expr_in(body, &pool, "Integral")?;
    let tt = expr_in(t, &pool, "Integral")?;
    match (lo, hi) {
        (Some(l), Some(h)) => {
            let (l, h) = (
                expr_in(l, &pool, "Integral")?,
                expr_in(h, &pool, "Integral")?,
            );
            one_expr(py, &pool, |p| alkahest_core::pde::integral(b, tt, l, h, p))
        }
        (None, None) => one_expr(py, &pool, |p| {
            alkahest_core::pde::indefinite_integral(b, tt, p)
        }),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "Integral takes (f, x) or (f, t, lo, hi)",
        )),
    }
}

/// Private: `Indexed(base, index)`.
#[pyfunction]
fn _pde_indexed(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(base)?;
    let (b, i) = (
        expr_in(base, &pool, "Indexed")?,
        expr_in(index, &pool, "Indexed")?,
    );
    one_expr(py, &pool, |p| alkahest_core::pde::indexed(b, i, p))
}

/// Private: `KroneckerDelta(i, j)` (folded when decided).
#[pyfunction]
fn _pde_kronecker(
    py: Python<'_>,
    i: &Bound<'_, PyAny>,
    j: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(i)?;
    let (a, b) = (
        expr_in(i, &pool, "KroneckerDelta")?,
        expr_in(j, &pool, "KroneckerDelta")?,
    );
    one_expr(py, &pool, |p| {
        Ok(alkahest_core::pde::kronecker_delta(a, b, p))
    })
}

/// Private: `RootOfFamily(F, lam, n, branch)`.
#[pyfunction]
fn _pde_root_of_family(
    py: Python<'_>,
    f: &Bound<'_, PyAny>,
    lam: &Bound<'_, PyAny>,
    n: &Bound<'_, PyAny>,
    branch: i64,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(lam)?;
    let (ff, l, nn) = (
        expr_in(f, &pool, "RootOfFamily")?,
        expr_in(lam, &pool, "RootOfFamily")?,
        expr_in(n, &pool, "RootOfFamily")?,
    );
    one_expr(py, &pool, |p| {
        alkahest_core::pde::root_of_family(ff, l, nn, branch, p)
    })
}

/// Private: the member with index `k` of a `RootOfFamily`.
#[pyfunction]
fn _pde_root_family_member(py: Python<'_>, fam: &Bound<'_, PyAny>, k: i64) -> PyResult<Py<PyAny>> {
    let pool = first_pool(fam)?;
    let e = expr_in(fam, &pool, "RootOfFamily.member")?;
    one_expr(py, &pool, |p| alkahest_core::pde::member(e, k, p))
}

/// Private: certified enclosure `(lo, hi)` of a `RootOfFamily` member as
/// decimal strings rounded outward, `bits` bits wide.
#[pyfunction]
#[pyo3(signature = (root, bits=64))]
fn _pde_root_family_enclosure(
    py: Python<'_>,
    root: &Bound<'_, PyAny>,
    bits: u32,
) -> PyResult<(String, String)> {
    let pool = first_pool(root)?;
    let e = expr_in(root, &pool, "RootOfFamily.enclosure")?;
    let p = pool.borrow(py);
    let (lo, hi) = alkahest_core::pde::root_family_enclosure(e, bits, &p.inner).map_err(pde_err)?;
    let digits = (f64::from(bits) * std::f64::consts::LOG10_2).ceil() as usize + 3;
    Ok((
        lo.to_string_radix_round(10, Some(digits), rug::float::Round::Down),
        hi.to_string_radix_round(10, Some(digits), rug::float::Round::Up),
    ))
}

/// Private: expand one `Sum` with integer bounds.
#[pyfunction]
fn _pde_expand_sum(py: Python<'_>, e: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let pool = first_pool(e)?;
    let id = expr_in(e, &pool, "expand_sum")?;
    one_expr(py, &pool, |p| alkahest_core::pde::expand_sum(id, p))
}

/// Private: expand every finite `Sum` in `e`.
#[pyfunction]
fn _pde_expand_finite_sums(py: Python<'_>, e: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let pool = first_pool(e)?;
    let id = expr_in(e, &pool, "expand_finite_sums")?;
    one_expr(py, &pool, |p| alkahest_core::pde::expand_finite_sums(id, p))
}

/// Private: partial sums up to index `n`.
#[pyfunction]
fn _pde_truncate(py: Python<'_>, e: &Bound<'_, PyAny>, n: i64) -> PyResult<Py<PyAny>> {
    let pool = first_pool(e)?;
    let id = expr_in(e, &pool, "truncate")?;
    one_expr(py, &pool, |p| alkahest_core::pde::truncate(id, n, p))
}

/// Private: the term with index `n` of a `Sum`.
#[pyfunction]
fn _pde_coefficient(
    py: Python<'_>,
    e: &Bound<'_, PyAny>,
    n: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(e)?;
    let id = expr_in(e, &pool, "coefficient")?;
    let nn = expr_in(n, &pool, "coefficient")?;
    one_expr(py, &pool, |p| alkahest_core::pde::coefficient(id, nn, p))
}

/// Private: fold sums, with `(index, lo, hi)` range assumptions.
#[pyfunction]
#[pyo3(signature = (e, ranges=None))]
fn _pde_simplify_sums(
    py: Python<'_>,
    e: &Bound<'_, PyAny>,
    ranges: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(e)?;
    let id = expr_in(e, &pool, "simplify_sums")?;
    let mut rs = Vec::new();
    if let Some(r) = ranges.filter(|o| !o.is_none()) {
        for item in r.try_iter()? {
            let item = item?;
            rs.push((
                expr_in(&item.get_item(0)?, &pool, "simplify_sums")?,
                expr_in(&item.get_item(1)?, &pool, "simplify_sums")?,
                expr_in(&item.get_item(2)?, &pool, "simplify_sums")?,
            ));
        }
    }
    one_expr(py, &pool, |p| {
        Ok(alkahest_core::pde::simplify_sums(id, &rs, p))
    })
}

/// Private: `Σⱼ ∂²u/∂xⱼ²` over `x_lo … x_hi`.
#[pyfunction]
fn _pde_laplacian_indexed(
    py: Python<'_>,
    u: &Bound<'_, PyAny>,
    base: &Bound<'_, PyAny>,
    lo: &Bound<'_, PyAny>,
    hi: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = first_pool(base)?;
    let (uu, b, l, h) = (
        expr_in(u, &pool, "laplacian_indexed")?,
        expr_in(base, &pool, "laplacian_indexed")?,
        expr_in(lo, &pool, "laplacian_indexed")?,
        expr_in(hi, &pool, "laplacian_indexed")?,
    );
    one_expr(py, &pool, |p| {
        alkahest_core::pde::laplacian_indexed(uu, b, l, h, p)
    })
}

// ---------------------------------------------------------------------------
// pdsolve
// ---------------------------------------------------------------------------

fn side_condition_tuple(
    py: Python<'_>,
    c: &alkahest_core::SideCondition,
    pool: &Py<PyExprPool>,
) -> PyResult<Py<PyAny>> {
    use alkahest_core::SideCondition as S;
    let rendered = {
        let p = pool.borrow(py);
        c.display_with(&p.inner).to_string()
    };
    let (kind, e) = match c {
        S::NonZero(e) => ("nonzero", *e),
        S::Positive(e) => ("positive", *e),
        S::InDomain(e, _) => ("in_domain", *e),
    };
    PyTuple::new(
        py,
        [
            kind.into_py_any(py)?,
            wrap(py, e, pool),
            rendered.into_py_any(py)?,
        ],
    )?
    .into_py_any(py)
}

fn solution_dict<'py>(
    py: Python<'py>,
    s: &PdeSolution,
    pool: &Py<PyExprPool>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("unknown", wrap(py, s.unknown, pool))?;
    d.set_item("value", wrap(py, s.value, pool))?;
    d.set_item("implicit", s.implicit)?;
    d.set_item("kind", s.kind.as_str())?;
    d.set_item("method", s.method)?;
    d.set_item("arbitrary_functions", s.arbitrary_functions.clone())?;
    d.set_item("constants", wrap_list(py, &s.constants, pool)?)?;
    let sc = PyList::empty(py);
    for c in &s.side_conditions {
        sc.append(side_condition_tuple(py, c, pool)?)?;
    }
    d.set_item("side_conditions", sc)?;
    d.set_item(
        "verification",
        verification_dict(py, &s.verification, pool)?,
    )?;
    d.set_item("steps", s.steps.clone())?;
    let alts = PyList::empty(py);
    for a in &s.alternatives {
        alts.append(solution_dict(py, a, pool)?)?;
    }
    d.set_item("alternatives", alts)?;
    Ok(d)
}

/// Private: `pdsolve` / `pdsolve_all` on a validated problem; a list of
/// solution dicts.
#[pyfunction]
#[pyo3(signature = (equations, unknowns, conditions=None, hint=None, prefix="F", all=false))]
fn _pde_solve<'py>(
    py: Python<'py>,
    equations: &Bound<'_, PyAny>,
    unknowns: &Bound<'_, PyAny>,
    conditions: Option<&Bound<'_, PyAny>>,
    hint: Option<String>,
    prefix: &str,
    all: bool,
) -> PyResult<Bound<'py, PyList>> {
    let pool = first_pool(unknowns)?;
    let eqs = exprs_in(equations, &pool, "pdsolve")?;
    let us = exprs_in(unknowns, &pool, "pdsolve")?;
    let conds = match conditions.filter(|o| !o.is_none()) {
        Some(c) => conditions_from(py, c, &pool)?,
        None => Vec::new(),
    };
    let mut opts = PdsolveOptions::default().with_function_prefix(prefix);
    if let Some(h) = hint {
        opts = opts.with_hint(h);
    }
    let sols = {
        let p = pool.borrow(py);
        let pl = &p.inner;
        py.detach(|| -> Result<Vec<PdeSolution>, PdeError> {
            let mut prob = PdeProblem::new(&eqs, &us, pl)?;
            for c in conds {
                prob = match c.kind {
                    ConditionKind::Initial => prob.ic(c.lhs, c.rhs, pl),
                    _ => prob.bc(c.lhs, c.rhs, pl),
                }?;
            }
            if all {
                pdsolve_all(&prob, &opts, pl)
            } else {
                pdsolve(&prob, &opts, pl).map(|s| vec![s])
            }
        })
        .map_err(pde_err)?
    };
    let out = PyList::empty(py);
    for s in &sols {
        out.append(solution_dict(py, s, &pool)?)?;
    }
    Ok(out)
}

/// Private: the valid `pdsolve` hints with their milestones and
/// descriptions.
#[pyfunction]
fn _pde_solve_hints() -> Vec<(&'static str, &'static str, &'static str)> {
    alkahest_core::pde::TIERS
        .iter()
        .map(|t| (t.name, t.milestone, t.description))
        .collect()
}

/// Register the PDE surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJetSpace>()?;
    m.add_function(wrap_pyfunction!(py_functions_to_jets, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_classify, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_domain_condition, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_check_problem, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_verify, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_solve, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_solve_hints, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_sum, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_integral, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_kronecker, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_root_of_family, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_root_family_member, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_root_family_enclosure, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_expand_sum, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_expand_finite_sums, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_truncate, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_simplify_sums, m)?)?;
    m.add_function(wrap_pyfunction!(_pde_laplacian_indexed, m)?)?;
    m.add("PdeError", m.py().get_type::<PyPdeError>())?;
    Ok(())
}
