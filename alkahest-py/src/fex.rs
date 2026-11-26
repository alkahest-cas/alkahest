//! PyO3 bindings for `alkahest_core::fex` — the Finite Expression Method
//! evaluation engine (experimental).
//!
//! Native surface (wrapped by the pure-Python `alkahest.fex` module):
//! `_FexTemplate`, `_FexProblem`, `_FexEvaluator` and `FexError`
//! (`E-FEX-001` … `E-FEX-008`).
//!
//! Arrays cross the boundary through `__array_interface__` (no NumPy binding
//! needed): opcode batches are C-contiguous `int64`, everything else
//! C-contiguous `float64`. Outputs are written into caller-allocated arrays,
//! so a NumPy (or DLPack-imported) buffer is filled in place, and the GIL is
//! released for the duration of the computation.

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use alkahest_core::fex::{
    canonical_key, canonical_skeletons, fit_batch, redundant_params, same_function, Evaluator,
    FexError, FitMethod, FitOptions, JetSymbols, LinearOperator, LossKind, NonlocalTerm,
    PdeOperator, Problem, ScalarField, SecondOrderForm, SlotKind, Template,
};
use alkahest_core::ExprId;

use crate::{PyExpr, PyExprPool};

pyo3::create_exception!(alkahest, PyFexError, crate::PyAlkahestError);

fn fex_err(e: FexError) -> PyErr {
    Python::attach(|py| {
        let exc = py.get_type::<PyFexError>();
        crate::make_structured_err(py, &exc, &e)
    })
}

/// A C-contiguous array seen through `__array_interface__`.
struct View {
    addr: usize,
    shape: Vec<usize>,
}

impl View {
    fn len(&self) -> usize {
        self.shape.iter().product()
    }
}

fn view(obj: &Bound<'_, PyAny>, typestr: &str, writable: bool, what: &str) -> PyResult<View> {
    let kind = if typestr.ends_with("i8") {
        "int64"
    } else {
        "float64"
    };
    let bad = |why: &str| {
        PyTypeError::new_err(format!(
            "{what} must be a C-contiguous {kind} NumPy array ({why})"
        ))
    };
    let iface = obj
        .getattr("__array_interface__")
        .map_err(|_| bad("it has no __array_interface__"))?;
    let d = iface
        .cast::<PyDict>()
        .map_err(|_| bad("__array_interface__ is not a dict"))?;
    let get = |key: &str| d.get_item(key).ok().flatten();
    let ts: String = get("typestr")
        .and_then(|t| t.extract().ok())
        .ok_or_else(|| bad("no typestr"))?;
    let native = if cfg!(target_endian = "little") {
        format!("<{}", &typestr[1..])
    } else {
        format!(">{}", &typestr[1..])
    };
    if ts != native {
        return Err(bad(&format!("dtype {ts}")));
    }
    if get("mask").is_some_and(|m| !m.is_none()) {
        return Err(bad("masked arrays are not supported"));
    }
    let shape: Vec<usize> = get("shape")
        .and_then(|t| t.extract().ok())
        .ok_or_else(|| bad("no shape"))?;
    if let Some(strides) = get("strides").filter(|s| !s.is_none()) {
        let strides: Vec<isize> = strides.extract().map_err(|_| bad("unreadable strides"))?;
        let mut expect = 8isize;
        for (&n, &st) in shape.iter().zip(&strides).rev() {
            if n > 1 && st != expect {
                return Err(bad("not C-contiguous"));
            }
            expect = expect.saturating_mul(n as isize);
        }
    }
    let (addr, readonly): (usize, bool) = get("data")
        .and_then(|t| t.extract().ok())
        .ok_or_else(|| bad("no data pointer"))?;
    if writable && readonly {
        return Err(PyValueError::new_err(format!("{what} is read-only")));
    }
    if addr % 8 != 0 && shape.iter().product::<usize>() > 0 {
        return Err(bad("misaligned data"));
    }
    Ok(View { addr, shape })
}

/// Copy a float64 array into a Vec (the caller keeps `obj` alive).
fn read_f64(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let v = view(obj, "<f8", false, what)?;
    let n = v.len();
    let data = if n == 0 {
        Vec::new()
    } else {
        // SAFETY: `view` checked dtype, contiguity and alignment; `obj` is
        // alive for this call and we copy before returning.
        unsafe { std::slice::from_raw_parts(v.addr as *const f64, n).to_vec() }
    };
    Ok((data, v.shape))
}

fn read_i64(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let v = view(obj, "<i8", false, what)?;
    let n = v.len();
    let raw: &[i64] = if n == 0 {
        &[]
    } else {
        // SAFETY: as in `read_f64`.
        unsafe { std::slice::from_raw_parts(v.addr as *const i64, n) }
    };
    let mut out = Vec::with_capacity(n);
    for &x in raw {
        if x < 0 {
            return Err(fex_err(FexError::Opcode {
                detail: format!("negative opcode {x}"),
            }));
        }
        out.push(x as usize);
    }
    Ok((out, v.shape))
}

/// A writable float64 output buffer of exactly `n` elements.
///
/// # Safety
/// The returned slice aliases `obj`'s memory; the caller must keep `obj`
/// alive and not resize it while the slice is in use.
unsafe fn write_f64<'a>(obj: &Bound<'_, PyAny>, n: usize, what: &str) -> PyResult<&'a mut [f64]> {
    let v = view(obj, "<f8", true, what)?;
    if v.len() != n {
        return Err(fex_err(FexError::Points {
            detail: format!("{what} has {} elements, need {n}", v.len()),
        }));
    }
    if n == 0 {
        return Ok(&mut []);
    }
    Ok(std::slice::from_raw_parts_mut(v.addr as *mut f64, n))
}

fn expr_ids(
    py: Python<'_>,
    items: &[PyRef<'_, PyExpr>],
    pool: &Py<PyExprPool>,
) -> PyResult<Vec<ExprId>> {
    items
        .iter()
        .map(|e| {
            if e.pool.as_ptr() != pool.as_ptr() {
                Err(PyValueError::new_err(
                    "all expressions must come from the same ExprPool",
                ))
            } else {
                let _ = py;
                Ok(e.id)
            }
        })
        .collect()
}

/// A FEX template: tree shape, operator tables, dimension.
#[pyclass(name = "_FexTemplate", module = "alkahest", frozen)]
pub struct PyFexTemplate {
    pub(crate) inner: Arc<Template>,
}

#[pymethods]
impl PyFexTemplate {
    #[new]
    fn new(shape: &str, unary: Vec<String>, binary: Vec<String>, dim: usize) -> PyResult<Self> {
        let u: Vec<&str> = unary.iter().map(String::as_str).collect();
        let b: Vec<&str> = binary.iter().map(String::as_str).collect();
        let t = Template::from_names(shape, &u, &b, dim).map_err(fex_err)?;
        Ok(PyFexTemplate { inner: Arc::new(t) })
    }

    #[getter]
    fn shape(&self) -> String {
        self.inner.shape().to_string()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn n_slots(&self) -> usize {
        self.inner.n_slots()
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    #[getter]
    fn unary(&self) -> Vec<String> {
        self.inner
            .unary_table()
            .iter()
            .map(|u| u.name().to_string())
            .collect()
    }

    #[getter]
    fn binary(&self) -> Vec<String> {
        self.inner
            .binary_table()
            .iter()
            .map(|u| u.name().to_string())
            .collect()
    }

    #[getter]
    fn slot_arities(&self) -> Vec<usize> {
        self.inner.slot_arities()
    }

    #[getter]
    fn slot_kinds(&self) -> Vec<&'static str> {
        self.inner
            .slots()
            .iter()
            .map(|s| match s.kind {
                SlotKind::Leaf => "leaf",
                SlotKind::Unary => "unary",
                _ => "binary",
            })
            .collect()
    }

    #[getter]
    fn param_offsets(&self) -> Vec<usize> {
        self.inner.slots().iter().map(|s| s.param_offset).collect()
    }

    #[getter]
    fn n_skeletons(&self) -> u128 {
        self.inner.n_skeletons()
    }

    fn describe(&self, e: Vec<usize>) -> PyResult<String> {
        self.inner.describe(&e).map_err(fex_err)
    }

    fn canonical_key(&self, e: Vec<usize>) -> PyResult<String> {
        canonical_key(&self.inner, &e).map_err(fex_err)
    }

    fn redundant_params(&self, e: Vec<usize>) -> PyResult<Vec<Option<f64>>> {
        redundant_params(&self.inner, &e).map_err(fex_err)
    }

    #[pyo3(signature = (limit = 1_000_000))]
    fn canonical_skeletons(&self, limit: u128) -> PyResult<Vec<(String, Vec<usize>)>> {
        canonical_skeletons(&self.inner, limit).map_err(fex_err)
    }

    fn instantiate(
        &self,
        py: Python<'_>,
        e: Vec<usize>,
        theta: Vec<f64>,
        coords: Vec<PyRef<'_, PyExpr>>,
    ) -> PyResult<PyExpr> {
        let pool = coords
            .first()
            .ok_or_else(|| PyValueError::new_err("coords must not be empty"))?
            .pool
            .clone_ref(py);
        let ids = expr_ids(py, &coords, &pool)?;
        let id = {
            let p = pool.borrow(py);
            self.inner
                .instantiate(&e, &theta, &ids, &p.inner)
                .map_err(fex_err)?
        };
        Ok(PyExpr { id, pool })
    }

    fn instantiate_symbolic(
        &self,
        py: Python<'_>,
        e: Vec<usize>,
        params: Vec<PyRef<'_, PyExpr>>,
        coords: Vec<PyRef<'_, PyExpr>>,
    ) -> PyResult<PyExpr> {
        let pool = coords
            .first()
            .ok_or_else(|| PyValueError::new_err("coords must not be empty"))?
            .pool
            .clone_ref(py);
        let ids = expr_ids(py, &coords, &pool)?;
        let ps = expr_ids(py, &params, &pool)?;
        let id = {
            let p = pool.borrow(py);
            self.inner
                .instantiate_with(&e, &ps, &ids, &p.inner)
                .map_err(fex_err)?
        };
        Ok(PyExpr { id, pool })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (e1, theta1, e2, theta2, coords, max_den = 1000, tol = 1e-9))]
    fn same_function(
        &self,
        py: Python<'_>,
        e1: Vec<usize>,
        theta1: Vec<f64>,
        e2: Vec<usize>,
        theta2: Vec<f64>,
        coords: Vec<PyRef<'_, PyExpr>>,
        max_den: i64,
        tol: f64,
    ) -> PyResult<Option<bool>> {
        let pool = coords
            .first()
            .ok_or_else(|| PyValueError::new_err("coords must not be empty"))?
            .pool
            .clone_ref(py);
        let ids = expr_ids(py, &coords, &pool)?;
        let p = pool.borrow(py);
        same_function(
            &self.inner,
            &e1,
            &theta1,
            &e2,
            &theta2,
            &ids,
            &p.inner,
            max_den,
            tol,
        )
        .map_err(fex_err)
    }
}

fn field(
    py: Python<'_>,
    v: Option<&Bound<'_, PyAny>>,
    coords: &[ExprId],
    pool: Option<&Py<PyExprPool>>,
) -> PyResult<ScalarField> {
    let Some(v) = v else {
        return Ok(ScalarField::Zero);
    };
    if v.is_none() {
        return Ok(ScalarField::Zero);
    }
    if let Ok(e) = v.extract::<PyRef<PyExpr>>() {
        let pool = pool.ok_or_else(|| {
            PyValueError::new_err("an expression source term needs the coordinate symbols")
        })?;
        if e.pool.as_ptr() != pool.as_ptr() {
            return Err(PyValueError::new_err(
                "source terms must come from the same ExprPool as the coordinates",
            ));
        }
        let p = pool.borrow(py);
        return ScalarField::from_expr(e.id, coords, &p.inner).map_err(fex_err);
    }
    if let Ok(c) = v.extract::<f64>() {
        return Ok(if c == 0.0 {
            ScalarField::Zero
        } else {
            ScalarField::Const(c)
        });
    }
    Err(PyTypeError::new_err(
        "a source term must be a number, an Expr in the coordinates, or None",
    ))
}

/// A PDE problem for FEX scoring.
#[pyclass(name = "_FexProblem", module = "alkahest", frozen)]
pub struct PyFexProblem {
    pub(crate) inner: Arc<Problem>,
}

#[pymethods]
impl PyFexProblem {
    /// Build a problem.
    ///
    /// `kind` is `"poisson"`, `"transport"` (needs `grad_coeffs`),
    /// `"cubic_schrodinger"`, `"linear"` (needs `linear=(lap, grad, u, cubic,
    /// potential)`), or `"expr"` (needs `operator` and the jet symbols).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        dim, kind, *, rhs = None, bc = None, potential = None, coords = None,
        grad_coeffs = None, linear = None, operator = None, u = None, lap = None,
        grad = None, ut = None, time = None, bc_weight = 100.0, loss = "least_squares",
        power = 3.0, nonlocal_terms = None, second_order = None
    ))]
    fn new(
        py: Python<'_>,
        dim: usize,
        kind: &str,
        rhs: Option<&Bound<'_, PyAny>>,
        bc: Option<&Bound<'_, PyAny>>,
        potential: Option<&Bound<'_, PyAny>>,
        coords: Option<Vec<PyRef<'_, PyExpr>>>,
        grad_coeffs: Option<Vec<f64>>,
        linear: Option<(f64, Vec<f64>, f64, f64, bool)>,
        operator: Option<PyRef<'_, PyExpr>>,
        u: Option<PyRef<'_, PyExpr>>,
        lap: Option<PyRef<'_, PyExpr>>,
        grad: Option<Vec<PyRef<'_, PyExpr>>>,
        ut: Option<PyRef<'_, PyExpr>>,
        time: Option<usize>,
        bc_weight: f64,
        loss: &str,
        power: f64,
        nonlocal_terms: Option<Vec<(Vec<f64>, Vec<f64>, f64, f64)>>,
        second_order: Option<(Vec<f64>, Vec<(usize, usize, f64)>)>,
    ) -> PyResult<Self> {
        let pool: Option<Py<PyExprPool>> = coords
            .as_ref()
            .and_then(|c| c.first())
            .map(|e| e.pool.clone_ref(py));
        let cids: Vec<ExprId> = match (&coords, &pool) {
            (Some(c), Some(p)) => expr_ids(py, c, p)?,
            _ => Vec::new(),
        };
        if !cids.is_empty() && cids.len() != dim {
            return Err(fex_err(FexError::Dimension {
                detail: format!("{} coordinate symbols for dimension {dim}", cids.len()),
            }));
        }
        let op = match kind {
            "poisson" => PdeOperator::poisson(),
            "transport" => PdeOperator::transport(grad_coeffs.ok_or_else(|| {
                PyValueError::new_err("kind='transport' needs grad_coeffs")
            })?),
            "cubic_schrodinger" => PdeOperator::cubic_schrodinger(),
            "linear" => {
                let (l, g, uu, c, pot) = linear
                    .ok_or_else(|| PyValueError::new_err("kind='linear' needs linear=(...)"))?;
                PdeOperator::Linear(LinearOperator {
                    lap: l,
                    grad: g,
                    u: uu,
                    cubic: c,
                    potential: pot,
                })
            }
            "expr" => {
                let pool = pool.as_ref().ok_or_else(|| {
                    PyValueError::new_err("kind='expr' needs the coordinate symbols")
                })?;
                let one = |e: Option<PyRef<'_, PyExpr>>, name: &str| -> PyResult<ExprId> {
                    let e = e.ok_or_else(|| {
                        PyValueError::new_err(format!("kind='expr' needs {name}"))
                    })?;
                    Ok(expr_ids(py, std::slice::from_ref(&e), pool)?[0])
                };
                let opid = one(operator, "operator")?;
                let js = JetSymbols {
                    u: one(u, "u")?,
                    lap: one(lap, "lap")?,
                    grad: expr_ids(
                        py,
                        &grad.ok_or_else(|| PyValueError::new_err("kind='expr' needs grad"))?,
                        pool,
                    )?,
                    coords: cids.clone(),
                    ut: match ut {
                        Some(e) => Some(expr_ids(py, std::slice::from_ref(&e), pool)?[0]),
                        None => None,
                    },
                };
                let p = pool.borrow(py);
                PdeOperator::from_expr(opid, &js, time, &p.inner).map_err(fex_err)?
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown problem kind {other:?}; use poisson, transport, cubic_schrodinger, linear or expr"
                )))
            }
        };
        let rhs = field(py, rhs, &cids, pool.as_ref())?;
        let bc = field(py, bc, &cids, pool.as_ref())?;
        let pot = field(py, potential, &cids, pool.as_ref())?;
        let loss = match loss {
            "least_squares" | "ls" => LossKind::LeastSquares,
            "normalized" => LossKind::Normalized { power },
            "rayleigh" => LossKind::Rayleigh,
            "centered" | "least_squares_centered" => LossKind::CenteredLeastSquares,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown loss {other:?}; use least_squares, normalized, rayleigh or centered"
                )))
            }
        };
        let mut p = Problem::new(dim, op, rhs, bc)
            .map_err(fex_err)?
            .with_potential(pot)
            .with_bc_weight(bc_weight)
            .with_loss(loss);
        if let Some(t) = time {
            p = p.with_time(t).map_err(fex_err)?;
        }
        if let Some((diag, off)) = second_order {
            p = p
                .with_second_order(SecondOrderForm { diag, off })
                .map_err(fex_err)?;
        }
        if let Some(terms) = nonlocal_terms {
            let terms = terms
                .into_iter()
                .map(|(scale, shift, u, lap)| NonlocalTerm {
                    scale,
                    shift,
                    u,
                    lap,
                })
                .collect();
            p = p.with_nonlocal(terms).map_err(fex_err)?;
        }
        Ok(PyFexProblem { inner: Arc::new(p) })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn time(&self) -> Option<usize> {
        self.inner.time()
    }

    #[getter]
    fn loss(&self) -> &'static str {
        match self.inner.loss() {
            LossKind::LeastSquares => "least_squares",
            LossKind::Normalized { .. } => "normalized",
            LossKind::Rayleigh => "rayleigh",
            LossKind::CenteredLeastSquares => "centered",
            _ => "other",
        }
    }

    #[getter]
    fn operator_kind(&self) -> &'static str {
        match self.inner.operator() {
            PdeOperator::Linear(_) => "linear",
            _ => "compiled",
        }
    }

    #[getter]
    fn n_nonlocal(&self) -> usize {
        self.inner.nonlocal().len()
    }

    #[getter]
    fn order(&self) -> u8 {
        self.inner.operator().order()
    }
}

/// The compiled scoring kernel for one template, problem and point set.
#[pyclass(name = "_FexEvaluator", module = "alkahest", frozen)]
pub struct PyFexEvaluator {
    inner: Arc<Evaluator>,
    ns: usize,
    np: usize,
}

impl PyFexEvaluator {
    fn batch(
        &self,
        e: &Bound<'_, PyAny>,
        theta: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<usize>, Vec<f64>, usize)> {
        let (es, esh) = read_i64(e, "E")?;
        let (th, _) = read_f64(theta, "Theta")?;
        let b = if esh.len() == 2 {
            esh[0]
        } else if esh.len() == 1 && self.ns > 0 {
            esh[0] / self.ns
        } else {
            0
        };
        if es.len() != b * self.ns || th.len() != b * self.np {
            return Err(fex_err(FexError::Points {
                detail: format!(
                    "E must be (batch, {}) and Theta (batch, {}); got {} and {} elements",
                    self.ns,
                    self.np,
                    es.len(),
                    th.len()
                ),
            }));
        }
        Ok((es, th, b))
    }
}

#[pymethods]
impl PyFexEvaluator {
    #[new]
    #[pyo3(signature = (template, problem, interior, boundary = None))]
    fn new(
        py: Python<'_>,
        template: &PyFexTemplate,
        problem: &PyFexProblem,
        interior: &Bound<'_, PyAny>,
        boundary: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dim = template.inner.dim();
        let (xi, shi) = read_f64(interior, "interior")?;
        if shi.len() != 2 || shi[1] != dim {
            return Err(fex_err(FexError::Points {
                detail: format!("interior must have shape (n, {dim}), got {shi:?}"),
            }));
        }
        let xb = match boundary {
            Some(b) if !b.is_none() => {
                let (xb, shb) = read_f64(b, "boundary")?;
                if shb.len() != 2 || shb[1] != dim {
                    return Err(fex_err(FexError::Points {
                        detail: format!("boundary must have shape (n, {dim}), got {shb:?}"),
                    }));
                }
                xb
            }
            _ => Vec::new(),
        };
        let t = template.inner.clone();
        let p = problem.inner.clone();
        let ev = py
            .detach(move || Evaluator::new(&t, &p, &xi, &xb))
            .map_err(fex_err)?;
        Ok(PyFexEvaluator {
            ns: template.inner.n_slots(),
            np: template.inner.n_params(),
            inner: Arc::new(ev),
        })
    }

    #[getter]
    fn n_residuals(&self) -> usize {
        self.inner.n_residuals()
    }

    #[getter]
    fn n_interior(&self) -> usize {
        self.inner.n_interior()
    }

    #[getter]
    fn n_boundary(&self) -> usize {
        self.inner.n_boundary()
    }

    /// Fill `R` (batch × M) and optionally `J` (batch × M × P); returns one
    /// validity flag per candidate.
    #[pyo3(signature = (e, theta, r, j = None))]
    fn residual_and_jacobian_into(
        &self,
        py: Python<'_>,
        e: &Bound<'_, PyAny>,
        theta: &Bound<'_, PyAny>,
        r: &Bound<'_, PyAny>,
        j: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<bool>> {
        let (es, th, b) = self.batch(e, theta)?;
        let m = self.inner.n_residuals();
        // SAFETY: `r` / `j` are kept alive by the caller's references for the
        // duration of this call; the GIL is released but the arrays cannot
        // be freed while borrowed here.
        let rs = unsafe { write_f64(r, b * m, "R")? };
        let js = match j {
            Some(j) if !j.is_none() => Some(unsafe { write_f64(j, b * m * self.np, "J")? }),
            _ => None,
        };
        let ev = self.inner.clone();
        let rs_addr = rs.as_mut_ptr() as usize;
        let rs_len = rs.len();
        let js_raw = js.map(|s| (s.as_mut_ptr() as usize, s.len()));
        py.detach(move || {
            // SAFETY: see above; the slices are rebuilt from the same
            // pointers and lengths.
            let rs = unsafe { std::slice::from_raw_parts_mut(rs_addr as *mut f64, rs_len) };
            let js =
                js_raw.map(|(a, n)| unsafe { std::slice::from_raw_parts_mut(a as *mut f64, n) });
            ev.residual_and_jacobian_batch(&es, &th, rs, js)
        })
        .map_err(fex_err)
    }

    /// Fill `losses` (batch) and optionally `grads` (batch × P).
    #[pyo3(signature = (e, theta, losses, grads = None))]
    fn loss_into(
        &self,
        py: Python<'_>,
        e: &Bound<'_, PyAny>,
        theta: &Bound<'_, PyAny>,
        losses: &Bound<'_, PyAny>,
        grads: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let (es, th, b) = self.batch(e, theta)?;
        // SAFETY: as in `residual_and_jacobian_into`.
        let ls = unsafe { write_f64(losses, b, "losses")? };
        let gs = match grads {
            Some(g) if !g.is_none() => Some(unsafe { write_f64(g, b * self.np, "grads")? }),
            _ => None,
        };
        let ev = self.inner.clone();
        let out = py
            .detach(move || match gs {
                Some(g) => ev.loss_and_grad_batch(&es, &th, g),
                None => ev.loss_batch(&es, &th),
            })
            .map_err(fex_err)?;
        ls.copy_from_slice(&out);
        Ok(())
    }

    /// `(u, grad, lap)` of one candidate at the given points, written into
    /// `u` (n), `grad` (n × d), `lap` (n).
    #[allow(clippy::too_many_arguments)]
    fn jets_into(
        &self,
        py: Python<'_>,
        e: Vec<usize>,
        theta: Vec<f64>,
        points: &Bound<'_, PyAny>,
        u: &Bound<'_, PyAny>,
        grad: &Bound<'_, PyAny>,
        lap: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (x, _) = read_f64(points, "points")?;
        let ev = self.inner.clone();
        let jets = py
            .detach(move || ev.jets_at(&e, &theta, &x))
            .map_err(fex_err)?;
        let n = jets.u.len();
        // SAFETY: the output arrays are alive for this call.
        unsafe {
            write_f64(u, n, "u")?.copy_from_slice(&jets.u);
            write_f64(grad, jets.grad.len(), "grad")?.copy_from_slice(&jets.grad);
            write_f64(lap, n, "lap")?.copy_from_slice(&jets.lap);
        }
        Ok(())
    }

    /// Fit a batch; returns a list of `(theta, loss, iterations, evaluations)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (e, theta0, method = "lm", max_iter = 50, lr = 1e-3, adam_iters = 20, bfgs_iters = 20, tol = 1e-30, freeze_redundant = true))]
    fn fit_batch<'py>(
        &self,
        py: Python<'py>,
        e: &Bound<'py, PyAny>,
        theta0: &Bound<'py, PyAny>,
        method: &str,
        max_iter: usize,
        lr: f64,
        adam_iters: usize,
        bfgs_iters: usize,
        tol: f64,
        freeze_redundant: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let (es, th, _) = self.batch(e, theta0)?;
        let method = match method {
            "lm" => FitMethod::Lm { max_iter },
            "adam" => FitMethod::Adam {
                iters: max_iter,
                lr,
            },
            "bfgs" => FitMethod::Bfgs { iters: max_iter },
            "adam_bfgs" | "paper" => FitMethod::AdamBfgs {
                adam: adam_iters,
                lr,
                bfgs: bfgs_iters,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown fit method {other:?}; use lm, adam, bfgs or adam_bfgs"
                )))
            }
        };
        let opts = FitOptions {
            method,
            tol,
            freeze_redundant,
        };
        let ev = self.inner.clone();
        let res = py
            .detach(move || fit_batch(&ev, &es, &th, &opts))
            .map_err(fex_err)?;
        let items: Vec<(Vec<f64>, f64, usize, usize)> = res
            .into_iter()
            .map(|r| (r.theta, r.loss, r.iterations, r.evaluations))
            .collect();
        PyList::new(py, items)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFexTemplate>()?;
    m.add_class::<PyFexProblem>()?;
    m.add_class::<PyFexEvaluator>()?;
    m.add("FexError", m.py().get_type::<PyFexError>())?;
    Ok(())
}
