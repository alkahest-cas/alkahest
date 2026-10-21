//! PyO3 bindings for `alkahest_core::tensor` — metric → curvature on a
//! coordinate chart (F27).
//!
//! The native surface is private (`_TensorMetric`, `_tensor_canonical`,
//! `_tensor_derivative`); `alkahest.tensor` builds the public `Metric`,
//! `Tensor` and index classes on top of it. Refusals arrive as
//! `alkahest.TensorError` with a stable `E-TENSOR-NNN` `.code`; budget trips
//! as `BudgetExceededError`.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use alkahest_core::tensor::{
    canonical_derivative as core_canonical_derivative, canonical_form as core_canonical_form,
    Metric, TensorError,
};
use alkahest_core::ExprId;

use crate::{coerce_substituent, guard_depth_all, PyExpr, PyExprPool};

pyo3::create_exception!(alkahest, PyTensorError, crate::PyAlkahestError);

fn tensor_err(e: TensorError) -> PyErr {
    Python::with_gil(|py| {
        if let Some(trip) = e.budget_trip() {
            let exc_type = py.get_type_bound::<crate::PyBudgetExceededError>();
            return crate::make_structured_err(py, &exc_type, &trip);
        }
        let exc_type = py.get_type_bound::<PyTensorError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

fn wrap(py: Python<'_>, pool: &Py<PyExprPool>, id: ExprId) -> PyObject {
    PyExpr {
        id,
        pool: pool.clone_ref(py),
    }
    .into_py(py)
}

/// The pool of the first `Expr` found anywhere in `ob` (recursing into lists/tuples).
fn find_pool(py: Python<'_>, ob: &Bound<'_, PyAny>) -> Option<Py<PyExprPool>> {
    if let Ok(e) = ob.extract::<PyRef<PyExpr>>() {
        return Some(e.pool.clone_ref(py));
    }
    if let Ok(l) = ob.downcast::<PyList>() {
        return l.iter().find_map(|x| find_pool(py, &x));
    }
    if let Ok(t) = ob.downcast::<PyTuple>() {
        return t.iter().find_map(|x| find_pool(py, &x));
    }
    None
}

fn vector(py: Python<'_>, pool: &Py<PyExprPool>, ob: &Bound<'_, PyAny>) -> PyResult<Vec<ExprId>> {
    let items: Vec<Bound<'_, PyAny>> = ob.extract().map_err(|_| {
        PyTypeError::new_err("expected a sequence of expressions (Expr, int or Fraction)")
    })?;
    let ids = items
        .iter()
        .map(|x| coerce_substituent(pool, x, py))
        .collect::<PyResult<Vec<ExprId>>>()?;
    guard_depth_all(&pool.borrow(py).inner, &ids)?;
    Ok(ids)
}

fn nest2(py: Python<'_>, pool: &Py<PyExprPool>, t: Vec<Vec<ExprId>>) -> PyObject {
    let rows: Vec<PyObject> = t
        .into_iter()
        .map(|r| {
            let v: Vec<PyObject> = r.into_iter().map(|e| wrap(py, pool, e)).collect();
            PyList::new_bound(py, v).into_py(py)
        })
        .collect();
    PyList::new_bound(py, rows).into_py(py)
}

fn nest3(py: Python<'_>, pool: &Py<PyExprPool>, t: Vec<Vec<Vec<ExprId>>>) -> PyObject {
    let v: Vec<PyObject> = t.into_iter().map(|m| nest2(py, pool, m)).collect();
    PyList::new_bound(py, v).into_py(py)
}

fn nest4(py: Python<'_>, pool: &Py<PyExprPool>, t: Vec<Vec<Vec<Vec<ExprId>>>>) -> PyObject {
    let v: Vec<PyObject> = t.into_iter().map(|m| nest3(py, pool, m)).collect();
    PyList::new_bound(py, v).into_py(py)
}

/// Native metric (private; use :class:`alkahest.Metric`).
#[pyclass(name = "_TensorMetric", module = "alkahest", unsendable)]
pub(crate) struct PyTensorMetric {
    inner: Metric,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyTensorMetric {
    #[new]
    fn new(py: Python<'_>, g: &Bound<'_, PyAny>, coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        let pool = find_pool(py, coords)
            .or_else(|| find_pool(py, g))
            .ok_or_else(|| PyTypeError::new_err("coordinates must be Expr symbols"))?;
        let cs = vector(py, &pool, coords)?;
        let rows: Vec<Bound<'_, PyAny>> = g
            .extract()
            .map_err(|_| PyTypeError::new_err("the metric must be a list of rows"))?;
        let mut gm = Vec::with_capacity(rows.len());
        for row in &rows {
            gm.push(vector(py, &pool, row)?);
        }
        let inner = {
            let p = pool.borrow(py);
            Metric::new(&gm, &cs, &p.inner).map_err(tensor_err)?
        };
        Ok(PyTensorMetric { inner, pool })
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn coords(&self, py: Python<'_>) -> Vec<PyObject> {
        self.inner
            .coords()
            .iter()
            .map(|&c| wrap(py, &self.pool, c))
            .collect()
    }

    fn timings(&self) -> Vec<(String, f64)> {
        self.inner
            .timings()
            .into_iter()
            .map(|(n, t)| (n.to_string(), t))
            .collect()
    }

    fn components(&self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self.inner.components(&p.inner).map_err(tensor_err)?;
        Ok(nest2(py, &self.pool, t))
    }

    fn determinant(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let e = self.inner.determinant(&p.inner).map_err(tensor_err)?;
        Ok(wrap(py, &self.pool, e))
    }

    fn inverse(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self.inner.inverse(&p.inner).map_err(tensor_err)?;
        Ok(nest2(py, &self.pool, t))
    }

    fn christoffel(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self.inner.christoffel(&p.inner).map_err(tensor_err)?;
        Ok(nest3(py, &self.pool, t))
    }

    fn christoffel_first_kind(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self
            .inner
            .christoffel_first_kind(&p.inner)
            .map_err(tensor_err)?;
        Ok(nest3(py, &self.pool, t))
    }

    fn riemann(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self.inner.riemann(&p.inner).map_err(tensor_err)?;
        Ok(nest4(py, &self.pool, t))
    }

    fn riemann_lower(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self.inner.riemann_lower(&p.inner).map_err(tensor_err)?;
        Ok(nest4(py, &self.pool, t))
    }

    fn ricci(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self.inner.ricci(&p.inner).map_err(tensor_err)?;
        Ok(nest2(py, &self.pool, t))
    }

    fn ricci_scalar(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let e = self.inner.ricci_scalar(&p.inner).map_err(tensor_err)?;
        Ok(wrap(py, &self.pool, e))
    }

    fn is_ricci_flat(&mut self) -> PyResult<bool> {
        self.inner.is_ricci_flat().map_err(tensor_err)
    }

    fn einstein(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self.inner.einstein(&p.inner).map_err(tensor_err)?;
        Ok(nest2(py, &self.pool, t))
    }

    fn weyl(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let t = self.inner.weyl(&p.inner).map_err(tensor_err)?;
        Ok(nest4(py, &self.pool, t))
    }

    fn kretschmann(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let p = self.pool.borrow(py);
        let e = self.inner.kretschmann(&p.inner).map_err(tensor_err)?;
        Ok(wrap(py, &self.pool, e))
    }

    fn geodesic_accelerations(
        &mut self,
        py: Python<'_>,
        velocities: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyObject>> {
        let vs = vector(py, &self.pool, velocities)?;
        let p = self.pool.borrow(py);
        let out = self
            .inner
            .geodesic_accelerations(&vs, &p.inner)
            .map_err(tensor_err)?;
        Ok(out.into_iter().map(|e| wrap(py, &self.pool, e)).collect())
    }
}

/// Canonical form of each expression in the tensor layer's exact field
/// (private; used by `alkahest.tensor`).
#[pyfunction]
#[pyo3(name = "_tensor_canonical")]
fn py_tensor_canonical(
    py: Python<'_>,
    exprs: &Bound<'_, PyAny>,
    coords: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let Some(pool) = find_pool(py, coords).or_else(|| find_pool(py, exprs)) else {
        return Err(PyTypeError::new_err("coordinates must be Expr symbols"));
    };
    let es = vector(py, &pool, exprs)?;
    let cs = vector(py, &pool, coords)?;
    let out = {
        let p = pool.borrow(py);
        core_canonical_form(&es, &cs, &p.inner).map_err(tensor_err)?
    };
    Ok(out.into_iter().map(|e| wrap(py, &pool, e)).collect())
}

/// `∂e/∂coords[var]` for each expression, in canonical form (private).
#[pyfunction]
#[pyo3(name = "_tensor_derivative")]
fn py_tensor_derivative(
    py: Python<'_>,
    exprs: &Bound<'_, PyAny>,
    coords: &Bound<'_, PyAny>,
    var: usize,
) -> PyResult<Vec<PyObject>> {
    let Some(pool) = find_pool(py, coords).or_else(|| find_pool(py, exprs)) else {
        return Err(PyTypeError::new_err("coordinates must be Expr symbols"));
    };
    let es = vector(py, &pool, exprs)?;
    let cs = vector(py, &pool, coords)?;
    let out = {
        let p = pool.borrow(py);
        core_canonical_derivative(&es, &cs, var, &p.inner).map_err(tensor_err)?
    };
    Ok(out.into_iter().map(|e| wrap(py, &pool, e)).collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("TensorError", m.py().get_type_bound::<PyTensorError>())?;
    m.add_class::<PyTensorMetric>()?;
    m.add_function(wrap_pyfunction!(py_tensor_canonical, m)?)?;
    m.add_function(wrap_pyfunction!(py_tensor_derivative, m)?)?;
    Ok(())
}
