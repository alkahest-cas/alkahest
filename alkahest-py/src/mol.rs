//! PyO3 bindings for `alkahest_core::pde::mol` — the method-of-lines runtime
//! (stencil-kernel right-hand sides, exact sparse Jacobians, native RK45 /
//! BDF integrators).
//!
//! Private surface: `alkahest.experimental.method_of_lines` builds the
//! program description in Python and drives it through `_MolProgram`.

use std::sync::Arc;

use alkahest_core::kernel::ExprId;
use alkahest_core::pde::mol::{
    integrate, MolDenseOp, MolError, MolInput, MolIntegrateOptions, MolKernelOp, MolMethod, MolOp,
    MolProgram, MolTarget,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule, PyTuple};

use crate::{array_view, extract_expr_in, jit_compile_error_to_py, PyExpr, PyExprPool};

fn mol_err(e: MolError) -> PyErr {
    match e {
        MolError::Integration(m) => PyRuntimeError::new_err(m),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn get<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("op is missing {key:?}")))
}

fn read_f64s(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    if let Ok(v) = array_view(obj, false, what) {
        let n = v.len();
        // SAFETY: `array_view` validated a live, aligned, C-contiguous float64
        // buffer of `n` elements; it is copied before `obj` can change.
        let s = unsafe { std::slice::from_raw_parts(v.addr as *const f64, n) };
        return Ok(s.to_vec());
    }
    obj.extract()
}

fn compile(
    py: Python<'_>,
    exprs: &Bound<'_, PyAny>,
    syms: &[ExprId],
    pool: &Py<PyExprPool>,
) -> PyResult<Arc<alkahest_core::CompiledVecFn>> {
    let ids: Vec<ExprId> = exprs
        .try_iter()?
        .map(|e| Ok(extract_expr_in(&e?, pool, "method_of_lines", "a kernel")?.id))
        .collect::<PyResult<_>>()?;
    let p = pool.borrow(py);
    let f = alkahest_core::compile_many(&ids, syms, &p.inner).map_err(jit_compile_error_to_py)?;
    Ok(Arc::new(f))
}

fn f64_bytes<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyBytes> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    PyBytes::new(py, &b)
}

fn usize_bytes<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyBytes> {
    let b: Vec<u8> = v.iter().flat_map(|&x| (x as u64).to_le_bytes()).collect();
    PyBytes::new(py, &b)
}

fn region(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    obj.extract()
}

fn parse_op(py: Python<'_>, d: &Bound<'_, PyDict>) -> PyResult<MolOp> {
    let kind: String = get(d, "kind")?.extract()?;
    match kind.as_str() {
        "dense" => Ok(MolOp::Dense(MolDenseOp {
            src_field: get(d, "src")?.extract()?,
            dst_field: get(d, "dst")?.extract()?,
            axis: get(d, "axis")?.extract()?,
            region: region(&get(d, "region")?)?,
            src_lo: get(d, "src_lo")?.extract()?,
            src_len: get(d, "src_len")?.extract()?,
            matrix: read_f64s(&get(d, "matrix")?, "matrix")?,
        })),
        "kernel" => {
            let expr = get(d, "expr")?;
            let first: PyRef<PyExpr> = expr.extract()?;
            let pool = first.pool.clone_ref(py);
            drop(first);
            let syms: Vec<ExprId> = get(d, "syms")?
                .try_iter()?
                .map(|e| Ok(extract_expr_in(&e?, &pool, "method_of_lines", "a symbol")?.id))
                .collect::<PyResult<_>>()?;
            let kernel = compile(py, &PyList::new(py, [expr])?.into_any(), &syms, &pool)?;
            let mut inputs = Vec::new();
            for item in get(d, "inputs")?.try_iter()? {
                let item = item?;
                let t = item.cast::<PyTuple>()?;
                let tag: String = t.get_item(0)?.extract()?;
                inputs.push(match tag.as_str() {
                    "field" => MolInput::Field {
                        field: t.get_item(1)?.extract()?,
                        offset: t.get_item(2)?.extract()?,
                    },
                    "coef" => MolInput::Coef(t.get_item(1)?.extract()?),
                    "time" => MolInput::Time,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown input tag {other:?}"
                        )))
                    }
                });
            }
            let (tk, tf): (String, usize) = get(d, "target")?.extract()?;
            let target = match tk.as_str() {
                "rhs" => MolTarget::Rhs(tf),
                "fill" => MolTarget::Fill(tf),
                other => return Err(PyValueError::new_err(format!("unknown target {other:?}"))),
            };
            let (jac_kernel, jac_inputs) = match d.get_item("jac")? {
                Some(j) if !j.is_none() => {
                    let ji: Vec<usize> = get(d, "jac_inputs")?.extract()?;
                    if ji.is_empty() {
                        (None, vec![])
                    } else {
                        (Some(compile(py, &j, &syms, &pool)?), ji)
                    }
                }
                _ => (None, vec![]),
            };
            Ok(MolOp::Kernel(MolKernelOp {
                kernel,
                inputs,
                target,
                region: region(&get(d, "region")?)?,
                jac_kernel,
                jac_inputs,
            }))
        }
        other => Err(PyValueError::new_err(format!("unknown op kind {other:?}"))),
    }
}

/// A compiled method-of-lines program (private; see
/// `alkahest.experimental.MolSystem`).
#[pyclass(name = "_MolProgram", frozen)]
struct PyMolProgram {
    inner: Arc<MolProgram>,
}

#[pymethods]
impl PyMolProgram {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        shape: Vec<usize>,
        n_fields: usize,
        coefs: &Bound<'_, PyList>,
        state_fields: Vec<usize>,
        state_flats: Vec<usize>,
        algebraic: Vec<bool>,
        ops: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        let coefs: Vec<Vec<f64>> = coefs
            .iter()
            .map(|c| read_f64s(&c, "coefficient array"))
            .collect::<PyResult<_>>()?;
        let ops: Vec<MolOp> = ops
            .iter()
            .map(|o| parse_op(py, o.cast::<PyDict>()?))
            .collect::<PyResult<_>>()?;
        if state_fields.len() != state_flats.len() {
            return Err(PyValueError::new_err(
                "state field and index lists differ in length",
            ));
        }
        let states = state_fields.into_iter().zip(state_flats).collect();
        let p = MolProgram::new(shape, n_fields, coefs, states, algebraic, ops).map_err(mol_err)?;
        Ok(PyMolProgram { inner: Arc::new(p) })
    }

    #[getter]
    fn n_states(&self) -> usize {
        self.inner.n_states()
    }

    #[getter]
    fn n_ops(&self) -> usize {
        self.inner.n_ops()
    }

    /// `out[:] = f(t, y)`; `y` and `out` are float64 NumPy arrays.
    fn rhs_into(
        &self,
        py: Python<'_>,
        t: f64,
        y: &Bound<'_, PyAny>,
        out: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let yv = array_view(y, false, "y")?;
        let ov = array_view(out, true, "out")?;
        let n = self.inner.n_states();
        if yv.len() != n || ov.len() != n {
            return Err(PyValueError::new_err(format!(
                "y and out must have {n} elements, got {} and {}",
                yv.len(),
                ov.len()
            )));
        }
        let (ya, oa) = (yv.addr, ov.addr);
        if ya == oa {
            return Err(PyValueError::new_err("out must not alias y"));
        }
        let prog = self.inner.clone();
        py.detach(move || {
            // SAFETY: both buffers were validated (live, aligned, `n` float64
            // values, `out` writable and distinct from `y`); the caller keeps
            // the arrays alive across this call.
            let ys = unsafe { std::slice::from_raw_parts(ya as *const f64, n) };
            let os = unsafe { std::slice::from_raw_parts_mut(oa as *mut f64, n) };
            prog.rhs(t, ys, os)
        })
        .map_err(mol_err)
    }

    /// The Jacobian at `(t, y)` as `(indptr, indices, data)`: little-endian
    /// `uint64`, `uint64` and `float64` bytes (for `numpy.frombuffer`).
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let y = read_f64s(y, "y")?;
        let prog = self.inner.clone();
        let m = py.detach(move || prog.jacobian(t, &y)).map_err(mol_err)?;
        (
            usize_bytes(py, &m.indptr),
            usize_bytes(py, &m.indices),
            f64_bytes(py, &m.data),
        )
            .into_pyobject(py)
    }

    /// Every padded field array after the boundary fills, as `float64` bytes.
    fn fields<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let y = read_f64s(y, "y")?;
        let f = self.inner.fields(t, &y).map_err(mol_err)?;
        Ok(f.iter().map(|v| f64_bytes(py, v)).collect())
    }

    /// Integrate natively; returns `(t, y_flat, stats, band)`.
    #[pyo3(signature = (t0, y0, t_eval, method="bdf", rtol=1e-6, atol=1e-9, h0=None,
                        max_step=f64::INFINITY, max_steps=10_000_000, max_order=5))]
    #[allow(clippy::too_many_arguments)]
    fn integrate<'py>(
        &self,
        py: Python<'py>,
        t0: f64,
        y0: &Bound<'_, PyAny>,
        t_eval: &Bound<'_, PyAny>,
        method: &str,
        rtol: f64,
        atol: f64,
        h0: Option<f64>,
        max_step: f64,
        max_steps: usize,
        max_order: usize,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let method = match method {
            "rk45" => MolMethod::Rk45,
            "bdf" => MolMethod::Bdf,
            "euler" => MolMethod::Euler,
            "rk4" => MolMethod::Rk4,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown native method {other:?} (rk45, bdf, euler, rk4)"
                )))
            }
        };
        let y0 = read_f64s(y0, "y0")?;
        let t_eval = read_f64s(t_eval, "t_eval")?;
        let opts = MolIntegrateOptions {
            method,
            rtol,
            atol,
            h0,
            max_step,
            max_steps,
            max_order,
            ..Default::default()
        };
        let prog = self.inner.clone();
        let sol = py
            .detach(move || integrate(&prog, t0, &y0, &t_eval, &opts))
            .map_err(mol_err)?;
        let stats = PyDict::new(py);
        stats.set_item("steps", sol.stats.steps)?;
        stats.set_item("rejected", sol.stats.rejected)?;
        stats.set_item("rhs_evals", sol.stats.rhs_evals)?;
        stats.set_item("jac_evals", sol.stats.jac_evals)?;
        stats.set_item("factorizations", sol.stats.factorizations)?;
        stats.set_item("newton_iters", sol.stats.newton_iters)?;
        let band = sol.band;
        (sol.t, f64_bytes(py, &sol.y), stats, band).into_pyobject(py)
    }
}

/// Register the method-of-lines runtime on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMolProgram>()?;
    Ok(())
}
