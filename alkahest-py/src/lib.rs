use alkahest_core::{
    adjoint_system as core_adjoint_system,
    cad_lift as core_cad_lift,
    cad_project as core_cad_project,
    // Phase 21 — JIT
    compile as core_compile,
    decide_expr as core_decide_expr,
    emit_horner_c as core_emit_horner_c,
    emit_stablehlo as core_emit_stablehlo,
    eval_interp as core_eval_interp,
    factor_univariate_mod_p as core_factor_univariate_mod_p,
    grad as core_grad,
    guess_integer_relation as core_guess_integer_relation,
    // Phase 24 — Horner form
    horner as core_horner,
    jacobian as core_jacobian,
    lattice_reduce_rows as core_lattice_reduce_rows,
    lattice_reduce_rows_with_delta as core_lattice_reduce_rows_with_delta,
    lower_to_first_order as core_lower_to_first_order,
    pantelides as core_pantelides,
    // Phase 27 — poly_normal
    poly_normal as core_poly_normal,
    // V2-4 — Real root isolation
    real_roots_symbolic as core_real_roots_symbolic,
    refine_root as core_refine_root,
    resistor as core_resistor,
    // V2-2 — Resultants and subresultant PRS
    resultant as core_resultant,
    // V3-3 — FOFormula / satisfiability
    satisfiable as core_satisfiable,
    sensitivity_system as core_sensitivity_system,
    // V2-3 — Sparse interpolation
    sparse_interpolate as core_sparse_interpolate,
    sparse_interpolate_univariate as core_sparse_interpolate_univariate,
    subresultant_prs as core_subresultant_prs,
    solve_linear_recurrence_homogeneous as core_solve_linear_recurrence_homogeneous,
    subs as core_subs,
    sum_definite as core_sum_definite,
    sum_indefinite as core_sum_indefinite,
    verify_wz_pair as core_verify_wz_pair,
    WzPair,
    // Phase 22 — Ball arithmetic
    ArbBall as CoreArbBall,
    // V2-9 — CAD / real QE
    CadError,
    Capabilities,
    Domain,
    Event,
    ExprId,
    ExprPool,
    FactorError,
    HybridODE,
    IntervalEval as CoreIntervalEval,
    LatticeError,
    Matrix,
    MatrixError,
    MultiPoly,
    MultiPolyFactorization,
    OdeError,
    Pattern,
    Port,
    PrimitiveRegistry,
    PslqError,
    RationalFunction,
    RealRootError,
    RewriteRule,
    RootInterval as CoreRootInterval,
    Satisfiability as CoreSatisfiability,
    ScalarODE,
    System as AcausalSystem,
    UniPoly,
    UniPolyFactorModP,
    UniPolyFactorization,
    DAE,
    ODE,
};

#[cfg(feature = "cuda")]
use alkahest_core::compile_cuda as core_compile_cuda;
use alkahest_core::kernel::expr::PredicateKind;
// V2-1 — Modular / CRT framework
use alkahest_core::modular::{
    lift_crt as core_lift_crt, mignotte_bound as core_mignotte_bound,
    rational_reconstruction as core_rational_reconstruction, reduce_mod as core_reduce_mod,
    select_lucky_prime as core_select_lucky_prime, ModularError, MultiPolyFp,
};
use alkahest_core::{
    diff as core_diff, diff_forward as core_diff_forward, integrate as core_integrate, load_from,
    log_exp_rules, match_pattern as core_match_pattern, simplify as core_simplify,
    simplify_egraph as core_simplify_egraph, simplify_egraph_with as core_simplify_egraph_with,
    simplify_with as core_simplify_with, trig_rules, AlkahestError as AlkahestErrorTrait,
    DiffError, EgraphConfig, IntegrationError, IoError, LinearRecurrenceError, PatternRule,
    ResultantError, SimplifyConfig, SizeCost, SparseInterpError, SumError,
};
use rug::{Integer, Rational};
use pyo3::exceptions::{PyOverflowError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// V1-3: Structured Python exception hierarchy
// ---------------------------------------------------------------------------

// V1-3: structured exception hierarchy.
// Base inherits from ValueError for backward compat with existing `except ValueError` tests.
pyo3::create_exception!(alkahest, PyAlkahestError, pyo3::exceptions::PyValueError);
pyo3::create_exception!(alkahest, PyConversionError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyDomainError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyDiffError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyPoolError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyIntegrationError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyMatrixError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyModularError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyOdeError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyDaeError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyJitError, PyAlkahestError);
pyo3::create_exception!(alkahest, PySolverError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyCudaError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyIoError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyParseError, PyAlkahestError);
// V2-7 — Polynomial factorization
pyo3::create_exception!(alkahest, PyFactorError, PyAlkahestError);
// V2-2 — Resultants
pyo3::create_exception!(alkahest, PyResultantError, PyAlkahestError);
// V2-3 — Sparse interpolation
pyo3::create_exception!(alkahest, PySparseInterpError, PyAlkahestError);
// V2-4 — Real root isolation
pyo3::create_exception!(alkahest, PyRealRootError, PyAlkahestError);
// V2-6 — LLL + integer relations
pyo3::create_exception!(alkahest, PyLatticeError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyPslqError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyCadError, PyAlkahestError);
pyo3::create_exception!(alkahest, PySumError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyLinearRecurrenceError, PyAlkahestError);

/// Build a structured exception with `.code`, `.remediation`, `.span` attributes.
fn make_structured_err<E: AlkahestErrorTrait>(
    _py: Python<'_>,
    exc_type: &pyo3::Bound<'_, pyo3::types::PyType>,
    e: &E,
) -> PyErr {
    let msg = e.to_string();
    let remediation = e.remediation().unwrap_or("");
    let full_msg = if remediation.is_empty() {
        msg
    } else {
        format!("{msg}\nRemediation: {remediation}")
    };
    let exc = exc_type.call1((full_msg,)).unwrap();
    exc.setattr("code", e.code()).ok();
    exc.setattr("remediation", e.remediation()).ok();
    exc.setattr("span", e.span()).ok();
    PyErr::from_value_bound(exc)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_domain(s: &str) -> Domain {
    match s {
        "complex" => Domain::Complex,
        "integer" => Domain::Integer,
        "positive" => Domain::Positive,
        "nonneg" | "nonnegative" => Domain::NonNegative,
        "nonzero" => Domain::NonZero,
        _ => Domain::Real,
    }
}

fn diff_error_to_py(e: DiffError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyDiffError>();
        make_structured_err(py, &exc_type, &e)
    })
}

#[allow(dead_code)]
fn io_error_to_py(e: IoError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyIoError>();
        make_structured_err(py, &exc_type, &e)
    })
}

#[cfg(feature = "groebner-cuda")]
fn gpu_groebner_error_to_py(e: alkahest_core::experimental::GpuGroebnerError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PySolverError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn integrate_error_to_py(e: IntegrationError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyIntegrationError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn conv_error_to_py(e: alkahest_core::ConversionError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyConversionError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn factor_error_to_py(e: FactorError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyFactorError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn resultant_error_to_py(e: ResultantError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyResultantError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn sparse_interp_error_to_py(e: SparseInterpError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PySparseInterpError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn real_root_error_to_py(e: RealRootError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyRealRootError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn cad_error_to_py(e: CadError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyCadError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn sum_error_to_py(e: SumError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PySumError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn linear_recurrence_error_to_py(e: LinearRecurrenceError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyLinearRecurrenceError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn modular_error_to_py(e: ModularError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyModularError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn lattice_error_to_py(e: LatticeError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyLatticeError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn pslq_error_to_py(e: PslqError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyPslqError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn matrix_error_to_py(e: MatrixError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyMatrixError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn ode_error_to_py(e: OdeError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyOdeError>();
        make_structured_err(py, &exc_type, &e)
    })
}

// ---------------------------------------------------------------------------
// PyExprPool
// ---------------------------------------------------------------------------

#[pyclass(name = "ExprPool")]
struct PyExprPool {
    inner: ExprPool,
}

#[pymethods]
impl PyExprPool {
    #[new]
    fn new() -> Self {
        PyExprPool {
            inner: ExprPool::new(),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(&self, _exc_type: PyObject, _exc_val: PyObject, _exc_tb: PyObject) -> bool {
        false
    }

    fn symbol(slf: PyRef<'_, Self>, name: &str, domain: Option<&str>) -> PyExpr {
        let dom = parse_domain(domain.unwrap_or("real"));
        let id = slf.inner.symbol(name, dom);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    fn integer(slf: PyRef<'_, Self>, n: i64) -> PyExpr {
        let id = slf.inner.integer(n);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    fn rational(slf: PyRef<'_, Self>, p: i64, q: i64) -> PyExpr {
        let id = slf.inner.rational(p, q);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    fn float(slf: PyRef<'_, Self>, value: f64, prec: Option<u32>) -> PyExpr {
        let id = slf.inner.float(value, prec.unwrap_or(53));
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    // PA-9 — Predicate constructors
    fn lt(slf: PyRef<'_, Self>, a: PyExpr, b: PyExpr) -> PyExpr {
        let id = slf.inner.pred_lt(a.id, b.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn le(slf: PyRef<'_, Self>, a: PyExpr, b: PyExpr) -> PyExpr {
        let id = slf.inner.pred_le(a.id, b.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn gt(slf: PyRef<'_, Self>, a: PyExpr, b: PyExpr) -> PyExpr {
        let id = slf.inner.pred_gt(a.id, b.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn ge(slf: PyRef<'_, Self>, a: PyExpr, b: PyExpr) -> PyExpr {
        let id = slf.inner.pred_ge(a.id, b.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn pred_eq(slf: PyRef<'_, Self>, a: PyExpr, b: PyExpr) -> PyExpr {
        let id = slf.inner.pred_eq(a.id, b.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn pred_ne(slf: PyRef<'_, Self>, a: PyExpr, b: PyExpr) -> PyExpr {
        let id = slf.inner.pred_ne(a.id, b.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn pred_and(slf: PyRef<'_, Self>, args: Vec<PyExpr>) -> PyExpr {
        let ids: Vec<ExprId> = args.iter().map(|e| e.id).collect();
        let id = slf.inner.pred_and(ids);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn pred_or(slf: PyRef<'_, Self>, args: Vec<PyExpr>) -> PyExpr {
        let ids: Vec<ExprId> = args.iter().map(|e| e.id).collect();
        let id = slf.inner.pred_or(ids);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn pred_not(slf: PyRef<'_, Self>, a: PyExpr) -> PyExpr {
        let id = slf.inner.pred_not(a.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn pred_true(slf: PyRef<'_, Self>) -> PyExpr {
        let id = slf.inner.pred_true();
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }
    fn pred_false(slf: PyRef<'_, Self>) -> PyExpr {
        let id = slf.inner.pred_false();
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    /// Universal quantifier: ``∀ var . body`` (first-order logic).
    fn forall(slf: PyRef<'_, Self>, var: PyExpr, body: PyExpr) -> PyExpr {
        let id = slf.inner.forall(var.id, body.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    /// Existential quantifier: ``∃ var . body``.
    fn exists(slf: PyRef<'_, Self>, var: PyExpr, body: PyExpr) -> PyExpr {
        let id = slf.inner.exists(var.id, body.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    // V1-16: ExprPool persistence bindings
    /// Write the pool to `path` atomically (temp + rename).  Raises `IoError`
    /// on any filesystem failure.
    fn save_to(&self, path: &str) -> PyResult<()> {
        self.inner.checkpoint(path).map_err(io_error_to_py)
    }

    /// Load a persisted pool from `path`.  Returns a new `ExprPool`.
    /// Raises `FileNotFoundError` if `path` does not exist, `IoError` for
    /// other failures.
    #[staticmethod]
    fn load_from(path: &str) -> PyResult<PyExprPool> {
        match load_from(path) {
            Ok(Some(inner)) => Ok(PyExprPool { inner }),
            Ok(None) => Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "pool file not found: {path}"
            ))),
            Err(e) => Err(io_error_to_py(e)),
        }
    }
}

// ---------------------------------------------------------------------------
// PyExpr
// ---------------------------------------------------------------------------

#[pyclass(name = "Expr")]
#[derive(Clone)]
struct PyExpr {
    id: alkahest_core::ExprId,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyExpr {
    fn __eq__(&self, other: PyRef<PyExpr>) -> bool {
        self.id == other.id
    }

    fn __hash__(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.id.hash(&mut h);
        h.finish()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        self.pool.borrow(py).inner.display(self.id).to_string()
    }

    fn __str__(&self, py: Python<'_>) -> String {
        self.pool.borrow(py).inner.display(self.id).to_string()
    }

    // ------------------------------------------------------------------
    // Arithmetic — accept Expr, int, or float on the right-hand side.
    // Return py.NotImplemented() for unrecognised types so Python can
    // try the reflected operation on the other operand.
    // ------------------------------------------------------------------

    fn __add__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyObject {
        match self.coerce_scalar(other, py) {
            Some(rhs) => {
                let id = self.pool.borrow(py).inner.add(vec![self.id, rhs]);
                PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            }
            None => py.NotImplemented(),
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyObject {
        match self.coerce_scalar(other, py) {
            Some(lhs) => {
                let id = self.pool.borrow(py).inner.add(vec![lhs, self.id]);
                PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            }
            None => py.NotImplemented(),
        }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyObject {
        match self.coerce_scalar(other, py) {
            Some(rhs) => {
                let pool = self.pool.borrow(py);
                let neg_one = pool.inner.integer(-1i32);
                let neg_rhs = pool.inner.mul(vec![neg_one, rhs]);
                let id = pool.inner.add(vec![self.id, neg_rhs]);
                drop(pool);
                PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            }
            None => py.NotImplemented(),
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyObject {
        // other - self
        match self.coerce_scalar(other, py) {
            Some(lhs) => {
                let pool = self.pool.borrow(py);
                let neg_one = pool.inner.integer(-1i32);
                let neg_self = pool.inner.mul(vec![neg_one, self.id]);
                let id = pool.inner.add(vec![lhs, neg_self]);
                drop(pool);
                PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            }
            None => py.NotImplemented(),
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyObject {
        match self.coerce_scalar(other, py) {
            Some(rhs) => {
                let id = self.pool.borrow(py).inner.mul(vec![self.id, rhs]);
                PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            }
            None => py.NotImplemented(),
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyObject {
        match self.coerce_scalar(other, py) {
            Some(lhs) => {
                let id = self.pool.borrow(py).inner.mul(vec![lhs, self.id]);
                PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            }
            None => py.NotImplemented(),
        }
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyObject {
        match self.coerce_scalar(other, py) {
            Some(rhs) => {
                let pool = self.pool.borrow(py);
                let neg_one = pool.inner.integer(-1i32);
                let inv = pool.inner.pow(rhs, neg_one);
                let id = pool.inner.mul(vec![self.id, inv]);
                drop(pool);
                PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            }
            None => py.NotImplemented(),
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyObject {
        // other / self  =  other * self^-1
        match self.coerce_scalar(other, py) {
            Some(lhs) => {
                let pool = self.pool.borrow(py);
                let neg_one = pool.inner.integer(-1i32);
                let inv_self = pool.inner.pow(self.id, neg_one);
                let id = pool.inner.mul(vec![lhs, inv_self]);
                drop(pool);
                PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            }
            None => py.NotImplemented(),
        }
    }

    fn __neg__(&self, py: Python<'_>) -> PyExpr {
        let pool = self.pool.borrow(py);
        let neg_one = pool.inner.integer(-1i32);
        let id = pool.inner.mul(vec![neg_one, self.id]);
        drop(pool);
        PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        }
    }

    fn __pow__(&self, exp: i64, _modulo: Option<PyObject>, py: Python<'_>) -> PyExpr {
        let pool = self.pool.borrow(py);
        let exp_id = pool.inner.integer(exp);
        let id = pool.inner.pow(self.id, exp_id);
        drop(pool);
        PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        }
    }

    fn pow_expr(&self, exp: &PyExpr, py: Python<'_>) -> PyExpr {
        let pool = self.pool.borrow(py);
        let id = pool.inner.pow(self.id, exp.id);
        drop(pool);
        PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        }
    }

    fn node_tag(&self, py: Python<'_>) -> String {
        let data = self.pool.borrow(py).inner.get(self.id);
        match data {
            alkahest_core::ExprData::Symbol { .. } => "symbol".to_string(),
            alkahest_core::ExprData::Integer(_) => "integer".to_string(),
            _ => "other".to_string(),
        }
    }

    // V2-20: expose expression tree structure for pure-Python pretty-printing.
    //
    // Returns a Python list [tag, arg...] describing this node:
    //   ["symbol",    name: str]
    //   ["integer",   value: str]
    //   ["rational",  numer: str, denom: str]
    //   ["float",     value: str]
    //   ["add",       [child: Expr, ...]]
    //   ["mul",       [child: Expr, ...]]
    //   ["pow",       base: Expr, exp: Expr]
    //   ["func",      name: str, [arg: Expr, ...]]
    //   ["piecewise", [[cond: Expr, val: Expr], ...], default: Expr]
    //   ["predicate", kind: str, [arg: Expr, ...]]
    fn node(&self, py: Python<'_>) -> PyObject {
        let data = {
            let pool = self.pool.borrow(py);
            pool.inner.get(self.id)
        };

        macro_rules! wrap {
            ($id:expr) => {
                PyExpr {
                    id: $id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py)
            };
        }

        macro_rules! ids_to_pylist {
            ($ids:expr) => {{
                let items: Vec<PyObject> = $ids.iter().map(|&id| wrap!(id)).collect();
                PyList::new_bound(py, items).into_py(py)
            }};
        }

        match data {
            alkahest_core::ExprData::Symbol { name, .. } => {
                PyList::new_bound(py, vec!["symbol".into_py(py), name.into_py(py)]).into_py(py)
            }
            alkahest_core::ExprData::Integer(n) => {
                PyList::new_bound(py, vec!["integer".into_py(py), n.0.to_string().into_py(py)])
                    .into_py(py)
            }
            alkahest_core::ExprData::Rational(r) => PyList::new_bound(
                py,
                vec![
                    "rational".into_py(py),
                    r.0.numer().to_string().into_py(py),
                    r.0.denom().to_string().into_py(py),
                ],
            )
            .into_py(py),
            alkahest_core::ExprData::Float(f) => PyList::new_bound(
                py,
                vec!["float".into_py(py), f.inner.to_string().into_py(py)],
            )
            .into_py(py),
            alkahest_core::ExprData::Add(args) => {
                PyList::new_bound(py, vec!["add".into_py(py), ids_to_pylist!(args)]).into_py(py)
            }
            alkahest_core::ExprData::Mul(args) => {
                PyList::new_bound(py, vec!["mul".into_py(py), ids_to_pylist!(args)]).into_py(py)
            }
            alkahest_core::ExprData::Pow { base, exp } => {
                PyList::new_bound(py, vec!["pow".into_py(py), wrap!(base), wrap!(exp)]).into_py(py)
            }
            alkahest_core::ExprData::Func { name, args } => PyList::new_bound(
                py,
                vec!["func".into_py(py), name.into_py(py), ids_to_pylist!(args)],
            )
            .into_py(py),
            alkahest_core::ExprData::Piecewise { branches, default } => {
                let br_items: Vec<PyObject> = branches
                    .iter()
                    .map(|&(cond, val)| {
                        PyTuple::new_bound(py, vec![wrap!(cond), wrap!(val)]).into_py(py)
                    })
                    .collect();
                PyList::new_bound(
                    py,
                    vec![
                        "piecewise".into_py(py),
                        PyList::new_bound(py, br_items).into_py(py),
                        wrap!(default),
                    ],
                )
                .into_py(py)
            }
            alkahest_core::ExprData::Predicate { kind, args } => {
                let kind_str = match kind {
                    PredicateKind::Lt => "lt",
                    PredicateKind::Le => "le",
                    PredicateKind::Gt => "gt",
                    PredicateKind::Ge => "ge",
                    PredicateKind::Eq => "eq",
                    PredicateKind::Ne => "ne",
                    PredicateKind::And => "and",
                    PredicateKind::Or => "or",
                    PredicateKind::Not => "not",
                    PredicateKind::True => "true",
                    PredicateKind::False => "false",
                };
                PyList::new_bound(
                    py,
                    vec![
                        "predicate".into_py(py),
                        kind_str.into_py(py),
                        ids_to_pylist!(args),
                    ],
                )
                .into_py(py)
            }
            alkahest_core::ExprData::Forall { var, body } => {
                PyList::new_bound(py, vec!["forall".into_py(py), wrap!(var), wrap!(body)])
                    .into_py(py)
            }
            alkahest_core::ExprData::Exists { var, body } => {
                PyList::new_bound(py, vec!["exists".into_py(py), wrap!(var), wrap!(body)])
                    .into_py(py)
            }
        }
    }
}

// Non-pymethod helpers for PyExpr.
impl PyExpr {
    // Coerce a Python scalar (Expr | int | float) to an interned ExprId.
    // Returns None for unrecognised types so callers can return NotImplemented.
    fn coerce_scalar(&self, ob: &Bound<'_, PyAny>, py: Python<'_>) -> Option<ExprId> {
        if let Ok(e) = ob.extract::<PyRef<PyExpr>>() {
            return Some(e.id);
        }
        let pool = self.pool.borrow(py);
        if let Ok(n) = ob.extract::<i64>() {
            return Some(pool.inner.integer(n));
        }
        if let Ok(f) = ob.extract::<f64>() {
            return Some(pool.inner.float(f, 53));
        }
        None
    }
}

// ---------------------------------------------------------------------------
// PyDerivedResult
// ---------------------------------------------------------------------------

#[pyclass(name = "DerivedResult")]
struct PyDerivedResult {
    value: PyExpr,
    derivation: String,
    steps_raw: Vec<(String, String, String, Vec<String>)>,
}

#[pymethods]
impl PyDerivedResult {
    #[getter]
    fn value(&self) -> PyExpr {
        self.value.clone()
    }

    #[getter]
    fn derivation(&self) -> &str {
        &self.derivation
    }

    #[getter]
    fn steps<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let list = PyList::empty_bound(py);
        for (rule, before, after, conds) in &self.steps_raw {
            let d = PyDict::new_bound(py);
            d.set_item("rule", rule).unwrap();
            d.set_item("before", before).unwrap();
            d.set_item("after", after).unwrap();
            d.set_item("side_conditions", conds).unwrap();
            list.append(d).unwrap();
        }
        list
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("DerivedResult(value={})", self.value.__repr__(py))
    }
}

fn make_derived_result(
    py: Python<'_>,
    derived: alkahest_core::DerivedExpr<alkahest_core::ExprId>,
    pool_py: Py<PyExprPool>,
) -> PyDerivedResult {
    let derivation = {
        let pool = pool_py.borrow(py);
        derived.log.display_with(&pool.inner).to_string()
    };
    let steps_raw: Vec<_> = {
        let pool = pool_py.borrow(py);
        derived
            .log
            .steps()
            .iter()
            .map(|step| {
                let before_str = pool.inner.display(step.before).to_string();
                let after_str = pool.inner.display(step.after).to_string();
                let conds: Vec<String> = step
                    .side_conditions
                    .iter()
                    .map(|c| c.display_with(&pool.inner).to_string())
                    .collect();
                (step.rule_name.to_string(), before_str, after_str, conds)
            })
            .collect()
    };
    let value = PyExpr {
        id: derived.value,
        pool: pool_py,
    };
    PyDerivedResult {
        value,
        derivation,
        steps_raw,
    }
}

// ---------------------------------------------------------------------------
// Module-level functions: named math functions
// ---------------------------------------------------------------------------

fn make_func(py: Python<'_>, name: &str, expr: PyRef<PyExpr>) -> PyExpr {
    let id = expr.pool.borrow(py).inner.func(name, vec![expr.id]);
    let pool = expr.pool.clone_ref(py);
    PyExpr { id, pool }
}

#[pyfunction]
fn sin(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "sin", expr)
}

#[pyfunction]
fn cos(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "cos", expr)
}

#[pyfunction]
fn exp(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "exp", expr)
}

#[pyfunction]
fn log(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "log", expr)
}

#[pyfunction]
fn sqrt(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "sqrt", expr)
}

// V1-12: expanded primitive registry
#[pyfunction]
fn tan(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "tan", expr)
}

#[pyfunction]
fn sinh(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "sinh", expr)
}

#[pyfunction]
fn cosh(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "cosh", expr)
}

#[pyfunction]
fn tanh(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "tanh", expr)
}

#[pyfunction]
fn asin(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "asin", expr)
}

#[pyfunction]
fn acos(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "acos", expr)
}

#[pyfunction]
fn atan(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "atan", expr)
}

#[pyfunction]
fn erf(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "erf", expr)
}

#[pyfunction]
fn erfc(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "erfc", expr)
}

#[pyfunction]
#[pyo3(name = "abs")]
fn abs_expr(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "abs", expr)
}

#[pyfunction]
fn sign(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "sign", expr)
}

#[pyfunction]
fn floor(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "floor", expr)
}

#[pyfunction]
fn ceil(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "ceil", expr)
}

#[pyfunction]
#[pyo3(name = "round")]
fn round_expr(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "round", expr)
}

#[pyfunction]
fn gamma(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "gamma", expr)
}

fn make_binary_func(py: Python<'_>, name: &str, a: PyRef<PyExpr>, b: PyRef<PyExpr>) -> PyExpr {
    let id = a.pool.borrow(py).inner.func(name, vec![a.id, b.id]);
    let pool = a.pool.clone_ref(py);
    PyExpr { id, pool }
}

#[pyfunction]
fn atan2(py: Python<'_>, y: PyRef<PyExpr>, x: PyRef<PyExpr>) -> PyExpr {
    make_binary_func(py, "atan2", y, x)
}

#[pyfunction]
#[pyo3(name = "min")]
fn min_expr(py: Python<'_>, a: PyRef<PyExpr>, b: PyRef<PyExpr>) -> PyExpr {
    make_binary_func(py, "min", a, b)
}

#[pyfunction]
#[pyo3(name = "max")]
fn max_expr(py: Python<'_>, a: PyRef<PyExpr>, b: PyRef<PyExpr>) -> PyExpr {
    make_binary_func(py, "max", a, b)
}

// ---------------------------------------------------------------------------
// Module-level: simplify and diff
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "simplify")]
fn py_simplify(py: Python<'_>, expr: PyRef<PyExpr>) -> PyDerivedResult {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_simplify(expr.id, &pool.inner)
    };
    let pool_py = expr.pool.clone_ref(py);
    make_derived_result(py, derived, pool_py)
}

/// Python-visible configuration for the e-graph simplifier.
///
/// All arguments are keyword-only with the same defaults as the Rust `EgraphConfig`.
#[pyclass(name = "EgraphConfig")]
#[derive(Clone)]
struct PyEgraphConfig {
    inner: EgraphConfig,
}

#[pymethods]
impl PyEgraphConfig {
    #[new]
    #[pyo3(signature = (
        shrink_iters = 5,
        explore_iters = 3,
        const_fold_iters = 3,
        node_limit = None,
        iter_limit = None,
        include_trig_rules = true,
        include_log_exp_rules = true,
    ))]
    fn new(
        shrink_iters: usize,
        explore_iters: usize,
        const_fold_iters: usize,
        node_limit: Option<usize>,
        iter_limit: Option<usize>,
        include_trig_rules: bool,
        include_log_exp_rules: bool,
    ) -> Self {
        PyEgraphConfig {
            inner: EgraphConfig {
                shrink_iters,
                explore_iters,
                const_fold_iters,
                node_limit,
                iter_limit,
                include_trig_rules,
                include_log_exp_rules,
            },
        }
    }

    #[getter]
    fn shrink_iters(&self) -> usize {
        self.inner.shrink_iters
    }

    #[getter]
    fn explore_iters(&self) -> usize {
        self.inner.explore_iters
    }

    #[getter]
    fn const_fold_iters(&self) -> usize {
        self.inner.const_fold_iters
    }

    #[getter]
    fn node_limit(&self) -> Option<usize> {
        self.inner.node_limit
    }

    #[getter]
    fn iter_limit(&self) -> Option<usize> {
        self.inner.iter_limit
    }

    #[getter]
    fn include_trig_rules(&self) -> bool {
        self.inner.include_trig_rules
    }

    #[getter]
    fn include_log_exp_rules(&self) -> bool {
        self.inner.include_log_exp_rules
    }
}

#[pyfunction]
#[pyo3(name = "simplify_egraph")]
fn py_simplify_egraph(py: Python<'_>, expr: PyRef<PyExpr>) -> PyDerivedResult {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_simplify_egraph(expr.id, &pool.inner)
    };
    let pool_py = expr.pool.clone_ref(py);
    make_derived_result(py, derived, pool_py)
}

/// Simplify using the e-graph backend with a custom [`EgraphConfig`].
///
/// Use this when you want to disable specific rule sets (e.g. trig or log/exp
/// rules) or tune the phase iteration counts.
#[pyfunction]
#[pyo3(name = "simplify_egraph_with")]
fn py_simplify_egraph_with(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    config: PyRef<PyEgraphConfig>,
) -> PyDerivedResult {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_simplify_egraph_with(expr.id, &pool.inner, &config.inner, &SizeCost)
    };
    let pool_py = expr.pool.clone_ref(py);
    make_derived_result(py, derived, pool_py)
}

#[pyfunction]
#[pyo3(name = "diff")]
fn py_diff(py: Python<'_>, expr: PyRef<PyExpr>, var: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_diff(expr.id, var.id, &pool.inner).map_err(diff_error_to_py)?
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py))
}

#[pyfunction]
#[pyo3(name = "diff_forward")]
fn py_diff_forward(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_diff_forward(expr.id, var.id, &pool.inner).map_err(diff_error_to_py)?
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py))
}

// ---------------------------------------------------------------------------
// V2-7 — Forward declarations (factorization result types reference these)
// ---------------------------------------------------------------------------

#[pyclass(name = "UniPoly")]
struct PyUniPoly {
    inner: UniPoly,
}

#[pyclass(name = "MultiPoly")]
struct PyMultiPoly {
    inner: MultiPoly,
}

// ---------------------------------------------------------------------------
// V2-7 — Factorization result types
// ---------------------------------------------------------------------------

#[pyclass(name = "UniPolyFactorization")]
struct PyUniPolyFactorization {
    inner: UniPolyFactorization,
}

#[pymethods]
impl PyUniPolyFactorization {
    #[getter]
    fn unit(&self) -> String {
        self.inner.unit.to_string()
    }

    fn factor_list(&self) -> Vec<(PyUniPoly, u32)> {
        self.inner
            .factors
            .iter()
            .map(|(p, e)| (PyUniPoly { inner: p.clone() }, *e))
            .collect()
    }
}

#[pyclass(name = "MultiPolyFactorization")]
struct PyMultiPolyFactorization {
    inner: MultiPolyFactorization,
}

#[pymethods]
impl PyMultiPolyFactorization {
    #[getter]
    fn unit(&self) -> String {
        self.inner.unit.to_string()
    }

    fn factor_list(&self) -> Vec<(PyMultiPoly, u32)> {
        self.inner
            .factors
            .iter()
            .map(|(p, e)| (PyMultiPoly { inner: p.clone() }, *e))
            .collect()
    }
}

#[pyclass(name = "UniPolyFactorModP")]
struct PyUniPolyFactorModP {
    inner: UniPolyFactorModP,
}

#[pymethods]
impl PyUniPolyFactorModP {
    #[getter]
    fn modulus(&self) -> u64 {
        self.inner.modulus
    }

    fn factor_list(&self) -> Vec<(Vec<u64>, u32)> {
        self.inner.factors.clone()
    }
}

// ---------------------------------------------------------------------------
// PyUniPoly
// ---------------------------------------------------------------------------

#[pymethods]
impl PyUniPoly {
    #[staticmethod]
    fn from_symbolic(py: Python<'_>, expr: PyRef<PyExpr>, var: PyRef<PyExpr>) -> PyResult<Self> {
        let pool = expr.pool.borrow(py);
        UniPoly::from_symbolic(expr.id, var.id, &pool.inner)
            .map(|p| PyUniPoly { inner: p })
            .map_err(conv_error_to_py)
    }

    fn coefficients(&self) -> Vec<i64> {
        self.inner.coefficients_i64()
    }

    fn degree(&self) -> i64 {
        self.inner.degree()
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn __add__(&self, other: PyRef<PyUniPoly>) -> PyUniPoly {
        PyUniPoly {
            inner: &self.inner + &other.inner,
        }
    }

    fn __sub__(&self, other: PyRef<PyUniPoly>) -> PyUniPoly {
        PyUniPoly {
            inner: &self.inner - &other.inner,
        }
    }

    fn __mul__(&self, other: PyRef<PyUniPoly>) -> PyUniPoly {
        PyUniPoly {
            inner: &self.inner * &other.inner,
        }
    }

    fn __pow__(&self, exp: u32, _modulo: Option<PyObject>) -> PyUniPoly {
        PyUniPoly {
            inner: self.inner.pow(exp),
        }
    }

    fn gcd(&self, other: PyRef<PyUniPoly>) -> PyResult<PyUniPoly> {
        self.inner
            .gcd(&other.inner)
            .map(|p| PyUniPoly { inner: p })
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("gcd: variable mismatch"))
    }

    /// Factor over ℤ (FLINT).
    fn factor_z(&self) -> PyResult<PyUniPolyFactorization> {
        self.inner
            .factor_z()
            .map(|inner| PyUniPolyFactorization { inner })
            .map_err(factor_error_to_py)
    }

    fn __repr__(&self) -> String {
        format!("UniPoly({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// PyMultiPoly
// ---------------------------------------------------------------------------

#[pymethods]
impl PyMultiPoly {
    #[staticmethod]
    fn from_symbolic(
        py: Python<'_>,
        expr: PyRef<PyExpr>,
        vars: Vec<PyRef<PyExpr>>,
    ) -> PyResult<Self> {
        let var_ids: Vec<_> = vars.iter().map(|v| v.id).collect();
        let pool = expr.pool.borrow(py);
        MultiPoly::from_symbolic(expr.id, var_ids, &pool.inner)
            .map(|p| PyMultiPoly { inner: p })
            .map_err(conv_error_to_py)
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn total_degree(&self) -> u32 {
        self.inner.total_degree()
    }

    fn integer_content(&self) -> String {
        self.inner.integer_content().to_string()
    }

    fn __add__(&self, other: PyRef<PyMultiPoly>) -> PyResult<PyMultiPoly> {
        if !self.inner.compatible_with(&other.inner) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "MultiPoly arithmetic requires matching variable lists",
            ));
        }
        Ok(PyMultiPoly {
            inner: self.inner.clone() + other.inner.clone(),
        })
    }

    fn __sub__(&self, other: PyRef<PyMultiPoly>) -> PyResult<PyMultiPoly> {
        if !self.inner.compatible_with(&other.inner) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "MultiPoly arithmetic requires matching variable lists",
            ));
        }
        Ok(PyMultiPoly {
            inner: self.inner.clone() - other.inner.clone(),
        })
    }

    fn __mul__(&self, other: PyRef<PyMultiPoly>) -> PyResult<PyMultiPoly> {
        if !self.inner.compatible_with(&other.inner) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "MultiPoly arithmetic requires matching variable lists",
            ));
        }
        Ok(PyMultiPoly {
            inner: self.inner.clone() * other.inner.clone(),
        })
    }

    /// Factor over ℤ (multivariate FLINT).
    fn factor_z(&self) -> PyResult<PyMultiPolyFactorization> {
        self.inner
            .factor_z()
            .map(|inner| PyMultiPolyFactorization { inner })
            .map_err(factor_error_to_py)
    }

    fn __repr__(&self) -> String {
        format!("MultiPoly({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// PyRationalFunction
// ---------------------------------------------------------------------------

#[pyclass(name = "RationalFunction")]
struct PyRationalFunction {
    inner: RationalFunction,
}

#[pymethods]
impl PyRationalFunction {
    #[staticmethod]
    fn from_symbolic(
        py: Python<'_>,
        numer: PyRef<PyExpr>,
        denom: PyRef<PyExpr>,
        vars: Vec<PyRef<PyExpr>>,
    ) -> PyResult<Self> {
        let var_ids: Vec<_> = vars.iter().map(|v| v.id).collect();
        let pool = numer.pool.borrow(py);
        RationalFunction::from_symbolic(numer.id, denom.id, var_ids, &pool.inner)
            .map(|r| PyRationalFunction { inner: r })
            .map_err(conv_error_to_py)
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn numer(&self) -> PyMultiPoly {
        PyMultiPoly {
            inner: self.inner.numer.clone(),
        }
    }

    fn denom(&self) -> PyMultiPoly {
        PyMultiPoly {
            inner: self.inner.denom.clone(),
        }
    }

    fn __add__(&self, other: PyRef<PyRationalFunction>) -> PyResult<PyRationalFunction> {
        (self.inner.clone() + other.inner.clone())
            .map(|r| PyRationalFunction { inner: r })
            .map_err(conv_error_to_py)
    }

    fn __sub__(&self, other: PyRef<PyRationalFunction>) -> PyResult<PyRationalFunction> {
        (self.inner.clone() - other.inner.clone())
            .map(|r| PyRationalFunction { inner: r })
            .map_err(conv_error_to_py)
    }

    fn __mul__(&self, other: PyRef<PyRationalFunction>) -> PyResult<PyRationalFunction> {
        (self.inner.clone() * other.inner.clone())
            .map(|r| PyRationalFunction { inner: r })
            .map_err(conv_error_to_py)
    }

    fn __truediv__(&self, other: PyRef<PyRationalFunction>) -> PyResult<PyRationalFunction> {
        (self.inner.clone() / other.inner.clone())
            .map(|r| PyRationalFunction { inner: r })
            .map_err(conv_error_to_py)
    }

    fn __neg__(&self) -> PyRationalFunction {
        PyRationalFunction {
            inner: -self.inner.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("RationalFunction({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "integrate")]
fn py_integrate(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_integrate(expr.id, var.id, &pool.inner).map_err(integrate_error_to_py)?
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py))
}

#[pyfunction]
#[pyo3(name = "sum_indefinite")]
fn py_sum_indefinite(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_sum_indefinite(expr.id, k.id, &pool.inner).map_err(sum_error_to_py)?
    };
    Ok(make_derived_result(py, derived, expr.pool.clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "sum_definite")]
fn py_sum_definite(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
    lo: PyRef<PyExpr>,
    hi: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_sum_definite(expr.id, k.id, lo.id, hi.id, &pool.inner).map_err(sum_error_to_py)?
    };
    Ok(make_derived_result(py, derived, expr.pool.clone_ref(py)))
}

#[pyfunction]
#[pyo3(name = "solve_linear_recurrence_homogeneous")]
fn py_solve_linear_recurrence_homogeneous(
    py: Python<'_>,
    n: PyRef<PyExpr>,
    coeffs: Vec<(i64, i64)>,
    initials: Vec<PyRef<PyExpr>>,
) -> PyResult<PyExpr> {
    let rat_coeffs: Vec<Rational> = coeffs
        .into_iter()
        .map(|(a, b)| Rational::from((Integer::from(a), Integer::from(b))))
        .collect();
    let init_ids: Vec<ExprId> = initials.iter().map(|e| e.id).collect();
    let pool_py = n.pool.clone_ref(py);
    let closed = {
        let pool = pool_py.borrow(py);
        core_solve_linear_recurrence_homogeneous(&pool.inner, n.id, &rat_coeffs, &init_ids)
            .map_err(linear_recurrence_error_to_py)?
            .closed_form
    };
    Ok(PyExpr {
        id: closed,
        pool: pool_py,
    })
}

#[pyfunction]
#[pyo3(name = "verify_wz_pair")]
fn py_verify_wz_pair(
    py: Python<'_>,
    f: PyRef<PyExpr>,
    g: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
) -> PyResult<bool> {
    let _ = py;
    let pool = f.pool.borrow(py);
    let pair = WzPair { f: f.id, g: g.id };
    Ok(core_verify_wz_pair(&pair, n.id, k.id, &pool.inner))
}

/// `alkahest.match_pattern(pattern_expr, expr) -> list[dict[str, Expr]]`
///
/// Find all AC-aware matches of `pattern_expr` anywhere in `expr`.
/// Each match is returned as a dict mapping wildcard names to matched
/// sub-expressions.  Wildcards are symbols whose names start with a
/// lower-case letter.
#[pyfunction]
fn match_pattern(py: Python<'_>, pattern_expr: PyRef<PyExpr>, expr: PyRef<PyExpr>) -> PyObject {
    let pool_py = pattern_expr.pool.clone_ref(py);
    let matches = {
        let pool = pool_py.borrow(py);
        let pat = Pattern::from_expr(pattern_expr.id);
        core_match_pattern(&pat, expr.id, &pool.inner)
    };
    let out = PyList::empty_bound(py);
    for subst in matches {
        let d = PyDict::new_bound(py);
        for (name, id) in subst.bindings {
            let expr_py = PyExpr {
                id,
                pool: pool_py.clone_ref(py),
            };
            d.set_item(name, expr_py.into_py(py)).unwrap();
        }
        out.append(d).unwrap();
    }
    out.into_py(py)
}

// ---------------------------------------------------------------------------
// PyRewriteRule — wrapper for user-defined pattern rules (R-5)
// ---------------------------------------------------------------------------

#[pyclass(name = "RewriteRule")]
struct PyRewriteRule {
    inner: PatternRule,
}

/// `alkahest.make_rule(lhs, rhs)` — create a rewrite rule from two expressions.
///
/// Symbols whose names start with a lower-case letter in `lhs` are wildcards
/// that bind to any sub-expression; the same names in `rhs` are replaced by
/// the bound values.
///
/// Example::
///
///     rule = alkahest.make_rule(a*b + a*c, a*(b + c))  # factoring rule
///     result = alkahest.simplify_with(expr, [rule])
#[pyfunction]
fn make_rule(py: Python<'_>, lhs: PyRef<PyExpr>, rhs: PyRef<PyExpr>) -> PyRewriteRule {
    let pool = lhs.pool.borrow(py);
    let _ = pool; // borrow released below
    drop(pool);
    let lhs_id = lhs.id;
    let rhs_id = rhs.id;
    PyRewriteRule {
        inner: PatternRule::new(Pattern::from_expr(lhs_id), rhs_id),
    }
}

/// `alkahest.simplify_with(expr, rules)` — simplify using a custom rule list.
///
/// `rules` is a list of `RewriteRule` objects (e.g. from `make_rule`).
/// The default arithmetic rules are NOT included; combine with
/// `alkahest.default_rules()` if needed.
#[pyfunction]
#[pyo3(name = "simplify_with")]
fn py_simplify_with(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    rules: Vec<PyRef<PyRewriteRule>>,
) -> PyDerivedResult {
    // We can't easily collect into `Vec<Box<dyn RewriteRule>>` due to trait
    // object lifetime constraints from PyO3, so we re-implement the engine loop.
    // Build a list of lhs/rhs pairs and apply PatternRule inline.
    let pool_py = expr.pool.clone_ref(py);
    let lhs_rhs: Vec<(ExprId, ExprId)> = rules
        .iter()
        .map(|r| (r.inner.lhs.root, r.inner.rhs))
        .collect();

    let derived = {
        let pool = pool_py.borrow(py);
        // Build boxed rules list
        let boxed: Vec<Box<dyn RewriteRule>> = lhs_rhs
            .into_iter()
            .map(|(lhs, rhs)| {
                Box::new(PatternRule::new(Pattern::from_expr(lhs), rhs)) as Box<dyn RewriteRule>
            })
            .collect();
        core_simplify_with(expr.id, &pool.inner, &boxed, SimplifyConfig::default())
    };
    make_derived_result(py, derived, pool_py)
}

/// `alkahest.simplify_expanded(expr)` — simplify with distributive expansion.
///
/// Applies `(a + b) * c → a*c + b*c` in addition to all default rules.
#[pyfunction]
#[pyo3(name = "simplify_expanded")]
fn py_simplify_expanded(py: Python<'_>, expr: PyRef<PyExpr>) -> PyDerivedResult {
    let derived = {
        let pool = expr.pool.borrow(py);
        alkahest_core::simplify_expanded(expr.id, &pool.inner)
    };
    let pool_py = expr.pool.clone_ref(py);
    make_derived_result(py, derived, pool_py)
}

/// `alkahest.simplify_trig(expr)` — simplify with trigonometric identities.
#[pyfunction]
#[pyo3(name = "simplify_trig")]
fn py_simplify_trig(py: Python<'_>, expr: PyRef<PyExpr>) -> PyDerivedResult {
    let derived = {
        let pool = expr.pool.borrow(py);
        let rules = trig_rules();
        core_simplify_with(expr.id, &pool.inner, &rules, SimplifyConfig::default())
    };
    let pool_py = expr.pool.clone_ref(py);
    make_derived_result(py, derived, pool_py)
}

/// `alkahest.simplify_log_exp(expr)` — simplify with log/exp identities.
#[pyfunction]
#[pyo3(name = "simplify_log_exp")]
fn py_simplify_log_exp(py: Python<'_>, expr: PyRef<PyExpr>) -> PyDerivedResult {
    let derived = {
        let pool = expr.pool.borrow(py);
        let rules = log_exp_rules();
        core_simplify_with(expr.id, &pool.inner, &rules, SimplifyConfig::default())
    };
    let pool_py = expr.pool.clone_ref(py);
    make_derived_result(py, derived, pool_py)
}

// ---------------------------------------------------------------------------
// V5-1 — Lean 4 certificate exporter
// ---------------------------------------------------------------------------

/// `alkahest.to_lean(expr) -> str`
///
/// Simplify `expr` and generate a Lean 4 proof certificate from the
/// resulting derivation log.  Returns a string containing the complete
/// `.lean` source file (Mathlib imports + one example per rewrite step).
#[pyfunction]
#[pyo3(name = "to_lean")]
fn py_to_lean(py: Python<'_>, expr: PyRef<PyExpr>) -> String {
    let pool_py = expr.pool.clone_ref(py);
    let derived = {
        let pool = pool_py.borrow(py);
        core_simplify(expr.id, &pool.inner)
    };
    let pool = pool_py.borrow(py);
    alkahest_core::emit_lean(&derived, &pool.inner)
}

// ---------------------------------------------------------------------------
// subs — substitution primitive (R-6)
// ---------------------------------------------------------------------------

/// `alkahest.subs(expr, mapping)` — replace sub-expressions.
///
/// `mapping` is a dict mapping `Expr` → `Expr`.  Every sub-expression of
/// `expr` that appears as a key is replaced by the corresponding value.
///
/// Example::
///
///     result = alkahest.subs(expr, {x: pool.integer(3)})
#[pyfunction]
#[pyo3(name = "subs")]
fn py_subs(py: Python<'_>, expr: PyRef<PyExpr>, mapping: &Bound<'_, PyDict>) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let mut map: HashMap<ExprId, ExprId> = HashMap::new();
    for (k, v) in mapping.iter() {
        let key_expr: PyRef<PyExpr> = k.extract()?;
        let val_expr: PyRef<PyExpr> = v.extract()?;
        map.insert(key_expr.id, val_expr.id);
    }
    let result_id = {
        let pool = pool_py.borrow(py);
        core_subs(expr.id, &map, &pool.inner)
    };
    Ok(PyExpr {
        id: result_id,
        pool: pool_py,
    })
}

/// Returns the alkahest-core version string.
#[pyfunction]
fn version() -> &'static str {
    alkahest_core::version()
}

// ---------------------------------------------------------------------------
// Phase 14: grad — reverse-mode AD
// ---------------------------------------------------------------------------

/// `alkahest.grad(expr, vars)` — compute all partial derivatives at once.
///
/// Returns a list of `Expr` objects `[∂expr/∂vars[0], ∂expr/∂vars[1], …]`.
///
/// Uses reverse-mode (adjoint) accumulation: O(DAG size) regardless of
/// the number of variables, vs. O(#vars × DAG size) for repeated `diff`.
#[pyfunction]
#[pyo3(name = "grad")]
fn py_grad(py: Python<'_>, expr: PyRef<PyExpr>, vars: Vec<PyRef<PyExpr>>) -> Vec<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let grads = {
        let pool = pool_py.borrow(py);
        core_grad(expr.id, &var_ids, &pool.inner)
    };
    grads
        .into_iter()
        .map(|id| PyExpr {
            id,
            pool: pool_py.clone_ref(py),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 15: Matrix and jacobian
// ---------------------------------------------------------------------------

#[pyclass(name = "Matrix")]
struct PyMatrix {
    inner: Matrix,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyMatrix {
    // Allow Matrix([[expr, expr], [expr, expr]]) in addition to from_rows.
    #[new]
    fn __new__(py: Python<'_>, rows: Vec<Vec<PyRef<PyExpr>>>) -> PyResult<PyMatrix> {
        PyMatrix::from_rows(py, rows)
    }

    #[staticmethod]
    fn from_rows(py: Python<'_>, rows: Vec<Vec<PyRef<PyExpr>>>) -> PyResult<PyMatrix> {
        if rows.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Matrix must have at least one row",
            ));
        }
        let pool_py: Py<PyExprPool> = rows[0][0].pool.clone_ref(py);
        let data: Vec<Vec<ExprId>> = rows
            .iter()
            .map(|row| row.iter().map(|e| e.id).collect())
            .collect();
        let m = Matrix::new(data).map_err(matrix_error_to_py)?;
        Ok(PyMatrix {
            inner: m,
            pool: pool_py,
        })
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }

    /// Shape of the matrix as a ``(rows, cols)`` tuple.
    fn shape(&self) -> (usize, usize) {
        (self.inner.rows, self.inner.cols)
    }

    /// Sum of diagonal entries (trace).
    fn trace(&self, py: Python<'_>) -> PyResult<PyExpr> {
        if self.inner.rows != self.inner.cols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "trace() requires a square matrix",
            ));
        }
        let pool = self.pool.borrow(py);
        let diag: Vec<ExprId> = (0..self.inner.rows).map(|i| self.inner.get(i, i)).collect();
        let id = pool.inner.add(diag);
        drop(pool);
        Ok(PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        })
    }

    fn get(&self, py: Python<'_>, r: usize, c: usize) -> PyExpr {
        PyExpr {
            id: self.inner.get(r, c),
            pool: self.pool.clone_ref(py),
        }
    }

    fn transpose(&self, py: Python<'_>) -> PyMatrix {
        PyMatrix {
            inner: self.inner.transpose(),
            pool: self.pool.clone_ref(py),
        }
    }

    fn __add__(&self, py: Python<'_>, other: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
        let pool = self.pool.borrow(py);
        let m = self
            .inner
            .add(&other.inner, &pool.inner)
            .map_err(matrix_error_to_py)?;
        drop(pool);
        Ok(PyMatrix {
            inner: m,
            pool: self.pool.clone_ref(py),
        })
    }

    fn __sub__(&self, py: Python<'_>, other: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
        let pool = self.pool.borrow(py);
        let m = self
            .inner
            .sub(&other.inner, &pool.inner)
            .map_err(matrix_error_to_py)?;
        drop(pool);
        Ok(PyMatrix {
            inner: m,
            pool: self.pool.clone_ref(py),
        })
    }

    fn __matmul__(&self, py: Python<'_>, other: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
        let pool = self.pool.borrow(py);
        let m = self
            .inner
            .mul(&other.inner, &pool.inner)
            .map_err(matrix_error_to_py)?;
        drop(pool);
        Ok(PyMatrix {
            inner: m,
            pool: self.pool.clone_ref(py),
        })
    }

    fn det(&self, py: Python<'_>) -> PyResult<PyExpr> {
        let pool = self.pool.borrow(py);
        let d = self.inner.det(&pool.inner).map_err(matrix_error_to_py)?;
        drop(pool);
        Ok(PyExpr {
            id: d,
            pool: self.pool.clone_ref(py),
        })
    }

    fn simplify(&self, py: Python<'_>) -> PyMatrix {
        let pool = self.pool.borrow(py);
        let m = self.inner.simplify_entries(&pool.inner);
        drop(pool);
        PyMatrix {
            inner: m,
            pool: self.pool.clone_ref(py),
        }
    }

    fn to_list(&self, py: Python<'_>) -> Vec<Vec<PyExpr>> {
        (0..self.inner.rows)
            .map(|r| {
                (0..self.inner.cols)
                    .map(|c| PyExpr {
                        id: self.inner.get(r, c),
                        pool: self.pool.clone_ref(py),
                    })
                    .collect()
            })
            .collect()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        self.inner.display(&pool.inner)
    }
}

/// `alkahest.jacobian(f_vec, x_vec)` — compute the Jacobian matrix.
#[pyfunction]
#[pyo3(name = "jacobian")]
fn py_jacobian(
    py: Python<'_>,
    f_vec: Vec<PyRef<PyExpr>>,
    x_vec: Vec<PyRef<PyExpr>>,
) -> PyResult<PyMatrix> {
    if f_vec.is_empty() || x_vec.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "f_vec and x_vec must be non-empty",
        ));
    }
    let pool_py = f_vec[0].pool.clone_ref(py);
    let f_ids: Vec<ExprId> = f_vec.iter().map(|e| e.id).collect();
    let x_ids: Vec<ExprId> = x_vec.iter().map(|e| e.id).collect();
    let m = {
        let pool = pool_py.borrow(py);
        core_jacobian(&f_ids, &x_ids, &pool.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    Ok(PyMatrix {
        inner: m,
        pool: pool_py,
    })
}

// ---------------------------------------------------------------------------
// Phase 16: ODE
// ---------------------------------------------------------------------------

#[pyclass(name = "ODE")]
struct PyODE {
    inner: ODE,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyODE {
    // Allow ODE([state_vars], [rhs], time_var) directly in addition to ODE.new(...)
    #[new]
    fn __new__(
        py: Python<'_>,
        state_vars: Vec<PyRef<PyExpr>>,
        rhs: Vec<PyRef<PyExpr>>,
        time_var: PyRef<PyExpr>,
    ) -> PyResult<PyODE> {
        PyODE::new(py, state_vars, rhs, time_var)
    }

    #[staticmethod]
    fn new(
        py: Python<'_>,
        state_vars: Vec<PyRef<PyExpr>>,
        rhs: Vec<PyRef<PyExpr>>,
        time_var: PyRef<PyExpr>,
    ) -> PyResult<PyODE> {
        let pool_py = time_var.pool.clone_ref(py);
        let state_ids: Vec<ExprId> = state_vars.iter().map(|e| e.id).collect();
        let rhs_ids: Vec<ExprId> = rhs.iter().map(|e| e.id).collect();
        let ode = {
            let pool = pool_py.borrow(py);
            ODE::new(state_ids, rhs_ids, time_var.id, &pool.inner).map_err(ode_error_to_py)?
        };
        Ok(PyODE {
            inner: ode,
            pool: pool_py,
        })
    }

    fn with_ic(&self, py: Python<'_>, var: PyRef<PyExpr>, value: PyRef<PyExpr>) -> PyODE {
        PyODE {
            inner: self.inner.clone().with_ic(var.id, value.id),
            pool: self.pool.clone_ref(py),
        }
    }

    fn order(&self) -> usize {
        self.inner.order()
    }

    fn is_autonomous(&self, py: Python<'_>) -> bool {
        let pool = self.pool.borrow(py);
        self.inner.is_autonomous(&pool.inner)
    }

    fn state_vars(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.inner
            .state_vars
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    fn rhs(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.inner
            .rhs
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    fn derivatives(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.inner
            .derivatives
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    fn simplify_rhs(&self, py: Python<'_>) -> PyODE {
        let pool = self.pool.borrow(py);
        let new_ode = self.inner.simplify_rhs(&pool.inner);
        drop(pool);
        PyODE {
            inner: new_ode,
            pool: self.pool.clone_ref(py),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        format!("ODE(\n{}\n)", self.inner.display(&pool.inner))
    }
}

/// `alkahest.lower_to_first_order(var, rhs, order, time_var)` — lower a scalar ODE to first-order.
#[pyfunction]
#[pyo3(name = "lower_to_first_order")]
fn py_lower_to_first_order(
    py: Python<'_>,
    var: PyRef<PyExpr>,
    rhs: PyRef<PyExpr>,
    order: usize,
    time_var: PyRef<PyExpr>,
) -> PyResult<PyODE> {
    let pool_py = var.pool.clone_ref(py);
    let scalar = ScalarODE {
        var: var.id,
        aux_vars: vec![],
        rhs: rhs.id,
        time_var: time_var.id,
        order,
    };
    let ode = {
        let pool = pool_py.borrow(py);
        core_lower_to_first_order(&scalar, &pool.inner).map_err(ode_error_to_py)?
    };
    Ok(PyODE {
        inner: ode,
        pool: pool_py,
    })
}

// ---------------------------------------------------------------------------
// Phase 19: Sensitivity analysis
// ---------------------------------------------------------------------------

#[pyclass(name = "SensitivitySystem")]
struct PySensitivitySystem {
    ode: PyODE,
    original_dim: usize,
    n_params: usize,
}

#[pymethods]
impl PySensitivitySystem {
    #[getter]
    fn extended_ode(&self, py: Python<'_>) -> PyODE {
        PyODE {
            inner: self.ode.inner.clone(),
            pool: self.ode.pool.clone_ref(py),
        }
    }

    #[getter]
    fn original_dim(&self) -> usize {
        self.original_dim
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.n_params
    }

    fn __repr__(&self) -> String {
        format!(
            "SensitivitySystem(dim={}, n_params={})",
            self.original_dim, self.n_params
        )
    }
}

/// `alkahest.sensitivity_system(ode, params)` — build the forward sensitivity ODE.
#[pyfunction]
#[pyo3(name = "sensitivity_system")]
fn py_sensitivity_system(
    py: Python<'_>,
    ode: PyRef<PyODE>,
    params: Vec<PyRef<PyExpr>>,
) -> PyResult<PySensitivitySystem> {
    let pool_py = ode.pool.clone_ref(py);
    let param_ids: Vec<ExprId> = params.iter().map(|e| e.id).collect();
    let sys = {
        let pool = pool_py.borrow(py);
        core_sensitivity_system(&ode.inner, &param_ids, &pool.inner).map_err(ode_error_to_py)?
    };
    Ok(PySensitivitySystem {
        ode: PyODE {
            inner: sys.extended_ode,
            pool: pool_py,
        },
        original_dim: sys.original_dim,
        n_params: sys.n_params,
    })
}

/// `alkahest.adjoint_system(ode, objective_grad)` — build the adjoint ODE.
#[pyfunction]
#[pyo3(name = "adjoint_system")]
fn py_adjoint_system(
    py: Python<'_>,
    ode: PyRef<PyODE>,
    objective_grad: Vec<PyRef<PyExpr>>,
) -> PyResult<PyODE> {
    let pool_py = ode.pool.clone_ref(py);
    let grad_ids: Vec<ExprId> = objective_grad.iter().map(|e| e.id).collect();
    let adj = {
        let pool = pool_py.borrow(py);
        core_adjoint_system(&ode.inner, &grad_ids, &pool.inner).map_err(ode_error_to_py)?
    };
    Ok(PyODE {
        inner: adj.adjoint_ode,
        pool: pool_py,
    })
}

// ---------------------------------------------------------------------------
// Phase 17: DAE
// ---------------------------------------------------------------------------

#[pyclass(name = "DAE")]
struct PyDAE {
    inner: DAE,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyDAE {
    #[staticmethod]
    fn new(
        py: Python<'_>,
        equations: Vec<PyRef<PyExpr>>,
        variables: Vec<PyRef<PyExpr>>,
        derivatives: Vec<PyRef<PyExpr>>,
        time_var: PyRef<PyExpr>,
    ) -> PyDAE {
        let pool_py = time_var.pool.clone_ref(py);
        let eq_ids: Vec<ExprId> = equations.iter().map(|e| e.id).collect();
        let var_ids: Vec<ExprId> = variables.iter().map(|e| e.id).collect();
        let deriv_ids: Vec<ExprId> = derivatives.iter().map(|e| e.id).collect();
        let dae = DAE::new(eq_ids, var_ids, deriv_ids, time_var.id);
        PyDAE {
            inner: dae,
            pool: pool_py,
        }
    }

    fn n_equations(&self) -> usize {
        self.inner.n_equations()
    }
    fn n_variables(&self) -> usize {
        self.inner.n_variables()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        format!("DAE(\n{}\n)", self.inner.display(&pool.inner))
    }
}

/// `alkahest.pantelides(dae)` — apply the Pantelides index-reduction algorithm.
#[pyfunction]
#[pyo3(name = "pantelides")]
fn py_pantelides(py: Python<'_>, dae: PyRef<PyDAE>) -> PyResult<PyDAE> {
    let pool_py = dae.pool.clone_ref(py);
    let result = {
        let pool = pool_py.borrow(py);
        core_pantelides(&dae.inner, &pool.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    Ok(PyDAE {
        inner: result.reduced_dae,
        pool: pool_py,
    })
}

// ---------------------------------------------------------------------------
// Phase 20: Hybrid ODE / Events
// ---------------------------------------------------------------------------

#[pyclass(name = "Event")]
struct PyEvent {
    inner: Event,
}

#[pymethods]
impl PyEvent {
    #[staticmethod]
    fn new(
        _py: Python<'_>,
        name: &str,
        condition: PyRef<PyExpr>,
        reset_map: Vec<(PyRef<PyExpr>, PyRef<PyExpr>)>,
    ) -> PyEvent {
        let reset: Vec<(ExprId, ExprId)> = reset_map.iter().map(|(v, e)| (v.id, e.id)).collect();
        PyEvent {
            inner: Event::new(name, condition.id, reset),
        }
    }

    fn rising(mut slf: PyRefMut<'_, Self>) {
        slf.inner.direction = 1;
    }

    fn falling(mut slf: PyRefMut<'_, Self>) {
        slf.inner.direction = -1;
    }

    fn __repr__(&self) -> String {
        format!(
            "Event(name='{}', direction={})",
            self.inner.name, self.inner.direction
        )
    }
}

#[pyclass(name = "HybridODE")]
struct PyHybridODE {
    inner: HybridODE,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyHybridODE {
    #[staticmethod]
    fn new(py: Python<'_>, ode: PyRef<PyODE>) -> PyHybridODE {
        let pool_py = ode.pool.clone_ref(py);
        PyHybridODE {
            inner: HybridODE::new(ode.inner.clone()),
            pool: pool_py,
        }
    }

    fn add_event(&self, py: Python<'_>, event: PyRef<PyEvent>) -> PyHybridODE {
        let new_inner = self.inner.clone().add_event(event.inner.clone());
        PyHybridODE {
            inner: new_inner,
            pool: self.pool.clone_ref(py),
        }
    }

    fn n_events(&self) -> usize {
        self.inner.events.len()
    }

    fn guards(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.inner
            .guards()
            .into_iter()
            .map(|id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        self.inner.display(&pool.inner)
    }
}

// ---------------------------------------------------------------------------
// Phase 18: Acausal components
// ---------------------------------------------------------------------------

#[pyclass(name = "Port")]
#[derive(Clone)]
struct PyPort {
    inner: Port,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyPort {
    #[getter]
    fn potential(&self, py: Python<'_>) -> PyExpr {
        PyExpr {
            id: self.inner.potential,
            pool: self.pool.clone_ref(py),
        }
    }

    #[getter]
    fn flow(&self, py: Python<'_>) -> PyExpr {
        PyExpr {
            id: self.inner.flow,
            pool: self.pool.clone_ref(py),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
}

#[pyclass(name = "AcausalSystem")]
struct PyAcausalSystem {
    inner: AcausalSystem,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyAcausalSystem {
    #[new]
    fn new(_py: Python<'_>, pool: PyRef<PyExprPool>) -> PyAcausalSystem {
        let pool_py: Py<PyExprPool> = pool.into();
        PyAcausalSystem {
            inner: AcausalSystem::new(),
            pool: pool_py,
        }
    }

    fn flatten(&self, py: Python<'_>, time_var: PyRef<PyExpr>) -> PyDAE {
        let pool = self.pool.borrow(py);
        let dae = self.inner.flatten(time_var.id, &pool.inner);
        drop(pool);
        PyDAE {
            inner: dae,
            pool: self.pool.clone_ref(py),
        }
    }
}

/// `alkahest.resistor(name, resistance)` — create a resistor component.
#[pyfunction]
#[pyo3(name = "resistor")]
fn py_resistor(py: Python<'_>, name: &str, resistance: PyRef<PyExpr>) -> PyResult<PyObject> {
    // Return as a Python dict for simplicity
    let pool = resistance.pool.borrow(py);
    let comp = core_resistor(name, resistance.id, &pool.inner);
    drop(pool);
    // Pack as dict: {"name": name, "n_equations": N, "n_ports": M}
    let d = PyDict::new_bound(py);
    d.set_item("name", comp.name.clone())?;
    d.set_item("n_equations", comp.equations.len())?;
    d.set_item("n_ports", comp.ports.len())?;
    Ok(d.into_py(py))
}

// ---------------------------------------------------------------------------
// Phase 21 — JIT compiled evaluation
// ---------------------------------------------------------------------------

/// Compile a symbolic expression to a fast native function.
///
/// Returns a callable Python object (PyCompiledFn).
#[pyfunction]
#[pyo3(name = "compile_expr")]
fn py_compile_expr(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    inputs: &Bound<'_, PyList>,
) -> PyResult<PyCompiledFn> {
    let pool = expr.pool.borrow(py);
    let input_ids: Vec<ExprId> = inputs
        .iter()
        .map(|item| {
            let e: PyRef<PyExpr> = item.extract()?;
            Ok(e.id)
        })
        .collect::<PyResult<_>>()?;
    drop(pool);

    let pool_ref = expr.pool.borrow(py);
    let compiled = core_compile(expr.id, &input_ids, &pool_ref.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    drop(pool_ref);

    Ok(PyCompiledFn {
        snapshot: compiled,
        n_inputs: input_ids.len(),
    })
}

/// Evaluate a symbolic expression numerically using the interpreter.
///
/// `bindings` is a dict mapping Expr → float.
#[pyfunction]
#[pyo3(name = "eval_expr")]
fn py_eval_expr(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    bindings: &Bound<'_, PyDict>,
) -> PyResult<f64> {
    let pool = expr.pool.borrow(py);
    let mut env = std::collections::HashMap::new();
    for (key, value) in bindings.iter() {
        let var: PyRef<PyExpr> = key.extract()?;
        let val: f64 = value.extract()?;
        env.insert(var.id, val);
    }
    let result = core_eval_interp(expr.id, &env, &pool.inner).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "expression could not be evaluated (unbound variable or unsupported node)",
        )
    })?;
    Ok(result)
}

#[pyclass(name = "CompiledFn", unsendable)]
struct PyCompiledFn {
    snapshot: alkahest_core::CompiledFn,
    n_inputs: usize,
}

#[pymethods]
impl PyCompiledFn {
    /// Call the compiled function with a list of float inputs.
    fn __call__(&self, inputs: Vec<f64>) -> PyResult<f64> {
        if inputs.len() != self.n_inputs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.n_inputs,
                inputs.len()
            )));
        }
        Ok(self.snapshot.call(&inputs))
    }

    #[getter]
    fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    fn __repr__(&self) -> String {
        format!("<CompiledFn n_inputs={}>", self.n_inputs)
    }

    /// Batch-evaluate over N points (Phase 25 — NumPy/JAX array evaluation).
    ///
    /// `inputs_flat` is a flat list of length `n_vars * n_points` laid out
    /// var-major: `[x0[0], x0[1], …, x1[0], x1[1], …]`.
    ///
    /// Returns a flat list of N outputs.  In Python this is wrapped by the
    /// `alkahest.numpy_eval` helper that handles the buffer-protocol conversion.
    fn call_batch_raw(
        &self,
        inputs_flat: Vec<f64>,
        n_vars: usize,
        n_points: usize,
    ) -> PyResult<Vec<f64>> {
        if inputs_flat.len() != n_vars * n_points {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "inputs_flat length {} != n_vars({}) * n_points({})",
                inputs_flat.len(),
                n_vars,
                n_points
            )));
        }
        if n_vars != self.n_inputs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} variables, got {}",
                self.n_inputs, n_vars
            )));
        }
        let cols: Vec<&[f64]> = (0..n_vars)
            .map(|i| &inputs_flat[i * n_points..(i + 1) * n_points])
            .collect();
        let mut output = vec![0.0f64; n_points];
        self.snapshot.call_batch(&cols, &mut output);
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Phase 22 — Ball arithmetic
// ---------------------------------------------------------------------------

#[pyclass(name = "ArbBall")]
#[derive(Clone)]
struct PyArbBall {
    inner: CoreArbBall,
}

#[pymethods]
impl PyArbBall {
    /// Create a real ball `[mid ± rad]`.
    #[new]
    #[pyo3(signature = (mid, rad=0.0, prec=128))]
    fn new(mid: f64, rad: f64, prec: u32) -> Self {
        PyArbBall {
            inner: CoreArbBall::from_midpoint_radius(mid, rad, prec),
        }
    }

    #[getter]
    fn mid(&self) -> f64 {
        self.inner.mid_f64()
    }

    #[getter]
    fn rad(&self) -> f64 {
        self.inner.rad_f64()
    }

    #[getter]
    fn lo(&self) -> f64 {
        self.inner.lo().to_f64()
    }

    #[getter]
    fn hi(&self) -> f64 {
        self.inner.hi().to_f64()
    }

    fn contains(&self, v: f64) -> bool {
        self.inner.contains(v)
    }

    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    fn __add__(&self, other: &PyArbBall) -> PyArbBall {
        PyArbBall {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }

    fn __sub__(&self, other: &PyArbBall) -> PyArbBall {
        PyArbBall {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    fn __mul__(&self, other: &PyArbBall) -> PyArbBall {
        PyArbBall {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }

    fn __truediv__(&self, other: &PyArbBall) -> PyResult<PyArbBall> {
        (self.inner.clone() / other.inner.clone())
            .map(|b| PyArbBall { inner: b })
            .ok_or_else(|| {
                pyo3::exceptions::PyZeroDivisionError::new_err("division by a ball containing zero")
            })
    }

    fn __neg__(&self) -> PyArbBall {
        PyArbBall {
            inner: -self.inner.clone(),
        }
    }

    fn sin(&self) -> PyArbBall {
        PyArbBall {
            inner: self.inner.sin(),
        }
    }
    fn cos(&self) -> PyArbBall {
        PyArbBall {
            inner: self.inner.cos(),
        }
    }
    fn exp(&self) -> PyArbBall {
        PyArbBall {
            inner: self.inner.exp(),
        }
    }

    fn log(&self) -> PyResult<PyArbBall> {
        self.inner
            .log()
            .map(|b| PyArbBall { inner: b })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "log undefined for a ball containing non-positive values",
                )
            })
    }

    fn sqrt(&self) -> PyResult<PyArbBall> {
        self.inner
            .sqrt()
            .map(|b| PyArbBall { inner: b })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "sqrt undefined for a ball containing negative values",
                )
            })
    }

    fn __repr__(&self) -> String {
        format!(
            "ArbBall({:.6} ± {:.2e})",
            self.inner.mid_f64(),
            self.inner.rad_f64()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Evaluate a symbolic expression using rigorous interval (ball) arithmetic.
///
/// `bindings` is a dict mapping Expr → ArbBall.
#[pyfunction]
#[pyo3(name = "interval_eval")]
fn py_interval_eval(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    bindings: &Bound<'_, PyDict>,
    prec: Option<u32>,
) -> PyResult<PyArbBall> {
    let prec = prec.unwrap_or(128);
    let pool = expr.pool.borrow(py);
    let mut eval = CoreIntervalEval::new(prec);
    for (key, value) in bindings.iter() {
        let var: PyRef<PyExpr> = key.extract()?;
        let ball: PyRef<PyArbBall> = value.extract()?;
        eval.bind(var.id, ball.inner.clone());
    }
    let result = eval.eval(expr.id, &pool.inner).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "expression could not be evaluated with ball arithmetic",
        )
    })?;
    Ok(PyArbBall { inner: result })
}

// ---------------------------------------------------------------------------
// Phase 23 — Parallel simplification
// ---------------------------------------------------------------------------

/// Simplify an expression using parallel bottom-up traversal (requires
/// the `parallel` feature to be enabled at build time for speedup;
/// falls back to sequential otherwise).
#[pyfunction]
#[pyo3(name = "simplify_par")]
fn py_simplify_par(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let pool_ref = expr.pool.borrow(py);
    #[cfg(feature = "parallel")]
    let result = alkahest_core::simplify_par(expr.id, &pool_ref.inner);
    #[cfg(not(feature = "parallel"))]
    let result = alkahest_core::simplify(expr.id, &pool_ref.inner);
    drop(pool_ref);
    Ok(make_derived_result(py, result, expr.pool.clone_ref(py)))
}

// ---------------------------------------------------------------------------
// Phase 24 — Horner-form code emission
// ---------------------------------------------------------------------------

/// Convert a polynomial expression to Horner form.
///
/// `expr` must be a univariate polynomial in `var`.
/// Returns a new `Expr` in Horner form `a₀ + x*(a₁ + x*(…))`.
#[pyfunction]
#[pyo3(name = "horner")]
fn py_horner(py: Python<'_>, expr: PyRef<PyExpr>, var: PyRef<PyExpr>) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let result = {
        let pool = pool_py.borrow(py);
        core_horner(expr.id, var.id, &pool.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    Ok(PyExpr {
        id: result,
        pool: pool_py,
    })
}

/// Emit a C function that evaluates the Horner form of a polynomial.
///
/// Returns a `str` containing a complete C function definition.
///
/// Parameters
/// ----------
/// expr : Expr
///     A univariate polynomial in `var`.
/// var : Expr
///     The polynomial variable.
/// var_name : str
///     The C variable name (default ``"x"``).
/// fn_name : str
///     The C function name (default ``"eval_poly"``).
#[pyfunction]
#[pyo3(name = "emit_c")]
#[pyo3(signature = (expr, var, var_name="x", fn_name="eval_poly"))]
fn py_emit_c(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
    var_name: &str,
    fn_name: &str,
) -> PyResult<String> {
    let pool = expr.pool.borrow(py);
    core_emit_horner_c(expr.id, var.id, var_name, fn_name, &pool.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// Phase 25 — NumPy / batch evaluation: call_batch_raw is merged into PyCompiledFn above.

// ---------------------------------------------------------------------------
// Phase 26 — collect_like_terms
// ---------------------------------------------------------------------------

/// Collect like terms in an expression: `2*x + 3*x → 5*x`.
///
/// Runs the `SubSelf` (collect_add_terms) and `ConstFold` rewrite rules
/// on `expr`.  This is a post-expansion pass — call after
/// `simplify_expanded` if you want full polynomial simplification.
#[pyfunction]
#[pyo3(name = "collect_like_terms")]
fn py_collect_like_terms(py: Python<'_>, expr: PyRef<PyExpr>) -> PyDerivedResult {
    use alkahest_core::{rules_for_config, simplify_with};
    let pool_py = expr.pool.clone_ref(py);
    let derived = {
        let pool = pool_py.borrow(py);
        let rules = rules_for_config(&SimplifyConfig::default());
        simplify_with(expr.id, &pool.inner, &rules, SimplifyConfig::default())
    };
    make_derived_result(py, derived, pool_py)
}

// ---------------------------------------------------------------------------
// Phase 27 — poly_normal
// ---------------------------------------------------------------------------

/// Normalize a polynomial expression to canonical sum-of-products form.
///
/// Converts `expr` to a [`MultiPoly`] (expanding all products, collecting
/// like terms) then converts back to a symbolic expression.  The result is
/// in sorted monomial order.
///
/// Returns `ValueError` if `expr` is not a polynomial in `vars`.
///
/// Example::
///
///     poly_normal((x+1)*(x-1), [x])  →  Expr for x² - 1
#[pyfunction]
#[pyo3(name = "poly_normal")]
fn py_poly_normal(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    vars: Vec<PyRef<PyExpr>>,
) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let result = {
        let pool = pool_py.borrow(py);
        core_poly_normal(expr.id, var_ids, &pool.inner).map_err(conv_error_to_py)?
    };
    Ok(PyExpr {
        id: result,
        pool: pool_py,
    })
}

// ---------------------------------------------------------------------------
// V2-2 — Resultants and subresultant PRS
// ---------------------------------------------------------------------------

/// Compute the resultant of two polynomial expressions with respect to a
/// variable.
///
/// Both ``p`` and ``q`` must be polynomial expressions with integer
/// coefficients.  The returned expression is:
///
/// - An integer constant in the **univariate** case (only ``var`` appears).
/// - A polynomial in the remaining variables in the **multivariate** case
///   (``var`` has been eliminated).
///
/// The returned :class:`DerivedResult` carries a ``"Resultant"`` derivation
/// step tagged with the Lean 4 theorem
/// ``Polynomial.resultant_eq_zero_iff_common_root``.
///
/// Raises :class:`ResultantError` if either input is not a polynomial with
/// integer coefficients.
///
/// Example::
///
///     pool = ExprPool()
///     x = pool.symbol("x")
///     y = pool.symbol("y")
///     p = x**2 + y**2 - pool.integer(1)
///     q = y - x
///     r = resultant(p, q, y)
///     # r.value == 2*x^2 - 1
#[pyfunction]
#[pyo3(name = "resultant")]
fn py_resultant(
    py: Python<'_>,
    p: PyRef<PyExpr>,
    q: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let pool_py = p.pool.clone_ref(py);
    let derived = {
        let pool = pool_py.borrow(py);
        core_resultant(p.id, q.id, var.id, &pool.inner).map_err(resultant_error_to_py)?
    };
    Ok(make_derived_result(py, derived, pool_py))
}

/// Compute the subresultant polynomial remainder sequence of two univariate
/// polynomials with integer coefficients.
///
/// Returns a Python ``list`` of :class:`Expr` objects ordered
/// ``[p, q, S₂, S₃, …, Sₖ]``.
///
/// Both polynomials must be **univariate** in ``var`` with integer
/// coefficients.  Multivariate inputs raise :class:`ResultantError`.
///
/// Example::
///
///     pool = ExprPool()
///     x = pool.symbol("x")
///     p = x**2 - pool.integer(1)
///     q = x - pool.integer(1)
///     prs = subresultant_prs(p, q, x)
///     # prs == [p, q, last_element]
#[pyfunction]
#[pyo3(name = "subresultant_prs")]
fn py_subresultant_prs(
    py: Python<'_>,
    p: PyRef<PyExpr>,
    q: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
) -> PyResult<Vec<PyExpr>> {
    let pool_py = p.pool.clone_ref(py);
    let derived = {
        let pool = pool_py.borrow(py);
        core_subresultant_prs(p.id, q.id, var.id, &pool.inner).map_err(resultant_error_to_py)?
    };

    let py_exprs: Vec<PyExpr> = derived
        .value
        .into_iter()
        .map(|id| PyExpr {
            id,
            pool: pool_py.clone_ref(py),
        })
        .collect();
    Ok(py_exprs)
}

// ---------------------------------------------------------------------------
// V2-4 — Real root isolation Python bindings
// ---------------------------------------------------------------------------

/// A closed rational interval `[lo, hi]` isolating exactly one real root.
///
/// For an exact rational root `r`, ``lo == hi == r``.
#[pyclass(name = "RootInterval", module = "alkahest")]
struct PyRootInterval {
    inner: CoreRootInterval,
}

#[pymethods]
impl PyRootInterval {
    /// Lower bound as a float (may be slightly inexact).
    #[getter]
    fn lo(&self) -> f64 {
        self.inner.lo_f64()
    }

    /// Upper bound as a float (may be slightly inexact).
    #[getter]
    fn hi(&self) -> f64 {
        self.inner.hi_f64()
    }

    /// Exact lower bound as ``(numerator_str, denominator_str)``.
    fn lo_exact(&self) -> (String, String) {
        self.inner.lo_exact()
    }

    /// Exact upper bound as ``(numerator_str, denominator_str)``.
    fn hi_exact(&self) -> (String, String) {
        self.inner.hi_exact()
    }

    fn __repr__(&self) -> String {
        let lo = self.inner.lo_f64();
        let hi = self.inner.hi_f64();
        if lo == hi {
            format!("RootInterval({lo})")
        } else {
            format!("RootInterval({lo}, {hi})")
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

fn core_interval_to_py(iv: CoreRootInterval) -> PyRootInterval {
    PyRootInterval { inner: iv }
}

/// Isolate all real roots of a polynomial expression.
///
/// Returns a list of :class:`RootInterval` objects sorted by lower endpoint.
/// Each interval contains exactly one real root of the squarefree part of
/// ``poly``.  Repeated roots appear once each.
///
/// Parameters
/// ----------
/// poly : Expr
///     A univariate polynomial expression with integer coefficients.
/// var : Expr
///     The polynomial variable.
///
/// Returns
/// -------
/// list[RootInterval]
///
/// Raises
/// ------
/// RealRootError
///     If ``poly`` is not a polynomial with integer coefficients, or is the
///     zero polynomial.
///
/// Example::
///
///     pool = ExprPool()
///     x = pool.symbol("x")
///     roots = real_roots(x**2 - pool.integer(4), x)
///     # roots ≈ [RootInterval(-2.0, -2.0), RootInterval(2.0, 2.0)]
#[pyfunction]
#[pyo3(name = "real_roots")]
fn py_real_roots(
    py: Python<'_>,
    poly: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
) -> PyResult<Vec<PyRootInterval>> {
    let pool = poly.pool.borrow(py);
    let intervals =
        core_real_roots_symbolic(poly.id, var.id, &pool.inner).map_err(real_root_error_to_py)?;
    Ok(intervals.into_iter().map(core_interval_to_py).collect())
}

/// Refine a :class:`RootInterval` to at least ``prec`` bits of precision.
///
/// Uses bisection with floating-point Horner evaluation.  For exact rational
/// roots (``lo == hi``), returns a zero-radius ball.
///
/// Parameters
/// ----------
/// poly : Expr
///     The same polynomial passed to :func:`real_roots`.
/// interval : RootInterval
///     One element of the list returned by :func:`real_roots`.
/// var : Expr
///     The polynomial variable.
/// prec : int
///     Desired precision in bits (minimum 53, clamped to ``max(53, prec)``).
///
/// Returns
/// -------
/// ArbBall
///     Rigorous floating-point ball containing the root.
///
/// Example::
///
///     pool = ExprPool()
///     x = pool.symbol("x")
///     ivs = real_roots(x**2 - pool.integer(2), x)
///     ball = refine_root(x**2 - pool.integer(2), ivs[1], x, 53)
///     # ball.mid ≈ 1.4142135623730951
#[pyfunction]
#[pyo3(name = "refine_root")]
fn py_refine_root(
    py: Python<'_>,
    poly: PyRef<PyExpr>,
    interval: PyRef<PyRootInterval>,
    var: PyRef<PyExpr>,
) -> PyResult<PyArbBall> {
    let pool = poly.pool.borrow(py);
    let uni = UniPoly::from_symbolic(poly.id, var.id, &pool.inner)
        .map_err(|e| real_root_error_to_py(RealRootError::NotAPolynomial(e)))?;
    let ball = core_refine_root(&uni, &interval.inner, 53);
    Ok(PyArbBall { inner: ball })
}

/// Factor a dense univariate polynomial over :math:`\mathbb{F}_p` from ascending
/// integer coefficients (reduced mod ``p``).
#[pyfunction]
#[pyo3(name = "factor_univariate_mod_p")]
fn py_factor_univariate_mod_p(coeffs: Vec<i64>, modulus: u64) -> PyResult<PyUniPolyFactorModP> {
    core_factor_univariate_mod_p(&coeffs, modulus)
        .map(|inner| PyUniPolyFactorModP { inner })
        .map_err(factor_error_to_py)
}

// ---------------------------------------------------------------------------
// V2-3 — Sparse interpolation Python bindings
// ---------------------------------------------------------------------------

/// Recover a sparse univariate polynomial over ``F_p`` from black-box
/// evaluations using the Ben-Or/Tiwari (Prony-style) algorithm.
///
/// Parameters
/// ----------
/// eval : callable
///     Black-box oracle ``x ↦ f(x) mod p``.  Called with a single
///     ``int`` argument and must return an ``int``.
/// term_bound : int
///     Upper bound ``T`` on the number of nonzero terms.  Exactly
///     ``2·T`` oracle calls are made.
/// prime : int
///     Field characteristic ``p``.  Must satisfy ``p > 2·T`` and
///     ``p > max_degree(f)``.
///
/// Returns
/// -------
/// list[tuple[int, int]]
///     List of ``(coefficient, exponent)`` pairs.
///
/// Raises
/// ------
/// SparseInterpError
///     On invalid prime, prime too small, or inconsistent oracle.
///
/// Example::
///
///     # Recover x^100 + 3·x^17 + 5 from 6 evaluations.
///     p = 997
///     def f(x): return (x**100 + 3*x**17 + 5) % p
///     terms = sparse_interp_univariate(f, 3, p)
///     # terms ≈ [(1, 100), (3, 17), (5, 0)]
#[pyfunction]
#[pyo3(name = "sparse_interp_univariate")]
fn py_sparse_interp_univariate(
    py: Python<'_>,
    eval: Bound<'_, pyo3::types::PyAny>,
    term_bound: usize,
    prime: u64,
) -> PyResult<Vec<(u64, u32)>> {
    let rust_eval = |x: u64| -> u64 {
        let result = eval
            .call1((x,))
            .expect("sparse_interp_univariate: oracle call failed");
        result
            .extract::<u64>()
            .expect("sparse_interp_univariate: oracle must return int")
    };
    let terms = core_sparse_interpolate_univariate(&rust_eval, term_bound, prime)
        .map_err(sparse_interp_error_to_py)?;
    let _ = py; // suppress unused warning
    Ok(terms)
}

/// Recover a sparse multivariate polynomial over ``F_p`` from black-box
/// evaluations using Zippel's variable-by-variable algorithm.
///
/// Parameters
/// ----------
/// eval : callable
///     Black-box oracle ``(x₁, …, xₙ) ↦ f(x₁, …, xₙ) mod p``.
///     Called with a Python ``list[int]`` (one int per variable) and
///     must return an ``int``.
/// vars : list[Expr]
///     Symbolic variable expressions in the same order as the
///     coordinates passed to ``eval``.
/// term_bound : int
///     Upper bound ``T`` on the number of nonzero terms.
/// degree_bound : int
///     Upper bound ``D`` on the degree of each individual variable.
///     For the dense fallback, set ``D ≤ T``.
/// prime : int
///     Field characteristic ``p``.  Must satisfy ``p > 2·T`` and
///     ``p > D``.
/// seed : int, optional
///     PRNG seed for random evaluation points (default 0).  Change
///     the seed to recover from occasional Vandermonde singularities.
///
/// Returns
/// -------
/// MultiPolyFp
///     Recovered polynomial with coefficients in ``[0, p)``.
///     On 20-variable inputs this is typically ≥ 5× faster in oracle
///     calls than dense interpolation.
///
/// Raises
/// ------
/// SparseInterpError
///     On invalid prime, prime too small, or inconsistent oracle.
///
/// Example::
///
///     pool = ExprPool()
///     x, y = pool.symbol("x"), pool.symbol("y")
///     p = 1009
///     def f(pt):
///         x_, y_ = pt
///         return (x_ * y_ + 3) % p
///     result = sparse_interp(f, [x, y], term_bound=4, degree_bound=3, prime=p)
#[pyfunction]
#[pyo3(name = "sparse_interp")]
#[pyo3(signature = (eval, vars, term_bound, degree_bound, prime, seed=0))]
fn py_sparse_interp(
    py: Python<'_>,
    eval: Bound<'_, pyo3::types::PyAny>,
    vars: Vec<PyRef<PyExpr>>,
    term_bound: usize,
    degree_bound: u32,
    prime: u64,
    seed: u64,
) -> PyResult<PyMultiPolyFp> {
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();

    let rust_eval = |pt: &[u64]| -> u64 {
        let py_list = pyo3::types::PyList::new_bound(py, pt.iter().copied());
        let result = eval
            .call1((py_list,))
            .expect("sparse_interp: oracle call failed");
        result
            .extract::<u64>()
            .expect("sparse_interp: oracle must return int")
    };

    let fp = core_sparse_interpolate(&rust_eval, var_ids, term_bound, degree_bound, prime, seed)
        .map_err(sparse_interp_error_to_py)?;
    Ok(PyMultiPolyFp { inner: fp })
}

// ---------------------------------------------------------------------------
// PA-9 — Piecewise Python bindings
// ---------------------------------------------------------------------------

/// Build a piecewise expression from a list of (condition_expr, value_expr) pairs
/// and a default value.
///
/// Conditions must be ``Predicate`` expressions built with the pool's predicate
/// constructors (``pool.lt``, ``pool.le``, ``pool.gt``, ``pool.ge``, etc.).
#[pyfunction(name = "piecewise")]
fn py_piecewise(
    py: Python<'_>,
    branches: Vec<(PyRef<PyExpr>, PyRef<PyExpr>)>,
    default: PyRef<PyExpr>,
) -> PyExpr {
    let pool_py = default.pool.clone_ref(py);
    let rust_branches: Vec<(ExprId, ExprId)> = branches.iter().map(|(c, v)| (c.id, v.id)).collect();
    let id = {
        let pool = pool_py.borrow(py);
        pool.inner.piecewise(rust_branches, default.id)
    };
    PyExpr { id, pool: pool_py }
}

// ---------------------------------------------------------------------------
// V3-3 — First-order logic (FOFormula)
// ---------------------------------------------------------------------------

fn require_same_pool(py: Python<'_>, a: &PyExpr, b: &PyExpr) -> PyResult<()> {
    if !a.pool.is(&b.pool) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expressions must belong to the same ExprPool",
        ));
    }
    let _ = py;
    Ok(())
}

/// Return ``False`` if unsatisfiable, ``True`` if satisfiable with no witness
/// variables, a ``dict`` of symbol → rational string if a witness is found, or
/// ``None`` if the fragment is unsupported.
#[pyfunction(name = "satisfiable")]
fn py_satisfiable(py: Python<'_>, formula: PyRef<PyExpr>) -> PyResult<PyObject> {
    let pool = formula.pool.borrow(py);
    let out: PyObject = match core_satisfiable(formula.id, &pool.inner) {
        CoreSatisfiability::Unsat => false.to_object(py),
        CoreSatisfiability::Unknown => py.None(),
        CoreSatisfiability::Sat(m) => {
            if m.is_empty() {
                true.to_object(py)
            } else {
                let d = PyDict::new_bound(py);
                for (k, v) in m {
                    d.set_item(k, v)?;
                }
                d.into_py(py)
            }
        }
    };
    Ok(out)
}

/// Logical conjunction of two predicate expressions (same pool).
#[pyfunction(name = "And")]
fn py_logic_and(py: Python<'_>, a: PyRef<PyExpr>, b: PyRef<PyExpr>) -> PyResult<PyExpr> {
    require_same_pool(py, &a, &b)?;
    let pool_py = a.pool.clone_ref(py);
    let id = pool_py.borrow(py).inner.pred_and(vec![a.id, b.id]);
    Ok(PyExpr { id, pool: pool_py })
}

/// Logical disjunction (same pool).
#[pyfunction(name = "Or")]
fn py_logic_or(py: Python<'_>, a: PyRef<PyExpr>, b: PyRef<PyExpr>) -> PyResult<PyExpr> {
    require_same_pool(py, &a, &b)?;
    let pool_py = a.pool.clone_ref(py);
    let id = pool_py.borrow(py).inner.pred_or(vec![a.id, b.id]);
    Ok(PyExpr { id, pool: pool_py })
}

/// Logical negation.
#[pyfunction(name = "Not")]
fn py_logic_not(_py: Python<'_>, a: PyRef<PyExpr>) -> PyExpr {
    let pool_py = a.pool.clone_ref(_py);
    let id = pool_py.borrow(_py).inner.pred_not(a.id);
    PyExpr { id, pool: pool_py }
}

/// ``∀ var . body`` (same pool).
#[pyfunction(name = "Forall")]
fn py_forall(py: Python<'_>, var: PyRef<PyExpr>, body: PyRef<PyExpr>) -> PyResult<PyExpr> {
    require_same_pool(py, &var, &body)?;
    let pool_py = var.pool.clone_ref(py);
    let id = pool_py.borrow(py).inner.forall(var.id, body.id);
    Ok(PyExpr { id, pool: pool_py })
}

/// ``∃ var . body`` (same pool).
#[pyfunction(name = "Exists")]
fn py_exists(py: Python<'_>, var: PyRef<PyExpr>, body: PyRef<PyExpr>) -> PyResult<PyExpr> {
    require_same_pool(py, &var, &body)?;
    let pool_py = var.pool.clone_ref(py);
    let id = pool_py.borrow(py).inner.exists(var.id, body.id);
    Ok(PyExpr { id, pool: pool_py })
}

fn cad_witness_symbol_name(pool: &ExprPool, sym: ExprId) -> PyResult<String> {
    match pool.get(sym) {
        alkahest_core::ExprData::Symbol { name, .. } => Ok(name.clone()),
        _ => Err(PyTypeError::new_err(
            "CAD witness uses non-symbol ExprId (internal error)",
        )),
    }
}

/// Decide a closed polynomial sentence over ℝ (one outer `\forall`/`\exists`; purely
/// polynomial body in the bound symbol with integer coefficients).
///
/// Returns ``(truth, witness_or_none)`` where ``witness`` maps symbol names to
/// rational decimal strings when an existential sentence is deduced satisfied.
#[pyfunction(name = "decide")]
fn py_decide(py: Python<'_>, formula: PyRef<PyExpr>) -> PyResult<(bool, PyObject)> {
    let pool_py = formula.pool.clone_ref(py);
    let bor = pool_py.borrow(py);
    let inner = &bor.inner;
    let r = core_decide_expr(formula.id, inner).map_err(cad_error_to_py)?;
    let wit: PyObject = match r.witness {
        None => py.None(),
        Some(m) => {
            let d = PyDict::new_bound(py);
            for (sym, rat) in m {
                let name = cad_witness_symbol_name(inner, sym)?;
                d.set_item(name, rat.to_string())?;
            }
            d.into_py(py)
        }
    };
    Ok((r.truth, wit))
}

/// Brown-style CAD projection polynomials after eliminating ``elim_var``.
#[pyfunction(name = "cad_project")]
fn py_cad_project(
    py: Python<'_>,
    polys: Vec<PyRef<PyExpr>>,
    elim_var: PyRef<PyExpr>,
) -> PyResult<Vec<PyExpr>> {
    let pool_py = elim_var.pool.clone_ref(py);
    for p in &polys {
        if !p.pool.is(&pool_py) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cad_project expects all Expr in the same ExprPool",
            ));
        }
    }
    let ids: Vec<ExprId> = polys.iter().map(|e| e.id).collect();
    let bor = pool_py.borrow(py);
    let out = core_cad_project(ids.as_slice(), elim_var.id, &bor.inner).map_err(cad_error_to_py)?;
    Ok(out
        .into_iter()
        .map(|id| PyExpr {
            id,
            pool: pool_py.clone_ref(py),
        })
        .collect())
}

#[pyfunction(name = "cad_lift")]
fn py_cad_lift(
    py: Python<'_>,
    polys: Vec<PyRef<PyExpr>>,
    main_var: PyRef<PyExpr>,
) -> PyResult<Vec<PyRootInterval>> {
    let pool_py = main_var.pool.clone_ref(py);
    for p in &polys {
        if !p.pool.is(&pool_py) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cad_lift expects all Expr in the same ExprPool",
            ));
        }
    }
    let ids: Vec<ExprId> = polys.iter().map(|e| e.id).collect();
    let bor = pool_py.borrow(py);
    let intervals =
        core_cad_lift(ids.as_slice(), main_var.id, &bor.inner).map_err(cad_error_to_py)?;
    Ok(intervals.into_iter().map(core_interval_to_py).collect())
}

// ---------------------------------------------------------------------------
// PA-5 — Primitive registry Python bindings
// ---------------------------------------------------------------------------

/// Python-visible wrapper around [`PrimitiveRegistry`].
#[pyclass(name = "PrimitiveRegistry")]
struct PyPrimitiveRegistry {
    inner: PrimitiveRegistry,
}

#[pymethods]
impl PyPrimitiveRegistry {
    /// Create a default registry pre-populated with all built-in primitives.
    #[new]
    fn new() -> Self {
        PyPrimitiveRegistry {
            inner: PrimitiveRegistry::default_registry(),
        }
    }

    /// Return a registry pre-populated with Alkahest's built-in primitives.
    #[staticmethod]
    fn default_registry() -> Self {
        PyPrimitiveRegistry {
            inner: PrimitiveRegistry::default_registry(),
        }
    }

    /// Return the capability bitfield for a named primitive as a dict.
    fn capabilities(&self, name: &str) -> std::collections::HashMap<String, bool> {
        let caps = self.inner.capabilities(name);
        [
            ("simplify", caps.contains(Capabilities::SIMPLIFY)),
            ("diff_forward", caps.contains(Capabilities::DIFF_FORWARD)),
            ("diff_reverse", caps.contains(Capabilities::DIFF_REVERSE)),
            ("numeric_f64", caps.contains(Capabilities::NUMERIC_F64)),
            ("numeric_ball", caps.contains(Capabilities::NUMERIC_BALL)),
            ("lower_llvm", caps.contains(Capabilities::LOWER_LLVM)),
            ("lean_theorem", caps.contains(Capabilities::LEAN_THEOREM)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    /// Return True if a primitive with this name is registered.
    fn is_registered(&self, name: &str) -> bool {
        self.inner.is_registered(name)
    }

    /// Return the coverage report as a Markdown string.
    fn coverage_report_markdown(&self) -> String {
        self.inner.coverage_report().to_markdown()
    }

    /// Return the coverage report as a list of dicts.
    fn coverage_report(&self) -> Vec<std::collections::HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            self.inner
                .coverage_report()
                .rows
                .into_iter()
                .map(|row| {
                    let caps = row.caps;
                    [
                        ("name", row.name.into_py(py)),
                        (
                            "simplify",
                            caps.contains(Capabilities::SIMPLIFY).into_py(py),
                        ),
                        (
                            "diff_forward",
                            caps.contains(Capabilities::DIFF_FORWARD).into_py(py),
                        ),
                        (
                            "diff_reverse",
                            caps.contains(Capabilities::DIFF_REVERSE).into_py(py),
                        ),
                        (
                            "numeric_f64",
                            caps.contains(Capabilities::NUMERIC_F64).into_py(py),
                        ),
                        (
                            "numeric_ball",
                            caps.contains(Capabilities::NUMERIC_BALL).into_py(py),
                        ),
                        (
                            "lower_llvm",
                            caps.contains(Capabilities::LOWER_LLVM).into_py(py),
                        ),
                        (
                            "lean_theorem",
                            caps.contains(Capabilities::LEAN_THEOREM).into_py(py),
                        ),
                    ]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect()
                })
                .collect()
        })
    }

    fn __repr__(&self) -> String {
        let report = self.inner.coverage_report();
        format!("PrimitiveRegistry({} primitives)", report.rows.len())
    }
}

// ---------------------------------------------------------------------------
// V5-2 — StableHLO/XLA bridge
// ---------------------------------------------------------------------------

/// `alkahest.to_stablehlo(expr, inputs) -> str`
///
/// Lower a symbolic expression to a StableHLO MLIR text module.
///
/// Parameters
/// ----------
/// expr : Expr
///     The expression to lower.
/// inputs : list[Expr]
///     The input variables (become function arguments in order).
/// fn_name : str, optional
///     The MLIR function name (default "alkahest_fn").
///
/// Returns
/// -------
/// str
///     Complete MLIR text module.
#[pyfunction]
#[pyo3(name = "to_stablehlo")]
#[pyo3(signature = (expr, inputs, fn_name="alkahest_fn"))]
fn py_to_stablehlo(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    inputs: Vec<PyRef<PyExpr>>,
    fn_name: &str,
) -> String {
    let pool_py = expr.pool.clone_ref(py);
    let pool = pool_py.borrow(py);
    let input_ids: Vec<ExprId> = inputs.iter().map(|e| e.id).collect();
    core_emit_stablehlo(expr.id, &input_ids, fn_name, &pool.inner)
}

// ---------------------------------------------------------------------------
// V5-3 — NVPTX JIT backend
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[pyclass(name = "CudaCompiledFn")]
struct PyCudaCompiledFn {
    ptx: String,
    n_inputs: usize,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyCudaCompiledFn {
    #[getter]
    fn ptx(&self) -> &str {
        &self.ptx
    }

    #[getter]
    fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    fn __repr__(&self) -> String {
        format!(
            "<CudaCompiledFn n_inputs={} ptx_len={}>",
            self.n_inputs,
            self.ptx.len()
        )
    }
}

/// `alkahest.compile_cuda(expr, inputs) -> CudaCompiledFn`
///
/// Compile a symbolic expression to a CUDA GPU kernel targeting the NVPTX
/// backend. Returns a `CudaCompiledFn` object whose `.ptx` attribute contains
/// the generated PTX assembly.
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(name = "compile_cuda")]
fn py_compile_cuda(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    inputs: &Bound<'_, PyList>,
) -> PyResult<PyCudaCompiledFn> {
    let input_ids: Vec<ExprId> = inputs
        .iter()
        .map(|item| {
            let e: PyRef<PyExpr> = item.extract()?;
            Ok(e.id)
        })
        .collect::<PyResult<_>>()?;

    let pool_ref = expr.pool.borrow(py);
    let compiled = core_compile_cuda(expr.id, &input_ids, &pool_ref.inner).map_err(|e| {
        Python::with_gil(|py2| {
            let exc_type = py2.get_type_bound::<PyCudaError>();
            make_structured_err(py2, &exc_type, &e)
        })
    })?;
    drop(pool_ref);

    Ok(PyCudaCompiledFn {
        ptx: compiled.ptx,
        n_inputs: compiled.n_inputs,
    })
}

// ---------------------------------------------------------------------------
// V5-11 — Gröbner basis
// ---------------------------------------------------------------------------

#[cfg(feature = "groebner")]
use alkahest_core::{
    dae_index_reduce, expr_to_gbpoly, primary_decomposition, radical as core_ideal_radical,
    rosenfeld_groebner_with_options, DaeIndexReduction, GbPoly, GroebnerBasis, MonomialOrder,
};

#[cfg(feature = "groebner")]
#[pyclass(name = "GbPoly")]
struct PyGbPoly {
    inner: GbPoly,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyGbPoly {
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn n_vars(&self) -> usize {
        self.inner.n_vars
    }

    fn __repr__(&self) -> String {
        format!("GbPoly(n_terms={})", self.inner.terms.len())
    }
}

#[cfg(feature = "groebner")]
#[pyclass(name = "GroebnerBasis")]
struct PyGroebnerBasis {
    inner: GroebnerBasis,
    /// Pool used when this basis was computed from expressions (None for bases
    /// returned by `solve()` which had no variable context stored).
    pool: Option<Py<PyExprPool>>,
    /// Variables in the order used for exponent vectors — populated by `compute()`.
    var_ids: Vec<ExprId>,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyGroebnerBasis {
    /// Compute a Gröbner basis (lex order) for the polynomial system
    /// `polys = 0` in the given variables.
    ///
    /// Parameters
    /// ----------
    /// polys : list[Expr]
    ///     Polynomial expressions, each representing `p(vars) = 0`.
    /// vars : list[Expr]
    ///     Symbolic variables (must be ``Symbol``).
    #[staticmethod]
    fn compute(
        py: Python<'_>,
        polys: Vec<PyRef<PyExpr>>,
        vars: Vec<PyRef<PyExpr>>,
    ) -> PyResult<PyGroebnerBasis> {
        if polys.is_empty() || vars.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GroebnerBasis.compute requires at least one polynomial and one variable",
            ));
        }
        let pool_py = polys[0].pool.clone_ref(py);
        let pool = pool_py.borrow(py);
        let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
        let mut gb_polys = Vec::with_capacity(polys.len());
        for p in &polys {
            let gbp = expr_to_gbpoly(p.id, &var_ids, &pool.inner)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            gb_polys.push(gbp);
        }
        drop(pool);
        let inner = GroebnerBasis::compute(gb_polys, MonomialOrder::Lex);
        Ok(PyGroebnerBasis {
            inner,
            pool: Some(pool_py),
            var_ids,
        })
    }

    /// Gröbner basis via Faugère's F5 (signature-based reduction, V2-8).
    ///
    /// Same calling convention as :meth:`compute`; polynomial term order is lex.
    /// Module signatures use lex on the monomial part × generator index.
    #[staticmethod]
    fn compute_f5(
        py: Python<'_>,
        polys: Vec<PyRef<PyExpr>>,
        vars: Vec<PyRef<PyExpr>>,
    ) -> PyResult<PyGroebnerBasis> {
        if polys.is_empty() || vars.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GroebnerBasis.compute_f5 requires at least one polynomial and one variable",
            ));
        }
        let pool_py = polys[0].pool.clone_ref(py);
        let pool = pool_py.borrow(py);
        let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
        let mut gb_polys = Vec::with_capacity(polys.len());
        for p in &polys {
            let gbp = expr_to_gbpoly(p.id, &var_ids, &pool.inner)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            gb_polys.push(gbp);
        }
        drop(pool);
        let inner = GroebnerBasis::compute_f5(gb_polys, MonomialOrder::Lex);
        Ok(PyGroebnerBasis {
            inner,
            pool: Some(pool_py),
            var_ids,
        })
    }

    fn reduce(&self, p: PyRef<PyGbPoly>) -> PyGbPoly {
        PyGbPoly {
            inner: self.inner.reduce(&p.inner),
        }
    }

    /// Test membership.  Accepts either a ``GbPoly`` or an ``Expr``; when
    /// passing an ``Expr`` the basis must have been created via ``compute()``
    /// so that the variable order is known.
    fn contains(&self, py: Python<'_>, p: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(gbp) = p.downcast::<PyGbPoly>() {
            return Ok(self.inner.contains(&gbp.borrow().inner));
        }
        if let Ok(expr) = p.downcast::<PyExpr>() {
            let pool_py = self.pool.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "GroebnerBasis has no variable context; use GroebnerBasis.compute() to build one that accepts Expr",
                )
            })?;
            let pool = pool_py.borrow(py);
            let gbp = expr_to_gbpoly(expr.borrow().id, &self.var_ids, &pool.inner)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            return Ok(self.inner.contains(&gbp));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "contains() expects a GbPoly or an Expr",
        ))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("GroebnerBasis(n_generators={})", self.inner.len())
    }
}

#[cfg(feature = "groebner")]
fn py_monomial_order_for_dae(order: Option<&str>) -> MonomialOrder {
    order
        .and_then(MonomialOrder::from_str)
        .unwrap_or(MonomialOrder::GRevLex)
}

/// V2-13 — Rosenfeld–Gröbner-style differential elimination result.
#[cfg(feature = "groebner")]
#[pyclass(name = "RosenfeldGroebnerResult")]
struct PyRosenfeldGroebnerResult {
    #[pyo3(get)]
    consistent: bool,
    #[pyo3(get)]
    truncated: bool,
    #[pyo3(get)]
    prolongation_rounds: usize,
    working_dae: DAE,
    final_basis: Option<GroebnerBasis>,
    pool: Py<PyExprPool>,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyRosenfeldGroebnerResult {
    fn working_dae(&self, py: Python<'_>) -> PyDAE {
        PyDAE {
            inner: self.working_dae.clone(),
            pool: self.pool.clone_ref(py),
        }
    }

    fn final_basis(&self, py: Python<'_>) -> PyResult<Option<Py<PyGroebnerBasis>>> {
        match &self.final_basis {
            None => Ok(None),
            Some(gb) => Ok(Some(Py::new(
                py,
                PyGroebnerBasis {
                    inner: gb.clone(),
                    pool: Some(self.pool.clone_ref(py)),
                    var_ids: vec![],
                },
            )?)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RosenfeldGroebnerResult(consistent={}, truncated={})",
            self.consistent, self.truncated
        )
    }
}

#[cfg(feature = "groebner")]
#[pyclass(name = "DaeIndexReduction")]
struct PyDaeIndexReduction {
    inner: DaeIndexReduction,
    pool: Py<PyExprPool>,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyDaeIndexReduction {
    #[getter]
    fn used_pantelides(&self) -> bool {
        matches!(self.inner, DaeIndexReduction::Pantelides(_))
    }

    #[getter]
    fn used_rosenfeld_groebner(&self) -> bool {
        matches!(self.inner, DaeIndexReduction::Rosenfeld(_))
    }

    /// Pantelides-reduced DAE if Pantelides succeeded; else Rosenfeld working DAE.
    fn dae(&self, py: Python<'_>) -> PyDAE {
        let dae = match &self.inner {
            DaeIndexReduction::Pantelides(p) => p.reduced_dae.clone(),
            DaeIndexReduction::Rosenfeld(r) => r.working_dae.clone(),
        };
        PyDAE {
            inner: dae,
            pool: self.pool.clone_ref(py),
        }
    }

    fn rosenfeld_groebner_result(&self, py: Python<'_>) -> Option<Py<PyRosenfeldGroebnerResult>> {
        match &self.inner {
            DaeIndexReduction::Rosenfeld(r) => Py::new(
                py,
                PyRosenfeldGroebnerResult {
                    consistent: r.consistent,
                    truncated: r.truncated,
                    prolongation_rounds: r.prolongation_rounds,
                    working_dae: r.working_dae.clone(),
                    final_basis: r.final_basis.clone(),
                    pool: self.pool.clone_ref(py),
                },
            )
            .ok(),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            DaeIndexReduction::Pantelides(p) => format!(
                "DaeIndexReduction(pantelides, differentiation_steps={})",
                p.differentiation_steps
            ),
            DaeIndexReduction::Rosenfeld(_) => "DaeIndexReduction(rosenfeld_groebner)".to_string(),
        }
    }
}

#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "rosenfeld_groebner", signature = (dae, order=None, max_prolong_rounds=None))]
fn py_rosenfeld_groebner(
    py: Python<'_>,
    dae: PyRef<PyDAE>,
    order: Option<&str>,
    max_prolong_rounds: Option<usize>,
) -> PyResult<PyRosenfeldGroebnerResult> {
    let pool_py = dae.pool.clone_ref(py);
    let r = {
        let pool = pool_py.borrow(py);
        rosenfeld_groebner_with_options(&dae.inner, &pool.inner, py_monomial_order_for_dae(order), max_prolong_rounds.unwrap_or(8))
    };
    let r = r.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyRosenfeldGroebnerResult {
        consistent: r.consistent,
        truncated: r.truncated,
        prolongation_rounds: r.prolongation_rounds,
        working_dae: r.working_dae,
        final_basis: r.final_basis,
        pool: pool_py,
    })
}

#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "dae_index_reduce", signature = (dae, order=None))]
fn py_dae_index_reduce(py: Python<'_>, dae: PyRef<PyDAE>, order: Option<&str>) -> PyResult<PyDaeIndexReduction> {
    let pool_py = dae.pool.clone_ref(py);
    let inner = {
        let pool = pool_py.borrow(py);
        dae_index_reduce(&dae.inner, &pool.inner, py_monomial_order_for_dae(order))
    };
    let inner = inner.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyDaeIndexReduction {
        inner,
        pool: pool_py,
    })
}

#[cfg(feature = "groebner")]
#[pyclass(name = "PrimaryComponent")]
struct PyPrimaryComponent {
    primary: GroebnerBasis,
    associated_prime: GroebnerBasis,
    pool: Py<PyExprPool>,
    var_ids: Vec<ExprId>,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyPrimaryComponent {
    /// Gröbner basis of the primary component.
    #[pyo3(name = "primary")]
    fn py_primary(&self, py: Python<'_>) -> PyResult<Py<PyGroebnerBasis>> {
        Py::new(
            py,
            PyGroebnerBasis {
                inner: self.primary.clone(),
                pool: Some(self.pool.clone_ref(py)),
                var_ids: self.var_ids.clone(),
            },
        )
    }

    /// Gröbner basis of the associated prime (√Q).
    #[pyo3(name = "associated_prime")]
    fn py_associated_prime(&self, py: Python<'_>) -> PyResult<Py<PyGroebnerBasis>> {
        Py::new(
            py,
            PyGroebnerBasis {
                inner: self.associated_prime.clone(),
                pool: Some(self.pool.clone_ref(py)),
                var_ids: self.var_ids.clone(),
            },
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PrimaryComponent(primary_generators={}, associated_generators={})",
            self.primary.len(),
            self.associated_prime.len()
        )
    }
}

#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "primary_decomposition", signature = (polys, vars))]
fn py_primary_decomposition(
    py: Python<'_>,
    polys: Vec<PyRef<PyExpr>>,
    vars: Vec<PyRef<PyExpr>>,
) -> PyResult<Vec<Py<PyPrimaryComponent>>> {
    if polys.is_empty() || vars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "primary_decomposition requires at least one polynomial and one variable",
        ));
    }
    let pool_py = polys[0].pool.clone_ref(py);
    let pool = pool_py.borrow(py);
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let mut gb_polys = Vec::with_capacity(polys.len());
    for p in &polys {
        let gbp = expr_to_gbpoly(p.id, &var_ids, &pool.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        gb_polys.push(gbp);
    }
    drop(pool);
    let comps = primary_decomposition(gb_polys, MonomialOrder::Lex).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    })?;
    let mut out = Vec::with_capacity(comps.len());
    for c in comps {
        out.push(Py::new(
            py,
            PyPrimaryComponent {
                primary: c.primary,
                associated_prime: c.associated_prime,
                pool: pool_py.clone_ref(py),
                var_ids: var_ids.clone(),
            },
        )?);
    }
    Ok(out)
}

/// Radical √I of the ideal generated by `polys` (same variable order as `vars`).
#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "radical", signature = (polys, vars))]
fn py_ideal_radical(
    py: Python<'_>,
    polys: Vec<PyRef<PyExpr>>,
    vars: Vec<PyRef<PyExpr>>,
) -> PyResult<Py<PyGroebnerBasis>> {
    if polys.is_empty() || vars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "radical requires at least one polynomial and one variable",
        ));
    }
    let pool_py = polys[0].pool.clone_ref(py);
    let pool = pool_py.borrow(py);
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let mut gb_polys = Vec::with_capacity(polys.len());
    for p in &polys {
        let gbp = expr_to_gbpoly(p.id, &var_ids, &pool.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        gb_polys.push(gbp);
    }
    drop(pool);
    let gb = core_ideal_radical(gb_polys, MonomialOrder::Lex).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    })?;
    Py::new(
        py,
        PyGroebnerBasis {
            inner: gb,
            pool: Some(pool_py),
            var_ids,
        },
    )
}

// ---------------------------------------------------------------------------
// V1-4 — Polynomial system solver
// ---------------------------------------------------------------------------

#[cfg(feature = "groebner")]
use alkahest_core::{solve_polynomial_system, SolutionSet, RegularChain, triangularize};

/// `alkahest.solve(equations, vars, *, numeric=False) -> list[dict] | GroebnerBasis | list`
///
/// Solve a zero-dimensional polynomial system.
///
/// Parameters
/// ----------
/// equations : list[Expr]
///     Each expression represents `p(vars) = 0`.
/// vars : list[Expr]
///     The symbolic variables to solve for (must be symbols).
/// numeric : bool, default False
///     When ``False`` (default), each solution dict maps ``Expr → Expr``
///     (symbolic).  When ``True``, values are cast to ``float`` (legacy
///     behaviour; useful for quick numerical checks).
///
/// Returns
/// -------
/// list[dict]
///     Each dict maps a variable ``Expr`` to a solution value.
///     Returns an empty list when no solution exists.
/// GroebnerBasis
///     When the system has infinitely many solutions (parametric ideal).
#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "solve", signature = (equations, vars, numeric = false))]
fn py_solve(
    py: Python<'_>,
    equations: Vec<PyRef<PyExpr>>,
    vars: Vec<PyRef<PyExpr>>,
    numeric: bool,
) -> PyResult<PyObject> {
    if equations.is_empty() || vars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "solve requires at least one equation and one variable",
        ));
    }
    let pool_py = equations[0].pool.clone_ref(py);
    let eq_ids: Vec<ExprId> = equations.iter().map(|e| e.id).collect();
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();

    let result = {
        let pool = pool_py.borrow(py);
        solve_polynomial_system(eq_ids, var_ids.clone(), &pool.inner)
    };

    match result {
        Err(e) => Python::with_gil(|py2| {
            let exc_type = py2.get_type_bound::<PySolverError>();
            Err(make_structured_err(py2, &exc_type, &e))
        }),
        Ok(SolutionSet::NoSolution) => Ok(pyo3::types::PyList::empty_bound(py).into()),
        Ok(SolutionSet::Parametric(gb)) => Ok(PyGroebnerBasis {
            inner: gb,
            pool: None,
            var_ids: vec![],
        }
        .into_py(py)),
        Ok(SolutionSet::Finite(solutions)) => {
            let list = pyo3::types::PyList::empty_bound(py);
            let pool = pool_py.borrow(py);
            for sol in solutions {
                let d = pyo3::types::PyDict::new_bound(py);
                for (i, val) in sol.iter().enumerate() {
                    let var_expr = PyExpr {
                        id: var_ids[i],
                        pool: pool_py.clone_ref(py),
                    };
                    if numeric {
                        // Legacy numeric path: cast solution ExprId to f64.
                        let env: std::collections::HashMap<ExprId, f64> =
                            std::collections::HashMap::new();
                        let f = alkahest_core::jit::eval_interp(*val, &env, &pool.inner)
                            .unwrap_or(f64::NAN);
                        d.set_item(var_expr.into_py(py), f)?;
                    } else {
                        // Symbolic path: wrap the solution ExprId as a PyExpr.
                        let val_expr = PyExpr {
                            id: *val,
                            pool: pool_py.clone_ref(py),
                        };
                        d.set_item(var_expr.into_py(py), val_expr.into_py(py))?;
                    }
                }
                list.append(d)?;
            }
            Ok(list.into())
        }
    }
}

// ---------------------------------------------------------------------------
// V2-11 — Regular chains / triangular decomposition
// ---------------------------------------------------------------------------

#[cfg(feature = "groebner")]
#[pyclass(name = "RegularChain")]
struct PyRegularChain {
    inner: RegularChain,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyRegularChain {
    #[getter]
    fn n_vars(&self) -> usize {
        self.inner.n_vars
    }

    /// Gröbner-style polynomial tiles (``GbPoly``), ascending by main variable.
    fn polys(&self) -> Vec<PyGbPoly> {
        self.inner
            .polys
            .iter()
            .map(|p| PyGbPoly { inner: p.clone() })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "RegularChain(n_vars={}, n_polys={})",
            self.inner.n_vars,
            self.inner.len()
        )
    }
}

/// Lex-basis triangular decomposition (possibly split on factored univariates).
#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "triangularize", signature = (equations, vars))]
fn py_triangularize(
    py: Python<'_>,
    equations: Vec<PyRef<PyExpr>>,
    vars: Vec<PyRef<PyExpr>>,
) -> PyResult<PyObject> {
    if equations.is_empty() || vars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "triangularize requires at least one equation and one variable",
        ));
    }
    let pool_py = equations[0].pool.clone_ref(py);
    let eq_ids: Vec<ExprId> = equations.iter().map(|e| e.id).collect();
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();

    let result = {
        let pool = pool_py.borrow(py);
        triangularize(eq_ids, var_ids, &pool.inner)
    };

    match result {
        Err(e) => Python::with_gil(|py2| {
            let exc_type = py2.get_type_bound::<PySolverError>();
            Err(make_structured_err(py2, &exc_type, &e))
        }),
        Ok(chains) => {
            let list = pyo3::types::PyList::empty_bound(py);
            for chain in chains {
                list.append(
                    PyRegularChain { inner: chain }.into_py(py),
                )?;
            }
            Ok(list.into())
        }
    }
}

// ---------------------------------------------------------------------------
// V2-1 — Modular / CRT framework
// ---------------------------------------------------------------------------

#[pyclass(name = "MultiPolyFp")]
struct PyMultiPolyFp {
    inner: MultiPolyFp,
}

#[pymethods]
impl PyMultiPolyFp {
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn total_degree(&self) -> u32 {
        self.inner.total_degree()
    }

    #[getter]
    fn modulus(&self) -> u64 {
        self.inner.modulus
    }

    /// Return the polynomial's terms as a ``dict`` mapping exponent tuples
    /// to coefficients.  Exponent tuples have trailing zeros removed.
    ///
    /// Example::
    ///
    ///     fp = modular_reduce(poly, 101)
    ///     for exp_tuple, coeff in fp.terms.items():
    ///         print(exp_tuple, coeff)
    #[getter]
    fn terms<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyDict> {
        let dict = pyo3::types::PyDict::new_bound(py);
        for (exp, &coeff) in &self.inner.terms {
            let key = pyo3::types::PyTuple::new_bound(py, exp.iter().copied());
            dict.set_item(key, coeff).unwrap();
        }
        dict
    }

    fn __repr__(&self) -> String {
        format!("MultiPolyFp({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Reduce a polynomial over ℤ to F_p = ℤ/pℤ.
///
/// Returns a `MultiPolyFp` with coefficients in [0, p).
/// Raises `ModularError` if `p` is not prime.
#[pyfunction]
#[pyo3(name = "modular_reduce")]
fn py_modular_reduce(poly: PyRef<PyMultiPoly>, p: u64) -> PyResult<PyMultiPolyFp> {
    core_reduce_mod(&poly.inner, p)
        .map(|fp| PyMultiPolyFp { inner: fp })
        .map_err(modular_error_to_py)
}

/// Reconstruct a polynomial over ℤ from modular images via CRT.
///
/// `polys` and `primes` must have the same length.
/// All images must share the same variable list.
#[pyfunction]
#[pyo3(name = "modular_lift_crt")]
fn py_modular_lift_crt(
    polys: Vec<PyRef<PyMultiPolyFp>>,
    primes: Vec<u64>,
) -> PyResult<PyMultiPoly> {
    if polys.len() != primes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polys and primes must have the same length",
        ));
    }
    let images: Vec<(MultiPolyFp, u64)> = polys
        .iter()
        .zip(primes.iter())
        .map(|(p, &prime)| (p.inner.clone(), prime))
        .collect();
    core_lift_crt(&images)
        .map(|mp| PyMultiPoly { inner: mp })
        .map_err(modular_error_to_py)
}

/// Rational reconstruction: find a/b ≡ n (mod m) with small |a| and b.
///
/// Returns `(a_str, b_str)` as decimal strings (convert with `int()`),
/// or `None` if no rational with norm ≤ ⌊√(m/2)⌋ exists.
/// Both `n_str` and `m_str` are decimal integer strings.
#[pyfunction]
#[pyo3(name = "modular_rational_reconstruction")]
fn py_modular_rational_reconstruction(
    n_str: &str,
    m_str: &str,
) -> PyResult<Option<(String, String)>> {
    use rug::{Complete, Integer};
    let n = Integer::parse(n_str)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid integer for n"))?
        .complete();
    let m = Integer::parse(m_str)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid integer for m"))?
        .complete();
    Ok(core_rational_reconstruction(&n, &m).map(|(a, b)| (a.to_string(), b.to_string())))
}

/// Compute the Mignotte coefficient bound for a polynomial.
///
/// Returns the bound as a decimal integer string (use `int()` to convert).
#[pyfunction]
#[pyo3(name = "modular_mignotte_bound")]
fn py_modular_mignotte_bound(poly: PyRef<PyMultiPoly>) -> String {
    core_mignotte_bound(&poly.inner).to_string()
}

/// Select the smallest lucky prime not in `used` that does not divide `avoid_divisor_str`.
///
/// `avoid_divisor_str` is a decimal integer string. Pass `"0"` for no constraint.
#[pyfunction]
#[pyo3(name = "modular_select_lucky_prime")]
fn py_modular_select_lucky_prime(avoid_divisor_str: &str, used: Vec<u64>) -> PyResult<u64> {
    use rug::{Complete, Integer};
    let avoid = Integer::parse(avoid_divisor_str)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid integer for avoid_divisor"))?
        .complete();
    Ok(core_select_lucky_prime(&avoid, &used))
}

/// LLL‑reduce rows of integers (same ambient dimension across rows).
#[pyfunction]
#[pyo3(name = "lat_lll_reduce_rows", signature=(rows, delta_num=None, delta_den=None))]
fn py_lat_lll_reduce_rows(
    rows: Vec<Vec<i64>>,
    delta_num: Option<i64>,
    delta_den: Option<i64>,
) -> PyResult<Vec<Vec<i64>>> {
    use rug::Integer;
    let basis: Vec<Vec<Integer>> = rows
        .into_iter()
        .map(|r| r.into_iter().map(Integer::from).collect())
        .collect();
    let reduced = match (delta_num, delta_den) {
        (Some(n), Some(d)) if d != 0 => {
            let delta = rug::Rational::from((n, d));
            core_lattice_reduce_rows_with_delta(&basis, delta).map_err(lattice_error_to_py)?
        }
        _ => core_lattice_reduce_rows(&basis).map_err(lattice_error_to_py)?,
    };
    reduced
        .into_iter()
        .map(|r| {
            r.into_iter()
                .map(|z| {
                    z.to_i64()
                        .ok_or_else(|| PyOverflowError::new_err("LLL matrix entry overflows i64"))
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()
}

/// Search for `[aᵢ]` such that Σ aᵢ constantsᵢ ≈ 0 (mixed `float` / decimal strings).
///
/// Typical high‑precision literals: `"1.644934066848226436472415166646025189219…"` matched with
/// `precision_bits≈664` for ~200 decimals.
#[pyfunction]
#[pyo3(name = "guess_relation", signature=(constants, precision_bits=664, max_abs_coeff=None))]
fn py_guess_relation(
    constants: Bound<'_, PyAny>,
    precision_bits: u32,
    max_abs_coeff: Option<u128>,
) -> PyResult<Option<Vec<i64>>> {
    use rug::ops::CompleteRound;
    use rug::Float;
    let list = constants
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("constants must be a list"))?;
    let n = list.len();
    let mut xs: Vec<Float> = Vec::with_capacity(n);
    for i in 0..n {
        let item = list.get_item(i)?;
        if let Ok(v) = item.extract::<f64>() {
            xs.push(Float::with_val(precision_bits, v));
        } else if let Ok(s) = item.extract::<String>() {
            xs.push(
                Float::parse(s.trim())
                    .map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(
                            "could not parse decimal string as floating constant",
                        )
                    })?
                    .complete(precision_bits),
            );
        } else {
            return Err(PyTypeError::new_err(
                "each constant must be a float or decimal string",
            ));
        }
    }
    let rel = core_guess_integer_relation(&xs, precision_bits, max_abs_coeff)
        .map_err(pslq_error_to_py)?;
    Ok(match rel {
        None => None,
        Some(coeffs) => {
            let mut out = Vec::with_capacity(coeffs.len());
            for z in coeffs {
                let v = z.to_i64().ok_or_else(|| {
                    PyOverflowError::new_err("coefficient overflows i64; report for bigint output")
                })?;
                out.push(v);
            }
            Some(out)
        }
    })
}

#[pymodule]
fn alkahest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_egraph, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_egraph_with, m)?)?;
    m.add_class::<PyEgraphConfig>()?;
    m.add_function(wrap_pyfunction!(py_simplify_with, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_expanded, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_trig, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_log_exp, m)?)?;
    m.add_function(wrap_pyfunction!(py_diff, m)?)?;
    m.add_function(wrap_pyfunction!(py_diff_forward, m)?)?;
    m.add_function(wrap_pyfunction!(py_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(py_sum_indefinite, m)?)?;
    m.add_function(wrap_pyfunction!(py_sum_definite, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve_linear_recurrence_homogeneous, m)?)?;
    m.add_function(wrap_pyfunction!(py_verify_wz_pair, m)?)?;
    m.add_function(wrap_pyfunction!(match_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(make_rule, m)?)?;
    m.add_function(wrap_pyfunction!(py_subs, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    // V1-12: expanded primitive registry
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(asin, m)?)?;
    m.add_function(wrap_pyfunction!(acos, m)?)?;
    m.add_function(wrap_pyfunction!(atan, m)?)?;
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(erfc, m)?)?;
    m.add_function(wrap_pyfunction!(abs_expr, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(round_expr, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(atan2, m)?)?;
    m.add_function(wrap_pyfunction!(min_expr, m)?)?;
    m.add_function(wrap_pyfunction!(max_expr, m)?)?;
    m.add_class::<PyExprPool>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyDerivedResult>()?;
    m.add_class::<PyUniPoly>()?;
    m.add_class::<PyMultiPoly>()?;
    m.add_class::<PyUniPolyFactorization>()?;
    m.add_class::<PyMultiPolyFactorization>()?;
    m.add_class::<PyUniPolyFactorModP>()?;
    m.add_class::<PyRationalFunction>()?;
    m.add_class::<PyRewriteRule>()?;
    // Phase 14
    m.add_function(wrap_pyfunction!(py_grad, m)?)?;
    // Phase 15
    m.add_function(wrap_pyfunction!(py_jacobian, m)?)?;
    m.add_class::<PyMatrix>()?;
    // Phase 16
    m.add_class::<PyODE>()?;
    m.add_function(wrap_pyfunction!(py_lower_to_first_order, m)?)?;
    // Phase 17
    m.add_class::<PyDAE>()?;
    m.add_function(wrap_pyfunction!(py_pantelides, m)?)?;
    // Phase 18
    m.add_class::<PyPort>()?;
    m.add_class::<PyAcausalSystem>()?;
    m.add_function(wrap_pyfunction!(py_resistor, m)?)?;
    // Phase 19
    m.add_class::<PySensitivitySystem>()?;
    m.add_function(wrap_pyfunction!(py_sensitivity_system, m)?)?;
    m.add_function(wrap_pyfunction!(py_adjoint_system, m)?)?;
    // Phase 20
    m.add_class::<PyEvent>()?;
    m.add_class::<PyHybridODE>()?;
    // Phase 21 — JIT
    m.add_function(wrap_pyfunction!(py_compile_expr, m)?)?;
    m.add_function(wrap_pyfunction!(py_eval_expr, m)?)?;
    m.add_class::<PyCompiledFn>()?;
    // Phase 22 — Ball arithmetic
    m.add_class::<PyArbBall>()?;
    m.add_function(wrap_pyfunction!(py_interval_eval, m)?)?;
    // Phase 23 — Parallel simplification
    m.add_function(wrap_pyfunction!(py_simplify_par, m)?)?;
    // Phase 24 — Horner form
    m.add_function(wrap_pyfunction!(py_horner, m)?)?;
    m.add_function(wrap_pyfunction!(py_emit_c, m)?)?;
    // Phase 26 — collect_like_terms
    m.add_function(wrap_pyfunction!(py_collect_like_terms, m)?)?;
    // Phase 27 — poly_normal
    m.add_function(wrap_pyfunction!(py_poly_normal, m)?)?;
    // PA-5 — Primitive registry
    m.add_class::<PyPrimitiveRegistry>()?;
    // PA-9 — Piecewise
    m.add_function(wrap_pyfunction!(py_piecewise, m)?)?;
    m.add_function(wrap_pyfunction!(py_satisfiable, m)?)?;
    m.add_function(wrap_pyfunction!(py_logic_and, m)?)?;
    m.add_function(wrap_pyfunction!(py_logic_or, m)?)?;
    m.add_function(wrap_pyfunction!(py_logic_not, m)?)?;
    m.add_function(wrap_pyfunction!(py_forall, m)?)?;
    m.add_function(wrap_pyfunction!(py_exists, m)?)?;
    m.add_function(wrap_pyfunction!(py_decide, m)?)?;
    m.add_function(wrap_pyfunction!(py_cad_project, m)?)?;
    m.add_function(wrap_pyfunction!(py_cad_lift, m)?)?;
    // V5-1 — Lean 4 certificate exporter
    m.add_function(wrap_pyfunction!(py_to_lean, m)?)?;
    // V5-2 — StableHLO/XLA bridge
    m.add_function(wrap_pyfunction!(py_to_stablehlo, m)?)?;
    // V5-3 — NVPTX JIT backend
    #[cfg(feature = "cuda")]
    {
        m.add_class::<PyCudaCompiledFn>()?;
        m.add_function(wrap_pyfunction!(py_compile_cuda, m)?)?;
    }
    // V5-11 — Gröbner basis / V1-16 — GroebnerBasis.compute
    #[cfg(feature = "groebner")]
    {
        m.add_class::<PyGbPoly>()?;
        m.add_class::<PyGroebnerBasis>()?;
        m.add_class::<PyRosenfeldGroebnerResult>()?;
        m.add_class::<PyDaeIndexReduction>()?;
        m.add_class::<PyPrimaryComponent>()?;
        m.add_class::<PyRegularChain>()?;
        m.add_function(wrap_pyfunction!(py_solve, m)?)?;
        m.add_function(wrap_pyfunction!(py_triangularize, m)?)?;
        m.add_function(wrap_pyfunction!(py_primary_decomposition, m)?)?;
        m.add_function(wrap_pyfunction!(py_ideal_radical, m)?)?;
        m.add_function(wrap_pyfunction!(py_rosenfeld_groebner, m)?)?;
        m.add_function(wrap_pyfunction!(py_dae_index_reduce, m)?)?;
    }
    // V2-2 — Resultants and subresultant PRS
    m.add_function(wrap_pyfunction!(py_resultant, m)?)?;
    m.add_function(wrap_pyfunction!(py_subresultant_prs, m)?)?;
    // V2-3 — Sparse interpolation
    m.add_function(wrap_pyfunction!(py_sparse_interp_univariate, m)?)?;
    m.add_function(wrap_pyfunction!(py_sparse_interp, m)?)?;
    // V2-4 — Real root isolation
    m.add_class::<PyRootInterval>()?;
    m.add_function(wrap_pyfunction!(py_real_roots, m)?)?;
    m.add_function(wrap_pyfunction!(py_refine_root, m)?)?;
    m.add_function(wrap_pyfunction!(py_factor_univariate_mod_p, m)?)?;
    // V2-6 — Lattice / integer relations
    m.add_function(wrap_pyfunction!(py_lat_lll_reduce_rows, m)?)?;
    m.add_function(wrap_pyfunction!(py_guess_relation, m)?)?;
    // V2-1 — Modular / CRT framework
    m.add_class::<PyMultiPolyFp>()?;
    m.add_function(wrap_pyfunction!(py_modular_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(py_modular_lift_crt, m)?)?;
    m.add_function(wrap_pyfunction!(py_modular_rational_reconstruction, m)?)?;
    m.add_function(wrap_pyfunction!(py_modular_mignotte_bound, m)?)?;
    m.add_function(wrap_pyfunction!(py_modular_select_lucky_prime, m)?)?;
    // V1-3 — Structured exception hierarchy
    m.add("AlkahestError", m.py().get_type_bound::<PyAlkahestError>())?;
    m.add(
        "ConversionError",
        m.py().get_type_bound::<PyConversionError>(),
    )?;
    m.add("DomainError", m.py().get_type_bound::<PyDomainError>())?;
    m.add("DiffError", m.py().get_type_bound::<PyDiffError>())?;
    m.add("PoolError", m.py().get_type_bound::<PyPoolError>())?;
    m.add(
        "IntegrationError",
        m.py().get_type_bound::<PyIntegrationError>(),
    )?;
    m.add("MatrixError", m.py().get_type_bound::<PyMatrixError>())?;
    m.add("ModularError", m.py().get_type_bound::<PyModularError>())?;
    m.add("OdeError", m.py().get_type_bound::<PyOdeError>())?;
    m.add("DaeError", m.py().get_type_bound::<PyDaeError>())?;
    m.add("JitError", m.py().get_type_bound::<PyJitError>())?;
    m.add("SolverError", m.py().get_type_bound::<PySolverError>())?;
    m.add("CudaError", m.py().get_type_bound::<PyCudaError>())?;
    m.add("IoError", m.py().get_type_bound::<PyIoError>())?;
    m.add("ParseError", m.py().get_type_bound::<PyParseError>())?;
    m.add("FactorError", m.py().get_type_bound::<PyFactorError>())?;
    m.add(
        "ResultantError",
        m.py().get_type_bound::<PyResultantError>(),
    )?;
    m.add(
        "SparseInterpError",
        m.py().get_type_bound::<PySparseInterpError>(),
    )?;
    m.add("RealRootError", m.py().get_type_bound::<PyRealRootError>())?;
    m.add("LatticeError", m.py().get_type_bound::<PyLatticeError>())?;
    m.add("PslqError", m.py().get_type_bound::<PyPslqError>())?;
    m.add("CadError", m.py().get_type_bound::<PyCadError>())?;
    m.add("SumError", m.py().get_type_bound::<PySumError>())?;
    m.add(
        "LinearRecurrenceError",
        m.py().get_type_bound::<PyLinearRecurrenceError>(),
    )?;
    // V1-15: compile-time flag so Python tests can skip egraph-dependent assertions.
    m.add("HAS_EGRAPH", cfg!(feature = "egraph"))?;
    Ok(())
}
