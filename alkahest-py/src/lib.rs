use alkahest_core::{
    adjoint_system as core_adjoint_system,
    cad_lift as core_cad_lift,
    cad_project as core_cad_project,
    // Rational-function cancel/together
    cancel as core_cancel,
    capacitor as core_capacitor,
    // Phase 21 — JIT
    compile as core_compile,
    decide_expr as core_decide_expr,
    emit_expr_c as core_emit_expr_c,
    emit_expr_c_vec as core_emit_expr_c_vec,
    emit_horner_c as core_emit_horner_c,
    emit_stablehlo as core_emit_stablehlo,
    eval_interp_checked as core_eval_interp_checked,
    factor_univariate_mod_p as core_factor_univariate_mod_p,
    // V2-3 — Sparse interpolation and sparse modular GCD
    gcd_sparse_modular as core_gcd_sparse_modular,
    grad as core_grad,
    guess_integer_relation as core_guess_integer_relation,
    // Phase 24 — Horner form
    horner as core_horner,
    jacobian as core_jacobian,
    jit_available as core_jit_available,
    lattice_reduce_rows as core_lattice_reduce_rows,
    lattice_reduce_rows_with_delta as core_lattice_reduce_rows_with_delta,
    lower_to_first_order as core_lower_to_first_order,
    pantelides as core_pantelides,
    // Phase 27 — poly_normal
    poly_normal as core_poly_normal,
    product_definite as core_product_definite,
    product_indefinite as core_product_indefinite,
    // V2-4 — Real root isolation
    real_roots_symbolic as core_real_roots_symbolic,
    refine_root as core_refine_root,
    residue as core_residue,
    resistor as core_resistor,
    // V2-2 — Resultants and subresultant PRS
    resultant as core_resultant,
    // Parametric Routh–Hurwitz stability conditions
    routh_hurwitz as core_routh_hurwitz,
    // V3-3 — FOFormula / satisfiability
    satisfiable as core_satisfiable,
    sensitivity_system as core_sensitivity_system,
    solve_linear_recurrence_homogeneous as core_solve_linear_recurrence_homogeneous,
    sparse_interpolate as core_sparse_interpolate,
    sparse_interpolate_univariate as core_sparse_interpolate_univariate,
    subresultant_prs as core_subresultant_prs,
    subs as core_subs,
    sum_definite as core_sum_definite,
    sum_indefinite as core_sum_indefinite,
    together as core_together,
    verify_wz_pair as core_verify_wz_pair,
    voltage_source as core_voltage_source,
    // Phase 22 — Ball arithmetic
    ArbBall as CoreArbBall,
    // V2-9 — CAD / real QE
    CadError,
    Capabilities,
    CompileCache as CoreCompileCache,
    Component,
    Domain,
    EigenError,
    Event,
    ExprId,
    ExprPool,
    FactorError,
    HybridODE,
    IntervalEval as CoreIntervalEval,
    LatticeError,
    LinearAlgebraError,
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
    WzPair,
    DAE,
    ODE,
};

use alkahest_core::kernel::expr::PredicateKind;
use alkahest_core::kernel::fold_predicates as core_fold_predicates;
use alkahest_core::kernel::ExprData;
use alkahest_core::pattern::{
    match_pattern_with_config as core_match_pattern_with_config, MatchConfig,
};
#[cfg(feature = "cuda")]
use alkahest_core::{compile_cuda as core_compile_cuda, CudaCompiledFn as CoreCudaCompiledFn};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
// V2-1 — Modular / CRT framework
use alkahest_core::deriv::RewriteStep;
use alkahest_core::modular::{
    lift_crt as core_lift_crt, mignotte_bound as core_mignotte_bound,
    rational_reconstruction as core_rational_reconstruction, reduce_mod as core_reduce_mod,
    select_lucky_prime as core_select_lucky_prime, ModularError, MultiPolyFp,
};
use alkahest_core::{
    apart as core_apart, diff as core_diff, diff_forward as core_diff_forward,
    eval_complex_f64 as core_eval_complex_f64, eval_exact_rational as core_eval_exact_rational,
    eval_f64 as core_eval_f64, eval_interval as core_eval_interval, integrate as core_integrate,
    integrate_definite as core_integrate_definite, limit as core_limit, load_from,
    rsolve as core_rsolve, series as core_series, simplify as core_simplify,
    simplify_batch as core_simplify_batch, simplify_egraph as core_simplify_egraph,
    simplify_egraph_with as core_simplify_egraph_with, simplify_log_exp as core_simplify_log_exp,
    simplify_trig_normal_form as core_simplify_trig_normal_form,
    simplify_with as core_simplify_with, trig_rules,
    verify_antiderivative_status as core_verify_antiderivative_status,
    AlkahestError as AlkahestErrorTrait, AntiderivativeVerification, ApartError,
    AssumptionContext as CoreAssumptionContext, AssumptionError, ComplexF64, DerivedExpr,
    DiffError, EgraphConfig, GaussRat, IntegrationError, IoError,
    LimitDirection as CoreLimitDirection, LimitError, LinearRecurrenceError, PatternRule,
    ProductError, ResidueError, ResultantError, RsolveError, SeriesError, SideCondition,
    SimplifyConfig, SizeCost, SparseGcdError, SparseInterpError, SumError,
};
// Experimental calculus / ODE / transform surface (PyO3 bindings deferred at
// landing time — see PRs #152–#161). These mirror the Rust `experimental`
// re-exports; the Python surface lives under `alkahest.experimental`.
use alkahest_core::calculus::asymptotic::{
    asymptotic_expand as core_asymptotic_expand, AsymptoticError as CoreAsymptoticError,
};
use alkahest_core::calculus::fps::{Fps as CoreFps, FpsError as CoreFpsError};
use alkahest_core::calculus::multilimit::{
    multilimit as core_multilimit, MultiLimit as CoreMultiLimit,
};
use alkahest_core::ode::dsolve::{
    dsolve as core_dsolve, DsolveError as CoreDsolveError, OdeInput as CoreOdeInput,
};
use alkahest_core::ode::numeric::{
    integrate_rk4 as core_integrate_rk4, integrate_rk45 as core_integrate_rk45,
    NumericOdeError as CoreNumericOdeError, OdeTrajectory as CoreOdeTrajectory,
    Rk45Options as CoreRk45Options, Rk4Options as CoreRk4Options,
};
use alkahest_core::ode::series_solve::{
    series_solve as core_series_solve, PointKind as CorePointKind,
    SeriesError as CoreSeriesSolveError, SeriesOde as CoreSeriesOde,
};
use alkahest_core::transform::{
    fourier_transform as core_fourier_transform, inverse_fourier_transform as core_ifourier,
    inverse_laplace_transform as core_ilaplace, inverse_z_transform as core_iztransform,
    laplace_transform as core_laplace, z_transform as core_ztransform,
    FourierError as CoreFourierError, LaplaceError as CoreLaplaceError,
    ZTransformError as CoreZTransformError,
};
// P1 item 9 — rigorous global bounds (Taylor models / validated numerics)
use alkahest_core::validated::bounds::{
    bound_on_box as core_bound_on_box, verified_integral as core_verified_integral,
    verified_no_roots as core_verified_no_roots, verified_sign as core_verified_sign,
    BoundOptions as CoreBoundOptions, IntegralOptions as CoreIntegralOptions,
    SignPredicate as CoreSignPredicate, Verdict as CoreVerdict,
};
use alkahest_core::validated::ValidatedError as CoreValidatedError;
// P1 item 8 — positivity certificates (SOS / Positivstellensatz)
use alkahest_core::real::sos::{
    prove_nonneg as core_prove_nonneg, sos_decompose as core_sos_decompose,
    PositivityCertificate as CorePositivityCertificate, SosError as CoreSosError,
    SosOpts as CoreSosOpts,
};
// P1 item 7 — creative telescoping / holonomic (D-finite) machinery
use alkahest_core::holonomic::{
    boundary_term as core_boundary_term, boundary_verdict as core_boundary_verdict,
    natural_limits as core_natural_limits, zeilberger_search as core_zeilberger_search,
    BoundaryStatus as CoreBoundaryStatus, BoundaryVerdict as CoreBoundaryVerdict,
    HolonomicError as CoreHolonomicError, OrderSearch as CoreOrderSearch,
    ZeilbergerOpts as CoreZeilbergerOpts, ZeilbergerResult as CoreZeilbergerResult,
};
// M4(b) — q-analogue creative telescoping (q-Zeilberger)
use alkahest_core::holonomic::qzeil::{
    cyclotomic_polynomial as core_cyclotomic_polynomial,
    q_specialize_at_root_of_unity as core_q_specialize_at_root_of_unity,
    q_zeilberger as core_q_zeilberger, QBoundaryStatus as CoreQBoundaryStatus,
    QCertificate as CoreQCertificate, QHolonomicError as CoreQHolonomicError,
    QRootOfUnitySpecialization as CoreQRootOfUnitySpecialization,
    QZeilbergerOpts as CoreQZeilbergerOpts,
};
// M4 — double-sum (Apagodu–Zeilberger) creative telescoping
use alkahest_core::holonomic::telescoping2d::{
    boundary_status_2d as core_boundary_status_2d, boundary_status_md as core_boundary_status_md,
    telescope2d_search as core_telescope2d_search, telescope_md_search as core_telescope_md_search,
    Telescoping2dError as CoreTelescoping2dError, Telescoping2dOpts as CoreTelescoping2dOpts,
    Telescoping2dResult as CoreTelescoping2dResult, TelescopingMdOpts as CoreTelescopingMdOpts,
    TelescopingMdResult as CoreTelescopingMdResult,
};
// M6 — modular / p-adic evaluation of holonomic sequences
use alkahest_core::holonomic::modular::{
    binomial_mod as core_binomial_mod, ModularError as CoreHolonomicModularError,
    ModularEvaluation as CoreModularEvaluation, ModularRecurrence as CoreModularRecurrence,
};
// M5 — recurrence -> asymptotics (Poincaré–Perron)
use alkahest_core::holonomic::asymptotics::{
    asymptotics_from_recurrence as core_asymptotics_from_recurrence,
    ConnectionConstant as CoreConnectionConstant,
    RecurrenceAsymptotics as CoreRecurrenceAsymptotics,
};
// P1 item 10 — asymptotic expansion at scale
use alkahest_core::calculus::euler_maclaurin::euler_maclaurin as core_euler_maclaurin;
use alkahest_core::calculus::singularity::coefficient_asymptotics as core_coefficient_asymptotics;
// V3-1 — Integer number theory
use alkahest_core::number_theory::{
    discrete_log as nt_discrete_log, factorint as nt_factorint, isprime as nt_isprime,
    jacobi_symbol as nt_jacobi_symbol, nextprime as nt_nextprime, nthroot_mod as nt_nthroot_mod,
    totient as nt_totient, NumberTheoryError as CoreNumberTheoryError,
    QuadraticDirichlet as CoreQuadraticDirichlet,
};
// The buffer protocol is not in the stable ABI before 3.11, and PyO3 gates
// `pyo3::buffer` on exactly this cfg. The Python layer already falls back to
// `call_batch_raw` when these methods are absent (see `_dlpack._call_batch`),
// so a limited-API 3.9/3.10 build loses the bulk-copy fast path and nothing else.
#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyInt, PyList, PyTuple};
use rug::{Complete, Integer, Rational};
use std::collections::BTreeMap;
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
pyo3::create_exception!(alkahest, PyAssumptionError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyDepthLimitError, PyAlkahestError);

fn depth_error_to_py(e: alkahest_core::DepthLimitError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyDepthLimitError>();
        make_structured_err(py, &exc_type, &e)
    })
}

/// Refuse an expression too deeply nested to recurse over.
///
/// O(1) — the depth is cached on the pool node.  See
/// [`alkahest_core::kernel::depth`] for why every recursive consumer needs
/// this: without it a deep enough argument overflows the native stack, and a
/// stack overflow is a `SIGSEGV`, not an exception, so the caller's
/// `except Exception` never runs.
fn guard_depth(pool: &ExprPool, id: ExprId) -> PyResult<()> {
    alkahest_core::check_expr_depth(pool, id).map_err(depth_error_to_py)
}

/// [`guard_depth`] for an expression still wrapped in its `PyExpr`.
fn guard_expr_depth(py: Python<'_>, expr: &PyExpr) -> PyResult<()> {
    guard_depth(&expr.pool.borrow(py).inner, expr.id)
}

/// Largest `n_pts` a plotting call will accept.
///
/// The renderers hand `n_pts` straight to `Vec::with_capacity`, so without a
/// ceiling a Python `int` becomes either a capacity-overflow panic or an
/// allocation the OOM killer resolves — neither of which a caller can catch.
/// 10 million points is already far past any useful SVG.
const MAX_PLOT_POINTS: usize = 10_000_000;

/// Largest series / Taylor-model order any entry point will accept.
///
/// Order sizes a coefficient vector, and Rust's allocator **aborts** on
/// failure — `SIGABRT`, no unwinding, nothing to catch.  `series(sin(x), x, 0,
/// 2**31 - 1)` did exactly that.  A truncation degree past this is not a
/// computation anyone is waiting for; 2^20 coefficients is already minutes of
/// exact-rational work.
const MAX_SERIES_ORDER: usize = 1 << 20;

/// Reject an order that would size an allocation out of the process.
fn checked_order(what: &str, order: usize) -> PyResult<usize> {
    if order > MAX_SERIES_ORDER {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what} must be at most {MAX_SERIES_ORDER} (got {order})"
        )));
    }
    Ok(order)
}

/// Largest floating-point precision, in bits, any entry point will accept.
///
/// `rug::Float::with_val` **panics** outside `[1, i32::MAX]`, and several
/// call sites double the precision internally, so the ceiling is set two
/// octaves below `i32::MAX` to leave room for that.  16 Mibit is ~5 million
/// decimal digits.
const MAX_PRECISION_BITS: u32 = 1 << 24;

/// Validate a user-supplied precision before it reaches `rug`.
///
/// `Float::with_val(0, x)` and `Float::with_val(huge, x)` both panic, and a
/// panic crossing PyO3 becomes `pyo3_runtime.PanicException`, which derives
/// from `BaseException` — so `except Exception` in a caller's loop does not
/// catch it and the loop dies.  Every entry point taking a `prec` /
/// `precision_bits` argument goes through here instead.
fn checked_prec(prec: u32) -> PyResult<u32> {
    if prec == 0 || prec > MAX_PRECISION_BITS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "precision must be between 1 and {MAX_PRECISION_BITS} bits (got {prec})"
        )));
    }
    Ok(prec)
}

/// Parse a Python ``int`` (any size) into an interned integer expression.
fn integer_into_pool(pool: &ExprPool, n: &Bound<'_, PyAny>) -> PyResult<ExprId> {
    Ok(pool.integer(big_integer_from_py(n)?))
}

/// A Python int of any size as an exact `rug::Integer`.
///
/// `extract::<i64>()` first for the common case, then the decimal string, which
/// is what makes arbitrary precision work. `pool.integer` has accepted bignums
/// this way for a long time; `pool.rational` did not, and took `i64` directly —
/// so `pool.rational(math.factorial(30), 7)` raised `OverflowError: Python int
/// too large to convert to C long` while `pool.integer(math.factorial(30))` was
/// fine. Factorial- and binomial-scale numerators are ordinary in this domain,
/// and the kernel's `ExprPool::rational` already takes `impl Into<rug::Integer>`
/// — only the binding was narrow.
fn big_integer_from_py(n: &Bound<'_, PyAny>) -> PyResult<Integer> {
    if let Ok(v) = n.extract::<i64>() {
        return Ok(Integer::from(v));
    }
    let s = n.str()?.to_string();
    Integer::parse(&s).map(Integer::from).map_err(|_| {
        PyOverflowError::new_err(format!("integer literal out of range or invalid: {s}"))
    })
}

fn pool_mismatch_err() -> PyErr {
    PyPoolError::new_err(
        "expressions belong to different ExprPool instances; combine only symbols \
         and values created from the same pool",
    )
}

/// True when *id* is a literal integer/rational/float zero.
fn expr_is_literal_zero(pool: &ExprPool, id: ExprId) -> bool {
    match pool.get(id) {
        ExprData::Integer(n) => n.0.is_zero(),
        ExprData::Rational(r) => r.0.is_zero(),
        ExprData::Float(f) => f.inner == 0.0,
        _ => false,
    }
}

/// Coerce a substitution value (``Expr``, ``DerivedResult``, ``int``, or ``float``).
fn coerce_substituent(
    pool_py: &Py<PyExprPool>,
    ob: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<ExprId> {
    if let Ok(e) = ob.extract::<PyRef<PyExpr>>() {
        if !e.pool.is(pool_py) {
            return Err(pool_mismatch_err());
        }
        return Ok(e.id);
    }
    if let Ok(dr) = ob.downcast::<PyDerivedResult>() {
        let dr = dr.borrow();
        if !dr.value.pool.is(pool_py) {
            return Err(pool_mismatch_err());
        }
        return Ok(dr.value.id);
    }
    let pool = pool_py.borrow(py);
    if let Ok(n) = ob.extract::<i64>() {
        return Ok(pool.inner.integer(n));
    }
    if let Ok(f) = ob.extract::<f64>() {
        return Ok(pool.inner.float(f, 53));
    }
    integer_into_pool(&pool.inner, ob)
}

fn expr_is_zero(py: Python<'_>, expr: &PyExpr) -> bool {
    let pool = expr.pool.borrow(py);
    expr_is_literal_zero(&pool.inner, expr.id)
}

fn expr_ids_equal(a: &PyExpr, b: &PyExpr) -> bool {
    a.pool.is(&b.pool) && a.id == b.id
}
pyo3::create_exception!(alkahest, PyIntegrationError, PyAlkahestError);
pyo3::create_exception!(alkahest, PySeriesError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyLimitError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyMatrixError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyEigenError, PyMatrixError);
pyo3::create_exception!(alkahest, PyLinearAlgebraError, PyMatrixError);
pyo3::create_exception!(alkahest, PyModularError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyOdeError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyDaeError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyJitError, PyAlkahestError);
pyo3::create_exception!(alkahest, PySolverError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyHomotopyError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyCudaError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyIoError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyParseError, PyAlkahestError);
// V2-7 — Polynomial factorization
pyo3::create_exception!(alkahest, PyFactorError, PyAlkahestError);
// V2-2 — Resultants
pyo3::create_exception!(alkahest, PyResultantError, PyAlkahestError);
// V2-3 — Sparse interpolation
pyo3::create_exception!(alkahest, PySparseInterpError, PyAlkahestError);
pyo3::create_exception!(alkahest, PySparseGcdError, PyAlkahestError);
// V2-4 — Real root isolation
pyo3::create_exception!(alkahest, PyRealRootError, PyAlkahestError);
// V2-6 — LLL + integer relations
pyo3::create_exception!(alkahest, PyLatticeError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyPslqError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyCadError, PyAlkahestError);
// P1 item 9 — rigorous global bounds (Taylor models / validated numerics)
pyo3::create_exception!(alkahest, PyValidatedError, PyAlkahestError);
// P1 item 8 — positivity certificates (SOS / Positivstellensatz)
pyo3::create_exception!(alkahest, PySosError, PyAlkahestError);
pyo3::create_exception!(alkahest, PySumError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyProductError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyNumberTheoryError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyLinearRecurrenceError, PyAlkahestError);
pyo3::create_exception!(alkahest, PyRsolveError, PyAlkahestError);
#[cfg(feature = "groebner")]
pyo3::create_exception!(alkahest, PyDiophantineError, PyAlkahestError);
// M9 — Gröbner bases over Q(params); E-PARAMGB-004 is a refusal, not a fault.
#[cfg(feature = "groebner")]
pyo3::create_exception!(alkahest, PyParamGroebnerError, PyAlkahestError);
// P1 search plumbing item 4 — budgets, cancellation, determinism
pyo3::create_exception!(alkahest, PyBudgetExceededError, PyAlkahestError);
// P1 item 7 — creative telescoping / holonomic (D-finite) machinery
pyo3::create_exception!(alkahest, PyHolonomicError, PyAlkahestError);

/// Build a structured exception with `.code`, `.remediation`, `.span` attributes.
fn make_structured_err<E: AlkahestErrorTrait>(
    _py: Python<'_>,
    exc_type: &pyo3::Bound<'_, pyo3::types::PyType>,
    e: &E,
) -> PyErr {
    let msg = e.to_string();
    let code = e.code();
    let remediation = e.remediation().unwrap_or("");
    // Prefix the stable code so `str(exc)` / logs are greppable without reading `.code`.
    let full_msg = if remediation.is_empty() {
        format!("[{code}] {msg}")
    } else {
        format!("[{code}] {msg}\nRemediation: {remediation}")
    };
    let exc = exc_type.call1((full_msg,)).unwrap();
    exc.setattr("code", code).ok();
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

/// Python-visible symbol domain (matches :class:`alkahest_core::kernel::Domain`).
#[pyclass(name = "Domain")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyDomain {
    #[pyo3(name = "Real")]
    Real,
    #[pyo3(name = "Complex")]
    Complex,
    #[pyo3(name = "Integer")]
    Integer,
    #[pyo3(name = "Positive")]
    Positive,
    #[pyo3(name = "NonNegative")]
    NonNegative,
    #[pyo3(name = "NonZero")]
    NonZero,
}

#[pymethods]
impl PyDomain {
    fn __str__(&self) -> &'static str {
        self.wire_str()
    }

    fn __repr__(&self) -> String {
        format!("Domain.{self:?}")
    }
}

impl PyDomain {
    const fn wire_str(self) -> &'static str {
        match self {
            PyDomain::Real => "real",
            PyDomain::Complex => "complex",
            PyDomain::Integer => "integer",
            PyDomain::Positive => "positive",
            PyDomain::NonNegative => "nonneg",
            PyDomain::NonZero => "nonzero",
        }
    }
}

impl From<PyDomain> for Domain {
    fn from(d: PyDomain) -> Self {
        match d {
            PyDomain::Real => Domain::Real,
            PyDomain::Complex => Domain::Complex,
            PyDomain::Integer => Domain::Integer,
            PyDomain::Positive => Domain::Positive,
            PyDomain::NonNegative => Domain::NonNegative,
            PyDomain::NonZero => Domain::NonZero,
        }
    }
}

impl From<Domain> for PyDomain {
    fn from(d: Domain) -> Self {
        match d {
            Domain::Real => PyDomain::Real,
            Domain::Complex => PyDomain::Complex,
            Domain::Integer => PyDomain::Integer,
            Domain::Positive => PyDomain::Positive,
            Domain::NonNegative => PyDomain::NonNegative,
            Domain::NonZero => PyDomain::NonZero,
        }
    }
}

fn parse_domain_arg(ob: Option<&Bound<'_, PyAny>>) -> PyResult<Domain> {
    let Some(ob) = ob else {
        return Ok(Domain::Real);
    };
    if ob.is_none() {
        return Ok(Domain::Real);
    }
    if let Ok(d) = ob.extract::<PyDomain>() {
        return Ok(d.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return Ok(parse_domain(&s));
    }
    Err(PyTypeError::new_err(
        "domain must be a str (e.g. 'real') or alkahest.Domain",
    ))
}

/// The domain set by the innermost ``alkahest.context(domain=...)`` frame, if any.
///
/// Read back out of the Python thread-local rather than mirrored in a Rust
/// thread-local on purpose: `alkahest._context` owns the context stack (it also
/// has `_overlay`, which pushes frames `context()` never sees), so a second copy
/// of the state here could silently disagree with what `active_domain()` reports.
/// Any failure to reach it — the pure-Python layer not importable, a
/// non-domain value in the frame — degrades to `None`, i.e. the historical
/// `Domain::Real` default.
fn ambient_domain(py: Python<'_>) -> Option<Domain> {
    // Fast path. `pool.symbol` is a hot path — interning is measured directly
    // by `bench_codspeed.py::test_intern_symbol_cached` — and calling into
    // Python to ask for the ambient domain costs ~2 µs per symbol, which made
    // that benchmark 4.6x slower when this lookup was unconditional.
    //
    // `ACTIVE_CONTEXT_FRAMES` counts frames pushed by `alkahest._context`
    // across *all* threads, so it is deliberately conservative: it can say
    // "somebody, somewhere, has a context open" and send a thread that has no
    // context of its own down the slow path, but it can never report zero while
    // a frame is live. Zero therefore means no `context()` or `_overlay()`
    // block is open anywhere and the answer is `None` without touching Python.
    // The stack itself stays the single source of truth — this counts frames,
    // it does not mirror their contents.
    if ACTIVE_CONTEXT_FRAMES.load(Ordering::Acquire) == 0 {
        return None;
    }
    let module = py.import_bound("alkahest._context").ok()?;
    let value = module.call_method0("active_domain").ok()?;
    if value.is_none() {
        return None;
    }
    parse_domain_arg(Some(&value)).ok()
}

/// Number of live `alkahest._context` frames, across every thread.
///
/// Maintained by `_note_context_push` / `_note_context_pop`, which
/// `alkahest._context` calls at the four places it mutates its stack (both
/// `context()` and `_overlay()`, each popping in a `finally`). Only ever read
/// as "is this zero", so an over-count costs a slow path and never a wrong
/// answer.
static ACTIVE_CONTEXT_FRAMES: AtomicUsize = AtomicUsize::new(0);

/// Record that `alkahest._context` pushed a frame. See [`ACTIVE_CONTEXT_FRAMES`].
#[pyfunction]
#[pyo3(name = "_note_context_push")]
fn py_note_context_push() {
    ACTIVE_CONTEXT_FRAMES.fetch_add(1, Ordering::Release);
}

/// Record that `alkahest._context` popped a frame. Saturates at zero so an
/// unbalanced pop cannot wrap the counter into a permanently non-zero state.
#[pyfunction]
#[pyo3(name = "_note_context_pop")]
fn py_note_context_pop() {
    let _ = ACTIVE_CONTEXT_FRAMES.fetch_update(Ordering::Release, Ordering::Acquire, |n| {
        Some(n.saturating_sub(1))
    });
}

/// `parse_domain_arg`, but an absent/`None` argument falls back to the ambient
/// `alkahest.context(domain=...)` before falling back to [`Domain::Real`].
///
/// This is what makes `pool.symbol("x")` and `alkahest.symbol("x")` agree on the
/// sort inside a `context(domain=...)` block; they used to disagree, and the
/// disagreement was invisible until an SMT export declared `Real` where the
/// caller had asked for `Int`.
fn resolve_domain_arg(py: Python<'_>, ob: Option<&Bound<'_, PyAny>>) -> PyResult<Domain> {
    match ob.filter(|o| !o.is_none()) {
        Some(explicit) => parse_domain_arg(Some(explicit)),
        None => Ok(ambient_domain(py).unwrap_or(Domain::Real)),
    }
}

fn py_int_decimal(v: &Bound<'_, PyAny>) -> PyResult<String> {
    v.str()?.extract::<String>()
}

fn diff_error_to_py(e: DiffError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyDiffError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn assumption_error_to_py(e: AssumptionError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyAssumptionError>();
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

fn integrate_error_to_py(e: IntegrationError) -> PyErr {
    // A budget/cancellation trip is not an integration failure — raise the
    // dedicated `BudgetExceededError` (E-BUDGET-*) instead of `IntegrationError`
    // so callers can catch it uniformly regardless of which engine tripped it.
    // (Budget trips are encoded inside `NotImplemented` for Rust semver; see
    // `IntegrationError::is_budget`.)
    if e.is_budget() {
        return Python::with_gil(|py| {
            let exc_type = py.get_type_bound::<PyBudgetExceededError>();
            make_structured_err(py, &exc_type, &e)
        });
    }
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyIntegrationError>();
        make_structured_err(py, &exc_type, &e)
    })
}

// ---------------------------------------------------------------------------
// P1 search plumbing item 4 — budgets, cancellation, determinism
//
// `alkahest_core::budget::enter` returns an RAII guard that must be dropped
// on the same thread it was created on (the underlying stack is
// thread-local). `alkahest.context(budget=...)` is a `@contextmanager`, so
// its `push`/`pop` calls always run on the thread that entered the `with`
// block — there is no `push`/`pop` pair here without a matching thread, and
// no guard object crosses the Python/Rust boundary. Each `push_budget` call
// stores its guard on a Rust-side thread-local stack; `pop_budget` pops and
// drops the most recent one.
// ---------------------------------------------------------------------------

thread_local! {
    static PY_BUDGET_GUARDS: std::cell::RefCell<Vec<alkahest_core::budget::BudgetGuard>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Push a [`alkahest_core::budget::Budget`] onto this thread's active-budget
/// stack. Pair with [`py_pop_budget`]; `alkahest.context(budget=...)` calls
/// both around its `with` block.
#[pyfunction]
#[pyo3(name = "push_budget")]
#[pyo3(signature = (wall_ms=None, max_steps=None, seed=None))]
fn py_push_budget(wall_ms: Option<f64>, max_steps: Option<u64>, seed: Option<u64>) -> PyResult<()> {
    let mut budget = alkahest_core::budget::Budget::new();
    if let Some(ms) = wall_ms {
        // `Duration::from_secs_f64` panics above `u64::MAX` seconds, and
        // `wall_ms=1e30` is a plausible way to spell "effectively unlimited".
        // Saturate at ~100 years instead of dying.
        if !ms.is_finite() || ms < 0.0 {
            return Err(PyValueError::new_err(
                "wall_ms must be a finite, non-negative number of milliseconds",
            ));
        }
        const MAX_WALL_SECS: f64 = 100.0 * 365.0 * 24.0 * 3600.0;
        let secs = (ms / 1000.0).min(MAX_WALL_SECS);
        budget.wall = Some(std::time::Duration::from_secs_f64(secs));
    }
    budget.max_steps = max_steps;
    budget.seed = seed;
    let guard = alkahest_core::budget::enter(budget);
    PY_BUDGET_GUARDS.with(|g| g.borrow_mut().push(guard));
    Ok(())
}

/// Pop the most recently pushed [`alkahest_core::budget::Budget`] from this
/// thread's active-budget stack, dropping its guard.
#[pyfunction]
#[pyo3(name = "pop_budget")]
fn py_pop_budget() -> PyResult<()> {
    let popped = PY_BUDGET_GUARDS.with(|g| g.borrow_mut().pop());
    if popped.is_none() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "pop_budget() called with no active budget on this thread",
        ));
    }
    Ok(())
}

/// `True` if a [`alkahest_core::budget::Budget`] is active on this thread.
#[pyfunction]
#[pyo3(name = "is_budget_active")]
fn py_is_budget_active() -> bool {
    alkahest_core::budget::is_active()
}

/// The seed of the innermost active budget on this thread, or `None`.
#[pyfunction]
#[pyo3(name = "budget_seed")]
fn py_budget_seed() -> Option<u64> {
    alkahest_core::budget::seed()
}

/// Request cancellation of the current cooperative operation(s), process-wide.
#[pyfunction]
#[pyo3(name = "request_cancel")]
fn py_request_cancel() {
    alkahest_core::budget::request_cancel();
}

/// Clear a previously requested cancellation.
#[pyfunction]
#[pyo3(name = "clear_cancel")]
fn py_clear_cancel() {
    alkahest_core::budget::clear_cancel();
}

/// `True` if [`py_request_cancel`] was called and not yet cleared.
#[pyfunction]
#[pyo3(name = "is_cancelled")]
fn py_is_cancelled() -> bool {
    alkahest_core::budget::is_cancelled()
}

fn series_error_to_py(e: SeriesError) -> PyErr {
    // The series engine reports "the requested order is out of reach" as
    // `InvalidOrder` (`SeriesError` is an exhaustive public enum, so it cannot
    // carry a `Truncated` variant without a major semver break) and records the
    // cause out of band. Recover it here so a work-ceiling trip raises
    // `E-SERIES-003` and a budget trip raises the same `BudgetExceededError`
    // (`E-BUDGET-*`) every other engine raises, instead of masquerading as
    // "you passed order 0".
    if matches!(e, SeriesError::InvalidOrder) {
        if let Some(r) = alkahest_core::calculus::series::take_series_refusal() {
            return Python::with_gil(|py| match r.budget() {
                Some(b) => {
                    let exc_type = py.get_type_bound::<PyBudgetExceededError>();
                    make_structured_err(py, &exc_type, &b)
                }
                None => {
                    let exc_type = py.get_type_bound::<PySeriesError>();
                    make_structured_err(py, &exc_type, &r)
                }
            });
        }
    }
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PySeriesError>();
        make_structured_err(py, &exc_type, &e)
    })
}

/// Convert a `PrimaryDecompositionError` into a Python exception, recovering a
/// refusal recorded out of band.
///
/// `radical` and `primary_decomposition` report "I cannot certify this" as
/// `PrimaryDecompositionError::Factorization` (the enum is public and
/// exhaustive, so it cannot grow a `NotCertifiable` variant without a major
/// semver break) and record the real reason for `take_ideal_refusal`. Recover
/// it here so a refusal raises its own `E-IDEAL-005` / `E-IDEAL-006` instead of
/// an uncoded `ValueError` that autoresearch loops cannot branch on.
///
/// `PyAlkahestError` subclasses `ValueError`, so callers catching `ValueError`
/// are unaffected.
#[cfg(feature = "groebner")]
fn ideal_error_to_py(e: alkahest_core::ideal::PrimaryDecompositionError) -> PyErr {
    use alkahest_core::ideal::PrimaryDecompositionError;
    if matches!(e, PrimaryDecompositionError::Factorization(_)) {
        if let Some(r) = alkahest_core::ideal::take_ideal_refusal() {
            return Python::with_gil(|py| {
                let exc_type = py.get_type_bound::<PyAlkahestError>();
                make_structured_err(py, &exc_type, &r)
            });
        }
    }
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyAlkahestError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn limit_error_to_py(e: LimitError) -> PyErr {
    // The limit engine reports a budget/cancellation trip as `DepthExceeded`
    // (`LimitError` is an exhaustive public enum, so it cannot carry a
    // `Budget` variant without a major semver break) and records the cause
    // out-of-band. Recover it here so a budget trip raises the same
    // `BudgetExceededError` (`E-BUDGET-*`) every other engine raises, instead
    // of masquerading as "the limit is too hard".
    if matches!(e, LimitError::DepthExceeded) {
        if let Some(b) = alkahest_core::calculus::limits::last_budget_trip() {
            return Python::with_gil(|py| {
                let exc_type = py.get_type_bound::<PyBudgetExceededError>();
                make_structured_err(py, &exc_type, &b)
            });
        }
    }
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyLimitError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn parse_limit_direction(dir: Option<&str>) -> CoreLimitDirection {
    match dir.unwrap_or("+-").trim() {
        "+" => CoreLimitDirection::Plus,
        "-" => CoreLimitDirection::Minus,
        "+-" | "" => CoreLimitDirection::Bidirectional,
        _ => CoreLimitDirection::Bidirectional,
    }
}

/// Build a `DomainError` with `.code`/`.remediation` attributes for failures
/// that don't come from a type implementing `AlkahestError` (e.g.
/// [`alkahest_core::InterpEvalError`], which is intentionally lightweight
/// since it's an interpreter-internal detail, not a user-facing subsystem).
fn domain_error(py: Python<'_>, code: &str, message: String, remediation: &str) -> PyErr {
    coded_error::<PyDomainError>(py, code, message, remediation)
}

/// [`domain_error`] for any other exception class in the hierarchy.
///
/// Same shape as [`make_structured_err`] — `[CODE] message`, plus `.code`,
/// `.remediation` and `.span` attributes — for failures raised at the Python
/// boundary, where there is no Rust error type implementing `AlkahestError` to
/// hand it. The alternative is letting whatever PyO3 happened to produce
/// escape, which is how `residue` came to raise a bare `AttributeError`: not
/// an `AlkahestError`, so invisible to `except ak.AlkahestError`, and carrying
/// no code for a caller to branch on.
fn coded_error<E>(py: Python<'_>, code: &str, message: String, remediation: &str) -> PyErr
where
    E: pyo3::type_object::PyTypeInfo,
{
    let exc_type = py.get_type_bound::<E>();
    let full_msg = format!("[{code}] {message}\nRemediation: {remediation}");
    let exc = exc_type.call1((full_msg,)).unwrap();
    exc.setattr("code", code).ok();
    exc.setattr("remediation", remediation).ok();
    exc.setattr("span", py.None()).ok();
    PyErr::from_value_bound(exc)
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

fn sparse_gcd_error_to_py(e: SparseGcdError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PySparseGcdError>();
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

fn product_error_to_py(e: ProductError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyProductError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn number_theory_error_to_py(e: CoreNumberTheoryError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyNumberTheoryError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn linear_recurrence_error_to_py(e: LinearRecurrenceError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyLinearRecurrenceError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn rsolve_error_to_py(e: RsolveError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyRsolveError>();
        make_structured_err(py, &exc_type, &e)
    })
}

#[cfg(feature = "groebner")]
fn diophantine_error_to_py(e: DiophantineError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyDiophantineError>();
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
    // `SingularMatrix` covers two refusals: a determinant proven zero
    // (`E-MAT-003`) and one that could be proven neither way (`E-MAT-004`).
    // `MatrixError` is an exhaustive public enum, so the second cannot have its
    // own variant without a major semver break; it is recorded out of band
    // instead — the same arrangement `limit_error_to_py` uses for budget trips.
    if matches!(e, MatrixError::SingularMatrix) {
        if let Some(r) = alkahest_core::matrix::take_zero_test_refusal() {
            return Python::with_gil(|py| {
                let exc_type = py.get_type_bound::<PyMatrixError>();
                make_structured_err(py, &exc_type, &r)
            });
        }
    }
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyMatrixError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn eigen_error_to_py(e: EigenError) -> PyErr {
    // `KernelComputationFailed` reaches Python from two places: the eigenvector
    // nullspace refused because one entry's vanishing is undecidable
    // (`E-LINALG-010` — fixable by substituting concrete parameters), or the
    // computed columns did not assemble (`E-EIGEN-006`). `EigenError` is an
    // exhaustive public enum, so the first travels out of band exactly as it
    // does for `nullspace`; see `linear_algebra_error_to_py`. The exception
    // class stays `EigenError` — only the code and the message get specific.
    if matches!(e, EigenError::KernelComputationFailed) {
        if let Some(r) = alkahest_core::matrix::take_zero_test_refusal() {
            return Python::with_gil(|py| {
                let exc_type = py.get_type_bound::<PyEigenError>();
                make_structured_err(py, &exc_type, &r)
            });
        }
    }
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyEigenError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn linear_algebra_error_to_py(e: LinearAlgebraError) -> PyErr {
    // `UnsupportedField` covers both "entries are not rational constants"
    // (`E-LINALG-007`) and "a pivot could be proven neither zero nor non-zero"
    // (`E-LINALG-010`); see `matrix_error_to_py` for why the second has no
    // variant of its own.
    if matches!(e, LinearAlgebraError::UnsupportedField) {
        if let Some(r) = alkahest_core::matrix::take_zero_test_refusal() {
            return Python::with_gil(|py| {
                let exc_type = py.get_type_bound::<PyLinearAlgebraError>();
                make_structured_err(py, &exc_type, &r)
            });
        }
    }
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyLinearAlgebraError>();
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

    /// Free symbol. ``domain`` defaults to the ambient
    /// ``alkahest.context(domain=...)`` and only then to ``Domain.Real``.
    ///
    /// The fallback exists because the domain is not decoration: it picks the
    /// SMT-LIB sort (``Int`` vs ``Real``) and hence the logic (``QF_NIA`` vs
    /// ``QF_NRA``), so a pool symbol built inside a ``domain=Domain.Integer``
    /// block used to change the question being asked without changing any
    /// status a caller reads. Pass ``domain="real"`` explicitly to opt out.
    #[pyo3(signature = (name, domain=None, commutative=None))]
    fn symbol(
        slf: PyRef<'_, Self>,
        name: &str,
        domain: Option<&Bound<'_, PyAny>>,
        commutative: Option<bool>,
    ) -> PyResult<PyExpr> {
        let commutative = commutative.unwrap_or(true);
        let dom = resolve_domain_arg(slf.py(), domain)?;
        let id = slf.inner.symbol_commutative(name, dom, commutative);
        let pool: Py<PyExprPool> = slf.into();
        Ok(PyExpr { id, pool })
    }

    /// Canonical imaginary unit `I` (`Domain.Complex` symbol named ``"I"``).
    ///
    /// In ``evaluate(..., mode="complex")`` this auto-binds to ``1j`` when
    /// unbound. Symbolic simplification folds ``I**2 → -1``, etc.
    fn imaginary_unit(slf: PyRef<'_, Self>) -> PyExpr {
        let id = slf.inner.imaginary_unit();
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    fn integer(slf: PyRef<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let id = integer_into_pool(&slf.inner, n)?;
        let pool: Py<PyExprPool> = slf.into();
        Ok(PyExpr { id, pool })
    }

    /// Exact rational `p/q`. Both take Python ints of any size, as
    /// [`integer`] does — see `big_integer_from_py`.
    fn rational(
        slf: PyRef<'_, Self>,
        p: &Bound<'_, PyAny>,
        q: &Bound<'_, PyAny>,
    ) -> PyResult<PyExpr> {
        let (num, den) = (big_integer_from_py(p)?, big_integer_from_py(q)?);
        if den == 0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "pool.rational(p, q) needs a non-zero denominator",
            ));
        }
        let id = slf.inner.rational(num, den);
        let pool: Py<PyExprPool> = slf.into();
        Ok(PyExpr { id, pool })
    }

    /// Apply a named primitive or symbolic function: ``pool.func("sin", [x])``, ``pool.func("f", [n])``.
    #[pyo3(name = "func")]
    fn apply_named(slf: PyRef<'_, Self>, name: &str, args: Vec<PyExpr>) -> PyExpr {
        let ids: Vec<ExprId> = args.iter().map(|e| e.id).collect();
        let id = slf.inner.func(name, ids);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    /// Build an addition node: ``pool.add([x, y, z])`` → `x + y + z`.
    ///
    /// Children are sorted canonically so ``pool.add([b, a]) == pool.add([a, b])``.
    fn add(slf: PyRef<'_, Self>, args: Vec<PyExpr>) -> PyExpr {
        let ids: Vec<ExprId> = args.iter().map(|e| e.id).collect();
        let id = slf.inner.add(ids);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    /// Build a multiplication node: ``pool.mul([x, y, z])`` → `x * y * z`.
    fn mul(slf: PyRef<'_, Self>, args: Vec<PyExpr>) -> PyExpr {
        let ids: Vec<ExprId> = args.iter().map(|e| e.id).collect();
        let id = slf.inner.mul(ids);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    fn float(slf: PyRef<'_, Self>, value: f64, prec: Option<u32>) -> PyResult<PyExpr> {
        let prec = checked_prec(prec.unwrap_or(53))?;
        let id = slf.inner.float(value, prec);
        let pool: Py<PyExprPool> = slf.into();
        Ok(PyExpr { id, pool })
    }

    /// `O(arg)` — Landau remainder bound (V2-15 series API).
    fn big_o(slf: PyRef<'_, Self>, arg: PyExpr) -> PyExpr {
        let id = slf.inner.big_o(arg.id);
        let pool: Py<PyExprPool> = slf.into();
        PyExpr { id, pool }
    }

    /// Canonical `+∞` for ``limit(..., oo)`` (Unicode ∞, V2-16).
    fn pos_infinity(slf: PyRef<'_, Self>) -> PyExpr {
        let id = slf.inner.pos_infinity();
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

    // Every renderer below walks the expression recursively, so each one is a
    // stack overflow — i.e. a `SIGSEGV`, not an exception — on a deep enough
    // tree.  `guard_depth` turns that into a catchable `DepthLimitError`.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let pool = self.pool.borrow(py);
        guard_depth(&pool.inner, self.id)?;
        Ok(pool.inner.display(self.id).to_string())
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let pool = self.pool.borrow(py);
        guard_depth(&pool.inner, self.id)?;
        Ok(pool.inner.display(self.id).to_string())
    }

    fn display_latex(&self, py: Python<'_>) -> PyResult<String> {
        let pool = self.pool.borrow(py);
        guard_depth(&pool.inner, self.id)?;
        Ok(alkahest_core::render_latex(self.id, &pool.inner))
    }

    fn display_unicode(&self, py: Python<'_>) -> PyResult<String> {
        let pool = self.pool.borrow(py);
        guard_depth(&pool.inner, self.id)?;
        Ok(alkahest_core::render_unicode(self.id, &pool.inner))
    }

    // ------------------------------------------------------------------
    // Arithmetic — accept Expr, int, or float on the right-hand side.
    // Return py.NotImplemented() for unrecognised types so Python can
    // try the reflected operation on the other operand.
    // ------------------------------------------------------------------

    fn __add__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        match self.coerce_scalar(other, py)? {
            Some(rhs) => {
                let id = self.pool.borrow(py).inner.add(vec![self.id, rhs]);
                Ok(PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py))
            }
            None => Ok(py.NotImplemented()),
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        match self.coerce_scalar(other, py)? {
            Some(lhs) => {
                let id = self.pool.borrow(py).inner.add(vec![lhs, self.id]);
                Ok(PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py))
            }
            None => Ok(py.NotImplemented()),
        }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        match self.coerce_scalar(other, py)? {
            Some(rhs) => {
                let pool = self.pool.borrow(py);
                let neg_one = pool.inner.integer(-1i32);
                let neg_rhs = pool.inner.mul(vec![neg_one, rhs]);
                let id = pool.inner.add(vec![self.id, neg_rhs]);
                drop(pool);
                Ok(PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py))
            }
            None => Ok(py.NotImplemented()),
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        // other - self
        match self.coerce_scalar(other, py)? {
            Some(lhs) => {
                let pool = self.pool.borrow(py);
                let neg_one = pool.inner.integer(-1i32);
                let neg_self = pool.inner.mul(vec![neg_one, self.id]);
                let id = pool.inner.add(vec![lhs, neg_self]);
                drop(pool);
                Ok(PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py))
            }
            None => Ok(py.NotImplemented()),
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        match self.coerce_scalar(other, py)? {
            Some(rhs) => {
                let id = self.pool.borrow(py).inner.mul(vec![self.id, rhs]);
                Ok(PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py))
            }
            None => Ok(py.NotImplemented()),
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        match self.coerce_scalar(other, py)? {
            Some(lhs) => {
                let id = self.pool.borrow(py).inner.mul(vec![lhs, self.id]);
                Ok(PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py))
            }
            None => Ok(py.NotImplemented()),
        }
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        match self.coerce_scalar(other, py)? {
            Some(rhs) => {
                let pool = self.pool.borrow(py);
                if expr_is_literal_zero(&pool.inner, rhs) {
                    return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "division by zero",
                    ));
                }
                let neg_one = pool.inner.integer(-1i32);
                let inv = pool.inner.pow(rhs, neg_one);
                let id = pool.inner.mul(vec![self.id, inv]);
                drop(pool);
                Ok(PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py))
            }
            None => Ok(py.NotImplemented()),
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        // other / self  =  other * self^-1
        match self.coerce_scalar(other, py)? {
            Some(lhs) => {
                let pool = self.pool.borrow(py);
                if expr_is_literal_zero(&pool.inner, self.id) {
                    return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "division by zero",
                    ));
                }
                let neg_one = pool.inner.integer(-1i32);
                let inv_self = pool.inner.pow(self.id, neg_one);
                let id = pool.inner.mul(vec![lhs, inv_self]);
                drop(pool);
                Ok(PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                }
                .into_py(py))
            }
            None => Ok(py.NotImplemented()),
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

    fn __pow__(
        &self,
        exp: &Bound<'_, PyAny>,
        _modulo: Option<PyObject>,
        py: Python<'_>,
    ) -> PyObject {
        // Accept Python int/float literals and Expr exponents.
        let pool = self.pool.borrow(py);
        let exp_id = if let Ok(n) = exp.extract::<i64>() {
            pool.inner.integer(n)
        } else if let Ok(x) = exp.extract::<f64>() {
            // IEEE float literal → pool float node (complex eval uses principal Log).
            pool.inner.float(x, 53)
        } else if let Ok(expr_ref) = exp.extract::<PyRef<PyExpr>>() {
            expr_ref.id
        } else {
            drop(pool);
            return py.NotImplemented();
        };
        let id = pool.inner.pow(self.id, exp_id);
        drop(pool);
        PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        }
        .into_py(py)
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
            alkahest_core::ExprData::BigO(inner) => {
                PyList::new_bound(py, vec!["big_o".into_py(py), wrap!(inner)]).into_py(py)
            }
            alkahest_core::ExprData::RootSum { poly, var, body } => PyList::new_bound(
                py,
                vec!["root_sum".into_py(py), wrap!(poly), wrap!(var), wrap!(body)],
            )
            .into_py(py),
        }
    }
}

// Non-pymethod helpers for PyExpr.
impl PyExpr {
    // Coerce a Python scalar (Expr | int | float) to an interned ExprId.
    // Returns None for unrecognised types so callers can return NotImplemented.
    fn coerce_scalar(&self, ob: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Option<ExprId>> {
        if let Ok(e) = ob.extract::<PyRef<PyExpr>>() {
            if !e.pool.is(&self.pool) {
                return Err(pool_mismatch_err());
            }
            return Ok(Some(e.id));
        }
        let pool = self.pool.borrow(py);
        if let Ok(n) = ob.extract::<i64>() {
            return Ok(Some(pool.inner.integer(n)));
        }
        if let Ok(f) = ob.extract::<f64>() {
            return Ok(Some(pool.inner.float(f, 53)));
        }
        if ob.is_instance_of::<PyInt>() {
            return Ok(Some(integer_into_pool(&pool.inner, ob)?));
        }
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// V2-15 — Truncated series / Laurent + BigO
// ---------------------------------------------------------------------------

#[pyclass(name = "Series")]
#[derive(Clone)]
struct PySeries {
    expr: PyExpr,
}

#[pymethods]
impl PySeries {
    #[getter]
    fn expr(&self) -> PyExpr {
        self.expr.clone()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.expr.pool.borrow(py);
        format!("Series({})", pool.inner.display(self.expr.id))
    }

    fn __str__(&self, py: Python<'_>) -> String {
        let pool = self.expr.pool.borrow(py);
        pool.inner.display(self.expr.id).to_string()
    }
}

// ---------------------------------------------------------------------------
// PyFps — lazy formal power series over ℚ (experimental, PR #155)
// ---------------------------------------------------------------------------

/// Coerce a Python `int`, `fractions.Fraction`, or `(numer, denom)` tuple into a
/// rug `Rational`. (Floats are intentionally rejected — Fps is exact over ℚ.)
fn py_to_rational(ob: &Bound<'_, PyAny>) -> PyResult<Rational> {
    if let Ok(v) = ob.extract::<i64>() {
        return Ok(Rational::from(v));
    }
    if let Ok((n, d)) = ob.extract::<(i64, i64)>() {
        if d == 0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Fps coefficient denominator is zero",
            ));
        }
        return Ok(Rational::from((Integer::from(n), Integer::from(d))));
    }
    // Fraction-like: has integer `numerator` / `denominator` attributes.
    if let (Ok(n), Ok(d)) = (ob.getattr("numerator"), ob.getattr("denominator")) {
        let ns = n.str()?.to_string();
        let ds = d.str()?.to_string();
        let nz = Integer::parse(&ns)
            .map_err(|_| PyTypeError::new_err(format!("invalid numerator: {ns}")))?;
        let dz = Integer::parse(&ds)
            .map_err(|_| PyTypeError::new_err(format!("invalid denominator: {ds}")))?;
        let nz = Integer::from(nz);
        let dz = Integer::from(dz);
        if dz == 0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Fps coefficient denominator is zero",
            ));
        }
        return Ok(Rational::from((nz, dz)));
    }
    // Bare big integer.
    let s = ob.str()?.to_string();
    let z = Integer::parse(&s)
        .map_err(|_| PyTypeError::new_err(format!("cannot coerce {s} to a rational")))?;
    Ok(Rational::from(Integer::from(z)))
}

fn py_seq_to_rationals(seq: &Bound<'_, PyAny>) -> PyResult<Vec<Rational>> {
    let mut out = Vec::new();
    for item in seq.iter()? {
        out.push(py_to_rational(&item?)?);
    }
    Ok(out)
}

/// Lazy formal power series `∑ aₙ xⁿ` over ℚ with memoized coefficients
/// (experimental; mirrors the Rust :rust:`Fps`).
///
/// Exact-rational only — coefficients are Python `int` / `fractions.Fraction`.
/// Expression-backed series (`Fps.from_expr`) are snapshotted to a finite order
/// at construction (the Rust `Fps<'p>` borrows the pool, which cannot cross the
/// Python boundary), so coefficients past that order read as `0`; pass a larger
/// `order` for deeper work. All other constructors and operations are fully lazy
/// over `Fps<'static>`.
#[pyclass(name = "Fps", unsendable)]
#[derive(Clone)]
struct PyFps {
    inner: CoreFps<'static>,
}

#[pymethods]
impl PyFps {
    /// Series from explicit ascending rational coefficients of a polynomial.
    #[staticmethod]
    fn from_poly(coeffs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let cs = py_seq_to_rationals(coeffs)?;
        Ok(PyFps {
            inner: CoreFps::from_poly(&cs),
        })
    }

    /// Series of `p(x)/q(x)` from ascending coefficient lists `num` / `den`
    /// (requires `den[0] != 0`).
    #[staticmethod]
    fn from_rational(num: &Bound<'_, PyAny>, den: &Bound<'_, PyAny>) -> PyResult<Self> {
        let n = py_seq_to_rationals(num)?;
        let d = py_seq_to_rationals(den)?;
        Ok(PyFps {
            inner: CoreFps::from_rational(&n, &d).map_err(fps_error_to_py)?,
        })
    }

    /// Snapshot the series of `expr` in `var` about `0` to `order` coefficients.
    ///
    /// Coefficients of index `>= order` read as `0` (see the class note).
    #[staticmethod]
    #[pyo3(signature = (expr, var, order=32))]
    fn from_expr(
        py: Python<'_>,
        expr: PyRef<PyExpr>,
        var: PyRef<PyExpr>,
        order: usize,
    ) -> PyResult<Self> {
        let order = checked_order("Fps order", order)?;
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let fps = CoreFps::from_expr(expr.id, var.id, &pool.inner).map_err(fps_error_to_py)?;
        let coeffs = fps.coeffs(order);
        Ok(PyFps {
            inner: CoreFps::from_poly(&coeffs),
        })
    }

    /// The zero series.
    #[staticmethod]
    fn zero() -> Self {
        PyFps {
            inner: CoreFps::zero(),
        }
    }

    /// The constant series `c`.
    #[staticmethod]
    fn constant(c: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyFps {
            inner: CoreFps::constant(py_to_rational(c)?),
        })
    }

    /// The series `x`.
    #[staticmethod]
    fn x() -> Self {
        PyFps {
            inner: CoreFps::x(),
        }
    }

    /// `exp(x) = ∑ xⁿ/n!`.
    #[staticmethod]
    fn exp_series() -> Self {
        PyFps {
            inner: CoreFps::exp_series(),
        }
    }

    /// `sin(x) = ∑ (−1)ᵏ x^{2k+1}/(2k+1)!`.
    #[staticmethod]
    fn sin_series() -> Self {
        PyFps {
            inner: CoreFps::sin_series(),
        }
    }

    /// `cos(x) = ∑ (−1)ᵏ x^{2k}/(2k)!`.
    #[staticmethod]
    fn cos_series() -> Self {
        PyFps {
            inner: CoreFps::cos_series(),
        }
    }

    /// `log(1+x) = ∑_{n≥1} (−1)^{n+1} xⁿ/n`.
    #[staticmethod]
    fn log1p_series() -> Self {
        PyFps {
            inner: CoreFps::log1p_series(),
        }
    }

    /// `atan(x) = ∑_{k≥0} (−1)ᵏ x^{2k+1}/(2k+1)`.
    #[staticmethod]
    fn atan_series() -> Self {
        PyFps {
            inner: CoreFps::atan_series(),
        }
    }

    /// Binomial series `(1+x)^α = ∑ C(α,n) xⁿ` for rational `α`.
    #[staticmethod]
    fn binomial_series(alpha: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyFps {
            inner: CoreFps::binomial_series(py_to_rational(alpha)?),
        })
    }

    /// The `n`-th coefficient `aₙ` as a Python `int` / `Fraction`.
    fn coeff(&self, py: Python<'_>, n: usize) -> PyResult<PyObject> {
        // The memoising probe behind `coeff` sizes a coefficient vector from
        // `n`; an unchecked Python int aborts the process in the allocator.
        let n = checked_order("Fps coefficient index", n)?;
        rational_to_py(py, &self.inner.coeff(n))
    }

    /// The first `n` coefficients `[a₀, …, a_{n-1}]`.
    fn coeffs(&self, py: Python<'_>, n: usize) -> PyResult<PyObject> {
        let n = checked_order("Fps coefficient count", n)?;
        let out = PyList::empty_bound(py);
        for c in self.inner.coeffs(n) {
            out.append(rational_to_py(py, &c)?)?;
        }
        Ok(out.into_py(py))
    }

    /// Truncate to a symbolic `Expr` of degree `< order` in `var` (with an
    /// `O(varᵒʳᵈᵉʳ)` tail).
    fn to_expr(&self, py: Python<'_>, var: PyRef<PyExpr>, order: u32) -> PyResult<PyExpr> {
        checked_order("Fps order", order as usize)?;
        let pool_py = var.pool.clone_ref(py);
        let id = {
            let pool = pool_py.borrow(py);
            self.inner.to_expr(var.id, order, &pool.inner)
        };
        Ok(PyExpr { id, pool: pool_py })
    }

    /// Sum `self + other`.
    fn add(&self, other: &PyFps) -> Self {
        PyFps {
            inner: self.inner.add(&other.inner),
        }
    }

    /// Difference `self - other`.
    fn sub(&self, other: &PyFps) -> Self {
        PyFps {
            inner: self.inner.sub(&other.inner),
        }
    }

    /// Cauchy product `self * other`.
    fn mul(&self, other: &PyFps) -> Self {
        PyFps {
            inner: self.inner.mul(&other.inner),
        }
    }

    /// Scale every coefficient by the rational `c`.
    fn scale(&self, c: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyFps {
            inner: self.inner.scale(py_to_rational(c)?),
        })
    }

    /// Quotient `self / other` (requires `other(0) != 0`).
    fn div(&self, other: &PyFps) -> PyResult<Self> {
        Ok(PyFps {
            inner: self.inner.div(&other.inner).map_err(fps_error_to_py)?,
        })
    }

    /// Multiplicative inverse `1/self` (requires `self(0) != 0`).
    fn inverse(&self) -> PyResult<Self> {
        Ok(PyFps {
            inner: self.inner.inverse().map_err(fps_error_to_py)?,
        })
    }

    /// Composition `self ∘ g` (requires `g(0) = 0`).
    fn compose(&self, g: &PyFps) -> PyResult<Self> {
        Ok(PyFps {
            inner: self.inner.compose(&g.inner).map_err(fps_error_to_py)?,
        })
    }

    /// Compositional inverse (reversion) of `self` (requires `self(0) = 0`,
    /// `self'(0) != 0`).
    fn revert(&self) -> PyResult<Self> {
        Ok(PyFps {
            inner: self.inner.revert().map_err(fps_error_to_py)?,
        })
    }

    /// Formal derivative.
    fn derivative(&self) -> Self {
        PyFps {
            inner: self.inner.derivative(),
        }
    }

    /// Formal integral (zero constant term).
    fn integral(&self) -> Self {
        PyFps {
            inner: self.inner.integral(),
        }
    }

    fn __repr__(&self) -> String {
        "Fps(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Explicit assumptions
// ---------------------------------------------------------------------------

/// Pool-scoped assumptions for conservative simplification (stable top-level).
#[pyclass(name = "Assumptions")]
struct PyAssumptions {
    pool: Py<PyExprPool>,
    inner: CoreAssumptionContext,
}

#[pymethods]
impl PyAssumptions {
    #[new]
    fn new(pool: Py<PyExprPool>) -> Self {
        Self {
            pool,
            inner: CoreAssumptionContext::new(),
        }
    }

    /// Add a predicate to this context.
    ///
    /// Only positive and non-zero facts authorize conditional rewrites. Other
    /// predicates remain provenance for contradiction detection.
    fn refine(&mut self, py: Python<'_>, predicate: PyRef<PyExpr>) -> PyResult<()> {
        if !predicate.pool.is(&self.pool) {
            return Err(pool_mismatch_err());
        }
        let pool = self.pool.borrow(py);
        self.inner
            .refine(predicate.id, &pool.inner)
            .map_err(assumption_error_to_py)
    }

    /// Simplify an expression under this explicit assumption context.
    fn simplify(&self, py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
        if !expr.pool.is(&self.pool) {
            return Err(pool_mismatch_err());
        }
        let derived = {
            let pool = self.pool.borrow(py);
            self.inner.simplify(expr.id, &pool.inner)
        };
        Ok(make_derived_result(
            py,
            derived,
            self.pool.clone_ref(py),
            None,
        ))
    }

    /// True when this context has a strict-positivity fact for `expr` —
    /// either an explicit `refine(x > 0)` or `expr` itself being declared
    /// with `Domain.Positive`.
    ///
    /// Agent-facing helper for composing assumptions with other APIs (e.g.
    /// filtering `solve` roots) without exposing the internal `SideCondition`
    /// representation.
    fn is_positive(&self, py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<bool> {
        if !expr.pool.is(&self.pool) {
            return Err(pool_mismatch_err());
        }
        if self
            .inner
            .facts()
            .iter()
            .any(|f| matches!(f, SideCondition::Positive(id) if *id == expr.id))
        {
            return Ok(true);
        }
        let pool = self.pool.borrow(py);
        Ok(matches!(
            pool.inner.get(expr.id),
            ExprData::Symbol {
                domain: Domain::Positive,
                ..
            }
        ))
    }

    /// Predicates explicitly asserted in this context.
    #[getter]
    fn predicates<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let predicates = PyList::empty_bound(py);
        for &id in self.inner.predicates() {
            predicates
                .append(
                    Py::new(
                        py,
                        PyExpr {
                            id,
                            pool: self.pool.clone_ref(py),
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        predicates
    }
}

// ---------------------------------------------------------------------------
// PyDerivedResult
// ---------------------------------------------------------------------------

// Result envelope schema versions (P1 search-plumbing item 6).
//
// Bump `RESULT_SCHEMA_VERSION` for any change to the *envelope* shape
// returned by `DerivedResult.to_dict` / `to_json` (added/removed/renamed
// top-level keys: `kind`, `value`, `verification`, `certificate_status`,
// `steps`, `has_certificate`, ...). Bump `STEPS_SCHEMA_VERSION` for any
// change to the shape of a single `steps` entry (full-mode field names
// `rule`/`before`/`after`/`side_conditions`, or the compact-mode short-key
// mapping `r`/`s`). These are independent so a caller that only reads
// `.value` / `.verification` doesn't need to re-check its parsing when a
// `steps` internal detail changes, and vice versa. Both are exposed as
// module-level attributes (`alkahest.RESULT_SCHEMA_VERSION`) and as
// `DerivedResult` class attributes; see `docs/mdbook/src/derivations.md`.
const RESULT_SCHEMA_VERSION: u32 = 1;
const STEPS_SCHEMA_VERSION: u32 = 1;

#[pyclass(name = "DerivedResult")]
struct PyDerivedResult {
    value: PyExpr,
    derivation: String,
    steps_raw: Vec<(String, String, String, Vec<String>)>,
    raw: DerivedExpr<ExprId>,
    /// Differentiation variable when this result comes from :func:`diff`.
    wrt: Option<ExprId>,
    /// Integrand and variable for lazy exact antiderivative verification.
    integration_verification_input: Option<(ExprId, ExprId)>,
    /// `(integrand, var, lower, upper)` for a definite integral, used to emit an
    /// interval-FTC Lean certificate (`∫ a..b f = F b − F a`) lazily.
    definite_integration_input: Option<(ExprId, ExprId, ExprId, ExprId)>,
}

#[pymethods]
impl PyDerivedResult {
    /// Envelope schema version for :meth:`to_dict` / :meth:`to_json`. See
    /// module-level ``alkahest.RESULT_SCHEMA_VERSION``.
    #[classattr]
    const SCHEMA_VERSION: u32 = crate::RESULT_SCHEMA_VERSION;

    /// Schema version of each entry in ``.steps`` / the ``steps`` key of
    /// :meth:`to_dict`. See module-level ``alkahest.STEPS_SCHEMA_VERSION``.
    #[classattr]
    const STEPS_SCHEMA_VERSION: u32 = crate::STEPS_SCHEMA_VERSION;

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

    /// Lean 4 proof certificate (``.lean`` source), when the derivation log is
    /// certifiable without ``sorry`` and without false unwrapped equalities.
    ///
    /// Returns ``None`` when the log is empty, records an integration (where
    /// ``integrand = F`` would be false), or would require an admission (B3).
    #[getter]
    fn certificate(&self, py: Python<'_>) -> Option<String> {
        // Integration derivation logs construct antiderivatives; emitting them
        // as rewrite equalities is unsound (e.g. `sin x = -cos x`). They are
        // instead certified via the FTC derivative relation
        // `deriv (fun x => F) x = f` (Part A).
        if let Some((integrand, var)) = self.integration_verification_input {
            let pool_py = self.value.pool.clone_ref(py);
            let pool = pool_py.borrow(py);
            let src =
                alkahest_core::emit_integration_cert(self.raw.value, integrand, var, &pool.inner);
            if src.is_empty() || src.contains("sorry") || src.contains("admit") {
                return None;
            }
            return Some(src);
        }
        // Definite integrals certify the sound interval-FTC equation
        // `∫ x in a..b, f x = F b - F a` (never emitted as a false rewrite).
        if let Some((integrand, var, lower, upper)) = self.definite_integration_input {
            let pool_py = self.value.pool.clone_ref(py);
            let pool = pool_py.borrow(py);
            let src = alkahest_core::emit_definite_integration_cert(
                integrand,
                var,
                lower,
                upper,
                &pool.inner,
            );
            if src.is_empty() || src.contains("sorry") || src.contains("admit") {
                return None;
            }
            return Some(src);
        }
        if self.raw.log.is_empty() {
            return None;
        }
        let pool_py = self.value.pool.clone_ref(py);
        let pool = pool_py.borrow(py);
        let src = alkahest_core::emit_lean_expr_wrt(&self.raw, &pool.inner, self.wrt);
        if src.is_empty() || src.contains("sorry") || src.contains("admit") {
            return None;
        }
        Some(src)
    }

    /// Why this result does (or does not) carry a Lean certificate.
    ///
    /// Returns a dict with:
    ///
    /// * ``certifiable`` — ``True`` iff :attr:`certificate` is not ``None``.
    /// * ``reason`` — stable reason code: ``"emitted"``,
    ///   ``"withheld_no_derivation"`` (nothing was rewritten, so there is
    ///   nothing to prove), ``"withheld_integration_shape"`` (the integral
    ///   falls outside the FTC fragment the emitter can encode), or
    ///   ``"withheld_uncertifiable_step"``.
    /// * ``blocking_steps`` — for ``withheld_uncertifiable_step``, the list of
    ///   ``{"index", "rule", "before", "after"}`` records whose rewrite rule has
    ///   no sound Lean encoding today. This is a *diagnostic*: an empty list
    ///   with ``certifiable=False`` means the whole-derivation gate withheld
    ///   for a reason not attributable to a single step.
    ///
    /// The certifiability boundary this reports is exactly the one tabulated by
    /// :func:`alkahest.certificate_coverage`.
    #[getter]
    fn certificate_status<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let status = PyDict::new_bound(py);
        let certifiable = self.certificate(py).is_some();
        let blocking = PyList::empty_bound(py);

        let reason = if certifiable {
            "emitted"
        } else if self.integration_verification_input.is_some()
            || self.definite_integration_input.is_some()
        {
            "withheld_integration_shape"
        } else if self.raw.log.is_empty() {
            "withheld_no_derivation"
        } else {
            let pool_py = self.value.pool.clone_ref(py);
            let pool = pool_py.borrow(py);
            for (index, step) in self.raw.log.steps().iter().enumerate() {
                if alkahest_core::step_is_certifiable(step, self.wrt, &pool.inner) {
                    continue;
                }
                let record = PyDict::new_bound(py);
                record.set_item("index", index).unwrap();
                record.set_item("rule", step.rule_name).unwrap();
                record
                    .set_item("before", pool.inner.display(step.before).to_string())
                    .unwrap();
                record
                    .set_item("after", pool.inner.display(step.after).to_string())
                    .unwrap();
                blocking.append(record).unwrap();
            }
            "withheld_uncertifiable_step"
        };

        status.set_item("certifiable", certifiable).unwrap();
        status.set_item("reason", reason).unwrap();
        status.set_item("blocking_steps", blocking).unwrap();
        status
    }

    /// Machine-readable evidence metadata for this derived result.
    ///
    /// A generated Lean source artifact is not a statement that Lean has
    /// checked it. An indefinite integration result is ``exactly_verified``
    /// only when its symbolic derivative residual simplifies to zero.
    #[getter]
    fn verification<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let metadata = PyDict::new_bound(py);
        let lean_certificate = {
            if let Some((integrand, var)) = self.integration_verification_input {
                // Integrals certify via the FTC derivative relation (Part A).
                let pool_py = self.value.pool.clone_ref(py);
                let pool = pool_py.borrow(py);
                let src = alkahest_core::emit_integration_cert(
                    self.raw.value,
                    integrand,
                    var,
                    &pool.inner,
                );
                if src.is_empty() || src.contains("sorry") || src.contains("admit") {
                    None
                } else {
                    Some(src)
                }
            } else if let Some((integrand, var, lower, upper)) = self.definite_integration_input {
                // Definite integrals certify via the interval FTC (Part A).
                let pool_py = self.value.pool.clone_ref(py);
                let pool = pool_py.borrow(py);
                let src = alkahest_core::emit_definite_integration_cert(
                    integrand,
                    var,
                    lower,
                    upper,
                    &pool.inner,
                );
                if src.is_empty() || src.contains("sorry") || src.contains("admit") {
                    None
                } else {
                    Some(src)
                }
            } else if self.raw.log.is_empty() {
                None
            } else {
                let pool_py = self.value.pool.clone_ref(py);
                let pool = pool_py.borrow(py);
                let src = alkahest_core::emit_lean_expr_wrt(&self.raw, &pool.inner, self.wrt);
                if src.is_empty() || src.contains("sorry") || src.contains("admit") {
                    None
                } else {
                    Some(src)
                }
            }
        };
        let has_certificate = lean_certificate.is_some();
        let integration_verification =
            self.integration_verification_input
                .and_then(|(integrand, var)| {
                    let pool_py = self.value.pool.clone_ref(py);
                    let pool = pool_py.borrow(py);
                    core_verify_antiderivative_status(self.raw.value, integrand, var, &pool.inner)
                });
        metadata
            .set_item(
                "status",
                if integration_verification == Some(AntiderivativeVerification::Exact) {
                    "exactly_verified"
                } else if integration_verification == Some(AntiderivativeVerification::Numeric) {
                    "numerically_checked"
                } else if has_certificate {
                    "certificate_available"
                } else {
                    "unverified"
                },
            )
            .unwrap();
        metadata
            .set_item(
                "evidence",
                if integration_verification == Some(AntiderivativeVerification::Exact) {
                    "antiderivative_derivative_identity"
                } else if integration_verification == Some(AntiderivativeVerification::Numeric) {
                    "antiderivative_numeric_samples"
                } else if has_certificate {
                    "derivation_log"
                } else {
                    "none"
                },
            )
            .unwrap();
        metadata.set_item("externally_verified", false).unwrap();
        metadata
            .set_item(
                "artifact_format",
                if has_certificate { Some("lean4") } else { None },
            )
            .unwrap();

        let side_conditions = PyList::empty_bound(py);
        for (_, _, _, conditions) in &self.steps_raw {
            for condition in conditions {
                side_conditions.append(condition).unwrap();
            }
        }
        metadata
            .set_item("side_conditions", side_conditions)
            .unwrap();
        metadata
            .set_item(
                "method",
                if integration_verification == Some(AntiderivativeVerification::Exact) {
                    "in_kernel_symbolic_residual"
                } else if integration_verification == Some(AntiderivativeVerification::Numeric) {
                    "floating_point_samples"
                } else {
                    "derivation_log"
                },
            )
            .unwrap();
        metadata
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("DerivedResult(value={})", self.value.__repr__(py)?))
    }

    fn __bool__(&self, py: Python<'_>) -> bool {
        !expr_is_zero(py, &self.value)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, _py: Python<'_>) -> PyResult<bool> {
        if let Ok(other_dr) = other.downcast::<PyDerivedResult>() {
            return Ok(expr_ids_equal(&self.value, &other_dr.borrow().value));
        }
        if let Ok(other_expr) = other.extract::<PyRef<PyExpr>>() {
            return Ok(expr_ids_equal(&self.value, &other_expr));
        }
        Ok(false)
    }

    /// Stable, versioned dict envelope for machine/agent consumers.
    ///
    /// ``mode="full"`` (default) carries the same information as
    /// ``.steps`` / ``.verification`` / ``.certificate_status`` combined
    /// under one discriminated envelope (``kind="alkahest.derived_result"``,
    /// versioned by :attr:`SCHEMA_VERSION` / :attr:`STEPS_SCHEMA_VERSION`).
    ///
    /// ``mode="compact"`` is strictly smaller and intended for hot loops /
    /// tight agent context budgets:
    ///
    /// * ``steps`` entries drop ``before``/``after`` (usually the largest
    ///   strings in a derivation — the single biggest token cost) and use
    ///   short keys: ``r`` (rule name) and ``s`` (``side_conditions``,
    ///   *omitted entirely* when the list is empty).
    /// * ``verification`` is pruned to ``status`` and
    ///   ``externally_verified`` only — the two fields that carry the
    ///   honesty signal (whether this result is verified, and whether that
    ///   verification happened out-of-process). ``verification["status"]``
    ///   is **never** renamed, abbreviated, or omitted in compact mode.
    /// * ``certificate_status`` is pruned to ``certifiable`` and
    ///   ``reason``; the ``blocking_steps`` diagnostic list (which repeats
    ///   ``before``/``after`` expression text) is dropped.
    ///
    /// Neither mode ever includes the Lean certificate source text — use
    /// the :attr:`certificate` getter for that. ``has_certificate`` (a
    /// bool) plus ``certificate_status.reason`` is enough to know whether a
    /// certificate exists and why not, without paying for the source.
    ///
    /// Raises ``ValueError`` for any ``mode`` other than ``"full"`` /
    /// ``"compact"``.
    #[pyo3(signature = (mode="full"))]
    fn to_dict<'py>(&self, py: Python<'py>, mode: &str) -> PyResult<Bound<'py, PyDict>> {
        let compact = derived_result_mode_is_compact(mode)?;

        let has_certificate = self.certificate(py).is_some();
        let verification_full = self.verification(py);
        let certificate_status_full = self.certificate_status(py);

        let out = PyDict::new_bound(py);
        out.set_item("kind", "alkahest.derived_result")?;
        out.set_item("schema_version", RESULT_SCHEMA_VERSION)?;
        out.set_item("steps_schema_version", STEPS_SCHEMA_VERSION)?;
        out.set_item("value", self.value.__str__(py)?)?;

        if compact {
            let verification = PyDict::new_bound(py);
            verification.set_item(
                "status",
                verification_full
                    .get_item("status")?
                    .expect("verification always sets status"),
            )?;
            verification.set_item(
                "externally_verified",
                verification_full
                    .get_item("externally_verified")?
                    .expect("verification always sets externally_verified"),
            )?;
            out.set_item("verification", verification)?;

            let certificate_status = PyDict::new_bound(py);
            certificate_status.set_item(
                "certifiable",
                certificate_status_full
                    .get_item("certifiable")?
                    .expect("certificate_status always sets certifiable"),
            )?;
            certificate_status.set_item(
                "reason",
                certificate_status_full
                    .get_item("reason")?
                    .expect("certificate_status always sets reason"),
            )?;
            out.set_item("certificate_status", certificate_status)?;
        } else {
            out.set_item("verification", verification_full)?;
            out.set_item("certificate_status", certificate_status_full)?;
        }

        out.set_item("steps", self.steps_dict_list(py, compact))?;
        out.set_item("has_certificate", has_certificate)?;
        Ok(out)
    }

    /// ``json.dumps(self.to_dict(mode=mode))``, via Python's own ``json``
    /// module so the output matches what an agent's own ``json.dumps``
    /// would produce. See :meth:`to_dict` for the schema.
    #[pyo3(signature = (mode="full"))]
    fn to_json(&self, py: Python<'_>, mode: &str) -> PyResult<String> {
        derived_result_mode_is_compact(mode)?;
        let dict = self.to_dict(py, mode)?;
        let json = PyModule::import_bound(py, "json")?;
        json.getattr("dumps")?.call1((dict,))?.extract()
    }
}

/// Shared `mode` validation for `to_dict` / `to_json`. Returns `true` for
/// `"compact"`, `false` for `"full"`.
fn derived_result_mode_is_compact(mode: &str) -> PyResult<bool> {
    match mode {
        "full" => Ok(false),
        "compact" => Ok(true),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "DerivedResult.to_dict/to_json: mode must be 'full' or 'compact', got {other:?}"
        ))),
    }
}

impl PyDerivedResult {
    /// `steps` list for [`PyDerivedResult::to_dict`]. Full mode mirrors the
    /// `.steps` getter exactly (`rule`/`before`/`after`/`side_conditions`);
    /// compact mode uses short keys and drops `before`/`after`, omitting
    /// `s` entirely when `side_conditions` is empty.
    fn steps_dict_list<'py>(&self, py: Python<'py>, compact: bool) -> Bound<'py, PyList> {
        let list = PyList::empty_bound(py);
        for (rule, before, after, conds) in &self.steps_raw {
            let d = PyDict::new_bound(py);
            if compact {
                d.set_item("r", rule).unwrap();
                if !conds.is_empty() {
                    d.set_item("s", conds).unwrap();
                }
            } else {
                d.set_item("rule", rule).unwrap();
                d.set_item("before", before).unwrap();
                d.set_item("after", after).unwrap();
                d.set_item("side_conditions", conds).unwrap();
            }
            list.append(d).unwrap();
        }
        list
    }
}

fn make_derived_result(
    py: Python<'_>,
    derived: alkahest_core::DerivedExpr<alkahest_core::ExprId>,
    pool_py: Py<PyExprPool>,
    wrt: Option<ExprId>,
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
        raw: derived,
        wrt,
        integration_verification_input: None,
        definite_integration_input: None,
    }
}

/// Post-process a :class:`DerivedResult` with algebraic :func:`simplify` when
/// ``context(simplify=True)`` is active (Python layer calls this).
#[pyfunction]
#[pyo3(name = "_derived_result_context_simplify")]
fn py_derived_result_context_simplify(
    py: Python<'_>,
    dr: PyRef<PyDerivedResult>,
) -> PyResult<PyDerivedResult> {
    let pool_py = dr.value.pool.clone_ref(py);
    let simplified = {
        let pool = pool_py.borrow(py);
        core_simplify(dr.value.id, &pool.inner)
    };
    if simplified.value == dr.value.id {
        return Ok(PyDerivedResult {
            value: dr.value.clone(),
            derivation: dr.derivation.clone(),
            steps_raw: dr.steps_raw.clone(),
            raw: dr.raw.clone(),
            wrt: dr.wrt,
            integration_verification_input: dr.integration_verification_input,
            definite_integration_input: dr.definite_integration_input,
        });
    }
    let mut log = dr.raw.log.clone();
    log.push(RewriteStep::simple(
        "context_simplify",
        dr.raw.value,
        simplified.value,
    ));
    let merged = DerivedExpr {
        value: simplified.value,
        log: log.merge(simplified.log),
    };
    let mut result = make_derived_result(py, merged, pool_py, dr.wrt);
    result.integration_verification_input = dr.integration_verification_input;
    result.definite_integration_input = dr.definite_integration_input;
    Ok(result)
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

/// Symbolic complex conjugation (stable top-level export).
#[pyfunction]
fn conjugate(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "conjugate", expr)
}

/// Symbolic real-part constructor (stable top-level export).
#[pyfunction]
fn re(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "re", expr)
}

/// Symbolic imaginary-part constructor (stable top-level export).
#[pyfunction]
fn im(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "im", expr)
}

/// Principal argument (`Arg ∈ (−π, π]`; stable top-level export).
///
/// Only domain-safe literal simplifications are applied; branch-cut cases
/// remain symbolic.
#[pyfunction]
fn arg(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "arg", expr)
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
fn asinh(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "asinh", expr)
}

#[pyfunction]
fn acosh(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "acosh", expr)
}

#[pyfunction]
fn atanh(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "atanh", expr)
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

/// Principal-branch Lambert W₀(x), with W(x)·e^W(x) = x.
///
/// Stable top-level export (also re-exported from ``alkahest.experimental``).
#[pyfunction]
fn lambert_w(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "lambert_w", expr)
}

/// Digamma ψ(x) = Γ′(x)/Γ(x).
///
/// Stable top-level export (also re-exported from ``alkahest.experimental``).
#[pyfunction]
fn digamma(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "digamma", expr)
}

/// Bessel function of the first kind, order 0: J₀(x).
///
/// Stable top-level export (also re-exported from ``alkahest.experimental``).
#[pyfunction]
fn bessel_j0(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "bessel_j0", expr)
}

/// Bessel function of the first kind, order 1: J₁(x).
///
/// Stable top-level export (also re-exported from ``alkahest.experimental``).
#[pyfunction]
fn bessel_j1(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "bessel_j1", expr)
}

/// Heaviside step `θ(x)` (registered primitive; `θ(0) = 1/2`).
///
/// Surfaced under `alkahest.experimental` to avoid mutating the frozen
/// top-level `__all__` (the constructor pairs with the experimental Laplace
/// transform — PR #152).
#[pyfunction]
fn heaviside(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "heaviside", expr)
}

/// Dirac delta `δ(x)` (registered primitive; derivative of `heaviside`).
///
/// Surfaced under `alkahest.experimental` (see [`heaviside`]).
#[pyfunction]
fn dirac_delta(py: Python<'_>, expr: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "diracdelta", expr)
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

fn make_ternary_func(
    py: Python<'_>,
    name: &str,
    a: PyRef<PyExpr>,
    b: PyRef<PyExpr>,
    c: PyRef<PyExpr>,
) -> PyExpr {
    let id = a.pool.borrow(py).inner.func(name, vec![a.id, b.id, c.id]);
    let pool = a.pool.clone_ref(py);
    PyExpr { id, pool }
}

// ── Elliptic special functions (parameter convention m = k²) ──────────────────

/// Complete elliptic integral of the first kind, `EllipticK(m)`.
#[pyfunction]
fn elliptic_k(py: Python<'_>, m: PyRef<PyExpr>) -> PyExpr {
    make_func(py, "EllipticK", m)
}

/// Elliptic integral of the second kind.
///
/// `elliptic_e(m)` is the *complete* integral `EllipticE(m)`.
/// `elliptic_e(phi, m)` is the *incomplete* integral `EllipticE(phi, m)`.
#[pyfunction]
#[pyo3(signature = (arg1, arg2=None))]
fn elliptic_e(py: Python<'_>, arg1: PyRef<PyExpr>, arg2: Option<PyRef<PyExpr>>) -> PyExpr {
    match arg2 {
        None => make_func(py, "EllipticE", arg1),
        Some(m) => make_binary_func(py, "EllipticE", arg1, m),
    }
}

/// Incomplete elliptic integral of the first kind, `EllipticF(phi, m)`.
#[pyfunction]
fn elliptic_f(py: Python<'_>, phi: PyRef<PyExpr>, m: PyRef<PyExpr>) -> PyExpr {
    make_binary_func(py, "EllipticF", phi, m)
}

/// Incomplete elliptic integral of the third kind, `EllipticPi(n, phi, m)`.
#[pyfunction]
fn elliptic_pi(py: Python<'_>, n: PyRef<PyExpr>, phi: PyRef<PyExpr>, m: PyRef<PyExpr>) -> PyExpr {
    make_ternary_func(py, "EllipticPi", n, phi, m)
}

// ---------------------------------------------------------------------------
// Module-level: simplify and diff
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "simplify")]
fn py_simplify(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_simplify(expr.id, &pool.inner)
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, None))
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
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        shrink_iters = 5,
        explore_iters = 3,
        const_fold_iters = 3,
        node_limit = None,
        iter_limit = None,
        include_trig_rules = true,
        include_log_exp_rules = true,
        disjoint_schedule = true,
    ))]
    fn new(
        shrink_iters: usize,
        explore_iters: usize,
        const_fold_iters: usize,
        node_limit: Option<usize>,
        iter_limit: Option<usize>,
        include_trig_rules: bool,
        include_log_exp_rules: bool,
        disjoint_schedule: bool,
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
                disjoint_schedule,
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

    #[getter]
    fn disjoint_schedule(&self) -> bool {
        self.inner.disjoint_schedule
    }
}

#[pyfunction]
#[pyo3(name = "simplify_egraph")]
fn py_simplify_egraph(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_simplify_egraph(expr.id, &pool.inner)
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, None))
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
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_simplify_egraph_with(expr.id, &pool.inner, &config.inner, &SizeCost)
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, None))
}

#[pyfunction]
#[pyo3(name = "diff")]
fn py_diff(py: Python<'_>, expr: PyRef<PyExpr>, var: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_diff(expr.id, var.id, &pool.inner).map_err(diff_error_to_py)?
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, Some(var.id)))
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
        guard_depth(&pool.inner, expr.id)?;
        core_diff_forward(expr.id, var.id, &pool.inner).map_err(diff_error_to_py)?
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, Some(var.id)))
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
    /// When set (e.g. from `from_symbolic`), `__str__` uses user symbol names.
    pool: Option<Py<PyExprPool>>,
}

fn merge_mp_pool(a: &Option<Py<PyExprPool>>, b: &Option<Py<PyExprPool>>) -> Option<Py<PyExprPool>> {
    a.clone().or_else(|| b.clone())
}

// ---------------------------------------------------------------------------
// V2-7 — Factorization result types
// ---------------------------------------------------------------------------

#[pyclass(name = "UniPolyFactorization")]
struct PyUniPolyFactorization {
    inner: UniPolyFactorization,
    original: UniPoly,
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

    /// Exact in-kernel reconstruction evidence for this factorization.
    #[getter]
    fn verification<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        factor_verification_dict(py, self.inner.verifies_product(&self.original))
    }
}

#[pyclass(name = "MultiPolyFactorization")]
struct PyMultiPolyFactorization {
    inner: MultiPolyFactorization,
    original: MultiPoly,
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
            .map(|(p, e)| {
                (
                    PyMultiPoly {
                        inner: p.clone(),
                        pool: None,
                    },
                    *e,
                )
            })
            .collect()
    }

    /// Exact in-kernel reconstruction evidence for this factorization.
    #[getter]
    fn verification<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        factor_verification_dict(py, self.inner.verifies_product(&self.original))
    }
}

fn factor_verification_dict<'py>(py: Python<'py>, verified: bool) -> Bound<'py, PyDict> {
    let result = PyDict::new_bound(py);
    result
        .set_item(
            "status",
            if verified {
                "exactly_verified"
            } else {
                "unverified"
            },
        )
        .unwrap();
    result.set_item("evidence", "factor_product").unwrap();
    result
        .set_item("method", "in_kernel_exact_reconstruction")
        .unwrap();
    result.set_item("lean_checked", false).unwrap();
    result
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

    /// Construct a `UniPoly` from coefficients (constant term first).
    ///
    /// Each coefficient may be a Python ``int`` or an integer ``Expr`` from the
    /// same pool as *var*.
    ///
    /// ```python
    /// p = ExprPool()
    /// x = p.symbol("x")
    /// # -1 + x^2  (coefficients in ascending degree order)
    /// poly = UniPoly.from_coefficients([-1, 0, 1], x)
    /// # also accepted:
    /// poly = UniPoly.from_coefficients([p.integer(-1), p.integer(0), p.integer(1)], x)
    /// ```
    ///
    /// Raises `TypeError` if any coefficient is not an int / integer expression.
    /// Raises `OverflowError` if any coefficient overflows `i64`.
    #[staticmethod]
    fn from_coefficients(
        py: Python<'_>,
        coefficients: Vec<Bound<'_, PyAny>>,
        var: PyRef<'_, PyExpr>,
    ) -> PyResult<Self> {
        let pool = var.pool.borrow(py);
        let mut i64_coeffs: Vec<i64> = Vec::with_capacity(coefficients.len());
        for (idx, coeff) in coefficients.iter().enumerate() {
            if let Ok(n) = coeff.extract::<i64>() {
                i64_coeffs.push(n);
                continue;
            }
            if let Ok(expr) = coeff.extract::<PyRef<PyExpr>>() {
                match pool.inner.get(expr.id) {
                    alkahest_core::ExprData::Integer(bi) => {
                        let n = bi.0.to_i64().ok_or_else(|| {
                            PyOverflowError::new_err(format!(
                                "UniPoly.from_coefficients: coefficient at index {idx} overflows i64"
                            ))
                        })?;
                        i64_coeffs.push(n);
                        continue;
                    }
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "UniPoly.from_coefficients: coefficient at index {idx} is not an integer expression"
                        )));
                    }
                }
            }
            return Err(PyTypeError::new_err(format!(
                "UniPoly.from_coefficients: coefficient at index {idx} must be int or integer Expr"
            )));
        }
        let coeffs = alkahest_core::FlintPoly::from_coefficients(&i64_coeffs);
        Ok(PyUniPoly {
            inner: UniPoly {
                var: var.id,
                coeffs,
            },
        })
    }

    /// Coefficients in ascending degree order, as exact Python `int`s.
    ///
    /// Lossless for coefficients of any size. This used to go through
    /// `coefficients_i64()`, which does not merely saturate — it returned `0`
    /// for anything past `i64`, so `2**100 * x**2 + 1` came back as
    /// `[1, 0, 0]`: a quadratic reading as the constant `1`, with no exception
    /// and no flag. That is exactly the silent-error class this project gates
    /// against, and it is reachable from ordinary use, since `factor_z`,
    /// resultants and pseudo-division all grow coefficients past 64 bits.
    fn coefficients(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let int_cls = py.get_type_bound::<PyInt>();
        self.inner
            .coefficients()
            .into_iter()
            .map(|c| Ok(int_cls.call1((c.to_string(),))?.into_py(py)))
            .collect()
    }

    /// Leading (highest-degree) coefficient, as an exact Python `int`.
    ///
    /// `0` for the zero polynomial, matching `degree == -1` there.  Lossless
    /// for coefficients of any size, as `coefficients()` also now is — so both
    /// are safe on the output of `factor_z`, resultants and pseudo-division,
    /// where the coefficients grow past 64 bits.
    ///
    /// A property, not a method: it is a single FLINT coefficient read.
    #[getter]
    fn leading_coeff(&self, py: Python<'_>) -> PyResult<PyObject> {
        let int_cls = py.get_type_bound::<PyInt>();
        Ok(int_cls
            .call1((self.inner.leading_coeff().to_string(),))?
            .into_py(py))
    }

    /// Degree of the polynomial (`-1` for the zero polynomial).
    #[getter]
    fn degree(&self) -> i64 {
        self.inner.degree()
    }

    /// True if this is the zero polynomial.
    #[getter]
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

    fn __floordiv__(&self, other: PyRef<PyUniPoly>) -> PyResult<PyUniPoly> {
        self.inner
            .pseudo_divrem(&other.inner)
            .map(|(q, _)| PyUniPoly { inner: q })
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("//: variable mismatch"))
    }

    fn __mod__(&self, other: PyRef<PyUniPoly>) -> PyResult<PyUniPoly> {
        self.inner
            .pseudo_divrem(&other.inner)
            .map(|(_, r)| PyUniPoly { inner: r })
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("%: variable mismatch"))
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
            .map(|inner| PyUniPolyFactorization {
                inner,
                original: self.inner.clone(),
            })
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
    /// Build a multivariate polynomial from a symbolic expression.
    ///
    /// If *vars* is omitted, free symbols of *expr* are used (sorted by
    /// internal id for a deterministic order).
    #[staticmethod]
    #[pyo3(signature = (expr, vars=None))]
    fn from_symbolic(
        py: Python<'_>,
        expr: PyRef<PyExpr>,
        vars: Option<Vec<PyRef<PyExpr>>>,
    ) -> PyResult<Self> {
        let pool = expr.pool.borrow(py);
        let var_ids: Vec<_> = match vars {
            Some(v) => v.iter().map(|v| v.id).collect(),
            None => alkahest_core::collect_free_vars(expr.id, &pool.inner),
        };
        MultiPoly::from_symbolic(expr.id, var_ids, &pool.inner)
            .map(|p| PyMultiPoly {
                inner: p,
                pool: Some(expr.pool.clone_ref(py)),
            })
            .map_err(conv_error_to_py)
    }

    /// True if this is the zero polynomial.
    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Highest total degree over all terms (`0` for the zero polynomial).
    #[getter]
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
            pool: merge_mp_pool(&self.pool, &other.pool),
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
            pool: merge_mp_pool(&self.pool, &other.pool),
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
            pool: merge_mp_pool(&self.pool, &other.pool),
        })
    }

    fn primitive_part(&self) -> PyMultiPoly {
        PyMultiPoly {
            inner: self.inner.primitive_part(),
            pool: self.pool.clone(),
        }
    }

    /// GCD over ℤ (multivariate FLINT).
    fn gcd(&self, other: PyRef<PyMultiPoly>) -> PyResult<PyMultiPoly> {
        self.inner
            .gcd(&other.inner)
            .map(|inner| PyMultiPoly {
                inner,
                pool: merge_mp_pool(&self.pool, &other.pool),
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "gcd: incompatible variable lists or zero polynomial",
                )
            })
    }

    /// Factor over ℤ (multivariate FLINT).
    fn factor_z(&self) -> PyResult<PyMultiPolyFactorization> {
        self.inner
            .factor_z()
            .map(|inner| PyMultiPolyFactorization {
                inner,
                original: self.inner.clone(),
            })
            .map_err(factor_error_to_py)
    }

    fn __repr__(&self) -> String {
        format!("MultiPoly({})", self.inner)
    }

    fn __str__(&self, py: Python<'_>) -> String {
        if let Some(ref pool_py) = self.pool {
            let pool = pool_py.borrow(py);
            self.inner.display_with(&pool.inner)
        } else {
            self.inner.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// PyRationalFunction
// ---------------------------------------------------------------------------

/// Convert a `UniPoly` into a univariate `MultiPoly` over the same variable.
///
/// `MultiPoly` stores terms as `BTreeMap<Vec<u32>, rug::Integer>` where the
/// key is the exponent vector (one entry per variable, trailing zeros omitted).
/// The constant term has key `vec![]`.
fn unipoly_to_multipoly(p: &UniPoly) -> MultiPoly {
    let vars = vec![p.var];
    let coeffs = p.coefficients(); // Vec<rug::Integer>, ascending degree
    let mut terms = std::collections::BTreeMap::new();
    for (i, coeff) in coeffs.into_iter().enumerate() {
        if coeff != 0 {
            let exp: Vec<u32> = if i == 0 { vec![] } else { vec![i as u32] };
            terms.insert(exp, coeff);
        }
    }
    MultiPoly { vars, terms }
}

#[pyclass(name = "RationalFunction")]
struct PyRationalFunction {
    inner: RationalFunction,
    pool: Option<Py<PyExprPool>>,
}

#[pymethods]
impl PyRationalFunction {
    /// Construct a `RationalFunction` from two `UniPoly` numerator / denominator.
    ///
    /// ```python
    /// p = ExprPool()
    /// x = p.symbol("x")
    /// numer = UniPoly.from_coefficients([p.integer(-1), p.integer(0), p.integer(1)], x)  # x²-1
    /// denom = UniPoly.from_coefficients([p.integer(-1), p.integer(1)], x)                # x-1
    /// rf = RationalFunction(numer, denom)  # → (x+1)  after GCD reduction
    /// ```
    ///
    /// Raises `ValueError` if `denom` is the zero polynomial.
    #[new]
    fn from_unipolys(numer: PyRef<'_, PyUniPoly>, denom: PyRef<'_, PyUniPoly>) -> PyResult<Self> {
        let n = unipoly_to_multipoly(&numer.inner);
        let d = unipoly_to_multipoly(&denom.inner);
        RationalFunction::new(n, d)
            .map(|r| PyRationalFunction {
                inner: r,
                pool: None,
            })
            .map_err(conv_error_to_py)
    }

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
            .map(|r| PyRationalFunction {
                inner: r,
                pool: Some(numer.pool.clone_ref(py)),
            })
            .map_err(conv_error_to_py)
    }

    /// True if the numerator is the zero polynomial.
    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    fn numer(&self) -> PyMultiPoly {
        PyMultiPoly {
            inner: self.inner.numer.clone(),
            pool: self.pool.clone(),
        }
    }

    fn denom(&self) -> PyMultiPoly {
        PyMultiPoly {
            inner: self.inner.denom.clone(),
            pool: self.pool.clone(),
        }
    }

    fn __add__(&self, other: PyRef<PyRationalFunction>) -> PyResult<PyRationalFunction> {
        (self.inner.clone() + other.inner.clone())
            .map(|r| PyRationalFunction {
                inner: r,
                pool: merge_mp_pool(&self.pool, &other.pool),
            })
            .map_err(conv_error_to_py)
    }

    fn __sub__(&self, other: PyRef<PyRationalFunction>) -> PyResult<PyRationalFunction> {
        (self.inner.clone() - other.inner.clone())
            .map(|r| PyRationalFunction {
                inner: r,
                pool: merge_mp_pool(&self.pool, &other.pool),
            })
            .map_err(conv_error_to_py)
    }

    fn __mul__(&self, other: PyRef<PyRationalFunction>) -> PyResult<PyRationalFunction> {
        (self.inner.clone() * other.inner.clone())
            .map(|r| PyRationalFunction {
                inner: r,
                pool: merge_mp_pool(&self.pool, &other.pool),
            })
            .map_err(conv_error_to_py)
    }

    fn __truediv__(&self, other: PyRef<PyRationalFunction>) -> PyResult<PyRationalFunction> {
        (self.inner.clone() / other.inner.clone())
            .map(|r| PyRationalFunction {
                inner: r,
                pool: merge_mp_pool(&self.pool, &other.pool),
            })
            .map_err(conv_error_to_py)
    }

    fn __neg__(&self) -> PyRationalFunction {
        PyRationalFunction {
            inner: -self.inner.clone(),
            pool: self.pool.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("RationalFunction({})", self.inner)
    }

    fn __str__(&self, py: Python<'_>) -> String {
        if let Some(ref pool_py) = self.pool {
            let pool = pool_py.borrow(py);
            self.inner.display_with(&pool.inner)
        } else {
            self.inner.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Integrate, with the GIL **released** for the duration of the core call.
///
/// `integrate` is one of the two engines that honour `alkahest.Budget` /
/// `request_cancel()` (see `docs/mdbook/src/budgets.md`). Holding the GIL for
/// the whole run made that promise half-true: a watchdog thread calling
/// `request_cancel()` could not execute a single bytecode until the call it
/// wanted to cancel had already finished, so only a flag set *before* the call
/// was ever observed — the opposite of what a fan-out search loop needs.
///
/// The idiom is `py_simplify_par`'s, and the safety argument is the same one:
/// `ExprPool` is `Send + Sync` and interns through a lock-free index, and this
/// is strictly weaker than what `simplify_par` already does (Rayon workers on
/// the same pool, concurrently). Nothing under `core_integrate` touches a
/// `Python` token. The budget itself is thread-local, and `allow_threads` does
/// not move the work to another thread — it only drops the GIL on this one — so
/// the active `Budget` frame is still the caller's.
#[pyfunction]
#[pyo3(name = "integrate")]
fn py_integrate(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool_ref = expr.pool.borrow(py);
        guard_depth(&pool_ref.inner, expr.id)?;
        // Bind out of the `PyRef` first: it carries a `Python` marker and so is
        // not `Sync`, but the pool and ids themselves are safe to send.
        let (id, var_id, pool) = (expr.id, var.id, &pool_ref.inner);
        py.allow_threads(|| core_integrate(id, var_id, pool))
            .map_err(integrate_error_to_py)?
    };
    let pool_py = expr.pool.clone_ref(py);
    let mut result = make_derived_result(py, derived, pool_py, None);
    result.integration_verification_input = Some((expr.id, var.id));
    Ok(result)
}

#[pyfunction]
#[pyo3(name = "integrate_definite")]
fn py_integrate_definite(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
    lower: PyRef<PyExpr>,
    upper: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        core_integrate_definite(expr.id, var.id, lower.id, upper.id, &pool.inner)
            .map_err(integrate_error_to_py)?
    };
    let pool_py = expr.pool.clone_ref(py);
    let mut result = make_derived_result(py, derived, pool_py, None);
    result.definite_integration_input = Some((expr.id, var.id, lower.id, upper.id));
    Ok(result)
}

#[pyfunction]
#[pyo3(name = "apart")]
fn py_apart(py: Python<'_>, expr: PyRef<PyExpr>, var: PyRef<PyExpr>) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_apart(expr.id, var.id, &pool.inner).map_err(apart_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// `ResidueError` has an inherent `code()` but does not implement the
/// `AlkahestError` trait, so this cannot go through [`make_structured_err`].
/// It still has to produce an `AlkahestError` subclass carrying `.code`:
/// raising a bare `ValueError` with the code glued into the message made the
/// code unreadable except by string-matching. `AlkahestError` subclasses
/// `ValueError`, so `except ValueError` keeps working.
fn residue_error_to_py(e: ResidueError) -> PyErr {
    Python::with_gil(|py| {
        coded_error::<PyAlkahestError>(
            py,
            e.code(),
            e.to_string(),
            match e {
                ResidueError::NotRational => {
                    "input must be a rational function of the variable over ℚ"
                }
                ResidueError::ZeroDenominator => "denominator must be non-zero",
                ResidueError::PoleOrderTooHigh { .. } => {
                    "pole order exceeds supported bound; essential singularities are out of scope"
                }
                ResidueError::DivisionByZero => {
                    "division by zero during Laurent coefficient extraction"
                }
            },
        )
    })
}

/// `residue(f, z, point)` was handed a `point` that is not an exact constant.
///
/// `E-RESIDUE-005` is raised only at the Python boundary — the Rust `residue`
/// takes an already-parsed `GaussRat` and cannot reach this state — so it is
/// deliberately absent from `alkahest-core`'s `REGISTRY`, on the same footing
/// as `E-SMT-001`/`E-SMT-003`/`E-SMT-004` in `alkahest/smt.py` and
/// `E-BATCH-001` in `alkahest/_batch.py`.
fn residue_point_error(py: Python<'_>, point: &Bound<'_, PyAny>) -> PyErr {
    coded_error::<PyAlkahestError>(
        py,
        "E-RESIDUE-005",
        format!(
            "residue: the point must be an exact constant in ℚ(i), got {}",
            py_type_name(point)
        ),
        "pass an int, a fractions.Fraction, a complex with integral parts, or a \
         (re, im) pair of rationals. A symbolic Expr is not accepted — residue \
         evaluates at one point, so substitute a value first",
    )
}
fn rational_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Rational> {
    if let Ok(i) = ob.extract::<i64>() {
        return Ok(Rational::from(i));
    }
    exact_binding(ob)
}
fn parse_gauss_point(ob: &Bound<'_, PyAny>) -> PyResult<GaussRat> {
    if let Ok(t) = ob.downcast::<PyTuple>() {
        if t.len() == 2 {
            return Ok(GaussRat::from_re_im(
                rational_from_py(&t.get_item(0)?)?,
                rational_from_py(&t.get_item(1)?)?,
            ));
        }
    }
    if let Ok(c) = ob.downcast::<PyComplex>() {
        let (re, im) = (c.real(), c.imag());
        if re.fract() == 0.0 && im.fract() == 0.0 {
            return Ok(GaussRat::from_re_im(
                Rational::from(re as i64),
                Rational::from(im as i64),
            ));
        }
    }
    Ok(GaussRat::from_re_im(
        rational_from_py(ob)?,
        Rational::from(0),
    ))
}
fn try_complex_binding(ob: &Bound<'_, PyAny>) -> Option<ComplexF64> {
    if let Some(v) = try_strict_complex_binding(ob) {
        return Some(v);
    }
    // Real scalars are valid only when the caller already selected complex mode.
    if let Ok(x) = ob.extract::<f64>() {
        return Some(ComplexF64::new(x, 0.0));
    }
    if let Ok(x) = ob.extract::<i64>() {
        return Some(ComplexF64::new(x as f64, 0.0));
    }
    None
}

/// True complex values that should trigger auto-mode complex evaluation.
/// Deliberately excludes reals/`Fraction` (those extract as `f64`) so auto
/// mode still prefers exact rational / f64 backends.
fn try_strict_complex_binding(ob: &Bound<'_, PyAny>) -> Option<ComplexF64> {
    if let Ok(c) = ob.downcast::<PyComplex>() {
        return Some(ComplexF64::new(c.real(), c.imag()));
    }
    if let Ok(t) = ob.downcast::<PyTuple>() {
        if t.len() == 2 {
            let re: f64 = t.get_item(0).ok()?.extract().ok()?;
            let im: f64 = t.get_item(1).ok()?.extract().ok()?;
            return Some(ComplexF64::new(re, im));
        }
    }
    None
}
#[pyfunction]
#[pyo3(name = "residue")]
fn py_residue(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
    point: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let gauss = parse_gauss_point(point).map_err(|_| residue_point_error(py, point))?;
    let id = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_residue(expr.id, var.id, gauss, &pool.inner).map_err(residue_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

fn apart_error_to_py(e: ApartError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// `alkahest.series(expr, var, point, order) -> Series`
///
/// Truncated Taylor / Laurent expansion of *expr* in *var* about *point*, with
/// an explicit ``O(h^order)`` remainder.
///
/// The expansion is **bounded**: it honours an active :class:`alkahest.Budget`
/// and, with none, an internal work ceiling. Coefficients are formed by
/// repeated differentiation without re-simplifying, so an expression whose
/// derivatives do not close (nested radicals, in particular) grows by a
/// constant factor per coefficient and a high order is unreachable rather than
/// slow. Running out of room raises :exc:`alkahest.SeriesError` with code
/// ``E-SERIES-003`` (or :exc:`alkahest.BudgetExceededError` when a budget
/// stopped it) — never a shorter series, which would wear an ``O(·)`` label
/// nothing bounded.
#[pyfunction]
#[pyo3(name = "series")]
fn py_series(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
    point: &Bound<'_, PyAny>,
    order: u32,
) -> PyResult<PySeries> {
    let pool_py = expr.pool.clone_ref(py);
    let point_id = coerce_substituent(&pool_py, point, py)?;
    let id = {
        let pool_ref = pool_py.borrow(py);
        guard_depth(&pool_ref.inner, expr.id)?;
        checked_order("series order", order as usize)?;
        // GIL released for the core call, like `limit` and `integrate`: the
        // coefficient loop honours `Budget`, and a `request_cancel()` from
        // another Python thread cannot reach it while this one holds the GIL.
        let (id, var_id, pool) = (expr.id, var.id, &pool_ref.inner);
        py.allow_threads(|| core_series(id, var_id, point_id, order, pool))
            .map_err(series_error_to_py)?
            .expr()
    };
    Ok(PySeries {
        expr: PyExpr { id, pool: pool_py },
    })
}

/// Take a limit, with the GIL **released** for the duration of the core call.
///
/// `limit` is the other budget-honouring engine; see `py_integrate` for why
/// holding the GIL made `request_cancel()` unable to reach a running call, and
/// for the safety argument (identical here — `core_limit` takes `&ExprPool` and
/// no `Python` token, and the budget/work-ceiling state it uses is thread-local
/// to *this* thread, which `allow_threads` does not change).
#[pyfunction]
#[pyo3(name = "limit", signature = (expr, var, point, dir=None))]
fn py_limit(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
    point: PyRef<PyExpr>,
    dir: Option<&str>,
) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let d = parse_limit_direction(dir);
    let id = {
        let pool_ref = pool_py.borrow(py);
        guard_depth(&pool_ref.inner, expr.id)?;
        // Bind out of the `PyRef` first: it carries a `Python` marker and so is
        // not `Sync`, but the pool and ids themselves are safe to send.
        let (id, var_id, point_id, pool) = (expr.id, var.id, point.id, &pool_ref.inner);
        py.allow_threads(|| core_limit(id, var_id, point_id, d, pool))
            .map_err(limit_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

// ===========================================================================
// Experimental calculus / ODE / transform surface (PyO3 bindings, PRs #152–#161)
//
// Exposed via `alkahest.experimental`. Conversions follow the integrate/apart
// idiom: borrow the input `Expr`'s pool, call the core routine, and wrap the
// resulting `ExprId`s back into `Expr` against the same pool.
// ===========================================================================

/// Convert a rug `Rational` into a Python `int` (when integral) or
/// `fractions.Fraction` (otherwise), so Fps coefficients are exact in Python.
fn rational_to_py(py: Python<'_>, r: &Rational) -> PyResult<PyObject> {
    let numer = r.numer().to_string();
    let denom = r.denom().to_string();
    if *r.denom() == 1 {
        let int_cls = py.get_type_bound::<PyInt>();
        return Ok(int_cls.call1((numer,))?.into_py(py));
    }
    let fractions = py.import_bound("fractions")?;
    let frac = fractions.getattr("Fraction")?;
    // Fraction(str) accepts the "numer/denom" form; the two-argument form
    // requires Rational instances, not strings.
    Ok(frac.call1((format!("{numer}/{denom}"),))?.into_py(py))
}

fn dsolve_error_to_py(e: CoreDsolveError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn laplace_error_to_py(e: CoreLaplaceError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn fourier_error_to_py(e: CoreFourierError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn ztransform_error_to_py(e: CoreZTransformError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn asymptotic_error_to_py(e: CoreAsymptoticError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn series_solve_error_to_py(e: CoreSeriesSolveError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
fn fps_error_to_py(e: CoreFpsError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// `experimental.dsolve(equation, x, y, [y', y'', …])` — solve a scalar ODE.
///
/// `equation` is interpreted as `equation = 0`, written in terms of the
/// independent variable `x`, the unknown `y`, and the derivative symbols
/// `derivs` (`derivs[0] = y'`, …). Returns a list of solution dicts with keys
/// `y_of_x` (the `Expr` for `y(x)`), `constants` (list of `Expr`), and `method`.
#[pyfunction]
#[pyo3(name = "dsolve")]
fn py_dsolve(
    py: Python<'_>,
    equation: PyRef<PyExpr>,
    x: PyRef<PyExpr>,
    y: PyRef<PyExpr>,
    derivs: Vec<PyExpr>,
) -> PyResult<PyObject> {
    let pool_py = equation.pool.clone_ref(py);
    let result = {
        let pool = pool_py.borrow(py);
        let input = CoreOdeInput {
            x: x.id,
            y: y.id,
            derivs: derivs.iter().map(|e| e.id).collect(),
            equation: equation.id,
        };
        core_dsolve(&input, &pool.inner).map_err(dsolve_error_to_py)?
    };
    let out = PyList::empty_bound(py);
    for sol in result.solutions {
        let d = PyDict::new_bound(py);
        d.set_item(
            "y_of_x",
            PyExpr {
                id: sol.y_of_x,
                pool: pool_py.clone_ref(py),
            }
            .into_py(py),
        )?;
        let consts = PyList::empty_bound(py);
        for c in sol.constants {
            consts.append(
                PyExpr {
                    id: c,
                    pool: pool_py.clone_ref(py),
                }
                .into_py(py),
            )?;
        }
        d.set_item("constants", consts)?;
        d.set_item("method", sol.method)?;
        out.append(d)?;
    }
    Ok(out.into_py(py))
}

// ---------------------------------------------------------------------------
// Numeric ODE integrators (experimental, Phase 16b)
// ---------------------------------------------------------------------------

/// Convert a [`CoreNumericOdeError`] to a Python exception.
fn numeric_ode_error_to_py(e: CoreNumericOdeError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyOdeError>();
        make_structured_err(py, &exc_type, &e)
    })
}

/// Sampled ODE trajectory returned by :func:`ode_integrate_rk4` and
/// :func:`ode_integrate_rk45`.
///
/// Attributes
/// ----------
/// t : list[float]
///     Time points (length = number of accepted steps + 1, including ``t_start``).
/// y : list[list[float]]
///     State values at each time point.  ``y[i][j]`` is the value of state
///     variable ``j`` at time ``t[i]``.
///
/// Methods
/// -------
/// t_final() → float | None
///     The last time point.
/// y_final() → list[float] | None
///     The state vector at the last time point.
#[pyclass(name = "OdeTrajectory")]
struct PyOdeTrajectory {
    inner: CoreOdeTrajectory,
}

#[pymethods]
impl PyOdeTrajectory {
    #[getter]
    fn t(&self, py: Python<'_>) -> PyObject {
        PyList::new_bound(py, &self.inner.t).into_py(py)
    }

    #[getter]
    fn y(&self, py: Python<'_>) -> PyObject {
        let rows: Vec<PyObject> = self
            .inner
            .y
            .iter()
            .map(|row| PyList::new_bound(py, row).into_py(py))
            .collect();
        PyList::new_bound(py, rows).into_py(py)
    }

    /// The final time point, or ``None`` for an empty trajectory.
    #[getter]
    fn t_final(&self) -> Option<f64> {
        self.inner.t_final()
    }

    fn y_final(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner
            .y_final()
            .map(|v| PyList::new_bound(py, v).into_py(py))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "OdeTrajectory(steps={}, t_final={:?})",
            self.inner.len(),
            self.inner.t_final()
        )
    }
}

/// ``experimental.ode_integrate_rk4(ode, y0, t_start, t_end, h, max_steps)``
///
/// Integrate a first-order ODE system using the classical 4th-order fixed-step
/// Runge–Kutta method.
///
/// Parameters
/// ----------
/// ode : ODE
///     First-order system ``dy/dt = f(t, y)`` built with :class:`alkahest.ODE`.
/// y0 : list[float]
///     Initial conditions; one value per state variable.
/// t_start : float
///     Start of the integration interval.
/// t_end : float
///     End of the integration interval; must satisfy ``t_end > t_start``.
/// h : float, optional
///     Fixed step size (default ``0.01``).
/// max_steps : int, optional
///     Maximum number of steps (default ``1_000_000``).
///
/// Returns
/// -------
/// OdeTrajectory
///     Sampled trajectory with ``.t`` and ``.y`` arrays.
///
/// Raises
/// ------
/// OdeError
///     On any integration failure (non-finite value, max steps exceeded, etc.).
///
/// Example::
///
///     import alkahest as A
///     from alkahest import experimental as ex
///     p = A.ExprPool()
///     t, y = p.symbol("t"), p.symbol("y")
///     ode = A.ODE([y], [y], t)   # dy/dt = y
///     traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 1.0, h=0.001)
///     import math
///     assert abs(traj.y_final()[0] - math.e) < 1e-6
#[pyfunction]
#[pyo3(name = "ode_integrate_rk4", signature = (ode, y0, t_start, t_end, h=0.01, max_steps=1_000_000))]
fn py_ode_integrate_rk4(
    py: Python<'_>,
    ode: PyRef<PyODE>,
    y0: Vec<f64>,
    t_start: f64,
    t_end: f64,
    h: f64,
    max_steps: usize,
) -> PyResult<PyOdeTrajectory> {
    let opts = CoreRk4Options { h, max_steps };
    let traj = {
        let pool = ode.pool.borrow(py);
        core_integrate_rk4(&ode.inner, &y0, t_start, t_end, &opts, &pool.inner)
            .map_err(numeric_ode_error_to_py)?
    };
    Ok(PyOdeTrajectory { inner: traj })
}

/// ``experimental.ode_integrate_rk45(ode, y0, t_start, t_end, ...)``
///
/// Integrate a first-order ODE system using the adaptive Dormand–Prince
/// RK4(5) method with automatic step-size control.
///
/// Parameters
/// ----------
/// ode : ODE
///     First-order system ``dy/dt = f(t, y)`` built with :class:`alkahest.ODE`.
/// y0 : list[float]
///     Initial conditions; one value per state variable.
/// t_start : float
///     Start of the integration interval.
/// t_end : float
///     End of the integration interval; must satisfy ``t_end > t_start``.
/// h_init : float, optional
///     Initial step size (default ``0.01``).
/// h_min : float, optional
///     Minimum allowable step size (default ``1e-12``).
/// h_max : float, optional
///     Maximum allowable step size (default ``1.0``).
/// rtol : float, optional
///     Relative tolerance (default ``1e-6``).
/// atol : float, optional
///     Absolute tolerance (default ``1e-9``).
/// max_steps : int, optional
///     Maximum number of accepted steps (default ``1_000_000``).
///
/// Returns
/// -------
/// OdeTrajectory
///     Sampled trajectory with ``.t`` and ``.y`` arrays.
///
/// Raises
/// ------
/// OdeError
///     On any integration failure (step size too small, non-finite value, etc.).
///
/// Example::
///
///     import alkahest as A
///     from alkahest import experimental as ex
///     p = A.ExprPool()
///     t, y = p.symbol("t"), p.symbol("y")
///     ode = A.ODE([y], [y], t)   # dy/dt = y
///     traj = ex.ode_integrate_rk45(ode, [1.0], 0.0, 1.0, rtol=1e-9, atol=1e-12)
///     import math
///     assert abs(traj.y_final()[0] - math.e) < 1e-8
#[pyfunction]
#[pyo3(name = "ode_integrate_rk45", signature = (
    ode, y0, t_start, t_end,
    h_init=0.01, h_min=1e-12, h_max=1.0, rtol=1e-6, atol=1e-9, max_steps=1_000_000
))]
#[allow(clippy::too_many_arguments)]
fn py_ode_integrate_rk45(
    py: Python<'_>,
    ode: PyRef<PyODE>,
    y0: Vec<f64>,
    t_start: f64,
    t_end: f64,
    h_init: f64,
    h_min: f64,
    h_max: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> PyResult<PyOdeTrajectory> {
    let opts = CoreRk45Options {
        h_init,
        h_min,
        h_max,
        rtol,
        atol,
        max_steps,
    };
    let traj = {
        let pool = ode.pool.borrow(py);
        core_integrate_rk45(&ode.inner, &y0, t_start, t_end, &opts, &pool.inner)
            .map_err(numeric_ode_error_to_py)?
    };
    Ok(PyOdeTrajectory { inner: traj })
}

/// `experimental.laplace_transform(f, t, s)` → `Expr` for `L{f}(s)`.
#[pyfunction]
#[pyo3(name = "laplace_transform")]
fn py_laplace_transform(
    py: Python<'_>,
    f: PyRef<PyExpr>,
    t: PyRef<PyExpr>,
    s: PyRef<PyExpr>,
) -> PyResult<PyExpr> {
    let pool_py = f.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        core_laplace(f.id, t.id, s.id, &pool.inner).map_err(laplace_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// `experimental.inverse_laplace_transform(F, s, t)` → `Expr` for `L⁻¹{F}(t)`.
#[pyfunction]
#[pyo3(name = "inverse_laplace_transform")]
fn py_inverse_laplace_transform(
    py: Python<'_>,
    big_f: PyRef<PyExpr>,
    s: PyRef<PyExpr>,
    t: PyRef<PyExpr>,
) -> PyResult<PyExpr> {
    let pool_py = big_f.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        core_ilaplace(big_f.id, s.id, t.id, &pool.inner).map_err(laplace_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// `experimental.fourier_transform(f, x, xi)` → `Expr` for `F{f}(ξ)` (unitary,
/// ordinary-frequency convention).
#[pyfunction]
#[pyo3(name = "fourier_transform")]
fn py_fourier_transform(
    py: Python<'_>,
    f: PyRef<PyExpr>,
    x: PyRef<PyExpr>,
    xi: PyRef<PyExpr>,
) -> PyResult<PyExpr> {
    let pool_py = f.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        core_fourier_transform(f.id, x.id, xi.id, &pool.inner).map_err(fourier_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// `experimental.inverse_fourier_transform(g, xi, x)` → `Expr` for `F⁻¹{g}(x)`.
#[pyfunction]
#[pyo3(name = "inverse_fourier_transform")]
fn py_inverse_fourier_transform(
    py: Python<'_>,
    g: PyRef<PyExpr>,
    xi: PyRef<PyExpr>,
    x: PyRef<PyExpr>,
) -> PyResult<PyExpr> {
    let pool_py = g.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        core_ifourier(g.id, xi.id, x.id, &pool.inner).map_err(fourier_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// `experimental.z_transform(a, n, z)` → `Expr` for the unilateral `Z{a[n]}(z)`.
#[pyfunction]
#[pyo3(name = "z_transform")]
fn py_z_transform(
    py: Python<'_>,
    a: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
    z: PyRef<PyExpr>,
) -> PyResult<PyExpr> {
    let pool_py = a.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        core_ztransform(a.id, n.id, z.id, &pool.inner).map_err(ztransform_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// `experimental.inverse_z_transform(X, z, n)` → `Expr` for `Z⁻¹{X}[n]`.
#[pyfunction]
#[pyo3(name = "inverse_z_transform")]
fn py_inverse_z_transform(
    py: Python<'_>,
    big_x: PyRef<PyExpr>,
    z: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
) -> PyResult<PyExpr> {
    let pool_py = big_x.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        core_iztransform(big_x.id, z.id, n.id, &pool.inner).map_err(ztransform_error_to_py)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// `experimental.multilimit(f, x, y, a, b)` — two-variable limit.
///
/// Returns a dict with key `status` in `{"value", "dne", "undecided"}`:
/// - `value`: also `value` (`Expr`);
/// - `dne`: also `path_a` / `path_b`, each a dict with `description` (str),
///   `value` (`Expr`), and `value_numeric` (float);
/// - `undecided`: no further keys.
#[pyfunction]
#[pyo3(name = "multilimit")]
fn py_multilimit(
    py: Python<'_>,
    f: PyRef<PyExpr>,
    x: PyRef<PyExpr>,
    y: PyRef<PyExpr>,
    a: PyRef<PyExpr>,
    b: PyRef<PyExpr>,
) -> PyResult<PyObject> {
    let pool_py = f.pool.clone_ref(py);
    let result = {
        let pool = pool_py.borrow(py);
        core_multilimit(f.id, x.id, y.id, a.id, b.id, &pool.inner)
    };
    let d = PyDict::new_bound(py);
    match result {
        CoreMultiLimit::Value(v) => {
            d.set_item("status", "value")?;
            d.set_item(
                "value",
                PyExpr {
                    id: v,
                    pool: pool_py.clone_ref(py),
                }
                .into_py(py),
            )?;
        }
        CoreMultiLimit::DoesNotExist { path_a, path_b } => {
            d.set_item("status", "dne")?;
            let mk =
                |w: &alkahest_core::calculus::multilimit::PathWitness| -> PyResult<Py<PyDict>> {
                    let pw = PyDict::new_bound(py);
                    pw.set_item("description", w.description.clone())?;
                    pw.set_item(
                        "value",
                        PyExpr {
                            id: w.value,
                            pool: pool_py.clone_ref(py),
                        }
                        .into_py(py),
                    )?;
                    pw.set_item("value_numeric", w.value_numeric)?;
                    Ok(pw.into())
                };
            d.set_item("path_a", mk(&path_a)?)?;
            d.set_item("path_b", mk(&path_b)?)?;
        }
        CoreMultiLimit::Undecided => {
            d.set_item("status", "undecided")?;
        }
    }
    Ok(d.into_py(py))
}

/// `experimental.asymptotic_expand(f, var, n_terms)` — asymptotic expansion at
/// `+∞`, returning a list of term `Expr`s (most significant first).
#[pyfunction]
#[pyo3(name = "asymptotic_expand")]
fn py_asymptotic_expand(
    py: Python<'_>,
    f: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
    n_terms: usize,
) -> PyResult<PyObject> {
    let pool_py = f.pool.clone_ref(py);
    let terms = {
        let pool = pool_py.borrow(py);
        core_asymptotic_expand(f.id, var.id, n_terms, &pool.inner)
            .map_err(asymptotic_error_to_py)?
            .term_exprs()
    };
    let out = PyList::empty_bound(py);
    for id in terms {
        out.append(
            PyExpr {
                id,
                pool: pool_py.clone_ref(py),
            }
            .into_py(py),
        )?;
    }
    Ok(out.into_py(py))
}

/// `experimental.series_solve(x, p, q, r, x0, order)` — power-series / Frobenius
/// solution of `p·y'' + q·y' + r·y = 0` about `x0`.
///
/// Returns a dict with `kind` (`"ordinary"` / `"regular_singular"`), `order`
/// (int), `x0` (`Expr`), and `solutions`: a list of dicts each with `exponent`
/// (Fraction/int), `coeffs` (list of Fraction/int), `log_coeff`
/// (Fraction/int or `None`), and `expr` (the truncated symbolic `Expr`).
#[pyfunction]
#[pyo3(name = "series_solve")]
fn py_series_solve(
    py: Python<'_>,
    x: PyRef<PyExpr>,
    p: PyRef<PyExpr>,
    q: PyRef<PyExpr>,
    r: PyRef<PyExpr>,
    x0: PyRef<PyExpr>,
    order: usize,
) -> PyResult<PyObject> {
    let pool_py = x.pool.clone_ref(py);
    let pool = pool_py.borrow(py);
    let ode = CoreSeriesOde::new(x.id, p.id, q.id, r.id);
    let result =
        core_series_solve(&ode, x0.id, order, &pool.inner).map_err(series_solve_error_to_py)?;
    let d = PyDict::new_bound(py);
    d.set_item(
        "kind",
        match result.kind {
            CorePointKind::Ordinary => "ordinary",
            CorePointKind::RegularSingular => "regular_singular",
        },
    )?;
    d.set_item("order", result.order)?;
    d.set_item(
        "x0",
        PyExpr {
            id: result.x0,
            pool: pool_py.clone_ref(py),
        }
        .into_py(py),
    )?;
    let sols = PyList::empty_bound(py);
    for s in &result.solutions {
        let sd = PyDict::new_bound(py);
        sd.set_item("exponent", rational_to_py(py, &s.exponent)?)?;
        let coeffs = PyList::empty_bound(py);
        for c in &s.coeffs {
            coeffs.append(rational_to_py(py, c)?)?;
        }
        sd.set_item("coeffs", coeffs)?;
        match &s.log_coeff {
            Some(c) => sd.set_item("log_coeff", rational_to_py(py, c)?)?,
            None => sd.set_item("log_coeff", py.None())?,
        }
        let expr_id = s.to_expr(x.id, x0.id, order, &pool.inner);
        sd.set_item(
            "expr",
            PyExpr {
                id: expr_id,
                pool: pool_py.clone_ref(py),
            }
            .into_py(py),
        )?;
        sols.append(sd)?;
    }
    d.set_item("solutions", sols)?;
    Ok(d.into_py(py))
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
        guard_depth(&pool.inner, expr.id)?;
        core_sum_indefinite(expr.id, k.id, &pool.inner).map_err(sum_error_to_py)?
    };
    Ok(make_derived_result(
        py,
        derived,
        expr.pool.clone_ref(py),
        None,
    ))
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
        guard_depth(&pool.inner, expr.id)?;
        core_sum_definite(expr.id, k.id, lo.id, hi.id, &pool.inner).map_err(sum_error_to_py)?
    };
    Ok(make_derived_result(
        py,
        derived,
        expr.pool.clone_ref(py),
        None,
    ))
}

#[pyfunction]
#[pyo3(name = "product_indefinite")]
fn py_product_indefinite(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_product_indefinite(expr.id, k.id, &pool.inner).map_err(product_error_to_py)?
    };
    Ok(make_derived_result(
        py,
        derived,
        expr.pool.clone_ref(py),
        None,
    ))
}

#[pyfunction]
#[pyo3(name = "product_definite")]
fn py_product_definite(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
    lo: PyRef<PyExpr>,
    hi: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_product_definite(expr.id, k.id, lo.id, hi.id, &pool.inner)
            .map_err(product_error_to_py)?
    };
    Ok(make_derived_result(
        py,
        derived,
        expr.pool.clone_ref(py),
        None,
    ))
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
#[pyo3(name = "rsolve", signature = (equation, n, seq_name, initials=None))]
fn py_rsolve(
    py: Python<'_>,
    equation: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
    seq_name: &str,
    initials: Option<PyObject>,
) -> PyResult<PyExpr> {
    let init_storage: Option<BTreeMap<i64, ExprId>> = match initials {
        None => None,
        Some(obj) => {
            let b = obj.bind(py);
            if b.is_none() {
                None
            } else {
                let d = b.downcast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err("initials must be dict[int, Expr] or None")
                })?;
                let mut m = BTreeMap::new();
                for (k, v) in d.iter() {
                    let ki: i64 = k.extract()?;
                    let ve: PyRef<PyExpr> = v.extract()?;
                    m.insert(ki, ve.id);
                }
                Some(m)
            }
        }
    };
    let pool_py = equation.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        let raw = core_rsolve(
            &pool.inner,
            equation.id,
            n.id,
            seq_name,
            init_storage.as_ref(),
        )
        .map_err(rsolve_error_to_py)?;
        core_simplify(raw, &pool.inner).value
    };
    Ok(PyExpr { id, pool: pool_py })
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

// ---------------------------------------------------------------------------
// P1 item 7 — creative telescoping / holonomic (D-finite) machinery
// ---------------------------------------------------------------------------

fn holonomic_error_to_py(e: CoreHolonomicError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyHolonomicError>();
        make_structured_err(py, &exc_type, &e)
    })
}

/// Modular-evaluation errors, raised as the *same* Python `HolonomicError`.
///
/// `ModularError` is a separate Rust enum only because `HolonomicError` is
/// public and exhaustive, so adding variants to it is a major-version break.
/// The Python surface deliberately does not reflect that split: the codes
/// (`E-HOLO-006`/`007`/`008`) and the exception class are what a caller
/// catches, and both are unchanged.
fn holonomic_modular_error_to_py(e: CoreHolonomicModularError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyHolonomicError>();
        make_structured_err(py, &exc_type, &e)
    })
}

/// A **verified** Zeilberger certificate, returned by :func:`alkahest.zeilberger`.
///
/// Carries the recurrence coefficients ``a_0(n), …, a_J(n)`` and the rational
/// certificate ``R(n, k)`` satisfying, as an exact identity in ``Q(n)(k)``,
///
/// ``Σ_i a_i(n)·F(n+i, k) = G(n, k+1) − G(n, k)``  with  ``G = R·F``.
///
/// That identity is re-checked exactly before this object is constructed — a
/// returned certificate is a proof, not a numerical match.
///
/// **It is an identity in ``k``, and only that.**  A recurrence for the *sum*
/// ``S(n) = Σ_{k=k_lo}^{k_hi} F(n,k)`` is a second statement, and it is decided
/// separately over the range in :attr:`limits` — see :attr:`boundary`, which is
/// ``"vanishes"``, ``"nonzero"`` or ``"unknown"``:
///
/// * ``"vanishes"`` — proved; ``Σ_i a_i(n)·S(n+i) = 0`` holds for the sum.
/// * ``"nonzero"`` — proved; the recurrence is the **inhomogeneous**
///   ``Σ_i a_i(n)·S(n+i) = b(n)`` with ``b(n)`` returned as
///   :attr:`boundary_rhs`.  Still a theorem, just not the homogeneous one.
/// * ``"unknown"`` — neither was established.  **Nothing** follows about the
///   sum; :attr:`boundary_reason` says what stopped the proof.
///
/// The distinction is not cosmetic.  For ``Σ_{k=0}^{n} C(n,k)/(k+1)`` the
/// certificate is perfectly valid and the homogeneous recurrence is *false* —
/// the true relation is ``(n+2)·S(n+1) − (2n+2)·S(n) = 1``.  Reading a
/// recurrence off a certificate without this verdict is how a valid certificate
/// becomes a false theorem.
///
/// A verdict is also not a statement about every ``n``.  The range in
/// :attr:`limits` need not *be* a range at every ``n`` — ``k = 3..n−3`` runs
/// backwards at ``n = 3, 4`` — and the telescoping needs ``G = R·F`` to be
/// finite at every integer ``k`` in it.  :attr:`boundary_valid_from` and
/// :attr:`certificate_poles` carry those two bounds, rather than leaving a bare
/// ``b(n)`` with an implied "for every ``n``" attached to it.
#[pyclass(name = "ZeilbergerCertificate")]
struct PyZeilbergerCertificate {
    order: usize,
    order_is_minimal: bool,
    coeff_ids: Vec<ExprId>,
    certificate_id: ExprId,
    boundary_id: ExprId,
    pool: Py<PyExprPool>,
    derivation: String,
    /// Kept so that :meth:`boundary_at` can re-decide the hypothesis over a
    /// different range without re-running the search.
    result: CoreZeilbergerResult,
    term_id: ExprId,
    n_id: ExprId,
    k_id: ExprId,
    limits: (ExprId, ExprId),
    verdict: CoreBoundaryVerdict,
}

#[pymethods]
impl PyZeilbergerCertificate {
    /// Recurrence order ``J``; ``len(coeffs) == order + 1``.
    ///
    /// Minimal only when :attr:`order_is_minimal` says so — the default search
    /// does not establish it.
    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    /// Whether the search **established** that no lower-order relation exists.
    ///
    /// ``True`` means every order below :attr:`order` was refused at every
    /// certificate degree up to ``max_degree``.  ``False`` means *not
    /// established* — never "a lower order exists", since a lower-order
    /// relation that had been found would have been returned instead.
    ///
    /// The default search visits the ``(order, degree)`` grid cheapest-first,
    /// so it can reach a cheap order-2 probe before an expensive order-1 one;
    /// an order-2 result therefore does not rule out order 1 and this flag is
    /// ``False``.  It is ``True`` for order 1 (nothing is lower) and whenever
    /// the cost-ordered plan happened to exhaust every lower order first.  Pass
    /// ``minimal=True`` to :func:`alkahest.zeilberger` to search
    /// order-ascending and get the claim, at the cost of the low-order sweep.
    #[getter]
    fn order_is_minimal(&self) -> bool {
        self.order_is_minimal
    }

    /// ``[a_0(n), …, a_J(n)]`` — polynomial coefficients of the recurrence.
    #[getter]
    fn coeffs(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.coeff_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// ``R(n, k)`` — the rational certificate, with ``G(n,k) = R(n,k)·F(n,k)``.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyExpr {
        let _ = py;
        PyExpr {
            id: self.certificate_id,
            pool: self.pool.clone_ref(py),
        }
    }

    /// ``G(n, k) = R(n, k)·F(n, k)`` — the telescoped quantity.
    ///
    /// The recurrence for a *sum* over ``k = k_lo .. k_hi`` is
    /// ``Σ_i a_i(n)·S(n+i) = G(n, k_hi+1) − G(n, k_lo)``; substitute the two
    /// endpoints here to find out whether that difference vanishes, which is the
    /// hypothesis listed in :attr:`side_conditions`.
    #[getter]
    fn boundary_term(&self, py: Python<'_>) -> PyExpr {
        let _ = py;
        PyExpr {
            id: self.boundary_id,
            pool: self.pool.clone_ref(py),
        }
    }

    /// The summation range the boundary verdict is about, as
    /// ``(k_lo, k_hi)``.
    ///
    /// It defaults to ``(0, n)`` — the ``Σ_{k=0}^{n}`` convention the classical
    /// identities and the OEIS formula field use — and is echoed here so the
    /// assumption is on the record rather than silent.  A caller summing over
    /// anything else passes ``limits=`` and gets a verdict about *that* range.
    #[getter]
    fn limits(&self, py: Python<'_>) -> (PyExpr, PyExpr) {
        (
            PyExpr {
                id: self.limits.0,
                pool: self.pool.clone_ref(py),
            },
            PyExpr {
                id: self.limits.1,
                pool: self.pool.clone_ref(py),
            },
        )
    }

    /// ``"vanishes"``, ``"nonzero"`` or ``"unknown"`` — whether a recurrence for
    /// the *sum* over :attr:`limits` follows from this certificate, and which.
    ///
    /// See the class documentation for what each verdict licenses.  Only
    /// ``"unknown"`` means *no* statement about the sum may be made.
    #[getter]
    fn boundary(&self) -> &'static str {
        self.verdict.tag()
    }

    /// The smallest ``n`` this verdict is claimed for, or ``None``.
    ///
    /// A verdict is a statement about ``S(n)``, and the range in :attr:`limits`
    /// is not a range at every ``n``: ``k = 3..n−3`` runs *backwards* at
    /// ``n = 3`` and ``n = 4``, where a sum over it is ``0`` under one reading
    /// and a signed sum under the other.  The relation is false there, so those
    /// ``n`` are excluded instead of being claimed — this attribute is where the
    /// exclusion is recorded, and ``None`` means none was needed.
    ///
    /// It is a bound on ``n``, not a promise about it: the standing conditions
    /// in :attr:`side_conditions` still apply above it, and it only says
    /// anything next to a verdict — when :attr:`boundary` is ``"unknown"``
    /// nothing is claimed at any ``n`` regardless of what this holds.
    #[getter]
    fn boundary_valid_from(&self) -> Option<i64> {
        self.verdict.valid_from
    }

    /// Integer points ``k`` inside :attr:`limits` where the telescoping breaks,
    /// as expressions in ``n``.
    ///
    /// ``G(n,k) = R(n,k)·F(n,k)`` has to be finite at every integer ``k`` in the
    /// range for ``Σ_k (G(n,k+1) − G(n,k))`` to collapse to the two endpoints.
    /// A pole of the certificate at an interior point — ``k = (n+3)/2`` for
    /// ``C(n,k)/(n−2k+1)`` over ``k = 0..n``, an integer for every odd ``n`` —
    /// breaks it in the *middle* of the sum, where no boundary value can see it.
    /// Poles of the summand itself, which leave ``S(n)`` undefined, are listed
    /// here too.
    ///
    /// Non-empty implies :attr:`boundary` is ``"unknown"``.  Empty is not a
    /// proof that there are none: locations with a denominator past ``4``, or
    /// that only enter the range for large ``n``, are not searched for.
    #[getter]
    fn certificate_poles(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.verdict
            .certificate_poles
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// ``b(n)`` in ``Σ_i a_i(n)·S(n+i) = b(n)``, or ``None``.
    ///
    /// Present exactly when :attr:`boundary` is ``"nonzero"``.  When the verdict
    /// is ``"vanishes"`` the right-hand side is ``0`` and this is ``None``;
    /// when it is ``"unknown"`` there is no recurrence for the sum to write down.
    #[getter]
    fn boundary_rhs(&self, py: Python<'_>) -> Option<PyExpr> {
        match &self.verdict.status {
            CoreBoundaryStatus::Nonzero { rhs, .. } => Some(PyExpr {
                id: *rhs,
                pool: self.pool.clone_ref(py),
            }),
            _ => None,
        }
    }

    /// Why the boundary verdict came out as it did, in one sentence.
    ///
    /// For ``"unknown"`` this is what stopped the proof — "the limits were not
    /// supplied", "the certificate has a pole at the endpoint" — which is what
    /// tells a caller whether to retry with a better range or close the branch.
    #[getter]
    fn boundary_reason(&self) -> String {
        match &self.verdict.status {
            CoreBoundaryStatus::Vanishes => {
                "the boundary difference was proved to vanish in exact arithmetic".to_string()
            }
            CoreBoundaryStatus::Nonzero { witness_n, .. } => format!(
                "the boundary difference is not identically zero: b({witness_n}) != 0 in exact \
                 arithmetic"
            ),
            CoreBoundaryStatus::Unknown { reason } => reason.clone(),
        }
    }

    /// Whether a recurrence for the sum may be read off at all.
    ///
    /// ``True`` for ``"vanishes"`` (homogeneous) and ``"nonzero"``
    /// (inhomogeneous, with :attr:`boundary_rhs`), ``False`` for ``"unknown"``.
    #[getter]
    fn implies_sum_recurrence(&self) -> bool {
        self.verdict.implies_sum_recurrence()
    }

    /// Re-decide the boundary hypothesis over a different summation range.
    ///
    /// Returns a ``dict`` with the same keys as the attributes above —
    /// ``boundary``, ``rhs``, ``reason``, ``valid_from``, ``certificate_poles``,
    /// ``side_conditions`` — without re-running the search, which is the
    /// expensive half.  Use it to ask what the *same* certificate says about
    /// ``k = 0..n-1`` as well as ``k = 0..n``.
    #[pyo3(signature = (k_lo, k_hi))]
    fn boundary_at(
        &self,
        py: Python<'_>,
        k_lo: &Bound<'_, PyAny>,
        k_hi: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyDict>> {
        let lo = coerce_limit(py, &self.pool, k_lo, "k_lo")?;
        let hi = coerce_limit(py, &self.pool, k_hi, "k_hi")?;
        let (verdict, conditions) = {
            let pool = self.pool.borrow(py);
            let verdict = core_boundary_verdict(
                &self.result,
                self.term_id,
                self.n_id,
                self.k_id,
                Some((lo, hi)),
                &pool.inner,
            );
            let range = format_range(&pool.inner, lo, hi);
            let conditions = verdict.side_conditions(&range, &pool.inner);
            (verdict, conditions)
        };
        let out = PyDict::new_bound(py);
        out.set_item("boundary", verdict.tag())?;
        let rhs = match &verdict.status {
            CoreBoundaryStatus::Nonzero { rhs, .. } => Some(Py::new(
                py,
                PyExpr {
                    id: *rhs,
                    pool: self.pool.clone_ref(py),
                },
            )?),
            _ => None,
        };
        out.set_item("rhs", rhs)?;
        out.set_item(
            "reason",
            match &verdict.status {
                CoreBoundaryStatus::Vanishes => {
                    "the boundary difference was proved to vanish in exact arithmetic".to_string()
                }
                CoreBoundaryStatus::Nonzero { witness_n, .. } => {
                    format!("the boundary difference is not identically zero: b({witness_n}) != 0")
                }
                CoreBoundaryStatus::Unknown { reason } => reason.clone(),
            },
        )?;
        out.set_item("valid_from", verdict.valid_from)?;
        let poles: Vec<Py<PyExpr>> = verdict
            .certificate_poles
            .iter()
            .map(|&id| {
                Py::new(
                    py,
                    PyExpr {
                        id,
                        pool: self.pool.clone_ref(py),
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        out.set_item("certificate_poles", poles)?;
        out.set_item("side_conditions", conditions)?;
        Ok(out.unbind())
    }

    /// Hypotheses the certificate does **not** establish, as plain strings.
    ///
    /// Mirrors ``DerivedResult.verification["side_conditions"]``.  This tracks
    /// :attr:`boundary`: a discharged hypothesis, a refuted one and an open one
    /// read differently, so a loop that only looks at this list still cannot
    /// mistake the three.  It is never empty — even a proved boundary is a
    /// statement about the ``n`` at which everything involved is defined, and a
    /// permanent record of which range was assumed.
    #[getter]
    fn side_conditions(&self, py: Python<'_>) -> Vec<String> {
        let pool = self.pool.borrow(py);
        let range = format_range(&pool.inner, self.limits.0, self.limits.1);
        self.verdict.side_conditions(&range, &pool.inner)
    }

    /// Human-readable derivation log for the search that produced this.
    #[getter]
    fn derivation(&self) -> String {
        self.derivation.clone()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        let coeffs: Vec<String> = self
            .coeff_ids
            .iter()
            .map(|&id| pool.inner.display(id).to_string())
            .collect();
        format!(
            "ZeilbergerCertificate(order={}{}, boundary={}, coeffs=[{}], certificate={})",
            self.order,
            if self.order_is_minimal {
                " [minimal]"
            } else {
                ""
            },
            self.verdict.tag(),
            coeffs.join(", "),
            pool.inner.display(self.certificate_id)
        )
    }
}

/// A summation limit written as an `Expr` or as a plain Python `int`.
fn coerce_limit(
    py: Python<'_>,
    pool_py: &Py<PyExprPool>,
    v: &Bound<'_, PyAny>,
    which: &str,
) -> PyResult<ExprId> {
    if let Ok(e) = v.extract::<PyRef<PyExpr>>() {
        return Ok(e.id);
    }
    if let Ok(i) = v.extract::<i64>() {
        let pool = pool_py.borrow(py);
        return Ok(pool.inner.integer(i));
    }
    Err(PyTypeError::new_err(format!(
        "{which} must be an alkahest Expr or an int, got {}",
        v.get_type()
    )))
}

fn format_range(pool: &ExprPool, lo: ExprId, hi: ExprId) -> String {
    format!("k = {}..{}", pool.display(lo), pool.display(hi))
}

/// `alkahest.zeilberger(term, n, k, *, limits=None, max_order=4, max_degree=16, minimal=False) -> ZeilbergerCertificate`
///
/// Zeilberger's algorithm (creative telescoping) for a proper hypergeometric
/// term ``F(n, k)``.  Returns a **verified** certificate: the recurrence
/// ``Σ_i a_i(n)·F(n+i,k) = ΔG`` with ``G = R·F`` is re-checked as an exact
/// identity in ``Q(n)(k)`` before it is returned.
///
/// The verified statement is that identity in ``k``.  A recurrence for the
/// **sum** ``S(n) = Σ_{k=k_lo}^{k_hi} F(n,k)`` is a separate claim, and it is
/// decided here rather than left to the caller:
/// :attr:`~alkahest.ZeilbergerCertificate.boundary` is ``"vanishes"`` (the
/// homogeneous recurrence holds), ``"nonzero"`` (the inhomogeneous one does,
/// with ``b(n)`` in
/// :attr:`~alkahest.ZeilbergerCertificate.boundary_rhs`) or ``"unknown"``
/// (nothing follows about the sum).
///
/// ``limits`` is the summation range as ``(k_lo, k_hi)``, each an ``Expr`` or an
/// ``int``.  It **defaults to** ``(0, n)`` — ``Σ_{k=0}^{n}``, the convention the
/// classical identities and the OEIS formula field use — and the range actually
/// used is echoed back on
/// :attr:`~alkahest.ZeilbergerCertificate.limits`, so the assumption is on the
/// record rather than silent.  The verdict is about *that* range and changes
/// with it: truncating ``Σ_{k=0}^{n}`` to ``Σ_{k=0}^{n-1}`` generally turns
/// ``"vanishes"`` into ``"nonzero"``.  A range this analysis cannot place —
/// endpoints that are not integer-affine in ``n`` — is ``"unknown"``, never
/// ``"vanishes"``.
///
/// ``max_order`` and ``max_degree`` are upper **bounds**, not starting points.
/// The search visits the ``(order, degree)`` grid by iterative deepening,
/// cheapest candidate first, and returns the first relation that passes exact
/// verification — so raising either bound widens what can be found without
/// slowing down an input that was already decided at a low order and degree.
///
/// **The order it returns is not claimed to be minimal.**  Cheapest-first means
/// a cheap order-2 probe can be reached before an expensive order-1 one, so an
/// order-2 result does not rule out an order-1 relation;
/// :attr:`~alkahest.ZeilbergerCertificate.order_is_minimal` is ``False`` to say
/// so rather than leaving it to be assumed.  Pass ``minimal=True`` to search
/// **order-ascending** instead — every degree ``0..=max_degree`` at order ``J``
/// is refused before order ``J+1`` is tried — which makes a returned order
/// genuinely minimal and sets the flag.  That is the hopeless low-order sweep
/// the default plan exists to avoid, and it costs accordingly; ask for it when
/// minimality is the result, not as a habit.
///
/// Raises :exc:`alkahest.HolonomicError` rather than guessing when ``term`` is
/// outside the proper hypergeometric class (``E-HOLO-001``) or when the bounded
/// search is exhausted (``E-HOLO-002``).
// Eight parameters, of which five are keyword-only with defaults on the Python
// side. `clippy::too_many_arguments` is about call sites that have to remember
// an order; this signature has none, so collapsing it into an options struct
// would only put a Rust shape between the caller and the keywords they type.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    name = "zeilberger",
    signature = (term, n, k, *, limits = None, max_order = 4, max_degree = 16, minimal = false)
)]
fn py_zeilberger(
    py: Python<'_>,
    term: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
    limits: Option<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
    max_order: usize,
    max_degree: usize,
    minimal: bool,
) -> PyResult<PyZeilbergerCertificate> {
    let pool_py = term.pool.clone_ref(py);
    let opts = CoreZeilbergerOpts {
        max_order,
        max_degree,
    };
    let search = if minimal {
        CoreOrderSearch::MinimalOrder
    } else {
        CoreOrderSearch::CostOrdered
    };
    // The default is stated, not inferred: `Σ_{k=0}^{n}`, echoed back on the
    // result so a caller summing over something else sees the mismatch.
    let limits = match limits {
        Some((lo, hi)) => (
            coerce_limit(py, &pool_py, &lo, "limits[0]")?,
            coerce_limit(py, &pool_py, &hi, "limits[1]")?,
        ),
        None => {
            let pool = pool_py.borrow(py);
            core_natural_limits(n.id, &pool.inner)
        }
    };
    let (
        order,
        order_is_minimal,
        coeff_ids,
        certificate_id,
        boundary_id,
        derivation,
        result,
        status,
    ) = {
        let pool = pool_py.borrow(py);
        let derived = core_zeilberger_search(term.id, n.id, k.id, &pool.inner, &opts, search)
            .map_err(holonomic_error_to_py)?;
        let derivation = derived.log.display_with(&pool.inner).to_string();
        let report = derived.value;
        let boundary = core_boundary_term(&report.result, term.id, &pool.inner);
        let status = core_boundary_verdict(
            &report.result,
            term.id,
            n.id,
            k.id,
            Some(limits),
            &pool.inner,
        );
        (
            report.result.order,
            report.order_is_minimal,
            report.result.coeffs.clone(),
            report.result.certificate,
            boundary,
            derivation,
            report.result,
            status,
        )
    };
    Ok(PyZeilbergerCertificate {
        order,
        order_is_minimal,
        coeff_ids,
        certificate_id,
        boundary_id,
        pool: pool_py,
        derivation,
        result,
        term_id: term.id,
        n_id: n.id,
        k_id: k.id,
        limits,
        verdict: status,
    })
}

// ---------------------------------------------------------------------------
// M4(b) — q-analogue creative telescoping (q-Zeilberger)
// ---------------------------------------------------------------------------

/// A **verified** ``q``-Zeilberger certificate, from
/// :func:`alkahest.experimental.q_zeilberger`.
///
/// The recurrence ``Σ_i a_i(q**n)·F(n+i,k) = G(n,k+1) − G(n,k)`` with
/// ``G = R·F`` is re-checked as an exact identity in ``Q(q)(q**n)(q**k)``
/// before this object is constructed, exactly as the classical
/// :class:`~alkahest.ZeilbergerCertificate` is.
///
/// **The verdict on the sum is two-valued here**, not three:
/// :attr:`boundary` is ``"vanishes"`` (proved: ``Σ_i a_i(q**n)·S(n+i) = 0``
/// for ``S(n) = Σ_{k ∈ Z} F(n,k)``, a finite sum over the proved
/// :attr:`support` window) or ``"unknown"`` (nothing may be claimed about the
/// sum). There is no ``"nonzero"`` arm: an inhomogeneity ``b(n)`` for a
/// ``q``-sum needs endpoint values that are not rational in ``q**n``, and an
/// unproved ``b(n)`` would be worse than none.
///
/// ``q`` is treated as **transcendental** throughout. A verdict is an identity
/// in ``Q(q)``; it does not license specialising ``q`` to a root of unity,
/// which is a separate step with its own hypotheses.
#[pyclass(name = "QZeilbergerCertificate")]
struct PyQZeilbergerCertificate {
    order: usize,
    order_is_minimal: bool,
    probes: usize,
    coeff_ids: Vec<ExprId>,
    certificate_id: ExprId,
    pool: Py<PyExprPool>,
    derivation: String,
    q_id: ExprId,
    cert: CoreQCertificate,
    n_min: i64,
}

#[pymethods]
impl PyQZeilbergerCertificate {
    /// Recurrence order ``J``; ``len(coeffs) == order + 1``.
    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    /// Whether the search **established** that no lower-order relation exists.
    ///
    /// ``False`` means *not established*, never "a lower order exists" — the
    /// same convention as the classical certificate's.
    #[getter]
    fn order_is_minimal(&self) -> bool {
        self.order_is_minimal
    }

    /// How many ``(order, degree)`` probes the search made.
    #[getter]
    fn probes(&self) -> usize {
        self.probes
    }

    /// ``[a_0, …, a_J]`` — the recurrence coefficients, as expressions in
    /// ``q`` and ``q**n``.
    #[getter]
    fn coeffs(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.coeff_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// ``R`` — the rational certificate, with ``G(n,k) = R·F(n,k)``.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyExpr {
        PyExpr {
            id: self.certificate_id,
            pool: self.pool.clone_ref(py),
        }
    }

    /// ``"vanishes"`` or ``"unknown"`` — whether a recurrence for the *sum*
    /// follows from this certificate.
    #[getter]
    fn boundary(&self) -> &'static str {
        self.cert.boundary.tag()
    }

    /// Whether a recurrence for the sum may be read off at all.
    #[getter]
    fn implies_sum_recurrence(&self) -> bool {
        self.cert.boundary.implies_sum_recurrence()
    }

    /// Why the boundary verdict came out as it did, in one sentence.
    #[getter]
    fn boundary_reason(&self) -> String {
        match &self.cert.boundary {
            CoreQBoundaryStatus::Vanishes { n_min, .. } => format!(
                "the summand was proved to have finite support in k and to be finite at every \
                 integer k, so the sum over all integer k obeys the homogeneous recurrence for \
                 every n >= {n_min}"
            ),
            CoreQBoundaryStatus::Unknown { reason } => reason.clone(),
        }
    }

    /// The proved support window ``(lo, hi)`` in ``k`` as strings in ``n``, or
    /// ``None`` when the verdict is ``"unknown"`` or the window is not a single
    /// affine bound on each side.
    ///
    /// ``S(n) = Σ_{k ∈ Z} F(n,k)`` is the sum the verdict is about, and this is
    /// the finite range it equals: the summand was proved to vanish outside it.
    #[getter]
    fn support(&self) -> Option<(String, String)> {
        match &self.cert.boundary {
            CoreQBoundaryStatus::Vanishes { support, .. } => support.clone(),
            CoreQBoundaryStatus::Unknown { .. } => None,
        }
    }

    /// Hypotheses the certificate does **not** establish, as plain strings.
    #[getter]
    fn side_conditions(&self) -> Vec<String> {
        self.cert.boundary.side_conditions()
    }

    /// Human-readable derivation log.
    #[getter]
    fn derivation(&self) -> String {
        self.derivation.clone()
    }

    /// ``S(n0) = Σ_{k ∈ Z} F(n0, k)`` as an exact polynomial in ``q``.
    ///
    /// Computed from the definition of the ``q``-Pochhammer symbol, **not**
    /// through the shift quotients the search used — which is what makes
    /// checking the returned recurrence against these values an independent
    /// check rather than a restatement of the certificate.
    ///
    /// Raises :exc:`alkahest.HolonomicError` when the support window is not
    /// established, since then the sum is not a finite sum to evaluate.
    fn sum_term(&self, py: Python<'_>, n0: i64) -> PyResult<PyExpr> {
        let pool = self.pool.borrow(py);
        let value = self
            .cert
            .term
            .sum_at(n0, self.n_min)
            .map_err(q_holonomic_error_to_py)?;
        let id = alkahest_core::holonomic::hyperterm::rn_to_expr(&pool.inner, self.q_id, &value);
        let id = alkahest_core::simplify::simplify(id, &pool.inner).value;
        Ok(PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        })
    }

    /// ``specialize_at_root_of_unity(d, n) -> QRootOfUnitySpecialization``
    ///
    /// Decide whether the proved ``Q(q)`` recurrence survives ``q = ζ_d``, a
    /// primitive ``d``-th root of unity — the step between what this
    /// certificate proves and the ``q``-supercongruence literature.
    ///
    /// The hypotheses (no pole in any ``a_i(q**n)`` or ``S(n+i)`` at ``ζ_d``)
    /// are decided **exactly**, by polynomial divisibility by ``Φ_d(q)`` over
    /// ``Q``; nothing is evaluated numerically. The result carries a
    /// three-valued verdict — ``"specializes"``, ``"obstructed"``,
    /// ``"unknown"`` — and refuses to offer a specialised value when a
    /// hypothesis fails.
    ///
    /// ``d = 1`` means ``ζ_1 = 1``: the classical ``q → 1`` limit.
    #[pyo3(signature = (d, n))]
    fn specialize_at_root_of_unity(
        &self,
        py: Python<'_>,
        d: u32,
        n: i64,
    ) -> PyResult<PyQRootOfUnitySpecialization> {
        let spec = core_q_specialize_at_root_of_unity(&self.cert, d, n)
            .map_err(q_holonomic_error_to_py)?;
        Ok(PyQRootOfUnitySpecialization {
            spec,
            pool: self.pool.clone_ref(py),
            q_id: self.q_id,
        })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        let coeffs: Vec<String> = self
            .coeff_ids
            .iter()
            .map(|&id| pool.inner.display(id).to_string())
            .collect();
        format!(
            "QZeilbergerCertificate(order={}{}, boundary={}, coeffs=[{}])",
            self.order,
            if self.order_is_minimal {
                " [minimal]"
            } else {
                ""
            },
            self.cert.boundary.tag(),
            coeffs.join(", ")
        )
    }
}

/// The verdict on specialising a ``q``-Zeilberger certificate at a primitive
/// ``d``-th root of unity, from
/// :meth:`~alkahest.experimental.QZeilbergerCertificate.specialize_at_root_of_unity`.
///
/// ``q``-Zeilberger proves identities with ``q`` **transcendental**. Setting
/// ``q = ζ_d`` is a separate step with its own hypotheses, and specialising at
/// a point where a denominator vanishes produces a confidently wrong statement.
/// This object is that step, taken as a decision:
///
/// * :attr:`status` ``== "specializes"`` — proved. Every ``a_i(q**n)`` and
///   every ``S(n+i)`` was shown to have non-negative ``Φ_d``-adic valuation, so
///   the specialisation map is defined on all of them, and
///   ``Σ_i a_i(ζ**n)·S_ζ(n+i) = 0`` in ``Q(ζ_d)`` — re-checked in exact
///   cyclotomic arithmetic before this object was built.
/// * ``"obstructed"`` — a pole at ``ζ_d`` was **exhibited**. No specialised
///   value is offered. This is not a proof that the specialised identity is
///   false, only that this route to it is blocked.
/// * ``"unknown"`` — nothing follows.
///
/// Three further things are reported rather than hidden, because each of them
/// makes a true verdict mean less than it looks:
/// :attr:`is_vacuous` (every coefficient died, so the recurrence is ``0 = 0``),
/// :attr:`leading_coefficient_survives` (``False`` means the recurrence no
/// longer determines the last value), and :attr:`support_shrinks` (``q``-Lucas
/// killed terms the generic identity needs — ``[2;1]_q = 1 + q`` is non-zero in
/// ``Q(q)`` and zero at ``ζ_2``).
///
/// :meth:`sum_valuation` is the ``q``-supercongruence quantity: it is the exact
/// integer ``v`` with ``Φ_d(q)**v`` dividing ``S(n)`` and ``Φ_d(q)**(v+1)`` not.
#[pyclass(name = "QRootOfUnitySpecialization")]
struct PyQRootOfUnitySpecialization {
    spec: CoreQRootOfUnitySpecialization,
    pool: Py<PyExprPool>,
    q_id: ExprId,
}

impl PyQRootOfUnitySpecialization {
    fn poly_to_expr(
        &self,
        py: Python<'_>,
        p: &alkahest_core::matrix::normal_form::RatUniPoly,
    ) -> PyExpr {
        let pool = self.pool.borrow(py);
        let id = alkahest_core::holonomic::hyperterm::ratuni_to_expr(&pool.inner, self.q_id, p);
        PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        }
    }

    fn checked_index(&self, i: usize) -> PyResult<usize> {
        if !self.spec.specializes() {
            return Err(PyValueError::new_err(format!(
                "the specialisation is \"{}\", so no specialised value exists: {}",
                self.spec.status.tag(),
                self.spec.status.reason()
            )));
        }
        if i >= self.spec.sums.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "shift index {i} is out of range for a recurrence of order {}",
                self.spec.sums.len().saturating_sub(1)
            )));
        }
        Ok(i)
    }
}

#[pymethods]
impl PyQRootOfUnitySpecialization {
    /// The order ``d`` of the root of unity.
    #[getter]
    fn d(&self) -> u32 {
        self.spec.d
    }

    /// The index ``n`` the verdict is about; the recurrence relates
    /// ``n … n + order``.
    #[getter]
    fn n(&self) -> i64 {
        self.spec.n0
    }

    /// ``"specializes"``, ``"obstructed"`` or ``"unknown"``.
    #[getter]
    fn status(&self) -> &'static str {
        self.spec.status.tag()
    }

    /// Whether a specialised recurrence may be claimed at all.
    #[getter]
    fn specializes(&self) -> bool {
        self.spec.specializes()
    }

    /// Why the verdict came out as it did; ``""`` when it specialises.
    #[getter]
    fn reason(&self) -> String {
        self.spec.status.reason().to_string()
    }

    /// Whether the specialised recurrence is ``0 = 0``.
    ///
    /// ``True`` means every ``a_i(ζ**n)`` vanished. The statement is still a
    /// theorem; it simply constrains nothing, and reading :attr:`specializes`
    /// without reading this would be claiming more than there is.
    #[getter]
    fn is_vacuous(&self) -> bool {
        self.spec.is_vacuous()
    }

    /// Whether the leading coefficient ``a_J(ζ**n)`` survives, i.e. whether the
    /// specialised recurrence still determines ``S_ζ(n+J)``.
    #[getter]
    fn leading_coefficient_survives(&self) -> bool {
        self.spec.leading_coefficient_survives()
    }

    /// Whether ``S_ζ`` is also the sum of the *specialised summands*.
    ///
    /// ``False`` means some summand inside the window has a pole at ``ζ_d``:
    /// :meth:`sum_value` is still the correct image of the exact ``Q(q)`` sum,
    /// but writing it as ``Σ_k F_ζ(n,k)`` would be writing down an undefined
    /// expression.
    #[getter]
    fn is_termwise_regular(&self) -> bool {
        self.spec.is_termwise_regular()
    }

    /// Whether ``q``-Lucas killed at least one term the generic identity needs.
    #[getter]
    fn support_shrinks(&self) -> bool {
        self.spec.support_shrinks()
    }

    /// The proved generic support window ``(lo, hi)`` in ``k`` at this ``n``,
    /// or ``None`` when it was not established.
    #[getter]
    fn window(&self) -> Option<(i64, i64)> {
        self.spec.window
    }

    /// The ``k`` inside that window at which the summand is **still non-zero**
    /// at ``ζ_d`` — the effective window, which ``q``-Lucas can shrink.
    #[getter]
    fn effective_support(&self) -> Vec<i64> {
        self.spec.effective_support.clone()
    }

    /// Hypotheses and caveats this verdict does not discharge, as plain strings.
    #[getter]
    fn side_conditions(&self) -> Vec<String> {
        self.spec.side_conditions()
    }

    /// ``Φ_d(q)`` — the cyclotomic polynomial the arithmetic is modulo.
    ///
    /// Exposed so a caller can redo the whole check by hand: two elements of
    /// ``Q(ζ_d)`` are equal exactly when their canonical representatives (what
    /// :meth:`sum_value` returns, of degree ``< φ(d)``) are equal.
    fn modulus(&self, py: Python<'_>) -> PyExpr {
        self.poly_to_expr(py, self.spec.field.modulus())
    }

    /// ``S_ζ(n + i)`` — the specialised sum, as its canonical representative in
    /// ``Q[q]`` of degree ``< φ(d)``.
    ///
    /// Raises :exc:`ValueError` unless the verdict is ``"specializes"``: an
    /// obstructed specialisation has no value to report, and returning one
    /// anyway is exactly the mistake this class exists to prevent.
    #[pyo3(signature = (i = 0))]
    fn sum_value(&self, py: Python<'_>, i: usize) -> PyResult<PyExpr> {
        let i = self.checked_index(i)?;
        Ok(self.poly_to_expr(py, &self.spec.sums[i].poly))
    }

    /// ``a_i(ζ**n)`` — the specialised recurrence coefficient.
    #[pyo3(signature = (i = 0))]
    fn coefficient(&self, py: Python<'_>, i: usize) -> PyResult<PyExpr> {
        let i = self.checked_index(i)?;
        Ok(self.poly_to_expr(py, &self.spec.coeffs[i].poly))
    }

    /// ``v_{Φ_d}(S(n + i))`` — the exact ``Φ_d``-adic valuation of the
    /// **generic** sum, or ``None`` when ``S(n+i)`` is identically zero.
    ///
    /// This is the ``q``-supercongruence statement in its exact form:
    /// ``v >= r`` is precisely ``Φ_d(q)**r`` divides ``S(n+i)``, and ``v < 0``
    /// is the pole that obstructs specialisation. Available even when the
    /// verdict is ``"obstructed"`` — a negative valuation *is* the obstruction.
    #[pyo3(signature = (i = 0))]
    fn sum_valuation(&self, i: usize) -> PyResult<Option<i64>> {
        self.spec.sum_valuations.get(i).copied().ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("shift index {i} is out of range"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "QRootOfUnitySpecialization(d={}, n={}, status={}{}{})",
            self.spec.d,
            self.spec.n0,
            self.spec.status.tag(),
            if self.spec.is_vacuous() {
                ", vacuous"
            } else if self.spec.specializes() && !self.spec.leading_coefficient_survives() {
                ", leading coefficient vanishes"
            } else {
                ""
            },
            if self.spec.support_shrinks() {
                ", support shrinks"
            } else {
                ""
            }
        )
    }
}

/// `alkahest.experimental.cyclotomic_polynomial(pool, d, var=None) -> Expr`
///
/// The ``d``-th cyclotomic polynomial ``Φ_d(q)``, monic of degree ``φ(d)``,
/// with exact integer coefficients.
///
/// This is the modulus the root-of-unity machinery works over: ``Φ_d`` is
/// irreducible over ``Q``, so for a polynomial ``p`` the statement
/// ``p(ζ_d) = 0`` is *exactly* the divisibility ``Φ_d | p``, which is how
/// :meth:`~alkahest.experimental.QZeilbergerCertificate.specialize_at_root_of_unity`
/// decides its hypotheses without evaluating anything numerically.
///
/// ``var`` names the variable; it defaults to a symbol called ``q``.
#[pyfunction]
#[pyo3(name = "cyclotomic_polynomial", signature = (pool, d, var = None))]
fn py_cyclotomic_polynomial(
    py: Python<'_>,
    pool: Py<PyExprPool>,
    d: u32,
    var: Option<PyRef<PyExpr>>,
) -> PyResult<PyExpr> {
    if d == 0 {
        return Err(PyValueError::new_err(
            "the order of a root of unity must be at least 1",
        ));
    }
    if d > alkahest_core::holonomic::qzeil::MAX_CYCLOTOMIC_ORDER {
        return Err(PyValueError::new_err(format!(
            "the order must be at most {}, got {d}",
            alkahest_core::holonomic::qzeil::MAX_CYCLOTOMIC_ORDER
        )));
    }
    let phi = core_cyclotomic_polynomial(d);
    let id = {
        let p = pool.borrow(py);
        let v = match &var {
            Some(v) => v.id,
            None => p.inner.symbol("q", alkahest_core::kernel::Domain::Real),
        };
        alkahest_core::holonomic::hyperterm::ratuni_to_expr(&p.inner, v, &phi)
    };
    Ok(PyExpr {
        id,
        pool: pool.clone_ref(py),
    })
}

fn q_holonomic_error_to_py(e: CoreQHolonomicError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyHolonomicError>();
        make_structured_err(py, &exc_type, &e)
    })
}

/// `alkahest.experimental.q_zeilberger(term, q, n, k, *, max_order=3, max_degree=6, minimal=False, n_min=0) -> QZeilbergerCertificate`
///
/// ``q``-Zeilberger's algorithm: a **verified** ``q``-recurrence for a
/// ``q``-hypergeometric term ``F(n, k)``, plus a verdict on whether it carries
/// over to the sum.
///
/// The supported class, enforced by the parser rather than assumed:
///
/// ```text
/// F(n,k) = R(q**n, q**k) · z**k · w**n · q**(A*k² + B*n*k + C*n² + D*k + E*n)
///          · Π_j qpochhammer(u_j, d_j, v_j)**e_j
/// ```
///
/// written with the function heads ``pool.func("qbinomial", [N, K])`` (the
/// Gaussian binomial) and ``pool.func("qpochhammer", [u, d, v])`` (meaning
/// ``(q**u; q**d)_v``), powers of ``q`` whose exponent is a degree-≤2
/// polynomial in ``n`` and ``k``, powers with a base free of ``n`` and ``k``,
/// and any rational function of ``q``, ``q**n``, ``q**k``.
///
/// Anything else raises :exc:`alkahest.HolonomicError` rather than being
/// answered: ``E-HOLO-020`` outside the class (a bare ``n`` or ``k``, a
/// ``gamma``, a ``sin``), ``E-HOLO-021`` when the bounded search is exhausted,
/// ``E-HOLO-023`` for a malformed call, and ``E-HOLO-024`` for an input that
/// looks like the class but whose shift quotient is not rational — the
/// canonical case being a ``q``-Pochhammer whose first argument shifts by
/// something its base does not divide, e.g. ``(q; q**2)_k`` under ``k ↦ k+1``.
///
/// ``n_min`` is the smallest ``n`` the boundary verdict is asserted for; it
/// defaults to ``0`` and is echoed in
/// :attr:`~alkahest.experimental.QZeilbergerCertificate.side_conditions`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    name = "q_zeilberger",
    signature = (term, q, n, k, *, max_order = 3, max_degree = 6, minimal = false, n_min = 0)
)]
fn py_q_zeilberger(
    py: Python<'_>,
    term: PyRef<PyExpr>,
    q: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
    max_order: usize,
    max_degree: usize,
    minimal: bool,
    n_min: i64,
) -> PyResult<PyQZeilbergerCertificate> {
    let pool_py = term.pool.clone_ref(py);
    let opts = CoreQZeilbergerOpts {
        max_order,
        max_degree,
        search: if minimal {
            CoreOrderSearch::MinimalOrder
        } else {
            CoreOrderSearch::CostOrdered
        },
        n_min,
    };
    let (cert, derivation) = {
        let pool = pool_py.borrow(py);
        let derived = core_q_zeilberger(term.id, q.id, n.id, k.id, &pool.inner, &opts)
            .map_err(q_holonomic_error_to_py)?;
        let derivation = derived.log.display_with(&pool.inner).to_string();
        (derived.value, derivation)
    };
    Ok(PyQZeilbergerCertificate {
        order: cert.report.result.order,
        order_is_minimal: cert.report.order_is_minimal,
        probes: cert.report.probes,
        coeff_ids: cert.report.result.coeffs.clone(),
        certificate_id: cert.report.result.certificate,
        pool: pool_py,
        derivation,
        q_id: q.id,
        cert,
        n_min,
    })
}

// ---------------------------------------------------------------------------
// M4 — double-sum (Apagodu–Zeilberger) creative telescoping
// ---------------------------------------------------------------------------

fn telescoping2d_error_to_py(e: CoreTelescoping2dError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyHolonomicError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn format_rect_range(pool: &ExprPool, lo: ExprId, hi: ExprId, var: &str) -> String {
    format!("{var} = {}..{}", pool.display(lo), pool.display(hi))
}

/// A **verified** double-sum creative-telescoping certificate, returned by
/// :func:`alkahest.experimental.telescope2d`.
///
/// Carries the recurrence coefficients ``a_0(n), …, a_J(n)`` and two rational
/// certificates ``cert1``, ``cert2`` satisfying, as an exact identity in
/// ``Q(n,j,k)``,
///
/// ``Σ_i a_i(n)·F(n+i,j,k) = Δ_j G_1 + Δ_k G_2``,
/// ``G_1 = cert1·F``, ``G_2 = cert2·F``.
///
/// That identity is re-checked exactly before this object is constructed. It
/// says nothing on its own about ``S(n) = Σ_j Σ_k F(n,j,k)`` over a stated
/// rectangle — call :meth:`boundary_status` to decide that, separately, the
/// same three-valued discipline as :class:`~alkahest.ZeilbergerCertificate`
/// uses for the single-sum case.
///
/// This is a genuinely scoped-down engine, not a full Wegschaider reduction:
/// see the module docs in ``alkahest_cas::holonomic::telescoping2d`` (Rust)
/// for exactly what proper hypergeometric class, ansatz degree budget,
/// certificate-denominator ansatz and boundary restrictions (constant
/// rectangles only) it operates under.
#[pyclass(name = "Telescoping2dCertificate")]
struct PyTelescoping2dCertificate {
    order: usize,
    coeff_ids: Vec<ExprId>,
    cert1_id: ExprId,
    cert2_id: ExprId,
    pool: Py<PyExprPool>,
    term_id: ExprId,
    n_id: ExprId,
    j_id: ExprId,
    k_id: ExprId,
    result: CoreTelescoping2dResult,
}

#[pymethods]
impl PyTelescoping2dCertificate {
    /// Recurrence order ``J``; ``len(coeffs) == order + 1``.
    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    /// ``[a_0(n), …, a_J(n)]`` — polynomial coefficients of the recurrence.
    #[getter]
    fn coeffs(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.coeff_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// ``c_1(n,j,k)``, with ``G_1 = c_1·F``.
    #[getter]
    fn cert1(&self, py: Python<'_>) -> PyExpr {
        let _ = py;
        PyExpr {
            id: self.cert1_id,
            pool: self.pool.clone_ref(py),
        }
    }

    /// ``c_2(n,j,k)``, with ``G_2 = c_2·F``.
    #[getter]
    fn cert2(&self, py: Python<'_>) -> PyExpr {
        let _ = py;
        PyExpr {
            id: self.cert2_id,
            pool: self.pool.clone_ref(py),
        }
    }

    /// Decide the boundary hypothesis for the double sum
    /// ``S(n) = Σ_{j=j_lo}^{j_hi} Σ_{k=k_lo}^{k_hi} F(n,j,k)``.
    ///
    /// ``j_lo, j_hi, k_lo, k_hi`` must be **integer constants** (``Expr`` or
    /// plain ``int``) — not expressions in ``n``. This is a real limitation,
    /// not unfinished polish: see the Rust module docs for
    /// ``telescoping2d::boundary`` for why, and for the standard workaround
    /// (a fixed bound safely larger than the true combinatorial support) when
    /// the natural range is `n`-dependent, e.g. ``j = 0..n``.
    ///
    /// Returns a ``dict`` with keys ``status`` (``"vanishes"``, ``"nonzero"``
    /// or ``"unknown"`` — though this version never produces ``"nonzero"``,
    /// see the class docs), ``implies_sum_recurrence`` and
    /// ``side_conditions``.
    #[pyo3(signature = (j_lo, j_hi, k_lo, k_hi))]
    fn boundary_status(
        &self,
        py: Python<'_>,
        j_lo: &Bound<'_, PyAny>,
        j_hi: &Bound<'_, PyAny>,
        k_lo: &Bound<'_, PyAny>,
        k_hi: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyDict>> {
        let jlo = coerce_limit(py, &self.pool, j_lo, "j_lo")?;
        let jhi = coerce_limit(py, &self.pool, j_hi, "j_hi")?;
        let klo = coerce_limit(py, &self.pool, k_lo, "k_lo")?;
        let khi = coerce_limit(py, &self.pool, k_hi, "k_hi")?;
        let (status, j_range, k_range) = {
            let pool = self.pool.borrow(py);
            let status = core_boundary_status_2d(
                &self.result,
                self.term_id,
                self.n_id,
                self.j_id,
                self.k_id,
                (jlo, jhi),
                (klo, khi),
                &pool.inner,
            );
            (
                status,
                format_rect_range(&pool.inner, jlo, jhi, "j"),
                format_rect_range(&pool.inner, klo, khi, "k"),
            )
        };
        let out = PyDict::new_bound(py);
        out.set_item("status", status.tag())?;
        out.set_item("implies_sum_recurrence", status.implies_sum_recurrence())?;
        out.set_item(
            "side_conditions",
            status.side_conditions(&j_range, &k_range),
        )?;
        Ok(out.unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        let coeffs: Vec<String> = self
            .coeff_ids
            .iter()
            .map(|&id| pool.inner.display(id).to_string())
            .collect();
        format!(
            "Telescoping2dCertificate(order={}, coeffs=[{}], cert1={}, cert2={})",
            self.order,
            coeffs.join(", "),
            pool.inner.display(self.cert1_id),
            pool.inner.display(self.cert2_id)
        )
    }
}

/// `alkahest.experimental.telescope2d(term, n, j, k, *, max_order=2, max_a_degree=2, max_cert_degree=3) -> Telescoping2dCertificate`
///
/// Double-sum creative telescoping (Apagodu–Zeilberger) for a proper
/// hypergeometric term ``F(n, j, k)`` — the two-bound-index generalization of
/// :func:`alkahest.zeilberger`. Returns a **verified** certificate: the
/// identity ``Σ_i a_i(n)·F(n+i,j,k) = Δ_j G_1 + Δ_k G_2`` is re-checked
/// exactly in ``Q(n,j,k)`` before it is returned.
///
/// The supported class: rational prefactor times ``z_j**j·z_k**k·w**n`` times
/// ``gamma(a*n+b*j+c*k+d)**e`` factors with integer ``a, b, c`` (also reached
/// via ``factorial``, ``binomial``, ``pochhammer``) — exactly
/// :func:`alkahest.zeilberger`'s class, generalized from two indices to
/// three. No more than two bound indices, and no general Wegschaider
/// reduction; see the class docs and the Rust module docs
/// (``alkahest_cas::holonomic::telescoping2d``) for the full, honestly-stated
/// scope.
///
/// ``max_order``, ``max_a_degree`` (degree bound on each ``a_i(n)``) and
/// ``max_cert_degree`` (box degree bound, in each of ``n, j, k``
/// independently, on the two certificate numerators) are genuine upper
/// bounds on a plain ascending search — raising them admits harder inputs.
/// Raises :exc:`alkahest.HolonomicError` (``E-HOLO-040`` outside the
/// supported class, ``E-HOLO-041`` when the bounded search is exhausted,
/// ``E-HOLO-042`` for a malformed call) rather than guessing.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    name = "telescope2d",
    signature = (term, n, j, k, *, max_order = 2, max_a_degree = 2, max_cert_degree = 3)
)]
fn py_telescope2d(
    py: Python<'_>,
    term: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
    j: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
    max_order: usize,
    max_a_degree: usize,
    max_cert_degree: usize,
) -> PyResult<PyTelescoping2dCertificate> {
    let pool_py = term.pool.clone_ref(py);
    let opts = CoreTelescoping2dOpts {
        max_order,
        max_a_degree,
        max_cert_degree,
    };
    let result = {
        let pool = pool_py.borrow(py);
        core_telescope2d_search(term.id, n.id, j.id, k.id, &pool.inner, &opts)
            .map_err(telescoping2d_error_to_py)?
    };
    Ok(PyTelescoping2dCertificate {
        order: result.order,
        coeff_ids: result.coeffs.clone(),
        cert1_id: result.cert1,
        cert2_id: result.cert2,
        pool: pool_py,
        term_id: term.id,
        n_id: n.id,
        j_id: j.id,
        k_id: k.id,
        result,
    })
}

/// A **verified** `m`-bound-index creative-telescoping certificate, returned
/// by :func:`alkahest.experimental.telescope_md`. The general form of
/// :class:`Telescoping2dCertificate` (`m = 2` is that class's special case,
/// unchanged and still returned by :func:`~alkahest.experimental.telescope2d`
/// itself).
///
/// Carries the recurrence coefficients ``a_0(n), …, a_J(n)`` and `m` rational
/// certificates ``c_1, …, c_m`` satisfying, as an exact identity in
/// ``Q(n, x_1, …, x_m)``, ``Σ_i a_i(n)·F(n+i,x) = Σ_t Δ_t(c_t·F)``. That
/// identity is re-checked exactly before this object is constructed. It says
/// nothing on its own about the `m`-fold sum over a stated box — call
/// :meth:`boundary_status` to decide that, separately.
///
/// See the Rust module docs (``alkahest_cas::holonomic::telescoping2d``) for
/// the complete, honestly-stated scope: proper hypergeometric summands only,
/// no general Wegschaider reduction, a fixed (non-minimal) certificate
/// denominator, constant-box-only boundary analysis, and the resource
/// ceilings (`MAX_ANSATZ_UNKNOWNS`, `MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS`)
/// that keep a search with no certificate in reach a fast, honest refusal
/// rather than an unbounded computation.
#[pyclass(name = "TelescopingMdCertificate")]
struct PyTelescopingMdCertificate {
    order: usize,
    coeff_ids: Vec<ExprId>,
    cert_ids: Vec<ExprId>,
    pool: Py<PyExprPool>,
    term_id: ExprId,
    n_id: ExprId,
    index_ids: Vec<ExprId>,
    result: CoreTelescopingMdResult,
}

#[pymethods]
impl PyTelescopingMdCertificate {
    /// Recurrence order ``J``; ``len(coeffs()) == order + 1``.
    #[getter]
    fn order(&self) -> usize {
        self.order
    }

    /// ``[a_0(n), …, a_J(n)]`` — polynomial coefficients of the recurrence.
    fn coeffs(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.coeff_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// ``[c_1(n,x), …, c_m(n,x)]``, one per bound index in the order they
    /// were supplied to :func:`~alkahest.experimental.telescope_md`.
    fn certs(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.cert_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// Decide the boundary hypothesis for the `m`-fold sum
    /// ``S(n) = Σ_{x_1} … Σ_{x_m} F(n,x)`` over the box
    /// ``x_t = limits[t][0] .. limits[t][1]``.
    ///
    /// ``limits`` must have exactly one ``(lo, hi)`` pair per bound index,
    /// in the same order they were supplied to
    /// :func:`~alkahest.experimental.telescope_md`; each bound is an
    /// **integer constant** (``Expr`` or plain ``int``), never an expression
    /// in ``n`` — see the Rust module docs for
    /// ``telescoping2d::boundary`` for why, and for the standard workaround
    /// when the natural range is `n`-dependent.
    ///
    /// Returns a ``dict`` with keys ``status`` (``"vanishes"``, ``"nonzero"``
    /// or ``"unknown"`` — though this version never produces ``"nonzero"``,
    /// see the class docs), ``implies_sum_recurrence`` and
    /// ``side_conditions``.
    #[pyo3(signature = (limits))]
    fn boundary_status(
        &self,
        py: Python<'_>,
        limits: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
    ) -> PyResult<Py<PyDict>> {
        if limits.len() != self.index_ids.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} (lo, hi) limit pairs, one per bound index, got {}",
                self.index_ids.len(),
                limits.len()
            )));
        }
        let mut coerced: Vec<(ExprId, ExprId)> = Vec::with_capacity(limits.len());
        for (i, (lo, hi)) in limits.iter().enumerate() {
            let lo_id = coerce_limit(py, &self.pool, lo, &format!("limits[{i}][0]"))?;
            let hi_id = coerce_limit(py, &self.pool, hi, &format!("limits[{i}][1]"))?;
            coerced.push((lo_id, hi_id));
        }
        let (status, ranges) = {
            let pool = self.pool.borrow(py);
            let status = core_boundary_status_md(
                &self.result,
                self.term_id,
                self.n_id,
                &self.index_ids,
                &coerced,
                &pool.inner,
            );
            let ranges: Vec<String> = self
                .index_ids
                .iter()
                .zip(coerced.iter())
                .map(|(&idx, &(lo, hi))| {
                    format!(
                        "{} = {}..{}",
                        pool.inner.display(idx),
                        pool.inner.display(lo),
                        pool.inner.display(hi)
                    )
                })
                .collect();
            (status, ranges)
        };
        let out = PyDict::new_bound(py);
        out.set_item("status", status.tag())?;
        out.set_item("implies_sum_recurrence", status.implies_sum_recurrence())?;
        out.set_item("side_conditions", status.side_conditions(&ranges))?;
        Ok(out.unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        let coeffs: Vec<String> = self
            .coeff_ids
            .iter()
            .map(|&id| pool.inner.display(id).to_string())
            .collect();
        let certs: Vec<String> = self
            .cert_ids
            .iter()
            .map(|&id| pool.inner.display(id).to_string())
            .collect();
        format!(
            "TelescopingMdCertificate(order={}, coeffs=[{}], certs=[{}])",
            self.order,
            coeffs.join(", "),
            certs.join(", ")
        )
    }
}

/// `alkahest.experimental.telescope_md(term, n, indices, *, max_order=2, max_a_degree=2, max_cert_degree=2) -> TelescopingMdCertificate`
///
/// Creative telescoping (Apagodu–Zeilberger) for a proper hypergeometric term
/// ``F(n, x_1, …, x_m)`` with an arbitrary number `m ≥ 1` of bound indices —
/// the general form of :func:`telescope2d` (`m = 2`), which remains the
/// semver-stable special case with its own dedicated function. Returns a
/// **verified** certificate: the identity
/// ``Σ_i a_i(n)·F(n+i,x) = Σ_t Δ_t(c_t·F)`` is re-checked exactly in
/// ``Q(n,x_1,…,x_m)`` before it is returned.
///
/// ``indices`` is the list of bound-index symbols ``[x_1, …, x_m]``, in the
/// order the returned certificate's ``certs()`` and any
/// :meth:`~alkahest.experimental.TelescopingMdCertificate.boundary_status`
/// call use. The supported summand class, the certificate ansatz's degree
/// budget, the fixed (non-minimal) certificate denominator and the
/// constant-box-only boundary analysis are exactly :func:`telescope2d`'s,
/// generalized from two indices to `m`; see the Rust module docs
/// (``alkahest_cas::holonomic::telescoping2d``) for the complete, honestly-
/// stated scope, including the resource ceilings that keep a search with no
/// certificate in reach a fast refusal rather than an unbounded computation
/// as `m` or `max_cert_degree` grow — raising `m` or the certificate degree
/// bound grows the search space (and the risk of hitting those ceilings)
/// much faster than in the two-index case.
///
/// Raises :exc:`alkahest.HolonomicError` (``E-HOLO-040`` outside the
/// supported class, ``E-HOLO-041`` when the bounded search is exhausted —
/// including when a resource ceiling, not genuine non-existence, is the
/// reason — ``E-HOLO-042`` for a malformed call, e.g. an empty ``indices``
/// or a repeated symbol) rather than guessing.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    name = "telescope_md",
    signature = (term, n, indices, *, max_order = 2, max_a_degree = 2, max_cert_degree = 2)
)]
fn py_telescope_md(
    py: Python<'_>,
    term: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
    indices: Vec<PyRef<PyExpr>>,
    max_order: usize,
    max_a_degree: usize,
    max_cert_degree: usize,
) -> PyResult<PyTelescopingMdCertificate> {
    let pool_py = term.pool.clone_ref(py);
    let index_ids: Vec<ExprId> = indices.iter().map(|e| e.id).collect();
    let opts = CoreTelescopingMdOpts {
        max_order,
        max_a_degree,
        max_cert_degree,
    };
    let result = {
        let pool = pool_py.borrow(py);
        core_telescope_md_search(term.id, n.id, &index_ids, &pool.inner, &opts)
            .map_err(telescoping2d_error_to_py)?
    };
    Ok(PyTelescopingMdCertificate {
        order: result.order,
        coeff_ids: result.coeffs.clone(),
        cert_ids: result.certs.clone(),
        pool: pool_py,
        term_id: term.id,
        n_id: n.id,
        index_ids,
        result,
    })
}

// ---------------------------------------------------------------------------
// P1 item 10 — asymptotic expansion at scale
// ---------------------------------------------------------------------------

/// An asymptotic expansion together with the evidence for it.
///
/// `terms` is ordered most-significant first, so the claim is
/// ``f ~ terms[0] + terms[1] + …``. The remaining fields exist so a caller can
/// audit *why* it should believe that: which hypotheses were mechanically
/// checked versus merely assumed, and how the truncated expansion compared
/// against an independently computed reference.
#[pyclass(name = "AsymptoticReport")]
struct PyAsymptoticReport {
    method: String,
    term_ids: Vec<ExprId>,
    rigor: String,
    hypotheses: Vec<(String, String)>,
    verification: Vec<(f64, f64, f64, f64)>,
    derivation: Vec<String>,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyAsymptoticReport {
    /// Name of the method that produced the expansion.
    #[getter]
    fn method(&self) -> String {
        self.method.clone()
    }

    /// Ordered terms, most significant first.
    #[getter]
    fn terms(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.term_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// The most significant term.
    #[getter]
    fn leading(&self, py: Python<'_>) -> Option<PyExpr> {
        self.term_ids.first().map(|&id| PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        })
    }

    /// ``"proved"`` or ``"numerically_consistent"`` — how much of the result
    /// the implemented method actually establishes.
    #[getter]
    fn rigor(&self) -> String {
        self.rigor.clone()
    }

    /// ``[(status, statement)]`` with status ``"checked"`` or ``"assumed"``.
    #[getter]
    fn hypotheses(&self) -> Vec<(String, String)> {
        self.hypotheses.clone()
    }

    /// True when every listed hypothesis was mechanically checked.
    #[getter]
    fn all_hypotheses_checked(&self) -> bool {
        self.hypotheses.iter().all(|(s, _)| s == "checked")
    }

    /// ``[(at, reference, approximation, relative_error)]`` from the gate.
    #[getter]
    fn verification(&self) -> Vec<(f64, f64, f64, f64)> {
        self.verification.clone()
    }

    /// Worst relative error observed by the numeric gate.
    #[getter]
    fn max_relative_error(&self) -> Option<f64> {
        self.verification
            .iter()
            .map(|v| v.3)
            .fold(None, |acc, e| Some(acc.map_or(e, |a: f64| f64::max(a, e))))
    }

    /// Ordered, human-readable derivation log.
    #[getter]
    fn derivation(&self) -> Vec<String> {
        self.derivation.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "AsymptoticReport(method={:?}, terms={}, rigor={:?})",
            self.method,
            self.term_ids.len(),
            self.rigor
        )
    }
}

/// `alkahest.euler_maclaurin(summand, k, a, n, *, corrections=2) -> AsymptoticReport`
///
/// Asymptotic expansion of ``Σ_{k=a}^{n} f(k)`` as ``n → ∞`` by the
/// Euler–Maclaurin formula. For ``f(k) = 1/k`` this recovers
/// ``H_n ~ log n + γ + 1/(2n) − 1/(12n²) + …``.
///
/// The additive constant (γ above) is **not** determined by Euler–Maclaurin
/// from the ``n``-side terms; it is fitted numerically from the exact sum and
/// the returned report says so — ``rigor`` is ``"numerically_consistent"`` and
/// the fitted constant appears as an explicitly assumed hypothesis.
///
/// Raises :exc:`alkahest.AsymptoticError` rather than guessing when the
/// summand has no symbolic antiderivative or no term survives the numeric gate.
#[pyfunction]
#[pyo3(name = "euler_maclaurin", signature = (summand, k, a, n, *, corrections = 2))]
fn py_euler_maclaurin(
    py: Python<'_>,
    summand: PyRef<PyExpr>,
    k: PyRef<PyExpr>,
    a: i64,
    n: PyRef<PyExpr>,
    corrections: usize,
) -> PyResult<PyAsymptoticReport> {
    let pool_py = summand.pool.clone_ref(py);
    let r = {
        let pool = pool_py.borrow(py);
        core_euler_maclaurin(summand.id, k.id, a, n.id, corrections, &pool.inner)
            .map_err(asymptotic_error_to_py)?
    };
    Ok(PyAsymptoticReport {
        method: r.method.to_string(),
        term_ids: r.terms.clone(),
        rigor: r.rigor.tag().to_string(),
        hypotheses: r
            .hypotheses
            .iter()
            .map(|h| (h.status.tag().to_string(), h.statement.clone()))
            .collect(),
        verification: r
            .verification
            .iter()
            .map(|v| (v.at, v.reference, v.approximation, v.relative_error))
            .collect(),
        derivation: r.derivation.clone(),
        pool: pool_py,
    })
}

/// `alkahest.experimental.coefficient_asymptotics(gf, z, n) -> AsymptoticReport`
///
/// Growth of ``[zⁿ] f(z)`` for a rational generating function, by singularity
/// analysis: the coefficient asymptotics are governed by the pole of smallest
/// modulus. ``1/(1 - z - z²)`` gives the Fibonacci growth ``φⁿ/√5``.
///
/// Declines (:exc:`alkahest.AsymptoticError`) rather than guessing when the
/// dominant singularity is not unique — several poles of equal modulus make the
/// coefficients oscillate and no single power-law term describes them — when it
/// is complex, or when the input is not rational.
#[pyfunction]
#[pyo3(name = "coefficient_asymptotics", signature = (gf, z, n))]
fn py_coefficient_asymptotics(
    py: Python<'_>,
    gf: PyRef<PyExpr>,
    z: PyRef<PyExpr>,
    n: PyRef<PyExpr>,
) -> PyResult<PyAsymptoticReport> {
    let pool_py = gf.pool.clone_ref(py);
    let r = {
        let pool = pool_py.borrow(py);
        core_coefficient_asymptotics(gf.id, z.id, n.id, &pool.inner)
            .map_err(asymptotic_error_to_py)?
    };
    Ok(PyAsymptoticReport {
        method: r.method.to_string(),
        term_ids: r.terms.clone(),
        rigor: r.rigor.tag().to_string(),
        hypotheses: r
            .hypotheses
            .iter()
            .map(|h| (h.status.tag().to_string(), h.statement.clone()))
            .collect(),
        verification: r
            .verification
            .iter()
            .map(|v| (v.at, v.reference, v.approximation, v.relative_error))
            .collect(),
        derivation: r.derivation.clone(),
        pool: pool_py,
    })
}

// ---------------------------------------------------------------------------
// M5 — recurrence -> asymptotics (Poincaré–Perron)
// ---------------------------------------------------------------------------

/// Asymptotics of a P-recursive sequence, read off its recurrence.
///
/// Returned by :func:`alkahest.experimental.asymptotics_from_recurrence`.  The
/// object is arranged around one distinction, because it is the distinction a
/// research loop gets wrong:
///
/// * **Derived** — :attr:`growth_rate`, :attr:`polynomial_exponent`,
///   :meth:`roots`, :attr:`verdict`.  These are functions of the coefficient
///   polynomials and of nothing else: Poincaré–Perron applied to the
///   characteristic polynomial.  When the root is rational they are available
///   *exactly* as :attr:`growth_rate_exact` and
///   :attr:`polynomial_exponent_exact`.
/// * **Fitted** — :attr:`connection_constant`, and only that.  `C` in
///   ``u(n) ~ C·ρⁿ·n^α`` is determined by the initial conditions, not by the
///   recurrence, so it is extrapolated numerically from the exact terms.
///   :attr:`connection_constant_converged` says whether the extrapolation
///   agreed with a second one from a smaller range, and
///   :attr:`connection_constant_drift` says by how much.
///
/// :meth:`evidence` returns both halves as a dict for logging next to a result,
/// and :meth:`report` returns the family's usual
/// :class:`~alkahest.experimental.AsymptoticReport` with the hypotheses, the
/// numeric corroboration and the derivation log.
#[pyclass(name = "RecurrenceAsymptotics")]
struct PyRecurrenceAsymptotics {
    inner: CoreRecurrenceAsymptotics,
    growth_rate_exact_id: Option<ExprId>,
    polynomial_exponent_exact_id: Option<ExprId>,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyRecurrenceAsymptotics {
    /// Recurrence order ``J``.
    #[getter]
    fn order(&self) -> usize {
        self.inner.characteristic.order
    }

    /// ``D`` — the largest degree among the coefficient polynomials.
    #[getter]
    fn coefficient_degree(&self) -> usize {
        self.inner.characteristic.coefficient_degree
    }

    /// ``"single_dominant_root"``, ``"equal_modulus_roots"``,
    /// ``"repeated_dominant_root"``, ``"degenerate_leading_coefficient"`` or
    /// ``"eventually_zero"``.
    ///
    /// Only the first gives a growth law.  The others are the hypotheses of
    /// Poincaré–Perron failing, reported rather than assumed away: equal-modulus
    /// roots make the solutions oscillate, and answering with one of them would
    /// be a wrong answer with a confident face on it.
    #[getter]
    fn verdict(&self) -> &'static str {
        self.inner.characteristic.verdict.tag()
    }

    /// One sentence saying what :attr:`verdict` means for the caller.
    #[getter]
    fn verdict_reason(&self) -> String {
        self.inner.characteristic.verdict.explanation()
    }

    /// ``ρ`` — the dominant characteristic root, or ``None``.
    ///
    /// **Derived.**  ``None`` exactly when :attr:`verdict` is not
    /// ``"single_dominant_root"``.
    #[getter]
    fn growth_rate(&self) -> Option<f64> {
        self.inner.characteristic.growth_rate
    }

    /// ``ρ`` as an exact rational :class:`~alkahest.Expr`, when it is one.
    ///
    /// ``None`` means "not a rational number, or not established" — Apéry's
    /// ``17 + 12√2`` is real and simple and has no entry here.
    #[getter]
    fn growth_rate_exact(&self, py: Python<'_>) -> Option<PyExpr> {
        self.growth_rate_exact_id.map(|id| PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        })
    }

    /// ``α`` in ``u(n) ~ C·ρⁿ·n^α``, or ``None``.
    ///
    /// **Derived**, from ``α = −χ₁(ρ)/(ρ·χ'(ρ))``.  ``-0.5`` for the central
    /// binomial coefficients, ``-1.5`` for Catalan, Motzkin and Apéry.
    #[getter]
    fn polynomial_exponent(&self) -> Option<f64> {
        self.inner.characteristic.polynomial_exponent
    }

    /// ``α`` as an exact rational :class:`~alkahest.Expr`, when ``ρ`` is
    /// rational (then so is ``α``).
    #[getter]
    fn polynomial_exponent_exact(&self, py: Python<'_>) -> Option<PyExpr> {
        self.polynomial_exponent_exact_id.map(|id| PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        })
    }

    /// ``C`` — **fitted**, never derived.  ``None`` when it was not fitted.
    ///
    /// Read :attr:`connection_constant_converged` before quoting it.  A value
    /// with ``converged`` ``False`` is evidence about the fit, not a result.
    #[getter]
    fn connection_constant(&self) -> Option<f64> {
        self.inner.connection.map(|c| c.value)
    }

    /// Whether the connection constant agreed with a second extrapolation from
    /// a smaller range of indices.
    #[getter]
    fn connection_constant_converged(&self) -> bool {
        self.inner.connection.is_some_and(|c| c.converged)
    }

    /// How far the two extrapolations of ``C`` differ, relative to their size.
    #[getter]
    fn connection_constant_drift(&self) -> Option<f64> {
        self.inner.connection.map(|c| c.relative_drift)
    }

    /// Largest index the connection constant was fitted at.
    #[getter]
    fn connection_constant_fitted_at(&self) -> Option<i64> {
        self.inner.connection.map(|c| c.fitted_at)
    }

    /// Whether the supplied terms were seen to follow the *dominant* root.
    ///
    /// ``None`` when no terms were supplied.  Poincaré's conclusion is that
    /// ``u(n+1)/u(n)`` tends to *some* characteristic root; ``False`` is the
    /// real answer that the sequence's dominant component vanishes, as it does
    /// for the constant solution of ``u(n+2) = 3u(n+1) − 2u(n)``.
    #[getter]
    fn follows_dominant_root(&self) -> Option<bool> {
        self.inner.follows_dominant_root
    }

    /// ``C·ρⁿ·n^α`` as an :class:`~alkahest.Expr`, or ``None``.
    ///
    /// Present only when the verdict gave a single law, the terms followed the
    /// dominant root, the constant converged and the result passed the numeric
    /// gate.  The constant inside it is fitted — see
    /// :attr:`connection_constant`.
    #[getter]
    fn leading_term(&self, py: Python<'_>) -> Option<PyExpr> {
        self.inner.leading_term.map(|id| PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        })
    }

    /// Whether the enumeration of integer zeros of the leading coefficient was
    /// exhaustive.
    #[getter]
    fn singular_indices_complete(&self) -> bool {
        self.inner.characteristic.singular_indices_complete
    }

    /// Worst relative error observed by the numeric gate.
    #[getter]
    fn max_relative_error(&self) -> Option<f64> {
        self.inner.max_relative_error()
    }

    /// Every root of the characteristic polynomial, modulus-descending.
    ///
    /// ``[(re, im, modulus, multiplicity)]``.  The multiplicity is **exact** —
    /// it comes from the squarefree decomposition of ``χ`` over ``ℚ``, not from
    /// clustering the numeric roots.
    fn roots(&self) -> Vec<(f64, f64, f64, usize)> {
        self.inner
            .characteristic
            .roots
            .iter()
            .map(|r| (r.re, r.im, r.modulus, r.multiplicity))
            .collect()
    }

    /// Integer ``n ≥ start`` at which the leading coefficient vanishes.
    ///
    /// Poincaré–Perron needs it non-zero for large ``n``; since it is a
    /// polynomial there are finitely many exceptions and the theorem applies
    /// beyond the largest.  See :attr:`singular_indices_complete`.
    fn singular_indices(&self) -> Vec<i64> {
        self.inner.characteristic.singular_indices.clone()
    }

    /// The derived and the fitted halves, as a dict, for logging next to a
    /// result.
    ///
    /// Sibling of :meth:`alkahest.GuessedRecurrence.evidence`.  ``derived``
    /// holds what follows from the recurrence, ``fitted`` holds the connection
    /// constant and how well it converged; a loop that records this cannot
    /// later mistake one for the other.
    fn evidence(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let out = PyDict::new_bound(py);

        let derived = PyDict::new_bound(py);
        derived.set_item("order", self.inner.characteristic.order)?;
        derived.set_item("verdict", self.inner.characteristic.verdict.tag())?;
        derived.set_item("growth_rate", self.inner.characteristic.growth_rate)?;
        derived.set_item(
            "polynomial_exponent",
            self.inner.characteristic.polynomial_exponent,
        )?;
        derived.set_item("roots", self.roots())?;
        derived.set_item("singular_indices", self.singular_indices())?;
        out.set_item("derived", derived)?;

        let fitted = PyDict::new_bound(py);
        fitted.set_item("connection_constant", self.connection_constant())?;
        fitted.set_item("converged", self.connection_constant_converged())?;
        fitted.set_item("relative_drift", self.connection_constant_drift())?;
        fitted.set_item("fitted_at", self.connection_constant_fitted_at())?;
        fitted.set_item(
            "refit_at",
            self.inner
                .connection
                .map(|c: CoreConnectionConstant| c.refit_at),
        )?;
        out.set_item("fitted", fitted)?;

        out.set_item("follows_dominant_root", self.inner.follows_dominant_root)?;
        out.set_item("max_relative_error", self.max_relative_error())?;
        Ok(out.unbind())
    }

    /// The result as the asymptotics family's usual
    /// :class:`~alkahest.experimental.AsymptoticReport`.
    ///
    /// This is where the hypotheses, the numeric corroboration and the
    /// derivation log live, so there is exactly one place to read them from.
    /// ``terms`` is empty when :attr:`leading_term` is ``None``; ``rigor`` is
    /// always ``"numerically_consistent"``, because the modulus separation is
    /// decided numerically and the constant is fitted.
    fn report(&self, py: Python<'_>) -> PyAsymptoticReport {
        let r = self.inner.report();
        PyAsymptoticReport {
            method: r.method.to_string(),
            term_ids: r.terms.clone(),
            rigor: r.rigor.tag().to_string(),
            hypotheses: r
                .hypotheses
                .iter()
                .map(|h| (h.status.tag().to_string(), h.statement.clone()))
                .collect(),
            verification: r
                .verification
                .iter()
                .map(|v| (v.at, v.reference, v.approximation, v.relative_error))
                .collect(),
            derivation: r.derivation.clone(),
            pool: self.pool.clone_ref(py),
        }
    }

    fn __repr__(&self) -> String {
        let verdict = self.inner.characteristic.verdict.tag();
        match self.inner.characteristic.growth_rate {
            Some(rho) => format!(
                "RecurrenceAsymptotics(verdict={verdict:?}, growth_rate={rho}, \
                 polynomial_exponent={}, connection_constant={} [fitted])",
                self.inner
                    .characteristic
                    .polynomial_exponent
                    .map_or_else(|| "None".to_string(), |a| a.to_string()),
                self.connection_constant()
                    .map_or_else(|| "None".to_string(), |c| c.to_string()),
            ),
            None => format!("RecurrenceAsymptotics(verdict={verdict:?}, no growth law)"),
        }
    }
}

/// One recurrence coefficient: an `Expr` in `n`, or ascending integer
/// coefficients of a polynomial in `n`.
///
/// The second form exists for `GuessedRecurrence.coeffs`, whose entries are
/// arbitrary-size Python ints. Routing them through `big_integer_from_py` keeps
/// them exact; building the same polynomial with Python arithmetic
/// (`c * n**j`) silently turns anything past 2⁵³ into a float, which for a
/// recurrence fitted to `(2n)!`-scale terms is most of them.
fn coerce_recurrence_coefficient(
    py: Python<'_>,
    pool_py: &Py<PyExprPool>,
    n: ExprId,
    v: &Bound<'_, PyAny>,
    which: usize,
) -> PyResult<ExprId> {
    if let Ok(e) = v.extract::<PyRef<PyExpr>>() {
        if !e.pool.is(pool_py) {
            return Err(pool_mismatch_err());
        }
        return Ok(e.id);
    }
    let bad_shape = || {
        PyTypeError::new_err(format!(
            "coeffs[{which}] must be an alkahest Expr or a sequence of integers \
             (ascending coefficients of a polynomial in n), got {}",
            v.get_type()
        ))
    };
    // A `str` iterates as characters, so `extract::<Vec<_>>()` would accept one
    // and then fail deep inside the integer parse with a message about the
    // wrong thing.
    if v.is_instance_of::<pyo3::types::PyString>() || v.is_instance_of::<pyo3::types::PyBytes>() {
        return Err(bad_shape());
    }
    let items: Vec<Bound<'_, PyAny>> = v.extract().map_err(|_| bad_shape())?;
    let pool = pool_py.borrow(py);
    let mut terms: Vec<ExprId> = Vec::new();
    for (j, item) in items.iter().enumerate() {
        if item.is_instance_of::<pyo3::types::PyFloat>() {
            return Err(PyTypeError::new_err(format!(
                "coeffs[{which}][{j}] is a float; a recurrence coefficient must be an exact \
                 integer — the characteristic polynomial is computed in exact arithmetic and \
                 a rounded coefficient describes a different recurrence"
            )));
        }
        let c = big_integer_from_py(item)?;
        if c == 0 {
            continue;
        }
        let lit = pool.inner.integer(c);
        terms.push(if j == 0 {
            lit
        } else {
            let power = pool.inner.pow(n, pool.inner.integer(j as i64));
            pool.inner.mul(vec![lit, power])
        });
    }
    Ok(if terms.is_empty() {
        pool.inner.integer(0_i32)
    } else {
        core_simplify(pool.inner.add(terms), &pool.inner).value
    })
}

/// One sequence term: a Python int, or a `(numerator, denominator)` pair.
///
/// A `float` is refused rather than converted, for the reason
/// `guess_holonomic` refuses one: `0.1` is not one tenth, and the arithmetic
/// downstream of here is exact and would happily fit a growth law to a
/// different sequence.
fn coerce_sequence_term(v: &Bound<'_, PyAny>, which: usize) -> PyResult<rug::Rational> {
    if v.is_instance_of::<pyo3::types::PyFloat>() {
        return Err(PyTypeError::new_err(format!(
            "terms[{which}] is a float; sequence terms must be exact (an int, or a \
             (numerator, denominator) pair) — a growth law fitted to rounded terms is a \
             growth law for a different sequence"
        )));
    }
    if let Ok((num, den)) = v.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>() {
        let (num, den) = (big_integer_from_py(&num)?, big_integer_from_py(&den)?);
        if den == 0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(format!(
                "terms[{which}] has a zero denominator"
            )));
        }
        return Ok(rug::Rational::from((num, den)));
    }
    Ok(rug::Rational::from(big_integer_from_py(v)?))
}

/// `alkahest.experimental.asymptotics_from_recurrence(coeffs, n, *, terms=None, start=0)`
///
/// Growth of the sequence satisfying ``Σ_i coeffs[i](n)·u(n+i) = 0``, by
/// Poincaré–Perron.  ``coeffs`` are the coefficient polynomials ``p_0 … p_J``,
/// each an ``Expr`` in ``n`` or a sequence of ascending integer coefficients;
/// ``terms`` are the exact leading terms with ``terms[0] = u(start)``.
///
/// The growth rate ``ρ`` and the polynomial exponent ``α`` are **derived** from
/// the recurrence.  The connection constant ``C`` in ``u(n) ~ C·ρⁿ·n^α`` is
/// **not** — it depends on the initial conditions — so it is extrapolated
/// numerically from the terms and reported separately, the way
/// :func:`~alkahest.experimental.euler_maclaurin` reports its additive
/// constant.  Pass no terms and you still get ``ρ``, ``α`` and the roots.
///
/// A degenerate case is *reported*, not refused and not papered over:
/// equal-modulus roots, a repeated dominant root, a leading coefficient whose
/// top-degree part vanishes, or an eventually-zero sequence each set
/// :attr:`~alkahest.experimental.RecurrenceAsymptotics.verdict` and leave
/// ``growth_rate`` as ``None``.
///
/// Raises :exc:`alkahest.AsymptoticError` only for malformed input: fewer than
/// two coefficients, a coefficient that is not a polynomial in ``n`` over
/// ``ℚ``, or a characteristic polynomial all of whose roots are zero.
#[pyfunction]
#[pyo3(
    name = "asymptotics_from_recurrence",
    signature = (coeffs, n, *, terms = None, start = 0)
)]
fn py_asymptotics_from_recurrence(
    py: Python<'_>,
    coeffs: Vec<Bound<'_, PyAny>>,
    n: PyRef<PyExpr>,
    terms: Option<Vec<Bound<'_, PyAny>>>,
    start: i64,
) -> PyResult<PyRecurrenceAsymptotics> {
    let pool_py = n.pool.clone_ref(py);
    let mut coeff_ids = Vec::with_capacity(coeffs.len());
    for (i, c) in coeffs.iter().enumerate() {
        coeff_ids.push(coerce_recurrence_coefficient(py, &pool_py, n.id, c, i)?);
    }
    let mut exact_terms = Vec::new();
    for (i, t) in terms.unwrap_or_default().iter().enumerate() {
        exact_terms.push(coerce_sequence_term(t, i)?);
    }

    let (inner, growth_rate_exact_id, polynomial_exponent_exact_id) = {
        let pool = pool_py.borrow(py);
        let inner =
            core_asymptotics_from_recurrence(&coeff_ids, n.id, &exact_terms, start, &pool.inner)
                .map_err(asymptotic_error_to_py)?;
        let rho = inner
            .characteristic
            .growth_rate_exact
            .as_ref()
            .map(|q| pool.inner.rational(q.numer().clone(), q.denom().clone()));
        let alpha = inner
            .characteristic
            .polynomial_exponent_exact
            .as_ref()
            .map(|q| pool.inner.rational(q.numer().clone(), q.denom().clone()));
        (inner, rho, alpha)
    };
    Ok(PyRecurrenceAsymptotics {
        inner,
        growth_rate_exact_id,
        polynomial_exponent_exact_id,
        pool: pool_py,
    })
}

/// `alkahest.match_pattern(pattern_expr, expr) -> list[dict[str, Expr]]`
///
/// Find all AC-aware matches of `pattern_expr` anywhere in `expr`.
/// Each match is returned as a dict mapping wildcard names to matched
/// sub-expressions.  Wildcards are symbols whose names start with a
/// lower-case letter.  Set `wildcards=False` to treat every symbol as a literal
/// (e.g. when the pattern is a single variable like `x`).
#[pyfunction]
#[pyo3(signature = (pattern_expr, expr, *, wildcards = true))]
fn match_pattern(
    py: Python<'_>,
    pattern_expr: PyRef<PyExpr>,
    expr: PyRef<PyExpr>,
    wildcards: bool,
) -> PyResult<PyObject> {
    let pool_py = pattern_expr.pool.clone_ref(py);
    let matches = {
        let pool = pool_py.borrow(py);
        // The matcher recurses over both sides, so both need the guard.
        alkahest_core::check_expr_depths(&pool.inner, &[pattern_expr.id, expr.id])
            .map_err(depth_error_to_py)?;
        let pat = Pattern::from_expr(pattern_expr.id);
        core_match_pattern_with_config(&pat, expr.id, &pool.inner, MatchConfig { wildcards })
    };
    let out = PyList::empty_bound(py);
    for subst in matches {
        let d = PyDict::new_bound(py);
        for (name, id) in subst.bindings {
            let expr_py = PyExpr {
                id,
                pool: pool_py.clone_ref(py),
            };
            d.set_item(name, expr_py.into_py(py))?;
        }
        out.append(d)?;
    }
    Ok(out.into_py(py))
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
    make_derived_result(py, derived, pool_py, None)
}

/// `alkahest.simplify_expanded(expr)` — simplify with distributive expansion.
///
/// Applies `(a + b) * c → a*c + b*c` in addition to all default rules.
#[pyfunction]
#[pyo3(name = "simplify_expanded")]
fn py_simplify_expanded(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        alkahest_core::simplify_expanded(expr.id, &pool.inner)
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, None))
}

/// `alkahest.simplify_trig(expr)` — simplify with trigonometric identities.
#[pyfunction]
#[pyo3(name = "simplify_trig")]
fn py_simplify_trig(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let rules = trig_rules();
        core_simplify_with(expr.id, &pool.inner, &rules, SimplifyConfig::default())
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, None))
}

/// `alkahest.simplify_trig_normal_form(expr)` — reduce to a trig normal form.
///
/// Runs the full algebraic core *with bounded polynomial expansion* plus the
/// sin/cos-polynomial trig identities — argument-sign normalization and the
/// Pythagorean identity, including its multi-angle case — driven to a fixed
/// point. Unlike :func:`simplify_trig` (trig identities only, no expansion),
/// this composes product expansion, constant folding, like-term collection,
/// and Pythagorean reduction into a single call.
///
/// The headline use case is verifying orthogonality of a rotation
/// (direction-cosine) matrix: every entry of ``R.T @ R - I`` collapses to ``0``.
/// It reduces in the sin/cos monomial basis and does not introduce
/// compound-angle (``sin(2u)``, ``sin(u+v)``, …) forms. This bundle is heavier
/// than :func:`simplify` and is opt-in.
#[pyfunction]
#[pyo3(name = "simplify_trig_normal_form")]
fn py_simplify_trig_normal_form(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_simplify_trig_normal_form(expr.id, &pool.inner)
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, None))
}

/// `alkahest.simplify_log_exp(expr, assumptions=None)` — simplify with log/exp identities.
///
/// Unconditional rules: `log(exp(x))→x`, `exp(x)·exp(y)→exp(x+y)`.
/// Branch-cut identities (`exp(log(x))→x`, sum/power/quotient of logs) require
/// positivity facts from ``assumptions`` or ``Domain.Positive`` symbols.
#[pyfunction]
#[pyo3(name = "simplify_log_exp", signature = (expr, assumptions=None))]
fn py_simplify_log_exp(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    assumptions: Option<PyRef<PyAssumptions>>,
) -> PyResult<PyDerivedResult> {
    if let Some(ref a) = assumptions {
        if !a.pool.is(&expr.pool) {
            return Err(pool_mismatch_err());
        }
    }
    let derived = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let facts = match &assumptions {
            Some(a) => a.inner.facts().to_vec(),
            None => Vec::new(),
        };
        core_simplify_log_exp(expr.id, &pool.inner, &facts)
    };
    let pool_py = expr.pool.clone_ref(py);
    Ok(make_derived_result(py, derived, pool_py, None))
}

// ---------------------------------------------------------------------------
// V5-1 — Lean 4 certificate exporter
// ---------------------------------------------------------------------------

/// `alkahest.to_lean(expr_or_result) -> str`
///
/// Generate a Lean 4 proof certificate from a derivation.
///
/// Accepts either:
///
/// - An :class:`Expr` — the default simplifier runs first, then the
///   derivation log is certified.  Equivalent to
///   ``to_lean(simplify(expr))``.
/// - A :class:`DerivedResult` returned by any simplifier (e.g.
///   ``simplify_trig``, ``simplify_log_exp``, ``simplify_with``) — the
///   log already recorded in that result is certified directly, so the
///   choice of simplifier is fully under caller control.
///
/// Returns a string containing the complete ``.lean`` source file
/// (Mathlib imports + one ``example`` per rewrite step).
///
/// Example::
///
///     result = alkahest.simplify_trig(sin(x)**2 + cos(x)**2)
///     print(alkahest.to_lean(result))  # certifies via trig identities
#[pyfunction]
#[pyo3(name = "to_lean")]
fn py_to_lean(py: Python<'_>, arg: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(derived_bound) = arg.downcast::<PyDerivedResult>() {
        let d = derived_bound.borrow();
        // `expr_to_lean` recurses once per level with no cap.
        guard_expr_depth(py, &d.value)?;
        // Integration results certify via the FTC derivative relation
        // `deriv (fun x => F) x = f` rather than a false `f = F` equality.
        if let Some((integrand, var)) = d.integration_verification_input {
            let pool_py = d.value.pool.clone_ref(py);
            let pool = pool_py.borrow(py);
            let src =
                alkahest_core::emit_integration_cert(d.raw.value, integrand, var, &pool.inner);
            if src.contains("sorry") || src.contains("admit") {
                return Ok(String::new());
            }
            return Ok(src);
        }
        // Definite integrals certify via the interval FTC equation.
        if let Some((integrand, var, lower, upper)) = d.definite_integration_input {
            let pool_py = d.value.pool.clone_ref(py);
            let pool = pool_py.borrow(py);
            let src = alkahest_core::emit_definite_integration_cert(
                integrand,
                var,
                lower,
                upper,
                &pool.inner,
            );
            if src.contains("sorry") || src.contains("admit") {
                return Ok(String::new());
            }
            return Ok(src);
        }
        let pool_py = d.value.pool.clone_ref(py);
        let pool = pool_py.borrow(py);
        let src = alkahest_core::emit_lean_expr_wrt(&d.raw, &pool.inner, d.wrt);
        if src.contains("sorry") || src.contains("admit") {
            return Ok(String::new());
        }
        return Ok(src);
    }
    if let Ok(expr_bound) = arg.downcast::<PyExpr>() {
        let expr = expr_bound.borrow();
        let pool_py = expr.pool.clone_ref(py);
        let derived = {
            let pool = pool_py.borrow(py);
            guard_depth(&pool.inner, expr.id)?;
            core_simplify(expr.id, &pool.inner)
        };
        // Part C: the default simplifier may leave the expression untouched
        // (e.g. `exp(log(x))`, whose rewrite lives in `simplify_log_exp`, not
        // the default set). Emitting `example : e = e := rfl` in that case reads
        // as a proven theorem about `e` when nothing was actually derived. When
        // no non-trivial rewrite occurred, withhold instead of presenting a
        // vacuous reflexive identity as a certificate.
        if derived.log.is_empty() {
            return Ok(String::new());
        }
        let pool = pool_py.borrow(py);
        return Ok(alkahest_core::emit_lean(&derived, &pool.inner));
    }
    Err(PyTypeError::new_err(
        "to_lean() expects an Expr or DerivedResult",
    ))
}

// ---------------------------------------------------------------------------
// P2-3 — SMT-LIB 2 exporter (the export half of the SMT/SAT bridge)
// ---------------------------------------------------------------------------

fn smtlib_error_to_py(e: alkahest_core::logic::smtlib::SmtLibError) -> PyErr {
    Python::with_gil(|py| {
        // Deliberately *not* a dedicated PyO3 exception class: `SmtError` is a
        // pure-Python class in `alkahest/exceptions.py` and is not in
        // `_NATIVE_EXCEPTION_OVERLAY`, so a native `PySmtError` would be a
        // *different* class from the `ak.SmtError` callers write `except` for.
        // `alkahest/smt.py` re-raises this as `SmtError`, preserving `.code`.
        let exc_type = py.get_type_bound::<PyAlkahestError>();
        make_structured_err(py, &exc_type, &e)
    })
}

/// `alkahest.to_smtlib(formula, logic="auto", *, check_sat=True, get_model=True) -> str`
///
/// Export a predicate (or quantified) :class:`Expr` as a complete, runnable
/// SMT-LIB 2 script — the SMT counterpart of :func:`alkahest.to_lean`.
///
/// ``logic`` is ``"auto"`` (infer the weakest logic that fits) or one of
/// ``QF_LIA``, ``QF_NIA``, ``QF_LRA``, ``QF_NRA``, ``QF_LIRA``, ``QF_NIRA``,
/// their quantified counterparts, or ``ALL``.  A named logic that is too weak
/// for the formula is an error rather than a downgrade.
///
/// Raises with the stable code ``E-SMT-002`` when the formula is outside the
/// exportable fragment; :func:`alkahest.smt.supported` is the plan-ahead
/// predicate that answers the same question without raising.
///
/// Example::
///
///     f = alkahest.And(pool.gt(x * x, pool.integer(2)), pool.lt(x, pool.integer(0)))
///     print(alkahest.to_smtlib(f))
#[pyfunction]
#[pyo3(
    name = "to_smtlib",
    signature = (formula, logic = "auto", *, check_sat = true, get_model = true)
)]
fn py_to_smtlib(
    py: Python<'_>,
    formula: PyRef<PyExpr>,
    logic: &str,
    check_sat: bool,
    get_model: bool,
) -> PyResult<String> {
    let pool = formula.pool.borrow(py);
    guard_depth(&pool.inner, formula.id)?;
    let opts = alkahest_core::logic::smtlib::SmtLibOptions {
        logic: if logic == "auto" { None } else { Some(logic) },
        check_sat,
        get_model,
    };
    alkahest_core::logic::smtlib::to_smtlib(formula.id, &pool.inner, &opts)
        .map_err(smtlib_error_to_py)
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
        let key_id = coerce_substituent(&pool_py, &k, py)?;
        let val_id = coerce_substituent(&pool_py, &v, py)?;
        map.insert(key_id, val_id);
    }
    let result_id = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let substituted = core_subs(expr.id, &map, &pool.inner);
        core_fold_predicates(substituted, &pool.inner)
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
fn py_grad(py: Python<'_>, expr: PyRef<PyExpr>, vars: Vec<PyRef<PyExpr>>) -> PyResult<Vec<PyExpr>> {
    let pool_py = expr.pool.clone_ref(py);
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let grads = {
        let pool = pool_py.borrow(py);
        // Reverse-mode is the shallowest walker in the library: its post-order
        // DFS overflowed an 8 MiB stack at depth 4 687 (see kernel::depth).
        guard_depth(&pool.inner, expr.id)?;
        core_grad(expr.id, &var_ids, &pool.inner)
    };
    Ok(grads
        .into_iter()
        .map(|id| PyExpr {
            id,
            pool: pool_py.clone_ref(py),
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Phase 15: Matrix and jacobian
// ---------------------------------------------------------------------------

#[pyclass(name = "Matrix", subclass)]
struct PyMatrix {
    inner: Matrix,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyMatrix {
    // Allow Matrix([[expr, expr], [expr, expr]]) in addition to from_rows.
    #[new]
    fn __new__(py: Python<'_>, rows: Vec<Vec<Bound<'_, PyAny>>>) -> PyResult<PyMatrix> {
        PyMatrix::from_rows(py, rows)
    }

    /// Build a matrix from a 2D list of entries.
    ///
    /// Each entry may be an :class:`Expr`, :class:`DerivedResult`, Python
    /// ``int``, or Python ``float``. Bare numeric literals are coerced into
    /// the same :class:`ExprPool` as the first :class:`Expr`/``DerivedResult``
    /// found anywhere in `rows` (consistent with how arithmetic operators
    /// coerce scalars). If `rows` contains no `Expr`/`DerivedResult` at all
    /// (e.g. an all-integer matrix), the pool cannot be inferred and a
    /// `TypeError` is raised — pass at least one `Expr`/`DerivedResult`
    /// entry (or use `ExprPool.integer`/`pool.matrix_of(...)`-style helpers)
    /// so the pool can be determined.
    #[staticmethod]
    fn from_rows(py: Python<'_>, rows: Vec<Vec<Bound<'_, PyAny>>>) -> PyResult<PyMatrix> {
        if rows.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Matrix must have at least one row",
            ));
        }
        // Find the pool from the first Expr/DerivedResult entry anywhere in `rows`.
        let mut pool_py: Option<Py<PyExprPool>> = None;
        for row in &rows {
            for entry in row {
                if let Ok(e) = entry.extract::<PyRef<PyExpr>>() {
                    pool_py = Some(e.pool.clone_ref(py));
                    break;
                }
                if let Ok(dr) = entry.downcast::<PyDerivedResult>() {
                    pool_py = Some(dr.borrow().value.pool.clone_ref(py));
                    break;
                }
            }
            if pool_py.is_some() {
                break;
            }
        }
        let pool_py = pool_py.ok_or_else(|| {
            PyTypeError::new_err(
                "Matrix.from_rows could not determine an ExprPool: at least one entry must \
                 be an Expr or DerivedResult (bare int/float entries are coerced into that \
                 pool, but the pool cannot be inferred from numbers alone)",
            )
        })?;
        let data: Vec<Vec<ExprId>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| coerce_substituent(&pool_py, entry, py))
                    .collect::<PyResult<Vec<ExprId>>>()
            })
            .collect::<PyResult<Vec<Vec<ExprId>>>>()?;
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

    fn get(&self, py: Python<'_>, r: usize, c: usize) -> PyResult<PyExpr> {
        // `Matrix::get` indexes `data[r * cols + c]` with no bounds check, so
        // an out-of-range subscript was a panic — a `BaseException` on the
        // Python side — rather than the `IndexError` a caller expects.  Note
        // `r * cols` also wraps in release, which would have silently returned
        // a different element for huge `r`.
        let (rows, cols) = (self.inner.rows, self.inner.cols);
        if r >= rows || c >= cols {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "matrix index ({r}, {c}) out of range for a {rows}x{cols} matrix"
            )));
        }
        Ok(PyExpr {
            id: self.inner.get(r, c),
            pool: self.pool.clone_ref(py),
        })
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
        self.matmul_impl(py, &other)
    }

    /// `other @ self` (only reached when `other` is a `Matrix` that did not
    /// itself handle `@`). Provided for symmetry with `__matmul__`.
    fn __rmatmul__(&self, py: Python<'_>, other: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
        other.matmul_impl(py, self)
    }

    /// `self * other`.
    ///
    /// Follows the SymPy / CAS convention: if `other` is a :class:`Matrix`,
    /// this is the **matrix product** (identical to ``self @ other``, with the
    /// same inner-dimension check). If `other` is a scalar — an :class:`Expr`,
    /// :class:`DerivedResult`, Python ``int``, or ``float`` — every entry is
    /// scaled by it. There is no elementwise (Hadamard) product on ``*``; use
    /// :meth:`hadamard` for that.
    fn __mul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        if let Ok(rhs) = other.downcast::<PyMatrix>() {
            return Ok(self.matmul_impl(py, &rhs.borrow())?.into_py(py));
        }
        match self.coerce_matrix_scalar(other, py)? {
            Some(scalar) => Ok(self.scale_impl(py, scalar).into_py(py)),
            None => Ok(py.NotImplemented()),
        }
    }

    /// `other * self` — scalar multiplication with the scalar on the left
    /// (e.g. ``2 * A`` or ``expr * A``). Matrix-on-the-left products are
    /// handled by the left operand's ``*`` / ``@``.
    fn __rmul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        match self.coerce_matrix_scalar(other, py)? {
            Some(scalar) => Ok(self.scale_impl(py, scalar).into_py(py)),
            None => Ok(py.NotImplemented()),
        }
    }

    /// Matrix product ``self @ other`` (SymPy-style named alias).
    ///
    /// Same result as ``self @ other`` and ``self * other``. Raises a
    /// `MatrixError` (E-MAT-001) if the inner dimensions do not match.
    fn multiply(&self, py: Python<'_>, other: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
        self.matmul_impl(py, &other)
    }

    /// Scale every entry by `scalar` (an :class:`Expr`, :class:`DerivedResult`,
    /// Python ``int``, or ``float``). Same as ``self * scalar``.
    fn scalar_mul(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyMatrix> {
        match self.coerce_matrix_scalar(other, py)? {
            Some(scalar) => Ok(self.scale_impl(py, scalar)),
            None => Err(PyTypeError::new_err(
                "scalar_mul expects a scalar (Expr, DerivedResult, int, or float)",
            )),
        }
    }

    /// Elementwise (Hadamard) product — multiply corresponding entries.
    ///
    /// Requires both matrices to have the same shape. This is *not* the
    /// matrix product; use ``@`` / ``*`` / :meth:`multiply` for that. Exposed
    /// only as a named method (never on an operator) so ``*`` unambiguously
    /// means the matrix product, per the CAS convention.
    fn hadamard(&self, py: Python<'_>, other: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
        if self.inner.rows != other.inner.rows || self.inner.cols != other.inner.cols {
            return Err(matrix_error_to_py(MatrixError::DimensionMismatch {
                msg: format!(
                    "cannot take Hadamard product of {}×{} and {}×{}: shapes must match",
                    self.inner.rows, self.inner.cols, other.inner.rows, other.inner.cols
                ),
            }));
        }
        let pool = self.pool.borrow(py);
        let rows = self.inner.rows;
        let cols = self.inner.cols;
        let data: Vec<Vec<ExprId>> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        pool.inner
                            .mul(vec![self.inner.get(r, c), other.inner.get(r, c)])
                    })
                    .collect()
            })
            .collect();
        drop(pool);
        let m = Matrix::new(data).map_err(matrix_error_to_py)?;
        Ok(PyMatrix {
            inner: m,
            pool: self.pool.clone_ref(py),
        })
    }

    /// ``self ** n`` for a non-negative integer `n` (repeated matrix product).
    ///
    /// ``A ** 0`` is the identity matrix of the same size; ``A ** 1`` is `A`;
    /// ``A ** n`` is ``A @ A @ ... @ A`` (`n` factors). Requires a square
    /// matrix. Negative or non-integer exponents raise `TypeError`.
    fn __pow__(
        &self,
        py: Python<'_>,
        exp: &Bound<'_, PyAny>,
        modulo: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        if !modulo.is_none() {
            return Ok(py.NotImplemented());
        }
        let n: i64 = match exp.extract::<i64>() {
            Ok(n) => n,
            Err(_) => return Ok(py.NotImplemented()),
        };
        if n < 0 {
            return Err(PyTypeError::new_err(
                "matrix power requires a non-negative integer exponent (matrix inverse for \
                 negative powers is not supported via **; use .inverse())",
            ));
        }
        if self.inner.rows != self.inner.cols {
            return Err(matrix_error_to_py(MatrixError::DimensionMismatch {
                msg: format!(
                    "cannot raise a non-square {}×{} matrix to a power",
                    self.inner.rows, self.inner.cols
                ),
            }));
        }
        let pool = self.pool.borrow(py);
        // `A**0` is the identity; for `n >= 1` start from `A` (not `I @ A`) so
        // that `A**1` is exactly `A` and `A**n` matches `A @ A @ ... @ A`.
        let mut acc = if n == 0 {
            Matrix::identity(self.inner.rows, &pool.inner)
        } else {
            self.inner.clone()
        };
        for _ in 1..n {
            acc = acc
                .mul(&self.inner, &pool.inner)
                .map_err(matrix_error_to_py)?;
        }
        drop(pool);
        Ok(PyMatrix {
            inner: acc,
            pool: self.pool.clone_ref(py),
        }
        .into_py(py))
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

    /// `det(λI − M)` and the fresh λ symbol (`Expr`) used in that polynomial.
    fn characteristic_polynomial_lambda_minus_m(
        &self,
        py: Python<'_>,
    ) -> PyResult<(PyExpr, PyExpr)> {
        let pool = self.pool.borrow(py);
        let (poly, lam) = self
            .inner
            .characteristic_polynomial_lambda_minus_m(&pool.inner)
            .map_err(eigen_error_to_py)?;
        drop(pool);
        let pq = self.pool.clone_ref(py);
        Ok((
            PyExpr {
                id: poly,
                pool: pq.clone_ref(py),
            },
            PyExpr { id: lam, pool: pq },
        ))
    }

    /// Dictionary mapping each eigenvalue expression to its algebraic multiplicity.
    fn eigenvals(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pool = self.pool.borrow(py);
        let pairs = self
            .inner
            .eigenvalues(&pool.inner)
            .map_err(eigen_error_to_py)?;
        drop(pool);
        let out = PyDict::new_bound(py);
        for (e, mult) in pairs {
            let key = PyExpr {
                id: e,
                pool: self.pool.clone_ref(py),
            }
            .into_py(py);
            out.set_item(key, mult)?;
        }
        Ok(out.into())
    }

    /// SymPy-style triples `(eigenvalue, multiplicity, [column eigenvectors …])`.
    fn eigenvects(&self, py: Python<'_>) -> PyResult<Vec<(PyExpr, usize, Vec<PyMatrix>)>> {
        let pool = self.pool.borrow(py);
        let triples = self
            .inner
            .eigenvectors(&pool.inner)
            .map_err(eigen_error_to_py)?;
        drop(pool);
        Ok(triples
            .into_iter()
            .map(|(e, mult, vecs)| {
                (
                    PyExpr {
                        id: e,
                        pool: self.pool.clone_ref(py),
                    },
                    mult,
                    vecs.into_iter()
                        .map(|m| PyMatrix {
                            inner: m,
                            pool: self.pool.clone_ref(py),
                        })
                        .collect(),
                )
            })
            .collect())
    }

    /// `(P, D)` with `M @ P == P @ D` when the matrix is diagonalizable.
    fn diagonalize(&self, py: Python<'_>) -> PyResult<(PyMatrix, PyMatrix)> {
        let pool = self.pool.borrow(py);
        let (p, d) = self
            .inner
            .diagonalize(&pool.inner)
            .map_err(eigen_error_to_py)?;
        drop(pool);
        let pq = self.pool.clone_ref(py);
        Ok((
            PyMatrix {
                inner: p,
                pool: pq.clone_ref(py),
            },
            PyMatrix { inner: d, pool: pq },
        ))
    }

    fn nullspace(&self, py: Python<'_>) -> PyResult<Vec<PyMatrix>> {
        let pool = self.pool.borrow(py);
        let bas = self
            .inner
            .nullspace(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        Ok(bas
            .into_iter()
            .map(|m| PyMatrix {
                inner: m,
                pool: self.pool.clone_ref(py),
            })
            .collect())
    }

    fn rank(&self, py: Python<'_>) -> PyResult<usize> {
        let pool = self.pool.borrow(py);
        self.inner
            .rank(&pool.inner)
            .map_err(linear_algebra_error_to_py)
    }

    fn rref(&self, py: Python<'_>) -> PyResult<PyMatrix> {
        let pool = self.pool.borrow(py);
        let r = self
            .inner
            .rref(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        Ok(PyMatrix {
            inner: r,
            pool: self.pool.clone_ref(py),
        })
    }

    fn column_space(&self, py: Python<'_>) -> PyResult<Vec<PyMatrix>> {
        let pool = self.pool.borrow(py);
        let bas = self
            .inner
            .column_space(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        Ok(bas
            .into_iter()
            .map(|m| PyMatrix {
                inner: m,
                pool: self.pool.clone_ref(py),
            })
            .collect())
    }

    fn row_space(&self, py: Python<'_>) -> PyResult<Vec<PyMatrix>> {
        let pool = self.pool.borrow(py);
        let bas = self
            .inner
            .row_space(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        Ok(bas
            .into_iter()
            .map(|m| PyMatrix {
                inner: m,
                pool: self.pool.clone_ref(py),
            })
            .collect())
    }

    fn lu(&self, py: Python<'_>) -> PyResult<(PyMatrix, PyMatrix, Vec<usize>)> {
        let pool = self.pool.borrow(py);
        let lu = self
            .inner
            .lu(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        let pq = self.pool.clone_ref(py);
        Ok((
            PyMatrix {
                inner: lu.l,
                pool: pq.clone_ref(py),
            },
            PyMatrix {
                inner: lu.u,
                pool: pq,
            },
            lu.perm,
        ))
    }

    fn qr(&self, py: Python<'_>) -> PyResult<(PyMatrix, PyMatrix)> {
        let pool = self.pool.borrow(py);
        let qr = self
            .inner
            .qr(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        let pq = self.pool.clone_ref(py);
        Ok((
            PyMatrix {
                inner: qr.q,
                pool: pq.clone_ref(py),
            },
            PyMatrix {
                inner: qr.r,
                pool: pq,
            },
        ))
    }

    fn cholesky(&self, py: Python<'_>) -> PyResult<PyMatrix> {
        let pool = self.pool.borrow(py);
        let l = self
            .inner
            .cholesky(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        Ok(PyMatrix {
            inner: l,
            pool: self.pool.clone_ref(py),
        })
    }

    fn jordan_form(&self, py: Python<'_>) -> PyResult<(PyMatrix, PyMatrix)> {
        let pool = self.pool.borrow(py);
        let (p, j) = self
            .inner
            .jordan_form(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        let pq = self.pool.clone_ref(py);
        Ok((
            PyMatrix {
                inner: p,
                pool: pq.clone_ref(py),
            },
            PyMatrix { inner: j, pool: pq },
        ))
    }

    fn rational_canonical_form(&self, py: Python<'_>) -> PyResult<(PyMatrix, PyMatrix)> {
        let pool = self.pool.borrow(py);
        let (p, c) = self
            .inner
            .rational_canonical_form(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        let pq = self.pool.clone_ref(py);
        Ok((
            PyMatrix {
                inner: p,
                pool: pq.clone_ref(py),
            },
            PyMatrix { inner: c, pool: pq },
        ))
    }

    fn minimal_polynomial(&self, py: Python<'_>) -> PyResult<PyExpr> {
        let pool = self.pool.borrow(py);
        let (poly, _lam) = self
            .inner
            .minimal_polynomial(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        Ok(PyExpr {
            id: poly,
            pool: self.pool.clone_ref(py),
        })
    }

    fn matrix_exp(&self, py: Python<'_>) -> PyResult<PyMatrix> {
        let pool = self.pool.borrow(py);
        let expm = self
            .inner
            .matrix_exp(&pool.inner)
            .map_err(linear_algebra_error_to_py)?;
        drop(pool);
        Ok(PyMatrix {
            inner: expm,
            pool: self.pool.clone_ref(py),
        })
    }

    fn inverse(&self, py: Python<'_>) -> PyResult<PyMatrix> {
        let pool = self.pool.borrow(py);
        let inv = self
            .inner
            .inverse(&pool.inner)
            .map_err(matrix_error_to_py)?;
        drop(pool);
        Ok(PyMatrix {
            inner: inv,
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

// Private (non-`#[pymethods]`) helpers shared by the multiplication surface.
impl PyMatrix {
    /// Matrix product `self @ other`, reused by `@`, `*`, `**`, and `multiply`.
    fn matmul_impl(&self, py: Python<'_>, other: &PyMatrix) -> PyResult<PyMatrix> {
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

    /// Scale every entry by an already-coerced `ExprId` scalar.
    fn scale_impl(&self, py: Python<'_>, scalar: ExprId) -> PyMatrix {
        let pool = self.pool.borrow(py);
        let m = self.inner.scale(scalar, &pool.inner);
        drop(pool);
        PyMatrix {
            inner: m,
            pool: self.pool.clone_ref(py),
        }
    }

    /// Coerce `ob` into a scalar `ExprId` in this matrix's pool.
    ///
    /// Accepts `Expr`, `DerivedResult`, Python `int` (including bignum), and
    /// `float`. Returns `Ok(None)` for anything else so operators can fall
    /// back to `NotImplemented`. Pool mismatches raise (like the rest of the
    /// API).
    fn coerce_matrix_scalar(
        &self,
        ob: &Bound<'_, PyAny>,
        py: Python<'_>,
    ) -> PyResult<Option<ExprId>> {
        if let Ok(e) = ob.extract::<PyRef<PyExpr>>() {
            if !e.pool.is(&self.pool) {
                return Err(pool_mismatch_err());
            }
            return Ok(Some(e.id));
        }
        if let Ok(dr) = ob.downcast::<PyDerivedResult>() {
            let dr = dr.borrow();
            if !dr.value.pool.is(&self.pool) {
                return Err(pool_mismatch_err());
            }
            return Ok(Some(dr.value.id));
        }
        let pool = self.pool.borrow(py);
        if let Ok(n) = ob.extract::<i64>() {
            return Ok(Some(pool.inner.integer(n)));
        }
        if let Ok(f) = ob.extract::<f64>() {
            return Ok(Some(pool.inner.float(f, 53)));
        }
        if ob.is_instance_of::<PyInt>() {
            return Ok(Some(integer_into_pool(&pool.inner, ob)?));
        }
        Ok(None)
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
        alkahest_core::check_expr_depths(&pool.inner, &f_ids).map_err(depth_error_to_py)?;
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

/// Symbolic system of first-order ordinary differential equations.
///
/// Represents the system ``d(state_vars)/dt = rhs`` driven by ``time_var``.
///
/// Parameters
/// ----------
/// state_vars : list[Expr]
///     Symbolic state variables (e.g. ``[x, y]``).
/// rhs : list[Expr]
///     Right-hand side expressions, one per state variable.
/// time_var : Expr
///     The independent time variable.
///
/// Example::
///
///     p = alkahest.ExprPool()
///     t, x = p.symbol("t"), p.symbol("x")
///     ode = alkahest.ODE([x], [-x], t)   # dx/dt = -x
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

    /// Build an explicit first-order system ``d(state_vars)/dt = rhs``.
    ///
    /// Parameters
    /// ----------
    /// state_vars : list[Expr]
    ///     Symbolic state variables.
    /// rhs : list[Expr]
    ///     Right-hand sides, one per state variable and in the same order.
    /// time_var : Expr
    ///     The independent variable.
    ///
    /// All three arguments are **positional**; there are no keyword forms.
    /// ``ODE(state_vars, rhs, time_var)`` is accepted as a synonym.
    /// For an implicit or constrained system use :class:`DAE` instead, and for
    /// a scalar higher-order equation use :func:`lower_to_first_order`.
    ///
    /// Example::
    ///
    ///     # Harmonic oscillator as a first-order system: x' = v, v' = -x
    ///     ode = alkahest.ODE.new([x, v], [v, -x], t)
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

    /// Number of state variables (the order of the system).
    #[getter]
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

/// `alkahest.lower_to_first_order(var, rhs, order, time_var)` — lower a scalar
/// higher-order ODE to an equivalent first-order system.
///
/// Takes the *pieces* of the equation, not an :class:`ODE` object: the scalar
/// unknown, the right-hand side, the order of the derivative on the left, and
/// the independent variable.  All four are required and positional.
///
/// Parameters
/// ----------
/// var : Expr
///     The scalar unknown, e.g. ``x``.
/// rhs : Expr
///     Right-hand side of ``d^order(var)/dt^order = rhs``.
/// order : int
///     Order of the equation (``2`` for ``x''``).
/// time_var : Expr
///     The independent variable.
///
/// Returns
/// -------
/// ODE
///     A first-order system whose state is ``[var, var', …, var^(order-1)]``.
///
/// Example::
///
///     # x'' = -4x  →  [x' = x', (x')' = -4x]
///     ode = alkahest.lower_to_first_order(x, -4 * x, 2, t)
///     ode.order          # 2
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

/// Symbolic differential-algebraic equation system.
///
/// A DAE is a system of implicit equations mixing differential and algebraic
/// constraints: ``F(t, variables, derivatives) = 0``.
///
/// Parameters (via :meth:`DAE.new`)
/// ---------------------------------
/// equations : list[Expr]
///     Implicit equations (each equals zero).
/// variables : list[Expr]
///     Algebraic and state variables.
/// derivatives : list[Expr]
///     Derivative expressions corresponding to ``variables``.
/// time_var : Expr
///     The independent time variable.
///
/// Use :func:`pantelides` to reduce the differential index before simulation.
///
/// Attributes
/// ----------
/// n_equations : int
/// n_variables : int
/// time_var : Expr
/// index : int | None
///     Differentiation index, set by :func:`pantelides` on the DAE it
///     returns; ``None`` on a DAE that has not been index-reduced.
///
/// Example::
///
///     p = alkahest.ExprPool()
///     t, x, dx = p.symbol("t"), p.symbol("x"), p.symbol("dx/dt")
///     dae = alkahest.DAE.new([dx - x], [x], [dx], t)   # x' = x
///     dae.equations()      # [-x + dx/dt]
#[pyclass(name = "DAE")]
struct PyDAE {
    inner: DAE,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyDAE {
    /// Build a DAE from implicit equations.
    ///
    /// Parameters
    /// ----------
    /// equations : list[Expr]
    ///     Implicit equations, each meaning ``g = 0``.  Write ``x' = f`` as
    ///     ``dx - f``, using a *separate symbol* for the derivative.
    /// variables : list[Expr]
    ///     Dependent variables (state and algebraic).
    /// derivatives : list[Expr]
    ///     Symbol standing for the time derivative of ``variables[i]``, e.g.
    ///     ``pool.symbol("dx/dt")``.  Alkahest does not parse the name; the
    ///     positional pairing is what makes it a derivative.
    /// time_var : Expr
    ///     The independent variable.
    ///
    /// Example::
    ///
    ///     # Pendulum in Cartesian coordinates (index 3):
    ///     #   x' = u,  u' = -lam*x,  x**2 + y**2 - 1 = 0
    ///     dae = alkahest.DAE.new(
    ///         [dx - u, du + lam * x, x**2 + y**2 - one],
    ///         [x, u, lam],
    ///         [dx, du, dlam],
    ///         t,
    ///     )
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

    /// Number of equations in the system.
    #[getter]
    fn n_equations(&self) -> usize {
        self.inner.n_equations()
    }

    /// Number of dependent variables in the system.
    #[getter]
    fn n_variables(&self) -> usize {
        self.inner.n_variables()
    }

    /// The independent (time) variable.
    #[getter]
    fn time_var(&self, py: Python<'_>) -> PyExpr {
        PyExpr {
            id: self.inner.time_var,
            pool: self.pool.clone_ref(py),
        }
    }

    /// Differentiation index, or ``None`` if it has not been computed.
    ///
    /// :func:`pantelides` sets it on the DAE it returns: `0` means the input
    /// already had a perfect structural matching, `k` means `k` rounds of
    /// differentiation were needed.
    #[getter]
    fn index(&self) -> Option<usize> {
        self.inner.index
    }

    /// The implicit equations, each meaning ``g = 0``.
    ///
    /// After :func:`pantelides` or :func:`rosenfeld_groebner` this is how you
    /// see which equations were added by differentiation.
    fn equations(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.inner
            .equations
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// The dependent variables, parallel to :meth:`derivatives`.
    fn variables(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.inner
            .variables
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// The derivative symbols: ``derivatives()[i]`` is ``d(variables()[i])/dt``.
    ///
    /// Index reduction appends higher jets here (``d2x/dt2``, …), so this is
    /// how the extra variables of a prolonged system get names.
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

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        format!("DAE(\n{}\n)", self.inner.display(&pool.inner))
    }
}

/// `alkahest.pantelides(dae)` — structural index reduction of a DAE.
///
/// Repeatedly differentiates the equations that a maximum bipartite matching
/// leaves unmatched, until every equation is matched to a variable, and
/// returns the **reduced** :class:`DAE` — not a report object.
///
/// The reduced system has more equations and more derivative jets than the
/// input; read them with :meth:`DAE.equations` and :meth:`DAE.derivatives`.
/// :attr:`DAE.index` on the result is the number of differentiation rounds it
/// took (``0`` = the input already matched).
///
/// Raises `ValueError` (`E-DAE-002`) if the index exceeds 10 — try
/// :func:`dae_index_reduce`, which falls back to :func:`rosenfeld_groebner`.
///
/// This is a *structural* algorithm: it looks at which variables occur in
/// which equations, not at whether the coefficients actually make the system
/// solvable.
///
/// Example::
///
///     reduced = alkahest.pantelides(dae)
///     reduced.index            # differentiation rounds used
///     reduced.equations()      # original equations plus the differentiated ones
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

/// Ordinary differential equation system with discrete events.
///
/// A :class:`HybridODE` wraps a continuous :class:`ODE` and a list of
/// :class:`Event` objects.  Each event specifies a guard condition and a
/// reset map that is applied when the guard crosses zero.
///
/// Construction::
///
///     hybrid = alkahest.HybridODE.new(ode)
///     hybrid = hybrid.add_event(event)
///
/// Attributes
/// ----------
/// n_events : int
///     Number of discrete events attached to this system.
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

    /// Number of registered discrete events.
    #[getter]
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

/// A physical component (resistor, capacitor, voltage source, …) with named
/// :class:`Port` connectors and internal constitutive equations.
///
/// Construct components via :func:`resistor`, :func:`capacitor`, or
/// :func:`voltage_source`, then register them on an :class:`AcausalSystem`
/// with :meth:`AcausalSystem.add_component` and wire them together with
/// :meth:`AcausalSystem.connect`.
#[pyclass(name = "Component")]
#[derive(Clone)]
struct PyComponent {
    inner: Component,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyComponent {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Number of constitutive equations contributed by this component.
    #[getter]
    fn n_equations(&self) -> usize {
        self.inner.equations.len()
    }

    /// Number of external connection ports.
    #[getter]
    fn n_ports(&self) -> usize {
        self.inner.ports.len()
    }

    /// All ports, in declaration order.
    fn ports(&self, py: Python<'_>) -> Vec<PyPort> {
        self.inner
            .ports
            .iter()
            .map(|p| PyPort {
                inner: p.clone(),
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// Look up a port by its full name (e.g. `"R1.p"`), or `None` if absent.
    fn port(&self, py: Python<'_>, name: &str) -> Option<PyPort> {
        self.inner.port(name).map(|p| PyPort {
            inner: p.clone(),
            pool: self.pool.clone_ref(py),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Component(name={:?}, n_ports={}, n_equations={})",
            self.inner.name,
            self.inner.ports.len(),
            self.inner.equations.len()
        )
    }
}

/// Acausal component-based modelling system.
///
/// An :class:`AcausalSystem` aggregates components connected through
/// :class:`Port` objects (potential/flow pairs).  Add components with
/// :meth:`add_component`, wire ports together with :meth:`connect`, and
/// call :meth:`flatten` to convert the component network into an equivalent
/// :class:`DAE` suitable for simulation or index reduction.
///
/// Example (RC circuit)::
///
///     p = alkahest.ExprPool()
///     t = p.symbol("t")
///
///     src = alkahest.voltage_source("V1", p.symbol("Vs"))
///     res = alkahest.resistor("R1", p.symbol("R"))
///     cap = alkahest.capacitor("C1", p.symbol("C"))
///
///     sys = alkahest.AcausalSystem(p)
///     sys.add_component(src["component"])
///     sys.add_component(res["component"])
///     sys.add_component(cap["component"])
///
///     sys.connect(src["component"].port("V1.p"), res["component"].port("R1.p"))
///     sys.connect(res["component"].port("R1.n"), cap["component"].port("C1.p"))
///     sys.connect(cap["component"].port("C1.n"), src["component"].port("V1.n"))
///
///     dae = sys.flatten(t)
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

    /// Add a component (e.g. from :func:`resistor`, :func:`capacitor`,
    /// :func:`voltage_source`) to the system.
    fn add_component(&mut self, component: PyRef<PyComponent>) {
        self.inner.add_component(component.inner.clone());
    }

    /// Connect two ports: equates their potentials and balances their flows
    /// (`a.potential == b.potential`, `a.flow + b.flow == 0`).
    fn connect(&mut self, port_a: PyRef<PyPort>, port_b: PyRef<PyPort>) {
        self.inner.connect(&port_a.inner, &port_b.inner);
    }

    /// Flatten all component and connection equations into a :class:`DAE`.
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

/// Pack a core `Component` into the dict shape returned by `resistor`,
/// `capacitor`, and `voltage_source`: `{"name", "n_equations", "n_ports",
/// "component"}`, where `"component"` is a :class:`Component` instance that
/// can be passed to `AcausalSystem.add_component` and `.port(name)`.
fn component_to_pydict(
    py: Python<'_>,
    comp: Component,
    pool: Py<PyExprPool>,
) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("name", comp.name.clone())?;
    d.set_item("n_equations", comp.equations.len())?;
    d.set_item("n_ports", comp.ports.len())?;
    d.set_item("component", PyComponent { inner: comp, pool }.into_py(py))?;
    Ok(d.into_py(py))
}

/// `alkahest.resistor(name, resistance)` — create a resistor component.
///
/// Returns a dict `{"name", "n_equations", "n_ports", "component"}`; the
/// `"component"` entry is a :class:`Component` usable with
/// `AcausalSystem.add_component` and `.port(name)`.
#[pyfunction]
#[pyo3(name = "resistor")]
fn py_resistor(py: Python<'_>, name: &str, resistance: PyRef<PyExpr>) -> PyResult<PyObject> {
    let pool_py = resistance.pool.clone_ref(py);
    let pool = resistance.pool.borrow(py);
    let comp = core_resistor(name, resistance.id, &pool.inner);
    drop(pool);
    component_to_pydict(py, comp, pool_py)
}

/// `alkahest.capacitor(name, capacitance)` — create an ideal capacitor
/// component (`C * dv/dt = i`).
///
/// Returns a dict `{"name", "n_equations", "n_ports", "component"}`; the
/// `"component"` entry is a :class:`Component` usable with
/// `AcausalSystem.add_component` and `.port(name)`.
#[pyfunction]
#[pyo3(name = "capacitor")]
fn py_capacitor(py: Python<'_>, name: &str, capacitance: PyRef<PyExpr>) -> PyResult<PyObject> {
    let pool_py = capacitance.pool.clone_ref(py);
    let pool = capacitance.pool.borrow(py);
    let comp = core_capacitor(name, capacitance.id, &pool.inner);
    drop(pool);
    component_to_pydict(py, comp, pool_py)
}

/// `alkahest.voltage_source(name, voltage)` — create an ideal voltage
/// source component (`v_p - v_n = V`).
///
/// Returns a dict `{"name", "n_equations", "n_ports", "component"}`; the
/// `"component"` entry is a :class:`Component` usable with
/// `AcausalSystem.add_component` and `.port(name)`.
#[pyfunction]
#[pyo3(name = "voltage_source")]
fn py_voltage_source(py: Python<'_>, name: &str, voltage: PyRef<PyExpr>) -> PyResult<PyObject> {
    let pool_py = voltage.pool.clone_ref(py);
    let pool = voltage.pool.borrow(py);
    let comp = core_voltage_source(name, voltage.id, &pool.inner);
    drop(pool);
    component_to_pydict(py, comp, pool_py)
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
    if !core_jit_available() {
        let warnings = py.import_bound("warnings")?;
        warnings.call_method1(
            "warn",
            (
                "JIT compilation (LLVM) is not available in this build; \
                 compile_expr() is falling back to the tree-walking interpreter. \
                 For native performance install a release wheel tagged +jit (see README), \
                 or rebuild with LLVM 15 via \
                 maturin develop --manifest-path alkahest-py/Cargo.toml --features jit.",
                py.get_type_bound::<pyo3::exceptions::PyRuntimeWarning>(),
                // stack level 2 so the warning points at the caller's site
                2i32,
            ),
        )?;
    }

    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
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
        inner: Arc::new(compiled),
    })
}

/// Return True if any native JIT backend is available in this build.
///
/// When False, `compile_expr` falls back to the tree-walking interpreter and
/// emits a RuntimeWarning.
#[pyfunction]
#[pyo3(name = "jit_is_available")]
fn py_jit_is_available() -> bool {
    core_jit_available()
}

/// Return the Cargo feature flags compiled into this extension.
///
/// This reports the installed artifact, not the project defaults or the
/// availability of Python fallback functions.
///
/// # Every key must name something a caller can reach
///
/// A capability bit exists so an agent can decide what to use without probing.
/// That makes an unreachable `true` the same class of defect as a silent wrong
/// answer, and a bit that correlates with nothing at all only marginally
/// better. Two keys were dropped in 3.8 for failing that test, and
/// `tests/test_agent_contract.py::test_every_advertised_feature_has_an_entry_point`
/// now walks this map and refuses any key without a named entry point:
///
/// - `groebner_cuda` reported `--features groebner-cuda`, but the GPU Gröbner
///   kernel has no PyO3 binding at all — `GroebnerBasis` exposes only CPU
///   methods and `compute_groebner_basis_gpu` is reachable from Rust only. No
///   Python observation could distinguish `true` from `false`.
/// - `numpy` reported a Cargo feature that gated the `numpy` crate, which
///   this crate never used. `alkahest.numpy_eval` works through the buffer
///   protocol regardless, so the bit was `false` on every build that shipped
///   and predicted nothing about NumPy support either way.
#[pyfunction]
#[pyo3(name = "_build_features")]
fn py_build_features() -> std::collections::HashMap<String, bool> {
    [
        ("egraph", cfg!(feature = "egraph")),
        ("groebner", cfg!(feature = "groebner")),
        // Retain the Cargo feature names for compatibility and expose
        // backend-specific names so callers need not infer what `jit` means.
        // `cuda` implies `alkahest-core/jit` (see alkahest-core's Cargo.toml:
        // `cuda = ["jit", "dep:cudarc"]`), so a build with `--features cuda`
        // links the LLVM backend even though *this* crate's own `jit` feature
        // is off. Reporting `cfg!(feature = "jit")` alone therefore said
        // `llvm_jit: false` on a build that demonstrably emits NVPTX — the
        // capability contract has to describe what is linked, not which flag
        // the caller happened to name.
        ("jit", cfg!(feature = "jit") || cfg!(feature = "cuda")),
        ("cranelift", cfg!(feature = "cranelift")),
        ("llvm_jit", cfg!(feature = "jit") || cfg!(feature = "cuda")),
        ("cranelift_jit", cfg!(feature = "cranelift")),
        ("parallel", cfg!(feature = "parallel")),
        // `cuda` stays: it is falsifiable. `true` guarantees `ak.compile_cuda`
        // and `ak.CudaCompiledFn` exist, `false` guarantees they do not.
        ("cuda", cfg!(feature = "cuda")),
    ]
    .into_iter()
    .map(|(name, enabled)| (name.to_string(), enabled))
    .collect()
}

/// Evaluate a symbolic expression numerically using the interpreter.
///
/// `bindings` is a dict mapping Expr → float.
#[pyfunction]
#[pyo3(name = "eval_expr")]
fn py_eval_expr(
    py: Python<'_>,
    expr: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyDict>,
) -> PyResult<f64> {
    let (expr_id, pool_py) = if let Ok(dr) = expr.downcast::<PyDerivedResult>() {
        let dr = dr.borrow();
        (dr.value.id, dr.value.pool.clone_ref(py))
    } else {
        let e: PyRef<PyExpr> = expr.extract()?;
        (e.id, e.pool.clone_ref(py))
    };
    let pool = pool_py.borrow(py);
    guard_depth(&pool.inner, expr_id)?;
    let mut env = std::collections::HashMap::new();
    for (key, value) in bindings.iter() {
        let var: PyRef<PyExpr> = key.extract()?;
        let val: f64 = value.extract()?;
        env.insert(var.id, val);
    }
    core_eval_interp_checked(expr_id, &env, &pool.inner).map_err(|e| match e {
        alkahest_core::InterpEvalError::Unevaluable => pyo3::exceptions::PyValueError::new_err(
            "expression could not be evaluated (unbound variable or unsupported node)",
        ),
        alkahest_core::InterpEvalError::NonFinite => domain_error(
            py,
            "E-EVAL-009",
            "expression is undefined at this point: substituting the bindings produced a \
             non-finite result (e.g. division by zero). This is not the same as a removable- \
             singularity limit — if you want the limiting value, call cancel() first and \
             evaluate the simplified expression."
                .to_string(),
            "result is not finite",
        ),
    })
}

#[pyclass(name = "CompiledFn", unsendable)]
struct PyCompiledFn {
    /// Shared ownership — multiple `PyCompiledFn` objects from a `CompileCache`
    /// reference the same compiled code without recompilation.
    inner: Arc<alkahest_core::CompiledFn>,
}

#[pymethods]
impl PyCompiledFn {
    /// Evaluate at **one point**, given as a single sequence of floats.
    ///
    /// `f([1.0, 2.0])` — one argument holding every input value, not one
    /// argument per input. To evaluate over arrays of points instead, use
    /// `alkahest.numpy_eval(f, xs, ys)`, which takes the arrays as separate
    /// positional arguments.
    #[pyo3(signature = (*args))]
    fn __call__(&self, args: &Bound<'_, PyTuple>) -> PyResult<f64> {
        let n = self.inner.n_inputs;
        // Spelled as *args so that `f(1.0, 2.0)` — the natural guess — is
        // answered with the convention rather than with PyO3's arity message.
        if args.len() != 1 {
            return Err(PyTypeError::new_err(format!(
                "CompiledFn takes one argument: the {n} input value(s) as a single sequence, \
                 e.g. f([1.0, 2.0]) — got {} positional arguments. To evaluate over arrays of \
                 points use alkahest.numpy_eval(f, xs, ys), which does take one argument per \
                 input variable.",
                args.len()
            )));
        }
        let arg = args.get_item(0)?;
        let inputs: Vec<f64> = arg.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "CompiledFn takes the {n} input value(s) as a single sequence, e.g. f([1.0, 2.0]); \
                 got {}. To evaluate over arrays of points use alkahest.numpy_eval(f, xs, ys).",
                py_type_name(&arg)
            ))
        })?;
        if inputs.len() != n {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}; CompiledFn evaluates one point per call — the \
                 sequence holds one value per input variable, not one point per element (use \
                 alkahest.numpy_eval(f, xs, ys) for a batch)",
                n,
                inputs.len()
            )));
        }
        Ok(self.inner.call(&inputs))
    }

    #[getter]
    fn n_inputs(&self) -> usize {
        self.inner.n_inputs
    }

    fn __repr__(&self) -> String {
        format!("<CompiledFn n_inputs={}>", self.inner.n_inputs)
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
        // Checked: the product wraps in release, and a wrapped product that
        // happens to equal `inputs_flat.len()` let a `2**63`-long slice range
        // through to panic a few lines below.
        let expected = n_vars.checked_mul(n_points);
        if expected != Some(inputs_flat.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "inputs_flat length {} != n_vars({}) * n_points({})",
                inputs_flat.len(),
                n_vars,
                n_points
            )));
        }
        if n_vars != self.inner.n_inputs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} variables, got {}",
                self.inner.n_inputs, n_vars
            )));
        }
        let cols: Vec<&[f64]> = (0..n_vars)
            .map(|i| &inputs_flat[i * n_points..(i + 1) * n_points])
            .collect();
        let mut output = vec![0.0f64; n_points];
        self.inner.call_batch(&cols, &mut output);
        Ok(output)
    }

    /// Parallel batch evaluation using Rayon (requires `--features parallel`).
    ///
    /// Identical to :meth:`call_batch_raw` but distributes the N points across
    /// all available CPU cores.  The GIL is released during evaluation so other
    /// Python threads are not blocked.
    ///
    /// Use :func:`alkahest.numpy_eval_par` for a NumPy-friendly wrapper.
    #[cfg(feature = "parallel")]
    fn call_batch_raw_par(
        &self,
        py: Python<'_>,
        inputs_flat: Vec<f64>,
        n_vars: usize,
        n_points: usize,
    ) -> PyResult<Vec<f64>> {
        // Checked: the product wraps in release, and a wrapped product that
        // happens to equal `inputs_flat.len()` let a `2**63`-long slice range
        // through to panic a few lines below.
        let expected = n_vars.checked_mul(n_points);
        if expected != Some(inputs_flat.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "inputs_flat length {} != n_vars({}) * n_points({})",
                inputs_flat.len(),
                n_vars,
                n_points
            )));
        }
        if n_vars != self.inner.n_inputs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} variables, got {}",
                self.inner.n_inputs, n_vars
            )));
        }
        let cols: Vec<&[f64]> = (0..n_vars)
            .map(|i| &inputs_flat[i * n_points..(i + 1) * n_points])
            .collect();
        let mut output = vec![0.0f64; n_points];
        // Release the GIL for the duration of parallel evaluation so other
        // Python threads can run while Rayon works on the native side.
        py.allow_threads(|| {
            self.inner.call_batch_par(&cols, &mut output);
        });
        Ok(output)
    }

    /// Fast-path batch evaluation directly against buffer-protocol objects
    /// (NumPy `float64` arrays, `array.array('d', …)`, memoryviews, …).
    ///
    /// Unlike :meth:`call_batch_raw`, this never materialises a Python list
    /// or `float` object per element: each input array's contents are copied
    /// into a native `Vec<f64>` in one bulk `memcpy`-style operation (via
    /// PyO3's buffer protocol), the compiled function runs with the GIL
    /// released, and the result is written back into the caller-supplied
    /// `output` buffer the same way.
    ///
    /// `inputs` must contain exactly `n_inputs` buffers, each of length
    /// `n_points`, C-contiguous, and holding `float64` elements. `output`
    /// must be a writable buffer of the same shape (e.g. `numpy.empty`).
    ///
    /// This is the method used internally by :func:`alkahest.numpy_eval`;
    /// most callers should use that helper rather than calling this
    /// directly.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    fn call_batch_buffer(
        &self,
        py: Python<'_>,
        inputs: Vec<PyBuffer<f64>>,
        output: PyBuffer<f64>,
    ) -> PyResult<()> {
        let n_points = output.item_count();
        let cols = extract_batch_columns(&inputs, self.inner.n_inputs, n_points, py)?;
        let col_refs: Vec<&[f64]> = cols.iter().map(|v| v.as_slice()).collect();

        let mut out_vec = vec![0.0f64; n_points];
        py.allow_threads(|| {
            self.inner.call_batch(&col_refs, &mut out_vec);
        });

        output.copy_from_slice(py, &out_vec).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to write output buffer: {e}"))
        })
    }

    /// Parallel counterpart to :meth:`call_batch_buffer` (requires
    /// `--features parallel`). Distributes the `n_points` evaluations across
    /// all available CPU cores via Rayon; the GIL is released for the
    /// duration of the native call.
    ///
    /// Used internally by :func:`alkahest.numpy_eval_par`.
    #[cfg(all(feature = "parallel", any(not(Py_LIMITED_API), Py_3_11)))]
    fn call_batch_buffer_par(
        &self,
        py: Python<'_>,
        inputs: Vec<PyBuffer<f64>>,
        output: PyBuffer<f64>,
    ) -> PyResult<()> {
        let n_points = output.item_count();
        let cols = extract_batch_columns(&inputs, self.inner.n_inputs, n_points, py)?;
        let col_refs: Vec<&[f64]> = cols.iter().map(|v| v.as_slice()).collect();

        let mut out_vec = vec![0.0f64; n_points];
        py.allow_threads(|| {
            self.inner.call_batch_par(&col_refs, &mut out_vec);
        });

        output.copy_from_slice(py, &out_vec).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to write output buffer: {e}"))
        })
    }
}

/// Shared validation + bulk-copy helper for [`PyCompiledFn::call_batch_buffer`]
/// and [`PyCompiledFn::call_batch_buffer_par`].
///
/// Copies each input buffer into an owned `Vec<f64>` via a single
/// `PyBuffer::to_vec` bulk copy (no per-element Python object boxing), after
/// validating shapes so the native `call_batch` call is safe.
#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
fn extract_batch_columns(
    inputs: &[PyBuffer<f64>],
    n_inputs: usize,
    n_points: usize,
    py: Python<'_>,
) -> PyResult<Vec<Vec<f64>>> {
    if inputs.len() != n_inputs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {} input array(s), got {}",
            n_inputs,
            inputs.len()
        )));
    }
    inputs
        .iter()
        .map(|buf| {
            if buf.item_count() != n_points {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "all input arrays must have length {n_points}, got {}",
                    buf.item_count()
                )));
            }
            if !buf.is_c_contiguous() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "input arrays must be C-contiguous (use np.ascontiguousarray)",
                ));
            }
            buf.to_vec(py).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("failed to read input buffer: {e}"))
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Compiled expression cache
// ---------------------------------------------------------------------------

/// Content-addressed cache of JIT-compiled functions.
///
/// Because Alkahest hash-conses expressions, the same expression tree always
/// produces the same ``ExprId``.  ``CompileCache`` exploits this to skip
/// recompilation: the first ``compile()`` call JIT-compiles the expression;
/// subsequent calls with the same ``(expr, inputs)`` pair return the cached
/// result in O(1) time.
///
/// Example::
///
///     cache = alkahest.CompileCache()
///     x = alkahest.symbol("x")
///     expr = x ** 2
///
///     f = cache.compile(expr, [x])   # compiles
///     g = cache.compile(expr, [x])   # cache hit — same CompiledFn
///
///     assert f(3.0) == 9.0
///     print(cache.stats())           # {'len': 1, 'compiles': 1, 'hits': 1, 'hit_rate': 0.5}
#[pyclass(name = "CompileCache", unsendable)]
struct PyCompileCache {
    inner: CoreCompileCache,
}

#[pymethods]
impl PyCompileCache {
    /// Create a new, empty compile cache.
    #[new]
    fn new() -> Self {
        PyCompileCache {
            inner: CoreCompileCache::new(),
        }
    }

    /// Compile `expr` with the given `inputs`, returning a cached :class:`CompiledFn`.
    ///
    /// The first call for a given ``(expr, inputs)`` pair JIT-compiles the
    /// expression.  Subsequent calls return the same :class:`CompiledFn`
    /// without recompilation.
    ///
    /// Parameters
    /// ----------
    /// expr : Expr
    ///     The expression to compile.
    /// inputs : list[Expr]
    ///     Ordered list of symbolic input variables.
    ///
    /// Returns
    /// -------
    /// CompiledFn
    fn compile(
        &mut self,
        py: Python<'_>,
        expr: PyRef<PyExpr>,
        inputs: &Bound<'_, PyList>,
    ) -> PyResult<PyCompiledFn> {
        let input_ids: Vec<ExprId> = inputs
            .iter()
            .map(|item| {
                let e: PyRef<PyExpr> = item.extract()?;
                Ok(e.id)
            })
            .collect::<PyResult<_>>()?;

        let pool_ref = expr.pool.borrow(py);
        let arc = self
            .inner
            .compile(expr.id, &input_ids, &pool_ref.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        drop(pool_ref);

        Ok(PyCompiledFn { inner: arc })
    }

    /// Number of ``(expr, inputs)`` pairs currently cached.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// ``True`` if the cache contains no entries.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Remove all cached entries.
    ///
    /// Live :class:`CompiledFn` objects already returned to Python callers
    /// remain valid — they hold an independent reference to the compiled code.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// ``True`` if a compiled function for ``(expr, inputs)`` is in the cache.
    fn contains(
        &self,
        py: Python<'_>,
        expr: PyRef<PyExpr>,
        inputs: &Bound<'_, PyList>,
    ) -> PyResult<bool> {
        let input_ids: Vec<ExprId> = inputs
            .iter()
            .map(|item| {
                let e: PyRef<PyExpr> = item.extract()?;
                Ok(e.id)
            })
            .collect::<PyResult<_>>()?;
        let _ = py;
        Ok(self.inner.contains(expr.id, &input_ids))
    }

    /// Return a dict with cache statistics.
    ///
    /// Keys: ``len``, ``compiles``, ``hits``, ``hit_rate``.
    fn stats(&self) -> std::collections::HashMap<&'static str, f64> {
        let mut m = std::collections::HashMap::new();
        m.insert("len", self.inner.len() as f64);
        m.insert("compiles", self.inner.compile_count() as f64);
        m.insert("hits", self.inner.hit_count() as f64);
        m.insert("hit_rate", self.inner.hit_rate());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "<CompileCache len={} hits={} compiles={}>",
            self.inner.len(),
            self.inner.hit_count(),
            self.inner.compile_count(),
        )
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
    fn new(mid: f64, rad: f64, prec: u32) -> PyResult<Self> {
        // `rug::Float::with_val` panics on prec 0, and the radius path inside
        // `from_midpoint_radius` doubles it — see `checked_prec`.
        let prec = checked_prec(prec)?;
        Ok(PyArbBall {
            inner: CoreArbBall::from_midpoint_radius(mid, rad, prec),
        })
    }

    #[getter]
    fn mid(&self) -> f64 {
        self.inner.mid_f64()
    }

    /// Radius, rounded **up** to `f64`.
    ///
    /// Nearest-rounding here would report a radius smaller than the true one
    /// and turn a valid enclosure into an invalid one at the Python boundary.
    #[getter]
    fn rad(&self) -> f64 {
        self.inner.rad.to_f64_round(rug::float::Round::Up)
    }

    /// Lower endpoint, rounded **down** to `f64`.
    ///
    /// The ball is computed at `prec` bits — far finer than `f64` — so a
    /// nearest-rounded endpoint can land *inside* the true interval. A caller
    /// writing `lo <= v <= hi` would then get `False` for a value the ball
    /// genuinely encloses, which defeats the entire purpose of rigorous
    /// arithmetic. Rounding outward keeps the `f64` view a valid enclosure of
    /// the exact one, at the cost of being at most one ulp wider.
    #[getter]
    fn lo(&self) -> f64 {
        self.inner.lo().to_f64_round(rug::float::Round::Down)
    }

    /// Upper endpoint, rounded **up** to `f64`. See [`lo`].
    #[getter]
    fn hi(&self) -> f64 {
        self.inner.hi().to_f64_round(rug::float::Round::Up)
    }

    fn contains(&self, v: f64) -> bool {
        self.inner.contains(v)
    }

    /// True if the ball has zero radius (a single exact value).
    #[getter]
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
    let prec = checked_prec(prec.unwrap_or(128))?;
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
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

/// Structured result returned by the unified evaluator (stable top-level).
#[pyclass(name = "EvaluationResult")]
struct PyEvaluationResult {
    value: PyObject,
    status: String,
    backend: String,
    requested_mode: String,
    requested_precision_bits: Option<u32>,
    achieved_precision_bits: Option<u32>,
    enclosure: Option<Py<PyArbBall>>,
    reason: Option<String>,
}

#[pymethods]
impl PyEvaluationResult {
    #[getter]
    fn value(&self, py: Python<'_>) -> PyObject {
        self.value.clone_ref(py)
    }
    #[getter]
    fn status(&self) -> &str {
        &self.status
    }
    #[getter]
    fn backend(&self) -> &str {
        &self.backend
    }
    #[getter]
    fn requested_mode(&self) -> &str {
        &self.requested_mode
    }
    #[getter]
    fn requested_precision_bits(&self) -> Option<u32> {
        self.requested_precision_bits
    }
    #[getter]
    fn achieved_precision_bits(&self) -> Option<u32> {
        self.achieved_precision_bits
    }
    #[getter]
    fn enclosure(&self, py: Python<'_>) -> Option<Py<PyArbBall>> {
        self.enclosure.as_ref().map(|ball| ball.clone_ref(py))
    }
    #[getter]
    fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
    #[getter]
    fn is_enclosure(&self) -> bool {
        self.enclosure.is_some()
    }
}

/// The Python type name of `value`, for error messages. Never fails: a type
/// whose `__name__` cannot be read is reported as `<unknown>` rather than
/// replacing the caller's real error with a second one.
fn py_type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

/// `str(value.<attr>)`, or `None` if the attribute is missing or unreadable.
fn attr_as_string(value: &Bound<'_, PyAny>, attr: &str) -> Option<String> {
    value.getattr(attr).ok()?.str().ok()?.extract().ok()
}

fn exact_binding(value: &Bound<'_, PyAny>) -> PyResult<Rational> {
    if let Ok(integer) = value.extract::<i64>() {
        return Ok(Rational::from(integer));
    }
    // `value.getattr("numerator")?` used to propagate a bare `AttributeError`
    // for anything that is neither an int nor a `Fraction` — including an
    // `Expr`, which is the obvious thing to pass as `residue(f, z, point)`.
    // That error named this function's implementation rather than the caller's
    // mistake, and `AttributeError` is not an `AlkahestError`, so it escaped
    // `except ak.AlkahestError` entirely. Probe instead of propagate.
    let (numerator, denominator) = match (
        attr_as_string(value, "numerator"),
        attr_as_string(value, "denominator"),
    ) {
        (Some(n), Some(d)) => (n, d),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "exact bindings must be int or fractions.Fraction, got {}",
                py_type_name(value)
            )))
        }
    };
    let numerator = Integer::parse(numerator)
        .map_err(|_| PyTypeError::new_err("exact bindings must be int or fractions.Fraction"))?
        .complete();
    let denominator = Integer::parse(denominator)
        .map_err(|_| PyTypeError::new_err("exact bindings must be int or fractions.Fraction"))?
        .complete();
    if denominator == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "exact binding denominator must be non-zero",
        ));
    }
    Ok(Rational::from((numerator, denominator)))
}

#[pyfunction]
#[pyo3(name = "evaluate", signature = (expr, bindings, *, mode = "auto", precision_bits = None))]
fn py_evaluate(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    bindings: &Bound<'_, PyDict>,
    mode: &str,
    precision_bits: Option<u32>,
) -> PyResult<PyEvaluationResult> {
    if !matches!(mode, "auto" | "exact" | "f64" | "interval" | "complex") {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mode must be 'auto', 'exact', 'f64', 'complex', or 'interval'",
        ));
    }
    if let Some(p) = precision_bits {
        checked_prec(p)?;
    }
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    let wants_interval = mode == "interval"
        || (mode == "auto"
            && (precision_bits.is_some()
                || bindings
                    .iter()
                    .any(|(_, v)| v.is_instance_of::<PyArbBall>())));
    if wants_interval {
        let precision = precision_bits.unwrap_or(128);
        let mut evaluator = CoreIntervalEval::new(precision);
        for (key, value) in bindings.iter() {
            let var: PyRef<PyExpr> = key.extract()?;
            let ball: PyRef<PyArbBall> = value
                .extract()
                .map_err(|_| PyTypeError::new_err("interval bindings must be ArbBall values"))?;
            evaluator.bind(var.id, ball.inner.clone());
        }
        return Ok(match core_eval_interval(expr.id, &pool.inner, &evaluator) {
            Ok(ball) => {
                let py_ball = Py::new(py, PyArbBall { inner: ball })?;
                PyEvaluationResult {
                    value: py_ball.clone_ref(py).into_py(py),
                    status: "ok".into(),
                    backend: "mpfr_ball".into(),
                    requested_mode: mode.into(),
                    requested_precision_bits: precision_bits,
                    achieved_precision_bits: Some(precision),
                    enclosure: Some(py_ball),
                    reason: None,
                }
            }
            Err(error) => PyEvaluationResult {
                value: py.None(),
                status: "unsupported".into(),
                backend: "none".into(),
                requested_mode: mode.into(),
                requested_precision_bits: precision_bits,
                achieved_precision_bits: None,
                enclosure: None,
                reason: Some(error.reason.code().to_owned()),
            },
        });
    }

    let wants_complex = mode == "complex"
        || (mode == "auto"
            && bindings
                .iter()
                .any(|(_, v)| try_strict_complex_binding(&v).is_some()));
    if wants_complex {
        let mut env = std::collections::HashMap::new();
        for (key, value) in bindings.iter() {
            let var: PyRef<PyExpr> = key.extract()?;
            env.insert(
                var.id,
                try_complex_binding(&value).ok_or_else(|| {
                    PyTypeError::new_err(
                        "complex bindings must be complex numbers (or real scalars in mode='complex')",
                    )
                })?,
            );
        }
        return Ok(match core_eval_complex_f64(expr.id, &pool.inner, &env) {
            Ok(value) => {
                let pc = PyComplex::from_doubles_bound(py, value.re, value.im);
                PyEvaluationResult {
                    value: pc.into_py(py),
                    status: "ok".into(),
                    backend: "interpreter_complex_f64".into(),
                    requested_mode: mode.into(),
                    requested_precision_bits: precision_bits,
                    achieved_precision_bits: Some(53),
                    enclosure: None,
                    reason: None,
                }
            }
            Err(error) => PyEvaluationResult {
                value: py.None(),
                status: "unsupported".into(),
                backend: "none".into(),
                requested_mode: mode.into(),
                requested_precision_bits: precision_bits,
                achieved_precision_bits: None,
                enclosure: None,
                reason: Some(error.reason.agent_code().to_owned()),
            },
        });
    }
    let mut exact = std::collections::HashMap::new();
    let mut exact_possible = true;
    for (key, value) in bindings.iter() {
        let var: PyRef<PyExpr> = key.extract()?;
        match exact_binding(&value) {
            Ok(v) => {
                exact.insert(var.id, v);
            }
            Err(_) => {
                exact_possible = false;
                break;
            }
        }
    }
    if mode == "exact" || (mode == "auto" && exact_possible) {
        return Ok(
            match core_eval_exact_rational(expr.id, &pool.inner, &exact) {
                Ok(value) => {
                    let fraction = py
                        .import_bound("fractions")?
                        .getattr("Fraction")?
                        .call1((format!("{}/{}", value.numer(), value.denom()),))?;
                    PyEvaluationResult {
                        value: fraction.into_py(py),
                        status: "ok".into(),
                        backend: "exact_rational".into(),
                        requested_mode: mode.into(),
                        requested_precision_bits: precision_bits,
                        achieved_precision_bits: None,
                        enclosure: None,
                        reason: None,
                    }
                }
                Err(error) => PyEvaluationResult {
                    value: py.None(),
                    status: "unsupported".into(),
                    backend: "none".into(),
                    requested_mode: mode.into(),
                    requested_precision_bits: precision_bits,
                    achieved_precision_bits: None,
                    enclosure: None,
                    reason: Some(error.reason.code().to_owned()),
                },
            },
        );
    }
    let mut env = std::collections::HashMap::new();
    for (key, value) in bindings.iter() {
        let var: PyRef<PyExpr> = key.extract()?;
        env.insert(var.id, value.extract::<f64>()?);
    }
    Ok(match core_eval_f64(expr.id, &pool.inner, &env) {
        Ok(value) => PyEvaluationResult {
            value: value.into_py(py),
            status: "ok".into(),
            backend: "interpreter_f64".into(),
            requested_mode: mode.into(),
            requested_precision_bits: precision_bits,
            achieved_precision_bits: Some(53),
            enclosure: None,
            reason: None,
        },
        Err(error) => PyEvaluationResult {
            value: py.None(),
            status: "unsupported".into(),
            backend: "none".into(),
            requested_mode: mode.into(),
            requested_precision_bits: precision_bits,
            achieved_precision_bits: None,
            enclosure: None,
            reason: Some(error.reason.code().to_owned()),
        },
    })
}

// ---------------------------------------------------------------------------
// Phase 23 — Parallel simplification
// ---------------------------------------------------------------------------

/// Simplify an expression using a recursive fork-join traversal.
///
/// Requires the ``parallel`` feature at build time; without it this falls back
/// to sequential :func:`simplify`, so the call is always available. Check
/// ``alkahest.capabilities()["features"]["parallel"]`` to tell which you have.
///
/// Fork-join keeps each subtree on one worker, which suits **wide** expressions
/// — a large sum or product of independent terms. For deep, narrow expressions
/// prefer :func:`simplify_redex`, or let :func:`simplify_auto` choose.
///
/// The result matches :func:`simplify` exactly. The derivation log may vary in
/// order between runs, because two workers can reach the same node
/// concurrently; :func:`simplify_redex` does not have that property.
#[pyfunction]
#[pyo3(name = "simplify_par")]
fn py_simplify_par(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let pool_ref = expr.pool.borrow(py);
    guard_depth(&pool_ref.inner, expr.id)?;
    // Bind out of the `PyRef` first: it carries a `Python` marker and so is
    // not `Sync`, but the pool and id themselves are safe to send.
    #[cfg(feature = "parallel")]
    let result = {
        let (id, pool) = (expr.id, &pool_ref.inner);
        py.allow_threads(|| alkahest_core::simplify_par(id, pool))
    };
    #[cfg(not(feature = "parallel"))]
    let result = alkahest_core::simplify(expr.id, &pool_ref.inner);
    drop(pool_ref);
    Ok(make_derived_result(
        py,
        result,
        expr.pool.clone_ref(py),
        None,
    ))
}

/// Simplify an expression by scheduling independent rewrites level by level.
///
/// Requires the ``parallel`` feature at build time; without it this falls back
/// to sequential :func:`simplify`, so the call is always available.
///
/// Nodes are bucketed by height, so every node at a given height is rewritten
/// concurrently regardless of its type. That suits **deep** expressions, where
/// :func:`simplify_par` finds no wide node to fork on and ends up running
/// essentially sequentially. It also visits each node exactly once, so unlike
/// :func:`simplify_par` the derivation log is **deterministic** — identical
/// across runs and across CPU counts.
///
/// The result matches :func:`simplify` exactly.
#[pyfunction]
#[pyo3(name = "simplify_redex")]
fn py_simplify_redex(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let pool_ref = expr.pool.borrow(py);
    guard_depth(&pool_ref.inner, expr.id)?;
    // Bind out of the `PyRef` first: it carries a `Python` marker and so is
    // not `Sync`, but the pool and id themselves are safe to send.
    #[cfg(feature = "parallel")]
    let result = {
        let (id, pool) = (expr.id, &pool_ref.inner);
        py.allow_threads(|| alkahest_core::simplify_redex(id, pool))
    };
    #[cfg(not(feature = "parallel"))]
    let result = alkahest_core::simplify(expr.id, &pool_ref.inner);
    drop(pool_ref);
    Ok(make_derived_result(
        py,
        result,
        expr.pool.clone_ref(py),
        None,
    ))
}

/// Simplify in parallel, choosing the strategy from the expression's shape.
///
/// Requires the ``parallel`` feature at build time; without it this falls back
/// to sequential :func:`simplify`, so the call is always available.
///
/// Dispatches to :func:`simplify_par` for wide expressions when enough CPU
/// cores are available, and to :func:`simplify_redex` otherwise. Use
/// :func:`simplify_strategy` to see which it would pick without running it.
///
/// This is the one to reach for if you do not want to think about shape. The
/// result matches :func:`simplify` exactly.
#[pyfunction]
#[pyo3(name = "simplify_auto")]
fn py_simplify_auto(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    let pool_ref = expr.pool.borrow(py);
    guard_depth(&pool_ref.inner, expr.id)?;
    // Bind out of the `PyRef` first: it carries a `Python` marker and so is
    // not `Sync`, but the pool and id themselves are safe to send.
    #[cfg(feature = "parallel")]
    let result = {
        let (id, pool) = (expr.id, &pool_ref.inner);
        py.allow_threads(|| alkahest_core::simplify_auto(id, pool))
    };
    #[cfg(not(feature = "parallel"))]
    let result = alkahest_core::simplify(expr.id, &pool_ref.inner);
    drop(pool_ref);
    Ok(make_derived_result(
        py,
        result,
        expr.pool.clone_ref(py),
        None,
    ))
}

/// Report which strategy :func:`simplify_auto` would use for ``expr``.
///
/// Returns ``"fork_join"`` (:func:`simplify_par`), ``"level_scheduled"``
/// (:func:`simplify_redex`), or ``"sequential"`` when the extension was built
/// without the ``parallel`` feature and both parallel entry points fall back to
/// :func:`simplify`.
///
/// The answer depends on the number of worker threads available as well as the
/// expression, so it can differ between machines.
#[pyfunction]
#[pyo3(name = "simplify_strategy")]
fn py_simplify_strategy(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<String> {
    #[cfg(feature = "parallel")]
    {
        let pool_ref = expr.pool.borrow(py);
        guard_depth(&pool_ref.inner, expr.id)?;
        let strategy = alkahest_core::choose_strategy(expr.id, &pool_ref.inner);
        Ok(match strategy {
            alkahest_core::Strategy::ForkJoin => "fork_join".to_string(),
            alkahest_core::Strategy::LevelScheduled => "level_scheduled".to_string(),
        })
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (py, expr);
        Ok("sequential".to_string())
    }
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
        guard_depth(&pool.inner, expr.id)?;
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
/// var : Expr or list[Expr]
///     The polynomial variable. `emit_c` only supports univariate
///     polynomials, so a list/tuple is accepted as a convenience but must
///     contain exactly one `Expr` (e.g. `[x]` is equivalent to `x`).
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
    var: &Bound<'_, PyAny>,
    var_name: &str,
    fn_name: &str,
) -> PyResult<String> {
    let var_id = extract_univariate_var(var)?;
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    core_emit_horner_c(expr.id, var_id, var_name, fn_name, &pool.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Extract a single `Expr`'s id from `var`, which may be an `Expr` directly
/// or a one-element `list`/`tuple` containing an `Expr` (a common but
/// incorrect guess for APIs that expect a single variable).
fn extract_univariate_var(var: &Bound<'_, PyAny>) -> PyResult<ExprId> {
    if let Ok(e) = var.extract::<PyRef<PyExpr>>() {
        return Ok(e.id);
    }
    if let Ok(seq) = var.extract::<Vec<PyRef<PyExpr>>>() {
        return match seq.len() {
            1 => Ok(seq[0].id),
            n => Err(PyTypeError::new_err(format!(
                "var must be a single Expr (or a one-element list/tuple); \
                 got a sequence of length {n}. emit_c only supports univariate \
                 polynomials."
            ))),
        };
    }
    Err(PyTypeError::new_err(
        "var must be an Expr (the polynomial variable), or a one-element \
         list/tuple containing one, e.g. emit_c(expr, x) or emit_c(expr, [x])",
    ))
}

// ---------------------------------------------------------------------------
// Transcendental C emission — general expression DAG walker
// ---------------------------------------------------------------------------

/// Emit a C function that evaluates a symbolic expression including
/// transcendental functions.
///
/// Unlike :func:`emit_c` (which is restricted to univariate polynomials and
/// uses Horner form), ``emit_c_expr`` supports arbitrary expressions including
/// ``sin``, ``cos``, ``exp``, ``log``, ``sqrt``, ``tan``, ``atan2``, ``erf``,
/// ``floor``, ``ceil``, and more.  The emitted code requires ``#include <math.h>``.
///
/// Parameters
/// ----------
/// expr : Expr
///     The symbolic expression to compile.
/// vars : Expr | list[Expr]
///     The symbolic variables (in argument order).  A single ``Expr`` is treated
///     as a one-element list.
/// var_names : str | list[str], optional
///     C parameter names for each variable.  If omitted, defaults to the
///     symbolic name of each variable.  Must have the same length as *vars*.
/// fn_name : str, optional
///     The C function name (default ``"f"``).
///
/// Returns
/// -------
/// str
///     A complete C function definition.
///
/// Raises
/// ------
/// ValueError
///     If the expression calls a function with no ``<math.h>`` equivalent
///     (e.g. ``diracdelta``, elliptic integrals) or references a symbol not
///     listed in *vars*.
///
/// Examples
/// --------
/// >>> from alkahest import pool, sin, emit_c_expr
/// >>> p = pool(); x = p.symbol("x")
/// >>> print(emit_c_expr(sin(x) + x**2, x))
/// double f(double x) {
///     return (sin(x) + (x * x));
/// }
#[pyfunction]
#[pyo3(name = "emit_c_expr")]
#[pyo3(signature = (expr, vars, var_names=None, fn_name="f"))]
fn py_emit_c_expr(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    vars: &Bound<'_, PyAny>,
    var_names: Option<&Bound<'_, PyAny>>,
    fn_name: &str,
) -> PyResult<String> {
    // Collect variable ExprIds.
    let var_ids: Vec<ExprId> = if let Ok(e) = vars.extract::<PyRef<PyExpr>>() {
        vec![e.id]
    } else if let Ok(seq) = vars.extract::<Vec<PyRef<PyExpr>>>() {
        seq.iter().map(|e| e.id).collect()
    } else {
        return Err(PyTypeError::new_err(
            "vars must be an Expr or a list of Expr",
        ));
    };

    // Collect (or derive) C parameter names.
    let pool_guard = expr.pool.borrow(py);
    guard_depth(&pool_guard.inner, expr.id)?;
    let c_names: Vec<String> = if let Some(names_obj) = var_names {
        if let Ok(s) = names_obj.extract::<String>() {
            vec![s]
        } else if let Ok(seq) = names_obj.extract::<Vec<String>>() {
            seq
        } else {
            return Err(PyTypeError::new_err(
                "var_names must be a str or list of str",
            ));
        }
    } else {
        // Default: use the symbolic name of each variable.
        var_ids
            .iter()
            .map(|&id| match pool_guard.inner.get(id) {
                alkahest_core::kernel::ExprData::Symbol { name, .. } => name,
                _ => "v".to_string(),
            })
            .collect()
    };

    if var_ids.len() != c_names.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "vars and var_names must have the same length (got {} and {})",
            var_ids.len(),
            c_names.len()
        )));
    }

    let name_refs: Vec<&str> = c_names.iter().map(String::as_str).collect();
    core_emit_expr_c(expr.id, &var_ids, &name_refs, fn_name, &pool_guard.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Emit a C function that writes multiple symbolic expressions into an output
/// array.
///
/// Computes ``len(exprs)`` values and writes them to ``double *out``:
///
/// .. code-block:: c
///
///    void fn_name(double x, double y, …, double *out);
///    /* out[0] = exprs[0], out[1] = exprs[1], … */
///
/// All expressions share the same input variables and the same ``<math.h>``
/// support as :func:`emit_c_expr`.
///
/// Parameters
/// ----------
/// exprs : list[Expr]
///     The symbolic expressions to evaluate (one per output component).
/// vars : Expr | list[Expr]
///     The symbolic variables (in argument order).
/// var_names : str | list[str], optional
///     C parameter names for each variable.  Defaults to the symbolic names.
/// fn_name : str, optional
///     The C function name (default ``"eval_vec"``).
///
/// Returns
/// -------
/// str
///     A complete C function definition with a ``void`` return type.
///
/// Raises
/// ------
/// ValueError
///     Same conditions as :func:`emit_c_expr`.
///
/// Examples
/// --------
/// >>> from alkahest import pool, sin, cos, emit_c_vec
/// >>> p = pool(); x = p.symbol("x")
/// >>> print(emit_c_vec([sin(x), cos(x)], x))
/// void eval_vec(double x, double *out) {
///     out[0] = sin(x);
///     out[1] = cos(x);
/// }
#[pyfunction]
#[pyo3(name = "emit_c_vec")]
#[pyo3(signature = (exprs, vars, var_names=None, fn_name="eval_vec"))]
fn py_emit_c_vec(
    py: Python<'_>,
    exprs: Vec<PyRef<PyExpr>>,
    vars: &Bound<'_, PyAny>,
    var_names: Option<&Bound<'_, PyAny>>,
    fn_name: &str,
) -> PyResult<String> {
    if exprs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "emit_c_vec requires at least one expression",
        ));
    }

    // All exprs must share the same pool; use the first one.
    let pool_guard = exprs[0].pool.borrow(py);
    let expr_ids: Vec<ExprId> = exprs.iter().map(|e| e.id).collect();
    alkahest_core::check_expr_depths(&pool_guard.inner, &expr_ids).map_err(depth_error_to_py)?;

    // Collect variable ExprIds.
    let var_ids: Vec<ExprId> = if let Ok(e) = vars.extract::<PyRef<PyExpr>>() {
        vec![e.id]
    } else if let Ok(seq) = vars.extract::<Vec<PyRef<PyExpr>>>() {
        seq.iter().map(|e| e.id).collect()
    } else {
        return Err(PyTypeError::new_err(
            "vars must be an Expr or a list of Expr",
        ));
    };

    // Collect (or derive) C parameter names.
    let c_names: Vec<String> = if let Some(names_obj) = var_names {
        if let Ok(s) = names_obj.extract::<String>() {
            vec![s]
        } else if let Ok(seq) = names_obj.extract::<Vec<String>>() {
            seq
        } else {
            return Err(PyTypeError::new_err(
                "var_names must be a str or list of str",
            ));
        }
    } else {
        var_ids
            .iter()
            .map(|&id| match pool_guard.inner.get(id) {
                alkahest_core::kernel::ExprData::Symbol { name, .. } => name,
                _ => "v".to_string(),
            })
            .collect()
    };

    if var_ids.len() != c_names.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "vars and var_names must have the same length (got {} and {})",
            var_ids.len(),
            c_names.len()
        )));
    }

    let name_refs: Vec<&str> = c_names.iter().map(String::as_str).collect();
    core_emit_expr_c_vec(&expr_ids, &var_ids, &name_refs, fn_name, &pool_guard.inner)
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
fn py_collect_like_terms(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    use alkahest_core::{rules_for_config, simplify_with};
    let pool_py = expr.pool.clone_ref(py);
    let derived = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let rules = rules_for_config(&SimplifyConfig::default());
        simplify_with(expr.id, &pool.inner, &rules, SimplifyConfig::default())
    };
    Ok(make_derived_result(py, derived, pool_py, None))
}

// ---------------------------------------------------------------------------
// V3-2 — Non-commutative Pauli / Clifford simplification helpers
// ---------------------------------------------------------------------------

/// Simplify with default arithmetic rules plus the Pauli product table on ``sx``, ``sy``, ``sz``.
#[pyfunction]
#[pyo3(name = "simplify_pauli")]
fn py_simplify_pauli(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyDerivedResult> {
    use alkahest_core::algebra::noncommutative::pauli_product_rules;
    use alkahest_core::{rules_for_config, simplify_with};
    let pool_py = expr.pool.clone_ref(py);
    let derived = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let mut rules = rules_for_config(&SimplifyConfig::default());
        rules.extend(pauli_product_rules());
        simplify_with(expr.id, &pool.inner, &rules, SimplifyConfig::default())
    };
    Ok(make_derived_result(py, derived, pool_py, None))
}

/// Simplify with default rules plus orthogonal Clifford anticommutation on ``cliff_e1``, ``cliff_e2``.
#[pyfunction]
#[pyo3(name = "simplify_clifford_orthogonal")]
fn py_simplify_clifford_orthogonal(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
) -> PyResult<PyDerivedResult> {
    use alkahest_core::algebra::noncommutative::clifford_orthogonal_rules;
    use alkahest_core::{rules_for_config, simplify_with};
    let pool_py = expr.pool.clone_ref(py);
    let derived = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let mut rules = rules_for_config(&SimplifyConfig::default());
        rules.extend(clifford_orthogonal_rules());
        simplify_with(expr.id, &pool.inner, &rules, SimplifyConfig::default())
    };
    Ok(make_derived_result(py, derived, pool_py, None))
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
        guard_depth(&pool.inner, expr.id)?;
        core_poly_normal(expr.id, var_ids, &pool.inner).map_err(conv_error_to_py)?
    };
    Ok(PyExpr {
        id: result,
        pool: pool_py,
    })
}

/// Cancel common polynomial factors in a rational expression.
///
/// Combines an expression built from ``+``, ``-``, ``*``, ``/`` and integer
/// powers of polynomials in *vars* over a common denominator, then divides the
/// numerator and denominator by their polynomial GCD.  Any sub-expression that
/// is not a polynomial in *vars* (a function call like ``sin(x)``, a symbol not
/// listed in *vars*, or a base with a symbolic exponent) is treated as an
/// opaque generator.
///
/// If *vars* is omitted, free symbols of *expr* are used.
///
/// Examples::
///
///     cancel((x**2 - 1) / (x - 1), [x])   # -> x + 1
///     cancel((x**2 - 1) / (x - 1))        # vars inferred
///     cancel(1/x + 1/(x + 1), [x])         # -> (2*x + 1) / (x**2 + x)
///     cancel(x / x, [x])                   # -> 1
///
/// Limitations: generators are matched structurally (``sin(x)`` and
/// ``sin(2*x/2)`` are distinct), and bases raised to symbolic exponents are
/// opaque as a whole.
#[pyfunction]
#[pyo3(name = "cancel", signature = (expr, vars=None))]
fn py_cancel(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    vars: Option<Vec<PyRef<PyExpr>>>,
) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let result = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let var_ids: Vec<ExprId> = match vars {
            Some(v) => v.iter().map(|v| v.id).collect(),
            None => alkahest_core::collect_free_vars(expr.id, &pool.inner),
        };
        core_cancel(expr.id, var_ids, &pool.inner).map_err(conv_error_to_py)?
    };
    Ok(PyExpr {
        id: result,
        pool: pool_py,
    })
}

/// Combine a rational expression over a single common denominator.
///
/// Behaves like :func:`cancel` (the numerator/denominator GCD is divided out by
/// the underlying rational-function constructor); provided as a companion name
/// for callers that want the explicit "put over a common denominator" intent.
///
/// If *vars* is omitted, free symbols of *expr* are used.
///
/// Example::
///
///     together(1/x + 1/(x + 1), [x])   # -> (2*x + 1) / (x**2 + x)
#[pyfunction]
#[pyo3(name = "together", signature = (expr, vars=None))]
fn py_together(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    vars: Option<Vec<PyRef<PyExpr>>>,
) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let result = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let var_ids: Vec<ExprId> = match vars {
            Some(v) => v.iter().map(|v| v.id).collect(),
            None => alkahest_core::collect_free_vars(expr.id, &pool.inner),
        };
        core_together(expr.id, var_ids, &pool.inner).map_err(conv_error_to_py)?
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
        alkahest_core::check_expr_depths(&pool.inner, &[p.id, q.id]).map_err(depth_error_to_py)?;
        core_resultant(p.id, q.id, var.id, &pool.inner).map_err(resultant_error_to_py)?
    };
    Ok(make_derived_result(py, derived, pool_py, None))
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
        alkahest_core::check_expr_depths(&pool.inner, &[p.id, q.id]).map_err(depth_error_to_py)?;
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
    guard_depth(&pool.inner, poly.id)?;
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
    guard_depth(&pool.inner, poly.id)?;
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
    // The oracle is arbitrary user code, so it can raise anything — and the
    // core signature is infallible, so an `.expect()` here turned every one of
    // those exceptions into a `PanicException` (a `BaseException`, which a
    // caller's `except Exception` does not catch).  Park the first error and
    // re-raise it after the algorithm returns.
    let oracle_err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
    let rust_eval = |x: u64| -> u64 {
        if oracle_err.borrow().is_some() {
            return 0;
        }
        match eval.call1((x,)).and_then(|r| r.extract::<u64>()) {
            Ok(v) => v,
            Err(e) => {
                *oracle_err.borrow_mut() = Some(e);
                0
            }
        }
    };
    let terms = core_sparse_interpolate_univariate(&rust_eval, term_bound, prime);
    if let Some(e) = oracle_err.into_inner() {
        return Err(e);
    }
    let terms = terms.map_err(sparse_interp_error_to_py)?;
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

    // See `sparse_interp_univariate`: a raising oracle must not become a
    // `PanicException`.
    let oracle_err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
    let rust_eval = |pt: &[u64]| -> u64 {
        if oracle_err.borrow().is_some() {
            return 0;
        }
        let py_list = pyo3::types::PyList::new_bound(py, pt.iter().copied());
        match eval.call1((py_list,)).and_then(|r| r.extract::<u64>()) {
            Ok(v) => v,
            Err(e) => {
                *oracle_err.borrow_mut() = Some(e);
                0
            }
        }
    };

    let fp = core_sparse_interpolate(&rust_eval, var_ids, term_bound, degree_bound, prime, seed);
    if let Some(e) = oracle_err.into_inner() {
        return Err(e);
    }
    let fp = fp.map_err(sparse_interp_error_to_py)?;
    Ok(PyMultiPolyFp {
        inner: fp,
        pool: None,
    })
}

/// Compute the primitive GCD of two multivariate polynomials over ℤ using
/// sparse interpolation and the Chinese Remainder Theorem (Zippel method).
///
/// Both polynomials must have the same variable list.  The result is the
/// primitive part of ``gcd(f, g)`` with positive leading coefficient.
///
/// Parameters
/// ----------
/// f, g : MultiPoly
///     Input polynomials with the same variable list.
/// term_bound : int
///     Upper bound on the number of nonzero terms in the GCD.
/// degree_bound : int
///     Upper bound on the per-variable degree of the GCD in ``x₂,…,xₙ``.
///     The degree in ``x₁`` is probed automatically.
/// seed : int, optional
///     PRNG seed.  Change on failure (default 0).
///
/// Returns
/// -------
/// MultiPoly
///     The primitive GCD with positive leading coefficient.
///
/// Raises
/// ------
/// SparseGcdError
///     If the interpolation fails or the variable lists are incompatible.
///
/// Examples
/// --------
/// ::
///
///     pool = ExprPool()
///     x, y = pool.symbol("x"), pool.symbol("y")
///     f = MultiPoly.from_symbolic((x - pool.integer(1)) * (x + y), [x, y])
///     g = MultiPoly.from_symbolic((x + pool.integer(1)) * (x + y), [x, y])
///     h = gcd_sparse(f, g, term_bound=4, degree_bound=3)
///     # h represents x + y
#[pyfunction]
#[pyo3(name = "gcd_sparse")]
#[pyo3(signature = (f, g, term_bound, degree_bound, seed=0))]
fn py_gcd_sparse(
    f: PyRef<PyMultiPoly>,
    g: PyRef<PyMultiPoly>,
    term_bound: usize,
    degree_bound: u32,
    seed: u64,
) -> PyResult<PyMultiPoly> {
    let result = core_gcd_sparse_modular(&f.inner, &g.inner, term_bound, degree_bound, seed)
        .map_err(sparse_gcd_error_to_py)?;
    Ok(PyMultiPoly {
        inner: result,
        pool: merge_mp_pool(&f.pool, &g.pool),
    })
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
    guard_depth(&pool.inner, formula.id)?;
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
    guard_depth(inner, formula.id)?;
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

// ---------------------------------------------------------------------------
// P1 item 9 — rigorous global bounds (Taylor models / validated numerics)
// ---------------------------------------------------------------------------

fn validated_error_to_py(e: CoreValidatedError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyValidatedError>();
        make_structured_err(py, &exc_type, &e)
    })
}

// ---------------------------------------------------------------------------
// P1 item 8 — positivity certificates (SOS / Positivstellensatz)
// ---------------------------------------------------------------------------

fn sos_error_to_py(e: CoreSosError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PySosError>();
        make_structured_err(py, &exc_type, &e)
    })
}

fn verdict_str(v: CoreVerdict) -> &'static str {
    match v {
        CoreVerdict::True => "true",
        CoreVerdict::False => "false",
        CoreVerdict::Undecided => "undecided",
    }
}

/// A **rigorous** enclosure: the true value is guaranteed to lie in
/// ``[lower, upper]``.
///
/// Returned by :func:`alkahest.bound_on_box` and
/// :func:`alkahest.verified_integral`. The bound may be *wide*, but it is
/// never *wrong* — that is the whole contract. Check
/// :attr:`budget_exhausted` to see whether it is as tight as you asked for.
#[pyclass(name = "Enclosure")]
struct PyEnclosure {
    #[pyo3(get)]
    lower: f64,
    #[pyo3(get)]
    upper: f64,
    #[pyo3(get)]
    budget_exhausted: bool,
    #[pyo3(get)]
    subdivisions: usize,
}

#[pymethods]
impl PyEnclosure {
    /// Width of the enclosure.
    #[getter]
    fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// True if `v` is inside the rigorous enclosure.
    fn contains(&self, v: f64) -> bool {
        self.lower <= v && v <= self.upper
    }

    fn __repr__(&self) -> String {
        format!(
            "Enclosure([{}, {}], budget_exhausted={}, subdivisions={})",
            self.lower, self.upper, self.budget_exhausted, self.subdivisions
        )
    }
}

/// An exact, re-checkable proof that a polynomial is non-negative.
///
/// Returned by :func:`alkahest.sos_decompose` and :func:`alkahest.prove_nonneg`.
/// The identity it carries has already been re-expanded and compared against
/// the target exactly; :meth:`verify` re-runs that check on demand, so a
/// downstream consumer never has to trust the search that produced it.
#[pyclass(name = "PositivityCertificate")]
struct PyPositivityCertificate {
    inner: CorePositivityCertificate,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyPositivityCertificate {
    /// ``"sos"``, ``"handelman"`` or ``"putinar"``.
    #[getter]
    fn kind(&self) -> String {
        self.inner.kind.as_str().to_string()
    }

    /// The basis degree (unconstrained) or Handelman level actually used.
    #[getter]
    fn degree(&self) -> u32 {
        self.inner.degree
    }

    /// Number of squares in the decomposition.
    #[getter]
    fn num_squares(&self) -> usize {
        self.inner.num_squares()
    }

    /// The certificate identity, e.g. ``p = 1*(x - y)^2 + 2*(y)^2``.
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity_string()
    }

    /// The claim being certified, e.g. ``p >= 0 on {g_0 >= 0, g_1 >= 0}``.
    #[getter]
    fn claim(&self) -> String {
        self.inner.claim_string()
    }

    /// How the search proceeded — the audit trail.
    #[getter]
    fn log(&self) -> Vec<String> {
        self.inner.log.clone()
    }

    /// The right-hand side of the identity as an :class:`Expr`.
    #[getter]
    fn expression(&self, py: Python<'_>) -> PyExpr {
        let id = {
            let pool = self.pool.borrow(py);
            self.inner.to_expr(&pool.inner)
        };
        PyExpr {
            id,
            pool: self.pool.clone_ref(py),
        }
    }

    /// Re-run the exact verification. Always ``True`` for a certificate that
    /// was returned (the search refuses rather than returning an unverified
    /// one); exposed so a caller can check independently.
    fn verify(&self) -> PyResult<bool> {
        match self.inner.verify() {
            Ok(()) => Ok(true),
            Err(why) => Err(sos_error_to_py(CoreSosError::VerificationFailed(why))),
        }
    }

    /// Lean 4 rendering of the identity, or ``None`` when it cannot be emitted
    /// soundly.
    fn to_lean(&self) -> Option<String> {
        self.inner.to_lean()
    }

    fn __repr__(&self) -> String {
        format!(
            "PositivityCertificate(kind={}, degree={}, squares={})",
            self.inner.kind.as_str(),
            self.inner.degree,
            self.inner.num_squares()
        )
    }
}

fn parse_box(
    py: Python<'_>,
    boxes: Vec<(PyRef<PyExpr>, f64, f64)>,
) -> (Py<PyExprPool>, Vec<(ExprId, f64, f64)>) {
    let pool = boxes[0].0.pool.clone_ref(py);
    let out = boxes.iter().map(|(v, lo, hi)| (v.id, *lo, *hi)).collect();
    (pool, out)
}

/// Whether the validated-bounds subsystem can bound an expression at all.
///
/// Returned by :func:`alkahest.bounds_supported`. Truthy exactly when every
/// construct in the expression has a rigorous Taylor-model rule, so it drops
/// into ``if ak.bounds_supported(f):``.
///
/// Attributes
/// ----------
/// supported : bool
///     The verdict. ``bool(self)`` is the same value.
/// blocker : str or None
///     The evaluator's own description of the **first** construct it has no
///     rule for (``"function `bessel_j0`"``), or ``None`` when supported.
/// functions : list of str
///     Every function in the expression with no Taylor-model rule, sorted.
///     Empty when ``supported`` is ``True`` — and possibly empty when it is
///     ``False``, if the blocker is a node kind rather than a function.
/// detail : str
///     One-sentence human explanation.
#[pyclass(name = "BoundsSupport", frozen)]
struct PyBoundsSupport {
    #[pyo3(get)]
    supported: bool,
    #[pyo3(get)]
    blocker: Option<String>,
    #[pyo3(get)]
    functions: Vec<String>,
    #[pyo3(get)]
    detail: String,
}

#[pymethods]
impl PyBoundsSupport {
    fn __bool__(&self) -> bool {
        self.supported
    }

    /// The verdict as a plain dict, for logging and JSON.
    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("supported", self.supported)?;
        d.set_item("blocker", self.blocker.clone())?;
        d.set_item("functions", self.functions.clone())?;
        d.set_item("detail", &self.detail)?;
        Ok(d.into_py(py))
    }

    fn __repr__(&self) -> String {
        match &self.blocker {
            None => "BoundsSupport(supported=True)".to_string(),
            Some(what) => format!("BoundsSupport(supported=False, blocker={what:?})"),
        }
    }
}

/// `alkahest.bounds_supported(expr) -> BoundsSupport`
///
/// Can the validated-bounds subsystem bound this expression at all?
///
/// :func:`bound_on_box`, :func:`verified_integral`, :func:`verified_no_roots`
/// and :func:`verified_sign` all evaluate through the same Taylor-model
/// evaluator, and it refuses any construct it has no rigorous rule for with
/// ``E-VALIDATED-001``. This answers that question *without running the
/// bound*, so a planning loop can pick a certifiable route instead of
/// discovering the boundary by hitting it.
///
/// The answer is produced by running the real evaluator on a probe box, so it
/// cannot drift from what :func:`bound_on_box` does. The same information per
/// primitive is the ``taylor_model`` flag in
/// ``capabilities()["primitives"]``.
///
/// **What it does not promise.** ``True`` means no ``E-VALIDATED-001``. A
/// supported function can still be refused on a *particular box* for a
/// domain violation (``E-VALIDATED-003``, e.g. ``log`` on ``[-2, -1]``) or a
/// non-finite enclosure (``E-VALIDATED-004``) — those depend on the box, not
/// on the expression, so no box-free predicate can rule them out.
///
/// Note this is a different question from :func:`alkahest.certifiable`, which
/// asks whether an operation emits a **Lean** certificate. A rigorous
/// enclosure is not a Lean proof term, and the validated subsystem is not in
/// the certificate ledger; conflating the two under one predicate would make
/// a ``True`` mean two different kinds of evidence.
///
/// Examples
/// --------
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); x = p.symbol("x")
/// >>> bool(ak.bounds_supported(ak.sin(x) * ak.exp(x)))
/// True
/// >>> answer = ak.bounds_supported(ak.bessel_j0(x))
/// >>> bool(answer), answer.functions
/// (False, ['bessel_j0'])
#[pyfunction]
#[pyo3(name = "bounds_supported")]
fn py_bounds_supported(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyBoundsSupport> {
    guard_expr_depth(py, &expr)?;
    let pool_py = expr.pool.clone_ref(py);
    let pool = pool_py.borrow(py);
    let blocker = alkahest_core::taylor_model_refusal(expr.id, &pool.inner);
    let functions = alkahest_core::taylor_model_blockers(expr.id, &pool.inner);
    let detail = match &blocker {
        None => "every construct has a rigorous Taylor-model rule; the validated-bounds \
                 entry points will not refuse this expression with E-VALIDATED-001 (a \
                 particular box may still hit a domain violation)"
            .to_string(),
        Some(what) => format!(
            "the validated-bounds subsystem has no rigorous Taylor-model rule for \
             {what}; bound_on_box would refuse with E-VALIDATED-001"
        ),
    };
    Ok(PyBoundsSupport {
        supported: blocker.is_none(),
        blocker,
        functions,
        detail,
    })
}

/// `alkahest.bound_on_box(expr, box, *, order=6, prec=128, tol=1e-9, max_subdivisions=2048)`
///
/// Rigorous enclosure of the **range** of `expr` over an axis-aligned box,
/// via Taylor models plus Moore–Skelboe branch-and-bound.
///
/// `box` is a list of `(variable, lo, hi)`. The returned
/// :class:`Enclosure` is a guaranteed outer bound; when the work budget runs
/// out it is returned anyway with ``budget_exhausted=True`` — wide but true.
#[pyfunction]
#[pyo3(name = "bound_on_box", signature = (expr, r#box, *, order = 6, prec = 128, tol = 1e-9, max_subdivisions = 2048))]
fn py_bound_on_box(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    r#box: Vec<(PyRef<PyExpr>, f64, f64)>,
    order: usize,
    prec: u32,
    tol: f64,
    max_subdivisions: usize,
) -> PyResult<PyEnclosure> {
    if r#box.is_empty() {
        return Err(PyValueError::new_err(
            "bound_on_box: the box must constrain at least one variable",
        ));
    }
    let prec = checked_prec(prec)?;
    guard_expr_depth(py, &expr)?;
    let (pool_py, boxes) = parse_box(py, r#box);
    let opts = CoreBoundOptions {
        order,
        prec,
        tol,
        max_subdivisions,
    };
    let pool = pool_py.borrow(py);
    let r =
        core_bound_on_box(expr.id, &pool.inner, &boxes, &opts).map_err(validated_error_to_py)?;
    Ok(PyEnclosure {
        lower: r.lower(),
        upper: r.upper(),
        budget_exhausted: r.budget_exhausted,
        subdivisions: r.subdivisions,
    })
}

/// `alkahest.verified_integral(expr, var, a, b, *, order=8, prec=128, tol=1e-9, max_subdivisions=4096)`
///
/// Rigorous enclosure of ``∫_a^b expr d(var)`` by Taylor-model quadrature.
///
/// Unlike :func:`integrate_definite`, this never needs a closed form — and
/// unlike a floating-point quadrature, the answer is a *theorem*: the true
/// integral is guaranteed to lie in the returned interval. Refuses on
/// singular or improper integrands rather than guessing.
///
/// A **removable** singularity is not a refusal: an integrand written as
/// ``N(x)/D(x)`` with ``N(p) = D(p) = 0`` and ``D'(p) != 0`` — ``log(1+x)/x``
/// on ``[0, 1]``, ``sin(x)/x`` on ``[-1, 1]`` — is enclosed via Cauchy's mean
/// value theorem, and the value returned is the integral of the continuous
/// extension. The two zeros are checked *symbolically*, so a genuine pole is
/// never mistaken for a removable one.
///
/// A singularity that is integrable but not removable (``-log(x)`` on
/// ``[0, 1]``, ``1/sqrt(1-x*x)`` on ``[0, 1]``) is still refused: the integral
/// exists, but no rigorous enclosure of the *integrand* does. The
/// :class:`ValidatedError` message says which of the two situations it is.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "verified_integral", signature = (expr, var, a, b, *, order = 8, prec = 128, tol = 1e-9, max_subdivisions = 4096))]
fn py_verified_integral(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
    a: f64,
    b: f64,
    order: usize,
    prec: u32,
    tol: f64,
    max_subdivisions: usize,
) -> PyResult<PyEnclosure> {
    let prec = checked_prec(prec)?;
    guard_expr_depth(py, &expr)?;
    let pool_py = expr.pool.clone_ref(py);
    let opts = CoreIntegralOptions {
        order,
        prec,
        tol,
        max_subdivisions,
    };
    let pool = pool_py.borrow(py);
    let r = core_verified_integral(expr.id, &pool.inner, var.id, a, b, &opts)
        .map_err(validated_error_to_py)?;
    Ok(PyEnclosure {
        lower: r.lower(),
        upper: r.upper(),
        budget_exhausted: r.budget_exhausted,
        subdivisions: r.subdivisions,
    })
}

/// `alkahest.verified_no_roots(expr, box, ...) -> "true" | "false" | "undecided"`
///
/// Three-valued: ``"true"`` proves `expr` has no root anywhere in the box,
/// ``"false"`` proves it does, and ``"undecided"`` means neither could be
/// established within the budget. The third is never collapsed into the
/// other two.
///
/// ``"false"`` is certified by the intermediate value theorem: the box is
/// first proven free of poles and branch cuts (so `expr` is continuous on it),
/// then the box is subdivided until two points are found at which `expr` is
/// rigorously proven to have opposite signs. A box is convex, so the segment
/// joining them stays inside it and `expr` must vanish somewhere along it.
/// Because the two points need not be the box's own endpoints, an even number
/// of roots no longer defeats the test — ``x*x - 2`` on ``[-2, 2]`` is
/// ``"false"``.
///
/// A root that never produces a sign change — a double root such as
/// ``(x-1)**2`` — stays ``"undecided"``: no witness exists, and none is
/// invented.
#[pyfunction]
#[pyo3(name = "verified_no_roots", signature = (expr, r#box, *, order = 6, prec = 128, tol = 1e-9, max_subdivisions = 2048))]
fn py_verified_no_roots(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    r#box: Vec<(PyRef<PyExpr>, f64, f64)>,
    order: usize,
    prec: u32,
    tol: f64,
    max_subdivisions: usize,
) -> PyResult<String> {
    if r#box.is_empty() {
        return Err(PyValueError::new_err(
            "verified_no_roots: the box must constrain at least one variable",
        ));
    }
    let prec = checked_prec(prec)?;
    guard_expr_depth(py, &expr)?;
    let (pool_py, boxes) = parse_box(py, r#box);
    let opts = CoreBoundOptions {
        order,
        prec,
        tol,
        max_subdivisions,
    };
    let pool = pool_py.borrow(py);
    let v = core_verified_no_roots(expr.id, &pool.inner, &boxes, &opts)
        .map_err(validated_error_to_py)?;
    Ok(verdict_str(v).to_string())
}

/// `alkahest.verified_sign(expr, box, predicate, ...) -> "true" | "false" | "undecided"`
///
/// `predicate` is one of ``"positive"``, ``"negative"``, ``"nonnegative"``,
/// ``"nonpositive"``. A ``"false"`` verdict is itself certified — either the
/// enclosure proves the predicate fails everywhere, or a rigorously evaluated
/// point witnesses the failure.
///
/// An inequality that is **tight at an endpoint** of the box is still decided.
/// Subdivision alone cannot do it — where the margin vanishes, every enclosure
/// of the range straddles zero — so the box is split: a collar at the endpoint
/// is handled by a truncated Taylor expansion with a proven Lagrange remainder,
/// the rest by branch-and-bound. That covers the classical sharp trigonometric
/// inequalities (Cusa–Huygens, Mitrinović–Adamović, Wilker, Huygens, Jordan) on
/// boxes reaching ``x = 0``. Tightness in the *interior* is not covered and
/// stays ``"undecided"``.
///
/// ``tol`` sets the tolerance of the *enclosure*, not of the verdict: it is an
/// absolute width, so it does not bound how close to zero the answer may be.
/// Once the enclosure has been computed, the search is re-run with the sign
/// itself as the stopping rule, which is what decides an inequality whose
/// margin is narrower than ``tol``.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "verified_sign", signature = (expr, r#box, predicate, *, order = 6, prec = 128, tol = 1e-9, max_subdivisions = 2048))]
fn py_verified_sign(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    r#box: Vec<(PyRef<PyExpr>, f64, f64)>,
    predicate: &str,
    order: usize,
    prec: u32,
    tol: f64,
    max_subdivisions: usize,
) -> PyResult<String> {
    if r#box.is_empty() {
        return Err(PyValueError::new_err(
            "verified_sign: the box must constrain at least one variable",
        ));
    }
    let prec = checked_prec(prec)?;
    guard_expr_depth(py, &expr)?;
    let pred = match predicate {
        "positive" => CoreSignPredicate::Positive,
        "negative" => CoreSignPredicate::Negative,
        "nonnegative" => CoreSignPredicate::NonNegative,
        "nonpositive" => CoreSignPredicate::NonPositive,
        other => {
            return Err(PyValueError::new_err(format!(
                "verified_sign: predicate must be one of 'positive', 'negative', \
                 'nonnegative', 'nonpositive', got {other:?}"
            )))
        }
    };
    let (pool_py, boxes) = parse_box(py, r#box);
    let opts = CoreBoundOptions {
        order,
        prec,
        tol,
        max_subdivisions,
    };
    let pool = pool_py.borrow(py);
    let v = core_verified_sign(expr.id, &pool.inner, &boxes, pred, &opts)
        .map_err(validated_error_to_py)?;
    Ok(verdict_str(v).to_string())
}

/// `alkahest.sos_decompose(expr, vars, *, basis_degree=None) -> PositivityCertificate`
///
/// Exact rational sum-of-squares decomposition ``p = Σ_j σ_j·q_j²``.
///
/// Raises :exc:`alkahest.SosError` rather than guessing: ``E-SOS-003`` with a
/// witness point when ``p`` is negative somewhere, ``E-SOS-002`` when no
/// certificate of the searched shape exists at this basis degree (which is
/// *not* a proof that none exists — see the docs).
#[pyfunction]
#[pyo3(name = "sos_decompose", signature = (expr, vars, *, basis_degree = None))]
fn py_sos_decompose(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    vars: Vec<PyRef<PyExpr>>,
    basis_degree: Option<u32>,
) -> PyResult<PyPositivityCertificate> {
    let pool_py = expr.pool.clone_ref(py);
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let opts = CoreSosOpts {
        basis_degree,
        ..Default::default()
    };
    let inner = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_sos_decompose(expr.id, &var_ids, &pool.inner, &opts).map_err(sos_error_to_py)?
    };
    Ok(PyPositivityCertificate {
        inner,
        pool: pool_py,
    })
}

/// `alkahest.prove_nonneg(expr, vars, *, constraints=(), basis_degree=None, level=2)`
///
/// Prove ``p ≥ 0`` on ``{x : g_i(x) ≥ 0}`` with a Handelman-style certificate
/// ``p = Σ_α c_α·Π g_i^{α_i}`` (``c_α ≥ 0``). With no constraints this is
/// :func:`sos_decompose`.
#[pyfunction]
#[pyo3(name = "prove_nonneg", signature = (expr, vars, *, constraints = None, basis_degree = None, level = 2))]
fn py_prove_nonneg(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    vars: Vec<PyRef<PyExpr>>,
    constraints: Option<Vec<PyRef<PyExpr>>>,
    basis_degree: Option<u32>,
    level: u32,
) -> PyResult<PyPositivityCertificate> {
    let pool_py = expr.pool.clone_ref(py);
    guard_expr_depth(py, &expr)?;
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let cons: Vec<ExprId> = constraints
        .map(|cs| cs.iter().map(|c| c.id).collect())
        .unwrap_or_default();
    let opts = CoreSosOpts {
        basis_degree,
        level,
    };
    let inner = {
        let pool = pool_py.borrow(py);
        core_prove_nonneg(expr.id, &cons, &var_ids, &pool.inner, &opts).map_err(sos_error_to_py)?
    };
    Ok(PyPositivityCertificate {
        inner,
        pool: pool_py,
    })
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

/// Symbolic / parametric Routh–Hurwitz stability analysis.
///
/// Given a characteristic polynomial ``poly`` in the analysis variable ``var``
/// whose coefficients may be symbolic expressions in free parameters, returns a
/// dict with:
///
/// - ``"degree"``: the degree of ``poly`` in ``var``;
/// - ``"first_column"``: the Routh-array first-column entries (top to bottom) as
///   ``Expr`` objects in the parameters;
/// - ``"condition"``: the stability condition as a single predicate ``Expr`` — a
///   conjunction of ``entry > 0`` over the non-trivial first-column entries.
///
/// For example ``s**2 + a*s + b`` yields the condition ``a > 0 ∧ b > 0`` and
/// ``s**3 + a*s**2 + b*s + c`` yields ``a > 0 ∧ a*b - c > 0 ∧ c > 0``.
#[pyfunction(name = "routh_hurwitz")]
fn py_routh_hurwitz(py: Python<'_>, poly: PyRef<PyExpr>, var: PyRef<PyExpr>) -> PyResult<PyObject> {
    let pool_py = poly.pool.clone_ref(py);
    if !var.pool.is(&pool_py) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "routh_hurwitz expects poly and var in the same ExprPool",
        ));
    }
    let rh = {
        let bor = pool_py.borrow(py);
        core_routh_hurwitz(poly.id, var.id, &bor.inner).map_err(cad_error_to_py)?
    };
    let first_column: Vec<PyObject> = rh
        .first_column
        .iter()
        .map(|&id| {
            PyExpr {
                id,
                pool: pool_py.clone_ref(py),
            }
            .into_py(py)
        })
        .collect();
    let condition_id = {
        let bor = pool_py.borrow(py);
        rh.condition_expr(&bor.inner)
    };
    let condition = PyExpr {
        id: condition_id,
        pool: pool_py.clone_ref(py),
    }
    .into_py(py);
    let d = PyDict::new_bound(py);
    d.set_item("degree", rh.degree)?;
    d.set_item("first_column", first_column)?;
    d.set_item("condition", condition)?;
    Ok(d.into_py(py))
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
            ("taylor_model", caps.contains(Capabilities::TAYLOR_MODEL)),
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
                        // Not implied by `numeric_ball`: ball arithmetic is
                        // pointwise, a Taylor model needs a rule with a
                        // rigorous remainder. Derived by running the
                        // validated evaluator, so it cannot drift from what
                        // `bound_on_box` actually accepts.
                        (
                            "taylor_model",
                            caps.contains(Capabilities::TAYLOR_MODEL).into_py(py),
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
) -> PyResult<String> {
    let pool_py = expr.pool.clone_ref(py);
    let pool = pool_py.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    let input_ids: Vec<ExprId> = inputs.iter().map(|e| e.id).collect();
    Ok(core_emit_stablehlo(
        expr.id,
        &input_ids,
        fn_name,
        &pool.inner,
    ))
}

// ---------------------------------------------------------------------------
// V5-3 — NVPTX JIT backend
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[pyclass(name = "CudaCompiledFn")]
struct PyCudaCompiledFn {
    inner: CoreCudaCompiledFn,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyCudaCompiledFn {
    #[getter]
    fn ptx(&self) -> &str {
        self.inner.ptx_source()
    }

    #[getter]
    fn n_inputs(&self) -> usize {
        self.inner.n_inputs
    }

    fn __repr__(&self) -> String {
        format!(
            "<CudaCompiledFn n_inputs={} ptx_len={}>",
            self.inner.n_inputs,
            self.inner.ptx.len()
        )
    }

    /// Evaluate the compiled kernel on CUDA device 0 for ``N`` independent points.
    ///
    /// ``inputs`` is a list of length ``n_inputs``.  Each entry is a 1-D sequence
    /// of ``N`` values for that variable (column-major / SoA: one array per
    /// symbolic input).  Returns a Python ``list`` of ``N`` outputs.
    ///
    /// Equivalent to ``call_batch_on(0, inputs)``.
    #[pyo3(name = "call_batch")]
    fn call_batch_py(&self, inputs: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        self.eval_on_device(0, inputs)
    }

    /// Evaluate the compiled kernel on a specific CUDA device ordinal.
    ///
    /// Same contract as :meth:`call_batch`, which is this method with
    /// ``device = 0``. The PTX is device-independent — it is generated once and
    /// each device gets its own lazily-loaded module — so the same
    /// ``CudaCompiledFn`` can be driven across every device on the host.
    ///
    /// `alkahest-core` has always been able to target a chosen device
    /// (`CudaCompiledFn::call_batch_on`, exercised by
    /// `nvptx_gpu::nvptx_multi_device_both_3090s`), but the binding only ever
    /// exposed device 0, so on a multi-GPU host every device but the first was
    /// unreachable from Python.
    ///
    /// Raises :class:`CudaError` if *device* is not a valid ordinal on this
    /// host.
    #[pyo3(name = "call_batch_on")]
    fn call_batch_on_py(&self, device: usize, inputs: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        self.eval_on_device(device, inputs)
    }
}

#[cfg(feature = "cuda")]
impl PyCudaCompiledFn {
    /// Shared body of `call_batch` / `call_batch_on`: validate the SoA input
    /// columns, then dispatch to the requested device ordinal.
    fn eval_on_device(&self, device: usize, inputs: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        if inputs.len() != self.inner.n_inputs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} input columns, got {}",
                self.inner.n_inputs,
                inputs.len()
            )));
        }
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(self.inner.n_inputs);
        for item in inputs.iter() {
            let col: Vec<f64> = item.extract()?;
            cols.push(col);
        }
        let n_pts = if cols.is_empty() { 0 } else { cols[0].len() };
        if cols.iter().any(|c| c.len() != n_pts) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all input columns must have the same length",
            ));
        }
        let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        let mut out = vec![0.0f64; n_pts];
        self.inner
            .call_batch_on(device, &col_refs, &mut out)
            .map_err(|e| {
                Python::with_gil(|py2| {
                    let exc_type = py2.get_type_bound::<PyCudaError>();
                    make_structured_err(py2, &exc_type, &e)
                })
            })?;
        Ok(out)
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

    Ok(PyCudaCompiledFn { inner: compiled })
}

/// `alkahest.cuda_device_count() -> int`
///
/// Number of CUDA devices visible to this process; `0` when none are.
///
/// The valid arguments to :meth:`CudaCompiledFn.call_batch_on` are exactly
/// ``range(cuda_device_count())``. Without this, the only way to discover the
/// range was to launch on an ordinal and catch ``E-CUDA-003`` — the workaround
/// `docs/mdbook/src/gpu.md` had to document while no verified implementation
/// existed.
///
/// Never raises: every "no GPU here" shape (no driver, no device, driver too
/// old) reports `0`, which is the single answer a caller acts on.
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(name = "cuda_device_count")]
fn py_cuda_device_count() -> usize {
    alkahest_core::cuda_device_count()
}

// ---------------------------------------------------------------------------
// V5-11 — Gröbner basis
// ---------------------------------------------------------------------------

#[cfg(feature = "groebner")]
use alkahest_core::{
    dae_index_reduce_ranked, expr_to_gbpoly, gbpoly_to_expr, primary_decomposition,
    radical as core_ideal_radical, rosenfeld_groebner_ranked, DaeIndexReduction, GbPoly,
    GroebnerBasis, MonomialOrder, ParamGbPoly, ParamGroebnerBasis, ParamGroebnerError, ParamPoly,
    QParam,
};

/// A sparse multivariate polynomial over ℚ, as used by the Gröbner machinery.
///
/// A `GbPoly` stores exponent *vectors*, not variable names, so reading one
/// back needs the variable list its exponent slots refer to.  Polynomials
/// handed out by Alkahest carry that list with them (see :meth:`variables`),
/// so :meth:`to_expr` normally takes no arguments; pass `vars` explicitly only
/// for a polynomial built against a different list.
///
/// Attributes
/// ----------
/// is_zero : bool
/// n_vars : int
/// n_terms : int
#[cfg(feature = "groebner")]
#[pyclass(name = "GbPoly")]
struct PyGbPoly {
    inner: GbPoly,
    /// Pool that `var_ids` belong to, when the variable context is known.
    pool: Option<Py<PyExprPool>>,
    /// Variables in the order used for exponent vectors.
    var_ids: Vec<ExprId>,
}

#[cfg(feature = "groebner")]
impl PyGbPoly {
    /// Wrap a core polynomial together with the variable context that names
    /// its exponent slots.
    fn with_ctx(
        py: Python<'_>,
        inner: GbPoly,
        pool: Option<&Py<PyExprPool>>,
        var_ids: &[ExprId],
    ) -> PyGbPoly {
        PyGbPoly {
            inner,
            pool: pool.map(|p| p.clone_ref(py)),
            var_ids: var_ids.to_vec(),
        }
    }
}

/// Resolve the `(pool, var_ids)` pair for an `Expr` conversion: an explicit
/// `vars` argument wins, otherwise the stored context, otherwise an error.
#[cfg(feature = "groebner")]
fn resolve_gb_ctx(
    py: Python<'_>,
    stored_pool: Option<&Py<PyExprPool>>,
    stored_vars: &[ExprId],
    vars: Option<Vec<PyRef<PyExpr>>>,
    what: &str,
) -> PyResult<(Py<PyExprPool>, Vec<ExprId>)> {
    match vars {
        Some(v) if !v.is_empty() => Ok((v[0].pool.clone_ref(py), v.iter().map(|e| e.id).collect())),
        _ => match stored_pool {
            Some(p) => Ok((p.clone_ref(py), stored_vars.to_vec())),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{what} carries no variable context; pass vars=[...] naming exponent slots 0, 1, …"
            ))),
        },
    }
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyGbPoly {
    /// True if this is the zero polynomial.
    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Number of variables in the ambient ring.
    #[getter]
    fn n_vars(&self) -> usize {
        self.inner.n_vars
    }

    /// Number of non-zero terms.
    #[getter]
    fn n_terms(&self) -> usize {
        self.inner.terms.len()
    }

    /// The variables naming this polynomial's exponent slots, in order.
    ///
    /// Empty when the polynomial carries no variable context — see
    /// :meth:`to_expr`.
    fn variables(&self, py: Python<'_>) -> Vec<PyExpr> {
        match &self.pool {
            None => vec![],
            Some(pool) => self
                .var_ids
                .iter()
                .map(|&id| PyExpr {
                    id,
                    pool: pool.clone_ref(py),
                })
                .collect(),
        }
    }

    /// The terms as ``(exponents, coefficient)`` pairs.
    ///
    /// `exponents` is a tuple of `int` parallel to :meth:`variables`;
    /// `coefficient` is an exact Python `int` or `fractions.Fraction`.
    /// Terms come in ascending exponent-vector order.
    ///
    /// Example::
    ///
    ///     p = alkahest.expr_to_gbpoly(x**2 - 3*y, [x, y])
    ///     p.terms()   # [((0, 1), -3), ((2, 0), 1)]
    fn terms(&self, py: Python<'_>) -> PyResult<Vec<(PyObject, PyObject)>> {
        let mut out = Vec::with_capacity(self.inner.terms.len());
        for (exp, coeff) in &self.inner.terms {
            let exps = pyo3::types::PyTuple::new_bound(py, exp.iter().map(|&e| e as u64));
            out.push((exps.into_py(py), rational_to_py(py, coeff)?));
        }
        Ok(out)
    }

    /// Convert back to an :class:`Expr`.
    ///
    /// Parameters
    /// ----------
    /// vars : list[Expr], optional
    ///     Variables naming exponent slots 0, 1, ….  Defaults to
    ///     :meth:`variables`, which is what you want for a polynomial that came
    ///     out of a :class:`GroebnerBasis`, a :class:`RegularChain` or
    ///     :func:`rosenfeld_groebner`.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If no variable context is available, or `vars` names fewer variables
    ///     than the polynomial actually uses.
    ///
    /// Example::
    ///
    ///     gb = alkahest.GroebnerBasis.compute([x**2 - y, x - y], [x, y])
    ///     [g.to_expr() for g in gb]
    #[pyo3(signature = (vars=None))]
    fn to_expr(&self, py: Python<'_>, vars: Option<Vec<PyRef<PyExpr>>>) -> PyResult<PyExpr> {
        let (pool_py, var_ids) =
            resolve_gb_ctx(py, self.pool.as_ref(), &self.var_ids, vars, "GbPoly")?;
        let id = {
            let pool = pool_py.borrow(py);
            gbpoly_to_expr(&self.inner, &var_ids, &pool.inner)
        };
        match id {
            Some(id) => Ok(PyExpr { id, pool: pool_py }),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "GbPoly is over {} variables but only {} were named; \
                 pass the full vars list it was built with",
                self.inner.n_vars,
                var_ids.len()
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!("GbPoly(n_terms={})", self.inner.terms.len())
    }
}

/// `alkahest.expr_to_gbpoly(expr, vars)` — convert a polynomial :class:`Expr`
/// into the :class:`GbPoly` representation.
///
/// The inverse of :meth:`GbPoly.to_expr`.  Exponent slot `i` of the result
/// refers to ``vars[i]``, and the polynomial remembers `vars`, so it can be fed
/// straight to :meth:`GroebnerBasis.reduce`, :meth:`GroebnerBasis.contains` or
/// :meth:`GroebnerBasis.compute_raw`.
///
/// Raises `ValueError` if *expr* is not polynomial in *vars* — a free symbol
/// outside *vars*, a negative or symbolic exponent, or a transcendental call.
///
/// Example::
///
///     p = alkahest.expr_to_gbpoly(x**2 + y**2 - pool.integer(1), [x, y])
///     gb = alkahest.GroebnerBasis.compute_raw([p])
#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "expr_to_gbpoly", signature = (expr, vars))]
fn py_expr_to_gbpoly(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    vars: Vec<PyRef<PyExpr>>,
) -> PyResult<PyGbPoly> {
    if vars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expr_to_gbpoly requires at least one variable",
        ));
    }
    let pool_py = expr.pool.clone_ref(py);
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let inner = {
        let pool = pool_py.borrow(py);
        expr_to_gbpoly(expr.id, &var_ids, &pool.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };
    Ok(PyGbPoly {
        inner,
        pool: Some(pool_py),
        var_ids,
    })
}

/// A computed Gröbner basis for a polynomial ideal.
///
/// The basis is a **sequence**: it supports `len()`, integer indexing and
/// iteration, yielding its generators as :class:`GbPoly`.  Read the generators
/// back as expressions with :meth:`to_exprs`, or one at a time with
/// :meth:`GbPoly.to_expr` — that is how an elimination ideal (a ``"lex"`` basis,
/// or the differential elimination performed by :func:`rosenfeld_groebner`) is
/// turned into readable relations.
///
/// Attributes
/// ----------
/// order : str
///     Monomial order the generators are reduced under: ``"lex"``, ``"grlex"``
///     or ``"grevlex"``.
///
/// Example::
///
///     gb = alkahest.GroebnerBasis.compute([x**2 + y**2 - one, x - y], [x, y])
///     len(gb)                     # number of generators
///     [g.to_expr() for g in gb]   # generators as Expr
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
impl PyGroebnerBasis {
    /// A generator wrapped with this basis's variable context.
    fn wrap(&self, py: Python<'_>, p: GbPoly) -> PyGbPoly {
        PyGbPoly::with_ctx(py, p, self.pool.as_ref(), &self.var_ids)
    }

    /// Accept a `GbPoly` as-is, or convert an `Expr` using this basis's
    /// variable ordering.
    fn coerce_to_gbpoly(&self, py: Python<'_>, p: &Bound<'_, PyAny>) -> PyResult<GbPoly> {
        if let Ok(gbp) = p.downcast::<PyGbPoly>() {
            return Ok(gbp.borrow().inner.clone());
        }
        if let Ok(expr) = p.downcast::<PyExpr>() {
            let pool_py = self.pool.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "GroebnerBasis has no variable context; use GroebnerBasis.compute() to build one that accepts Expr, or pass a GbPoly from expr_to_gbpoly()",
                )
            })?;
            let pool = pool_py.borrow(py);
            return expr_to_gbpoly(expr.borrow().id, &self.var_ids, &pool.inner)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a GbPoly or an Expr",
        ))
    }
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyGroebnerBasis {
    /// Compute a Gröbner basis for the polynomial system `polys = 0` in
    /// the given variables.
    ///
    /// Parameters
    /// ----------
    /// polys : list[Expr]
    ///     Polynomial expressions, each representing ``p(vars) = 0``.
    /// vars : list[Expr]
    ///     Symbolic variables (must be ``Symbol``).
    /// order : str, optional
    ///     Monomial order: ``"lex"`` (default), ``"grevlex"``, or ``"grlex"``.
    ///     When ``"lex"`` is requested and the ideal is 0-dimensional, the
    ///     grevlex-then-FGLM strategy is used automatically (much faster than
    ///     direct lex Buchberger for 3+ variable systems).
    /// params : list[Expr], optional
    ///     Symbols to put in the **coefficient field** rather than the ring
    ///     (M9).  With ``params`` the computation runs in ``Q(params)[vars]``
    ///     instead of ``Q[vars, params]``, and the return type is a
    ///     :class:`ParametricGroebnerBasis` — the same sequence protocol, plus
    ///     :meth:`~ParametricGroebnerBasis.conditions` and
    ///     :meth:`~ParametricGroebnerBasis.specialize`.  That class is
    ///     experimental; :class:`GroebnerBasis` is unchanged when ``params`` is
    ///     omitted or empty.
    ///
    /// Example::
    ///
    ///     # parameters as ring variables — a is a 3rd variable
    ///     gb = alkahest.GroebnerBasis.compute([a*x - y, x + y - one], [x, y, a])
    ///     # parameters in the coefficient field
    ///     gb = alkahest.GroebnerBasis.compute([a*x - y, x + y - one], [x, y], params=[a])
    #[staticmethod]
    #[pyo3(signature = (polys, vars, order=None, params=None))]
    fn compute(
        py: Python<'_>,
        polys: Vec<PyRef<PyExpr>>,
        vars: Vec<PyRef<PyExpr>>,
        order: Option<&str>,
        params: Option<Vec<PyRef<PyExpr>>>,
    ) -> PyResult<PyObject> {
        if let Some(params) = params {
            if !params.is_empty() {
                let basis = PyParamGroebnerBasis::build(py, polys, vars, params, order)?;
                return Ok(Py::new(py, basis)?.into_py(py));
            }
        }
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
        let parsed_order = order
            .and_then(MonomialOrder::from_str)
            .unwrap_or(MonomialOrder::Lex);
        let inner = match parsed_order {
            MonomialOrder::Lex => GroebnerBasis::compute_lex(gb_polys),
            other => GroebnerBasis::compute(gb_polys, other),
        };
        Ok(Py::new(
            py,
            PyGroebnerBasis {
                inner,
                pool: Some(pool_py),
                var_ids,
            },
        )?
        .into_py(py))
    }

    /// Gröbner basis via Faugère's F5 (signature-based reduction, V2-8).
    ///
    /// Parameters
    /// ----------
    /// polys, vars
    ///     Same as :meth:`compute`.
    /// order : str, optional
    ///     Monomial order: ``"lex"`` (default), ``"grevlex"``, or ``"grlex"``.
    #[staticmethod]
    #[pyo3(signature = (polys, vars, order=None))]
    fn compute_f5(
        py: Python<'_>,
        polys: Vec<PyRef<PyExpr>>,
        vars: Vec<PyRef<PyExpr>>,
        order: Option<&str>,
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
        let parsed_order = order
            .and_then(MonomialOrder::from_str)
            .unwrap_or(MonomialOrder::Lex);
        let inner = GroebnerBasis::compute_f5(gb_polys, parsed_order);
        Ok(PyGroebnerBasis {
            inner,
            pool: Some(pool_py),
            var_ids,
        })
    }

    /// Low-level entry point that accepts already-converted ``GbPoly`` objects,
    /// bypassing the :func:`expr_to_gbpoly` conversion. Useful when the
    /// polynomial representation is already known (e.g., from ``MultiPoly``
    /// reconstruction).
    ///
    /// The variable context of the *first* input polynomial, if it has one, is
    /// carried onto the resulting basis, so a basis built from
    /// :func:`expr_to_gbpoly` output stays readable via :meth:`to_exprs`.
    ///
    /// Parameters
    /// ----------
    /// gb_polys : list[GbPoly]
    ///     Already-converted polynomial objects.
    /// order : str, optional
    ///     Monomial order: ``"lex"`` (default), ``"grevlex"``, or ``"grlex"``.
    ///     The same grevlex-then-FGLM strategy is applied for ``"lex"``.
    #[staticmethod]
    #[pyo3(signature = (gb_polys, order=None))]
    fn compute_raw(
        py: Python<'_>,
        gb_polys: Vec<PyRef<PyGbPoly>>,
        order: Option<&str>,
    ) -> PyResult<PyGroebnerBasis> {
        if gb_polys.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GroebnerBasis.compute_raw requires at least one GbPoly",
            ));
        }
        let pool = gb_polys[0].pool.as_ref().map(|p| p.clone_ref(py));
        let var_ids = gb_polys[0].var_ids.clone();
        let raw: Vec<GbPoly> = gb_polys.iter().map(|p| p.inner.clone()).collect();
        let parsed_order = order
            .and_then(MonomialOrder::from_str)
            .unwrap_or(MonomialOrder::Lex);
        let inner = match parsed_order {
            MonomialOrder::Lex => GroebnerBasis::compute_lex(raw),
            other => GroebnerBasis::compute(raw, other),
        };
        Ok(PyGroebnerBasis {
            inner,
            pool,
            var_ids,
        })
    }

    /// The monomial order the generators are reduced under.
    #[getter]
    fn order(&self) -> &'static str {
        self.inner.order().as_str()
    }

    /// The elimination ideal `I ∩ k[remaining vars]`, as a `GroebnerBasis`.
    ///
    /// Drops every generator whose support mentions one of *vars*.  Under a
    /// ``"lex"`` basis with the eliminated variables ordered **first**, what is
    /// left is a Gröbner basis for the elimination ideal — the relations among
    /// the remaining variables alone.  Read them with :meth:`to_exprs`.
    ///
    /// The basis must know its variable ordering; *vars* must be among
    /// :meth:`variables`.
    ///
    /// This is the implicitization move: parametrize a curve or surface, then
    /// eliminate the parameters.
    ///
    /// Example::
    ///
    ///     # Implicitize (t, t**2): eliminate t from {x - t, y - t**2}
    ///     gb = alkahest.GroebnerBasis.compute([x - t, y - t**2], [t, x, y])
    ///     gb.eliminate([t]).to_exprs()   # [((y * -1) + x^2)]  i.e. y = x**2
    fn eliminate(&self, py: Python<'_>, vars: Vec<PyRef<PyExpr>>) -> PyResult<PyGroebnerBasis> {
        if self.pool.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GroebnerBasis has no variable context; use GroebnerBasis.compute() to build one that can eliminate",
            ));
        }
        let mut indices = Vec::with_capacity(vars.len());
        for v in &vars {
            match self.var_ids.iter().position(|&id| id == v.id) {
                Some(i) => indices.push(i),
                None => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "eliminate() was given a variable this basis is not written over; \
                         see GroebnerBasis.variables()",
                    ))
                }
            }
        }
        Ok(PyGroebnerBasis {
            inner: self.inner.eliminate(&indices),
            pool: self.pool.as_ref().map(|p| p.clone_ref(py)),
            var_ids: self.var_ids.clone(),
        })
    }

    /// The variables naming exponent slots 0, 1, … of the generators.
    ///
    /// Empty when the basis carries no variable context (a basis built by
    /// :meth:`compute_raw` from context-free polynomials).
    fn variables(&self, py: Python<'_>) -> Vec<PyExpr> {
        match &self.pool {
            None => vec![],
            Some(pool) => self
                .var_ids
                .iter()
                .map(|&id| PyExpr {
                    id,
                    pool: pool.clone_ref(py),
                })
                .collect(),
        }
    }

    /// The basis generators as :class:`GbPoly`, interreduced and monic.
    ///
    /// Equivalent to ``list(basis)``.
    fn polynomials(&self, py: Python<'_>) -> Vec<PyGbPoly> {
        self.inner
            .generators()
            .iter()
            .map(|p| self.wrap(py, p.clone()))
            .collect()
    }

    /// The basis generators as :class:`Expr`, each meaning ``g = 0``.
    ///
    /// This is the read path for elimination: with a ``"lex"`` basis, the
    /// generators free of the eliminated variables are the eliminated
    /// relations.
    ///
    /// Parameters
    /// ----------
    /// vars : list[Expr], optional
    ///     Override the stored variable ordering — see :meth:`GbPoly.to_expr`.
    ///
    /// Example::
    ///
    ///     gb = alkahest.GroebnerBasis.compute([x**2 + y**2 - one, x - y], [x, y])
    ///     gb.to_exprs()
    #[pyo3(signature = (vars=None))]
    fn to_exprs(&self, py: Python<'_>, vars: Option<Vec<PyRef<PyExpr>>>) -> PyResult<Vec<PyExpr>> {
        let (pool_py, var_ids) =
            resolve_gb_ctx(py, self.pool.as_ref(), &self.var_ids, vars, "GroebnerBasis")?;
        let ids: Option<Vec<ExprId>> = {
            let pool = pool_py.borrow(py);
            self.inner
                .generators()
                .iter()
                .map(|g| gbpoly_to_expr(g, &var_ids, &pool.inner))
                .collect()
        };
        match ids {
            Some(ids) => Ok(ids
                .into_iter()
                .map(|id| PyExpr {
                    id,
                    pool: pool_py.clone_ref(py),
                })
                .collect()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "basis is over more variables than the {} named; \
                 pass the full vars list it was built with",
                var_ids.len()
            ))),
        }
    }

    /// Reduce a polynomial modulo this basis and return the remainder.
    ///
    /// Accepts a :class:`GbPoly` or an :class:`Expr`; passing an ``Expr``
    /// requires the basis to know its variable ordering (see
    /// :meth:`variables`).  The remainder is a :class:`GbPoly` — call
    /// :meth:`GbPoly.to_expr` on it to read it back.  It is zero exactly when
    /// :meth:`contains` is true.
    fn reduce(&self, py: Python<'_>, p: &Bound<'_, PyAny>) -> PyResult<PyGbPoly> {
        let poly = self.coerce_to_gbpoly(py, p)?;
        Ok(self.wrap(py, self.inner.reduce(&poly)))
    }

    /// Test membership.  Accepts either a ``GbPoly`` or an ``Expr``; when
    /// passing an ``Expr`` the basis must have been created via ``compute()``
    /// so that the variable order is known.
    fn contains(&self, py: Python<'_>, p: &Bound<'_, PyAny>) -> PyResult<bool> {
        let poly = self.coerce_to_gbpoly(py, p)?;
        Ok(self.inner.contains(&poly))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Generator `i` as a :class:`GbPoly`; negative indices count from the end.
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<PyGbPoly> {
        let n = self.inner.len() as isize;
        let i = if index < 0 { index + n } else { index };
        if i < 0 || i >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "GroebnerBasis index out of range",
            ));
        }
        Ok(self.wrap(py, self.inner.generators()[i as usize].clone()))
    }

    /// Iterate over the generators as :class:`GbPoly`.
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = pyo3::types::PyList::empty_bound(py);
        for p in self.inner.generators() {
            list.append(Py::new(py, self.wrap(py, p.clone()))?)?;
        }
        Ok(list.as_any().iter()?.into_py(py))
    }

    fn __repr__(&self) -> String {
        format!("GroebnerBasis(n_generators={})", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// M9 — Gröbner bases over the coefficient field Q(params)
// ---------------------------------------------------------------------------

#[cfg(feature = "groebner")]
fn param_groebner_error_to_py(e: ParamGroebnerError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyParamGroebnerError>();
        make_structured_err(py, &exc_type, &e)
    })
}

/// `ParamPoly` → `Expr` over the parameter symbols.
#[cfg(feature = "groebner")]
fn parampoly_to_expr(p: &ParamPoly, params: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let mut terms: Vec<ExprId> = Vec::with_capacity(p.terms.len());
    for (exp, coeff) in &p.terms {
        let mut factors: Vec<ExprId> = Vec::new();
        for (i, &e) in exp.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let v = *params.get(i)?;
            factors.push(if e == 1 {
                v
            } else {
                pool.pow(v, pool.integer(e))
            });
        }
        if factors.is_empty() || *coeff != 1 {
            factors.insert(0, pool.integer(coeff.clone()));
        }
        terms.push(if factors.len() == 1 {
            factors[0]
        } else {
            pool.mul(factors)
        });
    }
    Some(match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    })
}

/// `QParam` → `Expr`, as `num` or `num * den**-1`.
#[cfg(feature = "groebner")]
fn qparam_to_expr(c: &QParam, params: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let num = parampoly_to_expr(c.numerator(), params, pool)?;
    if c.denominator().is_one() {
        return Some(num);
    }
    let den = parampoly_to_expr(c.denominator(), params, pool)?;
    let inv = pool.pow(den, pool.integer(-1_i32));
    Some(pool.mul(vec![num, inv]))
}

/// `ParamGbPoly` → `Expr`: the variables carry exponents, the parameters ride
/// in the coefficients.
#[cfg(feature = "groebner")]
fn paramgbpoly_to_expr(
    p: &ParamGbPoly,
    vars: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
) -> Option<ExprId> {
    let mut terms: Vec<ExprId> = Vec::with_capacity(p.terms.len());
    for (exp, coeff) in &p.terms {
        let c = qparam_to_expr(coeff, params, pool)?;
        let mut factors: Vec<ExprId> = Vec::new();
        for (i, &e) in exp.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let v = *vars.get(i)?;
            factors.push(if e == 1 {
                v
            } else {
                pool.pow(v, pool.integer(e))
            });
        }
        if factors.is_empty() || !coeff.is_one() {
            factors.insert(0, c);
        }
        terms.push(if factors.len() == 1 {
            factors[0]
        } else {
            pool.mul(factors)
        });
    }
    Some(match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    })
}

/// A polynomial in the ring variables whose coefficients are rational
/// functions of the parameters.
///
/// The parametric counterpart of :class:`GbPoly`: exponent slots name the
/// *variables* only, because the parameters live in the coefficient field.
/// Read it back with :meth:`to_expr` exactly as you would a :class:`GbPoly`.
///
/// Attributes
/// ----------
/// is_zero : bool
/// n_vars : int
/// n_params : int
/// n_terms : int
#[cfg(feature = "groebner")]
#[pyclass(name = "ParametricGbPoly")]
struct PyParamGbPoly {
    inner: ParamGbPoly,
    pool: Option<Py<PyExprPool>>,
    var_ids: Vec<ExprId>,
    param_ids: Vec<ExprId>,
}

#[cfg(feature = "groebner")]
impl PyParamGbPoly {
    fn with_ctx(
        py: Python<'_>,
        inner: ParamGbPoly,
        pool: Option<&Py<PyExprPool>>,
        var_ids: &[ExprId],
        param_ids: &[ExprId],
    ) -> PyParamGbPoly {
        PyParamGbPoly {
            inner,
            pool: pool.map(|p| p.clone_ref(py)),
            var_ids: var_ids.to_vec(),
            param_ids: param_ids.to_vec(),
        }
    }

    fn require_pool(&self, what: &str) -> PyResult<&Py<PyExprPool>> {
        self.pool.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{what} carries no variable context; rebuild it with \
                 ParametricGroebnerBasis.compute()"
            ))
        })
    }
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyParamGbPoly {
    /// True if this is the zero polynomial.
    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Number of ring variables.
    #[getter]
    fn n_vars(&self) -> usize {
        self.inner.n_vars
    }

    /// Number of parameters in the coefficient field.
    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params
    }

    /// Number of non-zero terms.
    #[getter]
    fn n_terms(&self) -> usize {
        self.inner.n_terms()
    }

    /// The variables naming exponent slots 0, 1, ….
    fn variables(&self, py: Python<'_>) -> Vec<PyExpr> {
        match &self.pool {
            None => vec![],
            Some(pool) => self
                .var_ids
                .iter()
                .map(|&id| PyExpr {
                    id,
                    pool: pool.clone_ref(py),
                })
                .collect(),
        }
    }

    /// The parameters of the coefficient field, in order.
    fn parameters(&self, py: Python<'_>) -> Vec<PyExpr> {
        match &self.pool {
            None => vec![],
            Some(pool) => self
                .param_ids
                .iter()
                .map(|&id| PyExpr {
                    id,
                    pool: pool.clone_ref(py),
                })
                .collect(),
        }
    }

    /// The terms as ``(exponents, coefficient)`` pairs.
    ///
    /// `exponents` is a tuple of `int` parallel to :meth:`variables`;
    /// `coefficient` is an :class:`Expr` in the parameters — a rational
    /// function, not necessarily a polynomial.  That is the whole difference
    /// from :meth:`GbPoly.terms`, whose coefficients are numbers.
    fn terms(&self, py: Python<'_>) -> PyResult<Vec<(PyObject, PyExpr)>> {
        let pool_py = self.require_pool("ParametricGbPoly")?.clone_ref(py);
        let mut out = Vec::with_capacity(self.inner.terms.len());
        {
            let pool = pool_py.borrow(py);
            for (exp, coeff) in &self.inner.terms {
                let exps = pyo3::types::PyTuple::new_bound(py, exp.iter().map(|&e| e as u64));
                let id = qparam_to_expr(coeff, &self.param_ids, &pool.inner).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "coefficient mentions more parameters than were named",
                    )
                })?;
                out.push((
                    exps.into_py(py),
                    PyExpr {
                        id,
                        pool: pool_py.clone_ref(py),
                    },
                ));
            }
        }
        Ok(out)
    }

    /// Convert back to an :class:`Expr`, coefficients and all.
    ///
    /// Denominators appear as ``den**-1`` factors, so the result is a rational
    /// expression in the parameters and a polynomial in the variables.
    ///
    /// Example::
    ///
    ///     gb = alkahest.GroebnerBasis.compute([a*x - y, x + y - 1], [x, y], params=[a])
    ///     [g.to_expr() for g in gb]
    fn to_expr(&self, py: Python<'_>) -> PyResult<PyExpr> {
        let pool_py = self.require_pool("ParametricGbPoly")?.clone_ref(py);
        let id = {
            let pool = pool_py.borrow(py);
            paramgbpoly_to_expr(&self.inner, &self.var_ids, &self.param_ids, &pool.inner)
        };
        match id {
            Some(id) => Ok(PyExpr { id, pool: pool_py }),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "polynomial is over more variables or parameters than were named",
            )),
        }
    }

    /// Substitute rational values for the parameters, giving a :class:`GbPoly`.
    ///
    /// Raises `ParamGroebnerError` (``E-PARAMGB-004``) when a coefficient has a
    /// pole at that point.
    fn specialize(&self, py: Python<'_>, values: Vec<Bound<'_, PyAny>>) -> PyResult<PyGbPoly> {
        let vals = values
            .iter()
            .map(py_to_rational)
            .collect::<PyResult<Vec<Rational>>>()?;
        if vals.len() != self.inner.n_params {
            return Err(param_groebner_error_to_py(ParamGroebnerError::WrongArity {
                expected: self.inner.n_params,
                got: vals.len(),
            }));
        }
        let p = self.inner.specialize(&vals).ok_or_else(|| {
            param_groebner_error_to_py(ParamGroebnerError::Degenerate { vanishing: vec![] })
        })?;
        Ok(PyGbPoly::with_ctx(py, p, self.pool.as_ref(), &self.var_ids))
    }

    fn __repr__(&self) -> String {
        format!(
            "ParametricGbPoly(n_terms={}, n_params={})",
            self.inner.n_terms(),
            self.inner.n_params
        )
    }
}

/// A Gröbner basis computed with the parameters in the **coefficient field**.
///
/// `GroebnerBasis.compute(polys, vars)` puts everything in `Q[vars]`, so a
/// parameter has to be declared as one more ring variable; the elimination then
/// runs in `Q[vars, params]`.  Here the parameters are moved into `Q(params)`
/// instead: they never enter the monomial order, never generate S-pairs, and
/// never enlarge the staircase.  For differential elimination — reading the
/// input-output equations of an ODE model out of a state elimination — that is
/// the difference between a computation that finishes and one that does not.
///
/// The object is a **sequence** of :class:`ParametricGbPoly`, like
/// :class:`GroebnerBasis`, and supports the same `len()` / indexing /
/// iteration / :meth:`to_exprs` / :meth:`eliminate` read path.
///
/// **The result is generic.** A leading coefficient can be a non-zero element
/// of `Q(params)` and still vanish for particular parameter values, and there
/// the basis says nothing. :meth:`conditions` lists the polynomials whose
/// non-vanishing was assumed; the basis holds at exactly the parameter points
/// where none of them vanishes. :meth:`specialize` refuses on the rest rather
/// than returning something that is not a basis.
///
/// Attributes
/// ----------
/// order : str
///     Monomial order: ``"lex"``, ``"grlex"`` or ``"grevlex"``.
/// n_params : int
///     Number of parameters in the coefficient field.
///
/// Example::
///
///     gb = alkahest.GroebnerBasis.compute([a*x - y, x + y - one], [x, y], params=[a])
///     [g.to_expr() for g in gb]        # coefficients are rational in `a`
///     [c for c in gb.conditions()]     # [a + 1] — the basis is silent at a = -1
///     gb.specialize([3])               # an ordinary GroebnerBasis over ℚ
#[cfg(feature = "groebner")]
#[pyclass(name = "ParametricGroebnerBasis")]
struct PyParamGroebnerBasis {
    inner: ParamGroebnerBasis,
    pool: Py<PyExprPool>,
    var_ids: Vec<ExprId>,
    param_ids: Vec<ExprId>,
}

#[cfg(feature = "groebner")]
impl PyParamGroebnerBasis {
    fn wrap(&self, py: Python<'_>, p: ParamGbPoly) -> PyParamGbPoly {
        PyParamGbPoly::with_ctx(py, p, Some(&self.pool), &self.var_ids, &self.param_ids)
    }

    /// Shared constructor for `ParametricGroebnerBasis.compute` and for
    /// `GroebnerBasis.compute(..., params=[...])`.
    fn build(
        py: Python<'_>,
        polys: Vec<PyRef<PyExpr>>,
        vars: Vec<PyRef<PyExpr>>,
        params: Vec<PyRef<PyExpr>>,
        order: Option<&str>,
    ) -> PyResult<PyParamGroebnerBasis> {
        if polys.is_empty() || vars.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a parametric Gröbner basis needs at least one polynomial and one variable",
            ));
        }
        let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
        let param_ids: Vec<ExprId> = params.iter().map(|p| p.id).collect();
        if let Some(clash) = param_ids.iter().find(|p| var_ids.contains(p)) {
            let _ = clash;
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a symbol cannot be both a ring variable and a coefficient-field parameter",
            ));
        }
        let mut all_ids = var_ids.clone();
        all_ids.extend_from_slice(&param_ids);

        let pool_py = polys[0].pool.clone_ref(py);
        let mut gens = Vec::with_capacity(polys.len());
        {
            let pool = pool_py.borrow(py);
            for p in &polys {
                let gbp = expr_to_gbpoly(p.id, &all_ids, &pool.inner)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let pg = ParamGbPoly::from_gbpoly(&gbp, var_ids.len(), param_ids.len())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "internal: polynomial arity does not match vars + params",
                        )
                    })?;
                gens.push(pg);
            }
        }
        let parsed_order = order
            .and_then(MonomialOrder::from_str)
            .unwrap_or(MonomialOrder::Lex);
        let inner =
            ParamGroebnerBasis::compute(gens, parsed_order).map_err(param_groebner_error_to_py)?;
        Ok(PyParamGroebnerBasis {
            inner,
            pool: pool_py,
            var_ids,
            param_ids,
        })
    }

    fn rational_values(&self, values: &[Bound<'_, PyAny>]) -> PyResult<Vec<Rational>> {
        let vals = values
            .iter()
            .map(py_to_rational)
            .collect::<PyResult<Vec<Rational>>>()?;
        if vals.len() != self.inner.n_params() {
            return Err(param_groebner_error_to_py(ParamGroebnerError::WrongArity {
                expected: self.inner.n_params(),
                got: vals.len(),
            }));
        }
        Ok(vals)
    }
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyParamGroebnerBasis {
    /// Compute a Gröbner basis in ``Q(params)[vars]``.
    ///
    /// Parameters
    /// ----------
    /// polys : list[Expr]
    ///     Polynomial expressions, each meaning ``p = 0``.  They must be
    ///     polynomial in *vars* and in *params*.
    /// vars : list[Expr]
    ///     The ring variables — the ones the monomial order sees.
    /// params : list[Expr]
    ///     Symbols to place in the coefficient field.  Must be disjoint from
    ///     *vars*.
    /// order : str, optional
    ///     ``"lex"`` (default), ``"grevlex"`` or ``"grlex"``.  Lex is what
    ///     elimination needs: order the variables to eliminate first.
    #[staticmethod]
    #[pyo3(signature = (polys, vars, params, order=None))]
    fn compute(
        py: Python<'_>,
        polys: Vec<PyRef<PyExpr>>,
        vars: Vec<PyRef<PyExpr>>,
        params: Vec<PyRef<PyExpr>>,
        order: Option<&str>,
    ) -> PyResult<PyParamGroebnerBasis> {
        PyParamGroebnerBasis::build(py, polys, vars, params, order)
    }

    /// The monomial order the generators are reduced under.
    #[getter]
    fn order(&self) -> &'static str {
        self.inner.order().as_str()
    }

    /// Number of parameters in the coefficient field.
    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    /// The ring variables naming exponent slots 0, 1, ….
    fn variables(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.var_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// The parameters of the coefficient field, in order.
    fn parameters(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.param_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// The polynomials in the parameters whose non-vanishing this basis
    /// assumed, as :class:`Expr`.
    ///
    /// Each is irreducible, primitive and has a positive leading coefficient.
    /// The basis is valid at exactly the parameter points where **none** of
    /// them vanishes; the degeneracy locus is the union of the hypersurfaces
    /// they cut out.  An empty list means the basis holds everywhere.
    ///
    /// The list is *sufficient, not necessary* — a point on the locus may still
    /// be fine, but this computation cannot see that, and the honest report is
    /// the hypothesis it actually used.
    ///
    /// Example::
    ///
    ///     gb = alkahest.GroebnerBasis.compute([a*x - y, x + y - one], [x, y], params=[a])
    ///     [str(c) for c in gb.conditions()]   # ['a + 1']
    fn conditions(&self, py: Python<'_>) -> PyResult<Vec<PyExpr>> {
        let pool = self.pool.borrow(py);
        let mut out = Vec::with_capacity(self.inner.conditions().len());
        for c in self.inner.conditions() {
            let id = parampoly_to_expr(c, &self.param_ids, &pool.inner).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "condition mentions more parameters than were named",
                )
            })?;
            out.push(PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            });
        }
        Ok(out)
    }

    /// The conditions that vanish at *values* — empty exactly when the basis
    /// applies at that parameter point.
    fn vanishing_conditions(
        &self,
        py: Python<'_>,
        values: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyExpr>> {
        let vals = self.rational_values(&values)?;
        let pool = self.pool.borrow(py);
        let mut out = Vec::new();
        for c in self.inner.vanishing_conditions(&vals) {
            let id = parampoly_to_expr(&c, &self.param_ids, &pool.inner).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "condition mentions more parameters than were named",
                )
            })?;
            out.push(PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            });
        }
        Ok(out)
    }

    /// True when *values* lies off the degeneracy locus.
    fn is_regular_at(&self, values: Vec<Bound<'_, PyAny>>) -> PyResult<bool> {
        let vals = self.rational_values(&values)?;
        Ok(self.inner.is_regular_at(&vals))
    }

    /// Substitute rational values for the parameters, giving an ordinary
    /// :class:`GroebnerBasis` over ℚ.
    ///
    /// Off the degeneracy locus the result is exactly what
    /// :meth:`GroebnerBasis.compute` would return for the specialised system.
    /// On it, this raises `ParamGroebnerError` with ``.code ==
    /// "E-PARAMGB-004"`` rather than handing back something that is not a
    /// basis — check first with :meth:`is_regular_at` if that is a normal
    /// outcome for your caller.
    fn specialize(
        &self,
        py: Python<'_>,
        values: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyGroebnerBasis> {
        let vals = self.rational_values(&values)?;
        let gens = self
            .inner
            .specialize(&vals)
            .map_err(param_groebner_error_to_py)?;
        Ok(PyGroebnerBasis {
            inner: GroebnerBasis::from_generators(gens, self.inner.order()),
            pool: Some(self.pool.clone_ref(py)),
            var_ids: self.var_ids.clone(),
        })
    }

    /// The elimination ideal `I ∩ Q(params)[remaining vars]`.
    ///
    /// Same contract as :meth:`GroebnerBasis.eliminate`: under a ``"lex"``
    /// basis with the eliminated variables ordered **first**, the generators
    /// free of them generate the elimination ideal.  The conditions travel with
    /// the result — eliminating does not make the hypotheses go away.
    fn eliminate(
        &self,
        py: Python<'_>,
        vars: Vec<PyRef<PyExpr>>,
    ) -> PyResult<PyParamGroebnerBasis> {
        let mut indices = Vec::with_capacity(vars.len());
        for v in &vars {
            match self.var_ids.iter().position(|&id| id == v.id) {
                Some(i) => indices.push(i),
                None => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "eliminate() was given a symbol this basis is not written over; \
                         parameters cannot be eliminated — they are in the coefficient field",
                    ))
                }
            }
        }
        Ok(PyParamGroebnerBasis {
            inner: self.inner.eliminate(&indices),
            pool: self.pool.clone_ref(py),
            var_ids: self.var_ids.clone(),
            param_ids: self.param_ids.clone(),
        })
    }

    /// The generators as :class:`ParametricGbPoly`. Equivalent to ``list(basis)``.
    fn polynomials(&self, py: Python<'_>) -> Vec<PyParamGbPoly> {
        self.inner
            .generators()
            .iter()
            .map(|p| self.wrap(py, p.clone()))
            .collect()
    }

    /// The generators as :class:`Expr`, each meaning ``g = 0``.
    fn to_exprs(&self, py: Python<'_>) -> PyResult<Vec<PyExpr>> {
        let ids: Option<Vec<ExprId>> = {
            let pool = self.pool.borrow(py);
            self.inner
                .generators()
                .iter()
                .map(|g| paramgbpoly_to_expr(g, &self.var_ids, &self.param_ids, &pool.inner))
                .collect()
        };
        match ids {
            Some(ids) => Ok(ids
                .into_iter()
                .map(|id| PyExpr {
                    id,
                    pool: self.pool.clone_ref(py),
                })
                .collect()),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "basis is over more variables or parameters than were named",
            )),
        }
    }

    /// Reduce a polynomial modulo this basis; the remainder is a
    /// :class:`ParametricGbPoly`.
    ///
    /// Accepts a :class:`ParametricGbPoly` or an :class:`Expr`.
    fn reduce(&self, py: Python<'_>, p: &Bound<'_, PyAny>) -> PyResult<PyParamGbPoly> {
        let poly = self.coerce(py, p)?;
        Ok(self.wrap(py, self.inner.reduce(&poly)))
    }

    /// Ideal membership: true exactly when :meth:`reduce` gives zero.
    fn contains(&self, py: Python<'_>, p: &Bound<'_, PyAny>) -> PyResult<bool> {
        let poly = self.coerce(py, p)?;
        Ok(self.inner.contains(&poly))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Generator `i`; negative indices count from the end.
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<PyParamGbPoly> {
        let n = self.inner.len() as isize;
        let i = if index < 0 { index + n } else { index };
        if i < 0 || i >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "ParametricGroebnerBasis index out of range",
            ));
        }
        Ok(self.wrap(py, self.inner.generators()[i as usize].clone()))
    }

    /// Iterate over the generators.
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = pyo3::types::PyList::empty_bound(py);
        for p in self.inner.generators() {
            list.append(Py::new(py, self.wrap(py, p.clone()))?)?;
        }
        Ok(list.as_any().iter()?.into_py(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "ParametricGroebnerBasis(n_generators={}, n_params={}, n_conditions={})",
            self.inner.len(),
            self.inner.n_params(),
            self.inner.conditions().len()
        )
    }
}

#[cfg(feature = "groebner")]
impl PyParamGroebnerBasis {
    /// Accept a `ParametricGbPoly` as-is, or convert an `Expr` against this
    /// basis's variable and parameter lists.
    fn coerce(&self, py: Python<'_>, p: &Bound<'_, PyAny>) -> PyResult<ParamGbPoly> {
        if let Ok(pg) = p.downcast::<PyParamGbPoly>() {
            return Ok(pg.borrow().inner.clone());
        }
        if let Ok(expr) = p.downcast::<PyExpr>() {
            let mut all_ids = self.var_ids.clone();
            all_ids.extend_from_slice(&self.param_ids);
            let pool = self.pool.borrow(py);
            let gbp = expr_to_gbpoly(expr.borrow().id, &all_ids, &pool.inner)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            return ParamGbPoly::from_gbpoly(&gbp, self.var_ids.len(), self.param_ids.len())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "internal: polynomial arity does not match vars + params",
                    )
                });
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a ParametricGbPoly or an Expr",
        ))
    }
}

#[cfg(feature = "groebner")]
fn py_monomial_order_for_dae(order: Option<&str>) -> MonomialOrder {
    order
        .and_then(MonomialOrder::from_str)
        .unwrap_or(MonomialOrder::GRevLex)
}

/// V2-13 — result of Rosenfeld–Gröbner-style differential elimination.
///
/// Returned by :func:`rosenfeld_groebner`.  The eliminated relations are in
/// :meth:`final_basis`; read them with ``result.final_basis().to_exprs()``.
///
/// Attributes
/// ----------
/// consistent : bool
///     ``False`` iff the unit ideal was reached — the system has no common
///     jet solution over ℚ, i.e. the equations are contradictory.
/// truncated : bool
///     ``True`` if prolongation stopped at ``max_prolong_rounds`` rather than
///     because the differential chain saturated.  A truncated basis is a
///     *sound* set of consequences of the system but need not be complete, so
///     "not in the basis" does not mean "not a consequence".
/// prolongation_rounds : int
///     Number of prolongation rounds that contributed new relations.
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
    /// Jet variables indexing the exponent vectors of `final_basis`.
    var_ids: Vec<ExprId>,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyRosenfeldGroebnerResult {
    /// The prolonged :class:`DAE`: the input system plus the derivative jets
    /// introduced while differentiating it.
    fn working_dae(&self, py: Python<'_>) -> PyDAE {
        PyDAE {
            inner: self.working_dae.clone(),
            pool: self.pool.clone_ref(py),
        }
    }

    /// The jet variables indexing the basis, in exponent-slot order.
    ///
    /// These are the symbols the elimination actually ran over — the time
    /// variable, the declared states and derivatives, and every higher jet
    /// (``d2x/dt2``, …) introduced by prolongation.
    fn variables(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.var_ids
            .iter()
            .map(|&id| PyExpr {
                id,
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// The saturated Gröbner basis, or ``None`` when the system is
    /// inconsistent.
    ///
    /// The returned :class:`GroebnerBasis` knows its variable ordering, so
    /// ``final_basis().to_exprs()`` gives the eliminated relations as
    /// :class:`Expr`.
    ///
    /// Example::
    ///
    ///     r = alkahest.rosenfeld_groebner(dae, max_prolong_rounds=1)
    ///     for eq in r.final_basis().to_exprs():
    ///         print(eq, "= 0")
    fn final_basis(&self, py: Python<'_>) -> PyResult<Option<Py<PyGroebnerBasis>>> {
        match &self.final_basis {
            None => Ok(None),
            Some(gb) => Ok(Some(Py::new(
                py,
                PyGroebnerBasis {
                    inner: gb.clone(),
                    pool: Some(self.pool.clone_ref(py)),
                    var_ids: self.var_ids.clone(),
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
    /// Jet variables for the Gröbner fallback; empty when Pantelides won.
    var_ids: Vec<ExprId>,
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
                    var_ids: self.var_ids.clone(),
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

/// `alkahest.rosenfeld_groebner(dae, order=None, max_prolong_rounds=None)` —
/// Rosenfeld–Gröbner-style differential elimination.
///
/// Prolongs the system (differentiates each equation, introducing higher jets
/// as new indeterminates) and computes a Gröbner basis after each round, until
/// differentiating adds nothing new to the ideal or the round budget runs out.
/// The basis is the set of *algebraic consequences* of the differential
/// system — the input–output relations elimination is after.
///
/// Parameters
/// ----------
/// dae : DAE
///     The system, polynomial in its variables and derivative symbols.
/// order : str, optional
///     Monomial order — ``"grevlex"`` (default), ``"grlex"`` or ``"lex"``.
///     Use ``"lex"`` when you want elimination-ordered generators.
/// max_prolong_rounds : int, optional
///     Prolongation budget (default 8).  Nonlinear jets often do not saturate
///     in finitely many algebraic steps, so hitting the budget is normal and
///     sets :attr:`RosenfeldGroebnerResult.truncated`.
///
/// Returns
/// -------
/// RosenfeldGroebnerResult
///     Read the relations with ``result.final_basis().to_exprs()``.
///
/// Example::
///
///     t, x, dx = p.symbol("t"), p.symbol("x"), p.symbol("dx/dt")
///     dae = alkahest.DAE.new([dx - x], [x], [dx], t)
///     r = alkahest.rosenfeld_groebner(dae, max_prolong_rounds=1)
///     r.consistent                      # True
///     r.final_basis().to_exprs()        # the eliminated relations, as Expr
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
        rosenfeld_groebner_ranked(
            &dae.inner,
            &pool.inner,
            py_monomial_order_for_dae(order),
            max_prolong_rounds.unwrap_or(8),
        )
    };
    let (r, ranking) = r.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyRosenfeldGroebnerResult {
        consistent: r.consistent,
        truncated: r.truncated,
        prolongation_rounds: r.prolongation_rounds,
        working_dae: r.working_dae,
        final_basis: r.final_basis,
        pool: pool_py,
        var_ids: ranking.vars,
    })
}

#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "dae_index_reduce", signature = (dae, order=None))]
fn py_dae_index_reduce(
    py: Python<'_>,
    dae: PyRef<PyDAE>,
    order: Option<&str>,
) -> PyResult<PyDaeIndexReduction> {
    let pool_py = dae.pool.clone_ref(py);
    let out = {
        let pool = pool_py.borrow(py);
        dae_index_reduce_ranked(&dae.inner, &pool.inner, py_monomial_order_for_dae(order))
    };
    let (inner, ranking) =
        out.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyDaeIndexReduction {
        inner,
        pool: pool_py,
        var_ids: ranking.map(|r| r.vars).unwrap_or_default(),
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
    let comps = primary_decomposition(gb_polys, MonomialOrder::Lex).map_err(ideal_error_to_py)?;
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

/// Radical √I of the ideal generated by `polys`.
///
/// If *vars* is omitted, free symbols across all *polys* are used (sorted by
/// internal id). At least one free symbol must be present.
#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "radical", signature = (polys, vars=None))]
fn py_ideal_radical(
    py: Python<'_>,
    polys: Vec<PyRef<PyExpr>>,
    vars: Option<Vec<PyRef<PyExpr>>>,
) -> PyResult<Py<PyGroebnerBasis>> {
    if polys.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "radical requires at least one polynomial",
        ));
    }
    let pool_py = polys[0].pool.clone_ref(py);
    let pool = pool_py.borrow(py);
    let var_ids: Vec<ExprId> = match vars {
        Some(v) => {
            if v.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "radical requires at least one variable",
                ));
            }
            v.iter().map(|v| v.id).collect()
        }
        None => {
            let mut set = std::collections::BTreeSet::new();
            for p in &polys {
                for v in alkahest_core::collect_free_vars(p.id, &pool.inner) {
                    set.insert(v);
                }
            }
            let ids: Vec<ExprId> = set.into_iter().collect();
            if ids.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "radical: no free symbols found to use as variables",
                ));
            }
            ids
        }
    };
    let mut gb_polys = Vec::with_capacity(polys.len());
    for p in &polys {
        let gbp = expr_to_gbpoly(p.id, &var_ids, &pool.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        gb_polys.push(gbp);
    }
    drop(pool);
    let gb = core_ideal_radical(gb_polys, MonomialOrder::Lex).map_err(ideal_error_to_py)?;
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
use alkahest_core::{
    diophantine as core_diophantine, solve_numerical, solve_polynomial_system,
    solve_transcendental, triangularize, CertifiedPoint, DiophantineError,
    DiophantineSolution as CoreDiophantineSolution, HomotopyError, HomotopyOpts, RegularChain,
    SolutionSet, TranscendentalOutcome,
};

#[cfg(feature = "groebner")]
fn homotopy_err_to_py(e: HomotopyError) -> PyErr {
    Python::with_gil(|py| match &e {
        HomotopyError::Algebraic(se) => {
            let exc = py.get_type_bound::<PySolverError>();
            make_structured_err(py, &exc, se)
        }
        _ => {
            let exc = py.get_type_bound::<PyHomotopyError>();
            make_structured_err(py, &exc, &e)
        }
    })
}

/// One homotopy endpoint with optional Smale diagnostics.
#[cfg(feature = "groebner")]
#[pyclass(name = "CertifiedSolution")]
struct PyCertifiedSolution {
    inner: CertifiedPoint,
    var_ids: Vec<ExprId>,
    pool: Py<PyExprPool>,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyCertifiedSolution {
    #[getter]
    fn coordinates(&self) -> Vec<f64> {
        self.inner.coordinates.clone()
    }

    #[getter]
    fn max_residual(&self) -> f64 {
        self.inner.max_residual_f64
    }

    #[getter]
    fn smale_alpha(&self) -> Option<f64> {
        self.inner.smale_alpha
    }

    #[getter]
    fn smale_certified(&self) -> bool {
        self.inner.smale_certified
    }

    /// Map each variable ``Expr`` to its coordinate ``float`` (same ``vars`` order as ``solve``).
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);
        for (i, &c) in self.inner.coordinates.iter().enumerate() {
            let vx = PyExpr {
                id: self.var_ids[i],
                pool: self.pool.clone_ref(py),
            };
            d.set_item(vx.into_py(py), c)?;
        }
        Ok(d.into())
    }

    fn enclosures(&self) -> Vec<PyArbBall> {
        self.inner
            .enclosure
            .iter()
            .map(|b| PyArbBall { inner: b.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CertifiedSolution(residual={:?}, certified={})",
            self.inner.max_residual_f64, self.inner.smale_certified,
        )
    }
}

/// Result of [`diophantine`](py_diophantine): a parametric linear family,
/// a finite list of integer points, or a Pell-type description.
#[cfg(feature = "groebner")]
#[pyclass(name = "DiophantineSolution")]
struct PyDiophantineSolution {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    parameter: Option<Py<PyExpr>>,
    /// ``x(t), y(t), …`` when ``kind == "parametric_linear"``.
    #[pyo3(get)]
    parametric: Option<Vec<Py<PyExpr>>>,
    /// List of coordinate tuples when ``kind == "finite"``.
    #[pyo3(get)]
    points: Option<Vec<Vec<Py<PyExpr>>>>,
    /// Coefficient ``D`` in ``x² - D·y² = 1`` or ``x² - D·y² = N``.
    #[pyo3(get)]
    pell_d: Option<Py<PyExpr>>,
    /// Fundamental unit ``(x0, y0)`` when ``kind == "pell_fundamental"``.
    #[pyo3(get)]
    fundamental: Option<(Py<PyExpr>, Py<PyExpr>)>,
    /// Right-hand ``N`` in ``x² - D·y² = N`` when ``kind == "pell_generalized"``.
    #[pyo3(get)]
    pell_n: Option<Py<PyExpr>>,
    /// A particular solution ``(x0, y0)`` when ``kind == "pell_generalized"``.
    #[pyo3(get)]
    pell_particular: Option<(Py<PyExpr>, Py<PyExpr>)>,
    /// Unit ``(ux, uy)`` with ``ux² - D·uy² = 1`` when ``kind == "pell_generalized"``.
    #[pyo3(get)]
    pell_unit: Option<(Py<PyExpr>, Py<PyExpr>)>,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyDiophantineSolution {
    fn __repr__(&self) -> String {
        format!("DiophantineSolution(kind={:?})", self.kind)
    }
}

#[cfg(feature = "groebner")]
fn diophantine_core_to_py(
    py: Python<'_>,
    sol: CoreDiophantineSolution,
    pool_py: Py<PyExprPool>,
) -> PyResult<PyDiophantineSolution> {
    let wrap = |id: ExprId| {
        Py::new(
            py,
            PyExpr {
                id,
                pool: pool_py.clone_ref(py),
            },
        )
    };
    match sol {
        CoreDiophantineSolution::ParametricLinear { parameter, values } => {
            let mut parametric = Vec::with_capacity(values.len());
            for id in values {
                parametric.push(wrap(id)?);
            }
            Ok(PyDiophantineSolution {
                kind: "parametric_linear".into(),
                parameter: Some(wrap(parameter)?),
                parametric: Some(parametric),
                points: None,
                pell_d: None,
                fundamental: None,
                pell_n: None,
                pell_particular: None,
                pell_unit: None,
            })
        }
        CoreDiophantineSolution::Finite(rows) => {
            let mut pts = Vec::with_capacity(rows.len());
            for row in rows {
                let mut pyrow = Vec::with_capacity(row.len());
                for id in row {
                    pyrow.push(wrap(id)?);
                }
                pts.push(pyrow);
            }
            Ok(PyDiophantineSolution {
                kind: "finite".into(),
                parameter: None,
                parametric: None,
                points: Some(pts),
                pell_d: None,
                fundamental: None,
                pell_n: None,
                pell_particular: None,
                pell_unit: None,
            })
        }
        CoreDiophantineSolution::PellFundamental { d, x0, y0 } => Ok(PyDiophantineSolution {
            kind: "pell_fundamental".into(),
            parameter: None,
            parametric: None,
            points: None,
            pell_d: Some(wrap(d)?),
            fundamental: Some((wrap(x0)?, wrap(y0)?)),
            pell_n: None,
            pell_particular: None,
            pell_unit: None,
        }),
        CoreDiophantineSolution::PellGeneralized {
            d,
            n,
            x0,
            y0,
            unit_x,
            unit_y,
        } => Ok(PyDiophantineSolution {
            kind: "pell_generalized".into(),
            parameter: None,
            parametric: None,
            points: None,
            pell_d: Some(wrap(d)?),
            fundamental: None,
            pell_n: Some(wrap(n)?),
            pell_particular: Some((wrap(x0)?, wrap(y0)?)),
            pell_unit: Some((wrap(unit_x)?, wrap(unit_y)?)),
        }),
        CoreDiophantineSolution::NoSolution => Ok(PyDiophantineSolution {
            kind: "no_solution".into(),
            parameter: None,
            parametric: None,
            points: None,
            pell_d: None,
            fundamental: None,
            pell_n: None,
            pell_particular: None,
            pell_unit: None,
        }),
    }
}

/// `alkahest.diophantine(equation, vars)` — integer solutions of a binary
/// Diophantine equation.
///
/// Solves `equation = 0` over the integers in exactly two unknowns.
/// Supported patterns:
///
/// * **Linear** — returns a parametric family ``x(t), y(t)``
///   (``kind == "parametric_linear"``).
/// * **Sum of two squares** — ``x² + y² = n`` (finite list of points).
/// * **Pell / generalized Pell** — ``x² − D·y² = 1`` or ``= N``.
///
/// Raises ``DiophantineError`` when the equation is not a polynomial in
/// *vars*, has non-integer coefficients, is an unsupported pattern, or has
/// no integer solution.
///
/// Example::
///
///     sol = diophantine(x + 2*y - 5, [x, y])
///     # sol.kind == "parametric_linear"; sol.parametric = [x(t), y(t)]
#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "diophantine", signature = (equation, vars))]
fn py_diophantine(
    py: Python<'_>,
    equation: PyRef<PyExpr>,
    vars: Vec<PyRef<PyExpr>>,
) -> PyResult<PyDiophantineSolution> {
    if vars.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "diophantine requires exactly two Expr variables",
        ));
    }
    let pool_py = equation.pool.clone_ref(py);
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let sol = {
        let pool = pool_py.borrow(py);
        core_diophantine(&pool.inner, equation.id, &var_ids)
    }
    .map_err(diophantine_error_to_py)?;
    diophantine_core_to_py(py, sol, pool_py)
}

#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "solve_numerical", signature = (
    equations,
    vars,
    *,
    max_bezout_paths=None,
    certify_prec_bits=None,
))]
fn py_solve_numerical(
    py: Python<'_>,
    equations: Vec<PyRef<PyExpr>>,
    vars: Vec<PyRef<PyExpr>>,
    max_bezout_paths: Option<usize>,
    certify_prec_bits: Option<u32>,
) -> PyResult<Vec<PyCertifiedSolution>> {
    if equations.is_empty() || vars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "solve_numerical requires at least one equation and one variable",
        ));
    }
    let pool_py = equations[0].pool.clone_ref(py);
    let eq_ids: Vec<ExprId> = equations.iter().map(|e| e.id).collect();
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    let mut opts = HomotopyOpts::default();
    if let Some(m) = max_bezout_paths {
        opts.max_bezout_paths = m;
    }
    if let Some(b) = certify_prec_bits {
        opts.certify_prec_bits = b;
    }
    let pts = {
        let pool = pool_py.borrow(py);
        solve_numerical(&eq_ids, &var_ids, &pool.inner, &opts)
    };
    match pts {
        Err(e) => Err(homotopy_err_to_py(e)),
        Ok(v) => Ok(v
            .into_iter()
            .map(|inner| PyCertifiedSolution {
                inner,
                var_ids: var_ids.clone(),
                pool: pool_py.clone_ref(py),
            })
            .collect()),
    }
}

/// `alkahest.solve(equations, vars, *, numeric=False, method="groebner")`
///
/// Solve a zero-dimensional polynomial system.
///
/// Parameters
/// ----------
/// equations : list[Expr]
///     Each expression represents `p(vars) = 0`.
/// vars : list[Expr]
///     Variables to solve for (symbols). Free symbols that appear in
///     *equations* but are omitted from *vars* are treated as parameters;
///     solutions may be expressions in those symbols (e.g. ``solve([x**2 - y],
///     [x])`` → ``±sqrt(y)``).
/// numeric : bool, default False
///     Used when ``method="groebner"``: symbolic ``Expr`` values vs ``float``.
///     When Lex back-substitution hits a degree > 2 univariate, ``numeric=True``
///     falls back to homotopy continuation (same as ``method="homotopy"``).
/// method : str, default ``"groebner"``
///     ``"groebner"`` — Lex basis + triangular back-substitution.
///     ``"homotopy"`` — total-degree homotopy continuation in ``ℂⁿ`` followed
///     by Newton projection to real tuples (always ``float`` dict values).
///
/// Returns
/// -------
/// list[dict]
///     Each dict maps a variable ``Expr`` to ``Expr`` (symbolic Groebner) or
///     ``float`` (Groebner with ``numeric=True``, or ``method="homotopy"``).
///     Solutions are a *set*: a double root is one entry, not two. Every
///     parameter-free tuple has been substituted back into the equations you
///     passed and could not be shown to violate them. A tuple containing a free
///     parameter is **not** a number and is returned unverified, under the
///     non-vanishing hypotheses reported by
///     :func:`alkahest.solve_side_conditions` — ``solve([a*x - b], [x])`` is
///     ``b/a`` *for* ``a ≠ 0``, and that condition is listed there.
/// GroebnerBasis
///     When ``method="groebner"`` and no finite solution list could be
///     produced — usually a positive-dimensional ideal, but also when the
///     Lex basis admits no complete triangular elimination in *vars*. It is
///     "here is the ideal" rather than a claim that the solutions are
///     infinite.
#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "solve", signature = (equations, vars, *, numeric = false, method = "groebner"))]
fn py_solve(
    py: Python<'_>,
    equations: Vec<PyRef<PyExpr>>,
    vars: Vec<PyRef<PyExpr>>,
    numeric: bool,
    method: &str,
) -> PyResult<PyObject> {
    if equations.is_empty() || vars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "solve requires at least one equation and one variable",
        ));
    }
    let pool_py = equations[0].pool.clone_ref(py);
    let eq_ids: Vec<ExprId> = equations.iter().map(|e| e.id).collect();
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.id).collect();
    alkahest_core::check_expr_depths(&pool_py.borrow(py).inner, &eq_ids)
        .map_err(depth_error_to_py)?;
    // `solve_side_conditions()` must describe *this* call, including the paths
    // below that never reach the symbolic solver (homotopy, transcendental).
    reset_solve_side_conditions();

    if method == "homotopy" {
        let opts = HomotopyOpts::default();
        let pts = {
            let pool = pool_py.borrow(py);
            solve_numerical(&eq_ids, &var_ids, &pool.inner, &opts)
        };
        return match pts {
            Err(e) => Err(homotopy_err_to_py(e)),
            Ok(points) => {
                let list = pyo3::types::PyList::empty_bound(py);
                for p in points {
                    let d = pyo3::types::PyDict::new_bound(py);
                    for (i, &val) in p.coordinates.iter().enumerate() {
                        let var_expr = PyExpr {
                            id: var_ids[i],
                            pool: pool_py.clone_ref(py),
                        };
                        d.set_item(var_expr.into_py(py), val)?;
                    }
                    list.append(d)?;
                }
                Ok(list.into())
            }
        };
    }

    if method != "groebner" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown solve method: {method:?} (expected 'groebner' or 'homotopy')"
        )));
    }

    // Transcendental pre-processing: for a single equation in a single unknown
    // containing exp/log/Lambert-W/trig, try the scoped closed-form solver before
    // handing off to the polynomial path (which would reject any transcendental).
    // On `Unsupported` we fall straight through to `solve_polynomial_system`.
    if eq_ids.len() == 1 && var_ids.len() == 1 {
        let trans = {
            let pool = pool_py.borrow(py);
            solve_transcendental(eq_ids[0], var_ids[0], &pool.inner)
        };
        if let TranscendentalOutcome::Solved(values) = trans {
            // Each value is a solution for the single variable.
            let solutions: Vec<Vec<ExprId>> = values.into_iter().map(|v| vec![v]).collect();
            let result: Result<SolutionSet, alkahest_core::SolverError> =
                Ok(SolutionSet::Finite(solutions));
            return finite_solutions_to_py(py, result, &pool_py, &var_ids, &var_ids, numeric);
        }
    }

    let result = {
        let pool = pool_py.borrow(py);
        let r = solve_polynomial_system(eq_ids.clone(), var_ids.clone(), &pool.inner);
        // Whatever the back-substitution had to assume about a parametric
        // leading coefficient, rendered while the pool is in hand — see
        // `py_solve_side_conditions`.
        capture_solve_side_conditions(&pool.inner);
        r
    };

    // B5: `numeric=True` means the caller accepts floats — when Lex back-substitution
    // hits a degree > 2 univariate, fall through to homotopy instead of raising.
    if numeric {
        if let Err(alkahest_core::SolverError::HighDegree(_)) = &result {
            let opts = HomotopyOpts::default();
            let pts = {
                let pool = pool_py.borrow(py);
                solve_numerical(&eq_ids, &var_ids, &pool.inner, &opts)
            };
            return match pts {
                Err(e) => Err(homotopy_err_to_py(e)),
                Ok(points) => {
                    let list = pyo3::types::PyList::empty_bound(py);
                    for p in points {
                        let d = pyo3::types::PyDict::new_bound(py);
                        for (i, &val) in p.coordinates.iter().enumerate() {
                            let var_expr = PyExpr {
                                id: var_ids[i],
                                pool: pool_py.clone_ref(py),
                            };
                            d.set_item(var_expr.into_py(py), val)?;
                        }
                        list.append(d)?;
                    }
                    Ok(list.into())
                }
            };
        }
    }

    // A `Parametric` basis is indexed by the solve variables *followed by* the
    // free parameters — the same concatenation `solve_polynomial_system` builds
    // its exponent vectors from. Without it the returned basis cannot be read.
    let basis_var_ids: Vec<ExprId> = {
        let pool = pool_py.borrow(py);
        let mut all = var_ids.clone();
        all.extend(alkahest_core::solver::collect_parameters(
            &eq_ids,
            &var_ids,
            &pool.inner,
        ));
        all
    };
    finite_solutions_to_py(py, result, &pool_py, &var_ids, &basis_var_ids, numeric)
}

#[cfg(feature = "groebner")]
thread_local! {
    /// Hypotheses recorded by the most recent `solve` on this thread, rendered
    /// against the pool that call used.
    static SOLVE_SIDE_CONDITIONS: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Reset both ends of the side-condition channel at the start of a `solve`.
#[cfg(feature = "groebner")]
fn reset_solve_side_conditions() {
    let _ = alkahest_core::solver::take_solve_side_conditions();
    SOLVE_SIDE_CONDITIONS.with(|c| c.borrow_mut().clear());
}

/// Render the hypotheses the core solver just assumed, for
/// [`py_solve_side_conditions`].
#[cfg(feature = "groebner")]
fn capture_solve_side_conditions(pool: &alkahest_core::ExprPool) {
    let rendered: Vec<String> = alkahest_core::solver::take_solve_side_conditions()
        .iter()
        .map(|c| c.display_with(pool).to_string())
        .collect();
    SOLVE_SIDE_CONDITIONS.with(|c| *c.borrow_mut() = rendered);
}

/// `alkahest.solve_side_conditions() -> list[str]`
///
/// The hypotheses the most recent :func:`alkahest.solve` on this thread
/// **assumed** in order to return the solutions it did — one string per
/// condition, e.g. ``"a ≠ 0"``.
///
/// ``solve([a*x - b], [x])`` returns ``b/a``, which is the solution *for
/// ``a ≠ 0``*: at ``a = 0`` the equation reads ``-b = 0``, so there is either no
/// solution (``b ≠ 0``) or every ``x`` (``b = 0``), and neither of those is
/// ``b/a``. That generic-parameter reading is deliberate and useful, but a
/// parametric tuple is not a number, so it is returned **unverified** — nothing
/// substitutes it back — and the hypothesis is the only signal a caller can
/// audit.
///
/// This mirrors ``DerivedResult.verification["side_conditions"]`` and
/// :attr:`alkahest.ZeilbergerCertificate.side_conditions`; ``solve`` returns
/// plain ``dict`` s, which cannot carry the attribute, so it is reported beside
/// the result instead.
///
/// An empty list means the solver *proved* every coefficient it divided by to
/// be non-zero — not that it did not look. Reset by each ``solve`` call, so
/// read it before the next one; repeated reads of the same call agree.
#[cfg(feature = "groebner")]
#[pyfunction]
#[pyo3(name = "solve_side_conditions")]
fn py_solve_side_conditions() -> Vec<String> {
    SOLVE_SIDE_CONDITIONS.with(|c| c.borrow().clone())
}

/// Shared formatting for a [`SolutionSet`] result into the Python return shape
/// (list of dicts, a `GroebnerBasis`, or a structured error).
#[cfg(feature = "groebner")]
fn finite_solutions_to_py(
    py: Python<'_>,
    result: Result<SolutionSet, alkahest_core::SolverError>,
    pool_py: &Py<PyExprPool>,
    var_ids: &[ExprId],
    basis_var_ids: &[ExprId],
    numeric: bool,
) -> PyResult<PyObject> {
    match result {
        Err(e) => Python::with_gil(|py2| {
            let exc_type = py2.get_type_bound::<PySolverError>();
            Err(make_structured_err(py2, &exc_type, &e))
        }),
        Ok(SolutionSet::NoSolution) => Ok(pyo3::types::PyList::empty_bound(py).into()),
        Ok(SolutionSet::Parametric(gb)) => Ok(PyGroebnerBasis {
            inner: gb,
            pool: Some(pool_py.clone_ref(py)),
            var_ids: basis_var_ids.to_vec(),
        }
        .into_py(py)),
        Ok(SolutionSet::Finite(solutions)) => {
            let list = pyo3::types::PyList::empty_bound(py);
            let pool = pool_py.borrow(py);
            // Symbolic solutions are simplified before hand-off. Sibling roots
            // (±√disc) and per-variable back-substitutions share large
            // subexpressions, so simplify them as one batch with a shared memo
            // rather than re-simplifying common subtrees once per value.
            let simplified: Vec<Vec<ExprId>> = if numeric {
                Vec::new()
            } else {
                let flat: Vec<ExprId> = solutions.iter().flatten().copied().collect();
                let mut out = core_simplify_batch(&flat, &pool.inner)
                    .into_iter()
                    .map(|d| d.value);
                solutions
                    .iter()
                    .map(|sol| (0..sol.len()).map(|_| out.next().unwrap()).collect())
                    .collect()
            };
            for (s, sol) in solutions.iter().enumerate() {
                let d = pyo3::types::PyDict::new_bound(py);
                for (i, val) in sol.iter().enumerate() {
                    let var_expr = PyExpr {
                        id: var_ids[i],
                        pool: pool_py.clone_ref(py),
                    };
                    if numeric {
                        let env: std::collections::HashMap<ExprId, f64> =
                            std::collections::HashMap::new();
                        let f = alkahest_core::jit::eval_interp(*val, &env, &pool.inner)
                            .unwrap_or(f64::NAN);
                        d.set_item(var_expr.into_py(py), f)?;
                    } else {
                        let val_expr = PyExpr {
                            id: simplified[s][i],
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
/// One triangular component of a decomposition, as returned by
/// :func:`triangularize`.
///
/// The component's polynomials are :class:`GbPoly`; each carries the variable
/// ordering :func:`triangularize` was called with, so ``p.to_expr()`` reads it
/// back as an :class:`Expr`.
#[pyclass(name = "RegularChain")]
struct PyRegularChain {
    inner: RegularChain,
    pool: Option<Py<PyExprPool>>,
    var_ids: Vec<ExprId>,
}

#[cfg(feature = "groebner")]
#[pymethods]
impl PyRegularChain {
    #[getter]
    fn n_vars(&self) -> usize {
        self.inner.n_vars
    }

    /// Gröbner-style polynomial tiles (``GbPoly``), ascending by main variable.
    ///
    /// Each tile knows the variables it is written over, so
    /// ``[p.to_expr() for p in chain.polys()]`` gives the triangular system as
    /// :class:`Expr` (see :meth:`to_exprs`).
    fn polys(&self, py: Python<'_>) -> Vec<PyGbPoly> {
        self.inner
            .polys
            .iter()
            .map(|p| PyGbPoly::with_ctx(py, p.clone(), self.pool.as_ref(), &self.var_ids))
            .collect()
    }

    /// The variables the chain is written over, in exponent-slot order.
    fn variables(&self, py: Python<'_>) -> Vec<PyExpr> {
        match &self.pool {
            None => vec![],
            Some(pool) => self
                .var_ids
                .iter()
                .map(|&id| PyExpr {
                    id,
                    pool: pool.clone_ref(py),
                })
                .collect(),
        }
    }

    /// The triangular system as :class:`Expr`, each meaning ``p = 0``.
    fn to_exprs(&self, py: Python<'_>) -> PyResult<Vec<PyExpr>> {
        let (pool_py, var_ids) =
            resolve_gb_ctx(py, self.pool.as_ref(), &self.var_ids, None, "RegularChain")?;
        let ids: Option<Vec<ExprId>> = {
            let pool = pool_py.borrow(py);
            self.inner
                .polys
                .iter()
                .map(|p| gbpoly_to_expr(p, &var_ids, &pool.inner))
                .collect()
        };
        ids.map(|ids| {
            ids.into_iter()
                .map(|id| PyExpr {
                    id,
                    pool: pool_py.clone_ref(py),
                })
                .collect()
        })
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "regular chain is over more variables than were named",
            )
        })
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
        triangularize(eq_ids, var_ids.clone(), &pool.inner)
    };

    match result {
        Err(e) => Python::with_gil(|py2| {
            let exc_type = py2.get_type_bound::<PySolverError>();
            // `triangularize` reports "this needs a splitting decomposition" as
            // `NotPolynomial` (the enum is public and exhaustive) and records the
            // real reason out of band. Recover it so the refusal raises its own
            // `E-SOLVE-004` rather than `E-SOLVE-001`, which means something else
            // entirely — a genuinely non-polynomial equation.
            if matches!(e, alkahest_core::SolverError::NotPolynomial(_)) {
                if let Some(r) = alkahest_core::solver::regular_chains::take_triangularize_refusal()
                {
                    return Err(make_structured_err(py2, &exc_type, &r));
                }
            }
            Err(make_structured_err(py2, &exc_type, &e))
        }),
        Ok(chains) => {
            let list = pyo3::types::PyList::empty_bound(py);
            for chain in chains {
                list.append(
                    PyRegularChain {
                        inner: chain,
                        pool: Some(pool_py.clone_ref(py)),
                        var_ids: var_ids.clone(),
                    }
                    .into_py(py),
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
    /// When set, carries the originating pool so variable names survive a
    /// `reduce_mod` → `lift_crt` round-trip (otherwise vars fall back to `x0`).
    pool: Option<Py<PyExprPool>>,
}

#[pymethods]
impl PyMultiPolyFp {
    /// True if this is the zero polynomial.
    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Highest total degree over all terms (`0` for the zero polynomial).
    #[getter]
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

// ---------------------------------------------------------------------------
// V3-1 — Integer number theory
// ---------------------------------------------------------------------------

/// Quadratic Dirichlet character modulo an odd square-free conductor \(q \ge 3\).
///
/// Matches the Jacobi symbol \((\\cdot \\mid q)\) on residues coprime to \(q\) (otherwise `0`).
#[pyclass(name = "DirichletChi", module = "alkahest")]
struct PyDirichletChi {
    inner: CoreQuadraticDirichlet,
}

#[pymethods]
impl PyDirichletChi {
    #[new]
    #[pyo3(text_signature = "(conductor:int)")]
    fn py_new(conductor: &Bound<'_, PyAny>) -> PyResult<Self> {
        let c = py_int_decimal(conductor)?;
        CoreQuadraticDirichlet::new(&c)
            .map(|inner| PyDirichletChi { inner })
            .map_err(number_theory_error_to_py)
    }

    #[getter]
    fn conductor(&self) -> String {
        self.inner.conductor()
    }

    fn eval(&self, n: &Bound<'_, PyAny>) -> PyResult<i32> {
        let ns = py_int_decimal(n)?;
        self.inner.eval(&ns).map_err(number_theory_error_to_py)
    }

    fn __repr__(&self) -> String {
        format!("DirichletChi({})", self.inner.conductor())
    }
}

/// Provable/prime-tested `bool` (`fmpz_is_prime`).
#[pyfunction]
#[pyo3(name = "nt_isprime", signature = (n))]
fn py_nt_isprime(n: &Bound<'_, PyAny>) -> PyResult<bool> {
    let s = py_int_decimal(n)?;
    nt_isprime(&s).map_err(number_theory_error_to_py)
}

/// Factorisation payload: `(sign, [(prime_decimal, exponent), ...])`.
#[pyfunction]
#[pyo3(name = "nt_factorint", signature = (n))]
fn py_nt_factorint(n: &Bound<'_, PyAny>) -> PyResult<(i32, Vec<(String, u64)>)> {
    let s = py_int_decimal(n)?;
    nt_factorint(&s).map_err(number_theory_error_to_py)
}

/// Next prime strictly after `n` (full proof when `proved` is `True`).
#[pyfunction]
#[pyo3(name = "nt_nextprime", signature = (n, proved = true))]
fn py_nt_nextprime(n: &Bound<'_, PyAny>, proved: bool) -> PyResult<String> {
    let s = py_int_decimal(n)?;
    nt_nextprime(&s, proved).map_err(number_theory_error_to_py)
}

/// Euler φ(`n`).
#[pyfunction]
#[pyo3(name = "nt_totient", signature = (n))]
fn py_nt_totient(n: &Bound<'_, PyAny>) -> PyResult<String> {
    let s = py_int_decimal(n)?;
    nt_totient(&s).map_err(number_theory_error_to_py)
}

/// Jacobi symbol (`a`|`n`).
#[pyfunction]
#[pyo3(name = "nt_jacobi", signature = (a, n))]
fn py_nt_jacobi(a: &Bound<'_, PyAny>, n: &Bound<'_, PyAny>) -> PyResult<i32> {
    let sa = py_int_decimal(a)?;
    let sn = py_int_decimal(n)?;
    nt_jacobi_symbol(&sa, &sn).map_err(number_theory_error_to_py)
}

/// Modular `k`th root modulo prime `p` (sqrt path or inversion when `\gcd(k,p-1)=1`).
#[pyfunction]
#[pyo3(name = "nt_nthroot_mod", signature = (a, k, p))]
fn py_nt_nthroot_mod(a: &Bound<'_, PyAny>, k: u64, p: &Bound<'_, PyAny>) -> PyResult<String> {
    let sa = py_int_decimal(a)?;
    let sp = py_int_decimal(p)?;
    nt_nthroot_mod(&sa, k, &sp).map_err(number_theory_error_to_py)
}

/// Smallest nonnegative `e` with `base**e ≡ residue \pmod modulus` (`modulus` prime).
#[pyfunction]
#[pyo3(name = "nt_discrete_log", signature = (residue, base, modulus))]
fn py_nt_discrete_log(
    residue: &Bound<'_, PyAny>,
    base: &Bound<'_, PyAny>,
    modulus: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let sr = py_int_decimal(residue)?;
    let sb = py_int_decimal(base)?;
    let sm = py_int_decimal(modulus)?;
    nt_discrete_log(&sr, &sb, &sm).map_err(number_theory_error_to_py)
}

/// Reduce a polynomial over ℤ to F_p = ℤ/pℤ.
///
/// Returns a `MultiPolyFp` with coefficients in [0, p).
/// Raises `ModularError` if `p` is not prime.
#[pyfunction]
#[pyo3(name = "modular_reduce")]
fn py_modular_reduce(poly: PyRef<PyMultiPoly>, p: u64) -> PyResult<PyMultiPolyFp> {
    core_reduce_mod(&poly.inner, p)
        .map(|fp| PyMultiPolyFp {
            inner: fp,
            pool: poly.pool.clone(),
        })
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
    // Preserve variable names from the first image's originating pool so the
    // reconstructed polynomial round-trips to the same string representation.
    let pool = polys.first().and_then(|p| p.pool.clone());
    core_lift_crt(&images)
        .map(|mp| PyMultiPoly { inner: mp, pool })
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
    let precision_bits = checked_prec(precision_bits)?;
    let list = constants
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("constants must be a list"))?;
    let n = list.len();
    let mut xs: Vec<Float> = Vec::with_capacity(n);
    for i in 0..n {
        let item = list.get_item(i)?;
        // Python `int` is checked *before* `f64`. `extract::<f64>()` succeeds
        // for an int and rounds it: `2**60 + 1` arrived as `2**60`, and
        // `guess_relation([2**60+1, 2**60, 1])` then returned `[-1, 1, 0]`,
        // whose residual over the values actually supplied is `-1`, not `0`
        // (the true relation is `[-1, 1, 1]`). `relation_confidence` reported
        // `credible=True` with `available_digits=inf`, because `_supplied_bits`
        // treats an int as exact — which it is, right up until this line threw
        // the low bits away. Ints take the same decimal-string route as
        // strings, so the two input forms mean the same thing.
        if item.is_instance_of::<pyo3::types::PyInt>() {
            let s = item.str()?.to_string();
            xs.push(
                Float::parse(s.trim())
                    .map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(
                            "could not parse integer constant as a floating constant",
                        )
                    })?
                    .complete(precision_bits),
            );
        } else if let Ok(v) = item.extract::<f64>() {
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

// ---------------------------------------------------------------------------
// Plot — SVG polyline and Graphviz DOT renderers
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "plot_svg", signature = (expr, var, lo, hi, width=640, height=400, n_pts=300, padding=10))]
#[allow(clippy::too_many_arguments)]
fn py_plot_svg(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    var: PyRef<PyExpr>,
    lo: f64,
    hi: f64,
    width: u32,
    height: u32,
    n_pts: usize,
    padding: u32,
) -> PyResult<String> {
    // `n_pts` reaches `Vec::with_capacity` in the renderer, so an unchecked
    // Python int is a capacity-overflow panic or an OOM kill, not an error.
    if n_pts > MAX_PLOT_POINTS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_pts must be at most {MAX_PLOT_POINTS} (got {n_pts})"
        )));
    }
    let pool_ref = expr.pool.borrow(py);
    guard_depth(&pool_ref.inner, expr.id)?;
    Ok(alkahest_core::render_svg_opts(
        &pool_ref.inner,
        expr.id,
        var.id,
        lo,
        hi,
        width,
        height,
        n_pts,
        padding,
    ))
}

#[pyfunction]
#[pyo3(name = "plot_dot")]
fn py_plot_dot(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<String> {
    let pool_ref = expr.pool.borrow(py);
    guard_depth(&pool_ref.inner, expr.id)?;
    Ok(alkahest_core::render_dot(&pool_ref.inner, expr.id))
}

// ---------------------------------------------------------------------------
// M6 — modular / p-adic evaluation of holonomic sequences
// ---------------------------------------------------------------------------

/// The result of evaluating a :class:`alkahest.ModularRecurrence`, with the
/// evidence that makes its residues trustworthy.
///
/// :attr:`singular_indices` is the field to read when a residue is surprising:
/// it lists the steps where the recurrence's leading coefficient was not a
/// unit mod ``p`` and the working precision had to absorb the loss.
#[pyclass(name = "ModularEvaluation")]
struct PyModularEvaluation {
    inner: CoreModularEvaluation,
}

#[pymethods]
impl PyModularEvaluation {
    /// The prime the evaluation ran at.
    #[getter]
    fn prime(&self) -> u64 {
        self.inner.prime()
    }

    /// ``k`` — the precision that was asked for, and delivered.
    #[getter]
    fn precision(&self) -> u32 {
        self.inner.precision()
    }

    /// ``K >= k`` — the precision the forward pass actually ran at.
    ///
    /// ``working_precision - precision`` is the total ``p``-adic precision lost
    /// to singular steps, and is ``0`` for a recurrence whose leading
    /// coefficient is a unit throughout.
    #[getter]
    fn working_precision(&self) -> u32 {
        self.inner.working_precision()
    }

    /// ``p**k``.
    #[getter]
    fn modulus(&self) -> u64 {
        self.inner.modulus()
    }

    /// How many singular steps there were, in total.
    #[getter]
    fn n_singular(&self) -> u64 {
        self.inner.n_singular()
    }

    /// How many forward steps the evaluation took.
    #[getter]
    fn steps(&self) -> u64 {
        self.inner.steps()
    }

    /// The residues, one per requested index, each in ``[0, p**k)``.
    fn residues(&self) -> Vec<u64> {
        self.inner.residues().to_vec()
    }

    /// Indices ``n`` where ``p`` divides the leading coefficient ``a_J(n)``.
    ///
    /// Truncated to the first 64; :attr:`n_singular` is the full count.
    fn singular_indices(&self) -> Vec<i64> {
        self.inner.singular_indices().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModularEvaluation(residues={:?}, prime={}, precision={}, \
             working_precision={}, n_singular={}, steps={})",
            self.inner.residues(),
            self.inner.prime(),
            self.inner.precision(),
            self.inner.working_precision(),
            self.inner.n_singular(),
            self.inner.steps()
        )
    }
}

/// Read one initial value as an exact rational ``(numerator, denominator)``.
///
/// ``int`` and :class:`fractions.Fraction` both carry ``.numerator`` /
/// ``.denominator``, so both work and nothing else does — in particular a
/// ``float`` is refused rather than converted. ``0.1`` is not one tenth, and a
/// sequence started from a binary approximation of the value you meant is a
/// different sequence, silently, because everything downstream is exact.
fn exact_rational_from_py(value: &Bound<'_, PyAny>, where_: &str) -> PyResult<(Integer, Integer)> {
    let (num, den) = match (value.getattr("numerator"), value.getattr("denominator")) {
        (Ok(n), Ok(d)) => (n, d),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "{where_} must be an exact rational (int or fractions.Fraction), \
                 got {}; a float cannot be one, and evaluating a recurrence from \
                 rounded initial values evaluates a different sequence",
                value.get_type().name()?
            )))
        }
    };
    Ok((big_integer_from_py(&num)?, big_integer_from_py(&den)?))
}

fn integer_poly_from_py(value: &Bound<'_, PyAny>, where_: &str) -> PyResult<Vec<Integer>> {
    let mut out = Vec::new();
    for (i, item) in value.iter()?.enumerate() {
        let item = item?;
        out.push(big_integer_from_py(&item).map_err(|_| {
            PyTypeError::new_err(format!(
                "{where_}[{i}] must be an int; coefficient polynomials are over Z, \
                 so clear denominators through the whole relation first"
            ))
        })?);
    }
    Ok(out)
}

/// A P-recursive recurrence prepared for evaluation modulo prime powers.
///
/// Holds ``Σ_{i=0}^{J} a_i(n)·S(n+i) = b(n)`` with integer polynomial
/// coefficients, plus the ``J`` initial values ``S(start) … S(start+J-1)``.
/// Nothing about ``p`` is fixed at construction, so one object serves an entire
/// supercongruence sweep.
///
/// ``coeffs[i]`` is ``a_i`` written **lowest-degree coefficient first** — the
/// same convention as :attr:`alkahest.GuessedRecurrence.coeffs`, so a fitted
/// recurrence is handed straight over.
///
/// Why this is Rust and not Python: it is exact modular arithmetic in a hot
/// loop over machine words, and the `p`-adic precision accounting that makes a
/// singular index safe has to happen inside that loop. Per ``CONTRIBUTING.md``
/// § *Rust vs Python*, points 2 and 5 of the Rust column.
///
/// **The recurrence is a hypothesis about your sequence.** This class checks
/// that it is well formed and that every forward step is determined
/// ``p``-adically; it cannot check that your sequence satisfies it. Certify
/// with :func:`alkahest.zeilberger`, or fit and confirm with
/// :func:`alkahest.guess_holonomic`.
///
/// >>> import alkahest as ak
/// >>> # Apéry A005259: (n+2)³A(n+2) = (34n³+153n²+231n+117)A(n+1) − (n+1)³A(n)
/// >>> apery = ak.ModularRecurrence(
/// ...     [[1, 3, 3, 1], [-117, -231, -153, -34], [8, 12, 6, 1]],
/// ...     [1, 5],
/// ... )
/// >>> apery.value_mod(12, 13, 3)              # A(p−1) ≡ 1 (mod p³), p = 13
/// 1
/// >>> apery.value_mod(10006, 10007, 3)        # …and at p = 10007, in ~5 ms
/// 1
#[pyclass(name = "ModularRecurrence")]
struct PyModularRecurrence {
    inner: CoreModularRecurrence,
}

#[pymethods]
impl PyModularRecurrence {
    #[new]
    #[pyo3(signature = (coeffs, initial, *, rhs = None, start = 0))]
    fn new(
        coeffs: &Bound<'_, PyAny>,
        initial: &Bound<'_, PyAny>,
        rhs: Option<&Bound<'_, PyAny>>,
        start: i64,
    ) -> PyResult<Self> {
        let mut polys = Vec::new();
        for (i, item) in coeffs.iter()?.enumerate() {
            polys.push(integer_poly_from_py(&item?, &format!("coeffs[{i}]"))?);
        }
        let rhs = match rhs {
            Some(r) => integer_poly_from_py(r, "rhs")?,
            None => Vec::new(),
        };
        let mut inits = Vec::new();
        for (j, item) in initial.iter()?.enumerate() {
            inits.push(exact_rational_from_py(&item?, &format!("initial[{j}]"))?);
        }
        let inner = CoreModularRecurrence::new(polys, rhs, inits, start)
            .map_err(holonomic_modular_error_to_py)?;
        Ok(Self { inner })
    }

    /// Recurrence order ``J``; ``len(coeffs()) == order + 1``.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// Largest degree of any coefficient polynomial, the right-hand side
    /// included.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Index ``n`` that ``initial[0]`` belongs to.
    #[getter]
    fn start(&self) -> i64 {
        self.inner.start()
    }

    /// Whether ``b(n)`` is identically zero.
    #[getter]
    fn is_homogeneous(&self) -> bool {
        self.inner.is_homogeneous()
    }

    /// ``[a_0, …, a_J]``, each a list of exact ints, lowest degree first.
    fn coeffs(&self, py: Python<'_>) -> PyResult<Vec<Vec<PyObject>>> {
        self.inner
            .coefficients()
            .iter()
            .map(|poly| integers_to_py(py, poly))
            .collect()
    }

    /// ``b(n)`` as a list of exact ints, lowest degree first; ``[]`` when
    /// homogeneous.
    fn rhs(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        integers_to_py(py, self.inner.inhomogeneity())
    }

    /// ``[S(start), …, S(start+J-1)]`` as ints or :class:`fractions.Fraction`.
    fn initial(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let int_cls = py.get_type_bound::<PyInt>();
        let fraction_cls = py.import_bound("fractions")?.getattr("Fraction")?;
        self.inner
            .initial_values()
            .iter()
            .map(|(num, den)| {
                let n = int_cls.call1((num.to_string(),))?;
                if *den == 1 {
                    Ok(n.into_py(py))
                } else {
                    let d = int_cls.call1((den.to_string(),))?;
                    Ok(fraction_cls.call1((n, d))?.into_py(py))
                }
            })
            .collect()
    }

    /// ``S(n) mod p**k``, computed from the recurrence without ever forming
    /// ``S(n)`` over the integers.
    ///
    /// :raises HolonomicError: ``E-HOLO-006`` when ``p**k`` is not a supported
    ///     modulus, ``E-HOLO-007`` when a step does not determine the next term
    ///     as a ``p``-adic integer, ``E-HOLO-008`` when the working precision
    ///     the singular steps demand is past the machine-word backend.
    fn value_mod(&self, n: i64, p: u64, k: u32) -> PyResult<u64> {
        self.inner
            .value_mod(n, p, k)
            .map_err(holonomic_modular_error_to_py)
    }

    /// ``[S(n) mod p**k for n in indices]``, in **one** forward pass.
    ///
    /// The indices are sorted internally and the residues come back in the
    /// order they were asked for, so evaluating a scattered set of indices
    /// costs one run to the largest of them rather than one run each.
    fn values_mod(&self, indices: Vec<i64>, p: u64, k: u32) -> PyResult<Vec<u64>> {
        let (sorted, back) = sorted_unique_with_index(&indices);
        let evaluation = self
            .inner
            .evaluate(&sorted, p, k)
            .map_err(holonomic_modular_error_to_py)?;
        Ok(back
            .iter()
            .map(|&slot| evaluation.residues()[slot])
            .collect())
    }

    /// Like :meth:`values_mod`, but returns the full
    /// :class:`alkahest.ModularEvaluation` with its precision accounting.
    fn evaluate(&self, indices: Vec<i64>, p: u64, k: u32) -> PyResult<PyModularEvaluation> {
        let (sorted, _) = sorted_unique_with_index(&indices);
        Ok(PyModularEvaluation {
            inner: self
                .inner
                .evaluate(&sorted, p, k)
                .map_err(holonomic_modular_error_to_py)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ModularRecurrence(order={}, degree={}, start={}, homogeneous={})",
            self.inner.order(),
            self.inner.degree(),
            self.inner.start(),
            self.inner.is_homogeneous()
        )
    }
}

fn integers_to_py(py: Python<'_>, values: &[Integer]) -> PyResult<Vec<PyObject>> {
    let int_cls = py.get_type_bound::<PyInt>();
    values
        .iter()
        .map(|c| Ok(int_cls.call1((c.to_string(),))?.into_py(py)))
        .collect()
}

/// Sort and de-duplicate, returning the sorted indices and, for each original
/// position, the slot it landed in.
fn sorted_unique_with_index(indices: &[i64]) -> (Vec<i64>, Vec<usize>) {
    let mut sorted: Vec<i64> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let back = indices
        .iter()
        .map(|n| sorted.partition_point(|s| s < n))
        .collect();
    (sorted, back)
}

/// ``binomial(a, b) mod p**k``, exactly, for ``p`` prime.
///
/// Uses the Andrew Granville / Davis–Webb factorisation of ``n!`` into its
/// ``p``-free part, which at ``k = 1`` is Lucas' theorem exactly. The cost is
/// ``O(p·k³ + log_p(a)·p·k)`` and does not grow with ``a`` beyond the
/// logarithm, so ``a`` far larger than ``p`` is the ordinary case rather than
/// the hard one.
///
/// ``b < 0`` and ``b > a`` are not errors: the binomial coefficient is ``0``.
///
/// :raises HolonomicError: ``E-HOLO-006`` when ``p`` is not prime, ``k < 1``,
///     or ``p**k >= 2**62``; ``E-HOLO-008`` when the work budget would be
///     exceeded.
///
/// >>> import alkahest as ak
/// >>> ak.binomial_mod(2 * 11 - 1, 10, 11, 3)   # Wolstenholme
/// 1
/// >>> ak.binomial_mod(1_000_000, 3, 7, 4)
/// 2261
/// >>> ak.binomial_mod(5, 9, 7, 4)
/// 0
#[pyfunction]
#[pyo3(name = "binomial_mod", signature = (a, b, p, k))]
fn py_binomial_mod(a: u64, b: i128, p: u64, k: u32) -> PyResult<u64> {
    core_binomial_mod(a, b, p, k).map_err(holonomic_modular_error_to_py)
}

#[pymodule]
fn alkahest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(py_derived_result_context_simplify, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_egraph, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_egraph_with, m)?)?;
    m.add_class::<PyEgraphConfig>()?;
    m.add_function(wrap_pyfunction!(py_simplify_with, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_expanded, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_trig, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_trig_normal_form, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_log_exp, m)?)?;
    m.add_function(wrap_pyfunction!(py_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(py_diff, m)?)?;
    m.add_function(wrap_pyfunction!(py_diff_forward, m)?)?;
    m.add_function(wrap_pyfunction!(py_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(py_integrate_definite, m)?)?;
    m.add_function(wrap_pyfunction!(py_apart, m)?)?;
    m.add_function(wrap_pyfunction!(py_residue, m)?)?;
    m.add_function(wrap_pyfunction!(py_series, m)?)?;
    m.add_function(wrap_pyfunction!(py_limit, m)?)?;
    m.add_function(wrap_pyfunction!(py_sum_indefinite, m)?)?;
    m.add_function(wrap_pyfunction!(py_sum_definite, m)?)?;
    m.add_function(wrap_pyfunction!(py_product_indefinite, m)?)?;
    m.add_function(wrap_pyfunction!(py_product_definite, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve_linear_recurrence_homogeneous, m)?)?;
    m.add_function(wrap_pyfunction!(py_rsolve, m)?)?;
    m.add_function(wrap_pyfunction!(py_verify_wz_pair, m)?)?;
    // P1 item 10 — asymptotic expansion at scale
    m.add_class::<PyAsymptoticReport>()?;
    m.add_function(wrap_pyfunction!(py_euler_maclaurin, m)?)?;
    m.add_function(wrap_pyfunction!(py_coefficient_asymptotics, m)?)?;
    // P1 item 7 — creative telescoping / holonomic (D-finite) machinery
    m.add_class::<PyZeilbergerCertificate>()?;
    m.add_function(wrap_pyfunction!(py_zeilberger, m)?)?;
    // M4(b) — q-analogue creative telescoping
    m.add_class::<PyQZeilbergerCertificate>()?;
    m.add_class::<PyQRootOfUnitySpecialization>()?;
    m.add_function(wrap_pyfunction!(py_q_zeilberger, m)?)?;
    m.add_function(wrap_pyfunction!(py_cyclotomic_polynomial, m)?)?;
    // M4 — double-sum (Apagodu–Zeilberger) creative telescoping
    m.add_class::<PyTelescoping2dCertificate>()?;
    m.add_function(wrap_pyfunction!(py_telescope2d, m)?)?;
    // M4 extension — arbitrary-m-bound-index generalization
    m.add_class::<PyTelescopingMdCertificate>()?;
    m.add_function(wrap_pyfunction!(py_telescope_md, m)?)?;
    // M6 — modular / p-adic evaluation of holonomic sequences
    m.add_class::<PyModularRecurrence>()?;
    m.add_class::<PyModularEvaluation>()?;
    m.add_function(wrap_pyfunction!(py_binomial_mod, m)?)?;
    // M5 — recurrence -> asymptotics (Poincaré–Perron)
    m.add_class::<PyRecurrenceAsymptotics>()?;
    m.add_function(wrap_pyfunction!(py_asymptotics_from_recurrence, m)?)?;
    m.add_function(wrap_pyfunction!(match_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(make_rule, m)?)?;
    m.add_function(wrap_pyfunction!(py_subs, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(conjugate, m)?)?;
    m.add_function(wrap_pyfunction!(re, m)?)?;
    m.add_function(wrap_pyfunction!(im, m)?)?;
    m.add_function(wrap_pyfunction!(arg, m)?)?;
    // V1-12: expanded primitive registry
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(asin, m)?)?;
    m.add_function(wrap_pyfunction!(acos, m)?)?;
    m.add_function(wrap_pyfunction!(atan, m)?)?;
    m.add_function(wrap_pyfunction!(asinh, m)?)?;
    m.add_function(wrap_pyfunction!(acosh, m)?)?;
    m.add_function(wrap_pyfunction!(atanh, m)?)?;
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(erfc, m)?)?;
    m.add_function(wrap_pyfunction!(abs_expr, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(round_expr, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(lambert_w, m)?)?;
    m.add_function(wrap_pyfunction!(digamma, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_j0, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_j1, m)?)?;
    m.add_function(wrap_pyfunction!(heaviside, m)?)?;
    m.add_function(wrap_pyfunction!(dirac_delta, m)?)?;
    // Experimental calculus / ODE / transform surface (PRs #152–#161).
    m.add_function(wrap_pyfunction!(py_dsolve, m)?)?;
    m.add_function(wrap_pyfunction!(py_laplace_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_inverse_laplace_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_fourier_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_inverse_fourier_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_z_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_inverse_z_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_multilimit, m)?)?;
    m.add_function(wrap_pyfunction!(py_asymptotic_expand, m)?)?;
    m.add_function(wrap_pyfunction!(py_series_solve, m)?)?;
    m.add_class::<PyFps>()?;
    m.add_function(wrap_pyfunction!(atan2, m)?)?;
    m.add_function(wrap_pyfunction!(min_expr, m)?)?;
    m.add_function(wrap_pyfunction!(max_expr, m)?)?;
    m.add_function(wrap_pyfunction!(elliptic_k, m)?)?;
    m.add_function(wrap_pyfunction!(elliptic_e, m)?)?;
    m.add_function(wrap_pyfunction!(elliptic_f, m)?)?;
    m.add_function(wrap_pyfunction!(elliptic_pi, m)?)?;
    m.add_class::<PyDomain>()?;
    m.add_class::<PyExprPool>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyEvaluationResult>()?;
    m.add_class::<PyAssumptions>()?;
    m.add_class::<PyDerivedResult>()?;
    m.add_class::<PySeries>()?;
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
    // Phase 16b — numeric ODE integrators
    m.add_class::<PyOdeTrajectory>()?;
    m.add_function(wrap_pyfunction!(py_ode_integrate_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(py_ode_integrate_rk45, m)?)?;
    // Phase 17
    m.add_class::<PyDAE>()?;
    m.add_function(wrap_pyfunction!(py_pantelides, m)?)?;
    // Phase 18
    m.add_class::<PyPort>()?;
    m.add_class::<PyComponent>()?;
    m.add_class::<PyAcausalSystem>()?;
    m.add_function(wrap_pyfunction!(py_resistor, m)?)?;
    m.add_function(wrap_pyfunction!(py_capacitor, m)?)?;
    m.add_function(wrap_pyfunction!(py_voltage_source, m)?)?;
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
    m.add_function(wrap_pyfunction!(py_jit_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(py_build_features, m)?)?;
    m.add_class::<PyCompiledFn>()?;
    m.add_class::<PyCompileCache>()?;
    // Phase 22 — Ball arithmetic
    m.add_class::<PyArbBall>()?;
    m.add_function(wrap_pyfunction!(py_interval_eval, m)?)?;
    // Phase 23 — Parallel simplification
    m.add_function(wrap_pyfunction!(py_simplify_par, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_redex, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_auto, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_strategy, m)?)?;
    // Phase 24 — Horner form
    m.add_function(wrap_pyfunction!(py_horner, m)?)?;
    m.add_function(wrap_pyfunction!(py_emit_c, m)?)?;
    // Transcendental C emission (general DAG walker)
    m.add_function(wrap_pyfunction!(py_emit_c_expr, m)?)?;
    m.add_function(wrap_pyfunction!(py_emit_c_vec, m)?)?;
    // Phase 26 — collect_like_terms
    m.add_function(wrap_pyfunction!(py_collect_like_terms, m)?)?;
    // V3-2 — Pauli / Clifford (non-commutative helpers)
    m.add_function(wrap_pyfunction!(py_simplify_pauli, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_clifford_orthogonal, m)?)?;
    // Phase 27 — poly_normal
    m.add_function(wrap_pyfunction!(py_poly_normal, m)?)?;
    // Rational-function cancel/together
    m.add_function(wrap_pyfunction!(py_cancel, m)?)?;
    m.add_function(wrap_pyfunction!(py_together, m)?)?;
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
    // P1 item 9 — rigorous global bounds (Taylor models / validated numerics)
    m.add_class::<PyEnclosure>()?;
    m.add_class::<PyBoundsSupport>()?;
    m.add_function(wrap_pyfunction!(py_bounds_supported, m)?)?;
    m.add_function(wrap_pyfunction!(py_bound_on_box, m)?)?;
    m.add_function(wrap_pyfunction!(py_verified_integral, m)?)?;
    m.add_function(wrap_pyfunction!(py_verified_no_roots, m)?)?;
    m.add_function(wrap_pyfunction!(py_verified_sign, m)?)?;
    // P1 item 8 — positivity certificates (SOS / Positivstellensatz)
    m.add_class::<PyPositivityCertificate>()?;
    m.add_function(wrap_pyfunction!(py_sos_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(py_prove_nonneg, m)?)?;
    m.add_function(wrap_pyfunction!(py_cad_lift, m)?)?;
    m.add_function(wrap_pyfunction!(py_routh_hurwitz, m)?)?;
    // V5-1 — Lean 4 certificate exporter
    m.add_function(wrap_pyfunction!(py_to_lean, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_smtlib, m)?)?;
    // V5-2 — StableHLO/XLA bridge
    m.add_function(wrap_pyfunction!(py_to_stablehlo, m)?)?;
    // V5-3 — NVPTX JIT backend
    #[cfg(feature = "cuda")]
    {
        m.add_class::<PyCudaCompiledFn>()?;
        m.add_function(wrap_pyfunction!(py_compile_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(py_cuda_device_count, m)?)?;
    }
    // V5-11 — Gröbner basis / V1-16 — GroebnerBasis.compute
    #[cfg(feature = "groebner")]
    {
        m.add_class::<PyGbPoly>()?;
        m.add_class::<PyGroebnerBasis>()?;
        // M9 — coefficient fields for elimination
        m.add_class::<PyParamGbPoly>()?;
        m.add_class::<PyParamGroebnerBasis>()?;
        m.add_class::<PyRosenfeldGroebnerResult>()?;
        m.add_class::<PyDaeIndexReduction>()?;
        m.add_class::<PyPrimaryComponent>()?;
        m.add_class::<PyRegularChain>()?;
        m.add_class::<PyCertifiedSolution>()?;
        m.add_class::<PyDiophantineSolution>()?;
        m.add_function(wrap_pyfunction!(py_solve_numerical, m)?)?;
        m.add_function(wrap_pyfunction!(py_diophantine, m)?)?;
        m.add_function(wrap_pyfunction!(py_solve, m)?)?;
        m.add_function(wrap_pyfunction!(py_solve_side_conditions, m)?)?;
        m.add_function(wrap_pyfunction!(py_triangularize, m)?)?;
        m.add_function(wrap_pyfunction!(py_primary_decomposition, m)?)?;
        m.add_function(wrap_pyfunction!(py_ideal_radical, m)?)?;
        m.add_function(wrap_pyfunction!(py_rosenfeld_groebner, m)?)?;
        m.add_function(wrap_pyfunction!(py_dae_index_reduce, m)?)?;
        m.add_function(wrap_pyfunction!(py_expr_to_gbpoly, m)?)?;
    }
    // V2-2 — Resultants and subresultant PRS
    m.add_function(wrap_pyfunction!(py_resultant, m)?)?;
    m.add_function(wrap_pyfunction!(py_subresultant_prs, m)?)?;
    // V2-3 — Sparse interpolation and sparse modular GCD
    m.add_function(wrap_pyfunction!(py_sparse_interp_univariate, m)?)?;
    m.add_function(wrap_pyfunction!(py_sparse_interp, m)?)?;
    m.add_function(wrap_pyfunction!(py_gcd_sparse, m)?)?;
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
    // V3-1 — Integer number theory
    m.add_class::<PyDirichletChi>()?;
    m.add_function(wrap_pyfunction!(py_nt_isprime, m)?)?;
    m.add_function(wrap_pyfunction!(py_nt_factorint, m)?)?;
    m.add_function(wrap_pyfunction!(py_nt_nextprime, m)?)?;
    m.add_function(wrap_pyfunction!(py_nt_totient, m)?)?;
    m.add_function(wrap_pyfunction!(py_nt_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(py_nt_nthroot_mod, m)?)?;
    m.add_function(wrap_pyfunction!(py_nt_discrete_log, m)?)?;
    // Plot — SVG and DOT renderers
    m.add_function(wrap_pyfunction!(py_plot_svg, m)?)?;
    m.add_function(wrap_pyfunction!(py_plot_dot, m)?)?;
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
        "AssumptionError",
        m.py().get_type_bound::<PyAssumptionError>(),
    )?;
    m.add(
        "DepthLimitError",
        m.py().get_type_bound::<PyDepthLimitError>(),
    )?;
    m.add(
        "IntegrationError",
        m.py().get_type_bound::<PyIntegrationError>(),
    )?;
    m.add("SeriesError", m.py().get_type_bound::<PySeriesError>())?;
    m.add("LimitError", m.py().get_type_bound::<PyLimitError>())?;
    m.add("MatrixError", m.py().get_type_bound::<PyMatrixError>())?;
    m.add("EigenError", m.py().get_type_bound::<PyEigenError>())?;
    m.add(
        "LinearAlgebraError",
        m.py().get_type_bound::<PyLinearAlgebraError>(),
    )?;
    m.add("ModularError", m.py().get_type_bound::<PyModularError>())?;
    m.add("OdeError", m.py().get_type_bound::<PyOdeError>())?;
    m.add("DaeError", m.py().get_type_bound::<PyDaeError>())?;
    m.add("JitError", m.py().get_type_bound::<PyJitError>())?;
    m.add("SolverError", m.py().get_type_bound::<PySolverError>())?;
    m.add("HomotopyError", m.py().get_type_bound::<PyHomotopyError>())?;
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
    m.add(
        "SparseGcdError",
        m.py().get_type_bound::<PySparseGcdError>(),
    )?;
    m.add("RealRootError", m.py().get_type_bound::<PyRealRootError>())?;
    m.add("LatticeError", m.py().get_type_bound::<PyLatticeError>())?;
    m.add("PslqError", m.py().get_type_bound::<PyPslqError>())?;
    m.add("CadError", m.py().get_type_bound::<PyCadError>())?;
    // P1 item 9 — rigorous global bounds (Taylor models / validated numerics)
    m.add(
        "ValidatedError",
        m.py().get_type_bound::<PyValidatedError>(),
    )?;
    // P1 item 8 — positivity certificates (SOS / Positivstellensatz)
    m.add("SosError", m.py().get_type_bound::<PySosError>())?;
    m.add("SumError", m.py().get_type_bound::<PySumError>())?;
    m.add("ProductError", m.py().get_type_bound::<PyProductError>())?;
    // P1 item 7 — creative telescoping / holonomic (D-finite) machinery
    m.add(
        "HolonomicError",
        m.py().get_type_bound::<PyHolonomicError>(),
    )?;
    m.add(
        "NumberTheoryError",
        m.py().get_type_bound::<PyNumberTheoryError>(),
    )?;
    m.add(
        "LinearRecurrenceError",
        m.py().get_type_bound::<PyLinearRecurrenceError>(),
    )?;
    m.add("RsolveError", m.py().get_type_bound::<PyRsolveError>())?;
    #[cfg(feature = "groebner")]
    m.add(
        "DiophantineError",
        m.py().get_type_bound::<PyDiophantineError>(),
    )?;
    #[cfg(feature = "groebner")]
    m.add(
        "ParamGroebnerError",
        m.py().get_type_bound::<PyParamGroebnerError>(),
    )?;
    // P1 search plumbing item 4 — budgets, cancellation, determinism
    m.add(
        "BudgetExceededError",
        m.py().get_type_bound::<PyBudgetExceededError>(),
    )?;
    m.add_function(wrap_pyfunction!(py_push_budget, m)?)?;
    m.add_function(wrap_pyfunction!(py_note_context_push, m)?)?;
    m.add_function(wrap_pyfunction!(py_note_context_pop, m)?)?;
    m.add_function(wrap_pyfunction!(py_pop_budget, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_budget_active, m)?)?;
    m.add_function(wrap_pyfunction!(py_budget_seed, m)?)?;
    m.add_function(wrap_pyfunction!(py_request_cancel, m)?)?;
    m.add_function(wrap_pyfunction!(py_clear_cancel, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_cancelled, m)?)?;
    // V1-15: compile-time flag so Python tests can skip egraph-dependent assertions.
    m.add("HAS_EGRAPH", cfg!(feature = "egraph"))?;
    // P1 search-plumbing item 6: versioned DerivedResult.to_dict/to_json envelope.
    m.add("RESULT_SCHEMA_VERSION", RESULT_SCHEMA_VERSION)?;
    m.add("STEPS_SCHEMA_VERSION", STEPS_SCHEMA_VERSION)?;
    Ok(())
}
