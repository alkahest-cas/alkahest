import contextlib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _meta_version

from . import (
    lattice,
    modular,  # noqa: F401 — V2-1: expose alkahest.modular submodule
    number_theory,
)
from ._context import (
    active_domain,
    active_pool,
    context,
    get_context_value,
    simplify_enabled,
    symbol,
)
from ._dlpack import _to_numpy
from ._parse import parse
from ._pretty import latex, unicode_str
from ._product import Product
from ._pytree import (
    TreeDef,
    flatten_exprs,
    map_exprs,
    unflatten_exprs,
)
from ._transform import (
    CompiledGradTracedFn,
    CompiledTracedFn,
    GradTracedFn,
    TracedFn,
    grad,
    jit,
    trace,
    trace_fn,
)
from .alkahest import (  # noqa: F401
    # Phase 17: DAE
    DAE,
    HAS_EGRAPH,
    # Phase 16: ODE
    ODE,
    # Phase 18: Acausal modelling
    AcausalSystem,
    # V3-3: First-order logic + V2-9 CAD/QE
    And,
    # Phase 22: Ball arithmetic
    ArbBall,
    CompileCache,
    # Phase 21: JIT compiled evaluation
    CompiledFn,
    # Core expression types
    DerivedResult,
    # V1-15: EgraphConfig and simplify_egraph_with
    EgraphConfig,
    # Phase 20: Hybrid systems
    Event,
    Exists,
    Expr,
    ExprPool,
    Forall,
    HybridODE,
    # Phase 15: Symbolic matrices
    Matrix,
    # Polynomial types
    MultiPoly,
    MultiPolyFactorization,
    # V2-1: Modular / CRT framework
    MultiPolyFp,
    Not,
    Or,
    Port,
    # PA-5: Primitive registry
    PrimitiveRegistry,
    RationalFunction,
    # Rewrite rules
    RewriteRule,
    # V2-4: Real root isolation (VAS)
    RootInterval,
    # Phase 19: Sensitivity analysis
    SensitivitySystem,
    Series,
    UniPoly,
    UniPolyFactorization,
    # V2-7: Polynomial factorization
    UniPolyFactorModP,
    abs,  # symbolic abs — use alkahest.abs(expr); shadows Python builtin within this module
    acos,
    adjoint_system,
    asin,
    atan,
    atan2,
    cad_lift,
    cad_project,
    ceil,
    # Phase 26: collect_like_terms
    collect_like_terms,
    compile_expr,
    # Math functions
    cos,
    cosh,
    decide,
    # Core operations
    diff,
    diff_forward,
    emit_c,
    erf,
    erfc,
    eval_expr,
    exp,
    factor_univariate_mod_p,
    floor,
    gamma,
    # V2-3: Sparse interpolation and sparse modular GCD
    gcd_sparse,
    # V2-6: Approximate integer relations (LLL-backed heuristic — not Ferguson–Bailey PSLQ)
    guess_relation,
    # Phase 24: Horner-form code emission
    horner,
    integrate,
    interval_eval,
    jacobian,
    jit_is_available,
    limit,
    log,
    lower_to_first_order,
    make_rule,
    match_pattern,
    max,  # symbolic max(a, b) — shadows Python builtin within this module
    min,  # symbolic min(a, b) — shadows Python builtin within this module
    pantelides,
    # PA-9: Piecewise
    piecewise,
    # Phase 27: poly_normal
    poly_normal,
    product_definite,
    product_indefinite,
    real_roots,
    refine_root,
    resistor,
    # V2-2: Resultants and subresultant PRS
    resultant,
    round,  # symbolic round — use alkahest.round(expr)
    rsolve,
    satisfiable,
    sensitivity_system,
    series,
    sign,
    simplify,
    simplify_clifford_orthogonal,
    simplify_egraph,
    simplify_egraph_with,
    simplify_expanded,
    simplify_log_exp,
    simplify_par,
    simplify_pauli,
    simplify_trig,
    simplify_with,
    sin,
    sinh,
    solve_linear_recurrence_homogeneous,
    sparse_interp,
    sparse_interp_univariate,
    sqrt,
    subresultant_prs,
    subs,
    sum_definite,
    sum_indefinite,
    # V1-12: expanded primitive registry
    tan,
    tanh,
    # V5-1: Lean 4 certificate exporter
    to_lean,
    # V5-2: StableHLO/XLA bridge
    to_stablehlo,
    verify_wz_pair,
    version,
)
from .alkahest import (
    # Phase 14: Reverse-mode AD (symbolic gradient; for traced-fn gradient use alkahest.grad)
    grad as symbolic_grad,
)

# V5-7: JAX primitive integration (optional — requires JAX)
with contextlib.suppress(ImportError):
    from ._jax import to_jax  # noqa: F401

# V1-4 / V1-16: Polynomial system solver + Gröbner basis (optional — requires groebner feature)
with contextlib.suppress(ImportError):
    from .alkahest import (
        CertifiedSolution,
        DaeIndexReduction,
        DiophantineSolution,
        GbPoly,
        GroebnerBasis,
        RegularChain,
        RosenfeldGroebnerResult,
        dae_index_reduce,
        diophantine,
        rosenfeld_groebner,
        solve,
        solve_numerical,
        triangularize,
    )

# V1-3 — Structured exception hierarchy: bind pure-Python stubs first, then replace
# each name with the compiled class when ``alkahest.alkahest`` exports it.  A bulk
# ``from .alkahest import (..., DiophantineError, ...)`` fails entirely when optional
# features omit a symbol (``DiophantineError`` is registered only with the groebner
# feature); that used to downgrade *every* error class to stubs and drop names that
# have no stub (CadError, …).
from .exceptions import (
    AlkahestError,
    CadError,
    ConversionError,
    DaeError,
    DiffError,
    DiophantineError,
    DomainError,
    EigenError,
    FactorError,
    IntegrationError,
    IoError,
    JitError,
    LatticeError,
    LimitError,
    LinearRecurrenceError,
    MatrixError,
    NumberTheoryError,
    OdeError,
    ParseError,
    PoolError,
    ProductError,
    PslqError,
    RealRootError,
    ResultantError,
    RsolveError,
    SeriesError,
    SolverError,
    SparseGcdError,
    SparseInterpError,
    SumError,
)

_NATIVE_EXCEPTION_OVERLAY: tuple[str, ...] = (
    "AlkahestError",
    "CadError",
    "ConversionError",
    "DaeError",
    "DiffError",
    "DiophantineError",
    "DomainError",
    "EigenError",
    "FactorError",
    "IntegrationError",
    "IoError",
    "JitError",
    "LatticeError",
    "LimitError",
    "LinearRecurrenceError",
    "MatrixError",
    "NumberTheoryError",
    "OdeError",
    "PoolError",
    "ProductError",
    "PslqError",
    "RealRootError",
    "ResultantError",
    "RsolveError",
    "SeriesError",
    "SolverError",
    "SparseGcdError",
    "SparseInterpError",
)

try:
    from . import alkahest as _alkahest_native
except ImportError:
    pass
else:
    _globals = globals()
    for _overlay_name in _NATIVE_EXCEPTION_OVERLAY:
        _native_cls = getattr(_alkahest_native, _overlay_name, None)
        if _native_cls is not None:
            _globals[_overlay_name] = _native_cls

try:
    __version__ = _meta_version("alkahest")
except _PackageNotFoundError:
    __version__ = "unknown"


def numpy_eval(compiled_fn, *arrays):
    """Vectorised evaluation of a :class:`CompiledFn` over arrays.

    Phase 25 / PA-8 — NumPy / JAX / PyTorch / DLPack array evaluation.

    Parameters
    ----------
    compiled_fn : CompiledFn
        A compiled function returned by :func:`compile_expr`.
    *arrays : array-like
        One array per input variable.  All arrays must have the same number
        of elements.  Accepts NumPy arrays, PyTorch CPU tensors, JAX arrays,
        CuPy arrays (host copy made), or anything with ``__dlpack__`` or
        ``__array__``.

    Returns
    -------
    numpy.ndarray
        Output values with the same shape as the first input array.

    Example
    -------
    >>> import numpy as np
    >>> import alkahest
    >>> p = alkahest.ExprPool()
    >>> x = p.symbol("x")
    >>> f = alkahest.compile_expr(x ** 2, [x])
    >>> xs = np.linspace(0, 1, 1_000_000)
    >>> ys = alkahest.numpy_eval(f, xs)   # vectorised; ≥100× faster than a loop
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy_eval requires NumPy.  Install it with: pip install numpy") from exc

    n_vars = compiled_fn.n_inputs
    if len(arrays) != n_vars:
        raise ValueError(f"expected {n_vars} input array(s), got {len(arrays)}")

    # Use DLPack-aware conversion for each input.
    first_raw = np.asarray(arrays[0])
    out_shape = first_raw.shape if first_raw.ndim > 0 else ()

    flat_arrays = [_to_numpy(a).ravel() for a in arrays]
    n_points = flat_arrays[0].size
    if any(a.size != n_points for a in flat_arrays):
        raise ValueError("all input arrays must have the same number of elements")

    inputs_flat = [v for arr in flat_arrays for v in arr.tolist()]
    result_flat = compiled_fn.call_batch_raw(inputs_flat, n_vars, n_points)
    result = np.array(result_flat, dtype=np.float64)
    if out_shape:
        result = result.reshape(out_shape)
    return result


def numpy_eval_par(compiled_fn, *arrays):
    """Parallel vectorised evaluation of a :class:`CompiledFn` over arrays.

    Identical to :func:`numpy_eval` but distributes the N evaluation points
    across all available CPU cores using Rayon.  The Python GIL is released
    during evaluation so other threads are not blocked.

    Requires the ``parallel`` build feature
    (``maturin develop --features parallel``).  Falls back silently to the
    sequential :func:`numpy_eval` when the feature is not compiled in.

    Parameters
    ----------
    compiled_fn : CompiledFn
        A compiled function returned by :func:`compile_expr` or
        :class:`CompileCache`.
    *arrays : array-like
        One array per input variable.  All arrays must have the same number
        of elements.  Accepts NumPy arrays, PyTorch CPU tensors, JAX arrays,
        or anything with ``__dlpack__`` or ``__array__``.

    Returns
    -------
    numpy.ndarray
        Output values with the same shape as the first input array.

    Notes
    -----
    For small N (< ~1 000 points) thread-scheduling overhead may exceed the
    computation time; :func:`numpy_eval` is faster in that regime.

    Example
    -------
    >>> import numpy as np
    >>> import alkahest
    >>> p = alkahest.ExprPool()
    >>> x = p.symbol("x")
    >>> f = alkahest.compile_expr(x ** 2, [x])
    >>> xs = np.linspace(0, 1, 10_000_000)
    >>> ys = alkahest.numpy_eval_par(f, xs)   # multi-core evaluation
    """
    # Use the parallel path if call_batch_raw_par is available (parallel feature).
    if not hasattr(compiled_fn, "call_batch_raw_par"):
        return numpy_eval(compiled_fn, *arrays)

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy_eval_par requires NumPy.  Install it with: pip install numpy"
        ) from exc

    n_vars = compiled_fn.n_inputs
    if len(arrays) != n_vars:
        raise ValueError(f"expected {n_vars} input array(s), got {len(arrays)}")

    first_raw = np.asarray(arrays[0])
    out_shape = first_raw.shape if first_raw.ndim > 0 else ()

    flat_arrays = [_to_numpy(a).ravel() for a in arrays]
    n_points = flat_arrays[0].size
    if any(a.size != n_points for a in flat_arrays):
        raise ValueError("all input arrays must have the same number of elements")

    inputs_flat = [v for arr in flat_arrays for v in arr.tolist()]
    result_flat = compiled_fn.call_batch_raw_par(inputs_flat, n_vars, n_points)
    result = np.array(result_flat, dtype=np.float64)
    if out_shape:
        result = result.reshape(out_shape)
    return result


__all__ = [
    # Phase 17
    "DAE",
    "HAS_EGRAPH",
    # Phase 16
    "ODE",
    # Phase 18
    "AcausalSystem",
    # Exceptions (V1-3 — stable diagnostic codes)
    "AlkahestError",
    "AlkahestError",
    "And",
    # Phase 22
    "ArbBall",
    "CadError",
    "CertifiedSolution",
    "CompileCache",
    # Phase 21
    "CompiledFn",
    "CompiledGradTracedFn",
    "CompiledTracedFn",
    "ConversionError",
    "DaeError",
    "DaeIndexReduction",
    "DerivedResult",
    "DiffError",
    "DiophantineError",
    "DiophantineSolution",
    "DomainError",
    "EgraphConfig",
    "EigenError",
    # Phase 20
    "Event",
    "Exists",
    "Expr",
    # Core
    "ExprPool",
    "FactorError",
    "Forall",
    "GbPoly",
    "GradTracedFn",
    "GroebnerBasis",
    "HybridODE",
    "IntegrationError",
    # V1-16: IoError
    "IoError",
    "JitError",
    "LatticeError",
    "LimitError",
    "LinearRecurrenceError",
    # Phase 15
    "Matrix",
    "MatrixError",
    "MultiPoly",
    "MultiPolyFactorization",
    "Not",
    "NumberTheoryError",
    "OdeError",
    "Or",
    "ParseError",
    "PoolError",
    "Port",
    # PA-5
    "PrimitiveRegistry",
    "Product",
    "ProductError",
    "PslqError",
    "RationalFunction",
    "RealRootError",
    "RegularChain",
    "ResultantError",
    # Rules
    "RewriteRule",
    "RootInterval",
    "RosenfeldGroebnerResult",
    "RsolveError",
    # Phase 19
    "SensitivitySystem",
    "Series",
    "SeriesError",
    "SolverError",
    "SparseGcdError",
    "SparseInterpError",
    "SumError",
    # PA-7
    "TracedFn",
    # PA-10
    "TreeDef",
    # Polynomials
    "UniPoly",
    "UniPolyFactorModP",
    "UniPolyFactorization",
    "__version__",
    "abs",
    "acos",
    "active_domain",
    "active_pool",
    "adjoint_system",
    "asin",
    "atan",
    "cad_lift",
    "cad_project",
    "ceil",
    # Phase 26
    "collect_like_terms",
    "compile_expr",
    # RW-7
    "context",
    "cos",
    "cosh",
    "dae_index_reduce",
    # V3-3 — FOFormula / V2-9 CAD
    "decide",
    # Calculus
    "diff",
    "diff_forward",
    "diophantine",
    "emit_c",
    "erf",
    "erfc",
    "eval_expr",
    "exp",
    "factor_univariate_mod_p",
    "flatten_exprs",
    "floor",
    # V2-3
    "gcd_sparse",
    "get_context_value",
    "grad",
    # V2-6
    "guess_relation",
    # Phase 24
    "horner",
    "integrate",
    "interval_eval",
    "jacobian",
    "jit",
    "jit_is_available",
    # V2-20
    "latex",
    "lattice",
    "limit",
    "log",
    "lower_to_first_order",
    "make_rule",
    "map_exprs",
    # Pattern matching & substitution
    "match_pattern",
    "number_theory",
    # Phase 25
    "numpy_eval",
    "numpy_eval_par",
    "pantelides",
    # V2-21
    "parse",
    # PA-9
    "piecewise",
    # Phase 27
    "poly_normal",
    "product_definite",
    "product_indefinite",
    # V2-4
    "real_roots",
    "refine_root",
    "resistor",
    # V2-2
    "resultant",
    # V2-13 — Differential algebra / Rosenfeld–Gröbner (requires groebner)
    "rosenfeld_groebner",
    "round",
    "rsolve",
    "satisfiable",
    "sensitivity_system",
    "series",
    "sign",
    # Simplification
    "simplify",
    "simplify_clifford_orthogonal",
    "simplify_egraph",
    "simplify_egraph_with",
    "simplify_enabled",
    "simplify_expanded",
    "simplify_log_exp",
    # Phase 23
    "simplify_par",
    "simplify_pauli",
    "simplify_trig",
    "simplify_with",
    # Math functions (core 5)
    "sin",
    "sinh",
    # V1-4 / V1-16: Polynomial system solver + Gröbner basis (requires groebner feature)
    "solve",
    "solve_linear_recurrence_homogeneous",
    "solve_numerical",
    "sparse_interp",
    "sparse_interp_univariate",
    "sqrt",
    "subresultant_prs",
    "subs",
    "sum_definite",
    "sum_indefinite",
    "symbol",
    "symbolic_grad",
    # V1-12: expanded primitives
    "tan",
    "tanh",
    # V5-1
    "to_lean",
    # V5-2
    "to_stablehlo",
    "trace",
    "trace_fn",
    # V2-11 — Regular chains / triangular decomposition
    "triangularize",
    "unflatten_exprs",
    "unicode_str",
    "verify_wz_pair",
    "version",
]
