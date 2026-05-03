from . import modular  # noqa: F401 — V2-1: expose alkahest.modular submodule
from ._context import (  # noqa: F401
    active_domain,
    active_pool,
    context,
    get_context_value,
    simplify_enabled,
    symbol,
)
from ._dlpack import _to_numpy  # noqa: F401
from ._parse import parse  # noqa: F401
from ._pretty import latex, unicode_str  # noqa: F401
from ._pytree import (  # noqa: F401
    TreeDef,
    flatten_exprs,
    map_exprs,
    unflatten_exprs,
)
from ._transform import (  # noqa: F401
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
    # Phase 22: Ball arithmetic
    ArbBall,
    # Phase 21: JIT compiled evaluation
    CompiledFn,
    # Core expression types
    DerivedResult,
    # V1-15: EgraphConfig and simplify_egraph_with
    EgraphConfig,
    # Phase 20: Hybrid systems
    Event,
    Expr,
    ExprPool,
    HybridODE,
    # Phase 15: Symbolic matrices
    Matrix,
    # Polynomial types
    MultiPoly,
    # V2-1: Modular / CRT framework
    MultiPolyFp,
    Port,
    # PA-5: Primitive registry
    PrimitiveRegistry,
    RationalFunction,
    # Rewrite rules
    RewriteRule,
    # Phase 19: Sensitivity analysis
    SensitivitySystem,
    UniPoly,
    abs,  # symbolic abs — use alkahest.abs(expr); shadows Python builtin within this module
    acos,
    adjoint_system,
    asin,
    atan,
    atan2,
    ceil,
    # Phase 26: collect_like_terms
    collect_like_terms,
    compile_expr,
    # Math functions
    cos,
    cosh,
    # Core operations
    diff,
    diff_forward,
    emit_c,
    erf,
    erfc,
    eval_expr,
    exp,
    floor,
    gamma,
    # Phase 24: Horner-form code emission
    horner,
    integrate,
    interval_eval,
    jacobian,
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
    resistor,
    # V2-2: Resultants and subresultant PRS
    resultant,
    round,  # symbolic round — use alkahest.round(expr)
    # V2-3: Sparse interpolation
    sparse_interp,
    sparse_interp_univariate,
    sensitivity_system,
    sign,
    simplify,
    simplify_egraph,
    simplify_egraph_with,
    simplify_expanded,
    simplify_log_exp,
    # Phase 23: Parallel simplification
    simplify_par,
    simplify_trig,
    simplify_with,
    sin,
    sinh,
    sqrt,
    subresultant_prs,
    subs,
    # V1-12: expanded primitive registry
    tan,
    tanh,
    # V5-2: StableHLO/XLA bridge
    to_stablehlo,
    version,
)
from .alkahest import (  # noqa: F401
    # Phase 14: Reverse-mode AD (symbolic gradient; for traced-fn gradient use alkahest.grad)
    grad as symbolic_grad,
)

# V5-7: JAX primitive integration (optional — requires JAX)
try:
    from ._jax import to_jax  # noqa: F401
except ImportError:
    pass

# V1-4 / V1-16: Polynomial system solver + Gröbner basis (optional — requires groebner feature)
try:
    from .alkahest import GbPoly, GroebnerBasis, solve  # noqa: F401
except ImportError:
    pass

# V1-16: IoError (always present in the native module)
try:
    from .alkahest import IoError  # noqa: F401
except ImportError:
    pass
# Import exception classes from the native module (V1-3).
# The native module registers proper PyO3 exception classes with .code/.remediation/.span.
# Fall back to the pure-Python stubs in exceptions.py only if the native module
# doesn't export them (e.g. when running with a stale .so).
try:
    from .alkahest import (  # noqa: F401
        AlkahestError,
        ConversionError,
        DaeError,
        DiffError,
        DomainError,
        IntegrationError,
        JitError,
        MatrixError,
        OdeError,
        PoolError,
        ResultantError,
        SparseInterpError,
    )
except ImportError:
    from .exceptions import (  # noqa: F401
        AlkahestError,
        ConversionError,
        DaeError,
        DiffError,
        DomainError,
        IntegrationError,
        JitError,
        MatrixError,
        OdeError,
        PoolError,
    )
from .exceptions import ParseError, SolverError  # noqa: F401  (pure-Python only for now)


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
        raise ImportError(
            "numpy_eval requires NumPy.  Install it with: pip install numpy"
        ) from exc

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


__all__ = [
    # Exceptions (V1-3 — stable diagnostic codes)
    "AlkahestError",
    "AlkahestError",
    "ConversionError",
    "DomainError",
    "DiffError",
    "PoolError",
    "IntegrationError",
    "MatrixError",
    "OdeError",
    "DaeError",
    "JitError",
    "SolverError",
    "version",
    # Core
    "ExprPool",
    "Expr",
    "DerivedResult",
    # Polynomials
    "UniPoly",
    "MultiPoly",
    "RationalFunction",
    # Rules
    "RewriteRule",
    # Simplification
    "simplify",
    "simplify_egraph",
    "simplify_egraph_with",
    "EgraphConfig",
    "HAS_EGRAPH",
    "simplify_expanded",
    "simplify_trig",
    "simplify_log_exp",
    "simplify_with",
    # Calculus
    "diff",
    "diff_forward",
    "integrate",
    "symbolic_grad",
    # Pattern matching & substitution
    "match_pattern",
    "make_rule",
    "subs",
    # Math functions (core 5)
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    # V1-12: expanded primitives
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "erf",
    "erfc",
    "abs",
    "sign",
    "floor",
    "ceil",
    "round",
    # Phase 15
    "Matrix",
    "jacobian",
    # Phase 16
    "ODE",
    "lower_to_first_order",
    # Phase 17
    "DAE",
    "pantelides",
    # Phase 18
    "AcausalSystem",
    "Port",
    "resistor",
    # Phase 19
    "SensitivitySystem",
    "sensitivity_system",
    "adjoint_system",
    # Phase 20
    "Event",
    "HybridODE",
    # Phase 21
    "CompiledFn",
    "compile_expr",
    "eval_expr",
    # Phase 22
    "ArbBall",
    "interval_eval",
    # Phase 23
    "simplify_par",
    # Phase 24
    "horner",
    "emit_c",
    # Phase 25
    "numpy_eval",
    # Phase 26
    "collect_like_terms",
    # Phase 27
    "poly_normal",
    # V2-2
    "resultant",
    "subresultant_prs",
    "ResultantError",
    # V2-3
    "sparse_interp",
    "sparse_interp_univariate",
    "SparseInterpError",
    # PA-5
    "PrimitiveRegistry",
    # PA-7
    "TracedFn",
    "CompiledTracedFn",
    "GradTracedFn",
    "trace",
    "grad",
    "jit",
    "trace_fn",
    # PA-10
    "TreeDef",
    "flatten_exprs",
    "unflatten_exprs",
    "map_exprs",
    # PA-9
    "piecewise",
    # V5-2
    "to_stablehlo",
    # V1-4 / V1-16: Polynomial system solver + Gröbner basis (requires groebner feature)
    "solve",
    "GroebnerBasis",
    "GbPoly",
    # V1-16: IoError
    "IoError",
    # RW-7
    "context",
    "symbol",
    "active_pool",
    "active_domain",
    "simplify_enabled",
    "get_context_value",
    # V2-20
    "latex",
    "unicode_str",
    # V2-21
    "parse",
    "ParseError",
]
