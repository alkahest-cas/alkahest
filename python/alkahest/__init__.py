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
from ._plot import (
    plot,
    plot3d,
    plot_dag,
    plot_implicit,
    plot_parametric,
    plot_roots,
    plot_series,
    plot_svg,
)
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
    # Explicit positive/nonzero refinement for conservative simplification
    Assumptions,
    CompileCache,
    # Phase 21: JIT compiled evaluation
    CompiledFn,
    Component,
    # Core expression types
    DerivedResult,
    Domain,
    # V1-15: EgraphConfig and simplify_egraph_with
    EgraphConfig,
    # Unified exact / f64 / complex / interval evaluation
    EvaluationResult,
    # Phase 20: Hybrid systems
    Event,
    Exists,
    Expr,
    ExprPool,
    Forall,
    HybridODE,
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
    _build_features,
    _derived_result_context_simplify,
    abs,  # symbolic abs — use alkahest.abs(expr); shadows Python builtin within this module
    acos,
    acosh,
    adjoint_system,
    # Partial-fraction decomposition over ℚ
    apart,
    # Symbolic complex constructors (principal Arg)
    arg,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bessel_j0,
    bessel_j1,
    cad_lift,
    cad_project,
    # Rational-function cancel/together
    cancel,
    # Phase 18: Acausal modelling (component constructors)
    capacitor,
    ceil,
    # Phase 26: collect_like_terms
    collect_like_terms,
    compile_expr,
    conjugate,
    # Math functions
    cos,
    cosh,
    decide,
    # Core operations
    diff,
    diff_forward,
    digamma,
    # Elliptic special functions (parameter convention m = k²)
    elliptic_e,
    elliptic_f,
    elliptic_k,
    elliptic_pi,
    emit_c,
    emit_c_expr,
    emit_c_vec,
    erf,
    erfc,
    eval_expr,
    evaluate,
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
    im,
    integrate,
    interval_eval,
    jacobian,
    jit_is_available,
    lambert_w,
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
    re,
    real_roots,
    refine_root,
    # Rational meromorphic residues at points in ℚ(i)
    residue,
    resistor,
    # V2-2: Resultants and subresultant PRS
    resultant,
    round,  # symbolic round — use alkahest.round(expr)
    # Parametric Routh–Hurwitz stability conditions
    routh_hurwitz,
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
    simplify_trig_normal_form,
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
    together,
    verify_wz_pair,
    version,
    voltage_source,
)
from .alkahest import (
    # Phase 15: Symbolic matrices
    Matrix as _NativeMatrix,
)
from .alkahest import (
    # Phase 14: reverse-mode partials on Expr (native name `grad`; exported as symbolic_grad)
    grad as _native_symbolic_grad,
)
from .alkahest import (
    integrate_definite as _native_integrate_definite,
)

# V5-7: JAX primitive integration (optional — requires JAX)
with contextlib.suppress(ImportError):
    from ._jax import to_jax  # noqa: F401

# V1-4 / V1-16: Polynomial system solver + Gröbner basis
# groebner is a default Cargo feature since 2.3.1 — present in all PyPI wheels.
# contextlib.suppress is kept as a safety net for custom builds with --no-default-features.
with contextlib.suppress(ImportError):
    from .alkahest import (
        CertifiedSolution,
        DaeIndexReduction,
        DiophantineSolution,
        GbPoly,
        GroebnerBasis,
        PrimaryComponent,
        RegularChain,
        RosenfeldGroebnerResult,
        dae_index_reduce,
        diophantine,
        primary_decomposition,
        radical,
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
    AssumptionError,
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
    LinearAlgebraError,
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
    "AssumptionError",
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
    "LinearAlgebraError",
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


# ---------------------------------------------------------------------------
# DerivedResult coercion — fix P1: subs/diff/etc. with DerivedResult first arg
# ---------------------------------------------------------------------------


def symbolic_grad(expr, vars):
    """Symbolic partial derivatives of *expr* with respect to each variable in *vars*.

    This is **not** the same as :func:`grad` from :mod:`alkahest._transform`, which
    differentiates a :class:`~alkahest.TracedFn` produced by :func:`~alkahest.trace`
    and returns a :class:`~alkahest.GradTracedFn` for numeric evaluation (often
    composed with :func:`~alkahest.jit`).

    Parameters
    ----------
    expr : Expr
        Expression to differentiate.
    vars : list[Expr]
        Variables; one partial derivative per entry.

    Returns
    -------
    list[Expr]
        ``[∂expr/∂vars[0], ∂expr/∂vars[1], …]`` (plain expressions, no step log).

    See Also
    --------
    diff : single-variable derivative with :class:`~alkahest.DerivedResult` steps.
    grad : JAX-style gradient of a traced Python function.
    jacobian : matrix of partials for a vector-valued function.

    Example
    -------
    >>> p = ExprPool(); x, y = p.symbol("x"), p.symbol("y")
    >>> symbolic_grad(x**2 + y**2, [x, y])  # [2*x, 2*y]
    """
    return _native_symbolic_grad(expr, vars)


def _coerce_expr(x):
    """Return ``x.value`` if *x* is a :class:`DerivedResult`, else *x* unchanged.

    Internal helper used by the wrappers below so agents can write::

        r = diff(x**2, x)
        subs(r, {x: pool.integer(1)})   # r is a DerivedResult; auto-coerced

    rather than having to remember ``.value`` everywhere.
    """
    if isinstance(x, DerivedResult):
        return x.value
    return x


def _coerce_bound(value, like):
    """Coerce a definite-integral bound to an :class:`Expr` in *like*'s pool.

    *value* may already be an :class:`Expr`/:class:`DerivedResult` (returned as
    is, after ``.value`` coercion), or a Python ``int``/``float``, which is lifted
    into the same :class:`ExprPool` as *like* via ``like * 0 + value`` (the
    arithmetic operators coerce the scalar and the ``like * 0`` term vanishes).
    """
    value = _coerce_expr(value)
    if isinstance(value, Expr):
        return value
    return like * 0 + value


def _maybe_context_simplify(result):
    """When ``context(simplify=True)`` is active, algebraically simplify *result*.

    Merges the operation's derivation log with a ``context_simplify`` step and
    any steps from :func:`simplify`.  No-op for non-:class:`DerivedResult`
    values or when the flag is off.
    """
    if not simplify_enabled() or not isinstance(result, DerivedResult):
        return result
    return _derived_result_context_simplify(result)


# ---------------------------------------------------------------------------
# Matrix — accept bare int/float entries in `Matrix(...)` / `Matrix.from_rows(...)`
# ---------------------------------------------------------------------------


def _coerce_matrix_rows(rows):
    """Return *rows* unchanged unless every entry is a plain ``int``/``float``.

    The native ``Matrix.from_rows`` can infer an :class:`ExprPool` and coerce
    bare numbers as long as *some* entry in `rows` is already an
    :class:`Expr`/:class:`DerivedResult`.  If `rows` is entirely numeric
    (e.g. ``[[0, 1], [-1, 0]]``), there is no pool to infer it from, so we
    fall back to :func:`active_pool` (set via ``with alkahest.context(pool=...)``)
    and lift each entry into that pool with ``pool.integer``/``pool.float``.
    """
    has_expr = any(isinstance(entry, (Expr, DerivedResult)) for row in rows for entry in row)
    if has_expr:
        return rows
    pool = active_pool()
    if pool is None:
        # No pool anywhere — let the native constructor raise its clear error.
        return rows

    def lift(entry):
        if isinstance(entry, bool):
            # bool is a subclass of int; treat as int (0/1).
            return pool.integer(int(entry))
        if isinstance(entry, int):
            return pool.integer(entry)
        if isinstance(entry, float):
            return pool.float(entry)
        return entry

    return [[lift(entry) for entry in row] for row in rows]


class Matrix(_NativeMatrix):
    """Symbolic matrix.  See :class:`alkahest.alkahest.Matrix` for full docs.

    This thin wrapper additionally accepts a fully-numeric ``rows`` (every
    entry a Python ``int``/``float``, e.g. ``Matrix([[0, 1], [-1, 0]])``) by
    coercing entries into :func:`active_pool` (the pool set via
    ``with alkahest.context(pool=...)``).  Mixed int/``Expr`` rows already
    work without this wrapper, since the native constructor infers the pool
    from any :class:`Expr`/:class:`DerivedResult` entry.
    """

    def __new__(cls, rows):
        return _NativeMatrix.__new__(cls, _coerce_matrix_rows(rows))

    @staticmethod
    def from_rows(rows):
        return _NativeMatrix.from_rows(_coerce_matrix_rows(rows))


# Wrap key calculus/simplification functions so they accept DerivedResult or
# Expr as their first argument.  The native Rust functions only accept Expr.

_native_diff = diff


def diff(expr, var):
    """Differentiate *expr* with respect to *var*.

    Parameters
    ----------
    expr : Expr or DerivedResult
        Expression to differentiate.  If a :class:`DerivedResult` is passed
        (e.g. the output of a previous :func:`diff` call) its ``.value`` is
        used automatically.
    var : Expr
        The differentiation variable.

    Returns
    -------
    DerivedResult
        Contains the derivative in ``.value`` and a step log in ``.steps``.

    Example
    -------
    >>> p = ExprPool(); x = p.symbol("x")
    >>> d = diff(x**3, x)   # DerivedResult; d.value == 3*x^2
    >>> diff(d, x).value    # second derivative — passes DerivedResult directly
    """
    return _maybe_context_simplify(_native_diff(_coerce_expr(expr), _coerce_expr(var)))


_native_integrate = integrate


def integrate(expr, var, a=None, b=None):
    """Indefinite or definite integral of *expr* with respect to *var*.

    With two arguments this returns the indefinite integral (antiderivative).
    When both bounds *a* and *b* are supplied it returns the definite integral
    ``∫_a^b expr d(var)`` via the fundamental theorem of calculus,
    ``F(b) − F(a)`` where ``F`` is the antiderivative.

    Parameters
    ----------
    expr : Expr or DerivedResult
        Integrand.  :class:`DerivedResult` is auto-coerced to ``.value``.
    var : Expr
        Integration variable.
    a, b : Expr, optional
        Lower and upper bounds.  Provide **both** for a definite integral.

    Returns
    -------
    DerivedResult

    Raises
    ------
    ValueError
        If exactly one of *a*, *b* is provided.

    Notes
    -----
    The definite form is the elementary FTC wrapper: it requires an elementary
    antiderivative that is finite at both bounds.  Improper integrals, poles
    between the bounds, and the residue-theorem route are not handled; in those
    cases the underlying integration error is propagated rather than guessed.

    For an indefinite integral, ``result.verification`` reports
    ``"exactly_verified"`` only when the kernel proves the symbolic residual
    ``diff(result, var) - expr`` is zero. A successful result with
    ``"unverified"`` may instead have passed the integrator's separate numeric
    soundness gate. Definite integral results have no antiderivative
    verification metadata.

    Example
    -------
    >>> p = ExprPool(); x = p.symbol("x")
    >>> integrate(x**2, x).value         # x^3/3
    >>> integrate(x**2, x, 0, 1).value   # 1/3
    """
    if (a is None) != (b is None):
        raise ValueError(
            "integrate: provide both bounds for a definite integral, or neither "
            "for the indefinite integral"
        )
    var = _coerce_expr(var)
    if a is None:
        return _maybe_context_simplify(_native_integrate(_coerce_expr(expr), var))
    return _maybe_context_simplify(
        _native_integrate_definite(
            _coerce_expr(expr),
            var,
            _coerce_bound(a, var),
            _coerce_bound(b, var),
        )
    )


_native_apart = apart


def apart(expr, var):
    """Partial-fraction decomposition of *expr* as a rational function of *var*.

    Decomposes ``p(x)/q(x)`` into ``poly_part + Σ A_ij(x) / f_i(x)**j`` where the
    ``f_i`` are the distinct **ℚ-irreducible** factors of the denominator and
    ``deg A_ij < deg f_i``.

    Parameters
    ----------
    expr : Expr or DerivedResult
        A rational function of *var*.  :class:`DerivedResult` is coerced to
        ``.value``.
    var : Expr
        The variable to decompose in.

    Returns
    -------
    Expr

    Raises
    ------
    ValueError
        If *expr* is not a rational function of *var*, or denominator
        factorization fails.

    Notes
    -----
    Decomposition is over ℚ: irreducible quadratics (and higher) are kept
    intact, not split into complex/algebraic linear factors.

    Example
    -------
    >>> p = ExprPool(); x = p.symbol("x")
    >>> apart(1 / (x**2 - 1), x)   # 1/(2(x-1)) - 1/(2(x+1))
    """
    return _native_apart(_coerce_expr(expr), _coerce_expr(var))


_native_subs = subs


def subs(expr, mapping):
    """Substitute variables in *expr* according to *mapping*.

    Parameters
    ----------
    expr : Expr or DerivedResult
        Source expression.  :class:`DerivedResult` is auto-coerced to ``.value``.
    mapping : dict[Expr, Expr | DerivedResult | int | float]
        Substitution map. Values (and keys that are symbols) may be
        :class:`Expr`, :class:`DerivedResult`, or Python ``int``/``float``
        (coerced into the expression pool).

    Returns
    -------
    Expr

    Example
    -------
    >>> p = ExprPool(); x = p.symbol("x")
    >>> subs(x**2, {x: 3})         # int values coerced — OK
    >>> r = diff(x**2, x)          # DerivedResult
    >>> subs(r, {x: 1})            # DerivedResult first arg — OK
    """
    return _native_subs(_coerce_expr(expr), mapping)


_native_simplify = simplify


def simplify(expr):
    """Apply the algebraic rewrite-rule simplifier.

    Parameters
    ----------
    expr : Expr or DerivedResult

    Returns
    -------
    DerivedResult

    Example
    -------
    >>> p = ExprPool(); x = p.symbol("x")
    >>> simplify(x + 0).value   # x
    """
    return _native_simplify(_coerce_expr(expr))


_native_simplify_egraph = simplify_egraph


def simplify_egraph(expr):
    """E-graph equality-saturation simplifier.

    Parameters
    ----------
    expr : Expr or DerivedResult

    Returns
    -------
    DerivedResult
    """
    return _native_simplify_egraph(_coerce_expr(expr))


_native_simplify_trig = simplify_trig


def simplify_trig(expr):
    """Trigonometric identity simplifier.

    Parameters
    ----------
    expr : Expr or DerivedResult

    Returns
    -------
    DerivedResult
    """
    return _native_simplify_trig(_coerce_expr(expr))


_native_simplify_trig_normal_form = simplify_trig_normal_form


def simplify_trig_normal_form(expr):
    """Trigonometric normal-form simplifier.

    Runs the full algebraic core *with bounded polynomial expansion* plus the
    sin/cos-polynomial trig identities (argument-sign normalization and the
    Pythagorean identity, including its multi-angle case), driven to a fixed
    point.  Unlike :func:`simplify_trig` (identities only, no expansion), this
    composes product expansion, constant folding, like-term collection, and
    Pythagorean reduction in a single call.

    The headline use case is verifying orthogonality of a rotation
    (direction-cosine) matrix: every entry of ``R.T @ R - I`` collapses to
    ``0``.  It reduces in the sin/cos monomial basis and does not introduce
    compound-angle (``sin(2u)``, ``sin(u+v)``, …) forms.  This bundle is heavier
    than :func:`simplify` and is opt-in.

    Parameters
    ----------
    expr : Expr or DerivedResult

    Returns
    -------
    DerivedResult
    """
    return _native_simplify_trig_normal_form(_coerce_expr(expr))


_native_simplify_log_exp = simplify_log_exp


def simplify_log_exp(expr):
    """Logarithm / exponential simplifier.

    Parameters
    ----------
    expr : Expr or DerivedResult

    Returns
    -------
    DerivedResult
    """
    return _native_simplify_log_exp(_coerce_expr(expr))


_native_simplify_expanded = simplify_expanded


def simplify_expanded(expr):
    """Expand-then-simplify.

    Parameters
    ----------
    expr : Expr or DerivedResult

    Returns
    -------
    DerivedResult
    """
    return _native_simplify_expanded(_coerce_expr(expr))


_native_collect_like_terms = collect_like_terms


def collect_like_terms(expr):
    """Collect like terms in a sum.

    Parameters
    ----------
    expr : Expr or DerivedResult

    Returns
    -------
    DerivedResult
    """
    return _native_collect_like_terms(_coerce_expr(expr))


_native_series = series


def series(expr, var, point, order):
    """Truncated Taylor/Laurent series expansion.

    Parameters
    ----------
    expr : Expr or DerivedResult
    var : Expr
        Expansion variable.
    point : Expr
        Expansion point.
    order : int
        Truncation order.

    Returns
    -------
    Series
    """
    return _native_series(_coerce_expr(expr), _coerce_expr(var), _coerce_expr(point), order)


_native_limit = limit


def limit(expr, var, point, dir=None):
    """Compute the limit of *expr* as *var* → *point*.

    Parameters
    ----------
    expr : Expr or DerivedResult
    var : Expr
    point : Expr
    dir : str or None
        ``"+"`` for right limit, ``"-"`` for left limit.

    Returns
    -------
    Expr
    """
    return _native_limit(_coerce_expr(expr), _coerce_expr(var), _coerce_expr(point), dir)


_native_sum_indefinite = sum_indefinite


def sum_indefinite(expr, k):
    """Indefinite symbolic sum of *expr* with respect to index *k*."""
    return _maybe_context_simplify(_native_sum_indefinite(_coerce_expr(expr), _coerce_expr(k)))


_native_sum_definite = sum_definite


def sum_definite(expr, k, lo, hi):
    """Definite symbolic sum of *expr* for *k* from *lo* to *hi* (inclusive)."""
    return _maybe_context_simplify(
        _native_sum_definite(
            _coerce_expr(expr),
            _coerce_expr(k),
            _coerce_expr(lo),
            _coerce_expr(hi),
        )
    )


_native_product_indefinite = product_indefinite


def product_indefinite(expr, k):
    """Indefinite symbolic product of *expr* with respect to index *k*."""
    return _maybe_context_simplify(_native_product_indefinite(_coerce_expr(expr), _coerce_expr(k)))


_native_product_definite = product_definite


def product_definite(expr, k, lo, hi):
    """Definite symbolic product of *expr* for *k* from *lo* to *hi* (inclusive)."""
    return _maybe_context_simplify(
        _native_product_definite(
            _coerce_expr(expr),
            _coerce_expr(k),
            _coerce_expr(lo),
            _coerce_expr(hi),
        )
    )


_native_rsolve = rsolve


def rsolve(equation, n, seq_name, initials=None):
    """Solve a linear recurrence; *equation* may be :class:`DerivedResult`."""
    return _native_rsolve(_coerce_expr(equation), _coerce_expr(n), seq_name, initials)


_native_poly_normal = poly_normal


def poly_normal(expr, vars):
    """Expand and collect *expr* as a polynomial in *vars*."""
    return _native_poly_normal(_coerce_expr(expr), [_coerce_expr(v) for v in vars])


_native_cancel = cancel
_native_together = together


def cancel(expr, vars=None):
    """Combine *expr* over a common denominator and cancel common polynomial factors.

    Non-polynomial sub-expressions (e.g. ``sin(x)`` or symbols not in *vars*) are
    treated as opaque generators. Accepts :class:`DerivedResult` as *expr*.

    If *vars* is omitted, free symbols of *expr* are inferred.
    """
    coerced = _coerce_expr(expr)
    if vars is None:
        return _native_cancel(coerced)
    return _native_cancel(coerced, [_coerce_expr(v) for v in vars])


def together(expr, vars=None):
    """Combine *expr* over a single common denominator (alias of :func:`cancel`).

    If *vars* is omitted, free symbols of *expr* are inferred.
    """
    coerced = _coerce_expr(expr)
    if vars is None:
        return _native_together(coerced)
    return _native_together(coerced, [_coerce_expr(v) for v in vars])


_native_eval_expr = eval_expr


def eval_expr(expr, bindings):
    """Numerically evaluate *expr*; accepts :class:`DerivedResult` as *expr*."""
    return _native_eval_expr(_coerce_expr(expr), bindings)


# ---------------------------------------------------------------------------
# Groebner stubs — fallback for custom builds using --no-default-features.
# Since 2.3.1, groebner is a default Cargo feature and all standard PyPI
# wheels include it; these stubs should never fire in a normal install.
# Agents that read __all__ see these names; they get a clear error message
# rather than an AttributeError if somehow the feature is missing.
# ---------------------------------------------------------------------------

if "solve" not in dir():

    def solve(*_args, **_kwargs):
        """Polynomial system solver.

        This symbol is not available because the native extension was built
        without the ``groebner`` Cargo feature (e.g. via
        ``--no-default-features``). Standard ``pip install alkahest`` wheels
        always include it. To fix a custom build::

            maturin develop --features groebner
        """
        raise ImportError(
            "alkahest.solve is unavailable — the native extension was built "
            "without the groebner feature (--no-default-features). "
            "Standard PyPI wheels include groebner by default. "
            "See https://alkahest-cas.github.io/alkahest/ for details."
        )

    def solve_numerical(*_args, **_kwargs):
        """Numerical solver (groebner feature missing from this build)."""
        raise ImportError(
            "alkahest.solve_numerical is unavailable — groebner feature missing. "
            "See alkahest.solve.__doc__ for details."
        )

    class GroebnerBasis:
        """Gröbner basis type (groebner feature missing from this build).

        Raises ImportError on instantiation.
        """

        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "alkahest.GroebnerBasis is unavailable — groebner feature missing. "
                "See alkahest.solve.__doc__ for details."
            )


# ---------------------------------------------------------------------------
# capabilities() — single probe function for agent session start
# ---------------------------------------------------------------------------


def capabilities() -> dict:
    """Return the versioned agent contract for this installed build.

    Agents should call this once at session start before selecting an
    operation. The ``features`` mapping is authoritative for the installed
    extension; it does not infer availability from project defaults or Python
    fallback functions.

    Returns
    -------
    dict
        ``contract_version`` identifies this schema. ``groebner``, ``jit``,
        ``egraph``, and ``parallel`` are compatibility feature booleans.
        ``features`` contains installed Cargo features and explicit
        ``llvm_jit`` / ``cranelift_jit`` backend flags; ``primitives`` is
        deterministic per-primitive implementation coverage, and
        ``verification`` describes available evidence artifacts and checkers.
        Symbolic linear algebra (``Matrix.rref``, ``rank``, ``nullspace``,
        ``jordan_form``, ``minimal_polynomial``, LU/QR/Cholesky, etc.) is available
        on :class:`Matrix`; unsupported inputs raise :class:`LinearAlgebraError`
        with stable ``E-LINALG-*`` codes.

    Example
    -------
    >>> import alkahest as ak
    >>> caps = ak.capabilities()
    >>> if caps["groebner"]:
    ...     result = ak.solve([x**2 - 1], [x])
    >>> if not caps["jit"]:
    ...     print("JIT unavailable; compile_expr will run in interpreter mode")
    """
    features = _build_features()
    primitive_rows = PrimitiveRegistry.default_registry().coverage_report()
    primitive_rows.sort(key=lambda row: row["name"])
    return {
        "contract_version": 1,
        # Compatibility keys: report what this extension was compiled with,
        # even where a Python-level fallback exists.
        "groebner": features["groebner"],
        "jit": features["llvm_jit"] or features["cranelift_jit"],
        "egraph": features["egraph"],
        "parallel": features["parallel"],
        "features": features,
        "primitives": primitive_rows,
        "verification": {
            "statuses": [
                "lean_checked",
                "certificate_available",
                "exactly_verified",
                "numerically_checked",
                "unverified",
            ],
            "artifacts": {"lean4_source": True},
            "checkers": {"lean4": "external"},
        },
    }


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
    "And",
    # Phase 22
    "ArbBall",
    "AssumptionError",
    "Assumptions",
    "CadError",
    "CertifiedSolution",
    "CompileCache",
    # Phase 21
    "CompiledFn",
    "CompiledGradTracedFn",
    "CompiledTracedFn",
    "Component",
    "ConversionError",
    "DaeError",
    "DaeIndexReduction",
    "DerivedResult",
    "DiffError",
    "DiophantineError",
    "DiophantineSolution",
    "Domain",
    "DomainError",
    "EgraphConfig",
    "EigenError",
    "EvaluationResult",
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
    "LinearAlgebraError",
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
    "PrimaryComponent",
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
    "acosh",
    "active_domain",
    "active_pool",
    "adjoint_system",
    "apart",
    "arg",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bessel_j0",
    "bessel_j1",
    "cad_lift",
    "cad_project",
    "cancel",
    "capabilities",
    "capacitor",
    "ceil",
    # Phase 26
    "collect_like_terms",
    "compile_expr",
    "conjugate",
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
    "digamma",
    "diophantine",
    # Elliptic special functions (parameter convention m = k²)
    "elliptic_e",
    "elliptic_f",
    "elliptic_k",
    "elliptic_pi",
    "emit_c",
    "emit_c_expr",
    "emit_c_vec",
    "erf",
    "erfc",
    "eval_expr",
    "evaluate",
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
    "im",
    "integrate",
    "interval_eval",
    "jacobian",
    "jit",
    "jit_is_available",
    "lambert_w",
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
    "plot",
    "plot3d",
    "plot_dag",
    "plot_implicit",
    "plot_parametric",
    "plot_roots",
    "plot_series",
    "plot_svg",
    # Phase 27
    "poly_normal",
    "primary_decomposition",
    "product_definite",
    "product_indefinite",
    "radical",
    "re",
    # V2-4
    "real_roots",
    "refine_root",
    "residue",
    "resistor",
    # V2-2
    "resultant",
    # V2-13 — Differential algebra / Rosenfeld–Gröbner (groebner default feature)
    "rosenfeld_groebner",
    "round",
    "routh_hurwitz",
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
    "simplify_trig_normal_form",
    "simplify_with",
    # Math functions (core 5)
    "sin",
    "sinh",
    # V1-4 / V1-16: Polynomial system solver + Gröbner basis (default since 2.3.1)
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
    "together",
    "trace",
    "trace_fn",
    # V2-11 — Regular chains / triangular decomposition
    "triangularize",
    "unflatten_exprs",
    "unicode_str",
    "verify_wz_pair",
    "version",
    "voltage_source",
]
