"""JAX-style transformation layer over the Alkahest Rust kernel.

PA-7 — ``cas.trace`` / ``cas.grad`` / ``cas.jit`` Python façade.

All transformations are pure Python wrappers over the existing Rust functions.
No kernel changes are required.

Example
-------
>>> import alkahest
>>> p = alkahest.ExprPool()
>>>
>>> @alkahest.trace(p)
... def f(x, y):
...     return x**2 + alkahest.sin(y)
>>>
>>> grad_f = alkahest.grad(f)              # TracedFn computing ∂f/∂x, ∂f/∂y
>>> fast_f = alkahest.jit(f)              # CompiledTracedFn backed by LLVM JIT
>>>
>>> import numpy as np
>>> xs = np.linspace(0, 1, 1_000_000)
>>> ys = fast_f(xs, xs)                 # vectorised, zero-copy
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Callable

from .alkahest import (
    Expr,
    ExprPool,
    compile_expr,
)

# ---------------------------------------------------------------------------
# TracedFn — a symbolic function built by tracing a Python callable
# ---------------------------------------------------------------------------


class TracedFn:
    """A symbolic function produced by tracing a Python callable.

    The callable is invoked *once* with symbolic :class:`Expr` arguments to
    capture the computation graph.  Subsequent numeric evaluations run through
    the Rust interpreter or (if :func:`jit` is applied) the LLVM JIT.

    Attributes
    ----------
    expr : Expr
        The symbolic output expression.
    pool : ExprPool
        The expression pool that owns all nodes.
    symbols : list[Expr]
        The symbolic input variables, one per positional argument.
    """

    def __init__(
        self,
        expr: Expr,
        pool: ExprPool,
        symbols: list[Expr],
        *,
        names: list[str] | None = None,
    ):
        self.expr = expr
        self.pool = pool
        self.symbols = symbols
        self._names: list[str] = names or [f"x{i}" for i in range(len(symbols))]

    # ── numeric evaluation ────────────────────────────────────────────────────

    def __call__(self, *values):
        """Evaluate numerically.

        Values may be plain Python floats *or* NumPy / JAX / PyTorch arrays.
        When array inputs are detected, the call is automatically vectorised
        via the batch path (same as :func:`~alkahest.numpy_eval`).
        """
        if len(values) != len(self.symbols):
            raise ValueError(f"expected {len(self.symbols)} argument(s), got {len(values)}")
        # Array-path: delegate to numpy_eval
        try:
            import numpy as np

            arr_inputs = [np.asarray(v) for v in values]
            if any(a.ndim > 0 for a in arr_inputs):
                from . import numpy_eval

                compiled = compile_expr(self.expr, self.symbols)
                return numpy_eval(compiled, *values)
        except ImportError:
            pass
        # Scalar path: build env dict and call eval_interp
        from .alkahest import eval_expr  # noqa: PLC0415

        env = {sym: float(val) for sym, val in zip(self.symbols, values)}
        return eval_expr(self.expr, env)

    # ── representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        arg_str = ", ".join(self._names)
        return f"TracedFn({arg_str}) → {self.expr}"


# ---------------------------------------------------------------------------
# CompiledTracedFn — JIT-compiled variant of TracedFn
# ---------------------------------------------------------------------------


class CompiledTracedFn:
    """A :class:`TracedFn` whose evaluation is backed by LLVM JIT.

    Created via :func:`jit`.  Falls back to the Rust interpreter if the
    ``jit`` feature is not compiled in.
    """

    def __init__(self, traced: TracedFn):
        self._traced = traced
        self._compiled = compile_expr(traced.expr, traced.symbols)

    @property
    def expr(self) -> Expr:
        return self._traced.expr

    @property
    def pool(self) -> ExprPool:
        return self._traced.pool

    @property
    def symbols(self) -> list[Expr]:
        return self._traced.symbols

    def __call__(self, *values):
        """Evaluate; routes to the batch path for array inputs."""
        if len(values) != len(self.symbols):
            raise ValueError(f"expected {len(self.symbols)} argument(s), got {len(values)}")
        try:
            import numpy as np

            arr_inputs = [np.asarray(v) for v in values]
            if any(a.ndim > 0 for a in arr_inputs):
                from . import numpy_eval

                return numpy_eval(self._compiled, *values)
            # Scalar
            return self._compiled([float(v) for v in values])
        except ImportError:
            return self._compiled([float(v) for v in values])

    def __repr__(self) -> str:
        return f"CompiledTracedFn({self._traced!r})"


# ---------------------------------------------------------------------------
# GradTracedFn — gradient of a TracedFn
# ---------------------------------------------------------------------------


class GradTracedFn:
    """Gradient of a :class:`TracedFn`.

    The gradient is computed *symbolically* via the Rust ``grad`` or ``diff``
    function and then wrapped as a new :class:`TracedFn` for each output.
    """

    def __init__(
        self,
        traced: TracedFn,
        wrt: list[Expr] | None,
    ):
        from .alkahest import diff as _diff  # noqa: PLC0415

        self._traced = traced
        self._wrt = wrt or traced.symbols

        self._grad_exprs: list[Expr] = []
        for sym in self._wrt:
            dr = _diff(traced.expr, sym)
            self._grad_exprs.append(dr.value)

    def __call__(self, *values) -> list:
        """Return the gradient as a list, one value per ``wrt`` variable."""
        if len(values) != len(self._traced.symbols):
            raise ValueError(f"expected {len(self._traced.symbols)} argument(s), got {len(values)}")
        try:
            import numpy as np

            arr_inputs = [np.asarray(v) for v in values]
            if any(a.ndim > 0 for a in arr_inputs):
                from . import numpy_eval

                results = []
                for g_expr in self._grad_exprs:
                    compiled = compile_expr(g_expr, self._traced.symbols)
                    results.append(numpy_eval(compiled, *values))
                return results
        except ImportError:
            pass
        from .alkahest import eval_expr  # noqa: PLC0415

        env = {sym: float(val) for sym, val in zip(self._traced.symbols, values)}
        return [eval_expr(g_expr, env) for g_expr in self._grad_exprs]

    def __repr__(self) -> str:
        wrt_names = [str(w) for w in self._wrt]
        return f"GradTracedFn(wrt=[{', '.join(wrt_names)}]) of {self._traced!r}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trace(
    pool: ExprPool,
    *,
    names: Sequence[str] | None = None,
    domain: str = "real",
) -> Callable:
    """Decorator that symbolically traces a Python function.

    Parameters
    ----------
    pool : ExprPool
        The expression pool to use for symbolic variables.
    names : sequence of str, optional
        Names for the symbolic arguments.  Inferred from the function
        signature when omitted.
    domain : str
        Domain for the symbolic variables (``"real"``, ``"complex"`` …).

    Returns
    -------
    Callable
        A decorator; when applied to a function it returns a
        :class:`TracedFn`.

    Example
    -------
    >>> @alkahest.trace(p)
    ... def f(x, y):
    ...     return x**2 + alkahest.sin(y)
    """

    def decorator(fn: Callable) -> TracedFn:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        sym_names = list(names) if names else params

        if len(sym_names) != len(params):
            raise ValueError(
                f"trace: {len(sym_names)} names supplied but function has {len(params)} parameters"
            )

        syms = [pool.symbol(n, domain) for n in sym_names]
        expr = fn(*syms)

        return TracedFn(expr, pool, syms, names=sym_names)

    return decorator


def grad(
    fn: TracedFn | Callable,
    *,
    wrt: Sequence[Expr] | None = None,
) -> GradTracedFn:
    """Return a new callable that computes the gradient of ``fn``.

    Parameters
    ----------
    fn : TracedFn or callable decorated with :func:`trace`
        The function to differentiate.
    wrt : sequence of Expr, optional
        Variables to differentiate with respect to.  Defaults to all
        input variables of ``fn``.

    Returns
    -------
    GradTracedFn
        A callable returning ``[∂f/∂wrt[0], ∂f/∂wrt[1], …]``.
    """
    if not isinstance(fn, TracedFn):
        raise TypeError(
            f"grad() expects a TracedFn (decorated with @alkahest.trace), got {type(fn).__name__}"
        )
    return GradTracedFn(fn, list(wrt) if wrt is not None else None)


def jit(fn: TracedFn | Callable) -> CompiledTracedFn:
    """JIT-compile a :class:`TracedFn` using the LLVM backend.

    Parameters
    ----------
    fn : TracedFn or callable decorated with :func:`trace`
        The function to compile.

    Returns
    -------
    CompiledTracedFn
        A callable backed by LLVM JIT (or the Rust interpreter if ``--features
        jit`` was not enabled at build time).
    """
    if not isinstance(fn, TracedFn):
        raise TypeError(
            f"jit() expects a TracedFn (decorated with @alkahest.trace), got {type(fn).__name__}"
        )
    return CompiledTracedFn(fn)


# ---------------------------------------------------------------------------
# Convenience: functional-style (non-decorator) trace
# ---------------------------------------------------------------------------


def trace_fn(
    fn: Callable,
    pool: ExprPool,
    *,
    names: Sequence[str] | None = None,
    domain: str = "real",
) -> TracedFn:
    """Trace ``fn`` without using it as a decorator.

    Equivalent to ``trace(pool, names=names, domain=domain)(fn)``.
    """
    return trace(pool, names=names, domain=domain)(fn)


__all__ = [
    "TracedFn",
    "CompiledTracedFn",
    "GradTracedFn",
    "trace",
    "grad",
    "jit",
    "trace_fn",
]
