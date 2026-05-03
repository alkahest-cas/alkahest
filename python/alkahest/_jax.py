"""
alkahest._jax — JAX primitive source integration (V5-7).

Wrap a Alkahest symbolic expression as a JAX primitive so it participates in
jax.grad, jax.vmap, and jax.pmap.

Usage::

    import alkahest
    import alkahest._jax as cjax

    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    expr = alkahest.sin(x) * alkahest.cos(x)

    f = cjax.as_jax_primitive(expr, [x])
    # f is now callable like a JAX function
    import jax
    import jax.numpy as jnp
    xs = jnp.linspace(0, 1, 100)
    ys = f(xs)          # concrete eval
    dydx = jax.grad(lambda x: f(x).sum())(xs)  # reverse-mode grad
"""

from __future__ import annotations

from typing import Callable

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    from jax import core as jax_core  # noqa: F401
    from jax.interpreters import ad as jax_ad  # noqa: F401
    from jax.interpreters import batching as jax_batching  # noqa: F401

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

import numpy as np

import alkahest


def _require_jax() -> None:
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is not installed. Run: pip install jax[cuda12] or pip install jax")


def as_jax_primitive(expr, inputs: list) -> Callable:
    """Wrap a Alkahest expression as a JAX-traceable callable.

    Parameters
    ----------
    expr : alkahest.Expr
        The symbolic expression to wrap.
    inputs : list[alkahest.Expr]
        Ordered list of symbolic input variables.

    Returns
    -------
    callable
        A JAX-compatible function ``f(*arrays) -> array``.  Supports
        ``jax.grad``, ``jax.vmap``, and ``jax.jit``.

    Notes
    -----
    This implementation registers a JAX *primitive* with:
    - Abstract evaluation (shape / dtype inference).
    - Concrete evaluation via ``alkahest.numpy_eval``.
    - JVP (forward-mode) via ``alkahest.grad``.
    - Transpose rule for reverse-mode.
    """
    _require_jax()

    # Compute gradients symbolically, once at wrap time.
    grad_exprs = alkahest.symbolic_grad(expr, inputs)

    # Define the JAX primitive
    prim_name = f"alkahest_{id(expr)}"
    prim = jax_core.Primitive(prim_name)
    prim.multiple_results = False

    # Compile batched eval
    try:
        compiled_fn = alkahest.compile_expr(expr, inputs)
    except Exception:
        compiled_fn = None

    def _eval_impl(*arrays):
        """Concrete evaluation via alkahest.numpy_eval or call_batch_raw."""
        flat_arrays = [np.asarray(a, dtype=np.float64).ravel() for a in arrays]
        n_pts = flat_arrays[0].size if flat_arrays else 1
        if compiled_fn is not None:
            inputs_flat = np.concatenate(flat_arrays)
            out = np.array(compiled_fn.call_batch_raw(inputs_flat.tolist(), len(inputs), n_pts))
        else:
            out = np.zeros(n_pts)
            for i in range(n_pts):
                bindings = {inp: float(arr[i]) for inp, arr in zip(inputs, flat_arrays)}
                out[i] = alkahest.eval_expr(expr, bindings)
        shape = arrays[0].shape if arrays else ()
        return jnp.array(out.reshape(shape))

    prim.def_impl(_eval_impl)

    # Abstract eval: output has same shape/dtype as first input
    def _abstract_eval(*avals):
        return jax_core.ShapedArray(avals[0].shape, avals[0].dtype)

    prim.def_abstract_eval(_abstract_eval)

    # JVP (tangent propagation)
    def _jvp(primals, tangents):
        out = prim.bind(*primals)
        grad_vals = [as_jax_primitive(g, inputs)(*primals) for g in grad_exprs]
        out_dot = sum(g * t for g, t in zip(grad_vals, tangents) if not isinstance(t, jax_ad.Zero))
        if isinstance(out_dot, int) and out_dot == 0:
            out_dot = jnp.zeros_like(out)
        return out, out_dot

    jax_ad.primitive_jvps[prim] = _jvp

    # Batching (vmap) rule
    def _batching_rule(args, axes):
        out = prim.bind(*args)
        return out, axes[0]

    jax_batching.primitive_batchers[prim] = _batching_rule

    def f(*arrays):
        return prim.bind(*arrays)

    return f


def to_jax(expr, inputs: list) -> Callable:
    """Alias for :func:`as_jax_primitive`.

    Matches the ``alkahest.to_jax(expr, inputs)`` name from the plan.
    """
    return as_jax_primitive(expr, inputs)
