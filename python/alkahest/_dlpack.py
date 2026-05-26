"""DLPack / array-protocol bridge for :class:`~alkahest.CompiledFn`.

PA-8 — DLPack + ``__array__`` protocols on ``CompiledFn``.

This module provides ``_to_numpy`` which converts any array-like object
(NumPy, PyTorch, JAX, CuPy …) to a contiguous ``float64`` NumPy array using
the most efficient path available:

1. DLPack (``__dlpack__`` / ``numpy.from_dlpack``) — zero-copy when the
   tensor is on CPU and already float64; otherwise a copy to host is made.
2. ``__array__`` protocol — any object that implements ``__array__`` is
   accepted (NumPy, JAX arrays, …).
3. Fallback — ``numpy.asarray``.

The patched :func:`numpy_eval` in :mod:`alkahest` delegates to ``_to_numpy``
before calling ``call_batch_raw``.

Usage
-----
The module is imported automatically by :mod:`alkahest.__init__`.  You do not
need to call ``_to_numpy`` directly; it is used internally by
:func:`alkahest.numpy_eval`.

Example
-------
>>> import alkahest, numpy as np
>>> p = alkahest.ExprPool()
>>> x = p.symbol("x")
>>> f = alkahest.compile_expr(x**2, [x])
>>>
>>> # NumPy
>>> alkahest.numpy_eval(f, np.linspace(0, 1, 100))
>>>
>>> # PyTorch (CPU tensor)
>>> import torch
>>> alkahest.numpy_eval(f, torch.linspace(0, 1, 100))  # zero-copy via DLPack
>>>
>>> # JAX
>>> import jax.numpy as jnp
>>> alkahest.numpy_eval(f, jnp.linspace(0, 1, 100))
"""

from __future__ import annotations

from typing import Any


def _to_numpy(x: Any):
    """Convert any array-like to a contiguous ``float64`` NumPy array.

    Conversion priority
    -------------------
    1. Already a ``numpy.ndarray`` with ``dtype=float64`` and C-contiguous →
       returned as-is (zero-copy).
    2. Has ``__dlpack__`` → ``numpy.from_dlpack`` then cast to float64.
    3. Has ``__array__`` → ``numpy.asarray`` then ensure float64 / contiguous.
    4. Fallback → ``numpy.asarray``.

    Parameters
    ----------
    x : array-like
        Input array or scalar.

    Returns
    -------
    numpy.ndarray
        Contiguous, C-order, ``dtype=float64`` array.
    """
    import numpy as np

    # Fast-path: already a compatible ndarray.
    if isinstance(x, np.ndarray):
        if x.dtype == np.float64 and x.flags["C_CONTIGUOUS"]:
            return x
        return np.ascontiguousarray(x, dtype=np.float64)

    # DLPack path (PyTorch, CuPy, JAX on CPU, …).
    if hasattr(x, "__dlpack__"):
        try:
            arr = np.from_dlpack(x)
            return np.ascontiguousarray(arr, dtype=np.float64)
        except Exception:
            # Gracefully fall through if DLPack conversion fails (e.g. CUDA).
            pass

    # __array__ protocol (JAX arrays, etc.).
    if hasattr(x, "__array__"):
        return np.ascontiguousarray(np.asarray(x), dtype=np.float64)

    # Generic fallback.
    return np.ascontiguousarray(x, dtype=np.float64)


def numpy_eval_dlpack(compiled_fn, *arrays):
    """Vectorised evaluation supporting DLPack and ``__array__`` inputs.

    Replaces the Phase-25 :func:`alkahest.numpy_eval` with a version that
    accepts PyTorch / JAX / CuPy tensors in addition to NumPy arrays.

    Parameters
    ----------
    compiled_fn : CompiledFn
        A compiled function returned by :func:`alkahest.compile_expr`.
    *arrays : array-like
        One array per input variable.  All arrays must have the same number
        of elements.  Accepts NumPy arrays, PyTorch CPU tensors, JAX arrays,
        or anything with a ``__dlpack__`` or ``__array__`` method.

    Returns
    -------
    numpy.ndarray
        Output values with the same shape as the first input array.
    """
    import numpy as np

    n_vars = compiled_fn.n_inputs
    if len(arrays) != n_vars:
        raise ValueError(f"expected {n_vars} input array(s), got {len(arrays)}")

    # Capture original shape for reshaping at the end.
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


__all__ = ["_to_numpy", "numpy_eval_dlpack"]
