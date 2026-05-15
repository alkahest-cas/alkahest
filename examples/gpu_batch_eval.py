#!/usr/bin/env python3
r"""Embarrassingly parallel math: CPU vector batch vs NVIDIA GPU (NVPTX).

Build a symbolic expression once, then evaluate it on millions of independent
(x, y) samples. Each output depends only on its own inputs, so the problem
is ideal for SIMD/CPU batching and for CUDA (data-parallel map semantics).

**CPU path** (always): :func:`alkahest.compile_expr` + :func:`alkahest.numpy_eval`
runs a fused native batch loop over all points.

**GPU path** (optional): :func:`alkahest.compile_cuda` returns a
:class:`~alkahest.CudaCompiledFn` whose :meth:`~alkahest.CudaCompiledFn.call_batch`
launches the generated NVPTX kernel on device 0.

Requirements for CUDA
---------------------
Rebuild the extension with JIT + NVPTX (see ``README.md`` / ``docs/features.md``)::

    maturin develop --manifest-path alkahest-py/Cargo.toml --release \\
        --features "jit cuda"

You need NVIDIA drivers and a GPU compatible with the shipped PTX target
(production builds target ``sm_86`` / Ampere class). If ``libdevice`` is not
found automatically, set ``ALKAHEST_LIBDEVICE_PATH`` to ``libdevice.10.bc``.

Notes
-----
- :func:`alkahest.to_jax` / ``jax.jit`` is excellent for array programs and AD,
  but the Alkahest JAX primitive dispatches concrete evaluation through the
  **CPU** batch path — use ``compile_cuda`` when you specifically want device
  kernels from symbolic expressions.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

import alkahest as ak


def _build_xy_expr(pool: ak.ExprPool):
    x = pool.symbol("x")
    y = pool.symbol("y")
    scale = pool.rational(1, 100)
    expr = ak.sin(x) * ak.cos(y) + (x * x + y * y) * scale
    return expr, x, y


def main() -> None:
    n = 1_048_576
    print("Alkahest GPU / CPU batch evaluation demo")
    print(f"  points per run: {n:_}")

    pool = ak.ExprPool()
    expr, x, y = _build_xy_expr(pool)
    inputs = [x, y]

    rng = np.random.default_rng(0)
    xs = rng.standard_normal(n, dtype=np.float64)
    ys = rng.standard_normal(n, dtype=np.float64)

    # --- CPU: compiled batch (vector-friendly, runs in native code) ----------
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="JIT compilation .* not available.*",
            category=RuntimeWarning,
        )
        f_cpu = ak.compile_expr(expr, inputs)
    t0 = time.perf_counter()
    out_cpu = ak.numpy_eval(f_cpu, xs, ys)
    t_cpu = time.perf_counter() - t0
    print(f"\nCPU  (compile_expr + numpy_eval): {t_cpu * 1000:.2f} ms")

    # --- Optional: NVPTX / CUDA ------------------------------------------------
    compile_cuda = getattr(ak, "compile_cuda", None)
    if compile_cuda is None:
        print(
            "\nCUDA: skipped (this wheel was built without the `cuda` feature).\n"
            "Rebuild with: maturin develop --manifest-path alkahest-py/Cargo.toml "
            '--release --features "jit cuda"'
        )
        return

    try:
        cu = compile_cuda(expr, inputs)
        t0 = time.perf_counter()
        out_gpu = np.asarray(cu.call_batch([xs, ys]), dtype=np.float64)
        t_gpu = time.perf_counter() - t0
    except Exception as exc:
        print(f"\nCUDA path failed (compile or launch): {exc}")
        return

    err = float(np.max(np.abs(out_gpu - out_cpu)))

    print(f"GPU  (compile_cuda + call_batch): {t_gpu * 1000:.2f} ms")
    print(f"max |GPU - CPU|: {err:.3e}")
    if err < 1e-9:
        print("numerical match: ok")
    else:
        print("warning: mismatch exceeds 1e-9 (investigate dtype / kernel)")


if __name__ == "__main__":
    main()
