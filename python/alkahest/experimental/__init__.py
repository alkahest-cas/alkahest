"""
alkahest.experimental — APIs that are not yet semver-stable.

Functions and classes in this module may change signature, be renamed, or be
removed between minor versions without a deprecation cycle.  Graduate to
``alkahest.*`` once the API has been exercised in production.

Current experimental surface:
- :func:`to_lean` — Lean 4 certificate export (V5-1)
- :func:`to_stablehlo` — StableHLO / XLA bridge (V5-2)
- :func:`to_jax` — JAX primitive integration (V5-7, requires JAX)
- :class:`GroebnerBasis`, :class:`GbPoly` — parallel F4 Gröbner basis (V5-11,
  requires ``groebner`` feature)
- :func:`solve` — polynomial system solver (V1-4, requires ``groebner`` feature)
- :func:`compile_cuda` — NVPTX codegen + GPU batch evaluation via
  :class:`CudaCompiledFn` (requires ``cuda`` + ``jit`` features)
"""

from __future__ import annotations

# Re-export everything from the stable module for convenience
from alkahest import to_stablehlo  # noqa: F401

try:
    from alkahest.alkahest import to_lean  # noqa: F401
except ImportError:
    pass

try:
    from alkahest._jax import to_jax  # noqa: F401
except ImportError:
    pass

try:
    from alkahest.alkahest import GbPoly, GroebnerBasis, solve  # noqa: F401
except ImportError:
    pass

try:
    from alkahest.alkahest import CudaCompiledFn, compile_cuda  # noqa: F401
except ImportError:
    pass

__all__ = [
    "to_stablehlo",
    "to_lean",
    "to_jax",
    "GbPoly",
    "GroebnerBasis",
    "solve",
    "compile_cuda",
    "CudaCompiledFn",
]
