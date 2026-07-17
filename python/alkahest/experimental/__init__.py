"""
alkahest.experimental — APIs that are not yet semver-stable.

Functions and classes in this module may change signature, be renamed, or be
removed between minor versions without a deprecation cycle.  Graduate to
``alkahest.*`` once the API has been exercised in production.

Current experimental surface (linear algebra on ``Matrix``):
- ``Matrix.rref``, ``nullspace``, ``rank``, ``column_space``, ``row_space``
- ``Matrix.lu``, ``qr``, ``cholesky``
- ``Matrix.jordan_form``, ``rational_canonical_form``, ``minimal_polynomial``,
  ``matrix_exp``, ``inverse``

Agents should probe ``alkahest.capabilities()`` at session start; symbolic LA
methods live on :class:`~alkahest.Matrix` and raise :class:`~alkahest.LinearAlgebraError`
(``E-LINALG-*``) when an operation is unsupported for the input field or shape.

Other experimental surface:
- :class:`Assumptions` — explicit positive/nonzero refinement for conservative
  simplification
- :func:`to_lean` — Lean 4 certificate export (V5-1)
- :func:`to_stablehlo` — StableHLO / XLA bridge (V5-2)
- :func:`to_jax` — JAX primitive integration (V5-7, requires JAX)
- :func:`evaluate` — unified exact, f64, and interval evaluation
- :func:`conjugate`, :func:`re`, :func:`im`, :func:`arg` — symbolic complex
  constructors (principal Arg only; no numeric/JIT evaluation yet)
- :class:`GroebnerBasis`, :class:`GbPoly` — parallel F4 Gröbner basis (V5-11,
  requires ``groebner`` feature)
- :func:`solve` — polynomial system solver (V1-4, requires ``groebner`` feature)
- :func:`compile_cuda` — NVPTX codegen + GPU batch evaluation via
  :class:`CudaCompiledFn` (requires ``cuda`` + ``jit`` features)

Calculus / ODE / transform surface (PyO3 bindings deferred at landing,
PRs #152–#161):
- :func:`heaviside`, :func:`dirac_delta` — distribution primitive constructors
  (placed here rather than at the top level so the frozen ``__all__`` is
  untouched).
- :func:`dsolve` — classical symbolic ODE solver (#153).
- :func:`laplace_transform` / :func:`inverse_laplace_transform` (#152).
- :func:`fourier_transform` / :func:`inverse_fourier_transform` (#158).
- :func:`z_transform` / :func:`inverse_z_transform` (#159).
- :func:`multilimit` — two-variable limits (#156).
- :func:`asymptotic_expand` — asymptotic expansion at infinity (#161).
- :func:`series_solve` — power-series / Frobenius ODE solutions (#160).
- :class:`Fps` — lazy formal power series over ℚ (#155).
- :func:`lambert_w`, :func:`digamma`, :func:`bessel_j0`, :func:`bessel_j1` —
  special-function foundation primitives (principal Lambert W₀, digamma ψ,
  Bessel J₀/J₁).

Numeric ODE integrators (Phase 16b):
- :func:`ode_integrate_rk4` — fixed-step 4th-order Runge–Kutta integrator.
- :func:`ode_integrate_rk45` — adaptive Dormand–Prince RK4(5) integrator.
- :class:`OdeTrajectory` — sampled trajectory returned by the integrators.
"""

from __future__ import annotations

import contextlib

# Re-export everything from the stable module for convenience
from alkahest import to_stablehlo

# Calculus / ODE / transform surface (always built into the extension).
from alkahest.alkahest import (
    Assumptions,
    EvaluationResult,
    Fps,
    OdeTrajectory,
    arg,
    asymptotic_expand,
    bessel_j0,
    bessel_j1,
    conjugate,
    digamma,
    dirac_delta,
    dsolve,
    evaluate,
    fourier_transform,
    heaviside,
    im,
    inverse_fourier_transform,
    inverse_laplace_transform,
    inverse_z_transform,
    lambert_w,
    laplace_transform,
    multilimit,
    ode_integrate_rk4,
    ode_integrate_rk45,
    re,
    series_solve,
    z_transform,
)

with contextlib.suppress(ImportError):
    from alkahest.alkahest import to_lean

with contextlib.suppress(ImportError):
    from alkahest._jax import to_jax

with contextlib.suppress(ImportError):
    from alkahest.alkahest import GbPoly, GroebnerBasis, solve

with contextlib.suppress(ImportError):
    from alkahest.alkahest import CudaCompiledFn, compile_cuda

__all__ = [
    "Assumptions",
    "CudaCompiledFn",
    "EvaluationResult",
    "Fps",
    "GbPoly",
    "GroebnerBasis",
    "OdeTrajectory",
    "arg",
    "asymptotic_expand",
    "bessel_j0",
    "bessel_j1",
    "compile_cuda",
    "conjugate",
    "digamma",
    "dirac_delta",
    "dsolve",
    "evaluate",
    "fourier_transform",
    "heaviside",
    "im",
    "inverse_fourier_transform",
    "inverse_laplace_transform",
    "inverse_z_transform",
    "lambert_w",
    "laplace_transform",
    "multilimit",
    "ode_integrate_rk4",
    "ode_integrate_rk45",
    "re",
    "series_solve",
    "solve",
    "to_jax",
    "to_lean",
    "to_stablehlo",
    "z_transform",
]
