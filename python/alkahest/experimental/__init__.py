"""
alkahest.experimental — APIs that are not yet semver-stable.

Functions and classes in this module may change signature, be renamed, or be
removed between minor versions without a deprecation cycle.  Graduate to
``alkahest.*`` once the API has been exercised in production.

Matrix linear algebra (``Matrix.rref``, ``nullspace``, ``rank``, ``lu``, ``qr``,
``cholesky``, ``jordan_form``, ``matrix_exp``, …) lives on the stable
:class:`~alkahest.Matrix` type and raises :class:`~alkahest.LinearAlgebraError`
(``E-LINALG-*``) when unsupported. Agents should probe
``alkahest.capabilities()`` at session start.

Graduated to the top-level stable surface (still re-exported here for
backward compatibility):
- :class:`Assumptions`, :func:`evaluate` / :class:`EvaluationResult`
- :func:`conjugate`, :func:`re`, :func:`im`, :func:`arg`
- :func:`residue`
- :func:`lambert_w`, :func:`digamma`, :func:`bessel_j0`, :func:`bessel_j1`
- :func:`solve`, :class:`GroebnerBasis`, :class:`GbPoly` (require ``groebner``)
- :func:`to_lean`, :func:`to_stablehlo`

Remaining experimental surface:
- :func:`to_jax` — JAX primitive integration (requires JAX)
- :func:`compile_cuda` / :class:`CudaCompiledFn` — NVPTX codegen (requires
  ``cuda`` + ``jit``)

Calculus / ODE / transform surface:
- :func:`heaviside`, :func:`dirac_delta` — distribution primitive constructors
- :func:`dsolve` — classical symbolic ODE solver (#153)
- :func:`laplace_transform` / :func:`inverse_laplace_transform` (#152)
- :func:`fourier_transform` / :func:`inverse_fourier_transform` (#158)
- :func:`z_transform` / :func:`inverse_z_transform` (#159)
- :func:`multilimit` — two-variable limits (#156)
- :func:`asymptotic_expand` — asymptotic expansion at infinity (#161)
- :func:`series_solve` — power-series / Frobenius ODE solutions (#160)
- :class:`Fps` — lazy formal power series over ℚ (#155)

Numeric ODE integrators (Phase 16b):
- :func:`ode_integrate_rk4` — fixed-step 4th-order Runge–Kutta integrator
- :func:`ode_integrate_rk45` — adaptive Dormand–Prince RK4(5) integrator
- :class:`OdeTrajectory` — sampled trajectory returned by the integrators
"""

from __future__ import annotations

import contextlib

# Graduated symbols — re-exported from the stable top-level for callers that
# still import ``alkahest.experimental``.
from alkahest import (
    Assumptions,
    EvaluationResult,
    arg,
    bessel_j0,
    bessel_j1,
    conjugate,
    digamma,
    evaluate,
    im,
    lambert_w,
    re,
    residue,
    to_stablehlo,
)

# Calculus / ODE / transform surface (still experimental).
from alkahest.alkahest import (
    Fps,
    OdeTrajectory,
    asymptotic_expand,
    dirac_delta,
    dsolve,
    fourier_transform,
    heaviside,
    inverse_fourier_transform,
    inverse_laplace_transform,
    inverse_z_transform,
    laplace_transform,
    multilimit,
    ode_integrate_rk4,
    ode_integrate_rk45,
    series_solve,
    z_transform,
)

with contextlib.suppress(ImportError):
    from alkahest import to_lean

with contextlib.suppress(ImportError):
    from alkahest._jax import to_jax

with contextlib.suppress(ImportError):
    from alkahest import GbPoly, GroebnerBasis, solve

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
    "residue",
    "series_solve",
    "solve",
    "to_jax",
    "to_lean",
    "to_stablehlo",
    "z_transform",
]
