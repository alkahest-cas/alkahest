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
- :func:`asymptotics_from_recurrence` — Poincaré–Perron growth of a P-recursive
  sequence, with the derived growth rate/exponent kept apart from the fitted
  connection constant (M5)
- :func:`series_solve` — power-series / Frobenius ODE solutions (#160)
- :class:`Fps` — lazy formal power series over ℚ (#155)

``q``-analogue creative telescoping (M4b):
- :func:`q_zeilberger` / :class:`QZeilbergerCertificate` — ``q``-Zeilberger for
  ``q``-hypergeometric sums (Gaussian binomials, ``q``-Pochhammer symbols),
  with the certificate re-checked as an exact identity in ``Q(q)(q**n)(q**k)``
  and a two-valued verdict on whether it carries over to the sum
- :func:`qbinomial`, :func:`qpochhammer` — builders for the two function heads
  the engine recognises
- :meth:`~alkahest.experimental.QZeilbergerCertificate.specialize_at_root_of_unity`
  / :class:`QRootOfUnitySpecialization` (M4) — the step from a ``Q(q)``
  identity to ``q = ζ_d``, a primitive ``d``-th root of unity: the
  ``q``-supercongruence literature. Pole and vanishing hypotheses are decided
  exactly by polynomial divisibility by ``Φ_d(q)`` over ``Q``, never
  numerically, and the verdict is three-valued (``"specializes"``,
  ``"obstructed"``, ``"unknown"``) rather than a silent specialisation
- :func:`cyclotomic_polynomial` — ``Φ_d(q)``, the modulus the root-of-unity
  arithmetic works over, exposed so a caller can redo the divisibility check
  by hand

Double-sum (Apagodu–Zeilberger) creative telescoping (M4):
- :func:`telescope2d` / :class:`Telescoping2dCertificate` — creative
  telescoping for a proper hypergeometric term ``F(n,j,k)`` with **two** bound
  indices, generalizing :func:`alkahest.zeilberger` from one summation index
  to two: finds ``a_0(n), …, a_J(n)`` and two rational certificates ``c_1,
  c_2`` with ``Σ_i a_i(n)·F(n+i,j,k) = Δ_j(c_1·F) + Δ_k(c_2·F)``, re-checked
  as an exact identity in ``Q(n,j,k)`` before being returned
- :meth:`~alkahest.experimental.Telescoping2dCertificate.boundary_status` —
  the two-dimensional boundary/corner analysis: the boundary of a rectangle
  is **four one-dimensional strip sums**, not four corner evaluations (read
  the Rust module docs, ``alkahest_cas::holonomic::telescoping2d::boundary``,
  for the derivation). Only **constant** (not ``n``-dependent) rectangles are
  supported, and only a sufficient "each strip vanishes pointwise" criterion
  is checked — real, stated restrictions, not unfinished polish
- :func:`telescope_md` / :class:`TelescopingMdCertificate` (since 3.10) — the
  general form of :func:`telescope2d` for an arbitrary number ``m >= 1`` of
  bound indices, not just two: ``telescope_md(term, n, [x_1, ..., x_m])``
  finds ``m`` rational certificates ``c_1, ..., c_m`` with
  ``Σ_i a_i(n)·F(n+i,x) = Σ_t Δ_t(c_t·F)``, and
  :meth:`~alkahest.experimental.TelescopingMdCertificate.boundary_status`
  decides the box-sum boundary (``2m`` face sums, the ``m``-dimensional
  generalization of the four-strip-sum result, **not** ``2**m`` corner
  evaluations). Same proper-hypergeometric-only, fixed-denominator,
  constant-box-only scope as ``telescope2d`` — no genuinely broader summand
  class. Raising ``m`` or ``max_cert_degree`` grows the ansatz search space
  fast (a certificate numerator is a box of
  ``(max_cert_degree + 1)**(m + 1)`` unknowns, and there are ``m`` of them),
  so this module also enforces two resource ceilings on the underlying exact
  linear solve — a single search probe is refused outright above 400
  unknowns, and the total work spent on probes at or above 150 unknowns in
  one search call is capped to 300 — so a search with no certificate in
  reach at all comes back as a fast, honest ``SearchExhausted`` (naming the
  ceiling when that, not genuine non-existence, is why) rather than running
  unboundedly long. See the Rust module docs
  (``alkahest_cas::holonomic::telescoping2d``) for the complete, honestly-
  stated scope and the exact ceiling values
- This is a genuinely scoped-down engine: proper hypergeometric summands
  only, no general Wegschaider reduction, a bounded-degree ansatz search
  rather than a minimal Gosper normal form. See the Rust module docs for the
  complete, honest limitations list

Novelty filtering (:mod:`alkahest.experimental.novelty`):
- :class:`RecurrenceClaim` — a recurrence in a normal form two presentations
  of the same fact share, plus a stable ``claim_hash`` to dedupe on
- :class:`QRecurrenceClaim` — the same for a ``q``-recurrence, whose
  coefficients live in ``Q(q, q^n)`` rather than ``Q[n]``
- :func:`check_novelty` / :class:`NoveltyVerdict` — was this claim already
  written down? Three-valued, and a negative is never reported as "novel"
- :class:`OeisCache` (offline, the tested path) and :class:`OeisWeb` (opt-in
  network) as sources

Coefficient fields for elimination (M9):
- :class:`ParametricGroebnerBasis` / :class:`ParametricGbPoly` — a Gröbner
  basis in ``Q(params)[vars]`` rather than ``Q[vars, params]``, reachable as
  ``GroebnerBasis.compute(polys, vars, params=[...])``.  The basis is generic,
  so it reports the hypersurfaces its leading coefficients assumed non-zero
  (``conditions()``) and refuses to ``specialize()`` on them instead of
  returning something that is not a basis

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

# M4(b) — q-analogue creative telescoping.  The engine is in the kernel; the
# two term builders are sugar over `pool.func`.
from alkahest._qterm import qbinomial, qpochhammer

# M5 — recurrence -> asymptotics.  The dispatch over the three shapes a
# recurrence arrives in is Python; the mathematics is in the kernel.
from alkahest._recurrence_asymptotics import (
    RecurrenceAsymptotics,
    asymptotics_from_recurrence,
)

# Calculus / ODE / transform surface (still experimental).
from alkahest.alkahest import (
    # P1 item 10 — asymptotic expansion at scale
    AsymptoticReport,
    Fps,
    OdeTrajectory,
    QRootOfUnitySpecialization,
    QZeilbergerCertificate,
    Telescoping2dCertificate,
    TelescopingMdCertificate,
    asymptotic_expand,
    # P1 item 10 — asymptotic expansion at scale
    coefficient_asymptotics,
    cyclotomic_polynomial,
    dirac_delta,
    dsolve,
    euler_maclaurin,
    fourier_transform,
    heaviside,
    inverse_fourier_transform,
    inverse_laplace_transform,
    inverse_z_transform,
    laplace_transform,
    multilimit,
    ode_integrate_rk4,
    ode_integrate_rk45,
    q_zeilberger,
    series_solve,
    telescope2d,
    telescope_md,
    z_transform,
)

# M11 — novelty filtering.  Claim normalisation, a stable hash, and a lookup
# whose negative is never reported as "novel".  Pure Python by CONTRIBUTING's
# rule: HTTP, JSON, and parsing a third party's prose.
from alkahest.experimental import novelty
from alkahest.experimental.novelty import (
    NoveltyMatch,
    NoveltyVerdict,
    OeisCache,
    OeisEntry,
    OeisWeb,
    QRecurrenceClaim,
    RecurrenceClaim,
    check_novelty,
)

with contextlib.suppress(ImportError):
    from alkahest import to_lean

with contextlib.suppress(ImportError):
    from alkahest._jax import to_jax

with contextlib.suppress(ImportError):
    from alkahest import GbPoly, GroebnerBasis, solve

# M9 — Gröbner bases over the coefficient field Q(params).  Registered by the
# extension only on `groebner` builds, hence the suppressed import.
with contextlib.suppress(ImportError):
    from alkahest.alkahest import ParametricGbPoly, ParametricGroebnerBasis

with contextlib.suppress(ImportError):
    from alkahest.alkahest import CudaCompiledFn, compile_cuda

__all__ = [
    "Assumptions",
    # P1 item 10 — asymptotic expansion at scale
    "AsymptoticReport",
    "CudaCompiledFn",
    "EvaluationResult",
    "Fps",
    "GbPoly",
    "GroebnerBasis",
    # M11 — novelty filtering
    "NoveltyMatch",
    "NoveltyVerdict",
    "OdeTrajectory",
    # M11 — novelty filtering
    "OeisCache",
    "OeisEntry",
    "OeisWeb",
    # M9 — coefficient fields for elimination
    "ParametricGbPoly",
    "ParametricGroebnerBasis",
    # M11 — novelty filtering
    "QRecurrenceClaim",
    # M4 — root-of-unity specialisation
    "QRootOfUnitySpecialization",
    # M4(b) — q-analogue creative telescoping
    "QZeilbergerCertificate",
    # M5 — recurrence -> asymptotics
    "RecurrenceAsymptotics",
    # M11 — novelty filtering
    "RecurrenceClaim",
    # M4 — double-sum (Apagodu-Zeilberger) creative telescoping
    "Telescoping2dCertificate",
    "TelescopingMdCertificate",
    "arg",
    "asymptotic_expand",
    # M5 — recurrence -> asymptotics
    "asymptotics_from_recurrence",
    "bessel_j0",
    "bessel_j1",
    # M11 — novelty filtering
    "check_novelty",
    # P1 item 10 — asymptotic expansion at scale
    "coefficient_asymptotics",
    "compile_cuda",
    "conjugate",
    # M4 — root-of-unity specialisation
    "cyclotomic_polynomial",
    "digamma",
    "dirac_delta",
    "dsolve",
    "euler_maclaurin",
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
    # M11 — novelty filtering (the module itself, for `novelty.RecordedRecurrence`
    # and the status tables)
    "novelty",
    "ode_integrate_rk4",
    "ode_integrate_rk45",
    # M4(b) — q-analogue creative telescoping
    "q_zeilberger",
    "qbinomial",
    "qpochhammer",
    "re",
    "residue",
    "series_solve",
    "solve",
    # M4 — double-sum (Apagodu-Zeilberger) creative telescoping
    "telescope2d",
    "telescope_md",
    "to_jax",
    "to_lean",
    "to_stablehlo",
    "z_transform",
]
