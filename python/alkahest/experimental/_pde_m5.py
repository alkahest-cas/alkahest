"""PDE milestone M5: Green's functions and fundamental solutions.

Everything here is experimental and re-exported from
:mod:`alkahest.experimental`.

Convention
----------
A Green's function solves ``L G = δ`` for the operator exactly as written
(Mathematica's ``GreenFunction`` convention): ``Δ G = δ`` gives
``G = log(r)/(2π)`` in the plane.  Evolution kernels are causal and solve
``L G = δ(x) δ(t)``; Helmholtz kernels are outgoing.  The Sturm–Liouville
Green's functions are for ``𝓛u = −[(p u')' + q u]`` (``𝓛φₙ = λₙ w φₙ``).

Fundamental solutions
---------------------
:func:`fundamental_solution` (``"laplace"``, ``"biharmonic"``,
``"helmholtz"``, ``"modified_helmholtz"``, ``"heat"``, ``"wave"``,
``"klein_gordon"``, ``"telegraph"``) in 1–3 space dimensions, or with a
symbolic ``dimension=n`` for Laplace and heat::

    >>> import alkahest as ak, alkahest.experimental as akx
    >>> p = ak.ExprPool(); x, y = p.symbol("x"), p.symbol("y")
    >>> fs = akx.fundamental_solution("laplace", [x, y])
    >>> fs.kernel          # log(sqrt(x^2 + y^2)) / (2 pi)
    >>> fs.verified, fs.identity

Boundary-value problems
-----------------------
:func:`image_green_function` (half- and quarter-spaces, strips as an image
``Sum``, the disk and ball by Kelvin inversion with Poisson kernels),
:func:`green_function` (a :class:`SturmLiouville` problem: the eigenfunction
series — static, heat or wave — or the closed form by variation of
parameters) and :func:`green_solve` (``u = ∫ G f + boundary terms`` for a
:class:`PDE`).
"""

from __future__ import annotations

from alkahest.alkahest import (
    _pde_fundamental_solution,
    _pde_green_solve,
    _pde_image_green,
    _pde_sl_green,
)

__all__ = [
    "FundamentalSolution",
    "GreenFunction",
    "GreenSolution",
    "fundamental_solution",
    "green_function",
    "green_solve",
    "image_green_function",
]


class _Record:
    """Attribute access over the dict returned by the extension."""

    def __init__(self, d: dict):
        self._d = d

    @property
    def verified(self) -> bool:
        """True only when every check is an exact proof."""
        return bool(self._d["verified"])

    def failed_checks(self) -> list:
        """The checks that are not ``proved_zero``."""
        return [c for c in self._d["checks"] if c["status"] != "proved_zero"]


class FundamentalSolution(_Record):
    """An entry of the fundamental-solution table.

    ``operator``, ``dimension``, ``space`` (the variables ``x − ξ``),
    ``time``, ``radius`` (``r``), ``kernel`` (the distribution, with its
    ``heaviside`` / ``diracdelta`` factors), ``radial`` (``G(r[, t])`` on the
    support), ``smooth`` (the Cartesian kernel on the support), ``support``,
    ``identity`` (``"Δ G = δ(x)"``, …), ``identity_proof``, ``derivation``,
    ``checks`` (dicts ``what`` / ``status`` / ``method`` / ``detail``),
    ``status``, ``side_conditions`` and ``references``.
    """

    def __init__(self, d: dict):
        super().__init__(d)
        self.operator = d.get("operator")
        self.dimension = d.get("dimension")
        self.space = d.get("space")
        self.time = d.get("time")
        self.radius = d.get("radius")
        self.kernel = d.get("kernel")
        self.radial = d.get("radial")
        self.smooth = d.get("smooth")
        self.support = d.get("support")
        self.identity = d.get("identity")
        self.identity_proof = d.get("identity_proof")
        self.derivation = d.get("derivation")
        self.checks = d.get("checks")
        self.status = d.get("status")
        self.side_conditions = d.get("side_conditions")
        self.references = d.get("references")

    def __repr__(self) -> str:
        return f"FundamentalSolution({self.operator}, {self.identity}: G = {self.kernel})"


class GreenFunction(_Record):
    """A Green's function of a boundary-value problem.

    ``operator``, ``method`` (``"images"``, ``"image_series"``,
    ``"conformal_map"``, ``"kelvin"``, ``"eigenfunction_series"``,
    ``"modified_eigenfunction_series"``, ``"variation_of_parameters"``),
    ``space``, ``source`` (``ξ``), ``time``, ``value`` (``G``: closed form or
    an infinite ``Sum``), ``series``, ``branches`` (``(x < ξ, x > ξ)``),
    ``poisson_kernel`` and ``boundary_vars``, ``checks``, ``status``,
    ``side_conditions``, ``notes`` and ``references``.
    """

    def __init__(self, d: dict):
        super().__init__(d)
        self.operator = d.get("operator")
        self.method = d.get("method")
        self.space = d.get("space")
        self.source = d.get("source")
        self.time = d.get("time")
        self.value = d.get("value")
        self.series = d.get("series")
        self.branches = d.get("branches")
        self.poisson_kernel = d.get("poisson_kernel")
        self.boundary_vars = d.get("boundary_vars")
        self.checks = d.get("checks")
        self.status = d.get("status")
        self.side_conditions = d.get("side_conditions")
        self.notes = d.get("notes")
        self.references = d.get("references")

    def __repr__(self) -> str:
        return f"GreenFunction({self.operator}, {self.method}: G = {self.value})"


class GreenSolution(_Record):
    """The result of :func:`green_solve`.

    ``unknown``, ``kind`` (``"closed_form"`` / ``"integral_representation"``),
    ``method``, ``value`` (with ``Integral(…)`` nodes for a representation),
    ``closed_part``, ``terms`` (dicts ``role`` / ``kernel`` / ``density`` /
    ``vars`` / ``bounds`` / ``closed``), ``green`` (the Green's function used)
    and ``source``, ``checks``, ``status``, ``side_conditions``, ``notes`` and
    ``steps`` — the shape of the ``PdeSolution`` of ``pdsolve``.
    """

    def __init__(self, d: dict):
        super().__init__(d)
        self.unknown = d.get("unknown")
        self.kind = d.get("kind")
        self.method = d.get("method")
        self.value = d.get("value")
        self.closed_part = d.get("closed_part")
        self.terms = d.get("terms")
        self.green = d.get("green")
        self.source = d.get("source")
        self.checks = d.get("checks")
        self.status = d.get("status")
        self.side_conditions = d.get("side_conditions")
        self.notes = d.get("notes")
        self.steps = d.get("steps")

    def __repr__(self) -> str:
        return f"GreenSolution({self.kind}, {self.method}: u = {self.value})"


def fundamental_solution(
    operator: str, space=None, time=None, *, dimension=None, radius=None, **params
) -> FundamentalSolution:
    """The fundamental solution of ``operator``.

    ``space`` is the list of space variables (they stand for ``x − ξ``; 1, 2
    or 3 of them), ``time`` the time variable of an evolution operator.
    Parameters by keyword: ``k`` (Helmholtz, modified Helmholtz),
    ``diffusivity`` (heat), ``speed`` (wave, Klein–Gordon, telegraph),
    ``mass`` (Klein–Gordon), ``damping`` (telegraph, ``u_tt + 2a u_t −
    c² u_xx``).  ``dimension=n, radius=r`` (symbols) give the radial kernel in
    ``n`` dimensions for Laplace and heat.
    """
    if dimension is None and not space:
        raise ValueError("give the space variables, or dimension= and radius=")
    return FundamentalSolution(
        _pde_fundamental_solution(
            operator, list(space or []), time, params or None, dimension, radius
        )
    )


def image_green_function(
    operator: str, space, source, region, walls, time=None, **params
) -> GreenFunction:
    """The Green's function on a region by the method of images.

    ``region``: ``("half_space", axis)``, ``("quarter_space", a, b)``,
    ``("strip", axis, width)``, ``("disk", R)`` or ``("ball", R)`` (axes index
    ``space``); ``walls``: one ``"dirichlet"`` / ``"neumann"`` per wall (the
    wall at 0 first for a strip).  ``source`` is the source point ``ξ``.
    """
    if isinstance(walls, str):
        walls = [walls]
    return GreenFunction(
        _pde_image_green(
            operator, list(space), list(source), tuple(region), list(walls), time, params or None
        )
    )


def green_function(problem, xi, kind: str = "closed", *, diffusivity=None, speed=None, time=None):
    """The Green's function of a :class:`SturmLiouville` problem.

    ``kind``: ``"closed"`` (variation of parameters, separated regular
    conditions), ``"static"`` (the eigenfunction series ``Σ φₙ(x)φₙ(ξ)/(λₙ‖φₙ‖²)``,
    or the modified Green's function when ``λ = 0`` is an eigenvalue),
    ``"heat"`` (``diffusivity=``, ``time=``) or ``"wave"`` (``speed=``,
    ``time=``).
    """
    if getattr(problem, "var", None) is None:
        raise ValueError("green_function needs a SturmLiouville problem built with its variable")
    bc = problem.bc
    periodic = isinstance(bc, str) and bc.lower() == "periodic"
    left, right = ("dirichlet", "dirichlet") if periodic else bc
    from alkahest.experimental._pde_m3 import _end

    pool = problem.var._pool
    a, b = problem.interval
    param = diffusivity if kind == "heat" else speed
    return GreenFunction(
        _pde_sl_green(
            problem.var,
            problem.p,
            problem.q,
            problem.w,
            a,
            b,
            _end(left, pool),
            _end(right, pool),
            periodic,
            xi,
            kind,
            param,
            time,
        )
    )


def green_solve(pde) -> GreenSolution:
    """Solve a :class:`PDE` by its Green's function: ``u = ∫ G f +
    boundary terms``.

    Two-point problems ``a₂u'' + a₁u' + a₀u = F`` on ``Interval(x, a, b)`` with
    ``αu + βu' = γ`` at each end; the heat (``uₜ − κuₓₓ = Q``) and wave
    (``uₜₜ − c²uₓₓ = Q``) equations on ``Line(x) * HalfLine(t, 0)``; Poisson's
    equation on ``Line(x) * HalfLine(y, 0)`` with ``u(x, 0) = h``.
    """
    if pde.domain is None:
        raise ValueError("declare the domain: pde.on(Interval(x, 0, L))")
    return GreenSolution(
        _pde_green_solve(pde.equations, pde.unknowns, pde.domain.spec(), list(pde._conditions))
    )
