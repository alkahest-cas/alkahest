"""PDE milestone M4: separation of variables and eigenfunction-series
solutions on bounded domains.

Everything here is experimental and re-exported from
:mod:`alkahest.experimental`.

Separability
------------
:func:`separate` substitutes ``u = X(x)·T(t)…`` (or ``X(x) + T(t)`` with
``kind="additive"``) and tests whether ``L[XT]/(XT)`` splits, trying the
classical multipliers (``r²`` for the polar Laplacian, ``r² sin²θ`` for the
spherical one)::

    >>> import alkahest as ak, alkahest.experimental as akx
    >>> p = ak.ExprPool(); x, t = p.symbol("x"), p.symbol("t")
    >>> u = akx.Function("u")(x, t)
    >>> s = akx.separate(akx.D(u, t) - akx.D(u, x, 2), u)
    >>> [e["equation"] for e in s.equations]   # X'' + mu1 X, T' + mu1 T

Series solutions
----------------
:func:`separation_solve` solves a linear :class:`PDE` posed on intervals
(``Interval``, ``Box``), curvilinear regions (:class:`PolarDisk`,
:class:`Annulus`, :class:`Sector`, :class:`Cylinder`,
:class:`SphericalBall`, or :class:`Curvilinear`) or a Cartesian ``Disk`` /
``Ball``, times at most one ``HalfLine`` in time::

    >>> L, k = p.symbol("L", "positive"), p.symbol("k", "positive")
    >>> pde = (akx.PDE(akx.D(u, t) - k*akx.D(u, x, 2), u)
    ...        .on(akx.Interval(x, 0, L) * akx.HalfLine(t, 0))
    ...        .bc(u.subs({x: 0}), 0).bc(u.subs({x: L}), 0)
    ...        .ic(u.subs({t: 0}), x*(L - x)))
    >>> sol = akx.separation_solve(pde)
    >>> sol.value          # Sum(... sin(n*pi*x/L) exp(-k (n pi/L)^2 t), n, 1, oo)
    >>> sol.verified, sol.convergence
    >>> sol.truncate(5)    # explicit partial sum
    >>> f = sol.compile(20, [x, t], {L: 1, k: 1})   # fast numeric evaluation

Every mode is checked to satisfy the PDE identically in its indices, every
coefficient by an exactly checked antiderivative, and the projection of the
data is compared numerically on the first members (which can only refute).
The truncated series is never claimed equal to the data pointwise;
``convergence`` (``"uniform"`` / ``"l2"``) and ``notes`` say how the data are
attained (Gibbs phenomenon at an incompatible boundary).
"""

from __future__ import annotations

import alkahest as _ak
from alkahest.alkahest import (
    _pde_curvilinear_laplacian,
    _pde_numeric_truncation,
    _pde_separate,
    _pde_separation_solve,
)
from alkahest.experimental._pde import _Domain

__all__ = [
    "Annulus",
    "Curvilinear",
    "Cylinder",
    "ModeGroup",
    "PolarDisk",
    "Sector",
    "Separation",
    "SeriesSolution",
    "SphericalBall",
    "curvilinear_laplacian",
    "is_separable",
    "separate",
    "separation_solve",
]


def _pool_of(*objs):
    for o in objs:
        if isinstance(o, _ak.Expr):
            return o._pool
        if isinstance(o, (list, tuple)):
            pl = _pool_of(*o)
            if pl is not None:
                return pl
    return None


def _as_expr(v, pool):
    if isinstance(v, _ak.Expr):
        return v
    if isinstance(v, bool):
        raise TypeError("a bool is not an expression")
    if isinstance(v, int):
        return pool.integer(v)
    if isinstance(v, float):
        return pool.float(v)
    if isinstance(v, str):
        return _ak.parse(v, pool)
    raise TypeError(f"cannot use {v!r} as an expression")


# ---------------------------------------------------------------------------
# Curvilinear domains
# ---------------------------------------------------------------------------


class Curvilinear(_Domain):
    """A coordinate box in ``"polar"`` ``(r, θ)``, ``"cylindrical"``
    ``(r, θ, z)`` or ``"spherical"`` ``(r, θ, φ)`` coordinates (``θ`` the polar
    angle in spherical coordinates).  ``coords[i]`` may be ``None`` for a
    coordinate the unknown does not depend on (radial and axisymmetric
    problems); ``bounds[i] = (lo, hi)``.  An angular range of length ``2π`` is
    periodic; ``r = 0`` and the poles ``θ = 0, π`` are singular ends where
    boundedness is implied.
    """

    __slots__ = ("bounds", "coords", "system")

    def __init__(self, system: str, coords, bounds):
        self.system = system
        self.coords = list(coords)
        pool = _pool_of([c for c in self.coords if c is not None])
        self.bounds = [(_as_expr(lo, pool), _as_expr(hi, pool)) for lo, hi in bounds]

    def spec(self):
        return ("curvilinear", self.system, self.coords, self.bounds)

    def vars(self):
        return [c for c in self.coords if c is not None]


def _pi(pool):
    return _ak.parse("pi", pool)


def PolarDisk(r, theta, radius) -> Curvilinear:
    """The disk ``0 ≤ r < radius``, ``θ ∈ (−π, π)`` (``theta=None`` for a
    radial problem)."""
    pool = r._pool
    pi = _pi(pool)
    return Curvilinear("polar", [r, theta], [(0, radius), (-pi, pi)])


def Annulus(r, theta, inner, outer) -> Curvilinear:
    """``inner < r < outer``, ``θ ∈ (−π, π)``."""
    pi = _pi(r._pool)
    return Curvilinear("polar", [r, theta], [(inner, outer), (-pi, pi)])


def Sector(r, theta, radius, angle, start=0) -> Curvilinear:
    """``0 ≤ r < radius``, ``start < θ < start + angle`` (boundary conditions
    on the two edges are needed)."""
    pool = r._pool
    start = _as_expr(start, pool)
    return Curvilinear("polar", [r, theta], [(0, radius), (start, start + _as_expr(angle, pool))])


def Cylinder(r, theta, z, radius, z_range) -> Curvilinear:
    """``0 ≤ r < radius``, ``θ ∈ (−π, π)``, ``z_range[0] < z < z_range[1]``
    (``theta=None`` for an axisymmetric problem)."""
    pi = _pi(r._pool)
    return Curvilinear(
        "cylindrical", [r, theta, z], [(0, radius), (-pi, pi), (z_range[0], z_range[1])]
    )


def SphericalBall(r, theta, phi, radius, inner=0) -> Curvilinear:
    """The ball ``inner ≤ r < radius`` in spherical coordinates (``θ`` polar,
    ``φ`` azimuth; ``None`` for a coordinate the unknown does not depend
    on)."""
    pi = _pi(r._pool)
    return Curvilinear("spherical", [r, theta, phi], [(inner, radius), (0, pi), (-pi, pi)])


def curvilinear_laplacian(f, system: str, coords):
    """The Laplacian of ``f`` in ``"polar"``, ``"cylindrical"`` or
    ``"spherical"`` coordinates (``None`` for a coordinate ``f`` does not
    depend on)."""
    return _pde_curvilinear_laplacian(f, system, list(coords))


# ---------------------------------------------------------------------------
# Separation
# ---------------------------------------------------------------------------


class Separation:
    """The result of :func:`separate`: ``kind``, ``unknown``, ``ansatz``
    (``X(x)*T(t)`` or ``X(x) + T(t)``), ``factors``, ``constants`` and
    ``equations`` — one dict per variable, in the order they were split off:
    ``{"var", "factor", "equation" (= 0), "constant" (None for the last),
    "multiplier"}``."""

    __slots__ = ("ansatz", "constants", "equations", "factors", "kind", "unknown")

    def __init__(self, d: dict):
        self.kind = d["kind"]
        self.unknown = d["unknown"]
        self.ansatz = d["ansatz"]
        self.factors = d["factors"]
        self.constants = d["constants"]
        self.equations = d["equations"]

    @property
    def odes(self) -> list:
        """The separated ODEs (each ``= 0``)."""
        return [e["equation"] for e in self.equations]

    def __repr__(self) -> str:
        return (
            f"Separation({self.kind}: {self.ansatz}; "
            + "; ".join(f"{e['equation']} = 0" for e in self.equations)
            + ")"
        )


def separate(
    equation, unknown=None, kind: str = "product", *, prefer=None, last=None
) -> Separation:
    """Separate ``equation`` (an ``Expr`` read as ``= 0``) in the arguments of
    ``unknown``.  ``prefer``: variables to split off first; ``last``: one to
    keep for last.  Raises ``PdeError`` ``E-PDE-200`` when it does not
    separate."""
    from alkahest.experimental._pde import PDE, _residual

    eq = _residual(equation)
    if unknown is None:
        unknown = PDE(eq).unknowns[0]
    return Separation(_pde_separate(eq, unknown, kind, prefer, last))


def is_separable(equation, unknown=None, kind: str = "product") -> bool:
    """Whether :func:`separate` succeeds."""
    try:
        separate(equation, unknown, kind)
    except Exception as e:  # PdeError
        if getattr(e, "code", "") == "E-PDE-200":
            return False
        raise
    return True


class ModeGroup:
    """One group of modes: ``indices`` (``(n, first)`` pairs), ``upper`` (the
    last value of each index: ``oo``, or finite when the range was split around
    an exceptional index), ``eigenvalues``
    (``(separation constant, value)``), ``eigenfunctions`` (product
    eigenfunctions; cosine and sine variants for a periodic direction),
    ``weight``, ``norms_squared``, ``coefficients`` (per eigenfunction, a dict
    ``{"ic" | "ic1" | "left" | "right" | "source": c}``), ``amplitudes``
    (``T(s)`` per eigenfunction), ``term`` (the summand), ``value`` (the
    nested ``Sum``), ``closed`` and ``status``."""

    __slots__ = (
        "amplitudes",
        "closed",
        "coefficients",
        "eigenfunctions",
        "eigenvalues",
        "indices",
        "norms_squared",
        "status",
        "term",
        "upper",
        "value",
        "weight",
    )
    value: object

    def __init__(self, d: dict):
        for k in self.__slots__:
            setattr(self, k, d[k])

    def __repr__(self) -> str:
        return f"ModeGroup({self.value})"


class SeriesSolution:
    """The result of :func:`separation_solve`.

    ``value`` (``u``), ``kind`` (``"series"`` / ``"closed_form"``),
    ``method``, ``steady_state`` (lifting or steady state, or ``None``),
    ``modes`` (:class:`ModeGroup`), ``separation`` (:class:`Separation`),
    ``verification`` (``status``, ``verified``, ``checks``: tuples ``(what,
    group, zero_status, method, detail)``), ``projection`` (numeric checks of
    the coefficients against quadratures; they may only refute),
    ``convergence`` (``"uniform"`` / ``"l2"``), ``side_conditions``,
    ``notes``, ``alternatives`` (``("poisson_integral", expr)`` for the
    disk), ``coordinates`` / ``curvilinear_value`` (for a problem posed on a
    Cartesian ``Disk`` / ``Ball``), ``constants`` (arbitrary constants of a
    Neumann problem).
    """

    __slots__ = (
        "alternatives",
        "constants",
        "convergence",
        "coordinates",
        "curvilinear_value",
        "kind",
        "method",
        "modes",
        "notes",
        "projection",
        "separation",
        "side_conditions",
        "steady_state",
        "value",
        "verification",
    )

    def __init__(self, d: dict):
        self.value = d["value"]
        self.kind = d["kind"]
        self.method = d["method"]
        self.steady_state = d["steady_state"]
        self.modes = [ModeGroup(g) for g in d["modes"]]
        self.separation = None if d["separation"] is None else Separation(d["separation"])
        self.verification = d["verification"]
        self.projection = d["projection"]
        self.convergence = d["convergence"]
        self.side_conditions = d["side_conditions"]
        self.notes = d["notes"]
        self.alternatives = d["alternatives"]
        self.coordinates = d["coordinates"]
        self.curvilinear_value = d["curvilinear_value"]
        self.constants = d["constants"]

    @property
    def verified(self) -> bool:
        """``True`` when every mode, amplitude, eigen-condition and
        coefficient check is a proof (``verification["status"] ==
        "exactly_verified"``)."""
        return self.verification["verified"]

    @property
    def status(self) -> str:
        return self.verification["status"]

    def truncate(self, n: int):
        """The partial sum with every index ``≤ n`` (an explicit expression;
        ``RootOfFamily`` members stay symbolic)."""
        from alkahest.experimental import truncate as _truncate

        return _truncate(self.value, n)

    def numeric(self, n: int, params: dict | None = None):
        """The partial sum with indices ``≤ n``, ``params`` substituted,
        ``RootOfFamily`` members replaced by floats (certified enclosures'
        midpoints) and unevaluated coefficient integrals by quadratures."""
        return _pde_numeric_truncation(self.value, n, params)

    def compile(self, n: int, inputs, params: dict | None = None, **kw):
        """A compiled function of ``inputs`` (JIT; sums are evaluated in
        multivariate Horner form) evaluating the numeric partial sum
        :meth:`numeric` — for plots and cross-checks."""
        return _ak.compile_expr(self.numeric(n, params), list(inputs), **kw)

    def __repr__(self) -> str:
        return f"SeriesSolution({self.kind}, {self.status}: u = {self.value})"


def separation_solve(
    pde, *, check_members: int = 4, steady_state: bool = True, index_names=None
) -> SeriesSolution:
    """Solve a linear :class:`PDE` on a bounded separable domain by separation
    of variables and eigenfunction expansion (see the module docs).

    ``check_members``: members per index compared in the numeric projection
    check (0 to skip); ``steady_state``: prefer a steady state to a
    polynomial lifting for time-independent boundary data; ``index_names``:
    summation index names in separation order.

    Raises ``PdeError`` ``E-PDE-200…209`` (and ``E-PDE-15x`` from the
    Sturm–Liouville solver) when the problem is outside the method.
    """
    if pde.domain is None:
        raise ValueError("declare the domain: pde.on(Interval(x, 0, L) * HalfLine(t, 0))")
    return SeriesSolution(
        _pde_separation_solve(
            pde.equations,
            pde.unknowns,
            pde.domain.spec(),
            list(pde._conditions),
            check_members,
            steady_state,
            None if index_names is None else list(index_names),
        )
    )
