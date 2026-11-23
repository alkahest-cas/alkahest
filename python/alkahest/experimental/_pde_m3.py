"""PDE milestone M3: the Sturm–Liouville eigenproblem solver and the
transform methods.

Everything here is experimental and re-exported from
:mod:`alkahest.experimental`.

Sturm–Liouville
---------------
:func:`sturm_liouville` solves ``(p X')' + q X + λ w X = 0`` on ``(a, b)``
with separated end conditions (``"dirichlet"``, ``"neumann"``,
``("robin", alpha, beta)`` for ``alpha*X + beta*X' = 0``, and ``"bounded"`` /
``"decay"`` at a singular or infinite end) or periodic ones::

    >>> import alkahest as ak, alkahest.experimental as akx
    >>> p = ak.ExprPool(); x = p.symbol("x"); L = p.symbol("L", "positive")
    >>> es = akx.sturm_liouville(x, (0, L), ("dirichlet", "dirichlet"))
    >>> es.families[0].eigenvalue        # (n*pi/L)**2
    >>> es.families[0].eigenfunctions[0]  # sin(n*pi*x/L)

A Robin end gives a :func:`RootOfFamily` with a pole-free defining equation
(``h*sin(nu*L) + nu*cos(nu*L) = 0``) and, once the parameters are numbers,
certified enclosures (:meth:`EigenFamily.enclosures`).  Bessel, Legendre,
Chebyshev, Hermite and Laguerre problems are recognised from ``p, q, w`` and
the ends, or built with :meth:`SturmLiouville.bessel` etc.

Transform methods
-----------------
:func:`transform_solve` takes a :class:`PDE` on ``Line(x) * HalfLine(t, 0)``
(Fourier in ``x``) or ``HalfLine(x, 0) * HalfLine(t, 0)`` (Laplace in ``t`` for
boundary data, sine / cosine transforms for initial data) and returns a
:class:`TransformSolution`: a closed form, or a kernel representation
``∫ G(x - s, t) f(s) ds`` (``kind == "integral_representation"``) whose kernel
was verified by substitution.
"""

from __future__ import annotations

import contextlib

import alkahest as _ak
from alkahest.alkahest import (
    _pde_sl_enclosures,
    _pde_sl_from_ode,
    _pde_sturm_liouville,
    _pde_transform_solve,
    _sincos_transform,
)

__all__ = [
    "EigenFamily",
    "EigenMode",
    "Eigensystem",
    "SturmLiouville",
    "TransformSolution",
    "fourier_cosine_transform",
    "fourier_sine_transform",
    "inverse_fourier_cosine_transform",
    "inverse_fourier_sine_transform",
    "sturm_liouville",
    "transform_solve",
]


def _pool_of(*objs):
    for o in objs:
        if isinstance(o, _ak.Expr):
            return o._pool
        if isinstance(o, (list, tuple)):
            p = _pool_of(*o)
            if p is not None:
                return p
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
    if isinstance(v, str) and v in ("oo", "inf", "+oo"):
        return pool.symbol("oo", "positive")
    if isinstance(v, str) and v in ("-oo", "-inf"):
        return -pool.symbol("oo", "positive")
    raise TypeError(f"cannot use {v!r} as an expression")


def _end(spec, pool):
    if isinstance(spec, str):
        return spec
    if isinstance(spec, (tuple, list)) and len(spec) == 3 and str(spec[0]).lower() == "robin":
        return ("robin", _as_expr(spec[1], pool), _as_expr(spec[2], pool))
    raise ValueError(
        "an end condition is 'dirichlet', 'neumann', 'bounded', 'decay' or ('robin', alpha, beta)"
    )


class EigenFamily:
    """One family ``(λ_n, X_n)``, ``n = first, first + 1, …``.

    Attributes: ``index`` (the integer symbol ``n``), ``first``,
    ``frequency`` (``ν_n``: closed form or a ``RootOfFamily``),
    ``frequency_var``, ``eigenvalue`` (``λ_n``), ``eigenvalue_of_frequency``
    (``λ`` as a function of ``ν``), ``eigenfunctions`` (one; two for periodic
    problems), ``norms_squared`` (``∫ w X_n²``, ``None`` when it did not
    close), ``closed_form``, ``defining_equation`` (``F(ν) = 0``, pole-free),
    ``enclosures`` (certified ``(n, lo, hi)`` of the first eigenvalues when the
    problem is numeric) and ``verification`` (``status``, ``ode``, ``left``,
    ``right``, ``method``, ``members``).
    """

    __slots__ = (
        "closed_form",
        "defining_equation",
        "eigenfunctions",
        "eigenfunctions_of_frequency",
        "eigenvalue",
        "eigenvalue_of_frequency",
        "enclosures",
        "first",
        "frequency",
        "frequency_var",
        "index",
        "norms_squared",
        "verification",
    )

    def __init__(self, d: dict):
        for k in self.__slots__:
            setattr(self, k, d[k])

    def frequency_at(self, k: int):
        """``ν_k``: the closed form at ``n = k``, or the ``RootOfFamily``
        member."""
        if self.closed_form:
            return _ak.simplify(_ak.subs(self.frequency, {self.index: k}))
        from alkahest.experimental._pde import root_family_member

        return root_family_member(self.frequency, k)

    def eigenvalue_at(self, k: int):
        """``λ_k``."""
        return _ak.simplify(
            _ak.subs(self.eigenvalue_of_frequency, {self.frequency_var: self.frequency_at(k)})
        )

    def eigenfunctions_at(self, k: int) -> list:
        """The eigenfunction(s) of ``λ_k``."""
        nu = self.frequency_at(k)
        out = []
        for f in self.eigenfunctions_of_frequency:
            e = _ak.subs(f, {self.frequency_var: nu})
            with contextlib.suppress(Exception):  # no orthogonal polynomial inside
                e = _ak.expand_orthopoly(e)
            out.append(_ak.simplify(e))
        return out

    def enclosures_with(self, count: int, params: dict | None = None, bits: int = 64) -> list:
        """Certified enclosures ``(n, lo, hi)`` of the first ``count``
        eigenvalues after substituting ``params`` (symbol → number)."""
        return _pde_sl_enclosures(
            self.index,
            self.first,
            self.frequency_var,
            self.frequency,
            self.eigenvalue_of_frequency,
            self.closed_form,
            count,
            params or {},
            bits,
        )

    def __repr__(self) -> str:
        return (
            f"EigenFamily(n >= {self.first}: lambda_n = {self.eigenvalue}, "
            f"X_n = {self.eigenfunctions[0]}, {self.verification['status']})"
        )


class EigenMode:
    """An isolated eigenpair (``λ₀`` itself, possibly under a ``condition``)."""

    __slots__ = ("condition", "eigenfunction", "eigenvalue", "norm_squared", "verification")

    def __init__(self, d: dict):
        for k in self.__slots__:
            setattr(self, k, d[k])

    def __repr__(self) -> str:
        cond = "" if self.condition is None else f" if {self.condition}"
        return f"EigenMode(lambda = {self.eigenvalue}, X = {self.eigenfunction}{cond})"


class Eigensystem:
    """The solution of a Sturm–Liouville problem.

    ``kind`` (``"regular_constant"``, ``"regular"``, ``"periodic"``,
    ``"bessel"``, ``"legendre"``, ``"chebyshev"``, ``"hermite"``,
    ``"laguerre"``), ``var``, ``weight``, ``interval``, ``threshold`` (``λ₀``,
    where the frequency vanishes), ``families`` (:class:`EigenFamily`),
    ``modes`` (:class:`EigenMode`), ``lower`` (``{"status": "none" |
    "conditional" | "present", "condition", "family"}`` — eigenvalues below
    ``λ₀``), ``side_conditions`` and ``notes``.
    """

    def __init__(self, d: dict):
        self.kind = d["kind"]
        self.var = d["var"]
        self.weight = d["weight"]
        self.interval = d["interval"]
        self.threshold = d["threshold"]
        self.families = [EigenFamily(f) for f in d["families"]]
        self.modes = [EigenMode(m) for m in d["modes"]]
        lower = dict(d["lower"])
        if "family" in lower:
            lower["family"] = EigenFamily(lower["family"])
        self.lower = lower
        self.side_conditions = d["side_conditions"]
        self.notes = d["notes"]

    @property
    def verified(self) -> bool:
        """Every family and mode exactly verified."""
        return all(f.verification["verified"] for f in self.families) and all(
            m.verification == "exactly_verified" for m in self.modes if m.condition is None
        )

    def __repr__(self) -> str:
        return (
            f"Eigensystem({self.kind}, families={self.families}, modes={self.modes}, "
            f"lower={self.lower['status']})"
        )


class SturmLiouville:
    """A Sturm–Liouville problem ``(p X')' + q X + λ w X = 0`` on
    ``interval`` with ``bc = (left, right)`` or ``bc = "periodic"``."""

    def __init__(self, var, interval, bc, *, p=1, q=0, w=1):
        pool = _pool_of(var, list(interval), p, q, w)
        self.var = var
        self.interval = tuple(_as_expr(e, pool) for e in interval)
        self.p, self.q, self.w = (_as_expr(e, pool) for e in (p, q, w))
        self.bc = bc
        self._ode = None

    @classmethod
    def from_ode(cls, equation, unknown, eigenvalue, interval, bc):
        """Read ``A X'' + B X' + (C + λ D) X = 0`` off ``equation`` (an
        ``Expr`` in the applied unknown ``X(x)`` and its derivatives)."""
        self = cls.__new__(cls)
        pool = _pool_of(unknown)
        self.var = None
        self.interval = tuple(_as_expr(e, pool) for e in interval)
        self.bc = bc
        self._ode = (equation, unknown, eigenvalue)
        return self

    @classmethod
    def bessel(cls, r, order, radius, bc="dirichlet"):
        """``(r R')' − (ν²/r) R + λ r R = 0`` on ``(0, radius)``, bounded at 0."""
        pool = _pool_of(r)
        nu = _as_expr(order, pool)
        return cls(r, (0, radius), ("bounded", bc), p=r, q=-(nu**2) / r, w=r)

    @classmethod
    def legendre(cls, x):
        """``((1 − x²) X')' + λ X = 0`` on ``[−1, 1]``."""
        return cls(x, (-1, 1), ("bounded", "bounded"), p=1 - x**2)

    @classmethod
    def chebyshev(cls, x):
        """``(√(1−x²) X')' + λ X / √(1−x²) = 0`` on ``[−1, 1]``."""
        s = _ak.sqrt(1 - x**2)
        return cls(x, (-1, 1), ("bounded", "bounded"), p=s, w=1 / s)

    @classmethod
    def hermite(cls, x):
        """``(e^{−x²} X')' + λ e^{−x²} X = 0`` on ``ℝ``."""
        g = _ak.exp(-(x**2))
        return cls(x, ("-oo", "oo"), ("decay", "decay"), p=g, w=g)

    @classmethod
    def laguerre(cls, x):
        """``(x e^{−x} X')' + λ e^{−x} X = 0`` on ``(0, ∞)``."""
        e = _ak.exp(-x)
        return cls(x, (0, "oo"), ("bounded", "decay"), p=x * e, w=e)

    def solve(self, *, enclosures: int = 5, bits: int = 64, index: str = "n") -> Eigensystem:
        """Solve; see :func:`sturm_liouville`."""
        periodic = isinstance(self.bc, str) and self.bc.lower() == "periodic"
        if periodic:
            left = right = "dirichlet"
        else:
            left, right = self.bc
        a, b = self.interval
        pool = _pool_of(a, b, self.var, self._ode)
        left, right = _end(left, pool), _end(right, pool)
        if self._ode is not None:
            eq, u, lam = self._ode
            d = _pde_sl_from_ode(eq, u, lam, a, b, left, right, periodic, enclosures, bits, index)
        else:
            d = _pde_sturm_liouville(
                self.var,
                self.p,
                self.q,
                self.w,
                a,
                b,
                left,
                right,
                periodic,
                enclosures,
                bits,
                index,
            )
        return Eigensystem(d)


def sturm_liouville(
    var, interval, bc, *, p=1, q=0, w=1, enclosures: int = 5, bits: int = 64, index: str = "n"
) -> Eigensystem:
    """Solve ``(p X')' + q X + λ w X = 0`` on ``interval = (a, b)``.

    ``bc`` is ``(left, right)`` — each ``"dirichlet"``, ``"neumann"``,
    ``("robin", alpha, beta)`` (``alpha*X + beta*X' = 0`` at that end, literal
    signs), ``"bounded"`` or ``"decay"`` (singular / infinite ends of the
    recognised classical problems) — or ``"periodic"``.

    Returns an :class:`Eigensystem`.  Raises ``PdeError`` with ``E-PDE-150``
    (unsupported), ``151`` (bad end condition), ``152`` (degenerate) or ``153``
    (an eigenpair refuted by its exact check — never returned).
    """
    return SturmLiouville(var, interval, bc, p=p, q=q, w=w).solve(
        enclosures=enclosures, bits=bits, index=index
    )


class TransformSolution:
    """The result of :func:`transform_solve`.

    ``kind`` (``"closed_form"`` / ``"integral_representation"``), ``method``,
    ``value`` (``u``, with ``Integral(integrand, s, lo, hi)`` nodes for a
    representation), ``closed_part``, ``terms`` (``{"kernel", "density",
    "var", "lower", "upper"}`` per integral), ``transformed`` (the solution in
    transform space), ``transform_var``, ``verification`` (``status``,
    ``verified``, ``checks``: ``(what, index, zero_status)``),
    ``side_conditions`` and ``notes``.
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    @property
    def verified(self) -> bool:
        return self.verification["verified"]

    def __repr__(self) -> str:
        return f"TransformSolution({self.kind}, {self.method}: u = {self.value})"


def transform_solve(pde, method: str = "auto") -> TransformSolution:
    """Solve a linear constant-coefficient :class:`PDE` by a transform method.

    ``method``: ``"auto"`` (by the domain), ``"fourier"`` (``x`` on a line,
    ``t > 0``), ``"laplace"`` (``x > 0``, ``t > 0``, zero initial data,
    boundary data at ``x = 0``) or ``"image"`` (``x > 0``, homogeneous
    Dirichlet / Neumann end — sine / cosine transform — and initial data).
    """
    if pde.domain is None:
        raise ValueError("declare the domain: pde.on(Line(x) * HalfLine(t, 0))")
    return TransformSolution(
        _pde_transform_solve(
            pde.equations, pde.unknowns, pde.domain.spec(), list(pde._conditions), method
        )
    )


def fourier_sine_transform(f, x, w):
    """``F_s{f}(w) = ∫₀^∞ f(x) sin(w x) dx`` (table-based)."""
    return _sincos_transform(f, x, w, "sine")


def fourier_cosine_transform(f, x, w):
    """``F_c{f}(w) = ∫₀^∞ f(x) cos(w x) dx`` (table-based)."""
    return _sincos_transform(f, x, w, "cosine")


def inverse_fourier_sine_transform(F, w, x):
    """``f(x) = (2/π) ∫₀^∞ F(w) sin(w x) dw``."""
    return _sincos_transform(F, w, x, "sine", True)


def inverse_fourier_cosine_transform(F, w, x):
    """``f(x) = (2/π) ∫₀^∞ F(w) cos(w x) dw``."""
    return _sincos_transform(F, w, x, "cosine", True)
