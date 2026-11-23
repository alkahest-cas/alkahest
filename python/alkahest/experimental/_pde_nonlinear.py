"""Nonlinear PDEs (milestone M6): Lie point symmetries, conservation laws,
reductions to ODEs and linearising transformations.

The equivalents of Maple PDEtools' ``DeterminingPDE`` / ``Infinitesimals``
(:func:`determining_equations`, :func:`pde_symmetries`),
``ConservedCurrents`` (:func:`conservation_laws`), ``TWSolutions``
(:func:`traveling_wave_solutions`), ``InvariantSolutions`` /
``SimilaritySolutions`` (:func:`invariant_solutions`,
:func:`similarity_solutions`).

Everything returned has been **verified exactly**: each symmetry generator by
recomputing ``pr X(F)`` on solutions, each conservation law by
``D_t T + D_x X = Λ·F`` as an identity, each solution by substitution into
the equation (anything that does not verify is discarded), each
transformation as a jet identity.  Refusals raise :class:`PdeError` with
codes ``E-PDE-300 … E-PDE-349``::

    >>> import alkahest as ak, alkahest.experimental as akx
    >>> p = ak.ExprPool(); x, t = p.symbol("x"), p.symbol("t")
    >>> u = akx.Function("u")(x, t)
    >>> S = akx.pde_symmetries(akx.D(u, t) + u * akx.D(u, x) + akx.D(u, x, 3), u)
    >>> len(S.generators), S.complete      # KdV: 4 generators
    (4, True)

Expressions of symmetries and conservation laws are written with the
independent variables and *jet symbols* ``u``, ``u_x``, ``u_xx``, …
(:class:`JetSpace` converts them back to derivative nodes).
"""

from __future__ import annotations

import alkahest as _ak
from alkahest.alkahest import (
    _pde_cole_hopf,
    _pde_commutator,
    _pde_conservation_laws,
    _pde_conserved_flux,
    _pde_determining_equations,
    _pde_euler_operator,
    _pde_hodograph,
    _pde_hopf_lax,
    _pde_miura,
    _pde_multipliers,
    _pde_reduce_by_generator,
    _pde_reduced_solutions,
    _pde_symmetries,
    _pde_verify_conservation_law,
    _pde_verify_symmetry,
)
from alkahest.experimental._pde import _residual, _unknowns_of

__all__ = [
    "ConservationLaw",
    "PointSymmetries",
    "ReducedSolution",
    "SymmetryGenerator",
    "ansatz_solutions",
    "cole_hopf",
    "commutator",
    "conservation_laws",
    "conservation_multipliers",
    "conserved_flux",
    "determining_equations",
    "euler_operator",
    "hodograph",
    "hopf_lax",
    "invariant_solutions",
    "miura",
    "pde_symmetries",
    "reduce_by_generator",
    "similarity_solutions",
    "traveling_wave_solutions",
    "verify_conservation_law",
    "verify_symmetry",
]


def _eq_u(eq, u):
    e = _residual(eq)
    if u is None:
        found = _unknowns_of([e])
        if len(found) != 1:
            raise ValueError("give the unknown u(x, t) explicitly")
        u = found[0]
    return e, u


def _indep(u) -> list:
    node = u.node()
    if node[0] != "func":
        raise TypeError("the unknown must be an applied function u(x, t)")
    return list(node[2])


class SymmetryGenerator:
    """A point symmetry ``X = Σ ξⁱ ∂_{xᵢ} + η ∂_u``.

    Attributes: ``xi`` (list, one per independent variable), ``eta``,
    ``verified`` (``pr X(F) = 0`` on solutions proved exactly), ``operator``
    (text such as ``"x∂_x + 2t∂_t"``)."""

    __slots__ = ("eta", "operator", "verified", "xi")

    def __init__(self, d: dict):
        self.xi = list(d["xi"])
        self.eta = d["eta"]
        self.verified = bool(d["verified"])
        self.operator = d["operator"]

    @property
    def components(self) -> list:
        """``[ξ¹, …, ξ^p, η]``."""
        return [*self.xi, self.eta]

    def __repr__(self) -> str:
        return f"SymmetryGenerator({self.operator})"


class PointSymmetries:
    """The Lie point symmetries of one equation (one parameter case).

    Attributes
    ----------
    generators : list[SymmetryGenerator]
        A basis of the finite-dimensional part, each verified.
    structure_constants : dict or None
        ``{(a, b, c): C}`` (1-based) with ``[X_a, X_b] = Σ_c C_abc X_c``;
        ``None`` when a commutator is not in the span.
    infinite : dict or None
        ``{"components", "functions", "constraints", "kind"}``: generators
        with arbitrary functions ``F1(…)`` subject to linear PDEs.  ``kind``
        is ``"superposition"`` (``α(x)∂_u``, ``α`` any solution of a linear
        equation), ``"passive"`` (the general solution of a completed linear
        system, e.g. the conformal group of the wave equation) or
        ``"residual"`` (left unsolved; ``complete`` is then false).
    complete : bool
        The answer is the whole point-symmetry algebra.
    case : list[tuple]
        Parameter choices of this case (``[(m, -4/3)]``); empty for the
        generic case.
    excluded : list[tuple]
        Parameter values where the generic answer may change (each analysed
        as its own case by ``pde_symmetries(..., cases=True)``).
    determining_equations : list[Expr]
    notes : list[str]
    """

    def __init__(self, d: dict):
        self.independent = list(d["independent"])
        self.unknown = d["unknown"]
        self.dependent = d["dependent"]
        self.equation = d["equation"]
        self.generators = [SymmetryGenerator(g) for g in d["generators"]]
        sc = d["structure_constants"]
        self.structure_constants = (
            None if sc is None else {(a + 1, b + 1, c + 1): v for (a, b, c, v) in sc}
        )
        self.infinite = d["infinite"]
        self.complete = bool(d["complete"])
        self.case = list(d["case"])
        self.nonzero = list(d["nonzero"])
        self.excluded = list(d["excluded"])
        self.determining_equations = list(d["determining_equations"])
        self.notes = list(d["notes"])

    def __len__(self) -> int:
        return len(self.generators)

    def __iter__(self):
        return iter(self.generators)

    @property
    def dimension(self):
        """The dimension of the finite part, or ``float('inf')`` when an
        infinite-dimensional part is present."""
        return float("inf") if self.infinite else len(self.generators)

    def lie_algebra(self, names: str = "X", *, check_jacobi: bool = True):
        """The finite part as an :class:`alkahest.LieAlgebra` on
        noncommutative generators ``X1, X2, …`` with ``[X_a, X_b] =
        Σ C_abc X_c`` (stored as ``f_abc = -i C_abc``)."""
        if self.structure_constants is None:
            raise ValueError("the generators do not close under commutation")
        pool = _pool_of_expr(self.dependent)
        gens = [pool.symbol(f"{names}{i + 1}", commutative=False) for i in range(len(self))]
        table = {k: -pool.imaginary_unit() * v for k, v in self.structure_constants.items()}
        return _ak.lie_algebra(gens, table, check_jacobi=check_jacobi)

    def __repr__(self) -> str:
        inf = f" + infinite[{self.infinite['kind']}]" if self.infinite else ""
        case = f", case={self.case}" if self.case else ""
        ops = ", ".join(g.operator for g in self.generators)
        return f"PointSymmetries([{ops}]{inf}, complete={self.complete}{case})"


def _pool_of_expr(e):
    return e._pool


def pde_symmetries(eq, u=None, *, cases: bool = False):
    """The Lie point symmetries of one PDE ``eq = 0`` in one unknown.

    Returns the generic-case :class:`PointSymmetries`; ``cases=True`` returns
    the list of every parameter case (group classification: e.g. ``m = -4/3``
    and ``m = 0`` for ``u_t = (u^m u_x)_x``).

    Method: determining equations from the prolongation formula on the jet
    space, split by the parametric derivatives; RIF-style heuristics
    (splitting, trivial-equation integration, elimination, ``dsolve`` for
    linear ODEs) with a linear differential-Gröbner completion as backstop;
    case splits on parameters.  Each generator is verified independently.
    Raises :class:`PdeError` ``E-PDE-300`` / ``E-PDE-301``.
    """
    e, u = _eq_u(eq, u)
    out = [PointSymmetries(d) for d in _pde_symmetries(e, u)]
    if cases:
        return out
    for c in out:
        if not c.case:
            return c
    return out[0]


def determining_equations(eq, u=None) -> list:
    """The determining equations of the point symmetries (``= 0``), in the
    infinitesimals ``xi_x(x, t, u)``, …, ``eta(x, t, u)``."""
    e, u = _eq_u(eq, u)
    return list(_pde_determining_equations(e, u))


def verify_symmetry(eq, u, generator) -> str:
    """Zero status (``"proved_zero"`` for a symmetry) of ``pr X(F)`` on
    solutions; ``generator`` is a :class:`SymmetryGenerator` or the list
    ``[ξ¹, …, ξ^p, η]`` written in the variables and the symbol ``u``."""
    e, u = _eq_u(eq, u)
    comps = generator.components if isinstance(generator, SymmetryGenerator) else list(generator)
    return _pde_verify_symmetry(e, u, comps)


def commutator(variables, X, Y) -> list:
    """``[X, Y]`` of vector fields given by components over ``variables``."""
    X = X.components if isinstance(X, SymmetryGenerator) else X
    Y = Y.components if isinstance(Y, SymmetryGenerator) else Y
    return list(_pde_commutator(list(variables), list(X), list(Y)))


class ConservationLaw:
    """``Σ_i D_{x_i} Φ^i = Λ·F``: ``multiplier`` (Λ), ``components``
    (``Φ^i`` per independent variable), ``density`` / ``flux`` (the ``t``
    component and the rest), ``verified`` (proved exactly)."""

    __slots__ = ("components", "independent", "multiplier", "verified")

    def __init__(self, d: dict, independent: list):
        self.multiplier = d["multiplier"]
        self.components = list(d["components"])
        self.verified = bool(d["verified"])
        self.independent = independent

    def _time(self):
        names = [str(v) for v in self.independent]
        return names.index("t") if "t" in names else len(names) - 1

    @property
    def density(self):
        """The component along ``t`` (or the last variable)."""
        return self.components[self._time()] if self.components else None

    @property
    def flux(self):
        """The remaining component(s): one ``Expr`` in two variables, a list
        otherwise."""
        if not self.components:
            return None
        rest = [c for i, c in enumerate(self.components) if i != self._time()]
        return rest[0] if len(rest) == 1 else rest

    def __repr__(self) -> str:
        return (
            f"ConservationLaw(multiplier={self.multiplier}, density={self.density}, "
            f"flux={self.flux}, verified={self.verified})"
        )


def conservation_multipliers(eq, u=None, order: int = 2, *, cases: bool = False):
    """Multipliers ``Λ`` with ``E_u(Λ·F) ≡ 0``, depending on the variables,
    ``u`` and the non-leading derivatives up to ``order``: a dict
    ``{"multipliers", "infinite", "complete", "coordinates", "case",
    "notes"}`` (generic case; ``cases=True`` for all)."""
    e, u = _eq_u(eq, u)
    out = list(_pde_multipliers(e, u, order))
    if cases:
        return out
    for c in out:
        if not c["case"]:
            return c
    return out[0]


def conservation_laws(eq, u=None, order: int = 2) -> list:
    """Every conservation law with a multiplier of order ``≤ order`` (the
    generic case), each with its verified density and flux; a multiplier
    whose flux was not recovered is kept with ``verified=False``."""
    e, u = _eq_u(eq, u)
    ind = _indep(u)
    return [ConservationLaw(d, ind) for d in _pde_conservation_laws(e, u, order)]


def conserved_flux(eq, u, multiplier) -> ConservationLaw:
    """The density/flux of the conservation law with ``multiplier``
    (``E-PDE-306`` when it is not a divergence)."""
    e, u = _eq_u(eq, u)
    return ConservationLaw(_pde_conserved_flux(e, u, multiplier), _indep(u))


def euler_operator(lagrangian, u):
    """The variational derivative ``E_u(L) = Σ_J (−D)_J ∂L/∂u_J``."""
    return _pde_euler_operator(lagrangian, u)


def verify_conservation_law(eq, u, multiplier, components) -> str:
    """Zero status of ``Σ D_i Φ^i − Λ·F`` (``"proved_zero"`` when it is a
    conservation law)."""
    e, u = _eq_u(eq, u)
    return _pde_verify_conservation_law(e, u, multiplier, list(components))


class ReducedSolution:
    """A verified solution family from a reduction: ``value`` (``u =``),
    ``method`` (``"traveling_wave"``, ``"tanh"``, ``"sech"``, ``"tan"``,
    ``"similarity"``, ``"invariant"``), ``constants``, ``assumptions``,
    ``steps``, ``verification``."""

    __slots__ = ("assumptions", "constants", "method", "steps", "value", "verification")

    def __init__(self, d: dict):
        self.value = d["value"]
        self.method = d["method"]
        self.constants = list(d["constants"])
        self.assumptions = list(d["assumptions"])
        self.steps = list(d["steps"])
        self.verification = d["verification"]

    @property
    def verified(self) -> bool:
        return bool(self.verification.get("verified"))

    def __repr__(self) -> str:
        return f"ReducedSolution(u = {self.value}, method={self.method!r})"


def traveling_wave_solutions(eq, u=None) -> list:
    """Traveling waves ``u = F(x − ct)``: the reduced ODE integrated as far
    as it is exact and solved by ``dsolve`` (general solutions first),
    separatrices of autonomous second-order ODEs (solitons, kinks), then the
    tanh / sech / tan expansion methods (special speeds).  Each verified."""
    e, u = _eq_u(eq, u)
    return [ReducedSolution(d) for d in _pde_reduced_solutions(e, u, "traveling_wave")]


def ansatz_solutions(eq, u=None, method: str = "tanh") -> list:
    """Expansion-method solutions ``u = Σ aᵢ Y(k(x − ct))ⁱ`` with ``Y`` =
    ``tanh``, ``sech`` or ``tan``; the pole order from balancing and the
    coefficient system solved by Gröbner bases.  Each verified."""
    e, u = _eq_u(eq, u)
    return [ReducedSolution(d) for d in _pde_reduced_solutions(e, u, method)]


def invariant_solutions(eq, u, generator) -> list:
    """Solutions invariant under ``generator`` (a :class:`SymmetryGenerator`
    or ``[ξ_x, ξ_t, η]``).  Each verified."""
    e, u = _eq_u(eq, u)
    comps = generator.components if isinstance(generator, SymmetryGenerator) else list(generator)
    return [ReducedSolution(d) for d in _pde_reduced_solutions(e, u, "invariant", comps)]


def similarity_solutions(eq, u=None) -> list:
    """Invariant solutions under every point-symmetry generator and under
    combinations ``X + λY`` made exact by the choice of ``λ`` (the heat
    source and Barenblatt solutions).  Each verified."""
    e, u = _eq_u(eq, u)
    return [ReducedSolution(d) for d in _pde_reduced_solutions(e, u, "similarity")]


def reduce_by_generator(eq, u, generator) -> dict:
    """The reduction by a generator: ``{"variable", "symbol", "form",
    "ode", "function"}`` — similarity variable ``s(x, t)``, the invariant
    form ``u = ψ(x, t, F(s))`` and the ODE for ``F``."""
    e, u = _eq_u(eq, u)
    comps = generator.components if isinstance(generator, SymmetryGenerator) else list(generator)
    return dict(_pde_reduce_by_generator(e, u, comps))


def cole_hopf(eq, u=None) -> dict:
    """Cole–Hopf for ``u_t + a u u_x = ν u_xx``: ``{"substitution",
    "heat_equation", "theta", "viscosity", "coefficient", "verified"}``."""
    e, u = _eq_u(eq, u)
    return dict(_pde_cole_hopf(e, u))


def miura(eq, v, u) -> dict:
    """The Miura map of ``v_t + a v² v_x + b v_xxx = 0``: ``u = α v² + β v_x``
    solving a KdV equation in ``u`` — ``{"substitution", "kdv",
    "kdv_unknown", "verified"}``."""
    e, v = _eq_u(eq, v)
    return dict(_pde_miura(e, v, u))


def hodograph(equations, unknowns) -> dict:
    """The hodograph transformation of a 2×2 homogeneous quasilinear
    system: the linear system for ``x(u, v)``, ``t(u, v)``."""
    eqs = [_residual(q) for q in equations]
    return dict(_pde_hodograph(eqs, list(unknowns)))


def hopf_lax(eq, u=None) -> dict:
    """The Hopf–Lax data of ``u_t + H(u_x) = 0``: Hamiltonian, Lagrangian
    (Legendre transform, duality verified) and the objective
    ``g(y) + t L((x − y)/t)`` minimised over ``y``."""
    e, u = _eq_u(eq, u)
    return dict(_pde_hopf_lax(e, u))
