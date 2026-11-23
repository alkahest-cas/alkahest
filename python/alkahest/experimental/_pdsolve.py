"""``pdsolve`` — the symbolic PDE solver (milestone M1: first-order equations).

The solver is a dispatcher over *method tiers* (``pdsolve_hints()``); every
answer it returns has been verified exactly by substitution — explicit
solutions through :func:`verify_pde_solution`, implicit relations
``Φ(x, u) = 0`` by implicit differentiation — and anything that does not
verify is discarded.  Refusals raise :class:`PdeError` with codes
``E-PDE-050 … E-PDE-099``::

    >>> import alkahest as ak, alkahest.experimental as akx
    >>> p = ak.ExprPool(); x, y = p.symbol("x"), p.symbol("y")
    >>> u = akx.Function("u")
    >>> sol = akx.pdsolve(2 * akx.D(u(x, y), x) + 3 * akx.D(u(x, y), y), u(x, y))
    >>> sol.kind, sol.verified
    ('general', True)
"""

from __future__ import annotations

from alkahest.alkahest import _pde_solve, _pde_solve_hints
from alkahest.experimental._pde import PDE, _condition, _residual, _unknowns_of

__all__ = ["PdeSolution", "pdsolve", "pdsolve_hints"]


class PdeSolution:
    """A verified answer of :func:`pdsolve`.

    Attributes
    ----------
    value : Expr
        ``u = value`` — or, when :attr:`implicit` is true, the relation
        ``value = 0`` (written with the applied unknown ``u(x, y)`` in it).
    implicit : bool
        Whether :attr:`value` is a relation.
    kind : str
        ``"general"`` (arbitrary functions ``F1, …``), ``"particular"`` (a
        Cauchy problem's solution), ``"implicit"`` (``Φ(ξ₁, ξ₂) = 0`` or an
        implicit Cauchy solution), ``"complete_integral"`` (constants
        ``C1, …``), ``"singular_integral"``.
    method : str
        The tier that produced it (a valid ``hint``).
    arbitrary_functions : list[str]
    constants : list[Expr]
    side_conditions : list[tuple[str, Expr, str]]
        ``(kind, expr, rendered)`` with kind ``"nonzero"`` / ``"positive"``:
        a non-characteristic Jacobian, the breaking time ``t_b - t > 0``, a
        non-vanishing ``Φ_u``.
    verification : dict
        As :func:`verify_pde_solution` (``status`` is always
        ``"exactly_verified"`` for a returned solution).
    steps : list[str]
    alternatives : list[PdeSolution]
        Further verified answers (a singular integral next to a complete
        integral).
    """

    __slots__ = (
        "alternatives",
        "arbitrary_functions",
        "constants",
        "implicit",
        "kind",
        "method",
        "side_conditions",
        "steps",
        "unknown",
        "value",
        "verification",
    )

    def __init__(self, d: dict):
        self.unknown = d["unknown"]
        self.value = d["value"]
        self.implicit = d["implicit"]
        self.kind = d["kind"]
        self.method = d["method"]
        self.arbitrary_functions = list(d["arbitrary_functions"])
        self.constants = list(d["constants"])
        self.side_conditions = list(d["side_conditions"])
        self.verification = d["verification"]
        self.steps = list(d["steps"])
        self.alternatives = [PdeSolution(a) for a in d["alternatives"]]

    @property
    def verified(self) -> bool:
        """``True`` — only exactly verified answers are returned."""
        return bool(self.verification.get("verified"))

    @property
    def rhs(self):
        """The explicit ``u = rhs``; ``None`` for an implicit relation."""
        return None if self.implicit else self.value

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__} | {
            "alternatives": [a.to_dict() for a in self.alternatives]
        }

    def __repr__(self) -> str:
        rel = f"{self.value} = 0" if self.implicit else f"{self.unknown} = {self.value}"
        return f"PdeSolution({rel}, kind={self.kind!r}, method={self.method!r})"


def pdsolve_hints() -> list:
    """The ``hint`` names :func:`pdsolve` accepts, in the order it tries
    them, as ``(name, milestone, description)`` tuples."""
    return list(_pde_solve_hints())


def pdsolve(problem_or_eq, u=None, hint=None, *, ics=None, bcs=None, solvefun="F", all=False):
    """Solve a PDE, returning a verified :class:`PdeSolution`.

    ``problem_or_eq`` is a :class:`PDE` (with ``.ic(...)`` data) or an
    equation (``Expr`` read as ``= 0``, an ``(lhs, rhs)`` pair or an
    equality); ``u`` the applied unknown ``u(x, y)`` (inferred when the
    equation differentiates one function).  ``ics`` / ``bcs`` add data as
    ``(lhs, rhs)`` pairs — for a first-order equation, Cauchy data
    ``u(x, 0) = f(x)`` on a non-characteristic curve.

    ``hint`` restricts to one method (see :func:`pdsolve_hints`);
    ``solvefun`` is the prefix of the arbitrary functions (``"F"`` →
    ``F1, F2, …``); ``all=True`` returns a list with one verified answer per
    applicable method.

    Milestone M1 covers first-order equations: linear and semilinear
    ``Σ aᵢ u_{xᵢ} = c(x, u)`` in any number of variables (characteristics,
    first integrals, ``dsolve`` along characteristics), quasilinear equations
    (Lagrange, implicit ``Φ(ξ₁, ξ₂) = 0``), Cauchy problems (with the
    non-characteristic Jacobian and, for ``u_t + c(u) u_x = 0``, the breaking
    time as side conditions), and fully nonlinear equations (Clairaut,
    ``f(p, q) = 0``, ``f(u, p, q) = 0``, separable / Hamilton–Jacobi,
    Lagrange–Charpit; singular integrals by envelope).

    Raises :class:`PdeError`: ``E-PDE-050`` (no method for this problem, or
    an unknown hint), ``E-PDE-051`` (no candidate verified), ``E-PDE-052``
    (data on a characteristic), ``E-PDE-053`` (data that cannot be imposed).
    """
    if isinstance(problem_or_eq, PDE):
        pde = problem_or_eq
        if u is not None:
            us = list(u) if isinstance(u, (list, tuple)) else [u]
            pde = PDE(pde.equations, us, _domain=pde.domain, _conditions=pde._conditions)
    else:
        eq = _residual(problem_or_eq)
        if u is None:
            found = _unknowns_of([eq])
            u = found[0] if len(found) == 1 else found
        pde = PDE([eq], u)
    conds = list(pde._conditions)
    for c in ics or ():
        conds.append(_condition("ic", c, None))
    for c in bcs or ():
        conds.append(_condition("bc", c, None))
    out = _pde_solve(pde.equations, pde.unknowns, conds, hint, solvefun, all)
    sols = [PdeSolution(d) for d in out]
    return sols if all else sols[0]
