"""Second-order linear PDEs in two independent variables (milestone M2).

``A u_xx + B u_xy + C u_yy + D u_x + E u_y + F u = G``:

* :func:`canonical_form` / :func:`canonical_forms` — characteristic
  coordinates ``ξ = φ(x, y)``, ``η = ψ(x, y)`` from
  ``A dy² − B dx dy + C dx² = 0`` and the hyperbolic ``w_ξη + … = 0``,
  parabolic ``w_ηη + … = 0`` or elliptic ``w_αα + w_ββ + … = 0`` normal form
  (one per region of constant type for variable coefficients);
* :func:`solve_second_order` — closed-form general solutions in arbitrary
  functions (an ODE in one variable; the full symbol factored into
  first-order operators — d'Alembert's ``F(x + ct) + G(x − ct)`` is a case;
  the shift ``u = e^{αx+βy} v``; the Laplace cascade; the canonical form),
  d'Alembert's formula for Cauchy data, and — labelled ``"particular"`` — the
  exponential family ``e^{kx + ω(k) y}`` of the dispersion relation when no
  general solution exists in closed form (heat, Klein–Gordon);
* :func:`laplace_cascade` / :func:`laplace_invariants` — the Laplace cascade
  for ``u_xy + a u_x + b u_y + c u = s`` with variable coefficients, under a
  step budget (termination has no a priori bound: ``E-PDE-103``).

Every returned solution has been substituted back with
:func:`~alkahest.experimental.verify_pde_solution` (arbitrary functions kept
symbolic) and is ``exactly_verified``; anything else raises
:class:`~alkahest.experimental.PdeError` with a code in ``E-PDE-100…149``.

    >>> import alkahest as ak, alkahest.experimental as akx
    >>> p = ak.ExprPool(); x, t, c = p.symbol("x"), p.symbol("t"), p.symbol("c")
    >>> u = akx.Function("u")(x, t)
    >>> s = akx.solve_second_order(akx.D(u, t, 2) - c**2 * akx.D(u, x, 2), u)
    >>> s.kind, s.verification["status"]
    ('general', 'exactly_verified')
"""

from __future__ import annotations

from alkahest.alkahest import (
    _pde2_canonical_form,
    _pde2_laplace_cascade,
    _pde2_laplace_invariants,
    _pde2_solve,
)
from alkahest.experimental._pde import PDE, _condition, _err, _residual, _unknowns_of

__all__ = [
    "CanonicalForm",
    "SecondOrderSolution",
    "canonical_form",
    "canonical_forms",
    "laplace_cascade",
    "laplace_invariants",
    "solve_second_order",
]


class CanonicalForm:
    """A canonical form: ``type``, ``transformation`` ``[φ, ψ]`` (the new
    variables ``new_variables`` are ``ξ = φ(x, y)``, ``η = ψ(x, y)``),
    ``inverse`` (``[x(ξ, η), y(ξ, η)]`` when found), ``unknown`` ``w(ξ, η)``
    with ``u(x, y) = w(φ, ψ)``, ``equation`` (read ``= 0``, principal
    coefficient one), ``coefficients`` ``[A', B', C', D', E', F', G']``,
    ``in_new_variables`` (whether every coefficient is a function of ``ξ, η``),
    ``multiplier`` and ``conditions``; ``region`` for a mixed-type equation."""

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def type(self) -> str:
        return self._d["type"]

    def to_dict(self) -> dict:
        return dict(self._d)

    def __repr__(self) -> str:
        phi, psi = self._d["transformation"]
        xi, eta = self._d["new_variables"]
        return (
            f"CanonicalForm({self._d['type']}, {xi} = {phi}, {eta} = {psi}: "
            f"{self._d['equation']} = 0)"
        )


class SecondOrderSolution:
    """A verified solution: ``solution`` (the expression for ``u``),
    ``kind`` (``"general"`` or ``"particular"``), ``method``,
    ``arbitrary_functions``, ``parameters`` (free parameters of a family),
    ``conditions`` (side conditions), ``steps``, ``canonical`` (a
    :class:`CanonicalForm` or ``None``) and ``verification`` (the dict of
    :func:`~alkahest.experimental.verify_pde_solution`, always
    ``exactly_verified``)."""

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        d = dict(d)
        if d.get("canonical") is not None:
            d["canonical"] = CanonicalForm(d["canonical"])
        self._d = d

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def value(self):
        """Alias of ``solution``, as on ``DerivedResult``."""
        return self._d["solution"]

    def to_dict(self) -> dict:
        d = dict(self._d)
        if d.get("canonical") is not None:
            d["canonical"] = d["canonical"].to_dict()
        return d

    def __repr__(self) -> str:
        return (
            f"SecondOrderSolution({self._d['kind']}, method={self._d['method']!r}, "
            f"u = {self._d['solution']})"
        )


def _eq_and_unknown(equation, unknown):
    if isinstance(equation, PDE):
        if len(equation.equations) != 1 or len(equation.unknowns) != 1:
            raise _err("E-PDE-008", "one equation in one unknown is needed")
        return equation.equations[0], equation.unknowns[0], list(equation._conditions)
    eq = _residual(equation)
    if unknown is None:
        us = _unknowns_of([eq])
        if len(us) != 1:
            raise _err(
                "E-PDE-001",
                f"pass the unknown explicitly: the equation differentiates {len(us)} functions",
            )
        unknown = us[0]
    return eq, unknown, []


def canonical_form(equation, unknown=None) -> CanonicalForm:
    """The canonical form of a linear second-order equation whose type is the
    same on the whole plane.  ``E-PDE-102`` when the type varies (use
    :func:`canonical_forms`), ``E-PDE-101`` when a characteristic ODE has no
    closed-form first integral."""
    eq, u, _ = _eq_and_unknown(equation, unknown)
    return CanonicalForm(_pde2_canonical_form(eq, u, False)[0])


def canonical_forms(equation, unknown=None) -> list:
    """One :class:`CanonicalForm` per region of constant type (Tricomi's
    ``y u_xx + u_yy = 0``: hyperbolic for ``y < 0``, elliptic for ``y > 0``)."""
    eq, u, _ = _eq_and_unknown(equation, unknown)
    return [CanonicalForm(d) for d in _pde2_canonical_form(eq, u, True)]


def solve_second_order(
    equation,
    unknown=None,
    *,
    ics=None,
    max_cascade_steps: int = 8,
    prefix: str = "F",
    allow_particular: bool = True,
) -> SecondOrderSolution:
    """Solve a linear second-order equation in two variables in closed form.

    ``equation`` is a :class:`~alkahest.experimental.PDE` (whose initial
    conditions, if any, are used) or an equation (``Expr`` read as ``= 0`` or
    an ``(lhs, rhs)`` pair) with ``unknown``.  ``ics`` adds Cauchy data as
    ``(lhs, rhs)`` pairs: ``(u(x, 0), f)`` and ``(D(u, t).subs(t, 0), g)``.

    Returns a :class:`SecondOrderSolution` (``kind="general"`` in arbitrary
    functions ``F1, F2, …``, or ``"particular"`` for a Cauchy problem or an
    exponential family).  Raises :class:`~alkahest.experimental.PdeError`:
    ``E-PDE-100`` outside the class, ``E-PDE-103`` when the Laplace cascade
    spends ``max_cascade_steps``, ``E-PDE-104`` when no method applies,
    ``E-PDE-107`` for unsupported conditions.
    """
    eq, u, conds = _eq_and_unknown(equation, unknown)
    for c in ics or ():
        conds.append(_condition("ic", c, None))
    d = _pde2_solve(eq, u, conds or None, max_cascade_steps, prefix, allow_particular)
    return SecondOrderSolution(d)


def laplace_cascade(equation, unknown=None, *, max_steps: int = 8) -> SecondOrderSolution:
    """Solve ``u_xy + a u_x + b u_y + c u = s`` (variable coefficients) by
    Laplace transformations until an invariant vanishes.  ``E-PDE-103`` when
    none does within ``max_steps``; ``E-PDE-108`` for another principal
    part."""
    eq, u, _ = _eq_and_unknown(equation, unknown)
    return SecondOrderSolution(_pde2_laplace_cascade(eq, u, max_steps))


def laplace_invariants(equation, unknown=None) -> tuple:
    """The Laplace invariants ``(h, k) = (a_x + ab − c, b_y + ab − c)`` of
    ``u_xy + a u_x + b u_y + c u = s``."""
    eq, u, _ = _eq_and_unknown(equation, unknown)
    return _pde2_laplace_invariants(eq, u)
