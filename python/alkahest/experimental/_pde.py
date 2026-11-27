"""PDE foundation layer (milestone M0): unknowns, derivatives, jets, problem
objects, classification and verification.

Everything here is experimental and re-exported from
:mod:`alkahest.experimental`.  The mathematics lives in the Rust crate
(``alkahest_cas::pde``); the problem objects below are thin, immutable Python
values that validate through it.

Representation
--------------
An unknown is an applied undefined function ``u(x, t)`` and its partial
derivatives are the kernel's ``Derivative(u(x, t), x, t)`` nodes, canonical per
argument slot (mixed partials commute by construction)::

    >>> import alkahest as ak, alkahest.experimental as akx
    >>> p = ak.ExprPool(); x, t = p.symbol("x"), p.symbol("t")
    >>> u = akx.Function("u")
    >>> akx.D(u(x, t), x, t)
    Derivative(u(x, t), x, t)
    >>> heat = akx.PDE(akx.D(u(x, t), t) - akx.D(u(x, t), x, 2), u(x, t))
    >>> heat.classify().named[0]["name"]
    'heat'
"""

from __future__ import annotations

import alkahest as _ak
from alkahest import alkahest as _native
from alkahest.alkahest import (
    JetSpace,
    PdeError,
    _pde_check_problem,
    _pde_classify,
    _pde_domain_condition,
    _pde_verify,
    functions_to_jets,
)

__all__ = [
    "PDE",
    "ArbitraryFunctions",
    "Ball",
    "Box",
    "D",
    "Disk",
    "Function",
    "HalfLine",
    "Indexed",
    "IndexedBase",
    "Integral",
    "Interval",
    "JetSpace",
    "KroneckerDelta",
    "Line",
    "PdeError",
    "PdeInfo",
    "Product",
    "RootOfFamily",
    "Sum",
    "classify_pde",
    "coefficient",
    "expand_finite_sums",
    "expand_sum",
    "functions_to_jets",
    "laplacian_indexed",
    "root_family_enclosure",
    "root_family_member",
    "simplify_sums",
    "truncate",
    "verify_pde_solution",
]


def _err(code: str, message: str) -> PdeError:
    e = PdeError(f"[{code}] {message}")
    e.code = code
    e.remediation = None
    e.span = None
    return e


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
    if pool is None:
        raise TypeError(f"cannot tell which ExprPool {v!r} belongs to: pass at least one Expr")
    if isinstance(v, bool):
        raise TypeError("a bool is not an expression")
    if isinstance(v, int):
        return pool.integer(v)
    if isinstance(v, float):
        return pool.float(v)
    from fractions import Fraction

    if isinstance(v, Fraction):
        return pool.rational(v.numerator, v.denominator)
    raise TypeError(f"expected an Expr or a number, got {type(v).__name__}")


# ---------------------------------------------------------------------------
# Unknowns and derivatives
# ---------------------------------------------------------------------------


class Function:
    """An undefined function name: ``Function("u")(x, t)`` is the applied
    unknown ``u(x, t)``.

    Any name the kernel attaches no meaning to (not ``sin``, not a reserved
    head such as ``Derivative`` or ``Sum``) is an undefined function: a smooth
    function about which nothing else is known.  It differentiates through the
    ``Derivative`` head and prints and parses back (declare it to
    :func:`alkahest.parse` with ``symbols={"u": Function("u")}``).
    """

    __slots__ = ("name",)
    _alkahest_undefined_function = True

    def __init__(self, name: str):
        if not isinstance(name, str) or not _ak._is_undefined_function_name(name):
            raise _err(
                "E-PDE-001",
                f"{name!r} is not available as an undefined function name "
                "(it is built in, reserved, or not an identifier)",
            )
        self.name = name

    def __call__(self, *args):
        if not args:
            raise _err("E-PDE-001", f"{self.name}() needs at least one argument")
        pool = _pool_of(*args)
        if pool is None:
            pool = _ak.default_pool() if hasattr(_ak, "default_pool") else None
        if pool is None:
            raise TypeError(f"{self.name}(...) needs at least one Expr argument")
        return pool.func(self.name, [_as_expr(a, pool) for a in args])

    def __repr__(self) -> str:
        return f"Function({self.name!r})"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Function) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("alkahest.Function", self.name))


def D(expr, *spec):
    """Partial derivative, SymPy style: ``D(f, x, t)`` is ``∂²f/∂x∂t``,
    ``D(f, x, 3)`` is ``∂³f/∂x³``, ``D(f, (x, 2), t)`` mixes both.

    Returns the derivative as an ``Expr`` (not a ``DerivedResult``).  With no
    variables, ``expr`` itself.
    """
    cur = expr.value if hasattr(expr, "value") and not isinstance(expr, _ak.Expr) else expr
    last = None
    for s in spec:
        if isinstance(s, tuple):
            if len(s) != 2 or not isinstance(s[1], int) or s[1] < 0:
                raise ValueError("a (var, n) pair needs a non-negative integer n")
            var, n = s
            for _ in range(n):
                cur = _ak.diff(cur, var).value
            last = var
        elif isinstance(s, int) and not isinstance(s, bool):
            if last is None:
                raise ValueError("an order must follow a variable: D(f, x, 2)")
            if s < 1:
                raise ValueError("an order after a variable must be at least 1")
            for _ in range(s - 1):
                cur = _ak.diff(cur, last).value
        else:
            cur = _ak.diff(cur, s).value
            last = s
    return cur


# ---------------------------------------------------------------------------
# Arbitrary functions (D6)
# ---------------------------------------------------------------------------


def _names_in(expr, out: set) -> None:
    stack = [expr]
    seen = set()
    while stack:
        e = stack.pop()
        if not isinstance(e, _ak.Expr) or e in seen:
            continue
        seen.add(e)
        node = e.node()
        if node[0] == "symbol":
            out.add(str(e))
        elif node[0] == "func":
            out.add(node[1])
            stack.extend(node[2])
        else:
            for part in node[1:]:
                if isinstance(part, list):
                    stack.extend(part)
                elif isinstance(part, _ak.Expr):
                    stack.append(part)


class ArbitraryFunctions:
    """Fresh arbitrary-function names ``F1, F2, …`` for general solutions —
    the PDE counterpart of ``dsolve``'s ``C1, C2, …``.

    ``ArbitraryFunctions(avoid=[exprs])`` skips every name already used by a
    function or symbol of ``exprs``; ``.fresh(x - c*t)`` returns
    ``F1(x - c*t)``.  The results are ordinary undefined functions, so
    :func:`verify_pde_solution` keeps them symbolic.
    """

    def __init__(self, prefix: str = "F", avoid=()):
        self.prefix = prefix
        self._next = 1
        self._taken: set = set()
        for e in avoid:
            _names_in(e, self._taken)

    def fresh_name(self) -> str:
        while True:
            name = f"{self.prefix}{self._next}"
            self._next += 1
            if name not in self._taken and _ak._is_undefined_function_name(name):
                self._taken.add(name)
                return name

    def fresh_function(self) -> Function:
        return Function(self.fresh_name())

    def fresh(self, *args):
        return self.fresh_function()(*args)


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------


class _Domain:
    __slots__ = ()

    def spec(self) -> tuple:  # pragma: no cover - abstract
        raise NotImplementedError

    def __mul__(self, other: _Domain) -> Product:
        if not isinstance(other, _Domain):
            return NotImplemented
        return Product(self, other)

    def condition(self):
        """Membership as a predicate expression in the coordinates."""
        return _pde_domain_condition(self.spec())

    def vars(self) -> list:
        raise NotImplementedError  # pragma: no cover

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.spec()[1:]}"


class Interval(_Domain):
    """``lo < var < hi``."""

    __slots__ = ("hi", "lo", "var")

    def __init__(self, var, lo, hi):
        pool = var._pool
        self.var, self.lo, self.hi = var, _as_expr(lo, pool), _as_expr(hi, pool)

    def spec(self):
        return ("interval", self.var, self.lo, self.hi)

    def vars(self):
        return [self.var]


class HalfLine(_Domain):
    """``var > end`` (default) or ``var < end`` with ``upper=True``."""

    __slots__ = ("end", "upper", "var")

    def __init__(self, var, end=0, upper: bool = False):
        self.var, self.end, self.upper = var, _as_expr(end, var._pool), bool(upper)

    def spec(self):
        return ("half_line", self.var, self.end, self.upper)

    def vars(self):
        return [self.var]


class Line(_Domain):
    """All of ℝ in ``var``."""

    __slots__ = ("var",)

    def __init__(self, var):
        self.var = var

    def spec(self):
        return ("line", self.var)

    def vars(self):
        return [self.var]


class Box(_Domain):
    """``Box([x, y], [(0, a), (0, b)])`` — a product of intervals."""

    __slots__ = ("bounds", "variables")

    def __init__(self, variables, bounds):
        self.variables = list(variables)
        pool = _pool_of(self.variables)
        self.bounds = [(_as_expr(lo, pool), _as_expr(hi, pool)) for lo, hi in bounds]

    def spec(self):
        return ("box", self.variables, self.bounds)

    def vars(self):
        return list(self.variables)


class Disk(_Domain):
    """``Disk((x, y), radius, center=(0, 0))`` — the open disk.

    :func:`separation_solve` rewrites an isotropic equation on it in polar
    coordinates; a boundary condition ``u(x, y, …) = g`` (the unknown at its
    own arguments) is Dirichlet data on the circle.
    """

    __slots__ = ("center", "radius", "variables")

    def __init__(self, variables, radius, center=(0, 0)):
        self.variables = list(variables)
        pool = _pool_of(self.variables)
        self.radius = _as_expr(radius, pool)
        self.center = [_as_expr(c, pool) for c in center]

    def spec(self):
        return ("disk", self.variables, self.center, self.radius)

    def vars(self):
        return list(self.variables)


class Ball(_Domain):
    """``Ball([x, y, z], radius, center=None)`` — the open ball (origin by
    default).  Like :class:`Disk`: :func:`separation_solve` works in polar
    (two dimensions) or spherical (three) coordinates."""

    __slots__ = ("center", "radius", "variables")

    def __init__(self, variables, radius, center=None):
        self.variables = list(variables)
        pool = _pool_of(self.variables)
        self.radius = _as_expr(radius, pool)
        if center is None:
            center = [0] * len(self.variables)
        self.center = [_as_expr(c, pool) for c in center]

    def spec(self):
        return ("ball", self.variables, self.center, self.radius)

    def vars(self):
        return list(self.variables)


class Product(_Domain):
    """A product of domains in disjoint coordinates: ``Interval(x, 0, L) *
    HalfLine(t, 0)``."""

    __slots__ = ("parts",)

    def __init__(self, *parts):
        flat = []
        for p in parts:
            flat.extend(p.parts if isinstance(p, Product) else [p])
        self.parts = tuple(flat)

    def spec(self):
        return ("product", [p.spec() for p in self.parts])

    def vars(self):
        return [v for p in self.parts for v in p.vars()]

    def __repr__(self) -> str:
        return " * ".join(repr(p) for p in self.parts)


# ---------------------------------------------------------------------------
# Equations
# ---------------------------------------------------------------------------


def _residual(eq):
    """``eq`` as an expression read as ``= 0``: an ``Expr``, an ``(lhs, rhs)``
    pair, or an equality predicate ``lhs = rhs``."""
    if isinstance(eq, (tuple, list)) and len(eq) == 2:
        lhs, rhs = eq
        pool = _pool_of(lhs, rhs)
        return _as_expr(lhs, pool) - _as_expr(rhs, pool)
    if hasattr(eq, "value") and not isinstance(eq, _ak.Expr):
        eq = eq.value
    if not isinstance(eq, _ak.Expr):
        raise TypeError(f"expected an equation (Expr or (lhs, rhs)), got {type(eq).__name__}")
    node = eq.node()
    if node[0] == "predicate" and str(node[1]).lower() in ("eq", "="):
        a, b = node[2]
        return a - b
    return eq


def _condition(kind: str, lhs, rhs):
    if rhs is None:
        if isinstance(lhs, (tuple, list)) and len(lhs) == 2:
            lhs, rhs = lhs
        else:
            node = lhs.node() if isinstance(lhs, _ak.Expr) else None
            if node is not None and node[0] == "predicate" and str(node[1]).lower() in ("eq", "="):
                lhs, rhs = node[2]
            else:
                rhs = 0
    pool = _pool_of(lhs, rhs)
    return (kind, _as_expr(lhs, pool), _as_expr(rhs, pool))


def _unknowns_of(eqs) -> list:
    """The applied unknowns differentiated in ``eqs`` (for when none are
    given)."""
    found = []
    stack = list(eqs)
    seen = set()
    while stack:
        e = stack.pop()
        if not isinstance(e, _ak.Expr) or e in seen:
            continue
        seen.add(e)
        node = e.node()
        if node[0] == "func":
            if node[1] == "Derivative" and node[2]:
                inner = node[2][0]
                if inner not in found:
                    found.append(inner)
            stack.extend(node[2])
        else:
            for part in node[1:]:
                if isinstance(part, list):
                    stack.extend(part)
                elif isinstance(part, _ak.Expr):
                    stack.append(part)
    return found


class PdeInfo:
    """Result of :func:`classify_pde`.

    Attributes: ``order``, ``linearity`` (``"linear"``, ``"semilinear"``,
    ``"quasilinear"``, ``"fully_nonlinear"``), ``homogeneous`` and
    ``constant_coefficients`` (``None`` unless linear), ``type``
    (``"elliptic"``, ``"parabolic"``, ``"hyperbolic"``, ``"ultrahyperbolic"``,
    ``"mixed"``, ``"solution_dependent"``, ``"undetermined"``; ``None`` unless
    second order), ``type_conditions``, ``principal_symbol``,
    ``discriminant``, ``regions`` (dicts with ``type``, ``condition``,
    ``var``, ``lower``, ``upper``), ``named`` (dicts with ``name``,
    ``parameters``, ``conditions``, ``time``), ``equation_in_jets``,
    ``unknown``, ``independent``.
    """

    def __init__(self, d: dict):
        self._d = dict(d)
        self.unknown = d["unknown"]
        self.independent = d["independent"]
        self.equation_in_jets = d["equation_in_jets"]
        self.order = d["order"]
        self.linearity = d["linearity"]
        self.homogeneous = d["homogeneous"]
        self.constant_coefficients = d["constant_coefficients"]
        self.type = d["type"]
        self.type_conditions = d["type_conditions"]
        self.principal_symbol = d["principal_symbol"]
        self.discriminant = d["discriminant"]
        self.regions = d["regions"]
        self.named = d["named"]

    @property
    def names(self) -> list:
        """The matched equation names, e.g. ``["heat"]``."""
        return [m["name"] for m in self._d["named"]]

    def to_dict(self) -> dict:
        """The classification as a plain dict (expressions as ``Expr``)."""
        return dict(self._d)

    def __repr__(self) -> str:
        bits = [f"order={self.order}", f"linearity={self.linearity!r}"]
        if self.type is not None:
            bits.append(f"type={self.type!r}")
        if self.named:
            bits.append(f"named={self.names!r}")
        return f"PdeInfo({', '.join(bits)})"


def classify_pde(equation, unknown=None) -> PdeInfo:
    """Classify a scalar PDE: order, linearity class, homogeneity, constant
    coefficients, elliptic / parabolic / hyperbolic type (with regions when
    the coefficients vary), and named-equation matching (heat, wave, Laplace,
    Poisson, Helmholtz, transport, Burgers, KdV, Fisher–KPP, sine-Gordon,
    Klein–Gordon, telegraph).

    ``equation`` is an ``Expr`` read as ``= 0`` or an ``(lhs, rhs)`` pair;
    ``unknown`` the applied unknown ``u(x, t)`` (inferred when the equation
    differentiates exactly one function).  Raises :class:`PdeError`.
    """
    if isinstance(equation, PDE):
        return equation.classify()
    eq = _residual(equation)
    if unknown is None:
        us = _unknowns_of([eq])
        if len(us) != 1:
            raise _err(
                "E-PDE-001",
                f"pass the unknown explicitly: the equation differentiates {len(us)} functions",
            )
        unknown = us[0]
    return PdeInfo(_pde_classify(eq, unknown))


class PDE:
    """A PDE problem: ``PDE(eq | [eqs], u | [u, v])``, with
    ``.on(domain)``, ``.bc(lhs, rhs=0)`` and ``.ic(lhs, rhs=0)``.

    Equations are ``Expr`` (read as ``= 0``) or ``(lhs, rhs)`` pairs; unknowns
    are applied undefined functions ``u(x, t)`` (inferred from the
    equations when omitted).  The object is immutable — ``on``, ``bc`` and
    ``ic`` return a new problem — and validated by the Rust layer on every
    construction, so a malformed problem raises :class:`PdeError`
    (``E-PDE-001 … E-PDE-011``) at the point it is written.
    """

    __slots__ = ("_conditions", "_domain", "equations", "independent", "order", "unknowns")

    def __init__(self, equations, unknowns=None, *, _domain=None, _conditions=()):
        eqs = equations if isinstance(equations, list) else [equations]
        self.equations = [_residual(e) for e in eqs]
        if unknowns is None:
            us = _unknowns_of(self.equations)
        elif isinstance(unknowns, (list, tuple)):
            us = list(unknowns)
        else:
            us = [unknowns]
        self.unknowns = us
        self._domain = _domain
        self._conditions = tuple(_conditions)
        info = _pde_check_problem(
            self.equations,
            self.unknowns,
            None if _domain is None else _domain.spec(),
            list(self._conditions),
        )
        self.order = info["order"]
        self.independent = info["independent"]

    def _with(self, domain=None, conditions=None) -> PDE:
        return PDE(
            self.equations,
            self.unknowns,
            _domain=self._domain if domain is None else domain,
            _conditions=self._conditions if conditions is None else conditions,
        )

    def on(self, domain: _Domain) -> PDE:
        """The same problem on ``domain`` (``Interval``, ``HalfLine``, ``Box``,
        ``Disk``, ``Ball``, a ``Product`` of them)."""
        if not isinstance(domain, _Domain):
            raise TypeError("on() takes an Interval, HalfLine, Line, Box, Disk, Ball or Product")
        return self._with(domain=domain)

    def bc(self, lhs, rhs=None) -> PDE:
        """Add the boundary condition ``lhs = rhs`` (``rhs`` defaults to 0;
        ``lhs`` may also be an ``(lhs, rhs)`` pair)."""
        return self._with(conditions=(*self._conditions, _condition("bc", lhs, rhs)))

    def ic(self, lhs, rhs=None) -> PDE:
        """Add the initial condition ``lhs = rhs``."""
        return self._with(conditions=(*self._conditions, _condition("ic", lhs, rhs)))

    @property
    def domain(self):
        return self._domain

    @property
    def bcs(self) -> list:
        return [(lhs, rhs) for k, lhs, rhs in self._conditions if k == "bc"]

    @property
    def ics(self) -> list:
        return [(lhs, rhs) for k, lhs, rhs in self._conditions if k == "ic"]

    def jet_space(self, max_order: int | None = None) -> JetSpace:
        """The problem's jet space (order :attr:`order` by default)."""
        return JetSpace(
            self.independent, self.unknowns, self.order if max_order is None else max_order
        )

    def classify(self) -> PdeInfo:
        """:func:`classify_pde` of a scalar problem (``E-PDE-012`` for a
        system)."""
        if len(self.equations) != 1 or len(self.unknowns) != 1:
            raise _err(
                "E-PDE-012",
                "classification is defined for one equation in one unknown; this system has "
                f"{len(self.equations)} equations in {len(self.unknowns)} unknowns",
            )
        return PdeInfo(_pde_classify(self.equations[0], self.unknowns[0]))

    def verify(self, solution) -> dict:
        """:func:`verify_pde_solution` against this problem's equations and
        conditions."""
        return verify_pde_solution(self, solution)

    def __repr__(self) -> str:
        eqs = ", ".join(f"{e} = 0" for e in self.equations)
        us = ", ".join(str(u) for u in self.unknowns)
        s = f"PDE([{eqs}], [{us}]"
        if self._domain is not None:
            s += f", on={self._domain!r}"
        if self._conditions:
            s += f", conditions={len(self._conditions)}"
        return s + ")"


def verify_pde_solution(pde, solution, bcs=None, ics=None, unknown=None) -> dict:
    """Check a candidate solution by substitution into the PDE, the boundary
    conditions and the initial conditions.

    ``pde`` is a :class:`PDE` or an equation (``Expr`` / ``(lhs, rhs)``);
    ``solution`` an expression (or a list, one per unknown); ``bcs`` / ``ics``
    extra conditions as ``(lhs, rhs)`` pairs or expressions read as ``= 0``.

    Returns a dict in the vocabulary of ``integrate``'s ``verification``:
    ``status`` is ``"exactly_verified"`` (every residual proved zero —
    symbolic, never sampled), ``"refuted"`` (some residual proved nonzero),
    ``"numerically_checked"`` (no refutation, some residual only
    ``probably_zero`` — evidence, **not** proof) or ``"unverified"``;
    ``verified`` is ``True`` only for ``exactly_verified``.  ``pde`` / ``bc``
    / ``ic`` list each residual with its ``zero_test`` status
    (``proved_zero`` / ``proved_nonzero`` / ``probably_zero`` / ``unknown``).
    Arbitrary functions (``F1(x - c*t)``) stay symbolic and are listed under
    ``arbitrary_functions``.
    """
    if not isinstance(pde, PDE):
        eq = _residual(pde)
        pde = PDE([eq], unknown)
    sols = list(solution) if isinstance(solution, (list, tuple)) else [solution]
    conds = list(pde._conditions)
    for c in bcs or ():
        conds.append(_condition("bc", c, None))
    for c in ics or ():
        conds.append(_condition("ic", c, None))
    pool = _pool_of(pde.unknowns)
    sols = [_as_expr(s, pool) for s in sols]
    return _pde_verify(pde.equations, pde.unknowns, sols, conds)


# ---------------------------------------------------------------------------
# Reserved heads: Sum / Indexed / KroneckerDelta / RootOfFamily (D4 / D5)
# ---------------------------------------------------------------------------


def Sum(body, k, lo, hi):
    """``Σ_{k=lo}^{hi} body`` — a binder: ``k`` is bound in ``body`` and
    :func:`alkahest.subs` respects that (a key mentioning ``k`` does not reach
    inside; a value that would be captured renames ``k``).

    ``lo`` / ``hi`` may be symbolic (a dimension ``d``) or ``oo``.  ``diff``
    works term by term (for bounds free of the variable), numeric evaluation
    expands once the bounds evaluate to integers, and :func:`truncate`,
    :func:`coefficient`, :func:`expand_sum` and :func:`simplify_sums` do the
    rest.  Raises :class:`PdeError` ``E-PDE-020`` for a malformed sum.
    """
    pool = _pool_of(k, body, lo, hi)
    return _native._pde_sum(_as_expr(body, pool), k, _as_expr(lo, pool), _as_expr(hi, pool))


def Integral(f, t, lo=None, hi=None):
    """An unevaluated integral: ``Integral(f, t, lo, hi)`` is
    ``∫_{lo}^{hi} f dt`` (a binder: ``t`` is bound in ``f``), and
    ``Integral(f, x)`` the indefinite integral, defined as
    ``Integral(f(t), t, 0, x)``.

    ``diff`` applies the Leibniz rule / fundamental theorem of calculus
    (``d/dx Integral(f(x), x) = f(x)``), so answers containing integrals are
    verified exactly; numeric evaluation uses Gauss–Legendre quadrature.
    ``pdsolve`` returns integrals of this head where a characteristic
    quadrature has no closed form (an arbitrary ``f(x)``, a non-elementary
    integrand).  Raises :class:`PdeError` ``E-PDE-054`` for a malformed
    integral.
    """
    pool = _pool_of(t, f, lo, hi)
    if lo is None and hi is None:
        return _native._pde_integral(_as_expr(f, pool), t)
    if lo is None or hi is None:
        raise TypeError("Integral takes (f, x) or (f, t, lo, hi)")
    return _native._pde_integral(_as_expr(f, pool), t, _as_expr(lo, pool), _as_expr(hi, pool))


def Indexed(base, index):
    """The indexed symbol ``base[index]`` — a variable ``diff`` can
    differentiate with respect to, with ``∂x_i/∂x_j = KroneckerDelta(i, j)``.
    Bind ``Indexed(x, 3)`` in an ``eval_expr`` environment to give ``x_3`` a
    value."""
    return _native._pde_indexed(base, _as_expr(index, base._pool))


class IndexedBase:
    """``x = IndexedBase("x", pool)``; ``x[i]`` is :func:`Indexed` ``(x, i)``.

    ``IndexedBase(sym)`` also accepts an existing symbol."""

    __slots__ = ("symbol",)

    def __init__(self, name, pool=None):
        if isinstance(name, _ak.Expr):
            self.symbol = name
        else:
            if pool is None:
                pool = _ak.default_pool() if hasattr(_ak, "default_pool") else None
            if pool is None:
                raise TypeError("IndexedBase(name) needs a pool")
            self.symbol = pool.symbol(name)

    def __getitem__(self, index):
        return Indexed(self.symbol, index)

    def __repr__(self) -> str:
        return f"IndexedBase({str(self.symbol)!r})"


def KroneckerDelta(i, j):
    """``δ_ij``; folds to ``1`` / ``0`` for equal nodes or distinct integer
    literals."""
    pool = _pool_of(i, j)
    return _native._pde_kronecker(_as_expr(i, pool), _as_expr(j, pool))


def RootOfFamily(f, lam, n, branch: int = 0):
    """The ``n``-th root of ``f(lam, n) = 0``: ``branch=0`` counts positive
    roots in increasing order (``n = 1, 2, …``), ``branch=1`` negative roots in
    decreasing order.  ``lam`` is bound in ``f``; ``n`` may be a symbol (the
    whole family) or a positive integer.  :func:`root_family_enclosure` gives
    a certified enclosure; ``diff`` with respect to a parameter of ``f`` is
    exact (implicit differentiation).  Raises ``E-PDE-030`` when malformed.
    """
    pool = _pool_of(lam, f)
    return _native._pde_root_of_family(_as_expr(f, pool), lam, _as_expr(n, pool), int(branch))


def root_family_member(family, k: int):
    """The member with index ``k`` of a ``RootOfFamily`` with symbolic
    index."""
    return _native._pde_root_family_member(family, int(k))


def root_family_enclosure(root, bits: int = 64):
    """Certified enclosure ``(lo, hi)`` (decimal strings, rounded outward) of
    a ``RootOfFamily`` with an integer index and numeric parameters, at most
    ``2**-bits`` wide relative to the root.  Raises ``E-PDE-031`` when the
    root cannot be certified (a pole of ``f``, a multiple root) — it never
    guesses."""
    return _native._pde_root_family_enclosure(root, int(bits))


def expand_sum(s):
    """An explicit sum of the terms of the ``Sum`` ``s`` (integer bounds;
    ``E-PDE-021`` otherwise)."""
    return _native._pde_expand_sum(s)


def expand_finite_sums(expr):
    """``expr`` with every ``Sum`` whose bounds are integer literals
    expanded; the others are left alone."""
    return _native._pde_expand_finite_sums(expr)


def truncate(expr, n: int):
    """Partial sums: every ``Sum(body, k, lo, hi)`` in ``expr`` becomes the
    explicit sum of its terms with ``k <= n`` (for a series from ``k = 1``,
    its first ``n`` terms)."""
    return _native._pde_truncate(expr, int(n))


def coefficient(s, n):
    """The term with index ``n`` of the ``Sum`` ``s`` (``0`` when ``n`` is
    provably outside the range; ``E-PDE-021`` when undecidable)."""
    return _native._pde_coefficient(s, _as_expr(n, s._pool))


def simplify_sums(expr, ranges=None):
    """Fold the sums in ``expr``: empty and single-term ranges, ``k``-free
    summands (``Σ_{i=1}^{d} c = c·d``), linearity, and
    ``Σ_k f(k)·KroneckerDelta(k, j) = f(j)`` when ``j`` is known to lie in the
    range — an integer inside literal bounds, or by an entry
    ``(j, lo, hi)`` of ``ranges``."""
    return _native._pde_simplify_sums(expr, ranges)


def laplacian_indexed(u, base, lo, hi):
    """``Σ_j ∂²u/∂x_j²`` over the indexed coordinates ``x_lo … x_hi`` (``hi``
    may be a symbol ``d``): ``laplacian_indexed(Sum(x[i]**2/2, i, 1, d), x, 1,
    d)`` is ``d`` for every ``d``."""
    if isinstance(base, IndexedBase):
        base = base.symbol
    pool = base._pool
    return _native._pde_laplacian_indexed(u, base, _as_expr(lo, pool), _as_expr(hi, pool))
