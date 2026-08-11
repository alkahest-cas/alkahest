"""Ansatz families — structured conjecture generation for search loops.

P2 item 1 — see ``docs/mdbook/src/ansatz.md``.

Stage 1 of an autoresearch loop is *generate*: propose a structured family of
candidate objects ("every polynomial of degree ≤ 3 in x and y", "a Padé
approximant of type (2, 2)", "a quadratic Lyapunov candidate"), then either
sweep it numerically or solve for the coefficients that make some residual
vanish. Agents hand-roll this constantly, and the hand-rolled version is
usually wrong in one of three specific ways:

* it loses the distinction between an *unknown coefficient* and an
  *independent variable*, so the follow-up substitution solves for the wrong
  thing;
* it assumes the first *m* sample points give *m* independent equations, which
  is false exactly when the family is degenerate — the interesting case;
* it never substitutes the fitted answer back, so a fit that only satisfies the
  sampled constraints is reported as if it satisfied the identity.

This module is that plumbing, done once. :class:`Ansatz` keeps the unknowns and
the variables apart; :func:`fit` over-samples and reads the rank off the
reduced row echelon form rather than assuming independence; and ``fit(..., certify="residual")``
— the default — substitutes the solution back and only reports
``"exactly_verified"`` when the residual provably normalises to zero.

Honesty invariants
------------------
**Solving may be heuristic; checking is exact.** The linear system is built by
*collocation* (probing the residual at sample points), which is a sufficient
condition for identical vanishing only for polynomial residuals of bounded
degree. So the fit is never trusted on its own: it is substituted back and
normalised. A solution that normalises to zero is
``verification["status"] == "exactly_verified"``; one that does not is returned
**intact** but labelled ``"numerically_checked"``, with the surviving residual
stated in ``verification["residual"]``. There is no status meaning "solved".

**Inconsistent is a result, not a malfunction.** When no member of the family
can satisfy the constraints, :func:`fit` raises
:class:`~alkahest.AnsatzError` ``E-ANSATZ-003``. For a loop that is a *closed
branch* — a positive finding worth recording — in the same spirit as a
non-elementarity verdict from :func:`alkahest.integrate`.

**Underdetermined is also a result.** When the rank is below the number of
unknowns, the members that work form a positive-dimensional family. That is
often the answer the loop wanted, so :attr:`AnsatzSolution.free` returns the
free parameters rather than picking an arbitrary member of the space.

Quick start
-----------
>>> import alkahest as ak
>>> from alkahest.ansatz import polynomial, fit
>>> pool = ak.ExprPool()
>>> x = pool.symbol("x")
>>> A = polynomial(pool, [x], degree=2)
>>> len(A.unknowns)
3
>>> target = x**2 - pool.integer(3) * x + pool.integer(2)
>>> sol = fit(A, A.expr - target)
>>> sol.status
'exactly_verified'
>>> sol.rank, sol.free
(3, ())

Everything here is pure Python composed from primitives that are already fast
in Rust (``Matrix.rref``, ``simplify``, ``subs``), so it works in a build
without the ``groebner`` feature — see :func:`fit` for the one path that needs
it, and refuses rather than degrading when it is absent.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from .exceptions import AnsatzError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Iterator, Mapping, Sequence

__all__ = [
    "DEFAULT_MAX_MEMBERS",
    "DEFAULT_MAX_TERMS",
    "DEFAULT_SEED",
    "Ansatz",
    "AnsatzSolution",
    "certify_nonneg",
    "enumerate_family",
    "exponential_polynomial",
    "fit",
    "linear_combination",
    "polynomial",
    "quadratic_form",
    "rational",
]

#: Default ceiling on the number of unknown coefficients a family may carry.
#: ``C(n + d, d)`` is a combinatorial explosion, so every constructor is
#: bounded; exceeding the bound raises ``E-ANSATZ-002`` *before* anything is
#: materialised. Raise it deliberately, per call, rather than globally.
DEFAULT_MAX_TERMS = 256

#: Default ceiling on the number of members :func:`enumerate_family` will
#: produce. Enumeration is lazy *and* bounded — the bound is checked from the
#: family size before the first member is built.
DEFAULT_MAX_MEMBERS = 100_000

#: Seed used for sample-point selection when no :class:`~alkahest.Budget`
#: carrying one is active. Fixing it means two runs on two machines fit at the
#: same points; entering ``ak.context(budget=ak.Budget(seed=...))`` overrides
#: it (see :func:`alkahest.budget_seed`).
DEFAULT_SEED = 0x5EED_A115


def _ak() -> Any:
    """Resolve the parent package at call time.

    ``alkahest/__init__.py`` imports this module during its own
    initialisation, so a module-level ``import alkahest`` would bind a
    half-built namespace. Mirrors ``alkahest._batch._ak``.
    """
    import alkahest

    return alkahest


def _value(obj: Any) -> Any:
    """Coerce a :class:`~alkahest.DerivedResult` to its ``.value``.

    Several kernel entry points (``cancel``, ``together``, ...) return a bare
    ``Expr`` in some builds and a ``DerivedResult`` in others; this normalises
    both.
    """
    value = getattr(obj, "value", None)
    if value is not None and hasattr(obj, "verification"):
        return value
    return obj


# ---------------------------------------------------------------------------
# Expression inspection
# ---------------------------------------------------------------------------


class _NotExact(Exception):
    """The expression has no exact rational value (a symbol, or transcendental)."""


class _Undefined(Exception):
    """The expression is mathematically undefined here (a vanishing denominator)."""


_MAX_EXPONENT = 4096


def _rational_value(expr: Any) -> Fraction:
    """Evaluate a closed arithmetic expression exactly.

    Parameters
    ----------
    expr : Expr
        Expression built from integers, rationals, floats, ``+``, ``*``, and
        integer powers.

    Returns
    -------
    Fraction

    Raises
    ------
    _NotExact
        A free symbol, a function application, or an irrational power.
    _Undefined
        A negative power of zero — i.e. the point is a pole. This is the check
        that lets :func:`fit` *skip* a sample point instead of quietly building
        a row out of ``2 * 0^-1``.
    """
    node = expr.node()
    tag = node[0]
    if tag == "integer":
        return Fraction(int(node[1]))
    if tag == "rational":
        return Fraction(int(node[1]), int(node[2]))
    if tag == "float":
        return Fraction(str(node[1]))
    if tag == "add":
        total = Fraction(0)
        for arg in node[1]:
            total += _rational_value(arg)
        return total
    if tag == "mul":
        product = Fraction(1)
        for arg in node[1]:
            product *= _rational_value(arg)
        return product
    if tag == "pow":
        exponent = _rational_value(node[2])
        if exponent.denominator != 1 or abs(exponent) > _MAX_EXPONENT:
            raise _NotExact(f"non-integer or oversized exponent in {expr}")
        base = _rational_value(node[1])
        if base == 0 and exponent < 0:
            raise _Undefined(f"zero denominator in {expr}")
        return base ** int(exponent)
    raise _NotExact(f"no exact rational value for {expr}")


def _free_symbols(expr: Any, out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``{name: Expr}`` for every free symbol reachable from *expr*."""
    found: dict[str, Any] = {} if out is None else out
    stack = [expr]
    while stack:
        current = stack.pop()
        node = current.node()
        if node[0] == "symbol":
            found.setdefault(str(node[1]), current)
            continue
        for item in node[1:]:
            if isinstance(item, (list, tuple)):
                stack.extend(i for i in item if hasattr(i, "node"))
            elif hasattr(item, "node"):
                stack.append(item)
    return found


def _degree_in(expr: Any, names: frozenset[str]) -> int | None:
    """Total degree of *expr* in the symbols *names*, or ``None`` if > 1.

    A purely syntactic, conservative test. ``0`` means the unknowns do not
    occur, ``1`` means the expression is affine in them, and ``None`` means
    "nonlinear, or not decidable by inspection" — which :func:`fit` treats as
    nonlinear, since the cost of guessing wrong is a silently wrong fit.
    """
    node = expr.node()
    tag = node[0]
    if tag == "symbol":
        return 1 if str(node[1]) in names else 0
    if tag in {"integer", "rational", "float"}:
        return 0
    if tag == "add":
        best = 0
        for arg in node[1]:
            degree = _degree_in(arg, names)
            if degree is None:
                return None
            best = max(best, degree)
        return best
    if tag == "mul":
        total = 0
        for arg in node[1]:
            degree = _degree_in(arg, names)
            if degree is None:
                return None
            total += degree
        return None if total > 1 else total
    if tag == "pow":
        if _degree_in(node[2], names) != 0:
            return None  # an unknown in the exponent
        base = _degree_in(node[1], names)
        if base == 0:
            return 0
        if base is None:
            return None
        try:
            exponent = _rational_value(node[2])
        except (_NotExact, _Undefined):
            return None
        return base if exponent == 1 else None
    if tag == "func":
        for arg in node[2]:
            if _degree_in(arg, names) != 0:
                return None  # an unknown inside sin/exp/...
        return 0
    return None if _free_symbols(expr).keys() & names else 0


def _is_affine(expr: Any, names: frozenset[str]) -> bool:
    """True when *expr* is affine (degree ≤ 1) in the symbols *names*."""
    return _degree_in(expr, names) is not None


def _unknown_denominators(expr: Any, names: frozenset[str]) -> list[tuple[Any, int]]:
    """Sub-expressions ``base**-k`` whose *base* involves the unknowns.

    These are what makes a Padé-style residual nonlinear, and multiplying them
    out is what makes it linear again — see :func:`_clear_denominators`.
    """
    found: dict[str, tuple[Any, int]] = {}
    stack = [expr]
    while stack:
        current = stack.pop()
        node = current.node()
        if node[0] == "pow":
            try:
                exponent = _rational_value(node[2])
            except (_NotExact, _Undefined):
                exponent = Fraction(0)
            base = node[1]
            if exponent < 0 and exponent.denominator == 1 and _free_symbols(base).keys() & names:
                key = str(base)
                power = -int(exponent)
                previous = found.get(key)
                if previous is None or previous[1] < power:
                    found[key] = (base, power)
        for item in node[1:]:
            if isinstance(item, (list, tuple)):
                stack.extend(i for i in item if hasattr(i, "node"))
            elif hasattr(item, "node"):
                stack.append(item)
    return [found[key] for key in sorted(found)]


def _clear_denominators(expr: Any, names: frozenset[str]) -> tuple[Any, str] | None:
    """Multiply out unknown-bearing denominators, returning ``(expr, note)``.

    ``p/q − f`` is not linear in the coefficients of *p* and *q*; ``p − f·q``
    is. This is the transform that keeps the Padé / rational-function case on
    the default (linear, ``groebner``-free) path, and it is applied
    automatically by :func:`fit` — see design note D2 in the module docs.

    Returns ``None`` when there is nothing to clear or the result is still not
    affine.
    """
    ak = _ak()
    denominators = _unknown_denominators(expr, names)
    if not denominators:
        return None
    scaled = expr
    parts: list[str] = []
    for base, power in denominators:
        for _ in range(power):
            scaled = scaled * base
        parts.append(f"({base})^{power}" if power != 1 else f"({base})")
    for transform in (ak.cancel, ak.together, lambda e: ak.simplify(e)):
        try:
            candidate = _value(transform(scaled))
        except Exception:
            continue
        if _is_affine(candidate, names):
            return candidate, "multiplied through by " + " * ".join(parts)
    return None


# ---------------------------------------------------------------------------
# Deterministic sample points
# ---------------------------------------------------------------------------


def _resolve_seed(seed: int | None) -> int:
    """Seed to use: the explicit one, else the active budget's, else the default."""
    if seed is not None:
        return int(seed)
    try:
        from ._budget import budget_seed
    except ImportError:  # pragma: no cover - defensive
        return DEFAULT_SEED
    try:
        active = budget_seed()
    except Exception:
        active = None
    return DEFAULT_SEED if active is None else int(active)


def _lcg(seed: int) -> Iterator[int]:
    """A deterministic 64-bit LCG.

    Deliberately *not* :mod:`random`: this stream must be identical across
    Python versions and platforms so that a fit recorded in a claim graph can
    be reproduced from its seed alone.
    """
    state = (int(seed) ^ 0x9E3779B97F4A7C15) & 0xFFFF_FFFF_FFFF_FFFF
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFF_FFFF_FFFF_FFFF
        yield (state >> 17) & 0xFFFF_FFFF


def _sample_points(n_vars: int, seed: int) -> Iterator[tuple[Fraction, ...]]:
    """Yield distinct deterministic rational points in ``n_vars`` variables."""
    stream = _lcg(seed)
    seen: set[tuple[Fraction, ...]] = set()
    while True:
        point = tuple(Fraction(next(stream) % 25 - 12, 1 + next(stream) % 4) for _ in range(n_vars))
        if point in seen:
            continue
        seen.add(point)
        yield point


def _to_expr(pool: Any, value: Fraction) -> Any:
    """Intern a :class:`~fractions.Fraction` into *pool*."""
    if value.denominator == 1:
        return pool.integer(value.numerator)
    return pool.rational(value.numerator, value.denominator)


# ---------------------------------------------------------------------------
# Ansatz
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ansatz:
    """A parametric family of expressions with named unknown coefficients.

    An ansatz is an object rather than a bare :class:`~alkahest.Expr` because a
    bare expression loses the distinction between an *unknown coefficient* and
    an *independent variable*, and every downstream step needs it. Construct
    one with :func:`polynomial`, :func:`rational`, :func:`linear_combination`,
    :func:`exponential_polynomial`, or :func:`quadratic_form` rather than
    directly.

    Attributes
    ----------
    expr : Expr
        The family member with symbolic coefficients, e.g.
        ``c_0 + c_1*x + c_2*x^2``.
    unknowns : tuple of Expr
        The coefficient symbols, in a deterministic order (graded, then
        lexicographic by exponent).
    basis : tuple of Expr
        Parallel to *unknowns*: the expression each unknown multiplies. For
        :func:`rational` this is the numerator basis followed by the
        denominator basis, since the two blocks multiply into different places
        — read ``metadata["numerator_terms"]`` for the split.
    vars : tuple of Expr
        The independent variables. Disjoint from *unknowns* by construction.
    family : str
        Which constructor produced this, e.g. ``"polynomial"``.
    name : str
        Prefix used for the coefficient symbol names.
    pool : ExprPool
        Pool everything is interned in.
    metadata : dict
        Family-specific detail (degree, rates, the numerator/denominator split).

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.ansatz import polynomial
    >>> pool = ak.ExprPool()
    >>> x, y = pool.symbol("x"), pool.symbol("y")
    >>> A = polynomial(pool, [x, y], degree=1, name="c")
    >>> A.names
    ('c_0_0', 'c_1_0', 'c_0_1')
    >>> A.expr
    (c_0_0 + (x * c_1_0) + (y * c_0_1))
    """

    expr: Any
    unknowns: tuple[Any, ...]
    basis: tuple[Any, ...]
    vars: tuple[Any, ...]
    family: str
    name: str
    pool: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of unknown coefficients."""
        return len(self.unknowns)

    def __repr__(self) -> str:
        variables = ", ".join(str(v) for v in self.vars)
        return f"Ansatz({self.family}, vars=({variables}), unknowns={len(self.unknowns)})"

    @property
    def names(self) -> tuple[str, ...]:
        """Coefficient symbol names, parallel to :attr:`unknowns`.

        Names are **predictable**, never gensym-ed: an agent that cannot
        predict them cannot write the follow-up call.
        """
        return tuple(str(u) for u in self.unknowns)

    @property
    def symbols(self) -> dict[str, Any]:
        """``{name: Expr}`` for the unknowns, for keying an
        :meth:`instantiate` mapping by name."""
        return dict(zip(self.names, self.unknowns))

    def instantiate(self, assignment: Mapping[Any, Any], *, simplify: bool = True) -> Any:
        """Substitute coefficient values, returning a concrete member.

        Parameters
        ----------
        assignment : mapping
            ``{Expr or str: Expr or int or float}``. Unknowns left out stay
            symbolic, which is how an underdetermined fit is returned.
        simplify : bool
            Fold the substituted expression (so a zeroed coefficient really
            disappears). Turn off for a hot enumeration whose consumer
            normalises anyway.

        Returns
        -------
        Expr

        Examples
        --------
        >>> import alkahest as ak
        >>> from alkahest.ansatz import polynomial
        >>> pool = ak.ExprPool()
        >>> x = pool.symbol("x")
        >>> A = polynomial(pool, [x], degree=1)
        >>> A.instantiate({"c_0": 2, "c_1": -1})
        (2 + (x * -1))
        """
        substituted = _substitute(self.pool, self.expr, assignment, self.symbols)
        return _value(_ak().simplify(substituted)) if simplify else substituted

    def residual(self, target: Any) -> Any:
        """The residual whose vanishing means "this member equals *target*".

        For most families this is just ``self.expr - target``. For
        :func:`rational` it is the **denominator-cleared** form
        ``numerator - target * denominator``, which is affine in the unknowns
        where ``p/q - f`` is not. Passing the naive difference to :func:`fit`
        works too — it clears denominators itself — but this states the intent.

        Examples
        --------
        >>> import alkahest as ak
        >>> from alkahest.ansatz import rational
        >>> pool = ak.ExprPool()
        >>> x = pool.symbol("x")
        >>> A = rational(pool, [x], num_degree=1, den_degree=1)
        >>> A.residual(pool.integer(1))
        (a_0 + (x * a_1) + ((1 + (x * b_1)) * -1))
        """
        ak = _ak()
        target = _value(target)
        numerator = self.metadata.get("numerator")
        denominator = self.metadata.get("denominator")
        if numerator is not None and denominator is not None:
            return _value(ak.simplify(numerator - target * denominator))
        return self.expr - target

    def with_prefix(self, name: str) -> Ansatz:
        """A copy of this family with the coefficient prefix *name*.

        Useful when two ansätze are fitted jointly and their coefficient names
        would otherwise collide.
        """
        return _rebuild(self, name)


def _substitute(
    pool: Any, expr: Any, assignment: Mapping[Any, Any], symbols: Mapping[str, Any]
) -> Any:
    """``ak.subs`` with string keys and :class:`~fractions.Fraction` values allowed."""
    ak = _ak()
    mapping: dict[Any, Any] = {}
    for key, value in assignment.items():
        symbol = symbols.get(key) if isinstance(key, str) else key
        if symbol is None:
            raise AnsatzError(
                f"[E-ANSATZ-001] unknown coefficient name {key!r}",
                code="E-ANSATZ-001",
                remediation=f"expected one of {sorted(symbols)}",
            )
        mapping[symbol] = _to_expr(pool, value) if isinstance(value, Fraction) else value
    if not mapping:
        return expr
    return _value(ak.subs(expr, mapping))


# ---------------------------------------------------------------------------
# Family constructors
# ---------------------------------------------------------------------------


def _exponents(n_vars: int, degree: int, *, min_degree: int = 0) -> list[tuple[int, ...]]:
    """Exponent tuples with ``min_degree <= |a| <= degree``, graded then lex."""
    out: list[tuple[int, ...]] = []
    for total in range(min_degree, degree + 1):
        block = [c for c in itertools.product(range(total + 1), repeat=n_vars) if sum(c) == total]
        out.extend(sorted(block, reverse=True))
    return out


def _count_exponents(n_vars: int, degree: int, *, min_degree: int = 0) -> int:
    """``len(_exponents(...))`` without materialising it."""
    upper = math.comb(n_vars + degree, n_vars)
    lower = math.comb(n_vars + min_degree - 1, n_vars) if min_degree > 0 else 0
    return upper - lower


def _check_size(count: int, max_terms: int, what: str) -> None:
    if count > max_terms:
        raise AnsatzError(
            f"[E-ANSATZ-002] {what} needs {count} unknown coefficients, "
            f"which exceeds max_terms={max_terms}",
            code="E-ANSATZ-002",
            remediation=(
                "lower the degree, or pass an explicit max_terms= if you really want a "
                f"{count}-term family — nothing is materialised until this check passes"
            ),
        )


def _check_collision(names: Sequence[str], reserved: Mapping[str, Any], name: str) -> None:
    """Refuse a coefficient prefix whose names collide with existing symbols.

    A family named ``c`` fitted against an expression that already contains a
    symbol ``c_0`` silently solves for the wrong thing. There is deliberately
    no gensym fallback: unpredictable names are unusable from a follow-up call.

    ``reserved`` is what this module can see — the family's own variables plus
    every free symbol of the expressions handed to the constructor, plus
    anything the caller declared. The pool itself exposes no symbol listing, so
    a symbol that exists in the pool but appears in none of those is not
    detected here; ``fit``'s back-substitution check is the backstop.
    """
    duplicates = [n for n in names if n in reserved]
    if duplicates:
        raise AnsatzError(
            f"[E-ANSATZ-001] coefficient name(s) {sorted(set(duplicates))} already "
            f"denote symbols in this problem; the fit would solve for the wrong thing",
            code="E-ANSATZ-001",
            remediation=(
                f"pass a different name= (currently {name!r}) — coefficient names are "
                "never gensym-ed, because an agent that cannot predict them cannot "
                "write the follow-up call"
            ),
        )
    seen: set[str] = set()
    for candidate in names:
        if candidate in seen:
            raise AnsatzError(
                f"[E-ANSATZ-001] coefficient name {candidate!r} generated twice",
                code="E-ANSATZ-001",
                remediation=("internal naming clash; report it with the call that caused it"),
            )
        seen.add(candidate)


def _reserved_from(exprs: Iterable[Any], extra: Iterable[Any]) -> dict[str, Any]:
    reserved: dict[str, Any] = {}
    for expr in exprs:
        _free_symbols(_value(expr), reserved)
    for item in extra:
        if isinstance(item, str):
            reserved.setdefault(item, None)
        else:
            _free_symbols(_value(item), reserved)
    return reserved


def _coefficient_name(name: str, exponents: Sequence[int], *, univariate: bool) -> str:
    if univariate:
        return f"{name}_{exponents[0]}"
    return name + "".join(f"_{e}" for e in exponents)


def _make_terms(
    pool: Any,
    variables: Sequence[Any],
    exponents: Sequence[Sequence[int]],
    name: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[str, ...]]:
    """Build ``(unknowns, basis, names)`` for a monomial family."""
    univariate = len(variables) == 1
    names = tuple(_coefficient_name(name, e, univariate=univariate) for e in exponents)
    unknowns = tuple(pool.symbol(n) for n in names)
    basis: list[Any] = []
    for combo in exponents:
        term = pool.integer(1)
        for var, power in zip(variables, combo):
            for _ in range(power):
                term = term * var
        basis.append(term)
    return unknowns, tuple(basis), names


def _sum_terms(pool: Any, unknowns: Sequence[Any], basis: Sequence[Any]) -> Any:
    total = pool.integer(0)
    for coefficient, term in zip(unknowns, basis):
        total = total + coefficient * term
    return _value(_ak().simplify(total))


def polynomial(
    pool: Any,
    vars: Sequence[Any],
    degree: int,
    *,
    name: str = "c",
    min_degree: int = 0,
    max_terms: int = DEFAULT_MAX_TERMS,
    reserved: Sequence[Any] = (),
) -> Ansatz:
    """Every polynomial of total degree ≤ *degree* in *vars*, with unknown coefficients.

    Parameters
    ----------
    pool : ExprPool
        Pool the coefficient symbols are interned in.
    vars : sequence of Expr
        Independent variables, in the order the exponent tuples index them.
    degree : int
        Maximum **total** degree.
    name : str
        Coefficient prefix. Names are ``f"{name}_{i}"`` for one variable and
        ``f"{name}_{i}_{j}..."`` for several — always predictable, never
        gensym-ed. Collides → ``E-ANSATZ-001``.
    min_degree : int
        Drop monomials of total degree below this (``min_degree=degree`` gives
        the homogeneous forms).
    max_terms : int
        Refuse rather than materialise a family bigger than this
        (``E-ANSATZ-002``). ``C(n + d, d)`` grows fast.
    reserved : sequence of Expr or str
        Extra symbols (or names) the coefficients must not collide with — pass
        the target function here when it carries symbols of its own.

    Returns
    -------
    Ansatz

    Raises
    ------
    AnsatzError
        ``E-ANSATZ-001`` on a name collision, ``E-ANSATZ-002`` if the family
        exceeds *max_terms*.

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.ansatz import polynomial
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> A = polynomial(pool, [x], degree=3)
    >>> A.names
    ('c_0', 'c_1', 'c_2', 'c_3')
    >>> try:
    ...     polynomial(pool, [x], degree=99, max_terms=10)
    ... except ak.AnsatzError as exc:
    ...     exc.code
    'E-ANSATZ-002'
    """
    variables = tuple(_value(v) for v in vars)
    if not variables:
        raise AnsatzError(
            "[E-ANSATZ-001] polynomial() needs at least one variable",
            code="E-ANSATZ-001",
            remediation="pass vars=[x] (or use linear_combination for a basis with no variables)",
        )
    if degree < 0 or min_degree < 0 or min_degree > degree:
        raise ValueError("polynomial() requires 0 <= min_degree <= degree")
    count = _count_exponents(len(variables), degree, min_degree=min_degree)
    _check_size(
        count,
        max_terms,
        f"polynomial(degree={degree}, {len(variables)} variable(s))",
    )
    exponents = _exponents(len(variables), degree, min_degree=min_degree)
    unknowns, basis, names = _make_terms(pool, variables, exponents, name)
    _check_collision(names, _reserved_from(variables, reserved), name)
    return Ansatz(
        expr=_sum_terms(pool, unknowns, basis),
        unknowns=unknowns,
        basis=basis,
        vars=variables,
        family="polynomial",
        name=name,
        pool=pool,
        metadata={
            "degree": degree,
            "min_degree": min_degree,
            "exponents": exponents,
            "max_terms": max_terms,
        },
    )


def rational(
    pool: Any,
    vars: Sequence[Any],
    num_degree: int,
    den_degree: int,
    *,
    name: str = "a",
    den_name: str = "b",
    monic_denominator: bool = True,
    max_terms: int = DEFAULT_MAX_TERMS,
    reserved: Sequence[Any] = (),
) -> Ansatz:
    """A rational-function family ``p / q`` — the Padé case, kept linear.

    ``p/q ≈ f`` is *not* linear in the coefficients of ``p`` and ``q``, but
    ``p − f·q = 0`` is linear in them jointly. This constructor records the
    numerator/denominator split so :meth:`Ansatz.residual` (and :func:`fit`,
    which clears denominators itself) can apply that transform, keeping the
    default solve path exact linear algebra with no ``groebner`` dependency.

    Parameters
    ----------
    pool, vars, max_terms, reserved
        As :func:`polynomial`.
    num_degree, den_degree : int
        Total degrees of the numerator and denominator.
    name, den_name : str
        Coefficient prefixes for the numerator and denominator.
    monic_denominator : bool
        When true (default), the denominator's constant term is fixed to ``1``.
        Without a normalisation ``p/q`` and ``(λp)/(λq)`` are the same function,
        so **every** such fit would report a spurious extra free parameter.

    Returns
    -------
    Ansatz
        ``.expr`` is the quotient; ``.metadata["numerator"]`` and
        ``["denominator"]`` are the two polynomials, and
        ``["numerator_terms"]`` is how many of ``.unknowns`` belong to the
        numerator.

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.ansatz import rational
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> A = rational(pool, [x], num_degree=1, den_degree=1)
    >>> A.names
    ('a_0', 'a_1', 'b_1')
    >>> A.expr
    ((a_0 + (x * a_1)) * (1 + (x * b_1))^-1)
    """
    variables = tuple(_value(v) for v in vars)
    if not variables:
        raise AnsatzError(
            "[E-ANSATZ-001] rational() needs at least one variable",
            code="E-ANSATZ-001",
            remediation="pass vars=[x]",
        )
    if num_degree < 0 or den_degree < 0:
        raise ValueError("rational() requires non-negative degrees")
    den_min = 1 if monic_denominator else 0
    count = _count_exponents(len(variables), num_degree) + _count_exponents(
        len(variables), den_degree, min_degree=den_min
    )
    _check_size(
        count,
        max_terms,
        f"rational(num_degree={num_degree}, den_degree={den_degree})",
    )
    num_exponents = _exponents(len(variables), num_degree)
    den_exponents = _exponents(len(variables), den_degree, min_degree=den_min)
    num_unknowns, num_basis, num_names = _make_terms(pool, variables, num_exponents, name)
    den_unknowns, den_basis, den_names = _make_terms(pool, variables, den_exponents, den_name)
    _check_collision(
        num_names + den_names,
        _reserved_from(variables, reserved),
        f"{name}/{den_name}",
    )
    numerator = _sum_terms(pool, num_unknowns, num_basis)
    denominator = _sum_terms(pool, den_unknowns, den_basis)
    if monic_denominator:
        denominator = _value(_ak().simplify(pool.integer(1) + denominator))
    return Ansatz(
        expr=numerator / denominator,
        unknowns=num_unknowns + den_unknowns,
        basis=num_basis + den_basis,
        vars=variables,
        family="rational",
        name=name,
        pool=pool,
        metadata={
            "num_degree": num_degree,
            "den_degree": den_degree,
            "numerator": numerator,
            "denominator": denominator,
            "numerator_terms": len(num_unknowns),
            "den_name": den_name,
            "monic_denominator": monic_denominator,
            "max_terms": max_terms,
        },
    )


def linear_combination(
    pool: Any,
    basis: Sequence[Any],
    *,
    vars: Sequence[Any] | None = None,
    name: str = "c",
    max_terms: int = DEFAULT_MAX_TERMS,
    reserved: Sequence[Any] = (),
) -> Ansatz:
    """``Σ cᵢ · basisᵢ`` — the general escape hatch.

    Whenever the family is "an unknown linear combination of *these* things"
    — a Lyapunov basis, a set of candidate integrals, ζ(3) and π³ and log³2 —
    this is the constructor. Everything else in this module is a convenience
    wrapper that computes a basis for you.

    Parameters
    ----------
    pool : ExprPool
    basis : sequence of Expr
        The basis functions. Not checked for linear independence — a dependent
        basis simply shows up as free parameters in the fit, which is the
        honest answer.
    vars : sequence of Expr, optional
        Independent variables. Inferred from the basis's free symbols (sorted
        by name) when omitted.
    name, max_terms, reserved
        As :func:`polynomial`.

    Returns
    -------
    Ansatz

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.ansatz import linear_combination
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> A = linear_combination(pool, [ak.sin(x), ak.cos(x)])
    >>> A.expr
    ((sin(x) * c_0) + (cos(x) * c_1))
    """
    terms = tuple(_value(b) for b in basis)
    if not terms:
        raise AnsatzError(
            "[E-ANSATZ-001] linear_combination() needs a non-empty basis",
            code="E-ANSATZ-001",
            remediation="pass basis=[...] with at least one expression",
        )
    _check_size(len(terms), max_terms, f"linear_combination({len(terms)} basis functions)")
    names = tuple(f"{name}_{i}" for i in range(len(terms)))
    unknowns = tuple(pool.symbol(n) for n in names)
    symbols = _reserved_from(terms, reserved)
    _check_collision(names, symbols, name)
    if vars is None:
        variables = tuple(symbols[key] for key in sorted(symbols) if symbols[key] is not None)
    else:
        variables = tuple(_value(v) for v in vars)
    return Ansatz(
        expr=_sum_terms(pool, unknowns, terms),
        unknowns=unknowns,
        basis=terms,
        vars=variables,
        family="linear_combination",
        name=name,
        pool=pool,
        metadata={"max_terms": max_terms},
    )


def exponential_polynomial(
    pool: Any,
    var: Any,
    rates: Sequence[Any],
    *,
    degree: int | Sequence[int] = 0,
    name: str = "c",
    max_terms: int = DEFAULT_MAX_TERMS,
    reserved: Sequence[Any] = (),
) -> Ansatz:
    """``Σᵢ pᵢ(x)·e^{λᵢ x}`` with polynomial ``pᵢ`` of the given degree.

    The standard ansatz for a linear ODE or recurrence whose characteristic
    roots ``λᵢ`` are already known — the polynomial factor's degree is the
    multiplicity minus one. Because the ``λᵢ`` are *given*, the family stays
    linear in the unknowns and fits through the default path.

    Parameters
    ----------
    pool : ExprPool
    var : Expr
        The independent variable.
    rates : sequence of Expr or int
        The exponents ``λᵢ``. Duplicates are allowed but pointless — raise the
        corresponding *degree* instead.
    degree : int or sequence of int
        Degree of each polynomial factor; a scalar applies to every rate.
    name, max_terms, reserved
        As :func:`polynomial`. Coefficient names are ``f"{name}_{i}_{k}"`` for
        rate *i* and power *k*.

    Returns
    -------
    Ansatz

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.ansatz import exponential_polynomial
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> A = exponential_polynomial(pool, x, [1, -1])
    >>> A.names
    ('c_0_0', 'c_1_0')
    >>> A.expr
    ((exp(x) * c_0_0) + (exp((x * -1)) * c_1_0))
    """
    ak = _ak()
    variable = _value(var)
    rate_exprs = [r if hasattr(r, "node") else pool.integer(int(r)) for r in rates]
    if not rate_exprs:
        raise AnsatzError(
            "[E-ANSATZ-001] exponential_polynomial() needs at least one rate",
            code="E-ANSATZ-001",
            remediation="pass rates=[lambda_0, ...]",
        )
    if isinstance(degree, int):
        degrees = [degree] * len(rate_exprs)
    else:
        degrees = [int(d) for d in degree]
        if len(degrees) != len(rate_exprs):
            raise ValueError("exponential_polynomial(): len(degree) must match len(rates)")
    if any(d < 0 for d in degrees):
        raise ValueError("exponential_polynomial() requires non-negative degrees")
    count = sum(d + 1 for d in degrees)
    _check_size(count, max_terms, f"exponential_polynomial({len(rate_exprs)} rates)")

    names: list[str] = []
    basis: list[Any] = []
    for index, (rate, deg) in enumerate(zip(rate_exprs, degrees)):
        envelope = ak.exp(rate * variable)
        for power in range(deg + 1):
            term = envelope
            for _ in range(power):
                term = term * variable
            names.append(f"{name}_{index}_{power}")
            basis.append(_value(ak.simplify(term)))
    unknowns = tuple(pool.symbol(n) for n in names)
    _check_collision(tuple(names), _reserved_from([variable, *rate_exprs], reserved), name)
    return Ansatz(
        expr=_sum_terms(pool, unknowns, basis),
        unknowns=unknowns,
        basis=tuple(basis),
        vars=(variable,),
        family="exponential_polynomial",
        name=name,
        pool=pool,
        metadata={"rates": tuple(rate_exprs), "degrees": tuple(degrees), "max_terms": max_terms},
    )


def quadratic_form(
    pool: Any,
    vars: Sequence[Any],
    *,
    name: str = "q",
    max_terms: int = DEFAULT_MAX_TERMS,
    reserved: Sequence[Any] = (),
) -> Ansatz:
    """``Σ_{i ≤ j} q_ij · xᵢ xⱼ`` — the Lyapunov-candidate family.

    Only the upper triangle carries an unknown, because ``q_ij xᵢxⱼ`` and
    ``q_ji xⱼxᵢ`` are the same term and carrying both would make every fit
    report spurious free parameters.

    This module's job **ends** when it has produced the candidate. Deciding
    whether a fitted form is non-negative is :func:`alkahest.prove_nonneg` /
    :func:`alkahest.sos_decompose`'s job, and :func:`certify_nonneg` is the
    one-line hand-off — positivity is not reimplemented here.

    Parameters
    ----------
    pool, vars, name, max_terms, reserved
        As :func:`polynomial`. Coefficient names are ``f"{name}_{i}_{j}"`` with
        ``i <= j`` indexing *vars*.

    Returns
    -------
    Ansatz

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.ansatz import quadratic_form
    >>> pool = ak.ExprPool()
    >>> x, y = pool.symbol("x"), pool.symbol("y")
    >>> A = quadratic_form(pool, [x, y])
    >>> A.names
    ('q_0_0', 'q_0_1', 'q_1_1')
    """
    variables = tuple(_value(v) for v in vars)
    if not variables:
        raise AnsatzError(
            "[E-ANSATZ-001] quadratic_form() needs at least one variable",
            code="E-ANSATZ-001",
            remediation="pass vars=[x, y]",
        )
    pairs = [(i, j) for i in range(len(variables)) for j in range(i, len(variables))]
    _check_size(len(pairs), max_terms, f"quadratic_form({len(variables)} variables)")
    names = tuple(f"{name}_{i}_{j}" for i, j in pairs)
    unknowns = tuple(pool.symbol(n) for n in names)
    basis = tuple(variables[i] * variables[j] for i, j in pairs)
    _check_collision(names, _reserved_from(variables, reserved), name)
    return Ansatz(
        expr=_sum_terms(pool, unknowns, basis),
        unknowns=unknowns,
        basis=basis,
        vars=variables,
        family="quadratic_form",
        name=name,
        pool=pool,
        metadata={"pairs": tuple(pairs), "max_terms": max_terms},
    )


def _rebuild(ansatz: Ansatz, name: str) -> Ansatz:
    """Re-run *ansatz*'s constructor with a different coefficient prefix."""
    meta = ansatz.metadata
    if ansatz.family == "polynomial":
        return polynomial(
            ansatz.pool,
            ansatz.vars,
            meta["degree"],
            name=name,
            min_degree=meta["min_degree"],
            max_terms=meta["max_terms"],
        )
    if ansatz.family == "rational":
        return rational(
            ansatz.pool,
            ansatz.vars,
            meta["num_degree"],
            meta["den_degree"],
            name=name,
            den_name=meta["den_name"] + "_" + name,
            monic_denominator=meta["monic_denominator"],
            max_terms=meta["max_terms"],
        )
    if ansatz.family == "exponential_polynomial":
        return exponential_polynomial(
            ansatz.pool,
            ansatz.vars[0],
            meta["rates"],
            degree=meta["degrees"],
            name=name,
            max_terms=meta["max_terms"],
        )
    if ansatz.family == "quadratic_form":
        return quadratic_form(ansatz.pool, ansatz.vars, name=name, max_terms=meta["max_terms"])
    return linear_combination(
        ansatz.pool,
        ansatz.basis,
        vars=ansatz.vars,
        name=name,
        max_terms=meta.get("max_terms", DEFAULT_MAX_TERMS),
    )


# ---------------------------------------------------------------------------
# Enumeration (stage 2 material)
# ---------------------------------------------------------------------------


def enumerate_family(
    ansatz: Ansatz,
    coeffs: Iterable[Any] = (-1, 0, 1),
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> Iterator[Any]:
    """Lazily enumerate concrete members over a finite coefficient set.

    Enumeration and fitting stay separate on purpose: this feeds *stage 2* of a
    search loop (generate candidates, then hammer them with
    :func:`alkahest.compile_expr` or :func:`alkahest.batch_map`), while
    :func:`fit` is *stage 3* (turn data into a symbolic claim). Fusing them
    produces an API that does neither well.

    The bound is checked **before** the first member is built, from
    ``len(coeffs) ** len(ansatz)`` — enumeration is lazy *and* bounded.

    Parameters
    ----------
    ansatz : Ansatz
    coeffs : iterable
        Values each coefficient ranges over. Materialised once.
    max_members : int
        Refuse (``E-ANSATZ-002``) rather than start an enumeration longer than
        this.

    Yields
    ------
    Expr
        One member per assignment, coefficients varying last-unknown-fastest.

    Raises
    ------
    AnsatzError
        ``E-ANSATZ-002`` when the family size exceeds *max_members*.

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.ansatz import enumerate_family, polynomial
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> A = polynomial(pool, [x], degree=1)
    >>> [str(m) for m in enumerate_family(A, [0, 1])]
    ['0', 'x', '1', '(x + 1)']
    >>> try:
    ...     next(enumerate_family(A, range(10 ** 6), max_members=10))
    ... except ak.AnsatzError as exc:
    ...     exc.code
    'E-ANSATZ-002'
    """
    values = list(coeffs)
    total = len(values) ** len(ansatz.unknowns)
    _check_size(total, max_members, f"enumerate_family over {len(values)} value(s)")
    for assignment in itertools.product(values, repeat=len(ansatz.unknowns)):
        yield ansatz.instantiate(dict(zip(ansatz.unknowns, assignment)))


# ---------------------------------------------------------------------------
# AnsatzSolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnsatzSolution:
    """The outcome of fitting an :class:`Ansatz`.

    Shaped so that :meth:`alkahest.research.ResearchSession.record` accepts it
    unchanged: it exposes ``.value``, ``.steps``, ``.verification``, and
    ``.certificate`` exactly as a :class:`~alkahest.DerivedResult` does, and
    ``verification["status"]`` is one of the existing
    :data:`alkahest.research.STATUS_BADGES` keys — no new vocabulary.

    Attributes
    ----------
    expr : Expr
        The fitted member. When the system is underdetermined this still
        contains the free unknowns, symbolically.
    assignment : dict
        ``{unknown: Expr}`` for the determined coefficients. Values may mention
        the free unknowns.
    free : tuple of Expr
        Unknowns the constraints do not pin down. Non-empty means the members
        that work form a positive-dimensional family — often the interesting
        answer, so no arbitrary member is picked for you.
    rank : int
        Rank of the collocation system; ``rank == len(ansatz)`` iff the fit is
        unique.
    status : str
        Mirrors ``verification["status"]``: ``"exactly_verified"``,
        ``"numerically_checked"``, or ``"unverified"``.
    verification : dict
        ``{"status", "evidence", "method", "externally_verified", ...}``.
        ``"residual"`` carries the surviving normal form when the exact check
        did not close, and ``"max_abs_residual"`` the worst numeric sample.
    steps : tuple of dict
        Derivation log in the ``STEP_FIELDS`` schema.
    residual : Expr
        The residual as fitted (after any denominator clearing).
    check : dict
        A re-verification recipe in the shape
        :meth:`alkahest.research.ClaimGraph.verify` understands
        (``{"kind": "zero", "expr": ...}`` — the back-substituted residual,
        which must normalise to zero). Pass it through as
        ``session.record(solution, check=solution.check)`` so a graph loaded
        from disk months later can re-derive the check rather than trust it.
    points : tuple
        The rows' provenance, as nested rational strings: the sample points
        actually used for ``certify="residual"``, or the multi-indices
        ``α`` for ``certify="exact"``. Enough to reproduce the system.
    ansatz : Ansatz
        The family this came from.
    certificate : None
        Always ``None``: this module emits no Lean certificate. Withholding
        beats emitting one nothing checked.
    """

    expr: Any
    assignment: dict[Any, Any]
    free: tuple[Any, ...]
    rank: int
    status: str
    verification: dict[str, Any]
    steps: tuple[dict[str, Any], ...]
    residual: Any
    check: dict[str, Any]
    points: tuple[tuple[str, ...], ...]
    ansatz: Ansatz
    certificate: None = None

    @property
    def value(self) -> Any:
        """Alias of :attr:`expr` — the attribute ``ResearchSession`` reads."""
        return self.expr

    @property
    def determined(self) -> bool:
        """True when the constraints pin down a unique member."""
        return not self.free

    @property
    def badge(self) -> str:
        """Honest one-line rendering of :attr:`status`."""
        from .research import STATUS_BADGES

        return STATUS_BADGES.get(self.status, "unrecognised status")

    def __repr__(self) -> str:
        return (
            f"AnsatzSolution({self.ansatz.family}, rank={self.rank}, "
            f"free={len(self.free)}, status={self.status!r})"
        )


# ---------------------------------------------------------------------------
# The linear system
# ---------------------------------------------------------------------------


def _probe(
    pool: Any,
    residual: Any,
    unknowns: Sequence[Any],
    bindings: Mapping[Any, Any],
) -> list[Any]:
    """One row: ``[C_0, ..., C_{m-1}, -R_0]`` for the residual under *bindings*.

    Because the residual is affine in the unknowns, evaluating it with every
    unknown set to zero gives the constant term ``R_0``, and setting unknown
    *j* to one (the rest zero) gives ``R_0 + C_j``. That is ``subs`` and
    nothing else — no coefficient collection over a symbolic ring is needed.
    """
    ak = _ak()
    zeros = {u: pool.integer(0) for u in unknowns}
    base = _entry(_value(ak.subs(residual, {**zeros, **bindings})))
    row: list[Any] = []
    for unknown in unknowns:
        probe = dict(zeros)
        probe[unknown] = pool.integer(1)
        shifted = _entry(_value(ak.subs(residual, {**probe, **bindings})))
        row.append(_difference(pool, shifted, base))
    row.append(_negate(pool, base))
    return row


def _fold_exp(expr: Any) -> Any:
    """Rewrite ``e^a · e^b → e^{a+b}`` and ``(e^a)^k → e^{ka}`` bottom-up.

    The kernel's ``simplify_log_exp`` merges some of these but not all — it
    leaves ``e^{-9/2} · e^{9/2}`` standing inside an n-ary product — and an
    exponential ansatz's elimination is *made* of such products. Without this
    the entries grow without ever cancelling, so a genuine zero row is never
    recognised. Only ``add``/``mul``/``pow``/``exp`` nodes are rebuilt;
    anything else is returned untouched.
    """
    ak = _ak()
    node = expr.node()
    tag = node[0]
    if tag == "add":
        total = None
        for arg in node[1]:
            folded = _fold_exp(arg)
            total = folded if total is None else total + folded
        return expr if total is None else total
    if tag == "mul":
        exponents: list[Any] = []
        rest: list[Any] = []
        for arg in node[1]:
            folded = _fold_exp(arg)
            inner = folded.node()
            if inner[0] == "func" and str(inner[1]) == "exp" and len(inner[2]) == 1:
                exponents.append(inner[2][0])
            else:
                rest.append(folded)
        if len(exponents) > 1:
            combined = exponents[0]
            for extra in exponents[1:]:
                combined = combined + extra
            rest.append(ak.exp(_value(ak.simplify(combined))))
        elif exponents:
            rest.append(ak.exp(exponents[0]))
        if not rest:
            return expr
        product = rest[0]
        for factor in rest[1:]:
            product = product * factor
        return product
    if tag == "pow":
        base = _fold_exp(node[1])
        inner = base.node()
        try:
            power = _rational_value(node[2])
        except (_NotExact, _Undefined):
            power = None
        if (
            power is not None
            and power.denominator == 1
            and abs(power) <= _MAX_EXPONENT
            and inner[0] == "func"
            and str(inner[1]) == "exp"
            and len(inner[2]) == 1
        ):
            return ak.exp(_value(ak.simplify(inner[2][0] * int(power))))
        return base ** node[2]
    if tag == "func" and str(node[1]) == "exp" and len(node[2]) == 1:
        return ak.exp(_fold_exp(node[2][0]))
    return expr


def _number_like(sample: Any, value: Fraction) -> Any:
    """A numeric literal interned in the same pool as *sample*."""
    result = sample * 0 + value.numerator
    if value.denominator != 1:
        result = result / value.denominator
    return _value(_ak().simplify(result))


def _split_term(term: Any) -> tuple[Fraction, list[Any]]:
    """Split a product into its rational coefficient and its other factors."""
    node = term.node()
    if node[0] == "mul":
        coefficient = Fraction(1)
        factors: list[Any] = []
        for factor in node[1]:
            try:
                coefficient *= _rational_value(factor)
            except (_NotExact, _Undefined):
                factors.append(_collect_rational(factor))
        return coefficient, factors
    try:
        return _rational_value(term), []
    except (_NotExact, _Undefined):
        return Fraction(1), [_collect_rational(term)]


def _collect_rational(expr: Any) -> Any:
    """Collect like terms whose coefficients are rational.

    The kernel's collector is over ℤ: ``simplify`` leaves
    ``t·31/3 + t·(−31/3)`` standing and ``cancel``/``poly_normal`` refuse it
    outright with ``E-POLY-002`` (non-integer coefficient). Elimination
    produces exactly that shape at every step, so without this the entries
    never cancel and a genuine zero row is never recognised. Grouping is
    syntactic — two terms merge only when their non-numeric factors render
    identically — so this can only ever *find* cancellations, never assert one
    that is not there.
    """
    node = expr.node()
    if node[0] == "mul":
        coefficient, factors = _split_term(expr)
        return _rebuild_term(expr, coefficient, factors)
    if node[0] != "add":
        return expr
    groups: dict[tuple[str, ...], tuple[Fraction, list[Any]]] = {}
    order: list[tuple[str, ...]] = []
    for term in node[1]:
        coefficient, factors = _split_term(term)
        key = tuple(sorted(str(f) for f in factors))
        if key in groups:
            groups[key] = (groups[key][0] + coefficient, groups[key][1])
        else:
            groups[key] = (coefficient, factors)
            order.append(key)
    result = None
    for key in order:
        coefficient, factors = groups[key]
        if coefficient == 0:
            continue
        term = _rebuild_term(expr, coefficient, factors)
        result = term if result is None else result + term
    return _number_like(expr, Fraction(0)) if result is None else result


def _rebuild_term(sample: Any, coefficient: Fraction, factors: Sequence[Any]) -> Any:
    if not factors:
        return _number_like(sample, coefficient)
    product = factors[0]
    for factor in factors[1:]:
        product = product * factor
    if coefficient == 1:
        return product
    product = product * coefficient.numerator
    if coefficient.denominator != 1:
        product = product / coefficient.denominator
    return product


def _normalise(expr: Any) -> Any:
    """Best-effort normal form for a transcendental matrix entry.

    ``simplify`` alone leaves ``a + (-1 * (a + b))`` uncollected and
    ``e^a · e^b`` uncombined, and an uncollected entry becomes a phantom pivot
    in :func:`_solve_rows`, so the distributing and exp/log-collecting
    simplifiers each get a pass. Cheap: entries are point evaluations, not
    whole residuals.
    """
    ak = _ak()
    current = expr
    for _ in range(3):
        before = str(current)
        for transform in (_fold_exp, _collect_rational, ak.simplify_expanded, ak.simplify):
            try:
                current = _value(transform(current))
            except Exception:
                continue
        after = str(current)
        if after == "0" or after == before:
            break
    return current


def _entry(expr: Any) -> Fraction | Any:
    """Normalise one matrix entry to a :class:`~fractions.Fraction` when possible.

    ``_Undefined`` deliberately propagates: it means the sample point is a
    pole, and the caller's job is to skip that point rather than to build a row
    out of ``2 * 0^-1``.
    """
    try:
        return _rational_value(expr)
    except _NotExact:
        pass
    normalised = _normalise(expr)
    try:
        return _rational_value(normalised)
    except (_NotExact, _Undefined):
        return normalised


def _as_expr(pool: Any, value: Any) -> Any:
    """Intern a mixed ``Fraction``/``Expr`` entry as an :class:`~alkahest.Expr`."""
    return _to_expr(pool, value) if isinstance(value, Fraction) else value


def _reciprocal(pool: Any, value: Any) -> Any:
    """``1 / value``, pushed inwards so exponentials stay collectable.

    The kernel does not rewrite ``exp(a)^-1`` as ``exp(-a)``, so a plain
    ``x ** -1`` during elimination produces entries that
    :func:`alkahest.simplify_log_exp` can no longer combine — and an entry that
    will not combine is an entry whose zero test fails. Distributing the
    reciprocal over products and turning ``exp(a)`` into ``exp(-a)`` here keeps
    every intermediate in a shape the exp collector can close.
    """
    if isinstance(value, Fraction):
        return _to_expr(pool, Fraction(1) / value)
    node = value.node()
    tag = node[0]
    if tag == "func" and str(node[1]) == "exp" and len(node[2]) == 1:
        return _ak().exp(node[2][0] * pool.integer(-1))
    if tag == "mul":
        product = pool.integer(1)
        for factor in node[1]:
            product = product * _reciprocal(pool, factor)
        return product
    if tag == "pow":
        try:
            exponent = _rational_value(node[2])
        except (_NotExact, _Undefined):
            return value ** pool.integer(-1)
        if exponent.denominator == 1 and abs(exponent) <= _MAX_EXPONENT:
            return _reciprocal(pool, node[1]) ** pool.integer(int(exponent))
        return value ** pool.integer(-1)
    if tag in {"integer", "rational"}:
        return _to_expr(pool, Fraction(1) / _rational_value(value))
    return value ** pool.integer(-1)


def _divide(pool: Any, value: Any, divisor: Any) -> Any:
    """``value / divisor`` for mixed ``Fraction``/``Expr`` entries."""
    if isinstance(value, Fraction) and isinstance(divisor, Fraction):
        return value / divisor
    return _normalise(_as_expr(pool, value) * _reciprocal(pool, divisor))


def _difference(pool: Any, left: Any, right: Any) -> Any:
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left - right
    return _entry(_as_expr(pool, left) - _as_expr(pool, right))


def _negate(pool: Any, value: Any) -> Any:
    if isinstance(value, Fraction):
        return -value
    return _entry(_as_expr(pool, value) * pool.integer(-1))


def _is_zero_entry(value: Any) -> bool:
    """Decide whether a reduced-matrix entry is zero, conservatively.

    Rational entries — the common case, and every entry of a polynomial or
    rational fit — are decided exactly. A transcendental entry is put through
    :func:`alkahest.simplify` first, because ``rref`` leaves cancelling terms
    like ``t + (-1 * t)`` uncollected and treating those as pivots would
    manufacture phantom rank. Anything the simplifier cannot reduce is treated
    as **non-zero**, which costs rank rather than inventing a solution.
    """
    if isinstance(value, Fraction):
        return value == 0
    try:
        return _rational_value(value) == 0
    except (_NotExact, _Undefined):
        pass
    if str(value).strip() == "0":
        return True
    return str(_normalise(value)).strip() == "0"


def _rows_to_matrix(pool: Any, rows: Sequence[Sequence[Any]]) -> Any:
    """Intern a mixed ``Fraction``/``Expr`` row set as an :class:`~alkahest.Matrix`."""
    ak = _ak()
    interned = [[_as_expr(pool, cell) for cell in row] for row in rows]
    return ak.Matrix.from_rows(interned)


def _collocation_rows(
    ansatz: Ansatz,
    residual: Any,
    *,
    seed: int,
    n_rows: int,
    max_points: int,
) -> tuple[list[list[Any]], list[tuple[Fraction, ...]], int]:
    """Build the system by evaluating the residual at deterministic points.

    Points where the residual is undefined (a pole of the target, a vanishing
    denominator) are **skipped and resampled**, with a bounded retry count, so
    a bad draw degrades into a slightly smaller system rather than a row of
    nonsense.
    """
    pool = ansatz.pool
    rows: list[list[Any]] = []
    used: list[tuple[Fraction, ...]] = []
    skipped = 0
    stream = _sample_points(len(ansatz.vars), seed)
    attempts = 0
    while len(rows) < n_rows and attempts < max_points:
        attempts += 1
        point = next(stream)
        bindings = {var: _to_expr(pool, value) for var, value in zip(ansatz.vars, point)}
        try:
            row = _probe(pool, residual, ansatz.unknowns, bindings)
        except _Undefined:
            skipped += 1
            continue
        except Exception:
            skipped += 1
            continue
        rows.append(row)
        used.append(point)
    return rows, used, skipped


def _derivative_rows(
    ansatz: Ansatz,
    residual: Any,
    *,
    seed: int,
    degree_bound: int,
    max_rows: int,
) -> tuple[list[list[Any]], list[tuple[Fraction, ...]], int]:
    """Build the system by Taylor-coefficient extraction (``certify="exact"``).

    ``R ≡ 0`` iff every coefficient ``∂^α R / α!`` vanishes, so each multi-index
    ``|α| ≤ degree_bound`` contributes one equation. Exact for polynomial
    residuals of that degree — no sampling argument required — at the cost of
    one differentiation per multi-index.
    """
    ak = _ak()
    pool = ansatz.pool
    multi = _exponents(len(ansatz.vars), degree_bound)[:max_rows]
    stream = _sample_points(len(ansatz.vars), seed)
    base_point = tuple(Fraction(0) for _ in ansatz.vars)
    rows: list[list[Any]] = []
    used: list[tuple[Fraction, ...]] = []
    skipped = 0
    for attempt in range(8):
        bindings = {var: _to_expr(pool, value) for var, value in zip(ansatz.vars, base_point)}
        rows, used = [], []
        ok = True
        for alpha in multi:
            derivative = residual
            try:
                for var, order in zip(ansatz.vars, alpha):
                    for _ in range(order):
                        derivative = _value(ak.diff(derivative, var))
                rows.append(_probe(pool, derivative, ansatz.unknowns, bindings))
            except _Undefined:
                ok = False
                break
            except Exception:
                ok = False
                break
            used.append(tuple(Fraction(a) for a in alpha))
        if ok and rows:
            return rows, used, skipped
        skipped += 1
        base_point = next(stream)
        if attempt == 7:  # pragma: no cover - exhausted
            break
    return rows, used, skipped


def _default_degree_bound(ansatz: Ansatz, wanted: int) -> int:
    """Smallest multi-index degree giving at least *wanted* Taylor equations."""
    n_vars = len(ansatz.vars)
    for degree in range(64):
        if _count_exponents(n_vars, degree) >= wanted:
            return degree
    return 64  # pragma: no cover - unreachable for any realistic family


def _is_exact_system(rows: Sequence[Sequence[Any]]) -> bool:
    """True when every entry is a :class:`~fractions.Fraction`.

    Polynomial, rational and quadratic-form fits always land here; a family
    whose basis takes transcendental values at rational points (an exponential
    ansatz) does not.
    """
    return all(isinstance(cell, Fraction) for row in rows for cell in row)


def _symbolic_rref(pool: Any, rows: Sequence[Sequence[Any]], width: int) -> list[list[Any]]:
    """Row-reduce a system whose entries are not all rational.

    :meth:`alkahest.Matrix.rref` is used for the exact-rational case, which is
    the overwhelming majority and the one the design specifies. It is *not*
    used here: its internal zero test does not apply the exp/log collector, so
    on a matrix of ``e^{λ x_k}`` entries it fails to cancel a redundant row and
    reports a phantom pivot in the augmented column — i.e. a spurious
    "inconsistent". This does the same elimination with :func:`_normalise` as
    the zero test instead.
    """
    matrix = [[_as_expr(pool, cell) for cell in row] for row in rows]
    pivot_row = 0
    for column in range(width):
        candidate = next(
            (i for i in range(pivot_row, len(matrix)) if not _is_zero_entry(matrix[i][column])),
            None,
        )
        if candidate is None:
            continue
        matrix[pivot_row], matrix[candidate] = matrix[candidate], matrix[pivot_row]
        pivot = matrix[pivot_row][column]
        matrix[pivot_row] = [_divide(pool, cell, pivot) for cell in matrix[pivot_row]]
        for index in range(len(matrix)):
            if index == pivot_row or _is_zero_entry(matrix[index][column]):
                continue
            factor = matrix[index][column]
            matrix[index] = [
                _normalise(matrix[index][j] - factor * matrix[pivot_row][j])
                for j in range(width + 1)
            ]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return matrix


def _numerically_inconsistent(rows: Sequence[Sequence[Any]], width: int) -> bool:
    """Corroborate an inconsistency verdict in floating point.

    Only consulted for a system that is **not** over ℚ, where an apparent
    inconsistency may be nothing but a zero test the simplifier could not
    settle. Claiming ``E-ANSATZ-003`` — "no member of this family works" — on
    that basis would be exactly the kind of confident wrong answer this package
    exists to avoid, so the claim is only made when an independent numeric rank
    comparison agrees. Failure to evaluate returns ``False``: no corroboration,
    no claim.
    """
    ak = _ak()
    try:
        matrix = [
            [
                float(cell) if isinstance(cell, Fraction) else float(ak.eval_expr(cell, {}))
                for cell in row
            ]
            for row in rows
        ]
    except Exception:
        return False
    return _float_rank(matrix, width) < _float_rank(matrix, width + 1)


def _float_rank(matrix: Sequence[Sequence[float]], columns: int) -> int:
    """Rank of the first *columns* columns, by float Gaussian elimination."""
    work = [list(row[:columns]) for row in matrix]
    scale = max((abs(v) for row in work for v in row), default=0.0)
    tolerance = 1e-9 * (scale if scale > 0 else 1.0)
    rank = 0
    for column in range(columns):
        pivot = max(range(rank, len(work)), key=lambda i: abs(work[i][column]), default=None)
        if pivot is None or abs(work[pivot][column]) <= tolerance:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        head = work[rank]
        for index in range(len(work)):
            if index == rank or abs(work[index][column]) <= tolerance:
                continue
            factor = work[index][column] / head[column]
            work[index] = [a - factor * b for a, b in zip(work[index], head)]
        rank += 1
    return rank


def _solve_rows(
    ansatz: Ansatz, rows: Sequence[Sequence[Any]]
) -> tuple[dict[Any, Any], tuple[Any, ...], int]:
    """Reduce the augmented system and read off the solution.

    Rank is **read off the reduced form**, never assumed from the number of
    points: assuming the first *m* samples are independent is the specific bug
    in every hand-rolled version of this, and it is why :func:`fit` over-samples.
    """
    pool = ansatz.pool
    unknowns = ansatz.unknowns
    width = len(unknowns)
    exact = _is_exact_system(rows)
    if exact:
        reduced = _rows_to_matrix(pool, rows).rref().to_list()
    else:
        reduced = _symbolic_rref(pool, rows, width)

    pivots: list[tuple[int, int]] = []
    inconsistent = False
    for index, row in enumerate(reduced):
        for column, cell in enumerate(row):
            if _is_zero_entry(cell):
                continue
            if column == width:
                inconsistent = True
            else:
                pivots.append((index, column))
            break
    if inconsistent and (exact or _numerically_inconsistent(rows, width)):
        raise AnsatzError(
            "[E-ANSATZ-003] the constraints are inconsistent: no member of this "
            f"{ansatz.family} family satisfies them",
            code="E-ANSATZ-003",
            remediation=(
                "this is a result, not a malfunction — record the closed branch. "
                "To reopen it, enlarge the family (raise the degree, add basis "
                "functions) or weaken the constraint"
            ),
        )

    pivot_columns = {column for _, column in pivots}
    free = tuple(u for index, u in enumerate(unknowns) if index not in pivot_columns)
    assignment: dict[Any, Any] = {}
    for row_index, column in pivots:
        row = reduced[row_index]
        value = _as_expr(pool, row[width])
        for other in range(column + 1, width):
            if other in pivot_columns or _is_zero_entry(row[other]):
                continue
            value = value - _as_expr(pool, row[other]) * unknowns[other]
        assignment[unknowns[column]] = _normalise(value)
    return assignment, free, len(pivots)


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------


def _normal_form(expr: Any, variables: Sequence[Any]) -> Any:
    """Best-effort normal form for the back-substituted residual.

    ``poly_normal`` is the strongest normaliser when it applies (a genuine
    polynomial normal form over ℤ), so it gets first refusal; otherwise the
    entry normaliser, which also collects transcendental like terms, decides.
    Only a form that renders as ``0`` is ever treated as a proof.
    """
    ak = _ak()
    simplified = _normalise(expr)
    if str(simplified).strip() == "0":
        return simplified
    names = _free_symbols(simplified)
    try:
        return ak.poly_normal(simplified, [names[key] for key in sorted(names)] or list(variables))
    except Exception:
        return simplified


def _numeric_residual(
    expr: Any, variables: Sequence[Any], free: Sequence[Any], seed: int, samples: int
) -> float | None:
    """Largest ``|residual|`` over fresh sample points, or ``None`` if unevaluable."""
    ak = _ak()
    stream = _sample_points(len(variables) + len(free), seed ^ 0xA5A5_A5A5)
    worst: float | None = None
    for _ in range(samples):
        point = next(stream)
        bindings = {symbol: float(value) for symbol, value in zip((*variables, *free), point)}
        try:
            magnitude = abs(float(ak.eval_expr(expr, bindings)))
        except Exception:
            continue
        worst = magnitude if worst is None else max(worst, magnitude)
    return worst


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def fit(
    ansatz: Ansatz,
    residual: Any,
    *,
    certify: str = "residual",
    seed: int | None = None,
    oversample: int | None = None,
    max_points: int | None = None,
    degree_bound: int | None = None,
    tolerance: float = 1e-8,
    samples: int = 5,
) -> AnsatzSolution:
    """Solve for the coefficients that make *residual* vanish identically.

    The default path is **exact linear algebra, not Gröbner**. Undetermined
    coefficients, series matching, Lyapunov forms and Padé are all linear in
    the unknowns once denominators are cleared, so the system is built by
    probing with :func:`alkahest.subs` and reduced with
    :meth:`alkahest.Matrix.rref` over exact rationals. Nothing here needs the
    ``groebner`` feature; the one path that does (a residual genuinely
    nonlinear in the unknowns) refuses with ``E-ANSATZ-004`` when it is absent
    rather than degrading silently.

    Parameters
    ----------
    ansatz : Ansatz
        The family to fit.
    residual : Expr or DerivedResult
        The expression that must vanish identically in ``ansatz.vars`` —
        typically ``ansatz.expr - target``. If it is not affine in the unknowns
        because of an unknown-bearing denominator (the Padé case), the
        denominator is cleared automatically and the fact is recorded in
        :attr:`AnsatzSolution.steps`.
    certify : {"residual", "exact", "none"}
        ``"residual"`` (default) builds the system by collocation and checks it
        by exact back-substitution. ``"exact"`` builds the system by
        Taylor-coefficient extraction instead, so the *system itself* is exact
        for polynomial residuals, and still back-substitutes. ``"none"`` skips
        the check entirely and returns ``status="unverified"`` — for hot loops
        that will verify downstream, never for anything recorded as a result.
    seed : int, optional
        Sample-point seed. Defaults to :func:`alkahest.budget_seed` (so
        ``with ak.context(budget=ak.Budget(seed=7))`` controls it), and to
        :data:`DEFAULT_SEED` when no budget is active. Two runs with the same
        seed use the same points on any machine.
    oversample : int, optional
        Extra equations beyond the unknown count. Defaults to
        ``max(4, len(ansatz))``. Over-sampling is what lets the rank be *read*
        off the ``rref`` instead of assumed.
    max_points : int, optional
        Cap on sample-point draws, including those skipped for being poles.
    degree_bound : int, optional
        Multi-index degree for ``certify="exact"``. Defaults to the family's
        own total degree plus two.
    tolerance : float
        Numeric residual below which a non-normalising fit is still reported as
        ``"numerically_checked"`` rather than as a stated failure.
    samples : int
        Fresh points used for that numeric check.

    Returns
    -------
    AnsatzSolution

    Raises
    ------
    AnsatzError
        ``E-ANSATZ-003`` when the system is inconsistent — **no member of the
        family satisfies the constraints**, which is a positive result for a
        search loop and should be recorded as a closed branch.
        ``E-ANSATZ-004`` when the residual is nonlinear in the unknowns and
        this build has no ``groebner`` feature to escalate to.

    Examples
    --------
    Round-trip a known polynomial:

    >>> import alkahest as ak
    >>> from alkahest.ansatz import fit, polynomial
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> A = polynomial(pool, [x], degree=2)
    >>> sol = fit(A, A.expr - (x**2 + pool.integer(1)))
    >>> sol.expr
    (1 + x^2)
    >>> sol.status, sol.free
    ('exactly_verified', ())

    An underdetermined family returns its free parameters rather than an
    arbitrary member:

    >>> B = polynomial(pool, [x], degree=3, name="d")
    >>> under = fit(B, ak.diff(B.expr, x).value - pool.integer(0))
    >>> under.rank, [str(f) for f in under.free]
    (3, ['d_0'])
    """
    ak = _ak()
    if certify not in {"residual", "exact", "none"}:
        raise ValueError("fit(certify=...) must be 'residual', 'exact', or 'none'")

    unknowns = ansatz.unknowns
    names = frozenset(ansatz.names)
    target_residual = _value(residual)
    steps: list[dict[str, Any]] = []

    # -- linearise -------------------------------------------------------
    if not _is_affine(target_residual, names):
        cleared = _clear_denominators(target_residual, names)
        if cleared is not None:
            steps.append(
                {
                    "rule": "ansatz_clear_denominator",
                    "before": str(target_residual),
                    "after": str(cleared[0]),
                    "side_conditions": [f"{cleared[1]} != 0"],
                }
            )
            target_residual = cleared[0]
        else:
            return _fit_nonlinear(ansatz, target_residual, seed=seed, steps=steps)

    resolved_seed = _resolve_seed(seed)
    width = len(unknowns)
    extra = max(4, width) if oversample is None else int(oversample)
    n_rows = width + extra
    point_cap = 8 * n_rows + 32 if max_points is None else int(max_points)

    bound = _default_degree_bound(ansatz, n_rows) if degree_bound is None else int(degree_bound)
    if certify == "exact":
        rows, used, skipped = _derivative_rows(
            ansatz, target_residual, seed=resolved_seed, degree_bound=bound, max_rows=point_cap
        )
        method = "taylor_extraction"
    else:
        rows, used, skipped = _collocation_rows(
            ansatz,
            target_residual,
            seed=resolved_seed,
            n_rows=n_rows,
            max_points=point_cap,
        )
        method = "collocation"
        if rows and not _is_exact_system(rows):
            # Collocating a transcendental basis gives entries like e^{λ x_k},
            # and exact linear algebra over those is only as good as the
            # simplifier's zero test. Taylor extraction at the origin usually
            # gives the *same* constraints over ℚ (e^0 = 1), so prefer it when
            # it does — an exact system is worth one extra attempt.
            exact_rows, exact_used, exact_skipped = _derivative_rows(
                ansatz,
                target_residual,
                seed=resolved_seed,
                degree_bound=bound,
                max_rows=point_cap,
            )
            if exact_rows and _is_exact_system(exact_rows):
                rows, used, skipped = exact_rows, exact_used, exact_skipped
                method = "taylor_extraction"

    if not rows:
        raise AnsatzError(
            "[E-ANSATZ-003] could not evaluate the residual at any sample point, so the "
            "family cannot be constrained at all",
            code="E-ANSATZ-003",
            remediation=(
                "the residual may be undefined everywhere the sampler looked; pass a "
                "different seed=, raise max_points=, or check the target for a pole"
            ),
        )
    steps.append(
        {
            "rule": f"ansatz_{method}",
            "before": str(target_residual),
            "after": f"{len(rows)} x {width + 1} augmented system over Q",
            "side_conditions": [f"{skipped} point(s) skipped as undefined"] if skipped else [],
        }
    )

    assignment, free, rank = _solve_rows(ansatz, rows)
    steps.append(
        {
            "rule": "ansatz_rref",
            "before": f"{len(rows)} equations in {width} unknowns",
            "after": f"rank {rank}, {len(free)} free parameter(s)",
            "side_conditions": [],
        }
    )

    fitted = _value(ak.simplify(_substitute(ansatz.pool, ansatz.expr, assignment, ansatz.symbols)))
    verification = _certify(
        ansatz,
        target_residual,
        assignment,
        free,
        certify=certify,
        method=method,
        seed=resolved_seed,
        tolerance=tolerance,
        samples=samples,
        steps=steps,
    )
    return AnsatzSolution(
        expr=fitted,
        assignment=assignment,
        free=free,
        rank=rank,
        status=verification["status"],
        verification=verification,
        steps=tuple(steps),
        residual=target_residual,
        check=_check_recipe(verification),
        points=tuple(tuple(str(v) for v in point) for point in used),
        ansatz=ansatz,
    )


def _check_recipe(verification: Mapping[str, Any]) -> dict[str, Any]:
    """A ``ClaimGraph.verify`` recipe, or ``{}`` when nothing was checked.

    ``certify="none"`` deliberately yields no recipe: a re-verification pass
    must report such a claim as ``skipped``, not re-derive a check that was
    never run.
    """
    substituted = verification.get("substituted")
    return {"kind": "zero", "expr": str(substituted)} if substituted else {}


def _certify(
    ansatz: Ansatz,
    residual: Any,
    assignment: Mapping[Any, Any],
    free: Sequence[Any],
    *,
    certify: str,
    method: str,
    seed: int,
    tolerance: float,
    samples: int,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Substitute the fit back and grade it, honestly.

    Solving is heuristic (collocation proves identical vanishing only for
    polynomial residuals of bounded degree); checking is exact and cheap. So
    the grade comes from the check, never from the solve.
    """
    if certify == "none":
        return {
            "status": "unverified",
            "evidence": "none",
            "method": f"ansatz_{method}",
            "externally_verified": False,
            "side_conditions": [],
        }

    substituted = _substitute(ansatz.pool, residual, assignment, ansatz.symbols)
    normal = _normal_form(substituted, (*ansatz.vars, *free))
    if str(normal).strip() == "0":
        steps.append(
            {
                "rule": "ansatz_back_substitution",
                "before": str(substituted),
                "after": "0",
                "side_conditions": [],
            }
        )
        return {
            "status": "exactly_verified",
            "evidence": "symbolic_residual_zero",
            "method": f"ansatz_{method}+back_substitution",
            "externally_verified": False,
            "side_conditions": [],
            "residual": "0",
            "substituted": str(substituted),
        }

    worst = _numeric_residual(substituted, ansatz.vars, free, seed, samples)
    steps.append(
        {
            "rule": "ansatz_back_substitution",
            "before": str(substituted),
            "after": str(normal),
            "side_conditions": ["residual did not normalise to 0"],
        }
    )
    if worst is None:
        evidence = (
            f"residual did not normalise to 0 (got {normal}) and could not be sampled "
            "numerically; the fit satisfies the collocation constraints only"
        )
    elif worst <= tolerance:
        evidence = (
            f"floating-point samples only: |residual| <= {worst:.3g} at {samples} fresh "
            f"point(s); the exact normal form is {normal}, not 0"
        )
    else:
        evidence = (
            f"the fit does NOT satisfy the residual: |residual| = {worst:.6g} at a fresh "
            f"point, exact normal form {normal}. It satisfies only the sampled constraints"
        )
    return {
        "status": "numerically_checked",
        "evidence": evidence,
        "method": f"ansatz_{method}+back_substitution",
        "externally_verified": False,
        "side_conditions": [],
        "residual": str(normal),
        "max_abs_residual": worst,
        "substituted": str(substituted),
    }


def _fit_nonlinear(
    ansatz: Ansatz, residual: Any, *, seed: int | None, steps: list[dict[str, Any]]
) -> AnsatzSolution:
    """Escalate a genuinely nonlinear residual to :func:`alkahest.solve`.

    Reached only when the residual is nonlinear in the unknowns *and* clearing
    denominators did not fix it. Requires a ``groebner`` build; refuses with
    ``E-ANSATZ-004`` otherwise instead of pretending the linear path applies.
    """
    ak = _ak()
    if not ak.capabilities().get("groebner", False):
        raise AnsatzError(
            "[E-ANSATZ-004] the residual is nonlinear in the unknowns "
            f"{list(ansatz.names)} and escalating to alkahest.solve needs a groebner build, "
            "which this is not",
            code="E-ANSATZ-004",
            remediation=(
                "rebuild with --features groebner, or reformulate the family so the "
                "residual is affine in the unknowns (for p/q use ansatz.rational, whose "
                "denominator-clearing transform keeps it linear)"
            ),
        )
    resolved_seed = _resolve_seed(seed)
    width = len(ansatz.unknowns)
    # Over-sample here too: one equation per unknown can be satisfiable while
    # the identity is not, and an overdetermined Groebner solve returning no
    # solution is what makes E-ANSATZ-003 reachable on this path at all.
    wanted = width + max(2, width)
    equations: list[Any] = []
    used: list[tuple[Fraction, ...]] = []
    stream = _sample_points(len(ansatz.vars), resolved_seed)
    attempts = 0
    while len(equations) < wanted and attempts < 8 * wanted + 32:
        attempts += 1
        point = next(stream)
        bindings = {var: _to_expr(ansatz.pool, value) for var, value in zip(ansatz.vars, point)}
        try:
            equation = _value(ak.subs(residual, bindings))
            _rational_value(_value(ak.subs(equation, dict.fromkeys(ansatz.unknowns, 0))))
        except _Undefined:
            continue
        except _NotExact:
            pass
        except Exception:
            continue
        equations.append(equation)
        used.append(point)

    steps.append(
        {
            "rule": "ansatz_nonlinear_escalation",
            "before": str(residual),
            "after": f"{len(equations)} polynomial equation(s) handed to alkahest.solve",
            "side_conditions": ["residual is nonlinear in the unknowns"],
        }
    )
    solutions = ak.solve(equations, list(ansatz.unknowns))
    if not solutions:
        raise AnsatzError(
            "[E-ANSATZ-003] the constraints are inconsistent: no member of this "
            f"{ansatz.family} family satisfies them (nonlinear system, no solutions)",
            code="E-ANSATZ-003",
            remediation=(
                "this is a result, not a malfunction — record the closed branch, or "
                "enlarge the family and re-fit"
            ),
        )
    chosen = min(solutions, key=lambda s: sorted((str(k), str(v)) for k, v in s.items()))
    assignment = {key: _value(value) for key, value in chosen.items()}
    fitted = _value(ak.simplify(_substitute(ansatz.pool, ansatz.expr, assignment, ansatz.symbols)))
    verification = _certify(
        ansatz,
        residual,
        assignment,
        (),
        certify="residual",
        method="groebner_escalation",
        seed=resolved_seed,
        tolerance=1e-8,
        samples=5,
        steps=steps,
    )
    return AnsatzSolution(
        expr=fitted,
        assignment=assignment,
        free=(),
        rank=len(assignment),
        status=verification["status"],
        verification=verification,
        steps=tuple(steps),
        residual=residual,
        check=_check_recipe(verification),
        points=tuple(tuple(str(v) for v in point) for point in used),
        ansatz=ansatz,
    )


# ---------------------------------------------------------------------------
# Hand-off to positivity
# ---------------------------------------------------------------------------


def certify_nonneg(
    candidate: Any,
    vars: Sequence[Any] | None = None,
    *,
    constraints: Sequence[Any] = (),
    **kwargs: Any,
) -> Any:
    """Hand a fitted candidate to :func:`alkahest.prove_nonneg`.

    The ansatz module's job ends when it has produced the candidate. Positivity
    is decided by the existing certificate machinery — this is a one-line
    adapter that unwraps an :class:`AnsatzSolution` or :class:`Ansatz` and
    forwards, so that a Lyapunov workflow reads

    ``cert = certify_nonneg(fit(quadratic_form(pool, [x, y]), residual))``

    and every outcome (a :class:`~alkahest.PositivityCertificate`, an
    ``E-SOS-003`` refutation with a witness, an ``E-SOS-002`` "no certificate
    of this shape at this degree") comes back unmodified. Nothing about
    positivity is reimplemented or reinterpreted here.

    Parameters
    ----------
    candidate : AnsatzSolution, Ansatz, Expr or DerivedResult
        The expression to certify. An :class:`AnsatzSolution` still carrying
        free parameters is refused — an undetermined form is not a candidate.
    vars : sequence of Expr, optional
        Variables; taken from the ansatz when *candidate* carries one.
    constraints : sequence of Expr
        Forwarded as ``prove_nonneg(constraints=...)``.
    **kwargs
        Forwarded verbatim.

    Returns
    -------
    PositivityCertificate

    Raises
    ------
    AnsatzError
        ``E-ANSATZ-003`` if the candidate still has free parameters.
    SosError
        Whatever :func:`alkahest.prove_nonneg` raises — passed through.
    """
    ak = _ak()
    variables = vars
    if isinstance(candidate, AnsatzSolution):
        if candidate.free:
            raise AnsatzError(
                "[E-ANSATZ-003] this fit still has free parameters "
                f"{[str(f) for f in candidate.free]}, so it is a family, not a candidate",
                code="E-ANSATZ-003",
                remediation=(
                    "instantiate the free parameters first, e.g. "
                    "solution.ansatz.instantiate({...}), then certify that member"
                ),
            )
        variables = variables if variables is not None else candidate.ansatz.vars
        expr = candidate.expr
    elif isinstance(candidate, Ansatz):
        variables = variables if variables is not None else candidate.vars
        expr = candidate.expr
    else:
        expr = _value(candidate)
    if variables is None:
        symbols = _free_symbols(expr)
        variables = [symbols[key] for key in sorted(symbols)]
    return ak.prove_nonneg(expr, list(variables), constraints=list(constraints), **kwargs)
