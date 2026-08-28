"""Cross-CAS differential testing — ask a second CAS the same question.

Certificates cover the fragment where Alkahest can *prove* its answer. Outside
that fragment there is nothing checking the answer at all, and the failure that
matters for a search loop is not a crash but a **silent error**: a confident,
plausible, wrong result that the loop then builds a hundred derived claims on
top of (see ``tests/silent_errors/README.md``). An independent implementation
is the cheapest instrument that catches exactly those.

This module runs one query through Alkahest and through an oracle and reports
whether they agree — as a four-valued outcome, never a boolean:

``agree``
    Both systems answered, and a named comparison rung settled it.
``diverge``
    Both systems answered and the answers are not the same. The record names
    **two suspects**, carries the witness point and both values, and never
    words itself as "Alkahest is right".
``incomparable``
    The question could not be posed identically to both systems (an
    untranslatable node, an assumption context with no faithful mapping, an
    operation one side refused). **Never** collapsed into ``agree``.
``unavailable``
    No oracle is installed. Also never ``agree`` — a loop that believes it is
    cross-checking and is not is worse off than one that knows it isn't.

Quick start
-----------
>>> import alkahest as ak
>>> from alkahest.crosscheck import check, oracles
>>> "sympy" in oracles()          # None when SymPy is not installed
True
>>> pool = ak.ExprPool()
>>> x = pool.symbol("x")
>>> out = check("integrate", ak.sin(x), x)
>>> out.outcome in {"agree", "unavailable"}
True

Design notes
------------
*One translator, not four.* ``tests/`` already carries four hand-rolled
``_expr_to_sympy`` helpers, each covering a different subset. :class:`Translator`
is the one implementation; the tests are meant to migrate onto it. Building the
QA harness and the runtime mode separately would produce two translators that
disagree, which is the worst possible outcome for a tool whose entire job is
detecting disagreement.

*Total-or-refuse.* Every tag :meth:`alkahest.Expr.node` can emit appears in
:data:`Translator._DISPATCH`; an unknown tag, an unmapped primitive, or an
assumption context with no faithful counterpart raises
:class:`~alkahest.CrossCheckError` (``E-XCHECK-001``), which :func:`check` turns
into ``incomparable``. **False divergences are worse than no signal** — they
train both the loop and the team to ignore the alarm, and a best-effort
translator manufactures them by construction.

*Opt-in per call site.* There is deliberately no ``context(crosscheck=True)``:
that would put an oracle round-trip on every call, and stage-2 falsification
runs millions of times. Call :func:`check` where you want it — at stage-5
recording frequency, hundreds of times, not millions.

See ``docs/mdbook/src/crosscheck.md``.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .exceptions import CrossCheckError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Mapping, Sequence

__all__ = [
    "FROZEN_CORPUS",
    "FUNCTION_MAP",
    "NODE_TAGS",
    "OPERATIONS",
    "OUTCOMES",
    "PREDICATE_KINDS",
    "REFUSED_FUNCTIONS",
    "RUNG_NAMES",
    "SAMPLE_BOX",
    "SWEEP_OPERATIONS",
    "CrossCheck",
    "Divergence",
    "FrozenCase",
    "Operation",
    "Oracle",
    "SweepReport",
    "SymPyOracle",
    "SymPyTranslator",
    "Translator",
    "check",
    "oracles",
    "register_oracle",
    "run_frozen_corpus",
    "sweep",
    "to_sympy",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The four outcomes, in decreasing order of how much they let you conclude.
#: ``incomparable`` and ``unavailable`` are *not* weaker forms of ``agree``;
#: they say the check did not happen, and code that treats them as agreement
#: has reintroduced the exact failure this module exists to prevent.
OUTCOMES = ("agree", "diverge", "incomparable", "unavailable")

#: Comparison ladder. The rung that settled a check is always recorded, because
#: the rungs differ in what they license: 1, 2 and 4 are proofs of agreement
#: within their scope, while 3 only ever *fails to refute* at the points sampled.
RUNG_NAMES: dict[int, str] = {
    1: "syntactic",
    2: "symbolic",
    3: "rigorous_numeric",
    4: "invariant",
}

RUNG_SYNTACTIC = 1
RUNG_SYMBOLIC = 2
RUNG_NUMERIC = 3
RUNG_INVARIANT = 4

#: Every tag :meth:`alkahest.Expr.node` can emit. Authoritative mirror of the
#: ``ExprData`` match in ``alkahest-py/src/lib.rs``; ``tests/test_crosscheck.py``
#: re-derives the set from that source so the table cannot silently drift when a
#: new node kind lands.
NODE_TAGS = frozenset(
    {
        "symbol",
        "integer",
        "rational",
        "float",
        "add",
        "mul",
        "pow",
        "func",
        "piecewise",
        "predicate",
        "forall",
        "exists",
        "big_o",
        "root_sum",
    }
)

#: Every ``kind`` a ``predicate`` node can carry.
PREDICATE_KINDS = frozenset(
    {"lt", "le", "gt", "ge", "eq", "ne", "and", "or", "not", "true", "false"}
)

#: Alkahest primitive name -> SymPy callable name, for the primitives whose
#: conventions provably coincide. Deliberately *not* ``getattr(sympy, name)``:
#: that is how a translator ends up quietly mapping ``ceil`` onto nothing, or
#: ``EllipticK`` onto a function of a different argument.
FUNCTION_MAP: dict[str, str] = {
    "abs": "Abs",
    "acos": "acos",
    "acosh": "acosh",
    "arg": "arg",
    "asin": "asin",
    "asinh": "asinh",
    "atan": "atan",
    "atan2": "atan2",
    "atanh": "atanh",
    "ceil": "ceiling",
    "conjugate": "conjugate",
    "cos": "cos",
    "cosh": "cosh",
    "digamma": "digamma",
    "diracdelta": "DiracDelta",
    "erf": "erf",
    "erfc": "erfc",
    "exp": "exp",
    "floor": "floor",
    "gamma": "gamma",
    "im": "im",
    "lambert_w": "LambertW",
    "log": "log",
    "max": "Max",
    "min": "Min",
    "re": "re",
    "sign": "sign",
    "sin": "sin",
    "sinh": "sinh",
    "sqrt": "sqrt",
    "tan": "tan",
    "tanh": "tanh",
    # Exponential-integral family. Alkahest and SymPy both follow DLMF §6.2 —
    # same principal values, same branch cut on the negative reals, same
    # `li(z) = Ei(log z)` — so these are name-for-name. NB SymPy's `Li` (capital
    # L) is the *offset* logarithmic integral `li(x) - li(2)`; mapping `li` onto
    # it would manufacture a constant divergence, which is exactly the failure
    # this table exists to prevent.
    "Ei": "Ei",
    "li": "li",
    "Si": "Si",
    "Ci": "Ci",
    "Shi": "Shi",
    "Chi": "Chi",
}

#: Bessel functions of fixed integer order, which SymPy spells with the order as
#: an argument. Faithful, just not a name-for-name rename.
_FIXED_ORDER_BESSEL: dict[str, int] = {"bessel_j0": 0, "bessel_j1": 1}

#: Primitives with a *known* convention mismatch, refused with the reason spelt
#: out. Each of these would translate "successfully" and then manufacture
#: divergences that say nothing about either system's correctness.
REFUSED_FUNCTIONS: dict[str, str] = {
    "heaviside": (
        "SymPy's Heaviside(0) defaults to 1/2 while Alkahest fixes no value there, "
        "so the two agree everywhere except on a point the comparison would sample"
    ),
    "round": (
        "SymPy has no symbolic rounding function with a documented half-way rule to compare against"
    ),
    "EllipticK": "elliptic integrals differ in modulus-vs-parameter convention between systems",
    "EllipticE": "elliptic integrals differ in modulus-vs-parameter convention between systems",
    "EllipticF": "elliptic integrals differ in modulus-vs-parameter convention between systems",
    "EllipticPi": "elliptic integrals differ in modulus-vs-parameter convention between systems",
}

#: Pool symbols with a reserved meaning. ``ExprPool.pos_infinity()`` and
#: ``ExprPool.imaginary_unit()`` intern ordinary symbols with these names, so a
#: translator that treats them as free variables silently asks the oracle a
#: different question.
RESERVED_SYMBOLS = ("∞", "I")


# ---------------------------------------------------------------------------
# Lazy package handles (this module is imported from ``alkahest/__init__.py``)
# ---------------------------------------------------------------------------


def _ak() -> Any:
    import alkahest

    return alkahest


def _refuse(message: str, *, code: str = "E-XCHECK-001", remediation: str) -> CrossCheckError:
    return CrossCheckError(message, code=code, remediation=remediation)


# ---------------------------------------------------------------------------
# The translator
# ---------------------------------------------------------------------------


class Translator(ABC):
    """Walk an :class:`alkahest.Expr` into another CAS's term language.

    Subclass once per oracle. The walk, the dispatch table, and the refusal
    discipline live here; a subclass supplies only the leaf and node
    constructors of its target language.

    The dispatch table is **total over** :data:`NODE_TAGS` — that is the
    property that makes a divergence informative, and
    ``tests/test_crosscheck.py`` asserts it against the tags the Rust binding
    can actually emit rather than against this list, so a new node kind fails
    the build instead of quietly translating to nothing.

    Assumptions
    -----------
    :meth:`translate` accepts the governing :class:`alkahest.Assumptions` and
    maps its predicates onto per-symbol flags. A predicate with no faithful
    counterpart (a relation between two symbols, a disjunction, a bound on a
    composite expression) raises rather than being dropped: dropping it asks
    the oracle a *weaker* question and any divergence that follows is an
    artefact of the harness.
    """

    #: Node tag -> handler method name. Total over :data:`NODE_TAGS`.
    _DISPATCH: Mapping[str, str] = {
        "symbol": "_symbol",
        "integer": "_integer",
        "rational": "_rational",
        "float": "_float",
        "add": "_add",
        "mul": "_mul",
        "pow": "_pow",
        "func": "_func",
        "piecewise": "_piecewise",
        "predicate": "_predicate",
        "forall": "_forall",
        "exists": "_exists",
        "big_o": "_big_o",
        "root_sum": "_root_sum",
    }

    def __init__(self) -> None:
        self._flags: dict[str, dict[str, bool]] = {}

    # -- public entry point -------------------------------------------------

    def translate(self, expr: Any, *, assumptions: Any = None) -> Any:
        """Translate *expr*, honouring *assumptions*, or refuse.

        Parameters
        ----------
        expr : Expr or DerivedResult
            The expression to translate.
        assumptions : Assumptions, optional
            Governing assumption context. When ``None``, the ambient
            :func:`alkahest.active_assumptions` context is used, so a check run
            inside ``with alkahest.context(assumptions=...)`` asks the oracle
            the same conditioned question the caller asked Alkahest.

        Returns
        -------
        object
            A term in the oracle's language.

        Raises
        ------
        CrossCheckError
            ``E-XCHECK-001`` for an unknown node tag, an unmapped primitive, or
            an assumption with no faithful counterpart.
        """
        ak = _ak()
        if isinstance(expr, ak.DerivedResult):
            expr = expr.value
        if assumptions is None:
            assumptions = ak.active_assumptions()
        self._flags = self._symbol_flags(assumptions)
        return self._walk(expr)

    def assumption_flags(self, assumptions: Any) -> dict[str, dict[str, bool]]:
        """Public view of the symbol-flag mapping derived from *assumptions*."""
        return self._symbol_flags(assumptions)

    # -- the walk -----------------------------------------------------------

    def _walk(self, expr: Any) -> Any:
        node = expr.node()
        tag = node[0]
        handler = self._DISPATCH.get(tag)
        if handler is None:
            raise _refuse(
                f"no faithful translation for node tag {tag!r}",
                remediation=(
                    "alkahest.crosscheck refuses rather than guessing: a best-effort "
                    "translation of an unknown node manufactures divergences that say "
                    "nothing about either system. Add the tag to Translator._DISPATCH "
                    "with a mapping you can defend, or cross-check a different route."
                ),
            )
        return getattr(self, handler)(node)

    def _walk_all(self, exprs: Sequence[Any]) -> list[Any]:
        return [self._walk(e) for e in exprs]

    # -- assumptions --------------------------------------------------------

    def _symbol_flags(self, assumptions: Any) -> dict[str, dict[str, bool]]:
        """Map assumption predicates onto per-symbol flags, or refuse."""
        if assumptions is None:
            return {}
        flags: dict[str, dict[str, bool]] = {}
        for predicate in assumptions.predicates:
            node = predicate.node()
            if node[0] != "predicate":
                raise _refuse(
                    f"assumption {predicate} is not a predicate",
                    remediation="Assumption contexts must contain predicate expressions.",
                )
            kind, args = node[1], node[2]
            name = self._assumption_flag(kind, args)
            if name is None:
                raise _refuse(
                    f"no faithful translation for the active assumption {predicate}",
                    remediation=(
                        "Only sign and non-zero conditions on a bare symbol map onto "
                        "oracle symbol flags. Drop the assumption for the duration of "
                        "the check, or accept outcome='incomparable' — asking the "
                        "oracle a weaker question would turn every legitimate "
                        "refinement into a false divergence."
                    ),
                )
            symbol_name, flag = name
            flags.setdefault(symbol_name, {})[flag] = True
        return flags

    @staticmethod
    def _assumption_flag(kind: str, args: Sequence[Any]) -> tuple[str, str] | None:
        """``(symbol_name, flag)`` for a mappable predicate, else ``None``.

        Mappable shapes are exactly ``sym <rel> 0`` for the four order relations
        and ``sym != 0`` — the conditions that correspond one-for-one to a
        symbol-level flag in every CAS that has such flags. Everything else
        (relations between symbols, conditions on composites, boolean
        combinations, quantified statements) has no per-symbol counterpart.
        """
        flag_by_kind = {
            "gt": "positive",
            "ge": "nonnegative",
            "lt": "negative",
            "le": "nonpositive",
            "ne": "nonzero",
        }
        flag = flag_by_kind.get(kind)
        if flag is None or len(args) != 2:
            return None
        lhs, rhs = args
        if lhs.node()[0] != "symbol":
            return None
        rhs_node = rhs.node()
        if rhs_node[0] != "integer" or int(rhs_node[1]) != 0:
            return None
        return str(lhs.node()[1]), flag

    # -- leaf and node constructors (per oracle) ----------------------------

    @abstractmethod
    def _symbol(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _integer(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _rational(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _float(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _add(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _mul(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _pow(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _func(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _piecewise(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _predicate(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _forall(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _exists(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _big_o(self, node: Sequence[Any]) -> Any: ...

    @abstractmethod
    def _root_sum(self, node: Sequence[Any]) -> Any: ...


class SymPyTranslator(Translator):
    """:class:`Translator` into SymPy's expression language."""

    def __init__(self, sympy_module: Any) -> None:
        super().__init__()
        self.sympy = sympy_module

    # -- leaves -------------------------------------------------------------

    def _symbol(self, node: Sequence[Any]) -> Any:
        name = str(node[1])
        if name == "∞":
            return self.sympy.oo
        if name == "I":
            return self.sympy.I
        return self.sympy.Symbol(name, **self._flags.get(name, {}))

    def _integer(self, node: Sequence[Any]) -> Any:
        return self.sympy.Integer(int(node[1]))

    def _rational(self, node: Sequence[Any]) -> Any:
        return self.sympy.Rational(int(node[1]), int(node[2]))

    def _float(self, node: Sequence[Any]) -> Any:
        # Alkahest prints floats at full precision; ``Float(str)`` keeps every
        # digit, where ``Float(float(...))`` would re-round through binary.
        return self.sympy.Float(str(node[1]))

    # -- structure ----------------------------------------------------------

    def _add(self, node: Sequence[Any]) -> Any:
        return self.sympy.Add(*self._walk_all(node[1]))

    def _mul(self, node: Sequence[Any]) -> Any:
        return self.sympy.Mul(*self._walk_all(node[1]))

    def _pow(self, node: Sequence[Any]) -> Any:
        return self.sympy.Pow(self._walk(node[1]), self._walk(node[2]))

    def _func(self, node: Sequence[Any]) -> Any:
        name = str(node[1])
        args = self._walk_all(node[2])
        reason = REFUSED_FUNCTIONS.get(name)
        if reason is not None:
            raise _refuse(
                f"no faithful translation for primitive {name!r}: {reason}",
                remediation=(
                    "Cross-check a route that avoids this primitive. Translating it "
                    "anyway would produce divergences caused by the convention "
                    "mismatch rather than by either system being wrong."
                ),
            )
        order = _FIXED_ORDER_BESSEL.get(name)
        if order is not None:
            return self.sympy.besselj(order, *args)
        mapped = FUNCTION_MAP.get(name)
        if mapped is None:
            raise _refuse(
                f"no faithful translation for primitive {name!r}",
                remediation=(
                    "Add it to alkahest.crosscheck.FUNCTION_MAP once you have checked "
                    "that the two systems use the same branch and the same argument "
                    "convention, or to REFUSED_FUNCTIONS with the reason they do not."
                ),
            )
        return getattr(self.sympy, mapped)(*args)

    def _piecewise(self, node: Sequence[Any]) -> Any:
        branches = [(self._walk(value), self._walk(cond)) for cond, value in node[1]]
        branches.append((self._walk(node[2]), self.sympy.true))
        return self.sympy.Piecewise(*branches)

    def _predicate(self, node: Sequence[Any]) -> Any:
        kind = str(node[1])
        args = self._walk_all(node[2])
        builders: dict[str, Callable[..., Any]] = {
            "lt": self.sympy.StrictLessThan,
            "le": self.sympy.LessThan,
            "gt": self.sympy.StrictGreaterThan,
            "ge": self.sympy.GreaterThan,
            "eq": self.sympy.Eq,
            "ne": self.sympy.Ne,
            "and": self.sympy.And,
            "or": self.sympy.Or,
            "not": self.sympy.Not,
        }
        if kind == "true":
            return self.sympy.true
        if kind == "false":
            return self.sympy.false
        builder = builders.get(kind)
        if builder is None:
            raise _refuse(
                f"no faithful translation for predicate kind {kind!r}",
                remediation="Extend SymPyTranslator._predicate with a defended mapping.",
            )
        return builder(*args)

    def _forall(self, node: Sequence[Any]) -> Any:
        return _refuse_quantifier("∀")

    def _exists(self, node: Sequence[Any]) -> Any:
        return _refuse_quantifier("∃")

    def _big_o(self, node: Sequence[Any]) -> Any:
        return self.sympy.O(self._walk(node[1]))

    def _root_sum(self, node: Sequence[Any]) -> Any:
        poly = self._walk(node[1])
        var = self._walk(node[2])
        body = self._walk(node[3])
        return self.sympy.RootSum(self.sympy.Poly(poly, var), self.sympy.Lambda(var, body))


def _refuse_quantifier(label: str) -> Any:
    """Refuse to translate a quantifier.

    SymPy has no first-order quantifier that composes with its arithmetic —
    ``Q``/``ask`` is an assumption query, not a term former, and neither
    ``simplify`` nor ``evalf`` means anything on one. Encoding ``∀`` as anything
    SymPy *will* accept produces an object the comparison ladder cannot use,
    and every rung would then answer "not equal" for reasons that have nothing
    to do with the mathematics.
    """
    raise _refuse(
        f"no faithful translation for a {label} quantifier",
        remediation=(
            "SymPy has no term-level quantifier to compare against. Cross-check the "
            "quantifier-free matrix, or use alkahest.decide, whose answer is a truth "
            "value the ladder can compare."
        ),
    )


def to_sympy(expr: Any, *, assumptions: Any = None) -> Any:
    """Translate an Alkahest expression into SymPy, or refuse.

    The one translator this package ships, exposed directly for callers who
    want the term rather than a comparison. Total over :data:`NODE_TAGS`:
    anything it cannot map faithfully raises rather than degrading.

    Parameters
    ----------
    expr : Expr or DerivedResult
        Expression to translate.
    assumptions : Assumptions, optional
        Governing assumption context; defaults to
        :func:`alkahest.active_assumptions`.

    Returns
    -------
    sympy.Basic

    Raises
    ------
    CrossCheckError
        ``E-XCHECK-002`` if SymPy is not installed, ``E-XCHECK-001`` if the
        expression or the assumption context has no faithful mapping.

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.crosscheck import oracles, to_sympy
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> if oracles()["sympy"]:
    ...     str(to_sympy(ak.sin(x) ** 2))
    ... else:
    ...     'sin(x)**2'
    'sin(x)**2'
    """
    oracle = SymPyOracle()
    return oracle.translate(expr, assumptions=assumptions)


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


class Oracle(ABC):
    """An independent CAS the comparator can interrogate.

    The comparator talks to oracles **only** through this interface, so a
    second implementation (Wolfram, Maxima, a pinned older Alkahest) drops in
    without the ladder changing. Two consequences worth stating:

    * every method may answer "I don't know" — :meth:`is_zero` returns ``None``,
      :meth:`run` raises — and the comparator turns that into ``incomparable``
      rather than guessing;
    * :meth:`lift` is optional in effect but not in signature: it is what lets
      rung 2 be attempted *independently in both systems*, which is what makes
      "either one proved it" a defensible notion of agreement.

    Attributes
    ----------
    name : str
        Stable short identifier, used as the key in :func:`oracles`.
    """

    name: str = "oracle"

    # -- availability -------------------------------------------------------

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """Is this oracle usable in this process right now?"""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string of the backing system. Recorded on every result."""

    # -- language -----------------------------------------------------------

    @abstractmethod
    def translate(self, expr: Any, *, assumptions: Any = None) -> Any:
        """Alkahest :class:`~alkahest.Expr` -> a term in this oracle's language."""

    @abstractmethod
    def lift(self, obj: Any, pool: Any) -> Any:
        """This oracle's term -> an Alkahest :class:`~alkahest.Expr`, or refuse."""

    @abstractmethod
    def render(self, obj: Any) -> str:
        """Human-readable rendering, for witnesses and reports."""

    @abstractmethod
    def canonical(self, obj: Any) -> str:
        """Canonical string form, for rung 1."""

    # -- operations ---------------------------------------------------------

    @abstractmethod
    def supports(self, operation: str) -> bool:
        """Can this oracle answer *operation* at all?"""

    @abstractmethod
    def run(self, operation: str, args: Mapping[str, Any]) -> Any:
        """Perform *operation* in this oracle's own engine.

        Raises
        ------
        CrossCheckError
            ``E-XCHECK-004`` when the oracle declines or returns an
            unevaluated form; a refusal is not a divergence.
        """

    # -- reasoning primitives the ladder needs ------------------------------

    @abstractmethod
    def is_zero(self, obj: Any) -> bool | None:
        """``True``/``False`` if decided, ``None`` if the oracle could not."""

    @abstractmethod
    def diff(self, obj: Any, var: Any) -> Any:
        """Derivative of *obj* with respect to the translated variable *var*."""

    @abstractmethod
    def subs(self, obj: Any, bindings: Mapping[Any, Any]) -> Any:
        """Substitute translated terms for translated symbols."""

    @abstractmethod
    def subs_by_name(self, obj: Any, point: Mapping[str, float]) -> Any:
        """Substitute numbers for free symbols matched **by name**.

        Distinct from :meth:`subs` because the comparator only ever knows the
        Alkahest symbol *names* at a witness point, and rebuilding an oracle
        symbol from a name would have to re-derive its assumption flags — get
        that wrong and the substitution silently misses.
        """

    @abstractmethod
    def to_float(self, obj: Any) -> float | None:
        """Numeric value of *obj*, or ``None`` when it has none."""


class SymPyOracle(Oracle):
    """SymPy as the reference implementation.

    The first oracle, and the one the frozen corpus is pinned against. Its
    version is recorded on **every** record: without that the corpus rots
    silently the first time SymPy changes an answer, which it will.

    Raises
    ------
    CrossCheckError
        ``E-XCHECK-002`` on construction when SymPy is not importable. The
        absence is loud by construction — there is no code path in which a
        missing SymPy yields agreement.
    """

    name = "sympy"

    def __init__(self) -> None:
        self.sympy = _import_sympy()
        self._translator = SymPyTranslator(self.sympy)

    # -- availability -------------------------------------------------------

    @classmethod
    def available(cls) -> bool:
        try:
            _import_sympy()
        except CrossCheckError:
            return False
        return True

    @property
    def version(self) -> str:
        return str(self.sympy.__version__)

    # -- language -----------------------------------------------------------

    def translate(self, expr: Any, *, assumptions: Any = None) -> Any:
        return self._translator.translate(expr, assumptions=assumptions)

    def lift(self, obj: Any, pool: Any) -> Any:
        """SymPy term -> Alkahest ``Expr``, refusing anything not exactly representable.

        Deliberately narrow. A lift that "mostly works" would put an
        approximated value into the Alkahest-side rung-2 attempt, and a wrong
        answer there is indistinguishable from the divergence the module exists
        to find.
        """
        sp = self.sympy
        ak = _ak()

        def go(o: Any) -> Any:
            if o is sp.oo:
                return pool.pos_infinity()
            if o is sp.I:
                return pool.imaginary_unit()
            if o.is_Integer:
                return pool.integer(int(o))
            if o.is_Rational:
                return pool.rational(int(o.p), int(o.q))
            if o.is_Float:
                return pool.float(float(o))
            if o.is_Symbol:
                return pool.symbol(str(o.name))
            if o.is_Add:
                return _fold(go(a) for a in o.args)
            if o.is_Mul:
                terms = [go(a) for a in o.args]
                out = terms[0]
                for t in terms[1:]:
                    out = out * t
                return out
            if o.is_Pow:
                return go(o.base) ** go(o.exp)
            head = type(o).__name__
            inverse = _SYMPY_TO_ALKAHEST.get(head)
            if inverse is not None and hasattr(ak, inverse):
                return getattr(ak, inverse)(*[go(a) for a in o.args])
            raise _refuse(
                f"cannot lift SymPy {head} back into Alkahest exactly",
                remediation=(
                    "The oracle's answer stays in the oracle for this check; the "
                    "Alkahest-side symbolic rung is skipped rather than run against an "
                    "approximation."
                ),
            )

        def _fold(parts: Iterator[Any]) -> Any:
            items = list(parts)
            out = items[0]
            for item in items[1:]:
                out = out + item
            return out

        return go(self.sympy.sympify(obj))

    def render(self, obj: Any) -> str:
        return str(obj)

    def canonical(self, obj: Any) -> str:
        return self.sympy.srepr(self.sympy.sympify(obj))

    # -- operations ---------------------------------------------------------

    def supports(self, operation: str) -> bool:
        return operation in _SYMPY_RUNNERS

    def run(self, operation: str, args: Mapping[str, Any]) -> Any:
        runner = _SYMPY_RUNNERS.get(operation)
        if runner is None:
            raise _refuse(
                f"SymPy oracle has no implementation of {operation!r}",
                code="E-XCHECK-004",
                remediation=(
                    "Add it to alkahest.crosscheck._SYMPY_RUNNERS together with a "
                    "comparison rung in OPERATIONS."
                ),
            )
        try:
            result = runner(self.sympy, args)
        except CrossCheckError:
            raise
        except Exception as exc:
            raise _refuse(
                f"SymPy declined {operation}: {type(exc).__name__}: {exc}",
                code="E-XCHECK-004",
                remediation=(
                    "The oracle refused. That is not a divergence and is not recorded "
                    "as one; the check reports outcome='incomparable'."
                ),
            ) from exc
        self._reject_unevaluated(operation, result)
        return result

    def _reject_unevaluated(self, operation: str, result: Any) -> None:
        """An unevaluated ``Integral``/``Sum``/``Limit`` is a refusal, not an answer."""
        sp = self.sympy
        unevaluated = (sp.Integral, sp.Sum, sp.Limit, sp.Derivative)
        candidates = result if isinstance(result, (list, tuple)) else [result]
        for item in candidates:
            if isinstance(item, dict):
                items = list(item.values())
            elif isinstance(item, (list, tuple)):
                items = list(item)
            else:
                items = [item]
            for value in items:
                if isinstance(value, sp.Basic) and value.has(*unevaluated):
                    raise _refuse(
                        f"SymPy returned an unevaluated form for {operation}: {value}",
                        code="E-XCHECK-004",
                        remediation=(
                            "SymPy did not answer. Reported as 'incomparable' — an "
                            "unevaluated Integral is a refusal and comparing against it "
                            "would fabricate a divergence."
                        ),
                    )

    # -- reasoning primitives ----------------------------------------------

    def is_zero(self, obj: Any) -> bool | None:
        sp = self.sympy
        try:
            simplified = sp.simplify(sp.expand(obj))
        except Exception:
            return None
        if simplified == 0:
            return True
        try:
            verdict = simplified.is_zero
        except Exception:
            return None
        return None if verdict is None else bool(verdict)

    def diff(self, obj: Any, var: Any) -> Any:
        return self.sympy.diff(obj, var)

    def subs(self, obj: Any, bindings: Mapping[Any, Any]) -> Any:
        return self.sympy.sympify(obj).subs(dict(bindings))

    def subs_by_name(self, obj: Any, point: Mapping[str, float]) -> Any:
        term = self.sympy.sympify(obj)
        bindings = {s: point[str(s)] for s in term.free_symbols if str(s) in point}
        return term.subs(bindings)

    def to_float(self, obj: Any) -> float | None:
        try:
            value = complex(self.sympy.sympify(obj).evalf(30))
        except Exception:
            return None
        if abs(value.imag) > 1e-18:
            return None
        real = value.real
        if real != real or real in (float("inf"), float("-inf")):
            return None
        return real


#: SymPy head name -> the ``alkahest`` callable that rebuilds it, for
#: :meth:`SymPyOracle.lift`. Only the round-trippable half of
#: :data:`FUNCTION_MAP`; anything absent makes ``lift`` refuse.
_SYMPY_TO_ALKAHEST: dict[str, str] = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "erf": "erf",
    "gamma": "gamma",
    "Abs": "abs",
    "sign": "sign",
    # Exponential-integral family (see FUNCTION_MAP above for why the
    # conventions coincide).
    "Ei": "exp_integral_ei",
    "li": "log_integral",
    "Si": "sin_integral",
    "Ci": "cos_integral",
    "Shi": "sinh_integral",
    "Chi": "cosh_integral",
}


def _import_sympy() -> Any:
    try:
        import sympy
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise CrossCheckError(
            "no cross-check oracle is installed (SymPy import failed)",
            code="E-XCHECK-002",
            remediation=(
                "pip install sympy. Until then alkahest.crosscheck.check returns "
                "outcome='unavailable' — never 'agree'. Call "
                "alkahest.crosscheck.oracles() at session start to find out before "
                "you plan around the check."
            ),
        ) from exc
    return sympy


# -- SymPy operation runners ------------------------------------------------
#
# One per entry in OPERATIONS. Keyed by the same names as the alkahest module
# namespace, so `check("integrate", ...)` runs `alkahest.integrate` on one side
# and this table's `integrate` on the other.


def _sympy_diff(sp: Any, args: Mapping[str, Any]) -> Any:
    return sp.diff(args["expr"], args["var"])


def _sympy_integrate(sp: Any, args: Mapping[str, Any]) -> Any:
    if args.get("lower") is not None:
        return sp.integrate(args["expr"], (args["var"], args["lower"], args["upper"]))
    return sp.integrate(args["expr"], args["var"])


def _sympy_simplify(sp: Any, args: Mapping[str, Any]) -> Any:
    return sp.simplify(args["expr"])


def _sympy_expand(sp: Any, args: Mapping[str, Any]) -> Any:
    return sp.expand(args["expr"])


def _sympy_limit(sp: Any, args: Mapping[str, Any]) -> Any:
    return sp.limit(args["expr"], args["var"], args["point"])


def _sympy_series(sp: Any, args: Mapping[str, Any]) -> Any:
    return sp.series(args["expr"], args["var"], args["point"], args["order"]).removeO()


def _sympy_sum_indefinite(sp: Any, args: Mapping[str, Any]) -> Any:
    from sympy.concrete.gosper import gosper_sum

    out = gosper_sum(args["expr"], args["var"])
    if out is None:
        raise _refuse(
            "SymPy's Gosper implementation found no hypergeometric antidifference",
            code="E-XCHECK-004",
            remediation="A refusal, not a divergence; reported as 'incomparable'.",
        )
    return out


def _sympy_solve(sp: Any, args: Mapping[str, Any]) -> Any:
    solutions = sp.solve(list(args["equations"]), list(args["vars"]), dict=True)
    return [{str(k): v for k, v in sol.items()} for sol in solutions]


_SYMPY_RUNNERS: dict[str, Callable[[Any, Mapping[str, Any]], Any]] = {
    "diff": _sympy_diff,
    "integrate": _sympy_integrate,
    "simplify": _sympy_simplify,
    "simplify_expanded": _sympy_expand,
    "limit": _sympy_limit,
    "series": _sympy_series,
    "sum_indefinite": _sympy_sum_indefinite,
    "solve": _sympy_solve,
}


# ---------------------------------------------------------------------------
# Oracle registry
# ---------------------------------------------------------------------------

_ORACLE_CLASSES: dict[str, type[Oracle]] = {"sympy": SymPyOracle}


def register_oracle(oracle_cls: type[Oracle]) -> None:
    """Register an :class:`Oracle` implementation under its ``name``.

    The comparator resolves oracles through this registry, so a new backend is
    a class plus one call — no change to the ladder, the outcomes, or the
    frozen corpus machinery.

    Parameters
    ----------
    oracle_cls : type[Oracle]
        Class (not instance); it is constructed lazily, and only when
        :meth:`Oracle.available` says it can be.
    """
    _ORACLE_CLASSES[oracle_cls.name] = oracle_cls


def oracles() -> dict[str, str | None]:
    """Which oracles are installed, and at what version.

    Reports every *known* oracle, including the absent ones — an agent must be
    able to tell that SymPy is missing **before** it plans around a check that
    will only ever return ``unavailable``.

    Returns
    -------
    dict
        ``{"sympy": "1.14.0", "wolfram": None}``-shaped; ``None`` means
        "known to this build, not installed here".

    Examples
    --------
    >>> from alkahest.crosscheck import oracles
    >>> sorted(oracles())
    ['sympy']
    """
    out: dict[str, str | None] = {}
    for name, cls in sorted(_ORACLE_CLASSES.items()):
        if not cls.available():
            out[name] = None
            continue
        try:
            out[name] = cls().version
        except Exception:  # pragma: no cover - a broken install is "absent"
            out[name] = None
    return out


def _resolve_oracle(oracle: Oracle | str | None) -> Oracle | None:
    """Return a usable oracle, or ``None`` when none is installed."""
    if isinstance(oracle, Oracle):
        return oracle
    names = [oracle] if isinstance(oracle, str) else sorted(_ORACLE_CLASSES)
    for name in names:
        cls = _ORACLE_CLASSES.get(name)
        if cls is None or not cls.available():
            continue
        try:
            return cls()
        except CrossCheckError:  # pragma: no cover - available() already checked
            continue
    return None


# ---------------------------------------------------------------------------
# Operation table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Operation:
    """How one operation is posed to both systems, and how it is compared.

    Attributes
    ----------
    name : str
        Matches the ``alkahest`` module attribute of the same name.
    rungs : tuple of int
        Ladder rungs to attempt, **in order**. Rung 4 leads wherever it exists,
        because it sidesteps equal-up-to-a-constant, up-to-ordering and
        up-to-a-unit entirely — the three things that make naive comparison of
        antiderivatives, solution sets and factorisations produce nothing but
        noise.
    invariant : str or None
        Name of the rung-4 invariant, or ``None`` when the operation has none.
    result : {"expr", "solutions"}
        Shape of the answer, which decides how it is compared and rendered.
    normalize : str or None
        Operation-specific normalisation applied to the Alkahest answer before
        the ladder runs — currently only ``"strip_big_o"``. This is *not* a
        rung: it removes a difference in how the two systems spell the same
        answer, and it must never remove a difference in the answer itself.
    """

    name: str
    rungs: tuple[int, ...]
    invariant: str | None
    result: str = "expr"
    normalize: str | None = None


#: Operations with a defined comparison rung. Anything else raises
#: ``E-XCHECK-003`` rather than being compared by a generic fallback: a generic
#: fallback is precisely how a harness starts reporting divergences that are
#: really just two systems using different normal forms.
OPERATIONS: dict[str, Operation] = {
    # d/dx has no invariant that is not itself the thing under test
    # (integrating back is weaker than the derivative it would check), so the
    # ladder runs 1 -> 2 -> 3.
    "diff": Operation("diff", (1, 2, 3), None),
    # Antiderivatives agree up to a constant; differentiating both removes the
    # constant and turns the comparison into an exact one.
    "integrate": Operation("integrate", (4, 1, 2, 3), "antiderivative"),
    # A simplifier's contract is that it preserves value. Checking each
    # system's output against the shared *input* compares the two claims
    # without ever asking whether two normal forms happen to coincide.
    "simplify": Operation("simplify", (4, 1, 2, 3), "value_preserving"),
    "simplify_expanded": Operation("simplify_expanded", (4, 1, 2, 3), "value_preserving"),
    "limit": Operation("limit", (1, 2, 3), None),
    # The two systems spell the remainder differently — Alkahest carries an
    # explicit O() term inside the expression, SymPy has removeO(). Stripping
    # it is a normalisation, not a rung: what is compared afterwards is still
    # the whole truncated polynomial, coefficient for coefficient.
    "series": Operation("series", (1, 2, 3), None, normalize="strip_big_o"),
    # Gosper: S(k+1) - S(k) = t(k) pins the antidifference up to a constant.
    "sum_indefinite": Operation("sum_indefinite", (4, 1, 2, 3), "antidifference"),
    # Substituting solutions back checks each system's answers on their own
    # terms; the set comparison then finds *missed* solutions, which is the
    # divergence worth having.
    "solve": Operation("solve", (4,), "substitution", result="solutions"),
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    """Two systems answered, and the answers are not the same.

    The wording is constrained on purpose. A divergence names **two suspects**:
    it is evidence that one of the two is wrong, and on its own it is never
    evidence about *which*. :attr:`support` carries whatever the rigorous
    escalation could establish, and its default is ``"unresolved"``.

    Attributes
    ----------
    operation : str
        Operation that diverged.
    oracle, oracle_version : str
        Which independent system, at which version. Without the version a
        recorded divergence rots the first time the oracle changes an answer.
    point : dict
        Witness: symbol name -> value at which the two disagree. Empty when
        the divergence is symbolic rather than pointwise.
    alkahest_value, oracle_value : str
        Both values, rendered. Never one without the other.
    alkahest_enclosure : tuple of float, optional
        Rigorous ball-arithmetic enclosure computed on the Alkahest side.
    support : {"unresolved", "alkahest_supported", "oracle_supported"}
        What the rigorous escalation established. ``"oracle_supported"`` means
        an operation invariant was **rigorously refuted on the Alkahest side**
        — a silent-error finding, and the case that should be routed into
        ``tests/silent_errors/corpus.py``.
    region : dict or None
        When :func:`alkahest.verified_sign` certified that the failing
        invariant residual keeps one sign across the whole sampling box, the
        box: ``{"x": [0.35, 2.65]}``. That upgrades a finding from "wrong at
        this point" to "wrong on this interval", which is a much shorter
        argument to hand a reviewer.
    detail : str
        One sentence, phrased symmetrically.
    """

    operation: str
    oracle: str
    oracle_version: str
    point: dict[str, float] = field(default_factory=dict)
    alkahest_value: str = ""
    oracle_value: str = ""
    alkahest_enclosure: tuple[float, float] | None = None
    support: str = "unresolved"
    region: dict[str, list[float]] | None = None
    detail: str = ""

    @property
    def silent_error_candidate(self) -> bool:
        """Is this a case for ``tests/silent_errors/corpus.py``?

        True exactly when rigorous ball arithmetic refuted the *Alkahest* side
        of an operation invariant. That is the finding worth converting into a
        permanent regression gate; everything else is a lead.
        """
        return self.support == "oracle_supported"

    def statement(self) -> str:
        """One-line rendering that attributes blame to neither system."""
        where = (
            " at " + ", ".join(f"{k}={v!r}" for k, v in sorted(self.point.items()))
            if self.point
            else ""
        )
        return (
            f"{self.operation}: alkahest and {self.oracle} {self.oracle_version} "
            f"disagree{where} — alkahest {self.alkahest_value!r} vs "
            f"{self.oracle} {self.oracle_value!r}; {self.detail}"
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "operation": self.operation,
            "oracle": self.oracle,
            "oracle_version": self.oracle_version,
            "point": dict(self.point),
            "alkahest_value": self.alkahest_value,
            "oracle_value": self.oracle_value,
            "alkahest_enclosure": list(self.alkahest_enclosure)
            if self.alkahest_enclosure
            else None,
            "support": self.support,
            "region": self.region,
            "silent_error_candidate": self.silent_error_candidate,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CrossCheck:
    """The outcome of one differential check.

    Attributes
    ----------
    operation : str
        Operation checked.
    outcome : {"agree", "diverge", "incomparable", "unavailable"}
        See the module docstring. Truthiness is deliberately **not** defined on
        this class: ``if check(...)`` would read as "it agreed" and would be
        true for three of the four outcomes under any obvious convention.
    rung : int or None
        Which rung settled it (see :data:`RUNG_NAMES`); ``None`` when nothing
        did.
    reason : str
        Stable short code for *why* — ``"invariant_holds"``,
        ``"no_oracle"``, ``"untranslatable"``, ``"alkahest_refused"``,
        ``"oracle_refused"``, ``"ladder_exhausted"``, and so on. Machine
        triage should switch on this, not on :attr:`detail`.
    detail : str
        One human sentence.
    oracle, oracle_version : str or None
        Which system answered, at which version.
    alkahest_value, oracle_value : str or None
        Both answers, rendered.
    witness : dict or None
        ``{"point", "alkahest", "oracle", "enclosure"}`` when a pointwise
        witness exists.
    divergence : Divergence or None
        Populated iff ``outcome == "diverge"``.
    conclusive : bool
        Whether the settling rung *proves* what it reports. Rung 3 agreement is
        sampled, so it is reported as agreement with ``conclusive=False``
        rather than being silently promoted or silently discarded.
    elapsed_ms : float
        Wall-clock cost of the check, oracle round trip included.
    """

    operation: str
    outcome: str
    rung: int | None = None
    reason: str = ""
    detail: str = ""
    oracle: str | None = None
    oracle_version: str | None = None
    alkahest_value: str | None = None
    oracle_value: str | None = None
    witness: dict[str, Any] | None = None
    divergence: Divergence | None = None
    conclusive: bool = False
    elapsed_ms: float = 0.0

    @property
    def rung_name(self) -> str | None:
        """Human name of :attr:`rung` (see :data:`RUNG_NAMES`)."""
        return RUNG_NAMES.get(self.rung) if self.rung is not None else None

    @property
    def checked(self) -> bool:
        """Did a comparison actually happen?

        ``False`` for ``incomparable`` and ``unavailable``. Provided so the
        common mistake — treating "no signal" as "clean" — has to be written
        out explicitly.
        """
        return self.outcome in ("agree", "diverge")

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "operation": self.operation,
            "outcome": self.outcome,
            "rung": self.rung,
            "rung_name": self.rung_name,
            "reason": self.reason,
            "detail": self.detail,
            "oracle": self.oracle,
            "oracle_version": self.oracle_version,
            "alkahest_value": self.alkahest_value,
            "oracle_value": self.oracle_value,
            "witness": self.witness,
            "divergence": self.divergence.as_dict() if self.divergence else None,
            "conclusive": self.conclusive,
        }

    def __repr__(self) -> str:
        return (
            f"CrossCheck({self.outcome!r}, operation={self.operation!r}, "
            f"rung={self.rung_name!r}, reason={self.reason!r})"
        )


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------


def _walk_expr(expr: Any) -> Iterator[Any]:
    """Yield *expr* and every subexpression, depth-first."""
    yield expr
    node = expr.node()
    tag = node[0]
    if tag in ("add", "mul"):
        for arg in node[1]:
            yield from _walk_expr(arg)
    elif tag == "func":
        for arg in node[2]:
            yield from _walk_expr(arg)
    elif tag == "pow":
        yield from _walk_expr(node[1])
        yield from _walk_expr(node[2])
    elif tag == "big_o":
        yield from _walk_expr(node[1])
    elif tag == "predicate":
        for arg in node[2]:
            yield from _walk_expr(arg)
    elif tag in ("forall", "exists"):
        yield from _walk_expr(node[1])
        yield from _walk_expr(node[2])
    elif tag == "piecewise":
        for cond, value in node[1]:
            yield from _walk_expr(cond)
            yield from _walk_expr(value)
        yield from _walk_expr(node[2])
    elif tag == "root_sum":
        yield from _walk_expr(node[1])
        yield from _walk_expr(node[2])
        yield from _walk_expr(node[3])


def _free_symbols(expr: Any) -> dict[str, Any]:
    """Free symbols of *expr*, excluding the reserved ``∞`` / ``I`` names."""
    out: dict[str, Any] = {}
    for sub in _walk_expr(expr):
        node = sub.node()
        if node[0] == "symbol":
            name = str(node[1])
            if name not in RESERVED_SYMBOLS:
                out[name] = sub
    return out


def _is_exact_zero(expr: Any) -> bool:
    node = expr.node()
    return node[0] == "integer" and int(node[1]) == 0


def _unwrap(value: Any) -> Any:
    """Reduce an Alkahest return value to the expression it carries.

    ``DerivedResult`` exposes ``.value``; ``Series`` exposes ``.expr``. Both
    are unwrapped, because the ladder compares expressions and a caller should
    not have to know which wrapper an operation happens to return.
    """
    if isinstance(value, (list, tuple, dict)) or hasattr(value, "node"):
        return value
    for attr in ("value", "expr"):
        inner = getattr(value, attr, None)
        if inner is not None and hasattr(inner, "node"):
            return inner
    return value


def _alkahest_is_zero(expr: Any) -> bool | None:
    """Try to prove ``expr == 0`` with Alkahest's own simplifiers."""
    ak = _ak()
    for route in (ak.simplify, ak.simplify_expanded):
        try:
            out = _unwrap(route(expr))
        except Exception:
            continue
        if hasattr(out, "node") and _is_exact_zero(out):
            return True
    return None


# ---------------------------------------------------------------------------
# Rigorous numeric machinery
# ---------------------------------------------------------------------------

#: Absolute slack allowed when comparing a rigorous Alkahest enclosure against
#: an oracle's floating value. The enclosure is rigorous; the oracle's number is
#: not, so the slack is entirely the oracle's error budget. Without it every
#: check would "diverge" on the oracle's last ulp — the exact false-alarm shape
#: that trains people to ignore the alarm.
ORACLE_SLACK_ABS = 1e-9
ORACLE_SLACK_REL = 1e-9

#: Default sample count and seed for the pointwise rungs. The seed is only used
#: when no :class:`alkahest.Budget` seed is active, so a sweep run under
#: ``context(budget=Budget(seed=...))`` is reproducible from that seed alone.
DEFAULT_POINTS = 5
DEFAULT_SEED = 20260810


def _seed(explicit: int | None = None) -> int:
    if explicit is not None:
        return explicit
    ak = _ak()
    try:
        active = ak.budget_seed()
    except Exception:  # pragma: no cover - native budget unavailable
        active = None
    return DEFAULT_SEED if active is None else int(active)


#: The band every sample point is drawn from, and the box the region-level
#: refutation is certified over. Positive and away from zero on purpose:
#: negative and near-zero arguments put ``log``, ``sqrt`` and negative powers on
#: or across a branch cut, where the two systems legitimately differ and the
#: resulting "divergence" would be about conventions rather than correctness.
SAMPLE_BOX = (0.35, 2.65)


def _sample_points(names: Sequence[str], rng: random.Random, count: int) -> list[dict[str, float]]:
    """Deterministic sample points inside :data:`SAMPLE_BOX`."""
    lo, hi = SAMPLE_BOX
    return [{name: round(rng.uniform(lo, hi), 6) for name in names} for _ in range(count)]


def _region_refutes_zero(expr: Any, symbols: Mapping[str, Any]) -> dict[str, list[float]] | None:
    """Certify that *expr* keeps one sign across :data:`SAMPLE_BOX`, or ``None``.

    :func:`alkahest.verified_sign` answers this over a whole box in one call,
    which is strictly more than the pointwise enclosure can say. It is used
    only to *strengthen* a refutation that ball arithmetic already made — never
    to create one — so the primary rung stays the one that also yields the
    enclosure a witness needs.
    """
    ak = _ak()
    if not symbols:
        return None
    lo, hi = SAMPLE_BOX
    box = [(symbol, lo, hi) for _name, symbol in sorted(symbols.items())]
    for predicate in ("positive", "negative"):
        try:
            verdict = ak.verified_sign(expr, box, predicate)
        except Exception:
            return None
        if verdict == "true":
            return {name: [lo, hi] for name in sorted(symbols)}
    return None


def _enclose(expr: Any, point: Mapping[str, float], symbols: Mapping[str, Any]) -> Any:
    """Rigorous ball enclosure of *expr* at *point*, or ``None``."""
    ak = _ak()
    try:
        bindings = {symbols[name]: ak.ArbBall(value, 0.0) for name, value in point.items()}
        return ak.interval_eval(expr, bindings)
    except Exception:
        return None


def _ball_bounds(ball: Any) -> tuple[float, float] | None:
    try:
        lo, hi = float(ball.lo), float(ball.hi)
    except Exception:  # pragma: no cover - defensive
        return None
    if lo != lo or hi != hi:
        return None
    return (lo, hi)


def _excludes(bounds: tuple[float, float], value: float) -> bool:
    """Does a rigorous enclosure exclude *value*, allowing the oracle its slack?"""
    lo, hi = bounds
    slack = ORACLE_SLACK_ABS + ORACLE_SLACK_REL * max(abs(lo), abs(hi), abs(value))
    return value < lo - slack or value > hi + slack


def _excludes_zero(bounds: tuple[float, float]) -> bool:
    """Rigorous: the enclosure is built entirely from Alkahest expressions."""
    lo, hi = bounds
    return lo > 0.0 or hi < 0.0


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------


def check(
    operation: str,
    *args: Any,
    oracle: Oracle | str | None = None,
    assumptions: Any = None,
    points: int = DEFAULT_POINTS,
    seed: int | None = None,
    pool: Any = None,
    **kwargs: Any,
) -> CrossCheck:
    """Run one query through Alkahest and through an oracle, and compare.

    Parameters
    ----------
    operation : str
        Name of an :data:`OPERATIONS` entry; also the ``alkahest`` module
        attribute that is called on the Alkahest side.
    *args
        The arguments you would pass to ``alkahest.<operation>``.
    oracle : Oracle or str, optional
        Which oracle to use. Defaults to the first installed one.
    assumptions : Assumptions, optional
        Governing assumption context; defaults to
        :func:`alkahest.active_assumptions`. An assumption with no faithful
        mapping into the oracle's language yields ``incomparable``.
    points : int
        Sample count for the rigorous-numeric rung.
    seed : int, optional
        Sampling seed. Defaults to the active :class:`alkahest.Budget` seed,
        then to :data:`DEFAULT_SEED`, so a check is reproducible.
    pool : ExprPool, optional
        Pool the arguments live in; defaults to :func:`alkahest.active_pool`.
        Only needed to attempt the **Alkahest side** of the symbolic rung — the
        oracle's answer has to be lifted back into a pool to be subtracted
        there. Without it the rung still runs, in the oracle only, and the
        record says which system proved what.
    **kwargs
        Forwarded to the Alkahest call.

    Returns
    -------
    CrossCheck
        Four-valued: ``agree`` / ``diverge`` / ``incomparable`` /
        ``unavailable``. **A missing oracle is ``unavailable``, never
        ``agree``.**

    Raises
    ------
    CrossCheckError
        ``E-XCHECK-003`` if *operation* has no defined comparison rung.
        Untranslatable input and missing oracles are *outcomes*, not
        exceptions; an unknown operation is a caller mistake.

    Examples
    --------
    >>> import alkahest as ak
    >>> from alkahest.crosscheck import check
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> out = check("diff", ak.sin(x), x)
    >>> out.outcome in {"agree", "unavailable"}
    True
    >>> out.outcome == "agree" or out.reason == "no_oracle"
    True
    """
    started = time.perf_counter()
    spec = OPERATIONS.get(operation)
    if spec is None:
        raise CrossCheckError(
            f"no comparison rung is defined for operation {operation!r}",
            code="E-XCHECK-003",
            remediation=(
                "Comparable operations: "
                + ", ".join(sorted(OPERATIONS))
                + ". A generic structural fallback is deliberately absent — it would "
                "report the two systems' differing normal forms as divergences."
            ),
        )

    resolved = _resolve_oracle(oracle)
    if resolved is None:
        return CrossCheck(
            operation=operation,
            outcome="unavailable",
            reason="no_oracle",
            detail=(
                "no cross-check oracle is installed, so nothing was compared; this is not agreement"
            ),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    try:
        return _compare(
            spec,
            resolved,
            args,
            kwargs,
            assumptions=assumptions,
            points=points,
            seed=seed,
            pool=pool,
            started=started,
        )
    except CrossCheckError as exc:
        reason = {
            "E-XCHECK-001": "untranslatable",
            "E-XCHECK-002": "no_oracle",
            # 003 is the caller error "this operation has no comparison rung",
            # raised before any oracle work; 004 is "the oracle declined". They
            # were one code, which meant a caller could not tell "I asked for
            # something unsupported" from "the oracle had nothing to say".
            "E-XCHECK-003": "unsupported_operation",
            "E-XCHECK-004": "oracle_refused",
        }.get(str(getattr(exc, "code", "")), "incomparable")
        return CrossCheck(
            operation=operation,
            outcome="unavailable" if reason == "no_oracle" else "incomparable",
            reason=reason,
            detail=str(exc),
            oracle=resolved.name,
            oracle_version=_safe_version(resolved),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


def _safe_version(oracle: Oracle) -> str | None:
    try:
        return oracle.version
    except Exception:  # pragma: no cover - defensive
        return None


def _compare(
    spec: Operation,
    oracle: Oracle,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    assumptions: Any,
    points: int,
    seed: int | None,
    pool: Any,
    started: float,
) -> CrossCheck:
    ak = _ak()
    version = _safe_version(oracle)

    def finish(**fields: Any) -> CrossCheck:
        fields.setdefault("oracle", oracle.name)
        fields.setdefault("oracle_version", version)
        fields["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return CrossCheck(operation=spec.name, **fields)

    if not oracle.supports(spec.name):
        raise _refuse(
            f"oracle {oracle.name!r} cannot answer {spec.name!r}",
            code="E-XCHECK-004",
            remediation="Use a different oracle, or a different operation.",
        )

    # --- pose the same question to both sides ------------------------------
    # Translation first: if the question cannot be posed identically there is
    # nothing to compare, and running the Alkahest side anyway would tempt a
    # caller to read the (uncompared) value as endorsed.
    oracle_args = _translate_args(spec, oracle, args, kwargs, assumptions=assumptions)

    try:
        alkahest_raw = getattr(ak, spec.name)(*args, **kwargs)
    except Exception as exc:
        return finish(
            outcome="incomparable",
            reason="alkahest_refused",
            detail=(
                f"alkahest declined {spec.name} ({type(exc).__name__}: {exc}); "
                "a refusal is not a divergence"
            ),
        )

    oracle_answer = oracle.run(spec.name, oracle_args)

    if spec.result == "solutions":
        return _compare_solutions(spec, oracle, alkahest_raw, oracle_answer, oracle_args, finish)

    alkahest_answer = _unwrap(alkahest_raw)
    if not hasattr(alkahest_answer, "node"):
        return finish(
            outcome="incomparable",
            reason="unsupported_result_shape",
            detail=(
                f"alkahest returned {type(alkahest_answer).__name__}, which has no comparison rung"
            ),
        )

    if spec.normalize == "strip_big_o":
        stripped = _strip_big_o(alkahest_answer)
        if stripped is None:
            return finish(
                outcome="incomparable",
                reason="empty_after_normalisation",
                detail="the alkahest series has no terms below its O() remainder",
            )
        alkahest_answer = stripped

    ctx = _Comparison(
        spec=spec,
        oracle=oracle,
        alkahest_answer=alkahest_answer,
        oracle_answer=oracle_answer,
        oracle_args=oracle_args,
        source_args=args,
        assumptions=assumptions,
        points=points,
        rng=random.Random(_seed(seed)),
        version=version or "unknown",
        pool=pool if pool is not None else _active_pool(),
    )

    for rung in spec.rungs:
        verdict = ctx.attempt(rung)
        if verdict is not None:
            return finish(**verdict)

    return finish(
        outcome="incomparable",
        reason="ladder_exhausted",
        alkahest_value=str(alkahest_answer),
        oracle_value=oracle.render(oracle_answer),
        detail=(
            "no rung settled it: the two forms are not syntactically equal, neither "
            "system proved the difference zero, and no point could be evaluated "
            "rigorously — reported as incomparable rather than agreement"
        ),
    )


def _translate_args(
    spec: Operation,
    oracle: Oracle,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    assumptions: Any,
) -> dict[str, Any]:
    """Build the oracle-side argument bundle, refusing anything untranslatable."""

    def tr(expr: Any) -> Any:
        return oracle.translate(expr, assumptions=assumptions)

    if spec.name == "solve":
        equations, variables = args[0], args[1]
        return {
            "equations": [tr(e) for e in equations],
            "vars": [tr(v) for v in variables],
        }

    out: dict[str, Any] = {"expr": tr(args[0])}
    if spec.name in ("simplify", "simplify_expanded"):
        return out
    if len(args) > 1:
        out["var"] = tr(args[1])
    if spec.name == "integrate":
        out["lower"] = tr(args[2]) if len(args) > 2 else None
        out["upper"] = tr(args[3]) if len(args) > 3 else None
    elif spec.name == "limit":
        out["point"] = tr(args[2]) if len(args) > 2 else None
    elif spec.name == "series":
        out["point"] = tr(args[2]) if len(args) > 2 else None
        out["order"] = int(args[3]) if len(args) > 3 else int(kwargs.get("order", 6))
    return out


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResidualVerdict:
    """One side's answer to "does this invariant residual vanish?".

    ``proved`` is separate from ``status`` on purpose: "no sampled point
    refuted it" and "a normaliser proved it zero" are both ``"holds"``, and
    reporting the first as though it were the second is precisely how a
    differential harness starts producing false confidence.
    """

    status: str
    proved: bool
    via: str
    point: dict[str, float] | None = None
    bounds: tuple[float, float] | None = None
    region: dict[str, list[float]] | None = None


@dataclass
class _Comparison:
    """Per-check state shared by the rungs."""

    spec: Operation
    oracle: Oracle
    alkahest_answer: Any
    oracle_answer: Any
    oracle_args: Mapping[str, Any]
    source_args: Sequence[Any]
    assumptions: Any
    points: int
    rng: random.Random
    version: str
    pool: Any = None

    def attempt(self, rung: int) -> dict[str, Any] | None:
        """Run one rung; ``None`` means "inconclusive, try the next"."""
        return {
            RUNG_SYNTACTIC: self._syntactic,
            RUNG_SYMBOLIC: self._symbolic,
            RUNG_NUMERIC: self._numeric,
            RUNG_INVARIANT: self._invariant,
        }[rung]()

    # -- shared -------------------------------------------------------------

    @property
    def may_refute(self) -> bool:
        """May the answer-comparing rungs (1–3) report a *divergence*?

        Only when the operation's answer is pinned exactly. Where an invariant
        exists, the operation leaves freedom the answers are allowed to differ
        by — ``+ C`` for an antiderivative, the additive constant in a Gosper
        antidifference — so "the two answers are not equal" is not evidence of
        disagreement, and reporting it as one would make the harness fire on
        every correctly-integrated example that happened to reach these rungs.

        Rung 4 runs first precisely so it settles those cases. When it comes
        back inconclusive, rungs 1–3 may still *confirm* agreement (equal
        answers agree under any freedom) but must fall through rather than
        refute.
        """
        return self.spec.invariant is None

    @property
    def rendered(self) -> tuple[str, str]:
        return str(self.alkahest_answer), self.oracle.render(self.oracle_answer)

    def _translated_answer(self) -> Any | None:
        try:
            return self.oracle.translate(self.alkahest_answer, assumptions=self.assumptions)
        except CrossCheckError:
            return None

    # -- rung 1 -------------------------------------------------------------

    def _syntactic(self) -> dict[str, Any] | None:
        translated = self._translated_answer()
        if translated is None:
            return None
        try:
            same = self.oracle.canonical(translated) == self.oracle.canonical(self.oracle_answer)
        except Exception:
            return None
        if not same:
            return None
        ak_value, oracle_value = self.rendered
        return {
            "outcome": "agree",
            "rung": RUNG_SYNTACTIC,
            "reason": "identical_canonical_form",
            "detail": "both systems produced the same canonical form",
            "alkahest_value": ak_value,
            "oracle_value": oracle_value,
            "conclusive": True,
        }

    # -- rung 2 -------------------------------------------------------------

    def _symbolic(self) -> dict[str, Any] | None:
        """``simplify(a - b) == 0``, attempted independently in both systems.

        Either system proving it counts as agreement — the point of running two
        implementations is that they fail to normalise different things.
        """
        ak_value, oracle_value = self.rendered
        prover: str | None = None

        translated = self._translated_answer()
        if translated is not None:
            verdict = self.oracle.is_zero(translated - self.oracle_answer)
            if verdict is True:
                prover = self.oracle.name
            elif verdict is False:
                if not self.may_refute:
                    return None
                return self._symbolic_divergence(ak_value, oracle_value)

        if prover is None:
            lifted = self._lift_oracle_answer()
            if lifted is not None and _alkahest_is_zero(self.alkahest_answer - lifted):
                prover = "alkahest"

        if prover is None:
            return None
        return {
            "outcome": "agree",
            "rung": RUNG_SYMBOLIC,
            "reason": "difference_proved_zero",
            "detail": f"{prover} proved the difference of the two answers is identically zero",
            "alkahest_value": ak_value,
            "oracle_value": oracle_value,
            "conclusive": True,
        }

    def _symbolic_divergence(self, ak_value: str, oracle_value: str) -> dict[str, Any]:
        divergence = Divergence(
            operation=self.spec.name,
            oracle=self.oracle.name,
            oracle_version=self.version,
            alkahest_value=ak_value,
            oracle_value=oracle_value,
            support="unresolved",
            detail=(
                f"{self.oracle.name} determined the difference of the two answers is "
                "not identically zero; which of the two is wrong is not established here"
            ),
        )
        return {
            "outcome": "diverge",
            "rung": RUNG_SYMBOLIC,
            "reason": "difference_proved_nonzero",
            "detail": divergence.detail,
            "alkahest_value": ak_value,
            "oracle_value": oracle_value,
            "divergence": divergence,
            "conclusive": True,
        }

    def _lift_oracle_answer(self) -> Any | None:
        if self.pool is None:
            return None
        try:
            return self.oracle.lift(self.oracle_answer, self.pool)
        except Exception:
            return None

    # -- rung 3 -------------------------------------------------------------

    def _numeric(self) -> dict[str, Any] | None:
        """Sample and evaluate with rigorous ball arithmetic.

        The asymmetry is the point: a ball is a *rigorous* enclosure of the
        Alkahest answer, so a value outside it is a real disagreement rather
        than float noise — while agreement here is only ever "not refuted at
        the points sampled", which is why :attr:`CrossCheck.conclusive` is
        ``False`` for this rung.
        """
        symbols = _free_symbols(self.alkahest_answer)
        ak_value, oracle_value = self.rendered
        checked = 0
        for point in _sample_points(sorted(symbols), self.rng, self.points):
            ball = _enclose(self.alkahest_answer, point, symbols)
            bounds = _ball_bounds(ball) if ball is not None else None
            if bounds is None:
                continue
            value = self._oracle_at(point)
            if value is None:
                continue
            checked += 1
            if _excludes(bounds, value):
                if not self.may_refute:
                    # The answers differ at this point, but the operation lets
                    # them (see `may_refute`). Not agreement either — fall
                    # through to `incomparable` rather than call it clean.
                    return None
                divergence = Divergence(
                    operation=self.spec.name,
                    oracle=self.oracle.name,
                    oracle_version=self.version,
                    point=dict(point),
                    alkahest_value=f"{ak_value} = {bounds[0]!r}..{bounds[1]!r}",
                    oracle_value=f"{oracle_value} = {value!r}",
                    alkahest_enclosure=bounds,
                    support="unresolved",
                    detail=(
                        "the rigorous enclosure of the alkahest answer excludes the "
                        f"{self.oracle.name} value at this point; one of the two is "
                        "wrong and this rung does not say which"
                    ),
                )
                return {
                    "outcome": "diverge",
                    "rung": RUNG_NUMERIC,
                    "reason": "enclosure_excludes_oracle_value",
                    "detail": divergence.detail,
                    "alkahest_value": ak_value,
                    "oracle_value": oracle_value,
                    "witness": _witness(divergence),
                    "divergence": divergence,
                    "conclusive": True,
                }
        if checked == 0:
            return None
        return {
            "outcome": "agree",
            "rung": RUNG_NUMERIC,
            "reason": "not_refuted_numerically",
            "detail": (
                f"rigorous enclosures at {checked} sampled point(s) contain the "
                f"{self.oracle.name} value; sampling cannot prove agreement, only fail "
                "to refute it"
            ),
            "alkahest_value": ak_value,
            "oracle_value": oracle_value,
            "conclusive": False,
        }

    def _oracle_at(self, point: Mapping[str, float]) -> float | None:
        try:
            substituted = self.oracle.subs_by_name(self.oracle_answer, point)
        except Exception:
            return None
        return self.oracle.to_float(substituted)

    # -- rung 4 -------------------------------------------------------------

    def _invariant(self) -> dict[str, Any] | None:
        builder = _INVARIANTS.get(self.spec.invariant or "")
        if builder is None:
            return None
        try:
            residuals = builder(self)
        except CrossCheckError:
            raise
        except Exception:
            return None
        if residuals is None:
            return None
        ak_residual, oracle_residual = residuals

        ak_side = self._residual_verdict(ak_residual)
        oracle_side = self._oracle_residual_verdict(oracle_residual)
        ak_verdict, oracle_verdict = ak_side.status, oracle_side.status
        ak_point, ak_bounds, ak_region = ak_side.point, ak_side.bounds, ak_side.region

        ak_value, oracle_value = self.rendered
        if ak_verdict == "holds" and oracle_verdict == "holds":
            proved = ak_side.proved and oracle_side.proved
            return {
                "outcome": "agree",
                "rung": RUNG_INVARIANT,
                "reason": "invariant_holds" if proved else "invariant_not_refuted",
                "detail": (
                    f"both answers satisfy the {self.spec.invariant} invariant "
                    f"({ak_side.via} / {oracle_side.via}), so they agree up to the "
                    "freedom the operation leaves"
                    if proved
                    else (
                        f"neither answer's {self.spec.invariant} residual could be "
                        f"refuted ({ak_side.via} / {oracle_side.via}), but neither was "
                        "proved zero either — this is sampled evidence, not a proof"
                    )
                ),
                "alkahest_value": ak_value,
                "oracle_value": oracle_value,
                "conclusive": proved,
            }
        if ak_verdict == "refuted" or oracle_verdict == "refuted":
            support = {
                ("refuted", "holds"): "oracle_supported",
                ("holds", "refuted"): "alkahest_supported",
            }.get((ak_verdict, oracle_verdict), "unresolved")
            detail = {
                "oracle_supported": (
                    f"the alkahest answer fails the {self.spec.invariant} invariant under "
                    f"rigorous ball arithmetic while the {self.oracle.name} answer satisfies "
                    "it — a silent-error candidate for tests/silent_errors/corpus.py"
                ),
                "alkahest_supported": (
                    f"the {self.oracle.name} answer fails the {self.spec.invariant} "
                    "invariant while the alkahest answer satisfies it"
                ),
                "unresolved": (
                    f"at least one answer fails the {self.spec.invariant} invariant; the "
                    "evidence does not establish which system is at fault"
                ),
            }[support]
            if ak_region is not None:
                detail += (
                    "; the alkahest residual is certified to keep one sign across "
                    f"{SAMPLE_BOX[0]}..{SAMPLE_BOX[1]}, so the failure is not pointwise"
                )
            divergence = Divergence(
                operation=self.spec.name,
                oracle=self.oracle.name,
                oracle_version=self.version,
                point=dict(ak_point or {}),
                alkahest_value=ak_value,
                oracle_value=oracle_value,
                alkahest_enclosure=ak_bounds,
                support=support,
                region=ak_region,
                detail=detail,
            )
            return {
                "outcome": "diverge",
                "rung": RUNG_INVARIANT,
                "reason": f"invariant_failed_{support}",
                "detail": detail,
                "alkahest_value": ak_value,
                "oracle_value": oracle_value,
                "witness": _witness(divergence),
                "divergence": divergence,
                "conclusive": True,
            }
        return None

    def _residual_verdict(self, residual: Any) -> _ResidualVerdict:
        """Does the Alkahest-side invariant residual vanish?

        Three routes, tried in order of what they license:

        1. Alkahest's own simplifiers prove it zero — a proof, and the one
           route with no second system in it.
        2. Rigorous ball arithmetic **excludes** zero at a sampled point — also
           a proof, of the opposite. This residual is built purely from
           Alkahest expressions, so there is no oracle float in it to blame:
           a refutation here says the Alkahest answer is wrong, full stop.
        3. The oracle's normaliser is asked about the *translated* residual.
           This still cross-checks Alkahest's engine — the residual came out of
           Alkahest's own ``diff``/``subs`` — it just borrows a second
           normaliser to decide it, which is the whole point of having one.

        Falling through all three leaves sampled non-refutation, reported with
        ``proved=False`` so rung-4 agreement built on it is not dressed up as a
        proof.
        """
        if residual is None:
            return _ResidualVerdict("unknown", False, "no residual available")
        if _alkahest_is_zero(residual):
            return _ResidualVerdict("holds", True, "alkahest proved the residual zero")
        symbols = _free_symbols(residual)
        checked = 0
        for point in _sample_points(sorted(symbols), self.rng, self.points):
            ball = _enclose(residual, point, symbols)
            bounds = _ball_bounds(ball) if ball is not None else None
            if bounds is None:
                continue
            checked += 1
            if _excludes_zero(bounds):
                return _ResidualVerdict(
                    "refuted",
                    True,
                    "a rigorous enclosure of the alkahest residual excludes zero",
                    point=dict(point),
                    bounds=bounds,
                    region=_region_refutes_zero(residual, symbols),
                )
        try:
            translated = self.oracle.translate(residual, assumptions=self.assumptions)
        except CrossCheckError:
            translated = None
        if translated is not None:
            verdict = self.oracle.is_zero(translated)
            if verdict is True:
                return _ResidualVerdict(
                    "holds", True, f"{self.oracle.name} normalised the alkahest residual to zero"
                )
            if verdict is False and checked == 0:
                # No rigorous enclosure was available, so this is the oracle's
                # word alone; recorded as a refutation but not as a proof.
                return _ResidualVerdict(
                    "refuted",
                    False,
                    f"{self.oracle.name} determined the alkahest residual is non-zero",
                )
        if checked:
            return _ResidualVerdict(
                "holds", False, f"zero is inside every enclosure at {checked} sampled point(s)"
            )
        return _ResidualVerdict("unknown", False, "the residual could not be decided or evaluated")

    def _oracle_residual_verdict(self, residual: Any) -> _ResidualVerdict:
        if residual is None:
            return _ResidualVerdict("unknown", False, "no residual available")
        verdict = self.oracle.is_zero(residual)
        if verdict is True:
            return _ResidualVerdict(
                "holds", True, f"{self.oracle.name} proved its own residual zero"
            )
        if verdict is False:
            return _ResidualVerdict(
                "refuted", True, f"{self.oracle.name} determined its own residual is non-zero"
            )
        return _ResidualVerdict("unknown", False, f"{self.oracle.name} could not decide")


def _witness(divergence: Divergence) -> dict[str, Any]:
    return {
        "point": dict(divergence.point),
        "alkahest": divergence.alkahest_value,
        "oracle": divergence.oracle_value,
        "enclosure": list(divergence.alkahest_enclosure) if divergence.alkahest_enclosure else None,
    }


def _active_pool() -> Any:
    """The ambient pool, or ``None``.

    :class:`alkahest.Expr` deliberately does not expose the pool it was interned
    in, so the only routes to one are the caller's ``pool=`` argument and the
    ambient context. When neither is available the pool-dependent half of the
    ladder is skipped rather than guessed at — mixing pools raises ``E-POOL-*``,
    and a check must not fail for a reason that has nothing to do with the
    mathematics.
    """
    ak = _ak()
    try:
        return ak.active_pool()
    except Exception:  # pragma: no cover - defensive
        return None


# -- the rung-4 invariants ---------------------------------------------------
#
# Each returns ``(alkahest_residual_expr | None, oracle_residual_obj | None)``;
# both must vanish. Returning ``None`` on a side means "this side could not be
# checked", which makes the rung inconclusive rather than a divergence.


def _inv_antiderivative(ctx: _Comparison) -> tuple[Any, Any] | None:
    """``d/dx F - f == 0`` — the check that ignores ``+ C``."""
    ak = _ak()
    integrand, var = ctx.source_args[0], ctx.source_args[1]
    if len(ctx.source_args) > 2:  # definite integral: no antiderivative to check
        return None
    ak_residual = _unwrap(ak.diff(ctx.alkahest_answer, var)) - integrand
    oracle_residual = (
        ctx.oracle.diff(ctx.oracle_answer, ctx.oracle_args["var"]) - ctx.oracle_args["expr"]
    )
    return (ak_residual, oracle_residual)


def _inv_value_preserving(ctx: _Comparison) -> tuple[Any, Any] | None:
    """``simplified - original == 0`` — a simplifier's whole contract."""
    original = ctx.source_args[0]
    ak_residual = ctx.alkahest_answer - original
    oracle_residual = ctx.oracle_answer - ctx.oracle_args["expr"]
    return (ak_residual, oracle_residual)


def _inv_antidifference(ctx: _Comparison) -> tuple[Any, Any] | None:
    """``S(k+1) - S(k) - t(k) == 0`` — Gosper's defining property."""
    ak = _ak()
    term, var = ctx.source_args[0], ctx.source_args[1]
    shifted = _unwrap(ak.subs(ctx.alkahest_answer, {var: var + 1}))
    ak_residual = shifted - ctx.alkahest_answer - term
    ovar = ctx.oracle_args["var"]
    oracle_residual = (
        ctx.oracle.subs(ctx.oracle_answer, {ovar: ovar + 1})
        - ctx.oracle_answer
        - ctx.oracle_args["expr"]
    )
    return (ak_residual, oracle_residual)


_INVARIANTS: dict[str, Callable[[_Comparison], tuple[Any, Any] | None]] = {
    "antiderivative": _inv_antiderivative,
    "value_preserving": _inv_value_preserving,
    "antidifference": _inv_antidifference,
}


def _strip_big_o(expr: Any) -> Any | None:
    """Drop top-level ``O(...)`` terms from a series expression."""
    node = expr.node()
    if node[0] == "big_o":
        return None
    if node[0] != "add":
        return expr
    kept = [term for term in node[1] if term.node()[0] != "big_o"]
    if not kept:
        return None
    out = kept[0]
    for term in kept[1:]:
        out = out + term
    return out


# ---------------------------------------------------------------------------
# solve: substitute both solution sets back
# ---------------------------------------------------------------------------


def _compare_solutions(
    spec: Operation,
    oracle: Oracle,
    alkahest_raw: Any,
    oracle_answer: Any,
    oracle_args: Mapping[str, Any],
    finish: Callable[..., CrossCheck],
) -> CrossCheck:
    """Rung 4 for ``solve``: verify both sides, then compare the sets.

    Solution *sets* compare badly by construction — ordering, radical form, and
    which branch a root is written in all differ. Substituting back checks each
    system's answers on their own terms, and only then does the set comparison
    mean something: with both sides verified, a size difference is a genuinely
    **missed solution**, not a formatting artefact.
    """
    if not isinstance(alkahest_raw, list):
        return finish(
            outcome="incomparable",
            reason="unsupported_result_shape",
            detail=f"alkahest.solve returned {type(alkahest_raw).__name__}",
        )

    equations = list(oracle_args["equations"])
    ak_sets: list[frozenset[tuple[str, str]]] = []
    for solution in alkahest_raw:
        bindings = {oracle.translate(k): oracle.translate(v) for k, v in solution.items()}
        translated = {str(k): oracle.translate(v) for k, v in solution.items()}
        for equation in equations:
            residual = oracle.subs(equation, bindings)
            if oracle.is_zero(residual) is False:
                divergence = Divergence(
                    operation=spec.name,
                    oracle=oracle.name,
                    oracle_version=_safe_version(oracle) or "unknown",
                    point={str(k): 0.0 for k in solution},
                    alkahest_value=_render_solutions(alkahest_raw),
                    oracle_value=_render_oracle_solutions(oracle, oracle_answer),
                    support="oracle_supported",
                    detail=(
                        f"an alkahest solution does not satisfy the system when "
                        f"substituted back ({oracle.name} evaluated the residual as "
                        "non-zero) — a silent-error candidate"
                    ),
                )
                return finish(
                    outcome="diverge",
                    rung=RUNG_INVARIANT,
                    reason="invariant_failed_oracle_supported",
                    detail=divergence.detail,
                    alkahest_value=divergence.alkahest_value,
                    oracle_value=divergence.oracle_value,
                    witness=_witness(divergence),
                    divergence=divergence,
                    conclusive=True,
                )
        ak_sets.append(frozenset((k, oracle.canonical(v)) for k, v in translated.items()))

    oracle_sets = [
        frozenset((k, oracle.canonical(v)) for k, v in solution.items())
        for solution in oracle_answer
    ]

    if len(ak_sets) != len(oracle_sets):
        divergence = Divergence(
            operation=spec.name,
            oracle=oracle.name,
            oracle_version=_safe_version(oracle) or "unknown",
            alkahest_value=_render_solutions(alkahest_raw),
            oracle_value=_render_oracle_solutions(oracle, oracle_answer),
            support="unresolved",
            detail=(
                f"alkahest returned {len(ak_sets)} solution(s) and {oracle.name} returned "
                f"{len(oracle_sets)}; every alkahest solution verified, so one of the two "
                "systems has a different solution *set*, and this does not say which is "
                "complete"
            ),
        )
        return finish(
            outcome="diverge",
            rung=RUNG_INVARIANT,
            reason="solution_set_size_differs",
            detail=divergence.detail,
            alkahest_value=divergence.alkahest_value,
            oracle_value=divergence.oracle_value,
            divergence=divergence,
            conclusive=True,
        )

    return finish(
        outcome="agree",
        rung=RUNG_INVARIANT,
        reason="invariant_holds",
        detail=(
            "every alkahest solution satisfies the system when substituted back, and both "
            "systems returned the same number of solutions"
        ),
        alkahest_value=_render_solutions(alkahest_raw),
        oracle_value=_render_oracle_solutions(oracle, oracle_answer),
        conclusive=True,
    )


def _render_solutions(solutions: Sequence[Mapping[Any, Any]]) -> str:
    rendered = [
        "{" + ", ".join(f"{k}: {v}" for k, v in sorted(s.items(), key=lambda kv: str(kv[0]))) + "}"
        for s in solutions
    ]
    return "[" + ", ".join(rendered) + "]"


def _render_oracle_solutions(oracle: Oracle, solutions: Sequence[Mapping[str, Any]]) -> str:
    rendered = [
        "{" + ", ".join(f"{k}: {oracle.render(v)}" for k, v in sorted(s.items())) + "}"
        for s in solutions
    ]
    return "[" + ", ".join(rendered) + "]"


# ---------------------------------------------------------------------------
# Tier 1: the seeded sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepReport:
    """Result of a seeded differential sweep.

    Attributes
    ----------
    seed : int
        **Print this.** A sweep is only useful as a bug report if the run that
        found something can be reproduced exactly.
    oracle, oracle_version : str or None
        Which system, at which version. A sweep that turns red after a SymPy
        upgrade is a different event from one that turns red after an Alkahest
        change, and only the recorded version tells them apart.
    checks : tuple of CrossCheck
        Every check run, in order.
    """

    seed: int
    oracle: str | None
    oracle_version: str | None
    checks: tuple[CrossCheck, ...]

    @property
    def findings(self) -> tuple[CrossCheck, ...]:
        """Checks whose outcome was ``diverge``."""
        return tuple(c for c in self.checks if c.outcome == "diverge")

    @property
    def silent_error_candidates(self) -> tuple[CrossCheck, ...]:
        """Findings where Alkahest's side was rigorously refuted."""
        return tuple(
            c for c in self.findings if c.divergence and c.divergence.silent_error_candidate
        )

    def counts(self) -> dict[str, int]:
        """Outcome histogram, total over :data:`OUTCOMES`."""
        out = dict.fromkeys(OUTCOMES, 0)
        for c in self.checks:
            out[c.outcome] = out.get(c.outcome, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, suitable for filing as a CI artifact."""
        return {
            "seed": self.seed,
            "oracle": self.oracle,
            "oracle_version": self.oracle_version,
            "counts": self.counts(),
            "checks": [c.as_dict() for c in self.checks],
        }

    def summary(self) -> str:
        """One-screen report; print it from the nightly job."""
        counts = self.counts()
        lines = [
            f"crosscheck sweep: seed={self.seed} oracle={self.oracle} "
            f"version={self.oracle_version}",
            "  " + "  ".join(f"{k}={counts[k]}" for k in OUTCOMES),
        ]
        for finding in self.findings:
            lines.append(f"  DIVERGE {finding.operation}: {finding.detail}")
        return "\n".join(lines)


#: Operations the default sweep exercises. Chosen so the *comparator* is what
#: gets stressed — every one of these has a rung 4 or a rigorous rung 3.
#:
#: ``limit`` is deliberately absent. The original reason — the call could run
#: away and the kernel held the GIL throughout, so nothing in-process could
#: bound it — no longer holds: ``limit`` now has cooperative checkpoints and an
#: internal work ceiling, and its binding releases the GIL, so
#: ``context(budget=...)`` and ``request_cancel()`` both reach it. What is left
#: is that its comparator is weaker than the three below, which is a reason to
#: promote findings by hand rather than to sweep it randomly.
SWEEP_OPERATIONS = ("diff", "integrate", "simplify")


def sweep(
    *,
    seed: int | None = None,
    cases: int = 40,
    operations: Sequence[str] = SWEEP_OPERATIONS,
    oracle: Oracle | str | None = None,
    pool: Any = None,
    points: int = DEFAULT_POINTS,
) -> SweepReport:
    """Run a seeded differential sweep and report what diverged.

    This is the **nightly** tier of the two-tier arrangement described in
    ``docs/mdbook/src/crosscheck.md``. It is deliberately not a per-PR gate: it
    is randomised, and an oracle upgrade would turn it red for reasons that have
    nothing to do with the pull request under review. Its findings are meant to
    be promoted into :data:`FROZEN_CORPUS`, which *is* the per-PR gate.

    .. warning::

       **Neither side is bounded, and this module does not pretend otherwise.**
       Alkahest's heavy engines hold the GIL, so a non-terminating call cannot
       be timed out from Python: a worker thread cannot be stopped, and
       abandoning one wedges the interpreter anyway. SymPy is no better placed.
       Run the nightly job under an OS-level timeout, and wrap the sweep in
       ``context(budget=...)`` for the engines that *are* cooperative
       (:func:`alkahest.integrate`, best-effort :func:`alkahest.simplify` — see
       ``docs/mdbook/src/budgets.md``). :data:`SWEEP_OPERATIONS` is chosen to
       stay clear of the paths known not to terminate.

    Parameters
    ----------
    seed : int, optional
        Defaults to the active :class:`alkahest.Budget` seed, then to
        :data:`DEFAULT_SEED`. Always recorded on the report.
    cases : int
        Number of expressions to generate.
    operations : sequence of str
        Which :data:`OPERATIONS` entries to exercise.
    oracle : Oracle or str, optional
        Defaults to the first installed oracle. With none installed the report
        is all ``unavailable`` — the sweep does not pretend to have run.
    pool : ExprPool, optional
        Pool to build candidates in; a fresh one by default.
    points : int
        Sample count passed to :func:`check`.
    Returns
    -------
    SweepReport
    """
    ak = _ak()
    resolved_seed = _seed(seed)
    rng = random.Random(resolved_seed)
    pool = pool if pool is not None else ak.ExprPool()
    x = pool.symbol("x")
    resolved = _resolve_oracle(oracle)

    checks: list[CrossCheck] = []
    for index in range(cases):
        operation = operations[index % len(operations)]
        expr = _random_expr(pool, x, rng)
        args = (expr,) if operation == "simplify" else (expr, x)
        checks.append(
            check(
                operation,
                *args,
                oracle=resolved,
                points=points,
                seed=resolved_seed + index,
                pool=pool,
            )
        )
    return SweepReport(
        seed=resolved_seed,
        oracle=resolved.name if resolved else None,
        oracle_version=_safe_version(resolved) if resolved else None,
        checks=tuple(checks),
    )


def _random_expr(pool: Any, x: Any, rng: random.Random, depth: int = 0) -> Any:
    """A small deterministic grammar of expressions both systems should handle.

    Kept simple on purpose. The job of the sweep is to exercise the
    *comparator* — translation, the ladder, the witness machinery — not to find
    the hardest integrand in the world; a corpus tuned to whatever the current
    build happens to be good at measures the corpus, not the build.
    """
    ak = _ak()
    if depth >= 2 or rng.random() < 0.45:
        choice = rng.randrange(6)
        if choice == 0:
            return x ** pool.integer(rng.randint(1, 4))
        if choice == 1:
            return ak.sin(x)
        if choice == 2:
            return ak.cos(x)
        if choice == 3:
            return ak.exp(x)
        if choice == 4:
            return pool.integer(rng.randint(1, 5)) * x
        return pool.integer(rng.randint(-4, 4))
    left = _random_expr(pool, x, rng, depth + 1)
    right = _random_expr(pool, x, rng, depth + 1)
    return left + right if rng.random() < 0.6 else left * right


# ---------------------------------------------------------------------------
# Tier 2: the frozen corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenCase:
    """One previously-observed comparison, pinned to an oracle version range.

    The per-PR half of the two-tier arrangement. Every case records the oracle
    version range its expectation was established against: without that the
    corpus rots silently the first time the oracle changes an answer, which it
    will, and a red gate would then be indistinguishable from a real regression.

    Attributes
    ----------
    id : str
        Stable, never reused.
    operation : str
        An :data:`OPERATIONS` key.
    build : callable
        ``pool -> tuple`` of arguments for :func:`check`. Takes the pool so each
        run gets a fresh one.
    expected : str
        Expected :attr:`CrossCheck.outcome`.
    expected_reason : str or None
        Expected :attr:`CrossCheck.reason`, when the case is pinning *why*.
    oracle : str
        Oracle name the expectation belongs to.
    oracle_versions : str
        Comma-separated PEP-440-ish range, e.g. ``">=1.12,<2"``. A case whose
        range excludes the installed version is skipped, not failed.
    found_by : str
        ``"seeded sweep, seed=..."`` or a hand-written provenance note. The
        ratchet: a divergence the nightly finds must land here.
    note : str
        What this case is protecting.
    """

    id: str
    operation: str
    build: Callable[[Any], tuple[Any, ...]]
    expected: str
    oracle: str = "sympy"
    oracle_versions: str = ">=1.12"
    found_by: str = ""
    note: str = ""
    expected_reason: str | None = None

    def applies_to(self, version: str) -> bool:
        """Does this case's expectation cover *version* of its oracle?"""
        return _version_in_range(version, self.oracle_versions)


def _parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(text).split("."):
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _version_in_range(version: str, spec: str) -> bool:
    """Tiny comparator for ``">=1.12,<2"``-style ranges.

    Deliberately dependency-free: pulling ``packaging`` in for five operators
    would put a new install requirement on a module whose entire point is to
    work in whatever environment the loop happens to have.
    """
    actual = _parse_version(version)
    for clause in (c.strip() for c in spec.split(",") if c.strip()):
        for operator in ("!=", ">=", "<=", "==", ">", "<"):
            if clause.startswith(operator):
                bound = _parse_version(clause[len(operator) :])
                width = max(len(actual), len(bound))
                left = actual + (0,) * (width - len(actual))
                right = bound + (0,) * (width - len(bound))
                ok = {
                    ">=": left >= right,
                    "<=": left <= right,
                    "==": left == right,
                    "!=": left != right,
                    ">": left > right,
                    "<": left < right,
                }[operator]
                if not ok:
                    return False
                break
        else:  # pragma: no cover - malformed spec
            raise ValueError(f"unparseable version clause {clause!r}")
    return True


def _case_polynomial_antiderivative(pool: Any) -> tuple[Any, ...]:
    x = pool.symbol("x")
    return (x ** pool.integer(3) + pool.integer(2) * x, x)


def _case_trig_antiderivative(pool: Any) -> tuple[Any, ...]:
    ak = _ak()
    x = pool.symbol("x")
    return (ak.sin(x) * ak.cos(x), x)


def _case_chain_rule(pool: Any) -> tuple[Any, ...]:
    ak = _ak()
    x = pool.symbol("x")
    return (ak.sin(ak.exp(x)), x)


def _case_pythagorean_simplify(pool: Any) -> tuple[Any, ...]:
    ak = _ak()
    x = pool.symbol("x")
    return (ak.sin(x) ** pool.integer(2) + ak.cos(x) ** pool.integer(2),)


def _case_gosper_linear(pool: Any) -> tuple[Any, ...]:
    k = pool.symbol("k")
    return (k, k)


def _case_quadratic_solve(pool: Any) -> tuple[Any, ...]:
    x = pool.symbol("x")
    return ([x ** pool.integer(2) - pool.integer(4)], [x])


def _case_exp_series(pool: Any) -> tuple[Any, ...]:
    ak = _ak()
    x = pool.symbol("x")
    return (ak.exp(x), x, pool.integer(0), 5)


def _case_root_sum_antiderivative(pool: Any) -> tuple[Any, ...]:
    x = pool.symbol("x")
    return (x / (x ** pool.integer(4) + pool.integer(1)), x)


def _case_pythagorean_derivative(pool: Any) -> tuple[Any, ...]:
    ak = _ak()
    x = pool.symbol("x")
    return (ak.sin(ak.exp(x)) * ak.cos(x), x)


#: Previously-observed comparisons, re-run on every pull request.
#:
#: This is the ratchet's downstream end: a divergence the nightly sweep finds
#: must be promoted into a case here (with ``found_by`` naming the seed), or it
#: only ever gets exercised by a job nobody reads. Cases are added, never
#: silently deleted — an expectation that changes is a re-pin with a new
#: ``oracle_versions`` range, and the old range says what used to be true.
FROZEN_CORPUS: tuple[FrozenCase, ...] = (
    FrozenCase(
        id="integrate_polynomial",
        operation="integrate",
        build=_case_polynomial_antiderivative,
        expected="agree",
        expected_reason="invariant_holds",
        found_by="hand-written baseline",
        note=(
            "The rung-4 control. Both antiderivatives differ by a constant and by "
            "term order, so any rung below 4 would report noise here."
        ),
    ),
    FrozenCase(
        id="integrate_sin_cos",
        operation="integrate",
        build=_case_trig_antiderivative,
        expected="agree",
        expected_reason="invariant_holds",
        found_by="hand-written baseline",
        note=(
            "sin(x)cos(x) has three standard antiderivatives that differ by "
            "constants; differentiating both is the only comparison that survives it."
        ),
    ),
    FrozenCase(
        id="integrate_root_sum",
        operation="integrate",
        build=_case_root_sum_antiderivative,
        expected="agree",
        expected_reason="invariant_holds",
        found_by="hand probe, 2026-08-11: the only observed route through the root_sum node",
        note=(
            "Alkahest answers with a RootSum and SymPy with atan(x^2)/2. Nothing below "
            "rung 4 settles it, ball arithmetic cannot evaluate a RootSum at all, and "
            "the case only lands on 'agree' because the invariant rung falls back to "
            "the oracle's normaliser for the alkahest-side residual. It is the case "
            "that pins that fallback, and the root_sum translation with it."
        ),
    ),
    FrozenCase(
        id="diff_chain_rule",
        operation="diff",
        build=_case_chain_rule,
        expected="agree",
        found_by="hand-written baseline",
        note="Composite derivative — exercises the ladder where no invariant exists.",
    ),
    FrozenCase(
        id="diff_product_of_composites",
        operation="diff",
        build=_case_pythagorean_derivative,
        expected="agree",
        found_by="hand-written baseline",
        note=(
            "Product and chain rule together; the two systems order the factors "
            "differently, so this exercises rungs 2 and 3 rather than rung 1."
        ),
    ),
    FrozenCase(
        id="simplify_pythagorean",
        operation="simplify",
        build=_case_pythagorean_simplify,
        expected="agree",
        expected_reason="invariant_holds",
        found_by="hand-written baseline",
        note=(
            "SymPy folds sin^2 + cos^2 to 1 and Alkahest may not. That is a strength "
            "difference, not a divergence, and the value-preserving invariant is what "
            "keeps the gate from mistaking one for the other."
        ),
    ),
    FrozenCase(
        id="sum_indefinite_linear",
        operation="sum_indefinite",
        build=_case_gosper_linear,
        expected="agree",
        expected_reason="invariant_holds",
        found_by="hand-written baseline",
        note=(
            "Gosper antidifferences are pinned only up to a constant, and the two "
            "systems pick different ones; the telescoping invariant removes it."
        ),
    ),
    FrozenCase(
        id="solve_quadratic",
        operation="solve",
        build=_case_quadratic_solve,
        expected="agree",
        expected_reason="invariant_holds",
        found_by="hand-written baseline",
        note="Substitute-back plus a set-size comparison; catches a missed root.",
    ),
    FrozenCase(
        id="series_exp",
        operation="series",
        build=_case_exp_series,
        expected="agree",
        found_by="hand-written baseline",
        note="Exercises the big_o node and the truncation-convention invariant.",
    ),
)


def run_frozen_corpus(
    *,
    oracle: Oracle | str | None = None,
    cases: Sequence[FrozenCase] = FROZEN_CORPUS,
) -> list[tuple[FrozenCase, CrossCheck | None]]:
    """Re-run :data:`FROZEN_CORPUS` against the installed oracle.

    Parameters
    ----------
    oracle : Oracle or str, optional
        Defaults to the first installed oracle.
    cases : sequence of FrozenCase
        Defaults to the whole corpus.

    Returns
    -------
    list of (FrozenCase, CrossCheck or None)
        ``None`` means the case does not apply to the installed oracle version
        and was skipped — a skip, never a pass.
    """
    ak = _ak()
    resolved = _resolve_oracle(oracle)
    version = _safe_version(resolved) if resolved else None
    out: list[tuple[FrozenCase, CrossCheck | None]] = []
    for case in cases:
        if resolved is None or version is None or not case.applies_to(version):
            out.append((case, None))
            continue
        if case.oracle != resolved.name:
            out.append((case, None))
            continue
        pool = ak.ExprPool()
        with ak.context(pool=pool):
            out.append((case, check(case.operation, *case.build(pool), oracle=resolved)))
    return out
