"""alkahest.units — physical units, quantities and dimensional analysis.

A :class:`Unit` is a product of named units with **rational** exponents over
the seven SI base dimensions, carrying an **exact** scale relative to SI
(``1 inch = 127/5000 m``, ``1 deg = π/180 rad`` — never a float). A
:class:`Quantity` pairs a value (an ``Expr``, ``int`` or ``Fraction``) with a
unit, and its arithmetic checks dimensions:

* ``+`` / ``-`` need equal dimensions (``UnitError`` ``E-UNIT-001`` names both)
  and convert the right operand exactly into the left operand's unit;
* ``*`` / ``/`` / ``**`` (rational exponent) combine units;
* :func:`exp`, :func:`log`, :func:`sin`, … need a dimensionless argument
  (``E-UNIT-004``); an angle in ``deg`` is converted to ``rad`` first;
* ``degC`` / ``degF`` are affine scales: they can be :func:`convert`\\ ed and
  nothing else (``E-UNIT-003``).

:func:`dimension_of` checks a whole symbolic formula given each symbol's unit,
and :func:`diff` / :func:`integrate` carry units through calculus
(``∫ C dt`` of a concentration over time is in ``mg*h/L``).

Example
-------
>>> from alkahest import ExprPool
>>> from alkahest.units import Quantity, convert
>>> convert(Quantity(1, "inch"), "m").value
Fraction(127, 5000)
"""

from __future__ import annotations

from fractions import Fraction
from numbers import Rational as _Rational

from .alkahest import (
    Expr,
    ExprPool,
    Unit,
    UnitError,
    _unit_affine_map,
    dimension_of,
    known_units,
    parse_unit,
)

__all__ = [
    "Quantity",
    "Unit",
    "UnitError",
    "acos",
    "apply",
    "asin",
    "atan",
    "convert",
    "cos",
    "cosh",
    "diff",
    "dimension_of",
    "exp",
    "integrate",
    "known_units",
    "log",
    "parse_unit",
    "simplify_units",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "to_base",
]

_DEFAULT_POOL: ExprPool | None = None


def _default_pool() -> ExprPool:
    global _DEFAULT_POOL
    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = ExprPool()
    return _DEFAULT_POOL


def _as_unit(u) -> Unit:
    if isinstance(u, Unit):
        return u
    if isinstance(u, str):
        return parse_unit(u)
    raise TypeError(f"expected a Unit or a unit string, got {type(u).__name__}")


def _unwrap(v):
    """Accept Expr, DerivedResult (.value), int, Fraction, float."""
    if isinstance(v, Quantity):
        raise TypeError("a Quantity cannot be the value of another Quantity")
    if isinstance(v, bool):
        raise TypeError("a bool is not a quantity value")
    if isinstance(v, (Expr, int, Fraction, float)):
        return v
    if hasattr(v, "value") and isinstance(v.value, Expr):  # DerivedResult
        return v.value
    if isinstance(v, _Rational):
        return Fraction(v.numerator, v.denominator)
    raise TypeError(
        f"a quantity value must be an Expr, int, Fraction or float, got {type(v).__name__}"
    )


def _normalize_number(x):
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x)
    return x


def _lift(v, like: Expr) -> Expr:
    """``v`` as an ``Expr`` in the pool of ``like``."""
    if isinstance(v, Expr):
        return v
    return _unit_affine_map(v, 1, 0, 0, like)


def _affine(v, ratio: Fraction, pi_power: Fraction, shift: Fraction, like=None):
    """``ratio·π**pi_power·v + shift``, exactly."""
    if ratio == 1 and pi_power == 0 and shift == 0:
        return v
    if not isinstance(v, Expr):
        if pi_power == 0:
            if isinstance(v, float):
                return v * float(ratio) + float(shift)
            return _normalize_number(v * ratio + shift)
        if not isinstance(like, Expr):
            like = _default_pool().integer(0)
        return _unit_affine_map(v, ratio, pi_power, shift, like)
    return _unit_affine_map(v, ratio, pi_power, shift)


class Quantity:
    """A value together with its :class:`Unit`.

    Parameters
    ----------
    value : Expr, int, Fraction or float
        The magnitude. Use ``int`` / ``Fraction`` / ``Expr`` for exact work; a
        ``float`` is kept as given (conversions never introduce one).
    unit : Unit or str, default ``"1"``
        The unit, e.g. ``"mg"``, ``"L/h"``, ``Unit("m/s^2")``.

    Arithmetic checks dimensions (see the module docstring). Comparison with
    ``==`` is identity; compare ``convert(a, b.unit).value`` with ``b.value``
    instead.

    Example
    -------
    >>> m = Quantity(2, "kg"); a = Quantity(3, "m/s^2")
    >>> (m * a).unit == Unit("N")
    True
    """

    __slots__ = ("unit", "value")

    def __init__(self, value, unit="1"):
        self.value = _unwrap(value)
        self.unit = _as_unit(unit)

    # -- presentation ---------------------------------------------------------
    def __repr__(self) -> str:
        return f"Quantity({self.value!r}, {str(self.unit)!r})"

    def __str__(self) -> str:
        u = str(self.unit)
        return f"{self.value}" if u == "1" else f"{self.value} {u}"

    @property
    def dimension(self) -> str:
        """The dimension of :attr:`unit`, e.g. ``"mass/length^3"``."""
        return self.unit.dimension

    # -- conversions ----------------------------------------------------------
    def convert(self, unit, *, _like=None) -> Quantity:
        """Express in ``unit`` (exact). See :func:`convert`."""
        target = _as_unit(unit)
        ratio, pi_power, shift = self.unit.conversion(target)
        return Quantity(_affine(self.value, ratio, pi_power, shift, _like), target)

    def to_base(self) -> Quantity:
        """Express in coherent SI base units (``J → kg*m^2/s^2``)."""
        return self.convert(self.unit.to_base())

    def simplify_units(self) -> Quantity:
        """Merge factors of equal dimension (``mg/kg*g → mg``, ``km/m → 1``)."""
        ratio, pi_power, unit = self.unit.simplify()
        return Quantity(_affine(self.value, ratio, pi_power, Fraction(0)), unit)

    def simplify(self) -> Quantity:
        """Simplify the value (``alkahest.simplify``); the unit is unchanged."""
        if not isinstance(self.value, Expr):
            return self
        from . import simplify as _simplify

        return Quantity(_simplify(self.value).value, self.unit)

    # -- arithmetic -----------------------------------------------------------
    @staticmethod
    def _coerce(other):
        if isinstance(other, Quantity):
            return other
        if isinstance(other, Unit):
            return None
        try:
            return Quantity(other, Unit.dimensionless())
        except TypeError:
            return None

    def _sum(self, other, sign: int, op: str):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        self.unit.check_same_dimension(o.unit, op)
        for u in (self.unit, o.unit):
            if u.is_affine:
                u.require_pure_number("a sum or difference")
        rhs = o.convert(self.unit, _like=self.value).value
        return Quantity(self.value + rhs if sign > 0 else self.value - rhs, self.unit)

    def __add__(self, other):
        return self._sum(other, 1, "add")

    def __radd__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else o._sum(self, 1, "add")

    def __sub__(self, other):
        return self._sum(other, -1, "subtract")

    def __rsub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else o._sum(self, -1, "subtract")

    def __neg__(self):
        if self.unit.is_affine:
            self.unit.require_pure_number("a negation")
        return Quantity(-self.value, self.unit)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Quantity(self.value, self.unit * other)
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return Quantity(self.value * o.value, self.unit * o.unit)

    def __rmul__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return Quantity(o.value * self.value, o.unit * self.unit)

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return Quantity(self.value, self.unit / other)
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        unit = self.unit / o.unit
        return Quantity(_divide(self.value, o.value), unit)

    def __rtruediv__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        unit = o.unit / self.unit
        return Quantity(_divide(o.value, self.value), unit)

    def __pow__(self, exponent):
        if isinstance(exponent, Quantity):
            exponent.unit.require_pure_number("exponent")
            exponent = exponent.convert(Unit.dimensionless()).value
        if isinstance(exponent, (int, Fraction)) and not isinstance(exponent, bool):
            q = Fraction(exponent)
            unit = self.unit**q
            v = self.value
            e = int(q) if q.denominator == 1 else q
            if isinstance(v, Expr):
                return Quantity(v**e, unit)
            if isinstance(v, float):
                return Quantity(v ** float(q), unit)
            if q.denominator == 1:
                return Quantity(_normalize_number(Fraction(v) ** q.numerator), unit)
            return Quantity(_lift(v, _default_pool().integer(0)) ** q, unit)
        # A symbolic exponent needs a pure-number base.
        self.unit.require_pure_number("base of a symbolic power")
        base = self.convert(Unit.dimensionless()).value
        return Quantity(base**exponent, Unit.dimensionless())


def _divide(a, b):
    if isinstance(a, Expr) or isinstance(b, Expr):
        return a / b
    if isinstance(a, float) or isinstance(b, float):
        return a / b
    return _normalize_number(Fraction(a) / Fraction(b))


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------


def convert(q: Quantity, unit) -> Quantity:
    """Express ``q`` in ``unit``, exactly.

    The value is multiplied by the exact ratio of the two scales (and, between
    ``degC``/``degF``/``K``/``degR``, shifted by the exact offset).

    Raises ``UnitError`` ``E-UNIT-001`` when the dimensions differ.

    Example
    -------
    >>> convert(Quantity(100, "degC"), "degF").value
    212
    """
    return q.convert(unit)


def to_base(q: Quantity) -> Quantity:
    """Express ``q`` in coherent SI base units (``kg``, ``m``, ``s``, ``A``,
    ``K``, ``mol``, ``cd``); ``degC``/``degF`` go to ``K``."""
    return q.to_base()


def simplify_units(q: Quantity) -> Quantity:
    """Merge the factors of ``q.unit`` that share a dimension into the first
    one (``mg/kg*g → mg``, ``L/h*min → L``, ``km/m → 1``), folding the exact
    conversion ratio into the value."""
    return q.simplify_units()


def _as_quantity(q) -> Quantity:
    return q if isinstance(q, Quantity) else Quantity(q, Unit.dimensionless())


def apply(func, q, name: str | None = None) -> Quantity:
    """Apply a function that needs a pure-number argument (``exp``, ``log``,
    trigonometric, …) to ``q``.

    A dimensionless ``q`` with a scale (``deg``, ``%``) is converted to a pure
    number first, so ``sin(Quantity(90, "deg"))`` is ``sin(π/2)``. A
    dimensional ``q`` is refused with ``UnitError`` ``E-UNIT-004``.
    """
    q = _as_quantity(q)
    label = name or getattr(func, "__name__", "the function")
    if not q.unit.same_dimension(Unit.dimensionless()):
        q.unit.require_pure_number(f"argument of {label}")
    if q.unit.is_affine:
        q.unit.require_pure_number(f"argument of {label}")
    v = q.convert(Unit.dimensionless()).value
    if not isinstance(v, Expr):
        v = _lift(v, _default_pool().integer(0))
    return Quantity(func(v), Unit.dimensionless())


def _fn(name: str):
    def f(q):
        from . import alkahest as _native

        return apply(getattr(_native, name), q, name)

    f.__name__ = name
    f.__qualname__ = name
    f.__doc__ = (
        f"``{name}`` of a dimensionless quantity (``UnitError`` ``E-UNIT-004`` "
        "otherwise; angles in ``deg`` are converted to ``rad``)."
    )
    return f


exp = _fn("exp")
log = _fn("log")
sin = _fn("sin")
cos = _fn("cos")
tan = _fn("tan")
asin = _fn("asin")
acos = _fn("acos")
atan = _fn("atan")
sinh = _fn("sinh")
cosh = _fn("cosh")
tanh = _fn("tanh")


def sqrt(q) -> Quantity:
    """``q ** (1/2)``; the unit's scale must have an exact square root
    (``E-UNIT-007`` otherwise, e.g. ``km``)."""
    return _as_quantity(q) ** Fraction(1, 2)


def _var_parts(var) -> tuple[Expr, Unit]:
    if isinstance(var, Quantity):
        if not isinstance(var.value, Expr):
            raise TypeError("the variable must be a Quantity whose value is a symbol")
        return var.value, var.unit
    if isinstance(var, Expr):
        return var, Unit.dimensionless()
    raise TypeError("the variable must be a Quantity (symbol, unit) or an Expr")


def diff(q, var) -> Quantity:
    """``d q / d var`` with unit ``q.unit / var.unit``.

    ``var`` is a :class:`Quantity` whose value is a symbol, e.g.
    ``Quantity(t, "h")``.

    Example
    -------
    >>> # x(t) in m, t in s  →  dx/dt in m/s
    """
    from . import diff as _diff

    q = _as_quantity(q)
    v, vu = _var_parts(var)
    if q.unit.is_affine:
        q.unit.require_pure_number("a derivative")
    d = _diff(_lift(q.value, v), v).value
    return Quantity(d, q.unit / vu)


_UNIT_FREE_BOUNDS = {"0", "oo", "-oo", "∞", "-∞", "(-1 * oo)", "(oo * -1)", "(-1 * ∞)", "(∞ * -1)"}


def _bound(b, vu: Unit):
    if isinstance(b, Quantity):
        return b.convert(vu).value
    if vu.same_dimension(Unit.dimensionless()) and vu.is_pure_number:
        return b
    if isinstance(b, int) and b == 0:
        return b
    if isinstance(b, Expr) and str(b) in _UNIT_FREE_BOUNDS:
        return b
    # Anything else is a number whose unit is unknown.
    Unit.dimensionless().check_same_dimension(vu, "use a plain number as a bound of")
    return b  # pragma: no cover - the line above always raises here


def integrate(q, var, lower=None, upper=None) -> Quantity:
    """``∫ q d var`` with unit ``q.unit · var.unit``.

    ``var`` is a :class:`Quantity` whose value is a symbol, e.g.
    ``Quantity(t, "h")``. Bounds may be quantities (converted exactly into
    ``var``'s unit) or the unit-free ``0`` and ``±oo``; a bare number for a
    dimensional variable is refused (``E-UNIT-001``), because its unit is
    unknown.

    Example
    -------
    >>> # AUC: ∫_0^∞ C(t) dt with C in mg/L and t in h is in mg*h/L.
    """
    from . import integrate as _integrate

    q = _as_quantity(q)
    v, vu = _var_parts(var)
    if q.unit.is_affine:
        q.unit.require_pure_number("an integral")
    unit = q.unit * vu
    value = _lift(q.value, v)
    if (lower is None) != (upper is None):
        raise ValueError("pass both bounds for a definite integral, or neither")
    if lower is None:
        return Quantity(_integrate(value, v).value, unit)
    a, b = _bound(lower, vu), _bound(upper, vu)
    return Quantity(_integrate(value, v, a, b).value, unit)
