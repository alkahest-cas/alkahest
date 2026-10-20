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
from .alkahest import unify_units as _unify_units

__all__ = [
    "Quantity",
    "Unit",
    "UnitError",
    "acos",
    "apply",
    "asin",
    "atan",
    "concentration_to_ph",
    "convert",
    "cos",
    "cosh",
    "db_to_ratio",
    "diff",
    "dimension_of",
    "exp",
    "integrate",
    "known_units",
    "log",
    "np_to_ratio",
    "parse_unit",
    "ph_to_concentration",
    "ratio_to_db",
    "ratio_to_np",
    "simplify_units",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "to_base",
    "unify_units",
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

    def simplify_units(self, derived: bool = True, allowed=None) -> Quantity:
        """Merge factors of equal dimension (``mg/kg*g → mg``, ``km/m → 1``)
        and, with ``derived=True`` (default), rename an exact match to a named
        SI derived unit (``kg*m/s^2 → N``). See :func:`simplify_units`."""
        ratio, pi_power, unit = self.unit.simplify(
            derived, None if allowed is None else list(allowed)
        )
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


def simplify_units(q: Quantity, derived: bool = True, allowed=None) -> Quantity:
    """Merge the factors of ``q.unit`` that share a dimension into the first
    one (``mg/kg*g → mg``, ``L/h*min → L``, ``km/m → 1``), folding the exact
    conversion ratio into the value; then, unless ``derived=False``, rename the
    result to the named SI derived unit it **exactly** equals.

    Parameters
    ----------
    q : Quantity
    derived : bool, default True
        ``kg*m/s^2 → N``, ``kg*m^2/s^2 → J``, ``A*s → C``, ``kg*m^2/(A*s^3) →
        V``; a power-of-1000 prefix is allowed (``g*m/s^2 → mN``). The
        candidates are ``N, Pa, J, W, C, V, ohm, F, H, T, Wb, S, Hz`` (their
        dimensions are pairwise distinct, so there is no ambiguity among them;
        ``N*m`` is a ``J``). A unit that is already a single named factor,
        carries a dimensionless factor (``rad/s`` is not ``Hz``) or whose scale
        is not a power of 1000 times the derived unit (``kg*cm/s^2``) is left
        as is. ``Gy``, ``Sv`` and ``Bq`` are never produced.
    allowed : iterable of str, optional
        Restrict the candidates, e.g. ``allowed=["N", "J"]``.

    Example
    -------
    >>> simplify_units(Quantity(3, "kg*m/s^2")).unit
    Unit('N')
    >>> simplify_units(Quantity(3, "kg*m/s^2"), derived=False).unit
    Unit('kg*m/s^2')
    """
    return q.simplify_units(derived, allowed)


def unify_units(expr, units, prefer: str = "finest") -> Quantity:
    """Like :func:`dimension_of`, but convert a sum whose terms share a
    dimension and differ in unit instead of refusing it (``E-UNIT-006``).

    ``a`` in ``h`` plus ``b`` in ``min`` becomes ``60*a + b`` in ``min``: each
    such term is multiplied by the exact ratio of its unit to the target unit
    given by ``prefer`` — ``"finest"`` (default, the smallest unit among the
    terms, so factors are integers wherever possible), ``"coarsest"`` or
    ``"first"`` (the first term in canonical order). ``min``/``max``/``atan2``
    arguments and ``Piecewise`` values are treated the same way.

    Returns a :class:`Quantity` whose ``value`` is the rewritten formula and
    whose ``unit`` is the unit that value is in when every symbol is in its
    listed unit. A formula without mixed sums comes back unchanged; every other
    refusal of :func:`dimension_of` stands (``E-UNIT-001``, ``-003``, ``-004``,
    ``-005``, ``-007``).

    Example
    -------
    >>> p = ExprPool(); a, b = p.symbol("a"), p.symbol("b")
    >>> q = unify_units(a + b, {a: "h", b: "min"})   # 60*a + b
    >>> str(q.unit)
    'min'
    """
    value, unit = _unify_units(expr, units, prefer)
    return Quantity(value, unit)


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


# ---------------------------------------------------------------------------
# Logarithmic scales: explicit conversions only (no arithmetic)
# ---------------------------------------------------------------------------
#
# dB, Np and pH are not units with a scale factor: ``10 dB + 10 dB`` is not
# ``20 dB`` of anything (powers multiply), so ``Unit("dB")`` is refused
# (E-UNIT-002). Convert to a plain ratio / concentration, do the arithmetic
# there, and convert back. (``Unit("pH")`` is the SI picohenry, not pH.)


def _log_arg(x):
    if isinstance(x, Quantity):
        x.unit.require_pure_number("a logarithmic-scale value")
        x = x.value
    if isinstance(x, bool):
        raise TypeError("a bool is not a number")
    return x


def _kind(kind: str, power: int, field: int) -> int:
    if kind == "power":
        return power
    if kind == "field":
        return field
    raise ValueError("kind must be 'power' or 'field'")


def _pow10(q: Fraction):
    """``10 ** q`` for an integer ``q`` (int, or Fraction when negative)."""
    return int(10**q) if q >= 0 else Fraction(1, 10 ** (-int(q)))


def _exact_log10(r: Fraction):
    """``k`` when ``r == 10**k``, else ``None``."""
    n, d = r.numerator, r.denominator
    if d == 1:
        m, sign = n, 1
    elif n == 1:
        m, sign = d, -1
    else:
        return None
    k = 0
    while m % 10 == 0:
        m //= 10
        k += 1
    return sign * k if m == 1 else None


def _as_ratio_expr(r: Fraction) -> Expr:
    p = _default_pool()
    if r.denominator == 1:
        return p.integer(r.numerator)
    return p.rational(r.numerator, r.denominator)


def _positive_fraction(r) -> Fraction:
    r = Fraction(r)
    if r <= 0:
        raise ValueError("the ratio must be positive")
    return r


def db_to_ratio(db, kind: str = "power"):
    """The ratio a level in decibels stands for: ``10**(db/10)`` for a power
    quantity (``kind="power"``, default) and ``10**(db/20)`` for a field
    quantity (``kind="field"``: voltage, current, pressure, amplitude).

    Exact: ``db_to_ratio(30)`` is ``1000`` and ``db_to_ratio(3)`` the symbolic
    ``10**(3/10)`` — never a float unless the input is one. ``db`` may be a
    number, an ``Expr`` or a pure-number :class:`Quantity`. The inverse is
    :func:`ratio_to_db`. There is no ``dB`` unit and no dB arithmetic:
    ``Unit("dB")`` is refused with a pointer here.

    Example
    -------
    >>> db_to_ratio(20, "field")
    10
    """
    d = _kind(kind, 10, 20)
    x = _log_arg(db)
    if isinstance(x, float):
        return 10.0 ** (x / d)
    if isinstance(x, Expr):
        return x._pool.integer(10) ** (x / d)
    q = Fraction(x) / d
    if q.denominator == 1:
        return _pow10(q)
    return _default_pool().integer(10) ** q


def ratio_to_db(ratio, kind: str = "power"):
    """The level in decibels of a positive ratio: ``10·log10(ratio)`` for a
    power quantity, ``20·log10(ratio)`` for a field quantity. Exact for exact
    powers of ten (``ratio_to_db(1000)`` is ``30``), symbolic otherwise
    (``Expr`` in, ``10*log(r)/log(10)`` out). ``ValueError`` for a ratio ≤ 0.

    Example
    -------
    >>> ratio_to_db(100, "field")
    40
    """
    from . import alkahest as _native

    f = _kind(kind, 10, 20)
    r = _log_arg(ratio)
    if isinstance(r, float):
        import math

        return f * math.log10(r)
    if isinstance(r, Expr):
        return f * _native.log(r) / _native.log(r._pool.integer(10))
    r = _positive_fraction(r)
    k = _exact_log10(r)
    if k is not None:
        return f * k
    p = _default_pool()
    return f * _native.log(_as_ratio_expr(r)) / _native.log(p.integer(10))


def np_to_ratio(np_, kind: str = "field"):
    """The ratio a level in nepers stands for: ``exp(np)`` for a field
    quantity (default) and ``exp(2·np)`` for a power quantity. Exact
    (``exp(...)`` stays symbolic; ``0`` Np is ``1``)."""
    from . import alkahest as _native

    k = _kind(kind, 2, 1)
    x = _log_arg(np_)
    if isinstance(x, float):
        import math

        return math.exp(k * x)
    if not isinstance(x, Expr):
        if x == 0:
            return 1
        x = _lift(x, _default_pool().integer(0))
    return _native.exp(x if k == 1 else k * x)


def ratio_to_np(ratio, kind: str = "field"):
    """The level in nepers of a positive ratio: ``ln(ratio)`` for a field
    quantity, ``ln(ratio)/2`` for a power quantity. ``ratio_to_np(1)`` is
    ``0``; ``ValueError`` for a ratio ≤ 0."""
    from . import alkahest as _native

    k = _kind(kind, 2, 1)
    r = _log_arg(ratio)
    if isinstance(r, float):
        import math

        return math.log(r) / k
    if not isinstance(r, Expr):
        r = _positive_fraction(r)
        if r == 1:
            return 0
        r = _as_ratio_expr(r)
    v = _native.log(r)
    return v if k == 1 else v / k


def ph_to_concentration(ph) -> Quantity:
    """The hydrogen-ion concentration of a pH as a :class:`Quantity` in
    ``mol/L``: ``10**(-pH)`` (pH 7 is ``Fraction(1, 10**7)``), taking the
    IUPAC standard state of 1 mol/L (activity ≈ concentration). The inverse
    is :func:`concentration_to_ph`."""
    x = _log_arg(ph)
    if isinstance(x, float):
        return Quantity(10.0 ** (-x), "mol/L")
    if isinstance(x, Expr):
        return Quantity(x._pool.integer(10) ** (-x), "mol/L")
    q = -Fraction(x)
    if q.denominator == 1:
        return Quantity(_pow10(q), "mol/L")
    return Quantity(_default_pool().integer(10) ** q, "mol/L")


def concentration_to_ph(c):
    """The pH of a hydrogen-ion concentration: ``-log10(c / (1 mol/L))``.
    ``c`` is a :class:`Quantity` of amount per volume (any unit; it is
    converted exactly to ``mol/L`` first). Exact for powers of ten
    (``Fraction(1, 10**7)`` mol/L → ``7``), symbolic otherwise; ``UnitError``
    ``E-UNIT-001`` for another dimension, ``ValueError`` for a concentration
    ≤ 0."""
    from . import alkahest as _native

    if not isinstance(c, Quantity):
        raise TypeError("pass the concentration as a Quantity, e.g. Quantity(x, 'mol/L')")
    v = c.convert("mol/L").value
    if isinstance(v, float):
        import math

        return -math.log10(v)
    if isinstance(v, Expr):
        return -_native.log(v) / _native.log(v._pool.integer(10))
    v = _positive_fraction(v)
    k = _exact_log10(v)
    if k is not None:
        return -k
    return -_native.log(_as_ratio_expr(v)) / _native.log(_default_pool().integer(10))
