Units API
=========

.. currentmodule:: alkahest

Physical units, quantities and dimensional analysis (:mod:`alkahest.units`).
Every scale is exact — a rational times a power of π — so no conversion ever
introduces a float. See the *Units and dimensional analysis* chapter of the
user guide for the full list of units and rules.

Units
-----

.. class:: Unit(spec: str)

   A product of named units with rational exponents over the seven SI base
   dimensions, with an exact scale relative to SI. Parsed from strings such as
   ``"mg*h/L"``, ``"m/s^2"``, ``"kg m s^-2"``, ``"m^(1/2)"``, ``"µg/mL"``; combine
   with ``*``, ``/`` and ``**`` (``int`` or ``Fraction`` exponent). Equality is
   semantic (``Unit("N") == Unit("kg*m/s^2")``).

   Properties: ``name``, ``dimension`` (e.g. ``"mass/length^3"``),
   ``dimension_exponents``, ``factors``, ``scale`` (``(ratio, pi_power)``),
   ``is_affine``, ``is_pure_number``. Methods: ``conversion(target)`` →
   ``(ratio, pi_power, shift)``, ``to_base()``, ``simplify()``,
   ``same_dimension(other)``, ``check_same_dimension(other, operation)``,
   ``require_pure_number(context)``.

.. function:: parse_unit(spec: str) -> Unit

   Parse a unit string. Raises :class:`UnitError` ``E-UNIT-002`` for a
   malformed string or unknown unit (``"m2"`` is refused, not read as
   ``m^2``) and ``E-UNIT-003`` for ``degC``/``degF`` inside a compound unit.

Quantities
----------

.. class:: Quantity(value, unit="1")

   A value (``Expr``, ``int``, ``Fraction`` or ``float``) with a :class:`Unit`.
   ``+``/``-`` require equal dimensions (``E-UNIT-001``, naming both) and
   convert the right operand exactly; ``*``, ``/`` and rational ``**`` combine
   units. Methods ``convert(unit)``, ``to_base()``, ``simplify_units()``,
   ``simplify()``.

   Example::

      F = Quantity(2, "kg") * Quantity(3, "m/s^2")
      F.unit == Unit("N")                             # True
      convert_units(Quantity(1, "inch"), "m").value   # Fraction(127, 5000)
      convert_units(Quantity(100, "degC"), "degF").value   # 212

.. function:: convert_units(q: Quantity, unit) -> Quantity

   ``alkahest.units.convert``. Express ``q`` in ``unit``, exactly (affine
   temperature scales included). ``E-UNIT-001`` when the dimensions differ.

.. function:: to_base_units(q: Quantity) -> Quantity

   ``alkahest.units.to_base``. Express ``q`` in coherent SI base units.

.. function:: simplify_units(q: Quantity) -> Quantity

   Merge factors of equal dimension (``mg/kg·kg → mg``, ``km/m → 1``),
   folding the exact ratio into the value.

Dimensional analysis
--------------------

.. function:: dimension_of(expr: Expr, units: dict) -> Unit

   The unit of ``expr`` given the unit of each symbol (keys may be any
   sub-expression, such as ``C(t)``). Checks the whole formula: sums need
   equal dimensions (``E-UNIT-001``) and equal units (``E-UNIT-006``),
   transcendental arguments and symbolic exponents must be pure numbers
   (``E-UNIT-004``), every symbol needs a unit (``E-UNIT-005``).

   Example::

      C = F*D*ka/(V*(ka - ke)) * (exp(-ke*t) - exp(-ka*t))
      dimension_of(C, {F: "1", D: "mg", ka: "1/h", ke: "1/h", V: "L", t: "h"})
      # Unit('mg/L')

Calculus with units
-------------------

``alkahest.units.diff(q, var)`` and ``alkahest.units.integrate(q, var,
lower=None, upper=None)`` take the variable as a ``Quantity`` whose value is a
symbol, and return ``q.unit / var.unit`` and ``q.unit · var.unit``
respectively. Bounds are quantities (converted into the variable's unit) or the
unit-free ``0`` and ``±oo``. ``alkahest.units.exp``, ``log``, ``sin``, … (and
``apply(f, q)``) require a dimensionless argument and convert ``deg``/``%`` to a
pure number first.

Refusals raise :exc:`UnitError` (``E-UNIT-001`` … ``E-UNIT-007``); see
:doc:`errors`.
