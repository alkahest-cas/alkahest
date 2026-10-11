"""G10 — physical units, quantities and dimensional analysis (alkahest.units).

Oracles: an exact hand-checked conversion table (SI brochure / NIST SP 811
definitions), ``pint`` when it is installed, SymPy for the pharmacokinetic AUC,
and mpmath for the angle conversion.
"""

from __future__ import annotations

import math
from fractions import Fraction

import alkahest
import pytest
from alkahest import ExprPool, Quantity, Unit, UnitError, dimension_of, parse_unit
from alkahest import units as U


def _code(excinfo) -> str:
    return excinfo.value.code


# ---------------------------------------------------------------------------
# Exactness and conversion table
# ---------------------------------------------------------------------------

EXACT_TABLE = [
    # (from, to, exact ratio)
    ("inch", "m", Fraction(127, 5000)),
    ("ft", "inch", Fraction(12)),
    ("mi", "ft", Fraction(5280)),
    ("lb", "kg", Fraction(45359237, 10**8)),
    ("lbf", "N", Fraction(45359237, 10**8) * Fraction(980665, 10**5)),
    ("psi", "Pa", Fraction(44482216152605, 10**13) / Fraction(64516, 10**8)),
    ("atm", "Pa", Fraction(101325)),
    ("bar", "Pa", Fraction(100000)),
    ("atm", "Torr", Fraction(760)),
    ("mmHg", "Pa", Fraction(133322387415, 10**9)),
    ("mph", "m/s", Fraction(1397, 3125)),
    ("kWh", "J", Fraction(3_600_000)),
    ("kcal", "J", Fraction(4184)),
    ("eV", "J", Fraction(1602176634, 10**28)),
    ("gal", "L", Fraction(3785411784, 10**9)),
    ("hp", "W", Fraction(74569987158227022, 10**14)),
    ("mg", "g", Fraction(1, 1000)),
    ("µg", "mg", Fraction(1, 1000)),
    ("μg", "mg", Fraction(1, 1000)),
    ("ug", "mg", Fraction(1, 1000)),
    ("mcg", "ug", Fraction(1)),
    ("mL", "L", Fraction(1, 1000)),
    ("L", "m^3", Fraction(1, 1000)),
    ("L/h", "mL/min", Fraction(50, 3)),
    ("mg*h/L", "ug*min/mL", Fraction(60)),
    ("mg/kg", "ug/g", Fraction(1)),
    ("M", "mol/L", Fraction(1)),
    ("nM", "mol/m^3", Fraction(1, 10**6)),
    ("day", "h", Fraction(24)),
    ("kg m s^-2", "N", Fraction(1)),
    ("N*m", "J", Fraction(1)),
    ("V*A", "W", Fraction(1)),
    ("ohm", "V/A", Fraction(1)),
    ("kΩ", "Ω", Fraction(1000)),
    ("degR", "K", Fraction(5, 9)),
    ("delta_degF", "delta_degC", Fraction(5, 9)),
    ("%", "1", Fraction(1, 100)),
]


@pytest.mark.parametrize(("a", "b", "ratio"), EXACT_TABLE)
def test_exact_conversion_table(a, b, ratio):
    r, pi_power, shift = Unit(a).conversion(b)
    assert (r, pi_power, shift) == (ratio, 0, 0)
    assert isinstance(r, (int, Fraction))
    q = U.convert(Quantity(1, a), b)
    assert q.value == ratio
    assert not isinstance(q.value, float)


def test_inch_is_exactly_127_over_5000_metre():
    assert U.convert(Quantity(1, "inch"), "m").value == Fraction(127, 5000)
    assert U.convert(Quantity(Fraction(5000, 127), "inch"), "m").value == 1


PINT_NAMES = {
    "%": "percent",
    "deg": "degree",
    "Ω": "ohm",
    "kΩ": "kiloohm",
    "μg": "microgram",
    "µg": "microgram",
    "mcg": "microgram",
    "ug": "microgram",
    "1": "dimensionless",
    "Torr": "torr",
}


def _pint(u: str) -> str:
    return PINT_NAMES.get(u, u).replace("^", "**")


@pytest.mark.parametrize(("a", "b", "ratio"), [*EXACT_TABLE, ("deg", "rad", None)])
def test_conversions_match_pint(a, b, ratio):
    pint = pytest.importorskip("pint")
    ureg = pint.UnitRegistry()
    expected = ureg.Quantity(1, _pint(a)).to(_pint(b)).magnitude
    r, pi_power, _ = Unit(a).conversion(b)
    got = float(r) * math.pi ** float(pi_power)
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("value", "a", "b", "expected"),
    [
        (100, "degC", "degF", 212),
        (-40, "degF", "degC", -40),
        (0, "degC", "K", Fraction(27315, 100)),
        (32, "degF", "K", Fraction(27315, 100)),
        (Fraction(27315, 100), "K", "degC", 0),
        (0, "K", "degF", Fraction(-45967, 100)),
        (491, "degR", "degF", Fraction(3133, 100)),
    ],
)
def test_affine_temperature_conversion_is_exact(value, a, b, expected):
    got = U.convert(Quantity(value, a), b).value
    assert got == expected
    pint = pytest.importorskip("pint")
    ureg = pint.UnitRegistry()
    assert float(got) == pytest.approx(
        ureg.Quantity(float(value), a).to(b).magnitude, rel=1e-12, abs=1e-12
    )


def test_affine_temperature_refused_in_arithmetic():
    t = Quantity(20, "degC")
    for bad in (
        lambda: t + Quantity(5, "degC"),
        lambda: t - Quantity(5, "degC"),
        lambda: t * 2,
        lambda: 2 * t,
        lambda: t / Quantity(1, "s"),
        lambda: t**2,
        lambda: -t,
        lambda: parse_unit("degC/s"),
        lambda: parse_unit("J/(kg*degC)"),
        lambda: Unit("degC") * Unit("m"),
    ):
        with pytest.raises(UnitError) as ei:
            bad()
        assert _code(ei) == "E-UNIT-003"
    # Differences are fine in K or delta_degC.
    d = U.convert(t, "K") - U.convert(Quantity(5, "degC"), "K")
    assert d.value == 15
    assert d.unit == Unit("delta_degC")


def test_degree_to_radian_is_exact_pi_over_180():
    import mpmath

    q = U.convert(Quantity(90, "deg"), "rad")
    assert "pi" in str(q.value)
    assert alkahest.eval_expr(q.value, {}) == pytest.approx(float(mpmath.pi / 2), rel=1e-15)
    assert Unit("deg").scale == (Fraction(1, 180), Fraction(1))
    # sin(90 deg) = 1
    s = U.sin(Quantity(90, "deg"))
    assert s.unit.is_pure_number
    assert alkahest.eval_expr(s.value, {}) == pytest.approx(1.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Dimension checking
# ---------------------------------------------------------------------------


def test_adding_different_dimensions_names_both():
    with pytest.raises(UnitError) as ei:
        Quantity(1, "mg") + Quantity(1, "L")
    assert _code(ei) == "E-UNIT-001"
    msg = str(ei.value)
    assert "mass" in msg
    assert "length^3" in msg
    with pytest.raises(UnitError) as ei:
        Quantity(1, "m") - 1
    assert _code(ei) == "E-UNIT-001"
    with pytest.raises(UnitError) as ei:
        U.convert(Quantity(1, "m"), "s")
    assert _code(ei) == "E-UNIT-001"


def test_addition_converts_exactly_into_the_left_unit():
    s = Quantity(1, "m") + Quantity(1, "inch")
    assert s.unit == Unit("m")
    assert str(s.unit) == "m"
    assert s.value == Fraction(5127, 5000)
    s = Quantity(1, "h") + Quantity(30, "min")
    assert s.value == Fraction(3, 2)


def test_functions_need_dimensionless_arguments():
    p = ExprPool()
    x = p.symbol("x")
    with pytest.raises(UnitError) as ei:
        U.exp(Quantity(x, "m"))
    assert _code(ei) == "E-UNIT-004"
    with pytest.raises(UnitError) as ei:
        U.sin(Quantity(x, "s"))
    assert _code(ei) == "E-UNIT-004"
    # radians and ratios are dimensionless
    U.sin(Quantity(x, "rad"))
    r = U.log(Quantity(x, "m") / Quantity(1, "km"))
    assert r.unit.is_pure_number


def test_parse_forms_and_refusals():
    assert parse_unit("m/s^2") == parse_unit("m*s**-2") == parse_unit("m s⁻²")
    assert parse_unit("kg m s^-2") == Unit("N")
    assert str(parse_unit("kg m s^-2")) == "kg*m/s^2"
    assert str(parse_unit("mg*h/L")) == "mg*h/L"
    assert parse_unit("mg*h/L").dimension == "mass*time/length^3"
    assert parse_unit("m^(1/2)").dimension_exponents == {"length": Fraction(1, 2)}
    assert Unit("J").to_base() == Unit("kg*m^2/s^2")
    assert str(Unit("J").to_base()) == "kg*m^2/s^2"
    for bad in ["", "m2", "furlongs", "m^", "m^1/2", "(m", "3m", "0*m"]:
        with pytest.raises(UnitError) as ei:
            parse_unit(bad)
        assert _code(ei) == "E-UNIT-002", bad


def test_rational_powers_and_irrational_scale_refusal():
    assert Unit("cm^2") ** Fraction(1, 2) == Unit("cm")
    assert str(U.sqrt(Quantity(4, "m^2")).unit) == "m"
    with pytest.raises(UnitError) as ei:
        Unit("km") ** Fraction(1, 2)
    assert _code(ei) == "E-UNIT-007"


def test_simplify_units():
    q = (Quantity(5, "mg/kg") * Quantity(70, "kg")).simplify_units()
    assert str(q.unit) == "mg"
    assert q.value == 350
    q = (Quantity(2, "mg/kg") * Quantity(3, "g")).simplify_units()
    assert str(q.unit) == "mg"
    assert q.value == Fraction(6, 1000)
    q = U.simplify_units(Quantity(3, "km") / Quantity(1, "m"))
    assert q.unit.is_pure_number
    assert q.value == 3000
    q = (Quantity(6, "L/h") * Quantity(30, "min")).simplify_units()
    assert str(q.unit) == "L"
    assert q.value == 3


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def test_mechanics_force_energy_power():
    m = Quantity(2, "kg")
    a = Quantity(3, "m/s^2")
    F = m * a
    assert F.unit == Unit("N")
    assert F.value == 6
    W = F * Quantity(5, "m")
    assert W.unit == Unit("J")
    KE = Fraction(1, 2) * m * Quantity(4, "m/s") ** 2
    assert KE.unit == Unit("J")
    assert KE.value == 16
    total = (W + KE).convert("kWh")
    assert total.value == Fraction(46, 3_600_000)
    P = (W + KE) / Quantity(2, "s")
    assert P.unit == Unit("W")
    assert U.convert(P, "W").value == 23
    # psi from lbf / in^2
    p = Quantity(1, "lbf") / Quantity(1, "inch") ** 2
    assert U.convert(p, "psi").value == 1


def test_symbolic_mechanics_dimension_of():
    p = ExprPool()
    m, v, g, h, k, x = (p.symbol(n) for n in ("m", "v", "g", "h", "k", "x"))
    E = Fraction(1, 2) * m * v**2 + m * g * h + Fraction(1, 2) * k * x**2
    units = {m: "kg", v: "m/s", g: "m/s^2", h: "m", k: "N/m", x: "m"}
    assert dimension_of(E, units) == Unit("J")
    with pytest.raises(UnitError) as ei:
        dimension_of(m * v + m * g * h, units)
    assert _code(ei) == "E-UNIT-001"


# ---------------------------------------------------------------------------
# Pharmacokinetics (report §4.1)
# ---------------------------------------------------------------------------


def _bateman(p):
    F, D, ka, ke, V, t = (p.symbol(n) for n in ("F", "D", "ka", "ke", "V", "t"))
    C = F * D * ka / (V * (ka - ke)) * (alkahest.exp(-ke * t) - alkahest.exp(-ka * t))
    units = {F: "1", D: "mg", ka: "1/h", ke: "1/h", V: "L", t: "h"}
    return C, units, (F, D, ka, ke, V, t)


def test_bateman_has_concentration_units():
    p = ExprPool()
    C, units, _ = _bateman(p)
    u = dimension_of(C, units)
    assert u == Unit("mg/L")
    assert u.dimension == "mass/length^3"


def test_bateman_with_inconsistent_time_units_is_refused():
    p = ExprPool()
    C, units, (*_, t) = _bateman(p)
    units[t] = "min"  # ke·t is then (1/h)·min — not a pure number
    with pytest.raises(UnitError) as ei:
        dimension_of(C, units)
    assert _code(ei) == "E-UNIT-004"
    units[t] = "h"
    units.pop(next(k for k in units if str(k) == "V"))
    with pytest.raises(UnitError) as ei:
        dimension_of(C, units)
    assert _code(ei) == "E-UNIT-005"


def test_auc_carries_concentration_times_time():
    """∫_0^∞ C(t) dt = F·D/(V·ke) in mg·h/L — checked against SymPy."""
    sympy = pytest.importorskip("sympy")
    p = ExprPool()
    C, units, (F, D, ka, ke, V, t) = _bateman(p)
    unit = dimension_of(C, units)
    assert unit == Unit("mg/L")
    # The symbolic ∫_0^∞ needs ke, ka > 0; evaluate at a numeric parameter point.
    vals = {F: 1, D: 100, ka: Fraction(3, 2), ke: Fraction(1, 5), V: 20}
    mapping = {k: p.rational(v.numerator, v.denominator) for k, v in vals.items()}
    Cq = Quantity(alkahest.subs(C, mapping), unit)
    tq = Quantity(t, "h")
    auc = U.integrate(Cq, tq, 0, Quantity(p.pos_infinity(), "h"))
    assert auc.unit == Unit("mg*h/L")
    assert str(auc.unit) == "mg*h/L"
    sF, sD, ska, ske, sV, st = sympy.symbols("F D ka ke V t", positive=True)
    Cs = sF * sD * ska / (sV * (ska - ske)) * (sympy.exp(-ske * st) - sympy.exp(-ska * st))
    point = {sF: 1, sD: 100, ska: sympy.Rational(3, 2), ske: sympy.Rational(1, 5), sV: 20}
    ref = sympy.integrate(Cs.subs(point), (st, 0, sympy.oo))
    assert ref == 25  # F·D/(V·ke)
    got = alkahest.simplify(auc.value).value
    assert alkahest.eval_expr(got, {}) == pytest.approx(float(ref), rel=1e-12)
    # and in µg·min/mL the number is 60× larger, exactly
    auc2 = U.convert(auc, "ug*min/mL")
    r, _, _ = Unit("mg*h/L").conversion("ug*min/mL")
    assert r == 60
    assert auc2.unit == Unit("ug*min/mL")


def test_clearance_and_dose_rate():
    CL = Quantity(5, "L/h")
    Css = Quantity(2, "mg/L")
    rate = (CL * Css).simplify_units()
    assert rate.unit == Unit("mg/h")
    assert rate.value == 10
    dose = (rate * Quantity(1, "day")).simplify_units()
    assert U.convert(dose, "g").value == Fraction(240, 1000)


def test_diff_carries_units():
    p = ExprPool()
    t = p.symbol("t")
    x = Quantity(3 * t**2, "m")
    v = U.diff(x, Quantity(t, "s"))
    assert v.unit == Unit("m/s")
    a = U.diff(v, Quantity(t, "s"))
    assert a.unit == Unit("m/s^2")
    assert str(alkahest.simplify(a.value).value) == "6"


def test_integrate_refuses_bare_numeric_bounds_for_dimensional_variable():
    p = ExprPool()
    t = p.symbol("t")
    with pytest.raises(UnitError) as ei:
        U.integrate(Quantity(t, "mg/L"), Quantity(t, "h"), 0, 5)
    assert _code(ei) == "E-UNIT-001"
    q = U.integrate(Quantity(t, "mg/L"), Quantity(t, "h"), 0, Quantity(30, "min"))
    assert q.unit == Unit("mg*h/L")
    assert str(alkahest.simplify(q.value).value) == "1/8"


def test_error_class_and_registry():
    assert issubclass(UnitError, alkahest.AlkahestError)
    with pytest.raises(alkahest.AlkahestError):
        parse_unit("nonsense_unit")


def test_unit_pickles():
    import pickle

    u = Unit("mg*h/L")
    assert pickle.loads(pickle.dumps(u)) == u
    assert hash(Unit("N")) == hash(Unit("kg*m/s^2"))


def test_integrate_accepts_unit_free_infinite_bound():
    p = ExprPool()
    t = p.symbol("t")
    k = Fraction(1, 5)
    q = U.integrate(Quantity(alkahest.exp(-k * t), "mg/L"), Quantity(t, "h"), 0, p.pos_infinity())
    assert q.unit == Unit("mg*h/L")
    assert alkahest.eval_expr(alkahest.simplify(q.value).value, {}) == pytest.approx(5.0)
