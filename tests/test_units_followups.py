"""F18 — units follow-ups: mixed-unit sums, derived-unit naming, units through
Derivative nodes, logarithmic-scale conversions.

Oracles: ``pint`` when installed, an exact hand-checked table (SI brochure)
otherwise, and ``math`` for the logarithmic scales.
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
# (1) Mixed-unit sums are converted, not refused
# ---------------------------------------------------------------------------


def test_dimension_of_still_refuses_hours_plus_minutes():
    p = ExprPool()
    a, b = p.symbol("a"), p.symbol("b")
    with pytest.raises(UnitError) as ei:
        dimension_of(a + b, {a: "h", b: "min"})
    assert _code(ei) == "E-UNIT-006"


@pytest.mark.parametrize(
    ("prefer", "unit", "expected_per_h_min"),
    [
        ("finest", "min", lambda ha, mb: 60 * ha + mb),
        ("coarsest", "h", lambda ha, mb: ha + Fraction(mb, 60)),
    ],
)
def test_unify_hours_plus_minutes(prefer, unit, expected_per_h_min):
    p = ExprPool()
    a, b = p.symbol("a"), p.symbol("b")
    q = U.unify_units(a + b, {a: "h", b: "min"}, prefer=prefer)
    assert q.unit == Unit(unit)
    assert str(q.unit) == unit
    for ha, mb in [(2, 30), (0, 5), (Fraction(3, 2), 7)]:
        got = alkahest.eval_expr(q.value, {a: float(ha), b: float(mb)})
        assert got == pytest.approx(float(expected_per_h_min(ha, mb)), rel=1e-12)


def test_unify_matches_pint():
    pint = pytest.importorskip("pint")
    ureg = pint.UnitRegistry()
    p = ExprPool()
    a, b = p.symbol("a"), p.symbol("b")
    q = U.unify_units(a + b, {a: "h", b: "min"})
    for ha, mb in [(2, 30), (7, 11)]:
        total = ureg.Quantity(ha, "hour") + ureg.Quantity(mb, "minute")
        want = total.to("minute").magnitude
        assert alkahest.eval_expr(q.value, {a: float(ha), b: float(mb)}) == pytest.approx(want)


def test_unify_first_policy_and_mixed_prefix():
    p = ExprPool()
    x, y, z = p.symbol("x"), p.symbol("y"), p.symbol("z")
    # canonical order decides "first"; the result must be consistent either way
    q = U.unify_units(x + y + z, {x: "km", y: "m", z: "mm"}, prefer="first")
    assert q.unit.dimension == "length"
    vals = {x: 1.0, y: 1.0, z: 1.0}
    total_m = 1000.0 + 1.0 + 0.001
    ratio = float(Unit("m").conversion(q.unit)[0])
    assert alkahest.eval_expr(q.value, vals) == pytest.approx(total_m * ratio, rel=1e-12)
    q = U.unify_units(x + y + z, {x: "km", y: "m", z: "mm"})
    assert str(q.unit) == "mm"
    assert alkahest.eval_expr(q.value, vals) == pytest.approx(1_001_001.0)


def test_unify_nested_piecewise_and_max():
    p = ExprPool()
    a, b, m = p.symbol("a"), p.symbol("b"), p.symbol("m")
    q = U.unify_units(m * (a + b), {a: "h", b: "min", m: "kg"})
    assert q.unit == Unit("kg*min")
    q = U.unify_units((a + b) ** 2, {a: "h", b: "min"})
    assert q.unit == Unit("min^2")
    # sqrt of a minute-valued sum has an irrational scale: still refused
    with pytest.raises(UnitError) as ei:
        U.unify_units(alkahest.sqrt(a + b), {a: "h", b: "min"})
    assert _code(ei) == "E-UNIT-007"
    # h has an exact square root (3600 = 60^2), so this one is fine
    q = U.unify_units(alkahest.sqrt(a + b), {a: "h", b: "h"})
    assert q.unit == Unit("h") ** Fraction(1, 2)


def test_unify_leaves_clean_formulas_alone_and_keeps_refusals():
    p = ExprPool()
    a, b, x, c, k = (p.symbol(n) for n in "abxck")
    e = a + b
    q = U.unify_units(e, {a: "h", b: "h"})
    assert q.value == e
    assert str(q.unit) == "h"
    with pytest.raises(UnitError) as ei:
        U.unify_units(a + x, {a: "h", x: "m"})
    assert _code(ei) == "E-UNIT-001"
    with pytest.raises(UnitError) as ei:
        U.unify_units(c + k, {c: "degC", k: "degF"})
    assert _code(ei) == "E-UNIT-003"
    with pytest.raises(UnitError) as ei:
        U.unify_units(a + b, {a: "h"})
    assert _code(ei) == "E-UNIT-005"
    with pytest.raises(UnitError) as ei:
        U.unify_units(alkahest.sin(a + b), {a: "h", b: "min"})
    assert _code(ei) == "E-UNIT-004"
    with pytest.raises(ValueError):
        U.unify_units(a + b, {a: "h", b: "min"}, prefer="nope")
    # irrational scale stays refused
    with pytest.raises(UnitError) as ei:
        Unit("km") ** Fraction(1, 2)
    assert _code(ei) == "E-UNIT-007"


# ---------------------------------------------------------------------------
# (2) Derived-unit naming
# ---------------------------------------------------------------------------

DERIVED = [
    ("kg*m/s^2", "N"),
    ("kg/(m*s^2)", "Pa"),
    ("kg*m^2/s^2", "J"),
    ("N*m", "J"),
    ("kg*m^2/s^3", "W"),
    ("A*s", "C"),
    ("kg*m^2/(A*s^3)", "V"),
    ("kg*m^2/(A^2*s^3)", "ohm"),
    ("A^2*s^4/(kg*m^2)", "F"),
    ("kg*m^2/(A^2*s^2)", "H"),
    ("kg/(A*s^2)", "T"),
    ("kg*m^2/(A*s^2)", "Wb"),
    ("A^2*s^3/(kg*m^2)", "S"),
    ("1/s", "Hz"),
    ("g*m/s^2", "mN"),
    ("kg*km/s^2", "kN"),
    ("g*m^2/s^2", "mJ"),
]


@pytest.mark.parametrize(("src", "name"), DERIVED)
def test_derived_naming_table(src, name):
    q = U.simplify_units(Quantity(3, src))
    assert str(q.unit) == name
    # exact: the value is unchanged and the unit is semantically the original
    assert q.value == 3
    assert q.unit == Unit(src)


@pytest.mark.parametrize(("src", "name"), DERIVED)
def test_derived_naming_matches_pint(src, name):
    pint = pytest.importorskip("pint")
    ureg = pint.UnitRegistry()

    def p(u):
        return u.replace("^", "**").replace("ohm", "ohm").replace("µ", "u")

    got = ureg.Quantity(1, p(src)).to(p(name)).magnitude
    assert got == pytest.approx(1.0, rel=1e-12)


def test_derived_naming_value_scales_exactly():
    # 2 g * 3 m / s^2 is 6 mN; the value stays 6 and the unit is mN
    q = U.simplify_units(Quantity(2, "g") * Quantity(3, "m/s^2"))
    assert str(q.unit) == "mN"
    assert q.value == 6
    q = U.simplify_units(Quantity(5, "mg/kg") * Quantity(70, "kg"))
    assert str(q.unit) == "mg"
    assert q.value == 350


@pytest.mark.parametrize(
    "src", ["kg*cm/s^2", "rad/s", "m/s", "N", "kPa", "mg*h/L", "m^2/s^2", "deg/s"]
)
def test_derived_naming_leaves_inexact_or_other_units(src):
    q = U.simplify_units(Quantity(3, src))
    # (merging of same-dimension factors may change the spelling, never to a derived name)
    assert str(q.unit) not in {"Hz", "Gy", "Sv", "Bq"} or src == "1/s"
    if src in {"N", "kPa", "m/s", "rad/s", "m^2/s^2", "deg/s", "mg*h/L"}:
        assert str(q.unit) == src
    assert q.unit == Unit(src)


def test_derived_opt_out_and_allow_list():
    q = Quantity(3, "kg*m/s^2")
    assert str(U.simplify_units(q, derived=False).unit) == "kg*m/s^2"
    assert str(q.simplify_units(derived=False).unit) == "kg*m/s^2"
    assert str(U.simplify_units(q, allowed=["J", "W"]).unit) == "kg*m/s^2"
    assert str(U.simplify_units(q, allowed=["N"]).unit) == "N"
    # Unit-level API keeps the old default
    assert str(Unit("kg*m/s^2").simplify()[2]) == "kg*m/s^2"
    assert str(Unit("kg*m/s^2").simplify(derived=True)[2]) == "N"
    assert Unit("kg*m/s^2").to_derived() == Unit("N")
    assert str(Unit("kg*m/s^2").to_derived()) == "N"
    assert Unit("m/s").to_derived() is None
    assert Unit("kg*cm/s^2").to_derived() is None


# ---------------------------------------------------------------------------
# (3) Units through Derivative nodes
# ---------------------------------------------------------------------------


def test_dimension_of_derivative_nodes():
    p = ExprPool()
    t, x, mass = p.symbol("t"), p.symbol("x"), p.symbol("mass")
    f = p.func("f", [t])
    d1 = alkahest.diff(f, t).value
    d2 = alkahest.diff(d1, t).value
    assert "Derivative" in str(d1)
    units = {t: "s", f: "m", mass: "kg"}
    assert dimension_of(d1, units) == Unit("m/s")
    assert dimension_of(d2, units) == Unit("m/s^2")
    # Newton's second law as an equation of units
    assert dimension_of(mass * d2, units) == Unit("N")
    # kinetic + potential-style sums need matching dimensions
    with pytest.raises(UnitError) as ei:
        dimension_of(d1 + f, units)
    assert _code(ei) == "E-UNIT-001"
    # minutes scale the unit
    assert dimension_of(d1, {t: "min", f: "m"}) == Unit("m/min")
    # partials
    g = p.func("g", [x, t])
    gx = alkahest.diff(g, x).value
    gxt = alkahest.diff(gx, t).value
    gxx = alkahest.diff(gx, x).value
    units = {t: "s", x: "m", g: "K"}
    assert dimension_of(gx, units) == Unit("K/m")
    assert dimension_of(gxt, units) == Unit("K/(m*s)")
    assert dimension_of(gxx, units) == Unit("K/m^2")
    # the heat equation  g_t - alpha g_xx  is consistent with alpha in m^2/s
    alpha = p.symbol("alpha")
    gt = alkahest.diff(g, t).value
    heat = gt - alpha * gxx
    assert dimension_of(heat, {**units, alpha: "m^2/s"}) == Unit("K/s")
    with pytest.raises(UnitError) as ei:
        dimension_of(heat, {**units, alpha: "m/s"})
    assert _code(ei) == "E-UNIT-001"


def test_derivative_of_unknown_function_unit_is_refused():
    p = ExprPool()
    t = p.symbol("t")
    f = p.func("f", [t])
    d1 = alkahest.diff(f, t).value
    with pytest.raises(UnitError) as ei:
        dimension_of(d1, {t: "s"})
    assert _code(ei) == "E-UNIT-005"


def test_derivative_unit_matches_quantity_diff():
    # the node-level rule agrees with units.diff on a concrete symbol
    p = ExprPool()
    t, x = p.symbol("t"), p.symbol("x")
    d = U.diff(Quantity(x**2, "m"), Quantity(t, "s"))
    assert d.unit == Unit("m/s")


# ---------------------------------------------------------------------------
# (4) Logarithmic scales: explicit conversion functions only
# ---------------------------------------------------------------------------


def test_log_units_are_not_units():
    for name in ("dB", "Np", "dBm"):
        with pytest.raises(UnitError) as ei:
            parse_unit(name)
        assert _code(ei) == "E-UNIT-002"
        assert "logarithmic" in str(ei.value)
    # pH reads as the SI picohenry; it is documented, not a pH scale
    assert parse_unit("pH").dimension == parse_unit("H").dimension


@pytest.mark.parametrize(
    ("db", "kind", "want"),
    [
        (0, "power", 1),
        (10, "power", 10),
        (30, "power", 1000),
        (20, "field", 10),
        (-20, "power", Fraction(1, 100)),
        (-40, "field", Fraction(1, 100)),
    ],
)
def test_db_to_ratio_exact(db, kind, want):
    assert U.db_to_ratio(db, kind) == want


def test_db_roundtrip_and_symbolic():
    p = ExprPool()
    assert U.ratio_to_db(1000) == 30
    assert U.ratio_to_db(100, "field") == 40
    assert U.ratio_to_db(Fraction(1, 100)) == -20
    assert U.ratio_to_db(1) == 0
    # 3 dB is ten^(3/10): symbolic, and numerically 1.9952623...
    r = U.db_to_ratio(3)
    assert not isinstance(r, float)
    assert alkahest.eval_expr(r, {}) == pytest.approx(10**0.3, rel=1e-14)
    # ratio 2 -> 10 log10 2 dB
    d = U.ratio_to_db(2)
    assert alkahest.eval_expr(d, {}) == pytest.approx(10 * math.log10(2), rel=1e-14)
    # symbolic argument
    x = p.symbol("x")
    assert alkahest.eval_expr(U.db_to_ratio(x), {x: 6.0}) == pytest.approx(10**0.6)
    assert alkahest.eval_expr(U.ratio_to_db(x, "field"), {x: 5.0}) == pytest.approx(
        20 * math.log10(5)
    )
    # floats stay floats
    assert U.db_to_ratio(3.0) == pytest.approx(10**0.3)
    with pytest.raises(ValueError):
        U.ratio_to_db(0)
    with pytest.raises(ValueError):
        U.ratio_to_db(-1)
    with pytest.raises(ValueError):
        U.db_to_ratio(3, "bogus")
    # a dimensional value cannot be a level
    with pytest.raises(UnitError) as ei:
        U.db_to_ratio(Quantity(3, "m"))
    assert _code(ei) == "E-UNIT-004"


def test_nepers():
    assert U.np_to_ratio(0) == 1
    assert alkahest.eval_expr(U.np_to_ratio(1), {}) == pytest.approx(math.e)
    assert alkahest.eval_expr(U.np_to_ratio(1, "power"), {}) == pytest.approx(math.e**2)
    assert U.ratio_to_np(1) == 0
    assert alkahest.eval_expr(U.ratio_to_np(2), {}) == pytest.approx(math.log(2))
    assert alkahest.eval_expr(U.ratio_to_np(2, "power"), {}) == pytest.approx(math.log(2) / 2)
    # 1 Np (field) = 20 log10(e) dB
    np_in_db = alkahest.eval_expr(U.ratio_to_db(U.np_to_ratio(1), "field"), {})
    assert np_in_db == pytest.approx(20 * math.log10(math.e))


def test_ph_conversions():
    c = U.ph_to_concentration(7)
    assert c.unit == Unit("mol/L")
    assert c.value == Fraction(1, 10**7)
    assert U.ph_to_concentration(0).value == 1
    assert U.concentration_to_ph(c) == 7
    # any concentration unit converts exactly to mol/L first
    assert U.concentration_to_ph(Quantity(Fraction(1, 10**4), "mmol/mL")) == 4
    assert U.concentration_to_ph(Quantity(Fraction(1, 10**6), "mol/L")) == 6
    assert U.concentration_to_ph(Quantity(1, "mol/m^3")) == 3
    # non-integral pH is symbolic, numerically right
    c = U.ph_to_concentration(Fraction(15, 2))
    assert alkahest.eval_expr(c.value, {}) == pytest.approx(10**-7.5)
    ph = U.concentration_to_ph(Quantity(Fraction(3, 10**5), "mol/L"))
    assert alkahest.eval_expr(ph, {}) == pytest.approx(-math.log10(3e-5), rel=1e-14)
    with pytest.raises(UnitError) as ei:
        U.concentration_to_ph(Quantity(1, "g"))
    assert _code(ei) == "E-UNIT-001"
    with pytest.raises(ValueError):
        U.concentration_to_ph(Quantity(0, "mol/L"))


def test_new_functions_exported():
    for n in (
        "unify_units",
        "db_to_ratio",
        "ratio_to_db",
        "np_to_ratio",
        "ratio_to_np",
        "ph_to_concentration",
        "concentration_to_ph",
    ):
        assert n in U.__all__
        assert callable(getattr(U, n))
