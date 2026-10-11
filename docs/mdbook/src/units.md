# Units and dimensional analysis

`alkahest.units` attaches physical units to values and to symbolic formulas,
checks dimensional consistency, and converts between units **exactly**.

```python
from fractions import Fraction
import alkahest
from alkahest import ExprPool, Quantity, Unit, dimension_of
from alkahest import units as U

U.convert(Quantity(1, "inch"), "m").value        # Fraction(127, 5000)
U.convert(Quantity(100, "degC"), "degF").value   # 212
(Quantity(2, "kg") * Quantity(3, "m/s^2")).unit == Unit("N")   # True
Quantity(1, "mg") + Quantity(1, "L")             # UnitError E-UNIT-001
```

## Units

A `Unit` is a product of named units raised to **rational** exponents. Behind
the name it carries a dimension over the seven SI base dimensions (length,
mass, time, current, temperature, amount, luminous intensity) and an exact
scale relative to the coherent SI unit, stored as `ratio · π^k` with a rational
`ratio` — so `1 lbf = 4.4482216152605 N` and `1 deg = π/180 rad` are exact,
and no conversion ever introduces a float.

Unit strings:

| Form | Example |
|---|---|
| product / quotient | `"mg*h/L"`, `"kg m s^-2"`, `"kg*m^2/(s^2*A)"` |
| exponents | `"m/s^2"`, `"m**2"`, `"m s⁻²"`, `"m^(1/2)"` (fractions in parentheses) |
| SI prefixes | `y z a f p n µ/μ/u m c d da h k M G T P E Z Y` on SI units (`mg`, `kPa`, `µL`, `mM`, `kWh`, `keV`) |
| numeric factors | `"1000*m"` |

`/` divides by the single factor that follows it (`a/b*c` is `a·c/b`). A name
followed directly by a digit (`"m2"`) is refused rather than guessed at.

Known units (`alkahest.units.known_units()` lists them): the SI base units and
`Hz N Pa J W C V Ω/ohm F H T Wb S`; `L M(=mol/L) min h day week deg rad sr eV
cal Wh bar Å % ppm`; `mcg cc`; `inch/in ft yd mi lb oz lbf psi atm Torr mmHg mph
gal hp g0`; `K degR delta_degC delta_degF`, and the affine `degC/°C`,
`degF/°F`.

Equality is semantic: `Unit("N") == Unit("kg*m/s^2")`, while `str()` keeps the
spelling. `Unit.to_base()` gives the coherent SI unit in base units
(`J → kg*m^2/s^2`).

## Quantities

`Quantity(value, unit)` pairs an `Expr`, `int`, `Fraction` (or, if you insist,
`float`) with a unit.

* `+` and `-` need equal dimensions — `E-UNIT-001` names both — and convert the
  right operand exactly into the left one's unit
  (`Quantity(1, "h") + Quantity(30, "min")` is `3/2 h`).
* `*`, `/` and `**` with an `int` or `Fraction` exponent combine units.
* `U.exp`, `U.log`, `U.sin`, … (or `U.apply(f, q)`) need a dimensionless
  argument (`E-UNIT-004`); one in `deg` or `%` is converted to a pure number
  first, so `U.sin(Quantity(90, "deg"))` is `sin(π/2)`.
* `convert_units(q, unit)` / `q.convert(unit)`, `to_base_units(q)`, and
  `simplify_units(q)`, which merges factors of the same dimension
  (`mg/kg·kg → mg`, `L/h·min → L`, `km/m → 1`).

### Temperature

`K` and `degR` are ordinary multiplicative units, and so are the difference
units `delta_degC` (= K) and `delta_degF`. `degC` and `degF` are **affine**: their
zero is not absolute zero, so `2 · 20 °C` and `20 °C + 5 °C` have no meaning.
They may only stand alone and be converted, exactly
(`convert_units(Quantity(32, "degF"), "K")` is `5463/20`); any product, power,
sum, negation, or compound unit such as `J/(kg*degC)` raises `E-UNIT-003`.
Write heat capacities with `K` or `delta_degC`.

## Checking a formula

`dimension_of(expr, units)` takes a mapping from each symbol (or any
sub-expression, such as an applied function `C(t)`) to its unit, checks the
whole formula and returns the unit its value is in:

```python
p = ExprPool()
F, D, ka, ke, V, t = (p.symbol(n) for n in ("F", "D", "ka", "ke", "V", "t"))
C = F*D*ka/(V*(ka - ke)) * (alkahest.exp(-ke*t) - alkahest.exp(-ka*t))
units = {F: "1", D: "mg", ka: "1/h", ke: "1/h", V: "L", t: "h"}
dimension_of(C, units)          # Unit('mg/L')

units[t] = "min"
dimension_of(C, units)          # UnitError E-UNIT-004: ke·t is in min/h
```

Rules: the terms of a sum must have the same dimension (`E-UNIT-001`) **and**
the same unit (`E-UNIT-006` — `t_h + t_min` is consistent dimensionally but
its value depends on which number is in which unit); every argument of a
transcendental function and every non-literal exponent must be a pure number
(`E-UNIT-004`); `pi`, `E`, `EulerGamma`, `Catalan`, `GoldenRatio` and `I` are
pure numbers; an exact `0` term fits any sum; a symbol or function with no
known unit is refused (`E-UNIT-005`). `abs`, `re`, `im` keep their argument's
unit, `sqrt`/`cbrt` take a root, `min`/`max` need equal units, `sign` and
`Heaviside` accept anything. Piecewise conditions are not checked, only the
branch values.

## Calculus with units

`U.diff(q, var)` and `U.integrate(q, var, lower, upper)` take the variable as a
`Quantity` whose value is a symbol, and carry the units through:

```python
Cq = Quantity(C, dimension_of(C, units))
auc = U.integrate(Cq, Quantity(t, "h"), 0, Quantity(p.pos_infinity(), "h"))
auc.unit                          # Unit('mg*h/L')   (value F·D/(V·ke))
U.convert(auc, "ug*min/mL")       # exactly 60× the number
```

Bounds are quantities (converted exactly into the variable's unit) or the
unit-free `0` and `±oo`; a bare number such as `5` for a variable in hours is
refused with `E-UNIT-001`, because its unit is unknown.

## Errors

| Code | Meaning |
|---|---|
| `E-UNIT-001` | different dimensions where equal ones are needed (both named) |
| `E-UNIT-002` | malformed unit string or unknown unit name |
| `E-UNIT-003` | an affine temperature (`degC`, `degF`) in arithmetic or a compound unit |
| `E-UNIT-004` | a function argument or symbolic exponent that is not a pure number |
| `E-UNIT-005` | a symbol or function whose unit is not known |
| `E-UNIT-006` | a sum of terms with the same dimension but different units |
| `E-UNIT-007` | a fractional power whose scale has no exact rational root (`km^(1/2)`) |

## Rust

The same engine is `alkahest_cas::units`: `parse_unit`, `Unit`
(`mul`/`div`/`pow`/`conversion`/`to_base`/`simplify`), `Quantity` (arithmetic
over `ExprId`s) and `dimension_of(expr, &HashMap<ExprId, Unit>, &pool)`.
