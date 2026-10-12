//! Physical units and dimensional analysis (G10).
//!
//! A [`Unit`] is a product of named units raised to **rational** exponents.
//! Behind the name it carries
//!
//! * a [`Dimension`] — a rational exponent for each of the seven SI base
//!   dimensions (length, mass, time, current, temperature, amount, luminous
//!   intensity);
//! * an **exact** scale factor relative to the coherent SI unit of that
//!   dimension, stored as `r · π^k` with `r` a positive rational and `k` a
//!   rational — never a float, so `1 inch = 127/5000 m` and
//!   `1 deg = π/180 rad` are exact;
//! * for the two affine temperature scales (`degC`, `degF`) an additive
//!   offset. Those units may only stand alone and only be converted with
//!   [`Quantity::convert`]; putting one inside a product, power or sum is the classic
//!   unit trap (`2 °C · 3` is not `6 °C` of anything) and is refused
//!   (`E-UNIT-003`).
//!
//! A [`Quantity`] is an expression value paired with a unit. Adding or
//! subtracting quantities requires equal dimensions (`E-UNIT-001`, naming both),
//! multiplication, division and rational powers combine units, and
//! [`dimension_of`] checks a whole symbolic formula given the unit of each
//! symbol — every function argument (`exp`, `log`, `sin`, …) must be
//! dimensionless with scale 1 (`E-UNIT-004`); radians are dimensionless.
//!
//! Unit strings are parsed by [`parse_unit`]: `"mg*h/L"`, `"m/s^2"`,
//! `"kg m s^-2"`, `"m^(1/2)"`, `"µg/mL"`, `"kWh"`, `"°C"`.

mod derived;
mod dimension;
mod parse;
mod table;
#[cfg(test)]
mod tests;

pub use derived::DERIVED_UNITS;
pub use dimension::{dimension_of, unify_units, UnifyPolicy};
pub use parse::parse_unit;
pub use table::known_units;

use crate::kernel::{ExprId, ExprPool};
use rug::ops::Pow;
use rug::{Integer, Rational};
use std::fmt;

/// Names of the seven SI base dimensions, in the order [`Dimension`] stores them.
pub const BASE_DIMENSIONS: [&str; 7] = [
    "length",
    "mass",
    "time",
    "current",
    "temperature",
    "amount",
    "luminous_intensity",
];

/// The coherent SI base unit of each dimension in [`BASE_DIMENSIONS`].
pub const BASE_UNITS: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];

/// Order in which [`Unit::to_base`] lists base units (`kg*m^2/s^2`).
const BASE_DISPLAY_ORDER: [usize; 7] = [1, 0, 2, 3, 4, 5, 6];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a units operation was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnitError {
    /// `E-UNIT-001` — an operation needs equal dimensions and got two different
    /// ones (addition, subtraction, comparison, conversion).
    IncompatibleDimensions {
        /// What was being done, e.g. `"add"` or `"convert"`.
        operation: String,
        /// Left unit as written.
        left_unit: String,
        /// Left dimension, e.g. `mass/length^3`.
        left_dimension: String,
        /// Right unit as written.
        right_unit: String,
        /// Right dimension.
        right_dimension: String,
    },
    /// `E-UNIT-002` — a unit string could not be parsed, or names no known unit.
    Parse {
        /// The input string.
        input: String,
        /// What went wrong.
        message: String,
    },
    /// `E-UNIT-003` — an affine temperature unit (`degC`, `degF`) was used in
    /// arithmetic or inside a compound unit.
    AffineUnit {
        /// The affine unit.
        unit: String,
        /// What was attempted.
        context: String,
    },
    /// `E-UNIT-004` — a function argument or a symbolic exponent is not a pure
    /// number (dimensionless with scale 1).
    NotDimensionless {
        /// Where it happened, e.g. ``"argument of exp"``.
        context: String,
        /// The unit found there.
        unit: String,
        /// Its dimension.
        dimension: String,
    },
    /// `E-UNIT-005` — the units of a sub-expression cannot be determined (a
    /// symbol with no unit, an unknown function, an unsupported node).
    UnknownUnit {
        /// The sub-expression, rendered.
        expr: String,
        /// Why.
        reason: String,
    },
    /// `E-UNIT-006` — the terms of a sum have the same dimension but different
    /// units, so the formula's numeric value depends on which unit is meant.
    MixedScales {
        /// The sum, rendered.
        expr: String,
        /// One term's unit.
        left_unit: String,
        /// Another term's unit.
        right_unit: String,
    },
    /// `E-UNIT-007` — a fractional power of a unit whose scale factor has no
    /// exact rational root (e.g. `km^(1/2)`).
    IrrationalScale {
        /// The unit.
        unit: String,
        /// The exponent, rendered.
        exponent: String,
    },
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitError::IncompatibleDimensions {
                operation,
                left_unit,
                left_dimension,
                right_unit,
                right_dimension,
            } => write!(
                f,
                "cannot {operation} `{left_unit}` [{left_dimension}] and `{right_unit}` [{right_dimension}]: the dimensions differ"
            ),
            UnitError::Parse { input, message } => {
                write!(f, "cannot parse unit `{input}`: {message}")
            }
            UnitError::AffineUnit { unit, context } => write!(
                f,
                "`{unit}` is an affine temperature scale (its zero is not absolute zero) and cannot be used in {context}; convert it to K first"
            ),
            UnitError::NotDimensionless {
                context,
                unit,
                dimension,
            } => {
                if dimension == "dimensionless" {
                    write!(
                        f,
                        "the {context} must be a pure number, but it is in `{unit}`, which is dimensionless with scale ≠ 1; convert it (e.g. to rad) first"
                    )
                } else {
                    write!(
                        f,
                        "the {context} must be dimensionless, but it has unit `{unit}` [{dimension}]"
                    )
                }
            }
            UnitError::UnknownUnit { expr, reason } => {
                write!(f, "cannot determine the unit of `{expr}`: {reason}")
            }
            UnitError::MixedScales {
                expr,
                left_unit,
                right_unit,
            } => write!(
                f,
                "the terms of `{expr}` have the same dimension but different units (`{left_unit}` and `{right_unit}`), so the formula's value depends on which unit each number is in; convert to a common unit first"
            ),
            UnitError::IrrationalScale { unit, exponent } => write!(
                f,
                "`({unit})^({exponent})` has an irrational scale factor relative to SI, which cannot be represented exactly; rewrite it in a unit whose scale has an exact root"
            ),
        }
    }
}

impl std::error::Error for UnitError {}

impl crate::errors::AlkahestError for UnitError {
    fn code(&self) -> &'static str {
        match self {
            UnitError::IncompatibleDimensions { .. } => "E-UNIT-001",
            UnitError::Parse { .. } => "E-UNIT-002",
            UnitError::AffineUnit { .. } => "E-UNIT-003",
            UnitError::NotDimensionless { .. } => "E-UNIT-004",
            UnitError::UnknownUnit { .. } => "E-UNIT-005",
            UnitError::MixedScales { .. } => "E-UNIT-006",
            UnitError::IrrationalScale { .. } => "E-UNIT-007",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            UnitError::IncompatibleDimensions { .. } => {
                "only quantities of the same dimension can be added, compared or converted; check the formula or the units assigned to its symbols"
            }
            UnitError::Parse { .. } => {
                "write units as a product such as \"mg*h/L\", \"m/s^2\" or \"kg m s^-2\"; exponents follow ^ or **, fractional ones in parentheses: \"m^(1/2)\""
            }
            UnitError::AffineUnit { .. } => {
                "°C and °F are affine scales; convert to K (or use delta_degC / delta_degF for temperature differences) before doing arithmetic"
            }
            UnitError::NotDimensionless { .. } => {
                "exp, log, trigonometric and other transcendental functions need a pure-number argument; divide by a reference quantity, or convert angles to rad"
            }
            UnitError::UnknownUnit { .. } => {
                "give every symbol (or applied function such as C(t)) a unit in the units mapping"
            }
            UnitError::MixedScales { .. } => {
                "assign the same unit to every symbol of one dimension (e.g. all times in h), or let alkahest.units.unify_units convert the mixed sum exactly to a common unit"
            }
            UnitError::IrrationalScale { .. } => {
                "use the coherent SI unit (e.g. m^(1/2) instead of km^(1/2)) for fractional powers"
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Dimension
// ---------------------------------------------------------------------------

/// Rational exponents over the seven SI base dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Dimension {
    exps: [Rational; 7],
}

impl Dimension {
    /// The dimension of a pure number.
    pub fn dimensionless() -> Self {
        Self::default()
    }

    /// The dimension with integer exponents `e` (order of [`BASE_DIMENSIONS`]).
    pub fn from_ints(e: [i32; 7]) -> Self {
        Dimension {
            exps: e.map(Rational::from),
        }
    }

    /// The exponent of base dimension `i` (order of [`BASE_DIMENSIONS`]).
    pub fn exponent(&self, i: usize) -> &Rational {
        &self.exps[i]
    }

    /// `true` for a pure number.
    pub fn is_dimensionless(&self) -> bool {
        self.exps.iter().all(|e| *e == 0)
    }

    fn mul(&self, other: &Dimension) -> Dimension {
        let mut exps = self.exps.clone();
        for (a, b) in exps.iter_mut().zip(other.exps.iter()) {
            *a += b;
        }
        Dimension { exps }
    }

    fn pow(&self, q: &Rational) -> Dimension {
        let mut exps = self.exps.clone();
        for a in exps.iter_mut() {
            *a *= q;
        }
        Dimension { exps }
    }
}

fn fmt_exponent(e: &Rational) -> String {
    if *e.denom() == 1 {
        e.numer().to_string()
    } else {
        format!("({}/{})", e.numer(), e.denom())
    }
}

/// Render `name^e` products as `a*b^2/(c*d^(1/2))`-style text.
fn fmt_product<'a>(items: impl Iterator<Item = (&'a str, &'a Rational)>, unit_one: &str) -> String {
    let mut num: Vec<String> = Vec::new();
    let mut den: Vec<String> = Vec::new();
    for (name, e) in items {
        if *e == 0 {
            continue;
        }
        let (target, a) = if *e > 0 {
            (&mut num, e.clone())
        } else {
            (&mut den, Rational::from(-e))
        };
        if a == 1 {
            target.push(name.to_string());
        } else {
            target.push(format!("{name}^{}", fmt_exponent(&a)));
        }
    }
    let n = if num.is_empty() {
        unit_one.to_string()
    } else {
        num.join("*")
    };
    match den.len() {
        0 => n,
        1 => format!("{n}/{}", den[0]),
        _ => format!("{n}/({})", den.join("*")),
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "dimensionless");
        }
        let s = fmt_product(BASE_DIMENSIONS.iter().copied().zip(self.exps.iter()), "1");
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Scale
// ---------------------------------------------------------------------------

/// An exact positive scale factor `ratio · π^pi_power`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Scale {
    /// The rational part (positive).
    pub ratio: Rational,
    /// The power of π.
    pub pi_power: Rational,
}

impl Scale {
    /// The factor 1.
    pub fn one() -> Self {
        Scale {
            ratio: Rational::from(1),
            pi_power: Rational::new(),
        }
    }

    /// `true` when the factor is exactly 1.
    pub fn is_one(&self) -> bool {
        self.ratio == 1 && self.pi_power == 0
    }

    fn mul(&self, o: &Scale) -> Scale {
        Scale {
            ratio: Rational::from(&self.ratio * &o.ratio),
            pi_power: Rational::from(&self.pi_power + &o.pi_power),
        }
    }

    fn div(&self, o: &Scale) -> Scale {
        Scale {
            ratio: Rational::from(&self.ratio / &o.ratio),
            pi_power: Rational::from(&self.pi_power - &o.pi_power),
        }
    }

    /// `self^q`, or `None` when the rational part has no exact `q`-th power.
    fn pow(&self, q: &Rational) -> Option<Scale> {
        let ratio = rational_pow(&self.ratio, q)?;
        Some(Scale {
            ratio,
            pi_power: Rational::from(&self.pi_power * q),
        })
    }

    /// Build the factor as an expression in `pool`.
    pub fn to_expr(&self, pool: &ExprPool) -> ExprId {
        let r = rational_expr(pool, &self.ratio);
        if self.pi_power == 0 {
            return r;
        }
        let pi = pool.symbol("pi", crate::kernel::Domain::Real);
        let pe = pool.pow(pi, rational_expr(pool, &self.pi_power));
        if self.ratio == 1 {
            pe
        } else {
            pool.mul(vec![r, pe])
        }
    }
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ratio)?;
        if self.pi_power != 0 {
            write!(f, "*pi^{}", fmt_exponent(&self.pi_power))?;
        }
        Ok(())
    }
}

/// An integer or rational literal node for `r`.
pub(crate) fn rational_expr(pool: &ExprPool, r: &Rational) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

/// Exact `n`-th root of a non-negative integer, if it exists.
fn exact_root(x: &Integer, n: u32) -> Option<Integer> {
    if n == 1 {
        return Some(x.clone());
    }
    let r = x.clone().root(n);
    if Pow::pow(r.clone(), n) == *x {
        Some(r)
    } else {
        None
    }
}

/// Exact `x^q` for positive rational `x`, if rational.
fn rational_pow(x: &Rational, q: &Rational) -> Option<Rational> {
    let den = q.denom().to_u32()?;
    let num = q.numer().clone();
    let nr = exact_root(x.numer(), den)?;
    let dr = exact_root(x.denom(), den)?;
    let root = Rational::from((nr, dr));
    let abs = num.clone().abs().to_u32()?;
    let p: Rational = Pow::pow(root, abs);
    Some(if num < 0 { p.recip() } else { p })
}

// ---------------------------------------------------------------------------
// Unit
// ---------------------------------------------------------------------------

/// A physical unit: a product of named units with rational exponents, an exact
/// scale relative to SI, and a [`Dimension`].
///
/// Equality (`==`) is **semantic**: two units are equal when they have the same
/// dimension, scale and offset, whatever they are called (`N == kg*m/s^2`).
#[derive(Clone, Debug)]
pub struct Unit {
    factors: Vec<(String, Rational)>,
    scale: Scale,
    dim: Dimension,
    /// For affine temperature units: `K = scale·x + offset`.
    offset: Option<Rational>,
}

impl PartialEq for Unit {
    fn eq(&self, o: &Unit) -> bool {
        self.dim == o.dim && self.scale == o.scale && self.offset == o.offset
    }
}

impl Eq for Unit {}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Unit {
    /// The unit of a pure number (written `1`).
    pub fn dimensionless() -> Self {
        Unit {
            factors: Vec::new(),
            scale: Scale::one(),
            dim: Dimension::dimensionless(),
            offset: None,
        }
    }

    /// Parse a unit string; see [`parse_unit`].
    pub fn parse(s: &str) -> Result<Unit, UnitError> {
        parse_unit(s)
    }

    pub(crate) fn atom(name: &str, scale: Scale, dim: Dimension, offset: Option<Rational>) -> Unit {
        Unit {
            factors: vec![(name.to_string(), Rational::from(1))],
            scale,
            dim,
            offset,
        }
    }

    /// The unit's name as a product, e.g. `mg*h/L`; `1` when dimensionless and
    /// unnamed.
    pub fn name(&self) -> String {
        fmt_product(self.factors.iter().map(|(n, e)| (n.as_str(), e)), "1")
    }

    /// The named factors and their exponents.
    pub fn factors(&self) -> &[(String, Rational)] {
        &self.factors
    }

    /// The dimension.
    pub fn dimension(&self) -> &Dimension {
        &self.dim
    }

    /// The exact scale relative to the coherent SI unit of the same dimension.
    pub fn scale(&self) -> &Scale {
        &self.scale
    }

    /// `true` for `degC`/`degF`, whose zero is not absolute zero.
    pub fn is_affine(&self) -> bool {
        self.offset.is_some()
    }

    /// The affine offset in kelvin (`K = scale·x + offset`), if any.
    pub fn offset(&self) -> Option<&Rational> {
        self.offset.as_ref()
    }

    /// `true` when this is a pure number with scale 1 (e.g. `1`, `rad`, `m/m`).
    pub fn is_pure_number(&self) -> bool {
        self.dim.is_dimensionless() && self.scale.is_one() && self.offset.is_none()
    }

    fn refuse_affine(&self, context: &str) -> Result<(), UnitError> {
        if self.is_affine() {
            Err(UnitError::AffineUnit {
                unit: self.name(),
                context: context.to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn merged_factors(&self, other: &[(String, Rational)], sign: i32) -> Vec<(String, Rational)> {
        let mut out = self.factors.clone();
        for (n, e) in other {
            let e = Rational::from(e * sign);
            if let Some(slot) = out.iter_mut().find(|(m, _)| m == n) {
                slot.1 += e;
            } else {
                out.push((n.clone(), e));
            }
        }
        out.retain(|(_, e)| *e != 0);
        out
    }

    /// `self · other`. Refuses affine units (`E-UNIT-003`).
    pub fn mul(&self, other: &Unit) -> Result<Unit, UnitError> {
        self.refuse_affine("a product")?;
        other.refuse_affine("a product")?;
        Ok(Unit {
            factors: self.merged_factors(&other.factors, 1),
            scale: self.scale.mul(&other.scale),
            dim: self.dim.mul(&other.dim),
            offset: None,
        })
    }

    /// `self / other`. Refuses affine units (`E-UNIT-003`).
    pub fn div(&self, other: &Unit) -> Result<Unit, UnitError> {
        self.refuse_affine("a quotient")?;
        other.refuse_affine("a quotient")?;
        Ok(Unit {
            factors: self.merged_factors(&other.factors, -1),
            scale: self.scale.div(&other.scale),
            dim: self.dim.mul(&other.dim.pow(&Rational::from(-1))),
            offset: None,
        })
    }

    /// `self^q` for rational `q`. Refuses affine units (`E-UNIT-003`) and a
    /// scale with no exact rational `q`-th power (`E-UNIT-007`).
    pub fn pow(&self, q: &Rational) -> Result<Unit, UnitError> {
        if *q == 1 {
            return Ok(self.clone());
        }
        self.refuse_affine("a power")?;
        let scale = self
            .scale
            .pow(q)
            .ok_or_else(|| UnitError::IrrationalScale {
                unit: self.name(),
                exponent: fmt_exponent(q),
            })?;
        let mut factors = self.factors.clone();
        for (_, e) in factors.iter_mut() {
            *e *= q;
        }
        factors.retain(|(_, e)| *e != 0);
        Ok(Unit {
            factors,
            scale,
            dim: self.dim.pow(q),
            offset: None,
        })
    }

    fn incompatible(&self, other: &Unit, operation: &str) -> UnitError {
        UnitError::IncompatibleDimensions {
            operation: operation.to_string(),
            left_unit: self.name(),
            left_dimension: self.dim.to_string(),
            right_unit: other.name(),
            right_dimension: other.dim.to_string(),
        }
    }

    /// `Ok(())` when `self` and `other` have the same dimension, else
    /// `E-UNIT-001` naming both (with `operation` in the message).
    pub fn check_same_dimension(&self, other: &Unit, operation: &str) -> Result<(), UnitError> {
        if self.dim == other.dim {
            Ok(())
        } else {
            Err(self.incompatible(other, operation))
        }
    }

    /// The exact factor `f` with `x [self] = f·x [target]` for multiplicative
    /// units. Refuses different dimensions (`E-UNIT-001`) and affine units
    /// (`E-UNIT-003`; use [`Unit::conversion`]).
    pub fn conversion_factor(&self, target: &Unit) -> Result<Scale, UnitError> {
        self.check_same_dimension(target, "convert")?;
        self.refuse_affine("a multiplicative conversion")?;
        target.refuse_affine("a multiplicative conversion")?;
        Ok(self.scale.div(&target.scale))
    }

    /// The exact affine map `x [self] = factor·x + shift [target]`, valid for
    /// every unit pair of equal dimension (`shift` is zero unless an affine
    /// temperature unit is involved). Refuses different dimensions
    /// (`E-UNIT-001`).
    pub fn conversion(&self, target: &Unit) -> Result<(Scale, Rational), UnitError> {
        self.check_same_dimension(target, "convert")?;
        let factor = self.scale.div(&target.scale);
        let zero = Rational::new();
        let o1 = self.offset.as_ref().unwrap_or(&zero);
        let o2 = target.offset.as_ref().unwrap_or(&zero);
        // K = s1·x + o1 = s2·y + o2  ⇒  y = (s1/s2)·x + (o1 − o2)/s2.
        // Affine units are temperature-only, so π never enters the shift.
        let shift = Rational::from(o1 - o2) / &target.scale.ratio;
        Ok((factor, shift))
    }

    /// The coherent SI unit of the same dimension, written in base units
    /// (`kg*m^2/s^2` for `J`). Affine units map to `K`.
    pub fn to_base(&self) -> Unit {
        let mut factors = Vec::new();
        for &i in BASE_DISPLAY_ORDER.iter() {
            let e = self.dim.exponent(i);
            if *e != 0 {
                factors.push((BASE_UNITS[i].to_string(), e.clone()));
            }
        }
        Unit {
            factors,
            scale: Scale::one(),
            dim: self.dim.clone(),
            offset: None,
        }
    }

    /// Merge factors of equal dimension into the first one of that dimension
    /// (`mg/kg*g → mg`, `km/m → 1`), and fold numeric literal factors.
    ///
    /// Returns `(factor, unit)` with `x [self] = factor·x [unit]`. Affine units
    /// are returned unchanged.
    pub fn simplify(&self) -> (Scale, Unit) {
        if self.is_affine() || self.factors.len() < 2 {
            return (Scale::one(), self.clone());
        }
        // Resolve each factor back to an atomic unit.
        let mut groups: Vec<(String, Unit, Rational)> = Vec::new();
        for (name, e) in &self.factors {
            let Ok(atom) = parse::parse_atom(name) else {
                return (Scale::one(), self.clone());
            };
            if name.starts_with(|c: char| c.is_ascii_digit()) {
                // numeric literal — folded entirely
                continue;
            }
            if !atom.dim.is_dimensionless() {
                if let Some(g) = groups.iter_mut().find(|g| g.1.dim == atom.dim) {
                    g.2 += e;
                    continue;
                }
                if let Some(g) = groups
                    .iter_mut()
                    .find(|g| g.1.dim == atom.dim.pow(&Rational::from(-1)))
                {
                    g.2 -= e;
                    continue;
                }
            }
            groups.push((name.clone(), atom, e.clone()));
        }
        let mut out = Unit::dimensionless();
        for (_, atom, e) in &groups {
            if *e == 0 {
                continue;
            }
            match atom.pow(e).and_then(|p| out.mul(&p)) {
                Ok(u) => out = u,
                Err(_) => return (Scale::one(), self.clone()),
            }
        }
        debug_assert_eq!(out.dim, self.dim);
        (self.scale.div(&out.scale), out)
    }
}

// ---------------------------------------------------------------------------
// Quantity
// ---------------------------------------------------------------------------

/// A value (any expression) together with its [`Unit`].
#[derive(Clone, Debug)]
pub struct Quantity {
    /// The numeric or symbolic magnitude.
    pub value: ExprId,
    /// The unit the magnitude is expressed in.
    pub unit: Unit,
}

fn scaled(pool: &ExprPool, x: ExprId, f: &Scale) -> ExprId {
    if f.is_one() {
        x
    } else {
        pool.mul(vec![f.to_expr(pool), x])
    }
}

impl Quantity {
    /// Pair `value` with `unit`.
    pub fn new(value: ExprId, unit: Unit) -> Self {
        Quantity { value, unit }
    }

    fn arith_guard(&self, other: &Quantity, what: &str) -> Result<(), UnitError> {
        self.unit.refuse_affine(what)?;
        other.unit.refuse_affine(what)
    }

    /// `self + other`, in `self`'s unit (`other` is converted exactly).
    /// Refuses different dimensions (`E-UNIT-001`) and affine units
    /// (`E-UNIT-003`).
    pub fn add(&self, other: &Quantity, pool: &ExprPool) -> Result<Quantity, UnitError> {
        self.unit.check_same_dimension(&other.unit, "add")?;
        self.arith_guard(other, "a sum")?;
        let f = other.unit.conversion_factor(&self.unit)?;
        let rhs = scaled(pool, other.value, &f);
        Ok(Quantity::new(
            pool.add(vec![self.value, rhs]),
            self.unit.clone(),
        ))
    }

    /// `self − other`, in `self`'s unit. Same refusals as [`Quantity::add`].
    pub fn sub(&self, other: &Quantity, pool: &ExprPool) -> Result<Quantity, UnitError> {
        self.unit.check_same_dimension(&other.unit, "subtract")?;
        self.arith_guard(other, "a difference")?;
        let f = other.unit.conversion_factor(&self.unit)?;
        let rhs = scaled(pool, other.value, &f);
        let neg = pool.mul(vec![pool.integer(-1), rhs]);
        Ok(Quantity::new(
            pool.add(vec![self.value, neg]),
            self.unit.clone(),
        ))
    }

    /// `self · other`; the units multiply.
    pub fn mul(&self, other: &Quantity, pool: &ExprPool) -> Result<Quantity, UnitError> {
        let unit = self.unit.mul(&other.unit)?;
        Ok(Quantity::new(pool.mul(vec![self.value, other.value]), unit))
    }

    /// `self / other`; the units divide.
    pub fn div(&self, other: &Quantity, pool: &ExprPool) -> Result<Quantity, UnitError> {
        let unit = self.unit.div(&other.unit)?;
        let inv = pool.pow(other.value, pool.integer(-1));
        Ok(Quantity::new(pool.mul(vec![self.value, inv]), unit))
    }

    /// `self^q` for a rational exponent.
    pub fn pow(&self, q: &Rational, pool: &ExprPool) -> Result<Quantity, UnitError> {
        let unit = self.unit.pow(q)?;
        Ok(Quantity::new(
            pool.pow(self.value, rational_expr(pool, q)),
            unit,
        ))
    }

    /// Express in `target` (exact; affine temperature scales allowed).
    pub fn convert(&self, target: &Unit, pool: &ExprPool) -> Result<Quantity, UnitError> {
        let (f, shift) = self.unit.conversion(target)?;
        let mut v = scaled(pool, self.value, &f);
        if shift != 0 {
            v = pool.add(vec![v, rational_expr(pool, &shift)]);
        }
        Ok(Quantity::new(v, target.clone()))
    }

    /// Express in coherent SI base units.
    pub fn to_base(&self, pool: &ExprPool) -> Result<Quantity, UnitError> {
        self.convert(&self.unit.to_base(), pool)
    }

    /// Merge same-dimension factors of the unit (see [`Unit::simplify`]).
    pub fn simplify_units(&self, pool: &ExprPool) -> Quantity {
        let (f, unit) = self.unit.simplify();
        Quantity::new(scaled(pool, self.value, &f), unit)
    }
}
