//! Dimensional analysis of a symbolic formula.

use super::{Unit, UnitError};
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Rational;
use std::collections::HashMap;

/// Named constants that are pure numbers.
const CONSTANT_SYMBOLS: &[&str] = &["pi", "E", "EulerGamma", "Catalan", "GoldenRatio", "I"];

/// Functions whose arguments must be pure numbers (dimensionless, scale 1) and
/// whose value is a pure number.
const PURE_FUNCTIONS: &[&str] = &[
    "exp",
    "log",
    "ln",
    "log10",
    "log2",
    "sin",
    "cos",
    "tan",
    "sec",
    "csc",
    "cot",
    "asin",
    "acos",
    "atan",
    "asec",
    "acsc",
    "acot",
    "sinh",
    "cosh",
    "tanh",
    "sech",
    "csch",
    "coth",
    "asinh",
    "acosh",
    "atanh",
    "asech",
    "acsch",
    "acoth",
    "erf",
    "erfc",
    "erfi",
    "gamma",
    "lgamma",
    "loggamma",
    "digamma",
    "polygamma",
    "beta",
    "zeta",
    "lambertw",
    "LambertW",
    "W",
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    "bessel_j0",
    "bessel_j1",
    "Si",
    "Ci",
    "Ei",
    "li",
    "fresnels",
    "fresnelc",
    "dilog",
    "polylog",
    "bessel_j",
    "bessel_y",
    "bessel_i",
    "bessel_k",
    "spherical_jn",
    "spherical_yn",
    "airy_ai",
    "airy_bi",
    "airy_ai_prime",
    "airy_bi_prime",
    "hurwitz_zeta",
    "legendre_p",
    "assoc_legendre_p",
    "hermite_h",
    "hermite_he",
    "laguerre_l",
    "assoc_laguerre_l",
    "chebyshev_t",
    "chebyshev_u",
    "gegenbauer_c",
    "jacobi_p",
    "factorial",
    "binomial",
    "floor",
    "ceiling",
    "ceil",
    "round",
    "frac",
    "sinc",
    "elliptic_k",
    "elliptic_e",
];

/// Functions of any argument whose value is a pure number.
const SIGN_FUNCTIONS: &[&str] = &["sign", "sgn", "Heaviside", "heaviside", "arg"];

/// Functions whose value carries the unit of their (single) argument.
const SAME_UNIT_FUNCTIONS: &[&str] = &["abs", "Abs", "re", "im", "conjugate"];

struct Walker<'a> {
    pool: &'a ExprPool,
    units: &'a HashMap<ExprId, Unit>,
    memo: HashMap<ExprId, Unit>,
}

impl Walker<'_> {
    fn show(&self, id: ExprId) -> String {
        self.pool.display(id).to_string()
    }

    fn require_pure(&mut self, id: ExprId, context: String) -> Result<(), UnitError> {
        let u = self.unit(id)?;
        if u.is_pure_number() {
            Ok(())
        } else {
            Err(UnitError::NotDimensionless {
                context,
                unit: u.name(),
                dimension: u.dimension().to_string(),
            })
        }
    }

    /// The common unit of `ids`, which must agree in dimension and scale.
    fn common(&mut self, whole: ExprId, ids: &[ExprId], op: &str) -> Result<Unit, UnitError> {
        let mut first: Option<Unit> = None;
        for &t in ids {
            if matches!(self.pool.get(t), ExprData::Integer(ref n) if n.0 == 0) {
                continue; // an exact 0 has every unit
            }
            let u = self.unit(t)?;
            match &first {
                None => first = Some(u),
                Some(f) => {
                    if f.dimension() != u.dimension() {
                        return Err(UnitError::IncompatibleDimensions {
                            operation: format!("{op} (in `{}`)", self.show(whole)),
                            left_unit: f.name(),
                            left_dimension: f.dimension().to_string(),
                            right_unit: u.name(),
                            right_dimension: u.dimension().to_string(),
                        });
                    }
                    if f != &u {
                        return Err(UnitError::MixedScales {
                            expr: self.show(whole),
                            left_unit: f.name(),
                            right_unit: u.name(),
                        });
                    }
                }
            }
        }
        Ok(first.unwrap_or_else(Unit::dimensionless))
    }

    fn unit(&mut self, id: ExprId) -> Result<Unit, UnitError> {
        if let Some(u) = self.units.get(&id) {
            return Ok(u.clone());
        }
        if let Some(u) = self.memo.get(&id) {
            return Ok(u.clone());
        }
        let u = self.compute(id)?;
        self.memo.insert(id, u.clone());
        Ok(u)
    }

    fn unknown(&self, id: ExprId, reason: &str) -> UnitError {
        UnitError::UnknownUnit {
            expr: self.show(id),
            reason: reason.to_string(),
        }
    }

    fn compute(&mut self, id: ExprId) -> Result<Unit, UnitError> {
        match self.pool.get(id) {
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => {
                Ok(Unit::dimensionless())
            }
            ExprData::Symbol { name, .. } => {
                if CONSTANT_SYMBOLS.contains(&name.as_str()) || self.pool.is_imaginary_unit(id) {
                    Ok(Unit::dimensionless())
                } else {
                    Err(self.unknown(id, "the symbol has no unit in the units mapping"))
                }
            }
            ExprData::Add(terms) => self.common(id, &terms, "add"),
            ExprData::Mul(factors) => {
                let mut acc = Unit::dimensionless();
                for f in factors {
                    let u = self.unit(f)?;
                    acc = acc.mul(&u)?;
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp } => {
                let bu = self.unit(base)?;
                let q: Option<Rational> = match self.pool.get(exp) {
                    ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
                    ExprData::Rational(r) => Some(r.0.clone()),
                    _ => None,
                };
                match q {
                    Some(q) => bu.pow(&q),
                    None => {
                        if !bu.is_pure_number() {
                            return Err(UnitError::NotDimensionless {
                                context: format!(
                                    "base of `{}` (an exponent that is not a rational literal needs a pure-number base)",
                                    self.show(id)
                                ),
                                unit: bu.name(),
                                dimension: bu.dimension().to_string(),
                            });
                        }
                        self.require_pure(exp, format!("exponent of `{}`", self.show(id)))?;
                        Ok(Unit::dimensionless())
                    }
                }
            }
            ExprData::Func { name, args } => {
                let n = name.as_str();
                if PURE_FUNCTIONS.contains(&n) {
                    for &a in &args {
                        self.require_pure(a, format!("argument of {n} in `{}`", self.show(id)))?;
                    }
                    Ok(Unit::dimensionless())
                } else if SIGN_FUNCTIONS.contains(&n) {
                    for &a in &args {
                        self.unit(a)?;
                    }
                    Ok(Unit::dimensionless())
                } else if SAME_UNIT_FUNCTIONS.contains(&n) && args.len() == 1 {
                    self.unit(args[0])
                } else if (n == "sqrt" || n == "cbrt") && args.len() == 1 {
                    let q = Rational::from((1, if n == "sqrt" { 2 } else { 3 }));
                    self.unit(args[0])?.pow(&q)
                } else if matches!(n, "min" | "max" | "Min" | "Max") && !args.is_empty() {
                    self.common(id, &args, "compare")
                } else if matches!(n, "atan2") && args.len() == 2 {
                    self.common(id, &args, "compare")?;
                    Ok(Unit::dimensionless())
                } else {
                    Err(self.unknown(
                        id,
                        "the function is not known to the units checker; give the applied function a unit in the units mapping",
                    ))
                }
            }
            ExprData::Piecewise { branches, default } => {
                let mut vals: Vec<ExprId> = branches.iter().map(|b| b.1).collect();
                vals.push(default);
                self.common(id, &vals, "join the branches")
            }
            _ => Err(self.unknown(id, "this kind of expression has no unit")),
        }
    }
}

/// The unit of `expr`, given the unit of each symbol (or of any
/// sub-expression, such as an applied function `C(t)`) in `units`.
///
/// This checks **dimensional consistency** of the whole formula: the terms of
/// every sum must have the same dimension (`E-UNIT-001`, naming both) and the
/// same unit (`E-UNIT-006`); every argument of `exp`, `log`, `sin`, … and every
/// non-literal exponent must be a pure number — dimensionless with scale 1, so
/// an angle must be in `rad` (`E-UNIT-004`); a symbol or function with no
/// known unit is refused (`E-UNIT-005`). `pi`, `E`, `EulerGamma`, `Catalan`,
/// `GoldenRatio` and `I` are pure numbers. An exact `0` term is accepted in
/// any sum. Piecewise conditions are not checked, only the branch values.
///
/// The returned unit is the one the formula's value is in when every symbol
/// is given in its listed unit (e.g. `mg/L` for the Bateman equation with the
/// dose in `mg` and the volume in `L`).
pub fn dimension_of(
    expr: ExprId,
    units: &HashMap<ExprId, Unit>,
    pool: &ExprPool,
) -> Result<Unit, UnitError> {
    let mut w = Walker {
        pool,
        units,
        memo: HashMap::new(),
    };
    w.unit(expr)
}
