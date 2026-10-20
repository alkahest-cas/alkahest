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
    "zeta_prime",
    "hankel1",
    "hankel2",
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

    /// `Derivative(f(u…), k…)`: `unit(f(u…)) / Π unit(uᵢ)^kᵢ`.
    fn derivative(&mut self, id: ExprId) -> Result<Unit, UnitError> {
        let Some(v) = crate::diff::undefined::view(id, self.pool) else {
            return Err(self.unknown(id, "a malformed Derivative node"));
        };
        let mut acc = self.unit(v.inner)?;
        for (&a, &k) in v.args.iter().zip(v.orders.iter()) {
            if k == 0 {
                continue;
            }
            let au = self.unit(a)?;
            acc = acc.div(&au.pow(&Rational::from(k))?)?;
        }
        Ok(acc)
    }

    /// `Integral(f, x)` and `Integral(f, x, a, b)`: `unit(f)·unit(x)`, with the
    /// bounds in the unit of `x`.
    fn integral(&mut self, id: ExprId, args: &[ExprId]) -> Result<Unit, UnitError> {
        if args.len() != 2 && args.len() != 4 {
            return Err(self.unknown(id, "an Integral node needs (f, x) or (f, x, a, b)"));
        }
        let fu = self.unit(args[0])?;
        let xu = self.unit(args[1])?;
        if args.len() == 4 {
            self.common(id, &[args[1], args[2], args[3]], "integrate between bounds")?;
        }
        fu.mul(&xu)
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
                } else if n == crate::diff::undefined::DERIVATIVE_FUNC {
                    self.derivative(id)
                } else if n == "Integral" {
                    self.integral(id, &args)
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

/// Which unit a sum of same-dimension, different-unit terms is converted to by
/// [`unify_units`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnifyPolicy {
    /// The unit with the smallest scale among the terms (`h + min` → `min`):
    /// conversion factors are then the most likely to be integers.
    Finest,
    /// The unit with the largest scale among the terms (`h + min` → `h`).
    Coarsest,
    /// The unit of the first term in the sum's canonical order.
    First,
}

fn scale_value(s: &super::Scale) -> rug::Float {
    use rug::float::Constant;
    use rug::ops::Pow;
    let prec = 256;
    let mut v = rug::Float::with_val(prec, &s.ratio);
    if s.pi_power != 0 {
        let pi = rug::Float::with_val(prec, Constant::Pi);
        let e = rug::Float::with_val(prec, &s.pi_power);
        v *= rug::Float::with_val(prec, pi.pow(&e));
    }
    v
}

struct Unifier<'a> {
    w: Walker<'a>,
    memo: HashMap<ExprId, ExprId>,
    policy: UnifyPolicy,
    /// Every node of the input formula. A converted term `c·t` is in the
    /// target unit while the node `c·t` read at face value would be in `t`'s
    /// unit, so converted terms are pinned in the walker's memo; that is only
    /// sound if the same node does not also occur naturally.
    originals: std::collections::HashSet<ExprId>,
}

fn collect_nodes(id: ExprId, pool: &ExprPool, out: &mut std::collections::HashSet<ExprId>) {
    if !out.insert(id) {
        return;
    }
    match pool.get(id) {
        ExprData::Add(v) | ExprData::Mul(v) => v.iter().for_each(|&c| collect_nodes(c, pool, out)),
        ExprData::Pow { base, exp } => {
            collect_nodes(base, pool, out);
            collect_nodes(exp, pool, out);
        }
        ExprData::Func { args, .. } => args.iter().for_each(|&c| collect_nodes(c, pool, out)),
        ExprData::Piecewise { branches, default } => {
            for (c, v) in branches {
                collect_nodes(c, pool, out);
                collect_nodes(v, pool, out);
            }
            collect_nodes(default, pool, out);
        }
        _ => {}
    }
}

impl Unifier<'_> {
    /// Convert each term to the policy's unit of its dimension. `whole` names
    /// the sum in errors.
    fn unify_terms(
        &mut self,
        whole: ExprId,
        terms: Vec<ExprId>,
        op: &str,
    ) -> Result<Vec<ExprId>, UnitError> {
        let mut units: Vec<Option<Unit>> = Vec::new();
        for &t in &terms {
            if matches!(self.w.pool.get(t), ExprData::Integer(ref n) if n.0 == 0) {
                units.push(None);
            } else {
                units.push(Some(self.w.unit(t)?));
            }
        }
        let present: Vec<Unit> = units.iter().flatten().cloned().collect();
        let Some(first) = present.first() else {
            return Ok(terms);
        };
        if present.iter().all(|u| u == first) {
            return Ok(terms);
        }
        // Differing dimensions are E-UNIT-001 (reported by the plain checker);
        // affine units are E-UNIT-003.
        if present.iter().any(|u| u.dimension() != first.dimension()) {
            self.w.common(whole, &terms, op)?;
        }
        for u in &present {
            if u.is_affine() {
                return Err(UnitError::AffineUnit {
                    unit: u.name(),
                    context: format!("a sum (in `{}`)", self.w.show(whole)),
                });
            }
        }
        let mut target: &Unit = first;
        let mut best = scale_value(target.scale());
        for u in present.iter().skip(1) {
            let v = scale_value(u.scale());
            let better = match self.policy {
                UnifyPolicy::First => false,
                UnifyPolicy::Finest => v < best,
                UnifyPolicy::Coarsest => v > best,
            };
            if better {
                target = u;
                best = v;
            }
        }
        let target = target.clone();
        let mut out = Vec::with_capacity(terms.len());
        for (t, u) in terms.into_iter().zip(units) {
            match u {
                Some(u) if u != target => {
                    let f = u.conversion_factor(&target)?;
                    let c = self.w.pool.mul(vec![f.to_expr(self.w.pool), t]);
                    let clash = self.originals.contains(&c)
                        || self.w.memo.get(&c).is_some_and(|m| *m != target);
                    if clash {
                        return Err(UnitError::UnknownUnit {
                            expr: self.w.show(whole),
                            reason: format!(
                                "converting a term to `{}` produced `{}`, which also occurs unconverted in the formula; rename one of the symbols or convert by hand",
                                target.name(),
                                self.w.show(c)
                            ),
                        });
                    }
                    self.w.memo.insert(c, target.clone());
                    out.push(c);
                }
                _ => out.push(t),
            }
        }
        Ok(out)
    }

    fn rebuild(&mut self, id: ExprId) -> Result<ExprId, UnitError> {
        if self.w.units.contains_key(&id) {
            return Ok(id);
        }
        if let Some(&r) = self.memo.get(&id) {
            return Ok(r);
        }
        let pool = self.w.pool;
        let out = match pool.get(id) {
            ExprData::Add(terms) => {
                let new = self.rebuild_all(&terms)?;
                let new = self.unify_terms(id, new, "add")?;
                pool.add(new)
            }
            ExprData::Mul(fs) => {
                let new = self.rebuild_all(&fs)?;
                pool.mul(new)
            }
            ExprData::Pow { base, exp } => {
                let b = self.rebuild(base)?;
                let e = self.rebuild(exp)?;
                pool.pow(b, e)
            }
            ExprData::Func { name, args } => {
                if name == crate::diff::undefined::DERIVATIVE_FUNC || name == "Integral" {
                    id
                } else {
                    let mut new = self.rebuild_all(&args)?;
                    if matches!(name.as_str(), "min" | "max" | "Min" | "Max" | "atan2") {
                        new = self.unify_terms(id, new, "compare")?;
                    }
                    pool.func(name, new)
                }
            }
            ExprData::Piecewise { branches, default } => {
                let mut vals: Vec<ExprId> = Vec::new();
                for b in &branches {
                    vals.push(self.rebuild(b.1)?);
                }
                vals.push(self.rebuild(default)?);
                let vals = self.unify_terms(id, vals, "join the branches")?;
                let nd = vals[vals.len() - 1];
                let nb = branches
                    .iter()
                    .zip(vals.iter())
                    .map(|(b, &v)| (b.0, v))
                    .collect();
                pool.piecewise(nb, nd)
            }
            _ => id,
        };
        self.memo.insert(id, out);
        Ok(out)
    }

    fn rebuild_all(&mut self, ids: &[ExprId]) -> Result<Vec<ExprId>, UnitError> {
        ids.iter().map(|&i| self.rebuild(i)).collect()
    }
}

/// Like [`dimension_of`], but a sum whose terms share a dimension and differ
/// in unit (`t₁[h] + t₂[min]`) is **converted** instead of refused with
/// `E-UNIT-006`: every such term is multiplied by the exact ratio of its scale
/// to the unit chosen by `policy`. Returns the rewritten expression, whose
/// value is in the returned unit when every symbol is in its listed unit, and
/// that unit. The same conversion applies to the arguments of `min`/`max`/
/// `atan2` and to the values of a `Piecewise`. An expression without mixed
/// sums comes back unchanged. Differing dimensions (`E-UNIT-001`), affine
/// units in a sum (`E-UNIT-003`) and everything else [`dimension_of`] refuses
/// stay refused.
pub fn unify_units(
    expr: ExprId,
    units: &HashMap<ExprId, Unit>,
    pool: &ExprPool,
    policy: UnifyPolicy,
) -> Result<(ExprId, Unit), UnitError> {
    let mut u = Unifier {
        w: Walker {
            pool,
            units,
            memo: HashMap::new(),
        },
        memo: HashMap::new(),
        policy,
        originals: std::collections::HashSet::new(),
    };
    collect_nodes(expr, pool, &mut u.originals);
    let e = u.rebuild(expr)?;
    let unit = u.w.unit(e)?;
    Ok((e, unit))
}
