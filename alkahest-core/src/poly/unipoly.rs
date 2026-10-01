use super::error::ConversionError;
use super::exponent;
use crate::flint::{integer::FlintInteger, FlintPoly};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::kernel::{IdMap, IdSeen};
use rug::{Integer, Rational};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Sub};

// ---------------------------------------------------------------------------
// Intermediate coefficient map used only during `from_symbolic`.
// Maps degree → integer coefficient.  Zero entries are always removed.
// ---------------------------------------------------------------------------

type CoeffMap = BTreeMap<u32, Integer>;

/// Coefficient map for parsing univariate polynomials with ℚ coefficients.
type CoeffRatMap = BTreeMap<u32, Rational>;

fn coeffmap_add(mut a: CoeffMap, b: CoeffMap) -> CoeffMap {
    for (deg, coeff) in b {
        let entry = a.entry(deg).or_insert_with(|| rug::Integer::from(0));
        *entry += coeff;
        if *entry == 0 {
            a.remove(&deg);
        }
    }
    a
}

/// `ExponentTooLarge` if a product degree exceeds `u32::MAX` (it used to wrap:
/// `x^(2^31)·x^(2^31)` became the constant `1`).
fn coeffmap_mul(a: &CoeffMap, b: &CoeffMap) -> Result<CoeffMap, ConversionError> {
    let mut result = CoeffMap::new();
    for (&da, ca) in a {
        for (&db, cb) in b {
            let prod = ca.clone() * cb.clone();
            if prod == 0 {
                continue;
            }
            let d = exponent::add(da, db)?;
            let entry = result.entry(d).or_insert_with(|| rug::Integer::from(0));
            *entry += prod;
            if *entry == 0 {
                result.remove(&d);
            }
        }
    }
    Ok(result)
}

fn coeffmap_pow(base: &CoeffMap, n: u32) -> Result<CoeffMap, ConversionError> {
    if n >= 2 {
        if let Some((&top, _)) = base.last_key_value() {
            // At most `top·n + 1` distinct degrees; each map entry carries a
            // `u32` key, an `Integer` header and the B-tree's own overhead.
            let degrees = f64::from(top) * f64::from(n) + 1.0;
            let terms = super::size::power_term_bound(base.len(), n, degrees);
            let bits = f64::from(n) * super::size::log2_l1_norm(base.values());
            super::size::check_power_size(terms, bits, 48)?;
        }
    }
    coeffmap_pow_unchecked(base, n)
}

fn coeffmap_pow_unchecked(base: &CoeffMap, n: u32) -> Result<CoeffMap, ConversionError> {
    if n == 0 {
        let mut one = CoeffMap::new();
        one.insert(0, rug::Integer::from(1));
        return Ok(one);
    }
    if n == 1 {
        return Ok(base.clone());
    }
    let half = coeffmap_pow_unchecked(base, n / 2)?;
    let mut result = coeffmap_mul(&half, &half)?;
    if n % 2 == 1 {
        result = coeffmap_mul(&result, base)?;
    }
    Ok(result)
}

/// Largest degree [`UniPoly`] will materialise densely: 2^26, i.e. half a
/// GiB of coefficient slots before a single limb is stored.
///
/// `UniPoly` is a dense `fmpz_poly`, so `x^n + 1` costs `8·(n + 1)` bytes up
/// front, and FLINT aborts the whole process when that allocation fails —
/// `UniPoly.from_symbolic(x^(2^31) + 1)` asked for 16 GiB and took the
/// interpreter with it, even under `Budget(max_bytes=...)`. Conversions refuse
/// a larger degree with `E-POLY-004` before FLINT is called, and a smaller one
/// that would not fit the active memory budget (see `check_dense_degree`).
pub const MAX_DENSE_DEGREE: u32 = 1 << 26;

/// Refuse (with `ExponentTooLarge`) to allocate a dense polynomial of degree
/// `deg` above [`MAX_DENSE_DEGREE`], or one whose `8·(deg + 1)`-byte
/// coefficient array would pass the active `Budget(max_bytes=...)` or the
/// process's address-space headroom. Checked *before* FLINT allocates, since
/// FLINT cannot fail gracefully once called. Degrees below 2^16 (half a MiB)
/// skip the memory probes, which cost a `/proc` read.
pub(crate) fn check_dense_degree(deg: u64) -> Result<(), ConversionError> {
    if deg > u64::from(MAX_DENSE_DEGREE) {
        return Err(ConversionError::ExponentTooLarge);
    }
    if deg < 1 << 16 {
        return Ok(());
    }
    let bytes = (deg + 1) * std::mem::size_of::<u64>() as u64;
    if let Some(limit) = crate::budget::max_bytes() {
        if crate::budget::bytes_used().saturating_add(bytes) > limit {
            return Err(ConversionError::ExponentTooLarge);
        }
    }
    use crate::budget::memory::{address_space_limit, address_space_used, reserve_bytes};
    if let (Some(limit), Some(used)) = (address_space_limit(), address_space_used()) {
        if used
            .saturating_add(bytes)
            .saturating_add(reserve_bytes(limit))
            >= limit
        {
            return Err(ConversionError::ExponentTooLarge);
        }
    }
    Ok(())
}

fn coeffmap_to_flintpoly(map: &CoeffMap) -> Result<FlintPoly, ConversionError> {
    if let Some((&top, _)) = map.last_key_value() {
        check_dense_degree(u64::from(top))?;
    }
    let mut poly = FlintPoly::new();
    for (&deg, coeff) in map {
        let fi = FlintInteger::from_rug(coeff);
        poly.set_coeff_flint(deg as usize, &fi);
    }
    Ok(poly)
}

fn coeffmap_rat_add(mut a: CoeffRatMap, b: CoeffRatMap) -> CoeffRatMap {
    for (deg, coeff) in b {
        let entry = a.entry(deg).or_insert_with(|| Rational::from(0));
        *entry += coeff;
        if *entry == 0 {
            a.remove(&deg);
        }
    }
    a
}

fn coeffmap_rat_mul(a: &CoeffRatMap, b: &CoeffRatMap) -> Result<CoeffRatMap, ConversionError> {
    let mut result = CoeffRatMap::new();
    for (&da, ca) in a {
        for (&db, cb) in b {
            let prod = ca.clone() * cb.clone();
            if prod == 0 {
                continue;
            }
            let d = exponent::add(da, db)?;
            let entry = result.entry(d).or_insert_with(|| Rational::from(0));
            *entry += prod;
            if *entry == 0 {
                result.remove(&d);
            }
        }
    }
    Ok(result)
}

fn coeffmap_rat_pow(base: &CoeffRatMap, n: u32) -> Result<CoeffRatMap, ConversionError> {
    if n == 0 {
        let mut one = CoeffRatMap::new();
        one.insert(0, Rational::from(1));
        return Ok(one);
    }
    if n == 1 {
        return Ok(base.clone());
    }
    let half = coeffmap_rat_pow(base, n / 2)?;
    let mut result = coeffmap_rat_mul(&half, &half)?;
    if n % 2 == 1 {
        result = coeffmap_rat_mul(&result, base)?;
    }
    Ok(result)
}

/// Scale each ℚ coefficient so all become integers after multiplying by `lcm`; returns ℤ coeff map.
fn rat_coeffmap_to_integer(map: &CoeffRatMap) -> Result<CoeffMap, ConversionError> {
    let mut den_lcm = Integer::from(1);
    for r in map.values() {
        if r == &Rational::from(0) {
            continue;
        }
        den_lcm = den_lcm.lcm(&r.denom().clone());
    }
    let mut out = CoeffMap::new();
    let lcm_rat = Rational::from(&den_lcm);
    for (deg, r) in map {
        if r == &Rational::from(0) {
            continue;
        }
        let scaled = r.clone() * lcm_rat.clone();
        if *scaled.denom() != 1 {
            return Err(ConversionError::NonIntegerCoefficient);
        }
        let n = scaled.numer().clone();
        if n != 0 {
            out.insert(*deg, n);
        }
    }
    Ok(out)
}

fn expr_to_univariate_rat_coeffs(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<CoeffRatMap, ConversionError> {
    match pool.get(expr) {
        ExprData::Symbol { .. } if expr == var => {
            let mut map = CoeffRatMap::new();
            map.insert(1, Rational::from(1));
            Ok(map)
        }
        ExprData::Symbol { name, .. } => Err(ConversionError::UnexpectedSymbol(name.clone())),
        ExprData::Integer(n) => {
            let mut map = CoeffRatMap::new();
            if n.0 != 0 {
                map.insert(0, Rational::from(&n.0));
            }
            Ok(map)
        }
        ExprData::Rational(br) => {
            let mut map = CoeffRatMap::new();
            let r = br.0.clone();
            if r != 0 {
                map.insert(0, r);
            }
            Ok(map)
        }
        ExprData::Float(_) => Err(ConversionError::NonIntegerCoefficient),
        ExprData::Add(args) => {
            let mut acc = CoeffRatMap::new();
            for &arg in &args {
                let sub = expr_to_univariate_rat_coeffs(arg, var, pool)?;
                acc = coeffmap_rat_add(acc, sub);
            }
            Ok(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = CoeffRatMap::new();
            acc.insert(0, Rational::from(1));
            for &arg in &args {
                let sub = expr_to_univariate_rat_coeffs(arg, var, pool)?;
                acc = coeffmap_rat_mul(&acc, &sub)?;
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(n) => {
                let n_u32 = exponent::exponent_u32(&n.0)?;
                let base_coeffs = expr_to_univariate_rat_coeffs(base, var, pool)?;
                coeffmap_rat_pow(&base_coeffs, n_u32)
            }
            _ => Err(ConversionError::NonConstantExponent),
        },
        ExprData::Func { name, .. } => Err(ConversionError::NonPolynomialFunction(name.clone())),
        ExprData::Piecewise { .. } => Err(ConversionError::NonPolynomialFunction(
            "Piecewise".to_string(),
        )),
        ExprData::Predicate { .. } => Err(ConversionError::NonPolynomialFunction(
            "Predicate".to_string(),
        )),
        ExprData::Forall { .. } | ExprData::Exists { .. } => Err(
            ConversionError::NonPolynomialFunction("quantifier".to_string()),
        ),
        ExprData::BigO(_) => Err(ConversionError::NonPolynomialFunction("BigO".to_string())),
        ExprData::RootSum { .. } => Err(ConversionError::NonPolynomialFunction(
            "RootSum".to_string(),
        )),
    }
}

fn expr_to_univariate_coeffs(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<CoeffMap, ConversionError> {
    match pool.get(expr) {
        // var itself → x^1
        ExprData::Symbol { .. } if expr == var => {
            let mut map = CoeffMap::new();
            map.insert(1, rug::Integer::from(1));
            Ok(map)
        }
        // Any other symbol is a free variable — not a valid integer coefficient
        ExprData::Symbol { name, .. } => Err(ConversionError::UnexpectedSymbol(name)),
        // Integer constant
        ExprData::Integer(n) => {
            let mut map = CoeffMap::new();
            if n.0 != 0 {
                map.insert(0, n.0.clone());
            }
            Ok(map)
        }
        // A `Rational` node whose value is integral (denominator 1) can arise
        // from un-collapsed arithmetic; treat it as the integer numerator.
        ExprData::Rational(r) if *r.0.denom() == 1 => {
            let mut map = CoeffMap::new();
            let n = r.0.numer().clone();
            if n != 0 {
                map.insert(0, n);
            }
            Ok(map)
        }
        ExprData::Rational(_) | ExprData::Float(_) => Err(ConversionError::NonIntegerCoefficient),
        // n-ary sum: recurse and accumulate
        ExprData::Add(args) => {
            let mut acc = CoeffMap::new();
            for &arg in &args {
                let sub = expr_to_univariate_coeffs(arg, var, pool)?;
                acc = coeffmap_add(acc, sub);
            }
            Ok(acc)
        }
        // n-ary product: recurse and convolve
        ExprData::Mul(args) => {
            let mut acc = CoeffMap::new();
            acc.insert(0, rug::Integer::from(1));
            for &arg in &args {
                let sub = expr_to_univariate_coeffs(arg, var, pool)?;
                acc = coeffmap_mul(&acc, &sub)?;
            }
            Ok(acc)
        }
        // Power with a constant non-negative integer exponent
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(n) => {
                let n_u32 = exponent::exponent_u32(&n.0)?;
                let base_coeffs = expr_to_univariate_coeffs(base, var, pool)?;
                coeffmap_pow(&base_coeffs, n_u32)
            }
            _ => Err(ConversionError::NonConstantExponent),
        },
        ExprData::Func { name, .. } => Err(ConversionError::NonPolynomialFunction(name)),
        ExprData::Piecewise { .. } => Err(ConversionError::NonPolynomialFunction(
            "Piecewise".to_string(),
        )),
        ExprData::Predicate { .. } => Err(ConversionError::NonPolynomialFunction(
            "Predicate".to_string(),
        )),
        ExprData::Forall { .. } | ExprData::Exists { .. } => Err(
            ConversionError::NonPolynomialFunction("quantifier".to_string()),
        ),
        ExprData::BigO(_) => Err(ConversionError::NonPolynomialFunction("BigO".to_string())),
        ExprData::RootSum { .. } => Err(ConversionError::NonPolynomialFunction(
            "RootSum".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Direct FLINT construction used by `from_symbolic`.
//
// Walks the expression in the same order as `expr_to_univariate_coeffs`, so a
// non-polynomial input fails with the same error, but does the arithmetic on
// dense `fmpz_poly`s. Results for DAG nodes reached more than once are
// memoised, so shared subexpressions (a Chebyshev-style recurrence, say) cost
// linear rather than exponential time.
//
// Dense arithmetic allocates one word per degree, whereas the coefficient map
// is sparse. When an intermediate degree would exceed `DENSE_DEGREE_LIMIT`
// the builder gives up and the caller reruns the sparse map algorithm, which
// keeps behaviour for huge sparse exponents — including terms that cancel,
// and the map's own u32 degree arithmetic — exactly as it was.
// ---------------------------------------------------------------------------

/// Largest intermediate degree the dense FLINT builder will materialise.
const DENSE_DEGREE_LIMIT: i64 = 1 << 22;

enum BuildError {
    Conversion(ConversionError),
    /// An intermediate degree exceeded [`DENSE_DEGREE_LIMIT`].
    TooSparse,
}

impl From<ConversionError> for BuildError {
    fn from(e: ConversionError) -> Self {
        BuildError::Conversion(e)
    }
}

/// One expression node, read out of the pool before recursing.
enum UniNode {
    Var,
    Small(i64),
    Big(Integer),
    Add(Vec<ExprId>),
    Mul(Vec<ExprId>),
    Pow(ExprId, ExprId),
    Fail(ConversionError),
}

struct UniBuilder<'a> {
    var: ExprId,
    pool: &'a ExprPool,
    /// Compound nodes converted at least once.
    ///
    /// Keyed with the `ExprId` hasher and inline while small: every
    /// compound node is inserted here, and on a small polynomial a growing
    /// SipHash set was a visible share of the whole conversion.
    seen: IdSeen,
    /// Results for compound nodes reached a second time, i.e. shared ones.
    /// Only shared nodes are stored, so a tree pays no copying and holds no
    /// extra intermediates; a DAG node is computed at most twice.
    memo: IdMap<FlintPoly>,
}

impl<'a> UniBuilder<'a> {
    fn new(var: ExprId, pool: &'a ExprPool) -> Self {
        UniBuilder {
            var,
            pool,
            seen: IdSeen::default(),
            memo: IdMap::default(),
        }
    }

    fn node(&self, expr: ExprId) -> UniNode {
        let var = self.var;
        // Same case split as `expr_to_univariate_coeffs`, same errors.
        self.pool.with(expr, |d| match d {
            ExprData::Symbol { .. } if expr == var => UniNode::Var,
            ExprData::Symbol { name, .. } => {
                UniNode::Fail(ConversionError::UnexpectedSymbol(name.clone()))
            }
            ExprData::Integer(n) => match n.0.to_i64() {
                Some(v) => UniNode::Small(v),
                None => UniNode::Big(n.0.clone()),
            },
            ExprData::Rational(r) if *r.0.denom() == 1 => match r.0.numer().to_i64() {
                Some(v) => UniNode::Small(v),
                None => UniNode::Big(r.0.numer().clone()),
            },
            ExprData::Rational(_) | ExprData::Float(_) => {
                UniNode::Fail(ConversionError::NonIntegerCoefficient)
            }
            ExprData::Add(args) => UniNode::Add(args.clone()),
            ExprData::Mul(args) => UniNode::Mul(args.clone()),
            ExprData::Pow { base, exp } => UniNode::Pow(*base, *exp),
            ExprData::Func { name, .. } => {
                UniNode::Fail(ConversionError::NonPolynomialFunction(name.clone()))
            }
            ExprData::Piecewise { .. } => UniNode::Fail(ConversionError::NonPolynomialFunction(
                "Piecewise".to_string(),
            )),
            ExprData::Predicate { .. } => UniNode::Fail(ConversionError::NonPolynomialFunction(
                "Predicate".to_string(),
            )),
            ExprData::Forall { .. } | ExprData::Exists { .. } => UniNode::Fail(
                ConversionError::NonPolynomialFunction("quantifier".to_string()),
            ),
            ExprData::BigO(_) => {
                UniNode::Fail(ConversionError::NonPolynomialFunction("BigO".to_string()))
            }
            ExprData::RootSum { .. } => UniNode::Fail(ConversionError::NonPolynomialFunction(
                "RootSum".to_string(),
            )),
        })
    }

    fn build(&mut self, expr: ExprId) -> Result<FlintPoly, BuildError> {
        let node = match self.node(expr) {
            UniNode::Var => return Ok(FlintPoly::from_coefficients(&[0, 1])),
            UniNode::Small(v) => return Ok(FlintPoly::from_coefficients(&[v])),
            UniNode::Big(n) => return Ok(FlintPoly::from_rug_coefficients(&[n])),
            UniNode::Fail(e) => return Err(e.into()),
            compound => compound,
        };
        if let Some(p) = self.memo.get(&expr) {
            return Ok(p.clone());
        }
        let result = match node {
            UniNode::Add(args) => {
                let mut acc = FlintPoly::new();
                for arg in args {
                    let sub = self.build(arg)?;
                    acc = if acc.is_zero() { sub } else { &acc + &sub };
                }
                acc
            }
            UniNode::Mul(args) => {
                let mut acc: Option<FlintPoly> = None;
                for arg in args {
                    let sub = self.build(arg)?;
                    acc = Some(match acc {
                        None => sub,
                        Some(a) => {
                            if !a.is_zero() && !sub.is_zero() {
                                let d = a.degree() + sub.degree();
                                if d > DENSE_DEGREE_LIMIT {
                                    return Err(BuildError::TooSparse);
                                }
                                check_dense_degree(d as u64)?;
                            }
                            &a * &sub
                        }
                    });
                }
                acc.unwrap_or_else(|| FlintPoly::from_coefficients(&[1]))
            }
            UniNode::Pow(base, exp) => {
                // The exponent is checked before the base is converted, as in
                // `expr_to_univariate_coeffs`.
                let n = match self.pool.get(exp) {
                    ExprData::Integer(n) => n.0,
                    _ => return Err(ConversionError::NonConstantExponent.into()),
                };
                let n_u32 = exponent::exponent_u32(&n)?;
                let b = self.build(base)?;
                if b.degree() > 0 {
                    let d = b.degree() * i64::from(n_u32);
                    if d > DENSE_DEGREE_LIMIT {
                        return Err(BuildError::TooSparse);
                    }
                    check_dense_degree(d as u64)?;
                }

                match n_u32 {
                    1 => b,
                    _ => b
                        .checked_pow(n_u32)
                        .ok_or(ConversionError::ExponentTooLarge)?,
                }
            }
            UniNode::Var | UniNode::Small(_) | UniNode::Big(_) | UniNode::Fail(_) => {
                unreachable!("atoms return early")
            }
        };
        if !self.seen.insert(expr) {
            self.memo.insert(expr, result.clone());
        }
        Ok(result)
    }
}

/// `expr` as a dense ℤ\[var\] polynomial: FLINT arithmetic when every
/// intermediate degree is at most [`DENSE_DEGREE_LIMIT`], the sparse
/// coefficient map otherwise. Both give the same polynomial, or the same error.
fn expr_to_flintpoly(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<FlintPoly, ConversionError> {
    match UniBuilder::new(var, pool).build(expr) {
        Ok(p) => Ok(p),
        Err(BuildError::Conversion(e)) => Err(e),
        Err(BuildError::TooSparse) => {
            coeffmap_to_flintpoly(&expr_to_univariate_coeffs(expr, var, pool)?)
        }
    }
}

// ---------------------------------------------------------------------------
// UniPoly
// ---------------------------------------------------------------------------

/// Dense univariate polynomial over ℤ, backed by FLINT's `fmpz_poly_t`.
///
/// Coefficients are in ascending degree order: index 0 is the constant term.
/// The variable is tracked as an `ExprId` so conversion back to symbolic
/// form remains possible.
#[derive(Clone)]
pub struct UniPoly {
    pub var: ExprId,
    pub coeffs: FlintPoly,
}

impl UniPoly {
    pub fn zero(var: ExprId) -> Self {
        UniPoly {
            var,
            coeffs: FlintPoly::new(),
        }
    }

    pub fn constant(var: ExprId, c: i64) -> Self {
        UniPoly {
            var,
            coeffs: if c == 0 {
                FlintPoly::new()
            } else {
                FlintPoly::from_coefficients(&[c])
            },
        }
    }

    /// Convert a symbolic expression to a `UniPoly` in `var`.
    /// Returns `Err` if the expression is not a polynomial in `var`
    /// with integer coefficients.
    pub fn from_symbolic(
        expr: ExprId,
        var: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, ConversionError> {
        let coeffs = expr_to_flintpoly(expr, var, pool)?;
        Ok(UniPoly { var, coeffs })
    }

    /// Like [`Self::from_symbolic`], but after integer parsing fails, interprets
    /// coefficients as rationals, multiplies through by the least common denominator,
    /// and returns the resulting primitive ℤ polynomial (same roots in ℚ as the input).
    pub fn from_symbolic_clear_denoms(
        expr: ExprId,
        var: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, ConversionError> {
        match Self::from_symbolic(expr, var, pool) {
            Ok(p) => Ok(p),
            Err(ConversionError::NonIntegerCoefficient) => {
                let map = expr_to_univariate_rat_coeffs(expr, var, pool)?;
                let intmap = rat_coeffmap_to_integer(&map)?;
                let coeffs = coeffmap_to_flintpoly(&intmap)?;

                Ok(UniPoly { var, coeffs })
            }
            Err(e) => Err(e),
        }
    }

    /// Coefficient vector in ascending degree order (constant term first).
    pub fn coefficients(&self) -> Vec<rug::Integer> {
        (0..self.coeffs.length())
            .map(|i| self.coeffs.get_coeff_flint(i).to_rug())
            .collect()
    }

    /// Coefficient vector as `i64`. Overflows silently for large coefficients —
    /// use `coefficients()` for lossless access.
    pub fn coefficients_i64(&self) -> Vec<i64> {
        self.coeffs.coefficients()
    }

    /// Coefficient vector as `i64`, or `None` if any coefficient overflows i64.
    pub fn coefficients_i64_checked(&self) -> Option<Vec<i64>> {
        self.coefficients()
            .into_iter()
            .map(|c| c.to_i64())
            .collect()
    }

    /// Leading (highest-degree) coefficient, exact. `0` for the zero
    /// polynomial, matching `degree() == -1` there.
    pub fn leading_coeff(&self) -> rug::Integer {
        self.coeffs.leading_coeff_fmpz().to_rug()
    }

    pub fn degree(&self) -> i64 {
        self.coeffs.degree()
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_zero()
    }

    /// `self^exp`.
    ///
    /// # Panics
    ///
    /// If the result's degree would pass the dense ceiling (see
    /// [`Self::checked_pow`]). FLINT used to be asked for the allocation and
    /// abort the process; a panic can at least be caught.
    pub fn pow(&self, exp: u32) -> Self {
        self.checked_pow(exp)
            .unwrap_or_else(|e| panic!("UniPoly::pow: {e} (E-POLY-004)"))
    }

    /// `self^exp`, or [`ConversionError::ExponentTooLarge`] if the result's
    /// degree would exceed [`MAX_DENSE_DEGREE`], or its estimated size
    /// (degree times coefficient bits, see [`FlintPoly::checked_pow`]) would
    /// not fit the machine or the active memory budget.
    pub fn checked_pow(&self, exp: u32) -> Result<Self, ConversionError> {
        let deg = u64::try_from(self.degree()).unwrap_or(0);
        check_dense_degree(deg.saturating_mul(u64::from(exp)))?;
        Ok(UniPoly {
            var: self.var,
            coeffs: self
                .coeffs
                .checked_pow(exp)
                .ok_or(ConversionError::ExponentTooLarge)?,
        })
    }

    /// `self * rhs`, or [`ConversionError::ExponentTooLarge`] if the product's
    /// degree would exceed [`MAX_DENSE_DEGREE`] or the active memory budget
    /// (checked before FLINT allocates, since FLINT aborts on failure).
    ///
    /// # Panics
    ///
    /// If the operands have different variables (as for `*`).
    pub fn checked_mul(&self, rhs: &Self) -> Result<Self, ConversionError> {
        if !self.is_zero() && !rhs.is_zero() {
            let d = u64::try_from(self.degree()).unwrap_or(0)
                + u64::try_from(rhs.degree()).unwrap_or(0);
            check_dense_degree(d)?;
        }
        Ok(self * rhs)
    }

    /// Pseudo-division: returns `(quotient, remainder)` satisfying
    /// `lc(other)^d * self = quotient * other + remainder`.
    /// Returns `None` if the variables differ.
    pub fn pseudo_divrem(&self, other: &Self) -> Option<(Self, Self)> {
        if self.var != other.var {
            return None;
        }
        let (q_coeffs, r_coeffs, _) = self.coeffs.pseudo_divrem(&other.coeffs);
        Some((
            UniPoly {
                var: self.var,
                coeffs: q_coeffs,
            },
            UniPoly {
                var: self.var,
                coeffs: r_coeffs,
            },
        ))
    }

    /// GCD of two polynomials over the same variable (up to scalar units).
    /// Returns `None` if the variables differ.
    pub fn gcd(&self, other: &Self) -> Option<Self> {
        if self.var != other.var {
            return None;
        }
        Some(UniPoly {
            var: self.var,
            coeffs: self.coeffs.gcd(&other.coeffs),
        })
    }

    /// Formal derivative with respect to the tracked degree variable ([`UniPoly::var`]).
    pub fn derivative(&self) -> Self {
        UniPoly {
            var: self.var,
            coeffs: self.coeffs.derivative(),
        }
    }

    /// Rebuild a symbolic sum of nonzero monomial terms in [`Self::var`] (`c·x^k` with `ℤ` coeffs).
    pub fn to_symbolic_expr(&self, pool: &ExprPool) -> ExprId {
        let coeffs = self.coefficients(); // ascending degree
        let var = self.var;
        if coeffs.is_empty() {
            return pool.integer(0_i32);
        }
        let summands: Vec<ExprId> = coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(deg, coeff)| {
                let c_id = pool.integer(coeff.clone());
                if deg == 0 {
                    c_id
                } else {
                    let exp_id = pool.integer(deg as i64);
                    let x_pow = if deg == 1 { var } else { pool.pow(var, exp_id) };
                    if *coeff == 1 {
                        x_pow
                    } else if *coeff == -1 {
                        pool.mul(vec![pool.integer(-1_i32), x_pow])
                    } else {
                        pool.mul(vec![c_id, x_pow])
                    }
                }
            })
            .collect();

        match summands.len() {
            0 => pool.integer(0_i32),
            1 => summands[0],
            _ => pool.add(summands),
        }
    }

    /// Squarefree kernel: divides out `gcd(p, p')` repeatedly until trivial.
    /// Constant and zero polynomials return a clone unchanged.
    pub fn squarefree_part(&self) -> Self {
        if self.is_zero() || self.degree() <= 0 {
            return self.clone();
        }
        let mut p = self.clone();
        loop {
            let d = p.derivative();
            if d.is_zero() {
                break;
            }
            let Some(g) = p.gcd(&d) else {
                break;
            };
            if g.degree() <= 0 {
                break;
            }
            p = UniPoly {
                var: p.var,
                coeffs: p.coeffs.div_exact(&g.coeffs),
            };
        }
        p
    }

    /// Multiply then divide by `\gcd(u, v)` (least common multiple over ℤ\[x\] up to a unit).
    pub fn lcm_poly(&self, other: &Self) -> Self {
        same_var(self, other);
        let prod = self * other;
        let g = self.gcd(other).unwrap();
        UniPoly {
            var: self.var,
            coeffs: prod.coeffs.div_exact(&g.coeffs),
        }
    }

    /// Evaluate at a rational point using Horner's method.
    pub fn eval_rational(&self, x: &rug::Rational) -> rug::Rational {
        let n = self.coeffs.length();
        if n == 0 {
            return rug::Rational::from(0);
        }
        let mut acc = rug::Rational::from((
            self.coeffs.get_coeff_flint(n - 1).to_rug(),
            rug::Integer::from(1),
        ));
        for idx in (0..n.saturating_sub(1)).rev() {
            acc = acc * x.clone()
                + rug::Rational::from((
                    self.coeffs.get_coeff_flint(idx).to_rug(),
                    rug::Integer::from(1),
                ));
        }
        acc
    }
}

impl PartialEq for UniPoly {
    fn eq(&self, other: &Self) -> bool {
        self.var == other.var && self.coeffs == other.coeffs
    }
}
impl Eq for UniPoly {}

// ---------------------------------------------------------------------------
// Arithmetic — same variable required; panics on variable mismatch
// ---------------------------------------------------------------------------

fn same_var(a: &UniPoly, b: &UniPoly) {
    assert_eq!(
        a.var, b.var,
        "UniPoly arithmetic requires both operands to share the same variable"
    );
}

impl Add for UniPoly {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        &self + &rhs
    }
}
impl<'b> Add<&'b UniPoly> for &UniPoly {
    type Output = UniPoly;
    fn add(self, rhs: &'b UniPoly) -> UniPoly {
        same_var(self, rhs);
        UniPoly {
            var: self.var,
            coeffs: &self.coeffs + &rhs.coeffs,
        }
    }
}

impl Sub for UniPoly {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        &self - &rhs
    }
}
impl<'b> Sub<&'b UniPoly> for &UniPoly {
    type Output = UniPoly;
    fn sub(self, rhs: &'b UniPoly) -> UniPoly {
        same_var(self, rhs);
        UniPoly {
            var: self.var,
            coeffs: &self.coeffs - &rhs.coeffs,
        }
    }
}

impl Mul for UniPoly {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}
impl<'b> Mul<&'b UniPoly> for &UniPoly {
    type Output = UniPoly;
    fn mul(self, rhs: &'b UniPoly) -> UniPoly {
        same_var(self, rhs);
        UniPoly {
            var: self.var,
            coeffs: &self.coeffs * &rhs.coeffs,
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for UniPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.coeffs)
    }
}

impl fmt::Debug for UniPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UniPoly(var={:?}, {})", self.var, self.coeffs)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use proptest::prelude::*;

    fn pool_and_var() -> (ExprPool, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        (p, x)
    }

    // --- from_symbolic ---

    #[test]
    fn from_symbolic_quadratic() {
        // x^2 + 2*x + 1
        let (p, x) = pool_and_var();
        let two = p.integer(2_i32);
        let one = p.integer(1_i32);
        let xsq = p.pow(x, p.integer(2_i32));
        let two_x = p.mul(vec![two, x]);
        let expr = p.add(vec![xsq, two_x, one]);
        let poly = UniPoly::from_symbolic(expr, x, &p).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![1, 2, 1]);
    }

    #[test]
    fn leading_coeff_is_exact_and_zero_for_the_zero_polynomial() {
        let (p, x) = pool_and_var();
        // 3*x^2 + 1
        let xsq = p.pow(x, p.integer(2_i32));
        let expr = p.add(vec![p.mul(vec![p.integer(3_i32), xsq]), p.integer(1_i32)]);
        let poly = UniPoly::from_symbolic(expr, x, &p).unwrap();
        assert_eq!(poly.leading_coeff(), rug::Integer::from(3));

        // Beyond i64, where `coefficients_i64` truncates.
        let big: rug::Integer = rug::Integer::from(1) << 100;
        let big_expr = p.mul(vec![p.integer(big.clone()), xsq]);
        let big_poly = UniPoly::from_symbolic(big_expr, x, &p).unwrap();
        assert_eq!(big_poly.leading_coeff(), big);

        let zero = UniPoly::from_symbolic(p.integer(0_i32), x, &p).unwrap();
        assert_eq!(zero.degree(), -1);
        assert_eq!(zero.leading_coeff(), rug::Integer::ZERO);
    }

    #[test]
    fn from_symbolic_constant() {
        let (p, x) = pool_and_var();
        let five = p.integer(5_i32);
        let poly = UniPoly::from_symbolic(five, x, &p).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![5]);
    }

    #[test]
    fn from_symbolic_zero() {
        let (p, x) = pool_and_var();
        let zero = p.integer(0_i32);
        let poly = UniPoly::from_symbolic(zero, x, &p).unwrap();
        assert!(poly.is_zero());
    }

    #[test]
    fn from_symbolic_identity() {
        // p(x) = x  →  coefficients [0, 1]
        let (p, x) = pool_and_var();
        let poly = UniPoly::from_symbolic(x, x, &p).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![0, 1]);
    }

    #[test]
    fn from_symbolic_integer_valued_rational_node() {
        // An un-collapsed Rational(4, 1) node should be accepted as the
        // integer constant 4.
        let (p, x) = pool_and_var();
        let four = p.rational(4_i32, 1_i32);
        let poly = UniPoly::from_symbolic(four, x, &p).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![4]);
    }

    #[test]
    fn from_symbolic_non_integer_rational_still_rejected() {
        // A genuine non-integer Rational(1, 2) node must still error.
        let (p, x) = pool_and_var();
        let half = p.rational(1_i32, 2_i32);
        assert!(matches!(
            UniPoly::from_symbolic(half, x, &p),
            Err(ConversionError::NonIntegerCoefficient)
        ));
    }

    #[test]
    fn from_symbolic_free_symbol_error() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let expr = p.add(vec![x, y]);
        assert!(matches!(
            UniPoly::from_symbolic(expr, x, &p),
            Err(ConversionError::UnexpectedSymbol(_))
        ));
    }

    #[test]
    fn from_symbolic_negative_exponent_error() {
        let (p, x) = pool_and_var();
        let neg_one = p.integer(-1_i32);
        let expr = p.pow(x, neg_one);
        assert!(matches!(
            UniPoly::from_symbolic(expr, x, &p),
            Err(ConversionError::NegativeExponent)
        ));
    }

    #[test]
    fn from_symbolic_clear_denoms_rational_linear() {
        // λ/2 + 1  →  clears to λ + 2
        let (p, x) = pool_and_var();
        let half = p.rational(1, 2);
        let term = p.mul(vec![half, x]);
        let expr = p.add(vec![term, p.integer(1_i32)]);
        let poly = UniPoly::from_symbolic_clear_denoms(expr, x, &p).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![2, 1]);
    }

    #[test]
    fn from_symbolic_power_of_poly() {
        // (x + 1)^2 = x^2 + 2x + 1
        let (p, x) = pool_and_var();
        let one = p.integer(1_i32);
        let x_plus_1 = p.add(vec![x, one]);
        let two = p.integer(2_i32);
        let expr = p.pow(x_plus_1, two);
        let poly = UniPoly::from_symbolic(expr, x, &p).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![1, 2, 1]);
    }

    // --- arithmetic ---

    #[test]
    fn add_polys() {
        let (p, x) = pool_and_var();
        let a = UniPoly::from_symbolic(p.add(vec![x, p.integer(1_i32)]), x, &p).unwrap();
        let b = UniPoly::from_symbolic(p.add(vec![x, p.integer(-1_i32)]), x, &p).unwrap();
        let sum = &a + &b;
        assert_eq!(sum.coefficients_i64(), vec![0, 2]);
    }

    #[test]
    fn sub_polys() {
        let (p, x) = pool_and_var();
        let a = UniPoly::from_symbolic(
            p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(1_i32)]),
            x,
            &p,
        )
        .unwrap();
        let b = UniPoly::from_symbolic(p.integer(1_i32), x, &p).unwrap();
        let diff = &a - &b;
        assert_eq!(diff.coefficients_i64(), vec![0, 0, 1]);
    }

    #[test]
    fn mul_polys() {
        // (x+1)*(x-1) = x^2 - 1
        let (p, x) = pool_and_var();
        let a = UniPoly::from_symbolic(p.add(vec![x, p.integer(1_i32)]), x, &p).unwrap();
        let b = UniPoly::from_symbolic(p.add(vec![x, p.integer(-1_i32)]), x, &p).unwrap();
        let prod = &a * &b;
        assert_eq!(prod.coefficients_i64(), vec![-1, 0, 1]);
    }

    #[test]
    fn pow_poly() {
        let (p, x) = pool_and_var();
        let xp1 = UniPoly::from_symbolic(p.add(vec![x, p.integer(1_i32)]), x, &p).unwrap();
        let q = xp1.pow(3);
        assert_eq!(q.coefficients_i64(), vec![1, 3, 3, 1]);
    }

    #[test]
    fn gcd_polys() {
        // gcd(x^2 - 1, x - 1) should have degree 1
        let (p, x) = pool_and_var();
        let x2m1 = UniPoly::from_symbolic(
            p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(-1_i32)]),
            x,
            &p,
        )
        .unwrap();
        let xm1 = UniPoly::from_symbolic(p.add(vec![x, p.integer(-1_i32)]), x, &p).unwrap();
        let g = x2m1.gcd(&xm1).unwrap();
        assert_eq!(g.degree(), 1);
    }

    // --- display ---

    #[test]
    fn display_linear() {
        let (p, x) = pool_and_var();
        let poly = UniPoly::from_symbolic(p.add(vec![x, p.integer(1_i32)]), x, &p).unwrap();
        let s = poly.to_string();
        assert!(s.contains('x'), "display should mention x: {s}");
    }

    // --- FLINT builder vs the coefficient-map algorithm ---

    /// The pre-FLINT `from_symbolic`: sparse coefficient map, then dense.
    fn reference(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<FlintPoly, ConversionError> {
        expr_to_univariate_coeffs(expr, var, pool).and_then(|m| coeffmap_to_flintpoly(&m))
    }

    /// The pre-FLINT `FlintPoly::derivative`, coefficient by coefficient.
    fn reference_derivative(p: &FlintPoly) -> FlintPoly {
        let deg = p.degree();
        let mut result = FlintPoly::new();
        for i in 1..=deg.max(0) as usize {
            let c = p.get_coeff_flint(i).to_rug() * i as i64;
            result.set_coeff_flint(i - 1, &FlintInteger::from_rug(&c));
        }
        result
    }

    /// Expression recipe, built into a pool once the strategy has run.
    #[derive(Debug, Clone)]
    enum Recipe {
        Var,
        Other,
        Int(i64),
        Big(i64, u32),
        Rat(i64, i64),
        Sin,
        Add(Vec<Recipe>),
        Mul(Vec<Recipe>),
        Pow(Box<Recipe>, i64),
        PowVar(Box<Recipe>),
    }

    fn recipe() -> impl Strategy<Value = Recipe> {
        let leaf = prop_oneof![
            10 => Just(Recipe::Var),
            1 => Just(Recipe::Other),
            4 => (-5i64..=5).prop_map(Recipe::Int),
            2 => any::<i64>().prop_map(Recipe::Int),
            1 => (any::<i64>(), 60u32..200).prop_map(|(a, b)| Recipe::Big(a, b)),
            1 => (-6i64..=6, 1i64..=3).prop_map(|(a, b)| Recipe::Rat(a, b)),
            1 => Just(Recipe::Sin),
        ];
        leaf.prop_recursive(4, 40, 4, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..4).prop_map(Recipe::Add),
                prop::collection::vec(inner.clone(), 0..4).prop_map(Recipe::Mul),
                (inner.clone(), -1i64..=5).prop_map(|(b, n)| Recipe::Pow(Box::new(b), n)),
                inner.prop_map(|b| Recipe::PowVar(Box::new(b))),
            ]
        })
    }

    fn build(r: &Recipe, p: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
        match r {
            Recipe::Var => x,
            Recipe::Other => y,
            Recipe::Int(n) => p.integer(*n),
            Recipe::Big(a, b) => p.integer(Integer::from(*a) << *b),
            Recipe::Rat(a, b) => p.rational(*a, *b),
            Recipe::Sin => p.func("sin", vec![x]),
            Recipe::Add(v) => p.add(v.iter().map(|c| build(c, p, x, y)).collect()),
            Recipe::Mul(v) => p.mul(v.iter().map(|c| build(c, p, x, y)).collect()),
            Recipe::Pow(b, n) => p.pow(build(b, p, x, y), p.integer(*n)),
            Recipe::PowVar(b) => p.pow(build(b, p, x, y), x),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1024))]

        /// Same polynomial, or the same error, as the coefficient-map path.
        #[test]
        fn flint_builder_matches_coeffmap(r in recipe()) {
            let p = ExprPool::new();
            let x = p.symbol("x", Domain::Real);
            let y = p.symbol("y", Domain::Real);
            let e = build(&r, &p, x, y);
            prop_assert_eq!(expr_to_flintpoly(e, x, &p), reference(e, x, &p));
        }

        #[test]
        fn derivative_matches_per_coefficient(
            coeffs in prop::collection::vec((any::<i64>(), 0u32..130), 0..40)
        ) {
            let c: Vec<Integer> = coeffs.iter().map(|&(a, s)| Integer::from(a) << s).collect();
            let f = FlintPoly::from_rug_coefficients(&c);
            prop_assert_eq!(f.derivative(), reference_derivative(&f));
        }
    }

    /// `T_{n+1} = 2x·T_n − T_{n−1}` shares each `T_n` between two parents, so
    /// an unmemoised walk is exponential in `n`.
    #[test]
    fn from_symbolic_is_linear_on_shared_dag() {
        let (p, x) = pool_and_var();
        let two_x = p.mul(vec![p.integer(2_i32), x]);
        let (mut t0, mut t1) = (p.integer(1_i32), x);
        let (mut f0, mut f1) = (
            FlintPoly::from_coefficients(&[1]),
            FlintPoly::from_coefficients(&[0, 1]),
        );
        let fx2 = FlintPoly::from_coefficients(&[0, 2]);
        for _ in 0..60 {
            let t2 = p.add(vec![
                p.mul(vec![two_x, t1]),
                p.mul(vec![p.integer(-1_i32), t0]),
            ]);
            let f2 = &(&fx2 * &f1) - &f0;
            (t0, t1, f0, f1) = (t1, t2, f1, f2);
        }
        let start = std::time::Instant::now();
        let got = UniPoly::from_symbolic(t1, x, &p).unwrap();
        assert!(
            start.elapsed().as_secs() < 5,
            "T_61 took {:?}",
            start.elapsed()
        );
        assert_eq!(got.coeffs, f1);
        assert_eq!(got.degree(), 61);
    }

    /// Past the dense-degree limit the sparse map takes over, so a huge
    /// exponent that cancels or is multiplied by zero stays cheap.
    #[test]
    fn huge_sparse_degree_falls_back_to_the_map() {
        let (p, x) = pool_and_var();
        let big = p.pow(x, p.integer(3_000_000_000_u64));
        let zero_times = p.mul(vec![p.integer(0_i32), big]);
        assert!(UniPoly::from_symbolic(zero_times, x, &p).unwrap().is_zero());
        let cancels = p.add(vec![
            big,
            p.mul(vec![p.integer(-1_i32), big]),
            p.integer(7_i32),
        ]);
        let got = UniPoly::from_symbolic(cancels, x, &p).unwrap();
        assert_eq!(got.coefficients_i64(), vec![7]);
        // An error after the huge factor is still the coefficient map's error.
        let with_sin = p.mul(vec![big, p.func("sin", vec![x])]);
        assert_eq!(
            UniPoly::from_symbolic(with_sin, x, &p).err(),
            reference(with_sin, x, &p).err()
        );
    }

    // --- Exponent overflow and the dense-degree ceiling (audit A1, B4) ---

    const B31: u64 = 1 << 31;

    #[test]
    fn product_degree_past_u32_is_refused_not_wrapped() {
        // x^(2^31) · x^(2^31) - 4: the map path used to wrap the degree to 0
        // and see the constant -3 (so `real_roots` found no roots).
        let (p, x) = pool_and_var();
        let h = p.pow(x, p.integer(B31));
        let w = p.add(vec![p.mul(vec![h, h]), p.integer(-4_i32)]);
        assert_eq!(
            UniPoly::from_symbolic(w, x, &p).err(),
            Some(ConversionError::ExponentTooLarge)
        );
        // Same through the ℚ-coefficient path.
        let wq = p.add(vec![p.mul(vec![h, h]), p.rational(1, 2)]);
        assert_eq!(
            UniPoly::from_symbolic_clear_denoms(wq, x, &p).err(),
            Some(ConversionError::ExponentTooLarge)
        );
        assert_eq!(
            coeffmap_mul(
                &CoeffMap::from([(u32::MAX, Integer::from(1))]),
                &CoeffMap::from([(1, Integer::from(1))])
            ),
            Err(ConversionError::ExponentTooLarge)
        );
    }

    #[test]
    fn dense_degree_ceiling_refuses_before_flint_allocates() {
        // x^(2^31) + 1 asked FLINT for 16 GiB and aborted the process.
        let (p, x) = pool_and_var();
        for e in [
            u64::from(MAX_DENSE_DEGREE) + 1,
            B31,
            1 << 32,
            1 << 63,
            u64::MAX,
        ] {
            let f = p.add(vec![p.pow(x, p.integer(e)), p.integer(1_i32)]);
            assert_eq!(
                UniPoly::from_symbolic(f, x, &p).err(),
                Some(ConversionError::ExponentTooLarge),
                "degree {e}"
            );
        }
        // The ceiling itself is allowed by the check (not allocated here).
        assert_eq!(check_dense_degree(u64::from(MAX_DENSE_DEGREE)), Ok(()));
        // A huge degree that cancels is still fine: nothing dense is built.
        let big = p.pow(x, p.integer(B31));
        let cancels = p.add(vec![big, p.mul(vec![p.integer(-1_i32), big])]);
        assert!(UniPoly::from_symbolic(cancels, x, &p).unwrap().is_zero());
    }

    #[test]
    fn pow_past_the_dense_ceiling_is_refused_before_flint_allocates() {
        // PyUniPoly.__pow__((x+1), 2^31) aborted inside FLINT.
        let (p, x) = pool_and_var();
        let xp1 = UniPoly::from_symbolic(p.add(vec![x, p.integer(1_i32)]), x, &p).unwrap();
        assert_eq!(
            xp1.checked_pow(1 << 31).err(),
            Some(ConversionError::ExponentTooLarge)
        );
        assert_eq!(
            xp1.checked_pow(u32::MAX).err(),
            Some(ConversionError::ExponentTooLarge)
        );
        assert_eq!(xp1.checked_pow(3).unwrap(), xp1.pow(3));
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| xp1.pow(1 << 31)));
        assert!(r.is_err());
    }

    /// Follow-up to #413: the degree ceiling stopped `x^(2^31)`, but a genuine
    /// binomial to a moderate power — `(x+1)^(2^21)`, two million
    /// coefficients of up to two million bits, half a terabyte — went
    /// straight to `fmpz_poly_pow` and took the process down. Its size is now
    /// estimated first and the power refused with `E-POLY-004`.
    #[test]
    fn binomial_power_past_memory_is_refused_before_flint_allocates() {
        let (p, x) = pool_and_var();
        let xp1_expr = p.add(vec![x, p.integer(1_i32)]);
        let xp1 = UniPoly::from_symbolic(xp1_expr, x, &p).unwrap();
        let huge = 1_u32 << 21;
        // Only meaningful where half a terabyte is more than the machine has;
        // anywhere else the unbudgeted power is legitimately allowed.
        let fits = crate::budget::memory::physical_memory().is_none_or(|m| m > 1 << 39);
        if !fits {
            assert_eq!(
                xp1.checked_pow(huge).err(),
                Some(ConversionError::ExponentTooLarge)
            );
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| xp1.coeffs.pow(huge)));
            assert!(r.is_err(), "FlintPoly::pow must refuse, not allocate");
            // The same power reached through the symbolic converter.
            let e = p.pow(xp1_expr, p.integer(huge));
            assert_eq!(
                UniPoly::from_symbolic(e, x, &p).err(),
                Some(ConversionError::ExponentTooLarge)
            );
        }
        // A power that fits is untouched: C(4096, 2048) is the middle term.
        let small = xp1.checked_pow(4096).unwrap();
        assert_eq!(small.degree(), 4096);
        // ... and refused under a memory budget it would not fit (about
        // 2 MiB of coefficients against 1 MiB).
        let _g = crate::budget::enter_with_memory(crate::budget::Budget::default(), Some(1 << 20));
        assert_eq!(
            xp1.checked_pow(4096).err(),
            Some(ConversionError::ExponentTooLarge)
        );
        let e = p.pow(xp1_expr, p.integer(4096));
        assert_eq!(
            UniPoly::from_symbolic(e, x, &p).err(),
            Some(ConversionError::ExponentTooLarge)
        );
    }

    #[test]
    fn checked_mul_respects_the_dense_ceiling() {
        let (p, x) = pool_and_var();
        let f = UniPoly::from_symbolic(
            p.add(vec![p.pow(x, p.integer(1 << 16)), p.integer(1)]),
            x,
            &p,
        )
        .unwrap();
        assert_eq!(f.checked_mul(&f).unwrap(), &f * &f);
        let zero = UniPoly::zero(x);
        assert!(f.checked_mul(&zero).unwrap().is_zero());
        let _g = crate::budget::enter_with_memory(crate::budget::Budget::default(), Some(1 << 20));
        assert_eq!(
            f.checked_mul(&f).err(),
            Some(ConversionError::ExponentTooLarge)
        );
    }

    #[test]
    fn dense_degree_respects_the_memory_budget() {
        let (p, x) = pool_and_var();
        // 2^21 goes through the dense FLINT builder, 2^23 through the sparse
        // map; they need 16 and 64 MiB of slots, and a 1 MiB budget refuses
        // both before anything is allocated.
        let f = |e: u32| p.add(vec![p.pow(x, p.integer(e)), p.integer(1_i32)]);
        {
            let _g =
                crate::budget::enter_with_memory(crate::budget::Budget::default(), Some(1 << 20));
            for e in [1_u32 << 21, 1 << 23] {
                assert_eq!(
                    UniPoly::from_symbolic(f(e), x, &p).err(),
                    Some(ConversionError::ExponentTooLarge),
                    "degree {e}"
                );
            }
        }
        let got = UniPoly::from_symbolic(f(1 << 21), x, &p).unwrap();
        assert_eq!(got.degree(), 1 << 21);
    }

    #[test]
    fn from_symbolic_power_edge_cases() {
        let (p, x) = pool_and_var();
        let zero = p.integer(0_i32);
        // 0^0 = 1, as in the coefficient map.
        let e = p.pow(zero, zero);
        assert_eq!(
            UniPoly::from_symbolic(e, x, &p).unwrap().coefficients_i64(),
            vec![1]
        );
        // Exponent beyond u32.
        let e = p.pow(x, p.integer(1_u64 << 40));
        assert_eq!(
            UniPoly::from_symbolic(e, x, &p).err(),
            Some(ConversionError::ExponentTooLarge)
        );
        // The exponent is checked before the base: 1/2 inside a negative power.
        let e = p.pow(p.rational(1, 2), p.integer(-1_i32));
        assert_eq!(
            UniPoly::from_symbolic(e, x, &p).err(),
            Some(ConversionError::NegativeExponent)
        );
    }

    /// Timing comparison, run by hand:
    /// `cargo test --release -p alkahest-cas --lib unipoly::tests::timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn timing_from_symbolic_and_derivative() {
        fn best(mut f: impl FnMut()) -> f64 {
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        }
        let (p, x) = pool_and_var();
        for n in [400_i32, 1000] {
            let e = p.pow(p.add(vec![x, p.integer(1_i32)]), p.integer(n));
            let new = best(|| drop(expr_to_flintpoly(e, x, &p).unwrap()));
            let old = best(|| drop(reference(e, x, &p).unwrap()));
            println!("from_symbolic((x+1)^{n}): map {old:.3} ms, flint {new:.3} ms");
            let f = expr_to_flintpoly(e, x, &p).unwrap();
            let new = best(|| drop(f.derivative()));
            let old = best(|| drop(reference_derivative(&f)));
            println!("derivative deg {n}: per-coeff {old:.3} ms, fmpz_poly_derivative {new:.3} ms");
        }
        let two_x = p.mul(vec![p.integer(2_i32), x]);
        let (mut t0, mut t1) = (p.integer(1_i32), x);
        for n in 1..=24 {
            let t2 = p.add(vec![
                p.mul(vec![two_x, t1]),
                p.mul(vec![p.integer(-1_i32), t0]),
            ]);
            (t0, t1) = (t1, t2);
            if n % 4 == 0 {
                let new = best(|| drop(expr_to_flintpoly(t1, x, &p).unwrap()));
                let old = best(|| drop(reference(t1, x, &p).unwrap()));
                println!(
                    "Chebyshev T_{}: map {old:.3} ms, flint+memo {new:.3} ms",
                    n + 1
                );
            }
        }
    }
}
