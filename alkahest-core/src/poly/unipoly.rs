use super::error::ConversionError;
use crate::flint::{integer::FlintInteger, FlintPoly};
use crate::kernel::{ExprData, ExprId, ExprPool};
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

fn coeffmap_mul(a: &CoeffMap, b: &CoeffMap) -> CoeffMap {
    let mut result = CoeffMap::new();
    for (&da, ca) in a {
        for (&db, cb) in b {
            let prod = ca.clone() * cb.clone();
            if prod == 0 {
                continue;
            }
            let entry = result
                .entry(da + db)
                .or_insert_with(|| rug::Integer::from(0));
            *entry += prod;
            if *entry == 0 {
                result.remove(&(da + db));
            }
        }
    }
    result
}

fn coeffmap_pow(base: &CoeffMap, n: u32) -> CoeffMap {
    if n == 0 {
        let mut one = CoeffMap::new();
        one.insert(0, rug::Integer::from(1));
        return one;
    }
    if n == 1 {
        return base.clone();
    }
    let half = coeffmap_pow(base, n / 2);
    let mut result = coeffmap_mul(&half, &half);
    if n % 2 == 1 {
        result = coeffmap_mul(&result, base);
    }
    result
}

fn coeffmap_to_flintpoly(map: &CoeffMap) -> FlintPoly {
    let mut poly = FlintPoly::new();
    for (&deg, coeff) in map {
        let fi = FlintInteger::from_rug(coeff);
        poly.set_coeff_flint(deg as usize, &fi);
    }
    poly
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

fn coeffmap_rat_mul(a: &CoeffRatMap, b: &CoeffRatMap) -> CoeffRatMap {
    let mut result = CoeffRatMap::new();
    for (&da, ca) in a {
        for (&db, cb) in b {
            let prod = ca.clone() * cb.clone();
            if prod == 0 {
                continue;
            }
            let entry = result.entry(da + db).or_insert_with(|| Rational::from(0));
            *entry += prod;
            if *entry == 0 {
                result.remove(&(da + db));
            }
        }
    }
    result
}

fn coeffmap_rat_pow(base: &CoeffRatMap, n: u32) -> CoeffRatMap {
    if n == 0 {
        let mut one = CoeffRatMap::new();
        one.insert(0, Rational::from(1));
        return one;
    }
    if n == 1 {
        return base.clone();
    }
    let half = coeffmap_rat_pow(base, n / 2);
    let mut result = coeffmap_rat_mul(&half, &half);
    if n % 2 == 1 {
        result = coeffmap_rat_mul(&result, base);
    }
    result
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
                acc = coeffmap_rat_mul(&acc, &sub);
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(n) => {
                if n.0 < 0 {
                    return Err(ConversionError::NegativeExponent);
                }
                let n_u32 = n.0.to_u32().ok_or(ConversionError::ExponentTooLarge)?;
                let base_coeffs = expr_to_univariate_rat_coeffs(base, var, pool)?;
                Ok(coeffmap_rat_pow(&base_coeffs, n_u32))
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
                acc = coeffmap_mul(&acc, &sub);
            }
            Ok(acc)
        }
        // Power with a constant non-negative integer exponent
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(n) => {
                if n.0 < 0 {
                    return Err(ConversionError::NegativeExponent);
                }
                let n_u32 = n.0.to_u32().ok_or(ConversionError::ExponentTooLarge)?;
                let base_coeffs = expr_to_univariate_coeffs(base, var, pool)?;
                Ok(coeffmap_pow(&base_coeffs, n_u32))
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
        let map = expr_to_univariate_coeffs(expr, var, pool)?;
        let coeffs = coeffmap_to_flintpoly(&map);
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
                let coeffs = coeffmap_to_flintpoly(&intmap);
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

    pub fn degree(&self) -> i64 {
        self.coeffs.degree()
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_zero()
    }

    pub fn pow(&self, exp: u32) -> Self {
        UniPoly {
            var: self.var,
            coeffs: self.coeffs.pow(exp),
        }
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
                    let x_pow = if deg == 1 {
                        var
                    } else {
                        pool.pow(var, exp_id)
                    };
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
}
