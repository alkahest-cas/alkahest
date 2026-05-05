use super::error::ConversionError;
use crate::flint::mpoly::{FlintMPoly, FlintMPolyCtx};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// Exponent vector: ascending by variable index.
// Invariant: trailing zeros are stripped so that the zero polynomial has no
// terms and the constant 1 has a single entry with key vec![].
// ---------------------------------------------------------------------------

type Exponents = Vec<u32>;
type TermMap = BTreeMap<Exponents, rug::Integer>;

fn termmap_add(mut a: TermMap, b: TermMap) -> TermMap {
    for (exp, coeff) in b {
        let entry = a
            .entry(exp.clone())
            .or_insert_with(|| rug::Integer::from(0));
        *entry += coeff;
        if *entry == 0 {
            a.remove(&exp);
        }
    }
    a
}

fn termmap_mul(a: &TermMap, b: &TermMap) -> TermMap {
    let mut result = TermMap::new();
    for (ea, ca) in a {
        for (eb, cb) in b {
            let prod = ca.clone() * cb.clone();
            if prod == 0 {
                continue;
            }
            let len = ea.len().max(eb.len());
            let mut exp = vec![0u32; len];
            for (i, &e) in ea.iter().enumerate() {
                exp[i] += e;
            }
            for (i, &e) in eb.iter().enumerate() {
                exp[i] += e;
            }
            // strip trailing zeros
            while exp.last() == Some(&0) {
                exp.pop();
            }
            let entry = result
                .entry(exp.clone())
                .or_insert_with(|| rug::Integer::from(0));
            *entry += prod;
            if *entry == 0 {
                result.remove(&exp);
            }
        }
    }
    result
}

fn termmap_neg(map: TermMap) -> TermMap {
    map.into_iter().map(|(k, v)| (k, -v)).collect()
}

fn termmap_pow(base: &TermMap, n: u32) -> TermMap {
    if n == 0 {
        let mut one = TermMap::new();
        one.insert(vec![], rug::Integer::from(1));
        return one;
    }
    if n == 1 {
        return base.clone();
    }
    let half = termmap_pow(base, n / 2);
    let mut result = termmap_mul(&half, &half);
    if n % 2 == 1 {
        result = termmap_mul(&result, base);
    }
    result
}

fn expr_to_multivariate_coeffs(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<TermMap, ConversionError> {
    // Extract node data in a single lock acquisition, then release before recursing.
    enum NodeInfo {
        Symbol { idx: Option<usize>, name: String },
        Integer(rug::Integer),
        NonIntCoeff,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow { base: ExprId, exp: ExprId },
        Func(String),
    }

    let info = pool.with(expr, |data| match data {
        ExprData::Symbol { name, .. } => NodeInfo::Symbol {
            idx: vars.iter().position(|&v| v == expr),
            name: name.clone(),
        },
        ExprData::Integer(n) => NodeInfo::Integer(n.0.clone()),
        ExprData::Rational(_) | ExprData::Float(_) => NodeInfo::NonIntCoeff,
        ExprData::Add(args) => NodeInfo::Add(args.clone()),
        ExprData::Mul(args) => NodeInfo::Mul(args.clone()),
        ExprData::Pow { base, exp } => NodeInfo::Pow {
            base: *base,
            exp: *exp,
        },
        ExprData::Func { name, .. } => NodeInfo::Func(name.clone()),
        ExprData::Piecewise { .. }
        | ExprData::Predicate { .. }
        | ExprData::Forall { .. }
        | ExprData::Exists { .. }
        | ExprData::BigO(_) => NodeInfo::Func("piecewise_or_predicate".to_string()),
    });

    match info {
        NodeInfo::Symbol { idx: Some(idx), .. } => {
            let mut exp = vec![0u32; idx + 1];
            exp[idx] = 1;
            let mut map = TermMap::new();
            map.insert(exp, rug::Integer::from(1));
            Ok(map)
        }
        NodeInfo::Symbol { name, .. } => Err(ConversionError::UnexpectedSymbol(name)),
        NodeInfo::Integer(n) => {
            let mut map = TermMap::new();
            if n != 0 {
                map.insert(vec![], n);
            }
            Ok(map)
        }
        NodeInfo::NonIntCoeff => Err(ConversionError::NonIntegerCoefficient),
        NodeInfo::Add(args) => {
            let mut acc = TermMap::new();
            for arg in args {
                let sub = expr_to_multivariate_coeffs(arg, vars, pool)?;
                acc = termmap_add(acc, sub);
            }
            Ok(acc)
        }
        NodeInfo::Mul(args) => {
            let mut acc: TermMap = {
                let mut m = TermMap::new();
                m.insert(vec![], rug::Integer::from(1));
                m
            };
            for arg in args {
                let sub = expr_to_multivariate_coeffs(arg, vars, pool)?;
                acc = termmap_mul(&acc, &sub);
            }
            Ok(acc)
        }
        NodeInfo::Pow { base, exp } => {
            // Read the exponent without holding the pool lock during recursion.
            let n = pool
                .with(exp, |data| match data {
                    ExprData::Integer(n) => Some(n.0.clone()),
                    _ => None,
                })
                .ok_or(ConversionError::NonConstantExponent)?;
            if n < 0 {
                return Err(ConversionError::NegativeExponent);
            }
            let n_u32 = n.to_u32().ok_or(ConversionError::ExponentTooLarge)?;
            let base_coeffs = expr_to_multivariate_coeffs(base, vars, pool)?;
            Ok(termmap_pow(&base_coeffs, n_u32))
        }
        NodeInfo::Func(name) => Err(ConversionError::NonPolynomialFunction(name)),
    }
}

// ---------------------------------------------------------------------------
// MultiPoly
// ---------------------------------------------------------------------------

/// Sparse multivariate polynomial over ℤ.
///
/// `vars` fixes the variable ordering; the exponent key `[e0, e1, …]` means
/// `vars[0]^e0 * vars[1]^e1 * …`.  Trailing zeros in the exponent vector are
/// always stripped so structural equality reduces to map equality.
#[derive(Clone, PartialEq, Eq)]
pub struct MultiPoly {
    pub vars: Vec<ExprId>,
    pub terms: TermMap,
}

impl MultiPoly {
    pub fn zero(vars: Vec<ExprId>) -> Self {
        MultiPoly {
            vars,
            terms: TermMap::new(),
        }
    }

    pub fn constant(vars: Vec<ExprId>, c: i64) -> Self {
        let mut terms = TermMap::new();
        if c != 0 {
            terms.insert(vec![], rug::Integer::from(c));
        }
        MultiPoly { vars, terms }
    }

    pub fn from_symbolic(
        expr: ExprId,
        vars: Vec<ExprId>,
        pool: &ExprPool,
    ) -> Result<Self, ConversionError> {
        let terms = expr_to_multivariate_coeffs(expr, &vars, pool)?;
        Ok(MultiPoly { vars, terms })
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn total_degree(&self) -> u32 {
        self.terms
            .keys()
            .map(|exp| exp.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// GCD of all integer coefficients (content). Returns 0 for the zero polynomial.
    pub fn integer_content(&self) -> rug::Integer {
        self.terms.values().fold(rug::Integer::from(0), |acc, c| {
            rug::Integer::from(acc.gcd_ref(c))
        })
    }

    /// Primitive part: divide all coefficients by the integer content.
    pub fn primitive_part(&self) -> Self {
        let g = self.integer_content();
        if g == 0 {
            return self.clone();
        }
        self.div_integer(&g)
    }

    /// Returns `true` if both polynomials have the same variable list and can be combined.
    pub fn compatible_with(&self, other: &Self) -> bool {
        self.vars == other.vars
    }

    /// Compute the GCD of two compatible multivariate polynomials using FLINT.
    ///
    /// Returns `None` if the polynomials have different variable lists, if either
    /// is zero, or if FLINT's GCD algorithm fails (which is exceedingly rare).
    ///
    /// The returned GCD is normalised so that its leading coefficient is positive.
    pub fn gcd(&self, other: &Self) -> Option<Self> {
        if !self.compatible_with(other) {
            return None;
        }
        if self.is_zero() || other.is_zero() {
            return None;
        }

        let nvars = self.vars.len();

        // Build a FLINT context and convert both polynomials.
        let ctx = FlintMPolyCtx::new(nvars.max(1));

        let a = multi_to_flint(self, &ctx);
        let b = multi_to_flint(other, &ctx);

        let g = a.gcd(&b, &ctx)?;

        // Convert the GCD back to MultiPoly
        let terms = g.terms(nvars.max(1), &ctx);
        let mut gcd = MultiPoly {
            vars: self.vars.clone(),
            terms,
        };

        // Normalise: make the leading coefficient positive
        if let Some((_, lc)) = gcd.terms.iter().next_back() {
            if *lc < 0 {
                gcd = -gcd;
            }
        }

        Some(gcd)
    }

    /// Convert back to a symbolic expression in the given pool.
    ///
    /// Produces a canonical sum-of-products: each term is `coeff * var[0]^e0 * var[1]^e1 * …`.
    /// The zero polynomial maps to `Integer(0)`.
    pub fn to_expr(&self, pool: &ExprPool) -> ExprId {
        if self.terms.is_empty() {
            return pool.integer(0_i32);
        }
        let summands: Vec<ExprId> = self
            .terms
            .iter()
            .map(|(exps, coeff)| {
                let coeff_id = pool.integer(coeff.clone());
                let mut factors = vec![coeff_id];
                for (i, &e) in exps.iter().enumerate() {
                    if e == 0 || i >= self.vars.len() {
                        continue;
                    }
                    let var = self.vars[i];
                    let exp_id = pool.integer(e);
                    factors.push(if e == 1 { var } else { pool.pow(var, exp_id) });
                }
                match factors.len() {
                    0 => pool.integer(1_i32),
                    1 => factors[0],
                    _ => pool.mul(factors),
                }
            })
            .collect();

        match summands.len() {
            0 => pool.integer(0_i32),
            1 => summands[0],
            _ => pool.add(summands),
        }
    }

    /// Divide all coefficients by `d` (exact division — caller ensures divisibility).
    pub fn div_integer(&self, d: &rug::Integer) -> Self {
        debug_assert!(
            self.terms.values().all(|v| v.is_divisible(d)),
            "div_integer: not all coefficients are divisible by {d}"
        );
        let terms = self
            .terms
            .iter()
            .map(|(k, v)| (k.clone(), rug::Integer::from(v.div_exact_ref(d))))
            .collect();
        MultiPoly {
            vars: self.vars.clone(),
            terms,
        }
    }
}

/// Convert a `MultiPoly` to a `FlintMPoly` in the given context.
pub(crate) fn multi_to_flint_pub(p: &MultiPoly, ctx: &FlintMPolyCtx) -> FlintMPoly {
    multi_to_flint(p, ctx)
}

fn multi_to_flint(p: &MultiPoly, ctx: &FlintMPolyCtx) -> FlintMPoly {
    let nvars = p.vars.len().max(1);
    let mut fp = FlintMPoly::new(ctx);
    for (exp, coeff) in &p.terms {
        let mut exp_u64 = vec![0u64; nvars];
        for (i, &e) in exp.iter().enumerate() {
            if i < nvars {
                exp_u64[i] = e as u64;
            }
        }
        fp.push_term(coeff, &exp_u64, ctx);
    }
    fp.finish(ctx);
    fp
}

fn same_vars(a: &MultiPoly, b: &MultiPoly) {
    assert_eq!(
        a.vars, b.vars,
        "MultiPoly arithmetic requires both operands to share the same variable list"
    );
}

impl Neg for MultiPoly {
    type Output = Self;
    fn neg(self) -> Self {
        MultiPoly {
            vars: self.vars,
            terms: termmap_neg(self.terms),
        }
    }
}

impl Add for MultiPoly {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        same_vars(&self, &rhs);
        MultiPoly {
            vars: self.vars.clone(),
            terms: termmap_add(self.terms, rhs.terms),
        }
    }
}

impl Sub for MultiPoly {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        same_vars(&self, &rhs);
        MultiPoly {
            vars: self.vars.clone(),
            terms: termmap_add(self.terms, termmap_neg(rhs.terms)),
        }
    }
}

impl Mul for MultiPoly {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        same_vars(&self, &rhs);
        MultiPoly {
            vars: self.vars.clone(),
            terms: termmap_mul(&self.terms, &rhs.terms),
        }
    }
}

impl fmt::Display for MultiPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        // BTreeMap iterates in lexicographic key order (lowest degree first)
        for (exp, coeff) in &self.terms {
            if !first {
                if *coeff > 0 {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                }
            } else if *coeff < 0 {
                write!(f, "-")?;
            }
            first = false;

            let abs_coeff = rug::Integer::from(coeff.abs_ref());
            let has_vars = exp.iter().any(|&e| e > 0);
            if abs_coeff != 1 || !has_vars {
                write!(f, "{abs_coeff}")?;
            }
            for (i, &e) in exp.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                // Use generic xi notation since we don't have ExprPool here
                let var_label = format!("x{i}");
                if e == 1 {
                    write!(f, "{var_label}")?;
                } else {
                    write!(f, "{var_label}^{e}")?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for MultiPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultiPoly(vars={:?}, {})", self.vars, self)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn pool_xy() -> (ExprPool, ExprId, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        (p, x, y)
    }

    #[test]
    fn univariate_from_symbolic() {
        // x^2 + 2x + 1
        let (p, x, y) = pool_xy();
        let xsq = p.pow(x, p.integer(2_i32));
        let two_x = p.mul(vec![p.integer(2_i32), x]);
        let expr = p.add(vec![xsq, two_x, p.integer(1_i32)]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &p).unwrap();
        // constant term
        assert_eq!(poly.terms[&vec![]], rug::Integer::from(1));
        // x^1 term
        assert_eq!(poly.terms[&vec![1]], rug::Integer::from(2));
        // x^2 term
        assert_eq!(poly.terms[&vec![2]], rug::Integer::from(1));
    }

    #[test]
    fn bivariate_from_symbolic() {
        // x*y
        let (p, x, y) = pool_xy();
        let expr = p.mul(vec![x, y]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &p).unwrap();
        assert_eq!(poly.terms[&vec![1, 1]], rug::Integer::from(1));
        assert_eq!(poly.terms.len(), 1);
    }

    #[test]
    fn zero_poly() {
        let (_p, x, y) = pool_xy();
        let zero = MultiPoly::zero(vec![x, y]);
        assert!(zero.is_zero());
    }

    #[test]
    fn add_polys() {
        let (p, x, y) = pool_xy();
        let a = MultiPoly::from_symbolic(x, vec![x, y], &p).unwrap();
        let b = MultiPoly::from_symbolic(y, vec![x, y], &p).unwrap();
        let sum = a + b;
        assert_eq!(sum.terms[&vec![1]], rug::Integer::from(1)); // x
        assert_eq!(sum.terms[&vec![0, 1]], rug::Integer::from(1)); // y
    }

    #[test]
    fn mul_polys() {
        // (x + 1) * (x - 1) = x^2 - 1
        let (p, x, y) = pool_xy();
        let a = MultiPoly::from_symbolic(p.add(vec![x, p.integer(1_i32)]), vec![x, y], &p).unwrap();
        let b =
            MultiPoly::from_symbolic(p.add(vec![x, p.integer(-1_i32)]), vec![x, y], &p).unwrap();
        let prod = a * b;
        assert_eq!(prod.terms[&vec![]], rug::Integer::from(-1));
        assert_eq!(prod.terms[&vec![2]], rug::Integer::from(1));
        assert!(!prod.terms.contains_key(&vec![1]));
    }

    #[test]
    fn integer_content() {
        // 6x + 4 → content = 2
        let (p, x, y) = pool_xy();
        let expr = p.add(vec![p.mul(vec![p.integer(6_i32), x]), p.integer(4_i32)]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &p).unwrap();
        assert_eq!(poly.integer_content(), rug::Integer::from(2));
    }

    #[test]
    fn primitive_part() {
        // 6x + 4 → primitive part = 3x + 2
        let (p, x, y) = pool_xy();
        let expr = p.add(vec![p.mul(vec![p.integer(6_i32), x]), p.integer(4_i32)]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &p).unwrap();
        let pp = poly.primitive_part();
        assert_eq!(pp.terms[&vec![]], rug::Integer::from(2));
        assert_eq!(pp.terms[&vec![1]], rug::Integer::from(3));
    }

    #[test]
    fn free_symbol_error() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.symbol("z", Domain::Real);
        let expr = p.add(vec![x, z]);
        assert!(matches!(
            MultiPoly::from_symbolic(expr, vec![x], &p),
            Err(ConversionError::UnexpectedSymbol(_))
        ));
    }
}
