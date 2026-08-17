//! Sparse multivariate polynomials over ℚ.
//!
//! [`crate::poly::MultiPoly`] is the project's workhorse sparse polynomial, but
//! it is over ℤ.  Positivity certificates are intrinsically rational — the
//! multipliers `σ_j` in `p = Σ σ_j q_j²` almost never come out integral — so
//! this module carries its own exact ℚ representation.  Everything here is
//! closed under the operations the certificate verifier needs (add, multiply,
//! scale, square) and nothing here ever rounds.

use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::{Integer, Rational};
use std::collections::BTreeMap;
use std::fmt;

/// Exponent vector, one entry per variable (fixed-length, no trailing-zero
/// stripping — the length always equals `nvars`, which keeps the monomial
/// ordering total and the arithmetic branch-free).
pub type Exponents = Vec<u32>;

/// A sparse polynomial over ℚ in a fixed number of variables.
///
/// Invariant: `terms` never contains a zero coefficient, and every key has
/// length exactly `nvars`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RatPoly {
    nvars: usize,
    terms: BTreeMap<Exponents, Rational>,
}

impl RatPoly {
    pub fn zero(nvars: usize) -> Self {
        RatPoly {
            nvars,
            terms: BTreeMap::new(),
        }
    }

    pub fn constant(nvars: usize, c: Rational) -> Self {
        let mut p = RatPoly::zero(nvars);
        if c != 0 {
            p.terms.insert(vec![0; nvars], c);
        }
        p
    }

    pub fn one(nvars: usize) -> Self {
        RatPoly::constant(nvars, Rational::from(1))
    }

    /// `x_1² + … + x_nvars²` — the building block of the Reznick multiplier
    /// family `(x_1² + … + x_nvars²)^N`. Zero only at the origin and strictly
    /// positive everywhere else, which is exactly what the multiplier
    /// certificate's verification argument needs.
    pub fn sum_of_squares(nvars: usize) -> Self {
        let mut p = RatPoly::zero(nvars);
        for i in 0..nvars {
            let mut e = vec![0u32; nvars];
            e[i] = 2;
            p.terms.insert(e, Rational::from(1));
        }
        p
    }

    /// `coeff * x^exps`.
    pub fn monomial(nvars: usize, exps: Exponents, coeff: Rational) -> Self {
        debug_assert_eq!(exps.len(), nvars);
        let mut p = RatPoly::zero(nvars);
        if coeff != 0 {
            p.terms.insert(exps, coeff);
        }
        p
    }

    pub fn nvars(&self) -> usize {
        self.nvars
    }

    pub fn terms(&self) -> &BTreeMap<Exponents, Rational> {
        &self.terms
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// The polynomial's value viewed as a rational constant, if it is one.
    pub fn as_constant(&self) -> Option<Rational> {
        match self.terms.len() {
            0 => Some(Rational::from(0)),
            1 => {
                let (e, c) = self.terms.iter().next().unwrap();
                if e.iter().all(|&v| v == 0) {
                    Some(c.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// `Some(d)` iff every term has total degree exactly `d` (a form). The
    /// zero polynomial is homogeneous of every degree, so it is reported
    /// homogeneous of degree `0` (harmless: callers only use this to decide
    /// whether a *non-trivial* degree restriction on a monomial basis is
    /// still complete, and a zero target is handled separately everywhere
    /// upstream of that decision).
    pub fn is_homogeneous(&self) -> Option<u32> {
        let mut degs = self.terms.keys().map(|e| e.iter().sum::<u32>());
        let first = degs.next().unwrap_or(0);
        if degs.all(|d| d == first) {
            Some(first)
        } else {
            None
        }
    }

    pub fn total_degree(&self) -> u32 {
        self.terms
            .keys()
            .map(|e| e.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// Highest exponent of variable `i` occurring anywhere.
    pub fn degree_in(&self, i: usize) -> u32 {
        self.terms.keys().map(|e| e[i]).max().unwrap_or(0)
    }

    pub fn coeff(&self, exps: &[u32]) -> Rational {
        self.terms
            .get(exps)
            .cloned()
            .unwrap_or_else(|| Rational::from(0))
    }

    fn insert_add(&mut self, exps: Exponents, c: Rational) {
        if c == 0 {
            return;
        }
        match self.terms.get_mut(&exps) {
            Some(slot) => {
                *slot += c;
                if *slot == 0 {
                    self.terms.remove(&exps);
                }
            }
            None => {
                self.terms.insert(exps, c);
            }
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.nvars, other.nvars);
        let mut out = self.clone();
        for (e, c) in &other.terms {
            out.insert_add(e.clone(), c.clone());
        }
        out
    }

    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.nvars, other.nvars);
        let mut out = self.clone();
        for (e, c) in &other.terms {
            out.insert_add(e.clone(), -c.clone());
        }
        out
    }

    pub fn neg(&self) -> Self {
        RatPoly {
            nvars: self.nvars,
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), -c.clone()))
                .collect(),
        }
    }

    pub fn scale(&self, k: &Rational) -> Self {
        if *k == 0 {
            return RatPoly::zero(self.nvars);
        }
        RatPoly {
            nvars: self.nvars,
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), Rational::from(c * k)))
                .collect(),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.nvars, other.nvars);
        let mut out = RatPoly::zero(self.nvars);
        for (ea, ca) in &self.terms {
            for (eb, cb) in &other.terms {
                let e: Exponents = ea.iter().zip(eb).map(|(a, b)| a + b).collect();
                out.insert_add(e, Rational::from(ca * cb));
            }
        }
        out
    }

    pub fn square(&self) -> Self {
        self.mul(self)
    }

    pub fn pow(&self, n: u32) -> Self {
        let mut acc = RatPoly::one(self.nvars);
        for _ in 0..n {
            acc = acc.mul(self);
        }
        acc
    }

    /// Evaluate at a rational point (one value per variable).
    pub fn eval(&self, point: &[Rational]) -> Rational {
        debug_assert_eq!(point.len(), self.nvars);
        let mut acc = Rational::from(0);
        for (e, c) in &self.terms {
            let mut term = c.clone();
            for (i, &k) in e.iter().enumerate() {
                for _ in 0..k {
                    term *= &point[i];
                }
            }
            acc += term;
        }
        acc
    }

    /// Least common multiple of all coefficient denominators.
    pub fn denominator_lcm(&self) -> Integer {
        let mut l = Integer::from(1);
        for c in self.terms.values() {
            l.lcm_mut(c.denom());
        }
        l
    }

    // -----------------------------------------------------------------------
    // Conversion
    // -----------------------------------------------------------------------

    /// Convert a symbolic expression to a rational polynomial in `vars`.
    ///
    /// Returns `Err(reason)` for anything that is not a polynomial with
    /// rational coefficients in exactly those variables.
    pub fn from_expr(expr: ExprId, vars: &[ExprId], pool: &ExprPool) -> Result<Self, String> {
        let nvars = vars.len();
        pool.with(expr, |data| match data {
            ExprData::Integer(n) => Ok(RatPoly::constant(nvars, Rational::from(n.0.clone()))),
            ExprData::Rational(r) => Ok(RatPoly::constant(nvars, r.0.clone())),
            ExprData::Symbol { name, .. } => match vars.iter().position(|&v| v == expr) {
                Some(i) => {
                    let mut e = vec![0; nvars];
                    e[i] = 1;
                    Ok(RatPoly::monomial(nvars, e, Rational::from(1)))
                }
                None => Err(format!(
                    "symbol `{name}` is not among the declared variables; \
                     pass it in `vars` or eliminate it first"
                )),
            },
            ExprData::Add(args) => {
                let mut acc = RatPoly::zero(nvars);
                for &a in args {
                    acc = acc.add(&RatPoly::from_expr(a, vars, pool)?);
                }
                Ok(acc)
            }
            ExprData::Mul(args) => {
                let mut acc = RatPoly::one(nvars);
                for &a in args {
                    acc = acc.mul(&RatPoly::from_expr(a, vars, pool)?);
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp } => {
                let k = pool.with(*exp, |d| match d {
                    ExprData::Integer(n) => n.0.to_i32(),
                    _ => None,
                });
                match k {
                    Some(k) if k >= 0 => {
                        let b = RatPoly::from_expr(*base, vars, pool)?;
                        Ok(b.pow(k as u32))
                    }
                    // A negative exponent is still polynomial when the base is a
                    // constant: `x**2 / 4` parses as `x**2 * 4**(-1)`, and
                    // rational coefficients written as divisions are the common
                    // case, not an exotic one.
                    Some(k) => {
                        let b = RatPoly::from_expr(*base, vars, pool)?;
                        let c = b.as_constant().ok_or_else(|| {
                            "a negative exponent is only polynomial when its base is constant; \
                             clear the denominator first (multiply through)"
                                .to_string()
                        })?;
                        if c == 0 {
                            return Err("division by zero in the target polynomial".to_string());
                        }
                        let mut acc = Rational::from(1);
                        for _ in 0..k.unsigned_abs() {
                            acc /= c.clone();
                        }
                        Ok(RatPoly::constant(nvars, acc))
                    }
                    None => Err(
                        "only integer exponents are supported in positivity certificates"
                            .to_string(),
                    ),
                }
            }
            ExprData::Float(_) => Err("floating-point coefficients are not exact; \
                                       rationalize them before certifying positivity"
                .to_string()),
            other => Err(format!(
                "expression is not polynomial (node: {})",
                node_kind(other)
            )),
        })
    }

    /// Convert back to a symbolic expression.
    pub fn to_expr(&self, vars: &[ExprId], pool: &ExprPool) -> ExprId {
        if self.terms.is_empty() {
            return pool.integer(0);
        }
        let one = Rational::from(1);
        let summands: Vec<ExprId> = self
            .terms
            .iter()
            .rev()
            .map(|(exps, c)| {
                let mut factors: Vec<ExprId> = Vec::new();
                let is_unit = *c == one;
                let has_vars = exps.iter().any(|&e| e > 0);
                if !is_unit || !has_vars {
                    factors.push(rational_expr(c, pool));
                }
                for (i, &e) in exps.iter().enumerate() {
                    if e == 0 {
                        continue;
                    }
                    factors.push(if e == 1 {
                        vars[i]
                    } else {
                        let ex = pool.integer(e);
                        pool.pow(vars[i], ex)
                    });
                }
                match factors.len() {
                    0 => pool.integer(1),
                    1 => factors[0],
                    _ => pool.mul(factors),
                }
            })
            .collect();
        match summands.len() {
            1 => summands[0],
            _ => pool.add(summands),
        }
    }

    /// Human-readable rendering with the given variable names.
    pub fn display(&self, names: &[String]) -> String {
        if self.terms.is_empty() {
            return "0".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        for (exps, c) in self.terms.iter().rev() {
            let has_vars = exps.iter().any(|&e| e > 0);
            let mut s = String::new();
            if !has_vars || *c != 1 {
                s.push_str(&format_rational(c));
                if has_vars {
                    s.push('*');
                }
            }
            let mut first = true;
            for (i, &e) in exps.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                if !first {
                    s.push('*');
                }
                first = false;
                s.push_str(&names[i]);
                if e > 1 {
                    s.push_str(&format!("^{e}"));
                }
            }
            parts.push(s);
        }
        parts.join(" + ")
    }

    /// Lean 4 term rendering.  Uses full-precision integer literals (never
    /// truncates to machine words) so the emitted certificate always states
    /// exactly the identity that was verified.
    pub fn to_lean(&self, names: &[String]) -> String {
        if self.terms.is_empty() {
            return "(0 : ℝ)".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        for (exps, c) in self.terms.iter().rev() {
            let mut factors: Vec<String> = Vec::new();
            let has_vars = exps.iter().any(|&e| e > 0);
            if !has_vars || *c != 1 {
                factors.push(lean_rational(c));
            }
            for (i, &e) in exps.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                factors.push(if e == 1 {
                    format!("({} : ℝ)", names[i])
                } else {
                    format!("({} : ℝ) ^ ({e} : ℕ)", names[i])
                });
            }
            parts.push(factors.join(" * "));
        }
        format!("({})", parts.join(" + "))
    }
}

fn node_kind(data: &ExprData) -> &'static str {
    match data {
        ExprData::Func { .. } => "function application",
        ExprData::Predicate { .. } => "predicate",
        ExprData::Piecewise { .. } => "piecewise",
        ExprData::Forall { .. } | ExprData::Exists { .. } => "quantifier",
        ExprData::BigO(_) => "big-O",
        ExprData::RootSum { .. } => "root sum",
        _ => "unsupported",
    }
}

pub(crate) fn rational_expr(c: &Rational, pool: &ExprPool) -> ExprId {
    if *c.denom() == 1 {
        pool.integer(c.numer().clone())
    } else {
        pool.rational(c.numer().clone(), c.denom().clone())
    }
}

pub(crate) fn format_rational(c: &Rational) -> String {
    if *c.denom() == 1 {
        c.numer().to_string()
    } else {
        format!("{}/{}", c.numer(), c.denom())
    }
}

pub(crate) fn lean_rational(c: &Rational) -> String {
    if *c.denom() == 1 {
        format!("({} : ℝ)", c.numer())
    } else {
        format!("(({} : ℝ) / ({} : ℝ))", c.numer(), c.denom())
    }
}

impl fmt::Display for RatPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<String> = (0..self.nvars).map(|i| format!("x{i}")).collect();
        write!(f, "{}", self.display(&names))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn r(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    #[test]
    fn add_mul_are_exact() {
        // (x + 1/2)^2 = x^2 + x + 1/4
        let x = RatPoly::monomial(1, vec![1], Rational::from(1));
        let half = RatPoly::constant(1, r(1, 2));
        let sq = x.add(&half).square();
        assert_eq!(sq.coeff(&[2]), Rational::from(1));
        assert_eq!(sq.coeff(&[1]), Rational::from(1));
        assert_eq!(sq.coeff(&[0]), r(1, 4));
    }

    #[test]
    fn cancellation_removes_terms() {
        let x = RatPoly::monomial(1, vec![1], Rational::from(1));
        assert!(x.sub(&x).is_zero());
    }

    #[test]
    fn eval_matches_expansion() {
        // p = 2x^2 y - y + 3
        let mut p = RatPoly::zero(2);
        p = p.add(&RatPoly::monomial(2, vec![2, 1], Rational::from(2)));
        p = p.add(&RatPoly::monomial(2, vec![0, 1], Rational::from(-1)));
        p = p.add(&RatPoly::constant(2, Rational::from(3)));
        // at (2, 3): 2*4*3 - 3 + 3 = 24
        assert_eq!(p.eval(&[Rational::from(2), Rational::from(3)]), 24);
    }

    #[test]
    fn from_expr_roundtrip() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // (x - y)^2
        let two = pool.integer(2);
        let neg = pool.integer(-1);
        let diff = pool.add(vec![x, pool.mul(vec![neg, y])]);
        let e = pool.pow(diff, two);
        let p = RatPoly::from_expr(e, &[x, y], &pool).unwrap();
        assert_eq!(p.coeff(&[2, 0]), 1);
        assert_eq!(p.coeff(&[1, 1]), -2);
        assert_eq!(p.coeff(&[0, 2]), 1);

        let back = p.to_expr(&[x, y], &pool);
        let q = RatPoly::from_expr(back, &[x, y], &pool).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn from_expr_accepts_rational_literals() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let quarter = pool.rational(1, 4);
        let e = pool.add(vec![pool.pow(x, pool.integer(2)), quarter]);
        let p = RatPoly::from_expr(e, &[x], &pool).unwrap();
        assert_eq!(p.coeff(&[0]), r(1, 4));
    }

    #[test]
    fn from_expr_rejects_transcendental() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let s = pool.func("sin", vec![x]);
        assert!(RatPoly::from_expr(s, &[x], &pool).is_err());
    }

    #[test]
    fn from_expr_rejects_undeclared_symbol() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = pool.add(vec![x, y]);
        assert!(RatPoly::from_expr(e, &[x], &pool).is_err());
    }
}
