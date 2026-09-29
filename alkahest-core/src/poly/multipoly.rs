use super::error::ConversionError;
use crate::flint::mpoly::{FlintMPoly, FlintMPolyCtx};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

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

/// Schoolbook product over the `BTreeMap`. The reference algorithm, and the
/// faster one for small operands.
fn termmap_mul_schoolbook(a: &TermMap, b: &TermMap) -> TermMap {
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

/// Power by repeated squaring with the schoolbook product. The reference
/// algorithm, and the faster one for small operands.
fn termmap_pow_schoolbook(base: &TermMap, n: u32) -> TermMap {
    if n == 0 {
        let mut one = TermMap::new();
        one.insert(vec![], rug::Integer::from(1));
        return one;
    }
    if n == 1 {
        return base.clone();
    }
    let half = termmap_pow_schoolbook(base, n / 2);
    let mut result = termmap_mul_schoolbook(&half, &half);
    if n % 2 == 1 {
        result = termmap_mul_schoolbook(&result, base);
    }
    result
}

// ---------------------------------------------------------------------------
// FLINT-backed product and power.
//
// Above a small size the `BTreeMap` schoolbook loop (two rug clones and an
// exponent `Vec` per term pair) loses to converting both operands to
// `fmpz_mpoly`, multiplying there, and converting back. The result map is the
// same: `fmpz_mpoly` drops zero terms and combines like monomials, and
// `FlintMPoly::terms` strips trailing zero exponents, exactly the invariants
// the schoolbook loop maintains. The FLINT context is sized from the longest
// exponent key actually present (not from `vars`), so no exponent is dropped.
// ---------------------------------------------------------------------------

/// Minimum `|a|·|b|` term pairs for [`termmap_mul`] to use FLINT. Measured
/// (release, three variables): dense operands, whose products combine, favour
/// FLINT from about 100 pairs; fully sparse ones, dominated by converting every
/// result term back, break even between 256 and 4096 pairs and are within ~20%
/// below that.
const FLINT_MUL_MIN_PAIRS: usize = 128;

/// Minimum `|base|^n` for [`termmap_pow`] to use FLINT. Measured: 2 terms `^3`
/// and 3 terms `^2` are faster in the schoolbook loop; 2 terms `^8`, 3 terms
/// `^4` and 4 terms `^3` in FLINT.
const FLINT_POW_MIN_WORK: u64 = 16;

/// Per-variable maximum exponent over the keys of `t`, padded to `nvars`.
fn max_exponents(t: &TermMap, nvars: usize) -> Vec<u64> {
    let mut m = vec![0u64; nvars];
    for exp in t.keys() {
        for (slot, &e) in m.iter_mut().zip(exp) {
            *slot = (*slot).max(u64::from(e));
        }
    }
    m
}

fn termmap_nvars(t: &TermMap) -> usize {
    t.keys().map(Vec::len).max().unwrap_or(0)
}

fn termmap_to_flint(t: &TermMap, ctx: &Arc<FlintMPolyCtx>) -> FlintMPoly {
    let nvars = ctx.nvars();
    let mut fp = FlintMPoly::new(Arc::clone(ctx));
    let mut e = vec![0u64; nvars];
    for (exp, c) in t {
        e.iter_mut().for_each(|v| *v = 0);
        for (slot, &x) in e.iter_mut().zip(exp) {
            *slot = u64::from(x);
        }
        fp.push_term(c, &e);
    }
    fp.finish();
    fp
}

/// `a * b` through FLINT, or `None` when some result exponent would not fit
/// the `u32` keys (the schoolbook loop then reproduces the old behaviour).
fn termmap_mul_flint(a: &TermMap, b: &TermMap) -> Option<TermMap> {
    let nvars = termmap_nvars(a).max(termmap_nvars(b)).max(1);
    let (ma, mb) = (max_exponents(a, nvars), max_exponents(b, nvars));
    if ma.iter().zip(&mb).any(|(x, y)| x + y > u64::from(u32::MAX)) {
        return None;
    }
    let ctx = FlintMPolyCtx::new(nvars);
    let prod = termmap_to_flint(a, &ctx).mul(&termmap_to_flint(b, &ctx));
    Some(prod.terms())
}

/// `base^n` (`n ≥ 2`) through FLINT, or `None` on exponent overflow or a FLINT
/// failure.
fn termmap_pow_flint(base: &TermMap, n: u32) -> Option<TermMap> {
    let nvars = termmap_nvars(base).max(1);
    if max_exponents(base, nvars)
        .iter()
        .any(|&x| x * u64::from(n) > u64::from(u32::MAX))
    {
        return None;
    }
    let ctx = FlintMPolyCtx::new(nvars);
    Some(termmap_to_flint(base, &ctx).pow_ui(u64::from(n))?.terms())
}

fn termmap_mul(a: &TermMap, b: &TermMap) -> TermMap {
    if a.len().saturating_mul(b.len()) >= FLINT_MUL_MIN_PAIRS {
        if let Some(r) = termmap_mul_flint(a, b) {
            return r;
        }
    }
    termmap_mul_schoolbook(a, b)
}

fn termmap_pow(base: &TermMap, n: u32) -> TermMap {
    let work = (base.len() as u64).saturating_pow(n);
    if n >= 2 && base.len() >= 2 && work >= FLINT_POW_MIN_WORK {
        if let Some(r) = termmap_pow_flint(base, n) {
            return r;
        }
    }
    termmap_pow_schoolbook(base, n)
}

/// Per-call state for [`expr_to_multivariate_coeffs`].
struct MultiBuild {
    /// FLINT-backed product/power above the size thresholds; `false` gives the
    /// plain schoolbook algorithm (the test reference).
    fast: bool,
    /// Compound nodes converted at least once.
    seen: HashSet<ExprId>,
    /// Results for compound nodes reached a second time (shared DAG nodes), so
    /// each node is converted at most twice. Empty when `memoize` is off.
    memo: HashMap<ExprId, TermMap>,
    memoize: bool,
}

impl MultiBuild {
    fn new() -> Self {
        MultiBuild {
            fast: true,
            seen: HashSet::new(),
            memo: HashMap::new(),
            memoize: true,
        }
    }

    fn mul(&self, a: &TermMap, b: &TermMap) -> TermMap {
        if self.fast {
            termmap_mul(a, b)
        } else {
            termmap_mul_schoolbook(a, b)
        }
    }

    fn pow(&self, base: &TermMap, n: u32) -> TermMap {
        if self.fast {
            termmap_pow(base, n)
        } else {
            termmap_pow_schoolbook(base, n)
        }
    }
}

fn expr_to_multivariate_coeffs(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<TermMap, ConversionError> {
    build_multi(expr, vars, pool, &mut MultiBuild::new())
}

fn build_multi(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
    st: &mut MultiBuild,
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
        // A `Rational` node whose value is integral (denominator 1) can arise
        // from un-collapsed arithmetic; treat it as the integer numerator.
        ExprData::Rational(r) if *r.0.denom() == 1 => NodeInfo::Integer(r.0.numer().clone()),
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
        | ExprData::RootSum { .. }
        | ExprData::BigO(_) => NodeInfo::Func("piecewise_or_predicate".to_string()),
    });

    if matches!(
        info,
        NodeInfo::Add(_) | NodeInfo::Mul(_) | NodeInfo::Pow { .. }
    ) {
        if let Some(t) = st.memo.get(&expr) {
            return Ok(t.clone());
        }
    }

    let result = match info {
        NodeInfo::Symbol { idx: Some(idx), .. } => {
            let mut exp = vec![0u32; idx + 1];
            exp[idx] = 1;
            let mut map = TermMap::new();
            map.insert(exp, rug::Integer::from(1));
            return Ok(map);
        }
        NodeInfo::Symbol { name, .. } => return Err(ConversionError::UnexpectedSymbol(name)),
        NodeInfo::Integer(n) => {
            let mut map = TermMap::new();
            if n != 0 {
                map.insert(vec![], n);
            }
            return Ok(map);
        }
        NodeInfo::NonIntCoeff => return Err(ConversionError::NonIntegerCoefficient),
        NodeInfo::Add(args) => {
            let mut acc = TermMap::new();
            for arg in args {
                let sub = build_multi(arg, vars, pool, st)?;
                acc = termmap_add(acc, sub);
            }
            acc
        }
        NodeInfo::Mul(args) => {
            let mut acc: TermMap = {
                let mut m = TermMap::new();
                m.insert(vec![], rug::Integer::from(1));
                m
            };
            for arg in args {
                let sub = build_multi(arg, vars, pool, st)?;
                acc = st.mul(&acc, &sub);
            }
            acc
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
            let base_coeffs = build_multi(base, vars, pool, st)?;
            st.pow(&base_coeffs, n_u32)
        }
        NodeInfo::Func(name) => return Err(ConversionError::NonPolynomialFunction(name)),
    };
    if st.memoize && !st.seen.insert(expr) {
        st.memo.insert(expr, result.clone());
    }
    Ok(result)
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

        let a = multi_to_flint(self, Arc::clone(&ctx));
        let b = multi_to_flint(other, Arc::clone(&ctx));

        let g = a.gcd(&b)?;

        // Convert the GCD back to MultiPoly
        let terms = g.terms();
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
        let one = rug::Integer::from(1);
        let summands: Vec<ExprId> = self
            .terms
            .iter()
            .map(|(exps, coeff)| {
                let mut factors = Vec::new();
                // Omit a unit coefficient when there are variable factors so
                // cancel((x²-1)/(x-1)) prints as `x + 1`, not `1 + (x * 1)`.
                if coeff != &one {
                    factors.push(pool.integer(coeff.clone()));
                }
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

    /// Degree of `self` in `vars[idx]` (0 for the zero polynomial).
    pub fn degree_in(&self, idx: usize) -> u32 {
        self.terms
            .keys()
            .map(|exp| exp.get(idx).copied().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }

    /// Partial derivative with respect to `vars[idx]`.
    pub fn partial_derivative(&self, idx: usize) -> Self {
        let mut terms = TermMap::new();
        for (exp, coeff) in &self.terms {
            let e = exp.get(idx).copied().unwrap_or(0);
            if e == 0 {
                continue;
            }
            let mut new_exp = exp.clone();
            new_exp[idx] = e - 1;
            while new_exp.last() == Some(&0) {
                new_exp.pop();
            }
            let c = coeff.clone() * rug::Integer::from(e);
            let entry = terms
                .entry(new_exp.clone())
                .or_insert_with(|| rug::Integer::from(0));
            *entry += c;
            if *entry == 0 {
                terms.remove(&new_exp);
            }
        }
        MultiPoly {
            vars: self.vars.clone(),
            terms,
        }
    }

    /// Factor into irreducible factors over ℤ (FLINT's Bernardin–Monagan EEZ).
    ///
    /// Returns `(unit, [(factor, multiplicity), …])` such that `self` equals
    /// `unit` times the product of each `factor` raised to its `multiplicity`.
    /// Returns `None` for the zero polynomial or if FLINT's factoriser fails.
    pub fn factor_irreducible(&self) -> Option<(rug::Integer, Vec<(MultiPoly, u32)>)> {
        if self.is_zero() {
            return None;
        }
        let nvars = self.vars.len().max(1);
        let ctx = FlintMPolyCtx::new(nvars);
        let fp = multi_to_flint(self, Arc::clone(&ctx));
        let mut fac = crate::flint::mpoly::FlintMPolyFactor::new(Arc::clone(&ctx));
        if !fac.factor(&fp) || !fac.constant_den_is_one() {
            return None;
        }
        let unit = fac.unit().to_rug();
        let mut out = Vec::new();
        for i in 0..fac.len() {
            let base = fac.base_at(i);
            let mult = fac.exp_at(i);
            out.push((
                MultiPoly {
                    vars: self.vars.clone(),
                    terms: base.terms(),
                },
                mult,
            ));
        }
        Some((unit, out))
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
pub(crate) fn multi_to_flint_pub(p: &MultiPoly, ctx: Arc<FlintMPolyCtx>) -> FlintMPoly {
    multi_to_flint(p, ctx)
}

fn multi_to_flint(p: &MultiPoly, ctx: Arc<FlintMPolyCtx>) -> FlintMPoly {
    let nvars = p.vars.len().max(1);
    let mut fp = FlintMPoly::new(ctx);
    for (exp, coeff) in &p.terms {
        let mut exp_u64 = vec![0u64; nvars];
        for (i, &e) in exp.iter().enumerate() {
            if i < nvars {
                exp_u64[i] = e as u64;
            }
        }
        fp.push_term(coeff, &exp_u64);
    }
    fp.finish();
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

fn var_label(pool: &ExprPool, var: ExprId, index: usize) -> String {
    pool.with(var, |data| match data {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => format!("x{index}"),
    })
}

impl MultiPoly {
    /// Pretty-print using symbol names from *pool* (falls back to `x0`, `x1`, …).
    pub fn display_with(&self, pool: &ExprPool) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let mut out = String::new();
        let mut first = true;
        for (exp, coeff) in &self.terms {
            if !first {
                if *coeff > 0 {
                    out.push_str(" + ");
                } else {
                    out.push_str(" - ");
                }
            } else if *coeff < 0 {
                out.push('-');
            }
            first = false;

            let abs_coeff = rug::Integer::from(coeff.abs_ref());
            let has_vars = exp.iter().any(|&e| e > 0);
            if abs_coeff != 1 || !has_vars {
                out.push_str(&abs_coeff.to_string());
            }
            for (i, &e) in exp.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                let label = if i < self.vars.len() {
                    var_label(pool, self.vars[i], i)
                } else {
                    format!("x{i}")
                };
                if e == 1 {
                    out.push_str(&label);
                } else {
                    out.push_str(&format!("{label}^{e}"));
                }
            }
        }
        out
    }
}

impl fmt::Display for MultiPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
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
    fn display_with_uses_symbol_names() {
        let (p, x, y) = pool_xy();
        let expr = p.mul(vec![p.add(vec![x, y]), p.add(vec![x, p.integer(-1_i32)])]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &p).unwrap();
        let s = poly.display_with(&p);
        assert!(s.contains('x') && s.contains('y'));
        assert!(!s.contains("x0") && !s.contains("x1"));
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

    // --- FLINT product / power vs the schoolbook reference ---

    use proptest::prelude::*;

    /// A coefficient: small, word-sized or multi-limb, either sign, sometimes 0.
    fn coeff() -> impl Strategy<Value = rug::Integer> {
        prop_oneof![
            4 => (-9i64..=9).prop_map(rug::Integer::from),
            2 => any::<i64>().prop_map(rug::Integer::from),
            1 => (any::<i64>(), 64u32..300).prop_map(|(a, s)| rug::Integer::from(a) << s),
        ]
    }

    /// Arbitrary term map, not necessarily canonical: keys may carry trailing
    /// zeros and coefficients may be zero, as a hand-built `MultiPoly` can.
    fn raw_termmap(nvars: usize, max_terms: usize) -> impl Strategy<Value = TermMap> {
        prop::collection::btree_map(
            prop::collection::vec(0u32..6, 0..=nvars),
            coeff(),
            0..=max_terms,
        )
    }

    /// Canonical term map: trailing zeros stripped, zero coefficients dropped.
    fn termmap(nvars: usize, max_terms: usize) -> impl Strategy<Value = TermMap> {
        raw_termmap(nvars, max_terms).prop_map(|t| {
            let mut out = TermMap::new();
            for (mut k, c) in t {
                while k.last() == Some(&0) {
                    k.pop();
                }
                if c != 0 {
                    out = termmap_add(out, TermMap::from([(k, c)]));
                }
            }
            out
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn flint_mul_matches_schoolbook(
            (a, b) in (1usize..7).prop_flat_map(|n| (raw_termmap(n, 40), raw_termmap(n, 40))),
        ) {
            let expect = termmap_mul_schoolbook(&a, &b);
            prop_assert_eq!(termmap_mul_flint(&a, &b), Some(expect.clone()));
            prop_assert_eq!(termmap_mul(&a, &b), expect);
        }

        #[test]
        fn flint_pow_matches_schoolbook(
            base in (1usize..5).prop_flat_map(|n| termmap(n, 6)),
            n in 0u32..6,
        ) {
            let expect = termmap_pow_schoolbook(&base, n);
            if n >= 2 {
                prop_assert_eq!(termmap_pow_flint(&base, n), Some(expect.clone()));
            }
            prop_assert_eq!(termmap_pow(&base, n), expect);
        }

        /// `MultiPoly * MultiPoly` over the public type.
        #[test]
        fn multipoly_mul_matches_schoolbook(
            a in termmap(3, 30),
            b in termmap(3, 30),
        ) {
            let (_p, x, y) = pool_xy();
            let z = _p.symbol("z", Domain::Real);
            let vars = vec![x, y, z];
            let pa = MultiPoly { vars: vars.clone(), terms: a.clone() };
            let pb = MultiPoly { vars, terms: b.clone() };
            prop_assert_eq!((pa * pb).terms, termmap_mul_schoolbook(&a, &b));
        }

        /// `from_symbolic` (FLINT arithmetic, DAG memo) against the schoolbook,
        /// unmemoised walk: same map or same error.
        #[test]
        fn from_symbolic_matches_schoolbook(r in recipe()) {
            let p = ExprPool::new();
            let vs: Vec<ExprId> = ["x", "y", "z"].iter().map(|n| p.symbol(*n, Domain::Real)).collect();
            let w = p.symbol("w", Domain::Real);
            let e = build(&r, &p, &vs, w);
            let fast = expr_to_multivariate_coeffs(e, &vs, &p);
            let mut slow = MultiBuild::new();
            slow.fast = false;
            slow.memoize = false;
            prop_assert_eq!(fast, build_multi(e, &vs, &p, &mut slow));
        }
    }

    #[derive(Debug, Clone)]
    enum Recipe {
        Var(usize),
        Free,
        Int(i64),
        Big(i64, u32),
        Rat(i64, i64),
        Sin,
        Add(Vec<Recipe>),
        Mul(Vec<Recipe>),
        Pow(Box<Recipe>, i64),
    }

    fn recipe() -> impl Strategy<Value = Recipe> {
        let leaf = prop_oneof![
            10 => (0usize..3).prop_map(Recipe::Var),
            1 => Just(Recipe::Free),
            5 => (-5i64..=5).prop_map(Recipe::Int),
            1 => (any::<i64>(), 64u32..200).prop_map(|(a, s)| Recipe::Big(a, s)),
            1 => (-6i64..=6, 1i64..=3).prop_map(|(a, b)| Recipe::Rat(a, b)),
            1 => Just(Recipe::Sin),
        ];
        leaf.prop_recursive(4, 40, 5, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..5).prop_map(Recipe::Add),
                prop::collection::vec(inner.clone(), 0..5).prop_map(Recipe::Mul),
                (inner, -1i64..=5).prop_map(|(b, n)| Recipe::Pow(Box::new(b), n)),
            ]
        })
    }

    fn build(r: &Recipe, p: &ExprPool, vs: &[ExprId], w: ExprId) -> ExprId {
        match r {
            Recipe::Var(i) => vs[*i],
            Recipe::Free => w,
            Recipe::Int(n) => p.integer(*n),
            Recipe::Big(a, s) => p.integer(rug::Integer::from(*a) << *s),
            Recipe::Rat(a, b) => p.rational(*a, *b),
            Recipe::Sin => p.func("sin", vec![vs[0]]),
            Recipe::Add(v) => p.add(v.iter().map(|c| build(c, p, vs, w)).collect()),
            Recipe::Mul(v) => p.mul(v.iter().map(|c| build(c, p, vs, w)).collect()),
            Recipe::Pow(b, n) => p.pow(build(b, p, vs, w), p.integer(*n)),
        }
    }

    #[test]
    fn keys_longer_than_vars_are_kept() {
        // A hand-built map may index past `vars`; FLINT must not drop it.
        let a = TermMap::from([(vec![0, 0, 0, 2], rug::Integer::from(3))]);
        let b: TermMap = (0..70u32)
            .map(|i| (vec![i], rug::Integer::from(i + 1)))
            .collect();
        assert_eq!(
            termmap_mul_flint(&a, &b).unwrap(),
            termmap_mul_schoolbook(&a, &b)
        );
    }

    #[test]
    fn exponent_overflow_defers_to_schoolbook() {
        let top = u32::MAX - 1;
        let a = TermMap::from([(vec![top], rug::Integer::from(1))]);
        let one = TermMap::from([(vec![1], rug::Integer::from(1))]);
        // x^(MAX-1) · x fits exactly.
        assert_eq!(
            termmap_mul_flint(&a, &one).unwrap(),
            TermMap::from([(vec![u32::MAX], rug::Integer::from(1))])
        );
        // x^(MAX-1) · x^2 does not; FLINT declines.
        let two = TermMap::from([(vec![2], rug::Integer::from(1))]);
        assert!(termmap_mul_flint(&a, &two).is_none());
        assert!(termmap_pow_flint(&a, 2).is_none());
    }

    #[test]
    fn zero_and_trivial_powers() {
        let zero = TermMap::new();
        let p: TermMap = [(vec![1], 2), (vec![0, 1], -3), (vec![], 5)]
            .into_iter()
            .map(|(k, c)| (k, rug::Integer::from(c)))
            .collect();
        for n in 0..4 {
            assert_eq!(termmap_pow(&zero, n), termmap_pow_schoolbook(&zero, n));
            assert_eq!(termmap_pow(&p, n), termmap_pow_schoolbook(&p, n));
        }
        assert!(termmap_mul(&zero, &p).is_empty());
        assert_eq!(termmap_mul_flint(&zero, &p), Some(TermMap::new()));
    }

    /// Shared subexpressions are converted once, not once per path.
    #[test]
    fn from_symbolic_is_linear_on_shared_dag() {
        let (p, x, y) = pool_xy();
        let (mut t0, mut t1) = (p.integer(1_i32), p.add(vec![x, y]));
        for _ in 0..40 {
            let t2 = p.add(vec![p.mul(vec![x, t1]), p.mul(vec![y, t0])]);
            (t0, t1) = (t1, t2);
        }
        let start = std::time::Instant::now();
        let poly = MultiPoly::from_symbolic(t1, vec![x, y], &p).unwrap();
        assert!(start.elapsed().as_secs() < 5, "took {:?}", start.elapsed());
        assert_eq!(poly.total_degree(), 41);
    }

    /// Timing and crossover measurement, run by hand:
    /// `cargo test --release -p alkahest-cas --lib multipoly::tests::timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn timing_mul_pow_from_symbolic() {
        fn best(reps: usize, mut f: impl FnMut()) -> f64 {
            (0..reps)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::MAX, f64::min)
        }
        let p = ExprPool::new();
        let vs: Vec<ExprId> = ["x", "y", "z"]
            .iter()
            .map(|n| p.symbol(*n, Domain::Real))
            .collect();
        let one = p.integer(1_i32);
        let s = p.add(vec![vs[0], vs[1], vs[2], one]);
        // Crossover: dense (x+y+z+1)^k squared, k small.
        for k in 1..=4u32 {
            let t = expr_to_multivariate_coeffs(p.pow(s, p.integer(k)), &vs, &p).unwrap();
            let fl = best(200, || drop(termmap_mul_flint(&t, &t)));
            let sb = best(200, || drop(termmap_mul_schoolbook(&t, &t)));
            println!(
                "mul {:>4} pairs: schoolbook {sb:>9.1} us, flint {fl:>9.1} us",
                t.len() * t.len()
            );
            let n = 3;
            let fl = best(200, || drop(termmap_pow_flint(&t, n)));
            let sb = best(200, || drop(termmap_pow_schoolbook(&t, n)));
            println!(
                "pow {:>3} terms ^{n}: schoolbook {sb:>9.1} us, flint {fl:>9.1} us",
                t.len()
            );
        }
        for (a_len, b_len) in [
            (2usize, 2usize),
            (4, 4),
            (8, 8),
            (8, 16),
            (16, 16),
            (64, 64),
            (300, 300),
        ] {
            let a: TermMap = (0..a_len as u32)
                .map(|i| (vec![i, 1], rug::Integer::from(i + 1)))
                .collect();
            let b: TermMap = (0..b_len as u32)
                .map(|i| (vec![1, i], rug::Integer::from(2 * i + 1)))
                .collect();
            let fl = best(500, || drop(termmap_mul_flint(&a, &b)));
            let sb = best(500, || drop(termmap_mul_schoolbook(&a, &b)));
            println!(
                "sparse mul {:>4} pairs: schoolbook {sb:>9.2} us, flint {fl:>9.2} us",
                a_len * b_len
            );
        }
        for (terms, n) in [(2usize, 2u32), (2, 3), (2, 8), (3, 2), (3, 4)] {
            let base: TermMap = (0..terms)
                .map(|i| {
                    let mut k = vec![0u32; i + 1];
                    k[i] = 1;
                    (k, rug::Integer::from(i as i64 + 1))
                })
                .collect();
            let fl = best(500, || drop(termmap_pow_flint(&base, n)));
            let sb = best(500, || drop(termmap_pow_schoolbook(&base, n)));
            println!("pow {terms} terms ^{n}: schoolbook {sb:>9.2} us, flint {fl:>9.2} us");
        }
        for k in [6u32, 10] {
            let t = expr_to_multivariate_coeffs(p.pow(s, p.integer(k)), &vs, &p).unwrap();
            let fl = best(5, || drop(termmap_mul(&t, &t)));
            let sb = best(3, || drop(termmap_mul_schoolbook(&t, &t)));
            println!(
                "p*p, p=(x+y+z+1)^{k} ({} terms): schoolbook {:.3} ms, flint {:.3} ms",
                t.len(),
                sb / 1e3,
                fl / 1e3
            );
        }
        let e = p.pow(s, p.integer(20_i32));
        let fl = best(5, || drop(expr_to_multivariate_coeffs(e, &vs, &p).unwrap()));
        let sb = best(3, || {
            let mut st = MultiBuild::new();
            st.fast = false;
            drop(build_multi(e, &vs, &p, &mut st).unwrap())
        });
        println!(
            "from_symbolic((x+y+z+1)^20): schoolbook {:.3} ms, flint {:.3} ms",
            sb / 1e3,
            fl / 1e3
        );
    }
}
