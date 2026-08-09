//! Recognising *proper hypergeometric terms* and computing their exact shift ratios.
//!
//! A proper hypergeometric term in `(n, k)` is
//!
//! ```text
//! F(n, k) = R(n, k) · z^k · w^n · ∏_j Γ(a_j·n + b_j·k + c_j)^(e_j)
//! ```
//!
//! with `R ∈ Q(n, k)`, `z, w ∈ Q \ {0}`, `a_j, b_j ∈ Z`, `c_j ∈ Q`, `e_j ∈ Z`.
//! This is exactly the class for which both shift quotients
//! `F(n, k+1)/F(n, k)` and `F(n+i, k)/F(n, k)` are *rational functions* that can
//! be written down exactly — which is what Zeilberger's algorithm consumes.
//!
//! The parser is deliberately strict: anything it cannot place in this class is
//! refused rather than approximated.

use super::qfield::{rn_add, rn_int, rn_is_zero, rn_mul, rn_one, rn_rat, rn_var, PolyK, RatK, Rn};
use super::HolonomicError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Rational;

/// One `Γ(a·n + b·k + c)^e` factor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GammaFactor {
    pub a: i64,
    pub b: i64,
    pub c: Rational,
    pub e: i32,
}

/// A parsed proper hypergeometric term.
#[derive(Clone, Debug)]
pub struct ProperTerm {
    /// Rational-function prefactor `R(n, k)`.
    pub rat: RatK,
    /// Base of the `z^k` factor.
    pub z: Rational,
    /// Base of the `w^n` factor.
    pub w: Rational,
    pub gammas: Vec<GammaFactor>,
}

/// Largest integer exponent accepted on a sub-term (guards against blow-up).
const MAX_POW: i32 = 32;

impl ProperTerm {
    fn one() -> Self {
        ProperTerm {
            rat: RatK::one(),
            z: Rational::from(1),
            w: Rational::from(1),
            gammas: Vec::new(),
        }
    }

    fn from_ratk(r: RatK) -> Self {
        ProperTerm {
            rat: r,
            z: Rational::from(1),
            w: Rational::from(1),
            gammas: Vec::new(),
        }
    }

    fn mul(&self, other: &ProperTerm) -> ProperTerm {
        let mut gammas = self.gammas.clone();
        gammas.extend(other.gammas.iter().cloned());
        ProperTerm {
            rat: self.rat.mul(&other.rat),
            z: self.z.clone() * other.z.clone(),
            w: self.w.clone() * other.w.clone(),
            gammas,
        }
    }

    fn pow(&self, e: i32) -> Option<ProperTerm> {
        if e.unsigned_abs() > MAX_POW as u32 {
            return None;
        }
        let gammas = self
            .gammas
            .iter()
            .map(|g| {
                Some(GammaFactor {
                    a: g.a,
                    b: g.b,
                    c: g.c.clone(),
                    e: g.e.checked_mul(e)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(ProperTerm {
            rat: self.rat.pow_i32(e)?,
            z: rat_pow(&self.z, e)?,
            w: rat_pow(&self.w, e)?,
            gammas,
        })
    }

    /// `F(n, k+1) / F(n, k)` as an exact element of `Q(n, k)`.
    pub fn ratio_k(&self) -> Result<RatK, HolonomicError> {
        let shifted = self.rat.shift_k(1);
        let mut acc = shifted.div(&self.rat).ok_or_else(|| {
            HolonomicError::NotProperHypergeometric("term vanishes identically".into())
        })?;
        acc = acc.mul(&RatK::from_rn(rn_rat(self.z.clone())));
        for g in &self.gammas {
            let arg = gamma_arg_poly(g);
            let step = gamma_shift_ratio(&arg, g.b)?;
            acc = acc.mul(&step.pow_i32(g.e).ok_or_else(|| {
                HolonomicError::NotProperHypergeometric("gamma factor is identically zero".into())
            })?);
        }
        Ok(acc)
    }

    /// `F(n+i, k) / F(n, k)` as an exact element of `Q(n, k)`.
    pub fn ratio_n(&self, i: i64) -> Result<RatK, HolonomicError> {
        if i == 0 {
            return Ok(RatK::one());
        }
        let shifted = self.rat.shift_n(i);
        let mut acc = shifted.div(&self.rat).ok_or_else(|| {
            HolonomicError::NotProperHypergeometric("term vanishes identically".into())
        })?;
        acc = acc.mul(&RatK::from_rn(rn_rat(rat_pow_i64(&self.w, i)?)));
        for g in &self.gammas {
            let arg = gamma_arg_poly(g);
            let shift = g
                .a
                .checked_mul(i)
                .ok_or_else(|| HolonomicError::SearchExhausted("gamma shift overflow".into()))?;
            let step = gamma_shift_ratio(&arg, shift)?;
            acc = acc.mul(&step.pow_i32(g.e).ok_or_else(|| {
                HolonomicError::NotProperHypergeometric("gamma factor is identically zero".into())
            })?);
        }
        Ok(acc)
    }

    /// Parse an expression into the proper hypergeometric class.
    pub fn parse(
        expr: ExprId,
        n: ExprId,
        k: ExprId,
        pool: &ExprPool,
    ) -> Result<ProperTerm, HolonomicError> {
        parse_rec(expr, n, k, pool, 0)
    }
}

fn gamma_arg_poly(g: &GammaFactor) -> PolyK {
    // a·n + b·k + c  as a polynomial in k over Q(n)
    let const_term = rn_add(&rn_mul(&rn_int(g.a), &rn_var()), &rn_rat(g.c.clone()));
    PolyK::from_coeffs(vec![const_term, rn_int(g.b)])
}

/// `Γ(x + s) / Γ(x)` for integer `s`, as an exact rational function.
fn gamma_shift_ratio(x: &PolyK, s: i64) -> Result<RatK, HolonomicError> {
    if s == 0 {
        return Ok(RatK::one());
    }
    if s.unsigned_abs() > 512 {
        return Err(HolonomicError::SearchExhausted(format!(
            "gamma argument shift {s} exceeds the supported limit of 512"
        )));
    }
    let mut prod = PolyK::one();
    if s > 0 {
        for t in 0..s {
            prod = prod.mul(&x.add(&PolyK::constant(rn_int(t))));
        }
        Ok(RatK::from_poly(prod))
    } else {
        for t in 1..=(-s) {
            prod = prod.mul(&x.add(&PolyK::constant(rn_int(-t))));
        }
        RatK::from_poly(prod)
            .inv()
            .ok_or_else(|| HolonomicError::NotProperHypergeometric("gamma pole".into()))
    }
}

fn rat_pow(q: &Rational, e: i32) -> Option<Rational> {
    rat_pow_i64(q, e as i64).ok()
}

fn rat_pow_i64(q: &Rational, e: i64) -> Result<Rational, HolonomicError> {
    if e == 0 {
        return Ok(Rational::from(1));
    }
    if e.unsigned_abs() > 1024 {
        return Err(HolonomicError::SearchExhausted(
            "exponential factor exponent exceeds the supported limit".into(),
        ));
    }
    if *q == 0 {
        if e < 0 {
            return Err(HolonomicError::NotProperHypergeometric(
                "zero base raised to a negative power".into(),
            ));
        }
        return Ok(Rational::from(0));
    }
    let base = if e < 0 { q.clone().recip() } else { q.clone() };
    let mut acc = Rational::from(1);
    for _ in 0..e.unsigned_abs() {
        acc *= base.clone();
    }
    Ok(acc)
}

const MAX_PARSE_DEPTH: usize = 64;

fn parse_rec(
    expr: ExprId,
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ProperTerm, HolonomicError> {
    if depth > MAX_PARSE_DEPTH {
        return Err(HolonomicError::NotProperHypergeometric(
            "expression nests deeper than the parser supports".into(),
        ));
    }
    // Fast path: a purely rational sub-expression.
    if let Some(r) = as_ratk(expr, n, k, pool, 0) {
        return Ok(ProperTerm::from_ratk(r));
    }
    match pool.get(expr) {
        ExprData::Mul(args) => {
            let mut acc = ProperTerm::one();
            for a in args {
                acc = acc.mul(&parse_rec(a, n, k, pool, depth + 1)?);
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => {
            if let Some(e) = as_i32(exp, pool) {
                let b = parse_rec(base, n, k, pool, depth + 1)?;
                return b.pow(e).ok_or_else(|| {
                    HolonomicError::NotProperHypergeometric(format!(
                        "exponent {e} is outside the supported range (|e| ≤ {MAX_POW})"
                    ))
                });
            }
            // c^(α·n + β·k + γ) with rational c
            let Some(c) = as_rational(base, pool) else {
                return Err(HolonomicError::NotProperHypergeometric(format!(
                    "power with symbolic exponent needs a rational base, got {}",
                    pool.display(base)
                )));
            };
            if c == 0 {
                return Err(HolonomicError::NotProperHypergeometric(
                    "0 raised to a symbolic power".into(),
                ));
            }
            let (alpha, beta, gamma) = affine_parts(exp, n, k, pool).ok_or_else(|| {
                HolonomicError::NotProperHypergeometric(format!(
                    "exponent {} is not integer-affine in the two indices",
                    pool.display(exp)
                ))
            })?;
            if *gamma.clone().denom() != 1 {
                return Err(HolonomicError::NotProperHypergeometric(
                    "constant part of an exponential exponent must be an integer".into(),
                ));
            }
            let gi: i64 = gamma
                .numer()
                .to_i64()
                .ok_or_else(|| HolonomicError::SearchExhausted("exponent too large".into()))?;
            Ok(ProperTerm {
                rat: RatK::from_rn(rn_rat(rat_pow_i64(&c, gi)?)),
                z: rat_pow_i64(&c, beta)?,
                w: rat_pow_i64(&c, alpha)?,
                gammas: Vec::new(),
            })
        }
        ExprData::Func { name, args } => parse_func(&name, &args, n, k, pool),
        other => Err(HolonomicError::NotProperHypergeometric(format!(
            "unsupported node {other:?} in {}",
            pool.display(expr)
        ))),
    }
}

fn parse_func(
    name: &str,
    args: &[ExprId],
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Result<ProperTerm, HolonomicError> {
    let gamma_of = |arg: ExprId, e: i32| -> Result<GammaFactor, HolonomicError> {
        let (a, b, c) = affine_parts(arg, n, k, pool).ok_or_else(|| {
            HolonomicError::NotProperHypergeometric(format!(
                "gamma argument {} is not integer-affine in the two indices",
                pool.display(arg)
            ))
        })?;
        Ok(GammaFactor { a, b, c, e })
    };
    let one_plus = |arg: ExprId| -> ExprId { pool.add(vec![arg, pool.integer(1_i32)]) };
    match (name, args.len()) {
        ("gamma", 1) => Ok(ProperTerm {
            rat: RatK::one(),
            z: Rational::from(1),
            w: Rational::from(1),
            gammas: vec![gamma_of(args[0], 1)?],
        }),
        ("factorial", 1) => Ok(ProperTerm {
            rat: RatK::one(),
            z: Rational::from(1),
            w: Rational::from(1),
            gammas: vec![gamma_of(one_plus(args[0]), 1)?],
        }),
        ("binomial", 2) => {
            let top = one_plus(args[0]);
            let bot = one_plus(args[1]);
            let rest = pool.add(vec![
                args[0],
                pool.mul(vec![args[1], pool.integer(-1_i32)]),
                pool.integer(1_i32),
            ]);
            Ok(ProperTerm {
                rat: RatK::one(),
                z: Rational::from(1),
                w: Rational::from(1),
                gammas: vec![gamma_of(top, 1)?, gamma_of(bot, -1)?, gamma_of(rest, -1)?],
            })
        }
        ("pochhammer", 2) => {
            // (a)_m = Γ(a + m)/Γ(a)
            let sum = pool.add(vec![args[0], args[1]]);
            Ok(ProperTerm {
                rat: RatK::one(),
                z: Rational::from(1),
                w: Rational::from(1),
                gammas: vec![gamma_of(sum, 1)?, gamma_of(args[0], -1)?],
            })
        }
        _ => Err(HolonomicError::NotProperHypergeometric(format!(
            "function `{name}/{}` is not part of the proper hypergeometric class \
             (supported: gamma, factorial, binomial, pochhammer)",
            args.len()
        ))),
    }
}

/// Evaluate an expression inside the field `Q(n)(k)`, or `None` if it leaves it.
pub fn as_ratk(expr: ExprId, n: ExprId, k: ExprId, pool: &ExprPool, depth: usize) -> Option<RatK> {
    if depth > MAX_PARSE_DEPTH {
        return None;
    }
    if expr == k {
        return Some(RatK::k());
    }
    if expr == n {
        return Some(RatK::from_rn(rn_var()));
    }
    match pool.get(expr) {
        ExprData::Integer(i) => Some(RatK::from_rn(rn_rat(Rational::from(i.0.clone())))),
        ExprData::Rational(r) => Some(RatK::from_rn(rn_rat(r.0.clone()))),
        ExprData::Add(args) => {
            let mut acc = RatK::zero();
            for a in args {
                acc = acc.add(&as_ratk(a, n, k, pool, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = RatK::one();
            for a in args {
                acc = acc.mul(&as_ratk(a, n, k, pool, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let e = as_i32(exp, pool)?;
            if e.unsigned_abs() > MAX_POW as u32 {
                return None;
            }
            as_ratk(base, n, k, pool, depth + 1)?.pow_i32(e)
        }
        _ => None,
    }
}

fn as_rational(expr: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(expr) {
        ExprData::Integer(i) => Some(Rational::from(i.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

fn as_i32(expr: ExprId, pool: &ExprPool) -> Option<i32> {
    match pool.get(expr) {
        ExprData::Integer(i) => i.0.to_i32(),
        _ => None,
    }
}

/// Decompose an expression as `a·n + b·k + c` with `a, b ∈ Z` and `c ∈ Q`.
pub fn affine_parts(
    expr: ExprId,
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Option<(i64, i64, Rational)> {
    let r = as_ratk(expr, n, k, pool, 0)?;
    if r.den.degree() != 0 {
        return None;
    }
    let den_c = r.den.coeff(0);
    if rn_is_zero(&den_c) {
        return None;
    }
    let inv = super::qfield::rn_inv(&den_c)?;
    let num = r.num.scale(&inv);
    if num.degree() > 1 {
        return None;
    }
    let b_rn = num.coeff(1);
    let b = rn_as_rational(&b_rn)?;
    if *b.clone().denom() != 1 {
        return None;
    }
    let b_i = b.numer().to_i64()?;
    let (a, c) = rn_as_linear(&num.coeff(0))?;
    if *a.clone().denom() != 1 {
        return None;
    }
    let a_i = a.numer().to_i64()?;
    Some((a_i, b_i, c))
}

fn rn_as_rational(r: &Rn) -> Option<Rational> {
    if r.num.degree() > 0 || r.den.degree() > 0 {
        return None;
    }
    let num = r
        .num
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(|| Rational::from(0));
    let den = r
        .den
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(|| Rational::from(0));
    if den == 0 {
        return None;
    }
    Some(num / den)
}

/// Write `r ∈ Q(n)` as `a·n + c`, or `None` if it is not linear in `n`.
fn rn_as_linear(r: &Rn) -> Option<(Rational, Rational)> {
    if r.den.degree() > 0 {
        return None;
    }
    let den = r
        .den
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(|| Rational::from(0));
    if den == 0 {
        return None;
    }
    if r.num.degree() > 1 {
        return None;
    }
    let c0 = r
        .num
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(|| Rational::from(0));
    let c1 = r
        .num
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(|| Rational::from(0));
    Some((c1 / den.clone(), c0 / den))
}

// ---------------------------------------------------------------------------
// Algebra → expression bridge
// ---------------------------------------------------------------------------

fn rational_to_expr(pool: &ExprPool, q: &Rational) -> ExprId {
    let (num, den) = (q.numer().clone(), q.denom().clone());
    if den == 1 {
        pool.integer(num)
    } else {
        pool.rational(num, den)
    }
}

/// A `Q[n]` polynomial as an expression in `n`.
pub fn ratuni_to_expr(
    pool: &ExprPool,
    n: ExprId,
    p: &crate::matrix::normal_form::RatUniPoly,
) -> ExprId {
    let mut terms = Vec::new();
    for (deg, c) in p.coeffs.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        let ce = rational_to_expr(pool, c);
        let t = match deg {
            0 => ce,
            1 => pool.mul(vec![ce, n]),
            d => pool.mul(vec![ce, pool.pow(n, pool.integer(d as i64))]),
        };
        terms.push(t);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// An element of `Q(n)` as an expression in `n`.
pub fn rn_to_expr(pool: &ExprPool, n: ExprId, r: &Rn) -> ExprId {
    let num = ratuni_to_expr(pool, n, &r.num);
    if r.den.degree() == 0 && r.den.coeffs.first().map(|c| *c == 1).unwrap_or(false) {
        return num;
    }
    let den = ratuni_to_expr(pool, n, &r.den);
    pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
}

/// A `Q(n)[k]` polynomial as an expression in `n` and `k`.
pub fn polyk_to_expr(pool: &ExprPool, n: ExprId, k: ExprId, p: &PolyK) -> ExprId {
    let mut terms = Vec::new();
    for (deg, c) in p.coeffs.iter().enumerate() {
        if rn_is_zero(c) {
            continue;
        }
        let ce = rn_to_expr(pool, n, c);
        let t = match deg {
            0 => ce,
            1 => pool.mul(vec![ce, k]),
            d => pool.mul(vec![ce, pool.pow(k, pool.integer(d as i64))]),
        };
        terms.push(t);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// An element of `Q(n, k)` as an expression.
pub fn ratk_to_expr(pool: &ExprPool, n: ExprId, k: ExprId, r: &RatK) -> ExprId {
    let num = polyk_to_expr(pool, n, k, &r.num);
    if r.den.degree() == 0 {
        let c = r.den.coeff(0);
        if rn_is_zero(&c) {
            return num;
        }
        if r.den.eq_poly(&PolyK::one()) {
            return num;
        }
    }
    let den = polyk_to_expr(pool, n, k, &r.den);
    pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
}

/// Convenience: `1` as a `Q(n)` element (used by callers building terms).
pub fn rn_unit() -> Rn {
    rn_one()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn nk(pool: &ExprPool) -> (ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("k", Domain::Real),
        )
    }

    fn binom(pool: &ExprPool, n: ExprId, k: ExprId) -> ExprId {
        let g1 = pool.func("gamma", vec![pool.add(vec![n, pool.integer(1_i32)])]);
        let g2 = pool.func("gamma", vec![pool.add(vec![k, pool.integer(1_i32)])]);
        let g3 = pool.func(
            "gamma",
            vec![pool.add(vec![
                n,
                pool.mul(vec![k, pool.integer(-1_i32)]),
                pool.integer(1_i32),
            ])],
        );
        pool.mul(vec![
            g1,
            pool.pow(g2, pool.integer(-1_i32)),
            pool.pow(g3, pool.integer(-1_i32)),
        ])
    }

    #[test]
    fn binomial_ratios_are_exact() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let t = ProperTerm::parse(binom(&pool, n, k), n, k, &pool).expect("parse C(n,k)");
        // F(n,k+1)/F(n,k) = (n-k)/(k+1)
        let rk = t.ratio_k().expect("k ratio");
        let want = RatK {
            num: PolyK::from_coeffs(vec![rn_var(), rn_int(-1)]),
            den: PolyK::from_coeffs(vec![rn_one(), rn_one()]),
        }
        .normalize();
        assert!(rk.eq_ratk(&want), "got {rk:?}");
        // F(n+1,k)/F(n,k) = (n+1)/(n+1-k)
        let rn1 = t.ratio_n(1).expect("n ratio");
        let want2 = RatK {
            num: PolyK::constant(rn_add(&rn_var(), &rn_one())),
            den: PolyK::from_coeffs(vec![rn_add(&rn_var(), &rn_one()), rn_int(-1)]),
        }
        .normalize();
        assert!(rn1.eq_ratk(&want2), "got {rn1:?}");
    }

    #[test]
    fn binomial_func_head_parses_like_gammas() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let via_func = ProperTerm::parse(pool.func("binomial", vec![n, k]), n, k, &pool)
            .expect("parse binomial()");
        let via_gamma = ProperTerm::parse(binom(&pool, n, k), n, k, &pool).expect("parse gammas");
        assert!(via_func
            .ratio_k()
            .unwrap()
            .eq_ratk(&via_gamma.ratio_k().unwrap()));
        assert!(via_func
            .ratio_n(1)
            .unwrap()
            .eq_ratk(&via_gamma.ratio_n(1).unwrap()));
    }

    #[test]
    fn geometric_factor_is_recognised() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        // (-1)^k · 2^n
        let e = pool.mul(vec![
            pool.pow(pool.integer(-1_i32), k),
            pool.pow(pool.integer(2_i32), n),
        ]);
        let t = ProperTerm::parse(e, n, k, &pool).expect("parse");
        assert_eq!(t.z, Rational::from(-1));
        assert_eq!(t.w, Rational::from(2));
        assert!(t.ratio_k().unwrap().eq_ratk(&RatK::from_rn(rn_int(-1))));
        assert!(t.ratio_n(1).unwrap().eq_ratk(&RatK::from_rn(rn_int(2))));
    }

    #[test]
    fn non_hypergeometric_input_is_refused() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let e = pool.func("sin", vec![k]);
        assert!(ProperTerm::parse(e, n, k, &pool).is_err());
        // symbolic base with symbolic exponent
        let x = pool.symbol("x", Domain::Real);
        let e2 = pool.pow(x, k);
        assert!(ProperTerm::parse(e2, n, k, &pool).is_err());
    }
}
