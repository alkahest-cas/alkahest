//! Symbolic discrete products (∏) over ℚ(k) with ℤ-linear factorisation (V2-22).
//!
//! \(\prod_{k=m}^{n} q(k)\) for \(q\) a rational whose numerator/denominator split into
//! linear factors over ℤ telescopes via \(\sum \Delta\logΓ(k+r) = \logΓ(n+r+1)-\logΓ(m+r)\)
//! and integer leading-coefficient powers \(a^{e(n-m+1)}\).

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::flint::{integer::FlintInteger, FlintPoly};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;
use crate::poly::factor::UniPolyFactorization;
use crate::poly::UniPoly;
use crate::simplify::engine::simplify;
use crate::sum::ratfunc::RatFunc;
use rug::{Integer, Rational};
use std::fmt;

fn simp(pool: &ExprPool, e: ExprId) -> ExprId {
    simplify(e, pool).value
}

/// Errors raised by discrete product evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductError {
    /// Term is not a supported rational function of the index.
    NotRationalTerm(String),
    /// ℤ-factorisation failed.
    Factorization,
    /// An irreducible ℤ-factor has degree > 1.
    NonLinearFactor,
    /// Bound substitution failed (mirrors summation).
    BoundSubstitution(String),
}

impl fmt::Display for ProductError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProductError::NotRationalTerm(s) => write!(f, "product: unsupported term shape: {s}"),
            ProductError::Factorization => write!(f, "product: polynomial factorisation failed"),
            ProductError::NonLinearFactor => {
                write!(
                    f,
                    "product: term has a non-linear irreducible factor over ℤ"
                )
            }
            ProductError::BoundSubstitution(s) => write!(f, "product: bound substitution: {s}"),
        }
    }
}

impl std::error::Error for ProductError {}

impl crate::errors::AlkahestError for ProductError {
    fn code(&self) -> &'static str {
        match self {
            ProductError::NotRationalTerm(_) => "E-PROD-001",
            ProductError::Factorization => "E-PROD-002",
            ProductError::NonLinearFactor => "E-PROD-003",
            ProductError::BoundSubstitution(_) => "E-PROD-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some("supported: ∏ q(k) for q ∈ ℚ(k) factoring into ℤ-linear terms; no irreducible quadratics in k")
    }
}

fn rational_to_expr(pool: &ExprPool, r: &Rational) -> ExprId {
    let n = r.numer().clone();
    let d = r.denom().clone();
    if d == Integer::from(1) {
        pool.integer(n)
    } else {
        pool.rational(n, d)
    }
}

fn ratuni_poly_to_univ(p: &RatUniPoly, var: ExprId) -> Result<UniPoly, ProductError> {
    if p.is_zero() {
        return Ok(UniPoly::zero(var));
    }
    let mut lcm = Integer::from(1u32);
    for c in &p.coeffs {
        if *c != Rational::from(0) {
            lcm = lcm.lcm(&Integer::from(c.denom().clone()));
        }
    }
    let scale = Rational::from(&lcm);
    let mut max_i = p.coeffs.len().saturating_sub(1);
    let mut rug_coeffs = vec![Integer::from(0); max_i + 1];
    for (i, c) in p.coeffs.iter().enumerate() {
        if *c == Rational::from(0) {
            continue;
        }
        let scaled = c.clone() * scale.clone();
        if *scaled.denom() != Integer::from(1) {
            return Err(ProductError::NotRationalTerm(
                "could not clear denominators".into(),
            ));
        }
        rug_coeffs[i] = scaled.numer().clone();
        max_i = max_i.max(i);
    }
    rug_coeffs.truncate(max_i + 1);
    let coeffs: Vec<FlintInteger> = rug_coeffs.iter().map(FlintInteger::from_rug).collect();
    let mut fp = FlintPoly::new();
    for (i, ci) in coeffs.iter().enumerate() {
        if !ci.to_rug().is_zero() {
            fp.set_coeff_flint(i, ci);
        }
    }
    Ok(UniPoly { var, coeffs: fp })
}

fn expr_to_ratfunc(term: ExprId, k: ExprId, pool: &ExprPool) -> Result<RatFunc, ProductError> {
    let term = simp(pool, term);
    if term == k {
        return Ok(RatFunc {
            num: RatUniPoly::x(),
            den: RatUniPoly::one(),
        }
        .normalize());
    }
    match pool.get(term).clone() {
        ExprData::Integer(n) => Ok(RatFunc::scalar(Rational::from(&n.0))),
        ExprData::Rational(br) => Ok(RatFunc::scalar(br.0.clone())),
        ExprData::Symbol { name, .. } => {
            if term == k {
                Ok(RatFunc {
                    num: RatUniPoly::x(),
                    den: RatUniPoly::one(),
                }
                .normalize())
            } else {
                Err(ProductError::NotRationalTerm(format!(
                    "free symbol `{name}` — term must be unary rational in k",
                )))
            }
        }
        ExprData::Add(_) => {
            let p = UniPoly::from_symbolic_clear_denoms(term, k, pool).map_err(|e| {
                ProductError::NotRationalTerm(format!("polynomial expected in k: {e}"))
            })?;
            let coeffs: Vec<Rational> = p.coefficients().into_iter().map(Rational::from).collect();
            Ok(RatFunc::from_poly(RatUniPoly { coeffs }.trim()).normalize())
        }
        ExprData::Pow { base, exp } => {
            let e_i = match pool.get(exp) {
                ExprData::Integer(n) => n
                    .0
                    .to_i32()
                    .ok_or_else(|| ProductError::NotRationalTerm("exponent out of range".into()))?,
                _ => {
                    return Err(ProductError::NotRationalTerm(
                        "non-constant exponent".into(),
                    ))
                }
            };
            let base_rf = expr_to_ratfunc(base, k, pool)?;
            if e_i >= 0 {
                let ee = u32::try_from(e_i)
                    .map_err(|_| ProductError::NotRationalTerm("exponent overflow".into()))?;
                let mut acc = RatFunc::one();
                for _ in 0..ee {
                    acc = acc.mul_ratfunc(&base_rf);
                }
                Ok(acc.normalize())
            } else {
                let inv = base_rf
                    .inv()
                    .ok_or_else(|| ProductError::NotRationalTerm("invert zero".into()))?;
                let ee =
                    u32::try_from(-e_i).map_err(|_| ProductError::NotRationalTerm("exp".into()))?;
                let mut acc = RatFunc::one();
                for _ in 0..ee {
                    acc = acc.mul_ratfunc(&inv);
                }
                Ok(acc.normalize())
            }
        }
        ExprData::Mul(args) => {
            let mut acc = RatFunc::one();
            for &a in &args {
                acc = acc.mul_ratfunc(&expr_to_ratfunc(a, k, pool)?);
            }
            Ok(acc.normalize())
        }
        _ => Err(ProductError::NotRationalTerm(
            "expression is not a rational function of k with integer poly factors".into(),
        )),
    }
}

fn factor_univ(p: &UniPoly) -> Result<UniPolyFactorization, ProductError> {
    p.factor_z().map_err(|_| ProductError::Factorization)
}

/// ∏ fac over one side of a rational (numerator or denominator).
fn definite_side_from_factorization(
    pool: &ExprPool,
    fac: &UniPolyFactorization,
    lo: ExprId,
    hi: ExprId,
    delta_n: ExprId,
) -> Result<ExprId, ProductError> {
    let mut parts: Vec<ExprId> = Vec::new();
    let u = &fac.unit;
    if *u == Integer::from(-1) {
        parts.push(pool.pow(pool.integer(-1_i32), delta_n.clone()));
    } else if *u != Integer::from(1) {
        parts.push(pool.pow(pool.integer(u.clone()), delta_n.clone()));
    }

    for (fact, ee) in &fac.factors {
        let expo = *ee as i64;
        let d = fact.degree().max(0) as usize;
        match d {
            0 => {
                let cz = match fact.coefficients().first() {
                    Some(c) => c.clone(),
                    None => Integer::from(1),
                };
                if cz == Integer::from(1) {
                    continue;
                }
                if cz == Integer::from(-1) {
                    if expo.rem_euclid(2) != 0 {
                        parts.push(pool.pow(pool.integer(-1_i32), delta_n.clone()));
                    }
                    continue;
                }
                let exp_e = pool.integer(expo);
                parts.push(pool.pow(
                    pool.integer(cz.clone()),
                    simp(pool, pool.mul(vec![delta_n.clone(), exp_e])),
                ));
            }
            1 => {
                let coeffs = fact.coefficients();
                let aa = coeffs.get(1).cloned().unwrap_or_else(|| Integer::from(0));
                let bb = coeffs.get(0).cloned().unwrap_or_else(|| Integer::from(0));
                if aa == Integer::from(0) {
                    return Err(ProductError::NotRationalTerm("degenerate linear".into()));
                }
                let c_rat = Rational::from((bb, aa.clone()));
                let one = Rational::from(1);
                let hi_shift = rational_to_expr(pool, &(one.clone() + c_rat.clone()));
                let lo_shift = rational_to_expr(pool, &c_rat);
                let lead_exp = simp(pool, pool.mul(vec![delta_n.clone(), pool.integer(expo)]));
                let gh = pool.func(
                    "gamma",
                    vec![simp(pool, pool.add(vec![hi.clone(), hi_shift]))],
                );
                let gl = pool.func(
                    "gamma",
                    vec![simp(pool, pool.add(vec![lo.clone(), lo_shift]))],
                );
                let ratio = simp(pool, pool.mul(vec![gh, pool.pow(gl, pool.integer(-1_i32))]));
                parts.push(pool.pow(pool.integer(aa.clone()), lead_exp));
                if expo != 0 {
                    parts.push(pool.pow(ratio, pool.integer(expo)));
                }
            }
            _ => return Err(ProductError::NonLinearFactor),
        }
    }

    match parts.len() {
        0 => Ok(pool.integer(1_i32)),
        1 => Ok(simp(pool, parts[0])),
        _ => Ok(simp(pool, pool.mul(parts))),
    }
}

/// Indefinite multiplicative antiderivative for one polynomial side.
fn indefinite_side_from_factorization(
    pool: &ExprPool,
    fac: &UniPolyFactorization,
    k: ExprId,
) -> Result<ExprId, ProductError> {
    let mut parts: Vec<ExprId> = Vec::new();
    let u = &fac.unit;
    if *u == Integer::from(-1) {
        parts.push(pool.pow(pool.integer(-1_i32), k.clone()));
    } else if *u != Integer::from(1) {
        parts.push(pool.pow(pool.integer(u.clone()), k.clone()));
    }

    for (fact, ee) in &fac.factors {
        let expo = *ee as i64;
        let d = fact.degree().max(0) as usize;
        match d {
            0 => {
                let cz = match fact.coefficients().first() {
                    Some(c) => c.clone(),
                    None => Integer::from(1),
                };
                if cz == Integer::from(1) {
                    continue;
                }
                if cz == Integer::from(-1) {
                    if expo.rem_euclid(2) != 0 {
                        parts.push(pool.pow(pool.integer(-1_i32), k.clone()));
                    }
                    continue;
                }
                let exp_e = pool.integer(expo);
                parts.push(pool.pow(
                    pool.integer(cz.clone()),
                    simp(pool, pool.mul(vec![k.clone(), exp_e])),
                ));
            }
            1 => {
                let coeffs = fact.coefficients();
                let aa = coeffs.get(1).cloned().unwrap_or_else(|| Integer::from(0));
                let bb = coeffs.get(0).cloned().unwrap_or_else(|| Integer::from(0));
                if aa == Integer::from(0) {
                    return Err(ProductError::NotRationalTerm("degenerate linear".into()));
                }
                let c_rat = Rational::from((bb, aa.clone()));
                let lo_shift = rational_to_expr(pool, &c_rat);
                let gamma_k = pool.func(
                    "gamma",
                    vec![simp(pool, pool.add(vec![k.clone(), lo_shift]))],
                );
                let lead_exp_k = simp(pool, pool.mul(vec![k.clone(), pool.integer(expo)]));
                parts.push(pool.pow(pool.integer(aa), lead_exp_k));
                parts.push(pool.pow(gamma_k, pool.integer(expo)));
            }
            _ => return Err(ProductError::NonLinearFactor),
        }
    }

    match parts.len() {
        0 => Ok(pool.integer(1_i32)),
        1 => Ok(simp(pool, parts[0])),
        _ => Ok(simp(pool, pool.mul(parts))),
    }
}

/// ∏_{k=lo}^{hi} term(k); inclusive `[lo, hi]`.
pub fn product_definite(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProductError> {
    let rf = expr_to_ratfunc(term, k, pool)?;
    if rf.num.is_zero() {
        let z = simp(pool, pool.integer(0_i32));
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple("product_definite_zero", term, z));
        return Ok(DerivedExpr::with_log(z, log));
    }

    let univ_n = ratuni_poly_to_univ(&rf.num, k)?;
    let univ_d = ratuni_poly_to_univ(&rf.den, k)?;
    let fac_n = factor_univ(&univ_n)?;
    let fac_d = factor_univ(&univ_d)?;

    let one = pool.integer(1_i32);
    let delta_n = simp(
        pool,
        pool.add(vec![
            hi.clone(),
            pool.mul(vec![lo.clone(), pool.integer(-1)]),
            one,
        ]),
    );

    let top =
        definite_side_from_factorization(pool, &fac_n, lo.clone(), hi.clone(), delta_n.clone())?;
    let bot =
        definite_side_from_factorization(pool, &fac_d, lo.clone(), hi.clone(), delta_n.clone())?;
    let q = simp(
        pool,
        pool.mul(vec![top.clone(), pool.pow(bot, pool.integer(-1_i32))]),
    );

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("product_definite", term, q));
    Ok(DerivedExpr::with_log(q, log))
}

/// Witness `Z(k)` with \(Z(k+1)/Z(k)=term(k)\) (after canonical simplification).
pub fn product_indefinite(
    term: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProductError> {
    let rf = expr_to_ratfunc(term, k, pool)?;
    if rf.num.is_zero() {
        return Err(ProductError::NotRationalTerm(
            "indefinite product of zero unsupported".into(),
        ));
    }
    let fac_n = factor_univ(&ratuni_poly_to_univ(&rf.num, k)?)?;
    let fac_d = factor_univ(&ratuni_poly_to_univ(&rf.den, k)?)?;

    let top = indefinite_side_from_factorization(pool, &fac_n, k.clone())?;
    let bot = indefinite_side_from_factorization(pool, &fac_d, k.clone())?;

    let q = simp(
        pool,
        pool.mul(vec![top, pool.pow(bot, pool.integer(-1_i32))]),
    );

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("product_indefinite", term, q.clone()));
    Ok(DerivedExpr::with_log(q, log))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;
    use rug::Float;
    use std::collections::HashMap;

    fn gamma64(x: f64) -> f64 {
        Float::with_val(53, x).gamma().to_f64()
    }

    fn eval_g(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
        match pool.get(expr).clone() {
            ExprData::Func { name, args } if name == "gamma" && args.len() == 1 => {
                Some(gamma64(eval_g(args[0], env, pool)?))
            }
            ExprData::Add(args) => {
                let mut s = 0.0f64;
                for &a in &args {
                    s += eval_g(a, env, pool)?;
                }
                Some(s)
            }
            ExprData::Mul(args) => {
                let mut p = 1.0f64;
                for a in args {
                    p *= eval_g(a, env, pool)?;
                }
                Some(p)
            }
            ExprData::Pow { base, exp } => {
                Some(eval_g(base, env, pool)?.powf(eval_interp(exp, env, pool)?))
            }
            _ => eval_interp(expr, env, pool),
        }
    }

    #[test]
    fn product_linear_k_matches_factorial_gamma() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let n = pool.symbol("n", Domain::Real);
        let lo = pool.integer(1_i32);
        let p = product_definite(k, k, lo, n.clone(), &pool).expect("prod");
        let want = simp(
            &pool,
            pool.func(
                "gamma",
                vec![simp(&pool, pool.add(vec![n.clone(), pool.integer(1)]))],
            ),
        );
        for ni in 2..14 {
            let mut env = HashMap::new();
            env.insert(n, ni as f64);
            let pv = eval_g(p.value, &env, &pool).unwrap();
            let wv = eval_g(want, &env, &pool).unwrap();
            assert!(
                (pv - wv).abs() < 1e-6 * wv.abs().max(1.0),
                "n={ni}: pv={pv} wv={wv}"
            );
        }
    }

    #[test]
    fn wallis_partial_product_ratios() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let n = pool.symbol("n", Domain::Real);
        let two = pool.integer(2_i32);
        let km1 = simp(&pool, pool.add(vec![k.clone(), pool.integer(-1)]));
        let kp1 = simp(&pool, pool.add(vec![k.clone(), pool.integer(1)]));
        let k2 = simp(&pool, pool.pow(k.clone(), pool.integer(2)));
        let term = simp(
            &pool,
            pool.mul(vec![
                simp(&pool, pool.mul(vec![km1, kp1])),
                pool.pow(k2, pool.integer(-1)),
            ]),
        );

        let p = product_definite(term, k, two.clone(), n.clone(), &pool).expect("wallis");
        for ni in 3..36 {
            let mut env = HashMap::new();
            env.insert(n, ni as f64);
            let pv = eval_g(p.value, &env, &pool).unwrap();
            let want = (ni + 1) as f64 / (2.0 * ni as f64);
            assert!(
                (pv - want).abs() < 1e-5 * want.max(1.0),
                "n={}: got {}",
                ni,
                pv
            );
        }
    }
}
