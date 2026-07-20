//! Extract rational hypergeometric shift ratios `F(k+1)/F(k)` from symbolic terms.

use super::poly_aux::compose_affine;
use super::ratfunc::RatFunc;
use super::SumError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;
use crate::poly::UniPoly;
use rug::Rational;

fn rat_poly_from_unipoly(p: &UniPoly) -> RatUniPoly {
    let coeffs: Vec<Rational> = p.coefficients().into_iter().map(Rational::from).collect();
    RatUniPoly { coeffs }.trim()
}

fn poly_shift_ratio(p: &RatUniPoly) -> RatFunc {
    let shifted = compose_affine(p, &Rational::from(1), &Rational::from(1));
    RatFunc {
        num: shifted,
        den: p.clone(),
    }
    .normalize()
}

/// Γ(a·k+b + a)/Γ(a·k+b) for integer step `a ≥ 0`.
fn gamma_linear_ratio(a_step: i64, b: Rational) -> Result<RatFunc, SumError> {
    if a_step == 0 {
        return Ok(RatFunc::one());
    }
    if a_step < 0 {
        return Err(SumError::NotHypergeometric(
            "gamma linear argument with negative slope not supported".into(),
        ));
    }
    let z = RatUniPoly::x();
    let mut num = RatUniPoly::one();
    for t in 0..a_step {
        let coeff_k = Rational::from(a_step);
        let const_term = b.clone() + Rational::from(t);
        let lin = &(&RatUniPoly::constant(coeff_k) * &z) + &RatUniPoly::constant(const_term);
        num = num * lin;
    }
    Ok(RatFunc {
        num,
        den: RatUniPoly::one(),
    }
    .normalize())
}

pub fn hypergeom_ratio(term: ExprId, k: ExprId, pool: &ExprPool) -> Result<RatFunc, SumError> {
    ratio_factor(term, k, pool)
}

fn ratio_product(term: ExprId, k: ExprId, pool: &ExprPool) -> Result<RatFunc, SumError> {
    let data = pool.get(term);
    if let ExprData::Mul(args) = data {
        let mut acc = RatFunc::one();
        for &a in &args {
            acc = acc.mul_ratfunc(&ratio_factor(a, k, pool)?);
        }
        Ok(acc.normalize())
    } else {
        ratio_factor(term, k, pool)
    }
}

fn ratio_factor(f: ExprId, k: ExprId, pool: &ExprPool) -> Result<RatFunc, SumError> {
    if f == k {
        let num = compose_affine(&RatUniPoly::x(), &Rational::from(1), &Rational::from(1));
        return Ok(RatFunc {
            num,
            den: RatUniPoly::x(),
        }
        .normalize());
    }

    match pool.get(f) {
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => Ok(RatFunc::one()),
        ExprData::Symbol { .. } => {
            if f == k {
                let num = compose_affine(&RatUniPoly::x(), &Rational::from(1), &Rational::from(1));
                Ok(RatFunc {
                    num,
                    den: RatUniPoly::x(),
                }
                .normalize())
            } else {
                Ok(RatFunc::one())
            }
        }
        ExprData::Pow { base, exp } => {
            // Geometric / exponential-in-k: c^k or c^{a·k+b} with c free of k.
            if is_free_of_k(base, k, pool) && !is_free_of_k(exp, k, pool) {
                return geometric_base_ratio(base, exp, k, pool);
            }
            let e = match pool.get(exp) {
                ExprData::Integer(n) => {
                    n.0.to_i32()
                        .ok_or_else(|| SumError::NotHypergeometric("exponent too large".into()))?
                }
                _ => return Err(SumError::NotHypergeometric("non-integer exponent".into())),
            };
            if e == 0 {
                return Ok(RatFunc::one());
            }
            let rb = ratio_factor(base, k, pool)?;
            if e > 0 {
                let mut acc = rb.clone();
                for _ in 1..e {
                    acc = acc.mul_ratfunc(&rb);
                }
                Ok(acc)
            } else {
                let inv = rb.inv().ok_or_else(|| {
                    SumError::NotHypergeometric("zero base raised to a negative power".into())
                })?;
                let mut acc = inv.clone();
                for _ in 1..(-e) {
                    acc = acc.mul_ratfunc(&inv);
                }
                Ok(acc)
            }
        }
        ExprData::Mul(_) => ratio_product(f, k, pool),
        ExprData::Add(_) => {
            let p = UniPoly::from_symbolic(f, k, pool).map_err(|e| {
                SumError::NotHypergeometric(format!("expected polynomial in k: {e}"))
            })?;
            let rp = rat_poly_from_unipoly(&p);
            Ok(poly_shift_ratio(&rp))
        }
        ExprData::Func { name, args } => {
            if name == "gamma" && args.len() == 1 {
                let p = UniPoly::from_symbolic(args[0], k, pool).map_err(|e| {
                    SumError::NotHypergeometric(format!(
                        "gamma argument must be polynomial in k: {e}"
                    ))
                })?;
                if p.degree() > 1 {
                    return Err(SumError::NotHypergeometric(
                        "gamma argument must be linear in k".into(),
                    ));
                }
                let coeffs = p.coefficients();
                let b = if coeffs.is_empty() {
                    Rational::from(0)
                } else {
                    Rational::from(coeffs[0].clone())
                };
                let a = if coeffs.len() > 1 {
                    coeffs[1]
                        .to_i64()
                        .ok_or_else(|| SumError::NotHypergeometric("gamma slope".into()))?
                } else {
                    0_i64
                };
                return gamma_linear_ratio(a, b);
            }
            Err(SumError::NotHypergeometric(format!(
                "unsupported hypergeometric factor `{name}`"
            )))
        }
        _ => Err(SumError::NotHypergeometric(
            "unsupported expression shape".into(),
        )),
    }
}

fn is_free_of_k(expr: ExprId, k: ExprId, pool: &ExprPool) -> bool {
    if expr == k {
        return false;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => args.iter().all(|&a| is_free_of_k(a, k, pool)),
        ExprData::Pow { base, exp } => is_free_of_k(base, k, pool) && is_free_of_k(exp, k, pool),
        ExprData::Func { args, .. } => args.iter().all(|&a| is_free_of_k(a, k, pool)),
        _ => true,
    }
}

fn const_to_rational(expr: ExprId, pool: &ExprPool) -> Result<Rational, SumError> {
    match pool.get(expr) {
        ExprData::Integer(n) => Ok(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Ok(r.0.clone()),
        ExprData::Mul(args) => {
            let mut acc = Rational::from(1);
            for &a in &args {
                acc *= const_to_rational(a, pool)?;
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => {
            let b = const_to_rational(base, pool)?;
            let e = match pool.get(exp) {
                ExprData::Integer(n) => n.0.to_i32().ok_or_else(|| {
                    SumError::NotHypergeometric("constant exponent too large".into())
                })?,
                _ => {
                    return Err(SumError::NotHypergeometric(
                        "geometric base must be a rational constant".into(),
                    ))
                }
            };
            if e >= 0 {
                let mut p = Rational::from(1);
                for _ in 0..e {
                    p *= b.clone();
                }
                Ok(p)
            } else {
                if b.is_zero() {
                    return Err(SumError::NotHypergeometric("zero geometric base".into()));
                }
                let mut p = Rational::from(1);
                for _ in 0..(-e) {
                    p *= b.clone();
                }
                Ok(Rational::from(1) / p)
            }
        }
        _ => Err(SumError::NotHypergeometric(
            "geometric base must be a rational constant".into(),
        )),
    }
}

/// `F(k) = c^{a·k+b}` with `c` free of `k` ⇒ `F(k+1)/F(k) = c^a`.
fn geometric_base_ratio(
    base: ExprId,
    exp: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Result<RatFunc, SumError> {
    let c = const_to_rational(base, pool)?;
    if c.is_zero() {
        return Err(SumError::NotHypergeometric(
            "geometric term with zero base".into(),
        ));
    }
    let a = affine_slope_in_k(exp, k, pool)?;
    if a == 0 {
        return Ok(RatFunc::one());
    }
    let mut pow = Rational::from(1);
    let steps = a.unsigned_abs();
    for _ in 0..steps {
        pow *= c.clone();
    }
    if a > 0 {
        Ok(RatFunc::scalar(pow))
    } else {
        Ok(RatFunc::scalar(Rational::from(1) / pow))
    }
}

/// Extract integer slope `a` from an affine exponent `a·k + b` (`b` free of `k`).
fn affine_slope_in_k(exp: ExprId, k: ExprId, pool: &ExprPool) -> Result<i64, SumError> {
    if exp == k {
        return Ok(1);
    }
    match pool.get(exp) {
        ExprData::Add(args) => {
            let mut slope = 0_i64;
            for &a in &args {
                slope += affine_slope_in_k(a, k, pool)?;
            }
            Ok(slope)
        }
        ExprData::Mul(args) => {
            let mut slope_part = None;
            let mut coeff = Rational::from(1);
            for &a in &args {
                if !is_free_of_k(a, k, pool) {
                    if slope_part.is_some() {
                        return Err(SumError::NotHypergeometric(
                            "geometric exponent must be affine in k".into(),
                        ));
                    }
                    slope_part = Some(affine_slope_in_k(a, k, pool)?);
                } else {
                    coeff *= const_to_rational(a, pool)?;
                }
            }
            match slope_part {
                Some(s) => {
                    let scaled = coeff * Rational::from(s);
                    if *scaled.denom() != 1 {
                        return Err(SumError::NotHypergeometric(
                            "geometric exponent slope must be an integer".into(),
                        ));
                    }
                    scaled.numer().to_i64().ok_or_else(|| {
                        SumError::NotHypergeometric(
                            "geometric exponent slope must be an integer".into(),
                        )
                    })
                }
                None => Ok(0),
            }
        }
        _ if is_free_of_k(exp, k, pool) => Ok(0),
        _ => Err(SumError::NotHypergeometric(
            "geometric exponent must be affine in k".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::sum::gosper_certificate;

    #[test]
    fn ratio_k_times_gamma_k_plus_1() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let g = pool.func("gamma", vec![pool.add(vec![k, pool.integer(1_i32)])]);
        let term = pool.mul(vec![k, g]);
        let r = hypergeom_ratio(term, k, &pool).unwrap();
        let cert = gosper_certificate(&r).expect("gosper");
        assert!(!cert.num.is_zero());
    }

    #[test]
    fn ratio_geometric_two_pow_k() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let term = pool.pow(pool.integer(2_i32), k);
        let r = hypergeom_ratio(term, k, &pool).expect("2^k is hypergeometric");
        assert_eq!(r.den.degree(), 0);
        assert_eq!(r.num.coeffs[0], Rational::from(2));
        assert!(gosper_certificate(&r).is_some());
    }

    #[test]
    fn ratio_inv_k_times_k_plus_1() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let den = pool.mul(vec![k, pool.add(vec![k, pool.integer(1_i32)])]);
        let term = pool.pow(den, pool.integer(-1_i32));
        let r = hypergeom_ratio(term, k, &pool).expect("1/(k(k+1)) ratio");
        assert!(gosper_certificate(&r).is_some());
    }
}
