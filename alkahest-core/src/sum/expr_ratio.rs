//! Extract rational hypergeometric shift ratios `F(k+1)/F(k)` from symbolic terms.

use super::poly_aux::compose_affine;
use super::ratfunc::RatFunc;
use super::SumError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;
use crate::poly::UniPoly;
use rug::{Integer, Rational};

fn rat_poly_from_unipoly(p: &UniPoly) -> RatUniPoly {
    let coeffs: Vec<Rational> = p
        .coefficients()
        .into_iter()
        .map(|c| Rational::from(Integer::from(c)))
        .collect();
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
            let mut acc = rb.clone();
            for _ in 1..e {
                acc = acc.mul_ratfunc(&rb);
            }
            Ok(acc)
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
                    Rational::from(Integer::from(coeffs[0].clone()))
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
}
