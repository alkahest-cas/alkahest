//! Residue of a rational meromorphic function at a point in ℚ(i).

use rug::Rational;

use crate::integrate::risch::poly_rde::trim;
use crate::integrate::risch::rational_rde::{expr_to_qrational, poly_div_exact, poly_gcd};
use crate::kernel::{ExprId, ExprPool};

use super::gauss::{pole_order, rational_laurent_residue, GaussPoly, GaussRat};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidueError {
    NotRational,
    ZeroDenominator,
    PoleOrderTooHigh { order: u32 },
    DivisionByZero,
}

impl ResidueError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NotRational => "E-RESIDUE-001",
            Self::ZeroDenominator => "E-RESIDUE-002",
            Self::PoleOrderTooHigh { .. } => "E-RESIDUE-003",
            Self::DivisionByZero => "E-RESIDUE-004",
        }
    }
}

impl std::fmt::Display for ResidueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRational => write!(f, "residue: not a rational function of the variable"),
            Self::ZeroDenominator => write!(f, "residue: zero denominator"),
            Self::PoleOrderTooHigh { order } => write!(f, "residue: pole order {order} too high"),
            Self::DivisionByZero => write!(f, "residue: division by zero during extraction"),
        }
    }
}

impl std::error::Error for ResidueError {}

const MAX_POLE_ORDER: u32 = 32;

pub fn residue(
    expr: ExprId,
    var: ExprId,
    point: GaussRat,
    pool: &ExprPool,
) -> Result<ExprId, ResidueError> {
    let (mut num, mut den) = expr_to_qrational(expr, var, pool).ok_or(ResidueError::NotRational)?;
    num = trim(num);
    den = trim(den);
    if den.is_empty() || den.iter().all(|c| *c == 0) {
        return Err(ResidueError::ZeroDenominator);
    }
    let g = poly_gcd(&num, &den);
    num = poly_div_exact(&num, &g);
    den = poly_div_exact(&den, &g);
    if !GaussRat::eval_poly(&den, &point).is_zero() {
        return Ok(pool.integer(0_i32));
    }
    let m = pole_order(&den, &point);
    if m == 0 {
        return Ok(pool.integer(0_i32));
    }
    if m > MAX_POLE_ORDER {
        return Err(ResidueError::PoleOrderTooHigh { order: m });
    }
    let value = rational_laurent_residue(
        &GaussPoly::from_rational_poly(&num),
        &GaussPoly::from_rational_poly(&den),
        &point,
        m,
    )
    .ok_or(ResidueError::DivisionByZero)?;
    Ok(gauss_to_expr(&value, pool))
}

fn gauss_to_expr(g: &GaussRat, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    if g.re != 0 {
        let (n, d) = g.re.clone().into_numer_denom();
        terms.push(if d == 1 {
            pool.integer(n)
        } else {
            pool.rational(n, d)
        });
    }
    if g.im != 0 {
        let i = pool.imaginary_unit();
        let im = if g.im == Rational::from(1) {
            i
        } else if g.im == Rational::from(-1) {
            pool.mul(vec![pool.integer(-1_i32), i])
        } else {
            let (n, d) = g.im.clone().into_numer_denom();
            let coeff = if d == 1 {
                pool.integer(n)
            } else {
                pool.rational(n, d)
            };
            pool.mul(vec![coeff, i])
        };
        terms.push(im);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn pt(re: i64, im: i64) -> GaussRat {
        GaussRat::from_re_im(Rational::from(re), Rational::from(im))
    }

    #[test]
    fn residue_simple_pole_at_origin() {
        let pool = ExprPool::new();
        let z = pool.symbol("z", Domain::Complex);
        let r = residue(pool.pow(z, pool.integer(-1_i32)), z, pt(0, 0), &pool).unwrap();
        assert_eq!(pool.display(r).to_string(), "1");
    }

    #[test]
    fn residue_double_pole_is_zero() {
        let pool = ExprPool::new();
        let z = pool.symbol("z", Domain::Complex);
        let a = pool.integer(2_i32);
        let expr = pool.pow(
            pool.add(vec![z, pool.mul(vec![pool.integer(-1_i32), a])]),
            pool.integer(-2_i32),
        );
        let r = residue(expr, z, pt(2, 0), &pool).unwrap();
        assert_eq!(pool.display(r).to_string(), "0");
    }

    #[test]
    fn residue_at_i_for_reciprocal_quadratic() {
        let pool = ExprPool::new();
        let z = pool.symbol("z", Domain::Complex);
        let den = pool.add(vec![pool.pow(z, pool.integer(2_i32)), pool.integer(1_i32)]);
        let r = residue(pool.pow(den, pool.integer(-1_i32)), z, pt(0, 1), &pool).unwrap();
        let expected = pool.mul(vec![pool.rational(-1, 2), pool.imaginary_unit()]);
        assert_eq!(
            pool.display(r).to_string(),
            pool.display(expected).to_string()
        );
    }

    #[test]
    fn non_rational_declines() {
        let pool = ExprPool::new();
        let z = pool.symbol("z", Domain::Complex);
        assert_eq!(
            residue(pool.func("sin", vec![z]), z, pt(0, 0), &pool),
            Err(ResidueError::NotRational)
        );
    }
}
