//! Rational functions ℚ(k) for hypergeometric term ratios.

use super::poly_aux::compose_affine;
use crate::matrix::normal_form::RatUniPoly;
use rug::Rational;
use std::ops::{Add, Mul, Neg};

/// Reduced quotient `num / den` of coprime polynomials (`den` monic where possible).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RatFunc {
    pub num: RatUniPoly,
    pub den: RatUniPoly,
}

impl RatFunc {
    pub fn zero() -> Self {
        Self {
            num: RatUniPoly::zero(),
            den: RatUniPoly::one(),
        }
    }

    pub fn one() -> Self {
        Self {
            num: RatUniPoly::one(),
            den: RatUniPoly::one(),
        }
    }

    pub fn scalar(z: Rational) -> Self {
        Self {
            num: RatUniPoly::constant(z),
            den: RatUniPoly::one(),
        }
    }

    pub fn from_poly(p: RatUniPoly) -> Self {
        Self {
            num: p,
            den: RatUniPoly::one(),
        }
    }

    pub fn normalize(mut self) -> Self {
        if self.num.is_zero() {
            return Self::zero();
        }
        let g = RatUniPoly::gcd(&self.num, &self.den);
        if !g.is_zero() && g.degree() >= 0 {
            if let Some(nn) = super::poly_aux::poly_exact_div(&self.num, &g) {
                self.num = nn;
            }
            if let Some(dd) = super::poly_aux::poly_exact_div(&self.den, &g) {
                self.den = dd;
            }
        }
        // Make denominator monic
        if !self.den.is_zero() {
            let lc = self.den.leading_coeff();
            if lc != 1 {
                let inv = Rational::from(1) / lc.clone();
                self.num = scale_poly(&self.num, &inv);
                self.den = scale_poly(&self.den, &inv);
            }
        }
        self
    }

    pub fn mul_ratfunc(&self, other: &RatFunc) -> RatFunc {
        RatFunc {
            num: self.num.clone() * other.num.clone(),
            den: self.den.clone() * other.den.clone(),
        }
        .normalize()
    }

    pub fn inv(&self) -> Option<RatFunc> {
        if self.num.is_zero() {
            return None;
        }
        Some(
            RatFunc {
                num: self.den.clone(),
                den: self.num.clone(),
            }
            .normalize(),
        )
    }

    /// Affine shift of argument: `self(a·k+b)` as rational function in `k`.
    pub fn compose_affine_arg(&self, a: &Rational, b: &Rational) -> RatFunc {
        RatFunc {
            num: compose_affine(&self.num, a, b),
            den: compose_affine(&self.den, a, b),
        }
        .normalize()
    }
}

pub fn scale_poly(p: &RatUniPoly, z: &Rational) -> RatUniPoly {
    if p.is_zero() || z == &Rational::from(1) {
        return p.clone();
    }
    let coeffs: Vec<Rational> = p.coeffs.iter().map(|c| c.clone() * z.clone()).collect();
    RatUniPoly { coeffs }.trim()
}

impl Neg for RatFunc {
    type Output = RatFunc;
    fn neg(self) -> RatFunc {
        RatFunc {
            num: scale_poly(&self.num, &Rational::from(-1)),
            den: self.den,
        }
    }
}

impl Add for RatFunc {
    type Output = RatFunc;
    fn add(self, rhs: RatFunc) -> RatFunc {
        let num = &(&self.num * &rhs.den) + &(&rhs.num * &self.den);
        let den = &self.den * &rhs.den;
        RatFunc { num, den }.normalize()
    }
}

impl Mul for RatFunc {
    type Output = RatFunc;
    fn mul(self, rhs: RatFunc) -> RatFunc {
        self.mul_ratfunc(&rhs)
    }
}
