//! Exact arithmetic in ℚ(i).

use rug::{Integer, Rational};

use crate::integrate::risch::poly_rde::{poly_deriv, trim, QPoly};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GaussRat {
    pub re: Rational,
    pub im: Rational,
}

impl GaussRat {
    pub fn zero() -> Self {
        Self {
            re: Rational::from(0),
            im: Rational::from(0),
        }
    }
    pub fn one() -> Self {
        Self {
            re: Rational::from(1),
            im: Rational::from(0),
        }
    }
    pub fn from_re_im(re: Rational, im: Rational) -> Self {
        Self { re, im }
    }
    pub fn is_zero(&self) -> bool {
        self.re == 0 && self.im == 0
    }
    pub fn add(&self, o: &Self) -> Self {
        Self {
            re: self.re.clone() + o.re.clone(),
            im: self.im.clone() + o.im.clone(),
        }
    }
    pub fn sub(&self, o: &Self) -> Self {
        Self {
            re: self.re.clone() - o.re.clone(),
            im: self.im.clone() - o.im.clone(),
        }
    }
    pub fn mul(&self, o: &Self) -> Self {
        Self {
            re: self.re.clone() * o.re.clone() - self.im.clone() * o.im.clone(),
            im: self.re.clone() * o.im.clone() + self.im.clone() * o.re.clone(),
        }
    }
    pub fn div(&self, o: &Self) -> Option<Self> {
        let den = o.re.clone() * o.re.clone() + o.im.clone() * o.im.clone();
        if den == 0 {
            return None;
        }
        Some(Self {
            re: (self.re.clone() * o.re.clone() + self.im.clone() * o.im.clone()) / &den,
            im: (self.im.clone() * o.re.clone() - self.re.clone() * o.im.clone()) / &den,
        })
    }
    pub fn neg(&self) -> Self {
        Self {
            re: -self.re.clone(),
            im: -self.im.clone(),
        }
    }
    pub fn eval_poly(p: &QPoly, z: &Self) -> Self {
        let mut acc = Self::zero();
        let mut power = Self::one();
        for c in p.iter() {
            acc = acc.add(
                &Self {
                    re: c.clone(),
                    im: Rational::from(0),
                }
                .mul(&power),
            );
            power = power.mul(z);
        }
        acc
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GaussPoly {
    pub coeffs: Vec<GaussRat>,
}

impl GaussPoly {
    pub fn from_rational_poly(p: &QPoly) -> Self {
        Self {
            coeffs: p
                .iter()
                .map(|c| GaussRat {
                    re: c.clone(),
                    im: Rational::from(0),
                })
                .collect(),
        }
    }
    pub fn shift_var(z0: &GaussRat) -> Self {
        Self {
            coeffs: vec![z0.neg(), GaussRat::one()],
        }
    }
    pub fn mul(&self, o: &Self) -> Self {
        let mut out = vec![GaussRat::zero(); self.coeffs.len() + o.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in o.coeffs.iter().enumerate() {
                out[i + j] = out[i + j].add(&a.mul(b));
            }
        }
        Self {
            coeffs: trim_gauss(out),
        }
    }
    pub fn pow(&self, n: u32) -> Self {
        let mut acc = Self {
            coeffs: vec![GaussRat::one()],
        };
        let mut base = self.clone();
        let mut exp = n;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.mul(&base);
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base);
            }
        }
        acc
    }
    pub fn deriv(&self) -> Self {
        let mut out = Vec::new();
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            let s = Rational::from(i as i64);
            out.push(GaussRat {
                re: c.re.clone() * &s,
                im: c.im.clone() * &s,
            });
        }
        Self { coeffs: out }
    }
    pub fn eval(&self, z: &GaussRat) -> GaussRat {
        self.coeffs
            .iter()
            .rev()
            .fold(GaussRat::zero(), |acc, c| acc.mul(z).add(c))
    }
    pub fn taylor_coeffs(&self, z0: &GaussRat, order: u32) -> Vec<GaussRat> {
        let mut coeffs = Vec::new();
        let mut cur = self.clone();
        let mut fact = GaussRat::one();
        for k in 0..=order {
            coeffs.push(cur.eval(z0).div(&fact).unwrap_or_else(GaussRat::zero));
            fact = fact.mul(&GaussRat::from_re_im(
                Rational::from((k + 1) as i64),
                Rational::from(0),
            ));
            cur = cur.deriv();
        }
        coeffs
    }
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let a = trim_gauss(self.coeffs.clone());
        let b = trim_gauss(divisor.coeffs.clone());
        let bd = gauss_degree(&b);
        let ad = gauss_degree(&a);
        if ad < bd {
            return (Self { coeffs: vec![] }, Self { coeffs: a });
        }
        let mut r = a;
        let mut q = vec![GaussRat::zero(); (ad - bd + 1) as usize];
        let lcb = b[bd as usize].clone();
        while gauss_degree(&r) >= bd {
            let rd = gauss_degree(&r);
            let exp = (rd - bd) as usize;
            let coeff = r[rd as usize].div(&lcb).unwrap();
            q[exp] = coeff.clone();
            let mut term = vec![GaussRat::zero(); exp + b.len()];
            for (j, bc) in b.iter().enumerate() {
                term[exp + j] = bc.mul(&coeff);
            }
            r = gauss_sub(&r, &term);
        }
        (
            Self {
                coeffs: trim_gauss(q),
            },
            Self {
                coeffs: trim_gauss(r),
            },
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GaussTaylor {
    pub coeffs: Vec<GaussRat>,
}

impl GaussTaylor {
    pub fn mul(&self, o: &Self) -> Self {
        let n = self.coeffs.len().max(o.coeffs.len());
        let mut out = vec![GaussRat::zero(); n];
        for i in 0..n {
            for j in 0..=i {
                if j < self.coeffs.len() && (i - j) < o.coeffs.len() {
                    out[i] = out[i].add(&self.coeffs[j].mul(&o.coeffs[i - j]));
                }
            }
        }
        Self { coeffs: out }
    }
    pub fn inv(&self) -> Option<Self> {
        if self.coeffs.first()?.is_zero() {
            return None;
        }
        let n = self.coeffs.len();
        let mut out = vec![GaussRat::zero(); n];
        out[0] = GaussRat::one().div(&self.coeffs[0])?;
        for i in 1..n {
            let mut sum = GaussRat::zero();
            for j in 1..=i {
                sum = sum.add(&self.coeffs[j].mul(&out[i - j]));
            }
            out[i] = sum.neg().div(&self.coeffs[0])?;
        }
        Some(Self { coeffs: out })
    }
}

pub fn rational_laurent_residue(
    num: &GaussPoly,
    den: &GaussPoly,
    z0: &GaussRat,
    pole_order: u32,
) -> Option<GaussRat> {
    let zm = GaussPoly::shift_var(z0).pow(pole_order);
    let (b, rem) = den.div_rem(&zm);
    if !rem.coeffs.iter().all(|c| c.is_zero()) {
        return None;
    }
    let order = pole_order - 1;
    let h = GaussTaylor {
        coeffs: num.taylor_coeffs(z0, order),
    }
    .mul(
        &GaussTaylor {
            coeffs: b.taylor_coeffs(z0, order),
        }
        .inv()?,
    );
    h.coeffs.get(order as usize).cloned()
}

fn gauss_degree(p: &[GaussRat]) -> i64 {
    let t = trim_gauss(p.to_vec());
    (t.len() as i64) - 1
}

fn trim_gauss(mut p: Vec<GaussRat>) -> Vec<GaussRat> {
    while p.len() > 1 && p.last().is_some_and(|c| c.is_zero()) {
        p.pop();
    }
    if p.is_empty() {
        vec![GaussRat::zero()]
    } else {
        p
    }
}

fn gauss_sub(a: &[GaussRat], b: &[GaussRat]) -> Vec<GaussRat> {
    let n = a.len().max(b.len());
    let mut out = vec![GaussRat::zero(); n];
    for (i, c) in a.iter().enumerate() {
        out[i] = c.clone();
    }
    for (i, c) in b.iter().enumerate() {
        out[i] = out[i].sub(c);
    }
    trim_gauss(out)
}

pub fn pole_order(den: &QPoly, z0: &GaussRat) -> u32 {
    let mut m = 0u32;
    let mut d = den.clone();
    while GaussRat::eval_poly(&d, z0).is_zero() {
        m += 1;
        d = poly_deriv(&d);
        if trim(d.clone()).is_empty() {
            break;
        }
    }
    m
}

pub fn factorial(n: u32) -> Rational {
    let mut acc = Integer::from(1);
    for k in 2..=n {
        acc *= Integer::from(k);
    }
    Rational::from(acc)
}
