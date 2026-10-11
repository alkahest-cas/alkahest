//! Exact coefficients of the classical orthogonal polynomials.
//!
//! Every family is generated from a closed-form sum (DLMF §18.5) rather than
//! from the three-term recurrence, so the cost is `O(n)` coefficient products
//! and no division by a parameter-dependent quantity ever happens: a Jacobi
//! polynomial with `a + b = −2` or a Gegenbauer polynomial with `a = 0` is
//! produced from the same formula as every other, with no special case.
//!
//! Coefficients live in [`ParamPoly`], a polynomial with rational
//! coefficients in at most two family parameters (`α`, `a`, `b`). A parameter
//! that is a rational literal enters as a constant, so the common case stays
//! one rational per coefficient.

use std::collections::BTreeMap;

use rug::{Integer, Rational};

use super::error::SpecialFunctionError;

/// A polynomial with rational coefficients in up to two parameters
/// `p₀`, `p₁`, keyed by the exponent pair `(i, j)` of `p₀ⁱ·p₁ʲ`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ParamPoly(pub(crate) BTreeMap<(u32, u32), Rational>);

impl ParamPoly {
    pub(crate) fn zero() -> Self {
        ParamPoly(BTreeMap::new())
    }

    pub(crate) fn constant(q: Rational) -> Self {
        let mut m = BTreeMap::new();
        if q != 0 {
            m.insert((0, 0), q);
        }
        ParamPoly(m)
    }

    /// The parameter `p_i` itself (`i ∈ {0, 1}`).
    pub(crate) fn var(i: usize) -> Self {
        let mut m = BTreeMap::new();
        m.insert(if i == 0 { (1, 0) } else { (0, 1) }, Rational::from(1));
        ParamPoly(m)
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// The value, when this is a constant.
    pub(crate) fn as_constant(&self) -> Option<Rational> {
        match self.0.len() {
            0 => Some(Rational::new()),
            1 => self.0.get(&(0, 0)).cloned(),
            _ => None,
        }
    }

    fn add_term(&mut self, key: (u32, u32), c: Rational) {
        if c == 0 {
            return;
        }
        let slot = self.0.entry(key).or_default();
        *slot += c;
        if *slot == 0 {
            self.0.remove(&key);
        }
    }

    pub(crate) fn add_assign(&mut self, other: &ParamPoly) {
        for (k, c) in &other.0 {
            self.add_term(*k, c.clone());
        }
    }

    pub(crate) fn scaled(&self, q: &Rational) -> ParamPoly {
        if *q == 0 {
            return ParamPoly::zero();
        }
        ParamPoly(
            self.0
                .iter()
                .map(|(k, c)| (*k, Rational::from(c * q)))
                .collect(),
        )
    }

    pub(crate) fn mul(&self, other: &ParamPoly) -> ParamPoly {
        let mut out = ParamPoly::zero();
        for (&(i1, j1), c1) in &self.0 {
            for (&(i2, j2), c2) in &other.0 {
                out.add_term((i1 + i2, j1 + j2), Rational::from(c1 * c2));
            }
        }
        out
    }

    /// `self + q`.
    pub(crate) fn plus(&self, q: &Rational) -> ParamPoly {
        let mut out = self.clone();
        out.add_term((0, 0), q.clone());
        out
    }
}

/// How a family parameter enters the coefficients: a rational literal, or
/// the symbolic parameter `p_i`.
#[derive(Clone, Debug)]
pub(crate) enum ParamVal {
    Const(Rational),
    Sym(usize),
}

impl ParamVal {
    fn poly(&self) -> ParamPoly {
        match self {
            ParamVal::Const(q) => ParamPoly::constant(q.clone()),
            ParamVal::Sym(i) => ParamPoly::var(*i),
        }
    }
}

fn q(n: impl Into<Integer>) -> Rational {
    Rational::from(n.into())
}

fn fact(n: u32) -> Integer {
    Integer::from(Integer::factorial(n))
}

fn binom(n: u32, k: u32) -> Integer {
    Integer::from(Integer::binomial_u(n, k))
}

fn tick(k: u32) -> Result<(), SpecialFunctionError> {
    // A cooperative checkpoint every 32 outer iterations: cheap when no
    // budget is active, and it keeps a degree-1000 expansion cancellable.
    if k % 32 == 31 {
        crate::budget::check()?;
    }
    Ok(())
}

/// `P_n(x) = 2⁻ⁿ Σ_k (−1)ᵏ C(n,k) C(2n−2k, n) x^{n−2k}` (DLMF 18.5.8).
pub(crate) fn legendre(n: u32) -> Result<Vec<Rational>, SpecialFunctionError> {
    let mut c = vec![Rational::new(); n as usize + 1];
    let two_n = Integer::from(1) << n;
    for k in 0..=n / 2 {
        tick(k)?;
        let mut t = binom(n, k) * binom(2 * n - 2 * k, n);
        if k % 2 == 1 {
            t = -t;
        }
        c[(n - 2 * k) as usize] = Rational::from((t, two_n.clone()));
    }
    Ok(c)
}

/// Physicists' Hermite `H_n(x) = n! Σ_k (−1)ᵏ (2x)^{n−2k} / (k! (n−2k)!)`
/// (DLMF 18.5.13).
pub(crate) fn hermite_h(n: u32) -> Result<Vec<Rational>, SpecialFunctionError> {
    let mut c = vec![Rational::new(); n as usize + 1];
    let nf = fact(n);
    for k in 0..=n / 2 {
        tick(k)?;
        let mut num = Integer::from(&nf << (n - 2 * k));
        if k % 2 == 1 {
            num = -num;
        }
        let den = fact(k) * fact(n - 2 * k);
        c[(n - 2 * k) as usize] = Rational::from((num, den));
    }
    Ok(c)
}

/// Probabilists' Hermite `He_n(x) = n! Σ_k (−1)ᵏ x^{n−2k} / (k! (n−2k)! 2ᵏ)`
/// (DLMF 18.5.14 with 18.7.11's scaling).
pub(crate) fn hermite_he(n: u32) -> Result<Vec<Rational>, SpecialFunctionError> {
    let mut c = vec![Rational::new(); n as usize + 1];
    let nf = fact(n);
    for k in 0..=n / 2 {
        tick(k)?;
        let mut num = nf.clone();
        if k % 2 == 1 {
            num = -num;
        }
        let den = (fact(k) * fact(n - 2 * k)) << k;
        c[(n - 2 * k) as usize] = Rational::from((num, den));
    }
    Ok(c)
}

/// `T_n(x) = (n/2) Σ_k (−1)ᵏ (n−k−1)! / (k! (n−2k)!) (2x)^{n−2k}`, `T_0 = 1`
/// (DLMF 18.5.11).
pub(crate) fn chebyshev_t(n: u32) -> Result<Vec<Rational>, SpecialFunctionError> {
    if n == 0 {
        return Ok(vec![q(1)]);
    }
    let mut c = vec![Rational::new(); n as usize + 1];
    for k in 0..=n / 2 {
        tick(k)?;
        let mut num = (fact(n - k - 1) * n) << (n - 2 * k);
        if k % 2 == 1 {
            num = -num;
        }
        let den = (fact(k) * fact(n - 2 * k)) << 1u32;
        c[(n - 2 * k) as usize] = Rational::from((num, den));
    }
    Ok(c)
}

/// `U_n(x) = Σ_k (−1)ᵏ C(n−k, k) (2x)^{n−2k}` (DLMF 18.5.10).
pub(crate) fn chebyshev_u(n: u32) -> Result<Vec<Rational>, SpecialFunctionError> {
    let mut c = vec![Rational::new(); n as usize + 1];
    for k in 0..=n / 2 {
        tick(k)?;
        let mut num = binom(n - k, k) << (n - 2 * k);
        if k % 2 == 1 {
            num = -num;
        }
        c[(n - 2 * k) as usize] = Rational::from(num);
    }
    Ok(c)
}

/// `Π_{j=k+1}^{n} (p + j)` for every `k = 0..=n` (index `k`), i.e. the
/// rising factorial `(p + k + 1)_{n−k}`.
fn tail_products(p: &ParamPoly, n: u32) -> Result<Vec<ParamPoly>, SpecialFunctionError> {
    let mut out = vec![ParamPoly::zero(); n as usize + 1];
    out[n as usize] = ParamPoly::constant(q(1));
    for k in (0..n).rev() {
        tick(n - k)?;
        out[k as usize] = out[k as usize + 1].mul(&p.plus(&q(k + 1)));
    }
    Ok(out)
}

/// Generalised Laguerre
/// `L_n^{(α)}(x) = Σ_k (−1)ᵏ (α+k+1)_{n−k} / ((n−k)! k!) xᵏ` (DLMF 18.5.12).
pub(crate) fn laguerre(n: u32, alpha: &ParamVal) -> Result<Vec<ParamPoly>, SpecialFunctionError> {
    let tails = tail_products(&alpha.poly(), n)?;
    let mut c = Vec::with_capacity(n as usize + 1);
    for (k, tail) in tails.iter().enumerate() {
        let k = k as u32;
        tick(k)?;
        let mut num = Integer::from(1);
        if k % 2 == 1 {
            num = -num;
        }
        let s = Rational::from((num, fact(n - k) * fact(k)));
        c.push(tail.scaled(&s));
    }
    Ok(c)
}

/// Gegenbauer
/// `C_n^{(a)}(x) = Σ_k (−1)ᵏ (a)_{n−k} / (k! (n−2k)!) (2x)^{n−2k}` (DLMF 18.5.10).
pub(crate) fn gegenbauer(n: u32, a: &ParamVal) -> Result<Vec<ParamPoly>, SpecialFunctionError> {
    let a = a.poly();
    // rising[j] = (a)_j
    let mut rising = Vec::with_capacity(n as usize + 1);
    rising.push(ParamPoly::constant(q(1)));
    for j in 0..n {
        tick(j)?;
        let next = rising[j as usize].mul(&a.plus(&q(j)));
        rising.push(next);
    }
    let mut c = vec![ParamPoly::zero(); n as usize + 1];
    for k in 0..=n / 2 {
        tick(k)?;
        let mut num = Integer::from(1) << (n - 2 * k);
        if k % 2 == 1 {
            num = -num;
        }
        let s = Rational::from((num, fact(k) * fact(n - 2 * k)));
        c[(n - 2 * k) as usize] = rising[(n - k) as usize].scaled(&s);
    }
    Ok(c)
}

/// Jacobi, from the hypergeometric form (DLMF 18.5.7)
/// `P_n^{(a,b)}(x) = Σ_k C(n,k)(−1)ᵏ (n+a+b+1)_k (a+k+1)_{n−k} / n! · tᵏ`,
/// `t = (1 − x)/2`, expanded in `x` by Horner's rule.
pub(crate) fn jacobi(
    n: u32,
    a: &ParamVal,
    b: &ParamVal,
) -> Result<Vec<ParamPoly>, SpecialFunctionError> {
    let ap = a.poly();
    let mut s = ap.clone();
    s.add_assign(&b.poly());
    let s = s.plus(&q(n + 1)); // n + a + b + 1
    let tails = tail_products(&ap, n)?;
    let nf = fact(n);
    // ck[k] = C(n,k) (−1)^k (s)_k (a+k+1)_{n−k} / n!
    let mut rising = ParamPoly::constant(q(1));
    let mut ck = Vec::with_capacity(n as usize + 1);
    for k in 0..=n {
        tick(k)?;
        let mut num = binom(n, k);
        if k % 2 == 1 {
            num = -num;
        }
        let scale = Rational::from((num, nf.clone()));
        ck.push(rising.mul(&tails[k as usize]).scaled(&scale));
        rising = rising.mul(&s.plus(&q(k)));
    }
    // Horner in t = 1/2 − x/2.
    let half = Rational::from((1, 2));
    let mut res: Vec<ParamPoly> = vec![ck[n as usize].clone()];
    for k in (0..n).rev() {
        tick(n - k)?;
        let mut next = vec![ParamPoly::zero(); res.len() + 1];
        for (i, c) in res.iter().enumerate() {
            let h = c.scaled(&half);
            next[i].add_assign(&h);
            next[i + 1].add_assign(&h.scaled(&q(-1)));
        }
        next[0].add_assign(&ck[k as usize]);
        res = next;
    }
    Ok(res)
}

/// The `m`-th derivative of a dense rational polynomial.
pub(crate) fn derivative(c: &[Rational], m: u32) -> Vec<Rational> {
    let m = m as usize;
    if m >= c.len() {
        return vec![];
    }
    (m..c.len())
        .map(|k| {
            // falling factorial k (k−1) … (k−m+1)
            let mut f = Integer::from(1);
            for j in 0..m {
                f *= (k - j) as u64;
            }
            Rational::from(&c[k] * f)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    #[test]
    fn low_degree_tables() {
        // P_3 = (5x³ − 3x)/2
        assert_eq!(
            legendre(3).unwrap(),
            vec![r(0, 1), r(-3, 2), r(0, 1), r(5, 2)]
        );
        // H_3 = 8x³ − 12x
        assert_eq!(
            hermite_h(3).unwrap(),
            vec![r(0, 1), r(-12, 1), r(0, 1), r(8, 1)]
        );
        // He_4 = x⁴ − 6x² + 3
        assert_eq!(
            hermite_he(4).unwrap(),
            vec![r(3, 1), r(0, 1), r(-6, 1), r(0, 1), r(1, 1)]
        );
        // T_4 = 8x⁴ − 8x² + 1, U_3 = 8x³ − 4x
        assert_eq!(
            chebyshev_t(4).unwrap(),
            vec![r(1, 1), r(0, 1), r(-8, 1), r(0, 1), r(8, 1)]
        );
        assert_eq!(
            chebyshev_u(3).unwrap(),
            vec![r(0, 1), r(-4, 1), r(0, 1), r(8, 1)]
        );
    }

    #[test]
    fn laguerre_two() {
        // L_2(x) = (x² − 4x + 2)/2
        let c: Vec<_> = laguerre(2, &ParamVal::Const(r(0, 1)))
            .unwrap()
            .into_iter()
            .map(|p| p.as_constant().unwrap())
            .collect();
        assert_eq!(c, vec![r(1, 1), r(-2, 1), r(1, 2)]);
    }

    #[test]
    fn jacobi_reduces_to_legendre_and_gegenbauer_to_chebyshev_u() {
        for n in 0..9u32 {
            let j: Vec<_> = jacobi(n, &ParamVal::Const(r(0, 1)), &ParamVal::Const(r(0, 1)))
                .unwrap()
                .into_iter()
                .map(|p| p.as_constant().unwrap())
                .collect();
            assert_eq!(j, legendre(n).unwrap(), "n = {n}");
            let g: Vec<_> = gegenbauer(n, &ParamVal::Const(r(1, 1)))
                .unwrap()
                .into_iter()
                .map(|p| p.as_constant().unwrap())
                .collect();
            assert_eq!(g, chebyshev_u(n).unwrap(), "n = {n}");
        }
    }
}
