//! Partial-fraction decomposition (`apart`) of a univariate rational function
//! over ℚ.
//!
//! Given a rational function `p(x)/q(x)` in a chosen variable `x`, [`apart`]
//! produces the decomposition
//!
//! ```text
//!   p/q = poly_part(x)  +  Σ_i Σ_{j=1}^{e_i}  A_{ij}(x) / f_i(x)^j
//! ```
//!
//! where the `f_i` are the **distinct ℚ-irreducible factors** of the denominator
//! `q = ∏_i f_i^{e_i}`, and each numerator satisfies `deg A_{ij} < deg f_i`.
//!
//! ## Algorithm (Bronstein 2005, §1.5)
//!
//! 1. Parse `expr` into `(num, den)` over ℚ and reduce to lowest terms
//!    (`gcd(num, den) = 1`).
//! 2. Polynomial part via long division: `num = quo·den + rem`, `deg rem < deg den`.
//! 3. Factor `den` over ℚ into `∏ f_i^{e_i}` (FLINT integer factorization of the
//!    primitive part).
//! 4. Split `rem/∏ f_i^{e_i}` into `Σ A_i / f_i^{e_i}` with `deg A_i < deg f_i^{e_i}`
//!    via the standard pairwise-coprime CRT (extended Euclid) partial fraction.
//! 5. Expand each `A_i / f_i^{e_i}` into `Σ_{j=1}^{e_i} A_{ij} / f_i^j` via the
//!    `f_i`-adic expansion (repeated division by `f_i`).
//!
//! ## Scope
//!
//! Decomposition is over ℚ only: irreducible quadratics (and higher-degree
//! irreducible factors) are **kept intact**, not split into complex/algebraic
//! linear factors.  The `ℚ(α)`-splitting variant is future work.
//!
//! This reuses the ℚ\[x\] polynomial machinery from the rational Risch
//! integrator (`crate::integrate::risch::{poly_rde, rational_rde}`).

use rug::{Integer, Rational};

use crate::kernel::{ExprId, ExprPool};
use crate::poly::UniPoly;

use crate::integrate::risch::poly_rde::{
    degree, poly_mul, poly_one, poly_scale, poly_zero, qpoly_to_expr, trim, QPoly,
};
use crate::integrate::risch::rational_rde::{
    expr_to_qrational, poly_div_exact, poly_divrem, poly_gcd, poly_monic, poly_sub,
};

use super::error::ConversionError;

/// Errors from [`apart`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApartError {
    /// The input is not a rational function of `var` over ℚ (e.g. it contains a
    /// transcendental generator, a foreign symbol, or a non-integer exponent).
    NotRational,
    /// The denominator is the zero polynomial.
    ZeroDenominator,
    /// FLINT factorization of the denominator failed.
    FactorizationFailed,
}

impl std::fmt::Display for ApartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApartError::NotRational => {
                write!(f, "apart: input is not a rational function of the variable")
            }
            ApartError::ZeroDenominator => write!(f, "apart: zero denominator"),
            ApartError::FactorizationFailed => {
                write!(f, "apart: denominator factorization failed")
            }
        }
    }
}

impl std::error::Error for ApartError {}

impl From<ApartError> for ConversionError {
    fn from(_: ApartError) -> Self {
        ConversionError::ZeroDenominator
    }
}

/// Partial-fraction decomposition of `expr` as a univariate rational function of
/// `var` over ℚ.
///
/// Returns an [`ExprId`] equal to the input but written as
/// `poly_part + Σ A_{ij} / f_i^j`, where the `f_i` are the distinct ℚ-irreducible
/// factors of the denominator and `deg A_{ij} < deg f_i`.
///
/// # Errors
///
/// - [`ApartError::NotRational`] if `expr` is not a rational function of `var`.
/// - [`ApartError::FactorizationFailed`] if denominator factorization fails.
///
/// # Examples
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::poly::apart;
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// // 1/(x² − 1) = 1/(2(x−1)) − 1/(2(x+1))
/// let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
/// let f = pool.pow(den, pool.integer(-1_i32));
/// let pf = apart(f, x, &pool).unwrap();
/// // pf recombines to the original; here we just check it succeeds.
/// assert!(pool.display(pf).to_string().contains('x'));
/// ```
pub fn apart(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<ExprId, ApartError> {
    let (num, den) = expr_to_qrational(expr, var, pool).ok_or(ApartError::NotRational)?;
    let num = trim(num);
    let den = trim(den);
    if den.is_empty() {
        return Err(ApartError::ZeroDenominator);
    }

    // Reduce to lowest terms and make the denominator monic. Making `den`
    // monic scales it by `1/lc`, so `num` must be scaled by the same factor
    // to preserve `num/den` (otherwise the result is off by a factor of
    // `lc`).
    let g = poly_gcd(&num, &den);
    let num = poly_div_exact(&num, &g);
    let den = poly_div_exact(&den, &g);
    let den_trimmed = trim(den.clone());
    let lc = if degree(&den_trimmed) >= 0 {
        den_trimmed[degree(&den_trimmed) as usize].clone()
    } else {
        Rational::from(1)
    };
    let num = poly_scale(&num, &(Rational::from(1) / lc));
    let den = poly_monic(&den);

    // Constant (degree-0) denominator: nothing to decompose.  After monic
    // normalisation the denominator is the constant 1, so the result is `num`.
    if degree(&den) < 1 {
        return Ok(qpoly_to_expr(&num, var, pool));
    }

    // Polynomial part: num = quo·den + rem.
    let (quo, rem) = poly_divrem(&num, &den);
    let rem = trim(rem);

    let mut terms: Vec<ExprId> = Vec::new();
    if !trim(quo.clone()).is_empty() {
        terms.push(qpoly_to_expr(&quo, var, pool));
    }

    if !rem.is_empty() {
        // Factor the (monic) denominator over ℚ: den = ∏ f_i^{e_i}.
        let factors = factor_monic_q(&den, var, pool)?; // (monic irreducible f_i, e_i)

        // Per-factor moduli f_i^{e_i} (pairwise coprime).
        let moduli: Vec<QPoly> = factors.iter().map(|(f, e)| poly_pow(f, *e)).collect();

        // Split rem/∏ moduli into Σ A_i / moduli_i  (deg A_i < deg moduli_i).
        let parts = partial_fractions(&rem, &moduli);

        for ((f, e), a_i) in factors.iter().zip(parts.iter()) {
            // f-adic expansion: A_i / f^e = Σ_{j=1}^{e} A_{ij} / f^j with
            // deg A_{ij} < deg f.
            let coeffs = factor_adic_expansion(a_i, f, *e);
            for (j, a_ij) in coeffs.iter().enumerate() {
                let a_ij = trim(a_ij.clone());
                if a_ij.is_empty() {
                    continue;
                }
                let pow = j + 1; // coeffs[0] is the coefficient of 1/f^1
                let f_expr = qpoly_to_expr(f, var, pool);
                let num_expr = qpoly_to_expr(&a_ij, var, pool);
                let denom_expr = pool.pow(f_expr, pool.integer(-(pow as i32)));
                terms.push(pool.mul(vec![num_expr, denom_expr]));
            }
        }
    }

    Ok(match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    })
}

/// `p^n` for `n ≥ 0`.
fn poly_pow(p: &QPoly, n: u32) -> QPoly {
    let mut acc = poly_one();
    for _ in 0..n {
        acc = poly_mul(&acc, p);
    }
    acc
}

/// LCM of all coefficient denominators of `p`.
fn lcm_denoms(p: &QPoly) -> Integer {
    let mut l = Integer::from(1);
    for c in p.iter() {
        l = l.lcm(c.denom());
    }
    l
}

/// Factor a monic ℚ-polynomial `d` into its **distinct** monic irreducible
/// factors over ℚ with multiplicities: `d = ∏ (f_i, e_i)` with `f_i` monic
/// irreducible and pairwise distinct.
fn factor_monic_q(
    d: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Vec<(QPoly, u32)>, ApartError> {
    // Clear denominators so FLINT sees an integer polynomial; the factorization
    // over ℚ is the same up to the rational leading unit.
    let m = lcm_denoms(d);
    let d_int = poly_scale(d, &Rational::from(m));
    let d_expr = qpoly_to_expr(&d_int, var, pool);
    let up =
        UniPoly::from_symbolic(d_expr, var, pool).map_err(|_| ApartError::FactorizationFailed)?;
    let fac = up.factor_z().map_err(|_| ApartError::FactorizationFailed)?;
    let mut factors: Vec<(QPoly, u32)> = Vec::new();
    for (f, mult) in fac.factors {
        let qp: QPoly = f
            .coefficients()
            .iter()
            .map(|c| Rational::from(c.clone()))
            .collect();
        let qp = poly_monic(&qp);
        if degree(&qp) < 1 {
            continue; // unit factor
        }
        factors.push((qp, mult));
    }
    if factors.is_empty() {
        return Err(ApartError::FactorizationFailed);
    }
    Ok(factors)
}

/// Extended Euclid over ℚ\[x\]: returns `(g, s, t)` with `s·a + t·b = g`,
/// `g = gcd(a, b)` (monic).
fn ext_gcd(a: &QPoly, b: &QPoly) -> (QPoly, QPoly, QPoly) {
    let mut r0 = trim(a.clone());
    let mut r1 = trim(b.clone());
    let mut s0 = poly_one();
    let mut s1 = poly_zero();
    let mut t0 = poly_zero();
    let mut t1 = poly_one();
    while !r1.is_empty() {
        let (q, r) = poly_divrem(&r0, &r1);
        let new_s = poly_sub(&s0, &poly_mul(&q, &s1));
        let new_t = poly_sub(&t0, &poly_mul(&q, &t1));
        r0 = r1;
        r1 = r;
        s0 = s1;
        s1 = new_s;
        t0 = t1;
        t1 = new_t;
    }
    // Normalise to a monic gcd.
    let d = degree(&r0);
    if d >= 0 {
        let inv = Rational::from(1) / r0[d as usize].clone();
        r0 = poly_scale(&r0, &inv);
        s0 = poly_scale(&s0, &inv);
        t0 = poly_scale(&t0, &inv);
    }
    (r0, s0, t0)
}

/// Reduce `a mod m` (the remainder of `a / m`).
fn poly_mod(a: &QPoly, m: &QPoly) -> QPoly {
    let (_, r) = poly_divrem(a, m);
    trim(r)
}

/// Partial-fraction split over pairwise-coprime moduli: returns `A_i` with
/// `num / ∏ m_i = Σ A_i / m_i` and `deg A_i < deg m_i`.
///
/// `num` must satisfy `deg num < Σ deg m_i` for a strictly-proper result; the
/// caller guarantees this by peeling off the polynomial part first.
fn partial_fractions(num: &QPoly, moduli: &[QPoly]) -> Vec<QPoly> {
    let n = moduli.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![poly_mod(num, &moduli[0])];
    }
    let mut result = Vec::with_capacity(n);
    let mut cur = trim(num.clone());
    for i in 0..n - 1 {
        let mi = &moduli[i];
        let rest = moduli[i + 1..]
            .iter()
            .fold(poly_one(), |acc, m| poly_mul(&acc, m));
        // mi and rest are coprime; ext_gcd gives s·mi + t·rest = 1, so
        // A_i ≡ cur·t (mod mi).
        let (_g, _s, t) = ext_gcd(mi, &rest);
        let ai = poly_mod(&poly_mul(&cur, &t), mi);
        // Subtract A_i·rest and divide by mi to continue with the remaining
        // moduli: cur ← (cur − A_i·rest) / mi.
        let next = poly_div_exact(&poly_sub(&cur, &poly_mul(&ai, &rest)), mi);
        result.push(ai);
        cur = next;
    }
    result.push(cur);
    result
}

/// `f`-adic expansion of `a / f^e`: returns coefficients `[A_1, A_2, …, A_e]`
/// (each with `deg < deg f`) such that
/// `a / f^e = Σ_{j=1}^{e} A_j / f^j`, i.e. `a = Σ_{j=1}^{e} A_j · f^{e-j}`.
///
/// Computed by repeated division by `f`: write `a` in base `f` as
/// `a = c_0 + c_1·f + … + c_{e-1}·f^{e-1}` (deg c_k < deg f), then
/// `A_j = c_{e-j}`.
fn factor_adic_expansion(a: &QPoly, f: &QPoly, e: u32) -> Vec<QPoly> {
    let mut digits: Vec<QPoly> = Vec::with_capacity(e as usize);
    let mut cur = trim(a.clone());
    for _ in 0..e {
        let (q, r) = poly_divrem(&cur, f);
        digits.push(trim(r)); // c_0, c_1, …
        cur = trim(q);
    }
    // A_j = c_{e-j}  ⇒  reverse the digit list so index 0 is the 1/f^1 term.
    digits.reverse();
    digits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprData, ExprPool};

    fn pool() -> (ExprPool, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        (p, x)
    }

    /// Numeric evaluator for verification (Integer/Rational/Add/Mul/Pow only).
    fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args.iter().map(|&a| eval(a, x, xv, pool)).product(),
            ExprData::Pow { base, exp } => {
                let b = eval(base, x, xv, pool);
                if let ExprData::Integer(n) = pool.get(exp) {
                    if let Some(k) = n.0.to_i32() {
                        return b.powi(k);
                    }
                }
                b.powf(eval(exp, x, xv, pool))
            }
            other => panic!("eval: unsupported {other:?}"),
        }
    }

    /// Assert `apart(f)` equals `f` numerically at several non-pole points.
    fn assert_equiv(f: ExprId, pf: ExprId, x: ExprId, pool: &ExprPool) {
        for &xv in &[1.7_f64, 2.3, 3.9, -2.5, 5.1] {
            let lhs = eval(f, x, xv, pool);
            let rhs = eval(pf, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "apart ≠ input at x={xv}: {lhs} vs {rhs} (pf = {})",
                pool.display(pf)
            );
        }
    }

    #[test]
    fn one_over_x2_minus_1() {
        // 1/(x²−1) = 1/(2(x−1)) − 1/(2(x+1)).
        let (p, x) = pool();
        let den = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(-1_i32)]);
        let f = p.pow(den, p.integer(-1_i32));
        let pf = apart(f, x, &p).unwrap();
        assert_equiv(f, pf, x, &p);
    }

    #[test]
    fn improper_x3_over_x2_minus_1() {
        // x³/(x²−1) = x + 1/(2(x−1)) + 1/(2(x+1)).
        let (p, x) = pool();
        let den = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(-1_i32)]);
        let f = p.mul(vec![
            p.pow(x, p.integer(3_i32)),
            p.pow(den, p.integer(-1_i32)),
        ]);
        let pf = apart(f, x, &p).unwrap();
        assert_equiv(f, pf, x, &p);
        // Must contain a bare polynomial part (the `x` term).
        let s = p.display(pf).to_string();
        assert!(s.contains('x'), "expected a polynomial part: {s}");
    }

    #[test]
    fn repeated_factor() {
        // (x+1)/((x−1)²(x+2)).
        let (p, x) = pool();
        let xm1 = p.add(vec![x, p.integer(-1_i32)]);
        let xp2 = p.add(vec![x, p.integer(2_i32)]);
        let den = p.mul(vec![p.pow(xm1, p.integer(2_i32)), xp2]);
        let num = p.add(vec![x, p.integer(1_i32)]);
        let f = p.mul(vec![num, p.pow(den, p.integer(-1_i32))]);
        let pf = apart(f, x, &p).unwrap();
        assert_equiv(f, pf, x, &p);
    }

    #[test]
    fn irreducible_quadratic_kept() {
        // 1/((x−1)(x²+1)) — the quadratic stays intact over ℚ.
        let (p, x) = pool();
        let xm1 = p.add(vec![x, p.integer(-1_i32)]);
        let x2p1 = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(1_i32)]);
        let den = p.mul(vec![xm1, x2p1]);
        let f = p.pow(den, p.integer(-1_i32));
        let pf = apart(f, x, &p).unwrap();
        assert_equiv(f, pf, x, &p);
        // The result must not introduce sqrt/i — quadratic stays over ℚ.
        let s = p.display(pf).to_string();
        assert!(!s.contains("sqrt"), "should not split quadratic: {s}");
    }

    #[test]
    fn high_multiplicity() {
        // x/(x−1)³ = 1/(x−1)² + 1/(x−1)³.
        let (p, x) = pool();
        let xm1 = p.add(vec![x, p.integer(-1_i32)]);
        let f = p.mul(vec![x, p.pow(xm1, p.integer(-3_i32))]);
        let pf = apart(f, x, &p).unwrap();
        assert_equiv(f, pf, x, &p);
    }

    #[test]
    fn already_polynomial() {
        // x² + 1 has no proper part; apart returns it as-is.
        let (p, x) = pool();
        let f = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(1_i32)]);
        let pf = apart(f, x, &p).unwrap();
        assert_equiv(f, pf, x, &p);
    }

    #[test]
    fn not_rational_errors() {
        // exp(x)/(x²−1) is not a rational function of x.
        let (p, x) = pool();
        let den = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(-1_i32)]);
        let f = p.mul(vec![p.func("exp", vec![x]), p.pow(den, p.integer(-1_i32))]);
        assert_eq!(apart(f, x, &p), Err(ApartError::NotRational));
    }
}
