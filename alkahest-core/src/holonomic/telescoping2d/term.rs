//! Recognising *proper hypergeometric terms* `F(n, j, k)` in three affine
//! indices, and computing their exact shift ratios.
//!
//! This is [`super::super::hyperterm`] generalized from two indices to three:
//!
//! ```text
//! F(n, j, k) = R(n,j,k) · z_j^j · z_k^k · w^n · ∏_i Γ(a_i·n + b_i·j + c_i·k + d_i)^(e_i)
//! ```
//!
//! with `R ∈ Q(n,j,k)`, `z_j, z_k, w ∈ Q \ {0}`, `a_i, b_i, c_i ∈ Z`,
//! `d_i ∈ Q`, `e_i ∈ Z`. This is exactly the class for which the three shift
//! quotients `F(n+1,j,k)/F(n,j,k)`, `F(n,j+1,k)/F(n,j,k)` and
//! `F(n,j,k+1)/F(n,j,k)` — and, more generally, any `F(·+i,·,·)/F(·,·,·)` in a
//! single index — are rational functions in `Q(n,j,k)`, which is what the
//! double-sum ansatz search in [`super::search`] consumes.
//!
//! The parser deliberately reuses the supported-function-head set of
//! [`super::super::hyperterm`] (`gamma`, `factorial`, `binomial`,
//! `pochhammer`) and its strictness: anything outside the class above is
//! refused rather than approximated.

use super::poly::{Axis, Poly3, Rat3};
use super::Telescoping2dError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Rational;

/// One `Γ(a·n + b·j + c·k + d)^e` factor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GammaFactor3 {
    pub a: i64,
    pub b: i64,
    pub c: i64,
    pub d: Rational,
    pub e: i32,
}

fn gamma_arg_poly3(g: &GammaFactor3) -> Poly3 {
    Poly3::var(Axis::N)
        .scale(&Rational::from(g.a))
        .add(&Poly3::var(Axis::J).scale(&Rational::from(g.b)))
        .add(&Poly3::var(Axis::K).scale(&Rational::from(g.c)))
        .add(&Poly3::constant(g.d.clone()))
}

/// A parsed proper hypergeometric term `F(n, j, k)`.
#[derive(Clone, Debug)]
pub struct ProperTerm3 {
    pub rat: Rat3,
    pub zj: Rational,
    pub zk: Rational,
    pub w: Rational,
    pub gammas: Vec<GammaFactor3>,
}

const MAX_POW: i32 = 32;
const MAX_PARSE_DEPTH: usize = 64;
const MAX_FOLD_BITS: u32 = 4096;

impl ProperTerm3 {
    fn one() -> Self {
        ProperTerm3 {
            rat: Rat3::one(),
            zj: Rational::from(1),
            zk: Rational::from(1),
            w: Rational::from(1),
            gammas: Vec::new(),
        }
    }

    fn from_rat3(r: Rat3) -> Self {
        ProperTerm3 {
            rat: r,
            zj: Rational::from(1),
            zk: Rational::from(1),
            w: Rational::from(1),
            gammas: Vec::new(),
        }
    }

    fn mul(&self, other: &ProperTerm3) -> ProperTerm3 {
        let mut gammas = self.gammas.clone();
        gammas.extend(other.gammas.iter().cloned());
        ProperTerm3 {
            rat: self.rat.mul(&other.rat),
            zj: self.zj.clone() * other.zj.clone(),
            zk: self.zk.clone() * other.zk.clone(),
            w: self.w.clone() * other.w.clone(),
            gammas,
        }
    }

    fn pow(&self, e: i32) -> Option<ProperTerm3> {
        if e.unsigned_abs() > MAX_POW as u32 {
            return None;
        }
        let gammas = self
            .gammas
            .iter()
            .map(|g| {
                Some(GammaFactor3 {
                    a: g.a,
                    b: g.b,
                    c: g.c,
                    d: g.d.clone(),
                    e: g.e.checked_mul(e)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(ProperTerm3 {
            rat: self.rat.pow_i32(e)?,
            zj: rat_pow(&self.zj, e)?,
            zk: rat_pow(&self.zk, e)?,
            w: rat_pow(&self.w, e)?,
            gammas,
        })
    }

    /// `F(·+i·axis, ·, ·) / F(·,·,·)` — the shift quotient in a single index,
    /// as an exact element of `Q(n,j,k)`.
    pub fn ratio_axis(&self, axis: Axis, i: i64) -> Result<Rat3, Telescoping2dError> {
        if i == 0 {
            return Ok(Rat3::one());
        }
        let shifted = self.rat.shift(axis, i);
        let mut acc = shifted.div(&self.rat).ok_or_else(|| {
            Telescoping2dError::NotProperHypergeometric("term vanishes identically".into())
        })?;
        let base = match axis {
            Axis::N => &self.w,
            Axis::J => &self.zj,
            Axis::K => &self.zk,
        };
        acc = acc.mul(&Rat3::from_rational(rat_pow_i64(base, i)?));
        for g in &self.gammas {
            let arg = gamma_arg_poly3(g);
            let coeff = match axis {
                Axis::N => g.a,
                Axis::J => g.b,
                Axis::K => g.c,
            };
            let shift = coeff.checked_mul(i).ok_or_else(|| {
                Telescoping2dError::SearchExhausted("gamma shift overflow".into())
            })?;
            let step = gamma_shift_ratio3(&arg, shift)?;
            acc = acc.mul(&step.pow_i32(g.e).ok_or_else(|| {
                Telescoping2dError::NotProperHypergeometric(
                    "gamma factor is identically zero".into(),
                )
            })?);
        }
        Ok(acc)
    }

    pub fn parse(
        expr: ExprId,
        n: ExprId,
        j: ExprId,
        k: ExprId,
        pool: &ExprPool,
    ) -> Result<ProperTerm3, Telescoping2dError> {
        parse_rec(expr, n, j, k, pool, 0)
    }
}

/// `Γ(x + s) / Γ(x)` for integer `s`, as an exact rational function.
fn gamma_shift_ratio3(x: &Poly3, s: i64) -> Result<Rat3, Telescoping2dError> {
    if s == 0 {
        return Ok(Rat3::one());
    }
    if s.unsigned_abs() > 512 {
        return Err(Telescoping2dError::SearchExhausted(format!(
            "gamma argument shift {s} exceeds the supported limit of 512"
        )));
    }
    let mut prod = Poly3::one();
    if s > 0 {
        for t in 0..s {
            prod = prod.mul(&x.add(&Poly3::from_i64(t)));
        }
        Ok(Rat3::from_poly(prod))
    } else {
        for t in 1..=(-s) {
            prod = prod.mul(&x.add(&Poly3::from_i64(-t)));
        }
        Rat3::from_poly(prod)
            .inv()
            .ok_or_else(|| Telescoping2dError::NotProperHypergeometric("gamma pole".into()))
    }
}

fn rat_pow(q: &Rational, e: i32) -> Option<Rational> {
    rat_pow_i64(q, e as i64).ok()
}

fn rat_pow_i64(q: &Rational, e: i64) -> Result<Rational, Telescoping2dError> {
    if e == 0 {
        return Ok(Rational::from(1));
    }
    if e.unsigned_abs() > 1024 {
        return Err(Telescoping2dError::SearchExhausted(
            "exponential factor exponent exceeds the supported limit".into(),
        ));
    }
    if *q == 0 {
        if e < 0 {
            return Err(Telescoping2dError::NotProperHypergeometric(
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

fn parse_rec(
    expr: ExprId,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ProperTerm3, Telescoping2dError> {
    if depth > MAX_PARSE_DEPTH {
        return Err(Telescoping2dError::NotProperHypergeometric(
            "expression nests deeper than the parser supports".into(),
        ));
    }
    if let Some(r) = as_rat3(expr, n, j, k, pool, 0) {
        return Ok(ProperTerm3::from_rat3(r));
    }
    match pool.get(expr) {
        ExprData::Mul(args) => {
            let mut acc = ProperTerm3::one();
            for a in args {
                acc = acc.mul(&parse_rec(a, n, j, k, pool, depth + 1)?);
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => {
            if let Some(e) = as_i32(exp, pool) {
                let b = parse_rec(base, n, j, k, pool, depth + 1)?;
                return b.pow(e).ok_or_else(|| {
                    Telescoping2dError::NotProperHypergeometric(format!(
                        "exponent {e} is outside the supported range (|e| <= {MAX_POW})"
                    ))
                });
            }
            let Some(c) = as_rational_folded(base, pool, 0) else {
                return Err(Telescoping2dError::NotProperHypergeometric(format!(
                    "power with symbolic exponent needs a rational base, got {}",
                    pool.display(base)
                )));
            };
            if c == 0 {
                return Err(Telescoping2dError::NotProperHypergeometric(
                    "0 raised to a symbolic power".into(),
                ));
            }
            let (alpha, beta, delta, gamma_c) =
                affine_parts3(exp, n, j, k, pool).ok_or_else(|| {
                    Telescoping2dError::NotProperHypergeometric(format!(
                        "exponent {} is not integer-affine in the three indices",
                        pool.display(exp)
                    ))
                })?;
            if *gamma_c.clone().denom() != 1 {
                return Err(Telescoping2dError::NotProperHypergeometric(
                    "constant part of an exponential exponent must be an integer".into(),
                ));
            }
            let gi: i64 = gamma_c
                .numer()
                .to_i64()
                .ok_or_else(|| Telescoping2dError::SearchExhausted("exponent too large".into()))?;
            Ok(ProperTerm3 {
                rat: Rat3::from_rational(rat_pow_i64(&c, gi)?),
                zj: rat_pow_i64(&c, beta)?,
                zk: rat_pow_i64(&c, delta)?,
                w: rat_pow_i64(&c, alpha)?,
                gammas: Vec::new(),
            })
        }
        ExprData::Func { name, args } => parse_func(&name, &args, n, j, k, pool),
        other => Err(Telescoping2dError::NotProperHypergeometric(format!(
            "unsupported node {other:?} in {}",
            pool.display(expr)
        ))),
    }
}

fn parse_func(
    name: &str,
    args: &[ExprId],
    n: ExprId,
    j: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Result<ProperTerm3, Telescoping2dError> {
    let gamma_of = |arg: ExprId, e: i32| -> Result<GammaFactor3, Telescoping2dError> {
        let (a, b, c, d) = affine_parts3(arg, n, j, k, pool).ok_or_else(|| {
            Telescoping2dError::NotProperHypergeometric(format!(
                "gamma argument {} is not integer-affine in the three indices",
                pool.display(arg)
            ))
        })?;
        Ok(GammaFactor3 { a, b, c, d, e })
    };
    let one_plus = |arg: ExprId| -> ExprId { pool.add(vec![arg, pool.integer(1_i32)]) };
    match (name, args.len()) {
        ("gamma", 1) => Ok(ProperTerm3 {
            rat: Rat3::one(),
            zj: Rational::from(1),
            zk: Rational::from(1),
            w: Rational::from(1),
            gammas: vec![gamma_of(args[0], 1)?],
        }),
        ("factorial", 1) => Ok(ProperTerm3 {
            rat: Rat3::one(),
            zj: Rational::from(1),
            zk: Rational::from(1),
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
            Ok(ProperTerm3 {
                rat: Rat3::one(),
                zj: Rational::from(1),
                zk: Rational::from(1),
                w: Rational::from(1),
                gammas: vec![gamma_of(top, 1)?, gamma_of(bot, -1)?, gamma_of(rest, -1)?],
            })
        }
        ("pochhammer", 2) => {
            let sum = pool.add(vec![args[0], args[1]]);
            Ok(ProperTerm3 {
                rat: Rat3::one(),
                zj: Rational::from(1),
                zk: Rational::from(1),
                w: Rational::from(1),
                gammas: vec![gamma_of(sum, 1)?, gamma_of(args[0], -1)?],
            })
        }
        _ => Err(Telescoping2dError::NotProperHypergeometric(format!(
            "function `{name}/{}` is not part of the proper hypergeometric class \
             (supported: gamma, factorial, binomial, pochhammer)",
            args.len()
        ))),
    }
}

/// Evaluate an expression inside the field `Q(n, j, k)`, or `None` if it
/// leaves it.
pub fn as_rat3(
    expr: ExprId,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Option<Rat3> {
    if depth > MAX_PARSE_DEPTH {
        return None;
    }
    if expr == n {
        return Some(Rat3::from_poly(Poly3::var(Axis::N)));
    }
    if expr == j {
        return Some(Rat3::from_poly(Poly3::var(Axis::J)));
    }
    if expr == k {
        return Some(Rat3::from_poly(Poly3::var(Axis::K)));
    }
    match pool.get(expr) {
        ExprData::Integer(i) => Some(Rat3::from_rational(Rational::from(i.0.clone()))),
        ExprData::Rational(r) => Some(Rat3::from_rational(r.0.clone())),
        ExprData::Add(args) => {
            let mut acc = Rat3::zero();
            for a in args {
                acc = acc.add(&as_rat3(a, n, j, k, pool, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = Rat3::one();
            for a in args {
                acc = acc.mul(&as_rat3(a, n, j, k, pool, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let e = as_i32(exp, pool)?;
            if e.unsigned_abs() > MAX_POW as u32 {
                return None;
            }
            as_rat3(base, n, j, k, pool, depth + 1)?.pow_i32(e)
        }
        _ => None,
    }
}

fn fold_fits(q: &Rational) -> bool {
    q.numer().significant_bits() <= MAX_FOLD_BITS && q.denom().significant_bits() <= MAX_FOLD_BITS
}

fn as_rational_folded(expr: ExprId, pool: &ExprPool, depth: usize) -> Option<Rational> {
    if depth > MAX_PARSE_DEPTH {
        return None;
    }
    match pool.get(expr) {
        ExprData::Integer(i) => Some(Rational::from(i.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        ExprData::Add(args) => {
            let mut acc = Rational::from(0);
            for a in args {
                acc += as_rational_folded(a, pool, depth + 1)?;
                if !fold_fits(&acc) {
                    return None;
                }
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = Rational::from(1);
            for a in args {
                acc *= as_rational_folded(a, pool, depth + 1)?;
                if !fold_fits(&acc) {
                    return None;
                }
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let e = as_i32(exp, pool)?;
            if e.unsigned_abs() > MAX_POW as u32 {
                return None;
            }
            let b = as_rational_folded(base, pool, depth + 1)?;
            let bits = b
                .numer()
                .significant_bits()
                .max(b.denom().significant_bits());
            if bits.checked_mul(e.unsigned_abs())? > MAX_FOLD_BITS {
                return None;
            }
            let r = rat_pow(&b, e)?;
            fold_fits(&r).then_some(r)
        }
        _ => None,
    }
}

fn as_i32(expr: ExprId, pool: &ExprPool) -> Option<i32> {
    match pool.get(expr) {
        ExprData::Integer(i) => i.0.to_i32(),
        _ => None,
    }
}

/// Decompose an expression as `a·n + b·j + c·k + d` with `a, b, c ∈ Z` and
/// `d ∈ Q`.
pub fn affine_parts3(
    expr: ExprId,
    n: ExprId,
    j: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Option<(i64, i64, i64, Rational)> {
    let r = as_rat3(expr, n, j, k, pool, 0)?;
    let den_c = r.den.as_constant()?;
    if den_c == 0 {
        return None;
    }
    let inv = Rational::from(1) / den_c;
    let num = r.num.scale(&inv);
    // Total degree at most 1: every stored exponent triple sums to <= 1.
    if num.terms.keys().any(|(en, ej, ek)| en + ej + ek > 1) {
        return None;
    }
    let coeff_of = |axis_e: (u32, u32, u32)| -> Rational {
        num.terms
            .get(&axis_e)
            .cloned()
            .unwrap_or_else(|| Rational::from(0))
    };
    let a = coeff_of((1, 0, 0));
    let b = coeff_of((0, 1, 0));
    let c = coeff_of((0, 0, 1));
    let d = coeff_of((0, 0, 0));
    if *a.clone().denom() != 1 || *b.clone().denom() != 1 || *c.clone().denom() != 1 {
        return None;
    }
    let a_i = a.numer().to_i64()?;
    let b_i = b.numer().to_i64()?;
    let c_i = c.numer().to_i64()?;
    Some((a_i, b_i, c_i, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn njk(pool: &ExprPool) -> (ExprId, ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("j", Domain::Real),
            pool.symbol("k", Domain::Real),
        )
    }

    fn binom(pool: &ExprPool, top: ExprId, bot: ExprId) -> ExprId {
        pool.func("binomial", vec![top, bot])
    }

    /// `F(n,j,k) = C(n,j)*C(j,k)`: a genuine proper hypergeometric term in
    /// three indices, and the base of the non-separable worked example in
    /// `search.rs`/`mod.rs`.
    #[test]
    fn double_binomial_product_parses_and_has_exact_ratios() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let f = pool.mul(vec![binom(&pool, n, j), binom(&pool, j, k)]);
        let term = ProperTerm3::parse(f, n, j, k, &pool).expect("proper hypergeometric");

        // ratio_j = F(n,j+1,k)/F(n,j,k) should equal
        // (n-j)/(j+1) * (j+1)!/(j+1-k)! / (j!/(j-k)!) as a rational function;
        // check it numerically at a sample point instead of re-deriving the
        // closed form, since that *is* what the exact machinery computes.
        let rj = term.ratio_axis(Axis::J, 1).expect("ratio_j");
        let (nn, jj, kk) = (Rational::from(6), Rational::from(3), Rational::from(2));
        let got = rj.num.eval(&nn, &jj, &kk) / rj.den.eval(&nn, &jj, &kk);
        // C(6,4)*C(4,2) / (C(6,3)*C(3,2)) = 15*6 / (20*3) = 90/60 = 3/2
        assert_eq!(got, Rational::from((3, 2)));
    }

    #[test]
    fn refuses_non_hypergeometric_input() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let bad = pool.func("sin", vec![pool.mul(vec![n, j, k])]);
        let err = ProperTerm3::parse(bad, n, j, k, &pool).expect_err("not hypergeometric");
        assert!(matches!(
            err,
            Telescoping2dError::NotProperHypergeometric(_)
        ));
    }
}
