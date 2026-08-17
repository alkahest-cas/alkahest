//! Recognising *proper hypergeometric terms* `F(n, x_1, …, x_m)` in `m + 1`
//! affine indices, for an arbitrary number `m ≥ 1` of bound indices, and
//! computing their exact shift ratios.
//!
//! This is [`super::super::hyperterm`] generalized from one bound index to
//! `m`:
//!
//! ```text
//! F(n, x_1, …, x_m) = R(n,x) · ∏_t z_t^{x_t} · w^n · ∏_i Γ(a_i·n + Σ_t b_{i,t}·x_t + d_i)^(e_i)
//! ```
//!
//! with `R ∈ Q(n,x)`, `z_t, w ∈ Q \ {0}`, `a_i, b_{i,t} ∈ Z`, `d_i ∈ Q`,
//! `e_i ∈ Z`. This is exactly the class for which every shift quotient
//! `F(·+1 in one axis, ·) / F(·)` — and, more generally, any
//! `F(n+i,x)/F(n,x)` — is a rational function in `Q(n,x_1,…,x_m)`, which is
//! what the ansatz search in [`super::search`] consumes.
//!
//! The parser deliberately reuses the supported-function-head set of
//! [`super::super::hyperterm`] (`gamma`, `factorial`, `binomial`,
//! `pochhammer`) and its strictness: anything outside the class above is
//! refused rather than approximated.

use super::poly::{Axis, PolyM, RatM, AXIS_N};
use super::Telescoping2dError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Rational;

/// One `Γ(a·n + Σ_t b_t·x_t + d)^e` factor. `coeffs[0]` is the coefficient of
/// `n`; `coeffs[1..]` are the coefficients of the `m` bound indices, in order
/// — length `m + 1`, matching [`super::poly`]'s axis convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GammaFactorM {
    pub coeffs: Vec<i64>,
    pub d: Rational,
    pub e: i32,
}

fn gamma_arg_polym(g: &GammaFactorM, num_axes: usize) -> PolyM {
    let mut p = PolyM::constant(g.d.clone(), num_axes);
    for (axis, &c) in g.coeffs.iter().enumerate() {
        if c != 0 {
            p = p.add(&PolyM::var(axis, num_axes).scale(&Rational::from(c)));
        }
    }
    p
}

/// A parsed proper hypergeometric term `F(n, x_1, …, x_m)`.
#[derive(Clone, Debug)]
pub struct ProperTermM {
    /// Number of bound indices.
    pub m: usize,
    pub rat: RatM,
    pub w: Rational,
    /// `z_t` for `t = 0..m`, one per bound index (in caller order).
    pub z: Vec<Rational>,
    pub gammas: Vec<GammaFactorM>,
}

const MAX_POW: i32 = 32;
const MAX_PARSE_DEPTH: usize = 64;
const MAX_FOLD_BITS: u32 = 4096;

impl ProperTermM {
    fn num_axes(&self) -> usize {
        self.m + 1
    }

    fn one(m: usize) -> Self {
        ProperTermM {
            m,
            rat: RatM::one(m + 1),
            w: Rational::from(1),
            z: vec![Rational::from(1); m],
            gammas: Vec::new(),
        }
    }

    fn from_ratm(r: RatM, m: usize) -> Self {
        ProperTermM {
            m,
            rat: r,
            w: Rational::from(1),
            z: vec![Rational::from(1); m],
            gammas: Vec::new(),
        }
    }

    fn mul(&self, other: &ProperTermM) -> ProperTermM {
        debug_assert_eq!(self.m, other.m);
        let mut gammas = self.gammas.clone();
        gammas.extend(other.gammas.iter().cloned());
        let z = self
            .z
            .iter()
            .zip(other.z.iter())
            .map(|(a, b)| a.clone() * b.clone())
            .collect();
        ProperTermM {
            m: self.m,
            rat: self.rat.mul(&other.rat),
            w: self.w.clone() * other.w.clone(),
            z,
            gammas,
        }
    }

    fn pow(&self, e: i32) -> Option<ProperTermM> {
        if e.unsigned_abs() > MAX_POW as u32 {
            return None;
        }
        let gammas = self
            .gammas
            .iter()
            .map(|g| {
                Some(GammaFactorM {
                    coeffs: g.coeffs.clone(),
                    d: g.d.clone(),
                    e: g.e.checked_mul(e)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let z = self
            .z
            .iter()
            .map(|zt| rat_pow(zt, e))
            .collect::<Option<Vec<_>>>()?;
        Some(ProperTermM {
            m: self.m,
            rat: self.rat.pow_i32(e)?,
            w: rat_pow(&self.w, e)?,
            z,
            gammas,
        })
    }

    /// `F(·+i·axis, ·) / F(·)` — the shift quotient in a single axis (`0` for
    /// `n`, `1..=m` for a bound index), as an exact element of
    /// `Q(n,x_1,…,x_m)`.
    pub fn ratio_axis(&self, axis: Axis, i: i64) -> Result<RatM, Telescoping2dError> {
        if i == 0 {
            return Ok(RatM::one(self.num_axes()));
        }
        let shifted = self.rat.shift(axis, i);
        let mut acc = shifted.div(&self.rat).ok_or_else(|| {
            Telescoping2dError::NotProperHypergeometric("term vanishes identically".into())
        })?;
        let base = if axis == AXIS_N {
            &self.w
        } else {
            &self.z[axis - 1]
        };
        acc = acc.mul(&RatM::from_rational(rat_pow_i64(base, i)?, self.num_axes()));
        for g in &self.gammas {
            let arg = gamma_arg_polym(g, self.num_axes());
            let coeff = g.coeffs[axis];
            let shift = coeff.checked_mul(i).ok_or_else(|| {
                Telescoping2dError::SearchExhausted("gamma shift overflow".into())
            })?;
            let step = gamma_shift_ratiom(&arg, shift, self.num_axes())?;
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
        indices: &[ExprId],
        pool: &ExprPool,
    ) -> Result<ProperTermM, Telescoping2dError> {
        if indices.is_empty() {
            return Err(Telescoping2dError::InvalidInput(
                "at least one bound index is required".into(),
            ));
        }
        parse_rec(expr, n, indices, pool, 0)
    }
}

/// `Γ(x + s) / Γ(x)` for integer `s`, as an exact rational function.
fn gamma_shift_ratiom(x: &PolyM, s: i64, num_axes: usize) -> Result<RatM, Telescoping2dError> {
    if s == 0 {
        return Ok(RatM::one(num_axes));
    }
    if s.unsigned_abs() > 512 {
        return Err(Telescoping2dError::SearchExhausted(format!(
            "gamma argument shift {s} exceeds the supported limit of 512"
        )));
    }
    let mut prod = PolyM::one(num_axes);
    if s > 0 {
        for t in 0..s {
            prod = prod.mul(&x.add(&PolyM::from_i64(t, num_axes)));
        }
        Ok(RatM::from_poly(prod, num_axes))
    } else {
        for t in 1..=(-s) {
            prod = prod.mul(&x.add(&PolyM::from_i64(-t, num_axes)));
        }
        RatM::from_poly(prod, num_axes)
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
    indices: &[ExprId],
    pool: &ExprPool,
    depth: usize,
) -> Result<ProperTermM, Telescoping2dError> {
    let m = indices.len();
    if depth > MAX_PARSE_DEPTH {
        return Err(Telescoping2dError::NotProperHypergeometric(
            "expression nests deeper than the parser supports".into(),
        ));
    }
    if let Some(r) = as_ratm(expr, n, indices, pool, 0) {
        return Ok(ProperTermM::from_ratm(r, m));
    }
    match pool.get(expr) {
        ExprData::Mul(args) => {
            let mut acc = ProperTermM::one(m);
            for a in args {
                acc = acc.mul(&parse_rec(a, n, indices, pool, depth + 1)?);
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => {
            if let Some(e) = as_i32(exp, pool) {
                let b = parse_rec(base, n, indices, pool, depth + 1)?;
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
            let (coeffs, gamma_c) = affine_partsm(exp, n, indices, pool).ok_or_else(|| {
                Telescoping2dError::NotProperHypergeometric(format!(
                    "exponent {} is not integer-affine in the bound indices",
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
            let z = coeffs[1..]
                .iter()
                .map(|&bt| rat_pow_i64(&c, bt))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ProperTermM {
                m,
                rat: RatM::from_rational(rat_pow_i64(&c, gi)?, m + 1),
                w: rat_pow_i64(&c, coeffs[0])?,
                z,
                gammas: Vec::new(),
            })
        }
        ExprData::Func { name, args } => parse_func(&name, &args, n, indices, pool),
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
    indices: &[ExprId],
    pool: &ExprPool,
) -> Result<ProperTermM, Telescoping2dError> {
    let m = indices.len();
    let gamma_of = |arg: ExprId, e: i32| -> Result<GammaFactorM, Telescoping2dError> {
        let (coeffs, d) = affine_partsm(arg, n, indices, pool).ok_or_else(|| {
            Telescoping2dError::NotProperHypergeometric(format!(
                "gamma argument {} is not integer-affine in the bound indices",
                pool.display(arg)
            ))
        })?;
        Ok(GammaFactorM { coeffs, d, e })
    };
    let one_plus = |arg: ExprId| -> ExprId { pool.add(vec![arg, pool.integer(1_i32)]) };
    let plain = |gammas: Vec<GammaFactorM>| ProperTermM {
        m,
        rat: RatM::one(m + 1),
        w: Rational::from(1),
        z: vec![Rational::from(1); m],
        gammas,
    };
    match (name, args.len()) {
        ("gamma", 1) => Ok(plain(vec![gamma_of(args[0], 1)?])),
        ("factorial", 1) => Ok(plain(vec![gamma_of(one_plus(args[0]), 1)?])),
        ("binomial", 2) => {
            let top = one_plus(args[0]);
            let bot = one_plus(args[1]);
            let rest = pool.add(vec![
                args[0],
                pool.mul(vec![args[1], pool.integer(-1_i32)]),
                pool.integer(1_i32),
            ]);
            Ok(plain(vec![
                gamma_of(top, 1)?,
                gamma_of(bot, -1)?,
                gamma_of(rest, -1)?,
            ]))
        }
        ("pochhammer", 2) => {
            let sum = pool.add(vec![args[0], args[1]]);
            Ok(plain(vec![gamma_of(sum, 1)?, gamma_of(args[0], -1)?]))
        }
        _ => Err(Telescoping2dError::NotProperHypergeometric(format!(
            "function `{name}/{}` is not part of the proper hypergeometric class \
             (supported: gamma, factorial, binomial, pochhammer)",
            args.len()
        ))),
    }
}

/// Evaluate an expression inside the field `Q(n, x_1, …, x_m)`, or `None` if
/// it leaves it.
pub fn as_ratm(
    expr: ExprId,
    n: ExprId,
    indices: &[ExprId],
    pool: &ExprPool,
    depth: usize,
) -> Option<RatM> {
    if depth > MAX_PARSE_DEPTH {
        return None;
    }
    let num_axes = indices.len() + 1;
    if expr == n {
        return Some(RatM::from_poly(PolyM::var(AXIS_N, num_axes), num_axes));
    }
    for (t, &idx_expr) in indices.iter().enumerate() {
        if expr == idx_expr {
            return Some(RatM::from_poly(PolyM::var(t + 1, num_axes), num_axes));
        }
    }
    match pool.get(expr) {
        ExprData::Integer(i) => Some(RatM::from_rational(Rational::from(i.0.clone()), num_axes)),
        ExprData::Rational(r) => Some(RatM::from_rational(r.0.clone(), num_axes)),
        ExprData::Add(args) => {
            let mut acc = RatM::from_rational(Rational::from(0), num_axes);
            for a in args {
                acc = acc.add(&as_ratm(a, n, indices, pool, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = RatM::one(num_axes);
            for a in args {
                acc = acc.mul(&as_ratm(a, n, indices, pool, depth + 1)?);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let e = as_i32(exp, pool)?;
            if e.unsigned_abs() > MAX_POW as u32 {
                return None;
            }
            as_ratm(base, n, indices, pool, depth + 1)?.pow_i32(e)
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

/// Decompose an expression as `a·n + Σ_t b_t·x_t + d` with `a, b_t ∈ Z` and
/// `d ∈ Q`. Returns `(coeffs, d)` with `coeffs[0] = a`, `coeffs[1..] = b_t`.
pub fn affine_partsm(
    expr: ExprId,
    n: ExprId,
    indices: &[ExprId],
    pool: &ExprPool,
) -> Option<(Vec<i64>, Rational)> {
    let r = as_ratm(expr, n, indices, pool, 0)?;
    let den_c = r.den.as_constant()?;
    if den_c == 0 {
        return None;
    }
    let inv = Rational::from(1) / den_c;
    let num = r.num.scale(&inv);
    let num_axes = indices.len() + 1;
    // Total degree at most 1: every stored exponent tuple sums to <= 1.
    if num.terms.keys().any(|e| e.iter().sum::<u32>() > 1) {
        return None;
    }
    let coeff_of = |axis: usize| -> Rational {
        let mut e = vec![0u32; num_axes];
        e[axis] = 1;
        num.terms
            .get(&e)
            .cloned()
            .unwrap_or_else(|| Rational::from(0))
    };
    let mut coeffs = Vec::with_capacity(num_axes);
    for axis in 0..num_axes {
        let c = coeff_of(axis);
        if *c.clone().denom() != 1 {
            return None;
        }
        coeffs.push(c.numer().to_i64()?);
    }
    let zero_e = vec![0u32; num_axes];
    let d = num
        .terms
        .get(&zero_e)
        .cloned()
        .unwrap_or_else(|| Rational::from(0));
    Some((coeffs, d))
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
        let term = ProperTermM::parse(f, n, &[j, k], &pool).expect("proper hypergeometric");

        // ratio_j = F(n,j+1,k)/F(n,j,k) (axis 1 is j) should equal
        // (n-j)/(j+1) * (j+1)!/(j+1-k)! / (j!/(j-k)!) as a rational function;
        // check it numerically at a sample point instead of re-deriving the
        // closed form, since that *is* what the exact machinery computes.
        let rj = term.ratio_axis(1, 1).expect("ratio_j");
        let vals = [Rational::from(6), Rational::from(3), Rational::from(2)];
        let got = rj.num.eval(&vals) / rj.den.eval(&vals);
        // C(6,4)*C(4,2) / (C(6,3)*C(3,2)) = 15*6 / (20*3) = 90/60 = 3/2
        assert_eq!(got, Rational::from((3, 2)));
    }

    #[test]
    fn refuses_non_hypergeometric_input() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let bad = pool.func("sin", vec![pool.mul(vec![n, j, k])]);
        let err = ProperTermM::parse(bad, n, &[j, k], &pool).expect_err("not hypergeometric");
        assert!(matches!(
            err,
            Telescoping2dError::NotProperHypergeometric(_)
        ));
    }

    /// `m = 3` bound indices: `F(n,x,y,z) = C(n,x)*C(x,y)*C(y,z)`, a chain of
    /// three coupled binomial transforms — exercises a `num_axes = 4` parse.
    #[test]
    fn three_index_chain_parses_and_has_exact_ratios() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let f = pool.mul(vec![
            binom(&pool, n, x),
            binom(&pool, x, y),
            binom(&pool, y, z),
        ]);
        let term = ProperTermM::parse(f, n, &[x, y, z], &pool).expect("proper hypergeometric");
        assert_eq!(term.m, 3);
        let rz = term.ratio_axis(3, 1).expect("ratio_z");
        // F(n,x,y,z+1)/F(n,x,y,z) = C(y,z+1)/C(y,z) at a sample point.
        let vals = [
            Rational::from(8),
            Rational::from(5),
            Rational::from(4),
            Rational::from(1),
        ];
        let got = rz.num.eval(&vals) / rz.den.eval(&vals);
        // C(4,2)/C(4,1) = 6/4 = 3/2
        assert_eq!(got, Rational::from((3, 2)));
    }
}
