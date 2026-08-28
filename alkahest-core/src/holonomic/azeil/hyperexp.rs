//! Recognising *hyperexponential* terms and computing their exact logarithmic
//! derivative and shift ratios.
//!
//! `F` is **hyperexponential** in `x` when `F'/F` is a rational function —
//! equivalently `F = exp(∫ r)` for a rational `r`. That logarithmic derivative
//! is the whole content of `F` as far as differential creative telescoping is
//! concerned, the way `F(n,k+1)/F(n,k)` is in the discrete case.
//!
//! The class this module parses, the differential mirror of
//! [`super::super::hyperterm::ProperTerm`], is
//!
//! ```text
//! F(n, x) = R(n, x) · wⁿ · exp(η(n, x)) · ∏_j B_j(n, x)^(α_j·n + β_j)
//! ```
//!
//! with `R, η, B_j ∈ Q(n)(x)`, `w ∈ Q \ {0}`, `α_j ∈ Z`, `β_j ∈ Q`. Its two
//! certificates are:
//!
//! - [`HyperExpTerm::theta`] — `∂_x F / F ∈ Q(n)(x)`, always available:
//!   ```text
//!   θ = R'/R + ∂_x η + Σ_j (α_j·n + β_j)·B_j'/B_j
//!   ```
//! - [`HyperExpTerm::ratio_n`] — `F(n+i, x)/F(n, x) ∈ Q(n)(x)`, available only
//!   when `F` is *also* hypergeometric in `n`:
//!   ```text
//!   F(n+i,x)/F(n,x) = (R(n+i,x)/R(n,x)) · wⁱ · ∏_j B_j^(α_j·i)
//!   ```
//!
//! The two are deliberately split, and the split is not cosmetic. `exp(n·x)` is
//! hyperexponential in `x` (`θ = n`) and its `n`-ratio is `eˣ`, which is not
//! rational: [`HyperExpTerm::theta`] succeeds and [`HyperExpTerm::ratio_n`]
//! refuses with [`DiffTelescopingError::NotHypergeometricInN`]. A caller doing
//! indefinite integration (`super::dgosper`) only needs the first; creative
//! telescoping (`super::search`) needs both, and gets a *specific* refusal
//! rather than "not hyperexponential" when only the second is missing.
//!
//! # Why `η` and the `B_j` may carry `n` for `theta` but not for `ratio_n`
//!
//! `∂_x` is `Q(n)`-linear, so any `n`-dependence is inert for the logarithmic
//! derivative. The `n`-shift is a different matter: `exp(η(n+i,x) − η(n,x))` is
//! rational only when `η(n+i,·) = η(n,·)`, and `B(n+i,x)^(…)/B(n,x)^(…)` only
//! when the base is `n`-free (or the exponent is an `n`-free integer, in which
//! case the parser has already folded the factor into `R`). Both conditions are
//! *checked* by [`HyperExpTerm::ratio_n`] rather than assumed at parse time,
//! which is what lets the parser accept the wider `x`-only class.
//!
//! # Branches
//!
//! `B^β` for non-integer `β` is a formal symbol here, with `(B^β)' = β·B'/B·B^β`
//! by fiat. Every identity this module underwrites is therefore an identity of
//! formal hyperexponential expressions; turning it into an identity of
//! *functions* needs a consistent branch of `B^β` on the domain in question.
//! See the [module docs](super) — this is stated, not silently assumed.

use super::DiffTelescopingError;
use crate::holonomic::hyperterm::{affine_parts, as_ratk, ratk_to_expr};
use crate::holonomic::qfield::{ratk_deriv_k, rn_add, rn_int, rn_mul, rn_rat, rn_var, RatK};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use rug::Rational;

/// Largest integer exponent accepted on a sub-term (guards against blow-up).
const MAX_POW: i32 = 32;
/// Largest `|α·i|` accepted when forming an `n`-shift ratio.
const MAX_SHIFT_POW: i64 = 512;
/// Largest nesting depth the parser will descend.
const MAX_PARSE_DEPTH: usize = 64;

/// One `B(n,x)^(α·n + β)` factor.
///
/// `α = 0` with `β ∈ Z` never reaches here: the parser folds an integer power
/// into the rational prefactor instead, which is what keeps
/// [`HyperExpTerm::ratio_n`]'s "the base must be `n`-free" check from rejecting
/// perfectly ordinary rational integrands.
#[derive(Clone, Debug)]
pub struct PowerFactor {
    /// The base `B`, an exact element of `Q(n)(x)`.
    pub base: RatK,
    /// Coefficient of `n` in the exponent.
    pub alpha: i64,
    /// Constant part of the exponent.
    pub beta: Rational,
}

/// A parsed hyperexponential term
/// `F = rat · w^n · exp(eta) · ∏ powers[j].base ^ (α_j·n + β_j)`.
#[derive(Clone, Debug)]
pub struct HyperExpTerm {
    /// Rational-function prefactor `R(n, x)`.
    pub rat: RatK,
    /// Base of the `wⁿ` factor.
    pub w: Rational,
    /// Exponent of the `exp(η)` factor.
    pub eta: RatK,
    /// The `B^(α·n + β)` factors.
    pub powers: Vec<PowerFactor>,
}

impl HyperExpTerm {
    fn one() -> Self {
        HyperExpTerm {
            rat: RatK::one(),
            w: Rational::from(1),
            eta: RatK::zero(),
            powers: Vec::new(),
        }
    }

    fn from_ratk(r: RatK) -> Self {
        HyperExpTerm {
            rat: r,
            w: Rational::from(1),
            eta: RatK::zero(),
            powers: Vec::new(),
        }
    }

    fn mul(&self, other: &HyperExpTerm) -> HyperExpTerm {
        let mut powers = self.powers.clone();
        powers.extend(other.powers.iter().cloned());
        HyperExpTerm {
            rat: self.rat.mul(&other.rat),
            w: self.w.clone() * other.w.clone(),
            eta: self.eta.add(&other.eta),
            powers,
        }
    }

    fn pow(&self, e: i32) -> Result<HyperExpTerm, DiffTelescopingError> {
        if e.unsigned_abs() > MAX_POW as u32 {
            return Err(DiffTelescopingError::NotHyperexponential(format!(
                "exponent {e} is outside the supported range (|e| ≤ {MAX_POW})"
            )));
        }
        let rat = self.rat.pow_i32(e).ok_or_else(|| {
            DiffTelescopingError::NotHyperexponential(
                "a factor raised to a negative power is identically zero".into(),
            )
        })?;
        let scale = RatK::from_rn(rn_int(e as i64));
        let powers = self
            .powers
            .iter()
            .map(|p| {
                Ok(PowerFactor {
                    base: p.base.clone(),
                    alpha: p.alpha.checked_mul(e as i64).ok_or_else(|| {
                        DiffTelescopingError::NotHyperexponential(
                            "exponent overflow in a power factor".into(),
                        )
                    })?,
                    beta: p.beta.clone() * Rational::from(e),
                })
            })
            .collect::<Result<Vec<_>, DiffTelescopingError>>()?;
        Ok(HyperExpTerm {
            rat,
            w: rat_pow(&self.w, e as i64)?,
            eta: self.eta.mul(&scale),
            powers,
        })
    }

    /// `∂_x F / F` as an exact element of `Q(n)(x)`.
    ///
    /// This is the stage-1 certificate: it *is* the statement that `F` is
    /// hyperexponential, written down exactly.
    pub fn theta(&self) -> Result<RatK, DiffTelescopingError> {
        if self.rat.is_zero() {
            return Err(DiffTelescopingError::NotHyperexponential(
                "the term is identically zero, so F'/F is undefined".into(),
            ));
        }
        let mut acc = log_deriv(&self.rat)?;
        acc = acc.add(&ratk_deriv_k(&self.eta));
        for p in &self.powers {
            let e = RatK::from_rn(rn_add(
                &rn_mul(&rn_int(p.alpha), &rn_var()),
                &rn_rat(p.beta.clone()),
            ));
            acc = acc.add(&e.mul(&log_deriv(&p.base)?));
        }
        Ok(acc)
    }

    /// `F(n+i, x) / F(n, x)` as an exact element of `Q(n)(x)`.
    ///
    /// Refuses with [`DiffTelescopingError::NotHypergeometricInN`] when the
    /// ratio is not rational — which is a statement about `F`, not about this
    /// parser's reach, and closes the branch for every algorithm in this
    /// family.
    pub fn ratio_n(&self, i: i64) -> Result<RatK, DiffTelescopingError> {
        if i == 0 {
            return Ok(RatK::one());
        }
        if self.rat.is_zero() {
            return Err(DiffTelescopingError::NotHyperexponential(
                "the term is identically zero".into(),
            ));
        }
        if !self.eta.shift_n(i).eq_ratk(&self.eta) {
            return Err(DiffTelescopingError::NotHypergeometricInN(
                "the argument of exp depends on n, so F(n+i,x)/F(n,x) contains \
                 exp(eta(n+i,x) - eta(n,x)), which is not a rational function of x"
                    .into(),
            ));
        }
        let mut acc = self.rat.shift_n(i).div(&self.rat).ok_or_else(|| {
            DiffTelescopingError::NotHyperexponential("the term is identically zero".into())
        })?;
        acc = acc.mul(&RatK::from_rn(rn_rat(rat_pow(&self.w, i)?)));
        for p in &self.powers {
            if !p.base.shift_n(i).eq_ratk(&p.base) {
                return Err(DiffTelescopingError::NotHypergeometricInN(
                    "the base of a symbolic power depends on n, so the n-shift ratio is not \
                     a rational function of x"
                        .into(),
                ));
            }
            let shift = p.alpha.checked_mul(i).ok_or_else(|| {
                DiffTelescopingError::NotHypergeometricInN("exponent overflow".into())
            })?;
            if shift == 0 {
                continue;
            }
            if shift.abs() > MAX_SHIFT_POW {
                return Err(DiffTelescopingError::SearchExhausted(format!(
                    "n-shift exponent {shift} exceeds the supported limit of {MAX_SHIFT_POW}"
                )));
            }
            let step = p.base.pow_i32(shift as i32).ok_or_else(|| {
                DiffTelescopingError::NotHyperexponential(
                    "a power factor's base is identically zero".into(),
                )
            })?;
            acc = acc.mul(&step);
        }
        Ok(acc)
    }

    /// Parse an expression into the hyperexponential class in `(n, x)`.
    ///
    /// `n` may be any symbol that does not occur in `expr` when only the
    /// `x`-side is wanted; see [`hyperexp_log_derivative`].
    pub fn parse(
        expr: ExprId,
        n: ExprId,
        x: ExprId,
        pool: &ExprPool,
    ) -> Result<HyperExpTerm, DiffTelescopingError> {
        if n == x {
            return Err(DiffTelescopingError::InvalidInput(
                "the outer index n and the integration variable x must be distinct symbols".into(),
            ));
        }
        parse_rec(expr, n, x, pool, 0)
    }
}

/// `p'/p` for a nonzero `p ∈ Q(n)(x)`.
fn log_deriv(p: &RatK) -> Result<RatK, DiffTelescopingError> {
    ratk_deriv_k(p).div(p).ok_or_else(|| {
        DiffTelescopingError::NotHyperexponential(
            "a factor of the term is identically zero, so its logarithmic derivative is \
             undefined"
                .into(),
        )
    })
}

fn rat_pow(q: &Rational, e: i64) -> Result<Rational, DiffTelescopingError> {
    if e == 0 {
        return Ok(Rational::from(1));
    }
    if e.unsigned_abs() > 1024 {
        return Err(DiffTelescopingError::NotHyperexponential(
            "exponential factor exponent exceeds the supported limit".into(),
        ));
    }
    if *q == 0 {
        if e < 0 {
            return Err(DiffTelescopingError::NotHyperexponential(
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
    x: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<HyperExpTerm, DiffTelescopingError> {
    if depth > MAX_PARSE_DEPTH {
        return Err(DiffTelescopingError::NotHyperexponential(
            "expression nests deeper than the parser supports".into(),
        ));
    }
    // Fast path: a purely rational sub-expression contributes only to `R`.
    if let Some(r) = as_ratk(expr, n, x, pool, 0) {
        return Ok(HyperExpTerm::from_ratk(r));
    }
    match pool.get(expr) {
        ExprData::Mul(args) => {
            let mut acc = HyperExpTerm::one();
            for a in args {
                acc = acc.mul(&parse_rec(a, n, x, pool, depth + 1)?);
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => parse_pow(base, exp, n, x, pool, depth),
        ExprData::Func { name, args } => parse_func(&name, &args, n, x, pool, depth),
        ExprData::Add(_) => Err(DiffTelescopingError::NotHyperexponential(format!(
            "a sum of hyperexponential terms is not itself hyperexponential (unless it is \
             rational, which {} is not)",
            pool.display(expr)
        ))),
        other => Err(DiffTelescopingError::NotHyperexponential(format!(
            "unsupported node {other:?} in {}",
            pool.display(expr)
        ))),
    }
}

fn parse_pow(
    base: ExprId,
    exp: ExprId,
    n: ExprId,
    x: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<HyperExpTerm, DiffTelescopingError> {
    // An integer literal exponent is a plain power of whatever the base parses to.
    if let ExprData::Integer(i) = pool.get(exp) {
        let Some(e) = i.0.to_i32() else {
            return Err(DiffTelescopingError::NotHyperexponential(
                "integer exponent does not fit in i32".into(),
            ));
        };
        return parse_rec(base, n, x, pool, depth + 1)?.pow(e);
    }

    // Otherwise the exponent must be `α·n + β` with `α ∈ Z`, `β ∈ Q`, and in
    // particular free of `x`: `B(x)^x` is not hyperexponential (its logarithmic
    // derivative contains `log B`).
    let (alpha, x_coeff, beta) = affine_parts(exp, n, x, pool).ok_or_else(|| {
        DiffTelescopingError::NotHyperexponential(format!(
            "exponent {} is not of the form a*n + b with integer a and rational b",
            pool.display(exp)
        ))
    })?;
    if x_coeff != 0 {
        return Err(DiffTelescopingError::NotHyperexponential(format!(
            "exponent {} depends on the integration variable; c**x is hyperexponential only \
             over a field containing log(c), which Q(n)(x) is not",
            pool.display(exp)
        )));
    }

    // `α = 0` and `β ∈ Z`: an ordinary rational power. Fold it into `R` so that
    // `ratio_n`'s "the base must be n-free" check never sees it.
    if alpha == 0 {
        if let Some(b) = rational_to_i32(&beta) {
            let inner = parse_rec(base, n, x, pool, depth + 1)?;
            return inner.pow(b);
        }
    }

    let b = as_ratk(base, n, x, pool, 0).ok_or_else(|| {
        DiffTelescopingError::NotHyperexponential(format!(
            "a symbolic power needs a rational-function base, got {}",
            pool.display(base)
        ))
    })?;
    if b.is_zero() {
        return Err(DiffTelescopingError::NotHyperexponential(
            "zero raised to a symbolic power".into(),
        ));
    }
    // A base that is a nonzero `x`-free constant is a `wⁿ` factor, not a power
    // factor: it contributes nothing to `θ` and `w^i` to the shift ratio.
    if let Some(c) = ratk_as_rational(&b) {
        let bi = rational_to_i32(&beta).ok_or_else(|| {
            DiffTelescopingError::NotHyperexponential(
                "a rational base raised to a non-integer constant power leaves Q (e.g. 2**(1/2))"
                    .into(),
            )
        })?;
        return Ok(HyperExpTerm {
            rat: RatK::from_rn(rn_rat(rat_pow(&c, bi as i64)?)),
            w: rat_pow(&c, alpha)?,
            eta: RatK::zero(),
            powers: Vec::new(),
        });
    }
    Ok(HyperExpTerm {
        rat: RatK::one(),
        w: Rational::from(1),
        eta: RatK::zero(),
        powers: vec![PowerFactor {
            base: b,
            alpha,
            beta,
        }],
    })
}

fn parse_func(
    name: &str,
    args: &[ExprId],
    n: ExprId,
    x: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<HyperExpTerm, DiffTelescopingError> {
    match (name, args.len()) {
        ("exp", 1) => {
            let eta = as_ratk(args[0], n, x, pool, 0).ok_or_else(|| {
                DiffTelescopingError::NotHyperexponential(format!(
                    "exp(...) is hyperexponential only when its argument is a rational \
                     function of the indices; got exp({})",
                    pool.display(args[0])
                ))
            })?;
            Ok(HyperExpTerm {
                rat: RatK::one(),
                w: Rational::from(1),
                eta,
                powers: Vec::new(),
            })
        }
        ("sqrt", 1) => {
            let inner = parse_rec(args[0], n, x, pool, depth + 1)?;
            // sqrt(F) is hyperexponential whenever F is, with every exponent
            // halved — but only the rational-power part can be halved exactly,
            // so a non-square rational prefactor becomes a power factor.
            let mut out = HyperExpTerm {
                rat: RatK::one(),
                w: Rational::from(1),
                eta: inner
                    .eta
                    .mul(&RatK::from_rn(rn_rat(Rational::from((1, 2))))),
                powers: inner
                    .powers
                    .iter()
                    .map(|p| {
                        if p.alpha % 2 != 0 {
                            return Err(DiffTelescopingError::NotHyperexponential(
                                "sqrt of a factor with an odd n-exponent leaves the class \
                                 (its n-shift ratio would be a square root)"
                                    .into(),
                            ));
                        }
                        Ok(PowerFactor {
                            base: p.base.clone(),
                            alpha: p.alpha / 2,
                            beta: p.beta.clone() / Rational::from(2),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            };
            if inner.w != 1 {
                return Err(DiffTelescopingError::NotHyperexponential(
                    "sqrt of a w**n factor leaves Q".into(),
                ));
            }
            if !inner.rat.eq_ratk(&RatK::one()) {
                out.powers.push(PowerFactor {
                    base: inner.rat.clone(),
                    alpha: 0,
                    beta: Rational::from((1, 2)),
                });
            }
            Ok(out)
        }
        _ => Err(DiffTelescopingError::NotHyperexponential(format!(
            "function `{name}/{}` is not part of the hyperexponential class \
             (supported heads: exp, sqrt)",
            args.len()
        ))),
    }
}

fn rational_to_i32(q: &Rational) -> Option<i32> {
    if *q.clone().denom() != 1 {
        return None;
    }
    q.numer().to_i32()
}

/// A `Q(n)(x)` element that is a plain rational number, or `None`.
fn ratk_as_rational(r: &RatK) -> Option<Rational> {
    if r.num.degree() > 0 || r.den.degree() > 0 {
        return None;
    }
    let num = r.num.coeff(0);
    let den = r.den.coeff(0);
    let q = crate::holonomic::qfield::rn_div(&num, &den)?;
    if q.num.degree() > 0 || q.den.degree() > 0 {
        return None;
    }
    let a = q.num.coeffs.first().cloned()?;
    let b = q.den.coeffs.first().cloned()?;
    if b == 0 {
        return None;
    }
    Some(a / b)
}

/// Stage-1 entry point for the univariate case: `F'(x)/F(x)` as an expression
/// in `x`, for a hyperexponential `F`.
///
/// The returned expression is the exact logarithmic-derivative certificate:
/// `F = exp(∫ θ)` with `θ` rational, which is the definition of
/// hyperexponential. No `n` is involved, so a fresh internal symbol stands in
/// for the outer index and cannot appear in the result.
pub fn hyperexp_log_derivative(
    f: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DiffTelescopingError> {
    let n = fresh_outer_index(pool);
    let term = HyperExpTerm::parse(f, n, x, pool)?;
    let theta = term.theta()?;
    Ok(ratk_to_expr(pool, n, x, &theta))
}

/// A symbol for the outer index in calls that have no genuine outer index.
///
/// The name is deliberately not a plausible user symbol. Nothing in a
/// univariate call can *produce* an occurrence of it, so its only effect is to
/// give the shared bivariate machinery a second variable to be constant in.
pub(super) fn fresh_outer_index(pool: &ExprPool) -> ExprId {
    pool.symbol("__azeil_no_outer_index", Domain::Real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holonomic::hyperterm::ratk_to_expr;
    use crate::kernel::Domain;

    fn nx(pool: &ExprPool) -> (ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("x", Domain::Real),
        )
    }

    fn theta_str(expr: ExprId, pool: &ExprPool) -> String {
        let (n, x) = nx(pool);
        let t = HyperExpTerm::parse(expr, n, x, pool)
            .expect("term parses")
            .theta()
            .expect("theta exists");
        format!("{}", pool.display(ratk_to_expr(pool, n, x, &t)))
    }

    /// `θ` is a *function*, so comparing it against a hand-written expected
    /// value is done by exact `Q(n)(x)` equality, not by string matching.
    fn assert_theta_eq(expr: ExprId, expected: ExprId, pool: &ExprPool) {
        let (n, x) = nx(pool);
        let t = HyperExpTerm::parse(expr, n, x, pool)
            .expect("term parses")
            .theta()
            .expect("theta exists");
        let want = as_ratk(expected, n, x, pool, 0).expect("expected value is rational");
        assert!(
            t.eq_ratk(&want),
            "theta mismatch: got {}, want {}",
            pool.display(ratk_to_expr(pool, n, x, &t)),
            pool.display(expected)
        );
    }

    #[test]
    fn power_of_x_with_rational_exponent() {
        // F = x^(3/2)  ⇒  F'/F = (3/2)/x
        let pool = ExprPool::new();
        let (_, x) = nx(&pool);
        let f = pool.pow(x, pool.rational(3, 2));
        let expected = pool.mul(vec![pool.rational(3, 2), pool.pow(x, pool.integer(-1_i32))]);
        assert_theta_eq(f, expected, &pool);
    }

    #[test]
    fn exp_of_a_polynomial() {
        // F = exp(x^2 - 3x)  ⇒  F'/F = 2x - 3
        let pool = ExprPool::new();
        let (_, x) = nx(&pool);
        let arg = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(-3_i32), x]),
        ]);
        let f = pool.func("exp", vec![arg]);
        let expected = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(-3_i32),
        ]);
        assert_theta_eq(f, expected, &pool);
    }

    #[test]
    fn one_minus_x_to_the_n() {
        // F = (1-x)^n  ⇒  F'/F = -n/(1-x)
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let base = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let f = pool.pow(base, n);
        let expected = pool.mul(vec![
            pool.integer(-1_i32),
            n,
            pool.pow(base, pool.integer(-1_i32)),
        ]);
        assert_theta_eq(f, expected, &pool);
    }

    #[test]
    fn x_to_the_n_times_exp_minus_x() {
        // F = x^n e^{-x}  ⇒  F'/F = n/x - 1, and F(n+1,x)/F(n,x) = x.
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, n),
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), x])]),
        ]);
        let expected = pool.add(vec![
            pool.mul(vec![n, pool.pow(x, pool.integer(-1_i32))]),
            pool.integer(-1_i32),
        ]);
        assert_theta_eq(f, expected, &pool);

        let term = HyperExpTerm::parse(f, n, x, &pool).expect("parses");
        let r1 = term.ratio_n(1).expect("hypergeometric in n");
        assert!(r1.eq_ratk(&RatK::k()), "F(n+1,x)/F(n,x) must be x");
    }

    #[test]
    fn gaussian_moment_term() {
        // F = x^{2n} e^{-x^2}: θ = 2n/x - 2x, ratio = x^2.
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, pool.mul(vec![pool.integer(2_i32), n])),
            pool.func(
                "exp",
                vec![pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))])],
            ),
        ]);
        let expected = pool.add(vec![
            pool.mul(vec![
                pool.integer(2_i32),
                n,
                pool.pow(x, pool.integer(-1_i32)),
            ]),
            pool.mul(vec![pool.integer(-2_i32), x]),
        ]);
        assert_theta_eq(f, expected, &pool);

        let term = HyperExpTerm::parse(f, n, x, &pool).expect("parses");
        let r1 = term.ratio_n(1).expect("hypergeometric in n");
        let want = RatK::k().mul(&RatK::k());
        assert!(r1.eq_ratk(&want), "F(n+1,x)/F(n,x) must be x^2");
    }

    #[test]
    fn rational_multiple_and_product() {
        // F = (x/(x+1)) · x^n · (1-x)^n: θ must be the sum of the three parts.
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let one_minus = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let f = pool.mul(vec![
            x,
            pool.pow(pool.add(vec![x, pool.integer(1_i32)]), pool.integer(-1_i32)),
            pool.pow(x, n),
            pool.pow(one_minus, n),
        ]);
        let expected = pool.add(vec![
            pool.pow(x, pool.integer(-1_i32)),
            pool.mul(vec![
                pool.integer(-1_i32),
                pool.pow(pool.add(vec![x, pool.integer(1_i32)]), pool.integer(-1_i32)),
            ]),
            pool.mul(vec![n, pool.pow(x, pool.integer(-1_i32))]),
            pool.mul(vec![
                pool.integer(-1_i32),
                n,
                pool.pow(one_minus, pool.integer(-1_i32)),
            ]),
        ]);
        assert_theta_eq(f, expected, &pool);
    }

    #[test]
    fn w_to_the_n_factor() {
        // F = 2^n x^n: θ = n/x (the 2^n is x-free), ratio = 2x.
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![pool.pow(pool.integer(2_i32), n), pool.pow(x, n)]);
        let expected = pool.mul(vec![n, pool.pow(x, pool.integer(-1_i32))]);
        assert_theta_eq(f, expected, &pool);

        let term = HyperExpTerm::parse(f, n, x, &pool).expect("parses");
        let r1 = term.ratio_n(1).expect("hypergeometric in n");
        let want = RatK::k().mul(&RatK::from_rn(rn_int(2)));
        assert!(r1.eq_ratk(&want), "F(n+1,x)/F(n,x) must be 2x");
    }

    #[test]
    fn sqrt_is_a_half_power() {
        // F = sqrt(1 - x^2): θ = -x/(1-x^2).
        let pool = ExprPool::new();
        let (_, x) = nx(&pool);
        let inner = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))]),
        ]);
        let f = pool.func("sqrt", vec![inner]);
        let expected = pool.mul(vec![
            pool.integer(-1_i32),
            x,
            pool.pow(inner, pool.integer(-1_i32)),
        ]);
        assert_theta_eq(f, expected, &pool);
    }

    #[test]
    fn univariate_entry_point() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
        let theta = hyperexp_log_derivative(f, x, &pool).expect("hyperexponential");
        // 1/x + 1
        let s = format!("{}", pool.display(theta));
        assert!(s.contains('x'), "log derivative should mention x, got {s}");
    }

    // --- negative tests: the class boundary --------------------------------

    #[test]
    fn log_is_refused() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.func("log", vec![x]);
        let err = HyperExpTerm::parse(f, n, x, &pool).expect_err("log is not hyperexponential");
        assert!(matches!(err, DiffTelescopingError::NotHyperexponential(_)));
    }

    #[test]
    fn sum_of_hyperexponentials_is_refused() {
        // e^x + e^{2x} is not hyperexponential: its logarithmic derivative is
        // not rational. Refusing is the point — a wrong θ here would poison
        // every downstream certificate.
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.add(vec![
            pool.func("exp", vec![x]),
            pool.func("exp", vec![pool.mul(vec![pool.integer(2_i32), x])]),
        ]);
        let err = HyperExpTerm::parse(f, n, x, &pool).expect_err("a sum is refused");
        assert!(matches!(err, DiffTelescopingError::NotHyperexponential(_)));
    }

    #[test]
    fn x_in_the_exponent_of_a_constant_is_refused() {
        // 2^x = exp(x log 2) is hyperexponential over Q(log 2)(x), not over Q(x).
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.pow(pool.integer(2_i32), x);
        let err = HyperExpTerm::parse(f, n, x, &pool).expect_err("2**x is refused");
        assert!(matches!(err, DiffTelescopingError::NotHyperexponential(_)));
    }

    #[test]
    fn exp_of_n_times_x_is_hyperexponential_but_not_hypergeometric_in_n() {
        // The instructive case: θ exists, ratio_n does not.
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.func("exp", vec![pool.mul(vec![n, x])]);
        let term = HyperExpTerm::parse(f, n, x, &pool).expect("hyperexponential in x");
        let theta = term.theta().expect("theta exists");
        assert!(
            theta.eq_ratk(&RatK::from_rn(rn_var())),
            "theta must be n, got {}",
            pool.display(ratk_to_expr(&pool, n, x, &theta))
        );
        let err = term
            .ratio_n(1)
            .expect_err("ratio in n is exp(x), not rational");
        assert!(matches!(err, DiffTelescopingError::NotHypergeometricInN(_)));
    }

    #[test]
    fn sin_is_refused() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.func("sin", vec![x]);
        let err = HyperExpTerm::parse(f, n, x, &pool).expect_err("sin is not hyperexponential");
        assert!(matches!(err, DiffTelescopingError::NotHyperexponential(_)));
    }

    #[test]
    fn identical_symbols_are_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let err = HyperExpTerm::parse(x, x, x, &pool).expect_err("n and x must differ");
        assert!(matches!(err, DiffTelescopingError::InvalidInput(_)));
    }

    #[test]
    fn theta_of_a_rational_function_is_its_log_derivative() {
        // A rational F is hyperexponential with θ = F'/F — the degenerate but
        // legitimate case.
        let pool = ExprPool::new();
        let (_, x) = nx(&pool);
        let f = pool.pow(pool.add(vec![x, pool.integer(1_i32)]), pool.integer(3_i32));
        let expected = pool.mul(vec![
            pool.integer(3_i32),
            pool.pow(pool.add(vec![x, pool.integer(1_i32)]), pool.integer(-1_i32)),
        ]);
        assert_theta_eq(f, expected, &pool);
        let _ = theta_str(f, &pool);
    }
}
