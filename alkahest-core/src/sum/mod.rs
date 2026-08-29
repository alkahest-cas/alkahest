//! Creative telescoping / Zeilberger-style symbolic summation (V2-10).
//!
//! Gosper indefinite summation for hypergeometric terms — ratios `F(k+1)/F(k)`
//! that reduce to rational functions of `k`.  Includes constant-coefficient
//! homogeneous recurrence solving (order ≤ 2), explicit [`rsolve`] for linear
//! difference equations (V2-18), and optional WZ pair verification.

mod expr_ratio;
pub(crate) mod gosper;
mod poly_aux;
mod product;
mod ratfunc;
mod recurrence;
mod rsolve;
mod special;

pub use expr_ratio::hypergeom_ratio;
pub use gosper::{gosper_certificate, gosper_normal_form};
pub use product::{product_definite, product_indefinite, ProductError};
pub use ratfunc::RatFunc;
pub use recurrence::{
    solve_linear_recurrence_homogeneous, LinearRecurrenceError, RecurrenceSolution,
};
pub use rsolve::{rsolve, RsolveError};

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::subs::subs;
use crate::kernel::{ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;
use crate::simplify::engine::simplify;
use std::collections::HashMap;
use std::fmt;

fn simp(pool: &ExprPool, e: ExprId) -> ExprId {
    simplify(e, pool).value
}

/// Errors from symbolic summation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SumError {
    /// Term is not hypergeometric or ratio extraction failed.
    NotHypergeometric(String),
    /// Gosper's algorithm does not apply (no rational certificate).
    NotGosperSummable,
    /// Difference-variable substitution failed building bounds.
    BoundSubstitution(String),
}

impl fmt::Display for SumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SumError::NotHypergeometric(s) => write!(f, "sum: not hypergeometric: {s}"),
            SumError::NotGosperSummable => write!(f, "sum: term is not Gosper-summable"),
            SumError::BoundSubstitution(s) => write!(f, "sum: bound substitution: {s}"),
        }
    }
}

impl std::error::Error for SumError {}

impl crate::errors::AlkahestError for SumError {
    fn code(&self) -> &'static str {
        match self {
            SumError::NotHypergeometric(_) => "E-SUM-001",
            SumError::NotGosperSummable => "E-SUM-002",
            SumError::BoundSubstitution(_) => "E-SUM-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "supported indefinite sums are hypergeometric terms built from polynomials in k, products, and gamma(linear(k)); Zeilberger automation is partial — use verify_wz_pair for certificates. For sums to infinity, only recognized Basel-family even p-series (e.g. sum_definite(1/k**2, k, 1, pool.pos_infinity())) resolve to a closed form (pi^2/6, pi^4/90, ...); other improper sums are refused rather than guessed",
        )
    }
}

fn rat_poly_to_expr(pool: &ExprPool, k: ExprId, p: &RatUniPoly) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::new();
    for (deg, coeff) in p.coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let coeff_q = coeff.clone();
        let numer = coeff_q.numer();
        let denom = coeff_q.denom();
        let coeff_expr = if *denom == 1 {
            pool.integer(numer.clone())
        } else {
            pool.rational(numer.clone(), denom.clone())
        };
        let pow_id = if deg == 0 {
            coeff_expr
        } else if deg == 1 {
            pool.mul(vec![coeff_expr, k])
        } else {
            pool.mul(vec![coeff_expr, pool.pow(k, pool.integer(deg as i64))])
        };
        terms.push(pow_id);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

fn ratfunc_to_expr(pool: &ExprPool, k: ExprId, r: &RatFunc) -> ExprId {
    let num_e = rat_poly_to_expr(pool, k, &r.num);
    if r.den.is_zero() || r.den.degree() == 0 && r.den.coeffs.is_empty() {
        return num_e;
    }
    let den_e = rat_poly_to_expr(pool, k, &r.den);
    pool.mul(vec![num_e, pool.pow(den_e, pool.integer(-1_i32))])
}

/// Indefinite Gosper sum: find `G(k)` with `G(k+1)-G(k)=term` when `term` is hypergeometric in `k`.
pub fn sum_indefinite(
    term: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, SumError> {
    let ratio = hypergeom_ratio(term, k, pool)?;
    let cert = gosper_certificate(&ratio).ok_or(SumError::NotGosperSummable)?;
    let cert_e = ratfunc_to_expr(pool, k, &cert);
    let g = simp(pool, pool.mul(vec![term, cert_e]));
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("gosper_indefinite", term, g));
    Ok(DerivedExpr::with_log(g, log))
}

/// Definite sum `∑_{k=lo}^{hi} term(k)` when Gosper applies (upper bound inclusive).
///
/// When `hi` is [`ExprPool::pos_infinity`] this is an infinite sum. Gosper's
/// algorithm never applies there — its antidifference, even when it exists,
/// is a rational-times-hypergeometric term with no reason to vanish at `∞`,
/// so blindly substituting `k = ∞` (like [`crate::integrate::integrate_definite`]
/// warns against for the analogous FTC bound) would fabricate a meaningless
/// value. Instead an infinite upper bound is checked against a small table of
/// recognized closed forms — currently the Basel-family even p-series
/// `Σ_{k=1}^{∞} c/k^{2m} = c·ζ(2m)` (Basel-family even zeta values),
/// e.g. `sum_definite(1/k**2, k, 1, pool.pos_infinity())` → `π²/6`. Anything
/// else with an infinite bound honestly returns
/// [`SumError::NotGosperSummable`] rather than a wrong or unresolved-`∞`
/// answer.
pub fn sum_definite(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, SumError> {
    if hi == pool.pos_infinity() {
        let value = special::basel_family_closed_form(term, k, lo, hi, pool)
            .ok_or(SumError::NotGosperSummable)?;
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple("basel_zeta_even", term, value));
        return Ok(DerivedExpr::with_log(value, log));
    }
    let ind = sum_indefinite(term, k, pool)?;
    let g = ind.value;
    let one = pool.integer(1_i32);
    let hi_p1 = simp(pool, pool.add(vec![hi, one]));

    let mut m_upper = HashMap::new();
    m_upper.insert(k, hi_p1);
    let upper = simp(pool, subs(g, &m_upper, pool));

    let mut m_lower = HashMap::new();
    m_lower.insert(k, lo);
    let lower = simp(pool, subs(g, &m_lower, pool));

    // A pole of the *summand* strictly between the bounds is invisible in the
    // telescoped difference — `G(hi+1) − G(lo)` never mentions the interior
    // indices — so it has to be looked for in the summand itself, the same way
    // `integrate::engine`'s interior-pole guards look at the integrand rather
    // than at `F(b) − F(a)`.  `Σ_{k=1}^{10} 1/((k−3)(k−2))` telescoped to a
    // clean `−5/8` while its `k = 2` and `k = 3` terms divide by zero.
    if let Some(bad) = interior_undefined_index(term, k, lo, hi, pool) {
        return Err(SumError::BoundSubstitution(format!(
            "the summand is undefined at k = {bad}, which lies inside the summation \
             range: that term of the sum is a division by zero, so the sum has no \
             value and the telescoped difference G(hi+1) - G(lo) is not it",
        )));
    }

    let diff = simp(
        pool,
        pool.add(vec![upper, pool.mul(vec![lower, pool.integer(-1_i32)])]),
    );
    // `Σ_{k=0}^{6} 1/(k(k+1))` telescopes to `-1/7 + 0^{-1}`: the `k = 0` term
    // is undefined, so the antidifference has a pole inside the summation
    // range and the telescoping difference is not the sum.  A residual
    // `0^{negative}` is exactly that pole surviving into the answer; it is not
    // a value and must not be returned as one.
    if contains_zero_to_negative_power(diff, pool) {
        return Err(SumError::BoundSubstitution(format!(
            "the antidifference has a pole inside the summation range \
             (telescoping gave {}, which contains a division by zero) — a term of the \
             sum is undefined for some integer between the bounds, so the telescoped \
             difference G(hi+1) - G(lo) is not the sum",
            pool.display(diff),
        )));
    }
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("gosper_definite_telescope", term, diff));
    Ok(DerivedExpr::with_log(diff, log))
}

/// Largest number of individual indices [`interior_undefined_index`] will
/// substitute into when it cannot narrow the candidates structurally.
///
/// The structural pass below is exact and range-independent for a summand whose
/// negative powers are polynomials in `k`, which is every rational summand; this
/// bound only limits the brute-force fallback used for shapes it cannot parse,
/// so a very long range with an exotic summand is answered "no opinion" rather
/// than made slow.
const MAX_POLE_SCAN: i64 = 2048;

/// `expr` as an `i64` when it is an integer literal.
fn const_i64(pool: &ExprPool, e: ExprId) -> Option<i64> {
    match pool.get(e) {
        crate::kernel::ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    }
}

/// Collect the bases of every `X^negative` node, i.e. every denominator.
fn negative_power_bases(expr: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    use crate::kernel::ExprData;
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let negative_exp = match pool.get(exp) {
                ExprData::Integer(n) => n.0 < 0,
                ExprData::Rational(r) => r.0 < 0,
                _ => false,
            };
            if negative_exp {
                out.push(base);
            }
            negative_power_bases(base, pool, out);
            negative_power_bases(exp, pool, out);
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            for &x in xs.iter() {
                negative_power_bases(x, pool, out);
            }
        }
        ExprData::Func { args, .. } => {
            for &a in args.iter() {
                negative_power_bases(a, pool, out);
            }
        }
        _ => {}
    }
}

/// Integer roots of `p` in `[lo, hi]`, read off its ℤ-factorisation.
fn integer_roots_in(p: &crate::poly::UniPoly, lo: i64, hi: i64) -> Vec<i64> {
    let Ok(fac) = p.factor_z() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (fact, _) in &fac.factors {
        if fact.degree() != 1 {
            continue;
        }
        let coeffs = fact.coefficients();
        let (Some(b), Some(a)) = (coeffs.first(), coeffs.get(1)) else {
            continue;
        };
        if *a == 0 {
            continue;
        }
        // root = -b/a, an integer only when a divides b.
        let (q, r) = (-b.clone()).div_rem(a.clone());
        if r != 0 {
            continue;
        }
        if let Some(root) = q.to_i64() {
            if root >= lo && root <= hi {
                out.push(root);
            }
        }
    }
    out
}

/// The smallest integer in `[lo, hi]` at which `term` is undefined, when that
/// can be established; `None` means *no opinion*, never *no pole*.
///
/// Refusal must rest on positive evidence, so this returns an index only after
/// substituting it and seeing an actual `0^{negative}` survive simplification.
/// Candidates come from the roots of the summand's own denominators, so the cost
/// does not scale with the length of the range.
fn interior_undefined_index(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Option<i64> {
    let (lo_i, hi_i) = (const_i64(pool, lo)?, const_i64(pool, hi)?);
    if lo_i > hi_i {
        return None;
    }

    // Simplify first: `(k−2)/(k−2)` is `1`, and the caller asked about the
    // summand, not about an unreduced spelling of it.
    let term = simp(pool, term);

    let mut bases = Vec::new();
    negative_power_bases(term, pool, &mut bases);

    let mut candidates: Vec<i64> = Vec::new();
    let mut unparsed = false;
    for base in bases {
        match crate::poly::UniPoly::from_symbolic_clear_denoms(base, k, pool) {
            Ok(p) if p.degree() >= 1 => candidates.extend(integer_roots_in(&p, lo_i, hi_i)),
            // A constant denominator has no root; anything else (a `gamma`, a
            // `2^k`, a nested quotient) is outside this pass's reach.
            Ok(_) => {}
            Err(_) => unparsed = true,
        }
    }

    if unparsed && hi_i.saturating_sub(lo_i) < MAX_POLE_SCAN {
        candidates.extend(lo_i..=hi_i);
    }
    candidates.sort_unstable();
    candidates.dedup();

    for j in candidates {
        let mut m = HashMap::new();
        m.insert(k, pool.integer(j));
        if contains_zero_to_negative_power(simp(pool, subs(term, &m, pool)), pool) {
            return Some(j);
        }
    }
    None
}

/// True when `expr` contains a `0^n` node with `n` negative — an unresolved
/// division by zero that survived simplification.
fn contains_zero_to_negative_power(expr: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::ExprData;
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let zero_base = matches!(pool.get(base), ExprData::Integer(n) if n.0 == 0);
            let negative_exp = match pool.get(exp) {
                ExprData::Integer(n) => n.0 < 0,
                ExprData::Rational(r) => r.0 < 0,
                _ => false,
            };
            (zero_base && negative_exp)
                || contains_zero_to_negative_power(base, pool)
                || contains_zero_to_negative_power(exp, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            xs.iter().any(|&x| contains_zero_to_negative_power(x, pool))
        }
        ExprData::Func { args, .. } => args
            .iter()
            .any(|&a| contains_zero_to_negative_power(a, pool)),
        _ => false,
    }
}

/// Witness `(F, G)` for Zeilberger/WZ-style telescoping in `k`:
/// checks `F(n+1,k)-F(n,k) = G(n,k+1)-G(n,k)` after clearing denominators by cross-multiplication.
///
/// Requires `n`, `k` distinct symbols. Uses [`simplify`] and structural equality; dense normalization
/// for general `binom`/`gamma` identities is not guaranteed without extra rewrite rules.
#[derive(Clone, Debug)]
pub struct WzPair {
    pub f: ExprId,
    pub g: ExprId,
}

pub fn verify_wz_pair(pair: &WzPair, n: ExprId, k: ExprId, pool: &ExprPool) -> bool {
    let k1 = simp(pool, pool.add(vec![k, pool.integer(1_i32)]));
    let n1 = simp(pool, pool.add(vec![n, pool.integer(1_i32)]));

    let mut mn = HashMap::new();
    mn.insert(n, n1);
    let f_n1_k = simp(pool, subs(pair.f, &mn, pool));

    let lhs = simp(
        pool,
        pool.add(vec![f_n1_k, pool.mul(vec![pair.f, pool.integer(-1_i32)])]),
    );

    let mut mk = HashMap::new();
    mk.insert(k, k1);
    let g_n_k1 = simp(pool, subs(pair.g, &mk, pool));

    let rhs = simp(
        pool,
        pool.add(vec![g_n_k1, pool.mul(vec![pair.g, pool.integer(-1_i32)])]),
    );

    lhs == rhs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::ExprId;
    use crate::kernel::{Domain, ExprData};
    use std::collections::HashMap;

    fn eval_with_gamma(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
        match pool.get(expr) {
            ExprData::Func { name, args } if name == "gamma" && args.len() == 1 => {
                let x = eval_with_gamma(args[0], env, pool)?;
                Some(rug::Float::with_val(53, x).gamma().to_f64())
            }
            ExprData::Add(args) => {
                let mut sum = 0.0f64;
                for &a in &args {
                    sum += eval_with_gamma(a, env, pool)?;
                }
                Some(sum)
            }
            ExprData::Mul(args) => {
                let mut prod = 1.0f64;
                for &a in &args {
                    prod *= eval_with_gamma(a, env, pool)?;
                }
                Some(prod)
            }
            ExprData::Pow { base, exp } => {
                Some(eval_with_gamma(base, env, pool)?.powf(eval_with_gamma(exp, env, pool)?))
            }
            _ => eval_interp(expr, env, pool),
        }
    }

    #[test]
    fn indefinite_k_gamma_k_plus_1() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let gkp1 = pool.func("gamma", vec![pool.add(vec![k, pool.integer(1_i32)])]);
        let term = simp(&pool, pool.mul(vec![k, gkp1]));
        let r = sum_indefinite(term, k, &pool).expect("gosper");
        assert!(pool.with(r.value, |d| matches!(
            d,
            ExprData::Func { .. } | ExprData::Mul(_)
        )));
    }

    #[test]
    fn definite_sum_kfactorial_telescope() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let n = pool.symbol("n", Domain::Real);
        let zero = pool.integer(0_i32);
        let gkp1 = pool.func("gamma", vec![pool.add(vec![k, pool.integer(1_i32)])]);
        let term = simp(&pool, pool.mul(vec![k, gkp1]));
        let s = sum_definite(term, k, zero, n, &pool).expect("definite");
        let expected = simp(
            &pool,
            pool.add(vec![
                pool.func("gamma", vec![pool.add(vec![n, pool.integer(2_i32)])]),
                pool.integer(-1_i32),
            ]),
        );
        for ni in 0..=8 {
            let mut env = HashMap::new();
            env.insert(n, ni as f64);
            let sv = eval_with_gamma(s.value, &env, &pool).expect("sum eval");
            let ev = eval_with_gamma(expected, &env, &pool).expect("expected eval");
            assert!(
                (sv - ev).abs() < 1e-5 * ev.abs().max(1.0),
                "n={ni}: got {sv} want {ev}"
            );
        }
    }

    #[test]
    fn indefinite_sum_of_k() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let r = sum_indefinite(k, k, &pool).expect("Σk indefinite");
        // G(k) = k(k-1)/2
        for ki in 1..=10 {
            let mut env = HashMap::new();
            env.insert(k, ki as f64);
            let gv = eval_interp(r.value, &env, &pool).expect("G eval");
            let expected = (ki * (ki - 1)) as f64 / 2.0;
            assert!((gv - expected).abs() < 1e-9, "G({ki})={gv} want {expected}");
        }
    }

    #[test]
    fn definite_sum_of_k_one_to_ten() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let lo = pool.integer(1_i32);
        let hi = pool.integer(10_i32);
        let s = sum_definite(k, k, lo, hi, &pool).expect("Σ_{1}^{10} k");
        let v = eval_interp(s.value, &HashMap::new(), &pool).expect("eval");
        assert!((v - 55.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn definite_geometric_two_pow_k() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let term = pool.pow(pool.integer(2_i32), k);
        let lo = pool.integer(0_i32);
        let hi = pool.integer(5_i32);
        let s = sum_definite(term, k, lo, hi, &pool).expect("Σ 2^k");
        let v = eval_interp(s.value, &HashMap::new(), &pool).expect("eval");
        assert!((v - 63.0).abs() < 1e-9, "got {v}"); // 1+2+4+8+16+32
    }

    /// A pole of the summand *strictly inside* the range makes the sum
    /// undefined; the telescoped difference does not know that.
    #[test]
    fn interior_pole_is_refused_not_telescoped() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let i = |v: i32| pool.integer(v);

        // Σ_{k=1}^{10} 1/((k−3)(k−2)) — the k = 2 and k = 3 terms divide by zero.
        let den = simp(
            &pool,
            pool.mul(vec![
                simp(&pool, pool.add(vec![k, i(-3)])),
                simp(&pool, pool.add(vec![k, i(-2)])),
            ]),
        );
        let term = simp(&pool, pool.pow(den, i(-1)));
        let err = sum_definite(term, k, i(1), i(10), &pool).expect_err("must refuse");
        assert!(matches!(err, SumError::BoundSubstitution(_)));
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-SUM-003");

        // Σ_{k=−2}^{5} 1/(k(k+1)) — poles at k = −1 and k = 0.
        let den = simp(
            &pool,
            pool.mul(vec![k, simp(&pool, pool.add(vec![k, i(1)]))]),
        );
        let term = simp(&pool, pool.pow(den, i(-1)));
        assert!(sum_definite(term, k, i(-2), i(5), &pool).is_err());
    }

    /// The nearest convergent neighbours must still evaluate: the guard has to
    /// fire on a pole, not on the shape.
    #[test]
    fn poles_outside_the_range_do_not_block_the_sum() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let i = |v: i32| pool.integer(v);

        // Σ_{k=1}^{10} 1/(k(k+1)) = 1 − 1/11 = 10/11.
        let den = simp(
            &pool,
            pool.mul(vec![k, simp(&pool, pool.add(vec![k, i(1)]))]),
        );
        let term = simp(&pool, pool.pow(den, i(-1)));
        let s = sum_definite(term, k, i(1), i(10), &pool).expect("no pole in [1, 10]");
        let v = eval_interp(s.value, &HashMap::new(), &pool).expect("eval");
        assert!((v - 10.0 / 11.0).abs() < 1e-12, "got {v}");

        // Σ_{k=4}^{10} 1/((k−3)(k−2)): poles at 2 and 3, both below the range.
        let den = simp(
            &pool,
            pool.mul(vec![
                simp(&pool, pool.add(vec![k, i(-3)])),
                simp(&pool, pool.add(vec![k, i(-2)])),
            ]),
        );
        let term = simp(&pool, pool.pow(den, i(-1)));
        let s = sum_definite(term, k, i(4), i(10), &pool).expect("no pole in [4, 10]");
        let v = eval_interp(s.value, &HashMap::new(), &pool).expect("eval");
        // Σ_{k=4}^{10} (1/(k−3) − 1/(k−2)) = 1 − 1/8 = 7/8.
        assert!((v - 0.875).abs() < 1e-12, "got {v}");
    }

    #[test]
    fn wz_pair_zero_is_certificate() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let z = pool.integer(0_i32);
        let pair = WzPair { f: z, g: z };
        assert!(verify_wz_pair(&pair, n, k, &pool));
    }
}
