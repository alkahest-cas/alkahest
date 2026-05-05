//! Creative telescoping / Zeilberger-style symbolic summation (V2-10).
//!
//! Gosper indefinite summation for hypergeometric terms — ratios `F(k+1)/F(k)`
//! that reduce to rational functions of `k`.  Includes constant-coefficient
//! homogeneous recurrence solving (order ≤ 2) and optional WZ pair verification.

mod expr_ratio;
mod gosper;
mod poly_aux;
mod ratfunc;
mod recurrence;

pub use expr_ratio::hypergeom_ratio;
pub use gosper::{gosper_certificate, gosper_normal_form};
pub use ratfunc::RatFunc;
pub use recurrence::{
    solve_linear_recurrence_homogeneous, LinearRecurrenceError, RecurrenceSolution,
};

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::subs::subs;
use crate::kernel::{ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;
use crate::simplify::engine::simplify;
use rug::{Integer, Rational};
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
            "supported indefinite sums are hypergeometric terms built from polynomials in k, products, and gamma(linear(k)); Zeilberger automation is partial — use verify_wz_pair for certificates",
        )
    }
}

fn rat_poly_to_expr(pool: &ExprPool, k: ExprId, p: &RatUniPoly) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::new();
    for (deg, coeff) in p.coeffs.iter().enumerate() {
        if *coeff == Rational::from(0) {
            continue;
        }
        let coeff_q = coeff.clone();
        let numer = coeff_q.numer();
        let denom = coeff_q.denom();
        let coeff_expr = if *denom == Integer::from(1) {
            pool.integer(Integer::from(numer.clone()))
        } else {
            pool.rational(Integer::from(numer.clone()), Integer::from(denom.clone()))
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
pub fn sum_definite(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, SumError> {
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

    let diff = simp(
        pool,
        pool.add(vec![upper, pool.mul(vec![lower, pool.integer(-1_i32)])]),
    );
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("gosper_definite_telescope", term, diff));
    Ok(DerivedExpr::with_log(diff, log))
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
    fn wz_pair_zero_is_certificate() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let z = pool.integer(0_i32);
        let pair = WzPair { f: z, g: z };
        assert!(verify_wz_pair(&pair, n, k, &pool));
    }
}
