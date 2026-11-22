//! Limits at `+∞` of expressions built from `gamma`, `loggamma`, `digamma`
//! and `polygamma` (hence `factorial` and `binomial`, which parse to gamma).
//!
//! Each special function is replaced by an *asymptotically equivalent*
//! elementary expression, and the limit of the result is computed by the rest
//! of the engine (Gruntz, for the `exp`/`log` shapes Stirling produces):
//!
//! | function, `u → +∞` | replacement | error |
//! |---|---|---|
//! | `gamma(u)^c` | `exp(c·S(u))`, `S(u) = (u − ½)log u − u + ½log 2π` | factor `exp(c·O(1/u)) → 1` |
//! | `loggamma(u)` | `S(u)` | `O(1/u)` additive; factor `→ 1` since `S → ∞` |
//! | `digamma(u)` | `log u` | `O(1/u)` additive; factor `→ 1` |
//! | `polygamma(m, u)`, `m ≥ 1` | `(−1)^{m+1}(m − 1)!·u^{−m}` | factor `→ 1` |
//!
//! The replacement is only sound where the error provably vanishes in the
//! limit, so the shapes are restricted:
//!
//! * **multiplicative position** — the expression is a product of factors
//!   (and constant powers of products). `lim f·g = lim f'·g` when `f/f' → 1`,
//!   whatever `g` does (including `±∞` and `0`). Every special factor must
//!   have its argument tend to `+∞` and its exponent free of the variable
//!   (`Γ(x)^x` is refused: `x·O(1/x)` does not vanish).
//! * **additive position** — at the top level only, a term `c·loggamma(u)` or
//!   `c·digamma(u)` (`c` free of the variable) may be replaced by its
//!   additive approximation, whose error `c·O(1/u) → 0`.
//!
//! Anything else — a gamma inside a sum term (`Γ(x+1)/Γ(x) − x`, whose answer
//! lives in the `O(1/u)` terms), inside a function, or with a non-constant
//! exponent — is not rewritten, and the caller falls through to its other
//! rules.

use crate::kernel::subs::mentions_var;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

const SPECIAL: [&str; 4] = ["gamma", "loggamma", "digamma", "polygamma"];

/// Does `expr` mention one of the four functions with `var` in an argument?
pub(crate) fn mentions_gamma_family(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            (SPECIAL.contains(&name.as_str()) && args.iter().any(|&a| mentions_var(a, var, pool)))
                || args.iter().any(|&a| mentions_gamma_family(a, var, pool))
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            xs.iter().any(|&x| mentions_gamma_family(x, var, pool))
        }
        ExprData::Pow { base, exp } => {
            mentions_gamma_family(base, var, pool) || mentions_gamma_family(exp, var, pool)
        }
        _ => false,
    }
}

/// `S(u) = (u − ½)·log u − u + ½·log(2π)`.
fn stirling_log(u: ExprId, pool: &ExprPool) -> ExprId {
    let half = pool.rational(1, 2);
    let neg_half = pool.rational(-1, 2);
    let pi = pool.symbol("pi", Domain::Real);
    pool.add(vec![
        pool.mul(vec![pool.add(vec![u, neg_half]), pool.func("log", vec![u])]),
        pool.mul(vec![pool.integer(-1_i32), u]),
        pool.mul(vec![
            half,
            pool.func("log", vec![pool.mul(vec![pool.integer(2_i32), pi])]),
        ]),
    ])
}

/// `log(P(x))` for a polynomial `P` over ℚ of degree `d ≥ 1` with positive
/// leading coefficient `c` → `log c + d·log x + log(P(x)/(c·x^d))`, recursively.
///
/// An identity for `x > 0`. Stirling's exponents are differences such as
/// `x·log(x + ½) − x·log x`, whose cancellation is invisible until the `x·log x`
/// parts are split off; afterwards the remainder `x·log(1 + 1/(2x))` is an
/// ordinary power-scale limit.
pub(crate) fn split_polynomial_logs(e: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    use crate::calculus::asymptotic_common::{as_rational_function, qp_trim, rational_to_expr};
    if !mentions_var(e, var, pool) {
        return e;
    }
    match pool.get(e) {
        ExprData::Func { name, args } if name == "log" && args.len() == 1 => {
            let u = args[0];
            let Some(q) = as_rational_function(u, var, pool) else {
                return e;
            };
            let num = qp_trim(q.num);
            let den = qp_trim(q.den);
            if den.len() != 1 || num.len() < 2 || den[0] == 0 {
                return e;
            }
            let d = num.len() - 1;
            let lc = num[d].clone() / den[0].clone();
            if lc <= 0 {
                return e;
            }
            let scale = den[0].clone() * &lc;
            let rest: Vec<ExprId> = num
                .iter()
                .enumerate()
                .filter(|(_, c)| **c != 0)
                .map(|(i, c)| {
                    let coef = rational_to_expr(&(c.clone() / scale.clone()), pool);
                    let p = i as i64 - d as i64;
                    if p == 0 {
                        coef
                    } else {
                        pool.mul(vec![coef, pool.pow(var, pool.integer(p))])
                    }
                })
                .collect();
            pool.add(vec![
                pool.func("log", vec![rational_to_expr(&lc, pool)]),
                pool.mul(vec![pool.integer(d as i64), pool.func("log", vec![var])]),
                pool.func("log", vec![pool.add(rest)]),
            ])
        }
        ExprData::Add(xs) => pool.add(
            xs.iter()
                .map(|&x| split_polynomial_logs(x, var, pool))
                .collect(),
        ),
        ExprData::Mul(xs) => pool.mul(
            xs.iter()
                .map(|&x| split_polynomial_logs(x, var, pool))
                .collect(),
        ),
        _ => e,
    }
}

/// Rewrites the gamma family into elementary asymptotic equivalents; see the
/// module documentation. `goes_to_infinity(u)` decides whether an argument
/// tends to `+∞` (the caller's own limit routine).
pub(crate) struct Rewriter<'a, F: FnMut(ExprId) -> bool> {
    pub(crate) var: ExprId,
    pub(crate) pool: &'a ExprPool,
    pub(crate) goes_to_infinity: F,
}

impl<F: FnMut(ExprId) -> bool> Rewriter<'_, F> {
    fn free(&self, e: ExprId) -> bool {
        !mentions_var(e, self.var, self.pool)
    }

    fn clean(&self, e: ExprId) -> bool {
        !mentions_gamma_family(e, self.var, self.pool)
    }

    /// The whole expression: a top-level sum gets the additive rules for its
    /// `c·loggamma(u)` / `c·digamma(u)` terms, anything else the
    /// multiplicative ones.
    pub(crate) fn top(&mut self, e: ExprId) -> Option<ExprId> {
        let pool = self.pool;
        if let ExprData::Add(xs) = pool.get(e) {
            let mut out = Vec::with_capacity(xs.len());
            for &t in xs.iter() {
                if self.clean(t) {
                    out.push(t);
                    continue;
                }
                out.push(self.additive_term(t)?);
            }
            return Some(pool.add(out));
        }
        self.factor(e)
    }

    /// `c·loggamma(u)` → `c·S(u)`, `c·digamma(u)` → `c·log u`.
    fn additive_term(&mut self, t: ExprId) -> Option<ExprId> {
        let pool = self.pool;
        let factors: Vec<ExprId> = match pool.get(t) {
            ExprData::Mul(xs) => xs.to_vec(),
            _ => vec![t],
        };
        let mut out = Vec::with_capacity(factors.len());
        let mut special = 0;
        for f in factors {
            if self.free(f) {
                out.push(f);
                continue;
            }
            let ExprData::Func { name, args } = pool.get(f) else {
                return None;
            };
            if args.len() != 1 || !(self.goes_to_infinity)(args[0]) {
                return None;
            }
            let u = args[0];
            match name.as_str() {
                "loggamma" => out.push(stirling_log(u, pool)),
                "digamma" => out.push(pool.func("log", vec![u])),
                _ => return None,
            }
            special += 1;
        }
        (special == 1).then(|| pool.mul(out))
    }

    /// Multiplicative position: an asymptotic equivalent of `e`.
    fn factor(&mut self, e: ExprId) -> Option<ExprId> {
        let pool = self.pool;
        if self.clean(e) {
            return Some(e);
        }
        match pool.get(e) {
            ExprData::Mul(xs) => {
                let mut out = Vec::with_capacity(xs.len());
                for &x in xs.iter() {
                    out.push(self.factor(x)?);
                }
                Some(pool.mul(out))
            }
            ExprData::Pow { base, exp } => {
                if !self.free(exp) || !self.clean(exp) {
                    return None;
                }
                if let ExprData::Func { name, args } = pool.get(base) {
                    if name == "gamma" && args.len() == 1 {
                        let u = args[0];
                        if !(self.goes_to_infinity)(u) {
                            return None;
                        }
                        let arg = pool.mul(vec![exp, stirling_log(u, pool)]);
                        return Some(pool.func("exp", vec![arg]));
                    }
                }
                let b = self.factor(base)?;
                Some(pool.pow(b, exp))
            }
            ExprData::Func { name, args } => {
                let (u, m) = match (name.as_str(), args.as_slice()) {
                    ("gamma" | "loggamma" | "digamma", [u]) => (*u, None),
                    ("polygamma", [m, u]) => (*u, Some(*m)),
                    _ => return None,
                };
                if !(self.goes_to_infinity)(u) {
                    return None;
                }
                Some(match (name.as_str(), m) {
                    ("gamma", _) => pool.func("exp", vec![stirling_log(u, pool)]),
                    ("loggamma", _) => stirling_log(u, pool),
                    ("digamma", _) => pool.func("log", vec![u]),
                    (_, Some(m)) => {
                        let ExprData::Integer(mi) = pool.get(m) else {
                            return None;
                        };
                        let mi = mi.0.to_u32().filter(|&v| (1..=64).contains(&v))?;
                        // (−1)^{m+1} (m − 1)! u^{−m}
                        let mut fact = rug::Integer::from(1);
                        for j in 2..mi {
                            fact *= j;
                        }
                        if mi % 2 == 0 {
                            fact = -fact;
                        }
                        pool.mul(vec![
                            pool.integer(fact),
                            pool.pow(u, pool.integer(-i64::from(mi))),
                        ])
                    }
                    _ => return None,
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::calculus::limits::{limit, LimitDirection};
    use crate::kernel::{Domain, ExprId, ExprPool};

    fn lim(src: &str) -> Result<String, String> {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = std::collections::HashMap::from([("x".to_string(), x)]);
        let e: ExprId = crate::parse::parse(src, &pool, &mut syms).unwrap();
        limit(
            e,
            x,
            pool.pos_infinity(),
            LimitDirection::Bidirectional,
            &pool,
        )
        .map(|v| pool.display(v).to_string())
        .map_err(|e| e.to_string())
    }

    #[test]
    fn gamma_half_ratio() {
        assert_eq!(lim("gamma(x+1/2)/(gamma(x)*sqrt(x))").unwrap(), "1");
    }

    #[test]
    fn gamma_shift_ratio() {
        assert_eq!(lim("gamma(x+2)/(gamma(x)*x^2)").unwrap(), "1");
        assert_eq!(lim("gamma(x)/gamma(x+1)").unwrap(), "0");
    }

    #[test]
    fn loggamma_and_digamma_growth() {
        assert_eq!(lim("loggamma(x)/(x*log(x))").unwrap(), "1");
        assert_eq!(lim("digamma(x)/log(x)").unwrap(), "1");
    }

    #[test]
    fn factorial_parses_to_gamma() {
        assert_eq!(lim("factorial(x)/gamma(x+1)").unwrap(), "1");
    }

    #[test]
    fn gamma_in_a_sum_term_is_not_rewritten() {
        // Γ(x+1)/Γ(x) − x = 0; the leading Stirling term alone would give a
        // non-zero constant, so a wrong rewrite would show here.
        if let Ok(v) = lim("gamma(x+1)/gamma(x) - x") {
            assert_eq!(v, "0");
        }
    }
}
