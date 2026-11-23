//! Binomial differentials `∫ x^m·(a + b·x^n)^p dx` — Chebyshev's theorem.
//!
//! For rational `m, n, p` (`n ≠ 0`, `p ∉ ℤ`) and nonzero rational `a, b`,
//! Chebyshev (1853) proved the integral is elementary **iff** one of
//!
//! ```text
//!   p ∈ ℤ,      (m+1)/n ∈ ℤ,      (m+1)/n + p ∈ ℤ
//! ```
//!
//! holds.  With `u = x^n` (`dx = (1/n)·u^{1/n−1} du`) the integral becomes
//! `(1/n)·∫ u^q·(a + b·u)^p du`, `q = (m+1)/n − 1`, and the three cases are
//! exactly the ones the genus-0 parametrisation of a single linear-fractional
//! radical closes:
//!
//! * `q ∈ ℤ` — one radical `(a+bu)^p`;
//! * `q + p ∈ ℤ` — `u^{q+p}·((a+bu)/u)^p`, one radical of a Möbius radicand;
//! * (`p ∈ ℤ` is not a binomial differential in this module's sense.)
//!
//! The `u`-integral goes to the full engine, `u := x^n` is substituted back and
//! the result is gated by `d/dx F = f` against the original integrand.  In the
//! remaining case the integral is certified **non-elementary** — but only on
//! request of a caller that has already exhausted the elementary and
//! special-function routes, so a better answer (an elliptic closed form) is
//! never displaced by a refusal.

use rug::Rational;

use super::super::risch::poly_rde::is_free_of_var;
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

const U_NAME: &str = "$binom_u$";

/// `c·x^m·(a + b·x^n)^p` with every quantity rational.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Binomial {
    c: Rational,
    m: Rational,
    a: Rational,
    b: Rational,
    n: Rational,
    p: Rational,
}

fn rat_of(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

/// `x^r` (bare `x`, `sqrt(x)`, `x^r`) → `r`.
fn var_power(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<Rational> {
    if e == var {
        return Some(Rational::from(1));
    }
    match pool.get(e) {
        ExprData::Pow { base, exp } if base == var => rat_of(exp, pool),
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && args[0] == var =>
        {
            Some(Rational::from((1, 2)))
        }
        ExprData::Func { ref name, ref args }
            if name == "cbrt" && args.len() == 1 && args[0] == var =>
        {
            Some(Rational::from((1, 3)))
        }
        _ => None,
    }
}

/// `b·x^n` → `(b, n)`.
fn monomial(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<(Rational, Rational)> {
    if let Some(n) = var_power(e, var, pool) {
        return Some((Rational::from(1), n));
    }
    if let ExprData::Mul(args) = pool.get(e) {
        let mut b = Rational::from(1);
        let mut n: Option<Rational> = None;
        for &f in args.iter() {
            if let Some(r) = rat_of(f, pool) {
                b *= r;
            } else if let Some(k) = var_power(f, var, pool) {
                if n.replace(k).is_some() {
                    return None;
                }
            } else {
                return None;
            }
        }
        return n.map(|n| (b, n));
    }
    None
}

/// `a + b·x^n` → `(a, b, n)`.
fn binomial_base(
    e: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(Rational, Rational, Rational)> {
    let ExprData::Add(args) = pool.get(e) else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    let (c0, c1) = (args[0], args[1]);
    let (a, mono) = match (rat_of(c0, pool), rat_of(c1, pool)) {
        (Some(a), None) => (a, c1),
        (None, Some(a)) => (a, c0),
        _ => return None,
    };
    let (b, n) = monomial(mono, var, pool)?;
    (a != 0 && b != 0 && n != 0).then_some((a, b, n))
}

/// Recognise `c·x^m·(a + b·x^n)^p` with `p ∉ ℤ`.
pub(super) fn recognise(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<Binomial> {
    let factors: Vec<ExprId> = match pool.get(expr) {
        ExprData::Mul(args) => args.to_vec(),
        _ => vec![expr],
    };
    let mut c = Rational::from(1);
    let mut m = Rational::from(0);
    let mut base: Option<(Rational, Rational, Rational, Rational)> = None;
    for f in factors {
        if let Some(r) = rat_of(f, pool) {
            c *= r;
            continue;
        }
        if let Some(k) = var_power(f, var, pool) {
            m += k;
            continue;
        }
        let (inner, p) = match pool.get(f) {
            ExprData::Pow { base, exp } => (base, rat_of(exp, pool)?),
            ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
                (args[0], Rational::from((1, 2)))
            }
            ExprData::Func { ref name, ref args } if name == "cbrt" && args.len() == 1 => {
                (args[0], Rational::from((1, 3)))
            }
            _ => return None,
        };
        // A power of the bare binomial, or of `sqrt(binomial)`.
        let (inner, p) = match pool.get(inner) {
            ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
                (args[0], p / Rational::from(2))
            }
            _ => (inner, p),
        };
        let (a, b, n) = binomial_base(inner, var, pool)?;
        match &mut base {
            None => base = Some((a, b, n, p)),
            Some((a0, b0, n0, p0)) if *a0 == a && *b0 == b && *n0 == n => *p0 += p,
            Some(_) => return None,
        }
    }
    let (a, b, n, p) = base?;
    if p.is_integer() {
        return None;
    }
    Some(Binomial { c, m, a, b, n, p })
}

/// Chebyshev's elementarity condition.
pub(super) fn chebyshev_elementary(bi: &Binomial) -> bool {
    let k = (bi.m.clone() + Rational::from(1)) / bi.n.clone();
    bi.p.is_integer() || k.is_integer() || (k + bi.p.clone()).is_integer()
}

fn rexpr(r: &Rational, pool: &ExprPool) -> ExprId {
    super::super::risch::poly_rde::rational_to_expr(r, pool)
}

/// Integrate an elementary binomial differential by `u = x^n`.
pub(super) fn try_binomial(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    let u = pool.symbol(U_NAME, Domain::Real);
    if var == u {
        return None;
    }
    let bi = recognise(expr, var, pool)?;
    if !chebyshev_elementary(&bi) {
        return None;
    }
    let one = Rational::from(1);
    let q = (bi.m.clone() + &one) / bi.n.clone() - &one;
    let lin = pool.add(vec![
        rexpr(&bi.a, pool),
        pool.mul(vec![rexpr(&bi.b, pool), u]),
    ]);
    // (c/n)·u^q·(a+bu)^p, written with a single radical.
    let core = if q.is_integer() {
        pool.mul(vec![
            pool.pow(u, rexpr(&q, pool)),
            pool.pow(lin, rexpr(&bi.p, pool)),
        ])
    } else {
        let qp = q.clone() + &bi.p;
        if !qp.is_integer() {
            return None;
        }
        let frac = pool.mul(vec![lin, pool.pow(u, pool.integer(-1_i32))]);
        pool.mul(vec![
            pool.pow(u, rexpr(&qp, pool)),
            pool.pow(frac, rexpr(&bi.p, pool)),
        ])
    };
    let scale = bi.c.clone() / bi.n.clone();
    let integrand_u = simplify(pool.mul(vec![rexpr(&scale, pool), core]), pool).value;
    let inner = crate::integrate::engine::integrate(integrand_u, u, pool).ok()?;
    let mut map = std::collections::HashMap::new();
    map.insert(u, pool.pow(var, rexpr(&bi.n, pool)));
    let f = simplify(crate::kernel::subs::subs(inner.value, &map, pool), pool).value;
    if !is_free_of_var(f, u, pool) {
        return None;
    }
    // `G(x^n)` is right on `x > 0`.  For an even integer `n` it is right on
    // `x < 0` too when `f` is odd; when `f` is even the antiderivative must be
    // odd, i.e. `sign(x)·G(x^n)`, with the sign written as `x/√(x²)` (locally
    // constant, so its derivative vanishes off `x = 0`).  The two-sided gate
    // decides which, if either, is right.
    let f = if crate::integrate::engine::verify_antiderivative_status(f, expr, var, pool).is_some()
    {
        f
    } else if bi.n.is_integer() && bi.n.numer().is_even() {
        let sign = pool.mul(vec![
            var,
            pool.pow(
                pool.func("sqrt", vec![pool.pow(var, pool.integer(2_i32))]),
                pool.integer(-1_i32),
            ),
        ]);
        let g = simplify(pool.mul(vec![sign, f]), pool).value;
        crate::integrate::engine::verify_antiderivative_status(g, expr, var, pool)?;
        g
    } else {
        return None;
    };
    let mut log = DerivationLog::new();
    log = log.merge(inner.log);
    log.push(RewriteStep::simple("alg_binomial_chebyshev", expr, f));
    Some(DerivedExpr::with_log(f, log))
}

/// `Some(msg)` when `expr` is a binomial differential that Chebyshev's theorem
/// proves non-elementary.
pub(super) fn chebyshev_non_elementary(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<String> {
    let bi = recognise(expr, var, pool)?;
    if chebyshev_elementary(&bi) {
        return None;
    }
    Some(format!(
        "∫ x^m·(a + b·x^n)^p dx with m = {}, n = {}, p = {}: none of p, (m+1)/n, \
         (m+1)/n + p is an integer, so by Chebyshev's theorem on binomial \
         differentials the integral is not elementary",
        bi.m, bi.n, bi.p
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str, pool: &ExprPool) -> (ExprId, ExprId) {
        let x = pool.symbol("x", Domain::Real);
        let mut syms = std::collections::HashMap::from([("x".to_owned(), x)]);
        (crate::parse::parse(src, pool, &mut syms).unwrap(), x)
    }

    #[test]
    fn recognises_and_classifies() {
        let pool = ExprPool::new();
        let (e, x) = parse("x^(-2/3)*(1+x^(1/3))^(1/2)", &pool);
        let e = simplify(e, &pool).value;
        let bi = recognise(e, x, &pool).expect("binomial");
        assert!(chebyshev_elementary(&bi)); // (m+1)/n = 1
        let (e, x) = parse("(1+x^4)^(-1/4)", &pool);
        let bi = recognise(simplify(e, &pool).value, x, &pool).expect("binomial");
        assert!(chebyshev_elementary(&bi)); // (m+1)/n + p = 0
        let (e, x) = parse("x*sqrt(1+x^3)", &pool);
        let bi = recognise(simplify(e, &pool).value, x, &pool).expect("binomial");
        assert!(!chebyshev_elementary(&bi)); // 2/3, 7/6
    }

    #[test]
    fn integrates_elementary_cases() {
        // `(1+x⁴)^(-1/4)` is even, so its antiderivative is odd: `u = x⁴`
        // alone loses the sign of `x`, and the `x/√(x²)` factor restores it.
        for src in [
            "x^(-2/3)*(1+x^(1/3))^(1/2)",
            "x^(-2)*(1+x^2)^(-3/2)",
            "sqrt(1+x^3)/x",
        ] {
            let pool = ExprPool::new();
            let (e, x) = parse(src, &pool);
            let e = simplify(e, &pool).value;
            assert!(try_binomial(e, x, &pool).is_some(), "{src}");
        }
    }
}
