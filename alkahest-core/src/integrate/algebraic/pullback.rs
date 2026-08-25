//! The **power pullback** `u = x^k`: integrands of the shape
//! `f(x) = x^{k−1}·g(x^k)`.
//!
//! # What it is for
//!
//! `∫x dx/√(1−x⁴)` is `½asin(x²)`, and every route in this module used to
//! decline it — one of them by *certifying it non-elementary*.  The reason is
//! structural rather than accidental: `y² = 1−x⁴` is a genus-1 curve, the
//! differential `x dx/y` is a pure logarithmic differential whose residues
//! `±i` live at the two places over `∞`, and getting at them needs the residue
//! machinery to work over `ℚ(i)`.  Under `u = x²` the same integral is
//! `½∫du/√(1−u²)` on a **genus-0** curve, which the `arcsin` route already
//! closes in one step.
//!
//! The substitution is available whenever `f` is invariant in the right sense:
//! `f(x) = x^{k−1}·g(x^k)` gives `x^{k−1}dx = du/k`, hence
//!
//! ```text
//!     ∫f(x) dx = (1/k)·∫g(u) du,        u = x^k.
//! ```
//!
//! Since `u = x^k` and `x = u^{1/k}` are both algebraic, elementarity transfers
//! **both** ways: `∫g du` elementary ⟺ `∫f dx` elementary.  Only the forward
//! direction is used here (a verdict is never imported), which keeps this route
//! incapable of contradicting a certificate.
//!
//! # How the shape is recognised
//!
//! [`split`] rewrites the expression tree bottom-up into the normal form
//! `expr = x^r · G(u)` with `0 ≤ r < k` — a bare `x` contributes `r = 1`,
//! `x^{jk+r'}` folds `j` copies of `u` out, a sum insists its summands agree on
//! `r`, and anything under a fractional power or a function call must already
//! have `r = 0`.  Reaching `r = k−1` with `G` free of `x` *is* the shape.  This
//! is exact structural matching, not a numeric fit: it cannot report a match
//! that is not there.
//!
//! # Where it sits, and what it will not do
//!
//! It runs **last**, and only when the algebraic engine returned
//! `NotImplemented`.  Two consequences, both deliberate:
//!
//! * No currently-solved integral changes form.  The genus-0/1 machinery
//!   upstream produces better-shaped answers when it can, and keeps them.
//! * It never overrides a `NonElementary` verdict.  A sound certificate is a
//!   theorem, and a route that could talk one down would be a route that could
//!   talk a *correct* one down.  The practical cost is that
//!   `∫x dx/√(1−x⁶)` — genuinely non-elementary, and now certified so — is not
//!   re-expressed as `½·EllipticF(…)` even though the pullback could.
//!
//! Every emission is gated on a numeric `d/dx F = f` check against the
//! **original** integrand in `x`, so a mis-recognised shape can only decline.

use std::collections::HashMap;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::risch::poly_rde::is_free_of_var;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// Substituted variable.  The `$…$` fencing matches the convention in
/// `parametrize` (`$param_s$`) so it cannot collide with a user symbol, and
/// reusing one name across nesting levels is what makes the recursion guard in
/// [`try_power_pullback`] a single equality test.
const U_NAME: &str = "$pullback_u$";

/// Largest `k` considered.  `u = x^k` divides the degree of the radicand by
/// `k`, so the interesting range is small; the bound only stops a pathological
/// expression from driving a long search.
const MAX_K: i64 = 12;

/// Try `∫f dx = (1/k)·∫g(u) du` with `u = x^k`, for `f = x^{k−1}·g(x^k)`.
///
/// `None` means the shape does not apply (the caller keeps its own verdict);
/// `Some(Ok(..))` is a gate-verified antiderivative.  Never returns
/// `Some(Err(..))` — this route has no verdict of its own to report.
pub(super) fn try_power_pullback(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    let u = pool.symbol(U_NAME, Domain::Real);
    if var == u {
        return None; // already inside a pullback: do not nest
    }
    // This route exists to drop the genus of an *algebraic curve*.  Without a
    // radical whose radicand moves with `x` there is no curve to drop, and the
    // shape test alone would fire on things like `∫dx/(x√2)` — where `1/x` is
    // trivially `x^{k−1}·u^{−1}` for every `k` — and answer `log(x¹²)/12`.
    if !has_variable_radical(expr, var, pool) {
        return None;
    }

    // Largest `k` first: it drops the genus furthest, and taking the maximum is
    // also what keeps the substituted integrand from matching the same pattern
    // again.
    for k in (2..=MAX_K).rev() {
        let Some((g, r)) = split(expr, var, u, k, pool) else {
            continue;
        };
        if r != k - 1 || !is_free_of_var(g, var, pool) {
            continue;
        }
        let g = simplify(g, pool).value;
        if is_free_of_var(g, u, pool) && !is_free_of_var(expr, var, pool) {
            continue; // degenerate match: nothing was actually pulled back
        }
        let Ok(inner) = crate::integrate::engine::integrate(g, u, pool) else {
            continue;
        };

        // F(x) = (1/k)·H(x^k).
        let mut map = HashMap::new();
        map.insert(u, pool.pow(var, pool.integer(k as i32)));
        let back = crate::kernel::subs::subs(inner.value, &map, pool);
        let scaled = pool.mul(vec![
            pool.pow(pool.integer(k as i32), pool.integer(-1_i32)),
            back,
        ]);
        let f = simplify(scaled, pool);

        if !verify(f.value, expr, var, pool) {
            continue;
        }
        let mut log = DerivationLog::new();
        log = log.merge(inner.log);
        log = log.merge(f.log.clone());
        log.push(RewriteStep::simple("alg_power_pullback", expr, f.value));
        return Some(DerivedExpr::with_log(f.value, log));
    }
    None
}

/// Rewrite `expr` as `x^r · G(u)` with `u = x^k` and `0 ≤ r < k`, returning
/// `(G, r)`.  `None` when no such split exists.
///
/// The recursion is the whole recogniser: every construct either consumes its
/// children's residual `x`-degree (products and integer powers, which can fold
/// whole multiples of `k` into `u`) or demands that it be zero (fractional
/// powers, function arguments), and sums demand agreement.
fn split(expr: ExprId, var: ExprId, u: ExprId, k: i64, pool: &ExprPool) -> Option<(ExprId, i64)> {
    let one = pool.integer(1_i32);
    if expr == var {
        return Some((one, 1));
    }
    if is_free_of_var(expr, var, pool) {
        return Some((expr, 0));
    }
    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut parts = Vec::with_capacity(args.len());
            let mut deg: Option<i64> = None;
            for a in args.iter() {
                let (g, r) = split(*a, var, u, k, pool)?;
                if *deg.get_or_insert(r) != r {
                    return None; // summands disagree on the residual x-degree
                }
                parts.push(g);
            }
            Some((pool.add(parts), deg.unwrap_or(0)))
        }
        ExprData::Mul(args) => {
            let mut parts = Vec::with_capacity(args.len() + 1);
            let mut total = 0_i64;
            for a in args.iter() {
                let (g, r) = split(*a, var, u, k, pool)?;
                parts.push(g);
                total = total.checked_add(r)?;
            }
            let (q, r) = divmod(total, k);
            if q != 0 {
                parts.push(pool.pow(u, pool.integer(i32::try_from(q).ok()?)));
            }
            Some((pool.mul(parts), r))
        }
        ExprData::Pow { base, exp } => {
            let (g, rb) = split(base, var, u, k, pool)?;
            match pool.get(exp) {
                ExprData::Integer(n) => {
                    let n = n.0.to_i64()?;
                    let total = rb.checked_mul(n)?;
                    let (q, r) = divmod(total, k);
                    let mut parts = vec![pool.pow(g, exp)];
                    if q != 0 {
                        parts.push(pool.pow(u, pool.integer(i32::try_from(q).ok()?)));
                    }
                    Some((pool.mul(parts), r))
                }
                // A fractional (or symbolic) exponent cannot absorb a residual
                // `x`: `(x^r·G)^{p/q}` is not `x^{integer}·(…)`.
                _ if rb == 0 => Some((pool.pow(g, exp), 0)),
                _ => None,
            }
        }
        ExprData::Func { ref name, ref args } => {
            let mut parts = Vec::with_capacity(args.len());
            for a in args.iter() {
                let (g, r) = split(*a, var, u, k, pool)?;
                if r != 0 {
                    return None; // `f(x^r·G)` is not a function of `u` alone
                }
                parts.push(g);
            }
            Some((pool.func(name.clone(), parts), 0))
        }
        _ => None,
    }
}

/// Does `expr` contain a radical `base^{p/q}` (`q > 1`) whose radicand depends
/// on `var`?
fn has_variable_radical(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if is_free_of_var(expr, var, pool) {
        return false;
    }
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if (name == "sqrt" || name == "cbrt") && args.len() == 1 =>
        {
            if !is_free_of_var(args[0], var, pool) {
                return true;
            }
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.denom() != 1 && !is_free_of_var(base, var, pool) {
                    return true;
                }
            }
        }
        _ => {}
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().any(|&c| has_variable_radical(c, var, pool))
        }
        ExprData::Pow { base, exp } => {
            has_variable_radical(base, var, pool) || has_variable_radical(exp, var, pool)
        }
        _ => false,
    }
}

/// Euclidean division with a non-negative remainder: `n = q·k + r`, `0 ≤ r < k`.
fn divmod(n: i64, k: i64) -> (i64, i64) {
    let mut q = n / k;
    let mut r = n % k;
    if r < 0 {
        r += k;
        q -= 1;
    }
    (q, r)
}

/// Numeric gate: `d/dx F = f` at real samples where both sides evaluate.
///
/// Uses the interpreter over the primitive registry, so an `asin`/`atan`/
/// `EllipticF` answer is checked as readily as a logarithmic one.  At least
/// three agreeing samples are required, and any disagreement — not just a
/// shortage of samples — rejects.
fn verify(f: ExprId, integrand: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let Ok(df) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let d = simplify(df.value, pool).value;
    let mut checked = 0;
    // Spread across both `|x| < 1` and `|x| > 1`: `√(1−x⁴)` is real only on the
    // first, `√(x⁴−1)` only on the second, and three usable samples are needed
    // either way.
    for &xv in &[
        0.13_f64, 0.29, 0.41, 0.57, 0.68, 0.83, 1.15, 1.4, 1.7, 2.3, 3.1, 4.2,
    ] {
        let mut env = HashMap::new();
        env.insert(var, xv);
        let (Some(lhs), Some(rhs)) = (
            crate::jit::eval_interp(d, &env, pool),
            crate::jit::eval_interp(integrand, &env, pool),
        ) else {
            continue;
        };
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        if (lhs - rhs).abs() > 1e-7 * (1.0 + rhs.abs()) {
            return false;
        }
        checked += 1;
    }
    checked >= 3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::engine::{integrate, IntegrationError};

    fn parse(src: &str, pool: &ExprPool) -> ExprId {
        let mut syms = HashMap::new();
        let x = pool.symbol("x", Domain::Real);
        syms.insert("x".to_string(), x);
        crate::parse::parse(src, pool, &mut syms).expect("parse")
    }

    /// `d/dx F = f` at samples inside the real domain of the integrand.
    fn assert_antiderivative(src: &str, samples: &[f64]) -> String {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = parse(src, &pool);
        let res = integrate(f, x, &pool).unwrap_or_else(|e| panic!("{src}: {e:?}"));
        let d = simplify(crate::diff::diff(res.value, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for &xv in samples {
            let mut env = HashMap::new();
            env.insert(x, xv);
            let (Some(lhs), Some(rhs)) = (
                crate::jit::eval_interp(d, &env, &pool),
                crate::jit::eval_interp(f, &env, &pool),
            ) else {
                continue;
            };
            assert!(
                (lhs - rhs).abs() < 1e-7 * (1.0 + rhs.abs()),
                "{src}: d/dx F = {lhs} ≠ {rhs} = f at x = {xv}"
            );
            checked += 1;
        }
        assert!(checked >= 3, "{src}: only {checked} samples evaluated");
        pool.display(res.value).to_string()
    }

    /// The family in the bug report: `∫x^{k−1}/√(1−x^{2k}) = (1/k)asin(x^k)`.
    #[test]
    fn asin_pullback_family() {
        for (k, src) in [
            (2, "x/sqrt(1-x^4)"),
            (3, "x^2/sqrt(1-x^6)"),
            (4, "x^3/sqrt(1-x^8)"),
        ] {
            let out = assert_antiderivative(src, &[0.2, 0.4, 0.6, 0.75]);
            assert!(!out.is_empty(), "k = {k}");
        }
    }

    /// Scaled radicands: `∫x dx/√(c−x⁴) = ½asin(x²/√c)`.
    ///
    /// `∫x dx/√(1−4x⁴)` is *not* here: the pulled-back `∫du/√(1−4u²)` is a gap
    /// in the genus-0 quadratic route itself (a non-unit leading coefficient),
    /// unrelated to the pullback.  It declines honestly (`E-INT-001`) where it
    /// used to be certified non-elementary.
    #[test]
    fn asin_pullback_scaled() {
        assert_antiderivative("x/sqrt(4-x^4)", &[0.2, 0.5, 0.9, 1.2]);
        assert_antiderivative("x/sqrt(9-x^4)", &[0.3, 0.7, 1.1, 1.5]);
    }

    /// The `has_variable_radical` gate: a constant radicand is not a curve, so
    /// `∫dx/(x√2)` must not be "pulled back" to `log(x¹²)/(12√2)`.
    #[test]
    fn refuses_a_constant_radicand() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = parse("1/(x*sqrt(2))", &pool);
        assert!(!has_variable_radical(f, x, &pool));
        assert!(try_power_pullback(f, x, &pool).is_none());
    }

    /// The `asinh` half of the family — `∫x dx/√(1+x⁴)`, previously an honest
    /// decline rather than a false certificate, and now closed.
    #[test]
    fn asinh_pullback() {
        assert_antiderivative("x/sqrt(1+x^4)", &[0.2, 0.7, 1.3, 2.1]);
        assert_antiderivative("x^2/sqrt(1+x^6)", &[0.2, 0.7, 1.3, 2.1]);
    }

    /// Second-kind weights pull back too: `∫x√(1−x⁴) dx`.
    #[test]
    fn second_kind_pullback() {
        assert_antiderivative("x*sqrt(1-x^4)", &[0.2, 0.4, 0.6, 0.8]);
        assert_antiderivative("x*sqrt(1+x^4)", &[0.2, 0.9, 1.4, 2.2]);
        assert_antiderivative("x^3*sqrt(1-x^8)", &[0.2, 0.4, 0.6, 0.8]);
    }

    /// `split` is exact structural matching.
    #[test]
    fn split_recognises_and_refuses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let u = pool.symbol(U_NAME, Domain::Real);

        // x·(1 − x⁴)^{−1/2} = x¹·G(u), u = x².
        let f = parse("x/sqrt(1-x^4)", &pool);
        let (g, r) = split(f, x, u, 2, &pool).expect("k = 2 splits");
        assert_eq!(r, 1);
        assert!(is_free_of_var(g, x, &pool));
        // …but not with k = 4: the residual degree is 1, not 3.
        assert_eq!(split(f, x, u, 4, &pool).map(|(_, r)| r), Some(1));

        // A bare `x` under the radical blocks the substitution.
        let bad = parse("x/sqrt(1-x^3)", &pool);
        assert_eq!(split(bad, x, u, 2, &pool).map(|(_, r)| r), None);

        // Summands must agree on the residual degree.
        let mixed = parse("x + x^2", &pool);
        assert!(split(mixed, x, u, 2, &pool).is_none());
    }

    /// Negative powers fold correctly: `1/x = x^{k−1}·u^{−1}` for `k = 2`.
    #[test]
    fn split_handles_negative_exponents() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let u = pool.symbol(U_NAME, Domain::Real);
        let f = parse("1/x", &pool);
        let (g, r) = split(f, x, u, 2, &pool).expect("splits");
        assert_eq!(r, 1);
        assert!(is_free_of_var(g, x, &pool));
    }

    /// The route declines rather than nesting when handed its own variable.
    #[test]
    fn refuses_to_nest() {
        let pool = ExprPool::new();
        let u = pool.symbol(U_NAME, Domain::Real);
        let f = pool.mul(vec![u, pool.pow(u, pool.integer(2_i32))]);
        assert!(try_power_pullback(f, u, &pool).is_none());
    }

    /// `∫x dx/√(1−x⁶)` is genuinely non-elementary (`½∫du/√(1−u³)` is an
    /// elliptic integral of the first kind), and the pullback must not talk
    /// that certificate down.
    #[test]
    fn does_not_override_a_certificate() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = parse("x/sqrt(1-x^6)", &pool);
        let res = integrate(f, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "∫x dx/√(1−x⁶) must stay NonElementary; got {res:?}"
        );
    }
}
