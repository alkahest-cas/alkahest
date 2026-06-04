//! Genus-0 reduction by rational parametrization — Risch milestone **M2**.
//!
//! A single radical generator `y = (a·x + b)^{1/n}` with a **linear** radicand
//! defines a genus-0 curve `yⁿ = a·x + b` with the trivial rational
//! parametrization `x = (sⁿ − b)/a`, `y = s`.  Substituting `dx = (n/a)·s^{n−1} ds`
//! turns `∫ R(x, y) dx` into `∫ R((sⁿ−b)/a, s)·(n/a)s^{n−1} ds`, an integrand that
//! is **rational in `s`** — hence always elementary and handled by the ordinary
//! rational/Risch engine.  Back-substituting `s = (a·x+b)^{1/n}` recovers the
//! antiderivative.
//!
//! This covers the cubic-and-higher radical genus-0 cases the simple-radical
//! integral part (MA) cannot finish — e.g. `∫ ∛x/(x+1) dx`, `∫ 1/(x·∛(x+1)) dx`,
//! `∫ ∛x/(1+∛x) dx` — including their **logarithmic part**, which MA omitted (and
//! for which it previously returned a *wrong* `NonElementary`).
//!
//! Scope: a single radical with radicand linear in `x` (any index `n ≥ 2`).
//! Higher-degree radicands (`yⁿ = p(x)`, `deg p ≥ 2`) are generally higher genus
//! and are out of scope here.  Sound by construction: the result is accepted only
//! after a numeric `d/dx F = integrand` check.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::integrate::risch::poly_rde::{
    degree, expr_to_qpoly, is_free_of_var, qpoly_to_expr, rational_to_expr,
};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use std::collections::HashMap;

/// Try the genus-0 parametrization of a single linear-radicand radical.  Returns
/// `None` when the integrand is not of this shape (caller falls through).
pub(super) fn try_parametrize_genus0(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, radicand) = detect_single_radical(expr, var, pool)?;
    let p = expr_to_qpoly(radicand, var, pool)?; // radicand as ℚ[x]
    if degree(&p) != 1 {
        return None; // only linear radicands are genus-0 by this parametrization
    }
    let b = p[0].clone();
    let a = p[1].clone();
    if a == 0 {
        return None;
    }

    // s = (a·x + b)^{1/n};  x = (sⁿ − b)/a;  dx = (n/a)·s^{n−1} ds.
    let s = pool.symbol("$param_s$", Domain::Real);
    let s_n = pool.pow(s, pool.integer(n as i32));
    let neg_b = rational_to_expr(&(-b.clone()), pool);
    let inv_a = rational_to_expr(&a.clone().recip(), pool);
    let x_of_s = pool.mul(vec![pool.add(vec![s_n, neg_b]), inv_a]);

    // Rewrite the integrand directly in `s`: every standalone `x` becomes `x(s)`
    // and every power `(a·x+b)^{c/d}` of the radicand becomes `s^{c·n/d}`, so no
    // un-reducible `(sⁿ)^{1/n}` is ever formed.  `None` if some subterm is not
    // expressible (e.g. another, unrelated radical).
    let core = to_s(expr, var, &p, n, s, x_of_s, pool)?;
    let n_over_a = rational_to_expr(&(rug::Rational::from(n as i64) * a.recip()), pool);
    let dx_ds = pool.mul(vec![n_over_a, pool.pow(s, pool.integer((n as i32) - 1))]);
    let integrand_s = simplify(pool.mul(vec![core, dx_ds]), pool).value;

    // Integrate the rational-in-`s` integrand (always elementary), then
    // back-substitute s = (a·x+b)^{1/n}.
    let f_s = match crate::integrate::engine::integrate(integrand_s, s, pool) {
        Ok(d) => d.value,
        Err(_) => return None,
    };
    let radical_expr = pool.pow(qpoly_to_expr(&p, var, pool), pool.rational(1, n as i32));
    let mut back = HashMap::new();
    back.insert(s, radical_expr);
    let f_x = simplify(crate::kernel::subs(f_s, &back, pool), pool).value;

    // Soundness gate: d/dx F = integrand numerically (where the radicand > 0).
    if !verify_derivative(f_x, expr, var, &p, n, pool) {
        return None;
    }

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "algebraic_genus0_parametrize",
        expr,
        f_x,
    ));
    Some(Ok(DerivedExpr::with_log(f_x, log)))
}

/// Find the unique `x`-dependent radical generator and return `(n, radicand)`.
/// `None` if there is no such generator or more than one distinct one.
fn detect_single_radical(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(usize, ExprId)> {
    let mut found: Vec<(usize, ExprId)> = Vec::new();
    scan(expr, var, pool, &mut found);
    let mut distinct: Vec<(usize, ExprId)> = Vec::new();
    for (n, r) in found {
        if !distinct.iter().any(|&(m, q)| m == n && q == r) {
            distinct.push((n, r));
        }
    }
    if distinct.len() == 1 {
        Some(distinct.remove(0))
    } else {
        None
    }
}

fn scan(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<(usize, ExprId)>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && !is_free_of_var(args[0], var, pool) =>
        {
            out.push((2, args[0]));
        }
        ExprData::Func { ref name, ref args }
            if name == "cbrt" && args.len() == 1 && !is_free_of_var(args[0], var, pool) =>
        {
            out.push((3, args[0]));
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if let Some(den) = r.0.denom().to_i64() {
                    if den >= 2 && !is_free_of_var(base, var, pool) {
                        out.push((den as usize, base));
                        return;
                    }
                }
            }
            scan(base, var, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan(a, var, pool, out);
            }
        }
        _ => {}
    }
}

/// Rewrite `expr` (rational in `x` and the single radical `(a·x+b)^{1/n}`) as a
/// rational function of `s`, where `s = (a·x+b)^{1/n}`: standalone `x → x(s)`, and
/// any power `(a·x+b)^{c/d}` of the radicand → `s^{c·n/d}`.  Returns `None` if a
/// subterm is not expressible this way (a different radical, a transcendental of
/// `x`, or a fractional power with `d ∤ c·n`).
#[allow(clippy::too_many_arguments)]
fn to_s(
    expr: ExprId,
    var: ExprId,
    p: &[rug::Rational],
    n: usize,
    s: ExprId,
    x_of_s: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if expr == var {
        return Some(x_of_s);
    }
    if is_free_of_var(expr, var, pool) {
        return Some(expr); // constant in x (incl. other symbols / numbers)
    }
    // `(radicand)^{c/d}` → `s^{c·n/d}` when that exponent is an integer.
    let radical_power = |base: ExprId, c: i64, d: i64, pool: &ExprPool| -> Option<ExprId> {
        let rb = expr_to_qpoly(base, var, pool)?;
        if degree(&rb) >= 1 && trim_eq(&rb, p) && (c * n as i64) % d == 0 {
            Some(pool.pow(s, pool.integer(((c * n as i64) / d) as i32)))
        } else {
            None
        }
    };
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            radical_power(args[0], 1, 2, pool)
        }
        ExprData::Func { ref name, ref args } if name == "cbrt" && args.len() == 1 => {
            radical_power(args[0], 1, 3, pool)
        }
        ExprData::Add(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_s(a, var, p, n, s, x_of_s, pool))
                .collect::<Option<_>>()?;
            Some(pool.add(v))
        }
        ExprData::Mul(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_s(a, var, p, n, s, x_of_s, pool))
                .collect::<Option<_>>()?;
            Some(pool.mul(v))
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let inner = to_s(base, var, p, n, s, x_of_s, pool)?;
                Some(pool.pow(inner, pool.integer(m.0.to_i64()? as i32)))
            }
            ExprData::Rational(r) => {
                radical_power(base, r.0.numer().to_i64()?, r.0.denom().to_i64()?, pool)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Equality of two `ℚ[x]` polynomials up to trailing zeros.
fn trim_eq(a: &[rug::Rational], b: &[rug::Rational]) -> bool {
    let z = rug::Rational::from(0);
    let la = a.iter().rposition(|c| *c != z).map_or(0, |i| i + 1);
    let lb = b.iter().rposition(|c| *c != z).map_or(0, |i| i + 1);
    la == lb && a[..la] == b[..lb]
}

/// Numeric `d/dx F = integrand` check at sample points where the radicand
/// `a·x + b > 0` (so the principal real branch of the radical is well defined).
fn verify_derivative(
    f: ExprId,
    integrand: ExprId,
    var: ExprId,
    p: &[rug::Rational],
    _n: usize,
    pool: &ExprPool,
) -> bool {
    let Ok(df) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    let a = p[1].to_f64();
    let b = p[0].to_f64();
    let mut checked = 0;
    for &xv in &[0.3_f64, 0.8, 1.6, 2.7, 3.9] {
        if a * xv + b <= 1e-6 {
            continue; // radicand not positive: skip
        }
        let (Some(lhs), Some(rhs)) = (eval(ds, var, xv, pool), eval(integrand, var, xv, pool))
        else {
            return false;
        };
        if !lhs.is_finite() || !rhs.is_finite() || (lhs - rhs).abs() > 1e-6 * (1.0 + rhs.abs()) {
            return false;
        }
        checked += 1;
    }
    checked >= 2
}

fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> Option<f64> {
    if expr == x {
        return Some(xv);
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => Some(r.0.to_f64()),
        ExprData::Add(args) => args
            .iter()
            .try_fold(0.0, |s, &a| Some(s + eval(a, x, xv, pool)?)),
        ExprData::Mul(args) => args
            .iter()
            .try_fold(1.0, |s, &a| Some(s * eval(a, x, xv, pool)?)),
        ExprData::Pow { base, exp } => Some(eval(base, x, xv, pool)?.powf(eval(exp, x, xv, pool)?)),
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let a = eval(args[0], x, xv, pool)?;
            match name.as_str() {
                "exp" => Some(a.exp()),
                "log" => Some(a.ln()),
                "sqrt" => Some(a.sqrt()),
                "cbrt" => Some(a.cbrt()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(build: impl Fn(&ExprPool, ExprId) -> ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = build(&pool, x);
        let r = crate::integrate::engine::integrate(f, x, &pool);
        assert!(r.is_ok(), "expected elementary; got {r:?}");
        // d/dx F = f at a positive sample.
        let g = r.unwrap().value;
        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in &[1.3_f64, 2.4, 3.7] {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(f, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }
    }

    #[test]
    fn cbrt_x_over_x_plus_1() {
        // ∫ ∛x/(x+1) dx — elementary (was wrongly NonElementary before M2).
        check(|p, x| {
            let num = p.func("cbrt", vec![x]);
            let den = p.add(vec![x, p.integer(1)]);
            p.mul(vec![num, p.pow(den, p.integer(-1))])
        });
    }

    #[test]
    fn one_over_x_cbrt_x_plus_1() {
        // ∫ 1/(x·∛(x+1)) dx.
        check(|p, x| {
            let xp1 = p.add(vec![x, p.integer(1)]);
            let cb = p.func("cbrt", vec![xp1]);
            p.pow(p.mul(vec![x, cb]), p.integer(-1))
        });
    }

    #[test]
    fn cbrt_x_over_one_plus_cbrt_x() {
        // ∫ ∛x/(1+∛x) dx.
        check(|p, x| {
            let cb = p.func("cbrt", vec![x]);
            let den = p.add(vec![p.integer(1), cb]);
            p.mul(vec![cb, p.pow(den, p.integer(-1))])
        });
    }

    #[test]
    fn x_two_thirds() {
        // ∫ x^(2/3) dx = (3/5) x^(5/3).
        check(|p, x| p.pow(x, p.rational(2, 3)));
    }

    #[test]
    fn fifth_root_of_linear() {
        // ∫ (2x+1)^(1/5) dx = (5/12)(2x+1)^(6/5).
        check(|p, x| {
            let lin = p.add(vec![p.mul(vec![p.integer(2), x]), p.integer(1)]);
            p.pow(lin, p.rational(1, 5))
        });
    }
}
