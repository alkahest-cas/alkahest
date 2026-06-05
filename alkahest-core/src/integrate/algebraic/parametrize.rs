//! Genus-0 reduction by rational parametrization — Risch milestones **M2 / MC0**.
//!
//! A single radical generator `y = r(x)^{1/n}` whose radicand `r` is a
//! **linear-fractional** function `r = (a₁x+a₀)/(b₁x+b₀)` (numerator and
//! denominator each of degree ≤ 1) defines a genus-0 curve `yⁿ = r(x)`.  Solving
//! `sⁿ = r(x)` for `x` gives the rational parametrization
//!
//! ```text
//!   x(s) = (a₀ − b₀·sⁿ) / (b₁·sⁿ − a₁),   y = s,
//! ```
//! and substituting `dx = x'(s) ds` turns `∫ R(x, y) dx` into an integrand that is
//! **rational in `s`** — hence always elementary and handled by the ordinary
//! rational/Risch engine.  Back-substituting `s = r(x)^{1/n}` recovers the
//! antiderivative.  The pure polynomial-linear case (`b₁ = 0`, M2) — `∛x/(x+1)`,
//! `x^{2/3}`, … — and the genuinely fractional case (MC0) — `√((1−x)/(1+x))`,
//! `∛((x+1)/(x−1))`, … — are the same formula.
//!
//! This covers the cubic-and-higher radical genus-0 cases the simple-radical
//! integral part (MA) cannot finish, **including their logarithmic part** (which
//! MA omitted — previously returning a *wrong* `NonElementary` for e.g.
//! `∫ ∛x/(x+1) dx`).
//!
//! Scope: a single radical with linear-fractional radicand (any index `n ≥ 2`).
//! Radicands with ≥ 2 distinct finite zeros/poles (`yⁿ = p(x)`, `deg ≥ 2`,
//! non-Möbius) are generally higher genus and out of scope.  Sound by
//! construction: the result is accepted only after a numeric `d/dx F = integrand`
//! check.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::integrate::risch::poly_rde::{degree, is_free_of_var, poly_mul, rational_to_expr, trim};
use crate::integrate::risch::rational_rde::expr_to_qrational;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use std::collections::HashMap;

type QPoly = Vec<rug::Rational>;

/// Try the genus-0 parametrization of a single linear-fractional-radicand
/// radical.  Returns `None` when the integrand is not of this shape (caller falls
/// through).
pub(super) fn try_parametrize_genus0(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, radicand) = detect_single_radical(expr, var, pool)?;
    // Radicand as a reduced fraction num/den, each of degree ≤ 1.
    let (num, den) = expr_to_qrational(radicand, var, pool)?;
    let (num, den) = (trim(num), trim(den));
    if degree(&num) > 1 || degree(&den) > 1 || (degree(&num) < 1 && degree(&den) < 1) {
        return None;
    }
    let coeff = |p: &QPoly, i: usize| p.get(i).cloned().unwrap_or_else(|| rug::Rational::from(0));
    let (a0, a1) = (coeff(&num, 0), coeff(&num, 1));
    let (b0, b1) = (coeff(&den, 0), coeff(&den, 1));

    // s = r(x)^{1/n};  x(s) = (a₀ − b₀·sⁿ)/(b₁·sⁿ − a₁).
    let s = pool.symbol("$param_s$", Domain::Real);
    let s_n = pool.pow(s, pool.integer(n as i32));
    let lin = |c1: &rug::Rational, c0: &rug::Rational| {
        // c1·sⁿ + c0
        pool.add(vec![
            pool.mul(vec![rational_to_expr(c1, pool), s_n]),
            rational_to_expr(c0, pool),
        ])
    };
    let x_num = lin(&-b0, &a0); // −b₀·sⁿ + a₀
    let x_den = lin(&b1, &-a1.clone()); // b₁·sⁿ − a₁
    if degree(&num) < 1 && b1 == 0 {
        return None; // x would not depend on s
    }
    let x_of_s = pool.mul(vec![x_num, pool.pow(x_den, pool.integer(-1))]);

    // Rewrite the integrand directly in `s`: standalone `x → x(s)`, and every
    // power `r(x)^{c/d}` of the radicand → `s^{c·n/d}`, so no un-reducible
    // `(sⁿ)^{1/n}` is ever formed.
    let core = to_s(expr, var, &num, &den, n, s, x_of_s, pool)?;
    let dx_ds = simplify(crate::diff::diff(x_of_s, s, pool).ok()?.value, pool).value;
    let integrand_s = simplify(pool.mul(vec![core, dx_ds]), pool).value;

    // Integrate the rational-in-`s` integrand (always elementary), then
    // back-substitute s = r(x)^{1/n}.
    let f_s = match crate::integrate::engine::integrate(integrand_s, s, pool) {
        Ok(d) => d.value,
        Err(_) => return None,
    };
    let radical_expr = pool.pow(radicand, pool.rational(1, n as i32));
    let mut back = HashMap::new();
    back.insert(s, radical_expr);
    let f_x = simplify(crate::kernel::subs(f_s, &back, pool), pool).value;

    // Soundness gate: d/dx F = integrand numerically (where the radicand > 0).
    if !verify_derivative(f_x, expr, radicand, var, pool) {
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

/// Rewrite `expr` (rational in `x` and the single radical `r(x)^{1/n}`, `r =
/// num/den`) as a rational function of `s`, where `s = r(x)^{1/n}`: standalone
/// `x → x(s)`, and any power `r(x)^{c/d}` of the radicand → `s^{c·n/d}`.  Returns
/// `None` if a subterm is not expressible this way (a different radical, a
/// transcendental of `x`, or a fractional power with `d ∤ c·n`).
#[allow(clippy::too_many_arguments)]
fn to_s(
    expr: ExprId,
    var: ExprId,
    num: &QPoly,
    den: &QPoly,
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
    // `r(x)^{c/d}` → `s^{c·n/d}` when `base = r` (as a fraction) and the exponent
    // is an integer.
    let radical_power = |base: ExprId, c: i64, d: i64, pool: &ExprPool| -> Option<ExprId> {
        if same_fraction(base, num, den, var, pool) && (c * n as i64) % d == 0 {
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
                .map(|&a| to_s(a, var, num, den, n, s, x_of_s, pool))
                .collect::<Option<_>>()?;
            Some(pool.add(v))
        }
        ExprData::Mul(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_s(a, var, num, den, n, s, x_of_s, pool))
                .collect::<Option<_>>()?;
            Some(pool.mul(v))
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let inner = to_s(base, var, num, den, n, s, x_of_s, pool)?;
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

/// Is `base` equal, as a rational function of `x`, to the fraction `num/den`?
/// Tested by cross-multiplication so unequal scalings are *not* matched.
fn same_fraction(base: ExprId, num: &QPoly, den: &QPoly, var: ExprId, pool: &ExprPool) -> bool {
    let Some((bn, bd)) = expr_to_qrational(base, var, pool) else {
        return false;
    };
    // base nontrivial in x (so it really is the radicand, not a constant).
    if degree(&trim(bn.clone())) < 1 && degree(&trim(bd.clone())) < 1 {
        return false;
    }
    trim(poly_mul(&bn, den)) == trim(poly_mul(num, &bd))
}

/// Numeric `d/dx F = integrand` check at sample points where the radicand is
/// positive (so the principal real branch of the radical is well defined).
fn verify_derivative(
    f: ExprId,
    integrand: ExprId,
    radicand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> bool {
    let Ok(df) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    let mut checked = 0;
    for &xv in &[0.3_f64, 0.8, 1.6, 2.7, 3.9, 5.1] {
        match eval(radicand, var, xv, pool) {
            Some(r) if r > 1e-6 && r.is_finite() => {}
            _ => continue, // radicand not safely positive: skip
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
        check_at(build, &[1.3, 2.4, 3.7]);
    }

    fn check_at(build: impl Fn(&ExprPool, ExprId) -> ExprId, samples: &[f64]) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = build(&pool, x);
        let r = crate::integrate::engine::integrate(f, x, &pool);
        assert!(r.is_ok(), "expected elementary; got {r:?}");
        // d/dx F = f at samples (chosen inside the radicand-positive domain).
        let g = r.unwrap().value;
        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in samples {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(f, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }
    }

    /// MC0 (Möbius radicand): `∫ √((1−x)/(1+x)) dx` — genus-0 via `x = (1−s²)/(1+s²)`.
    /// Radicand positive on `(−1, 1)`.
    #[test]
    fn sqrt_mobius_one_minus_x_over_one_plus_x() {
        check_at(
            |p, x| {
                let num = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), x])]);
                let den = p.add(vec![p.integer(1), x]);
                let ratio = p.mul(vec![num, p.pow(den, p.integer(-1))]);
                p.func("sqrt", vec![ratio])
            },
            &[0.2, 0.55, 0.85],
        );
    }

    /// MC0: `∫ ∛((x+1)/(x−1)) dx` — radicand positive for `x > 1`.
    #[test]
    fn cbrt_mobius_x_plus_1_over_x_minus_1() {
        check_at(
            |p, x| {
                let num = p.add(vec![x, p.integer(1)]);
                let den = p.add(vec![x, p.integer(-1)]);
                let ratio = p.mul(vec![num, p.pow(den, p.integer(-1))]);
                p.func("cbrt", vec![ratio])
            },
            &[1.7, 2.6, 4.3],
        );
    }

    /// MC0: `∫ 1/((1+x)·√((1−x)/(1+x))) dx` — a rational weight times the Möbius
    /// radical.  Radicand positive on `(−1, 1)`.
    #[test]
    fn weighted_sqrt_mobius() {
        check_at(
            |p, x| {
                let num = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), x])]);
                let den = p.add(vec![p.integer(1), x]);
                let ratio = p.mul(vec![num, p.pow(den, p.integer(-1))]);
                let rad = p.func("sqrt", vec![ratio]);
                let w = p.pow(p.add(vec![p.integer(1), x]), p.integer(-1));
                p.mul(vec![w, p.pow(rad, p.integer(-1))])
            },
            &[0.2, 0.55, 0.85],
        );
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
