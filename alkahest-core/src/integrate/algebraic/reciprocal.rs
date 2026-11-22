//! The **reciprocal pullback** `t = x + c/x` for quartic radicands that are
//! invariant under `x ↦ c/x`.
//!
//! When `a(x) = a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀` satisfies `a₀ = c²a₄` and
//! `a₁ = c·a₃` for a rational `c`, then `x⁴·a(c/x) = c²·a(x)` and
//!
//! ```text
//!   a(x) = a₄·x²·S(t),   S(t) = t² + (a₃/a₄)·t + a₂/a₄ − 2c,   t = x + c/x,
//! ```
//!
//! so `Y = y/x` satisfies `Y² = a₄·S(t)` — a **genus-0** conic.  With
//! `dt = (1 − c/x²) dx`,
//!
//! ```text
//!   ∫ B(x)·y dx = ∫ G·Y dt,        G = B·x³/(x² − c),
//! ```
//!
//! which is a function of `t` exactly when `G` is invariant under
//! `σ: x ↦ c/x` (σ fixes `t` and `Y`).  `∫(x²+1)dx/((1−x²)√(x⁴+1))`
//! (`c = −1`) is the classic: its log part lives over `ℚ(√2)`, out of reach of
//! the rational residue machinery, but in `t` it is `∫−dt/(t·√(t²+2))`.
//!
//! An invariant `G = N/D` is rewritten in `t` by reducing modulo
//! `x² − t·x + c` (`N ≡ n₀ + n₁x`, `D ≡ d₀ + d₁x` over `ℚ[t]`) and multiplying
//! by the conjugate `x ↦ t − x`; invariance is exactly `n₁d₀ = n₀d₁`.
//! The rational part `A(x)` is integrated on its own.  The antiderivative in
//! `t` is rewritten with `t := x + c/x` and `√(a₄S(t)) := √a/x` (which squares
//! to `a₄S(t)` identically), then gated by `d/dx F = f` against the original
//! integrand.

use rug::Rational;

use super::super::risch::poly_rde::{
    degree, poly_add, poly_mul, poly_scale, qpoly_to_expr, rational_to_expr, trim, QPoly,
};
use super::super::risch::rational_rde::{expr_to_qrational, poly_sub};
use super::decompose::decompose_sqrt;
use super::poly_utils::is_zero_expr;
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

const T_NAME: &str = "$recip_t$";

fn q(n: i64) -> Rational {
    Rational::from(n)
}

/// `p(x) mod (x² − t·x + c)` as `(p₀(t), p₁(t))` with `p ≡ p₀ + p₁·x`.
fn reduce(p: &QPoly, c: &Rational) -> (QPoly, QPoly) {
    // x^k ≡ α_k + β_k x;  x^{k+1} ≡ −c·β_k + (α_k + t·β_k)·x.
    let t: QPoly = vec![q(0), q(1)];
    let (mut al, mut be): (QPoly, QPoly) = (vec![q(1)], vec![]);
    let (mut r0, mut r1): (QPoly, QPoly) = (vec![], vec![]);
    for coef in p {
        r0 = poly_add(&r0, &poly_scale(&al, coef));
        r1 = poly_add(&r1, &poly_scale(&be, coef));
        let nal = poly_scale(&be, &(-c.clone()));
        let nbe = poly_add(&al, &poly_mul(&t, &be));
        al = trim(nal);
        be = trim(nbe);
    }
    (trim(r0), trim(r1))
}

/// `N/D` invariant under `x ↦ c/x`, rewritten as a rational function of `t`.
fn to_t(num: &QPoly, den: &QPoly, c: &Rational) -> Option<(QPoly, QPoly)> {
    let (n0, n1) = reduce(num, c);
    let (d0, d1) = reduce(den, c);
    if !trim(poly_sub(&poly_mul(&n1, &d0), &poly_mul(&n0, &d1))).is_empty() {
        return None; // not invariant
    }
    let t: QPoly = vec![q(0), q(1)];
    let d0_d1t = poly_add(&d0, &poly_mul(&d1, &t));
    let rn = poly_add(&poly_mul(&n0, &d0_d1t), &poly_scale(&poly_mul(&n1, &d1), c));
    let rd = poly_add(&poly_mul(&d0, &d0_d1t), &poly_scale(&poly_mul(&d1, &d1), c));
    let rd = trim(rd);
    if rd.is_empty() {
        return None;
    }
    Some((trim(rn), rd))
}

fn rat_sqrt(r: &Rational) -> Option<Rational> {
    if *r < 0 {
        return None;
    }
    let (n, d) = (r.numer().clone(), r.denom().clone());
    let (ns, ds) = (n.clone().sqrt(), d.clone().sqrt());
    if rug::Integer::from(&ns * &ns) == n && rug::Integer::from(&ds * &ds) == d {
        Some(Rational::from((ns, ds)))
    } else {
        None
    }
}

fn ratfn_expr(n: &QPoly, d: &QPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![
        qpoly_to_expr(n, var, pool),
        pool.pow(qpoly_to_expr(d, var, pool), pool.integer(-1_i32)),
    ])
}

/// Try `t = x + c/x`.  `None` when the shape does not apply or the gate fails.
pub(super) fn try_reciprocal_pullback(
    expr: ExprId,
    sqrt_id: ExprId,
    p_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    let t = pool.symbol(T_NAME, Domain::Real);
    if var == t {
        return None;
    }
    let a = trim(crate::integrate::risch::poly_rde::expr_to_qpoly_expanded(
        p_expr, var, pool,
    )?);
    if degree(&a) != 4 {
        return None;
    }
    let (a_raw, b_raw) = decompose_sqrt(expr, sqrt_id, p_expr, pool)?;
    let a_part = simplify(a_raw, pool).value;
    let b_part = simplify(b_raw, pool).value;
    if is_zero_expr(b_part, pool) {
        return None;
    }
    let (bn, bd) = expr_to_qrational(b_part, var, pool)?;
    let a4 = a[4].clone();
    let c_abs = rat_sqrt(&(a[0].clone() / &a4))?;
    if c_abs == 0 {
        return None;
    }
    for c in [c_abs.clone(), -c_abs.clone()] {
        if a[1] != c.clone() * &a[3] {
            continue;
        }
        // G = B·x³/(x² − c)
        let gn = poly_mul(&bn, &vec![q(0), q(0), q(0), q(1)]);
        let gd = poly_mul(&bd, &vec![-c.clone(), q(0), q(1)]);
        let Some((tn, td)) = to_t(&gn, &gd, &c) else {
            continue;
        };
        // S(t)·a₄ = a₄t² + a₃t + a₂ − 2c·a₄.
        let s_t: QPoly = vec![
            a[2].clone() - q(2) * c.clone() * &a4,
            a[3].clone(),
            a4.clone(),
        ];
        let s_expr = qpoly_to_expr(&s_t, t, pool);
        let integrand_t = simplify(
            pool.mul(vec![
                ratfn_expr(&tn, &td, t, pool),
                pool.func("sqrt", vec![s_expr]),
            ]),
            pool,
        )
        .value;
        let Ok(inner) = crate::integrate::engine::integrate(integrand_t, t, pool) else {
            continue;
        };
        // Back-substitute: √(a₄S(t)) := √a/x first, then t := x + c/x.
        let y_over_x = pool.mul(vec![sqrt_id, pool.pow(var, pool.integer(-1_i32))]);
        let f_t = replace_sqrt(inner.value, &s_t, t, y_over_x, pool);
        let t_of_x = pool.add(vec![
            var,
            pool.mul(vec![
                rational_to_expr(&c, pool),
                pool.pow(var, pool.integer(-1_i32)),
            ]),
        ]);
        let mut map = std::collections::HashMap::new();
        map.insert(t, t_of_x);
        let f_b = crate::kernel::subs::subs(f_t, &map, pool);
        let f_a = if is_zero_expr(a_part, pool) {
            pool.integer(0_i32)
        } else {
            match crate::integrate::engine::integrate(a_part, var, pool) {
                Ok(d) => d.value,
                Err(_) => return None,
            }
        };
        let f = simplify(pool.add(vec![f_a, f_b]), pool).value;
        if mentions(f, t, pool) {
            continue;
        }
        if !super::parametrize::verify_derivative_pub(f, expr, p_expr, var, pool) {
            continue;
        }
        let mut log = DerivationLog::new();
        log = log.merge(inner.log);
        log.push(RewriteStep::simple("alg_reciprocal_pullback", expr, f));
        return Some(DerivedExpr::with_log(f, log));
    }
    None
}

fn mentions(e: ExprId, s: ExprId, pool: &ExprPool) -> bool {
    !crate::integrate::risch::poly_rde::is_free_of_var(e, s, pool)
}

/// Replace `√g` / `g^{k/2}` with `g ≡ s` (as polynomials in `t`) by `back^k`.
fn replace_sqrt(e: ExprId, s: &QPoly, t: ExprId, back: ExprId, pool: &ExprPool) -> ExprId {
    let is_s = |g: ExprId| -> bool {
        expr_to_qrational(g, t, pool)
            .is_some_and(|(n, d)| trim(poly_mul(&n, &vec![q(1)])) == trim(poly_mul(s, &d)))
    };
    match pool.get(e) {
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && is_s(args[0]) =>
        {
            back
        }
        ExprData::Func { ref name, ref args } => pool.func(
            name,
            args.iter()
                .map(|&a| replace_sqrt(a, s, t, back, pool))
                .collect(),
        ),
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.denom() == 2 && is_s(base) {
                    if let Some(k) = r.0.numer().to_i32() {
                        return pool.pow(back, pool.integer(k));
                    }
                }
            }
            pool.pow(replace_sqrt(base, s, t, back, pool), exp)
        }
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| replace_sqrt(a, s, t, back, pool))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| replace_sqrt(a, s, t, back, pool))
                .collect(),
        ),
        ExprData::RootSum { .. } => e,
        _ => e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> Option<ExprId> {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = std::collections::HashMap::from([("x".to_owned(), x)]);
        let e = crate::parse::parse(src, &pool, &mut syms).ok()?;
        let r = super::super::integrate_algebraic(e, x, &pool).ok()?;
        Some(r.value)
    }

    #[test]
    fn invariant_rewrite() {
        // x² + 1/x² ... as N/D = (x⁴+1)/x², c = 1 → t² − 2.
        let num: QPoly = vec![q(1), q(0), q(0), q(0), q(1)];
        let den: QPoly = vec![q(0), q(0), q(1)];
        let (n, d) = to_t(&num, &den, &q(1)).unwrap();
        // (t² − 2)/1 up to normalisation
        let lc = d[0].clone();
        assert_eq!(degree(&d), 0);
        let n: QPoly = n.iter().map(|v| v.clone() / &lc).collect();
        assert_eq!(n, vec![q(-2), q(0), q(1)]);
        // x itself is not invariant.
        assert!(to_t(&vec![q(0), q(1)], &vec![q(1)], &q(1)).is_none());
    }

    #[test]
    fn charlwood_38_and_39() {
        assert!(run("(x^2+1)/((1-x^2)*sqrt(x^4+1))").is_some());
        assert!(run("(1-x^2)/((x^2+1)*sqrt(x^4+1))").is_some());
    }
}
