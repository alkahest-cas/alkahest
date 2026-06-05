//! End-to-end genus-1 algebraic integration: emit the antiderivative
//! `∫ R(x,y) dx = g + (1/N)·log(u)` (Risch **MC**, capstone).
//!
//! Ties the whole genus-1 stack together for `y² = a(x)` (cubic):
//! 1. **Hermite-on-curve** ([`super::hermite_curve`]) → algebraic part `g` plus a
//!    third-kind remainder `h` (simple poles);
//! 2. **residue divisor** ([`super::residues`]) of `h dx`;
//! 3. **FIND-ORDER** ([`super::find_order`]) — the divisor class must be torsion;
//! 4. **Miller's algorithm** ([`super::elliptic`]) → the log argument `u` on `E`,
//!    back-translated to the original `(x, y)`.
//!
//! The result `g + (1/N)·log(u)` is accepted only after an exact numeric
//! `d/dx F = integrand` check — sound regardless of the holomorphic-part subtlety
//! (a leftover first-kind differential, which would make the integral
//! non-elementary, makes the check fail and the function return `None`).

use rug::{Integer, Rational};

use super::super::risch::alg_field::AlgElem;
use super::super::risch::poly_rde::{degree, qpoly_to_expr, rational_to_expr, trim, QPoly};
use super::elliptic::{short_weierstrass, EllFactor, Point};
use super::find_order::{find_order_placed, genus, FindOrder};
use super::hermite_curve::hermite_reduce_radical;
use super::residues::residue_divisor_placed;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// Integrate `∫ (integrand) dx` on the genus-1 curve `y² = a(x)` (cubic `a`),
/// returning the symbolic antiderivative `g + (1/N)·log(u)` when it is elementary
/// (verified `d/dx F = integrand`), else `None`.
pub fn integrate_genus1_log(
    a: &QPoly,
    integrand: &AlgElem,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let a = trim(a.clone());
    if genus(2, &a) != Some(1) || degree(&a) != 3 {
        return None;
    }

    // 1–3. Hermite → (g, h); residue divisor; torsion decision.
    let (g_alg, h) = hermite_reduce_radical(2, &a, integrand)?;
    let divisor = residue_divisor_placed(2, &a, &h);
    if !matches!(
        find_order_placed(2, &a, &divisor),
        FindOrder::Principal { .. }
    ) {
        return None;
    }

    // 4. Abel–Jacobi image and its order; build u for the principal divisor.
    let (e, map) = short_weierstrass(&a)?;
    let mut l = Integer::from(1);
    for r in &divisor {
        l = l.lcm(r.residue.value.denom());
    }
    let mut pairs: Vec<(Point, i64)> = Vec::new();
    for r in &divisor {
        let scaled = r.residue.value.clone() * Rational::from(l.clone());
        let coeff = scaled.numer().to_i64()?;
        let pt = if r.residue.at_infinity {
            Point::Infinity
        } else {
            let (x, y) = map(&r.residue.point, &r.y_coord);
            Point::Affine(x, y)
        };
        pairs.push((pt, coeff));
    }
    let mut s = Point::Infinity;
    for (p, c) in &pairs {
        let t = if *c >= 0 {
            e.mul(*c as u64, p)
        } else {
            e.mul((-c) as u64, &e.neg(p))
        };
        s = e.add(&s, &t);
    }
    let n_s = e.order(&s)?; // class order (Mazur ≤ 12)
    let scaled: Vec<(Point, i64)> = pairs
        .iter()
        .map(|(p, c)| (p.clone(), c * n_s as i64))
        .collect();
    let u_e = e.general_miller(&scaled)?;

    // Back-translate u (on E, coords X = a₃x + a₂/3, Y = a₃y) to (x, √a).
    let a3 = a[3].clone();
    let a2 = a.get(2).cloned().unwrap_or_else(|| Rational::from(0));
    let a_expr = qpoly_to_expr(&a, var, pool);
    let y_sym = pool.func("sqrt", vec![a_expr]);
    let to_expr = |f: &EllFactor| -> ExprId {
        match f {
            EllFactor::Vertical(x0) => pool.add(vec![
                pool.mul(vec![rational_to_expr(&a3, pool), var]),
                rational_to_expr(&(a2.clone() / Rational::from(3)), pool),
                rational_to_expr(&(-x0.clone()), pool),
            ]),
            EllFactor::Line(lam, nu) => pool.add(vec![
                pool.mul(vec![rational_to_expr(&a3, pool), y_sym]),
                pool.mul(vec![rational_to_expr(&(-(lam.clone() * &a3)), pool), var]),
                rational_to_expr(
                    &(-(lam.clone() * &a2 / Rational::from(3)) - nu.clone()),
                    pool,
                ),
            ]),
        }
    };
    let product = |fs: &[EllFactor]| -> ExprId {
        if fs.is_empty() {
            pool.integer(1_i32)
        } else {
            pool.mul(fs.iter().map(&to_expr).collect())
        }
    };
    let u = if u_e.den.is_empty() {
        product(&u_e.num)
    } else {
        pool.mul(vec![
            product(&u_e.num),
            pool.pow(product(&u_e.den), pool.integer(-1_i32)),
        ])
    };

    // F = g + (1/(N·L))·log(u).
    let log_u = pool.func("log", vec![u]);
    let coeff = rational_to_expr(
        &Rational::from((Integer::from(1), Integer::from(n_s) * &l)),
        pool,
    );
    let g_expr = alg_to_expr(&g_alg, y_sym, var, pool);
    let f = simplify(pool.add(vec![g_expr, pool.mul(vec![coeff, log_u])]), pool).value;

    // Soundness gate: d/dx F = integrand numerically (where a(x) > 0).
    if verify(f, integrand, &a, var, pool) {
        Some(f)
    } else {
        None
    }
}

/// `Σⱼ gⱼ(x)·yʲ` (AlgElem, `y = √a`) → symbolic.
fn alg_to_expr(g: &AlgElem, y_sym: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for (j, c) in g.iter().enumerate() {
        if c.numer().is_empty() {
            continue;
        }
        let num = qpoly_to_expr(c.numer(), var, pool);
        let coeff = if c.denom().len() == 1 && c.denom()[0] == 1 {
            num
        } else {
            pool.mul(vec![
                num,
                pool.pow(qpoly_to_expr(c.denom(), var, pool), pool.integer(-1_i32)),
            ])
        };
        let term = if j == 0 {
            coeff
        } else {
            pool.mul(vec![coeff, pool.pow(y_sym, pool.integer(j as i32))])
        };
        terms.push(term);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

fn verify(f: ExprId, integrand: &AlgElem, a: &QPoly, var: ExprId, pool: &ExprPool) -> bool {
    let Ok(df) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    let mut checked = 0;
    for &xv in &[0.3_f64, 1.4, 2.7, 3.6, 4.9] {
        let av = eval_qpoly(a, xv);
        if av <= 1e-6 {
            continue; // need √a real
        }
        let ya = av.sqrt();
        let Some(lhs) = eval(ds, var, xv, pool) else {
            return false;
        };
        let rhs = eval_alg(integrand, xv, ya);
        if !lhs.is_finite() || !rhs.is_finite() || (lhs - rhs).abs() > 1e-6 * (1.0 + rhs.abs()) {
            return false;
        }
        checked += 1;
    }
    checked >= 2
}

fn eval_qpoly(p: &QPoly, xv: f64) -> f64 {
    p.iter().rev().fold(0.0, |acc, c| acc * xv + c.to_f64())
}

fn eval_alg(g: &AlgElem, xv: f64, yv: f64) -> f64 {
    g.iter()
        .enumerate()
        .map(|(j, c)| {
            let num = eval_qpoly(c.numer(), xv);
            let den = eval_qpoly(c.denom(), xv);
            (num / den) * yv.powi(j as i32)
        })
        .sum()
}

/// Numeric eval; `sqrt`/`cbrt` take the principal real branch (matching the
/// `+√a` branch used by [`eval_alg`] for the integrand).
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
            let v = eval(args[0], x, xv, pool)?;
            match name.as_str() {
                "exp" => Some(v.exp()),
                "log" => Some(v.ln()),
                "sqrt" => Some(v.sqrt()),
                "cbrt" => Some(v.cbrt()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::risch::alg_field::RatFn;
    use super::*;
    use crate::kernel::Domain;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    /// `∫ [1/(2x) + (1/(2x(x³+1)))·y] dx` on `y² = x³+1`
    ///   = (1/3)·log(√(x³+1) − 1).
    /// The residue divisor is `(0,1) − O`, class order 3 ((0,1) is 3-torsion),
    /// so FIND-ORDER = Principal{3} and Miller yields `u = y − 1`.
    #[test]
    fn elliptic_log_third_order() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = qp(&[1, 0, 0, 1]); // x³ + 1
                                   // integrand = 1/(2x)  +  y/(2x(x³+1)).
        let integrand = vec![
            RatFn::new(qp(&[1]), qp(&[0, 2])),          // 1/(2x)
            RatFn::new(qp(&[1]), qp(&[0, 2, 0, 0, 2])), // 1/(2x + 2x⁴) = 1/(2x(x³+1))
        ];
        let f = integrate_genus1_log(&a, &integrand, x, &pool).expect("elementary log");
        // d/dx F = integrand is checked inside; assert it really matches here too.
        let ds = simplify(crate::diff::diff(f, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.7_f64, 1.5, 2.9] {
            let av: f64 = a.iter().rev().fold(0.0, |acc, c| acc * xv + c.to_f64());
            let ya = av.sqrt();
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval_alg(&integrand, xv, ya);
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {}",
                pool.display(f)
            );
        }
    }

    /// `∫ y/(x²(x³+1)) dx`-style integrand whose class is non-torsion-free... a
    /// genuinely non-elementary case returns `None`: `∫ dx/√(x³+1)` (elliptic
    /// integral of the first kind, holomorphic — no log part).
    #[test]
    fn first_kind_not_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = qp(&[1, 0, 0, 1]);
        // 1/y = y/(x³+1).
        let integrand = vec![RatFn::int(0), RatFn::new(qp(&[1]), qp(&[1, 0, 0, 1]))];
        assert!(integrate_genus1_log(&a, &integrand, x, &pool).is_none());
    }
}
