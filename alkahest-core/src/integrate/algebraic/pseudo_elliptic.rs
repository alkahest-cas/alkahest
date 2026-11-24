//! Pseudo-elliptic third-kind integrals `∫ N(t)/(D(t)·√P(t)) dt` through a
//! norm equation.
//!
//! When the residue divisor of `ω = N/(D·y) dt` on `y² = P` is torsion, the
//! logarithmic part is `c·log(h)` for a function `h = p + q·y` whose zeros and
//! poles lie over the roots of `D` (and over ∞).  Its norm is then
//!
//! ```text
//! p² − q²·P = λ·Dᵏ        (log form,  F = c·log((p + q y)/(p − q y)))
//! p² + q²·P = λ·Dᵏ        (atan form, F = c·atan(q y / p))
//! ```
//!
//! for the torsion order `k`.  The second is the first with `p ↦ i·p`: it is
//! what a divisor *anti*-invariant under complex conjugation gives, as for
//! `∫ dt/((1 + t²)·√(t⁴ + 3t² + 3)) = ⅓·atan(t·√(t⁴ + 3t² + 3))` (Charlwood
//! #5 after `t = tan x`), whose residues `±i/2` sit over `t = ±i`.
//!
//! Both are **linear** problems once a square root `S` of `±P` modulo `Dᵏ` is
//! known: `Dᵏ | p² ∓ q²P` holds when `p ≡ q·S (mod Dᵏ)`.  With
//! `deg p ≤ n/2`, `deg q ≤ (n − deg P)/2` for `n = k·deg D`, the norm has degree
//! at most `n` and is divisible by `Dᵏ`, so it *is* `λ·Dᵏ`.  `S` is lifted from
//! a root modulo `D` by Newton's iteration; the root modulo `D` is looked for
//! among `c·tʲ` (a rational multiple of a power of `t`), which is where the
//! classical pseudo-elliptic families have it.
//!
//! The candidate is accepted only when `F′` is a constant multiple of `ω`
//! *exactly* (polynomial identity), and the caller gates it again by
//! differentiation.  Nothing here decides non-elementarity: a miss is a decline.

use rug::Rational;

use crate::integrate::risch::number_field::{mod_inverse, poly_mod};
use crate::integrate::risch::poly_rde::{
    degree, poly_add, poly_deriv, poly_mul, poly_scale, qpoly_to_expr, rational_to_expr, trim,
    QPoly,
};
use crate::integrate::risch::rational_rde::{poly_divrem, poly_gcd, poly_pow, poly_sub};
use crate::kernel::{ExprId, ExprPool};

/// Largest torsion order tried.
const MAX_ORDER: usize = 12;
/// Largest `k·deg D` (size of the linear system).
const MAX_N: i64 = 48;

fn rat(n: i64) -> Rational {
    Rational::from(n)
}

fn monomial(c: Rational, j: usize) -> QPoly {
    let mut v = vec![Rational::from(0); j + 1];
    v[j] = c;
    v
}

/// Exact rational square root, `None` when `r` is not a square in ℚ.
fn rational_sqrt(r: &Rational) -> Option<Rational> {
    if *r < 0 {
        return None;
    }
    let (n, d) = (r.numer().clone(), r.denom().clone());
    let (sn, sd) = (n.clone().sqrt(), d.clone().sqrt());
    (sn.clone() * &sn == n && sd.clone() * &sd == d).then(|| Rational::from((sn, sd)))
}

/// A root `S₀` of `S² ≡ r (mod D)` of the shape `c·tʲ`, `j < deg D`.
fn monomial_sqrt_mod(r: &QPoly, d: &QPoly) -> Option<QPoly> {
    let dd = degree(d) as usize;
    let t = monomial(rat(1), 1);
    for j in 0..dd {
        // r·t^{−2j} mod D must be a rational square c².
        let tj2 = poly_mod(&poly_pow(&t, 2 * j as u32), d);
        let inv = mod_inverse(&tj2, d)?;
        let q = trim(poly_mod(&poly_mul(r, &inv), d));
        if degree(&q) > 0 {
            continue;
        }
        let c2 = q.first().cloned().unwrap_or_else(|| Rational::from(0));
        if c2 == 0 {
            continue;
        }
        if let Some(c) = rational_sqrt(&c2) {
            return Some(monomial(c, j));
        }
    }
    None
}

/// Newton-lift `S₀² ≡ r (mod D)` to `S² ≡ r (mod Dᵐ)`.
fn hensel_sqrt(s0: &QPoly, r: &QPoly, d: &QPoly, m: u32) -> Option<QPoly> {
    let mut s = s0.clone();
    let mut prec = 1_u32;
    while prec < m {
        prec = (2 * prec).min(m);
        let modulus = poly_pow(d, prec);
        let err = poly_sub(&poly_mul(&s, &s), r);
        let two_s = poly_scale(&s, &rat(2));
        let inv = mod_inverse(&poly_mod(&two_s, &modulus), &modulus)?;
        let step = poly_mod(&poly_mul(&err, &inv), &modulus);
        s = trim(poly_mod(&poly_sub(&s, &step), &modulus));
    }
    let check = poly_mod(&poly_sub(&poly_mul(&s, &s), r), &poly_pow(d, m));
    trim(check).is_empty().then_some(s)
}

/// A non-zero vector in the null space of `rows × cols` matrix `a`
/// (row-major), preferring one whose `q`-part (columns `≥ split`) is non-zero.
fn null_vectors(mut a: Vec<Vec<Rational>>, cols: usize) -> Vec<Vec<Rational>> {
    let rows = a.len();
    let mut pivots = Vec::new();
    let mut r = 0;
    for c in 0..cols {
        let Some(p) = (r..rows).find(|&i| a[i][c] != 0) else {
            continue;
        };
        a.swap(r, p);
        let inv = Rational::from(1) / a[r][c].clone();
        for v in a[r].iter_mut() {
            *v *= &inv;
        }
        for i in 0..rows {
            if i != r && a[i][c] != 0 {
                let f = a[i][c].clone();
                for j in 0..cols {
                    let t = f.clone() * &a[r][j];
                    a[i][j] -= t;
                }
            }
        }
        pivots.push(c);
        r += 1;
        if r == rows {
            break;
        }
    }
    let free: Vec<usize> = (0..cols).filter(|c| !pivots.contains(c)).collect();
    free.iter()
        .map(|&f| {
            let mut v = vec![Rational::from(0); cols];
            v[f] = Rational::from(1);
            for (i, &pc) in pivots.iter().enumerate() {
                v[pc] = -a[i][f].clone();
            }
            v
        })
        .collect()
}

/// `(p, q, λ)` with `p ≡ q·S (mod Dᵏ)`, `p² − sign·q²P = λ·Dᵏ`, `q ≠ 0`, `λ ≠ 0`.
fn norm_solution(
    p_rad: &QPoly,
    d: &QPoly,
    s: &QPoly,
    k: u32,
    sign: i64,
) -> Option<(QPoly, QPoly, Rational)> {
    let n = degree(d) * k as i64;
    let dp = degree(p_rad);
    if n < dp || n > MAX_N {
        return None;
    }
    let (np, nq) = ((n / 2 + 1) as usize, ((n - dp) / 2 + 1) as usize);
    let modulus = poly_pow(d, k);
    let cols = np + nq;
    // Column j: the residue of the j-th basis unknown modulo Dᵏ.
    let mut columns: Vec<QPoly> = Vec::with_capacity(cols);
    for j in 0..np {
        columns.push(poly_mod(&monomial(rat(1), j), &modulus));
    }
    for j in 0..nq {
        let ts = poly_mul(&monomial(rat(-1), j), s);
        columns.push(poly_mod(&ts, &modulus));
    }
    let a: Vec<Vec<Rational>> = (0..n as usize)
        .map(|i| {
            columns
                .iter()
                .map(|c| c.get(i).cloned().unwrap_or_else(|| Rational::from(0)))
                .collect()
        })
        .collect();
    for v in null_vectors(a, cols) {
        let p = trim(v[..np].to_vec());
        let q = trim(v[np..].to_vec());
        if q.is_empty() || p.is_empty() {
            continue;
        }
        let norm = poly_sub(
            &poly_mul(&p, &p),
            &poly_scale(&poly_mul(&poly_mul(&q, &q), p_rad), &rat(sign)),
        );
        let (quo, rem) = poly_divrem(&norm, &modulus);
        let quo = trim(quo);
        if !trim(rem).is_empty() || degree(&quo) != 0 {
            continue;
        }
        return Some((p, q, quo[0].clone()));
    }
    None
}

/// `c` with `a = c·b` exactly, `None` when `a/b` is not a constant.
fn constant_ratio(a: &QPoly, b: &QPoly) -> Option<Rational> {
    let (a, b) = (trim(a.clone()), trim(b.clone()));
    if a.is_empty() || b.is_empty() || degree(&a) != degree(&b) {
        return None;
    }
    let c = a[a.len() - 1].clone() / b[b.len() - 1].clone();
    trim(poly_sub(&a, &poly_scale(&b, &c)))
        .is_empty()
        .then_some(c)
}

/// `∫ (num/den)/√P dt` as `c·atan(q√P/p)` or `c·log((p + q√P)/(p − q√P))`,
/// when the integrand has only simple poles away from `P = 0` and its residue
/// divisor is torsion of order `≤ 12` in the shape this module recognises.
pub(super) fn integrate_pseudo_elliptic(
    num: &QPoly,
    den: &QPoly,
    p_rad: &QPoly,
    sqrt_id: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let g = poly_gcd(num, den);
    let (num, _) = poly_divrem(num, &g);
    let (den, _) = poly_divrem(den, &g);
    let (num, den) = (trim(num), trim(den));
    let dd = degree(&den);
    // Simple poles only, none on the branch locus, and no pole over ∞.
    if dd < 1 || num.is_empty() || degree(&num) >= dd {
        return None;
    }
    if degree(&poly_gcd(&den, &poly_deriv(&den))) > 0 || degree(&poly_gcd(&den, p_rad)) > 0 {
        return None;
    }
    let p_prime = poly_deriv(p_rad);
    for sign in [1_i64, -1] {
        // log form (sign = 1) needs √P mod D, atan form (sign = −1) √(−P).
        let r = poly_scale(p_rad, &rat(sign));
        let r_mod = poly_mod(&r, &den);
        let Some(s0) = monomial_sqrt_mod(&r_mod, &den) else {
            continue;
        };
        let Some(s) = hensel_sqrt(&s0, &r, &den, MAX_ORDER as u32) else {
            continue;
        };
        for k in 1..=MAX_ORDER as u32 {
            crate::budget::check().ok()?;
            let modulus = poly_pow(&den, k);
            let sk = poly_mod(&s, &modulus);
            let Some((p, q, lambda)) = norm_solution(p_rad, &den, &sk, k, sign) else {
                continue;
            };
            // F′ = c·[2p q′P + p q P′ − 2p′ q P]/(m·λ·Dᵏ·y), m = 2 (atan) or 1 (log),
            // which must be c·num/(den·y).
            let core = poly_sub(
                &poly_add(
                    &poly_scale(&poly_mul(&poly_mul(&p, &poly_deriv(&q)), p_rad), &rat(2)),
                    &poly_mul(&poly_mul(&p, &q), &p_prime),
                ),
                &poly_scale(&poly_mul(&poly_mul(&poly_deriv(&p), &q), p_rad), &rat(2)),
            );
            let m = if sign == -1 { rat(2) } else { rat(1) };
            // num·m·λ·D^{k−1} = c·core
            let lhs = poly_scale(
                &poly_mul(&num, &poly_pow(&den, k - 1)),
                &(m * lambda.clone()),
            );
            let Some(c) = constant_ratio(&lhs, &core) else {
                continue;
            };
            let p_e = qpoly_to_expr(&p, var, pool);
            let qy = pool.mul(vec![qpoly_to_expr(&q, var, pool), sqrt_id]);
            let body = if sign == -1 {
                pool.func(
                    "atan",
                    vec![pool.mul(vec![qy, pool.pow(p_e, pool.integer(-1_i32))])],
                )
            } else {
                let neg_qy = pool.mul(vec![pool.integer(-1_i32), qy]);
                pool.func(
                    "log",
                    vec![pool.mul(vec![
                        pool.add(vec![p_e, qy]),
                        pool.pow(pool.add(vec![p_e, neg_qy]), pool.integer(-1_i32)),
                    ])],
                )
            };
            return Some(pool.mul(vec![rational_to_expr(&c, pool), body]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn q(v: &[i64]) -> QPoly {
        v.iter().map(|&c| Rational::from(c)).collect()
    }

    fn check(num: &[i64], den: &[i64], p: &[i64], at: &[f64]) -> ExprId {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let p_rad = q(p);
        let y = pool.func("sqrt", vec![qpoly_to_expr(&p_rad, t, &pool)]);
        let f = integrate_pseudo_elliptic(&q(num), &q(den), &p_rad, y, t, &pool)
            .expect("pseudo-elliptic antiderivative");
        let df = crate::simplify::engine::simplify(
            crate::diff::diff(f, t, &pool).unwrap().value,
            &pool,
        )
        .value;
        let integrand = pool.mul(vec![
            qpoly_to_expr(&q(num), t, &pool),
            pool.pow(qpoly_to_expr(&q(den), t, &pool), pool.integer(-1_i32)),
            pool.pow(y, pool.integer(-1_i32)),
        ]);
        for &xv in at {
            let mut env = std::collections::HashMap::new();
            env.insert(t, xv);
            let a = crate::jit::eval_interp(df, &env, &pool).unwrap();
            let b = crate::jit::eval_interp(integrand, &env, &pool).unwrap();
            assert!((a - b).abs() < 1e-10 * (1.0 + b.abs()), "{a} vs {b} at {xv}");
        }
        f
    }

    /// Charlwood #5 after `t = tan x`: `∫ dt/((1+t²)√(t⁴+3t²+3)) = ⅓·atan(t·√…)`.
    #[test]
    fn charlwood_5_reduced() {
        check(&[1], &[1, 0, 1], &[3, 0, 3, 0, 1], &[-1.3, -0.2, 0.4, 2.5]);
    }

    /// No square root of `±P` modulo `D` of the recognised shape: a decline.
    #[test]
    fn declines_outside_its_shape() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let p_rad = q(&[1, 0, 0, 0, 1]);
        let y = pool.func("sqrt", vec![qpoly_to_expr(&p_rad, t, &pool)]);
        // ∫ dt/((t − 2)√(t⁴+1)): residue divisor not torsion of the recognised shape.
        assert!(integrate_pseudo_elliptic(&q(&[1]), &q(&[-2, 1]), &p_rad, y, t, &pool).is_none());
    }
}
