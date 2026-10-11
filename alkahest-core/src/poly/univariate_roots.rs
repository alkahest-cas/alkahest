//! Exact roots of a univariate polynomial by factoring first.
//!
//! The solver and the eigenvalue routine both need "every root of this
//! polynomial, exactly". Doing that by formula on the whole polynomial stops
//! at degree 2 (degree 3–4 by Cardano/Ferrari, at a heavy price in nested
//! radicals and casus irreducibilis). Factoring first removes most of the
//! problem: `s³ + 6s² + 11s + 6` is `(s + 1)(s + 2)(s + 3)`, and the Legendre
//! polynomial `P₃` is `x·(5x² − 3)/2`.
//!
//! Each irreducible factor is then solved by the first of these that applies:
//!
//! 1. degree 1 — the rational root;
//! 2. degree 2 — the quadratic formula;
//! 3. an *even* polynomial `q(x²)` whose `q` is itself solvable this way —
//!    `±√r` for each root `r` of `q` (biquadratics such as `P₄`, `x⁴ + 1`, and
//!    recursively `x⁸ + 1`);
//! 4. a pure cubic `a·x³ + c` — the real cube root times `1, ω, ω̄`;
//! 5. a palindromic quartic — `y = x + 1/x` reduces it to two quadratics;
//! 6. anything else — one [`RootOf`](super::root_of) per root, in the
//!    documented root order.
//!
//! Over ℚ that is complete: every root of every polynomial is returned. With
//! symbolic (parameter) coefficients only steps 1–3 exist, and a factor that
//! needs anything more is reported back as its degree so the caller can refuse
//! with its own error code.
//!
//! Repeated roots are reported **once**, with the multiplicity of their factor
//! beside them; callers that return a solution *set* drop the multiplicity.

use rug::{Integer, Rational};

use super::multipoly::MultiPoly;
use super::root_of::{make_root_of, ordered_roots, rational_expr, ORDER_PREC};
use crate::kernel::{ExprId, ExprPool};

/// Distinct roots of the non-zero polynomial with integer coefficients
/// `coeffs` (ascending), each with its multiplicity.
///
/// `Err(d)` names the degree of an irreducible factor that could not be
/// solved — only possible when this build has no root isolation, or the factor
/// is past the `RootOf` degree cap.
pub(crate) fn roots_over_q(
    coeffs: &[Integer],
    pool: &ExprPool,
) -> Result<Vec<(ExprId, u32)>, usize> {
    let p = crate::flint::FlintPoly::from_rug_coefficients(coeffs);
    if p.is_zero() {
        return Err(0);
    }
    let (_, facs) = p.factor_over_z().map_err(|()| coeffs.len() - 1)?;
    let mut out = Vec::new();
    for (f, e) in facs {
        let mut c: Vec<Integer> = (0..f.length())
            .map(|i| f.get_coeff_flint(i).to_rug())
            .collect();
        if c.last().is_some_and(|l| *l < 0) {
            for v in c.iter_mut() {
                *v = -v.clone();
            }
        }
        for r in irreducible_roots(&c, pool)? {
            out.push((r, e));
        }
    }
    Ok(out)
}

/// All roots of an irreducible `c` (ascending, degree ≥ 1).
pub(crate) fn irreducible_roots(c: &[Integer], pool: &ExprPool) -> Result<Vec<ExprId>, usize> {
    let degree = c.len() - 1;
    if degree == 0 {
        return Ok(Vec::new());
    }
    if let Some(r) = radical_roots(c, pool) {
        return Ok(r);
    }
    let n = ordered_roots(c, ORDER_PREC).ok_or(degree)?.len();
    if n != degree {
        return Err(degree);
    }
    Ok((0..degree).map(|k| make_root_of(c, k, pool)).collect())
}

/// Roots of `c` in radicals by steps 1–5 of the module docs, or `None`.
fn radical_roots(c: &[Integer], pool: &ExprPool) -> Option<Vec<ExprId>> {
    let degree = c.len() - 1;
    match degree {
        0 => Some(Vec::new()),
        1 => Some(vec![rational_expr(
            &Rational::from((-c[0].clone(), c[1].clone())),
            pool,
        )]),
        2 => Some(integer_quadratic_roots(&c[2], &c[1], &c[0], pool)),
        _ => {
            if c.iter().skip(1).step_by(2).all(|v| *v == 0) {
                // q(y) with y = x²
                let q: Vec<Integer> = c.iter().step_by(2).cloned().collect();
                let inner = radical_roots(&q, pool)?;
                return Some(plus_minus_sqrt(&inner, pool));
            }
            if degree == 3 && c[1] == 0 && c[2] == 0 {
                return Some(pure_cubic_roots(&c[0], &c[3], pool));
            }
            if degree == 4 && c[0] == c[4] && c[1] == c[3] {
                return Some(palindromic_quartic_roots(c, pool));
            }
            None
        }
    }
}

/// `a·x² + b·x + c = 0` over ℤ, with the square part of the discriminant
/// pulled out of the radical: `x = −b/2a ± (k/2a)·√m`, `D = k²·m`.
fn integer_quadratic_roots(a: &Integer, b: &Integer, c: &Integer, pool: &ExprPool) -> Vec<ExprId> {
    let disc = Integer::from(b * b) - Integer::from(4u32) * a * c;
    let two_a = Integer::from(a * 2u32);
    let centre = Rational::from((-b.clone(), two_a.clone()));
    if disc == 0 {
        return vec![rational_expr(&centre, pool)];
    }
    let (k, m) = split_square(&disc);
    let scale = Rational::from((k, two_a));
    if m == 1 {
        return vec![
            rational_expr(&(centre.clone() + &scale), pool),
            rational_expr(&(centre - scale), pool),
        ];
    }
    let s = pool.func("sqrt", vec![pool.integer(m)]);
    let centre_e = rational_expr(&centre, pool);
    let mut out = Vec::with_capacity(2);
    for sc in [scale.clone(), -scale] {
        let term = pool.mul(vec![rational_expr(&sc, pool), s]);
        out.push(if centre == 0 {
            term
        } else {
            pool.add(vec![centre_e, term])
        });
    }
    out
}

/// `n = k²·m` with `k > 0` and `m` carrying the sign of `n`, the square part
/// found by trial division (complete for `|n| < 10¹²`; beyond that only the
/// small square factors are pulled out, which leaves a correct if less
/// reduced radical).
fn split_square(n: &Integer) -> (Integer, Integer) {
    let mut m = Integer::from(n.abs_ref());
    let mut k = Integer::from(1);
    if m.is_perfect_square() {
        return (m.sqrt(), Integer::from(n.signum_ref()));
    }
    let mut p: u32 = 2;
    while p <= 1_000_000 {
        let pp = Integer::from(p) * p;
        if pp > m {
            break;
        }
        while m.is_divisible(&pp) {
            m /= &pp;
            k *= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n.is_negative() {
        m = -m;
    }
    (k, m)
}

/// `±√r` for each `r`.
fn plus_minus_sqrt(inner: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::with_capacity(2 * inner.len());
    for &r in inner {
        let s = pool.func("sqrt", vec![r]);
        out.push(s);
        out.push(pool.mul(vec![pool.integer(-1), s]));
    }
    out
}

/// `a·x² + b·x + c = 0` by the formula. Two roots, unless the discriminant is
/// an exact rational zero.
pub(crate) fn quadratic_roots(a: ExprId, b: ExprId, c: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let neg_one = pool.integer(-1);
    let disc = pool.add(vec![
        pool.pow(b, pool.integer(2)),
        pool.mul(vec![pool.integer(-4), a, c]),
    ]);
    let inv_two_a = pool.pow(pool.mul(vec![pool.integer(2), a]), neg_one);
    let neg_b = pool.mul(vec![neg_one, b]);
    if exact_rational(disc, pool).is_some_and(|d| d == 0) {
        return vec![pool.mul(vec![neg_b, inv_two_a])];
    }
    let s = pool.func("sqrt", vec![disc]);
    vec![
        pool.mul(vec![pool.add(vec![neg_b, s]), inv_two_a]),
        pool.mul(vec![
            pool.add(vec![neg_b, pool.mul(vec![neg_one, s])]),
            inv_two_a,
        ]),
    ]
}

/// `a·x³ + c = 0`: `x = t^{1/3}·{1, ω, ω̄}` with `t = −c/a`, the real cube root
/// taken explicitly so that no branch convention is involved.
fn pure_cubic_roots(c0: &Integer, c3: &Integer, pool: &ExprPool) -> Vec<ExprId> {
    let t = Rational::from((-c0.clone(), c3.clone()));
    let mag = rational_expr(&Rational::from(t.abs_ref()), pool);
    let cbrt = pool.pow(mag, pool.rational(1, 3));
    let real = if t < 0 {
        pool.mul(vec![pool.integer(-1), cbrt])
    } else {
        cbrt
    };
    // ω = (−1 ± √−3)/2
    let half = pool.rational(1, 2);
    let neg_half = pool.rational(-1, 2);
    let s = pool.func("sqrt", vec![pool.integer(-3)]);
    let w1 = pool.add(vec![neg_half, pool.mul(vec![half, s])]);
    let w2 = pool.add(vec![neg_half, pool.mul(vec![neg_half, s])]);
    vec![real, pool.mul(vec![real, w1]), pool.mul(vec![real, w2])]
}

/// `a·x⁴ + b·x³ + c·x² + b·x + a = 0`. Dividing by `x²` (zero is not a root,
/// as `a ≠ 0`) gives `a·y² + b·y + (c − 2a) = 0` in `y = x + 1/x`, and each
/// root `y` gives `x² − y·x + 1 = 0`.
fn palindromic_quartic_roots(c: &[Integer], pool: &ExprPool) -> Vec<ExprId> {
    let a = pool.integer(c[4].clone());
    let b = pool.integer(c[3].clone());
    let cc = pool.integer(&c[2] - Integer::from(&c[4] * 2u32));
    let one = pool.integer(1);
    let mut out = Vec::with_capacity(4);
    for y in quadratic_roots(a, b, cc, pool) {
        let neg_y = pool.mul(vec![pool.integer(-1), y]);
        out.extend(quadratic_roots(one, neg_y, one, pool));
    }
    out
}

/// Exact value of an expression built from rationals by `+`, `·` and integer
/// powers, or `None`.
pub(crate) fn exact_rational(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    use crate::kernel::ExprData;
    match pool.get(e) {
        ExprData::Integer(n) => Some(Rational::from(n.0)),
        ExprData::Rational(r) => Some(r.0),
        ExprData::Add(args) => args.iter().try_fold(Rational::new(), |acc, &a| {
            Some(acc + exact_rational(a, pool)?)
        }),
        ExprData::Mul(args) => args.iter().try_fold(Rational::from(1), |acc, &a| {
            Some(acc * exact_rational(a, pool)?)
        }),
        ExprData::Pow { base, exp } => {
            let ExprData::Integer(k) = pool.get(exp) else {
                return None;
            };
            let k = k.0.to_i32()?;
            let b = exact_rational(base, pool)?;
            if k < 0 && b == 0 {
                return None;
            }
            use rug::ops::Pow;
            Some(b.pow(k))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Symbolic (parameter) coefficients
// ---------------------------------------------------------------------------

/// Distinct roots in `p.vars[var_pos]` of a multivariate integer polynomial
/// whose other variables are symbolic parameters, each with its multiplicity.
///
/// `p` is factored over ℤ[x, params] first. A factor free of `x` is a
/// condition on the parameters only and contributes no root (the generic
/// reading — the caller states the non-vanishing hypothesis it rests on).
/// A factor free of the parameters goes through [`roots_over_q`]. A factor
/// that mentions both is solved by the quadratic formula, or as an even
/// polynomial in `x` whose half is quadratic; anything larger is `Err(degree)`.
pub(crate) fn roots_with_parameters(
    p: &MultiPoly,
    var_pos: usize,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, u32)>, usize> {
    let fac = p.factor_z().map_err(|_| p.degree_in(var_pos) as usize)?;
    let mut out = Vec::new();
    for (f, e) in &fac.factors {
        let d = f.degree_in(var_pos) as usize;
        if d == 0 {
            continue;
        }
        let only_x = f
            .terms
            .keys()
            .all(|exp| exp.iter().enumerate().all(|(i, &k)| i == var_pos || k == 0));
        if only_x {
            let mut c = vec![Integer::new(); d + 1];
            for (exp, coeff) in &f.terms {
                let k = exp.get(var_pos).copied().unwrap_or(0) as usize;
                c[k] = coeff.clone();
            }
            for (r, m) in roots_over_q(&c, pool)? {
                out.push((r, m * e));
            }
            continue;
        }
        let coeffs = coefficient_exprs(f, var_pos, d, pool);
        let roots = symbolic_radical_roots(&coeffs, pool).ok_or(d)?;
        for r in roots {
            out.push((r, *e));
        }
    }
    Ok(out)
}

/// Coefficients of `x^0 … x^d` in `f` as expressions in the parameters.
fn coefficient_exprs(f: &MultiPoly, var_pos: usize, d: usize, pool: &ExprPool) -> Vec<ExprId> {
    let mut terms: Vec<Vec<ExprId>> = vec![Vec::new(); d + 1];
    for (exp, coeff) in &f.terms {
        let k = exp.get(var_pos).copied().unwrap_or(0) as usize;
        let mut factors = vec![pool.integer(coeff.clone())];
        for (i, &e) in exp.iter().enumerate() {
            if i == var_pos || e == 0 {
                continue;
            }
            let v = f.vars[i];
            factors.push(if e == 1 {
                v
            } else {
                pool.pow(v, pool.integer(e))
            });
        }
        terms[k].push(if factors.len() == 1 {
            factors[0]
        } else {
            pool.mul(factors)
        });
    }
    terms
        .into_iter()
        .map(|t| match t.len() {
            0 => pool.integer(0),
            1 => t[0],
            _ => pool.add(t),
        })
        .collect()
}

/// Roots of `Σ coeffs[k]·x^k` with symbolic coefficients: the quadratic
/// formula, or `±√r` over the roots of the half-degree polynomial when every
/// odd coefficient is an exact zero. `None` beyond that.
pub(crate) fn symbolic_radical_roots(coeffs: &[ExprId], pool: &ExprPool) -> Option<Vec<ExprId>> {
    let is_zero = |e: ExprId| exact_rational(e, pool).is_some_and(|v| v == 0);
    let mut d = coeffs.len().checked_sub(1)?;
    while d > 0 && is_zero(coeffs[d]) {
        d -= 1;
    }
    match d {
        0 => Some(Vec::new()),
        1 => {
            let neg = pool.mul(vec![pool.integer(-1), coeffs[0]]);
            Some(vec![
                pool.mul(vec![neg, pool.pow(coeffs[1], pool.integer(-1))])
            ])
        }
        2 => Some(quadratic_roots(coeffs[2], coeffs[1], coeffs[0], pool)),
        _ if (1..=d).step_by(2).all(|k| is_zero(coeffs[k])) => {
            let half: Vec<ExprId> = coeffs[..=d].iter().step_by(2).copied().collect();
            let inner = symbolic_radical_roots(&half, pool)?;
            Some(plus_minus_sqrt(&inner, pool))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    /// Evaluate every root and check `|p(r)|` is tiny relative to the terms.
    fn assert_all_roots(c: &[i64], roots: &[(ExprId, u32)], pool: &ExprPool) {
        for &(r, _) in roots {
            let z = crate::eval::eval_complex_f64(r, pool, &HashMap::new())
                .unwrap_or_else(|e| panic!("{}: {e:?}", pool.display(r)));
            let (mut re, mut im, mut mag) = (0.0f64, 0.0f64, 0.0f64);
            let az = z.re.hypot(z.im);
            for &k in c.iter().rev() {
                let (nr, ni) = (re * z.re - im * z.im + k as f64, re * z.im + im * z.re);
                re = nr;
                im = ni;
                mag = mag * az + (k as f64).abs();
            }
            assert!(
                re.hypot(im) <= 1e-9 * mag.max(1.0),
                "{} is not a root of {c:?}: residual {re}+{im}i",
                pool.display(r)
            );
        }
    }

    #[test]
    fn a_cubic_with_three_rational_roots() {
        let pool = ExprPool::new();
        let c = [6, 11, 6, 1];
        let r = roots_over_q(&ints(&c), &pool).unwrap();
        let mut vals: Vec<ExprId> = r.iter().map(|p| p.0).collect();
        vals.sort();
        let mut want = vec![pool.integer(-1), pool.integer(-2), pool.integer(-3)];
        want.sort();
        assert_eq!(vals, want);
    }

    #[test]
    fn repeated_roots_carry_their_multiplicity() {
        let pool = ExprPool::new();
        // x³ (x − 1)²
        let r = roots_over_q(&ints(&[0, 0, 0, 1, -2, 1]), &pool).unwrap();
        assert_eq!(r.len(), 2);
        assert!(r.contains(&(pool.integer(0), 3)));
        assert!(r.contains(&(pool.integer(1), 2)));
    }

    #[test]
    fn biquadratic_and_cyclotomic_quartics_are_radicals() {
        let pool = ExprPool::new();
        for c in [
            vec![3, 0, -30, 0, 35], // 8·P₄
            vec![1, 0, 0, 0, 1],    // x⁴ + 1
            vec![1, 1, 1, 1, 1],    // Φ₅, palindromic
            vec![1, 0, 0, 0, 0, 0, 0, 0, 1],
            vec![-2, 0, 0, 1], // x³ − 2
            vec![5, 0, 0, 1],  // x³ + 5
        ] {
            let r = roots_over_q(&ints(&c), &pool).unwrap();
            assert_eq!(r.len(), c.len() - 1, "{c:?}");
            assert!(
                r.iter()
                    .all(|&(e, _)| !pool.display(e).to_string().contains("RootOf")),
                "{c:?}"
            );
            assert_all_roots(&c, &r, &pool);
        }
    }

    #[cfg(flint_arb)]
    #[test]
    fn an_irreducible_quintic_is_root_of() {
        let pool = ExprPool::new();
        let c = [-1, -1, 0, 0, 0, 1]; // x⁵ − x − 1
        let r = roots_over_q(&ints(&c), &pool).unwrap();
        assert_eq!(r.len(), 5);
        for (k, &(e, m)) in r.iter().enumerate() {
            assert_eq!(m, 1);
            assert_eq!(
                pool.display(e).to_string(),
                format!("RootOf(x^5 - x - 1, {k})")
            );
        }
        assert_all_roots(&c, &r, &pool);
    }
}
