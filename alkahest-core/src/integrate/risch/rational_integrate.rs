//! Rational-function integration via Rothstein–Trager (Risch Gap 3).
//!
//! Computes the **logarithmic part** of `∫ A(x)/D(x) dx` for a rational function
//! over ℚ.  The classical pipeline is:
//!
//! 1. **Polynomial division** `A = Q·D + R` (deg R < deg D): `∫Q` is elementary.
//! 2. **Rothstein–Trager** on the proper part `R/D` with `D` squarefree, monic,
//!    `gcd(R, D) = 1`:
//!    ```text
//!      R(t) = res_x(R − t·D′, D),
//!      ∫ R/D dx = Σ_{c : R(c)=0}  c · log( gcd_x(R − c·D′, D) ).
//!    ```
//!    The roots `c` of the **resultant** are the residues; each contributes one
//!    logarithm whose argument is a polynomial GCD.
//!
//! ## Scope of this implementation
//!
//! - **Hermite reduction** (Yun squarefree factorization + partial fractions +
//!   per-factor reduction) handles **repeated factors**, peeling off the rational
//!   part of the antiderivative and leaving a proper fraction with a squarefree
//!   denominator for the logarithmic part.
//! - **Rothstein–Trager** then produces the logs for the squarefree remainder
//!   when its resultant roots are **rational** (a ℚ-linear combination of `log`s
//!   of ℚ-polynomials).
//! - Returns `None` (so the caller falls back) when the resultant has
//!   **non-rational roots** — e.g. `1/(x²+1)` (residues `±i/2`), whose
//!   antiderivative needs `arctan`/algebraic-number logs. (Handled separately in
//!   the arctan path once `atan` differentiation is available.)
//!
//! References: Rothstein (1976); Trager (1976); Bronstein (2005) §2.2–2.5;
//! SymPy `sympy/integrals/risch.py` (`residue_reduce`, `log_part`).

use rug::{Integer, Rational};

use crate::kernel::{Domain, ExprId, ExprPool};
use crate::poly::UniPoly;

use super::poly_rde::{
    degree, poly_add, poly_deriv, poly_integrate, poly_mul, poly_one, poly_scale, poly_zero,
    qpoly_to_expr, rational_to_expr, trim, QPoly,
};
use super::rational_rde::{
    expr_to_qrational, poly_div_exact, poly_divrem, poly_gcd, poly_monic, poly_sub,
};

/// Attempt to integrate `expr` as a rational function of `var`.
///
/// Returns `Some(F)` with the antiderivative (no constant of integration) when
/// the Rothstein–Trager path succeeds, or `None` when `expr` is not a rational
/// function or falls outside the supported subset (see module docs), so the
/// caller can fall back to its existing behaviour.
pub fn try_integrate_rational(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (a, d) = expr_to_qrational(expr, var, pool)?;
    let a = trim(a);
    let d = trim(d);
    if d.is_empty() {
        return None; // division by zero — malformed
    }
    if degree(&d) < 1 {
        return None; // polynomial integrand — leave to the rule engine
    }

    // Normalise: make D monic and reduce A/D to lowest terms.
    let lc_inv = Rational::from(1) / d[degree(&d) as usize].clone();
    let d = poly_scale(&d, &lc_inv);
    let a = poly_scale(&a, &lc_inv);
    let g = poly_gcd(&a, &d);
    let a = poly_div_exact(&a, &g);
    let d = poly_monic(&poly_div_exact(&d, &g));
    if degree(&d) < 1 {
        return None; // reduced to a polynomial
    }

    // Polynomial part: A = Q·D + R.
    let (q, r) = poly_divrem(&a, &d);
    let poly_int = poly_integrate(&q);
    let r = trim(r);

    let mut terms: Vec<ExprId> = Vec::new();
    if !trim(poly_int.clone()).is_empty() {
        terms.push(qpoly_to_expr(&poly_int, var, pool));
    }

    if !r.is_empty() {
        // Hermite reduction: split R/D into a rational part (added directly) plus a
        // proper fraction H/Drad with a *squarefree* denominator for the log part.
        let (rational_terms, h, drad) = hermite_reduce(&r, &d, var, pool)?;
        terms.extend(rational_terms);

        let h = trim(h);
        if !h.is_empty() {
            // Reduce H/Drad and apply Rothstein–Trager to the squarefree remainder.
            let g = poly_gcd(&h, &drad);
            let h = poly_div_exact(&h, &g);
            let drad = poly_monic(&poly_div_exact(&drad, &g));
            if degree(&drad) >= 1 {
                let dprime = poly_deriv(&drad);
                let logs = rothstein_trager(&h, &drad, &dprime, var, pool)?;
                terms.extend(logs);
            }
        }
    }

    Some(match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    })
}

// ---------------------------------------------------------------------------
// Hermite reduction (Bronstein §2.2)
// ---------------------------------------------------------------------------

/// Remainder of `a mod m`.
fn poly_mod(a: &QPoly, m: &QPoly) -> QPoly {
    poly_divrem(a, m).1
}

/// Extended GCD over ℚ[x]: returns `(g, s, t)` with `s·a + t·b = g`, `g` monic.
fn ext_gcd(a: &QPoly, b: &QPoly) -> (QPoly, QPoly, QPoly) {
    let (mut old_r, mut r) = (trim(a.clone()), trim(b.clone()));
    let (mut old_s, mut s) = (poly_one(), poly_zero());
    let (mut old_t, mut t) = (poly_zero(), poly_one());
    while !r.is_empty() {
        let (q, rem) = poly_divrem(&old_r, &r);
        old_r = r;
        r = rem;
        let ns = poly_sub(&old_s, &poly_mul(&q, &s));
        old_s = s;
        s = ns;
        let nt = poly_sub(&old_t, &poly_mul(&q, &t));
        old_t = t;
        t = nt;
    }
    let dg = degree(&old_r);
    if dg < 0 {
        return (poly_zero(), old_s, old_t);
    }
    let inv = Rational::from(1) / old_r[dg as usize].clone();
    (
        poly_scale(&old_r, &inv),
        poly_scale(&old_s, &inv),
        poly_scale(&old_t, &inv),
    )
}

/// Inverse of `w` modulo `v` (requires `gcd(w, v) = 1`), else `None`.
fn mod_inverse(w: &QPoly, v: &QPoly) -> Option<QPoly> {
    let (g, s, _t) = ext_gcd(w, v);
    if degree(&g) != 0 {
        return None; // not coprime
    }
    Some(poly_mod(&s, v))
}

/// Yun squarefree factorization of a monic polynomial: returns `(Vᵢ, i)` for
/// each non-constant `Vᵢ`, with `f = ∏ Vᵢ^i` and the `Vᵢ` squarefree & coprime.
fn yun(f: &QPoly) -> Option<Vec<(QPoly, usize)>> {
    let f = poly_monic(f);
    if degree(&f) <= 0 {
        return Some(vec![]);
    }
    let fp = poly_deriv(&f);
    let a0 = poly_gcd(&f, &fp);
    if degree(&a0) == 0 {
        return Some(vec![(f, 1)]); // already squarefree
    }
    let mut b = poly_div_exact(&f, &a0);
    let c = poly_div_exact(&fp, &a0);
    let mut d = poly_sub(&c, &poly_deriv(&b));
    let mut result = Vec::new();
    let mut i = 1usize;
    let cap = degree(&f) as usize + 2;
    while degree(&b) > 0 {
        if i > cap {
            return None; // safety: should never trigger in characteristic 0
        }
        let vi = poly_gcd(&b, &d);
        if degree(&vi) > 0 {
            result.push((poly_monic(&vi), i));
        }
        let b_next = poly_div_exact(&b, &vi);
        let c_next = poly_div_exact(&d, &vi);
        d = poly_sub(&c_next, &poly_deriv(&b_next));
        b = b_next;
        i += 1;
    }
    Some(result)
}

/// Partial-fraction decomposition over pairwise-coprime moduli: returns `Aᵢ`
/// with `num/∏mᵢ = Σ Aᵢ/mᵢ` and `deg Aᵢ < deg mᵢ`.
fn partial_fractions(num: &QPoly, moduli: &[QPoly]) -> Option<Vec<QPoly>> {
    let n = moduli.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(vec![poly_mod(num, &moduli[0])]);
    }
    let mut result = Vec::with_capacity(n);
    let mut cur = trim(num.clone());
    for i in 0..n - 1 {
        let mi = &moduli[i];
        let rest = moduli[i + 1..]
            .iter()
            .fold(poly_one(), |acc, m| poly_mul(&acc, m));
        let (g, _s, t) = ext_gcd(mi, &rest);
        if degree(&g) != 0 {
            return None; // moduli not coprime
        }
        let ai = poly_mod(&poly_mul(&cur, &t), mi);
        let s = poly_div_exact(&poly_sub(&cur, &poly_mul(&ai, &rest)), mi);
        result.push(ai);
        cur = s;
    }
    result.push(cur);
    Some(result)
}

/// Hermite reduction of `aᵢ / V^k` (`V` squarefree, the surrounding cofactor is 1).
///
/// Returns the rational-part terms `B / V^p` (as `(B, p)`) and the leftover
/// numerator over `V^1` (the contribution to the squarefree logarithmic part).
fn hermite_factor(ai: &QPoly, v: &QPoly, k: usize) -> Option<(Vec<(QPoly, usize)>, QPoly)> {
    let vp = poly_deriv(v);
    let mut a = trim(ai.clone());
    let mut terms = Vec::new();
    let mut power = k;
    while power >= 2 {
        let factor = Rational::from((power - 1) as i64);
        let coeff = poly_mod(&poly_scale(&vp, &factor), v);
        let inv = mod_inverse(&coeff, v)?;
        let b = poly_mod(&poly_mul(&poly_scale(&a, &Rational::from(-1)), &inv), v);
        // numerator = a − (B'·V − (k−1)·V'·B)
        let inner = poly_sub(
            &poly_mul(&poly_deriv(&b), v),
            &poly_scale(&poly_mul(&vp, &b), &factor),
        );
        a = poly_div_exact(&poly_sub(&a, &inner), v);
        terms.push((b, power - 1));
        power -= 1;
    }
    Some((terms, a))
}

/// Full Hermite reduction of a proper fraction `r/d` (`d` monic, `gcd(r,d)=1`).
///
/// Returns `(rational_terms, H, Drad)`: the symbolic rational part, and the
/// proper fraction `H/Drad` (`Drad` squarefree) feeding the logarithmic part.
fn hermite_reduce(
    r: &QPoly,
    d: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(Vec<ExprId>, QPoly, QPoly)> {
    let sqf = yun(d)?;
    if sqf.is_empty() {
        return Some((vec![], r.clone(), d.clone()));
    }
    let moduli: Vec<QPoly> = sqf
        .iter()
        .map(|(v, i)| {
            let mut m = poly_one();
            for _ in 0..*i {
                m = poly_mul(&m, v);
            }
            m
        })
        .collect();
    let parts = partial_fractions(r, &moduli)?;

    let drad: QPoly = sqf.iter().fold(poly_one(), |acc, (v, _)| poly_mul(&acc, v));
    let mut rational_terms: Vec<ExprId> = Vec::new();
    let mut h = poly_zero();

    for ((v, i), ai) in sqf.iter().zip(parts.iter()) {
        let cofactor = poly_div_exact(&drad, v); // Drad / V_i
        if *i == 1 {
            // Already squarefree: the whole part feeds the log part.
            h = poly_add(&h, &poly_mul(ai, &cofactor));
            continue;
        }
        let (terms, leftover) = hermite_factor(ai, v, *i)?;
        let v_expr = qpoly_to_expr(v, var, pool);
        for (b, p) in terms {
            let b_expr = qpoly_to_expr(&b, var, pool);
            let v_pow = pool.pow(v_expr, pool.integer(-(p as i32)));
            rational_terms.push(pool.mul(vec![b_expr, v_pow]));
        }
        h = poly_add(&h, &poly_mul(&leftover, &cofactor));
    }

    Some((rational_terms, trim(h), drad))
}

/// Rothstein–Trager logarithmic part of `∫ R/D dx` (`D` squarefree, monic,
/// `gcd(R, D) = 1`).  Returns `None` if any resultant root is non-rational.
fn rothstein_trager(
    r: &QPoly,
    d: &QPoly,
    dprime: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    // Build R(t) = res_x(R − t·D′, D) over ℤ.  Clear denominators of R and D′ by a
    // *single* common factor L so the resultant's roots are exactly the residues.
    let l = lcm_denoms(&[r, dprime]);
    let r_int = poly_scale(r, &Rational::from(l.clone()));
    let dp_int = poly_scale(dprime, &Rational::from(l));
    let m = lcm_denoms(&[d]);
    let d_int = poly_scale(d, &Rational::from(m));

    let t = pool.symbol("$rt_param$", Domain::Complex);
    let r_expr = qpoly_to_expr(&r_int, var, pool);
    let dp_expr = qpoly_to_expr(&dp_int, var, pool);
    // P(x, t) = R(x) − t·D′(x)
    let p_expr = pool.add(vec![
        r_expr,
        pool.mul(vec![pool.integer(-1_i32), t, dp_expr]),
    ]);
    let d_expr = qpoly_to_expr(&d_int, var, pool);

    let res = crate::poly::resultant(p_expr, d_expr, var, pool).ok()?;
    let rt_poly = UniPoly::from_symbolic(res.value, t, pool).ok()?;
    if rt_poly.degree() < 1 {
        return None;
    }
    let fac = rt_poly.factor_z().ok()?;

    // Collect distinct rational roots (residues); bail on any non-linear factor.
    let mut residues: Vec<Rational> = Vec::new();
    for (f, _mult) in &fac.factors {
        if f.degree() != 1 {
            return None; // non-rational residue — needs algebraic extension / arctan
        }
        let coeffs = f.coefficients(); // ascending: [b, a] for a·t + b
        let b = Rational::from(coeffs[0].clone());
        let a = Rational::from(coeffs[1].clone());
        let c = -b / a;
        if !residues.contains(&c) {
            residues.push(c);
        }
    }

    // Each residue c contributes c·log(gcd_x(R − c·D′, D)).
    let mut logs: Vec<ExprId> = Vec::new();
    let mut covered = 0i64;
    for c in &residues {
        let shifted = poly_sub(r, &poly_scale(dprime, c));
        let gc = poly_monic(&poly_gcd(&shifted, d));
        if degree(&gc) <= 0 {
            continue;
        }
        covered += degree(&gc);
        let arg = qpoly_to_expr(&gc, var, pool);
        let log_arg = pool.func("log", vec![arg]);
        let term = if *c == 1 {
            log_arg
        } else {
            pool.mul(vec![rational_to_expr(c, pool), log_arg])
        };
        logs.push(term);
    }

    // Soundness guard: every root of D must be accounted for by some residue.
    if covered != degree(d) {
        return None;
    }
    Some(logs)
}

/// LCM of all coefficient denominators across the given polynomials.
fn lcm_denoms(polys: &[&QPoly]) -> Integer {
    let mut l = Integer::from(1);
    for p in polys {
        for c in p.iter() {
            l = l.lcm(c.denom());
        }
    }
    l
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ExprData;

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    /// Numeric evaluator for verification (Integer/Rational/Add/Mul/Pow/log).
    fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args.iter().map(|&a| eval(a, x, xv, pool)).product(),
            ExprData::Pow { base, exp } => eval(base, x, xv, pool).powf(eval(exp, x, xv, pool)),
            ExprData::Func { ref name, ref args } if name == "log" && args.len() == 1 => {
                eval(args[0], x, xv, pool).ln()
            }
            other => panic!("eval: unsupported {other:?}"),
        }
    }

    /// Verify d/dx F = integrand numerically at several points.
    fn verify(expr: ExprId, antideriv: ExprId, x: ExprId, pool: &ExprPool) {
        let d = crate::diff::diff(antideriv, x, pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, pool).value;
        for &xv in &[1.7_f64, 2.3, 3.9] {
            let lhs = eval(ds, x, xv, pool);
            let rhs = eval(expr, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs} (F={})",
                pool.display(antideriv)
            );
        }
    }

    #[test]
    fn one_over_x2_minus_1() {
        // ∫ 1/(x²−1) dx = ½ log(x−1) − ½ log(x+1).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = poly_to_expr(
            &[Rational::from(-1), Rational::from(0), Rational::from(1)],
            x,
            &pool,
        );
        let f = pool.mul(vec![
            pool.integer(1_i32),
            pool.pow(den, pool.integer(-1_i32)),
        ]);
        let result = try_integrate_rational(f, x, &pool).expect("rational integral");
        verify(f, result, x, &pool);
    }

    #[test]
    fn two_x_over_x2_plus_1() {
        // ∫ 2x/(x²+1) dx = log(x²+1).  (Residue c = 1, rational.)
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2p1 = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.mul(vec![
            pool.integer(2_i32),
            x,
            pool.pow(x2p1, pool.integer(-1_i32)),
        ]);
        let result = try_integrate_rational(f, x, &pool).expect("rational integral");
        verify(f, result, x, &pool);
    }

    #[test]
    fn one_over_x2_minus_x() {
        // ∫ 1/(x²−x) dx = ∫ 1/(x(x−1)) dx = −log(x) + log(x−1).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = poly_to_expr(
            &[Rational::from(0), Rational::from(-1), Rational::from(1)],
            x,
            &pool,
        );
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("rational integral");
        verify(f, result, x, &pool);
    }

    #[test]
    fn improper_with_log_part() {
        // ∫ x³/(x²−1) dx = x²/2 + ½ log(x²−1)  (improper: polynomial + log part).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = poly_to_expr(
            &[Rational::from(-1), Rational::from(0), Rational::from(1)],
            x,
            &pool,
        );
        let f = pool.mul(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.pow(den, pool.integer(-1_i32)),
        ]);
        let result = try_integrate_rational(f, x, &pool).expect("rational integral");
        verify(f, result, x, &pool);
    }

    #[test]
    fn complex_residues_unsupported() {
        // ∫ 1/(x²+1) dx needs arctan (residues ±i/2) — not handled here → None.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        assert!(try_integrate_rational(f, x, &pool).is_none());
    }

    #[test]
    fn hermite_one_over_x_plus_1_squared() {
        // ∫ 1/(x+1)² dx = −1/(x+1)  (pure rational part, no log).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
        let f = pool.pow(xp1, pool.integer(-2_i32));
        let result = try_integrate_rational(f, x, &pool).expect("Hermite reduction");
        verify(f, result, x, &pool);
    }

    #[test]
    fn hermite_repeated_with_log_part() {
        // ∫ 1/(x²(x+1)) dx = −1/x − log(x) + log(x+1): rational part + two logs.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
        let den = pool.mul(vec![x2, xp1]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("Hermite + RT");
        verify(f, result, x, &pool);
        assert!(
            pool.display(result).to_string().contains("log"),
            "expected a logarithmic part"
        );
    }

    #[test]
    fn hermite_cubic_repeated() {
        // ∫ x/(x−1)³ dx — repeated linear factor of multiplicity 3.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let xm1 = pool.add(vec![x, pool.integer(-1_i32)]);
        let f = pool.mul(vec![x, pool.pow(xm1, pool.integer(-3_i32))]);
        let result = try_integrate_rational(f, x, &pool).expect("Hermite reduction");
        verify(f, result, x, &pool);
    }

    #[test]
    fn hermite_mixed_multiplicity() {
        // ∫ (2x+1)/((x+1)²(x+2)) dx — multiplicity 2 and 1 factors together.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
        let xp2 = pool.add(vec![x, pool.integer(2_i32)]);
        let den = pool.mul(vec![pool.pow(xp1, pool.integer(2_i32)), xp2]);
        let num = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let f = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
        let result = try_integrate_rational(f, x, &pool).expect("Hermite + RT");
        verify(f, result, x, &pool);
    }

    fn poly_to_expr(coeffs: &[Rational], x: ExprId, pool: &ExprPool) -> ExprId {
        qpoly_to_expr(&coeffs.to_vec(), x, pool)
    }
}
