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
//! - Otherwise, a partial-fraction pass factors the squarefree denominator over
//!   ℚ and integrates each irreducible factor:
//!   * linear → `log`;
//!   * irreducible quadratic → `log` + `arctan` (negative discriminant) or `log`
//!     with `√Δ` coefficients (positive discriminant);
//!   * irreducible factor of **degree ≥ 3** → a [`crate::kernel::ExprData::RootSum`]
//!     over the degree-≥3 algebraic residues (Lazard–Rioboo–Trager): the residue
//!     minimal polynomial is an irreducible factor of `R(t) = res_x(N − t·P', P)`,
//!     and the log argument `gcd_x(N − t·P', P)` is computed in the number field
//!     `ℚ[t]/Q(t)`.
//!
//! Thus `∫ A/D` is complete for every denominator that factors over ℚ — the only
//! `None` results are non-rational integrands handled elsewhere.
//!
//! References: Rothstein (1976); Trager (1976); Lazard & Rioboo (1990);
//! Bronstein (2005) §2.2–2.5; SymPy `sympy/integrals/risch.py`.

use rug::{Integer, Rational};

use crate::kernel::{Domain, ExprId, ExprPool};
use crate::poly::UniPoly;

use super::number_field::{ext_gcd, mod_inverse, poly_mod, NumberField};
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
                // Rothstein–Trager for rational residues; otherwise fall back to a
                // partial-fraction pass that emits log + arctan for irreducible
                // quadratic factors.
                let logs = match rothstein_trager(&h, &drad, &dprime, var, pool) {
                    Some(logs) => logs,
                    None => partial_fraction_log_arctan(&h, &drad, var, pool)?,
                };
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

// The ℚ[x] modular helpers `poly_mod`, `ext_gcd`, and `mod_inverse` live in
// [`super::number_field`] (shared with the number-field arithmetic below).

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
// Degree-≥3 algebraic residues: RootSum via Rothstein–Trager over ℚ[t]/Q
// ---------------------------------------------------------------------------

/// Build the Rothstein–Trager resultant `R(t) = res_x(num − t·dprime, d)` as a
/// ℚ-polynomial in the parameter symbol `param`.
fn resultant_param_poly(
    num: &QPoly,
    dprime: &QPoly,
    d: &QPoly,
    var: ExprId,
    param: ExprId,
    pool: &ExprPool,
) -> Option<QPoly> {
    // Clear denominators of num and dprime by one common factor so the parameter
    // roots are exactly the residues; d by its own factor (constant in t).
    let l = lcm_denoms(&[num, dprime]);
    let num_int = poly_scale(num, &Rational::from(l.clone()));
    let dp_int = poly_scale(dprime, &Rational::from(l));
    let m = lcm_denoms(&[d]);
    let d_int = poly_scale(d, &Rational::from(m));

    let num_expr = qpoly_to_expr(&num_int, var, pool);
    let dp_expr = qpoly_to_expr(&dp_int, var, pool);
    let p_expr = pool.add(vec![
        num_expr,
        pool.mul(vec![pool.integer(-1_i32), param, dp_expr]),
    ]);
    let d_expr = qpoly_to_expr(&d_int, var, pool);
    let res = crate::poly::resultant(p_expr, d_expr, var, pool).ok()?;
    let up = UniPoly::from_symbolic(res.value, param, pool).ok()?;
    if up.degree() < 1 {
        return None;
    }
    Some(
        up.coefficients()
            .iter()
            .map(|c| Rational::from(c.clone()))
            .collect(),
    )
}

/// The Lazard–Rioboo–Trager log argument `S(t, x) = gcd_x(num − t·dprime, d)`
/// computed over the number field `K = ℚ[t]/Q` (see [`super::number_field`]),
/// returned as a symbolic polynomial in `x` whose coefficients are expressions
/// in the root symbol `rvar`.
fn alg_log_argument(
    num: &QPoly,
    dprime: &QPoly,
    d: &QPoly,
    q: &QPoly,
    var: ExprId,
    rvar: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let k = NumberField::new(q.clone());
    let width = (degree(num).max(degree(dprime)) + 1).max(0) as usize;
    // A = num − t·dprime  (K-poly in x): A[i] = num[i] − dprime[i]·t.
    let a: Vec<QPoly> = (0..width)
        .map(|i| {
            let ni = coeff_at(num, i);
            let ppi = coeff_at(dprime, i);
            k.reduce(&vec![ni, -ppi])
        })
        .collect();
    // B = d  (K-poly in x): each coefficient is a K-constant.
    let b: Vec<QPoly> = d
        .iter()
        .map(|c| {
            if *c == 0 {
                poly_zero()
            } else {
                vec![c.clone()]
            }
        })
        .collect();

    let s = k.kpoly_gcd(&a, &b)?;
    if NumberField::kdeg(&s) < 1 {
        return None; // no nontrivial common factor — not a valid residue
    }
    // Build Σ_i coeff_i(rvar) · x^i.
    let mut terms: Vec<ExprId> = Vec::new();
    for (i, c) in s.iter().enumerate() {
        if NumberField::is_zero(c) {
            continue;
        }
        let c_expr = qpoly_to_expr(c, rvar, pool);
        let term = match i {
            0 => c_expr,
            1 => pool.mul(vec![c_expr, var]),
            _ => pool.mul(vec![c_expr, pool.pow(var, pool.integer(i as i32))]),
        };
        terms.push(term);
    }
    Some(match terms.len() {
        0 => return None,
        1 => terms[0],
        _ => pool.add(terms),
    })
}

/// Coefficient of `x^i` in a QPoly, or 0.
fn coeff_at(p: &QPoly, i: usize) -> Rational {
    p.get(i).cloned().unwrap_or_else(|| Rational::from(0))
}

// ---------------------------------------------------------------------------
// Non-rational residues: irreducible-quadratic → log + arctan
// ---------------------------------------------------------------------------

/// Factor a monic polynomial over ℚ into its monic irreducible factors
/// (multiplicities expanded), via FLINT integer factorization.
fn factor_monic_q(d: &QPoly, var: ExprId, pool: &ExprPool) -> Option<Vec<QPoly>> {
    let m = lcm_denoms(&[d]);
    let d_int = poly_scale(d, &Rational::from(m));
    let d_expr = qpoly_to_expr(&d_int, var, pool);
    let up = UniPoly::from_symbolic(d_expr, var, pool).ok()?;
    let fac = up.factor_z().ok()?;
    let mut factors = Vec::new();
    for (f, mult) in fac.factors {
        let qp: QPoly = f
            .coefficients()
            .iter()
            .map(|c| Rational::from(c.clone()))
            .collect();
        let qp = poly_monic(&qp);
        for _ in 0..mult {
            factors.push(qp.clone());
        }
    }
    Some(factors)
}

/// Coefficient of `x^i`, or 0.
fn at(p: &QPoly, i: usize) -> Rational {
    p.get(i).cloned().unwrap_or_else(|| Rational::from(0))
}

/// `c · e`, collapsing `c = 1`.
fn scaled(c: &Rational, e: ExprId, pool: &ExprPool) -> ExprId {
    if *c == 1 {
        e
    } else {
        pool.mul(vec![rational_to_expr(c, pool), e])
    }
}

/// Integrate a proper fraction `h/d` (`d` squarefree, monic, `gcd(h,d)=1`) by
/// partial fractions over the ℚ-irreducible factorization:
///   - **linear** factor `x+a` → `c·log(x+a)`;
///   - **irreducible quadratic** `x²+bx+c0` with discriminant `Δ = b²−4c0`:
///     * `Δ < 0` → `(M/2)·log(x²+bx+c0) + ((2N−Mb)/√(−Δ))·atan((2x+b)/√(−Δ))`;
///     * `Δ > 0` → `(M/2)·log(x²+bx+c0) + ((2N−Mb)/(2√Δ))·[log(2x+b−√Δ) − log(2x+b+√Δ)]`
///       (real irrational roots; the antiderivative carries `√Δ` coefficients).
///
/// Returns `None` for any irreducible factor of degree ≥ 3, which would require a
/// symbolic `RootSum` over the (degree ≥ 3) algebraic residues — see module docs.
fn partial_fraction_log_arctan(
    h: &QPoly,
    d: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    let factors = factor_monic_q(d, var, pool)?;
    if factors.is_empty() {
        return None;
    }
    let nums = partial_fractions(h, &factors)?;
    let mut terms: Vec<ExprId> = Vec::new();

    for (p, n) in factors.iter().zip(nums.iter()) {
        let n = trim(n.clone());
        if n.is_empty() {
            continue;
        }
        match degree(p) {
            1 => {
                // p = x + a (monic); n is a constant c → c·log(p).
                let p_expr = qpoly_to_expr(p, var, pool);
                let logp = pool.func("log", vec![p_expr]);
                terms.push(scaled(&at(&n, 0), logp, pool));
            }
            2 => {
                // p = x² + b·x + c0;  n = M·x + N.
                let b = at(p, 1);
                let c0 = at(p, 0);
                let big_m = at(&n, 1);
                let big_n = at(&n, 0);

                // Log part: (M/2)·log(x² + b·x + c0).
                if big_m != 0 {
                    let p_expr = qpoly_to_expr(p, var, pool);
                    let logp = pool.func("log", vec![p_expr]);
                    terms.push(scaled(&(big_m.clone() / Rational::from(2)), logp, pool));
                }

                let coeff_num = Rational::from(2) * big_n.clone() - big_m.clone() * b.clone();
                let disc = b.clone() * b.clone() - Rational::from(4) * c0.clone();
                let two_x_plus_b = pool.add(vec![
                    pool.mul(vec![pool.integer(2_i32), var]),
                    rational_to_expr(&b, pool),
                ]);

                if disc < 0 {
                    // Complex conjugate roots → arctan.
                    // (2N − Mb)/√(−Δ) · atan((2x + b)/√(−Δ)).
                    if coeff_num != 0 {
                        let neg_disc = Rational::from(-1) * disc.clone();
                        let sqrt = pool.func("sqrt", vec![rational_to_expr(&neg_disc, pool)]);
                        let sqrt_inv = pool.pow(sqrt, pool.integer(-1_i32));
                        let arg = pool.mul(vec![two_x_plus_b, sqrt_inv]);
                        let atan = pool.func("atan", vec![arg]);
                        let coeff = pool.mul(vec![rational_to_expr(&coeff_num, pool), sqrt_inv]);
                        terms.push(pool.mul(vec![coeff, atan]));
                    }
                } else {
                    // disc > 0 (irreducible ⇒ Δ not a perfect square): real irrational
                    // roots → log with √Δ.
                    // (2N − Mb)/(2√Δ) · [log(2x+b−√Δ) − log(2x+b+√Δ)].
                    if coeff_num != 0 {
                        let sqrt = pool.func("sqrt", vec![rational_to_expr(&disc, pool)]);
                        let sqrt_inv = pool.pow(sqrt, pool.integer(-1_i32));
                        let neg_sqrt = pool.mul(vec![pool.integer(-1_i32), sqrt]);
                        let arg_minus = pool.add(vec![two_x_plus_b, neg_sqrt]);
                        let arg_plus = pool.add(vec![two_x_plus_b, sqrt]);
                        let log_diff = pool.add(vec![
                            pool.func("log", vec![arg_minus]),
                            pool.mul(vec![pool.integer(-1_i32), pool.func("log", vec![arg_plus])]),
                        ]);
                        // coeff = (2N − Mb) / (2√Δ)
                        let half = Rational::from((1, 2));
                        let coeff =
                            pool.mul(vec![rational_to_expr(&(coeff_num * half), pool), sqrt_inv]);
                        terms.push(pool.mul(vec![coeff, log_diff]));
                    }
                }
            }
            _ => {
                // Irreducible factor of degree ≥ 3: residues are algebraic numbers
                // of degree ≥ 2 → emit a RootSum (Lazard–Rioboo–Trager).
                let pp = poly_deriv(p);
                let rvar = pool.symbol("$root$", Domain::Complex);
                let rt = resultant_param_poly(&n, &pp, p, var, rvar, pool)?;
                // Radical of R(t): distinct residues only.
                let rad = poly_monic(&poly_div_exact(&rt, &poly_gcd(&rt, &poly_deriv(&rt))));
                let factors_t = factor_monic_q(&rad, rvar, pool)?;
                for qf in &factors_t {
                    match degree(qf) {
                        d if d <= 0 => {}
                        1 => {
                            // Rational residue c (monic linear factor t + a₀ ⇒ c = −a₀).
                            let c = -coeff_at(qf, 0);
                            let shifted = poly_sub(&n, &poly_scale(&pp, &c));
                            let gc = poly_monic(&poly_gcd(&shifted, p));
                            if degree(&gc) >= 1 {
                                let logp = pool.func("log", vec![qpoly_to_expr(&gc, var, pool)]);
                                terms.push(scaled(&c, logp, pool));
                            }
                        }
                        _ => {
                            // RootSum(Q, t, t·log(S(t,x))).
                            let s_expr = alg_log_argument(&n, &pp, p, qf, var, rvar, pool)?;
                            let body = pool.mul(vec![rvar, pool.func("log", vec![s_expr])]);
                            let q_expr = qpoly_to_expr(qf, rvar, pool);
                            terms.push(pool.root_sum(q_expr, rvar, body));
                        }
                    }
                }
            }
        }
    }
    Some(terms)
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

    /// Numeric evaluator for verification (Integer/Rational/Add/Mul/Pow/log/…),
    /// including `RootSum` via a real-root environment binding `env`.
    fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        eval_env(expr, x, xv, &[], pool)
    }

    fn eval_env(expr: ExprId, x: ExprId, xv: f64, env: &[(ExprId, f64)], pool: &ExprPool) -> f64 {
        if expr == x {
            return xv;
        }
        for &(sym, val) in env {
            if expr == sym {
                return val;
            }
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_env(a, x, xv, env, pool)).sum(),
            ExprData::Mul(args) => {
                // Short-circuit on an exact-zero factor: `0 · log(negative)` must be
                // 0, not `0 · NaN`.  (Unsimplified `0·…` terms can appear inside a
                // differentiated RootSum body, which `simplify` leaves opaque.)
                let factors: Vec<f64> = args
                    .iter()
                    .map(|&a| eval_env(a, x, xv, env, pool))
                    .collect();
                if factors.contains(&0.0) {
                    0.0
                } else {
                    factors.iter().product()
                }
            }
            ExprData::Pow { base, exp } => {
                let b = eval_env(base, x, xv, env, pool);
                // Use integer power when possible — `powf` returns NaN for a
                // negative base with a (float-typed) integer exponent.
                if let ExprData::Integer(n) = pool.get(exp) {
                    if let Some(k) = n.0.to_i32() {
                        return b.powi(k);
                    }
                }
                b.powf(eval_env(exp, x, xv, env, pool))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval_env(args[0], x, xv, env, pool);
                match name.as_str() {
                    "log" => a.ln(),
                    "atan" => a.atan(),
                    "sqrt" => a.sqrt(),
                    other => panic!("eval: unsupported func {other}"),
                }
            }
            ExprData::RootSum { poly, var, body } => {
                // Σ over the real roots of `poly` of body[var := root].
                let coeffs = real_coeffs(poly, var, pool);
                real_roots_f64(&coeffs)
                    .into_iter()
                    .map(|r| {
                        let mut e = env.to_vec();
                        e.push((var, r));
                        eval_env(body, x, xv, &e, pool)
                    })
                    .sum()
            }
            other => panic!("eval: unsupported {other:?}"),
        }
    }

    /// Real coefficient vector (ascending) of a polynomial in `var`.
    fn real_coeffs(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<f64> {
        if expr == var {
            return vec![0.0, 1.0];
        }
        match pool.get(expr) {
            ExprData::Integer(n) => vec![n.0.to_f64()],
            ExprData::Rational(r) => vec![r.0.to_f64()],
            ExprData::Add(args) => {
                let mut acc = vec![0.0];
                for a in &args {
                    let c = real_coeffs(*a, var, pool);
                    if c.len() > acc.len() {
                        acc.resize(c.len(), 0.0);
                    }
                    for (i, v) in c.iter().enumerate() {
                        acc[i] += v;
                    }
                }
                acc
            }
            ExprData::Mul(args) => {
                let mut acc = vec![1.0];
                for a in &args {
                    let c = real_coeffs(*a, var, pool);
                    let mut prod = vec![0.0; acc.len() + c.len() - 1];
                    for (i, ai) in acc.iter().enumerate() {
                        for (j, cj) in c.iter().enumerate() {
                            prod[i + j] += ai * cj;
                        }
                    }
                    acc = prod;
                }
                acc
            }
            ExprData::Pow { base, exp } => {
                let k = match pool.get(exp) {
                    ExprData::Integer(n) => n.0.to_i64().unwrap(),
                    _ => panic!("real_coeffs: non-integer exponent"),
                };
                assert!(k >= 0, "real_coeffs: negative exponent");
                let c = real_coeffs(base, var, pool);
                let mut acc = vec![1.0];
                for _ in 0..k {
                    let mut prod = vec![0.0; acc.len() + c.len() - 1];
                    for (i, ai) in acc.iter().enumerate() {
                        for (j, cj) in c.iter().enumerate() {
                            prod[i + j] += ai * cj;
                        }
                    }
                    acc = prod;
                }
                acc
            }
            other => panic!("real_coeffs: unsupported {other:?}"),
        }
    }

    /// Real roots (bracket + bisection) of a polynomial with ascending `coeffs`.
    fn real_roots_f64(coeffs: &[f64]) -> Vec<f64> {
        let horner = |t: f64| coeffs.iter().rev().fold(0.0, |acc, c| acc * t + c);
        let (lo, hi, n) = (-60.0_f64, 60.0_f64, 240_000);
        let mut roots = Vec::new();
        let step = (hi - lo) / n as f64;
        let mut prev_t = lo;
        let mut prev_v = horner(lo);
        for i in 1..=n {
            let t = lo + step * i as f64;
            let v = horner(t);
            if prev_v == 0.0 {
                roots.push(prev_t);
            } else if prev_v * v < 0.0 {
                let (mut a, mut b) = (prev_t, t);
                for _ in 0..80 {
                    let m = 0.5 * (a + b);
                    if horner(a) * horner(m) <= 0.0 {
                        b = m;
                    } else {
                        a = m;
                    }
                }
                roots.push(0.5 * (a + b));
            }
            prev_t = t;
            prev_v = v;
        }
        roots
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
    fn arctan_one_over_x2_plus_1() {
        // ∫ 1/(x²+1) dx = atan(x).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("arctan path");
        verify(f, result, x, &pool);
        assert!(pool.display(result).to_string().contains("atan"));
    }

    #[test]
    fn arctan_one_over_x2_plus_4() {
        // ∫ 1/(x²+4) dx = ½·atan(x/2).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(4_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("arctan path");
        verify(f, result, x, &pool);
    }

    #[test]
    fn log_plus_arctan_mixed_numerator() {
        // ∫ (x+1)/(x²+1) dx = ½·log(x²+1) + atan(x): both a log and an arctan term.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let num = pool.add(vec![x, pool.integer(1_i32)]);
        let f = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
        let result = try_integrate_rational(f, x, &pool).expect("log + arctan");
        verify(f, result, x, &pool);
        let s = pool.display(result).to_string();
        assert!(
            s.contains("atan") && s.contains("log"),
            "expected log and atan: {s}"
        );
    }

    #[test]
    fn linear_and_quadratic_factors() {
        // ∫ 1/((x−1)(x²+1)) dx — a linear (log) and an irreducible quadratic
        // (log + arctan) factor together.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let xm1 = pool.add(vec![x, pool.integer(-1_i32)]);
        let x2p1 = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let den = pool.mul(vec![xm1, x2p1]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("mixed factors");
        verify(f, result, x, &pool);
    }

    #[test]
    fn real_irrational_roots_via_sqrt_log() {
        // ∫ 1/(x²−2) dx = (1/(2√2))·[log(2x−2√2) − log(2x+2√2)]: irreducible
        // quadratic with positive discriminant → log with √Δ.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-2_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("sqrt-log path");
        verify(f, result, x, &pool);
        assert!(pool.display(result).to_string().contains("sqrt"));
    }

    #[test]
    fn mixed_numerator_real_quadratic() {
        // ∫ (x+3)/(x²−2) dx = ½·log(x²−2) + (3/(2√2))·[log(2x−2√2) − log(2x+2√2)].
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-2_i32)]);
        let num = pool.add(vec![x, pool.integer(3_i32)]);
        let f = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
        let result = try_integrate_rational(f, x, &pool).expect("log + sqrt-log");
        verify(f, result, x, &pool);
    }

    #[test]
    fn degree_three_real_roots_root_sum() {
        // ∫ 1/(x³−3x+1) dx — irreducible over ℚ with three real roots, so the
        // residues are real algebraic numbers of degree 3 → a RootSum.  Verified
        // by differentiation (the derivative is real-valued for real residues).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-3_i32), x]),
            pool.integer(1_i32),
        ]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("RootSum path");
        assert!(
            pool.display(result).to_string().contains("RootSum"),
            "expected a RootSum: {}",
            pool.display(result)
        );
        // Avoid the denominator's real roots (≈ −1.88, 0.35, 1.53) as test points.
        let d = crate::diff::diff(result, x, &pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, &pool).value;
        for &xv in &[3.0_f64, 5.0, -4.0] {
            let lhs = eval(ds, x, xv, &pool);
            let rhs = eval(f, x, xv, &pool);
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn degree_three_with_one_real_root_via_root_sum() {
        // ∫ 1/(x³+x+1) dx — irreducible, one real + two complex roots.  Still
        // produces a RootSum; check it integrates (full numeric check would need
        // complex roots, covered structurally here).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            x,
            pool.integer(1_i32),
        ]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("RootSum path");
        assert!(pool.display(result).to_string().contains("RootSum"));
        // The derivative must differentiate cleanly (diff support for RootSum).
        let d = crate::diff::diff(result, x, &pool);
        assert!(d.is_ok());
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
