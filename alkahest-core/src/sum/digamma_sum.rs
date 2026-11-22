//! Finite sums of rational functions of `k` through the digamma family.
//!
//! Gosper's algorithm decides whether `Σ_{k=lo}^{hi} R(k)` has a *rational*
//! antidifference; for most rational `R` it has none (`Σ_{k=1}^{n} 1/k`).
//! Mathematica and SymPy still answer those, with polygamma functions, and so
//! does this module:
//!
//! * `R` is split into partial fractions `P(k) + Σ_ρ Σ_s c_{ρ,s}/(k − ρ)^s`
//!   (every pole linear in `k`; an irreducible quadratic factor is declined);
//! * the polynomial part goes back to Gosper, which always sums a polynomial;
//! * each `Σ_{k=lo}^{hi} (k − ρ)^{−s}` is `Z_s(lo − ρ) − Z_s(hi + 1 − ρ)` with
//!   `Z_1(z) = −ψ(z)` and `Z_s(z) = ζ(s, z) = (−1)^s ψ^{(s−1)}(z)/(s − 1)!`
//!   for `s ≥ 2`. At a literal positive integer or half-integer point the
//!   exact value is used (`ψ(1) = −γ`, so `Σ_{k=1}^{n} 1/k` is
//!   `digamma(n + 1) + EulerGamma`, which is `harmonic(n)`).
//!
//! A pole inside the range — a literal `lo − ρ` that is a non-positive integer
//! — is refused, not telescoped. Every closed form is checked numerically
//! against the partial sums it claims to equal (`S(lo + j)` for several `j`,
//! with any parameters bound to fixed values); a disagreement is a refusal.

use std::collections::HashMap;

use rug::Integer;

use super::closed_form::{
    digamma_without_gamma, hurwitz_zeta, literal, neg, partial_fractions, poly_coeffs,
    rational_piece, simp, Outcome,
};
use super::expr_ratio::is_free_of_k;
use crate::deriv::log::SideCondition;
use crate::kernel::subs::subs;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

/// Largest pole order summed.
const MAX_ORDER: u32 = 12;
/// Number of partial sums the closed form is checked against.
const CHECK_POINTS: i64 = 6;

/// `Z_s(z)`: `−ψ(z)` for `s = 1`, `ζ(s, z)` for `s ≥ 2`, exact at a literal
/// positive integer or half-integer `z`.
fn z_s(s: u32, z: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let euler = pool.symbol("EulerGamma", Domain::Real);
    if let Some(v) = literal(pool, z) {
        if v <= 0 && *v.denom() == 1 {
            return None; // a pole
        }
        if s == 1 {
            // −ψ(z) = γ − (ψ(z) + γ)
            if let Some(h) = digamma_without_gamma(z, pool) {
                return Some(pool.add(vec![euler, neg(pool, h)]));
            }
            return Some(neg(pool, pool.func("digamma", vec![z])));
        }
        if v > 0 {
            return hurwitz_zeta(s as usize, z, pool);
        }
    }
    Some(match s {
        1 => neg(pool, pool.func("digamma", vec![z])),
        2 => pool.func("trigamma", vec![z]),
        _ => {
            // (−1)^s ψ^{(s−1)}(z) / (s − 1)!
            let mut fact = Integer::from(1);
            for j in 2..s {
                fact *= j;
            }
            let sign = if s % 2 == 0 { 1 } else { -1 };
            pool.mul(vec![
                pool.rational(Integer::from(sign), fact),
                pool.func("polygamma", vec![pool.integer(i64::from(s - 1)), z]),
            ])
        }
    })
}

/// `Σ_{k=lo}^{hi} term` for a rational `term` (finite, possibly symbolic
/// bounds), or `NotApplicable`.
pub(super) fn finite_rational(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Outcome {
    // Build the closed form in a fresh upper-bound symbol so it can be checked
    // against partial sums whatever `hi` is, then substitute `hi`.
    let h = pool.symbol("__sum_hi", Domain::Real);
    let mut conditions = Vec::new();
    let Some(value) = closed_form_in(term, k, lo, h, pool, &mut conditions) else {
        return Outcome::NotApplicable;
    };
    match check_partial_sums(term, k, lo, h, value, pool) {
        Check::Agrees => {}
        Check::Disagrees => {
            return Outcome::Refused(
                "the digamma closed form of this rational sum disagreed with its partial \
                 sums numerically; it is not returned"
                    .to_string(),
            )
        }
        Check::Inconclusive => return Outcome::NotApplicable,
    }
    let value = simp(pool, subs(value, &HashMap::from([(h, hi)]), pool));
    Outcome::Sum {
        value,
        conditions,
        rule: "rational_sum_digamma",
    }
}

fn closed_form_in(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    h: ExprId,
    pool: &ExprPool,
    conditions: &mut Vec<SideCondition>,
) -> Option<ExprId> {
    let term = simp(pool, term);
    // Only a genuinely rational summand: some factor must be a negative power
    // of a polynomial in `k`, and nothing may be transcendental in `k`.
    if !is_rational_in(term, k, pool) {
        return None;
    }
    // Split into polynomial part + proper partial fractions when `apart` can;
    // otherwise work on the additive pieces as they are.
    let split = match crate::poly::apart_with_conditions(term, k, pool) {
        Ok((e, conds)) => {
            for c in conds {
                if !conditions.contains(&c) {
                    conditions.push(c);
                }
            }
            simp(pool, e)
        }
        Err(_) => term,
    };
    let pieces: Vec<ExprId> = match pool.get(split) {
        ExprData::Add(xs) => xs.to_vec(),
        _ => vec![split],
    };
    let one = pool.integer(1_i32);
    let h1 = pool.add(vec![h, one]);
    let mut poly_part = Vec::new();
    let mut out = Vec::new();
    let mut found_pole = false;
    for piece in pieces {
        if poly_coeffs(piece, k, pool).is_some() {
            poly_part.push(piece);
            continue;
        }
        let rp = rational_piece(piece, k, pool)?;
        if !rp.quads.is_empty() {
            return None;
        }
        let pf = partial_fractions(&rp, k, pool, conditions)?;
        for (rho, coeffs) in pf {
            for (idx, &c) in coeffs.iter().enumerate() {
                let s = idx as u32 + 1;
                if s > MAX_ORDER {
                    return None;
                }
                if super::closed_form::is_zero(pool, simp(pool, c)) {
                    continue;
                }
                found_pole = true;
                let z_lo = simp(pool, pool.add(vec![lo, neg(pool, rho)]));
                let z_hi = simp(pool, pool.add(vec![h1, neg(pool, rho)]));
                // A pole inside the range: refuse (no `Z_s` at a pole).
                let lower = z_s(s, z_lo, pool)?;
                let upper = z_s(s, z_hi, pool)?;
                out.push(pool.mul(vec![c, pool.add(vec![lower, neg(pool, upper)])]));
            }
        }
    }
    if !found_pole {
        return None;
    }
    if !poly_part.is_empty() {
        let p = simp(pool, pool.add(poly_part));
        let coeffs = poly_coeffs(p, k, pool)?;
        out.push(polynomial_sum(&coeffs, lo, h, pool)?);
    }
    Some(simp(pool, pool.add(out)))
}

/// `Σ_{k=lo}^{h} P(k)` for `P = Σ p_j k^j`: the antidifference
/// `Q = Σ a_i k^i` with `Q(k+1) − Q(k) = P(k)` solved from the top degree
/// down (`Σ_{i>j} C(i, j)·a_i = p_j`), then `Q(h + 1) − Q(lo)`.
fn polynomial_sum(p: &[ExprId], lo: ExprId, h: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let d = p.len();
    if d == 0 {
        return Some(pool.integer(0_i32));
    }
    if d > 32 {
        return None;
    }
    let binom =
        |n: usize, r: usize| -> Integer { Integer::from(Integer::binomial_u(n as u32, r as u32)) };
    // a[0..=d]; a[0] = 0.
    let mut a = vec![pool.integer(0_i32); d + 1];
    for j in (0..d).rev() {
        let mut acc = vec![p[j]];
        for (i, &ai) in a.iter().enumerate().skip(j + 2) {
            acc.push(pool.mul(vec![pool.integer(-binom(i, j)), ai]));
        }
        a[j + 1] = simp(
            pool,
            pool.mul(vec![pool.add(acc), pool.rational(1, (j + 1) as i64)]),
        );
    }
    let q_at = |t: ExprId| {
        let terms: Vec<ExprId> = (1..=d)
            .map(|i| pool.mul(vec![a[i], pool.pow(t, pool.integer(i as i64))]))
            .collect();
        pool.add(terms)
    };
    let h1 = pool.add(vec![h, pool.integer(1_i32)]);
    Some(simp(pool, pool.add(vec![q_at(h1), neg(pool, q_at(lo))])))
}

/// Rational in `k`: built from `k`, constants free of `k`, `+`, `·` and
/// integer powers.
fn is_rational_in(e: ExprId, k: ExprId, pool: &ExprPool) -> bool {
    if is_free_of_k(e, k, pool) {
        return true;
    }
    match pool.get(e) {
        ExprData::Symbol { .. } => e == k,
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().all(|&x| is_rational_in(x, k, pool)),
        ExprData::Pow { base, exp } => {
            matches!(pool.get(exp), ExprData::Integer(_)) && is_rational_in(base, k, pool)
        }
        _ => false,
    }
}

enum Check {
    Agrees,
    Disagrees,
    Inconclusive,
}

/// Compare the closed form `value(h)` with the partial sums
/// `Σ_{k=lo}^{lo+j} term` for `j = 0..CHECK_POINTS`, every other free symbol
/// bound to a fixed irrational-ish value.
fn check_partial_sums(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    h: ExprId,
    value: ExprId,
    pool: &ExprPool,
) -> Check {
    use crate::jit::eval_interp;
    let mut params = Vec::new();
    crate::eval::symbols::collect_free_symbols(term, pool, &mut params);
    crate::eval::symbols::collect_free_symbols(value, pool, &mut params);
    crate::eval::symbols::collect_free_symbols(lo, pool, &mut params);
    params.retain(|&s| s != k && s != h);
    // `lo` has to be numeric to enumerate partial sums: bind its symbols to
    // small integers (and every other parameter to a fixed real).
    let lo_syms: Vec<ExprId> = {
        let mut v = Vec::new();
        crate::eval::symbols::collect_free_symbols(lo, pool, &mut v);
        v
    };
    let mut env: HashMap<ExprId, f64> = HashMap::new();
    for (i, &p) in params.iter().enumerate() {
        let v = if lo_syms.contains(&p) {
            2.0 + i as f64
        } else {
            0.413_27 + 0.731 * i as f64
        };
        env.insert(p, v);
    }
    let Some(lo_v) = eval_interp(lo, &env, pool) else {
        return Check::Inconclusive;
    };
    if (lo_v - lo_v.round()).abs() > 1e-12 {
        return Check::Inconclusive;
    }
    let lo_i = lo_v.round();
    let mut partial = 0.0_f64;
    let mut agreed = 0;
    for j in 0..CHECK_POINTS {
        let kv = lo_i + j as f64;
        let mut e2 = env.clone();
        e2.insert(k, kv);
        let Some(t) = eval_interp(term, &e2, pool) else {
            return Check::Inconclusive;
        };
        if !t.is_finite() {
            return Check::Inconclusive;
        }
        partial += t;
        let mut e3 = env.clone();
        e3.insert(h, kv);
        let Some(v) = eval_interp(value, &e3, pool) else {
            continue;
        };
        if !v.is_finite() {
            continue;
        }
        let scale = 1.0_f64.max(partial.abs()).max(v.abs());
        if (v - partial).abs() > 1e-9 * scale {
            return Check::Disagrees;
        }
        agreed += 1;
    }
    if agreed >= 3 {
        Check::Agrees
    } else {
        Check::Inconclusive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;

    fn setup() -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let n = pool.symbol("n", Domain::Real);
        (pool, k, n)
    }

    fn sum_value(term: ExprId, k: ExprId, lo: ExprId, hi: ExprId, pool: &ExprPool) -> ExprId {
        super::super::sum_definite(term, k, lo, hi, pool)
            .expect("sums")
            .value
    }

    #[test]
    fn harmonic_number() {
        let (pool, k, n) = setup();
        let term = pool.pow(k, pool.integer(-1_i32));
        let v = sum_value(term, k, pool.integer(1_i32), n, &pool);
        let s = pool.display(v).to_string();
        assert!(s.contains("digamma") && s.contains("EulerGamma"), "{s}");
        let got = eval_interp(v, &HashMap::from([(n, 9.0)]), &pool).unwrap();
        let want: f64 = (1..=9).map(|j| 1.0 / j as f64).sum();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn literal_bounds_give_an_exact_rational() {
        let (pool, k, _) = setup();
        let term = pool.pow(k, pool.integer(-1_i32));
        let v = sum_value(term, k, pool.integer(1_i32), pool.integer(4_i32), &pool);
        assert_eq!(v, pool.rational(25, 12));
    }

    #[test]
    fn second_order_pole_uses_trigamma_and_zeta_two() {
        let (pool, k, n) = setup();
        let term = pool.pow(k, pool.integer(-2_i32));
        let v = sum_value(term, k, pool.integer(1_i32), n, &pool);
        let got = eval_interp(v, &HashMap::from([(n, 7.0)]), &pool).unwrap();
        let want: f64 = (1..=7).map(|j| 1.0 / (j * j) as f64).sum();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn mixed_partial_fractions_and_polynomial_part() {
        // (k^3 + 1)/(k (2k + 1)): polynomial part plus two simple poles.
        let (pool, k, n) = setup();
        let num = pool.add(vec![pool.pow(k, pool.integer(3_i32)), pool.integer(1_i32)]);
        let den = pool.mul(vec![
            k,
            pool.add(vec![
                pool.mul(vec![pool.integer(2_i32), k]),
                pool.integer(1_i32),
            ]),
        ]);
        let term = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
        let v = sum_value(term, k, pool.integer(1_i32), n, &pool);
        let got = eval_interp(v, &HashMap::from([(n, 8.0)]), &pool).unwrap();
        let want: f64 = (1..=8)
            .map(|j| {
                let j = j as f64;
                (j * j * j + 1.0) / (j * (2.0 * j + 1.0))
            })
            .sum();
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }

    #[test]
    fn symbolic_shift() {
        // Σ_{k=0}^{n} 1/(k + a)
        let (pool, k, n) = setup();
        let a = pool.symbol("a", Domain::Real);
        let term = pool.pow(pool.add(vec![k, a]), pool.integer(-1_i32));
        let v = sum_value(term, k, pool.integer(0_i32), n, &pool);
        let env = HashMap::from([(n, 5.0), (a, 0.75)]);
        let got = eval_interp(v, &env, &pool).unwrap();
        let want: f64 = (0..=5).map(|j| 1.0 / (j as f64 + 0.75)).sum();
        assert!((got - want).abs() < 1e-10, "{got} vs {want}");
    }

    #[test]
    fn pole_inside_the_range_is_refused() {
        // Σ_{k=0}^{n} 1/k: the k = 0 term divides by zero.
        let (pool, k, n) = setup();
        let term = pool.pow(k, pool.integer(-1_i32));
        assert!(super::super::sum_definite(term, k, pool.integer(0_i32), n, &pool).is_err());
    }

    #[test]
    fn quadratic_denominator_is_not_claimed() {
        let (pool, k, n) = setup();
        let den = pool.add(vec![pool.pow(k, pool.integer(2_i32)), pool.integer(1_i32)]);
        let term = pool.pow(den, pool.integer(-1_i32));
        assert!(matches!(
            finite_rational(term, k, pool.integer(1_i32), n, &pool),
            Outcome::NotApplicable
        ));
    }

    #[test]
    fn z_s_values() {
        let pool = ExprPool::new();
        // Z_1(1) = γ
        let v = z_s(1, pool.integer(1_i32), &pool).unwrap();
        assert_eq!(simp(&pool, v), pool.symbol("EulerGamma", Domain::Real));
        assert!(z_s(1, pool.integer(0_i32), &pool).is_none());
    }
}
