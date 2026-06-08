//! First-kind elliptic-integral *output* for genus-1 radicands.
//!
//! When `∫ c·dx/√(P(x))` with `P` a **cubic or quartic** polynomial is
//! genus-1 and **non-elementary**, the antiderivative is an incomplete elliptic
//! integral of the first kind.  This module reduces that integrand to Legendre
//! normal form and emits a closed form
//!
//! ```text
//! F_cand(x) = g · EllipticF(φ(x), m)
//! ```
//!
//! with `φ(x) = arcsin(S(x))` or `arccos(S(x))` for an explicit real Möbius `S`,
//! a modulus parameter `m` (Mathematica convention `m = k²`, matching the PR1
//! `EllipticF` primitive), and a constant `g` — all algebraic in the roots of
//! `P`.  The reduction follows the standard case analysis of Byrd & Friedman,
//! *Handbook of Elliptic Integrals* (§3.13x cubics, §3.14x–§3.16x quartics).
//!
//! # Soundness
//!
//! The reduction constants are **not trusted blindly**.  Every candidate is run
//! through a numeric verification gate (`verify`): its *symbolic* `d/dx` (via
//! the engine's `diff`, which differentiates `EllipticF` through the primitive
//! registry — `∂φ F = 1/√(1 − m·sin²φ)`, elementary) is sampled against the
//! integrand `c/√P` at points where `P > 0`.  The form is emitted **only** if
//! the gate passes; otherwise the caller falls through to `NonElementary`.  An
//! imperfect reduction constant can therefore never produce a wrong answer — it
//! merely declines.

use crate::integrate::risch::poly_rde::{expr_to_qpoly, is_free_of_var};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// A complex root, stored as `(re, im)`.
type Croot = (f64, f64);

/// Try to emit a first-kind `EllipticF` closed form for `∫ (a + b·√P) dx` when
/// the integrand reduces to the pure first-kind shape `c/√P` (`a = 0`,
/// `b = c/P` with `c` a constant) and `P` is a gate-verifiable cubic/quartic.
///
/// Returns the antiderivative `g·EllipticF(φ(x), m)` (numeric `g`, `m`,
/// real-Möbius `φ`) when the verification gate passes, else `None` (caller
/// falls through to the existing `NonElementary` path).
pub fn try_elliptic_output(
    a_part: ExprId,
    b_part: ExprId,
    p_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    // Restrict to the *pure first kind*: `∫ c·dx/√P`.  This is `a = 0` and
    // `b·√P = c/√P`, i.e. `b = c/P` with `c` free of `var`.
    if !is_zero(a_part, pool) {
        return None;
    }
    let bp = pool.mul(vec![b_part, p_expr]);
    let c_expr = simplify(bp, pool).value;
    if !is_free_of_var(c_expr, var, pool) {
        return None;
    }
    let c = eval_const(c_expr, pool)?;
    if !c.is_finite() || c == 0.0 {
        return None;
    }

    // Parse P to rational coefficients (ascending) and get its degree.
    let p_poly = expr_to_qpoly(p_expr, var, pool)?;
    let coeffs: Vec<f64> = p_poly.iter().map(|r| r.to_f64()).collect();
    let deg = coeffs.len().checked_sub(1)?;
    if deg != 3 && deg != 4 {
        return None;
    }
    let lead = *coeffs.last()?;
    if lead == 0.0 {
        return None;
    }

    // Find all complex roots and classify into real / complex-conjugate pairs.
    let roots = poly_roots(&coeffs)?;
    let (mut reals, pairs) = classify_roots(&roots);
    reals.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending

    let inv_sqrt_lead = 1.0 / lead.abs().sqrt();

    // Build the candidate reduction (g, m, phi-expr) per case.
    let cand = match (deg, reals.len(), pairs.len()) {
        (3, 3, 0) => cubic_three_real(&reals, inv_sqrt_lead, var, pool),
        (3, 1, 1) => cubic_one_real(reals[0], pairs[0], inv_sqrt_lead, var, pool),
        (4, 4, 0) => quartic_four_real(&reals, inv_sqrt_lead, var, pool),
        (4, 2, 1) => quartic_two_real(&reals, pairs[0], inv_sqrt_lead, var, pool),
        _ => return None, // e.g. all-complex quartic: declined (falls through)
    }?;
    let (g, m, phi) = cand;

    // F_cand = (c · g) · EllipticF(phi, m).
    let m_expr = float_to_expr(m, pool);
    let f = pool.func("EllipticF", vec![phi, m_expr]);
    let coeff = float_to_expr(c * g, pool);
    let f_cand = simplify(pool.mul(vec![coeff, f]), pool).value;

    // Soundness gate: d/dx F_cand = c/√P numerically where P > 0.
    if verify(f_cand, &coeffs, c, var, pool) {
        Some(f_cand)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Reduction cases (Byrd & Friedman normal forms)
// ---------------------------------------------------------------------------

/// Cubic, three real roots `e1 > e2 > e3` (region `x ≥ e1`, where `P > 0` for a
/// positive leading coefficient): `sin²φ = (e1−e3)/(x−e3)`,
/// `m = (e2−e3)/(e1−e3)`, `g = −2/√(e1−e3)`.
fn cubic_three_real(
    reals: &[f64],
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (e1, e2, e3) = (reals[0], reals[1], reals[2]);
    let denom = e1 - e3;
    if denom <= 0.0 {
        return None;
    }
    let g = -2.0 / denom.sqrt() * inv_sqrt_lead;
    let m = (e2 - e3) / denom;
    // φ = arcsin( √( (e1−e3)/(x−e3) ) )
    let x_minus_e3 = pool.add(vec![var, float_to_expr(-e3, pool)]);
    let ratio = pool.mul(vec![
        float_to_expr(e1 - e3, pool),
        pool.pow(x_minus_e3, pool.integer(-1_i32)),
    ]);
    let s = pool.func("sqrt", vec![ratio]);
    let phi = pool.func("asin", vec![s]);
    Some((g, m, phi))
}

/// Cubic, one real root `y1` and a complex pair `b1 ± i·a1` (region `x ≥ y1`):
/// `A = √((y1−b1)² + a1²)`, `g = 1/√A`, `m = (A + (b1−y1))/(2A)`,
/// `cos φ = (A − (x−y1))/(A + (x−y1))`.
fn cubic_one_real(
    y1: f64,
    pair: Croot,
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (b1, a1) = pair;
    let aa = ((y1 - b1).powi(2) + a1 * a1).sqrt();
    if aa <= 0.0 {
        return None;
    }
    let g = inv_sqrt_lead / aa.sqrt();
    let m = (aa + (b1 - y1)) / (2.0 * aa);
    // cos φ = (A − (x − y1)) / (A + (x − y1)); φ = arccos(...)
    let x_minus_y1 = pool.add(vec![var, float_to_expr(-y1, pool)]);
    let num = pool.add(vec![
        float_to_expr(aa, pool),
        pool.mul(vec![pool.integer(-1_i32), x_minus_y1]),
    ]);
    let den = pool.add(vec![float_to_expr(aa, pool), x_minus_y1]);
    let cosphi = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
    let phi = pool.func("acos", vec![cosphi]);
    Some((g, m, phi))
}

/// Quartic, four real roots `a > b > c > d` (region `c ≤ x ≤ b`, where `P > 0`):
/// `sn²φ = (b−d)(x−c)/((b−c)(x−d))`, `m = (b−c)(a−d)/((a−c)(b−d))`,
/// `g = 2/√((a−c)(b−d))`.
fn quartic_four_real(
    reals: &[f64],
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (a, b, c, d) = (reals[0], reals[1], reals[2], reals[3]);
    let ac = a - c;
    let bd = b - d;
    let bc = b - c;
    if ac <= 0.0 || bd <= 0.0 || bc <= 0.0 {
        return None;
    }
    let g = 2.0 / (ac * bd).sqrt() * inv_sqrt_lead;
    let m = bc * (a - d) / (ac * bd);
    // sin²φ = (b−d)(x−c) / ((b−c)(x−d))
    let x_minus_c = pool.add(vec![var, float_to_expr(-c, pool)]);
    let x_minus_d = pool.add(vec![var, float_to_expr(-d, pool)]);
    let num = pool.mul(vec![float_to_expr(bd, pool), x_minus_c]);
    let den = pool.mul(vec![float_to_expr(bc, pool), x_minus_d]);
    let ratio = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
    let s = pool.func("sqrt", vec![ratio]);
    let phi = pool.func("asin", vec![s]);
    Some((g, m, phi))
}

/// Quartic, two real roots `b1 > b2` and a complex pair `b3 ± i·a3`
/// (region `b2 ≤ x ≤ b1`, where `P > 0`):
/// `A1 = √((b1−b3)² + a3²)`, `A2 = √((b2−b3)² + a3²)`, `g = 1/√(A1·A2)`,
/// `m = ((A1+A2)² − (b1−b2)²)/(4·A1·A2)`,
/// `cos φ = ((b1−x)A2 − (x−b2)A1)/((b1−x)A2 + (x−b2)A1)`.
fn quartic_two_real(
    reals: &[f64],
    pair: Croot,
    inv_sqrt_lead: f64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(f64, f64, ExprId)> {
    let (b1, b2) = (reals[0], reals[1]);
    let (b3, a3) = pair;
    let aa1 = ((b1 - b3).powi(2) + a3 * a3).sqrt();
    let aa2 = ((b2 - b3).powi(2) + a3 * a3).sqrt();
    if aa1 <= 0.0 || aa2 <= 0.0 {
        return None;
    }
    let g = inv_sqrt_lead / (aa1 * aa2).sqrt();
    let m = ((aa1 + aa2).powi(2) - (b1 - b2).powi(2)) / (4.0 * aa1 * aa2);
    // cos φ = ((b1−x)A2 − (x−b2)A1) / ((b1−x)A2 + (x−b2)A1)
    let b1_minus_x = pool.add(vec![
        float_to_expr(b1, pool),
        pool.mul(vec![pool.integer(-1_i32), var]),
    ]);
    let x_minus_b2 = pool.add(vec![var, float_to_expr(-b2, pool)]);
    let t1 = pool.mul(vec![b1_minus_x, float_to_expr(aa2, pool)]);
    let t2 = pool.mul(vec![x_minus_b2, float_to_expr(aa1, pool)]);
    let num = pool.add(vec![t1, pool.mul(vec![pool.integer(-1_i32), t2])]);
    let den = pool.add(vec![t1, t2]);
    let cosphi = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
    let phi = pool.func("acos", vec![cosphi]);
    Some((g, m, phi))
}

// ---------------------------------------------------------------------------
// Verification gate
// ---------------------------------------------------------------------------

/// Numerically verify `d/dx F_cand = c/√P` at sample points where `P > 0`.
fn verify(f_cand: ExprId, coeffs: &[f64], c: f64, var: ExprId, pool: &ExprPool) -> bool {
    let Ok(df) = crate::diff::diff(f_cand, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    // Sample widely; keep only points where P > 0 and both sides are finite reals.
    let samples = [
        -3.5, -2.7, -1.6, -0.9, -0.4, 0.15, 0.3, 0.55, 0.7, 0.9, 1.1, 1.4, 1.9, 2.6, 3.4, 4.7,
    ];
    let mut checked = 0;
    for &xv in &samples {
        let pv = eval_poly(coeffs, xv);
        if pv <= 1e-6 {
            continue;
        }
        let rhs = c / pv.sqrt();
        let Some(lhs) = eval(ds, var, xv, pool) else {
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

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    super::poly_utils::is_zero_expr(expr, pool)
}

/// Evaluate a constant (var-free) expression to `f64`.
fn eval_const(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => Some(r.0.to_f64()),
        ExprData::Add(args) => args
            .iter()
            .try_fold(0.0, |s, &a| Some(s + eval_const(a, pool)?)),
        ExprData::Mul(args) => args
            .iter()
            .try_fold(1.0, |s, &a| Some(s * eval_const(a, pool)?)),
        ExprData::Pow { base, exp } => Some(eval_const(base, pool)?.powf(eval_const(exp, pool)?)),
        _ => None,
    }
}

/// Horner evaluation of a polynomial given ascending coefficients.
fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Numeric eval supporting the elementary functions produced by `diff` of the
/// candidate (`sin`, `cos`, `asin`, `acos`, `sqrt`; the `EllipticF` derivative
/// is rewritten to the elementary `1/√(1−m·sin²φ)` so no special-function eval
/// is needed).
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
                "sin" => Some(v.sin()),
                "cos" => Some(v.cos()),
                "tan" => Some(v.tan()),
                "asin" => Some(v.asin()),
                "acos" => Some(v.acos()),
                "atan" => Some(v.atan()),
                "sqrt" => Some(v.sqrt()),
                "exp" => Some(v.exp()),
                "log" => Some(v.ln()),
                "cbrt" => Some(v.cbrt()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Build an `ExprId` for an `f64` value as an exact rational reconstruction of
/// the float.  Keeping the constant rational (rather than a float literal the
/// engine has no node for) lets `simplify`/`diff` operate on it cleanly.
fn float_to_expr(v: f64, pool: &ExprPool) -> ExprId {
    // Exact small integers stay integer nodes.
    if v.fract() == 0.0 && v.abs() <= i32::MAX as f64 {
        return pool.integer(v as i32);
    }
    match rug::Rational::from_f64(v) {
        Some(r) => {
            let (num, den) = r.into_numer_denom();
            pool.rational(num, den)
        }
        None => pool.integer(0_i32),
    }
}

// ---------------------------------------------------------------------------
// Complex root finding (Durand–Kerner) + classification
// ---------------------------------------------------------------------------

/// Find all complex roots of a polynomial with ascending real coefficients
/// (degree 3 or 4) via Durand–Kerner iteration.
fn poly_roots(coeffs: &[f64]) -> Option<Vec<Croot>> {
    let n = coeffs.len() - 1;
    if n == 0 {
        return Some(vec![]);
    }
    let lead = *coeffs.last()?;
    // Monic normalized coefficients, ascending.
    let mono: Vec<f64> = coeffs.iter().map(|&c| c / lead).collect();

    // Initial guesses: powers of the classic Durand–Kerner seed 0.4 + 0.9i.
    let seed = (0.4_f64, 0.9_f64);
    let mut z: Vec<Croot> = (0..n).map(|k| cpow(seed, k as i32)).collect();

    for _ in 0..500 {
        let mut max_delta = 0.0_f64;
        for i in 0..n {
            let num = ceval(&mono, z[i]);
            let mut den = (1.0, 0.0);
            for j in 0..n {
                if i != j {
                    den = cmul(den, csub(z[i], z[j]));
                }
            }
            let delta = cdiv(num, den);
            z[i] = csub(z[i], delta);
            let d = (delta.0 * delta.0 + delta.1 * delta.1).sqrt();
            if d > max_delta {
                max_delta = d;
            }
        }
        if max_delta < 1e-14 {
            break;
        }
    }
    Some(z)
}

/// Classify roots into sorted real roots and complex-conjugate pairs
/// `(re, |im|)` (one entry per conjugate pair).
fn classify_roots(roots: &[Croot]) -> (Vec<f64>, Vec<Croot>) {
    let tol = 1e-7;
    let mut reals = Vec::new();
    let mut pairs = Vec::new();
    let mut used = vec![false; roots.len()];
    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        if roots[i].1.abs() < tol {
            reals.push(roots[i].0);
            used[i] = true;
        } else {
            // Find the conjugate partner.
            let mut best = None;
            let mut best_d = f64::INFINITY;
            for (j, used_j) in used.iter().enumerate().skip(i + 1) {
                if *used_j {
                    continue;
                }
                let d = (roots[j].0 - roots[i].0).abs() + (roots[j].1 + roots[i].1).abs();
                if d < best_d {
                    best_d = d;
                    best = Some(j);
                }
            }
            if let Some(j) = best {
                if best_d < 1e-5 {
                    pairs.push((roots[i].0, roots[i].1.abs()));
                    used[i] = true;
                    used[j] = true;
                }
            }
        }
    }
    (reals, pairs)
}

// Minimal complex arithmetic on `(re, im)` tuples.
fn cmul(a: Croot, b: Croot) -> Croot {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn csub(a: Croot, b: Croot) -> Croot {
    (a.0 - b.0, a.1 - b.1)
}
fn cdiv(a: Croot, b: Croot) -> Croot {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
fn cpow(a: Croot, n: i32) -> Croot {
    let mut r = (1.0, 0.0);
    for _ in 0..n {
        r = cmul(r, a);
    }
    r
}
/// Horner evaluation of a monic polynomial (ascending coeffs) at a complex point.
fn ceval(mono: &[f64], z: Croot) -> Croot {
    let mut acc = (0.0, 0.0);
    for &c in mono.iter().rev() {
        acc = cmul(acc, z);
        acc.0 += c;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// Assert `∫ c·dx/√P` emits an `EllipticF` form whose `d/dx` matches the
    /// integrand at sample points; return the form's display string.
    fn check_emits(p_expr: ExprId, var: ExprId, c: f64, pool: &ExprPool) -> Option<String> {
        let zero = pool.integer(0_i32);
        // b = c / P  ⇒ integrand = b·√P = c/√P.
        let c_e = float_to_expr(c, pool);
        let b = pool.mul(vec![c_e, pool.pow(p_expr, pool.integer(-1_i32))]);
        let f = try_elliptic_output(zero, b, p_expr, var, pool)?;
        let s = pool.display(f).to_string();
        assert!(s.contains("EllipticF"), "no EllipticF in {s}");
        Some(s)
    }

    #[test]
    fn cubic_x3_plus_1_emits_ellipticf() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let s = check_emits(p, x, 1.0, &pool).expect("∫dx/√(x³+1) should emit EllipticF");
        assert!(s.contains("EllipticF"), "{s}");
    }

    #[test]
    fn cubic_three_real_emits_ellipticf() {
        // x³ − x = x(x−1)(x+1): three real roots.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(x³−x) should emit EllipticF");
    }

    #[test]
    fn quartic_four_real_emits_ellipticf() {
        // (x²−1)(x²−4) = x⁴ − 5x² + 4: four real roots ±1, ±2.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(4_i32)),
            pool.mul(vec![pool.integer(-5_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.integer(4_i32),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√((x²−1)(x²−4)) should emit EllipticF");
    }

    #[test]
    fn quartic_two_real_pair_emits_ellipticf() {
        // 1 − x⁴ = (1−x²)(1+x²): two real roots ±1, complex pair ±i.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(4_i32))]),
        ]);
        check_emits(p, x, 1.0, &pool).expect("∫dx/√(1−x⁴) should emit EllipticF");
    }

    #[test]
    fn quintic_declined() {
        // x⁵+1 is genus-2: no degree-3/4 reduction ⇒ None (caller → NonElementary).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let zero = pool.integer(0_i32);
        let b = pool.pow(p, pool.integer(-1_i32));
        assert!(try_elliptic_output(zero, b, p, x, &pool).is_none());
    }
}
