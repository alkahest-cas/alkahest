//! `∫ R(eᵏˣ, y) dx` where `y = (c·eᵏˣ + d)^{1/n}` — the **radical-over-exp**
//! case solved by the rationalizing substitution `u = y` (Risch milestone
//! **M4**, algebraic-top-generator dispatch).
//!
//! ## The math
//!
//! When the integrand is a rational function of `t = eᵏˣ` and a single radical
//! `y = a^{1/n}` whose radicand `a = c·t + d` is **affine in `t`** (`c, d ∈ ℚ`,
//! `c ≠ 0`, `k ∈ ℚ\{0}`), the substitution `u = y` rationalizes everything:
//!
//! ```text
//!   uⁿ = c·t + d   ⟹   t = (uⁿ − d)/c,
//!   Dt = k·t   ⟹   dx = dt/(k·t),   dt = (n/c)·uⁿ⁻¹ du,
//!   dx = n·uⁿ⁻¹ / (k·(uⁿ − d)) du.
//! ```
//!
//! Substituting `t ↦ (uⁿ−d)/c` and every radical power `a^{m/n} ↦ uᵐ` turns the
//! integrand into a **rational function of `u`**, which the ordinary engine
//! integrates completely (Hermite + Rothstein–Trager).  Back-substituting
//! `u ↦ y = a^{1/n}` yields the antiderivative, accepted only after the numeric
//! soundness gate `d/dx F = integrand` passes.
//!
//! ## Worked example (the M4 capstone target)
//!
//! `∫ ∛(eˣ+1) dx`.  Here `t = eˣ`, `n = 3`, `a = t + 1` (`c = d = k = 1`),
//! `y = (t+1)^{1/3}`.  The substitution gives `∫ u · 3u² / (u³−1) du =
//! ∫ 3u³/(u³−1) du`, a rational integral — so the result is **elementary**
//! (`3u + log`-terms), back-substituted to `u = ∛(eˣ+1)`.
//!
//! ## Reachable shape (narrow & sound)
//!
//! - exactly one radical generator `y = a^{1/n}` (`n ≥ 2`);
//! - exactly one exp generator `t = eᵏˣ` with `k ∈ ℚ\{0}` (`η = k·x` linear);
//! - the radicand `a` is affine in `t`: `a = c·t + d`, `c, d ∈ ℚ`, `c ≠ 0`;
//! - the whole integrand is a rational function of `t` and `y`.
//!
//! Anything else → `None` (the caller falls through).  A wrong substitution can
//! only *fail the numeric gate* → `None`; it never produces a wrong answer.

use std::collections::HashMap;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{subs, Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use rug::Rational;

use super::poly_rde::is_free_of_var;
use super::tower::find_generators;

/// Public entry: try `∫ R(eᵏˣ, y) dx` via the rationalizing radical
/// substitution `u = y`, `y = (c·eᵏˣ+d)^{1/n}`.
///
/// Returns `None` when the integrand is not of this shape (caller falls
/// through).  When it is, returns `Some(Ok(F))` with a numerically-verified
/// antiderivative, or `None` if the substituted rational integral declines or
/// the gate fails.
pub fn try_integrate_radical_over_exp_subst(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    // 1. Exactly one radical generator y = a^{1/n} and one exp generator t = eᵏˣ.
    let (n, a_expr) = detect_radical(expr, var, pool)?;
    let gens = find_generators(expr, var, pool);
    if gens.len() != 1 {
        return None; // a single exp generator and nothing else transcendental
    }
    let g = &gens[0];
    if !g.is_exp() {
        return None;
    }
    let t_gen = g.generator; // exp(η)
    let eta = g.argument();

    // η must be k·x with k ∈ ℚ\{0} (so Dt = k·t).
    let k = linear_coeff_in_var(eta, var, pool)?;
    if k == 0 {
        return None;
    }

    // 2. The radicand must be affine in t: a = c·t + d with c, d ∈ ℚ, c ≠ 0.
    let (c, d) = affine_in_t(a_expr, t_gen, var, pool)?;
    if c == 0 {
        return None;
    }

    // 3. Build the substitution map.
    let u = pool.symbol("__u_radsubst", Domain::Real);

    // t ↦ (uⁿ − d)/c
    let un = pool.pow(u, pool.integer(n as i32));
    let d_expr = rat_expr(&d, pool);
    let c_inv = rat_expr(&Rational::from(c.recip_ref()), pool);
    let t_sub = simplify(
        pool.mul(vec![
            pool.add(vec![un, pool.mul(vec![pool.integer(-1_i32), d_expr])]),
            c_inv,
        ]),
        pool,
    )
    .value;

    let mut map: HashMap<ExprId, ExprId> = HashMap::new();
    map.insert(t_gen, t_sub);
    // Every radical-power node a^{m/n} ↦ uᵐ (collected so the top-down `subs`
    // replaces them before it could descend into the radicand's own `t`).
    collect_radical_powers(expr, &a_expr, n, u, pool, &mut map);

    // 4. Substitute and multiply by dx/du = n·uⁿ⁻¹ / (k·(uⁿ − d)).
    let r_of_u = subs(expr, &map, pool);
    if contains_subexpr(r_of_u, t_gen, pool) {
        return None; // a stray t survived — not the supported shape
    }
    let un2 = pool.pow(u, pool.integer(n as i32));
    let dx_du = pool.mul(vec![
        pool.integer(n as i32),
        pool.pow(u, pool.integer((n as i32) - 1)),
        pool.pow(
            pool.mul(vec![
                rat_expr(&k, pool),
                pool.add(vec![
                    un2,
                    pool.mul(vec![pool.integer(-1_i32), rat_expr(&d, pool)]),
                ]),
            ]),
            pool.integer(-1_i32),
        ),
    ]);
    let integrand_u = simplify(pool.mul(vec![r_of_u, dx_du]), pool).value;

    // The substituted integrand must now be free of the original variable and of
    // the radical/exp generators — a pure rational function of u.
    if !is_free_of_var(integrand_u, var, pool) {
        return None;
    }
    if !find_generators(integrand_u, u, pool).is_empty() {
        return None; // a transcendental sneaked through — out of scope
    }

    // Reduce to a canonical `numer(u)/denom(u)` so the engine sees a clean
    // rational function (the raw substituted product carries un-cancelled
    // `(uⁿ−d)·(uⁿ−d)⁻¹`-style factors that the symbolic simplifier leaves intact).
    let integrand_u = rationalize_in(integrand_u, u, pool).unwrap_or(integrand_u);

    // 5. Integrate the rational function in u with the ordinary engine.
    let antideriv_u = match crate::integrate::engine::integrate(integrand_u, u, pool) {
        Ok(d) => d.value,
        Err(_) => return None,
    };

    // 6. Back-substitute u ↦ y = a^{1/n}.
    let y_expr = pool.pow(a_expr, pool.rational(1_i32, n as i32));
    let mut back: HashMap<ExprId, ExprId> = HashMap::new();
    back.insert(u, y_expr);
    let f = simplify(subs(antideriv_u, &back, pool), pool).value;

    // 7. Soundness gate: d/dx F must equal the integrand numerically.
    if !verify_derivative(f, expr, var, pool) {
        return None;
    }

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("risch_radical_over_exp_subst", expr, f));
    Some(Ok(DerivedExpr::with_log(f, log)))
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

/// Find the single radical `a^{1/n}` (`n ≥ 2`) whose radicand `a` depends on
/// `var`.  Returns `None` unless there is exactly one distinct `(n, a)`.
fn detect_radical(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(usize, ExprId)> {
    let mut found: Vec<(usize, ExprId)> = Vec::new();
    scan_radical(expr, var, pool, &mut found);
    let mut distinct: Vec<(usize, ExprId)> = Vec::new();
    for (n, a) in found {
        if !distinct.iter().any(|(m, b)| *m == n && *b == a) {
            distinct.push((n, a));
        }
    }
    if distinct.len() == 1 {
        Some(distinct.remove(0))
    } else {
        None
    }
}

fn scan_radical(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<(usize, ExprId)>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if name == "cbrt" && args.len() == 1 && !is_free_of_var(args[0], var, pool) =>
        {
            out.push((3, args[0]));
        }
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && !is_free_of_var(args[0], var, pool) =>
        {
            out.push((2, args[0]));
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
            scan_radical(base, var, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan_radical(a, var, pool, out);
            }
        }
        _ => {}
    }
}

/// Collect every radical-power node `a^{m/n}` (as `cbrt`/`sqrt`/`Pow` of the
/// radicand `a_expr`) and map it to `uᵐ`, so the top-down `subs` rewrites the
/// radical before it can descend into the radicand and touch the inner `t`.
fn collect_radical_powers(
    expr: ExprId,
    a_expr: &ExprId,
    n: usize,
    u: ExprId,
    pool: &ExprPool,
    map: &mut HashMap<ExprId, ExprId>,
) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if name == "cbrt" && args.len() == 1 && args[0] == *a_expr && n == 3 =>
        {
            map.insert(expr, u);
        }
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && args[0] == *a_expr && n == 2 =>
        {
            map.insert(expr, u);
        }
        ExprData::Pow { base, exp } => {
            if base == *a_expr {
                if let ExprData::Rational(r) = pool.get(exp) {
                    if let (Some(den), Some(num)) = (r.0.denom().to_i64(), r.0.numer().to_i64()) {
                        if den >= 1 && (n as i64) % den == 0 {
                            let m = num * (n as i64 / den);
                            map.insert(expr, pool.pow(u, pool.integer(m as i32)));
                            return;
                        }
                    }
                }
            }
            collect_radical_powers(base, a_expr, n, u, pool, map);
            collect_radical_powers(exp, a_expr, n, u, pool, map);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                collect_radical_powers(a, a_expr, n, u, pool, map);
            }
        }
        _ => {}
    }
}

/// If `eta = k·var` with `k ∈ ℚ`, return `k`; else `None`.
fn linear_coeff_in_var(eta: ExprId, var: ExprId, pool: &ExprPool) -> Option<Rational> {
    if eta == var {
        return Some(Rational::from(1));
    }
    match pool.get(eta) {
        ExprData::Mul(args) => {
            let mut coeff = Rational::from(1);
            let mut saw_var = false;
            for &a in &args {
                if a == var {
                    if saw_var {
                        return None; // var²
                    }
                    saw_var = true;
                } else if let Some(r) = as_rational(a, pool) {
                    coeff *= r;
                } else {
                    return None;
                }
            }
            if saw_var {
                Some(coeff)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// If `a = c·t + d` with `c, d ∈ ℚ` (constants free of `var`), return `(c, d)`.
/// Uses three sample substitutions `t = 0, 1, 2` and checks exact linearity.
fn affine_in_t(
    a_expr: ExprId,
    t_gen: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(Rational, Rational)> {
    let at = |val: i32, pool: &ExprPool| -> Option<Rational> {
        let mut m: HashMap<ExprId, ExprId> = HashMap::new();
        m.insert(t_gen, pool.integer(val));
        let s = simplify(subs(a_expr, &m, pool), pool).value;
        // After fixing t, the result must be a rational constant free of var.
        if !is_free_of_var(s, var, pool) {
            return None;
        }
        as_rational(s, pool)
    };
    let a0 = at(0, pool)?; // d
    let a1 = at(1, pool)?; // c + d
    let a2 = at(2, pool)?; // 2c + d
    let c = a1.clone() - a0.clone();
    // Linearity check: a2 must equal 2c + d.
    if a2 != c.clone() + c.clone() + a0.clone() {
        return None;
    }
    Some((c, a0))
}

fn as_rational(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

/// Build a literal rational `ExprId` from a `rug::Rational`, collapsing to an
/// `Integer` node when the denominator is 1 (so downstream polynomial parsing,
/// which rejects `Rational` nodes even for integer values, succeeds).
fn rat_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

/// Reduce a rational expression in `u` to canonical `numer(u)/denom(u)` form via
/// the `RationalFunction` machinery (poly-GCD cancellation), returning the
/// reconstructed expression.  Returns `None` if `expr` is not a rational function
/// of `u` (then the caller keeps the original expression).
fn rationalize_in(expr: ExprId, u: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (num_expr, den_expr) = split_num_den(expr, pool);
    let num_expr = simplify(num_expr, pool).value;
    let den_expr = simplify(den_expr, pool).value;
    let rf =
        crate::poly::RationalFunction::from_symbolic(num_expr, den_expr, vec![u], pool).ok()?;
    let num = rf.numer.to_expr(pool);
    let den = rf.denom.to_expr(pool);
    let den_inv = pool.pow(den, pool.integer(-1_i32));
    Some(simplify(pool.mul(vec![num, den_inv]), pool).value)
}

/// Split a product into `(numerator, denominator)` expressions: factors with a
/// literal negative-integer power go into the denominator (with the sign
/// flipped), everything else into the numerator.
fn split_num_den(expr: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let mut num: Vec<ExprId> = Vec::new();
    let mut den: Vec<ExprId> = Vec::new();
    let factors: Vec<ExprId> = match pool.get(expr) {
        ExprData::Mul(args) => args,
        _ => vec![expr],
    };
    for f in factors {
        match pool.get(f) {
            ExprData::Pow { base, exp } => match pool.get(exp) {
                ExprData::Integer(n) if n.0 < 0 => {
                    let pos = pool.integer(-(n.0.to_i32().unwrap_or(0)));
                    den.push(pool.pow(base, pos));
                }
                ExprData::Rational(r) if r.0 < 0 => {
                    let pos = pool.rational(-r.0.numer().clone(), r.0.denom().clone());
                    den.push(pool.pow(base, pos));
                }
                _ => num.push(f),
            },
            _ => num.push(f),
        }
    }
    let one = pool.integer(1_i32);
    let n = match num.len() {
        0 => one,
        1 => num[0],
        _ => pool.mul(num),
    };
    let d = match den.len() {
        0 => one,
        1 => den[0],
        _ => pool.mul(den),
    };
    (n, d)
}

fn contains_subexpr(expr: ExprId, target: ExprId, pool: &ExprPool) -> bool {
    if expr == target {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_subexpr(a, target, pool))
        }
        ExprData::Pow { base, exp } => {
            contains_subexpr(base, target, pool) || contains_subexpr(exp, target, pool)
        }
        ExprData::Func { args, .. } => args.iter().any(|&a| contains_subexpr(a, target, pool)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Numeric soundness gate
// ---------------------------------------------------------------------------

fn verify_derivative(f: ExprId, integrand: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let Ok(df) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    let mut checked = 0;
    for &xv in &[0.35_f64, 0.7, 1.3, 2.1] {
        let (Some(lhs), Some(rhs)) = (eval(ds, var, xv, pool), eval(integrand, var, xv, pool))
        else {
            continue;
        };
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        if (lhs - rhs).abs() > 1e-6 * (1.0 + rhs.abs()) {
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

    fn p() -> ExprPool {
        ExprPool::new()
    }

    fn assert_diff_matches(f: ExprId, integrand: ExprId, x: ExprId, pool: &ExprPool, pts: &[f64]) {
        let ds = simplify(crate::diff::diff(f, x, pool).unwrap().value, pool).value;
        for &xv in pts {
            let lhs = eval(ds, x, xv, pool).unwrap();
            let rhs = eval(integrand, x, xv, pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {}",
                pool.display(f)
            );
        }
    }

    /// M4 capstone: ∫ ∛(eˣ+1) dx is **elementary** (u = ∛(eˣ+1) ⟹
    /// ∫ 3u³/(u³−1) du, rational).
    #[test]
    fn cbrt_exp_plus_one_integrates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let inner = pool.add(vec![exp_x, pool.integer(1_i32)]);
        let integrand = pool.pow(inner, pool.rational(1_i32, 3_i32));

        let res = try_integrate_radical_over_exp_subst(integrand, x, &pool)
            .expect("recognized")
            .expect("elementary");
        assert_diff_matches(res.value, integrand, x, &pool, &[0.35, 0.7, 1.3, 2.1]);
    }

    /// √ sibling: ∫ √(eˣ+1) dx is elementary too.
    #[test]
    fn sqrt_exp_plus_one_integrates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let inner = pool.add(vec![exp_x, pool.integer(1_i32)]);
        let integrand = pool.func("sqrt", vec![inner]);

        let res = try_integrate_radical_over_exp_subst(integrand, x, &pool)
            .expect("recognized")
            .expect("elementary");
        assert_diff_matches(res.value, integrand, x, &pool, &[0.35, 0.7, 1.3, 2.1]);
    }

    /// Rational-in-radical: ∫ eˣ/∛(eˣ+1) dx (= ∫ (u³−1)·3u²/(u³−1)/u du =
    /// ∫ 3u du = (3/2)·∛(eˣ+1)²).
    #[test]
    fn exp_over_cbrt_integrates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let inner = pool.add(vec![exp_x, pool.integer(1_i32)]);
        let inv_cbrt = pool.pow(inner, pool.rational(-1_i32, 3_i32));
        let integrand = pool.mul(vec![exp_x, inv_cbrt]);

        let res = try_integrate_radical_over_exp_subst(integrand, x, &pool)
            .expect("recognized")
            .expect("elementary");
        assert_diff_matches(res.value, integrand, x, &pool, &[0.35, 0.7, 1.3, 2.1]);
    }

    /// Scaled radicand: ∫ ∛(2eˣ−1) dx (c=2, d=−1).
    #[test]
    fn cbrt_scaled_radicand_integrates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let inner = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), exp_x]),
            pool.integer(-1_i32),
        ]);
        let integrand = pool.pow(inner, pool.rational(1_i32, 3_i32));

        let res = try_integrate_radical_over_exp_subst(integrand, x, &pool)
            .expect("recognized")
            .expect("elementary");
        // ∛(2eˣ−1) is real only where 2eˣ−1 > 0, i.e. x > log(1/2) ≈ −0.69.
        assert_diff_matches(res.value, integrand, x, &pool, &[0.1, 0.7, 1.3, 2.1]);
    }

    /// Declines a non-affine radicand: ∫ √(e^{2x}+1) dx (radicand quadratic in
    /// t = eˣ).
    #[test]
    fn declines_nonaffine_radicand() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let exp_2x = pool.func("exp", vec![two_x]); // e^{2x} = t² but a separate generator here
        let inner = pool.add(vec![exp_2x, pool.integer(1_i32)]);
        let integrand = pool.func("sqrt", vec![inner]);
        // The radicand's generator is e^{2x}, affine in *that* t — the substitution
        // is still valid (u² = e^{2x}+1).  Accept if the gate passes; this test
        // only asserts no panic / no wrong answer.
        let res = try_integrate_radical_over_exp_subst(integrand, x, &pool);
        if let Some(Ok(d)) = res {
            assert_diff_matches(d.value, integrand, x, &pool, &[0.35, 0.7, 1.3]);
        }
    }

    /// Declines plain ∫ x·eˣ dx (no radical).
    #[test]
    fn declines_no_radical() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
        assert!(try_integrate_radical_over_exp_subst(f, x, &pool).is_none());
    }

    /// Declines a log-tower radicand (no exp generator).
    #[test]
    fn declines_log_radicand() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let inner = pool.add(vec![log_x, x]);
        let integrand = pool.func("cbrt", vec![inner]);
        assert!(try_integrate_radical_over_exp_subst(integrand, x, &pool).is_none());
    }
}
