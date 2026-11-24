//! Jacobi elliptic functions and the Weierstrass `℘` function.
//!
//! These are the *inverses* of the elliptic integrals the crate already has
//! (`EllipticF`, …), and they are what an autonomous second-order equation
//! `y'' = f(y)` with a cubic or quartic energy integral solves to: `y'' = 6y²`
//! is `℘`, the undamped Duffing equation `y'' = y − y³` is `cn`, the pendulum
//! `y'' + sin y = 0` is `2·am`.
//!
//! | primitive | arguments | derivative in the first argument |
//! |---|---|---|
//! | `JacobiSN(u, m)` | `u`, parameter `m = k²` | `cn·dn` |
//! | `JacobiCN(u, m)` | | `−sn·dn` |
//! | `JacobiDN(u, m)` | | `−m·sn·cn` |
//! | `JacobiAmplitude(u, m)` | | `dn` |
//! | `WeierstrassP(z, g₂, g₃)` | invariants `g₂`, `g₃` | `WeierstrassPPrime(z, g₂, g₃)` |
//! | `WeierstrassPPrime(z, g₂, g₃)` | | `6℘² − g₂/2` |
//!
//! # Conventions
//!
//! The parameter convention is `m = k²` throughout, as for `EllipticF(φ, m)`
//! and Mathematica's `JacobiSN[u, m]`; `am(u | m)` is the inverse of
//! `F(φ | m)`, so `sn = sin(am)` and `cn = cos(am)` (DLMF 22.16.11).  `℘` is
//! parametrised by its invariants, `℘'² = 4℘³ − g₂℘ − g₃` (DLMF 23.3.10), as
//! Mathematica's `WeierstrassP[z, {g2, g3}]` — not by the half-period ratio
//! `τ` that `experimental.weierstrass_p(z, tau)` takes.
//!
//! # Derivatives in the parameters
//!
//! `∂/∂m sn(u | m)` and `∂/∂g₂ ℘` exist but are not elementary in the
//! functions themselves (they need `E(am | m)`, resp. `ζ` and `σ`).  The rules
//! here differentiate in the first argument only and **decline** when a
//! parameter depends on the differentiation variable, rather than dropping a
//! term.
//!
//! # Numerics
//!
//! Real `f64` values only, which is what the ODE verification gate samples.
//!
//! * `sn`, `cn`, `dn` come from the descending Landen (AGM) transformation
//!   (A&S 16.4, DLMF 22.20), valid for every real `m`: `m > 1` goes through
//!   the reciprocal-parameter transformation `sn(u | m) = sn(√m·u | 1/m)/√m`
//!   (DLMF 22.17.2–4) and `m < 0` through the AGM's own handling of a
//!   complementary parameter above one.
//! * `am` is `atan2(sn, cn)` made continuous: for `m < 1` the quarter period
//!   `K(m)` is computed by the AGM, and `am(u + 2K) = am(u) + π`; for `m > 1`
//!   `cn(u | m) = dn(√m·u | 1/m) > 0`, so the principal `atan` is already
//!   continuous (and `am` is bounded: the motion is a libration).
//! * `℘` and `℘'` use the Laurent series at a point `z/2ⁿ` small against the
//!   lattice (`|w|²·max(|g₂|^{1/2}, |g₃|^{1/3}) < 0.1`) and then `n`
//!   applications of the duplication formula on the curve
//!   `Y² = 4X³ − g₂X − g₃` — real arithmetic for real data.  A lattice point
//!   is a pole and comes back non-finite (refused by `numeric_f64`).

use super::Primitive;
use crate::kernel::{ExprData, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// f64 kernels
// ---------------------------------------------------------------------------

/// `(sn, cn, dn)(u | m)` for real `u`, `m`.
pub(crate) fn sncndn(u: f64, m: f64) -> Option<(f64, f64, f64)> {
    if !u.is_finite() || !m.is_finite() {
        return None;
    }
    if m == 1.0 {
        let s = 1.0 / u.cosh();
        return Some((u.tanh(), s, s));
    }
    if m == 0.0 {
        return Some((u.sin(), u.cos(), 1.0));
    }
    if m > 1.0 {
        // DLMF 22.17.2–4: sn(u|m) = sn(ku|1/m)/k, cn(u|m) = dn(ku|1/m),
        // dn(u|m) = cn(ku|1/m), k = √m.
        let k = m.sqrt();
        let (s, c, d) = landen(k * u, 1.0 - 1.0 / m)?;
        return Some((s / k, d, c));
    }
    landen(u, 1.0 - m)
}

/// Descending Landen / AGM for complementary parameter `mc = 1 − m > 0`
/// (Numerical Recipes `sncndn`, which is A&S 16.4 in iterative form).
fn landen(u: f64, mc: f64) -> Option<(f64, f64, f64)> {
    const CA: f64 = 1e-10;
    let mut em = [0.0f64; 24];
    let mut en = [0.0f64; 24];
    let mut a = 1.0f64;
    let mut emc = mc;
    let mut c = 1.0f64;
    let mut l = 0usize;
    let mut dn = 1.0f64;
    for i in 0..24 {
        l = i;
        em[i] = a;
        emc = emc.sqrt();
        en[i] = emc;
        c = 0.5 * (a + emc);
        if (a - emc).abs() <= CA * a {
            break;
        }
        emc *= a;
        a = c;
    }
    let u = u * c;
    let mut sn = u.sin();
    let mut cn = u.cos();
    if sn != 0.0 {
        let mut a = cn / sn;
        c *= a;
        for ii in (0..=l).rev() {
            let b = em[ii];
            a *= c;
            c *= dn;
            dn = (en[ii] + a) / (b + a);
            a = c / b;
        }
        let a = 1.0 / (c * c + 1.0).sqrt();
        sn = if sn >= 0.0 { a } else { -a };
        cn = c * sn;
    }
    if sn.is_finite() && cn.is_finite() && dn.is_finite() {
        Some((sn, cn, dn))
    } else {
        None
    }
}

/// Complete elliptic integral `K(m)` for `m < 1`, by the AGM.
fn complete_k(m: f64) -> Option<f64> {
    if m >= 1.0 || !m.is_finite() {
        return None;
    }
    let (mut a, mut b) = (1.0f64, (1.0 - m).sqrt());
    for _ in 0..64 {
        if (a - b).abs() <= 1e-16 * a {
            break;
        }
        let an = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = an;
    }
    Some(std::f64::consts::PI / (2.0 * a))
}

/// `am(u | m)`, continuous in `u`.
pub(crate) fn amplitude(u: f64, m: f64) -> Option<f64> {
    if !u.is_finite() || !m.is_finite() {
        return None;
    }
    if m == 1.0 {
        return Some(u.sinh().atan());
    }
    if m > 1.0 {
        // cn(u | m) = dn(√m·u | 1/m) > 0: the principal atan is continuous.
        let (s, c, _) = sncndn(u, m)?;
        return Some(s.atan2(c));
    }
    let k = complete_k(m)?;
    let n = (u / (2.0 * k)).round();
    let r = u - 2.0 * k * n;
    let (s, c, _) = sncndn(r, m)?;
    Some(n * std::f64::consts::PI + s.atan2(c))
}

/// `(℘, ℘')(z; g₂, g₃)` for real data.
pub(crate) fn weierstrass(z: f64, g2: f64, g3: f64) -> Option<(f64, f64)> {
    if !z.is_finite() || !g2.is_finite() || !g3.is_finite() || z == 0.0 {
        return None;
    }
    let scale = g2.abs().sqrt().max(g3.abs().cbrt());
    let mut w = z;
    let mut n = 0u32;
    while w * w * scale > 0.1 {
        w *= 0.5;
        n += 1;
        if n > 200 {
            return None;
        }
    }
    // Laurent coefficients: ℘ = w⁻² + Σ_{k≥2} c_k w^{2k−2} (DLMF 23.9.2–3).
    const K: usize = 14;
    let mut c = [0.0f64; K + 1];
    c[2] = g2 / 20.0;
    c[3] = g3 / 28.0;
    for k in 4..=K {
        let mut s = 0.0;
        for j in 2..=k - 2 {
            s += c[j] * c[k - j];
        }
        c[k] = 3.0 * s / (((2 * k + 1) * (k - 3)) as f64);
    }
    let w2 = w * w;
    let mut p = 1.0 / w2;
    let mut dp = -2.0 / (w2 * w);
    let mut wpow = 1.0; // w^{2k−4}
    for (k, &ck) in c.iter().enumerate().skip(2) {
        // c_k w^{2k−2} and its derivative (2k−2) c_k w^{2k−3}
        p += ck * wpow * w2;
        dp += ck * (2 * k - 2) as f64 * wpow * w;
        wpow *= w2;
    }
    for _ in 0..n {
        let lam = (6.0 * p * p - 0.5 * g2) / dp;
        let p2 = 0.25 * lam * lam - 2.0 * p;
        let dp2 = -(dp + lam * (p2 - p));
        p = p2;
        dp = dp2;
    }
    Some((p, dp))
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

fn is_lit(pool: &ExprPool, e: ExprId, v: i64) -> bool {
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == v)
}

/// Shared body of the five forward rules: `local(args) · d(first)/d(wrt)`,
/// declining when a parameter depends on `wrt`.
fn chain(
    args: &[ExprId],
    arity: usize,
    wrt: ExprId,
    pool: &ExprPool,
    local: impl FnOnce() -> ExprId,
) -> Option<ExprId> {
    if args.len() != arity {
        return None;
    }
    if args[1..]
        .iter()
        .any(|&p| crate::kernel::subs::mentions_var(p, wrt, pool))
    {
        return None;
    }
    let du = crate::diff::diff(args[0], wrt, pool).ok()?.value;
    if is_lit(pool, du, 0) {
        return Some(pool.integer(0_i32));
    }
    Some(pool.mul(vec![local(), du]))
}

/// Reverse rule matching [`chain`]: the cotangent of the first argument only
/// exists when the parameters are constants of the computation, which reverse
/// mode cannot know — so a parameter cotangent of zero would be a lie.  Only
/// the all-literal-parameter case is answered.
fn chain_rev(
    args: &[ExprId],
    arity: usize,
    cotan: ExprId,
    pool: &ExprPool,
    local: impl FnOnce() -> ExprId,
) -> Option<Vec<ExprId>> {
    if args.len() != arity {
        return None;
    }
    if !args[1..].iter().all(|&p| {
        matches!(
            pool.get(p),
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
        )
    }) {
        return None;
    }
    let mut out = vec![pool.mul(vec![cotan, local()])];
    out.extend(args[1..].iter().map(|_| pool.integer(0_i32)));
    Some(out)
}

/// `sn(u | m)`, the Jacobi elliptic sine.
pub struct JacobiSnPrimitive;
/// `cn(u | m)`, the Jacobi elliptic cosine.
pub struct JacobiCnPrimitive;
/// `dn(u | m)`, the Jacobi delta amplitude.
pub struct JacobiDnPrimitive;
/// `am(u | m)`, the Jacobi amplitude: the inverse of `EllipticF(φ, m)`.
pub struct JacobiAmPrimitive;

/// `℘(z; g₂, g₃)`, the Weierstrass elliptic function with invariants `g₂, g₃`.
pub struct WeierstrassPPrimitive;
/// `℘'(z; g₂, g₃)`.
pub struct WeierstrassPPrimePrimitive;

fn jac(name: &str, args: &[ExprId], pool: &ExprPool) -> ExprId {
    pool.func(name, args.to_vec())
}

fn pretty_call(name: &str, args: &[ExprId], pool: &ExprPool) -> String {
    let a: Vec<String> = args.iter().map(|&a| pool.display(a).to_string()).collect();
    format!("{name}({})", a.join(", "))
}

/// `u = 0` and `m = 0` closed forms, shared by the four Jacobi primitives.
fn jacobi_simplify(name: &str, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    if args.len() != 2 {
        return None;
    }
    let (u, m) = (args[0], args[1]);
    if is_lit(pool, u, 0) {
        return Some(match name {
            "JacobiCN" | "JacobiDN" => pool.integer(1_i32),
            _ => pool.integer(0_i32),
        });
    }
    if is_lit(pool, m, 0) {
        return Some(match name {
            "JacobiSN" => pool.func("sin", vec![u]),
            "JacobiCN" => pool.func("cos", vec![u]),
            "JacobiDN" => pool.integer(1_i32),
            _ => u,
        });
    }
    None
}

impl Primitive for JacobiSnPrimitive {
    fn name(&self) -> &'static str {
        "JacobiSN"
    }
    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        pretty_call("JacobiSN", args, pool)
    }
    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        jacobi_simplify("JacobiSN", args, pool)
    }
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        chain(args, 2, wrt, pool, || {
            pool.mul(vec![
                jac("JacobiCN", args, pool),
                jac("JacobiDN", args, pool),
            ])
        })
    }
    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        chain_rev(args, 2, cotan, pool, || {
            pool.mul(vec![
                jac("JacobiCN", args, pool),
                jac("JacobiDN", args, pool),
            ])
        })
    }
    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        let [u, m] = args else { return None };
        sncndn(*u, *m).map(|t| t.0)
    }
}

impl Primitive for JacobiCnPrimitive {
    fn name(&self) -> &'static str {
        "JacobiCN"
    }
    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        pretty_call("JacobiCN", args, pool)
    }
    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        jacobi_simplify("JacobiCN", args, pool)
    }
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        chain(args, 2, wrt, pool, || {
            neg(
                pool,
                pool.mul(vec![
                    jac("JacobiSN", args, pool),
                    jac("JacobiDN", args, pool),
                ]),
            )
        })
    }
    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        chain_rev(args, 2, cotan, pool, || {
            neg(
                pool,
                pool.mul(vec![
                    jac("JacobiSN", args, pool),
                    jac("JacobiDN", args, pool),
                ]),
            )
        })
    }
    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        let [u, m] = args else { return None };
        sncndn(*u, *m).map(|t| t.1)
    }
}

impl Primitive for JacobiDnPrimitive {
    fn name(&self) -> &'static str {
        "JacobiDN"
    }
    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        pretty_call("JacobiDN", args, pool)
    }
    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        jacobi_simplify("JacobiDN", args, pool)
    }
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        chain(args, 2, wrt, pool, || {
            neg(
                pool,
                pool.mul(vec![
                    args[1],
                    jac("JacobiSN", args, pool),
                    jac("JacobiCN", args, pool),
                ]),
            )
        })
    }
    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        chain_rev(args, 2, cotan, pool, || {
            neg(
                pool,
                pool.mul(vec![
                    args[1],
                    jac("JacobiSN", args, pool),
                    jac("JacobiCN", args, pool),
                ]),
            )
        })
    }
    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        let [u, m] = args else { return None };
        sncndn(*u, *m).map(|t| t.2)
    }
}

impl Primitive for JacobiAmPrimitive {
    fn name(&self) -> &'static str {
        "JacobiAmplitude"
    }
    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        pretty_call("JacobiAmplitude", args, pool)
    }
    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        jacobi_simplify("JacobiAmplitude", args, pool)
    }
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        chain(args, 2, wrt, pool, || jac("JacobiDN", args, pool))
    }
    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        chain_rev(args, 2, cotan, pool, || jac("JacobiDN", args, pool))
    }
    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        let [u, m] = args else { return None };
        amplitude(*u, *m)
    }
}

/// `6℘² − g₂/2`, the derivative of `℘'`.
fn wp_second(args: &[ExprId], pool: &ExprPool) -> ExprId {
    let p = jac("WeierstrassP", args, pool);
    pool.add(vec![
        pool.mul(vec![pool.integer(6_i32), pool.pow(p, pool.integer(2_i32))]),
        pool.mul(vec![pool.rational(-1, 2), args[1]]),
    ])
}

impl Primitive for WeierstrassPPrimitive {
    fn name(&self) -> &'static str {
        "WeierstrassP"
    }
    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        pretty_call("WeierstrassP", args, pool)
    }
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        chain(args, 3, wrt, pool, || jac("WeierstrassPPrime", args, pool))
    }
    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        chain_rev(args, 3, cotan, pool, || {
            jac("WeierstrassPPrime", args, pool)
        })
    }
    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        let [z, g2, g3] = args else { return None };
        weierstrass(*z, *g2, *g3)
            .map(|t| t.0)
            .filter(|v| v.is_finite())
    }
}

impl Primitive for WeierstrassPPrimePrimitive {
    fn name(&self) -> &'static str {
        "WeierstrassPPrime"
    }
    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        pretty_call("WeierstrassPPrime", args, pool)
    }
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        chain(args, 3, wrt, pool, || wp_second(args, pool))
    }
    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        chain_rev(args, 3, cotan, pool, || wp_second(args, pool))
    }
    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        let [z, g2, g3] = args else { return None };
        weierstrass(*z, *g2, *g3)
            .map(|t| t.1)
            .filter(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values: mpmath 1.3 `ellipfun` / `ellipf`, 30 digits.
    #[test]
    fn jacobi_values_match_reference() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            // u, m, sn, cn, dn
            (
                0.5,
                0.3,
                0.474215622711821,
                0.880408736426462,
                0.965678964745951,
            ),
            (
                1.7,
                0.9,
                0.952230542108019,
                0.305380082318197,
                0.428872119878411,
            ),
            (
                2.3,
                5.7,
                -0.367064946411401,
                0.930195315574098,
                0.481664772597264,
            ),
            (
                0.8,
                -1.5,
                0.79262059231055,
                0.609715176656505,
                1.39368974489737,
            ),
        ];
        for &(u, m, s, c, d) in cases {
            let (gs, gc, gd) = sncndn(u, m).unwrap();
            assert!((gs - s).abs() < 1e-11, "sn({u}|{m}) = {gs}, want {s}");
            assert!((gc - c).abs() < 1e-11, "cn({u}|{m}) = {gc}, want {c}");
            assert!((gd - d).abs() < 1e-11, "dn({u}|{m}) = {gd}, want {d}");
        }
    }

    #[test]
    fn jacobi_identities_and_derivatives_hold_for_every_parameter_regime() {
        for &m in &[-3.0, -0.5, 0.0, 0.25, 0.5, 0.99, 1.0, 1.7, 5.7, 12.3] {
            for &u in &[-7.3, -1.1, 0.0, 0.37, 1.9, 4.4, 11.0] {
                let (s, c, d) = sncndn(u, m).unwrap();
                assert!((s * s + c * c - 1.0).abs() < 1e-12, "sn²+cn² at {u}|{m}");
                assert!(
                    (d * d + m * s * s - 1.0).abs() < 1e-11,
                    "dn²+m sn² at {u}|{m}"
                );
                // Central differences against the derivative rules.
                let h = 1e-5;
                let (sp, cp, dp) = sncndn(u + h, m).unwrap();
                let (sm, cm, dm) = sncndn(u - h, m).unwrap();
                let tol = 1e-7 * (1.0 + m.abs());
                assert!(
                    ((sp - sm) / (2.0 * h) - c * d).abs() < tol,
                    "sn' at {u}|{m}"
                );
                assert!(
                    ((cp - cm) / (2.0 * h) + s * d).abs() < tol,
                    "cn' at {u}|{m}"
                );
                assert!(
                    ((dp - dm) / (2.0 * h) + m * s * c).abs() < tol,
                    "dn' at {u}|{m}"
                );
                let a = amplitude(u, m).unwrap();
                assert!((a.sin() - s).abs() < 1e-11 && (a.cos() - c).abs() < 1e-11);
                let da = (amplitude(u + h, m).unwrap() - amplitude(u - h, m).unwrap()) / (2.0 * h);
                assert!((da - d).abs() < tol, "am' at {u}|{m}");
            }
        }
    }

    #[test]
    fn weierstrass_satisfies_its_differential_equation() {
        for &(g2, g3) in &[
            (0.0, 1.0),
            (0.0, -4.3),
            (2.0, 0.5),
            (-1.0, 3.0),
            (12.0, -7.0),
        ] {
            for &z in &[0.13, 0.4, 0.71, -0.55, 1.3] {
                let Some((p, dp)) = weierstrass(z, g2, g3) else {
                    continue;
                };
                if !p.is_finite() || p.abs() > 1e6 {
                    continue;
                }
                let rhs = 4.0 * p * p * p - g2 * p - g3;
                assert!(
                    (dp * dp - rhs).abs() < 1e-8 * (1.0 + rhs.abs()),
                    "℘'² vs 4℘³−g₂℘−g₃ at z={z}, g=({g2},{g3}): {} vs {rhs}",
                    dp * dp
                );
                let h = 1e-5;
                let fd = (weierstrass(z + h, g2, g3).unwrap().0
                    - weierstrass(z - h, g2, g3).unwrap().0)
                    / (2.0 * h);
                assert!((fd - dp).abs() < 1e-5 * (1.0 + dp.abs()), "℘' at z={z}");
            }
        }
        // Lemniscatic case g₂ = 4, g₃ = 0 has real half-period ω = 1.311028777…,
        // where ℘'(ω) = 0 and ℘(ω) = e₁ = 1.
        let omega = 1.311_028_777_146_06;
        let (p, dp) = weierstrass(omega, 4.0, 0.0).unwrap();
        assert!(
            (p - 1.0).abs() < 1e-9 && dp.abs() < 1e-7,
            "℘(ω) = {p}, ℘'(ω) = {dp}"
        );
    }

    #[test]
    fn derivative_rules_are_registered() {
        let pool = ExprPool::new();
        let u = pool.symbol("u", crate::kernel::Domain::Real);
        let m = pool.symbol("m", crate::kernel::Domain::Real);
        let sn = pool.func("JacobiSN", vec![u, m]);
        let d = crate::diff::diff(sn, u, &pool).unwrap().value;
        let s = pool.display(d).to_string();
        assert!(s.contains("JacobiCN") && s.contains("JacobiDN"), "{s}");
        // A parameter that depends on the variable is declined, not dropped.
        let sn_m = pool.func("JacobiSN", vec![u, u]);
        let r = crate::diff::diff(sn_m, u, &pool);
        if let Ok(v) = r {
            let s = pool.display(v.value).to_string();
            assert!(!s.contains("JacobiCN(u, u) * JacobiDN"), "{s}");
        }
        let g2 = pool.integer(0_i32);
        let g3 = pool.symbol("g", crate::kernel::Domain::Real);
        let wp = pool.func("WeierstrassP", vec![u, g2, g3]);
        let d2 = crate::diff::diff(crate::diff::diff(wp, u, &pool).unwrap().value, u, &pool)
            .unwrap()
            .value;
        let s = pool.display(d2).to_string();
        assert!(s.contains("WeierstrassP(") && !s.contains("Prime"), "{s}");
    }
}
