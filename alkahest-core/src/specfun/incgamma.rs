//! The Euler Beta function and the incomplete gamma functions (F11).
//!
//! | node | DLMF | definition |
//! |---|---|---|
//! | `beta(a, b)` | 5.12.1 | `B(a, b) = Γ(a)Γ(b)/Γ(a+b) = ∫₀¹ t^{a−1}(1−t)^{b−1} dt` |
//! | `uppergamma(s, z)` | 8.2.2 | `Γ(s, z) = ∫_z^∞ t^{s−1}e^{−t} dt` |
//! | `lowergamma(s, z)` | 8.2.1 | `γ(s, z) = ∫₀^z t^{s−1}e^{−t} dt` |
//!
//! The spellings and argument orders are SymPy's (`beta`, `uppergamma`,
//! `lowergamma`); the values are mpmath's `beta`, `gammainc(s, z)` (upper)
//! and `gammainc(s, 0, z)` (lower), and Arb's `arb_hypgeom_gamma_upper` /
//! `_lower`. `Γ(s, z) + γ(s, z) = Γ(s)`.
//!
//! # Exact forms (`simplify`)
//!
//! * `B(a, b)` at literal arguments whose `Γ` values are exact (positive
//!   integers and half-integers) is the `Γ` ratio; with one argument a positive
//!   integer `n ≤` [`MAX_INT_REDUCTION`] it is the rational function
//!   `(n−1)!/(a(a+1)⋯(a+n−1))` in the other.
//! * `Γ(s, z)` and `γ(s, z)` for a positive integer `s = n` are elementary:
//!   `Γ(n, z) = (n−1)! e^{−z} Σ_{k<n} z^k/k!` (DLMF 8.4.8); for `s = ½` they are
//!   `√π erfc(√z)` and `√π erf(√z)` (8.4.6), and for `s = m + ½` the recurrence
//!   `Γ(s+1, z) = sΓ(s, z) + z^s e^{−z}` (8.8.2) reduces them to that.
//! * `Γ(s, 0) = Γ(s)` and `γ(s, 0) = 0` for a literal `s > 0`;
//!   `Γ(s, ∞) = 0`, `γ(s, ∞) = Γ(s)`.
//!
//! # Derivatives
//!
//! `∂_z Γ(s, z) = −z^{s−1}e^{−z}`, `∂_z γ(s, z) = z^{s−1}e^{−z}`;
//! `∂_a B(a, b) = B(a, b)(ψ(a) − ψ(a+b))` and symmetrically in `b`. A
//! derivative of an incomplete gamma in `s` has no closed form here (it is a
//! Meijer-G function) and declines with `E-DIFF-001`.
//!
//! # Refusals
//!
//! `E-SPECFN-006` for a literal pole: `B(a, b)` with `a` or `b` a
//! non-positive integer, `Γ(s, 0)` for a literal `s ≤ 0`.

use rug::{Integer, Rational};

use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use crate::orthopoly::{int_literal, mk_add, mk_mul, rat_node, rational_literal};
use crate::primitive::Primitive;

use super::SpecialFunctionError;

/// Largest positive-integer argument for which `B(a, n)` is rewritten as a
/// rational function of `a`, and the largest `n` (or `m` in `m + ½`) for
/// which an incomplete gamma is expanded into elementary functions.
pub(crate) const MAX_INT_REDUCTION: i64 = 32;

fn exp_of(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("exp", vec![e])
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    mk_mul(pool, vec![pool.integer(-1), e])
}

fn sqrt_pi(pool: &ExprPool) -> ExprId {
    pool.func("sqrt", vec![crate::eval::symbols::pi_symbol(pool)])
}

fn is_literal_zero(e: ExprId, pool: &ExprPool) -> bool {
    rational_literal(pool, e).is_some_and(|q| q == 0)
}

fn nonpositive_int(e: ExprId, pool: &ExprPool) -> bool {
    rational_literal(pool, e).is_some_and(|q| *q.denom() == 1 && q <= 0)
}

/// `s` as `(m, half)`: `s = m` (`half = false`) or `s = m + ½`, with
/// `0 ≤ m ≤` [`MAX_INT_REDUCTION`] (and `m ≥ 1` for the integer case).
fn small_order(s: ExprId, pool: &ExprPool) -> Option<(i64, bool)> {
    let q = rational_literal(pool, s)?;
    if *q.denom() == 1 {
        let n = q.numer().to_i64()?;
        return (1..=MAX_INT_REDUCTION).contains(&n).then_some((n, false));
    }
    if *q.denom() == 2 {
        let m = (q - Rational::from((1, 2))).numer().to_i64()?;
        return (0..=MAX_INT_REDUCTION).contains(&m).then_some((m, true));
    }
    None
}

fn factorial(n: u32) -> Integer {
    Integer::from(Integer::factorial(n))
}

/// `e^{−z} Σ_{k<n} z^k/k!`.
fn exp_partial_sum(n: i64, z: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::with_capacity(n as usize);
    for k in 0..n {
        let c = Rational::from((Integer::from(1), factorial(k as u32)));
        let zk = match k {
            0 => pool.integer(1),
            1 => z,
            _ => pool.pow(z, pool.integer(k)),
        };
        terms.push(mk_mul(pool, vec![rat_node(pool, &c), zk]));
    }
    mk_mul(pool, vec![exp_of(neg(z, pool), pool), mk_add(pool, terms)])
}

/// The elementary form of `Γ(s, z)` (`upper = true`) or `γ(s, z)` for an
/// order `s` accepted by [`small_order`].
fn incomplete_elementary(upper: bool, s: ExprId, z: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (m, half) = small_order(s, pool)?;
    if !half {
        // Γ(n, z) = (n−1)! e^{−z} Σ_{k<n} z^k/k!;  γ(n, z) = (n−1)! − Γ(n, z).
        let f = pool.integer(factorial((m - 1) as u32));
        let up = mk_mul(pool, vec![f, exp_partial_sum(m, z, pool)]);
        return Some(if upper {
            up
        } else {
            mk_add(pool, vec![f, neg(up, pool)])
        });
    }
    // s = ½ first, then Γ(s+1, z) = sΓ(s, z) + z^s e^{−z} and
    // γ(s+1, z) = sγ(s, z) − z^s e^{−z}.
    let root = pool.func("sqrt", vec![z]);
    let base = pool.func(if upper { "erfc" } else { "erf" }, vec![root]);
    let mut acc = mk_mul(pool, vec![sqrt_pi(pool), base]);
    let ez = exp_of(neg(z, pool), pool);
    for j in 0..m {
        // From order sj = j + ½ to sj + 1.
        let sj = Rational::from((2 * j + 1, 2));
        let zs = pool.pow(z, rat_node(pool, &sj));
        let tail = mk_mul(pool, vec![zs, ez]);
        let scaled = mk_mul(pool, vec![rat_node(pool, &sj), acc]);
        acc = if upper {
            mk_add(pool, vec![scaled, tail])
        } else {
            mk_add(pool, vec![scaled, neg(tail, pool)])
        };
    }
    Some(acc)
}

/// The exact value of `Γ(s, z)` for the cases in the module docs.
pub(crate) fn uppergamma_value(s: ExprId, z: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if z == pool.pos_infinity() {
        return Some(pool.integer(0));
    }
    if is_literal_zero(z, pool) {
        let q = rational_literal(pool, s)?;
        return (q > 0).then(|| {
            super::gamma::gamma_value(s, pool).unwrap_or_else(|| pool.func("gamma", vec![s]))
        });
    }
    incomplete_elementary(true, s, z, pool)
}

/// The exact value of `γ(s, z)` for the cases in the module docs.
pub(crate) fn lowergamma_value(s: ExprId, z: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if is_literal_zero(z, pool) {
        let q = rational_literal(pool, s)?;
        return (q > 0).then(|| pool.integer(0));
    }
    if z == pool.pos_infinity() {
        let q = rational_literal(pool, s)?;
        return (q > 0).then(|| {
            super::gamma::gamma_value(s, pool).unwrap_or_else(|| pool.func("gamma", vec![s]))
        });
    }
    incomplete_elementary(false, s, z, pool)
}

/// `(n−1)!/(a(a+1)⋯(a+n−1))` when `n` is a small positive integer literal.
fn beta_integer_partner(a: ExprId, n: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let k = int_literal(pool, n)?.to_i64()?;
    if !(1..=MAX_INT_REDUCTION).contains(&k) {
        return None;
    }
    let mut f = vec![pool.integer(factorial((k - 1) as u32))];
    for j in 0..k {
        let shifted = if j == 0 {
            a
        } else {
            pool.add(vec![a, pool.integer(j)])
        };
        f.push(pool.pow(shifted, pool.integer(-1)));
    }
    Some(mk_mul(pool, f))
}

/// The exact value of `B(a, b)`, if any (module docs).
pub(crate) fn beta_value(a: ExprId, b: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let (Some(qa), Some(qb)) = (rational_literal(pool, a), rational_literal(pool, b)) {
        let ab = rat_node(pool, &(qa + qb));
        if let (Some(ga), Some(gb), Some(gab)) = (
            super::gamma::gamma_value(a, pool),
            super::gamma::gamma_value(b, pool),
            super::gamma::gamma_value(ab, pool),
        ) {
            return Some(mk_mul(pool, vec![ga, gb, pool.pow(gab, pool.integer(-1))]));
        }
    }
    beta_integer_partner(a, b, pool).or_else(|| beta_integer_partner(b, a, pool))
}

fn pole(function: &'static str, at: String) -> SpecialFunctionError {
    SpecialFunctionError::Pole { function, at }
}

/// The Euler Beta function `B(a, b) = Γ(a)Γ(b)/Γ(a+b)` (DLMF 5.12.1).
///
/// Exact values where the `Γ` ratio is exact, and the rational function
/// `(n−1)!/(a(a+1)⋯(a+n−1))` when one argument is a positive integer `n`;
/// the node `beta(a, b)` otherwise.
///
/// # Errors
///
/// `E-SPECFN-006` when `a` or `b` is a non-positive integer literal.
pub fn beta(a: ExprId, b: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    for (v, name) in [(a, "a"), (b, "b")] {
        if nonpositive_int(v, pool) {
            return Err(pole("beta", format!("{name} = {}", pool.display(v))));
        }
    }
    crate::budget::check()?;
    Ok(beta_value(a, b, pool).unwrap_or_else(|| pool.func("beta", vec![a, b])))
}

/// The upper incomplete gamma function `Γ(s, z) = ∫_z^∞ t^{s−1}e^{−t} dt`
/// (DLMF 8.2.2), principal branch.
///
/// Elementary for a positive integer `s` and in `erfc` for a half-integer
/// `s > 0`; `Γ(s, 0) = Γ(s)` for a literal `s > 0`.
///
/// # Errors
///
/// `E-SPECFN-006` for `Γ(s, 0)` with a literal `s ≤ 0`.
pub fn uppergamma(s: ExprId, z: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    if is_literal_zero(z, pool) && rational_literal(pool, s).is_some_and(|q| q <= 0) {
        return Err(pole(
            "uppergamma",
            format!("s = {}, z = 0", pool.display(s)),
        ));
    }
    crate::budget::check()?;
    Ok(uppergamma_value(s, z, pool).unwrap_or_else(|| pool.func("uppergamma", vec![s, z])))
}

/// The lower incomplete gamma function `γ(s, z) = ∫₀^z t^{s−1}e^{−t} dt`
/// (DLMF 8.2.1), principal branch.
///
/// Elementary for a positive integer `s` and in `erf` for a half-integer
/// `s > 0`; `γ(s, 0) = 0` for a literal `s > 0`.
///
/// # Errors
///
/// `E-SPECFN-006` for a literal `s` that is a non-positive integer (`γ` has
/// poles there as a function of `s`).
pub fn lowergamma(s: ExprId, z: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    if nonpositive_int(s, pool) {
        return Err(pole("lowergamma", format!("s = {}", pool.display(s))));
    }
    crate::budget::check()?;
    Ok(lowergamma_value(s, z, pool).unwrap_or_else(|| pool.func("lowergamma", vec![s, z])))
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// `z^{s−1} e^{−z}`, the integrand of both incomplete gamma functions.
fn kernel_at(s: ExprId, z: ExprId, pool: &ExprPool) -> ExprId {
    let sm1 = pool.add(vec![s, pool.integer(-1)]);
    mk_mul(pool, vec![pool.pow(z, sm1), exp_of(neg(z, pool), pool)])
}

fn incomplete_diff(upper: bool, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let [s, z] = args else { return None };
    if crate::kernel::subs::mentions_var(*s, wrt, pool) {
        return None;
    }
    let dz = crate::diff::diff(*z, wrt, pool).ok()?.value;
    let k = kernel_at(*s, *z, pool);
    let k = if upper { neg(k, pool) } else { k };
    Some(mk_mul(pool, vec![k, dz]))
}

/// Registry entry for `uppergamma(s, z)`.
pub struct UpperGammaPrimitive;

impl Primitive for UpperGammaPrimitive {
    fn name(&self) -> &'static str {
        "uppergamma"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        match args {
            [s, z] => format!("Γ({}, {})", pool.display(*s), pool.display(*z)),
            _ => "uppergamma(?)".into(),
        }
    }

    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        let [s, z] = args else { return None };
        uppergamma_value(*s, *z, pool)
    }

    /// `∂_z Γ(s, z) = −z^{s−1}e^{−z}`; `∂_s` declines.
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        incomplete_diff(true, args, wrt, pool)
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        match args {
            [s, z] => incgamma_f64(true, *s, *z),
            _ => None,
        }
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        match args {
            [s, z] => incgamma_ball(true, s, z),
            _ => None,
        }
    }
}

/// Registry entry for `lowergamma(s, z)`.
pub struct LowerGammaPrimitive;

impl Primitive for LowerGammaPrimitive {
    fn name(&self) -> &'static str {
        "lowergamma"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        match args {
            [s, z] => format!("γ({}, {})", pool.display(*s), pool.display(*z)),
            _ => "lowergamma(?)".into(),
        }
    }

    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        let [s, z] = args else { return None };
        lowergamma_value(*s, *z, pool)
    }

    /// `∂_z γ(s, z) = z^{s−1}e^{−z}`; `∂_s` declines.
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        incomplete_diff(false, args, wrt, pool)
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        match args {
            [s, z] => incgamma_f64(false, *s, *z),
            _ => None,
        }
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        match args {
            [s, z] => incgamma_ball(false, s, z),
            _ => None,
        }
    }
}

/// Registry entry for `beta(a, b)`.
pub struct BetaPrimitive;

impl Primitive for BetaPrimitive {
    fn name(&self) -> &'static str {
        "beta"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        match args {
            [a, b] => format!("B({}, {})", pool.display(*a), pool.display(*b)),
            _ => "beta(?)".into(),
        }
    }

    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        let [a, b] = args else { return None };
        beta_value(*a, *b, pool)
    }

    /// `d B(a, b) = B(a, b)·[(ψ(a) − ψ(a+b))·da + (ψ(b) − ψ(a+b))·db]`.
    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        let [a, b] = args else { return None };
        let da = crate::diff::diff(*a, wrt, pool).ok()?.value;
        let db = crate::diff::diff(*b, wrt, pool).ok()?.value;
        let psi_ab = pool.func("digamma", vec![pool.add(vec![*a, *b])]);
        let part = |v: ExprId, dv: ExprId| {
            let d = mk_add(pool, vec![pool.func("digamma", vec![v]), neg(psi_ab, pool)]);
            mk_mul(pool, vec![d, dv])
        };
        let node = pool.func("beta", vec![*a, *b]);
        Some(mk_mul(
            pool,
            vec![node, mk_add(pool, vec![part(*a, da), part(*b, db)])],
        ))
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        match args {
            [a, b] => beta_f64(*a, *b),
            _ => None,
        }
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        match args {
            [a, b] => beta_ball(a, b),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Numerics (Arb)
// ---------------------------------------------------------------------------

/// `Γ(s, z)` / `γ(s, z)` in `f64`, certified to 53 bits by Arb; `None` where
/// the value is not real or not finite, or without Arb.
pub(crate) fn incgamma_f64(upper: bool, s: f64, z: f64) -> Option<f64> {
    if !s.is_finite() || !z.is_finite() {
        return None;
    }
    #[cfg(flint_arb)]
    {
        arb_inc::incgamma_f64(upper, s, z)
    }
    #[cfg(not(flint_arb))]
    {
        let _ = upper;
        None
    }
}

pub(crate) fn incgamma_ball(upper: bool, s: &ArbBall, z: &ArbBall) -> Option<ArbBall> {
    #[cfg(flint_arb)]
    {
        arb_inc::incgamma_ball(upper, s, z)
    }
    #[cfg(not(flint_arb))]
    {
        let _ = (upper, s, z);
        None
    }
}

pub(crate) fn beta_f64(a: f64, b: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    #[cfg(flint_arb)]
    {
        arb_inc::beta_f64(a, b)
    }
    #[cfg(not(flint_arb))]
    {
        None
    }
}

pub(crate) fn beta_ball(a: &ArbBall, b: &ArbBall) -> Option<ArbBall> {
    #[cfg(flint_arb)]
    {
        arb_inc::beta_ball(a, b)
    }
    #[cfg(not(flint_arb))]
    {
        let _ = (a, b);
        None
    }
}

#[cfg(flint_arb)]
mod arb_inc {
    use super::super::numeric::arb_impl::{available, certified_f64, from_arb, to_arb};
    use crate::ball::ArbBall;
    use crate::flint::acb::Arb;
    use crate::flint::arb as ffi;

    fn incgamma(upper: bool, s: &Arb, z: &Arb, prec: i64) -> Option<Arb> {
        let mut out = Arb::new();
        unsafe {
            if upper {
                ffi::arb_hypgeom_gamma_upper(out.as_mut_ptr(), s.as_ptr(), z.as_ptr(), 0, prec);
            } else {
                ffi::arb_hypgeom_gamma_lower(out.as_mut_ptr(), s.as_ptr(), z.as_ptr(), 0, prec);
            }
        }
        out.is_finite().then_some(out)
    }

    fn beta(a: &Arb, b: &Arb, prec: i64) -> Option<Arb> {
        let mut ga = Arb::new();
        let mut gb = Arb::new();
        let mut rab = Arb::new();
        let ab = a.add(b, prec);
        unsafe {
            ffi::arb_gamma(ga.as_mut_ptr(), a.as_ptr(), prec);
            ffi::arb_gamma(gb.as_mut_ptr(), b.as_ptr(), prec);
            ffi::arb_rgamma(rab.as_mut_ptr(), ab.as_ptr(), prec);
        }
        let v = ga.mul(&gb, prec).mul(&rab, prec);
        v.is_finite().then_some(v)
    }

    fn f64_retry(f: impl Fn(i64) -> Option<Arb>) -> Option<f64> {
        if !available() {
            return None;
        }
        for prec in [128_i64, 256, 512] {
            if let Some(v) = f(prec).as_ref().and_then(certified_f64) {
                return Some(v);
            }
        }
        None
    }

    pub(super) fn incgamma_f64(upper: bool, s: f64, z: f64) -> Option<f64> {
        let (sb, zb) = (Arb::from_f64(s), Arb::from_f64(z));
        f64_retry(|prec| incgamma(upper, &sb, &zb, prec))
    }

    pub(super) fn beta_f64(a: f64, b: f64) -> Option<f64> {
        let (ab, bb) = (Arb::from_f64(a), Arb::from_f64(b));
        f64_retry(|prec| beta(&ab, &bb, prec))
    }

    pub(super) fn incgamma_ball(upper: bool, s: &ArbBall, z: &ArbBall) -> Option<ArbBall> {
        if !available() {
            return None;
        }
        let prec = s.prec.max(z.prec);
        let v = incgamma(upper, &to_arb(s)?, &to_arb(z)?, i64::from(prec) + 16)?;
        from_arb(&v, prec)
    }

    pub(super) fn beta_ball(a: &ArbBall, b: &ArbBall) -> Option<ArbBall> {
        if !available() {
            return None;
        }
        let prec = a.prec.max(b.prec);
        let v = beta(&to_arb(a)?, &to_arb(b)?, i64::from(prec) + 16)?;
        from_arb(&v, prec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplify::simplify;

    fn ev(e: ExprId, pool: &ExprPool) -> f64 {
        crate::jit::eval_interp(e, &std::collections::HashMap::new(), pool).unwrap()
    }

    #[test]
    fn beta_exact_values() {
        let p = ExprPool::new();
        // B(2, 3) = 1!2!/4! = 1/12
        let b = beta(p.integer(2), p.integer(3), &p).unwrap();
        assert_eq!(simplify(b, &p).value, p.rational(1, 12));
        // B(1/2, 1/2) = π
        let b = beta(p.rational(1, 2), p.rational(1, 2), &p).unwrap();
        assert!((ev(b, &p) - std::f64::consts::PI).abs() < 1e-14);
        // B(a, 1) = 1/a
        let a = p.symbol("a", crate::kernel::Domain::Real);
        let b = beta(a, p.integer(1), &p).unwrap();
        assert_eq!(b, p.pow(a, p.integer(-1)));
        assert!(beta(p.integer(0), a, &p).is_err());
    }

    #[test]
    fn incomplete_gamma_special_values() {
        let p = ExprPool::new();
        let z = p.symbol("z", crate::kernel::Domain::Real);
        let g = uppergamma(p.integer(1), z, &p).unwrap();
        assert_eq!(g, p.func("exp", vec![neg(z, &p)]));
        let g0 = uppergamma(p.rational(5, 2), p.integer(0), &p).unwrap();
        assert!((ev(g0, &p) - 1.329_340_388_179_137).abs() < 1e-14);
        assert!(uppergamma(p.integer(0), p.integer(0), &p).is_err());
        assert_eq!(
            lowergamma(p.rational(1, 3), p.integer(0), &p).unwrap(),
            p.integer(0)
        );
        let half = uppergamma(p.rational(1, 2), z, &p).unwrap();
        assert!(p.display(half).to_string().contains("erfc"));
    }

    #[cfg(flint_arb)]
    #[test]
    fn incomplete_gamma_numeric_against_mpmath() {
        // mpmath.gammainc(2.5, 1.5) and gammainc(2.5, 0, 1.5).
        let up = incgamma_f64(true, 2.5, 1.5).unwrap();
        let lo = incgamma_f64(false, 2.5, 1.5).unwrap();
        assert!((up - 0.930_519_442_786_792).abs() < 1e-14, "{up}");
        assert!((up + lo - 1.329_340_388_179_137).abs() < 1e-14);
        // Elementary forms agree with Arb at half-integer and integer orders.
        let p = ExprPool::new();
        for (s, zv) in [
            (p.rational(7, 2), 0.8),
            (p.integer(4), 2.3),
            (p.rational(1, 2), 1.1),
        ] {
            let sv = ev(s, &p);
            let z = p.float(zv, 53);
            let u = uppergamma_value(s, z, &p).unwrap();
            let l = lowergamma_value(s, z, &p).unwrap();
            assert!((ev(u, &p) - incgamma_f64(true, sv, zv).unwrap()).abs() < 1e-12);
            assert!((ev(l, &p) - incgamma_f64(false, sv, zv).unwrap()).abs() < 1e-12);
        }
        // mpmath.beta(0.3, 2.7)
        assert!((beta_f64(0.3, 2.7).unwrap() - 2.310_517_136_083_31).abs() < 1e-12);
    }

    #[test]
    fn derivatives() {
        let p = ExprPool::new();
        let z = p.symbol("z", crate::kernel::Domain::Real);
        let s = p.symbol("s", crate::kernel::Domain::Real);
        let g = p.func("uppergamma", vec![s, z]);
        let d = crate::diff::diff(g, z, &p).unwrap().value;
        assert!(!p.display(d).to_string().contains("uppergamma"));
        assert!(crate::diff::diff(g, s, &p).is_err());
    }
}
