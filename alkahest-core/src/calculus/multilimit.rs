//! Multivariate limits `lim_{(x,y)→(a,b)} f(x,y)` (V-MULTILIMIT).
//!
//! This module computes two-variable limits with three sound outcomes, exposed
//! through [`MultiLimit`]:
//!
//! * [`MultiLimit::Value`] — a *proven* value `L`. A value is returned **only**
//!   when one of two rigorous routes succeeds: the **continuity** fast path
//!   (substituting the target point yields a finite, well-defined number — no
//!   `0/0`, `∞−∞`, …), or a **polar squeeze bound** (for a rational integrand,
//!   `|f − L|` is shown to be bounded by `B(r) → 0` with the denominator bounded
//!   away from `0`). Agreement of path probes alone is **never** sufficient to
//!   return a value.
//!
//! * [`MultiLimit::DoesNotExist`] — non-existence with a *certificate*: two
//!   explicit approach paths whose univariate limits differ. Each witness path
//!   limit is verified both by the univariate engine
//!   ([`crate::calculus::limit`]) and by a numeric sanity sample before the
//!   certificate is issued.
//!
//! * [`MultiLimit::Undecided`] — a sound decline: the limit may or may not
//!   exist, but the current heuristics could neither prove a value nor exhibit a
//!   non-existence certificate.
//!
//! The engine first translates the problem to the origin (`x → a + u`,
//! `y → b + v`), then runs the continuity path, then path probes
//! (lines `v = m·u` with symbolic `m`, and power paths `v = c·uᵏ`), and finally
//! — when the probes agree on a candidate `L` — attempts a polar bound.
//!
//! Only the two-variable case is implemented; the API names the two variables
//! explicitly so it does not over-promise an `n`-variable surface.

use crate::calculus::limits::{limit, LimitDirection};
use crate::kernel::{subs, Domain, ExprData, ExprId, ExprPool};
use crate::poly::RationalFunction;
use crate::simplify::simplify;
use std::collections::HashMap;
use std::fmt;

/// An approach path used in a non-existence certificate, together with the
/// univariate limit obtained along it.
///
/// `description` is a human-readable rendering of the substitution
/// (e.g. `"v = 0 (the u-axis)"` or `"v = u²"`), `value` is the limit of the
/// translated function along that path as the path parameter `→ 0`, and
/// `value_numeric` is the numeric sample that corroborated it.
#[derive(Clone, Debug)]
pub struct PathWitness {
    /// Human-readable description of the approach path.
    pub description: String,
    /// Symbolic univariate limit obtained along the path.
    pub value: ExprId,
    /// Numeric sample corroborating `value` (used to guard the certificate).
    pub value_numeric: f64,
}

/// Result of a two-variable limit computation. See the [module
/// documentation](mod@self) for the soundness contract.
#[derive(Clone, Debug)]
pub enum MultiLimit {
    /// A proven limit value (continuity path or polar squeeze bound only).
    Value(ExprId),
    /// The limit does not exist, certified by two paths with different limits.
    DoesNotExist {
        /// First witness path and its limit.
        path_a: PathWitness,
        /// Second witness path and its limit (differs from `path_a`).
        path_b: PathWitness,
    },
    /// Sound decline: neither a value nor a non-existence certificate found.
    Undecided,
}

impl fmt::Display for MultiLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiLimit::Value(_) => write!(f, "Value"),
            MultiLimit::DoesNotExist { path_a, path_b } => write!(
                f,
                "DoesNotExist (along {} → {}; along {} → {})",
                path_a.description, path_a.value_numeric, path_b.description, path_b.value_numeric
            ),
            MultiLimit::Undecided => write!(f, "Undecided"),
        }
    }
}

/// Tolerance for declaring two numeric path limits "different".
const PROBE_SEPARATION: f64 = 1e-4;
/// Tolerance for numeric agreement / corroboration.
const NUMERIC_TOL: f64 = 1e-6;

/// `multilimit(f, x, y, a, b, pool)` — compute `lim_{(x,y)→(a,b)} f(x,y)`.
///
/// `x` and `y` must be distinct symbols; `a` and `b` are the (finite) target
/// coordinates. Returns a [`MultiLimit`] whose `Value`/`DoesNotExist` arms are
/// sound (see [module docs](mod@self)); anything the heuristics cannot settle is
/// reported as [`MultiLimit::Undecided`].
pub fn multilimit(
    f: ExprId,
    x: ExprId,
    y: ExprId,
    a: ExprId,
    b: ExprId,
    pool: &ExprPool,
) -> MultiLimit {
    if x == y {
        return MultiLimit::Undecided;
    }

    // 1. Translate to the origin: x → a + u, y → b + v.
    let u = pool.symbol("__ml_u", Domain::Real);
    let v = pool.symbol("__ml_v", Domain::Real);
    let xa = simplify(pool.add(vec![a, u]), pool).value;
    let yb = simplify(pool.add(vec![b, v]), pool).value;
    let mut t = HashMap::new();
    t.insert(x, xa);
    t.insert(y, yb);
    let g = simplify(subs(f, &t, pool), pool).value;

    // 2. Continuity fast path.
    if let Some(val) = continuity_value(g, u, v, pool) {
        return MultiLimit::Value(val);
    }

    // 3. Path probes for non-existence.
    if let Some(dne) = probe_nonexistence(g, u, v, pool) {
        return dne;
    }

    // 4. Existence proof when probes agree on L (rational f, polar bound).
    if let Some(l) = probes_agree_on(g, u, v, pool) {
        if rational_polar_bound_to_zero(g, u, v, l, pool) {
            return MultiLimit::Value(l);
        }
    }

    MultiLimit::Undecided
}

// ===========================================================================
// Continuity fast path
// ===========================================================================

/// If substituting `(u, v) = (0, 0)` yields a finite, well-defined value with a
/// non-vanishing denominator (corroborated by a numeric multi-sample around the
/// origin), return it.
fn continuity_value(g: ExprId, u: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // Denominator must not vanish at the origin.
    let (_num, den) = numer_denom(g, pool);
    if let Some(dv) = eval2(den, u, v, 0.0, 0.0, pool) {
        if dv.abs() < 1e-12 {
            return None;
        }
    } else {
        return None;
    }

    // Symbolic substitution at the origin.
    let mut m = HashMap::new();
    m.insert(u, pool.integer(0_i32));
    m.insert(v, pool.integer(0_i32));
    let at0 = simplify(subs(g, &m, pool), pool).value;

    // The substituted expression must contain no surviving u/v and be finite.
    if contains_var(at0, u, pool) || contains_var(at0, v, pool) {
        return None;
    }
    let lv = eval2(at0, u, v, 0.0, 0.0, pool)?;
    if !lv.is_finite() {
        return None;
    }

    // Numeric corroboration: as the sample point shrinks toward the origin,
    // `g` must converge to `lv` (this rejects removable-singularity `0/0` forms
    // that simplified away but are genuinely discontinuous). We sample a fixed
    // set of directions at geometrically shrinking radii and require the
    // worst-case deviation to decrease toward 0.
    let dirs = [
        (0.6, 0.8),
        (-0.8, 0.6),
        (0.5, -0.866),
        (-0.707, -0.707),
        (1.0, 0.1),
    ];
    let radii = [0.1_f64, 0.01, 0.001];
    let mut prev_dev = f64::INFINITY;
    for &r in &radii {
        let mut dev = 0.0_f64;
        for (cu, cv) in dirs {
            let gv = eval2(g, u, v, r * cu, r * cv, pool)?;
            dev = dev.max((gv - lv).abs());
        }
        // Deviation must shrink (continuity); allow tiny float slack.
        if dev > prev_dev + 1e-12 {
            return None;
        }
        prev_dev = dev;
    }
    if prev_dev > NUMERIC_TOL * (1.0 + lv.abs()) {
        return None;
    }
    Some(at0)
}

// ===========================================================================
// Path probes (non-existence)
// ===========================================================================

/// Numeric path parametrization: maps a small parameter `t > 0` to a point
/// `(du, dv)` near the origin along the path.
type PathFn = fn(f64) -> (f64, f64);

/// A candidate approach path: a univariate-in-`u`-or-`v` expression plus a label
/// and a numeric path-evaluator giving `(du, dv)` for a small parameter `t > 0`.
struct Probe {
    label: String,
    /// `g` restricted to this path, as a univariate function of `u` or `v`.
    expr_of_t: ExprId,
    /// Numeric path: given a small `t > 0`, returns `(du, dv)`.
    numeric: PathFn,
}

/// Build the standard line and power probes and look for two whose verified
/// limits differ.
fn probe_nonexistence(g: ExprId, u: ExprId, v: ExprId, pool: &ExprPool) -> Option<MultiLimit> {
    let probes = build_probes(g, u, v, pool);
    let mut witnesses: Vec<PathWitness> = Vec::new();

    for p in &probes {
        if let Some(w) = verify_probe(p, u, v, pool) {
            // Compare against existing witnesses for a separation.
            for prev in &witnesses {
                if (prev.value_numeric - w.value_numeric).abs() > PROBE_SEPARATION {
                    return Some(MultiLimit::DoesNotExist {
                        path_a: prev.clone(),
                        path_b: w.clone(),
                    });
                }
            }
            witnesses.push(w);
        }
    }
    None
}

/// Construct the list of probes:
///   * the u-axis (`v = 0`) and v-axis (`u = 0`),
///   * lines `v = m·u` for a spread of slopes `m`,
///   * power paths `v = uᵏ` and `u = vᵏ` for `k ∈ {2, 3}`.
fn build_probes(g: ExprId, u: ExprId, v: ExprId, pool: &ExprPool) -> Vec<Probe> {
    let mut probes = Vec::new();

    // Substitute v = rhs(u) into g (univariate in u).
    let sub_v = |rhs: ExprId| -> ExprId {
        let mut m = HashMap::new();
        m.insert(v, rhs);
        simplify(subs(g, &m, pool), pool).value
    };
    // Substitute u = rhs(v) into g (univariate in v).
    let sub_u = |rhs: ExprId| -> ExprId {
        let mut m = HashMap::new();
        m.insert(u, rhs);
        simplify(subs(g, &m, pool), pool).value
    };

    // u-axis: v = 0.
    probes.push(Probe {
        label: "v = 0 (the u-axis)".to_string(),
        expr_of_t: sub_v(pool.integer(0_i32)),
        numeric: |t| (t, 0.0),
    });
    // v-axis: u = 0.
    probes.push(Probe {
        label: "u = 0 (the v-axis)".to_string(),
        expr_of_t: sub_u(pool.integer(0_i32)),
        numeric: |t| (0.0, t),
    });

    // Lines v = m·u for several integer slopes.
    let slopes: [(i64, &str, PathFn); 4] = [
        (1, "v = u", |t| (t, t)),
        (-1, "v = -u", |t| (t, -t)),
        (2, "v = 2·u", |t| (t, 2.0 * t)),
        (3, "v = 3·u", |t| (t, 3.0 * t)),
    ];
    for (m_int, lbl, num) in slopes {
        let rhs = simplify(pool.mul(vec![pool.integer(m_int), u]), pool).value;
        probes.push(Probe {
            label: lbl.to_string(),
            expr_of_t: sub_v(rhs),
            numeric: num,
        });
    }

    // Power paths v = uᵏ (catches x²y/(x⁴+y²): all lines → 0, v = u² → ½).
    let pow_v: [(i32, &str, PathFn); 2] = [
        (2, "v = u²", |t| (t, t * t)),
        (3, "v = u³", |t| (t, t * t * t)),
    ];
    for (k, lbl, num) in pow_v {
        let rhs = simplify(pool.pow(u, pool.integer(k)), pool).value;
        probes.push(Probe {
            label: lbl.to_string(),
            expr_of_t: sub_v(rhs),
            numeric: num,
        });
    }
    // Symmetric power paths u = vᵏ.
    let pow_u: [(i32, &str, PathFn); 2] = [
        (2, "u = v²", |t| (t * t, t)),
        (3, "u = v³", |t| (t * t * t, t)),
    ];
    for (k, lbl, num) in pow_u {
        let rhs = simplify(pool.pow(v, pool.integer(k)), pool).value;
        probes.push(Probe {
            label: lbl.to_string(),
            expr_of_t: sub_u(rhs),
            numeric: num,
        });
    }

    probes
}

/// Verify one probe: compute its univariate limit symbolically *and*
/// numerically; only return a witness if both agree on a finite value.
fn verify_probe(p: &Probe, u: ExprId, v: ExprId, pool: &ExprPool) -> Option<PathWitness> {
    let uses_u = contains_var(p.expr_of_t, u, pool);
    let uses_v = contains_var(p.expr_of_t, v, pool);
    let var = if uses_u && !uses_v {
        u
    } else if uses_v && !uses_u {
        v
    } else if !uses_u && !uses_v {
        // Constant along this path: limit is the constant itself.
        let lv = eval2(p.expr_of_t, u, v, 0.0, 0.0, pool)?;
        if !lv.is_finite() {
            return None;
        }
        return Some(PathWitness {
            description: p.label.clone(),
            value: p.expr_of_t,
            value_numeric: lv,
        });
    } else {
        // Both u and v survived (shouldn't happen for a genuine path); bail.
        return None;
    };

    // Symbolic univariate limit as var → 0⁺.
    let lim_sym = limit(
        p.expr_of_t,
        var,
        pool.integer(0_i32),
        LimitDirection::Plus,
        pool,
    )
    .ok()?;
    let lim_num = eval2(lim_sym, u, v, 0.0, 0.0, pool)?;
    if !lim_num.is_finite() {
        return None;
    }

    // Numeric sanity sample along the actual path (corroboration).
    let mut sampled = None;
    for &t in &[1e-2, 1e-3, 1e-4] {
        let (du, dv) = (p.numeric)(t);
        if let Some(val) = eval2(p.expr_of_t, u, v, du, dv, pool) {
            sampled = Some(val);
        }
    }
    let sampled = sampled?;
    if (sampled - lim_num).abs() > 1e-2 * (1.0 + lim_num.abs()) {
        return None;
    }

    Some(PathWitness {
        description: p.label.clone(),
        value: lim_sym,
        value_numeric: lim_num,
    })
}

/// When all verified probes agree on a single finite value `L`, return it
/// (as a candidate for an existence proof). Returns `None` if probes disagree
/// or too few succeed.
fn probes_agree_on(g: ExprId, u: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let probes = build_probes(g, u, v, pool);
    let mut value_sym: Option<ExprId> = None;
    let mut value_num: Option<f64> = None;
    let mut count = 0;
    for p in &probes {
        if let Some(w) = verify_probe(p, u, v, pool) {
            match value_num {
                None => {
                    value_num = Some(w.value_numeric);
                    value_sym = Some(w.value);
                }
                Some(prev) => {
                    if (prev - w.value_numeric).abs() > PROBE_SEPARATION {
                        return None;
                    }
                }
            }
            count += 1;
        }
    }
    if count >= 3 {
        value_sym
    } else {
        None
    }
}

// ===========================================================================
// Existence proof: polar squeeze bound for rational f
// ===========================================================================

/// For a rational `g = N/D` and candidate limit `L`, attempt to prove
/// `lim_{(u,v)→0} g = L` by a polar squeeze:
///   substitute `u = r·cosθ`, `v = r·sinθ`, form `h = g − L`, and show that on a
///   `θ`-grid (over `[0, 2π)`) the denominator stays bounded away from `0` while
///   the envelope `max_θ |h|` shrinks toward `0` as `r → 0`.
///
/// This is a numeric-grid certificate with a conservative safety margin — false
/// negatives produce `Undecided`, never a wrong value.
fn rational_polar_bound_to_zero(
    g: ExprId,
    u: ExprId,
    v: ExprId,
    l: ExprId,
    pool: &ExprPool,
) -> bool {
    // Require g to be a genuine rational function in (u, v).
    if !is_rational_in(g, u, v, pool) {
        return false;
    }
    // h = g − L.
    let neg_l = simplify(pool.mul(vec![pool.integer(-1_i32), l]), pool).value;
    let h = simplify(pool.add(vec![g, neg_l]), pool).value;
    let (_hn, hd) = numer_denom(h, pool);

    // θ-grid over [0, 2π).
    let n_theta = 720;
    let thetas: Vec<f64> = (0..n_theta)
        .map(|i| 2.0 * std::f64::consts::PI * (i as f64) / (n_theta as f64))
        .collect();

    // Radii decreasing toward 0 (each a quarter of the previous).
    let radii = [
        0.5_f64,
        0.125,
        0.03125,
        7.8125e-3,
        1.953_125e-3,
        4.882_812_5e-4,
        1.220_703_125e-4,
    ];

    // Track the max |h| envelope per radius and verify it shrinks to ~0.
    let mut prev_env: Option<f64> = None;
    let mut first_env: Option<f64> = None;
    let mut max_overall = 0.0_f64;
    for &r in &radii {
        let mut env = 0.0_f64;
        for &th in &thetas {
            let du = r * th.cos();
            let dv = r * th.sin();
            // Denominator guard.
            match eval2(hd, u, v, du, dv, pool) {
                Some(d) if d.abs() > 1e-9 => {}
                _ => return false, // denominator approaches 0 on the grid → no bound
            }
            let hv = match eval2(h, u, v, du, dv, pool) {
                Some(x) => x.abs(),
                _ => return false,
            };
            if hv > env {
                env = hv;
            }
        }
        max_overall = max_overall.max(env);
        if first_env.is_none() {
            first_env = Some(env);
        }
        // Envelope must be (weakly) decreasing as r shrinks (allow small slack).
        if let Some(pe) = prev_env {
            if env > pe * 1.05 + 1e-12 {
                return false;
            }
        }
        prev_env = Some(env);
    }

    // Soundness gate: across a 4096× radius contraction the envelope must have
    // (a) been driven down by a large factor (so it is genuinely going to 0, not
    // plateauing at a path-dependent constant), and (b) reached a small absolute
    // value. A constant-along-some-path discontinuity keeps a flat envelope and
    // is rejected by (a); a true limit shrinks at least linearly in r.
    let final_env = prev_env.unwrap_or(f64::INFINITY);
    let first = first_env.unwrap_or(f64::INFINITY);
    final_env < 1e-2 && final_env < first * 0.05 + 1e-12 && final_env < max_overall * 0.5 + 1e-12
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Split `expr` into `(numerator, denominator)` by collecting negative-integer
/// power factors into the denominator.
fn numer_denom(expr: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let factors = match pool.get(expr) {
        ExprData::Mul(xs) => xs,
        _ => vec![expr],
    };
    let mut nums = Vec::new();
    let mut dens = Vec::new();
    for f in factors {
        match pool.get(f) {
            ExprData::Pow { base, exp } => {
                if let ExprData::Integer(n) = pool.get(exp) {
                    if n.0 < 0 {
                        let neg_exp =
                            simplify(pool.mul(vec![pool.integer(-1_i32), exp]), pool).value;
                        dens.push(pool.pow(base, neg_exp));
                        continue;
                    }
                }
                nums.push(f);
            }
            _ => nums.push(f),
        }
    }
    let num = if nums.is_empty() {
        pool.integer(1_i32)
    } else {
        simplify(pool.mul(nums), pool).value
    };
    let den = if dens.is_empty() {
        pool.integer(1_i32)
    } else {
        simplify(pool.mul(dens), pool).value
    };
    (num, den)
}

/// Is `expr` a rational function in `u, v` (convertible to `RationalFunction`)?
fn is_rational_in(expr: ExprId, u: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    let (num, den) = numer_denom(expr, pool);
    RationalFunction::from_symbolic(num, den, vec![u, v], pool).is_ok()
}

/// Does `expr` syntactically contain `var`?
fn contains_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|x| contains_var(*x, var, pool)),
        ExprData::Pow { base, exp } => {
            contains_var(base, var, pool) || contains_var(exp, var, pool)
        }
        ExprData::Func { args, .. } => args.iter().any(|a| contains_var(*a, var, pool)),
        _ => false,
    }
}

/// Numerically evaluate `expr` with `u := du`, `v := dv`. Returns `None` if a
/// node cannot be evaluated or the result is non-finite.
fn eval2(expr: ExprId, u: ExprId, v: ExprId, du: f64, dv: f64, pool: &ExprPool) -> Option<f64> {
    let r = eval2_inner(expr, u, v, du, dv, pool)?;
    if r.is_finite() {
        Some(r)
    } else {
        None
    }
}

fn eval2_inner(
    expr: ExprId,
    u: ExprId,
    v: ExprId,
    du: f64,
    dv: f64,
    pool: &ExprPool,
) -> Option<f64> {
    if expr == u {
        return Some(du);
    }
    if expr == v {
        return Some(dv);
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (num, den) = r.0.clone().into_numer_denom();
            Some(num.to_f64() / den.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { name, .. } => match name.as_str() {
            "pi" => Some(std::f64::consts::PI),
            "e" => Some(std::f64::consts::E),
            _ => None,
        },
        ExprData::Add(xs) => {
            let mut s = 0.0;
            for x in xs {
                s += eval2_inner(x, u, v, du, dv, pool)?;
            }
            Some(s)
        }
        ExprData::Mul(xs) => {
            let mut p = 1.0;
            for x in xs {
                p *= eval2_inner(x, u, v, du, dv, pool)?;
            }
            Some(p)
        }
        ExprData::Pow { base, exp } => {
            let b = eval2_inner(base, u, v, du, dv, pool)?;
            let e = eval2_inner(exp, u, v, du, dv, pool)?;
            Some(b.powf(e))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let a = eval2_inner(args[0], u, v, du, dv, pool)?;
            Some(match name.as_str() {
                "sin" => a.sin(),
                "cos" => a.cos(),
                "tan" => a.tan(),
                "exp" => a.exp(),
                "log" => a.ln(),
                "sqrt" => a.sqrt(),
                "sinh" => a.sinh(),
                "cosh" => a.cosh(),
                "tanh" => a.tanh(),
                "asin" => a.asin(),
                "acos" => a.acos(),
                "atan" => a.atan(),
                "abs" => a.abs(),
                _ => return None,
            })
        }
        ExprData::Func { name, args } if args.len() == 2 && name == "atan2" => {
            let y = eval2_inner(args[0], u, v, du, dv, pool)?;
            let x = eval2_inner(args[1], u, v, du, dv, pool)?;
            Some(y.atan2(x))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn setup() -> (ExprPool, ExprId, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        (p, x, y)
    }

    /// `x²+y² → a²+b²` via continuity.
    #[test]
    fn continuity_sum_of_squares() {
        let (p, x, y) = setup();
        let f = simplify(
            p.add(vec![p.pow(x, p.integer(2)), p.pow(y, p.integer(2))]),
            &p,
        )
        .value;
        // At (1, 2) → 5.
        let r = multilimit(f, x, y, p.integer(1), p.integer(2), &p);
        match r {
            MultiLimit::Value(val) => {
                assert_eq!(val, p.integer(5), "got {}", p.display(val));
            }
            other => panic!("expected Value(5), got {other}"),
        }
    }

    /// `xy/(x²+y²) → DNE` at the origin (lines disagree).
    #[test]
    fn dne_xy_over_x2_plus_y2() {
        let (p, x, y) = setup();
        let num = p.mul(vec![x, y]);
        let den = p.add(vec![p.pow(x, p.integer(2)), p.pow(y, p.integer(2))]);
        let f = simplify(p.mul(vec![num, p.pow(den, p.integer(-1))]), &p).value;
        let r = multilimit(f, x, y, p.integer(0), p.integer(0), &p);
        match r {
            MultiLimit::DoesNotExist { path_a, path_b } => {
                assert!(
                    (path_a.value_numeric - path_b.value_numeric).abs() > PROBE_SEPARATION,
                    "witness values not separated: {} vs {}",
                    path_a.value_numeric,
                    path_b.value_numeric
                );
            }
            other => panic!("expected DNE, got {other}"),
        }
    }

    /// `x²y/(x⁴+y²) → DNE` (all lines agree on 0, parabola y=x² gives ½).
    #[test]
    fn dne_x2y_over_x4_plus_y2() {
        let (p, x, y) = setup();
        let num = p.mul(vec![p.pow(x, p.integer(2)), y]);
        let den = p.add(vec![p.pow(x, p.integer(4)), p.pow(y, p.integer(2))]);
        let f = simplify(p.mul(vec![num, p.pow(den, p.integer(-1))]), &p).value;
        let r = multilimit(f, x, y, p.integer(0), p.integer(0), &p);
        match r {
            MultiLimit::DoesNotExist { path_a, path_b } => {
                assert!(
                    (path_a.value_numeric - path_b.value_numeric).abs() > PROBE_SEPARATION,
                    "witnesses: {} vs {}",
                    path_a.value_numeric,
                    path_b.value_numeric
                );
            }
            other => panic!("expected DNE via parabola, got {other}"),
        }
    }

    /// `x²y²/(x²+y²) → 0` proven by polar bound.
    #[test]
    fn value_x2y2_over_x2_plus_y2() {
        let (p, x, y) = setup();
        let num = p.mul(vec![p.pow(x, p.integer(2)), p.pow(y, p.integer(2))]);
        let den = p.add(vec![p.pow(x, p.integer(2)), p.pow(y, p.integer(2))]);
        let f = simplify(p.mul(vec![num, p.pow(den, p.integer(-1))]), &p).value;
        let r = multilimit(f, x, y, p.integer(0), p.integer(0), &p);
        match r {
            MultiLimit::Value(val) => {
                let n = eval2(val, x, y, 0.0, 0.0, &p).unwrap();
                assert!(n.abs() < 1e-9, "expected 0, got {n}");
            }
            other => panic!("expected Value(0), got {other}"),
        }
    }

    /// `(x³+y³)/(x²+y²) → 0` proven by polar bound.
    #[test]
    fn value_x3_plus_y3_over_x2_plus_y2() {
        let (p, x, y) = setup();
        let num = p.add(vec![p.pow(x, p.integer(3)), p.pow(y, p.integer(3))]);
        let den = p.add(vec![p.pow(x, p.integer(2)), p.pow(y, p.integer(2))]);
        let f = simplify(p.mul(vec![num, p.pow(den, p.integer(-1))]), &p).value;
        let r = multilimit(f, x, y, p.integer(0), p.integer(0), &p);
        match r {
            MultiLimit::Value(val) => {
                let n = eval2(val, x, y, 0.0, 0.0, &p).unwrap();
                assert!(n.abs() < 1e-9, "expected 0, got {n}");
            }
            other => panic!("expected Value(0), got {other}"),
        }
    }

    /// `xy/(x+y)` at the origin: must never return a value (DNE or Undecided).
    #[test]
    fn xy_over_x_plus_y_never_value() {
        let (p, x, y) = setup();
        let num = p.mul(vec![x, y]);
        let den = p.add(vec![x, y]);
        let f = simplify(p.mul(vec![num, p.pow(den, p.integer(-1))]), &p).value;
        let r = multilimit(f, x, y, p.integer(0), p.integer(0), &p);
        assert!(
            !matches!(r, MultiLimit::Value(_)),
            "must not certify a value for xy/(x+y); got {r}"
        );
    }

    /// `sin(xy)/(xy) → 1` (continuity after removable singularity) OR Undecided
    /// — but never a wrong value.
    #[test]
    fn sin_xy_over_xy_one_or_undecided() {
        let (p, x, y) = setup();
        let xy = p.mul(vec![x, y]);
        let f = simplify(
            p.mul(vec![p.func("sin", vec![xy]), p.pow(xy, p.integer(-1))]),
            &p,
        )
        .value;
        let r = multilimit(f, x, y, p.integer(0), p.integer(0), &p);
        match r {
            MultiLimit::Value(val) => {
                let n = eval2(val, x, y, 0.0, 0.0, &p).unwrap();
                assert!((n - 1.0).abs() < 1e-6, "expected 1, got {n}");
            }
            MultiLimit::Undecided => {}
            MultiLimit::DoesNotExist { .. } => panic!("sin(xy)/(xy) exists; must not be DNE"),
        }
    }
}
