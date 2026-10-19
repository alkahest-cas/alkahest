//! Antiderivatives that are continuous wherever the integrand is.
//!
//! A substitution antiderivative is an antiderivative on each interval where
//! the substitution is smooth, and nowhere else. `t = tan(x/2)` and `t = tan x`
//! leave `atan(g(tan …))` terms that jump by `±π` where the tangent runs off to
//! infinity; a rational substitution (`u = √cot x`, Euler's substitutions)
//! leaves `atan(N/D)` terms that jump where `D` vanishes. The integrand is
//! perfectly regular there, so the fundamental theorem does not apply across
//! the jump, and `F(b) − F(a)` is not the integral (Jeffrey, *Integration to
//! obtain expressions valid on domains of maximum extent*, ISSAC 1993; Jeffrey
//! and Rich, *The evaluation of trigonometric integrals avoiding spurious
//! discontinuities*, TOMS 1994).
//!
//! Two repairs live here, one for each place a jump can hurt:
//!
//! * [`periodic_step_correction`] — **indefinite** results. When every jump of
//!   `F` sits on the pole lattice of a `tan`/`cot`/`sec`/`csc` of a linear
//!   argument, the jump heights are computed *exactly* by one-sided limits and
//!   cancelled by a sawtooth `J·x/T − (J/2π)·atan2(sin φ, cos φ)`, whose
//!   derivative is zero and whose jumps are `+J` on the lattice. The result is
//!   an antiderivative on every interval on which the integrand is continuous.
//! * [`definite_jump_correction`] — **definite** results, any route. Each
//!   `c·atan(g)` term of `F` (in linear position) is scanned for real
//!   crossings of `g` through `±∞`; each confirmed crossing contributes exactly
//!   `c·(±π)` (and `c·(±2π)` for an `atan2` branch-cut crossing), so the value
//!   `F(b) − F(a) − Σ jumps` is exact even though the crossing points are only
//!   located numerically — only their *count and direction* enter the answer.
//!   The corrected `F` is re-scanned for any jump that remains, and the value
//!   is cross-checked against tanh–sinh quadrature before it is accepted.
//!
//! Both are repairs, never sources of new claims: each returns `None` the
//! moment anything it needs cannot be established, and the caller keeps its
//! previous behaviour (a coded refusal, or the local-validity label).

use std::cell::Cell;
use std::collections::HashMap;

use rug::{Integer, Rational};

use crate::calculus::trig_pole::{one_sided_value, pi_multiple, Side};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// Rule name of the global sawtooth correction (see [`periodic_step_correction`]).
pub(crate) const STEP_CORRECTION_RULE: &str = "int_continuous_step_correction";

thread_local! {
    /// Set while the definite-integration path computes its antiderivative.
    ///
    /// That path splits the interval at substitution poles and takes one-sided
    /// limits there, which is the exact form of the correction; a sawtooth
    /// `atan2` term evaluated *at* its own branch point by plain substitution
    /// would hand it the left limit from both sides.
    static RAW_ANTIDERIVATIVE: Cell<bool> = const { Cell::new(false) };
}

/// Guard: while alive, [`step_correction_enabled`] is `false` on this thread.
pub(crate) struct RawAntiderivative {
    prev: bool,
}

impl RawAntiderivative {
    pub(crate) fn enter() -> Self {
        let prev = RAW_ANTIDERIVATIVE.with(|c| c.replace(true));
        RawAntiderivative { prev }
    }
}

impl Drop for RawAntiderivative {
    fn drop(&mut self) {
        RAW_ANTIDERIVATIVE.with(|c| c.set(self.prev));
    }
}

/// Whether indefinite results should be made globally continuous.
pub(crate) fn step_correction_enabled() -> bool {
    !RAW_ANTIDERIVATIVE.with(|c| c.get())
}

fn mentions(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    crate::kernel::subs::mentions_var(e, var, pool)
}

fn closed_value(e: ExprId, pool: &ExprPool) -> Option<f64> {
    crate::jit::eval_interp(e, &HashMap::new(), pool).filter(|v| v.is_finite())
}

fn eval_at(e: ExprId, var: ExprId, x: f64, pool: &ExprPool) -> Option<f64> {
    let mut env = HashMap::new();
    env.insert(var, x);
    crate::jit::eval_interp(e, &env, pool).filter(|v| v.is_finite())
}

fn pi(pool: &ExprPool) -> ExprId {
    crate::eval::symbols::pi_symbol(pool)
}

// ---------------------------------------------------------------------------
// Indefinite: sawtooth correction on a trig-pole lattice
// ---------------------------------------------------------------------------

/// Largest number of distinct pole classes per period handled.
const MAX_POLE_CLASSES: usize = 16;

/// Half-width of the window on which a corrected antiderivative is checked.
const CHECK_WINDOW: f64 = 20.0;

/// `(α, q)` with `u = α·var + q·π`, both rational, `α ≠ 0`.
fn linear_pi_form(u: ExprId, var: ExprId, pool: &ExprPool) -> Option<(Rational, Rational)> {
    let d = crate::diff::diff(u, var, pool).ok()?.value;
    let alpha_e = simplify(d, pool).value;
    if mentions(alpha_e, var, pool) {
        return None;
    }
    let alpha = match pool.get(alpha_e) {
        ExprData::Integer(n) => Rational::from(n.0.clone()),
        ExprData::Rational(r) => r.0.clone(),
        _ => return None,
    };
    if alpha == 0 {
        return None;
    }
    let mut at0 = HashMap::new();
    at0.insert(var, pool.integer(0_i32));
    let beta = simplify(crate::kernel::subs(u, &at0, pool), pool).value;
    let q = pi_multiple(beta, pool)?;
    Some((alpha, q))
}

/// Arguments of every `tan`/`cot`/`sec`/`csc` in `f` that mention `var`.
fn trig_pole_args(f: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<(String, ExprId)>) {
    match pool.get(f) {
        ExprData::Func { name, args } => {
            if args.len() == 1
                && matches!(name.as_str(), "tan" | "cot" | "sec" | "csc")
                && mentions(args[0], var, pool)
                && !out.iter().any(|(n, a)| *n == name && *a == args[0])
            {
                out.push((name.clone(), args[0]));
            }
            for a in args {
                trig_pole_args(a, var, pool, out);
            }
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            for x in xs {
                trig_pole_args(x, var, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            trig_pole_args(base, var, pool, out);
            trig_pole_args(exp, var, pool, out);
        }
        _ => {}
    }
}

fn rat_gcd(a: &Rational, b: &Rational) -> Rational {
    // gcd(p/q, r/s) = gcd(p·s, r·q) / (q·s)
    let num = Integer::from(a.numer() * b.denom()).gcd(&Integer::from(b.numer() * a.denom()));
    Rational::from((num, Integer::from(a.denom() * b.denom())))
}

fn rat_lcm(a: &Rational, b: &Rational) -> Rational {
    // lcm(a, b) = a·b / gcd(a, b), all positive here.
    Rational::from(a * b) / rat_gcd(a, b)
}

/// `r mod m` in `[0, m)`, `m > 0`.
fn rat_mod(r: &Rational, m: &Rational) -> Rational {
    let q = Rational::from(r / m);
    let fl = q.floor();
    r - m * fl
}

/// Make `f` — an antiderivative of `integrand` — continuous across every pole
/// of the `tan`/`cot`/`sec`/`csc` terms it carries at which the integrand is
/// finite.
///
/// With the poles on the lattice `p + kT` (all of them rational multiples of
/// `π`, `T` the common period of the jump pattern), the jump at each class
/// `p` is `J = F(p⁻) − F(p⁺)`, computed exactly by one-sided limits and
/// confirmed equal at `p ± T`. The returned antiderivative is
///
/// ```text
/// F + Σ_p J_p·( x/T − atan2(sin φ_p, cos φ_p)/(2π) ),   φ_p = 2π(x − p)/T + π,
/// ```
///
/// whose added terms have derivative zero and jump by `+J_p` exactly at
/// `p + kT`. `None` when there is nothing to correct, when any jump cannot be
/// established exactly, or when the corrected result fails the derivative
/// gate or still jumps on `[−20, 20]`.
pub(crate) fn periodic_step_correction(
    f: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let mut found = Vec::new();
    trig_pole_args(f, var, pool, &mut found);
    if found.is_empty() {
        return None;
    }
    // Pole lattices, in units of π: x/π = (c − q + k)/α, c = 1/2 for tan/sec,
    // 0 for cot/csc. The jump pattern repeats with period 2/|α| at worst
    // (sec/csc alternate sign), so take the lcm of those.
    let mut lattices: Vec<(Rational, Rational)> = Vec::new(); // (offset, step)
    let mut period: Option<Rational> = None;
    for (name, u) in &found {
        let (alpha, q) = linear_pi_form(*u, var, pool)?;
        let c = if matches!(name.as_str(), "tan" | "sec") {
            Rational::from((1, 2))
        } else {
            Rational::new()
        };
        let step = alpha.clone().abs().recip();
        let offset = Rational::from(&c - &q) / &alpha;
        let t_i = Rational::from(&step * 2u32);
        period = Some(match period {
            None => t_i,
            Some(p) => rat_lcm(&p, &t_i),
        });
        lattices.push((offset, step));
    }
    let period = period?;
    let mut classes: Vec<Rational> = Vec::new();
    for (offset, step) in &lattices {
        let count = Rational::from(&period / step);
        if !count.is_integer() {
            return None;
        }
        let n = count.numer().to_usize()?;
        if n > MAX_POLE_CLASSES {
            return None;
        }
        for k in 0..n {
            let p = rat_mod(&(offset + Rational::from(step * k as u32)), &period);
            if !classes.contains(&p) {
                classes.push(p);
            }
        }
        if classes.len() > MAX_POLE_CLASSES {
            return None;
        }
    }

    let pi_e = pi(pool);
    let pi_f = std::f64::consts::PI;
    let t_f = period.to_f64() * pi_f;
    let jump_at = |p: &Rational| -> Option<Option<ExprId>> {
        let p_f = p.to_f64() * pi_f;
        // An integrand that is not finite near `p` is singular there; `F` owes
        // no continuity across it.
        let h = 1e-7 * (1.0 + p_f.abs());
        if eval_at(integrand, var, p_f - h, pool).is_none()
            || eval_at(integrand, var, p_f + h, pool).is_none()
        {
            return Some(None);
        }
        let p_e = pool.mul(vec![rational_expr(p, pool), pi_e]);
        let left = one_sided_value(f, var, p_e, Side::Below, pool)?;
        let right = one_sided_value(f, var, p_e, Side::Above, pool)?;
        let j = simplify(
            pool.add(vec![left, pool.mul(vec![pool.integer(-1_i32), right])]),
            pool,
        )
        .value;
        if mentions(j, var, pool) {
            return None;
        }
        Some(Some(j))
    };

    let mut correction = Vec::new();
    for p in &classes {
        let Some(j) = jump_at(p)? else { continue };
        let j_f = closed_value(j, pool)?;
        if j_f.abs() <= 1e-12 {
            continue;
        }
        // The same jump one period either side: the pattern is periodic.
        for shift in [1_i32, -1] {
            let q = p + Rational::from(&period * shift);
            let Some(jq) = jump_at(&q)? else { return None };
            let jq_f = closed_value(jq, pool)?;
            if (jq_f - j_f).abs() > 1e-9 * (1.0 + j_f.abs()) {
                return None;
            }
        }
        // J·x/T − (J/2π)·atan2(sin φ, cos φ), φ = 2π(x − p)/T + π.
        let two_pi_over_t = pool.mul(vec![
            pool.integer(2_i32),
            rational_expr(&period.clone().recip(), pool),
        ]);
        let shift = pool.mul(vec![rational_expr(p, pool), pi_e]);
        let phi = simplify(
            pool.add(vec![
                pool.mul(vec![
                    two_pi_over_t,
                    pool.add(vec![var, pool.mul(vec![pool.integer(-1_i32), shift])]),
                ]),
                pi_e,
            ]),
            pool,
        )
        .value;
        let saw = pool.func(
            "atan2",
            vec![pool.func("sin", vec![phi]), pool.func("cos", vec![phi])],
        );
        let lin = pool.mul(vec![
            j,
            var,
            pool.pow(
                pool.mul(vec![rational_expr(&period, pool), pi_e]),
                pool.integer(-1_i32),
            ),
        ]);
        let sawterm = pool.mul(vec![
            pool.integer(-1_i32),
            j,
            pool.pow(
                pool.mul(vec![pool.integer(2_i32), pi_e]),
                pool.integer(-1_i32),
            ),
            saw,
        ]);
        correction.push(lin);
        correction.push(sawterm);
    }
    if correction.is_empty() {
        return None;
    }
    let mut terms = vec![f];
    terms.extend(correction);
    let corrected = simplify(pool.add(terms), pool).value;

    // Gates: still an antiderivative, and no jump left at any lattice point of
    // the check window where the integrand is finite.
    if super::engine::verify_antiderivative_status(corrected, integrand, var, pool).is_none() {
        return None;
    }
    if t_f <= 0.0 || !t_f.is_finite() {
        return None;
    }
    for p in &classes {
        let p_f = p.to_f64() * pi_f;
        let k0 = ((-CHECK_WINDOW - p_f) / t_f).floor() as i64;
        let k1 = ((CHECK_WINDOW - p_f) / t_f).ceil() as i64;
        if k1 - k0 > 4096 {
            return None;
        }
        for k in k0..=k1 {
            let c = p_f + k as f64 * t_f;
            let h = 1e-7 * (1.0 + c.abs());
            if eval_at(integrand, var, c - h, pool).is_none()
                || eval_at(integrand, var, c + h, pool).is_none()
            {
                continue;
            }
            let (Some(l), Some(r)) = (
                eval_at(corrected, var, c - h, pool),
                eval_at(corrected, var, c + h, pool),
            ) else {
                return None;
            };
            if (r - l).abs() > 1e-4 * (1.0 + l.abs().max(r.abs())) {
                return None;
            }
        }
    }
    Some(corrected)
}

fn rational_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    if r.is_integer() {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

// ---------------------------------------------------------------------------
// Definite: Jeffrey's count of branch crossings
// ---------------------------------------------------------------------------

/// Grid cells used to look for branch crossings of one `atan` term.
const CROSSING_CELLS: usize = 4096;

/// A term `coef·atan(arg)` (or `coef·atan2(arg, den)`) of an antiderivative.
struct AtanTerm {
    coef: ExprId,
    arg: ExprId,
    /// `Some(d)` for `atan2(arg, d)`.
    den: Option<ExprId>,
}

/// Collect the `atan`/`atan2` terms of `f` that sit in *linear* position (a
/// sum of `var`-free multiples of them). `false` when an `atan` of `var`
/// sits anywhere else — inside a product with another `var`-dependent factor,
/// a power, another function — whose jump this module cannot size.
fn atan_terms(
    f: ExprId,
    coef: ExprId,
    var: ExprId,
    pool: &ExprPool,
    out: &mut Vec<AtanTerm>,
) -> bool {
    if !mentions(f, var, pool) {
        return true;
    }
    match pool.get(f) {
        ExprData::Add(xs) => xs.iter().all(|&x| atan_terms(x, coef, var, pool, out)),
        ExprData::Mul(xs) => {
            let (dep, free): (Vec<ExprId>, Vec<ExprId>) =
                xs.iter().partition(|&&x| mentions(x, var, pool));
            if dep.len() == 1 {
                let mut c = free;
                c.push(coef);
                atan_terms(dep[0], pool.mul(c), var, pool, out)
            } else {
                !dep.iter().any(|&d| contains_atan(d, var, pool))
            }
        }
        ExprData::Func { name, args } if name == "atan" && args.len() == 1 => {
            if contains_atan(args[0], var, pool) {
                return false;
            }
            out.push(AtanTerm {
                coef,
                arg: args[0],
                den: None,
            });
            true
        }
        ExprData::Func { name, args } if name == "atan2" && args.len() == 2 => {
            if contains_atan(args[0], var, pool) || contains_atan(args[1], var, pool) {
                return false;
            }
            out.push(AtanTerm {
                coef,
                arg: args[0],
                den: Some(args[1]),
            });
            true
        }
        _ => !contains_atan(f, var, pool),
    }
}

fn contains_atan(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if !mentions(e, var, pool) {
        return false;
    }
    match pool.get(e) {
        ExprData::Func { name, args } => {
            matches!(name.as_str(), "atan" | "atan2" | "acot")
                || args.iter().any(|&a| contains_atan(a, var, pool))
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| contains_atan(x, var, pool)),
        ExprData::Pow { base, exp } => {
            contains_atan(base, var, pool) || contains_atan(exp, var, pool)
        }
        _ => false,
    }
}

/// Branch crossings of `θ(x)` on `(lo, hi)`: each a point where `θ` jumps by
/// (numerically) `±period` between limits that sit on the ends of its range.
/// Returns `(location, sign)` pairs, or `None` when a discontinuity of any
/// other size is found (a jump of the argument itself, which this does not
/// size) or too little of the interval evaluates.
// `!(c0 < mid && mid < c1)` is the NaN-safe "bracket exhausted" test.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn branch_crossings(
    theta: &dyn Fn(f64) -> Option<f64>,
    lo: f64,
    hi: f64,
    period: f64,
) -> Option<Vec<(f64, i32)>> {
    let half = 0.5 * period;
    let n = CROSSING_CELLS;
    let xs: Vec<f64> = (0..=n)
        .map(|i| lo + (hi - lo) * (i as f64) / (n as f64))
        .collect();
    let vals: Vec<Option<f64>> = xs.iter().map(|&x| theta(x)).collect();
    if vals.iter().filter(|v| v.is_some()).count() * 4 < vals.len() * 3 {
        return None;
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        // Next pair of consecutive evaluable samples.
        let Some(a) = vals[i] else {
            i += 1;
            continue;
        };
        let mut j = i + 1;
        while j <= n && vals[j].is_none() {
            j += 1;
        }
        if j > n {
            break;
        }
        let b = vals[j].expect("checked");
        if (b - a).abs() > half {
            // Bisect towards the jump; keep the half with the larger change.
            let (mut c0, mut c1, mut v0, mut v1) = (xs[i], xs[j], a, b);
            for _ in 0..200 {
                let mut mid = 0.5 * (c0 + c1);
                if !(c0 < mid && mid < c1) {
                    break;
                }
                let mut vm = theta(mid);
                if vm.is_none() {
                    // Exactly on the crossing: step off it.
                    mid = c0 + 0.499 * (c1 - c0);
                    vm = theta(mid);
                }
                let Some(vm) = vm else { return None };
                if (vm - v0).abs() >= (v1 - vm).abs() {
                    c1 = mid;
                    v1 = vm;
                } else {
                    c0 = mid;
                    v0 = vm;
                }
            }
            let d = v1 - v0;
            if (d.abs() - period).abs() > 1e-6 * period
                || (v0.abs() - half).abs() > 1e-6 * period
                || (v1.abs() - half).abs() > 1e-6 * period
            {
                // A steep but continuous stretch ends with a small step; any
                // other surviving step is a discontinuity this cannot size.
                if d.abs() < 1e-3 * period {
                    i = j;
                    continue;
                }
                return None;
            }
            out.push((0.5 * (c0 + c1), if d > 0.0 { 1 } else { -1 }));
        }
        i = j;
    }
    Some(out)
}

/// The exact total `Σ (F(x⁺) − F(x⁻))` of the branch-crossing jumps of `f` on
/// `(lo, hi)`, so that `∫_lo^hi = F(hi) − F(lo) − Σ`.
///
/// `None` when the jumps of `f` are not all accounted for this way: an `atan`
/// in a position whose jump cannot be sized, a crossing of unexpected height,
/// or a corrected `f` that still jumps somewhere (re-scanned with the same
/// test that found the original jump). A returned correction has also been
/// checked against tanh–sinh quadrature of the integrand by the caller
/// (see [`quadrature_agrees`]).
pub(crate) fn definite_jump_correction(
    f: ExprId,
    integrand: ExprId,
    var: ExprId,
    lo: f64,
    hi: f64,
    pool: &ExprPool,
) -> Option<ExprId> {
    if !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return None;
    }
    let mut terms = Vec::new();
    if !atan_terms(f, pool.integer(1_i32), var, pool, &mut terms) || terms.is_empty() {
        return None;
    }
    let pi_f = std::f64::consts::PI;
    let no_params = HashMap::new();
    // (location, numeric jump of F) for the re-scan; exact pieces for the sum.
    let mut located: Vec<(f64, f64)> = Vec::new();
    let mut exact: Vec<ExprId> = Vec::new();
    for term in &terms {
        let c = simplify(term.coef, pool).value;
        let c_f = closed_value(c, pool)?;
        let num = crate::jit::Sampler::new(term.arg, var, &no_params, pool);
        let crossings = match term.den {
            None => {
                let theta = |x: f64| num.eval(x).filter(|v| !v.is_nan()).map(f64::atan);
                branch_crossings(&theta, lo, hi, pi_f)?
            }
            Some(d) => {
                let den = crate::jit::Sampler::new(d, var, &no_params, pool);
                let theta = |x: f64| {
                    let (n, d) = (num.eval(x)?, den.eval(x)?);
                    (n.is_finite() && d.is_finite() && (n != 0.0 || d != 0.0)).then(|| n.atan2(d))
                };
                branch_crossings(&theta, lo, hi, 2.0 * pi_f)?
            }
        };
        let unit: i32 = if term.den.is_some() { 2 } else { 1 };
        let mut total = 0_i64;
        for (at, sign) in crossings {
            total += i64::from(sign * unit);
            located.push((at, c_f * f64::from(sign) * f64::from(unit) * pi_f));
        }
        if total != 0 {
            exact.push(pool.mul(vec![pool.integer(total), c, pi(pool)]));
        }
    }
    if located.is_empty() {
        return None;
    }

    // Re-scan the corrected antiderivative for anything left over.
    let f_sampler = crate::jit::Sampler::new(f, var, &no_params, pool);
    let g_sampler = crate::jit::Sampler::new(integrand, var, &no_params, pool);
    let big_f = |x: f64| -> Option<f64> {
        let v = f_sampler.eval(x).filter(|v| v.is_finite())?;
        let steps: f64 = located
            .iter()
            .filter(|(at, _)| *at < x)
            .map(|(_, j)| *j)
            .sum();
        Some(v - steps)
    };
    let small_f = |x: f64| g_sampler.eval(x).filter(|v| v.is_finite());
    if super::engine::jump_scan(&big_f, &small_f, lo, hi).is_some() {
        return None;
    }
    if exact.is_empty() {
        return Some(pool.integer(0_i32));
    }
    Some(simplify(pool.add(exact), pool).value)
}

/// Does the closed value `value` agree with `∫_lo^hi integrand` computed by
/// composite tanh–sinh quadrature? Used as an independent check on a value
/// assembled by [`definite_jump_correction`].
///
/// Tanh–sinh rather than Gauss–Legendre because the integrands that reach
/// here often have an integrable algebraic endpoint singularity (`√x` at
/// `0`), which tanh–sinh resolves to near machine precision and a fixed
/// Gauss rule only to a few digits. Sixteen panels keep the nodes clustered
/// at every panel end, so a kink-free but sharply varying interior is still
/// resolved. A non-finite sample inside the interval is a refusal.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN-safe node test
pub(crate) fn quadrature_agrees(
    integrand: ExprId,
    var: ExprId,
    lo: f64,
    hi: f64,
    value: ExprId,
    pool: &ExprPool,
) -> bool {
    const PANELS: usize = 16;
    const STEP: f64 = 1.0 / 32.0;
    const T_MAX: f64 = 3.5;
    let Some(v) = closed_value(value, pool) else {
        return false;
    };
    if !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return false;
    }
    let sampler = crate::jit::Sampler::new(integrand, var, &HashMap::new(), pool);
    let half_pi = std::f64::consts::FRAC_PI_2;
    let width = (hi - lo) / PANELS as f64;
    let n = (T_MAX / STEP) as i64;
    let mut total = 0.0;
    let mut scale = 0.0;
    for k in 0..PANELS {
        let a = lo + width * k as f64;
        let b = if k + 1 == PANELS { hi } else { a + width };
        let r = 0.5 * (b - a);
        for j in -n..=n {
            let t = j as f64 * STEP;
            let u = half_pi * t.sinh();
            let cu = u.cosh();
            // Distance of the node from the nearer panel end, without the
            // cancellation of `1 − tanh u`.
            let gap = r * (-u.abs()).exp() / cu;
            let x = if t >= 0.0 { b - gap } else { a + gap };
            if !(x > a && x < b) {
                continue;
            }
            let w = STEP * r * half_pi * t.cosh() / (cu * cu);
            if w == 0.0 {
                continue;
            }
            let Some(y) = sampler.eval(x).filter(|y| y.is_finite()) else {
                // At the very ends an integrable singularity may overflow;
                // anywhere else a non-finite integrand is no basis for a check.
                if gap < 1e-12 * (b - a) {
                    continue;
                }
                return false;
            };
            total += w * y;
            scale += w * y.abs();
        }
    }
    (total - v).abs() <= 1e-9 * (1.0 + scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::engine::{integrate, integrate_definite};
    use crate::kernel::Domain;

    fn sqrt2(pool: &ExprPool) -> ExprId {
        pool.pow(pool.integer(2_i32), pool.rational(1_i32, 2_i32))
    }

    fn val(e: ExprId, x: ExprId, at: f64, pool: &ExprPool) -> f64 {
        eval_at(e, x, at, pool).expect("finite value")
    }

    /// `∫ dx/(1 + sin²x) = atan(√2·tan x)/√2` jumps by `π/√2` at every
    /// `π/2 + kπ`; the correction removes every one of those jumps.
    #[test]
    fn sawtooth_correction_removes_tan_pole_jumps() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let s2 = sqrt2(&pool);
        let f = pool.mul(vec![
            pool.pow(s2, pool.integer(-1_i32)),
            pool.func("atan", vec![pool.mul(vec![s2, pool.func("tan", vec![x])])]),
        ]);
        let sin2 = pool.pow(pool.func("sin", vec![x]), pool.integer(2_i32));
        let integrand = pool.pow(
            pool.add(vec![pool.integer(1_i32), sin2]),
            pool.integer(-1_i32),
        );
        let fixed = periodic_step_correction(f, integrand, x, &pool).expect("correctable");
        let h = 1e-8;
        for k in -5..=5 {
            let c = std::f64::consts::FRAC_PI_2 + k as f64 * std::f64::consts::PI;
            let jump = val(fixed, x, c + h, &pool) - val(fixed, x, c - h, &pool);
            assert!(jump.abs() < 1e-6, "jump {jump} at {c}");
        }
        // F(b) − F(a) is the integral over many periods: π/√2 per period.
        let span = val(fixed, x, 3.0 * std::f64::consts::PI, &pool) - val(fixed, x, 0.0, &pool);
        assert!((span - 3.0 * std::f64::consts::PI / 2f64.sqrt()).abs() < 1e-10);
    }

    /// `tan x` is the antiderivative of `sec²x`, which is itself singular at
    /// every pole: nothing to correct.
    #[test]
    fn no_correction_where_the_integrand_is_singular() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("tan", vec![x]);
        let integrand = pool.pow(pool.func("cos", vec![x]), pool.integer(-2_i32));
        assert!(periodic_step_correction(f, integrand, x, &pool).is_none());
    }

    /// `−atan(1/x)` is an antiderivative of `1/(1 + x²)` on each half-line
    /// and jumps by `−π` at `0`: the count of crossings repairs the value.
    #[test]
    fn jeffrey_count_repairs_an_atan_of_a_rational_function() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![
            pool.integer(-1_i32),
            pool.func("atan", vec![pool.pow(x, pool.integer(-1_i32))]),
        ]);
        let integrand = pool.pow(
            pool.add(vec![pool.integer(1_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.integer(-1_i32),
        );
        let c = definite_jump_correction(f, integrand, x, -1.0, 1.0, &pool).expect("sized");
        let c_f = closed_value(c, &pool).unwrap();
        assert!((c_f + std::f64::consts::PI).abs() < 1e-14, "{c_f}");
        let value = (val(f, x, 1.0, &pool) - val(f, x, -1.0, &pool)) - c_f;
        assert!((value - std::f64::consts::FRAC_PI_2).abs() < 1e-14);
        assert!(quadrature_agrees(
            integrand,
            x,
            -1.0,
            1.0,
            pool.mul(vec![pool.rational(1_i32, 2_i32), pi(&pool)]),
            &pool
        ));
    }

    /// A steep but continuous `atan` is not a crossing.
    #[test]
    fn steep_atan_is_not_a_crossing() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let g = pool.mul(vec![
            pool.integer(100_000_000_i64),
            pool.add(vec![x, pool.rational(-1_i32, 2_i32)]),
        ]);
        let theta_e = pool.func("atan", vec![g]);
        let sampler = crate::jit::Sampler::new(theta_e, x, &HashMap::new(), &pool);
        let theta = |t: f64| sampler.eval(t);
        let found = branch_crossings(&theta, 0.0, 1.0, std::f64::consts::PI).unwrap();
        assert!(found.is_empty(), "{found:?}");
    }

    /// End to end: the report's `cot^{3/2}` value, and definite integrals
    /// across the poles of `t = tan x` and `t = tan(x/2)`.
    #[test]
    fn definite_values_across_substitution_poles() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let cot = pool.pow(pool.func("tan", vec![x]), pool.integer(-1_i32));
        let cases: Vec<(ExprId, ExprId, ExprId, f64)> = vec![
            (
                pool.pow(cot, pool.rational(3_i32, 2_i32)),
                pool.rational(1_i32, 5_i32),
                pool.integer(1_i32),
                2.172_029_668_733_5,
            ),
            (
                pool.pow(
                    pool.add(vec![
                        pool.integer(1_i32),
                        pool.pow(pool.func("sin", vec![x]), pool.integer(2_i32)),
                    ]),
                    pool.integer(-1_i32),
                ),
                pool.integer(0_i32),
                pool.integer(5_i32),
                3.477_966_477_74,
            ),
            (
                pool.pow(
                    pool.add(vec![pool.integer(2_i32), pool.func("cos", vec![x])]),
                    pool.integer(-2_i32),
                ),
                pool.integer(0_i32),
                pool.integer(10_i32),
                4.148_293_626_75,
            ),
        ];
        for (f, a, b, want) in cases {
            let r = integrate_definite(f, x, a, b, &pool)
                .unwrap_or_else(|e| panic!("{}: {e}", pool.display(f)));
            let got = closed_value(r.value, &pool).unwrap();
            assert!(
                (got - want).abs() < 1e-9,
                "{}: {got} vs {want}",
                pool.display(f)
            );
        }
    }

    /// The indefinite `∫ dx/(1 + sin²x)` is continuous on [−20, 20].
    #[test]
    fn indefinite_bioche_result_is_global() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(
            pool.add(vec![
                pool.integer(1_i32),
                pool.pow(pool.func("sin", vec![x]), pool.integer(2_i32)),
            ]),
            pool.integer(-1_i32),
        );
        let big_f = integrate(f, x, &pool).expect("integrable").value;
        let mut prev = val(big_f, x, -20.0, &pool);
        let n = 4000;
        for i in 1..=n {
            let t = -20.0 + 40.0 * i as f64 / n as f64;
            let v = val(big_f, x, t, &pool);
            // |f| ≤ 1, so a step of 0.01 moves F by at most 0.01.
            assert!(
                (v - prev).abs() <= 0.0101,
                "jump near {t}: {}",
                pool.display(big_f)
            );
            prev = v;
        }
    }
}
