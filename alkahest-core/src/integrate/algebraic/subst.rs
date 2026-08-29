//! Rationalizing substitutions `uⁿ = g(x)` for radicals with a
//! **non-polynomial** radicand, closed by the shared
//! [`crate::integrate::gate`].
//!
//! # Scope
//!
//! `∫√(tan x) dx` is a textbook elementary integral — the classical
//! `u² = tan x` rationalization — and both Mathematica and SymPy return it.
//! Alkahest declined it: every algebraic route here first asks for the
//! radicand as a polynomial in `x`
//! (`genus_zero::integrate_b_sqrt_high_degree` refuses with *"radicand P is
//! not a polynomial in the variable"*), and `tan x` is not one.
//!
//! This module closes that gap for the family of radicands whose derivative is
//! a **rational function of themselves**:
//!
//! | `g(x)` | `g′` as a function of `g` |
//! |---|---|
//! | `tan(a·x+b)`  | `a·(1 + g²)` |
//! | `cot(a·x+b)`  | `−a·(1 + g²)` |
//! | `tanh(a·x+b)` | `a·(1 − g²)` |
//! | `exp(a·x+b)`  | `a·g` |
//!
//! For such a `g`, setting `u = g(x)^{1/n}` gives `uⁿ = g`, hence
//! `n·u^{n−1} du = g′ dx` and
//!
//! ```text
//!     dx = n·u^{n−1} / R(uⁿ) du,      where g′ = R(g).
//! ```
//!
//! If the integrand is a rational expression in `g` and `g^{1/n}` alone — no
//! bare `x` anywhere — the substitution turns it into a **rational function of
//! `u`**, which is always elementary.
//!
//! # Method — two proposers, one gate
//!
//! Everything here only *proposes*; nothing is emitted that the gate did not
//! check against the **original** integrand in `x`.
//!
//! 1. **Substitution proposer.**  Find the radical core `g` and the root order
//!    `n`, identify `g`'s family, rewrite the integrand in `u`, and confirm the
//!    result really is a rational function of `u` (which also guarantees the
//!    recursive integration below cannot re-enter this route).
//! 2. **`u`-antiderivative proposers**, tried in order:
//!    * the existing engine, with `RootSum` output suppressed — it gives the
//!      nicest closed forms when it can;
//!    * failing that, a **real partial-fraction ansatz**: `log(u − r)` at each
//!      real pole, `log(u² − 2αu + α²+β²)` and `atan((u − α)/β)` at each
//!      complex-conjugate pair, plus a polynomial ladder, with the
//!      coefficients fitted by least squares and snapped to rationals
//!      ([`crate::integrate::gate::fit_blocks`]).  This is the same
//!      propose-fit-verify pattern the elliptic route uses, applied to the
//!      place where Rothstein–Trager would otherwise return an unevaluable
//!      `RootSum` — `∫2u²/(u⁴+1) du` is exactly that case, and it is the one
//!      `∫√(tan x) dx` lands on.
//! 3. **Back-substitute** `u ↦ g(x)^{1/n}` and run the gate on
//!    `d/dx F(x) = f(x)` over the region where the substitution is real
//!    (`g > 0` for even `n`).
//!
//! # Honest limitations
//!
//! * **The radicand family is a closed table, not a decision procedure.**
//!   `g′ = R(g)` for rational `R` is the exact condition, but this module
//!   recognises it only for `tan`/`cot`/`tanh`/`exp` of a *linear* argument.
//!   `g = sin x` fails (its derivative `cos x = ±√(1−g²)` is algebraic, not
//!   rational, in `g`); `g = log x` fails (`1/x = 1/e^g`).  Polynomial
//!   radicands are deliberately excluded — they belong to the genus-0/1
//!   machinery, which produces better forms.
//! * **A bare `x` outside the radical is refused.**  `∫x·√(tan x) dx` would
//!   need `x = ψ(u)`, which is transcendental in `u`, so the substituted
//!   integrand would not be rational.  The rewrite refuses rather than guess.
//! * **The Weierstrass substitution `t = tan(x/2)` is not implemented here.**
//!   It is a different shape — it rationalizes `R(sin x, cos x)` rather than a
//!   radical — and the transcendental engine already owns that route.
//! * **The real partial-fraction ansatz needs simple poles.**  Repeated poles
//!   add `1/(u−r)^k` blocks, but the fit is only attempted when the design is
//!   well conditioned; a high-multiplicity denominator declines.
//! * **The emitted form is verified, not canonical.**  The gate certifies
//!   `d/dx F = f` on the sampled region; it does not claim `F` is continuous
//!   across the branch cuts of `tan`, and the `log` arguments are written
//!   without absolute values, so `F` is real only on the component the gate
//!   actually sampled.  This is the same convention the rest of the integrator
//!   uses.

use std::collections::HashMap;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::integrate::gate;
use crate::integrate::risch::poly_rde::{expr_to_qpoly, is_free_of_var};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::elliptic_output::{classify_roots, float_to_expr, poly_roots};

/// Symbol used for the substituted variable.  The `$…$` fencing matches the
/// convention already used by `parametrize` (`$param_s$`) so it cannot collide
/// with a user symbol.
const U_NAME: &str = "$subst_u$";

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Try to integrate `expr` by a rationalizing substitution `uⁿ = g(x)`.
///
/// Returns `None` when the shape is not one this route handles (so the caller
/// falls through unchanged), and `Some(Err(..))` only when the route was
/// applicable but could not be closed — never a wrong answer.
pub(super) fn try_rationalizing_substitution(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (g, n) = radical_core(expr, var, pool)?;
    let family = Family::detect(g, var, pool)?;

    let u = pool.symbol(U_NAME, Domain::Real);
    let rewritten = rewrite_in_u(expr, var, g, n, u, pool)?;

    // dx = n·u^{n−1} / R(uⁿ) du.
    let jacobian = {
        let un1 = pool.pow(u, pool.integer(n as i32 - 1));
        let r = family.derivative_in_u(u, n, pool);
        pool.mul(vec![
            pool.integer(n as i32),
            un1,
            pool.pow(r, pool.integer(-1_i32)),
        ])
    };
    let u_integrand = simplify(pool.mul(vec![rewritten, jacobian]), pool).value;

    // The substituted integrand must be a genuine rational function of `u`.
    // This is both the mathematical precondition and the recursion guard: a
    // rational integrand carries no radical, so the recursive `integrate` call
    // below cannot come back here.
    let (num, den) =
        crate::integrate::risch::rational_rde::expr_to_qrational(u_integrand, u, pool)?;
    let num_f: Vec<f64> = num.iter().map(|r| r.to_f64()).collect();
    let den_f: Vec<f64> = den.iter().map(|r| r.to_f64()).collect();
    if den_f.iter().all(|&c| c == 0.0) {
        return None;
    }

    // ── Propose antiderivatives in `u` ─────────────────────────────────────
    let mut proposals: Vec<ExprId> = Vec::new();
    if let Some(f_u) = engine_antiderivative(u_integrand, u, pool) {
        proposals.push(f_u);
    }
    if let Some(f_u) = real_form_antiderivative(&num_f, &den_f, u, pool) {
        proposals.push(f_u);
    }
    if proposals.is_empty() {
        return Some(Err(IntegrationError::NotImplemented(format!(
            "rationalizing substitution u^{n} = {} produced a rational integrand \
             this engine cannot integrate",
            pool.display(g)
        ))));
    }

    // ── Back-substitute and gate-verify in `x` ─────────────────────────────
    let root = nth_root_expr(g, n, pool);
    let back = |f_u: ExprId| -> ExprId {
        let mut map = HashMap::new();
        map.insert(u, root);
        simplify(crate::kernel::subs::subs(f_u, &map, pool), pool).value
    };
    let candidates: Vec<ExprId> = proposals.into_iter().map(back).collect();

    let samples = x_samples();
    let in_domain = |x: f64| -> bool {
        match gate::eval_at(g, var, x, pool) {
            Some(v) if v.is_finite() => n % 2 == 1 || v > 1e-4,
            _ => false,
        }
    };
    let domain = gate::Domain::from_samples(samples.clone())
        .with_predicate(in_domain)
        .with_boxes(domain_boxes(
            &samples,
            &|x| match gate::eval_at(g, var, x, pool) {
                Some(v) if v.is_finite() => n % 2 == 1 || v > 1e-4,
                _ => false,
            },
        ));
    let target = gate::Target::symbolic(expr);
    let accepted = gate::verify_first(candidates, &target, var, &domain, &gate_options(), pool)?;

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "rationalizing_substitution",
        expr,
        accepted.antiderivative,
    ));
    Some(Ok(DerivedExpr {
        value: accepted.antiderivative,
        log,
    }))
}

/// Gate configuration for this route.
///
/// Stricter than the elliptic route's default in the one dimension that costs
/// nothing: `min_points` is raised from 3 to 8, because the substitution has a
/// wide, densely sampled real domain and there is no reason to accept thin
/// evidence.  The rigorous enclosure tier is `BestEffort` here — the residual
/// of a `log`/`atan` candidate is a shallow elementary expression that Taylor
/// models handle in milliseconds, unlike the deeply nested elliptic residuals.
fn gate_options() -> gate::GateOptions {
    gate::GateOptions {
        tolerance: 1e-7,
        min_points: 8,
        symbolic: true,
        egraph: false,
        enclosure: gate::EnclosurePolicy::BestEffort(gate::EnclosureBudget::cheap()),
        min_strength: gate::Strength::Sampled,
    }
}

/// A dense real grid for the `x`-level gate, deliberately irrational-ish so it
/// does not land on `tan`'s poles or on `0`.
fn x_samples() -> Vec<f64> {
    let mut xs = Vec::new();
    let mut x = -6.0_f64;
    while x < 6.0 {
        xs.push(x);
        x += 0.0731;
    }
    xs
}

/// Maximal in-domain runs of the sample grid, inset, as enclosure boxes.
///
/// The gate pre-screens these itself, so an over-generous box costs an
/// attempt and nothing else; what matters is that a box never straddles a
/// domain boundary, which the inset guarantees.
fn domain_boxes(samples: &[f64], in_domain: &dyn Fn(f64) -> bool) -> Vec<(f64, f64)> {
    let mut boxes = Vec::new();
    let mut run: Option<(f64, f64)> = None;
    for &x in samples {
        if in_domain(x) {
            run = Some(match run {
                Some((lo, _)) => (lo, x),
                None => (x, x),
            });
        } else if let Some((lo, hi)) = run.take() {
            push_box(&mut boxes, lo, hi);
        }
    }
    if let Some((lo, hi)) = run.take() {
        push_box(&mut boxes, lo, hi);
    }
    boxes.sort_by(|a: &(f64, f64), b: &(f64, f64)| {
        (b.1 - b.0)
            .partial_cmp(&(a.1 - a.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    boxes.truncate(3);
    boxes
}

fn push_box(boxes: &mut Vec<(f64, f64)>, lo: f64, hi: f64) {
    let inset = 0.12 * (hi - lo);
    let (a, b) = (lo + inset, hi - inset);
    if b - a > 0.2 {
        boxes.push((a, b));
    }
}

// ---------------------------------------------------------------------------
// The radical core
// ---------------------------------------------------------------------------

/// The single radical base `g` and the root order `n` (the lcm of every
/// fractional exponent's denominator over `g`).
///
/// Refuses when there is more than one distinct base, when the base is
/// polynomial in `var` (the genus-0/1 routes own that and emit better forms),
/// or when there is no radical at all.
fn radical_core(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, usize)> {
    let mut found: Vec<(usize, ExprId)> = Vec::new();
    scan_radicals(expr, var, pool, &mut found);
    if found.is_empty() {
        return None;
    }
    let base = found[0].1;
    if found.iter().any(|&(_, b)| b != base) {
        return None;
    }
    if expr_to_qpoly(base, var, pool).is_some() {
        return None; // polynomial radicand: not this route's business
    }
    let n = found
        .iter()
        .try_fold(1usize, |acc, &(d, _)| lcm(acc, d).filter(|&v| v <= 12))?;
    if n < 2 {
        return None;
    }
    Some((base, n))
}

fn lcm(a: usize, b: usize) -> Option<usize> {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    a.checked_div(gcd(a, b)).map(|q| q * b)
}

fn scan_radicals(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<(usize, ExprId)>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && !is_free_of_var(args[0], var, pool) =>
        {
            out.push((2, args[0]));
        }
        ExprData::Func { ref name, ref args }
            if name == "cbrt" && args.len() == 1 && !is_free_of_var(args[0], var, pool) =>
        {
            out.push((3, args[0]));
        }
        ExprData::Func { ref args, .. } => {
            for &a in args.iter() {
                scan_radicals(a, var, pool, out);
            }
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
            scan_radicals(base, var, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan_radicals(a, var, pool, out);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Radicand families whose derivative is rational in themselves
// ---------------------------------------------------------------------------

/// `g = h(a·x + b)` with `g′ = R(g)` for a rational `R`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `tan`: `g′ = a(1 + g²)`.
    Tan,
    /// `cot`: `g′ = −a(1 + g²)`.
    Cot,
    /// `tanh`: `g′ = a(1 − g²)`.
    Tanh,
    /// `exp`: `g′ = a·g`.
    Exp,
}

struct Family {
    kind: Kind,
    /// Numerator and denominator of the exact rational slope `a`.
    a: rug::Rational,
}

impl Family {
    fn detect(g: ExprId, var: ExprId, pool: &ExprPool) -> Option<Family> {
        let ExprData::Func { name, args } = pool.get(g) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        let kind = match name.as_str() {
            "tan" => Kind::Tan,
            "cot" => Kind::Cot,
            "tanh" => Kind::Tanh,
            "exp" => Kind::Exp,
            _ => return None,
        };
        // The argument must be linear in `var` with an exact rational slope.
        let p = expr_to_qpoly(args[0], var, pool)?;
        if p.len() > 2 {
            return None;
        }
        let a = p.get(1)?.clone();
        if a == 0 {
            return None;
        }
        Some(Family { kind, a })
    }

    /// `g′` written as a function of `u`, using `g = uⁿ`.
    fn derivative_in_u(&self, u: ExprId, n: usize, pool: &ExprPool) -> ExprId {
        let a = rational_expr(&self.a, pool);
        let un = pool.pow(u, pool.integer(n as i32));
        let u2n = pool.pow(u, pool.integer(2 * n as i32));
        let one = pool.integer(1_i32);
        match self.kind {
            Kind::Tan => pool.mul(vec![a, pool.add(vec![one, u2n])]),
            Kind::Cot => pool.mul(vec![pool.integer(-1_i32), a, pool.add(vec![one, u2n])]),
            Kind::Tanh => pool.mul(vec![
                a,
                pool.add(vec![one, pool.mul(vec![pool.integer(-1_i32), u2n])]),
            ]),
            Kind::Exp => pool.mul(vec![a, un]),
        }
    }
}

fn rational_expr(r: &rug::Rational, pool: &ExprPool) -> ExprId {
    pool.rational(r.numer().clone(), r.denom().clone())
}

// ---------------------------------------------------------------------------
// Rewriting the integrand in `u`
// ---------------------------------------------------------------------------

/// Replace every occurrence of `g^{p/q}` by `u^{n·p/q}` (and `g` itself by
/// `uⁿ`), refusing anything that still mentions `var`.
fn rewrite_in_u(
    expr: ExprId,
    var: ExprId,
    g: ExprId,
    n: usize,
    u: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if expr == g {
        return Some(pool.pow(u, pool.integer(n as i32)));
    }
    if is_free_of_var(expr, var, pool) {
        return Some(expr);
    }
    if expr == var {
        return None; // a bare `x` cannot be written in `u`
    }
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if args.len() == 1 && args[0] == g => {
            let d = match name.as_str() {
                "sqrt" => 2usize,
                "cbrt" => 3usize,
                _ => return None,
            };
            if n % d != 0 {
                return None;
            }
            Some(pool.pow(u, pool.integer((n / d) as i32)))
        }
        ExprData::Pow { base, exp } if base == g => {
            let e = match pool.get(exp) {
                ExprData::Integer(i) => rug::Rational::from(i.0.clone()),
                ExprData::Rational(r) => r.0.clone(),
                _ => return None,
            };
            let scaled = e * rug::Rational::from(n as u32);
            if *scaled.denom() != 1 {
                return None;
            }
            let k = scaled.numer().to_i32()?;
            Some(pool.pow(u, pool.integer(k)))
        }
        ExprData::Pow { base, exp } => {
            if !is_free_of_var(exp, var, pool) {
                return None;
            }
            let b = rewrite_in_u(base, var, g, n, u, pool)?;
            Some(pool.pow(b, exp))
        }
        ExprData::Add(args) => {
            let mapped: Option<Vec<ExprId>> = args
                .iter()
                .map(|&a| rewrite_in_u(a, var, g, n, u, pool))
                .collect();
            Some(pool.add(mapped?))
        }
        ExprData::Mul(args) => {
            let mapped: Option<Vec<ExprId>> = args
                .iter()
                .map(|&a| rewrite_in_u(a, var, g, n, u, pool))
                .collect();
            Some(pool.mul(mapped?))
        }
        _ => None,
    }
}

/// `g(x)^{1/n}` as an expression, preferring the named `sqrt`/`cbrt` heads.
fn nth_root_expr(g: ExprId, n: usize, pool: &ExprPool) -> ExprId {
    match n {
        2 => pool.func("sqrt", vec![g]),
        3 => pool.func("cbrt", vec![g]),
        _ => pool.pow(
            g,
            pool.rational(rug::Integer::from(1), rug::Integer::from(n as u32)),
        ),
    }
}

// ---------------------------------------------------------------------------
// Proposer 1 — the existing engine, without `RootSum`
// ---------------------------------------------------------------------------

/// Integrate the rational `u`-integrand with the existing engine.
///
/// `RootSum` output is suppressed: a `RootSum` candidate is unevaluable by the
/// gate's numeric tier and opaque to `simplify`, so it could never clear the
/// gate — building one would be pure waste, and the guard makes the engine
/// decline early instead.  When it declines, proposer 2 takes over.
fn engine_antiderivative(u_integrand: ExprId, u: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let _guard = crate::integrate::risch::rational_integrate::RootSumSuppressed::enter();
    crate::integrate::integrate(u_integrand, u, pool)
        .ok()
        .map(|d| d.value)
        .filter(|&f| !contains_root_sum(f, pool))
}

fn contains_root_sum(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::RootSum { .. } => true,
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_root_sum(a, pool))
        }
        ExprData::Func { args, .. } => args.iter().any(|&a| contains_root_sum(a, pool)),
        ExprData::Pow { base, exp } => {
            contains_root_sum(base, pool) || contains_root_sum(exp, pool)
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Proposer 2 — real partial-fraction ansatz, fitted
// ---------------------------------------------------------------------------

/// Propose `∫ num/den du` in **real** closed form by fitting a
/// log / arctan / polynomial ansatz built from the poles of `den`.
///
/// This is the propose half of the pattern: the coefficients come from least
/// squares and mean nothing on their own.  The caller back-substitutes and
/// runs the gate on the result in `x`.
///
/// It exists because Rothstein–Trager answers `∫2u²/(u⁴+1) du` with a
/// `RootSum` over the roots of `t⁴ + 1/16` — correct, but unevaluable by the
/// gate and unreadable as an answer.  The same integral in the real basis
/// `{log(u²∓√2u+1), atan(√2u∓1)}` is the classical closed form.
fn real_form_antiderivative(
    num: &[f64],
    den: &[f64],
    u: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let blocks = real_blocks(num, den, u, pool)?;
    if blocks.is_empty() || blocks.len() > 12 {
        return None;
    }
    let samples = u_samples(den);
    let target = |uv: f64| -> Option<f64> {
        let d = eval_poly(den, uv);
        if d.abs() < 1e-6 {
            return None;
        }
        let v = eval_poly(num, uv) / d;
        v.is_finite().then_some(v)
    };
    let fit_opts = gate::FitOptions::default();
    let coeffs = gate::fit_blocks(&blocks, u, &samples, &target, &fit_opts, pool)?;
    gate::assemble(
        &blocks,
        &coeffs,
        &fit_opts,
        &|c, p| float_to_expr(c, p),
        pool,
    )
}

/// The real ansatz basis for `∫ num/den du`.
fn real_blocks(num: &[f64], den: &[f64], u: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let deg_den = trimmed_degree(den)?;
    let deg_num = trimmed_degree(num).unwrap_or(0);
    if deg_den == 0 || deg_den > 8 {
        return None;
    }
    let roots = poly_roots(&den[..=deg_den])?;
    let (reals, pairs) = classify_roots(&roots);

    let mut blocks: Vec<ExprId> = Vec::new();

    // Polynomial part: `∫ u^j du` contributes `u^{j+1}`, so a numerator of
    // degree `deg_num ≥ deg_den` needs the ladder up to `deg_num−deg_den+1`.
    if deg_num >= deg_den {
        for j in 1..=(deg_num - deg_den + 1) {
            blocks.push(pool.pow(u, pool.integer(j as i32)));
        }
    }

    // Real poles: `log(u − r)`, plus `1/(u−r)^k` for repeated ones.
    let mut seen: Vec<f64> = Vec::new();
    for r in reals {
        let mult = 1 + seen.iter().filter(|&&s| (s - r).abs() < 1e-6).count();
        seen.push(r);
        if mult == 1 {
            let ur = pool.add(vec![u, float_to_expr(-r, pool)]);
            blocks.push(pool.func("log", vec![ur]));
        } else {
            let ur = pool.add(vec![u, float_to_expr(-r, pool)]);
            blocks.push(pool.pow(ur, pool.integer(-(mult as i32 - 1))));
        }
    }

    // Complex-conjugate pairs `α ± iβ`: the real quadratic
    // `Q = u² − 2αu + (α²+β²)` contributes `log Q` and `atan((u−α)/β)`.
    let mut seen_pairs: Vec<(f64, f64)> = Vec::new();
    for (alpha, beta) in pairs {
        if beta.abs() < 1e-9 {
            return None;
        }
        let mult = 1 + seen_pairs
            .iter()
            .filter(|&&(a, b)| (a - alpha).abs() < 1e-6 && (b - beta).abs() < 1e-6)
            .count();
        seen_pairs.push((alpha, beta));
        let q = pool.add(vec![
            pool.pow(u, pool.integer(2_i32)),
            pool.mul(vec![float_to_expr(-2.0 * alpha, pool), u]),
            float_to_expr(alpha * alpha + beta * beta, pool),
        ]);
        if mult == 1 {
            blocks.push(pool.func("log", vec![q]));
            let shifted = pool.mul(vec![
                pool.add(vec![u, float_to_expr(-alpha, pool)]),
                float_to_expr(1.0 / beta, pool),
            ]);
            blocks.push(pool.func("atan", vec![shifted]));
        } else {
            let k = pool.integer(-(mult as i32 - 1));
            blocks.push(pool.pow(q, k));
            blocks.push(pool.mul(vec![u, pool.pow(q, k)]));
        }
    }
    Some(blocks)
}

/// Sample abscissae for the `u`-fit: a dense grid away from the poles of
/// `den` and from the real roots (where a `log(u − r)` block is undefined).
fn u_samples(den: &[f64]) -> Vec<f64> {
    let reals = trimmed_degree(den)
        .and_then(|d| poly_roots(&den[..=d]))
        .map(|r| classify_roots(&r).0)
        .unwrap_or_default();
    let lo = reals
        .iter()
        .fold(-3.0_f64, |m, &r| if r + 0.4 > m { r + 0.4 } else { m });
    let mut xs = Vec::new();
    let mut x = lo;
    while x < lo + 7.0 {
        if eval_poly(den, x).abs() > 1e-3 {
            xs.push(x);
        }
        x += 0.0913;
    }
    xs
}

fn trimmed_degree(p: &[f64]) -> Option<usize> {
    p.iter().rposition(|&c| c.abs() > 1e-14)
}

fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        (pool, x)
    }

    /// The headline case: `∫√(tan x) dx`, which declined before this route
    /// existed ("radicand P is not a polynomial in the variable").
    #[test]
    fn sqrt_tan_integrates_and_gate_verifies() {
        let (pool, x) = setup();
        let e = pool.func("sqrt", vec![pool.func("tan", vec![x])]);
        let got = crate::integrate::integrate(e, x, &pool).expect("∫√(tan x) dx should close");
        let s = pool.display(got.value).to_string();
        assert!(s.contains("atan"), "expected an arctan term: {s}");
        assert!(s.contains("log"), "expected a log term: {s}");
        assert!(!s.contains("RootSum"), "should not be a RootSum: {s}");

        // Independent re-verification: d/dx F = √(tan x) on (0, π/2).
        let d = crate::diff::diff(got.value, x, &pool).unwrap().value;
        let ds = simplify(d, &pool).value;
        let mut checked = 0;
        for k in 1..30 {
            let xv = k as f64 * 0.05;
            if xv >= 1.5 {
                break;
            }
            let lhs = gate::eval_at(ds, x, xv, &pool).expect("derivative evaluates");
            let rhs = xv.tan().sqrt();
            assert!(
                (lhs - rhs).abs() < 1e-8 * (1.0 + rhs.abs()),
                "x={xv}: {lhs} vs {rhs}"
            );
            checked += 1;
        }
        assert!(checked >= 20);
    }

    /// `∫dx/√(tan x)` — the reciprocal radical, same substitution.
    #[test]
    fn one_over_sqrt_tan_integrates() {
        let (pool, x) = setup();
        let e = pool.pow(
            pool.func("sqrt", vec![pool.func("tan", vec![x])]),
            pool.integer(-1_i32),
        );
        let got = crate::integrate::integrate(e, x, &pool).expect("∫dx/√(tan x) should close");
        let d = crate::diff::diff(got.value, x, &pool).unwrap().value;
        let ds = simplify(d, &pool).value;
        for k in 3..25 {
            let xv = k as f64 * 0.06;
            let lhs = gate::eval_at(ds, x, xv, &pool).expect("derivative evaluates");
            let rhs = 1.0 / xv.tan().sqrt();
            assert!(
                (lhs - rhs).abs() < 1e-8 * (1.0 + rhs.abs()),
                "x={xv}: {lhs} vs {rhs}"
            );
        }
    }

    /// `tanh` is the other member of the table that the engine's dispatch
    /// actually routes here (`exp` is claimed by the Risch engine first), so
    /// it is the test that shows the family is a family and not a special
    /// case for `tan`.
    #[test]
    fn sqrt_tanh_integrates() {
        let (pool, x) = setup();
        let e = pool.func("sqrt", vec![pool.func("tanh", vec![x])]);
        let got = crate::integrate::integrate(e, x, &pool).expect("∫√(tanh x) dx should close");
        let s = pool.display(got.value).to_string();
        assert!(s.contains("log") && s.contains("atan"), "{s}");
        let ds = simplify(crate::diff::diff(got.value, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for k in 1..30 {
            let xv = k as f64 * 0.13;
            let lhs = gate::eval_at(ds, x, xv, &pool).expect("derivative evaluates");
            let rhs = xv.tanh().sqrt();
            assert!(
                (lhs - rhs).abs() < 1e-8 * (1.0 + rhs.abs()),
                "x={xv}: {lhs} vs {rhs}"
            );
            checked += 1;
        }
        assert!(checked >= 25);
    }

    /// `cot` is in the table mathematically but has no entry in the primitive
    /// registry at all — no derivative rule, no numeric kernel — so nothing
    /// downstream can differentiate or evaluate the candidate and the route
    /// declines.  Pinned so that registering `cot` later is a visible change
    /// rather than a silent one.
    #[test]
    fn sqrt_cot_declines_for_want_of_a_registered_primitive() {
        let (pool, x) = setup();
        let e = pool.func("sqrt", vec![pool.func("cot", vec![x])]);
        assert!(crate::integrate::integrate(e, x, &pool).is_err());
    }

    /// A bare `x` outside the radical makes the substituted integrand
    /// non-rational, so the route must refuse rather than guess.
    #[test]
    fn bare_x_outside_the_radical_is_refused() {
        let (pool, x) = setup();
        let e = pool.mul(vec![x, pool.func("sqrt", vec![pool.func("tan", vec![x])])]);
        assert!(try_rationalizing_substitution(e, x, &pool).is_none());
    }

    /// A polynomial radicand belongs to the genus-0/1 machinery; this route
    /// must not intercept it.
    #[test]
    fn polynomial_radicand_is_left_to_the_algebraic_engine() {
        let (pool, x) = setup();
        let p = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let e = pool.func("sqrt", vec![p]);
        assert!(try_rationalizing_substitution(e, x, &pool).is_none());
        // …and `∫√(x³+1) dx` still reaches the elliptic route unchanged.
        let got = crate::integrate::integrate(e, x, &pool);
        let s = got.map(|d| pool.display(d.value).to_string()).unwrap();
        assert!(s.contains("Elliptic"), "{s}");
    }

    /// `g = sin x` has `g′ = √(1−g²)`, algebraic rather than rational in `g`:
    /// outside the table, so the route declines.
    #[test]
    fn sqrt_sin_declines() {
        let (pool, x) = setup();
        let e = pool.func("sqrt", vec![pool.func("sin", vec![x])]);
        assert!(try_rationalizing_substitution(e, x, &pool).is_none());
    }

    /// The real-form proposer on the integral that motivates it:
    /// `∫2u²/(u⁴+1) du`, where Rothstein–Trager answers with a `RootSum`.
    #[test]
    fn real_form_closes_two_u_squared_over_u4_plus_one() {
        let (pool, u) = setup();
        let num = [0.0, 0.0, 2.0];
        let den = [1.0, 0.0, 0.0, 0.0, 1.0];
        let f = real_form_antiderivative(&num, &den, u, &pool).expect("real form should fit");
        let d = crate::diff::diff(f, u, &pool).unwrap().value;
        let ds = simplify(d, &pool).value;
        for k in -25..25 {
            let uv = k as f64 * 0.11;
            let lhs = gate::eval_at(ds, u, uv, &pool).expect("evaluates");
            let rhs = 2.0 * uv * uv / (uv.powi(4) + 1.0);
            assert!(
                (lhs - rhs).abs() < 1e-8 * (1.0 + rhs.abs()),
                "u={uv}: {lhs} vs {rhs}"
            );
        }
    }

    /// The `u`-substitution route must reach a **rigorous** verdict, not just
    /// a sampled one: the residual of a `log`/`atan` candidate is a shallow
    /// elementary expression that Taylor models bound over a whole interval.
    #[test]
    fn sqrt_tan_candidate_is_enclosure_verified() {
        let (pool, x) = setup();
        let e = pool.func("sqrt", vec![pool.func("tan", vec![x])]);
        let got = crate::integrate::integrate(e, x, &pool).expect("closes");
        let g = pool.func("tan", vec![x]);
        let samples = x_samples();
        let pred = |v: f64| match gate::eval_at(g, x, v, &pool) {
            Some(w) if w.is_finite() => w > 1e-4,
            _ => false,
        };
        let domain = gate::Domain::from_samples(samples.clone())
            .with_predicate(pred)
            .with_boxes(domain_boxes(&samples, &pred));
        assert!(!domain.boxes().is_empty());
        let opts = gate::GateOptions {
            enclosure: gate::EnclosurePolicy::Required(gate::EnclosureBudget::thorough()),
            min_strength: gate::Strength::Enclosure,
            ..gate_options()
        };
        let verdict = gate::verify(
            got.value,
            &gate::Target::symbolic(e),
            x,
            &domain,
            &opts,
            &pool,
        );
        match verdict {
            gate::Verdict::EnclosureVerified {
                ref boxes,
                residual_bound,
                ..
            } => {
                assert!(!boxes.is_empty());
                assert!(residual_bound <= 1e-8, "bound {residual_bound:e}");
            }
            other => panic!("expected a rigorous enclosure, got {other:?}"),
        }
    }
}
