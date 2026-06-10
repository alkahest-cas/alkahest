//! End-to-end integration of a **radical whose radicand involves the
//! transcendental** — the Risch MD case, e.g. tutorial Example 15
//! `∫ (3(x+eˣ)^{1/3} + (2x²+3x)eˣ + 5x²)/(x(x+eˣ)^{1/3}) dx = 3x(x+eˣ)^{2/3} + 3log x`.
//!
//! The log analogue works the same way over a `t = log(h)` tower (e.g.
//! `∫ (1+1/x)/(3(x+log x)^{2/3}) dx = (x+log x)^{1/3}`).
//!
//! Ties together the M0 generic quotient ring, the [`ExpTowerField`] /
//! [`LogTowerField`] towers ([`super::tower_field`]), and the `solve_tower_rde`
//! tower Risch-DE solver:
//!
//! 1. detect the outermost radical `y = a^{1/n}` whose radicand `a` involves a
//!    single transcendental generator `t = exp(η)` or `t = log(h)`;
//! 2. parse `a` and the integrand's coefficients into `ℚ(x)(t)` ([`TExpr`]) and
//!    decompose the integrand over the power basis `{1, y, …, y^{n−1}}` via the
//!    generic `Quotient` (reducing `yⁿ = a`);
//! 3. per power: `i = 0` is `∫c₀ dx` (recurse into the engine); `i ≥ 1` is the
//!    tower Risch DE `vᵢ' + (i/n)(a'/a)·vᵢ = cᵢ` solved by `solve_tower_rde`;
//! 4. reconstruct `F = Σ vᵢ·a^{i/n} + ∫c₀` and **verify `d/dx F = integrand`
//!    numerically** — so the whole path is sound (a wrong reconstruction or an
//!    unsolved component yields `None`, never an incorrect antiderivative).
//!
//! Scope: a single exp/log generator, polynomial `η`/`h`, and per-component
//! solutions within `solve_tower_rde`'s ansatz.  Anything else → `None`.
//!
//! ## M4 PR2 — the trait-recursive multi-generator case
//!
//! [`try_integrate_exp_times_radical_over_tower`] adds the first genuinely
//! *multi-generator* path: `∫ exp(η)·R dx` where the **outer** transcendental
//! coefficient `exp(η)` (`η` a polynomial in `x`) multiplies a rational function
//! `R` over a radical `y = a^{1/n}` whose radicand involves a **separate**
//! transcendental `t = log(h)` (or `exp`).  Seeking `F = exp(η)·w`, the integral
//! becomes the twisted Risch DE `D(w) + η'·w = R` over
//! `Quotient<LogTowerField|ExpTowerField>`; decomposing over the radical power
//! basis gives per-component `D(wᵢ) + (η' + (i/n)a'/a)·wᵢ = Rᵢ`, each solved by
//! **descending one tower level** through [`DifferentialField`]`::rational_rde`
//! on the coefficient field — the M4 recursion.  Same numeric soundness gate; a
//! `None` falls through (never a wrong answer).  This catches integrands the
//! single-generator path declines (e.g.
//! `∫[eˣ√(x+log x)+eˣ(1+1/x)/(2√(x+log x))]dx = eˣ√(x+log x)`); it is purely
//! additive.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::alg_field::{RatFn, RationalFunctionField};
use super::diff_field::DifferentialField;
use super::exp_case::build_rational;
use super::number_field::{
    gdegree, gext_gcd, gpoly_divrem, gpoly_mul, gtrim, CoeffField, Quotient,
};
use super::poly_rde::{contains_subexpr, expr_to_qpoly, is_free_of_var, poly_deriv};
use super::radical_ext::RadicalExt;
use super::tower::find_generators;
use super::tower_field::{solve_tower_rde_generic, ExpTowerField, LogTowerField, TExpr};

use rug::Rational;
use std::collections::HashMap;

/// Element of the radical extension over the tower: coefficients of
/// `1, y, …, y^{n−1}`, each in `ℚ(x)(t)`.
type TowerElem = Vec<TExpr>;

/// Backwards-compatible alias of [`try_integrate_radical_over_transcendental`]
/// (originally exp-only; now also handles log towers).  Retained as a stable
/// public symbol.
pub fn try_integrate_radical_over_exp(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    try_integrate_radical_over_transcendental(expr, var, pool)
}

/// Public entry: try to integrate a radical whose radicand involves a single
/// **exp or log** transcendental generator.  Returns `None` when the integrand
/// is not of this shape, so the caller falls through to the ordinary dispatch.
pub fn try_integrate_radical_over_transcendental(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, a_expr) = detect_radical_with_transcendental_radicand(expr, var, pool)?;

    // The radicand must involve exactly one transcendental generator.
    let gens = find_generators(a_expr, var, pool);
    if gens.len() != 1 {
        return None;
    }
    let g = &gens[0];
    if g.is_exp() {
        // t = exp(η),  Dt = η'·t.
        let eta = g.argument();
        let t_gen = pool.func("exp", vec![eta]);
        let eta_poly = expr_to_qpoly(eta, var, pool)?;
        let field = ExpTowerField::new(RatFn::from_poly(&poly_deriv(&eta_poly)));
        integrate_radical_over_tower(&field, t_gen, expr, n, a_expr, var, pool)
    } else if g.is_log() {
        // t = log(h),  Dt = h'/h.
        let h = g.argument();
        let t_gen = pool.func("log", vec![h]);
        let h_poly = expr_to_qpoly(h, var, pool)?;
        let dh_over_h = RatFn::new(poly_deriv(&h_poly), h_poly);
        let field = LogTowerField::new(dh_over_h);
        integrate_radical_over_tower(&field, t_gen, expr, n, a_expr, var, pool)
    } else {
        None
    }
}

// ===========================================================================
// M4 PR2 — multi-generator recursive integrator
//   ∫ exp(η)·(radical over a log/exp tower) dx
// ===========================================================================

/// Try to integrate `∫ exp(η)·R dx` where `R` is a rational function over a
/// radical `y = a^{1/n}` whose radicand `a` involves a **single, different**
/// transcendental `t = log(h)` (or `t = exp(…)`), and `η` is a polynomial in
/// `x` alone.
///
/// This is the M4 multi-generator case: the integrand mixes the *outer*
/// transcendental coefficient `exp(η)` with a radical over a *separate* tower.
/// Seeking an antiderivative `F = exp(η)·w` with `w` in the radical extension,
/// `D(F) = exp(η)·(D(w) + η'·w)`, so we must solve the twisted Risch DE
/// `D(w) + η'·w = R` in `Quotient<LogTowerField>`.  The radical power basis
/// `{1, y, …, y^{n−1}}` diagonalizes the twist (`D(yⁱ) = (i/n)(a'/a)·yⁱ`),
/// giving per-component equations
///
/// ```text
///   D(wᵢ) + (η' + (i/n)·a'/a)·wᵢ = Rᵢ      over ℚ(x)(t),
/// ```
///
/// each solved by **descending one tower level** via
/// [`DifferentialField::rational_rde`] on the coefficient field (the M4
/// recursion).  The reconstructed `F = exp(η)·Σ wᵢ yⁱ` is accepted **only**
/// after the numeric soundness gate `d/dx F = integrand` passes; otherwise
/// `None` (the caller falls through, never a wrong answer).
///
/// Returns `None` when the integrand is not of this shape.
pub fn try_integrate_exp_times_radical_over_tower(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    // The radical and its radicand `a` (single transcendental generator `t`).
    let (n, a_expr) = detect_radical_with_transcendental_radicand(expr, var, pool)?;
    let inner_gens = find_generators(a_expr, var, pool);
    if inner_gens.len() != 1 {
        return None;
    }
    let inner = &inner_gens[0];

    // The outer factor exp(η): an exp generator of the *whole* integrand that is
    // not the radicand's generator and whose argument η is a polynomial in x.
    let all_gens = find_generators(expr, var, pool);
    let mut eta: Option<ExprId> = None;
    for g in &all_gens {
        if !g.is_exp() {
            continue;
        }
        if g.generator == inner.generator {
            continue; // that's the radicand's own generator, not the outer factor
        }
        // η must be a polynomial in x (no other generators inside it).
        if expr_to_qpoly(g.argument(), var, pool).is_some() {
            if eta.is_some() {
                return None; // more than one candidate outer exp — out of scope
            }
            eta = Some(g.argument());
        }
    }
    let eta = eta?;
    let exp_eta = pool.func("exp", vec![eta]);

    // η' as a ℚ(x) element.
    let eta_poly = expr_to_qpoly(eta, var, pool)?;
    let eta_prime = RatFn::from_poly(&poly_deriv(&eta_poly));

    // Build the inner tower field (the radicand lives here).
    if inner.is_log() {
        let h = inner.argument();
        let t_gen = pool.func("log", vec![h]);
        let h_poly = expr_to_qpoly(h, var, pool)?;
        let dh_over_h = RatFn::new(poly_deriv(&h_poly), h_poly);
        let field = LogTowerField::new(dh_over_h);
        integrate_exp_times_radical(
            &field, t_gen, exp_eta, eta_prime, expr, n, a_expr, var, pool,
        )
    } else if inner.is_exp() {
        let inner_eta = inner.argument();
        let t_gen = pool.func("exp", vec![inner_eta]);
        let inner_eta_poly = expr_to_qpoly(inner_eta, var, pool)?;
        let field = ExpTowerField::new(RatFn::from_poly(&poly_deriv(&inner_eta_poly)));
        integrate_exp_times_radical(
            &field, t_gen, exp_eta, eta_prime, expr, n, a_expr, var, pool,
        )
    } else {
        None
    }
}

/// Core of [`try_integrate_exp_times_radical_over_tower`], generic over the
/// inner tower field `F` (which must be a [`DifferentialField`] so we can
/// recurse via [`DifferentialField::rational_rde`]).
#[allow(clippy::too_many_arguments)]
fn integrate_exp_times_radical<F>(
    field: &F,
    t_gen: ExprId,
    exp_eta: ExprId,
    eta_prime: RatFn,
    expr: ExprId,
    n: usize,
    a_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>>
where
    F: CoeffField<Elem = TExpr> + DifferentialField<Elem = TExpr> + Clone,
{
    // R = integrand / exp(η): divide out the outer exp factor so the remaining
    // coefficients are pure rational-in-(x, t) (free of exp(η)).  Divide each
    // additive term separately and simplify it so the exp(η)·exp(η)⁻¹
    // cancellation actually fires (a single top-level multiply may not
    // distribute), then re-assemble; finally require R to be free of exp(η).
    let neg1 = pool.integer(-1_i32);
    let inv_exp_eta = pool.pow(exp_eta, neg1);
    let div_term = |t: ExprId, pool: &ExprPool| -> ExprId {
        simplify(pool.mul(vec![t, inv_exp_eta]), pool).value
    };
    let r_expr = match pool.get(expr) {
        ExprData::Add(args) => {
            let parts: Vec<ExprId> = args.iter().map(|&t| div_term(t, pool)).collect();
            simplify(pool.add(parts), pool).value
        }
        _ => div_term(expr, pool),
    };
    // R must no longer mention exp(η); otherwise the cancellation did not
    // succeed and this is not the supported shape.
    if contains_subexpr(r_expr, exp_eta, pool) {
        return None;
    }

    // The quotient ring ℚ(x)(t)[y]/(yⁿ − a) (for decomposing R over the
    // power basis), and the *same* extension as a generic `DifferentialField`
    // (for the per-component RDE descent — M4 PR3).
    let a_tx = expr_to_texpr(field, a_expr, var, t_gen, pool)?;
    let mut modulus = vec![<F as CoeffField>::zero(field); n + 1];
    modulus[0] = <F as CoeffField>::neg(field, &a_tx);
    modulus[n] = <F as CoeffField>::one(field);
    let q = Quotient::new(field.clone(), modulus);
    let ext = RadicalExt::new(field.clone(), a_tx, n);

    // Decompose R over the power basis {1, y, …, y^{n−1}}.
    let r_elem = decompose_over_tower_radical(r_expr, n, a_expr, &q, field, var, t_gen, pool)?;

    // Solve D(w) + η'·w = R over the radical extension in ONE call: the generic
    // `RadicalExt::rational_rde` (M4 PR3) performs the per-component descent
    //   D(wᵢ) + (η' + (i/n)·a'/a)·wᵢ = Rᵢ      over ℚ(x)(t)
    // through `DifferentialField::rational_rde` on the lower field, and
    // self-verifies the assembled solution in-field.  `f = η'` is a *base*
    // scalar (the diagonal twist is baked in per component), so this is exactly
    // the supported `f ∈ base` case.
    let eta_prime_elem = vec![TExpr::from_ratfn(eta_prime)]; // η' ∈ base, component 0
    let w_elem =
        <RadicalExt<F> as DifferentialField>::rational_rde(&ext, &eta_prime_elem, &r_elem)?;

    // Reconstruct F = exp(η)·w from the solution components wᵢ.
    let mut w_terms: Vec<ExprId> = Vec::new();
    for (i, wi) in w_elem.iter().enumerate() {
        if <F as CoeffField>::is_zero(field, wi) {
            continue;
        }
        let wi_expr = texpr_to_expr(wi, var, t_gen, pool);
        if i == 0 {
            w_terms.push(wi_expr);
        } else {
            let yi = pool.pow(a_expr, pool.rational(i as i32, n as i32));
            w_terms.push(pool.mul(vec![wi_expr, yi]));
        }
    }

    if w_terms.is_empty() {
        return None;
    }
    let w_raw = if w_terms.len() == 1 {
        w_terms[0]
    } else {
        pool.add(w_terms)
    };
    // F = exp(η)·w.
    let f = simplify(pool.mul(vec![exp_eta, w_raw]), pool).value;

    // Soundness gate: d/dx F must equal the integrand numerically.
    if !verify_derivative(f, expr, var, pool) {
        return None;
    }
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "risch_exp_times_radical_over_tower",
        expr,
        f,
    ));
    Some(Ok(DerivedExpr::with_log(f, log)))
}

/// The field-generic integration core: decompose over the radical, solve each
/// component (`i=0` → engine, `i≥1` → `solve_tower_rde`), reconstruct, and
/// verify `d/dx F = integrand`.
fn integrate_radical_over_tower<F>(
    field: &F,
    t_gen: ExprId,
    expr: ExprId,
    n: usize,
    a_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>>
where
    F: CoeffField<Elem = TExpr> + Clone,
{
    // Radicand a as a ℚ(x)(t) element, and the quotient ℚ(x)(t)[y]/(yⁿ − a).
    let a_tx = expr_to_texpr(field, a_expr, var, t_gen, pool)?;
    let mut modulus = vec![field.zero(); n + 1];
    modulus[0] = field.neg(&a_tx);
    modulus[n] = field.one();
    let q = Quotient::new(field.clone(), modulus);

    // Decompose the integrand over the power basis {1, y, …, y^{n-1}}.
    let elem = decompose_over_tower_radical(expr, n, a_expr, &q, field, var, t_gen, pool)?;

    // P1 — residue criterion (Bronstein tutorial §3.4, eq 16): a *cheap,
    // standalone* NonElementary certifier.  If it fires, the integral is
    // provably non-elementary and we skip the (doomed) solve below.
    if residue_criterion_certifies_nonelementary(&elem, &a_tx, t_gen, n, var, pool) {
        return Some(Err(IntegrationError::NonElementary(
            "radical-over-transcendental integrand fails the residue criterion \
             (Bronstein eq 16): R ∤ κ(R) in K[z], so no elementary antiderivative exists"
                .to_string(),
        )));
    }

    // a'/a in the tower (for ωᵢ).
    let a_prime = field.derivation(&a_tx);
    let a_inv = field.inv(&a_tx)?;
    let log_deriv_a = field.mul(&a_prime, &a_inv);

    let mut terms: Vec<ExprId> = Vec::new();
    let mut log = DerivationLog::new();

    for (i, ci) in elem.iter().enumerate() {
        if field.is_zero(ci) {
            continue;
        }
        if i == 0 {
            // eq 23: ∫ c₀ dx — recurse into the engine.
            let c0_expr = texpr_to_expr(ci, var, t_gen, pool);
            match crate::integrate::engine::integrate(c0_expr, var, pool) {
                Ok(d) => terms.push(d.value),
                Err(_) => return None,
            }
        } else {
            // eq 24: vᵢ' + (i/n)(a'/a)·vᵢ = cᵢ  over ℚ(x)(t).
            let scale = TExpr::from_ratfn(RatFn::new(
                vec![Rational::from(i as i64)],
                vec![Rational::from(n as i64)],
            ));
            let omega = field.mul(&scale, &log_deriv_a);
            let vi = solve_tower_rde_generic(field, &omega, ci)?;
            if field.is_zero(&vi) {
                continue;
            }
            let vi_expr = texpr_to_expr(&vi, var, t_gen, pool);
            let yi = pool.pow(a_expr, pool.rational(i as i32, n as i32));
            terms.push(pool.mul(vec![vi_expr, yi]));
        }
    }

    let raw = match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    };
    let f = simplify(raw, pool).value;

    // Soundness gate: d/dx F must equal the integrand numerically.
    if !verify_derivative(f, expr, var, pool) {
        return None;
    }
    log.push(RewriteStep::simple(
        "risch_radical_over_transcendental",
        expr,
        f,
    ));
    Some(Ok(DerivedExpr::with_log(f, log)))
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Find a single radical generator `a^{1/n}` (`n ≥ 2`) whose radicand `a`
/// depends on `var` and contains a transcendental (exp/log) generator.
pub(super) fn detect_radical_with_transcendental_radicand(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(usize, ExprId)> {
    let mut found: Vec<(usize, ExprId)> = Vec::new();
    scan(expr, var, pool, &mut found);
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

fn scan(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<(usize, ExprId)>) {
    let is_transcendental_radicand = |a: ExprId, pool: &ExprPool| {
        !is_free_of_var(a, var, pool) && !find_generators(a, var, pool).is_empty()
    };
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if name == "cbrt" && args.len() == 1 && is_transcendental_radicand(args[0], pool) =>
        {
            out.push((3, args[0]));
        }
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && is_transcendental_radicand(args[0], pool) =>
        {
            out.push((2, args[0]));
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if let Some(den) = r.0.denom().to_i64() {
                    if den >= 2 && is_transcendental_radicand(base, pool) {
                        out.push((den as usize, base));
                        return;
                    }
                }
            }
            scan(base, var, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan(a, var, pool, out);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Decomposition over y (coefficients in the tower ℚ(x)(t))
// ---------------------------------------------------------------------------

fn elem_pow<F: CoeffField<Elem = TExpr>>(
    q: &Quotient<F>,
    base: &[TExpr],
    m: i64,
) -> Option<TowerElem> {
    if m == 0 {
        return Some(q.from_int(1));
    }
    if m < 0 {
        let inv = q.inv(base)?;
        return elem_pow(q, &inv, -m);
    }
    let mut acc = q.from_int(1);
    for _ in 0..m {
        acc = q.mul(&acc, base);
    }
    Some(acc)
}

#[allow(clippy::too_many_arguments)]
fn decompose_over_tower_radical<F: CoeffField<Elem = TExpr>>(
    expr: ExprId,
    n: usize,
    a_expr: ExprId,
    q: &Quotient<F>,
    field: &F,
    var: ExprId,
    exp_gen: ExprId,
    pool: &ExprPool,
) -> Option<TowerElem> {
    // A subexpression free of the radical is a coefficient in ℚ(x)(t).
    if let Some(tx) = expr_to_texpr(field, expr, var, exp_gen, pool) {
        return Some(q.reduce(&[tx]));
    }

    let generator = || vec![field.zero(), field.one()];
    match pool.get(expr) {
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let m = m.0.to_i64()?;
                let b =
                    decompose_over_tower_radical(base, n, a_expr, q, field, var, exp_gen, pool)?;
                elem_pow(q, &b, m)
            }
            ExprData::Rational(r) => {
                if base == a_expr {
                    let den = r.0.denom().to_i64()?;
                    let num = r.0.numer().to_i64()?;
                    if den >= 1 && (n as i64) % den == 0 {
                        return elem_pow(q, &generator(), num * (n as i64 / den));
                    }
                }
                None
            }
            _ => None,
        },
        ExprData::Func { ref name, ref args }
            if name == "cbrt" && args.len() == 1 && args[0] == a_expr && n % 3 == 0 =>
        {
            elem_pow(q, &generator(), (n / 3) as i64)
        }
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && args[0] == a_expr && n % 2 == 0 =>
        {
            elem_pow(q, &generator(), (n / 2) as i64)
        }
        ExprData::Add(args) => {
            let mut acc = q.from_int(0);
            for &a in &args {
                let t = decompose_over_tower_radical(a, n, a_expr, q, field, var, exp_gen, pool)?;
                acc = q.add(&acc, &t);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = q.from_int(1);
            for &a in &args {
                let t = decompose_over_tower_radical(a, n, a_expr, q, field, var, exp_gen, pool)?;
                acc = q.mul(&acc, &t);
            }
            Some(acc)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Symbolic ↔ TExpr conversion
// ---------------------------------------------------------------------------

fn ratfn_const(r: &Rational) -> RatFn {
    RatFn::from_poly(&vec![r.clone()])
}

fn texpr_pow<F: CoeffField<Elem = TExpr>>(field: &F, b: &TExpr, m: i64) -> Option<TExpr> {
    if m == 0 {
        return Some(field.one());
    }
    if m < 0 {
        let inv = field.inv(b)?;
        return texpr_pow(field, &inv, -m);
    }
    let mut acc = field.one();
    for _ in 0..m {
        acc = field.mul(&acc, b);
    }
    Some(acc)
}

/// Parse a symbolic expression that is a rational function in `x` and the single
/// exponential generator `t = exp(η)` into a [`TExpr`].  Returns `None` if the
/// expression is not of that form (e.g. it contains the radical, a different
/// transcendental, or an irrational constant).
fn expr_to_texpr<F: CoeffField<Elem = TExpr>>(
    field: &F,
    expr: ExprId,
    var: ExprId,
    exp_gen: ExprId,
    pool: &ExprPool,
) -> Option<TExpr> {
    if expr == var {
        return Some(TExpr::from_ratfn(RatFn::from_poly(&vec![
            Rational::from(0),
            Rational::from(1),
        ])));
    }
    if expr == exp_gen {
        return Some(TExpr::t());
    }
    match pool.get(expr) {
        ExprData::Integer(nv) => Some(TExpr::from_ratfn(ratfn_const(&Rational::from(
            nv.0.clone(),
        )))),
        ExprData::Rational(r) => Some(TExpr::from_ratfn(ratfn_const(&r.0))),
        ExprData::Add(args) => {
            let mut acc = field.zero();
            for &a in &args {
                acc = field.add(&acc, &expr_to_texpr(field, a, var, exp_gen, pool)?);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = field.one();
            for &a in &args {
                acc = field.mul(&acc, &expr_to_texpr(field, a, var, exp_gen, pool)?);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let m = m.0.to_i64()?;
                let b = expr_to_texpr(field, base, var, exp_gen, pool)?;
                texpr_pow(field, &b, m)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Reconstruct a [`TExpr`] back to a symbolic expression in `x` and
/// `exp_gen = exp(η)`.
fn texpr_to_expr(tx: &TExpr, var: ExprId, exp_gen: ExprId, pool: &ExprPool) -> ExprId {
    let num_e = tpoly_to_expr(tx.numer(), var, exp_gen, pool);
    let den = tx.denom();
    // Denominator == 1 (single constant term equal to 1)?
    if den.len() == 1 && den[0] == RatFn::int(1) {
        return num_e;
    }
    let den_e = tpoly_to_expr(den, var, exp_gen, pool);
    pool.mul(vec![num_e, pool.pow(den_e, pool.integer(-1_i32))])
}

/// `Σⱼ cⱼ(x)·tʲ` (a `t`-polynomial over `ℚ(x)`) → symbolic, using
/// `tʲ = exp(η)^j`.
fn tpoly_to_expr(p: &[RatFn], var: ExprId, exp_gen: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::new();
    for (j, cj) in p.iter().enumerate() {
        if cj.numer().is_empty() {
            continue; // zero coefficient
        }
        let coeff = build_rational(cj.numer(), cj.denom(), var, pool);
        let term = if j == 0 {
            coeff
        } else {
            pool.mul(vec![coeff, pool.pow(exp_gen, pool.integer(j as i32))])
        };
        terms.push(term);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

// ---------------------------------------------------------------------------
// Numeric verification
// ---------------------------------------------------------------------------

fn verify_derivative(f: ExprId, integrand: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let Ok(df) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    for &xv in &[0.55_f64, 1.3, 2.1] {
        let (Some(lhs), Some(rhs)) = (eval(ds, var, xv, pool), eval(integrand, var, xv, pool))
        else {
            return false;
        };
        if (lhs - rhs).abs() > 1e-6 * (1.0 + rhs.abs()) {
            return false;
        }
    }
    true
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

// ---------------------------------------------------------------------------
// P1 — residue criterion (Bronstein tutorial §3.4, eq 16)
// ---------------------------------------------------------------------------

/// `ℚ(x)`-polynomial in the monomial `t` (ascending degree), the coefficient
/// type of the tower field.
type TPoly = Vec<RatFn>;

/// Decide whether the simple-radical-over-transcendental integrand
/// `f = Σᵢ cᵢ yⁱ` (`y = a^{1/n}`, `cᵢ ∈ ℚ(x)(t)`, single monomial `t`) is
/// **provably non-elementary** via Bronstein's residue criterion.
///
/// Writing the normal part as `G/D` over the power basis `wᵢ = yⁱ` (valid since
/// we require the radicand normal/squarefree, so `H = 1`), with `F = yⁿ − a`
/// the minimal polynomial, the criterion (eq 16) forms
///
/// ```text
///   R(z) = res_t( res_y(G − z·D′, F), D ) ∈ K[z],   K = ℚ(x),
/// ```
///
/// and proves: *if `f` has an elementary integral then `R | κ(R)` in `K[z]`*,
/// where `κ` applies `d/dx` coefficient-wise.  We therefore return `true`
/// (certified `NonElementary`) exactly when `R ∤ κ(R)`.
///
/// **Soundness.**
/// * The `pp_z` (primitive-part) step is omitted: the `z`-content of the inner
///   resultant is free of `z`, so after `res_t` it is a factor in `K` — a *unit*
///   in `K[z]` — and `R | κ(R)` is invariant under units in `K[z]`.
/// * Clearing denominators before each resultant only multiplies the output by
///   factors free of the elimination variable, i.e. further `z`-free units.
/// * `R | κ(R) ⟺ κ(R) = u·R` for some `u ∈ K` (because `deg_z κ(R) ≤ deg_z R`)
///   `⟺` all nonzero `z`-coefficients `rᵢ` share one logarithmic derivative
///   `⟺ Wᵢⱼ = rᵢ′·rⱼ − rᵢ·rⱼ′ = 0` for every pair.  We certify only when some
///   `Wᵢⱼ` is *definitely* nonzero (symbolic + numeric), so a false certificate
///   is impossible; any ambiguity yields `false` (inconclusive).
fn residue_criterion_certifies_nonelementary(
    elem: &[TExpr],
    a_tx: &TExpr,
    t_gen: ExprId,
    n: usize,
    var: ExprId,
    pool: &ExprPool,
) -> bool {
    let f = RationalFunctionField;

    // Classify the monomial t and obtain t′ symbolically.
    let t_is_exp =
        matches!(pool.get(t_gen), ExprData::Func { ref name, .. } if name.as_str() == "exp");
    let Ok(tp) = crate::diff::diff(t_gen, var, pool) else {
        return false;
    };
    let tprime = simplify(tp.value, pool).value;

    // D = common t-denominator of the cᵢ (the integrand's normal denominator in
    // the power basis).
    let mut d_tpoly: TPoly = vec![f.one()];
    for ci in elem {
        if ci.numer().is_empty() {
            continue;
        }
        d_tpoly = tpoly_lcm(&f, &d_tpoly, ci.denom());
    }
    if gdegree(&f, &d_tpoly) < 1 {
        return false; // no t-pole ⇒ residue criterion is vacuous here
    }

    // D must be *normal*: squarefree in t, and (when t = exp is special)
    // coprime to t.
    let dd_dt = formal_t_derivative(&f, &d_tpoly);
    let (g, _, _) = gext_gcd(&f, &d_tpoly, &dd_dt);
    if gdegree(&f, &g) > 0 {
        return false; // repeated t-factor: not Hermite-reduced here
    }
    if t_is_exp && (d_tpoly.is_empty() || f.eq(&d_tpoly[0], &f.zero())) {
        return false; // t | D is special, not normal
    }

    // Fresh indeterminates: T (the monomial), Y (the radical), Z (the RT param).
    let big_t = pool.symbol("$p1_t$", Domain::Complex);
    let big_y = pool.symbol("$p1_y$", Domain::Complex);
    let big_z = pool.symbol("$p1_z$", Domain::Complex);

    // D(x,T), a(x,T), t′(x,T).
    let d_sym = subst_t(
        tpoly_to_expr(&d_tpoly, var, t_gen, pool),
        t_gen,
        big_t,
        pool,
    );
    let a_sym = subst_t(texpr_to_expr(a_tx, var, t_gen, pool), t_gen, big_t, pool);
    let tprime_t = subst_t(tprime, t_gen, big_t, pool);

    // G(x,T,Y) = Σᵢ (cᵢ · D) Yⁱ   (numerators over the common denominator D).
    let mut g_terms: Vec<ExprId> = Vec::new();
    for (i, ci) in elem.iter().enumerate() {
        if ci.numer().is_empty() {
            continue;
        }
        let (quot, rem) = gpoly_divrem(&f, &d_tpoly, ci.denom());
        if !rem.is_empty() {
            return false; // D not a multiple of denom(cᵢ) — shouldn't happen
        }
        let gi = gpoly_mul(&f, ci.numer(), &quot);
        let gi_sym = subst_t(tpoly_to_expr(&gi, var, t_gen, pool), t_gen, big_t, pool);
        let term = if i == 0 {
            gi_sym
        } else {
            pool.mul(vec![gi_sym, pool.pow(big_y, pool.integer(i as i32))])
        };
        g_terms.push(term);
    }
    if g_terms.is_empty() {
        return false;
    }
    let g_sym = if g_terms.len() == 1 {
        g_terms[0]
    } else {
        pool.add(g_terms)
    };

    // D′ = ∂ₓD + (∂_T D)·t′(T)   (total derivative through the tower).
    let (Ok(dx), Ok(dt)) = (
        crate::diff::diff(d_sym, var, pool),
        crate::diff::diff(d_sym, big_t, pool),
    ) else {
        return false;
    };
    let dprime = pool.add(vec![dx.value, pool.mul(vec![dt.value, tprime_t])]);

    // P = G − Z·D′ ;  F = Yⁿ − a.
    let neg1 = pool.integer(-1_i32);
    let p_sym = pool.add(vec![g_sym, pool.mul(vec![neg1, big_z, dprime])]);
    let f_sym = pool.add(vec![
        pool.pow(big_y, pool.integer(n as i32)),
        pool.mul(vec![neg1, a_sym]),
    ]);

    // res_y(P, F), then res_t(·, D).  Clear denominators first: the resultant
    // needs polynomial inputs, and the dropped denominators are free of the
    // elimination variable (z-free units, harmless to the criterion).
    let p_num = numer_denom(p_sym, pool).0;
    let f_num = numer_denom(f_sym, pool).0;
    let Ok(res1) = crate::poly::resultant(p_num, f_num, big_y, pool) else {
        return false;
    };
    let d_num = numer_denom(d_sym, pool).0;
    let Ok(r_res) = crate::poly::resultant(res1.value, d_num, big_t, pool) else {
        return false;
    };
    let r = simplify(r_res.value, pool).value;

    // R(z) ∈ K[z] with K = ℚ(x): test R | κ(R) via the pairwise Wronskian.
    let coeffs = collect_z_coeffs(r, big_z, n, pool);
    for i in 0..coeffs.len() {
        let Ok(ri_p) = crate::diff::diff(coeffs[i], var, pool) else {
            return false;
        };
        for j in (i + 1)..coeffs.len() {
            let Ok(rj_p) = crate::diff::diff(coeffs[j], var, pool) else {
                return false;
            };
            // Wᵢⱼ = rᵢ′·rⱼ − rᵢ·rⱼ′
            let w = pool.add(vec![
                pool.mul(vec![ri_p.value, coeffs[j]]),
                pool.mul(vec![neg1, coeffs[i], rj_p.value]),
            ]);
            if definitely_nonzero(w, var, pool) {
                return true; // R ∤ κ(R) ⇒ certified NonElementary
            }
        }
    }
    false
}

/// `lcm` of two `ℚ(x)[t]` polynomials: `a·b / gcd(a, b)`.
fn tpoly_lcm(f: &RationalFunctionField, a: &TPoly, b: &TPoly) -> TPoly {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }
    let prod = gpoly_mul(f, a, b);
    let (g, _, _) = gext_gcd(f, a, b);
    gpoly_divrem(f, &prod, &g).0
}

/// Formal derivative `d/dt` of a `ℚ(x)[t]` polynomial (no `x`-derivation).
fn formal_t_derivative(f: &RationalFunctionField, p: &TPoly) -> TPoly {
    if p.len() <= 1 {
        return Vec::new();
    }
    let mut out: TPoly = Vec::with_capacity(p.len() - 1);
    for (j, cj) in p.iter().enumerate().skip(1) {
        out.push(f.mul(&RatFn::int(j as i64), cj));
    }
    gtrim(f, out)
}

/// Replace every occurrence of the generator `t_gen` by the fresh symbol `big_t`.
fn subst_t(e: ExprId, t_gen: ExprId, big_t: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(t_gen, big_t);
    crate::kernel::subs(e, &m, pool)
}

/// `z`-coefficients `r₀, …, r_{max_deg}` of a polynomial `r ∈ K[z]`, obtained as
/// `rₖ = (1/k!)·∂_z^k r |_{z=0}`.
fn collect_z_coeffs(r: ExprId, z: ExprId, max_deg: usize, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::with_capacity(max_deg + 1);
    let mut cur = r;
    let mut fact: i64 = 1; // k!
    for k in 0..=max_deg {
        let mut m = HashMap::new();
        m.insert(z, pool.integer(0_i32));
        let at0 = simplify(crate::kernel::subs(cur, &m, pool), pool).value;
        let ck = if k == 0 {
            at0
        } else {
            pool.mul(vec![
                at0,
                pool.pow(pool.integer(fact as i32), pool.integer(-1_i32)),
            ])
        };
        out.push(simplify(ck, pool).value);
        let Ok(d) = crate::diff::diff(cur, z, pool) else {
            break;
        };
        cur = simplify(d.value, pool).value;
        fact *= (k as i64) + 1;
    }
    out
}

/// Conservatively decide whether `e` is a *nonzero* function of `x`: it must not
/// simplify to the literal `0`, must evaluate finitely at every sample point,
/// and be clearly nonzero at one of them.  Any uncertainty (unevaluable, all
/// samples ≈ 0) returns `false`, so a true-zero expression can never be reported
/// nonzero — the property the residue certificate relies on for soundness.
fn definitely_nonzero(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let s = simplify(e, pool).value;
    if s == pool.integer(0_i32) {
        return false;
    }
    let mut saw_nonzero = false;
    for &xv in &[0.37_f64, 0.91, 1.73, 2.59, 3.42] {
        match eval(s, var, xv, pool) {
            Some(v) if v.is_finite() => {
                if v.abs() > 1e-9 {
                    saw_nonzero = true;
                }
            }
            Some(_) => {}         // ±∞ / NaN at this sample: skip
            None => return false, // cannot evaluate ⇒ not sure ⇒ inconclusive
        }
    }
    saw_nonzero
}

/// Split `e` into `(numerator, denominator)` over `±·∧` structure, clearing
/// integer-power denominators recursively.  No GCD cancellation is attempted
/// (extra common factors are harmless units for the residue criterion).
/// Non-integer powers and opaque functions are treated as atoms.
fn numer_denom(e: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    match pool.get(e) {
        ExprData::Add(args) => {
            let parts: Vec<(ExprId, ExprId)> = args.iter().map(|&a| numer_denom(a, pool)).collect();
            let common_den = pool.mul(parts.iter().map(|(_, d)| *d).collect());
            let mut num_terms = Vec::with_capacity(parts.len());
            for (i, (ni, _)) in parts.iter().enumerate() {
                let mut factors = vec![*ni];
                for (j, (_, dj)) in parts.iter().enumerate() {
                    if j != i {
                        factors.push(*dj);
                    }
                }
                num_terms.push(pool.mul(factors));
            }
            (pool.add(num_terms), common_den)
        }
        ExprData::Mul(args) => {
            let mut nums = Vec::with_capacity(args.len());
            let mut dens = Vec::with_capacity(args.len());
            for &a in &args {
                let (na, da) = numer_denom(a, pool);
                nums.push(na);
                dens.push(da);
            }
            (pool.mul(nums), pool.mul(dens))
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Integer(k) = pool.get(exp) {
                let ki = k.0.to_i64().unwrap_or(0);
                let (bn, bd) = numer_denom(base, pool);
                if ki >= 0 {
                    (pool.pow(bn, exp), pool.pow(bd, exp))
                } else {
                    let pe = pool.integer((-ki) as i32);
                    (pool.pow(bd, pe), pool.pow(bn, pe))
                }
            } else {
                (e, pool.integer(1_i32))
            }
        }
        _ => (e, pool.integer(1_i32)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    /// Tutorial Example 15:
    /// ∫ (3(x+eˣ)^{1/3} + (2x²+3x)eˣ + 5x²)/(x(x+eˣ)^{1/3}) dx
    ///   = 3x(x+eˣ)^{2/3} + 3 log x.
    fn example15_integrand(pool: &ExprPool, x: ExprId) -> ExprId {
        let exp_x = pool.func("exp", vec![x]);
        let a = pool.add(vec![x, exp_x]); // x + eˣ
        let y = pool.pow(a, pool.rational(1_i32, 3_i32)); // (x+eˣ)^{1/3}
        let x2 = pool.pow(x, pool.integer(2_i32));
        // numerator
        let t1 = pool.mul(vec![pool.integer(3_i32), y]);
        let coeff2 = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x2]),
            pool.mul(vec![pool.integer(3_i32), x]),
        ]); // 2x²+3x
        let t2 = pool.mul(vec![coeff2, exp_x]);
        let t3 = pool.mul(vec![pool.integer(5_i32), x2]);
        let num = pool.add(vec![t1, t2, t3]);
        // denominator x·(x+eˣ)^{1/3}
        let den = pool.mul(vec![x, y]);
        pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
    }

    #[test]
    fn example15_end_to_end() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = example15_integrand(&pool, x);
        let res = crate::integrate::engine::integrate(f, x, &pool);
        assert!(res.is_ok(), "Example 15 should integrate; got {res:?}");
        // d/dx F = f, verified numerically.
        let g = res.unwrap().value;
        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.6_f64, 1.4, 2.3] {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(f, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }
    }

    /// P1 residue criterion (Bronstein eq 16): `∫ √(log x)/(log x − x) dx` is
    /// non-elementary — its residue at the pole `log x = x` is `√x`, which is
    /// not constant.  Hand check with `t = log x`, `y = √t`, `a = t`, `n = 2`:
    /// `D = t − x`, `D′ = 1/x − 1`, `G = y`, so
    /// `R(z) = res_t(res_y(y − z·D′, y² − t), t − x) = (1/x − 1)²·z² − x`, and
    /// `κ(R) = −2(1−x)/x³·z² − 1`.  Then `W₀₂ = r₀′r₂ − r₀r₂′ = −(1−x)(3−x)/x² ≠ 0`,
    /// so `R ∤ κ(R)` and the integral is certified non-elementary.
    #[test]
    fn residue_criterion_certifies_sqrt_log() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let y = pool.pow(log_x, pool.rational(1, 2));
        let den = pool.add(vec![log_x, pool.mul(vec![pool.integer(-1), x])]);
        let f = pool.mul(vec![y, pool.pow(den, pool.integer(-1))]);

        // Direct filter.
        let r = try_integrate_radical_over_transcendental(f, x, &pool);
        assert!(
            matches!(r, Some(Err(IntegrationError::NonElementary(_)))),
            "P1 should certify NonElementary; got {r:?}"
        );
        // End-to-end through the public engine.
        let e = crate::integrate::engine::integrate(f, x, &pool);
        assert!(
            matches!(e, Err(IntegrationError::NonElementary(_))),
            "engine should report NonElementary; got {e:?}"
        );
    }

    /// Degree-3 analogue: `∫ ∛(x + eˣ)/(eˣ − x) dx` — non-elementary via eq 16.
    #[test]
    fn residue_criterion_certifies_cbrt_mixed() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let a = pool.add(vec![x, exp_x]);
        let y = pool.pow(a, pool.rational(1, 3));
        let den = pool.add(vec![exp_x, pool.mul(vec![pool.integer(-1), x])]);
        let f = pool.mul(vec![y, pool.pow(den, pool.integer(-1))]);
        let r = try_integrate_radical_over_transcendental(f, x, &pool);
        assert!(
            matches!(r, Some(Err(IntegrationError::NonElementary(_)))),
            "P1 should certify NonElementary; got {r:?}"
        );
    }

    /// Soundness control: P1 must NOT reject an *elementary* integrand that has
    /// a genuine `t`-pole.  `∫ 1/(2x·√(log x)) dx = √(log x)` (pole at `t = 0`).
    /// The residue criterion holds, so integration proceeds and succeeds.
    #[test]
    fn residue_criterion_keeps_elementary_with_pole() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let y = pool.pow(log_x, pool.rational(1, 2));
        let den = pool.mul(vec![pool.integer(2), x, y]);
        let f = pool.pow(den, pool.integer(-1)); // 1/(2x√(log x))

        let res = crate::integrate::engine::integrate(f, x, &pool);
        assert!(
            res.is_ok(),
            "elementary integrand must not be certified NonElementary; got {res:?}"
        );
        // d/dx F = f, verified numerically.
        let g = res.unwrap().value;
        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in &[1.4_f64, 2.3, 3.1] {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(f, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}"
            );
        }
    }

    #[test]
    fn plain_exp_not_intercepted() {
        // ∫ x·exp(x) dx has no radical → this path returns None.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
        assert!(try_integrate_radical_over_transcendental(f, x, &pool).is_none());
    }

    /// Log-tower analogue: ∫ (1/3)(1+1/x)·(x+log x)^{−2/3} dx = (x+log x)^{1/3}.
    /// d/dx ∛(x+log x) = (1+1/x)/(3(x+log x)^{2/3}).
    #[test]
    fn cbrt_of_x_plus_log_x_end_to_end() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let a = pool.add(vec![x, log_x]); // x + log x
        let coeff = pool.mul(vec![
            pool.rational(1_i32, 3_i32),
            pool.add(vec![pool.integer(1_i32), pool.pow(x, pool.integer(-1_i32))]),
        ]); // (1/3)(1 + 1/x)
        let integrand = pool.mul(vec![coeff, pool.pow(a, pool.rational(-2_i32, 3_i32))]);

        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(res.is_ok(), "∫ should be elementary; got {res:?}");
        let g = res.unwrap().value;
        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.7_f64, 1.5, 2.6] {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(integrand, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }
    }

    /// M4 PR2 headline — multi-generator recursive integrator.
    /// ∫ [ eˣ·√(x+log x) + eˣ·(1+1/x)/(2·√(x+log x)) ] dx = eˣ·√(x+log x).
    ///
    /// Outer transcendental coefficient `eˣ` multiplies a radical
    /// `y = √(x+log x)` over a *separate* log tower.  The integral reduces to the
    /// twisted Risch DE `D(w) + w = R` over `Quotient<LogTowerField>`, solved
    /// per radical component by descending one tower level via
    /// `LogTowerField::rational_rde` (the M4 trait recursion).
    #[test]
    fn exp_times_sqrt_x_plus_log_x_end_to_end() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let log_x = pool.func("log", vec![x]);
        let a = pool.add(vec![x, log_x]); // x + log x
        let y = pool.pow(a, pool.rational(1_i32, 2_i32)); // √(x+log x)
        let inv_y = pool.pow(a, pool.rational(-1_i32, 2_i32)); // 1/√(x+log x)

        // term1 = eˣ·√(x+log x)
        let term1 = pool.mul(vec![exp_x, y]);
        // term2 = eˣ·(1+1/x)/(2√(x+log x))
        let one_plus_inv_x = pool.add(vec![pool.integer(1_i32), pool.pow(x, pool.integer(-1_i32))]);
        let term2 = pool.mul(vec![
            exp_x,
            one_plus_inv_x,
            pool.rational(1_i32, 2_i32),
            inv_y,
        ]);
        let integrand = pool.add(vec![term1, term2]);

        // Direct path returns Some(Ok(...)).
        let direct = try_integrate_exp_times_radical_over_tower(integrand, x, &pool);
        assert!(
            matches!(direct, Some(Ok(_))),
            "headline should integrate via the multi-generator path; got {direct:?}"
        );

        // End-to-end through the public engine.
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(res.is_ok(), "headline should integrate; got {res:?}");
        let g = res.unwrap().value;

        // d/dx F = integrand, verified numerically.
        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.7_f64, 1.5, 2.6] {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(integrand, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }

        // F numerically equals eˣ·√(x+log x).
        let expected = pool.mul(vec![pool.func("exp", vec![x]), y]);
        for &xv in &[1.3_f64, 2.1, 3.4] {
            let fv = eval(g, x, xv, &pool).unwrap();
            let ev = eval(expected, x, xv, &pool).unwrap();
            assert!(
                (fv - ev).abs() < 1e-6 * (1.0 + ev.abs()),
                "x={xv}: F = {fv}, eˣ√(x+log x) = {ev}"
            );
        }
    }

    /// M4 PR3 new nesting case — radical over an *exp* tower (vs the headline's
    /// *log* tower), demonstrating the same tower-recursive descent now lives in
    /// the generic `RadicalExt::rational_rde`.
    ///
    /// ∫ [ eˣ·√(x+e^{x²}) + eˣ·(1+2x·e^{x²})/(2·√(x+e^{x²})) ] dx
    ///     = eˣ·√(x+e^{x²}).
    ///
    /// Outer transcendental coefficient `eˣ` multiplies a radical
    /// `y = √(x+e^{x²})` over a *separate, independent* exp tower `e^{x²}`.  The
    /// twisted Risch DE `D(w)+w = R` over `Quotient<ExpTowerField>` is solved
    /// per radical component by descending one tower level via
    /// `ExpTowerField::rational_rde` — through the generic `RadicalExt` impl.
    #[test]
    fn exp_times_sqrt_x_plus_exp_x_squared_end_to_end() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]); // e^{x²}
        let a = pool.add(vec![x, exp_x2]); // x + e^{x²}
        let y = pool.pow(a, pool.rational(1_i32, 2_i32)); // √(x+e^{x²})
        let inv_y = pool.pow(a, pool.rational(-1_i32, 2_i32));

        // term1 = eˣ·√(x+e^{x²})
        let term1 = pool.mul(vec![exp_x, y]);
        // a' = 1 + 2x·e^{x²};  term2 = eˣ·a'/(2√(x+e^{x²}))
        let a_prime = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(2_i32), x, exp_x2]),
        ]);
        let term2 = pool.mul(vec![exp_x, a_prime, pool.rational(1_i32, 2_i32), inv_y]);
        let integrand = pool.add(vec![term1, term2]);

        // Direct path integrates.
        let direct = try_integrate_exp_times_radical_over_tower(integrand, x, &pool);
        assert!(
            matches!(direct, Some(Ok(_))),
            "new exp-tower nesting case should integrate; got {direct:?}"
        );

        // End-to-end through the public engine.
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(res.is_ok(), "should integrate; got {res:?}");
        let g = res.unwrap().value;

        // d/dx F = integrand, verified numerically.
        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.4_f64, 0.8, 1.2] {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(integrand, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }
    }

    /// M4 capstone (target B): `∫∛(eˣ+1) dx` is **provably elementary** via the
    /// rationalizing substitution `u = ∛(eˣ+1)` and now integrates end-to-end
    /// through the public engine (was `NotImplemented` before the
    /// `radical_subst` hook).
    #[test]
    fn m4_capstone_cbrt_exp_plus_one() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let inner = pool.add(vec![exp_x, pool.integer(1_i32)]);
        let integrand = pool.pow(inner, pool.rational(1_i32, 3_i32));

        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(res.is_ok(), "∫∛(eˣ+1) dx must integrate; got {res:?}");
        let g = res.unwrap().value;

        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.35_f64, 0.7, 1.3, 2.1] {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(integrand, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }
    }

    /// M4 capstone (target B sibling): `∫ eˣ/∛(eˣ+1) dx` — radical in the
    /// denominator — also rationalizes (`∫ 3u du = (3/2)·∛(eˣ+1)²`).
    #[test]
    fn m4_capstone_exp_over_cbrt() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let inner = pool.add(vec![exp_x, pool.integer(1_i32)]);
        let inv_cbrt = pool.pow(inner, pool.rational(-1_i32, 3_i32));
        let integrand = pool.mul(vec![exp_x, inv_cbrt]);

        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(res.is_ok(), "∫ eˣ/∛(eˣ+1) dx must integrate; got {res:?}");
        let g = res.unwrap().value;

        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.35_f64, 0.7, 1.3, 2.1] {
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(integrand, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }
    }

    /// M4 capstone (target A): `∫ exp(x)/∛(x+log x) dx` — the radicand mixes two
    /// independent transcendentals (`x` and `log x`), so the rationalizing
    /// substitution does not apply and the `exp`/`log` interaction is genuinely
    /// unsupported.  The engine **declines** (no wrong elementary answer); this
    /// pins that honest decline.
    #[test]
    fn m4_capstone_exp_over_cbrt_x_plus_logx_declines() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let log_x = pool.func("log", vec![x]);
        let a = pool.add(vec![x, log_x]);
        let inv_cbrt = pool.pow(a, pool.rational(-1_i32, 3_i32));
        let integrand = pool.mul(vec![exp_x, inv_cbrt]);

        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            res.is_err(),
            "∫ exp(x)/∛(x+log x) dx is not reachable; expected decline, got {res:?}"
        );
    }

    /// The new multi-generator path declines (returns `None`) on the existing
    /// single-generator cases — it is purely additive.
    #[test]
    fn multigen_declines_single_generator_cases() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);

        // Example-15 style ∛(x+eˣ) integrand: no separate outer exp factor.
        let f15 = example15_integrand(&pool, x);
        assert!(
            try_integrate_exp_times_radical_over_tower(f15, x, &pool).is_none(),
            "multigen path must decline the single-generator exp radical case"
        );

        // Plain x·exp(x): no radical at all.
        let plain = pool.mul(vec![x, pool.func("exp", vec![x])]);
        assert!(
            try_integrate_exp_times_radical_over_tower(plain, x, &pool).is_none(),
            "multigen path must decline non-radical integrands"
        );
    }
}
