//! Algebraic-function Risch integration (V1-2).
//!
//! Extends the Risch integration engine to handle integrands containing
//! algebraic subterms: `sqrt(P(x))`, `P(x)^(p/q)`, and rational combinations
//! thereof over Q(x).
//!
//! Supports degree-2 algebraic extensions K = Q(x)\[y\]/(y² - P(x)):
//! - P of degree 0 (constant): trivial
//! - P of degree 1 (linear): complete elementary integration via substitution
//! - P of degree 2 (quadratic): genus-0 curve, always elementary
//! - P of degree ≥ 3: returns `NonElementary` (elliptic/hyperelliptic integrals)
//!
//! Higher-degree extensions (y^q = P, q > 2) return `UnsupportedExtensionDegree`.
//!
//! References:
//! - Trager (1984). Integration of algebraic functions. MIT PhD thesis.
//! - Bronstein (2005). Symbolic Integration I. Springer, chs. 10–11.

// Quadratic-tower field ℚ(α)[w]/(w²−c) + Galois automorphisms for the
// algebraic-base-point logarithmic part: `primitive_element` builds the field
// and `galois_quartic` its automorphisms, consumed by
// `genus_zero::try_alg_base_log` (the conjugate-divisor reduction).
mod alg_tower;
pub(super) mod decompose;
pub mod elliptic;
pub mod elliptic_output;
pub mod find_order;
pub mod genus1_log;
pub(super) mod genus_zero;
pub mod hermite_curve;
pub mod integral_basis;
mod jacobian_torsion;
pub(super) mod parametrize;
pub(super) mod poly_utils;
pub mod residues;
// Trager ℚ-basis logarithmic-part criterion: decomposition + per-component
// torsion over rational *and* algebraic places.  `trager_log_criterion_alg` is
// the engine consumer (via `genus_zero::integrate_b_sqrt_high_degree`); the
// rational-only `trager_log_criterion` is its specialization, kept for reuse and
// tests.
#[allow(dead_code)]
mod trager_log;
pub mod vanhoeij;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use decompose::decompose_sqrt;
use genus_zero::integrate_with_sqrt;
use poly_utils::is_zero_expr;

// ---------------------------------------------------------------------------
// Algebraic subterm detection
// ---------------------------------------------------------------------------

/// Returns `true` if `expr` contains any algebraic subterm (sqrt or fractional
/// power) that requires the algebraic integration path.
pub fn contains_algebraic_subterm(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } => {
            if name == "sqrt" {
                return true;
            }
            args.iter().any(|&a| contains_algebraic_subterm(a, pool))
        }
        ExprData::Pow { base, exp } => {
            // Any Rational exponent triggers algebraic detection
            if matches!(pool.get(exp), ExprData::Rational(_)) {
                return true;
            }
            // For integer exponent, recurse into the base only
            contains_algebraic_subterm(base, pool)
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_algebraic_subterm(a, pool))
        }
        _ => false,
    }
}

/// Returns `true` if `expr` contains a **radical function of `var`** that the
/// structural [`contains_algebraic_subterm`] misses — namely `cbrt(g(x))` (and
/// other named nth-root forms) where the radicand depends on `var`.
///
/// `contains_algebraic_subterm` recognizes only `sqrt` and `Pow` with a rational
/// exponent; a `cbrt` (or `nthroot`) *function* of a non-trivial argument needs
/// the algebraic engine too.  Constant radicands (`cbrt(3)`) are excluded so
/// they keep routing to the rule engine's constant rule (no regression).
pub fn contains_algebraic_func_of_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    use crate::integrate::risch::poly_rde::is_free_of_var;
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            if name == "cbrt" && !is_free_of_var(args[0], var, pool) {
                return true;
            }
            contains_algebraic_func_of_var(args[0], var, pool)
        }
        ExprData::Pow { base, .. } => contains_algebraic_func_of_var(base, var, pool),
        ExprData::Add(args) | ExprData::Mul(args) => args
            .iter()
            .any(|&a| contains_algebraic_func_of_var(a, var, pool)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Generator discovery
// ---------------------------------------------------------------------------

/// Walk `expr` and collect all algebraic generator IDs (sqrt or P^(1/2)).
fn collect_generators(expr: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } => {
            if name == "sqrt" && args.len() == 1 {
                out.push(expr);
                // Do not recurse into the argument — nested sqrt not supported
            } else {
                for &a in args.iter() {
                    collect_generators(a, pool, out);
                }
            }
        }
        ExprData::Pow { base, exp } => {
            if matches!(pool.get(exp), ExprData::Rational(_)) {
                out.push(expr);
            } else {
                collect_generators(base, pool, out);
                // exp is an integer, no need to recurse into it
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in args.iter() {
                collect_generators(a, pool, out);
            }
        }
        _ => {}
    }
}

/// Extract the radicand from a `sqrt(P)` or `P^(1/2)` expression.
fn get_radicand(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => Some(args[0]),
        ExprData::Pow { base, exp } => {
            // Check for Rational(1/2)
            match pool.get(exp) {
                ExprData::Rational(r) if r.0 == rug::Rational::from((1u32, 2u32)) => Some(base),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Find a unique algebraic generator in `expr`.
/// Returns `Some((sqrt_id, radicand_id))` when there is exactly one generator.
fn find_generator(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let mut generators = Vec::new();
    collect_generators(expr, pool, &mut generators);
    generators.sort_unstable();
    generators.dedup();
    if generators.len() != 1 {
        return None;
    }
    let sqrt_id = generators[0];
    let radicand = get_radicand(sqrt_id, pool)?;
    Some((sqrt_id, radicand))
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Symbolically integrate `expr` with respect to `var`, where `expr` contains
/// algebraic subterms (sqrt or fractional powers).
///
/// Precondition: `contains_algebraic_subterm(expr, pool)` is `true`.
pub fn integrate_algebraic(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    // M2: genus-0 reduction by rational parametrization.  A single radical with a
    // *linear* radicand `(a·x+b)^{1/n}` parametrizes as `x = (sⁿ−b)/a`, turning the
    // integrand rational in `s` (always elementary — incl. the logarithmic part the
    // simple-radical integral part below cannot finish).  Tried first so it fixes
    // those cases (and their previously wrong `NonElementary`).
    if let Some(res) = parametrize::try_parametrize_genus0(expr, var, pool) {
        return res;
    }

    // MA (Risch M0/M1): degree-≥3 simple radical `p(x)^{1/n}` over ℚ(x).  The
    // genus-0 sqrt engine below only covers degree 2; the simple-radical
    // integral part handles higher degrees (squarefree radicand).  Returns
    // `None` when not applicable, so degree-2 and unsupported cases fall through.
    if let Some(res) =
        crate::integrate::risch::simple_radical::try_integrate_simple_radical(expr, var, pool)
    {
        return res;
    }

    // Standard path: decompose `A(x) + B(x)·√P` and integrate each part.  When it
    // cannot express the integrand — e.g. a *rational* coefficient on a quadratic
    // radical, `∫ dx/((x²−1)√(x²+1))` — fall back to the genus-0 Euler
    // substitution, which rationalizes the whole `∫ R(x,√(quadratic)) dx`.  The
    // decompose path is tried first so polynomial-coefficient cases keep their
    // nicer closed forms.
    match integrate_via_decompose(expr, var, pool) {
        Err(IntegrationError::NotImplemented(_)) => {
            if let Some(res) = parametrize::try_euler_quadratic(expr, var, pool) {
                return res;
            }
            integrate_via_decompose(expr, var, pool)
        }
        other => other,
    }
}

/// The standard algebraic path: find the single `√P` generator, decompose the
/// integrand as `A(x) + B(x)·√P`, and integrate each part (with the genus-1
/// capstone for `deg P ≥ 3`).
fn integrate_via_decompose(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    let mut log = DerivationLog::new();

    // Step 1: Find the unique sqrt generator y = sqrt(P(x))
    let (sqrt_id, p_expr) = find_generator(expr, pool).ok_or_else(|| {
        IntegrationError::NotImplemented(
            "algebraic integrator requires exactly one sqrt(P(x)) generator; \
             multiple or nested generators are not supported in v1.1"
                .to_string(),
        )
    })?;

    // Step 2: Validate extension degree (only degree-2 extensions in v1.1)
    // Higher-degree generators like P^(1/3) are rejected here.
    if let ExprData::Pow { exp, .. } = pool.get(sqrt_id) {
        if let ExprData::Rational(r) = pool.get(exp) {
            let q = r.0.denom().to_u32().unwrap_or(0);
            if q != 2 {
                return Err(IntegrationError::UnsupportedExtensionDegree(q));
            }
        }
    }

    // Step 3: Decompose integrand as A(x) + B(x)·sqrt(P)
    let (a_raw, b_raw) = decompose_sqrt(expr, sqrt_id, p_expr, pool).ok_or_else(|| {
        IntegrationError::NotImplemented(
            "could not decompose integrand into A(x) + B(x)·sqrt(P(x)); \
             expression structure is not supported"
                .to_string(),
        )
    })?;

    // Simplify both parts so that subsequent pattern matching works on canonical forms.
    // (e.g. field inversion produces Mul(-1, 1, Pow(Mul(-1, P), -1)) which simplifies to P^-1)
    let a_part = simplify(a_raw, pool).value;
    let b_part = simplify(b_raw, pool).value;

    // MC1 (genus-1): for `y² = P(x)` with `P` a cubic, the rational part `A(x)` and
    // the algebraic part `B(x)·y` couple into a *single* log (e.g.
    // `∫[1/(2x) + √(x³+1)/(2x(x³+1))] dx = ⅓·log(√(x³+1)−1)`), so the combined
    // integrand must go through the genus-1 capstone rather than integrating `A`
    // and `B·y` separately.  `integrate_genus1_log` self-guards on genus/degree and
    // verifies `d/dx F = integrand`, so a non-elementary integral (e.g.
    // `∫dx/√(x³+1)`) returns `None` and falls through to the `NonElementary` path
    // below; degree-2 (genus-0) curves also return `None` here, preserving the
    // existing genus-0 engine.
    if !is_zero_expr(b_part, pool) {
        if let Some(f) = try_genus1_log(a_part, b_part, p_expr, var, pool) {
            let simplified = simplify(f, pool);
            log = log.merge(simplified.log);
            log.push(RewriteStep::simple(
                "genus1_elliptic_log",
                expr,
                simplified.value,
            ));
            return Ok(DerivedExpr::with_log(simplified.value, log));
        }

        // PR2: first-kind elliptic-integral *output*.  When the integrand is the
        // pure first-kind shape `∫ c·dx/√P` with `P` a genus-1 cubic/quartic that
        // the genus-1 log capstone above proved non-elementary, emit the Legendre
        // normal-form `c·g·EllipticF(φ(x), m)`.  Gated by numeric `d/dx F =
        // integrand` verification inside `try_elliptic_output`, so it returns
        // `None` (falling through to the `NonElementary` path) unless the closed
        // form is verified — never a wrong answer.
        if let Some(f) = elliptic_output::try_elliptic_output(a_part, b_part, p_expr, var, pool) {
            let simplified = simplify(f, pool);
            log = log.merge(simplified.log);
            log.push(RewriteStep::simple(
                "genus1_elliptic_firstkind_output",
                expr,
                simplified.value,
            ));
            return Ok(DerivedExpr::with_log(simplified.value, log));
        }
    }

    let zero = pool.integer(0_i32);

    // Step 4: Integrate the rational part ∫ A(x) dx
    let int_a = if is_zero_expr(a_part, pool) {
        zero
    } else {
        crate::integrate::engine::integrate_raw(a_part, var, pool, &mut log)?
    };

    // Step 5: Integrate the algebraic part ∫ B(x)·sqrt(P) dx
    let int_b = if is_zero_expr(b_part, pool) {
        zero
    } else {
        integrate_with_sqrt(b_part, p_expr, sqrt_id, var, pool, &mut log)?
    };

    // Step 6: Combine
    let raw = match (is_zero_expr(int_a, pool), is_zero_expr(int_b, pool)) {
        (true, true) => zero,
        (false, true) => int_a,
        (true, false) => int_b,
        (false, false) => pool.add(vec![int_a, int_b]),
    };

    // Step 7: Simplify and record derivation
    let simplified = simplify(raw, pool);
    log = log.merge(simplified.log);
    log.push(RewriteStep::simple(
        "algebraic_risch",
        expr,
        simplified.value,
    ));

    Ok(DerivedExpr::with_log(simplified.value, log))
}

/// Attempt the genus-1 (MC1) capstone for an integrand `A(x) + B(x)·√P` over the
/// curve `y² = P(x)`.  Parses `P` to a [`QPoly`] and `A`, `B` to rational
/// functions over ℚ(x), assembles the combined integrand as an `AlgElem`
/// `[A, B]` (`y = √P`), and calls [`genus1_log::integrate_genus1_log`], which
/// returns the antiderivative only when it is elementary (cubic `P`, torsion
/// residue divisor, `d/dx F = integrand` verified) — otherwise `None`.
fn try_genus1_log(
    a_part: ExprId,
    b_part: ExprId,
    p_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    use crate::integrate::risch::alg_field::RatFn;
    use crate::integrate::risch::poly_rde::expr_to_qpoly;
    use crate::integrate::risch::rational_rde::expr_to_qrational;

    let p_poly = expr_to_qpoly(p_expr, var, pool)?;
    let a_rat = if is_zero_expr(a_part, pool) {
        RatFn::int(0)
    } else {
        let (num, den) = expr_to_qrational(a_part, var, pool)?;
        RatFn::new(num, den)
    };
    let (b_num, b_den) = expr_to_qrational(b_part, var, pool)?;
    let integrand = vec![a_rat, RatFn::new(b_num, b_den)];
    genus1_log::integrate_genus1_log(&p_poly, &integrand, var, pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// Numeric eval on the principal real branch (`sqrt`/`log` real).
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
            ExprData::Pow { base, exp } => {
                Some(eval(base, x, xv, pool)?.powf(eval(exp, x, xv, pool)?))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let v = eval(args[0], x, xv, pool)?;
                match name.as_str() {
                    "sqrt" => Some(v.sqrt()),
                    "log" => Some(v.ln()),
                    "exp" => Some(v.exp()),
                    "cbrt" => Some(v.cbrt()),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// `∫ [1/(2x) + √(x³+1)/(2x(x³+1))] dx = ⅓·log(√(x³+1) − 1)`, end-to-end
    /// through the **public engine** (genus-1, `y² = x³+1`).
    #[test]
    fn genus1_elliptic_log_via_engine() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x3p1 = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let sq = pool.func("sqrt", vec![x3p1]);
        let half = pool.pow(pool.integer(2_i32), pool.integer(-1_i32));
        // A = 1/(2x)
        let a_part = pool.mul(vec![half, pool.pow(x, pool.integer(-1_i32))]);
        // B·√P = √(x³+1) / (2x(x³+1))
        let denom = pool.mul(vec![pool.integer(2_i32), x, x3p1]);
        let b_term = pool.mul(vec![sq, pool.pow(denom, pool.integer(-1_i32))]);
        let integrand = pool.add(vec![a_part, b_term]);

        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("genus-1 integrand should integrate elementarily");
        let f = res.value;

        // d/dx F = integrand at sample points where x³+1 > 0.
        let df = simplify(crate::diff::diff(f, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for &xv in &[0.7_f64, 1.5, 2.9] {
            let lhs = eval(df, x, xv, &pool).unwrap();
            let rhs = eval(integrand, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {}",
                pool.display(f)
            );
            checked += 1;
        }
        assert!(checked >= 2);
    }

    /// `∫ 5x⁴·√(x⁵+1) dx = ⅔(x⁵+1)^{3/2}` — a **genus-2** (deg P = 5) integrand
    /// that is nonetheless elementary (the polynomial-`B` integral part).  This
    /// used to be wrongly reported `NonElementary`; the integral-part solver now
    /// returns `Q·√P`.  Verified by `d/dx F = integrand`.
    #[test]
    fn quintic_poly_b_integral_part_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x5p1 = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let sq = pool.func("sqrt", vec![x5p1]);
        let integrand = pool.mul(vec![
            pool.integer(5_i32),
            pool.pow(x, pool.integer(4_i32)),
            sq,
        ]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("polynomial-B integral part is elementary");
        let df = simplify(crate::diff::diff(res.value, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.3_f64, 0.8, 1.4] {
            let lhs = eval(df, x, xv, &pool).unwrap();
            let rhs = eval(integrand, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}"
            );
        }
    }

    /// `∫ x·√(x⁵+1) dx` is genuinely non-elementary (the integral-part ansatz has
    /// no polynomial solution ⇒ a residual `∫dx/√(x⁵+1)`), and `x⁵+1` is
    /// squarefree — so the engine soundly reports `NonElementary`.
    #[test]
    fn quintic_poly_b_genuinely_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x5p1 = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let sq = pool.func("sqrt", vec![x5p1]);
        let integrand = pool.mul(vec![x, sq]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// **Rational** weight `B` on a genus-2 curve: `∫ 5x⁴/(2√(x⁵+1)) dx =
    /// √(x⁵+1)` — the integral part `b·√P` (`b=1`) found via the rational Risch
    /// DE `b' + (P'/2P)b = B`.  Previously hit the blind `NonElementary` shortcut.
    #[test]
    fn quintic_rational_b_integral_part_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x5p1 = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let inv_sqrt = pool.pow(pool.func("sqrt", vec![x5p1]), pool.integer(-1_i32));
        let half = pool.pow(pool.integer(2_i32), pool.integer(-1_i32));
        let integrand = pool.mul(vec![
            pool.integer(5_i32),
            pool.pow(x, pool.integer(4_i32)),
            half,
            inv_sqrt,
        ]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("rational-B integral part is elementary");
        let df = simplify(crate::diff::diff(res.value, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.4_f64, 0.9, 1.6] {
            let lhs = eval(df, x, xv, &pool).unwrap();
            let rhs = eval(integrand, x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}"
            );
        }
    }

    /// `∫ dx/√(x⁵+1)` — genus-2 first-kind hyperelliptic: no logarithmic part
    /// (empty residue divisor) and no algebraic primitive (Risch DE unsolvable),
    /// so soundly `NonElementary`.
    #[test]
    fn quintic_first_kind_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x5p1 = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let integrand = pool.pow(pool.func("sqrt", vec![x5p1]), pool.integer(-1_i32));
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// `∫ √(x⁵+x+1)/(x−2) dx` — a **genus-2** integrand with an algebraic-sheet
    /// pole (rational base `x=2`, sheet `√a(2)=√35` irrational).  The end-to-end
    /// consumer collects the residues into `ℚ(√35)` and the Trager ℚ-basis
    /// criterion finds the `√35`-component `2[(2,√35)−∞]` non-torsion ⇒
    /// `NonElementary`.  **Oracle-confirmed:** FriCAS 1.3.7 returns this integral
    /// unevaluated (its complete Trager implementation ⇒ proven non-elementary).
    #[test]
    fn quintic_algebraic_pole_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(5_i32)),
            x,
            pool.integer(1_i32),
        ]);
        let sq = pool.func("sqrt", vec![p]);
        let integrand = pool.mul(vec![
            sq,
            pool.pow(
                pool.add(vec![x, pool.integer(-2_i32)]),
                pool.integer(-1_i32),
            ),
        ]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// **Compositum** of quadratic sheet fields: `∫ √(x⁵+x+1)/((x−2)(x−3)) dx`
    /// has algebraic-sheet poles at `x=2` (`√35`) and `x=3` (`√247`) — distinct
    /// fields.  The Trager components separate (a rational `1`-component + one
    /// `√d_i`-component each); the `√35`-component `2[(2,√35)−∞]` is non-torsion ⇒
    /// `NonElementary`.  **FriCAS-confirmed** (returns it unevaluated).
    #[test]
    fn quintic_compositum_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(5_i32)),
            x,
            pool.integer(1_i32),
        ]);
        let sq = pool.func("sqrt", vec![p]);
        let den = pool.mul(vec![
            pool.add(vec![x, pool.integer(-2_i32)]),
            pool.add(vec![x, pool.integer(-3_i32)]),
        ]);
        let integrand = pool.mul(vec![sq, pool.pow(den, pool.integer(-1_i32))]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// `∫ x³·√(x⁵+x+1)/(x−2) dx` — likewise non-elementary (algebraic-sheet pole,
    /// non-torsion component); FriCAS-confirmed.
    #[test]
    fn quintic_algebraic_pole_weighted_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(5_i32)),
            x,
            pool.integer(1_i32),
        ]);
        let sq = pool.func("sqrt", vec![p]);
        let integrand = pool.mul(vec![
            pool.pow(x, pool.integer(3_i32)),
            sq,
            pool.pow(
                pool.add(vec![x, pool.integer(-2_i32)]),
                pool.integer(-1_i32),
            ),
        ]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// **Algebraic base point** (degree-4 Galois tower): `∫ √(x⁵−4x+3)/(x²−2) dx`
    /// has a pole at the irrational base `x=±√2`; since `a(√2)=3`, the residue
    /// field is `ℚ(√2,√3)` (Galois).  The conjugate-divisor reduction builds the
    /// four automorphisms, decomposes the conjugate residues over ℚ, and finds a
    /// non-torsion component ⇒ `NonElementary`.  **FriCAS-confirmed**.
    #[test]
    fn quintic_algebraic_base_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(5_i32)),
            pool.mul(vec![pool.integer(-4_i32), x]),
            pool.integer(3_i32),
        ]);
        let sq = pool.func("sqrt", vec![p]);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-2_i32)]); // x²−2
        let integrand = pool.mul(vec![sq, pool.pow(den, pool.integer(-1_i32))]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// A **non-Galois** algebraic base: `∫ √(x⁵+x+1)/(x²−2) dx` has a pole at the
    /// irrational base `x=±√2` with `a(√2)=1+5√2`, so `N(c)=(1+5√2)(1−5√2)=−49`
    /// and the residue field `ℚ(√2,√(1+5√2))` is *not* Galois — its closure
    /// `L=K(√(−49))=K(7i)` has degree 8.  The conjugate-divisor reduction builds
    /// `L` (each defining relation `α²=2, w²=c, v²=c̄` verified in `L`), forms the
    /// four orbit places/residues, decomposes over ℚ, and finds a non-torsion
    /// component ⇒ `NonElementary`.  (FriCAS times out on this degree-8 problem; the
    /// verdict rests on the verified-by-construction closure + Trager torsion test.)
    #[test]
    fn quintic_algebraic_base_non_galois_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(5_i32)),
            x,
            pool.integer(1_i32),
        ]);
        let sq = pool.func("sqrt", vec![p]);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-2_i32)]);
        let integrand = pool.mul(vec![sq, pool.pow(den, pool.integer(-1_i32))]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// **General quadratic base** `x²+b·x+c₀` (Galois): the `x→x+1` translate of
    /// `∫√(x⁵−4x+3)/(x²−2)` is `∫√(x⁵+5x⁴+10x³+10x²+x)/(x²+2x−1) dx` — the base is
    /// now `x²+2x−1` (`b=2≠0`).  Completing the square (`α=β−1`, `m=b²/4−c₀=2`)
    /// reduces it to the depressed `x²−2`, recovering the same Galois residue field
    /// `ℚ(√2,√3)`.  A translation cannot change elementarity, so this must stay
    /// `NonElementary`.
    #[test]
    fn quintic_general_quadratic_base_galois_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // x⁵ + 5x⁴ + 10x³ + 10x² + x  (= (x+1)⁵ − 4(x+1) + 3).
        let p = pool.add(vec![
            pool.pow(x, pool.integer(5_i32)),
            pool.mul(vec![pool.integer(5_i32), pool.pow(x, pool.integer(4_i32))]),
            pool.mul(vec![pool.integer(10_i32), pool.pow(x, pool.integer(3_i32))]),
            pool.mul(vec![pool.integer(10_i32), pool.pow(x, pool.integer(2_i32))]),
            x,
        ]);
        let sq = pool.func("sqrt", vec![p]);
        // x² + 2x − 1  (= (x+1)² − 2).
        let den = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(-1_i32),
        ]);
        let integrand = pool.mul(vec![sq, pool.pow(den, pool.integer(-1_i32))]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// **General quadratic base** (non-Galois): the `x→x+1` translate of
    /// `∫√(x⁵+x+1)/(x²−2)` is `∫√(x⁵+5x⁴+10x³+10x²+6x+3)/(x²+2x−1) dx` — base
    /// `x²+2x−1` (`b=2`), depressing to `x²−2` with the same non-Galois closure
    /// `K(7i)`.  Must remain `NonElementary` (a pure translation).
    #[test]
    fn quintic_general_quadratic_base_non_galois_non_elementary() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // x⁵ + 5x⁴ + 10x³ + 10x² + 6x + 3  (= (x+1)⁵ + (x+1) + 1).
        let p = pool.add(vec![
            pool.pow(x, pool.integer(5_i32)),
            pool.mul(vec![pool.integer(5_i32), pool.pow(x, pool.integer(4_i32))]),
            pool.mul(vec![pool.integer(10_i32), pool.pow(x, pool.integer(3_i32))]),
            pool.mul(vec![pool.integer(10_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.mul(vec![pool.integer(6_i32), x]),
            pool.integer(3_i32),
        ]);
        let sq = pool.func("sqrt", vec![p]);
        let den = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(-1_i32),
        ]);
        let integrand = pool.mul(vec![sq, pool.pow(den, pool.integer(-1_i32))]);
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "got {res:?}"
        );
    }

    /// `∫ dx/√(x³+1)` is a first-kind elliptic integral.  **PR2**: the genus-1
    /// log capstone proves it non-elementary, then the first-kind elliptic-output
    /// reduction (`elliptic_output::try_elliptic_output`) emits the Legendre
    /// normal form `c·EllipticF(φ(x), m)`.  We assert the result contains
    /// `EllipticF` and that its symbolic `d/dx` numerically matches the integrand
    /// `1/√(x³+1)` where `x³+1 > 0`.  (Previously this test asserted
    /// `NonElementary`; PR1 added the primitive, PR2 wires the output.)
    #[test]
    fn genus1_first_kind_emits_ellipticf_via_engine() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x3p1 = pool.add(vec![pool.pow(x, pool.integer(3_i32)), pool.integer(1_i32)]);
        let sq = pool.func("sqrt", vec![x3p1]);
        let integrand = pool.pow(sq, pool.integer(-1_i32));
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("∫dx/√(x³+1) should emit a first-kind EllipticF form");
        let f = res.value;
        assert!(
            pool.display(f).to_string().contains("EllipticF"),
            "expected EllipticF in {}",
            pool.display(f)
        );
        let df = simplify(crate::diff::diff(f, x, &pool).unwrap().value, &pool).value;
        let mut checked = 0;
        for &xv in &[0.5_f64, 1.0, 2.0, 3.0] {
            // x³+1 > 0 for these.
            let lhs = eval_ell(df, x, xv, &pool).unwrap();
            let rhs = 1.0 / (xv * xv * xv + 1.0).sqrt();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {}",
                pool.display(f)
            );
            checked += 1;
        }
        assert!(checked >= 3);
    }

    /// `∫ dx/√(x³−x)` (three real roots) — also reduces to a first-kind
    /// `EllipticF` form, verified by `d/dx F = integrand` on `x > 1` (`x³−x > 0`).
    #[test]
    fn genus1_three_real_emits_ellipticf_via_engine() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let sq = pool.func("sqrt", vec![p]);
        let integrand = pool.pow(sq, pool.integer(-1_i32));
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("∫dx/√(x³−x) should emit EllipticF");
        let f = res.value;
        assert!(
            pool.display(f).to_string().contains("EllipticF"),
            "{}",
            pool.display(f)
        );
        let df = simplify(crate::diff::diff(f, x, &pool).unwrap().value, &pool).value;
        for &xv in &[1.2_f64, 2.0, 4.0] {
            let lhs = eval_ell(df, x, xv, &pool).unwrap();
            let rhs = 1.0 / (xv * xv * xv - xv).sqrt();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}"
            );
        }
    }

    /// `∫ dx/√(1−x⁴)` (two real roots + complex pair) — first-kind quartic
    /// reduction, verified on `|x| < 1` (`1−x⁴ > 0`).
    #[test]
    fn genus1_quartic_emits_ellipticf_via_engine() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 1 − x⁴.
        let p = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(4_i32))]),
        ]);
        let sq = pool.func("sqrt", vec![p]);
        let integrand = pool.pow(sq, pool.integer(-1_i32));
        let res = crate::integrate::engine::integrate(integrand, x, &pool)
            .expect("∫dx/√(1−x⁴) should emit EllipticF");
        let f = res.value;
        assert!(
            pool.display(f).to_string().contains("EllipticF"),
            "{}",
            pool.display(f)
        );
        let df = simplify(crate::diff::diff(f, x, &pool).unwrap().value, &pool).value;
        for &xv in &[-0.8_f64, -0.2, 0.3, 0.8] {
            let lhs = eval_ell(df, x, xv, &pool).unwrap();
            let rhs = 1.0 / (1.0 - xv.powi(4)).sqrt();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}"
            );
        }
    }

    /// `∫ dx/√(x⁵+1)` is genus-2 (quintic): no first-kind cubic/quartic
    /// reduction applies, so the engine still soundly reports `NonElementary`.
    #[test]
    fn quintic_first_kind_still_non_elementary_via_engine() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.add(vec![pool.pow(x, pool.integer(5_i32)), pool.integer(1_i32)]);
        let sq = pool.func("sqrt", vec![p]);
        let integrand = pool.pow(sq, pool.integer(-1_i32));
        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "∫dx/√(x⁵+1) must stay NonElementary, got {res:?}"
        );
    }

    /// Numeric eval that also handles `sin`/`cos`/`asin`/`acos` so the symbolic
    /// `d/dx` of an `EllipticF` form (which rewrites to elementary terms) can be
    /// sampled.  Extends the `sqrt`/`log`/`exp`/`cbrt`-only `eval` above.
    fn eval_ell(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> Option<f64> {
        if expr == x {
            return Some(xv);
        }
        match pool.get(expr) {
            ExprData::Integer(n) => Some(n.0.to_f64()),
            ExprData::Rational(r) => Some(r.0.to_f64()),
            ExprData::Add(args) => args
                .iter()
                .try_fold(0.0, |s, &a| Some(s + eval_ell(a, x, xv, pool)?)),
            ExprData::Mul(args) => args
                .iter()
                .try_fold(1.0, |s, &a| Some(s * eval_ell(a, x, xv, pool)?)),
            ExprData::Pow { base, exp } => {
                Some(eval_ell(base, x, xv, pool)?.powf(eval_ell(exp, x, xv, pool)?))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let v = eval_ell(args[0], x, xv, pool)?;
                match name.as_str() {
                    "sin" => Some(v.sin()),
                    "cos" => Some(v.cos()),
                    "tan" => Some(v.tan()),
                    "asin" => Some(v.asin()),
                    "acos" => Some(v.acos()),
                    "atan" => Some(v.atan()),
                    "sqrt" => Some(v.sqrt()),
                    "log" => Some(v.ln()),
                    "exp" => Some(v.exp()),
                    "cbrt" => Some(v.cbrt()),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
