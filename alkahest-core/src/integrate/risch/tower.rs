//! Differential field tower representation and detection for the Risch algorithm.
//!
//! Models a tower of transcendental extensions over ℚ(x):
//!   K = ℚ(x)(t₁, t₂, …, tₙ)
//! where each tᵢ is either:
//!   - **Hyperexponential**: tᵢ = exp(ηᵢ), D(tᵢ) = D(ηᵢ) · tᵢ
//!   - **Hyperlogarithmic**:  tᵢ = log(hᵢ), D(tᵢ) = D(hᵢ)/hᵢ
//!
//! For the transcendental Risch algorithm, we build the tower bottom-up from a
//! symbolic expression and classify each generator.
//!
//! References: Bronstein (2005), §4.1–4.3.

use super::poly_rde::is_free_of_var;
use crate::kernel::{ExprData, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Tower level types
// ---------------------------------------------------------------------------

/// The kind of a transcendental extension generator.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtensionKind {
    /// tᵢ = exp(η) with D(t) = D(η)·t (hyperexponential).
    Exp { eta: ExprId },
    /// tᵢ = log(h) with D(t) = D(h)/h (hyperlogarithmic).
    Log { h: ExprId },
}

/// A single level in the differential field tower.
#[derive(Debug, Clone)]
pub struct TowerLevel {
    /// ExprId of the generator expression itself (e.g., `exp(x^2)`).
    pub generator: ExprId,
    /// Kind and inner argument.
    pub kind: ExtensionKind,
}

impl TowerLevel {
    /// The inner argument (η for Exp, h for Log).
    pub fn argument(&self) -> ExprId {
        match self.kind {
            ExtensionKind::Exp { eta } => eta,
            ExtensionKind::Log { h } => h,
        }
    }

    /// Returns true if this is an Exp extension.
    pub fn is_exp(&self) -> bool {
        matches!(self.kind, ExtensionKind::Exp { .. })
    }

    /// Returns true if this is a Log extension.
    pub fn is_log(&self) -> bool {
        matches!(self.kind, ExtensionKind::Log { .. })
    }
}

// ---------------------------------------------------------------------------
// Input-form normalisation helpers
// ---------------------------------------------------------------------------
//
// `a/b`, `a·b⁻¹` and `(a·b)⁻¹` denote the same function, but they are three
// different trees, and an exponent of `-1` does not always arrive as the
// literal `-1`: prefix negation is `(-1) · operand`, so `(-1)` built through
// the pool API or Python's `Expr.__neg__` is the unevaluated product `(1 · -1)`.
// (The parsers now fold that case — see `negate_literal` in `crate::parse` —
// but the builder API is public and every non-parser caller can still hand us
// the product.)  Every structural detector and matcher below keys on tree
// shape, so without a normalising view the *spelling* of an integrand decides
// its route through the integrator.  These two helpers give the detectors a
// spelling-independent reading of "integer exponent" and "power of a product".

/// The value of a **var-free integer-valued exponent**, folding the small
/// arithmetic a caller can leave behind (`pool.mul([1, -1])` for `-1`).
///
/// Only `Integer`, `Add`, `Mul` and non-negative integer `Pow` nodes are
/// folded — no simplifier is invoked, so this is cheap enough to call from a
/// detector on every node.  Returns `None` for anything that is not a literal
/// integer under that folding (including symbols, rationals, and `1/2`-style
/// exponents, which must *not* be distributed over a product).
pub fn literal_integer(expr: ExprId, pool: &ExprPool) -> Option<i64> {
    fn go(expr: ExprId, pool: &ExprPool, depth: u32) -> Option<i64> {
        if depth == 0 {
            return None;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_i64(),
            ExprData::Add(args) => args
                .iter()
                .try_fold(0_i64, |acc, &a| acc.checked_add(go(a, pool, depth - 1)?)),
            ExprData::Mul(args) => args
                .iter()
                .try_fold(1_i64, |acc, &a| acc.checked_mul(go(a, pool, depth - 1)?)),
            ExprData::Pow { base, exp } => {
                let b = go(base, pool, depth - 1)?;
                let e = go(exp, pool, depth - 1)?;
                let e = u32::try_from(e).ok()?;
                if e > 32 {
                    return None;
                }
                b.checked_pow(e)
            }
            _ => None,
        }
    }
    go(expr, pool, 8)
}

/// Build `base^n`, folding a nested integer power (`(a^m)^n → a^{m·n}`) so the
/// result is in the same shape a directly-written `a^{mn}` would have.
pub fn pow_integer(base: ExprId, n: i64, pool: &ExprPool) -> ExprId {
    if n == 1 {
        return base;
    }
    if let ExprData::Pow {
        base: inner,
        exp: inner_exp,
    } = pool.get(base)
    {
        if let Some(m) = literal_integer(inner_exp, pool) {
            if let Some(mn) = m.checked_mul(n) {
                if mn == 1 {
                    return inner;
                }
                return pool.pow(inner, pool.integer(mn));
            }
        }
    }
    if n == 0 {
        return pool.integer(1_i32);
    }
    pool.pow(base, pool.integer(n))
}

/// If `expr` is `(f₁ · f₂ · …)^n` with a **literal integer** `n`, return the
/// distributed factors `[f₁^n, f₂^n, …]`.
///
/// `(a·b)^n = a^n·b^n` holds for every integer `n` over a field, so this is a
/// pure re-spelling and not a domain-widening rewrite — which is exactly why
/// the exponent must be an integer (`(a·b)^(1/2) ≠ a^(1/2)·b^(1/2)` on the
/// negative reals).  Returns `None` when `expr` is not of that shape.
pub fn distribute_integer_pow_over_mul(expr: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let ExprData::Pow { base, exp } = pool.get(expr) else {
        return None;
    };
    let n = literal_integer(exp, pool)?;
    let ExprData::Mul(factors) = pool.get(base) else {
        return None;
    };
    Some(
        factors
            .iter()
            .map(|&f| pow_integer(f, n, pool))
            .collect::<Vec<_>>(),
    )
}

// ---------------------------------------------------------------------------
// Generator discovery
// ---------------------------------------------------------------------------

/// Walk `expr` and collect all distinct transcendental generators (exp/log).
///
/// Each unique generator appears exactly once in the output list, in the order
/// first encountered during a depth-first traversal.
pub fn find_generators(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<TowerLevel> {
    let mut generators = Vec::new();
    collect_generators(expr, var, pool, &mut generators);
    generators
}

fn collect_generators(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<TowerLevel>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let arg = args[0];
            match name.as_str() {
                "exp" => {
                    if !is_free_of_var(arg, var, pool) {
                        let level = TowerLevel {
                            generator: expr,
                            kind: ExtensionKind::Exp { eta: arg },
                        };
                        if !out.iter().any(|l| l.generator == expr) {
                            out.push(level);
                        }
                    }
                    // Recurse into the exponent for nested structure
                    collect_generators(arg, var, pool, out);
                }
                "log" => {
                    if !is_free_of_var(arg, var, pool) {
                        let level = TowerLevel {
                            generator: expr,
                            kind: ExtensionKind::Log { h: arg },
                        };
                        if !out.iter().any(|l| l.generator == expr) {
                            out.push(level);
                        }
                    }
                    collect_generators(arg, var, pool, out);
                }
                _ => {
                    for &a in args.iter() {
                        collect_generators(a, var, pool, out);
                    }
                }
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in args.iter() {
                collect_generators(a, var, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_generators(base, var, pool, out);
            collect_generators(exp, var, pool, out);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Expression decomposition for the exp tower
// ---------------------------------------------------------------------------

/// Try to decompose `expr` as `coefficient(x) * exp_gen^power`, where:
///   - `exp_gen` is an `exp(η)` ExprId
///   - `power` is an integer (the monomial power in the exp tower)
///   - `coefficient` is what remains after factoring out the exp_gen
///
/// Returns `None` if the expression cannot be so decomposed.
pub fn extract_exp_factor(expr: ExprId, exp_gen: ExprId, pool: &ExprPool) -> Option<(ExprId, i64)> {
    match pool.get(expr) {
        // expr IS the generator itself: coefficient = 1, power = 1
        _ if expr == exp_gen => Some((pool.integer(1_i32), 1)),

        ExprData::Mul(args) => {
            // Separate exp_gen factors from the rest.
            let mut exp_power: i64 = 0;
            let mut rest: Vec<ExprId> = Vec::new();

            for &a in &args {
                if a == exp_gen {
                    exp_power += 1;
                } else if let ExprData::Pow { base, exp } = pool.get(a) {
                    // The exponent is folded (a hand-built `-1` can arrive as
                    // `1 · -1`), so `/` and `^(-n)` decompose identically.
                    match (base == exp_gen)
                        .then(|| literal_integer(exp, pool))
                        .flatten()
                    {
                        Some(n) => exp_power += n,
                        None => rest.push(a),
                    }
                } else {
                    rest.push(a);
                }
            }

            if exp_power == 0 {
                return None; // No exp factor found.
            }

            let coeff = match rest.len() {
                0 => pool.integer(1_i32),
                1 => rest[0],
                _ => pool.mul(rest),
            };
            Some((coeff, exp_power))
        }

        ExprData::Pow { base, exp } => {
            if base == exp_gen {
                // exp_gen^n: coefficient = 1, power = n
                if let Some(n_i) = literal_integer(exp, pool) {
                    return Some((pool.integer(1_i32), n_i));
                }
            }
            // `(c · exp(η)^m)^n = c^n · exp(η)^{m·n}` — an integer power of a
            // product is the same monomial, just spelled differently.
            if let Some(n) = literal_integer(exp, pool) {
                if matches!(pool.get(base), ExprData::Mul(_) | ExprData::Pow { .. }) {
                    let (inner_coeff, m) = extract_exp_factor(base, exp_gen, pool)?;
                    return Some((pow_integer(inner_coeff, n, pool), m.checked_mul(n)?));
                }
            }
            None
        }

        _ => None,
    }
}

/// Decompose `expr` into its components relative to an exp generator.
///
/// Returns `(rational_part, exp_monomials)` where:
/// - `rational_part` is the sum of all terms in `expr` NOT involving `exp_gen`
/// - `exp_monomials` is a list of `(coefficient, k)` pairs meaning `coefficient * exp_gen^k`
///
/// Supports `expr` that is a sum of such terms.
pub fn decompose_wrt_exp(
    expr: ExprId,
    exp_gen: ExprId,
    pool: &ExprPool,
) -> (ExprId, Vec<(ExprId, i64)>) {
    use std::collections::BTreeMap;

    let zero = pool.integer(0_i32);

    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut rational_terms: Vec<ExprId> = Vec::new();
            // Combine coefficients of equal powers of `exp_gen`.  Without this,
            // ∫ (c₁·θ + c₀)·exp(η) written as a sum of products
            // (c₁·θ·exp(η) + c₀·exp(η)) is integrated term-by-term and each
            // summand can be mis-certified NonElementary even when the combined
            // coefficient admits a poly-in-log RDE solution (B2).
            let mut by_k: BTreeMap<i64, Vec<ExprId>> = BTreeMap::new();

            for &a in &args {
                if let Some((coeff, k)) = extract_exp_factor(a, exp_gen, pool) {
                    by_k.entry(k).or_default().push(coeff);
                } else {
                    rational_terms.push(a);
                }
            }

            let rational_part = match rational_terms.len() {
                0 => zero,
                1 => rational_terms[0],
                _ => pool.add(rational_terms),
            };
            let exp_terms: Vec<(ExprId, i64)> = by_k
                .into_iter()
                .map(|(k, coeffs)| {
                    let coeff = match coeffs.len() {
                        1 => coeffs[0],
                        _ => pool.add(coeffs),
                    };
                    (coeff, k)
                })
                .collect();
            (rational_part, exp_terms)
        }

        _ => {
            // Single term.
            if let Some((coeff, k)) = extract_exp_factor(expr, exp_gen, pool) {
                (zero, vec![(coeff, k)])
            } else {
                (expr, vec![])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Expression decomposition for the log tower
// ---------------------------------------------------------------------------

thread_local! {
    /// A [`crate::budget`] refusal raised inside [`decompose_log_inner`],
    /// whose `Option` return cannot carry one.
    ///
    /// Cleared at the top of every [`decompose_as_log_poly`] and taken by
    /// [`take_decompose_budget_trip`], so it can only ever describe the call
    /// that just returned. Without it a budget refusal and "this is not a
    /// polynomial in log(h)" reach the caller as the same bare `None`, and the
    /// resource refusal would be reported as a mathematical decline
    /// (`E-INT-*` instead of `E-BUDGET-*`) — the same out-of-band carrier
    /// `calculus::series::take_series_refusal` uses for the same reason.
    static DECOMPOSE_TRIP: std::cell::Cell<Option<crate::budget::BudgetError>> =
        const { std::cell::Cell::new(None) };
}

/// The budget refusal, if any, that ended the [`decompose_as_log_poly`] call
/// that just returned `None`. Consumes it.
pub fn take_decompose_budget_trip() -> Option<crate::budget::BudgetError> {
    DECOMPOSE_TRIP.with(|c| c.take())
}

/// Grow `coeffs` so that index `deg` exists, refusing rather than allocating
/// without a bound.
///
/// `deg` is derived from an exponent read out of the expression, so nothing
/// upstream constrains it: a negative exponent on the generator (`log(h)^-1`,
/// which reaches here from any integrand with the generator in a denominator)
/// becomes a `usize` near `2^64` at the cast in the caller, and the loop below
/// then pushes until the allocator gives up — a process abort, with no
/// checkpoint in between for a budget to intervene at. Asking
/// [`crate::budget::check_growth`] for the whole span *before* pushing any of
/// it is that checkpoint.
fn grow_coeffs_to(coeffs: &mut Vec<ExprId>, deg: usize, pool: &ExprPool) -> Option<()> {
    if coeffs.len() > deg {
        return Some(());
    }
    let wanted = (deg - coeffs.len() + 1) as u64;
    if let Err(e) = crate::budget::check_growth(wanted) {
        DECOMPOSE_TRIP.with(|c| c.set(Some(e)));
        return None;
    }
    while coeffs.len() <= deg {
        coeffs.push(pool.integer(0_i32));
    }
    Some(())
}

/// Try to decompose `expr` as a polynomial in `log_gen = log(h)`.
///
/// Returns `Some(coeffs)` where `coeffs[k]` is the coefficient of `log_gen^k`,
/// or `None` if the expression cannot be written as a polynomial in `log_gen`
/// — or if building the coefficient list would have cost more than the ambient
/// [`crate::budget`] allows, in which case
/// [`take_decompose_budget_trip`] returns the refusal.
pub fn decompose_as_log_poly(
    expr: ExprId,
    log_gen: ExprId,
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    // Maximum degree to try (practical bound).
    const MAX_LOG_DEGREE: usize = 20;

    DECOMPOSE_TRIP.with(|c| c.set(None));

    let mut coeffs = vec![pool.integer(0_i32); 1]; // coeffs[0] = constant term

    decompose_log_inner(
        expr,
        log_gen,
        pool,
        &mut coeffs,
        pool.integer(1_i32),
        MAX_LOG_DEGREE,
    )?;

    Some(coeffs)
}

/// Recursive helper: accumulate `factor * expr` into `coeffs`.
/// `factor` is a rational expression that multiplies the current subexpression.
fn decompose_log_inner(
    expr: ExprId,
    log_gen: ExprId,
    pool: &ExprPool,
    coeffs: &mut Vec<ExprId>,
    factor: ExprId,
    depth_limit: usize,
) -> Option<()> {
    if depth_limit == 0 {
        return None;
    }

    // Ensure coeffs has at least 1 element.
    if coeffs.is_empty() {
        coeffs.push(pool.integer(0_i32));
    }

    // expr IS the log generator: contributes factor to degree 1.
    if expr == log_gen {
        grow_coeffs_to(coeffs, 1, pool)?;
        let old = coeffs[1];
        coeffs[1] = pool.add(vec![old, factor]);
        return Some(());
    }

    // expr does not involve log_gen: contributes factor * expr to degree 0.
    if is_free_of_log(expr, log_gen, pool) {
        let term = if is_one(factor, pool) {
            expr
        } else {
            pool.mul(vec![factor, expr])
        };
        let old = coeffs[0];
        coeffs[0] = pool.add(vec![old, term]);
        return Some(());
    }

    match pool.get(expr) {
        ExprData::Add(args) => {
            for a in &args {
                decompose_log_inner(*a, log_gen, pool, coeffs, factor, depth_limit - 1)?;
            }
            Some(())
        }

        ExprData::Mul(args) => {
            // Separate: find log_gen (or power of log_gen) in the product.
            let mut log_power = 0i64;
            let mut other_factors: Vec<ExprId> = Vec::new();

            for &a in &args {
                if a == log_gen {
                    log_power += 1;
                } else if let ExprData::Pow { base, exp } = pool.get(a) {
                    if base == log_gen {
                        if let ExprData::Integer(n) = pool.get(exp) {
                            log_power += n.0.to_i64()?;
                        } else {
                            other_factors.push(a);
                        }
                    } else {
                        other_factors.push(a);
                    }
                } else {
                    other_factors.push(a);
                }
            }

            if log_power == 0 {
                // No log factor: treat the whole expression as a constant.
                let term = if is_one(factor, pool) {
                    expr
                } else {
                    pool.mul(vec![factor, expr])
                };
                let old = coeffs[0];
                coeffs[0] = pool.add(vec![old, term]);
                return Some(());
            }

            // Build the coefficient = factor * (other factors).
            let new_factor = if other_factors.is_empty() {
                factor
            } else if is_one(factor, pool) {
                match other_factors.len() {
                    1 => other_factors[0],
                    _ => pool.mul(other_factors),
                }
            } else {
                let mut f = other_factors;
                f.push(factor);
                pool.mul(f)
            };

            // `log_power` is an `i64` accumulated from exponents, so it is
            // negative whenever the generator sits in a *denominator*
            // (`log(x)^-1`).  `as usize` wraps that to ~2^64 and the growth
            // below then pushes zeros until the allocator aborts the process — a
            // 4 GiB request, reached from `∫ dx/(x·log x·(1 + log²(log x)))`.
            // A negative power is not representable as a polynomial in `log(h)`
            // — which is exactly what this decomposition builds — so decline.
            //
            // `grow_coeffs_to` is the independent second line of defence: it
            // meters the allocation against the budget, so a *positive* degree
            // large enough to run away is refused as `E-BUDGET-*` rather than
            // aborting. The two fixes are complementary — this one removes the
            // wrap, that one bounds what an honest large degree may allocate.
            let deg = usize::try_from(log_power).ok()?;
            grow_coeffs_to(coeffs, deg, pool)?;
            let old = coeffs[deg];
            coeffs[deg] = pool.add(vec![old, new_factor]);
            Some(())
        }

        ExprData::Pow { base, exp } => {
            if base == log_gen {
                // log_gen^n: pure power.
                if let ExprData::Integer(n) = pool.get(exp) {
                    if let Some(deg) = n.0.to_u32() {
                        grow_coeffs_to(coeffs, deg as usize, pool)?;
                        let old = coeffs[deg as usize];
                        coeffs[deg as usize] = pool.add(vec![old, factor]);
                        return Some(());
                    }
                }
            }
            None
        }

        _ => None,
    }
}

/// Returns true if `expr` is the integer 1.
fn is_one(expr: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(expr), ExprData::Integer(n) if n.0 == 1)
}

/// Returns true if `expr` syntactically does not involve `log_gen`.
fn is_free_of_log(expr: ExprId, log_gen: ExprId, pool: &ExprPool) -> bool {
    if expr == log_gen {
        return false;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().all(|&a| is_free_of_log(a, log_gen, pool))
        }
        ExprData::Pow { base, exp } => {
            is_free_of_log(base, log_gen, pool) && is_free_of_log(exp, log_gen, pool)
        }
        ExprData::Func { ref args, .. } => args.iter().all(|&a| is_free_of_log(a, log_gen, pool)),
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Degree estimation
// ---------------------------------------------------------------------------

/// Estimate the degree of `expr` as a polynomial in `var`.
/// Returns `None` if the expression is not a polynomial in `var`.
pub fn poly_degree(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<u32> {
    if expr == var {
        return Some(1);
    }
    if is_free_of_var(expr, var, pool) {
        return Some(0);
    }
    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut max_d = 0u32;
            for &a in &args {
                let d = poly_degree(a, var, pool)?;
                max_d = max_d.max(d);
            }
            Some(max_d)
        }
        ExprData::Mul(args) => {
            let mut total = 0u32;
            for &a in &args {
                let d = poly_degree(a, var, pool)?;
                total = total.checked_add(d)?;
            }
            Some(total)
        }
        ExprData::Pow { base, exp } if base == var => match pool.get(exp) {
            ExprData::Integer(n) => n.0.to_u32(),
            _ => None,
        },
        ExprData::Pow { base, .. } if is_free_of_var(base, var, pool) => Some(0),
        _ => None,
    }
}

#[cfg(test)]
mod log_power_underflow_tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    /// A log generator in a *denominator* gives `log_power < 0`.
    ///
    /// `log_power as usize` used to wrap that to ~2^64, and the zero-fill loop
    /// then pushed until the allocator aborted the process. A 4 GiB request was
    /// reachable from `∫ dx/(x·log x·(1 + log²(log x)))` — an elementary
    /// integral — and no `Budget(wall_ms=…)` could interrupt it, because the
    /// loop never returns to a cooperative checkpoint.
    ///
    /// This decomposition builds a *polynomial* in `log(h)`, so a negative
    /// power is simply not representable: it must decline, promptly.
    #[test]
    fn negative_log_power_declines_instead_of_allocating() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let logx = pool.func("log", vec![x]);

        for power in [-1_i32, -2, -7] {
            let f = pool.mul(vec![
                pool.pow(x, pool.integer(-1_i32)),
                pool.pow(logx, pool.integer(power)),
            ]);
            let got = decompose_as_log_poly(f, logx, &pool);
            assert!(
                got.is_none(),
                "log(x)^{power} is not a polynomial in log(x); expected a decline, got {got:?}"
            );
        }
    }

    /// The non-negative path is untouched: `x·log(x)^2` still decomposes.
    #[test]
    fn non_negative_log_power_still_decomposes() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let logx = pool.func("log", vec![x]);
        let f = pool.mul(vec![x, pool.pow(logx, pool.integer(2_i32))]);
        assert!(
            decompose_as_log_poly(f, logx, &pool).is_some(),
            "x·log(x)^2 is a polynomial in log(x) and must still decompose"
        );
    }
}
