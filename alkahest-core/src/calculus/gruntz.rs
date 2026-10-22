//! Gruntz comparability-graph algorithm for symbolic limits of exp-log combinations (V2-17).
//!
//! Reference: Gruntz (1996) "On Computing Limits in a Symbolic Manipulation System"
//!
//! # Algorithm (limit as var → +∞)
//!
//! 1. Collect all `exp(h)` subexpressions where `h` diverges to ±∞.
//! 2. Build the comparability ordering: `exp(h₁) ≻ exp(h₂)` iff `lim(h₁/h₂) = ∞`.
//! 3. Retain only the **maximal** (MRV) set — the fastest-growing elements.
//! 4. Pick ω ∈ MRV with ω → 0⁺ (inner h_ω → -∞).  If all inners → +∞, form ω = exp(-h).
//! 5. Rewrite expr: each element `exp(hₑ)` ∈ MRV becomes `ω^c`, c = lim(hₑ/h_ω).
//! 6. Expand the rewritten expression as a Laurent series in ω → 0⁺.
//! 7. Leading power > 0 → 0; < 0 → ±∞; = 0 → recurse on the coefficient.

use crate::calculus::limits::{checkpoint, limit, LimitDirection, LimitError};
use crate::calculus::series::local_expansion;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::rules::{CollectExp, ExpPow};
use crate::simplify::{rules_for_config, simplify, simplify_with, SimplifyConfig};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Recursion guard — prevents unbounded re-entry when Gruntz sub-limits
// trigger Gruntz again on simpler expressions.
// ---------------------------------------------------------------------------

thread_local! {
    static GRUNTZ_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

const GRUNTZ_MAX_DEPTH: u32 = 8;
const SERIES_TERMS: u32 = 8;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Try to compute `lim_{var → +∞} expr` using the Gruntz algorithm.
/// Returns `None` if not applicable or if a sub-step fails.
pub(crate) fn try_gruntz(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, LimitError> {
    let current = GRUNTZ_DEPTH.with(|d| d.get());
    if current >= GRUNTZ_MAX_DEPTH {
        return Ok(None);
    }
    checkpoint(pool)?;
    GRUNTZ_DEPTH.with(|d| d.set(current + 1));
    let result = gruntz_inner(expr, var, pool);
    GRUNTZ_DEPTH.with(|d| d.set(current));
    result
}

/// Collapse exp(h)^n → exp(n·h) and exp(a)·exp(b) → exp(a+b) before analysis.
/// These rules must NOT be in the default simplifier (they undo omega-power rewrites),
/// so we apply them here only as an input preprocessing step, combined with the full
/// default rule set so that exponent arithmetic is fully simplified.
fn preprocess_exp(expr: ExprId, pool: &ExprPool) -> ExprId {
    let cfg = SimplifyConfig::default();
    let mut rules = rules_for_config(&cfg);
    rules.push(Box::new(ExpPow));
    rules.push(Box::new(CollectExp));
    simplify_with(expr, pool, &rules, cfg).value
}

fn gruntz_inner(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<Option<ExprId>, LimitError> {
    // Exponentials in disguise — `aˣ`, `f(x)^{g(x)}`, `sinh`, `cosh` — are
    // written as `exp` so the mrv step sees them; left as they were, they
    // pass for lower-class coefficients (`sinh(x)/eˣ` read as `0`).
    let unhidden = rewrite_hidden_exponentials(expr, var, pool);
    // Collapse exp(h)^n → exp(n·h) and merge exp factors before analysis.
    let preprocessed = preprocess_exp(unhidden, pool);

    // Collect exp(h) subexpressions where h diverges
    let mut candidates: Vec<(ExprId, ExprId)> = Vec::new();
    collect_exp_subexprs(preprocessed, var, pool, &mut candidates)?;

    if candidates.is_empty() {
        // If preprocessing changed the expression (e.g. exp(x+1)/exp(x) → exp(1)),
        // delegate to limit() on the simplified form so it can use direct substitution.
        // (`(1 + 1/x)^{x²}/eˣ` becomes `exp(x²·log(1 + 1/x) − x)`, whose
        // exponent converges: no mrv set, just a continuous `exp`.)
        if preprocessed != expr {
            return match limit(
                preprocessed,
                var,
                pool.pos_infinity(),
                LimitDirection::Bidirectional,
                pool,
            ) {
                Ok(l) => Ok(Some(l)),
                Err(LimitError::DepthExceeded) => Err(LimitError::DepthExceeded),
                Err(_) => Ok(None),
            };
        }
        return Ok(None);
    }
    let expr = preprocessed;

    // Build MRV set: keep only the maximally-growing elements
    let mrv = build_mrv_set(candidates, var, pool)?;
    if mrv.is_empty() {
        return Ok(None);
    }

    // Find ω: the mrv element that → 0, or construct one
    let (omega_expr, omega_inner) = match find_omega(&mrv, var, pool)? {
        Some(p) => p,
        None => return Ok(None),
    };

    // Only exponentials enter the mrv set, so `var` itself (= exp(log var)) is
    // treated as a coefficient — sound only when ω beats every power of `var`.
    // `exp(log(x)/2)` is `√x`: same class as `x`, and `exp(log(x)/2)/√x`
    // would otherwise read as `ω⁻¹ · x^{-1/2}` → ∞.
    if !beats_every_power(omega_inner, var, pool)? {
        return Ok(None);
    }

    // Rewrite expr by substituting mrv elements as powers of ω. ω itself
    // becomes a fresh positive symbol `w`, so the leading-term analysis below
    // works with an ordinary variable and nothing the simplifier does to
    // `exp(h_ω)` can hide an occurrence of it.
    let w = pool.symbol(OMEGA_SYMBOL, Domain::Positive);
    let rewritten = match rewrite_in_omega(expr, &mrv, omega_expr, omega_inner, w, var, pool) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };

    // Leading term `coeff · ωᵖ` of the rewritten expression as ω → 0⁺.
    let ctx = OmegaCtx {
        w,
        h_omega: omega_inner,
        var,
        pool,
    };
    // Split every `log` whose leading order is not 1 into its explicit part
    // and a `log` of something tending to 1, so a sum like
    // `log(ω⁻¹ + ω) − x` cancels symbolically instead of in the leading
    // coefficients, where only the next order could decide it.
    let rewritten = match split_logs(rewritten, &ctx, false)? {
        Some(r) => simplify(r, pool).value,
        None => rewritten,
    };
    let Some(lt) = leading_term(rewritten, &ctx)? else {
        return Ok(None);
    };
    let coeff = lt.coeff;
    let Some(power) = power_sign(lt.power, pool) else {
        // A symbolic rate whose sign nobody stated: `e^{-k t}` with `k` free.
        crate::calculus::limits::note_missing_assumption(lt.power, var, pool);
        return Ok(None);
    };

    if power > 0 {
        return Ok(Some(pool.integer(0_i32)));
    }
    if power < 0 {
        // No established sign, no answer: decline the whole Gruntz route
        // rather than name an infinity by coin flip. `limits` has already
        // recorded the parameter whose sign was wanted.
        let Some(sign) = sign_of_coeff_at_inf(coeff, var, pool) else {
            crate::calculus::limits::note_missing_assumption(coeff, var, pool);
            return Ok(None);
        };
        return Ok(Some(signed_infinity(pool, sign)));
    }
    // No progress: the coefficient is the problem we were handed.
    if coeff == expr {
        return Ok(None);
    }

    // power == 0: the limit equals lim_{x→+∞} coeff(x)
    match limit(
        coeff,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    ) {
        Ok(l) => Ok(Some(l)),
        Err(LimitError::DepthExceeded) => Err(LimitError::DepthExceeded),
        Err(_) => Ok(None),
    }
}

/// `aᵍ` (with `g` depending on `var`) ↦ `exp(g·log a)`, `sinh u ↦ (eᵘ − e⁻ᵘ)/2`,
/// `cosh u ↦ (eᵘ + e⁻ᵘ)/2`, everywhere in `expr`. A base that is a negative
/// constant is left alone (`(−2)ˣ` is not an exponential of a real argument);
/// [`leading_term`] then refuses to treat it as a coefficient.
fn rewrite_hidden_exponentials(expr: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    if !contains_var(expr, var, pool) {
        return expr;
    }
    let exp = |a: ExprId| pool.func("exp".to_string(), vec![a]);
    match pool.get(expr) {
        ExprData::Add(xs) => pool.add(
            xs.iter()
                .map(|&x| rewrite_hidden_exponentials(x, var, pool))
                .collect(),
        ),
        ExprData::Mul(xs) => pool.mul(
            xs.iter()
                .map(|&x| rewrite_hidden_exponentials(x, var, pool))
                .collect(),
        ),
        ExprData::Pow { base, exp: e } => {
            let b = rewrite_hidden_exponentials(base, var, pool);
            let g = rewrite_hidden_exponentials(e, var, pool);
            let negative_const_base = !contains_var(b, var, pool)
                && crate::calculus::limits::quantity_sign(b, pool)
                    != Some(crate::simplify::Sign::Positive);
            if contains_var(g, var, pool) && !negative_const_base {
                let log_b = pool.func("log".to_string(), vec![b]);
                return exp(pool.mul(vec![g, log_b]));
            }
            pool.pow(b, g)
        }
        ExprData::Func { name, args } => {
            let na: Vec<ExprId> = args
                .iter()
                .map(|&a| rewrite_hidden_exponentials(a, var, pool))
                .collect();
            if na.len() == 1 && matches!(name.as_str(), "sinh" | "cosh") {
                let u = na[0];
                let minus_u = pool.mul(vec![pool.integer(-1_i32), u]);
                let other = if name == "sinh" {
                    pool.mul(vec![pool.integer(-1_i32), exp(minus_u)])
                } else {
                    exp(minus_u)
                };
                return pool.mul(vec![pool.rational(1, 2), pool.add(vec![exp(u), other])]);
            }
            pool.func(name.clone(), na)
        }
        _ => expr,
    }
}

/// `true` when a `var`-dependent node, left over as a coefficient after the
/// mrv rewrite, is known to grow slower than every exponential: built from
/// `var`, constants, sums, products, constant powers, `log`, and bounded or
/// algebraic heads. Anything else (`Γ(x)`, `x^x` that escaped the rewrite, an
/// unknown function) may be an exponential in disguise and is refused.
fn is_subexponential(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if !contains_var(e, var, pool) {
        return true;
    }
    match pool.get(e) {
        ExprData::Symbol { .. } => true,
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            xs.iter().all(|&x| is_subexponential(x, var, pool))
        }
        ExprData::Pow { base, exp } => {
            !contains_var(exp, var, pool) && is_subexponential(base, var, pool)
        }
        ExprData::Func { name, args } => {
            // `exp` nodes left here are below the mrv class by construction.
            matches!(
                name.as_str(),
                "exp"
                    | "log"
                    | "ln"
                    | "sqrt"
                    | "cbrt"
                    | "abs"
                    | "sin"
                    | "cos"
                    | "atan"
                    | "tanh"
                    | "erf"
            ) && args.iter().all(|&a| is_subexponential(a, var, pool))
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Step 1 — collect diverging exp subexpressions
// ---------------------------------------------------------------------------

/// Walk `expr` and collect `(exp(h), h)` pairs where `h` contains `var` and
/// `limit(h, var, +∞)` is ±∞.
fn collect_exp_subexprs(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    out: &mut Vec<(ExprId, ExprId)>,
) -> Result<(), LimitError> {
    match pool.get(expr) {
        ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
            let inner = args[0];
            if contains_var(inner, var, pool) {
                let lim = limit(
                    inner,
                    var,
                    pool.pos_infinity(),
                    LimitDirection::Bidirectional,
                    pool,
                );
                if let Ok(l) = lim {
                    if is_pos_inf(l, pool) || is_neg_inf(l, pool) {
                        out.push((expr, inner));
                        // Recurse into inner to discover nested exp terms
                        collect_exp_subexprs(inner, var, pool, out)?;
                        return Ok(());
                    }
                }
            }
            // Inner doesn't diverge — still recurse in case args have nested exp
            for a in args.iter() {
                collect_exp_subexprs(*a, var, pool, out)?;
            }
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            for x in xs {
                collect_exp_subexprs(x, var, pool, out)?;
            }
        }
        ExprData::Pow { base, exp } => {
            collect_exp_subexprs(base, var, pool, out)?;
            collect_exp_subexprs(exp, var, pool, out)?;
        }
        ExprData::Func { args, .. } => {
            for a in args {
                collect_exp_subexprs(a, var, pool, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 2 — comparability and MRV set construction
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Growth {
    Slower,
    Equal,
    Faster,
}

/// Compare growth rates of `exp(h1)` and `exp(h2)` via `lim(h1/h2, x→+∞)`.
fn compare_growth(
    h1: ExprId,
    h2: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Growth, LimitError> {
    if h1 == h2 {
        return Ok(Growth::Equal);
    }
    let ratio = simplify(pool.mul(vec![h1, pool.pow(h2, pool.integer(-1_i32))]), pool).value;
    let l = limit(
        ratio,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    )?;
    if is_zero(l, pool) {
        Ok(Growth::Slower)
    } else if is_pos_inf(l, pool) || is_neg_inf(l, pool) {
        Ok(Growth::Faster)
    } else {
        Ok(Growth::Equal)
    }
}

/// Retain only elements of `candidates` that are not dominated by any other.
fn build_mrv_set(
    candidates: Vec<(ExprId, ExprId)>,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, ExprId)>, LimitError> {
    // De-duplicate by exp ExprId
    let mut unique: Vec<(ExprId, ExprId)> = Vec::new();
    for c in candidates {
        if !unique.iter().any(|(e, _)| *e == c.0) {
            unique.push(c);
        }
    }
    if unique.len() == 1 {
        return Ok(unique);
    }

    let n = unique.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in 0..n {
            if i == j || dominated[i] {
                continue;
            }
            // O(n²) sub-limits, each of which can itself be arbitrarily hard.
            checkpoint(pool)?;
            let h_i = unique[i].1;
            let h_j = unique[j].1;
            if let Ok(Growth::Slower) = compare_growth(h_i, h_j, var, pool) {
                dominated[i] = true;
            }
        }
    }
    Ok(unique
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !dominated[*i])
        .map(|(_, v)| v)
        .collect())
}

// ---------------------------------------------------------------------------
// Step 3 — find ω
// ---------------------------------------------------------------------------

/// Return `(omega_expr, omega_inner)` where `omega_expr → 0⁺`.
///
/// Prefers an mrv element whose inner → -∞.  If none exists, constructs
/// `exp(-h)` from the first element whose inner → +∞.
fn find_omega(
    mrv: &[(ExprId, ExprId)],
    var: ExprId,
    pool: &ExprPool,
) -> Result<Option<(ExprId, ExprId)>, LimitError> {
    // Among equally good choices prefer an element whose exponent is a plain
    // generalised polynomial (`−2x` over `x²·log(1 + 1/x)`): every later step
    // divides by `h_ω`, and the polynomial one keeps those sub-limits simple.
    let simple = |h: ExprId| power_degree(h, var, pool).is_some();
    let mut to_zero: Vec<(ExprId, ExprId)> = Vec::new();
    let mut undecided = false;
    for &(e, h) in mrv {
        let lim = match limit(
            h,
            var,
            pool.pos_infinity(),
            LimitDirection::Bidirectional,
            pool,
        ) {
            Ok(l) => l,
            Err(LimitError::DepthExceeded) => return Err(LimitError::DepthExceeded),
            Err(_) => {
                undecided = true;
                continue;
            }
        };
        if is_neg_inf(lim, pool) {
            to_zero.push((e, h));
        } else if !is_pos_inf(lim, pool) {
            undecided = true;
        }
    }
    if let Some(&p) = to_zero.iter().find(|(_, h)| simple(*h)).or(to_zero.first()) {
        return Ok(Some(p));
    }
    if undecided {
        return Ok(None);
    }
    // All inners → +∞: build ω = exp(-h) from one of them
    if let Some(&(_, h)) = mrv.iter().find(|(_, h)| simple(*h)).or(mrv.first()) {
        let neg_h = simplify(pool.mul(vec![pool.integer(-1_i32), h]), pool).value;
        let omega = simplify(pool.func("exp".to_string(), vec![neg_h]), pool).value;
        return Ok(Some((omega, neg_h)));
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Step 4 — rewrite in ω
// ---------------------------------------------------------------------------

/// Name of the fresh positive symbol that stands for ω in the rewritten
/// expression. Never escapes [`gruntz_inner`]: every coefficient handed back to
/// [`limit`] is free of it by construction.
const OMEGA_SYMBOL: &str = "__gruntz_omega";

/// Build the rewritten expression: every mrv `exp(hₑ)` becomes
/// `exp(hₑ − c·h_ω) · w^c` with `c = lim hₑ/h_ω` and `w` standing for
/// `ω = exp(h_ω)`.
///
/// The factor `exp(hₑ − c·h_ω)` is the whole difference between `exp(hₑ)` and
/// `ω^c`; it belongs to a strictly lower comparability class, so it is part of
/// the leading *coefficient*, and dropping it (as this used to) changes the
/// answer whenever the coefficient survives to the end:
/// `e^{1−3t}/(e^{−3t} + e^{−4t}) → e`, not `0`.
fn rewrite_in_omega(
    expr: ExprId,
    mrv: &[(ExprId, ExprId)],
    omega_expr: ExprId,
    omega_inner: ExprId,
    w: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LimitError> {
    let mut subst: HashMap<ExprId, ExprId> = HashMap::new();
    subst.insert(omega_expr, w);
    // Pass 1: the power of ω each element carries. Same class means a finite,
    // nonzero, `var`-free ratio — a rational, or a symbolic one like `k₂/k₁`
    // whose sign the assumptions decide later; anything else (0, ∞, a ratio
    // that still moves) is not something the power bookkeeping can carry.
    let mut pending: Vec<(ExprId, ExprId, ExprId)> = Vec::new();
    for &(e, h_e) in mrv {
        if e == omega_expr {
            continue;
        }
        let c = exponent_relative_to_omega(h_e, omega_inner, var, pool)?;
        if is_zero(c, pool)
            || contains_var(c, var, pool)
            || is_pos_inf(c, pool)
            || is_neg_inf(c, pool)
            || crate::calculus::limits::has_unusable_node(c, pool)
        {
            return Err(LimitError::Unsupported);
        }
        pending.push((e, h_e, c));
    }
    // Pass 2: the lower-class factor, with any ω-class element inside its
    // exponent rewritten too (`exp(x + e^{−x})` ↦ `exp(w) · w⁻¹`).
    let base_subst = subst.clone();
    for (e, h_e, c) in pending {
        let rest = simplify(
            pool.add(vec![
                h_e,
                pool.mul(vec![pool.integer(-1_i32), c, omega_inner]),
            ]),
            pool,
        )
        .value;
        let rest = rewrite_node(rest, &base_subst, pool);
        let factor = if is_zero(rest, pool) {
            pool.pow(w, c)
        } else {
            pool.mul(vec![
                pool.func("exp".to_string(), vec![rest]),
                pool.pow(w, c),
            ])
        };
        subst.insert(e, simplify(factor, pool).value);
    }
    Ok(rewrite_node(expr, &subst, pool))
}

/// `true` when `exp(h)` outgrows (or, for `h → −∞`, out-decays) every power of
/// `var`: `log var / h → 0`.
///
/// Decided structurally when `h` is a generalised polynomial in `var` with a
/// positive leading degree (`−2x`, `√x`, `x² − x`), otherwise by the limit.
fn beats_every_power(h: ExprId, var: ExprId, pool: &ExprPool) -> Result<bool, LimitError> {
    if matches!(power_degree(h, var, pool), Some(d) if d > 0) {
        return Ok(true);
    }
    let ratio = simplify(
        pool.mul(vec![
            pool.func("log".to_string(), vec![var]),
            pool.pow(h, pool.integer(-1_i32)),
        ]),
        pool,
    )
    .value;
    match limit(
        ratio,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    ) {
        Ok(l) => Ok(is_zero(l, pool)),
        Err(LimitError::DepthExceeded) => Err(LimitError::DepthExceeded),
        Err(_) => Ok(false),
    }
}

/// `lim_{var→+∞}` of a generalised polynomial `Σ cᵢ·var^{rᵢ}` (radicals
/// included) with a unique top-degree term: `±∞` by that term's sign for a
/// positive degree, `0` for a negative one. `None` for anything else,
/// including a top term of degree 0 or of undetermined sign.
///
/// `lim_{x→∞} √x` was not computable at all — and with it every Gruntz step
/// over `exp(√x)`, whose exponent's divergence it is asked to establish.
pub(crate) fn generalized_polynomial_limit(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let d = power_degree(expr, var, pool)?;
    if d == 0 {
        return None;
    }
    if d < 0 {
        return Some(pool.integer(0_i32));
    }
    let s = monomial_sign(expr, var, pool)?;
    Some(signed_infinity(pool, s))
}

/// Eventual sign (as `var → +∞`) of a generalised polynomial, read off its
/// unique top-degree term.
fn monomial_sign(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<i8> {
    if !contains_var(e, var, pool) {
        return match crate::calculus::limits::quantity_sign(e, pool)? {
            crate::simplify::Sign::Positive => Some(1),
            crate::simplify::Sign::Negative => Some(-1),
            crate::simplify::Sign::Zero => None,
        };
    }
    if e == var {
        return Some(1);
    }
    match pool.get(e) {
        ExprData::Pow { base, exp } => {
            let s = monomial_sign(base, var, pool)?;
            if s > 0 {
                return Some(1);
            }
            let r = expr_as_rational(exp, pool)?;
            if *r.denom() != 1 {
                return None;
            }
            Some(if r.numer().is_even() { 1 } else { s })
        }
        ExprData::Mul(xs) => {
            let mut s = 1_i8;
            for x in xs {
                s *= monomial_sign(x, var, pool)?;
            }
            Some(s)
        }
        ExprData::Add(xs) => {
            let top = power_degree(e, var, pool)?;
            let lead = xs
                .iter()
                .find(|&&x| power_degree(x, var, pool).as_ref() == Some(&top))?;
            monomial_sign(*lead, var, pool)
        }
        ExprData::Func { name, args } if args.len() == 1 && name == "sqrt" => {
            (monomial_sign(args[0], var, pool)? > 0).then_some(1)
        }
        ExprData::Func { name, args } if args.len() == 1 && name == "cbrt" => {
            monomial_sign(args[0], var, pool)
        }
        _ => None,
    }
}

/// The exact degree of `e` as `var → ∞` when `e` is a sum of
/// `const · var^r` terms with a *unique* top-degree term (so nothing can
/// cancel it), else `None`.
fn power_degree(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    if !contains_var(e, var, pool) {
        return Some(rug::Rational::from(0));
    }
    if e == var {
        return Some(rug::Rational::from(1));
    }
    match pool.get(e) {
        ExprData::Pow { base, exp } => {
            let r = expr_as_rational(exp, pool)?;
            Some(power_degree(base, var, pool)? * r)
        }
        ExprData::Mul(xs) => {
            let mut d = rug::Rational::from(0);
            for x in xs {
                d += power_degree(x, var, pool)?;
            }
            Some(d)
        }
        ExprData::Add(xs) => {
            let ds: Vec<rug::Rational> = xs
                .iter()
                .map(|&x| power_degree(x, var, pool))
                .collect::<Option<_>>()?;
            let top = ds.iter().max()?.clone();
            (ds.iter().filter(|d| **d == top).count() == 1).then_some(top)
        }
        ExprData::Func { name, args } if args.len() == 1 && name == "sqrt" => {
            Some(power_degree(args[0], var, pool)? / rug::Rational::from(2))
        }
        ExprData::Func { name, args } if args.len() == 1 && name == "cbrt" => {
            Some(power_degree(args[0], var, pool)? / rug::Rational::from(3))
        }
        _ => None,
    }
}

fn exponent_relative_to_omega(
    h_e: ExprId,
    h_omega: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LimitError> {
    if h_e == h_omega {
        return Ok(pool.integer(1_i32));
    }
    let ratio = simplify(
        pool.mul(vec![h_e, pool.pow(h_omega, pool.integer(-1_i32))]),
        pool,
    )
    .value;
    limit(
        ratio,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    )
}

fn rewrite_node(expr: ExprId, subst: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> ExprId {
    if let Some(&rep) = subst.get(&expr) {
        return rep;
    }
    match pool.get(expr) {
        ExprData::Add(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| rewrite_node(*x, subst, pool)).collect();
            simplify(pool.add(ys), pool).value
        }
        ExprData::Mul(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| rewrite_node(*x, subst, pool)).collect();
            simplify(pool.mul(ys), pool).value
        }
        ExprData::Pow { base, exp } => {
            let b = rewrite_node(base, subst, pool);
            let e = rewrite_node(exp, subst, pool);
            // Flatten (x^a)^b → x^(a·b) so rational omega powers compose correctly.
            if let ExprData::Pow {
                base: inner_base,
                exp: inner_e,
            } = pool.get(b)
            {
                let combined = simplify(pool.mul(vec![inner_e, e]), pool).value;
                return simplify(pool.pow(inner_base, combined), pool).value;
            }
            simplify(pool.pow(b, e), pool).value
        }
        ExprData::Func { name, args } => {
            let na: Vec<ExprId> = args.iter().map(|a| rewrite_node(*a, subst, pool)).collect();
            simplify(pool.func(name.clone(), na), pool).value
        }
        _ => expr,
    }
}

// ---------------------------------------------------------------------------
// Step 5 — leading term of the rewritten expression as ω → 0⁺
// ---------------------------------------------------------------------------

/// What the leading-term analysis works against: the symbol `w` standing for
/// `ω = exp(h_ω)`, and `h_ω` itself, which is what `log ω` *is*.
struct OmegaCtx<'a> {
    w: ExprId,
    h_omega: ExprId,
    var: ExprId,
    pool: &'a ExprPool,
}

/// `e = coeff · ωᵖ · (1 + o(1))` as `ω → 0⁺`, with `coeff` free of ω.
///
/// For `p ≠ 0` the coefficient must lie in a strictly lower comparability
/// class than ω — otherwise "`p > 0` ⇒ the limit is 0" is false — and every
/// rule below preserves that, with one exception: `log ω = h_ω` puts the
/// *exponent* of ω into a coefficient, and `exp` of such a coefficient can be
/// ω itself in disguise (`exp(log ω) = ω`). `log_derived` marks coefficients
/// that carry an `h_ω`; [`leading_term_exp`] refuses to exponentiate them.
struct LeadingTerm {
    coeff: ExprId,
    power: ExprId,
    log_derived: bool,
}

/// Leading term of `expr` in ω, or `None` when no rule here can establish it
/// (the caller then declines the whole Gruntz route — never a guess).
///
/// Handles ω-free terms, `ω`, products, sums (minimum power; a cancelling
/// leading coefficient falls back to a Taylor expansion only where that
/// expansion is sound), rational powers, `exp` and `log`. `log` is the reason
/// this exists: `log(c·ωᵖ·(1 + o(1))) = p·h_ω + log c + o(1)`, a term of
/// order ω⁰ whose coefficient grows like `h_ω`. The Taylor expansion this used
/// to fall back to has no business near `log ω` — it read
/// `lim_{t→∞} log(e^{−2t})/t` as `0` (true −2) and, for `log(e^t)/t`,
/// differentiated without bound until the allocator gave up.
fn leading_term(expr: ExprId, ctx: &OmegaCtx<'_>) -> Result<Option<LeadingTerm>, LimitError> {
    let pool = ctx.pool;
    checkpoint(pool)?;
    let w = ctx.w;
    if !contains_var(expr, w, pool) {
        if !is_subexponential(expr, ctx.var, pool) {
            return Ok(None);
        }
        return Ok(Some(LeadingTerm {
            coeff: expr,
            power: pool.integer(0_i32),
            log_derived: false,
        }));
    }
    if expr == w {
        return Ok(Some(LeadingTerm {
            coeff: pool.integer(1_i32),
            power: pool.integer(1_i32),
            log_derived: false,
        }));
    }
    match pool.get(expr) {
        ExprData::Mul(xs) => {
            let mut coeffs = Vec::with_capacity(xs.len());
            let mut powers = Vec::with_capacity(xs.len());
            let mut log_derived = false;
            for &x in &xs {
                let Some(lt) = leading_term(x, ctx)? else {
                    return Ok(None);
                };
                powers.push(lt.power);
                log_derived |= lt.log_derived;
                coeffs.push(lt.coeff);
            }
            Ok(Some(LeadingTerm {
                coeff: simplify(pool.mul(coeffs), pool).value,
                power: simplify(pool.add(powers), pool).value,
                log_derived,
            }))
        }
        ExprData::Add(xs) => {
            let mut terms = Vec::with_capacity(xs.len());
            for &x in &xs {
                let Some(lt) = leading_term(x, ctx)? else {
                    return Ok(None);
                };
                terms.push(lt);
            }
            // The smallest power wins; deciding which one that is may take
            // the assumptions (`e^{-k₁t} − e^{-k₂t}` needs `k₁ < k₂`).
            let mut min_p = terms[0].power;
            for t in &terms[1..] {
                let d = simplify(
                    pool.add(vec![t.power, pool.mul(vec![pool.integer(-1_i32), min_p])]),
                    pool,
                )
                .value;
                match power_sign(d, pool) {
                    Some(s) if s < 0 => min_p = t.power,
                    Some(_) => {}
                    None => {
                        crate::calculus::limits::note_missing_assumption(d, ctx.var, pool);
                        return Ok(None);
                    }
                }
            }
            let lead: Vec<&LeadingTerm> = terms
                .iter()
                .filter(|t| {
                    t.power == min_p
                        || power_sign(power_diff(t.power, min_p, pool), pool) == Some(0)
                })
                .collect();
            let coeff = simplify(pool.add(lead.iter().map(|t| t.coeff).collect()), pool).value;
            if is_zero(coeff, pool) || vanishes_numerically(coeff, ctx) {
                // The leading orders cancel; the next one decides and only an
                // expansion can find it.
                return series_leading_term(expr, ctx);
            }
            Ok(Some(LeadingTerm {
                coeff,
                power: min_p,
                log_derived: lead.iter().any(|t| t.log_derived),
            }))
        }
        ExprData::Pow { base, exp } => {
            // A constant exponent — rational, or a parameter like `k`.
            if !contains_var(exp, w, pool) && !contains_var(exp, ctx.var, pool) {
                let Some(lt) = leading_term(base, ctx)? else {
                    return Ok(None);
                };
                // A negative or fractional power of a coefficient that is
                // really zero is not a leading term of anything.
                let harmless =
                    matches!(expr_as_rational(exp, pool), Some(r) if r >= 0 && *r.denom() == 1);
                if !harmless && vanishes_numerically(lt.coeff, ctx) {
                    return Ok(None);
                }
                let coeff = if lt.coeff == pool.integer(1_i32) {
                    lt.coeff
                } else {
                    simplify(pool.pow(lt.coeff, exp), pool).value
                };
                return Ok(Some(LeadingTerm {
                    coeff,
                    power: simplify(pool.mul(vec![lt.power, exp]), pool).value,
                    log_derived: lt.log_derived,
                }));
            }
            // `b^g` with `b` free of ω: `exp(g·log b)`.
            if !contains_var(base, w, pool) {
                let arg = simplify(
                    pool.mul(vec![exp, pool.func("log".to_string(), vec![base])]),
                    pool,
                )
                .value;
                return leading_term_exp(arg, ctx);
            }
            Ok(None)
        }
        ExprData::Func { name, args } if args.len() == 1 && name == "exp" => {
            leading_term_exp(args[0], ctx)
        }
        // `sqrt(A)`/`cbrt(A)` as `A^{1/2}`/`A^{1/3}`: the leading coefficient
        // is positive for a real `sqrt`, and `cbrt` is odd, so the principal
        // power of the leading term is the leading term of the root.
        ExprData::Func { name, args }
            if args.len() == 1 && matches!(name.as_str(), "sqrt" | "cbrt") =>
        {
            let k = if name == "sqrt" { 2 } else { 3 };
            let as_pow = pool.pow(args[0], pool.rational(1, k));
            if name == "cbrt" {
                // A negative radicand has a real cube root but no real
                // principal power; only take the shortcut when it is positive.
                let Some(lt) = leading_term(args[0], ctx)? else {
                    return Ok(None);
                };
                if crate::calculus::limits::quantity_sign(lt.coeff, pool)
                    != Some(crate::simplify::Sign::Positive)
                {
                    return Ok(None);
                }
            }
            leading_term(as_pow, ctx)
        }
        ExprData::Func { name, args } if args.len() == 1 && name == "log" => {
            let Some(lt) = leading_term(args[0], ctx)? else {
                return Ok(None);
            };
            if !log_coefficient_is_positive(lt.coeff, ctx) {
                return Ok(None);
            }
            let log_c = simplify(pool.func("log".to_string(), vec![lt.coeff]), pool).value;
            let p_h = simplify(pool.mul(vec![lt.power, ctx.h_omega]), pool).value;
            let l0 = simplify(pool.add(vec![p_h, log_c]), pool).value;
            if is_zero(l0, pool) || vanishes_numerically(l0, ctx) {
                // log(1 + u) with u → 0: `log(1 + u) = u·(1 + o(1))`, so the
                // leading term is u's — when u visibly vanishes.
                if is_zero(lt.power, pool) {
                    let u = expand_in_w(pool.add(vec![args[0], pool.integer(-1_i32)]), w, pool);
                    if let Some(lu) = leading_term(u, ctx)? {
                        if power_sign(lu.power, pool) == Some(1) {
                            return Ok(Some(lu));
                        }
                    }
                }
                // Otherwise a Taylor expansion, where it is sound.
                return series_leading_term(expr, ctx);
            }
            Ok(Some(LeadingTerm {
                coeff: l0,
                power: pool.integer(0_i32),
                log_derived: lt.log_derived || !is_zero(lt.power, pool),
            }))
        }
        _ => series_leading_term(expr, ctx),
    }
}

/// `log A ↦ p·h_ω + log c + log(A·ω^{−p}/c)` for every `log` with `A ~ c·ωᵖ`,
/// outside exponents. The last `log` has an argument tending to 1, which the
/// leading-term and Taylor steps handle; the explicit `p·h_ω` lets a sum cancel
/// the growing part exactly (`log(eˣ + e⁻ˣ) − x`).
///
/// Never applied inside `exp` or a power's exponent: `exp(p·h_ω)` is ω itself,
/// and once written out it would pass for a lower-class coefficient. Returns
/// `None` when nothing changed.
fn split_logs(
    expr: ExprId,
    ctx: &OmegaCtx<'_>,
    in_exponent: bool,
) -> Result<Option<ExprId>, LimitError> {
    let pool = ctx.pool;
    let w = ctx.w;
    if !contains_var(expr, w, pool) {
        return Ok(None);
    }
    let rebuild = |xs: &[ExprId], in_exp: bool| -> Result<Option<Vec<ExprId>>, LimitError> {
        let mut changed = false;
        let mut out = Vec::with_capacity(xs.len());
        for &x in xs {
            match split_logs(x, ctx, in_exp)? {
                Some(y) => {
                    changed = true;
                    out.push(y);
                }
                None => out.push(x),
            }
        }
        Ok(changed.then_some(out))
    };
    match pool.get(expr) {
        ExprData::Add(xs) => Ok(rebuild(&xs, in_exponent)?.map(|v| pool.add(v))),
        ExprData::Mul(xs) => Ok(rebuild(&xs, in_exponent)?.map(|v| pool.mul(v))),
        ExprData::Pow { base, exp } => {
            let b = split_logs(base, ctx, in_exponent)?;
            let e = split_logs(exp, ctx, true)?;
            if b.is_none() && e.is_none() {
                return Ok(None);
            }
            Ok(Some(pool.pow(b.unwrap_or(base), e.unwrap_or(exp))))
        }
        ExprData::Func { name, args } if args.len() == 1 && name == "log" && !in_exponent => {
            let a = split_logs(args[0], ctx, false)?.unwrap_or(args[0]);
            let Some(lt) = leading_term(a, ctx)? else {
                return Ok(None);
            };
            if is_zero(lt.power, pool) && lt.coeff == pool.integer(1_i32) {
                return Ok((a != args[0]).then(|| pool.func("log".to_string(), vec![a])));
            }
            if !log_coefficient_is_positive(lt.coeff, ctx) {
                return Ok(None);
            }
            let rest = expand_in_w(
                pool.mul(vec![
                    a,
                    pool.pow(w, pool.mul(vec![pool.integer(-1_i32), lt.power])),
                    pool.pow(lt.coeff, pool.integer(-1_i32)),
                ]),
                w,
                pool,
            );
            Ok(Some(pool.add(vec![
                pool.mul(vec![lt.power, ctx.h_omega]),
                pool.func("log".to_string(), vec![lt.coeff]),
                pool.func("log".to_string(), vec![rest]),
            ])))
        }
        ExprData::Func { name, args } => {
            let in_exp = in_exponent || name == "exp";
            Ok(rebuild(&args, in_exp)?.map(|v| pool.func(name.clone(), v)))
        }
        _ => Ok(None),
    }
}

/// Expand `e` and merge the powers of `w` in every product, so `w^{-c}·w^{c}`
/// with a symbolic `c` is `1` and `(w^{-c}·w^{c} + w^{1-c}) − 1` cancels —
/// which the general simplifier does not do for symbolic exponents.
fn expand_in_w(e: ExprId, w: ExprId, pool: &ExprPool) -> ExprId {
    fn merge(e: ExprId, w: ExprId, pool: &ExprPool) -> ExprId {
        match pool.get(e) {
            ExprData::Add(xs) => pool.add(xs.iter().map(|&x| merge(x, w, pool)).collect()),
            ExprData::Mul(xs) => {
                let mut exps = Vec::new();
                let mut others = Vec::new();
                for &x in &xs {
                    let x = merge(x, w, pool);
                    match pool.get(x) {
                        _ if x == w => exps.push(pool.integer(1_i32)),
                        ExprData::Pow { base, exp } if base == w => exps.push(exp),
                        ExprData::Integer(n) if n.0 == 1 => {}
                        _ => others.push(x),
                    }
                }
                if !exps.is_empty() {
                    let total = simplify(pool.add(exps), pool).value;
                    if !is_zero(total, pool) {
                        others.push(pool.pow(w, total));
                    }
                }
                match others.len() {
                    0 => pool.integer(1_i32),
                    1 => others[0],
                    _ => pool.mul(others),
                }
            }
            ExprData::Pow { base, exp } => {
                let b = merge(base, w, pool);
                if matches!(pool.get(b), ExprData::Integer(n) if n.0 == 1) {
                    return b;
                }
                pool.pow(b, exp)
            }
            _ => e,
        }
    }
    let expanded = crate::simplify::simplify_expanded(e, pool).value;
    simplify(merge(expanded, w, pool), pool).value
}

/// `log c` is only real for `c > 0`, and nothing here may assume it:
/// `log(−k·e^{−t})` is not `log(−k) − t` for a `k` of unknown sign.
fn log_coefficient_is_positive(c: ExprId, ctx: &OmegaCtx<'_>) -> bool {
    let pool = ctx.pool;
    let positive = if contains_var(c, ctx.var, pool) {
        let to_inf = limit(
            c,
            ctx.var,
            pool.pos_infinity(),
            LimitDirection::Bidirectional,
            pool,
        )
        .is_ok_and(|l| is_pos_inf(l, pool));
        to_inf || sign_of_coeff_at_inf(c, ctx.var, pool) == Some(1)
    } else {
        crate::calculus::limits::quantity_sign(c, pool) == Some(crate::simplify::Sign::Positive)
    };
    if !positive {
        crate::calculus::limits::note_missing_assumption(c, ctx.var, pool);
    }
    positive
}

/// Leading term of `exp(arg)`.
fn leading_term_exp(arg: ExprId, ctx: &OmegaCtx<'_>) -> Result<Option<LeadingTerm>, LimitError> {
    let pool = ctx.pool;
    let Some(lt) = leading_term(arg, ctx)? else {
        return Ok(None);
    };
    let Some(sign) = power_sign(lt.power, pool) else {
        return Ok(None);
    };
    if sign > 0 {
        // exp(o(1)) = 1 + o(1).
        return Ok(Some(LeadingTerm {
            coeff: pool.integer(1_i32),
            power: pool.integer(0_i32),
            log_derived: false,
        }));
    }
    if sign == 0 && !lt.log_derived {
        // exp(c + o(1)) = exp(c)·(1 + o(1)); exp(c) is lower class because the
        // node itself was not in the mrv set.
        return Ok(Some(LeadingTerm {
            coeff: simplify(pool.func("exp".to_string(), vec![lt.coeff]), pool).value,
            power: pool.integer(0_i32),
            log_derived: false,
        }));
    }
    // Either a faster-than-ω exponential the mrv step missed, or `exp` of an
    // `h_ω` — possibly ω itself. Neither is decided here.
    Ok(None)
}

/// Leading term from a Taylor/Laurent expansion in `w`, used only where the
/// expansion is sound (see [`series_is_sound`]).
fn series_leading_term(
    expr: ExprId,
    ctx: &OmegaCtx<'_>,
) -> Result<Option<LeadingTerm>, LimitError> {
    let pool = ctx.pool;
    if !series_is_sound(expr, ctx.w, pool) {
        return Ok(None);
    }
    let zero = pool.integer(0_i32);
    let expansion = match local_expansion(expr, ctx.w, zero, SERIES_TERMS, pool) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };
    checkpoint(pool)?;
    let base_val = expansion.valuation;
    for (k, coeff) in expansion.coeffs.iter().enumerate() {
        // `0·(pole)` simplifies to `0`; it is not a zero coefficient.
        if crate::calculus::limits::has_untamed_pole(*coeff, pool) {
            return Ok(None);
        }
        let c = simplify(*coeff, pool).value;
        if contains_var(c, ctx.w, pool) || crate::calculus::limits::has_unusable_node(c, pool) {
            return Ok(None);
        }
        if !is_zero(c, pool) {
            return Ok(Some(LeadingTerm {
                coeff: c,
                power: pool.integer(base_val + k as i32),
                log_derived: false,
            }));
        }
    }
    Ok(None)
}

/// `true` when a Taylor/Laurent expansion of `expr` in `w` at `0` means what it
/// says: `w` occurs only in polynomial / integer-power position and inside the
/// arguments of entire functions (`exp`, `sin`, `cos`, `sinh`, `cosh`) that
/// have no pole at `w = 0`. A `log`, a fractional power, or any other function
/// of a `w`-dependent argument is a possible branch point there, where a
/// Taylor loop reports nonsense or never terminates.
fn series_is_sound(expr: ExprId, w: ExprId, pool: &ExprPool) -> bool {
    if !contains_var(expr, w, pool) {
        return true;
    }
    match pool.get(expr) {
        ExprData::Symbol { .. } => true,
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().all(|&x| series_is_sound(x, w, pool)),
        ExprData::Pow { base, exp } => {
            matches!(pool.get(exp), ExprData::Integer(_)) && series_is_sound(base, w, pool)
        }
        ExprData::Func { name, args }
            if args.len() == 1 && matches!(name.as_str(), "log" | "ln") =>
        {
            // `log` is analytic wherever its argument is nonzero: sound when
            // the argument is, and its value at `w = 0` is a nonzero number.
            let a = args[0];
            if !series_is_sound(a, w, pool) || has_pole_in(a, w, pool) {
                return false;
            }
            let mut m = HashMap::with_capacity(1);
            m.insert(w, pool.integer(0_i32));
            let raw = crate::kernel::subs(a, &m, pool);
            if crate::calculus::limits::has_untamed_pole(raw, pool) {
                return false;
            }
            let at0 = simplify(raw, pool).value;
            !is_zero(at0, pool) && !crate::calculus::limits::has_unusable_node(at0, pool)
        }
        ExprData::Func { name, args } => {
            matches!(name.as_str(), "exp" | "sin" | "cos" | "sinh" | "cosh")
                && args
                    .iter()
                    .all(|&a| series_is_sound(a, w, pool) && !has_pole_in(a, w, pool))
        }
        _ => false,
    }
}

/// `true` when `expr` has a negative power of a `w`-dependent base, i.e. a
/// possible pole at `w = 0` — inside an entire function that is an essential
/// singularity, which no Taylor expansion represents.
fn has_pole_in(expr: ExprId, w: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let neg = match pool.get(exp) {
                ExprData::Integer(n) => n.0 < 0,
                ExprData::Rational(r) => r.0 < 0,
                _ => true,
            };
            (neg && contains_var(base, w, pool)) || has_pole_in(base, w, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| has_pole_in(x, w, pool)),
        ExprData::Func { args, .. } => args.iter().any(|&a| has_pole_in(a, w, pool)),
        _ => false,
    }
}

/// Sample points (in `var`) for the numeric sanity checks below: large enough
/// to be "towards +∞" for the coefficients Gruntz produces, small enough that
/// an `exp` of a lower-class argument does not overflow.
const SAMPLES: [f64; 3] = [7.389, 13.27, 23.71];

fn sample(e: ExprId, ctx: &OmegaCtx<'_>) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(SAMPLES.len());
    for &t in &SAMPLES {
        let mut env = HashMap::new();
        env.insert(ctx.var, t);
        out.push(crate::jit::eval_interp(e, &env, ctx.pool)?);
    }
    Some(out)
}

/// `true` when `e` (free of ω) evaluates to (numerically) zero at every sample
/// point — a cancellation `simplify` did not see. `false` when it cannot be
/// evaluated (a free parameter), which keeps the structural answer.
fn vanishes_numerically(e: ExprId, ctx: &OmegaCtx<'_>) -> bool {
    match sample(e, ctx) {
        Some(vs) => vs.iter().all(|v| v.is_finite() && v.abs() < 1e-12),
        None => false,
    }
}

fn power_diff(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    simplify(
        pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]),
        pool,
    )
    .value
}

/// Sign (`-1`, `0`, `1`) of a power of ω — a constant, rational or symbolic —
/// from its value or the assumptions; `None` when nothing establishes it.
///
/// A symbolic power is a ratio of rates (`k₂/k₁`) or a difference of two
/// (`k₂/k₁ − 1`); for the latter the sign algebra needs the positive
/// denominators cleared first (`(k₂ − k₁)` is what `k₂ > k₁` speaks about).
fn power_sign(p: ExprId, pool: &ExprPool) -> Option<i8> {
    use crate::simplify::Sign;
    if let Some(r) = expr_as_rational(p, pool) {
        return Some(r.cmp0() as i8);
    }
    let to_i8 = |s: Sign| match s {
        Sign::Positive => 1,
        Sign::Negative => -1,
        Sign::Zero => 0,
    };
    if let Some(s) = crate::calculus::limits::quantity_sign(p, pool) {
        return Some(to_i8(s));
    }
    let mut dens = Vec::new();
    positive_denominators(p, pool, &mut dens);
    if dens.is_empty() {
        return None;
    }
    dens.push(p);
    let cleared = crate::simplify::simplify_expanded(pool.mul(dens), pool).value;
    crate::calculus::limits::quantity_sign(cleared, pool).map(to_i8)
}

/// Bases of negative integer powers in `e` whose sign is assumed positive.
fn positive_denominators(e: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    match pool.get(e) {
        ExprData::Pow { base, exp } => {
            if let ExprData::Integer(n) = pool.get(exp) {
                if n.0 < 0
                    && crate::simplify::assumed_sign(base, pool)
                        == Some(crate::simplify::Sign::Positive)
                {
                    let d = pool.pow(base, pool.integer(-n.0.clone()));
                    if !out.contains(&d) {
                        out.push(d);
                    }
                }
            }
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            for x in xs {
                positive_denominators(x, pool, out);
            }
        }
        _ => {}
    }
}

fn expr_as_rational(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn contains_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|a| contains_var(*a, var, pool)),
        ExprData::Pow { base, exp } => {
            contains_var(base, var, pool) || contains_var(exp, var, pool)
        }
        ExprData::Func { args, .. } => args.iter().any(|a| contains_var(*a, var, pool)),
        _ => false,
    }
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0)
        || matches!(pool.get(e), ExprData::Rational(r) if r.0 == 0)
}

fn is_pos_inf(e: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::pool::POS_INFINITY_SYMBOL;
    matches!(pool.get(e), ExprData::Symbol { name, .. } if name == POS_INFINITY_SYMBOL)
}

fn is_neg_inf(e: ExprId, pool: &ExprPool) -> bool {
    let ExprData::Mul(args) = pool.get(e) else {
        return false;
    };
    let m1 = pool.integer(-1_i32);
    args.len() == 2
        && ((args[0] == m1 && is_pos_inf(args[1], pool))
            || (args[1] == m1 && is_pos_inf(args[0], pool)))
}

fn signed_infinity(pool: &ExprPool, sign: i8) -> ExprId {
    if sign < 0 {
        pool.mul(vec![pool.integer(-1_i32), pool.pos_infinity()])
    } else {
        pool.pos_infinity()
    }
}

/// Determine the sign of `coeff` as `var → +∞` (for the leading-term decision),
/// or `None` when nothing establishes it.
///
/// The MRV rewrite reduces the whole problem to `coeff · ωᵖ` with `ω → 0⁺`, so
/// for `p < 0` this sign *is* the answer: it chooses between `+∞` and `−∞`.
/// It used to end in `.unwrap_or(1)` — "assume positive" — which for a
/// coefficient carrying a free parameter (`lim_{t→∞} (−1/k)·e^{kt}`, a
/// perfectly ordinary shape for an improper integral's antiderivative) is a
/// coin flip reported as a result. `None` now means the caller must decline;
/// [`crate::calculus::limits`] records which parameter was missing.
fn sign_of_coeff_at_inf(coeff: ExprId, var: ExprId, pool: &ExprPool) -> Option<i8> {
    // Try computing lim(coeff) and reading its sign
    if let Ok(l) = limit(
        coeff,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    ) {
        if let Some(s) = structural_sign(l, pool) {
            return Some(s);
        }
        // `lim coeff = 0` says nothing about *which side* of zero the
        // coefficient approaches from, and that side is the whole question:
        // for `e^x/x` the MRV rewrite leaves `coeff = x⁻¹`, whose limit is 0
        // and whose eventual sign is what decides between `+∞` and `−∞`.
        //
        // The reciprocal settles it without guessing. `1/coeff → +∞` means
        // `1/coeff > 0` from some point on, hence `coeff > 0` from some point
        // on — and `lim 1/x⁻¹ = lim x = +∞` is a limit this engine computes
        // directly. Only taken when the limit really is zero, and only one
        // level deep, so it cannot ping-pong with itself.
        if is_zero(l, pool) {
            let reciprocal = simplify(pool.pow(coeff, pool.integer(-1_i32)), pool).value;
            if let Ok(r) = limit(
                reciprocal,
                var,
                pool.pos_infinity(),
                LimitDirection::Bidirectional,
                pool,
            ) {
                if is_pos_inf(r, pool) {
                    return Some(1);
                }
                if is_neg_inf(r, pool) {
                    return Some(-1);
                }
            }
        }
    }
    structural_sign(coeff, pool)
}

/// Sign of a `var`-free leading coefficient from structure, the ambient
/// assumptions, or its own numeric value; `None` when it is zero or
/// undetermined.
fn structural_sign(e: ExprId, pool: &ExprPool) -> Option<i8> {
    match crate::calculus::limits::coefficient_sign(e, pool)? {
        crate::simplify::Sign::Positive => Some(1),
        crate::simplify::Sign::Negative => Some(-1),
        crate::simplify::Sign::Zero => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculus::limits::{limit, LimitDirection};
    use crate::kernel::Domain;

    fn oo(pool: &ExprPool) -> ExprId {
        pool.pos_infinity()
    }

    #[test]
    fn gruntz_exp_neg_x_to_zero() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(-x) = 0
        let expr = p.func("exp".to_string(), vec![p.mul(vec![p.integer(-1_i32), x])]);
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(0_i32), "exp(-x): {}", p.display(r));
    }

    #[test]
    fn gruntz_x_times_exp_neg_x() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} x * exp(-x) = 0
        let neg_x = p.mul(vec![p.integer(-1_i32), x]);
        let expr = simplify(p.mul(vec![x, p.func("exp".to_string(), vec![neg_x])]), &p).value;
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(0_i32), "x*exp(-x): {}", p.display(r));
    }

    #[test]
    fn gruntz_exp_x_over_x_squared_is_inf() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(x) / x² = +∞
        let expr = simplify(
            p.mul(vec![
                p.func("exp".to_string(), vec![x]),
                p.pow(x, p.integer(-2_i32)),
            ]),
            &p,
        )
        .value;
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, oo(&p), "exp(x)/x²: {}", p.display(r));
    }

    #[test]
    fn gruntz_ratio_exp2x_over_exp3x_is_zero() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(2x) / exp(3x) = 0
        let e2x = p.func("exp".to_string(), vec![p.mul(vec![p.integer(2_i32), x])]);
        let e3x = p.func("exp".to_string(), vec![p.mul(vec![p.integer(3_i32), x])]);
        let expr = simplify(p.mul(vec![e2x, p.pow(e3x, p.integer(-1_i32))]), &p).value;
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(0_i32), "exp(2x)/exp(3x): {}", p.display(r));
    }

    #[test]
    fn gruntz_nested_exp_ratio() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(x+1) / exp(x) = exp(1)
        let e_x1 = p.func("exp".to_string(), vec![p.add(vec![x, p.integer(1_i32)])]);
        let e_x = p.func("exp".to_string(), vec![x]);
        let expr = simplify(p.mul(vec![e_x1, p.pow(e_x, p.integer(-1_i32))]), &p).value;
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        let expected = simplify(p.func("exp".to_string(), vec![p.integer(1_i32)]), &p).value;
        let r_s = simplify(r, &p).value;
        assert_eq!(r_s, expected, "exp(x+1)/exp(x): {}", p.display(r));
    }

    fn exp(p: &ExprPool, a: ExprId) -> ExprId {
        p.func("exp".to_string(), vec![a])
    }

    fn log(p: &ExprPool, a: ExprId) -> ExprId {
        p.func("log".to_string(), vec![a])
    }

    fn lin(p: &ExprPool, a: i64, x: ExprId) -> ExprId {
        p.mul(vec![p.integer(a), x])
    }

    fn over_x(p: &ExprPool, e: ExprId, x: ExprId) -> ExprId {
        p.mul(vec![e, p.pow(x, p.integer(-1_i32))])
    }

    fn lim_inf(p: &ExprPool, e: ExprId, x: ExprId) -> Result<ExprId, LimitError> {
        limit(e, x, oo(p), LimitDirection::Bidirectional, p).map(|r| simplify(r, p).value)
    }

    /// report10-21 N3: `log(e^{−2t})/t → −2`, not 0.
    #[test]
    fn log_of_exp_over_x_is_the_rate() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        for a in [-3_i64, -2, -1, 1, 2, 5] {
            let e = over_x(&p, log(&p, exp(&p, lin(&p, a, x))), x);
            let r = lim_inf(&p, e, x).unwrap();
            assert_eq!(r, p.integer(a), "log(exp({a}x))/x: {}", p.display(r));
        }
    }

    /// report10-21 N3: `log(e^{−2t} + e^{−3t})/t → −2`.
    #[test]
    fn log_of_exponential_sum_over_x_is_the_top_rate() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let s = p.add(vec![exp(&p, lin(&p, -2, x)), exp(&p, lin(&p, -3, x))]);
        let r = lim_inf(&p, over_x(&p, log(&p, s), x), x).unwrap();
        assert_eq!(r, p.integer(-2_i32), "{}", p.display(r));
        // log(e^x + 1)/x → 1 — report10-21 C1 (ran away in the Taylor loop).
        let s = p.add(vec![exp(&p, x), p.integer(1_i32)]);
        let r = lim_inf(&p, over_x(&p, log(&p, s), x), x).unwrap();
        assert_eq!(r, p.integer(1_i32), "{}", p.display(r));
    }

    /// The lower-class factor `exp(h_e − c·h_ω)` survives the rewrite:
    /// `e^{1−3x}/(e^{−3x} + e^{−4x}) → e`.
    #[test]
    fn mrv_rewrite_keeps_the_lower_class_factor() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let num = exp(&p, p.add(vec![p.integer(1_i32), lin(&p, -3, x)]));
        let den = p.add(vec![exp(&p, lin(&p, -3, x)), exp(&p, lin(&p, -4, x))]);
        let e = p.mul(vec![num, p.pow(den, p.integer(-1_i32))]);
        let r = lim_inf(&p, e, x).unwrap();
        let v = crate::jit::eval_interp(r, &HashMap::new(), &p).unwrap();
        assert!((v - std::f64::consts::E).abs() < 1e-12, "{}", p.display(r));
    }

    /// A rate of unknown order against another refuses instead of picking one.
    #[test]
    fn symbolic_rates_of_unknown_order_refuse() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let a = p.symbol("a", Domain::Positive);
        let b = p.symbol("b", Domain::Positive);
        let neg = |k: ExprId| p.mul(vec![p.integer(-1_i32), k, x]);
        let s = p.add(vec![
            exp(&p, neg(a)),
            p.mul(vec![p.integer(-1_i32), exp(&p, neg(b))]),
        ]);
        let e = over_x(&p, log(&p, s), x);
        assert!(lim_inf(&p, e, x).is_err());
    }

    /// `log ω` must never reach a Taylor expansion in ω.
    #[test]
    fn series_soundness_guard() {
        let p = ExprPool::new();
        let w = p.symbol("w", Domain::Positive);
        assert!(series_is_sound(p.add(vec![w, p.integer(1_i32)]), w, &p));
        assert!(series_is_sound(exp(&p, w), w, &p));
        assert!(!series_is_sound(log(&p, w), w, &p));
        assert!(!series_is_sound(p.pow(w, p.rational(1, 2)), w, &p));
        assert!(!series_is_sound(
            exp(&p, p.pow(w, p.integer(-1_i32))),
            w,
            &p
        ));
    }

    #[test]
    fn gruntz_nested_exp_exp_x_is_inf() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(exp(x)) = +∞
        let inner = p.func("exp".to_string(), vec![x]);
        let expr = p.func("exp".to_string(), vec![inner]);
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, oo(&p), "exp(exp(x)): {}", p.display(r));
    }
}
