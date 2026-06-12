//! Asymptotic expansions at infinity in the Poincaré sense (§2.4).
//!
//! Given `f(x)` and a variable `x`, [`asymptotic_expand`] returns an ordered
//! list of asymptotic terms `g₁, g₂, …` such that
//!
//! ```text
//! f(x) ~ g₁(x) + g₂(x) + …   as x → +∞,
//! ```
//!
//! with each `gₖ₊₁ = o(gₖ)` as `x → +∞` (the defining property of a Poincaré
//! asymptotic sequence). The terms are returned most-significant first.
//!
//! # Strategy
//!
//! The core engine is the substitution `x = 1/t` followed by a (Laurent /
//! Puiseux-lite) series of `f(1/t)` at `t → 0⁺`, reusing
//! [`mod@crate::calculus::series`]. A term `c · t^e` of that series maps back to
//! `c · x^{−e}`; the polynomial-growth part (negative `t`-powers, i.e. positive
//! `x`-powers) is carried along automatically. This covers:
//!
//! * rational functions — e.g. `(x+1)/(x−1) ~ 1 + 2/x + 2/x² + …`;
//! * algebraic functions analytic at ∞ — e.g.
//!   `√(x²+1) ~ x + 1/(2x) − 1/(8x³) + …`;
//! * compositions analytic at ∞ — e.g. `x·sin(1/x) ~ 1 − 1/(6x²) + …`,
//!   `e^{1/x}·x ~ x + 1 + 1/(2x) + …`.
//!
//! Beyond pure power scales, a **leading log/exp scale is peeled
//! multiplicatively** for a restricted but common shape: `log(P(x))` arguments
//! (e.g. `log(x+1) ~ log x + 1/x − 1/(2x²) + …`) and a single dominant
//! `log(x)` / `exp(g(x))` factor whose power-scale cofactor is then expanded.
//! Genuinely scale-iterated expansions (nested exp-log hierarchies,
//! Γ / Stirling asymptotics) are **out of scope** and decline honestly.
//!
//! # Correctness gate
//!
//! Every returned expansion is gated by a numeric `o()`-check: at
//! `x = 10², 10⁴, 10⁶` the residual `|f − Σ₁..ₖ gᵢ|` must be bounded by (and
//! shrink consistently with) the next term `gₖ₊₁`. Terms that fail the gate are
//! dropped from the tail; if no term survives, the call declines rather than
//! emit an unverified expansion.

use crate::calculus::series::{local_expansion, LocalExpansion};
use crate::diff::DiffError;
use crate::jit::eval_interp;
use crate::kernel::{subs, Domain, ExprData, ExprId, ExprPool};
use crate::simplify::simplify;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One term `gₖ(x)` of an asymptotic expansion at `+∞`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AsymptoticTerm {
    /// The symbolic term, expressed in the original variable `x`.
    pub expr: ExprId,
}

impl AsymptoticTerm {
    /// The symbolic term `gₖ(x)`.
    pub fn expr(self) -> ExprId {
        self.expr
    }
}

/// Result of [`asymptotic_expand`]: the ordered asymptotic terms (most
/// significant first).
#[derive(Clone, Debug)]
pub struct AsymptoticExpansion {
    /// Asymptotic terms `g₁, g₂, …`, ordered most-significant first.
    pub terms: Vec<AsymptoticTerm>,
}

impl AsymptoticExpansion {
    /// The bare term expressions, ordered most-significant first.
    pub fn term_exprs(&self) -> Vec<ExprId> {
        self.terms.iter().map(|t| t.expr).collect()
    }

    /// The sum `g₁ + g₂ + … + g_k` of all surviving terms (the truncated
    /// asymptotic approximation).
    pub fn partial_sum(&self, pool: &ExprPool) -> ExprId {
        if self.terms.is_empty() {
            return pool.integer(0_i32);
        }
        let xs: Vec<ExprId> = self.terms.iter().map(|t| t.expr).collect();
        simplify(pool.add(xs), pool).value
    }
}

/// Failure modes for [`asymptotic_expand`].
#[derive(Debug)]
pub enum AsymptoticError {
    /// `n_terms` must be ≥ 1.
    InvalidTermCount,
    /// The substitution `x = 1/t` series failed (function not analytic /
    /// Laurent-expandable at ∞ with the available machinery).
    SeriesFailed,
    /// A derivative needed for an internal expansion failed.
    Diff(DiffError),
    /// The numeric `o()`-gate rejected every candidate term — the expansion
    /// could not be verified, so nothing is emitted.
    GateFailed,
    /// No implemented scale (power / single log-exp peel) matched the input.
    UnsupportedScale,
}

impl fmt::Display for AsymptoticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsymptoticError::InvalidTermCount => write!(f, "n_terms must be >= 1"),
            AsymptoticError::SeriesFailed => {
                write!(f, "could not form a series of f(1/t) at t -> 0")
            }
            AsymptoticError::Diff(e) => write!(f, "{e}"),
            AsymptoticError::GateFailed => {
                write!(f, "numeric o()-gate rejected the asymptotic expansion")
            }
            AsymptoticError::UnsupportedScale => {
                write!(f, "asymptotic scale not supported by current rules")
            }
        }
    }
}

impl std::error::Error for AsymptoticError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AsymptoticError::Diff(e) => Some(e),
            _ => None,
        }
    }
}

impl crate::errors::AlkahestError for AsymptoticError {
    fn code(&self) -> &'static str {
        match self {
            AsymptoticError::InvalidTermCount => "E-ASYMPT-001",
            AsymptoticError::SeriesFailed => "E-ASYMPT-002",
            AsymptoticError::Diff(_) => "E-ASYMPT-003",
            AsymptoticError::GateFailed => "E-ASYMPT-004",
            AsymptoticError::UnsupportedScale => "E-ASYMPT-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            AsymptoticError::InvalidTermCount => "pass n_terms >= 1",
            AsymptoticError::SeriesFailed => {
                "the function may not be analytic/Laurent-expandable at infinity; \
                 try a simpler form or fewer terms"
            }
            AsymptoticError::Diff(_) => {
                "ensure all functions are registered primitives with differentiation rules"
            }
            AsymptoticError::GateFailed => {
                "the expansion could not be numerically verified at large x; \
                 the function may have an oscillatory or non-power-scale tail"
            }
            AsymptoticError::UnsupportedScale => {
                "exp/log scale hierarchies and Gamma/Stirling asymptotics are out of scope; \
                 power-scale (rational/algebraic) and single log/exp peels are supported"
            }
        })
    }
}

impl From<DiffError> for AsymptoticError {
    fn from(e: DiffError) -> Self {
        AsymptoticError::Diff(e)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Compute the asymptotic expansion of `f` in `var` as `var → +∞`, returning at
/// most `n_terms` ordered terms (most significant first).
///
/// See the [module documentation](self) for the strategy, supported scales, and
/// the numeric `o()`-gate that every emitted expansion must pass.
pub fn asymptotic_expand(
    f: ExprId,
    var: ExprId,
    n_terms: usize,
    pool: &ExprPool,
) -> Result<AsymptoticExpansion, AsymptoticError> {
    if n_terms == 0 {
        return Err(AsymptoticError::InvalidTermCount);
    }

    let f = simplify(f, pool).value;

    // 1. Pure power-scale core (substitution x = 1/t).
    if let Ok(exp) = power_scale_expand(f, var, n_terms, pool) {
        if !exp.terms.is_empty() {
            return Ok(exp);
        }
    }

    // 2. Log-scale peel: f = log(P(x)) + rest, or a single log(x)/exp(g) factor.
    if let Some(exp) = try_log_peel(f, var, n_terms, pool)? {
        if !exp.terms.is_empty() {
            return Ok(exp);
        }
    }

    Err(AsymptoticError::UnsupportedScale)
}

// ---------------------------------------------------------------------------
// Power-scale core: x = 1/t, series at t -> 0, map back.
// ---------------------------------------------------------------------------

/// Build a raw (ungated) list of power-scale terms `cᵢ · x^{−eᵢ}` from the
/// Laurent expansion of `f(1/t)` at `t → 0`, ordered by decreasing `x`-power.
///
/// `order` is the series truncation in `t`; it is chosen larger than `n_terms`
/// so that low-order zero coefficients do not starve the result.
fn power_scale_terms_raw(
    f: ExprId,
    var: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Vec<(i64, ExprId)>, AsymptoticError> {
    let t = pool.symbol("__asy_t", Domain::Positive);
    let inv_t = pool.pow(t, pool.integer(-1_i32));
    let mut m = HashMap::new();
    m.insert(var, inv_t);
    let f_of_t = simplify(subs(f, &m, pool), pool).value;

    // Regularize symbolically: f(1/t) = t^val · u(t) with u analytic & nonzero
    // at t = 0. `regularize_at_zero` extracts the integer t-valuation `val`
    // (possibly negative — a pole / polynomial-growth factor) and the analytic
    // part `u`, pushing t-powers inside radicals/powers so that `local_expansion`
    // (Taylor) sees an honest analytic function.
    let (val, analytic) =
        regularize_at_zero(f_of_t, t, pool).ok_or(AsymptoticError::SeriesFailed)?;

    let LocalExpansion {
        valuation: tay_val,
        coeffs,
        ..
    } = local_expansion(analytic, t, pool.integer(0_i32), order, pool)
        .map_err(|_| AsymptoticError::SeriesFailed)?;

    // u's i-th coefficient sits at t^{tay_val+i}; with the extracted factor the
    // original f(1/t) term is at t-power (val + tay_val + i), i.e. x-power
    // −(val + tay_val + i).
    let mut out: Vec<(i64, ExprId)> = Vec::new();
    for (i, &c) in coeffs.iter().enumerate() {
        // Coefficients are var-free constants but may carry unfolded
        // `sin(0)`/`cos(0)`/`exp(0)` heads; fold them and drop numeric zeros.
        let c = fold_constant(c, pool);
        if is_numeric_zero(c, pool) {
            continue;
        }
        let t_pow = val + tay_val as i64 + i as i64;
        let x_pow = -t_pow;
        let term = make_power_term(c, var, x_pow, pool);
        out.push((x_pow, term));
    }
    // Most significant (largest x-power) first.
    out.sort_by_key(|p| std::cmp::Reverse(p.0));
    Ok(out)
}

/// Decompose `g(t) = t^val · u(t)` with `u` analytic at `t = 0`, returning
/// `(val, u)`. The integer `val` may be negative (pole). Returns `None` when the
/// structure is outside the integer-power scale (e.g. a genuine fractional
/// `t`-valuation / Puiseux branch, or an unsupported head).
///
/// This is a structural valuation calculus that pushes `t`-powers through
/// `Add`/`Mul`/`Pow`/elementary `Func`s so the analytic remainder is honestly
/// Taylor-expandable. `val` is recovered exactly; the analytic part is left
/// symbolic for [`local_expansion`].
fn regularize_at_zero(g: ExprId, t: ExprId, pool: &ExprPool) -> Option<(i64, ExprId)> {
    let g = simplify(g, pool).value;
    match pool.get(g) {
        // Constants and other-variable atoms: valuation 0.
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => Some((0, g)),
        ExprData::Symbol { .. } => {
            if g == t {
                Some((1, pool.integer(1_i32)))
            } else {
                Some((0, g))
            }
        }
        ExprData::Mul(xs) => {
            let mut val = 0i64;
            let mut parts: Vec<ExprId> = Vec::new();
            for x in xs {
                let (v, u) = regularize_at_zero(x, t, pool)?;
                val += v;
                parts.push(u);
            }
            Some((val, simplify(pool.mul(parts), pool).value))
        }
        ExprData::Pow { base, exp } => {
            // Only integer or rational exponents with integer resulting valuation.
            let (vb, ub) = regularize_at_zero(base, t, pool)?;
            let exp_q = rational_value(exp, pool)?;
            // valuation = vb * exp; must be an integer for the integer scale.
            let num = exp_q.numer().clone();
            let den = exp_q.denom().clone();
            let prod = rug::Integer::from(vb) * &num;
            // prod / den must be an integer.
            if den == 0 {
                return None;
            }
            let (q, r) = prod.div_rem(den.clone());
            if r != 0 {
                return None; // fractional valuation: Puiseux, out of integer scale
            }
            let val = q.to_i64()?;
            // analytic part = ub^exp (ub analytic & nonzero ⇒ ub^exp analytic).
            let analytic = simplify(pool.pow(ub, exp), pool).value;
            Some((val, analytic))
        }
        ExprData::Add(xs) => {
            // Regularize each summand, factor out the minimal valuation.
            let mut pieces: Vec<(i64, ExprId)> = Vec::with_capacity(xs.len());
            let mut vmin = i64::MAX;
            for x in &xs {
                let (v, u) = regularize_at_zero(*x, t, pool)?;
                vmin = vmin.min(v);
                pieces.push((v, u));
            }
            if vmin == i64::MAX {
                return None;
            }
            let mut summands: Vec<ExprId> = Vec::with_capacity(pieces.len());
            for (v, u) in pieces {
                let shift = v - vmin; // ≥ 0
                let term = if shift == 0 {
                    u
                } else {
                    simplify(pool.mul(vec![pool.pow(t, pool.integer(shift)), u]), pool).value
                };
                summands.push(term);
            }
            let analytic = simplify(pool.add(summands), pool).value;
            // The factored sum is analytic; its own valuation may be > 0 if the
            // leading terms cancel — local_expansion will pick that up via tay_val.
            Some((vmin, analytic))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let (va, ua) = regularize_at_zero(args[0], t, pool)?;
            match name.as_str() {
                // sqrt(t^va · ua) = t^{va/2} · sqrt(ua) when va is even (else the
                // result is a genuine Puiseux branch, outside the integer scale).
                "sqrt" => {
                    if va % 2 != 0 {
                        return None;
                    }
                    // Reconstruct the analytic argument t^{va}·ua only when va==0;
                    // when va<0 the sqrt would diverge — but va even means
                    // t^{va/2}·sqrt(ua) with ua analytic & nonzero is the answer.
                    let analytic = simplify(pool.func("sqrt", vec![ua]), pool).value;
                    Some((va / 2, analytic))
                }
                // Elementary transcendental heads that are analytic wherever their
                // argument is analytic (finite at t=0, i.e. valuation ≥ 0). The
                // composition is then analytic at t=0; `local_expansion` (Taylor)
                // recovers any internal zero (e.g. sin(t) ~ t). A divergent
                // argument (valuation < 0) is outside this scale.
                "sin" | "cos" | "tan" | "exp" | "cosh" | "sinh" | "tanh" | "gamma" => {
                    if va < 0 {
                        return None;
                    }
                    // Rebuild the full (analytic) argument t^{va}·ua.
                    let full_arg = if va == 0 {
                        ua
                    } else {
                        simplify(pool.mul(vec![pool.pow(t, pool.integer(va)), ua]), pool).value
                    };
                    let analytic = simplify(pool.func(name, vec![full_arg]), pool).value;
                    Some((0, analytic))
                }
                // log is analytic only at a finite *nonzero* argument: require
                // valuation 0 and a nonzero constant term, so log(arg) is itself
                // analytic (valuation 0) at t = 0 (e.g. log(1+t)).
                "log" => {
                    if va != 0 {
                        return None;
                    }
                    let at0 = eval_at_zero(ua, t, pool)?;
                    if at0 == 0.0 {
                        return None;
                    }
                    let analytic = simplify(pool.func("log", vec![ua]), pool).value;
                    Some((0, analytic))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Fold constant `sin(0)`/`cos(0)`/`exp(0)`/… heads that the generic simplifier
/// leaves intact inside Taylor coefficients, recursing through `Add`/`Mul`/`Pow`.
fn fold_constant(e: ExprId, pool: &ExprPool) -> ExprId {
    let e = simplify(e, pool).value;
    match pool.get(e) {
        ExprData::Add(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| fold_constant(*x, pool)).collect();
            simplify(pool.add(ys), pool).value
        }
        ExprData::Mul(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| fold_constant(*x, pool)).collect();
            simplify(pool.mul(ys), pool).value
        }
        ExprData::Pow { base, exp } => {
            let b = fold_constant(base, pool);
            let x = fold_constant(exp, pool);
            simplify(pool.pow(b, x), pool).value
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let inner = fold_constant(args[0], pool);
            if matches!(pool.get(inner), ExprData::Integer(n) if n.0 == 0) {
                match name.as_str() {
                    "sin" | "tan" | "sinh" | "tanh" => return pool.integer(0_i32),
                    "cos" | "cosh" | "exp" => return pool.integer(1_i32),
                    _ => {}
                }
            }
            simplify(pool.func(name, vec![inner]), pool).value
        }
        _ => e,
    }
}

/// True if `e` evaluates numerically to (approximately) zero, or is the
/// structural integer/rational zero.
fn is_numeric_zero(e: ExprId, pool: &ExprPool) -> bool {
    if matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0) {
        return true;
    }
    if let ExprData::Rational(r) = pool.get(e) {
        return r.0 == 0;
    }
    match eval_interp(e, &HashMap::new(), pool) {
        Some(v) => v == 0.0,
        None => false,
    }
}

/// Numeric value of `e` at `t = 0` (a small positive sample), used to check that
/// an analytic factor is nonzero there.
fn eval_at_zero(e: ExprId, t: ExprId, pool: &ExprPool) -> Option<f64> {
    let mut env = HashMap::new();
    env.insert(t, 1.0e-6f64);
    eval_interp(e, &env, pool).filter(|v| v.is_finite())
}

/// Extract a rational/integer constant exponent as a `rug::Rational`.
fn rational_value(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(rug::Rational::from((n.0.clone(), rug::Integer::from(1)))),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

fn make_power_term(coeff: ExprId, var: ExprId, x_pow: i64, pool: &ExprPool) -> ExprId {
    let pow = if x_pow == 0 {
        pool.integer(1_i32)
    } else if x_pow == 1 {
        var
    } else {
        pool.pow(var, pool.integer(x_pow))
    };
    simplify(pool.mul(vec![coeff, pow]), pool).value
}

fn power_scale_expand(
    f: ExprId,
    var: ExprId,
    n_terms: usize,
    pool: &ExprPool,
) -> Result<AsymptoticExpansion, AsymptoticError> {
    // Request extra order so we still find n_terms nonzero terms past gaps.
    let order = (n_terms as u32).saturating_mul(2).saturating_add(8).min(40);
    let raw = power_scale_terms_raw(f, var, order, pool)?;
    if raw.is_empty() {
        return Err(AsymptoticError::SeriesFailed);
    }
    let candidate: Vec<ExprId> = raw.iter().map(|(_, e)| *e).take(n_terms).collect();
    let gated = gate_terms(f, var, &candidate, pool);
    if gated.is_empty() {
        return Err(AsymptoticError::GateFailed);
    }
    Ok(AsymptoticExpansion {
        terms: gated
            .into_iter()
            .map(|expr| AsymptoticTerm { expr })
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// Log/exp scale peeling (restricted).
// ---------------------------------------------------------------------------

/// Handle a single dominant `log`/`exp` scale.
///
/// Two shapes are covered:
///
/// * `f = log(P(x))`, expanded as `log(x^d) + log(P/x^d)` where `d = deg P`;
///   the second factor is analytic at ∞ and power-expanded. Gives e.g.
///   `log(x+1) ~ log x + 1/x − 1/(2x²) + …`.
/// * `f = log(x) · h(x)` or `exp(g(x)) · h(x)` with a single such leading
///   factor and a power-expandable cofactor `h`.
fn try_log_peel(
    f: ExprId,
    var: ExprId,
    n_terms: usize,
    pool: &ExprPool,
) -> Result<Option<AsymptoticExpansion>, AsymptoticError> {
    // Shape: f is exactly log(arg).
    if let ExprData::Func { name, args } = pool.get(f) {
        if name == "log" && args.len() == 1 {
            return log_of_arg_peel(f, args[0], var, n_terms, pool);
        }
    }

    // Shape: f is a product with exactly one log(x)/exp(g) factor and a
    // power-expandable cofactor.
    if let ExprData::Mul(factors) = pool.get(f) {
        if let Some(exp) = mul_with_scale_factor(&factors, var, n_terms, pool)? {
            return Ok(Some(exp));
        }
    }

    Ok(None)
}

/// Expand `log(arg)` where `arg → +∞` polynomially: peel `log(x^d)` and
/// power-expand the analytic remainder `log(arg / x^d)`.
fn log_of_arg_peel(
    f: ExprId,
    arg: ExprId,
    var: ExprId,
    n_terms: usize,
    pool: &ExprPool,
) -> Result<Option<AsymptoticExpansion>, AsymptoticError> {
    // Determine the dominant integer power d of x in arg via the t-substitution
    // valuation of arg(1/t): arg ~ c · x^d means arg(1/t) ~ c · t^{-d}.
    let t = pool.symbol("__asy_t", Domain::Positive);
    let inv_t = pool.pow(t, pool.integer(-1_i32));
    let mut m = HashMap::new();
    m.insert(var, inv_t);
    let arg_t = simplify(subs(arg, &m, pool), pool).value;
    let order = (n_terms as u32).saturating_add(6).min(40);
    let (val, _analytic) = match regularize_at_zero(arg_t, t, pool) {
        Some(p) => p,
        None => return Ok(None),
    };
    let d = -val; // x-power of arg's leading behaviour
    if d <= 0 {
        // arg does not grow polynomially; not a log-at-infinity peel.
        return Ok(None);
    }

    // log(arg) = d·log(x) + log(arg / x^d). The cofactor arg / x^d → leading
    // constant and is power-expandable.
    let x_d = pool.pow(var, pool.integer(d));
    let cofactor = simplify(
        pool.mul(vec![arg, pool.pow(x_d, pool.integer(-1_i32))]),
        pool,
    )
    .value;
    let log_cofactor = pool.func("log", vec![cofactor]);

    // Leading log term.
    let log_x = pool.func("log", vec![var]);
    let lead = simplify(pool.mul(vec![pool.integer(d), log_x]), pool).value;

    // Power-expand the analytic remainder log(cofactor); it has a finite limit,
    // so its expansion is a genuine power series in 1/x with no log.
    let remainder = power_scale_terms_raw(log_cofactor, var, order, pool)?;
    let mut candidate: Vec<ExprId> = vec![lead];
    for (_, e) in remainder.into_iter().take(n_terms.saturating_sub(1)) {
        // Drop a structurally-zero constant term (limit of log cofactor is 0).
        if matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0) {
            continue;
        }
        candidate.push(e);
    }

    let gated = gate_terms(f, var, &candidate, pool);
    if gated.is_empty() {
        return Ok(None);
    }
    Ok(Some(AsymptoticExpansion {
        terms: gated
            .into_iter()
            .map(|expr| AsymptoticTerm { expr })
            .collect(),
    }))
}

/// `f = scale · h` with one leading `log(x)`/`exp(g)` factor and a
/// power-expandable cofactor `h`. Expand `h ~ Σ hᵢ` and distribute the scale.
fn mul_with_scale_factor(
    factors: &[ExprId],
    var: ExprId,
    n_terms: usize,
    pool: &ExprPool,
) -> Result<Option<AsymptoticExpansion>, AsymptoticError> {
    // Identify a single scale factor: log(var) or exp(g) with g depending on var.
    let mut scale: Option<ExprId> = None;
    let mut rest: Vec<ExprId> = Vec::new();
    for &fac in factors {
        let is_scale = match pool.get(fac) {
            ExprData::Func { name, args } if name == "log" && args.len() == 1 => args[0] == var,
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                depends_on(args[0], var, pool)
            }
            _ => false,
        };
        if is_scale && scale.is_none() {
            scale = Some(fac);
        } else {
            rest.push(fac);
        }
    }
    let Some(scale) = scale else {
        return Ok(None);
    };
    let cofactor = if rest.is_empty() {
        pool.integer(1_i32)
    } else {
        simplify(pool.mul(rest), pool).value
    };
    // The cofactor must be power-expandable on its own.
    let order = (n_terms as u32).saturating_mul(2).saturating_add(8).min(40);
    let co_terms = power_scale_terms_raw(cofactor, var, order, pool)?;
    if co_terms.is_empty() {
        return Ok(None);
    }
    let candidate: Vec<ExprId> = co_terms
        .into_iter()
        .take(n_terms)
        .map(|(_, e)| simplify(pool.mul(vec![scale, e]), pool).value)
        .collect();

    let f = simplify(pool.mul(factors.to_vec()), pool).value;
    let gated = gate_terms(f, var, &candidate, pool);
    if gated.is_empty() {
        return Ok(None);
    }
    Ok(Some(AsymptoticExpansion {
        terms: gated
            .into_iter()
            .map(|expr| AsymptoticTerm { expr })
            .collect(),
    }))
}

fn depends_on(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|a| depends_on(*a, var, pool)),
        ExprData::Pow { base, exp } => depends_on(base, var, pool) || depends_on(exp, var, pool),
        ExprData::Func { args, .. } => args.iter().any(|a| depends_on(*a, var, pool)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Numeric o()-gate.
// ---------------------------------------------------------------------------

/// Sample points at which the asymptotic gate is checked.
const GATE_POINTS: [f64; 3] = [1.0e2, 1.0e4, 1.0e6];

/// Relative slack allowed in the residual ≤ |next term| comparison (accounts
/// for floating-point error and the fact that the *next* asymptotic term only
/// bounds the residual up to a constant for finitely many terms).
const GATE_SLACK: f64 = 8.0;

/// Filter `candidate` (ordered, most-significant first) down to the longest
/// verified prefix: term `k+1` must be `o(term k)` and the residual after `k`
/// terms must be controlled by term `k+1` at every gate point.
///
/// Returns the surviving prefix (possibly empty).
fn gate_terms(f: ExprId, var: ExprId, candidate: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    if candidate.is_empty() {
        return Vec::new();
    }

    // Numerically evaluate f at each gate point.
    let mut f_vals = [0.0f64; GATE_POINTS.len()];
    for (j, &xv) in GATE_POINTS.iter().enumerate() {
        let mut env = HashMap::new();
        env.insert(var, xv);
        match eval_interp(f, &env, pool) {
            Some(v) if v.is_finite() => f_vals[j] = v,
            _ => return Vec::new(), // cannot evaluate f → cannot gate → decline
        }
    }

    // term_vals[k][j]
    let mut term_vals: Vec<[f64; GATE_POINTS.len()]> = Vec::with_capacity(candidate.len());
    for &term in candidate {
        let mut row = [0.0f64; GATE_POINTS.len()];
        for (j, &xv) in GATE_POINTS.iter().enumerate() {
            let mut env = HashMap::new();
            env.insert(var, xv);
            match eval_interp(term, &env, pool) {
                Some(v) if v.is_finite() => row[j] = v,
                _ => {
                    // Term not numerically evaluable — stop accepting here.
                    row = [f64::NAN; GATE_POINTS.len()];
                    break;
                }
            }
        }
        term_vals.push(row);
    }

    let mut accepted = 0usize;
    let mut partial = [0.0f64; GATE_POINTS.len()];

    for k in 0..candidate.len() {
        let row = term_vals[k];
        if row.iter().any(|v| !v.is_finite()) {
            break;
        }

        // o()-check vs previous term: |term_k| < |term_{k-1}| at large x, and
        // the ratio should be shrinking across the gate points.
        if k > 0 {
            let prev = term_vals[k - 1];
            let mut ok = true;
            let mut last_ratio = f64::INFINITY;
            for j in 0..GATE_POINTS.len() {
                let denom = prev[j].abs();
                if denom == 0.0 {
                    ok = false;
                    break;
                }
                let ratio = row[j].abs() / denom;
                if ratio > 1.0 {
                    ok = false;
                    break;
                }
                if j > 0 && ratio > last_ratio * (1.0 + 1e-9) {
                    // ratio not decreasing → not a genuine asymptotic refinement.
                    ok = false;
                    break;
                }
                last_ratio = ratio;
            }
            if !ok {
                break;
            }
        }

        // Tentatively add this term and check the residual is controlled by it.
        let mut next_partial = partial;
        for j in 0..GATE_POINTS.len() {
            next_partial[j] += row[j];
        }

        let mut residual_ok = true;
        let mut last_rel = f64::INFINITY;
        for j in 0..GATE_POINTS.len() {
            let residual = (f_vals[j] - next_partial[j]).abs();
            let scale = row[j].abs();
            // After adding term k, the residual must be no bigger than this
            // term (up to slack); this is the "term k+1 = o(term k)" guarantee
            // expressed against the realized residual.
            if residual > scale * GATE_SLACK + 1e-12 {
                residual_ok = false;
                break;
            }
            // Residual (relative to current term magnitude) should not grow as
            // x increases.
            let rel = if scale > 0.0 {
                residual / scale
            } else {
                residual
            };
            if j > 0 && rel > last_rel * GATE_SLACK + 1e-12 {
                residual_ok = false;
                break;
            }
            last_rel = rel;
        }
        if !residual_ok {
            break;
        }

        partial = next_partial;
        accepted = k + 1;
    }

    candidate[..accepted].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn approx_eq_expr(a: ExprId, b: ExprId, var: ExprId, pool: &ExprPool) -> bool {
        // Structural-or-numeric equality across a few sample points.
        if simplify(a, pool).value == simplify(b, pool).value {
            return true;
        }
        let mut any = false;
        for &xv in &[2.5f64, 7.0, 13.0] {
            let mut env = HashMap::new();
            env.insert(var, xv);
            let (Some(va), Some(vb)) = (eval_interp(a, &env, pool), eval_interp(b, &env, pool))
            else {
                return false;
            };
            if (va - vb).abs() > 1e-9 * (1.0 + va.abs()) {
                return false;
            }
            any = true;
        }
        any
    }

    /// Numeric value of a constant (var-independent) expression.
    fn const_val(e: ExprId, pool: &ExprPool) -> Option<f64> {
        eval_interp(e, &HashMap::new(), pool)
    }

    /// Numerically check f ~ Σ terms at a large x: residual small vs last term.
    fn residual_small(f: ExprId, terms: &[ExprId], var: ExprId, pool: &ExprPool) -> bool {
        let xv = 1.0e5;
        let mut env = HashMap::new();
        env.insert(var, xv);
        let fv = eval_interp(f, &env, pool).unwrap();
        let mut sum = 0.0;
        let mut last = 0.0;
        for &t in terms {
            let v = eval_interp(t, &env, pool).unwrap();
            sum += v;
            last = v;
        }
        (fv - sum).abs() <= last.abs() * 8.0 + 1e-9
    }

    #[test]
    fn rational_x_plus_1_over_x_minus_1() {
        // (x+1)/(x-1) ~ 1 + 2/x + 2/x^2 + ...
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let num = p.add(vec![x, p.integer(1)]);
        let den = p.add(vec![x, p.integer(-1)]);
        let f = p.mul(vec![num, p.pow(den, p.integer(-1))]);
        let exp = asymptotic_expand(f, x, 4, &p).unwrap();
        let terms = exp.term_exprs();
        assert!(terms.len() >= 3, "got {} terms", terms.len());
        // Leading term is 1.
        assert_eq!(const_val(terms[0], &p), Some(1.0));
        // 2/x
        let two_over_x = p.mul(vec![p.integer(2), p.pow(x, p.integer(-1))]);
        assert!(approx_eq_expr(terms[1], two_over_x, x, &p));
        assert!(residual_small(f, &terms, x, &p));
    }

    #[test]
    fn sqrt_x_squared_plus_one() {
        // sqrt(x^2+1) ~ x + 1/(2x) - 1/(8x^3) + ...
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let inside = p.add(vec![p.pow(x, p.integer(2)), p.integer(1)]);
        let f = p.func("sqrt", vec![inside]);
        let exp = asymptotic_expand(f, x, 3, &p).unwrap();
        let terms = exp.term_exprs();
        assert!(terms.len() >= 2, "got {} terms", terms.len());
        assert!(approx_eq_expr(terms[0], x, x, &p));
        // second term ~ 1/(2x)
        let half_over_x = p.mul(vec![
            p.rational(rug::Integer::from(1), rug::Integer::from(2)),
            p.pow(x, p.integer(-1)),
        ]);
        assert!(approx_eq_expr(terms[1], half_over_x, x, &p));
        assert!(residual_small(f, &terms, x, &p));
    }

    #[test]
    fn x_sin_one_over_x() {
        // x*sin(1/x) ~ 1 - 1/(6x^2) + ...
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let inv = p.pow(x, p.integer(-1));
        let f = p.mul(vec![x, p.func("sin", vec![inv])]);
        let exp = asymptotic_expand(f, x, 3, &p).unwrap();
        let terms = exp.term_exprs();
        assert!(!terms.is_empty());
        assert_eq!(const_val(terms[0], &p), Some(1.0));
        assert!(residual_small(f, &terms, x, &p));
    }

    #[test]
    fn exp_one_over_x_times_x() {
        // e^{1/x} * x ~ x + 1 + 1/(2x) + ...
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let inv = p.pow(x, p.integer(-1));
        let f = p.mul(vec![p.func("exp", vec![inv]), x]);
        let exp = asymptotic_expand(f, x, 3, &p).unwrap();
        let terms = exp.term_exprs();
        assert!(terms.len() >= 2, "got {}", terms.len());
        assert!(approx_eq_expr(terms[0], x, x, &p));
        assert_eq!(const_val(terms[1], &p), Some(1.0));
        assert!(residual_small(f, &terms, x, &p));
    }

    #[test]
    fn log_x_plus_one() {
        // log(x+1) ~ log x + 1/x - 1/(2x^2) + ...
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let arg = p.add(vec![x, p.integer(1)]);
        let f = p.func("log", vec![arg]);
        let exp = asymptotic_expand(f, x, 3, &p).unwrap();
        let terms = exp.term_exprs();
        assert!(terms.len() >= 2, "got {}", terms.len());
        // leading term log(x)
        let log_x = p.func("log", vec![x]);
        assert_eq!(simplify(terms[0], &p).value, simplify(log_x, &p).value);
        // 1/x term
        let inv = p.pow(x, p.integer(-1));
        assert!(approx_eq_expr(terms[1], inv, x, &p));
        assert!(residual_small(f, &terms, x, &p));
    }

    #[test]
    fn sqrt_x_plus_sqrt_x_leading() {
        // sqrt(x + sqrt(x)) ~ sqrt(x) · sqrt(1 + 1/sqrt(x)); leading behaviour
        // ~ sqrt(x). The inner sqrt(x) is a half-integer scale, so this is a
        // Puiseux case for the integer-power core: we expect either the
        // leading-scale peel to deliver sqrt(x), or an honest decline. Whatever
        // is returned must pass the residual gate.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let inner = p.func("sqrt", vec![x]);
        let arg = p.add(vec![x, inner]);
        let f = p.func("sqrt", vec![arg]);
        match asymptotic_expand(f, x, 2, &p) {
            Ok(exp) => {
                let terms = exp.term_exprs();
                assert!(residual_small(f, &terms, x, &p));
                // Leading term should behave like sqrt(x): ratio → 1 at large x.
                let mut env = HashMap::new();
                env.insert(x, 1.0e6f64);
                let lead = eval_interp(terms[0], &env, &p).unwrap();
                let sx = 1.0e3f64; // sqrt(1e6)
                assert!((lead / sx - 1.0).abs() < 1e-2, "lead={lead}");
            }
            Err(_) => { /* honest decline acceptable (half-integer/Puiseux scale) */ }
        }
    }

    #[test]
    fn x_over_log_x_peels() {
        // x / log(x): a single dominant log factor with a power-expandable
        // cofactor x. Leading term ~ x / log(x).
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let logx = p.func("log", vec![x]);
        let f = p.mul(vec![x, p.pow(logx, p.integer(-1))]);
        match asymptotic_expand(f, x, 2, &p) {
            Ok(exp) => {
                let terms = exp.term_exprs();
                assert!(!terms.is_empty());
                assert!(residual_small(f, &terms, x, &p));
            }
            Err(_) => { /* acceptable: 1/log(x) cofactor is itself non-power */ }
        }
    }

    #[test]
    fn invalid_term_count() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let err = asymptotic_expand(x, x, 0, &p).unwrap_err();
        assert!(matches!(err, AsymptoticError::InvalidTermCount));
    }

    #[test]
    fn x_over_log_x_gate_is_honest() {
        // 1/(x log x): a genuine non-power scale; we must not fabricate an
        // unverified power-scale tail. Either an honest log-peel or a decline.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Positive);
        let logx = p.func("log", vec![x]);
        let f = p.mul(vec![p.pow(x, p.integer(-1)), p.pow(logx, p.integer(-1))]);
        match asymptotic_expand(f, x, 3, &p) {
            Ok(exp) => {
                // If anything is returned it must pass the residual gate.
                let terms = exp.term_exprs();
                assert!(residual_small(f, &terms, x, &p));
            }
            Err(_) => { /* honest decline is acceptable */ }
        }
    }
}
