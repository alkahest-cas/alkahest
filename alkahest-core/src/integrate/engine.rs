/// Symbolic integration — rule-based Risch subset.
///
/// Handles:
/// - Constants: `∫ c dx = c·x`
/// - Power rule: `∫ x^n dx = x^(n+1)/(n+1)` (`n ≠ -1`)
/// - Logarithm: `∫ x^(-1) dx = ln(x)`  (`∫ 1/x dx`)
/// - Sum rule: `∫ (f + g) dx = ∫f dx + ∫g dx`
/// - Constant-multiple rule: `∫ c·f dx = c · ∫f dx`
/// - Known functions: sin, cos, exp, 1/x
/// - Inverse-trig / inverse-hyperbolic via integration by parts: atan, asin,
///   acos, asinh, acosh, atanh (bare and `rest(x)·f(x)`)
///
/// Everything else returns `Err(IntegrationError::NotImplemented)`.
///
/// The result is simplified with the rule-based simplifier before returning.
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::{simplify, simplify_expanded};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrationError {
    /// The expression is outside the supported Risch subset.
    ///
    /// Also used as a **semver-safe carrier** for budget/cancellation trips
    /// (see [`IntegrationError::from`] for [`crate::budget::BudgetError`]):
    /// adding a dedicated `Budget` variant would be a major break on this
    /// exhaustive enum. Encoded messages start with the internal `[[budget]]`
    /// marker; use [`IntegrationError::is_budget`] /
    /// [`IntegrationError::budget_code`] to distinguish them from genuine
    /// "not implemented" declines. Python maps these to `BudgetExceededError`
    /// (`E-BUDGET-*`).
    NotImplemented(String),
    /// Division by zero would occur (e.g. power-rule with n=-1 on a non-x base).
    DivisionByZero,
    /// The algebraic extension has degree > 2 (v1.1 supports only sqrt / degree-2).
    UnsupportedExtensionDegree(u32),
    /// The integrand provably has no elementary antiderivative (e.g. elliptic integrals).
    NonElementary(String),
}

/// Prefix for [`IntegrationError::NotImplemented`] messages that encode a
/// [`crate::budget::BudgetError`]. Invisible to casual grepping of user-facing
/// "not implemented" strings; stripped from [`Display`].
const BUDGET_MARKER: &str = "[[budget]]";

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrationError::NotImplemented(msg) => {
                if let Some(rest) = msg.strip_prefix(BUDGET_MARKER) {
                    write!(f, "integrate: {rest}")
                } else {
                    write!(f, "integrate: not implemented: {msg}")
                }
            }
            IntegrationError::DivisionByZero => write!(f, "integrate: division by zero"),
            IntegrationError::UnsupportedExtensionDegree(q) => write!(
                f,
                "integrate: algebraic extension of degree {q} is not supported \
                 (v1.1 supports only degree-2 / sqrt extensions)"
            ),
            IntegrationError::NonElementary(msg) => {
                write!(f, "integrate: no elementary antiderivative exists: {msg}")
            }
        }
    }
}

impl std::error::Error for IntegrationError {}

impl From<crate::budget::BudgetError> for IntegrationError {
    fn from(e: crate::budget::BudgetError) -> Self {
        use crate::errors::AlkahestError;
        // Encode code + Display body so Python/callers keep E-BUDGET-* without
        // a new exhaustive-enum variant (cargo-semver-checks major).
        IntegrationError::NotImplemented(format!("{BUDGET_MARKER}[{}] {e}", e.code()))
    }
}

impl IntegrationError {
    /// `true` when this error encodes a budget/cancellation trip rather than a
    /// genuine "outside the Risch subset" decline.
    pub fn is_budget(&self) -> bool {
        matches!(self, IntegrationError::NotImplemented(msg) if msg.starts_with(BUDGET_MARKER))
    }

    /// The `E-BUDGET-*` code when [`is_budget`](Self::is_budget), else `None`.
    pub fn budget_code(&self) -> Option<&'static str> {
        let IntegrationError::NotImplemented(msg) = self else {
            return None;
        };
        let rest = msg.strip_prefix(BUDGET_MARKER)?;
        if rest.starts_with("[E-BUDGET-001]") {
            Some("E-BUDGET-001")
        } else if rest.starts_with("[E-BUDGET-002]") {
            Some("E-BUDGET-002")
        } else if rest.starts_with("[E-BUDGET-003]") {
            Some("E-BUDGET-003")
        } else {
            None
        }
    }

    /// A human-readable remediation hint for the user.
    pub fn remediation(&self) -> Option<&'static str> {
        if let Some(code) = self.budget_code() {
            return match code {
                "E-BUDGET-001" => Some(
                    "raise Budget(wall_ms=...), or accept a heuristic/numeric result for this \
                     candidate instead of an exact one",
                ),
                "E-BUDGET-002" => Some(
                    "raise Budget(max_steps=...), or accept a partial/heuristic result for this \
                     candidate instead of an exact one",
                ),
                "E-BUDGET-003" => Some(
                    "call alkahest.clear_cancel() (Python) or budget::clear_cancel() (Rust) before \
                     starting the next candidate",
                ),
                _ => None,
            };
        }
        match self {
            IntegrationError::NotImplemented(_) => Some(
                "only power, linearity, sin/cos/exp rules and algebraic (sqrt) rules \
                 are implemented; use a numeric integrator for arbitrary functions",
            ),
            IntegrationError::DivisionByZero => None,
            IntegrationError::UnsupportedExtensionDegree(_) => Some(
                "v1.1 supports sqrt(P(x)) only; higher-degree radicals (cbrt, nth-root) \
                 are planned for v2.0",
            ),
            IntegrationError::NonElementary(_) => Some(
                "this integrand has no closed-form antiderivative in terms of elementary \
                 functions; use a numeric integrator or elliptic-integral library",
            ),
        }
    }

    /// Optional source span `(start_byte, end_byte)` within the input text.
    pub fn span(&self) -> Option<(usize, usize)> {
        None
    }
}

impl crate::errors::AlkahestError for IntegrationError {
    fn code(&self) -> &'static str {
        if let Some(code) = self.budget_code() {
            return code;
        }
        match self {
            IntegrationError::NotImplemented(_) => "E-INT-001",
            IntegrationError::DivisionByZero => "E-INT-002",
            IntegrationError::UnsupportedExtensionDegree(_) => "E-INT-003",
            IntegrationError::NonElementary(_) => "E-INT-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        IntegrationError::remediation(self)
    }
}

// ---------------------------------------------------------------------------
// Logarithmic-derivative rule:  ∫ (h'/h)·log(h)^n dx
// ---------------------------------------------------------------------------

/// Integrate `∫ (h'/h)·log(h)^n dx` for an integer `n`.
///
/// With `θ = log(h)` the derivation gives `Dθ = h'/h`, so the integrand
/// `(h'/h)·θ^n = Dθ·θ^n` has antiderivative `θ^{n+1}/(n+1)` for `n ≠ −1` and
/// `log(θ) = log(log(h))` for `n = −1`.  This is the single-generator
/// logarithmic case of the Risch algorithm; it covers elementary integrands the
/// rule engine cannot reduce, e.g. `∫ 1/(x·log x) dx = log(log x)` and
/// `∫ 1/(x·log(x)^2) dx = −1/log(x)`.
///
/// Returns `Some(F)` only when the integrand matches the template exactly (the
/// coefficient equals `h'/h` as a rational function), so the result is always a
/// sound, differentiation-verifiable antiderivative; otherwise `None`.
fn try_log_derivative(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    use super::risch::poly_rde::{poly_mul, rational_to_expr, trim};
    use super::risch::rational_rde::expr_to_qrational;
    use super::risch::tower::find_generators;

    // The integrand must involve exactly one transcendental generator, log(h).
    let gens = find_generators(expr, var, pool);
    if gens.len() != 1 || !gens[0].is_log() {
        return None;
    }
    let theta = gens[0].generator; // log(h)
    let h = gens[0].argument(); // h

    // Write expr = coeff · θ^n with a nonzero integer n.
    let (coeff, n) = extract_log_power(expr, theta, pool)?;
    if n == 0 {
        return None;
    }

    // coeff must be a rational function of `var` (no θ inside).
    let (cn, cd) = expr_to_qrational(coeff, var, pool)?;

    // Require coeff == h'/h as rational functions.
    let hp = crate::diff::diff(h, var, pool).ok()?.value;
    let (hpn, hpd) = expr_to_qrational(hp, var, pool)?;
    let (hn, hd) = expr_to_qrational(h, var, pool)?;
    // h'/h = (hpn·hd) / (hpd·hn);  coeff == h'/h  ⇔  cn·(hpd·hn) == (hpn·hd)·cd.
    let rn = poly_mul(&hpn, &hd);
    let rd = poly_mul(&hpd, &hn);
    if trim(poly_mul(&cn, &rd)) != trim(poly_mul(&rn, &cd)) {
        return None;
    }

    // Antiderivative.
    if n == -1 {
        Some(pool.func("log", vec![theta])) // log(log(h))
    } else {
        let np1 = n + 1;
        let pow = pool.pow(theta, pool.integer(np1));
        let coeff_expr = rational_to_expr(&rug::Rational::from((1_i64, np1)), pool);
        Some(pool.mul(vec![coeff_expr, pow]))
    }
}

/// Decompose `expr` as `coeff · theta^n` for an integer `n`, returning
/// `(coeff, n)`.  `coeff` collects every factor other than integer powers of
/// `theta`.  Returns `None` if `theta` does not appear (or appears only with a
/// non-integer exponent).
fn extract_log_power(expr: ExprId, theta: ExprId, pool: &ExprPool) -> Option<(ExprId, i64)> {
    if expr == theta {
        return Some((pool.integer(1_i32), 1));
    }
    match pool.get(expr) {
        ExprData::Pow { base, exp } if base == theta => match pool.get(exp) {
            ExprData::Integer(m) => Some((pool.integer(1_i32), m.0.to_i64()?)),
            _ => None,
        },
        ExprData::Mul(args) => {
            let mut n: i64 = 0;
            let mut rest: Vec<ExprId> = Vec::new();
            for &a in &args {
                if a == theta {
                    n += 1;
                } else if let ExprData::Pow { base, exp } = pool.get(a) {
                    if base == theta {
                        match pool.get(exp) {
                            ExprData::Integer(m) => n += m.0.to_i64()?,
                            _ => rest.push(a),
                        }
                    } else {
                        rest.push(a);
                    }
                } else {
                    rest.push(a);
                }
            }
            if n == 0 {
                return None;
            }
            let coeff = match rest.len() {
                0 => pool.integer(1_i32),
                1 => rest[0],
                _ => pool.mul(rest),
            };
            Some((coeff, n))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the i64 value of an integer expression, or None.
fn as_integer(expr: ExprId, pool: &ExprPool) -> Option<i64> {
    pool.with(expr, |data| match data {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    })
}

/// Return `true` if `expr` does not involve `var` (is a constant w.r.t. `var`).
///
/// Internally memoises into `cache` (keyed by `ExprId`, valid for a fixed `var`).
/// Use [`is_free_of`] from call sites; [`is_free_of_inner`] is the recursive worker.
fn is_free_of(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let mut cache: HashMap<ExprId, bool> = HashMap::new();
    is_free_of_inner(expr, var, pool, &mut cache)
}

fn is_free_of_inner(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    cache: &mut HashMap<ExprId, bool>,
) -> bool {
    if expr == var {
        return false;
    }
    if let Some(&cached) = cache.get(&expr) {
        return cached;
    }
    let children: Vec<ExprId> = pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::Func { args, .. } => args.clone(),
        _ => vec![],
    });
    let result = children
        .into_iter()
        .all(|c| is_free_of_inner(c, var, pool, cache));
    cache.insert(expr, result);
    result
}

/// If `expr = a*var + b` where `a`, `b` are free of `var`, return `Some((a, b))`.
/// Returns `Some((1, 0))` when `expr == var`.
fn is_linear_in(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if expr == var {
        return Some((pool.integer(1_i32), pool.integer(0_i32)));
    }
    match pool.get(expr) {
        ExprData::Mul(args) => {
            let var_pos = args.iter().position(|&a| a == var)?;
            let others: Vec<ExprId> = args
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != var_pos)
                .map(|(_, &a)| a)
                .collect();
            let a = match others.len() {
                0 => pool.integer(1_i32),
                1 => others[0],
                _ => pool.mul(others),
            };
            if is_free_of(a, var, pool) {
                Some((a, pool.integer(0_i32)))
            } else {
                None
            }
        }
        ExprData::Add(args) => {
            let mut a_opt: Option<ExprId> = None;
            let mut b_parts: Vec<ExprId> = vec![];
            for &arg in &args {
                if arg == var {
                    if a_opt.is_some() {
                        return None;
                    }
                    a_opt = Some(pool.integer(1_i32));
                } else {
                    match pool.get(arg) {
                        ExprData::Mul(margs) => {
                            let vpos = margs.iter().position(|&m| m == var);
                            if let Some(vp) = vpos {
                                if a_opt.is_some() {
                                    return None;
                                }
                                let others: Vec<ExprId> = margs
                                    .iter()
                                    .enumerate()
                                    .filter(|&(i, _)| i != vp)
                                    .map(|(_, &m)| m)
                                    .collect();
                                let coeff = match others.len() {
                                    0 => pool.integer(1_i32),
                                    1 => others[0],
                                    _ => pool.mul(others),
                                };
                                if is_free_of(coeff, var, pool) {
                                    a_opt = Some(coeff);
                                } else {
                                    b_parts.push(arg);
                                }
                            } else if is_free_of(arg, var, pool) {
                                b_parts.push(arg);
                            } else {
                                return None;
                            }
                        }
                        _ if is_free_of(arg, var, pool) => b_parts.push(arg),
                        _ => return None,
                    }
                }
            }
            let a = a_opt?;
            let b = match b_parts.len() {
                0 => pool.integer(0_i32),
                1 => b_parts[0],
                _ => pool.add(b_parts),
            };
            Some((a, b))
        }
        _ => None,
    }
}

/// Match `∫ c * x * exp(x) dx = c * exp(x) * (x - 1)`.
///
/// Recognises any `Mul` containing exactly one `exp(var)` factor, exactly one
/// `var` factor, and zero or more constant (free-of-var) factors.
fn try_x_times_func(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<ExprId> {
    let args = match pool.get(expr) {
        ExprData::Mul(v) => v,
        _ => return None,
    };

    let exp_pos = args.iter().position(|&a| {
        pool.with(a, |d| match d {
            ExprData::Func { name, args } => name == "exp" && args.len() == 1 && args[0] == var,
            _ => false,
        })
    })?;

    let var_pos = args.iter().position(|&a| a == var)?;

    let others: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != exp_pos && i != var_pos)
        .map(|(_, &a)| a)
        .collect();
    if !others.iter().all(|&a| is_free_of(a, var, pool)) {
        return None;
    }

    // ∫ c * x * exp(x) dx = c * exp(x) * (x - 1)
    let exp_x = args[exp_pos];
    let x_minus_1 = pool.add(vec![var, pool.integer(-1_i32)]);
    let mut factors = vec![exp_x, x_minus_1];
    factors.extend_from_slice(&others);
    let result = pool.mul(factors);
    log.push(RewriteStep::simple("int_x_exp", expr, result));
    Some(result)
}

// ---------------------------------------------------------------------------
// Inverse-trigonometric integration by parts
// ---------------------------------------------------------------------------

/// `true` if `name` is one of the inverse-trigonometric or inverse-hyperbolic
/// functions handled by the IBP path (`atan`, `asin`, `acos`, `asinh`, `acosh`,
/// `atanh`).  All six have algebraic (rational-or-√-quadratic) derivatives, so
/// the IBP residual `∫ P·f'` closes through the existing rational/√-quadratic
/// engines.
fn is_inverse_trig(name: &str) -> bool {
    matches!(name, "atan" | "asin" | "acos" | "asinh" | "acosh" | "atanh")
}

/// `true` if `expr` contains an inverse-trigonometric or inverse-hyperbolic
/// function anywhere in its tree.  Used to guarantee the IBP residual is
/// inverse-trig-free, so the IBP branch cannot re-enter itself (termination).
fn contains_inverse_trig(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { name, args } => {
            is_inverse_trig(&name) || args.iter().any(|&a| contains_inverse_trig(a, pool))
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_inverse_trig(a, pool))
        }
        ExprData::Pow { base, exp } => {
            contains_inverse_trig(base, pool) || contains_inverse_trig(exp, pool)
        }
        _ => false,
    }
}

/// Derivative `f'(var)` for an inverse-trigonometric or inverse-hyperbolic `f`:
/// `atan'(x) = 1/(1+x²)`, `asin'(x) = 1/√(1−x²)`, `acos'(x) = −1/√(1−x²)`,
/// `asinh'(x) = 1/√(x²+1)`, `acosh'(x) = 1/√(x²−1)`, `atanh'(x) = 1/(1−x²)`.
fn inverse_trig_derivative(name: &str, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let x2 = pool.pow(var, pool.integer(2_i32));
    match name {
        "atan" => {
            // 1/(1 + x²)
            let denom = pool.add(vec![pool.integer(1_i32), x2]);
            Some(pool.pow(denom, pool.integer(-1_i32)))
        }
        "atanh" => {
            // 1/(1 − x²)
            let neg_x2 = pool.mul(vec![pool.integer(-1_i32), x2]);
            let denom = pool.add(vec![pool.integer(1_i32), neg_x2]);
            Some(pool.pow(denom, pool.integer(-1_i32)))
        }
        "asin" | "acos" => {
            // ±1/√(1 − x²)
            let neg_x2 = pool.mul(vec![pool.integer(-1_i32), x2]);
            let one_minus_x2 = pool.add(vec![pool.integer(1_i32), neg_x2]);
            let sqrt = pool.func("sqrt", vec![one_minus_x2]);
            let inv = pool.pow(sqrt, pool.integer(-1_i32));
            if name == "asin" {
                Some(inv)
            } else {
                Some(pool.mul(vec![pool.integer(-1_i32), inv]))
            }
        }
        "asinh" => {
            // 1/√(x² + 1)
            let x2_plus_one = pool.add(vec![x2, pool.integer(1_i32)]);
            let sqrt = pool.func("sqrt", vec![x2_plus_one]);
            Some(pool.pow(sqrt, pool.integer(-1_i32)))
        }
        "acosh" => {
            // 1/√(x² − 1)
            let x2_minus_one = pool.add(vec![x2, pool.integer(-1_i32)]);
            let sqrt = pool.func("sqrt", vec![x2_minus_one]);
            Some(pool.pow(sqrt, pool.integer(-1_i32)))
        }
        _ => None,
    }
}

/// Largest integer power `k` of an inverse-trig factor the IBP reduction will
/// attempt.  Each IBP step lowers `k` by one, so recursion always terminates;
/// this cap only bounds expression blow-up for pathological inputs (powers above
/// it decline cleanly rather than expanding a huge intermediate form).
const MAX_INVERSE_TRIG_POWER: i64 = 12;

thread_local! {
    /// Re-entry depth of [`try_inverse_trig_ibp`] on the current thread.  Needed
    /// because a `k ≥ 2` residual of a *rational*-derivative inverse function
    /// (atan/atanh) is `∫ log(1∓x²)/(1∓x²) dx`, which the Risch log-case
    /// integrates by parts back into `∫ atan(x)·(…) dx` — a product that re-enters
    /// this branch, forming a mutual-recursion cycle with no elementary fixed
    /// point.  Bounding the re-entry depth breaks the cycle so those genuinely
    /// non-elementary integrals decline cleanly instead of overflowing the stack.
    /// The elementary (algebraic-derivative) cases never re-enter, so the bound
    /// does not affect them.
    static INVERSE_TRIG_IBP_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Maximum re-entry depth for [`try_inverse_trig_ibp`].  The elementary
/// (asin/acos/asinh/acosh) reductions enter exactly once, so `1` suffices;
/// deeper re-entry only ever arises from the non-elementary atan²/atanh² cycle,
/// which must decline.
const INVERSE_TRIG_IBP_MAX_DEPTH: u32 = 1;

/// RAII guard that increments [`INVERSE_TRIG_IBP_DEPTH`] on construction and
/// decrements it on drop, so the depth is restored on every exit path (including
/// the `?` early returns in [`try_inverse_trig_ibp`]).
struct InverseTrigIbpDepthGuard;

impl Drop for InverseTrigIbpDepthGuard {
    fn drop(&mut self) {
        INVERSE_TRIG_IBP_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// If `a` is `f(var)` or `f(var)^k` for an inverse-trig `f` and integer `k ≥ 1`,
/// return `(fname, k)`.  A bare function is treated as `k = 1`.  Non-integer,
/// zero, or negative exponents, and any other shape, return `None`.
fn as_inverse_trig_power(a: ExprId, var: ExprId, pool: &ExprPool) -> Option<(String, i64)> {
    match pool.get(a) {
        ExprData::Func { name, args }
            if args.len() == 1 && args[0] == var && is_inverse_trig(&name) =>
        {
            Some((name, 1))
        }
        ExprData::Pow { base, exp } => {
            let k = as_integer(exp, pool)?;
            if k < 1 {
                return None;
            }
            match pool.get(base) {
                ExprData::Func { name, args }
                    if args.len() == 1 && args[0] == var && is_inverse_trig(&name) =>
                {
                    Some((name, k))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Identify the shape `∫ rest(x)·f(x)^k dx`: a single inverse-trig factor `f`
/// (argument exactly `var`) raised to an integer power `k ≥ 1`, times an
/// inverse-trig-free polynomial/rational `rest` (or `1`).  Returns
/// `(fname, k, rest)`, or `None` when the integrand is not of this form (no
/// inverse-trig factor, two of them, a non-integer power, or a `rest` that still
/// contains an inverse-trig subterm).
fn match_inverse_trig_power(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(String, i64, ExprId)> {
    match pool.get(expr) {
        // Bare ∫ f(x)^k dx (including the k = 1 function node).
        ExprData::Func { .. } | ExprData::Pow { .. } => {
            let (name, k) = as_inverse_trig_power(expr, var, pool)?;
            Some((name, k, pool.integer(1_i32)))
        }
        // Product ∫ rest(x)·f(x)^k dx with exactly one inverse-trig factor.
        ExprData::Mul(args) => {
            let mut found: Option<(usize, String, i64)> = None;
            for (i, &a) in args.iter().enumerate() {
                if let Some((name, k)) = as_inverse_trig_power(a, var, pool) {
                    if found.is_some() {
                        return None; // two inverse-trig factors — out of scope
                    }
                    found = Some((i, name, k));
                }
            }
            let (pos, name, k) = found?;
            let rest_factors: Vec<ExprId> = args
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != pos)
                .map(|(_, &a)| a)
                .collect();
            let rest = match rest_factors.len() {
                0 => pool.integer(1_i32),
                1 => rest_factors[0],
                _ => pool.mul(rest_factors),
            };
            // `rest` must be inverse-trig-free (any remaining inverse-trig factor
            // would be a second one, or nested — out of scope for this branch).
            if contains_inverse_trig(rest, pool) {
                return None;
            }
            Some((name, k, rest))
        }
        _ => None,
    }
}

/// Integrate `∫ coeff(x)·f(x)^k dx` for integer `k ≥ 0` by repeated integration
/// by parts on the inverse-trig power, where `coeff` is inverse-trig-free:
///
/// ```text
/// ∫ coeff·f^k dx = C·f^k − k·∫ (C·f')·f^{k−1} dx,   C = ∫ coeff dx.
/// ```
///
/// Each step lowers the power of `f` by one, so the recursion terminates; the
/// new coefficient `C·f'` is again inverse-trig-free (`f'` is rational or
/// algebraic-√).  At `k = 0` this is the base case `∫ coeff dx`, resolved
/// through the full [`integrate`] engine (rational, algebraic-√, or a clean
/// decline when the residual is non-elementary — e.g. the `atan²`/`atanh²`
/// residual `∫ log(1∓x²)/(1∓x²) dx`).  Returns `None` if any sub-integral
/// declines.
fn integrate_inverse_trig_power(
    coeff: ExprId,
    fname: &str,
    k: i64,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    // Base case: pure ∫ coeff dx (coeff is inverse-trig-free ⇒ no re-entry).
    if k <= 0 {
        return integrate_additive(coeff, var, pool);
    }

    let fprime = inverse_trig_derivative(fname, var, pool)?;

    // C = ∫ coeff dx (full engine, so rational/algebraic-√ residuals resolve).
    let cap = simplify(integrate_additive(coeff, var, pool)?, pool).value;

    // Main term C·f^k.
    let f = pool.func(fname, vec![var]);
    let fk = if k == 1 {
        f
    } else {
        pool.pow(f, pool.integer(k))
    };
    let main = pool.mul(vec![cap, fk]);

    // Residual −k·∫ (C·f')·f^{k−1} dx.  `C·f'` may reintroduce `f` (e.g.
    // `∫ x²/√(1−x²)` contributes an `asin` term), so it is not assumed
    // inverse-trig-free; the reduction is still valid and the recursion still
    // lowers the tracked power of `f` by one.
    // Expand so a reintroduced-`f` term separates from the algebraic part into a
    // top-level sum (e.g. `(asin − x√)/(2√) → asin/(2√) − x/2`); the base case
    // then integrates each summand independently through the full pipeline.
    let new_coeff = simplify_expanded(pool.mul(vec![cap, fprime]), pool).value;
    let residual = integrate_inverse_trig_power(new_coeff, fname, k - 1, var, pool)?;
    let neg = pool.mul(vec![pool.integer(-k), residual]);

    Some(pool.add(vec![main, neg]))
}

/// Integrate `∫ expr dx` term-by-term over a top-level sum, sending each summand
/// through the full [`integrate`] pipeline (rule engine → rational fallback →
/// derivative-divides u-substitution).  The plain [`Node::Add`] sum-rule only
/// runs the rule engine on each term, so an `f(x)·f'(x)` summand produced by the
/// inverse-trig IBP reduction (which needs the u-substitution fallback to close)
/// would be missed; splitting here routes each term through the fallback.
/// Returns `None` if any summand declines.
fn integrate_additive(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let ExprData::Add(args) = pool.get(expr) {
        let mut terms = Vec::with_capacity(args.len());
        for a in args {
            terms.push(integrate_additive(a, var, pool)?);
        }
        return Some(pool.add(terms));
    }
    integrate(expr, var, pool).ok().map(|d| d.value)
}

/// Integrate `∫ rest(x)·f(x)^k dx` by parts, where `f ∈ {atan, asin, acos,
/// asinh, acosh, atanh}` (argument exactly `var`), `k ≥ 1` is an integer, and
/// `rest` is an inverse-trig-free polynomial/rational factor (or `1`):
///
/// ```text
/// ∫ rest·f^k dx = P·f^k − k·∫ (P·f')·f^{k−1} dx,   P = ∫ rest dx.
/// ```
///
/// The reduction ([`integrate_inverse_trig_power`]) recurses, lowering the power
/// of `f` by one each step until the pure `∫ … dx` base case, and terminates.
/// Whether the whole thing closes depends on the derivative of `f`:
/// asin/acos/asinh/acosh have **algebraic** derivatives (`1/√(1∓x²)` /
/// `1/√(x²±1)`), so every residual resolves and powers such as `∫ asin(x)² dx`
/// are elementary; atan/atanh have **rational** derivatives (`1/(1±x²)`), and
/// for `k ≥ 2` the final residual is the non-elementary `∫ log(1∓x²)/(1∓x²) dx`,
/// so `∫ atan(x)² dx` / `∫ atanh(x)² dx` decline cleanly (the sub-integral
/// returns `None`).  The final antiderivative is soundness-gated by
/// [`verify_antiderivative`]: it is returned only if `d/dx result = integrand`,
/// so a wrong integral is never emitted.  Returns `None` (decline) when the
/// shape does not match or any sub-integral declines.
fn try_inverse_trig_ibp(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<ExprId> {
    let (fname, k, rest) = match_inverse_trig_power(expr, var, pool)?;

    // Bound intermediate blow-up; powers above the cap decline cleanly.
    if k > MAX_INVERSE_TRIG_POWER {
        return None;
    }

    // Break the atan²/atanh² mutual-recursion cycle with the Risch log-case.
    let depth = INVERSE_TRIG_IBP_DEPTH.with(|d| d.get());
    if depth >= INVERSE_TRIG_IBP_MAX_DEPTH {
        return None;
    }
    INVERSE_TRIG_IBP_DEPTH.with(|d| d.set(depth + 1));
    let _depth_guard = InverseTrigIbpDepthGuard;

    let result = integrate_inverse_trig_power(rest, &fname, k, var, pool)?;

    // Soundness gate: only emit when d/dx result equals the integrand.
    if !verify_antiderivative(result, expr, var, pool) {
        return None;
    }

    log.push(RewriteStep::simple("int_inverse_trig_ibp", expr, result));
    Some(result)
}

// ---------------------------------------------------------------------------
// Products of polynomial/exponential with a trigonometric factor (IBP)
// ---------------------------------------------------------------------------

/// Match `∫ p(x)·sin(a·x+b) dx` / `∫ p(x)·cos(a·x+b) dx` where `p` is a genuine
/// polynomial in `var` and the trig argument is linear (`a·x+b`, `a ≠ 0`), and
/// build the antiderivative by repeated integration by parts (each step lowers
/// `deg p` by one and terminates at a constant `p`).  Soundness-gated: the result
/// is returned only when its derivative equals the integrand.
///
/// Declines (returns `None`) on non-polynomial coefficients, a non-linear trig
/// argument, or two trig factors (product-of-trigs linearization is out of
/// scope), so nothing already handled elsewhere regresses.
fn try_poly_trig_ibp(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<ExprId> {
    let args = match pool.get(expr) {
        ExprData::Mul(v) => v,
        _ => return None,
    };

    // Exactly one sin/cos factor whose argument is linear (non-constant) in var.
    let mut found: Option<(usize, bool, ExprId)> = None; // (pos, is_sin, arg)
    for (i, &a) in args.iter().enumerate() {
        if let ExprData::Func { name, args: fargs } = pool.get(a) {
            if fargs.len() == 1 && (name == "sin" || name == "cos") {
                let arg = fargs[0];
                if is_linear_in(arg, var, pool).is_some() {
                    if found.is_some() {
                        return None; // two trig factors — out of scope
                    }
                    found = Some((i, name == "sin", arg));
                }
            }
        }
    }
    let (pos, is_sin, arg) = found?;

    // Remaining factors form the polynomial coefficient p.
    let rest_factors: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pos)
        .map(|(_, &a)| a)
        .collect();
    let p = match rest_factors.len() {
        0 => pool.integer(1_i32),
        1 => rest_factors[0],
        _ => pool.mul(rest_factors),
    };
    // Require a genuine polynomial coefficient (decline e.g. `exp(x)·sin(x)`,
    // which the exp·trig fast-path handles instead).
    if !is_polynomial_in(p, var, pool) {
        return None;
    }

    let result = integrate_poly_trig(p, is_sin, arg, var, pool)?;

    // Soundness gate: only emit when d/dx result equals the integrand.
    if !verify_antiderivative(result, expr, var, pool) {
        return None;
    }
    log.push(RewriteStep::simple("int_poly_trig_ibp", expr, result));
    Some(result)
}

/// Recursive integration-by-parts kernel for `∫ p·sin(arg)` / `∫ p·cos(arg)`
/// with `arg = a·x+b` linear in `var`.  Uses `∫ p·f = p·v − ∫ v·p'` where `v`
/// is the antiderivative of the trig part; each recursion differentiates `p`
/// (lowering its degree) and swaps sin↔cos, terminating once `p` is constant.
fn integrate_poly_trig(
    p: ExprId,
    is_sin: bool,
    arg: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let (a, _b) = is_linear_in(arg, var, pool)?;
    let a_inv = pool.pow(a, pool.integer(-1_i32));
    let neg_one = pool.integer(-1_i32);

    // v = antiderivative of the trig part:
    //   sin(arg) -> -cos(arg)/a ,  cos(arg) -> sin(arg)/a
    let v = if is_sin {
        let cos_arg = pool.func("cos", vec![arg]);
        pool.mul(vec![neg_one, a_inv, cos_arg])
    } else {
        let sin_arg = pool.func("sin", vec![arg]);
        pool.mul(vec![a_inv, sin_arg])
    };
    let pv = pool.mul(vec![p, v]);

    // Base case: p constant ⇒ p' = 0 ⇒ ∫ v·p' = 0.
    if is_free_of(p, var, pool) {
        return Some(pv);
    }

    // p' via differentiation (degree strictly decreases ⇒ termination).
    let dp = crate::diff::diff(p, var, pool).ok()?.value;
    let dp = simplify(dp, pool).value;

    // ∫ v·p':  v = -cos(arg)/a (sin case) ⇒ -a_inv·∫ p'·cos(arg);
    //          v =  sin(arg)/a (cos case) ⇒  a_inv·∫ p'·sin(arg).
    let inner = integrate_poly_trig(dp, !is_sin, arg, var, pool)?;
    let coeff = if is_sin {
        pool.mul(vec![neg_one, a_inv])
    } else {
        a_inv
    };
    let vp_integral = pool.mul(vec![coeff, inner]);

    // result = p·v − ∫ v·p'.
    let neg_vp = pool.mul(vec![neg_one, vp_integral]);
    Some(pool.add(vec![pv, neg_vp]))
}

/// Match `∫ exp(a·x+c)·sin(b·x+d) dx` / `∫ exp(a·x+c)·cos(b·x+d) dx` (constant
/// `a`, `b`) and build the cyclic integration-by-parts closed form directly:
///
/// ```text
/// ∫ exp(g)·sin(h) dx = exp(g)·(a·sin h − b·cos h)/(a² + b²)
/// ∫ exp(g)·cos(h) dx = exp(g)·(b·sin h + a·cos h)/(a² + b²)
/// ```
///
/// with `g = a·x+c`, `h = b·x+d`.  Constant extra factors are carried through.
/// Soundness-gated; declines anything outside this exact shape (e.g. a leftover
/// polynomial factor — triple products are out of scope).
fn try_exp_trig_ibp(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<ExprId> {
    let args = match pool.get(expr) {
        ExprData::Mul(v) => v,
        _ => return None,
    };

    let mut exp_factor: Option<(usize, ExprId)> = None; // (pos, g)
    let mut trig_factor: Option<(usize, bool, ExprId)> = None; // (pos, is_sin, h)
    for (i, &a) in args.iter().enumerate() {
        if let ExprData::Func { name, args: fargs } = pool.get(a) {
            if fargs.len() == 1 {
                let inner = fargs[0];
                if name == "exp" && is_linear_in(inner, var, pool).is_some() {
                    if exp_factor.is_some() {
                        return None;
                    }
                    exp_factor = Some((i, inner));
                    continue;
                }
                if (name == "sin" || name == "cos") && is_linear_in(inner, var, pool).is_some() {
                    if trig_factor.is_some() {
                        return None;
                    }
                    trig_factor = Some((i, name == "sin", inner));
                    continue;
                }
            }
        }
    }
    let (epos, g) = exp_factor?;
    let (tpos, is_sin, h) = trig_factor?;

    // Every other factor must be constant (free of var) — no leftover polynomial.
    let const_factors: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != epos && i != tpos)
        .map(|(_, &a)| a)
        .collect();
    if !const_factors.iter().all(|&a| is_free_of(a, var, pool)) {
        return None;
    }

    let (a, _c) = is_linear_in(g, var, pool)?;
    let (b, _d) = is_linear_in(h, var, pool)?;

    // Denominator a² + b².
    let two = pool.integer(2_i32);
    let a2 = pool.pow(a, two);
    let b2 = pool.pow(b, two);
    let denom = pool.add(vec![a2, b2]);
    let denom_inv = pool.pow(denom, pool.integer(-1_i32));

    let neg_one = pool.integer(-1_i32);
    let exp_g = pool.func("exp", vec![g]);
    let sin_h = pool.func("sin", vec![h]);
    let cos_h = pool.func("cos", vec![h]);

    let numerator = if is_sin {
        // a·sin h − b·cos h
        let a_sin = pool.mul(vec![a, sin_h]);
        let neg_b_cos = pool.mul(vec![neg_one, b, cos_h]);
        pool.add(vec![a_sin, neg_b_cos])
    } else {
        // b·sin h + a·cos h
        let b_sin = pool.mul(vec![b, sin_h]);
        let a_cos = pool.mul(vec![a, cos_h]);
        pool.add(vec![b_sin, a_cos])
    };

    let mut factors = vec![exp_g, numerator, denom_inv];
    factors.extend_from_slice(&const_factors);
    let result = pool.mul(factors);

    // Soundness gate: only emit when d/dx result equals the integrand.
    if !verify_antiderivative(result, expr, var, pool) {
        return None;
    }
    log.push(RewriteStep::simple("int_exp_trig_ibp", expr, result));
    Some(result)
}

// ---------------------------------------------------------------------------
// Trigonometric powers and products via Fourier linearization
// ---------------------------------------------------------------------------

/// Maximum combined trig degree (number of `sin`/`cos` factors) the Fourier
/// linearizer will expand.  The term count grows as `2^degree`, so this bounds
/// the work; beyond it the fast-path declines and the integrand falls through.
const MAX_TRIG_LINEARIZE_DEGREE: usize = 8;

/// A single term of a finite Fourier expansion: `coeff · f(arg)` with
/// `f ∈ {sin, cos}` and `arg` linear in the integration variable.
struct FourierTerm {
    coeff: ExprId,
    is_sin: bool,
    arg: ExprId,
}

/// Fast-path for `∫ sin^m(a·x+b)·cos^n(c·x+d) dx` (nonnegative integer powers,
/// linear arguments) — covering `sin²`, `cos²`, `sin³`, `sin²·cos²`,
/// different-frequency products like `sin(2x)·cos(x)`, … — plus the small
/// reciprocal-square family `∫ 1/cos² = tan`, `∫ 1/sin² = −cot`,
/// `∫ tan² = tan − x`.
///
/// The product/power case is rewritten into a linear combination of
/// `sin(k·x)`/`cos(k·x)`/constant via product-to-sum identities (a finite
/// Fourier expansion), then each term is integrated with the elementary
/// `∫ sin(k·x) = −cos(k·x)/k`, `∫ cos(k·x) = sin(k·x)/k`, `∫ c = c·x` rules.
/// Every emitted antiderivative is soundness-gated by [`verify_antiderivative`],
/// so a wrong result is never returned; unmatched shapes decline cleanly.
///
/// Terminates without recursing into [`integrate_raw`]: each linearized term is
/// a bare `sin`/`cos` of a linear argument, integrated in closed form here.
fn try_trig_power_product(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<ExprId> {
    // Small reciprocal-square / tan² table first (not Fourier-linearizable).
    if let Some(result) = trig_reciprocal_square_antiderivative(expr, var, pool) {
        if verify_antiderivative(result, expr, var, pool) {
            log.push(RewriteStep::simple("int_trig_reciprocal_sq", expr, result));
            return Some(result);
        }
    }

    // Product/power of sin/cos with linear arguments → Fourier linearization.
    let (coeff, factors) = collect_trig_product(expr, var, pool)?;
    // Require genuine linearization work (combined degree ≥ 2): bare `sin(x)` /
    // `cos(x)` keep their existing dedicated rules and are not intercepted here.
    if factors.len() < 2 || factors.len() > MAX_TRIG_LINEARIZE_DEGREE {
        return None;
    }

    let terms = fourier_expand(coeff, &factors, pool);
    let parts: Vec<ExprId> = terms
        .iter()
        .map(|t| integrate_fourier_term(t, var, pool))
        .collect();
    let result = pool.add(parts);

    // Soundness gate: only emit when d/dx result equals the integrand.
    if !verify_antiderivative(result, expr, var, pool) {
        return None;
    }
    log.push(RewriteStep::simple("int_trig_linearize", expr, result));
    Some(result)
}

/// Maximum power `n` for the reciprocal-trig reductions `∫ secⁿ` / `∫ cscⁿ`.
/// Caps the reduction-formula recursion so a pathological exponent cannot blow
/// up the emitted expression; higher powers decline cleanly.
const MAX_RECIP_TRIG_POWER: i64 = 8;

/// Fast-path for `∫ secⁿ` / `∫ cscⁿ` — integrands that are a **negative integer
/// power** of `sin`/`cos` of a linear argument `u = a·x + b`.
///
/// Because `sec`/`csc` desugar to reciprocals at parse time, the integrand
/// arrives as `cos(u)^(-n)` / `sin(u)^(-n)` (flattened) or as the *nested*
/// `(cos(u)^(-1))^m` shape produced by `sec(u)^m`. Both are recognized here; the
/// exponent is flattened (`(g^p)^q → g^(p·q)`) before dispatch.
///
/// Closed forms (`u = a·x + b`, each divided by `a` for the chain rule):
///   - `n = 1`: `∫ sec = log((1+sin)/cos)`, `∫ csc = log((1−cos)/sin)` — real
///     forms of `log|sec+tan|` and `log|tan(u/2)|`.
///   - `n = 2`: `∫ sec² = tan`, `∫ csc² = −cot`.
///   - `n ≥ 3`: the standard reduction formula, recursing down to the `n∈{1,2}`
///     base cases (capped at [`MAX_RECIP_TRIG_POWER`]).
///
/// Every emitted antiderivative is soundness-gated by [`verify_antiderivative`],
/// so a wrong result is never returned; positive powers (owned by the trig
/// linearization path) and non-linear arguments decline cleanly here.
fn try_reciprocal_trig_power(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<ExprId> {
    let (is_sin, u, n) = detect_reciprocal_trig_power(expr, pool)?;
    let (a, _b) = is_linear_in(u, var, pool)?;

    let u_integral = reciprocal_trig_u_integral(is_sin, u, n, pool)?;
    // Chain rule: ∫ f(a·x+b) dx = (1/a) · [∫ f(u) du].
    let a_inv = pool.pow(a, pool.integer(-1_i32));
    let result = pool.mul(vec![a_inv, u_integral]);

    // Soundness gate: only emit when d/dx result equals the integrand.
    if !verify_antiderivative(result, expr, var, pool) {
        return None;
    }
    log.push(RewriteStep::simple(
        "int_reciprocal_trig_power",
        expr,
        result,
    ));
    Some(result)
}

/// Detect a negative-integer power of `sin`/`cos`, flattening one optional level
/// of nesting `(g^p)^q → g^(p·q)`. Returns `(is_sin, arg, n)` with `n = −exp ≥ 1`,
/// or `None` for any other shape (including zero/positive exponents, which are
/// owned by other paths).
fn detect_reciprocal_trig_power(expr: ExprId, pool: &ExprPool) -> Option<(bool, ExprId, i64)> {
    let ExprData::Pow { base, exp } = pool.get(expr) else {
        return None;
    };
    let outer = as_integer(exp, pool)?;
    // Flatten one optional level of nesting: (g^p)^q → g^(p·q).
    let (fname, arg, total) = match pool.get(base) {
        ExprData::Func { name, args } if args.len() == 1 => (name, args[0], outer),
        ExprData::Pow {
            base: inner_base,
            exp: inner_exp,
        } => {
            let inner = as_integer(inner_exp, pool)?;
            let ExprData::Func { name, args } = pool.get(inner_base) else {
                return None;
            };
            if args.len() != 1 {
                return None;
            }
            (name, args[0], inner.checked_mul(outer)?)
        }
        _ => return None,
    };
    if fname != "sin" && fname != "cos" {
        return None;
    }
    // Only negative powers (the reciprocal family); positive/zero exponents are
    // handled by the trig linearization path.
    if total >= 0 {
        return None;
    }
    Some((fname == "sin", arg, -total))
}

/// Antiderivative of `secⁿ(u)` / `cscⁿ(u)` **with respect to `u`** (the caller
/// applies the chain-rule `1/a` factor). Returns `None` above the recursion cap.
fn reciprocal_trig_u_integral(is_sin: bool, u: ExprId, n: i64, pool: &ExprPool) -> Option<ExprId> {
    if !(1..=MAX_RECIP_TRIG_POWER).contains(&n) {
        return None;
    }
    Some(if is_sin {
        csc_u_integral(u, n, pool)
    } else {
        sec_u_integral(u, n, pool)
    })
}

/// `∫ secⁿ(u) du` via the reduction formula (`sec = 1/cos`), recursing to the
/// `n∈{1,2}` base cases. Assumes `1 ≤ n ≤ MAX_RECIP_TRIG_POWER`.
fn sec_u_integral(u: ExprId, n: i64, pool: &ExprPool) -> ExprId {
    let cos_u = pool.func("cos", vec![u]);
    match n {
        // ∫ sec(u) du = log((1+sin u)/cos u) = log|sec u + tan u|.
        1 => {
            let num = pool.add(vec![pool.integer(1_i32), pool.func("sin", vec![u])]);
            let inv_cos = pool.pow(cos_u, pool.integer(-1_i32));
            let arg = pool.mul(vec![num, inv_cos]);
            pool.func("log", vec![arg])
        }
        // ∫ sec²(u) du = tan(u).
        2 => pool.func("tan", vec![u]),
        // ∫ secⁿ = secⁿ⁻²·tan/(n−1) + (n−2)/(n−1)·∫secⁿ⁻².
        _ => {
            let sec_pow = pool.pow(cos_u, pool.integer(-((n - 2) as i32)));
            let tan_u = pool.func("tan", vec![u]);
            let term1 = pool.mul(vec![pool.rational(1_i32, (n - 1) as i32), sec_pow, tan_u]);
            let rec = sec_u_integral(u, n - 2, pool);
            let term2 = pool.mul(vec![pool.rational((n - 2) as i32, (n - 1) as i32), rec]);
            pool.add(vec![term1, term2])
        }
    }
}

/// `∫ cscⁿ(u) du` via the reduction formula (`csc = 1/sin`, `cot = cos/sin`),
/// recursing to the `n∈{1,2}` base cases. Assumes `1 ≤ n ≤ MAX_RECIP_TRIG_POWER`.
fn csc_u_integral(u: ExprId, n: i64, pool: &ExprPool) -> ExprId {
    let sin_u = pool.func("sin", vec![u]);
    let cos_u = pool.func("cos", vec![u]);
    match n {
        // ∫ csc(u) du = log((1−cos u)/sin u) = log|tan(u/2)| = −log|csc u + cot u|.
        1 => {
            let neg_cos = pool.mul(vec![pool.integer(-1_i32), cos_u]);
            let num = pool.add(vec![pool.integer(1_i32), neg_cos]);
            let inv_sin = pool.pow(sin_u, pool.integer(-1_i32));
            let arg = pool.mul(vec![num, inv_sin]);
            pool.func("log", vec![arg])
        }
        // ∫ csc²(u) du = −cot(u) = −cos(u)/sin(u).
        2 => {
            let inv_sin = pool.pow(sin_u, pool.integer(-1_i32));
            pool.mul(vec![pool.integer(-1_i32), cos_u, inv_sin])
        }
        // ∫ cscⁿ = −cscⁿ⁻²·cot/(n−1) + (n−2)/(n−1)·∫cscⁿ⁻².
        _ => {
            let csc_pow = pool.pow(sin_u, pool.integer(-((n - 2) as i32)));
            let inv_sin = pool.pow(sin_u, pool.integer(-1_i32));
            let cot_u = pool.mul(vec![cos_u, inv_sin]);
            let term1 = pool.mul(vec![pool.rational(-1_i32, (n - 1) as i32), csc_pow, cot_u]);
            let rec = csc_u_integral(u, n - 2, pool);
            let term2 = pool.mul(vec![pool.rational((n - 2) as i32, (n - 1) as i32), rec]);
            pool.add(vec![term1, term2])
        }
    }
}

/// Collect the constant coefficient and the list of `sin`/`cos` factors (with
/// linear arguments) making up a pure trig product/power.  Returns `None` if any
/// `var`-dependent factor is not a nonnegative integer power of `sin`/`cos` of a
/// linear argument, so polynomial·trig, exp·trig, `tan`, negative powers, etc.
/// are left to their dedicated paths.
fn collect_trig_product(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, Vec<(bool, ExprId)>)> {
    let factors: Vec<ExprId> = match pool.get(expr) {
        ExprData::Mul(args) => args,
        ExprData::Pow { .. } => vec![expr],
        _ => return None,
    };

    let mut coeff_factors: Vec<ExprId> = Vec::new();
    let mut trig: Vec<(bool, ExprId)> = Vec::new();
    for f in factors {
        if is_free_of(f, var, pool) {
            coeff_factors.push(f);
            continue;
        }
        if !push_trig_factor(f, var, pool, &mut trig) {
            return None;
        }
        // Guard the `2^degree` blow-up early on a large explicit power.
        if trig.len() > MAX_TRIG_LINEARIZE_DEGREE {
            return None;
        }
    }

    let coeff = match coeff_factors.len() {
        0 => pool.integer(1_i32),
        1 => coeff_factors[0],
        _ => pool.mul(coeff_factors),
    };
    Some((coeff, trig))
}

/// Push one `var`-dependent factor onto `trig` when it is `sin`/`cos` of a
/// linear argument raised to a nonnegative integer power; return `false`
/// otherwise (so the caller declines the whole integrand).
fn push_trig_factor(
    f: ExprId,
    var: ExprId,
    pool: &ExprPool,
    trig: &mut Vec<(bool, ExprId)>,
) -> bool {
    match pool.get(f) {
        ExprData::Func { name, args } if args.len() == 1 => {
            let is_sin = name == "sin";
            if (is_sin || name == "cos") && is_linear_in(args[0], var, pool).is_some() {
                trig.push((is_sin, args[0]));
                true
            } else {
                false
            }
        }
        ExprData::Pow { base, exp } => {
            let Some(n) = as_integer(exp, pool) else {
                return false;
            };
            if !(1..=MAX_TRIG_LINEARIZE_DEGREE as i64).contains(&n) {
                return false;
            }
            match pool.get(base) {
                ExprData::Func { name, args } if args.len() == 1 => {
                    let is_sin = name == "sin";
                    if (is_sin || name == "cos") && is_linear_in(args[0], var, pool).is_some() {
                        for _ in 0..n {
                            trig.push((is_sin, args[0]));
                        }
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// Expand `coeff · Π f_i(arg_i)` (each `f_i ∈ {sin, cos}`, `arg_i` linear) into a
/// finite Fourier sum `Σ c_j · g_j(θ_j)` via product-to-sum identities.  Every
/// output argument stays linear in the integration variable, so each term
/// integrates in closed form.
fn fourier_expand(coeff: ExprId, factors: &[(bool, ExprId)], pool: &ExprPool) -> Vec<FourierTerm> {
    let neg_one = pool.integer(-1_i32);
    let half = pool.rational(1_i32, 2_i32);
    // Seed with `coeff · cos(0)` (= coeff), the multiplicative identity.
    let mut terms = vec![FourierTerm {
        coeff,
        is_sin: false,
        arg: pool.integer(0_i32),
    }];

    for &(g_sin, u) in factors {
        let mut next: Vec<FourierTerm> = Vec::with_capacity(terms.len() * 2);
        for t in &terms {
            let hc = pool.mul(vec![half, t.coeff]);
            let neg_hc = pool.mul(vec![neg_one, hc]);
            let a = t.arg;
            let neg_a = pool.mul(vec![neg_one, a]);
            let neg_u = pool.mul(vec![neg_one, u]);
            let u_plus_a = simplify(pool.add(vec![u, a]), pool).value;
            let u_minus_a = simplify(pool.add(vec![u, neg_a]), pool).value;
            let a_minus_u = simplify(pool.add(vec![a, neg_u]), pool).value;
            match (g_sin, t.is_sin) {
                // sin(u)·cos(A) = ½[sin(u+A) + sin(u−A)]
                (true, false) => {
                    next.push(FourierTerm {
                        coeff: hc,
                        is_sin: true,
                        arg: u_plus_a,
                    });
                    next.push(FourierTerm {
                        coeff: hc,
                        is_sin: true,
                        arg: u_minus_a,
                    });
                }
                // sin(u)·sin(A) = ½[cos(u−A) − cos(u+A)]
                (true, true) => {
                    next.push(FourierTerm {
                        coeff: hc,
                        is_sin: false,
                        arg: u_minus_a,
                    });
                    next.push(FourierTerm {
                        coeff: neg_hc,
                        is_sin: false,
                        arg: u_plus_a,
                    });
                }
                // cos(u)·cos(A) = ½[cos(u−A) + cos(u+A)]
                (false, false) => {
                    next.push(FourierTerm {
                        coeff: hc,
                        is_sin: false,
                        arg: u_minus_a,
                    });
                    next.push(FourierTerm {
                        coeff: hc,
                        is_sin: false,
                        arg: u_plus_a,
                    });
                }
                // cos(u)·sin(A) = ½[sin(A+u) + sin(A−u)]
                (false, true) => {
                    next.push(FourierTerm {
                        coeff: hc,
                        is_sin: true,
                        arg: u_plus_a,
                    });
                    next.push(FourierTerm {
                        coeff: hc,
                        is_sin: true,
                        arg: a_minus_u,
                    });
                }
            }
        }
        terms = next;
    }
    terms
}

/// Integrate one Fourier term `c · f(arg)` (arg linear in `var`) in closed form:
/// `∫ c·sin(k·x+φ) = −c·cos(k·x+φ)/k`, `∫ c·cos(k·x+φ) = c·sin(k·x+φ)/k`, and
/// `∫ c·f(const) dx = c·f(const)·x` when `arg` is free of `var`.
fn integrate_fourier_term(t: &FourierTerm, var: ExprId, pool: &ExprPool) -> ExprId {
    match is_linear_in(t.arg, var, pool) {
        Some((a, _b)) => {
            let a_inv = pool.pow(a, pool.integer(-1_i32));
            if t.is_sin {
                // ∫ c·sin(arg) = −c·cos(arg)/a
                let cos_arg = pool.func("cos", vec![t.arg]);
                pool.mul(vec![pool.integer(-1_i32), t.coeff, a_inv, cos_arg])
            } else {
                // ∫ c·cos(arg) = c·sin(arg)/a
                let sin_arg = pool.func("sin", vec![t.arg]);
                pool.mul(vec![t.coeff, a_inv, sin_arg])
            }
        }
        None => {
            // arg free of var ⇒ f(arg) is constant ⇒ ∫ c·f(arg) dx = c·f(arg)·x.
            let name = if t.is_sin { "sin" } else { "cos" };
            let f = pool.func(name, vec![t.arg]);
            pool.mul(vec![t.coeff, f, var])
        }
    }
}

/// True when `expr` contains at least one `sin`/`cos`/`tan` applied to exactly
/// `var`.  Cheap pre-filter for the Weierstrass path so it never allocates the
/// half-angle symbol for a non-trig integrand.
fn contains_trig_of_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { name, args } if args.len() == 1 => {
            (matches!(name.as_str(), "sin" | "cos" | "tan") && args[0] == var)
                || contains_trig_of_var(args[0], var, pool)
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_trig_of_var(a, var, pool))
        }
        ExprData::Pow { base, exp } => {
            contains_trig_of_var(base, var, pool) || contains_trig_of_var(exp, var, pool)
        }
        _ => false,
    }
}

/// True when `expr` contains a genuine rational-trig denominator: a negative
/// integer power of an `Add` node that itself contains a trig function of `var`
/// (e.g. `(2+cos x)^(-1)`, `(sin x + cos x)^(-1)`, `(1+sin x)^(-2)`).
///
/// This is the trigger for the Weierstrass path.  It deliberately excludes bare
/// `sin`/`cos`/`tan`, pure powers/products of trig, and `secⁿ`/`cscⁿ`
/// (reciprocal powers of a single trig *function*, whose base is a `Func`, not an
/// `Add`) — all of which the dedicated fast-paths and rules already handle with
/// nicer closed forms.
fn has_rational_trig_denominator(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let negative = as_integer(exp, pool).map(|n| n < 0).unwrap_or(false);
            if negative
                && matches!(pool.get(base), ExprData::Add(_))
                && contains_trig_of_var(base, var, pool)
            {
                return true;
            }
            has_rational_trig_denominator(base, var, pool)
                || has_rational_trig_denominator(exp, var, pool)
        }
        ExprData::Add(args) | ExprData::Mul(args) => args
            .iter()
            .any(|&a| has_rational_trig_denominator(a, var, pool)),
        ExprData::Func { args, .. } => args
            .iter()
            .any(|&a| has_rational_trig_denominator(a, var, pool)),
        _ => false,
    }
}

/// Structurally rewrite `expr` — a rational function of `sin(var)`, `cos(var)`,
/// and `tan(var)` (argument exactly `var`) — into the half-angle variable `t`,
/// using `sin x = 2t/(1+t²)`, `cos x = (1−t²)/(1+t²)`, `tan x = 2t/(1−t²)`.
///
/// Returns `None` when `expr` is not rational in those trig functions of `var`:
/// e.g. it contains a bare `var`, an `exp(x)`/`log(x)`/inverse-trig call, a
/// power with a `var`-dependent exponent, or a trig call whose argument is not
/// exactly `var` (`sin(2x)`, `cos(x²)`, …).  Constants (free of `var`) pass
/// through unchanged.
fn weierstrass_rewrite(expr: ExprId, var: ExprId, t: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if is_free_of(expr, var, pool) {
        return Some(expr);
    }
    if expr == var {
        // A bare occurrence of the integration variable is not rational-in-trig.
        return None;
    }

    let one = pool.integer(1_i32);
    let two = pool.integer(2_i32);
    let neg_one = pool.integer(-1_i32);
    let t2 = pool.pow(t, two);
    let one_plus_t2 = pool.add(vec![one, t2]);
    let one_minus_t2 = pool.add(vec![one, pool.mul(vec![neg_one, t2])]);

    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut out = Vec::with_capacity(args.len());
            for a in args {
                out.push(weierstrass_rewrite(a, var, t, pool)?);
            }
            Some(pool.add(out))
        }
        ExprData::Mul(args) => {
            let mut out = Vec::with_capacity(args.len());
            for a in args {
                out.push(weierstrass_rewrite(a, var, t, pool)?);
            }
            Some(pool.mul(out))
        }
        ExprData::Pow { base, exp } => {
            // The exponent must be a constant (free of `var`) — e.g. the `−1` in
            // a denominator, or a positive integer power of sin/cos.
            if !is_free_of(exp, var, pool) {
                return None;
            }
            let new_base = weierstrass_rewrite(base, var, t, pool)?;
            Some(pool.pow(new_base, exp))
        }
        ExprData::Func { name, args } if args.len() == 1 && args[0] == var => match name.as_str() {
            "sin" => Some(pool.mul(vec![two, t, pool.pow(one_plus_t2, neg_one)])),
            "cos" => Some(pool.mul(vec![one_minus_t2, pool.pow(one_plus_t2, neg_one)])),
            "tan" => Some(pool.mul(vec![two, t, pool.pow(one_minus_t2, neg_one)])),
            _ => None,
        },
        _ => None,
    }
}

/// Integrate a rational function of `sin(var)`/`cos(var)`/`tan(var)` (single
/// frequency, argument exactly `var`) via the Weierstrass half-angle
/// substitution `t = tan(x/2)`:
///
/// ```text
/// sin x = 2t/(1+t²),  cos x = (1−t²)/(1+t²),  tan x = 2t/(1−t²),  dx = 2/(1+t²) dt.
/// ```
///
/// The integrand is rewritten as a rational function of `t`, integrated through
/// the full elementary pipeline (partial fractions / Rothstein–Trager / atan /
/// log), and back-substituted `t ↦ tan(x/2)`.
///
/// Placed *after* the dedicated trig fast-paths in [`integrate_raw`], so it only
/// catches genuinely rational-in-trig integrands those decline (e.g.
/// `1/(2+cos x)`); the nicer closed forms for `∫sin²`, `∫sec²`, `∫sin(2x)cos(x)`
/// are untouched.  Soundness-gated by [`verify_antiderivative`]: the candidate
/// is returned only when `d/dx result = integrand`, so a wrong antiderivative is
/// never produced.  Declines cleanly (`Ok(None)`) when the integrand is not
/// rational in trig or the `t`-integral does not close.
///
/// # Why this returns a `Result`
///
/// The `t`-integral is a *whole nested `integrate` call*, and the half-angle
/// substitution doubles the degree — `∫ 1/(sin⁹x + sin x + 1) dx` becomes a
/// degree-18 rational function, which measured **110 s** end to end. That inner
/// call has cooperative checkpoints of its own, but `.ok()?` threw their verdict
/// away exactly as `try_u_substitution` did, so the budget could not stop the
/// single most expensive route in the elementary integrator. A budget error now
/// propagates; a genuine decline still returns `Ok(None)`.
fn try_weierstrass_rational_trig(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<Option<ExprId>, IntegrationError> {
    // Only fire on genuine rational-trig integrands (a trig-containing sum in a
    // denominator); bare/product/power trig keep their nicer dedicated forms.
    if !has_rational_trig_denominator(expr, var, pool) {
        return Ok(None);
    }

    // Fresh half-angle variable t = tan(x/2).
    let t = pool.symbol("__weierstrass_t", crate::kernel::Domain::Real);

    // Rewrite the integrand as a rational function of t.
    let Some(g_body) = weierstrass_rewrite(expr, var, t, pool) else {
        return Ok(None);
    };

    // Jacobian: dx = 2/(1+t²) dt.
    let one = pool.integer(1_i32);
    let t2 = pool.pow(t, pool.integer(2_i32));
    let one_plus_t2 = pool.add(vec![one, t2]);
    let jac = pool.mul(vec![
        pool.integer(2_i32),
        pool.pow(one_plus_t2, pool.integer(-1_i32)),
    ]);
    let g = simplify(pool.mul(vec![g_body, jac]), pool).value;

    // Integrate the rational function in t through the full elementary pipeline.
    // `g` is rational in `t` with no trig of `t`, so this path cannot re-fire and
    // recursion is bounded.
    // This route ends at the `verify_antiderivative` gate below, which can never
    // accept a `RootSum` (`simplify` makes it an opaque atom and `eval_interp`
    // cannot evaluate one).  Tell the rational integrator so, and it declines
    // before paying for the Lazard–Rioboo–Trager number-field GCD instead of
    // after — same answer, without the dominant cost of this route.
    let inner = {
        let _no_root_sum = super::risch::rational_integrate::RootSumSuppressed::enter();
        match integrate(g, t, pool) {
            Ok(inner) => inner,
            // Not this route declining — the caller wants out.
            Err(e) if e.is_budget() => return Err(e),
            Err(_) => return Ok(None),
        }
    };

    // Back-substitute t = tan(x/2).
    let half = pool.rational(1_i32, 2_i32);
    let half_x = pool.mul(vec![half, var]);
    let tan_half = pool.func("tan", vec![half_x]);
    let mut back = HashMap::new();
    back.insert(t, tan_half);
    let result = simplify(crate::kernel::subs(inner.value, &back, pool), pool).value;

    // Soundness gate: d/dx(result) must equal the original integrand.
    if !verify_antiderivative(result, expr, var, pool) {
        return Ok(None);
    }
    log.push(RewriteStep::simple("int_weierstrass_trig", expr, result));
    Ok(Some(result))
}

/// Small explicit table for `∫ 1/cos²(u) = tan(u)/a`, `∫ 1/sin²(u) = −cot(u)/a`
/// (emitted as `−cos(u)/(a·sin(u))` so the result differentiates through the
/// registered primitives), and `∫ tan²(u) = tan(u)/a − x`, with `u = a·x+b`
/// linear in `var`.  Returns an unverified candidate; the caller gates it with
/// [`verify_antiderivative`].
fn trig_reciprocal_square_antiderivative(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let ExprData::Pow { base, exp } = pool.get(expr) else {
        return None;
    };
    let n = as_integer(exp, pool)?;
    let ExprData::Func { name, args } = pool.get(base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let u = args[0];
    let (a, _b) = is_linear_in(u, var, pool)?;
    let a_inv = pool.pow(a, pool.integer(-1_i32));
    let neg_one = pool.integer(-1_i32);

    match (name.as_str(), n) {
        // ∫ sec²(u) dx = tan(u)/a
        ("cos", -2) => {
            let tan_u = pool.func("tan", vec![u]);
            Some(pool.mul(vec![a_inv, tan_u]))
        }
        // ∫ csc²(u) dx = −cot(u)/a, written as −cos(u)/(a·sin(u)).
        ("sin", -2) => {
            let cos_u = pool.func("cos", vec![u]);
            let sin_inv = pool.pow(pool.func("sin", vec![u]), neg_one);
            Some(pool.mul(vec![neg_one, a_inv, cos_u, sin_inv]))
        }
        // ∫ tan²(u) dx = tan(u)/a − x
        ("tan", 2) => {
            let tan_u = pool.func("tan", vec![u]);
            let first = pool.mul(vec![a_inv, tan_u]);
            let neg_x = pool.mul(vec![neg_one, var]);
            Some(pool.add(vec![first, neg_x]))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Known non-elementary pre-check (Risch Gap 6)
// ---------------------------------------------------------------------------

/// Transcendental functions `f` for which `∫ f(linear)/poly dx` is a classic
/// non-elementary special function (Liouville's theorem):
///   - `exp` → exponential integral `Ei`
///   - `sin` → sine integral `Si`
///   - `cos` → cosine integral `Ci`
///   - `sinh` → hyperbolic sine integral `Shi`
///   - `cosh` → hyperbolic cosine integral `Chi`
fn special_integral_name(func: &str) -> Option<&'static str> {
    match func {
        "exp" => Some("Ei"),
        "sin" => Some("Si"),
        "cos" => Some("Ci"),
        "sinh" => Some("Shi"),
        "cosh" => Some("Chi"),
        _ => None,
    }
}

/// Return `true` if `exp` is a negative integer literal.
fn is_negative_integer(exp: ExprId, pool: &ExprPool) -> bool {
    as_integer(exp, pool).is_some_and(|n| n < 0)
}

/// Return `true` if `expr` is a polynomial in `var` (integer powers only).
fn is_polynomial_in(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var || is_free_of(expr, var, pool) {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().all(|&a| is_polynomial_in(a, var, pool))
        }
        ExprData::Pow { base, exp } => {
            is_polynomial_in(base, var, pool) && as_integer(exp, pool).is_some_and(|n| n >= 0)
        }
        _ => false,
    }
}

/// Return `true` if `base` is a non-constant polynomial in `var` that can appear
/// as a denominator in a known non-elementary form.  Dividing a special
/// transcendental `f(linear)` by *any* non-constant polynomial yields an
/// Ei/Si/Ci/Shi/Chi-family integral, so this is a sound `NonElementary`
/// certificate (Liouville's theorem), not a guess.
fn is_simple_denominator_base(base: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    !is_free_of(base, var, pool) && is_polynomial_in(base, var, pool)
}

/// Structural pre-check certifying that `expr` is a provably non-elementary
/// integrand of one of the classic special-function families.  Returns a
/// human-readable description (used in the `NonElementary` message) on a match.
///
/// Recognised forms (with `g`, `D` linear and non-constant in `var`, and every
/// other factor free of `var`):
///   - `c · f(g) · D^(-n)` with `f ∈ {exp, sin, cos, sinh, cosh}` → `Ei/Si/Ci/Shi/Chi`
///   - `c · log(g)^(-n)` → logarithmic integral `li`
///
/// These are non-elementary by Liouville's theorem (Bronstein 2005, §1.2).  The
/// matcher is intentionally narrow: the *only* `var`-dependent factors allowed
/// are the transcendental numerator and the polynomial denominator, so it never
/// fires on cancelling cases such as `x²·sin(x)/x = x·sin(x)` (elementary).
fn known_nonelementary(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<String> {
    // A single `log(g)^(-n)` factor (not wrapped in a Mul) is the bare `li` case.
    if let Some(msg) = match_log_denominator(expr, var, pool) {
        return Some(msg);
    }

    let args = match pool.get(expr) {
        ExprData::Mul(args) => args,
        _ => return None,
    };

    let mut special: Option<String> = None; // f(g) with f a special transcendental
    let mut has_poly_denom = false; // a D^(-n) factor
    let mut log_denom: Option<String> = None; // a log(g)^(-n) factor (li)

    for &a in &args {
        // Constant factor — always allowed.
        if is_free_of(a, var, pool) {
            continue;
        }

        // Transcendental numerator f(g), f special, g linear non-constant.
        if let ExprData::Func { ref name, ref args } = pool.get(a) {
            if args.len() == 1
                && special_integral_name(name).is_some()
                && is_linear_in(args[0], var, pool).is_some()
            {
                if special.is_some() {
                    return None; // two interacting specials — out of scope
                }
                special = Some(pool.display(a).to_string());
                continue;
            }
        }

        // Denominator factor D^(-n).
        if let ExprData::Pow { base, exp } = pool.get(a) {
            if is_negative_integer(exp, pool) {
                if let Some(msg) = match_log_denominator(a, var, pool) {
                    if log_denom.is_some() {
                        return None;
                    }
                    log_denom = Some(msg);
                    continue;
                }
                if is_simple_denominator_base(base, var, pool) {
                    has_poly_denom = true;
                    continue;
                }
            }
        }

        // Any other factor involving `var` breaks the recognised shape.
        return None;
    }

    if let (Some(f), true) = (&special, has_poly_denom) {
        return Some(format!(
            "{f} divided by a polynomial gives a special-function integral \
             (Ei/Si/Ci/Shi/Chi), which is not elementary (Liouville's theorem)"
        ));
    }

    if let Some(msg) = log_denom {
        return Some(msg);
    }

    None
}

/// Match a `log(linear)^(-n)` factor (`1/log` family → logarithmic integral `li`).
fn match_log_denominator(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<String> {
    let ExprData::Pow { base, exp } = pool.get(expr) else {
        return None;
    };
    if !is_negative_integer(exp, pool) {
        return None;
    }
    let ExprData::Func { ref name, ref args } = pool.get(base) else {
        return None;
    };
    if name == "log" && args.len() == 1 && is_linear_in(args[0], var, pool).is_some() {
        Some(format!(
            "1/{} is the logarithmic integral li, which is not elementary \
             (Liouville's theorem)",
            pool.display(base)
        ))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Core integration (no simplification yet)
// ---------------------------------------------------------------------------

/// Crate-internal entry to the rule-based integrator (no algebraic dispatch).
/// Used by the algebraic engine to integrate the rational part A(x).
pub(crate) fn integrate_raw(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // Cooperative checkpoint. This is the rule engine's dispatcher: it recurses
    // per summand (sum rule) and per non-constant factor (constant-multiple
    // rule), and several of the route helpers it tries below — the Weierstrass
    // half-angle substitution in particular — run a whole nested `integrate`.
    // Without a check here the only checkpoints on the elementary route were the
    // two at depth 0, so `∫ f₁ + … + f₈` of eight hard rational terms could not
    // be stopped between terms at all.
    crate::budget::check()?;

    // Fast-path: ∫ c * x * exp(x) dx = c * exp(x) * (x - 1)
    if let Some(result) = try_x_times_func(expr, var, pool, log) {
        return Ok(result);
    }

    // Inverse-trigonometric integration by parts:
    //   ∫ rest(x)·f(x) dx with f ∈ {atan, asin, acos} and arg == var.
    // Handles both the bare case (∫ atan(x) dx) and the product case
    // (∫ x·atan(x) dx). Soundness-gated inside the helper.
    if let Some(result) = try_inverse_trig_ibp(expr, var, pool, log) {
        return Ok(result);
    }

    // Polynomial × trig product via repeated integration by parts:
    //   ∫ p(x)·sin(a·x+b) dx, ∫ p(x)·cos(a·x+b) dx  (p polynomial, linear arg).
    // Soundness-gated inside the helper.
    if let Some(result) = try_poly_trig_ibp(expr, var, pool, log) {
        return Ok(result);
    }

    // Exponential × trig product via the cyclic IBP closed form:
    //   ∫ exp(a·x+c)·sin(b·x+d) dx, ∫ exp(a·x+c)·cos(b·x+d) dx.
    // Soundness-gated inside the helper.
    if let Some(result) = try_exp_trig_ibp(expr, var, pool, log) {
        return Ok(result);
    }

    // Powers and products of sin/cos (and the small 1/cos², 1/sin², tan² family):
    //   ∫ sin^m(a·x+b)·cos^n(c·x+d) dx via Fourier linearization + termwise
    //   integration, ∫ 1/cos² = tan, ∫ 1/sin² = −cot, ∫ tan² = tan − x.
    // Soundness-gated inside the helper; does not recurse into integrate_raw.
    if let Some(result) = try_trig_power_product(expr, var, pool, log) {
        return Ok(result);
    }

    // Negative integer powers of sin/cos (i.e. ∫ secⁿ / ∫ cscⁿ), which arrive as
    // reciprocal-power expressions because sec/csc desugar at parse time:
    //   ∫ 1/cos = log((1+sin)/cos), ∫ 1/sin = log((1−cos)/sin), ∫ sec² = tan,
    //   ∫ csc² = −cot, and ∫ secⁿ / ∫ cscⁿ (n ≥ 3) via the reduction formula.
    // Recognizes both the flattened `cos(x)^(-n)` and the nested `(cos(x)^(-1))^m`
    // shapes. Soundness-gated inside the helper.
    if let Some(result) = try_reciprocal_trig_power(expr, var, pool, log) {
        return Ok(result);
    }

    // Rational functions of sin/cos/tan (single frequency, argument `var`) via
    // the Weierstrass half-angle substitution t = tan(x/2).  Placed AFTER the
    // dedicated trig fast-paths so it only catches genuinely rational-in-trig
    // integrands they decline (e.g. 1/(2+cos x), 1/(1+sin x)); the nicer closed
    // forms for ∫sin², ∫sec², ∫sin(2x)cos(x) are preserved.  Soundness-gated in
    // the helper.
    if let Some(result) = try_weierstrass_rational_trig(expr, var, pool, log)? {
        return Ok(result);
    }

    // Snapshot node type without holding the lock during recursive calls.
    enum Node {
        IsVar,
        Constant,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow { base: ExprId, exp: ExprId },
        Func { name: String, arg: ExprId },
        Unknown,
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Symbol { .. } if expr == var => Node::IsVar,
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => Node::Constant,
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow {
            base: *base,
            exp: *exp,
        },
        ExprData::Func { name, args } if args.len() == 1 => Node::Func {
            name: name.clone(),
            arg: args[0],
        },
        _ => Node::Unknown,
    });

    match node {
        // ∫ x dx = x²/2
        Node::IsVar => {
            let two = pool.integer(2_i32);
            let inv_two = pool.pow(two, pool.integer(-1_i32));
            let result = pool.mul(vec![pool.pow(var, two), inv_two]);
            log.push(RewriteStep::simple("int_power_rule", expr, result));
            Ok(result)
        }

        // ∫ c dx = c*x  (c free of var)
        Node::Constant => {
            let result = pool.mul(vec![expr, var]);
            log.push(RewriteStep::simple("int_constant_rule", expr, result));
            Ok(result)
        }

        // Sum rule: ∫(f + g + …) = ∫f + ∫g + …
        Node::Add(args) => {
            let mut int_args = Vec::with_capacity(args.len());
            for a in &args {
                let ia = integrate_raw(*a, var, pool, log)?;
                int_args.push(ia);
            }
            let result = pool.add(int_args);
            log.push(RewriteStep::simple("int_sum_rule", expr, result));
            Ok(result)
        }

        // Constant-multiple / power rule for Mul
        Node::Mul(args) => {
            // Partition args into constants (free of var) and non-constants
            let (consts, non_consts): (Vec<ExprId>, Vec<ExprId>) =
                args.iter().partition(|&&a| is_free_of(a, var, pool));

            if non_consts.is_empty() {
                // All factors are constants — treat whole expression as constant
                let result = pool.mul(vec![expr, var]);
                log.push(RewriteStep::simple("int_constant_rule", expr, result));
                return Ok(result);
            }

            // Build the non-constant part
            let inner = match non_consts.len() {
                1 => non_consts[0],
                _ => pool.mul(non_consts.clone()),
            };

            // Build the constant factor
            let const_factor = match consts.len() {
                0 => None,
                1 => Some(consts[0]),
                _ => Some(pool.mul(consts.clone())),
            };

            // Guard against self-recursion: if no constant factor was split off,
            // `inner` is the same product we started with, and recursing would loop
            // forever (this previously crashed the process with a stack overflow on
            // inputs like `sin(x)/x` or `exp(x)/x`).  Bail out cleanly instead.
            if inner == expr {
                return Err(IntegrationError::NotImplemented(format!(
                    "∫ {} — irreducible product of var-dependent factors",
                    pool.display(expr)
                )));
            }

            // Integrate the non-constant part
            let int_inner = integrate_raw(inner, var, pool, log)?;

            let result = match const_factor {
                None => int_inner,
                Some(c) => {
                    let r = pool.mul(vec![c, int_inner]);
                    log.push(RewriteStep::simple("int_constant_multiple_rule", expr, r));
                    r
                }
            };
            Ok(result)
        }

        // Power rule: ∫ f^n dx
        Node::Pow { base, exp } => {
            // Check if exponent is a constant integer
            let n_opt = as_integer(exp, pool);

            if let Some(n) = n_opt {
                if base == var {
                    if n == -1 {
                        // ∫ x^(-1) dx = ln(x)
                        let result = pool.func("log", vec![var]);
                        log.push(RewriteStep::simple("log_rule", expr, result));
                        return Ok(result);
                    }
                    // ∫ x^n dx = x^(n+1) / (n+1)
                    let np1 = pool.integer(n + 1);
                    let inv_np1 = pool.pow(np1, pool.integer(-1_i32));
                    let result = pool.mul(vec![pool.pow(var, np1), inv_np1]);
                    log.push(RewriteStep::simple("int_power_rule", expr, result));
                    return Ok(result);
                }

                // ∫ 1/(a*x + b) dx = log(a*x + b) / a
                if n == -1 {
                    if let Some((a, _b)) = is_linear_in(base, var, pool) {
                        let log_base = pool.func("log", vec![base]);
                        let a_inv = pool.pow(a, pool.integer(-1_i32));
                        let result = pool.mul(vec![a_inv, log_base]);
                        log.push(RewriteStep::simple("int_linear_inv", expr, result));
                        return Ok(result);
                    }
                }

                // base is free of var: ∫ c^n dx = c^n * x
                if is_free_of(base, var, pool) {
                    let result = pool.mul(vec![expr, var]);
                    log.push(RewriteStep::simple("int_constant_rule", expr, result));
                    return Ok(result);
                }
            }

            Err(IntegrationError::NotImplemented(
                "∫ (expr)^(exp) where base or exp is non-trivial".to_string(),
            ))
        }

        // Named single-argument functions
        Node::Func { name, arg } => {
            if arg != var {
                // Only handle f(x) directly; chain rule is out of scope
                if is_free_of(arg, var, pool) {
                    // ∫ f(c) dx = f(c) * x
                    let result = pool.mul(vec![expr, var]);
                    log.push(RewriteStep::simple("int_constant_rule", expr, result));
                    return Ok(result);
                }
                // ∫ exp(a*x + b) dx = exp(a*x + b) / a
                if name == "exp" {
                    if let Some((a, _b)) = is_linear_in(arg, var, pool) {
                        let exp_expr = pool.func("exp", vec![arg]);
                        let a_inv = pool.pow(a, pool.integer(-1_i32));
                        let result = pool.mul(vec![a_inv, exp_expr]);
                        log.push(RewriteStep::simple("int_exp_linear", expr, result));
                        return Ok(result);
                    }
                }
                return Err(IntegrationError::NotImplemented(format!(
                    "∫ {name}(non-trivial arg) — chain rule not implemented"
                )));
            }
            match name.as_str() {
                // ∫ sin(x) dx = -cos(x)
                "sin" => {
                    let neg_one = pool.integer(-1_i32);
                    let result = pool.mul(vec![neg_one, pool.func("cos", vec![var])]);
                    log.push(RewriteStep::simple("int_sin", expr, result));
                    Ok(result)
                }
                // ∫ cos(x) dx = sin(x)
                "cos" => {
                    let result = pool.func("sin", vec![var]);
                    log.push(RewriteStep::simple("int_cos", expr, result));
                    Ok(result)
                }
                // ∫ exp(x) dx = exp(x)
                "exp" => {
                    let result = pool.func("exp", vec![var]);
                    log.push(RewriteStep::simple("int_exp", expr, result));
                    Ok(result)
                }
                // ∫ log(x) dx = x*log(x) - x  (integration by parts)
                "log" => {
                    let log_x = pool.func("log", vec![var]);
                    let x_log_x = pool.mul(vec![var, log_x]);
                    let neg_x = pool.mul(vec![pool.integer(-1_i32), var]);
                    let result = pool.add(vec![x_log_x, neg_x]);
                    log.push(RewriteStep::simple("int_log", expr, result));
                    Ok(result)
                }
                "sqrt" => Err(IntegrationError::NotImplemented(
                    "∫ sqrt(x) — not in the supported Risch subset".to_string(),
                )),
                other => Err(IntegrationError::NotImplemented(format!("∫ {other}(x)"))),
            }
        }

        Node::Unknown => Err(IntegrationError::NotImplemented(
            "unsupported expression node".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Symbolically integrate `expr` with respect to `var`.
///
/// Returns the antiderivative (without the constant of integration) after
/// applying the rule-based simplifier.  The derivation log records every
/// rule applied.
///
/// # Routing
///
/// Integrands are dispatched in this order:
///
/// 1. **Algebraic** (contains `sqrt` or fractional powers) → `algebraic` engine.
/// 2. **Transcendental Risch** (contains `exp(g)` with `deg(g) ≥ 2`, `poly·exp`,
///    `log^n` for `n ≥ 2`, or `poly·log`) → `risch` engine.
/// 3. **Rule-based** fallback for simpler cases already in the table.
///
/// # Supported operations (rule-based)
///
/// | Input              | Result                      | Rule                    |
/// |--------------------|-----------------------------|-------------------------|
/// | `c` (constant)     | `c·x`                       | `constant_rule`         |
/// | `x^n` (n≠-1)      | `x^(n+1)/(n+1)`             | `power_rule`            |
/// | `x^(-1)`           | `ln(x)`                     | `log_rule`              |
/// | `f + g`            | `∫f + ∫g`                   | `sum_rule`              |
/// | `c · f`            | `c · ∫f`                    | `constant_multiple_rule`|
/// | `sin(x)`           | `-cos(x)`                   | `int_sin`               |
/// | `cos(x)`           | `sin(x)`                    | `int_cos`               |
/// | `exp(x)`           | `exp(x)`                    | `int_exp`               |
/// | `exp(a*x + b)`     | `exp(a*x+b) / a`            | `int_exp_linear`        |
/// | `log(x)`           | `x*log(x) - x`              | `int_log`               |
/// | `x * exp(x)`       | `exp(x)*(x-1)`              | `int_x_exp`             |
/// | `1/(a*x + b)`      | `log(a*x+b) / a`            | `int_linear_inv`        |
/// | `atan(x)`          | `x*atan(x) - ½log(1+x²)`   | `int_inverse_trig_ibp`  |
/// | `asin(x)`          | `x*asin(x) + √(1-x²)`      | `int_inverse_trig_ibp`  |
/// | `acos(x)`          | `x*acos(x) - √(1-x²)`      | `int_inverse_trig_ibp`  |
/// | `asinh(x)`         | `x*asinh(x) - √(x²+1)`     | `int_inverse_trig_ibp`  |
/// | `acosh(x)`         | `x*acosh(x) - √(x²-1)`     | `int_inverse_trig_ibp`  |
/// | `atanh(x)`         | `x*atanh(x) + ½log(1-x²)`  | `int_inverse_trig_ibp`  |
/// | `rest(x)*atan(x)`  | IBP: `P*atan - ∫P·f'`      | `int_inverse_trig_ibp`  |
/// | `p(x)*sin(a·x+b)`  | repeated IBP (tabular)      | `int_poly_trig_ibp`     |
/// | `p(x)*cos(a·x+b)`  | repeated IBP (tabular)      | `int_poly_trig_ibp`     |
/// | `exp(a·x)*sin(b·x)`| cyclic IBP closed form      | `int_exp_trig_ibp`      |
/// | `exp(a·x)*cos(b·x)`| cyclic IBP closed form      | `int_exp_trig_ibp`      |
///
/// # Transcendental Risch (Risch engine)
///
/// | Input                      | Result                      | Condition              |
/// |----------------------------|-----------------------------|------------------------|
/// | `exp(g)`, deg(g) ≥ 2      | `v·exp(g)` (if elementary)  | Risch DE solvable      |
/// | `exp(g)`, deg(g) ≥ 2      | `NonElementary`             | Risch DE unsolvable    |
/// | `p(x)·exp(a·x+b)`, deg≥1  | polynomial · exp            | RDE / undetermined coeff. (`x·exp(x)` itself stays in the rule-based `int_x_exp` table) |
/// | `log(h)^n`, n ≥ 2         | polynomial in log           | IBP reduction          |
/// | `p(x)·log(h)`              | polynomial · log            | IBP reduction          |
///
/// # Verification
///
/// For all supported inputs, `diff(integrate(f, x), x)` should simplify to
/// `f` (modulo simplification of the constant rule).  The property tests in
/// this module verify this on random polynomials.
pub fn integrate(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    // Cooperative budget checkpoint (P1 search plumbing item 4): the single
    // entry point every public integration route passes through, so a fan-out
    // loop over many candidates can bound wall-clock/step cost or request
    // cancellation without waiting for an OS-level kill. No-op unless the
    // caller entered a `budget::Budget` — see `crate::budget`.
    crate::budget::check()?;

    // V1-2: Route algebraic integrands to the Trager/Risch algebraic engine.
    // For *mixed* algebraic+transcendental (e.g. exp(x)/sqrt(x²+1)) the Risch
    // engine handles the transcendental level and delegates base-field integrals
    // back to the algebraic engine, so only route to algebraic when there are NO
    // transcendental (exp/log) generators.
    let has_algebraic = super::algebraic::contains_algebraic_subterm(expr, pool)
        || super::algebraic::contains_algebraic_func_of_var(expr, var, pool);
    let has_transcendental = super::risch::contains_risch_form(expr, var, pool);
    // An inverse-trig factor (atan/asin/…·√…) is outside the pure-algebraic
    // engine's scope — it rejects such `B(x)·√(quadratic)` integrands.  Skip the
    // algebraic route in that case so the integrand falls through to the rule
    // engine and the derivative-divides u-substitution (which resolves the
    // `f(x)·f'(x)` sub-integrals produced by the inverse-trig IBP reduction).
    if has_algebraic && !has_transcendental && !contains_inverse_trig(expr, pool) {
        return super::algebraic::integrate_algebraic(expr, var, pool);
    }

    // V2+: Route transcendental Risch cases (exp polynomial, log powers, etc.)
    // Also covers mixed algebraic+transcendental (has_algebraic && has_transcendental).
    if has_transcendental {
        return super::risch::integrate_risch(expr, var, pool);
    }

    // Logarithmic-derivative rule: ∫ (h'/h)·log(h)^n dx (single-generator log
    // case, e.g. ∫ 1/(x·log x) dx = log(log x)).  This must precede the
    // `known_nonelementary` li pre-check below, which would otherwise mis-certify
    // ∫ 1/(x·log x) dx as the (non-elementary) logarithmic integral li — it is in
    // fact elementary because 1/x = (log x)'.  The rule fires only when the
    // coefficient equals h'/h exactly, so a match is always a correct, verifiable
    // antiderivative; genuinely non-elementary forms (1/log x, 1/((x+1)·log x))
    // do not match and fall through to the certification below.
    if let Some(result) = try_log_derivative(expr, var, pool) {
        let simplified = simplify(result, pool);
        let mut rlog = DerivationLog::new();
        rlog.push(RewriteStep::simple(
            "log_derivative_rule",
            expr,
            simplified.value,
        ));
        let final_log = rlog.merge(simplified.log);
        return Ok(DerivedExpr::with_log(simplified.value, final_log));
    }

    // Risch Gap 6: certify classic non-elementary special-function integrands
    // (Ei/Si/Ci/Shi/Chi/li) before the rule-based engine, which would otherwise
    // return the weaker `NotImplemented` verdict.
    if let Some(reason) = known_nonelementary(expr, var, pool) {
        return Err(IntegrationError::NonElementary(reason));
    }

    integrate_inner(expr, var, pool, 0)
}

/// Internal entry point that runs the full elementary pipeline — rule engine,
/// then the rational-function fallback, then the non-linear u-substitution
/// fallback — threading a recursion `depth` so u-substitution can recurse on the
/// reduced integrand without risking unbounded recursion.
///
/// `depth == 0` is the top-level call from [`integrate`]; u-substitution
/// increments it for the inner integral and only recurses while
/// `depth < U_SUBST_MAX_DEPTH`.
fn integrate_inner(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    depth: u32,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    // Cooperative budget checkpoint at the recursion boundary: u-substitution
    // re-enters `integrate_inner` (see `try_u_substitution` below), so this
    // one call site also bounds the recursive fallback chain, not just the
    // initial call. See `crate::budget` — P1 search plumbing item 4.
    crate::budget::check()?;

    let mut log = DerivationLog::new();
    match integrate_raw(expr, var, pool, &mut log) {
        Ok(raw) => {
            let simplified = simplify(raw, pool);
            let final_log = log.merge(simplified.log);
            Ok(DerivedExpr::with_log(simplified.value, final_log))
        }
        // A budget trip travels *as* a `NotImplemented` (see `IntegrationError`'s
        // carrier note), so it has to be split off ahead of the decline arm —
        // otherwise the fallbacks below read "the caller wants out" as "the rule
        // engine declined" and carry on spending the time the caller just asked
        // to stop spending.
        Err(e) if e.is_budget() => Err(e),
        Err(IntegrationError::NotImplemented(msg)) => {
            // Risch Gap 3: rational-function integration via Rothstein–Trager.
            // Tried as a fallback so simple cases keep their existing rules.
            if let Some(result) =
                super::risch::rational_integrate::try_integrate_rational(expr, var, pool)
            {
                let simplified = simplify(result, pool);
                let mut rlog = DerivationLog::new();
                rlog.push(RewriteStep::simple(
                    "rothstein_trager",
                    expr,
                    simplified.value,
                ));
                let final_log = rlog.merge(simplified.log);
                return Ok(DerivedExpr::with_log(simplified.value, final_log));
            }
            // `try_integrate_rational` returns a bare `None` both for "not a
            // rational function" and for "the budget tripped part-way" — it is
            // public API and cannot grow a `Result` without a major semver break.
            // Asking here is what turns the second into an honest `E-BUDGET-*`
            // instead of letting it fall through as a mathematical decline.
            crate::budget::check()?;
            // Non-linear substitution (derivative-divides heuristic):
            // ∫ f(g(x))·g'(x) dx = ∫ f(u) du with u = g(x).  Tried only after
            // the rules and the rational path have declined, so anything they
            // already solve is untouched.  The result is soundness-gated: it is
            // returned only when its derivative matches the integrand, so a
            // wrong antiderivative is never produced (a clean decline falls
            // through to the existing error).
            if let Some(result) = try_u_substitution(expr, var, pool, depth)? {
                let simplified = simplify(result, pool);
                let mut rlog = DerivationLog::new();
                rlog.push(RewriteStep::simple(
                    "u_substitution",
                    expr,
                    simplified.value,
                ));
                let final_log = rlog.merge(simplified.log);
                return Ok(DerivedExpr::with_log(simplified.value, final_log));
            }
            Err(IntegrationError::NotImplemented(msg))
        }
        Err(other) => Err(other),
    }
}

/// Definite integral `∫_lower^upper f dx` via the fundamental theorem of
/// calculus: `F(upper) − F(lower)` where `F = ∫ f dx`.
///
/// This is the elementary FTC wrapper: it computes an antiderivative with
/// [`integrate`], substitutes the bounds, and simplifies the difference.  It
/// handles only the case where the antiderivative exists and is finite at both
/// bounds.
///
/// It deliberately does **not** evaluate improper integrals or take the
/// residue-theorem route.  Where the integrand is a rational function, a pole on
/// `[lower, upper]` is *detected* and reported as an error rather than being
/// pushed through the FTC difference, which would yield a finite-looking but
/// wrong value (`∫_{-1}^{1} x^{-2} dx` would otherwise "evaluate" to `-2`, while
/// the integral diverges).  Non-polynomial denominators (`1/sin(x)`) are not
/// analysed and still fall through unchecked.
///
/// # Errors
///
/// Returns the same errors as [`integrate`]: [`IntegrationError::NonElementary`]
/// when no elementary antiderivative exists, or
/// [`IntegrationError::NotImplemented`] when the integrand is outside the
/// supported subset — including when a detected pole makes the integral
/// improper.
pub fn integrate_definite(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    // A pole of the integrand strictly inside (or at an endpoint of) the
    // interval makes the integral improper: `F` is discontinuous there, so the
    // FTC difference `F(b) - F(a)` is not the integral and is often a clean,
    // plausible, wrong number (`∫_{-1}^{1} x^{-2} dx` "=" `-1 - 1` = `-2`,
    // while the integral diverges). Detect that before substituting, so the
    // caller gets an error instead of a fabricated value.
    // Normalise the bounds first: a caller-supplied bound may be an unreduced
    // expression (the Python binding lifts a scalar as `var·0 + n`), and every
    // check below reasons about the bound's *value*.
    let lower = simplify(lower, pool).value;
    let upper = simplify(upper, pool).value;

    if let Some(reason) = interior_singularity(expr, var, lower, upper, pool) {
        return Err(IntegrationError::NotImplemented(reason));
    }

    // The exact check above only sees rational integrands.  `1/cos(x)^2` on
    // `[0, 2]` has a pole at `π/2` that no polynomial root isolation can find,
    // and the FTC difference `tan(2) - tan(0) = -2.185…` is a clean, plausible,
    // *negative* number for an integrand that is positive everywhere and whose
    // integral diverges.  Confirm blow-up numerically instead.
    if let Some(reason) = numeric_interior_singularity(expr, var, lower, upper, pool) {
        return Err(IntegrationError::NotImplemented(reason));
    }

    // Both checks above bind only `var`, so a free *parameter* in the integrand
    // turns each of them off — `interior_singularity` cannot build an integer
    // polynomial from a parametric denominator, and every numeric sample fails
    // with an unbound symbol. The FTC difference was then returned as though it
    // held for all parameter values, when for some of them the integral
    // diverges.
    if let Some(reason) = parametric_interior_singularity(expr, var, lower, upper, pool) {
        return Err(IntegrationError::NotImplemented(reason));
    }

    let antideriv = integrate(expr, var, pool)?;
    let f = antideriv.value;

    // The FTC needs `F` continuous on `[lower, upper]`, and none of the checks
    // above look at `F` at all — they look at the integrand. A bounded, smooth,
    // strictly positive integrand can still have an antiderivative that jumps
    // inside the interval, and then `F(b) - F(a)` is not the integral. The
    // Weierstrass substitution manufactures exactly that: every
    // `∫ dx/(a + b·cos x)` picks up a `tan(x/2)`, which jumps at `x = π`.
    if let Some(reason) = antiderivative_jump(f, expr, var, lower, upper, pool) {
        return Err(IntegrationError::NotImplemented(reason));
    }

    // F(upper) and F(lower). For a finite bound this is plain substitution; for
    // `±∞` (V2-16's canonical pos_infinity, or its negation) substitution would
    // silently treat `∞` as an ordinary free symbol and fabricate a
    // finite-looking but meaningless expression (e.g. `exp(-k·∞)`). Instead the
    // bound value is the *limit* of `F` as `var → bound`, computed via
    // [`crate::calculus::limit`]. If that limit cannot be determined, the
    // integral errors rather than returning a wrong answer.
    let f_upper = eval_bound(f, var, upper, pool)?;
    let f_lower = eval_bound(f, var, lower, pool)?;
    let neg_lower = pool.mul(vec![pool.integer(-1_i32), f_lower]);
    let diff_expr = pool.add(vec![f_upper, neg_lower]);

    let simplified = simplify(diff_expr, pool);

    // Last gate: for numeric bounds the answer is a closed numeric expression,
    // so it must denote a finite real.  `∫_{-1}^{1} x^{-1} dx` reduces to
    // `-log(-1)` and `∫_0^1 x^{-3/2} dx` to `-2 + 2·(0^{1/2})^{-1}`: both look
    // like values but denote nothing real, and both come from applying the FTC
    // where its hypotheses fail.  Refuse rather than hand back an expression
    // the evaluator itself rejects.
    if numeric_bound(lower, pool).is_some() && numeric_bound(upper, pool).is_some() {
        if let Some(reason) = non_real_closed_form(simplified.value, pool) {
            return Err(IntegrationError::NotImplemented(format!(
                "improper integral: the fundamental-theorem difference F(b) - F(a) = {} {reason}, \
                 so the antiderivative is not real and finite across [{}, {}] and the FTC does \
                 not apply (the integral diverges, or converges only as a principal value)",
                pool.display(simplified.value),
                pool.display(lower),
                pool.display(upper),
            )));
        }
    }

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "fundamental_theorem_of_calculus",
        expr,
        simplified.value,
    ));
    let final_log = antideriv.log.merge(log).merge(simplified.log);
    Ok(DerivedExpr::with_log(simplified.value, final_log))
}

/// Describe why a *closed* (symbol-free) numeric expression does not denote a
/// finite real, or `None` when it does — or when the question cannot be decided
/// (unbound symbols, functions the evaluator does not implement), in which case
/// the caller must not reject.
fn non_real_closed_form(expr: ExprId, pool: &ExprPool) -> Option<&'static str> {
    use crate::eval::UnsupportedReason;
    match crate::eval::eval_f64(expr, pool, &HashMap::new()) {
        Ok(_) => None,
        Err(e) => match e.reason {
            UnsupportedReason::NonFiniteResult => Some("is not a finite real number"),
            UnsupportedReason::ZeroToNegativePower => {
                Some("contains a division by zero (an unresolved pole)")
            }
            UnsupportedReason::UnsupportedExpression { kind: "branch_cut" } => {
                Some("leaves the real branch of a logarithm or root")
            }
            _ => None,
        },
    }
}

/// Number of grid samples used to look for a blow-up of the integrand.
const POLE_SCAN_SAMPLES: usize = 257;
/// Bisection refinements applied to a candidate blow-up.
const POLE_SCAN_REFINEMENTS: usize = 60;
/// The refined magnitude must exceed this before a pole is declared.
///
/// It cannot be much higher. `1/x` reaches only `1e16` before the nearest
/// probe runs out of `f64` resolution, so a threshold of `1e30` — the value
/// this held until 3.8 — is unreachable for every *simple* pole and the scan
/// could only ever see double poles. `∫_1^5 tan x dx` was returned as
/// `0.644` (its Cauchy principal value) for a divergent integral because of it.
const POLE_SCAN_MAGNITUDE: f64 = 1e13;
/// …and must exceed this multiple of the integrand's *typical* magnitude, so an
/// integrand that is merely large everywhere is never mistaken for a pole.
///
/// The baseline is the **median** of the coarse samples, not their maximum.
/// With the maximum, a grid point landing essentially on the pole defeats the
/// test — the "growth" has already happened before refinement starts. That is
/// not a corner case: on `[0, π]` sample 128 of 257 falls within `1e-5` of
/// `π/2`, which is exactly why `∫_0^π tan²x dx` came back as `-π`, a negative
/// number for a non-negative integrand.
const POLE_SCAN_GROWTH: f64 = 1e12;
/// Fraction of the interval width excluded at each end.  Endpoint singularities
/// are a different (and often convergent) story — `∫_0^1 log x dx = -1` is
/// perfectly well defined — and are handled by [`non_real_closed_form`] on the
/// resulting closed form, not here.
const POLE_SCAN_MARGIN: f64 = 1e-3;

/// Numerically confirm a singularity of `integrand` **strictly inside**
/// `(lower, upper)`.
///
/// This complements [`interior_singularity`], which is exact but only sees
/// rational integrands.  Here the integrand is sampled on a grid, the largest
/// magnitude is refined by repeated bracket shrinking, and a pole is reported
/// only when the magnitude both exceeds [`POLE_SCAN_MAGNITUDE`] and has grown by
/// a factor of [`POLE_SCAN_GROWTH`] over the refinement.  No function that is
/// bounded on the interval can pass that test, so a proper integral is never
/// rejected; an integrand the evaluator cannot handle numerically simply falls
/// through with `None`, exactly as before.
// The negated comparisons below (`!(width > 0.0)`, `!(lo_b < hi_b)`, …) are
// deliberate: they are NaN-safe bail-outs.  `!(width > 0.0)` is true when
// `width` is NaN and correctly abandons the scan, whereas clippy's suggested
// `width <= 0.0` is false for NaN and would let a degenerate interval through
// into the sampling loop.  Since this function's whole job is to decide whether
// an integral is safe to evaluate, failing open on NaN is exactly the bug it
// exists to prevent.
fn numeric_interior_singularity(
    integrand: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<String> {
    numeric_interior_singularity_at(integrand, var, lower, upper, &HashMap::new(), pool)
}

/// [`numeric_interior_singularity`] with the integrand's free *parameters*
/// pinned to concrete values by `params`.
///
/// The scan itself is unchanged; only the environment the integrand is
/// evaluated in gains the extra bindings. See
/// [`parametric_interior_singularity`] for why a pole found at one parameter
/// value is enough to refuse an answer returned for all of them.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn numeric_interior_singularity_at(
    integrand: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    params: &HashMap<ExprId, f64>,
    pool: &ExprPool,
) -> Option<String> {
    let (a, b) = (numeric_bound(lower, pool)?, numeric_bound(upper, pool)?);
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let width = hi - lo;
    if !(width > 0.0) || !width.is_finite() {
        return None;
    }
    let (scan_lo, scan_hi) = (lo + POLE_SCAN_MARGIN * width, hi - POLE_SCAN_MARGIN * width);
    if !(scan_lo < scan_hi) {
        return None;
    }

    // Sample through the tree-walking interpreter, not `eval::eval_f64`: the
    // latter knows only `sin`, `cos`, `exp`, `log` and `sqrt`, so every sample
    // of an integrand mentioning `tan` (or `abs`, `sinh`, `atan`, …) failed and
    // the scan reported "no opinion" for the whole family. `∫_0^2 sec²x dx` was
    // refused while `∫_0^2 tan²x dx` — the same function minus 1 — returned
    // `tan 2 - 2 = -4.19`, a negative number for a non-negative integrand whose
    // integral diverges. `eval_interp` covers the primitive vocabulary the
    // integrator itself works over.
    let at = |t: f64| -> Option<f64> {
        let mut bindings = params.clone();
        bindings.insert(var, t);
        crate::jit::eval_interp(integrand, &bindings, pool).filter(|v| v.is_finite())
    };

    // Coarse scan for the largest magnitude on the grid.
    let scan_width = scan_hi - scan_lo;
    let mut magnitudes: Vec<f64> = Vec::with_capacity(POLE_SCAN_SAMPLES);
    let mut center = f64::NAN;
    let mut m0 = 0.0f64;
    for i in 0..POLE_SCAN_SAMPLES {
        let t = scan_lo + scan_width * (i as f64 + 0.5) / POLE_SCAN_SAMPLES as f64;
        if let Some(v) = at(t) {
            magnitudes.push(v.abs());
            if v.abs() > m0 {
                m0 = v.abs();
                center = t;
            }
        }
    }
    // Nothing evaluated: the integrand is outside the numeric evaluator's
    // vocabulary, so this check has no opinion.
    if magnitudes.is_empty() || !center.is_finite() || m0 <= 0.0 {
        return None;
    }
    // Typical magnitude on the interval — the baseline the blow-up has to beat.
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let baseline = magnitudes[magnitudes.len() / 2].max(f64::MIN_POSITIVE);

    // Refine: keep shrinking a bracket around the running maximum.
    let mut half = scan_width / POLE_SCAN_SAMPLES as f64;
    let mut peak = m0;
    let mut peak_at = center;
    for _ in 0..POLE_SCAN_REFINEMENTS {
        let (mut lo_b, mut hi_b) = (peak_at - half, peak_at + half);
        lo_b = lo_b.max(scan_lo);
        hi_b = hi_b.min(scan_hi);
        if !(lo_b < hi_b) {
            break;
        }
        let step = (hi_b - lo_b) / 6.0;
        if !(step > 0.0) {
            break;
        }
        for j in 1..=5 {
            let t = lo_b + step * j as f64;
            if let Some(v) = at(t) {
                if v.abs() > peak {
                    peak = v.abs();
                    peak_at = t;
                }
            }
        }
        // Always shrink, even when this round found nothing bigger: once the
        // bracket is narrower than six times the distance to the pole none of
        // the probes can beat the incumbent, and stopping there would abandon
        // the search a few rounds before the blow-up becomes visible.
        half = step;
    }

    if peak > POLE_SCAN_MAGNITUDE && peak > POLE_SCAN_GROWTH * baseline {
        return Some(format!(
            "improper integral: the integrand blows up at {} ≈ {}, strictly inside the \
             interval of integration [{}, {}] (|integrand| exceeds {:e} there). The \
             fundamental-theorem difference F(b) - F(a) is not the value of this integral \
             — it diverges, or converges only as a principal value",
            pool.display(var),
            peak_at,
            lo,
            hi,
            peak,
        ));
    }
    None
}

/// Maximum number of free parameters a parametric pole scan will handle.
///
/// Beyond this the assignment grid is not worth its cost, and the scan simply
/// has no opinion — exactly as it does for an integrand the evaluator cannot
/// handle.
const POLE_SCAN_MAX_PARAMS: usize = 4;

/// Number of assignments tried per parameter in a parametric pole scan.
const POLE_SCAN_PARAM_VALUES: usize = 11;

/// Free symbols of `expr` other than `var`, in first-seen order.
///
/// `∞` is excluded: it is the canonical bound marker, not a parameter, and it
/// is never a value the numeric evaluator could bind.
fn free_parameters(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    fn walk(e: ExprId, var: ExprId, inf: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
        if e == var || e == inf {
            return;
        }
        match pool.get(e) {
            ExprData::Symbol { .. } => {
                if !out.contains(&e) {
                    out.push(e);
                }
            }
            ExprData::Add(xs) | ExprData::Mul(xs) => {
                for x in xs {
                    walk(x, var, inf, pool, out);
                }
            }
            ExprData::Pow { base, exp } => {
                walk(base, var, inf, pool, out);
                walk(exp, var, inf, pool, out);
            }
            ExprData::Func { args, .. } => {
                for a in args {
                    walk(a, var, inf, pool, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(expr, var, pool.pos_infinity(), pool, &mut out);
    out
}

/// Candidate values for a free parameter when scanning `[lo, hi]` for a pole.
///
/// Poles whose *location* is set by a parameter (`1/(x-a)²`, `1/(a·x-1)²`) sit
/// inside the interval only for parameter values related to the interval
/// itself, so the grid mixes points spread across `[lo, hi]` with a handful of
/// generic small magnitudes. The offsets are deliberately not round fractions:
/// a parameter landing *exactly* on a grid sample makes the integrand
/// non-finite there, which the scan discards rather than reports.
fn pole_scan_parameter_values(lo: f64, hi: f64) -> Vec<f64> {
    let width = hi - lo;
    let mut values = vec![
        lo + 0.137 * width,
        lo + 0.371 * width,
        lo + 0.613 * width,
        lo + 0.859 * width,
    ];
    values.extend_from_slice(&[-2.13, -1.07, -0.43, 0.43, 1.07, 2.13, 3.71]);
    debug_assert_eq!(values.len(), POLE_SCAN_PARAM_VALUES);
    values
}

/// Detect an interior pole that appears for *some* real value of the
/// integrand's free parameters.
///
/// [`numeric_interior_singularity`] binds only the integration variable, so an
/// integrand carrying any other free symbol evaluates to nothing at every
/// sample and the scan silently switches itself off. `∫_0^2 sec²x dx` is
/// correctly refused, while `∫_0^2 a·sec²x dx` returned `a·tan 2` — a
/// *negative* number, at `a = 1`, for the very integrand the plain scan exists
/// to catch. `∫_{-1}^{1} (x-a)^{-2} dx` is the same failure through the exact
/// route: `interior_singularity` needs integer coefficients, so a parametric
/// denominator falls through and the FTC difference is returned as if it held
/// for every `a`, including the `|a| < 1` where the integral diverges.
///
/// The result is reported for *all* parameter values, so exhibiting one real
/// value at which the integral is improper is enough to refuse it: the answer
/// carries no side condition that would exclude that value. Refusal is
/// therefore justified by the same blow-up evidence the plain scan uses — the
/// magnitude and growth thresholds mean no bounded integrand can trigger it —
/// and finding nothing simply falls through to the previous behaviour.
fn parametric_interior_singularity(
    integrand: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<String> {
    let params = free_parameters(integrand, var, pool);
    if params.is_empty() || params.len() > POLE_SCAN_MAX_PARAMS {
        return None;
    }
    let (a, b) = (numeric_bound(lower, pool)?, numeric_bound(upper, pool)?);
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let grid = pole_scan_parameter_values(lo, hi);

    // The assignment set is deliberately linear, not the full product: the
    // *diagonal* (every parameter at the same grid value, which is what finds
    // the pole of `1/(x-a-b)²`), plus each parameter swept alone with the
    // others held at a fixed non-degenerate value (which finds the pole of
    // `1/(x-a)²` however many other parameters ride along). A full grid would
    // be `11^n` scans for no extra coverage of the shapes that actually occur.
    const HELD: f64 = 1.07;
    let n = params.len();
    let mut assignments: Vec<Vec<f64>> = grid.iter().map(|v| vec![*v; n]).collect();
    if n > 1 {
        for i in 0..n {
            for v in &grid {
                let mut row = vec![HELD; n];
                row[i] = *v;
                assignments.push(row);
            }
        }
    }

    for assignment in assignments {
        let bindings: HashMap<ExprId, f64> = params
            .iter()
            .copied()
            .zip(assignment.iter().copied())
            .collect();
        if let Some(reason) =
            numeric_interior_singularity_at(integrand, var, lower, upper, &bindings, pool)
        {
            let at = params
                .iter()
                .zip(assignment.iter())
                .map(|(p, v)| format!("{} = {}", pool.display(*p), v))
                .collect::<Vec<_>>()
                .join(", ");
            return Some(format!(
                "{reason}. This holds at {at}; the answer would be returned for every value \
                 of {}, so it is refused rather than stated without the side condition that \
                 keeps the pole outside [{}, {}]",
                params
                    .iter()
                    .map(|p| pool.display(*p).to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                lo,
                hi,
            ));
        }
    }
    None
}

/// Cells the antiderivative is sampled over when looking for a jump.
const JUMP_SCAN_CELLS: usize = 257;
/// Sub-samples per cell used to estimate `sup |f|` on that cell.
const JUMP_SCAN_SUBSAMPLES: usize = 9;
/// A cell is *suspicious* once `|ΔF|` exceeds this multiple of `h·sup|f|`.
///
/// The mean value theorem gives `|ΔF| = h·|f(ξ)| ≤ h·sup|f|` wherever `F` is
/// differentiable, so the true value is `≤ 1`; the margin absorbs `sup|f|`
/// being *sampled* rather than computed.
const JUMP_SUSPICION_RATIO: f64 = 8.0;
/// Bisections applied to the most suspicious cell.
const JUMP_REFINEMENTS: usize = 50;
/// A jump is declared only once the ratio has grown past this.
///
/// This is what separates a genuine discontinuity from a narrow spike. Around
/// a jump, `|ΔF|` tends to the jump height while `h·sup|f|` tends to zero, so
/// the ratio grows like `1/h`. Around a spike — however tall — `F` is still
/// continuous, `|ΔF|` shrinks with the cell, and the ratio stays bounded.
const JUMP_CONFIRM_RATIO: f64 = 1e6;

/// Detect a jump discontinuity of the antiderivative `f` strictly inside
/// `(lower, upper)`, which makes `F(b) − F(a)` not the value of the integral.
///
/// This is the failure the other two guards structurally cannot see: they look
/// at the *integrand*, and here the integrand is perfectly well behaved.
/// `∫_0^{3.2} dx/(cos x − 3)` — integrand between `1/16` and `1/4`, so the
/// integral is between `0.2` and `0.8` — returned `−0.413`, because the
/// half-angle antiderivative carries a `tan(x/2)` that jumps at `x = π`. Over a
/// full period the same mechanism returns `0`: `∫_0^{2π} dx/(2 + cos x)` came
/// back as `−8e-17` where the value is `2π/√3 ≈ 3.63`.
///
/// Returns `None` — no opinion — whenever the question cannot be decided:
/// symbolic bounds, free parameters, or an `F` the interpreter cannot
/// evaluate. Refusal requires positive evidence, never absence of it.
// `!(width > 0.0)` and `!(bound > 0.0)` are NaN-safe bail-outs, exactly as in
// `numeric_interior_singularity`: they are *true* for NaN and correctly abandon
// the scan, whereas clippy's suggested `width <= 0.0` is false for NaN and would
// let a degenerate cell through into the ratio test.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn antiderivative_jump(
    f: ExprId,
    integrand: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<String> {
    if !free_parameters(f, var, pool).is_empty()
        || !free_parameters(integrand, var, pool).is_empty()
    {
        return None;
    }
    let (a, b) = (numeric_bound(lower, pool)?, numeric_bound(upper, pool)?);
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let width = hi - lo;
    if !(width > 0.0) || !width.is_finite() {
        return None;
    }

    let at = |e: ExprId, t: f64| -> Option<f64> {
        let mut bindings = HashMap::new();
        bindings.insert(var, t);
        crate::jit::eval_interp(e, &bindings, pool).filter(|v| v.is_finite())
    };
    // `sup |integrand|` over `[c0, c1]`, sampled. `None` when nothing on the
    // cell evaluates, which makes the cell undecidable rather than suspicious.
    let sup_f = |c0: f64, c1: f64| -> Option<f64> {
        let mut m: Option<f64> = None;
        for j in 0..JUMP_SCAN_SUBSAMPLES {
            let t = c0 + (c1 - c0) * (j as f64) / ((JUMP_SCAN_SUBSAMPLES - 1) as f64);
            if let Some(v) = at(integrand, t) {
                m = Some(m.map_or(v.abs(), |cur: f64| cur.max(v.abs())));
            }
        }
        m
    };
    // `|ΔF| / (h · sup|f|)` on `[c0, c1]`, together with `|ΔF|`.
    let ratio = |c0: f64, c1: f64| -> Option<(f64, f64)> {
        let (f0, f1) = (at(f, c0)?, at(f, c1)?);
        let jump = (f1 - f0).abs();
        let bound = (c1 - c0).abs() * sup_f(c0, c1)?;
        if !(bound > 0.0) || !bound.is_finite() || !jump.is_finite() {
            return None;
        }
        Some((jump / bound, jump))
    };

    // Coarse pass: find the most suspicious cell.
    let mut worst = 0.0_f64;
    let mut cell = (f64::NAN, f64::NAN);
    for i in 0..JUMP_SCAN_CELLS {
        let c0 = lo + width * (i as f64) / (JUMP_SCAN_CELLS as f64);
        let c1 = lo + width * ((i + 1) as f64) / (JUMP_SCAN_CELLS as f64);
        if let Some((r, _)) = ratio(c0, c1) {
            if r > worst {
                worst = r;
                cell = (c0, c1);
            }
        }
    }
    if worst < JUMP_SUSPICION_RATIO || !cell.0.is_finite() {
        return None;
    }

    // Refine: keep the half carrying the larger `|ΔF|`. A jump keeps its
    // height while the cell shrinks; a spike does not.
    let (mut c0, mut c1) = cell;
    let mut best = worst;
    for _ in 0..JUMP_REFINEMENTS {
        let mid = 0.5 * (c0 + c1);
        if !(c0 < mid && mid < c1) {
            break;
        }
        let left = ratio(c0, mid);
        let right = ratio(mid, c1);
        let take_left = match (&left, &right) {
            (Some((_, jl)), Some((_, jr))) => jl >= jr,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let (nc0, nc1) = if take_left { (c0, mid) } else { (mid, c1) };
        let Some((r, _)) = ratio(nc0, nc1) else { break };
        c0 = nc0;
        c1 = nc1;
        best = best.max(r);
    }

    (best > JUMP_CONFIRM_RATIO).then(|| {
        format!(
            "improper application of the fundamental theorem: the antiderivative {} is \
             discontinuous at {} ≈ {}, strictly inside [{}, {}] (its increment there exceeds \
             the integrand's own bound by a factor of {:.3e}, and grows as the bracket \
             shrinks). F(b) - F(a) therefore skips the jump and is not the value of this \
             integral",
            pool.display(f),
            pool.display(var),
            0.5 * (c0 + c1),
            lo,
            hi,
            best,
        )
    })
}

/// Split `expr` into `(numerator, denominator)` by collecting factors carrying a
/// negative integer power into the denominator.
fn split_numer_denom(expr: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let factors = match pool.get(expr) {
        ExprData::Mul(xs) => xs,
        _ => vec![expr],
    };
    let mut nums = Vec::new();
    let mut dens = Vec::new();
    for factor in factors {
        if let ExprData::Pow { base, exp } = pool.get(factor) {
            if let ExprData::Integer(n) = pool.get(exp) {
                if n.0 < 0 {
                    let positive = simplify(pool.mul(vec![pool.integer(-1_i32), exp]), pool).value;
                    dens.push(pool.pow(base, positive));
                    continue;
                }
            }
        }
        nums.push(factor);
    }
    let numer = if nums.is_empty() {
        pool.integer(1_i32)
    } else {
        simplify(pool.mul(nums), pool).value
    };
    let denom = if dens.is_empty() {
        pool.integer(1_i32)
    } else {
        simplify(pool.mul(dens), pool).value
    };
    (numer, denom)
}

/// Interpret `bound` as a concrete `f64`, if it is a numeric constant.
///
/// The bound is simplified first.  Callers do not always hand in a bare
/// literal: the Python binding lifts a plain `int`/`float` into the pool as
/// `var·0 + n`, which still *mentions* the integration variable.  Evaluating
/// that unsimplified fails with an unbound symbol, and every singularity check
/// keyed off this function would then silently switch itself off for every
/// Python caller.
fn numeric_bound(bound: ExprId, pool: &ExprPool) -> Option<f64> {
    let bound = simplify(bound, pool).value;
    let value = crate::eval::eval_f64(bound, pool, &std::collections::HashMap::new()).ok()?;
    value.is_finite().then_some(value)
}

/// Detect a singularity of `integrand` on the closed interval `[lower, upper]`.
///
/// Returns a human-readable description when a pole is found, or `None` when
/// there is none *or* when the question cannot be decided — the check must never
/// reject an integral that is actually proper, so every uncertain case falls
/// through to the previous behaviour.
///
/// Scope: rational integrands, whose poles are the real roots of the reduced
/// denominator. Common factors shared with the numerator are divided out first,
/// so removable singularities (`(x²-1)/(x-1)` at `x = 1`) are correctly *not*
/// reported. Non-polynomial denominators (`1/sin(x)`, `1/log(x)`) are not
/// analysed and still fall through.
fn interior_singularity(
    integrand: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<String> {
    // Symbolic bounds cannot be compared against root locations.
    let (a, b) = (numeric_bound(lower, pool)?, numeric_bound(upper, pool)?);
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };

    let simplified = simplify(integrand, pool).value;
    let (numer, denom) = split_numer_denom(simplified, pool);

    // A denominator free of the integration variable has no poles in `var`.
    let denom_poly = crate::poly::UniPoly::from_symbolic(denom, var, pool).ok()?;
    if denom_poly.degree() == 0 {
        return None;
    }

    // Divide out factors shared with the numerator: those singularities are
    // removable and must not be reported.
    // The pseudo-quotient has the same roots as the exact quotient (they differ
    // only by a constant factor), which is all this check needs.
    let reduced = crate::poly::UniPoly::from_symbolic(numer, var, pool)
        .ok()
        .and_then(|numer_poly| denom_poly.gcd(&numer_poly))
        .filter(|common| common.degree() > 0)
        .and_then(|common| denom_poly.pseudo_divrem(&common))
        .filter(|(_, remainder)| remainder.is_zero())
        .map(|(quotient, _)| quotient)
        .unwrap_or_else(|| denom_poly.clone());
    if reduced.degree() == 0 {
        return None;
    }

    for interval in crate::poly::real_roots(&reduced).ok()? {
        // `real_roots` isolates one root per interval. Report only when the
        // whole bracket lies inside `[lo, hi]`, so the root is *certainly*
        // inside; a partially overlapping bracket is ambiguous and is skipped
        // rather than risking a false rejection.
        if interval.lo_f64() >= lo && interval.hi_f64() <= hi {
            let location = if interval.lo_f64() == interval.hi_f64() {
                format!("{}", interval.lo_f64())
            } else {
                format!("in [{}, {}]", interval.lo_f64(), interval.hi_f64())
            };
            return Some(format!(
                "improper integral: the integrand has a pole at {} = {}, inside the \
                 interval of integration [{}, {}]. The integral does not converge \
                 (or converges only as a principal value), so the \
                 fundamental-theorem difference F(b) - F(a) is not its value",
                pool.display(var),
                location,
                lo,
                hi,
            ));
        }
    }
    None
}

/// Evaluate the antiderivative `f` at `bound` for the FTC difference.
///
/// For a finite `bound`, this is plain substitution. For `bound == +∞` (or
/// `-∞`, represented as `(-1)·(+∞)` per [`ExprPool::pos_infinity`]'s
/// documented convention), the value is `lim_{var→bound} f`, computed via
/// [`crate::calculus::limit`]. A limit that cannot be determined (or one that
/// is itself non-finite, i.e. the integral diverges) is reported as
/// [`IntegrationError::NotImplemented`] — never silently substituted as if `∞`
/// were an ordinary symbol.
fn eval_bound(
    f: ExprId,
    var: ExprId,
    bound: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, IntegrationError> {
    if is_infinite_bound(bound, pool) {
        let lim = crate::calculus::limit(
            f,
            var,
            bound,
            crate::calculus::LimitDirection::Bidirectional,
            pool,
        )
        .map_err(|e| {
            IntegrationError::NotImplemented(format!(
                "improper integral with an infinite bound: lim_{{{}→{}}} {} : {e}",
                pool.display(var),
                pool.display(bound),
                pool.display(f),
            ))
        })?;
        // The antiderivative diverges at this bound (the limit is itself `±∞`,
        // or — for forms `limit` cannot fully reduce — contains a residual
        // `0^{negative}` pole artifact). Either way the *definite* integral is
        // divergent or beyond what can be certified here: error rather than
        // feeding `∞`/an unresolved pole into the FTC subtraction, which would
        // simplify into a finite-looking (but meaningless) value.
        if expr_is_non_finite(lim, pool) {
            return Err(IntegrationError::NotImplemented(format!(
                "improper integral with an infinite bound: lim_{{{}→{}}} {} = {} is not finite (the improper integral may diverge)",
                pool.display(var),
                pool.display(bound),
                pool.display(f),
                pool.display(lim),
            )));
        }
        return Ok(lim);
    }
    Ok(subs_var(f, var, bound, pool))
}

/// True when `expr` is (or contains) `±∞` (the canonical [`ExprPool::pos_infinity`]
/// symbol) or an unresolved `0^{negative integer}` pole artifact — i.e. is not a
/// finite value, so it must not be used as an endpoint in the FTC subtraction.
fn expr_is_non_finite(expr: ExprId, pool: &ExprPool) -> bool {
    if expr == pool.pos_infinity() {
        return true;
    }
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            if let ExprData::Integer(n) = pool.get(exp) {
                if n.0 < 0 {
                    if let ExprData::Integer(b) = pool.get(base) {
                        if b.0 == 0 {
                            return true;
                        }
                    }
                }
            }
            expr_is_non_finite(base, pool) || expr_is_non_finite(exp, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|x| expr_is_non_finite(*x, pool)),
        ExprData::Func { args, .. } => args.iter().any(|a| expr_is_non_finite(*a, pool)),
        _ => false,
    }
}

/// True when `bound` is `+∞` (canonical [`ExprPool::pos_infinity`] symbol) or
/// `-∞` (`(-1)·(+∞)`, the documented convention for limits at minus infinity).
fn is_infinite_bound(bound: ExprId, pool: &ExprPool) -> bool {
    let pos_inf = pool.pos_infinity();
    if bound == pos_inf {
        return true;
    }
    if let ExprData::Mul(args) = pool.get(bound) {
        if args.len() == 2 {
            let m_one = pool.integer(-1_i32);
            return (args[0] == m_one && args[1] == pos_inf)
                || (args[1] == m_one && args[0] == pos_inf);
        }
    }
    false
}

/// Substitute `value` for `var` everywhere in `expr`.
fn subs_var(expr: ExprId, var: ExprId, value: ExprId, pool: &ExprPool) -> ExprId {
    let mut map = HashMap::new();
    map.insert(var, value);
    crate::kernel::subs(expr, &map, pool)
}

// ---------------------------------------------------------------------------
// Non-linear integration by substitution (u-substitution / derivative-divides)
// ---------------------------------------------------------------------------

/// Maximum recursion depth for nested u-substitutions.  The reduced integrand is
/// structurally simpler at each step, but the cap is the hard guarantee against
/// pathological inputs.
const U_SUBST_MAX_DEPTH: u32 = 3;

/// Maximum number of candidate inner functions `g` tried per call, so degenerate
/// inputs cannot cause combinatorial blow-up.
const U_SUBST_MAX_CANDIDATES: usize = 12;

/// Recognise `∫ f(g(x))·g'(x) dx` and solve it by `u = g(x)` (the
/// derivative-divides heuristic).
///
/// For each non-trivial inner function `g` (arguments of `Func` nodes, bases of
/// `Pow` nodes, and non-constant factors of a top-level `Mul`), divide the
/// integrand by `g'(x)`.  If the quotient depends on `x` only through `g`, the
/// integral reduces to `∫ (quotient with g↦u) du`, which is integrated
/// recursively and back-substituted (`u ↦ g`).
///
/// Every candidate result is **soundness-gated**: it is returned only when its
/// derivative equals the original integrand (structurally, or to ~1e-7 over
/// several real sample points).  A failing candidate is skipped; if none passes,
/// the function declines with `Ok(None)` and the caller reports its existing
/// error.
///
/// # Why this returns a `Result` and not just an `Option`
///
/// A failing candidate is skipped — but a *budget trip* is not a failing
/// candidate, it is the caller asking the whole call to stop. This loop used to
/// throw both away identically (`let Ok(inner) = … else { continue }`), which
/// silently defeated every cooperative checkpoint below the top level: with
/// `max_steps=2` — enough to clear the two depth-0 checks — a `request_cancel()`
/// or an exhausted wall clock was discarded and the search moved on to the next
/// of up to 12 candidates, each of which could take seconds. `integrate` was
/// therefore only interruptible in its first instants, whatever the binding did
/// about the GIL. A budget error now propagates; everything else still skips.
fn try_u_substitution(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    depth: u32,
) -> Result<Option<ExprId>, IntegrationError> {
    if depth >= U_SUBST_MAX_DEPTH {
        return Ok(None);
    }

    // Try the integrand as written, and a trig-expanded form (tan → sin·cos⁻¹,
    // etc.) so that `∫ tan x dx` exposes the inner function `g = cos x`.  The
    // soundness gate always checks against the original `expr`.
    let mut variants = vec![expr];
    let expanded = trig_expand(expr, pool);
    if expanded != expr {
        variants.push(expanded);
    }

    // `(g, reduced integrand)` pairs already attempted. The two variants
    // (`expr` and its trig-expanded form) very often reduce to the *same* inner
    // integral under the same `g` — `∫ cos x·sin¹²x/(sin¹⁷x + sin x + 1) dx`
    // reaches `∫ u¹²/(u¹⁷ + u + 1) du` from both — and since `u` is
    // hash-consed, that is literally the same `ExprId`, integrated twice for the
    // same verdict (measured: 5.6 s + 5.5 s of an 11.1 s call). The pair is the
    // key rather than the integrand alone because a different `g` back-
    // substitutes to a different candidate.
    let mut attempted: std::collections::HashSet<(ExprId, ExprId)> =
        std::collections::HashSet::new();

    for &form in &variants {
        let candidates = collect_usub_candidates(form, var, pool);

        for g in candidates.into_iter().take(U_SUBST_MAX_CANDIDATES) {
            // Cooperative checkpoint at the granularity that actually costs
            // something: each surviving candidate runs a full recursive
            // `integrate`, which can take seconds. Checking only at the
            // recursion boundary below is too late — once a budget has tripped,
            // `simplify` stops rewriting, so the candidate's quotient no longer
            // reduces, `is_free_of` rejects it, and it `continue`s without ever
            // reaching that boundary. The search would then run out of
            // candidates and report a *decline* for what is really a
            // cancellation.
            crate::budget::check()?;

            // g must contain var, must not be var itself, and must not be constant.
            if g == var || is_free_of(g, var, pool) {
                continue;
            }

            // g'(x)
            let Ok(dg_raw) = crate::diff::diff(g, var, pool) else {
                continue;
            };
            let dg = simplify(dg_raw.value, pool).value;
            if is_zero(dg, pool) {
                continue;
            }

            // quotient = form / g'.  Distribute the reciprocal over the factors
            // of `dg` (so `x · (2·x)⁻¹` becomes `x · 2⁻¹ · x⁻¹`, which the
            // simplifier cancels to `1/2`; a bare `(2·x)⁻¹` Pow node is not
            // cancelled factor-by-factor).
            let inv = reciprocal(dg, pool);
            let quotient = simplify(pool.mul(vec![form, inv]), pool).value;

            // Replace g with a fresh symbol u and check the quotient depends on
            // x only through g.
            let u = pool.symbol("__usub_u", crate::kernel::Domain::Real);
            let mut fwd = HashMap::new();
            fwd.insert(g, u);
            let replaced = crate::kernel::subs(quotient, &fwd, pool);
            if !is_free_of(replaced, var, pool) {
                continue;
            }
            if !attempted.insert((g, replaced)) {
                continue; // identical reduced integral, identical verdict
            }

            // Integrate the reduced integrand in u (full pipeline, deeper level).
            // As in the Weierstrass route, this candidate ends at the
            // `verify_antiderivative` gate below, which provably cannot accept a
            // `RootSum` — so suppress the Lazard–Rioboo–Trager number-field GCD
            // that would build one rather than paying for an answer that is
            // certain to be rejected.
            let inner = {
                let _no_root_sum = super::risch::rational_integrate::RootSumSuppressed::enter();
                match integrate_inner(replaced, u, pool, depth + 1) {
                    Ok(inner) => inner,
                    // Not this candidate declining — the caller wants out.
                    Err(e) if e.is_budget() => return Err(e),
                    Err(_) => continue,
                }
            };

            // Back-substitute u ↦ g.
            let mut back = HashMap::new();
            back.insert(u, g);
            let result = simplify(crate::kernel::subs(inner.value, &back, pool), pool).value;

            // Soundness gate: d/dx(result) must equal the original integrand.
            if verify_antiderivative(result, expr, var, pool) {
                return Ok(Some(result));
            }
        }
    }

    Ok(None)
}

/// Rewrite trigonometric functions in terms of `sin`/`cos` (e.g. `tan → sin·cos⁻¹`)
/// using the simplifier's `trig_rules` ruleset, so the derivative-divides search
/// can find inner functions such as `g = cos x` for `∫ tan x dx`.  Returns the
/// rewritten expression (equal to the input when no rule fires).
fn trig_expand(expr: ExprId, pool: &ExprPool) -> ExprId {
    use crate::simplify::engine::{simplify_with, SimplifyConfig};
    use crate::simplify::rulesets::trig_rules;
    let rules = trig_rules();
    simplify_with(expr, pool, &rules, SimplifyConfig::default()).value
}

/// Build `1/expr`, distributing the reciprocal over the factors of a `Mul` and
/// over an existing `Pow` exponent.  This produces a form the simplifier can
/// cancel against the numerator (a bare `Pow{Mul[..], -1}` node is not cancelled
/// factor-by-factor by the rule simplifier).
fn reciprocal(expr: ExprId, pool: &ExprPool) -> ExprId {
    let neg_one = pool.integer(-1_i32);
    match pool.get(expr) {
        ExprData::Mul(args) => {
            let inv_args: Vec<ExprId> = args.iter().map(|&a| reciprocal(a, pool)).collect();
            pool.mul(inv_args)
        }
        ExprData::Pow { base, exp } => {
            let neg_exp = pool.mul(vec![neg_one, exp]);
            pool.pow(base, neg_exp)
        }
        _ => pool.pow(expr, neg_one),
    }
}

/// Collect candidate inner functions `g` for u-substitution, in priority order
/// (larger / more composite candidates first).
fn collect_usub_candidates(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out: Vec<ExprId> = Vec::new();
    let mut seen: std::collections::HashSet<ExprId> = std::collections::HashSet::new();

    // Top-level Mul factors (lower priority, appended after structural ones).
    let mut factor_candidates: Vec<ExprId> = Vec::new();
    if let ExprData::Mul(args) = pool.get(expr) {
        for &a in &args {
            if a != var && !is_free_of(a, var, pool) && seen.insert(a) {
                factor_candidates.push(a);
            }
        }
    }

    collect_usub_inner(expr, var, pool, &mut out, &mut seen);

    // Larger candidates (more nodes) first so we prefer the most composite inner
    // function (e.g. x²+1 over x²).
    out.sort_by_key(|&c| std::cmp::Reverse(node_count(c, pool)));
    out.extend(factor_candidates);
    out
}

/// Recursively gather `Func` arguments and `Pow` bases that contain `var`.
fn collect_usub_inner(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    out: &mut Vec<ExprId>,
    seen: &mut std::collections::HashSet<ExprId>,
) {
    match pool.get(expr) {
        ExprData::Func { args, .. } => {
            for a in args {
                if a != var && !is_free_of(a, var, pool) && seen.insert(a) {
                    out.push(a);
                }
                collect_usub_inner(a, var, pool, out, seen);
            }
        }
        ExprData::Pow { base, exp } => {
            if base != var && !is_free_of(base, var, pool) && seen.insert(base) {
                out.push(base);
            }
            collect_usub_inner(base, var, pool, out, seen);
            collect_usub_inner(exp, var, pool, out, seen);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect_usub_inner(a, var, pool, out, seen);
            }
        }
        _ => {}
    }
}

/// Number of nodes in `expr` (a cheap structural-size proxy), used to order
/// candidates largest-first.
fn node_count(expr: ExprId, pool: &ExprPool) -> usize {
    1 + pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().map(|&a| node_count(a, pool)).sum::<usize>()
        }
        ExprData::Pow { base, exp } => node_count(*base, pool) + node_count(*exp, pool),
        _ => 0,
    })
}

/// `true` if `expr` is the integer `0`.
fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    as_integer(expr, pool) == Some(0)
}

/// Verify exactly that `d/dx(candidate) == integrand` after symbolic
/// simplification.
///
/// This is an in-kernel symbolic check. It does not use numeric sampling and
/// therefore returns `false` when equality cannot be established structurally.
pub fn verify_antiderivative_exact(
    candidate: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> bool {
    let Ok(d_raw) = crate::diff::diff(candidate, var, pool) else {
        return false;
    };
    let d = simplify(d_raw.value, pool).value;
    let neg = pool.mul(vec![pool.integer(-1_i32), integrand]);
    is_zero(simplify(pool.add(vec![d, neg]), pool).value, pool)
}

/// Evidence established by the antiderivative soundness gate.
///
/// Numeric sampling is a useful acceptance screen, but is deliberately distinct
/// from an in-kernel symbolic derivative identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiderivativeVerification {
    /// The symbolic residual `d/dx(candidate) - integrand` simplified to zero.
    Exact,
    /// Several finite floating-point samples agreed, but no exact identity was found.
    Numeric,
}

/// Soundness gate: verify `d/dx(candidate) == integrand`.
///
/// Accepts when `d/dx(candidate) − integrand` simplifies structurally to zero,
/// **or** when a numeric check agrees to ~1e-7 over several real sample points
/// (skipping points where either side is non-finite, e.g. singularities).  A
/// `candidate` whose derivative cannot be confirmed equal is rejected, so the
/// integrator never returns a wrong antiderivative.
pub fn verify_antiderivative_status(
    candidate: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<AntiderivativeVerification> {
    if verify_antiderivative_exact(candidate, integrand, var, pool) {
        return Some(AntiderivativeVerification::Exact);
    }

    // Numeric check at several sample points (irrational, to dodge poles).
    let Ok(d_raw) = crate::diff::diff(candidate, var, pool) else {
        return None;
    };
    let d = simplify(d_raw.value, pool).value;
    let samples = [0.3719_f64, 0.9137, 1.4231, 2.1719, 2.8123, 3.6411];
    let mut checked = 0_usize;
    for &xv in &samples {
        let mut env = HashMap::new();
        env.insert(var, xv);
        let (Some(dv), Some(fv)) = (
            crate::jit::eval_interp(d, &env, pool),
            crate::jit::eval_interp(integrand, &env, pool),
        ) else {
            // Unevaluable expression — cannot certify numerically.
            return None;
        };
        if !dv.is_finite() || !fv.is_finite() {
            continue; // near a singularity; skip this sample
        }
        let tol = 1e-7 * (1.0 + dv.abs().max(fv.abs()));
        if (dv - fv).abs() > tol {
            return None;
        }
        checked += 1;
    }

    // Require at least a couple of usable samples so an all-singular set cannot
    // vacuously pass.
    (checked >= 2).then_some(AntiderivativeVerification::Numeric)
}

fn verify_antiderivative(
    candidate: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> bool {
    verify_antiderivative_status(candidate, integrand, var, pool).is_some()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::diff;
    use crate::kernel::{Domain, ExprPool};
    use crate::poly::UniPoly;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    /// `∫ cos(x)·sinⁿ(x)/(sin^d(x) + sin x + 1) dx` — declined by every rule, so
    /// it goes to the two searches that cost real time: the Weierstrass
    /// half-angle route and, failing that, derivative-divides u-substitution.
    fn hard_trig_integrand(pool: &ExprPool, x: ExprId, n: i32, d: i32) -> ExprId {
        let s = pool.func("sin", vec![x]);
        let c = pool.func("cos", vec![x]);
        let den = pool.add(vec![pool.pow(s, pool.integer(d)), s, pool.integer(1_i32)]);
        pool.mul(vec![
            c,
            pool.pow(s, pool.integer(n)),
            pool.pow(den, pool.integer(-1_i32)),
        ])
    }

    /// A budget trip inside the u-substitution search must reach the caller.
    ///
    /// `try_u_substitution` used to discard every error from its recursive
    /// `integrate_inner` call — budget trips included — and move on to the next
    /// of up to twelve candidates, so the checkpoint at the recursion boundary
    /// did nothing.
    ///
    /// Called directly rather than through `integrate`, because which route
    /// `integrate` picks for a given integrand is not this test's business: the
    /// claim is about the search, so the search is what gets called.
    #[test]
    fn a_budget_trip_inside_u_substitution_propagates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let e = hard_trig_integrand(&pool, x, 6, 3);

        let _guard = crate::budget::enter(crate::budget::Budget::new().with_max_steps(0));
        let err = try_u_substitution(e, x, &pool, 0).expect_err("the budget must stop the search");
        assert!(err.is_budget(), "expected a budget trip, got {err:?}");
        assert_eq!(err.budget_code(), Some("E-BUDGET-002"));
    }

    /// Same claim for the Weierstrass half-angle route, which is where a hard
    /// rational-trig integrand actually spends its seconds: it runs a whole
    /// nested `integrate` on a doubled-degree rational function, and used to
    /// throw that call's budget verdict away with `.ok()?`.
    #[test]
    fn a_budget_trip_inside_the_weierstrass_route_propagates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let e = hard_trig_integrand(&pool, x, 6, 3);
        let mut log = DerivationLog::new();

        let _guard = crate::budget::enter(crate::budget::Budget::new().with_max_steps(0));
        let err = try_weierstrass_rational_trig(e, x, &pool, &mut log)
            .expect_err("the budget must stop the route");
        assert!(err.is_budget(), "expected a budget trip, got {err:?}");
    }

    /// End to end: a wall budget on the integrand that used to overshoot it by
    /// more than 10× must come back as a budget trip, not as a mathematical
    /// decline.
    ///
    /// The failure this pins is subtle and was live until the checkpoints went
    /// in: every route that gave up part-way reported `NotImplemented`, and
    /// because `NotImplemented` is *also* the budget carrier, a trip could be
    /// consumed by the next fallback and the caller would be told the integral
    /// is unsupported when in fact it was never finished. No wall-clock
    /// assertion here — only which verdict comes back.
    ///
    /// `(n, d)` was raised from `(12, 9)` to `(40, 31)` for 3.8: suppressing the
    /// `RootSum` the two verify-gated routes cannot use took `(12, 9)` from
    /// 3.7 s to 12 ms, which is inside the 50 ms budget, so the trip this test
    /// asserts stopped happening for the good reason. `(40, 31)` still costs
    /// about 5 s unbudgeted.
    #[test]
    fn a_wall_budget_stops_the_weierstrass_route_honestly() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let e = hard_trig_integrand(&pool, x, 40, 31);

        let _guard = crate::budget::enter(
            crate::budget::Budget::new().with_wall(std::time::Duration::from_millis(50)),
        );
        let err = integrate(e, x, &pool).expect_err("the budget must stop this call");
        assert!(
            err.is_budget(),
            "a wall-clock trip must be reported as one, not as a decline; got {err:?}"
        );
        assert_eq!(err.budget_code(), Some("E-BUDGET-001"));
    }

    /// The control: the propagation must not turn a *declining* candidate into
    /// an error. With no budget active the search still runs to its own verdict.
    #[test]
    fn u_substitution_still_declines_without_erroring() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let inner = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        // ∫ 2x·cos(x²) dx = sin(x²): u-substitution's bread and butter, and the
        // path where earlier candidates decline before the right one is found.
        let e = pool.mul(vec![two_x, pool.func("cos", vec![inner])]);
        let got = integrate(e, x, &pool).expect("u-substitution must still solve this");
        let expected = pool.func("sin", vec![inner]);
        assert_eq!(
            simplify(got.value, &pool).value,
            simplify(expected, &pool).value
        );
    }

    #[test]
    fn antiderivative_verification_distinguishes_numeric_evidence() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let candidate = pool.func("sqrt", vec![pool.pow(x, pool.integer(2_i32))]);
        let one = pool.integer(1_i32);

        assert_eq!(
            verify_antiderivative_status(candidate, one, x, &pool),
            Some(AntiderivativeVerification::Numeric)
        );
    }

    fn coeffs_equal(a: ExprId, b: ExprId, x: ExprId, pool: &ExprPool) -> bool {
        let ap = UniPoly::from_symbolic(a, x, pool);
        let bp = UniPoly::from_symbolic(b, x, pool);
        match (ap, bp) {
            (Ok(a), Ok(b)) => a.coefficients_i64() == b.coefficients_i64(),
            _ => a == b,
        }
    }

    // Verify the antiderivative: diff(∫f) should equal f (mod simplification).
    fn verify(expr: ExprId, x: ExprId, pool: &ExprPool) {
        let integral = integrate(expr, x, pool).unwrap();
        let deriv = diff(integral.value, x, pool).unwrap();
        assert!(
            coeffs_equal(deriv.value, expr, x, pool),
            "diff(integrate(f)) ≠ f for f = {}",
            pool.display(expr)
        );
    }

    #[test]
    fn integrate_constant() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // ∫ 5 dx = 5x
        let r = integrate(pool.integer(5_i32), x, &pool).unwrap();
        let expected = pool.mul(vec![pool.integer(5_i32), x]);
        assert!(coeffs_equal(r.value, expected, x, &pool));
    }

    #[test]
    fn integrate_x() {
        // ∫ x dx = x²/2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify(x, x, &pool);
    }

    #[test]
    fn integrate_x_squared() {
        // ∫ x² dx = x³/3
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        verify(x2, x, &pool);
    }

    #[test]
    fn integrate_polynomial() {
        // ∫ (x² + 2x) dx = x³/3 + x²
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(2_i32), x]),
        ]);
        let r = integrate(expr, x, &pool).unwrap();
        // Verify by differentiation
        let d = diff(r.value, x, &pool).unwrap();
        assert!(
            coeffs_equal(d.value, expr, x, &pool),
            "diff(∫(x²+2x)) ≠ x²+2x; got {}",
            pool.display(d.value)
        );
    }

    #[test]
    fn integrate_one_over_x() {
        // ∫ x^(-1) dx = log(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x_inv = pool.pow(x, pool.integer(-1_i32));
        let r = integrate(x_inv, x, &pool).unwrap();
        assert_eq!(r.value, pool.func("log", vec![x]));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "log_rule"));
    }

    #[test]
    fn integrate_sin() {
        // ∫ sin(x) dx = -cos(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let r = integrate(sin_x, x, &pool).unwrap();
        let neg_one = pool.integer(-1_i32);
        let expected = pool.mul(vec![neg_one, pool.func("cos", vec![x])]);
        assert_eq!(r.value, expected);
        assert!(r.log.steps().iter().any(|s| s.rule_name == "int_sin"));
    }

    #[test]
    fn integrate_cos() {
        // ∫ cos(x) dx = sin(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = integrate(pool.func("cos", vec![x]), x, &pool).unwrap();
        assert_eq!(r.value, pool.func("sin", vec![x]));
    }

    #[test]
    fn integrate_exp() {
        // ∫ exp(x) dx = exp(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = integrate(pool.func("exp", vec![x]), x, &pool).unwrap();
        assert_eq!(r.value, pool.func("exp", vec![x]));
    }

    #[test]
    fn integrate_constant_multiple() {
        // ∫ 3*x² dx = 3 * x³/3 = x³
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.integer(3_i32), pool.pow(x, pool.integer(2_i32))]);
        verify(expr, x, &pool);
    }

    #[test]
    fn integrate_not_implemented() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // ∫ sin(x²) dx has no elementary antiderivative and is outside the supported subset
        let x2 = pool.pow(x, pool.integer(2_i32));
        let err = integrate(pool.func("sin", vec![x2]), x, &pool);
        assert!(matches!(err, Err(IntegrationError::NotImplemented(_))));
    }

    // --- New rules (v0.5 Risch extension) ---

    #[test]
    fn integrate_log_x() {
        // ∫ log(x) dx = x*log(x) - x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let r = integrate(log_x, x, &pool).unwrap();
        assert!(
            r.log.steps().iter().any(|s| s.rule_name == "int_log"),
            "should have logged int_log step"
        );
        // Structural check: result contains log(x)
        let result_str = pool.display(r.value).to_string();
        assert!(
            result_str.contains("log"),
            "result should contain log: {result_str}"
        );
    }

    #[test]
    fn integrate_exp_linear_arg() {
        // ∫ exp(2*x) dx = exp(2*x) / 2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let two_x = pool.mul(vec![two, x]);
        let expr = pool.func("exp", vec![two_x]);
        let r = integrate(expr, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_exp_linear"),
            "should fire int_exp_linear"
        );
        // Structural check: result is 2^(-1) * exp(2*x)
        let result_str = pool.display(r.value).to_string();
        assert!(
            result_str.contains("exp"),
            "result should contain exp: {result_str}"
        );
    }

    #[test]
    fn integrate_x_times_exp_x() {
        // ∫ x * exp(x) dx = exp(x) * (x - 1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.func("exp", vec![x])]);
        let r = integrate(expr, x, &pool).unwrap();
        assert!(
            r.log.steps().iter().any(|s| s.rule_name == "int_x_exp"),
            "should fire int_x_exp"
        );
        let result_str = pool.display(r.value).to_string();
        assert!(
            result_str.contains("exp"),
            "result should contain exp: {result_str}"
        );
    }

    #[test]
    fn integrate_const_times_x_times_exp_x() {
        // ∫ 3 * x * exp(x) dx  — constant factor should be preserved
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let three = pool.integer(3_i32);
        let expr = pool.mul(vec![three, x, pool.func("exp", vec![x])]);
        let r = integrate(expr, x, &pool).unwrap();
        assert!(
            r.log.steps().iter().any(|s| s.rule_name == "int_x_exp"),
            "should fire int_x_exp for 3*x*exp(x)"
        );
    }

    /// Numeric evaluator supporting exp/sin/cos/tan and `log(abs(.))` (the
    /// `int_x_exp`-family / trig-substitution antiderivatives use these).
    fn eval_exp_trig(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_exp_trig(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args
                .iter()
                .map(|&a| eval_exp_trig(a, x, xv, pool))
                .product(),
            ExprData::Pow { base, exp } => {
                eval_exp_trig(base, x, xv, pool).powf(eval_exp_trig(exp, x, xv, pool))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval_exp_trig(args[0], x, xv, pool);
                match name.as_str() {
                    "exp" => a.exp(),
                    "sin" => a.sin(),
                    "cos" => a.cos(),
                    "tan" => a.tan(),
                    "sec" => 1.0 / a.cos(),
                    "log" => a.ln(),
                    "abs" => a.abs(),
                    other => panic!("eval_exp_trig: unsupported func {other}"),
                }
            }
            other => panic!("eval_exp_trig: unsupported node {other:?}"),
        }
    }

    /// Integrate and assert `d/dx F = f` numerically at a few sample points.
    fn verify_exp_trig(f: ExprId, x: ExprId, pool: &ExprPool) {
        let r = integrate(f, x, pool).unwrap_or_else(|e| panic!("expected elementary: {e:?}"));
        let d = diff(r.value, x, pool).unwrap();
        let ds = simplify(d.value, pool).value;
        for &xv in &[0.3_f64, 0.7, 1.1] {
            let lhs = eval_exp_trig(ds, x, xv, pool);
            let rhs = eval_exp_trig(f, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}\n  F = {}",
                pool.display(r.value)
            );
        }
    }

    #[test]
    fn integrate_x_times_exp_neg3x() {
        // ∫ x·exp(-3x) dx — Bug #1 (PR #153 dsolve fallback): the engine
        // previously declined this with "irreducible product of var-dependent
        // factors" because `try_x_times_func` only matches `exp(var)` exactly
        // (a=1) and `needs_exp_risch` only routes `poly·exp(linear)` to Risch
        // when the surrounding polynomial has degree ≥ 2.  For a≠1, x·exp(a·x)
        // (degree-1 poly) fell into neither path.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg3 = pool.integer(-3_i32);
        let neg3x = pool.mul(vec![neg3, x]);
        let expr = pool.mul(vec![x, pool.func("exp", vec![neg3x])]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_x_times_exp_2x_plus_1() {
        // ∫ x·exp(2x+1) dx — non-unit rate AND nonzero additive constant; also
        // outside `try_x_times_func` (eta = 2x+1 ≠ x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let two_x_plus_1 = pool.add(vec![two_x, pool.integer(1_i32)]);
        let expr = pool.mul(vec![x, pool.func("exp", vec![two_x_plus_1])]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_x_squared_times_exp_neg_x() {
        // ∫ x²·exp(-x) dx — degree-2 poly with non-unit rate (already routed to
        // Risch before this fix; regression check that it still works).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.func("exp", vec![neg_x]),
        ]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_x_times_exp_x_unaffected() {
        // ∫ x·exp(x) dx still goes through `int_x_exp` (basic engine), not Risch.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.func("exp", vec![x])]);
        let r = integrate(expr, x, &pool).unwrap();
        assert!(
            r.log.steps().iter().any(|s| s.rule_name == "int_x_exp"),
            "x*exp(x) should still fire int_x_exp"
        );
        verify_exp_trig(expr, x, &pool);
    }

    // -- Polynomial × trig products (int_poly_trig_ibp) -----------------------

    #[test]
    fn integrate_x_times_sin() {
        // ∫ x·sin(x) dx = sin(x) − x·cos(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.func("sin", vec![x])]);
        let r = integrate(expr, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_poly_trig_ibp"),
            "x·sin(x) should fire int_poly_trig_ibp"
        );
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_x_times_cos() {
        // ∫ x·cos(x) dx = cos(x) + x·sin(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.func("cos", vec![x])]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_x_squared_times_sin() {
        // ∫ x²·sin(x) dx (repeated IBP)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.func("sin", vec![x]),
        ]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_x_squared_times_cos() {
        // ∫ x²·cos(x) dx (repeated IBP)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.func("cos", vec![x]),
        ]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_poly_times_sin() {
        // ∫ (x²+1)·sin(x) dx
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let poly = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let expr = pool.mul(vec![poly, pool.func("sin", vec![x])]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_x_times_sin_linear_arg() {
        // ∫ x·sin(2x+1) dx — linear (non-unit) trig argument.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let arg = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let expr = pool.mul(vec![x, pool.func("sin", vec![arg])]);
        verify_exp_trig(expr, x, &pool);
    }

    // -- Exponential × trig products (int_exp_trig_ibp) -----------------------

    #[test]
    fn integrate_exp_times_sin() {
        // ∫ exp(x)·sin(x) dx = ½·exp(x)·(sin(x) − cos(x))
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.func("exp", vec![x]), pool.func("sin", vec![x])]);
        let r = integrate(expr, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_exp_trig_ibp"),
            "exp(x)·sin(x) should fire int_exp_trig_ibp"
        );
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_exp_times_cos() {
        // ∫ exp(x)·cos(x) dx = ½·exp(x)·(sin(x) + cos(x))
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.func("exp", vec![x]), pool.func("cos", vec![x])]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_exp2x_times_cos3x() {
        // ∫ exp(2x)·cos(3x) dx = exp(2x)·(3·sin(3x) + 2·cos(3x))/13
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let three_x = pool.mul(vec![pool.integer(3_i32), x]);
        let expr = pool.mul(vec![
            pool.func("exp", vec![two_x]),
            pool.func("cos", vec![three_x]),
        ]);
        verify_exp_trig(expr, x, &pool);
    }

    // -- Regressions: existing paths untouched by the new fast-paths ----------

    #[test]
    fn integrate_x_times_exp_x_still_int_x_exp() {
        // ∫ x·exp(x) dx still routes through int_x_exp (not the new trig paths).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.func("exp", vec![x])]);
        let r = integrate(expr, x, &pool).unwrap();
        assert!(
            r.log.steps().iter().any(|s| s.rule_name == "int_x_exp"),
            "x·exp(x) should still fire int_x_exp"
        );
    }

    #[test]
    fn integrate_log_still_works() {
        // ∫ log(x) dx = x·log(x) − x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_exp_trig(pool.func("log", vec![x]), x, &pool);
    }

    #[test]
    fn integrate_x_times_log_still_works() {
        // ∫ x·log(x) dx = x²·log(x)/2 − x²/4
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.func("log", vec![x])]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    #[ignore = "Bug #2 (PR #153 follow-up): ∫ tan(x)·sin(x) dx = ln|sec(x)+tan(x)| - sin(x) \
                requires a Pythagorean-identity rewrite (sin² = 1 - cos² to split \
                sin²/cos into sec - cos) plus `sec` integration support, neither of \
                which exist yet. Out of scope for the contained routing fix in this \
                PR; tracked separately."]
    fn integrate_tan_times_sin() {
        // ∫ tan(x)·sin(x) dx = ln|sec(x) + tan(x)| − sin(x) — Bug #2.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.func("tan", vec![x]), pool.func("sin", vec![x])]);
        verify_exp_trig(expr, x, &pool);
    }

    #[test]
    fn integrate_one_over_linear() {
        // ∫ 1/(2*x + 3) dx = log(2*x + 3) / 2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let three = pool.integer(3_i32);
        let linear = pool.add(vec![pool.mul(vec![two, x]), three]);
        let expr = pool.pow(linear, pool.integer(-1_i32));
        let r = integrate(expr, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_linear_inv"),
            "should fire int_linear_inv"
        );
        let result_str = pool.display(r.value).to_string();
        assert!(
            result_str.contains("log"),
            "result should contain log: {result_str}"
        );
    }

    #[test]
    fn integrate_x_cubed_plus_2x() {
        // ∫ (x³ + 2x) dx — antiderivative check
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(2_i32), x]),
        ]);
        verify(expr, x, &pool);
    }

    #[test]
    fn integrate_derivation_log_nonempty() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = integrate(pool.pow(x, pool.integer(2_i32)), x, &pool).unwrap();
        assert!(
            !r.log.is_empty(),
            "integration should produce a derivation log"
        );
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "int_power_rule"));
    }

    #[test]
    fn integrate_sqrt_x() {
        // ∫ sqrt(x) dx  should succeed (linear P)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let result = integrate(sqrt_x, x, &pool);
        match &result {
            Ok(r) => println!("sqrt(x) integral = {}", pool.display(r.value)),
            Err(e) => println!("ERROR: {e}"),
        }
        assert!(result.is_ok(), "∫ sqrt(x) dx failed: {:?}", result);
    }

    #[test]
    fn integrate_inv_sqrt_x() {
        // ∫ 1/sqrt(x) dx = 2·sqrt(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let inv_sqrt_x = pool.pow(sqrt_x, pool.integer(-1_i32));
        let result = integrate(inv_sqrt_x, x, &pool);
        match &result {
            Ok(r) => println!("1/sqrt(x) integral = {}", pool.display(r.value)),
            Err(e) => println!("ERROR: {e}"),
        }
        assert!(result.is_ok(), "∫ 1/sqrt(x) dx failed: {:?}", result);
    }

    #[test]
    fn integrate_sqrt_x2_plus_1() {
        // ∫ sqrt(x²+1) dx  should succeed (quadratic P)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let p_expr = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let sqrt_p = pool.func("sqrt", vec![p_expr]);
        let result = integrate(sqrt_p, x, &pool);
        match &result {
            Ok(r) => println!("sqrt(x^2+1) integral = {}", pool.display(r.value)),
            Err(e) => println!("ERROR: {e}"),
        }
        assert!(result.is_ok(), "∫ sqrt(x²+1) dx failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Risch Gap 6: crash fix + known-non-elementary certification
    // -----------------------------------------------------------------------

    /// Build `f(arg) / denom` as `Mul([f(arg), denom^(-1)])`.
    fn over(pool: &ExprPool, num: ExprId, denom: ExprId) -> ExprId {
        let inv = pool.pow(denom, pool.integer(-1_i32));
        pool.mul(vec![num, inv])
    }

    #[test]
    fn sin_over_x_is_nonelementary_not_crash() {
        // ∫ sin(x)/x dx = Si(x): previously stack-overflowed; must now certify NE.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = over(&pool, pool.func("sin", vec![x]), x);
        let r = integrate(f, x, &pool);
        assert!(
            matches!(r, Err(IntegrationError::NonElementary(_))),
            "∫ sin(x)/x dx should be NonElementary; got {r:?}"
        );
    }

    #[test]
    fn exp_over_x_is_nonelementary() {
        // ∫ exp(x)/x dx = Ei(x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = over(&pool, pool.func("exp", vec![x]), x);
        let r = integrate(f, x, &pool);
        assert!(
            matches!(r, Err(IntegrationError::NonElementary(_))),
            "∫ exp(x)/x dx should be NonElementary; got {r:?}"
        );
    }

    #[test]
    fn cos_over_linear_is_nonelementary() {
        // ∫ cos(x)/(2x+1) dx is a shifted Ci — non-elementary.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let denom = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let f = over(&pool, pool.func("cos", vec![x]), denom);
        let r = integrate(f, x, &pool);
        assert!(
            matches!(r, Err(IntegrationError::NonElementary(_))),
            "∫ cos(x)/(2x+1) dx should be NonElementary; got {r:?}"
        );
    }

    #[test]
    fn one_over_log_is_nonelementary() {
        // ∫ 1/log(x) dx = li(x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("log", vec![x]), pool.integer(-1_i32));
        let r = integrate(f, x, &pool);
        assert!(
            matches!(r, Err(IntegrationError::NonElementary(_))),
            "∫ 1/log(x) dx should be NonElementary; got {r:?}"
        );
    }

    #[test]
    fn exp_over_x_squared_is_nonelementary() {
        // ∫ exp(x)/x² dx — still an Ei-family non-elementary integral.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = over(&pool, pool.func("exp", vec![x]), x2);
        let r = integrate(f, x, &pool);
        assert!(
            matches!(r, Err(IntegrationError::NonElementary(_))),
            "∫ exp(x)/x² dx should be NonElementary; got {r:?}"
        );
    }

    #[test]
    fn log_over_x_is_elementary_not_misclassified() {
        // ∫ log(x)/x dx = log(x)²/2 is ELEMENTARY — the pre-check must NOT fire
        // (log is not in the special set; only 1/log triggers the li case).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = over(&pool, pool.func("log", vec![x]), x);
        let r = integrate(f, x, &pool);
        assert!(
            !matches!(r, Err(IntegrationError::NonElementary(_))),
            "∫ log(x)/x dx must not be flagged NonElementary; got {r:?}"
        );
    }

    #[test]
    fn x_times_sin_over_x_not_flagged() {
        // x·sin(x)/x = sin(x) is elementary; the extra `var` factor must block
        // the (otherwise tempting) Si pattern match.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let num = pool.mul(vec![x, pool.func("sin", vec![x])]);
        let f = over(&pool, num, x);
        // After construction this may auto-simplify, but the matcher itself must
        // not certify NonElementary on the raw structural form.
        assert!(
            known_nonelementary(f, x, &pool).is_none(),
            "x·sin(x)/x must not be certified NonElementary"
        );
    }

    #[test]
    fn rational_integration_via_fallback() {
        // ∫ 1/(x²−1) dx is solved by the Rothstein–Trager fallback (rule engine
        // returns NotImplemented first).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let r = integrate(f, x, &pool);
        assert!(
            r.is_ok(),
            "∫ 1/(x²−1) dx should integrate via fallback; got {r:?}"
        );
        // Result should contain logarithms.
        assert!(
            pool.display(r.unwrap().value).to_string().contains("log"),
            "expected log terms in the antiderivative"
        );
    }

    #[test]
    fn power_rule_not_regressed_by_fallback() {
        // ∫ x⁻² dx = −x⁻¹ must still come from the power rule, not the fallback.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(-2_i32));
        let r = integrate(f, x, &pool).unwrap();
        // d/dx result == x⁻².
        let d = diff(r.value, x, &pool).unwrap();
        for &xv in &[1.5_f64, 2.5] {
            let lhs = eval_simple(d.value, x, xv, &pool);
            assert!(
                (lhs - xv.powi(-2)).abs() < 1e-9,
                "power rule regressed at {xv}"
            );
        }
    }

    #[test]
    fn arctan_case_via_fallback() {
        // ∫ 1/(x²+1) dx = atan(x), via the Rothstein–Trager / arctan fallback.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let r = integrate(f, x, &pool);
        assert!(r.is_ok(), "∫ 1/(x²+1) dx should integrate; got {r:?}");
        assert!(pool.display(r.unwrap().value).to_string().contains("atan"));
    }

    fn eval_simple(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_simple(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args.iter().map(|&a| eval_simple(a, x, xv, pool)).product(),
            ExprData::Pow { base, exp } => {
                eval_simple(base, x, xv, pool).powf(eval_simple(exp, x, xv, pool))
            }
            other => panic!("eval_simple: unsupported {other:?}"),
        }
    }

    #[test]
    fn plain_sin_not_flagged() {
        // ∫ sin(x) dx = -cos(x): a bare special function (no denominator) is fine.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("sin", vec![x]);
        assert!(integrate(f, x, &pool).is_ok());
        assert!(known_nonelementary(f, x, &pool).is_none());
    }

    // -----------------------------------------------------------------------
    // Logarithmic-derivative rule: ∫ (h'/h)·log(h)^n dx
    // -----------------------------------------------------------------------

    /// Numeric evaluator supporting log (the rule emits log/log-of-log terms).
    fn eval_log(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_log(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args.iter().map(|&a| eval_log(a, x, xv, pool)).product(),
            ExprData::Pow { base, exp } => {
                eval_log(base, x, xv, pool).powf(eval_log(exp, x, xv, pool))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval_log(args[0], x, xv, pool);
                match name.as_str() {
                    "log" => a.ln(),
                    other => panic!("eval_log: unsupported func {other}"),
                }
            }
            other => panic!("eval_log: unsupported node {other:?}"),
        }
    }

    /// Integrate and assert `d/dx F = integrand` numerically at a few points > 1
    /// (so all logs are positive).
    fn verify_log(f: ExprId, x: ExprId, pool: &ExprPool) {
        let r = integrate(f, x, pool).unwrap_or_else(|e| panic!("expected elementary: {e:?}"));
        let d = diff(r.value, x, pool).unwrap();
        let ds = simplify(d.value, pool).value;
        for &xv in &[1.3_f64, 2.1, 3.4] {
            let lhs = eval_log(ds, x, xv, pool);
            let rhs = eval_log(f, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}\n  F = {}",
                pool.display(r.value)
            );
        }
    }

    #[test]
    fn log_derivative_one_over_x_log_x() {
        // ∫ 1/(x·log x) dx = log(log x)   (n = −1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let logx = pool.func("log", vec![x]);
        let f = pool.mul(vec![
            pool.pow(x, pool.integer(-1)),
            pool.pow(logx, pool.integer(-1)),
        ]);
        verify_log(f, x, &pool);
        let r = integrate(f, x, &pool).unwrap();
        assert!(
            pool.display(r.value).to_string().contains("log(log"),
            "expected log(log(x)); got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn log_derivative_negative_powers() {
        // ∫ 1/(x·log(x)^2) dx = −1/log(x)   (n = −2)
        // ∫ 1/(x·log(x)^3) dx = −1/(2·log(x)^2)   (n = −3)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let logx = pool.func("log", vec![x]);
        for m in [2_i32, 3] {
            let f = pool.mul(vec![
                pool.pow(x, pool.integer(-1)),
                pool.pow(logx, pool.integer(-m)),
            ]);
            verify_log(f, x, &pool);
        }
    }

    #[test]
    fn log_derivative_polynomial_argument() {
        // ∫ (2x/(x²+1))·1/log(x²+1) dx = log(log(x²+1))   (h = x²+1, n = −1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let h = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let logh = pool.func("log", vec![h]);
        let dh_over_h = pool.mul(vec![pool.integer(2_i32), x, pool.pow(h, pool.integer(-1))]);
        let f = pool.mul(vec![dh_over_h, pool.pow(logh, pool.integer(-1))]);
        verify_log(f, x, &pool);
    }

    #[test]
    fn log_derivative_does_not_misfire() {
        // The rule must fire ONLY when the coefficient is exactly h'/h.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let logx = pool.func("log", vec![x]);

        // ∫ 1/log(x) dx = li(x): coefficient 1 ≠ 1/x → must stay NonElementary.
        let f = pool.pow(logx, pool.integer(-1));
        assert!(
            matches!(
                integrate(f, x, &pool),
                Err(IntegrationError::NonElementary(_))
            ),
            "∫ 1/log(x) dx must remain NonElementary"
        );

        // ∫ x/log(x) dx: coefficient x ≠ 1/x → the rule must not produce a result.
        let f = pool.mul(vec![x, pool.pow(logx, pool.integer(-1))]);
        assert!(
            integrate(f, x, &pool).is_err(),
            "∫ x/log(x) dx must not be (mis)integrated by the log-derivative rule"
        );
    }

    // -----------------------------------------------------------------------
    // Definite integration (FTC wrapper)
    // -----------------------------------------------------------------------

    /// Minimal numeric evaluator for closed-form definite-integral results
    /// (Integer/Rational/Add/Mul/Pow/log/atan/sqrt; no free symbols expected).
    fn eval_num(expr: ExprId, pool: &ExprPool) -> f64 {
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_num(a, pool)).sum(),
            ExprData::Mul(args) => args.iter().map(|&a| eval_num(a, pool)).product(),
            ExprData::Pow { base, exp } => {
                let b = eval_num(base, pool);
                if let ExprData::Integer(n) = pool.get(exp) {
                    if let Some(k) = n.0.to_i32() {
                        return b.powi(k);
                    }
                }
                b.powf(eval_num(exp, pool))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval_num(args[0], pool);
                match name.as_str() {
                    "log" => a.ln(),
                    "atan" => a.atan(),
                    "sqrt" => a.sqrt(),
                    other => panic!("eval_num: unsupported func {other}"),
                }
            }
            other => panic!("eval_num: unsupported {other:?}"),
        }
    }

    fn assert_num(result: ExprId, expected: f64, pool: &ExprPool) {
        let got = eval_num(result, pool);
        assert!(
            (got - expected).abs() < 1e-9,
            "definite integral = {got}, expected {expected}"
        );
    }

    #[test]
    fn definite_x_squared_0_1() {
        // ∫_0^1 x² dx = 1/3.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(2_i32));
        let r = integrate_definite(f, x, pool.integer(0_i32), pool.integer(1_i32), &pool).unwrap();
        assert_num(r.value, 1.0 / 3.0, &pool);
    }

    #[test]
    fn definite_two_x_0_1() {
        // ∫_0^1 2x dx = 1.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![pool.integer(2_i32), x]);
        let r = integrate_definite(f, x, pool.integer(0_i32), pool.integer(1_i32), &pool).unwrap();
        assert_num(r.value, 1.0, &pool);
    }

    #[test]
    fn definite_one_over_x_1_2() {
        // ∫_1^2 1/x dx = log(2).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(-1_i32));
        let r = integrate_definite(f, x, pool.integer(1_i32), pool.integer(2_i32), &pool).unwrap();
        assert_num(r.value, 2.0_f64.ln(), &pool);
    }

    // ── Interior-pole detection ──────────────────────────────────────────────
    //
    // Before this guard existed, each of these returned a clean, plausible,
    // wrong value via the FTC difference instead of erroring: `∫_{-1}^{1} x^{-2}`
    // gave `-2`, and the two log cases gave residuals containing `log(-1)`.

    fn assert_improper(r: Result<DerivedExpr<ExprId>, IntegrationError>, what: &str) {
        match r {
            Err(IntegrationError::NotImplemented(msg)) => {
                assert!(
                    msg.contains("pole"),
                    "{what}: expected a pole diagnostic, got: {msg}"
                );
            }
            Err(other) => panic!("{what}: expected NotImplemented, got {other:?}"),
            Ok(value) => panic!("{what}: expected an error, got {:?}", value.value),
        }
    }

    #[test]
    fn definite_pole_at_origin_inverse_square_is_rejected() {
        // ∫_{-1}^{1} x^{-2} dx diverges; naive FTC gives F(1) − F(−1) = −2.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(-2_i32));
        assert_improper(
            integrate_definite(f, x, pool.integer(-1_i32), pool.integer(1_i32), &pool),
            "1/x^2 over [-1, 1]",
        );
    }

    #[test]
    fn definite_pole_at_origin_inverse_is_rejected() {
        // ∫_{-1}^{1} x^{-1} dx diverges; naive FTC gives −log(−1).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(-1_i32));
        assert_improper(
            integrate_definite(f, x, pool.integer(-1_i32), pool.integer(1_i32), &pool),
            "1/x over [-1, 1]",
        );
    }

    #[test]
    fn definite_interior_pole_away_from_origin_is_rejected() {
        // ∫_0^2 1/(x²−1) dx has a pole at x = 1, strictly inside.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        assert_improper(
            integrate_definite(f, x, pool.integer(0_i32), pool.integer(2_i32), &pool),
            "1/(x^2-1) over [0, 2]",
        );
    }

    #[test]
    fn definite_pole_at_endpoint_is_rejected() {
        // ∫_0^1 x^{-1} dx is improper at the lower endpoint.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(-1_i32));
        assert_improper(
            integrate_definite(f, x, pool.integer(0_i32), pool.integer(1_i32), &pool),
            "1/x over [0, 1]",
        );
    }

    #[test]
    fn definite_pole_outside_interval_still_integrates() {
        // The guard must not reject a proper integral: the pole of 1/(x²−1) at
        // x = ±1 lies outside [2, 3].
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let r = integrate_definite(f, x, pool.integer(2_i32), pool.integer(3_i32), &pool)
            .expect("pole outside the interval must not be rejected");
        // ∫_2^3 dx/(x²−1) = ½·ln((x−1)/(x+1)) |_2^3 = ½·(ln(1/2) − ln(1/3)).
        let expected = 0.5 * ((1.0_f64 / 2.0).ln() - (1.0_f64 / 3.0).ln());
        assert_num(r.value, expected, &pool);
    }

    #[test]
    fn removable_singularity_is_not_reported_as_a_pole() {
        // (x²−1)/(x−1) reduces to x+1, so x = 1 is removable. The guard must not
        // fire even though the raw denominator vanishes at 1.
        //
        // Asserted against `interior_singularity` directly rather than through
        // `integrate_definite`, because the integrator independently declines
        // this unsimplified product form ("irreducible product of var-dependent
        // factors"). What matters here is only that the guard stays silent.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let numer = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let denom = pool.add(vec![x, pool.integer(-1_i32)]);
        let f = pool.mul(vec![numer, pool.pow(denom, pool.integer(-1_i32))]);
        assert_eq!(
            interior_singularity(f, x, pool.integer(0_i32), pool.integer(2_i32), &pool),
            None,
            "a removable singularity must not be reported as a pole"
        );
    }

    #[test]
    fn non_polynomial_denominator_falls_through() {
        // 1/sin(x) has poles, but the denominator is not polynomial so the check
        // cannot analyse it. It must fall through silently rather than guess.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("sin", vec![x]), pool.integer(-1_i32));
        assert_eq!(
            interior_singularity(f, x, pool.integer(-1_i32), pool.integer(1_i32), &pool),
            None
        );
    }

    #[test]
    fn definite_polynomial_unaffected_by_pole_check() {
        // A pole-free integrand must be untouched: ∫_0^1 x² dx = 1/3.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(2_i32));
        let r = integrate_definite(f, x, pool.integer(0_i32), pool.integer(1_i32), &pool).unwrap();
        assert_num(r.value, 1.0 / 3.0, &pool);
    }

    #[test]
    fn definite_symbolic_bounds_are_not_rejected() {
        // Bounds that are not numeric cannot be compared against root
        // locations; the check must fall through rather than guess.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let f = pool.pow(x, pool.integer(2_i32));
        assert!(integrate_definite(f, x, pool.integer(0_i32), a, &pool).is_ok());
    }

    #[test]
    fn definite_sin_arctan_bounds() {
        // ∫_0^1 1/(x²+1) dx = atan(1) − atan(0) = π/4.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        let r = integrate_definite(f, x, pool.integer(0_i32), pool.integer(1_i32), &pool).unwrap();
        assert_num(r.value, std::f64::consts::FRAC_PI_4, &pool);
    }

    #[test]
    fn definite_nonelementary_propagates() {
        // ∫_0^1 exp(x²) dx — non-elementary antiderivative ⇒ must error, not a
        // (wrong) number.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        let r = integrate_definite(f, x, pool.integer(0_i32), pool.integer(1_i32), &pool);
        assert!(
            r.is_err(),
            "∫_0^1 exp(x²) dx must propagate the integration error, got {r:?}"
        );
    }

    #[test]
    fn definite_unsupported_propagates() {
        // ∫ sin(x)/x dx is non-elementary; the definite form must error too.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![
            pool.func("sin", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let r = integrate_definite(f, x, pool.integer(1_i32), pool.integer(2_i32), &pool);
        assert!(r.is_err(), "∫ sin(x)/x dx must error in definite form");
    }

    // -----------------------------------------------------------------------
    // Infinite bounds (V2-16 pos_infinity): never substitute `∞` as an
    // ordinary symbol — evaluate via `limit`, or error.
    // -----------------------------------------------------------------------

    #[test]
    fn definite_exp_neg_x_0_to_infinity() {
        // ∫_0^∞ exp(-x) dx = 1.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let f = pool.func("exp", vec![neg_x]);
        let r = integrate_definite(f, x, pool.integer(0_i32), pool.pos_infinity(), &pool)
            .unwrap_or_else(|e| panic!("∫_0^∞ exp(-x) dx should evaluate, got error: {e}"));
        assert_eq!(
            r.value,
            pool.integer(1_i32),
            "∫_0^∞ exp(-x) dx = 1, got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn definite_one_over_x_squared_one_to_infinity() {
        // ∫_1^∞ 1/x² dx = 1 (lim_{x→∞} -1/x = 0, so F(∞) - F(1) = 0 - (-1) = 1).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(-2_i32));
        let r = integrate_definite(f, x, pool.integer(1_i32), pool.pos_infinity(), &pool)
            .unwrap_or_else(|e| panic!("∫_1^∞ 1/x² dx should evaluate, got error: {e}"));
        assert_eq!(
            r.value,
            pool.integer(1_i32),
            "∫_1^∞ 1/x² dx = 1, got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn definite_one_over_x_diverges_at_infinity_errors() {
        // ∫_1^∞ 1/x dx = log(x)|_1^∞ diverges (log(x) → ∞). Must NOT fabricate
        // a finite-looking expression by substituting ∞ for x in log(x); must
        // error instead.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(x, pool.integer(-1_i32));
        let r = integrate_definite(f, x, pool.integer(1_i32), pool.pos_infinity(), &pool);
        match r {
            Err(IntegrationError::NotImplemented(_)) => {}
            other => {
                panic!("∫_1^∞ 1/x dx diverges; expected NotImplemented, got {other:?}")
            }
        }
    }

    #[test]
    fn definite_polynomial_diverges_at_infinity_errors() {
        // ∫_0^∞ x dx diverges (lim_{x→∞} x²/2 = ∞). Must error, not return ∞
        // or a finite-looking value from naive substitution.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = integrate_definite(x, x, pool.integer(0_i32), pool.pos_infinity(), &pool);
        assert!(
            matches!(r, Err(IntegrationError::NotImplemented(_))),
            "∫_0^∞ x dx diverges; expected NotImplemented, got {r:?}"
        );
    }

    #[test]
    fn definite_exp_neg_x_neg_infinity_to_zero() {
        // ∫_{-∞}^0 exp(x) dx = 1 — exercises the `-∞` (lower) bound.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("exp", vec![x]);
        let neg_inf = pool.mul(vec![pool.integer(-1_i32), pool.pos_infinity()]);
        let r = integrate_definite(f, x, neg_inf, pool.integer(0_i32), &pool)
            .unwrap_or_else(|e| panic!("∫_{{-∞}}^0 exp(x) dx should evaluate, got error: {e}"));
        assert_eq!(
            r.value,
            pool.integer(1_i32),
            "∫_{{-∞}}^0 exp(x) dx = 1, got {}",
            pool.display(r.value)
        );
    }

    // -----------------------------------------------------------------------
    // Non-linear u-substitution (derivative-divides heuristic)
    // -----------------------------------------------------------------------

    /// Numeric verification of an antiderivative for transcendental integrands
    /// (the `coeffs_equal` helper only handles polynomials).  Checks
    /// `d/dx(F) == f` to ~1e-7 over several non-singular real samples.
    fn verify_numeric(integrand: ExprId, x: ExprId, pool: &ExprPool) {
        let integral = integrate(integrand, x, pool)
            .unwrap_or_else(|e| panic!("integrate failed for {}: {e}", pool.display(integrand)));
        let deriv = diff(integral.value, x, pool).unwrap();
        let d = simplify(deriv.value, pool).value;
        let samples = [0.41_f64, 0.93, 1.37, 2.11, 2.83];
        let mut checked = 0;
        for &xv in &samples {
            let mut env = std::collections::HashMap::new();
            env.insert(x, xv);
            let (Some(dv), Some(fv)) = (
                crate::jit::eval_interp(d, &env, pool),
                crate::jit::eval_interp(integrand, &env, pool),
            ) else {
                continue;
            };
            if !dv.is_finite() || !fv.is_finite() {
                continue;
            }
            assert!(
                (dv - fv).abs() <= 1e-7 * (1.0 + dv.abs().max(fv.abs())),
                "diff(∫f) ≠ f at x={xv}: got {dv}, want {fv}, for f = {}, F = {}",
                pool.display(integrand),
                pool.display(integral.value),
            );
            checked += 1;
        }
        assert!(checked >= 2, "no usable samples to verify antiderivative");
    }

    #[test]
    fn usub_x_sin_x2() {
        // ∫ x·sin(x²) dx = −cos(x²)/2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.mul(vec![x, pool.func("sin", vec![x2])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn usub_2x_exp_x2() {
        // ∫ 2x·e^(x²) dx = e^(x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.mul(vec![pool.integer(2_i32), x, pool.func("exp", vec![x2])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn usub_x_exp_x2() {
        // ∫ x·e^(x²) dx = e^(x²)/2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.mul(vec![x, pool.func("exp", vec![x2])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn usub_lnx_over_x() {
        // ∫ (ln x)/x dx = (ln x)²/2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![
            pool.func("log", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn usub_tan_x() {
        // ∫ tan(x) dx = −ln(cos x)  (g = cos x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("tan", vec![x]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn usub_exp_cos_exp() {
        // ∫ e^x·cos(e^x) dx = sin(e^x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let ex = pool.func("exp", vec![x]);
        let f = pool.mul(vec![ex, pool.func("cos", vec![ex])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn usub_x_cos_x2_plus_1() {
        // ∫ x·cos(x²+1) dx = sin(x²+1)/2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let inner = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.mul(vec![x, pool.func("cos", vec![inner])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn usub_nonelementary_still_errors() {
        // ∫ e^(x²) dx has no elementary antiderivative — must NOT be fabricated.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.func("exp", vec![x2]);
        let r = integrate(f, x, &pool);
        assert!(
            r.is_err(),
            "∫ e^(x²) dx must error, got {:?}",
            r.map(|d| pool.display(d.value))
        );
    }

    #[test]
    fn usub_does_not_disturb_basic_rules() {
        // Pre-existing cases must still be solved (by the rules, not u-subst).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // ∫ sin x dx
        let sinx = pool.func("sin", vec![x]);
        verify_numeric(sinx, x, &pool);
        // ∫ x² dx
        let x2 = pool.pow(x, pool.integer(2_i32));
        verify(x2, x, &pool);
        // ∫ e^x dx
        let ex = pool.func("exp", vec![x]);
        verify_numeric(ex, x, &pool);
        // ∫ 1/x dx
        let inv = pool.pow(x, pool.integer(-1_i32));
        verify_numeric(inv, x, &pool);
    }

    // --- Inverse-trigonometric integration by parts (atan / asin / acos) ---

    #[test]
    fn integrate_atan() {
        // ∫ atan(x) dx = x·atan(x) − ½·log(1+x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("atan", vec![x]);
        let r = integrate(f, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_inverse_trig_ibp"),
            "should fire int_inverse_trig_ibp"
        );
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_x_times_atan() {
        // ∫ x·atan(x) dx = ½(x²+1)·atan(x) − x/2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("atan", vec![x])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_x_squared_times_atan() {
        // ∫ x²·atan(x) dx
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.mul(vec![x2, pool.func("atan", vec![x])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_atan_over_x_squared() {
        // ∫ atan(x)/x² dx = −atan(x)/x + log(x) − ½·log(1+x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x_inv2 = pool.pow(x, pool.integer(-2_i32));
        let f = pool.mul(vec![pool.func("atan", vec![x]), x_inv2]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_asin() {
        // ∫ asin(x) dx = x·asin(x) + √(1−x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("asin", vec![x]);
        let r = integrate(f, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_inverse_trig_ibp"),
            "should fire int_inverse_trig_ibp"
        );
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_x_times_asin() {
        // ∫ x·asin(x) dx
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("asin", vec![x])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_acos() {
        // ∫ acos(x) dx = x·acos(x) − √(1−x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("acos", vec![x]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_x_times_acos() {
        // ∫ x·acos(x) dx
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("acos", vec![x])]);
        verify_numeric(f, x, &pool);
    }

    // --- Inverse-hyperbolic integration by parts (asinh / acosh / atanh) ---

    #[test]
    fn integrate_asinh() {
        // ∫ asinh(x) dx = x·asinh(x) − √(x²+1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("asinh", vec![x]);
        let r = integrate(f, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_inverse_trig_ibp"),
            "should fire int_inverse_trig_ibp"
        );
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_acosh() {
        // ∫ acosh(x) dx = x·acosh(x) − √(x²−1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("acosh", vec![x]);
        let r = integrate(f, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_inverse_trig_ibp"),
            "should fire int_inverse_trig_ibp"
        );
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_atanh() {
        // ∫ atanh(x) dx = x·atanh(x) + ½·log(1−x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("atanh", vec![x]);
        let r = integrate(f, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_inverse_trig_ibp"),
            "should fire int_inverse_trig_ibp"
        );
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_x_times_asinh() {
        // ∫ x·asinh(x) dx (residual ∫ x²/√(x²+1) resolves via the √-quadratic engine)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("asinh", vec![x])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_x_times_atanh() {
        // ∫ x·atanh(x) dx (residual ∫ x²/(1−x²) resolves via the rational engine)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("atanh", vec![x])]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_inverse_hyperbolic_diff_table_ok() {
        // Regression: d/dx of each inverse-hyperbolic function is non-zero.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        for name in ["asinh", "acosh", "atanh"] {
            let d = diff(pool.func(name, vec![x]), x, &pool).unwrap();
            assert_ne!(
                d.value,
                pool.integer(0_i32),
                "d/dx {name}(x) must be non-zero"
            );
        }
    }

    // --- Integer powers of inverse functions (IBP reduction) ---

    #[test]
    fn integrate_asin_squared() {
        // ∫ asin(x)² dx = x·asin(x)² + 2√(1−x²)·asin(x) − 2x (algebraic derivative
        // ⇒ elementary).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("asin", vec![x]), pool.integer(2_i32));
        let r = integrate(f, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "int_inverse_trig_ibp"),
            "should fire int_inverse_trig_ibp"
        );
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_acos_squared() {
        // ∫ acos(x)² dx — elementary (algebraic derivative).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("acos", vec![x]), pool.integer(2_i32));
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_asinh_squared() {
        // ∫ asinh(x)² dx — elementary (algebraic derivative).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("asinh", vec![x]), pool.integer(2_i32));
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_acosh_squared() {
        // ∫ acosh(x)² dx — elementary (algebraic derivative).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("acosh", vec![x]), pool.integer(2_i32));
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_x_times_asin_squared() {
        // ∫ x·asin(x)² dx — elementary (algebraic derivative, polynomial factor).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let asin2 = pool.pow(pool.func("asin", vec![x]), pool.integer(2_i32));
        let f = pool.mul(vec![x, asin2]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_asin_cubed() {
        // ∫ asin(x)³ dx — elementary (deeper IBP recursion, still algebraic).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("asin", vec![x]), pool.integer(3_i32));
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_atan_squared_declines() {
        // ∫ atan(x)² dx is NON-elementary — must decline cleanly (no panic, no
        // wrong closed form).  The IBP residual ∫ log(1+x²)/(1+x²) dx is a
        // dilog-type non-elementary integral (rational derivative 1/(1+x²)).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("atan", vec![x]), pool.integer(2_i32));
        let r = integrate(f, x, &pool);
        assert!(
            r.is_err(),
            "∫ atan(x)² dx should decline, got {:?}",
            r.map(|d| pool.display(d.value))
        );
    }

    #[test]
    fn integrate_atanh_squared_declines() {
        // ∫ atanh(x)² dx is NON-elementary — the residual ∫ log(1−x²)/(1−x²) dx
        // is non-elementary (rational derivative 1/(1−x²)).  Decline cleanly.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("atanh", vec![x]), pool.integer(2_i32));
        let r = integrate(f, x, &pool);
        assert!(
            r.is_err(),
            "∫ atanh(x)² dx should decline, got {:?}",
            r.map(|d| pool.display(d.value))
        );
    }

    #[test]
    fn integrate_atan_diff_table_ok() {
        // Regression: d/dx atan(x) = 1/(1+x²), asin/acos non-zero (diff-table sanity).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        for name in ["atan", "asin", "acos"] {
            let d = diff(pool.func(name, vec![x]), x, &pool).unwrap();
            assert_ne!(
                d.value,
                pool.integer(0_i32),
                "d/dx {name}(x) must be non-zero"
            );
        }
    }

    // ---------------------------------------------------------------------
    // Trigonometric powers and products (Fourier linearization fast-path)
    // ---------------------------------------------------------------------

    fn sinp(x: ExprId, n: i32, pool: &ExprPool) -> ExprId {
        pool.pow(pool.func("sin", vec![x]), pool.integer(n))
    }
    fn cosp(x: ExprId, n: i32, pool: &ExprPool) -> ExprId {
        pool.pow(pool.func("cos", vec![x]), pool.integer(n))
    }

    #[test]
    fn integrate_sin_squared() {
        // ∫ sin²(x) dx = x/2 − sin(2x)/4
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(sinp(x, 2, &pool), x, &pool);
    }

    #[test]
    fn integrate_cos_squared() {
        // ∫ cos²(x) dx = x/2 + sin(2x)/4
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(cosp(x, 2, &pool), x, &pool);
    }

    #[test]
    fn integrate_sin_cubed() {
        // ∫ sin³(x) dx = cos³(x)/3 − cos(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(sinp(x, 3, &pool), x, &pool);
    }

    #[test]
    fn integrate_cos_cubed() {
        // ∫ cos³(x) dx = sin(x) − sin³(x)/3
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(cosp(x, 3, &pool), x, &pool);
    }

    #[test]
    fn integrate_sin_squared_cos_squared() {
        // ∫ sin²(x)·cos²(x) dx = x/8 − sin(4x)/32
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![sinp(x, 2, &pool), cosp(x, 2, &pool)]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_sin_2x_times_cos_x() {
        // ∫ sin(2x)·cos(x) dx  (product-to-sum of different frequencies)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let f = pool.mul(vec![
            pool.func("sin", vec![two_x]),
            pool.func("cos", vec![x]),
        ]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_sin_x_times_sin_2x() {
        // ∫ sin(x)·sin(2x) dx  (product-to-sum, cos family)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let f = pool.mul(vec![
            pool.func("sin", vec![x]),
            pool.func("sin", vec![two_x]),
        ]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_cos_x_times_cos_3x() {
        // ∫ cos(x)·cos(3x) dx  (product-to-sum, cos family)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let three_x = pool.mul(vec![pool.integer(3_i32), x]);
        let f = pool.mul(vec![
            pool.func("cos", vec![x]),
            pool.func("cos", vec![three_x]),
        ]);
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_sec_squared() {
        // ∫ 1/cos²(x) dx = tan(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(cosp(x, -2, &pool), x, &pool);
    }

    #[test]
    fn integrate_csc_squared() {
        // ∫ 1/sin²(x) dx = −cot(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(sinp(x, -2, &pool), x, &pool);
    }

    #[test]
    fn integrate_tan_squared() {
        // ∫ tan²(x) dx = tan(x) − x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("tan", vec![x]), pool.integer(2_i32));
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_sin_squared_linear_arg() {
        // ∫ sin²(2x+1) dx  (linear argument a·x+b)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let arg = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let f = pool.pow(pool.func("sin", vec![arg]), pool.integer(2_i32));
        verify_numeric(f, x, &pool);
    }

    #[test]
    fn integrate_trig_powers_do_not_regress_basics() {
        // The new fast-path must not disturb the already-working simple cases.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // ∫ sin(x), ∫ cos(x)
        verify_numeric(pool.func("sin", vec![x]), x, &pool);
        verify_numeric(pool.func("cos", vec![x]), x, &pool);
        // ∫ tan(x) = −log(cos x)
        verify_numeric(pool.func("tan", vec![x]), x, &pool);
        // ∫ sin(x)·cos(x)
        let sc = pool.mul(vec![pool.func("sin", vec![x]), pool.func("cos", vec![x])]);
        verify_numeric(sc, x, &pool);
        // ∫ x·sin(x)  (poly·trig IBP path still owns this)
        let xsin = pool.mul(vec![x, pool.func("sin", vec![x])]);
        verify_numeric(xsin, x, &pool);
    }

    #[test]
    fn integrate_unsupported_trig_shape_declines_cleanly() {
        // ∫ sin(x)/x is non-elementary; must decline (no panic), not fabricate.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![
            pool.func("sin", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        assert!(integrate(f, x, &pool).is_err(), "∫ sin(x)/x should decline");
        // ∫ 1/cos¹⁰(x): the reciprocal-trig reduction is capped at n ≤ 8, so a
        // power above the cap must decline cleanly rather than blow up.
        let sec10 = cosp(x, -10, &pool);
        assert!(
            integrate(sec10, x, &pool).is_err(),
            "∫ 1/cos¹⁰(x) is above the reduction cap — should decline, not panic"
        );
    }

    // ---------------------------------------------------------------------
    // Reciprocal trig powers: ∫ secⁿ / ∫ cscⁿ (negative sin/cos powers)
    // ---------------------------------------------------------------------

    /// `sec(x)^m` as it parses after desugaring: the nested `(cos(x)^(-1))^m`.
    fn nested_sec(x: ExprId, m: i32, pool: &ExprPool) -> ExprId {
        let sec = pool.pow(pool.func("cos", vec![x]), pool.integer(-1_i32));
        pool.pow(sec, pool.integer(m))
    }
    fn nested_csc(x: ExprId, m: i32, pool: &ExprPool) -> ExprId {
        let csc = pool.pow(pool.func("sin", vec![x]), pool.integer(-1_i32));
        pool.pow(csc, pool.integer(m))
    }

    #[test]
    fn integrate_sec_squared_nested() {
        // ∫ sec(x)² dx — the nested (cos(x)^(-1))^2 spelling must close to tan(x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(nested_sec(x, 2, &pool), x, &pool);
    }

    #[test]
    fn integrate_csc_squared_nested() {
        // ∫ csc(x)² dx — nested (sin(x)^(-1))^2 spelling must close to −cot(x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(nested_csc(x, 2, &pool), x, &pool);
    }

    #[test]
    fn integrate_sec_squared_flattened() {
        // ∫ 1/cos(x)² dx = tan(x) (flattened spelling still closes).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(cosp(x, -2, &pool), x, &pool);
    }

    #[test]
    fn integrate_sec() {
        // ∫ sec(x) dx = log((1+sin x)/cos x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(nested_sec(x, 1, &pool), x, &pool);
        verify_numeric(cosp(x, -1, &pool), x, &pool);
    }

    #[test]
    fn integrate_csc() {
        // ∫ csc(x) dx = log((1−cos x)/sin x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(nested_csc(x, 1, &pool), x, &pool);
        verify_numeric(sinp(x, -1, &pool), x, &pool);
    }

    #[test]
    fn integrate_sec_cubed() {
        // ∫ sec(x)³ dx via the reduction formula.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(nested_sec(x, 3, &pool), x, &pool);
        verify_numeric(cosp(x, -3, &pool), x, &pool);
    }

    #[test]
    fn integrate_csc_cubed() {
        // ∫ csc(x)³ dx via the reduction formula.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(nested_csc(x, 3, &pool), x, &pool);
        verify_numeric(sinp(x, -3, &pool), x, &pool);
    }

    #[test]
    fn integrate_sec_quartic() {
        // ∫ sec(x)⁴ dx (even power, recurses to the tan base case).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        verify_numeric(nested_sec(x, 4, &pool), x, &pool);
    }

    #[test]
    fn integrate_sec_linear_arg() {
        // ∫ sec(2x+1) dx — the chain-rule 1/a factor must be applied.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let arg = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let f = pool.pow(pool.func("cos", vec![arg]), pool.integer(-1_i32));
        verify_numeric(f, x, &pool);
    }

    // -----------------------------------------------------------------------
    // Weierstrass half-angle substitution: rational functions of sin/cos.
    // -----------------------------------------------------------------------

    /// True when the derivation log for `∫ integrand dx` contains the Weierstrass
    /// rule step (i.e. the half-angle path is what closed the integral).
    fn weierstrass_fired(integrand: ExprId, x: ExprId, pool: &ExprPool) -> bool {
        let integral = integrate(integrand, x, pool).unwrap();
        integral
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "int_weierstrass_trig")
    }

    #[test]
    fn weierstrass_one_over_2_plus_cos() {
        // ∫ 1/(2+cos x) dx = (2/√3)·atan(tan(x/2)/√3)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let denom = pool.add(vec![pool.integer(2_i32), pool.func("cos", vec![x])]);
        let f = pool.pow(denom, pool.integer(-1_i32));
        verify_numeric(f, x, &pool);
        assert!(weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_one_over_1_plus_sin() {
        // ∫ 1/(1+sin x) dx = −2/(1+tan(x/2))
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let denom = pool.add(vec![pool.integer(1_i32), pool.func("sin", vec![x])]);
        let f = pool.pow(denom, pool.integer(-1_i32));
        verify_numeric(f, x, &pool);
        assert!(weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_one_over_5_plus_4cos() {
        // ∫ 1/(5+4cos x) dx = (2/3)·atan(tan(x/2)/3)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let denom = pool.add(vec![
            pool.integer(5_i32),
            pool.mul(vec![pool.integer(4_i32), pool.func("cos", vec![x])]),
        ]);
        let f = pool.pow(denom, pool.integer(-1_i32));
        verify_numeric(f, x, &pool);
        assert!(weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_one_over_sin_plus_cos() {
        // ∫ 1/(sin x + cos x) dx
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let denom = pool.add(vec![pool.func("sin", vec![x]), pool.func("cos", vec![x])]);
        let f = pool.pow(denom, pool.integer(-1_i32));
        verify_numeric(f, x, &pool);
        assert!(weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_sin_over_1_plus_sin() {
        // ∫ sin x/(1+sin x) dx
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sinx = pool.func("sin", vec![x]);
        let denom = pool.add(vec![pool.integer(1_i32), sinx]);
        let f = pool.mul(vec![sinx, pool.pow(denom, pool.integer(-1_i32))]);
        verify_numeric(f, x, &pool);
        assert!(weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_one_over_2_plus_sin() {
        // ∫ 1/(2+sin x) dx = (2/√3)·atan((2·tan(x/2)+1)/√3)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let denom = pool.add(vec![pool.integer(2_i32), pool.func("sin", vec![x])]);
        let f = pool.pow(denom, pool.integer(-1_i32));
        verify_numeric(f, x, &pool);
        assert!(weierstrass_fired(f, x, &pool));
    }

    // Regression: the dedicated trig fast-paths keep their nicer closed forms —
    // the Weierstrass path must NOT intercept them.

    #[test]
    fn weierstrass_does_not_intercept_sin() {
        // ∫ sin x dx stays −cos(x), not a half-angle form.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("sin", vec![x]);
        let r = integrate(f, x, &pool).unwrap();
        let expected = pool.mul(vec![pool.integer(-1_i32), pool.func("cos", vec![x])]);
        assert!(coeffs_equal(r.value, expected, x, &pool));
        assert!(!weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_does_not_intercept_cos() {
        // ∫ cos x dx stays sin(x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("cos", vec![x]);
        let r = integrate(f, x, &pool).unwrap();
        assert_eq!(r.value, pool.func("sin", vec![x]));
        assert!(!weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_does_not_intercept_sin_squared() {
        // ∫ sin²x dx keeps the Fourier-linearized form (no half-angle).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = sinp(x, 2, &pool);
        verify_numeric(f, x, &pool);
        assert!(!weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_does_not_intercept_sec_squared() {
        // ∫ sec²x dx keeps the tan(x) closed form.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.pow(pool.func("cos", vec![x]), pool.integer(-2_i32));
        verify_numeric(f, x, &pool);
        assert!(!weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_does_not_intercept_tan() {
        // ∫ tan x dx = −log(cos x) via u-substitution, not half-angle.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("tan", vec![x]);
        verify_numeric(f, x, &pool);
        assert!(!weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_does_not_intercept_sin2x_cos_x() {
        // ∫ sin(2x)·cos(x) dx keeps the Fourier-linearized form.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let f = pool.mul(vec![
            pool.func("sin", vec![two_x]),
            pool.func("cos", vec![x]),
        ]);
        verify_numeric(f, x, &pool);
        assert!(!weierstrass_fired(f, x, &pool));
    }

    #[test]
    fn weierstrass_declines_non_rational_trig() {
        // ∫ sin(x)/x dx is non-elementary: the Weierstrass rewrite hits a bare
        // `x` and must decline cleanly (no panic, returns an error).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![
            pool.func("sin", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        assert!(integrate(f, x, &pool).is_err());
    }
}
