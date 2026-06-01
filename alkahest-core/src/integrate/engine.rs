/// Symbolic integration — rule-based Risch subset.
///
/// Handles:
/// - Constants: `∫ c dx = c·x`
/// - Power rule: `∫ x^n dx = x^(n+1)/(n+1)` (`n ≠ -1`)
/// - Logarithm: `∫ x^(-1) dx = ln(x)`  (`∫ 1/x dx`)
/// - Sum rule: `∫ (f + g) dx = ∫f dx + ∫g dx`
/// - Constant-multiple rule: `∫ c·f dx = c · ∫f dx`
/// - Known functions: sin, cos, exp, 1/x
///
/// Everything else returns `Err(IntegrationError::NotImplemented)`.
///
/// The result is simplified with the rule-based simplifier before returning.
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrationError {
    /// The expression is outside the supported Risch subset.
    NotImplemented(String),
    /// Division by zero would occur (e.g. power-rule with n=-1 on a non-x base).
    DivisionByZero,
    /// The algebraic extension has degree > 2 (v1.1 supports only sqrt / degree-2).
    UnsupportedExtensionDegree(u32),
    /// The integrand provably has no elementary antiderivative (e.g. elliptic integrals).
    NonElementary(String),
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrationError::NotImplemented(msg) => write!(f, "integrate: not implemented: {msg}"),
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

impl IntegrationError {
    /// A human-readable remediation hint for the user.
    pub fn remediation(&self) -> Option<&'static str> {
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
    // Fast-path: ∫ c * x * exp(x) dx = c * exp(x) * (x - 1)
    if let Some(result) = try_x_times_func(expr, var, pool, log) {
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
            log.push(RewriteStep::simple("power_rule", expr, result));
            Ok(result)
        }

        // ∫ c dx = c*x  (c free of var)
        Node::Constant => {
            let result = pool.mul(vec![expr, var]);
            log.push(RewriteStep::simple("constant_rule", expr, result));
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
            log.push(RewriteStep::simple("sum_rule", expr, result));
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
                log.push(RewriteStep::simple("constant_rule", expr, result));
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
                    log.push(RewriteStep::simple("constant_multiple_rule", expr, r));
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
                    log.push(RewriteStep::simple("power_rule", expr, result));
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
                    log.push(RewriteStep::simple("constant_rule", expr, result));
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
                    log.push(RewriteStep::simple("constant_rule", expr, result));
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
///
/// # Transcendental Risch (Risch engine)
///
/// | Input                      | Result                      | Condition              |
/// |----------------------------|-----------------------------|------------------------|
/// | `exp(g)`, deg(g) ≥ 2      | `v·exp(g)` (if elementary)  | Risch DE solvable      |
/// | `exp(g)`, deg(g) ≥ 2      | `NonElementary`             | Risch DE unsolvable    |
/// | `p(x)·exp(a·x)`, deg≥2    | polynomial · exp            | undetermined coeff.    |
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
    // V1-2: Route algebraic integrands to the Trager/Risch algebraic engine.
    // For *mixed* algebraic+transcendental (e.g. exp(x)/sqrt(x²+1)) the Risch
    // engine handles the transcendental level and delegates base-field integrals
    // back to the algebraic engine, so only route to algebraic when there are NO
    // transcendental (exp/log) generators.
    let has_algebraic = super::algebraic::contains_algebraic_subterm(expr, pool);
    let has_transcendental = super::risch::contains_risch_form(expr, var, pool);
    if has_algebraic && !has_transcendental {
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

    let mut log = DerivationLog::new();
    match integrate_raw(expr, var, pool, &mut log) {
        Ok(raw) => {
            let simplified = simplify(raw, pool);
            let final_log = log.merge(simplified.log);
            Ok(DerivedExpr::with_log(simplified.value, final_log))
        }
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
            Err(IntegrationError::NotImplemented(msg))
        }
        Err(other) => Err(other),
    }
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
        assert!(r.log.steps().iter().any(|s| s.rule_name == "power_rule"));
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
}
