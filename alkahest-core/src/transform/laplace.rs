//! Symbolic Laplace transform `L{f(t)}(s)` and inverse `L⁻¹{F(s)}(t)`.
//!
//! # Forward transform
//!
//! [`laplace_transform`] is a rule/table-based structural recursion over `f(t)`:
//!
//! | `f(t)`                 | `L{f}(s)`                       | rule              |
//! |------------------------|---------------------------------|-------------------|
//! | `c` (const)            | `c/s`                           | constant          |
//! | `t^n` (`n ∈ ℤ₊`)       | `n! / s^{n+1}`                  | power             |
//! | `e^{a t}`              | `1/(s−a)`                       | exponential       |
//! | `sin(b t)`             | `b/(s²+b²)`                     | sine              |
//! | `cos(b t)`             | `s/(s²+b²)`                     | cosine            |
//! | `sinh(b t)`            | `b/(s²−b²)`                     | hyperbolic sine   |
//! | `cosh(b t)`            | `s/(s²−b²)`                     | hyperbolic cosine |
//! | `α·f + β·g`            | `α·L{f} + β·L{g}`               | linearity         |
//! | `e^{a t}·f(t)`         | `F(s−a)`                        | s-shift theorem   |
//! | `t^n·f(t)`             | `(−1)^n F^{(n)}(s)`             | frequency-diff    |
//! | `θ(t−a)·g(t−a)`        | `e^{−a s} G(s)`                | t-shift (`a ≥ 0`) |
//! | `θ(t−a)`               | `e^{−a s}/s`                    | shifted step      |
//! | `δ(t−a)`               | `e^{−a s}`  (`δ(t) ↦ 1`)        | impulse           |
//!
//! The derivative rule `L{f^{(n)}} = sⁿF − sⁿ⁻¹f(0) − … − f^{(n−1)}(0)` is
//! exposed separately as [`laplace_derivative_rule`] for the ODE workflow (it
//! operates on the *symbol* `F = L{f}` plus initial values, since `f` itself is
//! an unknown function).
//!
//! # Inverse transform
//!
//! [`inverse_laplace_transform`] inverts a **proper rational** `F(s) = p(s)/q(s)`
//! by partial fractions (via [`crate::poly::apart`]) and a per-term table:
//!
//! | term in `F(s)`              | `L⁻¹` term                       |
//! |-----------------------------|----------------------------------|
//! | `A/(s−a)^n`                 | `A·t^{n−1} e^{a t}/(n−1)!`        |
//! | `(B s + C)/((s−p)²+ω²)`     | damped sin/cos (`n = 1`)         |
//! | `(B s + C)/((s−p)²+ω²)²`    | `t`-weighted damped sin/cos      |
//! | `e^{−a s} F(s)`             | `θ(t−a)·(L⁻¹F)(t−a)`             |
//!
//! A leading polynomial part of `F` (improper rational) maps back to derivatives
//! of `δ(t)`, which we decline rather than fabricate.
//!
//! # Caveats
//!
//! Both directions are **formal** — no convergence region is computed and no
//! existence side-conditions are attached.  Unrecognised forms return
//! [`LaplaceError::NoRule`] (forward) / [`LaplaceError::NotInvertible`] (inverse)
//! rather than guessing.

use rug::Integer;

use crate::kernel::{ExprData, ExprId, ExprPool};

/// Errors from the Laplace transform routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaplaceError {
    /// No forward rule matched `f(t)` (E-TRANSFORM-001).
    NoRule(String),
    /// The inverse-transform input is not a form the table can invert
    /// (E-TRANSFORM-002).
    NotInvertible(String),
    /// The frequency variable `s` and time variable `t` must be distinct
    /// symbols (E-TRANSFORM-003).
    SameVariable,
}

impl std::fmt::Display for LaplaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaplaceError::NoRule(m) => {
                write!(f, "laplace_transform: no rule for {m} [E-TRANSFORM-001]")
            }
            LaplaceError::NotInvertible(m) => write!(
                f,
                "inverse_laplace_transform: cannot invert {m} [E-TRANSFORM-002]"
            ),
            LaplaceError::SameVariable => write!(
                f,
                "laplace_transform: time and frequency variables must differ [E-TRANSFORM-003]"
            ),
        }
    }
}

impl std::error::Error for LaplaceError {}

// ===========================================================================
// Small helpers
// ===========================================================================

fn is_free_of(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    crate::integrate::risch::poly_rde::is_free_of_var(expr, var, pool)
}

/// `n!` as an interned integer.
fn factorial(n: u64, pool: &ExprPool) -> ExprId {
    let mut acc = Integer::from(1);
    for k in 2..=n {
        acc *= Integer::from(k);
    }
    pool.integer(acc)
}

/// Extract `a` from an expression that is `a·var + b` with `a, b` free of `var`,
/// returning `(a, b)`.  Returns `None` when the expression is not affine in
/// `var`.
fn as_affine(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if expr == var {
        return Some((pool.integer(1_i32), pool.integer(0_i32)));
    }
    if is_free_of(expr, var, pool) {
        return Some((pool.integer(0_i32), expr));
    }
    match pool.get(expr) {
        ExprData::Mul(_) => {
            let (a, b) = as_affine_term(expr, var, pool)?;
            // a single Mul term has no constant part
            if b == pool.integer(0_i32) {
                Some((a, pool.integer(0_i32)))
            } else {
                None
            }
        }
        ExprData::Add(args) => {
            let mut a_acc: Vec<ExprId> = Vec::new();
            let mut b_acc: Vec<ExprId> = Vec::new();
            for arg in args {
                if is_free_of(arg, var, pool) {
                    b_acc.push(arg);
                } else {
                    let (a, b) = as_affine_term(arg, var, pool)?;
                    if b != pool.integer(0_i32) {
                        return None;
                    }
                    a_acc.push(a);
                }
            }
            let a = match a_acc.len() {
                0 => pool.integer(0_i32),
                1 => a_acc[0],
                _ => pool.add(a_acc),
            };
            let b = match b_acc.len() {
                0 => pool.integer(0_i32),
                1 => b_acc[0],
                _ => pool.add(b_acc),
            };
            Some((a, b))
        }
        _ => None,
    }
}

/// Affine analysis of a single (non-Add) term: returns `(coeff, 0)` for a term
/// `coeff·var`, else `None` unless it is free of `var` (`(0, term)`).
fn as_affine_term(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if expr == var {
        return Some((pool.integer(1_i32), pool.integer(0_i32)));
    }
    if is_free_of(expr, var, pool) {
        return Some((pool.integer(0_i32), expr));
    }
    if let ExprData::Mul(args) = pool.get(expr) {
        let pos = args.iter().position(|&a| a == var)?;
        let others: Vec<ExprId> = args
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &a)| a)
            .collect();
        if others.iter().all(|&o| is_free_of(o, var, pool)) {
            let coeff = match others.len() {
                0 => pool.integer(1_i32),
                1 => others[0],
                _ => pool.mul(others),
            };
            return Some((coeff, pool.integer(0_i32)));
        }
    }
    None
}

/// Simplify and return the bare `ExprId`.
fn simp(expr: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::simplify(expr, pool).value
}

fn neg(expr: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), expr])
}

fn recip(expr: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(expr, pool.integer(-1_i32))
}

// ===========================================================================
// Forward transform
// ===========================================================================

/// Compute the Laplace transform `L{f(t)}(s) = ∫₀^∞ f(t) e^{−s t} dt`.
///
/// `t` is the time variable, `s` the frequency variable; both must be distinct
/// symbols.  This is a *formal* transform — see the [module docs](self) for the
/// rule table, caveats, and declines.
///
/// # Errors
///
/// - [`LaplaceError::SameVariable`] if `t == s`.
/// - [`LaplaceError::NoRule`] if no table rule matches `f(t)`.
///
/// # Examples
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::transform::laplace_transform;
///
/// let pool = ExprPool::new();
/// let t = pool.symbol("t", Domain::Real);
/// let s = pool.symbol("s", Domain::Real);
/// // L{1}(s) = 1/s
/// let one = pool.integer(1_i32);
/// let f = laplace_transform(one, t, s, &pool).unwrap();
/// assert_eq!(pool.display(f).to_string(), pool.display(pool.pow(s, pool.integer(-1_i32))).to_string());
/// ```
pub fn laplace_transform(
    f: ExprId,
    t: ExprId,
    s: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LaplaceError> {
    if t == s {
        return Err(LaplaceError::SameVariable);
    }
    let out = laplace_inner(f, t, s, pool, 0)?;
    Ok(simp(out, pool))
}

const MAX_DEPTH: usize = 32;

fn laplace_inner(
    f: ExprId,
    t: ExprId,
    s: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, LaplaceError> {
    if depth > MAX_DEPTH {
        return Err(LaplaceError::NoRule("recursion depth exceeded".into()));
    }

    // Constant (free of t): L{c} = c/s.
    if is_free_of(f, t, pool) {
        return Ok(pool.mul(vec![f, recip(s, pool)]));
    }

    // Bare t: L{t} = 1/s².
    if f == t {
        return Ok(recip(pool.pow(s, pool.integer(2_i32)), pool));
    }

    match pool.get(f) {
        // Linearity over sums.
        ExprData::Add(args) => {
            let mut terms = Vec::with_capacity(args.len());
            for a in args {
                terms.push(laplace_inner(a, t, s, pool, depth + 1)?);
            }
            Ok(pool.add(terms))
        }

        // Products: split off the t-free scalar (linearity), then dispatch the
        // remaining t-dependent factor through the structural product rules.
        ExprData::Mul(args) => laplace_mul(&args, t, s, pool, depth),

        // Pure power t^n.
        ExprData::Pow { base, exp } if base == t => {
            if let Some(n) = nonneg_int_exp(exp, pool) {
                // L{t^n} = n! / s^{n+1}
                let fact = factorial(n, pool);
                let denom = pool.pow(s, pool.integer(Integer::from(n + 1)));
                Ok(pool.mul(vec![fact, recip(denom, pool)]))
            } else {
                Err(LaplaceError::NoRule(format!(
                    "t^e with non-integer exponent: {}",
                    pool.display(f)
                )))
            }
        }

        ExprData::Func { name, args } if args.len() == 1 => {
            laplace_func(&name, args[0], t, s, pool, depth)
        }

        _ => Err(LaplaceError::NoRule(pool.display(f).to_string())),
    }
}

/// Non-negative integer value of an exponent `ExprId`, if it is one.
fn nonneg_int_exp(exp: ExprId, pool: &ExprPool) -> Option<u64> {
    if let ExprData::Integer(n) = pool.get(exp) {
        let n = n.0;
        if n >= 0 {
            return n.to_u64();
        }
    }
    None
}

/// Laplace transform of a product `∏ args`.
fn laplace_mul(
    args: &[ExprId],
    t: ExprId,
    s: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, LaplaceError> {
    // Pull out the constant (t-free) scalar prefactor.
    let (consts, rest): (Vec<ExprId>, Vec<ExprId>) =
        args.iter().partition(|&&a| is_free_of(a, t, pool));
    let scalar = match consts.len() {
        0 => None,
        1 => Some(consts[0]),
        _ => Some(pool.mul(consts.clone())),
    };

    let inner = match rest.len() {
        0 => {
            // Wholly constant — handled by caller, but be safe.
            let c = scalar.unwrap_or_else(|| pool.integer(1_i32));
            return Ok(pool.mul(vec![c, recip(s, pool)]));
        }
        1 => rest[0],
        _ => pool.mul(rest.clone()),
    };

    let transformed = laplace_product_body(inner, t, s, pool, depth)?;
    Ok(match scalar {
        Some(c) => pool.mul(vec![c, transformed]),
        None => transformed,
    })
}

/// Transform a t-dependent product with no constant scalar factor, applying the
/// structural product theorems (frequency-diff `t^n·f`, s-shift `e^{at}·f`,
/// t-shift `θ(t−a)·g(t−a)`).
fn laplace_product_body(
    body: ExprId,
    t: ExprId,
    s: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, LaplaceError> {
    let factors: Vec<ExprId> = match pool.get(body) {
        ExprData::Mul(a) => a,
        _ => vec![body],
    };

    // (1) e^{a t} · g(t)  →  G(s − a)   [s-shift theorem]
    for (i, &fac) in factors.iter().enumerate() {
        if let Some(a) = match_exp_linear(fac, t, pool) {
            let rest = remove_index(&factors, i, pool);
            let g_transform = laplace_inner(rest, t, s, pool, depth + 1)?;
            let s_minus_a = simp(pool.add(vec![s, neg(a, pool)]), pool);
            return Ok(subs_one(g_transform, s, s_minus_a, pool));
        }
    }

    // (2) θ(t − a) · g(t − a)  →  e^{−a s} G(s)   [t-shift / second shift]
    if let Some(res) = try_time_shift(&factors, t, s, pool, depth)? {
        return Ok(res);
    }

    // (3) t^n · g(t)  →  (−1)^n F^{(n)}(s)   [frequency differentiation]
    for (i, &fac) in factors.iter().enumerate() {
        if let Some(n) = match_t_power(fac, t, pool) {
            let rest = remove_index(&factors, i, pool);
            let mut g_transform = laplace_inner(rest, t, s, pool, depth + 1)?;
            for _ in 0..n {
                g_transform = crate::diff::diff(g_transform, s, pool)
                    .map_err(|_| LaplaceError::NoRule("frequency-diff failed".into()))?
                    .value;
            }
            let sign = if n % 2 == 0 {
                pool.integer(1_i32)
            } else {
                pool.integer(-1_i32)
            };
            return Ok(pool.mul(vec![sign, g_transform]));
        }
    }

    // No product theorem applied.  If `body` was not actually a product (a lone
    // factor, e.g. a bare `cos(b t)` whose t-free scalar was already peeled off
    // by `laplace_mul`), fall back to the structural table.  This cannot recurse
    // forever: `laplace_inner` only re-enters `laplace_product_body` for a `Mul`,
    // and here `body` is not a `Mul`.
    if !matches!(pool.get(body), ExprData::Mul(_)) {
        return laplace_inner(body, t, s, pool, depth + 1);
    }

    Err(LaplaceError::NoRule(pool.display(body).to_string()))
}

/// If `fac` is `e^{a·t}` (or `e^{a·t+b}` collapsed), return `a` (the linear
/// coefficient), with the additive constant folded into the coefficient via
/// `e^{b}` only when `b` is free of `t` — here we require pure `a·t`.
fn match_exp_linear(fac: ExprId, t: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let ExprData::Func { name, args } = pool.get(fac) {
        if name == "exp" && args.len() == 1 {
            let (a, b) = as_affine(args[0], t, pool)?;
            if b == pool.integer(0_i32) && a != pool.integer(0_i32) {
                return Some(a);
            }
        }
    }
    // exp written as a power e^(…) is normalised to Func("exp", …) in this CAS.
    None
}

/// If `fac` is `t^n` with `n ∈ ℤ₊`, or bare `t`, return `n`.
fn match_t_power(fac: ExprId, t: ExprId, pool: &ExprPool) -> Option<u64> {
    if fac == t {
        return Some(1);
    }
    if let ExprData::Pow { base, exp } = pool.get(fac) {
        if base == t {
            return nonneg_int_exp(exp, pool).filter(|&n| n >= 1);
        }
    }
    None
}

/// Recognise `θ(t − a) · g(t − a)` (any number of g-factors), returning
/// `e^{−a s} · L{g(t)}` when every t-dependent non-Heaviside factor is a
/// function of `(t − a)` with the *same* shift `a ≥ 0` as the Heaviside.
fn try_time_shift(
    factors: &[ExprId],
    t: ExprId,
    s: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<Option<ExprId>, LaplaceError> {
    // Find a Heaviside factor and its shift a (from θ(t − a)).
    let mut heaviside_idx = None;
    let mut shift = None;
    for (i, &fac) in factors.iter().enumerate() {
        if let ExprData::Func { name, args } = pool.get(fac) {
            if name == "heaviside" && args.len() == 1 {
                // arg = t − a  ⇒  (coeff 1)·t + (−a)
                if let Some((coeff, b)) = as_affine(args[0], t, pool) {
                    if coeff == pool.integer(1_i32) {
                        heaviside_idx = Some(i);
                        shift = Some(neg(b, pool)); // a = −b
                        break;
                    }
                }
            }
        }
    }
    let (hi, a) = match (heaviside_idx, shift) {
        (Some(hi), Some(a)) => (hi, simp(a, pool)),
        _ => return Ok(None),
    };

    // The remaining factors form g(t − a).  Substitute u = t + a (i.e. shift the
    // argument back) and transform g(u), then attach e^{−a s}.
    let rest = remove_index(factors, hi, pool);
    let exp_neg_as = pool.func("exp", vec![simp(neg(pool.mul(vec![a, s]), pool), pool)]);

    // g(t − a): replace t by (t + a) to recover g(t), require the result be a
    // genuine function of t (the standard second-shift form).  If `rest` is just
    // the constant 1 (bare shifted step), G(s) = 1/s.
    if rest == pool.integer(1_i32) {
        return Ok(Some(pool.mul(vec![exp_neg_as, recip(s, pool)])));
    }
    let t_plus_a = simp(pool.add(vec![t, a]), pool);
    let g_of_t = subs_one(rest, t, t_plus_a, pool);
    let g_transform = laplace_inner(simp(g_of_t, pool), t, s, pool, depth + 1)?;
    Ok(Some(pool.mul(vec![exp_neg_as, g_transform])))
}

/// Single-argument primitive functions: sin/cos/sinh/cosh/exp/heaviside/dirac.
fn laplace_func(
    name: &str,
    arg: ExprId,
    t: ExprId,
    s: ExprId,
    pool: &ExprPool,
    _depth: usize,
) -> Result<ExprId, LaplaceError> {
    // exp(a t) → 1/(s − a)
    if name == "exp" {
        let (a, b) = as_affine(arg, t, pool).ok_or_else(|| {
            LaplaceError::NoRule(format!("exp of non-affine argument: {}", pool.display(arg)))
        })?;
        if b != pool.integer(0_i32) {
            return Err(LaplaceError::NoRule(
                "exp(a t + b): nonzero constant offset".into(),
            ));
        }
        let denom = pool.add(vec![s, neg(a, pool)]);
        return Ok(recip(denom, pool));
    }

    // For sin/cos/sinh/cosh the argument must be a pure linear b·t.
    let trig = matches!(name, "sin" | "cos" | "sinh" | "cosh");
    if trig {
        let (b, off) = as_affine(arg, t, pool).ok_or_else(|| {
            LaplaceError::NoRule(format!(
                "{name} of non-affine argument: {}",
                pool.display(arg)
            ))
        })?;
        if off != pool.integer(0_i32) || b == pool.integer(0_i32) {
            return Err(LaplaceError::NoRule(format!(
                "{name}(b t): argument must be a nonzero multiple of t"
            )));
        }
        let b2 = pool.pow(b, pool.integer(2_i32));
        let s2 = pool.pow(s, pool.integer(2_i32));
        return Ok(match name {
            // sin(bt) = b/(s²+b²)
            "sin" => {
                let denom = pool.add(vec![s2, b2]);
                pool.mul(vec![b, recip(denom, pool)])
            }
            // cos(bt) = s/(s²+b²)
            "cos" => {
                let denom = pool.add(vec![s2, b2]);
                pool.mul(vec![s, recip(denom, pool)])
            }
            // sinh(bt) = b/(s²−b²)
            "sinh" => {
                let denom = pool.add(vec![s2, neg(b2, pool)]);
                pool.mul(vec![b, recip(denom, pool)])
            }
            // cosh(bt) = s/(s²−b²)
            "cosh" => {
                let denom = pool.add(vec![s2, neg(b2, pool)]);
                pool.mul(vec![s, recip(denom, pool)])
            }
            _ => unreachable!(),
        });
    }

    // θ(t − a) → e^{−a s}/s   (a ≥ 0 assumed; a = 0 ⇒ 1/s).
    if name == "heaviside" {
        let (coeff, b) = as_affine(arg, t, pool).ok_or_else(|| {
            LaplaceError::NoRule(format!(
                "heaviside of non-affine argument: {}",
                pool.display(arg)
            ))
        })?;
        if coeff != pool.integer(1_i32) {
            return Err(LaplaceError::NoRule(
                "heaviside(c·t − a): coefficient of t must be 1".into(),
            ));
        }
        let a = simp(neg(b, pool), pool); // a = −b
        let exp_neg_as = pool.func("exp", vec![simp(neg(pool.mul(vec![a, s]), pool), pool)]);
        return Ok(pool.mul(vec![exp_neg_as, recip(s, pool)]));
    }

    // δ(t − a) → e^{−a s}   (δ(t) ↦ 1).
    if name == "diracdelta" {
        let (coeff, b) = as_affine(arg, t, pool).ok_or_else(|| {
            LaplaceError::NoRule(format!(
                "diracdelta of non-affine argument: {}",
                pool.display(arg)
            ))
        })?;
        if coeff != pool.integer(1_i32) {
            return Err(LaplaceError::NoRule(
                "diracdelta(c·t − a): coefficient of t must be 1".into(),
            ));
        }
        let a = simp(neg(b, pool), pool);
        return Ok(pool.func("exp", vec![simp(neg(pool.mul(vec![a, s]), pool), pool)]));
    }

    Err(LaplaceError::NoRule(format!("{name}(...)")))
}

/// Substitute every occurrence of `from` with `to` in `expr`.
fn subs_one(expr: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut map = std::collections::HashMap::new();
    map.insert(from, to);
    crate::kernel::subs(expr, &map, pool)
}

/// Remove the factor at `idx` from `factors`, returning the product of the rest
/// (or `1` if none remain).
fn remove_index(factors: &[ExprId], idx: usize, pool: &ExprPool) -> ExprId {
    let rest: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != idx)
        .map(|(_, &f)| f)
        .collect();
    match rest.len() {
        0 => pool.integer(1_i32),
        1 => rest[0],
        _ => pool.mul(rest),
    }
}

// ===========================================================================
// Derivative rule (for the ODE workflow)
// ===========================================================================

/// The Laplace transform of the `order`-th derivative `f^{(order)}(t)` in terms
/// of `F = L{f}(s)` and the initial values `f(0), f'(0), …, f^{(order−1)}(0)`:
///
/// ```text
///   L{f^{(n)}} = sⁿ F − sⁿ⁻¹ f(0) − sⁿ⁻² f'(0) − … − f^{(n−1)}(0).
/// ```
///
/// `initial_values[k]` must be `f^{(k)}(0)`.  Because `f` is an *unknown*
/// function, this operates on the placeholder symbol `f_transform = F(s)` rather
/// than a concrete `f`; it is the building block consumed by an
/// (algebraic-solve) ODE-via-Laplace workflow.  Missing trailing initial values
/// default to `0`.
///
/// Returns the simplified expression for `L{f^{(order)}}(s)`.
pub fn laplace_derivative_rule(
    f_transform: ExprId,
    s: ExprId,
    order: u32,
    initial_values: &[ExprId],
    pool: &ExprPool,
) -> ExprId {
    // sⁿ F
    let s_n = pool.pow(s, pool.integer(order as i32));
    let mut terms = vec![pool.mul(vec![s_n, f_transform])];
    // − Σ_{k=0}^{n−1} s^{n−1−k} f^{(k)}(0)
    for k in 0..order {
        let f_k0 = initial_values
            .get(k as usize)
            .copied()
            .unwrap_or_else(|| pool.integer(0_i32));
        if f_k0 == pool.integer(0_i32) {
            continue;
        }
        let power = (order - 1 - k) as i32;
        let s_pow = pool.pow(s, pool.integer(power));
        terms.push(pool.mul(vec![pool.integer(-1_i32), s_pow, f_k0]));
    }
    simp(pool.add(terms), pool)
}

// ===========================================================================
// Inverse transform
// ===========================================================================

/// Compute the inverse Laplace transform `L⁻¹{F(s)}(t)` for a **proper rational**
/// `F(s)` (optionally times a delay factor `e^{−a s}`).
///
/// Strategy: factor out any `e^{−a s}` delay (→ Heaviside shift), partial-fraction
/// the rational part via [`crate::poly::apart`], then map each term through the
/// inverse table.  See the [module docs](self) for the table and caveats.
///
/// # Errors
///
/// - [`LaplaceError::SameVariable`] if `s == t`.
/// - [`LaplaceError::NotInvertible`] for non-rational `F`, an improper rational
///   (polynomial part ≠ 0 ⇒ derivatives of `δ`), or an irreducible denominator
///   factor of degree > 2.
pub fn inverse_laplace_transform(
    big_f: ExprId,
    s: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LaplaceError> {
    if s == t {
        return Err(LaplaceError::SameVariable);
    }

    // Peel a delay factor e^{−a s}: L⁻¹{e^{−a s} G(s)} = θ(t−a)·g(t−a).
    if let Some((a, g)) = split_delay(big_f, s, pool) {
        let g_inv = inverse_laplace_transform(g, s, t, pool)?;
        let t_minus_a = simp(pool.add(vec![t, neg(a, pool)]), pool);
        let shifted = subs_one(g_inv, t, t_minus_a, pool);
        let heaviside = pool.func("heaviside", vec![t_minus_a]);
        return Ok(simp(pool.mul(vec![heaviside, shifted]), pool));
    }

    // Rational route: partial fractions, then table per term.
    let pf = crate::poly::apart(big_f, s, pool)
        .map_err(|e| LaplaceError::NotInvertible(format!("apart failed: {e}")))?;

    let terms: Vec<ExprId> = match pool.get(pf) {
        ExprData::Add(args) => args,
        _ => vec![pf],
    };

    let mut out = Vec::with_capacity(terms.len());
    for term in terms {
        out.push(invert_term(term, s, t, pool)?);
    }
    Ok(simp(pool.add(out), pool))
}

/// Split `F = e^{−a s} · G(s)` returning `(a, G)`, or `None` if there is no
/// exponential delay factor.
fn split_delay(big_f: ExprId, s: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let factors: Vec<ExprId> = match pool.get(big_f) {
        ExprData::Mul(a) => a,
        _ => vec![big_f],
    };
    for (i, &fac) in factors.iter().enumerate() {
        if let ExprData::Func { name, args } = pool.get(fac) {
            if name == "exp" && args.len() == 1 {
                // arg should be −a·s (linear in s, no constant)
                if let Some((coeff, b)) = as_affine(args[0], s, pool) {
                    if b == pool.integer(0_i32) && coeff != pool.integer(0_i32) {
                        let a = simp(neg(coeff, pool), pool); // arg = −a s ⇒ a = −coeff
                        let g = remove_index(&factors, i, pool);
                        return Some((a, g));
                    }
                }
            }
        }
    }
    None
}

/// Invert a single partial-fraction term `A·(s−a)^{−n}` or a quadratic-denominator
/// term `(B s + C)/((s−p)² + ω²)`.
fn invert_term(
    term: ExprId,
    s: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LaplaceError> {
    // Split term = numer · denom_pow, where denom_pow = D(s)^{−n}.
    let (numer, base, n) = split_rational_term(term, pool)
        .ok_or_else(|| LaplaceError::NotInvertible(pool.display(term).to_string()))?;

    // A polynomial part (n == 0) is the transform of `δ(t)` (constant term) or
    // its derivatives (higher terms) — outside the rational table.  We decline
    // it rather than emit distributional `δ` output, keeping the inverse on
    // ordinary functions (matching the "proper rational" scope).
    if n == 0 {
        return Err(LaplaceError::NotInvertible(format!(
            "polynomial part {} (δ / derivatives of δ — improper rational)",
            pool.display(term)
        )));
    }

    match poly_degree(base, s, pool) {
        Some(1) => invert_linear_pole(numer, base, n, s, t, pool),
        Some(2) => invert_quadratic(numer, base, n, s, t, pool),
        _ => Err(LaplaceError::NotInvertible(format!(
            "denominator factor of degree > 2: {}",
            pool.display(base)
        ))),
    }
}

/// Decompose a term into `(numerator, denom_base, n)` with `term = numerator ·
/// denom_base^{−n}` and `n ≥ 0`, `numerator` free of any negative power of `s`.
fn split_rational_term(term: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, u64)> {
    let factors: Vec<ExprId> = match pool.get(term) {
        ExprData::Mul(a) => a,
        _ => vec![term],
    };
    let mut numer_parts: Vec<ExprId> = Vec::new();
    let mut base: Option<ExprId> = None;
    let mut n: u64 = 0;

    for &fac in &factors {
        if let ExprData::Pow { base: b, exp } = pool.get(fac) {
            if let ExprData::Integer(e) = pool.get(exp) {
                let ev = e.0;
                if ev < 0 {
                    // negative power → part of denominator
                    if base.is_some() && base != Some(b) {
                        // Two distinct denominator factors — not a single PF term.
                        return None;
                    }
                    base = Some(b);
                    n = (-ev).to_u64()?;
                    continue;
                }
            }
        }
        numer_parts.push(fac);
    }

    let numer = match numer_parts.len() {
        0 => pool.integer(1_i32),
        1 => numer_parts[0],
        _ => pool.mul(numer_parts),
    };
    match base {
        Some(b) => Some((numer, b, n)),
        None => Some((numer, pool.integer(1_i32), 0)),
    }
}

/// Degree of `base` as a polynomial in `s` (only handles `s`, `s ± c`, and
/// `s² + …` forms via structural inspection).  Returns `None` if not obviously
/// polynomial of degree 1 or 2.
fn poly_degree(base: ExprId, s: ExprId, pool: &ExprPool) -> Option<u64> {
    if base == s {
        return Some(1);
    }
    match pool.get(base) {
        ExprData::Add(args) => {
            let mut deg = 0u64;
            for a in args {
                deg = deg.max(monomial_degree(a, s, pool)?);
            }
            Some(deg)
        }
        ExprData::Pow { .. } | ExprData::Mul(_) => monomial_degree(base, s, pool),
        _ if is_free_of(base, s, pool) => Some(0),
        _ => None,
    }
}

fn monomial_degree(term: ExprId, s: ExprId, pool: &ExprPool) -> Option<u64> {
    if term == s {
        return Some(1);
    }
    if is_free_of(term, s, pool) {
        return Some(0);
    }
    match pool.get(term) {
        ExprData::Pow { base, exp } if base == s => nonneg_int_exp(exp, pool),
        ExprData::Mul(args) => {
            let mut deg = 0u64;
            for a in args {
                deg += monomial_degree(a, s, pool)?;
            }
            Some(deg)
        }
        _ => None,
    }
}

/// `L⁻¹{A/(s−a)^n} = A·t^{n−1} e^{a t}/(n−1)!`.
fn invert_linear_pole(
    numer: ExprId,
    base: ExprId,
    n: u64,
    s: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LaplaceError> {
    // numer must be free of s (deg < deg(base) = 1).
    if !is_free_of(numer, s, pool) {
        return Err(LaplaceError::NotInvertible(format!(
            "linear-pole numerator depends on s: {}",
            pool.display(numer)
        )));
    }
    // base = s − a (monic).  Extract a from (coeff·s + b): a = −b/coeff, require coeff = 1.
    let (coeff, b) = as_affine(base, s, pool)
        .ok_or_else(|| LaplaceError::NotInvertible(pool.display(base).to_string()))?;
    if coeff != pool.integer(1_i32) {
        return Err(LaplaceError::NotInvertible(
            "non-monic linear denominator".into(),
        ));
    }
    let a = simp(neg(b, pool), pool); // a = −b

    let exp_at = pool.func("exp", vec![pool.mul(vec![a, t])]);
    let mut parts = vec![numer, exp_at];
    if n >= 2 {
        let t_pow = pool.pow(t, pool.integer(Integer::from(n - 1)));
        parts.push(t_pow);
        let fact = factorial(n - 1, pool);
        parts.push(recip(fact, pool));
    }
    Ok(pool.mul(parts))
}

/// Invert `(B s + C)/((s−p)² + ω²)^n` for `n ∈ {1, 2}` into damped sin/cos.
///
/// ```text
///   n = 1:  e^{p t} ( B cos(ω t) + ((C + B p)/ω) sin(ω t) )
///
///   n = 2:  write B s + C = B(s−p) + (B p + C); then
///           L⁻¹{(s−p)/den²} = (t/(2ω)) e^{p t} sin(ω t)
///           L⁻¹{1/den²}     = e^{p t}/(2ω³) (sin(ω t) − ω t cos(ω t))
/// ```
///
/// Higher powers (`n ≥ 3`) are declined.  The `n = 2` case is required for
/// round-trips of `t·sin` / `t·cos` (frequency differentiation of the
/// quadratic table entries).
fn invert_quadratic(
    numer: ExprId,
    base: ExprId,
    n: u64,
    s: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LaplaceError> {
    if n != 1 && n != 2 {
        return Err(LaplaceError::NotInvertible(
            "repeated irreducible quadratic pole (n ≥ 3) not in table".into(),
        ));
    }
    // Write base = s² + β s + γ. Complete the square: (s + β/2)² + (γ − β²/4).
    let (alpha, beta, gamma) = quadratic_coeffs(base, s, pool)
        .ok_or_else(|| LaplaceError::NotInvertible(pool.display(base).to_string()))?;
    if alpha != pool.integer(1_i32) {
        return Err(LaplaceError::NotInvertible(
            "non-monic quadratic denominator".into(),
        ));
    }
    // p = −β/2 ; ω² = γ − β²/4 ; ω = sqrt(ω²).
    let half = pool.rational(1_i32, 2_i32);
    let p = simp(pool.mul(vec![neg(beta, pool), half]), pool);
    let beta2 = pool.pow(beta, pool.integer(2_i32));
    let quarter = pool.rational(1_i32, 4_i32);
    let omega_sq = simp(
        pool.add(vec![gamma, neg(pool.mul(vec![beta2, quarter]), pool)]),
        pool,
    );
    let omega = simp(pool.pow(omega_sq, half), pool);

    // numerator B s + C.
    let (bb, cc) = as_affine(numer, s, pool)
        .ok_or_else(|| LaplaceError::NotInvertible(pool.display(numer).to_string()))?;

    let exp_pt = pool.func("exp", vec![pool.mul(vec![p, t])]);
    let omega_t = pool.mul(vec![omega, t]);
    let sin_wt = pool.func("sin", vec![omega_t]);
    let cos_wt = pool.func("cos", vec![omega_t]);

    if n == 1 {
        // e^{p t} [ B cos(ω t) + ((C + B p)/ω) sin(ω t) ]
        let cos_term = pool.mul(vec![bb, cos_wt]);
        let bp = pool.mul(vec![bb, p]);
        let sin_coeff = pool.mul(vec![pool.add(vec![cc, bp]), recip(omega, pool)]);
        let sin_term = pool.mul(vec![sin_coeff, sin_wt]);
        return Ok(pool.mul(vec![exp_pt, pool.add(vec![cos_term, sin_term])]));
    }

    // n = 2: B·(t/(2ω)) e^{pt} sin(ωt)
    //      + (Bp+C)·e^{pt}/(2ω³)·(sin(ωt) − ωt cos(ωt)).
    let two = pool.integer(2_i32);
    let two_omega = pool.mul(vec![two, omega]);
    let bp_plus_c = pool.add(vec![pool.mul(vec![bb, p]), cc]);

    let t_sin = pool.mul(vec![bb, t, recip(two_omega, pool), sin_wt]);
    let sin_minus_wt_cos = pool.add(vec![sin_wt, neg(pool.mul(vec![omega, t, cos_wt]), pool)]);
    let omega3 = pool.mul(vec![omega, omega, omega]);
    let two_omega3 = pool.mul(vec![two, omega3]);
    let second = pool.mul(vec![bp_plus_c, recip(two_omega3, pool), sin_minus_wt_cos]);
    Ok(pool.mul(vec![exp_pt, pool.add(vec![t_sin, second])]))
}

/// Coefficients `(α, β, γ)` of a monic-or-scaled quadratic `α s² + β s + γ`.
fn quadratic_coeffs(base: ExprId, s: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId)> {
    let args: Vec<ExprId> = match pool.get(base) {
        ExprData::Add(a) => a,
        _ => vec![base],
    };
    let mut alpha = pool.integer(0_i32);
    let mut beta = pool.integer(0_i32);
    let mut gamma_parts: Vec<ExprId> = Vec::new();
    for term in args {
        match monomial_degree(term, s, pool)? {
            2 => alpha = monomial_coeff(term, s, 2, pool)?,
            1 => beta = monomial_coeff(term, s, 1, pool)?,
            0 => gamma_parts.push(term),
            _ => return None,
        }
    }
    let gamma = match gamma_parts.len() {
        0 => pool.integer(0_i32),
        1 => gamma_parts[0],
        _ => pool.add(gamma_parts),
    };
    Some((alpha, beta, gamma))
}

/// Coefficient of `s^deg` in a single monomial `c·s^deg`.
fn monomial_coeff(term: ExprId, s: ExprId, deg: u64, pool: &ExprPool) -> Option<ExprId> {
    if deg == 0 {
        return Some(term);
    }
    if deg == 1 && term == s {
        return Some(pool.integer(1_i32));
    }
    if let ExprData::Pow { base, exp } = pool.get(term) {
        if base == s && nonneg_int_exp(exp, pool) == Some(deg) {
            return Some(pool.integer(1_i32));
        }
    }
    if let ExprData::Mul(args) = pool.get(term) {
        let mut coeff_parts: Vec<ExprId> = Vec::new();
        let mut found = false;
        for a in args {
            if a == s && deg == 1 {
                found = true;
                continue;
            }
            if let ExprData::Pow { base, exp } = pool.get(a) {
                if base == s && nonneg_int_exp(exp, pool) == Some(deg) {
                    found = true;
                    continue;
                }
            }
            coeff_parts.push(a);
        }
        if found {
            return Some(match coeff_parts.len() {
                0 => pool.integer(1_i32),
                1 => coeff_parts[0],
                _ => pool.mul(coeff_parts),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests;
