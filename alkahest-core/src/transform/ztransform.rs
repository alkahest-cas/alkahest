//! Symbolic (unilateral) Z-transform `Z{a[n]}(z)` and inverse `Z⁻¹{A(z)}(n)`.
//!
//! # Forward transform
//!
//! [`z_transform`] computes the *unilateral* Z-transform
//!
//! ```text
//!   Z{a[n]}(z) = Σ_{n≥0} a[n] z^{−n}.
//! ```
//!
//! It is a rule/table-based structural recursion over `a[n]` (mirroring
//! [`crate::transform::laplace`]):
//!
//! | `a[n]`                 | `Z{a}(z)`                              | rule              |
//! |------------------------|----------------------------------------|-------------------|
//! | `c` (const)            | `c·z/(z−1)`                             | constant          |
//! | `n`                    | `z/(z−1)²`                              | ramp              |
//! | `n²`                   | `z(z+1)/(z−1)³`                         | quadratic ramp    |
//! | `aⁿ`                   | `z/(z−a)`                               | geometric         |
//! | `n·aⁿ`                 | `a z/(z−a)²`                            | scaled-diff geom. |
//! | `sin(ω n)`             | `z sin(ω) / (z² − 2 z cos(ω) + 1)`      | sine              |
//! | `cos(ω n)`             | `z(z − cos(ω)) / (z² − 2 z cos(ω) + 1)` | cosine            |
//! | `α·a[n] + β·b[n]`      | `α A(z) + β B(z)`                       | linearity         |
//! | `aⁿ·x[n]`              | `X(z/a)`                                | scaling theorem   |
//! | `n·x[n]`               | `−z·dX/dz`                              | differentiation   |
//!
//! The unilateral shift theorems are exposed separately (they operate on the
//! *symbol* `X = Z{x}` plus initial values, since `x` itself is an unknown
//! sequence — exactly as [`crate::transform::laplace::laplace_derivative_rule`]
//! does for the derivative rule):
//!
//! - [`z_shift_delay`]: `x[n−k] ↦ z^{−k} X(z)` (zero initial conditions assumed
//!   for the "missing" samples `x[−1], …, x[−k]`).
//! - [`z_shift_advance`]: the *unilateral* advance
//!   `x[n+1] ↦ z·X(z) − z·x[0]`, needed to translate difference equations
//!   `a[n+1] = a[n] + a[n−1]` (etc.) into algebraic equations in `Z{a}`.
//!
//! # Inverse transform
//!
//! [`inverse_z_transform`] inverts a **rational** `X(z)` by writing
//! `X(z)/z` in partial fractions (via [`crate::poly::apart`]), multiplying each
//! term back by `z`, and mapping the resulting `z/(z−a)^k` shapes through the
//! inverse table.  Before the rational path, it also matches the **forward
//! sinusoid table** forms
//!
//! ```text
//!   z·sin(ω)/(z² − 2z·cos(ω) + 1)      ↦  sin(ω n)
//!   z(z − cos(ω))/(z² − 2z·cos(ω) + 1) ↦  cos(ω n)
//! ```
//!
//! (including a constant amplitude factor).  Those expressions have
//! transcendental coefficients, so `apart` (which requires ℚ-coefficients)
//! cannot see them as rational functions — the direct table match is what
//! makes `Z⁻¹{Z{sin(ω n)}}` and `Z⁻¹{Z{cos(ω n)}}` round-trip.
//!
//! Rational inverse table:
//!
//! | term in `X(z)`           | `Z⁻¹` term (`n ≥ 0`)                  |
//! |---------------------------|----------------------------------------|
//! | `A·z/(z−a)`               | `A·aⁿ`                                  |
//! | `A·z/(z−a)²`              | `A·n·aⁿ⁻¹`  (rewritten as `(A/a)·n·aⁿ`) |
//! | `A·z/(z−1)`               | `A` (constant)                          |
//! | `(P z² + Q z)/(z² + b z + c)` (`b² − 4c < 0`) | `rⁿ(A cos θn + B sin θn)`, **real** |
//!
//! The last row covers an **irreducible quadratic** denominator — a
//! complex-conjugate pole pair `r e^{±iθ}` with `r = √c`, `θ = acos(−b/2√c)` —
//! and emits the **real** damped sinusoid (no imaginary unit in the output;
//! the `i² = −1` collapse happens inside the derivation, not the result).  For
//! example `X(z) = z/(z² − z + 1)` inverts to `(2/√3)·sin(π n / 3)`, which the
//! forward table round-trips back to `z/(z² − z + 1)`.
//!
//! Higher-order repeated poles `(z−a)^k`, `k ≥ 3`, *repeated* complex poles
//! (`k ≥ 2`), and quadratic denominators with **non-negative discriminant**
//! (real, possibly surd, roots — e.g. the Fibonacci denominator `z² − z − 1`,
//! discriminant `5`) remain declined (outside the table — see
//! [`ZTransformError`]).  Such surd-root cases factor only over an algebraic
//! extension and have no rational-coefficient closed form here.
//!
//! # Caveats
//!
//! Both directions are **formal**: this is the *unilateral* transform with no
//! region-of-convergence tracked, matching [`crate::transform::laplace`]'s
//! `noconds=True`-style convention. Unrecognised forms return
//! [`ZTransformError::NoRule`] (forward) / [`ZTransformError::NotInvertible`]
//! (inverse) rather than guessing.
//!
//! ## Declined table entries
//!
//! The planning document additionally lists `binomial(n+k−1, k−1)·aⁿ` (negative
//! binomial / generalized geometric series) and the Kronecker delta
//! `δ[n−k] ↦ z^{−k}`. Alkahest has no `binomial(·,·)` or discrete-delta
//! expression primitive, so both are **out of scope** for the
//! expression-pattern table here: there is nothing in the kernel's expression
//! algebra that would match `δ[n−k]` (it is not `DiracDelta`, which is the
//! *continuous* impulse used by [`crate::transform::laplace`], and a discrete
//! Kronecker delta is a different object). Adding either would require a new
//! primitive (out of scope for an additive, non-primitive-registry change).

use crate::kernel::{ExprData, ExprId, ExprPool};

/// Errors from the Z-transform routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZTransformError {
    /// No forward rule matched `a[n]` (E-TRANSFORM-101).
    NoRule(String),
    /// The inverse-transform input is not a form the table can invert
    /// (E-TRANSFORM-102).
    NotInvertible(String),
    /// The frequency variable `z` and discrete-index variable `n` must be
    /// distinct symbols (E-TRANSFORM-103).
    SameVariable,
}

impl std::fmt::Display for ZTransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZTransformError::NoRule(m) => {
                write!(f, "z_transform: no rule for {m} [E-TRANSFORM-101]")
            }
            ZTransformError::NotInvertible(m) => write!(
                f,
                "inverse_z_transform: cannot invert {m} [E-TRANSFORM-102]"
            ),
            ZTransformError::SameVariable => write!(
                f,
                "z_transform: index and frequency variables must differ [E-TRANSFORM-103]"
            ),
        }
    }
}

impl std::error::Error for ZTransformError {}

// ===========================================================================
// Small helpers (mirroring transform::laplace)
// ===========================================================================

fn is_free_of(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    crate::integrate::risch::poly_rde::is_free_of_var(expr, var, pool)
}

/// Extract `(a, b)` from `a·var + b` with `a, b` free of `var`. `None` if not
/// affine in `var`.
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

fn simp(expr: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::simplify(expr, pool).value
}

fn neg(expr: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), expr])
}

fn recip(expr: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(expr, pool.integer(-1_i32))
}

/// Substitute every occurrence of `from` with `to` in `expr`.
fn subs_one(expr: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut map = std::collections::HashMap::new();
    map.insert(from, to);
    crate::kernel::subs(expr, &map, pool)
}

/// Remove the factor at `idx` from `factors`, returning the product of the
/// rest (or `1` if none remain).
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

// ===========================================================================
// Forward transform
// ===========================================================================

const MAX_DEPTH: usize = 32;

/// Compute the unilateral Z-transform `Z{a[n]}(z) = Σ_{n≥0} a[n] z^{−n}`.
///
/// `n` is the discrete-index variable, `z` the transform variable; both must
/// be distinct symbols. This is a *formal* transform — see the
/// [module docs](self) for the rule table, caveats, and declines.
///
/// # Errors
///
/// - [`ZTransformError::SameVariable`] if `n == z`.
/// - [`ZTransformError::NoRule`] if no table rule matches `a[n]`.
///
/// # Examples
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::simplify::simplify;
/// use alkahest_cas::transform::z_transform;
///
/// let pool = ExprPool::new();
/// let n = pool.symbol("n", Domain::Real);
/// let z = pool.symbol("z", Domain::Real);
/// // Z{1}(z) = z/(z-1)
/// let one = pool.integer(1_i32);
/// let big_x = z_transform(one, n, z, &pool).unwrap();
/// let expected = pool.mul(vec![
///     z,
///     pool.pow(pool.add(vec![z, pool.integer(-1_i32)]), pool.integer(-1_i32)),
/// ]);
/// assert_eq!(
///     pool.display(big_x).to_string(),
///     pool.display(simplify(expected, &pool).value).to_string()
/// );
/// ```
pub fn z_transform(
    a: ExprId,
    n: ExprId,
    z: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, ZTransformError> {
    if n == z {
        return Err(ZTransformError::SameVariable);
    }
    let out = z_inner(a, n, z, pool, 0)?;
    Ok(simp(out, pool))
}

fn z_inner(
    a: ExprId,
    n: ExprId,
    z: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, ZTransformError> {
    if depth > MAX_DEPTH {
        return Err(ZTransformError::NoRule("recursion depth exceeded".into()));
    }

    // Constant (free of n): Z{c} = c·z/(z−1).
    if is_free_of(a, n, pool) {
        return Ok(pool.mul(vec![a, geometric_transform(pool.integer(1_i32), z, pool)]));
    }

    // Bare n: Z{n} = z/(z−1)².
    if a == n {
        return Ok(ramp_transform(z, pool));
    }

    match pool.get(a) {
        // Linearity over sums.
        ExprData::Add(args) => {
            let mut terms = Vec::with_capacity(args.len());
            for arg in args {
                terms.push(z_inner(arg, n, z, pool, depth + 1)?);
            }
            Ok(pool.add(terms))
        }

        // Products: split off the n-free scalar (linearity), then dispatch the
        // remaining n-dependent factor through the structural product rules.
        ExprData::Mul(args) => z_mul(&args, n, z, pool, depth),

        // n^2 (other integer powers of n are not in the table).
        ExprData::Pow { base, exp } if base == n => {
            if nonneg_int_exp(exp, pool) == Some(2) {
                Ok(quadratic_ramp_transform(z, pool))
            } else {
                Err(ZTransformError::NoRule(format!(
                    "n^e (only n and n^2 are tabulated): {}",
                    pool.display(a)
                )))
            }
        }

        // a^n.
        ExprData::Pow { base, exp } if exp == n => {
            if is_free_of(base, n, pool) {
                Ok(geometric_transform(base, z, pool))
            } else {
                Err(ZTransformError::NoRule(format!(
                    "base^n with base depending on n: {}",
                    pool.display(a)
                )))
            }
        }

        ExprData::Func { name, args } if args.len() == 1 => z_func(&name, args[0], n, z, pool),

        _ => Err(ZTransformError::NoRule(pool.display(a).to_string())),
    }
}

/// `Z{a^n}(z) = z/(z − a)`.
fn geometric_transform(base: ExprId, z: ExprId, pool: &ExprPool) -> ExprId {
    let denom = pool.add(vec![z, neg(base, pool)]);
    pool.mul(vec![z, recip(denom, pool)])
}

/// `Z{n}(z) = z/(z − 1)²`.
fn ramp_transform(z: ExprId, pool: &ExprPool) -> ExprId {
    let denom = pool.pow(pool.add(vec![z, pool.integer(-1_i32)]), pool.integer(2_i32));
    pool.mul(vec![z, recip(denom, pool)])
}

/// `Z{n²}(z) = z(z + 1) / (z − 1)³`.
fn quadratic_ramp_transform(z: ExprId, pool: &ExprPool) -> ExprId {
    let numer = pool.mul(vec![z, pool.add(vec![z, pool.integer(1_i32)])]);
    let denom = pool.pow(pool.add(vec![z, pool.integer(-1_i32)]), pool.integer(3_i32));
    pool.mul(vec![numer, recip(denom, pool)])
}

/// Z-transform of a product `∏ args`.
fn z_mul(
    args: &[ExprId],
    n: ExprId,
    z: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, ZTransformError> {
    // Pull out the constant (n-free) scalar prefactor.
    let (consts, rest): (Vec<ExprId>, Vec<ExprId>) =
        args.iter().partition(|&&a| is_free_of(a, n, pool));
    let scalar = match consts.len() {
        0 => None,
        1 => Some(consts[0]),
        _ => Some(pool.mul(consts.clone())),
    };

    let inner = match rest.len() {
        0 => {
            let c = scalar.unwrap_or_else(|| pool.integer(1_i32));
            return Ok(pool.mul(vec![c, geometric_transform(pool.integer(1_i32), z, pool)]));
        }
        1 => rest[0],
        _ => pool.mul(rest.clone()),
    };

    let transformed = z_product_body(inner, n, z, pool, depth)?;
    Ok(match scalar {
        Some(c) => pool.mul(vec![c, transformed]),
        None => transformed,
    })
}

/// Transform an n-dependent product with no constant scalar factor, applying
/// the structural product theorems (scaling `aⁿ·x[n]`, differentiation
/// `n·x[n]`).
fn z_product_body(
    body: ExprId,
    n: ExprId,
    z: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, ZTransformError> {
    let factors: Vec<ExprId> = match pool.get(body) {
        ExprData::Mul(a) => a,
        _ => vec![body],
    };

    // (1) aⁿ · x[n]  →  X(z/a)   [scaling theorem]
    for (i, &fac) in factors.iter().enumerate() {
        if let Some(a) = match_geometric(fac, n, pool) {
            let rest = remove_index(&factors, i, pool);
            let x_transform = z_inner(rest, n, z, pool, depth + 1)?;
            let z_over_a = simp(pool.mul(vec![z, recip(a, pool)]), pool);
            return Ok(subs_one(x_transform, z, z_over_a, pool));
        }
    }

    // (2) n · x[n]  →  −z · dX/dz   [differentiation theorem]
    for (i, &fac) in factors.iter().enumerate() {
        if let Some(k) = match_n_power(fac, n, pool) {
            let rest = remove_index(&factors, i, pool);
            let mut x_transform = z_inner(rest, n, z, pool, depth + 1)?;
            for _ in 0..k {
                let dxdz = crate::diff::diff(x_transform, z, pool)
                    .map_err(|_| ZTransformError::NoRule("differentiation theorem failed".into()))?
                    .value;
                x_transform = simp(pool.mul(vec![pool.integer(-1_i32), z, dxdz]), pool);
            }
            return Ok(x_transform);
        }
    }

    // No product theorem applied. If `body` was not actually a product (a lone
    // factor, e.g. a bare `cos(ω n)` whose n-free scalar was already peeled off
    // by `z_mul`), fall back to the structural table. This cannot recurse
    // forever: `z_inner` only re-enters `z_product_body` for a `Mul`, and here
    // `body` is not a `Mul`.
    if !matches!(pool.get(body), ExprData::Mul(_)) {
        return z_inner(body, n, z, pool, depth + 1);
    }

    Err(ZTransformError::NoRule(pool.display(body).to_string()))
}

/// If `fac` is `aⁿ` (with `a` free of `n`, `a` not `±1`/trivial), return `a`.
fn match_geometric(fac: ExprId, n: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let ExprData::Pow { base, exp } = pool.get(fac) {
        if exp == n && is_free_of(base, n, pool) {
            return Some(base);
        }
    }
    None
}

/// If `fac` is `n^k` with `k ∈ ℤ₊`, or bare `n`, return `k`.
fn match_n_power(fac: ExprId, n: ExprId, pool: &ExprPool) -> Option<u64> {
    if fac == n {
        return Some(1);
    }
    if let ExprData::Pow { base, exp } = pool.get(fac) {
        if base == n {
            return nonneg_int_exp(exp, pool).filter(|&k| k >= 1);
        }
    }
    None
}

/// Single-argument primitive functions: sin/cos.
fn z_func(
    name: &str,
    arg: ExprId,
    n: ExprId,
    z: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, ZTransformError> {
    if matches!(name, "sin" | "cos") {
        let (omega, off) = as_affine(arg, n, pool).ok_or_else(|| {
            ZTransformError::NoRule(format!(
                "{name} of non-affine argument: {}",
                pool.display(arg)
            ))
        })?;
        if off != pool.integer(0_i32) || omega == pool.integer(0_i32) {
            return Err(ZTransformError::NoRule(format!(
                "{name}(ω n): argument must be a nonzero multiple of n"
            )));
        }
        let cos_w = pool.func("cos", vec![omega]);
        let sin_w = pool.func("sin", vec![omega]);
        let z2 = pool.pow(z, pool.integer(2_i32));
        let two_z_cos = pool.mul(vec![pool.integer(2_i32), z, cos_w]);
        // z² − 2z·cos(ω) + 1
        let denom = pool.add(vec![z2, neg(two_z_cos, pool), pool.integer(1_i32)]);
        return Ok(match name {
            // sin(ωn) ↦ z·sin(ω) / (z² − 2z·cos(ω) + 1)
            "sin" => {
                let numer = pool.mul(vec![z, sin_w]);
                pool.mul(vec![numer, recip(denom, pool)])
            }
            // cos(ωn) ↦ z(z − cos(ω)) / (z² − 2z·cos(ω) + 1)
            "cos" => {
                let z_minus_cos = pool.add(vec![z, neg(cos_w, pool)]);
                let numer = pool.mul(vec![z, z_minus_cos]);
                pool.mul(vec![numer, recip(denom, pool)])
            }
            _ => unreachable!(),
        });
    }

    Err(ZTransformError::NoRule(format!("{name}(...)")))
}

// ===========================================================================
// Shift theorems (for the difference-equation workflow)
// ===========================================================================

/// The Z-transform of the **delay** `x[n − k]` (`k ≥ 1`, zero initial
/// conditions for `x[−1], …, x[−k]`) in terms of `X = Z{x}(z)`:
///
/// ```text
///   Z{x[n − k]}(z) = z^{−k} X(z).
/// ```
///
/// Because `x` is an *unknown* sequence, this operates on the placeholder
/// `x_transform = X(z)` rather than a concrete `x[n]`. Mirrors
/// [`crate::transform::laplace::laplace_derivative_rule`] for the ODE
/// workflow.
pub fn z_shift_delay(x_transform: ExprId, z: ExprId, k: u32, pool: &ExprPool) -> ExprId {
    if k == 0 {
        return x_transform;
    }
    let z_neg_k = pool.pow(z, pool.integer(-(k as i64)));
    simp(pool.mul(vec![z_neg_k, x_transform]), pool)
}

/// The Z-transform of the unilateral **advance** `x[n + 1]` in terms of
/// `X = Z{x}(z)` and the initial value `x[0]`:
///
/// ```text
///   Z{x[n + 1]}(z) = z·X(z) − z·x[0].
/// ```
///
/// More generally, the `order`-th advance `x[n + order]` is obtained by
/// repeated application of this rule:
///
/// ```text
///   Z{x[n + m]}(z) = z^m X(z) − Σ_{k=0}^{m−1} z^{m−k} x[k].
/// ```
///
/// `initial_values[k]` must be `x[k]` for `k = 0, …, order − 1`. Missing
/// trailing initial values default to `0`.
pub fn z_shift_advance(
    x_transform: ExprId,
    z: ExprId,
    order: u32,
    initial_values: &[ExprId],
    pool: &ExprPool,
) -> ExprId {
    // z^order X(z)
    let z_m = pool.pow(z, pool.integer(order as i64));
    let mut terms = vec![pool.mul(vec![z_m, x_transform])];
    // − Σ_{k=0}^{order−1} z^{order−k} x[k]
    for k in 0..order {
        let xk = initial_values
            .get(k as usize)
            .copied()
            .unwrap_or_else(|| pool.integer(0_i32));
        if xk == pool.integer(0_i32) {
            continue;
        }
        let power = (order - k) as i64;
        let z_pow = pool.pow(z, pool.integer(power));
        terms.push(pool.mul(vec![pool.integer(-1_i32), z_pow, xk]));
    }
    simp(pool.add(terms), pool)
}

// ===========================================================================
// Inverse transform
// ===========================================================================

/// Compute the inverse Z-transform `Z⁻¹{X(z)}(n)` for a **rational** `X(z)`.
///
/// Strategy: write `X(z)/z` in partial fractions (via [`crate::poly::apart`]),
/// multiply each term back by `z` (giving `z/(z−a)^k`-shaped terms), then map
/// each through the inverse table. See the [module docs](self) for the table
/// and caveats.
///
/// # Errors
///
/// - [`ZTransformError::SameVariable`] if `z == n`.
/// - [`ZTransformError::NotInvertible`] for non-rational `X`, or a
///   denominator factor outside the linear-pole table (repeated pole order
///   `≥ 3`, or irreducible quadratic).
pub fn inverse_z_transform(
    big_x: ExprId,
    z: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, ZTransformError> {
    if z == n {
        return Err(ZTransformError::SameVariable);
    }

    // Forward sinusoid table (transcendental coeffs — `apart` cannot see these).
    if let Some(seq) = try_match_sinusoid_table(big_x, z, n, pool) {
        return Ok(seq);
    }

    // X(z)/z, partial-fractioned in z.
    let x_over_z = simp(pool.mul(vec![big_x, recip(z, pool)]), pool);
    let pf = crate::poly::apart(x_over_z, z, pool)
        .map_err(|e| ZTransformError::NotInvertible(format!("apart failed: {e}")))?;

    let pf_terms: Vec<ExprId> = match pool.get(pf) {
        ExprData::Add(args) => args,
        _ => vec![pf],
    };

    let mut out = Vec::with_capacity(pf_terms.len());
    for term in pf_terms {
        // Multiply this X(z)/z term back by z.
        let term_z = simp(pool.mul(vec![term, z]), pool);
        out.push(invert_term(term_z, z, n, pool)?);
    }
    Ok(simp(pool.add(out), pool))
}

/// Match `X(z)` against the forward sin/cos table (unit-circle poles with
/// transcendental coefficients that `apart` rejects as non-rational).
///
/// Recognises `A · z · sin(ω) / (z² − 2 z cos(ω) + 1)` and
/// `A · z (z − cos(ω)) / (z² − 2 z cos(ω) + 1)` (any constant amplitude `A`
/// free of `z`), returning `A·sin(ω n)` / `A·cos(ω n)`.
fn try_match_sinusoid_table(
    big_x: ExprId,
    z: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let (numer, denom) = split_fraction(big_x, pool)?;
    let omega = match_unit_circle_denom(denom, z, pool)?;

    let omega_n = pool.mul(vec![omega, n]);
    if let Some(amp) = match_sin_table_numer(numer, z, omega, pool) {
        let sin_term = pool.func("sin", vec![omega_n]);
        return Some(match amp == pool.integer(1_i32) {
            true => sin_term,
            false => pool.mul(vec![amp, sin_term]),
        });
    }
    if let Some(amp) = match_cos_table_numer(numer, z, omega, pool) {
        let cos_term = pool.func("cos", vec![omega_n]);
        return Some(match amp == pool.integer(1_i32) {
            true => cos_term,
            false => pool.mul(vec![amp, cos_term]),
        });
    }
    None
}

/// Split `numer · denom^{−1}` (possibly flattened as a Mul of factors).
fn split_fraction(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let factors: Vec<ExprId> = match pool.get(expr) {
        ExprData::Mul(a) => a,
        ExprData::Pow { base, exp } if is_neg_one(exp, pool) => {
            return Some((pool.integer(1_i32), base));
        }
        _ => return None,
    };
    let mut denoms = Vec::new();
    let mut numers = Vec::new();
    for &fac in &factors {
        match pool.get(fac) {
            ExprData::Pow { base, exp } if is_neg_one(exp, pool) => denoms.push(base),
            _ => numers.push(fac),
        }
    }
    if denoms.len() != 1 {
        return None;
    }
    let numer = match numers.len() {
        0 => pool.integer(1_i32),
        1 => numers[0],
        _ => pool.mul(numers),
    };
    Some((numer, denoms[0]))
}

fn is_neg_one(exp: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(exp), ExprData::Integer(n) if n.0 == -1)
}

/// Recognise monic `z² − 2 z·cos(ω) + 1`, returning `ω`.
fn match_unit_circle_denom(denom: ExprId, z: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (b, c) = monic_quadratic_coeffs(denom, z, pool)?;
    if c != pool.integer(1_i32) {
        return None;
    }
    // cos(ω) = −b/2
    let cos_omega = simp(
        pool.mul(vec![neg(b, pool), pool.rational(1_i32, 2_i32)]),
        pool,
    );
    match pool.get(cos_omega) {
        ExprData::Func { name, args } if name == "cos" && args.len() == 1 => Some(args[0]),
        _ => None,
    }
}

/// Numer of the form `A · z · sin(ω)` with `A` free of `z`.
fn match_sin_table_numer(
    numer: ExprId,
    z: ExprId,
    omega: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let sin_w = pool.func("sin", vec![omega]);
    peel_factors(numer, &[z, sin_w], z, pool)
}

/// Numer of the form `A · z · (z − cos(ω))` with `A` free of `z`.
fn match_cos_table_numer(
    numer: ExprId,
    z: ExprId,
    omega: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let cos_w = pool.func("cos", vec![omega]);
    let z_minus_cos = simp(pool.add(vec![z, neg(cos_w, pool)]), pool);
    // Prefer the factored shape `z · (z − cos(ω))`.
    if let Some(amp) = peel_factors(numer, &[z, z_minus_cos], z, pool) {
        return Some(amp);
    }
    // Expanded `z² − cos(ω)·z` (= P z² + Q z with P = A, Q = −A cos(ω)).
    let (p_coeff, q_coeff) = quadratic_numer_pq(numer, z, pool)?;
    let want_q = simp(pool.mul(vec![neg(p_coeff, pool), cos_w]), pool);
    if simp(q_coeff, pool) == want_q {
        Some(p_coeff)
    } else {
        None
    }
}

/// Peel each of `needles` (by `ExprId` equality after a best-effort simplify of
/// the needle) out of a Mul, returning the product of leftover factors — which
/// must be free of `z`.  Returns `None` if any needle is missing.
fn peel_factors(numer: ExprId, needles: &[ExprId], z: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let mut factors: Vec<ExprId> = match pool.get(numer) {
        ExprData::Mul(a) => a,
        _ => vec![numer],
    };
    for &needle in needles {
        let needle = simp(needle, pool);
        let pos = factors.iter().position(|&f| simp(f, pool) == needle)?;
        factors.remove(pos);
    }
    let amp = match factors.len() {
        0 => pool.integer(1_i32),
        1 => factors[0],
        _ => pool.mul(factors),
    };
    if is_free_of(amp, z, pool) {
        Some(amp)
    } else {
        None
    }
}

/// Invert a single term `A·z^p·(z−a)^{−k}` (after re-multiplying the
/// `apart(X(z)/z)` term by `z`); the table only covers `p == 1`.
fn invert_term(
    term: ExprId,
    z: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, ZTransformError> {
    let (numer, base, k) = split_rational_term(term, pool)
        .ok_or_else(|| ZTransformError::NotInvertible(pool.display(term).to_string()))?;

    // A constant term (k == 0): Z⁻¹{c} would be `c·δ[n]`, which has no
    // expression-level representation here (see module docs on the
    // Kronecker delta) — decline rather than fabricate.
    if k == 0 {
        return Err(ZTransformError::NotInvertible(format!(
            "constant term {} (Kronecker delta δ[n] — no discrete-impulse primitive)",
            pool.display(term)
        )));
    }

    // Irreducible quadratic denominator (complex-conjugate poles) → real
    // damped sinusoid `rⁿ(A cos θn + B sin θn)`.  Handle this before the
    // linear-pole numerator shape check, since here the numerator is a genuine
    // degree-≤2 polynomial in z (e.g. `P z² + Q z`), not `A·zᵖ`.
    if poly_degree(base, z, pool) == Some(2) {
        return invert_quadratic_pole(numer, base, k, z, n, pool);
    }

    let (coeff, p) = split_z_power(numer, z, pool).ok_or_else(|| {
        ZTransformError::NotInvertible(format!(
            "linear-pole numerator not of the form A·z^p: {}",
            pool.display(numer)
        ))
    })?;
    if p != 1 {
        return Err(ZTransformError::NotInvertible(format!(
            "numerator power of z ({p}) not in the table (expected A·z)"
        )));
    }

    match poly_degree(base, z, pool) {
        Some(1) => invert_linear_pole(coeff, base, k, z, n, pool),
        Some(d) => Err(ZTransformError::NotInvertible(format!(
            "denominator factor of degree {d} (only linear poles are tabulated): {}",
            pool.display(base)
        ))),
        None => Err(ZTransformError::NotInvertible(
            pool.display(base).to_string(),
        )),
    }
}

/// Split `numer = coeff · z^p` with `coeff` free of `z` and `p ≥ 0` an
/// integer. Returns `None` if `numer` is not of this shape.
fn split_z_power(numer: ExprId, z: ExprId, pool: &ExprPool) -> Option<(ExprId, u64)> {
    if numer == z {
        return Some((pool.integer(1_i32), 1));
    }
    if is_free_of(numer, z, pool) {
        return Some((numer, 0));
    }
    match pool.get(numer) {
        ExprData::Pow { base, exp } if base == z => {
            nonneg_int_exp(exp, pool).map(|p| (pool.integer(1_i32), p))
        }
        ExprData::Mul(args) => {
            let mut coeff_parts: Vec<ExprId> = Vec::new();
            let mut p = 0u64;
            for a in args {
                if a == z {
                    p += 1;
                    continue;
                }
                if let ExprData::Pow { base, exp } = pool.get(a) {
                    if base == z {
                        p += nonneg_int_exp(exp, pool)?;
                        continue;
                    }
                }
                if !is_free_of(a, z, pool) {
                    return None;
                }
                coeff_parts.push(a);
            }
            let coeff = match coeff_parts.len() {
                0 => pool.integer(1_i32),
                1 => coeff_parts[0],
                _ => pool.mul(coeff_parts),
            };
            Some((coeff, p))
        }
        _ => None,
    }
}

/// Decompose a term into `(numerator, denom_base, k)` with
/// `term = numerator · denom_base^{−k}` and `k ≥ 0`, `numerator` free of any
/// negative power of `z`.
fn split_rational_term(term: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, u64)> {
    let factors: Vec<ExprId> = match pool.get(term) {
        ExprData::Mul(a) => a,
        _ => vec![term],
    };
    let mut numer_parts: Vec<ExprId> = Vec::new();
    let mut base: Option<ExprId> = None;
    let mut k: u64 = 0;

    for &fac in &factors {
        if let ExprData::Pow { base: b, exp } = pool.get(fac) {
            if let ExprData::Integer(e) = pool.get(exp) {
                let ev = e.0;
                if ev < 0 {
                    if base.is_some() && base != Some(b) {
                        return None;
                    }
                    base = Some(b);
                    k = (-ev).to_u64()?;
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
        Some(b) => Some((numer, b, k)),
        None => Some((numer, pool.integer(1_i32), 0)),
    }
}

/// Degree of `base` as a polynomial in `z` (handles `z`, `z ± c` forms via
/// structural inspection). Returns `None` if not obviously polynomial.
fn poly_degree(base: ExprId, z: ExprId, pool: &ExprPool) -> Option<u64> {
    if base == z {
        return Some(1);
    }
    match pool.get(base) {
        ExprData::Add(args) => {
            let mut deg = 0u64;
            for a in args {
                deg = deg.max(monomial_degree(a, z, pool)?);
            }
            Some(deg)
        }
        ExprData::Pow { .. } | ExprData::Mul(_) => monomial_degree(base, z, pool),
        _ if is_free_of(base, z, pool) => Some(0),
        _ => None,
    }
}

fn monomial_degree(term: ExprId, z: ExprId, pool: &ExprPool) -> Option<u64> {
    if term == z {
        return Some(1);
    }
    if is_free_of(term, z, pool) {
        return Some(0);
    }
    match pool.get(term) {
        ExprData::Pow { base, exp } if base == z => nonneg_int_exp(exp, pool),
        ExprData::Mul(args) => {
            let mut deg = 0u64;
            for a in args {
                deg += monomial_degree(a, z, pool)?;
            }
            Some(deg)
        }
        _ => None,
    }
}

/// Invert `A·z·(z−a)^{−k}`:
///
/// - `k == 1`: `Z⁻¹{A·z/(z−a)} = A·aⁿ` (for `a == 1` this is the constant `A`).
/// - `k == 2`: `Z⁻¹{A·z/(z−a)²} = (A/a)·n·aⁿ` (for `a ≠ 0`); for `a == 0` the
///   term is `A·z^{-1}`, which is declined (anti-causal / improper for the
///   unilateral table).
fn invert_linear_pole(
    numer: ExprId,
    base: ExprId,
    k: u64,
    z: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, ZTransformError> {
    // `numer` is the coefficient `A` (free of `z` by construction of
    // `split_z_power`). base = z − a (monic). Extract a from (coeff·z + b): a = −b/coeff,
    // require coeff = 1.
    let (coeff, b) = as_affine(base, z, pool)
        .ok_or_else(|| ZTransformError::NotInvertible(pool.display(base).to_string()))?;
    if coeff != pool.integer(1_i32) {
        return Err(ZTransformError::NotInvertible(
            "non-monic linear denominator".into(),
        ));
    }
    let a = simp(neg(b, pool), pool); // a = −b

    match k {
        1 => {
            if a == pool.integer(0_i32) {
                // A·z/z = A·z^0 → constant A·δ-like term at n=0 only; decline
                // (see module docs on Kronecker delta).
                return Err(ZTransformError::NotInvertible(
                    "A·z/z term reduces to a Kronecker delta (no discrete-impulse primitive)"
                        .into(),
                ));
            }
            if a == pool.integer(1_i32) {
                // A·z/(z−1) → A (constant sequence)
                return Ok(numer);
            }
            // A·aⁿ
            let a_pow_n = pool.pow(a, n);
            Ok(pool.mul(vec![numer, a_pow_n]))
        }
        2 => {
            if a == pool.integer(0_i32) {
                return Err(ZTransformError::NotInvertible(
                    "A·z/z² term has no causal-sequence inverse in the table".into(),
                ));
            }
            // (A/a)·n·aⁿ
            let coeff = simp(pool.mul(vec![numer, recip(a, pool)]), pool);
            let a_pow_n = pool.pow(a, n);
            Ok(pool.mul(vec![coeff, n, a_pow_n]))
        }
        _ => Err(ZTransformError::NotInvertible(format!(
            "repeated linear pole of order {k} (only k = 1, 2 are tabulated)"
        ))),
    }
}

/// Invert a term `numer · (z² + b z + c)^{−k}` whose denominator is an
/// *irreducible* quadratic (complex-conjugate poles `r e^{±iθ}`).  Produces the
/// **real** damped sinusoid
///
/// ```text
///   Z⁻¹{·}(n) = rⁿ (A cos(θ n) + B sin(θ n)),
/// ```
///
/// with `r = √c`, `θ = acos(−b / 2√c)`, and no imaginary unit in the output.
///
/// The denominator must be monic with `k == 1` (a single quadratic factor); the
/// discriminant `b² − 4c` must be a literal *negative* rational (genuine
/// complex pair).  Repeated complex poles (`k ≥ 2`), non-monic denominators,
/// real-surd roots (non-negative discriminant), or a non-literal discriminant
/// are declined.
fn invert_quadratic_pole(
    numer: ExprId,
    base: ExprId,
    k: u64,
    z: ExprId,
    n: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, ZTransformError> {
    if k != 1 {
        return Err(ZTransformError::NotInvertible(format!(
            "repeated complex-conjugate pole of order {k} (only k = 1 is tabulated)"
        )));
    }

    // base = z² + b z + c (monic, coefficients free of z).
    let (b, c) = monic_quadratic_coeffs(base, z, pool).ok_or_else(|| {
        ZTransformError::NotInvertible(format!(
            "non-monic / non-quadratic denominator: {}",
            pool.display(base)
        ))
    })?;

    // Discriminant must be a literal negative rational (true complex pair).
    // A non-negative discriminant means real (possibly surd) roots — declined
    // (e.g. the Fibonacci denominator z² − z − 1 has discriminant 5 > 0).
    let b2 = pool.pow(b, pool.integer(2_i32));
    let four_c = pool.mul(vec![pool.integer(4_i32), c]);
    let disc = simp(pool.add(vec![b2, neg(four_c, pool)]), pool);
    match literal_rational(disc, pool) {
        Some(d) if d < 0 => {}
        Some(_) => {
            return Err(ZTransformError::NotInvertible(format!(
                "real-root quadratic denominator (discriminant ≥ 0): {}",
                pool.display(base)
            )));
        }
        None => {
            return Err(ZTransformError::NotInvertible(format!(
                "quadratic denominator with non-literal discriminant: {}",
                pool.display(base)
            )));
        }
    }

    // numer = P z² + Q z (no constant term: every apart(X/z) term is re-scaled
    // by z, so the lowest power is z¹).  Reject anything outside that shape.
    let (p_coeff, q_coeff) = quadratic_numer_pq(numer, z, pool).ok_or_else(|| {
        ZTransformError::NotInvertible(format!(
            "complex-pole numerator not of the form P·z² + Q·z: {}",
            pool.display(numer)
        ))
    })?;

    // r = √c, cosθ = −b / (2r), sinθ = √(1 − cos²θ), θ = acos(cosθ).
    let half = pool.rational(1_i32, 2_i32);
    let r = simp(pool.pow(c, half), pool);
    let two_r = pool.mul(vec![pool.integer(2_i32), r]);
    let cos_theta = simp(pool.mul(vec![neg(b, pool), recip(two_r, pool)]), pool);
    // sinθ = (1 − cos²θ)^{1/2}  (θ ∈ (0, π) so sinθ > 0).
    let cos2 = pool.pow(cos_theta, pool.integer(2_i32));
    let sin_theta = simp(
        pool.pow(pool.add(vec![pool.integer(1_i32), neg(cos2, pool)]), half),
        pool,
    );
    let theta = pool.func("acos", vec![cos_theta]);
    let theta_n = simp(pool.mul(vec![theta, n]), pool);

    // Match P z² + Q z = A·z(z − r cosθ) + B·z·r sinθ:
    //   A = P,  B = (Q + A r cosθ) / (r sinθ).
    let a_amp = p_coeff;
    let r_cos = pool.mul(vec![r, cos_theta]);
    let r_sin = pool.mul(vec![r, sin_theta]);
    let b_amp = simp(
        pool.mul(vec![
            pool.add(vec![q_coeff, pool.mul(vec![a_amp, r_cos])]),
            recip(r_sin, pool),
        ]),
        pool,
    );

    // rⁿ (A cos(θn) + B sin(θn)).
    let r_pow_n = pool.pow(r, n);
    let cos_term = pool.mul(vec![a_amp, pool.func("cos", vec![theta_n])]);
    let sin_term = pool.mul(vec![b_amp, pool.func("sin", vec![theta_n])]);
    let combo = pool.add(vec![cos_term, sin_term]);
    Ok(simp(pool.mul(vec![r_pow_n, combo]), pool))
}

/// Extract `(b, c)` from a monic quadratic `z² + b z + c` (coefficients free of
/// `z`).  Returns `None` for a non-monic leading coefficient or a missing/extra
/// degree.
fn monic_quadratic_coeffs(base: ExprId, z: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let z2 = pool.pow(z, pool.integer(2_i32));
    let terms: Vec<ExprId> = match pool.get(base) {
        ExprData::Add(a) => a,
        _ => vec![base],
    };
    let mut a2: Option<ExprId> = None; // coeff of z²
    let mut b1_parts: Vec<ExprId> = Vec::new(); // coeffs of z
    let mut c0_parts: Vec<ExprId> = Vec::new(); // constant
    for term in terms {
        if is_free_of(term, z, pool) {
            c0_parts.push(term);
            continue;
        }
        if let Some(coeff) = monomial_coeff(term, z2, z, pool) {
            if a2.is_some() {
                return None;
            }
            a2 = Some(coeff);
            continue;
        }
        if let Some(coeff) = monomial_coeff(term, z, z, pool) {
            b1_parts.push(coeff);
            continue;
        }
        return None;
    }
    // Leading coefficient must be 1 (monic).
    if simp(a2?, pool) != pool.integer(1_i32) {
        return None;
    }
    let b = match b1_parts.len() {
        0 => pool.integer(0_i32),
        1 => b1_parts[0],
        _ => pool.add(b1_parts),
    };
    let c = match c0_parts.len() {
        0 => pool.integer(0_i32),
        1 => c0_parts[0],
        _ => pool.add(c0_parts),
    };
    Some((b, c))
}

/// Coefficient of `power` (e.g. `z` or `z²`) in a single (non-Add) `term`,
/// requiring every other factor to be free of `var`; `1` for the bare power.
fn monomial_coeff(term: ExprId, power: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if term == power {
        return Some(pool.integer(1_i32));
    }
    if let ExprData::Mul(args) = pool.get(term) {
        let pos = args.iter().position(|&m| m == power)?;
        let others: Vec<ExprId> = args
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &m)| m)
            .collect();
        if others.iter().all(|&o| is_free_of(o, var, pool)) {
            return Some(match others.len() {
                0 => pool.integer(1_i32),
                1 => others[0],
                _ => pool.mul(others),
            });
        }
    }
    None
}

/// Extract `(P, Q)` from a numerator `P·z² + Q·z` (no constant or higher term);
/// `None` otherwise.
fn quadratic_numer_pq(numer: ExprId, z: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let z2 = pool.pow(z, pool.integer(2_i32));
    let terms: Vec<ExprId> = match pool.get(numer) {
        ExprData::Add(a) => a,
        _ => vec![numer],
    };
    let mut p_parts: Vec<ExprId> = Vec::new();
    let mut q_parts: Vec<ExprId> = Vec::new();
    for term in terms {
        if let Some(coeff) = monomial_coeff(term, z2, z, pool) {
            p_parts.push(coeff);
            continue;
        }
        if let Some(coeff) = monomial_coeff(term, z, z, pool) {
            q_parts.push(coeff);
            continue;
        }
        return None; // constant or higher-degree numerator term
    }
    let p = match p_parts.len() {
        0 => pool.integer(0_i32),
        1 => p_parts[0],
        _ => pool.add(p_parts),
    };
    let q = match q_parts.len() {
        0 => pool.integer(0_i32),
        1 => q_parts[0],
        _ => pool.add(q_parts),
    };
    Some((p, q))
}

/// If `expr` is a literal rational (integer or ratio), return it.
fn literal_rational(expr: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
