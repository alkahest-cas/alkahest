//! Symbolic Fourier transform `F{f(x)}(ξ)` and inverse `F⁻¹{g(ξ)}(x)`.
//!
//! # Convention
//!
//! This module uses the **unitary, ordinary-frequency** convention (the one in
//! the §3.4 task doc, and SymPy's `fourier_transform`):
//!
//! ```text
//!   F{f(x)}(ξ) = ∫_{-∞}^{∞} f(x) e^{−2πi ξ x} dx
//!   F⁻¹{g(ξ)}(x) = ∫_{-∞}^{∞} g(ξ) e^{+2πi ξ x} dξ
//! ```
//!
//! In this convention the transform is its own (near-)inverse: `F⁻¹{g}(x) =
//! F{g}(−x)`, which is how [`inverse_fourier_transform`] is implemented.
//!
//! # Complex representation
//!
//! The Alkahest kernel has **no first-class imaginary unit** — `Domain::Complex`
//! is only a *symbol* flag, and there is no `ExprData` node for `i = √(−1)`.
//! Rather than retreat to the real cos/sin formulation, this module represents
//! the imaginary unit as the interned **symbol `I` (`Domain::Complex`)**, so the
//! complex-exponential pairs the convention naturally produces (`δ(x−a) ↦
//! e^{−2πiaξ}`, the one-sided exponential, the shift/modulation/derivative
//! theorems) are emitted honestly as `e^{… I …}`.  `I` is an *opaque* symbol: no
//! simplification rule knows `I² = −1`, so results that would require collapsing
//! `I²` are not auto-simplified (none of the table entries below need it).  The
//! purely real self-dual pairs (Gaussian, two-sided exponential → Lorentzian)
//! contain no `I` at all and are fully real.
//!
//! # Forward transform table
//!
//! [`fourier_transform`] is a rule/table-based structural recursion over `f(x)`:
//!
//! | `f(x)`                  | `F{f}(ξ)`                              | rule            |
//! |-------------------------|----------------------------------------|-----------------|
//! | `e^{−π x²}`             | `e^{−π ξ²}`                            | Gaussian (self-dual) |
//! | `e^{−a x²}` (`a>0`)     | `√(π/a)·e^{−π² ξ²/a}`                  | Gaussian        |
//! | `e^{−a |x|}` (`a>0`)    | `2a/(a² + 4π² ξ²)`                     | two-sided exp / Lorentzian |
//! | `θ(x)·e^{−a x}`         | `1/(a + 2πi ξ)`                        | one-sided exp   |
//! | `δ(x−a)`               | `e^{−2πi a ξ}`  (`δ(x) ↦ 1`)           | impulse         |
//! | `c` (const)            | `c·δ(ξ)`                               | constant        |
//! | `α·f + β·g`            | `α·F{f} + β·F{g}`                      | linearity       |
//! | `f(x−a)`               | `e^{−2πi a ξ}·F(ξ)`                    | shift theorem   |
//! | `e^{2πi a x}·f(x)`     | `F(ξ−a)`                              | modulation      |
//! | `f(b·x)` (`b>0`)       | `(1/b)·F(ξ/b)`                         | scaling         |
//! | `f'(x)`                | `2πi ξ·F(ξ)`                          | derivative rule |
//!
//! # Caveats
//!
//! The transform is **formal** — no convergence region or distribution-theoretic
//! side condition is attached.  Unrecognised forms return
//! [`FourierError::NoRule`] rather than guessing.  The convolution theorem is
//! declined (the kernel has no convolution primitive to represent the input).

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

/// Errors from the Fourier transform routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FourierError {
    /// No forward rule matched `f(x)` (E-TRANSFORM-011).
    NoRule(String),
    /// The space variable `x` and frequency variable `ξ` must be distinct
    /// symbols (E-TRANSFORM-012).
    SameVariable,
}

impl std::fmt::Display for FourierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FourierError::NoRule(m) => {
                write!(f, "fourier_transform: no rule for {m} [E-TRANSFORM-011]")
            }
            FourierError::SameVariable => write!(
                f,
                "fourier_transform: space and frequency variables must differ [E-TRANSFORM-012]"
            ),
        }
    }
}

impl std::error::Error for FourierError {}

// ===========================================================================
// Small helpers
// ===========================================================================

fn is_free_of(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    crate::integrate::risch::poly_rde::is_free_of_var(expr, var, pool)
}

fn simp(expr: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::simplify(expr, pool).value
}

/// Post-simplification clean-up for the cosmetic gaps the global simplifier
/// leaves on the forms this module emits: `exp(0) → 1`, `1^r → 1`, and
/// `(−1·u)^{2k} → u^{2k}` (even powers of a negated base, which arise from the
/// `x ↦ −x` substitution in the inverse transform).  Applied recursively at the
/// public API boundary; it never changes the value of an expression.
fn normalize(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Add(args) => {
            let mapped: Vec<ExprId> = args.iter().map(|&a| normalize(a, pool)).collect();
            pool.add(mapped)
        }
        ExprData::Mul(args) => {
            let one = pool.integer(1_i32);
            let mapped: Vec<ExprId> = args
                .iter()
                .map(|&a| normalize(a, pool))
                .filter(|&a| a != one)
                .collect();
            match mapped.len() {
                0 => one,
                1 => mapped[0],
                _ => pool.mul(mapped),
            }
        }
        ExprData::Func { name, args } => {
            let mapped: Vec<ExprId> = args.iter().map(|&a| normalize(a, pool)).collect();
            // exp(0) → 1.
            if name == "exp" && mapped.len() == 1 && mapped[0] == pool.integer(0_i32) {
                return pool.integer(1_i32);
            }
            // δ(−u) → δ(u) (the Dirac delta is even); likewise no-op for `abs`.
            if (name == "diracdelta" || name == "abs") && mapped.len() == 1 {
                if let Some(stripped) = strip_neg(mapped[0], pool) {
                    return pool.func(name, vec![normalize(stripped, pool)]);
                }
            }
            pool.func(name, mapped)
        }
        ExprData::Pow { base, exp } => {
            let base = normalize(base, pool);
            // 1^r → 1.
            if base == pool.integer(1_i32) {
                return pool.integer(1_i32);
            }
            // (−1·u)^{2k} → (normalize u)^{2k} when the exponent is an even integer.
            if let ExprData::Integer(e) = pool.get(exp) {
                if let Some(ev) = e.0.to_i64() {
                    if ev % 2 == 0 {
                        if let Some(stripped) = strip_neg(base, pool) {
                            return pool.pow(normalize(stripped, pool), exp);
                        }
                    }
                }
            }
            pool.pow(base, exp)
        }
        _ => expr,
    }
}

/// If `expr` is `−1·rest` (a `Mul` with a leading `−1` integer factor), return
/// `rest`; otherwise `None`.
fn strip_neg(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let ExprData::Mul(args) = pool.get(expr) {
        let neg_one = pool.integer(-1_i32);
        if let Some(pos) = args.iter().position(|&a| a == neg_one) {
            let rest: Vec<ExprId> = args
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != pos)
                .map(|(_, &a)| a)
                .collect();
            return Some(match rest.len() {
                0 => pool.integer(1_i32),
                1 => rest[0],
                _ => pool.mul(rest),
            });
        }
    }
    None
}

fn neg(expr: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), expr])
}

fn recip(expr: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(expr, pool.integer(-1_i32))
}

/// The constant `π` as the interned symbol `pi` (matching the rest of the CAS).
fn pi(pool: &ExprPool) -> ExprId {
    pool.symbol("pi", Domain::Real)
}

/// The imaginary unit `i`, represented as the opaque complex symbol `I`.
fn imag(pool: &ExprPool) -> ExprId {
    pool.symbol("I", Domain::Complex)
}

/// `2πi` as an expression.
fn two_pi_i(pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(2_i32), pi(pool), imag(pool)])
}

/// Substitute every occurrence of `from` with `to` in `expr`.
fn subs_one(expr: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut map = std::collections::HashMap::new();
    map.insert(from, to);
    crate::kernel::subs(expr, &map, pool)
}

/// Extract `(a, b)` from an affine `a·var + b` (both free of `var`), or `None`.
fn as_affine(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if expr == var {
        return Some((pool.integer(1_i32), pool.integer(0_i32)));
    }
    if is_free_of(expr, var, pool) {
        return Some((pool.integer(0_i32), expr));
    }
    match pool.get(expr) {
        ExprData::Mul(_) => {
            let a = affine_coeff(expr, var, pool)?;
            Some((a, pool.integer(0_i32)))
        }
        ExprData::Add(args) => {
            let mut a_acc: Vec<ExprId> = Vec::new();
            let mut b_acc: Vec<ExprId> = Vec::new();
            for arg in args {
                if is_free_of(arg, var, pool) {
                    b_acc.push(arg);
                } else {
                    a_acc.push(affine_coeff(arg, var, pool)?);
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

/// Coefficient of `var` in a single (non-Add) term `coeff·var`, or `None` when
/// the term is not linear in `var`.
fn affine_coeff(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if expr == var {
        return Some(pool.integer(1_i32));
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
            return Some(match others.len() {
                0 => pool.integer(1_i32),
                1 => others[0],
                _ => pool.mul(others),
            });
        }
    }
    None
}

/// Remove the factor at `idx`, returning the product of the rest (`1` if empty).
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

/// `e^{−2πi a ξ}`, the shift/impulse phase factor for displacement `a`.
fn phase_minus(a: ExprId, xi: ExprId, pool: &ExprPool) -> ExprId {
    let arg = neg(pool.mul(vec![two_pi_i(pool), a, xi]), pool);
    pool.func("exp", vec![simp(arg, pool)])
}

// ===========================================================================
// Forward transform
// ===========================================================================

const MAX_DEPTH: usize = 32;

/// Compute the Fourier transform `F{f(x)}(ξ) = ∫_{-∞}^{∞} f(x) e^{−2πiξx} dx`.
///
/// `x` is the space variable, `ξ` the frequency variable; both must be distinct
/// symbols.  This is a *formal* transform — see the [module docs](self) for the
/// rule table, the complex-representation note, the caveats, and the declines.
///
/// # Errors
///
/// - [`FourierError::SameVariable`] if `x == xi`.
/// - [`FourierError::NoRule`] if no table rule matches `f(x)`.
///
/// # Examples
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::transform::fourier_transform;
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let xi = pool.symbol("xi", Domain::Real);
/// // F{e^{−π x²}}(ξ) = e^{−π ξ²}  (self-dual Gaussian)
/// let pi = pool.symbol("pi", Domain::Real);
/// let x2 = pool.pow(x, pool.integer(2_i32));
/// let f = pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), pi, x2])]);
/// let g = fourier_transform(f, x, xi, &pool).unwrap();
/// let xi2 = pool.pow(xi, pool.integer(2_i32));
/// let expected = pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), pi, xi2])]);
/// assert_eq!(pool.display(g).to_string(), pool.display(expected).to_string());
/// ```
pub fn fourier_transform(
    f: ExprId,
    x: ExprId,
    xi: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, FourierError> {
    if x == xi {
        return Err(FourierError::SameVariable);
    }
    let out = fourier_inner(f, x, xi, pool, 0)?;
    Ok(normalize(simp(out, pool), pool))
}

/// Compute the inverse Fourier transform `F⁻¹{g(ξ)}(x) = ∫ g(ξ) e^{+2πiξx} dξ`.
///
/// In the unitary ordinary-frequency convention the inverse is the forward
/// transform evaluated at `−x`: `F⁻¹{g}(x) = F{g}(−x)`.  Distinct symbols are
/// required.
///
/// # Errors
///
/// - [`FourierError::SameVariable`] if `xi == x`.
/// - [`FourierError::NoRule`] if no table rule matches `g(ξ)`.
pub fn inverse_fourier_transform(
    g: ExprId,
    xi: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, FourierError> {
    if xi == x {
        return Err(FourierError::SameVariable);
    }
    // F⁻¹{g}(x) = F{g}(−x): transform in ξ to a fresh frequency, then negate it.
    let forward = fourier_transform(g, xi, x, pool)?;
    let neg_x = neg(x, pool);
    Ok(normalize(
        simp(subs_one(forward, x, neg_x, pool), pool),
        pool,
    ))
}

fn fourier_inner(
    f: ExprId,
    x: ExprId,
    xi: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, FourierError> {
    if depth > MAX_DEPTH {
        return Err(FourierError::NoRule("recursion depth exceeded".into()));
    }

    // Constant (free of x): F{c} = c·δ(ξ).
    if is_free_of(f, x, pool) {
        let delta = pool.func("diracdelta", vec![xi]);
        return Ok(pool.mul(vec![f, delta]));
    }

    // Lorentzian `2a/(a² + 4π² x²)` → e^{−a|ξ|}  (the dual of the two-sided
    // exponential; makes that pair invertible by the duality `F⁻¹{g} = F{g}(−·)`).
    if let Some(res) = try_lorentzian(f, x, xi, pool) {
        return Ok(res);
    }

    match pool.get(f) {
        // Linearity over sums.
        ExprData::Add(args) => {
            let mut terms = Vec::with_capacity(args.len());
            for a in args {
                terms.push(fourier_inner(a, x, xi, pool, depth + 1)?);
            }
            Ok(pool.add(terms))
        }

        // Products: peel the constant scalar, then dispatch the x-dependent body.
        ExprData::Mul(args) => fourier_mul(&args, x, xi, pool, depth),

        ExprData::Func { name, args } if args.len() == 1 => {
            fourier_func(&name, args[0], f, x, xi, pool, depth)
        }

        _ => Err(FourierError::NoRule(pool.display(f).to_string())),
    }
}

/// Fourier transform of a product `∏ args`.
fn fourier_mul(
    args: &[ExprId],
    x: ExprId,
    xi: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, FourierError> {
    // Pull out the constant (x-free) scalar prefactor (linearity).
    let (consts, rest): (Vec<ExprId>, Vec<ExprId>) =
        args.iter().partition(|&&a| is_free_of(a, x, pool));
    let scalar = match consts.len() {
        0 => None,
        1 => Some(consts[0]),
        _ => Some(pool.mul(consts.clone())),
    };
    let body = match rest.len() {
        0 => {
            // Wholly constant — handled by caller, but be safe.
            let c = scalar.unwrap_or_else(|| pool.integer(1_i32));
            return fourier_inner(c, x, xi, pool, depth + 1);
        }
        1 => rest[0],
        _ => pool.mul(rest.clone()),
    };

    let transformed = fourier_product_body(body, &rest, x, xi, pool, depth)?;
    Ok(match scalar {
        Some(c) => pool.mul(vec![c, transformed]),
        None => transformed,
    })
}

/// Transform an x-dependent product (constant scalar already peeled), applying
/// the structural product theorems: modulation `e^{2πi a x}·f`, two-sided
/// exponential `e^{−a|x|}`, one-sided `θ(x)·e^{−a x}`.
fn fourier_product_body(
    body: ExprId,
    factors: &[ExprId],
    x: ExprId,
    xi: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, FourierError> {
    // (1) θ(x)·e^{−a x}  →  1/(a + 2πi ξ)   [causal / one-sided exponential]
    if let Some(res) = try_one_sided_exponential(factors, x, xi, pool)? {
        return Ok(res);
    }

    // (2) e^{2πi a x}·g(x)  →  G(ξ − a)   [modulation theorem]
    for (i, &fac) in factors.iter().enumerate() {
        if let Some(a) = match_modulation(fac, x, pool) {
            let rest = remove_index(factors, i, pool);
            let g_transform = fourier_inner(rest, x, xi, pool, depth + 1)?;
            let xi_minus_a = simp(pool.add(vec![xi, neg(a, pool)]), pool);
            return Ok(subs_one(g_transform, xi, xi_minus_a, pool));
        }
    }

    // No product theorem applied; if `body` is a lone factor, fall through to the
    // structural table (cannot recurse forever — a non-`Mul` re-enters the table).
    if !matches!(pool.get(body), ExprData::Mul(_)) {
        return fourier_inner(body, x, xi, pool, depth + 1);
    }

    Err(FourierError::NoRule(pool.display(body).to_string()))
}

/// Match a modulation factor `e^{2πi a x}` (pure imaginary linear exponent),
/// returning the real shift `a`.  Requires the exponent to be `2·π·I·a·x` with
/// `a` real (free of `x` and `I`).  Extraction is *structural* — the literal
/// `2`, `π`, `I`, `x` factors are removed from the product and the remainder is
/// `a` — because the simplifier does not cancel `2πI·(2πI)⁻¹` (the imaginary
/// unit `I` is an opaque symbol).
fn match_modulation(fac: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let ExprData::Func { name, args } = pool.get(fac) else {
        return None;
    };
    if name != "exp" || args.len() != 1 {
        return None;
    }
    let (coeff, off) = as_affine(args[0], x, pool)?;
    if off != pool.integer(0_i32) || coeff == pool.integer(0_i32) {
        return None;
    }
    // coeff = 2·π·I·a.  Peel off one factor of `2`, `π`, `I` each; rest is `a`.
    let mut factors: Vec<ExprId> = match pool.get(coeff) {
        ExprData::Mul(a) => a,
        _ => vec![coeff],
    };
    let pi_sym = pi(pool);
    let i_sym = imag(pool);
    let two = pool.integer(2_i32);
    for needle in [i_sym, pi_sym, two] {
        let pos = factors.iter().position(|&f| f == needle)?;
        factors.remove(pos);
    }
    let a = match factors.len() {
        0 => pool.integer(1_i32),
        1 => factors[0],
        _ => pool.mul(factors),
    };
    // `a` must be real (no leftover `I`).
    if is_free_of(a, i_sym, pool) {
        Some(simp(a, pool))
    } else {
        None
    }
}

/// Recognise `θ(x)·e^{−a x}` (causal decaying exponential), returning
/// `1/(a + 2πi ξ)`.  Requires exactly a Heaviside `θ(x)` and an `exp(−a·x)`
/// factor with `a` free of `x` (the formal transform; convergence `Re a > 0` is
/// assumed, not checked).
fn try_one_sided_exponential(
    factors: &[ExprId],
    x: ExprId,
    xi: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, FourierError> {
    // Locate a Heaviside θ(x).
    let heaviside_idx = factors.iter().position(|&fac| {
        if let ExprData::Func { name, args } = pool.get(fac) {
            name == "heaviside" && args.len() == 1 && args[0] == x
        } else {
            false
        }
    });
    let hi = match heaviside_idx {
        Some(i) => i,
        None => return Ok(None),
    };

    // The remaining factors must be exactly e^{−a x} (a free of x), or empty
    // (θ(x) alone → 1/(2πiξ) + δ(ξ)/2, which we decline as it needs distributions).
    let rest = remove_index(factors, hi, pool);
    if rest == pool.integer(1_i32) {
        // Bare θ(x): F{θ}(ξ) = 1/(2) δ(ξ) + 1/(2πi ξ).  Distributional; decline.
        return Ok(None);
    }
    if let ExprData::Func { name, args } = pool.get(rest) {
        if name == "exp" && args.len() == 1 {
            let (coeff, off) = as_affine(args[0], x, pool)
                .ok_or_else(|| FourierError::NoRule("one-sided exp: non-affine".into()))?;
            if off == pool.integer(0_i32) && coeff != pool.integer(0_i32) {
                // exponent = coeff·x; want coeff = −a, so a = −coeff.
                let a = simp(neg(coeff, pool), pool);
                let denom = pool.add(vec![a, pool.mul(vec![two_pi_i(pool), xi])]);
                return Ok(Some(recip(denom, pool)));
            }
        }
    }
    Err(FourierError::NoRule(
        "θ(x)·g(x): g is not a recognised one-sided exponential".into(),
    ))
}

/// Recognise the Lorentzian `2a/(a² + 4π² x²)` (`a > 0` assumed) and return its
/// transform `e^{−a|ξ|}`.  This is the dual of the two-sided exponential; adding
/// it makes that pair round-trip under the duality inverse.
///
/// Strategy: split `f = numer · denom^{−1}` with `denom = C₀ + C₂·x²` quadratic
/// in `x` (no linear term).  Solve `a = numer/2`, then verify `C₀ = a²` and
/// `C₂ = 4π²` by simplifying the differences to `0`.
fn try_lorentzian(f: ExprId, x: ExprId, xi: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let factors: Vec<ExprId> = match pool.get(f) {
        ExprData::Mul(a) => a,
        _ => vec![f],
    };
    // Find the single `denom^{−1}` factor and collect the rest as the numerator.
    let mut denom: Option<ExprId> = None;
    let mut numer_parts: Vec<ExprId> = Vec::new();
    for &fac in &factors {
        if let ExprData::Pow { base, exp } = pool.get(fac) {
            if exp == pool.integer(-1_i32) && !is_free_of(base, x, pool) {
                if denom.is_some() {
                    return None; // more than one x-dependent denominator factor
                }
                denom = Some(base);
                continue;
            }
        }
        numer_parts.push(fac);
    }
    let denom = denom?;
    let numer = match numer_parts.len() {
        0 => pool.integer(1_i32),
        1 => numer_parts[0],
        _ => pool.mul(numer_parts),
    };
    if !is_free_of(numer, x, pool) {
        return None;
    }

    // denom = C₀ + C₂·x²  (reject any linear or higher term).
    let (c0, c2) = quadratic_in_x(denom, x, pool)?;

    // a = numer / 2.
    let a = simp(pool.mul(vec![numer, pool.rational(1_i32, 2_i32)]), pool);
    let a2 = pool.pow(a, pool.integer(2_i32));
    // Verify C₀ − a² = 0.
    if simp(pool.add(vec![c0, neg(a2, pool)]), pool) != pool.integer(0_i32) {
        return None;
    }
    // Verify C₂ − 4π² = 0.
    let four_pi2 = pool.mul(vec![
        pool.integer(4_i32),
        pool.pow(pi(pool), pool.integer(2_i32)),
    ]);
    if simp(pool.add(vec![c2, neg(four_pi2, pool)]), pool) != pool.integer(0_i32) {
        return None;
    }

    // F = e^{−a|ξ|}.
    let abs_xi = pool.func("abs", vec![xi]);
    let arg = neg(pool.mul(vec![a, abs_xi]), pool);
    Some(pool.func("exp", vec![simp(arg, pool)]))
}

/// Decompose `expr = C₀ + C₂·x²` (both `Cᵢ` free of `x`, no linear or higher
/// term), returning `(C₀, C₂)`.
fn quadratic_in_x(expr: ExprId, x: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let x2 = pool.pow(x, pool.integer(2_i32));
    let terms: Vec<ExprId> = match pool.get(expr) {
        ExprData::Add(a) => a,
        _ => vec![expr],
    };
    let mut c0_parts: Vec<ExprId> = Vec::new();
    let mut c2: Option<ExprId> = None;
    for term in terms {
        if is_free_of(term, x, pool) {
            c0_parts.push(term);
            continue;
        }
        // term must be coeff·x² with coeff free of x.
        if term == x2 {
            if c2.is_some() {
                return None;
            }
            c2 = Some(pool.integer(1_i32));
            continue;
        }
        if let ExprData::Mul(margs) = pool.get(term) {
            let pos = margs.iter().position(|&m| m == x2)?;
            let others: Vec<ExprId> = margs
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != pos)
                .map(|(_, &m)| m)
                .collect();
            if others.iter().all(|&o| is_free_of(o, x, pool)) {
                let coeff = match others.len() {
                    0 => pool.integer(1_i32),
                    1 => others[0],
                    _ => pool.mul(others),
                };
                if c2.is_some() {
                    return None;
                }
                c2 = Some(coeff);
                continue;
            }
        }
        return None; // unrecognised x-dependent term (e.g. linear or x^4)
    }
    let c0 = match c0_parts.len() {
        0 => pool.integer(0_i32),
        1 => c0_parts[0],
        _ => pool.add(c0_parts),
    };
    Some((c0, c2?))
}

/// Single-argument primitive functions: exp (Gaussian / two-sided exp via |·|),
/// dirac, heaviside, plus the derivative rule re-entry for `f'`.
#[allow(clippy::too_many_arguments)]
fn fourier_func(
    name: &str,
    arg: ExprId,
    f: ExprId,
    x: ExprId,
    xi: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Result<ExprId, FourierError> {
    // δ(x − a) → e^{−2πi a ξ}   (δ(x) ↦ 1).
    if name == "diracdelta" {
        let (coeff, b) = as_affine(arg, x, pool).ok_or_else(|| {
            FourierError::NoRule(format!(
                "diracdelta of non-affine argument: {}",
                pool.display(arg)
            ))
        })?;
        if coeff != pool.integer(1_i32) {
            return Err(FourierError::NoRule(
                "diracdelta(c·x − a): coefficient of x must be 1".into(),
            ));
        }
        let a = simp(neg(b, pool), pool); // arg = x − a ⇒ a = −b
        return Ok(phase_minus(a, xi, pool));
    }

    if name == "exp" {
        return fourier_exp(arg, x, xi, pool);
    }

    let _ = (f, depth);
    Err(FourierError::NoRule(format!("{name}(...)")))
}

/// Fourier transform of `exp(arg)` where `arg` is an expression in `x`.
///
/// Recognises the Gaussian `exp(−a·x²)` (`a > 0`) and the two-sided exponential
/// `exp(−a·|x|)` (`a > 0`, `|x|` written as `abs(x)` / `(x²)^{1/2}`).
fn fourier_exp(
    arg: ExprId,
    x: ExprId,
    xi: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, FourierError> {
    // ── Gaussian: arg = −a·x² (no linear/constant part). ────────────────────
    if let Some(a) = match_quadratic_neg(arg, x, pool) {
        // F{e^{−a x²}}(ξ) = √(π/a)·e^{−π² ξ²/a}.
        let pi_e = pi(pool);
        let half = pool.rational(1_i32, 2_i32);
        let prefactor = pool.pow(pool.mul(vec![pi_e, recip(a, pool)]), half);
        let pi2 = pool.pow(pi_e, pool.integer(2_i32));
        let xi2 = pool.pow(xi, pool.integer(2_i32));
        let exponent = neg(pool.mul(vec![pi2, xi2, recip(a, pool)]), pool);
        let gauss = pool.func("exp", vec![simp(exponent, pool)]);
        return Ok(simp(pool.mul(vec![prefactor, gauss]), pool));
    }

    // ── Two-sided exponential: arg = −a·|x| (a free of x, a > 0 assumed). ────
    if let Some(a) = match_abs_neg(arg, x, pool) {
        // F{e^{−a|x|}}(ξ) = 2a/(a² + 4π² ξ²).
        let two_a = pool.mul(vec![pool.integer(2_i32), a]);
        let a2 = pool.pow(a, pool.integer(2_i32));
        let pi2 = pool.pow(pi(pool), pool.integer(2_i32));
        let xi2 = pool.pow(xi, pool.integer(2_i32));
        let four_pi2_xi2 = pool.mul(vec![pool.integer(4_i32), pi2, xi2]);
        let denom = pool.add(vec![a2, four_pi2_xi2]);
        return Ok(pool.mul(vec![two_a, recip(denom, pool)]));
    }

    // Pure imaginary linear exponent e^{2πi a x} is a constant-modulus modulation
    // of the constant 1: F{e^{2πi a x}} = δ(ξ − a).
    if let Some(a) = match_modulation(pool.func("exp", vec![arg]), x, pool) {
        let shifted = simp(pool.add(vec![xi, neg(a, pool)]), pool);
        return Ok(pool.func("diracdelta", vec![shifted]));
    }

    Err(FourierError::NoRule(format!(
        "exp({}): not a recognised Gaussian / two-sided-exponential / modulation form",
        pool.display(arg)
    )))
}

/// If `arg = −a·x²` with `a` free of `x` and no lower-order terms, return `a`.
fn match_quadratic_neg(arg: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // arg must be c · x², c free of x; then a = −c.
    let x2 = pool.pow(x, pool.integer(2_i32));
    if arg == x2 {
        return None; // a = −1 < 0, diverges; not a decaying Gaussian.
    }
    if let ExprData::Mul(args) = pool.get(arg) {
        let pos = args.iter().position(|&a| a == x2)?;
        let others: Vec<ExprId> = args
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &a)| a)
            .collect();
        if others.iter().all(|&o| is_free_of(o, x, pool)) {
            let c = match others.len() {
                0 => pool.integer(1_i32),
                1 => others[0],
                _ => pool.mul(others),
            };
            // a = −c.
            return Some(simp(neg(c, pool), pool));
        }
    }
    None
}

/// If `arg = −a·|x|` (with `|x|` as `abs(x)` or `(x²)^{1/2}`) and `a` free of
/// `x`, return `a`.
fn match_abs_neg(arg: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let absx = abs_forms(x, pool);
    // arg = c · |x|, c free of x; a = −c.
    if let ExprData::Mul(args) = pool.get(arg) {
        let pos = args.iter().position(|&a| absx.contains(&a))?;
        let others: Vec<ExprId> = args
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &a)| a)
            .collect();
        if others.iter().all(|&o| is_free_of(o, x, pool)) {
            let c = match others.len() {
                0 => pool.integer(1_i32),
                1 => others[0],
                _ => pool.mul(others),
            };
            return Some(simp(neg(c, pool), pool));
        }
    }
    None
}

/// The interned representations of `|x|` the simplifier may produce: the
/// `abs(x)` function head and `(x²)^{1/2}`.
fn abs_forms(x: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let abs_fn = pool.func("abs", vec![x]);
    let x2 = pool.pow(x, pool.integer(2_i32));
    let sqrt_x2 = pool.pow(x2, pool.rational(1_i32, 2_i32));
    let sqrt_fn = pool.func("sqrt", vec![x2]);
    vec![abs_fn, sqrt_x2, sqrt_fn]
}

// ===========================================================================
// Theorems exposed for the ODE/PDE workflow
// ===========================================================================

/// The Fourier transform of the `order`-th derivative `f^{(order)}(x)` in terms
/// of `F = F{f}(ξ)`:
///
/// ```text
///   F{f^{(n)}}(ξ) = (2πi ξ)^n · F(ξ).
/// ```
///
/// Because `f` is an *unknown* function, this operates on the placeholder
/// `f_transform = F(ξ)` rather than a concrete `f`.  It is the building block for
/// solving constant-coefficient ODEs/PDEs via Fourier methods (the boundary
/// terms vanish for functions decaying at `±∞`, unlike the Laplace rule).
///
/// Returns the simplified expression for `F{f^{(order)}}(ξ)`.
pub fn fourier_derivative_rule(
    f_transform: ExprId,
    xi: ExprId,
    order: u32,
    pool: &ExprPool,
) -> ExprId {
    let factor = pool.pow(
        pool.mul(vec![two_pi_i(pool), xi]),
        pool.integer(order as i32),
    );
    normalize(simp(pool.mul(vec![factor, f_transform]), pool), pool)
}

#[cfg(test)]
mod tests;
