//! Bessel, Airy, Riemann/Hurwitz zeta and polylogarithm functions (G11b).
//!
//! # Nodes and conventions
//!
//! Every function is a registered primitive. Conventions are DLMF's, which
//! are also SymPy's, mpmath's and Arb's:
//!
//! | node | DLMF | notes |
//! |---|---|---|
//! | `bessel_j(ν, x)`, `bessel_y(ν, x)` | 10.2 | first and second kind |
//! | `bessel_i(ν, x)`, `bessel_k(ν, x)` | 10.25 | modified; `K_{−ν} = K_ν` |
//! | `spherical_jn(n, x)`, `spherical_yn(n, x)` | 10.47 | integer `n`; `j_n = √(π/(2x)) J_{n+½}` |
//! | `airy_ai(x)`, `airy_bi(x)` | 9.2 | and `airy_ai_prime`, `airy_bi_prime` |
//! | `zeta(s)` | 25.2 | Riemann; pole at `s = 1` |
//! | `hurwitz_zeta(s, a)` | 25.11 | `ζ(s, 1) = ζ(s)` |
//! | `polylog(s, z)` | 25.12 | `Li_s`; principal branch, cut `[1, ∞)` |
//!
//! `bessel_j(0, x)` and `bessel_j(1, x)` are spelled `bessel_j0(x)` and
//! `bessel_j1(x)` — the older unary primitives, which carry Taylor models for
//! the validated tier — and `polylog(2, z)` is `dilog(z)`, so one function
//! never has two spellings in the pool.
//!
//! # Exact forms
//!
//! * **Half-integer order** Bessel functions are elementary (DLMF 10.16.1,
//!   10.47, 10.49): with `expand = true` (the default of the constructors)
//!   `bessel_j(1/2, x) = √2·sin(x)/(√π·√x)`, `spherical_jn(n, x)` for an
//!   integer `n` is `a(1/x)·sin x + b(1/x)·cos x` with rational `a`, `b`, and
//!   `bessel_k(n + 1/2, x)` is `√(π/(2x))·e^{−x}·p(1/x)`. The coefficients
//!   come from the three-term recurrences over the rationals.
//! * **Integer order** Bessel functions are not elementary; the node is kept
//!   and canonicalised: `J_{−n} = (−1)ⁿ J_n`, `Y_{−n} = (−1)ⁿ Y_n`,
//!   `I_{−n} = I_n`, `K_{−ν} = K_ν`, parity in `x` for `J_n` and `I_n`, and
//!   `J_ν(0)`, `I_ν(0)` (`1` for `ν = 0`, `0` for `ν > 0`).
//! * **`ζ`**: `ζ(2n) = (−1)^{n+1} B_{2n} (2π)^{2n} / (2(2n)!)` and
//!   `ζ(−n) = (−1)ⁿ B_{n+1}/(n+1)` exactly (so `ζ(0) = −1/2`); odd
//!   `ζ(2n+1)` stays `zeta(2n+1)`. `ζ(−n, a) = −B_{n+1}(a)/(n+1)`,
//!   `ζ(s, 1/2) = (2^s − 1) ζ(s)`.
//! * **`Li_s`**: `Li_1(z) = −log(1 − z)`, `Li_0(z) = z/(1 − z)`, `Li_{−n}` the
//!   Eulerian rational function, `Li_s(1) = ζ(s)` (`s > 1`),
//!   `Li_s(−1) = −(1 − 2^{1−s}) ζ(s)`.
//!
//! # Derivatives
//!
//! `d/dx` in the variable only, by DLMF 10.6.2, 10.29.2, 10.51.2, 9.2.1,
//! 25.11.17, and `d/dz Li_s(z) = Li_{s−1}(z)/z` from the series. A derivative in the
//! **order** `ν`, in `s`, or of `zeta` itself has no closed form here: the
//! rule declines and `diff` refuses with `E-DIFF-001` instead of dropping the
//! term.
//!
//! # Refusals
//!
//! `E-SPECFN-006` for a pole (`zeta(1)`, `hurwitz_zeta(1, a)`,
//! `polylog(s, 1)` for `s ≤ 1`), `E-SPECFN-007` for a non-integer spherical
//! order, `E-SPECFN-004` for an explicit expansion past
//! [`MAX_EXPAND_ORDER`].

pub(crate) mod closed;
pub(crate) mod numeric;
pub(crate) mod primitives;

#[cfg(test)]
mod tests;

use rug::Rational;

use crate::kernel::{ExprData, ExprId, ExprPool};
pub use crate::orthopoly::SpecialFunctionError;
use crate::orthopoly::{int_literal, mk_mul, rational_literal};

use numeric::Kernel;

/// Largest `|order|` that [`bessel`] / [`spherical_bessel`] expand into
/// elementary functions; past it they refuse with `E-SPECFN-004` (build the
/// node with `expand = false` instead).
pub const MAX_EXPAND_ORDER: i64 = 200;

/// The cylinder Bessel functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BesselKind {
    /// `J_ν` (`bessel_j`).
    J,
    /// `Y_ν` (`bessel_y`).
    Y,
    /// `I_ν` (`bessel_i`).
    I,
    /// `K_ν` (`bessel_k`).
    K,
}

impl BesselKind {
    /// The primitive's name.
    pub fn name(self) -> &'static str {
        self.kernel().name()
    }

    fn kernel(self) -> Kernel {
        match self {
            BesselKind::J => Kernel::BesselJ,
            BesselKind::Y => Kernel::BesselY,
            BesselKind::I => Kernel::BesselI,
            BesselKind::K => Kernel::BesselK,
        }
    }
}

/// The spherical Bessel functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SphericalBesselKind {
    /// `j_n` (`spherical_jn`).
    J,
    /// `y_n` (`spherical_yn`).
    Y,
}

impl SphericalBesselKind {
    /// The primitive's name.
    pub fn name(self) -> &'static str {
        self.kernel().name()
    }

    fn kernel(self) -> Kernel {
        match self {
            SphericalBesselKind::J => Kernel::SphericalJ,
            SphericalBesselKind::Y => Kernel::SphericalY,
        }
    }
}

/// The Airy functions and their derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AiryKind {
    /// `Ai` (`airy_ai`).
    Ai,
    /// `Bi` (`airy_bi`).
    Bi,
    /// `Ai′` (`airy_ai_prime`).
    AiPrime,
    /// `Bi′` (`airy_bi_prime`).
    BiPrime,
}

impl AiryKind {
    /// The primitive's name.
    pub fn name(self) -> &'static str {
        self.kernel().name()
    }

    fn kernel(self) -> Kernel {
        match self {
            AiryKind::Ai => Kernel::AiryAi,
            AiryKind::Bi => Kernel::AiryBi,
            AiryKind::AiPrime => Kernel::AiryAiPrime,
            AiryKind::BiPrime => Kernel::AiryBiPrime,
        }
    }
}

/// `kernel(args)` in canonical form: the special values and reflections of
/// [`primitives::canonical`] applied, else the bare node.
pub(crate) fn node(kernel: Kernel, args: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    primitives::canonical(kernel, &args, pool).unwrap_or_else(|| pool.func(kernel.name(), args))
}

fn pi(pool: &ExprPool) -> ExprId {
    crate::eval::symbols::pi_symbol(pool)
}

fn is_literal(pool: &ExprPool, e: ExprId, v: i64) -> bool {
    rational_literal(pool, e).is_some_and(|q| q == v)
}

/// The elementary form of a half-integer-order Bessel function, or `None`
/// when `ν` is not a half-integer literal (or `x` is the literal `0`, where
/// the special values apply instead).
fn explicit_bessel(
    kind: BesselKind,
    nu: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, SpecialFunctionError> {
    let Some(q) = rational_literal(pool, nu) else {
        return Ok(None);
    };
    if *q.denom() != 2 || is_literal(pool, x, 0) {
        return Ok(None);
    }
    // ν = m + 1/2
    let m = Rational::from(&q - Rational::from((1, 2)));
    let m = m.numer().to_i64().filter(|m| m.abs() <= MAX_EXPAND_ORDER);
    let Some(m) = m else {
        return Err(SpecialFunctionError::TooLarge {
            what: format!("{} order", kind.name()),
            value: q.numer().to_i64().unwrap_or(i64::MAX) / 2,
            max: MAX_EXPAND_ORDER,
        });
    };
    let half = Rational::from((1, 2));
    let sqrt2 = pool.func("sqrt", vec![pool.integer(2)]);
    let e = match kind {
        BesselKind::J | BesselKind::Y | BesselKind::I => {
            let ((f0, fm1), rec, (s, c)) = match kind {
                BesselKind::J => (
                    closed::seeds_j(),
                    closed::Spherical::Ordinary,
                    ("sin", "cos"),
                ),
                BesselKind::Y => (
                    closed::seeds_y(),
                    closed::Spherical::Ordinary,
                    ("sin", "cos"),
                ),
                _ => (
                    closed::seeds_i(),
                    closed::Spherical::Modified,
                    ("sinh", "cosh"),
                ),
            };
            let pair = closed::spherical_pair(rec, f0, fm1, m)?;
            // F_{m+1/2}(x) = √(2x/π) f_m(x)
            mk_mul(
                pool,
                vec![
                    sqrt2,
                    pool.pow(pi(pool), pool.rational(-1, 2)),
                    closed::pair_expr(pool, &pair, x, &half, s, c),
                ],
            )
        }
        BesselKind::K => {
            let p = closed::k_half_poly(m)?;
            // K_{m+1/2}(x) = √(π/(2x)) e^{−x} p_m(1/x)
            mk_mul(
                pool,
                vec![
                    pool.rational(1, 2),
                    sqrt2,
                    pool.pow(pi(pool), pool.rational(1, 2)),
                    closed::laurent(pool, &p, x, &Rational::from((-1, 2))),
                    pool.func("exp", vec![pool.mul(vec![pool.integer(-1), x])]),
                ],
            )
        }
    };
    Ok(Some(e))
}

/// The elementary form of `j_n` / `y_n` for an integer literal `n`.
fn explicit_spherical(
    kind: SphericalBesselKind,
    n: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, SpecialFunctionError> {
    let Some(k) = int_literal(pool, n) else {
        return Ok(None);
    };
    if is_literal(pool, x, 0) {
        return Ok(None);
    }
    let Some(m) = k.to_i64().filter(|m| m.abs() <= MAX_EXPAND_ORDER) else {
        return Err(SpecialFunctionError::TooLarge {
            what: format!("{} order", kind.name()),
            value: k.to_i64().unwrap_or(i64::MAX),
            max: MAX_EXPAND_ORDER,
        });
    };
    let (f0, fm1) = match kind {
        SphericalBesselKind::J => closed::seeds_j(),
        SphericalBesselKind::Y => closed::seeds_y(),
    };
    let pair = closed::spherical_pair(closed::Spherical::Ordinary, f0, fm1, m)?;
    Ok(Some(closed::pair_expr(
        pool,
        &pair,
        x,
        &Rational::new(),
        "sin",
        "cos",
    )))
}

/// The cylinder Bessel function `kind_ν(x)`.
///
/// With `expand = true` a half-integer literal order gives the elementary
/// form (`bessel_j(1/2, x) = √2·sin(x)/(√π·√x)`); otherwise the canonical
/// node (`bessel_j(0, x)` is `bessel_j0(x)`, `J_{−n} = (−1)ⁿ J_n`, special
/// values at `x = 0`).
///
/// # Errors
///
/// `E-SPECFN-004` for an expansion past [`MAX_EXPAND_ORDER`];
/// `E-BUDGET-*` when the active budget runs out.
pub fn bessel(
    kind: BesselKind,
    nu: ExprId,
    x: ExprId,
    expand: bool,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    if expand {
        if let Some(e) = explicit_bessel(kind, nu, x, pool)? {
            return Ok(e);
        }
    }
    Ok(node(kind.kernel(), vec![nu, x], pool))
}

/// The spherical Bessel function `kind_n(x)`.
///
/// An integer literal `n` with `expand = true` gives the elementary form
/// (`spherical_jn(1, x) = sin(x)/x² − cos(x)/x`); a symbolic `n` (or
/// `expand = false`) the node.
///
/// # Errors
///
/// `E-SPECFN-007` for a non-integer literal order, `E-SPECFN-004` past
/// [`MAX_EXPAND_ORDER`], `E-BUDGET-*`.
pub fn spherical_bessel(
    kind: SphericalBesselKind,
    n: ExprId,
    x: ExprId,
    expand: bool,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    if rational_literal(pool, n).is_some_and(|q| *q.denom() != 1) {
        return Err(SpecialFunctionError::OrderNotInteger {
            function: kind.name(),
            value: pool.display(n).to_string(),
        });
    }
    if expand {
        if let Some(e) = explicit_spherical(kind, n, x, pool)? {
            return Ok(e);
        }
    }
    Ok(node(kind.kernel(), vec![n, x], pool))
}

/// The Airy function `kind(x)`; exact at `x = 0`
/// (`Ai(0) = 3^{−2/3}/Γ(2/3)`, DLMF 9.2.3–9.2.6).
pub fn airy(kind: AiryKind, x: ExprId, pool: &ExprPool) -> ExprId {
    node(kind.kernel(), vec![x], pool)
}

fn pole(function: &'static str, at: String) -> SpecialFunctionError {
    SpecialFunctionError::Pole { function, at }
}

/// The Riemann zeta function `ζ(s)`: exact at even positive and at
/// non-positive integers, the node elsewhere (`zeta(3)` stays `zeta(3)`).
///
/// # Errors
///
/// `E-SPECFN-006` at the pole `s = 1`; `E-BUDGET-*`.
pub fn zeta(s: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    if is_literal(pool, s, 1) {
        return Err(pole("zeta", "s = 1".into()));
    }
    crate::budget::check()?;
    Ok(node(Kernel::Zeta, vec![s], pool))
}

/// The Hurwitz zeta function `ζ(s, a) = Σ_{k≥0} (k + a)^{−s}` (continued).
///
/// # Errors
///
/// `E-SPECFN-006` at the pole `s = 1`; `E-BUDGET-*`.
pub fn hurwitz_zeta(s: ExprId, a: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    if is_literal(pool, s, 1) {
        return Err(pole("hurwitz_zeta", "s = 1".into()));
    }
    crate::budget::check()?;
    Ok(node(Kernel::HurwitzZeta, vec![s, a], pool))
}

/// The polylogarithm `Li_s(z)`, principal branch.
///
/// # Errors
///
/// `E-SPECFN-006` at `z = 1` for a literal `s ≤ 1` (where `Li_s` has a
/// pole); `E-BUDGET-*`.
pub fn polylog(s: ExprId, z: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    if is_literal(pool, z, 1) && rational_literal(pool, s).is_some_and(|q| q <= 1) {
        return Err(pole("polylog", format!("z = 1 (s = {})", pool.display(s))));
    }
    crate::budget::check()?;
    Ok(node(Kernel::Polylog, vec![s, z], pool))
}

/// Rewrite every Bessel node whose order has become a half-integer literal
/// (and every spherical Bessel node with an integer literal order) into its
/// elementary form, and re-canonicalise every `zeta` / `hurwitz_zeta` /
/// `polylog` node — what `subs(bessel_j(nu, x), {nu: 1/2})` needs.
///
/// # Errors
///
/// `E-SPECFN-004` past [`MAX_EXPAND_ORDER`]; `E-BUDGET-*`.
pub fn expand_special(expr: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    let mut memo = std::collections::HashMap::new();
    expand_rec(expr, pool, &mut memo)
}

fn rewrite_func(
    name: &str,
    args: &[ExprId],
    pool: &ExprPool,
) -> Result<Option<ExprId>, SpecialFunctionError> {
    let Some(k) = Kernel::from_name(name) else {
        return Ok(None);
    };
    if args.len() != k.arity() {
        return Ok(None);
    }
    let explicit = match k {
        Kernel::BesselJ => explicit_bessel(BesselKind::J, args[0], args[1], pool)?,
        Kernel::BesselY => explicit_bessel(BesselKind::Y, args[0], args[1], pool)?,
        Kernel::BesselI => explicit_bessel(BesselKind::I, args[0], args[1], pool)?,
        Kernel::BesselK => explicit_bessel(BesselKind::K, args[0], args[1], pool)?,
        Kernel::SphericalJ => explicit_spherical(SphericalBesselKind::J, args[0], args[1], pool)?,
        Kernel::SphericalY => explicit_spherical(SphericalBesselKind::Y, args[0], args[1], pool)?,
        _ => None,
    };
    Ok(explicit.or_else(|| primitives::canonical(k, args, pool)))
}

fn expand_rec(
    expr: ExprId,
    pool: &ExprPool,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
) -> Result<ExprId, SpecialFunctionError> {
    if let Some(&r) = memo.get(&expr) {
        return Ok(r);
    }
    crate::budget::check()?;
    let rec = |v: &[ExprId], memo: &mut std::collections::HashMap<ExprId, ExprId>| {
        v.iter()
            .map(|&a| expand_rec(a, pool, memo))
            .collect::<Result<Vec<_>, _>>()
    };
    let out = match pool.get(expr) {
        ExprData::Add(args) => {
            let a = rec(&args, memo)?;
            if a == args {
                expr
            } else {
                pool.add(a)
            }
        }
        ExprData::Mul(args) => {
            let a = rec(&args, memo)?;
            if a == args {
                expr
            } else {
                pool.mul(a)
            }
        }
        ExprData::Pow { base, exp } => {
            let (b, e) = (expand_rec(base, pool, memo)?, expand_rec(exp, pool, memo)?);
            if (b, e) == (base, exp) {
                expr
            } else {
                pool.pow(b, e)
            }
        }
        ExprData::Func { name, args } => {
            let a = rec(&args, memo)?;
            match rewrite_func(&name, &a, pool)? {
                Some(e) => e,
                None if a == args => expr,
                None => pool.func(name.clone(), a),
            }
        }
        ExprData::Piecewise { branches, default } => {
            let mut nb = Vec::with_capacity(branches.len());
            for (c, v) in &branches {
                nb.push((expand_rec(*c, pool, memo)?, expand_rec(*v, pool, memo)?));
            }
            let d = expand_rec(default, pool, memo)?;
            if nb == branches && d == default {
                expr
            } else {
                pool.piecewise(nb, d)
            }
        }
        ExprData::Predicate { kind, args } => {
            let a = rec(&args, memo)?;
            if a == args {
                expr
            } else {
                pool.predicate(kind, a)
            }
        }
        _ => expr,
    };
    memo.insert(expr, out);
    Ok(out)
}

/// Complex `f64` value of a special-function node of this module (and of
/// `bessel_j0` / `bessel_j1`), used by [`crate::eval::eval_complex_f64`].
/// `None` for any other name, a pole, or a value Arb could not certify.
pub(crate) fn complex_value(name: &str, args: &[(f64, f64)]) -> Option<(f64, f64)> {
    match name {
        "bessel_j0" | "bessel_j1" if args.len() == 1 => {
            let nu = if name == "bessel_j0" { 0.0 } else { 1.0 };
            numeric::eval_c64(Kernel::BesselJ, &[(nu, 0.0), args[0]])
        }
        _ => numeric::eval_c64(Kernel::from_name(name)?, args),
    }
}

/// Whether `name` is one of this module's primitives (or `bessel_j0/j1`),
/// i.e. whether [`complex_value`] can evaluate it.
pub(crate) fn has_complex_kernel(name: &str) -> bool {
    matches!(name, "bessel_j0" | "bessel_j1") || Kernel::from_name(name).is_some()
}
