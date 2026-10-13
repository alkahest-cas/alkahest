//! Hankel functions `H^{(1)}_ν(x) = J_ν(x) + i·Y_ν(x)` and
//! `H^{(2)}_ν(x) = J_ν(x) − i·Y_ν(x)` (DLMF 10.4).
//!
//! * A half-integer order has the elementary form
//!   (DLMF 10.49.6–10.49.7 with `H_{n+½} = √(2x/π)·h_n`)
//!
//!   ```text
//!   H⁽¹⁾_{n+½}(x) = √(2/(πx)) · (−i)^{n+1} · e^{ix}  · Σ_{k=0}^{n} i^k    (n+k)!/(k!(n−k)!) (2x)^{−k}
//!   H⁽²⁾_{n+½}(x) = √(2/(πx)) ·   i^{n+1}  · e^{−ix} · Σ_{k=0}^{n} (−i)^k (n+k)!/(k!(n−k)!) (2x)^{−k}
//!   ```
//!
//!   and a negative half-integer order the reflection
//!   `H⁽¹⁾_{−ν} = e^{iπν} H⁽¹⁾_ν`, `H⁽²⁾_{−ν} = e^{−iπν} H⁽²⁾_ν` (DLMF 10.4.6),
//!   which for `ν = n + ½` is the exact phase `i·(−1)ⁿ` (resp. its inverse).
//! * An integer order has no elementary form: the node stays, with
//!   `H_{−n} = (−1)ⁿ H_n`.
//! * The value is complex: `evaluate` takes the complex path (Arb's
//!   `acb_hypgeom_bessel_j/y`, principal branches).

use rug::{Integer, Rational};

use crate::kernel::{ExprId, ExprPool};
use crate::orthopoly::{mk_add, mk_mul, rational_literal, SpecialFunctionError};

use super::numeric::Kernel;
use super::MAX_EXPAND_ORDER;

/// Which Hankel function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HankelKind {
    /// `H⁽¹⁾_ν = J_ν + iY_ν` (`hankel1`).
    H1,
    /// `H⁽²⁾_ν = J_ν − iY_ν` (`hankel2`).
    H2,
}

impl HankelKind {
    /// The primitive's name.
    pub fn name(self) -> &'static str {
        self.kernel().name()
    }

    fn kernel(self) -> Kernel {
        match self {
            HankelKind::H1 => Kernel::HankelH1,
            HankelKind::H2 => Kernel::HankelH2,
        }
    }
}

/// `i^e` for `e` mod 4.
fn i_power(pool: &ExprPool, e: i64) -> ExprId {
    let i = pool.imaginary_unit();
    match e.rem_euclid(4) {
        0 => pool.integer(1),
        1 => i,
        2 => pool.integer(-1),
        _ => mk_mul(pool, vec![pool.integer(-1), i]),
    }
}

/// The elementary form of `H_ν(x)` for a half-integer literal `ν`, or `None`
/// (other orders, or `x` the literal `0`).
pub(crate) fn explicit_hankel(
    kind: HankelKind,
    nu: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, SpecialFunctionError> {
    let Some(q) = rational_literal(pool, nu) else {
        return Ok(None);
    };
    if *q.denom() != 2 || rational_literal(pool, x).is_some_and(|v| v == 0) {
        return Ok(None);
    }
    // ν = m + 1/2; for ν < 0 reflect to n = −m − 1 ≥ 0 and keep the phase.
    let m: Rational = &q - Rational::from((1, 2));
    let Some(m) = m.numer().to_i64() else {
        return Err(too_large(kind, &q));
    };
    let (n, reflect) = if m >= 0 { (m, false) } else { (-m - 1, true) };
    if n > MAX_EXPAND_ORDER {
        return Err(too_large(kind, &q));
    }
    let sgn: i64 = if kind == HankelKind::H1 { 1 } else { -1 };
    // Overall phase exponent (units of i): (−i)^{n+1} = i^{−(n+1)} for H1,
    // i^{n+1} for H2; the reflection multiplies by i^{sgn·(2n+1)}.
    let base = -sgn * (n + 1) + if reflect { sgn * (2 * n + 1) } else { 0 };
    let mut terms = Vec::with_capacity(n as usize + 1);
    for k in 0..=n {
        crate::budget::check()?;
        // c_k = (n+k)! / (k! (n−k)! 2^k)
        let num = Integer::factorial((n + k) as u32);
        let den = Integer::from(Integer::factorial(k as u32))
            * Integer::from(Integer::factorial((n - k) as u32))
            * (Integer::from(1) << (k as u32));
        let ck = Rational::from((Integer::from(num), den));
        let phase = i_power(pool, base + sgn * k);
        let xk = pool.pow(x, pool.integer(-k));
        terms.push(mk_mul(
            pool,
            vec![crate::orthopoly::rat_node(pool, &ck), phase, xk],
        ));
    }
    let i = pool.imaginary_unit();
    let arg = if kind == HankelKind::H1 {
        mk_mul(pool, vec![i, x])
    } else {
        mk_mul(pool, vec![pool.integer(-1), i, x])
    };
    let sqrt2 = pool.func("sqrt", vec![pool.integer(2)]);
    Ok(Some(mk_mul(
        pool,
        vec![
            sqrt2,
            pool.pow(crate::eval::symbols::pi_symbol(pool), pool.rational(-1, 2)),
            pool.pow(x, pool.rational(-1, 2)),
            pool.func("exp", vec![arg]),
            mk_add(pool, terms),
        ],
    )))
}

fn too_large(kind: HankelKind, q: &Rational) -> SpecialFunctionError {
    SpecialFunctionError::TooLarge {
        what: format!("{} order", kind.name()),
        value: q.numer().to_i64().unwrap_or(i64::MAX) / 2,
        max: MAX_EXPAND_ORDER,
    }
}

/// The Hankel function `kind_ν(x)`.
///
/// With `expand = true` a half-integer literal order gives the elementary form
/// (`hankel1(1/2, x) = −i·√2·e^{ix}/(√π·√x)`); otherwise the canonical node
/// (`H_{−n} = (−1)ⁿ H_n` for an integer `n`).
///
/// # Errors
///
/// `E-SPECFN-004` for an expansion past [`MAX_EXPAND_ORDER`](super::MAX_EXPAND_ORDER);
/// `E-BUDGET-*` when the active budget runs out.
pub fn hankel(
    kind: HankelKind,
    nu: ExprId,
    x: ExprId,
    expand: bool,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    if expand {
        if let Some(e) = explicit_hankel(kind, nu, x, pool)? {
            return Ok(e);
        }
    }
    Ok(super::node(kind.kernel(), vec![nu, x], pool))
}
