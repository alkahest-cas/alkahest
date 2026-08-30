//! Variation of parameters and reduction of order for linear ODEs.
//!
//! Both are pure quadrature: given a fundamental system of the homogeneous
//! equation, everything else is an integral.  Keeping them here — independent
//! of *how* the basis was found — is what lets the constant-coefficient,
//! Euler–Cauchy and general variable-coefficient classes share one particular
//! solution routine instead of each growing its own ansatz.
//!
//! # Variation of parameters
//!
//! For `y⁽ⁿ⁾ + p_{n−1} y⁽ⁿ⁻¹⁾ + … + p₀ y = g` with fundamental system
//! `y₁ … yₙ`,
//!
//! ```text
//! y_p = Σₖ yₖ · ∫ (Wₖ / W) · g dx
//! ```
//!
//! where `W` is the Wronskian determinant and `Wₖ` is `W` with its `k`-th
//! column replaced by `(0, …, 0, 1)ᵀ`.  At `n = 2` this is the textbook
//! `y_p = −y₁∫(y₂g/W) dx + y₂∫(y₁g/W) dx`.
//!
//! **Constants.** The antiderivatives are taken *without* a constant, so `y_p`
//! is a particular solution and the general solution is `Σ Cₖ yₖ + y_p` with
//! exactly the `n` constants the homogeneous part introduced.  Adding a
//! constant to one of the integrals would only shift `y_p` by a multiple of a
//! `yₖ`, which the `Cₖ` already span — the danger is the opposite one, of
//! letting a class allocate a constant *here* as well as in the homogeneous
//! solution, so this function allocates none.
//!
//! # Reduction of order
//!
//! Given one solution `y₁` of `y'' + P y' + Q y = 0`,
//!
//! ```text
//! y₂ = y₁ · ∫ e^{−∫P dx} / y₁² dx
//! ```
//!
//! is a second, independent solution.  This is the only route in `dsolve` to a
//! second-order equation with non-constant coefficients that is not
//! Euler–Cauchy; the first solution is found by a small ansatz search.

use super::{contains, ddx, div, exp_of, integrate_or_decline, is_zero, normalized, simp, sub};
use crate::kernel::{ExprId, ExprPool};
use crate::ode::dsolve::DsolveError;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Determinants over `ExprId`
// ---------------------------------------------------------------------------

/// Symbolic determinant by Laplace expansion along the first row.
///
/// The matrices here are Wronskians of a fundamental system, so `n ≤ 4` in
/// every caller and the `O(n!)` expansion is cheaper than setting up an
/// exact-division fraction-free elimination over `ExprId`.
fn det(m: &[Vec<ExprId>], pool: &ExprPool) -> ExprId {
    let n = m.len();
    match n {
        0 => pool.integer(1_i32),
        1 => m[0][0],
        2 => sub(
            simp(pool.mul(vec![m[0][0], m[1][1]]), pool),
            simp(pool.mul(vec![m[0][1], m[1][0]]), pool),
            pool,
        ),
        _ => {
            let mut terms = Vec::with_capacity(n);
            for j in 0..n {
                if is_zero(m[0][j], pool) {
                    continue;
                }
                let minor: Vec<Vec<ExprId>> = m[1..]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter(|(c, _)| *c != j)
                            .map(|(_, &v)| v)
                            .collect()
                    })
                    .collect();
                let sign = if j % 2 == 0 { 1_i32 } else { -1_i32 };
                terms.push(pool.mul(vec![pool.integer(sign), m[0][j], det(&minor, pool)]));
            }
            simp(pool.add(terms), pool)
        }
    }
}

// ---------------------------------------------------------------------------
// Variation of parameters
// ---------------------------------------------------------------------------

/// Particular solution of `y⁽ⁿ⁾ + … = g` for the fundamental system `basis`.
///
/// `g` must already be divided through by the leading coefficient.  Returns
/// `Ok(None)` when the Wronskian vanishes (the supplied functions are not a
/// fundamental system), and `Err(Unsupported)` when a required integral does
/// not close — never an unevaluated integral.
pub(crate) fn variation_of_parameters(
    basis: &[ExprId],
    g: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, DsolveError> {
    let n = basis.len();
    if n == 0 {
        return Ok(None);
    }
    // Derivative table: d[i][j] = dⁱ/dxⁱ basis[j], for i = 0..n.
    let mut d: Vec<Vec<ExprId>> = Vec::with_capacity(n);
    d.push(basis.to_vec());
    for i in 1..n {
        let prev = d[i - 1].clone();
        let mut row = Vec::with_capacity(n);
        for b in prev {
            row.push(ddx(b, x, pool)?);
        }
        d.push(row);
    }

    // Deliberately *not* normalised: `integrate_or_decline` tries the raw and
    // the normalised spellings of the whole integrand, and for a `{cos, sin}`
    // basis the raw `…/(cos²x + sin²x)` is currently the one the engine
    // closes.  Normalising here would throw that spelling away.
    let w = det(&d, pool);
    if is_zero(w, pool) {
        return Ok(None);
    }

    let mut terms = Vec::with_capacity(n);
    for k in 0..n {
        // Wₖ: replace column k by the last unit vector.
        let mut mk = d.clone();
        for (i, row) in mk.iter_mut().enumerate() {
            row[k] = pool.integer(i32::from(i + 1 == n));
        }
        let wk = det(&mk, pool);
        if is_zero(wk, pool) {
            continue;
        }
        // Two constructions of the same quotient: divided by the Wronskian as
        // the determinant produced it, and by its normalised form.  Neither
        // dominates — `{cos, sin}` wants the first, `{e^x, e^{−x}}` the
        // second — so both are offered.
        let num = simp(pool.mul(vec![wk, g]), pool);
        let raw = div(num, w, pool);
        let norm = div(num, normalized(w, pool), pool);
        let anti = super::integrate_first_of(&[raw, norm], x, pool)?;
        terms.push(pool.mul(vec![basis[k], anti]));
    }
    Ok(Some(simp(pool.add(terms), pool)))
}

// ---------------------------------------------------------------------------
// Reduction of order
// ---------------------------------------------------------------------------

/// Second solution `y₂ = y₁ ∫ e^{−∫P dx} / y₁² dx` of `y'' + P y' + Q y = 0`.
///
/// Returns `Ok(None)` if the result is not independent of `y₁`.
pub(crate) fn reduction_of_order(
    y1: ExprId,
    p: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, DsolveError> {
    if is_zero(y1, pool) {
        return Ok(None);
    }
    let int_p = integrate_or_decline(p, x, pool)?;
    let neg = simp(pool.mul(vec![pool.integer(-1_i32), int_p]), pool);
    let mu = exp_of(neg, pool);
    let y1sq = simp(pool.pow(y1, pool.integer(2_i32)), pool);
    let integrand = div(mu, y1sq, pool);
    let v = integrate_or_decline(integrand, x, pool)?;
    if is_zero(v, pool) {
        return Ok(None);
    }
    let y2 = simp(pool.mul(vec![y1, v]), pool);
    // Independence: y₂/y₁ must actually depend on x.
    let ratio = simp(div(y2, y1, pool), pool);
    if !contains(ratio, x, pool) {
        return Ok(None);
    }
    Ok(Some(y2))
}

// ---------------------------------------------------------------------------
// First-solution ansatz for a general second-order linear equation
// ---------------------------------------------------------------------------

/// Candidate shapes tried as `y₁` for `y'' + P y' + Q y = 0`.
///
/// Deliberately a short, fixed list rather than a search: the point of
/// reduction of order here is to reach the *quadrature*, and the textbook
/// variable-coefficient equations that are solvable in closed form at all
/// have a first solution of one of these shapes (`x^m` for Euler-like and
/// Legendre-like equations, `e^{rx}` for equations with a polynomial
/// coefficient in front of `y''`).  Anything else is declined, which is the
/// honest answer — `dsolve` does not claim to decide solvability.
fn y1_candidates(x: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = vec![pool.integer(1_i32), x];
    for m in [2_i32, 3, -1, -2] {
        out.push(simp(pool.pow(x, pool.integer(m)), pool));
    }
    out.push(simp(pool.pow(x, pool.rational(1_i32, 2_i32)), pool));
    for r in [1_i32, -1, 2, -2] {
        let rx = simp(pool.mul(vec![pool.integer(r), x]), pool);
        out.push(simp(pool.func("exp", vec![rx]), pool));
    }
    out.push(pool.func("sin", vec![x]));
    out.push(pool.func("cos", vec![x]));
    out.push(pool.func("log", vec![x]));
    out
}

/// Find one solution of `y'' + P y' + Q y = 0` from [`y1_candidates`].
pub(crate) fn find_first_solution(
    p: ExprId,
    q: ExprId,
    x: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, DsolveError> {
    for cand in y1_candidates(x, pool) {
        if is_zero(cand, pool) {
            continue;
        }
        let d1 = ddx(cand, x, pool)?;
        let d2 = ddx(d1, x, pool)?;
        let residual = normalized(
            simp(
                pool.add(vec![
                    d2,
                    simp(pool.mul(vec![p, d1]), pool),
                    simp(pool.mul(vec![q, cand]), pool),
                ]),
                pool,
            ),
            pool,
        );
        if is_zero(residual, pool) || numerically_zero(residual, x, pool) {
            return Ok(Some(cand));
        }
    }
    Ok(None)
}

/// `expr ≈ 0` at several positive sample points (poles are skipped).
///
/// Only used to *accept an ansatz*, never to accept a final answer: whatever
/// this admits still has to survive `residual_is_zero` on the original
/// equation.
fn numerically_zero(expr: ExprId, x: ExprId, pool: &ExprPool) -> bool {
    let mut checked = 0usize;
    for xv in [0.37_f64, 0.81, 1.29, 1.73, 2.11] {
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        env.insert(x, xv);
        match super::verify::eval(expr, &env, pool) {
            Some(v) if v.is_finite() => {
                if v.abs() > 1e-9 {
                    return false;
                }
                checked += 1;
            }
            Some(_) => {}
            None => return false,
        }
    }
    checked >= 3
}
