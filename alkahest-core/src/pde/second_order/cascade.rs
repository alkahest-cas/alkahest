//! The Laplace cascade for `u_xy + a u_x + b u_y + c u = s` with variable
//! coefficients (Darboux; Tsarev, arXiv cs/0501030, cs/0609075).
//!
//! With the Laplace invariants `h = a_x + ab − c` and `k = b_y + ab − c`,
//!
//! ```text
//! L = (∂x + b)(∂y + a) − h = (∂y + a)(∂x + b) − k.
//! ```
//!
//! When `h = 0` (or `k = 0`) the operator factors and the equation is solved
//! by two first-order quadratures.  Otherwise a **Laplace transformation**
//! `u₁ = (∂y + a) u` (or `u₋₁ = (∂x + b) u`) gives an equation of the same
//! shape with invariants `h₁ = 2h − k − (ln h)_xy`, `k₁ = h`; the cascade
//! runs in both directions until some invariant vanishes, and the solution
//! is carried back by `u = ((∂x + b) u₁ − s)/h` (resp. `((∂y + a)u₋₁ − s)/k`).
//! There is no a priori bound on the number of steps, so the search stops
//! after a configurable budget and refuses (`E-PDE-103`).
//!
//! The quadrature `∫ W(x, y) G(y) dy` of an arbitrary function that the
//! factored case needs is removed exactly when `W = p(x) q(y) r(x, y)` with
//! `r` polynomial in `y`: with `q G = H^{(d+1)}`, repeated integration by
//! parts gives `∫ W G dy = p · Σᵢ (−1)^i ∂ʸⁱ r · H^{(d−i)}(y)` — a general
//! solution free of integrals, in the arbitrary functions `F(x)` and `H(y)`.

use crate::kernel::{ExprId, ExprPool};

use super::super::error::PdeError;
use super::util::{
    d, div, exp, free_of, integ, is_zero, neg, poly_degree_in, simp, size, split_xy, sub, tidy,
};

/// One level of the chain: coefficients, source and invariants.
#[derive(Clone, Debug)]
struct Level {
    a: ExprId,
    b: ExprId,
    /// kept for debugging output
    #[allow(dead_code)]
    c: ExprId,
    s: ExprId,
    h: ExprId,
    k: ExprId,
}

/// `(h, k)` for `u_xy + a u_x + b u_y + c u`.
pub(crate) fn invariants(
    a: ExprId,
    b: ExprId,
    c: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Result<(ExprId, ExprId), PdeError> {
    let ab = pool.mul(vec![a, b]);
    let h = simp(pool.add(vec![d(a, x, pool)?, ab, neg(c, pool)]), pool);
    let k = simp(pool.add(vec![d(b, y, pool)?, ab, neg(c, pool)]), pool);
    Ok((h, k))
}

fn level(
    a: ExprId,
    b: ExprId,
    c: ExprId,
    s: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Result<Level, PdeError> {
    let t = |e: ExprId| compact(e, x, y, pool);
    let (a, b, c, s) = (t(a), t(b), t(c), t(s));
    let (h, k) = invariants(a, b, c, x, y, pool)?;
    let (h, k) = (t(h), t(k));
    for e in [a, b, c, s, h, k] {
        if size(e, pool) > MAX_COEFFICIENT_SIZE {
            return Err(PdeError::LaplaceBudgetExhausted(format!(
                "the Laplace invariants grew past {MAX_COEFFICIENT_SIZE} nodes after a \
                 transformation(s); the chain is abandoned"
            )));
        }
    }
    Ok(Level { a, b, c, s, h, k })
}

/// Coefficients beyond this many nodes stop the cascade (expression swell
/// is the practical face of a non-terminating chain).
const MAX_COEFFICIENT_SIZE: u64 = 1500;

/// [`tidy`], or one cancelled fraction when the coefficient is rational in
/// `x, y` — whichever is smaller.
fn compact(e: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> ExprId {
    let a = tidy(e, pool);
    match crate::poly::cancel::cancel(a, vec![x, y], pool) {
        Ok(r) => {
            let r = simp(r, pool);
            if size(r, pool) < size(a, pool) {
                r
            } else {
                a
            }
        }
        Err(_) => a,
    }
}

/// The outcome of a successful cascade.
pub(crate) struct CascadeOutcome {
    pub solution: ExprId,
    pub steps: Vec<String>,
    /// Signed number of Laplace transformations (positive: `u₁ = (∂y + a)u`
    /// direction).
    pub transforms: i32,
}

/// Run the cascade with at most `max_steps` Laplace transformations in
/// total (both directions together).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cascade(
    a: ExprId,
    b: ExprId,
    c: ExprId,
    s: ExprId,
    x: ExprId,
    y: ExprId,
    max_steps: usize,
    fresh: &mut dyn FnMut(ExprId) -> ExprId,
    pool: &ExprPool,
) -> Result<CascadeOutcome, PdeError> {
    let l0 = level(a, b, c, s, x, y, pool)?;
    let (h0, k0) = (l0.h, l0.k);
    let mut steps = vec![format!(
        "Laplace invariants h = {}, k = {}",
        pool.display(h0),
        pool.display(k0)
    )];
    let mut pos = vec![l0.clone()];
    let mut negs = vec![l0];
    let mut used = 0usize;
    let mut quad_failures: Vec<String> = Vec::new();
    loop {
        // Check the newest level of each chain.
        let p = pos.last().expect("non-empty").clone();
        if is_zero(p.h, pool) {
            match solve_h_zero(&p, x, y, fresh, pool) {
                Ok(u) => {
                    steps.push(format!(
                        "h vanishes after {} transformation(s) in the (∂y + a) direction",
                        pos.len() - 1
                    ));
                    let u = carry_back_pos(&pos, u, x, y, pool)?;
                    return Ok(CascadeOutcome {
                        solution: u,
                        steps,
                        transforms: (pos.len() - 1) as i32,
                    });
                }
                Err(e) => quad_failures.push(super::util::detail(&e)),
            }
        }
        let n = negs.last().expect("non-empty").clone();
        if is_zero(n.k, pool) {
            match solve_k_zero(&n, x, y, fresh, pool) {
                Ok(u) => {
                    steps.push(format!(
                        "k vanishes after {} transformation(s) in the (∂x + b) direction",
                        negs.len() - 1
                    ));
                    let u = carry_back_neg(&negs, u, x, y, pool)?;
                    return Ok(CascadeOutcome {
                        solution: u,
                        steps,
                        transforms: -((negs.len() - 1) as i32),
                    });
                }
                Err(e) => quad_failures.push(super::util::detail(&e)),
            }
        }
        if !quad_failures.is_empty() {
            return Err(PdeError::QuadratureOfArbitraryFunction(
                quad_failures.join("; "),
            ));
        }
        if used >= max_steps {
            return Err(PdeError::LaplaceBudgetExhausted(format!(
                "no Laplace invariant vanished within {max_steps} transformation(s); \
                 h = {}, k = {} (the chain may be infinite: termination has no a priori bound)",
                pool.display(h0),
                pool.display(k0)
            )));
        }
        // Advance the chain whose invariant expression is smaller (cheap
        // heuristic), alternating otherwise.
        let advance_pos = used % 2 == 0;
        if advance_pos {
            let nx = step_pos(&p, x, y, pool)?;
            pos.push(nx);
        } else {
            let nx = step_neg(&n, x, y, pool)?;
            negs.push(nx);
        }
        used += 1;
    }
}

/// `u₁ = (∂y + a) u`: `a₁ = a − h_y/h`, `b₁ = b`, `c₁ = b_y + a₁ b − h`,
/// `s₁ = s_y + a₁ s`.
fn step_pos(l: &Level, x: ExprId, y: ExprId, pool: &ExprPool) -> Result<Level, PdeError> {
    let a1 = simp(sub(l.a, div(d(l.h, y, pool)?, l.h, pool), pool), pool);
    let b1 = l.b;
    let c1 = simp(
        pool.add(vec![
            d(l.b, y, pool)?,
            pool.mul(vec![a1, l.b]),
            neg(l.h, pool),
        ]),
        pool,
    );
    let s1 = simp(
        pool.add(vec![d(l.s, y, pool)?, pool.mul(vec![a1, l.s])]),
        pool,
    );
    level(a1, b1, c1, s1, x, y, pool)
}

/// `u₋₁ = (∂x + b) u`: `b₋₁ = b − k_x/k`, `a₋₁ = a`, `c₋₁ = a_x + a b₋₁ − k`,
/// `s₋₁ = s_x + b₋₁ s`.
fn step_neg(l: &Level, x: ExprId, y: ExprId, pool: &ExprPool) -> Result<Level, PdeError> {
    let b1 = simp(sub(l.b, div(d(l.k, x, pool)?, l.k, pool), pool), pool);
    let a1 = l.a;
    let c1 = simp(
        pool.add(vec![
            d(l.a, x, pool)?,
            pool.mul(vec![l.a, b1]),
            neg(l.k, pool),
        ]),
        pool,
    );
    let s1 = simp(
        pool.add(vec![d(l.s, x, pool)?, pool.mul(vec![b1, l.s])]),
        pool,
    );
    level(a1, b1, c1, s1, x, y, pool)
}

/// `u_n = ((∂x + b_n) u_{n+1} − s_n) / h_n`, from the top of the chain down.
fn carry_back_pos(
    chain: &[Level],
    top: ExprId,
    x: ExprId,
    _y: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let mut u = top;
    for l in chain[..chain.len() - 1].iter().rev() {
        let num = pool.add(vec![d(u, x, pool)?, pool.mul(vec![l.b, u]), neg(l.s, pool)]);
        u = tidy(div(num, l.h, pool), pool);
    }
    Ok(simp(u, pool))
}

/// `u_n = ((∂y + a_n) u_{n−1} − s_n) / k_n`.
fn carry_back_neg(
    chain: &[Level],
    top: ExprId,
    _x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let mut u = top;
    for l in chain[..chain.len() - 1].iter().rev() {
        let num = pool.add(vec![d(u, y, pool)?, pool.mul(vec![l.a, u]), neg(l.s, pool)]);
        u = tidy(div(num, l.k, pool), pool);
    }
    Ok(simp(u, pool))
}

/// `h = 0`: `(∂x + b)(∂y + a) u = s`.
fn solve_h_zero(
    l: &Level,
    x: ExprId,
    y: ExprId,
    fresh: &mut dyn FnMut(ExprId) -> ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    factored_solve(l.a, l.b, l.s, x, y, fresh, pool)
}

/// `k = 0`: `(∂y + a)(∂x + b) u = s` — the `h = 0` case with `x ↔ y`,
/// `a ↔ b`.
fn solve_k_zero(
    l: &Level,
    x: ExprId,
    y: ExprId,
    fresh: &mut dyn FnMut(ExprId) -> ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    factored_solve(l.b, l.a, l.s, y, x, fresh, pool)
}

/// General solution of `(∂x + b)(∂y + a) u = s`:
/// `v = u_y + a u` solves `v_x + b v = s`, so `v = (G(y) + ∫ μ s dx)/μ`
/// with `μ = e^{∫b dx}`; then `u = (F(x) + ∫ ν v dy)/ν` with
/// `ν = e^{∫a dy}`.
pub(crate) fn factored_solve(
    a: ExprId,
    b: ExprId,
    s: ExprId,
    x: ExprId,
    y: ExprId,
    fresh: &mut dyn FnMut(ExprId) -> ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let quad = |what: &str, e: ExprId, v: ExprId| -> Result<ExprId, PdeError> {
        integ(e, v, pool).ok_or_else(|| {
            PdeError::QuadratureOfArbitraryFunction(format!(
                "the quadrature ∫ {} d{} ({what}) does not close",
                pool.display(e),
                pool.display(v)
            ))
        })
    };
    let ib = quad("integrating factor", b, x)?;
    let ia = quad("integrating factor", a, y)?;
    let mu = exp(ib, pool);
    let nu = exp(ia, pool);
    // W = ν/μ multiplies G(y) under ∫ dy.
    let w = simp(pool.mul(vec![nu, exp(neg(ib, pool), pool)]), pool);
    let g_part = arbitrary_quadrature(w, x, y, fresh, pool)?;
    let mut parts = vec![fresh(x), g_part];
    if !is_zero(s, pool) {
        let inner = quad("source", simp(pool.mul(vec![mu, s]), pool), x)?;
        let v_s = simp(pool.mul(vec![inner, exp(neg(ib, pool), pool)]), pool);
        let outer = quad("source", simp(pool.mul(vec![nu, v_s]), pool), y)?;
        parts.push(outer);
    }
    Ok(simp(
        pool.mul(vec![pool.add(parts), exp(neg(ia, pool), pool)]),
        pool,
    ))
}

/// `∫ W(x, y) G(y) dy` with `G` arbitrary, written without an integral as
/// `p(x) Σᵢ (−1)^i ∂ʸⁱ r · H^{(d−i)}(y)` when `W = p(x) q(y) r(x, y)` with
/// `r` a polynomial of degree `d` in `y` (the new arbitrary function `H` is
/// defined by `q G = H^{(d+1)}`).
fn arbitrary_quadrature(
    w: ExprId,
    x: ExprId,
    y: ExprId,
    fresh: &mut dyn FnMut(ExprId) -> ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let (p, _q, r) = split_xy(w, x, y, pool);
    let deg = if free_of(r, y, pool) {
        Some(0)
    } else {
        poly_degree_in(r, y, 6, pool)
    };
    let Some(deg) = deg else {
        return Err(PdeError::QuadratureOfArbitraryFunction(format!(
            "∫ {} · G(y) dy with G arbitrary has no integral-free form \
             (the weight does not split as p(x)·q(y)·polynomial in y)",
            pool.display(w)
        )));
    };
    let h = fresh(y);
    let mut terms = Vec::new();
    let mut ri = r;
    for i in 0..=deg {
        // H^{(deg − i)}(y)
        let mut hd = h;
        for _ in 0..(deg - i) {
            hd = crate::diff::diff(hd, y, pool)
                .map_err(|e| PdeError::NotRationalInJets(e.to_string()))?
                .value;
        }
        let sign = if i % 2 == 0 { 1 } else { -1 };
        terms.push(pool.mul(vec![pool.integer(sign), ri, hd]));
        ri = d(ri, y, pool)?;
    }
    Ok(simp(pool.mul(vec![p, pool.add(terms)]), pool))
}
