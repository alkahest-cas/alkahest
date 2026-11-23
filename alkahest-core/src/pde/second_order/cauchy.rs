//! The Cauchy problem `A u_xx + B u_xy + C u_yy = G`, `u(x, y₀) = f(x)`,
//! `u_y(x, y₀) = g(x)` for constant `A, B, C` with `B² − 4AC ≠ 0` and
//! `C ≠ 0` (the data line is not characteristic): d'Alembert's formula,
//! generalised.
//!
//! With `λ₁ ≠ λ₂` the roots of `C λ² + B λ + A = 0`, every solution of the
//! homogeneous equation is `F(x + λ₁ Y) + G(x + λ₂ Y)` with `Y = y − y₀`,
//! and the data give
//!
//! ```text
//! F = (Γ − λ₂ f)/(λ₁ − λ₂),   G = (λ₁ f − Γ)/(λ₁ − λ₂),   Γ = ∫ g dx.
//! ```
//!
//! For the wave equation `u_tt = c² u_xx` that is
//! `(f(x + ct) + f(x − ct))/2 + (Γ(x + ct) − Γ(x − ct))/(2c)`.  A source is
//! handled by a particular solution from the factored operator and data
//! corrected by it.

use crate::diff::undefined::view;
use crate::kernel::{ExprId, ExprPool};

use super::super::error::PdeError;
use super::super::problem::{ConditionKind, PdeCondition};
use super::first_order::{note_nonzero, Factor1};
use super::linear::Linear2;
use super::util::{d, div, free_of, free_of_all, integ, is_zero, neg, simp, sqrt, sub, subs1};

/// Cauchy data on a line `y = y₀`.
pub(crate) struct CauchyData {
    pub y0: ExprId,
    pub f: ExprId,
    pub g: ExprId,
}

/// Read `u(x, y₀) = f`, `u_y(x, y₀) = g` from the conditions; `None` when
/// they are not exactly that.
pub(crate) fn read_data(
    l: &Linear2,
    conditions: &[PdeCondition],
    pool: &ExprPool,
) -> Option<CauchyData> {
    let name = view(l.u, pool)?.name;
    let (mut f, mut g, mut y0) = (None, None, None);
    for c in conditions {
        if c.kind != ConditionKind::Initial && c.kind != ConditionKind::Boundary {
            return None;
        }
        let v = view(c.lhs, pool)?;
        if v.name != name || v.args.len() != 2 || v.args[0] != l.x {
            return None;
        }
        let at = v.args[1];
        if !free_of_all(at, &[l.x, l.y], pool) {
            return None;
        }
        if let Some(prev) = y0 {
            if prev != at {
                return None;
            }
        }
        y0 = Some(at);
        if !free_of(c.rhs, l.y, pool) {
            return None;
        }
        match v.orders.as_slice() {
            [0, 0] if f.is_none() => f = Some(c.rhs),
            [0, 1] if g.is_none() => g = Some(c.rhs),
            _ => return None,
        }
    }
    Some(CauchyData {
        y0: y0?,
        f: f?,
        g: g?,
    })
}

/// Solve the Cauchy problem; `Ok(None)` when it is outside this method.
pub(crate) fn solve(
    l: &Linear2,
    data: &CauchyData,
    conds: &mut Vec<ExprId>,
    pool: &ExprPool,
) -> Result<Option<(ExprId, Vec<String>)>, PdeError> {
    let [a, b, c, dd, e, f0] = l.coef;
    if !is_zero(dd, pool) || !is_zero(e, pool) || !is_zero(f0, pool) || !l.constant_coefficients(pool) {
        return Ok(None);
    }
    if is_zero(c, pool) {
        return Err(PdeError::CauchyDataUnsupported(format!(
            "the data line {} = {} is characteristic (no u_{}{} term)",
            pool.display(l.y),
            pool.display(data.y0),
            pool.display(l.y),
            pool.display(l.y)
        )));
    }
    let disc = simp(
        sub(pool.pow(b, pool.integer(2)), pool.mul(vec![pool.integer(4), a, c]), pool),
        pool,
    );
    if is_zero(disc, pool) {
        return Ok(None);
    }
    note_nonzero(c, conds, pool);
    note_nonzero(disc, conds, pool);
    let (x, y) = (l.x, l.y);
    let sq = sqrt(disc, pool);
    let two_c = pool.mul(vec![pool.integer(2), c]);
    let l1 = div(pool.add(vec![neg(b, pool), sq]), two_c, pool);
    let l2 = div(sub(neg(b, pool), sq, pool), two_c, pool);
    let mut steps = vec![format!(
        "characteristic speeds λ = {}, {} (roots of C λ² + B λ + A)",
        pool.display(l1),
        pool.display(l2)
    )];
    // Particular solution for a source: C(∂y − λ₁∂x)(∂y − λ₂∂x) u = G.
    let up = if l.homogeneous(pool) {
        pool.integer(0)
    } else {
        let f1 = Factor1 {
            p: simp(neg(l1, pool), pool),
            q: pool.integer(1),
            c: pool.integer(0),
        };
        let f2 = Factor1 {
            p: simp(neg(l2, pool), pool),
            q: pool.integer(1),
            c: pool.integer(0),
        };
        let g = div(l.rhs, c, pool);
        let Some(w1) = f1.particular(g, x, y, pool) else {
            return Err(PdeError::CauchyDataUnsupported(format!(
                "no particular solution found for the source {}",
                pool.display(l.rhs)
            )));
        };
        let Some(w) = f2.particular(w1, x, y, pool) else {
            return Err(PdeError::CauchyDataUnsupported(format!(
                "no particular solution found for the source {}",
                pool.display(l.rhs)
            )));
        };
        steps.push(format!("particular solution {}", pool.display(w)));
        w
    };
    let fdat = simp(sub(data.f, subs1(up, y, data.y0, pool), pool), pool);
    let gdat = simp(sub(data.g, subs1(d(up, y, pool)?, y, data.y0, pool), pool), pool);
    let gamma = integ(gdat, x, pool).ok_or_else(|| {
        PdeError::CauchyDataUnsupported(format!(
            "∫ {} d{} does not close, so d'Alembert's formula has no closed form",
            pool.display(gdat),
            pool.display(x)
        ))
    })?;
    let dl = sub(l1, l2, pool);
    let big_f = div(sub(gamma, pool.mul(vec![l2, fdat]), pool), dl, pool);
    let big_g = div(sub(pool.mul(vec![l1, fdat]), gamma, pool), dl, pool);
    let yy = sub(y, data.y0, pool);
    let at = |h: ExprId, lam: ExprId| {
        let arg = simp(pool.add(vec![x, pool.mul(vec![lam, yy])]), pool);
        subs1(h, x, arg, pool)
    };
    let u = simp(
        pool.add(vec![at(big_f, l1), at(big_g, l2), up]),
        pool,
    );
    steps.push("d'Alembert: u = F(x + λ₁(y − y₀)) + G(x + λ₂(y − y₀)) with F + G = f, λ₁F' + λ₂G' = g".into());
    Ok(Some((u, steps)))
}
