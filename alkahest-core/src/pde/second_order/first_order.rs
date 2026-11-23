//! Self-contained first-order helper for the factored operators: the kernel
//! and a particular solution of `p ∂x + q ∂y + c` with `p, q, c` free of
//! `x, y`, by characteristics.  (The general first-order solver is M1's.)

use crate::kernel::{ExprId, ExprPool};

use super::super::error::PdeError;
use super::util::{d, div, exp, integ, is_nonzero, is_zero, neg, simp, sub, subs1};

/// `p ∂x + q ∂y + c`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Factor1 {
    pub p: ExprId,
    pub q: ExprId,
    pub c: ExprId,
}

impl Factor1 {
    /// Apply to `w`.
    pub fn apply(&self, w: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
        Ok(simp(
            pool.add(vec![
                pool.mul(vec![self.p, d(w, x, pool)?]),
                pool.mul(vec![self.q, d(w, y, pool)?]),
                pool.mul(vec![self.c, w]),
            ]),
            pool,
        ))
    }

    /// Whether the `x`-coefficient is taken as nonzero.
    fn along_x(&self, pool: &ExprPool) -> bool {
        !is_zero(self.p, pool)
    }

    /// The characteristic invariant (`x − (p/q) y`, or `y` / `x` when a
    /// coefficient vanishes) and the multiplier `e^{−c x/p}` (or
    /// `e^{−c y/q}`), so that the kernel is `multiplier · F(invariant)`.
    /// The side conditions (`p ≠ 0`, `q ≠ 0` when not provable) are pushed
    /// to `conds`.
    pub fn kernel(
        &self,
        x: ExprId,
        y: ExprId,
        conds: &mut Vec<ExprId>,
        pool: &ExprPool,
    ) -> (ExprId, ExprId) {
        let pz = is_zero(self.p, pool);
        let qz = is_zero(self.q, pool);
        let invariant = if qz {
            y
        } else if pz {
            x
        } else {
            note_nonzero(self.q, conds, pool);
            simp(sub(x, pool.mul(vec![div(self.p, self.q, pool), y]), pool), pool)
        };
        let mult = if !pz {
            note_nonzero(self.p, conds, pool);
            exp(neg(pool.mul(vec![div(self.c, self.p, pool), x]), pool), pool)
        } else {
            note_nonzero(self.q, conds, pool);
            exp(neg(pool.mul(vec![div(self.c, self.q, pool), y]), pool), pool)
        };
        (invariant, mult)
    }

    /// The variable `v` with `(p∂x + q∂y)(v) ≠ 0` used for a repeated factor:
    /// `L(v·K) = (p or q)·K` when `L K = 0`.
    pub fn repeat_var(&self, x: ExprId, y: ExprId, pool: &ExprPool) -> ExprId {
        if self.along_x(pool) {
            x
        } else {
            y
        }
    }

    /// A particular solution of `(p ∂x + q ∂y + c) w = g`, integrating along
    /// the characteristics.  `None` when a quadrature does not close or the
    /// candidate does not check.
    pub fn particular(&self, g: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> Option<ExprId> {
        if is_zero(g, pool) {
            return Some(pool.integer(0));
        }
        let w = if self.along_x(pool) {
            let (p, q, c) = (self.p, self.q, self.c);
            // s = x, r = q x − p y  ⇒  y = (q s − r)/p.
            let r = pool.symbol("__alkahest_pde2_r", crate::kernel::Domain::Real);
            let y_of = div(sub(pool.mul(vec![q, x]), r, pool), p, pool);
            let g_sr = simp(subs1(g, y, y_of, pool), pool);
            let e = exp(pool.mul(vec![div(c, p, pool), x]), pool);
            let integrand = simp(pool.mul(vec![e, g_sr, pool.pow(p, pool.integer(-1))]), pool);
            let i = integ(integrand, x, pool)?;
            let w_sr = simp(pool.mul(vec![exp(neg(pool.mul(vec![div(c, p, pool), x]), pool), pool), i]), pool);
            let r_xy = sub(pool.mul(vec![q, x]), pool.mul(vec![p, y]), pool);
            simp(subs1(w_sr, r, r_xy, pool), pool)
        } else {
            let (q, c) = (self.q, self.c);
            if !is_nonzero(q, pool) && is_zero(q, pool) {
                return None;
            }
            let e = exp(pool.mul(vec![div(c, q, pool), y]), pool);
            let integrand = simp(pool.mul(vec![e, g, pool.pow(q, pool.integer(-1))]), pool);
            let i = integ(integrand, y, pool)?;
            simp(pool.mul(vec![exp(neg(pool.mul(vec![div(c, q, pool), y]), pool), pool), i]), pool)
        };
        let check = self.apply(w, x, y, pool).ok()?;
        is_zero(sub(check, g, pool), pool).then_some(w)
    }
}

/// Record `e ≠ 0` as a side condition unless it is provably nonzero.
pub(crate) fn note_nonzero(e: ExprId, conds: &mut Vec<ExprId>, pool: &ExprPool) {
    if is_nonzero(e, pool) {
        return;
    }
    let c = pool.predicate(
        crate::kernel::expr::PredicateKind::Ne,
        vec![simp(e, pool), pool.integer(0)],
    );
    if !conds.contains(&c) {
        conds.push(c);
    }
}
