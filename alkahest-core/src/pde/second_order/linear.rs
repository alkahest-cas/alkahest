//! A linear second-order equation in two independent variables, read off as
//! its coefficients: `A u_xx + B u_xy + C u_yy + D u_x + E u_y + F u = G`.

use crate::kernel::{ExprId, ExprPool};

use super::super::classify::{classify_pde, Linearity, PdeInfo};
use super::super::error::PdeError;
use super::super::jet::{decode_unknown, JetSpace};
use super::util::{d, free_of_all, is_zero, neg, simp, subs1};

/// `A u_xx + B u_xy + C u_yy + D u_x + E u_y + F u = G` in the unknown
/// `u(x, y)`.
#[derive(Clone, Debug)]
pub(crate) struct Linear2 {
    pub x: ExprId,
    pub y: ExprId,
    /// The applied unknown `u(x, y)`.
    pub u: ExprId,
    /// `[A, B, C, D, E, F]`.
    pub coef: [ExprId; 6],
    /// The right-hand side `G` (the source).
    pub rhs: ExprId,
}

impl Linear2 {
    pub fn a(&self) -> ExprId {
        self.coef[0]
    }
    pub fn b(&self) -> ExprId {
        self.coef[1]
    }
    pub fn c(&self) -> ExprId {
        self.coef[2]
    }
    pub fn dd(&self) -> ExprId {
        self.coef[3]
    }
    pub fn e(&self) -> ExprId {
        self.coef[4]
    }
    pub fn f(&self) -> ExprId {
        self.coef[5]
    }

    /// Read the equation off `equation = 0`.
    ///
    /// # Errors
    ///
    /// `E-PDE-100` unless the equation is linear, of order two, in exactly two
    /// independent variables; the M0 errors of [`classify_pde`] otherwise.
    pub fn from_equation(
        equation: ExprId,
        unknown: ExprId,
        pool: &ExprPool,
    ) -> Result<(Linear2, PdeInfo), PdeError> {
        let info = classify_pde(equation, unknown, pool)?;
        let v = decode_unknown(unknown, pool)?;
        if v.args.len() != 2 || info.order != 2 || info.linearity != Linearity::Linear {
            return Err(PdeError::NotSecondOrderLinear2D(format!(
                "{} = 0 is {} of order {} in {} independent variable(s); \
                 the second-order solvers need a linear equation of order two in two variables",
                pool.display(equation),
                info.linearity.as_str(),
                info.order,
                v.args.len()
            )));
        }
        let space = JetSpace::new(&v.args, &[unknown], 2, pool)?;
        let f = simp(space.to_jets(equation, pool)?, pool);
        let alphas: [[u32; 2]; 6] = [[2, 0], [1, 1], [0, 2], [1, 0], [0, 1], [0, 0]];
        let mut coef = [pool.integer(0); 6];
        let mut rest = f;
        for (k, alpha) in alphas.iter().enumerate() {
            let j = space.jet(0, alpha, pool)?;
            coef[k] = d(f, j, pool)?;
            rest = subs1(rest, j, pool.integer(0), pool);
        }
        let rhs = simp(neg(rest, pool), pool);
        Ok((
            Linear2 {
                x: v.args[0],
                y: v.args[1],
                u: unknown,
                coef,
                rhs,
            },
            info,
        ))
    }

    /// Whether every coefficient is free of `x` and `y`.
    pub fn constant_coefficients(&self, pool: &ExprPool) -> bool {
        self.coef
            .iter()
            .all(|&c| free_of_all(c, &[self.x, self.y], pool))
    }

    pub fn homogeneous(&self, pool: &ExprPool) -> bool {
        is_zero(self.rhs, pool)
    }

    /// The equation `L u − G` as an expression in the unknown's derivatives.
    pub fn equation(&self, pool: &ExprPool) -> Result<ExprId, PdeError> {
        let lu = self.apply_unknown(pool)?;
        Ok(pool.add(vec![lu, neg(self.rhs, pool)]))
    }

    fn apply_unknown(&self, pool: &ExprPool) -> Result<ExprId, PdeError> {
        let dd = |e, v| {
            crate::diff::diff(e, v, pool)
                .map(|r| r.value)
                .map_err(|err| PdeError::NotRationalInJets(err.to_string()))
        };
        let (x, y, u) = (self.x, self.y, self.u);
        let ux = dd(u, x)?;
        let uy = dd(u, y)?;
        let parts = [dd(ux, x)?, dd(ux, y)?, dd(uy, y)?, ux, uy, u];
        Ok(pool.add(
            self.coef
                .iter()
                .zip(parts)
                .filter(|(&c, _)| c != pool.integer(0))
                .map(|(&c, p)| pool.mul(vec![c, p]))
                .collect(),
        ))
    }

    /// The same operator with the roles of `x` and `y` exchanged.
    pub fn swapped(&self) -> Linear2 {
        let [a, b, c, dd, e, f] = self.coef;
        Linear2 {
            x: self.y,
            y: self.x,
            u: self.u,
            coef: [c, b, a, e, dd, f],
            rhs: self.rhs,
        }
    }
}
