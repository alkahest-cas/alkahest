//! Exact linear and mixed-integer programming over ℚ, with certificates.
//!
//! [`linprog`] solves
//!
//! ```text
//!   minimise (or maximise)  c · x
//!   subject to              A_ub x ≤ b_ub,   A_eq x = b_eq,   l ≤ x ≤ u
//! ```
//!
//! in exact rational arithmetic (`rug::Rational`) with a two-phase dense
//! simplex under **Bland's rule**, which cannot cycle; in exact arithmetic that
//! makes termination unconditional (no tolerances, no degeneracy stalls). Bounds
//! may be infinite on either side; the default bound of every variable is
//! `[0, ∞)`, as in `scipy.optimize.linprog`.
//!
//! Every outcome carries a certificate that [`verify`](OptimalCertificate::verify)
//! re-checks exactly against the caller's data, and [`linprog`] runs that check
//! itself before returning — a certificate that fails is reported as
//! `E-LP-006` and withheld:
//!
//! * **optimal** — a primal point `x` and dual multipliers `y_ub`, `y_eq`,
//!   `z_lower`, `z_upper` with
//!   `c = A_ubᵀ y_ub + A_eqᵀ y_eq + z_lower + z_upper`, the sign conditions of
//!   the sense (for a minimisation `y_ub ≤ 0`, `z_lower ≥ 0`, `z_upper ≤ 0`; all
//!   reversed for a maximisation — the same convention as SciPy's HiGHS
//!   `marginals`), exact complementary slackness, and equal primal and dual
//!   objectives. Weak duality then proves `x` optimal.
//! * **infeasible** — a Farkas vector `(y_ub ≥ 0, y_eq, w_lower ≥ 0,
//!   w_upper ≥ 0)` with `A_ubᵀ y_ub + A_eqᵀ y_eq − w_lower + w_upper = 0` and
//!   `b_ub·y_ub + b_eq·y_eq − l·w_lower + u·w_upper < 0`: summing the
//!   constraints with those weights gives `0 < 0`.
//! * **unbounded** — a feasible point `x₀` and a direction `d` with
//!   `A_ub d ≤ 0`, `A_eq d = 0`, `d_j ≥ 0` where `l_j` is finite, `d_j ≤ 0`
//!   where `u_j` is finite, and `c · d < 0` (`> 0` when maximising).
//!
//! [`milp`] adds integrality by branch and bound on top (depth-first until a
//! first incumbent, then best-bound), honouring the active
//! [`crate::budget::Budget`] at every node and pivot; it either proves its
//! incumbent optimal (the search closed every node, so the gap is exactly 0),
//! proves infeasibility, proves unboundedness with an integer point and an
//! integral ray, or refuses (`E-LP-005`, or the budget's `E-BUDGET-*`) with the
//! best incumbent and open bound it had.
//!
//! [`program_from_exprs`] builds a programme from alkahest expressions:
//! an affine objective and `≤ / ≥ / =` predicates, with rational coefficients.
//!
//! This module is independent of [`crate::real::sos::lp`], whose solver is
//! frozen for the positivity certificates reproduced against it and returns
//! neither duals, Farkas vectors nor rays.

mod error;
mod expr;
mod milp;
mod simplex;
#[cfg(test)]
mod tests;

pub use error::{Incumbent, LinprogError};
pub use expr::{linear_form, program_from_exprs};
pub use milp::{milp, MilpOptions, MilpOutcome, MilpSolution};

use rug::Rational;

/// Direction of optimisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    /// Minimise `c · x`.
    Minimize,
    /// Maximise `c · x`.
    Maximize,
}

/// The bounds of one decision variable; `None` is an infinite bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarBound {
    /// Lower bound, `None` for `−∞`.
    pub lower: Option<Rational>,
    /// Upper bound, `None` for `+∞`.
    pub upper: Option<Rational>,
}

impl VarBound {
    /// `[0, ∞)`, the default.
    pub fn nonnegative() -> Self {
        VarBound {
            lower: Some(Rational::new()),
            upper: None,
        }
    }

    /// `(−∞, ∞)`.
    pub fn free() -> Self {
        VarBound {
            lower: None,
            upper: None,
        }
    }

    /// `[lower, upper]` with either side possibly infinite.
    pub fn new(lower: Option<Rational>, upper: Option<Rational>) -> Self {
        VarBound { lower, upper }
    }
}

/// A linear programme `opt c·x  s.t.  A_ub x ≤ b_ub, A_eq x = b_eq, l ≤ x ≤ u`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearProgram {
    /// Objective coefficients, one per variable.
    pub c: Vec<Rational>,
    /// Inequality rows (`≤`).
    pub a_ub: Vec<Vec<Rational>>,
    /// Inequality right-hand sides.
    pub b_ub: Vec<Rational>,
    /// Equality rows.
    pub a_eq: Vec<Vec<Rational>>,
    /// Equality right-hand sides.
    pub b_eq: Vec<Rational>,
    /// One bound per variable.
    pub bounds: Vec<VarBound>,
    /// Minimise or maximise.
    pub sense: Sense,
}

impl LinearProgram {
    /// `minimise c · x` with every variable in `[0, ∞)` and no constraints.
    pub fn new(c: Vec<Rational>) -> Self {
        let n = c.len();
        LinearProgram {
            c,
            a_ub: Vec::new(),
            b_ub: Vec::new(),
            a_eq: Vec::new(),
            b_eq: Vec::new(),
            bounds: vec![VarBound::nonnegative(); n],
            sense: Sense::Minimize,
        }
    }

    /// Number of decision variables.
    pub fn nvars(&self) -> usize {
        self.c.len()
    }

    /// Check that every row, right-hand side and bound list has the right
    /// length (`E-LP-001`).
    pub fn validate(&self) -> Result<(), LinprogError> {
        let n = self.nvars();
        let shape = |detail: String| Err(LinprogError::Shape { detail });
        if self.a_ub.len() != self.b_ub.len() {
            return shape(format!(
                "A_ub has {} rows but b_ub has {} entries",
                self.a_ub.len(),
                self.b_ub.len()
            ));
        }
        if self.a_eq.len() != self.b_eq.len() {
            return shape(format!(
                "A_eq has {} rows but b_eq has {} entries",
                self.a_eq.len(),
                self.b_eq.len()
            ));
        }
        for (name, rows) in [("A_ub", &self.a_ub), ("A_eq", &self.a_eq)] {
            for (i, row) in rows.iter().enumerate() {
                if row.len() != n {
                    return shape(format!(
                        "{name} row {i} has {} entries, but c has {n}",
                        row.len()
                    ));
                }
            }
        }
        if self.bounds.len() != n {
            return shape(format!(
                "{} bounds given for {n} variables",
                self.bounds.len()
            ));
        }
        Ok(())
    }

    /// `c · x`.
    pub fn objective_at(&self, x: &[Rational]) -> Rational {
        dot(&self.c, x)
    }

    /// `true` when `x` satisfies every constraint and bound exactly.
    pub fn is_feasible(&self, x: &[Rational]) -> bool {
        self.feasibility_violation(x).is_none()
    }

    fn feasibility_violation(&self, x: &[Rational]) -> Option<String> {
        if x.len() != self.nvars() {
            return Some(format!("point has {} entries", x.len()));
        }
        for (i, (row, b)) in self.a_ub.iter().zip(&self.b_ub).enumerate() {
            if dot(row, x) > *b {
                return Some(format!("inequality row {i} violated"));
            }
        }
        for (i, (row, b)) in self.a_eq.iter().zip(&self.b_eq).enumerate() {
            if dot(row, x) != *b {
                return Some(format!("equality row {i} violated"));
            }
        }
        for (j, (bd, v)) in self.bounds.iter().zip(x).enumerate() {
            if bd.lower.as_ref().is_some_and(|l| v < l) {
                return Some(format!("lower bound of variable {j} violated"));
            }
            if bd.upper.as_ref().is_some_and(|u| v > u) {
                return Some(format!("upper bound of variable {j} violated"));
            }
        }
        None
    }
}

/// Outcome of [`linprog`]: each status carries its certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LpOutcome {
    /// An optimal vertex with its dual certificate.
    Optimal(OptimalCertificate),
    /// No point satisfies the constraints; the Farkas vector proves it.
    Infeasible(FarkasCertificate),
    /// The objective is unbounded on the feasible set; the ray proves it.
    Unbounded(UnboundedCertificate),
}

impl LpOutcome {
    /// Re-check this outcome's certificate exactly against `lp`.
    pub fn verify(&self, lp: &LinearProgram) -> Result<(), String> {
        match self {
            LpOutcome::Optimal(c) => c.verify(lp),
            LpOutcome::Infeasible(c) => c.verify(lp),
            LpOutcome::Unbounded(c) => c.verify(lp),
        }
    }
}

/// An optimal point together with a dual solution proving it optimal.
///
/// Stationarity `c = A_ubᵀ y_ub + A_eqᵀ y_eq + z_lower + z_upper` holds
/// exactly; for a minimisation `y_ub ≤ 0`, `z_lower ≥ 0`, `z_upper ≤ 0`, for a
/// maximisation the signs are reversed. Multipliers of infinite bounds are 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimalCertificate {
    /// The optimal point.
    pub x: Vec<Rational>,
    /// The optimal value `c · x`.
    pub objective: Rational,
    /// Multipliers of the inequality rows.
    pub y_ub: Vec<Rational>,
    /// Multipliers of the equality rows.
    pub y_eq: Vec<Rational>,
    /// Multipliers of the lower bounds.
    pub z_lower: Vec<Rational>,
    /// Multipliers of the upper bounds.
    pub z_upper: Vec<Rational>,
}

impl OptimalCertificate {
    /// Exact check: primal feasibility, dual feasibility (stationarity and
    /// signs), complementary slackness and equality of the two objectives.
    pub fn verify(&self, lp: &LinearProgram) -> Result<(), String> {
        let n = lp.nvars();
        if let Some(v) = lp.feasibility_violation(&self.x) {
            return Err(format!("primal point infeasible: {v}"));
        }
        if self.objective != lp.objective_at(&self.x) {
            return Err("reported objective is not c · x".into());
        }
        if self.y_ub.len() != lp.a_ub.len()
            || self.y_eq.len() != lp.a_eq.len()
            || self.z_lower.len() != n
            || self.z_upper.len() != n
        {
            return Err("dual vector has the wrong length".into());
        }
        // Sign of a multiplier that must be "≤ 0 when minimising".
        let max = lp.sense == Sense::Maximize;
        let nonpos = |v: &Rational| if max { *v >= 0 } else { *v <= 0 };
        let nonneg = |v: &Rational| if max { *v <= 0 } else { *v >= 0 };
        for (i, y) in self.y_ub.iter().enumerate() {
            if !nonpos(y) {
                return Err(format!("inequality multiplier {i} has the wrong sign"));
            }
            let slack = lp.b_ub[i].clone() - dot(&lp.a_ub[i], &self.x);
            if *y != 0 && slack != 0 {
                return Err(format!(
                    "complementary slackness fails on inequality row {i}"
                ));
            }
        }
        let mut dual_obj = dot(&lp.b_ub, &self.y_ub) + dot(&lp.b_eq, &self.y_eq);
        for j in 0..n {
            let bd = &lp.bounds[j];
            let (zl, zu) = (&self.z_lower[j], &self.z_upper[j]);
            if !nonneg(zl) || !nonpos(zu) {
                return Err(format!(
                    "bound multiplier of variable {j} has the wrong sign"
                ));
            }
            match &bd.lower {
                None if *zl != 0 => {
                    return Err(format!("variable {j} has no lower bound but a multiplier"))
                }
                Some(l) => {
                    if *zl != 0 && self.x[j] != *l {
                        return Err(format!("complementary slackness fails on lower bound {j}"));
                    }
                    dual_obj += Rational::from(l * zl);
                }
                None => {}
            }
            match &bd.upper {
                None if *zu != 0 => {
                    return Err(format!("variable {j} has no upper bound but a multiplier"))
                }
                Some(u) => {
                    if *zu != 0 && self.x[j] != *u {
                        return Err(format!("complementary slackness fails on upper bound {j}"));
                    }
                    dual_obj += Rational::from(u * zu);
                }
                None => {}
            }
        }
        // Stationarity.
        let mut grad: Vec<Rational> = (0..n)
            .map(|j| Rational::from(&self.z_lower[j] + &self.z_upper[j]))
            .collect();
        add_transpose(&mut grad, &lp.a_ub, &self.y_ub);
        add_transpose(&mut grad, &lp.a_eq, &self.y_eq);
        if grad != lp.c {
            return Err("stationarity c = Aᵀy + z fails".into());
        }
        if dual_obj != self.objective {
            return Err("primal and dual objectives differ".into());
        }
        Ok(())
    }
}

/// A Farkas certificate of infeasibility.
///
/// `A_ubᵀ y_ub + A_eqᵀ y_eq − w_lower + w_upper = 0` with `y_ub, w_lower,
/// w_upper ≥ 0` (and zero on infinite bounds), while
/// `b_ub·y_ub + b_eq·y_eq − l·w_lower + u·w_upper < 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FarkasCertificate {
    /// Weights of the inequality rows (`≥ 0`).
    pub y_ub: Vec<Rational>,
    /// Weights of the equality rows (any sign).
    pub y_eq: Vec<Rational>,
    /// Weights of the lower bounds (`≥ 0`).
    pub w_lower: Vec<Rational>,
    /// Weights of the upper bounds (`≥ 0`).
    pub w_upper: Vec<Rational>,
}

impl FarkasCertificate {
    /// The value `b_ub·y_ub + b_eq·y_eq − l·w_lower + u·w_upper`, which is
    /// negative for a valid certificate.
    pub fn gap(&self, lp: &LinearProgram) -> Rational {
        let mut s = dot(&lp.b_ub, &self.y_ub) + dot(&lp.b_eq, &self.y_eq);
        for (j, bd) in lp.bounds.iter().enumerate() {
            if let Some(l) = &bd.lower {
                s -= Rational::from(l * &self.w_lower[j]);
            }
            if let Some(u) = &bd.upper {
                s += Rational::from(u * &self.w_upper[j]);
            }
        }
        s
    }

    /// Exact check of the certificate.
    pub fn verify(&self, lp: &LinearProgram) -> Result<(), String> {
        let n = lp.nvars();
        if self.y_ub.len() != lp.a_ub.len()
            || self.y_eq.len() != lp.a_eq.len()
            || self.w_lower.len() != n
            || self.w_upper.len() != n
        {
            return Err("Farkas vector has the wrong length".into());
        }
        if self.y_ub.iter().any(|v| *v < 0) {
            return Err("negative weight on an inequality row".into());
        }
        for j in 0..n {
            let bd = &lp.bounds[j];
            let (wl, wu) = (&self.w_lower[j], &self.w_upper[j]);
            if *wl < 0 || *wu < 0 {
                return Err(format!("negative weight on a bound of variable {j}"));
            }
            if (bd.lower.is_none() && *wl != 0) || (bd.upper.is_none() && *wu != 0) {
                return Err(format!("weight on an infinite bound of variable {j}"));
            }
        }
        let mut comb: Vec<Rational> = (0..n)
            .map(|j| Rational::from(&self.w_upper[j] - &self.w_lower[j]))
            .collect();
        add_transpose(&mut comb, &lp.a_ub, &self.y_ub);
        add_transpose(&mut comb, &lp.a_eq, &self.y_eq);
        if comb.iter().any(|v| *v != 0) {
            return Err("the weighted constraint sum is not 0 · x".into());
        }
        if self.gap(lp) >= 0 {
            return Err("the weighted right-hand side is not negative".into());
        }
        Ok(())
    }
}

/// A certificate of unboundedness: a feasible point and an improving ray.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnboundedCertificate {
    /// A feasible point.
    pub point: Vec<Rational>,
    /// A direction of the recession cone along which the objective improves
    /// without bound.
    pub direction: Vec<Rational>,
}

impl UnboundedCertificate {
    /// Exact check of the certificate.
    pub fn verify(&self, lp: &LinearProgram) -> Result<(), String> {
        if let Some(v) = lp.feasibility_violation(&self.point) {
            return Err(format!("point infeasible: {v}"));
        }
        let d = &self.direction;
        if d.len() != lp.nvars() {
            return Err("direction has the wrong length".into());
        }
        if lp.a_ub.iter().any(|row| dot(row, d) > 0) {
            return Err("direction leaves an inequality".into());
        }
        if lp.a_eq.iter().any(|row| dot(row, d) != 0) {
            return Err("direction leaves an equality".into());
        }
        for (j, bd) in lp.bounds.iter().enumerate() {
            if (bd.lower.is_some() && d[j] < 0) || (bd.upper.is_some() && d[j] > 0) {
                return Err(format!("direction leaves a bound of variable {j}"));
            }
        }
        let slope = dot(&lp.c, d);
        let improving = match lp.sense {
            Sense::Minimize => slope < 0,
            Sense::Maximize => slope > 0,
        };
        if !improving {
            return Err("objective does not improve along the direction".into());
        }
        Ok(())
    }
}

/// Solve a linear programme exactly and return its certified outcome.
///
/// # Errors
/// * `E-LP-001` — malformed shapes.
/// * `E-BUDGET-*` — the active budget tripped (checked at every pivot).
/// * `E-LP-006` — the computed certificate failed its exact check (a bug;
///   the result is withheld).
///
/// # Example
/// ```
/// use alkahest_cas::linprog::{linprog, LinearProgram, LpOutcome, Sense};
/// use rug::Rational;
/// let q = |n: i64| Rational::from(n);
/// // maximise x + y  s.t.  x + 2y ≤ 4,  3x + y ≤ 6,  x, y ≥ 0
/// let mut lp = LinearProgram::new(vec![q(1), q(1)]);
/// lp.sense = Sense::Maximize;
/// lp.a_ub = vec![vec![q(1), q(2)], vec![q(3), q(1)]];
/// lp.b_ub = vec![q(4), q(6)];
/// match linprog(&lp).unwrap() {
///     LpOutcome::Optimal(cert) => assert_eq!(cert.objective, Rational::from((14, 5))),
///     other => panic!("{other:?}"),
/// }
/// ```
pub fn linprog(lp: &LinearProgram) -> Result<LpOutcome, LinprogError> {
    lp.validate()?;
    let outcome = simplex::solve(lp)?;
    outcome
        .verify(lp)
        .map_err(|detail| LinprogError::Internal { detail })?;
    Ok(outcome)
}

pub(crate) fn dot(a: &[Rational], b: &[Rational]) -> Rational {
    let mut s = Rational::new();
    for (x, y) in a.iter().zip(b) {
        if *x != 0 && *y != 0 {
            s += Rational::from(x * y);
        }
    }
    s
}

/// `acc += Aᵀ y`.
fn add_transpose(acc: &mut [Rational], a: &[Vec<Rational>], y: &[Rational]) {
    for (row, yi) in a.iter().zip(y) {
        if *yi == 0 {
            continue;
        }
        for (s, aij) in acc.iter_mut().zip(row) {
            if *aij != 0 {
                *s += Rational::from(aij * yi);
            }
        }
    }
}
