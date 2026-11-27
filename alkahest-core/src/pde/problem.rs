//! Problem objects: [`PdeProblem`] (equations + unknowns + domain +
//! boundary / initial conditions), [`PdeDomain`] and [`PdeCondition`], plus
//! the fresh-arbitrary-function generator for general solutions (plan D6).

use std::collections::HashSet;

use crate::kernel::{ExprData, ExprId, ExprPool};

use super::classify::{classify_pde, PdeInfo};
use super::error::PdeError;
use super::jet::{decode_unknown, push_children, JetSpace};

/// A spatial / spatio-temporal domain.
///
/// Intervals, half-lines and boxes are complete; [`PdeDomain::Disk`] and
/// [`PdeDomain::Ball`] carry their data for the separation-of-variables
/// milestone, which will add the polar / spherical machinery that uses them.
/// `#[non_exhaustive]`: cylinders, annuli and wedges will be added.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PdeDomain {
    /// `lo < var < hi`.
    Interval {
        /// The coordinate.
        var: ExprId,
        /// Lower end.
        lo: ExprId,
        /// Upper end.
        hi: ExprId,
    },
    /// `var > lo` (`upper == false`) or `var < lo` (`upper == true`).
    HalfLine {
        /// The coordinate.
        var: ExprId,
        /// The finite end.
        end: ExprId,
        /// `true` for `(−∞, end)`, `false` for `(end, ∞)`.
        upper: bool,
    },
    /// All of `ℝ` in `var`.
    Line {
        /// The coordinate.
        var: ExprId,
    },
    /// A product `∏ (loᵢ, hiᵢ)`.
    Box {
        /// The coordinates.
        vars: Vec<ExprId>,
        /// `(loᵢ, hiᵢ)` per coordinate.
        bounds: Vec<(ExprId, ExprId)>,
    },
    /// The open disk `|(x, y) − center| < radius`.
    Disk {
        /// The two coordinates.
        vars: [ExprId; 2],
        /// The centre.
        center: [ExprId; 2],
        /// The radius.
        radius: ExprId,
    },
    /// The open ball `|x − center| < radius` in `vars.len()` dimensions.
    Ball {
        /// The coordinates.
        vars: Vec<ExprId>,
        /// The centre.
        center: Vec<ExprId>,
        /// The radius.
        radius: ExprId,
    },
    /// A product of domains in disjoint coordinates (space × time, …).
    Product(Vec<PdeDomain>),
    /// A coordinate box in polar `(r, θ)`, cylindrical `(r, θ, z)` or
    /// spherical `(r, θ, φ)` coordinates (milestone M4): a disk, annulus or
    /// sector, a cylinder, a ball or spherical shell.  `coords[i]` is `None`
    /// for a coordinate the problem does not depend on (a radial or
    /// axisymmetric problem); `bounds[i]` is that coordinate's range (an
    /// angular range of length `2π` is periodic).
    Curvilinear {
        /// The coordinate system.
        system: super::separation::CoordinateSystem,
        /// The coordinates (`None` where the unknown does not depend on one).
        coords: Vec<Option<ExprId>>,
        /// `(lo, hi)` per coordinate.
        bounds: Vec<(ExprId, ExprId)>,
    },
}

fn is_symbol(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Symbol { .. }))
}

/// The value of a numeric literal, for ordering checks; `None` when symbolic.
fn literal(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0)),
        ExprData::Rational(q) => Some(q.0),
        _ => None,
    }
}

fn check_var(var: ExprId, pool: &ExprPool) -> Result<(), PdeError> {
    if is_symbol(var, pool) {
        Ok(())
    } else {
        Err(PdeError::InvalidDomain(format!(
            "coordinate {} is not a symbol",
            pool.display(var)
        )))
    }
}

fn check_order(lo: ExprId, hi: ExprId, pool: &ExprPool) -> Result<(), PdeError> {
    if lo == hi {
        return Err(PdeError::InvalidDomain(format!(
            "empty interval ({}, {})",
            pool.display(lo),
            pool.display(hi)
        )));
    }
    if let (Some(a), Some(b)) = (literal(lo, pool), literal(hi, pool)) {
        if a >= b {
            return Err(PdeError::InvalidDomain(format!(
                "empty interval ({}, {})",
                pool.display(lo),
                pool.display(hi)
            )));
        }
    }
    Ok(())
}

impl PdeDomain {
    /// `lo < var < hi`.  `E-PDE-010` when `var` is not a symbol or the
    /// interval is provably empty (symbolic ends are accepted).
    pub fn interval(
        var: ExprId,
        lo: ExprId,
        hi: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        check_var(var, pool)?;
        check_order(lo, hi, pool)?;
        Ok(PdeDomain::Interval { var, lo, hi })
    }

    /// `var > end` (`upper = false`) or `var < end` (`upper = true`).
    pub fn half_line(
        var: ExprId,
        end: ExprId,
        upper: bool,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        check_var(var, pool)?;
        Ok(PdeDomain::HalfLine { var, end, upper })
    }

    /// All of `ℝ` in `var`.
    pub fn line(var: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        check_var(var, pool)?;
        Ok(PdeDomain::Line { var })
    }

    /// The box `∏ (loᵢ, hiᵢ)`.
    pub fn box_(
        vars: &[ExprId],
        bounds: &[(ExprId, ExprId)],
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        if vars.is_empty() || vars.len() != bounds.len() {
            return Err(PdeError::InvalidDomain(format!(
                "a box needs one (lo, hi) per coordinate: {} coordinates, {} bounds",
                vars.len(),
                bounds.len()
            )));
        }
        for (i, (&v, &(lo, hi))) in vars.iter().zip(bounds).enumerate() {
            check_var(v, pool)?;
            if vars[..i].contains(&v) {
                return Err(PdeError::InvalidDomain(format!(
                    "coordinate {} repeated",
                    pool.display(v)
                )));
            }
            check_order(lo, hi, pool)?;
        }
        Ok(PdeDomain::Box {
            vars: vars.to_vec(),
            bounds: bounds.to_vec(),
        })
    }

    fn check_radius(radius: ExprId, pool: &ExprPool) -> Result<(), PdeError> {
        if literal(radius, pool).is_some_and(|r| r <= 0) {
            return Err(PdeError::InvalidDomain(format!(
                "radius {} is not positive",
                pool.display(radius)
            )));
        }
        Ok(())
    }

    /// The open disk of `radius` about `center` in the coordinates `vars`.
    pub fn disk(
        vars: [ExprId; 2],
        center: [ExprId; 2],
        radius: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        check_var(vars[0], pool)?;
        check_var(vars[1], pool)?;
        Self::check_radius(radius, pool)?;
        Ok(PdeDomain::Disk {
            vars,
            center,
            radius,
        })
    }

    /// The open ball of `radius` about `center` in the coordinates `vars`.
    pub fn ball(
        vars: &[ExprId],
        center: &[ExprId],
        radius: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        if vars.is_empty() || vars.len() != center.len() {
            return Err(PdeError::InvalidDomain(
                "a ball needs one centre coordinate per variable".into(),
            ));
        }
        for &v in vars {
            check_var(v, pool)?;
        }
        Self::check_radius(radius, pool)?;
        Ok(PdeDomain::Ball {
            vars: vars.to_vec(),
            center: center.to_vec(),
            radius,
        })
    }

    /// A coordinate box in curvilinear coordinates: `coords` / `bounds` per
    /// coordinate of `system` (see [`PdeDomain::Curvilinear`]).
    ///
    /// `E-PDE-010` for the wrong number of entries, a coordinate that is not
    /// a symbol, a negative inner radius, a polar / azimuthal range longer
    /// than `2π`, or a polar angle outside `[0, π]` (literal bounds).
    pub fn curvilinear(
        system: super::separation::CoordinateSystem,
        coords: &[Option<ExprId>],
        bounds: &[(ExprId, ExprId)],
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        use super::separation::CoordinateSystem as C;
        let n = system.dimension();
        if coords.len() != n || bounds.len() != n {
            return Err(PdeError::InvalidDomain(format!(
                "{} coordinates need {n} coordinates and {n} ranges",
                system.as_str()
            )));
        }
        if coords[0].is_none() {
            return Err(PdeError::InvalidDomain(
                "a curvilinear domain needs its radial coordinate".into(),
            ));
        }
        let mut seen = Vec::new();
        for (c, &(lo, hi)) in coords.iter().zip(bounds) {
            if let Some(v) = c {
                check_var(*v, pool)?;
                if seen.contains(v) {
                    return Err(PdeError::InvalidDomain(format!(
                        "coordinate {} repeated",
                        pool.display(*v)
                    )));
                }
                seen.push(*v);
            }
            check_order(lo, hi, pool)?;
        }
        if literal(bounds[0].0, pool).is_some_and(|r| r < 0) {
            return Err(PdeError::InvalidDomain("a negative inner radius".into()));
        }
        let two_pi = 2.0 * std::f64::consts::PI;
        let width = |(lo, hi): (ExprId, ExprId)| -> Option<f64> {
            let w = pool.add(vec![hi, pool.mul(vec![pool.integer(-1), lo])]);
            crate::eval::eval_f64(w, pool, &std::collections::HashMap::new()).ok()
        };
        let angular: Vec<usize> = match system {
            C::Polar | C::Cylindrical => vec![1],
            C::Spherical => vec![2],
        };
        for i in angular {
            if width(bounds[i]).is_some_and(|w| w > two_pi + 1e-12) {
                return Err(PdeError::InvalidDomain(
                    "an angular range longer than 2π".into(),
                ));
            }
        }
        if system == C::Spherical {
            let ev = |e| crate::eval::eval_f64(e, pool, &std::collections::HashMap::new()).ok();
            if ev(bounds[1].0).is_some_and(|v| v < -1e-12)
                || ev(bounds[1].1).is_some_and(|v| v > std::f64::consts::PI + 1e-12)
            {
                return Err(PdeError::InvalidDomain(
                    "the polar angle θ ranges within [0, π]".into(),
                ));
            }
        }
        Ok(PdeDomain::Curvilinear {
            system,
            coords: coords.to_vec(),
            bounds: bounds.to_vec(),
        })
    }

    /// The disk `0 ≤ r < radius` in polar coordinates `(r, θ)`, `θ ∈ (−π, π)`.
    pub fn polar_disk(
        r: ExprId,
        theta: Option<ExprId>,
        radius: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        let pi = pool.symbol("pi", crate::kernel::Domain::Real);
        Self::curvilinear(
            super::separation::CoordinateSystem::Polar,
            &[Some(r), theta],
            &[
                (pool.integer(0), radius),
                (pool.mul(vec![pool.integer(-1), pi]), pi),
            ],
            pool,
        )
    }

    /// The product of domains in disjoint coordinates.
    pub fn product(parts: Vec<PdeDomain>, pool: &ExprPool) -> Result<Self, PdeError> {
        let mut seen: Vec<ExprId> = Vec::new();
        for p in &parts {
            for v in p.vars() {
                if seen.contains(&v) {
                    return Err(PdeError::InvalidDomain(format!(
                        "coordinate {} appears in two factors of a product",
                        pool.display(v)
                    )));
                }
                seen.push(v);
            }
        }
        // Flatten nested products.
        let mut flat = Vec::new();
        for p in parts {
            match p {
                PdeDomain::Product(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        Ok(PdeDomain::Product(flat))
    }

    /// The coordinates the domain constrains.
    pub fn vars(&self) -> Vec<ExprId> {
        match self {
            PdeDomain::Interval { var, .. }
            | PdeDomain::HalfLine { var, .. }
            | PdeDomain::Line { var } => {
                vec![*var]
            }
            PdeDomain::Box { vars, .. } | PdeDomain::Ball { vars, .. } => vars.clone(),
            PdeDomain::Disk { vars, .. } => vars.to_vec(),
            PdeDomain::Product(ps) => ps.iter().flat_map(|p| p.vars()).collect(),
            PdeDomain::Curvilinear { coords, .. } => coords.iter().flatten().copied().collect(),
        }
    }

    /// Short stable name: `"interval"`, `"half_line"`, `"line"`, `"box"`,
    /// `"disk"`, `"ball"`, `"product"`, `"curvilinear"`.
    pub fn kind(&self) -> &'static str {
        match self {
            PdeDomain::Interval { .. } => "interval",
            PdeDomain::HalfLine { .. } => "half_line",
            PdeDomain::Line { .. } => "line",
            PdeDomain::Box { .. } => "box",
            PdeDomain::Disk { .. } => "disk",
            PdeDomain::Ball { .. } => "ball",
            PdeDomain::Product(_) => "product",
            PdeDomain::Curvilinear { .. } => "curvilinear",
        }
    }

    /// Membership as a predicate in the coordinates (`(0 < x) ∧ (x < L)`).
    pub fn condition(&self, pool: &ExprPool) -> ExprId {
        use crate::kernel::expr::PredicateKind as K;
        let lt = |a, b| pool.predicate(K::Lt, vec![a, b]);
        let and = |v: Vec<ExprId>| {
            if v.len() == 1 {
                v[0]
            } else {
                pool.predicate(K::And, v)
            }
        };
        let r2 = |vars: &[ExprId], center: &[ExprId]| {
            pool.add(
                vars.iter()
                    .zip(center)
                    .map(|(&x, &c)| {
                        pool.pow(
                            pool.add(vec![x, pool.mul(vec![pool.integer(-1), c])]),
                            pool.integer(2),
                        )
                    })
                    .collect(),
            )
        };
        match self {
            PdeDomain::Interval { var, lo, hi } => and(vec![lt(*lo, *var), lt(*var, *hi)]),
            PdeDomain::HalfLine { var, end, upper } => {
                if *upper {
                    lt(*var, *end)
                } else {
                    lt(*end, *var)
                }
            }
            PdeDomain::Line { .. } => pool.predicate(K::True, vec![]),
            PdeDomain::Box { vars, bounds } => and(vars
                .iter()
                .zip(bounds)
                .flat_map(|(&v, &(lo, hi))| [lt(lo, v), lt(v, hi)])
                .collect()),
            PdeDomain::Disk {
                vars,
                center,
                radius,
            } => lt(r2(vars, center), pool.pow(*radius, pool.integer(2))),
            PdeDomain::Ball {
                vars,
                center,
                radius,
            } => lt(r2(vars, center), pool.pow(*radius, pool.integer(2))),
            PdeDomain::Product(ps) => and(ps.iter().map(|p| p.condition(pool)).collect()),
            PdeDomain::Curvilinear { coords, bounds, .. } => {
                let mut parts = Vec::new();
                for (c, &(lo, hi)) in coords.iter().zip(bounds) {
                    if let Some(v) = c {
                        parts.push(lt(lo, *v));
                        parts.push(lt(*v, hi));
                    }
                }
                and(parts)
            }
        }
    }
}

/// Whether a condition is a boundary or an initial condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConditionKind {
    /// Holds on (part of) the spatial boundary.
    Boundary,
    /// Holds at the initial time.
    Initial,
}

/// A side condition `lhs = rhs` (`u(0, t) = 0`, `u_t(x, 0) = g(x)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PdeCondition {
    /// Boundary or initial.
    pub kind: ConditionKind,
    /// Left-hand side; mentions an unknown.
    pub lhs: ExprId,
    /// Right-hand side.
    pub rhs: ExprId,
}

impl PdeCondition {
    /// `lhs − rhs`, the residual that must vanish.
    pub fn residual(&self, pool: &ExprPool) -> ExprId {
        pool.add(vec![self.lhs, pool.mul(vec![pool.integer(-1), self.rhs])])
    }
}

/// Whether `expr` mentions one of the unknowns named in `names` (applied at
/// any arguments, or differentiated).
pub(crate) fn mentions_unknown(expr: ExprId, names: &[String], pool: &ExprPool) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![expr];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = crate::diff::undefined::view(n, pool) {
            if names.contains(&v.name) {
                return true;
            }
        }
        push_children(n, pool, &mut stack);
    }
    false
}

/// A PDE problem: a system of equations (each read as `= 0`) in one or more
/// unknowns, optionally with a domain and boundary / initial conditions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PdeProblem {
    equations: Vec<ExprId>,
    unknowns: Vec<ExprId>,
    space: JetSpace,
    order: u32,
    domain: Option<PdeDomain>,
    conditions: Vec<PdeCondition>,
}

impl PdeProblem {
    /// The problem `equations = 0` in `unknowns` (`u(x, t)`, …).  The
    /// independent variables are read off the unknowns.
    ///
    /// # Errors
    ///
    /// `E-PDE-008` for an empty system; `E-PDE-001` / `E-PDE-002` for
    /// malformed unknowns; `E-PDE-007` when an equation mentions no unknown or
    /// the system has no derivative at all; `E-PDE-003` when an unknown
    /// occurs off its declared arguments in an equation.
    pub fn new(
        equations: &[ExprId],
        unknowns: &[ExprId],
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        if equations.is_empty() {
            return Err(PdeError::SystemShape("no equations".into()));
        }
        let big = JetSpace::from_unknowns(unknowns, u32::MAX / 4, pool)?;
        let names: Vec<String> = unknowns
            .iter()
            .map(|&u| decode_unknown(u, pool).map(|v| v.name))
            .collect::<Result<_, _>>()?;
        let mut order = 0;
        for &e in equations {
            if !mentions_unknown(e, &names, pool) {
                return Err(PdeError::NotADifferentialEquation(format!(
                    "{} mentions none of the unknowns",
                    pool.display(e)
                )));
            }
            order = order.max(big.order_of(e, pool)?.unwrap_or(0));
        }
        if order == 0 {
            return Err(PdeError::NotADifferentialEquation(
                "the system has no derivative of an unknown".into(),
            ));
        }
        Ok(PdeProblem {
            equations: equations.to_vec(),
            unknowns: unknowns.to_vec(),
            space: big.prolonged(0).with_max_order(order),
            order,
            domain: None,
            conditions: Vec::new(),
        })
    }

    /// The same problem posed on `domain`.
    ///
    /// # Errors
    ///
    /// `E-PDE-010` when the domain constrains a coordinate that is not an
    /// independent variable.
    pub fn on(mut self, domain: PdeDomain, pool: &ExprPool) -> Result<Self, PdeError> {
        for v in domain.vars() {
            if !self.space.independent().contains(&v) {
                return Err(PdeError::InvalidDomain(format!(
                    "{} is not an independent variable of the problem",
                    pool.display(v)
                )));
            }
        }
        self.domain = Some(domain);
        Ok(self)
    }

    fn with_condition(
        mut self,
        kind: ConditionKind,
        lhs: ExprId,
        rhs: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        let names: Vec<String> = self
            .unknowns
            .iter()
            .map(|&u| decode_unknown(u, pool).map(|v| v.name))
            .collect::<Result<_, _>>()?;
        if !mentions_unknown(lhs, &names, pool) && !mentions_unknown(rhs, &names, pool) {
            return Err(PdeError::InvalidCondition(format!(
                "{} = {} mentions no unknown",
                pool.display(lhs),
                pool.display(rhs)
            )));
        }
        self.conditions.push(PdeCondition { kind, lhs, rhs });
        Ok(self)
    }

    /// Add the boundary condition `lhs = rhs`.  `E-PDE-011` when neither
    /// side mentions an unknown.
    pub fn bc(self, lhs: ExprId, rhs: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        self.with_condition(ConditionKind::Boundary, lhs, rhs, pool)
    }

    /// Add the initial condition `lhs = rhs`.  `E-PDE-011` when neither side
    /// mentions an unknown.
    pub fn ic(self, lhs: ExprId, rhs: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        self.with_condition(ConditionKind::Initial, lhs, rhs, pool)
    }

    /// The equations, each read as `= 0`.
    pub fn equations(&self) -> &[ExprId] {
        &self.equations
    }

    /// The unknowns.
    pub fn unknowns(&self) -> &[ExprId] {
        &self.unknowns
    }

    /// The independent variables.
    pub fn independent(&self) -> &[ExprId] {
        self.space.independent()
    }

    /// The highest derivative order of the system.
    pub fn order(&self) -> u32 {
        self.order
    }

    /// The jet space of the problem's order.
    pub fn jet_space(&self) -> &JetSpace {
        &self.space
    }

    /// The domain, if one was set.
    pub fn domain(&self) -> Option<&PdeDomain> {
        self.domain.as_ref()
    }

    /// The boundary and initial conditions, in the order added.
    pub fn conditions(&self) -> &[PdeCondition] {
        &self.conditions
    }

    /// [`classify_pde`] of a scalar problem.
    ///
    /// # Errors
    ///
    /// `E-PDE-012` for a system (more than one equation or unknown) — use
    /// [`Self::classify_system`] for those; as [`classify_pde`].
    pub fn classify(&self, pool: &ExprPool) -> Result<PdeInfo, PdeError> {
        if self.equations.len() != 1 || self.unknowns.len() != 1 {
            return Err(PdeError::ClassificationUnsupported(format!(
                "classification is defined for one equation in one unknown; this system has {} equations in {} unknowns",
                self.equations.len(),
                self.unknowns.len()
            )));
        }
        classify_pde(self.equations[0], self.unknowns[0], pool)
    }

    /// [`super::classify_pde_system`] of the problem's equations and
    /// unknowns (any number of each, a scalar problem included).
    ///
    /// # Errors
    ///
    /// As [`super::classify_pde_system`].
    pub fn classify_system(
        &self,
        pool: &ExprPool,
    ) -> Result<super::classify_system::PdeSystemInfo, PdeError> {
        super::classify_system::classify_pde_system(&self.equations, &self.unknowns, pool)
    }
}

// ---------------------------------------------------------------------------
// D6: arbitrary functions in general solutions
// ---------------------------------------------------------------------------

/// A generator of fresh arbitrary-function names `F1, F2, …` — the PDE
/// counterpart of `dsolve`'s integration constants `C1, C2, …`.
///
/// A name is skipped while it is taken: by a function or symbol of the
/// expressions given to [`ArbitraryFunctions::avoiding`], or by an earlier
/// call.  The functions are ordinary undefined functions, so they
/// differentiate through the `Derivative` head and verification keeps them
/// symbolic.
#[derive(Clone, Debug)]
pub struct ArbitraryFunctions {
    prefix: String,
    next: usize,
    taken: HashSet<String>,
}

impl Default for ArbitraryFunctions {
    fn default() -> Self {
        Self::new("F")
    }
}

impl ArbitraryFunctions {
    /// Names `{prefix}1, {prefix}2, …`.
    pub fn new(prefix: &str) -> Self {
        ArbitraryFunctions {
            prefix: prefix.to_string(),
            next: 1,
            taken: HashSet::new(),
        }
    }

    /// Also avoid every function and symbol name occurring in `exprs`.
    pub fn avoiding(mut self, exprs: &[ExprId], pool: &ExprPool) -> Self {
        let mut seen = HashSet::new();
        let mut stack: Vec<ExprId> = exprs.to_vec();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            pool.with(n, |d| match d {
                ExprData::Symbol { name, .. } | ExprData::Func { name, .. } => {
                    self.taken.insert(name.clone());
                }
                _ => {}
            });
            push_children(n, pool, &mut stack);
        }
        self
    }

    /// The next fresh name.
    pub fn fresh_name(&mut self) -> String {
        loop {
            let name = format!("{}{}", self.prefix, self.next);
            self.next += 1;
            if self.taken.insert(name.clone()) {
                return name;
            }
        }
    }

    /// A fresh arbitrary function applied to `args`: `F1(x - c t)`.
    pub fn fresh(&mut self, args: Vec<ExprId>, pool: &ExprPool) -> ExprId {
        let name = self.fresh_name();
        pool.func(name, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::kernel::Domain;

    #[test]
    fn problem_and_conditions() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let l = pool.symbol("L", Domain::Positive);
        let u = pool.func("u", vec![x, t]);
        let ut = crate::diff::diff(u, t, &pool).unwrap().value;
        let ux = crate::diff::diff(u, x, &pool).unwrap().value;
        let uxx = crate::diff::diff(ux, x, &pool).unwrap().value;
        let heat = pool.add(vec![ut, pool.mul(vec![pool.integer(-1), uxx])]);
        let zero = pool.integer(0);
        let dom = PdeDomain::product(
            vec![
                PdeDomain::interval(x, zero, l, &pool).unwrap(),
                PdeDomain::half_line(t, zero, false, &pool).unwrap(),
            ],
            &pool,
        )
        .unwrap();
        let p = PdeProblem::new(&[heat], &[u], &pool)
            .unwrap()
            .on(dom, &pool)
            .unwrap()
            .bc(pool.func("u", vec![zero, t]), zero, &pool)
            .unwrap()
            .bc(pool.func("u", vec![l, t]), zero, &pool)
            .unwrap()
            .ic(
                pool.func("u", vec![x, zero]),
                pool.func("sin", vec![x]),
                &pool,
            )
            .unwrap();
        assert_eq!(p.order(), 2);
        assert_eq!(p.conditions().len(), 3);
        assert_eq!(p.classify(&pool).unwrap().named[0].name, "heat");
        assert_eq!(p.domain().unwrap().kind(), "product");
        let bad = p.clone().bc(x, zero, &pool).unwrap_err();
        assert_eq!(bad.code(), "E-PDE-011");
        let y = pool.symbol("y", Domain::Real);
        let bad = p.on(PdeDomain::line(y, &pool).unwrap(), &pool).unwrap_err();
        assert_eq!(bad.code(), "E-PDE-010");
        assert_eq!(
            PdeDomain::interval(x, pool.integer(1), zero, &pool)
                .unwrap_err()
                .code(),
            "E-PDE-010"
        );
        assert_eq!(
            PdeProblem::new(&[x], &[u], &pool).unwrap_err().code(),
            "E-PDE-007"
        );
    }

    #[test]
    fn arbitrary_names_skip_taken() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f1 = pool.func("F1", vec![x]);
        let mut g = ArbitraryFunctions::default().avoiding(&[f1], &pool);
        assert_eq!(g.fresh_name(), "F2");
        let f = g.fresh(vec![x], &pool);
        assert_eq!(pool.display(f).to_string(), "F3(x)");
        assert!(crate::diff::is_undefined_function_name("F3"));
    }
}
