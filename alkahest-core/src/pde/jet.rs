//! The jet layer (plan decision D3): a multi-index bijection between
//! derivatives of unknown functions and plain *jet coordinates*.
//!
//! A PDE is written with applied undefined functions and the reserved
//! `Derivative` head ([`crate::diff::undefined`]):
//!
//! ```text
//! Derivative(u(x, t), 1, 1)  —  ∂²u/∂x∂t
//! ```
//!
//! Most algorithms want to treat `u`, `u_x`, `u_t`, `u_xt`, … as independent
//! variables instead: the classification reads coefficients off `∂F/∂u_xx`,
//! a symmetry computation needs the total derivative `D_x` on functions of the
//! jets, and `GroebnerBasis` / `rosenfeld_groebner` take polynomial systems in
//! plain symbols.  [`JetSpace`] is that coordinate system:
//!
//! * independent variables `x₁ … xₙ` (plain symbols), dependent functions
//!   `u¹ … uᵐ` (each applied to a subset of the `xᵢ`), and a maximum order;
//! * the jet `u_α` for a multi-index `α ∈ ℕⁿ` is a symbol with a stable,
//!   readable name — `u`, `u_x`, `u_t`, `u_xt`, `u_xx` (variables in the order
//!   they were declared, repeated by multiplicity; `u_x1_x2` when a variable
//!   name is longer than one character);
//! * [`JetSpace::to_jets`] / [`JetSpace::from_jets`] convert between the two
//!   pictures, and are inverse to each other;
//! * [`JetSpace::total_derivative`] is `D_{xᵢ} = ∂_{xᵢ} + Σ u_{α+eᵢ} ∂/∂u_α`,
//!   [`JetSpace::prolong`] the prolongation of a system to a higher order, and
//!   [`JetSpace::to_rational_system`] the numerator / non-vanishing
//!   denominator form a Gröbner engine accepts.
//!
//! # ODE conventions
//!
//! The ODE code predates this layer and uses three spellings of the same jet
//! coordinates: `y, y', y''` (`dsolve`'s [`crate::ode::dsolve::OdeInput`]), `q, q_dot,
//! q_ddot, q_d3` ([`crate::diff::functions_to_symbols`]) and `y, dy/dt,
//! ddy/dt/dt` (the DAE index reduction).  [`JetNaming`] reproduces each of
//! them for a one-variable jet space, so those entry points are adapters over
//! the same map ([`JetSpace::to_ode_input`], [`JetSpace::dae_pairs`]); the old
//! functions are unchanged.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::diff::diff_impl::diff;
use crate::diff::undefined::{build, view, UndefView};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

use super::error::PdeError;

/// A multi-index `α = (α₁, …, αₙ)`: `αᵢ` derivatives with respect to the
/// `i`-th independent variable of the [`JetSpace`].
pub type MultiIndex = Vec<u32>;

/// How jet coordinates are named.
///
/// [`JetNaming::Subscript`] is the PDE spelling and the default.  The other
/// three reproduce the ODE code's existing conventions and need a jet space
/// with exactly one independent variable (`E-PDE-009` otherwise).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JetNaming {
    /// `u`, `u_x`, `u_xt`, `u_xx` (`u_x1_x2` for multi-character variables).
    Subscript,
    /// `y`, `y'`, `y''`, … — `dsolve`'s [`crate::ode::dsolve::OdeInput`] spelling.
    Prime,
    /// `q`, `q_dot`, `q_ddot`, `q_d3`, … — [`crate::diff::functions_to_symbols`].
    Dot,
    /// `y`, `dy/dt`, `ddy/dt/dt`, … — the DAE index-reduction spelling
    /// (`d{name}/dt` iterated; the variable's own name replaces `t`).
    DaeDt,
}

/// One dependent function of a [`JetSpace`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct DepFunc {
    /// Function name `u`.
    name: String,
    /// The applied function `u(x, t)` as declared.
    applied: ExprId,
    /// `slots[j]` = index into the independent variables of argument `j`.
    slots: Vec<usize>,
}

/// One coordinate found by [`JetSpace::jets_in`]: unknown `dep` differentiated
/// by `alpha`, represented by the jet symbol `symbol`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JetCoordinate {
    /// Index of the unknown in [`JetSpace::unknowns`].
    pub dep: usize,
    /// The multi-index, one entry per independent variable.
    pub alpha: MultiIndex,
    /// The jet symbol `u_α`.
    pub symbol: ExprId,
}

/// A system converted by [`JetSpace::to_rational_system`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JetSystem {
    /// The numerators, each to be read as `= 0`; polynomial in
    /// [`Self::variables`], [`Self::independent`] and [`Self::parameters`].
    pub equations: Vec<ExprId>,
    /// Denominators that were cleared and must not vanish on a solution.
    pub nonzero: Vec<ExprId>,
    /// The jet coordinates that occur, highest-ranked first (orderly ranking:
    /// total order descending, then unknown, then multi-index descending).
    pub variables: Vec<ExprId>,
    /// The independent variables of the jet space.
    pub independent: Vec<ExprId>,
    /// Everything else the equations mention as an opaque generator: symbolic
    /// constants, and functions of the independent variables alone such as
    /// `sin(x)` (they are constants for the algebra, not for `D_x`).
    pub parameters: Vec<ExprId>,
}

/// A multi-index jet space `J^k(ℝⁿ, ℝᵐ)`: independent variables, dependent
/// functions and a maximum order.  See the [module documentation](self).
///
/// The maximum order bounds [`Self::coordinates`] and what [`Self::to_jets`]
/// accepts; [`Self::total_derivative`] may produce jets one order higher (it
/// maps `J^k` to `J^{k+1}`), and [`Self::prolonged`] returns the larger space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JetSpace {
    indep: Vec<ExprId>,
    indep_names: Vec<String>,
    deps: Vec<DepFunc>,
    max_order: u32,
    naming: JetNaming,
    domain: Domain,
}

fn symbol_name(e: ExprId, pool: &ExprPool) -> Option<String> {
    pool.with(e, |d| match d {
        ExprData::Symbol { name, .. } => Some(name.clone()),
        _ => None,
    })
}

/// Decode `u` as an applied undefined function of distinct plain symbols.
pub(crate) fn decode_unknown(u: ExprId, pool: &ExprPool) -> Result<UndefView, PdeError> {
    let shown = || pool.display(u).to_string();
    let v = view(u, pool).ok_or_else(|| PdeError::NotAnUnknown(shown()))?;
    if v.inner != u {
        return Err(PdeError::NotAnUnknown(format!(
            "{} is a derivative; pass the function itself",
            shown()
        )));
    }
    for (i, &a) in v.args.iter().enumerate() {
        if symbol_name(a, pool).is_none() || v.args[..i].contains(&a) {
            return Err(PdeError::NotAnUnknown(format!(
                "{}: every argument must be a distinct plain symbol",
                shown()
            )));
        }
    }
    Ok(v)
}

/// All multi-indices of total order `order` in `n` variables, in descending
/// lexicographic order (`(2,0)`, `(1,1)`, `(0,2)`).
pub fn multi_indices(n: usize, order: u32) -> Vec<MultiIndex> {
    fn rec(n: usize, left: u32, cur: &mut Vec<u32>, out: &mut Vec<MultiIndex>) {
        if cur.len() + 1 == n {
            cur.push(left);
            out.push(cur.clone());
            cur.pop();
            return;
        }
        for k in (0..=left).rev() {
            cur.push(k);
            rec(n, left - k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    if n == 0 {
        if order == 0 {
            out.push(Vec::new());
        }
        return out;
    }
    rec(n, order, &mut Vec::with_capacity(n), &mut out);
    out
}

impl JetSpace {
    /// The jet space of order `max_order` for the unknowns `unknowns`
    /// (applied undefined functions such as `u(x, t)`, `v(t)`) over the
    /// independent variables `indep`.
    ///
    /// # Errors
    ///
    /// `E-PDE-001` when an unknown is not an undefined function of distinct
    /// symbols; `E-PDE-002` when `indep` is empty or repeats a variable, an
    /// unknown's argument is not in `indep`, or two unknowns share a name.
    pub fn new(
        indep: &[ExprId],
        unknowns: &[ExprId],
        max_order: u32,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        if indep.is_empty() {
            return Err(PdeError::InconsistentVariables(
                "no independent variables".into(),
            ));
        }
        if unknowns.is_empty() {
            return Err(PdeError::InconsistentVariables("no unknowns".into()));
        }
        let mut indep_names = Vec::with_capacity(indep.len());
        for (i, &x) in indep.iter().enumerate() {
            let name = symbol_name(x, pool).ok_or_else(|| {
                PdeError::InconsistentVariables(format!(
                    "independent variable {} is not a symbol",
                    pool.display(x)
                ))
            })?;
            if indep[..i].contains(&x) || indep_names.contains(&name) {
                return Err(PdeError::InconsistentVariables(format!(
                    "independent variable {name} is repeated"
                )));
            }
            indep_names.push(name);
        }
        let mut deps: Vec<DepFunc> = Vec::with_capacity(unknowns.len());
        for &u in unknowns {
            let v = decode_unknown(u, pool)?;
            if deps.iter().any(|d| d.name == v.name) {
                return Err(PdeError::InconsistentVariables(format!(
                    "unknown {} is declared twice",
                    v.name
                )));
            }
            let mut slots = Vec::with_capacity(v.args.len());
            for &a in &v.args {
                let i = indep.iter().position(|&x| x == a).ok_or_else(|| {
                    PdeError::InconsistentVariables(format!(
                        "{}: argument {} is not an independent variable",
                        pool.display(u),
                        pool.display(a)
                    ))
                })?;
                slots.push(i);
            }
            deps.push(DepFunc {
                name: v.name,
                applied: u,
                slots,
            });
        }
        Ok(JetSpace {
            indep: indep.to_vec(),
            indep_names,
            deps,
            max_order,
            naming: JetNaming::Subscript,
            domain: Domain::Real,
        })
    }

    /// [`Self::new`] with the independent variables read off the unknowns'
    /// arguments, in order of first appearance.
    pub fn from_unknowns(
        unknowns: &[ExprId],
        max_order: u32,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        let mut indep = Vec::new();
        for &u in unknowns {
            for a in decode_unknown(u, pool)?.args {
                if !indep.contains(&a) {
                    indep.push(a);
                }
            }
        }
        Self::new(&indep, unknowns, max_order, pool)
    }

    /// The same space with jets named by `naming`.
    ///
    /// # Errors
    ///
    /// `E-PDE-009` for an ODE convention on a space with more than one
    /// independent variable.
    pub fn with_naming(mut self, naming: JetNaming) -> Result<Self, PdeError> {
        if naming != JetNaming::Subscript && self.indep.len() != 1 {
            return Err(PdeError::NamingNeedsOneVariable(format!(
                "{naming:?} naming needs exactly one independent variable, the space has {}",
                self.indep.len()
            )));
        }
        self.naming = naming;
        Ok(self)
    }

    /// The same space with jet symbols in `domain` (default [`Domain::Real`]).
    /// The domain is part of a symbol's identity, so it is part of the jet map.
    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// The same space with maximum order `max_order + k` — the space a
    /// `k`-fold prolongation lives in.
    pub fn prolonged(&self, k: u32) -> Self {
        let mut s = self.clone();
        s.max_order = self.max_order.saturating_add(k);
        s
    }

    /// The same space with maximum order exactly `max_order`.
    pub fn with_max_order(mut self, max_order: u32) -> Self {
        self.max_order = max_order;
        self
    }

    /// The independent variables.
    pub fn independent(&self) -> &[ExprId] {
        &self.indep
    }

    /// The unknowns as applied functions (`u(x, t)`), in declaration order.
    pub fn unknowns(&self) -> Vec<ExprId> {
        self.deps.iter().map(|d| d.applied).collect()
    }

    /// The maximum order.
    pub fn max_order(&self) -> u32 {
        self.max_order
    }

    /// The naming convention.
    pub fn naming(&self) -> JetNaming {
        self.naming
    }

    /// The domain of the jet symbols.
    pub fn domain(&self) -> Domain {
        self.domain
    }

    /// Whether unknown `dep` depends on independent variable `i`.
    pub fn depends_on(&self, dep: usize, i: usize) -> bool {
        self.deps.get(dep).is_some_and(|d| d.slots.contains(&i))
    }

    fn check(&self, dep: usize, alpha: &[u32]) -> Result<(), PdeError> {
        let d = self.deps.get(dep).ok_or_else(|| {
            PdeError::InconsistentVariables(format!(
                "unknown index {dep} out of range (the space has {})",
                self.deps.len()
            ))
        })?;
        if alpha.len() != self.indep.len() {
            return Err(PdeError::InconsistentVariables(format!(
                "multi-index of length {} for {} independent variables",
                alpha.len(),
                self.indep.len()
            )));
        }
        for (i, &k) in alpha.iter().enumerate() {
            if k > 0 && !d.slots.contains(&i) {
                return Err(PdeError::InconsistentVariables(format!(
                    "{} does not depend on {}",
                    d.name, self.indep_names[i]
                )));
            }
        }
        Ok(())
    }

    /// The name of the jet `u_α` of unknown `dep` (no validation).
    pub fn jet_name(&self, dep: usize, alpha: &[u32]) -> String {
        let name = &self.deps[dep].name;
        let order: u32 = alpha.iter().sum();
        match self.naming {
            JetNaming::Subscript => {
                if order == 0 {
                    return name.clone();
                }
                let single = self.indep_names.iter().all(|n| n.chars().count() == 1);
                let mut s = format!("{name}_");
                let mut first = true;
                for (i, &k) in alpha.iter().enumerate() {
                    for _ in 0..k {
                        if !single && !first {
                            s.push('_');
                        }
                        s.push_str(&self.indep_names[i]);
                        first = false;
                    }
                }
                s
            }
            JetNaming::Prime => format!("{name}{}", "'".repeat(order as usize)),
            JetNaming::Dot => match order {
                0 => name.clone(),
                1 => format!("{name}_dot"),
                2 => format!("{name}_ddot"),
                k => format!("{name}_d{k}"),
            },
            JetNaming::DaeDt => {
                let mut s = name.clone();
                for _ in 0..order {
                    s = format!("d{s}/d{}", self.indep_names[0]);
                }
                s
            }
        }
    }

    /// The jet coordinate `u_α` of unknown `dep` (a symbol).  Orders above
    /// [`Self::max_order`] are allowed here, so that a total derivative can
    /// name its output.
    ///
    /// # Errors
    ///
    /// `E-PDE-002` when `dep` is out of range, `alpha` has the wrong length,
    /// or differentiates with respect to a variable `dep` does not depend on.
    pub fn jet(&self, dep: usize, alpha: &[u32], pool: &ExprPool) -> Result<ExprId, PdeError> {
        self.check(dep, alpha)?;
        Ok(pool.symbol(self.jet_name(dep, alpha), self.domain))
    }

    /// The derivative `Derivative(u(x…), k…)` that the jet `u_α` stands for.
    ///
    /// # Errors
    ///
    /// As [`Self::jet`].
    pub fn derivative(
        &self,
        dep: usize,
        alpha: &[u32],
        pool: &ExprPool,
    ) -> Result<ExprId, PdeError> {
        self.check(dep, alpha)?;
        let d = &self.deps[dep];
        let orders: Vec<u32> = d.slots.iter().map(|&i| alpha[i]).collect();
        Ok(build(d.applied, &orders, pool))
    }

    /// `(dep, α)` when `sym` is a jet coordinate of this space (of any
    /// order), `None` otherwise.  The name must be the canonical spelling and
    /// the symbol must have the space's domain.
    pub fn decode_jet(&self, sym: ExprId, pool: &ExprPool) -> Option<(usize, MultiIndex)> {
        let (name, domain) = pool.with(sym, |d| match d {
            ExprData::Symbol { name, domain, .. } => Some((name.clone(), *domain)),
            _ => None,
        })?;
        if domain != self.domain {
            return None;
        }
        let n = self.indep.len();
        // Longest unknown name first, so `uv_x` is not read as `u` + `v_x`.
        let mut order: Vec<usize> = (0..self.deps.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.deps[i].name.len()));
        for dep in order {
            let base = &self.deps[dep].name;
            let Some(rest) = name.strip_prefix(base.as_str()) else {
                continue;
            };
            let mut alpha = vec![0u32; n];
            let ok = match self.naming {
                JetNaming::Subscript => {
                    if rest.is_empty() {
                        true
                    } else if let Some(s) = rest.strip_prefix('_') {
                        let single = self.indep_names.iter().all(|v| v.chars().count() == 1);
                        let parts: Vec<String> = if single {
                            s.chars().map(|c| c.to_string()).collect()
                        } else {
                            s.split('_').map(str::to_string).collect()
                        };
                        !parts.is_empty()
                            && parts.iter().all(|p| {
                                match self.indep_names.iter().position(|v| v == p) {
                                    Some(i) => {
                                        alpha[i] += 1;
                                        true
                                    }
                                    None => false,
                                }
                            })
                    } else {
                        false
                    }
                }
                JetNaming::Prime => {
                    alpha[0] = rest.chars().count() as u32;
                    rest.chars().all(|c| c == '\'')
                }
                JetNaming::Dot | JetNaming::DaeDt => {
                    // Decode by re-encoding: these spellings are not
                    // prefix-structured in a way worth parsing.
                    let max = self.max_order.saturating_add(8);
                    match (0..=max).find(|&k| self.jet_name(dep, &[k]) == name) {
                        Some(k) => {
                            alpha[0] = k;
                            true
                        }
                        None => false,
                    }
                }
            };
            if !ok || self.check(dep, &alpha).is_err() {
                continue;
            }
            // Only the canonical spelling (`u_xt`, never `u_tx`).
            if self.jet_name(dep, &alpha) == name {
                return Some((dep, alpha));
            }
        }
        None
    }

    /// `(dep, α)` for an applied unknown `u(x…)` or one of its derivatives
    /// at the declared arguments; `Ok(None)` for any node that is not an
    /// unknown of this space (an arbitrary coefficient function `a(x)` is
    /// not).
    ///
    /// # Errors
    ///
    /// `E-PDE-003` when the node is an unknown of this space evaluated at other
    /// arguments (`u(x, 0)`).
    pub fn decode_derivative(
        &self,
        node: ExprId,
        pool: &ExprPool,
    ) -> Result<Option<(usize, MultiIndex)>, PdeError> {
        let Some(v) = view(node, pool) else {
            return Ok(None);
        };
        let Some(dep) = self.deps.iter().position(|d| d.name == v.name) else {
            return Ok(None);
        };
        let d = &self.deps[dep];
        let declared = pool.with(d.applied, |data| match data {
            ExprData::Func { args, .. } => args.clone(),
            _ => vec![],
        });
        if v.args != declared {
            return Err(PdeError::OffJet(format!(
                "{} is {} evaluated at ({}), not at its declared arguments",
                pool.display(node),
                d.name,
                v.args
                    .iter()
                    .map(|&a| pool.display(a).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let mut alpha = vec![0u32; self.indep.len()];
        for (j, &k) in v.orders.iter().enumerate() {
            alpha[d.slots[j]] += k;
        }
        Ok(Some((dep, alpha)))
    }

    /// Every jet coordinate in the space up to [`Self::max_order`]: unknown by
    /// unknown, order by order, multi-indices descending.
    pub fn coordinates(&self, pool: &ExprPool) -> Vec<ExprId> {
        let mut out = Vec::new();
        for dep in 0..self.deps.len() {
            for k in 0..=self.max_order {
                for a in multi_indices(self.indep.len(), k) {
                    if let Ok(j) = self.jet(dep, &a, pool) {
                        out.push(j);
                    }
                }
            }
        }
        out
    }

    /// Replace every unknown and derivative of one in `expr` by its jet
    /// coordinate.  Symbols that already are jet coordinates of this space are
    /// left as they are, so the map is idempotent.
    ///
    /// # Errors
    ///
    /// `E-PDE-003` for an unknown off its declared arguments (`u(x, 0)`),
    /// `E-PDE-004` for a derivative above [`Self::max_order`].
    pub fn to_jets(&self, expr: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
        let mut map: HashMap<ExprId, ExprId> = HashMap::new();
        let mut seen = HashSet::new();
        let mut stack = vec![expr];
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some((dep, alpha)) = self.decode_derivative(n, pool)? {
                let order: u32 = alpha.iter().sum();
                if order > self.max_order {
                    return Err(PdeError::OrderExceeded(format!(
                        "{} has order {order}, the jet space has order {}",
                        pool.display(n),
                        self.max_order
                    )));
                }
                map.insert(n, self.jet(dep, &alpha, pool)?);
                continue;
            }
            push_children(n, pool, &mut stack);
        }
        Ok(crate::kernel::subs::subs(expr, &map, pool))
    }

    /// Replace every jet coordinate in `expr` by the derivative it stands for
    /// — the inverse of [`Self::to_jets`].
    pub fn from_jets(&self, expr: ExprId, pool: &ExprPool) -> ExprId {
        let mut map: HashMap<ExprId, ExprId> = HashMap::new();
        for sym in symbols_of(expr, pool) {
            if let Some((dep, alpha)) = self.decode_jet(sym, pool) {
                if let Ok(d) = self.derivative(dep, &alpha, pool) {
                    map.insert(sym, d);
                }
            }
        }
        if map.is_empty() {
            return expr;
        }
        crate::kernel::subs::subs(expr, &map, pool)
    }

    /// The jet coordinates occurring in `expr` (which may be written with
    /// jets, derivatives, or both), sorted by the orderly ranking.
    ///
    /// # Errors
    ///
    /// As [`Self::to_jets`].
    pub fn jets_in(&self, expr: ExprId, pool: &ExprPool) -> Result<Vec<JetCoordinate>, PdeError> {
        let e = self.to_jets(expr, pool)?;
        let mut found: Vec<JetCoordinate> = Vec::new();
        for sym in symbols_of(e, pool) {
            if let Some((dep, alpha)) = self.decode_jet(sym, pool) {
                found.push(JetCoordinate {
                    dep,
                    alpha,
                    symbol: sym,
                });
            }
        }
        found.sort_by(|a, b| ranking_cmp(a, b));
        found.dedup();
        Ok(found)
    }

    /// The highest total order of a jet coordinate in `expr`, `None` when it
    /// mentions none.
    ///
    /// # Errors
    ///
    /// As [`Self::to_jets`].
    pub fn order_of(&self, expr: ExprId, pool: &ExprPool) -> Result<Option<u32>, PdeError> {
        Ok(self
            .jets_in(expr, pool)?
            .iter()
            .map(|c| c.alpha.iter().sum())
            .max())
    }

    /// The total derivative `D_{xᵢ} F = ∂F/∂xᵢ + Σ_{u, α} u_{α+eᵢ} ∂F/∂u_α`
    /// of a function `F` on the jet space (written with jets, derivatives or
    /// both; returned in jets and simplified).  `i` indexes
    /// [`Self::independent`].
    ///
    /// On an expression written with derivative nodes this agrees with
    /// [`crate::diff::diff`] after [`Self::to_jets`]; the jet form is what the
    /// prolongation formula, the Euler operator and the symmetry machinery
    /// need, because there `u_x` and `u` are independent coordinates.
    ///
    /// # Errors
    ///
    /// `E-PDE-002` when `i` is out of range; as [`Self::to_jets`]; a
    /// `DiffError` from a coefficient that cannot be differentiated is
    /// reported as `E-PDE-006`.
    pub fn total_derivative(
        &self,
        f: ExprId,
        i: usize,
        pool: &ExprPool,
    ) -> Result<ExprId, PdeError> {
        if i >= self.indep.len() {
            return Err(PdeError::InconsistentVariables(format!(
                "variable index {i} out of range (the space has {})",
                self.indep.len()
            )));
        }
        // Accept jets above max_order in the input (an iterated total
        // derivative climbs one order per step).
        let big = self.prolonged(u32::MAX / 4);
        let f = big.to_jets(f, pool)?;
        let derr = |e: crate::diff::DiffError| {
            PdeError::NotRationalInJets(format!("cannot differentiate: {e}"))
        };
        let mut terms = vec![diff(f, self.indep[i], pool).map_err(derr)?.value];
        for c in big.jets_in(f, pool)? {
            if !self.depends_on(c.dep, i) {
                continue;
            }
            let partial = diff(f, c.symbol, pool).map_err(derr)?.value;
            if partial == pool.integer(0) {
                continue;
            }
            let mut a = c.alpha.clone();
            a[i] += 1;
            terms.push(pool.mul(vec![self.jet(c.dep, &a, pool)?, partial]));
        }
        let sum = if terms.len() == 1 {
            terms[0]
        } else {
            pool.add(terms)
        };
        Ok(crate::simplify::engine::simplify(sum, pool).value)
    }

    /// The iterated total derivative `D^α F`.
    ///
    /// # Errors
    ///
    /// As [`Self::total_derivative`].
    pub fn total_derivative_multi(
        &self,
        f: ExprId,
        alpha: &[u32],
        pool: &ExprPool,
    ) -> Result<ExprId, PdeError> {
        if alpha.len() != self.indep.len() {
            return Err(PdeError::InconsistentVariables(format!(
                "multi-index of length {} for {} independent variables",
                alpha.len(),
                self.indep.len()
            )));
        }
        let mut cur = self
            .to_jets(f, pool)
            .or_else(|_| self.prolonged(u32::MAX / 4).to_jets(f, pool))?;
        for (i, &k) in alpha.iter().enumerate() {
            for _ in 0..k {
                cur = self.total_derivative(cur, i, pool)?;
            }
        }
        Ok(cur)
    }

    /// The `k`-th prolongation of a system: every `D^β Fⱼ` with `|β| ≤ k`,
    /// equation by equation, `β` in graded descending order, in jets.
    /// Duplicates (and zeros) are dropped.
    ///
    /// # Errors
    ///
    /// As [`Self::total_derivative`].
    pub fn prolong(
        &self,
        equations: &[ExprId],
        k: u32,
        pool: &ExprPool,
    ) -> Result<Vec<ExprId>, PdeError> {
        let n = self.indep.len();
        let zero = pool.integer(0);
        let mut out: Vec<ExprId> = Vec::new();
        for &f in equations {
            // Memoised on β so each D^β F costs one total derivative.
            let mut level: HashMap<MultiIndex, ExprId> = HashMap::new();
            let base = self.prolonged(u32::MAX / 4).to_jets(f, pool)?;
            level.insert(vec![0; n], base);
            let mut all: Vec<(MultiIndex, ExprId)> = vec![(vec![0; n], base)];
            for ord in 1..=k {
                let mut next: HashMap<MultiIndex, ExprId> = HashMap::new();
                for beta in multi_indices(n, ord) {
                    // Differentiate a parent β − e_i (the first nonzero slot).
                    let i = beta.iter().position(|&b| b > 0).expect("ord ≥ 1");
                    let mut parent = beta.clone();
                    parent[i] -= 1;
                    let p = level[&parent];
                    let d = self.total_derivative(p, i, pool)?;
                    next.insert(beta.clone(), d);
                    all.push((beta, d));
                }
                level = next;
            }
            for (_, e) in all {
                if e != zero && !out.contains(&e) {
                    out.push(e);
                }
            }
        }
        Ok(out)
    }

    /// The system as polynomials in the jet coordinates, for
    /// `GroebnerBasis` / `rosenfeld_groebner`: each equation is converted to
    /// jets, put over a common denominator and cancelled; the numerator is
    /// the polynomial equation and a non-constant denominator is recorded as
    /// a non-vanishing condition.
    ///
    /// # Errors
    ///
    /// `E-PDE-006` when an equation is not rational in the jets (`sin(u)`,
    /// `exp(u_x)`, `u^(1/2)`); as [`Self::to_jets`].
    pub fn to_rational_system(
        &self,
        equations: &[ExprId],
        pool: &ExprPool,
    ) -> Result<JetSystem, PdeError> {
        if equations.is_empty() {
            return Err(PdeError::SystemShape("empty system".into()));
        }
        let mut jets_eqs = Vec::with_capacity(equations.len());
        let mut coords: Vec<JetCoordinate> = Vec::new();
        for &e in equations {
            let j = self.to_jets(e, pool)?;
            for c in self.jets_in(j, pool)? {
                if !coords.contains(&c) {
                    coords.push(c);
                }
            }
            jets_eqs.push(j);
        }
        coords.sort_by(ranking_cmp);
        let variables: Vec<ExprId> = coords.iter().map(|c| c.symbol).collect();
        let jet_set: HashSet<ExprId> = variables.iter().copied().collect();
        let mut gens_vars = variables.clone();
        gens_vars.extend(self.indep.iter().copied());
        let mut out = JetSystem {
            equations: Vec::new(),
            nonzero: Vec::new(),
            variables,
            independent: self.indep.clone(),
            parameters: Vec::new(),
        };
        let one = pool.integer(1);
        for (&orig, &j) in equations.iter().zip(&jets_eqs) {
            let (rf, gens) =
                crate::poly::cancel::rational_over_generators(j, gens_vars.clone(), pool).map_err(
                    |e| PdeError::NotRationalInJets(format!("{}: {e}", pool.display(orig))),
                )?;
            for &g in &gens[gens_vars.len()..] {
                if symbols_of(g, pool).iter().any(|s| jet_set.contains(s)) {
                    return Err(PdeError::NotRationalInJets(format!(
                        "{} contains {}, which is not rational in the jets",
                        pool.display(orig),
                        pool.display(self.from_jets(g, pool))
                    )));
                }
                if !out.parameters.contains(&g) {
                    out.parameters.push(g);
                }
            }
            out.equations.push(rf.numer.to_expr(pool));
            let den = rf.denom.to_expr(pool);
            if den != one
                && !matches!(pool.get(den), ExprData::Integer(_))
                && !out.nonzero.contains(&den)
            {
                out.nonzero.push(den);
            }
        }
        Ok(out)
    }

    // -- ODE adapters ------------------------------------------------------

    /// The ODE `equation = 0` in one unknown `y(x)` as `dsolve`'s
    /// [`crate::ode::dsolve::OdeInput`]: the derivatives become the `y'`, `y''`, …
    /// symbols `OdeInput` expects ([`JetNaming::Prime`], real domain), up to
    /// the highest order the equation mentions.
    ///
    /// # Errors
    ///
    /// `E-PDE-008` unless the space has exactly one unknown;
    /// `E-PDE-009` unless it has exactly one independent variable;
    /// `E-PDE-007` when the equation mentions no derivative of the unknown;
    /// as [`Self::to_jets`].
    pub fn to_ode_input(
        &self,
        equation: ExprId,
        pool: &ExprPool,
    ) -> Result<crate::ode::dsolve::OdeInput, PdeError> {
        if self.deps.len() != 1 {
            return Err(PdeError::SystemShape(
                "an OdeInput has exactly one unknown".into(),
            ));
        }
        let space = self
            .clone()
            .with_naming(JetNaming::Prime)?
            .with_domain(Domain::Real);
        let eq = space.to_jets(equation, pool)?;
        let order = space.order_of(eq, pool)?.unwrap_or(0);
        if order == 0 {
            return Err(PdeError::NotADifferentialEquation(format!(
                "{} mentions no derivative of {}",
                pool.display(equation),
                self.deps[0].name
            )));
        }
        let derivs = (1..=order)
            .map(|k| space.jet(0, &[k], pool))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(crate::ode::dsolve::OdeInput {
            x: self.indep[0],
            y: space.jet(0, &[0], pool)?,
            derivs,
            equation: eq,
        })
    }

    /// The DAE convention: `(variable, derivative)` pairs
    /// `(y, dy/dt), (dy/dt, ddy/dt/dt), …` for every unknown up to order
    /// [`Self::max_order`], named as the DAE index reduction names them
    /// ([`JetNaming::DaeDt`]).
    ///
    /// # Errors
    ///
    /// `E-PDE-009` unless the space has exactly one independent variable.
    pub fn dae_pairs(&self, pool: &ExprPool) -> Result<Vec<(ExprId, ExprId)>, PdeError> {
        let space = self.clone().with_naming(JetNaming::DaeDt)?;
        let mut out = Vec::new();
        for dep in 0..self.deps.len() {
            for k in 0..self.max_order {
                out.push((space.jet(dep, &[k], pool)?, space.jet(dep, &[k + 1], pool)?));
            }
        }
        Ok(out)
    }
}

/// Orderly ranking: higher total order first, then unknown index, then
/// multi-index descending.
fn ranking_cmp(a: &JetCoordinate, b: &JetCoordinate) -> std::cmp::Ordering {
    let oa: u32 = a.alpha.iter().sum();
    let ob: u32 = b.alpha.iter().sum();
    ob.cmp(&oa)
        .then(a.dep.cmp(&b.dep))
        .then_with(|| b.alpha.cmp(&a.alpha))
}

pub(crate) fn push_children(n: ExprId, pool: &ExprPool, stack: &mut Vec<ExprId>) {
    pool.with(n, |d| match d {
        ExprData::Add(xs) | ExprData::Mul(xs) => stack.extend_from_slice(xs),
        ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => {
            stack.extend_from_slice(args)
        }
        ExprData::Pow { base, exp } => {
            stack.push(*base);
            stack.push(*exp);
        }
        ExprData::Piecewise { branches, default } => {
            for &(c, v) in branches {
                stack.push(c);
                stack.push(v);
            }
            stack.push(*default);
        }
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
            stack.push(*var);
            stack.push(*body);
        }
        ExprData::BigO(a) => stack.push(*a),
        ExprData::RootSum { poly, var, body } => {
            stack.push(*poly);
            stack.push(*var);
            stack.push(*body);
        }
        _ => {}
    });
}

/// Every symbol node reachable from `expr` (bound ones included).
pub(crate) fn symbols_of(expr: ExprId, pool: &ExprPool) -> BTreeSet<ExprId> {
    let mut out = BTreeSet::new();
    let mut seen = HashSet::new();
    let mut stack = vec![expr];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if pool.with(n, |d| matches!(d, ExprData::Symbol { .. })) {
            out.insert(n);
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    out
}

// ---------------------------------------------------------------------------
// functions_to_jets — the multi-variable functions_to_symbols
// ---------------------------------------------------------------------------

/// One replaced node from [`functions_to_jets`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JetSymbol {
    /// The unknown `u(x, t)` this entry belongs to.
    pub func: ExprId,
    /// Derivative orders per argument slot of `func` (all zero for `u`).
    pub orders: Vec<u32>,
    /// The node that was replaced.
    pub node: ExprId,
    /// The symbol that replaced it.
    pub symbol: ExprId,
}

/// The multi-variable generalisation of
/// [`crate::diff::functions_to_symbols`]: replace each unknown `u(x, t)` in
/// `funcs` and every derivative of it occurring in `exprs` by a symbol named
/// by the jet convention (`u`, `u_x`, `u_xt`, …, in `domain`; a trailing `_`
/// is added while the name is taken by a symbol already in `exprs`).
///
/// Returns the rewritten expressions and the replacement table; substituting
/// each `symbol → node` back undoes the rewrite.  `functions_to_symbols`
/// (single variable, `q_dot` names, `E-DIFF-007` for more than one argument)
/// is unchanged.
///
/// # Errors
///
/// `E-PDE-001` / `E-PDE-002` for a malformed unknown; `E-PDE-003` when an
/// unknown occurs at arguments other than its declared ones.
pub fn functions_to_jets(
    exprs: &[ExprId],
    funcs: &[ExprId],
    domain: Domain,
    pool: &ExprPool,
) -> Result<(Vec<ExprId>, Vec<JetSymbol>), PdeError> {
    let space = JetSpace::from_unknowns(funcs, u32::MAX / 4, pool)?.with_domain(domain);
    let mut taken: Vec<String> = Vec::new();
    for &e in exprs {
        for s in symbols_of(e, pool) {
            if let Some(n) = symbol_name(s, pool) {
                taken.push(n);
            }
        }
    }
    // Collect (dep, α) occurring, in ranking order (low order first here, so
    // the table reads u, u_t, u_x, u_tt, …).
    let mut found: Vec<(usize, MultiIndex, ExprId)> = Vec::new();
    for &e in exprs {
        let mut seen = HashSet::new();
        let mut stack = vec![e];
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some((dep, alpha)) = space.decode_derivative(n, pool)? {
                if !found.iter().any(|(_, _, m)| *m == n) {
                    found.push((dep, alpha, n));
                }
                continue;
            }
            push_children(n, pool, &mut stack);
        }
    }
    found.sort_by(|a, b| {
        let oa: u32 = a.1.iter().sum();
        let ob: u32 = b.1.iter().sum();
        a.0.cmp(&b.0).then(oa.cmp(&ob)).then_with(|| b.1.cmp(&a.1))
    });
    let mut table = Vec::with_capacity(found.len());
    let mut map: HashMap<ExprId, ExprId> = HashMap::new();
    for (dep, alpha, node) in found {
        let mut name = space.jet_name(dep, &alpha);
        while taken.contains(&name) {
            name.push('_');
        }
        taken.push(name.clone());
        let sym = pool.symbol(name, domain);
        let d = &space.deps[dep];
        table.push(JetSymbol {
            func: d.applied,
            orders: d.slots.iter().map(|&i| alpha[i]).collect(),
            node,
            symbol: sym,
        });
        map.insert(node, sym);
    }
    let out = exprs
        .iter()
        .map(|&e| crate::kernel::subs::subs(e, &map, pool))
        .collect();
    Ok((out, table))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;

    fn setup() -> (ExprPool, ExprId, ExprId, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let u = pool.func("u", vec![x, t]);
        (pool, x, t, u)
    }

    fn d(e: ExprId, v: ExprId, pool: &ExprPool) -> ExprId {
        diff(e, v, pool).unwrap().value
    }

    #[test]
    fn names_are_stable_and_readable() {
        let (pool, x, t, u) = setup();
        let j = JetSpace::new(&[x, t], &[u], 3, &pool).unwrap();
        let names: Vec<String> = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]]
            .iter()
            .map(|a| j.jet_name(0, a))
            .collect();
        assert_eq!(names, ["u", "u_x", "u_t", "u_xt", "u_xx", "u_xxt"]);
        let x1 = pool.symbol("x1", Domain::Real);
        let x2 = pool.symbol("x2", Domain::Real);
        let w = pool.func("w", vec![x1, x2]);
        let j2 = JetSpace::new(&[x1, x2], &[w], 2, &pool).unwrap();
        assert_eq!(j2.jet_name(0, &[1, 1]), "w_x1_x2");
        assert_eq!(j2.jet_name(0, &[2, 0]), "w_x1_x1");
        let s = j2.jet(0, &[1, 1], &pool).unwrap();
        assert_eq!(j2.decode_jet(s, &pool), Some((0, vec![1, 1])));
    }

    #[test]
    fn bijection_round_trips() {
        let (pool, x, t, u) = setup();
        let j = JetSpace::new(&[x, t], &[u], 3, &pool).unwrap();
        let uxt = d(d(u, x, &pool), t, &pool);
        let uxx = d(d(u, x, &pool), x, &pool);
        let e = pool.add(vec![
            uxt,
            pool.mul(vec![u, uxx]),
            pool.func("sin", vec![d(u, t, &pool)]),
        ]);
        let je = j.to_jets(e, &pool).unwrap();
        assert!(!crate::diff::undefined::has_undefined_function(je, &pool));
        let shown = pool.display(je).to_string();
        assert!(
            shown.contains("u_xt") && shown.contains("u_xx") && shown.contains("u_t"),
            "{shown}"
        );
        assert_eq!(j.from_jets(je, &pool), e);
        // idempotent
        assert_eq!(j.to_jets(je, &pool).unwrap(), je);
        for c in j.coordinates(&pool) {
            let (dep, a) = j.decode_jet(c, &pool).unwrap();
            let node = j.derivative(dep, &a, &pool).unwrap();
            assert_eq!(j.to_jets(node, &pool).unwrap(), c);
        }
        assert_eq!(j.coordinates(&pool).len(), 10); // C(2+3, 3)
    }

    #[test]
    fn non_canonical_names_are_not_jets() {
        let (pool, x, t, u) = setup();
        let j = JetSpace::new(&[x, t], &[u], 3, &pool).unwrap();
        assert!(j
            .decode_jet(pool.symbol("u_tx", Domain::Real), &pool)
            .is_none());
        assert!(j
            .decode_jet(pool.symbol("u_y", Domain::Real), &pool)
            .is_none());
        assert!(j
            .decode_jet(pool.symbol("u_x", Domain::Complex), &pool)
            .is_none());
    }

    #[test]
    fn off_jet_and_order_errors() {
        let (pool, x, t, u) = setup();
        let j = JetSpace::new(&[x, t], &[u], 1, &pool).unwrap();
        let u0 = pool.func("u", vec![x, pool.integer(0)]);
        assert_eq!(j.to_jets(u0, &pool).unwrap_err().code(), "E-PDE-003");
        let uxx = d(d(u, x, &pool), x, &pool);
        assert_eq!(j.to_jets(uxx, &pool).unwrap_err().code(), "E-PDE-004");
        // an arbitrary coefficient function is not an unknown
        let a = pool.func("a", vec![x]);
        assert_eq!(j.to_jets(a, &pool).unwrap(), a);
        let bad = JetSpace::new(&[x], &[u], 1, &pool).unwrap_err();
        assert_eq!(bad.code(), "E-PDE-002");
        let notu = JetSpace::new(&[x], &[pool.func("sin", vec![x])], 1, &pool).unwrap_err();
        assert_eq!(notu.code(), "E-PDE-001");
    }

    #[test]
    fn total_derivative_matches_diff() {
        let (pool, x, t, u) = setup();
        let j = JetSpace::new(&[x, t], &[u], 2, &pool).unwrap();
        let ux = d(u, x, &pool);
        // F = x·u·u_x + sin(u_t)
        let f = pool.add(vec![
            pool.mul(vec![x, u, ux]),
            pool.func("sin", vec![d(u, t, &pool)]),
        ]);
        for (i, v) in [x, t].into_iter().enumerate() {
            let dj = j.total_derivative(f, i, &pool).unwrap();
            let want = j.prolonged(1).to_jets(d(f, v, &pool), &pool).unwrap();
            let diffe = pool.add(vec![dj, pool.mul(vec![pool.integer(-1), want])]);
            assert_eq!(
                crate::simplify::engine::simplify(diffe, &pool).value,
                pool.integer(0),
                "D_{i}"
            );
        }
    }

    #[test]
    fn prolongation_counts() {
        let (pool, x, t, u) = setup();
        let j = JetSpace::new(&[x, t], &[u], 2, &pool).unwrap();
        // heat equation u_t - u_xx
        let heat = pool.add(vec![
            d(u, t, &pool),
            pool.mul(vec![pool.integer(-1), d(d(u, x, &pool), x, &pool)]),
        ]);
        let p = j.prolong(&[heat], 2, &pool).unwrap();
        assert_eq!(p.len(), 6); // 1 + 2 + 3
        let ut = j.jet(0, &[0, 1], &pool).unwrap();
        let uxx = j.jet(0, &[2, 0], &pool).unwrap();
        assert!(p.contains(
            &crate::simplify::engine::simplify(
                pool.add(vec![ut, pool.mul(vec![pool.integer(-1), uxx])]),
                &pool
            )
            .value
        ));
    }

    #[test]
    fn rational_system() {
        let (pool, x, t, u) = setup();
        let j = JetSpace::new(&[x, t], &[u], 2, &pool).unwrap();
        let ux = d(u, x, &pool);
        // u_t - u_x/u = 0  →  u·u_t − u_x, u ≠ 0
        let e = pool.add(vec![
            d(u, t, &pool),
            pool.mul(vec![pool.integer(-1), ux, pool.pow(u, pool.integer(-1))]),
        ]);
        let sys = j.to_rational_system(&[e], &pool).unwrap();
        assert_eq!(sys.equations.len(), 1);
        assert_eq!(sys.nonzero, vec![j.jet(0, &[0, 0], &pool).unwrap()]);
        assert_eq!(sys.variables.len(), 3);
        let bad = pool.func("sin", vec![u]);
        assert_eq!(
            j.to_rational_system(&[bad], &pool).unwrap_err().code(),
            "E-PDE-006"
        );
        // a coefficient sin(x) is a parameter, not a failure
        let ok = pool.add(vec![d(u, t, &pool), pool.func("sin", vec![x])]);
        let s2 = j.to_rational_system(&[ok], &pool).unwrap();
        assert_eq!(s2.parameters.len(), 1);
    }

    #[test]
    fn ode_conventions_are_adapters() {
        let pool = ExprPool::new();
        let tt = pool.symbol("t", Domain::Real);
        let q = pool.func("q", vec![tt]);
        let qd = d(q, tt, &pool);
        let qdd = d(qd, tt, &pool);
        let j = JetSpace::new(&[tt], &[q], 3, &pool).unwrap();
        // functions_to_symbols' names
        let dot = j.clone().with_naming(JetNaming::Dot).unwrap();
        let e = pool.add(vec![qdd, q]);
        let (old, _) =
            crate::diff::functions_to_symbols(&[e], &[q], tt, Domain::Real, &pool).unwrap();
        assert_eq!(dot.to_jets(e, &pool).unwrap(), old[0]);
        // dsolve's y' convention: y'' + y = 0 → C1 sin + C2 cos
        let input = j.to_ode_input(e, &pool).unwrap();
        assert_eq!(input.derivs.len(), 2);
        assert_eq!(pool.display(input.derivs[1]).to_string(), "q''");
        assert!(crate::ode::dsolve::dsolve(&input, &pool).is_ok());
        // DAE spelling
        let pairs = j.dae_pairs(&pool).unwrap();
        assert_eq!(pool.display(pairs[0].1).to_string(), "dq/dt");
        assert_eq!(pool.display(pairs[1].1).to_string(), "ddq/dt/dt");
        // and the ODE spellings refuse a PDE space
        let (pool2, x, t, u) = setup();
        let pj = JetSpace::new(&[x, t], &[u], 2, &pool2).unwrap();
        assert_eq!(
            pj.with_naming(JetNaming::Prime).unwrap_err().code(),
            "E-PDE-009"
        );
    }

    #[test]
    fn functions_to_jets_generalises_functions_to_symbols() {
        let (pool, x, t, u) = setup();
        let uxt = d(d(u, x, &pool), t, &pool);
        let clash = pool.symbol("u_x", Domain::Real);
        let e = pool.add(vec![uxt, d(u, x, &pool), clash]);
        let (out, table) = functions_to_jets(&[e], &[u], Domain::Real, &pool).unwrap();
        let names: Vec<String> = table
            .iter()
            .map(|e| pool.display(e.symbol).to_string())
            .collect();
        assert_eq!(names, ["u_x_", "u_xt"]);
        let back: HashMap<ExprId, ExprId> = table.iter().map(|e| (e.symbol, e.node)).collect();
        assert_eq!(crate::kernel::subs::subs(out[0], &back, &pool), e);
    }

    #[test]
    fn mixed_dependency_unknowns() {
        let (pool, x, t, u) = setup();
        let v = pool.func("v", vec![t]);
        let j = JetSpace::new(&[x, t], &[u, v], 2, &pool).unwrap();
        assert!(j.jet(1, &[1, 0], &pool).is_err());
        let vt = d(v, t, &pool);
        assert_eq!(
            pool.display(j.to_jets(vt, &pool).unwrap()).to_string(),
            "v_t"
        );
        // D_x of v_t is 0
        assert_eq!(j.total_derivative(vt, 0, &pool).unwrap(), pool.integer(0));
    }
}
