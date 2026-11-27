//! Reading boundary and initial conditions as linear conditions on one face:
//! `Σ_k c_k ∂^k_v u(…, v = a, …) = g(other variables)`.

use std::collections::HashMap;

use crate::diff::undefined::view;
use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{Domain, ExprId, ExprPool};

use super::super::error::PdeError;
use super::super::jet::push_children;
use super::super::problem::PdeCondition;
use super::super::sturm_liouville::{d, simp};
use super::modes::sub;

/// A linear condition on the face `vars[slot] = at`.
#[derive(Clone, Debug)]
pub(crate) struct FaceCondition {
    pub slot: usize,
    pub at: ExprId,
    /// `coeffs[k]` multiplies `∂^k u` (normal derivative order `k`).
    pub coeffs: Vec<ExprId>,
    /// The data `g` (a function of the other variables).
    pub data: ExprId,
}

/// A condition relating two faces of one variable: `c·∂^k u(a) − c·∂^k u(b)
/// = 0` (half of a periodicity condition).
#[derive(Clone, Debug)]
pub(crate) struct PeriodicCondition {
    pub slot: usize,
    pub order: u32,
}

#[derive(Clone, Debug)]
pub(crate) enum Parsed {
    Face(FaceCondition),
    Periodic(PeriodicCondition),
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    simp(e, pool) == pool.integer(0)
}

/// Parse `c` for the unknown `name(vars…)`.
pub(crate) fn parse_condition(
    c: &PdeCondition,
    name: &str,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<Parsed, PdeError> {
    let r = c.residual(pool);
    let bad = |why: &str| {
        PdeError::SeparationCondition(format!(
            "{} = {}: {why}",
            pool.display(c.lhs),
            pool.display(c.rhs)
        ))
    };
    // Occurrences of the unknown.
    let mut occ: Vec<(ExprId, usize, ExprId, u32)> = Vec::new(); // node, slot, at, order
    let mut stack = vec![r];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if v.name == name {
                if v.args.len() != vars.len() {
                    return Err(bad("the unknown has the wrong number of arguments"));
                }
                let fixed: Vec<usize> = (0..vars.len()).filter(|&j| v.args[j] != vars[j]).collect();
                if fixed.len() != 1 {
                    return Err(bad(
                        "each occurrence of the unknown must fix exactly one variable",
                    ));
                }
                let j = fixed[0];
                if vars.iter().any(|&x| mentions_var(v.args[j], x, pool)) {
                    return Err(bad("the face must be a constant value of the variable"));
                }
                if (0..vars.len()).any(|i| i != j && v.orders[i] > 0) {
                    return Err(bad(
                        "only derivatives normal to the face (in the fixed variable) are supported",
                    ));
                }
                occ.push((n, j, simp(v.args[j], pool), v.orders[j]));
                continue;
            }
        }
        push_children(n, pool, &mut stack);
    }
    if occ.is_empty() {
        return Err(bad("the condition mentions no value of the unknown"));
    }
    let slot = occ[0].1;
    if occ.iter().any(|o| o.1 != slot) {
        return Err(bad("the condition involves two different faces"));
    }
    // Replace every occurrence by a symbol; read off coefficients.
    let mut map = HashMap::new();
    let mut syms = Vec::new();
    for (k, o) in occ.iter().enumerate() {
        let s = pool.symbol(format!("__bc_z{k}"), Domain::Real);
        map.insert(o.0, s);
        syms.push(s);
    }
    let lin = super::separate::replace_nodes(r, &map, pool);
    let zero0: HashMap<ExprId, ExprId> = syms.iter().map(|&s| (s, pool.integer(0))).collect();
    // The part with the unknown, without the data (which may be a
    // Piecewise the differentiation should not touch).
    let core = simp(sub(lin, subs(lin, &zero0, pool), pool), pool);
    let mut coef = Vec::new();
    for &s in &syms {
        let cf = simp(d(core, s, pool)?, pool);
        if syms.iter().any(|&z| mentions_var(cf, z, pool)) {
            return Err(bad("the condition is not linear in the unknown"));
        }
        if vars.iter().any(|&x| mentions_var(cf, x, pool)) {
            return Err(bad(
                "a coefficient of the unknown depends on a variable (separation needs constant \
                 coefficients on each face)",
            ));
        }
        coef.push(cf);
    }
    let zero: HashMap<ExprId, ExprId> = syms.iter().map(|&s| (s, pool.integer(0))).collect();
    let data = simp(
        pool.mul(vec![pool.integer(-1), subs(lin, &zero, pool)]),
        pool,
    );
    if mentions_var(data, vars[slot], pool) {
        return Err(bad("the data depends on the fixed variable"));
    }
    let mut points: Vec<ExprId> = occ.iter().map(|o| o.2).collect();
    points.sort();
    points.dedup();
    if points.len() == 1 {
        let maxo = occ.iter().map(|o| o.3).max().unwrap_or(0) as usize;
        let mut coeffs = vec![pool.integer(0); maxo + 1];
        for (o, &cf) in occ.iter().zip(&coef) {
            let k = o.3 as usize;
            coeffs[k] = simp(pool.add(vec![coeffs[k], cf]), pool);
        }
        return Ok(Parsed::Face(FaceCondition {
            slot,
            at: points[0],
            coeffs,
            data,
        }));
    }
    if points.len() == 2
        && occ.len() == 2
        && occ[0].3 == occ[1].3
        && is_zero(pool.add(vec![coef[0], coef[1]]), pool)
        && is_zero(data, pool)
    {
        return Ok(Parsed::Periodic(PeriodicCondition {
            slot,
            order: occ[0].3,
        }));
    }
    Err(bad(
        "a condition on two faces must be a periodicity condition u(a) = u(b) or u'(a) = u'(b)",
    ))
}
