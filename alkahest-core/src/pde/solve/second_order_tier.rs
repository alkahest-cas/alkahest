//! The M2 row of [`super::TIERS`]: `pdsolve` delegates a linear
//! second-order equation in two variables (and Cauchy data for it) to
//! [`crate::pde::second_order::solve_second_order_with`], which returns only
//! exactly verified answers; the dispatcher verifies them again.

use crate::deriv::log::SideCondition;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::pde::second_order::{solve_second_order_with, SecondOrderOptions};

use super::{Candidate, Ctx, PdeSolutionKind};

/// `e ≠ 0` side conditions become [`SideCondition::NonZero`]; other
/// predicates (a region) are recorded as steps.
fn side_condition(c: ExprId, pool: &ExprPool) -> Option<SideCondition> {
    pool.with(c, |d| match d {
        ExprData::Predicate {
            kind: PredicateKind::Ne,
            args,
        } if args.len() == 2 => Some(SideCondition::NonZero(args[0])),
        _ => None,
    })
}

pub(crate) fn solve(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let opts = SecondOrderOptions::new().with_prefix(&ctx.prefix);
    match solve_second_order_with(ctx.problem, &opts, pool) {
        Ok(s) => {
            let mut steps = vec![format!("M2 method: {}", s.method)];
            steps.extend(s.steps.iter().cloned());
            let mut side = Vec::new();
            for &c in &s.conditions {
                match side_condition(c, pool) {
                    Some(sc) => side.push(sc),
                    None => steps.push(format!("valid on {}", pool.display(c))),
                }
            }
            vec![Candidate {
                value: s.solution,
                implicit: false,
                kind: if s.kind == "general" {
                    PdeSolutionKind::General
                } else {
                    PdeSolutionKind::Particular
                },
                constants: s.parameters.clone(),
                side_conditions: side,
                steps,
                general: None,
            }]
        }
        Err(e) => {
            ctx.fail(format!("second_order: {e}"));
            Vec::new()
        }
    }
}
