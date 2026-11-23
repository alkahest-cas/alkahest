//! The milestone-M6 tiers of `pdsolve`: nonlinear equations in two
//! independent variables, by Cole–Hopf, traveling-wave and similarity
//! reductions (see [`crate::pde::nonlinear`]).  Every candidate is verified
//! again by the dispatcher.

use crate::kernel::ExprPool;

use super::{Candidate, Ctx, PdeSolutionKind};

/// Two independent variables, no side conditions, a nonlinear equation.
pub(crate) fn applies_nonlinear(c: &Ctx) -> bool {
    c.indep.len() == 2 && c.problem.conditions().is_empty() && c.nonlinear
}

fn candidates(sols: Vec<crate::pde::nonlinear::ReducedSolution>) -> Vec<Candidate> {
    sols.into_iter()
        .map(|r| {
            let mut steps = r.steps.clone();
            for a in &r.assumptions {
                steps.push(format!("assumption: {a}"));
            }
            Candidate {
                value: r.value,
                implicit: false,
                kind: PdeSolutionKind::Special,
                constants: r.constants,
                side_conditions: Vec::new(),
                steps,
                general: None,
            }
        })
        .collect()
}

pub(crate) fn solve_traveling_wave(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let eq = ctx.problem.equations()[0];
    match crate::pde::nonlinear::capped(60, || {
        crate::pde::nonlinear::traveling_wave_solutions(eq, ctx.unknown, pool)
    }) {
        Ok(s) => candidates(s),
        Err(e) => {
            ctx.fail(format!("traveling_wave: {e}"));
            Vec::new()
        }
    }
}

pub(crate) fn solve_similarity(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let eq = ctx.problem.equations()[0];
    match crate::pde::nonlinear::capped(60, || {
        crate::pde::nonlinear::similarity_solutions(eq, ctx.unknown, pool)
    }) {
        Ok(mut s) => {
            // Prefer solutions depending on both variables.
            let indep = ctx.indep.clone();
            s.sort_by_key(|r| {
                !indep
                    .iter()
                    .all(|&x| crate::pde::solve::util::contains(r.value, x, pool))
            });
            candidates(s)
        }
        Err(e) => {
            ctx.fail(format!("similarity: {e}"));
            Vec::new()
        }
    }
}

pub(crate) fn solve_cole_hopf(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let eq = ctx.problem.equations()[0];
    match crate::pde::linearize::cole_hopf_families(eq, ctx.unknown, pool) {
        Ok(fams) => fams
            .into_iter()
            .map(|(value, constants)| Candidate {
                value,
                implicit: false,
                kind: PdeSolutionKind::Special,
                constants,
                side_conditions: Vec::new(),
                steps: vec![
                    "Cole–Hopf u = −(2ν/a) θ_x/θ with θ = 1 + Σ exp(kᵢx + νkᵢ²t + δᵢ), \
                     θ_t = ν θ_xx"
                        .into(),
                ],
                general: None,
            })
            .collect(),
        Err(e) => {
            ctx.fail(format!("cole_hopf: {e}"));
            Vec::new()
        }
    }
}
