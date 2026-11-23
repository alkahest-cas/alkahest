//! `pdsolve` — the symbolic PDE solver (milestone M1: first-order equations).
//!
//! # Design: a dispatcher over method tiers
//!
//! [`pdsolve`] classifies the problem once ([`Ctx`]) and walks the static
//! table [`TIERS`] in order.  A tier is a named method with a cheap
//! applicability test and a solver that returns **candidates**; the
//! dispatcher, not the tier, decides what is returned:
//!
//! 1. every candidate goes through verification — explicit ones through
//!    M0's [`verify_pde_solution`], implicit relations `Φ(x, u) = 0` through
//!    implicit differentiation (`u_{xᵢ} = −Φ_{xᵢ}/Φ_u` substituted into the
//!    equation, residual proved zero, plus the data checked on the relation);
//! 2. only `exactly_verified` candidates survive — a failed route is
//!    discarded (and named in the refusal), never returned;
//! 3. when the problem carries Cauchy data, a tier's *general* solution is
//!    first specialised to the data ([`cauchy`]) and the specialised answer is
//!    what gets verified, data included.
//!
//! Refusals use `E-PDE-050 … E-PDE-099`.
//!
//! # Extension points (later milestones)
//!
//! * **A new method is a new [`Tier`] row** in [`TIERS`]: a name (it becomes a
//!   valid `hint`), the milestone that owns it, an `applies` predicate over
//!   [`Ctx`] and a `solve` function returning [`Candidate`]s.  Order in the
//!   table is preference order.  M2 (second order in two variables), M4
//!   (separation of variables) and M6 (nonlinear reductions) add rows; the
//!   first-order rows below are guarded by `ctx.order == 1`.
//! * **[`Ctx`]** is where per-problem analysis lives.  It holds the problem,
//!   the equation in jets and the jet symbols; a milestone needing more (the
//!   principal symbol for M2, the domain for M4) reads it from
//!   [`Ctx::problem`] or adds a lazily computed field.
//! * **[`PdeSolutionKind`]** is `#[non_exhaustive]`: series, integral
//!   representations and reductions (M3–M6) are new variants.
//! * **Verification** is not per tier: a tier can never return an
//!   unverified answer, whatever it does internally.
//!
//! # First-order tiers (this milestone)
//!
//! | hint | equation | method |
//! |---|---|---|
//! | `first_order_linear` | `Σ aᵢ(x) u_{xᵢ} = c(x, u)`, `n ≥ 2` | characteristic coordinates: `n − 1` first integrals `ξ`, the reduced ODE along characteristics by `dsolve`, constant → `F(ξ)` |
//! | `first_order_quasilinear` | `Σ aᵢ(x, u) u_{xᵢ} = c(x, u)` | Lagrange: `n` first integrals of `dxᵢ/aᵢ = du/c`, `Φ(ξ₁, …, ξₙ) = 0`, explicit when `u` isolates |
//! | `clairaut` | `u = Σ xᵢpᵢ + f(p)` | `u = Σ aᵢxᵢ + f(a)`, singular integral by envelope |
//! | `standard_f_p` | `f(p₁, …, pₙ) = 0` | `u = Σ aᵢxᵢ + b` |
//! | `standard_f_up` | `f(u, p, q) = 0` | `u = U(x + a y)` |
//! | `separable` | `Σ fᵢ(xᵢ, pᵢ) = 0` (Hamilton–Jacobi) | additive separation |
//! | `charpit` | `F(x, y, u, p, q) = 0` | Lagrange–Charpit: a first integral of Charpit's system, `p, q` solved, `du = p dx + q dy` integrated |

use std::collections::HashSet;

use crate::deriv::log::SideCondition;
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::error::PdeError;
use super::jet::{decode_unknown, JetSpace};
use super::problem::{ArbitraryFunctions, PdeCondition, PdeProblem};
use super::verify::{verify_pde_solution, PdeVerification, ResidualCheck, VerificationStatus};

pub(crate) mod cauchy;
pub(crate) mod integrals;
pub(crate) mod linear;
pub(crate) mod nonlinear;
pub(crate) mod util;

use util::*;

/// What a [`PdeSolution`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PdeSolutionKind {
    /// A general solution `u = …` containing arbitrary functions `F1, …`.
    General,
    /// A solution of the problem's side conditions (no arbitrary function).
    Particular,
    /// A relation `Φ(x, u) = 0` (the value is `Φ`): an implicit general
    /// solution `F1(ξ₁, ξ₂) = 0`, or an implicit solution of a Cauchy problem.
    Implicit,
    /// A complete integral: `n` arbitrary constants for `n` variables.
    CompleteIntegral,
    /// A singular integral: the envelope of a complete integral.
    SingularIntegral,
}

impl PdeSolutionKind {
    /// `"general"`, `"particular"`, `"implicit"`, `"complete_integral"`,
    /// `"singular_integral"`.
    pub fn as_str(self) -> &'static str {
        match self {
            PdeSolutionKind::General => "general",
            PdeSolutionKind::Particular => "particular",
            PdeSolutionKind::Implicit => "implicit",
            PdeSolutionKind::CompleteIntegral => "complete_integral",
            PdeSolutionKind::SingularIntegral => "singular_integral",
        }
    }
}

/// A verified answer of [`pdsolve`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PdeSolution {
    /// The unknown `u(x, y)`.
    pub unknown: ExprId,
    /// `u = value` when [`Self::implicit`] is false; otherwise the relation
    /// `value = 0`, written with the applied unknown `u(x, y)` in it.
    pub value: ExprId,
    /// Whether [`Self::value`] is a relation `Φ(x, u) = 0`.
    pub implicit: bool,
    /// What kind of solution this is.
    pub kind: PdeSolutionKind,
    /// The tier that produced it (a valid `hint`).
    pub method: &'static str,
    /// Arbitrary functions in the answer (`F1`, …), kept symbolic.
    pub arbitrary_functions: Vec<String>,
    /// Arbitrary constants of a complete integral (`C1`, …).
    pub constants: Vec<ExprId>,
    /// Hypotheses under which the answer is valid: non-characteristic data
    /// (`Jacobian ≠ 0`), the breaking time (`t_b − t > 0`), a non-vanishing
    /// `Φ_u` for an implicit relation.
    pub side_conditions: Vec<SideCondition>,
    /// The verification record (always `exactly_verified` for a returned
    /// solution).
    pub verification: PdeVerification,
    /// Human-readable derivation steps.
    pub steps: Vec<String>,
    /// Further verified answers of the same call (a singular integral next
    /// to a complete integral).
    pub alternatives: Vec<PdeSolution>,
}

/// Options of [`pdsolve`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PdsolveOptions {
    /// Restrict to one tier (see [`pdsolve_hints`]).
    pub hint: Option<String>,
    /// Prefix of the arbitrary functions (`"F"` → `F1, F2, …`).
    pub function_prefix: String,
}

impl Default for PdsolveOptions {
    fn default() -> Self {
        PdsolveOptions {
            hint: None,
            function_prefix: "F".to_string(),
        }
    }
}

impl PdsolveOptions {
    /// Options with a hint.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Options with an arbitrary-function prefix.
    pub fn with_function_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.function_prefix = prefix.into();
        self
    }
}

/// A candidate answer from a tier, before verification.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub value: ExprId,
    pub implicit: bool,
    pub kind: PdeSolutionKind,
    pub constants: Vec<ExprId>,
    pub side_conditions: Vec<SideCondition>,
    pub steps: Vec<String>,
    /// Structure kept for Cauchy fitting.
    pub general: Option<cauchy::GeneralForm>,
}

/// One solving method.  See the module docs, "Extension points".
pub struct Tier {
    /// The tier's name; also its `hint`.
    pub name: &'static str,
    /// The milestone that owns it.
    pub milestone: &'static str,
    /// One-line description.
    pub description: &'static str,
    applies: fn(&Ctx) -> bool,
    solve: fn(&mut Ctx, &ExprPool) -> Vec<Candidate>,
}

/// The method table, in preference order.
pub static TIERS: &[Tier] = &[
    Tier {
        name: "first_order_linear",
        milestone: "M1",
        description: "linear / semilinear: characteristic coordinates + dsolve along characteristics",
        applies: |c| c.order == 1 && c.affine.is_some(),
        solve: linear::solve_linear,
    },
    Tier {
        name: "first_order_quasilinear",
        milestone: "M1",
        description: "quasilinear: Lagrange's method, implicit general solution Φ(ξ₁, …, ξₙ) = 0",
        applies: |c| c.order == 1 && c.affine.is_some(),
        solve: linear::solve_quasilinear,
    },
    Tier {
        name: "clairaut",
        milestone: "M1",
        description: "Clairaut u = Σ xᵢpᵢ + f(p): complete and singular integrals",
        applies: |c| c.order == 1 && c.affine.is_none(),
        solve: nonlinear::solve_clairaut,
    },
    Tier {
        name: "standard_f_p",
        milestone: "M1",
        description: "f(p₁, …, pₙ) = 0: complete integral u = Σ aᵢxᵢ + b",
        applies: |c| c.order == 1 && c.affine.is_none(),
        solve: nonlinear::solve_f_p,
    },
    Tier {
        name: "standard_f_up",
        milestone: "M1",
        description: "f(u, p, q) = 0: complete integral through u = U(x + a y)",
        applies: |c| c.order == 1 && c.affine.is_none() && c.indep.len() == 2,
        solve: nonlinear::solve_f_up,
    },
    Tier {
        name: "separable",
        milestone: "M1",
        description: "additively separable Σ fᵢ(xᵢ, pᵢ) = 0 (Hamilton–Jacobi)",
        applies: |c| c.order == 1 && c.affine.is_none(),
        solve: nonlinear::solve_separable,
    },
    Tier {
        name: "charpit",
        milestone: "M1",
        description: "Lagrange–Charpit: first integral of Charpit's system, then du = p dx + q dy",
        applies: |c| c.order == 1 && c.affine.is_none() && c.indep.len() == 2,
        solve: nonlinear::solve_charpit,
    },
];

/// The valid `hint` names, in preference order.
pub fn pdsolve_hints() -> Vec<&'static str> {
    TIERS.iter().map(|t| t.name).collect()
}

/// `Σ Aᵢ(x, u) pᵢ = C(x, u)`: the coefficients of an equation affine in the
/// first derivatives.
#[derive(Clone, Debug)]
pub(crate) struct Affine {
    pub a: Vec<ExprId>,
    pub c: ExprId,
}

/// Per-problem analysis shared by the tiers.
pub struct Ctx<'a> {
    /// The problem being solved.
    pub problem: &'a PdeProblem,
    /// The unknown `u(x, y)`.
    pub unknown: ExprId,
    /// Its independent variables.
    pub indep: Vec<ExprId>,
    /// The (single) equation's order.
    pub order: u32,
    pub(crate) space: JetSpace,
    /// The jet symbol of `u` itself.
    pub(crate) w: ExprId,
    /// The jet symbols `u_{xᵢ}`.
    pub(crate) p: Vec<ExprId>,
    /// The equation in jets.
    pub(crate) f: ExprId,
    pub(crate) affine: Option<Affine>,
    pub(crate) taken: HashSet<String>,
    pub(crate) arb: ArbitraryFunctions,
    pub(crate) failures: Vec<String>,
}

impl<'a> Ctx<'a> {
    fn new(problem: &'a PdeProblem, opts: &PdsolveOptions, pool: &ExprPool) -> Result<Self, PdeError> {
        if problem.equations().len() != 1 || problem.unknowns().len() != 1 {
            return Err(PdeError::PdsolveUnsupported(format!(
                "pdsolve solves one equation in one unknown; got {} equations in {} unknowns",
                problem.equations().len(),
                problem.unknowns().len()
            )));
        }
        let unknown = problem.unknowns()[0];
        let indep = decode_unknown(unknown, pool)?.args;
        if indep.len() < 2 {
            return Err(PdeError::PdsolveUnsupported(format!(
                "{} has one independent variable: use dsolve",
                pool.display(unknown)
            )));
        }
        let order = problem.order();
        let space = JetSpace::new(&indep, &[unknown], order.max(1), pool)?;
        let f = simp(space.to_jets(problem.equations()[0], pool)?, pool);
        let w = space.jet(0, &vec![0; indep.len()], pool)?;
        let mut p = Vec::new();
        for i in 0..indep.len() {
            let mut alpha = vec![0u32; indep.len()];
            alpha[i] = 1;
            p.push(space.jet(0, &alpha, pool)?);
        }
        let mut all: Vec<ExprId> = problem.equations().to_vec();
        for c in problem.conditions() {
            all.push(c.lhs);
            all.push(c.rhs);
        }
        let mut taken = names_in(&all, pool);
        taken.insert(symbol_name(w, pool).unwrap_or_default());
        for &pi in &p {
            taken.insert(symbol_name(pi, pool).unwrap_or_default());
        }
        let arb = ArbitraryFunctions::new(&opts.function_prefix).avoiding(&all, pool);
        let mut ctx = Ctx {
            problem,
            unknown,
            indep,
            order,
            space,
            w,
            p,
            f,
            affine: None,
            taken,
            arb,
            failures: Vec::new(),
        };
        if order == 1 {
            ctx.affine = ctx.affine_form(pool);
        }
        Ok(ctx)
    }

    fn affine_form(&self, pool: &ExprPool) -> Option<Affine> {
        let mut a = Vec::new();
        for &pi in &self.p {
            let ai = d(self.f, pi, pool)?;
            if !free_of(ai, &self.p, pool) {
                return None;
            }
            a.push(ai);
        }
        let zero = pool.integer(0);
        let m = self.p.iter().map(|&pi| (pi, zero)).collect();
        let b = simp(subs_map(self.f, &m, pool), pool);
        if a.iter().all(|&ai| is_zero(ai, pool)) {
            return None;
        }
        Some(Affine {
            a,
            c: simp(neg(b, pool), pool),
        })
    }

    /// `expr` with the jet symbol `w` replaced by the applied unknown.
    pub(crate) fn with_unknown(&self, expr: ExprId, pool: &ExprPool) -> ExprId {
        subs1(expr, self.w, self.unknown, pool)
    }

    pub(crate) fn fresh_constant(&mut self, pool: &ExprPool) -> ExprId {
        fresh_symbol("C", &mut self.taken, pool)
    }

    pub(crate) fn fresh_internal(&mut self, pool: &ExprPool) -> ExprId {
        fresh_symbol("__pde_s", &mut self.taken, pool)
    }

    pub(crate) fn fail(&mut self, why: String) {
        self.failures.push(why);
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Solve `problem` (one equation, one unknown), returning the first verified
/// answer in tier order.  See the [module documentation](self).
///
/// # Errors
///
/// `E-PDE-050` for a problem outside every tier (or an unknown hint),
/// `E-PDE-051` when no candidate verified, `E-PDE-052` for Cauchy data on a
/// characteristic, `E-PDE-053` for side conditions that cannot be imposed.
pub fn pdsolve(
    problem: &PdeProblem,
    opts: &PdsolveOptions,
    pool: &ExprPool,
) -> Result<PdeSolution, PdeError> {
    let mut all = run(problem, opts, false, pool)?;
    Ok(all.remove(0))
}

/// Every verified answer of every applicable tier (one per tier at most),
/// in tier order.  Errors as [`pdsolve`].
pub fn pdsolve_all(
    problem: &PdeProblem,
    opts: &PdsolveOptions,
    pool: &ExprPool,
) -> Result<Vec<PdeSolution>, PdeError> {
    run(problem, opts, true, pool)
}

fn run(
    problem: &PdeProblem,
    opts: &PdsolveOptions,
    all: bool,
    pool: &ExprPool,
) -> Result<Vec<PdeSolution>, PdeError> {
    if let Some(h) = &opts.hint {
        if !TIERS.iter().any(|t| t.name == h) {
            return Err(PdeError::PdsolveUnsupported(format!(
                "unknown hint {h:?}; valid hints: {}",
                pdsolve_hints().join(", ")
            )));
        }
    }
    let mut ctx = Ctx::new(problem, opts, pool)?;
    if ctx.order != 1 {
        return Err(PdeError::PdsolveUnsupported(format!(
            "order-{} equations are not solved yet (milestone M1 covers first order)",
            ctx.order
        )));
    }
    let has_data = !problem.conditions().is_empty();
    let mut out: Vec<PdeSolution> = Vec::new();
    let mut applicable = 0;
    let mut cauchy_err: Option<PdeError> = None;
    for tier in TIERS {
        if let Some(h) = &opts.hint {
            if tier.name != h {
                continue;
            }
        }
        if !(tier.applies)(&ctx) {
            continue;
        }
        applicable += 1;
        let cands = (tier.solve)(&mut ctx, pool);
        if cands.is_empty() {
            ctx.fail(format!("{}: no candidate", tier.name));
        }
        let mut got = false;
        let mut expanded: Vec<Candidate> = Vec::new();
        for cand in cands {
            if has_data {
                match cauchy::impose(&mut ctx, cand, pool) {
                    Ok(cs) => expanded.extend(cs),
                    Err(e) => {
                        ctx.fail(format!("{}: {e}", tier.name));
                        if matches!(e, PdeError::CharacteristicData(_)) || cauchy_err.is_none() {
                            cauchy_err = Some(e);
                        }
                    }
                }
            } else {
                expanded.push(cand);
            }
        }
        for cand in expanded {
            match finalize(&ctx, tier.name, cand, pool) {
                Ok(sol) => {
                    out.push(sol);
                    got = true;
                    break;
                }
                Err(why) => ctx.fail(format!("{}: {why}", tier.name)),
            }
        }
        if got && !all {
            break;
        }
    }
    if out.is_empty() {
        if applicable == 0 {
            return Err(PdeError::PdsolveUnsupported(match &opts.hint {
                Some(h) => format!("the hint {h:?} does not apply to this equation"),
                None => "no first-order method applies to this equation".into(),
            }));
        }
        if let Some(e) = cauchy_err {
            return Err(e);
        }
        return Err(PdeError::NoVerifiedSolution(ctx.failures.join("; ")));
    }
    Ok(out)
}

/// Verify a candidate; `Some` only when exactly verified.
fn finalize(
    ctx: &Ctx,
    method: &'static str,
    cand: Candidate,
    pool: &ExprPool,
) -> Result<PdeSolution, String> {
    let value = tidy(simp(cand.value, pool), pool);
    let shown = || {
        let s = pool.display(value).to_string();
        let s: String = s.chars().take(160).collect();
        if cand.implicit {
            format!("{s} = 0")
        } else {
            format!("u = {s}")
        }
    };
    let (verification, mut side) = if cand.implicit {
        verify_implicit(ctx, value, cand.kind, pool)
            .ok_or_else(|| format!("{} is not a relation in u", shown()))?
    } else {
        let v = verify_pde_solution(ctx.problem, &[value], pool).map_err(|e| e.to_string())?;
        (v, Vec::new())
    };
    if verification.status != VerificationStatus::ExactlyVerified {
        let bad: Vec<String> = verification
            .checks
            .iter()
            .filter(|c| c.status != ZeroStatus::ProvedZero)
            .map(|c| format!("{} residual {}", c.kind, c.status.as_str()))
            .collect();
        return Err(format!(
            "candidate {} discarded ({}: {})",
            shown(),
            verification.status.as_str(),
            bad.join(", ")
        ));
    }
    let mut side_conditions = cand.side_conditions.clone();
    side_conditions.append(&mut side);
    let arbitrary_functions = verification.arbitrary_functions.clone();
    let mut sol = PdeSolution {
        unknown: ctx.unknown,
        value,
        implicit: cand.implicit,
        kind: cand.kind,
        method,
        arbitrary_functions,
        constants: cand.constants.clone(),
        side_conditions,
        verification,
        steps: cand.steps.clone(),
        alternatives: Vec::new(),
    };
    if cand.kind == PdeSolutionKind::CompleteIntegral && !cand.implicit {
        if let Some(sing) = nonlinear::singular_integral(ctx, value, &cand.constants, pool) {
            if let Ok(s) = finalize(ctx, method, sing, pool) {
                sol.steps
                    .push(format!("singular integral (envelope): u = {}", pool.display(s.value)));
                sol.alternatives.push(s);
            }
        }
    }
    Ok(sol)
}

/// Cosmetic clean-up of an answer before verification (which sees the
/// cleaned form, so nothing here needs to be trusted): `exp(c·log a) → a^c`,
/// `log(exp a) → a`.
fn tidy(e: ExprId, pool: &ExprPool) -> ExprId {
    use crate::kernel::ExprData;
    const ELEMENTARY: &[&str] = &[
        "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan",
    ];
    let rebuilt = match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| tidy(x, pool)).collect()),
        ExprData::Mul(xs) => {
            let xs: Vec<ExprId> = xs.iter().map(|&x| tidy(x, pool)).collect();
            merge_exps(xs, pool)
        }
        ExprData::Pow { base, exp } => pool.pow(tidy(base, pool), tidy(exp, pool)),
        ExprData::Func { name, args } => {
            let mut args: Vec<ExprId> = args.iter().map(|&a| tidy(a, pool)).collect();
            if ELEMENTARY.contains(&name.as_str())
                && args.len() == 1
                && pool.depth(args[0]) < 12
            {
                args[0] = crate::simplify::engine::simplify_expanded(args[0], pool).value;
            }
            if name == "exp" && args.len() == 1 {
                if let Some(r) = exp_of_log(args[0], pool) {
                    return r;
                }
            }
            if name == "log" && args.len() == 1 {
                if let ExprData::Func { name: n2, args: a2 } = pool.get(args[0]) {
                    if n2 == "exp" && a2.len() == 1 {
                        return a2[0];
                    }
                }
            }
            pool.func(name, args)
        }
        _ => return e,
    };
    if rebuilt == e {
        e
    } else {
        simp(rebuilt, pool)
    }
}

/// `exp(a)·exp(b)^k → exp(a + k b)` inside one product.
fn merge_exps(xs: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    use crate::kernel::ExprData;
    let mut args = Vec::new();
    let mut rest = Vec::new();
    for x in xs {
        let hit = match pool.get(x) {
            ExprData::Func { name, args: a } if name == "exp" && a.len() == 1 => Some(a[0]),
            ExprData::Pow { base, exp } if is_number(exp, pool) => match pool.get(base) {
                ExprData::Func { name, args: a } if name == "exp" && a.len() == 1 => {
                    Some(pool.mul(vec![exp, a[0]]))
                }
                _ => None,
            },
            _ => None,
        };
        match hit {
            Some(a) => args.push(a),
            None => rest.push(x),
        }
    }
    if args.len() >= 2 {
        let sum = pool.add(args);
        let sum = if pool.depth(sum) < 12 {
            crate::simplify::engine::simplify_expanded(sum, pool).value
        } else {
            simp(sum, pool)
        };
        let ex = exp_of_log(sum, pool).unwrap_or_else(|| pool.func("exp", vec![sum]));
        rest.push(ex);
    } else {
        rest.extend(args.into_iter().map(|a| pool.func("exp", vec![a])));
    }
    pool.mul(rest)
}

/// `exp(Σ cᵢ log aᵢ + r)` → `Π aᵢ^cᵢ · exp(r)` for numeric `cᵢ`.
fn exp_of_log(arg: ExprId, pool: &ExprPool) -> Option<ExprId> {
    use crate::kernel::ExprData;
    let terms = match pool.get(arg) {
        ExprData::Add(xs) => xs,
        _ => vec![arg],
    };
    let mut factors = Vec::new();
    let mut rest = Vec::new();
    for t in terms {
        let (coef, core) = match pool.get(t) {
            ExprData::Mul(xs) if xs.len() == 2 && is_number(xs[0], pool) => (xs[0], xs[1]),
            ExprData::Mul(xs) if xs.len() == 2 && is_number(xs[1], pool) => (xs[1], xs[0]),
            _ => (pool.integer(1), t),
        };
        match pool.get(core) {
            ExprData::Func { name, args } if name == "log" && args.len() == 1 => {
                factors.push(pool.pow(args[0], coef));
            }
            _ => rest.push(t),
        }
    }
    if factors.is_empty() {
        return None;
    }
    if !rest.is_empty() {
        factors.push(pool.func("exp", vec![pool.add(rest)]));
    }
    Some(simp(pool.mul(factors), pool))
}

/// Verify a relation `Φ(x, u(x)) = 0` by implicit differentiation, and the
/// problem's conditions on it.
fn verify_implicit(
    ctx: &Ctx,
    relation: ExprId,
    kind: PdeSolutionKind,
    pool: &ExprPool,
) -> Option<(PdeVerification, Vec<SideCondition>)> {
    let g = simp(ctx.space.to_jets(relation, pool).ok()?, pool);
    if !contains(g, ctx.w, pool) {
        return None;
    }
    let gw = d(g, ctx.w, pool)?;
    if is_zero(gw, pool) {
        return None;
    }
    let mut m = std::collections::HashMap::new();
    for (i, &x) in ctx.indep.iter().enumerate() {
        let gx = d(g, x, pool)?;
        m.insert(ctx.p[i], neg(div(gx, gw, pool), pool));
    }
    let r = simp(subs_map(ctx.f, &m, pool), pool);
    let zt = zero_test(r, pool, None);
    let mut checks = vec![ResidualCheck {
        kind: "pde",
        index: 0,
        residual: r,
        status: zt.status,
        method: if zt.status == ZeroStatus::ProvedZero {
            "implicit_differentiation"
        } else {
            zt.method
        },
        evidence: zt.evidence,
    }];
    let mut side = Vec::new();
    if !is_number(gw, pool) {
        side.push(SideCondition::NonZero(ctx.with_unknown(gw, pool)));
    }
    let (mut nb, mut ni) = (0, 0);
    for c in ctx.problem.conditions() {
        let (k, idx) = match c.kind {
            super::problem::ConditionKind::Initial => {
                ni += 1;
                ("ic", ni - 1)
            }
            _ => {
                nb += 1;
                ("bc", nb - 1)
            }
        };
        let (status, residual) = match implicit_condition_residual(ctx, relation, c, pool) {
            Some(r) => (zero_test(r, pool, None).status, r),
            None => (ZeroStatus::Unknown, c.residual(pool)),
        };
        checks.push(ResidualCheck {
            kind: k,
            index: idx,
            residual,
            status,
            method: "relation_on_data",
            evidence: Vec::new(),
        });
    }
    let status = if checks.iter().all(|c| c.status == ZeroStatus::ProvedZero) {
        VerificationStatus::ExactlyVerified
    } else if checks.iter().any(|c| c.status == ZeroStatus::ProvedNonzero) {
        VerificationStatus::Refuted
    } else {
        VerificationStatus::Unverified
    };
    let _ = kind;
    let mut arbitrary: Vec<String> = Vec::new();
    collect_arbitrary(relation, ctx, pool, &mut arbitrary);
    Some((
        PdeVerification {
            status,
            checks,
            arbitrary_functions: arbitrary,
        },
        side,
    ))
}

fn collect_arbitrary(e: ExprId, ctx: &Ctx, pool: &ExprPool, out: &mut Vec<String>) {
    let uname = decode_unknown(ctx.unknown, pool).map(|v| v.name).unwrap_or_default();
    let mut seen = HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = crate::diff::undefined::view(n, pool) {
            if v.name != uname && !out.contains(&v.name) {
                out.push(v.name.clone());
            }
            stack.extend(v.args);
            continue;
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    out.sort();
}

/// The data `u(a(s)) = φ(s)` on the relation: `Φ(a(s), φ(s))`, which must
/// vanish.  `None` for a condition on a derivative.
fn implicit_condition_residual(
    ctx: &Ctx,
    relation: ExprId,
    c: &PdeCondition,
    pool: &ExprPool,
) -> Option<ExprId> {
    let v = crate::diff::undefined::view(c.lhs, pool)?;
    if v.inner != c.lhs || v.args.len() != ctx.indep.len() {
        return None;
    }
    let g = simp(ctx.space.to_jets(relation, pool).ok()?, pool);
    let mut m = std::collections::HashMap::new();
    let mut tmp = Vec::new();
    let mut taken = ctx.taken.clone();
    for &x in &ctx.indep {
        let t = fresh_symbol("__pde_tmp", &mut taken, pool);
        tmp.push(t);
        m.insert(x, t);
    }
    let g1 = subs_map(g, &m, pool);
    let mut m2: std::collections::HashMap<ExprId, ExprId> =
        tmp.iter().copied().zip(v.args.iter().copied()).collect();
    m2.insert(ctx.w, c.rhs);
    Some(simp(subs_map(g1, &m2, pool), pool))
}

#[cfg(test)]
mod tests;
