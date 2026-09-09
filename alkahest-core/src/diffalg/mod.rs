//! V2-13 — Differential algebra / Rosenfeld–Gröbner-style differential elimination.
//!
//! Given a polynomial DAE in implicit form `g_i(t, y, y') = 0`, prolongation
//! appends formal time derivatives `D^k(g_i)` while tracking the derivative
//! state layout (same structural rule as [`crate::dae::pantelides`]).  After
//! each prolongation step, an ordinary Gröbner basis over ℚ captures the
//! algebraic constraints among the jet variables.  Inconsistent systems
//! yield the unit ideal (basis containing a non-zero constant).
//!
//! This is a **fragment** of the full Rosenfeld–Gröbner / regular differential
//! decomposition (no multi-branch saturation over initials here); it suffices
//! for consistency checking and complements Pantelides when the structural
//! index is high.
//!
//! References:
//! - Boulier et al., *Rosenfeld–Gröbner algorithm* (differential elimination).
//! - Hubert, *Differential algebra for comptroller generation*.

use crate::dae::{
    differentiate_equation, extend_dae_for_derivative_symbols, pantelides, DaeError,
    PantelidesResult, DAE,
};
use crate::errors::AlkahestError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::groebner::{
    GbPoly, GroebnerBasis, MonomialOrder, ParamGbPoly, ParamGroebnerBasis, ParamGroebnerError,
};
use crate::solver::SolverError;
use crate::solver::{expr_to_gbpoly, expr_to_param_gbpoly};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

/// Default prolongation depth (formal time derivatives chained per equation).
const DEFAULT_MAX_PROLONG_ROUNDS: usize = 8;

/// Ordering data for differential indeterminates: `vars[i]` maps to exponent
/// index `i` of [`GbPoly`] used in Gröbner steps.
#[derive(Clone, Debug)]
pub struct DifferentialRanking {
    pub vars: Vec<ExprId>,
}

/// Algebraic presentation of a finitely generated differential ideal (dense
/// ℚ-polynomial generators in a fixed jet basis).
#[derive(Clone, Debug)]
pub struct DifferentialIdeal {
    pub generators: Vec<GbPoly>,
}

/// Ordinary polynomial ring context for a fixed independent variable and ranked jets.
#[derive(Clone, Debug)]
pub struct DifferentialRing {
    pub time: ExprId,
    pub ranked_indeterminates: Vec<ExprId>,
}

/// One component produced by differential elimination — here the algebraic
/// Gröbner basis of a saturated ordinary ideal at the end of prolongation.
#[derive(Clone, Debug)]
pub struct RegularDifferentialChain {
    pub basis: GroebnerBasis,
}

/// Result of [`rosenfeld_groebner`] (single coherent component in this build).
#[derive(Clone, Debug)]
pub struct RosenfeldGroebnerResult {
    /// `false` iff the unit ideal was encountered (no common jet solution over ℚ).
    pub consistent: bool,
    /// Non-empty when [`Self::consistent`] is true (one entry in this implementation).
    pub chains: Vec<RegularDifferentialChain>,
    /// [`DAE`] state after prolongation (extra derivative jets may be present).
    pub working_dae: DAE,
    /// Final Gröbner basis when consistent.
    pub final_basis: Option<GroebnerBasis>,
    /// Number of prolongation rounds that added new relations.
    pub prolongation_rounds: usize,
    /// `true` if we stopped only because the prolongation budget was reached
    /// (the differential chain need not be saturated).
    pub truncated: bool,
}

/// Outcome of [`dae_index_reduce`]: Pantelides when it succeeds, otherwise a
/// Rosenfeld–Gröbner consistency pass when Pantelides hits its index cap.
#[derive(Clone, Debug)]
pub enum DaeIndexReduction {
    Pantelides(PantelidesResult),
    Rosenfeld(RosenfeldGroebnerResult),
}

/// Errors from differential-algebra operations.
#[derive(Debug, Clone)]
pub enum DiffAlgError {
    DiffError(String),
    NotPolynomial(String),
    EmptySystem,
}

impl fmt::Display for DiffAlgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffAlgError::DiffError(s) => write!(f, "differentiation error: {s}"),
            DiffAlgError::NotPolynomial(s) => write!(f, "not a polynomial: {s}"),
            DiffAlgError::EmptySystem => write!(f, "empty equation system"),
        }
    }
}

impl std::error::Error for DiffAlgError {}

impl AlkahestError for DiffAlgError {
    fn code(&self) -> &'static str {
        match self {
            DiffAlgError::DiffError(_) => "E-DIFFALG-001",
            DiffAlgError::NotPolynomial(_) => "E-DIFFALG-002",
            DiffAlgError::EmptySystem => "E-DIFFALG-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DiffAlgError::DiffError(_) => {
                Some("ensure the DAE is polynomial in its state and derivative symbols")
            }
            DiffAlgError::NotPolynomial(_) => {
                Some("declare all jet variables and parameters; remove transcendental functions")
            }
            DiffAlgError::EmptySystem => Some("pass at least one implicit equation"),
        }
    }
}

fn solver_err_to_diffalg(e: SolverError) -> DiffAlgError {
    DiffAlgError::NotPolynomial(e.to_string())
}

fn is_unit_ideal_gb(gb: &GroebnerBasis) -> bool {
    gb.generators().iter().any(|g| {
        g.terms.len() == 1
            && g.terms
                .keys()
                .next()
                .is_some_and(|e| e.iter().all(|&x| x == 0))
            && g.terms.values().next().is_some_and(|c| *c != 0)
    })
}

fn pad_gbpoly(p: &GbPoly, new_n: usize) -> GbPoly {
    if new_n == p.n_vars {
        return p.clone();
    }
    assert!(new_n > p.n_vars);
    let pad = new_n - p.n_vars;
    let mut terms = BTreeMap::new();
    for (e, c) in &p.terms {
        let mut ne = e.clone();
        ne.extend(std::iter::repeat(0u32).take(pad));
        terms.insert(ne, c.clone());
    }
    GbPoly {
        terms,
        n_vars: new_n,
    }
}

fn children(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::BigO(inner) => vec![*inner],
        _ => vec![],
    })
}

fn collect_symbols(
    expr: ExprId,
    pool: &ExprPool,
    seen: &mut HashSet<ExprId>,
    out: &mut Vec<ExprId>,
) {
    let is_sym = pool.with(expr, |d| matches!(d, ExprData::Symbol { .. }));
    if is_sym && seen.insert(expr) {
        out.push(expr);
    }
    for c in children(expr, pool) {
        collect_symbols(c, pool, seen, out);
    }
}

fn vars_for_dae(dae: &DAE, scratch: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut push = |id: ExprId| {
        if seen.insert(id) {
            out.push(id);
        }
    };
    push(dae.time_var);
    for i in 0..dae.variables.len() {
        push(dae.variables[i]);
        push(dae.derivatives[i]);
    }
    for &root in scratch {
        collect_symbols(root, pool, &mut seen, &mut out);
    }
    out
}

/// Append every symbol of `fresh` that `vars` does not already name.
///
/// Prolongation pads existing polynomials by extending their exponent vectors,
/// which is only meaningful if slot `i` keeps naming the same symbol from round
/// to round.  Recomputing the ranking from scratch does **not** guarantee that:
/// a jet reached only through the previous round's equations drops out of the
/// next round's scratch set, every later slot shifts down by one, and the
/// padded polynomials silently start asserting relations about other variables.
/// Merging instead of replacing keeps the ranking append-only, which is what
/// [`pad_gbpoly`] assumes.
fn merge_vars(vars: &mut Vec<ExprId>, fresh: Vec<ExprId>) {
    for v in fresh {
        if !vars.contains(&v) {
            vars.push(v);
        }
    }
}

fn polys_from_equations(
    eqs: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<GbPoly>, DiffAlgError> {
    eqs.iter()
        .map(|&eq| expr_to_gbpoly(eq, vars, pool).map_err(solver_err_to_diffalg))
        .collect()
}

/// Algebraic-only slice: Gröbner basis of the generators; empty `Vec` if the
/// ideal is `[1]`.
pub fn rosenfeld_groebner_algebraic(
    gens: Vec<GbPoly>,
    order: MonomialOrder,
) -> Result<Vec<RegularDifferentialChain>, DiffAlgError> {
    if gens.is_empty() {
        return Err(DiffAlgError::EmptySystem);
    }
    let gb = GroebnerBasis::compute(gens, order);
    if is_unit_ideal_gb(&gb) {
        return Ok(vec![]);
    }
    Ok(vec![RegularDifferentialChain { basis: gb }])
}

/// Rosenfeld-style prolongation + Gröbner stabilization for a polynomial DAE.
///
/// `max_prolong_rounds` bounds how many prolongation steps are attempted; if the
/// chain has not saturated, [`RosenfeldGroebnerResult::truncated`] is set — typical
/// nonlinear ODE jets do not stabilize in finitely many algebraic steps.
pub fn rosenfeld_groebner_with_options(
    dae: &DAE,
    pool: &ExprPool,
    order: MonomialOrder,
    max_prolong_rounds: usize,
) -> Result<RosenfeldGroebnerResult, DiffAlgError> {
    rosenfeld_groebner_ranked(dae, pool, order, max_prolong_rounds).map(|(result, _)| result)
}

/// [`rosenfeld_groebner_with_options`] plus the jet [`DifferentialRanking`] that
/// indexes the exponent vectors of [`RosenfeldGroebnerResult::final_basis`].
///
/// The elimination result is unreadable without this: a [`GbPoly`] stores only
/// exponent vectors, so `ranking.vars[i]` is what exponent slot `i` refers to.
/// Pair it with [`crate::solver::gbpoly_to_expr`] to recover the input–output
/// equations as [`ExprId`]s.
pub fn rosenfeld_groebner_ranked(
    dae: &DAE,
    pool: &ExprPool,
    order: MonomialOrder,
    max_prolong_rounds: usize,
) -> Result<(RosenfeldGroebnerResult, DifferentialRanking), DiffAlgError> {
    if dae.equations.is_empty() {
        return Err(DiffAlgError::EmptySystem);
    }

    let source_eqs = dae.equations.clone();
    let mut work = dae.clone();
    let mut scratch: Vec<ExprId> = source_eqs.clone();
    let mut vars = vars_for_dae(&work, &scratch, pool);
    let mut active = polys_from_equations(&work.equations, &vars, pool)?;

    let mut prolong_exprs = source_eqs.clone();
    let mut prolongation_rounds: usize = 0;

    for round in 0..max_prolong_rounds {
        let gb = GroebnerBasis::compute(active.clone(), order);
        if is_unit_ideal_gb(&gb) {
            return Ok((
                RosenfeldGroebnerResult {
                    consistent: false,
                    chains: vec![],
                    working_dae: work,
                    final_basis: None,
                    prolongation_rounds,
                    truncated: false,
                },
                DifferentialRanking { vars },
            ));
        }

        let mut next_prolong = Vec::with_capacity(prolong_exprs.len());
        for &eq in &prolong_exprs {
            let d_eq =
                differentiate_equation(eq, &work.variables, &work.derivatives, work.time_var, pool)
                    .map_err(|e| DiffAlgError::DiffError(e.to_string()))?;
            extend_dae_for_derivative_symbols(&mut work, d_eq, pool);
            next_prolong.push(d_eq);
        }
        prolong_exprs = next_prolong;
        scratch = source_eqs
            .iter()
            .copied()
            .chain(prolong_exprs.iter().copied())
            .collect();
        merge_vars(&mut vars, vars_for_dae(&work, &scratch, pool));
        let n = vars.len();
        for p in &mut active {
            *p = pad_gbpoly(p, n);
        }

        let gb_check = GroebnerBasis::compute(active.clone(), order);
        let mut to_add: Vec<GbPoly> = Vec::new();
        for &d_eq in &prolong_exprs {
            let p = expr_to_gbpoly(d_eq, &vars, pool).map_err(solver_err_to_diffalg)?;
            if !gb_check.contains(&p) {
                to_add.push(p);
            }
        }

        if to_add.is_empty() {
            let final_basis = GroebnerBasis::compute(active, order);
            let consistent = !is_unit_ideal_gb(&final_basis);
            let chains = if consistent {
                vec![RegularDifferentialChain {
                    basis: final_basis.clone(),
                }]
            } else {
                vec![]
            };
            return Ok((
                RosenfeldGroebnerResult {
                    consistent,
                    chains,
                    working_dae: work,
                    final_basis: if consistent { Some(final_basis) } else { None },
                    prolongation_rounds,
                    truncated: false,
                },
                DifferentialRanking { vars },
            ));
        }

        active.extend(to_add);
        prolongation_rounds += 1;

        if round + 1 == max_prolong_rounds {
            let final_basis = GroebnerBasis::compute(active, order);
            let consistent = !is_unit_ideal_gb(&final_basis);
            let chains = if consistent {
                vec![RegularDifferentialChain {
                    basis: final_basis.clone(),
                }]
            } else {
                vec![]
            };
            return Ok((
                RosenfeldGroebnerResult {
                    consistent,
                    chains,
                    working_dae: work,
                    final_basis: if consistent { Some(final_basis) } else { None },
                    prolongation_rounds,
                    truncated: true,
                },
                DifferentialRanking { vars },
            ));
        }
    }

    let final_basis = GroebnerBasis::compute(active, order);
    let consistent = !is_unit_ideal_gb(&final_basis);
    Ok((
        RosenfeldGroebnerResult {
            consistent,
            chains: if consistent {
                vec![RegularDifferentialChain {
                    basis: final_basis.clone(),
                }]
            } else {
                vec![]
            },
            working_dae: work,
            final_basis: if consistent { Some(final_basis) } else { None },
            prolongation_rounds,
            truncated: true,
        },
        DifferentialRanking { vars },
    ))
}

// ---------------------------------------------------------------------------
// M9 × V2-13 — differential elimination with the parameters in Q(params)
// ---------------------------------------------------------------------------

/// Knobs for [`rosenfeld_groebner_parametric`].
#[derive(Clone, Copy, Debug)]
pub struct ParametricProlongOpts<'a> {
    /// Monomial order for each round's basis.  `Lex` with the variables to
    /// eliminate ordered first is what elimination needs.
    pub order: MonomialOrder,
    /// Prolongation budget — the number of formal time derivatives taken.
    pub max_prolong_rounds: usize,
    /// The variables the caller intends to eliminate, e.g. the unobserved
    /// states of an ODE model.  Their whole jet chain is eliminated with them:
    /// naming `x` also names `dx/dt`, `d2x/dt2`, … as they appear.
    ///
    /// Empty disables both the informativeness check and [`Self::minimal`].
    pub eliminate: &'a [ExprId],
    /// Stop at the **first** prolongation round whose elimination ideal is
    /// non-trivial, instead of prolonging to the budget.
    ///
    /// See [`ParametricRosenfeldResult::minimal_prolongation_rounds`] for the
    /// scope of "first informative" — it is not a guarantee of minimality.
    pub minimal: bool,
}

impl Default for ParametricProlongOpts<'_> {
    fn default() -> Self {
        ParametricProlongOpts {
            order: MonomialOrder::Lex,
            max_prolong_rounds: DEFAULT_MAX_PROLONG_ROUNDS,
            eliminate: &[],
            minimal: false,
        }
    }
}

/// Result of [`rosenfeld_groebner_parametric`].
#[derive(Clone, Debug)]
pub struct ParametricRosenfeldResult {
    /// `false` iff the unit ideal was reached over `Q(params)`.
    pub consistent: bool,
    /// `true` if prolongation stopped on the budget, or on `minimal`, rather
    /// than because differentiating stopped adding relations.  A truncated
    /// basis is a sound set of consequences but need not be complete.
    pub truncated: bool,
    /// Number of prolongation rounds that contributed new relations.
    pub prolongation_rounds: usize,
    /// The prolonged [`DAE`]: the input plus every jet introduced.
    pub working_dae: DAE,
    /// The saturated basis over `Q(params)`, or `None` when inconsistent.
    pub final_basis: Option<ParamGroebnerBasis>,
    /// The lowest round count at which the elimination ideal with respect to
    /// [`ParametricProlongOpts::eliminate`] was non-empty, or `None` when no
    /// round was informative or no `eliminate` list was given.
    ///
    /// **Scope.** This is "the first round at which eliminating those variables
    /// leaves a generator", not a theorem about the differential ideal.  For a
    /// single-output model it coincides with the jet order the input–output
    /// relation needs; **for multi-output models the criterion is known to be
    /// wrong**, because one output can become informative several rounds before
    /// the others and the truncated basis then misses their relations.  Treat
    /// it as a cost signal, not a certificate.
    ///
    /// It matters because the cost is not gentle: on the SIR model one extra
    /// prolongation past the informative round takes the elimination from a
    /// single 4-term generator to thirteen generators of up to 233 terms with
    /// 30-digit rational-function coefficients — four orders of magnitude of
    /// time, for the same relation.
    pub minimal_prolongation_rounds: Option<usize>,
}

fn param_err_to_diffalg(e: ParamGroebnerError) -> DiffAlgError {
    DiffAlgError::NotPolynomial(e.to_string())
}

fn is_unit_ideal_param(gb: &ParamGroebnerBasis) -> bool {
    gb.generators().iter().any(|g| {
        g.terms.len() == 1
            && g.terms
                .keys()
                .next()
                .is_some_and(|e| e.iter().all(|&x| x == 0))
    })
}

/// The exponent slots of `vars` naming `roots` or any jet descended from one.
///
/// `dae.variables[i]` differentiates to `dae.derivatives[i]`, so following that
/// map to a fixed point turns "eliminate `x`" into "eliminate `x`, `dx/dt`,
/// `d2x/dt2`, …" — which is what a caller eliminating a state means.
fn jet_closure_slots(dae: &DAE, vars: &[ExprId], roots: &[ExprId]) -> Vec<usize> {
    let mut closed: HashSet<ExprId> = roots.iter().copied().collect();
    loop {
        let mut grew = false;
        for (i, v) in dae.variables.iter().enumerate() {
            if closed.contains(v) {
                if let Some(&d) = dae.derivatives.get(i) {
                    grew |= closed.insert(d);
                }
            }
        }
        if !grew {
            break;
        }
    }
    vars.iter()
        .enumerate()
        .filter(|(_, v)| closed.contains(v))
        .map(|(i, _)| i)
        .collect()
}

/// True when at least one generator is free of every slot in `slots` — i.e. the
/// elimination ideal is non-trivial.
fn param_elimination_is_informative(gb: &ParamGroebnerBasis, slots: &[usize]) -> bool {
    gb.generators().iter().any(|g| {
        !g.terms
            .keys()
            .any(|e| slots.iter().any(|&i| e.get(i).copied().unwrap_or(0) > 0))
    })
}

/// The jet chain `v, dv, d²v, …` of a state, `depth` derivatives deep, using
/// the same `d{name}/dt` naming convention prolongation itself uses.
fn jet_chain(v: ExprId, dv: ExprId, depth: usize, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = vec![v, dv];
    let mut cur = dv;
    for _ in 0..depth {
        let name = pool.with(cur, |d| match d {
            ExprData::Symbol { name, .. } => format!("d{name}/dt"),
            _ => "d?/dt".to_string(),
        });
        cur = pool.symbol(&name, crate::kernel::Domain::Real);
        out.push(cur);
    }
    out
}

/// A ranking with the whole jet tower laid out up front, eliminated states
/// first.
///
/// Two things need this.  Elimination by generator filtering is only valid
/// under a lex order that ranks the eliminated variables *above* the rest, and
/// [`vars_for_dae`] interleaves states with outputs instead.  And the ranking
/// has to be append-only across rounds for [`pad_param_gbpoly`] to mean
/// anything, which it cannot be if new jets of an eliminated state keep having
/// to be inserted in front of the outputs.  Laying the tower out to the
/// prolongation depth settles both; the jets that never get used are unused
/// variables in the ring, which cost nothing but a slot.
fn ranked_jet_vars(
    dae: &DAE,
    eliminate: &[ExprId],
    params: &[ExprId],
    depth: usize,
    pool: &ExprPool,
) -> Vec<ExprId> {
    let mut elim_first: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for (i, &v) in dae.variables.iter().enumerate() {
        let Some(&dv) = dae.derivatives.get(i) else {
            continue;
        };
        let chain = jet_chain(v, dv, depth, pool);
        if eliminate.contains(&v) {
            elim_first.extend(chain);
        } else {
            rest.extend(chain);
        }
    }
    let mut out: Vec<ExprId> = vec![dae.time_var];
    for v in elim_first.into_iter().chain(rest) {
        if !params.contains(&v) && !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

fn param_vars_for_dae(
    dae: &DAE,
    scratch: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
) -> Vec<ExprId> {
    vars_for_dae(dae, scratch, pool)
        .into_iter()
        .filter(|v| !params.contains(v))
        .collect()
}

fn pad_param_gbpoly(p: &ParamGbPoly, new_n: usize) -> ParamGbPoly {
    if new_n == p.n_vars {
        return p.clone();
    }
    assert!(new_n > p.n_vars);
    let pad = new_n - p.n_vars;
    ParamGbPoly {
        terms: p
            .terms
            .iter()
            .map(|(e, c)| {
                let mut e = e.clone();
                e.extend(std::iter::repeat(0u32).take(pad));
                (e, c.clone())
            })
            .collect(),
        n_vars: new_n,
        n_params: p.n_params,
    }
}

/// Rosenfeld-style prolongation with `params` in the **coefficient field**
/// `Q(params)` rather than as extra ring variables (M9 × V2-13).
///
/// [`rosenfeld_groebner_ranked`] puts every free symbol in the ring, so a model
/// parameter enlarges the monomial order, the pair schedule and the staircase.
/// Here the parameters are moved into the coefficients, which is the difference
/// between eliminating states from `Q[states, jets, params]` and from
/// `Q(params)[states, jets]` — the computation the input–output relations of an
/// ODE model actually need.
///
/// The returned basis is **generic** in the parameters, exactly as
/// [`ParamGroebnerBasis`] describes: read
/// [`ParamGroebnerBasis::conditions`] for the hypotheses it used.
///
/// `params` must be disjoint from the DAE's variables, derivatives and time
/// variable; a parameter listed there is dropped from the ring, so it must not
/// be one of the unknowns.
pub fn rosenfeld_groebner_parametric(
    dae: &DAE,
    pool: &ExprPool,
    params: &[ExprId],
    opts: ParametricProlongOpts<'_>,
) -> Result<(ParametricRosenfeldResult, DifferentialRanking), DiffAlgError> {
    if dae.equations.is_empty() {
        return Err(DiffAlgError::EmptySystem);
    }

    let source_eqs = dae.equations.clone();
    let mut work = dae.clone();
    let mut scratch: Vec<ExprId> = source_eqs.clone();
    let mut vars = if opts.eliminate.is_empty() {
        param_vars_for_dae(&work, &scratch, params, pool)
    } else {
        let mut v = ranked_jet_vars(
            &work,
            opts.eliminate,
            params,
            opts.max_prolong_rounds + 1,
            pool,
        );
        merge_vars(&mut v, param_vars_for_dae(&work, &scratch, params, pool));
        v
    };
    let mut active: Vec<ParamGbPoly> = work
        .equations
        .iter()
        .map(|&eq| expr_to_param_gbpoly(eq, &vars, params, pool).map_err(solver_err_to_diffalg))
        .collect::<Result<_, _>>()?;

    let mut prolong_exprs = source_eqs.clone();
    let mut minimal_prolongation_rounds: Option<usize> = None;

    // The budget counts prolongations, so `max + 1` bases get computed: one for
    // the unprolonged system and one after each round.
    for round in 0..=opts.max_prolong_rounds {
        let gb = ParamGroebnerBasis::compute(active.clone(), opts.order)
            .map_err(param_err_to_diffalg)?;

        if is_unit_ideal_param(&gb) {
            return Ok((
                ParametricRosenfeldResult {
                    consistent: false,
                    truncated: false,
                    prolongation_rounds: round,
                    working_dae: work,
                    final_basis: None,
                    minimal_prolongation_rounds,
                },
                DifferentialRanking { vars },
            ));
        }

        if !opts.eliminate.is_empty() {
            let slots = jet_closure_slots(&work, &vars, opts.eliminate);
            if minimal_prolongation_rounds.is_none()
                && param_elimination_is_informative(&gb, &slots)
            {
                minimal_prolongation_rounds = Some(round);
                if opts.minimal {
                    return Ok((
                        ParametricRosenfeldResult {
                            consistent: true,
                            truncated: true,
                            prolongation_rounds: round,
                            working_dae: work,
                            final_basis: Some(gb),
                            minimal_prolongation_rounds,
                        },
                        DifferentialRanking { vars },
                    ));
                }
            }
        }

        if round == opts.max_prolong_rounds {
            return Ok((
                ParametricRosenfeldResult {
                    consistent: true,
                    truncated: true,
                    prolongation_rounds: round,
                    working_dae: work,
                    final_basis: Some(gb),
                    minimal_prolongation_rounds,
                },
                DifferentialRanking { vars },
            ));
        }

        let mut next_prolong = Vec::with_capacity(prolong_exprs.len());
        for &eq in &prolong_exprs {
            let d_eq =
                differentiate_equation(eq, &work.variables, &work.derivatives, work.time_var, pool)
                    .map_err(|e| DiffAlgError::DiffError(e.to_string()))?;
            extend_dae_for_derivative_symbols(&mut work, d_eq, pool);
            next_prolong.push(d_eq);
        }
        prolong_exprs = next_prolong;
        scratch = source_eqs
            .iter()
            .copied()
            .chain(prolong_exprs.iter().copied())
            .collect();
        let old_n = vars.len();
        merge_vars(&mut vars, param_vars_for_dae(&work, &scratch, params, pool));
        let n = vars.len();
        for p in &mut active {
            *p = pad_param_gbpoly(p, n);
        }
        // The previous round's basis is still a basis over the wider ring, so
        // the new relations can be tested against it without recomputing.
        let gb_check = gb.extend_vars(n - old_n);

        let mut to_add: Vec<ParamGbPoly> = Vec::new();
        for &d_eq in &prolong_exprs {
            let p =
                expr_to_param_gbpoly(d_eq, &vars, params, pool).map_err(solver_err_to_diffalg)?;
            if !gb_check.contains(&p) {
                to_add.push(p);
            }
        }

        if to_add.is_empty() {
            // Saturated: differentiating adds nothing the ideal did not have.
            return Ok((
                ParametricRosenfeldResult {
                    consistent: true,
                    truncated: false,
                    prolongation_rounds: round,
                    working_dae: work,
                    final_basis: Some(gb_check),
                    minimal_prolongation_rounds,
                },
                DifferentialRanking { vars },
            ));
        }

        active.extend(to_add);
    }

    unreachable!("the `round == max_prolong_rounds` arm returns")
}

/// Calls [`rosenfeld_groebner_with_options`] with the default maximum prolongation rounds.
pub fn rosenfeld_groebner(
    dae: &DAE,
    pool: &ExprPool,
    order: MonomialOrder,
) -> Result<RosenfeldGroebnerResult, DiffAlgError> {
    rosenfeld_groebner_with_options(dae, pool, order, DEFAULT_MAX_PROLONG_ROUNDS)
}

/// Try Pantelides; on [`DaeError::IndexTooHigh`], run [`rosenfeld_groebner`].
pub fn dae_index_reduce(
    dae: &DAE,
    pool: &ExprPool,
    order: MonomialOrder,
) -> Result<DaeIndexReduction, DaeError> {
    dae_index_reduce_ranked(dae, pool, order).map(|(r, _)| r)
}

/// [`dae_index_reduce`] plus the jet [`DifferentialRanking`] for the Gröbner
/// fallback — `None` when Pantelides succeeded and no basis was built.
pub fn dae_index_reduce_ranked(
    dae: &DAE,
    pool: &ExprPool,
    order: MonomialOrder,
) -> Result<(DaeIndexReduction, Option<DifferentialRanking>), DaeError> {
    match pantelides(dae, pool) {
        Ok(p) => Ok((DaeIndexReduction::Pantelides(p), None)),
        Err(DaeError::IndexTooHigh) => {
            let (r, ranking) =
                rosenfeld_groebner_ranked(dae, pool, order, DEFAULT_MAX_PROLONG_ROUNDS).map_err(
                    |e| match e {
                        DiffAlgError::DiffError(s) | DiffAlgError::NotPolynomial(s) => {
                            DaeError::DiffError(s)
                        }
                        DiffAlgError::EmptySystem => DaeError::StructurallyInconsistent,
                    },
                )?;
            Ok((DaeIndexReduction::Rosenfeld(r), Some(ranking)))
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn algebraic_inconsistent_unit_ideal() {
        let p = pool();
        let _x = p.symbol("x", Domain::Real);
        let one_p = GbPoly::constant(rug::Rational::from(1), 1);
        let gx = GbPoly::monomial(vec![1], rug::Rational::from(1));
        let f = gx.add(&one_p); // x+1
        let g = gx.sub(&one_p); // x-1
        let chains = rosenfeld_groebner_algebraic(vec![f, g], MonomialOrder::Lex).unwrap();
        assert!(chains.is_empty());
    }

    #[test]
    fn lotka_volterra_first_order_consistent() {
        let p = pool();
        let t = p.symbol("t", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let dx = p.symbol("dx/dt", Domain::Real);
        let dy = p.symbol("dy/dt", Domain::Real);
        // x' = x - x*y, y' = x*y - y  (coefficients 1)
        let eq1 = p.add(vec![dx, p.mul(vec![p.integer(-1), x]), p.mul(vec![x, y])]);
        let eq2 = p.add(vec![dy, p.mul(vec![p.integer(-1), x, y]), y]);
        let dae = DAE::new(vec![eq1, eq2], vec![x, y], vec![dx, dy], t);
        // Budget 0 = only test the algebraic consistency of the declared first-order ideal (no prolongation).
        let r = rosenfeld_groebner_with_options(&dae, &p, MonomialOrder::GRevLex, 0).unwrap();
        assert!(r.consistent && r.final_basis.is_some());
        assert!(r.truncated);
    }

    #[test]
    fn contradictory_linear_equations_inconsistent() {
        let p = pool();
        let t = p.symbol("t", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let dy = p.symbol("dy/dt", Domain::Real);
        // dy - y = 0  and  dy - y - 1 = 0  → 1 ∈ ideal
        let eq1 = p.add(vec![dy, p.mul(vec![p.integer(-1), y])]);
        let eq2 = p.add(vec![dy, p.mul(vec![p.integer(-1), y]), p.integer(-1)]);
        let dae = DAE::new(vec![eq1, eq2], vec![y], vec![dy], t);
        let r = rosenfeld_groebner(&dae, &p, MonomialOrder::Lex).unwrap();
        assert!(!r.consistent);
    }

    /// Prolongation used to emit relations the system does not imply.
    ///
    /// Two independent causes, both visible on `R' = I, I' = -I` after two
    /// rounds:
    ///
    /// * a state whose derivative had already been promoted got its
    ///   contribution counted twice — once from its own `(I, I')` pair and once
    ///   from the `(I', I'')` pair the previous round added; and
    /// * a jet that was *not* promoted got its contribution dropped entirely,
    ///   because the next differentiation treated it as a constant.  `I''`
    ///   reaches the equations through `R`'s chain without `R'` ever appearing
    ///   in one, so `R'` never became a state and the chain stopped there.
    ///
    /// The first inflates a coefficient, the second deletes a term; together
    /// they collapsed the second prolongation of `R' = I` into `-2·I'' = 0`,
    /// which forces `I = 0` — a relation about a decaying exponential that is
    /// simply false.
    #[test]
    fn prolongation_does_not_invent_relations() {
        let p = pool();
        let t = p.symbol("t", Domain::Real);
        let i = p.symbol("I", Domain::Real);
        let r_ = p.symbol("R", Domain::Real);
        let di = p.symbol("dI/dt", Domain::Real);
        let dr = p.symbol("dR/dt", Domain::Real);
        // R' = I, I' = -I.
        let eq1 = p.add(vec![dr, p.mul(vec![p.integer(-1), i])]);
        let eq2 = p.add(vec![di, i]);
        let dae = DAE::new(vec![eq1, eq2], vec![r_, i], vec![dr, di], t);

        let (res, ranking) = rosenfeld_groebner_ranked(&dae, &p, MonomialOrder::Lex, 2).unwrap();
        assert!(res.consistent);
        let gb = res.final_basis.expect("consistent");

        // The genuine consequence.
        let d2i = p.symbol("ddI/dt/dt", Domain::Real);
        let truth = p.add(vec![d2i, p.mul(vec![p.integer(-1), i])]);
        let truth_poly = expr_to_gbpoly(truth, &ranking.vars, &p).unwrap();
        assert!(gb.contains(&truth_poly), "I'' - I is a consequence");

        // Nothing forces the state to vanish identically.
        for false_claim in [i, di, d2i] {
            let q = expr_to_gbpoly(false_claim, &ranking.vars, &p).unwrap();
            assert!(
                !gb.contains(&q),
                "prolongation asserted a jet of the state vanishes identically"
            );
        }
    }

    /// The jet ranking has to be append-only across prolongation rounds.
    ///
    /// It used to be recomputed from scratch each round, and
    /// [`vars_for_dae`] appends the symbols it scrapes out of the equations
    /// *after* the declared jets — so introducing `d²x/dt²` pushed the trailing
    /// parameter `a` one slot to the right, under polynomials that had already
    /// been padded on the assumption that slot `i` still meant what it meant
    /// last round.  On `x' = a·x`, one prolongation, that silently turned the
    /// input equation into a relation about `d²x/dt²` and lost it from the
    /// basis entirely.
    #[test]
    fn the_jet_ranking_is_append_only() {
        let p = pool();
        let t = p.symbol("t", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let dx = p.symbol("dx/dt", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        // x' = a·x, with `a` an ordinary ring variable (no params here).
        let eq = p.add(vec![dx, p.mul(vec![p.integer(-1), a, x])]);
        let dae = DAE::new(vec![eq], vec![x], vec![dx], t);

        let (res, ranking) =
            rosenfeld_groebner_ranked(&dae, &p, MonomialOrder::GRevLex, 1).unwrap();
        let gb = res.final_basis.expect("consistent");

        // The system's own equation is a consequence of itself.
        let src = expr_to_gbpoly(eq, &ranking.vars, &p).unwrap();
        assert!(gb.contains(&src), "the input equation left the ideal");

        // ...and x·x'' - x' is not: it would need a·x = 1.
        let d2x = p.symbol("ddx/dt/dt", Domain::Real);
        let bogus = p.add(vec![p.mul(vec![x, d2x]), p.mul(vec![p.integer(-1), dx])]);
        let q = expr_to_gbpoly(bogus, &ranking.vars, &p).unwrap();
        assert!(
            !gb.contains(&q),
            "a shifted exponent slot invented a relation"
        );
    }

    /// The parametric route reads the input–output relation of an ODE model
    /// straight out of the DAE, with the rate constant in `Q(a)` rather than as
    /// a fourth ring variable (2026-08-19 issue #16).
    #[test]
    fn parametric_prolongation_yields_the_io_relation() {
        let p = pool();
        let t = p.symbol("t", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let dx = p.symbol("dx/dt", Domain::Real);
        let dy = p.symbol("dy/dt", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        // x' = -a·x, y = x  =>  y' + a·y = 0.
        let eq1 = p.add(vec![dx, p.mul(vec![a, x])]);
        let eq2 = p.add(vec![y, p.mul(vec![p.integer(-1), x])]);
        let dae = DAE::new(vec![eq1, eq2], vec![x, y], vec![dx, dy], t);

        let (r, ranking) = rosenfeld_groebner_parametric(
            &dae,
            &p,
            &[a],
            ParametricProlongOpts {
                order: MonomialOrder::Lex,
                max_prolong_rounds: 3,
                eliminate: &[x],
                minimal: true,
            },
        )
        .unwrap();

        // One derivative of the output is enough, not three.
        assert_eq!(r.minimal_prolongation_rounds, Some(1));
        assert_eq!(r.prolongation_rounds, 1);
        assert!(
            !ranking.vars.contains(&a),
            "`a` must not be a ring variable"
        );

        let gb = r.final_basis.expect("consistent");
        let state_slots: Vec<usize> = ranking
            .vars
            .iter()
            .enumerate()
            .filter(|(_, &v)| {
                let name = p.with(v, |d| match d {
                    ExprData::Symbol { name, .. } => name.clone(),
                    _ => String::new(),
                });
                name.trim_start_matches('d').split('/').next() == Some("x")
            })
            .map(|(i, _)| i)
            .collect();
        let io = gb.eliminate(&state_slots);
        assert_eq!(io.len(), 1);

        let relation = p.add(vec![dy, p.mul(vec![a, y])]);
        let q = expr_to_param_gbpoly(relation, &ranking.vars, &[a], &p).unwrap();
        assert!(io.contains(&q), "y' + a·y = 0 is the input–output relation");
    }

    #[test]
    fn textbook_library_runs() {
        // Ten tiny polynomial DAE snapshots (autonomous, explicit first derivatives).
        let mut n_ok = 0;
        for _ in 0..10 {
            let p = pool();
            let t = p.symbol("t", Domain::Real);
            let x = p.symbol("x", Domain::Real);
            let y = p.symbol("y", Domain::Real);
            let dx = p.symbol("dx/dt", Domain::Real);
            let dy = p.symbol("dy/dt", Domain::Real);
            // Mixed batch: linear dynamics, coupling, and one inconsistent pair.
            let (eqs, v, d, consistent) = match n_ok % 3 {
                0 => {
                    // harmonic x' = y, y' = -x
                    let e1 = p.add(vec![dx, p.mul(vec![p.integer(-1), y])]);
                    let e2 = p.add(vec![dy, x]);
                    (vec![e1, e2], vec![x, y], vec![dx, dy], true)
                }
                1 => {
                    // decoupled exponentials as linear place-holders: x'=x, y'=-y
                    let e1 = p.add(vec![dx, p.mul(vec![p.integer(-1), x])]);
                    let e2 = p.add(vec![dy, y]);
                    (vec![e1, e2], vec![x, y], vec![dx, dy], true)
                }
                _ => {
                    let e1 = p.add(vec![dx, p.mul(vec![p.integer(-1), x])]);
                    let e2 = p.add(vec![dx, p.mul(vec![p.integer(-1), x]), p.integer(-1)]);
                    (vec![e1, e2], vec![x], vec![dx], false)
                }
            };
            let dae = DAE::new(eqs, v, d, t);
            let r = rosenfeld_groebner(&dae, &p, MonomialOrder::GRevLex).unwrap();
            assert_eq!(r.consistent, consistent);
            n_ok += 1;
        }
        assert_eq!(n_ok, 10);
    }
}
