//! Branch and bound for mixed-integer linear programmes, on top of the exact
//! simplex.
//!
//! Node selection is depth-first until the first integer-feasible point is
//! found (to get an incumbent quickly), then best-bound. Branching is on the
//! lowest-index integer variable whose LP value is fractional. Every node LP is
//! solved exactly and its certificate checked by [`super::linprog`], so a node
//! pruned as infeasible carries a verified Farkas vector and a node pruned by
//! bound carries a verified dual; when the open list empties, the incumbent is
//! optimal with a gap of exactly zero.
//!
//! Before branching, the root relaxation can be strengthened by rounds of
//! Gomory mixed-integer cuts (cut-and-branch): integer variables' bounds are
//! first rounded inward (valid for every integer point), then each round
//! solves the relaxation with the revised simplex, reads GMI cuts off the
//! optimal tableau ([`super::revised`] documents why each is valid), appends
//! them as inequality rows and re-solves from the previous basis (a dual
//! simplex). Cuts never remove an integer-feasible point, so the search on the
//! strengthened programme has the same integer optimum; the incumbent is
//! re-checked against the caller's programme before it is returned. Every node
//! LP is warm-started from its parent's basis.
//!
//! Unboundedness is decided by Meyer's theorem: for rational data, if the
//! integer hull is non-empty its recession cone equals that of the LP
//! relaxation. So once some node LP is unbounded along a ray `d` (a ray of the
//! root relaxation too, since branching only adds bounds) *and* an integer
//! point `x₀` is known, `x₀ + t·d` (with `d` scaled to be integral) is an
//! integer-feasible family whose objective is unbounded. Without an integer
//! point the search continues, and may stop at the node limit.

use rug::{Integer, Rational};

use super::{
    linprog_with, revised, Incumbent, LinearProgram, LinprogError, LpBasis, LpMethod, LpOptions,
    LpOutcome, Sense, UnboundedCertificate, VarBound,
};
use crate::budget;

/// Options for [`milp`]; build with [`MilpOptions::new`] and adjust fields.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MilpOptions {
    /// `true` for each variable constrained to be an integer.
    pub integrality: Vec<bool>,
    /// The maximum number of branch-and-bound nodes whose LP is solved
    /// before refusing with `E-LP-005`.
    pub max_nodes: usize,
    /// Rounds of Gomory mixed-integer cuts at the root (0 disables cutting).
    pub cut_rounds: usize,
    /// The maximum total number of cuts added at the root.
    pub max_cuts: usize,
    /// Warm-start each node LP from its parent's basis (revised simplex).
    pub warm_start: bool,
}

impl MilpOptions {
    /// The default node cap.
    pub const DEFAULT_MAX_NODES: usize = 100_000;
    /// The default number of root cut rounds.
    pub const DEFAULT_CUT_ROUNDS: usize = 8;
    /// The default cap on the number of root cuts.
    pub const DEFAULT_MAX_CUTS: usize = 200;

    /// Integrality flags with the default node cap, cut limits and warm starts.
    pub fn new(integrality: Vec<bool>) -> Self {
        MilpOptions {
            integrality,
            max_nodes: Self::DEFAULT_MAX_NODES,
            cut_rounds: Self::DEFAULT_CUT_ROUNDS,
            max_cuts: Self::DEFAULT_MAX_CUTS,
            warm_start: true,
        }
    }
}

/// A proven-optimal integer solution.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MilpSolution {
    /// The optimal point; integer-valued on every integer variable.
    pub x: Vec<Rational>,
    /// Its objective value.
    pub objective: Rational,
    /// Number of nodes whose LP was solved. The search tree closed, so the
    /// bound equals `objective` (gap exactly 0).
    pub nodes: usize,
    /// The Gomory cuts added at the root, as rows `a · x ≤ b` in the caller's
    /// variables. Each is satisfied by every integer-feasible point.
    pub cuts: Vec<(Vec<Rational>, Rational)>,
}

/// Outcome of [`milp`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MilpOutcome {
    /// An optimal integer solution, proven by a closed search tree.
    Optimal(MilpSolution),
    /// No integer-feasible point exists: every leaf of the closed search tree
    /// was LP-infeasible (each with a verified Farkas vector).
    Infeasible {
        /// Number of nodes whose LP was solved.
        nodes: usize,
    },
    /// The objective is unbounded over the integer points: `point` is
    /// integer-feasible and `direction` is integral on the integer variables,
    /// so every `point + t·direction` (`t ∈ ℕ`) is integer-feasible.
    Unbounded(UnboundedCertificate),
}

struct Node {
    bounds: Vec<VarBound>,
    /// The parent's final basis, for a warm start.
    basis: Option<LpBasis>,
    /// Parent LP value in minimisation form; `None` when unknown (−∞).
    bound: Option<Rational>,
}

fn is_integer(v: &Rational) -> bool {
    *v.denom() == 1
}

/// Solve a mixed-integer linear programme exactly by branch and bound.
///
/// # Errors
/// * `E-LP-001` — malformed shapes, or an integrality list of the wrong length.
/// * `E-LP-005` — `max_nodes` node LPs were solved without closing the tree;
///   the best incumbent (if any) and the best open bound are carried.
/// * `E-BUDGET-*` — the active budget tripped; incumbent and bound carried.
/// * `E-LP-006` — a certificate failed its exact check (a bug).
pub fn milp(lp: &LinearProgram, opts: &MilpOptions) -> Result<MilpOutcome, LinprogError> {
    lp.validate()?;
    let n = lp.nvars();
    if opts.integrality.len() != n {
        return Err(LinprogError::Shape {
            detail: format!(
                "integrality has {} entries for {n} variables",
                opts.integrality.len()
            ),
        });
    }
    let to_min = |v: &Rational| -> Rational {
        match lp.sense {
            Sense::Minimize => v.clone(),
            Sense::Maximize => -v.clone(),
        }
    };
    let from_min = &to_min;

    // Round integer variables' bounds inward (valid for every integer point),
    // then strengthen the root with Gomory cuts.
    let mut work = lp.clone();
    for (bd, &int) in work.bounds.iter_mut().zip(&opts.integrality) {
        if int {
            bd.lower = bd.lower.take().map(|l| l.ceil());
            bd.upper = bd.upper.take().map(|u| u.floor());
        }
    }
    let (cuts, root_basis) = root_cuts(&mut work, opts)?;
    let lp_orig = lp;
    let lp = &work;

    let mut open = vec![Node {
        bounds: lp.bounds.clone(),
        basis: root_basis,
        bound: None,
    }];
    let mut incumbent: Option<Incumbent> = None;
    let mut inc_min: Option<Rational> = None;
    let mut ray: Option<Vec<Rational>> = None;
    let mut nodes = 0usize;

    // Best bound over the open nodes (and `extra`), in the caller's sense.
    let best_open = |open: &[Node], extra: Option<&Node>, inc_min: &Option<Rational>| {
        let mut best: Option<Rational> = inc_min.clone();
        for node in open.iter().chain(extra) {
            match &node.bound {
                None => return None,
                Some(b) => {
                    if best.as_ref().map_or(true, |cur| b < cur) {
                        best = Some(b.clone());
                    }
                }
            }
        }
        best.map(|b| from_min(&b))
    };

    loop {
        if let (Some(d), Some(inc)) = (&ray, &incumbent) {
            return unbounded_certificate(lp_orig, opts, inc, d);
        }
        // Depth-first until an incumbent exists, then best bound.
        let next = if incumbent.is_none() {
            open.pop()
        } else {
            let mut best_i: Option<usize> = None;
            for (i, node) in open.iter().enumerate() {
                let better = match best_i {
                    None => true,
                    Some(k) => match (&node.bound, &open[k].bound) {
                        (None, Some(_)) => true,
                        (Some(a), Some(b)) => a < b,
                        _ => false,
                    },
                };
                if better {
                    best_i = Some(i);
                }
            }
            best_i.map(|i| open.swap_remove(i))
        };
        let Some(node) = next else { break };
        if let (Some(b), Some(z)) = (&node.bound, &inc_min) {
            if b >= z {
                continue;
            }
        }
        if nodes >= opts.max_nodes {
            return Err(LinprogError::NodeLimit {
                nodes,
                bound: best_open(&open, Some(&node), &inc_min),
                incumbent,
            });
        }
        let with_progress = |e: LinprogError, open: &[Node], node: &Node| match e {
            LinprogError::Budget { source, .. } => LinprogError::Budget {
                source,
                incumbent: incumbent.clone(),
                bound: best_open(open, Some(node), &inc_min),
            },
            other => other,
        };
        if let Err(e) = budget::check() {
            return Err(with_progress(e.into(), &open, &node));
        }
        nodes += 1;

        let mut sub = lp.clone();
        sub.bounds = node.bounds.clone();
        let lp_opts = match (&node.basis, opts.warm_start) {
            (Some(b), true) => LpOptions::default().with_basis(b.clone()),
            _ => LpOptions::default(),
        };
        let solved = match linprog_with(&sub, &lp_opts) {
            Ok(s) => s,
            Err(e) => return Err(with_progress(e, &open, &node)),
        };
        let child_basis = opts.warm_start.then_some(solved.basis);
        let outcome = solved.outcome;
        let (point, value) = match outcome {
            LpOutcome::Infeasible(_) => continue,
            LpOutcome::Optimal(cert) => {
                let v = to_min(&cert.objective);
                if inc_min.as_ref().is_some_and(|z| v >= *z) {
                    continue;
                }
                (cert.x, Some(v))
            }
            LpOutcome::Unbounded(cert) => {
                ray = Some(cert.direction);
                (cert.point, None)
            }
        };
        let frac = (0..n).find(|&j| opts.integrality[j] && !is_integer(&point[j]));
        let Some(j) = frac else {
            // Integer-feasible.
            let obj = lp.objective_at(&point);
            let v = to_min(&obj);
            if inc_min.as_ref().map_or(true, |z| v < *z) {
                inc_min = Some(v);
                incumbent = Some(Incumbent {
                    x: point,
                    objective: obj,
                });
            }
            continue;
        };
        let v = &point[j];
        let fl = v.clone().floor();
        let ce = v.clone().ceil();
        let bd = &node.bounds[j];
        // Up child first on the stack, so the down child is explored first.
        let mut up = node.bounds.clone();
        if bd.lower.as_ref().map_or(true, |l| *l < ce) {
            up[j].lower = Some(ce);
        }
        if up[j]
            .upper
            .as_ref()
            .map_or(true, |u| up[j].lower.as_ref().unwrap() <= u)
        {
            open.push(Node {
                bounds: up,
                basis: child_basis.clone(),
                bound: value.clone(),
            });
        }
        let mut down = node.bounds;
        if down[j].upper.as_ref().map_or(true, |u| *u > fl) {
            down[j].upper = Some(fl);
        }
        if down[j]
            .lower
            .as_ref()
            .map_or(true, |l| l <= down[j].upper.as_ref().unwrap())
        {
            open.push(Node {
                bounds: down,
                basis: child_basis,
                bound: value,
            });
        }
    }

    match incumbent {
        Some(inc) => {
            if !lp_orig.is_feasible(&inc.x)
                || (0..n).any(|j| opts.integrality[j] && !is_integer(&inc.x[j]))
            {
                return Err(LinprogError::Internal {
                    detail: "incumbent is not integer-feasible".into(),
                });
            }
            Ok(MilpOutcome::Optimal(MilpSolution {
                x: inc.x,
                objective: inc.objective,
                nodes,
                cuts,
            }))
        }
        None => Ok(MilpOutcome::Infeasible { nodes }),
    }
}

/// Rounds of Gomory mixed-integer cuts on the root relaxation of `work`,
/// appended to `work.a_ub`. Returns the cuts and the last root basis.
fn root_cuts(
    work: &mut LinearProgram,
    opts: &MilpOptions,
) -> Result<(Vec<(Vec<Rational>, Rational)>, Option<LpBasis>), LinprogError> {
    let mut cuts: Vec<(Vec<Rational>, Rational)> = Vec::new();
    let mut basis: Option<LpBasis> = None;
    let mut last: Option<Rational> = None;
    for _ in 0..opts.cut_rounds {
        if cuts.len() >= opts.max_cuts {
            break;
        }
        let (engine, outcome, _) = revised::solve(work, basis.as_ref())?;
        outcome
            .verify(work)
            .map_err(|detail| LinprogError::Internal { detail })?;
        let LpOutcome::Optimal(cert) = outcome else {
            break;
        };
        basis = Some(engine.basis_labels(work));
        // Stop when a round no longer moves the bound.
        if last.as_ref() == Some(&cert.objective) {
            break;
        }
        last = Some(cert.objective.clone());
        let fractional = cert
            .x
            .iter()
            .zip(&opts.integrality)
            .any(|(v, &int)| int && !is_integer(v));
        if !fractional {
            break;
        }
        let room = opts.max_cuts - cuts.len();
        let new = engine.gomory_cuts(work, &opts.integrality, &cert.x, room.min(50));
        if new.is_empty() {
            break;
        }
        for (row, rhs) in new {
            work.a_ub.push(row.clone());
            work.b_ub.push(rhs.clone());
            cuts.push((row, rhs));
        }
    }
    let _ = LpMethod::Revised;
    Ok((cuts, basis))
}

fn unbounded_certificate(
    lp: &LinearProgram,
    opts: &MilpOptions,
    inc: &Incumbent,
    d: &[Rational],
) -> Result<MilpOutcome, LinprogError> {
    // Clear denominators so the ray is integral everywhere.
    let mut l = Integer::from(1);
    for v in d {
        l.lcm_mut(v.denom());
    }
    let lr = Rational::from(l);
    let direction: Vec<Rational> = d.iter().map(|v| Rational::from(v * &lr)).collect();
    let cert = UnboundedCertificate {
        point: inc.x.clone(),
        direction,
    };
    cert.verify(lp)
        .map_err(|detail| LinprogError::Internal { detail })?;
    let n = lp.nvars();
    if (0..n).any(|j| {
        opts.integrality[j] && (!is_integer(&cert.point[j]) || !is_integer(&cert.direction[j]))
    }) {
        return Err(LinprogError::Internal {
            detail: "unbounded MILP certificate is not integral".into(),
        });
    }
    Ok(MilpOutcome::Unbounded(cert))
}
