//! Systems in which an unknown occurs only inside one invertible function, and
//! the switch to complex solutions.
//!
//! `exp(x) + y = 3, exp(x) − y = 1` is polynomial in `u = exp(x)` and `y`.
//! When every occurrence of an unknown is inside one kernel — `exp`, `log`,
//! `sin`, `cos`, `tan` of an affine argument in that unknown alone, or a power
//! `b^(a·x + c)` with a base free of the unknowns (`exp(2x) = exp(x)²` counts as
//! the same kernel) — the kernel is replaced by a fresh unknown, the polynomial
//! system is solved, and each value is inverted:
//!
//! * `exp(f) = u` → `f = log u`, needing `u > 0` over the reals (a value proved
//!   otherwise drops the tuple; an undecided one records `u > 0`); over ℂ
//!   `f = log u + 2πik` for `u ≠ 0`;
//! * `log(f) = u` → `f = exp u` (over ℂ only when `−π < Im u ≤ π`);
//! * `b^f = u` → `f = log u / log b`, with the same conditions as `exp`;
//! * `sin`, `cos`, `tan` → the periodic families of the scalar solver.
//!
//! Every tuple is substituted back into the equations as written at generic
//! values of the parameters and of the family indices; one that fails refuses
//! the call.

use super::*;
use crate::deriv::log::SideCondition;
use crate::kernel::Domain;
use crate::solver::{Solution, SolutionSet, SolverError};
use std::cell::Cell;

thread_local! {
    /// Set while [`with_complex_domain`] runs.
    static COMPLEX_DOMAIN: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with the transcendental solvers reporting **complex** solutions:
/// `exp(x) = c` gives `x = log c + 2πik` (`k ∈ ℤ`) rather than the one real
/// logarithm, `sin(x) = 2` gives its complex families rather than nothing, and
/// a Lambert-W equation (a root on each of infinitely many complex branches;
/// only the real `W₀`/`W₋₁` are available) is refused.
///
/// An unknown declared with [`Domain::Complex`] selects the same mode for its
/// own equation without this wrapper. The setting is per thread and is
/// restored when `f` returns.
pub fn with_complex_domain<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            COMPLEX_DOMAIN.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(COMPLEX_DOMAIN.with(|c| c.replace(true)));
    f()
}

/// Are complex solutions wanted for `var`?
pub(super) fn complex_mode(var: ExprId, pool: &ExprPool) -> bool {
    COMPLEX_DOMAIN.with(|c| c.get())
        || pool.with(var, |d| {
            matches!(
                d,
                ExprData::Symbol {
                    domain: Domain::Complex,
                    ..
                }
            )
        })
}

/// The invertible function an unknown sits inside.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Inv {
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    /// `base^(…)` with `base` free of the unknowns.
    Power(ExprId),
}

#[derive(Clone, Copy)]
struct Occurrence {
    kind: Inv,
    node: ExprId,
    /// `a·var + b`.
    a: ExprId,
    b: ExprId,
}

/// Walk `e`, recording every function of an unknown. `false` when an unknown
/// sits somewhere this module cannot invert.
fn collect(
    e: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
    depth: usize,
    out: &mut Vec<(usize, Occurrence)>,
) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    let owner = |arg: ExprId| -> Option<usize> {
        let hits: Vec<usize> = (0..vars.len())
            .filter(|&i| contains_var(arg, vars[i], pool, 0))
            .collect();
        (hits.len() == 1).then(|| hits[0])
    };
    match pool.get(e) {
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().all(|&a| collect(a, vars, pool, depth + 1, out))
        }
        ExprData::Pow { base, exp } => {
            let exp_free = vars.iter().all(|&v| !contains_var(exp, v, pool, 0));
            if exp_free {
                if vars.iter().all(|&v| !contains_var(base, v, pool, 0)) {
                    return true;
                }
                // A polynomial power of something holding an unknown.
                return pool.with(exp, |d| matches!(d, ExprData::Integer(_)))
                    && collect(base, vars, pool, depth + 1, out);
            }
            if vars.iter().any(|&v| contains_var(base, v, pool, 0)) {
                return false;
            }
            let Some(i) = owner(exp) else { return false };
            let Some((a, b)) = affine_in_var(exp, vars[i], pool) else {
                return false;
            };
            out.push((
                i,
                Occurrence {
                    kind: Inv::Power(base),
                    node: e,
                    a,
                    b,
                },
            ));
            true
        }
        ExprData::Func { name, args } => {
            if vars.iter().all(|&v| !contains_var(e, v, pool, 0)) {
                return true;
            }
            let kind = match name.as_str() {
                "exp" => Inv::Exp,
                "log" | "ln" => Inv::Log,
                "sin" => Inv::Sin,
                "cos" => Inv::Cos,
                "tan" => Inv::Tan,
                _ => return false,
            };
            if args.len() != 1 {
                return false;
            }
            let Some(i) = owner(args[0]) else {
                return false;
            };
            let Some((a, b)) = affine_in_var(args[0], vars[i], pool) else {
                return false;
            };
            out.push((
                i,
                Occurrence {
                    kind,
                    node: e,
                    a,
                    b,
                },
            ));
            true
        }
        _ => true,
    }
}

/// One unknown's substitution: `var` is replaced by `u`, its kernel is
/// `kind(a·var + b)`, and every occurrence `node` stands for `scale·uⁿ`.
struct Plan {
    var: ExprId,
    u: ExprId,
    kind: Inv,
    a: ExprId,
    b: ExprId,
    replace: Vec<(ExprId, ExprId)>,
}

/// Group the occurrences of one unknown into a single kernel.
fn plan_for(var: ExprId, occ: &[Occurrence], idx: usize, pool: &ExprPool) -> Option<Plan> {
    let u = pool.symbol(format!("__inv_u_{idx}__"), Domain::Complex);
    let first = occ[0];
    if occ.iter().all(|o| o.node == first.node) {
        return Some(Plan {
            var,
            u,
            kind: first.kind,
            a: first.a,
            b: first.b,
            replace: vec![(first.node, u)],
        });
    }
    // exp(nᵢ·a·x + bᵢ) = exp(bᵢ − nᵢ·b)·uⁿⁱ for u = exp(a·x + b), positive
    // integers nᵢ; likewise for a power of one base.
    if !occ
        .iter()
        .all(|o| o.kind == first.kind && matches!(o.kind, Inv::Exp | Inv::Power(_)))
    {
        return None;
    }
    let neg_one = pool.integer(-1_i32);
    'base: for base in occ {
        let inv_a = pool.pow(base.a, neg_one);
        let mut replace = Vec::new();
        for o in occ {
            let Some(r) = as_rational(simp(pool.mul(vec![o.a, inv_a]), pool), pool) else {
                continue 'base;
            };
            if *r.denom() != 1 || r <= 0 {
                continue 'base;
            }
            let n = r.numer().to_i32()?;
            let shift = simp(
                pool.add(vec![o.b, pool.mul(vec![pool.integer(-n), base.b])]),
                pool,
            );
            let power = pool.pow(u, pool.integer(n));
            let rep = if is_zero_expr(shift, pool) {
                power
            } else {
                let factor = match base.kind {
                    Inv::Power(b) => pool.pow(b, shift),
                    _ => pool.func("exp", vec![shift]),
                };
                pool.mul(vec![factor, power])
            };
            replace.push((o.node, rep));
        }
        return Some(Plan {
            var,
            u,
            kind: base.kind,
            a: base.a,
            b: base.b,
            replace,
        });
    }
    None
}

/// Names for family indices that clash with nothing in `exprs`.
struct IndexNames {
    taken: std::collections::HashSet<String>,
}

impl IndexNames {
    fn new(exprs: &[ExprId], pool: &ExprPool) -> Self {
        let mut taken = std::collections::HashSet::new();
        for &e in exprs {
            for s in crate::poly::collect_free_vars(e, pool) {
                if let ExprData::Symbol { name, .. } = pool.get(s) {
                    taken.insert(name);
                }
            }
        }
        IndexNames { taken }
    }

    fn fresh(&mut self, pool: &ExprPool) -> ExprId {
        let fixed = ["k", "n", "m", "j"].iter().map(|s| s.to_string());
        let numbered = (1..).map(|i| format!("k_{i}"));
        let name = fixed
            .chain(numbered)
            .find(|n| !self.taken.contains(n))
            .expect("an unused name exists");
        self.taken.insert(name.clone());
        pool.symbol(name, Domain::Integer)
    }
}

/// Outcome of inverting one kernel value.
enum Inverted {
    /// The values of the kernel's argument.
    Values(Vec<ExprId>),
    /// No value: the tuple is not a solution in this domain.
    Drop,
}

/// Rigorous real enclosure of a closed value, or `None`.
fn real_enclosure(u: ExprId, pool: &ExprPool) -> Option<(f64, f64)> {
    let free = crate::poly::collect_free_vars(u, pool);
    if free
        .iter()
        .any(|&s| !crate::eval::symbols::is_named_constant(s, pool))
    {
        return None;
    }
    let b = crate::ball::IntervalEval::new(256).eval(u, pool)?;
    let (lo, hi) = (b.lo().to_f64(), b.hi().to_f64());
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

fn is_closed(u: ExprId, pool: &ExprPool) -> bool {
    crate::poly::collect_free_vars(u, pool)
        .iter()
        .all(|&s| crate::eval::symbols::is_named_constant(s, pool))
}

/// Is the closed value `u` certainly not real? (`false` when undecided.)
fn certainly_non_real(u: ExprId, pool: &ExprPool) -> bool {
    if !is_closed(u, pool) {
        return false;
    }
    let mut ev = crate::solver::verify::CBallEval::default();
    matches!(ev.eval(u, pool), Ok(b) if !b.im_ball().contains(0.0))
}

/// Invert `kind(arg) = u` for `arg`.
fn invert(
    kind: Inv,
    u: ExprId,
    complex: bool,
    names: &mut IndexNames,
    indices: &mut Vec<ExprId>,
    pool: &ExprPool,
) -> Result<Inverted, String> {
    let pi = crate::eval::symbols::pi_symbol(pool);
    let neg = |e: ExprId| pool.mul(vec![pool.integer(-1_i32), e]);
    let two_pi_i_k = |names: &mut IndexNames, indices: &mut Vec<ExprId>| {
        let k = names.fresh(pool);
        indices.push(k);
        pool.mul(vec![pool.integer(2_i32), pi, pool.imaginary_unit(), k])
    };
    let closed = is_closed(u, pool);
    match kind {
        Inv::Exp | Inv::Power(_) => {
            let lg = |e: ExprId| match kind {
                Inv::Power(b) => pool.mul(vec![
                    pool.func("log", vec![e]),
                    pool.pow(pool.func("log", vec![b]), pool.integer(-1_i32)),
                ]),
                _ => pool.func("log", vec![e]),
            };
            if complex {
                if closed && crate::matrix::zero_test::zero_status(pool, u).is_proven_zero() {
                    return Ok(Inverted::Drop);
                }
                if !closed {
                    crate::solver::assume_nonzero(u);
                }
                let shift = two_pi_i_k(names, indices);
                let log_u = pool.add(vec![pool.func("log", vec![u]), shift]);
                let v = match kind {
                    Inv::Power(b) => pool.mul(vec![
                        log_u,
                        pool.pow(pool.func("log", vec![b]), pool.integer(-1_i32)),
                    ]),
                    _ => log_u,
                };
                return Ok(Inverted::Values(vec![v]));
            }
            if certainly_non_real(u, pool) {
                return Ok(Inverted::Drop);
            }
            if closed {
                match real_enclosure(u, pool) {
                    Some((lo, _)) if lo > 0.0 => {}
                    Some((_, hi)) if hi <= 0.0 => return Ok(Inverted::Drop),
                    _ => {
                        if crate::matrix::zero_test::zero_status(pool, u).is_proven_zero() {
                            return Ok(Inverted::Drop);
                        }
                        return Err(format!(
                            "it cannot decide whether {} is positive",
                            pool.display(u)
                        ));
                    }
                }
            } else {
                crate::solver::record_side_condition(SideCondition::Positive(u));
            }
            if let Inv::Power(b) = kind {
                if require_positive(b, pool) == Some(false) {
                    return Err(format!(
                        "the base {} is not positive, so its real powers are not defined",
                        pool.display(b)
                    ));
                }
            }
            Ok(Inverted::Values(vec![lg(u)]))
        }
        Inv::Log => {
            if complex {
                if closed {
                    let mut ev = crate::solver::verify::CBallEval::default();
                    if let Ok(b) = ev.eval(u, pool) {
                        let im = b.im_ball();
                        let pi_f = std::f64::consts::PI;
                        if im.lo().to_f64() > pi_f + 1e-9 || im.hi().to_f64() < -pi_f + 1e-9 {
                            return Ok(Inverted::Drop);
                        }
                        if !(im.hi().to_f64() < pi_f - 1e-9 && im.lo().to_f64() > -pi_f) {
                            return Err(format!(
                                "it cannot place Im {} relative to the branch cut of log",
                                pool.display(u)
                            ));
                        }
                    }
                }
            } else if certainly_non_real(u, pool) {
                return Ok(Inverted::Drop);
            }
            Ok(Inverted::Values(vec![pool.func("exp", vec![u])]))
        }
        Inv::Sin | Inv::Cos | Inv::Tan => {
            if !complex && certainly_non_real(u, pool) {
                return Ok(Inverted::Drop);
            }
            if !complex && kind != Inv::Tan {
                if let Some((lo, hi)) = real_enclosure(u, pool) {
                    if lo > 1.0 || hi < -1.0 {
                        return Ok(Inverted::Drop);
                    }
                }
            }
            if kind == Inv::Tan && complex && closed {
                let e = pool.add(vec![pool.pow(u, pool.integer(2_i32)), pool.integer(1_i32)]);
                if crate::matrix::zero_test::zero_status(pool, e).is_proven_zero() {
                    return Ok(Inverted::Drop);
                }
            }
            let k = names.fresh(pool);
            indices.push(k);
            let two_pi_k = pool.mul(vec![pool.integer(2_i32), pi, k]);
            let extreme = matches!(as_rational(u, pool), Some(r) if r == 1 || r == -1);
            let vals = match kind {
                Inv::Sin => {
                    let s = pool.func("asin", vec![u]);
                    let mut v = vec![pool.add(vec![s, two_pi_k])];
                    if !extreme {
                        v.push(pool.add(vec![pi, neg(s), two_pi_k]));
                    }
                    v
                }
                Inv::Cos => {
                    let s = pool.func("acos", vec![u]);
                    let mut v = vec![pool.add(vec![s, two_pi_k])];
                    if !extreme {
                        v.push(pool.add(vec![neg(s), two_pi_k]));
                    }
                    v
                }
                _ => vec![pool.add(vec![pool.func("atan", vec![u]), pool.mul(vec![pi, k])])],
            };
            Ok(Inverted::Values(vals))
        }
    }
}

/// Retry a system the polynomial path refused because an unknown sits inside
/// a function. `None` when the reduction does not apply.
pub(crate) fn solve_with_invertible_kernels(
    equations: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Option<Result<SolutionSet, SolverError>> {
    // One equation is the scalar transcendental solver's business (the
    // polynomial path keeps refusing it with its own code).
    if equations.len() < 2 {
        return None;
    }
    let mut occ: Vec<(usize, Occurrence)> = Vec::new();
    for &e in equations {
        if !collect(e, vars, pool, 0, &mut occ) {
            return None;
        }
    }
    if occ.is_empty() {
        return None;
    }
    let mut plans: Vec<Plan> = Vec::new();
    for (i, &v) in vars.iter().enumerate() {
        let mine: Vec<Occurrence> = occ
            .iter()
            .filter(|(j, _)| *j == i)
            .map(|(_, o)| *o)
            .collect();
        if mine.is_empty() {
            continue;
        }
        plans.push(plan_for(v, &mine, i, pool)?);
    }
    let mut new_eqs = Vec::with_capacity(equations.len());
    for &e in equations {
        let mut cur = e;
        for p in &plans {
            for &(node, rep) in &p.replace {
                cur = substitute_node(cur, node, rep, pool, 0)?;
            }
        }
        new_eqs.push(cur);
    }
    // Each substituted unknown must be gone: it occurred only in its kernel.
    if plans
        .iter()
        .any(|p| new_eqs.iter().any(|&e| contains_var(e, p.var, pool, 0)))
    {
        return None;
    }
    let new_vars: Vec<ExprId> = vars
        .iter()
        .map(|&v| plans.iter().find(|p| p.var == v).map_or(v, |p| p.u))
        .collect();

    let refuse = |why: String| -> Option<Result<SolutionSet, SolverError>> {
        Some(Err(SolverError::NotPolynomial(format!(
            "each unknown inside a function occurs only there, so the system was solved for the \
             function values, but {why}"
        ))))
    };
    let inner = crate::solver::solve_polynomial_system(new_eqs, new_vars.clone(), pool);
    let set = match inner {
        Err(e) => return Some(Err(e)),
        Ok(SolutionSet::NoSolution) => return Some(Ok(SolutionSet::NoSolution)),
        Ok(SolutionSet::Parametric(_)) => {
            return refuse("the solution set is not finite".into());
        }
        Ok(SolutionSet::Finite(s)) => s,
    };
    let kept_conditions = crate::solver::take_solve_side_conditions();

    let mut names = IndexNames::new(equations, pool);
    let mut indices: Vec<ExprId> = Vec::new();
    let mut out: Vec<Solution> = Vec::new();
    for sol in &set {
        // Candidate values per unknown; the tuple set is their product.
        let mut choices: Vec<Vec<ExprId>> = Vec::with_capacity(vars.len());
        let mut dropped = false;
        for (i, &v) in vars.iter().enumerate() {
            let Some(p) = plans.iter().find(|p| p.var == v) else {
                choices.push(vec![sol[i]]);
                continue;
            };
            let complex = complex_mode(v, pool);
            match invert(p.kind, sol[i], complex, &mut names, &mut indices, pool) {
                Err(why) => return refuse(why),
                Ok(Inverted::Drop) => {
                    dropped = true;
                    break;
                }
                Ok(Inverted::Values(args)) => choices.push(
                    args.into_iter()
                        .map(|arg| simp(solve_affine(&p.a, &p.b, arg, pool), pool))
                        .collect(),
                ),
            }
        }
        if dropped {
            continue;
        }
        let mut tuples: Vec<Solution> = vec![Vec::new()];
        for c in &choices {
            let mut next = Vec::with_capacity(tuples.len() * c.len());
            for t in &tuples {
                for &val in c {
                    let mut t2 = t.clone();
                    t2.push(val);
                    next.push(t2);
                }
            }
            tuples = next;
        }
        out.extend(tuples);
    }
    for t in &out {
        match crate::solver::opaque::spot_check(equations, vars, t, &indices, pool) {
            crate::solver::opaque::SpotCheck::Passed => {}
            _ => {
                return refuse(
                    "a returned tuple could not be confirmed by substitution into the equations"
                        .into(),
                )
            }
        }
    }
    for c in kept_conditions {
        match c {
            SideCondition::NonZero(e) => crate::solver::assume_nonzero(e),
            other => crate::solver::record_side_condition(other),
        }
    }
    for k in indices {
        crate::solver::record_side_condition(SideCondition::InDomain(k, Domain::Integer));
    }
    Some(Ok(SolutionSet::Finite(out)))
}
