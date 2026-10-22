//! Pryce's Σ-method — the structural analysis behind [`super::pantelides`].
//!
//! The previous implementation matched an equation to a variable whenever the
//! equation mentioned the variable *or any of its derivatives*, and reported
//! the number of matching rounds as the index.  That is not Pantelides: the
//! Cartesian pendulum matched after one round and came back as index 1 (the
//! textbook value is 3), and `x' = y, x = sin t` came back as index 0 (it is
//! 2).  The reduced systems it returned left the Lagrange multiplier
//! undetermined.
//!
//! This module does the analysis properly, in the form Pryce gives it
//! ("A simple structural analysis method for DAEs", BIT 41, 2001), which is
//! equivalent to Pantelides' algorithm and directly yields the quantities a
//! caller needs:
//!
//! 1. Group the variables into **jet chains**: `x`, `x'`, `x''`, … of one
//!    unknown function.  A variable listed as another's derivative is a jet of
//!    that chain, not a new unknown.
//! 2. The **signature matrix** `Σ[i][j]` is the highest derivative order of
//!    chain `j` in equation `i` (`-∞` if absent).
//! 3. A **highest-value transversal** `T` — an assignment of equations to
//!    chains maximising `Σ σ_{i,T(i)}` — exists iff the system is
//!    structurally non-singular.
//! 4. The smallest **offsets** `c_i, d_j ≥ 0` with `d_j − c_i ≥ σ_ij`
//!    (equality on `T`) say that equation `i` must be differentiated `c_i`
//!    times, after which chain `j` appears to order `d_j`.
//! 5. The **system Jacobian** `J_ij = ∂f_i / ∂x_j^{(d_j − c_i)}` must be
//!    non-singular; if it is not, the structural analysis has under-estimated
//!    the index and the "reduced" system does not determine the highest
//!    derivatives — that is refused, never returned.
//!
//! The structural index is `ν = max c_i + (1 if some d_j = 0 else 0)`, which
//! is the differentiation index whenever step 5 succeeds.

use super::{differentiate_equation, structurally_depends, DaeError, DaeStructureError, DAE};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use rug::Rational;
use std::collections::{HashMap, HashSet};

/// Largest number of times one equation may be differentiated.
pub(super) const MAX_DIFFERENTIATIONS: usize = 10;

/// `d{name}/dt` — the naming rule every DAE routine shares for a jet's successor.
pub(crate) fn next_jet_symbol(jet: ExprId, pool: &ExprPool) -> ExprId {
    let name = pool.with(jet, |d| match d {
        ExprData::Symbol { name, .. } => format!("d{name}/dt"),
        _ => "d?/dt".to_string(),
    });
    pool.symbol(&name, Domain::Real)
}

fn is_symbol(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Symbol { .. }))
}

fn show(e: ExprId, pool: &ExprPool) -> String {
    pool.display(e).to_string()
}

/// Check the shape of a DAE: every variable and derivative a distinct symbol,
/// no more derivatives than variables, and the time variable not among them.
///
/// Fewer derivatives than variables is legal — the trailing variables are
/// algebraic (see [`normalized_jets`]).
pub(crate) fn validate_dae_shape(dae: &DAE, pool: &ExprPool) -> Result<(), DaeStructureError> {
    let malformed = |s: String| Err(DaeStructureError::MalformedDae(s));
    if dae.derivatives.len() > dae.variables.len() {
        return malformed(format!(
            "{} derivative symbols for {} variables: derivatives[i] names the derivative \
             of variables[i], so there cannot be more of them (trailing variables without \
             a derivative are algebraic)",
            dae.derivatives.len(),
            dae.variables.len()
        ));
    }
    let mut seen_vars = HashSet::new();
    for &v in &dae.variables {
        if !is_symbol(v, pool) {
            return malformed(format!("variable `{}` is not a symbol", show(v, pool)));
        }
        if !seen_vars.insert(v) {
            return malformed(format!("variable `{}` is listed twice", show(v, pool)));
        }
        if v == dae.time_var {
            return malformed(format!(
                "the time variable `{}` is listed as a dependent variable",
                show(v, pool)
            ));
        }
    }
    let mut seen_derivs = HashSet::new();
    for (i, &d) in dae.derivatives.iter().enumerate() {
        if !is_symbol(d, pool) {
            return malformed(format!(
                "derivative `{}` is not a symbol; name the derivative with a fresh symbol \
                 such as `dx/dt`",
                show(d, pool)
            ));
        }
        if !seen_derivs.insert(d) {
            return malformed(format!(
                "`{}` is listed as the derivative of two variables",
                show(d, pool)
            ));
        }
        if d == dae.time_var || d == dae.variables[i] {
            return malformed(format!(
                "`{}` cannot be the derivative of `{}`",
                show(d, pool),
                show(dae.variables[i], pool)
            ));
        }
    }
    // An algebraic variable's derivative is synthesised as `d{name}/dt`; if
    // that symbol is already one of the unknowns, the two would silently merge
    // into one jet chain.
    for &v in &dae.variables[dae.derivatives.len()..] {
        let synth = next_jet_symbol(v, pool);
        if seen_vars.contains(&synth) || seen_derivs.contains(&synth) {
            return malformed(format!(
                "`{}` has no listed derivative, and the name its derivative would get, \
                 `{}`, is already used; list its derivative explicitly",
                show(v, pool),
                show(synth, pool)
            ));
        }
    }
    Ok(())
}

/// `(variables, derivatives)` of equal length: a trailing variable with no
/// listed derivative (an algebraic one) gets the synthesised `d{name}/dt`.
///
/// Callers must have run [`validate_dae_shape`]; with more derivatives than
/// variables the surplus is dropped here rather than indexed out of bounds.
pub(crate) fn normalized_jets(dae: &DAE, pool: &ExprPool) -> (Vec<ExprId>, Vec<ExprId>) {
    let vars = dae.variables.clone();
    let derivs = vars
        .iter()
        .enumerate()
        .map(|(i, &v)| match dae.derivatives.get(i) {
            Some(&d) => d,
            None => next_jet_symbol(v, pool),
        })
        .collect();
    (vars, derivs)
}

/// One unknown function and its known jets `x, x', x'', …`.
struct Chain {
    jets: Vec<ExprId>,
}

impl Chain {
    fn jet(&mut self, k: usize, pool: &ExprPool) -> ExprId {
        while self.jets.len() <= k {
            let last = *self.jets.last().expect("a chain has its base jet");
            self.jets.push(next_jet_symbol(last, pool));
        }
        self.jets[k]
    }
}

/// Group `(vars, derivs)` into jet chains.  A variable that is another
/// variable's derivative is a jet of that chain; a cycle (`x' = y, y' = x`
/// *as declarations*) has no base and is refused.
fn build_chains(
    vars: &[ExprId],
    derivs: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<Chain>, DaeStructureError> {
    let var_pos: HashMap<ExprId, usize> = vars.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let is_some_derivative: HashSet<ExprId> = derivs.iter().copied().collect();
    let mut covered = vec![false; vars.len()];
    let mut chains = Vec::new();
    for &root in vars.iter().filter(|v| !is_some_derivative.contains(v)) {
        let mut jets = vec![root];
        let mut cur = root;
        while let Some(&p) = var_pos.get(&cur) {
            covered[p] = true;
            cur = derivs[p];
            jets.push(cur);
        }
        chains.push(Chain { jets });
    }
    if let Some(p) = covered.iter().position(|c| !c) {
        return Err(DaeStructureError::MalformedDae(format!(
            "the derivative declarations through `{}` form a cycle, so no variable in it \
             is an unknown function",
            show(vars[p], pool)
        )));
    }
    Ok(chains)
}

fn collect_symbols(expr: ExprId, pool: &ExprPool, out: &mut HashSet<ExprId>) {
    let children = pool.with(expr, |data| match data {
        ExprData::Symbol { .. } => {
            out.insert(expr);
            vec![]
        }
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::BigO(inner) => vec![*inner],
        _ => vec![],
    });
    for c in children {
        collect_symbols(c, pool, out);
    }
}

/// Signature matrix: `sig[i][j] = Some(k)` when the highest jet of chain `j`
/// in equation `i` is the `k`-th derivative.
fn signature_matrix(
    equations: &[ExprId],
    chains: &mut [Chain],
    pool: &ExprPool,
) -> Vec<Vec<Option<usize>>> {
    // Jets beyond the declared ones only appear if the caller wrote them with
    // the `d{name}/dt` naming rule; look a fixed distance past the end.
    let lookahead = MAX_DIFFERENTIATIONS + 2;
    let mut jet_of: HashMap<ExprId, (usize, usize)> = HashMap::new();
    for (j, ch) in chains.iter_mut().enumerate() {
        let depth = ch.jets.len() + lookahead;
        for k in 0..depth {
            jet_of.entry(ch.jet(k, pool)).or_insert((j, k));
        }
    }
    equations
        .iter()
        .map(|&eq| {
            let mut syms = HashSet::new();
            collect_symbols(eq, pool, &mut syms);
            let mut row = vec![None; chains.len()];
            for s in syms {
                if let Some(&(j, k)) = jet_of.get(&s) {
                    row[j] = Some(row[j].map_or(k, |old: usize| old.max(k)));
                }
            }
            row
        })
        .collect()
}

/// Maximum-weight perfect assignment of rows to columns (Hungarian algorithm,
/// minimising `-σ`).  `None` when no perfect transversal over finite entries
/// exists — the system is structurally singular.
fn highest_value_transversal(sig: &[Vec<Option<usize>>]) -> Option<Vec<usize>> {
    let n = sig.len();
    if n == 0 {
        return Some(vec![]);
    }
    let max_sig = sig.iter().flatten().filter_map(|s| *s).max().unwrap_or(0) as i64;
    // Any finite assignment costs at least -n·max_sig, so a forbidden entry
    // costing more than n·max_sig + 1 above it is never chosen when avoidable.
    let forbidden = (n as i64 + 1) * (max_sig + 1) * 4 + 1;
    let cost = |i: usize, j: usize| -> i64 {
        match sig[i][j] {
            Some(s) => -(s as i64),
            None => forbidden,
        }
    };
    // e-maxx formulation, 1-based with a virtual column 0.
    let inf = i64::MAX / 4;
    let mut u = vec![0i64; n + 1];
    let mut v = vec![0i64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut row_to_col = vec![0usize; n];
    for j in 1..=n {
        row_to_col[p[j] - 1] = j - 1;
    }
    if (0..n).any(|i| sig[i][row_to_col[i]].is_none()) {
        return None;
    }
    Some(row_to_col)
}

/// Smallest offsets `(c, d)` for the transversal `t` (Pryce's fixed-point
/// iteration).  `None` when some `c_i` would exceed [`MAX_DIFFERENTIATIONS`].
fn offsets(sig: &[Vec<Option<usize>>], t: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
    let n = sig.len();
    let mut c = vec![0usize; n];
    // Each pass raises some `c_i` or stops, and every `c_i` is capped, so this
    // bound is never the reason the loop ends on a well-formed input.
    for _ in 0..=(n + 1) * (MAX_DIFFERENTIATIONS + 1) {
        let mut d = vec![0usize; n];
        for (i, row) in sig.iter().enumerate() {
            for (j, s) in row.iter().enumerate() {
                if let Some(s) = s {
                    d[j] = d[j].max(s + c[i]);
                }
            }
        }
        let next: Vec<usize> = (0..n)
            .map(|i| d[t[i]] - sig[i][t[i]].expect("transversal entries are finite"))
            .collect();
        if next.iter().any(|&ci| ci > MAX_DIFFERENTIATIONS) {
            return None;
        }
        if next == c {
            return Some((c, d));
        }
        c = next;
    }
    None
}

/// Deterministic small rationals for the generic-point Jacobian test.
struct SamplePoints {
    state: u64,
}

impl SamplePoints {
    fn next(&mut self) -> (i64, i64) {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let a = (self.state >> 33) as i64;
        let num = 2 + a % 89;
        let den = 1 + (a / 89) % 13;
        let sign = if (a / 1157) % 2 == 0 { 1 } else { -1 };
        (sign * num, den)
    }
}

fn exact_det(mut m: Vec<Vec<Rational>>) -> Rational {
    let n = m.len();
    let mut det = Rational::from(1);
    for col in 0..n {
        let Some(piv) = (col..n).find(|&r| m[r][col] != 0) else {
            return Rational::new();
        };
        if piv != col {
            m.swap(piv, col);
            det = -det;
        }
        let p = m[col][col].clone();
        det *= &p;
        let (top, rest) = m.split_at_mut(col + 1);
        let pivot_row = &top[col];
        for row in rest {
            if row[col] == 0 {
                continue;
            }
            let f = Rational::from(&row[col] / &p);
            for (dst, src) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                *dst -= Rational::from(&f * src);
            }
        }
    }
    det
}

/// `true` when the f64 matrix is numerically non-singular (relative test).
fn float_nonsingular(mut m: Vec<Vec<f64>>) -> bool {
    let n = m.len();
    let scale = m
        .iter()
        .flatten()
        .fold(0.0f64, |a, &b| a.max(b.abs()))
        .max(f64::MIN_POSITIVE);
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
            .expect("non-empty range");
        if m[piv][col].abs() <= 1e-10 * scale {
            return false;
        }
        m.swap(piv, col);
        let (top, rest) = m.split_at_mut(col + 1);
        let pivot_row = &top[col];
        for row in rest {
            let f = row[col] / pivot_row[col];
            for (dst, src) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                *dst -= f * src;
            }
        }
    }
    true
}

/// Outcome of evaluating the system Jacobian at sample points.
enum JacobianVerdict {
    Nonsingular,
    /// Zero determinant at every point that could be evaluated.
    Singular,
    /// No sample point could be evaluated (poles, unsupported functions).
    Undecided,
}

fn jacobian_verdict(entries: &[Vec<ExprId>], pool: &ExprPool) -> JacobianVerdict {
    let n = entries.len();
    if n == 0 {
        return JacobianVerdict::Nonsingular;
    }
    let mut syms = HashSet::new();
    for &e in entries.iter().flatten() {
        collect_symbols(e, pool, &mut syms);
    }
    let mut syms: Vec<ExprId> = syms.into_iter().collect();
    syms.sort_by_key(|s| show(*s, pool));
    let mut pts = SamplePoints {
        state: 0x9E37_79B9_7F4A_7C15,
    };
    let mut evaluated = 0usize;
    for _attempt in 0..8 {
        if evaluated >= 4 {
            break;
        }
        let point: Vec<(i64, i64)> = syms.iter().map(|_| pts.next()).collect();
        let exact: HashMap<ExprId, Rational> = syms
            .iter()
            .zip(&point)
            .map(|(&s, &(a, b))| (s, Rational::from((a, b))))
            .collect();
        let exact_matrix: Option<Vec<Vec<Rational>>> = entries
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&e| crate::eval::eval_exact_rational(e, pool, &exact).ok())
                    .collect()
            })
            .collect();
        if let Some(m) = exact_matrix {
            evaluated += 1;
            if exact_det(m) != 0 {
                return JacobianVerdict::Nonsingular;
            }
            continue;
        }
        let floats: HashMap<ExprId, f64> = syms
            .iter()
            .zip(&point)
            .map(|(&s, &(a, b))| (s, a as f64 / b as f64))
            .collect();
        let float_matrix: Option<Vec<Vec<f64>>> = entries
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&e| crate::eval::eval_f64(e, pool, &floats).ok())
                    .collect()
            })
            .collect();
        if let Some(m) = float_matrix {
            evaluated += 1;
            if float_nonsingular(m) {
                return JacobianVerdict::Nonsingular;
            }
        }
    }
    if evaluated == 0 {
        JacobianVerdict::Undecided
    } else {
        JacobianVerdict::Singular
    }
}

/// The Σ-method index reduction.  See the module documentation.
pub(super) fn sigma_reduce(
    dae: &DAE,
    pool: &ExprPool,
) -> Result<super::PantelidesResult, DaeStructureError> {
    validate_dae_shape(dae, pool)?;
    let (vars, derivs) = normalized_jets(dae, pool);
    let mut chains = build_chains(&vars, &derivs, pool)?;
    let n_eq = dae.equations.len();
    if n_eq != chains.len() {
        return Err(DaeStructureError::Dae(DaeError::StructurallyInconsistent));
    }
    let sig = signature_matrix(&dae.equations, &mut chains, pool);
    let t = highest_value_transversal(&sig)
        .ok_or(DaeStructureError::Dae(DaeError::StructurallyInconsistent))?;
    let (c, d) = offsets(&sig, &t).ok_or(DaeStructureError::Dae(DaeError::IndexTooHigh))?;

    // Working state: the declared variables plus every jet below `d_j`, each
    // paired with its successor, so `differentiate_equation` applies the full
    // chain rule to every jet that can occur.
    let mut w_vars = vars.clone();
    let mut w_derivs = derivs.clone();
    for (j, ch) in chains.iter_mut().enumerate() {
        for k in 0..d[j] {
            let jet = ch.jet(k, pool);
            if !w_vars.contains(&jet) {
                let next = ch.jet(k + 1, pool);
                w_vars.push(jet);
                w_derivs.push(next);
            }
        }
    }

    // f_i, f_i', …, f_i^{(c_i)}, grouped by differentiation order.
    let mut current: Vec<ExprId> = dae.equations.clone();
    let mut equations = dae.equations.clone();
    let mut sigma = vec![0usize; n_eq];
    let mut steps = 0usize;
    let max_c = c.iter().copied().max().unwrap_or(0);
    for k in 1..=max_c {
        for i in 0..n_eq {
            if c[i] >= k {
                let de = differentiate_equation(current[i], &w_vars, &w_derivs, dae.time_var, pool)
                    .map_err(|e| DaeStructureError::Dae(DaeError::DiffError(e.to_string())))?;
                current[i] = de;
                equations.push(de);
                sigma.push(k);
                steps += 1;
            }
        }
    }

    // System Jacobian J_ij = ∂f_i^{(c_i)} / ∂x_j^{(d_j)}.
    let mut entries = Vec::with_capacity(n_eq);
    for &fi in &current {
        let mut row = Vec::with_capacity(n_eq);
        for (j, ch) in chains.iter_mut().enumerate() {
            let top = ch.jet(d[j], pool);
            let e = if structurally_depends(fi, top, pool) {
                crate::diff::diff(fi, top, pool)
                    .map_err(|e| DaeStructureError::Dae(DaeError::DiffError(e.to_string())))?
                    .value
            } else {
                pool.integer(0_i32)
            };
            row.push(e);
        }
        entries.push(row);
    }
    if let JacobianVerdict::Singular = jacobian_verdict(&entries, pool) {
        let tops: Vec<String> = chains
            .iter_mut()
            .enumerate()
            .map(|(j, ch)| show(ch.jet(d[j], pool), pool))
            .collect();
        return Err(DaeStructureError::SingularJacobian(format!(
            "after differentiating equation i c_i = {c:?} times, the Jacobian with respect \
             to the highest derivatives [{}] is singular at every sample point, so the \
             reduced system does not determine them; the structural index under-estimates \
             the true index",
            tops.join(", ")
        )));
    }

    let index = max_c + usize::from(d.contains(&0));
    let mut reduced = DAE::new(equations, w_vars, w_derivs, dae.time_var);
    reduced.index = Some(index);
    Ok(super::PantelidesResult {
        reduced_dae: reduced,
        differentiation_steps: steps,
        sigma,
    })
}
