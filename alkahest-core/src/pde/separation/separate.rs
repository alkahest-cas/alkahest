//! The separability test and the separated ODEs (plan §8 M4, item 1).
//!
//! **Product separation.**  Substitute `u = X(x)·T(t)·…` (fresh undefined
//! functions, one per variable) into the equation and divide by the product.
//! In the jet coordinates of the factors (`X, X', X'', T, T'`, one symbol
//! each, the M0 jet idea specialised to one variable per factor) the result
//! `E = L[XT]/(XT)` is a function of `x`, `t` and the jets.  A variable `v`
//! *splits off* when, for some multiplier `g`, every term of the expanded
//! `g·E` mentions either only `v` and `v`'s jets or none of them; then
//! `g·E = A(v) + B(rest)`, so `A = −σμ` and `B = σμ` for a separation constant
//! `μ`.  The multiplier is tried from `1` and the reciprocals of the
//! jet-free coefficients of `E`'s terms — `r²` for the polar Laplacian,
//! `r² sin²θ` for the spherical one — which is the classical recipe.  The
//! test recurses on `B = σμ`, so `n` variables give `n − 1` constants.
//!
//! **Additive separation** (Hamilton–Jacobi): `u = X(x) + T(t) + …` and the
//! same splitting without the division.
//!
//! The sign `σ` is chosen so that a separated second-order equation reads
//! `P[X] + μ w X = 0` with `w > 0` when the sign of `w` can be decided — the
//! Sturm–Liouville convention of [`sturm_liouville`](mod@super::super::sturm_liouville), so that
//! oscillatory directions get positive eigenvalues.
//!
//! The splitting is structural after expansion: it never claims a separation
//! that does not hold (each part is checked to be free of the other group's
//! symbols), but it can miss one hidden by an unexpanded form such as
//! `log(x t)`.

use std::collections::HashMap;

use crate::diff::undefined::{build, replace_function, view};
use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

use super::super::error::PdeError;
use super::super::jet::{decode_unknown, push_children, symbols_of};
use super::super::sturm_liouville::{sign, simp, Sgn};

/// Product (`u = X(x)T(t)`) or additive (`u = X(x) + T(t)`) separation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SeparationKind {
    /// `u = Π Xᵢ(xᵢ)`.
    Product,
    /// `u = Σ Xᵢ(xᵢ)` (Hamilton–Jacobi).
    Additive,
}

impl SeparationKind {
    /// `"product"` or `"additive"`.
    pub fn as_str(self) -> &'static str {
        match self {
            SeparationKind::Product => "product",
            SeparationKind::Additive => "additive",
        }
    }
}

/// One separated ordinary differential equation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SeparatedOde {
    /// The variable.
    pub var: ExprId,
    /// Its factor `X(x)` (an applied undefined function).
    pub factor: ExprId,
    /// The ODE, `= 0`, in `factor` and its `Derivative`s.
    pub equation: ExprId,
    /// The separation constant introduced when this variable was split off
    /// (`None` for the variable left last).
    pub constant: Option<ExprId>,
    /// The multiplier `g` the remaining equation was multiplied by before
    /// this split.
    pub multiplier: ExprId,
}

/// The result of [`separate`](fn@separate).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Separation {
    /// Product or additive.
    pub kind: SeparationKind,
    /// The unknown `u(x, t, …)`.
    pub unknown: ExprId,
    /// `X(x)·T(t)` or `X(x) + T(t)`.
    pub ansatz: ExprId,
    /// The factors, in the unknown's argument order.
    pub factors: Vec<ExprId>,
    /// The separated ODEs, in the order the variables were split off.
    pub equations: Vec<SeparatedOde>,
    /// The separation constants, in order.
    pub constants: Vec<ExprId>,
}

/// Options of [`separate_with`].
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct SeparateOptions {
    /// Variables to try first, in this order (others after them).
    pub prefer: Vec<ExprId>,
    /// A variable that must be split off last (the time or the
    /// non-eigen direction of a boundary-value problem).
    pub last: Option<ExprId>,
    /// Base name of the separation constants (`mu` → `mu1, mu2, …`).
    pub constant_name: Option<String>,
    /// An extra symbol to add as `+ κ u` before separating (the spectral
    /// parameter of a full eigenfunction expansion); it then appears in the
    /// last equation.
    pub spectral: Option<ExprId>,
}

impl SeparateOptions {
    /// Default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Try these variables first.
    pub fn prefer(mut self, vars: &[ExprId]) -> Self {
        self.prefer = vars.to_vec();
        self
    }
    /// Split this variable off last.
    pub fn last(mut self, v: ExprId) -> Self {
        self.last = Some(v);
        self
    }
}

pub(crate) fn sym_name(e: ExprId, pool: &ExprPool) -> Option<String> {
    pool.with(e, |d| match d {
        ExprData::Symbol { name, .. } => Some(name.clone()),
        _ => None,
    })
}

/// Names of every symbol and applied function in `e`.
pub(crate) fn names_in(e: ExprId, pool: &ExprPool, out: &mut Vec<String>) {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        match pool.get(n) {
            ExprData::Symbol { name, .. } => out.push(name),
            ExprData::Func { name, .. } => {
                out.push(name);
            }
            _ => {}
        }
        push_children(n, pool, &mut stack);
    }
}

fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => "F".into(),
    }
}

/// A fresh name `base`, `base1`, … not in `taken` (and pushed onto it).
pub(crate) fn fresh_name(base: &str, taken: &mut Vec<String>) -> String {
    let mut name = base.to_string();
    let mut k = 0;
    while taken.contains(&name) {
        k += 1;
        name = format!("{base}{k}");
    }
    taken.push(name.clone());
    name
}

/// Jet symbols of one factor `F(v)`: `F, F', F'', …` up to `order`.
struct FactorJets {
    var: ExprId,
    factor: ExprId,
    syms: Vec<ExprId>,
}

impl FactorJets {
    fn nodes(&self, pool: &ExprPool) -> Vec<ExprId> {
        (0..self.syms.len() as u32)
            .map(|k| build(self.factor, &[k], pool))
            .collect()
    }
}

/// Replace every `Derivative(F(v), k)` (and `F(v)`) by its jet symbol.
fn to_jets(e: ExprId, jets: &[FactorJets], pool: &ExprPool) -> Result<ExprId, PdeError> {
    let mut map: HashMap<ExprId, ExprId> = HashMap::new();
    for fj in jets {
        for (node, &s) in fj.nodes(pool).into_iter().zip(&fj.syms) {
            map.insert(node, s);
        }
    }
    let out = replace_nodes(e, &map, pool);
    // Anything of a factor left over is a derivative above the jet order.
    let mut stack = vec![out];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if jets
                .iter()
                .any(|fj| view(fj.factor, pool).is_some_and(|w| w.name == v.name))
            {
                return Err(PdeError::SeparationUnsupported(format!(
                    "{} occurs at an argument other than its variable",
                    pool.display(n)
                )));
            }
        }
        push_children(n, pool, &mut stack);
    }
    Ok(out)
}

fn from_jets(e: ExprId, jets: &[FactorJets], pool: &ExprPool) -> ExprId {
    let mut map: HashMap<ExprId, ExprId> = HashMap::new();
    for fj in jets {
        for (node, &s) in fj.nodes(pool).into_iter().zip(&fj.syms) {
            map.insert(s, node);
        }
    }
    subs(e, &map, pool)
}

/// Structural replacement of whole nodes (memoised).
pub(crate) fn replace_nodes(e: ExprId, map: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> ExprId {
    fn go(
        e: ExprId,
        map: &HashMap<ExprId, ExprId>,
        pool: &ExprPool,
        memo: &mut HashMap<ExprId, ExprId>,
    ) -> ExprId {
        if let Some(&r) = map.get(&e) {
            return r;
        }
        if let Some(&r) = memo.get(&e) {
            return r;
        }
        let r = match pool.get(e) {
            ExprData::Add(xs) => pool.add(xs.iter().map(|&x| go(x, map, pool, memo)).collect()),
            ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| go(x, map, pool, memo)).collect()),
            ExprData::Pow { base, exp } => {
                pool.pow(go(base, map, pool, memo), go(exp, map, pool, memo))
            }
            ExprData::Func { name, args } => {
                pool.func(name, args.iter().map(|&x| go(x, map, pool, memo)).collect())
            }
            _ => e,
        };
        memo.insert(e, r);
        r
    }
    go(e, map, pool, &mut HashMap::new())
}

fn terms_of(e: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(e) {
        ExprData::Add(xs) => xs,
        _ => vec![e],
    }
}

fn factors_of(e: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(e) {
        ExprData::Mul(xs) => xs,
        _ => vec![e],
    }
}

fn mentions_any(e: ExprId, syms: &[ExprId], pool: &ExprPool) -> bool {
    let s = symbols_of(e, pool);
    syms.iter().any(|x| s.contains(x))
}

/// Expand fully (products over sums, powers of sums), so that a separable
/// expression has only pure terms.
pub(crate) fn expanded(e: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::simplify_expanded(e, pool).value
}

/// `g·e = A + B` with `A` mentioning only `a_syms` (among `a_syms ∪
/// b_syms`) and `B` only `b_syms`.  `None` when some term mixes them.
fn split_with(
    e: ExprId,
    g: ExprId,
    a_syms: &[ExprId],
    b_syms: &[ExprId],
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let ge = expanded(pool.mul(vec![g, e]), pool);
    let mut a = Vec::new();
    let mut b = Vec::new();
    for t in terms_of(ge, pool) {
        let ma = mentions_any(t, a_syms, pool);
        let mb = mentions_any(t, b_syms, pool);
        match (ma, mb) {
            (true, true) => return None,
            (true, false) => a.push(t),
            _ => b.push(t),
        }
    }
    if a.is_empty() {
        return None;
    }
    Some((simp(pool.add(a), pool), simp(pool.add(b), pool)))
}

/// Candidate multipliers: `1`, then `1/c` for the jet-free coefficient `c`
/// of every term of `e` that depends on a variable.
fn multipliers(e: ExprId, jet_syms: &[ExprId], vars: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    let mut out = vec![pool.integer(1)];
    for t in terms_of(expanded(e, pool), pool) {
        let coef: Vec<ExprId> = factors_of(t, pool)
            .into_iter()
            .filter(|&f| !mentions_any(f, jet_syms, pool) && mentions_any(f, vars, pool))
            .collect();
        if coef.is_empty() {
            continue;
        }
        let c = simp(pool.pow(pool.mul(coef), pool.integer(-1)), pool);
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// Separate `equation` (`= 0`) in `unknown` with default options.
///
/// # Errors
///
/// See [`separate_with`].
pub fn separate(
    equation: ExprId,
    unknown: ExprId,
    kind: SeparationKind,
    pool: &ExprPool,
) -> Result<Separation, PdeError> {
    separate_with(equation, unknown, kind, &SeparateOptions::default(), pool)
}

/// Whether `equation` separates (product or additive) in every argument of
/// `unknown`.
pub fn is_separable(
    equation: ExprId,
    unknown: ExprId,
    kind: SeparationKind,
    pool: &ExprPool,
) -> bool {
    separate(equation, unknown, kind, pool).is_ok()
}

/// Separate `equation` (`= 0`) in `unknown`: see the [module docs](self).
///
/// # Errors
///
/// * `E-PDE-001` for a malformed unknown;
/// * `E-PDE-200` when no variable splits off (with any multiplier tried);
/// * `E-PDE-201` for an unknown of one variable, or a factor evaluated off
///   its variable.
pub fn separate_with(
    equation: ExprId,
    unknown: ExprId,
    kind: SeparationKind,
    opts: &SeparateOptions,
    pool: &ExprPool,
) -> Result<Separation, PdeError> {
    let uv = decode_unknown(unknown, pool)?;
    let vars = uv.args.clone();
    if vars.len() < 2 {
        return Err(PdeError::SeparationUnsupported(format!(
            "{} has one variable: nothing to separate",
            pool.display(unknown)
        )));
    }
    let mut taken = Vec::new();
    names_in(equation, pool, &mut taken);
    if let Some(k) = opts.spectral {
        names_in(k, pool, &mut taken);
    }
    taken.push(uv.name.clone());
    // Order of the equation (bounds the jets per factor).
    let order = equation_order(equation, &uv.name, pool).max(1);
    let mut jets = Vec::new();
    let mut factors = Vec::new();
    for &v in &vars {
        let base = capitalised(&sym_name(v, pool).unwrap_or_else(|| "F".into()));
        let fname = fresh_name(&base, &mut taken);
        let f = pool.func(fname.clone(), vec![v]);
        let syms: Vec<ExprId> = (0..=order)
            .map(|k| {
                let n = fresh_name(&format!("__sep_{fname}_{k}"), &mut taken);
                pool.symbol(n, Domain::Real)
            })
            .collect();
        jets.push(FactorJets {
            var: v,
            factor: f,
            syms,
        });
        factors.push(f);
    }
    let ansatz = match kind {
        SeparationKind::Product => pool.mul(factors.clone()),
        SeparationKind::Additive => pool.add(factors.clone()),
    };
    let mut eq = equation;
    if let Some(k) = opts.spectral {
        eq = pool.add(vec![eq, pool.mul(vec![k, unknown])]);
    }
    let substituted = replace_function(eq, unknown, ansatz, pool).map_err(|e| {
        PdeError::SeparationUnsupported(format!("cannot substitute the ansatz: {e}"))
    })?;
    let mut e = to_jets(substituted, &jets, pool)?;
    if kind == SeparationKind::Product {
        let prod: Vec<ExprId> = jets.iter().map(|fj| fj.syms[0]).collect();
        e = pool.mul(vec![e, pool.pow(pool.mul(prod), pool.integer(-1))]);
    }
    e = simp(e, pool);

    let base = opts.constant_name.clone().unwrap_or_else(|| "mu".into());
    let mut remaining: Vec<usize> = (0..vars.len()).collect();
    let mut equations = Vec::new();
    let mut constants = Vec::new();
    let all_jets: Vec<ExprId> = jets.iter().flat_map(|fj| fj.syms.clone()).collect();
    while remaining.len() > 1 {
        // Order of candidates: preferred first, `last` never.
        let mut cands: Vec<usize> = Vec::new();
        for p in &opts.prefer {
            if let Some(i) = remaining.iter().copied().find(|&i| vars[i] == *p) {
                cands.push(i);
            }
        }
        for &i in &remaining {
            if !cands.contains(&i) {
                cands.push(i);
            }
        }
        cands.retain(|&i| Some(vars[i]) != opts.last);
        let rem_vars: Vec<ExprId> = remaining.iter().map(|&i| vars[i]).collect();
        let mut done = None;
        'outer: for &i in &cands {
            let a_syms: Vec<ExprId> = std::iter::once(vars[i])
                .chain(jets[i].syms.iter().copied())
                .collect();
            let b_syms: Vec<ExprId> = remaining
                .iter()
                .filter(|&&j| j != i)
                .flat_map(|&j| std::iter::once(vars[j]).chain(jets[j].syms.iter().copied()))
                .collect();
            for g in multipliers(e, &all_jets, &rem_vars, pool) {
                if let Some((a, b)) = split_with(e, g, &a_syms, &b_syms, pool) {
                    // The part of variable i must actually involve its jets.
                    if !mentions_any(a, &jets[i].syms, pool) {
                        continue;
                    }
                    done = Some((i, g, a, b));
                    break 'outer;
                }
            }
        }
        let Some((i, g, a, b)) = done else {
            return Err(PdeError::SeparationNotSeparable(format!(
                "{} = 0 does not separate in {} (no variable splits off with the multipliers \
                 tried)",
                pool.display(equation),
                rem_vars
                    .iter()
                    .map(|&v| pool.display(v).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        };
        let mu_name = fresh_name(&format!("{base}{}", constants.len() + 1), &mut taken);
        let mu = pool.symbol(mu_name, Domain::Real);
        let fj = &jets[i];
        // Normalise: A/a₂ when the leading coefficient a₂ is free of v.
        let (a_norm, scale) = if kind == SeparationKind::Product {
            normalise(a, fj, pool)
        } else {
            (a, pool.integer(1))
        };
        // A/s + σμ = 0 ;  B/s − σμ = 0.
        let mut sigma = 1i64;
        let ode_jets = |sg: i64| -> ExprId {
            let core = pool.add(vec![a_norm, pool.mul(vec![pool.integer(sg), mu])]);
            match kind {
                SeparationKind::Product => pool.mul(vec![core, fj.syms[0]]),
                SeparationKind::Additive => core,
            }
        };
        if kind == SeparationKind::Product {
            let o = expanded(ode_jets(1), pool);
            let lead = fj
                .syms
                .last()
                .and_then(|&l| coefficient_of(o, l, pool))
                .or_else(|| highest_coefficient(o, &fj.syms, pool));
            if let Some(lead) = lead {
                let wcoef = simp(
                    crate::diff::diff(o, mu, pool)
                        .map(|r| r.value)
                        .unwrap_or(mu),
                    pool,
                );
                let wcoef = simp(
                    pool.mul(vec![wcoef, pool.pow(fj.syms[0], pool.integer(-1))]),
                    pool,
                );
                let ratio = simp(
                    pool.mul(vec![wcoef, pool.pow(lead, pool.integer(-1))]),
                    pool,
                );
                if sign(ratio, pool) == Sgn::Neg {
                    sigma = -1;
                }
            }
        }
        let ode = simp(
            from_jets(expanded(ode_jets(sigma), pool), &jets, pool),
            pool,
        );
        equations.push(SeparatedOde {
            var: fj.var,
            factor: fj.factor,
            equation: ode,
            constant: Some(mu),
            multiplier: g,
        });
        constants.push(mu);
        // Rest: B/scale − σμ.
        let b_scaled = pool.mul(vec![b, pool.pow(scale, pool.integer(-1))]);
        e = simp(
            pool.add(vec![b_scaled, pool.mul(vec![pool.integer(-sigma), mu])]),
            pool,
        );
        remaining.retain(|&j| j != i);
    }
    let last = remaining[0];
    let fj = &jets[last];
    if !mentions_any(e, &fj.syms, pool) {
        return Err(PdeError::SeparationNotSeparable(format!(
            "after separating, no derivative of {} is left: the equation does not involve {}",
            pool.display(fj.factor),
            pool.display(fj.var)
        )));
    }
    let mut ode_j = match kind {
        SeparationKind::Product => pool.mul(vec![e, fj.syms[0]]),
        SeparationKind::Additive => e,
    };
    // Normalise the last equation by its leading coefficient when that is
    // a nonzero expression free of the variable (`T' + kμT`, not `−T'/k − μT`).
    if let Some(lead) = highest_coefficient(expanded(ode_j, pool), &fj.syms, pool) {
        if !mentions_var(lead, fj.var, pool) && !mentions_any(lead, &fj.syms, pool) {
            ode_j = pool.mul(vec![ode_j, pool.pow(lead, pool.integer(-1))]);
        }
    }
    let ode = simp(from_jets(expanded(ode_j, pool), &jets, pool), pool);
    equations.push(SeparatedOde {
        var: fj.var,
        factor: fj.factor,
        equation: ode,
        constant: None,
        multiplier: pool.integer(1),
    });
    Ok(Separation {
        kind,
        unknown,
        ansatz,
        factors,
        equations,
        constants,
    })
}

/// `A / a₂` and `a₂`, where `a₂` is the coefficient of the highest jet of the
/// factor in `A` (for product separation `A` is in `X^{(k)}/X` form), when it
/// is free of the factor's variable and nonzero; otherwise `(A, 1)`.
fn normalise(a: ExprId, fj: &FactorJets, pool: &ExprPool) -> (ExprId, ExprId) {
    let one = pool.integer(1);
    let Some(lead) = highest_coefficient(a, &fj.syms, pool) else {
        return (a, one);
    };
    // coefficient of X^{(k)}/X: strip the 1/X factor.
    let lead = simp(pool.mul(vec![lead, fj.syms[0]]), pool);
    if mentions_var(lead, fj.var, pool) || mentions_any(lead, &fj.syms, pool) {
        return (a, one);
    }
    if lead == pool.integer(0) {
        return (a, one);
    }
    (
        simp(pool.mul(vec![a, pool.pow(lead, pool.integer(-1))]), pool),
        lead,
    )
}

/// The coefficient of `s` in the expanded `e` (`∂e/∂s` when `e` is linear in
/// it); `None` when `e` does not mention `s`.
fn coefficient_of(e: ExprId, s: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if !mentions_var(e, s, pool) {
        return None;
    }
    let c = crate::diff::diff(e, s, pool).ok()?.value;
    let c = simp(c, pool);
    (c != pool.integer(0)).then_some(c)
}

/// The coefficient of the highest jet symbol that `e` mentions.
fn highest_coefficient(e: ExprId, syms: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    for &s in syms.iter().skip(1).rev() {
        if let Some(c) = coefficient_of(e, s, pool) {
            return Some(c);
        }
    }
    None
}

/// The highest total derivative order of `name` in `e`.
pub(crate) fn equation_order(e: ExprId, name: &str, pool: &ExprPool) -> u32 {
    let mut best = 0;
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            if v.name == name {
                best = best.max(v.orders.iter().sum());
                continue;
            }
        }
        push_children(n, pool, &mut stack);
    }
    best
}
