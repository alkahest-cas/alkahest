//! Linear overdetermined PDE systems for unknown functions — the
//! *determining systems* of Lie point symmetries and of conservation-law
//! multipliers (milestone M6).
//!
//! The unknowns are functions `f_k(z_{i₁}, …)` of a subset of the coordinate
//! symbols `z`; an equation is a linear combination `Σ c(z)·∂^α f_k = 0` with
//! coefficients that are expressions in `z` and in symbolic parameters.
//!
//! The solver is the RIF-style heuristic loop of Reid–Wittkopf–Boulton and of
//! CRACK / LIEPDE, in this order of preference:
//!
//! 1. **splitting** — an equation whose unknowns do not depend on `z_j` but
//!    whose coefficients do is split by the monomials in `z_j` (and in
//!    `z_j`-dependent transcendental generators such as `sin z_j`, `e^{z_j}`,
//!    `z_j^m`), which is an equivalence when those generators are known to be
//!    linearly independent and is recorded as a loss of completeness otherwise;
//! 2. **trivial equations** `∂_{z_j}^r f = 0` are integrated:
//!    `f = Σ_{l<r} z_j^l g_l(…)`;
//! 3. **elimination** of an unknown that occurs undifferentiated once;
//! 4. **integration** of `∂_{z_j}^r f = R` when `R` is integrable in `z_j`;
//! 5. a **linear ODE** in a single function of one variable goes to `dsolve`;
//! 6. mixed trivial equations `∂^α f = 0`.
//!
//! A coefficient that depends only on parameters and may vanish forces a
//! **case split** (`c ≠ 0` / each root of `c`) — the group-classification
//! mechanism.  What cannot be resolved is returned as residual equations,
//! never dropped.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::poly::multipoly::MultiPoly;

use super::jet::push_children;
use super::solve::util::{contains, d, is_number, is_zero, isolate, simp};

/// A multi-index over [`DetSys::vars`].
pub(crate) type Alpha = Vec<u32>;
/// `(function index, multi-index)`: the derivative `∂^α f_k`.
pub(crate) type Atom = (usize, Alpha);
/// A linear combination of derivative atoms with expression coefficients.
pub(crate) type Lin = BTreeMap<Atom, ExprId>;

/// An unknown function of the system.
#[derive(Clone, Debug)]
pub(crate) struct Func {
    pub name: String,
    /// Indices into [`DetSys::vars`], ascending.
    pub deps: Vec<usize>,
    pub dead: bool,
}

/// A determining system together with the expressions ("targets") of the
/// original unknowns in terms of the current functions.
#[derive(Clone, Debug)]
pub(crate) struct DetSys {
    pub vars: Vec<ExprId>,
    pub params: Vec<ExprId>,
    pub funcs: Vec<Func>,
    pub eqs: Vec<Lin>,
    pub targets: Vec<Lin>,
    pub complete: bool,
    pub notes: Vec<String>,
    /// Parameter factors assumed non-zero on this branch.
    pub nonzero: Vec<ExprId>,
    /// `param = value` choices made on this branch.
    pub case: Vec<(ExprId, ExprId)>,
    /// Special parameter values at which a generic split would change
    /// (`z^m` colliding with `z^k`): `(param expression, value)`.
    pub special: Vec<(ExprId, ExprId)>,
    pub steps: Vec<String>,
    /// The residual equations form a completed (passive) system: every
    /// integrability condition reduces to zero.
    pub passive: bool,
    normalized: Vec<bool>,
}

enum Nz {
    Yes,
    Split(ExprId),
}

enum Step {
    Done,
    None,
    Branch(ExprId),
}

/// Budget of the solver loop (steps per branch, branches overall).
const MAX_STEPS: usize = 600;
const MAX_BRANCHES: usize = 16;
const MAX_TERMS: usize = 4000;

/// `(k, α)` from an atom symbol `__dA{k}_{α…}`.
pub(crate) fn parse_atom_sym(sym: ExprId, pool: &ExprPool) -> Option<Atom> {
    let name = pool.with(sym, |d| match d {
        ExprData::Symbol { name, .. } => Some(name.clone()),
        _ => None,
    })?;
    let rest = name.strip_prefix("__dA")?;
    let mut it = rest.split('_');
    let k: usize = it.next()?.parse().ok()?;
    let alpha: Option<Vec<u32>> = it.map(|x| x.parse().ok()).collect();
    Some((k, alpha?))
}

/// When `e` holds fractional powers `z^(p/q)`, the expression with
/// `z = w^L` (`L` the lcm of the denominators) and the fresh symbol `w`.
fn root_variable(e: ExprId, z: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let mut l: u64 = 1;
    let mut stack = vec![e];
    let mut seen = HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Pow { base, exp } = pool.get(n) {
            if base == z {
                if let Some(q) = super::solve::util::as_rational(exp, pool) {
                    let den = q.denom().to_u64().unwrap_or(1);
                    l = num_lcm(l, den);
                }
            }
        }
        push_children(n, pool, &mut stack);
    }
    if l == 1 || l > 64 {
        return None;
    }
    let w = pool.symbol(format!("__root_{}", z.0), Domain::Real);
    Some((rewrite_root(e, z, w, l as i64, pool), w))
}

fn num_lcm(a: u64, b: u64) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    a / gcd(a, b) * b
}

fn rewrite_root(e: ExprId, z: ExprId, w: ExprId, l: i64, pool: &ExprPool) -> ExprId {
    if e == z {
        return pool.pow(w, pool.integer(l));
    }
    match pool.get(e) {
        ExprData::Pow { base, exp } if base == z => {
            if let Some(q) = super::solve::util::as_rational(exp, pool) {
                let k: rug::Rational = &q * rug::Rational::from(l);
                if *k.denom() == 1 {
                    return pool.pow(w, pool.integer(k.numer().clone()));
                }
            }
            pool.pow(pool.pow(w, pool.integer(l)), exp)
        }
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| rewrite_root(x, z, w, l, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| rewrite_root(x, z, w, l, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(
            rewrite_root(base, z, w, l, pool),
            rewrite_root(exp, z, w, l, pool),
        ),
        ExprData::Func { name, args } => pool.func(
            name,
            args.iter()
                .map(|&a| rewrite_root(a, z, w, l, pool))
                .collect(),
        ),
        _ => e,
    }
}

pub(crate) fn atom_sym(a: &Atom, pool: &ExprPool) -> ExprId {
    let idx: Vec<String> = a.1.iter().map(|x| x.to_string()).collect();
    pool.symbol(format!("__dA{}_{}", a.0, idx.join("_")), Domain::Real)
}

/// `Σ c·A` with the atoms as symbols, and the symbol ↔ atom list.
pub(crate) fn lin_to_symbolic(l: &Lin, pool: &ExprPool) -> (ExprId, Vec<(ExprId, Atom)>) {
    let mut terms = Vec::new();
    let mut atoms = Vec::new();
    for (a, &c) in l {
        let s = atom_sym(a, pool);
        atoms.push((s, a.clone()));
        terms.push(pool.mul(vec![c, s]));
    }
    let e = match terms.len() {
        0 => pool.integer(0),
        1 => terms[0],
        _ => pool.add(terms),
    };
    (e, atoms)
}

/// Rewrite `z^(q + s)` (numeric `q`, symbolic `s`) as `z^q · z^s` and
/// `z^(k·s)` as `(z^s)^k`, so a polynomial view sees one generator `z^s`.
pub(crate) fn canon_powers(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| canon_powers(x, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| canon_powers(x, pool)).collect()),
        ExprData::Pow { base, exp } => {
            let b = canon_powers(base, pool);
            let is_sym = pool.with(b, |d| matches!(d, ExprData::Symbol { .. }));
            if !is_sym || is_number(exp, pool) {
                return pool.pow(b, canon_powers(exp, pool));
            }
            let terms = match pool.get(exp) {
                ExprData::Add(xs) => xs,
                _ => vec![exp],
            };
            let mut q = Vec::new();
            let mut s = Vec::new();
            for t in terms {
                if is_number(t, pool) {
                    q.push(t);
                } else {
                    s.push(t);
                }
            }
            let s_e = if s.len() == 1 { s[0] } else { pool.add(s) };
            // z^(k·s0) → (z^s0)^k for an integer k.
            let sym_part = match pool.get(s_e) {
                ExprData::Mul(xs) if xs.len() == 2 && is_integer(xs[0], pool) => {
                    pool.pow(pool.pow(b, xs[1]), xs[0])
                }
                _ => pool.pow(b, s_e),
            };
            if q.is_empty() {
                sym_part
            } else {
                let qn = simp(pool.add(q), pool);
                pool.mul(vec![pool.pow(b, qn), sym_part])
            }
        }
        ExprData::Func { name, args } => {
            pool.func(name, args.iter().map(|&a| canon_powers(a, pool)).collect())
        }
        _ => e,
    }
}

fn is_integer(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Integer(_)))
}

fn mp_expr(vars: &[ExprId], terms: BTreeMap<Vec<u32>, rug::Integer>, pool: &ExprPool) -> ExprId {
    let mut t = BTreeMap::new();
    for (mut k, c) in terms {
        while k.last() == Some(&0) {
            k.pop();
        }
        if c != 0 {
            t.insert(k, c);
        }
    }
    MultiPoly {
        vars: vars.to_vec(),
        terms: t,
    }
    .to_expr(pool)
}

/// Integer gcd of all coefficients across several term maps, with the sign
/// of the first leading coefficient.
fn content(maps: &[&BTreeMap<Vec<u32>, rug::Integer>]) -> rug::Integer {
    let mut g = rug::Integer::from(0);
    let mut sign: Option<bool> = None;
    for m in maps {
        for c in m.values() {
            if sign.is_none() && *c != 0 {
                sign = Some(*c < 0);
            }
            g = g.gcd(c);
        }
    }
    if g == 0 {
        g = rug::Integer::from(1);
    }
    if sign == Some(true) {
        g = -g;
    }
    g
}

/// Whether a generator is "known transcendental" for independence purposes,
/// and its family name.
fn family(g: ExprId, z: ExprId, params: &[ExprId], pool: &ExprPool) -> Option<String> {
    match pool.get(g) {
        ExprData::Func { name, args } if args.len() == 1 => match name.as_str() {
            "exp" => Some("exp".into()),
            "log" if args[0] == z => Some("log".into()),
            "sin" | "cos" => Some(format!("trig:{}", args[0].0)),
            "sinh" | "cosh" => Some(format!("htrig:{}", args[0].0)),
            _ => None,
        },
        ExprData::Pow { base, exp } if base == z => {
            let exp_params = super::jet::symbols_of(exp, pool)
                .into_iter()
                .all(|s| params.contains(&s));
            if !is_number(exp, pool) && exp_params {
                Some(format!("pow:{}", exp.0))
            } else {
                None
            }
        }
        _ => None,
    }
}

impl DetSys {
    pub(crate) fn new(vars: Vec<ExprId>, params: Vec<ExprId>) -> Self {
        DetSys {
            vars,
            params,
            funcs: Vec::new(),
            eqs: Vec::new(),
            targets: Vec::new(),
            complete: true,
            notes: Vec::new(),
            nonzero: Vec::new(),
            case: Vec::new(),
            special: Vec::new(),
            steps: Vec::new(),
            passive: true,
            normalized: Vec::new(),
        }
    }

    pub(crate) fn add_func(&mut self, name: impl Into<String>, deps: Vec<usize>) -> usize {
        self.funcs.push(Func {
            name: name.into(),
            deps,
            dead: false,
        });
        self.funcs.len() - 1
    }

    fn fresh_func(&mut self, base: usize, deps: Vec<usize>) -> usize {
        let name = format!("{}_{}", self.funcs[base].name, self.funcs.len());
        self.add_func(name, deps)
    }

    pub(crate) fn push_eq(&mut self, eq: Lin) {
        self.eqs.push(eq);
        self.normalized.push(false);
    }

    pub(crate) fn zero_alpha(&self) -> Alpha {
        vec![0; self.vars.len()]
    }

    fn note(&mut self, s: impl Into<String>) {
        let s = s.into();
        if !self.notes.contains(&s) {
            self.notes.push(s);
        }
    }

    // -----------------------------------------------------------------
    // Linear-combination algebra
    // -----------------------------------------------------------------

    pub(crate) fn lin_add(l: &mut Lin, a: Atom, c: ExprId, pool: &ExprPool) {
        match l.get(&a) {
            Some(&old) => {
                l.insert(a, pool.add(vec![old, c]));
            }
            None => {
                l.insert(a, c);
            }
        }
    }

    /// `∂_{z_j} L`.
    pub(crate) fn dlin(&self, l: &Lin, j: usize, pool: &ExprPool) -> Lin {
        let z = self.vars[j];
        let mut out = Lin::new();
        for (a, &c) in l {
            if contains(c, z, pool) {
                if let Some(dc) = d(c, z, pool) {
                    if dc != pool.integer(0) {
                        Self::lin_add(&mut out, a.clone(), dc, pool);
                    }
                }
            }
            if self.funcs[a.0].deps.contains(&j) {
                let mut b = a.1.clone();
                b[j] += 1;
                Self::lin_add(&mut out, (a.0, b), c, pool);
            }
        }
        Self::simp_lin(out, pool)
    }

    fn simp_lin(l: Lin, pool: &ExprPool) -> Lin {
        let zero = pool.integer(0);
        l.into_iter()
            .map(|(a, c)| (a, simp(c, pool)))
            .filter(|(_, c)| *c != zero)
            .collect()
    }

    fn dlin_multi(&self, l: &Lin, alpha: &[u32], pool: &ExprPool) -> Lin {
        let mut cur = l.clone();
        for (j, &k) in alpha.iter().enumerate() {
            for _ in 0..k {
                cur = self.dlin(&cur, j, pool);
            }
        }
        cur
    }

    /// The variables a combination depends on (coefficients and functions).
    fn lin_deps(&self, l: &Lin, pool: &ExprPool) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        for (a, &c) in l {
            out.extend(self.funcs[a.0].deps.iter().copied());
            for (j, &z) in self.vars.iter().enumerate() {
                if contains(c, z, pool) {
                    out.insert(j);
                }
            }
        }
        out
    }

    /// Replace `f_k` by `l` everywhere (equations and targets); adds the
    /// constraints `∂_{z_j} l = 0` for variables `f_k` does not depend on.
    fn substitute(&mut self, k: usize, l: &Lin, pool: &ExprPool) {
        let deps_l = self.lin_deps(l, pool);
        let extra: Vec<usize> = deps_l
            .iter()
            .copied()
            .filter(|j| !self.funcs[k].deps.contains(j))
            .collect();
        let mut memo: HashMap<Alpha, Lin> = HashMap::new();
        let apply = |sys: &DetSys, e: &Lin, memo: &mut HashMap<Alpha, Lin>| -> Option<Lin> {
            if !e.keys().any(|a| a.0 == k) {
                return None;
            }
            let mut out = Lin::new();
            for (a, &c) in e {
                if a.0 != k {
                    Self::lin_add(&mut out, a.clone(), c, pool);
                    continue;
                }
                let dl = memo
                    .entry(a.1.clone())
                    .or_insert_with(|| sys.dlin_multi(l, &a.1, pool))
                    .clone();
                for (b, cb) in dl {
                    Self::lin_add(&mut out, b, pool.mul(vec![c, cb]), pool);
                }
            }
            Some(Self::simp_lin(out, pool))
        };
        for i in 0..self.eqs.len() {
            if let Some(n) = apply(self, &self.eqs[i], &mut memo) {
                self.eqs[i] = n;
                self.normalized[i] = false;
            }
        }
        for i in 0..self.targets.len() {
            if let Some(n) = apply(self, &self.targets[i], &mut memo) {
                self.targets[i] = n;
            }
        }
        self.funcs[k].dead = true;
        for j in extra {
            let c = self.dlin(l, j, pool);
            if !c.is_empty() {
                self.push_eq(c);
            }
        }
    }

    // -----------------------------------------------------------------
    // Normalisation
    // -----------------------------------------------------------------

    /// Clear denominators, remove the integer content, canonical sign; `None`
    /// for an identically zero equation.
    fn normalize(&self, eq: &Lin, pool: &ExprPool) -> Option<Lin> {
        if eq.is_empty() {
            return None;
        }
        let (expr, atoms) = lin_to_symbolic(eq, pool);
        let expr = canon_powers(expr, pool);
        let syms: Vec<ExprId> = atoms.iter().map(|a| a.0).collect();
        let m = syms.len();
        let fallback = || {
            let l = Self::simp_lin(eq.clone(), pool);
            let l: Lin = l.into_iter().filter(|(_, c)| !is_zero(*c, pool)).collect();
            if l.is_empty() {
                None
            } else {
                Some(l)
            }
        };
        let Ok((rf, gens)) = crate::poly::cancel::rational_over_generators(expr, syms, pool) else {
            return fallback();
        };
        if rf.numer.terms.len() > MAX_TERMS {
            return fallback();
        }
        let rest: Vec<ExprId> = gens[m..].to_vec();
        let mut per: Vec<BTreeMap<Vec<u32>, rug::Integer>> = vec![BTreeMap::new(); m];
        for (exps, c) in &rf.numer.terms {
            let get = |i: usize| exps.get(i).copied().unwrap_or(0);
            let hit: Vec<usize> = (0..m).filter(|&i| get(i) > 0).collect();
            if hit.len() != 1 || get(hit[0]) != 1 {
                return fallback();
            }
            let key: Vec<u32> = (m..gens.len()).map(get).collect();
            per[hit[0]].insert(key, c.clone());
        }
        let refs: Vec<&BTreeMap<Vec<u32>, rug::Integer>> = per.iter().collect();
        let g = content(&refs);
        let mut out = Lin::new();
        for (i, terms) in per.into_iter().enumerate() {
            if terms.is_empty() {
                continue;
            }
            let terms = terms
                .into_iter()
                .map(|(k, c)| (k, rug::Integer::from(c.div_exact_ref(&g))))
                .collect();
            out.insert(atoms[i].1.clone(), mp_expr(&rest, terms, pool));
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    fn cleanup(&mut self, pool: &ExprPool) {
        let mut eqs = Vec::new();
        let mut norm = Vec::new();
        let mut seen: HashSet<Vec<(Atom, ExprId)>> = HashSet::new();
        let old = std::mem::take(&mut self.eqs);
        let oldn = std::mem::take(&mut self.normalized);
        for (e, done) in old.into_iter().zip(oldn) {
            if !done && crate::budget::tripped().is_some() {
                // Out of budget: keep the equation as it is (`run` stops at
                // its next poll and marks the system incomplete).
                eqs.push(e);
                norm.push(false);
                continue;
            }
            let n = if done {
                Some(e)
            } else {
                self.normalize(&e, pool)
            };
            if let Some(n) = n {
                let key: Vec<(Atom, ExprId)> = n.iter().map(|(a, &c)| (a.clone(), c)).collect();
                if seen.insert(key) {
                    eqs.push(n);
                    norm.push(true);
                }
            }
        }
        self.eqs = eqs;
        self.normalized = norm;
        for t in self.targets.iter_mut() {
            let l = std::mem::take(t);
            *t = Self::simp_lin(l, pool);
        }
    }

    // -----------------------------------------------------------------
    // Non-vanishing of coefficients (case splits on parameters)
    // -----------------------------------------------------------------

    fn depends_on_vars(&self, e: ExprId, pool: &ExprPool) -> bool {
        self.vars.iter().any(|&z| contains(e, z, pool))
    }

    fn nonzero_kind(&self, c: ExprId, pool: &ExprPool) -> Nz {
        if is_number(c, pool) {
            return Nz::Yes;
        }
        if self.params.is_empty() {
            return Nz::Yes;
        }
        let Ok((rf, gens)) =
            crate::poly::cancel::rational_over_generators(canon_powers(c, pool), vec![], pool)
        else {
            return Nz::Yes;
        };
        let var_dep: Vec<bool> = gens
            .iter()
            .map(|&g| self.vars.iter().any(|&z| contains(g, z, pool)))
            .collect();
        let pidx: Vec<usize> = (0..gens.len()).filter(|&i| !var_dep[i]).collect();
        if pidx.is_empty() {
            return Nz::Yes;
        }
        let pvars: Vec<ExprId> = pidx.iter().map(|&i| gens[i]).collect();
        let mut groups: BTreeMap<Vec<u32>, BTreeMap<Vec<u32>, rug::Integer>> = BTreeMap::new();
        for (exps, coef) in &rf.numer.terms {
            let get = |i: usize| exps.get(i).copied().unwrap_or(0);
            let key: Vec<u32> = (0..gens.len()).filter(|&i| var_dep[i]).map(get).collect();
            let mut pk: Vec<u32> = pidx.iter().map(|&i| get(i)).collect();
            while pk.last() == Some(&0) {
                pk.pop();
            }
            groups.entry(key).or_default().insert(pk, coef.clone());
        }
        let mut g: Option<MultiPoly> = None;
        for (_, terms) in groups {
            let p = MultiPoly {
                vars: pvars.clone(),
                terms,
            };
            g = Some(match g {
                None => p,
                Some(q) => match q.gcd(&p) {
                    Some(r) => r,
                    None => return Nz::Yes,
                },
            });
        }
        let Some(g) = g else { return Nz::Yes };
        if g.total_degree() == 0 {
            return Nz::Yes;
        }
        let factors: Vec<ExprId> = match g.factor_irreducible() {
            Some((_, fs)) => fs.iter().map(|(f, _)| f.to_expr(pool)).collect(),
            None => vec![g.to_expr(pool)],
        };
        for f in factors {
            if is_number(f, pool) {
                continue;
            }
            let fs = simp(f, pool);
            let nf = simp(pool.mul(vec![pool.integer(-1), fs]), pool);
            if !self.nonzero.contains(&fs) && !self.nonzero.contains(&nf) {
                return Nz::Split(fs);
            }
        }
        Nz::Yes
    }

    fn subs_params(&mut self, p: ExprId, v: ExprId, pool: &ExprPool) {
        let mut m = HashMap::new();
        m.insert(p, v);
        let sub = |l: &Lin| -> Lin {
            Self::simp_lin(
                l.iter()
                    .map(|(a, &c)| (a.clone(), crate::kernel::subs::subs(c, &m, pool)))
                    .collect(),
                pool,
            )
        };
        self.eqs = self.eqs.iter().map(sub).collect();
        self.normalized = vec![false; self.eqs.len()];
        self.targets = self.targets.iter().map(sub).collect();
        self.params.retain(|&q| q != p);
    }

    /// Branches for a parameter factor `f` that might vanish.
    fn branch(&self, f: ExprId, pool: &ExprPool) -> Vec<DetSys> {
        let mut out = Vec::new();
        let mut a = self.clone();
        a.nonzero.push(f);
        a.steps.push(format!("case {} ≠ 0", pool.display(f)));
        out.push(a);
        let ps: Vec<ExprId> = self
            .params
            .iter()
            .copied()
            .filter(|&p| contains(f, p, pool))
            .collect();
        let mut solved = false;
        for p in ps {
            let vals = isolate(f, p, pool);
            let mut any = false;
            for v in vals {
                if contains(v, p, pool) {
                    continue;
                }
                let mut m = HashMap::new();
                m.insert(p, v);
                if !is_zero(crate::kernel::subs::subs(f, &m, pool), pool) {
                    continue;
                }
                let mut b = self.clone();
                b.subs_params(p, v, pool);
                b.case.push((p, v));
                b.steps
                    .push(format!("case {} = {}", pool.display(p), pool.display(v)));
                out.push(b);
                any = true;
            }
            if any {
                solved = true;
                break;
            }
        }
        if !solved {
            out[0].note(format!(
                "the special case {} = 0 was not analysed",
                pool.display(f)
            ));
            out[0].complete = false;
        }
        out
    }

    // -----------------------------------------------------------------
    // Steps
    // -----------------------------------------------------------------

    /// Split one equation by a variable none of its unknowns depends on.
    fn try_split(&mut self, pool: &ExprPool) -> bool {
        for i in 0..self.eqs.len() {
            let eq = self.eqs[i].clone();
            let mut fdeps = BTreeSet::new();
            for a in eq.keys() {
                fdeps.extend(self.funcs[a.0].deps.iter().copied());
            }
            for j in 0..self.vars.len() {
                if fdeps.contains(&j) {
                    continue;
                }
                let z = self.vars[j];
                if !eq.values().any(|&c| contains(c, z, pool)) {
                    continue;
                }
                if let Some(parts) = self.split_eq(&eq, j, pool) {
                    if parts.len() <= 1 {
                        // Nothing to split (e.g. a lone generator): leave it.
                        continue;
                    }
                    self.eqs.remove(i);
                    self.normalized.remove(i);
                    let n = parts.len();
                    for p in parts {
                        self.push_eq(p);
                    }
                    self.steps
                        .push(format!("split by {} into {n}", pool.display(z)));
                    return true;
                }
            }
        }
        false
    }

    /// The parts of `eq` by monomials in the `z_j`-dependent generators.
    fn split_eq(&mut self, eq: &Lin, j: usize, pool: &ExprPool) -> Option<Vec<Lin>> {
        let z0 = self.vars[j];
        let (expr, atoms) = lin_to_symbolic(eq, pool);
        let expr = canon_powers(expr, pool);
        // Fractional powers z^(p/q): split in w = z^(1/L), L = lcm(q).
        let (expr, z) = match root_variable(expr, z0, pool) {
            Some((e2, w)) => (e2, w),
            None => (expr, z0),
        };
        let syms: Vec<ExprId> = atoms.iter().map(|a| a.0).collect();
        let m = syms.len();
        let (rf, gens) = crate::poly::cancel::rational_over_generators(expr, syms, pool).ok()?;
        let dep: Vec<usize> = (m..gens.len())
            .filter(|&g| contains(gens[g], z, pool))
            .collect();
        if dep.is_empty() {
            return None;
        }
        // Independence of the monomials in the dependent generators.
        let mut fams: Vec<String> = Vec::new();
        let mut ok = true;
        let mut trig: Option<(usize, usize)> = None; // (sin idx, cos idx)
        let mut sym_pow: Option<(usize, ExprId)> = None;
        for &g in &dep {
            if gens[g] == z {
                continue;
            }
            match family(gens[g], z, &self.params, pool) {
                Some(f) => {
                    if f.starts_with("trig:") || f.starts_with("htrig:") {
                        let is_sin = matches!(pool.get(gens[g]),
                            ExprData::Func { ref name, .. } if name == "sin" || name == "sinh");
                        let t = trig.get_or_insert((usize::MAX, usize::MAX));
                        if is_sin {
                            t.0 = g;
                        } else {
                            t.1 = g;
                        }
                    }
                    if let Some(rest) = f.strip_prefix("pow:") {
                        let _ = rest;
                        if let ExprData::Pow { exp, .. } = pool.get(gens[g]) {
                            sym_pow = Some((g, exp));
                        }
                    }
                    if fams.contains(&f) && !f.starts_with("trig:") && !f.starts_with("htrig:") {
                        ok = false;
                    }
                    fams.push(f);
                }
                None => ok = false,
            }
        }
        let trig_fams: BTreeSet<&String> = fams
            .iter()
            .filter(|f| f.starts_with("trig:") || f.starts_with("htrig:"))
            .collect();
        if trig_fams.len() > 1 {
            ok = false;
        }
        // sin² → 1 − cos² (sinh² → cosh² − 1) when both are present.
        let mut terms = rf.numer.terms.clone();
        if let Some((si, ci)) = trig {
            if si != usize::MAX && ci != usize::MAX {
                let hyper = matches!(pool.get(gens[si]),
                    ExprData::Func { ref name, .. } if name == "sinh");
                let mut changed = true;
                let mut guard = 0;
                while changed && guard < 64 {
                    guard += 1;
                    changed = false;
                    let mut next: BTreeMap<Vec<u32>, rug::Integer> = BTreeMap::new();
                    for (e, c) in &terms {
                        let mut e = e.clone();
                        e.resize(gens.len(), 0);
                        if e[si] >= 2 {
                            changed = true;
                            let mut a = e.clone();
                            a[si] -= 2;
                            let mut b = a.clone();
                            b[ci] += 2;
                            // sin² = 1 − cos²;  sinh² = cosh² − 1.
                            let (ca, cb) = if hyper {
                                (-c.clone(), c.clone())
                            } else {
                                (c.clone(), -c.clone())
                            };
                            *next.entry(a).or_insert_with(|| rug::Integer::from(0)) += ca;
                            *next.entry(b).or_insert_with(|| rug::Integer::from(0)) += cb;
                        } else {
                            *next.entry(e).or_insert_with(|| rug::Integer::from(0)) += c;
                        }
                    }
                    next.retain(|_, c| *c != 0);
                    terms = next;
                }
            }
        }
        let zidx = (m..gens.len()).find(|&g| gens[g] == z);
        let mut groups: BTreeMap<Vec<u32>, Vec<BTreeMap<Vec<u32>, rug::Integer>>> = BTreeMap::new();
        for (exps, c) in &terms {
            let get = |i: usize| exps.get(i).copied().unwrap_or(0);
            let hit: Vec<usize> = (0..m).filter(|&i| get(i) > 0).collect();
            if hit.len() != 1 || get(hit[0]) != 1 {
                return None;
            }
            let key: Vec<u32> = dep.iter().map(|&g| get(g)).collect();
            let rest: Vec<u32> = (m..gens.len())
                .map(|g| if dep.contains(&g) { 0 } else { get(g) })
                .collect();
            let per = groups
                .entry(key)
                .or_insert_with(|| vec![BTreeMap::new(); m]);
            per[hit[0]].insert(rest, c.clone());
        }
        // Symbolic powers z^s next to integer powers of z: the split is
        // generic; record the exceptional values of s.
        if let (Some((pi, s)), Some(zi)) = (sym_pow, zidx) {
            let keys: Vec<(i64, i64)> = groups
                .keys()
                .map(|k| {
                    let at = |g: usize| dep.iter().position(|&x| x == g).map(|p| k[p] as i64);
                    (at(zi).unwrap_or(0), at(pi).unwrap_or(0))
                })
                .collect();
            for a in 0..keys.len() {
                for b in a + 1..keys.len() {
                    let (a1, b1) = keys[a];
                    let (a2, b2) = keys[b];
                    if b1 != b2 {
                        // a1 + b1 s = a2 + b2 s  ⇔  s = (a2 − a1)/(b1 − b2)
                        let v = pool.rational(a2 - a1, b1 - b2);
                        let v = simp(v, pool);
                        if !self.special.iter().any(|&(e, w)| e == s && w == v) {
                            self.special.push((s, v));
                        }
                    }
                }
            }
        } else if sym_pow.is_some() {
            // z^s with no integer powers beside it: generic in s as well
        }
        if !ok {
            self.complete = false;
            self.note(format!(
                "split by {} over generators not proven independent",
                pool.display(z)
            ));
        }
        let rest_vars: Vec<ExprId> = (m..gens.len())
            .map(|g| {
                if dep.contains(&g) {
                    pool.integer(1)
                } else {
                    gens[g]
                }
            })
            .collect();
        let mut out = Vec::new();
        for (_, per) in groups {
            let mut l = Lin::new();
            for (i, t) in per.into_iter().enumerate() {
                if t.is_empty() {
                    continue;
                }
                l.insert(atoms[i].1.clone(), mp_expr(&rest_vars, t, pool));
            }
            if !l.is_empty() {
                out.push(l);
            }
        }
        Some(out)
    }

    /// `c·∂^α f = 0` with `c ≠ 0` (pure α first; mixed when `mixed`).
    fn try_single(&mut self, pool: &ExprPool, mixed: bool) -> Step {
        for i in 0..self.eqs.len() {
            if self.eqs[i].len() != 1 {
                continue;
            }
            let (a, &c) = self.eqs[i].iter().next().unwrap();
            let a = a.clone();
            let support: Vec<usize> = (0..a.1.len()).filter(|&j| a.1[j] > 0).collect();
            if support.len() > 1 && !mixed {
                continue;
            }
            if !self.depends_on_vars(c, pool) {
                if let Nz::Split(f) = self.nonzero_kind(c, pool) {
                    return Step::Branch(f);
                }
            }
            self.eqs.remove(i);
            self.normalized.remove(i);
            let k = a.0;
            let deps = self.funcs[k].deps.clone();
            let mut l = Lin::new();
            if support.is_empty() {
                // f = 0.
            } else {
                for &j in &support {
                    let sub: Vec<usize> = deps.iter().copied().filter(|&x| x != j).collect();
                    for p in 0..a.1[j] {
                        let g = self.fresh_func(k, sub.clone());
                        let coef = pool.pow(self.vars[j], pool.integer(p as i64));
                        let coef = simp(coef, pool);
                        let za = self.zero_alpha();
                        Self::lin_add(&mut l, (g, za), coef, pool);
                    }
                }
            }
            self.steps
                .push(format!("{} = 0 integrated", self.atom_display(&a, pool)));
            self.substitute(k, &l, pool);
            return Step::Done;
        }
        Step::None
    }

    fn atom_display(&self, a: &Atom, pool: &ExprPool) -> String {
        let f = &self.funcs[a.0].name;
        let mut s = f.clone();
        let mut sub = String::new();
        for (j, &k) in a.1.iter().enumerate() {
            for _ in 0..k {
                sub.push_str(&pool.display(self.vars[j]).to_string());
            }
        }
        if !sub.is_empty() {
            s.push('_');
            s.push_str(&sub);
        }
        s
    }

    /// Eliminate an unknown occurring once, undifferentiated.
    fn try_eliminate(&mut self, pool: &ExprPool) -> Step {
        let mut order: Vec<usize> = (0..self.eqs.len()).collect();
        order.sort_by_key(|&i| self.eqs[i].len());
        for i in order {
            let eq = self.eqs[i].clone();
            let za = self.zero_alpha();
            // Prefer the function with the most variables.
            let mut cands: Vec<usize> = eq
                .keys()
                .filter(|a| a.1 == za && eq.keys().filter(|b| b.0 == a.0).count() == 1)
                .map(|a| a.0)
                .collect();
            cands.sort_by_key(|&k| std::cmp::Reverse(self.funcs[k].deps.len()));
            // Only the first candidate is tried: it either branches or is
            // eliminated.
            if let Some(&k) = cands.first() {
                let c = eq[&(k, za.clone())];
                if !self.depends_on_vars(c, pool) {
                    if let Nz::Split(f) = self.nonzero_kind(c, pool) {
                        return Step::Branch(f);
                    }
                } else if let Nz::Split(f) = self.nonzero_kind(c, pool) {
                    return Step::Branch(f);
                }
                let inv = pool.pow(c, pool.integer(-1));
                let mut l = Lin::new();
                for (b, &cb) in &eq {
                    if b.0 == k {
                        continue;
                    }
                    let v = simp(pool.mul(vec![pool.integer(-1), cb, inv]), pool);
                    l.insert(b.clone(), v);
                }
                self.eqs.remove(i);
                self.normalized.remove(i);
                self.steps
                    .push(format!("eliminated {}", self.funcs[k].name));
                self.substitute(k, &l, pool);
                return Step::Done;
            }
        }
        Step::None
    }

    /// `∫ L dz_j` term by term; `None` when a term is not integrable.
    fn int_lin(&self, l: &Lin, j: usize, pool: &ExprPool) -> Option<Lin> {
        let z = self.vars[j];
        let mut out = Lin::new();
        for (a, &c) in l {
            if self.funcs[a.0].deps.contains(&j) {
                if contains(c, z, pool) || a.1[j] == 0 {
                    return None;
                }
                let mut b = a.1.clone();
                b[j] -= 1;
                Self::lin_add(&mut out, (a.0, b), c, pool);
            } else {
                let i = super::solve::util::integrate(c, z, pool)?;
                let back = d(i, z, pool)?;
                if !is_zero(
                    pool.add(vec![back, pool.mul(vec![pool.integer(-1), c])]),
                    pool,
                ) {
                    return None;
                }
                Self::lin_add(&mut out, a.clone(), i, pool);
            }
        }
        Some(Self::simp_lin(out, pool))
    }

    /// `c ∂_{z_j}^r f + rest = 0` with `rest` integrable in `z_j`.
    fn try_integrate(&mut self, pool: &ExprPool) -> Step {
        let mut order: Vec<usize> = (0..self.eqs.len()).collect();
        order.sort_by_key(|&i| self.eqs[i].len());
        for i in order {
            let eq = self.eqs[i].clone();
            for (a, &c) in &eq {
                let k = a.0;
                if eq.keys().filter(|b| b.0 == k).count() != 1 {
                    continue;
                }
                let support: Vec<usize> = (0..a.1.len()).filter(|&j| a.1[j] > 0).collect();
                if support.len() != 1 {
                    continue;
                }
                let j = support[0];
                let r = a.1[j];
                let z = self.vars[j];
                if let Nz::Split(f) = self.nonzero_kind(c, pool) {
                    return Step::Branch(f);
                }
                let inv = pool.pow(c, pool.integer(-1));
                let mut rhs = Lin::new();
                for (b, &cb) in &eq {
                    if b.0 == k {
                        continue;
                    }
                    rhs.insert(
                        b.clone(),
                        simp(pool.mul(vec![pool.integer(-1), cb, inv]), pool),
                    );
                }
                let mut cur = Some(rhs);
                for _ in 0..r {
                    cur = cur.and_then(|l| self.int_lin(&l, j, pool));
                }
                let Some(mut l) = cur else { continue };
                let deps = self.funcs[k].deps.clone();
                let sub: Vec<usize> = deps.iter().copied().filter(|&x| x != j).collect();
                for p in 0..r {
                    let g = self.fresh_func(k, sub.clone());
                    let coef = simp(pool.pow(z, pool.integer(p as i64)), pool);
                    let za = self.zero_alpha();
                    Self::lin_add(&mut l, (g, za), coef, pool);
                }
                self.eqs.remove(i);
                self.normalized.remove(i);
                self.steps.push(format!(
                    "integrated {} in {}",
                    self.funcs[k].name,
                    pool.display(z)
                ));
                self.substitute(k, &l, pool);
                return Step::Done;
            }
        }
        Step::None
    }

    /// A linear ODE in one function of one variable, by `dsolve`.
    fn try_ode(&mut self, pool: &ExprPool) -> bool {
        for i in 0..self.eqs.len() {
            let eq = &self.eqs[i];
            let k = eq.keys().next().unwrap().0;
            if !eq.keys().all(|a| a.0 == k) || self.funcs[k].deps.is_empty() {
                continue;
            }
            // Every atom a pure derivative in one variable z_j; the other
            // variables of f are parameters of the ODE.
            let deps = self.funcs[k].deps.clone();
            let Some(j) = deps.iter().copied().find(|&j| {
                eq.keys()
                    .all(|a| a.1.iter().enumerate().all(|(i2, &o)| i2 == j || o == 0))
            }) else {
                continue;
            };
            let z = self.vars[j];
            if eq.values().any(|&c| {
                self.vars
                    .iter()
                    .enumerate()
                    .any(|(i2, &w)| !deps.contains(&i2) && contains(c, w, pool))
            }) {
                continue;
            }
            let order = eq.keys().map(|a| a.1[j]).max().unwrap_or(0) as usize;
            if order == 0 {
                continue;
            }
            let y = pool.symbol("__detsys_y", Domain::Real);
            let (mut input, derivs) = crate::ode::dsolve::OdeInput::higher_order(z, y, order, pool);
            let mut terms = Vec::new();
            for (a, &c) in eq {
                let r = a.1[j] as usize;
                let v = if r == 0 { y } else { derivs[r - 1] };
                terms.push(pool.mul(vec![c, v]));
            }
            input.equation = pool.add(terms);
            let Ok(res) = crate::ode::dsolve::dsolve(&input, pool) else {
                continue;
            };
            let Some(sol) = res.solutions.first() else {
                continue;
            };
            // Linear in the constants, no particular part.
            let mut basis = Vec::new();
            let mut zero_m = HashMap::new();
            for &cst in &sol.constants {
                zero_m.insert(cst, pool.integer(0));
            }
            let mut ok = true;
            for &cst in &sol.constants {
                match d(sol.y_of_x, cst, pool) {
                    Some(phi) if sol.constants.iter().all(|&o| !contains(phi, o, pool)) => {
                        basis.push(phi)
                    }
                    _ => ok = false,
                }
            }
            if !ok {
                continue;
            }
            let rest = crate::kernel::subs::subs(sol.y_of_x, &zero_m, pool);
            if !is_zero(rest, pool) {
                continue;
            }
            let recon = pool.add(
                sol.constants
                    .iter()
                    .zip(&basis)
                    .map(|(&c, &b)| pool.mul(vec![c, b]))
                    .collect(),
            );
            if !is_zero(
                pool.add(vec![recon, pool.mul(vec![pool.integer(-1), sol.y_of_x])]),
                pool,
            ) {
                continue;
            }
            if basis.len() < order {
                self.complete = false;
                self.note("an ODE of the determining system was solved only partially");
            }
            let rest_deps: Vec<usize> = deps.iter().copied().filter(|&x| x != j).collect();
            let mut l = Lin::new();
            for b in basis {
                let g = self.fresh_func(k, rest_deps.clone());
                let za = self.zero_alpha();
                l.insert((g, za), simp(b, pool));
            }
            self.eqs.remove(i);
            self.normalized.remove(i);
            self.steps
                .push(format!("solved an ODE for {}", self.funcs[k].name));
            self.substitute(k, &l, pool);
            return true;
        }
        false
    }

    // -----------------------------------------------------------------
    // Completion (linear differential Gröbner / Riquier–Janet)
    // -----------------------------------------------------------------

    fn rank(a: &Atom) -> (u32, usize, Alpha) {
        (a.1.iter().sum(), a.0, a.1.clone())
    }

    fn leader(eq: &Lin) -> (Atom, ExprId) {
        let (a, &c) = eq
            .iter()
            .max_by(|x, y| Self::rank(x.0).cmp(&Self::rank(y.0)))
            .expect("non-empty equation");
        (a.clone(), c)
    }

    fn lin_comb(&self, a: ExprId, x: &Lin, b: ExprId, y: &Lin, pool: &ExprPool) -> Lin {
        let mut out = Lin::new();
        for (k, &c) in x {
            Self::lin_add(&mut out, k.clone(), pool.mul(vec![a, c]), pool);
        }
        for (k, &c) in y {
            Self::lin_add(&mut out, k.clone(), pool.mul(vec![b, c]), pool);
        }
        Self::simp_lin(out, pool)
    }

    /// Fully reduce `e` by the leaders of `gs`; `Err(f)` asks for a case
    /// split on a leading coefficient.
    fn reduce_by(&self, mut e: Lin, gs: &[Lin], pool: &ExprPool) -> Result<Option<Lin>, ExprId> {
        for _ in 0..200 {
            if crate::budget::tripped().is_some() {
                // A partial reduction is still a consequence of the system.
                return Ok(Some(e));
            }
            let Some(n) = self.normalize(&e, pool) else {
                return Ok(None);
            };
            e = n;
            let mut atoms: Vec<Atom> = e.keys().cloned().collect();
            atoms.sort_by_key(|b| std::cmp::Reverse(Self::rank(b)));
            let mut hit = None;
            'find: for b in &atoms {
                for g in gs {
                    let (la, _) = Self::leader(g);
                    if la.0 == b.0 && b.1.iter().zip(&la.1).all(|(x, y)| x >= y) {
                        hit = Some((b.clone(), g.clone()));
                        break 'find;
                    }
                }
            }
            let Some((b, g)) = hit else {
                return Ok(Some(e));
            };
            let (la, cg) = Self::leader(&g);
            if let Nz::Split(f) = self.nonzero_kind(cg, pool) {
                return Err(f);
            }
            let ce = e[&b];
            let shift: Alpha = b.1.iter().zip(&la.1).map(|(x, y)| x - y).collect();
            let dg = self.dlin_multi(&g, &shift, pool);
            e = self.lin_comb(cg, &e, pool.mul(vec![pool.integer(-1), ce]), &dg, pool);
            if e.is_empty() {
                return Ok(None);
            }
        }
        Ok(Some(e))
    }

    /// One round of autoreduction plus integrability conditions.
    fn completion(&mut self, pool: &ExprPool) -> Step {
        const MAX_EQS: usize = 60;
        const MAX_ORDER: u32 = 10;
        if self.eqs.len() > MAX_EQS {
            self.passive = false;
            return Step::None;
        }
        // Autoreduce.
        let mut changed = false;
        let mut i = 0;
        while i < self.eqs.len() {
            if crate::budget::tripped().is_some() {
                self.passive = false;
                return Step::None;
            }
            let others: Vec<Lin> = self
                .eqs
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, e)| e.clone())
                .collect();
            match self.reduce_by(self.eqs[i].clone(), &others, pool) {
                Err(f) => return Step::Branch(f),
                Ok(None) => {
                    self.eqs.remove(i);
                    self.normalized.remove(i);
                    changed = true;
                    continue;
                }
                Ok(Some(r)) => {
                    if r != self.eqs[i] {
                        self.eqs[i] = r;
                        self.normalized[i] = true;
                        changed = true;
                    }
                }
            }
            i += 1;
        }
        if changed {
            self.steps.push("autoreduced".into());
            return Step::Done;
        }
        // Integrability conditions.
        let n = self.eqs.len();
        let mut new: Vec<Lin> = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                let (la, ca) = Self::leader(&self.eqs[a]);
                let (lb, cb) = Self::leader(&self.eqs[b]);
                if la.0 != lb.0 {
                    continue;
                }
                let g: Alpha = la.1.iter().zip(&lb.1).map(|(x, y)| *x.max(y)).collect();
                if g.iter().sum::<u32>() > MAX_ORDER {
                    self.passive = false;
                    continue;
                }
                let sa: Alpha = g.iter().zip(&la.1).map(|(x, y)| x - y).collect();
                let sb: Alpha = g.iter().zip(&lb.1).map(|(x, y)| x - y).collect();
                let da = self.dlin_multi(&self.eqs[a], &sa, pool);
                let db = self.dlin_multi(&self.eqs[b], &sb, pool);
                let sp = self.lin_comb(cb, &da, pool.mul(vec![pool.integer(-1), ca]), &db, pool);
                let mut basis = self.eqs.clone();
                basis.extend(new.iter().cloned());
                match self.reduce_by(sp, &basis, pool) {
                    Err(f) => return Step::Branch(f),
                    Ok(None) => {}
                    Ok(Some(r)) => new.push(r),
                }
                if crate::budget::tripped().is_some() {
                    self.passive = false;
                    return Step::None;
                }
            }
        }
        if new.is_empty() {
            return Step::None;
        }
        self.steps
            .push(format!("{} integrability condition(s)", new.len()));
        for e in new {
            self.push_eq(e);
        }
        Step::Done
    }

    /// Run the heuristics to a fixed point; one system per case branch.
    pub(crate) fn solve(self, pool: &ExprPool) -> Vec<DetSys> {
        let mut branches = 0usize;
        let mut out = Vec::new();
        let mut stack = vec![self];
        while let Some(mut s) = stack.pop() {
            match s.run(pool) {
                None => out.push(s),
                Some(f) => {
                    if branches >= MAX_BRANCHES {
                        s.complete = false;
                        s.note("case-split budget exhausted");
                        out.push(s);
                        continue;
                    }
                    branches += 1;
                    let mut bs = s.branch(f, pool);
                    bs.reverse();
                    stack.extend(bs);
                }
            }
        }
        out
    }

    /// The loop; `Some(f)` asks for a case split on the parameter factor `f`.
    fn run(&mut self, pool: &ExprPool) -> Option<ExprId> {
        for _ in 0..MAX_STEPS {
            if crate::budget::tripped().is_some() {
                self.complete = false;
                self.note("the active budget was exhausted");
                return None;
            }
            self.cleanup(pool);
            if self.eqs.is_empty() {
                return None;
            }
            if self.try_split(pool) {
                continue;
            }
            match self.try_single(pool, false) {
                Step::Done => continue,
                Step::Branch(f) => return Some(f),
                Step::None => {}
            }
            match self.try_eliminate(pool) {
                Step::Done => continue,
                Step::Branch(f) => return Some(f),
                Step::None => {}
            }
            match self.try_integrate(pool) {
                Step::Done => continue,
                Step::Branch(f) => return Some(f),
                Step::None => {}
            }
            if self.try_ode(pool) {
                continue;
            }
            match self.try_single(pool, true) {
                Step::Done => continue,
                Step::Branch(f) => return Some(f),
                Step::None => {}
            }
            match self.completion(pool) {
                Step::Done => continue,
                Step::Branch(f) => return Some(f),
                Step::None => {}
            }
            break;
        }
        // Residual equations are reported by the caller (they may be the
        // superposition part of a linear equation, which is complete).
        None
    }

    // -----------------------------------------------------------------
    // Reading the answer
    // -----------------------------------------------------------------

    /// Live functions mentioned by the targets or the residual equations.
    pub(crate) fn live_funcs(&self) -> BTreeSet<usize> {
        let mut s = BTreeSet::new();
        for l in self.targets.iter().chain(self.eqs.iter()) {
            for a in l.keys() {
                s.insert(a.0);
            }
        }
        s
    }

    /// Functions (constants included) that occur in residual equations.
    pub(crate) fn constrained(&self) -> BTreeSet<usize> {
        let mut s = BTreeSet::new();
        for l in &self.eqs {
            for a in l.keys() {
                s.insert(a.0);
            }
        }
        s
    }

    /// The targets with every function except `keep` set to zero and `keep`
    /// (a constant) set to one.
    pub(crate) fn targets_for_constant(&self, keep: usize, pool: &ExprPool) -> Vec<ExprId> {
        self.targets
            .iter()
            .map(|t| {
                t.iter()
                    .filter(|(a, _)| a.0 == keep)
                    .map(|(_, &c)| c)
                    .next()
                    .unwrap_or_else(|| pool.integer(0))
            })
            .collect()
    }

    /// A combination as an expression with the functions written as applied
    /// undefined functions `names[k](z_deps…)` and derivative nodes.
    pub(crate) fn lin_to_expr(
        &self,
        l: &Lin,
        names: &HashMap<usize, String>,
        pool: &ExprPool,
    ) -> ExprId {
        let mut terms = Vec::new();
        for (a, &c) in l {
            let f = &self.funcs[a.0];
            let name = names.get(&a.0).cloned().unwrap_or_else(|| f.name.clone());
            let atom = if f.deps.is_empty() {
                pool.symbol(name, Domain::Real)
            } else {
                let args: Vec<ExprId> = f.deps.iter().map(|&j| self.vars[j]).collect();
                let inner = pool.func(name, args);
                let orders: Vec<u32> = f.deps.iter().map(|&j| a.1[j]).collect();
                if orders.iter().all(|&o| o == 0) {
                    inner
                } else {
                    crate::diff::undefined::build(inner, &orders, pool)
                }
            };
            terms.push(pool.mul(vec![c, atom]));
        }
        match terms.len() {
            0 => pool.integer(0),
            1 => simp(terms[0], pool),
            _ => simp(pool.add(terms), pool),
        }
    }
}

/// Children of a node (re-export for sibling modules).
#[allow(dead_code)]
pub(crate) fn children(n: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut v = Vec::new();
    push_children(n, pool, &mut v);
    v
}
