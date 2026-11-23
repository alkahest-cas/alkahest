//! Lie point symmetries of a scalar PDE (milestone M6) — the equivalent of
//! Maple PDEtools' `DeterminingPDE` / `Infinitesimals`.
//!
//! # Method
//!
//! For `F(x, u, u^{(n)}) = 0` with independent variables `x = (x₁, …, x_p)`
//! and a vector field `X = Σ ξⁱ(x, u) ∂_{xᵢ} + η(x, u) ∂_u`:
//!
//! 1. the characteristic `Q = η − Σ ξⁱ u_{xᵢ}` and the prolongation
//!    coefficients `η^J = D_J Q + Σ ξⁱ u_{J+eᵢ}` are built with total
//!    derivatives on M0's [`JetSpace`], the infinitesimals kept as unknown
//!    functions of `(x, u)` (the chain rule `D_i ξ = ξ_{xᵢ} + u_{xᵢ} ξ_u`);
//! 2. `pr X(F)` is restricted to solutions by solving `F = 0` for a leading
//!    derivative `u_L` and replacing `u_L` and all its derivatives;
//! 3. the result is a polynomial in the remaining (parametric) derivatives;
//!    its coefficients are the **determining equations**, a linear
//!    overdetermined system for `ξ, η`;
//! 4. that system is solved by the RIF-style heuristics of the `detsys` module (splitting,
//!    trivial-equation integration, elimination, `dsolve` for linear ODEs),
//!    with case splits on parameters (group classification);
//! 5. every finite generator is **verified** independently: `pr X(F)` is
//!    recomputed with M0's total derivatives on the concrete `ξ, η` and must
//!    vanish on solutions exactly (`zero_test`).
//!
//! What the heuristics cannot resolve is reported, never dropped: the
//! remaining equations and the arbitrary functions they constrain form the
//! *infinite-dimensional part* (`η = α(x)∂_u` with `α` any solution, for a
//! linear equation: superposition) and the `complete` flag says whether the
//! finite list is the whole algebra.
//!
//! Errors: `E-PDE-300` (outside scope), `E-PDE-301` (not polynomial in the
//! derivatives), `E-PDE-302` (a generator that fails verification, from
//! [`verify_symmetry`] callers).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::detsys::{atom_sym, canon_powers, parse_atom_sym, Alpha, DetSys, Lin};
use super::error::PdeError;
use super::jet::{decode_unknown, symbols_of, JetSpace};
use super::solve::util::{contains, d, is_number, is_zero, simp};

// ---------------------------------------------------------------------------
// Jet helpers shared with the conservation-law module
// ---------------------------------------------------------------------------

/// One scalar unknown `u(x₁, …, x_p)` with its jet space.
#[derive(Clone, Debug)]
pub(crate) struct Jets {
    pub space: JetSpace,
    pub indep: Vec<ExprId>,
    pub unknown: ExprId,
    /// The jet symbol of `u` itself.
    pub w: ExprId,
}

impl Jets {
    pub(crate) fn new(unknown: ExprId, pool: &ExprPool) -> Result<Jets, PdeError> {
        let v = decode_unknown(unknown, pool)?;
        let indep = v.args;
        let space = JetSpace::new(&indep, &[unknown], 64, pool)?;
        let w = space.jet(0, &vec![0; indep.len()], pool)?;
        Ok(Jets {
            space,
            indep,
            unknown,
            w,
        })
    }

    pub(crate) fn p(&self) -> usize {
        self.indep.len()
    }

    pub(crate) fn jet(&self, alpha: &[u32], pool: &ExprPool) -> ExprId {
        self.space
            .jet(0, alpha, pool)
            .expect("jet within the prolonged space")
    }

    pub(crate) fn decode(&self, sym: ExprId, pool: &ExprPool) -> Option<Alpha> {
        self.space.decode_jet(sym, pool).map(|(_, a)| a)
    }

    /// Jet symbols in `e` with their multi-indices (order 0 included).
    pub(crate) fn jets_in(&self, e: ExprId, pool: &ExprPool) -> Vec<(Alpha, ExprId)> {
        let mut out: Vec<(Alpha, ExprId)> = symbols_of(e, pool)
            .into_iter()
            .filter_map(|s| self.decode(s, pool).map(|a| (a, s)))
            .collect();
        out.sort_by(|a, b| rank_key(&a.0).cmp(&rank_key(&b.0)));
        out
    }

    pub(crate) fn to_jets(&self, e: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
        self.space.to_jets(e, pool)
    }

    /// The total derivative `D_i e`, where `e` may contain jets and atom
    /// symbols of `sys` (unknown functions of `sys.vars`, each a coordinate
    /// that is an independent variable or a jet).
    pub(crate) fn total_d(
        &self,
        e: ExprId,
        i: usize,
        sys: Option<&DetSys>,
        pool: &ExprPool,
    ) -> ExprId {
        let mut terms = Vec::new();
        if let Some(t) = d(e, self.indep[i], pool) {
            terms.push(t);
        }
        for s in symbols_of(e, pool) {
            if let Some(a) = self.decode(s, pool) {
                let Some(pd) = d(e, s, pool) else { continue };
                if pd == pool.integer(0) {
                    continue;
                }
                let mut b = a.clone();
                b[i] += 1;
                terms.push(pool.mul(vec![pd, self.jet(&b, pool)]));
            } else if let (Some(sys), Some(atom)) = (sys, parse_atom_sym(s, pool)) {
                let Some(pd) = d(e, s, pool) else { continue };
                if pd == pool.integer(0) {
                    continue;
                }
                let mut chain = Vec::new();
                for &j in &sys.funcs[atom.0].deps {
                    let dz = self.total_d_coord(sys.vars[j], i, pool);
                    if dz == pool.integer(0) {
                        continue;
                    }
                    let mut b = atom.1.clone();
                    b[j] += 1;
                    chain.push(pool.mul(vec![atom_sym(&(atom.0, b), pool), dz]));
                }
                if !chain.is_empty() {
                    terms.push(pool.mul(vec![pd, pool.add(chain)]));
                }
            }
        }
        match terms.len() {
            0 => pool.integer(0),
            1 => simp(terms[0], pool),
            _ => simp(pool.add(terms), pool),
        }
    }

    /// `D_i z` for a coordinate `z` (an independent variable or a jet).
    pub(crate) fn total_d_coord(&self, z: ExprId, i: usize, pool: &ExprPool) -> ExprId {
        if z == self.indep[i] {
            return pool.integer(1);
        }
        if let Some(a) = self.decode(z, pool) {
            let mut b = a;
            b[i] += 1;
            return self.jet(&b, pool);
        }
        pool.integer(0)
    }

    pub(crate) fn total_d_multi(
        &self,
        e: ExprId,
        alpha: &[u32],
        sys: Option<&DetSys>,
        pool: &ExprPool,
    ) -> ExprId {
        let mut cur = e;
        for (i, &k) in alpha.iter().enumerate() {
            for _ in 0..k {
                cur = self.total_d(cur, i, sys, pool);
            }
        }
        cur
    }
}

/// Orderly ranking key: order, then the multi-index.
pub(crate) fn rank_key(a: &[u32]) -> (u32, Vec<u32>) {
    (a.iter().sum(), a.to_vec())
}

fn ge(a: &[u32], b: &[u32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x >= y)
}

/// The equation solved for a leading derivative, `u_L = G`, and the
/// substitution of `u_L` and its derivatives on solutions.
#[derive(Clone, Debug)]
pub(crate) struct Reducer {
    pub lead: Alpha,
    pub g: ExprId,
    memo: HashMap<Alpha, ExprId>,
}

impl Reducer {
    /// Choose the leading derivative of `f` (in jets): highest in the
    /// orderly ranking among those `f` is linear in, with a solved form free
    /// of its derivatives.
    pub(crate) fn new(jets: &Jets, f: ExprId, pool: &ExprPool) -> Result<Reducer, PdeError> {
        if let Some(r) = Self::evolutionary(jets, f, pool) {
            return Ok(r);
        }
        let mut cands: Vec<(Alpha, ExprId)> = jets
            .jets_in(f, pool)
            .into_iter()
            .filter(|(a, _)| a.iter().sum::<u32>() > 0)
            .collect();
        cands.sort_by(|a, b| rank_key(&b.0).cmp(&rank_key(&a.0)));
        let maxo = cands.first().map(|c| c.0.iter().sum::<u32>()).unwrap_or(0);
        for (a, s) in &cands {
            if a.iter().sum::<u32>() < maxo {
                // Prefer the highest order; a lower one only if no top-order
                // derivative works.
            }
            let Some(fa) = d(f, *s, pool) else { continue };
            if fa == pool.integer(0) || contains(fa, *s, pool) {
                continue;
            }
            let rest = simp(
                crate::kernel::subs::subs(f, &HashMap::from([(*s, pool.integer(0))]), pool),
                pool,
            );
            let g = simp(
                pool.mul(vec![pool.integer(-1), rest, pool.pow(fa, pool.integer(-1))]),
                pool,
            );
            if jets.jets_in(g, pool).iter().any(|(b, _)| ge(b, a)) {
                continue;
            }
            return Ok(Reducer {
                lead: a.clone(),
                g,
                memo: HashMap::new(),
            });
        }
        Err(PdeError::SymmetryUnsupported(format!(
            "{} cannot be solved for a leading derivative (it is not linear in any \
             derivative whose solved form is free of that derivative's derivatives)",
            pool.display(f)
        )))
    }

    /// Cauchy–Kovalevskaya form: `F` linear in a pure derivative `∂_t^k u`
    /// whose solved form has no derivative of order `≥ k` in `t` (a variable
    /// named `t` is tried first, then the others from the last).
    fn evolutionary(jets: &Jets, f: ExprId, pool: &ExprPool) -> Option<Reducer> {
        let p = jets.p();
        let mut order: Vec<usize> = (0..p).rev().collect();
        if let Some(ti) = jets
            .indep
            .iter()
            .position(|&x| pool.display(x).to_string() == "t")
        {
            order.retain(|&i| i != ti);
            order.insert(0, ti);
        }
        let js = jets.jets_in(f, pool);
        for i in order {
            let k = js.iter().map(|(a, _)| a[i]).max().unwrap_or(0);
            if k == 0 {
                continue;
            }
            let mut lead = vec![0; p];
            lead[i] = k;
            let Some((_, s)) = js.iter().find(|(a, _)| *a == lead) else {
                continue;
            };
            let Some(fa) = d(f, *s, pool) else { continue };
            if fa == pool.integer(0) || contains(fa, *s, pool) {
                continue;
            }
            // Other jets with t-order k (mixed ones) disqualify.
            if js.iter().any(|(a, _)| a[i] >= k && *a != lead) {
                continue;
            }
            if js
                .iter()
                .any(|(a, s2)| a[i] >= k && contains(fa, *s2, pool))
            {
                continue;
            }
            let rest = simp(
                crate::kernel::subs::subs(f, &HashMap::from([(*s, pool.integer(0))]), pool),
                pool,
            );
            let g = simp(
                pool.mul(vec![pool.integer(-1), rest, pool.pow(fa, pool.integer(-1))]),
                pool,
            );
            return Some(Reducer {
                lead,
                g,
                memo: HashMap::new(),
            });
        }
        None
    }

    pub(crate) fn reduced(&mut self, k: &[u32], jets: &Jets, pool: &ExprPool) -> ExprId {
        if k == self.lead.as_slice() {
            return self.g;
        }
        if let Some(&r) = self.memo.get(k) {
            return r;
        }
        let i = (0..k.len())
            .find(|&i| k[i] > self.lead[i])
            .expect("k > lead");
        let mut parent = k.to_vec();
        parent[i] -= 1;
        let base = self.reduced(&parent, jets, pool);
        let dr = jets.total_d(base, i, None, pool);
        let r = self.reduce(dr, jets, pool);
        self.memo.insert(k.to_vec(), r);
        r
    }

    /// Replace every jet that is a derivative of `u_L` by its value on
    /// solutions.
    pub(crate) fn reduce(&mut self, e: ExprId, jets: &Jets, pool: &ExprPool) -> ExprId {
        let mut cur = e;
        for _ in 0..16 {
            let hits: Vec<(Alpha, ExprId)> = jets
                .jets_in(cur, pool)
                .into_iter()
                .filter(|(a, _)| ge(a, &self.lead))
                .collect();
            if hits.is_empty() {
                return cur;
            }
            let mut m = HashMap::new();
            for (a, s) in hits {
                let v = self.reduced(&a, jets, pool);
                m.insert(s, v);
            }
            cur = simp(crate::kernel::subs::subs(cur, &m, pool), pool);
        }
        cur
    }
}

/// Symbols of `f` that are neither independent variables nor jets.
pub(crate) fn parameters_of(f: ExprId, jets: &Jets, pool: &ExprPool) -> Vec<ExprId> {
    symbols_of(f, pool)
        .into_iter()
        .filter(|&s| {
            !jets.indep.contains(&s)
                && jets.decode(s, pool).is_none()
                && parse_atom_sym(s, pool).is_none()
        })
        .collect()
}

/// Split `expr` (linear in atom symbols, polynomial in the jets `split`) into
/// one linear equation per monomial in `split`.
pub(crate) fn split_by_jets(
    expr: ExprId,
    split: &[ExprId],
    allowed: &[ExprId],
    jets: &Jets,
    pool: &ExprPool,
) -> Result<Vec<Lin>, PdeError> {
    let expr = canon_powers(expr, pool);
    let atoms: Vec<ExprId> = symbols_of(expr, pool)
        .into_iter()
        .filter(|&s| parse_atom_sym(s, pool).is_some())
        .collect();
    let m = atoms.len();
    let mut gv = atoms.clone();
    gv.extend_from_slice(split);
    let (rf, gens) = crate::poly::cancel::rational_over_generators(expr, gv, pool)
        .map_err(|e| PdeError::NotPolynomialInDerivatives(e.to_string()))?;
    let ns = split.len();
    for &g in &gens[m + ns..] {
        let js = jets.jets_in(g, pool);
        if js
            .iter()
            .any(|(a, s)| a.iter().sum::<u32>() > 0 && !allowed.contains(s))
            || atoms.iter().any(|&a| contains(g, a, pool))
        {
            return Err(PdeError::NotPolynomialInDerivatives(format!(
                "the symmetry condition contains {} — not polynomial in the derivatives",
                pool.display(g)
            )));
        }
    }
    for k in 0..m {
        if rf.denom.degree_in(k) > 0 {
            return Err(PdeError::NotPolynomialInDerivatives(
                "an infinitesimal occurs in a denominator".into(),
            ));
        }
    }
    let rest: Vec<ExprId> = gens[m + ns..].to_vec();
    let mut groups: BTreeMap<Vec<u32>, Vec<BTreeMap<Vec<u32>, rug::Integer>>> = BTreeMap::new();
    for (exps, c) in &rf.numer.terms {
        let get = |i: usize| exps.get(i).copied().unwrap_or(0);
        let hit: Vec<usize> = (0..m).filter(|&i| get(i) > 0).collect();
        if hit.len() != 1 || get(hit[0]) != 1 {
            return Err(PdeError::NotPolynomialInDerivatives(
                "the condition is not linear in the infinitesimals".into(),
            ));
        }
        let key: Vec<u32> = (m..m + ns).map(get).collect();
        let mut r: Vec<u32> = (m + ns..gens.len()).map(get).collect();
        while r.last() == Some(&0) {
            r.pop();
        }
        groups
            .entry(key)
            .or_insert_with(|| vec![BTreeMap::new(); m])[hit[0]]
            .insert(r, c.clone());
    }
    let mut out = Vec::new();
    for (_, per) in groups {
        let mut l = Lin::new();
        for (i, t) in per.into_iter().enumerate() {
            if t.is_empty() {
                continue;
            }
            let atom = parse_atom_sym(atoms[i], pool).expect("atom");
            let c = crate::poly::multipoly::MultiPoly {
                vars: rest.clone(),
                terms: t,
            }
            .to_expr(pool);
            l.insert(atom, c);
        }
        if !l.is_empty() {
            out.push(l);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A point-symmetry generator `X = Σ ξⁱ ∂_{xᵢ} + η ∂_u`, components written
/// in the independent variables and the plain symbol `u` (the jet symbol of
/// the unknown).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SymmetryGenerator {
    /// `ξⁱ`, one per independent variable.
    pub xi: Vec<ExprId>,
    /// `η`.
    pub eta: ExprId,
    /// Whether `pr X(F) = 0` on solutions was proved exactly.
    pub verified: bool,
}

impl SymmetryGenerator {
    /// All components `(ξ¹, …, ξ^p, η)`.
    pub fn components(&self) -> Vec<ExprId> {
        let mut v = self.xi.clone();
        v.push(self.eta);
        v
    }
}

/// The part of the symmetry algebra the solver left in terms of arbitrary
/// functions: components with functions `F1(…)`, and the linear PDEs those
/// functions must satisfy.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct InfiniteSymmetries {
    /// `(ξ¹, …, ξ^p, η)` containing the arbitrary functions / constants.
    pub components: Vec<ExprId>,
    /// The arbitrary functions (applied, `F1(x, t)`) and constrained
    /// constants (symbols).
    pub functions: Vec<ExprId>,
    /// Equations (`= 0`) the functions satisfy.
    pub constraints: Vec<ExprId>,
    /// `"superposition"` when it is `α(x)∂_u` with `α` any solution of the
    /// (linear) equation itself; `"residual"` otherwise.
    pub kind: &'static str,
}

/// The point symmetries of one equation, for one parameter case.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PointSymmetries {
    /// Independent variables.
    pub independent: Vec<ExprId>,
    /// The unknown `u(x, t)`.
    pub unknown: ExprId,
    /// The symbol standing for `u` in the components.
    pub dependent: ExprId,
    /// The equation in jets.
    pub equation: ExprId,
    /// Finite generators, a basis of the finite-dimensional part.
    pub generators: Vec<SymmetryGenerator>,
    /// `[X_a, X_b] = Σ_c C_{abc} X_c` as `(a, b, c, C)`, `a < b`, 0-based;
    /// `None` when some commutator is not in the span of the generators.
    pub structure_constants: Option<Vec<(usize, usize, usize, ExprId)>>,
    /// The infinite-dimensional / unresolved part, if any.
    pub infinite: Option<InfiniteSymmetries>,
    /// The finite generators and the infinite part are *all* point
    /// symmetries (every determining equation resolved, every split proved an
    /// equivalence).
    pub complete: bool,
    /// Parameter choices of this case (`m = -4/3`).
    pub case: Vec<(ExprId, ExprId)>,
    /// Parameter factors assumed non-zero in this case.
    pub nonzero: Vec<ExprId>,
    /// Parameter values excluded from this (generic) case because a split
    /// is not an equivalence there (`m ≠ 0`); each is reported as its own
    /// case when it could be analysed.
    pub excluded: Vec<(ExprId, ExprId)>,
    /// The determining equations (before solving), with the infinitesimals
    /// as `xi_x(x, t, u)`, …, `eta(x, t, u)`.
    pub determining_equations: Vec<ExprId>,
    /// Notes (incompleteness reasons, steps).
    pub notes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Determining equations
// ---------------------------------------------------------------------------

pub(crate) struct Determining {
    pub jets: Jets,
    pub f: ExprId,
    pub sys: DetSys,
    pub red: Reducer,
}

fn single_equation(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<(Jets, ExprId), PdeError> {
    let jets = Jets::new(unknown, pool)?;
    if jets.p() < 1 {
        return Err(PdeError::SymmetryUnsupported(
            "no independent variables".into(),
        ));
    }
    let f = simp(jets.to_jets(eq, pool)?, pool);
    if jets
        .jets_in(f, pool)
        .iter()
        .all(|(a, _)| a.iter().sum::<u32>() == 0)
    {
        return Err(PdeError::NotADifferentialEquation(format!(
            "{} has no derivative of {}",
            pool.display(eq),
            pool.display(unknown)
        )));
    }
    Ok((jets, f))
}

fn names_for(jets: &Jets, pool: &ExprPool) -> Vec<String> {
    let mut names: Vec<String> = jets
        .indep
        .iter()
        .map(|&x| format!("xi_{}", pool.display(x)))
        .collect();
    names.push("eta".into());
    names
}

pub(crate) fn determining(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<Determining, PdeError> {
    let (jets, f) = single_equation(eq, unknown, pool)?;
    let p = jets.p();
    let mut red = Reducer::new(&jets, f, pool)?;
    let params = parameters_of(f, &jets, pool);
    let mut vars = jets.indep.clone();
    vars.push(jets.w);
    let mut sys = DetSys::new(vars, params);
    let all: Vec<usize> = (0..=p).collect();
    for name in names_for(&jets, pool) {
        sys.add_func(name, all.clone());
    }
    let za = sys.zero_alpha();
    for k in 0..=p {
        let mut t = Lin::new();
        t.insert((k, za.clone()), pool.integer(1));
        sys.targets.push(t);
    }
    let a0 = |k: usize| atom_sym(&(k, za.clone()), pool);
    // Q = η − Σ ξⁱ u_i
    let mut q = vec![a0(p)];
    for i in 0..p {
        let mut e = vec![0; p];
        e[i] = 1;
        q.push(pool.mul(vec![pool.integer(-1), a0(i), jets.jet(&e, pool)]));
    }
    let q = simp(pool.add(q), pool);
    let mut dq: HashMap<Alpha, ExprId> = HashMap::new();
    dq.insert(vec![0; p], q);
    let mut terms = Vec::new();
    for i in 0..p {
        if let Some(fx) = d(f, jets.indep[i], pool) {
            if fx != pool.integer(0) {
                terms.push(pool.mul(vec![a0(i), fx]));
            }
        }
    }
    for (j, s) in jets.jets_in(f, pool) {
        let Some(fj) = d(f, s, pool) else { continue };
        if fj == pool.integer(0) {
            continue;
        }
        let dj = dq_of(&j, &mut dq, &jets, &sys, pool);
        let mut eta_j = vec![dj];
        for i in 0..p {
            let mut b = j.clone();
            b[i] += 1;
            eta_j.push(pool.mul(vec![a0(i), jets.jet(&b, pool)]));
        }
        terms.push(pool.mul(vec![pool.add(eta_j), fj]));
    }
    let pr = simp(pool.add(terms), pool);
    let pr = red.reduce(pr, &jets, pool);
    // Clear the denominators of the solved form before splitting.
    let split: Vec<ExprId> = jets
        .jets_in(pr, pool)
        .into_iter()
        .filter(|(a, _)| a.iter().sum::<u32>() > 0)
        .map(|(_, s)| s)
        .collect();
    let eqs = split_by_jets(pr, &split, &[], &jets, pool)?;
    for e in eqs {
        sys.push_eq(e);
    }
    Ok(Determining { jets, f, sys, red })
}

fn dq_of(
    j: &[u32],
    memo: &mut HashMap<Alpha, ExprId>,
    jets: &Jets,
    sys: &DetSys,
    pool: &ExprPool,
) -> ExprId {
    if let Some(&v) = memo.get(j) {
        return v;
    }
    let i = j.iter().position(|&k| k > 0).expect("nonzero");
    let mut parent = j.to_vec();
    parent[i] -= 1;
    let base = dq_of(&parent, memo, jets, sys, pool);
    let r = jets.total_d(base, i, Some(sys), pool);
    memo.insert(j.to_vec(), r);
    r
}

/// The determining equations of the point symmetries of `eq = 0` (one
/// equation, one unknown), with the infinitesimals written as
/// `xi_x(x, t, u)`, …, `eta(x, t, u)` and the dependent variable as the
/// symbol `u`.
///
/// # Errors
///
/// `E-PDE-300` for an equation that cannot be solved for a leading
/// derivative; `E-PDE-301` when the symmetry condition is not polynomial in
/// the derivatives; M0 errors for a malformed unknown.
pub fn determining_equations(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, PdeError> {
    let det = determining(eq, unknown, pool)?;
    Ok(det
        .sys
        .eqs
        .iter()
        .map(|l| det.sys.lin_to_expr(l, &HashMap::new(), pool))
        .collect())
}

// ---------------------------------------------------------------------------
// Verification and algebra
// ---------------------------------------------------------------------------

/// Whether `X = Σ ξⁱ∂_{xᵢ} + η∂_u` (components in `x` and the jet symbol
/// `u`) leaves `eq = 0` invariant: `pr X(F) = 0` on solutions, decided by an
/// independent computation through M0's total derivatives and `zero_test`.
///
/// # Errors
///
/// As [`determining_equations`]; `E-PDE-013` for the wrong number of
/// components.
pub fn verify_symmetry(
    eq: ExprId,
    unknown: ExprId,
    components: &[ExprId],
    pool: &ExprPool,
) -> Result<ZeroStatus, PdeError> {
    let (jets, f) = single_equation(eq, unknown, pool)?;
    let mut red = Reducer::new(&jets, f, pool)?;
    if components.len() != jets.p() + 1 {
        return Err(PdeError::InvalidCandidate(format!(
            "a generator needs {} components, got {}",
            jets.p() + 1,
            components.len()
        )));
    }
    let comps: Vec<ExprId> = components
        .iter()
        .map(|&c| jets.to_jets(c, pool))
        .collect::<Result<_, _>>()?;
    Ok(verify_components(&jets, f, &mut red, &comps, pool))
}

fn verify_components(
    jets: &Jets,
    f: ExprId,
    red: &mut Reducer,
    comps: &[ExprId],
    pool: &ExprPool,
) -> ZeroStatus {
    let p = jets.p();
    let mut q = vec![comps[p]];
    for i in 0..p {
        let mut e = vec![0; p];
        e[i] = 1;
        q.push(pool.mul(vec![pool.integer(-1), comps[i], jets.jet(&e, pool)]));
    }
    let q = simp(pool.add(q), pool);
    let mut terms = Vec::new();
    for i in 0..p {
        if let Some(fx) = d(f, jets.indep[i], pool) {
            terms.push(pool.mul(vec![comps[i], fx]));
        }
    }
    for (j, s) in jets.jets_in(f, pool) {
        let Some(fj) = d(f, s, pool) else {
            return ZeroStatus::Unknown;
        };
        if fj == pool.integer(0) {
            continue;
        }
        let Ok(dj) = jets.space.total_derivative_multi(q, &j, pool) else {
            return ZeroStatus::Unknown;
        };
        let mut eta_j = vec![dj];
        for i in 0..p {
            let mut b = j.clone();
            b[i] += 1;
            eta_j.push(pool.mul(vec![comps[i], jets.jet(&b, pool)]));
        }
        terms.push(pool.mul(vec![pool.add(eta_j), fj]));
    }
    let pr = simp(pool.add(terms), pool);
    let pr = red.reduce(pr, jets, pool);
    if pr == pool.integer(0) {
        return ZeroStatus::ProvedZero;
    }
    let ex = crate::simplify::engine::simplify_expanded(pr, pool).value;
    if ex == pool.integer(0) {
        return ZeroStatus::ProvedZero;
    }
    // Numerator of the rational form, for quotients from the solved form.
    if let Ok((n, _)) = crate::poly::cancel::together_parts(canon_powers(ex, pool), vec![], pool) {
        if n == pool.integer(0) {
            return ZeroStatus::ProvedZero;
        }
    }
    zero_test(ex, pool, None).status
}

/// The commutator `[X, Y]` of two vector fields on `(x, u)`.
pub fn commutator(vars: &[ExprId], x: &[ExprId], y: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    let apply = |v: &[ExprId], g: ExprId| -> ExprId {
        let mut t = Vec::new();
        for (k, &z) in vars.iter().enumerate() {
            if let Some(dg) = d(g, z, pool) {
                if dg != pool.integer(0) {
                    t.push(pool.mul(vec![v[k], dg]));
                }
            }
        }
        if t.is_empty() {
            pool.integer(0)
        } else {
            pool.add(t)
        }
    };
    (0..vars.len())
        .map(|c| {
            let a = apply(x, y[c]);
            let b = apply(y, x[c]);
            crate::simplify::engine::simplify_expanded(
                pool.add(vec![a, pool.mul(vec![pool.integer(-1), b])]),
                pool,
            )
            .value
        })
        .collect()
}

/// Coefficients `c` with `target = Σ c_k basis_k` (identically in `vars`),
/// over ℚ(parameters); `None` when not in the span.
pub(crate) fn decompose(
    target: &[ExprId],
    basis: &[Vec<ExprId>],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    let n = basis.len();
    if target.iter().all(|&t| is_zero(t, pool)) {
        return Some(vec![pool.integer(0); n]);
    }
    let mut taken = super::solve::util::names_in(
        &basis
            .iter()
            .flatten()
            .copied()
            .chain(target.iter().copied())
            .collect::<Vec<_>>(),
        pool,
    );
    let dirs: Vec<ExprId> = (0..target.len())
        .map(|_| super::solve::util::fresh_symbol("__sym_e", &mut taken, pool))
        .collect();
    let bsyms: Vec<ExprId> = (0..=n)
        .map(|_| super::solve::util::fresh_symbol("__sym_b", &mut taken, pool))
        .collect();
    let pack = |v: &[ExprId]| -> ExprId {
        pool.add(
            v.iter()
                .zip(&dirs)
                .map(|(&c, &e)| pool.mul(vec![c, e]))
                .collect(),
        )
    };
    let mut images: Vec<ExprId> = basis.iter().map(|b| pack(b)).collect();
    images.push(pack(target));
    let mut all_vars = vars.to_vec();
    all_vars.extend_from_slice(&dirs);
    let combos = super::solve::integrals::nullspace_combinations(&bsyms, &images, &all_vars, pool);
    for c in combos {
        let lt = d(c, bsyms[n], pool)?;
        if is_zero(lt, pool) {
            continue;
        }
        let mut out = Vec::new();
        for &b in &bsyms[..n] {
            let ck = d(c, b, pool)?;
            out.push(simp(
                pool.mul(vec![pool.integer(-1), ck, pool.pow(lt, pool.integer(-1))]),
                pool,
            ));
        }
        // Exact check.
        let ok = (0..target.len()).all(|i| {
            let mut s = vec![target[i]];
            for k in 0..n {
                s.push(pool.mul(vec![pool.integer(-1), out[k], basis[k][i]]));
            }
            is_zero(pool.add(s), pool)
        });
        if ok {
            return Some(out);
        }
    }
    None
}

/// A basis of the span of `gens` (drops linearly dependent ones).
fn independent_subset(gens: Vec<Vec<ExprId>>, vars: &[ExprId], pool: &ExprPool) -> Vec<usize> {
    let mut keep: Vec<usize> = Vec::new();
    for (i, g) in gens.iter().enumerate() {
        if g.iter().all(|&c| is_zero(c, pool)) {
            continue;
        }
        let basis: Vec<Vec<ExprId>> = keep.iter().map(|&k| gens[k].clone()).collect();
        if !basis.is_empty() && decompose(g, &basis, vars, pool).is_some() {
            continue;
        }
        keep.push(i);
    }
    keep
}

/// Scale a generator to integer, content-free coefficients with a positive
/// leading term.
fn primitive(comps: &[ExprId], vars: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    let mut taken = super::solve::util::names_in(comps, pool);
    let dirs: Vec<ExprId> = (0..comps.len())
        .map(|_| super::solve::util::fresh_symbol("__sym_e", &mut taken, pool))
        .collect();
    let packed = pool.add(
        comps
            .iter()
            .zip(&dirs)
            .map(|(&c, &e)| pool.mul(vec![c, e]))
            .collect(),
    );
    let Ok(rf) =
        crate::poly::cancel::rational_function_of(canon_powers(packed, pool), dirs.clone(), pool)
    else {
        return comps.to_vec();
    };
    // A denominator in the parameters only is a non-zero scale factor of
    // this (generic) case; one involving the variables is kept.
    let den = rf.denom.to_expr(pool);
    if vars
        .iter()
        .chain(dirs.iter())
        .any(|&v| contains(den, v, pool))
    {
        return comps.to_vec();
    }
    let num = rf.numer.primitive_part();
    let lead_neg = num
        .terms
        .iter()
        .next_back()
        .map(|(_, c)| *c < 0)
        .unwrap_or(false);
    let e = num.to_expr(pool);
    let e = if lead_neg {
        pool.mul(vec![pool.integer(-1), e])
    } else {
        e
    };
    dirs.iter()
        .map(|&dir| {
            d(e, dir, pool)
                .map(|v| simp(v, pool))
                .unwrap_or(pool.integer(0))
        })
        .collect()
}

fn size(e: ExprId, pool: &ExprPool) -> usize {
    let mut n = 0;
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        n += 1;
        super::jet::push_children(x, pool, &mut stack);
    }
    n
}

// ---------------------------------------------------------------------------
// The symmetry computation
// ---------------------------------------------------------------------------

/// The Lie point symmetries of `eq = 0`, one [`PointSymmetries`] per
/// parameter case (the generic case first, then special values such as
/// `m = -4/3` for `u_t = (u^m u_x)_x`).
///
/// # Errors
///
/// As [`determining_equations`].
pub fn point_symmetries(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<Vec<PointSymmetries>, PdeError> {
    let mut out = point_symmetries_inner(eq, unknown, pool)?;
    // Special parameter values flagged by generic splits.
    let mut specials: Vec<(ExprId, ExprId)> = Vec::new();
    for c in &out {
        for &(s, v) in &c.excluded {
            if !specials.contains(&(s, v)) {
                specials.push((s, v));
            }
        }
    }
    let mut seen: Vec<(ExprId, ExprId)> = out.iter().flat_map(|c| c.case.clone()).collect();
    for (s, v) in specials.into_iter().take(6) {
        // `s` is a parameter expression; solve it for one parameter.
        let params = symbols_of(s, pool);
        let Some(&p) = params.iter().next() else {
            continue;
        };
        let vals = super::solve::util::isolate(
            pool.add(vec![s, pool.mul(vec![pool.integer(-1), v])]),
            p,
            pool,
        );
        let Some(&pv) = vals.iter().find(|&&x| !contains(x, p, pool)) else {
            continue;
        };
        let pv = simp(pv, pool);
        if seen.contains(&(p, pv)) {
            continue;
        }
        seen.push((p, pv));
        let eq2 = simp(
            crate::kernel::subs::subs(eq, &HashMap::from([(p, pv)]), pool),
            pool,
        );
        if let Ok(cases) = point_symmetries_inner(eq2, unknown, pool) {
            for mut c in cases {
                c.case.insert(0, (p, pv));
                out.push(c);
            }
        }
    }
    Ok(out)
}

fn point_symmetries_inner(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<Vec<PointSymmetries>, PdeError> {
    let det = determining(eq, unknown, pool)?;
    let det_eqs: Vec<ExprId> = det
        .sys
        .eqs
        .iter()
        .map(|l| det.sys.lin_to_expr(l, &HashMap::new(), pool))
        .collect();
    let branches = det.sys.clone().solve(pool);
    let mut out = Vec::new();
    for b in branches {
        let mut red = det.red.clone();
        let jets = &det.jets;
        // The equation of this branch (parameters substituted).
        let f = if b.case.is_empty() {
            det.f
        } else {
            let m: HashMap<ExprId, ExprId> = b.case.iter().copied().collect();
            red = Reducer::new(
                jets,
                simp(crate::kernel::subs::subs(det.f, &m, pool), pool),
                pool,
            )?;
            simp(crate::kernel::subs::subs(det.f, &m, pool), pool)
        };
        out.push(read_branch(&b, jets, f, &mut red, &det_eqs, pool));
    }
    // Generic case first.
    out.sort_by_key(|c| c.case.len());
    Ok(out)
}

fn read_branch(
    b: &DetSys,
    jets: &Jets,
    f: ExprId,
    red: &mut Reducer,
    det_eqs: &[ExprId],
    pool: &ExprPool,
) -> PointSymmetries {
    let p = jets.p();
    let live = b.live_funcs();
    let constrained = b.constrained();
    let mut free_consts = Vec::new();
    let mut infinite_funcs = Vec::new();
    for &k in &live {
        if b.funcs[k].deps.is_empty() && !constrained.contains(&k) {
            free_consts.push(k);
        } else {
            infinite_funcs.push(k);
        }
    }
    let mut notes = b.notes.clone();
    let mut complete = b.complete;
    // Finite generators.
    let mut raw: Vec<Vec<ExprId>> = Vec::new();
    for &k in &free_consts {
        let comps = b.targets_for_constant(k, pool);
        raw.push(comps);
    }
    let vars = b.vars.clone();
    let keep = independent_subset(raw.clone(), &vars, pool);
    let mut gens: Vec<SymmetryGenerator> = Vec::new();
    for k in keep {
        let comps = primitive(&raw[k], &vars, pool);
        let st = verify_components(jets, f, red, &comps, pool);
        let verified = st == ZeroStatus::ProvedZero;
        if st == ZeroStatus::ProvedNonzero {
            notes.push(format!(
                "discarded a candidate generator that failed verification: {:?}",
                comps
                    .iter()
                    .map(|&c| pool.display(c).to_string())
                    .collect::<Vec<_>>()
            ));
            complete = false;
            continue;
        }
        gens.push(SymmetryGenerator {
            xi: comps[..p].to_vec(),
            eta: comps[p],
            verified,
        });
    }
    gens.sort_by_key(|g| g.components().iter().map(|&c| size(c, pool)).sum::<usize>());
    // Structure constants.
    let basis: Vec<Vec<ExprId>> = gens.iter().map(|g| g.components()).collect();
    let mut sc = Some(Vec::new());
    'outer: for a in 0..gens.len() {
        for bb in a + 1..gens.len() {
            let c = commutator(&vars, &basis[a], &basis[bb], pool);
            match decompose(&c, &basis, &vars, pool) {
                Some(coefs) => {
                    for (k, ck) in coefs.into_iter().enumerate() {
                        if !is_zero(ck, pool) {
                            sc.as_mut().unwrap().push((a, bb, k, ck));
                        }
                    }
                }
                None => {
                    sc = None;
                    break 'outer;
                }
            }
        }
    }
    // Infinite / residual part.
    let infinite = if infinite_funcs.is_empty() && b.eqs.is_empty() {
        None
    } else {
        let mut names = HashMap::new();
        let mut fnames = Vec::new();
        let (mut nf, mut nc) = (0, 0);
        for &k in &infinite_funcs {
            let name = if b.funcs[k].deps.is_empty() {
                nc += 1;
                format!("C{nc}")
            } else {
                nf += 1;
                format!("F{nf}")
            };
            names.insert(k, name.clone());
            let e = if b.funcs[k].deps.is_empty() {
                pool.symbol(name, Domain::Real)
            } else {
                pool.func(name, b.funcs[k].deps.iter().map(|&j| vars[j]).collect())
            };
            fnames.push(e);
        }
        let comps: Vec<ExprId> = b
            .targets
            .iter()
            .map(|t| {
                let l: Lin = t
                    .iter()
                    .filter(|(a, _)| infinite_funcs.contains(&a.0))
                    .map(|(a, &c)| (a.clone(), c))
                    .collect();
                b.lin_to_expr(&l, &names, pool)
            })
            .collect();
        let constraints: Vec<ExprId> = b
            .eqs
            .iter()
            .map(|l| b.lin_to_expr(l, &names, pool))
            .collect();
        let kind = if is_superposition(b, &infinite_funcs, jets, f, pool) {
            "superposition"
        } else {
            "residual"
        };
        let kind = if kind == "residual" && b.passive {
            notes.push(
                "the infinite part is the general solution of a completed (passive) linear \
                 system (see `infinite`)"
                    .into(),
            );
            "passive"
        } else {
            kind
        };
        if kind == "residual" {
            complete = false;
            notes.push("part of the determining system was left unsolved (see `infinite`)".into());
        }
        Some(InfiniteSymmetries {
            components: comps,
            functions: fnames,
            constraints,
            kind,
        })
    };
    let _ = BTreeSet::<usize>::new();
    PointSymmetries {
        independent: jets.indep.clone(),
        unknown: jets.unknown,
        dependent: jets.w,
        equation: f,
        generators: gens,
        structure_constants: sc,
        infinite,
        complete,
        case: b.case.clone(),
        nonzero: b.nonzero.clone(),
        excluded: b.special.clone(),
        determining_equations: det_eqs.to_vec(),
        notes,
    }
}

/// `η = α(x)` with `α` constrained by exactly the (linear, homogeneous)
/// equation itself.
fn is_superposition(b: &DetSys, inf: &[usize], jets: &Jets, f: ExprId, pool: &ExprPool) -> bool {
    let p = jets.p();
    if inf.len() != 1 || b.eqs.len() != 1 {
        return false;
    }
    let k = inf[0];
    if b.funcs[k].deps.contains(&p) {
        return false;
    }
    // Only in η.
    if b.targets[..p].iter().any(|t| t.keys().any(|a| a.0 == k)) {
        return false;
    }
    // F linear homogeneous in the jets.
    let lam = pool.symbol("__sym_lambda", Domain::Real);
    let m: HashMap<ExprId, ExprId> = jets
        .jets_in(f, pool)
        .into_iter()
        .map(|(_, s)| (s, pool.mul(vec![lam, s])))
        .collect();
    let scaled = crate::kernel::subs::subs(f, &m, pool);
    if !is_zero(
        pool.add(vec![scaled, pool.mul(vec![pool.integer(-1), lam, f])]),
        pool,
    ) {
        return false;
    }
    // The constraint is F[α] up to a constant factor.
    let eq = &b.eqs[0];
    let mut terms = Vec::new();
    for (a, &c) in eq {
        if a.0 != k {
            return false;
        }
        terms.push(pool.mul(vec![c, jets.jet(&a.1[..p], pool)]));
    }
    let g = simp(pool.add(terms), pool);
    let (Some((la, ls)), true) = (
        jets.jets_in(f, pool)
            .into_iter()
            .max_by(|a, b| rank_key(&a.0).cmp(&rank_key(&b.0))),
        true,
    ) else {
        return false;
    };
    let _ = la;
    let (Some(cf), Some(cg)) = (d(f, ls, pool), d(g, ls, pool)) else {
        return false;
    };
    if is_zero(cg, pool) {
        return false;
    }
    let ratio = simp(pool.mul(vec![cf, pool.pow(cg, pool.integer(-1))]), pool);
    if !is_number(ratio, pool) {
        return false;
    }
    is_zero(
        pool.add(vec![f, pool.mul(vec![pool.integer(-1), ratio, g])]),
        pool,
    )
}

/// Render a generator as `ξ¹∂_x + … + η∂_u` text.
pub fn generator_display(
    g: &SymmetryGenerator,
    independent: &[ExprId],
    dependent: ExprId,
    pool: &ExprPool,
) -> String {
    let mut parts = Vec::new();
    let mut push = |c: ExprId, v: ExprId| {
        if is_zero(c, pool) {
            return;
        }
        let cs = pool.display(c).to_string();
        let vs = pool.display(v).to_string();
        if cs == "1" {
            parts.push(format!("∂_{vs}"));
        } else if cs == "-1" {
            parts.push(format!("-∂_{vs}"));
        } else if cs.contains(' ') {
            parts.push(format!("({cs})∂_{vs}"));
        } else {
            parts.push(format!("{cs}∂_{vs}"));
        }
    };
    for (i, &x) in independent.iter().enumerate() {
        push(g.xi[i], x);
    }
    push(g.eta, dependent);
    if parts.is_empty() {
        "0".into()
    } else {
        parts.join(" + ").replace("+ -", "- ")
    }
}

#[allow(dead_code)]
fn unused(_: ExprData) {}
