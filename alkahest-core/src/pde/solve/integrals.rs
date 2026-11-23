//! First integrals of an autonomous vector field `V = Σ aᵢ(z) ∂/∂zᵢ` — the
//! characteristic system `dz₁/a₁ = … = dzₙ/aₙ` of a first-order PDE.
//!
//! A first integral is a non-constant `ξ(z)` with `V ξ = 0`.  Every candidate
//! is checked **exactly** (`V ξ` proved zero) before it is kept, and the kept
//! set is functionally independent (numeric Jacobian rank — used only to
//! *choose* among exactly-verified integrals, never to certify one).
//!
//! Sources, in order:
//! 1. coordinates whose component vanishes (`aᵢ ≡ 0` ⇒ `zᵢ` is an integral);
//! 2. pairs `dzⱼ/dzᵢ = aⱼ/aᵢ` solved by `dsolve`, after eliminating any
//!    third variable through an integral already found (the classical
//!    "use one integral to find the next" step), the integration constant
//!    isolated to give `ξ`;
//! 3. an ansatz search: `ξ = Σ cₖ bₖ` over monomials of degree ≤ 3 and
//!    `log zᵢ`, with `V ξ = 0` solved as a linear system over `ℚ(parameters)`
//!    — this finds the integrable combinations that the multiplier method
//!    finds by hand (`x + y + u`, `x² + y² + u²`, `log(xyz)`).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::kernel::{ExprId, ExprPool};
use crate::ode::dsolve::{dsolve_with, OdeInput, SolutionForm};
use crate::poly::multipoly::MultiPoly;
use crate::poly::rational::RationalFunction;
use crate::simplify::assumptions::AssumptionContext;

use super::util::*;

/// The characteristic vector field.
#[derive(Clone, Debug)]
pub(crate) struct VectorField {
    pub vars: Vec<ExprId>,
    pub comps: Vec<ExprId>,
}

impl VectorField {
    pub(crate) fn new(vars: Vec<ExprId>, comps: Vec<ExprId>, pool: &ExprPool) -> Self {
        let comps = comps.into_iter().map(|c| simp(c, pool)).collect();
        VectorField { vars, comps }
    }

    /// `V f`, simplified; `None` when a derivative is refused.
    pub(crate) fn apply(&self, f: ExprId, pool: &ExprPool) -> Option<ExprId> {
        let mut terms = Vec::new();
        for (&v, &a) in self.vars.iter().zip(&self.comps) {
            if !contains(f, v, pool) {
                continue;
            }
            let df = d(f, v, pool)?;
            terms.push(pool.mul(vec![a, df]));
        }
        Some(simp(pool.add(terms), pool))
    }

    /// `V ξ` is proved zero and `ξ` is not a constant.
    pub(crate) fn is_integral(&self, xi: ExprId, pool: &ExprPool) -> bool {
        if free_of(xi, &self.vars, pool) {
            return false;
        }
        match self.apply(xi, pool) {
            Some(r) => is_zero(r, pool),
            None => false,
        }
    }

    fn jacobian_row(&self, xi: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        self.vars.iter().map(|&v| d(xi, v, pool)).collect()
    }
}

/// The search state: integrals found so far (exactly verified, independent).
pub(crate) struct IntegralSearch<'a> {
    pub field: &'a VectorField,
    pub found: Vec<ExprId>,
    rows: Vec<Vec<ExprId>>,
    taken: HashSet<String>,
    pub notes: Vec<String>,
}

impl<'a> IntegralSearch<'a> {
    pub(crate) fn new(field: &'a VectorField, taken: HashSet<String>) -> Self {
        IntegralSearch {
            field,
            found: Vec::new(),
            rows: Vec::new(),
            taken,
            notes: Vec::new(),
        }
    }

    /// Keep `xi` when it is an exactly verified integral independent of the
    /// ones already kept.
    pub(crate) fn offer(&mut self, xi: ExprId, how: &str, pool: &ExprPool) -> bool {
        let xi = simp(xi, pool);
        if self.found.contains(&xi) || !self.field.is_integral(xi, pool) {
            return false;
        }
        let Some(row) = self.field.jacobian_row(xi, pool) else {
            return false;
        };
        let mut rows = self.rows.clone();
        rows.push(row.clone());
        let syms = free_symbols(&rows.iter().flatten().copied().collect::<Vec<_>>(), pool);
        if numeric_rank(&rows, &syms, pool) < rows.len() {
            return false;
        }
        self.rows.push(row);
        self.found.push(xi);
        self.notes
            .push(format!("first integral {} ({how})", pool.display(xi)));
        true
    }

    /// Find up to `need` independent first integrals.
    pub(crate) fn run(&mut self, need: usize, pool: &ExprPool) {
        let f = self.field;
        for (&v, &a) in f.vars.iter().zip(&f.comps) {
            if self.found.len() >= need {
                return;
            }
            if a == pool.integer(0) {
                self.offer(v, "its characteristic component vanishes", pool);
            }
        }
        // Pairs, repeated while they make progress (a new integral can
        // unlock a pair through elimination).
        let mut progress = true;
        let mut rounds = 0;
        while progress && self.found.len() < need && rounds < 4 {
            progress = false;
            rounds += 1;
            let n = f.vars.len();
            for i in 0..n {
                for j in 0..n {
                    if i == j || self.found.len() >= need {
                        continue;
                    }
                    if f.comps[i] == pool.integer(0) {
                        continue;
                    }
                    if let Some(xi) = self.pair_integral(i, j, pool) {
                        if self.offer(xi, "characteristic ODE", pool) {
                            progress = true;
                        }
                    }
                }
            }
        }
        if self.found.len() < need {
            for deg in 1..=3u32 {
                if self.found.len() >= need {
                    break;
                }
                for xi in ansatz_integrals(f, deg, pool) {
                    if self.found.len() >= need {
                        break;
                    }
                    self.offer(xi, "integrable combination", pool);
                }
            }
        }
    }

    /// An integral from `dzⱼ/dzᵢ = aⱼ/aᵢ`, eliminating other variables through
    /// the integrals already found.
    fn pair_integral(&mut self, i: usize, j: usize, pool: &ExprPool) -> Option<ExprId> {
        let f = self.field;
        let (zi, zj) = (f.vars[i], f.vars[j]);
        if f.comps[j] == pool.integer(0) {
            return None;
        }
        let mut ratio = simp(div(f.comps[j], f.comps[i], pool), pool);
        // Eliminate third variables.
        let mut back: HashMap<ExprId, ExprId> = HashMap::new();
        for (k, &zk) in f.vars.iter().enumerate() {
            if k == i || k == j || !contains(ratio, zk, pool) {
                continue;
            }
            let mut done = false;
            for &xi in &self.found.clone() {
                if !contains(xi, zk, pool) {
                    continue;
                }
                // Only integrals in zᵢ, zⱼ, z_k (and already-eliminated ones).
                let others = f
                    .vars
                    .iter()
                    .enumerate()
                    .any(|(m, &zm)| m != i && m != j && m != k && contains(xi, zm, pool));
                if others {
                    continue;
                }
                let c = fresh_symbol("__pde_k", &mut self.taken, pool);
                let Some(zk_of) = isolate_one(sub(xi, c, pool), zk, pool) else {
                    continue;
                };
                ratio = simp(subs1(ratio, zk, zk_of, pool), pool);
                back.insert(c, xi);
                done = true;
                break;
            }
            if !done {
                return None;
            }
        }
        if !f
            .vars
            .iter()
            .enumerate()
            .all(|(m, &zm)| m == i || m == j || !contains(ratio, zm, pool))
        {
            return None;
        }
        if has_undefined(ratio, pool) {
            return None;
        }
        // dzⱼ/dzᵢ = ratio.
        let constant = if !contains(ratio, zj, pool) {
            // Quadrature.
            let q = integrate(ratio, zi, pool)?;
            simp(sub(zj, q, pool), pool)
        } else if let Some((a, b)) = separate(ratio, zi, zj, pool) {
            // dzⱼ/dzᵢ = A(zᵢ)·B(zⱼ): ∫ dzⱼ/B − ∫ A dzᵢ, with this module's
            // quadrature (which handles e.g. ∫ dy/sinh(μy) for symbolic μ).
            let ib = integrate(recip(b, pool), zj, pool)?;
            let ia = integrate(a, zi, pool)?;
            simp(sub(ib, ia, pool), pool)
        } else {
            let (input, yp) = OdeInput::first_order(zi, zj, pool);
            let input = input.with_equation(sub(yp, ratio, pool));
            let rep = dsolve_with(&input, &AssumptionContext::new(), pool).ok()?;
            let mut got = None;
            for b in rep.branches {
                let c1 = *b.constants.first()?;
                let rel = match b.form {
                    SolutionForm::Explicit(y) => sub(y, zj, pool),
                    SolutionForm::Implicit(g) => g,
                };
                if let Some(cv) = isolate_one(rel, c1, pool) {
                    got = Some(cv);
                    break;
                }
            }
            got?
        };
        let xi = simp(subs_map(constant, &back, pool), pool);
        Some(xi)
    }
}

/// `ratio = A(zᵢ)·B(zⱼ)` by splitting the factors of a product.
fn separate(ratio: ExprId, zi: ExprId, zj: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let factors = match pool.get(ratio) {
        crate::kernel::ExprData::Mul(xs) => xs,
        _ => vec![ratio],
    };
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for f in factors {
        let (hi, hj) = (contains(f, zi, pool), contains(f, zj, pool));
        match (hi, hj) {
            (true, true) => return None,
            (false, true) => b.push(f),
            _ => a.push(f),
        }
    }
    if b.is_empty() {
        return None;
    }
    Some((simp(pool.mul(a), pool), simp(pool.mul(b), pool)))
}

// ---------------------------------------------------------------------------
// Ansatz search for integrable combinations
// ---------------------------------------------------------------------------

fn monomials(vars: &[ExprId], deg: u32, pool: &ExprPool) -> Vec<ExprId> {
    fn rec(
        vars: &[ExprId],
        start: usize,
        left: u32,
        cur: &mut Vec<ExprId>,
        out: &mut Vec<Vec<ExprId>>,
    ) {
        if left == 0 {
            out.push(cur.clone());
            return;
        }
        for k in start..vars.len() {
            cur.push(vars[k]);
            rec(vars, k, left - 1, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    for dgr in 1..=deg {
        let mut acc = Vec::new();
        rec(vars, 0, dgr, &mut Vec::new(), &mut acc);
        for m in acc {
            out.push(pool.mul(m));
        }
    }
    out
}

/// Integrals `Σ cₖ bₖ` with `V(Σ cₖ bₖ) = 0` identically.
pub(crate) fn ansatz_integrals(f: &VectorField, deg: u32, pool: &ExprPool) -> Vec<ExprId> {
    let mut basis = monomials(&f.vars, deg, pool);
    if deg == 1 {
        for &v in &f.vars {
            basis.push(pool.func("log", vec![v]));
        }
    }
    if basis.len() > 40 {
        return Vec::new();
    }
    let mut images = Vec::with_capacity(basis.len());
    for &b in &basis {
        match f.apply(b, pool) {
            Some(r) => images.push(r),
            None => return Vec::new(),
        }
    }
    nullspace_combinations(&basis, &images, &f.vars, pool)
}

/// `Σ cₖ basisₖ` for a basis of the solutions `c` of `Σ cₖ imagesₖ ≡ 0`
/// (identically in `vars`), coefficients in `ℚ(other symbols)`.
pub(crate) fn nullspace_combinations(
    basis: &[ExprId],
    images: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Vec<ExprId> {
    let m = basis.len();
    let mut taken = names_in(images, pool);
    let cs: Vec<ExprId> = (0..m)
        .map(|_| fresh_symbol("__pde_c", &mut taken, pool))
        .collect();
    let s = pool.add(
        cs.iter()
            .zip(images)
            .map(|(&c, &im)| pool.mul(vec![c, im]))
            .collect(),
    );
    let mut gens0 = cs.clone();
    gens0.extend_from_slice(vars);
    let Ok((rf, gens)) = crate::poly::cancel::rational_over_generators(s, gens0, pool) else {
        return Vec::new();
    };
    // The denominator must not involve the unknown coefficients.
    for k in 0..m {
        if rf.denom.degree_in(k) > 0 {
            return Vec::new();
        }
    }
    // Generators that depend on the variables vs parameters.
    let var_dep: Vec<bool> = gens
        .iter()
        .enumerate()
        .map(|(idx, &g)| idx >= m && !free_of(g, vars, pool))
        .collect();
    let param_idx: Vec<usize> = (m..gens.len()).filter(|&i| !var_dep[i]).collect();
    let param_vars: Vec<ExprId> = param_idx.iter().map(|&i| gens[i]).collect();
    // key (exponents on variable-dependent generators) → per-c coefficient.
    let mut rows: BTreeMap<Vec<u32>, Vec<MultiPoly>> = BTreeMap::new();
    for (exps, coeff) in &rf.numer.terms {
        let get = |i: usize| exps.get(i).copied().unwrap_or(0);
        let cdeg: Vec<usize> = (0..m).filter(|&k| get(k) > 0).collect();
        if cdeg.len() != 1 || get(cdeg[0]) != 1 {
            return Vec::new();
        }
        let k = cdeg[0];
        let key: Vec<u32> = (m..gens.len()).filter(|&i| var_dep[i]).map(get).collect();
        let pexp: Vec<u32> = param_idx.iter().map(|&i| get(i)).collect();
        let row = rows
            .entry(key)
            .or_insert_with(|| vec![MultiPoly::zero(param_vars.clone()); m]);
        let mut term = MultiPoly::zero(param_vars.clone());
        let mut e = pexp;
        while e.last() == Some(&0) {
            e.pop();
        }
        term.terms.insert(e, coeff.clone());
        row[k] = add_poly(&row[k], &term);
    }
    let one = MultiPoly::constant(param_vars.clone(), 1);
    let mut mat: Vec<Vec<RationalFunction>> = Vec::new();
    for (_, r) in rows {
        let mut row = Vec::with_capacity(m);
        for p in r {
            match RationalFunction::new(p, one.clone()) {
                Ok(rf) => row.push(rf),
                Err(_) => return Vec::new(),
            }
        }
        mat.push(row);
    }
    let Some(null) = nullspace(mat, m, &param_vars) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for vec in null {
        let mut terms = Vec::new();
        for (k, c) in vec.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let ce = div(c.numer.to_expr(pool), c.denom.to_expr(pool), pool);
            terms.push(pool.mul(vec![ce, basis[k]]));
        }
        if !terms.is_empty() {
            out.push(simp(pool.add(terms), pool));
        }
    }
    out
}

fn add_poly(a: &MultiPoly, b: &MultiPoly) -> MultiPoly {
    let mut terms = a.terms.clone();
    for (e, c) in &b.terms {
        let entry = terms
            .entry(e.clone())
            .or_insert_with(|| rug::Integer::from(0));
        *entry += c;
        if *entry == 0 {
            terms.remove(e);
        }
    }
    MultiPoly {
        vars: a.vars.clone(),
        terms,
    }
}

/// Basis of the right nullspace of `mat` (`rows × m`) over `ℚ(params)`.
fn nullspace(
    mut mat: Vec<Vec<RationalFunction>>,
    m: usize,
    params: &[ExprId],
) -> Option<Vec<Vec<RationalFunction>>> {
    let zero = || {
        RationalFunction::new(
            MultiPoly::zero(params.to_vec()),
            MultiPoly::constant(params.to_vec(), 1),
        )
        .ok()
    };
    let one = || {
        RationalFunction::new(
            MultiPoly::constant(params.to_vec(), 1),
            MultiPoly::constant(params.to_vec(), 1),
        )
        .ok()
    };
    let rows = mat.len();
    let mut pivots: Vec<usize> = Vec::new();
    let mut r = 0;
    for c in 0..m {
        if r >= rows {
            break;
        }
        let Some(p) = (r..rows).find(|&i| !mat[i][c].is_zero()) else {
            continue;
        };
        mat.swap(r, p);
        let piv = mat[r][c].clone();
        for cell in mat[r].iter_mut() {
            *cell = (cell.clone() / piv.clone()).ok()?;
        }
        for i in 0..rows {
            if i == r || mat[i][c].is_zero() {
                continue;
            }
            let f = mat[i][c].clone();
            let pivot_row = mat[r].clone();
            for (cell, pr) in mat[i].iter_mut().zip(pivot_row) {
                let t = (f.clone() * pr).ok()?;
                *cell = (cell.clone() - t).ok()?;
            }
        }
        pivots.push(c);
        r += 1;
    }
    let free: Vec<usize> = (0..m).filter(|c| !pivots.contains(c)).collect();
    let mut out = Vec::new();
    for &fc in &free {
        let mut v: Vec<RationalFunction> = (0..m).map(|_| zero()).collect::<Option<_>>()?;
        v[fc] = one()?;
        for (ri, &pc) in pivots.iter().enumerate() {
            v[pc] = -mat[ri][fc].clone();
        }
        out.push(v);
    }
    Some(out)
}
