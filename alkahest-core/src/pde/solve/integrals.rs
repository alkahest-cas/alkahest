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
            Some(r) => is_zero_with_integrals(r, pool),
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
            for (i, &ci) in f.comps.iter().enumerate() {
                if ci == pool.integer(0) {
                    continue;
                }
                for j in 0..n {
                    if i == j || self.found.len() >= need {
                        continue;
                    }
                    for xi in self.pair_integral(i, j, pool) {
                        if self.offer(xi, "characteristic ODE", pool) {
                            progress = true;
                            break;
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
    fn pair_integral(&mut self, i: usize, j: usize, pool: &ExprPool) -> Vec<ExprId> {
        let f = self.field;
        let (zi, zj) = (f.vars[i], f.vars[j]);
        if f.comps[j] == pool.integer(0) {
            return Vec::new();
        }
        let mut ratio = simp(div(f.comps[j], f.comps[i], pool), pool);
        let (mut ci, mut cj) = (f.comps[i], f.comps[j]);
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
                ci = simp(subs1(ci, zk, zk_of, pool), pool);
                cj = simp(subs1(cj, zk, zk_of, pool), pool);
                back.insert(c, xi);
                done = true;
                break;
            }
            if !done {
                return Vec::new();
            }
        }
        if !f
            .vars
            .iter()
            .enumerate()
            .all(|(m, &zm)| m == i || m == j || !contains(ratio, zm, pool))
        {
            return Vec::new();
        }
        // dzⱼ/dzᵢ = ratio.
        // Candidate "constants" of dzⱼ/dzᵢ = ratio (each a first integral
        // once the eliminated variables are substituted back).  Closed forms
        // first, then the same with unevaluated integrals: those verify
        // exactly by the fundamental theorem of calculus even where the zero
        // test cannot see through a closed form.
        let mut constants: Vec<ExprId> = Vec::new();
        let push = |c: Option<ExprId>, out: &mut Vec<ExprId>| {
            if let Some(c) = c {
                let c = simp(c, pool);
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        };
        if !contains(ratio, zj, pool) {
            push(
                quad(ratio, zi, pool).map(|q| sub(zj, q, pool)),
                &mut constants,
            );
            push(
                unevaluated(ratio, zi, pool).map(|q| sub(zj, q, pool)),
                &mut constants,
            );
        } else if let Some((a, b)) =
            separate(ratio, zi, zj, pool).or_else(|| separate_by_value(ratio, zi, zj, pool))
        {
            // dzⱼ/dzᵢ = A(zᵢ)·B(zⱼ): ∫ dzⱼ/B − ∫ A dzᵢ.
            let rb = recip(b, pool);
            if let (Some(ib), Some(ia)) = (quad(rb, zj, pool), quad(a, zi, pool)) {
                push(Some(sub(ib, ia, pool)), &mut constants);
            }
            if let (Some(ib), Some(ia)) = (unevaluated(rb, zj, pool), unevaluated(a, zi, pool)) {
                push(Some(sub(ib, ia, pool)), &mut constants);
            }
        } else {
            for c in linear_or_bernoulli(ratio, zi, zj, pool) {
                push(Some(c), &mut constants);
            }
            for c in linear_argument(ratio, zi, zj, pool) {
                push(Some(c), &mut constants);
            }
            // cⱼ dzᵢ − cᵢ dzⱼ = 0: exact, or exact after μ(zᵢ) / μ(zⱼ).
            for c in exact_with_factor(cj, neg(ci, pool), zi, zj, pool) {
                push(Some(c), &mut constants);
            }
            // The same on the cancelled numerator / denominator of the ratio
            // (the field may have been normalised by a coefficient).
            if let Ok((nu, de)) = crate::poly::cancel::together_parts(ratio, vec![zi, zj], pool) {
                if de != ci || nu != cj {
                    for c in exact_with_factor(nu, neg(de, pool), zi, zj, pool) {
                        push(Some(c), &mut constants);
                    }
                }
            }
            if constants.is_empty() && !has_undefined(ratio, pool) {
                for c in super::riccati::riccati_integrals(ratio, zi, zj, pool) {
                    push(Some(c), &mut constants);
                }
            }
            if constants.is_empty() && !has_undefined(ratio, pool) {
                let (input, yp) = OdeInput::first_order(zi, zj, pool);
                let input = input.with_equation(sub(yp, ratio, pool));
                if let Ok(rep) = dsolve_with(&input, &AssumptionContext::new(), pool) {
                    for b in rep.branches {
                        let Some(&c1) = b.constants.first() else {
                            continue;
                        };
                        let rel = match b.form {
                            SolutionForm::Explicit(y) => sub(y, zj, pool),
                            SolutionForm::Implicit(g) => g,
                        };
                        if let Some(cv) = isolate_one(rel, c1, pool) {
                            push(Some(cv), &mut constants);
                            break;
                        }
                    }
                }
            }
        }
        constants
            .into_iter()
            .map(|c| simp(subs_map(c, &back, pool), pool))
            .collect()
    }
}

/// `dzⱼ/dzᵢ = P zⱼ + Q` (linear) or `P zⱼ + Q zⱼ^m` (Bernoulli, `m`
/// possibly symbolic): integrating factor, with unevaluated integrals where
/// the quadratures do not close.  Returns first-integral candidates.
fn linear_or_bernoulli(ratio: ExprId, zi: ExprId, zj: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    let ratio = crate::simplify::engine::simplify_expanded(ratio, pool).value;
    let terms = match pool.get(ratio) {
        crate::kernel::ExprData::Add(xs) => xs,
        _ => vec![ratio],
    };
    // Split into Q (free of zj), P·zj, and R·zj^m.
    let (mut p, mut q, mut r) = (Vec::new(), Vec::new(), Vec::new());
    let mut m_exp: Option<ExprId> = None;
    for t in terms {
        if !contains(t, zj, pool) {
            q.push(t);
            continue;
        }
        let Some(dt) = d(t, zj, pool) else {
            return out;
        };
        if !contains(dt, zj, pool) {
            p.push(dt);
            continue;
        }
        // t = c·zj^m ? (structurally: factors zj^e, the rest free of zj)
        let fs = match pool.get(t) {
            crate::kernel::ExprData::Mul(xs) => xs,
            _ => vec![t],
        };
        let mut es = Vec::new();
        for f in fs {
            if f == zj {
                es.push(pool.integer(1));
            } else if let crate::kernel::ExprData::Pow { base, exp } = pool.get(f) {
                if base == zj && !contains(exp, zj, pool) {
                    es.push(exp);
                } else if contains(f, zj, pool) {
                    return out;
                }
            } else if contains(f, zj, pool) {
                return out;
            }
        }
        let m = simp(pool.add(es), pool);
        if contains(m, zi, pool) {
            return out;
        }
        if let Some(prev) = m_exp {
            if !is_zero(sub(prev, m, pool), pool) {
                return out;
            }
        }
        m_exp = Some(m);
        r.push(simp(div(t, pool.pow(zj, m), pool), pool));
    }
    let pz = simp(pool.add(p), pool);
    let (w, src, scale) = match (m_exp, q.is_empty()) {
        (None, _) => (zj, simp(pool.add(q), pool), pool.integer(1)),
        (Some(m), true) => {
            // w = zj^(1−m): w' = (1−m) P w + (1−m) R
            let one_m = simp(sub(pool.integer(1), m, pool), pool);
            if is_zero(one_m, pool) {
                return out;
            }
            (
                pool.pow(zj, one_m),
                simp(pool.mul(vec![one_m, pool.add(r)]), pool),
                one_m,
            )
        }
        _ => return out,
    };
    let pz = simp(pool.mul(vec![scale, pz]), pool);
    if contains(pz, zj, pool) || contains(src, zj, pool) {
        return out;
    }
    // ξ = w·e^{−∫P} − ∫ src·e^{−∫P}
    for unev in [false, true] {
        let ip = if unev {
            unevaluated(pz, zi, pool)
        } else {
            quad(pz, zi, pool)
        };
        let Some(ip) = ip else { continue };
        let mu = simp(pool.func("exp", vec![neg(ip, pool)]), pool);
        let integrand = pool.mul(vec![src, mu]);
        let iq = if unev {
            unevaluated(integrand, zi, pool)
        } else {
            quad(integrand, zi, pool)
        };
        let Some(iq) = iq else { continue };
        out.push(simp(sub(pool.mul(vec![w, mu]), iq, pool), pool));
    }
    out
}

/// `dzⱼ/dzᵢ = F(κ zᵢ + zⱼ)`: with `L = κ zᵢ + zⱼ`, `dL/dzᵢ = κ + F(L)` is
/// separable, giving `∫ dL/(κ + F(L)) − zᵢ`.
fn linear_argument(ratio: ExprId, zi: ExprId, zj: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    let (Some(ri), Some(rj)) = (d(ratio, zi, pool), d(ratio, zj, pool)) else {
        return out;
    };
    if is_zero(rj, pool) || is_zero(ri, pool) {
        return out;
    }
    let kappa = simp(div(ri, rj, pool), pool);
    if contains(kappa, zi, pool) || contains(kappa, zj, pool) {
        return out;
    }
    let l = crate::pde::integral::fresh_dummy("_L", &[ratio, zi, zj], pool);
    // zⱼ = L − κ zᵢ
    let fl = simp(
        subs1(ratio, zj, sub(l, pool.mul(vec![kappa, zi]), pool), pool),
        pool,
    );
    let fl = super::tidy(
        crate::simplify::engine::simplify_expanded(fl, pool).value,
        pool,
    );
    if contains(fl, zi, pool) {
        return out;
    }
    let integrand = recip(pool.add(vec![kappa, fl]), pool);
    let lv = pool.add(vec![pool.mul(vec![kappa, zi]), zj]);
    for q in [quad(integrand, l, pool), unevaluated(integrand, l, pool)]
        .into_iter()
        .flatten()
    {
        out.push(simp(sub(subs1(q, l, lv, pool), zi, pool), pool));
    }
    out
}

/// `M dzᵢ + N dzⱼ = 0`: the potential when exact, or after an integrating
/// factor depending on `zᵢ` or `zⱼ` alone.
fn exact_with_factor(m: ExprId, n: ExprId, zi: ExprId, zj: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    let (Some(my), Some(nx)) = (d(m, zj, pool), d(n, zi, pool)) else {
        return out;
    };
    let diff = simp(sub(my, nx, pool), pool);
    let mut factors: Vec<ExprId> = Vec::new();
    if is_zero(diff, pool) {
        factors.push(pool.integer(1));
    } else {
        // μ(zᵢ): μ′/μ = (M_y − N_x)/N;  μ(zⱼ): μ′/μ = (N_x − M_y)/M.
        let gi = simp(div(diff, n, pool), pool);
        if !contains(gi, zj, pool) {
            if let Some(ig) = integrate(gi, zi, pool) {
                factors.push(simp(pool.func("exp", vec![ig]), pool));
            }
        }
        let gj = simp(div(neg(diff, pool), m, pool), pool);
        if !contains(gj, zi, pool) {
            if let Some(ig) = integrate(gj, zj, pool) {
                factors.push(simp(pool.func("exp", vec![ig]), pool));
            }
        }
    }
    for mu in factors {
        let mm = simp(pool.mul(vec![mu, m]), pool);
        let nn = simp(pool.mul(vec![mu, n]), pool);
        let Some(p) = quad(mm, zi, pool) else {
            continue;
        };
        let Some(dp) = d(p, zj, pool) else { continue };
        let rest = simp(sub(nn, dp, pool), pool);
        let rest = crate::simplify::engine::simplify_expanded(rest, pool).value;
        if contains(rest, zi, pool) && !is_zero(d(rest, zi, pool).unwrap_or(rest), pool) {
            continue;
        }
        let rest = simp(subs1(rest, zi, pool.integer(1), pool), pool);
        let Some(q) = quad(rest, zj, pool) else {
            continue;
        };
        out.push(simp(pool.add(vec![p, q]), pool));
    }
    out
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
    if a.iter().all(|&f| !contains(f, zj, pool)) && b.iter().all(|&f| !contains(f, zi, pool)) {
        return Some((simp(pool.mul(a), pool), simp(pool.mul(b), pool)));
    }
    None
}

/// `ratio = A(zᵢ)·B(zⱼ)` for a ratio that is not a product of such factors
/// (`(y + 1)·f(x)` written as a sum): `B = ratio(zᵢ := c)`,
/// `A = ratio / B`, accepted when `A` is free of `zⱼ`.
fn separate_by_value(
    ratio: ExprId,
    zi: ExprId,
    zj: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    for c in [1i64, 2, 0] {
        let b = simp(subs1(ratio, zi, pool.integer(c), pool), pool);
        if is_zero(b, pool) || !contains(b, zj, pool) {
            continue;
        }
        let a = simp(div(ratio, b, pool), pool);
        let a = crate::poly::cancel::cancel(a, vec![zi, zj], pool).unwrap_or(a);
        if !contains(a, zj, pool) {
            return Some((a, b));
        }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// Local debugging aid (`-- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn debug_linear_characteristic() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let mut syms = std::collections::HashMap::new();
        syms.insert("x".to_string(), x);
        syms.insert("y".to_string(), y);
        syms.insert("f".to_string(), pool.func("f", vec![x]));
        syms.insert("g".to_string(), pool.func("g", vec![x]));
        for src in ["y*f(x) + g(x)", "y^2 - a^2*x^2 + a"] {
            let ratio = crate::parse::parse(src, &pool, &mut syms).unwrap();
            let field = VectorField::new(vec![x, y], vec![pool.integer(1), ratio], &pool);
            let cands = if src.contains("f(") {
                linear_or_bernoulli(ratio, x, y, &pool)
            } else {
                super::super::riccati::riccati_integrals(ratio, x, y, &pool)
            };
            for c in cands {
                let r = field.apply(c, &pool).unwrap();
                eprintln!(
                    "CAND {}\n  V = {}\n  zero = {}",
                    pool.display(c),
                    pool.display(r),
                    field.is_integral(c, &pool)
                );
                let mut search = IntegralSearch::new(&field, Default::default());
                eprintln!("  offer = {}", search.offer(c, "test", &pool));
                let row = field.jacobian_row(c, &pool).unwrap();
                let inst: Vec<ExprId> = row
                    .iter()
                    .map(|&e| instantiate_undefined(e, &pool))
                    .collect();
                for e in &inst {
                    let syms = free_symbols(&[*e], &pool);
                    eprintln!(
                        "  row {} -> {:?}",
                        pool.display(*e),
                        eval_at(*e, &sample_env(&syms, 0), &pool)
                    );
                }
            }
        }
    }

    #[test]
    fn linear_characteristic_with_arbitrary_coefficients() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let f = pool.func("f", vec![x]);
        let g = pool.func("g", vec![x]);
        let ratio = pool.add(vec![pool.mul(vec![y, f]), g]);
        let field = VectorField::new(vec![x, y], vec![pool.integer(1), ratio], &pool);
        let cands = linear_or_bernoulli(ratio, x, y, &pool);
        assert!(cands.iter().any(|&c| field.is_integral(c, &pool)));
    }
}
