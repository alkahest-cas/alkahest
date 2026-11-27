//! `classify_pde_system`: order, linearity and — for first-order square
//! systems — hyperbolicity / ellipticity from the principal symbol, plus
//! named-system matching (shallow water, 1-D Euler, Cauchy–Riemann, 1-D
//! Maxwell).
//!
//! # Principal symbol
//!
//! A first-order system `Fᵢ(x, U, ∂U) = 0` in `m` unknowns has the
//! coefficient matrices `(A_k)ᵢⱼ = ∂Fᵢ/∂(∂_{x_k} uⱼ)` — for a quasilinear
//! system `Σ_k A_k(x, U) ∂_{x_k} U + r(x, U) = 0` exactly the matrices of the
//! derivatives.  Its principal symbol is `P(ξ) = Σ_k ξ_k A_k` and the
//! **characteristic determinant** is `det P(ξ)` ([`PdeSystemInfo::characteristic_determinant`],
//! a polynomial in fresh symbols `ξ_x, ξ_t, …`).
//!
//! # Type (square systems)
//!
//! An *evolution variable* `t` is one whose matrix `A_t` is invertible
//! (preferring a variable named `t`).  The **characteristic speeds** are the
//! roots `λ` of `det(B − λ A_t) = 0`, `B = Σ_{k≠t} ξ_k A_k` (with `ξ = 1` in
//! two variables, so `λ` are the eigenvalues of `A_t⁻¹ A_x`).  They are
//! computed **exactly**: the determinant's numerator is factored over `ℤ`
//! with every other symbol a parameter, and the linear and quadratic factors
//! are solved in closed form.  Then, in two independent variables,
//!
//! * every speed real and the matrix diagonalisable → **hyperbolic**
//!   ([`PdeSystemInfo::strictly_hyperbolic`] when the speeds are distinct);
//!   a repeated speed is accepted only when the rank of `B − λA_t` is proved
//!   to drop by its multiplicity;
//! * no real speed → **elliptic** (Cauchy–Riemann);
//! * a quadratic factor whose discriminant `q` (normalised to
//!   `(c₁² − 4c₀c₂)/(4c₂²)`, so the speeds are `−c₁/(2c₂) ± √q`) has no
//!   provable sign: when `q` is a product of powers of symbols and unknowns
//!   with a positive coefficient (`g·h`, `γ·p/ρ`) the system is reported
//!   hyperbolic **under the condition `q > 0`** (in
//!   [`PdeSystemInfo::type_conditions`]); when it depends on the independent
//!   variables the type is mixed with [`PdeSystemInfo::regions`]; otherwise
//!   undetermined with regions.
//!
//! In more than two independent variables the speeds are still computed (they
//! depend on `ξ`), but the type is only asserted by the **symmetric
//! hyperbolic** criterion of Friedrichs: `A_t` a constant positive diagonal
//! matrix and every other `A_k` symmetric.  Anything else is `Undetermined`;
//! nothing is guessed.
//!
//! Every returned expression is written back in the unknowns
//! (`h(x, t)`, not the jet symbol `h`).

#![allow(clippy::needless_range_loop)]

use std::collections::{HashMap, HashSet};

use crate::diff::diff_impl::diff;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::{simplify, simplify_expanded};

use super::classify::{
    is_zero, region, sign_of, Linearity, NamedEquation, PdeType, Sign, TypeRegion,
};
use super::error::PdeError;
use super::jet::{decode_unknown, symbols_of, JetSpace};

/// Largest number of unknowns for which the type of a first-order system is
/// analysed (determinants are expanded by minors).
pub const MAX_SYSTEM_TYPE_SIZE: usize = 6;

/// Result of [`classify_pde_system`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PdeSystemInfo {
    /// The unknowns `u(x, t), v(x, t), …`.
    pub unknowns: Vec<ExprId>,
    /// The independent variables, in order of first appearance among the
    /// unknowns' arguments.
    pub independent: Vec<ExprId>,
    /// The equations in jet coordinates, each read as `= 0`.
    pub equations_in_jets: Vec<ExprId>,
    /// Highest derivative order of the system.
    pub order: u32,
    /// Highest derivative order of each equation.
    pub orders: Vec<u32>,
    /// Linearity class of the system (the weakest over the equations, with
    /// "highest order" meaning the system's order).
    pub linearity: Linearity,
    /// For a linear system, whether `U ≡ 0` solves it.
    pub homogeneous: Option<bool>,
    /// For a linear system, whether every coefficient is free of the
    /// independent variables.
    pub constant_coefficients: Option<bool>,
    /// Hyperbolic / elliptic / mixed / … for a first-order square system;
    /// `None` otherwise.
    pub pde_type: Option<PdeType>,
    /// For a hyperbolic system: whether the characteristic speeds are
    /// pairwise distinct (`None` when the type is not hyperbolic or not
    /// decided).
    pub strictly_hyperbolic: Option<bool>,
    /// Conditions under which [`Self::pde_type`] holds (`g·h(x, t) > 0` for
    /// shallow water, `det A_t ≠ 0`).
    pub type_conditions: Vec<ExprId>,
    /// Regions of constant type when the type varies or is undetermined.
    pub regions: Vec<TypeRegion>,
    /// The evolution variable `t` (an independent variable with `A_t`
    /// invertible).
    pub evolution_variable: Option<ExprId>,
    /// First-order systems: the coefficient matrices `A_k` (one per
    /// independent variable, rows = equations, columns = unknowns).
    pub coefficient_matrices: Vec<Vec<Vec<ExprId>>>,
    /// The symbols `ξ_k` of the principal symbol, one per independent
    /// variable.
    pub symbol_variables: Vec<ExprId>,
    /// `P(ξ) = Σ_k ξ_k A_k`.
    pub principal_symbol: Option<Vec<Vec<ExprId>>>,
    /// `det P(ξ)`.
    pub characteristic_determinant: Option<ExprId>,
    /// In two independent variables, `A_t⁻¹ A_x` (the characteristic
    /// matrix whose eigenvalues are the speeds).
    pub characteristic_matrix: Option<Vec<Vec<ExprId>>>,
    /// The characteristic speeds `λ` (roots of `det(B − λA_t)`) with their
    /// multiplicities, when every factor of the determinant could be solved.
    pub characteristic_speeds: Vec<(ExprId, usize)>,
    /// Matched named systems.
    pub named: Vec<NamedEquation>,
}

fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify(e, pool).value
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}

fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}

fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    simp(pool.mul(vec![a, pool.pow(b, pool.integer(-1))]), pool)
}

/// Rational-function normal form (numerator and denominator cancelled) when
/// `e` is rational in its symbols, else `simplify_expanded`.
fn norm(e: ExprId, pool: &ExprPool) -> ExprId {
    let e = simplify_expanded(e, pool).value;
    let vars = crate::poly::collect_free_vars(e, pool);
    match crate::poly::cancel::cancel(e, vars, pool) {
        Ok(c) => simp(c, pool),
        Err(_) => e,
    }
}

fn mentions_any(e: ExprId, set: &HashSet<ExprId>, pool: &ExprPool) -> bool {
    symbols_of(e, pool).iter().any(|s| set.contains(s))
}

fn d(e: ExprId, v: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    diff(e, v, pool)
        .map(|r| simp(r.value, pool))
        .map_err(|e| PdeError::NotRationalInJets(e.to_string()))
}

fn pred(kind: PredicateKind, a: ExprId, pool: &ExprPool) -> ExprId {
    pool.predicate(kind, vec![a, pool.integer(0)])
}

/// `det m` by expansion along the first row (`m` square, small).
pub(crate) fn det(m: &[Vec<ExprId>], pool: &ExprPool) -> ExprId {
    let n = m.len();
    if n == 0 {
        return pool.integer(1);
    }
    if n == 1 {
        return m[0][0];
    }
    if n == 2 {
        return simp(
            sub(
                pool.mul(vec![m[0][0], m[1][1]]),
                pool.mul(vec![m[0][1], m[1][0]]),
                pool,
            ),
            pool,
        );
    }
    let zero = pool.integer(0);
    let mut terms = Vec::new();
    for j in 0..n {
        if m[0][j] == zero {
            continue;
        }
        let minor: Vec<Vec<ExprId>> = m[1..]
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .filter(|&(k, _)| k != j)
                    .map(|(_, &x)| x)
                    .collect()
            })
            .collect();
        let t = pool.mul(vec![m[0][j], det(&minor, pool)]);
        terms.push(if j % 2 == 0 { t } else { neg(t, pool) });
    }
    simp(pool.add(terms), pool)
}

/// `m⁻¹` by the adjugate, when `det m` is not proved zero.
fn inverse(m: &[Vec<ExprId>], pool: &ExprPool) -> Option<Vec<Vec<ExprId>>> {
    let n = m.len();
    let dm = det(m, pool);
    if is_zero(dm, pool) {
        return None;
    }
    let mut inv = vec![vec![pool.integer(0); n]; n];
    for i in 0..n {
        for j in 0..n {
            // cofactor C_ji
            let minor: Vec<Vec<ExprId>> = m
                .iter()
                .enumerate()
                .filter(|&(r, _)| r != j)
                .map(|(_, row)| {
                    row.iter()
                        .enumerate()
                        .filter(|&(c, _)| c != i)
                        .map(|(_, &x)| x)
                        .collect()
                })
                .collect();
            let c = det(&minor, pool);
            let c = if (i + j) % 2 == 0 { c } else { neg(c, pool) };
            inv[i][j] = div(c, dm, pool);
        }
    }
    Some(inv)
}

fn matmul(a: &[Vec<ExprId>], b: &[Vec<ExprId>], pool: &ExprPool) -> Vec<Vec<ExprId>> {
    let n = a.len();
    let k = b.len();
    let m = if k == 0 { 0 } else { b[0].len() };
    (0..n)
        .map(|i| {
            (0..m)
                .map(|j| {
                    simp(
                        pool.add((0..k).map(|l| pool.mul(vec![a[i][l], b[l][j]])).collect()),
                        pool,
                    )
                })
                .collect()
        })
        .collect()
}

/// Whether every minor of size `size` of `m` is proved zero.
fn minors_vanish(m: &[Vec<ExprId>], size: usize, pool: &ExprPool) -> bool {
    fn combos(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut cur = Vec::new();
        fn rec(s: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if cur.len() == k {
                out.push(cur.clone());
                return;
            }
            for i in s..n {
                cur.push(i);
                rec(i + 1, n, k, cur, out);
                cur.pop();
            }
        }
        rec(0, n, k, &mut cur, &mut out);
        out
    }
    let n = m.len();
    if size == 0 || size > n {
        return size > n;
    }
    for rows in combos(n, size) {
        for cols in combos(n, size) {
            let sub: Vec<Vec<ExprId>> = rows
                .iter()
                .map(|&r| cols.iter().map(|&c| m[r][c]).collect())
                .collect();
            if !is_zero(det(&sub, pool), pool) {
                return false;
            }
        }
    }
    true
}

/// A product of powers of symbols (and of positive numbers) with a positive
/// numeric coefficient: positive whenever its symbols are.
fn positive_monomial(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Symbol { .. } => true,
        ExprData::Integer(n) => n.0.cmp0() == std::cmp::Ordering::Greater,
        ExprData::Rational(r) => r.0.cmp0() == std::cmp::Ordering::Greater,
        ExprData::Pow { base, exp } => {
            positive_monomial(base, pool)
                && pool.with(exp, |d| {
                    matches!(d, ExprData::Integer(_) | ExprData::Rational(_))
                })
        }
        ExprData::Mul(xs) => xs.iter().all(|&x| positive_monomial(x, pool)),
        _ => false,
    }
}

/// One factor of the characteristic polynomial, solved.
struct Factor {
    roots: Vec<ExprId>,
    multiplicity: usize,
    /// For a quadratic factor, the normalised discriminant `q`.
    q: Option<ExprId>,
    /// `false` when the factor could not be solved in closed form.
    solved: bool,
}

/// Factor `p(λ)` (numerator taken) over `ℤ` with the other symbols as
/// parameters and solve its linear and quadratic factors.
fn solve_char_poly(p: ExprId, lam: ExprId, pool: &ExprPool) -> Vec<Factor> {
    let num = crate::poly::cancel::together_parts(p, vec![lam], pool)
        .map(|(n, _)| n)
        .unwrap_or(p);
    let num = simplify_expanded(num, pool).value;
    let mut factors: Vec<(ExprId, usize)> = Vec::new();
    let mut vars = vec![lam];
    vars.extend(
        crate::poly::collect_free_vars(num, pool)
            .into_iter()
            .filter(|&v| v != lam),
    );
    match crate::poly::MultiPoly::from_symbolic(num, vars, pool)
        .ok()
        .and_then(|mp| mp.factor_z().ok())
    {
        Some(fac) => {
            for (f, e) in fac.factors {
                if f.degree_in(0) > 0 {
                    factors.push((f.to_expr(pool), e as usize));
                }
            }
        }
        None => factors.push((num, 1)),
    }
    let zero = pool.integer(0);
    let mut out = Vec::new();
    for (f, mult) in factors {
        let at0 = |e: ExprId| {
            simp(
                crate::kernel::subs::subs(e, &HashMap::from([(lam, zero)]), pool),
                pool,
            )
        };
        let Ok(f1) = d(f, lam, pool) else {
            out.push(Factor {
                roots: vec![],
                multiplicity: mult,
                q: None,
                solved: false,
            });
            continue;
        };
        let Ok(f2) = d(f1, lam, pool) else {
            out.push(Factor {
                roots: vec![],
                multiplicity: mult,
                q: None,
                solved: false,
            });
            continue;
        };
        let f3 = d(f2, lam, pool).ok();
        let c0 = at0(f);
        let c1 = at0(f1);
        if is_zero(f2, pool) && !symbols_of(f1, pool).contains(&lam) {
            // Linear.
            out.push(Factor {
                roots: vec![norm(neg(div(c0, c1, pool), pool), pool)],
                multiplicity: mult,
                q: None,
                solved: true,
            });
            continue;
        }
        if f3.is_some_and(|f3| is_zero(f3, pool)) {
            let c2 = norm(pool.mul(vec![pool.rational(1, 2), f2]), pool);
            let c1 = norm(c1, pool);
            let c0 = norm(c0, pool);
            let q = norm(
                div(
                    sub(
                        pool.pow(c1, pool.integer(2)),
                        pool.mul(vec![pool.integer(4), c0, c2]),
                        pool,
                    ),
                    pool.mul(vec![pool.integer(4), pool.pow(c2, pool.integer(2))]),
                    pool,
                ),
                pool,
            );
            let centre = norm(
                neg(div(c1, pool.mul(vec![pool.integer(2), c2]), pool), pool),
                pool,
            );
            let roots = if is_zero(q, pool) {
                vec![centre, centre]
            } else {
                let s = simp(pool.func("sqrt", vec![q]), pool);
                vec![
                    simp(pool.add(vec![centre, neg(s, pool)]), pool),
                    simp(pool.add(vec![centre, s]), pool),
                ]
            };
            out.push(Factor {
                roots,
                multiplicity: mult,
                q: Some(q),
                solved: true,
            });
            continue;
        }
        out.push(Factor {
            roots: vec![],
            multiplicity: mult,
            q: None,
            solved: false,
        });
    }
    out
}

/// Classify the system `equations = 0` in `unknowns`.  See the
/// [module documentation](self).
///
/// # Errors
///
/// `E-PDE-001` / `E-PDE-002` for malformed unknowns; `E-PDE-008` for an empty
/// system; `E-PDE-007` when an equation has no derivative of any unknown.
pub fn classify_pde_system(
    equations: &[ExprId],
    unknowns: &[ExprId],
    pool: &ExprPool,
) -> Result<PdeSystemInfo, PdeError> {
    if equations.is_empty() || unknowns.is_empty() {
        return Err(PdeError::SystemShape(
            "a system needs at least one equation and one unknown".into(),
        ));
    }
    for &u in unknowns {
        decode_unknown(u, pool)?;
    }
    let big = JetSpace::from_unknowns(unknowns, u32::MAX / 4, pool)?;
    let indep = big.independent().to_vec();
    let mut eqs = Vec::with_capacity(equations.len());
    let mut coords_all = Vec::new();
    let mut orders = Vec::new();
    for &e in equations {
        let f = simp(big.to_jets(e, pool)?, pool);
        let cs = big.jets_in(f, pool)?;
        let o = cs
            .iter()
            .map(|c| c.alpha.iter().sum::<u32>())
            .max()
            .unwrap_or(0);
        if o == 0 {
            return Err(PdeError::NotADifferentialEquation(format!(
                "{} has no derivative of the unknowns",
                pool.display(e)
            )));
        }
        orders.push(o);
        coords_all.extend(cs);
        eqs.push(f);
    }
    let order = *orders.iter().max().unwrap();
    let jet_set: HashSet<ExprId> = coords_all.iter().map(|c| c.symbol).collect();
    let top_set: HashSet<ExprId> = coords_all
        .iter()
        .filter(|c| c.alpha.iter().sum::<u32>() == order)
        .map(|c| c.symbol)
        .collect();
    let mut coords: Vec<_> = coords_all.clone();
    coords.sort_by_key(|c| c.symbol);
    coords.dedup_by_key(|c| c.symbol);

    let mut linear = true;
    let mut semilinear = true;
    let mut quasilinear = true;
    let mut partials: Vec<Vec<(ExprId, ExprId)>> = Vec::new();
    for &f in &eqs {
        let mut ps = Vec::new();
        for c in &coords {
            let p = d(f, c.symbol, pool)?;
            let is_top = c.alpha.iter().sum::<u32>() == order;
            if mentions_any(p, &jet_set, pool) {
                linear = false;
                if is_top {
                    semilinear = false;
                }
            }
            if is_top && mentions_any(p, &top_set, pool) {
                quasilinear = false;
            }
            ps.push((c.symbol, p));
        }
        partials.push(ps);
    }
    let linearity = if linear {
        Linearity::Linear
    } else if semilinear {
        Linearity::Semilinear
    } else if quasilinear {
        Linearity::Quasilinear
    } else {
        Linearity::FullyNonlinear
    };
    let zero = pool.integer(0);
    let at_zero: HashMap<ExprId, ExprId> = jet_set.iter().map(|&s| (s, zero)).collect();
    let indep_set: HashSet<ExprId> = indep.iter().copied().collect();
    let (homogeneous, constant_coefficients) = if linear {
        (
            Some(
                eqs.iter()
                    .all(|&f| is_zero(crate::kernel::subs::subs(f, &at_zero, pool), pool)),
            ),
            Some(
                partials
                    .iter()
                    .flatten()
                    .all(|(_, p)| !mentions_any(*p, &indep_set, pool)),
            ),
        )
    } else {
        (None, None)
    };

    let mut info = PdeSystemInfo {
        unknowns: unknowns.to_vec(),
        independent: indep.clone(),
        equations_in_jets: eqs.clone(),
        order,
        orders,
        linearity,
        homogeneous,
        constant_coefficients,
        pde_type: None,
        strictly_hyperbolic: None,
        type_conditions: Vec::new(),
        regions: Vec::new(),
        evolution_variable: None,
        coefficient_matrices: Vec::new(),
        symbol_variables: Vec::new(),
        principal_symbol: None,
        characteristic_determinant: None,
        characteristic_matrix: None,
        characteristic_speeds: Vec::new(),
        named: Vec::new(),
    };
    if order == 1 {
        let space = big.prolonged(0).with_max_order(1);
        first_order(&mut info, &space, &jet_set, pool)?;
        // Everything returned is written in the unknowns again.
        let back = |e: ExprId| space.from_jets(e, pool);
        let back_m = |m: &Vec<Vec<ExprId>>| -> Vec<Vec<ExprId>> {
            m.iter()
                .map(|r| r.iter().map(|&e| back(e)).collect())
                .collect()
        };
        info.coefficient_matrices = info.coefficient_matrices.iter().map(back_m).collect();
        info.principal_symbol = info.principal_symbol.as_ref().map(back_m);
        info.characteristic_matrix = info.characteristic_matrix.as_ref().map(back_m);
        info.characteristic_determinant = info.characteristic_determinant.map(back);
        info.type_conditions = info.type_conditions.iter().map(|&e| back(e)).collect();
        info.characteristic_speeds = info
            .characteristic_speeds
            .iter()
            .map(|&(e, m)| (back(e), m))
            .collect();
        for r in info.regions.iter_mut() {
            r.condition = back(r.condition);
        }
        for n in info.named.iter_mut() {
            for p in n.parameters.iter_mut() {
                p.1 = back(p.1);
            }
            n.conditions = n.conditions.iter().map(|&e| back(e)).collect();
        }
    }
    Ok(info)
}

fn xi_symbols(indep: &[ExprId], taken: &HashSet<String>, pool: &ExprPool) -> Vec<ExprId> {
    indep
        .iter()
        .map(|&x| {
            let base = pool.with(x, |d| match d {
                ExprData::Symbol { name, .. } => name.clone(),
                _ => "x".to_string(),
            });
            let mut name = format!("xi_{base}");
            while taken.contains(&name) {
                name.push('_');
            }
            pool.symbol(name, Domain::Real)
        })
        .collect()
}

fn names_in(exprs: &[ExprId], pool: &ExprPool) -> HashSet<String> {
    let mut out = HashSet::new();
    for &e in exprs {
        for s in symbols_of(e, pool) {
            pool.with(s, |d| {
                if let ExprData::Symbol { name, .. } = d {
                    out.insert(name.clone());
                }
            });
        }
    }
    out
}

fn first_order(
    info: &mut PdeSystemInfo,
    space: &JetSpace,
    jet_set: &HashSet<ExprId>,
    pool: &ExprPool,
) -> Result<(), PdeError> {
    let n_eq = info.equations_in_jets.len();
    let m = info.unknowns.len();
    let nv = info.independent.len();
    let zero = pool.integer(0);
    // A_k[i][j] = ∂F_i / ∂(u_j)_{x_k}  (zero when u_j does not depend on x_k).
    let mut mats = Vec::with_capacity(nv);
    for k in 0..nv {
        let mut a = vec![vec![zero; m]; n_eq];
        for (i, &f) in info.equations_in_jets.iter().enumerate() {
            for j in 0..m {
                if !space.depends_on(j, k) {
                    continue;
                }
                let mut alpha = vec![0u32; nv];
                alpha[k] = 1;
                let p = space.jet(j, &alpha, pool)?;
                a[i][j] = d(f, p, pool)?;
            }
        }
        mats.push(a);
    }
    let first_jets: HashSet<ExprId> = jet_set
        .iter()
        .copied()
        .filter(|&s| {
            space
                .decode_jet(s, pool)
                .is_some_and(|(_, a)| a.iter().sum::<u32>() == 1)
        })
        .collect();
    let mut taken = names_in(&info.equations_in_jets, pool);
    for &x in &info.independent {
        taken.extend(names_in(&[x], pool));
    }
    let xis = xi_symbols(&info.independent, &taken, pool);
    info.coefficient_matrices = mats.clone();
    info.symbol_variables = xis.clone();
    let sym: Vec<Vec<ExprId>> = (0..n_eq)
        .map(|i| {
            (0..m)
                .map(|j| {
                    simp(
                        pool.add(
                            (0..nv)
                                .map(|k| pool.mul(vec![xis[k], mats[k][i][j]]))
                                .collect(),
                        ),
                        pool,
                    )
                })
                .collect()
        })
        .collect();
    info.principal_symbol = Some(sym.clone());
    if n_eq != m || m > MAX_SYSTEM_TYPE_SIZE {
        return Ok(());
    }
    info.characteristic_determinant = Some(det(&sym, pool));
    // A principal symbol depending on first derivatives: fully nonlinear.
    if sym
        .iter()
        .flatten()
        .any(|&e| mentions_any(e, &first_jets, pool))
    {
        info.pde_type = Some(PdeType::SolutionDependent);
        return Ok(());
    }
    // Evolution variable.
    let mut order_k: Vec<usize> = (0..nv).collect();
    order_k.sort_by_key(|&k| {
        let is_t = pool.with(
            info.independent[k],
            |d| matches!(d, ExprData::Symbol { name, .. } if name == "t" || name == "tau"),
        );
        !is_t
    });
    let mut t_idx = None;
    for &k in &order_k {
        let dk = det(&mats[k], pool);
        if !is_zero(dk, pool) {
            t_idx = Some(k);
            if !matches!(sign_of(dk, pool), Some(Sign::Pos | Sign::Neg)) {
                info.type_conditions.push(pred(PredicateKind::Ne, dk, pool));
            }
            break;
        }
    }
    let Some(t) = t_idx else {
        info.pde_type = Some(PdeType::Undetermined);
        return Ok(());
    };
    info.evolution_variable = Some(info.independent[t]);
    let at = &mats[t];
    // B = Σ_{k≠t} ξ_k A_k  (ξ = 1 in two variables).
    let others: Vec<usize> = (0..nv).filter(|&k| k != t).collect();
    let b: Vec<Vec<ExprId>> = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| {
                    simp(
                        pool.add(
                            others
                                .iter()
                                .map(|&k| {
                                    if nv == 2 {
                                        mats[k][i][j]
                                    } else {
                                        pool.mul(vec![xis[k], mats[k][i][j]])
                                    }
                                })
                                .collect(),
                        ),
                        pool,
                    )
                })
                .collect()
        })
        .collect();
    let lam = {
        let mut name = "lambda".to_string();
        while taken.contains(&name) {
            name.push('_');
        }
        pool.symbol(name, Domain::Real)
    };
    let pencil: Vec<Vec<ExprId>> = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| simp(sub(b[i][j], pool.mul(vec![lam, at[i][j]]), pool), pool))
                .collect()
        })
        .collect();
    let charpoly = det(&pencil, pool);
    let factors = solve_char_poly(charpoly, lam, pool);
    let all_solved = factors.iter().all(|f| f.solved);
    let degree: usize = factors
        .iter()
        .map(|f| f.multiplicity * f.roots.len().max(1))
        .sum();
    if all_solved && degree == m {
        let mut speeds: Vec<(ExprId, usize)> = Vec::new();
        for f in &factors {
            for &r in &f.roots {
                if let Some(e) = speeds.iter_mut().find(|(s, _)| *s == r) {
                    e.1 += f.multiplicity;
                } else {
                    speeds.push((r, f.multiplicity));
                }
            }
        }
        info.characteristic_speeds = speeds;
    }
    if nv == 2 {
        if let Some(inv) = inverse(at, pool) {
            let cm = matmul(&inv, &b, pool);
            info.characteristic_matrix = Some(cm.clone());
            info.named = match_named(info, &cm, t, space, jet_set, pool);
        }
        decide_type_2d(info, &factors, all_solved && degree == m, &b, at, pool);
    } else {
        symmetric_hyperbolic(info, &mats, t, pool);
    }
    Ok(())
}

fn decide_type_2d(
    info: &mut PdeSystemInfo,
    factors: &[Factor],
    complete: bool,
    b: &[Vec<ExprId>],
    at: &[Vec<ExprId>],
    pool: &ExprPool,
) {
    if !complete {
        info.pde_type = Some(PdeType::Undetermined);
        return;
    }
    let indep_set: HashSet<ExprId> = info.independent.iter().copied().collect();
    let mut complex_count = 0usize;
    let mut conditions = Vec::new();
    let mut unknown_q: Vec<ExprId> = Vec::new();
    let mut distinct = true;
    for f in factors {
        if f.multiplicity > 1 {
            distinct = false;
        }
        match f.q {
            None => {}
            Some(q) => match sign_of(q, pool) {
                Some(Sign::Pos) => {}
                Some(Sign::Neg) => complex_count += 2 * f.multiplicity,
                Some(Sign::Zero) => {
                    distinct = false;
                }
                Some(Sign::NonNeg) => {
                    conditions.push(pred(PredicateKind::Ne, q, pool));
                }
                _ if is_zero(q, pool) => {
                    distinct = false;
                }
                _ if positive_monomial(q, pool) && !mentions_any(q, &indep_set, pool) => {
                    conditions.push(pred(PredicateKind::Gt, q, pool));
                }
                _ => unknown_q.push(q),
            },
        }
    }
    // Repeated roots across factors.
    let roots: Vec<ExprId> = factors
        .iter()
        .flat_map(|f| f.roots.iter().copied())
        .collect();
    for i in 0..roots.len() {
        for j in i + 1..roots.len() {
            if is_zero(sub(roots[i], roots[j], pool), pool) {
                distinct = false;
            }
        }
    }
    if !unknown_q.is_empty() {
        if unknown_q.len() == 1 && factors.len() == 1 && info.unknowns.len() == 2 {
            let q = unknown_q[0];
            info.regions = vec![
                region(PdeType::Hyperbolic, pred(PredicateKind::Gt, q, pool)),
                region(PdeType::Parabolic, pred(PredicateKind::Eq, q, pool)),
                region(PdeType::Elliptic, pred(PredicateKind::Lt, q, pool)),
            ];
            info.pde_type = Some(if mentions_any(q, &indep_set, pool) {
                PdeType::Mixed
            } else {
                PdeType::Undetermined
            });
        } else {
            info.pde_type = Some(PdeType::Undetermined);
        }
        return;
    }
    if complex_count == info.unknowns.len() {
        info.pde_type = Some(PdeType::Elliptic);
        return;
    }
    if complex_count > 0 {
        // Some real, some complex speeds: neither hyperbolic nor elliptic.
        info.pde_type = Some(PdeType::Undetermined);
        return;
    }
    // All real: diagonalisable?  A speed of multiplicity μ needs
    // rank(B − λA_t) = n − μ, i.e. every minor of size n − μ + 1 vanishes.
    let n = info.unknowns.len();
    if !distinct {
        for &(lam, mu) in &info.characteristic_speeds {
            if mu < 2 {
                continue;
            }
            let pencil: Vec<Vec<ExprId>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| simp(sub(b[i][j], pool.mul(vec![lam, at[i][j]]), pool), pool))
                        .collect()
                })
                .collect();
            if !minors_vanish(&pencil, n - mu + 1, pool) {
                // Real speeds but a missing eigenvector: weakly hyperbolic
                // at best, not hyperbolic.
                info.pde_type = Some(PdeType::Undetermined);
                return;
            }
        }
        if info.characteristic_speeds.len()
            + factors
                .iter()
                .filter(|f| f.q.is_some_and(|q| is_zero(q, pool)))
                .count()
            < roots.len()
            && info.characteristic_speeds.iter().all(|&(_, mu)| mu < 2)
        {
            // A repeated root the speed list did not merge (structurally
            // different spellings of one value): not decided.
            info.pde_type = Some(PdeType::Undetermined);
            return;
        }
    }
    info.pde_type = Some(PdeType::Hyperbolic);
    info.strictly_hyperbolic = Some(distinct);
    info.type_conditions.extend(conditions);
}

/// Friedrichs: `A_t` constant, diagonal and positive, every other `A_k`
/// symmetric ⇒ hyperbolic.
fn symmetric_hyperbolic(
    info: &mut PdeSystemInfo,
    mats: &[Vec<Vec<ExprId>>],
    t: usize,
    pool: &ExprPool,
) {
    let n = info.unknowns.len();
    let at = &mats[t];
    let indep_set: HashSet<ExprId> = info.independent.iter().copied().collect();
    let mut ok = true;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                ok &= matches!(sign_of(at[i][i], pool), Some(Sign::Pos))
                    && !mentions_any(at[i][i], &indep_set, pool);
            } else {
                ok &= is_zero(at[i][j], pool);
            }
        }
    }
    for (k, a) in mats.iter().enumerate() {
        if k == t {
            continue;
        }
        for i in 0..n {
            for j in i + 1..n {
                ok &= is_zero(sub(a[i][j], a[j][i], pool), pool);
            }
        }
    }
    info.pde_type = Some(if ok {
        PdeType::Hyperbolic
    } else {
        PdeType::Undetermined
    });
}

// ---------------------------------------------------------------------------
// Named systems
// ---------------------------------------------------------------------------

fn named(
    name: &str,
    params: Vec<(&str, ExprId)>,
    conds: Vec<ExprId>,
    time: Option<ExprId>,
) -> NamedEquation {
    NamedEquation {
        name: name.to_string(),
        parameters: params
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        conditions: conds,
        time,
    }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut out = Vec::new();
    for p in permutations(n - 1) {
        for i in 0..=p.len() {
            let mut q = p.clone();
            q.insert(i, n - 1);
            out.push(q);
        }
    }
    out
}

fn same(a: ExprId, b: ExprId, pool: &ExprPool) -> bool {
    is_zero(sub(a, b, pool), pool)
}

/// `true` when `F(U)` is free of jets and independent variables.
fn constant(e: ExprId, jets: &HashSet<ExprId>, indep: &HashSet<ExprId>, pool: &ExprPool) -> bool {
    !mentions_any(e, jets, pool) && !mentions_any(e, indep, pool)
}

fn match_named(
    info: &PdeSystemInfo,
    cm: &[Vec<ExprId>],
    t: usize,
    space: &JetSpace,
    jet_set: &HashSet<ExprId>,
    pool: &ExprPool,
) -> Vec<NamedEquation> {
    let n = cm.len();
    let zero = pool.integer(0);
    // Lower-order part: zero for every template below.
    let first: Vec<ExprId> = jet_set
        .iter()
        .copied()
        .filter(|&s| {
            space
                .decode_jet(s, pool)
                .is_some_and(|(_, a)| a.iter().sum::<u32>() == 1)
        })
        .collect();
    let at_first: HashMap<ExprId, ExprId> = first.iter().map(|&s| (s, zero)).collect();
    let homogeneous = info
        .equations_in_jets
        .iter()
        .all(|&f| is_zero(crate::kernel::subs::subs(f, &at_first, pool), pool));
    if !homogeneous {
        return Vec::new();
    }
    let indep_set: HashSet<ExprId> = info.independent.iter().copied().collect();
    let w: Vec<ExprId> = (0..n)
        .map(|j| {
            space
                .jet(j, &vec![0; info.independent.len()], pool)
                .unwrap_or(zero)
        })
        .collect();
    let time = Some(info.independent[t]);
    let mut out: Vec<NamedEquation> = Vec::new();
    let push = |e: NamedEquation, out: &mut Vec<NamedEquation>| {
        if !out.iter().any(|o| o.name == e.name) {
            out.push(e);
        }
    };
    for perm in permutations(n) {
        // M in the permuted unknown order.
        let mp: Vec<Vec<ExprId>> = (0..n)
            .map(|i| (0..n).map(|j| cm[perm[i]][perm[j]]).collect())
            .collect();
        let wp: Vec<ExprId> = perm.iter().map(|&j| w[j]).collect();
        let cst = |e: ExprId| constant(e, jet_set, &indep_set, pool);
        if n == 2 {
            let (a, b, c, dd) = (mp[0][0], mp[0][1], mp[1][0], mp[1][1]);
            // Cauchy–Riemann: M = ±[[0, −1], [1, 0]].
            if is_zero(a, pool)
                && is_zero(dd, pool)
                && ((same(b, pool.integer(-1), pool) && same(c, pool.integer(1), pool))
                    || (same(b, pool.integer(1), pool) && same(c, pool.integer(-1), pool)))
            {
                push(named("cauchy_riemann", vec![], vec![], None), &mut out);
                continue;
            }
            // 1-D Maxwell: M = [[0, a], [b, 0]], constant, a·b > 0.
            if is_zero(a, pool) && is_zero(dd, pool) && cst(b) && cst(c) {
                let s2 = simp(pool.mul(vec![b, c]), pool);
                let conds = match sign_of(s2, pool) {
                    Some(Sign::Pos) => vec![],
                    Some(Sign::Neg | Sign::Zero | Sign::NonPos) => continue,
                    _ => vec![pred(PredicateKind::Gt, s2, pool)],
                };
                push(
                    named("maxwell_1d", vec![("speed_squared", s2)], conds, time),
                    &mut out,
                );
                continue;
            }
            // Shallow water, primitive (h, u): M = [[u, h], [g, u]].
            let (h, u) = (wp[0], wp[1]);
            if same(a, u, pool)
                && same(dd, u, pool)
                && same(b, h, pool)
                && cst(c)
                && !is_zero(c, pool)
            {
                push(
                    named(
                        "shallow_water",
                        vec![("gravity", c)],
                        vec![pred(
                            PredicateKind::Gt,
                            simp(pool.mul(vec![c, h]), pool),
                            pool,
                        )],
                        time,
                    ),
                    &mut out,
                );
                continue;
            }
            // Shallow water, conservative (h, m = hu): Jacobian of
            // (m, m²/h + g h²/2).
            let mm = wp[1];
            if is_zero(a, pool) && same(b, pool.integer(1), pool) {
                let g = div(
                    pool.add(vec![
                        c,
                        div(
                            pool.pow(mm, pool.integer(2)),
                            pool.pow(h, pool.integer(2)),
                            pool,
                        ),
                    ]),
                    h,
                    pool,
                );
                if cst(g)
                    && !is_zero(g, pool)
                    && same(dd, div(pool.mul(vec![pool.integer(2), mm]), h, pool), pool)
                {
                    push(
                        named(
                            "shallow_water",
                            vec![("gravity", g)],
                            vec![pred(
                                PredicateKind::Gt,
                                simp(pool.mul(vec![g, h]), pool),
                                pool,
                            )],
                            time,
                        ),
                        &mut out,
                    );
                    continue;
                }
            }
        }
        if n == 3 {
            // Euler, primitive (ρ, u, p): M = [[u, ρ, 0], [0, u, 1/ρ], [0, γp, u]].
            let (rho, u, p) = (wp[0], wp[1], wp[2]);
            let gamma = div(mp[2][1], p, pool);
            let want = [
                [u, rho, zero],
                [zero, u, div(pool.integer(1), rho, pool)],
                [zero, mp[2][1], u],
            ];
            if cst(gamma)
                && !is_zero(gamma, pool)
                && (0..3).all(|i| (0..3).all(|j| same(mp[i][j], want[i][j], pool)))
            {
                push(
                    named(
                        "euler_1d",
                        vec![("gamma", gamma)],
                        vec![pred(
                            PredicateKind::Gt,
                            simp(div(pool.mul(vec![gamma, p]), rho, pool), pool),
                            pool,
                        )],
                        time,
                    ),
                    &mut out,
                );
                continue;
            }
            // Euler, conservative (ρ, m, E): Jacobian of
            // (m, m²/ρ + p, (E + p) m/ρ), p = (γ − 1)(E − m²/(2ρ)).
            let (rho, m, e) = (wp[0], wp[1], wp[2]);
            let gamma = simp(pool.add(vec![mp[1][2], pool.integer(1)]), pool);
            if cst(gamma) && !is_zero(sub(gamma, pool.integer(1), pool), pool) {
                let pr = pool.mul(vec![
                    sub(gamma, pool.integer(1), pool),
                    sub(
                        e,
                        div(
                            pool.pow(m, pool.integer(2)),
                            pool.mul(vec![pool.integer(2), rho]),
                            pool,
                        ),
                        pool,
                    ),
                ]);
                let flux = [
                    m,
                    pool.add(vec![div(pool.pow(m, pool.integer(2)), rho, pool), pr]),
                    div(pool.mul(vec![pool.add(vec![e, pr]), m]), rho, pool),
                ];
                let vars = [rho, m, e];
                let ok = (0..3).all(|i| {
                    (0..3).all(|j| d(flux[i], vars[j], pool).is_ok_and(|v| same(mp[i][j], v, pool)))
                });
                if ok {
                    push(
                        named(
                            "euler_1d",
                            vec![("gamma", gamma)],
                            vec![pred(
                                PredicateKind::Gt,
                                simp(div(pool.mul(vec![gamma, pr]), rho, pool), pool),
                                pool,
                            )],
                            time,
                        ),
                        &mut out,
                    );
                    continue;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct S {
        p: ExprPool,
    }

    fn app(s: &S, f: &str, args: &[ExprId]) -> ExprId {
        s.p.func(f, args.to_vec())
    }

    fn dv(s: &S, u: ExprId, v: ExprId) -> ExprId {
        diff(u, v, &s.p).unwrap().value
    }

    fn name_of(info: &PdeSystemInfo) -> Vec<String> {
        info.named.iter().map(|n| n.name.clone()).collect()
    }

    #[test]
    fn cauchy_riemann_is_elliptic() {
        let s = S { p: ExprPool::new() };
        let p = &s.p;
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let u = app(&s, "u", &[x, y]);
        let v = app(&s, "v", &[x, y]);
        let e1 = sub(dv(&s, u, x), dv(&s, v, y), p);
        let e2 = p.add(vec![dv(&s, u, y), dv(&s, v, x)]);
        let info = classify_pde_system(&[e1, e2], &[u, v], p).unwrap();
        assert_eq!(info.order, 1);
        assert_eq!(info.linearity, Linearity::Linear);
        assert_eq!(info.pde_type, Some(PdeType::Elliptic));
        assert_eq!(name_of(&info), vec!["cauchy_riemann"]);
        // det P(ξ) = ξx² + ξy².
        let det = info.characteristic_determinant.unwrap();
        let xs = &info.symbol_variables;
        let want = p.add(vec![p.pow(xs[0], p.integer(2)), p.pow(xs[1], p.integer(2))]);
        assert!(is_zero(sub(det, want, p), p), "{}", p.display(det));
    }

    #[test]
    fn shallow_water_is_hyperbolic_under_gh_positive() {
        let s = S { p: ExprPool::new() };
        let p = &s.p;
        let x = p.symbol("x", Domain::Real);
        let t = p.symbol("t", Domain::Real);
        let g = p.symbol("g", Domain::Real);
        let h = app(&s, "h", &[x, t]);
        let u = app(&s, "u", &[x, t]);
        // h_t + (h u)_x = 0, u_t + u u_x + g h_x = 0.
        let e1 = p.add(vec![dv(&s, h, t), dv(&s, p.mul(vec![h, u]), x)]);
        let e2 = p.add(vec![
            dv(&s, u, t),
            p.mul(vec![u, dv(&s, u, x)]),
            p.mul(vec![g, dv(&s, h, x)]),
        ]);
        let info = classify_pde_system(&[e1, e2], &[h, u], p).unwrap();
        assert_eq!(info.linearity, Linearity::Quasilinear);
        assert_eq!(info.pde_type, Some(PdeType::Hyperbolic));
        assert_eq!(info.strictly_hyperbolic, Some(true));
        assert_eq!(info.evolution_variable, Some(t));
        assert_eq!(info.characteristic_speeds.len(), 2);
        assert_eq!(name_of(&info), vec!["shallow_water"]);
        let cond = p.display(info.type_conditions[0]).to_string();
        assert!(cond.contains('g') && cond.contains("h(x, t)"), "{cond}");
        // speeds u ± √(g h)
        let ssum = p.add(info.characteristic_speeds.iter().map(|s| s.0).collect());
        assert!(is_zero(sub(ssum, p.mul(vec![p.integer(2), u]), p), p));
    }

    #[test]
    fn euler_primitive_and_maxwell() {
        let s = S { p: ExprPool::new() };
        let p = &s.p;
        let x = p.symbol("x", Domain::Real);
        let t = p.symbol("t", Domain::Real);
        let gamma = p.symbol("gamma", Domain::Real);
        let rho = app(&s, "rho", &[x, t]);
        let u = app(&s, "u", &[x, t]);
        let pr = app(&s, "p", &[x, t]);
        let e1 = p.add(vec![dv(&s, rho, t), dv(&s, p.mul(vec![rho, u]), x)]);
        let e2 = p.add(vec![
            dv(&s, u, t),
            p.mul(vec![u, dv(&s, u, x)]),
            p.mul(vec![p.pow(rho, p.integer(-1)), dv(&s, pr, x)]),
        ]);
        let e3 = p.add(vec![
            dv(&s, pr, t),
            p.mul(vec![u, dv(&s, pr, x)]),
            p.mul(vec![gamma, pr, dv(&s, u, x)]),
        ]);
        let info = classify_pde_system(&[e1, e2, e3], &[rho, u, pr], p).unwrap();
        assert_eq!(info.pde_type, Some(PdeType::Hyperbolic));
        assert_eq!(info.strictly_hyperbolic, Some(true));
        assert_eq!(info.characteristic_speeds.len(), 3);
        assert_eq!(name_of(&info), vec!["euler_1d"]);

        let c = p.symbol("c", Domain::Positive);
        let ef = app(&s, "E", &[x, t]);
        let bf = app(&s, "B", &[x, t]);
        let m1 = p.add(vec![
            dv(&s, ef, t),
            p.mul(vec![p.pow(c, p.integer(2)), dv(&s, bf, x)]),
        ]);
        let m2 = p.add(vec![dv(&s, bf, t), dv(&s, ef, x)]);
        let info = classify_pde_system(&[m1, m2], &[ef, bf], p).unwrap();
        assert_eq!(info.pde_type, Some(PdeType::Hyperbolic));
        assert_eq!(info.strictly_hyperbolic, Some(true));
        assert_eq!(name_of(&info), vec!["maxwell_1d"]);
        assert!(info.type_conditions.is_empty());
    }

    #[test]
    fn repeated_speed_needs_eigenvectors() {
        let s = S { p: ExprPool::new() };
        let p = &s.p;
        let x = p.symbol("x", Domain::Real);
        let t = p.symbol("t", Domain::Real);
        let u = app(&s, "u", &[x, t]);
        let v = app(&s, "v", &[x, t]);
        // Decoupled transport: hyperbolic, not strictly.
        let e1 = p.add(vec![dv(&s, u, t), dv(&s, u, x)]);
        let e2 = p.add(vec![dv(&s, v, t), dv(&s, v, x)]);
        let info = classify_pde_system(&[e1, e2], &[u, v], p).unwrap();
        assert_eq!(info.pde_type, Some(PdeType::Hyperbolic));
        assert_eq!(info.strictly_hyperbolic, Some(false));
        // Jordan block u_t + u_x + v_x = 0, v_t + v_x = 0: not hyperbolic.
        let e1 = p.add(vec![dv(&s, u, t), dv(&s, u, x), dv(&s, v, x)]);
        let info = classify_pde_system(&[e1, e2], &[u, v], p).unwrap();
        assert_eq!(info.pde_type, Some(PdeType::Undetermined));
    }

    #[test]
    fn three_dimensional_symmetric_hyperbolic_and_second_order() {
        let s = S { p: ExprPool::new() };
        let p = &s.p;
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let t = p.symbol("t", Domain::Real);
        let a = app(&s, "a", &[x, y, t]);
        let b = app(&s, "b", &[x, y, t]);
        let c = app(&s, "c", &[x, y, t]);
        // acoustics: a_t + b_x + c_y = 0, b_t + a_x = 0, c_t + a_y = 0.
        let e1 = p.add(vec![dv(&s, a, t), dv(&s, b, x), dv(&s, c, y)]);
        let e2 = p.add(vec![dv(&s, b, t), dv(&s, a, x)]);
        let e3 = p.add(vec![dv(&s, c, t), dv(&s, a, y)]);
        let info = classify_pde_system(&[e1, e2, e3], &[a, b, c], p).unwrap();
        assert_eq!(info.pde_type, Some(PdeType::Hyperbolic));
        // Second order: order/linearity only.
        let u = app(&s, "u", &[x, t]);
        let v = app(&s, "v", &[x, t]);
        let e1 = sub(dv(&s, u, t), dv(&s, dv(&s, u, x), x), p);
        let e2 = sub(dv(&s, v, t), p.mul(vec![u, v]), p);
        let info = classify_pde_system(&[e1, e2], &[u, v], p).unwrap();
        assert_eq!(info.order, 2);
        assert_eq!(info.orders, vec![2, 1]);
        assert_eq!(info.linearity, Linearity::Semilinear);
        assert_eq!(info.pde_type, None);
    }
}
