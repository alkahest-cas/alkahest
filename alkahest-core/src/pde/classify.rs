//! `classify_pde`: order, linearity class, homogeneity, coefficient type,
//! elliptic / parabolic / hyperbolic type (with regions for variable
//! coefficients), and named-equation matching.
//!
//! Everything is read off the equation in jet coordinates
//! ([`super::jet::JetSpace`]): with `F(x, u, u_α) = 0`,
//!
//! * the **order** is the highest `|α|` present;
//! * **linear** — every `∂F/∂u_α` is free of jets; **semilinear** — the
//!   highest-order ones are; **quasilinear** — the highest-order ones are free
//!   of highest-order jets; otherwise **fully nonlinear**;
//! * the **type** of a second-order equation comes from its principal symbol
//!   `Aᵢⱼ = ∂F/∂u_{eᵢ+eⱼ}` (halved off the diagonal).  In two variables the
//!   discriminant `Δ = B² − 4AC` (`B` the coefficient of `u_xy`) decides it
//!   pointwise; when `Δ` depends on one independent variable its real roots are
//!   isolated exactly and the line is cut into elliptic / parabolic /
//!   hyperbolic regions; when it depends on the solution the type is
//!   [`PdeType::SolutionDependent`].
//!
//! Named equations are recognised by decomposing `F` into jet monomials
//! (exactly, through the rational-function conversion), so `u_t = k·u_xx` is
//! the heat equation however it is spelled, scaled or ordered.

use std::collections::{BTreeMap, HashSet};

use crate::diff::diff_impl::diff;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::error::PdeError;
use super::jet::{decode_unknown, symbols_of, JetCoordinate, JetSpace, MultiIndex};

/// Linearity class, from most to least structured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Linearity {
    /// Affine in the unknown and all its derivatives.
    Linear,
    /// Highest-order part linear with coefficients depending on `x` only.
    Semilinear,
    /// Highest-order part linear, coefficients may depend on lower jets.
    Quasilinear,
    /// Anything else.
    FullyNonlinear,
}

impl Linearity {
    /// Stable lower-case name (`"linear"`, `"semilinear"`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            Linearity::Linear => "linear",
            Linearity::Semilinear => "semilinear",
            Linearity::Quasilinear => "quasilinear",
            Linearity::FullyNonlinear => "fully_nonlinear",
        }
    }
}

/// Type of a second-order equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PdeType {
    /// Principal symbol definite.
    Elliptic,
    /// Principal symbol degenerate (a zero eigenvalue).
    Parabolic,
    /// Principal symbol non-degenerate with exactly one eigenvalue of the
    /// minority sign.
    Hyperbolic,
    /// Non-degenerate with at least two eigenvalues of each sign.
    Ultrahyperbolic,
    /// Different types in different regions; see [`PdeInfo::regions`].
    Mixed,
    /// The principal coefficients depend on the solution (quasilinear or fully
    /// nonlinear): the type is a property of a solution, not of the equation.
    SolutionDependent,
    /// The type could not be decided (symbolic coefficients of unknown sign).
    Undetermined,
}

impl PdeType {
    /// Stable lower-case name.
    pub fn as_str(self) -> &'static str {
        match self {
            PdeType::Elliptic => "elliptic",
            PdeType::Parabolic => "parabolic",
            PdeType::Hyperbolic => "hyperbolic",
            PdeType::Ultrahyperbolic => "ultrahyperbolic",
            PdeType::Mixed => "mixed",
            PdeType::SolutionDependent => "solution_dependent",
            PdeType::Undetermined => "undetermined",
        }
    }
}

/// One region of constant type for a variable-coefficient equation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TypeRegion {
    /// The type on this region.
    pub pde_type: PdeType,
    /// The region as a predicate in the independent variables (and
    /// parameters): `Δ > 0`, `x < 0`, `(x > -1) ∧ (x < 1)`, `x = 0`, ….
    pub condition: ExprId,
    /// When the discriminant depends on a single independent variable, that
    /// variable; the region is then the interval [`Self::lower`] …
    /// [`Self::upper`] in it.
    pub var: Option<ExprId>,
    /// Lower end (exclusive), `None` for `−∞`.  Equal to [`Self::upper`] for a
    /// parabolic line `var = r`.  An exact rational or a `RootOf`.
    pub lower: Option<ExprId>,
    /// Upper end (exclusive), `None` for `+∞`.
    pub upper: Option<ExprId>,
}

/// A recognised named equation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct NamedEquation {
    /// Stable name: `"heat"`, `"wave"`, `"laplace"`, `"poisson"`,
    /// `"helmholtz"`, `"transport"`, `"burgers"`, `"inviscid_burgers"`,
    /// `"kdv"`, `"fisher_kpp"`, `"sine_gordon"`, `"klein_gordon"`,
    /// `"telegraph"`.
    pub name: String,
    /// The equation's constants in its textbook normalisation, e.g.
    /// `[("diffusivity", k)]` for `u_t = k Δu`, `[("speed_squared", c²)]` for
    /// `u_tt = c² Δu`, and `("source", f)` when there is a forcing term.
    pub parameters: Vec<(String, ExprId)>,
    /// Conditions on the parameters under which the name applies (for
    /// example `k > 0` when the diffusivity is a symbol of unknown sign).
    pub conditions: Vec<ExprId>,
    /// The variable read as time, when the name singles one out.
    pub time: Option<ExprId>,
}

/// Result of [`classify_pde`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PdeInfo {
    /// The unknown `u(x, …)`.
    pub unknown: ExprId,
    /// The independent variables, in the unknown's argument order.
    pub independent: Vec<ExprId>,
    /// The equation in jet coordinates (`u_t - u_xx`), read as `= 0`.
    pub equation_in_jets: ExprId,
    /// Highest derivative order.
    pub order: u32,
    /// Linearity class.
    pub linearity: Linearity,
    /// For a linear equation, whether `u ≡ 0` solves it (no source term);
    /// `None` otherwise.
    pub homogeneous: Option<bool>,
    /// For a linear equation, whether every coefficient of the operator is
    /// free of the independent variables; `None` otherwise.
    pub constant_coefficients: Option<bool>,
    /// Type of a second-order equation; `None` for other orders.
    pub pde_type: Option<PdeType>,
    /// Conditions under which [`Self::pde_type`] holds (`c ≠ 0` for
    /// `u_tt = c² u_xx`).
    pub type_conditions: Vec<ExprId>,
    /// The principal symbol `Aᵢⱼ` of a second-order equation (rows and
    /// columns follow [`Self::independent`]).
    pub principal_symbol: Option<Vec<Vec<ExprId>>>,
    /// `B² − 4AC` for a second-order equation in two variables.
    pub discriminant: Option<ExprId>,
    /// Regions of constant type when the type varies (or is undetermined).
    pub regions: Vec<TypeRegion>,
    /// Matched named equations (usually zero or one).
    pub named: Vec<NamedEquation>,
}

pub(crate) fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    let s = simplify(e, pool).value;
    if s == pool.integer(0) {
        return true;
    }
    matches!(
        crate::simplify::zero::zero_test(e, pool, None).status,
        crate::simplify::zero::ZeroStatus::ProvedZero
    )
}

fn mentions_any(e: ExprId, set: &HashSet<ExprId>, pool: &ExprPool) -> bool {
    symbols_of(e, pool).iter().any(|s| set.contains(s))
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}

fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    simplify(pool.mul(vec![a, pool.pow(b, pool.integer(-1))]), pool).value
}

/// Classify the scalar PDE `equation = 0` in the unknown `unknown`
/// (`u(x, t)`).  See the [module documentation](self).
///
/// # Errors
///
/// `E-PDE-001` for a malformed unknown; `E-PDE-003` when the unknown occurs
/// off its declared arguments; `E-PDE-007` when the equation has no
/// derivative of the unknown.
pub fn classify_pde(
    equation: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<PdeInfo, PdeError> {
    let v = decode_unknown(unknown, pool)?;
    let indep = v.args.clone();
    let big = JetSpace::new(&indep, &[unknown], u32::MAX / 4, pool)?;
    let f = simplify(big.to_jets(equation, pool)?, pool).value;
    let coords = big.jets_in(f, pool)?;
    let order = coords
        .iter()
        .map(|c| c.alpha.iter().sum::<u32>())
        .max()
        .unwrap_or(0);
    if order == 0 {
        return Err(PdeError::NotADifferentialEquation(format!(
            "{} has no derivative of {}",
            pool.display(equation),
            pool.display(unknown)
        )));
    }
    let space = big.prolonged(0);
    let jet_set: HashSet<ExprId> = coords.iter().map(|c| c.symbol).collect();
    let top_set: HashSet<ExprId> = coords
        .iter()
        .filter(|c| c.alpha.iter().sum::<u32>() == order)
        .map(|c| c.symbol)
        .collect();
    let derr = |e: crate::diff::DiffError| PdeError::NotRationalInJets(e.to_string());

    // Partial derivatives by every jet present.
    let mut partials: Vec<(JetCoordinate, ExprId)> = Vec::with_capacity(coords.len());
    for c in &coords {
        let p = simplify(diff(f, c.symbol, pool).map_err(derr)?.value, pool).value;
        partials.push((c.clone(), p));
    }
    let linear = partials
        .iter()
        .all(|(_, p)| !mentions_any(*p, &jet_set, pool));
    let top: Vec<&(JetCoordinate, ExprId)> = partials
        .iter()
        .filter(|(c, _)| c.alpha.iter().sum::<u32>() == order)
        .collect();
    let linearity = if linear {
        Linearity::Linear
    } else if top.iter().all(|(_, p)| !mentions_any(*p, &jet_set, pool)) {
        Linearity::Semilinear
    } else if top.iter().all(|(_, p)| !mentions_any(*p, &top_set, pool)) {
        Linearity::Quasilinear
    } else {
        Linearity::FullyNonlinear
    };

    let zero = pool.integer(0);
    let at_zero: std::collections::HashMap<ExprId, ExprId> =
        jet_set.iter().map(|&s| (s, zero)).collect();
    let indep_set: HashSet<ExprId> = indep.iter().copied().collect();
    let (homogeneous, constant_coefficients) = if linear {
        let f0 = crate::kernel::subs::subs(f, &at_zero, pool);
        (
            Some(is_zero(f0, pool)),
            Some(
                partials
                    .iter()
                    .all(|(_, p)| !mentions_any(*p, &indep_set, pool)),
            ),
        )
    } else {
        (None, None)
    };

    let mut info = PdeInfo {
        unknown,
        independent: indep.clone(),
        equation_in_jets: f,
        order,
        linearity,
        homogeneous,
        constant_coefficients,
        pde_type: None,
        type_conditions: Vec::new(),
        principal_symbol: None,
        discriminant: None,
        regions: Vec::new(),
        named: Vec::new(),
    };

    if order == 2 {
        classify_second_order(&mut info, &space, &partials, &jet_set, pool);
    }
    info.named = match_named(&info, &space, &jet_set, pool);
    Ok(info)
}

// ---------------------------------------------------------------------------
// Second-order type
// ---------------------------------------------------------------------------

/// Coarse sign of an expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sign {
    Pos,
    NonNeg,
    Zero,
    NonPos,
    Neg,
}

impl Sign {
    fn negate(self) -> Sign {
        match self {
            Sign::Pos => Sign::Neg,
            Sign::NonNeg => Sign::NonPos,
            Sign::Zero => Sign::Zero,
            Sign::NonPos => Sign::NonNeg,
            Sign::Neg => Sign::Pos,
        }
    }
    fn times(self, o: Sign) -> Sign {
        use Sign::*;
        match (self, o) {
            (Zero, _) | (_, Zero) => Zero,
            (Pos, s) | (s, Pos) => s,
            (Neg, s) | (s, Neg) => s.negate(),
            (NonNeg, NonNeg) | (NonPos, NonPos) => NonNeg,
            (NonNeg, NonPos) | (NonPos, NonNeg) => NonPos,
        }
    }
    fn plus(self, o: Sign) -> Option<Sign> {
        use Sign::*;
        Some(match (self, o) {
            (Zero, s) | (s, Zero) => s,
            (Pos, Pos) | (Pos, NonNeg) | (NonNeg, Pos) => Pos,
            (NonNeg, NonNeg) => NonNeg,
            (Neg, Neg) | (Neg, NonPos) | (NonPos, Neg) => Neg,
            (NonPos, NonPos) => NonPos,
            _ => return None,
        })
    }
}

/// Whether every symbol of `e` is real-valued by its domain.
fn real_valued(e: ExprId, pool: &ExprPool) -> bool {
    symbols_of(e, pool).iter().all(|&s| {
        pool.with(s, |d| match d {
            ExprData::Symbol { domain, .. } => *domain != Domain::Complex,
            _ => false,
        })
    })
}

/// A sign for `e` provable from its structure and symbol domains, or `None`.
pub(crate) fn sign_of(e: ExprId, pool: &ExprPool) -> Option<Sign> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(match n.0.cmp0() {
            std::cmp::Ordering::Less => Sign::Neg,
            std::cmp::Ordering::Equal => Sign::Zero,
            std::cmp::Ordering::Greater => Sign::Pos,
        }),
        ExprData::Rational(r) => Some(match r.0.cmp0() {
            std::cmp::Ordering::Less => Sign::Neg,
            std::cmp::Ordering::Equal => Sign::Zero,
            std::cmp::Ordering::Greater => Sign::Pos,
        }),
        ExprData::Symbol { domain, name, .. } => match domain {
            Domain::Positive => Some(Sign::Pos),
            Domain::NonNegative => Some(Sign::NonNeg),
            _ if name == "pi" || name == "π" => Some(Sign::Pos),
            _ => None,
        },
        ExprData::Pow { base, exp } => {
            let n = pool.with(exp, |d| match d {
                ExprData::Integer(n) => n.0.to_i64(),
                _ => None,
            });
            match n {
                Some(k) if k % 2 == 0 && real_valued(base, pool) => Some(
                    if sign_of(base, pool).is_some_and(|s| matches!(s, Sign::Pos | Sign::Neg)) {
                        Sign::Pos
                    } else {
                        Sign::NonNeg
                    },
                ),
                Some(k) => {
                    let s = sign_of(base, pool)?;
                    if k < 0 && matches!(s, Sign::Zero | Sign::NonNeg | Sign::NonPos) {
                        return None;
                    }
                    Some(if k % 2 == 0 { s.times(s) } else { s })
                }
                None => match sign_of(base, pool)? {
                    Sign::Pos => Some(Sign::Pos),
                    _ => None,
                },
            }
        }
        ExprData::Mul(xs) => xs
            .iter()
            .try_fold(Sign::Pos, |acc, &x| Some(acc.times(sign_of(x, pool)?))),
        ExprData::Add(xs) => {
            let mut acc = Sign::Zero;
            for x in xs {
                acc = acc.plus(sign_of(x, pool)?)?;
            }
            Some(acc)
        }
        ExprData::Func { name, args } if name == "exp" && real_valued(args[0], pool) => {
            Some(Sign::Pos)
        }
        _ => None,
    }
}

fn pred(kind: PredicateKind, a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.predicate(kind, vec![a, b])
}

fn classify_second_order(
    info: &mut PdeInfo,
    space: &JetSpace,
    partials: &[(JetCoordinate, ExprId)],
    jet_set: &HashSet<ExprId>,
    pool: &ExprPool,
) {
    let n = info.independent.len();
    let zero = pool.integer(0);
    let half = pool.rational(1, 2);
    let coeff = |alpha: &MultiIndex| -> ExprId {
        partials
            .iter()
            .find(|(c, _)| &c.alpha == alpha)
            .map(|(_, p)| *p)
            .unwrap_or(zero)
    };
    let mut a = vec![vec![zero; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut alpha = vec![0u32; n];
            alpha[i] += 1;
            alpha[j] += 1;
            if !space.depends_on(0, i) || !space.depends_on(0, j) {
                continue;
            }
            let c = coeff(&alpha);
            if i == j {
                a[i][i] = c;
            } else {
                let h = simplify(pool.mul(vec![half, c]), pool).value;
                a[i][j] = h;
                a[j][i] = h;
            }
        }
    }
    info.principal_symbol = Some(a.clone());
    let solution_dependent = a.iter().flatten().any(|&e| mentions_any(e, jet_set, pool));

    if n == 2 {
        // B = coefficient of u_xy = 2·A01.
        let b = simplify(pool.mul(vec![pool.integer(2), a[0][1]]), pool).value;
        let disc = simplify(
            pool.add(vec![
                pool.pow(b, pool.integer(2)),
                pool.mul(vec![pool.integer(-4), a[0][0], a[1][1]]),
            ]),
            pool,
        )
        .value;
        info.discriminant = Some(disc);
        if solution_dependent {
            info.pde_type = Some(PdeType::SolutionDependent);
            return;
        }
        classify_by_discriminant(info, disc, pool);
        return;
    }
    if solution_dependent {
        info.pde_type = Some(PdeType::SolutionDependent);
        return;
    }
    let indep_set: HashSet<ExprId> = info.independent.iter().copied().collect();
    if a.iter()
        .flatten()
        .any(|&e| mentions_any(e, &indep_set, pool))
    {
        info.pde_type = Some(PdeType::Undetermined);
        return;
    }
    info.pde_type = Some(match inertia(&a, pool) {
        Some((p, q, z)) => type_from_inertia(p, q, z),
        None => PdeType::Undetermined,
    });
}

fn type_from_inertia(p: usize, q: usize, z: usize) -> PdeType {
    if z > 0 {
        PdeType::Parabolic
    } else if p == 0 || q == 0 {
        PdeType::Elliptic
    } else if p.min(q) == 1 {
        PdeType::Hyperbolic
    } else {
        PdeType::Ultrahyperbolic
    }
}

/// `(positive, negative, zero)` eigenvalue counts of a constant symmetric
/// matrix: by sign analysis when it is diagonal, by symmetric Gaussian
/// elimination over ℚ when every entry is rational.
fn inertia(a: &[Vec<ExprId>], pool: &ExprPool) -> Option<(usize, usize, usize)> {
    let n = a.len();
    let zero = pool.integer(0);
    let diagonal = (0..n).all(|i| (0..n).all(|j| i == j || a[i][j] == zero));
    if diagonal {
        let (mut p, mut q, mut z) = (0, 0, 0);
        for (i, row) in a.iter().enumerate() {
            match sign_of(row[i], pool)? {
                Sign::Pos => p += 1,
                Sign::Neg => q += 1,
                Sign::Zero => z += 1,
                _ => return None,
            }
        }
        return Some((p, q, z));
    }
    let mut m: Vec<Vec<rug::Rational>> = Vec::with_capacity(n);
    for row in a {
        let mut r = Vec::with_capacity(n);
        for &e in row {
            r.push(match pool.get(e) {
                ExprData::Integer(k) => rug::Rational::from(k.0.clone()),
                ExprData::Rational(q) => q.0.clone(),
                _ => return None,
            });
        }
        m.push(r);
    }
    let (mut p, mut q) = (0, 0);
    let mut size = n;
    while size > 0 {
        // A nonzero diagonal pivot, else a 2×2 block on a nonzero
        // off-diagonal entry (one positive and one negative eigenvalue), else
        // the remaining block is zero.
        if let Some(k) = (0..size).find(|&k| m[k][k] != 0) {
            m.swap(k, size - 1);
            for row in m.iter_mut() {
                row.swap(k, size - 1);
            }
            let piv = m[size - 1][size - 1].clone();
            if piv > 0 {
                p += 1
            } else {
                q += 1
            }
            let pivot_row: Vec<rug::Rational> = m[size - 1][..size - 1].to_vec();
            for row in m.iter_mut().take(size - 1) {
                let f = rug::Rational::from(&row[size - 1] / &piv);
                for (x, pr) in row.iter_mut().take(size - 1).zip(&pivot_row) {
                    *x -= rug::Rational::from(&f * pr);
                }
            }
            size -= 1;
        } else if let Some((i, j)) = (0..size)
            .flat_map(|i| (0..size).map(move |j| (i, j)))
            .find(|&(i, j)| i != j && m[i][j] != 0)
        {
            // Replace row/col i by row/col i + row/col j: the new diagonal
            // entry 2·m[i][j] is nonzero (m[i][i] = m[j][j] = 0).
            let row_j: Vec<rug::Rational> = m[j][..size].to_vec();
            for (x, t) in m[i].iter_mut().zip(&row_j) {
                *x += t;
            }
            for row in m.iter_mut().take(size) {
                let t = row[j].clone();
                row[i] += t;
            }
        } else {
            break;
        }
    }
    Some((p, q, n - p - q))
}

fn classify_by_discriminant(info: &mut PdeInfo, disc: ExprId, pool: &ExprPool) {
    let zero = pool.integer(0);
    let gt = |e: ExprId| pred(PredicateKind::Gt, e, zero, pool);
    let lt = |e: ExprId| pred(PredicateKind::Lt, e, zero, pool);
    let eq = |e: ExprId| pred(PredicateKind::Eq, e, zero, pool);
    let ne = |e: ExprId| pred(PredicateKind::Ne, e, zero, pool);
    let generic_regions = |info: &mut PdeInfo| {
        info.regions = vec![
            region(PdeType::Hyperbolic, gt(disc)),
            region(PdeType::Parabolic, eq(disc)),
            region(PdeType::Elliptic, lt(disc)),
        ];
    };
    let depends: Vec<ExprId> = info
        .independent
        .iter()
        .copied()
        .filter(|&x| symbols_of(disc, pool).contains(&x))
        .collect();
    if depends.is_empty() {
        match sign_of(disc, pool) {
            Some(Sign::Pos) => info.pde_type = Some(PdeType::Hyperbolic),
            Some(Sign::Neg) => info.pde_type = Some(PdeType::Elliptic),
            Some(Sign::Zero) => info.pde_type = Some(PdeType::Parabolic),
            Some(Sign::NonNeg) => {
                info.pde_type = Some(PdeType::Hyperbolic);
                info.type_conditions.push(ne(disc));
            }
            Some(Sign::NonPos) => {
                info.pde_type = Some(PdeType::Elliptic);
                info.type_conditions.push(ne(disc));
            }
            None => {
                info.pde_type = Some(PdeType::Undetermined);
                generic_regions(info);
            }
        }
        return;
    }
    // One variable, rational polynomial coefficients: isolate the real roots.
    if depends.len() == 1 && symbols_of(disc, pool).len() == 1 {
        let x = depends[0];
        if let Ok(roots) = crate::poly::real_roots::real_roots_symbolic(disc, x, pool) {
            let mut root_exprs: Vec<ExprId> = Vec::with_capacity(roots.len());
            let mut ok = true;
            for (k, r) in roots.iter().enumerate() {
                if r.lo == r.hi {
                    root_exprs.push(rational_expr(&r.lo, pool));
                } else {
                    match crate::poly::root_of::root_of(disc, x, k, pool) {
                        Ok(e) => root_exprs.push(e),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if ok {
                info.regions = univariate_regions(disc, x, &roots, &root_exprs, pool);
                let types: HashSet<&'static str> =
                    info.regions.iter().map(|r| r.pde_type.as_str()).collect();
                info.pde_type = Some(if types.len() == 1 {
                    info.regions[0].pde_type
                } else {
                    PdeType::Mixed
                });
                return;
            }
        }
    }
    match sign_of(disc, pool) {
        Some(Sign::Pos) => info.pde_type = Some(PdeType::Hyperbolic),
        Some(Sign::Neg) => info.pde_type = Some(PdeType::Elliptic),
        Some(Sign::Zero) => info.pde_type = Some(PdeType::Parabolic),
        Some(Sign::NonNeg) => {
            info.pde_type = Some(PdeType::Mixed);
            info.regions = vec![
                region(PdeType::Hyperbolic, gt(disc)),
                region(PdeType::Parabolic, eq(disc)),
            ];
        }
        Some(Sign::NonPos) => {
            info.pde_type = Some(PdeType::Mixed);
            info.regions = vec![
                region(PdeType::Elliptic, lt(disc)),
                region(PdeType::Parabolic, eq(disc)),
            ];
        }
        None => {
            info.pde_type = Some(PdeType::Mixed);
            generic_regions(info);
        }
    }
}

pub(crate) fn region(pde_type: PdeType, condition: ExprId) -> TypeRegion {
    TypeRegion {
        pde_type,
        condition,
        var: None,
        lower: None,
        upper: None,
    }
}

fn rational_expr(r: &rug::Rational, pool: &ExprPool) -> ExprId {
    crate::diff::diff_impl::const_node(pool, r.clone())
}

fn sign_at(
    disc: ExprId,
    x: ExprId,
    at: &rug::Rational,
    pool: &ExprPool,
) -> Option<std::cmp::Ordering> {
    let mut m = std::collections::HashMap::new();
    m.insert(x, rational_expr(at, pool));
    let v = simplify(crate::kernel::subs::subs(disc, &m, pool), pool).value;
    match pool.get(v) {
        ExprData::Integer(n) => Some(n.0.cmp0()),
        ExprData::Rational(q) => Some(q.0.cmp0()),
        _ => None,
    }
}

fn type_of_sign(s: std::cmp::Ordering) -> PdeType {
    match s {
        std::cmp::Ordering::Greater => PdeType::Hyperbolic,
        std::cmp::Ordering::Less => PdeType::Elliptic,
        std::cmp::Ordering::Equal => PdeType::Parabolic,
    }
}

fn univariate_regions(
    disc: ExprId,
    x: ExprId,
    roots: &[crate::poly::real_roots::RootInterval],
    root_exprs: &[ExprId],
    pool: &ExprPool,
) -> Vec<TypeRegion> {
    let one = rug::Rational::from(1);
    let mut out = Vec::new();
    // Sample points: below the first root, between consecutive roots, above
    // the last.  Isolating intervals are disjoint, so hi_k ≤ lo_{k+1}.
    let m = roots.len();
    for k in 0..=m {
        let sample = if m == 0 {
            rug::Rational::from(0)
        } else if k == 0 {
            rug::Rational::from(&roots[0].lo - &one)
        } else if k == m {
            rug::Rational::from(&roots[m - 1].hi + &one)
        } else {
            rug::Rational::from(&roots[k - 1].hi + &roots[k].lo) / 2
        };
        let Some(s) = sign_at(disc, x, &sample, pool) else {
            continue;
        };
        let lower = (k > 0).then(|| root_exprs[k - 1]);
        let upper = (k < m).then(|| root_exprs[k]);
        let cond = match (lower, upper) {
            (None, None) => pool.predicate(PredicateKind::True, vec![]),
            (Some(l), None) => pred(PredicateKind::Gt, x, l, pool),
            (None, Some(u)) => pred(PredicateKind::Lt, x, u, pool),
            (Some(l), Some(u)) => pool.predicate(
                PredicateKind::And,
                vec![
                    pred(PredicateKind::Gt, x, l, pool),
                    pred(PredicateKind::Lt, x, u, pool),
                ],
            ),
        };
        out.push(TypeRegion {
            pde_type: type_of_sign(s),
            condition: cond,
            var: Some(x),
            lower,
            upper,
        });
        if k < m {
            out.push(TypeRegion {
                pde_type: PdeType::Parabolic,
                condition: pred(PredicateKind::Eq, x, root_exprs[k], pool),
                var: Some(x),
                lower: Some(root_exprs[k]),
                upper: Some(root_exprs[k]),
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Named equations
// ---------------------------------------------------------------------------

/// A jet monomial: sorted `(factor, exponent)` pairs, where a factor is a jet
/// symbol or a jet-dependent opaque generator such as `sin(u)`.  The empty
/// monomial is the source term.
type Monomial = Vec<(ExprId, u32)>;

/// `F` as `Σ coeff · monomial` with jet-free coefficients, exactly; `None`
/// when `F` is not polynomial in its jet factors.
fn decompose(
    f: ExprId,
    jets: &[ExprId],
    jet_set: &HashSet<ExprId>,
    pool: &ExprPool,
) -> Option<BTreeMap<Monomial, ExprId>> {
    let (rf, gens) = crate::poly::cancel::rational_over_generators(f, jets.to_vec(), pool).ok()?;
    let is_key: Vec<bool> = gens
        .iter()
        .enumerate()
        .map(|(i, &g)| i < jets.len() || mentions_any(g, jet_set, pool))
        .collect();
    if rf
        .denom
        .terms
        .keys()
        .any(|e| e.iter().enumerate().any(|(i, &k)| k > 0 && is_key[i]))
    {
        return None;
    }
    let den = rf.denom.to_expr(pool);
    let mut acc: BTreeMap<Monomial, Vec<ExprId>> = BTreeMap::new();
    for (exps, c) in rf.numer.terms.iter() {
        let mut key: Monomial = Vec::new();
        let mut factors = vec![pool.integer(c.clone())];
        for (i, &g) in gens.iter().enumerate() {
            let e = exps.get(i).copied().unwrap_or(0);
            if e == 0 {
                continue;
            }
            if is_key[i] {
                key.push((g, e));
            } else {
                factors.push(if e == 1 {
                    g
                } else {
                    pool.pow(g, pool.integer(e))
                });
            }
        }
        key.sort();
        acc.entry(key).or_default().push(pool.mul(factors));
    }
    Some(
        acc.into_iter()
            .map(|(k, v)| {
                let s = if v.len() == 1 { v[0] } else { pool.add(v) };
                (k, div(s, den, pool))
            })
            .collect(),
    )
}

struct Ctx<'a> {
    pool: &'a ExprPool,
    space: &'a JetSpace,
    terms: BTreeMap<Monomial, ExprId>,
    indep: Vec<ExprId>,
}

impl Ctx<'_> {
    fn jet(&self, alpha: &[u32]) -> ExprId {
        self.space
            .jet(0, alpha, self.pool)
            .expect("valid multi-index")
    }
    fn e(&self, i: usize, k: u32) -> MultiIndex {
        let mut a = vec![0; self.indep.len()];
        a[i] = k;
        a
    }
    fn mono(&self, factors: &[(ExprId, u32)]) -> Monomial {
        let mut m: Monomial = factors.to_vec();
        m.sort();
        m
    }
    fn coeff(&self, m: &Monomial) -> Option<ExprId> {
        self.terms.get(m).copied()
    }
    /// The monomials present, minus the source term.
    fn keys(&self) -> HashSet<Monomial> {
        self.terms
            .keys()
            .filter(|k| !k.is_empty())
            .cloned()
            .collect()
    }
    fn source(&self) -> Option<ExprId> {
        self.terms.get(&Vec::new()).copied()
    }
    fn constant(&self, e: ExprId) -> bool {
        let s = symbols_of(e, self.pool);
        !self.indep.iter().any(|x| s.contains(x))
    }
}

/// `r > 0`: `Some(None)` when proved, `Some(Some(cond))` when it is a
/// condition on symbols, `None` when refuted.
fn positive(r: ExprId, pool: &ExprPool) -> Option<Option<ExprId>> {
    // A complex coefficient (`i·u_t + u_xx`, Schrödinger) is never a
    // diffusivity or a squared speed, whatever its symbols turn out to be.
    if !real_valued(r, pool) {
        return None;
    }
    match sign_of(r, pool) {
        Some(Sign::Pos) => Some(None),
        Some(Sign::Neg | Sign::Zero | Sign::NonPos) => None,
        _ => Some(Some(pred(PredicateKind::Gt, r, pool.integer(0), pool))),
    }
}

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

fn match_named(
    info: &PdeInfo,
    space: &JetSpace,
    jet_set: &HashSet<ExprId>,
    pool: &ExprPool,
) -> Vec<NamedEquation> {
    let mut jets: Vec<ExprId> = jet_set.iter().copied().collect();
    jets.sort();
    let Some(terms) = decompose(info.equation_in_jets, &jets, jet_set, pool) else {
        return Vec::new();
    };
    if terms.values().any(|&c| is_zero(c, pool)) {
        return Vec::new();
    }
    let cx = Ctx {
        pool,
        space,
        terms,
        indep: info.independent.clone(),
    };
    // Every coefficient but the source term must be constant for a textbook
    // name (variable-coefficient transport is the one exception, below).
    let all_constant = cx
        .terms
        .iter()
        .all(|(k, &c)| k.is_empty() || cx.constant(c));
    let n = cx.indep.len();
    let time = cx.indep.iter().position(|&x| {
        pool.with(
            x,
            |d| matches!(d, ExprData::Symbol { name, .. } if name == "t" || name == "tau"),
        )
    });
    let mut out = Vec::new();
    let keys = cx.keys();
    let src = cx.source();
    let with_source = |mut p: Vec<(&'static str, ExprId)>, lead: ExprId| {
        if let Some(s) = src {
            p.push((
                "source",
                simplify(neg(div(s, lead, pool), pool), pool).value,
            ));
        }
        p
    };
    let u0 = cx.jet(&vec![0; n]);

    // --- elliptic family over all variables: Δu (+ k² u) (= f) ---------------
    if all_constant && info.order == 2 {
        let lap: Vec<Monomial> = (0..n)
            .map(|i| cx.mono(&[(cx.jet(&cx.e(i, 2)), 1)]))
            .collect();
        let mut rest: HashSet<Monomial> = keys.clone();
        if n >= 2 && lap.iter().all(|m| rest.remove(m)) {
            let a = cx.coeff(&lap[0]).unwrap();
            let equal = lap
                .iter()
                .all(|m| is_zero(pool.add(vec![cx.coeff(m).unwrap(), neg(a, pool)]), pool));
            let umono = cx.mono(&[(u0, 1)]);
            if equal && rest.is_empty() {
                // RHS f in Δu = f is −source/a.
                match src {
                    None => out.push(named("laplace", vec![], vec![], None)),
                    Some(s) => out.push(named(
                        "poisson",
                        vec![("source", simplify(neg(div(s, a, pool), pool), pool).value)],
                        vec![],
                        None,
                    )),
                }
            } else if equal && rest.len() == 1 && rest.contains(&umono) {
                let k2 = div(cx.coeff(&umono).unwrap(), a, pool);
                let mut p = vec![("k_squared", k2)];
                if let Some(s) = src {
                    p.push(("source", simplify(neg(div(s, a, pool), pool), pool).value));
                }
                out.push(named("helmholtz", p, vec![], None));
            }
        }
    }

    let Some(ti) = time else {
        return out;
    };
    let ut = cx.mono(&[(cx.jet(&cx.e(ti, 1)), 1)]);
    let utt = cx.mono(&[(cx.jet(&cx.e(ti, 2)), 1)]);
    let spatial: Vec<usize> = (0..n).filter(|&i| i != ti).collect();
    let tvar = Some(cx.indep[ti]);
    let lap_s: Vec<Monomial> = spatial
        .iter()
        .map(|&i| cx.mono(&[(cx.jet(&cx.e(i, 2)), 1)]))
        .collect();
    let umono = cx.mono(&[(u0, 1)]);
    let u2 = cx.mono(&[(u0, 2)]);
    // Equal spatial-Laplacian coefficients; returns the common one.
    let common_lap = |rest: &mut HashSet<Monomial>| -> Option<ExprId> {
        if lap_s.is_empty() || !lap_s.iter().all(|m| rest.remove(m)) {
            return None;
        }
        let a = cx.coeff(&lap_s[0])?;
        lap_s
            .iter()
            .all(|m| is_zero(pool.add(vec![cx.coeff(m).unwrap(), neg(a, pool)]), pool))
            .then_some(a)
    };

    // --- evolution: u_t = D Δu (+ …) ------------------------------------------
    if all_constant && keys.contains(&ut) && !keys.contains(&utt) {
        let lead = cx.coeff(&ut).unwrap();
        let mut rest = keys.clone();
        rest.remove(&ut);
        let mut r2 = rest.clone();
        if let Some(a) = common_lap(&mut r2) {
            let dcoef = simplify(neg(div(a, lead, pool), pool), pool).value;
            if r2.is_empty() {
                if let Some(c) = positive(dcoef, pool) {
                    out.push(named(
                        "heat",
                        with_source(vec![("diffusivity", dcoef)], lead),
                        c.into_iter().collect(),
                        tvar,
                    ));
                }
            } else if r2.len() == 2 && r2.contains(&umono) && r2.contains(&u2) && src.is_none() {
                // u_t = D Δu + r u − (r/K) u²
                let r = simplify(neg(div(cx.coeff(&umono).unwrap(), lead, pool), pool), pool).value;
                let q = div(cx.coeff(&u2).unwrap(), lead, pool); // = r/K
                let k = div(r, q, pool);
                let mut conds = Vec::new();
                if let Some(c) = positive(dcoef, pool) {
                    conds.extend(c);
                    if let Some(c2) = positive(k, pool) {
                        conds.extend(c2);
                        out.push(named(
                            "fisher_kpp",
                            vec![("diffusivity", dcoef), ("rate", r), ("capacity", k)],
                            conds,
                            tvar,
                        ));
                    }
                }
            }
        }
        // Burgers / KdV in one space dimension.
        if spatial.len() == 1 {
            let xi = spatial[0];
            let ux = cx.jet(&cx.e(xi, 1));
            let uux = cx.mono(&[(u0, 1), (ux, 1)]);
            let uxx = cx.mono(&[(cx.jet(&cx.e(xi, 2)), 1)]);
            let uxxx = cx.mono(&[(cx.jet(&cx.e(xi, 3)), 1)]);
            if rest.contains(&uux) && src.is_none() {
                let a = div(cx.coeff(&uux).unwrap(), lead, pool);
                let mut r3 = rest.clone();
                r3.remove(&uux);
                if r3.is_empty() {
                    out.push(named(
                        "inviscid_burgers",
                        vec![("advection", a)],
                        vec![],
                        tvar,
                    ));
                } else if r3.len() == 1 && r3.contains(&uxx) {
                    let nu =
                        simplify(neg(div(cx.coeff(&uxx).unwrap(), lead, pool), pool), pool).value;
                    if let Some(c) = positive(nu, pool) {
                        out.push(named(
                            "burgers",
                            vec![("advection", a), ("viscosity", nu)],
                            c.into_iter().collect(),
                            tvar,
                        ));
                    }
                } else if r3.len() == 1 && r3.contains(&uxxx) {
                    let b = div(cx.coeff(&uxxx).unwrap(), lead, pool);
                    out.push(named(
                        "kdv",
                        vec![("advection", a), ("dispersion", b)],
                        vec![],
                        tvar,
                    ));
                }
            }
        }
    }

    // --- transport: u_t + c·∇u = 0 (coefficients may vary) --------------------
    if info.order == 1 && info.linearity == Linearity::Linear && keys.contains(&ut) {
        let lead = cx.coeff(&ut).unwrap();
        let mut rest = keys.clone();
        rest.remove(&ut);
        let grads: Vec<(usize, Monomial)> = spatial
            .iter()
            .map(|&i| (i, cx.mono(&[(cx.jet(&cx.e(i, 1)), 1)])))
            .collect();
        let mut params: Vec<(String, ExprId)> = Vec::new();
        for (i, m) in &grads {
            if rest.remove(m) {
                let c = div(cx.coeff(m).unwrap(), lead, pool);
                params.push((format!("velocity_{}", pool.display(cx.indep[*i])), c));
            }
        }
        if !params.is_empty() && rest.is_empty() && src.is_none() {
            out.push(NamedEquation {
                name: "transport".into(),
                parameters: params,
                conditions: vec![],
                time: tvar,
            });
        }
    }

    // --- wave family: u_tt = c² Δu (+ a u_t + m² u …) ---------------------------
    if all_constant && keys.contains(&utt) {
        let lead = cx.coeff(&utt).unwrap();
        let mut rest = keys.clone();
        rest.remove(&utt);
        if let Some(a) = common_lap(&mut rest) {
            let c2 = simplify(neg(div(a, lead, pool), pool), pool).value;
            if let Some(cpos) = positive(c2, pool) {
                let cconds: Vec<ExprId> = cpos.into_iter().collect();
                let sin_u = pool.func("sin", vec![u0]);
                let sinm = cx.mono(&[(sin_u, 1)]);
                if rest.is_empty() {
                    out.push(named(
                        "wave",
                        with_source(vec![("speed_squared", c2)], lead),
                        cconds,
                        tvar,
                    ));
                } else if rest.len() == 1 && rest.contains(&umono) {
                    let m2 = div(cx.coeff(&umono).unwrap(), lead, pool);
                    out.push(named(
                        "klein_gordon",
                        with_source(vec![("speed_squared", c2), ("mass_squared", m2)], lead),
                        cconds,
                        tvar,
                    ));
                } else if rest.contains(&ut) && rest.iter().all(|m| *m == ut || *m == umono) {
                    let damping = div(cx.coeff(&ut).unwrap(), lead, pool);
                    let mut p = vec![("speed_squared", c2), ("damping", damping)];
                    if let Some(c) = cx.coeff(&umono) {
                        p.push(("reaction", div(c, lead, pool)));
                    }
                    out.push(named("telegraph", with_source(p, lead), cconds, tvar));
                } else if rest.len() == 1 && rest.contains(&sinm) && src.is_none() {
                    let m = div(cx.coeff(&sinm).unwrap(), lead, pool);
                    out.push(named(
                        "sine_gordon",
                        vec![("speed_squared", c2), ("coupling", m)],
                        cconds,
                        tvar,
                    ));
                }
            }
        }
    }
    // Light-cone sine-Gordon: u_xt = sin(u).
    if all_constant && n == 2 && src.is_none() {
        let uxt = cx.mono(&[(cx.jet(&[1, 1]), 1)]);
        let sinm = cx.mono(&[(pool.func("sin", vec![u0]), 1)]);
        if keys.len() == 2 && keys.contains(&uxt) && keys.contains(&sinm) {
            let m = simplify(
                neg(
                    div(cx.coeff(&sinm).unwrap(), cx.coeff(&uxt).unwrap(), pool),
                    pool,
                ),
                pool,
            )
            .value;
            out.push(named("sine_gordon", vec![("coupling", m)], vec![], tvar));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct S {
        pool: ExprPool,
        x: ExprId,
        y: ExprId,
        t: ExprId,
    }

    fn s() -> S {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        S { pool, x, y, t }
    }

    fn d(s: &S, e: ExprId, vs: &[ExprId]) -> ExprId {
        let mut c = e;
        for &v in vs {
            c = diff(c, v, &s.pool).unwrap().value;
        }
        c
    }

    #[test]
    fn heat_wave_laplace() {
        let s = s();
        let p = &s.pool;
        let u = p.func("u", vec![s.x, s.t]);
        let heat = p.add(vec![d(&s, u, &[s.t]), neg(d(&s, u, &[s.x, s.x]), p)]);
        let i = classify_pde(heat, u, p).unwrap();
        assert_eq!(i.order, 2);
        assert_eq!(i.linearity, Linearity::Linear);
        assert_eq!(i.homogeneous, Some(true));
        assert_eq!(i.constant_coefficients, Some(true));
        assert_eq!(i.pde_type, Some(PdeType::Parabolic));
        assert_eq!(i.named[0].name, "heat");

        let c = p.symbol("c", Domain::Real);
        let wave = p.add(vec![
            d(&s, u, &[s.t, s.t]),
            neg(
                p.mul(vec![p.pow(c, p.integer(2)), d(&s, u, &[s.x, s.x])]),
                p,
            ),
        ]);
        let i = classify_pde(wave, u, p).unwrap();
        assert_eq!(i.pde_type, Some(PdeType::Hyperbolic));
        assert_eq!(i.type_conditions.len(), 1);
        assert_eq!(i.named[0].name, "wave");

        let v = p.func("v", vec![s.x, s.y]);
        let lap = p.add(vec![d(&s, v, &[s.x, s.x]), d(&s, v, &[s.y, s.y])]);
        let i = classify_pde(lap, v, p).unwrap();
        assert_eq!(i.pde_type, Some(PdeType::Elliptic));
        assert_eq!(i.named[0].name, "laplace");
    }

    #[test]
    fn tricomi_regions() {
        let s = s();
        let p = &s.pool;
        let u = p.func("u", vec![s.x, s.y]);
        // u_yy + y·u_xx... Tricomi: y u_xx + u_yy = 0 → Δ = −4y.
        let e = p.add(vec![
            p.mul(vec![s.y, d(&s, u, &[s.x, s.x])]),
            d(&s, u, &[s.y, s.y]),
        ]);
        let i = classify_pde(e, u, p).unwrap();
        assert_eq!(i.pde_type, Some(PdeType::Mixed));
        let types: Vec<&str> = i.regions.iter().map(|r| r.pde_type.as_str()).collect();
        assert_eq!(types, ["hyperbolic", "parabolic", "elliptic"]);
        assert_eq!(i.regions[1].lower, Some(p.integer(0)));
        assert_eq!(i.constant_coefficients, Some(false));
    }

    #[test]
    fn irrational_region_ends_are_exact() {
        let s = s();
        let p = &s.pool;
        let u = p.func("u", vec![s.x, s.y]);
        // (x² − 2) u_xx + u_yy: Δ = −4(x² − 2): hyperbolic for |x| < √2.
        let a = p.add(vec![p.pow(s.x, p.integer(2)), p.integer(-2)]);
        let e = p.add(vec![
            p.mul(vec![a, d(&s, u, &[s.x, s.x])]),
            d(&s, u, &[s.y, s.y]),
        ]);
        let i = classify_pde(e, u, p).unwrap();
        let types: Vec<&str> = i.regions.iter().map(|r| r.pde_type.as_str()).collect();
        assert_eq!(
            types,
            [
                "elliptic",
                "parabolic",
                "hyperbolic",
                "parabolic",
                "elliptic"
            ]
        );
    }

    #[test]
    fn linearity_classes() {
        let s = s();
        let p = &s.pool;
        let u = p.func("u", vec![s.x, s.t]);
        let ut = d(&s, u, &[s.t]);
        let ux = d(&s, u, &[s.x]);
        let uxx = d(&s, u, &[s.x, s.x]);
        let semi = p.add(vec![ut, neg(uxx, p), p.pow(u, p.integer(3))]);
        assert_eq!(
            classify_pde(semi, u, p).unwrap().linearity,
            Linearity::Semilinear
        );
        let quasi = p.add(vec![ut, p.mul(vec![u, ux])]);
        assert_eq!(
            classify_pde(quasi, u, p).unwrap().linearity,
            Linearity::Quasilinear
        );
        let full = p.add(vec![ut, p.pow(ux, p.integer(2))]);
        assert_eq!(
            classify_pde(full, u, p).unwrap().linearity,
            Linearity::FullyNonlinear
        );
        let not_de = p.add(vec![u, s.x]);
        assert!(classify_pde(not_de, u, p).is_err());
    }

    #[test]
    fn inertia_of_constant_matrix() {
        let p = ExprPool::new();
        let i = |k: i64| p.integer(k);
        // [[0, 1], [1, 0]] → one +, one −.
        assert_eq!(
            inertia(&[vec![i(0), i(1)], vec![i(1), i(0)]], &p),
            Some((1, 1, 0))
        );
        // [[1,1,0],[1,1,0],[0,0,-1]] → +, 0, −
        let m = vec![
            vec![i(1), i(1), i(0)],
            vec![i(1), i(1), i(0)],
            vec![i(0), i(0), i(-1)],
        ];
        assert_eq!(inertia(&m, &p), Some((1, 1, 1)));
    }
}
