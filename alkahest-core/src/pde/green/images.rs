//! The method of images: half- and quarter-spaces, strips and slabs (image
//! series as M0's `Sum`), and the disk / ball by Kelvin inversion.

use std::collections::HashMap;

use crate::kernel::subs::subs;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::sturm_liouville::{replace_node, simp};
use super::super::sum::sum;
use super::{
    at, check, check_all, combine, dd, div, fresh, fundamental_solution, is_symbol, neg, pi,
    refuse_refuted, sq, sqrt, sub, trig_to_exp, GreenCheck, GreenFunction, GreenOperator,
};

/// A region bounded by flat walls or a sphere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageRegion {
    /// `x_axis > 0`.
    HalfSpace {
        /// The index of the normal coordinate in the space variables.
        axis: usize,
    },
    /// `x_a > 0, x_b > 0`.
    QuarterSpace {
        /// The two normal coordinates.
        axes: (usize, usize),
    },
    /// `0 < x_axis < width`.
    Strip {
        /// The normal coordinate.
        axis: usize,
        /// The width.
        width: ExprId,
    },
    /// `|x| < radius` in the plane (Laplace, Dirichlet).
    Disk {
        /// The radius.
        radius: ExprId,
    },
    /// `|x| < radius` in space (Laplace, Dirichlet).
    Ball {
        /// The radius.
        radius: ExprId,
    },
}

/// The condition on a wall.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WallCondition {
    /// `G = 0`.
    Dirichlet,
    /// `∂G/∂n = 0`.
    Neumann,
}

impl WallCondition {
    /// `"dirichlet"` or `"neumann"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            WallCondition::Dirichlet => "dirichlet",
            WallCondition::Neumann => "neumann",
        }
    }

    fn sign(&self) -> i64 {
        match self {
            WallCondition::Dirichlet => -1,
            WallCondition::Neumann => 1,
        }
    }
}

/// `e` with `space[i] → space[i] − pt[i]`.
fn shift(e: ExprId, space: &[ExprId], pt: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    for (&v, &p) in space.iter().zip(pt) {
        m.insert(v, sub(v, p, pool));
    }
    subs(e, &m, pool)
}

/// `diracdelta(a)` → an arbitrary profile `F(a)` (for differentiation in
/// checks: every identity proved for all `F` holds for `δ`).
fn undelta(e: ExprId, pool: &ExprPool) -> ExprId {
    let mut out = e;
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if let ExprData::Func { name, args } = pool.get(n) {
            if name == "diracdelta" && args.len() == 1 {
                out = replace_node(out, n, pool.func("__green_F", vec![args[0]]), pool);
                continue;
            }
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    out
}

fn wall_check(
    g: ExprId,
    var: ExprId,
    at_value: ExprId,
    cond: WallCondition,
    label: &str,
    pool: &ExprPool,
) -> Result<GreenCheck, PdeError> {
    let e = match cond {
        WallCondition::Dirichlet => g,
        WallCondition::Neumann => dd(g, var, pool)?,
    };
    Ok(check(
        "bc",
        at(e, var, at_value, pool),
        "substitution",
        format!(
            "{} on {label}",
            if cond == WallCondition::Dirichlet {
                "G = 0"
            } else {
                "∂G/∂n = 0"
            }
        ),
        pool,
    ))
}

/// The Green's function of `op` on a region by the method of images.
///
/// * `HalfSpace { axis }` with one wall condition and `QuarterSpace { axes }`
///   with two: every operator of the table (each is invariant under the
///   reflections `x_a → −x_a`).  Dirichlet half-spaces of the Laplacian also
///   get the Poisson kernel `P = ∂G/∂n_ξ` on the wall.
/// * `Strip { axis, width }` with two wall conditions: the heat operator (any
///   combination of Dirichlet / Neumann; the image series
///   `Σ_j σ^j [K(x − ξ − 2jL) + s₀ K(x + ξ − 2jL)]`), and the Laplacian with
///   Dirichlet walls (the series, and in the plane the closed form
///   `(1/4π) log[(cosh(πX/L) − cos(π(y−η)/L)) / (cosh(πX/L) − cos(π(y+η)/L))]`
///   from the conformal map `w = e^{πz/L}`).
///   The Helmholtz operator on a strip (2 variables) or slab (3 variables)
///   with Dirichlet / Neumann walls: the transverse mode series
///   `Σ φₙ(y)φₙ(η)/‖φₙ‖² gₙ` with outgoing longitudinal kernels of wavenumber
///   `κₙ = √(k² − λₙ)` (principal branch: the radiation condition), method
///   `"mode_series"`; `E-PDE-260` at a cutoff `k² = λₙ`.
/// * `Disk { radius }` / `Ball { radius }` with one Dirichlet condition: the
///   Laplacian, by Kelvin inversion, with the Poisson kernel
///   `(R² − |x|²) / (|S^{n−1}| R |x − ξ|ⁿ)`.
/// * `Disk { radius }` with a Neumann condition: the Neumann function
///   `N = (1/4π) log(|x − ξ|² Q/R²) − (|x|² + |ξ|²)/(4πR²)`
///   (`ΔN = δ − 1/(πR²)`, `∂N/∂n = 0`; method `"neumann_function"`), whose
///   `poisson_kernel` is the boundary kernel `−N(x, ξ)`, `|ξ| = R`, of the
///   Neumann data — see [`neumann_disk_solve`](super::neumann_disk_solve).
///
/// `space` are the field variables, `source` the source point (symbols), and
/// `time` the time variable of an evolution operator (the source time is 0).
///
/// # Errors
///
/// * `E-PDE-251` — mismatched lengths, non-symbol coordinates, a wall axis
///   out of range;
/// * `E-PDE-252` — a region / operator / condition combination without an
///   image construction here;
/// * `E-PDE-255` — a check refuted the construction (never returned).
pub fn image_green_function(
    op: GreenOperator,
    space: &[ExprId],
    time: Option<ExprId>,
    source: &[ExprId],
    region: ImageRegion,
    walls: &[WallCondition],
    pool: &ExprPool,
) -> Result<GreenFunction, PdeError> {
    let n = space.len();
    if source.len() != n {
        return Err(PdeError::GreenInvalidInput(format!(
            "a source point with {} coordinates for {n} space variables",
            source.len()
        )));
    }
    for &v in source {
        if !is_symbol(v, pool) || space.contains(&v) {
            return Err(PdeError::GreenInvalidInput(format!(
                "the source coordinate {} must be a symbol distinct from the space variables",
                pool.display(v)
            )));
        }
    }
    let axis_ok = |a: usize| -> Result<(), PdeError> {
        if a >= n {
            Err(PdeError::GreenInvalidInput(format!(
                "wall axis {a} for {n} space variables"
            )))
        } else {
            Ok(())
        }
    };
    let need_walls = |k: usize| -> Result<(), PdeError> {
        if walls.len() != k {
            Err(PdeError::GreenInvalidInput(format!(
                "{} wall conditions given, {k} expected",
                walls.len()
            )))
        } else {
            Ok(())
        }
    };
    match region {
        ImageRegion::HalfSpace { axis } => {
            axis_ok(axis)?;
            need_walls(1)?;
            reflections(op, space, time, source, &[(axis, walls[0])], pool)
        }
        ImageRegion::QuarterSpace { axes: (a, b) } => {
            axis_ok(a)?;
            axis_ok(b)?;
            if a == b {
                return Err(PdeError::GreenInvalidInput(
                    "a quarter-space needs two different axes".into(),
                ));
            }
            need_walls(2)?;
            reflections(
                op,
                space,
                time,
                source,
                &[(a, walls[0]), (b, walls[1])],
                pool,
            )
        }
        ImageRegion::Strip { axis, width } => {
            axis_ok(axis)?;
            need_walls(2)?;
            strip(op, space, time, source, axis, width, walls, pool)
        }
        ImageRegion::Disk { radius } | ImageRegion::Ball { radius } => {
            need_walls(1)?;
            let want = if matches!(region, ImageRegion::Disk { .. }) {
                2
            } else {
                3
            };
            if op == GreenOperator::Laplace
                && walls[0] == WallCondition::Neumann
                && n == 2
                && want == 2
            {
                return super::neumann_disk::neumann_function_disk(space, source, radius, pool);
            }
            if op != GreenOperator::Laplace || walls[0] != WallCondition::Dirichlet || n != want {
                return Err(PdeError::GreenRegionUnsupported(format!(
                    "Kelvin inversion is implemented for the Dirichlet Laplacian on the disk \
                     (2 variables) and the ball (3 variables), and the Neumann function for the \
                     Laplacian on the disk, not {} / {} / {n} variables",
                    op.name(),
                    walls[0].as_str()
                )));
            }
            kelvin(space, source, radius, pool)
        }
    }
}

/// The `pde` check of an image sum: every term is `K(x − p)` for a point `p`
/// outside the region, and `L` has constant coefficients, so
/// `L K(x − p) = (L K)(x − p) = 0` there by the table's own check of `K`.
fn translated(table: &[GreenCheck], count: usize) -> GreenCheck {
    let src: Vec<&GreenCheck> = table
        .iter()
        .filter(|c| c.what == "homogeneous" || c.what == "homogeneous_radial")
        .collect();
    let status = if !src.is_empty() && src.iter().all(|c| c.status == ZeroStatus::ProvedZero) {
        ZeroStatus::ProvedZero
    } else {
        ZeroStatus::Unknown
    };
    GreenCheck {
        what: "pde",
        status,
        method: "translation invariance: each term is K(x - p) and L K = 0 off the source (the table's check)".into(),
        detail: if count == 0 {
            "L S_j = 0 in the strip for every term of the series".into()
        } else {
            format!("L G = 0 in the region: {count} translates of K")
        },
    }
}

/// Half- and quarter-spaces: the reflection group of the walls.
fn reflections(
    op: GreenOperator,
    space: &[ExprId],
    time: Option<ExprId>,
    source: &[ExprId],
    walls: &[(usize, WallCondition)],
    pool: &ExprPool,
) -> Result<GreenFunction, PdeError> {
    let fs = fundamental_solution(op, space, time, pool)?;
    let smooth = undelta(fs.smooth, pool);
    let mut images: Vec<(Vec<ExprId>, i64)> = vec![(source.to_vec(), 1)];
    for &(a, w) in walls {
        let mut more = Vec::new();
        for (pt, s) in &images {
            let mut q = pt.clone();
            q[a] = neg(q[a], pool);
            more.push((q, s * w.sign()));
        }
        images.extend(more);
    }
    let term = |e: ExprId, pt: &[ExprId], s: i64| {
        let sh = shift(e, space, pt, pool);
        if s == 1 {
            sh
        } else {
            neg(sh, pool)
        }
    };
    let value = simp(
        pool.add(images.iter().map(|(p, s)| term(fs.kernel, p, *s)).collect()),
        pool,
    );
    let g_check = pool.add(images.iter().map(|(p, s)| term(smooth, p, *s)).collect());
    let mut checks = vec![translated(&fs.checks, images.len())];
    for &(a, w) in walls {
        checks.push(wall_check(
            g_check,
            space[a],
            pool.integer(0),
            w,
            &format!("{} = 0", pool.display(space[a])),
            pool,
        )?);
    }
    let side: Vec<ExprId> = walls
        .iter()
        .map(|&(a, _)| pool.pred_gt(source[a], pool.integer(0)))
        .chain(fs.side_conditions.iter().copied())
        .collect();
    let mut notes = vec![format!(
        "{} images of the source, reflected across the walls; every image lies outside the \
         region, so G − (fundamental solution at the source) solves the homogeneous equation \
         in it",
        images.len() - 1
    )];
    if op.is_time_dependent() {
        notes.push(
            "the causal (heaviside) factors of the images are the reflections of the source's, \
             so the wall conditions hold for the kernel as a distribution"
                .into(),
        );
    }
    // Poisson kernel of the Dirichlet Laplacian in a half-space.
    let mut poisson = None;
    let mut bvars = Vec::new();
    if op == GreenOperator::Laplace
        && walls.len() == 1
        && walls[0].1 == WallCondition::Dirichlet
        && space.len() >= 2
    {
        let a = walls[0].0;
        let p = simp(
            neg(
                at(
                    dd(g_check, source[a], pool)?,
                    source[a],
                    pool.integer(0),
                    pool,
                ),
                pool,
            ),
            pool,
        );
        checks.push(check(
            "poisson_kernel",
            GreenOperator::Laplace.apply(p, space, None, pool)?,
            "substitution",
            "the Poisson kernel P = −∂G/∂ξ_a|_{ξ_a = 0} is harmonic in x",
            pool,
        ));
        if space.len() == 2 {
            let b = 1 - a;
            let oo = pool.pos_infinity();
            match crate::integrate::integrate_definite(p, source[b], neg(oo, pool), oo, pool) {
                Ok(m) => checks.push(check(
                    "poisson_mass",
                    sub(m.value, pool.integer(1), pool),
                    "integrate",
                    "∫ P dξ = 1 over the wall",
                    pool,
                )),
                Err(_) => notes.push("∫ P over the wall did not close".into()),
            }
        }
        bvars = source
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != a)
            .map(|(_, &v)| v)
            .collect();
        poisson = Some(p);
    }
    refuse_refuted("the image Green's function", &checks)?;
    let status = combine(&checks);
    let mut references = vec![
        "Strauss, PDE §7.4 (half-space), §7.3 (images)".to_string(),
        "Haberman §9.5.6–9.5.8 (method of images)".to_string(),
        "Duffy, Green's Functions with Applications §5.2".to_string(),
    ];
    references.extend(fs.references.iter().cloned());
    Ok(GreenFunction {
        operator: op.name().into(),
        method: "images".into(),
        space: space.to_vec(),
        source: source.to_vec(),
        time,
        value,
        series: None,
        branches: None,
        poisson_kernel: poisson,
        boundary_vars: bvars,
        checks,
        status,
        side_conditions: side,
        notes,
        references,
    })
}

/// Strips / slabs: the image series.
#[allow(clippy::too_many_arguments)]
fn strip(
    op: GreenOperator,
    space: &[ExprId],
    time: Option<ExprId>,
    source: &[ExprId],
    axis: usize,
    width: ExprId,
    walls: &[WallCondition],
    pool: &ExprPool,
) -> Result<GreenFunction, PdeError> {
    let n = space.len();
    if let GreenOperator::Helmholtz { k } = op {
        return super::waveguide::helmholtz_waveguide(k, space, source, axis, width, walls, pool);
    }
    let heat = matches!(op, GreenOperator::Heat { .. });
    let laplace_dd = op == GreenOperator::Laplace
        && (n == 2 || n == 3)
        && walls.iter().all(|&w| w == WallCondition::Dirichlet);
    if !heat && !laplace_dd {
        return Err(PdeError::GreenRegionUnsupported(format!(
            "image series on a strip are implemented for the heat operator and the Dirichlet \
             Laplacian in 2 or 3 variables, not {} with {} / {}",
            op.name(),
            walls[0].as_str(),
            walls[1].as_str()
        )));
    }
    let fs = fundamental_solution(op, space, time, pool)?;
    let mut avoid: Vec<ExprId> = space.to_vec();
    avoid.extend(source);
    avoid.extend(time);
    avoid.push(width);
    avoid.extend(op.parameters().iter().map(|&(_, v)| v));
    let j = fresh("j", Domain::Integer, &avoid, pool);
    let s0 = walls[0].sign();
    let sigma = s0 * walls[1].sign();
    let two_l = pool.mul(vec![pool.integer(2), width]);
    // S(jj) with the sign factor `sp` (σ^jj).
    let summand = |e: ExprId, jj: ExprId, sp: ExprId| {
        let mut p1 = source.to_vec();
        p1[axis] = pool.add(vec![source[axis], pool.mul(vec![two_l, jj])]);
        let mut p2 = source.to_vec();
        p2[axis] = pool.add(vec![neg(source[axis], pool), pool.mul(vec![two_l, jj])]);
        let a = shift(e, space, &p1, pool);
        let b = shift(e, space, &p2, pool);
        pool.mul(vec![
            sp,
            pool.add(vec![a, pool.mul(vec![pool.integer(s0), b])]),
        ])
    };
    let sig_pow = |k: ExprId| {
        if sigma == 1 {
            pool.integer(1)
        } else {
            pool.pow(pool.integer(-1), k)
        }
    };
    let body = simp(summand(fs.kernel, j, sig_pow(j)), pool);
    let series = sum(
        body,
        j,
        neg(pool.pos_infinity(), pool),
        pool.pos_infinity(),
        pool,
    )?;
    let smooth = undelta(fs.smooth, pool);
    let mut checks = Vec::new();
    checks.push(translated(&fs.checks, 0));
    let xa = space[axis];
    // Wall 0: S(j) + S(−j) = 0 (with σ^{−j} = σ^j).
    let pair0 = |e: ExprId| -> Result<ExprId, PdeError> {
        let a = summand(e, j, sig_pow(j));
        let b = summand(e, neg(j, pool), sig_pow(j));
        Ok(match walls[0] {
            WallCondition::Dirichlet => at(pool.add(vec![a, b]), xa, pool.integer(0), pool),
            WallCondition::Neumann => at(
                pool.add(vec![dd(a, xa, pool)?, dd(b, xa, pool)?]),
                xa,
                pool.integer(0),
                pool,
            ),
        })
    };
    checks.push(check(
        "pairing",
        pair0(smooth)?,
        "substitution (term j paired with −j; σ^{−j} = σ^j)",
        format!(
            "{} at {} = 0 term by term",
            walls[0].as_str(),
            pool.display(xa)
        ),
        pool,
    ));
    // Wall L: S(j) + S(1 − j) = 0 (σ^{1−j} = σ·σ^j).
    let a = summand(smooth, j, sig_pow(j));
    let b = summand(
        smooth,
        sub(pool.integer(1), j, pool),
        pool.mul(vec![pool.integer(sigma), sig_pow(j)]),
    );
    let pl = match walls[1] {
        WallCondition::Dirichlet => at(pool.add(vec![a, b]), xa, width, pool),
        WallCondition::Neumann => at(
            pool.add(vec![dd(a, xa, pool)?, dd(b, xa, pool)?]),
            xa,
            width,
            pool,
        ),
    };
    checks.push(check(
        "pairing",
        pl,
        "substitution (term j paired with 1 − j; σ^{1−j} = σ·σ^j)",
        format!(
            "{} at {} = {} term by term",
            walls[1].as_str(),
            pool.display(xa),
            pool.display(width)
        ),
        pool,
    ));
    let mut notes = vec![
        "image series over j ∈ ℤ: the sources ξ + 2jL and the reflections −ξ + 2jL; the wall \
         conditions hold because the terms cancel in the pairs checked (j ↔ −j at the first \
         wall, j ↔ 1 − j at the second)"
            .to_string(),
    ];
    let mut value = series;
    let mut closed_series = None;
    let mut method = "image_series".to_string();
    if heat {
        notes.push(
            "for t > 0 the series and its derivatives converge absolutely and locally \
             uniformly (Gaussian tails), which justifies the term-wise checks and the \
             re-indexing"
                .into(),
        );
    } else if n == 3 {
        notes.push(
            "the paired summand is O(1/j²), so the series converges absolutely away from the \
             images"
                .into(),
        );
    } else {
        notes.push(
            "in the plane the paired summand is O(1/j): the series converges with symmetric \
             partial sums Σ_{|j| ≤ N}; the closed form below is its value"
                .into(),
        );
        // Closed form by the conformal map w = e^{πz/L}, z = x_b + i x_a.
        let b_axis = 1 - axis;
        let (xb, xi_b, xi_a) = (space[b_axis], source[b_axis], source[axis]);
        let pil = div(pi(pool), width, pool);
        let ch = pool.func("cosh", vec![pool.mul(vec![pil, sub(xb, xi_b, pool)])]);
        let cm = pool.func("cos", vec![pool.mul(vec![pil, sub(xa, xi_a, pool)])]);
        let cp = pool.func("cos", vec![pool.mul(vec![pil, pool.add(vec![xa, xi_a])])]);
        let closed = pool.mul(vec![
            div(
                pool.integer(1),
                pool.mul(vec![pool.integer(4), pi(pool)]),
                pool,
            ),
            pool.func("log", vec![div(sub(ch, cm, pool), sub(ch, cp, pool), pool)]),
        ]);
        let parts = [
            GreenOperator::Laplace.apply(
                pool.func("log", vec![sub(ch, cm, pool)]),
                space,
                None,
                pool,
            )?,
            GreenOperator::Laplace.apply(
                pool.func("log", vec![sub(ch, cp, pool)]),
                space,
                None,
                pool,
            )?,
        ];
        checks.push(check_all(
            "pde",
            &parts,
            "substitution (log(cosh X - cos Y) is harmonic off its zeros)",
            "the closed form is harmonic away from the source",
            pool,
        ));
        // log(N/D) = 0 on a wall iff N = D there.
        for (val, label) in [
            (pool.integer(0), "the first wall"),
            (width, "the second wall"),
        ] {
            let r = at(sub(cm, cp, pool), xa, val, pool);
            checks.push(check(
                "bc",
                r,
                "substitution (cos(z + m pi) = (-1)^m cos z)",
                format!("G = 0 on {label}: the two cosines agree"),
                pool,
            ));
        }
        // |w − ω|² = 2 e^{π(x+ξ)/L} (cosh(π(x−ξ)/L) − cos(π(y∓η)/L)).
        for (cos_term, label) in [(cm, "|w − ω|²"), (cp, "|w − ω̄|²")] {
            let sign = if label == "|w − ω|²" { 1 } else { -1 };
            let eta = pool.mul(vec![pool.integer(sign), xi_a]);
            let mod2 = pool.add(vec![
                pool.func("exp", vec![pool.mul(vec![pool.integer(2), pil, xb])]),
                pool.func("exp", vec![pool.mul(vec![pool.integer(2), pil, xi_b])]),
                neg(
                    pool.mul(vec![
                        pool.integer(2),
                        pool.func("exp", vec![pool.mul(vec![pil, pool.add(vec![xb, xi_b])])]),
                        pool.func("cos", vec![pool.mul(vec![pil, sub(xa, eta, pool)])]),
                    ]),
                    pool,
                ),
            ]);
            let want = pool.mul(vec![
                pool.integer(2),
                pool.func("exp", vec![pool.mul(vec![pil, pool.add(vec![xb, xi_b])])]),
                sub(ch, cos_term, pool),
            ]);
            checks.push(check(
                "conformal",
                trig_to_exp(sub(mod2, want, pool), pool),
                "substitution",
                format!("{label} for w = e^{{πz/L}}: G = (1/2π) log|w − ω|/|w − ω̄|"),
                pool,
            ));
        }
        notes.push(
            "closed form: the upper half-plane Green's function (1/2π) log|w − ω|/|w − ω̄| \
             composed with the conformal map w = e^{πz/L} of the strip, z = x + iy (checked: \
             the moduli identities); harmonicity and the logarithmic singularity are preserved \
             by conformal maps"
                .into(),
        );
        closed_series = Some(series);
        value = simp(closed, pool);
        method = "conformal_map".into();
    }
    refuse_refuted("the strip Green's function", &checks)?;
    let status = combine(&checks);
    let mut side = vec![
        pool.pred_gt(source[axis], pool.integer(0)),
        pool.pred_lt(source[axis], width),
        pool.pred_gt(width, pool.integer(0)),
    ];
    side.extend(fs.side_conditions.iter().copied());
    let mut references = vec![
        "Haberman §11.3 (heat equation on a finite interval by images)".to_string(),
        "Duffy, Green's Functions with Applications §5.2, §3.3".to_string(),
        "Morse & Feshbach, Methods of Theoretical Physics §7.2".to_string(),
    ];
    references.extend(fs.references.iter().cloned());
    Ok(GreenFunction {
        operator: op.name().into(),
        method,
        space: space.to_vec(),
        source: source.to_vec(),
        time,
        value,
        series: closed_series,
        branches: None,
        poisson_kernel: None,
        boundary_vars: Vec::new(),
        checks,
        status,
        side_conditions: side,
        notes,
        references,
    })
}

/// Disk and ball: Kelvin inversion.
fn kelvin(
    space: &[ExprId],
    source: &[ExprId],
    radius: ExprId,
    pool: &ExprPool,
) -> Result<GreenFunction, PdeError> {
    let n = space.len();
    let r2 = sq(radius, pool);
    let dot = |a: &[ExprId], b: &[ExprId]| {
        pool.add(
            a.iter()
                .zip(b)
                .map(|(&u, &v)| pool.mul(vec![u, v]))
                .collect(),
        )
    };
    let x2 = dot(space, space);
    let xi2 = dot(source, source);
    let xxi = dot(space, source);
    let d2 = pool.add(
        space
            .iter()
            .zip(source)
            .map(|(&u, &v)| sq(sub(u, v, pool), pool))
            .collect(),
    );
    // Q = |ξ|²|x|² − 2R² x·ξ + R⁴ = |ξ|² |x − ξ*|², ξ* = R² ξ / |ξ|².
    let q = pool.add(vec![
        pool.mul(vec![xi2, x2]),
        neg(pool.mul(vec![pool.integer(2), r2, xxi]), pool),
        sq(r2, pool),
    ]);
    let four_pi = pool.mul(vec![pool.integer(4), pi(pool)]);
    let g = if n == 2 {
        pool.mul(vec![
            div(pool.integer(1), four_pi, pool),
            pool.func("log", vec![div(pool.mul(vec![r2, d2]), q, pool)]),
        ])
    } else {
        pool.add(vec![
            neg(
                div(
                    pool.integer(1),
                    pool.mul(vec![four_pi, sqrt(d2, pool)]),
                    pool,
                ),
                pool,
            ),
            div(radius, pool.mul(vec![four_pi, sqrt(q, pool)]), pool),
        ])
    };
    let mut checks = Vec::new();
    // Each part separately: log(R²|x − ξ|²) − log Q in the plane,
    // −1/(4π|x − ξ|) + R/(4π√Q) in space.
    let parts: Vec<ExprId> = if n == 2 {
        vec![pool.func("log", vec![d2]), pool.func("log", vec![q])]
    } else {
        vec![
            pool.pow(d2, pool.rational(-1, 2)),
            pool.pow(q, pool.rational(-1, 2)),
        ]
    };
    let mut res = Vec::new();
    for part in parts {
        res.push(GreenOperator::Laplace.apply(part, space, None, pool)?);
    }
    checks.push(check_all(
        "pde",
        &res,
        "substitution (the source term and the image term separately)",
        "Δₓ G = 0 for x ≠ ξ in the open region",
        pool,
    ));
    let kel = sub(
        sub(q, pool.mul(vec![r2, d2]), pool),
        pool.mul(vec![sub(x2, r2, pool), sub(xi2, r2, pool)]),
        pool,
    );
    checks.push(check(
        "kelvin",
        kel,
        "polynomial identity",
        "|ξ|²|x − ξ*|² − R²|x − ξ|² = (|x|² − R²)(|ξ|² − R²): G = 0 on |x| = R, and the \
         image term is smooth for |x|, |ξ| < R",
        pool,
    ));
    // The boundary condition: on |x| = R (R → |x|), Q = R²|x − ξ|², so the
    // image term equals the source term (log ratio 0 / R/√Q = 1/|x − ξ|).
    let on_sphere = super::super::sturm_liouville::subs1(
        sub(q, pool.mul(vec![r2, d2]), pool),
        radius,
        sqrt(x2, pool),
        pool,
    );
    checks.push(check(
        "bc",
        super::expand_poly(simp(on_sphere, pool), pool),
        "substitution R = |x|",
        "G = 0 on |x| = R: Q = R²|x − ξ|² there",
        pool,
    ));
    // Poisson kernel P = ∂G/∂n_ξ = (ξ·∇_ξ G)/R on |ξ| = R.
    let omega = if n == 2 {
        pool.mul(vec![pool.integer(2), pi(pool)])
    } else {
        four_pi
    };
    let p = div(
        sub(r2, x2, pool),
        pool.mul(vec![
            omega,
            radius,
            pool.pow(sqrt(d2, pool), pool.integer(n as i64)),
        ]),
        pool,
    );
    // For the normal derivative write the image term with |ξ| |x − ξ*|
    // (= √Q) so that at the pole ξ = (R, 0, …) both radicals are |x − ξ|.
    let g_n = if n == 2 {
        g
    } else {
        let dstar2 = pool.add(
            space
                .iter()
                .zip(source)
                .map(|(&u, &v)| sq(sub(u, div(pool.mul(vec![r2, v]), xi2, pool), pool), pool))
                .collect(),
        );
        pool.add(vec![
            neg(
                div(
                    pool.integer(1),
                    pool.mul(vec![four_pi, sqrt(d2, pool)]),
                    pool,
                ),
                pool,
            ),
            div(
                radius,
                pool.mul(vec![four_pi, sqrt(xi2, pool), sqrt(dstar2, pool)]),
                pool,
            ),
        ])
    };
    let mut grad = Vec::new();
    for &v in source {
        grad.push(pool.mul(vec![v, dd(g_n, v, pool)?]));
    }
    let normal = div(pool.add(grad), radius, pool);
    // Both sides are invariant under rotations about the centre (they depend
    // on |x|, |ξ|, x·ξ only), so it suffices to compare them at the boundary
    // point ξ = (R, 0, …).
    let pole = |e: ExprId| {
        let mut m = HashMap::new();
        for (i, &v) in source.iter().enumerate() {
            m.insert(v, if i == 0 { radius } else { pool.integer(0) });
        }
        simp(subs(e, &m, pool), pool)
    };
    let diff = pole(sub(normal, p, pool));
    let mut pk = check(
        "poisson_kernel",
        diff,
        "substitution at ξ = (R, 0, …) (rotation invariance)",
        "P = ∂G/∂n_ξ on |ξ| = R",
        pool,
    );
    if pk.status != ZeroStatus::ProvedZero {
        pk.status = super::zstatus(super::expand_powers_pos(diff, pool), pool);
    }
    let harm = pole(GreenOperator::Laplace.apply(p, space, None, pool)?);
    let normal_ok = pk.status == ZeroStatus::ProvedZero;
    checks.push(pk);
    let mut hc = check(
        "poisson_kernel",
        harm,
        "substitution at ξ = (R, 0, …) (rotation invariance)",
        "P is harmonic in x for ξ on the sphere",
        pool,
    );
    if hc.status != ZeroStatus::ProvedZero
        && hc.status != ZeroStatus::ProvedNonzero
        && normal_ok
        && checks[0].status == ZeroStatus::ProvedZero
    {
        // P = ∂G/∂n_ξ (checked) and Δₓ commutes with ∂_ξ, so Δₓ P = ∂_n Δₓ G = 0.
        hc.status = ZeroStatus::ProvedZero;
        hc.method = "derived: P = ∂G/∂n_ξ (checked) and Δₓ G = 0 (checked)".into();
    }
    checks.push(hc);
    let area = pool.mul(vec![omega, pool.pow(radius, pool.integer(n as i64 - 1))]);
    let p0_on = {
        // x = 0 and the boundary point ξ = (R, 0, …).
        let mut m = HashMap::new();
        for &v in space {
            m.insert(v, pool.integer(0));
        }
        for (i, &v) in source.iter().enumerate() {
            m.insert(v, if i == 0 { radius } else { pool.integer(0) });
        }
        simp(subs(p, &m, pool), pool)
    };
    checks.push(check(
        "poisson_mass",
        sub(pool.mul(vec![area, p0_on]), pool.integer(1), pool),
        "substitution",
        "|∂B| · P(0, ξ) = 1 (mean-value property at the centre)",
        pool,
    ));
    refuse_refuted("the Kelvin Green's function", &checks)?;
    let status = combine(&checks);
    Ok(GreenFunction {
        operator: "laplace".into(),
        method: "kelvin".into(),
        space: space.to_vec(),
        source: source.to_vec(),
        time: None,
        value: simp(g, pool),
        series: None,
        branches: None,
        poisson_kernel: Some(simp(p, pool)),
        boundary_vars: source.to_vec(),
        checks,
        status,
        side_conditions: vec![pool.pred_lt(xi2, r2), pool.pred_gt(radius, pool.integer(0))],
        notes: vec![
            "Kelvin inversion: the image of ξ is ξ* = R² ξ / |ξ|², with strength chosen so that \
             G vanishes on the sphere; the Poisson kernel's arguments x and ξ are the field \
             point and a boundary point (|ξ| = R)"
                .into(),
            "u(x) = ∫_B G(x, ξ) f(ξ) dξ + ∮_{|ξ|=R} P(x, ξ) h(ξ) dS solves Δu = f, u = h on the \
             sphere (Green's representation formula)"
                .into(),
        ],
        references: vec![
            "Evans, PDE §2.2.4 (Green's function for a ball, Poisson's formula)".into(),
            "Strauss, PDE §7.4 (sphere), §6.3 (Poisson's formula)".into(),
            "Haberman §9.5.7 (circle)".into(),
        ],
    })
}
