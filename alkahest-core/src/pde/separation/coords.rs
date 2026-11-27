//! Separable curvilinear coordinates: polar, cylindrical and spherical
//! Laplacians, the map to Cartesian coordinates, and the rewrite of an
//! isotropic Cartesian equation on a [`PdeDomain::Disk`] / [`PdeDomain::Ball`]
//! into polar / spherical form.
//!
//! [`PdeDomain::Disk`]: super::super::PdeDomain::Disk
//! [`PdeDomain::Ball`]: super::super::PdeDomain::Ball

use std::collections::HashMap;

use crate::kernel::subs::subs;
use crate::kernel::{ExprId, ExprPool};

use super::super::error::PdeError;
use super::super::sturm_liouville::{d, simp};

/// A coordinate system of a [`PdeDomain::Curvilinear`] region.
///
/// * `Polar`: `(r, θ)`, `x = r cos θ`, `y = r sin θ`;
/// * `Cylindrical`: `(r, θ, z)`;
/// * `Spherical`: `(r, θ, φ)` with `θ ∈ [0, π]` the polar angle and `φ` the
///   azimuth: `x = r sin θ cos φ`, `y = r sin θ sin φ`, `z = r cos θ`.
///
/// [`PdeDomain::Curvilinear`]: super::super::PdeDomain::Curvilinear
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CoordinateSystem {
    /// `(r, θ)`.
    Polar,
    /// `(r, θ, z)`.
    Cylindrical,
    /// `(r, θ, φ)`, `θ` the polar angle.
    Spherical,
}

impl CoordinateSystem {
    /// `"polar"`, `"cylindrical"` or `"spherical"`.
    pub fn as_str(self) -> &'static str {
        match self {
            CoordinateSystem::Polar => "polar",
            CoordinateSystem::Cylindrical => "cylindrical",
            CoordinateSystem::Spherical => "spherical",
        }
    }

    /// The number of coordinates.
    pub fn dimension(self) -> usize {
        match self {
            CoordinateSystem::Polar => 2,
            _ => 3,
        }
    }

    /// Parse `"polar"`, `"cylindrical"`, `"spherical"`.
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "polar" => Some(CoordinateSystem::Polar),
            "cylindrical" => Some(CoordinateSystem::Cylindrical),
            "spherical" => Some(CoordinateSystem::Spherical),
            _ => None,
        }
    }
}

fn inv(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(e, pool.integer(-1))
}

/// The Laplacian of `f` in `system`; `coords[i]` is `None` for a coordinate
/// `f` does not depend on (radial and axisymmetric problems).
///
/// # Errors
///
/// `E-PDE-201` when the radius is absent or `coords` has the wrong length.
pub fn laplacian(
    f: ExprId,
    system: CoordinateSystem,
    coords: &[Option<ExprId>],
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    if coords.len() != system.dimension() {
        return Err(PdeError::SeparationUnsupported(format!(
            "{} coordinates need {} entries",
            system.as_str(),
            system.dimension()
        )));
    }
    let r = coords[0].ok_or_else(|| {
        PdeError::SeparationUnsupported("curvilinear coordinates need the radius".into())
    })?;
    let d1 = |v: ExprId| d(f, v, pool);
    let d2 = |v: ExprId| d(f, v, pool).and_then(|g| d(g, v, pool));
    let mut terms = Vec::new();
    match system {
        CoordinateSystem::Polar | CoordinateSystem::Cylindrical => {
            terms.push(d2(r)?);
            terms.push(pool.mul(vec![d1(r)?, inv(r, pool)]));
            if let Some(th) = coords[1] {
                terms.push(pool.mul(vec![d2(th)?, pool.pow(r, pool.integer(-2))]));
            }
            if system == CoordinateSystem::Cylindrical {
                if let Some(z) = coords[2] {
                    terms.push(d2(z)?);
                }
            }
        }
        CoordinateSystem::Spherical => {
            terms.push(d2(r)?);
            terms.push(pool.mul(vec![pool.integer(2), d1(r)?, inv(r, pool)]));
            if let Some(th) = coords[1] {
                let s = pool.func("sin", vec![th]);
                let c = pool.func("cos", vec![th]);
                terms.push(pool.mul(vec![d2(th)?, pool.pow(r, pool.integer(-2))]));
                terms.push(pool.mul(vec![
                    c,
                    inv(s, pool),
                    d1(th)?,
                    pool.pow(r, pool.integer(-2)),
                ]));
                if let Some(ph) = coords[2] {
                    terms.push(pool.mul(vec![
                        d2(ph)?,
                        pool.pow(r, pool.integer(-2)),
                        pool.pow(s, pool.integer(-2)),
                    ]));
                }
            } else if coords[2].is_some() {
                return Err(PdeError::SeparationUnsupported(
                    "a spherical problem that depends on φ must depend on θ".into(),
                ));
            }
        }
    }
    Ok(pool.add(terms))
}

/// `x = r cos θ`, `y = r sin θ` (`z = z`), or the spherical map, as
/// expressions in the curvilinear coordinates, about `center`.  A missing
/// angular coordinate is replaced by `0` (the problem does not depend on
/// it).
pub fn to_cartesian(
    system: CoordinateSystem,
    coords: &[ExprId],
    center: &[ExprId],
    pool: &ExprPool,
) -> Vec<ExprId> {
    let shift = |i: usize, e: ExprId| match center.get(i) {
        Some(&c) if c != pool.integer(0) => pool.add(vec![c, e]),
        _ => e,
    };
    let r = coords[0];
    match system {
        CoordinateSystem::Polar | CoordinateSystem::Cylindrical => {
            let th = coords[1];
            let mut v = vec![
                shift(0, pool.mul(vec![r, pool.func("cos", vec![th])])),
                shift(1, pool.mul(vec![r, pool.func("sin", vec![th])])),
            ];
            if system == CoordinateSystem::Cylindrical {
                v.push(shift(2, coords[2]));
            }
            v
        }
        CoordinateSystem::Spherical => {
            let (th, ph) = (coords[1], coords[2]);
            let st = pool.func("sin", vec![th]);
            vec![
                shift(0, pool.mul(vec![r, st, pool.func("cos", vec![ph])])),
                shift(1, pool.mul(vec![r, st, pool.func("sin", vec![ph])])),
                shift(2, pool.mul(vec![r, pool.func("cos", vec![th])])),
            ]
        }
    }
}

/// `r = |x − c|`, `θ = atan2(y − c_y, x − c_x)` (and the spherical
/// angles), for writing a curvilinear answer back in Cartesian coordinates.
pub fn from_cartesian(
    system: CoordinateSystem,
    cart: &[ExprId],
    center: &[ExprId],
    pool: &ExprPool,
) -> Vec<ExprId> {
    let rel: Vec<ExprId> = cart
        .iter()
        .enumerate()
        .map(|(i, &x)| match center.get(i) {
            Some(&c) if c != pool.integer(0) => {
                pool.add(vec![x, pool.mul(vec![pool.integer(-1), c])])
            }
            _ => x,
        })
        .collect();
    let sq = |e: ExprId| pool.pow(e, pool.integer(2));
    let half = pool.rational(1, 2);
    match system {
        CoordinateSystem::Polar => vec![
            pool.pow(pool.add(vec![sq(rel[0]), sq(rel[1])]), half),
            pool.func("atan2", vec![rel[1], rel[0]]),
        ],
        CoordinateSystem::Cylindrical => vec![
            pool.pow(pool.add(vec![sq(rel[0]), sq(rel[1])]), half),
            pool.func("atan2", vec![rel[1], rel[0]]),
            rel[2],
        ],
        CoordinateSystem::Spherical => {
            let rho = pool.pow(pool.add(vec![sq(rel[0]), sq(rel[1])]), half);
            vec![
                pool.pow(pool.add(vec![sq(rel[0]), sq(rel[1]), sq(rel[2])]), half),
                pool.func("atan2", vec![rho, rel[2]]),
                pool.func("atan2", vec![rel[1], rel[0]]),
            ]
        }
    }
}

/// Rewrite an *isotropic* equation in Cartesian coordinates — `A·Δu + R`
/// with `A` free of the spatial coordinates and `R` free of their
/// derivatives — into `system`.  `cart` are the Cartesian coordinates,
/// `curv` the new ones (same count), `u` the Cartesian unknown and `v` the
/// curvilinear one.
///
/// # Errors
///
/// `E-PDE-201` when the spatial principal part is not a multiple of the
/// Laplacian or first-order spatial derivatives occur.
#[allow(clippy::too_many_arguments)]
pub fn isotropic_to_curvilinear(
    eq: ExprId,
    u: ExprId,
    v: ExprId,
    cart: &[ExprId],
    curv: &[ExprId],
    center: &[ExprId],
    system: CoordinateSystem,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    use crate::diff::undefined::{build, view};
    let uv = view(u, pool).ok_or_else(|| PdeError::NotAnUnknown(pool.display(u).to_string()))?;
    let slots: Vec<usize> = cart
        .iter()
        .map(|c| uv.args.iter().position(|a| a == c))
        .collect::<Option<_>>()
        .ok_or_else(|| {
            PdeError::SeparationUnsupported("the disk / ball coordinates are not arguments".into())
        })?;
    // Replace every spatial derivative node by a symbol, read off A.
    let mut a: Option<ExprId> = None;
    let mut rest = eq;
    let mut lap_syms = Vec::new();
    for &s in &slots {
        let mut ord = vec![0u32; uv.args.len()];
        ord[s] = 2;
        let node = build(u, &ord, pool);
        let sym = pool.symbol(format!("__iso_{s}"), crate::kernel::Domain::Real);
        rest = super::separate::replace_nodes(rest, &HashMap::from([(node, sym)]), pool);
        lap_syms.push(sym);
    }
    // Any other derivative involving a spatial slot is not isotropic.
    let mut stack = vec![rest];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(w) = view(n, pool) {
            if w.name == uv.name && slots.iter().any(|&s| w.orders[s] > 0) {
                return Err(PdeError::SeparationUnsupported(format!(
                    "{} is not isotropic in the disk / ball coordinates ({} occurs)",
                    pool.display(eq),
                    pool.display(n)
                )));
            }
            if w.name == uv.name && w.args != uv.args {
                return Err(PdeError::SeparationUnsupported(format!(
                    "{} occurs off its arguments",
                    pool.display(n)
                )));
            }
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    for &sym in &lap_syms {
        let c = simp(d(rest, sym, pool)?, pool);
        if super::super::jet::symbols_of(c, pool)
            .iter()
            .any(|s| lap_syms.contains(s) || cart.contains(s))
        {
            return Err(PdeError::SeparationUnsupported(
                "the coefficient of the Laplacian depends on the coordinates".into(),
            ));
        }
        match a {
            None => a = Some(c),
            Some(a0) => {
                if simp(
                    pool.add(vec![a0, pool.mul(vec![pool.integer(-1), c])]),
                    pool,
                ) != pool.integer(0)
                {
                    return Err(PdeError::SeparationUnsupported(format!(
                        "{} is not isotropic: the second derivatives have different \
                         coefficients",
                        pool.display(eq)
                    )));
                }
            }
        }
    }
    let a = a.unwrap_or(pool.integer(0));
    let zero: HashMap<ExprId, ExprId> = lap_syms.iter().map(|&s| (s, pool.integer(0))).collect();
    let r0 = subs(rest, &zero, pool);
    // R with u → v, x → r cos θ, …
    let mut new_args = uv.args.clone();
    for (k, &s) in slots.iter().enumerate() {
        new_args[s] = curv[k];
    }
    let mut m: HashMap<ExprId, ExprId> = HashMap::new();
    let cmap = to_cartesian(system, curv, center, pool);
    for (k, &c) in cart.iter().enumerate() {
        m.insert(c, cmap[k]);
    }
    let r_new = subs(
        super::separate::replace_nodes(r0, &HashMap::from([(u, v)]), pool),
        &m,
        pool,
    );
    // Derivatives in the other variables (t) of u → of v.
    let r_new = crate::diff::undefined::replace_function(r_new, u, v, pool).unwrap_or(r_new);
    let coords: Vec<Option<ExprId>> = curv.iter().map(|&c| Some(c)).collect();
    let lap = laplacian(v, system, &coords, pool)?;
    let out = pool.add(vec![pool.mul(vec![a, lap]), r_new]);
    Ok(simp(out, pool))
}
