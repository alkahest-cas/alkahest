//! Vector calculus over orthogonal curvilinear coordinates.
//!
//! `grad`, `div`, `curl` and `∇²` for a scalar or vector field written in an
//! orthogonal coordinate system, plus the vector algebra
//! ([`dot`], [`cross`], [`norm`]) the differential operators are usually
//! combined with. Cartesian, cylindrical and spherical systems are built in;
//! [`Coordinates::from_embedding`] derives a new one from its Cartesian
//! parametrisation and **checks that it is orthogonal** before returning it.
//!
//! # Components are *physical* components
//!
//! A vector field is a `[ExprId; 3]` read in the local **orthonormal** frame
//! `(ê₁, ê₂, ê₃)`, i.e. `F = F₁ê₁ + F₂ê₂ + F₃ê₃` with `|êᵢ| = 1`. This is the
//! engineering convention and the one `sympy.vector` uses; it is *not* the
//! contravariant-component convention, where the basis vectors carry the scale
//! factors. Getting this wrong is a factor of `hᵢ` per component, so it is
//! stated here rather than left to be inferred.
//!
//! # The formulas
//!
//! With scale factors `h = (h₁, h₂, h₃)` and `H = h₁h₂h₃`:
//!
//! ```text
//! (∇f)ᵢ     = (1/hᵢ) ∂f/∂uᵢ
//! ∇·F       = (1/H) Σᵢ ∂/∂uᵢ ( (H/hᵢ) Fᵢ )
//! (∇×F)ᵢ    = (1/(h_j h_k)) [ ∂(h_k F_k)/∂u_j − ∂(h_j F_j)/∂u_k ],  (i,j,k) cyclic
//! ∇²f       = (1/H) Σᵢ ∂/∂uᵢ ( (H/hᵢ²) ∂f/∂uᵢ )
//! ```
//!
//! (Morse & Feshbach, *Methods of Theoretical Physics* I §1.3; Arfken & Weber,
//! *Mathematical Methods for Physicists* 7th ed. §3.10.)
//!
//! The **vector** Laplacian is `∇²F ≔ ∇(∇·F) − ∇×(∇×F)`
//! ([`vector_laplacian`]). In Cartesian coordinates that equals the
//! componentwise scalar Laplacian and is computed that way; in every other
//! system it does **not**, and applying the scalar Laplacian to each physical
//! component is a classic silent error this module declines to make.
//!
//! # What is not checked
//!
//! Scale factors say nothing about where a chart is *singular*. `∇×F` in
//! spherical coordinates divides by `r sin θ`, which vanishes on the polar
//! axis; the returned expression is the correct one everywhere the chart is
//! regular and is undefined (`E-EVAL-009` from `eval_expr`) on the axis, the
//! same way `1/x` is undefined at `0`. No pointwise domain condition is
//! attached.

use crate::diff::diff;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::fmt;

mod coords;
#[cfg(test)]
mod tests;

pub use coords::Coordinates;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a vector-calculus request could not be answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorError {
    /// A component could not be differentiated with respect to a coordinate.
    Differentiation(String),
    /// Two of the three coordinates are the same expression.
    ///
    /// `∂/∂u₁` and `∂/∂u₂` would then be the same operator, which turns `curl`
    /// into the zero field and `div` into a sum of the wrong partials — a
    /// clean, plausible, wrong answer.
    RepeatedCoordinate {
        /// Rendered form of the coordinate that appears twice.
        coordinate: String,
    },
    /// A coordinate is not a symbol, so it cannot be differentiated against.
    NotACoordinateSymbol {
        /// Rendered form of the offending entry.
        got: String,
    },
    /// The supplied embedding is not an orthogonal coordinate system: two
    /// tangent vectors `∂r/∂uᵢ`, `∂r/∂u_j` have a dot product that was not
    /// shown to vanish.
    ///
    /// Every formula in this module assumes orthogonality. Applying them to a
    /// skew chart returns numbers that look like a divergence and are not one.
    NonOrthogonal {
        /// The two coordinate indices (0-based) whose tangents are not
        /// orthogonal.
        pair: (usize, usize),
        /// Rendered form of the non-vanishing inner product.
        inner_product: String,
    },
    /// A scale factor `hᵢ` is identically zero, or could not be shown to be
    /// non-zero. Every operator divides by it.
    DegenerateScaleFactor {
        /// Which scale factor (0-based).
        index: usize,
        /// Rendered form of the scale factor.
        scale_factor: String,
        /// `true` when `hᵢ` was *proven* zero, `false` when the vanishing
        /// question could not be settled either way.
        proven_zero: bool,
    },
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::Differentiation(msg) => {
                write!(f, "could not differentiate a field component: {msg}")
            }
            VectorError::RepeatedCoordinate { coordinate } => write!(
                f,
                "coordinate `{coordinate}` appears twice; the three coordinates must be distinct symbols"
            ),
            VectorError::NotACoordinateSymbol { got } => write!(
                f,
                "`{got}` is not a symbol and cannot be used as a coordinate"
            ),
            VectorError::NonOrthogonal {
                pair,
                inner_product,
            } => write!(
                f,
                "the embedding is not orthogonal: tangent vectors {} and {} have inner product `{inner_product}`, which was not shown to vanish",
                pair.0, pair.1
            ),
            VectorError::DegenerateScaleFactor {
                index,
                scale_factor,
                proven_zero,
            } => {
                if *proven_zero {
                    write!(
                        f,
                        "scale factor h{} = `{scale_factor}` is identically zero; the chart is degenerate",
                        index + 1
                    )
                } else {
                    write!(
                        f,
                        "scale factor h{} = `{scale_factor}` could not be shown to be non-zero, and every operator divides by it",
                        index + 1
                    )
                }
            }
        }
    }
}

impl std::error::Error for VectorError {}

impl crate::errors::AlkahestError for VectorError {
    fn code(&self) -> &'static str {
        match self {
            VectorError::Differentiation(_) => "E-VEC-001",
            VectorError::RepeatedCoordinate { .. } => "E-VEC-002",
            VectorError::NotACoordinateSymbol { .. } => "E-VEC-003",
            VectorError::NonOrthogonal { .. } => "E-VEC-004",
            VectorError::DegenerateScaleFactor { .. } => "E-VEC-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            VectorError::Differentiation(_) => Some(
                "the field contains a function with no differentiation rule; register it in PrimitiveRegistry or rewrite the component",
            ),
            VectorError::RepeatedCoordinate { .. } => Some(
                "pass three distinct coordinate symbols, e.g. Coordinates::cylindrical(rho, phi, z)",
            ),
            VectorError::NotACoordinateSymbol { .. } => {
                Some("build each coordinate with `pool.symbol(..)` before constructing Coordinates")
            }
            VectorError::NonOrthogonal { .. } => Some(
                "these formulas hold only for orthogonal charts; supply an orthogonal embedding, or work in Cartesian coordinates and transform the result by hand",
            ),
            VectorError::DegenerateScaleFactor { .. } => Some(
                "restrict the chart to a region where the scale factor is non-zero, or re-parametrise; a chart that collapses a direction has no unique orthonormal frame there",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Vector algebra
// ---------------------------------------------------------------------------

/// `a · b` in an orthonormal frame: `Σ aᵢ bᵢ`, simplified.
///
/// Valid in any orthogonal coordinate system *because* the components are
/// physical (see the module docs): the metric is the identity in the
/// orthonormal frame.
pub fn dot(a: &[ExprId; 3], b: &[ExprId; 3], pool: &ExprPool) -> ExprId {
    let terms: Vec<ExprId> = (0..3).map(|i| pool.mul(vec![a[i], b[i]])).collect();
    simplify(pool.add(terms), pool).value
}

/// `a × b` in a **right-handed** orthonormal frame, simplified.
///
/// `(ê₁, ê₂, ê₃)` is assumed right-handed, which is what makes
/// `ê₁ × ê₂ = ê₃`. All three built-in charts are right-handed in the order
/// their constructors take: `(x, y, z)`, `(ρ, φ, z)` and `(r, θ, φ)`.
pub fn cross(a: &[ExprId; 3], b: &[ExprId; 3], pool: &ExprPool) -> [ExprId; 3] {
    let neg = pool.integer(-1_i32);
    let mut out = [a[0]; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let plus = pool.mul(vec![a[j], b[k]]);
        let minus = pool.mul(vec![neg, a[k], b[j]]);
        *slot = simplify(pool.add(vec![plus, minus]), pool).value;
    }
    out
}

/// `|a|² = a · a`, simplified.
pub fn norm_squared(a: &[ExprId; 3], pool: &ExprPool) -> ExprId {
    dot(a, a, pool)
}

/// `|a| = √(a · a)`, simplified.
///
/// The principal (non-negative) square root, which is the length only when the
/// components are real. For complex components this is `√(Σ aᵢ²)`, not the
/// Hermitian norm `√(Σ |aᵢ|²)`.
pub fn norm(a: &[ExprId; 3], pool: &ExprPool) -> ExprId {
    let sq = norm_squared(a, pool);
    simplify(pool.func("sqrt", vec![sq]), pool).value
}

/// `s · a`, simplified componentwise.
pub fn scale(s: ExprId, a: &[ExprId; 3], pool: &ExprPool) -> [ExprId; 3] {
    let mut out = [a[0]; 3];
    for i in 0..3 {
        out[i] = simplify(pool.mul(vec![s, a[i]]), pool).value;
    }
    out
}

/// `a + b`, simplified componentwise.
pub fn add(a: &[ExprId; 3], b: &[ExprId; 3], pool: &ExprPool) -> [ExprId; 3] {
    let mut out = [a[0]; 3];
    for i in 0..3 {
        out[i] = simplify(pool.add(vec![a[i], b[i]]), pool).value;
    }
    out
}

/// `a − b`, simplified componentwise.
pub fn sub(a: &[ExprId; 3], b: &[ExprId; 3], pool: &ExprPool) -> [ExprId; 3] {
    let neg = pool.integer(-1_i32);
    let mut out = [a[0]; 3];
    for i in 0..3 {
        let nb = pool.mul(vec![neg, b[i]]);
        out[i] = simplify(pool.add(vec![a[i], nb]), pool).value;
    }
    out
}

// ---------------------------------------------------------------------------
// Differential operators
// ---------------------------------------------------------------------------

fn d(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<ExprId, VectorError> {
    diff(expr, var, pool)
        .map(|r| r.value)
        .map_err(|e| VectorError::Differentiation(e.to_string()))
}

/// True for the interned literal `1`, which lets the Cartesian path skip a
/// pointless `1⁻¹ ·` wrapper on every component.
fn is_one(id: ExprId, pool: &ExprPool) -> bool {
    pool.with(id, |data| matches!(data, ExprData::Integer(n) if n.0 == 1))
}

fn divide(num: ExprId, den: ExprId, pool: &ExprPool) -> ExprId {
    if is_one(den, pool) {
        return num;
    }
    let inv = pool.pow(den, pool.integer(-1_i32));
    pool.mul(vec![inv, num])
}

fn times(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    if is_one(a, pool) {
        return b;
    }
    if is_one(b, pool) {
        return a;
    }
    pool.mul(vec![a, b])
}

/// `num/den`, normalised before anything differentiates it.
///
/// The weights these operators build are ratios of scale factors, and they
/// cancel more often than not: `H/h₂` is `ρ·ρ⁻¹ = 1` for the azimuthal index of
/// a cylindrical chart. Differentiating the un-cancelled form is correct but
/// fires the product rule on two factors that annihilate, and the debris
/// survives into the returned expression even after the final `simplify`. This
/// is a readability and cost measure only — every result is the same function
/// either way, which is what the textbook-form tests check.
fn weight(num: ExprId, den: ExprId, pool: &ExprPool) -> ExprId {
    if is_one(den, pool) {
        return num;
    }
    simplify(divide(num, den, pool), pool).value
}

/// `∇f` — the gradient of a scalar field, in physical components.
///
/// `(∇f)ᵢ = (1/hᵢ) ∂f/∂uᵢ`.
pub fn gradient(
    f: ExprId,
    coords: &Coordinates,
    pool: &ExprPool,
) -> Result<[ExprId; 3], VectorError> {
    let vars = coords.vars();
    let h = coords.scale_factors();
    let mut out = [f; 3];
    for i in 0..3 {
        let df = d(f, vars[i], pool)?;
        out[i] = simplify(divide(df, h[i], pool), pool).value;
    }
    Ok(out)
}

/// `∇·F` — the divergence of a vector field given in physical components.
///
/// `∇·F = (1/H) Σᵢ ∂/∂uᵢ ( (H/hᵢ) Fᵢ )` with `H = h₁h₂h₃`.
pub fn divergence(
    field: &[ExprId; 3],
    coords: &Coordinates,
    pool: &ExprPool,
) -> Result<ExprId, VectorError> {
    let vars = coords.vars();
    let h = coords.scale_factors();
    let big_h = coords.jacobian_factor(pool);
    let mut terms = Vec::with_capacity(3);
    for i in 0..3 {
        let weighted = times(weight(big_h, h[i], pool), field[i], pool);
        terms.push(d(weighted, vars[i], pool)?);
    }
    let total = pool.add(terms);
    Ok(simplify(divide(total, big_h, pool), pool).value)
}

/// `∇×F` — the curl of a vector field given in physical components.
///
/// `(∇×F)ᵢ = (1/(h_j h_k)) [ ∂(h_k F_k)/∂u_j − ∂(h_j F_j)/∂u_k ]` for `(i,j,k)`
/// a cyclic permutation of `(1,2,3)`. Right-handed frame; see [`cross`].
pub fn curl(
    field: &[ExprId; 3],
    coords: &Coordinates,
    pool: &ExprPool,
) -> Result<[ExprId; 3], VectorError> {
    let vars = coords.vars();
    let h = coords.scale_factors();
    let neg = pool.integer(-1_i32);
    let mut out = [field[0]; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let hk_fk = times(h[k], field[k], pool);
        let hj_fj = times(h[j], field[j], pool);
        let a = d(hk_fk, vars[j], pool)?;
        let b = d(hj_fj, vars[k], pool)?;
        let bracket = pool.add(vec![a, pool.mul(vec![neg, b])]);
        let den = simplify(times(h[j], h[k], pool), pool).value;
        *slot = simplify(divide(bracket, den, pool), pool).value;
    }
    Ok(out)
}

/// `∇²f` — the scalar (Laplace–Beltrami) Laplacian.
///
/// `∇²f = (1/H) Σᵢ ∂/∂uᵢ ( (H/hᵢ²) ∂f/∂uᵢ )`.
pub fn laplacian(f: ExprId, coords: &Coordinates, pool: &ExprPool) -> Result<ExprId, VectorError> {
    let vars = coords.vars();
    let h = coords.scale_factors();
    let big_h = coords.jacobian_factor(pool);
    let mut terms = Vec::with_capacity(3);
    for i in 0..3 {
        let df = d(f, vars[i], pool)?;
        let hi_sq = times(h[i], h[i], pool);
        let weighted = times(weight(big_h, hi_sq, pool), df, pool);
        terms.push(d(weighted, vars[i], pool)?);
    }
    let total = pool.add(terms);
    Ok(simplify(divide(total, big_h, pool), pool).value)
}

/// `∇²F` — the **vector** Laplacian, `∇(∇·F) − ∇×(∇×F)`.
///
/// In Cartesian coordinates this is the componentwise scalar Laplacian and is
/// computed that way (cheaper, and it leaves
/// `∇×(∇×F) = ∇(∇·F) − ∇²F` a real check rather than a tautology). In every
/// other chart the componentwise form is **wrong** — the basis vectors turn
/// from point to point, contributing `−F_φ/ρ²`-style terms — so the defining
/// identity is used instead.
pub fn vector_laplacian(
    field: &[ExprId; 3],
    coords: &Coordinates,
    pool: &ExprPool,
) -> Result<[ExprId; 3], VectorError> {
    if coords.is_cartesian(pool) {
        let mut out = [field[0]; 3];
        for i in 0..3 {
            out[i] = laplacian(field[i], coords, pool)?;
        }
        return Ok(out);
    }
    let div = divergence(field, coords, pool)?;
    let grad_div = gradient(div, coords, pool)?;
    let cc = curl(&curl(field, coords, pool)?, coords, pool)?;
    Ok(sub(&grad_div, &cc, pool))
}
