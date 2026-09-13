//! Orthogonal coordinate systems, described by their scale factors.
//!
//! An orthogonal chart `u ↦ r(u)` is fully characterised, for the purposes of
//! `grad`/`div`/`curl`/`∇²`, by the three **scale factors** (Lamé
//! coefficients)
//!
//! ```text
//! hᵢ = |∂r/∂uᵢ|
//! ```
//!
//! together with the fact that the tangent vectors `∂r/∂uᵢ` are mutually
//! orthogonal. Nothing else about the embedding enters the formulas. So this
//! type stores exactly that: three coordinate symbols and three scale factors.

use super::VectorError;
use crate::diff::diff;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::zero_test::{zero_status, ZeroStatus};
use crate::simplify::engine::simplify;

/// An orthogonal coordinate system: three coordinate symbols and their scale
/// factors.
///
/// Build one with [`Coordinates::cartesian`], [`Coordinates::cylindrical`],
/// [`Coordinates::spherical`] or [`Coordinates::from_embedding`]. There is no
/// public field and no public constructor that takes scale factors directly:
/// the three built-in charts carry textbook values, and anything else has to
/// come from an embedding whose orthogonality was checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coordinates {
    vars: [ExprId; 3],
    scale: [ExprId; 3],
    label: String,
}

fn validate_vars(vars: &[ExprId; 3], pool: &ExprPool) -> Result<(), VectorError> {
    for (i, &v) in vars.iter().enumerate() {
        let is_symbol = pool.with(v, |d| matches!(d, ExprData::Symbol { .. }));
        if !is_symbol {
            return Err(VectorError::NotACoordinateSymbol {
                got: pool.display(v).to_string(),
            });
        }
        for &w in &vars[..i] {
            if v == w {
                return Err(VectorError::RepeatedCoordinate {
                    coordinate: pool.display(v).to_string(),
                });
            }
        }
    }
    Ok(())
}

impl Coordinates {
    /// Cartesian `(x, y, z)` — scale factors `(1, 1, 1)`.
    ///
    /// # Errors
    ///
    /// `E-VEC-002` / `E-VEC-003` if the three arguments are not distinct
    /// symbols.
    pub fn cartesian(
        x: ExprId,
        y: ExprId,
        z: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, VectorError> {
        let vars = [x, y, z];
        validate_vars(&vars, pool)?;
        let one = pool.integer(1_i32);
        Ok(Coordinates {
            vars,
            scale: [one; 3],
            label: "cartesian".to_string(),
        })
    }

    /// Cylindrical `(ρ, φ, z)` — scale factors `(1, ρ, 1)`.
    ///
    /// `x = ρ cos φ`, `y = ρ sin φ`, `z = z`. Right-handed in this order.
    ///
    /// # Errors
    ///
    /// `E-VEC-002` / `E-VEC-003` if the three arguments are not distinct
    /// symbols.
    pub fn cylindrical(
        rho: ExprId,
        phi: ExprId,
        z: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, VectorError> {
        let vars = [rho, phi, z];
        validate_vars(&vars, pool)?;
        let one = pool.integer(1_i32);
        Ok(Coordinates {
            vars,
            scale: [one, rho, one],
            label: "cylindrical".to_string(),
        })
    }

    /// Spherical `(r, θ, φ)` — scale factors `(1, r, r sin θ)`.
    ///
    /// The **physics / ISO 80000-2** convention: `θ` is the polar angle
    /// measured from the `+z` axis and `φ` the azimuth, so
    /// `x = r sin θ cos φ`, `y = r sin θ sin φ`, `z = r cos θ`. Right-handed in
    /// the order `(r, θ, φ)`.
    ///
    /// If you want the other convention swap the two angles *and* remember
    /// that `(r, φ, θ)` is then left-handed, which flips the sign of every
    /// `curl` component.
    ///
    /// # Errors
    ///
    /// `E-VEC-002` / `E-VEC-003` if the three arguments are not distinct
    /// symbols.
    pub fn spherical(
        r: ExprId,
        theta: ExprId,
        phi: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, VectorError> {
        let vars = [r, theta, phi];
        validate_vars(&vars, pool)?;
        let one = pool.integer(1_i32);
        let sin_theta = pool.func("sin", vec![theta]);
        let r_sin_theta = pool.mul(vec![r, sin_theta]);
        Ok(Coordinates {
            vars,
            scale: [one, r, r_sin_theta],
            label: "spherical".to_string(),
        })
    }

    /// Derive a chart from its Cartesian embedding `(x(u), y(u), z(u))`,
    /// **verifying orthogonality** before returning it.
    ///
    /// `hᵢ = |∂r/∂uᵢ|`, and the pair `(i, j)` is accepted only when
    /// `∂r/∂uᵢ · ∂r/∂u_j` is *proven* to vanish — an undecided inner product is
    /// a refusal (`E-VEC-004`), not an assumption. Every formula in this
    /// module is false for a skew chart, and false in a way that produces a
    /// perfectly ordinary-looking expression.
    ///
    /// A scale factor that is identically zero, or whose non-vanishing could
    /// not be established, is `E-VEC-005`: the operators all divide by it.
    ///
    /// # Errors
    ///
    /// `E-VEC-001` (a component of the embedding is not differentiable),
    /// `E-VEC-002`/`E-VEC-003` (bad coordinates), `E-VEC-004` (not
    /// orthogonal), `E-VEC-005` (degenerate scale factor).
    pub fn from_embedding(
        vars: [ExprId; 3],
        embedding: [ExprId; 3],
        label: impl Into<String>,
        pool: &ExprPool,
    ) -> Result<Self, VectorError> {
        validate_vars(&vars, pool)?;

        // Tangent vectors ∂r/∂uᵢ.
        let mut tangent = [[embedding[0]; 3]; 3];
        for (i, &u) in vars.iter().enumerate() {
            for (k, &component) in embedding.iter().enumerate() {
                let partial = diff(component, u, pool)
                    .map_err(|e| VectorError::Differentiation(e.to_string()))?
                    .value;
                tangent[i][k] = simplify(partial, pool).value;
            }
        }

        // Orthogonality: every off-diagonal inner product must be *proven* zero.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let inner = super::dot(&tangent[i], &tangent[j], pool);
                let status = zero_status(pool, inner);
                if status != ZeroStatus::Zero {
                    return Err(VectorError::NonOrthogonal {
                        pair: (i, j),
                        inner_product: pool.display(inner).to_string(),
                    });
                }
            }
        }

        // Scale factors, and the division-by-zero guard.
        let mut scale = [embedding[0]; 3];
        for i in 0..3 {
            let h_sq = super::norm_squared(&tangent[i], pool);
            let status = zero_status(pool, h_sq);
            if status != ZeroStatus::NonZero {
                let h = simplify(pool.func("sqrt", vec![h_sq]), pool).value;
                return Err(VectorError::DegenerateScaleFactor {
                    index: i,
                    scale_factor: pool.display(h).to_string(),
                    proven_zero: status == ZeroStatus::Zero,
                });
            }
            scale[i] = simplify(pool.func("sqrt", vec![h_sq]), pool).value;
        }

        Ok(Coordinates {
            vars,
            scale,
            label: label.into(),
        })
    }

    /// The three coordinate symbols, in the order the constructor took them.
    pub fn vars(&self) -> [ExprId; 3] {
        self.vars
    }

    /// The three scale factors `(h₁, h₂, h₃)`.
    pub fn scale_factors(&self) -> [ExprId; 3] {
        self.scale
    }

    /// A human-readable name for the chart, for diagnostics only.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// `H = h₁h₂h₃`, the volume element's Jacobian factor, simplified.
    pub fn jacobian_factor(&self, pool: &ExprPool) -> ExprId {
        simplify(pool.mul(self.scale.to_vec()), pool).value
    }

    /// True when every scale factor is the literal `1`.
    ///
    /// An orthogonal chart with `h = (1, 1, 1)` has the constant metric `δᵢⱼ`,
    /// so its frame is parallel-transported and the componentwise scalar
    /// Laplacian *is* the vector Laplacian. That is the property
    /// [`super::vector_laplacian`] branches on, so it is derived from the
    /// scale factors rather than from a flag set at construction time.
    pub fn is_cartesian(&self, pool: &ExprPool) -> bool {
        self.scale
            .iter()
            .all(|&h| pool.with(h, |d| matches!(d, ExprData::Integer(n) if n.0 == 1)))
    }
}
