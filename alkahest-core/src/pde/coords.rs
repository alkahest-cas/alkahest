//! The n-D coordinate layer (plan decision D7): orthogonal charts on `ℝⁿ`
//! for any concrete `n`, and Cartesian `ℝᵈ` for a **symbolic** dimension `d`,
//! with `gradient`, `divergence` and `laplacian` acting on expressions that
//! contain unknowns and their `Derivative` nodes.
//!
//! The 3-D curvilinear [`crate::vector::Coordinates`] (with `curl` and the
//! vector Laplacian) is untouched; this module sits beside it.
//!
//! # Concrete dimension: [`CoordinatesN`]
//!
//! An orthogonal chart `u ↦ r(u)` on `ℝⁿ` is described by its coordinate
//! symbols and scale factors `hᵢ = |∂r/∂uᵢ|`; with `J = Πhᵢ`,
//!
//! ```text
//! (∇f)ᵢ = (1/hᵢ) ∂f/∂uᵢ                       (physical components)
//! ∇·F   = (1/J) Σᵢ ∂(J Fᵢ / hᵢ)/∂uᵢ
//! Δf    = (1/J) Σᵢ ∂(J/hᵢ² · ∂f/∂uᵢ)/∂uᵢ
//! ```
//!
//! Built-in charts are [`CoordinatesN::cartesian`], [`CoordinatesN::polar`]
//! and [`CoordinatesN::hyperspherical`] (textbook scale factors); anything
//! else comes from [`CoordinatesN::from_embedding`], whose orthogonality and
//! non-degenerate scale factors are **proved**, or the call refuses
//! (`E-PDE-041`).
//!
//! # Symbolic dimension: [`SymbolicCartesian`]
//!
//! `ℝᵈ` with coordinates `xᵢ = Indexed(x, i)`, `1 ≤ i ≤ d`, through the
//! reserved `Sum` / `Indexed` / `KroneckerDelta` heads of [`super::sum`]:
//! `∂f/∂xⱼ` for a free index `j`, `∇·F = Σⱼ ∂Fⱼ/∂xⱼ` for a field given by its
//! `j`-th component, and `Δf` ([`super::laplacian_indexed`]), so for example
//! `Δ Σᵢ xᵢ² = 2d`.

use crate::diff::diff_impl::diff;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::matrix::zero_test::{zero_status, ZeroStatus};
use crate::simplify::engine::simplify;

use super::error::PdeError;
use super::sum::{indexed, laplacian_indexed, simplify_sums, SUM};

fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify(e, pool).value
}

fn derr(e: crate::diff::DiffError) -> PdeError {
    PdeError::CoordinateChart(format!("differentiation failed: {e}"))
}

fn d(e: ExprId, v: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    Ok(simp(diff(e, v, pool).map_err(derr)?.value, pool))
}

fn inv(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(e, pool.integer(-1))
}

fn validate_vars(vars: &[ExprId], pool: &ExprPool) -> Result<(), PdeError> {
    if vars.is_empty() {
        return Err(PdeError::CoordinateChart(
            "a chart needs at least one coordinate".into(),
        ));
    }
    for (i, &v) in vars.iter().enumerate() {
        if !pool.with(v, |d| matches!(d, ExprData::Symbol { .. })) {
            return Err(PdeError::CoordinateChart(format!(
                "coordinate {} is not a symbol",
                pool.display(v)
            )));
        }
        if vars[..i].contains(&v) {
            return Err(PdeError::CoordinateChart(format!(
                "coordinate {} is repeated",
                pool.display(v)
            )));
        }
    }
    Ok(())
}

/// An orthogonal coordinate chart on `ℝⁿ` (concrete `n`): coordinate symbols
/// and their scale factors.  See the [module documentation](self).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatesN {
    vars: Vec<ExprId>,
    scale: Vec<ExprId>,
    label: String,
}

impl CoordinatesN {
    /// Cartesian `(x₁, …, xₙ)`: every scale factor is `1`.
    ///
    /// # Errors
    ///
    /// `E-PDE-041` when the coordinates are not distinct symbols.
    pub fn cartesian(vars: &[ExprId], pool: &ExprPool) -> Result<Self, PdeError> {
        validate_vars(vars, pool)?;
        Ok(CoordinatesN {
            vars: vars.to_vec(),
            scale: vec![pool.integer(1); vars.len()],
            label: "cartesian".into(),
        })
    }

    /// Plane polar `(r, θ)`, `x = r cos θ`, `y = r sin θ`: scale factors
    /// `(1, r)`.
    ///
    /// # Errors
    ///
    /// As [`Self::cartesian`].
    pub fn polar(r: ExprId, theta: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        Self::hyperspherical(r, &[theta], pool).map(|mut c| {
            c.label = "polar".into();
            c
        })
    }

    /// Hyperspherical `(r, φ₁, …, φₙ₋₁)` on `ℝⁿ`:
    /// `x₁ = r cos φ₁`, `x₂ = r sin φ₁ cos φ₂`, …,
    /// `xₙ₋₁ = r sin φ₁⋯sin φₙ₋₂ cos φₙ₋₁`, `xₙ = r sin φ₁⋯sin φₙ₋₁`, with
    /// scale factors `h_r = 1`, `h_{φₖ} = r Π_{j<k} sin φⱼ`.  For `n = 3`
    /// this is the physics convention `(r, θ, φ)`; for `n = 2` it is
    /// [`Self::polar`].
    ///
    /// # Errors
    ///
    /// As [`Self::cartesian`]; `E-PDE-041` also when no angle is given.
    pub fn hyperspherical(r: ExprId, angles: &[ExprId], pool: &ExprPool) -> Result<Self, PdeError> {
        if angles.is_empty() {
            return Err(PdeError::CoordinateChart(
                "hyperspherical coordinates need at least one angle".into(),
            ));
        }
        let mut vars = vec![r];
        vars.extend_from_slice(angles);
        validate_vars(&vars, pool)?;
        let mut scale = vec![pool.integer(1)];
        let mut prod = vec![r];
        for &phi in angles {
            scale.push(simp(pool.mul(prod.clone()), pool));
            prod.push(pool.func("sin", vec![phi]));
        }
        Ok(CoordinatesN {
            vars,
            scale,
            label: "hyperspherical".into(),
        })
    }

    /// The embedding `r(u) = (r₁(u), …, rₙ(u))` of an orthogonal chart.  The
    /// tangent vectors' pairwise inner products must be **proved** zero and
    /// each scale factor `hᵢ = √(|∂r/∂uᵢ|²)` proved non-zero.
    ///
    /// # Errors
    ///
    /// `E-PDE-041`: bad coordinates, an embedding whose length is not the
    /// number of coordinates, a non-orthogonal chart, a degenerate scale
    /// factor, or a component that cannot be differentiated.
    pub fn from_embedding(
        vars: &[ExprId],
        embedding: &[ExprId],
        label: impl Into<String>,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        validate_vars(vars, pool)?;
        if embedding.len() != vars.len() {
            return Err(PdeError::CoordinateChart(format!(
                "the embedding has {} components for {} coordinates",
                embedding.len(),
                vars.len()
            )));
        }
        let n = vars.len();
        let mut tangent = vec![vec![pool.integer(0); n]; n];
        for (i, &u) in vars.iter().enumerate() {
            for (k, &comp) in embedding.iter().enumerate() {
                tangent[i][k] = d(comp, u, pool)?;
            }
        }
        let dot = |a: &[ExprId], b: &[ExprId]| -> ExprId {
            simp(
                pool.add(
                    a.iter()
                        .zip(b)
                        .map(|(&x, &y)| pool.mul(vec![x, y]))
                        .collect(),
                ),
                pool,
            )
        };
        for i in 0..n {
            for j in i + 1..n {
                let ip = dot(&tangent[i], &tangent[j]);
                if zero_status(pool, ip) != ZeroStatus::Zero {
                    return Err(PdeError::CoordinateChart(format!(
                        "the chart is not orthogonal: ∂r/∂{} · ∂r/∂{} = {} is not proved zero",
                        pool.display(vars[i]),
                        pool.display(vars[j]),
                        pool.display(ip)
                    )));
                }
            }
        }
        let mut scale = Vec::with_capacity(n);
        for i in 0..n {
            let h2 = dot(&tangent[i], &tangent[i]);
            if zero_status(pool, h2) != ZeroStatus::NonZero {
                return Err(PdeError::CoordinateChart(format!(
                    "the scale factor of {} (h² = {}) is not proved non-zero",
                    pool.display(vars[i]),
                    pool.display(h2)
                )));
            }
            scale.push(simp(pool.func("sqrt", vec![h2]), pool));
        }
        Ok(CoordinatesN {
            vars: vars.to_vec(),
            scale,
            label: label.into(),
        })
    }

    /// The dimension `n`.
    pub fn dim(&self) -> usize {
        self.vars.len()
    }

    /// The coordinate symbols.
    pub fn vars(&self) -> &[ExprId] {
        &self.vars
    }

    /// The scale factors `hᵢ`.
    pub fn scale_factors(&self) -> &[ExprId] {
        &self.scale
    }

    /// The chart's name (`"cartesian"`, `"polar"`, `"hyperspherical"`, or the
    /// label given to [`Self::from_embedding`]).
    pub fn label(&self) -> &str {
        &self.label
    }

    /// `J = Πhᵢ`, the volume element's factor.
    pub fn jacobian_factor(&self, pool: &ExprPool) -> ExprId {
        simp(pool.mul(self.scale.clone()), pool)
    }

    /// Physical components `(1/hᵢ) ∂f/∂uᵢ` of `∇f`.  `f` may contain unknowns
    /// `u(x₁, …)` and their derivatives (as `Derivative` nodes).
    ///
    /// # Errors
    ///
    /// `E-PDE-041` when `diff` refuses.
    pub fn gradient(&self, f: ExprId, pool: &ExprPool) -> Result<Vec<ExprId>, PdeError> {
        self.vars
            .iter()
            .zip(&self.scale)
            .map(|(&u, &h)| Ok(simp(pool.mul(vec![d(f, u, pool)?, inv(h, pool)]), pool)))
            .collect()
    }

    /// `∇·F` of the field with physical components `F`.
    ///
    /// # Errors
    ///
    /// `E-PDE-041` when `F` has the wrong length or `diff` refuses.
    pub fn divergence(&self, field: &[ExprId], pool: &ExprPool) -> Result<ExprId, PdeError> {
        if field.len() != self.dim() {
            return Err(PdeError::CoordinateChart(format!(
                "the field has {} components in dimension {}",
                field.len(),
                self.dim()
            )));
        }
        let j = self.jacobian_factor(pool);
        let mut terms = Vec::with_capacity(field.len());
        for ((&u, &h), &fi) in self.vars.iter().zip(&self.scale).zip(field) {
            let flux = simp(pool.mul(vec![j, fi, inv(h, pool)]), pool);
            terms.push(d(flux, u, pool)?);
        }
        Ok(simp(pool.mul(vec![inv(j, pool), pool.add(terms)]), pool))
    }

    /// `Δf = ∇·∇f`.
    ///
    /// # Errors
    ///
    /// `E-PDE-041` when `diff` refuses.
    pub fn laplacian(&self, f: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
        let j = self.jacobian_factor(pool);
        let mut terms = Vec::with_capacity(self.dim());
        for (&u, &h) in self.vars.iter().zip(&self.scale) {
            let flux = simp(
                pool.mul(vec![j, pool.pow(h, pool.integer(-2)), d(f, u, pool)?]),
                pool,
            );
            terms.push(d(flux, u, pool)?);
        }
        Ok(simp(pool.mul(vec![inv(j, pool), pool.add(terms)]), pool))
    }
}

/// Cartesian `ℝᵈ` for a symbolic (or literal) dimension `d`: coordinates
/// `xᵢ = Indexed(x, i)`, `1 ≤ i ≤ d`.  See the [module documentation](self).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicCartesian {
    base: ExprId,
    dim: ExprId,
}

impl SymbolicCartesian {
    /// `ℝᵈ` with coordinates `Indexed(base, i)`.
    ///
    /// # Errors
    ///
    /// `E-PDE-041` when `base` is not a symbol, or `dim` mentions it.
    pub fn new(base: ExprId, dim: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        if !pool.with(base, |d| matches!(d, ExprData::Symbol { .. })) {
            return Err(PdeError::CoordinateChart(format!(
                "the coordinate base {} is not a symbol",
                pool.display(base)
            )));
        }
        if crate::kernel::subs::mentions_var(dim, base, pool) {
            return Err(PdeError::CoordinateChart(
                "the dimension mentions the coordinate base".into(),
            ));
        }
        Ok(SymbolicCartesian { base, dim })
    }

    /// The base symbol `x`.
    pub fn base(&self) -> ExprId {
        self.base
    }

    /// The dimension `d`.
    pub fn dim(&self) -> ExprId {
        self.dim
    }

    /// The coordinate `xᵢ = Indexed(x, i)`.
    ///
    /// # Errors
    ///
    /// Never for a valid chart (kept fallible for the `Indexed` constructor).
    pub fn coordinate(&self, i: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
        indexed(self.base, i, pool)
    }

    /// `|x|² = Σᵢ xᵢ²` with a fresh summation index.
    ///
    /// # Errors
    ///
    /// As [`Self::coordinate`].
    pub fn radius_squared(&self, pool: &ExprPool) -> Result<ExprId, PdeError> {
        let i = super::sum::fresh_index(pool.symbol("i", Domain::Integer), &[self.dim], pool);
        let xi = self.coordinate(i, pool)?;
        super::sum::sum(
            pool.pow(xi, pool.integer(2)),
            i,
            pool.integer(1),
            self.dim,
            pool,
        )
    }

    /// The `j`-th gradient component `∂f/∂xⱼ` (`j` a symbol, assumed in
    /// `1 … d`), with `Σᵢ g(i) δᵢⱼ` collapsed under that range.
    ///
    /// # Errors
    ///
    /// `E-PDE-022` / `E-PDE-041` when the term-wise derivative is refused.
    pub fn gradient_component(
        &self,
        f: ExprId,
        j: ExprId,
        pool: &ExprPool,
    ) -> Result<ExprId, PdeError> {
        let xj = self.coordinate(j, pool)?;
        let df = diff(f, xj, pool)
            .map_err(|e| PdeError::SumDifferentiation(e.to_string()))?
            .value;
        Ok(simplify_sums(df, &[(j, pool.integer(1), self.dim)], pool))
    }

    /// `∇·F = Σⱼ ∂Fⱼ/∂xⱼ` for the field whose `j`-th component is
    /// `component` (an expression in the index symbol `j`).
    ///
    /// # Errors
    ///
    /// As [`Self::gradient_component`].
    pub fn divergence(
        &self,
        component: ExprId,
        j: ExprId,
        pool: &ExprPool,
    ) -> Result<ExprId, PdeError> {
        let one = pool.integer(1);
        let xj = self.coordinate(j, pool)?;
        let df = diff(component, xj, pool)
            .map_err(|e| PdeError::SumDifferentiation(e.to_string()))?
            .value;
        let df = simplify_sums(df, &[(j, one, self.dim)], pool);
        Ok(simplify_sums(
            pool.func(SUM, vec![df, j, one, self.dim]),
            &[],
            pool,
        ))
    }

    /// `Δf = Σⱼ ∂²f/∂xⱼ²` ([`super::laplacian_indexed`]).
    ///
    /// # Errors
    ///
    /// As [`super::laplacian_indexed`].
    pub fn laplacian(&self, f: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
        laplacian_indexed(f, self.base, pool.integer(1), self.dim, pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplify::zero::{zero_test, ZeroStatus as Z};

    fn zero(e: ExprId, p: &ExprPool) -> bool {
        let s = simp(e, p);
        s == p.integer(0) || zero_test(s, p, None).status == Z::ProvedZero
    }

    fn sub(a: ExprId, b: ExprId, p: &ExprPool) -> ExprId {
        p.add(vec![a, p.mul(vec![p.integer(-1), b])])
    }

    #[test]
    fn cartesian_laplacian_of_unknown_in_four_dimensions() {
        let p = ExprPool::new();
        let xs: Vec<ExprId> = (1..=4)
            .map(|i| p.symbol(format!("x{i}"), Domain::Real))
            .collect();
        let c = CoordinatesN::cartesian(&xs, &p).unwrap();
        let u = p.func("u", xs.clone());
        let lap = c.laplacian(u, &p).unwrap();
        let mut want = Vec::new();
        for &x in &xs {
            want.push(diff(diff(u, x, &p).unwrap().value, x, &p).unwrap().value);
        }
        assert!(zero(sub(lap, p.add(want), &p), &p), "{}", p.display(lap));
        // div grad = laplacian.
        let g = c.gradient(u, &p).unwrap();
        let dv = c.divergence(&g, &p).unwrap();
        assert!(zero(sub(dv, lap, &p), &p));
        assert!(c.divergence(&g[..3], &p).is_err());
    }

    #[test]
    fn polar_and_hyperspherical() {
        let p = ExprPool::new();
        let r = p.symbol("r", Domain::Positive);
        let th = p.symbol("theta", Domain::Real);
        let c = CoordinatesN::polar(r, th, &p).unwrap();
        // Δ r² = 4 in the plane; Δ log r = 0.
        let lap = c.laplacian(p.pow(r, p.integer(2)), &p).unwrap();
        assert!(zero(sub(lap, p.integer(4), &p), &p));
        let lap = c.laplacian(p.func("log", vec![r]), &p).unwrap();
        assert!(zero(lap, &p), "{}", p.display(lap));
        // ℝ⁴: Δ r^{-2} = 0 (the fundamental solution r^{2−n}).
        let a1 = p.symbol("phi1", Domain::Real);
        let a2 = p.symbol("phi2", Domain::Real);
        let a3 = p.symbol("phi3", Domain::Real);
        let c4 = CoordinatesN::hyperspherical(r, &[a1, a2, a3], &p).unwrap();
        assert_eq!(c4.dim(), 4);
        let lap = c4.laplacian(p.pow(r, p.integer(-2)), &p).unwrap();
        assert!(zero(lap, &p), "{}", p.display(lap));
        // Radial unknown f(r): f'' + 3 f'/r.
        let f = p.func("f", vec![r]);
        let lap = c4.laplacian(f, &p).unwrap();
        let f1 = diff(f, r, &p).unwrap().value;
        let f2 = diff(f1, r, &p).unwrap().value;
        let want = p.add(vec![
            f2,
            p.mul(vec![p.integer(3), f1, p.pow(r, p.integer(-1))]),
        ]);
        assert!(zero(sub(lap, want, &p), &p), "{}", p.display(lap));
    }

    #[test]
    fn embedding_is_checked() {
        let p = ExprPool::new();
        let r = p.symbol("r", Domain::Positive);
        let th = p.symbol("theta", Domain::Real);
        let emb = [
            p.mul(vec![r, p.func("cos", vec![th])]),
            p.mul(vec![r, p.func("sin", vec![th])]),
        ];
        let c = CoordinatesN::from_embedding(&[r, th], &emb, "polar", &p).unwrap();
        let lap = c.laplacian(p.pow(r, p.integer(2)), &p).unwrap();
        assert!(zero(sub(lap, p.integer(4), &p), &p));
        // Skew chart: x = u + v, y = v.
        let u = p.symbol("u", Domain::Real);
        let v = p.symbol("v", Domain::Real);
        let e = CoordinatesN::from_embedding(&[u, v], &[p.add(vec![u, v]), v], "skew", &p);
        assert!(matches!(e, Err(PdeError::CoordinateChart(_))));
        assert!(CoordinatesN::cartesian(&[u, u], &p).is_err());
    }

    #[test]
    fn symbolic_dimension() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let dd = p.symbol("d", Domain::Positive);
        let c = SymbolicCartesian::new(x, dd, &p).unwrap();
        let r2 = c.radius_squared(&p).unwrap();
        // Δ |x|² = 2d.
        let lap = c.laplacian(r2, &p).unwrap();
        assert!(
            zero(sub(lap, p.mul(vec![p.integer(2), dd]), &p), &p),
            "{}",
            p.display(lap)
        );
        // ∂|x|²/∂xⱼ = 2xⱼ.
        let j = p.symbol("j", Domain::Integer);
        let g = c.gradient_component(r2, j, &p).unwrap();
        let xj = c.coordinate(j, &p).unwrap();
        assert!(
            zero(sub(g, p.mul(vec![p.integer(2), xj]), &p), &p),
            "{}",
            p.display(g)
        );
        // ∇·x = d.
        let dv = c.divergence(xj, j, &p).unwrap();
        assert!(zero(sub(dv, dd, &p), &p), "{}", p.display(dv));
    }
}
