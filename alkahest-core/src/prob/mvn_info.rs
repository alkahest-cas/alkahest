//! Information measures of the multivariate normal: differential entropy,
//! `KL(N₁ ‖ N₂)`, and the mutual information between two groups of components.
//!
//! These are exact algebra, not quadrature: a `d`-dimensional integral has no
//! place in this module's one-variable numeric gate. The formulas are the
//! textbook identities and the tests check them against `d = 1` (the
//! univariate, quadrature-gated `entropy` and `kl_divergence`), against
//! `scipy`/`numpy`, and against numerical integration in two dimensions.
//!
//! ```text
//! h(N(μ, Σ))           = ½ log((2πe)^d det Σ)
//! KL(N(μ₀,Σ₀)‖N(μ₁,Σ₁)) = ½ [ tr(Σ₁⁻¹Σ₀) + (μ₁-μ₀)ᵀΣ₁⁻¹(μ₁-μ₀) - d + log(det Σ₁/det Σ₀) ]
//! I(X_A; X_B)          = ½ log( det Σ_AA · det Σ_BB / det Σ_{A∪B} )
//!                      = -½ log det(I - Σ_AA⁻¹ Σ_AB Σ_BB⁻¹ Σ_BA)
//! ```
//!
//! The mutual information is written through the *canonical-correlation*
//! form, so a bivariate normal with correlation `ρ` comes out as the familiar
//! `-½ log(1 - ρ²)` rather than as a ratio of determinants that the simplifier
//! would have to cancel.

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprId, ExprPool};
use crate::matrix::Matrix;
use crate::simplify::{simplify, simplify_expanded};

use super::entropy::{base_factor, rescale};
use super::joint::MultivariateNormal;
use super::{verify, ProbError};

fn log(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("log", vec![arg])
}

/// The positive-definiteness constraints the caller has not discharged.
fn undecided(mvn: &MultivariateNormal, pool: &ExprPool) -> Vec<crate::deriv::SideCondition> {
    mvn.constraints(pool)
        .into_iter()
        .filter_map(|c| verify::undischarged(c, pool))
        .collect()
}

fn matrix_of(rows: Vec<Vec<ExprId>>) -> Matrix {
    Matrix::new(rows).expect("rectangular by construction")
}

fn det(m: &Matrix, pool: &ExprPool) -> Result<ExprId, ProbError> {
    m.det(pool)
        .map(|d| simplify_expanded(d, pool).value)
        .map_err(|e| ProbError::Unsupported(format!("determinant of a covariance block: {e}")))
}

fn inverse(m: &Matrix, pool: &ExprPool) -> Result<Matrix, ProbError> {
    m.inverse(pool).map_err(|e| {
        ProbError::Unsupported(format!("a covariance block could not be inverted: {e}"))
    })
}

fn mul(a: &Matrix, b: &Matrix, pool: &ExprPool) -> Result<Matrix, ProbError> {
    a.mul(b, pool)
        .map_err(|e| ProbError::Unsupported(format!("covariance block product: {e}")))
}

fn finish(
    rule: &'static str,
    value: ExprId,
    conds: Vec<crate::deriv::SideCondition>,
) -> DerivedExpr<ExprId> {
    let mut unique: Vec<crate::deriv::SideCondition> = Vec::new();
    for c in conds {
        if !unique.contains(&c) {
            unique.push(c);
        }
    }
    let mut log_ = DerivationLog::new();
    log_.push(RewriteStep::with_conditions(rule, value, value, unique));
    super::stash_prob_side_conditions(super::conditions_of(&log_));
    DerivedExpr::with_log(value, log_)
}

impl MultivariateNormal {
    /// The differential entropy `h = ½ log((2πe)^d det Σ)` in nats (or in the
    /// unit of `base`).
    ///
    /// Differential entropy, not Shannon entropy: it can be negative and it
    /// is not invariant under a change of variables (see the module notes on
    /// `entropy`). `h` is independent of the mean.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] for a numeric `base` that is `<= 0` or
    /// `1`; [`ProbError::Unsupported`] when `det Σ` cannot be formed.
    pub fn entropy(
        &self,
        base: Option<ExprId>,
        pool: &ExprPool,
    ) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        let (factor, base_conds) = base_factor(base, pool)?;
        let d = self.dim() as i64;
        let half_d = pool.rational(d, 2);
        let two_pi = pool.mul(vec![pool.integer(2), super::pi(pool)]);
        let d_det = det(&matrix_of(self.covariance_matrix().to_vec()), pool)?;
        // d/2 + (d/2) log(2π) + ½ log det Σ
        let raw = pool.add(vec![
            half_d,
            pool.mul(vec![half_d, log(two_pi, pool)]),
            pool.mul(vec![pool.rational(1, 2), log(d_det, pool)]),
        ]);
        let value = simplify(rescale(raw, factor, pool), pool).value;
        let mut conds = undecided(self, pool);
        conds.extend(base_conds);
        Ok(finish("prob_mvn_differential_entropy", value, conds))
    }

    /// `KL(self ‖ other) = ½[tr(Σ₁⁻¹Σ₀) + (μ₁-μ₀)ᵀΣ₁⁻¹(μ₁-μ₀) - d + log(det Σ₁/det Σ₀)]`.
    ///
    /// Two non-degenerate normals on the same space always have a finite
    /// divergence (their supports are both `ℝᵈ`). Not symmetric.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] when the dimensions differ or a covariance
    /// cannot be inverted symbolically; [`ProbError::InvalidParameter`] for a
    /// bad `base`.
    pub fn kl_divergence(
        &self,
        other: &MultivariateNormal,
        base: Option<ExprId>,
        pool: &ExprPool,
    ) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        if self.dim() != other.dim() {
            return Err(ProbError::Unsupported(format!(
                "kl_divergence: the laws have dimensions {} and {}",
                self.dim(),
                other.dim()
            )));
        }
        let (factor, base_conds) = base_factor(base, pool)?;
        let d = self.dim();
        let s0 = matrix_of(self.covariance_matrix().to_vec());
        let s1 = matrix_of(other.covariance_matrix().to_vec());
        let inv1 = inverse(&s1, pool)?;
        let prod = mul(&inv1, &s0, pool)?;
        let mut trace = Vec::with_capacity(d);
        for i in 0..d {
            trace.push(prod.get(i, i));
        }
        let diff: Vec<ExprId> = (0..d)
            .map(|i| {
                pool.add(vec![
                    other.mean_vector()[i],
                    pool.mul(vec![pool.integer(-1), self.mean_vector()[i]]),
                ])
            })
            .collect();
        let mut quad = Vec::with_capacity(d * d);
        for i in 0..d {
            for j in 0..d {
                quad.push(pool.mul(vec![diff[i], inv1.get(i, j), diff[j]]));
            }
        }
        let log_ratio = log(
            pool.mul(vec![
                det(&s1, pool)?,
                pool.pow(det(&s0, pool)?, pool.integer(-1)),
            ]),
            pool,
        );
        let raw = pool.mul(vec![
            pool.rational(1, 2),
            pool.add(vec![
                pool.add(trace),
                pool.add(quad),
                pool.integer(-(d as i64)),
                log_ratio,
            ]),
        ]);
        let value = simplify(rescale(raw, factor, pool), pool).value;
        let mut conds = undecided(self, pool);
        conds.extend(undecided(other, pool));
        conds.extend(base_conds);
        Ok(finish("prob_mvn_kl_divergence", value, conds))
    }

    /// The mutual information `I(X_A; X_B)` between the components `a` and
    /// the components `b` (disjoint, 0-based):
    /// `-½ log det(I - Σ_AA⁻¹ Σ_AB Σ_BB⁻¹ Σ_BA)`. For two scalar components
    /// with correlation `ρ` this is `-½ log(1 - ρ²)`.
    ///
    /// It is `0` exactly when `Σ_AB = 0`, and it is **not** the mutual
    /// information of the marginals alone: it depends on the covariance
    /// between the groups, which is what a joint law supplies.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] for an empty, repeated, overlapping or
    /// out-of-range index list, or a block that cannot be inverted
    /// symbolically; [`ProbError::InvalidParameter`] for a bad `base`.
    pub fn mutual_information(
        &self,
        a: &[usize],
        b: &[usize],
        base: Option<ExprId>,
        pool: &ExprPool,
    ) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        // Validates emptiness, range and repeats within each list.
        self.marginal(a)?;
        self.marginal(b)?;
        if a.iter().any(|i| b.contains(i)) {
            return Err(ProbError::Unsupported(
                "mutual_information: the two index groups overlap".to_string(),
            ));
        }
        let (factor, base_conds) = base_factor(base, pool)?;
        let cov = self.covariance_matrix();
        let block = |rows: &[usize], cols: &[usize]| {
            matrix_of(
                rows.iter()
                    .map(|&r| cols.iter().map(|&c| cov[r][c]).collect())
                    .collect(),
            )
        };
        let (saa, sbb, sab) = (block(a, a), block(b, b), block(a, b));
        let sba = sab.transpose();
        // C = Σ_AA⁻¹ Σ_AB Σ_BB⁻¹ Σ_BA
        let c = mul(
            &mul(
                &mul(&inverse(&saa, pool)?, &sab, pool)?,
                &inverse(&sbb, pool)?,
                pool,
            )?,
            &sba,
            pool,
        )?;
        let k = a.len();
        let mut rows = Vec::with_capacity(k);
        for i in 0..k {
            let mut row = Vec::with_capacity(k);
            for j in 0..k {
                let diag = if i == j {
                    pool.integer(1)
                } else {
                    pool.integer(0)
                };
                row.push(
                    simplify_expanded(
                        pool.add(vec![diag, pool.mul(vec![pool.integer(-1), c.get(i, j)])]),
                        pool,
                    )
                    .value,
                );
            }
            rows.push(row);
        }
        let gap = simplify(det(&matrix_of(rows), pool)?, pool).value;
        let raw = pool.mul(vec![pool.rational(-1, 2), log(gap, pool)]);
        let value = simplify(rescale(raw, factor, pool), pool).value;
        let mut conds = undecided(self, pool);
        conds.extend(base_conds);
        Ok(finish("prob_mvn_mutual_information", value, conds))
    }
}
