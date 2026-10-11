//! Joint laws: the multivariate normal, independent products of blocks,
//! covariance, and sums of independent variates in closed families.
//!
//! # What is modelled
//!
//! A joint law here is a list of **mutually independent blocks**
//! ([`JointBlock`]). Each block is either one variate with a univariate
//! [`Distribution`], or a vector of variates with a [`MultivariateNormal`]
//! law. Dependence lives *inside* a Gaussian block (through its covariance
//! matrix) and nowhere else. That is the whole dependence structure, and it is
//! stated by the caller's choice of blocks rather than guessed: a list of
//! marginals says nothing about their joint law, which is the reason the
//! univariate module refuses `E[XY]`.
//!
//! # How an expectation is computed
//!
//! The expression is expanded into a sum of products. Each product is split
//! into one factor per block (a factor touching two independent blocks is
//! split further only when it is an `exp` of a sum, which factors exactly;
//! anything else is refused), and independence gives
//! `E[∏_b g_b(X_b)] = ∏_b E[g_b(X_b)]`.
//!
//! * A univariate block's factor goes through [`super::expectation`], so it
//!   is checked against quadrature of its own defining integral like every
//!   other single-variable expectation.
//! * A Gaussian block's factor must be a monomial times `exp` of an affine
//!   form. The monomial moment is computed exactly by **Stein's identity**
//!   `E[X_i g(X)] = μ_i E[g(X)] + Σ_j Σ_ij E[∂_j g(X)]`, applied recursively;
//!   an `exp(tᵀX)` factor is absorbed by exponential tilting,
//!   `E[m(X) e^{tᵀX}] = M(t) · E_{μ+Σt, Σ}[m(X)]`. Both are identities, not
//!   approximations, and the result is a polynomial (times `M(t)`) in the
//!   entries of `μ` and `Σ`.
//!
//! The Gaussian route is **not** quadrature-gated: the numeric gate in this
//! module integrates in one variable. The identities above are exact
//! algebra, and the tests check them against Monte-Carlo and against the
//! closed forms numpy and SymPy compute.
//!
//! # Positive definiteness
//!
//! A covariance matrix is checked by Sylvester's criterion — every leading
//! principal minor positive. A minor that evaluates to a number `≤ 0` is
//! refused at construction (`E-PROB-001`); a symbolic minor cannot be decided
//! and is carried as a side condition on every derived quantity, exactly like
//! a symbolic `σ` on a univariate `Normal`.

use std::collections::HashMap;

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::Matrix;
use crate::simplify::{simplify, simplify_expanded};

use super::dists::{self, Sign};
use super::{sub, DistKind, Distribution, ProbError};

/// Largest dimension a [`MultivariateNormal`] may have. The determinant and
/// inverse of a symbolic covariance are expanded symbolically, and their size
/// grows factorially; past this the expressions stop being useful answers.
pub const MAX_MVN_DIM: usize = 8;

/// Most terms an expanded joint expectation may have.
const MAX_JOINT_TERMS: usize = 512;

/// Highest total degree of a Gaussian monomial moment. The same ceiling as the
/// univariate moment table, for the same reason: past it the answer is a
/// polynomial too large to be read.
const MAX_GAUSSIAN_DEGREE: u32 = super::MAX_MOMENT_ORDER;

fn exp(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("exp", vec![arg])
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0)
}

fn sum_or_zero(terms: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    if terms.is_empty() {
        pool.integer(0)
    } else {
        pool.add(terms)
    }
}

/// A polynomial result in its expanded, collected form.
fn tidy(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify_expanded(e, pool).value
}

// ---------------------------------------------------------------------------
// The multivariate normal
// ---------------------------------------------------------------------------

/// `N(μ, Σ)` on `ℝᵈ`, with symbolic mean vector and covariance matrix.
///
/// Constructed through [`MultivariateNormal::new`], which checks that `Σ` is
/// square, matches `μ`, is symmetric, and is not *decidably* indefinite.
/// Fields are private for the same reason as [`Distribution`]'s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultivariateNormal {
    mean: Vec<ExprId>,
    cov: Vec<Vec<ExprId>>,
}

impl MultivariateNormal {
    /// `N(μ, Σ)`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when `μ` is empty or longer than
    /// [`MAX_MVN_DIM`], when `Σ` is not `d × d`, when `Σ` is not symmetric
    /// (entries `(i, j)` and `(j, i)` must be the same expression), or when a
    /// leading principal minor of `Σ` evaluates to a number `≤ 0`. A symbolic
    /// minor is accepted and carried on [`MultivariateNormal::constraints`].
    pub fn new(
        mean: Vec<ExprId>,
        cov: Vec<Vec<ExprId>>,
        pool: &ExprPool,
    ) -> Result<Self, ProbError> {
        let d = mean.len();
        if d == 0 || d > MAX_MVN_DIM {
            return Err(ProbError::InvalidParameter {
                parameter: "mean",
                requirement: "a vector of 1 to 8 components",
            });
        }
        if cov.len() != d || cov.iter().any(|r| r.len() != d) {
            return Err(ProbError::InvalidParameter {
                parameter: "cov",
                requirement: "a d × d matrix, d = len(mean)",
            });
        }
        for i in 0..d {
            for j in (i + 1)..d {
                let diff = simplify(sub(cov[i][j], cov[j][i], pool), pool).value;
                if !is_zero(diff, pool) {
                    return Err(ProbError::InvalidParameter {
                        parameter: "cov",
                        requirement: "symmetric: entries (i, j) and (j, i) must be equal",
                    });
                }
            }
        }
        let out = MultivariateNormal { mean, cov };
        for minor in out.leading_minors(pool) {
            if matches!(dists::sign_of(minor, pool), Sign::Negative | Sign::Zero) {
                return Err(ProbError::InvalidParameter {
                    parameter: "cov",
                    requirement: "positive definite (every leading principal minor > 0)",
                });
            }
        }
        Ok(out)
    }

    /// The dimension `d`.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// The mean vector `μ`.
    pub fn mean_vector(&self) -> &[ExprId] {
        &self.mean
    }

    /// The covariance matrix `Σ`, row by row.
    pub fn covariance_matrix(&self) -> &[Vec<ExprId>] {
        &self.cov
    }

    /// `Σ`'s leading principal minors, simplified.
    fn leading_minors(&self, pool: &ExprPool) -> Vec<ExprId> {
        (1..=self.dim())
            .map(|k| {
                let rows: Vec<Vec<ExprId>> =
                    self.cov[..k].iter().map(|r| r[..k].to_vec()).collect();
                let m = Matrix::new(rows).expect("square by construction");
                let det = m.det(pool).expect("square by construction");
                tidy(det, pool)
            })
            .collect()
    }

    /// The positive-definiteness constraints `Δ_k > 0` on the leading
    /// principal minors, as predicates. Numeric ones were checked at
    /// construction; symbolic ones are the caller's to discharge.
    pub fn constraints(&self, pool: &ExprPool) -> Vec<ExprId> {
        let zero = pool.integer(0);
        self.leading_minors(pool)
            .into_iter()
            .map(|m| pool.pred_gt(m, zero))
            .collect()
    }

    /// The minors nobody has decided, as side conditions.
    fn undecided(&self, pool: &ExprPool) -> Vec<SideCondition> {
        self.leading_minors(pool)
            .into_iter()
            .filter(|&m| matches!(dists::sign_of(m, pool), Sign::Unknown))
            .map(SideCondition::Positive)
            .collect()
    }

    fn finish(
        &self,
        rule: &'static str,
        before: ExprId,
        value: ExprId,
        pool: &ExprPool,
    ) -> DerivedExpr<ExprId> {
        let mut log = DerivationLog::new();
        log.push(RewriteStep::with_conditions(
            rule,
            before,
            value,
            self.undecided(pool),
        ));
        super::stash_prob_side_conditions(super::conditions_of(&log));
        DerivedExpr::with_log(value, log)
    }

    fn matrix(&self) -> Matrix {
        Matrix::new(self.cov.clone()).expect("square by construction")
    }

    /// `tᵀΣt`.
    fn quad(&self, t: &[ExprId], pool: &ExprPool) -> ExprId {
        let mut terms = Vec::new();
        for i in 0..self.dim() {
            for j in 0..self.dim() {
                terms.push(pool.mul(vec![t[i], self.cov[i][j], t[j]]));
            }
        }
        sum_or_zero(terms, pool)
    }

    /// `μᵀt`.
    fn dot_mean(&self, t: &[ExprId], pool: &ExprPool) -> ExprId {
        sum_or_zero(
            self.mean
                .iter()
                .zip(t)
                .map(|(&m, &ti)| pool.mul(vec![m, ti]))
                .collect(),
            pool,
        )
    }

    fn check_len(&self, v: &[ExprId], what: &str) -> Result<(), ProbError> {
        if v.len() != self.dim() {
            return Err(ProbError::Unsupported(format!(
                "{what} has {} components; this multivariate normal has dimension {}",
                v.len(),
                self.dim()
            )));
        }
        Ok(())
    }

    /// The density `(2π)^{-d/2} det(Σ)^{-1/2} exp(-½(x-μ)ᵀΣ⁻¹(x-μ))`.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] when `xs` has the wrong length or `Σ` could
    /// not be inverted symbolically (its determinant could not be shown
    /// non-zero).
    pub fn pdf(&self, xs: &[ExprId], pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        self.check_len(xs, "the point")?;
        let m = self.matrix();
        let inv = m.inverse(pool).map_err(|e| {
            ProbError::Unsupported(format!("the covariance matrix could not be inverted: {e}"))
        })?;
        let det = simplify(m.det(pool).expect("square"), pool).value;
        let dev: Vec<ExprId> = xs
            .iter()
            .zip(&self.mean)
            .map(|(&x, &mu)| sub(x, mu, pool))
            .collect();
        let mut q = Vec::new();
        for i in 0..self.dim() {
            for j in 0..self.dim() {
                q.push(pool.mul(vec![dev[i], inv.get(i, j), dev[j]]));
            }
        }
        let q = simplify(sum_or_zero(q, pool), pool).value;
        let d = self.dim() as i64;
        let two_pi = pool.mul(vec![pool.integer(2), super::pi(pool)]);
        let norm = pool.mul(vec![
            pool.pow(two_pi, pool.rational(-d, 2)),
            pool.pow(det, pool.rational(-1, 2)),
        ]);
        let value = simplify(
            pool.mul(vec![
                norm,
                exp(pool.mul(vec![pool.rational(-1, 2), q]), pool),
            ]),
            pool,
        )
        .value;
        Ok(self.finish("prob_mvn_density", det, value, pool))
    }

    /// `M(t) = E[e^{tᵀX}] = exp(μᵀt + ½tᵀΣt)`, entire in `t`.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] when `ts` has the wrong length.
    pub fn moment_generating_function(
        &self,
        ts: &[ExprId],
        pool: &ExprPool,
    ) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        self.check_len(ts, "the argument")?;
        let arg = pool.add(vec![
            self.dot_mean(ts, pool),
            pool.mul(vec![pool.rational(1, 2), self.quad(ts, pool)]),
        ]);
        let value = exp(tidy(arg, pool), pool);
        Ok(self.finish("prob_mvn_moment_generating_function", arg, value, pool))
    }

    /// `φ(t) = E[e^{i tᵀX}] = exp(i μᵀt − ½tᵀΣt)`.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] when `ts` has the wrong length.
    pub fn characteristic_function(
        &self,
        ts: &[ExprId],
        pool: &ExprPool,
    ) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        self.check_len(ts, "the argument")?;
        let arg = pool.add(vec![
            pool.mul(vec![
                pool.imaginary_unit(),
                tidy(self.dot_mean(ts, pool), pool),
            ]),
            pool.mul(vec![pool.rational(-1, 2), tidy(self.quad(ts, pool), pool)]),
        ]);
        let value = exp(arg, pool);
        Ok(self.finish("prob_mvn_characteristic_function", arg, value, pool))
    }

    /// The law of `AX + b`: `N(Aμ + b, AΣAᵀ)`.
    ///
    /// `A` is `m × d`. The image is a *non-degenerate* normal only when `A`
    /// has full row rank, so the result goes back through
    /// [`MultivariateNormal::new`]: a decidably singular `AΣAᵀ` is refused
    /// rather than returned as a law with no density.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] on a dimension mismatch;
    /// [`ProbError::InvalidParameter`] when `AΣAᵀ` is decidably not positive
    /// definite.
    pub fn linear_transform(
        &self,
        a: &[Vec<ExprId>],
        b: Option<&[ExprId]>,
        pool: &ExprPool,
    ) -> Result<MultivariateNormal, ProbError> {
        let d = self.dim();
        if a.is_empty() || a.iter().any(|r| r.len() != d) {
            return Err(ProbError::Unsupported(format!(
                "A must be an m × {d} matrix to act on this multivariate normal"
            )));
        }
        let m = a.len();
        if let Some(b) = b {
            if b.len() != m {
                return Err(ProbError::Unsupported(format!(
                    "b has {} components; A has {m} rows",
                    b.len()
                )));
            }
        }
        let mut mean = Vec::with_capacity(m);
        for (r, row) in a.iter().enumerate() {
            let mut terms: Vec<ExprId> = row
                .iter()
                .zip(&self.mean)
                .map(|(&aij, &mu)| pool.mul(vec![aij, mu]))
                .collect();
            if let Some(b) = b {
                terms.push(b[r]);
            }
            mean.push(tidy(pool.add(terms), pool));
        }
        let mut cov = vec![vec![pool.integer(0); m]; m];
        for r in 0..m {
            for c in r..m {
                let mut terms = Vec::new();
                for i in 0..d {
                    for j in 0..d {
                        terms.push(pool.mul(vec![a[r][i], self.cov[i][j], a[c][j]]));
                    }
                }
                let v = tidy(sum_or_zero(terms, pool), pool);
                cov[r][c] = v;
                cov[c][r] = v;
            }
        }
        MultivariateNormal::new(mean, cov, pool)
    }

    /// The marginal law of the components `idx`, in the order given.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] for an empty, repeated or out-of-range index
    /// list.
    pub fn marginal(&self, idx: &[usize]) -> Result<MultivariateNormal, ProbError> {
        self.check_indices(idx, "marginal")?;
        Ok(MultivariateNormal {
            mean: idx.iter().map(|&i| self.mean[i]).collect(),
            cov: idx
                .iter()
                .map(|&i| idx.iter().map(|&j| self.cov[i][j]).collect())
                .collect(),
        })
    }

    fn check_indices(&self, idx: &[usize], what: &str) -> Result<(), ProbError> {
        if idx.is_empty() {
            return Err(ProbError::Unsupported(format!(
                "{what}: the index list is empty"
            )));
        }
        for (k, &i) in idx.iter().enumerate() {
            if i >= self.dim() {
                return Err(ProbError::Unsupported(format!(
                    "{what}: index {i} is out of range for dimension {}",
                    self.dim()
                )));
            }
            if idx[..k].contains(&i) {
                return Err(ProbError::Unsupported(format!(
                    "{what}: index {i} is repeated"
                )));
            }
        }
        Ok(())
    }

    /// The conditional law of the remaining components (in increasing index
    /// order) given `X[given] = values`:
    /// `N(μ₁ + Σ₁₂Σ₂₂⁻¹(a − μ₂), Σ₁₁ − Σ₁₂Σ₂₂⁻¹Σ₂₁)` — the Schur complement.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] for a bad index list, a `values` of the
    /// wrong length, conditioning on every component (there is nothing left
    /// to have a law), or a `Σ₂₂` that could not be inverted symbolically.
    pub fn conditional(
        &self,
        given: &[usize],
        values: &[ExprId],
        pool: &ExprPool,
    ) -> Result<MultivariateNormal, ProbError> {
        self.check_indices(given, "conditional")?;
        if values.len() != given.len() {
            return Err(ProbError::Unsupported(format!(
                "conditional: {} indices but {} values",
                given.len(),
                values.len()
            )));
        }
        let rest: Vec<usize> = (0..self.dim()).filter(|i| !given.contains(i)).collect();
        if rest.is_empty() {
            return Err(ProbError::Unsupported(
                "conditional: every component is conditioned on, so nothing is left to have a \
                 law"
                .to_string(),
            ));
        }
        let s22 = Matrix::new(
            given
                .iter()
                .map(|&i| given.iter().map(|&j| self.cov[i][j]).collect())
                .collect(),
        )
        .expect("square");
        let inv = s22.inverse(pool).map_err(|e| {
            ProbError::Unsupported(format!(
                "conditional: the covariance of the conditioning block could not be inverted: {e}"
            ))
        })?;
        let g = given.len();
        // W = Σ₁₂ Σ₂₂⁻¹, one row per remaining component.
        let w: Vec<Vec<ExprId>> = rest
            .iter()
            .map(|&r| {
                (0..g)
                    .map(|c| {
                        let terms: Vec<ExprId> = (0..g)
                            .map(|k| pool.mul(vec![self.cov[r][given[k]], inv.get(k, c)]))
                            .collect();
                        simplify(sum_or_zero(terms, pool), pool).value
                    })
                    .collect()
            })
            .collect();
        let mean: Vec<ExprId> = rest
            .iter()
            .enumerate()
            .map(|(ri, &r)| {
                let mut terms = vec![self.mean[r]];
                for c in 0..g {
                    terms.push(pool.mul(vec![w[ri][c], sub(values[c], self.mean[given[c]], pool)]));
                }
                simplify(pool.add(terms), pool).value
            })
            .collect();
        let n = rest.len();
        let mut cov = vec![vec![pool.integer(0); n]; n];
        for (ri, &r) in rest.iter().enumerate() {
            for (ci, &c) in rest.iter().enumerate().skip(ri) {
                let mut terms = vec![self.cov[r][c]];
                for k in 0..g {
                    terms.push(pool.mul(vec![pool.integer(-1), w[ri][k], self.cov[given[k]][c]]));
                }
                let v = simplify(pool.add(terms), pool).value;
                cov[ri][ci] = v;
                cov[ci][ri] = v;
            }
        }
        MultivariateNormal::new(mean, cov, pool)
    }

    /// `E[XᵀAX] = tr(AΣ) + μᵀAμ`, for any square `A` (not necessarily
    /// symmetric).
    ///
    /// # Errors
    ///
    /// [`ProbError::Unsupported`] when `A` is not `d × d`.
    pub fn quadratic_form_expectation(
        &self,
        a: &[Vec<ExprId>],
        pool: &ExprPool,
    ) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        let d = self.dim();
        if a.len() != d || a.iter().any(|r| r.len() != d) {
            return Err(ProbError::Unsupported(format!(
                "A must be {d} × {d} for a quadratic form in this multivariate normal"
            )));
        }
        let mut terms = Vec::new();
        for i in 0..d {
            for j in 0..d {
                // tr(AΣ) = Σ_ij A_ij Σ_ji, μᵀAμ = Σ_ij μ_i A_ij μ_j
                terms.push(pool.mul(vec![a[i][j], self.cov[j][i]]));
                terms.push(pool.mul(vec![self.mean[i], a[i][j], self.mean[j]]));
            }
        }
        let value = tidy(sum_or_zero(terms, pool), pool);
        Ok(self.finish("prob_mvn_quadratic_form_trace_identity", value, value, pool))
    }

    /// `E[∏ X_i^{α_i}]` by Stein's identity, memoised.
    fn monomial_moment(
        alpha: &[u32],
        mean: &[ExprId],
        cov: &[Vec<ExprId>],
        memo: &mut HashMap<Vec<u32>, ExprId>,
        pool: &ExprPool,
    ) -> ExprId {
        if let Some(&v) = memo.get(alpha) {
            return v;
        }
        let Some(i) = alpha.iter().position(|&a| a > 0) else {
            return pool.integer(1);
        };
        // E[X_i · X^β] = μ_i E[X^β] + Σ_j Σ_ij β_j E[X^{β - e_j}]
        let mut beta = alpha.to_vec();
        beta[i] -= 1;
        let mut terms = vec![pool.mul(vec![
            mean[i],
            Self::monomial_moment(&beta, mean, cov, memo, pool),
        ])];
        for j in 0..alpha.len() {
            if beta[j] == 0 {
                continue;
            }
            let mut gamma = beta.clone();
            gamma[j] -= 1;
            terms.push(pool.mul(vec![
                pool.integer(beta[j]),
                cov[i][j],
                Self::monomial_moment(&gamma, mean, cov, memo, pool),
            ]));
        }
        let v = tidy(pool.add(terms), pool);
        memo.insert(alpha.to_vec(), v);
        v
    }
}

// ---------------------------------------------------------------------------
// Blocks of a joint law
// ---------------------------------------------------------------------------

/// One block of a joint law. Blocks are **mutually independent**; dependence
/// exists only inside a [`JointBlock::Gaussian`] block.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum JointBlock {
    /// One variate (a symbol) with a univariate law.
    Univariate(ExprId, Distribution),
    /// A vector of variates (symbols, one per dimension) with a joint normal
    /// law.
    Gaussian(Vec<ExprId>, MultivariateNormal),
}

impl JointBlock {
    fn vars(&self) -> Vec<ExprId> {
        match self {
            JointBlock::Univariate(v, _) => vec![*v],
            JointBlock::Gaussian(vs, _) => vs.clone(),
        }
    }
}

fn validate_blocks(blocks: &[JointBlock], pool: &ExprPool) -> Result<(), ProbError> {
    if blocks.is_empty() {
        return Err(ProbError::Unsupported(
            "a joint law needs at least one block".to_string(),
        ));
    }
    let mut seen: Vec<ExprId> = Vec::new();
    for b in blocks {
        if let JointBlock::Gaussian(vs, mvn) = b {
            if vs.len() != mvn.dim() {
                return Err(ProbError::Unsupported(format!(
                    "a Gaussian block names {} variates for a {}-dimensional normal",
                    vs.len(),
                    mvn.dim()
                )));
            }
        }
        for v in b.vars() {
            if !matches!(pool.get(v), ExprData::Symbol { .. }) {
                return Err(ProbError::Unsupported(format!(
                    "`{}` is not a symbol; each variate of a joint law must be a bare symbol",
                    pool.display(v)
                )));
            }
            if seen.contains(&v) {
                return Err(ProbError::Unsupported(format!(
                    "`{}` is named by two blocks; a variate has one law",
                    pool.display(v)
                )));
            }
            seen.push(v);
        }
    }
    Ok(())
}

/// Which blocks' variates `e` mentions.
fn blocks_in(e: ExprId, blocks: &[JointBlock], pool: &ExprPool) -> Vec<usize> {
    blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.vars().iter().any(|&v| dists::contains(e, v, pool)))
        .map(|(i, _)| i)
        .collect()
}

struct Engine<'a> {
    blocks: &'a [JointBlock],
    pool: &'a ExprPool,
    log: DerivationLog,
    cache: HashMap<(usize, ExprId), ExprId>,
    gaussian_memo: HashMap<(usize, Vec<ExprId>), HashMap<Vec<u32>, ExprId>>,
}

impl<'a> Engine<'a> {
    fn new(blocks: &'a [JointBlock], pool: &'a ExprPool) -> Self {
        Engine {
            blocks,
            pool,
            log: DerivationLog::new(),
            cache: HashMap::new(),
            gaussian_memo: HashMap::new(),
        }
    }

    fn expect(&mut self, expr: ExprId) -> Result<ExprId, ProbError> {
        let pool = self.pool;
        let expanded = simplify_expanded(expr, pool).value;
        let terms: Vec<ExprId> = match pool.get(expanded) {
            ExprData::Add(args) => args.to_vec(),
            _ => vec![expanded],
        };
        if terms.len() > MAX_JOINT_TERMS {
            return Err(ProbError::Unsupported(format!(
                "the expression expanded to {} terms, past the ceiling of {MAX_JOINT_TERMS}",
                terms.len()
            )));
        }
        let mut out = Vec::with_capacity(terms.len());
        for term in terms {
            out.push(self.expect_term(term)?);
        }
        Ok(sum_or_zero(out, pool))
    }

    fn expect_term(&mut self, term: ExprId) -> Result<ExprId, ProbError> {
        let pool = self.pool;
        let factors: Vec<ExprId> = match pool.get(term) {
            ExprData::Mul(args) => args.to_vec(),
            _ => vec![term],
        };
        let mut constants = Vec::new();
        let mut groups: Vec<Vec<ExprId>> = vec![Vec::new(); self.blocks.len()];
        for f in factors {
            let hit = blocks_in(f, self.blocks, pool);
            match hit.as_slice() {
                [] => constants.push(f),
                [b] => groups[*b].push(f),
                _ => self.split_coupled(f, &mut constants, &mut groups)?,
            }
        }
        let mut product = constants;
        for (b, g) in groups.into_iter().enumerate() {
            if g.is_empty() {
                continue;
            }
            let g = pool.mul(g);
            product.push(self.expect_block(b, g)?);
        }
        Ok(if product.is_empty() {
            pool.integer(1)
        } else {
            pool.mul(product)
        })
    }

    /// `exp(Σ_b a_b)` with each `a_b` in one block factors as `∏_b exp(a_b)`.
    /// Anything else that mentions two independent blocks at once is not a
    /// product of per-block factors, and independence says nothing about it.
    fn split_coupled(
        &self,
        f: ExprId,
        constants: &mut Vec<ExprId>,
        groups: &mut [Vec<ExprId>],
    ) -> Result<(), ProbError> {
        let pool = self.pool;
        let refuse = || {
            ProbError::Unsupported(format!(
                "`{}` couples variates from independent blocks in a way that does not factor \
                 into per-block pieces; only products of per-block factors (and exp of a sum) \
                 are taken apart by independence",
                pool.display(f)
            ))
        };
        let ExprData::Func { name, args } = pool.get(f) else {
            return Err(refuse());
        };
        if name != "exp" || args.len() != 1 {
            return Err(refuse());
        }
        let arg = simplify_expanded(args[0], pool).value;
        let parts: Vec<ExprId> = match pool.get(arg) {
            ExprData::Add(a) => a.to_vec(),
            _ => vec![arg],
        };
        let mut per: Vec<Vec<ExprId>> = vec![Vec::new(); groups.len()];
        let mut free = Vec::new();
        for p in parts {
            match blocks_in(p, self.blocks, pool).as_slice() {
                [] => free.push(p),
                [b] => per[*b].push(p),
                _ => return Err(refuse()),
            }
        }
        if !free.is_empty() {
            constants.push(exp(pool.add(free), pool));
        }
        for (b, p) in per.into_iter().enumerate() {
            if !p.is_empty() {
                groups[b].push(exp(pool.add(p), pool));
            }
        }
        Ok(())
    }

    fn expect_block(&mut self, b: usize, g: ExprId) -> Result<ExprId, ProbError> {
        if let Some(&v) = self.cache.get(&(b, g)) {
            return Ok(v);
        }
        let v = match &self.blocks[b] {
            JointBlock::Univariate(x, dist) => {
                let r = super::expectation(g, *x, dist, self.pool)?;
                self.log = std::mem::take(&mut self.log).merge(r.log);
                r.value
            }
            JointBlock::Gaussian(vars, mvn) => {
                let (vars, mvn) = (vars.clone(), mvn.clone());
                self.gaussian(b, g, &vars, &mvn)?
            }
        };
        self.cache.insert((b, g), v);
        Ok(v)
    }

    /// `E[m(X)·e^{tᵀX + c}]` under `N(μ, Σ)`, for a monomial `m`.
    fn gaussian(
        &mut self,
        b: usize,
        g: ExprId,
        vars: &[ExprId],
        mvn: &MultivariateNormal,
    ) -> Result<ExprId, ProbError> {
        let pool = self.pool;
        let d = vars.len();
        let mut alpha = vec![0u32; d];
        let mut t: Vec<Vec<ExprId>> = vec![Vec::new(); d];
        let mut c: Vec<ExprId> = Vec::new();
        let refuse = |what: ExprId| {
            ProbError::Unsupported(format!(
                "under a multivariate-normal block, E[·] is computed for a polynomial in the \
                 variates times exp of an affine form; `{}` is neither",
                pool.display(what)
            ))
        };
        let factors: Vec<ExprId> = match pool.get(g) {
            ExprData::Mul(args) => args.to_vec(),
            _ => vec![g],
        };
        for f in factors {
            if let Some(i) = vars.iter().position(|&v| v == f) {
                alpha[i] += 1;
                continue;
            }
            match pool.get(f) {
                ExprData::Pow { base, exp: e } => {
                    let Some(i) = vars.iter().position(|&v| v == base) else {
                        return Err(refuse(f));
                    };
                    let k = match pool.get(e) {
                        ExprData::Integer(k) => k.0.to_u32(),
                        _ => None,
                    };
                    let Some(k) = k else {
                        return Err(refuse(f));
                    };
                    alpha[i] += k;
                }
                ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                    let arg = simplify_expanded(args[0], pool).value;
                    let parts: Vec<ExprId> = match pool.get(arg) {
                        ExprData::Add(a) => a.to_vec(),
                        _ => vec![arg],
                    };
                    for p in parts {
                        let present: Vec<usize> = (0..d)
                            .filter(|&i| dists::contains(p, vars[i], pool))
                            .collect();
                        match present.as_slice() {
                            [] => c.push(p),
                            [i] => {
                                let Some(coef) = linear_coefficient(p, vars[*i], pool) else {
                                    return Err(refuse(f));
                                };
                                t[*i].push(coef);
                            }
                            _ => return Err(refuse(f)),
                        }
                    }
                }
                _ => return Err(refuse(f)),
            }
        }
        let degree: u32 = alpha.iter().sum();
        if degree > MAX_GAUSSIAN_DEGREE {
            return Err(ProbError::Unsupported(format!(
                "Gaussian moment of total degree {degree} exceeds the ceiling of \
                 {MAX_GAUSSIAN_DEGREE}"
            )));
        }
        let t: Vec<ExprId> = t
            .into_iter()
            .map(|parts| tidy(sum_or_zero(parts, pool), pool))
            .collect();
        let tilted = t.iter().any(|&ti| !is_zero(ti, pool));
        let (mean, factor) = if tilted {
            // Tilting: E[m(X) e^{tᵀX}] = M(t)·E_{μ+Σt}[m(X)].
            let mean: Vec<ExprId> = (0..d)
                .map(|i| {
                    let mut terms = vec![mvn.mean[i]];
                    for j in 0..d {
                        terms.push(pool.mul(vec![mvn.cov[i][j], t[j]]));
                    }
                    tidy(pool.add(terms), pool)
                })
                .collect();
            let arg = tidy(
                pool.add(vec![
                    mvn.dot_mean(&t, pool),
                    pool.mul(vec![pool.rational(1, 2), mvn.quad(&t, pool)]),
                    sum_or_zero(c.clone(), pool),
                ]),
                pool,
            );
            (mean, exp(arg, pool))
        } else if c.is_empty() {
            (mvn.mean.clone(), pool.integer(1))
        } else {
            (mvn.mean.clone(), exp(pool.add(c.clone()), pool))
        };
        let memo = self.gaussian_memo.entry((b, mean.clone())).or_default();
        let m = MultivariateNormal::monomial_moment(&alpha, &mean, &mvn.cov, memo, pool);
        let value = simplify(pool.mul(vec![factor, m]), pool).value;
        self.log.push(RewriteStep::with_conditions(
            if tilted {
                "prob_gaussian_moment_by_stein_identity_after_exponential_tilting"
            } else {
                "prob_gaussian_moment_by_stein_identity"
            },
            g,
            value,
            mvn.undecided(pool),
        ));
        Ok(value)
    }
}

/// `c` when `term = c·v` with `c` free of `v`.
fn linear_coefficient(term: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if term == v {
        return Some(pool.integer(1));
    }
    let ExprData::Mul(args) = pool.get(term) else {
        return None;
    };
    let mut rest = Vec::new();
    let mut found = false;
    for a in args {
        if a == v && !found {
            found = true;
        } else if dists::contains(a, v, pool) {
            return None;
        } else {
            rest.push(a);
        }
    }
    if !found {
        return None;
    }
    Some(if rest.is_empty() {
        pool.integer(1)
    } else {
        pool.mul(rest)
    })
}

/// `E[expr]` under a joint law of mutually independent blocks.
///
/// `expr` is expanded; each term is split into per-block factors, which
/// independence lets be averaged separately. A univariate block's factor goes
/// through [`super::expectation`] (quadrature-verified). A Gaussian block's
/// factor must be a monomial in its variates times `exp` of an affine form,
/// and is computed exactly by Stein's identity and exponential tilting.
///
/// # Errors
///
/// [`ProbError::Unsupported`] for malformed blocks, a factor that couples two
/// independent blocks non-separably (`sin(X + Y)` with `X`, `Y` in different
/// blocks), a non-polynomial factor of a Gaussian block, or a Gaussian moment
/// of total degree above 12; and whatever [`super::expectation`] returns for a
/// univariate factor.
pub fn joint_expectation(
    expr: ExprId,
    blocks: &[JointBlock],
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    validate_blocks(blocks, pool)?;
    let mut engine = Engine::new(blocks, pool);
    let value = tidy(engine.expect(expr)?, pool);
    let mut log = engine.log;
    log.push(RewriteStep::simple(
        "prob_joint_expectation_by_block_independence",
        expr,
        value,
    ));
    super::stash_prob_side_conditions(super::conditions_of(&log));
    Ok(DerivedExpr::with_log(value, log))
}

/// `Cov[e₁, e₂] = E[e₁e₂] − E[e₁]E[e₂]` under a joint law of independent
/// blocks.
///
/// # Errors
///
/// As [`joint_expectation`].
pub fn covariance(
    e1: ExprId,
    e2: ExprId,
    blocks: &[JointBlock],
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    validate_blocks(blocks, pool)?;
    let mut engine = Engine::new(blocks, pool);
    let both = engine.expect(pool.mul(vec![e1, e2]))?;
    let m1 = engine.expect(e1)?;
    let m2 = if e2 == e1 { m1 } else { engine.expect(e2)? };
    let value = tidy(
        pool.add(vec![both, pool.mul(vec![pool.integer(-1), m1, m2])]),
        pool,
    );
    let mut log = engine.log;
    log.push(RewriteStep::simple(
        "prob_covariance_as_e_xy_minus_e_x_e_y",
        pool.mul(vec![e1, e2]),
        value,
    ));
    super::stash_prob_side_conditions(super::conditions_of(&log));
    Ok(DerivedExpr::with_log(value, log))
}

/// `Var[e] = Cov[e, e]` under a joint law of independent blocks — the
/// correlated counterpart of [`super::variance_affine_independent`], and not
/// restricted to affine `e`.
///
/// # Errors
///
/// As [`joint_expectation`].
pub fn joint_variance(
    e: ExprId,
    blocks: &[JointBlock],
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    covariance(e, e, blocks, pool)
}

// ---------------------------------------------------------------------------
// Sums of independent variates
// ---------------------------------------------------------------------------

/// Structural equality after simplification — the only notion of "same
/// parameter" that cannot be wrong. Two symbols that happen to be equal at
/// runtime are not the same parameter here.
fn same(a: ExprId, b: ExprId, pool: &ExprPool) -> bool {
    a == b || is_zero(simplify(sub(a, b, pool), pool).value, pool)
}

/// The law of `X₁ + … + X_n` for **independent** `Xᵢ`, where it stays in a
/// closed family (convolution read off the product of MGFs):
///
/// | summands | sum |
/// |---|---|
/// | `Normal(μᵢ, σᵢ)` | `Normal(Σμᵢ, √(Σσᵢ²))` |
/// | `Gamma(kᵢ, θ)`, `Exponential(1/θ)`, common scale | `Gamma(Σkᵢ, θ)` |
/// | `Poisson(λᵢ)` | `Poisson(Σλᵢ)` |
/// | `Binomial(nᵢ, p)`, `Bernoulli(p)`, common `p` | `Binomial(Σnᵢ, p)` |
/// | `NegativeBinomial(rᵢ, p)`, common `p` | `NegativeBinomial(Σrᵢ, p)` |
///
/// "Common" means the same expression (after simplification): two different
/// symbols are different parameters even if a caller means them to be equal.
///
/// # Errors
///
/// [`ProbError::Unsupported`] for an empty list, a mixture of families, or
/// mismatched scale/probability parameters — the sum exists, its MGF is
/// `∏Mᵢ(t)`, but it is not one of the laws this module names.
pub fn sum_independent(dists: &[Distribution], pool: &ExprPool) -> Result<Distribution, ProbError> {
    let Some(first) = dists.first() else {
        return Err(ProbError::Unsupported(
            "sum_independent needs at least one summand".to_string(),
        ));
    };
    if dists.len() == 1 {
        return Ok(first.clone());
    }
    let unsupported = |why: &str| {
        let names: Vec<&str> = dists.iter().map(|d| d.kind().name()).collect();
        ProbError::Unsupported(format!(
            "the sum of independent {} is not a law this module names ({why}); its moment \
             generating function is the product of the summands'",
            names.join(" + ")
        ))
    };
    let all = |k: &[DistKind]| dists.iter().all(|d| k.contains(&d.kind()));
    let params = |i: usize| -> Vec<ExprId> { dists.iter().map(|d| d.params()[i]).collect() };

    if all(&[DistKind::Normal]) {
        let mu = simplify(pool.add(params(0)), pool).value;
        let var = tidy(
            pool.add(
                params(1)
                    .into_iter()
                    .map(|s| pool.pow(s, pool.integer(2)))
                    .collect(),
            ),
            pool,
        );
        let sigma = simplify(pool.pow(var, pool.rational(1, 2)), pool).value;
        return Distribution::normal(mu, sigma, pool);
    }
    if all(&[DistKind::Poisson]) {
        let lam = simplify(pool.add(params(0)), pool).value;
        return Distribution::poisson(lam, pool);
    }
    if all(&[DistKind::Gamma, DistKind::Exponential]) {
        // (shape, scale) for each; Exponential(λ) is Gamma(1, 1/λ).
        let shape_scale: Vec<(ExprId, ExprId)> = dists
            .iter()
            .map(|d| match d.kind() {
                DistKind::Gamma => (d.params()[0], d.params()[1]),
                _ => (
                    pool.integer(1),
                    simplify(pool.pow(d.params()[0], pool.integer(-1)), pool).value,
                ),
            })
            .collect();
        let theta = shape_scale[0].1;
        if !shape_scale.iter().all(|&(_, s)| same(s, theta, pool)) {
            return Err(unsupported("the scales differ"));
        }
        let k = simplify(
            pool.add(shape_scale.iter().map(|&(k, _)| k).collect()),
            pool,
        )
        .value;
        return Distribution::gamma(k, theta, pool);
    }
    if all(&[DistKind::Binomial, DistKind::Bernoulli]) {
        let np: Vec<(ExprId, ExprId)> = dists
            .iter()
            .map(|d| match d.kind() {
                DistKind::Binomial => (d.params()[0], d.params()[1]),
                _ => (pool.integer(1), d.params()[0]),
            })
            .collect();
        let p = np[0].1;
        if !np.iter().all(|&(_, q)| same(q, p, pool)) {
            return Err(unsupported("the success probabilities differ"));
        }
        let n = simplify(pool.add(np.iter().map(|&(n, _)| n).collect()), pool).value;
        return Distribution::binomial(n, p, pool);
    }
    if all(&[DistKind::NegativeBinomial]) {
        let p = dists[0].params()[1];
        if !dists.iter().all(|d| same(d.params()[1], p, pool)) {
            return Err(unsupported("the success probabilities differ"));
        }
        let r = simplify(pool.add(params(0)), pool).value;
        return Distribution::negative_binomial(r, p, pool);
    }
    Err(unsupported(
        "the summands are not all from one closed family",
    ))
}
