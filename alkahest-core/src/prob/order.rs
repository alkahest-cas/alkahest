//! The minimum and maximum of `n` iid variates, for a symbolic `n`.
//!
//! For `X₁, …, X_n` iid with CDF `F` and density `f`,
//!
//! * `max` has CDF `F(x)ⁿ` and density `n F(x)ⁿ⁻¹ f(x)`;
//! * `min` has CDF `1 - (1 - F(x))ⁿ` and density `n (1 - F(x))ⁿ⁻¹ f(x)`.
//!
//! Those hold for every law with a closed-form CDF. The **moments** need one
//! more integral, and that closes for the two laws the table carries:
//!
//! | law | `E[max]` | `E[min]` | variance |
//! |---|---|---|---|
//! | `Uniform(a, b)` | `a + (b-a)n/(n+1)` | `a + (b-a)/(n+1)` | `(b-a)² n/((n+1)²(n+2))` (both) |
//! | `Exponential(λ)` | `(ψ(n+1) - ψ(1))/λ = H_n/λ` | `1/(nλ)` | `(ψ₁(1) - ψ₁(n+1))/λ²` ; `1/(nλ)²` |
//!
//! `H_n = ψ(n+1) - ψ(1)` is the harmonic number, written through the digamma
//! function so that it is defined — and correct — for a symbolic `n`. The
//! minimum of exponentials *is* an `Exponential(nλ)` and the extreme order
//! statistics of a `Uniform(0, 1)` *are* `Beta(n, 1)` and `Beta(1, n)`;
//! [`OrderStatistic::distribution`] returns those laws so the whole of this
//! module applies to them.
//!
//! Every moment is checked by the same numeric gate as a univariate one: the
//! claim against quadrature of `x · n F(x)ⁿ⁻¹ f(x)` over the base law's
//! support, at several parameter points **and several values of `n`**. The
//! formulas above are exact for real `n > 0` (the law with CDF `F(x)ⁿ` exists
//! for every real `n > 0`), so non-integer sample points are a stronger check,
//! not a weaker one.

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::simplify;

use super::dists::{self, Sign};
use super::{div, pi, sub, verify, DistKind, Distribution, ProbError};

/// Which extreme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OrderKind {
    /// `min(X₁, …, X_n)`.
    Min,
    /// `max(X₁, …, X_n)`.
    Max,
}

/// The law of the minimum or maximum of `n` iid copies of a base law.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderStatistic {
    base: Distribution,
    n: ExprId,
    kind: OrderKind,
}

/// The minimum or maximum of `n` iid copies of `dist`.
///
/// `n` is a positive integer literal or a symbolic expression; a symbolic `n`
/// carries `n > 0` and `n ∈ ℤ` as side conditions on everything derived.
///
/// # Errors
///
/// [`ProbError::InvalidParameter`] for a numeric `n` that is not a positive
/// integer; [`ProbError::Unsupported`] for a base law other than `Uniform` and
/// `Exponential` (the CDF-and-density formulas hold in general, but the moments
/// are tabled only for those two, and a law whose moments refuse is not worth
/// half-offering).
pub fn order_statistic(
    dist: &Distribution,
    n: ExprId,
    kind: OrderKind,
    pool: &ExprPool,
) -> Result<OrderStatistic, ProbError> {
    match pool.get(n) {
        ExprData::Integer(v) if v.0 >= 1 => {}
        _ if super::numeric_ball(n, pool).is_some() => {
            return Err(ProbError::InvalidParameter {
                parameter: "n",
                requirement: "a positive integer (a literal, or a symbolic expression)",
            })
        }
        _ => {}
    }
    if !matches!(dist.kind(), DistKind::Uniform | DistKind::Exponential) {
        return Err(ProbError::Unsupported(format!(
            "order statistics are tabled for Uniform and Exponential base laws; {} is not one \
             of them",
            dist.kind().name()
        )));
    }
    Ok(OrderStatistic {
        base: dist.clone(),
        n,
        kind,
    })
}

impl OrderStatistic {
    /// The base law.
    pub fn base(&self) -> &Distribution {
        &self.base
    }

    /// The sample size `n`.
    pub fn n(&self) -> ExprId {
        self.n
    }

    /// Minimum or maximum.
    pub fn kind(&self) -> OrderKind {
        self.kind
    }

    /// `n > 0`, `n ∈ ℤ` for a symbolic `n`, plus the base law's own
    /// undischarged constraints.
    fn conditions(&self, pool: &ExprPool) -> Vec<SideCondition> {
        let mut out = Vec::new();
        if !matches!(pool.get(self.n), ExprData::Integer(_)) {
            if matches!(dists::sign_of(self.n, pool), Sign::Unknown) {
                out.push(SideCondition::Positive(self.n));
            }
            out.push(SideCondition::InDomain(self.n, Domain::Integer));
        }
        for c in self.base.constraints(pool) {
            if let Some(side) = verify::undischarged(c, pool) {
                out.push(side);
            }
        }
        out
    }

    fn finish(
        &self,
        rule: &'static str,
        value: ExprId,
        extra: Vec<RewriteStep>,
        pool: &ExprPool,
    ) -> DerivedExpr<ExprId> {
        let mut log = DerivationLog::new();
        for s in extra {
            log.push(s);
        }
        log.push(RewriteStep::with_conditions(
            rule,
            value,
            value,
            self.conditions(pool),
        ));
        super::stash_prob_side_conditions(super::conditions_of(&log));
        DerivedExpr::with_log(value, log)
    }

    /// `F_{(·)}(x)` in terms of the base CDF `F`, written in-support.
    fn cdf_from(&self, base_cdf: ExprId, pool: &ExprPool) -> ExprId {
        let one = pool.integer(1);
        match self.kind {
            OrderKind::Max => pool.pow(base_cdf, self.n),
            OrderKind::Min => sub(one, pool.pow(sub(one, base_cdf, pool), self.n), pool),
        }
    }

    /// `P(M ≤ x)`. The base law's CDF is obtained from
    /// [`Distribution::cdf`] — so it is verified, and an argument that can be
    /// placed outside the support is answered exactly (`0` or `1`) — and then
    /// composed.
    ///
    /// # Errors
    ///
    /// Whatever [`Distribution::cdf`] returns for the base law.
    pub fn cdf(&self, x: ExprId, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
        let base = self.base.cdf(x, pool)?;
        let value = simplify(self.cdf_from(base.value, pool), pool).value;
        let steps = base.log.0.clone();
        Ok(self.finish("prob_order_statistic_cdf", value, steps, pool))
    }

    /// The density on the support, `n F(x)ⁿ⁻¹ f(x)` (max) or
    /// `n (1 - F(x))ⁿ⁻¹ f(x)` (min). Like [`Distribution::pdf`] it is the
    /// in-support formula and is not extended by zero.
    ///
    /// # Errors
    ///
    /// Whatever the base law's CDF table returns.
    pub fn pdf(&self, x: ExprId, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
        let value = simplify(self.density(x, pool)?, pool).value;
        Ok(self.finish("prob_order_statistic_density", value, Vec::new(), pool))
    }

    fn density(&self, x: ExprId, pool: &ExprPool) -> Result<ExprId, ProbError> {
        let one = pool.integer(1);
        let f = dists::pdf(&self.base, x, pool);
        let big_f = super::moments::cdf_claim(&self.base, x, pool)?;
        let n_minus_1 = sub(self.n, one, pool);
        let core = match self.kind {
            OrderKind::Max => pool.pow(big_f, n_minus_1),
            OrderKind::Min => pool.pow(sub(one, big_f, pool), n_minus_1),
        };
        Ok(pool.mul(vec![self.n, core, f]))
    }

    /// The weight `n F(x)ⁿ⁻¹` (max) or `n (1 - F(x))ⁿ⁻¹` (min) that turns
    /// the base density into this one — the verifier integrates
    /// `g(x)·weight·f(x)` against the base law.
    fn weight(&self, x: ExprId, pool: &ExprPool) -> Result<ExprId, ProbError> {
        let one = pool.integer(1);
        let big_f = super::moments::cdf_claim(&self.base, x, pool)?;
        let n_minus_1 = sub(self.n, one, pool);
        Ok(pool.mul(vec![
            self.n,
            match self.kind {
                OrderKind::Max => pool.pow(big_f, n_minus_1),
                OrderKind::Min => pool.pow(sub(one, big_f, pool), n_minus_1),
            },
        ]))
    }

    fn mean_claim(&self, pool: &ExprPool) -> ExprId {
        let p = self.base.params();
        let one = pool.integer(1);
        let n = self.n;
        let n1 = pool.add(vec![n, one]);
        match (self.base.kind(), self.kind) {
            (DistKind::Uniform, OrderKind::Max) => pool.add(vec![
                p[0],
                div(pool.mul(vec![sub(p[1], p[0], pool), n]), n1, pool),
            ]),
            (DistKind::Uniform, OrderKind::Min) => {
                pool.add(vec![p[0], div(sub(p[1], p[0], pool), n1, pool)])
            }
            // H_n / λ, H_n = ψ(n+1) - ψ(1).
            (DistKind::Exponential, OrderKind::Max) => div(
                sub(
                    pool.func("digamma", vec![n1]),
                    pool.func("digamma", vec![one]),
                    pool,
                ),
                p[0],
                pool,
            ),
            (DistKind::Exponential, OrderKind::Min) => {
                pool.pow(pool.mul(vec![n, p[0]]), pool.integer(-1))
            }
            _ => unreachable!("order_statistic admits only Uniform and Exponential"),
        }
    }

    fn variance_claim(&self, pool: &ExprPool) -> ExprId {
        let p = self.base.params();
        let one = pool.integer(1);
        let two = pool.integer(2);
        let n = self.n;
        match (self.base.kind(), self.kind) {
            (DistKind::Uniform, _) => div(
                pool.mul(vec![pool.pow(sub(p[1], p[0], pool), two), n]),
                pool.mul(vec![
                    pool.pow(pool.add(vec![n, one]), two),
                    pool.add(vec![n, two]),
                ]),
                pool,
            ),
            // Σ_{k≤n} 1/(kλ)² = (ψ₁(1) - ψ₁(n+1))/λ², ψ₁(1) = π²/6.
            (DistKind::Exponential, OrderKind::Max) => div(
                sub(
                    div(pool.pow(pi(pool), two), pool.integer(6), pool),
                    pool.func("trigamma", vec![pool.add(vec![n, one])]),
                    pool,
                ),
                pool.pow(p[0], two),
                pool,
            ),
            (DistKind::Exponential, OrderKind::Min) => {
                pool.pow(pool.mul(vec![n, p[0]]), pool.integer(-2))
            }
            _ => unreachable!("order_statistic admits only Uniform and Exponential"),
        }
    }

    /// `E[M]`, from the table above, verified against quadrature of
    /// `x · n F(x)ⁿ⁻¹ f(x)`.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unverified`] if the gate cannot confirm the table entry.
    pub fn mean(&self, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        let claim = simplify(self.mean_claim(pool), pool).value;
        let x = dists::fresh_var(&[claim, self.n], pool);
        let integrand = pool.mul(vec![
            x,
            self.weight(x, pool)?,
            dists::pdf(&self.base, x, pool),
        ]);
        let evidence = verify::check(claim, integrand, x, &self.base, pool)?;
        let step = verify::evidence_step(&evidence, claim, &self.base, pool);
        Ok(self.finish("prob_order_statistic_mean", claim, vec![step], pool))
    }

    /// `Var[M]`, from the table above, verified against quadrature of
    /// `(x - E[M])² · n F(x)ⁿ⁻¹ f(x)`.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unverified`] if the gate cannot confirm the table entry.
    pub fn variance(&self, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        let claim = simplify(self.variance_claim(pool), pool).value;
        let mean = simplify(self.mean_claim(pool), pool).value;
        let x = dists::fresh_var(&[claim, mean, self.n], pool);
        let integrand = pool.mul(vec![
            pool.pow(sub(x, mean, pool), pool.integer(2)),
            self.weight(x, pool)?,
            dists::pdf(&self.base, x, pool),
        ]);
        let evidence = verify::check(claim, integrand, x, &self.base, pool)?;
        let step = verify::evidence_step(&evidence, claim, &self.base, pool);
        Ok(self.finish("prob_order_statistic_variance", claim, vec![step], pool))
    }

    /// The order statistic as a named law, where it is one:
    /// `min` of `Exponential(λ)` is `Exponential(nλ)`; `max` and `min` of
    /// `Uniform(0, 1)` are `Beta(n, 1)` and `Beta(1, n)`. `None` otherwise
    /// (the `max` of exponentials, a `Uniform(a, b)` off the unit interval).
    pub fn distribution(&self, pool: &ExprPool) -> Option<Distribution> {
        let p = self.base.params();
        let is = |e: ExprId, v: i64| matches!(pool.get(e), ExprData::Integer(z) if z.0 == v);
        match (self.base.kind(), self.kind) {
            (DistKind::Exponential, OrderKind::Min) => {
                let rate = simplify(pool.mul(vec![self.n, p[0]]), pool).value;
                Distribution::exponential(rate, pool).ok()
            }
            (DistKind::Uniform, kind) if is(p[0], 0) && is(p[1], 1) => {
                let one = pool.integer(1);
                match kind {
                    OrderKind::Max => Distribution::beta(self.n, one, pool).ok(),
                    OrderKind::Min => Distribution::beta(one, self.n, pool).ok(),
                }
            }
            _ => None,
        }
    }
}
