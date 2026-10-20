//! The minimum, maximum and general `k`-th order statistic of `n` iid
//! variates.
//!
//! # The `k`-th smallest
//!
//! [`kth_order_statistic`] is the `k`-th smallest of `n` (`k = 1` the minimum,
//! `k = n` the maximum). For **any** continuous law with a closed-form CDF its
//! density is
//!
//! ```text
//! f_(k)(x) = n! / ((k-1)! (n-k)!) · F(x)^{k-1} (1 - F(x))^{n-k} · f(x)
//! ```
//!
//! and, for a literal `n`, its CDF is the binomial tail
//! `Σ_{j=k}^{n} C(n,j) F^j (1-F)^{n-j}`. Both are exact; the base law's CDF
//! comes from [`Distribution::cdf`], so it is verified, and a base law whose
//! CDF has no closed form (a `Gamma` with a non-integer shape) refuses.
//! A *discrete* base law is refused: with ties the formulas above are wrong.
//!
//! The **moments** of a `k`-th order statistic are the expectation of
//! `x · n!/((k-1)!(n-k)!) F^{k-1}(1-F)^{n-k}` under the base law; they are
//! computed by [`super::expectation`] — so by the integrator, and gated against
//! quadrature — and **refuse** (`E-PROB-003`) when it declines. That succeeds
//! for the polynomial/exponential CDFs (`Uniform`, `Exponential`, `Beta` and
//! `Gamma` with integer parameters) and is not expected to for the `erf` of a
//! `Normal`. It needs a literal `n` (at most [`MAX_ORDER_SAMPLE`]): the
//! integrand is expanded term by term.
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
use crate::simplify::{simplify, simplify_expanded};

use super::dists::{self, Sign};
use super::{div, pi, sub, verify, DistKind, Distribution, ProbError};

/// Largest literal sample size whose order-statistic CDF is written as an
/// explicit binomial tail, and whose moments are expanded for the integrator.
pub const MAX_ORDER_SAMPLE: u32 = 24;

/// Which extreme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OrderKind {
    /// `min(X₁, …, X_n)`.
    Min,
    /// `max(X₁, …, X_n)`.
    Max,
}

/// Which order statistic, internally: the two extremes, or the `k`-th smallest
/// for a rank strictly inside `1..n` (or not decidable). Private so that the
/// public [`OrderKind`] stays a fieldless enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rank {
    Min,
    Max,
    Kth(ExprId),
}

/// The law of the minimum, maximum or `k`-th smallest of `n` iid copies of a
/// base law.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderStatistic {
    base: Distribution,
    n: ExprId,
    kind: Rank,
}

/// The minimum or maximum of `n` iid copies of `dist`.
///
/// `n` is a positive integer literal or a symbolic expression; a symbolic `n`
/// carries `n > 0` and `n ∈ ℤ` as side conditions on everything derived. For
/// the `k`-th smallest see [`kth_order_statistic`].
///
/// # Errors
///
/// [`ProbError::InvalidParameter`] for a numeric `n` that is not a positive
/// integer; [`ProbError::Unsupported`] for a discrete base law (ties break the
/// density formula).
pub fn order_statistic(
    dist: &Distribution,
    n: ExprId,
    kind: OrderKind,
    pool: &ExprPool,
) -> Result<OrderStatistic, ProbError> {
    let rank = match kind {
        OrderKind::Min => Rank::Min,
        OrderKind::Max => Rank::Max,
    };
    build(dist, n, rank, pool)
}

/// The `k`-th smallest (1-based) of `n` iid copies of `dist`; `k = 1` is the
/// minimum and `k = n` the maximum, and a literal `k` equal to either is
/// normalised to it.
///
/// `n` and `k` are positive integer literals (`k <= n`) or symbolic
/// expressions; a symbolic `k` carries `k > 0`, `k ∈ ℤ` and `n - k + 1 > 0`.
///
/// # Errors
///
/// [`ProbError::InvalidParameter`] for a numeric `n` or `k` that is not a
/// positive integer, or a literal `k > n`; [`ProbError::Unsupported`] for a
/// discrete base law.
pub fn kth_order_statistic(
    dist: &Distribution,
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
) -> Result<OrderStatistic, ProbError> {
    match pool.get(k) {
        ExprData::Integer(v) if v.0 >= 1 => {}
        _ if super::numeric_ball(k, pool).is_some() => {
            return Err(ProbError::InvalidParameter {
                parameter: "k",
                requirement: "a positive integer (a literal, or a symbolic expression)",
            })
        }
        _ => {}
    }
    let (nl, kl) = (literal(n, pool), literal(k, pool));
    if let (Some(nn), Some(kk)) = (&nl, &kl) {
        if kk > nn {
            return Err(ProbError::InvalidParameter {
                parameter: "k",
                requirement: "at most n",
            });
        }
    }
    let rank = if kl.as_ref().is_some_and(|v| *v == 1) {
        Rank::Min
    } else if k == n || (nl.is_some() && nl == kl) {
        Rank::Max
    } else {
        Rank::Kth(k)
    };
    build(dist, n, rank, pool)
}

fn build(
    dist: &Distribution,
    n: ExprId,
    kind: Rank,
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
    if dist.support(pool).is_discrete() {
        return Err(ProbError::Unsupported(format!(
            "order statistics are for continuous base laws; {} has ties, which change the \
             density formula",
            dist.kind().name()
        )));
    }
    Ok(OrderStatistic {
        base: dist.clone(),
        n,
        kind,
    })
}

fn literal(e: ExprId, pool: &ExprPool) -> Option<rug::Integer> {
    match pool.get(e) {
        ExprData::Integer(v) => Some(v.0.clone()),
        _ => None,
    }
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

    /// Minimum or maximum. For a `k`-th order statistic strictly between the
    /// extremes (see [`kth_order_statistic`]) there is no such kind and this
    /// returns [`OrderKind::Max`]; check [`OrderStatistic::is_extreme`] first.
    pub fn kind(&self) -> OrderKind {
        match self.kind {
            Rank::Min => OrderKind::Min,
            Rank::Max | Rank::Kth(_) => OrderKind::Max,
        }
    }

    /// Is this the minimum or the maximum (as opposed to an interior `k`-th
    /// smallest)?
    pub fn is_extreme(&self) -> bool {
        !matches!(self.kind, Rank::Kth(_))
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
        if let Rank::Kth(k) = self.kind {
            if !matches!(pool.get(k), ExprData::Integer(_)) {
                if matches!(dists::sign_of(k, pool), Sign::Unknown) {
                    out.push(SideCondition::Positive(k));
                }
                out.push(SideCondition::InDomain(k, Domain::Integer));
                // k <= n for integers is n - k + 1 > 0.
                let room =
                    simplify(pool.add(vec![sub(self.n, k, pool), pool.integer(1)]), pool).value;
                if matches!(dists::sign_of(room, pool), Sign::Unknown) {
                    out.push(SideCondition::Positive(room));
                }
            }
        }
        for c in self.base.constraints(pool) {
            if let Some(side) = verify::undischarged(c, pool) {
                out.push(side);
            }
        }
        out
    }

    /// The literal `(n, k)` when both are small positive integers.
    fn literal_nk(&self, pool: &ExprPool) -> Option<(u32, u32)> {
        let n = dists::literal_count(self.n, pool)?;
        let k = match self.kind {
            Rank::Min => 1,
            Rank::Max => n,
            Rank::Kth(k) => dists::literal_count(k, pool)?,
        };
        (1..=MAX_ORDER_SAMPLE).contains(&n).then_some((n, k))
    }

    /// The rank `k` as an expression.
    pub fn rank(&self, pool: &ExprPool) -> ExprId {
        match self.kind {
            Rank::Min => pool.integer(1),
            Rank::Max => self.n,
            Rank::Kth(k) => k,
        }
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
    fn cdf_from(&self, base_cdf: ExprId, pool: &ExprPool) -> Result<ExprId, ProbError> {
        let one = pool.integer(1);
        Ok(match self.kind {
            Rank::Max => pool.pow(base_cdf, self.n),
            Rank::Min => sub(one, pool.pow(sub(one, base_cdf, pool), self.n), pool),
            Rank::Kth(_) => {
                // sum_{j=k}^{n} C(n,j) F^j (1-F)^{n-j}: the number of sample
                // points at or below x is Binomial(n, F(x)), and X_(k) <= x
                // exactly when at least k of them are.
                let Some((n, k)) = self.literal_nk(pool) else {
                    return Err(ProbError::NoClosedForm {
                        quantity: "the CDF of a k-th order statistic with a symbolic n or k",
                        missing: "the regularised incomplete beta function I_F(k, n - k + 1)",
                    });
                };
                let q = sub(one, base_cdf, pool);
                let terms = (k..=n)
                    .map(|j| {
                        pool.mul(vec![
                            pool.integer(rug::Integer::from(n).binomial(j)),
                            pool.pow(base_cdf, pool.integer(j)),
                            pool.pow(q, pool.integer(n - j)),
                        ])
                    })
                    .collect();
                pool.add(terms)
            }
        })
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
        let value = simplify(self.cdf_from(base.value, pool)?, pool).value;
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
        let f = dists::pdf(&self.base, x, pool);
        Ok(pool.mul(vec![self.weight(x, pool)?, f]))
    }

    /// The factor `c F^{k-1} (1-F)^{n-k}` that turns the base density into
    /// this one (`n F^{n-1}` for the maximum, `n (1-F)^{n-1}` for the
    /// minimum) -- the verifier integrates `g(x)*weight*f(x)` against the base
    /// law.
    fn weight(&self, x: ExprId, pool: &ExprPool) -> Result<ExprId, ProbError> {
        let one = pool.integer(1);
        let big_f = super::moments::cdf_claim(&self.base, x, pool)?;
        let n_minus_1 = simplify(sub(self.n, one, pool), pool).value;
        Ok(match self.kind {
            Rank::Max => pool.mul(vec![self.n, pool.pow(big_f, n_minus_1)]),
            Rank::Min => pool.mul(vec![self.n, pool.pow(sub(one, big_f, pool), n_minus_1)]),
            Rank::Kth(k) => {
                // n!/((k-1)!(n-k)!) -- an integer for literals, a Gamma ratio
                // otherwise.
                let coeff = match self.literal_nk(pool) {
                    Some((n, k)) => pool.integer(rug::Integer::from(n - 1).binomial(k - 1) * n),
                    None => div(
                        pool.func("gamma", vec![pool.add(vec![self.n, one])]),
                        pool.mul(vec![
                            pool.func("gamma", vec![k]),
                            pool.func("gamma", vec![pool.add(vec![sub(self.n, k, pool), one])]),
                        ]),
                        pool,
                    ),
                };
                pool.mul(vec![
                    coeff,
                    pool.pow(big_f, simplify(sub(k, one, pool), pool).value),
                    pool.pow(
                        sub(one, big_f, pool),
                        simplify(sub(self.n, k, pool), pool).value,
                    ),
                ])
            }
        })
    }

    /// `E[M^power]` by the general route: the expectation of
    /// `x^power * weight(x)` under the base law, which the integrator
    /// evaluates and the numeric gate checks. Needs a literal `n`.
    fn general_moment(
        &self,
        power: u32,
        pool: &ExprPool,
    ) -> Result<DerivedExpr<ExprId>, ProbError> {
        if self.literal_nk(pool).is_none() {
            return Err(ProbError::Unsupported(format!(
                "the moments of {} are computed for a literal sample size n <= {} (and a \
                 literal rank k); a symbolic n or k is tabled only for the extremes of \
                 Uniform and Exponential",
                self.describe(),
                MAX_ORDER_SAMPLE
            )));
        }
        let x = dists::fresh_var(&[self.n, self.rank(pool)], pool);
        let f = pool.mul(vec![
            pool.pow(x, pool.integer(power)),
            self.weight(x, pool)?,
        ]);
        super::expectation(f, x, &self.base, pool)
    }

    fn describe(&self) -> &'static str {
        match self.kind {
            Rank::Min => "the minimum",
            Rank::Max => "the maximum",
            Rank::Kth(_) => "a k-th order statistic",
        }
    }

    /// Is `E[M]`/`Var[M]` tabled in closed form for any `n`?
    fn tabled(&self, pool: &ExprPool) -> bool {
        (matches!(self.kind, Rank::Min | Rank::Max)
            && matches!(self.base.kind(), DistKind::Uniform | DistKind::Exponential))
            || self.normal_entry(pool).is_some()
    }

    /// The standard-normal order statistic `Z_(k:n)` for `n <= 3`, where its
    /// moments close: `(E[Z], Var[Z])` as expressions.
    ///
    /// `E[Z_(2:2)] = 1/√π`, `E[Z_(3:3)] = 3/(2√π)`, `Z_(2:3)` is symmetric
    /// (mean `0`), the lower order statistics are the negatives, and
    /// `E[Z_(2:2)²] = 1`, `E[Z_(3:3)²] = 1 + √3/(2π)`, `E[Z_(2:3)²] = 1 - √3/π`.
    /// For `n >= 4` the means involve `arctan`-type constants and the integrator
    /// does not close them: they refuse. Every entry is still checked against
    /// quadrature by [`OrderStatistic::mean`] / [`OrderStatistic::variance`].
    fn normal_entry(&self, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
        if self.base.kind() != DistKind::Normal {
            return None;
        }
        let (n, k) = self.literal_nk(pool)?;
        let pi = pi(pool);
        let inv_sqrt_pi = pool.pow(pi, pool.rational(-1, 2));
        let sqrt3_over_pi = pool.mul(vec![
            pool.pow(pool.integer(3), pool.rational(1, 2)),
            pool.pow(pi, pool.integer(-1)),
        ]);
        let one = pool.integer(1);
        // (sign, |mean| coefficient of 1/√π, variance)
        let upper = k * 2 > n + 1; // k is in the upper half
        let (mean_c, var) = match (n, k) {
            (1, _) => (None, one),
            (2, _) => (
                Some((pool.rational(1, 1), upper)),
                sub(one, pool.pow(pi, pool.integer(-1)), pool),
            ),
            (3, 2) => (None, sub(one, sqrt3_over_pi, pool)),
            (3, _) => (
                Some((pool.rational(3, 2), upper)),
                pool.add(vec![
                    one,
                    pool.mul(vec![pool.rational(1, 2), sqrt3_over_pi]),
                    pool.mul(vec![pool.rational(-9, 4), pool.pow(pi, pool.integer(-1))]),
                ]),
            ),
            _ => return None,
        };
        let e = match mean_c {
            None => pool.integer(0),
            Some((c, up)) => {
                let m = pool.mul(vec![c, inv_sqrt_pi]);
                if up {
                    m
                } else {
                    pool.mul(vec![pool.integer(-1), m])
                }
            }
        };
        Some((e, var))
    }

    fn mean_claim(&self, pool: &ExprPool) -> ExprId {
        let p = self.base.params();
        let one = pool.integer(1);
        let n = self.n;
        let n1 = pool.add(vec![n, one]);
        if let Some((e, _)) = self.normal_entry(pool) {
            return pool.add(vec![p[0], pool.mul(vec![p[1], e])]);
        }
        match (self.base.kind(), self.kind) {
            (DistKind::Uniform, Rank::Max) => pool.add(vec![
                p[0],
                div(pool.mul(vec![sub(p[1], p[0], pool), n]), n1, pool),
            ]),
            (DistKind::Uniform, Rank::Min) => {
                pool.add(vec![p[0], div(sub(p[1], p[0], pool), n1, pool)])
            }
            // H_n / λ, H_n = ψ(n+1) - ψ(1).
            (DistKind::Exponential, Rank::Max) => div(
                sub(
                    pool.func("digamma", vec![n1]),
                    pool.func("digamma", vec![one]),
                    pool,
                ),
                p[0],
                pool,
            ),
            (DistKind::Exponential, Rank::Min) => {
                pool.pow(pool.mul(vec![n, p[0]]), pool.integer(-1))
            }
            _ => unreachable!("only the tabled extremes reach the closed-form claims"),
        }
    }

    fn variance_claim(&self, pool: &ExprPool) -> ExprId {
        let p = self.base.params();
        let one = pool.integer(1);
        let two = pool.integer(2);
        let n = self.n;
        if let Some((_, var)) = self.normal_entry(pool) {
            return pool.mul(vec![pool.pow(p[1], two), var]);
        }
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
            (DistKind::Exponential, Rank::Max) => div(
                sub(
                    div(pool.pow(pi(pool), two), pool.integer(6), pool),
                    pool.func("trigamma", vec![pool.add(vec![n, one])]),
                    pool,
                ),
                pool.pow(p[0], two),
                pool,
            ),
            (DistKind::Exponential, Rank::Min) => {
                pool.pow(pool.mul(vec![n, p[0]]), pool.integer(-2))
            }
            _ => unreachable!("only the tabled extremes reach the closed-form claims"),
        }
    }

    /// `E[M]`. For the extremes of a `Uniform` or `Exponential` it is the
    /// closed form of the table above (any `n`, symbolic included); in every
    /// other case it is the expectation of `x * weight(x)` under the base
    /// law, computed by the integrator for a literal `n`. Either way it is
    /// verified against quadrature of `x * f_(k)(x)`.
    ///
    /// # Errors
    ///
    /// [`ProbError::Unverified`] if the gate cannot confirm the answer;
    /// [`ProbError::IntegralDidNotClose`] when the integrator declines (an
    /// `erf` CDF, say); [`ProbError::Unsupported`] for a symbolic `n` outside
    /// the table.
    pub fn mean(&self, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        if !self.tabled(pool) {
            let r = self.general_moment(1, pool)?;
            let value = r.value;
            return Ok(self.finish("prob_order_statistic_mean", value, r.log.0, pool));
        }
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

    /// `Var[M]`; routes as [`OrderStatistic::mean`]. On the general route it
    /// is `E[M^2] - E[M]^2`, both expectations gated.
    ///
    /// # Errors
    ///
    /// As [`OrderStatistic::mean`].
    pub fn variance(&self, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
        super::stash_prob_side_conditions(Vec::new());
        if !self.tabled(pool) {
            let m1 = self.general_moment(1, pool)?;
            let m2 = self.general_moment(2, pool)?;
            let value = simplify_expanded(
                sub(m2.value, pool.mul(vec![m1.value, m1.value]), pool),
                pool,
            )
            .value;
            let mut steps = m1.log.0;
            steps.extend(m2.log.0);
            return Ok(self.finish("prob_order_statistic_variance", value, steps, pool));
        }
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
            (DistKind::Exponential, Rank::Min) => {
                let rate = simplify(pool.mul(vec![self.n, p[0]]), pool).value;
                Distribution::exponential(rate, pool).ok()
            }
            (DistKind::Uniform, kind) if is(p[0], 0) && is(p[1], 1) => {
                let one = pool.integer(1);
                match kind {
                    Rank::Max => Distribution::beta(self.n, one, pool).ok(),
                    Rank::Min => Distribution::beta(one, self.n, pool).ok(),
                    Rank::Kth(k) => {
                        // U_(k:n) ~ Beta(k, n - k + 1).
                        let b = simplify(pool.add(vec![sub(self.n, k, pool), one]), pool).value;
                        Distribution::beta(k, b, pool).ok()
                    }
                }
            }
            _ => None,
        }
    }
}
