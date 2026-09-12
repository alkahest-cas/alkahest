//! Closed forms: means, variances, raw moments, CDFs, quantiles.
//!
//! Every entry here is a **claim**, and every public route out of this file
//! passes its claim through [`super::verify`] before returning it. The table
//! being a table is not an argument that it is right — a transcription slip
//! between `σ` and `σ²` is exactly the defect that survives code review and
//! poisons everything downstream — so nothing is taken on trust.
//!
//! Where a closed form does **not** exist inside this library's primitive set,
//! the answer is [`ProbError::NoClosedForm`] naming the missing function. The
//! `Gamma` CDF is the instructive case: it is `P(k, x/θ)`, the regularised
//! lower incomplete gamma, which alkahest does not have — except when `k` is a
//! positive integer, where the Erlang identity collapses it to a finite
//! elementary sum. So integer `k` returns a value and everything else refuses,
//! rather than the whole family refusing or (much worse) the whole family
//! returning the integer formula.

use rug::Integer;

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::simplify;

use super::dists;
use super::verify;
use super::{div, sub, DistKind, Distribution, ProbError, MAX_MOMENT_ORDER};

/// A fresh symbol to write the defining integrand against.
fn integration_var(dist: &Distribution, extra: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut seen: Vec<ExprId> = dist.params().to_vec();
    seen.extend_from_slice(extra);
    dists::fresh_var(&seen, pool)
}

fn exp(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("exp", vec![arg])
}

fn sq(a: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(a, pool.integer(2))
}

// ---------------------------------------------------------------------------
// Raw moments
// ---------------------------------------------------------------------------

/// `E[Xⁿ]` as an unverified claim, or `None` when this table has no entry.
fn raw_moment_claim(dist: &Distribution, n: u32, pool: &ExprPool) -> Option<ExprId> {
    let p = dist.params();
    let one = pool.integer(1);
    if n == 0 {
        return Some(one);
    }
    Some(match dist.kind() {
        DistKind::Normal => {
            // m₀ = 1, m₁ = μ, mₙ = μ·mₙ₋₁ + (n-1)σ²·mₙ₋₂ — the Gaussian moment
            // recursion, which is exact and needs no case split on parity.
            let (mu, sigma) = (p[0], p[1]);
            let s2 = sq(sigma, pool);
            let mut prev2 = one;
            let mut prev1 = mu;
            for i in 2..=n {
                let next = pool.add(vec![
                    pool.mul(vec![mu, prev1]),
                    pool.mul(vec![pool.integer(i - 1), s2, prev2]),
                ]);
                prev2 = prev1;
                prev1 = next;
            }
            if n == 1 {
                mu
            } else {
                prev1
            }
        }
        DistKind::LogNormal => {
            // E[Xⁿ] = e^{nμ + n²σ²/2}
            let (mu, sigma) = (p[0], p[1]);
            exp(
                pool.add(vec![
                    pool.mul(vec![pool.integer(n), mu]),
                    pool.mul(vec![
                        pool.rational(i64::from(n) * i64::from(n), 2),
                        sq(sigma, pool),
                    ]),
                ]),
                pool,
            )
        }
        DistKind::Uniform => {
            // (b^{n+1} - a^{n+1}) / ((n+1)(b-a))
            let (a, b) = (p[0], p[1]);
            let e = pool.integer(n + 1);
            div(
                sub(pool.pow(b, e), pool.pow(a, e), pool),
                pool.mul(vec![pool.integer(n + 1), sub(b, a, pool)]),
                pool,
            )
        }
        DistKind::Exponential => {
            // n! / λⁿ
            let lambda = p[0];
            div(
                pool.integer(factorial(n)),
                pool.pow(lambda, pool.integer(n)),
                pool,
            )
        }
        DistKind::Gamma => {
            // θⁿ · k(k+1)⋯(k+n-1)
            let (k, theta) = (p[0], p[1]);
            let mut factors = vec![pool.pow(theta, pool.integer(n))];
            for i in 0..n {
                factors.push(pool.add(vec![k, pool.integer(i)]));
            }
            pool.mul(factors)
        }
        DistKind::Beta => {
            // ∏_{i<n} (α+i)/(α+β+i)
            let (a, b) = (p[0], p[1]);
            let ab = pool.add(vec![a, b]);
            let mut num = Vec::new();
            let mut den = Vec::new();
            for i in 0..n {
                num.push(pool.add(vec![a, pool.integer(i)]));
                den.push(pool.add(vec![ab, pool.integer(i)]));
            }
            div(pool.mul(num), pool.mul(den), pool)
        }
        // X ∈ {0,1} ⇒ Xⁿ = X for every n ≥ 1.
        DistKind::Bernoulli => p[0],
        DistKind::Binomial => {
            // Σ_r S(n,r) · N^{(r)} · pʳ, with S the Stirling numbers of the
            // second kind and N^{(r)} the falling factorial: the factorial-
            // moment expansion, exact for every n.
            let (nn, prob) = (dists::count_of(p[0], pool), p[1]);
            let mut terms = Vec::new();
            for r in 0..=n {
                let s = stirling2(n, r);
                if s == 0 {
                    continue;
                }
                let ff = falling_factorial(nn, r);
                if ff == 0 {
                    continue;
                }
                terms.push(pool.mul(vec![pool.integer(s * ff), pool.pow(prob, pool.integer(r))]));
            }
            if terms.is_empty() {
                pool.integer(0)
            } else {
                pool.add(terms)
            }
        }
        DistKind::Poisson => {
            // Touchard: E[Xⁿ] = Σ_r S(n,r) λʳ.
            let lambda = p[0];
            let mut terms = Vec::new();
            for r in 0..=n {
                let s = stirling2(n, r);
                if s == 0 {
                    continue;
                }
                terms.push(pool.mul(vec![pool.integer(s), pool.pow(lambda, pool.integer(r))]));
            }
            pool.add(terms)
        }
    })
}

fn factorial(n: u32) -> Integer {
    let mut acc = Integer::from(1);
    for i in 2..=n {
        acc *= Integer::from(i);
    }
    acc
}

/// `N(N-1)⋯(N-r+1)`, zero once `r > N`.
fn falling_factorial(n: u32, r: u32) -> Integer {
    if r > n {
        return Integer::from(0);
    }
    let mut acc = Integer::from(1);
    for i in 0..r {
        acc *= Integer::from(n - i);
    }
    acc
}

/// Stirling numbers of the second kind, by the standard recurrence.
fn stirling2(n: u32, k: u32) -> Integer {
    if n == 0 && k == 0 {
        return Integer::from(1);
    }
    if k == 0 || k > n {
        return Integer::from(0);
    }
    let n = n as usize;
    let k = k as usize;
    let mut row = vec![Integer::from(0); k + 1];
    row[0] = Integer::from(1);
    for _ in 1..=n {
        let mut next = vec![Integer::from(0); k + 1];
        for j in 1..=k {
            next[j] = Integer::from(j) * row[j].clone() + row[j - 1].clone();
        }
        row = next;
    }
    row[k].clone()
}

/// `E[Xⁿ]` from the table, **without** the numeric gate.
///
/// Only for callers that verify the *assembled* result themselves — the
/// polynomial route in [`super::expect`] combines several of these and checks
/// the combination, so checking each one separately would be redundant work,
/// not extra assurance.
pub(crate) fn raw_moment_unchecked(
    dist: &Distribution,
    n: u32,
    pool: &ExprPool,
) -> Result<ExprId, ProbError> {
    if n > MAX_MOMENT_ORDER {
        return Err(ProbError::Unsupported(format!(
            "moment order {n} exceeds MAX_MOMENT_ORDER = {MAX_MOMENT_ORDER}"
        )));
    }
    raw_moment_claim(dist, n, pool).ok_or_else(|| {
        ProbError::Unsupported(format!("no raw-moment formula for {:?}", dist.kind()))
    })
}

pub(crate) fn moment(
    dist: &Distribution,
    n: u32,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    if n > MAX_MOMENT_ORDER {
        return Err(ProbError::Unsupported(format!(
            "moment order {n} exceeds MAX_MOMENT_ORDER = {MAX_MOMENT_ORDER}; past that the \
             verifying quadrature can no longer distinguish the right value from a wrong one, \
             and an unfailable check is not a check"
        )));
    }
    let claim = raw_moment_claim(dist, n, pool).ok_or_else(|| {
        ProbError::Unsupported(format!("no raw-moment formula for {:?}", dist.kind()))
    })?;
    let claim = simplify(claim, pool).value;

    let x = integration_var(dist, &[], pool);
    let f = pool.pow(x, pool.integer(n));
    let integrand = verify::definition_integrand(f, x, dist, pool);
    let evidence = verify::check(claim, integrand, x, dist, pool)?;

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("prob_raw_moment", x, claim));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    Ok(DerivedExpr::with_log(claim, log))
}

// ---------------------------------------------------------------------------
// Mean and variance
// ---------------------------------------------------------------------------

/// `E[X]`, in the form a reference would print it rather than as `moment(1)`.
pub(crate) fn mean_claim(dist: &Distribution, pool: &ExprPool) -> ExprId {
    let p = dist.params();
    let two = pool.integer(2);
    match dist.kind() {
        DistKind::Normal => p[0],
        DistKind::LogNormal => exp(pool.add(vec![p[0], div(sq(p[1], pool), two, pool)]), pool),
        DistKind::Uniform => div(pool.add(vec![p[0], p[1]]), two, pool),
        DistKind::Exponential => pool.pow(p[0], pool.integer(-1)),
        DistKind::Gamma => pool.mul(vec![p[0], p[1]]),
        DistKind::Beta => div(p[0], pool.add(vec![p[0], p[1]]), pool),
        DistKind::Bernoulli => p[0],
        DistKind::Binomial => pool.mul(vec![p[0], p[1]]),
        DistKind::Poisson => p[0],
    }
}

fn variance_claim(dist: &Distribution, pool: &ExprPool) -> ExprId {
    let p = dist.params();
    let one = pool.integer(1);
    let two = pool.integer(2);
    match dist.kind() {
        DistKind::Normal => sq(p[1], pool),
        DistKind::LogNormal => {
            // (e^{σ²} - 1)·e^{2μ+σ²}
            let s2 = sq(p[1], pool);
            pool.mul(vec![
                sub(exp(s2, pool), one, pool),
                exp(pool.add(vec![pool.mul(vec![two, p[0]]), s2]), pool),
            ])
        }
        DistKind::Uniform => div(sq(sub(p[1], p[0], pool), pool), pool.integer(12), pool),
        DistKind::Exponential => pool.pow(p[0], pool.integer(-2)),
        DistKind::Gamma => pool.mul(vec![p[0], sq(p[1], pool)]),
        DistKind::Beta => {
            // αβ / ((α+β)²(α+β+1))
            let ab = pool.add(vec![p[0], p[1]]);
            div(
                pool.mul(vec![p[0], p[1]]),
                pool.mul(vec![sq(ab, pool), pool.add(vec![ab, one])]),
                pool,
            )
        }
        DistKind::Bernoulli => pool.mul(vec![p[0], sub(one, p[0], pool)]),
        DistKind::Binomial => pool.mul(vec![p[0], p[1], sub(one, p[1], pool)]),
        DistKind::Poisson => p[0],
    }
}

pub(crate) fn mean(dist: &Distribution, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
    let claim = simplify(mean_claim(dist, pool), pool).value;
    let x = integration_var(dist, &[], pool);
    let integrand = verify::definition_integrand(x, x, dist, pool);
    let evidence = verify::check(claim, integrand, x, dist, pool)?;
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("prob_mean", x, claim));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    Ok(DerivedExpr::with_log(claim, log))
}

pub(crate) fn variance(
    dist: &Distribution,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    let claim = simplify(variance_claim(dist, pool), pool).value;
    let mu = simplify(mean_claim(dist, pool), pool).value;
    let x = integration_var(dist, &[], pool);
    // The *defining* integral, `∫(x - E[X])² p(x) dx`, deliberately written in
    // terms of the mean expression rather than as `E[X²] - E[X]²`: it makes
    // the check fail when either the mean or the variance entry is wrong,
    // rather than letting two consistent mistakes cancel.
    let centred = sq(sub(x, mu, pool), pool);
    let integrand = verify::definition_integrand(centred, x, dist, pool);
    let evidence = verify::check(claim, integrand, x, dist, pool)?;
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("prob_variance", x, claim));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    Ok(DerivedExpr::with_log(claim, log))
}

// ---------------------------------------------------------------------------
// CDF
// ---------------------------------------------------------------------------

/// `Φ(t) = ½(1 + erf(t/√2))`.
fn std_normal_cdf(t: ExprId, pool: &ExprPool) -> ExprId {
    let root2 = pool.func("sqrt", vec![pool.integer(2)]);
    let erf = pool.func("erf", vec![div(t, root2, pool)]);
    pool.mul(vec![
        pool.rational(1, 2),
        pool.add(vec![pool.integer(1), erf]),
    ])
}

/// A literal positive integer, or `None`.
fn positive_integer(e: ExprId, pool: &ExprPool) -> Option<u32> {
    match pool.get(e) {
        ExprData::Integer(n) if n.0 > 0 => n.0.to_u32(),
        _ => None,
    }
}

fn cdf_claim(dist: &Distribution, x: ExprId, pool: &ExprPool) -> Result<ExprId, ProbError> {
    let p = dist.params();
    let one = pool.integer(1);
    let neg1 = pool.integer(-1);
    Ok(match dist.kind() {
        DistKind::Normal => std_normal_cdf(div(sub(x, p[0], pool), p[1], pool), pool),
        DistKind::LogNormal => std_normal_cdf(
            div(sub(pool.func("log", vec![x]), p[0], pool), p[1], pool),
            pool,
        ),
        DistKind::Uniform => div(sub(x, p[0], pool), sub(p[1], p[0], pool), pool),
        DistKind::Exponential => sub(one, exp(pool.mul(vec![neg1, p[0], x]), pool), pool),
        DistKind::Gamma => {
            // Erlang only: with k a positive integer,
            //   F(x) = 1 - e^{-x/θ} Σ_{j<k} (x/θ)ʲ/j!
            // For any other k this is `P(k, x/θ)`, and alkahest has no
            // incomplete gamma to write it with.
            let Some(k) = positive_integer(p[0], pool) else {
                return Err(ProbError::NoClosedForm {
                    quantity: "the Gamma CDF with a non-integer shape",
                    missing: "the regularised lower incomplete gamma P(k, x)",
                });
            };
            let u = div(x, p[1], pool);
            let mut terms = Vec::new();
            for j in 0..k {
                terms.push(div(
                    pool.pow(u, pool.integer(j)),
                    pool.integer(factorial(j)),
                    pool,
                ));
            }
            sub(
                one,
                pool.mul(vec![exp(pool.mul(vec![neg1, u]), pool), pool.add(terms)]),
                pool,
            )
        }
        DistKind::Beta => {
            // Integer α, β only: I_x(α,β) is then the finite binomial tail
            //   Σ_{j=α}^{α+β-1} C(α+β-1, j) xʲ(1-x)^{α+β-1-j}.
            let (Some(a), Some(b)) = (positive_integer(p[0], pool), positive_integer(p[1], pool))
            else {
                return Err(ProbError::NoClosedForm {
                    quantity: "the Beta CDF with a non-integer parameter",
                    missing: "the regularised incomplete beta I_x(α, β)",
                });
            };
            let m = a + b - 1;
            let q = sub(one, x, pool);
            let mut terms = Vec::new();
            for j in a..=m {
                terms.push(pool.mul(vec![
                    pool.integer(Integer::from(m).binomial(j)),
                    pool.pow(x, pool.integer(j)),
                    pool.pow(q, pool.integer(m - j)),
                ]));
            }
            pool.add(terms)
        }
        DistKind::Bernoulli | DistKind::Binomial | DistKind::Poisson => {
            return Err(ProbError::NoClosedForm {
                quantity: "a discrete CDF",
                missing: "a step function over the integers, which this module does not model \
                          symbolically",
            })
        }
    })
}

/// Where `arg` sits relative to the support, when that can be decided.
///
/// A CDF is the one quantity in this module whose value *outside* the support
/// is not a matter of convention: `P(X ≤ x)` is `0` below it and `1` above it,
/// full stop. The closed forms in [`cdf_claim`] are the in-support branch only,
/// and evaluating one off-support does not fail — it produces a clean, wrong
/// number (`Gamma(3, 4/5)` at `x = -5` gives `-7396.87`, offered as a
/// probability). So a decidable argument is answered exactly here rather than
/// being pushed through a formula that does not apply to it.
enum Placement {
    Below,
    Above,
    Inside,
    /// Symbolic or otherwise undecidable — the in-support branch is returned
    /// with the restriction recorded as a side condition.
    Unknown,
}

fn placement(dist: &Distribution, arg: ExprId, pool: &ExprPool) -> Placement {
    let (lo, hi) = match dist.support(pool) {
        crate::prob::Support::Real => return Placement::Inside,
        crate::prob::Support::Positive => (Some(pool.integer(0)), None),
        crate::prob::Support::Interval(lo, hi) => (Some(lo), Some(hi)),
        // Discrete supports never reach here: `cdf_claim` refuses them first.
        _ => return Placement::Unknown,
    };
    let mut decided_inside = true;
    if let Some(lo) = lo {
        match dists::sign_of(sub(arg, lo, pool), pool) {
            dists::Sign::Negative => return Placement::Below,
            dists::Sign::Positive | dists::Sign::Zero => {}
            dists::Sign::Unknown => decided_inside = false,
        }
    }
    if let Some(hi) = hi {
        match dists::sign_of(sub(arg, hi, pool), pool) {
            dists::Sign::Positive | dists::Sign::Zero => return Placement::Above,
            dists::Sign::Negative => {}
            dists::Sign::Unknown => decided_inside = false,
        }
    }
    if decided_inside {
        Placement::Inside
    } else {
        Placement::Unknown
    }
}

pub(crate) fn cdf(
    dist: &Distribution,
    x: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    // Refuse the laws that have no closed form *before* anything else, so a
    // `Gamma` with a symbolic shape still reports E-PROB-004 rather than being
    // short-circuited by an out-of-support argument.
    let _ = cdf_claim(dist, x, pool)?;

    let placement = placement(dist, x, pool);
    if let Placement::Below | Placement::Above = placement {
        let value = pool.integer(match placement {
            Placement::Below => 0,
            _ => 1,
        });
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple("prob_cdf_outside_support", x, value));
        return Ok(DerivedExpr::with_log(value, log));
    }

    // Verify the CDF as a *function*: `check_cdf` reads levels and increments
    // off the claim at several abscissae, which it can only do when the
    // argument is a symbol it can substitute for. Building against a fresh one
    // and substituting afterwards is what lets `cdf(12/5)` be checked at all —
    // against the claim as a whole, over a range, rather than at the single
    // point the caller asked about.
    let z = integration_var(dist, &[x], pool);
    let checked = simplify(cdf_claim(dist, z, pool)?, pool).value;
    let evidence = verify::check_cdf(checked, z, dist, pool)?;

    let mut m = std::collections::HashMap::new();
    m.insert(z, x);
    let claim = simplify(crate::kernel::subs(checked, &m, pool), pool).value;

    let mut log = DerivationLog::new();
    if matches!(placement, Placement::Unknown) {
        // The closed form is the in-support branch. Where the argument could
        // not be placed, say so rather than letting the caller read a number
        // that is only a probability when the hypothesis holds.
        let mut conditions = Vec::new();
        match dist.support(pool) {
            crate::prob::Support::Positive => {
                conditions.push(SideCondition::Positive(x));
            }
            crate::prob::Support::Interval(lo, hi) => {
                conditions.push(SideCondition::Positive(simplify(sub(x, lo, pool), pool).value));
                conditions.push(SideCondition::Positive(simplify(sub(hi, x, pool), pool).value));
            }
            _ => {}
        }
        if !conditions.is_empty() {
            log.push(RewriteStep::with_conditions(
                "prob_cdf_argument_inside_support",
                x,
                claim,
                conditions,
            ));
        }
    }
    log.push(RewriteStep::simple("prob_cdf", x, claim));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    Ok(DerivedExpr::with_log(claim, log))
}

// ---------------------------------------------------------------------------
// Quantile
// ---------------------------------------------------------------------------

fn quantile_claim(
    dist: &Distribution,
    p_arg: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, ProbError> {
    let p = dist.params();
    let one = pool.integer(1);
    match dist.kind() {
        DistKind::Uniform => Ok(pool.add(vec![p[0], pool.mul(vec![p_arg, sub(p[1], p[0], pool)])])),
        DistKind::Exponential => Ok(div(
            pool.mul(vec![
                pool.integer(-1),
                pool.func("log", vec![sub(one, p_arg, pool)]),
            ]),
            p[0],
            pool,
        )),
        DistKind::Normal | DistKind::LogNormal => Err(ProbError::NoClosedForm {
            quantity: "the Normal/LogNormal quantile",
            missing: "the inverse error function erf⁻¹",
        }),
        DistKind::Gamma | DistKind::Beta => Err(ProbError::NoClosedForm {
            quantity: "the Gamma/Beta quantile",
            missing: "the inverse of an incomplete gamma / incomplete beta, which has no \
                      elementary closed form at all",
        }),
        DistKind::Bernoulli | DistKind::Binomial | DistKind::Poisson => {
            Err(ProbError::NoClosedForm {
                quantity: "a discrete quantile",
                missing: "a generalised inverse of a step function, which this module does not \
                          model symbolically",
            })
        }
    }
}

pub(crate) fn quantile(
    dist: &Distribution,
    p_arg: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    // `F⁻¹` is only defined on `[0, 1]`, and off it the closed forms produce a
    // number rather than failing: `Uniform(-2, 3).quantile(2)` evaluates to
    // `8`, outside the support it is supposed to be a point of. A probability
    // that can be decided is therefore checked, exactly as a distribution
    // parameter is.
    dists::require_probability(p_arg, "p", pool)?;

    let claim = simplify(quantile_claim(dist, p_arg, pool)?, pool).value;

    // As in `cdf`: verify the quantile as a function of a fresh symbol, over a
    // ladder of probabilities, then substitute. A literal `p` leaves
    // `check_quantile` with nothing to vary and no way to conclude.
    let z = integration_var(dist, &[p_arg], pool);
    let checked = simplify(quantile_claim(dist, z, pool)?, pool).value;
    let evidence = verify::check_quantile(checked, z, dist, pool)?;

    let mut log = DerivationLog::new();
    if matches!(dists::sign_of(p_arg, pool), dists::Sign::Unknown) {
        // A symbolic `p` cannot be placed in `[0, 1]`; carry the requirement
        // rather than assuming it.
        let one = pool.integer(1);
        log.push(RewriteStep::with_conditions(
            "prob_quantile_argument_is_a_probability",
            p_arg,
            claim,
            vec![
                SideCondition::Positive(p_arg),
                SideCondition::Positive(simplify(sub(one, p_arg, pool), pool).value),
            ],
        ));
    }
    log.push(RewriteStep::simple("prob_quantile", p_arg, claim));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    Ok(DerivedExpr::with_log(claim, log))
}
