//! The generating-function family: `M_X`, `G_X`, `K_X`, the cumulants, the
//! factorial moments, and the two shape statistics that fall out of them.
//!
//! # Existence is the content
//!
//! A table of generating functions is a page of a reference book, and a CAS
//! that returns one is worth nothing over the page. What is worth something is
//! the half of the statement the page prints in small type and the CAS drops:
//!
//! | ask | what a table-lookup returns | what is true |
//! |---|---|---|
//! | `M_X(t)` for `LogNormal(μ, σ)` | completing the square gives a clean `e^{…}` | `E[e^{tX}] = ∞` for **every** `t > 0`. The integral `∫e^{tx}e^{-(log x - μ)²/2σ²}dx/(xσ√2π)` diverges — `e^{tx}` beats a Gaussian in `log x`. There is no MGF |
//! | `M_X(t)` for `Exponential(λ)` | `λ/(λ - t)` | `λ/(λ - t)` **only for `t < λ`**. Outside the strip it is still a perfectly nice rational function, still finite, still plausible, and still not `E[e^{tX}]`, which is `+∞` there |
//! | `M_X(t)` for `Gamma(k, θ)` | `(1 - θt)^{-k}` | the same, with the strip `t < 1/θ` |
//! | `G_X(z)` for `Normal(μ, σ)` | `E[z^X] = E[e^{X log z}] = e^{μ log z + σ²log²z/2}` | nothing. `G_X(z) = Σ_k z^k P(X = k)` is a statement about a law on the non-negative integers; a normal puts zero mass on them and the sum is `0`. The formal manipulation answers a different question |
//! | `κ_n` for `LogNormal` | the moment–cumulant recursion runs fine | `κ_n = K^{(n)}(0)` and `K = log M` exists at no `t > 0`, so there is no function to differentiate. The recursion is computing the coefficients of a divergent series |
//!
//! The convergence strips are not in a docstring. `Exponential` and `Gamma`
//! report theirs through [`super::take_prob_side_conditions`] — the same
//! out-of-band channel `cdf` uses for a symbolic argument — and a `t` that can
//! be *decided* to sit outside the strip is [`ProbError::Divergent`] rather
//! than a formula. The laws whose MGF converges on the whole line say so in the
//! derivation log instead, so "no side conditions" means "checked and
//! unconditional", not "nobody looked".
//!
//! # What refuses
//!
//! | route | law | refusal |
//! |---|---|---|
//! | `M_X`, `K_X` | `LogNormal` | `E-PROB-006` for `t > 0` or an undecidable `t` — the integral diverges. `E-PROB-004` for a `t` decidably `≤ 0`, where `E[e^{tX}]` is finite but has no closed form |
//! | `M_X`, `K_X` | `Beta` | `E-PROB-004`: `M = ₁F₁(α; α+β; t)`, which this library does not have. The *cumulants* of a Beta are still returned — `K` is analytic at `0`, it just has no name here |
//! | `G_X` | every continuous law | `E-PROB-002`, naming the support. A category error, refused rather than formally manipulated |
//! | `κ_n` | `LogNormal` | `E-PROB-006`. Use [`skewness`] and [`excess_kurtosis`], which are defined from **central moments** and do exist for a log-normal |
//! | `κ_n` | `n > `[`MAX_CUMULANT_ORDER`] | `E-PROB-002` — see that constant |
//! | factorial moments | every continuous law | `E-PROB-002`: `E[X(X-1)⋯]` is `G^{(n)}(1)`, and there is no `G` |
//!
//! # How each claim is checked
//!
//! * `M_X`, `G_X`, `K_X` — [`super::verify::check_generating`], against
//!   `∫e^{tx}p(x)dx` / `Σz^k P(X=k)` at arguments drawn *inside the strip at
//!   each parameter point*, since the strip moves with `λ` and `θ`. `K` is
//!   checked as `e^{K(t)}` against the same integral, so the `log` is exercised
//!   rather than cancelled.
//! * `κ_n` — [`super::verify::check_cumulant`], against the expression of `κ_n`
//!   in the **central** moments, each of which is quadratured from its own
//!   defining integral. Deliberately not the same recursion over quadratured
//!   raw moments: that check agrees with a wrong recursion.
//! * factorial moments, [`skewness`], [`excess_kurtosis`] — [`super::verify::check`]
//!   against the defining expectation, `E[X^{(n)}]` and `E[((X-μ)/σ)^k]`.

use rug::Integer;

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::simplify;

use super::dists::{self, Sign};
use super::moments;
use super::verify::{self, CentralTerm};
use super::{div, sub, DistKind, Distribution, ProbError, MAX_MOMENT_ORDER};

/// Highest cumulant order [`cumulant`] will return.
///
/// A verification limit, in the same spirit as [`super::MAX_MOMENT_ORDER`] and
/// for a sharper reason. `κ_n` is assembled by the moment–cumulant recursion,
/// and the gate compares it against the standard expression of `κ_n` in the
/// central moments — a *different* identity on *quadratured* data, which is
/// what makes it able to catch an off-by-one in the recursion. That table of
/// identities is written out to `n = 6` in this module's private verifier;
/// extending the recursion past there without extending the table would leave
/// the extra orders checked by nothing, and an unchecked cumulant is not a
/// thing this module returns.
pub const MAX_CUMULANT_ORDER: u32 = 6;

fn exp(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("exp", vec![arg])
}

fn log(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("log", vec![arg])
}

// ---------------------------------------------------------------------------
// Where a moment generating function converges
// ---------------------------------------------------------------------------

/// The half-line an MGF converges on, as `gap > 0`, or `None` when it
/// converges on the whole line.
///
/// `Exponential(λ)` needs `λ - t > 0` and `Gamma(k, θ)` needs `1 - θt > 0`.
/// Everything else in this module's table is either supported on a bounded set
/// (`Uniform`, `Beta`, `Bernoulli`, `Binomial`) or has Gaussian/Poisson tails,
/// and its MGF is entire.
fn mgf_strip(dist: &Distribution, t: ExprId, pool: &ExprPool) -> Option<(ExprId, &'static str)> {
    let p = dist.params();
    match dist.kind() {
        DistKind::Exponential => Some((sub(p[0], t, pool), "t < lambda")),
        DistKind::Gamma => Some((
            sub(pool.integer(1), pool.mul(vec![p[1], t]), pool),
            "t < 1/theta",
        )),
        _ => None,
    }
}

/// Record the convergence region on `log`, or refuse when the argument can be
/// decided to sit outside it.
///
/// Three outcomes, and the third is the one that matters: an argument that
/// *cannot* be placed leaves with the strip attached as a
/// [`SideCondition::Positive`], so a caller reading
/// [`super::take_prob_side_conditions`] can tell a theorem from a conditional.
/// Assuming the convenient branch is what produces `λ/(λ - t)` for `t > λ`.
fn record_strip(
    log: &mut DerivationLog,
    claim: ExprId,
    dist: &Distribution,
    t: ExprId,
    what: &str,
    pool: &ExprPool,
) -> Result<(), ProbError> {
    let Some((gap, text)) = mgf_strip(dist, t, pool) else {
        log.push(RewriteStep::simple(
            "prob_mgf_converges_for_every_real_t",
            claim,
            claim,
        ));
        return Ok(());
    };
    let gap = simplify(gap, pool).value;
    match dists::sign_of(gap, pool) {
        Sign::Positive => log.push(RewriteStep::simple(
            "prob_mgf_argument_inside_the_convergence_strip",
            claim,
            claim,
        )),
        Sign::Negative | Sign::Zero => {
            return Err(ProbError::Divergent(format!(
                "the {what} of a {law} converges only for {text}, and the argument given is \
                 not: E[e^{{tX}}] = +∞ there. The closed form is still a finite, plausible \
                 number outside the strip, which is why it is refused rather than returned",
                law = dist.kind().name(),
            )))
        }
        Sign::Unknown => log.push(RewriteStep::with_conditions(
            "prob_mgf_argument_inside_the_convergence_strip",
            claim,
            claim,
            vec![SideCondition::Positive(gap)],
        )),
    }
    Ok(())
}

/// Arguments the gate checks an MGF at, **as expressions in the parameters**.
///
/// `λ/2` rather than `0.5`: the strip is `t < λ` and `λ` moves from one
/// parameter point to the next, so a constant ladder would step outside it and
/// report a correct closed form as divergent. See
/// [`super::verify::check_generating`].
fn mgf_ladder(dist: &Distribution, pool: &ExprPool) -> Vec<ExprId> {
    let p = dist.params();
    let half = pool.rational(1, 2);
    let neg1 = pool.integer(-1);
    match dist.kind() {
        DistKind::Exponential => vec![
            pool.mul(vec![half, p[0]]),
            pool.mul(vec![neg1, p[0]]),
            pool.mul(vec![pool.rational(9, 10), p[0]]),
        ],
        DistKind::Gamma => {
            let inv_theta = pool.pow(p[1], neg1);
            vec![
                pool.mul(vec![half, inv_theta]),
                pool.mul(vec![neg1, inv_theta]),
                pool.mul(vec![pool.rational(9, 10), inv_theta]),
            ]
        }
        _ => vec![half, pool.rational(-3, 4)],
    }
}

// ---------------------------------------------------------------------------
// Moment generating function
// ---------------------------------------------------------------------------

/// The MGF table, or the refusal that stands in its place.
fn mgf_claim(dist: &Distribution, t: ExprId, pool: &ExprPool) -> Result<ExprId, ProbError> {
    let p = dist.params();
    let one = pool.integer(1);
    let two = pool.integer(2);
    let neg1 = pool.integer(-1);
    Ok(match dist.kind() {
        // e^{μt + σ²t²/2}
        DistKind::Normal => exp(
            pool.add(vec![
                pool.mul(vec![p[0], t]),
                pool.mul(vec![
                    pool.rational(1, 2),
                    pool.pow(p[1], two),
                    pool.pow(t, two),
                ]),
            ]),
            pool,
        ),
        // (e^{bt} - e^{at}) / (t(b-a))
        DistKind::Uniform => div(
            sub(
                exp(pool.mul(vec![p[1], t]), pool),
                exp(pool.mul(vec![p[0], t]), pool),
                pool,
            ),
            pool.mul(vec![t, sub(p[1], p[0], pool)]),
            pool,
        ),
        // λ/(λ - t), for t < λ
        DistKind::Exponential => div(p[0], sub(p[0], t, pool), pool),
        // (1 - θt)^{-k}, for t < 1/θ
        DistKind::Gamma => pool.pow(
            sub(one, pool.mul(vec![p[1], t]), pool),
            pool.mul(vec![neg1, p[0]]),
        ),
        // 1 - p + p e^t
        DistKind::Bernoulli => pool.add(vec![
            sub(one, p[0], pool),
            pool.mul(vec![p[0], exp(t, pool)]),
        ]),
        // (1 - p + p e^t)^n
        DistKind::Binomial => pool.pow(
            pool.add(vec![
                sub(one, p[1], pool),
                pool.mul(vec![p[1], exp(t, pool)]),
            ]),
            p[0],
        ),
        // e^{λ(e^t - 1)}
        DistKind::Poisson => exp(pool.mul(vec![p[0], sub(exp(t, pool), one, pool)]), pool),
        DistKind::LogNormal => {
            return Err(lognormal_refusal(
                t,
                "moment generating function",
                "the log-normal moment generating function at a non-positive argument",
                pool,
            ))
        }
        DistKind::Beta => {
            return Err(ProbError::NoClosedForm {
                quantity: "the Beta moment generating function",
                missing: "the confluent hypergeometric function ₁F₁(α; α+β; t)",
            })
        }
    })
}

/// Why a log-normal has no MGF, told accurately for the `t` that was asked
/// about.
///
/// `E[e^{tX}]` is finite for `t ≤ 0` — it just has no closed form — and `+∞`
/// for every `t > 0`. Reporting the second reason for the first case would be
/// a lie in the direction that happens to be convenient; reporting the first
/// for the second would suggest the value merely needs a better integrator.
fn lognormal_refusal(
    t: ExprId,
    what: &'static str,
    quantity: &'static str,
    pool: &ExprPool,
) -> ProbError {
    match dists::sign_of(t, pool) {
        Sign::Negative | Sign::Zero => ProbError::NoClosedForm {
            quantity,
            missing: "nothing this library could add — E[e^{tX}] is finite for t ≤ 0 but is \
                      not expressible in elementary or standard special functions",
        },
        _ => ProbError::Divergent(format!(
            "a log-normal has no {what}: E[e^{{tX}}] = ∫e^{{tx}}p(x)dx diverges for every \
             t > 0, because e^{{tx}} outgrows a density that is Gaussian in log x. Completing \
             the square anyway yields a clean closed form that is the value of no integral. \
             (The log-normal is also the standard example of a law not determined by its \
             moments — Heyde 1963.)"
        )),
    }
}

/// `M_X(t) = E[e^{tX}]` in closed form, verified, with its region of
/// convergence reported rather than assumed.
///
/// # Errors
///
/// [`ProbError::Divergent`] for `LogNormal` at a positive or undecidable `t`,
/// and for an `Exponential`/`Gamma` argument decidably outside the strip.
/// [`ProbError::NoClosedForm`] for `Beta`. [`ProbError::Unverified`] if the
/// numeric gate could not confirm the table entry.
pub fn moment_generating_function(
    dist: &Distribution,
    t: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    let claim = simplify(mgf_claim(dist, t, pool)?, pool).value;

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "prob_moment_generating_function",
        t,
        claim,
    ));
    // Placed **before** the gate, not after. Outside the strip there is no
    // integral to quadrature: the run comes back divergent, or — at `t = λ`
    // exactly — with an unevaluable `λ/0`, and the caller is told the closed
    // form could not be *confirmed* when the truth is that the quantity does
    // not exist. E-PROB-006 and E-PROB-005 are different statements and the
    // order of these two lines is what keeps them apart.
    record_strip(&mut log, claim, dist, t, "moment generating function", pool)?;

    let x = dists::fresh_var(&[claim, t], pool);
    let integrand = verify::definition_integrand(exp(pool.mul(vec![t, x]), pool), x, dist, pool);
    let ladder = mgf_ladder(dist, pool);
    let evidence = verify::check_generating(claim, integrand, x, t, &ladder, dist, pool)?;
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log));
    Ok(DerivedExpr::with_log(claim, log))
}

// ---------------------------------------------------------------------------
// Cumulant generating function
// ---------------------------------------------------------------------------

/// `K_X(t) = log M_X(t)`, written the way a reference writes it rather than as
/// a `log` wrapped round [`mgf_claim`].
///
/// `-k log(1 - θt)` is the Gamma CGF; `log((1 - θt)^{-k})` is the same number
/// and a worse answer, and every downstream `diff` has to undo the wrapper
/// before it can do anything. The gate checks `e^{K}` against `∫e^{tx}p(x)dx`,
/// so writing the table out twice is checked rather than trusted.
fn cgf_claim(dist: &Distribution, t: ExprId, pool: &ExprPool) -> Result<ExprId, ProbError> {
    let p = dist.params();
    let one = pool.integer(1);
    let two = pool.integer(2);
    let neg1 = pool.integer(-1);
    Ok(match dist.kind() {
        // μt + σ²t²/2
        DistKind::Normal => pool.add(vec![
            pool.mul(vec![p[0], t]),
            pool.mul(vec![
                pool.rational(1, 2),
                pool.pow(p[1], two),
                pool.pow(t, two),
            ]),
        ]),
        // log((e^{bt} - e^{at}) / (t(b-a)))
        DistKind::Uniform => log(
            div(
                sub(
                    exp(pool.mul(vec![p[1], t]), pool),
                    exp(pool.mul(vec![p[0], t]), pool),
                    pool,
                ),
                pool.mul(vec![t, sub(p[1], p[0], pool)]),
                pool,
            ),
            pool,
        ),
        // log(λ/(λ - t)), for t < λ
        DistKind::Exponential => log(div(p[0], sub(p[0], t, pool), pool), pool),
        // -k log(1 - θt), for t < 1/θ
        DistKind::Gamma => pool.mul(vec![
            neg1,
            p[0],
            log(sub(one, pool.mul(vec![p[1], t]), pool), pool),
        ]),
        // log(1 - p + p e^t)
        DistKind::Bernoulli => log(
            pool.add(vec![
                sub(one, p[0], pool),
                pool.mul(vec![p[0], exp(t, pool)]),
            ]),
            pool,
        ),
        // n log(1 - p + p e^t)
        DistKind::Binomial => pool.mul(vec![
            p[0],
            log(
                pool.add(vec![
                    sub(one, p[1], pool),
                    pool.mul(vec![p[1], exp(t, pool)]),
                ]),
                pool,
            ),
        ]),
        // λ(e^t - 1)
        DistKind::Poisson => pool.mul(vec![p[0], sub(exp(t, pool), one, pool)]),
        DistKind::LogNormal => {
            return Err(lognormal_refusal(
                t,
                "cumulant generating function",
                "the log-normal cumulant generating function at a non-positive argument",
                pool,
            ))
        }
        DistKind::Beta => {
            return Err(ProbError::NoClosedForm {
                quantity: "the Beta cumulant generating function",
                missing: "the confluent hypergeometric function ₁F₁(α; α+β; t), of which K is \
                          the logarithm. The Beta *cumulants* do exist and are returned by \
                          `cumulant` — K is analytic at 0, it simply has no name here",
            })
        }
    })
}

/// `K_X(t) = log M_X(t)` in closed form, verified as `e^{K(t)} = E[e^{tX}]`.
///
/// Carries the same region of convergence as [`moment_generating_function`],
/// through the same channel, for the same reason: `log` of a divergent
/// expectation is not a cumulant generating function.
///
/// # Errors
///
/// As [`moment_generating_function`], except that `Beta` is refused here while
/// its [`cumulant`]s are not.
pub fn cumulant_generating_function(
    dist: &Distribution,
    t: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    let claim = simplify(cgf_claim(dist, t, pool)?, pool).value;

    let mut log_ = DerivationLog::new();
    log_.push(RewriteStep::simple(
        "prob_cumulant_generating_function",
        t,
        claim,
    ));
    // Before the gate — see `moment_generating_function`.
    record_strip(
        &mut log_,
        claim,
        dist,
        t,
        "cumulant generating function",
        pool,
    )?;

    let x = dists::fresh_var(&[claim, t], pool);
    let integrand = verify::definition_integrand(exp(pool.mul(vec![t, x]), pool), x, dist, pool);
    let ladder = mgf_ladder(dist, pool);
    // `e^{K}`, *not* simplified: the point is to push the claim's own `log`
    // through the numeric evaluator. A simplifier that cancelled `exp∘log`
    // would leave the `log` in the returned answer checked by nothing.
    let exponentiated = exp(claim, pool);
    let evidence = verify::check_generating(exponentiated, integrand, x, t, &ladder, dist, pool)?;
    log_.push(RewriteStep::simple(
        "prob_cgf_exponentiated_matches_the_moment_generating_integral",
        claim,
        claim,
    ));
    log_.push(verify::evidence_step(&evidence, claim, dist, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log_));
    Ok(DerivedExpr::with_log(claim, log_))
}

// ---------------------------------------------------------------------------
// Probability generating function
// ---------------------------------------------------------------------------

/// `G_X(z) = E[z^X] = Σ_k z^k P(X = k)` is a statement about a law on
/// `{0, 1, 2, …}`. Anything else is refused here rather than pushed through
/// `E[e^{X log z}]`, which answers a different question and returns a number.
fn require_integer_support(dist: &Distribution, pool: &ExprPool) -> Result<(), ProbError> {
    if dist.support(pool).is_discrete() {
        return Ok(());
    }
    Err(ProbError::Unsupported(format!(
        "the probability generating function G_X(z) = E[z^X] = Σ_k z^k P(X = k) is defined \
         for a law supported on the non-negative integers; {} is not one of them. The formal \
         rewrite E[z^X] = E[e^{{X log z}}] does produce a closed form, but it is the moment \
         generating function at log z and says nothing about P(X = k) — which is zero for \
         every k here. Ask for `moment_generating_function` if that is what was wanted",
        dist.kind().name()
    )))
}

fn pgf_claim(dist: &Distribution, z: ExprId, pool: &ExprPool) -> ExprId {
    let p = dist.params();
    let one = pool.integer(1);
    match dist.kind() {
        // 1 - p + pz
        DistKind::Bernoulli => pool.add(vec![sub(one, p[0], pool), pool.mul(vec![p[0], z])]),
        // (1 - p + pz)^n
        DistKind::Binomial => pool.pow(
            pool.add(vec![sub(one, p[1], pool), pool.mul(vec![p[1], z])]),
            p[0],
        ),
        // e^{λ(z - 1)}
        DistKind::Poisson => exp(pool.mul(vec![p[0], sub(z, one, pool)]), pool),
        _ => unreachable!("require_integer_support admits only the three integer laws"),
    }
}

/// Where `G_X` converges, as a log-entry name. All three integer laws here
/// converge on the whole plane — two because `G` is a polynomial, one because
/// a Poisson `G` is entire — so there is no side condition to raise, and the
/// derivation log says which of those it is rather than staying silent.
fn pgf_radius(dist: &Distribution) -> &'static str {
    match dist.kind() {
        DistKind::Bernoulli | DistKind::Binomial => {
            "prob_pgf_is_a_polynomial_so_converges_for_all_z"
        }
        _ => "prob_pgf_is_entire_so_converges_for_all_z",
    }
}

/// `G_X(z) = E[z^X]`, in closed form, verified against `Σ_k z^k P(X = k)`.
///
/// # Errors
///
/// [`ProbError::Unsupported`] for every law not supported on the non-negative
/// integers — a category error, refused by name rather than answered by formal
/// manipulation.
pub fn probability_generating_function(
    dist: &Distribution,
    z: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    require_integer_support(dist, pool)?;
    let claim = simplify(pgf_claim(dist, z, pool), pool).value;

    let k = dists::fresh_var(&[claim, z], pool);
    let integrand = verify::definition_integrand(pool.pow(z, k), k, dist, pool);
    let ladder = vec![pool.rational(1, 2), pool.rational(5, 4)];
    let evidence = verify::check_generating(claim, integrand, k, z, &ladder, dist, pool)?;

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "prob_probability_generating_function",
        z,
        claim,
    ));
    log.push(RewriteStep::simple(pgf_radius(dist), claim, claim));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log));
    Ok(DerivedExpr::with_log(claim, log))
}

// ---------------------------------------------------------------------------
// Factorial moments
// ---------------------------------------------------------------------------

fn factorial_moment_claim(dist: &Distribution, n: u32, pool: &ExprPool) -> ExprId {
    let p = dist.params();
    if n == 0 {
        return pool.integer(1);
    }
    match dist.kind() {
        // X ∈ {0, 1}, so X(X-1) = 0 and every order past the first vanishes.
        DistKind::Bernoulli => {
            if n == 1 {
                p[0]
            } else {
                pool.integer(0)
            }
        }
        // N^{(n)} pⁿ
        DistKind::Binomial => {
            let nn = dists::count_of(p[0], pool);
            pool.mul(vec![
                pool.integer(moments::falling_factorial(nn, n)),
                pool.pow(p[1], pool.integer(n)),
            ])
        }
        // λⁿ — the identity that makes the Poisson the distribution it is.
        DistKind::Poisson => pool.pow(p[0], pool.integer(n)),
        _ => unreachable!("require_integer_support admits only the three integer laws"),
    }
}

/// `E[X(X-1)⋯(X-n+1)] = G_X^{(n)}(1)`, in closed form, verified against the
/// defining sum.
///
/// # Errors
///
/// [`ProbError::Unsupported`] for a continuous law — the falling factorial of a
/// real variate is an expectation, but it is not `G^{(n)}(1)`, because there is
/// no `G` — and for `n` past [`super::MAX_MOMENT_ORDER`].
pub fn factorial_moment(
    dist: &Distribution,
    n: u32,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    require_integer_support(dist, pool)?;
    if n > MAX_MOMENT_ORDER {
        return Err(ProbError::Unsupported(format!(
            "factorial moment order {n} exceeds MAX_MOMENT_ORDER = {MAX_MOMENT_ORDER}"
        )));
    }
    let claim = simplify(factorial_moment_claim(dist, n, pool), pool).value;

    let k = dists::fresh_var(&[claim], pool);
    // ∏_{j<n}(k - j), written out rather than expanded: it is what the
    // factorial moment *is*, and the verifier must sum the definition.
    let mut factors: Vec<ExprId> = (0..n).map(|j| sub(k, pool.integer(j), pool)).collect();
    if factors.is_empty() {
        factors.push(pool.integer(1));
    }
    let integrand = verify::definition_integrand(pool.mul(factors), k, dist, pool);
    let evidence = verify::check(claim, integrand, k, dist, pool)?;

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("prob_factorial_moment", k, claim));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log));
    Ok(DerivedExpr::with_log(claim, log))
}

// ---------------------------------------------------------------------------
// Cumulants
// ---------------------------------------------------------------------------

/// `κ_n` from the raw moments, by `κ_n = μ'_n - Σ_{m<n} C(n-1, m-1) κ_m μ'_{n-m}`.
///
/// The recursion, not `n` differentiations of `K`: `K` for a `Uniform` is
/// `log((e^{bt} - e^{at})/(t(b-a)))`, whose `n`-th derivative at `t = 0` is a
/// `0/0` that no amount of symbolic differentiation resolves. The recursion is
/// uniform across the table, and the gate checks it against an identity that
/// is not itself a recursion.
fn cumulant_claim(dist: &Distribution, n: u32, pool: &ExprPool) -> Result<ExprId, ProbError> {
    let mut kappa: Vec<ExprId> = vec![pool.integer(0)];
    for m in 1..=n {
        let mut terms = vec![moments::raw_moment_unchecked(dist, m, pool)?];
        for j in 1..m {
            let c = Integer::from(m - 1).binomial(j - 1);
            let mu = moments::raw_moment_unchecked(dist, m - j, pool)?;
            terms.push(pool.mul(vec![pool.integer(-c), kappa[j as usize], mu]));
        }
        kappa.push(simplify(pool.add(terms), pool).value);
    }
    Ok(kappa[n as usize])
}

/// `κ_n` in the central moments, indexed by `n - 2`.
///
/// Kendall & Stuart, *The Advanced Theory of Statistics* I, §3.14. `κ₁` is the
/// mean and is not here; `κ₂ = μ₂` and `κ₃ = μ₃` are, so that the gate runs on
/// them too rather than trusting the two orders most likely to be right.
const CUMULANT_IN_CENTRAL_MOMENTS: [&[CentralTerm]; 5] = [
    // κ₂ = μ₂
    &[(1, &[(2, 1)])],
    // κ₃ = μ₃
    &[(1, &[(3, 1)])],
    // κ₄ = μ₄ - 3μ₂²
    &[(1, &[(4, 1)]), (-3, &[(2, 2)])],
    // κ₅ = μ₅ - 10μ₃μ₂
    &[(1, &[(5, 1)]), (-10, &[(3, 1), (2, 1)])],
    // κ₆ = μ₆ - 15μ₄μ₂ - 10μ₃² + 30μ₂³
    &[
        (1, &[(6, 1)]),
        (-15, &[(4, 1), (2, 1)]),
        (-10, &[(3, 2)]),
        (30, &[(2, 3)]),
    ],
];

/// The `n`-th cumulant `κ_n = K_X^{(n)}(0)`, in closed form, verified.
///
/// `κ₁` is the mean and `κ₂` the variance; for a `Normal` every `κ_n` with
/// `n ≥ 3` is `0`, and for a `Poisson(λ)` every `κ_n` is `λ`.
///
/// # Errors
///
/// [`ProbError::Divergent`] for `LogNormal`: `κ_n` is a derivative of
/// `K = log M` at the origin, `M` is infinite at every `t > 0`, and there is
/// therefore no function to differentiate. The moment–cumulant recursion runs
/// perfectly well on a log-normal's moments and computes the coefficients of a
/// divergent series, which is the silent error this refusal exists for — use
/// [`skewness`] and [`excess_kurtosis`], which are defined from central
/// moments and do exist there.
///
/// [`ProbError::Unsupported`] for `n = 0` (there is no `κ₀`) and for `n` past
/// [`MAX_CUMULANT_ORDER`].
pub fn cumulant(
    dist: &Distribution,
    n: u32,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    if n == 0 {
        return Err(ProbError::Unsupported(
            "there is no zeroth cumulant: K(0) = log M(0) = log 1 = 0 identically, for every \
             law, and reporting it as κ₀ would be an answer about nothing"
                .to_string(),
        ));
    }
    if n > MAX_CUMULANT_ORDER {
        return Err(ProbError::Unsupported(format!(
            "cumulant order {n} exceeds MAX_CUMULANT_ORDER = {MAX_CUMULANT_ORDER}; past that \
             there is no second, independent expression for κ_n in this module to check the \
             moment–cumulant recursion against, and an unchecked cumulant is not an answer"
        )));
    }
    if dist.kind() == DistKind::LogNormal {
        return Err(ProbError::Divergent(
            "a log-normal has no cumulants: κ_n = K^{(n)}(0) with K = log E[e^{tX}], and \
             E[e^{tX}] = +∞ for every t > 0, so K exists on no neighbourhood of the origin \
             and there is nothing to differentiate. The moment–cumulant recursion does run — \
             it computes the coefficients of a divergent series. Skewness and excess kurtosis \
             are defined from central moments instead, and `skewness`/`excess_kurtosis` \
             return them"
                .to_string(),
        ));
    }
    // κ₁ is the mean, and `mean` already states it the way a reference does
    // and checks it against ∫x·p(x)dx.
    if n == 1 {
        return moments::mean(dist, pool);
    }
    let claim = simplify(cumulant_claim(dist, n, pool)?, pool).value;
    let mean = simplify(moments::mean_claim(dist, pool), pool).value;
    let x = dists::fresh_var(&[claim, mean], pool);
    let terms = CUMULANT_IN_CENTRAL_MOMENTS[(n - 2) as usize];
    let evidence = verify::check_cumulant(claim, terms, mean, x, dist, pool)?;

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("prob_cumulant", mean, claim));
    log.push(RewriteStep::simple(
        "prob_cumulant_matches_its_central_moment_expression",
        claim,
        claim,
    ));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log));
    Ok(DerivedExpr::with_log(claim, log))
}

// ---------------------------------------------------------------------------
// Skewness and excess kurtosis
// ---------------------------------------------------------------------------

/// `μ_k = E[(X - μ)^k] = Σ_j C(k, j) μ'_j (-μ)^{k-j}`.
fn central_moment_claim(dist: &Distribution, k: u32, pool: &ExprPool) -> Result<ExprId, ProbError> {
    let mu = moments::mean_claim(dist, pool);
    let mut terms = Vec::with_capacity(k as usize + 1);
    for j in 0..=k {
        let c = Integer::from(k).binomial(j);
        let signed = if (k - j) % 2 == 0 { c } else { -c };
        terms.push(pool.mul(vec![
            pool.integer(signed),
            moments::raw_moment_unchecked(dist, j, pool)?,
            pool.pow(mu, pool.integer(k - j)),
        ]));
    }
    Ok(pool.add(terms))
}

/// Shared body of [`skewness`] and [`excess_kurtosis`].
///
/// `order` is `3` or `4`; `shift` is the `-3` that turns the fourth
/// standardised moment into the *excess* kurtosis. The claim is assembled from
/// the raw-moment table and checked against `E[((X - μ)/σ)^order]`, quadratured
/// from the density — so the standardisation, which is where the mistakes live
/// (`σ³` against `σ^{3/2}`, kurtosis against *excess* kurtosis), is checked
/// rather than asserted.
fn standardised_moment(
    dist: &Distribution,
    order: u32,
    shift: i32,
    rule: &'static str,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    let mu2 = central_moment_claim(dist, 2, pool)?;
    let mu_k = central_moment_claim(dist, order, pool)?;
    // σ^order = (μ₂)^{order/2}: a half-integer power, never `sqrt(μ₂)^order`,
    // so nothing has to decide a branch of the square root.
    let sigma_pow = pool.pow(mu2, pool.rational(i64::from(order), 2));
    let claim = simplify(
        pool.add(vec![div(mu_k, sigma_pow, pool), pool.integer(shift)]),
        pool,
    )
    .value;

    let mean = simplify(moments::mean_claim(dist, pool), pool).value;
    let x = dists::fresh_var(&[claim, mean], pool);
    let standardised = pool.pow(
        div(sub(x, mean, pool), pool.pow(mu2, pool.rational(1, 2)), pool),
        pool.integer(order),
    );
    let integrand = verify::definition_integrand(standardised, x, dist, pool);
    // The gate compares against `E[((X-μ)/σ)^order]`, so the `-3` has to come
    // back off before the comparison and go back on after.
    let unshifted = simplify(sub(claim, pool.integer(shift), pool), pool).value;
    let evidence = verify::check(unshifted, integrand, x, dist, pool)?;

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(rule, x, claim));
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log));
    Ok(DerivedExpr::with_log(claim, log))
}

/// `γ₁ = E[((X - μ)/σ)³] = κ₃/σ³`, in closed form, verified.
///
/// Defined here from the **central moments**, not from [`cumulant`], and the
/// difference is not cosmetic: a `LogNormal` has no cumulants — its CGF exists
/// at no `t > 0` — and a skewness of `(e^{σ²} + 2)√(e^{σ²} - 1)` that every
/// reference prints. Routing this through the CGF would refuse a quantity that
/// plainly exists.
///
/// # Errors
///
/// See [`ProbError`]. Every law in this module has a fourth moment, so the
/// usual refusal here is [`ProbError::Unverified`].
pub fn skewness(dist: &Distribution, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, ProbError> {
    standardised_moment(dist, 3, 0, "prob_skewness", pool)
}

/// `γ₂ = E[((X - μ)/σ)⁴] - 3 = κ₄/σ⁴`, in closed form, verified.
///
/// **Excess** kurtosis: `0` for a normal, not `3`. The two conventions differ
/// by exactly the constant a reader is least likely to notice — both are
/// positive numbers of the same magnitude for a heavy-tailed law — so the one
/// that is returned is the one whose name says which it is, and the `-3` is
/// applied after the gate has compared the fourth standardised moment against
/// its own quadrature.
///
/// # Errors
///
/// See [`ProbError`].
pub fn excess_kurtosis(
    dist: &Distribution,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    standardised_moment(dist, 4, -3, "prob_excess_kurtosis", pool)
}
