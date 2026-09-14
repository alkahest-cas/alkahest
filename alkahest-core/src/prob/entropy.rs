//! Entropy, Kullback–Leibler divergence, cross-entropy — and the refusals that
//! keep them from being plausible nonsense.
//!
//! # Two different quantities wear the same name
//!
//! For a discrete law this module computes the **Shannon entropy**
//! `H(X) = -Σ p log p`, which is a number of bits (or nats), is non-negative,
//! is zero exactly at a point mass, and is **invariant under relabelling** —
//! permuting the atoms cannot change it.
//!
//! For a continuous law it computes the **differential entropy**
//! `h(X) = -∫ f log f`, and almost none of that survives:
//!
//! * `h` is **not** the limit of `H` under finer and finer discretisation.
//!   Quantising `X` at width `Δ` gives `H_Δ ≈ h(X) - log Δ`, and the `-log Δ`
//!   diverges. The two differ by an infinite constant, so they are not
//!   comparable quantities and `h` is not "the information content of `X`".
//! * `h` is **not non-negative**. `h(Uniform(0, ½)) = log ½ < 0`.
//!   [`entropy`] returns that negative number rather than clamping it, and a
//!   `Uniform(0, b)` with symbolic `b` returns `log b`, of either sign,
//!   because neither sign is decidable.
//! * `h` is **not invariant under a change of variables**. Under `X = g(Y)`
//!   it shifts by `E[log|g'|]`. The table below contains its own witness:
//!   `LogNormal(μ, σ)` is `e^{Normal(μ, σ)}`, a smooth bijection onto `(0, ∞)`,
//!   and its entropy is `μ + ½log(2πeσ²)` — the normal's, shifted by exactly
//!   `E[log|dx/dy|] = μ`. Reading `h` as a relabelling-invariant "information
//!   content" would make those two equal, and they are not.
//!
//! The two are therefore *not* unified behind one formula here. Which one you
//! get is decided by [`Distribution::support`], and the derivation log says
//! which: `prob_shannon_entropy` or `prob_differential_entropy`.
//!
//! # `D(P‖Q)` is not a distance
//!
//! It is not symmetric (`D(Exponential(1)‖Exponential(4)) = 3·log 2 ≈ 2.386`
//! while `D(Exponential(4)‖Exponential(1)) = 1.5 - log 4·…`; the closed form is
//! asymmetric in `λ₁, λ₂` and the two numbers differ), it does not satisfy the
//! triangle inequality, and — the part that has to be *gated* rather than
//! documented — it is `+∞` whenever `supp P ⊄ supp Q`.
//!
//! `D(Uniform(0,1) ‖ Uniform(0,½))` is the canonical case. The closed form
//! `log((b₂-a₂)/(b₁-a₁))` evaluates there to `log ½ = -0.693`: finite,
//! clean, **negative**, and impossible — a KL divergence is non-negative by
//! Gibbs' inequality. Nothing in the arithmetic complains, because the formula
//! was derived under a containment hypothesis that the substitution silently
//! dropped. So [`kl_divergence`] resolves that hypothesis *before* it reaches
//! the formula:
//!
//! * decidably contained → the closed form, verified;
//! * decidably not contained → [`ProbError::Divergent`] (`E-PROB-006`) saying
//!   the value is `+∞`. A verdict, not an expression;
//! * undecidable (symbolic endpoints) → the closed form with the containment
//!   carried on [`super::take_prob_side_conditions`], so the caller can see
//!   which hypothesis their number rests on.
//!
//! # What is not here
//!
//! **Mutual information of a dependent pair.** `I(X;Y) = -½log(1-ρ²)` for a
//! bivariate normal is a one-line formula with nowhere to live: this module has
//! no joint distribution type (see [`super`]'s note on why), and inventing one
//! whose only inhabitant is the bivariate normal would be a formula in search
//! of a representation. [`mutual_information_independent`] covers the case the
//! marginals *do* determine — independence, where `I = 0` — and says in its
//! name that the assumption is the caller's, exactly as
//! [`super::variance_affine_independent`] does.

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::simplify::simplify;

use super::dists::{self, Sign};
use super::verify;
use super::{div, sub, DistKind, Distribution, ProbError};

/// Largest `Binomial` `n` whose entropy is assembled.
///
/// `H(Binomial(n, p))` has no closed form; what [`entropy`] returns is the
/// exact identity `n·H(Bernoulli(p)) - E[log C(n, k)]` with the residual
/// expectation written out over the support. That is `n - 1` terms, so it is a
/// value for a small `n` and an unreadable wall for a large one.
pub const MAX_BINOMIAL_ENTROPY_N: u32 = 64;

fn log(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("log", vec![arg])
}

/// Is `e` decidably zero?
///
/// [`dists::sign_of`] reads a rigorous enclosure, and an enclosure of the
/// *unsimplified* `1 + 1·(-1)` has a non-zero radius, so it reports `Unknown`
/// where the value is exactly `0`. Every "is this parameter at the edge of its
/// range" question below therefore simplifies first — without it
/// `Bernoulli(1)` is not recognised as the point mass it is, and its entropy
/// goes to the formula, where `0 log 0` is `NaN`.
fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    matches!(dists::sign_of(simplify(e, pool).value, pool), Sign::Zero)
}

fn gamma_fn(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("gamma", vec![arg])
}

fn digamma(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("digamma", vec![arg])
}

// ---------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------

/// The conversion from nats, and the hypotheses it needs.
///
/// `None` — the default — is nats, and is the right default for symbolic work:
/// `log` is the natural log everywhere else in this library, `h(Normal)` is
/// `½log(2πeσ²)` without a stray constant, and every identity below
/// (`H(P,Q) = H(P) + D(P‖Q)`, `D(P‖P) = 0`) holds verbatim. Bits are
/// `base = 2`, and the whole of the difference is a division by `log 2`.
///
/// A base is a number the caller supplies, so it gets the same treatment as
/// any other parameter: a numeric one is decided (`base ≤ 0` or `base = 1` is
/// [`ProbError::InvalidParameter`] — `log 1 = 0` and the conversion is a
/// division by zero), a symbolic one is carried.
fn base_factor(
    base: Option<ExprId>,
    pool: &ExprPool,
) -> Result<(Option<ExprId>, Vec<SideCondition>), ProbError> {
    let Some(b) = base else {
        return Ok((None, Vec::new()));
    };
    let mut conds = Vec::new();
    match dists::sign_of(b, pool) {
        Sign::Positive => {}
        Sign::Unknown => conds.push(SideCondition::Positive(b)),
        Sign::Negative | Sign::Zero => {
            return Err(ProbError::InvalidParameter {
                parameter: "base",
                requirement: "> 0",
            })
        }
    }
    let gap = simplify(sub(b, pool.integer(1), pool), pool).value;
    match dists::sign_of(gap, pool) {
        Sign::Zero => {
            return Err(ProbError::InvalidParameter {
                parameter: "base",
                requirement: "≠ 1 (log 1 = 0, so there is no such unit)",
            })
        }
        Sign::Unknown => conds.push(SideCondition::NonZero(gap)),
        Sign::Positive | Sign::Negative => {}
    }
    Ok((Some(pool.pow(log(b, pool), pool.integer(-1))), conds))
}

/// `x` in the requested unit. Applied to the claim **and** to the integrand the
/// claim is checked against, so the gate compares what is actually returned
/// rather than its natural-log ancestor.
fn rescale(x: ExprId, factor: Option<ExprId>, pool: &ExprPool) -> ExprId {
    match factor {
        Some(f) => pool.mul(vec![f, x]),
        None => x,
    }
}

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

/// `-p log p - (1-p) log(1-p)`, the two-atom Shannon entropy.
fn shannon_two(p: ExprId, pool: &ExprPool) -> ExprId {
    let neg1 = pool.integer(-1);
    let q = sub(pool.integer(1), p, pool);
    pool.add(vec![
        pool.mul(vec![neg1, p, log(p, pool)]),
        pool.mul(vec![neg1, q, log(q, pool)]),
    ])
}

/// `2πσ²`.
fn two_pi_sigma_sq(sigma: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![
        pool.integer(2),
        super::pi(pool),
        pool.pow(sigma, pool.integer(2)),
    ])
}

/// The table, in nats, one entry per law, written the way a reference writes it.
///
/// `Poisson` is absent on purpose and refuses: `H = λ(1 - log λ) + e^{-λ}Σ
/// λ^k log(k!)/k!` and the residual sum is not a standard special function.
/// Truncating it would produce a number that is wrong in the fourth digit and
/// indistinguishable from a right one.
fn entropy_claim(dist: &Distribution, pool: &ExprPool) -> Result<ExprId, ProbError> {
    let p = dist.params();
    let one = pool.integer(1);
    let two = pool.integer(2);
    let neg1 = pool.integer(-1);
    let half = pool.rational(1, 2);
    Ok(match dist.kind() {
        // ½log(2πσ²) + ½ = ½log(2πeσ²). Independent of μ: a shift moves the
        // density without changing its shape.
        DistKind::Normal => pool.add(vec![
            pool.mul(vec![half, log(two_pi_sigma_sq(p[1], pool), pool)]),
            half,
        ]),
        // μ + ½log(2πeσ²). The normal's, shifted by E[log|dx/dy|] = μ — the
        // whole of the change-of-variables correction, in one term.
        DistKind::LogNormal => pool.add(vec![
            p[0],
            pool.mul(vec![half, log(two_pi_sigma_sq(p[1], pool), pool)]),
            half,
        ]),
        // log(b - a): negative for a short interval, and returned so.
        DistKind::Uniform => log(sub(p[1], p[0], pool), pool),
        // 1 - log λ
        DistKind::Exponential => sub(one, log(p[0], pool), pool),
        // k + log θ + log Γ(k) + (1-k)ψ(k)
        DistKind::Gamma => pool.add(vec![
            p[0],
            log(p[1], pool),
            log(gamma_fn(p[0], pool), pool),
            pool.mul(vec![sub(one, p[0], pool), digamma(p[0], pool)]),
        ]),
        // log B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (α+β-2)ψ(α+β)
        DistKind::Beta => {
            let (a, b) = (p[0], p[1]);
            let ab = pool.add(vec![a, b]);
            let log_beta = log(
                pool.mul(vec![
                    gamma_fn(a, pool),
                    gamma_fn(b, pool),
                    pool.pow(gamma_fn(ab, pool), pool.integer(-1)),
                ]),
                pool,
            );
            pool.add(vec![
                log_beta,
                pool.mul(vec![neg1, sub(a, one, pool), digamma(a, pool)]),
                pool.mul(vec![neg1, sub(b, one, pool), digamma(b, pool)]),
                pool.mul(vec![sub(ab, two, pool), digamma(ab, pool)]),
            ])
        }
        DistKind::Bernoulli => shannon_two(p[0], pool),
        DistKind::Binomial => {
            // H = n·H(Bernoulli(p)) - E[log C(n, k)].
            //
            // Written this way rather than as the defining sum `-Σ P log P`,
            // and the difference is not cosmetic: the defining sum *is* what
            // the verifier computes, so returning it would make the numeric
            // gate compare an expression with itself and pass unconditionally.
            // This form is a genuine rearrangement — the `k log p` and
            // `(n-k) log q` parts summed in closed form, only the
            // coefficient's logarithm left over — so an error in the algebra
            // shows up as a disagreement.
            let n = dists::count_of(p[0], pool);
            if n > MAX_BINOMIAL_ENTROPY_N {
                return Err(ProbError::Unsupported(format!(
                    "H(Binomial(n, p)) has no closed form; the identity \
                     n·H(Bernoulli(p)) - E[log C(n, k)] leaves a residual sum of n - 1 terms, \
                     and n = {n} exceeds MAX_BINOMIAL_ENTROPY_N = {MAX_BINOMIAL_ENTROPY_N}"
                )));
            }
            let prob = p[1];
            let q = sub(one, prob, pool);
            let mut terms = vec![pool.mul(vec![p[0], shannon_two(prob, pool)])];
            for k in 1..n {
                let c = rug::Integer::from(n).binomial(k);
                let log_c = log(pool.integer(c.clone()), pool);
                terms.push(pool.mul(vec![
                    neg1,
                    pool.integer(c),
                    pool.pow(prob, pool.integer(k)),
                    pool.pow(q, pool.integer(n - k)),
                    log_c,
                ]));
            }
            pool.add(terms)
        }
        DistKind::Poisson => {
            return Err(ProbError::NoClosedForm {
                quantity: "the Poisson entropy",
                missing: "a closed form for e^{-λ} Σ_k λ^k log(k!)/k! — the residual sum \
                          after λ(1 - log λ) is not an elementary or standard special \
                          function, and only has asymptotic expansions",
            })
        }
    })
}

/// Is this law a point mass, so that its entropy is exactly `0`?
///
/// `Bernoulli(0)`, `Bernoulli(1)` and any `Binomial` with `p ∈ {0, 1}` or
/// `n = 0` put all their mass on one atom. `H = 0` there, by the `0 log 0 = 0`
/// convention — but the closed form contains `p log p`, which evaluates to
/// `NaN` at `p = 0`. So the degenerate case is answered exactly and the formula
/// is never reached.
fn degenerate_point_mass(dist: &Distribution, pool: &ExprPool) -> bool {
    let p = dist.params();
    let prob = match dist.kind() {
        DistKind::Bernoulli => p[0],
        DistKind::Binomial => {
            if dists::count_of(p[0], pool) == 0 {
                return true;
            }
            p[1]
        }
        _ => return false,
    };
    let one = pool.integer(1);
    is_zero(prob, pool) || is_zero(sub(prob, one, pool), pool)
}

/// `0 < p` and `p < 1`, when neither can be decided.
///
/// `H(Bernoulli(p))` is `0` at both endpoints of the parameter range and the
/// closed form is `NaN` there. For a symbolic `p` that is not a refusal — the
/// formula is right on the open interval, which is where a `Bernoulli` usually
/// lives — but it *is* a hypothesis, and the constructor's own `0 ≤ p ≤ 1` is
/// the non-strict version, so nothing else would record it.
fn open_probability_conditions(dist: &Distribution, pool: &ExprPool) -> Vec<SideCondition> {
    let p = dist.params();
    let prob = match dist.kind() {
        DistKind::Bernoulli => p[0],
        DistKind::Binomial => p[1],
        _ => return Vec::new(),
    };
    let one = pool.integer(1);
    let mut out = Vec::new();
    if matches!(
        dists::sign_of(simplify(prob, pool).value, pool),
        Sign::Unknown
    ) {
        out.push(SideCondition::Positive(prob));
    }
    let gap = simplify(sub(one, prob, pool), pool).value;
    if matches!(dists::sign_of(gap, pool), Sign::Unknown) {
        out.push(SideCondition::Positive(gap));
    }
    out
}

/// The entropy of `dist`, in nats unless `base` says otherwise, verified.
///
/// Shannon entropy `H = -Σ p log p` on a discrete support, differential
/// entropy `h = -∫ f log f` on a continuous one. **These are different
/// quantities** — see the module documentation before treating a value from
/// here as "the information content" of anything.
///
/// # Errors
///
/// * [`ProbError::NoClosedForm`] for `Poisson`: the residual sum
///   `e^{-λ}Σ λ^k log(k!)/k!` is not expressible here.
/// * [`ProbError::Unsupported`] for a `Binomial` past
///   [`MAX_BINOMIAL_ENTROPY_N`].
/// * [`ProbError::InvalidParameter`] for a numeric `base` that is `≤ 0` or `1`.
/// * [`ProbError::Unverified`] if the numeric gate could not confirm the table
///   entry against quadrature of `-∫ f log f`.
pub fn entropy(
    dist: &Distribution,
    base: Option<ExprId>,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    // Ahead of everything else, so a `Poisson` reports E-PROB-004 rather than
    // being short-circuited by a degenerate-parameter branch it does not have.
    let raw = entropy_claim(dist, pool)?;
    let (factor, base_conds) = base_factor(base, pool)?;

    let discrete = dist.support(pool).is_discrete();
    let rule = if discrete {
        "prob_shannon_entropy"
    } else {
        "prob_differential_entropy"
    };

    if degenerate_point_mass(dist, pool) {
        let zero = pool.integer(0);
        let mut log_ = DerivationLog::new();
        log_.push(RewriteStep::simple(
            "prob_entropy_of_a_point_mass_is_zero",
            zero,
            zero,
        ));
        super::stash_prob_side_conditions(super::conditions_of(&log_));
        return Ok(DerivedExpr::with_log(zero, log_));
    }

    let claim = simplify(rescale(raw, factor, pool), pool).value;

    let x = dists::fresh_var(dist.params(), pool);
    let density = dists::pdf(dist, x, pool);
    // `f(x) = -log p(x)`, so the integrand is the defining `-p log p`.
    let f = rescale(
        pool.mul(vec![pool.integer(-1), log(density, pool)]),
        factor,
        pool,
    );
    let integrand = verify::definition_integrand(f, x, dist, pool);
    let evidence = verify::check(claim, integrand, x, dist, pool)?;

    let mut log_ = DerivationLog::new();
    log_.push(RewriteStep::simple(rule, x, claim));
    let mut conds = open_probability_conditions(dist, pool);
    conds.extend(base_conds);
    if !conds.is_empty() {
        log_.push(RewriteStep::with_conditions(
            "prob_entropy_hypotheses",
            claim,
            claim,
            conds,
        ));
    }
    log_.push(verify::evidence_step(&evidence, claim, dist, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log_));
    Ok(DerivedExpr::with_log(claim, log_))
}

// ---------------------------------------------------------------------------
// Support containment — the gate `D(P‖Q)` needs and the formula cannot express
// ---------------------------------------------------------------------------

/// One hypothesis that `supp P ⊆ supp Q` reduces to: `gap ≥ 0`, or `gap > 0`
/// when `strict`.
struct Gate {
    gap: ExprId,
    strict: bool,
    /// Spelled out, for the refusal message and the side condition.
    reads: &'static str,
}

fn diverged(reads: &str) -> ProbError {
    ProbError::Divergent(format!(
        "D(P‖Q) = +∞: P charges a set Q gives probability zero ({reads} fails), so \
         log(dP/dQ) is +∞ on a set of positive P-measure. The same-family closed form is \
         derived under the containment hypothesis and returns a finite — often negative — \
         number when it is substituted through regardless, which no divergence is"
    ))
}

/// `p₂ > 0` and `1 - p₂ > 0`: a two-atom `Q` must charge both atoms that `P`
/// charges. Dropped where `P` itself does not charge one of them.
fn atom_gates(p1: ExprId, p2: ExprId, pool: &ExprPool) -> Vec<Gate> {
    let one = pool.integer(1);
    let mut out = Vec::new();
    // P charges {1} unless p₁ = 0.
    if !is_zero(p1, pool) {
        out.push(Gate {
            gap: p2,
            strict: true,
            reads: "q(1) > 0",
        });
    }
    // P charges {0} unless p₁ = 1.
    if !is_zero(sub(p1, one, pool), pool) {
        out.push(Gate {
            gap: sub(one, p2, pool),
            strict: true,
            reads: "q(0) > 0",
        });
    }
    out
}

/// Resolve `supp P ⊆ supp Q` as far as it can be resolved.
///
/// `Ok(gates)` are the hypotheses that could **not** be decided and must
/// therefore travel with the answer. `Err` is the decided-false case, and it is
/// a verdict — `+∞` — rather than a refusal to try.
fn support_gates(
    p: &Distribution,
    q: &Distribution,
    pool: &ExprPool,
) -> Result<Vec<Gate>, ProbError> {
    // `supp P ⊆ supp P` needs no hypothesis. Without this, `D(P‖P)` over a
    // `Bernoulli(p)` with symbolic `p` would publish `p > 0` and `1 - p > 0` —
    // true but irrelevant, and a caller who has to discharge irrelevant
    // hypotheses stops reading the relevant ones.
    if p == q {
        return Ok(Vec::new());
    }
    let (pp, qp) = (p.params(), q.params());
    let raw: Vec<Gate> = match p.kind() {
        // Identical supports, whatever the parameters: `ℝ`, `(0, ∞)`, `[0, 1]`,
        // `{0, 1, 2, …}`. Nothing to decide.
        DistKind::Normal
        | DistKind::LogNormal
        | DistKind::Exponential
        | DistKind::Gamma
        | DistKind::Beta
        | DistKind::Poisson => Vec::new(),
        // [a₁, b₁] ⊆ [a₂, b₂].
        DistKind::Uniform => vec![
            Gate {
                gap: sub(pp[0], qp[0], pool),
                strict: false,
                reads: "a₁ ≥ a₂",
            },
            Gate {
                gap: sub(qp[1], pp[1], pool),
                strict: false,
                reads: "b₁ ≤ b₂",
            },
        ],
        DistKind::Bernoulli => atom_gates(pp[0], qp[0], pool),
        DistKind::Binomial => {
            let (n1, n2) = (dists::count_of(pp[0], pool), dists::count_of(qp[0], pool));
            if n1 > n2 {
                return Err(diverged("{0…n₁} ⊆ {0…n₂}"));
            }
            atom_gates(pp[1], qp[1], pool)
        }
    };

    let mut carried = Vec::new();
    for g in raw {
        let gap = simplify(g.gap, pool).value;
        match dists::sign_of(gap, pool) {
            Sign::Positive => {}
            Sign::Zero if !g.strict => {}
            Sign::Zero | Sign::Negative => return Err(diverged(g.reads)),
            Sign::Unknown => carried.push(Gate { gap, ..g }),
        }
    }
    Ok(carried)
}

// ---------------------------------------------------------------------------
// Kullback–Leibler divergence
// ---------------------------------------------------------------------------

/// `D(P‖Q)` in nats, as a claim. `None` where the pair has no closed form.
fn kl_claim(p: &Distribution, q: &Distribution, pool: &ExprPool) -> Result<ExprId, ProbError> {
    let (a, b) = (p.params(), q.params());
    let two = pool.integer(2);
    let neg1 = pool.integer(-1);
    let half = pool.rational(1, 2);
    Ok(match p.kind() {
        // log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - ½.
        //
        // The log-normal's is the *same* expression: KL is invariant under a
        // common invertible change of variables (the Jacobians cancel inside
        // the log), which is precisely the property differential entropy does
        // not have. Two quantities built from the same integrand, one
        // invariant and one not.
        DistKind::Normal | DistKind::LogNormal => {
            let (m1, s1, m2, s2) = (a[0], a[1], b[0], b[1]);
            pool.add(vec![
                log(div(s2, s1, pool), pool),
                div(
                    pool.add(vec![pool.pow(s1, two), pool.pow(sub(m1, m2, pool), two)]),
                    pool.mul(vec![two, pool.pow(s2, two)]),
                    pool,
                ),
                pool.mul(vec![neg1, half]),
            ])
        }
        // log((b₂-a₂)/(b₁-a₁)), under [a₁,b₁] ⊆ [a₂,b₂].
        DistKind::Uniform => log(
            div(sub(b[1], b[0], pool), sub(a[1], a[0], pool), pool),
            pool,
        ),
        // log(λ₁/λ₂) + λ₂/λ₁ - 1. Manifestly asymmetric.
        DistKind::Exponential => pool.add(vec![
            log(div(a[0], b[0], pool), pool),
            div(b[0], a[0], pool),
            neg1,
        ]),
        // (k₁-k₂)ψ(k₁) + k₂log(θ₂/θ₁) + k₁(θ₁-θ₂)/θ₂ + log(Γ(k₂)/Γ(k₁))
        DistKind::Gamma => {
            let (k1, t1, k2, t2) = (a[0], a[1], b[0], b[1]);
            pool.add(vec![
                pool.mul(vec![sub(k1, k2, pool), digamma(k1, pool)]),
                pool.mul(vec![k2, log(div(t2, t1, pool), pool)]),
                pool.mul(vec![k1, div(sub(t1, t2, pool), t2, pool)]),
                log(div(gamma_fn(k2, pool), gamma_fn(k1, pool), pool), pool),
            ])
        }
        // log(B(α₂,β₂)/B(α₁,β₁)) + (α₁-α₂)(ψ(α₁)-ψ(α₁+β₁))
        //                        + (β₁-β₂)(ψ(β₁)-ψ(α₁+β₁))
        DistKind::Beta => {
            let (a1, b1, a2, b2) = (a[0], a[1], b[0], b[1]);
            let s1 = pool.add(vec![a1, b1]);
            // `log(B(α₂,β₂)/B(α₁,β₁))` as **one** product of Γ atoms and their
            // inverses, not as a quotient of two compound products. A `Mul`
            // raised to `-1` does not distribute over its factors, so the
            // quotient form leaves `D(P‖P)` as `log(B/B)` rather than `0`; the
            // flat form cancels factor by factor, which is what makes the
            // `D(P‖P) = 0` identity hold *symbolically* rather than only at a
            // parameter point.
            let inv = |e: ExprId| pool.pow(e, pool.integer(-1));
            let s2 = pool.add(vec![a2, b2]);
            pool.add(vec![
                log(
                    pool.mul(vec![
                        gamma_fn(a2, pool),
                        gamma_fn(b2, pool),
                        inv(gamma_fn(s2, pool)),
                        inv(gamma_fn(a1, pool)),
                        inv(gamma_fn(b1, pool)),
                        gamma_fn(s1, pool),
                    ]),
                    pool,
                ),
                pool.mul(vec![
                    sub(a1, a2, pool),
                    sub(digamma(a1, pool), digamma(s1, pool), pool),
                ]),
                pool.mul(vec![
                    sub(b1, b2, pool),
                    sub(digamma(b1, pool), digamma(s1, pool), pool),
                ]),
            ])
        }
        // p₁log(p₁/p₂) + (1-p₁)log((1-p₁)/(1-p₂))
        DistKind::Bernoulli => bernoulli_kl(a[0], b[0], pool),
        // n identical trials: D = n·D_Bernoulli(p₁‖p₂).
        DistKind::Binomial => {
            let (n1, n2) = (dists::count_of(a[0], pool), dists::count_of(b[0], pool));
            if n1 != n2 {
                return Err(ProbError::Unsupported(format!(
                    "D(Binomial({n1}, p₁) ‖ Binomial({n2}, p₂)) with n₁ ≠ n₂ has no closed \
                     form: log(dP/dQ) carries log(C({n1},k)/C({n2},k)), which does not sum \
                     in closed form. Equal n gives n·D(Bernoulli(p₁)‖Bernoulli(p₂))"
                )));
            }
            pool.mul(vec![a[0], bernoulli_kl(a[1], b[1], pool)])
        }
        // λ₁log(λ₁/λ₂) + λ₂ - λ₁.
        //
        // Worth noting against the entropy table: the Poisson *entropy* has no
        // closed form and its divergence does. The log of the density ratio
        // loses the `log k!` that made the entropy intractable, because it
        // appears in both densities and cancels.
        DistKind::Poisson => pool.add(vec![
            pool.mul(vec![a[0], log(div(a[0], b[0], pool), pool)]),
            b[0],
            pool.mul(vec![neg1, a[0]]),
        ]),
    })
}

fn bernoulli_kl(p1: ExprId, p2: ExprId, pool: &ExprPool) -> ExprId {
    let one = pool.integer(1);
    let (q1, q2) = (sub(one, p1, pool), sub(one, p2, pool));
    pool.add(vec![
        pool.mul(vec![p1, log(div(p1, p2, pool), pool)]),
        pool.mul(vec![q1, log(div(q1, q2, pool), pool)]),
    ])
}

/// `D(P‖Q) = ∫ p log(p/q)`, in nats unless `base` says otherwise, verified.
///
/// `P` and `Q` must be the same family; a cross-family divergence generally has
/// no closed form and is [`ProbError::Unsupported`] rather than a guess.
///
/// **Not a distance.** `D(P‖Q) ≠ D(Q‖P)`, and it is `+∞` when `supp P ⊄
/// supp Q` — see the module documentation for how that hypothesis is resolved
/// rather than assumed.
///
/// # Errors
///
/// * [`ProbError::Divergent`] (`E-PROB-006`) when the supports are decidably
///   not nested. The value is `+∞`; this is the verdict, not a refusal to try.
/// * [`ProbError::Unsupported`] for mismatched families, and for two
///   `Binomial`s with different `n`.
/// * [`ProbError::InvalidParameter`] for a bad `base`.
/// * [`ProbError::Unverified`] if the gate could not confirm the closed form.
pub fn kl_divergence(
    p: &Distribution,
    q: &Distribution,
    base: Option<ExprId>,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    let (claim, evidence, conds) = kl_parts(p, q, base, pool)?;
    let mut log_ = DerivationLog::new();
    log_.push(RewriteStep::simple("prob_kl_divergence", claim, claim));
    if !conds.is_empty() {
        log_.push(RewriteStep::with_conditions(
            "prob_kl_support_containment_and_parameters",
            claim,
            claim,
            conds,
        ));
    }
    log_.push(verify::evidence_step(&evidence, claim, p, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log_));
    Ok(DerivedExpr::with_log(claim, log_))
}

/// The shared body of [`kl_divergence`] and [`cross_entropy`]: the claim, the
/// evidence from checking it against `∫ p log(p/q)`, and the hypotheses that
/// could not be discharged.
fn kl_parts(
    p: &Distribution,
    q: &Distribution,
    base: Option<ExprId>,
    pool: &ExprPool,
) -> Result<(ExprId, verify::Evidence, Vec<SideCondition>), ProbError> {
    if p.kind() != q.kind() {
        return Err(ProbError::Unsupported(format!(
            "D({} ‖ {}) is a cross-family divergence; this table carries the same-family \
             closed forms only. ∫p log(p/q) between different families generally has no \
             elementary value, and the nearest same-family formula is not it",
            p.kind().name(),
            q.kind().name()
        )));
    }
    let raw = kl_claim(p, q, pool)?;
    let (factor, base_conds) = base_factor(base, pool)?;
    let gates = support_gates(p, q, pool)?;

    let claim = simplify(rescale(raw, factor, pool), pool).value;

    let x = dists::fresh_var(&[p.params(), q.params()].concat(), pool);
    let (dp, dq) = (dists::pdf(p, x, pool), dists::pdf(q, x, pool));
    let f = rescale(log(div(dp, dq, pool), pool), factor, pool);
    let integrand = pool.mul(vec![f, dp]);

    // The gate must never sample a parameter point outside the hypotheses the
    // answer is being returned under: at an `a₂ > a₁` the integrand is the
    // *wrong* integrand (it silently uses `q`'s formula off `q`'s support) and
    // would be compared against a closed form that does not claim to hold
    // there. So the undecided containment travels into the sampler as a
    // constraint, alongside `Q`'s own parameter constraints, which nothing
    // else in `verify` knows about.
    let mut extra = q.constraints(pool);
    let zero = pool.integer(0);
    for g in &gates {
        extra.push(if g.strict {
            pool.pred_gt(g.gap, zero)
        } else {
            pool.pred_ge(g.gap, zero)
        });
    }
    let evidence = verify::check_pair(claim, integrand, x, p, q, &extra, pool)?;

    let mut conds: Vec<SideCondition> = gates
        .iter()
        .map(|g| {
            if g.strict {
                SideCondition::Positive(g.gap)
            } else {
                SideCondition::InDomain(g.gap, Domain::NonNegative)
            }
        })
        .collect();
    for c in q.constraints(pool) {
        if let Some(side) = verify::undischarged(c, pool) {
            if !conds.contains(&side) {
                conds.push(side);
            }
        }
    }
    conds.extend(base_conds);
    Ok((claim, evidence, conds))
}

// ---------------------------------------------------------------------------
// Cross-entropy
// ---------------------------------------------------------------------------

/// `H(P, Q) = -∫ p log q = H(P) + D(P‖Q)`, verified.
///
/// Assembled from the *two* tables and then checked against `-E_P[log q]`,
/// which is neither of them. That makes the identity in the name an assertion
/// the gate can falsify rather than a definition: an error in either the
/// entropy entry or the divergence entry surfaces here as a disagreement, and
/// only a pair of errors that cancel exactly would survive.
///
/// # Errors
///
/// Whatever [`entropy`] or [`kl_divergence`] would refuse — in particular
/// `Poisson`, whose entropy has no closed form and whose cross-entropy
/// therefore does not either.
pub fn cross_entropy(
    p: &Distribution,
    q: &Distribution,
    base: Option<ExprId>,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    if p.kind() != q.kind() {
        return Err(ProbError::Unsupported(format!(
            "H({}, {}) is a cross-family cross-entropy; this table carries the same-family \
             closed forms only",
            p.kind().name(),
            q.kind().name()
        )));
    }
    let h = entropy_claim(p, pool)?;
    let d = kl_claim(p, q, pool)?;
    let (factor, base_conds) = base_factor(base, pool)?;
    let gates = support_gates(p, q, pool)?;

    if degenerate_point_mass(p, pool) {
        // `H(P) = 0`, so `H(P, Q) = D(P‖Q)` — and the entropy formula's
        // `p log p` would have been `NaN` here.
        return kl_divergence(p, q, base, pool);
    }

    let claim = simplify(rescale(pool.add(vec![h, d]), factor, pool), pool).value;

    let x = dists::fresh_var(&[p.params(), q.params()].concat(), pool);
    let (dp, dq) = (dists::pdf(p, x, pool), dists::pdf(q, x, pool));
    let f = rescale(
        pool.mul(vec![pool.integer(-1), log(dq, pool)]),
        factor,
        pool,
    );
    let integrand = pool.mul(vec![f, dp]);

    let mut extra = q.constraints(pool);
    let zero = pool.integer(0);
    for g in &gates {
        extra.push(if g.strict {
            pool.pred_gt(g.gap, zero)
        } else {
            pool.pred_ge(g.gap, zero)
        });
    }
    let evidence = verify::check_pair(claim, integrand, x, p, q, &extra, pool)?;

    let mut log_ = DerivationLog::new();
    log_.push(RewriteStep::simple(
        "prob_cross_entropy_is_entropy_plus_divergence",
        claim,
        claim,
    ));
    let mut conds: Vec<SideCondition> = gates
        .iter()
        .map(|g| {
            if g.strict {
                SideCondition::Positive(g.gap)
            } else {
                SideCondition::InDomain(g.gap, Domain::NonNegative)
            }
        })
        .collect();
    conds.extend(open_probability_conditions(p, pool));
    for c in q.constraints(pool) {
        if let Some(side) = verify::undischarged(c, pool) {
            if !conds.contains(&side) {
                conds.push(side);
            }
        }
    }
    conds.extend(base_conds);
    if !conds.is_empty() {
        log_.push(RewriteStep::with_conditions(
            "prob_cross_entropy_hypotheses",
            claim,
            claim,
            conds,
        ));
    }
    log_.push(verify::evidence_step(&evidence, claim, p, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log_));
    Ok(DerivedExpr::with_log(claim, log_))
}

// ---------------------------------------------------------------------------
// Mutual information
// ---------------------------------------------------------------------------

/// `I(X; Y) = 0`, **assuming `X` and `Y` are independent**.
///
/// The assumption is in the name because nothing here can check it: a pair of
/// marginals does not record whether they are independent, and under dependence
/// the answer is wrong by the whole of `I`, with nothing in the return value to
/// say so. If you cannot assert independence, do not call this.
///
/// There is deliberately no dependent case. `I = -½log(1-ρ²)` for a bivariate
/// normal would need a joint-distribution type, and this module does not have
/// one (see [`super`]); adding one whose only inhabitant existed to host this
/// formula would be a worse answer than not having the formula.
///
/// # Errors
///
/// [`ProbError::InvalidParameter`] for a numeric `base` that is `≤ 0` or `1` —
/// the only thing there is to get wrong, since `0` is `0` in every unit.
pub fn mutual_information_independent(
    x: &Distribution,
    y: &Distribution,
    base: Option<ExprId>,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    let (_, base_conds) = base_factor(base, pool)?;
    let zero = pool.integer(0);
    let mut log_ = DerivationLog::new();
    let mut conds = base_conds;
    for d in [x, y] {
        for c in d.constraints(pool) {
            if let Some(side) = verify::undischarged(c, pool) {
                if !conds.contains(&side) {
                    conds.push(side);
                }
            }
        }
    }
    log_.push(RewriteStep::with_conditions(
        "prob_mutual_information_under_assumed_independence",
        zero,
        zero,
        conds,
    ));
    super::stash_prob_side_conditions(super::conditions_of(&log_));
    Ok(DerivedExpr::with_log(zero, log_))
}
