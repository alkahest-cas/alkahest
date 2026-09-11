//! The distribution table: parameter checks, supports, densities, and the
//! change of variables each continuous distribution reduces through.
//!
//! Everything here is a *definition*. Nothing in this file is derived, and
//! nothing in this file is verified — it is what the verifier checks
//! **against**. The densities are written in the untransformed variable
//! exactly as a reference would state them, and the numeric gate in
//! [`super::verify`] quadratures those, so a mistake in the reduction below
//! cannot hide behind a matching mistake in the density.

use rug::Integer;

use crate::kernel::{ExprData, ExprId, ExprPool};

use super::{div, pi, sub, DistKind, Distribution, ProbError, Support};

// ---------------------------------------------------------------------------
// Deciding a numeric constraint
// ---------------------------------------------------------------------------

/// What could be decided about the sign of a parameter expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sign {
    Positive,
    Negative,
    Zero,
    /// The expression contains a free symbol, or its enclosure straddles zero.
    /// Not decidable here, so the constraint is **carried** rather than
    /// checked — see [`Distribution::constraints`].
    Unknown,
}

/// Sign of `expr` from a rigorous enclosure, `Unknown` when the enclosure does
/// not separate it from zero.
///
/// Deliberately one-sided in the safe direction: a free symbol yields
/// `Unknown`, which makes the constructor *accept* and the constraint travel
/// with the distribution. Refusing a symbolic `σ` would be a false refusal;
/// accepting a numeric `σ = -1` would be the silent error.
pub(crate) fn sign_of(expr: ExprId, pool: &ExprPool) -> Sign {
    let Some(ball) = super::numeric_ball(expr, pool) else {
        return Sign::Unknown;
    };
    if ball.is_exact() && ball.mid == 0 {
        return Sign::Zero;
    }
    if ball.lo() > 0 {
        Sign::Positive
    } else if ball.hi() < 0 {
        Sign::Negative
    } else {
        Sign::Unknown
    }
}

pub(crate) fn require_positive(
    expr: ExprId,
    parameter: &'static str,
    pool: &ExprPool,
) -> Result<(), ProbError> {
    match sign_of(expr, pool) {
        Sign::Positive | Sign::Unknown => Ok(()),
        Sign::Negative | Sign::Zero => Err(ProbError::InvalidParameter {
            parameter,
            requirement: "> 0",
        }),
    }
}

pub(crate) fn require_less(
    a: ExprId,
    b: ExprId,
    parameter: &'static str,
    requirement: &'static str,
    pool: &ExprPool,
) -> Result<(), ProbError> {
    match sign_of(sub(b, a, pool), pool) {
        Sign::Positive | Sign::Unknown => Ok(()),
        Sign::Negative | Sign::Zero => Err(ProbError::InvalidParameter {
            parameter,
            requirement,
        }),
    }
}

pub(crate) fn require_probability(
    expr: ExprId,
    parameter: &'static str,
    pool: &ExprPool,
) -> Result<(), ProbError> {
    let bad = ProbError::InvalidParameter {
        parameter,
        requirement: "0 ≤ p ≤ 1",
    };
    if matches!(sign_of(expr, pool), Sign::Negative) {
        return Err(bad);
    }
    let one = pool.integer(1);
    if matches!(sign_of(sub(expr, one, pool), pool), Sign::Positive) {
        return Err(bad);
    }
    Ok(())
}

/// `n` must be a literal non-negative integer: every route over a finite
/// discrete support enumerates `{0…n}`, so a symbolic `n` is not something
/// this module can compute with. Refusing at construction is better than
/// accepting and then failing on every query.
pub(crate) fn require_count(
    expr: ExprId,
    parameter: &'static str,
    pool: &ExprPool,
) -> Result<(), ProbError> {
    match pool.get(expr) {
        ExprData::Integer(n) if n.0 >= 0 && n.0 <= u32::MAX => Ok(()),
        _ => Err(ProbError::InvalidParameter {
            parameter,
            requirement: "a literal non-negative integer",
        }),
    }
}

/// The literal `n` of a `Binomial`, known to exist by [`require_count`].
pub(crate) fn count_of(expr: ExprId, pool: &ExprPool) -> u32 {
    match pool.get(expr) {
        ExprData::Integer(n) => n.0.to_u32().unwrap_or(0),
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Support and constraints
// ---------------------------------------------------------------------------

pub(crate) fn support(dist: &Distribution, pool: &ExprPool) -> Support {
    let p = dist.params();
    match dist.kind() {
        DistKind::Normal => Support::Real,
        DistKind::LogNormal | DistKind::Exponential | DistKind::Gamma => Support::Positive,
        DistKind::Uniform => Support::Interval(p[0], p[1]),
        DistKind::Beta => Support::Interval(pool.integer(0), pool.integer(1)),
        DistKind::Bernoulli => Support::IntegersUpTo(1),
        DistKind::Binomial => Support::IntegersUpTo(count_of(p[0], pool)),
        DistKind::Poisson => Support::NonNegativeIntegers,
    }
}

pub(crate) fn constraints(dist: &Distribution, pool: &ExprPool) -> Vec<ExprId> {
    let p = dist.params();
    let zero = pool.integer(0);
    let one = pool.integer(1);
    match dist.kind() {
        DistKind::Normal | DistKind::LogNormal => vec![pool.pred_gt(p[1], zero)],
        DistKind::Uniform => vec![pool.pred_lt(p[0], p[1])],
        DistKind::Exponential | DistKind::Poisson => vec![pool.pred_gt(p[0], zero)],
        DistKind::Gamma | DistKind::Beta => {
            vec![pool.pred_gt(p[0], zero), pool.pred_gt(p[1], zero)]
        }
        DistKind::Bernoulli => vec![pool.pred_ge(p[0], zero), pool.pred_le(p[0], one)],
        DistKind::Binomial => vec![
            pool.pred_ge(p[0], zero),
            pool.pred_ge(p[1], zero),
            pool.pred_le(p[1], one),
        ],
    }
}

// ---------------------------------------------------------------------------
// Densities
// ---------------------------------------------------------------------------

fn exp(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("exp", vec![arg])
}

fn sqrt(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("sqrt", vec![arg])
}

fn gamma_fn(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("gamma", vec![arg])
}

/// `1/√(2π)`, the normalising constant every Gaussian route shares.
pub(crate) fn inv_sqrt_two_pi(pool: &ExprPool) -> ExprId {
    let two_pi = pool.mul(vec![pool.integer(2), pi(pool)]);
    pool.pow(sqrt(two_pi, pool), pool.integer(-1))
}

/// The standard normal density `φ(z) = e^{-z²/2}/√(2π)`.
pub(crate) fn std_normal_density(z: ExprId, pool: &ExprPool) -> ExprId {
    let z2 = pool.pow(z, pool.integer(2));
    let kernel = exp(pool.mul(vec![pool.rational(-1, 2), z2]), pool);
    pool.mul(vec![inv_sqrt_two_pi(pool), kernel])
}

pub(crate) fn pdf(dist: &Distribution, x: ExprId, pool: &ExprPool) -> ExprId {
    let p = dist.params();
    let neg1 = pool.integer(-1);
    let two = pool.integer(2);
    match dist.kind() {
        DistKind::Normal => {
            // e^{-(x-μ)²/(2σ²)} / (σ√(2π))
            let (mu, sigma) = (p[0], p[1]);
            let dev = sub(x, mu, pool);
            let num = exp(
                div(
                    pool.mul(vec![neg1, pool.pow(dev, two)]),
                    pool.mul(vec![two, pool.pow(sigma, two)]),
                    pool,
                ),
                pool,
            );
            div(
                num,
                pool.mul(vec![sigma, sqrt(pool.mul(vec![two, pi(pool)]), pool)]),
                pool,
            )
        }
        DistKind::LogNormal => {
            // e^{-(log x - μ)²/(2σ²)} / (x σ√(2π))
            let (mu, sigma) = (p[0], p[1]);
            let dev = sub(pool.func("log", vec![x]), mu, pool);
            let num = exp(
                div(
                    pool.mul(vec![neg1, pool.pow(dev, two)]),
                    pool.mul(vec![two, pool.pow(sigma, two)]),
                    pool,
                ),
                pool,
            );
            div(
                num,
                pool.mul(vec![x, sigma, sqrt(pool.mul(vec![two, pi(pool)]), pool)]),
                pool,
            )
        }
        DistKind::Uniform => pool.pow(sub(p[1], p[0], pool), neg1),
        DistKind::Exponential => {
            let lambda = p[0];
            pool.mul(vec![lambda, exp(pool.mul(vec![neg1, lambda, x]), pool)])
        }
        DistKind::Gamma => {
            // x^{k-1} e^{-x/θ} / (Γ(k) θ^k)
            let (k, theta) = (p[0], p[1]);
            let num = pool.mul(vec![
                pool.pow(x, pool.add(vec![k, neg1])),
                exp(pool.mul(vec![neg1, div(x, theta, pool)]), pool),
            ]);
            div(
                num,
                pool.mul(vec![gamma_fn(k, pool), pool.pow(theta, k)]),
                pool,
            )
        }
        DistKind::Beta => {
            // x^{α-1}(1-x)^{β-1} Γ(α+β)/(Γ(α)Γ(β))
            let (a, b) = (p[0], p[1]);
            let one_minus_x = sub(pool.integer(1), x, pool);
            let core = pool.mul(vec![
                pool.pow(x, pool.add(vec![a, neg1])),
                pool.pow(one_minus_x, pool.add(vec![b, neg1])),
            ]);
            let norm = div(
                gamma_fn(pool.add(vec![a, b]), pool),
                pool.mul(vec![gamma_fn(a, pool), gamma_fn(b, pool)]),
                pool,
            );
            pool.mul(vec![norm, core])
        }
        DistKind::Bernoulli => {
            // p^x (1-p)^{1-x}
            let prob = p[0];
            let q = sub(pool.integer(1), prob, pool);
            pool.mul(vec![
                pool.pow(prob, x),
                pool.pow(q, sub(pool.integer(1), x, pool)),
            ])
        }
        DistKind::Binomial => {
            // C(n,x) p^x (1-p)^{n-x}
            let (n, prob) = (p[0], p[1]);
            let q = sub(pool.integer(1), prob, pool);
            let coeff = binomial_coefficient(n, x, pool);
            pool.mul(vec![coeff, pool.pow(prob, x), pool.pow(q, sub(n, x, pool))])
        }
        DistKind::Poisson => {
            // λ^x e^{-λ} / Γ(x+1)
            let lambda = p[0];
            let num = pool.mul(vec![
                pool.pow(lambda, x),
                exp(pool.mul(vec![neg1, lambda]), pool),
            ]);
            div(
                num,
                gamma_fn(pool.add(vec![x, pool.integer(1)]), pool),
                pool,
            )
        }
    }
}

/// `C(n, k)` — an exact integer when `k` is a literal, otherwise the `Γ`
/// expression. The literal path matters: every discrete route enumerates the
/// support, and `Γ(6)/(Γ(3)Γ(4))` is a float round-trip where `10` is exact.
fn binomial_coefficient(n: ExprId, k: ExprId, pool: &ExprPool) -> ExprId {
    if let (ExprData::Integer(nn), ExprData::Integer(kk)) = (pool.get(n), pool.get(k)) {
        if let (Some(nn), Some(kk)) = (nn.0.to_u32(), kk.0.to_u32()) {
            if kk <= nn {
                return pool.integer(Integer::from(nn).binomial(kk));
            }
            return pool.integer(0);
        }
    }
    let one = pool.integer(1);
    div(
        gamma_fn(pool.add(vec![n, one]), pool),
        pool.mul(vec![
            gamma_fn(pool.add(vec![k, one]), pool),
            gamma_fn(pool.add(vec![sub(n, k, pool), one]), pool),
        ]),
        pool,
    )
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// The change of variables a continuous distribution integrates through.
///
/// `x = ξ(z)` is **strictly increasing** for every distribution here — that is
/// what lets an integration bound in `x` be mapped to one in `z` by applying
/// `ζ = ξ⁻¹` and keeping the orientation. The monotonicity follows from the
/// distribution's own constraints (`σ > 0`, `λ > 0`, `θ > 0`), which is why
/// those constraints are checked or carried rather than assumed away.
pub(crate) struct Reduction {
    /// The reduction variable.
    pub z: ExprId,
    /// `ξ(z)`: the original variable as a function of `z`.
    pub x_of_z: ExprId,
    /// `ζ(x)`: the inverse, as an expression in the caller's `x`.
    pub z_of_x: ExprId,
    /// `p(ξ(z))·ξ'(z)` — the density of `z`, written directly.
    pub density: ExprId,
    /// Lower `z`-bound of the whole support (`-∞` allowed).
    pub lo: ExprId,
    /// Upper `z`-bound of the whole support (`∞` allowed).
    pub hi: ExprId,
}

/// A symbol name not occurring in `expr`, so the reduction variable cannot
/// capture one of the caller's.
pub(crate) fn fresh_var(exprs: &[ExprId], pool: &ExprPool) -> ExprId {
    for n in 0..64u32 {
        let name = if n == 0 {
            "_prob_z".to_string()
        } else {
            format!("_prob_z{n}")
        };
        let cand = pool.symbol(&name, crate::kernel::Domain::Real);
        if !exprs.iter().any(|&e| contains(e, cand, pool)) {
            return cand;
        }
    }
    pool.symbol("_prob_z63", crate::kernel::Domain::Real)
}

pub(crate) fn contains(expr: ExprId, target: ExprId, pool: &ExprPool) -> bool {
    if expr == target {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().any(|&a| contains(a, target, pool))
        }
        ExprData::Pow { base, exp } => contains(base, target, pool) || contains(exp, target, pool),
        _ => false,
    }
}

/// The reduction for a continuous distribution, or `None` for a discrete one.
pub(crate) fn reduction(dist: &Distribution, x: ExprId, pool: &ExprPool) -> Option<Reduction> {
    let p = dist.params();
    let z = fresh_var(&[x, p[0], *p.get(1).unwrap_or(&p[0])], pool);
    let inf = pool.pos_infinity();
    let neg_inf = pool.mul(vec![pool.integer(-1), inf]);
    let zero = pool.integer(0);
    let neg1 = pool.integer(-1);
    Some(match dist.kind() {
        DistKind::Normal => {
            let (mu, sigma) = (p[0], p[1]);
            Reduction {
                z,
                x_of_z: pool.add(vec![mu, pool.mul(vec![sigma, z])]),
                z_of_x: div(sub(x, mu, pool), sigma, pool),
                density: std_normal_density(z, pool),
                lo: neg_inf,
                hi: inf,
            }
        }
        DistKind::LogNormal => {
            let (mu, sigma) = (p[0], p[1]);
            Reduction {
                z,
                x_of_z: exp(pool.add(vec![mu, pool.mul(vec![sigma, z])]), pool),
                z_of_x: div(sub(pool.func("log", vec![x]), mu, pool), sigma, pool),
                density: std_normal_density(z, pool),
                lo: neg_inf,
                hi: inf,
            }
        }
        DistKind::Uniform => Reduction {
            z,
            x_of_z: z,
            z_of_x: x,
            density: pool.pow(sub(p[1], p[0], pool), neg1),
            lo: p[0],
            hi: p[1],
        },
        DistKind::Exponential => {
            let lambda = p[0];
            Reduction {
                z,
                x_of_z: div(z, lambda, pool),
                z_of_x: pool.mul(vec![lambda, x]),
                density: exp(pool.mul(vec![neg1, z]), pool),
                lo: zero,
                hi: inf,
            }
        }
        DistKind::Gamma => {
            let (k, theta) = (p[0], p[1]);
            Reduction {
                z,
                x_of_z: pool.mul(vec![theta, z]),
                z_of_x: div(x, theta, pool),
                density: div(
                    pool.mul(vec![
                        pool.pow(z, pool.add(vec![k, neg1])),
                        exp(pool.mul(vec![neg1, z]), pool),
                    ]),
                    gamma_fn(k, pool),
                    pool,
                ),
                lo: zero,
                hi: inf,
            }
        }
        DistKind::Beta => Reduction {
            z,
            x_of_z: z,
            z_of_x: x,
            density: pdf(dist, z, pool),
            lo: zero,
            hi: pool.integer(1),
        },
        DistKind::Bernoulli | DistKind::Binomial | DistKind::Poisson => return None,
    })
}
