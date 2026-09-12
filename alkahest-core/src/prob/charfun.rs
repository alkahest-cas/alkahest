//! Characteristic functions, `φ_X(t) = E[e^{itX}]`.
//!
//! # Why `φ` is a first-class attribute here and not a wrapper round the pdf
//!
//! `φ` and the density are two *independent* descriptions of a law, and the
//! implication only runs one way in practice. Every distribution in this module
//! has a density, so it would be possible to define `φ` as "the Fourier
//! transform of `pdf`" — and that definition would be a dead end the moment
//! anything harder arrives. A Lévy process is *defined* by its characteristic
//! exponent (Lévy–Khintchine); variance-gamma and Heston have closed-form `φ`
//! and no closed-form density at all, which is the entire reason Fourier option
//! pricing exists. A `φ` that can only be reached through a pdf cannot describe
//! any of them.
//!
//! So [`characteristic_function`] reads a table keyed on the distribution, in
//! the same way [`super::moments`] does. Adding a law that has `φ` and no `p`
//! is then an entry in that table, not a redesign.
//!
//! # Convention, pinned
//!
//! `φ_X(t) = E[e^{itX}] = ∫ p(x)e^{itx}dx`, while
//! [`crate::transform::fourier_transform`] is unitary ordinary-frequency,
//! `F{p}(ξ) = ∫p(x)e^{-2πiξx}dx`. Matching the exponents gives
//!
//! ```text
//! φ_X(t) = F{p}(-t / 2π)
//! ```
//!
//! — a `2π` scaling **and** a sign flip. Getting either half wrong produces a
//! function that is still smooth, still equal to `1` at `t = 0`, and still has
//! the right modulus for a symmetric law, so it would survive every casual
//! check. It is therefore not left as a comment: `verify::check_characteristic`
//! computes `F{p}(-t/2π)` through the actual transform, wherever the transform
//! has a rule for the density, and compares it against the table entry
//! numerically. A convention slip fails that comparison.
//!
//! # Inversion — read this before assuming a round trip
//!
//! There is **none** here. Recovering a density or an option price from `φ`
//! is an oscillatory integral along a contour
//! (`p(x) = (1/2π)∫φ(t)e^{-itx}dt`, or the Carr–Madan / Gil-Pelaez variants),
//! and [`crate::transform::inverse_fourier_transform`] is a small rule table,
//! not a contour integrator: it inverts what its table lists and declines the
//! rest. Nothing in this module calls it, and nothing here will turn a `φ` back
//! into a density or a price. `φ` is provided as a computed, verified object —
//! useful for moments (see [`super::moments`] and the `φ⁽ⁿ⁾(0) = iⁿE[Xⁿ]`
//! cross-check in the tests), for cumulants, and as the input to a numerical
//! inversion the caller supplies. It is not half of a round trip.
//!
//! # What refuses
//!
//! | law | why |
//! |---|---|
//! | `LogNormal` | `φ` has **no closed form** — not "not implemented": the series `Σ (it)ⁿe^{nμ+n²σ²/2}/n!` diverges for every `t ≠ 0` and the function is not expressible in elementary or standard special functions. `E-PROB-004` |
//! | `Beta` | `φ = ₁F₁(α; α+β; it)`, the confluent hypergeometric function, which this library does not have. `E-PROB-004` |

use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::simplify;

use super::dists;
use super::verify;
use super::{div, sub, DistKind, Distribution, ProbError};

/// `φ_X(t)` in closed form, verified.
///
/// # Errors
///
/// [`ProbError::NoClosedForm`] for `LogNormal` and `Beta` (see the module
/// docs), and [`ProbError::Unverified`] if the numeric gate could not confirm
/// the table entry.
pub fn characteristic_function(
    dist: &Distribution,
    t: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, ProbError> {
    super::stash_prob_side_conditions(Vec::new());
    let claim = simplify(claim_for(dist, t, pool)?, pool).value;
    let (evidence, fourier_agreed) = verify::check_characteristic(claim, t, dist, pool)?;
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "prob_characteristic_function",
        t,
        claim,
    ));
    if fourier_agreed {
        log.push(RewriteStep::simple(
            "prob_charfun_matches_fourier_transform_at_minus_t_over_two_pi",
            claim,
            claim,
        ));
    }
    log.push(verify::evidence_step(&evidence, claim, dist, pool));
    super::stash_prob_side_conditions(super::conditions_of(&log));
    Ok(DerivedExpr::with_log(claim, log))
}

/// `i`, the imaginary unit, as the kernel spells it.
fn imag(pool: &ExprPool) -> ExprId {
    pool.imaginary_unit()
}

fn exp(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("exp", vec![arg])
}

/// The table. One entry per law, written the way a reference writes it.
fn claim_for(dist: &Distribution, t: ExprId, pool: &ExprPool) -> Result<ExprId, ProbError> {
    let p = dist.params();
    let i = imag(pool);
    let one = pool.integer(1);
    let two = pool.integer(2);
    let neg1 = pool.integer(-1);
    Ok(match dist.kind() {
        // e^{iμt - σ²t²/2}
        DistKind::Normal => exp(
            pool.add(vec![
                pool.mul(vec![i, p[0], t]),
                pool.mul(vec![
                    pool.rational(-1, 2),
                    pool.pow(p[1], two),
                    pool.pow(t, two),
                ]),
            ]),
            pool,
        ),
        // (e^{ibt} - e^{iat}) / (it(b-a))
        DistKind::Uniform => div(
            sub(
                exp(pool.mul(vec![i, p[1], t]), pool),
                exp(pool.mul(vec![i, p[0], t]), pool),
                pool,
            ),
            pool.mul(vec![i, t, sub(p[1], p[0], pool)]),
            pool,
        ),
        // λ/(λ - it)
        DistKind::Exponential => div(p[0], sub(p[0], pool.mul(vec![i, t]), pool), pool),
        // (1 - iθt)^{-k}
        DistKind::Gamma => pool.pow(
            sub(one, pool.mul(vec![i, p[1], t]), pool),
            pool.mul(vec![neg1, p[0]]),
        ),
        // 1 - p + p e^{it}
        DistKind::Bernoulli => pool.add(vec![
            sub(one, p[0], pool),
            pool.mul(vec![p[0], exp(pool.mul(vec![i, t]), pool)]),
        ]),
        // (1 - p + p e^{it})^n
        DistKind::Binomial => pool.pow(
            pool.add(vec![
                sub(one, p[1], pool),
                pool.mul(vec![p[1], exp(pool.mul(vec![i, t]), pool)]),
            ]),
            p[0],
        ),
        // e^{λ(e^{it} - 1)}
        DistKind::Poisson => exp(
            pool.mul(vec![p[0], sub(exp(pool.mul(vec![i, t]), pool), one, pool)]),
            pool,
        ),
        DistKind::LogNormal => {
            return Err(ProbError::NoClosedForm {
                quantity: "the log-normal characteristic function",
                missing: "nothing this library could add — φ for a log-normal has no closed \
                          form in elementary or standard special functions, and its moment \
                          series diverges for every t ≠ 0",
            })
        }
        DistKind::Beta => {
            return Err(ProbError::NoClosedForm {
                quantity: "the Beta characteristic function",
                missing: "the confluent hypergeometric function ₁F₁(α; α+β; it)",
            })
        }
    })
}

/// The density extended by zero off the support, in the form
/// [`crate::transform::fourier_transform`] expects — or `None` where no
/// spelling of it is in that table.
///
/// Only used as an independent cross-check of the table entry above, so
/// `None` costs a piece of evidence and nothing else.
pub(crate) fn fourier_ready_density(
    dist: &Distribution,
    x: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    match dist.kind() {
        // Already supported on the whole line.
        DistKind::Normal => Some(dists::pdf(dist, x, pool)),
        // `θ(x)·λe^{-λx}` is the transform table's one-sided exponential.
        DistKind::Exponential => Some(pool.mul(vec![
            pool.func("heaviside", vec![x]),
            dists::pdf(dist, x, pool),
        ])),
        _ => None,
    }
}
