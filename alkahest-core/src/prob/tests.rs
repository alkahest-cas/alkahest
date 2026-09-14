//! Tests for the probability layer.
//!
//! Every expected value in this file comes from somewhere that is not
//! alkahest: a standard reference identity (`E[X²] = μ² + σ²`), a value
//! computed by hand, or a constant produced by `mpmath`/`sympy.stats` and
//! pasted in as a literal. Checking alkahest against alkahest measures
//! self-consistency, and self-consistency is exactly what a systematic error
//! preserves.

use rug::Float;

use super::quad::{eval_ball, VERIFY_PREC};
use super::*;
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::simplify::simplify;

fn pool() -> ExprPool {
    ExprPool::new()
}

fn sym(p: &ExprPool, name: &str) -> ExprId {
    p.symbol(name, Domain::Real)
}

/// Evaluate an expression at a named-symbol environment.
fn at(p: &ExprPool, e: ExprId, env: &[(&str, f64)]) -> f64 {
    let bindings: Vec<(ExprId, Float)> = env
        .iter()
        .map(|(n, v)| (sym(p, n), Float::with_val(VERIFY_PREC, *v)))
        .collect();
    eval_ball(e, &bindings, p, VERIFY_PREC)
        .unwrap_or_else(|| panic!("{} did not evaluate", p.display(e)))
        .mid
        .to_f64()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-11 * a.abs().max(b.abs()).max(1.0)
}

macro_rules! assert_close {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        assert!(close(a, b), "{a} != {b}");
    }};
}

// ---------------------------------------------------------------------------
// Construction: numeric constraints are decided, symbolic ones are carried
// ---------------------------------------------------------------------------

#[test]
fn a_numeric_parameter_outside_its_constraint_is_refused() {
    let p = pool();
    let mu = sym(&p, "mu");
    for bad in [p.integer(0), p.integer(-1), p.rational(-1, 2)] {
        let e = Distribution::normal(mu, bad, &p).unwrap_err();
        assert_eq!(e.code(), "E-PROB-001");
    }
    assert_eq!(
        Distribution::uniform(p.integer(1), p.integer(0), &p)
            .unwrap_err()
            .code(),
        "E-PROB-001"
    );
    assert_eq!(
        Distribution::beta(p.integer(-1), p.integer(2), &p)
            .unwrap_err()
            .code(),
        "E-PROB-001"
    );
    assert_eq!(
        Distribution::bernoulli(p.rational(3, 2), &p)
            .unwrap_err()
            .code(),
        "E-PROB-001"
    );
    // `n` must be a literal count: a symbolic one is refused at construction
    // rather than accepted and then failing on every query.
    assert_eq!(
        Distribution::binomial(sym(&p, "n"), p.rational(1, 2), &p)
            .unwrap_err()
            .code(),
        "E-PROB-001"
    );
}

#[test]
fn a_symbolic_parameter_is_carried_not_decided() {
    let p = pool();
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let d = Distribution::normal(mu, sigma, &p).expect("symbolic sigma is not decidable");
    let cs = d.constraints(&p);
    assert_eq!(cs.len(), 1);
    // It is `σ > 0`, as a predicate the caller can read and discharge.
    assert_eq!(format!("{}", p.display(cs[0])), "(sigma > 0)");
}

// ---------------------------------------------------------------------------
// Means and variances, against reference identities
// ---------------------------------------------------------------------------

#[test]
fn normal_mean_and_variance() {
    let p = pool();
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let d = Distribution::normal(mu, sigma, &p).unwrap();
    assert_eq!(d.mean(&p).unwrap().value, mu);
    let v = d.variance(&p).unwrap().value;
    assert_close!(at(&p, v, &[("sigma", 1.7), ("mu", 0.4)]), 1.7 * 1.7);
}

#[test]
fn log_normal_mean_is_exp_mu_plus_half_sigma_squared() {
    let p = pool();
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let d = Distribution::log_normal(mu, sigma, &p).unwrap();
    let m = d.mean(&p).unwrap().value;
    // mpmath: exp(0.3 + 0.5*0.8**2)
    assert_close!(
        at(&p, m, &[("mu", 0.3), ("sigma", 0.8)]),
        (0.3f64 + 0.5 * 0.64).exp()
    );
    let v = d.variance(&p).unwrap().value;
    // (e^{σ²}-1)e^{2μ+σ²}
    let expected = ((0.64f64).exp() - 1.0) * (2.0 * 0.3 + 0.64f64).exp();
    assert_close!(at(&p, v, &[("mu", 0.3), ("sigma", 0.8)]), expected);
}

#[test]
fn uniform_exponential_gamma_beta_means() {
    let p = pool();
    let (a, b) = (sym(&p, "a"), sym(&p, "b"));
    let u = Distribution::uniform(a, b, &p).unwrap();
    assert_close!(
        at(&p, u.mean(&p).unwrap().value, &[("a", -2.0), ("b", 3.0)]),
        0.5
    );
    assert_close!(
        at(
            &p,
            u.variance(&p).unwrap().value,
            &[("a", -2.0), ("b", 3.0)]
        ),
        25.0 / 12.0
    );

    let lam = sym(&p, "lambda");
    let e = Distribution::exponential(lam, &p).unwrap();
    assert_close!(at(&p, e.mean(&p).unwrap().value, &[("lambda", 2.5)]), 0.4);
    assert_close!(
        at(&p, e.variance(&p).unwrap().value, &[("lambda", 2.5)]),
        0.16
    );

    let (k, theta) = (sym(&p, "k"), sym(&p, "theta"));
    let g = Distribution::gamma(k, theta, &p).unwrap();
    assert_close!(
        at(&p, g.mean(&p).unwrap().value, &[("k", 3.5), ("theta", 0.7)]),
        3.5 * 0.7
    );
    assert_close!(
        at(
            &p,
            g.variance(&p).unwrap().value,
            &[("k", 3.5), ("theta", 0.7)]
        ),
        3.5 * 0.49
    );

    let (al, be) = (sym(&p, "alpha"), sym(&p, "beta"));
    let bd = Distribution::beta(al, be, &p).unwrap();
    assert_close!(
        at(
            &p,
            bd.mean(&p).unwrap().value,
            &[("alpha", 2.0), ("beta", 3.0)]
        ),
        0.4
    );
    assert_close!(
        at(
            &p,
            bd.variance(&p).unwrap().value,
            &[("alpha", 2.0), ("beta", 3.0)]
        ),
        6.0 / (25.0 * 6.0)
    );
}

#[test]
fn discrete_means_and_variances() {
    let p = pool();
    let prob = sym(&p, "p");
    let b = Distribution::bernoulli(prob, &p).unwrap();
    assert_eq!(b.mean(&p).unwrap().value, prob);
    assert_close!(at(&p, b.variance(&p).unwrap().value, &[("p", 0.3)]), 0.21);

    let bin = Distribution::binomial(p.integer(5), prob, &p).unwrap();
    assert_close!(at(&p, bin.mean(&p).unwrap().value, &[("p", 0.3)]), 1.5);
    assert_close!(at(&p, bin.variance(&p).unwrap().value, &[("p", 0.3)]), 1.05);

    let lam = sym(&p, "lambda");
    let po = Distribution::poisson(lam, &p).unwrap();
    assert_close!(at(&p, po.mean(&p).unwrap().value, &[("lambda", 2.5)]), 2.5);
    assert_close!(
        at(&p, po.variance(&p).unwrap().value, &[("lambda", 2.5)]),
        2.5
    );
}

// ---------------------------------------------------------------------------
// Raw moments, against values worked out independently
// ---------------------------------------------------------------------------

#[test]
fn standard_normal_moments_are_the_double_factorials() {
    let p = pool();
    let d = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    // E[X^{2n}] = (2n-1)!! for a standard normal; odd moments vanish.
    for (n, expected) in [
        (1u32, 0.0),
        (2, 1.0),
        (3, 0.0),
        (4, 3.0),
        (6, 15.0),
        (8, 105.0),
    ] {
        let m = d.moment(n, &p).unwrap().value;
        assert_close!(at(&p, m, &[]), expected);
    }
}

#[test]
fn non_central_normal_second_moment_is_mu_squared_plus_sigma_squared() {
    let p = pool();
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let d = Distribution::normal(mu, sigma, &p).unwrap();
    let m2 = d.moment(2, &p).unwrap().value;
    assert_close!(at(&p, m2, &[("mu", 1.3), ("sigma", 0.7)]), 1.3 * 1.3 + 0.49);
    // E[X³] = μ³ + 3μσ²
    let m3 = d.moment(3, &p).unwrap().value;
    assert_close!(
        at(&p, m3, &[("mu", 1.3), ("sigma", 0.7)]),
        1.3f64.powi(3) + 3.0 * 1.3 * 0.49
    );
}

#[test]
fn poisson_third_moment_is_the_touchard_polynomial() {
    let p = pool();
    let lam = sym(&p, "lambda");
    let d = Distribution::poisson(lam, &p).unwrap();
    // E[X³] = λ³ + 3λ² + λ
    let m3 = d.moment(3, &p).unwrap().value;
    let l = 2.5f64;
    assert_close!(at(&p, m3, &[("lambda", l)]), l * l * l + 3.0 * l * l + l);
}

#[test]
fn exponential_and_gamma_moments() {
    let p = pool();
    let lam = sym(&p, "lambda");
    let e = Distribution::exponential(lam, &p).unwrap();
    // E[Xⁿ] = n!/λⁿ
    assert_close!(
        at(&p, e.moment(4, &p).unwrap().value, &[("lambda", 1.75)]),
        24.0 / 1.75f64.powi(4)
    );
    let (k, theta) = (sym(&p, "k"), sym(&p, "theta"));
    let g = Distribution::gamma(k, theta, &p).unwrap();
    // E[X³] = θ³·k(k+1)(k+2)
    assert_close!(
        at(
            &p,
            g.moment(3, &p).unwrap().value,
            &[("k", 2.0), ("theta", 1.5)]
        ),
        1.5f64.powi(3) * 2.0 * 3.0 * 4.0
    );
}

#[test]
fn moment_order_past_the_verifiable_range_is_refused() {
    let p = pool();
    let d = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let e = d.moment(MAX_MOMENT_ORDER + 1, &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-002");
}

// ---------------------------------------------------------------------------
// CDFs
// ---------------------------------------------------------------------------

#[test]
fn normal_cdf_is_the_error_function_form() {
    let p = pool();
    let (mu, sigma, x) = (sym(&p, "mu"), sym(&p, "sigma"), sym(&p, "x"));
    let d = Distribution::normal(mu, sigma, &p).unwrap();
    let f = d.cdf(x, &p).unwrap().value;
    // mpmath (40 dps): ncdf(2.5, 1, 2) = 0.773372647623131800672…
    assert_close!(
        at(&p, f, &[("mu", 1.0), ("sigma", 2.0), ("x", 2.5)]),
        0.773_372_647_623_131_8
    );
    assert_close!(at(&p, f, &[("mu", 0.0), ("sigma", 1.0), ("x", 0.0)]), 0.5);
}

#[test]
fn log_normal_and_exponential_cdfs() {
    let p = pool();
    let x = sym(&p, "x");
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let ln = Distribution::log_normal(mu, sigma, &p).unwrap();
    let f = ln.cdf(x, &p).unwrap().value;
    // mpmath (40 dps): ncdf((log(1.5) - 0.2)/0.5) = 0.659438147374726910788…
    assert_close!(
        at(&p, f, &[("mu", 0.2), ("sigma", 0.5), ("x", 1.5)]),
        0.659_438_147_374_727
    );

    let lam = sym(&p, "lambda");
    let e = Distribution::exponential(lam, &p).unwrap();
    let f = e.cdf(x, &p).unwrap().value;
    // 1 - e^{-1.75·0.8}
    assert_close!(
        at(&p, f, &[("lambda", 1.75), ("x", 0.8)]),
        1.0 - (-1.4f64).exp()
    );
}

#[test]
fn erlang_cdf_closes_and_the_general_gamma_cdf_refuses() {
    let p = pool();
    let x = sym(&p, "x");
    let theta = sym(&p, "theta");
    let erlang = Distribution::gamma(p.integer(3), theta, &p).unwrap();
    let f = erlang.cdf(x, &p).unwrap().value;
    // mpmath (40 dps): 1 - e^{-5/2}(1 + 5/2 + (5/2)²/2) = 0.456186884116670482…
    assert_close!(
        at(&p, f, &[("theta", 2.0), ("x", 5.0)]),
        0.456_186_884_116_670_5
    );

    let k = sym(&p, "k");
    let g = Distribution::gamma(k, theta, &p).unwrap();
    let e = g.cdf(x, &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-004");
    assert!(format!("{e}").contains("incomplete gamma"));
}

#[test]
fn integer_beta_cdf_closes_and_the_general_one_refuses() {
    let p = pool();
    let x = sym(&p, "x");
    let bd = Distribution::beta(p.integer(2), p.integer(3), &p).unwrap();
    let f = bd.cdf(x, &p).unwrap().value;
    // mpmath (40 dps): ∫₀^{0.4} 12t(1-t)² dt = 0.5248 exactly
    assert_close!(at(&p, f, &[("x", 0.4)]), 0.5248);

    let general = Distribution::beta(sym(&p, "alpha"), p.integer(3), &p).unwrap();
    assert_eq!(general.cdf(x, &p).unwrap_err().code(), "E-PROB-004");
}

#[test]
fn uniform_cdf_is_linear() {
    let p = pool();
    let (a, b, x) = (sym(&p, "a"), sym(&p, "b"), sym(&p, "x"));
    let u = Distribution::uniform(a, b, &p).unwrap();
    let f = u.cdf(x, &p).unwrap().value;
    assert_close!(at(&p, f, &[("a", -2.0), ("b", 3.0), ("x", 0.5)]), 0.5);
}

/// A CDF is a probability, and the in-support closed form is not one off the
/// support — it is a clean wrong number. `Erlang(3, 4/5)` at `x = -5` evaluates
/// to `-7396.87`, which is exactly the shape of answer this library exists not
/// to give.
#[test]
fn a_cdf_below_or_above_the_support_is_zero_or_one_not_the_formula() {
    let p = pool();

    // Below a support that starts at 0.
    let g = Distribution::gamma(p.integer(3), p.rational(4, 5), &p).unwrap();
    for below in [-1, -5] {
        let f = g.cdf(p.integer(below), &p).unwrap().value;
        assert_eq!(at(&p, f, &[]), 0.0, "Erlang CDF at x = {below}");
    }

    let e = Distribution::exponential(p.rational(7, 4), &p).unwrap();
    assert_eq!(at(&p, e.cdf(p.integer(-3), &p).unwrap().value, &[]), 0.0);

    // Both ends of a compact support.
    let u = Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap();
    assert_eq!(at(&p, u.cdf(p.integer(-4), &p).unwrap().value, &[]), 0.0);
    assert_eq!(at(&p, u.cdf(p.integer(7), &p).unwrap().value, &[]), 1.0);

    let b = Distribution::beta(p.integer(2), p.integer(3), &p).unwrap();
    assert_eq!(
        at(&p, b.cdf(p.rational(-1, 2), &p).unwrap().value, &[]),
        0.0
    );
    assert_eq!(at(&p, b.cdf(p.rational(3, 2), &p).unwrap().value, &[]), 1.0);

    // The step says which branch was taken, so a reader of the log can tell
    // "outside the support" from "the formula happened to give 0".
    let r = g.cdf(p.integer(-5), &p).unwrap();
    assert!(r
        .log
        .0
        .iter()
        .any(|s| s.rule_name == "prob_cdf_outside_support"));
}

/// A *literal* in-support argument is verified and returned, not refused. The
/// CDF is checked as a function of a fresh symbol — over a ladder of abscissae
/// — and the caller's point substituted afterwards, because a claim with no
/// free variable gives the checker nothing to vary.
#[test]
fn a_cdf_at_a_literal_point_inside_the_support_is_verified_and_returned() {
    let p = pool();
    let g = Distribution::gamma(p.integer(3), p.rational(4, 5), &p).unwrap();
    let f = g.cdf(p.rational(12, 5), &p).unwrap();
    // mpmath (40 dps): quad(gamma pdf k=3 theta=4/5, [0, 12/5]) =
    // 0.57680991887315648468…
    assert_close!(at(&p, f.value, &[]), 0.576_809_918_873_156_5);
    assert!(f.log.0.iter().any(|s| s.rule_name == "prob_cdf"));

    let u = Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap();
    assert_close!(at(&p, u.cdf(p.rational(1, 2), &p).unwrap().value, &[]), 0.5);
}

/// A symbolic argument cannot be placed, so the in-support restriction travels
/// with the answer instead of being assumed.
#[test]
fn a_symbolic_cdf_argument_carries_the_in_support_condition() {
    let p = pool();
    let x = sym(&p, "x");

    let e = Distribution::exponential(sym(&p, "lambda"), &p).unwrap();
    let r = e.cdf(x, &p).unwrap();
    assert!(r.log.0.iter().any(
        |s| s.rule_name == "prob_cdf_argument_inside_support" && !s.side_conditions.is_empty()
    ));

    // A compact support needs both ends recorded, not just one.
    let u = Distribution::uniform(sym(&p, "a"), sym(&p, "b"), &p).unwrap();
    let r = u.cdf(x, &p).unwrap();
    let conds: usize = r
        .log
        .0
        .iter()
        .filter(|s| s.rule_name == "prob_cdf_argument_inside_support")
        .map(|s| s.side_conditions.len())
        .sum();
    assert_eq!(conds, 2);

    // A normal's support is the whole line: nothing to restrict, so nothing is
    // claimed. A condition that is always true is noise, not disclosure.
    let n = Distribution::normal(sym(&p, "mu"), sym(&p, "sigma"), &p).unwrap();
    assert!(!n
        .cdf(x, &p)
        .unwrap()
        .log
        .0
        .iter()
        .any(|s| s.rule_name == "prob_cdf_argument_inside_support"));
}

/// The out-of-band channel. The value is an `ExprId`, so a hypothesis has
/// nowhere in band to live; without this a conditional answer and a theorem are
/// indistinguishable at the call site.
///
/// Two *different* classes of hypothesis travel on it, and both matter:
/// an undischarged **parameter** constraint (`σ > 0` for a symbolic scale) and
/// an undischarged **argument** placement (`x` inside the support). The first
/// comes from `evidence_step`, the second from `cdf`/`quantile`/`expectation`.
#[test]
fn undischarged_hypotheses_are_readable_out_of_band() {
    use crate::prob::take_prob_side_conditions;

    let p = pool();
    let x = sym(&p, "x");

    // Literal parameter, symbolic argument: the *only* open hypothesis is that
    // `x` is in the support.
    let e = Distribution::exponential(p.rational(7, 4), &p).unwrap();
    let _ = e.cdf(x, &p).unwrap();
    assert_eq!(
        take_prob_side_conditions().len(),
        1,
        "a symbolic cdf argument over a literal-rate exponential publishes x > 0"
    );

    // Consuming: a second read of the same call is empty, so one call's
    // hypotheses cannot be re-read as a later call's.
    assert!(take_prob_side_conditions().is_empty());

    // Symbolic parameter *and* symbolic argument: both classes are published.
    let e = Distribution::exponential(sym(&p, "lambda"), &p).unwrap();
    let _ = e.cdf(x, &p).unwrap();
    assert_eq!(
        take_prob_side_conditions().len(),
        2,
        "lambda > 0 and x > 0 are both still open"
    );

    // A compact support publishes both ends.
    let u = Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap();
    let _ = u.cdf(x, &p).unwrap();
    assert_eq!(take_prob_side_conditions().len(), 2);

    // A decidable argument is answered exactly and claims nothing.
    let g = Distribution::gamma(p.integer(3), p.rational(4, 5), &p).unwrap();
    let _ = g.cdf(p.integer(-5), &p).unwrap();
    assert!(take_prob_side_conditions().is_empty());
    let _ = g.cdf(p.rational(12, 5), &p).unwrap();
    assert!(take_prob_side_conditions().is_empty());

    // A normal's support is the whole line, so with literal parameters there is
    // nothing left to assume at all.
    let n = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let _ = n.cdf(x, &p).unwrap();
    assert!(take_prob_side_conditions().is_empty());

    // A symbolic quantile argument publishes 0 <= p <= 1.
    let q = sym(&p, "q");
    let _ = u.quantile(q, &p).unwrap();
    assert_eq!(take_prob_side_conditions().len(), 2);

    // A refusal *clears* the channel rather than leaving the previous call's
    // hypotheses readable as its own. Prime the channel, then refuse.
    let _ = u.cdf(x, &p).unwrap();
    let _ = n
        .quantile(q, &p)
        .expect_err("the normal quantile needs erf-inverse");
    assert!(
        take_prob_side_conditions().is_empty(),
        "a refusal must not leave a stale hypothesis behind"
    );

    // And a route that records nothing clears it too.
    let _ = u.cdf(x, &p).unwrap();
    let _ = n.mean(&p).unwrap();
    assert!(take_prob_side_conditions().is_empty());
}

// ---------------------------------------------------------------------------
// Quantiles
// ---------------------------------------------------------------------------

#[test]
fn uniform_and_exponential_quantiles_close() {
    let p = pool();
    let (a, b, q) = (sym(&p, "a"), sym(&p, "b"), sym(&p, "q"));
    let u = Distribution::uniform(a, b, &p).unwrap();
    let f = u.quantile(q, &p).unwrap().value;
    assert_close!(at(&p, f, &[("a", -2.0), ("b", 3.0), ("q", 0.25)]), -0.75);

    let lam = sym(&p, "lambda");
    let e = Distribution::exponential(lam, &p).unwrap();
    let f = e.quantile(q, &p).unwrap().value;
    // -log(1-0.75)/2 = log(4)/2
    assert_close!(at(&p, f, &[("lambda", 2.0), ("q", 0.75)]), 4f64.ln() / 2.0);
}

#[test]
fn the_normal_quantile_refuses_for_want_of_an_inverse_error_function() {
    let p = pool();
    let q = sym(&p, "q");
    let d = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let e = d.quantile(q, &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-004");
    assert!(format!("{e}").contains("erf"));

    let g = Distribution::gamma(p.integer(2), p.integer(1), &p).unwrap();
    assert_eq!(g.quantile(q, &p).unwrap_err().code(), "E-PROB-004");
    let po = Distribution::poisson(p.integer(2), &p).unwrap();
    assert_eq!(po.quantile(q, &p).unwrap_err().code(), "E-PROB-004");
}

/// `p` is a probability. Off `[0, 1]` the closed forms do not fail, they answer:
/// `Uniform(-2, 3).quantile(2)` is `8`, a point outside the support it is
/// supposed to name.
#[test]
fn a_quantile_argument_outside_zero_one_is_refused() {
    let p = pool();
    let u = Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap();
    for bad in [p.integer(2), p.rational(-1, 2), p.integer(-3)] {
        let e = u.quantile(bad, &p).unwrap_err();
        assert_eq!(e.code(), "E-PROB-001");
    }
    let e = Distribution::exponential(p.rational(7, 4), &p).unwrap();
    assert_eq!(
        e.quantile(p.rational(3, 2), &p).unwrap_err().code(),
        "E-PROB-001"
    );

    // Inside, a literal still returns a verified value.
    // mpmath: -log(1 - 9/10)/(7/4) = 1.3157629102823118194…
    assert_close!(
        at(&p, e.quantile(p.rational(9, 10), &p).unwrap().value, &[]),
        1.315_762_910_282_311_8
    );
    assert_close!(
        at(&p, u.quantile(p.rational(1, 4), &p).unwrap().value, &[]),
        -0.75
    );

    // A symbolic `p` cannot be decided, so `0 ≤ p ≤ 1` is carried.
    let q = sym(&p, "q");
    let r = u.quantile(q, &p).unwrap();
    assert!(r
        .log
        .0
        .iter()
        .any(|s| s.rule_name == "prob_quantile_argument_is_a_probability"
            && s.side_conditions.len() == 2));
}

// ---------------------------------------------------------------------------
// E[f(X)]
// ---------------------------------------------------------------------------

#[test]
fn expectation_of_a_polynomial_matches_the_moment_table() {
    let p = pool();
    let x = sym(&p, "x");
    let d = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    // E[3X² + 2X + 1] = 3·1 + 0 + 1 = 4
    let f = p.add(vec![
        p.mul(vec![p.integer(3), p.pow(x, p.integer(2))]),
        p.mul(vec![p.integer(2), x]),
        p.integer(1),
    ]);
    let r = expectation(f, x, &d, &p).unwrap();
    assert_close!(at(&p, r.value, &[]), 4.0);
}

#[test]
fn expectation_of_an_exponential_is_the_moment_generating_function() {
    let p = pool();
    let x = sym(&p, "x");
    let d = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    // E[e^X] = e^{1/2} for a standard normal.
    let f = p.func("exp", vec![x]);
    let r = expectation(f, x, &d, &p).unwrap();
    assert_close!(at(&p, r.value, &[]), 0.5f64.exp());
}

#[test]
fn expectation_over_a_finite_discrete_support_is_the_sum() {
    let p = pool();
    let x = sym(&p, "x");
    let prob = sym(&p, "p");
    let d = Distribution::binomial(p.integer(4), prob, &p).unwrap();
    // E[X²] = np(1-p) + (np)² = 4p(1-p) + 16p²
    let f = p.pow(x, p.integer(2));
    let r = expectation(f, x, &d, &p).unwrap();
    let pv = 0.3f64;
    assert_close!(
        at(&p, r.value, &[("p", pv)]),
        4.0 * pv * (1.0 - pv) + (4.0 * pv).powi(2)
    );
}

#[test]
fn expectation_over_poisson_needs_a_polynomial_and_says_so() {
    let p = pool();
    let x = sym(&p, "x");
    let lam = sym(&p, "lambda");
    let d = Distribution::poisson(lam, &p).unwrap();
    // A polynomial closes.
    let f = p.pow(x, p.integer(2));
    let r = expectation(f, x, &d, &p).unwrap();
    let l = 1.75f64;
    assert_close!(at(&p, r.value, &[("lambda", l)]), l * l + l);
    // A non-polynomial does not, and is refused rather than truncated.
    let g = p.func("exp", vec![p.func("sin", vec![x])]);
    let e = expectation(g, x, &d, &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-002");
}

/// `E[f(X)]` over a law whose support has a *finite* upper endpoint.
///
/// `map_bound` used to decide which reduction endpoint a support endpoint maps
/// to by asking whether the support endpoint was `+∞`.  That is right for every
/// law that is unbounded above and wrong for `Uniform` and `Beta`: the upper
/// bound mapped to the reduction's *lower* endpoint, every integral collapsed
/// to `∫_a^a = 0`, and `E[f]` was `0` for every `f`.  The gate caught it as
/// `E-PROB-005` — a false refusal — except where the true value happens to be
/// `0`, where a wrong derivation returned a right-looking number.
///
/// Values by hand: `E[X] = 1`, `E[X²] = 4/3`, `E[3X²+2X+1] = 7` over
/// `Uniform(0, 2)`; `E[e^X] = (e² − 1)/2` there; `E[X²] = 3/10` over
/// `Beta(2, 2)`.
#[test]
fn expectation_over_a_bounded_support_uses_the_upper_endpoint() {
    let p = pool();
    let x = sym(&p, "x");
    let u = Distribution::uniform(p.integer(0), p.integer(2), &p).unwrap();

    assert_close!(at(&p, expectation(x, x, &u, &p).unwrap().value, &[]), 1.0);
    assert_close!(
        at(
            &p,
            expectation(p.pow(x, p.integer(2)), x, &u, &p)
                .unwrap()
                .value,
            &[]
        ),
        4.0 / 3.0
    );
    let f = p.add(vec![
        p.mul(vec![p.integer(3), p.pow(x, p.integer(2))]),
        p.mul(vec![p.integer(2), x]),
        p.integer(1),
    ]);
    assert_close!(at(&p, expectation(f, x, &u, &p).unwrap().value, &[]), 7.0);
    assert_close!(
        at(
            &p,
            expectation(p.func("exp", vec![x]), x, &u, &p)
                .unwrap()
                .value,
            &[]
        ),
        (std::f64::consts::E.powi(2) - 1.0) / 2.0
    );

    let b = Distribution::beta(p.integer(2), p.integer(2), &p).unwrap();
    assert_close!(
        at(
            &p,
            expectation(p.pow(x, p.integer(2)), x, &b, &p)
                .unwrap()
                .value,
            &[]
        ),
        0.3
    );
}

/// A kinked payoff over a bounded support: both pieces have to land on the
/// right interval, not just the one whose endpoint is interior.
///
/// `E[max(X − K, 0)]` for `X ~ Uniform(a, b)` and `a ≤ K ≤ b` is
/// `(b − K)²/(2(b − a))`.  The old bound mapping computed `(K − a)²/(2(b − a))`
/// — the mirror image, equal only at the midpoint `K = (a+b)/2`, which is
/// exactly where a hand-picked example would have been chosen.
#[test]
fn a_call_payoff_over_a_uniform_uses_the_upper_endpoint() {
    let p = pool();
    let x = sym(&p, "x");
    let u = Distribution::uniform(p.integer(0), p.integer(2), &p).unwrap();
    for (k_num, k_den, want) in [(1_i64, 2_i64, 0.5625_f64), (1, 1, 0.25), (3, 2, 0.0625)] {
        let k = p.rational(k_num, k_den);
        let payoff = p.func("max", vec![sub(x, k, &p), p.integer(0)]);
        let r = expectation(payoff, x, &u, &p).unwrap();
        assert_close!(at(&p, r.value, &[]), want);
    }
}

#[test]
fn a_divergent_expectation_is_reported_as_divergent() {
    let p = pool();
    let x = sym(&p, "x");
    let d = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    // E[e^{X²}] = ∫ e^{x²}e^{-x²/2}/√(2π) dx diverges. The formal symbolic
    // answer is a clean finite number; the gate refuses before it is returned.
    let f = p.func("exp", vec![p.pow(x, p.integer(2))]);
    let e = expectation(f, x, &d, &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-006");
}

// ---------------------------------------------------------------------------
// Black–Scholes
// ---------------------------------------------------------------------------

/// `E[max(S-K,0)]` for `S ~ LogNormal(μ,σ)`, derived rather than tabulated.
///
/// Reference value from the closed-form Black–Scholes expression
/// `e^{μ+σ²/2}Φ((μ+σ²-log K)/σ) - KΦ((μ-log K)/σ)`, evaluated with `mpmath` at
/// 40 dps at `μ = 0.05`, `σ = 0.3`, `K = 1.1`, and cross-checked there against
/// `mpmath.quad` of `∫_K^∞ (s-K)p(s)ds` — the two agree to all 40 digits.
#[test]
fn expectation_of_a_call_payoff_under_a_log_normal_is_black_scholes() {
    let p = pool();
    let s = sym(&p, "S");
    let (mu, sigma, k) = (sym(&p, "mu"), sym(&p, "sigma"), sym(&p, "K"));
    let d = Distribution::log_normal(mu, sigma, &p).unwrap();
    let payoff = p.func("max", vec![sub(s, k, &p), p.integer(0)]);
    let r = expectation(payoff, s, &d, &p).expect("the call payoff should derive");
    let value = at(&p, r.value, &[("mu", 0.05), ("sigma", 0.3), ("K", 1.1)]);
    assert_close!(value, 0.130_968_082_077_972_3);

    // The derivation records that the strike has to be inside the support.
    let has_condition = r
        .log
        .0
        .iter()
        .any(|s| s.rule_name == "prob_kink_inside_support");
    assert!(has_condition, "K > 0 should travel with the answer");
}

#[test]
fn a_put_payoff_derives_too_and_satisfies_put_call_parity() {
    let p = pool();
    let s = sym(&p, "S");
    let (mu, sigma, k) = (sym(&p, "mu"), sym(&p, "sigma"), sym(&p, "K"));
    let d = Distribution::log_normal(mu, sigma, &p).unwrap();
    let call = expectation(p.func("max", vec![sub(s, k, &p), p.integer(0)]), s, &d, &p)
        .unwrap()
        .value;
    let put = expectation(p.func("max", vec![sub(k, s, &p), p.integer(0)]), s, &d, &p)
        .unwrap()
        .value;
    let env = [("mu", 0.05), ("sigma", 0.3), ("K", 1.1)];
    // E[max(S-K,0)] - E[max(K-S,0)] = E[S] - K
    let parity = at(&p, call, &env) - at(&p, put, &env);
    let forward = (0.05f64 + 0.5 * 0.09).exp() - 1.1;
    assert_close!(parity, forward);
}

#[test]
fn a_normal_call_payoff_is_the_bachelier_formula() {
    let p = pool();
    let x = sym(&p, "x");
    let (mu, sigma, k) = (sym(&p, "mu"), sym(&p, "sigma"), sym(&p, "K"));
    let d = Distribution::normal(mu, sigma, &p).unwrap();
    let payoff = p.func("max", vec![sub(x, k, &p), p.integer(0)]);
    let r = expectation(payoff, x, &d, &p).unwrap();
    // Bachelier: (μ-K)Φ(d) + σφ(d) with d = (μ-K)/σ. mpmath at 40 dps with
    // μ=0.2, σ=0.9, K=0.5 gives 0.228812502869164756134…
    let v = at(&p, r.value, &[("mu", 0.2), ("sigma", 0.9), ("K", 0.5)]);
    assert_close!(v, 0.228_812_502_869_164_76);
}

// ---------------------------------------------------------------------------
// Linearity and the independence rule
// ---------------------------------------------------------------------------

#[test]
fn linearity_of_expectation_needs_no_independence() {
    let p = pool();
    let (x, y) = (sym(&p, "X"), sym(&p, "Y"));
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let lam = sym(&p, "lambda");
    let nd = Distribution::normal(mu, sigma, &p).unwrap();
    let ed = Distribution::exponential(lam, &p).unwrap();
    // E[2X - 3Y + 5] = 2μ - 3/λ + 5
    let expr = p.add(vec![
        p.mul(vec![p.integer(2), x]),
        p.mul(vec![p.integer(-3), y]),
        p.integer(5),
    ]);
    let r = expectation_affine(expr, &[(x, nd), (y, ed)], &p).unwrap();
    assert_close!(
        at(&p, r.value, &[("mu", 1.5), ("sigma", 0.5), ("lambda", 2.0)]),
        2.0 * 1.5 - 3.0 / 2.0 + 5.0
    );
}

#[test]
fn variance_of_an_independent_sum_adds_the_squares() {
    let p = pool();
    let (x, y) = (sym(&p, "X"), sym(&p, "Y"));
    let nd = Distribution::normal(p.integer(0), sym(&p, "sigma"), &p).unwrap();
    let ud = Distribution::uniform(p.integer(0), p.integer(1), &p).unwrap();
    let expr = p.add(vec![p.mul(vec![p.integer(2), x]), y]);
    let r = variance_affine_independent(expr, &[(x, nd), (y, ud)], &p).unwrap();
    // 4σ² + 1/12
    assert_close!(at(&p, r.value, &[("sigma", 1.5)]), 4.0 * 2.25 + 1.0 / 12.0);
}

#[test]
fn a_product_of_two_variates_is_refused_rather_than_guessed() {
    let p = pool();
    let (x, y) = (sym(&p, "X"), sym(&p, "Y"));
    let nd = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let nd2 = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let expr = p.mul(vec![x, y]);
    let e = expectation_affine(expr, &[(x, nd), (y, nd2)], &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-002");
    assert!(format!("{e}").contains("joint law"));
}

#[test]
fn a_nonlinear_function_of_one_variate_is_refused_by_the_affine_route() {
    let p = pool();
    let x = sym(&p, "X");
    let nd = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let expr = p.func("sin", vec![x]);
    let e = expectation_affine(expr, &[(x, nd)], &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-002");
}

// ---------------------------------------------------------------------------
// The gate has to be able to fail
// ---------------------------------------------------------------------------

/// A check that never fails is decoration. Feed the verifier a claim that is
/// wrong the way a real transcription slip is wrong — `σ` where `σ²` belongs —
/// and confirm it is rejected.
#[test]
fn the_numeric_gate_rejects_a_plausible_wrong_closed_form() {
    let p = pool();
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let d = Distribution::normal(mu, sigma, &p).unwrap();
    let x = super::dists::fresh_var(&[mu, sigma], &p);
    let centred = p.pow(sub(x, mu, &p), p.integer(2));
    let integrand = super::verify::definition_integrand(centred, x, &d, &p);

    // The true variance is σ². Claim σ.
    let wrong = sigma;
    let e = super::verify::check(wrong, integrand, x, &d, &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-005");

    // And the right one passes, at several parameter points.
    let right = p.pow(sigma, p.integer(2));
    let ev = super::verify::check(right, integrand, x, &d, &p).unwrap();
    assert!(ev.parameter_points >= 2);
    assert!(ev.worst_relative_error < 1e-12);
}

/// The same for a *scale* mistake, which survives the point `σ = 1` that a
/// lazier table would have been checked at.
#[test]
fn the_numeric_gate_rejects_a_form_that_is_right_only_at_sigma_one() {
    let p = pool();
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let d = Distribution::normal(mu, sigma, &p).unwrap();
    let x = super::dists::fresh_var(&[mu, sigma], &p);
    let integrand = super::verify::definition_integrand(p.pow(x, p.integer(2)), x, &d, &p);
    // E[X²] = μ² + σ²; claim μ² + 1, which is right at σ = 1 and nowhere else.
    let wrong = p.add(vec![p.pow(mu, p.integer(2)), p.integer(1)]);
    assert_eq!(
        super::verify::check(wrong, integrand, x, &d, &p)
            .unwrap_err()
            .code(),
        "E-PROB-005"
    );
}

// ---------------------------------------------------------------------------
// The density integrates to one — the identity every table entry rests on
// ---------------------------------------------------------------------------

#[test]
fn every_density_integrates_to_one() {
    let p = pool();
    let one = p.integer(1);
    let cases: Vec<Distribution> = vec![
        Distribution::normal(sym(&p, "mu"), sym(&p, "sigma"), &p).unwrap(),
        Distribution::log_normal(sym(&p, "mu"), sym(&p, "sigma"), &p).unwrap(),
        Distribution::uniform(sym(&p, "a"), sym(&p, "b"), &p).unwrap(),
        Distribution::exponential(sym(&p, "lambda"), &p).unwrap(),
        Distribution::gamma(sym(&p, "k"), sym(&p, "theta"), &p).unwrap(),
        Distribution::beta(sym(&p, "alpha"), sym(&p, "beta"), &p).unwrap(),
        Distribution::bernoulli(sym(&p, "p"), &p).unwrap(),
        Distribution::binomial(p.integer(4), sym(&p, "p"), &p).unwrap(),
        Distribution::poisson(sym(&p, "lambda"), &p).unwrap(),
    ];
    for d in &cases {
        let x = super::dists::fresh_var(d.params(), &p);
        let integrand = super::verify::definition_integrand(one, x, d, &p);
        let ev = super::verify::check(one, integrand, x, d, &p)
            .unwrap_or_else(|e| panic!("{}: {e}", d.kind().name()));
        assert!(
            ev.parameter_points >= 2,
            "{}: only {} points",
            d.kind().name(),
            ev.parameter_points
        );
    }
}

// ---------------------------------------------------------------------------
// Support and pdf shape
// ---------------------------------------------------------------------------

#[test]
fn supports_are_what_they_say() {
    let p = pool();
    let d = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    assert_eq!(d.support(&p), Support::Real);
    let d = Distribution::log_normal(p.integer(0), p.integer(1), &p).unwrap();
    assert_eq!(d.support(&p), Support::Positive);
    let d = Distribution::binomial(p.integer(7), p.rational(1, 2), &p).unwrap();
    assert_eq!(d.support(&p), Support::IntegersUpTo(7));
    assert!(d.support(&p).is_discrete());
    let d = Distribution::poisson(p.integer(1), &p).unwrap();
    assert_eq!(d.support(&p), Support::NonNegativeIntegers);
}

#[test]
fn the_pdf_is_the_textbook_density() {
    let p = pool();
    let x = sym(&p, "x");
    let d = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let f = simplify(d.pdf(x, &p), &p).value;
    // φ(0) = 1/√(2π)
    assert_close!(
        at(&p, f, &[("x", 0.0)]),
        1.0 / (2.0 * std::f64::consts::PI).sqrt()
    );
    let d = Distribution::poisson(p.integer(2), &p).unwrap();
    let k = sym(&p, "k");
    let f = d.pdf(k, &p);
    // P(X=3) for λ=2 is 8e^{-2}/6
    assert_close!(at(&p, f, &[("k", 3.0)]), 8.0 * (-2.0f64).exp() / 6.0);
}

// ---------------------------------------------------------------------------
// Characteristic functions
// ---------------------------------------------------------------------------

/// The claim `φ(t)` evaluated in ℂ at a named environment.
fn phi_at(p: &ExprPool, e: ExprId, env: &[(&str, f64)]) -> (f64, f64) {
    let bindings: Vec<(ExprId, Float)> = env
        .iter()
        .map(|(n, v)| (sym(p, n), Float::with_val(VERIFY_PREC, *v)))
        .collect();
    let v = super::cplx::eval_complex(e, &bindings, p, VERIFY_PREC)
        .unwrap_or_else(|| panic!("{} did not evaluate in C", p.display(e)));
    (v.re.to_f64(), v.im.to_f64())
}

/// The convention test the whole layer hinges on. `φ_Normal(t) = e^{iμt -
/// σ²t²/2}`; a `2π` or a sign slip against alkahest's ordinary-frequency
/// Fourier convention produces something that is still smooth, still `1` at
/// `t = 0`, and wrong everywhere else.
#[test]
fn the_normal_characteristic_function_pins_the_convention() {
    let p = pool();
    let t = sym(&p, "t");
    let (mu, sigma) = (sym(&p, "mu"), sym(&p, "sigma"));
    let d = Distribution::normal(mu, sigma, &p).unwrap();
    let r = d.characteristic_function(t, &p).unwrap();
    let (re, im) = phi_at(&p, r.value, &[("mu", 0.7), ("sigma", 1.1), ("t", 1.3)]);
    let m = (-(1.1f64 * 1.1) * (1.3 * 1.3) / 2.0).exp();
    let ph = 0.7f64 * 1.3;
    assert_close!(re, m * ph.cos());
    assert_close!(im, m * ph.sin());
    // φ(0) = 1 for every law — necessary, and on its own worth nothing, which
    // is why the gate's own ladder does not include it.
    let (re0, im0) = phi_at(&p, r.value, &[("mu", 0.7), ("sigma", 1.1), ("t", 0.0)]);
    assert_close!(re0, 1.0);
    assert_close!(im0, 0.0);
    // The gate ran the independent Fourier route, not just the definition.
    assert!(r
        .log
        .0
        .iter()
        .any(|s| s.rule_name == "prob_charfun_matches_fourier_transform_at_minus_t_over_two_pi"));
}

#[test]
fn exponential_gamma_uniform_characteristic_functions() {
    let p = pool();
    let t = sym(&p, "t");

    let lam = sym(&p, "lambda");
    let e = Distribution::exponential(lam, &p).unwrap();
    let phi = e.characteristic_function(t, &p).unwrap();
    // λ/(λ - it) = λ(λ + it)/(λ² + t²)
    let (lv, tv) = (1.6f64, 0.9f64);
    let den = lv * lv + tv * tv;
    let (re, im) = phi_at(&p, phi.value, &[("lambda", lv), ("t", tv)]);
    assert_close!(re, lv * lv / den);
    assert_close!(im, lv * tv / den);

    let (k, theta) = (sym(&p, "k"), sym(&p, "theta"));
    let g = Distribution::gamma(k, theta, &p).unwrap();
    let phi = g.characteristic_function(t, &p).unwrap();
    // (1 - iθt)^{-k} at k=2, θ=0.5, t=1.2: (1 - 0.6i)^{-2},
    // and (1-0.6i)² = 0.64 - 1.2i.
    let (re, im) = phi_at(&p, phi.value, &[("k", 2.0), ("theta", 0.5), ("t", 1.2)]);
    let d2 = 0.64f64 * 0.64 + 1.2 * 1.2;
    assert_close!(re, 0.64 / d2);
    assert_close!(im, 1.2 / d2);

    let (a, b) = (sym(&p, "a"), sym(&p, "b"));
    let u = Distribution::uniform(a, b, &p).unwrap();
    let phi = u.characteristic_function(t, &p).unwrap();
    // (e^{ibt} - e^{iat})/(it(b-a)) at a=0, b=1, t=1: (e^{i} - 1)/i
    let (re, im) = phi_at(&p, phi.value, &[("a", 0.0), ("b", 1.0), ("t", 1.0)]);
    assert_close!(re, 1f64.sin());
    assert_close!(im, 1.0 - 1f64.cos());
}

#[test]
fn discrete_characteristic_functions() {
    let p = pool();
    let t = sym(&p, "t");
    let prob = sym(&p, "p");

    let b = Distribution::bernoulli(prob, &p).unwrap();
    let phi = b.characteristic_function(t, &p).unwrap();
    let (pv, tv) = (0.3f64, 1.1f64);
    let (re, im) = phi_at(&p, phi.value, &[("p", pv), ("t", tv)]);
    assert_close!(re, 1.0 - pv + pv * tv.cos());
    assert_close!(im, pv * tv.sin());

    let bin = Distribution::binomial(p.integer(4), prob, &p).unwrap();
    let phi = bin.characteristic_function(t, &p).unwrap();
    let (re, im) = phi_at(&p, phi.value, &[("p", pv), ("t", tv)]);
    // ((1-p) + p e^{it})⁴
    let (br, bi) = (1.0 - pv + pv * tv.cos(), pv * tv.sin());
    let (mut ar, mut ai) = (1.0f64, 0.0f64);
    for _ in 0..4 {
        let nr = ar * br - ai * bi;
        ai = ar * bi + ai * br;
        ar = nr;
    }
    assert_close!(re, ar);
    assert_close!(im, ai);

    let lam = sym(&p, "lambda");
    let po = Distribution::poisson(lam, &p).unwrap();
    let phi = po.characteristic_function(t, &p).unwrap();
    let lv = 2.2f64;
    let (re, im) = phi_at(&p, phi.value, &[("lambda", lv), ("t", tv)]);
    // e^{λ(e^{it} - 1)}
    let m = (lv * (tv.cos() - 1.0)).exp();
    let ph = lv * tv.sin();
    assert_close!(re, m * ph.cos());
    assert_close!(im, m * ph.sin());
}

#[test]
fn the_log_normal_and_beta_characteristic_functions_refuse() {
    let p = pool();
    let t = sym(&p, "t");
    let d = Distribution::log_normal(p.integer(0), p.integer(1), &p).unwrap();
    let e = d.characteristic_function(t, &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-004");
    assert!(format!("{e}").contains("no closed form"));

    let b = Distribution::beta(p.integer(2), p.integer(3), &p).unwrap();
    let e = b.characteristic_function(t, &p).unwrap_err();
    assert_eq!(e.code(), "E-PROB-004");
    assert!(format!("{e}").contains("hypergeometric"));
}

/// The strongest check this layer has: every moment computed two ways that
/// share no code — by quadrature of `∫xⁿp(x)dx` through the moment table, and
/// by differentiating `φ` `n` times at the origin — and required to agree.
///
/// `φ⁽ⁿ⁾(0) = iⁿE[Xⁿ]`, so the `n`-th derivative is real for even `n` and
/// purely imaginary for odd `n`; the test reads off whichever part carries it.
///
/// `Uniform` is absent on purpose. Its `φ` is `(e^{ibt} - e^{iat})/(it(b-a))`,
/// which is entire but *written* with a removable singularity at the origin —
/// the derivatives exist as limits and cannot be read off by substituting
/// `t = 0`. That is a limitation of evaluating this particular form at a
/// point, not of the identity, and papering over it by evaluating near zero
/// would turn an exact cross-check into a finite-difference estimate.
#[test]
fn moments_from_the_characteristic_function_agree_with_the_moment_table() {
    let p = pool();
    let t = sym(&p, "t");
    let cases: Vec<(Distribution, Vec<(&str, f64)>)> = vec![
        (
            Distribution::normal(sym(&p, "mu"), sym(&p, "sigma"), &p).unwrap(),
            vec![("mu", 0.6), ("sigma", 1.3)],
        ),
        (
            Distribution::exponential(sym(&p, "lambda"), &p).unwrap(),
            vec![("lambda", 1.7)],
        ),
        (
            Distribution::gamma(p.integer(3), sym(&p, "theta"), &p).unwrap(),
            vec![("theta", 0.8)],
        ),
        (
            Distribution::bernoulli(sym(&p, "p"), &p).unwrap(),
            vec![("p", 0.42)],
        ),
        (
            Distribution::poisson(sym(&p, "lambda"), &p).unwrap(),
            vec![("lambda", 2.4)],
        ),
        (
            Distribution::binomial(p.integer(5), sym(&p, "p"), &p).unwrap(),
            vec![("p", 0.35)],
        ),
    ];
    for (d, env) in &cases {
        let phi = d.characteristic_function(t, &p).unwrap().value;
        let mut deriv = phi;
        for n in 1..=3u32 {
            deriv = simplify(crate::diff::diff(deriv, t, &p).unwrap().value, &p).value;
            let mut at_zero: Vec<(&str, f64)> = env.clone();
            at_zero.push(("t", 0.0));
            let (re, im) = phi_at(&p, deriv, &at_zero);
            // φ⁽ⁿ⁾(0) = iⁿ·E[Xⁿ]
            let from_phi = match n % 4 {
                0 => re,
                1 => im,
                2 => -re,
                _ => -im,
            };
            let from_table = at(&p, d.moment(n, &p).unwrap().value, env);
            assert!(
                close(from_phi, from_table),
                "{} moment {n}: phi route {from_phi} vs table {from_table}",
                d.kind().name()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Generating functions
// ---------------------------------------------------------------------------

/// Evaluate a complex-valued expression at a named-symbol environment.
fn at_complex(p: &ExprPool, e: ExprId, env: &[(&str, f64)]) -> (f64, f64) {
    let bindings: Vec<(ExprId, Float)> = env
        .iter()
        .map(|(n, v)| (sym(p, n), Float::with_val(VERIFY_PREC, *v)))
        .collect();
    let v = super::cplx::eval_complex(e, &bindings, p, VERIFY_PREC)
        .unwrap_or_else(|| panic!("{} did not evaluate in C", p.display(e)));
    (v.re.to_f64(), v.im.to_f64())
}

/// `expr` with `var` replaced by `by`.
fn subs1(p: &ExprPool, expr: ExprId, var: ExprId, by: ExprId) -> ExprId {
    let mut m = std::collections::HashMap::new();
    m.insert(var, by);
    crate::kernel::subs(expr, &m, p)
}

/// `d^n expr / d var^n`, simplified at each step so the expression does not
/// blow up before it can be evaluated.
fn nth_derivative(p: &ExprPool, expr: ExprId, var: ExprId, n: u32) -> ExprId {
    let mut e = expr;
    for _ in 0..n {
        e = simplify(
            crate::diff::diff(e, var, p).expect("differentiable").value,
            p,
        )
        .value;
    }
    e
}

#[test]
fn the_mgf_at_an_imaginary_argument_is_the_characteristic_function() {
    // Cross-check 1. `M_X(it) = φ_X(t)` by definition — `E[e^{(it)X}]` is
    // `E[e^{itX}]`. The two tables are written out independently in `genfun`
    // and `charfun`, so a slip in either shows up here.
    let p = pool();
    let t = sym(&p, "t");
    let s = sym(&p, "s");
    let i_t = p.mul(vec![p.imaginary_unit(), t]);
    let dists = [
        Distribution::normal(p.rational(7, 10), p.rational(11, 10), &p).unwrap(),
        Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap(),
        Distribution::exponential(p.rational(7, 4), &p).unwrap(),
        Distribution::gamma(p.integer(3), p.rational(4, 5), &p).unwrap(),
        Distribution::bernoulli(p.rational(3, 10), &p).unwrap(),
        Distribution::binomial(p.integer(5), p.rational(7, 20), &p).unwrap(),
        Distribution::poisson(p.rational(12, 5), &p).unwrap(),
    ];
    for d in &dists {
        let m = moment_generating_function(d, s, &p).unwrap().value;
        let phi = d.characteristic_function(t, &p).unwrap().value;
        for tv in [0.4, -0.9, 1.3] {
            let (mr, mi) = at_complex(&p, subs1(&p, m, s, i_t), &[("t", tv)]);
            let (pr, pi_) = at_complex(&p, phi, &[("t", tv)]);
            assert_close!(mr, pr);
            assert_close!(mi, pi_);
        }
    }
}

#[test]
fn derivatives_of_the_mgf_at_zero_are_the_raw_moments() {
    // Cross-check 2. `M⁽ⁿ⁾(0) = E[Xⁿ]`, reached by differentiating the MGF
    // rather than by reading the moment table — two routes that share no
    // closed form.
    let p = pool();
    let t = sym(&p, "t");
    let cases = [
        Distribution::normal(p.rational(1, 2), p.rational(3, 2), &p).unwrap(),
        Distribution::exponential(p.rational(7, 4), &p).unwrap(),
        Distribution::gamma(p.integer(3), p.rational(4, 5), &p).unwrap(),
        Distribution::poisson(p.rational(12, 5), &p).unwrap(),
        Distribution::binomial(p.integer(5), p.rational(7, 20), &p).unwrap(),
    ];
    for d in &cases {
        let m = moment_generating_function(d, t, &p).unwrap().value;
        for n in 1..=4u32 {
            let dn = subs1(&p, nth_derivative(&p, m, t, n), t, p.integer(0));
            let table = d.moment(n, &p).unwrap().value;
            assert_close!(at(&p, dn, &[]), at(&p, table, &[]));
        }
    }
}

#[test]
fn derivatives_of_the_pgf_at_one_are_the_factorial_moments() {
    // Cross-check 3. `G⁽ⁿ⁾(1) = E[X(X-1)⋯(X-n+1)]`.
    let p = pool();
    let z = sym(&p, "z");
    let cases = [
        Distribution::bernoulli(p.rational(3, 10), &p).unwrap(),
        Distribution::binomial(p.integer(5), p.rational(7, 20), &p).unwrap(),
        Distribution::poisson(p.rational(12, 5), &p).unwrap(),
    ];
    for d in &cases {
        let g = probability_generating_function(d, z, &p).unwrap().value;
        for n in 1..=4u32 {
            let dn = subs1(&p, nth_derivative(&p, g, z, n), z, p.integer(1));
            let fm = factorial_moment(d, n, &p).unwrap().value;
            assert_close!(at(&p, dn, &[]), at(&p, fm, &[]));
        }
    }
}

#[test]
fn the_pgf_at_e_to_the_t_is_the_mgf() {
    // Cross-check 4. `G_X(e^t) = E[(e^t)^X] = E[e^{tX}] = M_X(t)` for an
    // integer-valued `X`.
    let p = pool();
    let t = sym(&p, "t");
    let z = sym(&p, "z");
    let e_t = p.func("exp", vec![t]);
    let cases = [
        Distribution::bernoulli(p.rational(3, 10), &p).unwrap(),
        Distribution::binomial(p.integer(5), p.rational(7, 20), &p).unwrap(),
        Distribution::poisson(p.rational(12, 5), &p).unwrap(),
    ];
    for d in &cases {
        let g = probability_generating_function(d, z, &p).unwrap().value;
        let m = moment_generating_function(d, t, &p).unwrap().value;
        for tv in [0.4, -0.9, 1.3] {
            assert_close!(
                at(&p, subs1(&p, g, z, e_t), &[("t", tv)]),
                at(&p, m, &[("t", tv)])
            );
        }
    }
}

#[test]
fn the_first_two_cumulants_are_the_mean_and_the_variance() {
    // Cross-check 5a.
    let p = pool();
    let cases = [
        Distribution::normal(p.rational(1, 2), p.rational(3, 2), &p).unwrap(),
        Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap(),
        Distribution::exponential(p.rational(7, 4), &p).unwrap(),
        Distribution::gamma(p.integer(3), p.rational(4, 5), &p).unwrap(),
        Distribution::beta(p.integer(2), p.integer(3), &p).unwrap(),
        Distribution::bernoulli(p.rational(3, 10), &p).unwrap(),
        Distribution::binomial(p.integer(5), p.rational(7, 20), &p).unwrap(),
        Distribution::poisson(p.rational(12, 5), &p).unwrap(),
    ];
    for d in &cases {
        assert_close!(
            at(&p, cumulant(d, 1, &p).unwrap().value, &[]),
            at(&p, d.mean(&p).unwrap().value, &[])
        );
        assert_close!(
            at(&p, cumulant(d, 2, &p).unwrap().value, &[]),
            at(&p, d.variance(&p).unwrap().value, &[])
        );
    }
}

#[test]
fn every_normal_cumulant_past_the_second_is_zero() {
    // Cross-check 5b — the sharp one. A normal's CGF is exactly
    // `μt + σ²t²/2`, a quadratic, so `κ_n = 0` for every `n ≥ 3`. An
    // off-by-one anywhere in the moment–cumulant recursion produces a nonzero
    // value here, and the Gaussian moments it is built from are large
    // (`E[X⁶] = 15σ⁶ + …`), so the cancellation is not one a wrong recursion
    // can stumble into.
    let p = pool();
    let mu = sym(&p, "mu");
    let sigma = sym(&p, "sigma");
    let d = Distribution::normal(mu, sigma, &p).unwrap();
    for n in 3..=MAX_CUMULANT_ORDER {
        let k = cumulant(&d, n, &p).unwrap().value;
        for env in [
            [("mu", 0.0), ("sigma", 1.0)],
            [("mu", -1.5), ("sigma", 2.0)],
            [("mu", 0.25), ("sigma", 3.0)],
        ] {
            assert!(
                at(&p, k, &env).abs() < 1e-9,
                "kappa_{n} of a normal is {} at {env:?}, not 0",
                at(&p, k, &env)
            );
        }
    }
}

#[test]
fn every_poisson_cumulant_is_lambda() {
    // The Poisson's defining property: `K(t) = λ(e^t - 1)`, so every
    // derivative at the origin is `λ`. A second sharp test of the recursion,
    // and one whose right answer is not zero.
    let p = pool();
    let lam = sym(&p, "lambda");
    let d = Distribution::poisson(lam, &p).unwrap();
    for n in 1..=MAX_CUMULANT_ORDER {
        let k = cumulant(&d, n, &p).unwrap().value;
        for lv in [1.0, 0.4, 2.5] {
            assert_close!(at(&p, k, &[("lambda", lv)]), lv);
        }
    }
}

#[test]
fn skewness_and_excess_kurtosis_agree_with_the_cumulants() {
    // `γ₁ = κ₃/σ³` and `γ₂ = κ₄/σ⁴` wherever the cumulants exist. The two
    // routes are different by construction — the shape statistics are built
    // from central moments so that a `LogNormal`, which has no cumulants, is
    // still answered.
    let p = pool();
    let cases = [
        Distribution::normal(p.rational(1, 2), p.rational(3, 2), &p).unwrap(),
        Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap(),
        Distribution::exponential(p.rational(7, 4), &p).unwrap(),
        Distribution::gamma(p.integer(3), p.rational(4, 5), &p).unwrap(),
        Distribution::beta(p.integer(2), p.integer(3), &p).unwrap(),
        Distribution::bernoulli(p.rational(3, 10), &p).unwrap(),
        Distribution::poisson(p.rational(12, 5), &p).unwrap(),
    ];
    for d in &cases {
        let var = at(&p, d.variance(&p).unwrap().value, &[]);
        let k3 = at(&p, cumulant(d, 3, &p).unwrap().value, &[]);
        let k4 = at(&p, cumulant(d, 4, &p).unwrap().value, &[]);
        assert_close!(
            at(&p, skewness(d, &p).unwrap().value, &[]),
            k3 / var.powf(1.5)
        );
        assert_close!(
            at(&p, excess_kurtosis(d, &p).unwrap().value, &[]),
            k4 / (var * var)
        );
    }
}

#[test]
fn known_shape_statistics_match_their_textbook_values() {
    // Values from references, not from alkahest: a normal is mesokurtic
    // (γ₂ = 0); a uniform has γ₁ = 0, γ₂ = -6/5; an exponential has γ₁ = 2,
    // γ₂ = 6 at every rate; a Poisson has γ₁ = λ^{-1/2}, γ₂ = λ^{-1}.
    let p = pool();
    let n = Distribution::normal(p.rational(1, 2), p.rational(3, 2), &p).unwrap();
    assert_close!(at(&p, skewness(&n, &p).unwrap().value, &[]), 0.0);
    assert_close!(at(&p, excess_kurtosis(&n, &p).unwrap().value, &[]), 0.0);

    let u = Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap();
    assert_close!(at(&p, skewness(&u, &p).unwrap().value, &[]), 0.0);
    assert_close!(at(&p, excess_kurtosis(&u, &p).unwrap().value, &[]), -1.2);

    let lam = sym(&p, "lambda");
    let e = Distribution::exponential(lam, &p).unwrap();
    let (sk, ek) = (
        skewness(&e, &p).unwrap().value,
        excess_kurtosis(&e, &p).unwrap().value,
    );
    for lv in [1.0, 0.4, 2.5] {
        assert_close!(at(&p, sk, &[("lambda", lv)]), 2.0);
        assert_close!(at(&p, ek, &[("lambda", lv)]), 6.0);
    }

    let po = Distribution::poisson(lam, &p).unwrap();
    let (sk, ek) = (
        skewness(&po, &p).unwrap().value,
        excess_kurtosis(&po, &p).unwrap().value,
    );
    for lv in [1.0, 0.4, 2.5] {
        assert_close!(at(&p, sk, &[("lambda", lv)]), lv.powf(-0.5));
        assert_close!(at(&p, ek, &[("lambda", lv)]), 1.0 / lv);
    }
}

#[test]
fn the_lognormal_mgf_is_refused_rather_than_completed() {
    // The archetypal silent error: completing the square in `∫e^{tx}p(x)dx`
    // for a log-normal produces a clean closed form, and the integral it is
    // supposed to be the value of diverges for every `t > 0`.
    let p = pool();
    let t = sym(&p, "t");
    let d = Distribution::log_normal(p.integer(0), p.integer(1), &p).unwrap();
    // A symbolic `t` cannot be decided, and the undecidable branch is the
    // divergent one: an MGF has to exist on a neighbourhood of the origin.
    assert_eq!(
        moment_generating_function(&d, t, &p).unwrap_err().code(),
        "E-PROB-006"
    );
    assert_eq!(
        moment_generating_function(&d, p.integer(1), &p)
            .unwrap_err()
            .code(),
        "E-PROB-006"
    );
    assert_eq!(
        cumulant_generating_function(&d, t, &p).unwrap_err().code(),
        "E-PROB-006"
    );
    assert_eq!(cumulant(&d, 3, &p).unwrap_err().code(), "E-PROB-006");
    // `t ≤ 0` is a different statement: the expectation is finite there, and
    // what is missing is a closed form rather than a value.
    assert_eq!(
        moment_generating_function(&d, p.integer(-1), &p)
            .unwrap_err()
            .code(),
        "E-PROB-004"
    );
    // …but the shape statistics, which are central moments rather than
    // cumulants, do exist and are returned. γ₁ = (e^{σ²}+2)√(e^{σ²}-1) with
    // σ = 1 is 6.1848771858680. (Aitchison & Brown, *The Lognormal
    // Distribution*, §2.3.)
    assert_close!(
        at(&p, skewness(&d, &p).unwrap().value, &[]),
        (1.0f64.exp() + 2.0) * (1.0f64.exp() - 1.0).sqrt()
    );
}

#[test]
fn an_exponential_mgf_outside_its_strip_is_refused() {
    // `M(t) = λ/(λ - t)` only for `t < λ`. At `t = 2λ` the expression is
    // `-1` — finite, clean, and the value of no integral, since `E[e^{2λX}]`
    // is `+∞`.
    let p = pool();
    let d = Distribution::exponential(p.rational(7, 4), &p).unwrap();
    for bad in [p.rational(7, 2), p.rational(7, 4), p.integer(2)] {
        let e = moment_generating_function(&d, bad, &p).unwrap_err();
        assert_eq!(e.code(), "E-PROB-006");
        assert_eq!(
            cumulant_generating_function(&d, bad, &p)
                .unwrap_err()
                .code(),
            "E-PROB-006"
        );
    }
    // The control: inside the strip it is returned, unconditionally, and it is
    // the right number. λ/(λ - t) at λ = 7/4, t = 1/2 is 1.4.
    let good = moment_generating_function(&d, p.rational(1, 2), &p).unwrap();
    assert_close!(at(&p, good.value, &[]), 1.4);
    assert!(take_prob_side_conditions().is_empty());
}

#[test]
fn a_symbolic_mgf_argument_carries_its_convergence_strip() {
    // The condition cannot be decided, so it is *reported* — out of band,
    // through the same channel a symbolic `cdf` argument uses. A caller who
    // reads an empty list has been told the answer is unconditional.
    let p = pool();
    let t = sym(&p, "t");
    let lam = sym(&p, "lambda");

    let d = Distribution::exponential(lam, &p).unwrap();
    moment_generating_function(&d, t, &p).unwrap();
    let conds = take_prob_side_conditions();
    assert!(
        conds
            .iter()
            .any(|c| matches!(c, crate::deriv::SideCondition::Positive(_))),
        "expected the strip lambda - t > 0, got {conds:?}"
    );

    // A Gamma carries `1 - θt > 0` the same way.
    let theta = sym(&p, "theta");
    let g = Distribution::gamma(p.integer(3), theta, &p).unwrap();
    moment_generating_function(&g, t, &p).unwrap();
    assert!(!take_prob_side_conditions().is_empty());

    // A normal's MGF is entire: nothing to discharge, and the empty list is a
    // statement rather than a silence.
    let n = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    moment_generating_function(&n, t, &p).unwrap();
    assert!(take_prob_side_conditions().is_empty());
}

#[test]
fn the_pgf_of_a_continuous_law_is_a_category_error() {
    // `E[z^X] = E[e^{X log z}]` is a formal manipulation that returns a
    // number. That number is the MGF at `log z`, and `G_X` is a statement
    // about `P(X = k)` — which is zero for every `k` here.
    let p = pool();
    let z = sym(&p, "z");
    for d in [
        Distribution::normal(p.integer(0), p.integer(1), &p).unwrap(),
        Distribution::log_normal(p.integer(0), p.integer(1), &p).unwrap(),
        Distribution::uniform(p.integer(0), p.integer(1), &p).unwrap(),
        Distribution::exponential(p.integer(1), &p).unwrap(),
        Distribution::gamma(p.integer(3), p.integer(1), &p).unwrap(),
        Distribution::beta(p.integer(2), p.integer(3), &p).unwrap(),
    ] {
        assert_eq!(
            probability_generating_function(&d, z, &p)
                .unwrap_err()
                .code(),
            "E-PROB-002"
        );
        assert_eq!(
            factorial_moment(&d, 2, &p).unwrap_err().code(),
            "E-PROB-002"
        );
    }
    // The control: an integer-valued law answers. G(z) = e^{λ(z-1)}, and at
    // λ = 12/5, z = 1/2 that is e^{-1.2} = 0.30119421191220214.
    let po = Distribution::poisson(p.rational(12, 5), &p).unwrap();
    let g = probability_generating_function(&po, z, &p).unwrap();
    assert_close!(at(&p, g.value, &[("z", 0.5)]), (-1.2f64).exp());
}

#[test]
fn the_beta_cgf_refuses_but_its_cumulants_do_not() {
    // `M = ₁F₁(α; α+β; t)` has no name here, so `K` refuses. `K` is still
    // analytic at the origin — a Beta is bounded — so the cumulants exist and
    // are returned. Refusing them too would be a false refusal.
    let p = pool();
    let t = sym(&p, "t");
    let d = Distribution::beta(p.integer(2), p.integer(3), &p).unwrap();
    assert_eq!(
        moment_generating_function(&d, t, &p).unwrap_err().code(),
        "E-PROB-004"
    );
    assert_eq!(
        cumulant_generating_function(&d, t, &p).unwrap_err().code(),
        "E-PROB-004"
    );
    // Beta(2,3): μ = 2/5, σ² = αβ/((α+β)²(α+β+1)) = 6/150 = 1/25.
    assert_close!(at(&p, cumulant(&d, 2, &p).unwrap().value, &[]), 0.04);
    // γ₁ = 2(β-α)√(α+β+1)/((α+β+2)√(αβ)) = 2(1)√6/(7√6) = 2/7.
    assert_close!(at(&p, skewness(&d, &p).unwrap().value, &[]), 2.0 / 7.0);
}

#[test]
fn cumulant_orders_outside_the_checked_range_are_refused() {
    let p = pool();
    let d = Distribution::poisson(p.rational(12, 5), &p).unwrap();
    assert_eq!(cumulant(&d, 0, &p).unwrap_err().code(), "E-PROB-002");
    assert_eq!(
        cumulant(&d, MAX_CUMULANT_ORDER + 1, &p).unwrap_err().code(),
        "E-PROB-002"
    );
}

// ---------------------------------------------------------------------------
// Information theory
// ---------------------------------------------------------------------------
//
// Every literal in this section is from mpmath 40 dps or from a hand
// derivation, both recorded beside the value. The two identities
// (`H(P,Q) = H(P) + D(P‖Q)` and `D(P‖P) = 0`) are asserted *symbolically*,
// which is stronger than any number: they have to hold for every parameter at
// once, and they are what a transcription slip between the entropy table and
// the divergence table breaks.

/// `h(X)` for the continuous laws, against `scipy.stats.<dist>.entropy()` —
/// an independent implementation of the same quantity.
#[test]
fn differential_entropy_matches_an_independent_implementation() {
    let p = pool();
    let cases: Vec<(Distribution, f64)> = vec![
        // scipy.stats.norm(0, 1).entropy() = 1.4189385332046727
        (
            Distribution::normal(p.integer(0), p.integer(1), &p).unwrap(),
            1.4189385332046727,
        ),
        // scipy.stats.norm(2, 1.5).entropy() = 1.8244036413128367
        (
            Distribution::normal(p.integer(2), p.rational(3, 2), &p).unwrap(),
            1.8244036413128368,
        ),
        // scipy.stats.lognorm(s=0.5, scale=exp(0.2)).entropy() = 0.9257913526447273
        (
            Distribution::log_normal(p.rational(1, 5), p.rational(1, 2), &p).unwrap(),
            0.9257913526447272,
        ),
        // log(5); scipy.stats.uniform(-2, 5).entropy()
        (
            Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap(),
            1.60943791243413,
        ),
        // 1 - log(1.75); scipy.stats.expon(scale=1/1.75).entropy()
        (
            Distribution::exponential(p.rational(7, 4), &p).unwrap(),
            0.44038421206458056,
        ),
        // scipy.stats.gamma(3.5, scale=0.8).entropy() = 1.7199384494197568
        (
            Distribution::gamma(p.rational(7, 2), p.rational(4, 5), &p).unwrap(),
            1.7199384494197565,
        ),
        // scipy.stats.beta(2, 3).entropy() = -0.2349066497880
        (
            Distribution::beta(p.integer(2), p.integer(3), &p).unwrap(),
            -0.23490664978800016,
        ),
    ];
    for (d, want) in &cases {
        let h = d.entropy(None, &p).unwrap().value;
        assert_close!(at(&p, h, &[]), *want);
    }
}

/// `H(X)` for the discrete laws, against a sum done by hand.
#[test]
fn shannon_entropy_matches_the_defining_sum() {
    let p = pool();
    // -0.3 log 0.3 - 0.7 log 0.7
    let bern = Distribution::bernoulli(p.rational(3, 10), &p).unwrap();
    assert_close!(
        at(&p, bern.entropy(None, &p).unwrap().value, &[]),
        0.610864302054894
    );
    // scipy.stats.binom(5, 0.35).entropy() = 1.4642429369178962
    let binom = Distribution::binomial(p.integer(5), p.rational(7, 20), &p).unwrap();
    assert_close!(
        at(&p, binom.entropy(None, &p).unwrap().value, &[]),
        1.464242936917896
    );
}

/// `h(Normal(μ, σ)) = ½log(2πeσ²)`, and it does not depend on `μ`.
///
/// The `μ`-independence is asserted on the *expression*, not on two numbers: a
/// formula that happened to contain a `μ` which cancelled numerically at the
/// sampled points would pass a numeric check and fail this one.
#[test]
fn gaussian_entropy_is_half_log_two_pi_e_sigma_squared_and_free_of_mu() {
    let p = pool();
    let sigma = p.rational(3, 2);
    let mut seen = None;
    for mu in [
        p.integer(0),
        p.integer(3),
        p.rational(-7, 2),
        sym(&p, "mu_free"),
    ] {
        let h = Distribution::normal(mu, sigma, &p)
            .unwrap()
            .entropy(None, &p)
            .unwrap()
            .value;
        match seen {
            None => seen = Some(h),
            Some(first) => assert_eq!(first, h, "entropy depends on mu"),
        }
    }
    // ½log(2πe·2.25) = 1.8244036413128368 (mpmath 40 dps).
    assert_close!(at(&p, seen.unwrap(), &[]), 1.8244036413128368);
}

/// Differential entropy is **not** invariant under a change of variables: it
/// shifts by `E[log|dx/dy|]`.
///
/// `LogNormal(μ, σ) = e^{Normal(μ, σ)}` is a smooth bijection onto `(0, ∞)`,
/// so the shift is `E[Y] = μ`. Treating `h` as relabelling-invariant — which
/// Shannon entropy genuinely is — makes this difference `0`.
#[test]
fn differential_entropy_shifts_by_the_log_jacobian_under_exp() {
    let p = pool();
    let (mu, sigma) = (sym(&p, "mu_j"), sym(&p, "sigma_j"));
    let normal = Distribution::normal(mu, sigma, &p).unwrap();
    let log_normal = Distribution::log_normal(mu, sigma, &p).unwrap();
    let gap = simplify(
        sub(
            log_normal.entropy(None, &p).unwrap().value,
            normal.entropy(None, &p).unwrap().value,
            &p,
        ),
        &p,
    )
    .value;
    assert_eq!(gap, mu, "h(e^Y) - h(Y) should be exactly mu");
}

/// `D(P‖P) = 0`, symbolically, for every family.
#[test]
fn a_divergence_from_a_law_to_itself_is_exactly_zero() {
    let p = pool();
    let zero = p.integer(0);
    let families = [
        Distribution::normal(sym(&p, "m0"), sym(&p, "s0"), &p).unwrap(),
        Distribution::log_normal(sym(&p, "m0"), sym(&p, "s0"), &p).unwrap(),
        Distribution::uniform(sym(&p, "ua"), sym(&p, "ub"), &p).unwrap(),
        Distribution::exponential(sym(&p, "lam0"), &p).unwrap(),
        Distribution::gamma(sym(&p, "k0"), sym(&p, "th0"), &p).unwrap(),
        Distribution::beta(sym(&p, "al0"), sym(&p, "be0"), &p).unwrap(),
        Distribution::bernoulli(sym(&p, "pp0"), &p).unwrap(),
        Distribution::binomial(p.integer(5), sym(&p, "pb0"), &p).unwrap(),
        Distribution::poisson(sym(&p, "lm0"), &p).unwrap(),
    ];
    for d in &families {
        let v = kl_divergence(d, d, None, &p).unwrap().value;
        assert_eq!(
            simplify(v, &p).value,
            zero,
            "D(P‖P) for {} is {}",
            d.kind().name(),
            p.display(v)
        );
    }
}

/// `H(P, Q) = H(P) + D(P‖Q)`, symbolically.
///
/// The cross-entropy is assembled from both tables and then checked against
/// `-E_P[log q]`, which is neither of them, so this identity is a genuine
/// constraint tying the two tables together rather than a restatement of how
/// the value was built.
#[test]
fn cross_entropy_is_entropy_plus_divergence() {
    let p = pool();
    let zero = p.integer(0);
    let pairs: Vec<(Distribution, Distribution)> = vec![
        (
            Distribution::normal(sym(&p, "m1"), sym(&p, "s1"), &p).unwrap(),
            Distribution::normal(sym(&p, "m2"), sym(&p, "s2"), &p).unwrap(),
        ),
        (
            Distribution::exponential(sym(&p, "l1"), &p).unwrap(),
            Distribution::exponential(sym(&p, "l2"), &p).unwrap(),
        ),
        (
            Distribution::gamma(sym(&p, "k1"), sym(&p, "t1"), &p).unwrap(),
            Distribution::gamma(sym(&p, "k2"), sym(&p, "t2"), &p).unwrap(),
        ),
        (
            Distribution::beta(sym(&p, "a1"), sym(&p, "b1"), &p).unwrap(),
            Distribution::beta(sym(&p, "a2"), sym(&p, "b2"), &p).unwrap(),
        ),
        (
            Distribution::bernoulli(sym(&p, "q1"), &p).unwrap(),
            Distribution::bernoulli(sym(&p, "q2"), &p).unwrap(),
        ),
        (
            Distribution::binomial(p.integer(4), sym(&p, "r1"), &p).unwrap(),
            Distribution::binomial(p.integer(4), sym(&p, "r2"), &p).unwrap(),
        ),
    ];
    for (a, b) in &pairs {
        let ce = cross_entropy(a, b, None, &p).unwrap().value;
        let h = a.entropy(None, &p).unwrap().value;
        let d = kl_divergence(a, b, None, &p).unwrap().value;
        let gap = simplify(sub(ce, p.add(vec![h, d]), &p), &p).value;
        assert_eq!(gap, zero, "{} cross-entropy identity", a.kind().name());
    }
}

/// Gibbs: `D(P‖Q) ≥ 0`, with equality only on the diagonal.
///
/// Swept rather than sampled. A sign error in a closed form — the `Gamma`
/// entry's `(k₁-k₂)ψ(k₁)` and `k₁(θ₁-θ₂)/θ₂` have opposite signs and are easy
/// to transpose — stays positive near the diagonal and only goes negative
/// further out, so one point proves nothing.
#[test]
fn gibbs_every_divergence_is_non_negative() {
    let p = pool();
    let grid = [(1, 2), (1, 1), (3, 2), (5, 2)];
    let mut gammas = Vec::new();
    let mut normals = Vec::new();
    let mut bernoullis = Vec::new();
    for (a, b) in grid {
        gammas.push(Distribution::gamma(p.rational(a, 2), p.rational(b, 2), &p).unwrap());
        normals.push(Distribution::normal(p.rational(a - 3, 2), p.rational(b, 2), &p).unwrap());
    }
    for k in 1..10 {
        bernoullis.push(Distribution::bernoulli(p.rational(k, 10), &p).unwrap());
    }
    for family in [&gammas, &normals, &bernoullis] {
        for a in family.iter() {
            for b in family.iter() {
                let v = at(&p, kl_divergence(a, b, None, &p).unwrap().value, &[]);
                assert!(
                    v >= -1e-12,
                    "D({} ‖ {}) = {v} is negative",
                    a.kind().name(),
                    b.kind().name()
                );
                if a == b {
                    assert!(v.abs() < 1e-12, "D(P‖P) = {v}");
                }
            }
        }
    }
}

/// Closed-form divergences against an independent computation.
#[test]
fn divergences_match_independent_values() {
    let p = pool();
    // mpmath 40 dps of ∫p log(p/q), and the closed forms by hand.
    let cases: Vec<(Distribution, Distribution, f64)> = vec![
        // log(1.5) + (1 + 4)/(2·2.25) - 0.5
        (
            Distribution::normal(p.integer(0), p.integer(1), &p).unwrap(),
            Distribution::normal(p.integer(2), p.rational(3, 2), &p).unwrap(),
            1.016576219219275,
        ),
        // log(1/4) + 4 - 1 = 3 - 2log2
        (
            Distribution::exponential(p.integer(1), &p).unwrap(),
            Distribution::exponential(p.integer(4), &p).unwrap(),
            1.613705638880109,
        ),
        // log 4 + 1/4 - 1 — the *other* direction, deliberately: D is not
        // symmetric, and these two numbers differ by 3log4 - 3.
        (
            Distribution::exponential(p.integer(4), &p).unwrap(),
            Distribution::exponential(p.integer(1), &p).unwrap(),
            0.636294361119891,
        ),
        // log((3-(-2))/(1-0)) = log 5
        (
            Distribution::uniform(p.integer(0), p.integer(1), &p).unwrap(),
            Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap(),
            1.6094379124341,
        ),
        // 2.4 log 2.4 + 1 - 2.4
        (
            Distribution::poisson(p.rational(12, 5), &p).unwrap(),
            Distribution::poisson(p.integer(1), &p).unwrap(),
            0.701124969649360,
        ),
        // 0.5 log(5/7) + 0.5 log(5/3); scipy.stats.entropy([.5,.5],[.3,.7])
        (
            Distribution::bernoulli(p.rational(1, 2), &p).unwrap(),
            Distribution::bernoulli(p.rational(7, 10), &p).unwrap(),
            0.087176693572389,
        ),
        // 0.3 log(3/5) + 0.7 log(7/5) — the other direction, a different
        // number: scipy.stats.entropy([.3,.7],[.5,.5]) = 0.08228287850505175.
        (
            Distribution::bernoulli(p.rational(3, 10), &p).unwrap(),
            Distribution::bernoulli(p.rational(1, 2), &p).unwrap(),
            0.082282878505052,
        ),
    ];
    for (a, b, want) in &cases {
        let v = kl_divergence(a, b, None, &p).unwrap().value;
        assert_close!(at(&p, v, &[]), *want);
    }
}

/// `supp P ⊄ supp Q` makes `D = +∞`, and the closed form there is a clean
/// finite **negative** number that nothing in the arithmetic objects to.
#[test]
fn a_divergence_off_a_nested_support_is_infinite_not_the_formula() {
    let p = pool();
    let wide = Distribution::uniform(p.integer(0), p.integer(1), &p).unwrap();
    let narrow = Distribution::uniform(p.integer(0), p.rational(1, 2), &p).unwrap();
    // The formula, if it were applied: log((1/2 - 0)/(1 - 0)) = -log 2.
    assert_eq!(
        kl_divergence(&wide, &narrow, None, &p).unwrap_err().code(),
        "E-PROB-006"
    );
    // A Q whose support Q ⊅ P on the *left* too.
    let shifted = Distribution::uniform(p.rational(1, 4), p.rational(3, 4), &p).unwrap();
    assert_eq!(
        kl_divergence(&wide, &shifted, None, &p).unwrap_err().code(),
        "E-PROB-006"
    );
    // Discrete: Q gives the atom `0` no mass at all.
    let fair = Distribution::bernoulli(p.rational(1, 2), &p).unwrap();
    for degenerate in [p.integer(0), p.integer(1)] {
        let q = Distribution::bernoulli(degenerate, &p).unwrap();
        assert_eq!(
            kl_divergence(&fair, &q, None, &p).unwrap_err().code(),
            "E-PROB-006"
        );
    }
    // The control: nesting restored, a value comes back.
    let containing = Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap();
    assert_close!(
        at(
            &p,
            kl_divergence(&wide, &containing, None, &p).unwrap().value,
            &[]
        ),
        1.6094379124341
    );
}

/// An undecidable containment is **published**, not assumed.
#[test]
fn a_symbolic_containment_travels_as_a_side_condition() {
    let p = pool();
    let (a1, b1) = (sym(&p, "ca"), sym(&p, "cb"));
    let inner = Distribution::uniform(a1, b1, &p).unwrap();
    let outer = Distribution::uniform(p.integer(-2), p.integer(3), &p).unwrap();
    match kl_divergence(&inner, &outer, None, &p) {
        Ok(_) => {
            let conds = take_prob_side_conditions();
            assert!(
                conds.len() >= 2,
                "a symbolic [a, b] against [-2, 3] must publish both containments, got {conds:?}"
            );
        }
        // Refusing to verify is also acceptable — what is not acceptable is
        // answering with the containment silently assumed.
        Err(e) => assert_eq!(e.code(), "E-PROB-005"),
    }
}

/// The units convert, and a base that is not a base is refused.
#[test]
fn entropy_in_bits_is_entropy_in_nats_over_log_two() {
    let p = pool();
    let fair = Distribution::bernoulli(p.rational(1, 2), &p).unwrap();
    // A fair coin is one bit, by definition, and log 2 nats.
    assert_close!(
        at(&p, fair.entropy(Some(p.integer(2)), &p).unwrap().value, &[]),
        1.0
    );
    assert_close!(
        at(&p, fair.entropy(None, &p).unwrap().value, &[]),
        std::f64::consts::LN_2
    );
    // log 1 = 0: there is no base-1 logarithm and no base-1 entropy.
    for bad in [p.integer(1), p.integer(0), p.integer(-2)] {
        assert_eq!(
            fair.entropy(Some(bad), &p).unwrap_err().code(),
            "E-PROB-001"
        );
    }
}

/// The refusals, and their controls.
#[test]
fn information_theory_refusals_name_what_is_missing() {
    let p = pool();
    // The Poisson entropy's residual sum is not a standard function.
    let pois = Distribution::poisson(p.rational(12, 5), &p).unwrap();
    assert_eq!(pois.entropy(None, &p).unwrap_err().code(), "E-PROB-004");
    // …but its divergence closes: `log k!` cancels between the two densities.
    let pois2 = Distribution::poisson(p.integer(1), &p).unwrap();
    assert_close!(
        at(
            &p,
            kl_divergence(&pois, &pois2, None, &p).unwrap().value,
            &[]
        ),
        0.701124969649360
    );
    // Cross-family is not the same-family formula with the names swapped.
    let normal = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let expo = Distribution::exponential(p.integer(1), &p).unwrap();
    assert_eq!(
        kl_divergence(&normal, &expo, None, &p).unwrap_err().code(),
        "E-PROB-002"
    );
    // A Binomial past the assembly limit.
    let big = Distribution::binomial(p.integer(80), p.rational(1, 3), &p).unwrap();
    assert_eq!(big.entropy(None, &p).unwrap_err().code(), "E-PROB-002");
}

/// A point mass has entropy exactly `0` — the `0 log 0 = 0` convention, taken
/// before the closed form (which is `NaN` there) is reached.
#[test]
fn the_entropy_of_a_point_mass_is_zero_not_nan() {
    let p = pool();
    let zero = p.integer(0);
    for degenerate in [p.integer(0), p.integer(1)] {
        let d = Distribution::bernoulli(degenerate, &p).unwrap();
        assert_eq!(d.entropy(None, &p).unwrap().value, zero);
        let b = Distribution::binomial(p.integer(4), degenerate, &p).unwrap();
        assert_eq!(b.entropy(None, &p).unwrap().value, zero);
    }
    let none = Distribution::binomial(p.integer(0), p.rational(1, 3), &p).unwrap();
    assert_eq!(none.entropy(None, &p).unwrap().value, zero);
}

/// `I(X; Y) = 0` under an assumed independence, and the assumption is the
/// caller's — it is in the function's name, not checked.
#[test]
fn mutual_information_of_an_independent_pair_is_zero() {
    let p = pool();
    let x = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let y = Distribution::poisson(p.integer(3), &p).unwrap();
    assert_eq!(
        mutual_information_independent(&x, &y, None, &p)
            .unwrap()
            .value,
        p.integer(0)
    );
}

/// The entropy and divergence gates are **able to fail**.
///
/// A check that has never rejected anything is decoration. These feed the same
/// verifier the same defining integrals with a deliberately wrong claim — the
/// two mistakes a reader would most easily make — and require a refusal.
#[test]
fn the_gate_rejects_a_wrong_entropy_and_a_wrong_divergence() {
    let p = pool();
    let (mu, sigma) = (sym(&p, "mu_g"), sym(&p, "sigma_g"));
    let normal = Distribution::normal(mu, sigma, &p).unwrap();
    let x = dists::fresh_var(normal.params(), &p);
    let density = dists::pdf(&normal, x, &p);
    let integrand = p.mul(vec![
        p.mul(vec![p.integer(-1), p.func("log", vec![density])]),
        density,
    ]);

    // `½log(2πeσ)` instead of `½log(2πeσ²)` — right at σ = 1, wrong elsewhere,
    // and off by a smooth factor rather than an obvious one.
    let wrong = p.add(vec![
        p.mul(vec![
            p.rational(1, 2),
            p.func("log", vec![p.mul(vec![p.integer(2), pi(&p), sigma])]),
        ]),
        p.rational(1, 2),
    ]);
    let err = verify::check(simplify(wrong, &p).value, integrand, x, &normal, &p).unwrap_err();
    assert_eq!(err.code(), "E-PROB-005");

    // A divergence with the two scales the wrong way round: `log(σ₁/σ₂)`
    // instead of `log(σ₂/σ₁)`, which is still 0 on the diagonal and still
    // finite everywhere.
    let q = Distribution::normal(sym(&p, "mu_h"), sym(&p, "sigma_h"), &p).unwrap();
    let (dp, dq) = (dists::pdf(&normal, x, &p), dists::pdf(&q, x, &p));
    let kl_integrand = p.mul(vec![p.func("log", vec![div(dp, dq, &p)]), dp]);
    let good = kl_divergence(&normal, &q, None, &p).unwrap().value;
    let flipped = simplify(
        p.add(vec![
            good,
            p.mul(vec![
                p.integer(-2),
                p.func("log", vec![div(q.params()[1], sigma, &p)]),
            ]),
        ]),
        &p,
    )
    .value;
    let err = verify::check_pair(
        flipped,
        kl_integrand,
        x,
        &normal,
        &q,
        &q.constraints(&p),
        &p,
    )
    .unwrap_err();
    assert_eq!(err.code(), "E-PROB-005");
}
