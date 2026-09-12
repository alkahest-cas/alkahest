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
