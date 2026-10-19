//! Tests for the F23 follow-ups: general `k`-th order statistics, the
//! labelled asymptotic entropy of `Binomial` / `NegativeBinomial` /
//! `Poisson`, `E[xᵏ cos(ωx+φ)]` under a `Normal`, and the information
//! measures of the multivariate normal.
//!
//! Reference numbers were computed outside alkahest (mpmath quadrature,
//! numpy / scipy) and pasted in as literals.

use rug::Float;

use super::quad::{eval_ball, VERIFY_PREC};
use super::*;
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprId, ExprPool};

fn sym(p: &ExprPool, name: &str) -> ExprId {
    p.symbol(name, Domain::Real)
}

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

macro_rules! close {
    ($a:expr, $b:expr, $tol:expr) => {{
        let (a, b, tol): (f64, f64, f64) = ($a, $b, $tol);
        assert!(
            (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0),
            "{a} != {b}"
        );
    }};
}

// ---------------------------------------------------------------------------
// k-th order statistics
// ---------------------------------------------------------------------------

#[test]
fn kth_order_statistic_density_and_cdf_for_a_normal_base() {
    let p = ExprPool::new();
    let x = sym(&p, "x");
    let normal = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let os = kth_order_statistic(&normal, p.integer(5), p.integer(2), &p).unwrap();
    // scipy: 20 F (1-F)^3 f at 0.3, and the binomial tail
    close!(
        at(&p, os.pdf(x, &p).unwrap().value, &[("x", 0.3)]),
        0.26291489935968493,
        1e-12
    );
    close!(
        at(&p, os.cdf(x, &p).unwrap().value, &[("x", 0.3)]),
        0.9260068331017236,
        1e-12
    );
}

#[test]
fn kth_order_statistic_moments_via_the_integrator() {
    let p = ExprPool::new();
    let lam = sym(&p, "lam");
    let n = p.integer(3);
    let beta = Distribution::beta(p.integer(2), p.integer(3), &p).unwrap();
    let gamma = Distribution::gamma(p.integer(2), p.integer(1), &p).unwrap();
    let exp = Distribution::exponential(lam, &p).unwrap();
    // (law, k, mean, variance) -- mpmath quadrature
    let beta_cases = [
        (1, 0.23216783216783216, 0.015828367436759045),
        (2, 0.3928071928071928, 0.02052768410410768),
        (3, 0.5750249750249751, 0.024790833542082294),
    ];
    for (k, m, v) in beta_cases {
        let os = kth_order_statistic(&beta, n, p.integer(k), &p).unwrap();
        close!(at(&p, os.mean(&p).unwrap().value, &[]), m, 1e-12);
        close!(at(&p, os.variance(&p).unwrap().value, &[]), v, 1e-11);
    }
    let gamma_cases = [
        (1, 0.9629629629629629, 0.3813443072702332),
        (2, 1.8240740740740742, 0.8054698216735254),
        (3, 3.212962962962963, 2.2355109739368997),
    ];
    for (k, m, v) in gamma_cases {
        let os = kth_order_statistic(&gamma, n, p.integer(k), &p).unwrap();
        close!(at(&p, os.mean(&p).unwrap().value, &[]), m, 1e-12);
        close!(at(&p, os.variance(&p).unwrap().value, &[]), v, 1e-11);
    }
    // Exponential: Rényi representation, symbolic rate. k = 2 of 3:
    // mean 5/(6λ), variance 13/(36λ²).
    let os = kth_order_statistic(&exp, n, p.integer(2), &p).unwrap();
    close!(
        at(&p, os.mean(&p).unwrap().value, &[("lam", 2.0)]),
        5.0 / 12.0,
        1e-12
    );
    close!(
        at(&p, os.variance(&p).unwrap().value, &[("lam", 2.0)]),
        13.0 / 144.0,
        1e-12
    );
}

#[test]
fn normal_order_statistics_close_for_n_two_and_three_and_refuse_beyond() {
    let p = ExprPool::new();
    let normal = Distribution::normal(p.integer(1), p.integer(2), &p).unwrap();
    let pi = std::f64::consts::PI;
    // n = 3: Z_(3:3) has mean 3/(2√π), variance 1 + √3/(2π) - 9/(4π)
    let top = order_statistic(&normal, p.integer(3), OrderKind::Max, &p).unwrap();
    close!(
        at(&p, top.mean(&p).unwrap().value, &[]),
        1.0 + 2.0 * 1.5 / pi.sqrt(),
        1e-12
    );
    close!(
        at(&p, top.variance(&p).unwrap().value, &[]),
        4.0 * (1.0 + 3f64.sqrt() / (2.0 * pi) - 9.0 / (4.0 * pi)),
        1e-12
    );
    // median of 3: mean μ, variance σ²(1 - √3/π)
    let med = kth_order_statistic(&normal, p.integer(3), p.integer(2), &p).unwrap();
    close!(at(&p, med.mean(&p).unwrap().value, &[]), 1.0, 1e-12);
    close!(
        at(&p, med.variance(&p).unwrap().value, &[]),
        4.0 * (1.0 - 3f64.sqrt() / pi),
        1e-12
    );
    // n = 2 minimum: μ - σ/√π, σ²(1 - 1/π)
    let lo = order_statistic(&normal, p.integer(2), OrderKind::Min, &p).unwrap();
    close!(
        at(&p, lo.mean(&p).unwrap().value, &[]),
        1.0 - 2.0 / pi.sqrt(),
        1e-12
    );
    close!(
        at(&p, lo.variance(&p).unwrap().value, &[]),
        4.0 * (1.0 - 1.0 / pi),
        1e-12
    );
    // n = 4: the mean involves arctan constants the integrator does not close.
    let four = order_statistic(&normal, p.integer(4), OrderKind::Max, &p).unwrap();
    assert_eq!(four.mean(&p).unwrap_err().code(), "E-PROB-003");
}

#[test]
fn kth_order_statistic_validation_and_refusals() {
    let p = ExprPool::new();
    let (n, k, x) = (sym(&p, "n"), sym(&p, "k"), sym(&p, "x"));
    let normal = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    let err = |r: Result<OrderStatistic, ProbError>| r.unwrap_err().code();
    assert_eq!(
        err(kth_order_statistic(&normal, p.integer(3), p.integer(4), &p)),
        "E-PROB-001"
    );
    assert_eq!(
        err(kth_order_statistic(&normal, p.integer(3), p.integer(0), &p)),
        "E-PROB-001"
    );
    let poisson = Distribution::poisson(p.integer(2), &p).unwrap();
    assert_eq!(
        err(kth_order_statistic(
            &poisson,
            p.integer(3),
            p.integer(2),
            &p
        )),
        "E-PROB-002"
    );
    // extremes are normalised
    let one = kth_order_statistic(&normal, p.integer(5), p.integer(1), &p).unwrap();
    assert_eq!(one.kind(), OrderKind::Min);
    assert!(one.is_extreme());
    let top = kth_order_statistic(&normal, p.integer(5), p.integer(5), &p).unwrap();
    assert_eq!(top.kind(), OrderKind::Max);
    // symbolic n, k: density is a Gamma ratio, evaluates like the literal one
    let sym_os = kth_order_statistic(&normal, n, k, &p).unwrap();
    let pdf = sym_os.pdf(x, &p).unwrap().value;
    close!(
        at(&p, pdf, &[("n", 5.0), ("k", 2.0), ("x", 0.3)]),
        0.26291489935968493,
        1e-10
    );
    assert_eq!(sym_os.cdf(x, &p).unwrap_err().code(), "E-PROB-004");
    assert_eq!(sym_os.mean(&p).unwrap_err().code(), "E-PROB-002");
    // a Gamma with a non-integer shape has no closed-form CDF
    let g = Distribution::gamma(p.rational(5, 2), p.integer(1), &p).unwrap();
    let gos = kth_order_statistic(&g, p.integer(3), p.integer(2), &p).unwrap();
    assert_eq!(gos.cdf(x, &p).unwrap_err().code(), "E-PROB-004");
    // Uniform(0,1) k-th is Beta(k, n-k+1)
    let u = Distribution::uniform(p.integer(0), p.integer(1), &p).unwrap();
    let uk = kth_order_statistic(&u, p.integer(5), p.integer(2), &p).unwrap();
    assert_eq!(uk.distribution(&p).unwrap().kind(), DistKind::Beta);
}

// ---------------------------------------------------------------------------
// Asymptotic entropy
// ---------------------------------------------------------------------------

#[test]
fn asymptotic_entropy_of_the_lattice_laws() {
    let p = ExprPool::new();
    let (n, q, r, lam) = (sym(&p, "n"), sym(&p, "p"), sym(&p, "r"), sym(&p, "lam"));
    // exact sums computed with scipy
    let b = Distribution::binomial(n, q, &p).unwrap();
    close!(
        at(
            &p,
            entropy_asymptotic(&b, None, &p).unwrap().value,
            &[("n", 200.0), ("p", 1.0 / 3.0)]
        ),
        3.3158456856538785,
        1e-4
    );
    let nb = Distribution::negative_binomial(r, q, &p).unwrap();
    close!(
        at(
            &p,
            entropy_asymptotic(&nb, None, &p).unwrap().value,
            &[("r", 80.0), ("p", 1.0 / 3.0)]
        ),
        4.501473615629695,
        1e-4
    );
    let po = Distribution::poisson(lam, &p).unwrap();
    close!(
        at(
            &p,
            entropy_asymptotic(&po, None, &p).unwrap().value,
            &[("lam", 40.0)]
        ),
        3.2612680129376925,
        1e-4
    );
    // the exact entropy still refuses, and says where to go
    let err = entropy(&b, None, &p).unwrap_err();
    assert_eq!(err.code(), "E-PROB-004");
    assert!(err.to_string().contains("entropy_asymptotic"));
    // the derivation is labelled
    let d = entropy_asymptotic(&b, None, &p).unwrap();
    assert!(d
        .log
        .0
        .iter()
        .any(|s| s.rule_name == "prob_entropy_asymptotic_edgeworth"));
    let normal = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    assert_eq!(
        entropy_asymptotic(&normal, None, &p).unwrap_err().code(),
        "E-PROB-002"
    );
}

// ---------------------------------------------------------------------------
// E[x^k cos(wx + phi)] under a Normal
// ---------------------------------------------------------------------------

#[test]
fn polynomial_times_trig_under_a_normal_matches_mpmath() {
    let p = ExprPool::new();
    let x = sym(&p, "x");
    let normal = |m: (i64, i64), s: (i64, i64)| {
        Distribution::normal(p.rational(m.0, m.1), p.rational(s.0, s.1), &p).unwrap()
    };
    let pow = |k: i64| p.pow(x, p.integer(k));
    // E[x^3 cos 2x], N(3/10, 7/10)
    let f1 = p.mul(vec![
        pow(3),
        p.func("cos", vec![p.mul(vec![p.integer(2), x])]),
    ]);
    let e1 = expectation(f1, x, &normal((3, 10), (7, 10)), &p).unwrap();
    close!(at(&p, e1.value, &[]), -0.2846814968107432, 1e-11);
    // E[x^2 e^{x/2} sin(3x/2 + 1/5)], N(-2/5, 6/5)
    let f2 = p.mul(vec![
        pow(2),
        p.func("exp", vec![p.mul(vec![p.rational(1, 2), x])]),
        p.func(
            "sin",
            vec![p.add(vec![p.mul(vec![p.rational(3, 2), x]), p.rational(1, 5)])],
        ),
    ]);
    let e2 = expectation(f2, x, &normal((-2, 5), (6, 5)), &p).unwrap();
    close!(at(&p, e2.value, &[]), -0.17243464916207216, 1e-11);
    // E[x^4 cos x], N(0,1) = -2 e^{-1/2}
    let f3 = p.mul(vec![pow(4), p.func("cos", vec![x])]);
    let e3 = expectation(f3, x, &normal((0, 1), (1, 1)), &p).unwrap();
    close!(at(&p, e3.value, &[]), -2.0 * (-0.5f64).exp(), 1e-12);
    // E[x sin x], N(0,1) = e^{-1/2}
    let f4 = p.mul(vec![x, p.func("sin", vec![x])]);
    let e4 = expectation(f4, x, &normal((0, 1), (1, 1)), &p).unwrap();
    close!(at(&p, e4.value, &[]), (-0.5f64).exp(), 1e-12);
    // E[x^5 cos(3x + 7/10)], N(11/10, 1/2)
    let f5 = p.mul(vec![
        pow(5),
        p.func(
            "cos",
            vec![p.add(vec![p.mul(vec![p.integer(3), x]), p.rational(7, 10)])],
        ),
    ]);
    let e5 = expectation(f5, x, &normal((11, 10), (1, 2)), &p).unwrap();
    close!(at(&p, e5.value, &[]), 2.676502092736405, 1e-10);
}

// ---------------------------------------------------------------------------
// Multivariate normal information measures
// ---------------------------------------------------------------------------

fn mvn(p: &ExprPool, mean: &[(i64, i64)], cov: &[&[(i64, i64)]]) -> MultivariateNormal {
    MultivariateNormal::new(
        mean.iter().map(|&(a, b)| p.rational(a, b)).collect(),
        cov.iter()
            .map(|r| r.iter().map(|&(a, b)| p.rational(a, b)).collect())
            .collect(),
        p,
    )
    .unwrap()
}

#[test]
fn mvn_entropy_kl_and_mutual_information_match_numpy() {
    let p = ExprPool::new();
    let one = (1, 1);
    let z = (0, 1);
    let m0 = [(0, 1), (1, 1), (-1, 1)];
    let s0: [&[(i64, i64)]; 3] = [
        &[(2, 1), (1, 2), (1, 10)],
        &[(1, 2), one, (3, 10)],
        &[(1, 10), (3, 10), (3, 2)],
    ];
    let m1 = [(1, 2), z, one];
    let s1: [&[(i64, i64)]; 3] = [
        &[one, (1, 5), z],
        &[(1, 5), (3, 1), (1, 2)],
        &[z, (1, 2), one],
    ];
    let a = mvn(&p, &m0, &s0);
    let b = mvn(&p, &m1, &s1);
    // numpy: 3-d KL, and group mutual informations
    close!(
        at(&p, a.kl_divergence(&b, None, &p).unwrap().value, &[]),
        3.4159208491228323,
        1e-12
    );
    close!(
        at(&p, a.kl_divergence(&a, None, &p).unwrap().value, &[]),
        0.0,
        1e-12
    );
    close!(
        at(
            &p,
            a.mutual_information(&[0], &[1, 2], None, &p).unwrap().value,
            &[]
        ),
        0.0672725387276844,
        1e-11
    );
    close!(
        at(
            &p,
            a.mutual_information(&[0, 1], &[2], None, &p).unwrap().value,
            &[]
        ),
        0.03144454427446682,
        1e-11
    );
    // 2-d entropy: ½ log((2πe)² det Σ), det = 7/4
    let c = mvn(&p, &[z, z], &[&[(2, 1), (1, 2)], &[(1, 2), one]]);
    close!(
        at(&p, c.entropy(None, &p).unwrap().value, &[]),
        3.1176849603770567,
        1e-12
    );
    // refusals
    assert_eq!(
        a.mutual_information(&[0], &[0, 1], None, &p)
            .unwrap_err()
            .code(),
        "E-PROB-002"
    );
    assert_eq!(
        a.kl_divergence(&c, None, &p).unwrap_err().code(),
        "E-PROB-002"
    );
}

#[test]
fn bivariate_normal_mutual_information_is_minus_half_log_one_minus_rho_squared() {
    let p = ExprPool::new();
    let rho = sym(&p, "rho");
    let m = MultivariateNormal::new(
        vec![p.integer(0), p.integer(0)],
        vec![vec![p.integer(1), rho], vec![rho, p.integer(1)]],
        &p,
    )
    .unwrap();
    let mi = m.mutual_information(&[0], &[1], None, &p).unwrap().value;
    close!(
        at(&p, mi, &[("rho", 0.6)]),
        -0.5 * (1.0f64 - 0.36).ln(),
        1e-13
    );
    // its hypotheses (positive definiteness: 1 - rho^2 > 0) are published
    assert!(!take_prob_side_conditions().is_empty());
    // standard deviations cancel symbolically
    let (sa, sb) = (sym(&p, "sa"), sym(&p, "sb"));
    let cov = vec![
        vec![p.pow(sa, p.integer(2)), p.mul(vec![rho, sa, sb])],
        vec![p.mul(vec![rho, sa, sb]), p.pow(sb, p.integer(2))],
    ];
    let m2 = MultivariateNormal::new(vec![p.integer(0), p.integer(0)], cov, &p).unwrap();
    let mi2 = m2.mutual_information(&[0], &[1], None, &p).unwrap().value;
    let shown = p.display(mi2).to_string();
    assert!(!shown.contains("sa") && !shown.contains("sb"), "{shown}");
    close!(
        at(&p, mi2, &[("rho", 0.6), ("sa", 2.0), ("sb", 0.3)]),
        0.22314355131420974,
        1e-13
    );
}
