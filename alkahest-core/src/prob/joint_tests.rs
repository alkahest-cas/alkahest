//! Tests for the G10f additions: symbolic-`n` Binomial, NegativeBinomial,
//! the closed-form expectation table, simplified standardised moments, the
//! multivariate normal and independent-block joint laws, sums of independent
//! variates, and order statistics.
//!
//! Expected values are reference identities or numbers computed outside
//! alkahest (mpmath / scipy.stats), pasted in as literals.

use rug::Float;

use super::quad::{eval_ball, VERIFY_PREC};
use super::*;
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

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

macro_rules! assert_close {
    ($a:expr, $b:expr) => {{
        let (a, b): (f64, f64) = ($a, $b);
        assert!(
            (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0),
            "{a} != {b}"
        );
    }};
}

fn is_int(p: &ExprPool, e: ExprId, v: i64) -> bool {
    matches!(p.get(e), ExprData::Integer(n) if n.0 == v)
}

// ---------------------------------------------------------------------------
// Symbolic-n Binomial
// ---------------------------------------------------------------------------

#[test]
fn binomial_with_symbolic_n_has_closed_form_moments() {
    let p = ExprPool::new();
    let (n, q) = (sym(&p, "n"), sym(&p, "p"));
    let d = Distribution::binomial(n, q, &p).expect("symbolic n is accepted");
    assert_eq!(d.support(&p), Support::IntegersUpToSymbolic(n));
    let env = [("n", 7.0), ("p", 0.3)];
    assert_close!(at(&p, d.mean(&p).unwrap().value, &env), 2.1);
    assert_close!(at(&p, d.variance(&p).unwrap().value, &env), 1.47);
    // n integrality is published, not assumed silently.
    let conds = take_prob_side_conditions();
    assert!(
        conds.iter().any(
            |c| matches!(c, crate::deriv::SideCondition::InDomain(e, Domain::Integer) if *e == n)
        ),
        "{conds:?}"
    );
    // scipy.stats.binom(7, 0.3).moment(3)
    assert_close!(at(&p, d.moment(3, &p).unwrap().value, &env), 19.11);
    // skewness (1-2p)/sqrt(np(1-p)) = 0.4/sqrt(1.47)
    assert_close!(
        at(&p, d.skewness(&p).unwrap().value, &env),
        0.4 / 1.47f64.sqrt()
    );
    let t = sym(&p, "t");
    let mgf = d.moment_generating_function(t, &p).unwrap().value;
    let want = (0.7 + 0.3 * 0.4f64.exp()).powi(7);
    assert_close!(at(&p, mgf, &[("n", 7.0), ("p", 0.3), ("t", 0.4)]), want);
    assert_close!(
        at(&p, d.factorial_moment(2, &p).unwrap().value, &env),
        7.0 * 6.0 * 0.09
    );
}

#[test]
fn binomial_with_symbolic_n_refuses_what_needs_enumeration() {
    let p = ExprPool::new();
    let (n, q, x) = (sym(&p, "n"), sym(&p, "p"), sym(&p, "x"));
    let d = Distribution::binomial(n, q, &p).unwrap();
    assert_eq!(d.entropy(None, &p).unwrap_err().code(), "E-PROB-004");
    assert_eq!(d.cdf(x, &p).unwrap_err().code(), "E-PROB-004");
    let f = p.func("exp", vec![x]);
    assert_eq!(expectation(f, x, &d, &p).unwrap_err().code(), "E-PROB-002");
    // A polynomial f closes through the moments: E[X²] = np(1-p) + n²p².
    let r = expectation(p.pow(x, p.integer(2)), x, &d, &p).unwrap();
    assert_close!(at(&p, r.value, &[("n", 7.0), ("p", 0.3)]), 1.47 + 4.41);
}

// ---------------------------------------------------------------------------
// NegativeBinomial
// ---------------------------------------------------------------------------

#[test]
fn negative_binomial_moments_and_generating_functions() {
    let p = ExprPool::new();
    let (r, q, t, z) = (sym(&p, "r"), sym(&p, "p"), sym(&p, "t"), sym(&p, "z"));
    let d = Distribution::negative_binomial(r, q, &p).unwrap();
    let env = [("r", 3.0), ("p", 0.4)];
    // scipy.stats.nbinom(3, 0.4): mean 4.5, var 11.25, skew 1.1925695879998879
    assert_close!(at(&p, d.mean(&p).unwrap().value, &env), 4.5);
    assert_close!(at(&p, d.variance(&p).unwrap().value, &env), 11.25);
    assert_close!(
        at(&p, d.skewness(&p).unwrap().value, &env),
        1.192_569_587_999_887_9
    );
    let mgf = d.moment_generating_function(t, &p).unwrap().value;
    let want = (0.4 / (1.0 - 0.6 * 0.2f64.exp())).powi(3);
    assert_close!(at(&p, mgf, &[("r", 3.0), ("p", 0.4), ("t", 0.2)]), want);
    let pgf = d.probability_generating_function(z, &p).unwrap().value;
    assert_close!(
        at(&p, pgf, &[("r", 3.0), ("p", 0.4), ("z", 0.5)]),
        (0.4f64 / 0.7).powi(3)
    );
    assert_eq!(d.entropy(None, &p).unwrap_err().code(), "E-PROB-004");
    // Outside the strip t < -log(1-p) the MGF diverges.
    let d2 = Distribution::negative_binomial(p.integer(2), p.rational(1, 2), &p).unwrap();
    assert_eq!(
        d2.moment_generating_function(p.integer(1), &p)
            .unwrap_err()
            .code(),
        "E-PROB-006"
    );
    assert_eq!(
        Distribution::negative_binomial(p.integer(2), p.integer(0), &p)
            .unwrap_err()
            .code(),
        "E-PROB-001"
    );
}

#[test]
fn negative_binomial_kl_with_equal_shape() {
    let p = ExprPool::new();
    let a = Distribution::negative_binomial(p.integer(3), p.rational(2, 5), &p).unwrap();
    let b = Distribution::negative_binomial(p.integer(3), p.rational(1, 2), &p).unwrap();
    // 3 log(0.4/0.5) + 4.5 log(0.6/0.5)
    let want = 3.0 * (0.8f64).ln() + 4.5 * (1.2f64).ln();
    assert_close!(
        at(&p, kl_divergence(&a, &b, None, &p).unwrap().value, &[]),
        want
    );
}

// ---------------------------------------------------------------------------
// Standardised moments simplify
// ---------------------------------------------------------------------------

#[test]
fn normal_skewness_and_excess_kurtosis_are_literal_zero() {
    let p = ExprPool::new();
    let d = Distribution::normal(sym(&p, "mu"), sym(&p, "sigma"), &p).unwrap();
    let s = d.skewness(&p).unwrap().value;
    let k = d.excess_kurtosis(&p).unwrap().value;
    assert!(is_int(&p, s, 0), "{}", p.display(s));
    assert!(is_int(&p, k, 0), "{}", p.display(k));
    let e = Distribution::exponential(sym(&p, "lam"), &p).unwrap();
    // (λ⁻²)^{3/2} is not folded to λ⁻³ without λ > 0, so the exponential's
    // values are checked numerically rather than as literals.
    let env = [("lam", 1.7)];
    assert_close!(at(&p, e.skewness(&p).unwrap().value, &env), 2.0);
    assert_close!(at(&p, e.excess_kurtosis(&p).unwrap().value, &env), 6.0);
}

// ---------------------------------------------------------------------------
// Closed-form expectation table
// ---------------------------------------------------------------------------

#[test]
fn normal_tilted_and_trigonometric_expectations() {
    let p = ExprPool::new();
    let (mu, s, x) = (sym(&p, "mu"), sym(&p, "sigma"), sym(&p, "x"));
    let d = Distribution::normal(mu, s, &p).unwrap();
    let env = [("mu", 0.3), ("sigma", 0.7)];
    // E[X² e^X] = e^{μ+σ²/2}((μ+σ²)² + σ²)
    let f = p.mul(vec![p.pow(x, p.integer(2)), p.func("exp", vec![x])]);
    let got = at(&p, expectation(f, x, &d, &p).unwrap().value, &env);
    let m: f64 = 0.3 + 0.49;
    assert_close!(got, (0.3f64 + 0.245).exp() * (m * m + 0.49));
    // E[cos X] = e^{-σ²/2} cos μ
    let got = at(
        &p,
        expectation(p.func("cos", vec![x]), x, &d, &p)
            .unwrap()
            .value,
        &env,
    );
    assert_close!(got, (-0.245f64).exp() * 0.3f64.cos());
}

#[test]
fn gamma_power_log_and_reciprocal_expectations() {
    let p = ExprPool::new();
    let (k, th, x, s) = (sym(&p, "k"), sym(&p, "theta"), sym(&p, "x"), sym(&p, "s"));
    let d = Distribution::gamma(k, th, &p).unwrap();
    let env = [("k", 2.5), ("theta", 1.3)];
    // E[X²] = k(k+1)θ²
    let r = expectation(p.pow(x, p.integer(2)), x, &d, &p).unwrap();
    assert_close!(at(&p, r.value, &env), 2.5 * 3.5 * 1.69);
    // E[log X] = ψ(k) + log θ; ψ(2.5) = 0.70315664064524318723 (mpmath)
    let r = expectation(p.func("log", vec![x]), x, &d, &p).unwrap();
    assert_close!(at(&p, r.value, &env), 0.703_156_640_645_243_2 + 1.3f64.ln());
    // E[1/X] = 1/(θ(k-1)), carrying k - 1 > 0.
    let r = expectation(p.pow(x, p.integer(-1)), x, &d, &p).unwrap();
    assert_close!(at(&p, r.value, &env), 1.0 / (1.3 * 1.5));
    let conds = take_prob_side_conditions();
    assert!(!conds.is_empty());
    // E[X^s] = θ^s Γ(k+s)/Γ(k); Γ(3.2), Γ(2.5) from mpmath
    let r = expectation(p.pow(x, s), x, &d, &p).unwrap();
    assert_close!(
        at(&p, r.value, &[("k", 2.5), ("theta", 1.3), ("s", 0.7)]),
        1.3f64.powf(0.7) * 2.423_965_479_935_368_3 / 1.329_340_388_179_137
    );
    // Shape 1: E[1/X] diverges, decidably.
    let e1 = Distribution::gamma(p.integer(1), p.integer(2), &p).unwrap();
    assert_eq!(
        expectation(p.pow(x, p.integer(-1)), x, &e1, &p)
            .unwrap_err()
            .code(),
        "E-PROB-006"
    );
}

#[test]
fn beta_and_lognormal_log_expectations() {
    let p = ExprPool::new();
    let x = sym(&p, "x");
    let b = Distribution::beta(p.integer(2), p.integer(3), &p).unwrap();
    // ψ(2) - ψ(5) = -(1/2 + 1/3 + 1/4)
    let r = expectation(p.func("log", vec![x]), x, &b, &p).unwrap();
    assert_close!(at(&p, r.value, &[]), -(0.5 + 1.0 / 3.0 + 0.25));
    let ln = Distribution::log_normal(p.rational(1, 5), p.rational(1, 2), &p).unwrap();
    let r = expectation(p.func("log", vec![x]), x, &ln, &p).unwrap();
    assert_close!(at(&p, r.value, &[]), 0.2);
    let r = expectation(p.func("sqrt", vec![x]), x, &ln, &p).unwrap();
    assert_close!(at(&p, r.value, &[]), (0.1f64 + 0.125 * 0.25).exp());
}

// ---------------------------------------------------------------------------
// Multivariate normal
// ---------------------------------------------------------------------------

fn mvn2(p: &ExprPool) -> MultivariateNormal {
    let m = vec![sym(p, "m1"), sym(p, "m2")];
    let (a, b, c) = (sym(p, "s11"), sym(p, "s12"), sym(p, "s22"));
    MultivariateNormal::new(m, vec![vec![a, b], vec![b, c]], p).unwrap()
}

const ENV2: [(&str, f64); 5] = [
    ("m1", 0.4),
    ("m2", -1.1),
    ("s11", 2.0),
    ("s12", 0.6),
    ("s22", 1.5),
];

#[test]
fn mvn_construction_checks_symmetry_and_definiteness() {
    let p = ExprPool::new();
    let (a, b) = (sym(&p, "a"), sym(&p, "b"));
    let z = p.integer(0);
    let not_sym = MultivariateNormal::new(
        vec![z, z],
        vec![vec![p.integer(1), a], vec![b, p.integer(1)]],
        &p,
    );
    assert_eq!(not_sym.unwrap_err().code(), "E-PROB-001");
    // det = 1 - 4 < 0
    let indefinite = MultivariateNormal::new(
        vec![z, z],
        vec![
            vec![p.integer(1), p.integer(2)],
            vec![p.integer(2), p.integer(1)],
        ],
        &p,
    );
    assert_eq!(indefinite.unwrap_err().code(), "E-PROB-001");
    let m = mvn2(&p);
    assert_eq!(m.constraints(&p).len(), 2);
}

#[test]
fn mvn_moments_by_stein_identity() {
    let p = ExprPool::new();
    let m = mvn2(&p);
    let (x, y) = (sym(&p, "X"), sym(&p, "Y"));
    let blocks = vec![JointBlock::Gaussian(vec![x, y], m)];
    let e = |f: ExprId| at(&p, joint_expectation(f, &blocks, &p).unwrap().value, &ENV2);
    assert_close!(e(p.mul(vec![x, y])), 0.6 + 0.4 * -1.1);
    // Isserlis with mean: E[X²Y²] = (s11 + m1²)(s22 + m2²) + 2 s12² + 4 m1 m2 s12
    let want = (2.0 + 0.16) * (1.5 + 1.21) + 2.0 * 0.36 + 4.0 * 0.4 * -1.1 * 0.6;
    assert_close!(
        e(p.mul(vec![p.pow(x, p.integer(2)), p.pow(y, p.integer(2))])),
        want
    );
    // Cov(X + Y, X - Y) = s11 - s22
    let cov = covariance(
        p.add(vec![x, y]),
        p.add(vec![x, p.mul(vec![p.integer(-1), y])]),
        &blocks,
        &p,
    )
    .unwrap()
    .value;
    assert_close!(at(&p, cov, &ENV2), 0.5);
    // Var(2X - 3Y) = 4 s11 - 12 s12 + 9 s22
    let v = joint_variance(
        p.add(vec![
            p.mul(vec![p.integer(2), x]),
            p.mul(vec![p.integer(-3), y]),
        ]),
        &blocks,
        &p,
    )
    .unwrap()
    .value;
    assert_close!(at(&p, v, &ENV2), 8.0 - 7.2 + 13.5);
    // E[e^{X}] = M(1, 0) = e^{m1 + s11/2}
    assert_close!(e(p.func("exp", vec![x])), (0.4f64 + 1.0).exp());
    // E[X e^{Y}] = e^{m2 + s22/2}(m1 + s12)
    assert_close!(
        e(p.mul(vec![x, p.func("exp", vec![y])])),
        (-1.1f64 + 0.75).exp() * (0.4 + 0.6)
    );
    // A non-polynomial factor of a Gaussian block is refused.
    assert_eq!(
        joint_expectation(p.func("sin", vec![x]), &blocks, &p)
            .unwrap_err()
            .code(),
        "E-PROB-002"
    );
}

#[test]
fn mvn_conditional_marginal_transform_and_quadratic_form() {
    let p = ExprPool::new();
    let m = mvn2(&p);
    let a = sym(&p, "a");
    let c = m.conditional(&[1], &[a], &p).unwrap();
    let env: Vec<(&str, f64)> = ENV2.iter().copied().chain([("a", 0.25)]).collect();
    // μ₁ + s12/s22 (a - μ₂), s11 - s12²/s22
    assert_close!(
        at(&p, c.mean_vector()[0], &env),
        0.4 + 0.6 / 1.5 * (0.25 + 1.1)
    );
    assert_close!(at(&p, c.covariance_matrix()[0][0], &env), 2.0 - 0.36 / 1.5);
    let mg = m.marginal(&[1]).unwrap();
    assert_eq!(mg.mean_vector(), &[sym(&p, "m2")]);
    // A = [[1, 1]]: X + Y ~ N(m1 + m2, s11 + 2 s12 + s22)
    let one = p.integer(1);
    let t = m.linear_transform(&[vec![one, one]], None, &p).unwrap();
    assert_close!(at(&p, t.covariance_matrix()[0][0], &ENV2), 4.7);
    // A rank-deficient map is a degenerate normal and is refused.
    let numeric = MultivariateNormal::new(
        vec![p.integer(0), p.integer(0)],
        vec![
            vec![p.integer(2), p.integer(1)],
            vec![p.integer(1), p.integer(2)],
        ],
        &p,
    )
    .unwrap();
    assert_eq!(
        numeric
            .linear_transform(&[vec![one, one], vec![one, one]], None, &p)
            .unwrap_err()
            .code(),
        "E-PROB-001"
    );
    // E[XᵀAX] = tr(AΣ) + μᵀAμ with A = [[1, 2], [0, 3]]
    let qa = vec![
        vec![p.integer(1), p.integer(2)],
        vec![p.integer(0), p.integer(3)],
    ];
    let q = m.quadratic_form_expectation(&qa, &p).unwrap().value;
    let want = (2.0 + 2.0 * 0.6 + 3.0 * 1.5) + (0.16 + 2.0 * 0.4 * -1.1 + 3.0 * 1.21);
    assert_close!(at(&p, q, &ENV2), want);
    // Density at a point: scipy.stats.multivariate_normal([0.4,-1.1],
    // [[2,.6],[.6,1.5]]).pdf([0.9, -0.2]) = 0.07436017915741862
    let pdf = m
        .pdf(&[p.rational(9, 10), p.rational(-1, 5)], &p)
        .unwrap()
        .value;
    assert_close!(at(&p, pdf, &ENV2), 0.074_360_179_157_418_62);
}

#[test]
fn independent_blocks_factor_and_couple_only_inside_a_gaussian() {
    let p = ExprPool::new();
    let (x, y, mu, s, lam) = (
        sym(&p, "X"),
        sym(&p, "Y"),
        sym(&p, "mu"),
        sym(&p, "sigma"),
        sym(&p, "lam"),
    );
    let blocks = vec![
        JointBlock::Univariate(x, Distribution::normal(mu, s, &p).unwrap()),
        JointBlock::Univariate(y, Distribution::exponential(lam, &p).unwrap()),
    ];
    let env = [("mu", 0.5), ("sigma", 1.2), ("lam", 2.0)];
    let r = joint_expectation(p.mul(vec![x, y]), &blocks, &p).unwrap();
    assert_close!(at(&p, r.value, &env), 0.25);
    // Independent ⇒ Cov(X, Y) = 0, Var(X + Y) = σ² + 1/λ².
    let c = covariance(x, y, &blocks, &p).unwrap().value;
    assert_close!(at(&p, c, &env), 0.0);
    let v = joint_variance(p.add(vec![x, y]), &blocks, &p)
        .unwrap()
        .value;
    assert_close!(at(&p, v, &env), 1.44 + 0.25);
    // sin(X + Y) does not factor over independent blocks.
    assert_eq!(
        joint_expectation(p.func("sin", vec![p.add(vec![x, y])]), &blocks, &p)
            .unwrap_err()
            .code(),
        "E-PROB-002"
    );
}

// ---------------------------------------------------------------------------
// Sums of independent variates
// ---------------------------------------------------------------------------

#[test]
fn sums_of_independent_variates_in_closed_families() {
    let p = ExprPool::new();
    let (m1, m2, s1, s2) = (sym(&p, "m1"), sym(&p, "m2"), sym(&p, "s1"), sym(&p, "s2"));
    let n = sum_independent(
        &[
            Distribution::normal(m1, s1, &p).unwrap(),
            Distribution::normal(m2, s2, &p).unwrap(),
        ],
        &p,
    )
    .unwrap();
    assert_eq!(n.kind(), DistKind::Normal);
    let env = [("m1", 1.0), ("m2", 2.0), ("s1", 3.0), ("s2", 4.0)];
    assert_close!(at(&p, n.params()[0], &env), 3.0);
    assert_close!(at(&p, n.params()[1], &env), 5.0);
    let th = sym(&p, "theta");
    let g = sum_independent(
        &[
            Distribution::gamma(p.integer(2), th, &p).unwrap(),
            Distribution::exponential(p.pow(th, p.integer(-1)), &p).unwrap(),
        ],
        &p,
    )
    .unwrap();
    assert_eq!(g.kind(), DistKind::Gamma);
    assert!(is_int(&p, g.params()[0], 3));
    let q = sym(&p, "q");
    let (n1, n2) = (sym(&p, "n1"), sym(&p, "n2"));
    let b = sum_independent(
        &[
            Distribution::binomial(n1, q, &p).unwrap(),
            Distribution::binomial(n2, q, &p).unwrap(),
            Distribution::bernoulli(q, &p).unwrap(),
        ],
        &p,
    )
    .unwrap();
    assert_eq!(b.kind(), DistKind::Binomial);
    // Different scales: not a Gamma.
    let e = sum_independent(
        &[
            Distribution::exponential(p.integer(1), &p).unwrap(),
            Distribution::exponential(p.integer(2), &p).unwrap(),
        ],
        &p,
    );
    assert_eq!(e.unwrap_err().code(), "E-PROB-002");
}

// ---------------------------------------------------------------------------
// Order statistics
// ---------------------------------------------------------------------------

#[test]
fn order_statistics_with_symbolic_n() {
    let p = ExprPool::new();
    let (n, lam) = (sym(&p, "n"), sym(&p, "lam"));
    let u = Distribution::uniform(p.integer(0), p.integer(1), &p).unwrap();
    let mx = order_statistic(&u, n, OrderKind::Max, &p).unwrap();
    assert_close!(at(&p, mx.mean(&p).unwrap().value, &[("n", 4.0)]), 0.8);
    // n/((n+1)²(n+2)) at n = 4: 4/150
    assert_close!(
        at(&p, mx.variance(&p).unwrap().value, &[("n", 4.0)]),
        4.0 / 150.0
    );
    assert_eq!(mx.distribution(&p).unwrap().kind(), DistKind::Beta);
    let e = Distribution::exponential(lam, &p).unwrap();
    let emax = order_statistic(&e, n, OrderKind::Max, &p).unwrap();
    let env = [("n", 3.0), ("lam", 2.0)];
    assert_close!(
        at(&p, emax.mean(&p).unwrap().value, &env),
        (1.0 + 0.5 + 1.0 / 3.0) / 2.0
    );
    assert_close!(
        at(&p, emax.variance(&p).unwrap().value, &env),
        (1.0 + 0.25 + 1.0 / 9.0) / 4.0
    );
    let emin = order_statistic(&e, n, OrderKind::Min, &p).unwrap();
    assert_close!(at(&p, emin.mean(&p).unwrap().value, &env), 1.0 / 6.0);
    assert_eq!(emin.distribution(&p).unwrap().kind(), DistKind::Exponential);
    let x = sym(&p, "x");
    let cdf = emax.cdf(x, &p).unwrap().value;
    assert_close!(
        at(&p, cdf, &[("n", 3.0), ("lam", 2.0), ("x", 0.7)]),
        (1.0 - (-1.4f64).exp()).powi(3)
    );
    assert_eq!(
        order_statistic(&e, p.rational(5, 2), OrderKind::Max, &p)
            .unwrap_err()
            .code(),
        "E-PROB-001"
    );
    let normal = Distribution::normal(p.integer(0), p.integer(1), &p).unwrap();
    // A Normal base is accepted (density and CDF are exact); the moments of a
    // symbolic-n maximum are not tabled and refuse.
    let nm = order_statistic(&normal, n, OrderKind::Max, &p).unwrap();
    assert_eq!(nm.mean(&p).unwrap_err().code(), "E-PROB-002");
}
