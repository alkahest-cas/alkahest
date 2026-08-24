//! End-to-end checks of the continuous creative-telescoping engine through the
//! crate's `experimental` surface — the same route a downstream Rust caller
//! takes, rather than the crate-internal module paths the unit tests use.
//!
//! Each case pairs a classical definite integral with the recurrence it is
//! known to satisfy, and asserts both halves the engine is responsible for:
//! the certificate's recurrence coefficients, and the boundary verdict that
//! decides whether that recurrence transfers from the integrand to the
//! integral.

use alkahest_cas::experimental::{
    almkvist_zeilberger, dgosper, integral_boundary_status, AzOpts, IntegralBoundaryStatus,
    IntegrationLimit,
};
use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use std::collections::HashMap;

fn ratio_at(pool: &ExprPool, n: ExprId, num: ExprId, den: ExprId, ni: f64) -> f64 {
    let env = HashMap::from([(n, ni)]);
    let a = alkahest_cas::eval_f64(num, pool, &env).expect("coefficient evaluates");
    let b = alkahest_cas::eval_f64(den, pool, &env).expect("coefficient evaluates");
    a / b
}

/// `∫_0^∞ xⁿ·e^(−x) dx = n!`  ⇒  `f(n+1) = (n+1)·f(n)`, boundary vanishes.
#[test]
fn gamma_function_recurrence_end_to_end() {
    let pool = ExprPool::new();
    let n = pool.symbol("n", Domain::Real);
    let x = pool.symbol("x", Domain::Real);
    let f = pool.mul(vec![
        pool.pow(x, n),
        pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), x])]),
    ]);

    let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
    let r = &out.value;
    assert_eq!(r.order, 1);
    for ni in [1.0_f64, 4.0, 12.5] {
        let q = ratio_at(&pool, n, r.coeffs[0], r.coeffs[1], ni);
        assert!(
            (q + (ni + 1.0)).abs() < 1e-9,
            "a_0/a_1 must be -(n+1); got {q} at n={ni}"
        );
    }

    let status = integral_boundary_status(
        r,
        f,
        n,
        x,
        &pool,
        &IntegrationLimit::at(0),
        &IntegrationLimit::PosInfinity,
    )
    .expect("boundary analysis");
    assert_eq!(status.tag(), "vanishes");
    assert!(status.implies_integral_recurrence());
    // The one hypothesis left standing is exactly the convergence condition at
    // the origin, and it is reported rather than assumed.
    assert_eq!(status.conditions().len(), 1);
}

/// `∫_{−∞}^{∞} x^(2n)·e^(−x²) dx`  ⇒  `2·f(n+1) = (2n+1)·f(n)`, boundary
/// vanishes with no condition on `n` at all.
#[test]
fn gaussian_moments_end_to_end() {
    let pool = ExprPool::new();
    let n = pool.symbol("n", Domain::Real);
    let x = pool.symbol("x", Domain::Real);
    let f = pool.mul(vec![
        pool.pow(x, pool.mul(vec![pool.integer(2_i32), n])),
        pool.func(
            "exp",
            vec![pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))])],
        ),
    ]);

    let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
    let r = &out.value;
    assert_eq!(r.order, 1);
    for ni in [0.0_f64, 3.0, 8.5] {
        let q = ratio_at(&pool, n, r.coeffs[0], r.coeffs[1], ni);
        assert!(
            (q + (2.0 * ni + 1.0) / 2.0).abs() < 1e-9,
            "a_0/a_1 must be -(2n+1)/2; got {q} at n={ni}"
        );
    }

    let status = integral_boundary_status(
        r,
        f,
        n,
        x,
        &pool,
        &IntegrationLimit::NegInfinity,
        &IntegrationLimit::PosInfinity,
    )
    .expect("boundary analysis");
    assert_eq!(status.tag(), "vanishes");
    assert!(status.conditions().is_empty());
}

/// `∫_0^1 xⁿ dx = 1/(n+1)`: the certificate exists at order 0 and the boundary
/// term is `1/(n+1)`, *not* zero. The engine must return the inhomogeneous
/// reading — this is the case a module that collapsed `Unknown` into `Vanishes`
/// would turn into the false lemma `f(n) = 0`.
#[test]
fn nonvanishing_boundary_end_to_end() {
    let pool = ExprPool::new();
    let n = pool.symbol("n", Domain::Real);
    let x = pool.symbol("x", Domain::Real);
    let f = pool.pow(x, n);

    let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
    assert_eq!(out.value.order, 0);

    let status = integral_boundary_status(
        &out.value,
        f,
        n,
        x,
        &pool,
        &IntegrationLimit::at(0),
        &IntegrationLimit::at(1),
    )
    .expect("boundary analysis");
    match &status {
        IntegralBoundaryStatus::Nonzero { rhs, .. } => {
            for ni in [2.0_f64, 6.0] {
                let env = HashMap::from([(n, ni)]);
                let v = alkahest_cas::eval_f64(*rhs, &pool, &env).expect("rhs evaluates");
                assert!((v - 1.0 / (ni + 1.0)).abs() < 1e-12, "rhs must be 1/(n+1)");
            }
        }
        other => panic!("expected Nonzero, got {other:?}"),
    }
}

/// Bessel's three-term recurrence, read off the Schläfli contour
/// representation `J_n(2) = (1/2πi)∮ e^(x − 1/x)·x^(−n−1) dx`:
/// `J_n + J_(n+2) = (n+1)·J_(n+1)`, certificate `R = 1`.
#[test]
fn bessel_recurrence_end_to_end() {
    let pool = ExprPool::new();
    let n = pool.symbol("n", Domain::Real);
    let x = pool.symbol("x", Domain::Real);
    let arg = pool.add(vec![
        x,
        pool.mul(vec![
            pool.integer(-1_i32),
            pool.pow(x, pool.integer(-1_i32)),
        ]),
    ]);
    let f = pool.mul(vec![
        pool.func("exp", vec![arg]),
        pool.pow(
            x,
            pool.add(vec![
                pool.mul(vec![pool.integer(-1_i32), n]),
                pool.integer(-1_i32),
            ]),
        ),
    ]);

    let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
    let r = &out.value;
    assert_eq!(r.order, 2);
    for ni in [1.0_f64, 5.0, 9.5] {
        let q0 = ratio_at(&pool, n, r.coeffs[0], r.coeffs[2], ni);
        let q1 = ratio_at(&pool, n, r.coeffs[1], r.coeffs[2], ni);
        assert!((q0 - 1.0).abs() < 1e-9, "a_0/a_2 must be 1; got {q0}");
        assert!(
            (q1 + (ni + 1.0)).abs() < 1e-9,
            "a_1/a_2 must be -(n+1); got {q1}"
        );
    }
}

/// The differential-Gosper half, through the same surface:
/// `∫ x²·eˣ dx = (x²−2x+2)·eˣ`.
#[test]
fn dgosper_end_to_end() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let f = pool.mul(vec![
        pool.pow(x, pool.integer(2_i32)),
        pool.func("exp", vec![x]),
    ]);
    let r = dgosper(f, x, &pool, &AzOpts::default()).expect("certificate");
    // R = (x²−2x+2)/x²: check numerically at a couple of points, independently
    // of the exact arithmetic the engine already verified it with.
    for xi in [0.5_f64, 2.0, 7.0] {
        let env = HashMap::from([(x, xi)]);
        let got = alkahest_cas::eval_f64(r.value, &pool, &env).expect("R evaluates");
        let want = (xi * xi - 2.0 * xi + 2.0) / (xi * xi);
        assert!((got - want).abs() < 1e-12, "R at x={xi}: {got} vs {want}");
    }
}

/// Out of class in the `n` direction: `exp(n·x)/x` is hyperexponential in `x`
/// but its `n`-ratio is `eˣ`. The refusal must be typed and specific, not a
/// panic and not a wrong recurrence.
#[test]
fn out_of_class_refusal_end_to_end() {
    let pool = ExprPool::new();
    let n = pool.symbol("n", Domain::Real);
    let x = pool.symbol("x", Domain::Real);
    let f = pool.mul(vec![
        pool.func("exp", vec![pool.mul(vec![n, x])]),
        pool.pow(x, pool.integer(-1_i32)),
    ]);
    let err = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect_err("out of class");
    assert_eq!(
        alkahest_cas::errors::AlkahestError::code(&err),
        "E-HOLO-061",
        "the n-side refusal has its own stable code"
    );
}
