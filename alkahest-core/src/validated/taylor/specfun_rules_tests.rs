//! Tests for the F24 Taylor-model rules (`specfun_rules`).  The oracle is the
//! Arb value of the function at 200 sample points of every box: each must lie
//! inside the Taylor enclosure.

use super::specfun_rules::{airy_polys, stirling_first};
use super::*;
use crate::kernel::{Domain, ExprPool};
use crate::primitive::{taylor_model_supports, PrimitiveRegistry};

const P: u32 = 128;

fn fl(v: f64) -> Float {
    Float::with_val(P, v)
}

/// `name(lead…, x)` over `x ∈ [lo, hi]` at Taylor order `order`: the
/// enclosure must contain the Arb value at 200 sample points; the enclosure
/// is returned for width checks.
fn check(
    name: &str,
    lead: &[ExprId],
    pool: &ExprPool,
    lead_f64: &[f64],
    (lo, hi): (f64, f64),
    order: usize,
) -> ArbBall {
    let x = pool.symbol("x", Domain::Real);
    let mut args = lead.to_vec();
    args.push(x);
    let e = pool.func(name, args);
    let r = taylor_range(e, pool, &[(x, fl(lo), fl(hi))], order, P)
        .unwrap_or_else(|err| panic!("{name} on [{lo}, {hi}]: {err}"));
    let reg = PrimitiveRegistry::default_registry();
    for i in 0..=200 {
        let t = lo + (hi - lo) * f64::from(i) / 200.0;
        let mut a = lead_f64.to_vec();
        a.push(t);
        let want = reg
            .numeric_f64(name, &a)
            .unwrap_or_else(|| panic!("no f64 value for {name}{a:?}"));
        assert!(
            r.contains(want),
            "{name}{a:?} = {want} escaped the Taylor enclosure {r}"
        );
    }
    r
}

fn width(b: &ArbBall) -> f64 {
    2.0 * b.rad_f64()
}

#[test]
fn bessel_y_i_k_are_enclosed() {
    let pool = ExprPool::new();
    for nu in [0_i64, 1, 3] {
        let n = pool.integer(nu);
        check("bessel_y", &[n], &pool, &[nu as f64], (0.75, 1.0), 10);
        check("bessel_y", &[n], &pool, &[nu as f64], (0.5, 6.0), 12);
        check("bessel_k", &[n], &pool, &[nu as f64], (0.5, 3.0), 12);
        check("bessel_i", &[n], &pool, &[nu as f64], (-1.5, 2.0), 12);
        check("bessel_j", &[n], &pool, &[nu as f64], (0.0, 5.0), 12);
    }
    // Negative orders reflect.
    let m2 = pool.integer(-2);
    check("bessel_y", &[m2], &pool, &[-2.0], (1.0, 3.0), 10);
    check("bessel_k", &[m2], &pool, &[-2.0], (1.0, 3.0), 10);
    let m3 = pool.integer(-3);
    check("bessel_j", &[m3], &pool, &[-3.0], (0.0, 3.0), 10);
    check("bessel_i", &[m3], &pool, &[-3.0], (0.0, 3.0), 10);
}

#[test]
fn tight_on_a_small_box() {
    let pool = ExprPool::new();
    let one = pool.integer(1);
    let r = check("bessel_y", &[one], &pool, &[1.0], (1.0, 1.01), 8);
    assert!(width(&r) < 0.1, "width {}", width(&r));
    let x = pool.symbol("x", Domain::Real);
    let e = pool.func("airy_ai", vec![x]);
    let r = taylor_range(e, &pool, &[(x, fl(0.5), fl(0.51))], 8, P).unwrap();
    assert!(width(&r) < 1e-2, "width {}", width(&r));
}

#[test]
fn airy_family_is_enclosed() {
    let pool = ExprPool::new();
    for name in ["airy_ai", "airy_bi", "airy_ai_prime", "airy_bi_prime"] {
        check(name, &[], &pool, &[], (-3.0, 2.0), 12);
        check(name, &[], &pool, &[], (0.0, 1.0), 8);
    }
}

#[test]
fn zeta_polygamma_polylog_are_enclosed() {
    let pool = ExprPool::new();
    check("zeta", &[], &pool, &[], (2.0, 3.0), 10);
    check("zeta", &[], &pool, &[], (1.25, 2.0), 10);
    check("zeta", &[], &pool, &[], (0.2, 0.8), 10);
    check("zeta", &[], &pool, &[], (-3.0, -1.5), 10);
    for n in [0_i64, 1, 2, 5] {
        check(
            "polygamma",
            &[pool.integer(n)],
            &pool,
            &[n as f64],
            (0.5, 3.0),
            10,
        );
    }
    for (s, sf) in [
        (pool.integer(3), 3.0),
        (pool.rational(5, 2), 2.5),
        (pool.integer(-1), -1.0),
    ] {
        check("polylog", &[s], &pool, &[sf], (0.1, 0.6), 10);
        check("polylog", &[s], &pool, &[sf], (-0.9, -0.1), 10);
        // A box through the origin: the m0 = 0 coefficients are the series.
        check("polylog", &[s], &pool, &[sf], (-0.4, 0.4), 10);
    }
}

#[test]
fn orthogonal_polynomials_are_enclosed() {
    let pool = ExprPool::new();
    let i = |n: i64| pool.integer(n);
    check("legendre_p", &[i(5)], &pool, &[5.0], (-1.0, 1.0), 8);
    check("chebyshev_t", &[i(7)], &pool, &[7.0], (-1.0, 1.0), 8);
    check("chebyshev_u", &[i(4)], &pool, &[4.0], (-1.0, 1.0), 8);
    check("hermite_h", &[i(4)], &pool, &[4.0], (-1.5, 1.5), 8);
    check("hermite_he", &[i(5)], &pool, &[5.0], (-1.5, 1.5), 8);
    check("laguerre_l", &[i(4)], &pool, &[4.0], (0.0, 5.0), 8);
    check(
        "assoc_laguerre_l",
        &[i(3), pool.rational(1, 2)],
        &pool,
        &[3.0, 0.5],
        (0.0, 4.0),
        8,
    );
    check(
        "gegenbauer_c",
        &[i(4), pool.rational(3, 2)],
        &pool,
        &[4.0, 1.5],
        (-1.0, 1.0),
        8,
    );
    check(
        "jacobi_p",
        &[i(3), pool.rational(1, 2), pool.rational(-1, 3)],
        &pool,
        &[3.0, 0.5, -1.0 / 3.0],
        (-0.9, 0.9),
        8,
    );
    check(
        "assoc_legendre_p",
        &[i(3), i(1)],
        &pool,
        &[3.0, 1.0],
        (-0.8, 0.8),
        8,
    );
}

#[test]
fn composite_expression_and_two_variable_boxes() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    // J_1(x)·Y_1(y) + Ai(x − y) over a 2-D box.
    let j = pool.func("bessel_j", vec![pool.integer(1), x]);
    let yv = pool.func("bessel_y", vec![pool.integer(1), y]);
    let ai = pool.func(
        "airy_ai",
        vec![pool.add(vec![x, pool.mul(vec![pool.integer(-1), y])])],
    );
    let e = pool.add(vec![pool.mul(vec![j, yv]), ai]);
    let r = taylor_range(
        e,
        &pool,
        &[(x, fl(0.5), fl(1.0)), (y, fl(1.0), fl(2.0))],
        6,
        P,
    )
    .unwrap();
    let reg = PrimitiveRegistry::default_registry();
    for i in 0..=20 {
        for k in 0..=20 {
            let (xv, yv) = (0.5 + 0.5 * f64::from(i) / 20.0, 1.0 + f64::from(k) / 20.0);
            let v = reg.numeric_f64("bessel_j", &[1.0, xv]).unwrap()
                * reg.numeric_f64("bessel_y", &[1.0, yv]).unwrap()
                + reg.numeric_f64("airy_ai", &[xv - yv]).unwrap();
            assert!(r.contains(v), "{v} escaped {r}");
        }
    }
}

#[test]
fn refusals_are_coded_not_wrong() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let nu = pool.symbol("nu", Domain::Real);
    let go = |e: ExprId, lo: f64, hi: f64| {
        taylor_range(e, &pool, &[(x, fl(lo), fl(hi))], 6, P).unwrap_err()
    };
    // Y and K need x > 0.
    let one = pool.integer(1);
    assert!(matches!(
        go(pool.func("bessel_y", vec![one, x]), -1.0, 1.0),
        ValidatedError::DomainViolation { .. }
    ));
    assert!(matches!(
        go(pool.func("bessel_k", vec![one, x]), 0.0, 1.0),
        ValidatedError::DomainViolation { .. }
    ));
    // The pole of ζ and the branch point of Li_s.
    assert!(matches!(
        go(pool.func("zeta", vec![x]), 0.5, 1.5),
        ValidatedError::DomainViolation { .. }
    ));
    assert!(matches!(
        go(pool.func("polylog", vec![pool.integer(2), x]), 0.5, 1.0),
        ValidatedError::DomainViolation { .. }
    ));
    assert!(matches!(
        go(pool.func("polygamma", vec![one, x]), -0.5, 1.0),
        ValidatedError::DomainViolation { .. }
    ));
    // A symbolic or non-integer order has no rule.
    assert!(matches!(
        go(pool.func("bessel_j", vec![nu, x]), 1.0, 2.0),
        ValidatedError::InvalidInput { .. } | ValidatedError::UnboundSymbol { .. }
    ));
    assert!(matches!(
        go(pool.func("legendre_p", vec![nu, x]), 0.1, 0.2),
        ValidatedError::InvalidInput { .. } | ValidatedError::UnboundSymbol { .. }
    ));
    assert!(matches!(
        go(
            pool.func("bessel_y", vec![pool.rational(1, 2), x]),
            1.0,
            2.0
        ),
        ValidatedError::Unsupported { .. }
    ));
}

#[test]
fn capability_flags_follow_the_evaluator() {
    for name in [
        "legendre_p",
        "assoc_legendre_p",
        "hermite_h",
        "hermite_he",
        "laguerre_l",
        "assoc_laguerre_l",
        "chebyshev_t",
        "chebyshev_u",
        "gegenbauer_c",
        "jacobi_p",
        "bessel_j",
        "bessel_y",
        "bessel_i",
        "bessel_k",
        "airy_ai",
        "airy_bi",
        "airy_ai_prime",
        "airy_bi_prime",
        "zeta",
        "polylog",
        "polygamma",
    ] {
        assert!(taylor_model_supports(name), "{name}");
    }
    // No rule: honest false.
    for name in [
        "spherical_jn",
        "spherical_yn",
        "hurwitz_zeta",
        "hyper",
        "zeta_prime",
        "hankel1",
        "hankel2",
    ] {
        assert!(!taylor_model_supports(name), "{name}");
    }
}

#[test]
fn stirling_and_airy_tables() {
    // s(4, ·) = [0, -6, 11, -6, 1]
    let row: Vec<i64> = stirling_first(4)
        .iter()
        .map(|c| c.to_i64().unwrap())
        .collect();
    assert_eq!(row, vec![0, -6, 11, -6, 1]);
    // Ai'' = x·Ai ; Ai''' = Ai + x·Ai'
    let (p, q) = airy_polys(2);
    assert_eq!((p, q), (vec![Integer::new(), Integer::from(1)], vec![]));
    let (p, q) = airy_polys(3);
    assert_eq!(
        (p, q),
        (
            vec![Integer::from(1)],
            vec![Integer::new(), Integer::from(1)]
        )
    );
}
