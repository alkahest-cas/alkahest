//! Unit tests for the special-function asymptotics (F25). The Python suite
//! (`tests/test_asymptotics_special.py`) checks the same expansions against
//! SymPy's `series(…, x, oo, n)` and mpmath.

use super::super::asymptotic_expand;
use super::*;
use crate::kernel::Domain;

fn q(n: i64, d: i64) -> Rational {
    Rational::from((n, d))
}

/// Value of `e` at `x = xv` (f64).
fn at(e: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
    let mut env = HashMap::new();
    env.insert(x, xv);
    eval_interp(e, &env, pool).unwrap()
}

#[test]
fn stirling_gamma_coefficients_are_exact() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Positive);
    let s = gamma_product(&[(q(1, 1), q(0, 1), q(1, 1))], x, 6, &p).unwrap();
    let got: Vec<Rational> = s
        .terms
        .iter()
        .map(|t| rational_literal(&p, t.coeff).unwrap())
        .collect();
    // DLMF 5.11.3: 1, 1/12, 1/288, −139/51840, −571/2488320.
    assert_eq!(
        &got[..5],
        &[
            q(1, 1),
            q(1, 12),
            q(1, 288),
            q(-139, 51840),
            q(-571, 2_488_320)
        ]
    );
}

#[test]
fn hankel_and_airy_coefficients() {
    let a = hankel_a(&q(0, 1), 3);
    assert_eq!(a, vec![q(1, 1), q(-1, 8), q(9, 128), q(-225, 3072)]);
    let a = hankel_a(&q(1, 2), 3);
    assert_eq!(a[1], q(0, 1));
    let u = airy_u(3);
    assert_eq!(u, vec![q(1, 1), q(5, 72), q(385, 10368)]);
}

#[test]
fn loggamma_stirling_terms() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Positive);
    let f = p.func("loggamma", vec![x]);
    let terms = asymptotic_expand(f, x, 6, &p).unwrap().term_exprs();
    assert_eq!(terms.len(), 6);
    let xv = 10.0f64;
    let want = [
        xv * xv.ln(),
        -xv,
        -0.5 * xv.ln(),
        0.5 * (2.0 * std::f64::consts::PI).ln(),
        1.0 / (12.0 * xv),
        -1.0 / (360.0 * xv.powi(3)),
    ];
    for (k, (&t, w)) in terms.iter().zip(want).enumerate() {
        let v = at(t, x, xv, &p);
        assert!(
            (v - w).abs() <= 1e-12 * w.abs().max(1.0),
            "term {k}: {v} vs {w}"
        );
    }
}

#[test]
fn report_repros_now_expand() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Positive);
    for f in [
        p.func("gamma", vec![x]),
        p.func("bessel_j0", vec![x]),
        p.func("erfc", vec![x]),
        p.func("Ei", vec![x]),
        p.func("airy_ai", vec![x]),
        p.func("digamma", vec![x]),
        p.func("bessel_k", vec![p.integer(1), x]),
    ] {
        let e = asymptotic_expand(f, x, 4, &p)
            .unwrap_or_else(|err| panic!("{}: {err:?}", p.display(f)));
        assert_eq!(e.terms.len(), 4, "{}", p.display(f));
    }
}

#[test]
fn j0_leading_term_and_residual() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Positive);
    let f = p.func("bessel_j0", vec![x]);
    let terms = asymptotic_expand(f, x, 3, &p).unwrap().term_exprs();
    for xv in [50.0f64, 123.4, 1000.0] {
        let lead =
            (2.0 / (std::f64::consts::PI * xv)).sqrt() * (xv - std::f64::consts::FRAC_PI_4).cos();
        assert!((at(terms[0], x, xv, &p) - lead).abs() < 1e-12);
        let sum: f64 = terms.iter().map(|&t| at(t, x, xv, &p)).sum();
        // The residual after three terms is O(x^{-7/2}).
        assert!((at(f, x, xv, &p) - sum).abs() < 0.2 * xv.powf(-3.5), "{xv}");
    }
}

#[test]
fn upper_incomplete_gamma() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Positive);
    // Γ(1/2, x) = √π erfc(√x) ~ e^{−x} x^{−1/2} (1 − 1/(2x) + 3/(4x²) …)
    let f = p.func("uppergamma", vec![p.rational(1, 2), x]);
    let terms = asymptotic_expand(f, x, 3, &p).unwrap().term_exprs();
    assert_eq!(terms.len(), 3);
    for (k, c) in [1.0, -0.5, 0.75].into_iter().enumerate() {
        let xv = 20.0f64;
        let want = c * (-xv).exp() * xv.powf(-0.5 - k as f64);
        let got = at(terms[k], x, xv, &p);
        assert!(
            (got - want).abs() <= 1e-12 * want.abs(),
            "{k}: {got} vs {want}"
        );
    }
    // Γ(3, x) = e^{−x}(x² + 2x + 2): finite and exact.
    let f = p.func("uppergamma", vec![p.integer(3), x]);
    let terms = asymptotic_expand(f, x, 6, &p).unwrap().term_exprs();
    assert_eq!(terms.len(), 3);
}

#[test]
fn central_binomial() {
    let p = ExprPool::new();
    let n = p.symbol("n", Domain::Positive);
    let two_n = p.mul(vec![p.integer(2), n]);
    let f = p.func("binomial", vec![two_n, n]);
    let terms = asymptotic_expand(f, n, 3, &p).unwrap().term_exprs();
    assert_eq!(terms.len(), 3);
    // 4^n/√(πn) (1 − 1/(8n) + 1/(128n²)) at n = 20 vs C(40, 20).
    let sum: f64 = terms.iter().map(|&t| at(t, n, 20.0, &p)).sum();
    let exact = 137_846_528_820.0_f64;
    assert!((sum / exact - 1.0).abs() < 1e-5, "{sum}");
}

#[test]
fn declines_outside_the_rules() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Positive);
    let nu = p.symbol("nu", Domain::Real);
    let neg = p.mul(vec![p.integer(-1), x]);
    for f in [
        p.func("gamma", vec![neg]),
        p.func("gamma", vec![p.pow(x, p.integer(2))]),
        p.func("bessel_j", vec![nu, x]),
        p.pow(p.func("erfc", vec![x]), p.integer(-1)),
        // Different exponential scales in one sum.
        p.add(vec![p.func("gamma", vec![x]), p.func("erfc", vec![x])]),
    ] {
        assert!(asymptotic_expand(f, x, 3, &p).is_err(), "{}", p.display(f));
    }
}
