//! F24 tests: `zeta_prime`, the Hankel functions, the real spherical
//! harmonics and the LaTeX notation of the special-function families.

use std::collections::HashMap;

use super::numeric::{eval_f64, Kernel};
use super::*;
use crate::eval::{eval_complex_f64, ComplexF64};
use crate::kernel::{render_latex, Domain, ExprPool};
use crate::orthopoly::{real_spherical_harmonic_y, spherical_harmonic_y};
use rug::Rational;

fn sym(pool: &ExprPool, s: &str) -> ExprId {
    pool.symbol(s, Domain::Real)
}

fn cplx(e: ExprId, pool: &ExprPool, binds: &[(ExprId, f64)]) -> ComplexF64 {
    let mut env: HashMap<ExprId, ComplexF64> = binds
        .iter()
        .map(|&(k, v)| (k, ComplexF64::new(v, 0.0)))
        .collect();
    env.insert(
        crate::eval::symbols::pi_symbol(pool),
        ComplexF64::new(std::f64::consts::PI, 0.0),
    );
    eval_complex_f64(e, pool, &env).expect("evaluates")
}

/// Agreement to `tol` relative to the modulus (the small component of a
/// value like `1e5 − 5e-7·i` is only meaningful next to the large one).
fn close(a: ComplexF64, re: f64, im: f64, tol: f64) -> bool {
    let scale = 1.0 + re.hypot(im);
    (a.re - re).abs() <= tol * scale && (a.im - im).abs() <= tol * scale
}

#[cfg(flint_arb)]
#[test]
fn hankel_half_integer_closed_forms_match_the_kernel() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    for kind in [HankelKind::H1, HankelKind::H2] {
        for twice in [1_i64, 3, 5, 9, -1, -3, -7, -11] {
            let nu = pool.rational(twice, 2);
            let closed = hankel(kind, nu, x, true, &pool).unwrap();
            // The expansion is elementary: no Hankel node survives.
            assert!(!pool.display(closed).to_string().contains("hankel"));
            for xv in [0.4_f64, 1.7, 6.3] {
                let want = complex_value(kind.name(), &[(twice as f64 / 2.0, 0.0), (xv, 0.0)])
                    .unwrap_or_else(|| panic!("kernel {kind:?} {twice}/2 at {xv}"));
                let got = cplx(closed, &pool, &[(x, xv)]);
                assert!(
                    close(got, want.0, want.1, 1e-11),
                    "{kind:?} nu={twice}/2 x={xv}: closed {got:?} vs kernel {want:?}"
                );
            }
        }
    }
}

#[cfg(flint_arb)]
#[test]
fn hankel_is_j_plus_i_y() {
    // H1 = J + iY, H2 = J − iY for integer and fractional orders.
    for nu in [0.0, 1.0, 2.0, 0.3, 2.75] {
        for xv in [0.5, 2.0, 9.0] {
            let j = eval_f64(Kernel::BesselJ, &[nu, xv]).unwrap();
            let y = eval_f64(Kernel::BesselY, &[nu, xv]).unwrap();
            let h1 = complex_value("hankel1", &[(nu, 0.0), (xv, 0.0)]).unwrap();
            let h2 = complex_value("hankel2", &[(nu, 0.0), (xv, 0.0)]).unwrap();
            assert!(
                (h1.0 - j).abs() < 1e-12 * (1.0 + j.abs())
                    && (h1.1 - y).abs() < 1e-12 * (1.0 + y.abs())
            );
            assert!(
                (h2.0 - j).abs() < 1e-12 * (1.0 + j.abs())
                    && (h2.1 + y).abs() < 1e-12 * (1.0 + y.abs())
            );
        }
    }
}

#[test]
fn hankel_integer_order_reflection_and_nodes() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let h = |k: HankelKind, nu: i64| hankel(k, pool.integer(nu), x, true, &pool).unwrap();
    // H_{-3} = −H_3 (odd order flips the sign), H_{-2} = H_2.
    let h3 = h(HankelKind::H1, 3);
    assert_eq!(h(HankelKind::H1, -3), pool.mul(vec![pool.integer(-1), h3]));
    assert_eq!(h(HankelKind::H2, -2), h(HankelKind::H2, 2));
    // Integer orders stay nodes; the name round-trips through the arity table.
    assert_eq!(
        h(HankelKind::H1, 0),
        pool.func("hankel1", vec![pool.integer(0), x])
    );
    assert_eq!(crate::kernel::known_func_arity("hankel1"), Some((2, 2)));
    assert_eq!(crate::kernel::known_func_arity("zeta_prime"), Some((1, 1)));
}

#[cfg(flint_arb)]
#[test]
fn hankel_differentiates_in_the_argument_only() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let nu = sym(&pool, "nu");
    // d/dx H1_2(x) = (H1_1 − H1_3)/2.
    let e = hankel(HankelKind::H1, pool.integer(2), x, false, &pool).unwrap();
    let d = crate::diff::diff(e, x, &pool).unwrap().value;
    for xv in [0.8_f64, 3.1] {
        let got = cplx(d, &pool, &[(x, xv)]);
        let h = |n: f64| complex_value("hankel1", &[(n, 0.0), (xv, 0.0)]).unwrap();
        let (a, b) = (h(1.0), h(3.0));
        assert!(close(got, 0.5 * (a.0 - b.0), 0.5 * (a.1 - b.1), 1e-12));
    }
    // ∂/∂ν has no closed form: refused, never dropped.
    let sym_order = hankel(HankelKind::H1, nu, x, false, &pool).unwrap();
    assert!(crate::diff::diff(sym_order, nu, &pool).is_err());
}

#[test]
fn zeta_prime_is_the_derivative_of_zeta() {
    let pool = ExprPool::new();
    let s = sym(&pool, "s");
    let z = zeta(s, &pool).unwrap();
    let d = crate::diff::diff(z, s, &pool).unwrap().value;
    assert_eq!(d, pool.func("zeta_prime", vec![s]));
    // Chain rule.
    let two_s = pool.mul(vec![pool.integer(2), s]);
    let z2 = zeta(two_s, &pool).unwrap();
    let d2 = crate::diff::diff(z2, s, &pool).unwrap().value;
    assert_eq!(
        d2,
        pool.mul(vec![pool.integer(2), pool.func("zeta_prime", vec![two_s])])
    );
    // ζ″ has no node: refused, not dropped.
    let zp = zeta_prime(s, &pool).unwrap();
    assert!(crate::diff::diff(zp, s, &pool).is_err());
    // The pole.
    assert!(zeta_prime(pool.integer(1), &pool).is_err());
}

#[cfg(flint_arb)]
#[test]
fn zeta_prime_exact_values_agree_with_the_kernel() {
    let pool = ExprPool::new();
    for n in [0_i64, -1, 2, -2, -4, -6, -10] {
        let e = zeta_prime(pool.integer(n), &pool).unwrap();
        // An exact value has no `zeta_prime` node left.
        assert!(
            !pool.display(e).to_string().contains("zeta_prime"),
            "ζ′({n}) was not rewritten: {}",
            pool.display(e)
        );
        let want = eval_f64(Kernel::ZetaPrime, &[n as f64]).unwrap();
        let env = HashMap::from([(crate::eval::symbols::pi_symbol(&pool), std::f64::consts::PI)]);
        let got = crate::jit::eval_interp(e, &env, &pool)
            .unwrap_or_else(|| panic!("{n}: does not evaluate"));
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1e-3),
            "ζ′({n}): exact form {got} vs kernel {want}"
        );
    }
    // ζ′(3) has no known closed form: the node stays.
    let e = zeta_prime(pool.integer(3), &pool).unwrap();
    assert_eq!(e, pool.func("zeta_prime", vec![pool.integer(3)]));
}

#[cfg(flint_arb)]
#[test]
fn real_harmonics_are_the_cosine_and_sine_parts_of_the_complex_ones() {
    let pool = ExprPool::new();
    let (t, p) = (sym(&pool, "t"), sym(&pool, "p"));
    let q = |v: i64| Rational::from(v);
    for l in 0..=4_i64 {
        for m in -l..=l {
            let real = real_spherical_harmonic_y(&q(l), &q(m), t, p, &pool).unwrap();
            let am = m.abs();
            let cplxy = spherical_harmonic_y(&q(l), &q(am), t, p, &pool).unwrap();
            for (tv, pv) in [(0.7_f64, 0.3_f64), (2.1, -1.9), (1.4, 4.4)] {
                let c = cplx(cplxy, &pool, &[(t, tv), (p, pv)]);
                let r = cplx(real, &pool, &[(t, tv), (p, pv)]);
                let sign = if am % 2 == 0 { 1.0 } else { -1.0 };
                // m > 0: √2 (−1)^m Re Y_l^m ; m < 0: √2 (−1)^|m| Im Y_l^|m| ;
                // m = 0: Y_l^0 itself.
                let want = if m == 0 {
                    c.re
                } else if m > 0 {
                    std::f64::consts::SQRT_2 * sign * c.re
                } else {
                    std::f64::consts::SQRT_2 * sign * c.im
                };
                assert!(
                    (r.re - want).abs() < 1e-12 && r.im.abs() < 1e-12,
                    "Y({l},{m}) at ({tv},{pv}): {r:?} vs {want}"
                );
            }
        }
    }
    // |m| > l is zero; non-integer arguments are refused.
    assert_eq!(
        real_spherical_harmonic_y(&q(1), &q(2), t, p, &pool).unwrap(),
        pool.integer(0)
    );
    assert!(real_spherical_harmonic_y(&Rational::from((1, 2)), &q(0), t, p, &pool).is_err());
}

#[test]
fn latex_notation() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let i = |v: i64| pool.integer(v);
    let n = sym(&pool, "n");
    let a = sym(&pool, "a");
    let b = sym(&pool, "b");
    let cases: Vec<(ExprId, &str)> = vec![
        (
            pool.func("jacobi_p", vec![n, a, b, x]),
            r"P_{n}^{(a,b)}\!\left(x\right)",
        ),
        (
            pool.func("legendre_p", vec![n, x]),
            r"P_{n}\!\left(x\right)",
        ),
        (
            pool.func("assoc_legendre_p", vec![n, i(2), x]),
            r"P_{n}^{2}\!\left(x\right)",
        ),
        (pool.func("hermite_h", vec![n, x]), r"H_{n}\!\left(x\right)"),
        (
            pool.func("hermite_he", vec![n, x]),
            r"\mathit{He}_{n}\!\left(x\right)",
        ),
        (
            pool.func("laguerre_l", vec![n, x]),
            r"L_{n}\!\left(x\right)",
        ),
        (
            pool.func("assoc_laguerre_l", vec![n, a, x]),
            r"L_{n}^{(a)}\!\left(x\right)",
        ),
        (
            pool.func("chebyshev_t", vec![n, x]),
            r"T_{n}\!\left(x\right)",
        ),
        (
            pool.func("chebyshev_u", vec![n, x]),
            r"U_{n}\!\left(x\right)",
        ),
        (
            pool.func("gegenbauer_c", vec![n, a, x]),
            r"C_{n}^{(a)}\!\left(x\right)",
        ),
        (pool.func("bessel_j", vec![n, x]), r"J_{n}\!\left(x\right)"),
        (
            pool.func("bessel_y", vec![i(2), x]),
            r"Y_{2}\!\left(x\right)",
        ),
        (pool.func("bessel_i", vec![n, x]), r"I_{n}\!\left(x\right)"),
        (pool.func("bessel_k", vec![n, x]), r"K_{n}\!\left(x\right)"),
        (
            pool.func("spherical_jn", vec![n, x]),
            r"j_{n}\!\left(x\right)",
        ),
        (
            pool.func("hankel1", vec![n, x]),
            r"H^{(1)}_{n}\!\left(x\right)",
        ),
        (
            pool.func("hankel2", vec![n, x]),
            r"H^{(2)}_{n}\!\left(x\right)",
        ),
        (
            pool.func("airy_ai", vec![x]),
            r"\operatorname{Ai}\!\left(x\right)",
        ),
        (
            pool.func("airy_bi_prime", vec![x]),
            r"\operatorname{Bi}'\!\left(x\right)",
        ),
        (pool.func("zeta", vec![x]), r"\zeta\!\left(x\right)"),
        (pool.func("zeta_prime", vec![x]), r"\zeta'\!\left(x\right)"),
        (
            pool.func("hurwitz_zeta", vec![n, x]),
            r"\zeta\!\left(n, x\right)",
        ),
        (
            pool.func("polylog", vec![n, x]),
            r"\operatorname{Li}_{n}\!\left(x\right)",
        ),
        (
            pool.func("polygamma", vec![n, x]),
            r"\psi^{(n)}\!\left(x\right)",
        ),
    ];
    for (e, want) in cases {
        assert_eq!(render_latex(e, &pool), want);
    }
}
