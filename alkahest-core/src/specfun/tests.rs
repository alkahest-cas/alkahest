//! Tests for `crate::specfun` against the registry, `diff`, `simplify` and
//! reference values (mpmath at 60 digits, recorded below).

use super::numeric::{eval_ball, eval_f64, Kernel};
use super::*;
use crate::ball::ArbBall;
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprPool};
use crate::primitive::{Capabilities, PrimitiveRegistry};

fn sym(pool: &ExprPool, s: &str) -> ExprId {
    pool.symbol(s, Domain::Real)
}

fn eval_at(e: ExprId, x: ExprId, v: f64, pool: &ExprPool) -> f64 {
    let mut env = std::collections::HashMap::new();
    env.insert(x, v);
    env.insert(crate::eval::symbols::pi_symbol(pool), std::f64::consts::PI);
    crate::jit::eval_interp(e, &env, pool).expect("evaluates")
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// `(kernel, args, value)` — mpmath at 60 digits (`mp.besselj`, `mp.bessely`,
/// `mp.besseli`, `mp.besselk`, `mp.airyai/airybi` (with derivative 1),
/// `mp.zeta`, `mp.zeta(s, a)`, `mp.polylog`; spherical via `√(π/2x)·J_{n+½}`).
const REFERENCE: &[(Kernel, &[f64], &str)] = &[
    (
        Kernel::BesselJ,
        &[2.5, 1.75],
        "0.1721875621632292476866262489584838413660266846644849792",
    ),
    (
        Kernel::BesselJ,
        &[0.0, 2.404825557695773],
        "-6.108765259736730397081979074235388478630943890786180355e-17",
    ),
    (
        Kernel::BesselY,
        &[1.0, 3.25],
        "0.3797777737170835232247409812289534689349873389075971744",
    ),
    (
        Kernel::BesselI,
        &[0.5, 0.75],
        "0.757614986389914007324539011421669966100621965231539089",
    ),
    (
        Kernel::BesselK,
        &[1.5, 2.0],
        "0.1799066579520921710520547524551902743291756778653436361",
    ),
    (
        Kernel::BesselJ,
        &[3.0, -1.25],
        "-0.03686835964979467648671441625529456824407583367716643519",
    ),
    (
        Kernel::AiryAi,
        &[1.5],
        "0.07174949700810540967355541648967751360575177846439507536",
    ),
    (
        Kernel::AiryBi,
        &[-2.25],
        "-0.4539206867501173070455352154417670062727199719898462637",
    ),
    (
        Kernel::AiryAiPrime,
        &[0.5],
        "-0.2249105326646838931359969903285832148250296356108928371",
    ),
    (
        Kernel::AiryBiPrime,
        &[1.0],
        "0.9324359333927756329594514536744353442695653752386283955",
    ),
    (
        Kernel::Zeta,
        &[3.0],
        "1.202056903159594285399738161511449990764986292340498882",
    ),
    (
        Kernel::Zeta,
        &[0.5],
        "-1.460354508809586812889499152515298012467229331012581491",
    ),
    (
        Kernel::Zeta,
        &[-2.5],
        "0.008516928777850330542358567028344486936275990220074477766",
    ),
    (
        Kernel::HurwitzZeta,
        &[2.5, 0.75],
        "2.491542385511935224454122679634134474134214799102436647",
    ),
    (
        Kernel::Polylog,
        &[3.0, 0.5],
        "0.5372131936080402009406232255949658266704024993403781707",
    ),
    (
        Kernel::Polylog,
        &[2.5, -3.0],
        "-2.162700712002056662292591440719206032550808511302658459",
    ),
    (
        Kernel::ZetaPrime,
        &[2.5],
        "-0.3873419503262099727119923759310510131994822887468830534",
    ),
    (
        Kernel::ZetaPrime,
        &[0.5],
        "-3.92264613920915172747153144671459951373032397150650521",
    ),
    (
        Kernel::ZetaPrime,
        &[-2.5],
        "-0.006265736372189758403282758729725833176322370751750072669",
    ),
    (
        Kernel::LogGamma,
        &[7.25],
        "7.052185450738539444925749253133010245418207107270901606",
    ),
    (
        Kernel::LogGamma,
        &[0.375],
        "0.863073982270647462405089094134015495332470629348427135",
    ),
    (
        Kernel::LogGamma,
        &[150.5],
        "602.5139548705854119507378778307831007899261142754249016",
    ),
    (
        Kernel::SphericalJ,
        &[2.0, 1.5],
        "0.1273492836884082156472409472301944595991345023426743568",
    ),
    (
        Kernel::SphericalY,
        &[1.0, -2.0],
        "-0.3506120042760552509486178755806818739096272929550589115",
    ),
];

#[test]
fn every_kernel_is_registered_with_the_right_arity() {
    let reg = PrimitiveRegistry::default_registry();
    for k in Kernel::ALL {
        assert!(reg.is_registered(k.name()), "{}", k.name());
        assert_eq!(
            crate::kernel::known_func_arity(k.name()),
            Some((k.arity(), k.arity())),
            "{}",
            k.name()
        );
        let caps = reg.capabilities(k.name());
        // ζ″ has no node: zeta_prime must not claim a derivative.
        assert_eq!(
            caps.contains(Capabilities::DIFF_FORWARD),
            k != Kernel::ZetaPrime,
            "{}",
            k.name()
        );
        assert!(!caps.contains(Capabilities::DIFF_REVERSE), "{}", k.name());
        assert!(!caps.contains(Capabilities::LEAN_THEOREM), "{}", k.name());
        // F24: Taylor-model rules exist for the cylinder functions of integer
        // order, Airy, zeta and the polylogarithm (`validated::taylor`).
        let has_taylor = matches!(
            k,
            Kernel::BesselJ
                | Kernel::BesselY
                | Kernel::BesselI
                | Kernel::BesselK
                | Kernel::AiryAi
                | Kernel::AiryBi
                | Kernel::AiryAiPrime
                | Kernel::AiryBiPrime
                | Kernel::Zeta
                | Kernel::Polylog
                | Kernel::LogGamma
        );
        assert_eq!(
            caps.contains(Capabilities::TAYLOR_MODEL),
            has_taylor,
            "{}",
            k.name()
        );
        assert!(!caps.contains(Capabilities::LOWER_LLVM), "{}", k.name());
        #[cfg(flint_arb)]
        {
            // The Hankel pair is complex-valued: no real f64 / ball slot.
            let real_valued = !matches!(k, Kernel::HankelH1 | Kernel::HankelH2);
            assert_eq!(
                caps.contains(Capabilities::NUMERIC_F64),
                real_valued,
                "{}",
                k.name()
            );
            assert_eq!(
                caps.contains(Capabilities::NUMERIC_BALL),
                real_valued,
                "{}",
                k.name()
            );
        }
    }
}

#[cfg(flint_arb)]
#[test]
fn balls_at_200_bits_contain_the_reference_values() {
    use rug::Float;
    for &(k, args, want) in REFERENCE {
        let balls: Vec<ArbBall> = args.iter().map(|&v| ArbBall::from_f64(v, 200)).collect();
        let b = eval_ball(k, &balls).unwrap_or_else(|| panic!("{k:?} {args:?}"));
        let r = Float::with_val(300, Float::parse(want).unwrap());
        let diff = Float::with_val(300, &b.mid - &r).abs();
        // The reference carries 55 significant digits.
        let slack = Float::with_val(300, r.clone().abs() * 1e-54) + 1e-70;
        let bound = Float::with_val(300, &b.rad + &slack);
        assert!(diff <= bound, "{k:?} {args:?}: {} vs {want}", b.mid);
        assert!(b.rad < 1e-55, "{k:?} {args:?}: radius {}", b.rad);
        let f = eval_f64(k, args).unwrap_or_else(|| panic!("f64 {k:?} {args:?}"));
        let rf = r.to_f64();
        assert!(
            (f - rf).abs() <= 4.0 * f64::EPSILON * rf.abs(),
            "{k:?} {args:?}: {f} vs {rf}"
        );
    }
}

#[cfg(flint_arb)]
#[test]
fn values_off_the_real_domain_and_poles_are_declined() {
    // Y, K at x < 0; J_ν at x < 0 for non-integer ν; Li_s past the cut; ζ(1).
    assert_eq!(eval_f64(Kernel::BesselY, &[0.0, -1.0]), None);
    assert_eq!(eval_f64(Kernel::BesselK, &[0.0, -1.0]), None);
    assert_eq!(eval_f64(Kernel::BesselJ, &[0.5, -1.0]), None);
    assert_eq!(eval_f64(Kernel::Polylog, &[3.0, 2.0]), None);
    assert_eq!(eval_f64(Kernel::Zeta, &[1.0]), None);
    assert_eq!(eval_f64(Kernel::HurwitzZeta, &[1.0, 0.5]), None);
    // j_0(0) = 1, y_n(0) is singular.
    assert_eq!(eval_f64(Kernel::SphericalJ, &[0.0, 0.0]), Some(1.0));
    assert_eq!(eval_f64(Kernel::SphericalY, &[0.0, 0.0]), None);
    // Complex values exist where the real kernels decline.
    let (re, im) = complex_value("polylog", &[(3.0, 0.0), (2.0, 0.0)]).unwrap();
    // Li_3(2) = π²·log 2/4 − iπ·log²2/2 + 7ζ(3)/8 … just check it is non-real.
    assert!(re.is_finite() && im.abs() > 0.1);
}

#[test]
fn half_integer_bessel_expansions_match_the_kernels() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let reg = PrimitiveRegistry::dispatch_registry();
    for kind in [BesselKind::J, BesselKind::Y, BesselKind::I, BesselKind::K] {
        for num in [-7_i64, -5, -3, -1, 1, 3, 5, 9] {
            let e = bessel(kind, pool.rational(num, 2), x, true, &pool).unwrap();
            assert!(
                crate::kernel::subs::mentions_var(e, x, &pool),
                "{kind:?} {num}/2"
            );
            for xv in [0.3_f64, 1.7, 4.2] {
                let got = eval_at(e, x, xv, &pool);
                let Some(want) = reg.numeric_f64(kind.name(), &[num as f64 / 2.0, xv]) else {
                    continue; // no Arb in this build
                };
                assert!(
                    close(got, want, 1e-11),
                    "{kind:?} {num}/2 at {xv}: {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn spherical_expansions_match_the_kernels() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let reg = PrimitiveRegistry::dispatch_registry();
    for kind in [SphericalBesselKind::J, SphericalBesselKind::Y] {
        for n in -3_i64..=6 {
            let e = spherical_bessel(kind, pool.integer(n), x, true, &pool).unwrap();
            for xv in [0.6_f64, 2.5, -1.3] {
                let got = eval_at(e, x, xv, &pool);
                let Some(want) = reg.numeric_f64(kind.name(), &[n as f64, xv]) else {
                    continue;
                };
                assert!(
                    close(got, want, 1e-10),
                    "{kind:?} {n} at {xv}: {got} vs {want}"
                );
            }
        }
    }
    let err =
        spherical_bessel(SphericalBesselKind::J, pool.rational(1, 2), x, true, &pool).unwrap_err();
    assert_eq!(err.code(), "E-SPECFN-007");
    let err = bessel(BesselKind::J, pool.rational(1001, 2), x, true, &pool).unwrap_err();
    assert_eq!(err.code(), "E-SPECFN-004");
}

#[test]
fn bessel_canonical_forms() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let n = sym(&pool, "n");
    let b = |k, nu, x| bessel(k, nu, x, true, &pool).unwrap();
    assert_eq!(
        b(BesselKind::J, pool.integer(0), x),
        pool.func("bessel_j0", vec![x])
    );
    assert_eq!(
        b(BesselKind::J, pool.integer(1), x),
        pool.func("bessel_j1", vec![x])
    );
    // J_{−1} = −J_1
    assert_eq!(
        b(BesselKind::J, pool.integer(-1), x),
        pool.mul(vec![pool.integer(-1), pool.func("bessel_j1", vec![x])])
    );
    // J_{−2} = J_2, I_{−3} = I_3, K_{−5/2}: half-integer, so expanded; K_{−n}
    assert_eq!(
        b(BesselKind::J, pool.integer(-2), x),
        pool.func("bessel_j", vec![pool.integer(2), x])
    );
    assert_eq!(
        b(BesselKind::I, pool.integer(-3), x),
        pool.func("bessel_i", vec![pool.integer(3), x])
    );
    assert_eq!(
        b(BesselKind::K, pool.integer(-4), x),
        pool.func("bessel_k", vec![pool.integer(4), x])
    );
    let zero = pool.integer(0);
    assert_eq!(b(BesselKind::J, zero, zero), pool.integer(1));
    assert_eq!(b(BesselKind::J, pool.integer(3), zero), zero);
    assert_eq!(b(BesselKind::I, zero, zero), pool.integer(1));
    assert_eq!(b(BesselKind::I, pool.rational(3, 2), zero), zero);
    // Y_0(0) is singular: the node is kept.
    assert_eq!(
        b(BesselKind::Y, zero, zero),
        pool.func("bessel_y", vec![zero, zero])
    );
    // A symbolic order is kept.
    assert_eq!(b(BesselKind::J, n, x), pool.func("bessel_j", vec![n, x]));
    // Parity: J_3(−x) = −J_3(x).
    let mx = pool.mul(vec![pool.integer(-1), x]);
    assert_eq!(
        b(BesselKind::J, pool.integer(3), mx),
        pool.mul(vec![
            pool.integer(-1),
            pool.func("bessel_j", vec![pool.integer(3), x])
        ])
    );
}

#[test]
fn zeta_exact_values_and_pole() {
    let pool = ExprPool::new();
    let pi = crate::eval::symbols::pi_symbol(&pool);
    let z = |s: i64| zeta(pool.integer(s), &pool).unwrap();
    assert_eq!(
        z(2),
        pool.mul(vec![pool.rational(1, 6), pool.pow(pi, pool.integer(2))])
    );
    assert_eq!(
        z(4),
        pool.mul(vec![pool.rational(1, 90), pool.pow(pi, pool.integer(4))])
    );
    assert_eq!(z(0), pool.rational(-1, 2));
    assert_eq!(z(-1), pool.rational(-1, 12));
    assert_eq!(z(-2), pool.integer(0));
    assert_eq!(z(-3), pool.rational(1, 120));
    assert_eq!(z(3), pool.func("zeta", vec![pool.integer(3)]));
    let err = zeta(pool.integer(1), &pool).unwrap_err();
    assert_eq!(err.code(), "E-SPECFN-006");
    // ζ(s, 1) = ζ(s); ζ(s, 1/2) = (2^s − 1) ζ(s); ζ(3, 3) = ζ(3) − 1 − 1/8.
    let s = sym(&pool, "s");
    assert_eq!(
        hurwitz_zeta(s, pool.integer(1), &pool).unwrap(),
        pool.func("zeta", vec![s])
    );
    assert_eq!(
        hurwitz_zeta(pool.integer(3), pool.rational(1, 2), &pool).unwrap(),
        pool.mul(vec![
            pool.integer(7),
            pool.func("zeta", vec![pool.integer(3)])
        ])
    );
    assert_eq!(
        hurwitz_zeta(pool.integer(3), pool.integer(3), &pool).unwrap(),
        pool.add(vec![
            pool.func("zeta", vec![pool.integer(3)]),
            pool.rational(-9, 8)
        ])
    );
    assert_eq!(
        hurwitz_zeta(pool.integer(1), s, &pool).unwrap_err().code(),
        "E-SPECFN-006"
    );
}

#[test]
fn hurwitz_at_nonpositive_integers_is_a_bernoulli_polynomial() {
    let pool = ExprPool::new();
    let a = sym(&pool, "a");
    let reg = PrimitiveRegistry::dispatch_registry();
    for n in 0..6_i64 {
        let e = hurwitz_zeta(pool.integer(-n), a, &pool).unwrap();
        for av in [0.3_f64, 1.7, 2.25] {
            let got = eval_at(e, a, av, &pool);
            let Some(want) = reg.numeric_f64("hurwitz_zeta", &[-(n as f64), av]) else {
                continue;
            };
            assert!(close(got, want, 1e-12), "ζ(−{n}, {av}): {got} vs {want}");
        }
    }
}

#[test]
fn polylog_closed_forms() {
    let pool = ExprPool::new();
    let z = sym(&pool, "z");
    let reg = PrimitiveRegistry::dispatch_registry();
    assert_eq!(
        polylog(pool.integer(2), z, &pool).unwrap(),
        pool.func("dilog", vec![z])
    );
    for n in [1_i64, 0, -1, -2, -5] {
        let e = polylog(pool.integer(n), z, &pool).unwrap();
        for zv in [-0.7_f64, 0.25, 0.6] {
            let got = eval_at(e, z, zv, &pool);
            let Some(want) = reg.numeric_f64("polylog", &[n as f64, zv]) else {
                continue;
            };
            assert!(close(got, want, 1e-12), "Li_{n}({zv}): {got} vs {want}");
        }
    }
    let pi = crate::eval::symbols::pi_symbol(&pool);
    // Li_4(1) = π⁴/90; Li_3(−1) = −3/4·ζ(3)
    assert_eq!(
        polylog(pool.integer(4), pool.integer(1), &pool).unwrap(),
        pool.mul(vec![pool.rational(1, 90), pool.pow(pi, pool.integer(4))])
    );
    assert_eq!(
        polylog(pool.integer(3), pool.integer(-1), &pool).unwrap(),
        pool.mul(vec![
            pool.rational(-3, 4),
            pool.func("zeta", vec![pool.integer(3)])
        ])
    );
    assert_eq!(
        polylog(pool.integer(1), pool.integer(1), &pool)
            .unwrap_err()
            .code(),
        "E-SPECFN-006"
    );
}

#[test]
fn airy_values_at_zero() {
    let pool = ExprPool::new();
    let zero = pool.integer(0);
    let reg = PrimitiveRegistry::dispatch_registry();
    for kind in [
        AiryKind::Ai,
        AiryKind::Bi,
        AiryKind::AiPrime,
        AiryKind::BiPrime,
    ] {
        let e = airy(kind, zero, &pool);
        assert!(!pool.display(e).to_string().contains("airy"), "{kind:?}");
        let x = sym(&pool, "unused");
        let got = eval_at(e, x, 0.0, &pool);
        if let Some(want) = reg.numeric_f64(kind.name(), &[0.0]) {
            assert!(close(got, want, 1e-14), "{kind:?}: {got} vs {want}");
        }
    }
}

#[test]
fn derivatives() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let nu = sym(&pool, "nu");
    let d = |e| crate::diff::diff(e, x, &pool);
    // d/dx J_ν = (J_{ν−1} − J_{ν+1})/2
    let jn = pool.func("bessel_j", vec![nu, x]);
    let dj = d(jn).unwrap().value;
    let s = pool.display(dj).to_string();
    assert!(
        s.contains("bessel_j((nu + -1), x)") && s.contains("bessel_j((nu + 1), x)"),
        "{s}"
    );
    // The order derivative is refused, never dropped.
    let jx = pool.func("bessel_j", vec![x, pool.integer(2)]);
    assert!(d(jx).is_err());
    // ζ′(s) is the node zeta_prime, which has no derivative of its own;
    // a constant ζ(3) differentiates to 0.
    assert_eq!(
        d(pool.func("zeta", vec![x])).unwrap().value,
        pool.func("zeta_prime", vec![x])
    );
    assert!(d(pool.func("zeta_prime", vec![x])).is_err());
    assert_eq!(
        d(pool.func("zeta", vec![pool.integer(3)])).unwrap().value,
        pool.integer(0)
    );
    // Ai″ = x·Ai
    let ai = pool.func("airy_ai", vec![x]);
    let d2 = d(d(ai).unwrap().value).unwrap().value;
    let d2 = crate::simplify::simplify(d2, &pool).value;
    assert_eq!(
        d2,
        crate::simplify::simplify(pool.mul(vec![x, ai]), &pool).value
    );
    // d/dz Li_3(z) = Li_2(z)/z = dilog(z)/z
    let li3 = pool.func("polylog", vec![pool.integer(3), x]);
    let dl = d(li3).unwrap().value;
    assert!(
        pool.display(dl).to_string().contains("dilog"),
        "{}",
        pool.display(dl)
    );
}

/// Finite differences of the kernels against the derivative identities.
#[cfg(flint_arb)]
#[test]
fn derivative_identities_vs_finite_differences() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let cases: Vec<(Kernel, Vec<ExprId>, f64)> = vec![
        (Kernel::BesselJ, vec![pool.rational(7, 3), x], 1.3),
        (Kernel::BesselY, vec![pool.integer(2), x], 2.1),
        (Kernel::BesselI, vec![pool.rational(1, 3), x], 0.8),
        (Kernel::BesselK, vec![pool.rational(5, 4), x], 1.6),
        (Kernel::SphericalJ, vec![pool.integer(3), x], 1.9),
        (Kernel::SphericalY, vec![pool.integer(2), x], 2.4),
        (Kernel::AiryAi, vec![x], 0.7),
        (Kernel::AiryBiPrime, vec![x], -0.9),
        (Kernel::HurwitzZeta, vec![pool.rational(5, 2), x], 0.6),
        (Kernel::Polylog, vec![pool.rational(5, 2), x], 0.35),
    ];
    for (k, args, xv) in cases {
        let e = pool.func(k.name(), args);
        let de = crate::diff::diff(e, x, &pool).unwrap().value;
        let got = eval_at(de, x, xv, &pool);
        let h = 1e-5;
        let fd = (eval_at(e, x, xv + h, &pool) - eval_at(e, x, xv - h, &pool)) / (2.0 * h);
        assert!(close(got, fd, 1e-8), "{k:?} at {xv}: {got} vs {fd}");
    }
}

#[test]
fn expand_special_after_subs() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let nu = sym(&pool, "nu");
    let e = pool.func("bessel_j", vec![nu, x]);
    let e = crate::kernel::subs::subs(e, &[(nu, pool.rational(1, 2))].into_iter().collect(), &pool);
    let ex = expand_special(e, &pool).unwrap();
    assert!(
        pool.display(ex).to_string().contains("sin"),
        "{}",
        pool.display(ex)
    );
}

#[test]
fn taylor_series_of_integer_order_bessel_at_zero() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let zero = pool.integer(0);
    // J_2(x) = x²/8 − x⁴/96 + x⁶/3072 − …
    let e = pool.func("bessel_j", vec![pool.integer(2), x]);
    let s = crate::calculus::series(e, x, zero, 8, &pool).expect("series of J_2");
    let shown = pool.display(s.expr()).to_string();
    assert!(
        shown.contains("1/8") && shown.contains("-1/96") && shown.contains("1/3072"),
        "{shown}"
    );
    // J_0 through the unary alias (its J_1/x derivative used to be 0/0 at 0).
    let e = pool.func("bessel_j0", vec![x]);
    let s = crate::calculus::series(e, x, zero, 6, &pool).expect("series of J_0");
    let shown = pool.display(s.expr()).to_string();
    assert!(shown.contains("-1/4") && shown.contains("1/64"), "{shown}");
}
