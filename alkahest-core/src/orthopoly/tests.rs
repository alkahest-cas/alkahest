//! Integration tests for `crate::orthopoly` against the primitive registry,
//! `diff`, `simplify` and the numeric evaluators.

use super::*;
use crate::kernel::{Domain, ExprPool};
use crate::primitive::{Capabilities, PrimitiveRegistry};

fn sym(pool: &ExprPool, s: &str) -> ExprId {
    pool.symbol(s, Domain::Real)
}

fn eval_at(e: ExprId, x: ExprId, v: f64, pool: &ExprPool) -> f64 {
    let mut env = std::collections::HashMap::new();
    env.insert(x, v);
    crate::jit::eval_interp(e, &env, pool).expect("evaluates")
}

#[test]
fn every_family_is_registered_with_the_right_arity() {
    let reg = PrimitiveRegistry::default_registry();
    for f in OrthoFamily::ALL {
        assert!(reg.is_registered(f.name()), "{}", f.name());
        assert_eq!(
            crate::kernel::known_func_arity(f.name()),
            Some((f.arity(), f.arity())),
            "{}",
            f.name()
        );
        let caps = reg.capabilities(f.name());
        assert!(caps.contains(Capabilities::NUMERIC_F64), "{}", f.name());
        assert!(caps.contains(Capabilities::DIFF_FORWARD), "{}", f.name());
        // No reverse slot, no Lean theorem: none claimed. (F24: every family
        // has a Taylor-model rule for an integer literal degree.)
        assert!(!caps.contains(Capabilities::DIFF_REVERSE), "{}", f.name());
        assert!(!caps.contains(Capabilities::LEAN_THEOREM), "{}", f.name());
        assert!(caps.contains(Capabilities::TAYLOR_MODEL), "{}", f.name());
        assert!(!caps.contains(Capabilities::LOWER_LLVM), "{}", f.name());
        #[cfg(flint_arb)]
        assert!(caps.contains(Capabilities::NUMERIC_BALL), "{}", f.name());
    }
}

/// The explicit expansion and the `f64` kernel of the unexpanded node agree.
#[test]
fn explicit_expansion_matches_numeric_kernel() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let reg = PrimitiveRegistry::dispatch_registry();
    let params: [&[(i64, i64)]; 10] = [
        &[],
        &[(2, 1)],
        &[],
        &[],
        &[],
        &[(3, 2)],
        &[],
        &[],
        &[(5, 4)],
        &[(1, 2), (-1, 3)],
    ];
    for (f, ps) in OrthoFamily::ALL.into_iter().zip(params) {
        for n in 0..9 {
            let pids: Vec<ExprId> = ps.iter().map(|&(a, b)| pool.rational(a, b)).collect();
            let e = orthopoly(f, pool.integer(n), &pids, x, true, &pool).unwrap();
            for &xv in &[-0.8, -0.3, 0.1, 0.55, 0.9] {
                let mut args = vec![n as f64];
                args.extend(ps.iter().map(|&(a, b)| a as f64 / b as f64));
                args.push(xv);
                let want = reg.numeric_f64(f.name(), &args).unwrap();
                let got = eval_at(e, x, xv, &pool);
                assert!(
                    (got - want).abs() <= 1e-11 * (1.0 + want.abs()),
                    "{}({n}, {ps:?}, {xv}): explicit {got} vs kernel {want}",
                    f.name()
                );
            }
        }
    }
}

/// `d/dx` of the node agrees with a central finite difference of the kernel,
/// for integer and non-integer degree.
#[test]
fn node_derivative_matches_finite_difference() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let n = sym(&pool, "n");
    let cases: [(OrthoFamily, Vec<f64>); 10] = [
        (OrthoFamily::Legendre, vec![]),
        (OrthoFamily::AssocLegendre, vec![1.0]),
        (OrthoFamily::HermiteH, vec![]),
        (OrthoFamily::HermiteHe, vec![]),
        (OrthoFamily::Laguerre, vec![]),
        (OrthoFamily::AssocLaguerre, vec![0.5]),
        (OrthoFamily::ChebyshevT, vec![]),
        (OrthoFamily::ChebyshevU, vec![]),
        (OrthoFamily::Gegenbauer, vec![0.75]),
        (OrthoFamily::Jacobi, vec![0.5, 1.5]),
    ];
    let reg = PrimitiveRegistry::dispatch_registry();
    for (f, ps) in cases {
        let pids: Vec<ExprId> = ps.iter().map(|&p| pool.float(p, 53)).collect();
        let node = orthopoly(f, n, &pids, x, false, &pool).unwrap();
        let d = crate::diff::diff(node, x, &pool).unwrap().value;
        let degrees: &[f64] = if cfg!(flint_arb) {
            &[3.0, 6.0, 2.5]
        } else {
            &[3.0, 6.0]
        };
        for &nv in degrees {
            if nv.fract() != 0.0 && f == OrthoFamily::AssocLegendre {
                continue;
            }
            for &xv in &[-0.6, 0.2, 0.7] {
                let mut env = std::collections::HashMap::new();
                env.insert(x, xv);
                env.insert(n, nv);
                let Some(got) = crate::jit::eval_interp(d, &env, &pool) else {
                    panic!("{}: derivative did not evaluate at n={nv}", f.name())
                };
                let h = 1e-6;
                let mut a = vec![nv];
                a.extend(&ps);
                let mut lo = a.clone();
                lo.push(xv - h);
                let mut hi = a.clone();
                hi.push(xv + h);
                let fd = (reg.numeric_f64(f.name(), &hi).unwrap()
                    - reg.numeric_f64(f.name(), &lo).unwrap())
                    / (2.0 * h);
                assert!(
                    (got - fd).abs() <= 1e-6 * (1.0 + fd.abs()),
                    "{}(n={nv}, x={xv}): d/dx {got} vs FD {fd}",
                    f.name()
                );
            }
        }
    }
}

#[test]
fn diff_in_the_degree_is_refused_not_dropped() {
    let pool = ExprPool::new();
    let (n, x) = (sym(&pool, "n"), sym(&pool, "x"));
    let p = orthopoly(OrthoFamily::Legendre, n, &[], x, false, &pool).unwrap();
    assert!(crate::diff::diff(p, n, &pool).is_err());
    let g = orthopoly(
        OrthoFamily::Gegenbauer,
        pool.integer(3),
        &[n],
        x,
        false,
        &pool,
    )
    .unwrap();
    assert!(crate::diff::diff(g, n, &pool).is_err());
}

#[test]
fn simplify_special_values_and_rational_points() {
    let pool = ExprPool::new();
    let (n, x) = (sym(&pool, "n"), sym(&pool, "x"));
    let one = pool.integer(1);
    let s = |e| crate::simplify::simplify(e, &pool).value;
    let p1 = orthopoly(OrthoFamily::Legendre, n, &[], one, false, &pool).unwrap();
    assert_eq!(s(p1), one);
    let u1 = orthopoly(OrthoFamily::ChebyshevU, n, &[], one, false, &pool).unwrap();
    assert_eq!(s(u1), s(pool.add(vec![n, one])));
    // P_3(1/2) = (5/8 − 3/2)/2 = −7/16
    let p = orthopoly(
        OrthoFamily::Legendre,
        pool.integer(3),
        &[],
        pool.rational(1, 2),
        false,
        &pool,
    )
    .unwrap();
    assert_eq!(s(p), pool.rational(-7, 16));
    // Parity: H_3(−x) = −H_3(x)
    let neg_x = pool.mul(vec![pool.integer(-1), x]);
    let h = orthopoly(
        OrthoFamily::HermiteH,
        pool.integer(3),
        &[],
        neg_x,
        false,
        &pool,
    )
    .unwrap();
    let hx = orthopoly(OrthoFamily::HermiteH, pool.integer(3), &[], x, false, &pool).unwrap();
    assert_eq!(s(h), s(pool.mul(vec![pool.integer(-1), hx])));
}

#[test]
fn expand_orthopoly_after_substitution() {
    let pool = ExprPool::new();
    let (n, x) = (sym(&pool, "n"), sym(&pool, "x"));
    let p = orthopoly(OrthoFamily::Legendre, n, &[], x, true, &pool).unwrap();
    let mut m = std::collections::HashMap::new();
    m.insert(n, pool.integer(2));
    let p2 = crate::kernel::subs(p, &m, &pool);
    let e = expand_orthopoly(p2, &pool).unwrap();
    let want = orthopoly(OrthoFamily::Legendre, pool.integer(2), &[], x, true, &pool).unwrap();
    assert_eq!(e, want);
}

#[test]
fn symbolic_parameter_expansion() {
    // L_2^{(α)}(x) = (α+1)(α+2)/2 − (α+2)x + x²/2
    let pool = ExprPool::new();
    let (a, x) = (sym(&pool, "alpha"), sym(&pool, "x"));
    let e = orthopoly(
        OrthoFamily::AssocLaguerre,
        pool.integer(2),
        &[a],
        x,
        true,
        &pool,
    )
    .unwrap();
    for &(av, xv) in &[(0.3, 0.7), (-0.4, 2.5)] {
        let mut env = std::collections::HashMap::new();
        env.insert(a, av);
        env.insert(x, xv);
        let got = crate::jit::eval_interp(e, &env, &pool).unwrap();
        let want: f64 = (av + 1.0) * (av + 2.0) / 2.0 - (av + 2.0) * xv + xv * xv / 2.0;
        assert!((got - want).abs() < 1e-13, "{got} {want}");
    }
}

#[test]
fn degree_cap_is_a_coded_refusal() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let e = orthopoly(
        OrthoFamily::Legendre,
        pool.integer(MAX_EXPAND_DEGREE + 1),
        &[],
        x,
        true,
        &pool,
    )
    .unwrap_err();
    assert_eq!(crate::errors::AlkahestError::code(&e), "E-SPECFN-004");
    // The node is still available.
    assert!(orthopoly(
        OrthoFamily::Legendre,
        pool.integer(MAX_EXPAND_DEGREE + 1),
        &[],
        x,
        false,
        &pool
    )
    .is_ok());
}

#[cfg(flint_arb)]
#[test]
fn ball_encloses_the_f64_value_and_large_degree() {
    use crate::ball::ArbBall;
    let reg = PrimitiveRegistry::dispatch_registry();
    for f in OrthoFamily::ALL {
        let mut args = vec![7.0];
        args.extend(std::iter::repeat_n(0.25, f.n_params()));
        args.push(0.4);
        let v = reg.numeric_f64(f.name(), &args).unwrap();
        let balls: Vec<ArbBall> = args.iter().map(|&a| ArbBall::from_f64(a, 128)).collect();
        let b = reg.numeric_ball(f.name(), &balls).unwrap();
        assert!(
            (b.mid_f64() - v).abs() <= 1e-13 * (1.0 + v.abs()),
            "{}: ball {} vs f64 {v}",
            f.name(),
            b.mid_f64()
        );
        assert!(b.rad_f64() < 1e-25 * (1.0 + v.abs()), "{}", f.name());
    }
    // P_1000(0.3) — Arb's asymptotic path; compare with the f64 recurrence.
    let balls = [ArbBall::from_f64(1000.0, 128), ArbBall::from_f64(0.3, 128)];
    let b = reg.numeric_ball("legendre_p", &balls).unwrap();
    let v = reg.numeric_f64("legendre_p", &[1000.0, 0.3]).unwrap();
    assert!((b.mid_f64() - v).abs() < 1e-12, "{} {v}", b.mid_f64());
}
