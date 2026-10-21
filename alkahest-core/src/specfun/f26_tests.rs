//! Tests for F26: complex `hyper` values (₂F₁ continued past `|z| = 1`), the
//! ₂F₁ / ₁F₁ transformations, the new `expand_hyper` closed forms, Gauss's
//! digamma theorem and `ψ⁽ⁿ⁾` at every rational.
//!
//! Oracles: elementary closed forms evaluated independently (`log`, `exp`,
//! `sinh`), Arb's `acb_hypgeom_2f1` on the original node against the
//! transformed expression, and Arb's `acb_polygamma` / 200-bit balls against
//! the exact digamma / polygamma values.

use std::collections::HashMap;

use rug::Rational;

use super::gamma::polygamma_value;
use super::hyper::complex_value;
use super::*;
use crate::ball::{ArbBall, IntervalEval};
use crate::errors::AlkahestError;
use crate::eval::{eval_complex_f64, ComplexF64};
use crate::kernel::{Domain, ExprPool};
use crate::primitive::PrimitiveRegistry;

type C = (f64, f64);

fn close(got: C, want: C, tol: f64) -> bool {
    let d = ((got.0 - want.0).powi(2) + (got.1 - want.1).powi(2)).sqrt();
    let m = (want.0 * want.0 + want.1 * want.1).sqrt();
    d <= tol * (1.0 + m)
}

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cdiv(a: C, b: C) -> C {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn clog(a: C) -> C {
    (a.0.hypot(a.1).ln(), a.1.atan2(a.0))
}

/// `hyper(p, q, a…, b…, z)` argument list with real parameters.
fn hargs(a: &[f64], b: &[f64], z: C) -> Vec<C> {
    let mut v = vec![(a.len() as f64, 0.0), (b.len() as f64, 0.0)];
    v.extend(a.iter().map(|&x| (x, 0.0)));
    v.extend(b.iter().map(|&x| (x, 0.0)));
    v.push(z);
    v
}

/// A tiny deterministic generator (no `rand` dependency).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    /// A rational `n/d` in `(lo, hi)` with a small denominator.
    fn rational(&mut self, lo: f64, hi: f64) -> Rational {
        let d = 3 + (self.next() * 9.0) as i64;
        let n = (self.uniform(lo, hi) * d as f64).round() as i64;
        Rational::from((n, d))
    }
}

// ---------------------------------------------------------------------------
// Complex values
// ---------------------------------------------------------------------------

#[test]
fn complex_2f1_matches_log_inside_and_outside_the_unit_disk() {
    // ₂F₁(1, 1; 2; z) = −log(1 − z)/z, principal log.
    for z in [
        (0.3, 0.4),
        (-0.7, 0.2),
        (3.0, 2.0),
        (-5.0, -1.5),
        (0.5, -4.0),
        (1.5, 1e-3),
        (1.5, -1e-3),
    ] {
        let got = complex_value(&hargs(&[1.0, 1.0], &[2.0], z)).expect("value");
        let want = cdiv(clog((1.0 - z.0, -z.1)), z);
        let want = (-want.0, -want.1);
        assert!(close(got, want, 1e-14), "{z:?}: {got:?} vs {want:?}");
    }
}

#[test]
fn complex_2f1_on_the_cut_is_the_limit_from_below() {
    // z = 2 on the cut [1, ∞): −log(1 − (2 − i0))/2 = −log(−1 + i0)/2 = −iπ/2.
    let got = complex_value(&hargs(&[1.0, 1.0], &[2.0], (2.0, 0.0))).expect("value");
    assert!(
        close(got, (0.0, -std::f64::consts::FRAC_PI_2), 1e-14),
        "{got:?}"
    );
    let below = complex_value(&hargs(&[1.0, 1.0], &[2.0], (2.0, -1e-9))).unwrap();
    assert!(close(got, below, 1e-8), "{got:?} vs {below:?}");
}

#[test]
fn complex_1f1_0f1_and_3f2() {
    let z: C = (1.3, -2.1);
    // ₁F₁(1; 2; z) = (eᶻ − 1)/z
    let ez = (z.0.exp() * z.1.cos(), z.0.exp() * z.1.sin());
    let want = cdiv((ez.0 - 1.0, ez.1), z);
    let got = complex_value(&hargs(&[1.0], &[2.0], z)).unwrap();
    assert!(close(got, want, 1e-14), "{got:?} vs {want:?}");
    // ₀F₁(; 3/2; z²/4) = sinh(z)/z
    let z2 = cmul(z, z);
    let sh = (z.0.sinh() * z.1.cos(), z.0.cosh() * z.1.sin());
    let got = complex_value(&hargs(&[], &[1.5], (z2.0 / 4.0, z2.1 / 4.0))).unwrap();
    assert!(close(got, cdiv(sh, z), 1e-14), "{got:?}");
    // ₃F₂(1, 1, 1; 2, 2; w) = Li₂(w)/w inside the disk, declines outside.
    let w = (0.4, 0.5);
    let li = complex_value_kernel("polylog", &[(2.0, 0.0), w]);
    let got = complex_value(&hargs(&[1.0, 1.0, 1.0], &[2.0, 2.0], w)).unwrap();
    assert!(close(got, cdiv(li, w), 1e-14), "{got:?}");
    assert!(complex_value(&hargs(&[1.0, 1.0, 1.0], &[2.0, 2.0], (3.0, 1.0))).is_none());
    // A lower pole declines.
    assert!(complex_value(&hargs(&[0.5], &[-2.0], (0.3, 0.1))).is_none());
}

fn complex_value_kernel(name: &str, args: &[C]) -> C {
    super::complex_value(name, args).expect("kernel value")
}

#[test]
fn complex_evaluator_sees_hyper_and_gamma() {
    let pool = ExprPool::new();
    let z = pool.symbol("z", Domain::Complex);
    let e = hyper(
        &[pool.integer(1), pool.integer(1)],
        &[pool.integer(2)],
        z,
        &pool,
    )
    .unwrap();
    let env = HashMap::from([(z, ComplexF64::new(3.0, 2.0))]);
    let v = eval_complex_f64(e, &pool, &env).expect("complex value");
    let want = cdiv(clog((-2.0, -2.0)), (3.0, 2.0));
    assert!(close((v.re, v.im), (-want.0, -want.1), 1e-14));
    // Γ(½ + i) through acb_gamma: |Γ(½ + i)|² = π / cosh(π).
    let g = pool.func(
        "gamma",
        vec![pool.add(vec![
            pool.rational(1, 2),
            pool.mul(vec![pool.integer(1), z]),
        ])],
    );
    let env = HashMap::from([(z, ComplexF64::new(0.0, 1.0))]);
    let v = eval_complex_f64(g, &pool, &env).expect("complex gamma");
    let m2 = v.re * v.re + v.im * v.im;
    let want = std::f64::consts::PI / std::f64::consts::PI.cosh();
    assert!((m2 - want).abs() < 1e-14, "{m2} vs {want}");
}

// ---------------------------------------------------------------------------
// Transformations
// ---------------------------------------------------------------------------

/// Evaluate `expr` at `z = zv` in the complex evaluator.
fn at(expr: ExprId, z: ExprId, zv: C, pool: &ExprPool) -> C {
    let env = HashMap::from([(z, ComplexF64::new(zv.0, zv.1))]);
    let v = eval_complex_f64(expr, pool, &env)
        .unwrap_or_else(|e| panic!("{} at {zv:?}: {e:?}", pool.display(expr)));
    (v.re, v.im)
}

#[test]
fn every_2f1_transformation_holds_at_random_parameters_and_points() {
    let pool = ExprPool::new();
    let z = pool.symbol("z", Domain::Complex);
    let mut rng = Lcg(0xF26);
    let mut checked = 0;
    for _ in 0..12 {
        let (a, b, c) = (
            rng.rational(-2.0, 2.5),
            rng.rational(-2.0, 2.5),
            rng.rational(0.2, 3.0),
        );
        let lit = |q: &Rational| crate::orthopoly::rat_node(&pool, q);
        let Ok(node) = hyper(&[lit(&a), lit(&b)], &[lit(&c)], z, &pool) else {
            continue;
        };
        if !matches!(pool.get(node), crate::kernel::ExprData::Func { .. }) {
            continue;
        }
        for t in [
            HyperTransformation::Euler,
            HyperTransformation::PfaffA,
            HyperTransformation::PfaffB,
            HyperTransformation::OneMinusZ,
            HyperTransformation::InverseZ,
            HyperTransformation::InverseOneMinusZ,
            HyperTransformation::OneMinusInverseZ,
        ] {
            let rhs = match hyper_transform(node, t, &pool) {
                Ok(e) => e,
                Err(e) => {
                    // Only the logarithmic case may refuse.
                    assert_eq!(e.code(), "E-SPECFN-009", "{t:?}: {e}");
                    continue;
                }
            };
            for _ in 0..3 {
                // Off the real axis: inside every formula's region.
                let zv = (rng.uniform(-3.0, 3.0), rng.uniform(0.2, 2.5));
                let zv = if rng.next() < 0.5 { (zv.0, -zv.1) } else { zv };
                let lhs = at(node, z, zv, &pool);
                let got = at(rhs, z, zv, &pool);
                assert!(
                    close(got, lhs, 1e-9),
                    "{t:?} a={a} b={b} c={c} z={zv:?}: {got:?} vs {lhs:?}\n{}",
                    pool.display(rhs)
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 100, "only {checked} checks ran");
}

#[test]
fn quadratic_transformations_hold_in_their_regions() {
    let pool = ExprPool::new();
    let z = pool.symbol("z", Domain::Complex);
    let r = |n: i64, d: i64| pool.rational(n, d);
    let cases = [
        // c = 2b
        ([r(2, 7), r(3, 5)], r(6, 5)),
        // c = a − b + 1
        ([r(5, 4), r(1, 3)], r(23, 12)),
        // 2c = a + b + 1
        ([r(2, 3), r(-1, 5)], r(11, 15)),
    ];
    let mut rng = Lcg(0xA11);
    for (ab, c) in cases {
        let node = hyper(&ab, &[c], z, &pool).unwrap();
        let rhs = hyper_transform(node, HyperTransformation::Quadratic, &pool).unwrap();
        for _ in 0..6 {
            // |z| < ½ and Re z < ½: inside all three regions.
            let (rho, th) = (rng.uniform(0.05, 0.45), rng.uniform(-3.1, 3.1));
            let zv = (rho * th.cos(), rho * th.sin());
            let lhs = at(node, z, zv, &pool);
            let got = at(rhs, z, zv, &pool);
            assert!(close(got, lhs, 1e-11), "{zv:?}: {got:?} vs {lhs:?}");
        }
    }
    let other = hyper(&[r(1, 3), r(1, 7)], &[r(9, 5)], z, &pool).unwrap();
    let e = hyper_transform(other, HyperTransformation::Quadratic, &pool).unwrap_err();
    assert_eq!(e.code(), "E-SPECFN-009");
}

#[test]
fn kummer_and_refusals() {
    let pool = ExprPool::new();
    let z = pool.symbol("z", Domain::Complex);
    let node = hyper(&[pool.rational(2, 3)], &[pool.rational(7, 4)], z, &pool).unwrap();
    let rhs = hyper_transform(node, HyperTransformation::Kummer, &pool).unwrap();
    for zv in [(1.5, -0.5), (-4.0, 2.0), (0.2, 6.0)] {
        assert!(close(at(rhs, z, zv, &pool), at(node, z, zv, &pool), 1e-12));
    }
    // Logarithmic case: c − a − b = 0 for ₂F₁(1, 1; 2; z), a − b = 0 too.
    let log_case = hyper(
        &[pool.integer(1), pool.integer(1)],
        &[pool.integer(2)],
        z,
        &pool,
    )
    .unwrap();
    for t in [
        HyperTransformation::OneMinusZ,
        HyperTransformation::InverseZ,
        HyperTransformation::InverseOneMinusZ,
        HyperTransformation::OneMinusInverseZ,
    ] {
        let e = hyper_transform(log_case, t, &pool).unwrap_err();
        assert_eq!(e.code(), "E-SPECFN-009", "{t:?}");
    }
    // Wrong p, q and not a node at all.
    let e = hyper_transform(node, HyperTransformation::Euler, &pool).unwrap_err();
    assert_eq!(e.code(), "E-SPECFN-009");
    let e = hyper_transform(z, HyperTransformation::Euler, &pool).unwrap_err();
    assert_eq!(e.code(), "E-SPECFN-009");
    assert_eq!(
        HyperTransformation::from_name("z/(z-1)"),
        Some(HyperTransformation::PfaffA)
    );
    for t in HyperTransformation::ALL {
        assert_eq!(HyperTransformation::from_name(t.name()), Some(t));
    }
}

// ---------------------------------------------------------------------------
// expand_hyper: new closed forms
// ---------------------------------------------------------------------------

#[test]
fn expand_hyper_reaches_new_closed_forms() {
    let pool = ExprPool::new();
    let z = pool.symbol("z", Domain::Complex);
    let r = |n: i64, d: i64| pool.rational(n, d);
    let cases: Vec<(Vec<ExprId>, Vec<ExprId>)> = vec![
        // a, a + ½ families (A&S 15.1.9, .10, .13, .14)
        (vec![r(1, 3), r(5, 6)], vec![r(1, 2)]),
        (vec![r(1, 3), r(5, 6)], vec![r(3, 2)]),
        (vec![r(1, 3), r(5, 6)], vec![r(5, 3)]),
        (vec![r(1, 3), r(5, 6)], vec![r(2, 3)]),
        (vec![r(1, 2), pool.integer(1)], vec![pool.integer(2)]),
        // through Pfaff: ₂F₁(1, 1; 3/2; z) → atanh form of z/(z − 1)
        (vec![pool.integer(1), pool.integer(1)], vec![r(3, 2)]),
        // through Euler: c − a a negative integer → polynomial × power
        (vec![r(7, 2), r(1, 3)], vec![r(3, 2)]),
        // ₁F₁(a; 2a; z) → Bessel I, and Kummer to reach it
        (vec![r(1, 2)], vec![pool.integer(1)]),
        (vec![r(3, 4)], vec![r(3, 2)]),
    ];
    for (a, b) in cases {
        let node = hyper(&a, &b, z, &pool).unwrap();
        let e = expand_hyper(node, &pool).unwrap();
        assert!(
            !pool.display(e).to_string().contains("hyper"),
            "{} kept: {}",
            pool.display(node),
            pool.display(e)
        );
        for zv in [(0.3, 0.0), (-0.6, 0.0), (0.2, 0.5), (-0.4, -0.3)] {
            let want = at(node, z, zv, &pool);
            let got = at(e, z, zv, &pool);
            assert!(
                close(got, want, 1e-11),
                "{} = {} at {zv:?}: {got:?} vs {want:?}",
                pool.display(node),
                pool.display(e)
            );
        }
    }
}

#[test]
fn expand_hyper_uses_kummer_to_reach_erf() {
    // ₁F₁(1; 3/2; z) = eᶻ ₁F₁(½; 3/2; −z) = eᶻ √π erf(√z)/(2√z).
    let pool = ExprPool::new();
    let z = pool.symbol("z", Domain::Real);
    let node = hyper(&[pool.integer(1)], &[pool.rational(3, 2)], z, &pool).unwrap();
    let e = expand_hyper(node, &pool).unwrap();
    let s = pool.display(e).to_string();
    assert!(s.contains("erf") && !s.contains("hyper"), "{s}");
    let reg = PrimitiveRegistry::default_registry();
    for zv in [0.3_f64, 0.9, 2.5] {
        let got = crate::jit::eval_interp(e, &HashMap::from([(z, zv)]), &pool).unwrap();
        let want = reg.numeric_f64("hyper", &[1.0, 1.0, 1.0, 1.5, zv]).unwrap();
        assert!(
            (got - want).abs() < 1e-13 * want.abs(),
            "{zv}: {got} vs {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// Digamma and polygamma at rationals
// ---------------------------------------------------------------------------

fn ball(e: ExprId, pool: &ExprPool) -> ArbBall {
    crate::eval::eval_interval(e, pool, &IntervalEval::new(200))
        .unwrap_or_else(|_| panic!("no enclosure of {}", pool.display(e)))
}

fn arb_polygamma(n: u32, x: &Rational) -> ArbBall {
    let reg = PrimitiveRegistry::default_registry();
    reg.numeric_ball(
        "polygamma",
        &[
            ArbBall::from_f64(f64::from(n), 200),
            ArbBall::from_rational(x, 200),
        ],
    )
    .expect("Arb polygamma")
}

fn agree(a: &ArbBall, b: &ArbBall) -> bool {
    let d = (a.mid.to_f64() - b.mid.to_f64()).abs();
    d <= 1e-40 * (1.0 + b.mid.to_f64().abs())
}

#[test]
fn gauss_digamma_theorem_at_every_small_denominator() {
    let pool = ExprPool::new();
    for q in 2..=13_i64 {
        for p in 1..q {
            if Integer::from(p).gcd(&Integer::from(q)) != 1 {
                continue;
            }
            for shift in [0_i64, 2, -1] {
                let x = Rational::from((p, q)) + shift;
                let v = polygamma_value(0, crate::orthopoly::rat_node(&pool, &x), &pool)
                    .unwrap_or_else(|| panic!("ψ({x}) not exact"));
                let s = pool.display(v).to_string();
                assert!(!s.contains("digamma"), "ψ({x}) = {s}");
                assert!(
                    agree(&ball(v, &pool), &arb_polygamma(0, &x)),
                    "ψ({x}) = {s}"
                );
            }
        }
    }
    // Past the cap the node is kept.
    let big = pool.rational(1, MAX_GAUSS_TEST);
    assert!(polygamma_value(0, big, &pool).is_none());
}

const MAX_GAUSS_TEST: i64 = super::gamma::MAX_GAUSS_DENOMINATOR + 1;

use rug::Integer;

#[test]
fn polygamma_at_rationals_classical_or_hurwitz() {
    let pool = ExprPool::new();
    for n in 1..=6_u32 {
        for (p, q) in [
            (1, 3),
            (2, 3),
            (1, 4),
            (3, 4),
            (1, 6),
            (5, 6),
            (2, 5),
            (3, 7),
        ] {
            let x = Rational::from((p, q));
            for x in [x.clone(), x + 1] {
                let v = polygamma_value(n, crate::orthopoly::rat_node(&pool, &x), &pool)
                    .unwrap_or_else(|| panic!("ψ⁽{n}⁾({x}) not exact"));
                let s = pool.display(v).to_string();
                assert!(!s.contains("polygamma") && !s.contains("trigamma"), "{s}");
                // Classical where the theory says so: odd s = n + 1 at
                // denominators 3, 4, 6 has no Hurwitz node left.
                let classical = n % 2 == 0 && matches!(q, 3 | 4 | 6) || (n == 1 && q == 4);
                assert_eq!(!s.contains("hurwitz_zeta"), classical, "ψ⁽{n}⁾({x}) = {s}");
                assert!(
                    agree(&ball(v, &pool), &arb_polygamma(n, &x)),
                    "ψ⁽{n}⁾({x}) = {s}"
                );
            }
        }
    }
}

#[test]
fn expand_hyper_reaches_lowergamma() {
    // ₁F₁(a; a+1; −w) = a w^{−a} γ(a, w), and through Kummer
    // ₁F₁(1; a+1; w) = eʷ ₁F₁(a; a+1; −w).
    let pool = ExprPool::new();
    let w = pool.symbol("w", Domain::Real);
    let reg = PrimitiveRegistry::default_registry();
    let neg_w = pool.mul(vec![pool.integer(-1), w]);
    for (a, b, z, sign) in [
        (pool.rational(1, 3), pool.rational(4, 3), neg_w, -1.0),
        (pool.rational(5, 2), pool.rational(7, 2), neg_w, -1.0),
        (pool.integer(1), pool.rational(4, 3), w, 1.0),
    ] {
        let node = hyper(&[a], &[b], z, &pool).unwrap();
        let e = expand_hyper(node, &pool).unwrap();
        let s = pool.display(e).to_string();
        assert!(!s.contains("hyper"), "{s}");
        let av = crate::orthopoly::rational_literal(&pool, a)
            .unwrap()
            .to_f64();
        let bv = crate::orthopoly::rational_literal(&pool, b)
            .unwrap()
            .to_f64();
        for wv in [0.3_f64, 1.7, 4.0] {
            let got = crate::jit::eval_interp(e, &HashMap::from([(w, wv)]), &pool).unwrap();
            let want = reg
                .numeric_f64("hyper", &[1.0, 1.0, av, bv, sign * wv])
                .unwrap();
            assert!(
                (got - want).abs() < 1e-12 * want.abs(),
                "{s} at {wv}: {got} vs {want}"
            );
        }
    }
}
