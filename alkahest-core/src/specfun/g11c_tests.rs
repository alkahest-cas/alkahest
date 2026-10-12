//! Tests for G11c: named constants, `Γ`/`ψ`/`ψ⁽ⁿ⁾` exact values, Laurent
//! series at the poles, and the hypergeometric `ₚFq` node.
//!
//! Oracles: Arb (an independent route — `acb_polygamma`, `arb_hypgeom_*` —
//! against the exact closed forms), and mpmath at 40 digits for the Laurent
//! coefficients (recorded below).

// Reference values are quoted at mpmath's full 30 digits, deliberately.
#![allow(clippy::excessive_precision)]

use std::collections::HashMap;

use super::gamma::{contains_float, polygamma_node, polygamma_value, shift_off_poles};
use super::*;
use crate::ball::{ArbBall, IntervalEval};
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprPool};
use crate::primitive::PrimitiveRegistry;

fn sym(pool: &ExprPool, s: &str) -> ExprId {
    pool.symbol(s, Domain::Real)
}

/// A 200-bit enclosure of a constant expression.
fn ball(e: ExprId, pool: &ExprPool) -> ArbBall {
    crate::eval::eval_interval(e, pool, &IntervalEval::new(200))
        .unwrap_or_else(|_| panic!("no enclosure of {}", pool.display(e)))
}

fn overlaps(a: &ArbBall, b: &ArbBall) -> bool {
    a.lo().to_f64() <= b.hi().to_f64() && b.lo().to_f64() <= a.hi().to_f64()
}

/// `ψ⁽ⁿ⁾(x)` from Arb at 200 bits — an independent route.
fn arb_polygamma(n: u32, x: &rug::Rational) -> ArbBall {
    let reg = PrimitiveRegistry::default_registry();
    let xb = ArbBall::from_rational(x, 200);
    reg.numeric_ball("polygamma", &[ArbBall::from_f64(f64::from(n), 200), xb])
        .expect("Arb polygamma")
}

fn f64_of(e: ExprId, pool: &ExprPool) -> f64 {
    crate::jit::eval_interp(e, &HashMap::new(), pool).expect("evaluates")
}

// ---------------------------------------------------------------------------
// Named constants
// ---------------------------------------------------------------------------

#[test]
fn named_constants_print_parse_and_never_fold() {
    let pool = ExprPool::new();
    let g = euler_gamma(&pool);
    assert_eq!(pool.display(g).to_string(), "EulerGamma");
    assert_eq!(crate::kernel::render_latex(g, &pool), r"\gamma");
    assert_eq!(crate::kernel::render_latex(catalan(&pool), &pool), "G");
    let parsed =
        crate::parse::parse("EulerGamma + Catalan", &pool, &mut HashMap::new()).expect("parses");
    assert_eq!(parsed, pool.add(vec![g, catalan(&pool)]));
    // simplify keeps it symbolic.
    let s = crate::simplify::simplify(pool.mul(vec![pool.integer(2), g]), &pool).value;
    assert!(!contains_float(s, &pool));
    // Catalan and Glaisher enclosures (mpmath: catalan, glaisher).
    let c = ball(catalan(&pool), &pool);
    assert!((c.mid_f64() - 0.915_965_594_177_219_015_054_603_514_932).abs() < 1e-16);
    if let Ok(a) = crate::eval::eval_interval(glaisher(&pool), &pool, &IntervalEval::new(200)) {
        assert!((a.mid_f64() - 1.282_427_129_100_622_636_875_342_568_87).abs() < 1e-15);
        assert!(a.rad_f64() < 1e-50);
    }
}

#[test]
fn the_c_emitter_writes_the_constant_as_a_literal() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let e = pool.add(vec![x, euler_gamma(&pool)]);
    let c = crate::horner::emit_expr_c(e, &[x], &["x"], "f", &pool).expect("emits");
    assert!(c.contains("0.57721566490153"), "{c}");
}

// ---------------------------------------------------------------------------
// Exact values
// ---------------------------------------------------------------------------

#[test]
fn digamma_values_are_exact_and_match_arb() {
    let pool = ExprPool::new();
    let reg = PrimitiveRegistry::default_registry();
    let dig = reg.get("digamma").unwrap();
    for (n, d) in [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (1, 4),
        (3, 4),
        (1, 6),
        (5, 6),
    ] {
        for shift in [-4_i64, -1, 0, 1, 3, 17] {
            let q = rug::Rational::from((n, d)) + rug::Rational::from(shift);
            if q <= 0 && *q.denom() == 1 {
                continue;
            }
            let x = crate::orthopoly::rat_node(&pool, &q);
            let v = dig.simplify(&[x], &pool).expect("exact value");
            assert!(!contains_float(v, &pool), "float in ψ({q})");
            let got = ball(v, &pool);
            let want = arb_polygamma(0, &q);
            assert!(overlaps(&got, &want), "ψ({q}): {got} vs {want}");
            assert!(got.rad_f64() < 1e-50, "ψ({q}) is not tight: {got}");
        }
    }
    // The pole keeps its node.
    assert!(dig.simplify(&[pool.integer(0)], &pool).is_none());
    assert!(dig.simplify(&[pool.integer(-3)], &pool).is_none());
    // ψ(1) = −γ exactly.
    assert_eq!(
        dig.simplify(&[pool.integer(1)], &pool).unwrap(),
        pool.mul(vec![pool.integer(-1), euler_gamma(&pool)])
    );
}

#[test]
fn polygamma_values_are_exact_and_match_arb() {
    let pool = ExprPool::new();
    for n in 1..=6_u32 {
        for (num, den) in [(1, 1), (1, 2), (1, 4), (3, 4)] {
            if den == 4 && n != 1 {
                continue;
            }
            for shift in [-3_i64, 0, 2, 9] {
                let q = rug::Rational::from((num, den)) + rug::Rational::from(shift);
                if q <= 0 && *q.denom() == 1 {
                    continue;
                }
                let x = crate::orthopoly::rat_node(&pool, &q);
                let v = polygamma_value(n, x, &pool).expect("exact value");
                assert!(!contains_float(v, &pool));
                let got = ball(v, &pool);
                let want = arb_polygamma(n, &q);
                assert!(overlaps(&got, &want), "ψ^({n})({q}): {got} vs {want}");
            }
        }
    }
    // ψ₁(1) = π²/6, ψ₂(1) = −2ζ(3): odd zeta values stay symbolic.
    let one = pool.integer(1);
    let pi2_6 = pool.mul(vec![
        pool.rational(1, 6),
        pool.pow(crate::eval::symbols::pi_symbol(&pool), pool.integer(2)),
    ]);
    assert_eq!(
        crate::simplify::simplify(polygamma_node(1, one, &pool), &pool).value,
        crate::simplify::simplify(pi2_6, &pool).value
    );
    let z3 = node(numeric::Kernel::Zeta, vec![pool.integer(3)], &pool);
    assert_eq!(
        polygamma_node(2, one, &pool),
        crate::simplify::simplify(pool.mul(vec![pool.integer(-2), z3]), &pool).value
    );
}

#[test]
fn gamma_values_at_integers_and_half_integers() {
    let pool = ExprPool::new();
    let reg = PrimitiveRegistry::default_registry();
    let g = reg.get("gamma").unwrap();
    // Γ(30) = 29! does not fit an i64; it is still exact.
    let v = g.simplify(&[pool.integer(30)], &pool).unwrap();
    assert_eq!(
        v,
        pool.integer(rug::Integer::from(rug::Integer::factorial(29)))
    );
    let half = g.simplify(&[pool.rational(7, 2)], &pool).unwrap();
    assert!((f64_of(half, &pool) - 3.323_350_970_447_843).abs() < 1e-13);
}

#[test]
fn polygamma_constructor_refusals_and_spellings() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    assert_eq!(
        polygamma(pool.integer(0), x, &pool).unwrap(),
        pool.func("digamma", vec![x])
    );
    assert_eq!(
        polygamma(pool.integer(1), x, &pool).unwrap(),
        pool.func("trigamma", vec![x])
    );
    let e = polygamma(pool.rational(1, 2), x, &pool).unwrap_err();
    assert_eq!(e.code(), "E-SPECFN-008");
    let e = polygamma(pool.integer(-1), x, &pool).unwrap_err();
    assert_eq!(e.code(), "E-SPECFN-008");
    let e = polygamma(pool.integer(3), pool.integer(-2), &pool).unwrap_err();
    assert_eq!(e.code(), "E-SPECFN-006");
}

#[test]
fn the_polygamma_ladder_differentiates() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let n = sym(&pool, "n");
    let t = pool.func("trigamma", vec![x]);
    let d = crate::diff::diff(t, x, &pool).unwrap().value;
    assert_eq!(d, pool.func("polygamma", vec![pool.integer(2), x]));
    let d2 = crate::diff::diff(d, x, &pool).unwrap().value;
    assert_eq!(d2, pool.func("polygamma", vec![pool.integer(3), x]));
    // ∂/∂n declines.
    let pn = pool.func("polygamma", vec![n, x]);
    assert!(crate::diff::diff(pn, n, &pool).is_err());
    // Finite differences of ψ₂ against ψ₃ at x = 2.3.
    let reg = PrimitiveRegistry::default_registry();
    let h = 1e-5;
    let fd = (reg.numeric_f64("polygamma", &[2.0, 2.3 + h]).unwrap()
        - reg.numeric_f64("polygamma", &[2.0, 2.3 - h]).unwrap())
        / (2.0 * h);
    let exact = reg.numeric_f64("polygamma", &[3.0, 2.3]).unwrap();
    assert!((fd - exact).abs() < 1e-7 * exact.abs());
}

// ---------------------------------------------------------------------------
// Laurent series at the poles
// ---------------------------------------------------------------------------

/// `(valuation, f64 coefficients)` of `series(expr, var, point, order)`.
fn laurent(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    pool: &ExprPool,
) -> (i32, Vec<f64>) {
    let le =
        crate::calculus::series::local_expansion(expr, var, point, order, pool).expect("expands");
    for &c in &le.coeffs {
        assert!(
            !contains_float(c, pool),
            "float coefficient {}",
            pool.display(c)
        );
    }
    (
        le.valuation,
        le.coeffs.iter().map(|&c| f64_of(c, pool)).collect(),
    )
}

fn assert_coeffs(got: &[f64], want: &[f64]) {
    assert!(got.len() >= want.len(), "{got:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
            "c[{i}] = {g}, want {w}"
        );
    }
}

#[test]
fn gamma_has_a_laurent_series_at_zero_with_exact_coefficients() {
    let pool = ExprPool::new();
    let e = sym(&pool, "eps");
    let zero = pool.integer(0);
    let s = crate::calculus::series(pool.func("gamma", vec![e]), e, zero, 2, &pool).unwrap();
    assert!(!contains_float(s.expr(), &pool));
    let shown = pool.display(s.expr()).to_string();
    assert!(
        shown.contains("EulerGamma") && shown.contains("pi"),
        "{shown}"
    );
    // mpmath: taylor(gamma(1+e)) = 1, −γ, 0.98905…, −0.90747…, 0.98172…
    let (v, c) = laurent(pool.func("gamma", vec![e]), e, zero, 5, &pool);
    assert_eq!(v, -1);
    assert_coeffs(
        &c,
        &[
            1.0,
            -0.577_215_664_901_532_860_606_512_090_082,
            0.989_055_995_327_972_555_395_395_651_501,
            -0.907_479_076_080_886_289_016_560_167_356,
            0.981_728_086_834_400_187_336_380_294_022,
        ],
    );
}

#[test]
fn gamma_poles_at_negative_integers_have_residues() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    // Γ at −2: residue (−1)²/2! = 1/2 (mpmath reference).
    let (v, c) = laurent(pool.func("gamma", vec![x]), x, pool.integer(-2), 3, &pool);
    assert_eq!(v, -1);
    assert_coeffs(
        &c,
        &[
            0.5,
            0.461_392_167_549_233_569_696_743_954_959,
            0.936_616_248_987_836_632_242_813_758_189,
        ],
    );
    for k in 0..6_i64 {
        let (v, c) = laurent(pool.func("gamma", vec![x]), x, pool.integer(-k), 1, &pool);
        assert_eq!(v, -1);
        let fact: f64 = (1..=k).map(|j| j as f64).product();
        let want = if k % 2 == 0 { 1.0 } else { -1.0 } / fact;
        assert!((c[0] - want).abs() < 1e-14, "residue at −{k}");
    }
}

#[test]
fn digamma_and_trigamma_poles() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let zero = pool.integer(0);
    let (v, c) = laurent(pool.func("digamma", vec![x]), x, zero, 4, &pool);
    assert_eq!(v, -1);
    assert_coeffs(
        &c,
        &[
            -1.0,
            -0.577_215_664_901_532_860_606_512_090_082,
            1.644_934_066_848_226_436_472_415_166_65,
            -1.202_056_903_159_594_285_399_738_161_51,
        ],
    );
    let (v, c) = laurent(
        pool.func("trigamma", vec![x]),
        x,
        pool.integer(-1),
        4,
        &pool,
    );
    assert_eq!(v, -2);
    assert_coeffs(
        &c,
        &[
            1.0,
            0.0,
            2.644_934_066_848_226_436_472_415_166_65,
            -0.404_113_806_319_188_570_799_476_323_023,
        ],
    );
}

#[test]
fn dimensional_regularisation_example() {
    // m^{−2ε} Γ(ε) = 1/ε − γ − 2 log m + …  (mpmath at m = 3)
    let pool = ExprPool::new();
    let e = sym(&pool, "eps");
    let m = sym(&pool, "m");
    let f = pool.mul(vec![
        pool.pow(m, pool.mul(vec![pool.integer(-2), e])),
        pool.func("gamma", vec![e]),
    ]);
    let le = crate::calculus::series::local_expansion(f, e, pool.integer(0), 3, &pool).unwrap();
    assert_eq!(le.valuation, -1);
    let mut env = HashMap::new();
    env.insert(m, 3.0);
    let vals: Vec<f64> = le
        .coeffs
        .iter()
        .map(|&c| {
            assert!(!contains_float(c, &pool));
            crate::jit::eval_interp(c, &env, &pool).unwrap()
        })
        .collect();
    assert_coeffs(
        &vals,
        &[
            1.0,
            -2.774_440_242_237_752_243_397_002_563_93,
            4.671_226_362_298_251_891_970_893_343_48,
        ],
    );
}

#[test]
fn the_pole_shift_leaves_regular_points_alone() {
    let pool = ExprPool::new();
    let x = sym(&pool, "x");
    let xi = sym(&pool, "xi");
    assert!(shift_off_poles(
        pool.func("gamma", vec![pool.add(vec![xi, pool.integer(1)])]),
        xi,
        &pool
    )
    .is_none());
    assert!(shift_off_poles(pool.func("gamma", vec![x]), xi, &pool).is_none());
    assert!(shift_off_poles(pool.func("gamma", vec![xi]), xi, &pool).is_some());
}

// ---------------------------------------------------------------------------
// Hypergeometric
// ---------------------------------------------------------------------------

fn hyp(a: &[ExprId], b: &[ExprId], z: ExprId, pool: &ExprPool) -> ExprId {
    hyper(a, b, z, pool).expect("builds")
}

fn hyp_f64(a: &[f64], b: &[f64], z: f64) -> f64 {
    let reg = PrimitiveRegistry::default_registry();
    let mut args = vec![a.len() as f64, b.len() as f64];
    args.extend_from_slice(a);
    args.extend_from_slice(b);
    args.push(z);
    reg.numeric_f64("hyper", &args).expect("Arb pFq")
}

#[test]
fn hyper_canonical_forms_and_refusals() {
    let pool = ExprPool::new();
    let z = sym(&pool, "z");
    let a = sym(&pool, "a");
    // Common parameter cancels: ₂F₁(a, 1/2; a; z) = ₁F₀(1/2;; z).
    let e = hyp(&[a, pool.rational(1, 2)], &[a], z, &pool);
    assert_eq!(e, hyper::raw_node(&[pool.rational(1, 2)], &[], z, &pool));
    assert_eq!(
        hyp(&[a], &[pool.integer(3)], pool.integer(0), &pool),
        pool.integer(1)
    );
    assert_eq!(
        hyp(&[pool.integer(0), a], &[pool.integer(3)], z, &pool),
        pool.integer(1)
    );
    let err = hyper(&[a], &[pool.integer(-2)], z, &pool).unwrap_err();
    assert_eq!(err.code(), "E-SPECFN-006");
    // …unless the series terminates before the zero.
    assert!(hyper(&[pool.integer(-1)], &[pool.integer(-2)], z, &pool).is_ok());
    // Prints, and reads back.
    let e = hyp(&[a, pool.integer(3)], &[pool.rational(7, 3)], z, &pool);
    let shown = pool.display(e).to_string();
    assert_eq!(shown, "hyper(2, 1, a, 3, 7/3, z)");
    assert_eq!(
        crate::kernel::render_latex(e, &pool),
        r"{}_{2}F_{1}\!\left(a, 3; \frac{7}{3}; z\right)"
    );
}

#[test]
fn malformed_hyper_nodes_are_declined_not_indexed() {
    let pool = ExprPool::new();
    let z = sym(&pool, "z");
    let reg = PrimitiveRegistry::default_registry();
    let bad = pool.func("hyper", vec![pool.integer(2), pool.integer(1), z, z]);
    assert!(reg
        .get("hyper")
        .unwrap()
        .simplify(&[pool.integer(2), pool.integer(1), z, z], &pool)
        .is_none());
    assert!(crate::diff::diff(bad, z, &pool).is_err());
    assert!(reg.numeric_f64("hyper", &[2.0, 1.0, 0.5, 0.5]).is_none());
    assert_eq!(expand_hyper(bad, &pool).unwrap(), bad);
    let _ = crate::kernel::render_latex(bad, &pool);
}

#[test]
fn hyper_numbers_and_derivative() {
    let pool = ExprPool::new();
    let z = sym(&pool, "z");
    // mpmath: hyp2f1(1/2, 3, 7/3, 0.3), hyp1f1(0.5, 1.5, -2), hyp0f1(2.5, 1.2),
    // hyp3f2(1, 1, 1, 2, 2, 0.5) = 2·Li₂(1/2) = 1.164481…
    assert!((hyp_f64(&[0.5, 3.0], &[7.0 / 3.0], 0.3) - 1.266_276_825_407_982_6).abs() < 1e-14);
    assert!((hyp_f64(&[0.5], &[1.5], -2.0) - 0.598_144_006_661_304_1).abs() < 1e-14);
    assert!((hyp_f64(&[1.0, 1.0, 1.0], &[2.0, 2.0], 0.5) - 1.164_481_052_930_025).abs() < 1e-14);
    // Past the cut ₂F₁ is not real: declined.
    let reg = PrimitiveRegistry::default_registry();
    assert!(reg
        .numeric_f64("hyper", &[2.0, 1.0, 0.5, 0.5, 1.5, 2.0])
        .is_none());
    // d/dz by central differences.
    let e = hyp(
        &[pool.rational(1, 2), pool.integer(3)],
        &[pool.rational(7, 3)],
        z,
        &pool,
    );
    let d = crate::diff::diff(e, z, &pool).unwrap().value;
    let at = |v: f64| {
        let mut env = HashMap::new();
        env.insert(z, v);
        crate::jit::eval_interp(d, &env, &pool).unwrap()
    };
    let h = 1e-6;
    let fd = (hyp_f64(&[0.5, 3.0], &[7.0 / 3.0], 0.3 + h)
        - hyp_f64(&[0.5, 3.0], &[7.0 / 3.0], 0.3 - h))
        / (2.0 * h);
    assert!((at(0.3) - fd).abs() < 1e-7 * fd.abs());
}

#[test]
fn expand_hyper_closed_forms_match_the_series() {
    let pool = ExprPool::new();
    let z = sym(&pool, "z");
    let w = sym(&pool, "w");
    let r = |n, d| pool.rational(n, d);
    let i = |n: i64| pool.integer(n);
    let neg = |e: ExprId| pool.mul(vec![pool.integer(-1), e]);
    // (a, b, argument expression, f64 params a, b, f64 z-value given the
    // symbol value 0.37)
    #[allow(clippy::type_complexity)]
    let cases: Vec<(Vec<ExprId>, Vec<ExprId>, ExprId, Vec<f64>, Vec<f64>, f64)> = vec![
        (vec![], vec![], z, vec![], vec![], 0.37),
        (vec![r(3, 2)], vec![], z, vec![1.5], vec![], 0.37),
        (
            vec![i(1), i(1)],
            vec![i(2)],
            z,
            vec![1.0, 1.0],
            vec![2.0],
            0.37,
        ),
        (
            vec![r(1, 2), i(1)],
            vec![r(3, 2)],
            z,
            vec![0.5, 1.0],
            vec![1.5],
            0.37,
        ),
        (
            vec![r(1, 2), r(1, 2)],
            vec![r(3, 2)],
            z,
            vec![0.5, 0.5],
            vec![1.5],
            0.37,
        ),
        (vec![i(1)], vec![i(2)], z, vec![1.0], vec![2.0], 0.37),
        (
            vec![r(1, 2)],
            vec![r(3, 2)],
            neg(w),
            vec![0.5],
            vec![1.5],
            -0.37,
        ),
        (vec![], vec![r(1, 2)], z, vec![], vec![0.5], 0.37),
        (vec![], vec![r(3, 2)], z, vec![], vec![1.5], 0.37),
        (vec![], vec![r(7, 3)], z, vec![], vec![7.0 / 3.0], 0.37),
        (
            vec![],
            vec![r(7, 3)],
            neg(w),
            vec![],
            vec![7.0 / 3.0],
            -0.37,
        ),
        (
            vec![i(-4), r(1, 3)],
            vec![r(5, 2)],
            z,
            vec![-4.0, 1.0 / 3.0],
            vec![2.5],
            0.37,
        ),
    ];
    for (a, b, arg, fa, fb, zv) in cases {
        let e = hyp(&a, &b, arg, &pool);
        let x = expand_hyper(e, &pool).unwrap();
        assert!(
            !pool.display(x).to_string().contains("hyper"),
            "not expanded: {}",
            pool.display(x)
        );
        let mut env = HashMap::new();
        env.insert(z, 0.37);
        env.insert(w, 0.37);
        let got = crate::jit::eval_interp(x, &env, &pool).unwrap();
        let want = hyp_f64(&fa, &fb, zv);
        assert!(
            (got - want).abs() < 1e-12 * (1.0 + want.abs()),
            "{}: {got} vs {want}",
            pool.display(e)
        );
    }
}

#[test]
fn gauss_summation() {
    let pool = ExprPool::new();
    let one = pool.integer(1);
    // ₂F₁(1/2, 1/3; 2; 1) = Γ(2)Γ(7/6)/(Γ(3/2)Γ(5/3)).
    let e = hyp(
        &[pool.rational(1, 2), pool.rational(1, 3)],
        &[pool.integer(2)],
        one,
        &pool,
    );
    let x = expand_hyper(e, &pool).unwrap();
    let want = 1.159_595_266_963_928_4;
    assert!((f64_of(x, &pool) - want).abs() < 1e-13);
    // c − a − b ≤ 0 diverges: kept.
    let e = hyp(
        &[pool.integer(1), pool.integer(1)],
        &[pool.integer(2)],
        one,
        &pool,
    );
    assert!(pool
        .display(expand_hyper(e, &pool).unwrap())
        .to_string()
        .contains("hyper"));
    // Chu–Vandermonde: c − a a non-positive integer gives 0.
    let a = pool.rational(7, 2);
    let e = hyp(
        &[a, pool.rational(-5, 2)],
        &[pool.rational(3, 2)],
        one,
        &pool,
    );
    assert_eq!(expand_hyper(e, &pool).unwrap(), pool.integer(0));
}

#[test]
fn a_terminating_series_past_the_cap_is_refused() {
    let pool = ExprPool::new();
    let z = sym(&pool, "z");
    let e = hyp(&[pool.integer(-500)], &[pool.integer(3)], z, &pool);
    assert_eq!(expand_hyper(e, &pool).unwrap_err().code(), "E-SPECFN-004");
}
