//! Tests for the Laplace transform and its inverse.
//!
//! Forward transforms are checked against the analytic table; round-trips
//! `L⁻¹{L{f}}` are checked by `simplify`-to-equality (or, where exact structural
//! equality is brittle, by numeric sampling of `F(s) − F_expected(s)`).

use super::*;
use crate::kernel::{Domain, ExprPool};

/// Numeric evaluation of an expression in a single variable at `var = val`,
/// used for sampling-based equality checks.  Handles the function heads the
/// Laplace table emits (incl. `sinh`/`cosh`/`heaviside`), which the generic
/// `jit::eval_interp` does not.
fn eval_at(expr: ExprId, var: ExprId, val: f64, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (n, d) = r.0.clone().into_numer_denom();
            Some(n.to_f64() / d.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { .. } => {
            if expr == var {
                Some(val)
            } else {
                None
            }
        }
        ExprData::Add(args) => args
            .iter()
            .try_fold(0.0, |acc, &a| Some(acc + eval_at(a, var, val, pool)?)),
        ExprData::Mul(args) => args
            .iter()
            .try_fold(1.0, |acc, &a| Some(acc * eval_at(a, var, val, pool)?)),
        ExprData::Pow { base, exp } => {
            Some(eval_at(base, var, val, pool)?.powf(eval_at(exp, var, val, pool)?))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let x = eval_at(args[0], var, val, pool)?;
            Some(match name.as_str() {
                "sin" => x.sin(),
                "cos" => x.cos(),
                "tan" => x.tan(),
                "sinh" => x.sinh(),
                "cosh" => x.cosh(),
                "exp" => x.exp(),
                "log" => x.ln(),
                "sqrt" => x.sqrt(),
                "heaviside" => {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        0.0
                    } else {
                        0.5
                    }
                }
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Assert two single-variable expressions agree numerically at a set of sample
/// points (avoids brittle structural comparison after simplification).
fn assert_numeric_eq(a: ExprId, b: ExprId, var: ExprId, samples: &[f64], pool: &ExprPool) {
    for &x in samples {
        let va = eval_at(a, var, x, pool);
        let vb = eval_at(b, var, x, pool);
        match (va, vb) {
            (Some(va), Some(vb)) => assert!(
                (va - vb).abs() < 1e-6 * (1.0 + va.abs() + vb.abs()),
                "mismatch at {x}: {} = {va} vs {} = {vb}",
                pool.display(a),
                pool.display(b),
            ),
            _ => panic!(
                "could not numerically evaluate at {x}: {} / {}",
                pool.display(a),
                pool.display(b)
            ),
        }
    }
}

fn setup() -> (ExprPool, ExprId, ExprId) {
    let pool = ExprPool::new();
    let t = pool.symbol("t", Domain::Real);
    let s = pool.symbol("s", Domain::Real);
    (pool, t, s)
}

// ── forward table ──────────────────────────────────────────────────────────

#[test]
fn forward_constant() {
    let (pool, t, s) = setup();
    let f = pool.integer(5_i32);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    // 5/s
    let want = pool.mul(vec![pool.integer(5_i32), pool.pow(s, pool.integer(-1_i32))]);
    assert_numeric_eq(got, want, s, &[2.0, 3.0, 5.0], &pool);
}

#[test]
fn forward_t_power() {
    let (pool, t, s) = setup();
    // L{t^3} = 6/s^4
    let f = pool.pow(t, pool.integer(3_i32));
    let got = laplace_transform(f, t, s, &pool).unwrap();
    let want = pool.mul(vec![pool.integer(6_i32), pool.pow(s, pool.integer(-4_i32))]);
    assert_numeric_eq(got, want, s, &[2.0, 3.0, 5.0], &pool);
}

#[test]
fn forward_exp() {
    let (pool, t, s) = setup();
    // L{e^{2t}} = 1/(s-2)
    let f = pool.func("exp", vec![pool.mul(vec![pool.integer(2_i32), t])]);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    let want = pool.pow(
        pool.add(vec![s, pool.integer(-2_i32)]),
        pool.integer(-1_i32),
    );
    assert_numeric_eq(got, want, s, &[3.0, 4.0, 5.0], &pool);
}

#[test]
fn forward_sin_cos() {
    let (pool, t, s) = setup();
    let s2 = pool.pow(s, pool.integer(2_i32));
    // L{sin(3t)} = 3/(s²+9)
    let sin3t = pool.func("sin", vec![pool.mul(vec![pool.integer(3_i32), t])]);
    let got = laplace_transform(sin3t, t, s, &pool).unwrap();
    let want = pool.mul(vec![
        pool.integer(3_i32),
        pool.pow(
            pool.add(vec![s2, pool.integer(9_i32)]),
            pool.integer(-1_i32),
        ),
    ]);
    assert_numeric_eq(got, want, s, &[2.0, 3.0, 5.0], &pool);

    // L{cos(3t)} = s/(s²+9)
    let cos3t = pool.func("cos", vec![pool.mul(vec![pool.integer(3_i32), t])]);
    let got = laplace_transform(cos3t, t, s, &pool).unwrap();
    let want = pool.mul(vec![
        s,
        pool.pow(
            pool.add(vec![s2, pool.integer(9_i32)]),
            pool.integer(-1_i32),
        ),
    ]);
    assert_numeric_eq(got, want, s, &[2.0, 3.0, 5.0], &pool);
}

#[test]
fn forward_sinh_cosh() {
    let (pool, t, s) = setup();
    let s2 = pool.pow(s, pool.integer(2_i32));
    // L{sinh(2t)} = 2/(s²−4)
    let f = pool.func("sinh", vec![pool.mul(vec![pool.integer(2_i32), t])]);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    let want = pool.mul(vec![
        pool.integer(2_i32),
        pool.pow(
            pool.add(vec![s2, pool.integer(-4_i32)]),
            pool.integer(-1_i32),
        ),
    ]);
    assert_numeric_eq(got, want, s, &[3.0, 4.0, 5.0], &pool);

    // L{cosh(2t)} = s/(s²−4)
    let f = pool.func("cosh", vec![pool.mul(vec![pool.integer(2_i32), t])]);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    let want = pool.mul(vec![
        s,
        pool.pow(
            pool.add(vec![s2, pool.integer(-4_i32)]),
            pool.integer(-1_i32),
        ),
    ]);
    assert_numeric_eq(got, want, s, &[3.0, 4.0, 5.0], &pool);
}

#[test]
fn forward_linearity() {
    let (pool, t, s) = setup();
    // L{2t + 3} = 2/s² + 3/s
    let f = pool.add(vec![
        pool.mul(vec![pool.integer(2_i32), t]),
        pool.integer(3_i32),
    ]);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    let want = pool.add(vec![
        pool.mul(vec![pool.integer(2_i32), pool.pow(s, pool.integer(-2_i32))]),
        pool.mul(vec![pool.integer(3_i32), pool.pow(s, pool.integer(-1_i32))]),
    ]);
    assert_numeric_eq(got, want, s, &[2.0, 3.0, 5.0], &pool);
}

#[test]
fn forward_s_shift() {
    let (pool, t, s) = setup();
    // L{e^{2t} cos(3t)} = (s−2)/((s−2)²+9)
    let cos3t = pool.func("cos", vec![pool.mul(vec![pool.integer(3_i32), t])]);
    let exp2t = pool.func("exp", vec![pool.mul(vec![pool.integer(2_i32), t])]);
    let f = pool.mul(vec![exp2t, cos3t]);
    let got = laplace_transform(f, t, s, &pool).unwrap();

    let s_minus_2 = pool.add(vec![s, pool.integer(-2_i32)]);
    let sm2_sq = pool.pow(s_minus_2, pool.integer(2_i32));
    let want = pool.mul(vec![
        s_minus_2,
        pool.pow(
            pool.add(vec![sm2_sq, pool.integer(9_i32)]),
            pool.integer(-1_i32),
        ),
    ]);
    assert_numeric_eq(got, want, s, &[3.0, 4.0, 6.0], &pool);
}

#[test]
fn forward_t_times_exp_sin() {
    // Task test: L{t e^{2t} sin(3t)}.
    // L{sin(3t)} = 3/(s²+9); shift s→s−2: 3/((s−2)²+9);
    // ×t ⇒ −d/ds: 6(s−2)/((s−2)²+9)².
    let (pool, t, s) = setup();
    let sin3t = pool.func("sin", vec![pool.mul(vec![pool.integer(3_i32), t])]);
    let exp2t = pool.func("exp", vec![pool.mul(vec![pool.integer(2_i32), t])]);
    let f = pool.mul(vec![t, exp2t, sin3t]);
    let got = laplace_transform(f, t, s, &pool).unwrap();

    let s_minus_2 = pool.add(vec![s, pool.integer(-2_i32)]);
    let sm2_sq = pool.pow(s_minus_2, pool.integer(2_i32));
    let denom = pool.add(vec![sm2_sq, pool.integer(9_i32)]);
    let want = pool.mul(vec![
        pool.integer(6_i32),
        s_minus_2,
        pool.pow(denom, pool.integer(-2_i32)),
    ]);
    assert_numeric_eq(got, want, s, &[3.0, 4.0, 5.5], &pool);
}

#[test]
fn forward_heaviside_step() {
    let (pool, t, s) = setup();
    // L{θ(t−2)} = e^{−2s}/s
    let arg = pool.add(vec![t, pool.integer(-2_i32)]);
    let f = pool.func("heaviside", vec![arg]);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    let want = pool.mul(vec![
        pool.func("exp", vec![pool.mul(vec![pool.integer(-2_i32), s])]),
        pool.pow(s, pool.integer(-1_i32)),
    ]);
    assert_numeric_eq(got, want, s, &[1.0, 2.0, 3.0], &pool);
}

#[test]
fn forward_heaviside_shifted_function() {
    let (pool, t, s) = setup();
    // L{θ(t−1)·(t−1)} = e^{−s}·L{t} = e^{−s}/s²
    let tm1 = pool.add(vec![t, pool.integer(-1_i32)]);
    let f = pool.mul(vec![pool.func("heaviside", vec![tm1]), tm1]);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    let want = pool.mul(vec![
        pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), s])]),
        pool.pow(s, pool.integer(-2_i32)),
    ]);
    assert_numeric_eq(got, want, s, &[1.5, 2.0, 3.0], &pool);
}

#[test]
fn forward_dirac() {
    let (pool, t, s) = setup();
    // L{δ(t)} = 1
    let f = pool.func("diracdelta", vec![t]);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    assert_numeric_eq(got, pool.integer(1_i32), s, &[1.0, 2.0, 3.0], &pool);

    // L{δ(t−3)} = e^{−3s}
    let arg = pool.add(vec![t, pool.integer(-3_i32)]);
    let f = pool.func("diracdelta", vec![arg]);
    let got = laplace_transform(f, t, s, &pool).unwrap();
    let want = pool.func("exp", vec![pool.mul(vec![pool.integer(-3_i32), s])]);
    assert_numeric_eq(got, want, s, &[0.5, 1.0, 2.0], &pool);
}

#[test]
fn forward_same_variable_errors() {
    let (pool, t, _s) = setup();
    let f = pool.integer(1_i32);
    assert_eq!(
        laplace_transform(f, t, t, &pool),
        Err(LaplaceError::SameVariable)
    );
}

#[test]
fn forward_declines_unknown() {
    let (pool, t, s) = setup();
    // L{log(t)} is not in the table (it is −(γ + log s)/s) — decline.
    let f = pool.func("log", vec![t]);
    assert!(matches!(
        laplace_transform(f, t, s, &pool),
        Err(LaplaceError::NoRule(_))
    ));
}

// ── derivative rule ──────────────────────────────────────────────────────────

#[test]
fn derivative_rule_second_order() {
    // L{y'' + y} at the rule level with y(0)=y0, y'(0)=y1, F = L{y}.
    // L{y''} = s²F − s·y0 − y1 ; plus L{y} = F.
    let (pool, _t, s) = setup();
    let big_f = pool.symbol("F", Domain::Real);
    let y0 = pool.symbol("y0", Domain::Real);
    let y1 = pool.symbol("y1", Domain::Real);

    let l_ypp = laplace_derivative_rule(big_f, s, 2, &[y0, y1], &pool);
    // s²F − s·y0 − y1 ; compare via difference simplifying to 0.
    let want = pool.add(vec![
        pool.mul(vec![pool.pow(s, pool.integer(2_i32)), big_f]),
        pool.mul(vec![pool.integer(-1_i32), s, y0]),
        pool.mul(vec![pool.integer(-1_i32), y1]),
    ]);
    let diff =
        crate::simplify::simplify_expanded(pool.add(vec![l_ypp, neg(want, &pool)]), &pool).value;
    assert_eq!(
        diff,
        pool.integer(0_i32),
        "L{{y''}} mismatch: {}",
        pool.display(l_ypp)
    );
}

#[test]
fn derivative_rule_first_order_zero_ic() {
    // L{y'} with y(0)=0 is s·F.
    let (pool, _t, s) = setup();
    let big_f = pool.symbol("F", Domain::Real);
    let got = laplace_derivative_rule(big_f, s, 1, &[pool.integer(0_i32)], &pool);
    let want = pool.mul(vec![s, big_f]);
    let diff =
        crate::simplify::simplify_expanded(pool.add(vec![got, neg(want, &pool)]), &pool).value;
    assert_eq!(diff, pool.integer(0_i32), "got {}", pool.display(got));
}

// ── inverse transform ──────────────────────────────────────────────────────

#[test]
fn inverse_simple_pole() {
    let (pool, t, s) = setup();
    // L⁻¹{1/(s−2)} = e^{2t}
    let big_f = pool.pow(
        pool.add(vec![s, pool.integer(-2_i32)]),
        pool.integer(-1_i32),
    );
    let got = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    let want = pool.func("exp", vec![pool.mul(vec![pool.integer(2_i32), t])]);
    assert_numeric_eq(got, want, t, &[0.0, 0.5, 1.0], &pool);
}

#[test]
fn inverse_repeated_pole() {
    let (pool, t, s) = setup();
    // L⁻¹{1/(s−1)³} = t² e^{t}/2
    let base = pool.add(vec![s, pool.integer(-1_i32)]);
    let big_f = pool.pow(base, pool.integer(-3_i32));
    let got = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    let want = pool.mul(vec![
        pool.rational(1_i32, 2_i32),
        pool.pow(t, pool.integer(2_i32)),
        pool.func("exp", vec![t]),
    ]);
    assert_numeric_eq(got, want, t, &[0.0, 0.5, 1.0, 2.0], &pool);
}

#[test]
fn inverse_complex_poles() {
    let (pool, t, s) = setup();
    // L⁻¹{1/(s²+4)} = sin(2t)/2
    let s2 = pool.pow(s, pool.integer(2_i32));
    let big_f = pool.pow(
        pool.add(vec![s2, pool.integer(4_i32)]),
        pool.integer(-1_i32),
    );
    let got = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    let want = pool.mul(vec![
        pool.rational(1_i32, 2_i32),
        pool.func("sin", vec![pool.mul(vec![pool.integer(2_i32), t])]),
    ]);
    assert_numeric_eq(got, want, t, &[0.1, 0.5, 1.0, 2.0], &pool);
}

#[test]
fn inverse_damped_oscillation() {
    let (pool, t, s) = setup();
    // L⁻¹{ s / (s²+2s+5) }.  Denominator = (s+1)²+4, p=−1, ω=2.
    // numerator B s + C with B=1, C=0 ⇒ e^{−t}( cos2t + ((0+1·(−1))/2) sin2t )
    //   = e^{−t}( cos2t − (1/2) sin2t ).
    let s2 = pool.pow(s, pool.integer(2_i32));
    let denom = pool.add(vec![
        s2,
        pool.mul(vec![pool.integer(2_i32), s]),
        pool.integer(5_i32),
    ]);
    let big_f = pool.mul(vec![s, pool.pow(denom, pool.integer(-1_i32))]);
    let got = inverse_laplace_transform(big_f, s, t, &pool).unwrap();

    let exp_neg_t = pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), t])]);
    let two_t = pool.mul(vec![pool.integer(2_i32), t]);
    let want = pool.mul(vec![
        exp_neg_t,
        pool.add(vec![
            pool.func("cos", vec![two_t]),
            pool.mul(vec![
                pool.rational(-1_i32, 2_i32),
                pool.func("sin", vec![two_t]),
            ]),
        ]),
    ]);
    assert_numeric_eq(got, want, t, &[0.0, 0.3, 0.8, 1.5], &pool);
}

#[test]
fn inverse_proper_rational_repeated_and_complex() {
    // Task test: a proper rational with repeated + complex poles.
    // F(s) = 1 / ((s−1)² (s²+1)).  Verify by round-trip numeric sampling of
    // L{ L⁻¹{F} } == F.
    let (pool, t, s) = setup();
    let s2 = pool.pow(s, pool.integer(2_i32));
    let sm1 = pool.add(vec![s, pool.integer(-1_i32)]);
    let sm1_sq = pool.pow(sm1, pool.integer(2_i32));
    let quad = pool.add(vec![s2, pool.integer(1_i32)]);
    let denom = pool.mul(vec![sm1_sq, quad]);
    let big_f = pool.pow(denom, pool.integer(-1_i32));

    let f_of_t = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    // Round-trip: transform back and compare to F(s) numerically.
    let back = laplace_transform(f_of_t, t, s, &pool).unwrap();
    assert_numeric_eq(back, big_f, s, &[2.0, 3.0, 4.0, 5.0], &pool);
}

#[test]
fn inverse_delay_heaviside() {
    let (pool, t, s) = setup();
    // L⁻¹{ e^{−2s}/s } = θ(t−2)
    let big_f = pool.mul(vec![
        pool.func("exp", vec![pool.mul(vec![pool.integer(-2_i32), s])]),
        pool.pow(s, pool.integer(-1_i32)),
    ]);
    let got = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    // θ(t−2): 0 for t<2, 1 for t>2.
    let want = pool.func("heaviside", vec![pool.add(vec![t, pool.integer(-2_i32)])]);
    assert_numeric_eq(got, want, t, &[0.5, 1.0, 3.0, 4.0], &pool);
}

#[test]
fn inverse_declines_improper() {
    let (pool, t, s) = setup();
    // F(s) = s/(s−1) is improper (= 1 + 1/(s−1)); polynomial part ⇒ derivative
    // of δ, which we decline.
    let big_f = pool.mul(vec![
        s,
        pool.pow(
            pool.add(vec![s, pool.integer(-1_i32)]),
            pool.integer(-1_i32),
        ),
    ]);
    assert!(matches!(
        inverse_laplace_transform(big_f, s, t, &pool),
        Err(LaplaceError::NotInvertible(_))
    ));
}

// ── round-trips ──────────────────────────────────────────────────────────────

#[test]
fn round_trip_exp() {
    let (pool, t, s) = setup();
    // f(t) = e^{3t} ; L⁻¹{L{f}} = f.
    let f = pool.func("exp", vec![pool.mul(vec![pool.integer(3_i32), t])]);
    let big_f = laplace_transform(f, t, s, &pool).unwrap();
    let back = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    assert_numeric_eq(back, f, t, &[0.0, 0.2, 0.5, 1.0], &pool);
}

#[test]
fn round_trip_sin() {
    let (pool, t, s) = setup();
    let f = pool.func("sin", vec![pool.mul(vec![pool.integer(2_i32), t])]);
    let big_f = laplace_transform(f, t, s, &pool).unwrap();
    let back = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    assert_numeric_eq(back, f, t, &[0.1, 0.5, 1.0, 2.5], &pool);
}

#[test]
fn round_trip_t_squared() {
    let (pool, t, s) = setup();
    let f = pool.pow(t, pool.integer(2_i32));
    let big_f = laplace_transform(f, t, s, &pool).unwrap();
    let back = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    assert_numeric_eq(back, f, t, &[0.0, 0.5, 1.0, 2.0], &pool);
}

#[test]
fn round_trip_t_sin() {
    // Frequency-diff of L{sin} produces a repeated quadratic pole; n = 2
    // inverse is required for L⁻¹{L{t sin(ωt)}} = t sin(ωt).
    let (pool, t, s) = setup();
    let f = pool.mul(vec![
        t,
        pool.func("sin", vec![pool.mul(vec![pool.integer(2_i32), t])]),
    ]);
    let big_f = laplace_transform(f, t, s, &pool).unwrap();
    let back = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    assert_numeric_eq(back, f, t, &[0.25, 0.5, 1.0, 1.5, 2.0], &pool);
}

#[test]
fn round_trip_t_cos() {
    let (pool, t, s) = setup();
    let f = pool.mul(vec![
        t,
        pool.func("cos", vec![pool.mul(vec![pool.integer(3_i32), t])]),
    ]);
    let big_f = laplace_transform(f, t, s, &pool).unwrap();
    let back = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    assert_numeric_eq(back, f, t, &[0.25, 0.5, 1.0, 1.5, 2.0], &pool);
}

#[test]
fn round_trip_table_smoke() {
    // Cheap forward∘inverse identity checks on the core rational table.
    let (pool, t, s) = setup();
    let cases: Vec<ExprId> = vec![
        pool.integer(1_i32),
        t,
        pool.pow(t, pool.integer(2_i32)),
        pool.func("exp", vec![pool.mul(vec![pool.integer(-2_i32), t])]),
        pool.func("sin", vec![pool.mul(vec![pool.integer(5_i32), t])]),
        pool.func("cos", vec![t]),
        pool.mul(vec![
            pool.func("exp", vec![pool.mul(vec![pool.integer(2_i32), t])]),
            pool.func("sin", vec![pool.mul(vec![pool.integer(3_i32), t])]),
        ]),
        pool.mul(vec![t, pool.func("exp", vec![t])]),
    ];
    let samples = [0.3_f64, 0.7, 1.1, 1.9];
    for f in cases {
        let big_f = laplace_transform(f, t, s, &pool).unwrap();
        let back = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
        assert_numeric_eq(back, f, t, &samples, &pool);
    }
}

#[test]
fn inverse_hyperbolic_poles() {
    // Irreducible over ℚ with ω² < 0 must use sinh/cosh — not sin(√(−κ²)).
    let (pool, t, s) = setup();
    // L⁻¹{1/(s² − 2)} = sinh(√2 · t)/√2
    let den = pool.add(vec![pool.pow(s, pool.integer(2_i32)), pool.integer(-2_i32)]);
    let big_f = recip(den, &pool);
    let got = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    assert!(
        pool.display(got).to_string().contains("sinh"),
        "expected sinh, got {}",
        pool.display(got)
    );
    let kappa = (2.0_f64).sqrt();
    for &x in &[0.5_f64, 1.0, 1.5] {
        let va = eval_at(got, t, x, &pool).expect("finite");
        let vb = (kappa * x).sinh() / kappa;
        assert!(
            (va - vb).abs() < 1e-8,
            "1/(s²−2) at t={x}: {va} vs sinh/√2 = {vb}"
        );
    }

    // L⁻¹{s/(s² − 2)} = cosh(√2 · t)
    let big_f = pool.mul(vec![s, recip(den, &pool)]);
    let got = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    assert!(
        pool.display(got).to_string().contains("cosh"),
        "expected cosh, got {}",
        pool.display(got)
    );
    for &x in &[0.5_f64, 1.0, 1.5] {
        let va = eval_at(got, t, x, &pool).expect("finite");
        let vb = (kappa * x).cosh();
        assert!((va - vb).abs() < 1e-8, "s/(s²−2) at t={x}: {va} vs {vb}");
    }
}

#[test]
fn round_trip_sinh_irreducible() {
    // L{sinh(√2 · t)} →  √2/(s²−2); inverse must recover sinh (not NaN sin).
    let (pool, t, s) = setup();
    let sqrt2 = pool.pow(pool.integer(2_i32), pool.rational(1_i32, 2_i32));
    let f = pool.func("sinh", vec![pool.mul(vec![sqrt2, t])]);
    let big_f = laplace_transform(f, t, s, &pool).unwrap();
    let back = inverse_laplace_transform(big_f, s, t, &pool).unwrap();
    assert_numeric_eq(back, f, t, &[0.25, 0.5, 1.0, 1.5], &pool);
}

#[test]
fn decline_negative_heaviside_shift() {
    let (pool, t, s) = setup();
    // θ(t + 1) = θ(t − (−1)) — unilateral table requires a ≥ 0.
    let arg = pool.add(vec![t, pool.integer(1_i32)]);
    let f = pool.func("heaviside", vec![arg]);
    let err = laplace_transform(f, t, s, &pool).unwrap_err();
    assert!(matches!(err, LaplaceError::NoRule(_)), "{err}");
}

#[test]
fn decline_negative_dirac_shift() {
    let (pool, t, s) = setup();
    let arg = pool.add(vec![t, pool.integer(2_i32)]);
    let f = pool.func("diracdelta", vec![arg]);
    let err = laplace_transform(f, t, s, &pool).unwrap_err();
    assert!(matches!(err, LaplaceError::NoRule(_)), "{err}");
}

// ===========================================================================
// Symbolic parameters — inverse over ℚ(params)
// ===========================================================================
//
// Each of these checks the *value* of the answer against the closed form,
// numerically, at concrete parameter points; a partial-fraction identity that
// only holds symbolically is exactly the failure this module must not ship.
// Each also pins the hypotheses the answer rests on, because an answer whose
// validity silently depends on `ka ≠ ke` is the same defect wearing a hat.

mod symbolic_params {
    use super::*;
    use crate::deriv::SideCondition;
    use crate::simplify::assumptions::AssumptionContext;
    use crate::transform::inverse_laplace_transform_with_assumptions;
    use crate::transform::inverse_laplace_transform_with_conditions;
    use std::collections::HashMap;

    /// Evaluate an expression with several symbols bound.
    fn eval_env(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
        match pool.get(expr) {
            ExprData::Integer(n) => Some(n.0.to_f64()),
            ExprData::Rational(r) => {
                let (n, d) = r.0.clone().into_numer_denom();
                Some(n.to_f64() / d.to_f64())
            }
            ExprData::Float(f) => Some(f.inner.to_f64()),
            ExprData::Symbol { .. } => env.get(&expr).copied(),
            ExprData::Add(args) => args
                .iter()
                .try_fold(0.0, |acc, &a| Some(acc + eval_env(a, env, pool)?)),
            ExprData::Mul(args) => args
                .iter()
                .try_fold(1.0, |acc, &a| Some(acc * eval_env(a, env, pool)?)),
            ExprData::Pow { base, exp } => {
                Some(eval_env(base, env, pool)?.powf(eval_env(exp, env, pool)?))
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let x = eval_env(args[0], env, pool)?;
                Some(match name.as_str() {
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    "sinh" => x.sinh(),
                    "cosh" => x.cosh(),
                    "exp" => x.exp(),
                    "heaviside" => {
                        if x > 0.0 {
                            1.0
                        } else if x < 0.0 {
                            0.0
                        } else {
                            0.5
                        }
                    }
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    fn conds(cs: &[SideCondition], pool: &ExprPool) -> Vec<String> {
        cs.iter()
            .map(|c| c.display_with(pool).to_string())
            .collect()
    }

    fn sym(pool: &ExprPool, n: &str) -> ExprId {
        pool.symbol(n, Domain::Real)
    }

    /// `L⁻¹{1/(s² + ω²)} = sin(ωt)/ω`, for every real `ω ≠ 0`.
    #[test]
    fn undamped_oscillator_with_a_symbolic_frequency() {
        let (pool, t, s) = setup();
        let w = sym(&pool, "w");
        let den = pool.add(vec![
            pool.pow(s, pool.integer(2_i32)),
            pool.pow(w, pool.integer(2_i32)),
        ]);
        let (got, cs) =
            inverse_laplace_transform_with_conditions(recip(den, &pool), s, t, &pool).unwrap();

        for (wv, tv) in [(2.0_f64, 0.7_f64), (0.4, 3.1), (-1.5, 1.2)] {
            let env = HashMap::from([(w, wv), (t, tv)]);
            let got_v = eval_env(got, &env, &pool).expect("finite");
            let want = (wv * tv).sin() / wv;
            assert!(
                (got_v - want).abs() < 1e-9,
                "ω={wv}, t={tv}: {got_v} vs sin(ωt)/ω = {want} — {}",
                pool.display(got)
            );
        }
        // ω = 0 is a different function (`t`), so it must be excluded.
        assert_eq!(conds(&cs, &pool), vec!["w ≠ 0".to_string()]);
    }

    /// The second-order step response — the single most common computation in
    /// classical control.
    ///
    /// `L⁻¹{K/(s² + 2ζωs + ω²)} = K·e^{−ζωt}·sin(ω√(1−ζ²)·t)/(ω√(1−ζ²))`.
    #[test]
    fn second_order_underdamped_response() {
        let (pool, t, s) = setup();
        let (k, w, zeta) = (sym(&pool, "K"), sym(&pool, "w"), sym(&pool, "zeta"));
        let den = pool.add(vec![
            pool.pow(s, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(2_i32), zeta, w, s]),
            pool.pow(w, pool.integer(2_i32)),
        ]);
        let big_f = pool.mul(vec![k, recip(den, &pool)]);
        let (got, cs) = inverse_laplace_transform_with_conditions(big_f, s, t, &pool).unwrap();

        for (kv, wv, zv, tv) in [
            (1.0_f64, 2.0_f64, 0.3_f64, 0.8_f64),
            (2.5, 5.0, 0.1, 1.7),
            (0.75, 1.25, 0.7, 3.0),
        ] {
            let env = HashMap::from([(k, kv), (w, wv), (zeta, zv), (t, tv)]);
            let got_v = eval_env(got, &env, &pool).expect("finite");
            let wd = wv * (1.0 - zv * zv).sqrt();
            let want = kv * (-zv * wv * tv).exp() * (wd * tv).sin() / wd;
            assert!(
                (got_v - want).abs() < 1e-9 * (1.0 + want.abs()),
                "K={kv} ω={wv} ζ={zv} t={tv}: {got_v} vs {want} — {}",
                pool.display(got)
            );
        }

        // The hypotheses are the whole point: ω ≠ 0 (else there is no pole
        // pair), ζ ≠ ±1 (critical damping is `t·e^{−ωt}`, a different form),
        // and ω²(1−ζ²) > 0 — the under-damped condition |ζ| < 1 without which
        // the printed `sin` is a `sinh` in disguise.
        let strings = conds(&cs, &pool);
        assert!(strings.iter().any(|c| c == "w ≠ 0"), "{strings:?}");
        assert_eq!(
            strings.iter().filter(|c| c.ends_with("≠ 0")).count(),
            3,
            "expected ω ≠ 0 and ζ ≠ ±1: {strings:?}"
        );
        assert_eq!(
            strings.iter().filter(|c| c.ends_with("> 0")).count(),
            1,
            "expected the under-damped positivity condition: {strings:?}"
        );
    }

    /// The Bateman function — the single most common computation in
    /// pharmacokinetics — and the coincidence it rests on.
    #[test]
    fn bateman_two_compartment() {
        let (pool, t, s) = setup();
        let (d, ka, ke) = (sym(&pool, "D"), sym(&pool, "ka"), sym(&pool, "ke"));
        let den = pool.mul(vec![pool.add(vec![s, ka]), pool.add(vec![s, ke])]);
        let big_f = pool.mul(vec![d, ka, recip(den, &pool)]);
        let (got, cs) = inverse_laplace_transform_with_conditions(big_f, s, t, &pool).unwrap();

        for (dv, kav, kev, tv) in [
            (100.0_f64, 1.5_f64, 0.25_f64, 2.0_f64),
            (50.0, 0.4, 2.2, 0.7),
            (1.0, 3.0, 0.1, 5.5),
        ] {
            let env = HashMap::from([(d, dv), (ka, kav), (ke, kev), (t, tv)]);
            let got_v = eval_env(got, &env, &pool).expect("finite");
            let want = dv * kav * ((-kav * tv).exp() - (-kev * tv).exp()) / (kev - kav);
            assert!(
                (got_v - want).abs() < 1e-9 * (1.0 + want.abs()),
                "D={dv} ka={kav} ke={kev} t={tv}: {got_v} vs Bateman {want} — {}",
                pool.display(got)
            );
        }

        // At ka = ke the answer above is 0/0 and the true inverse is
        // `D·ka·t·e^{−ka t}`. Exactly one hypothesis, and it is that one.
        let strings = conds(&cs, &pool);
        assert_eq!(strings.len(), 1, "{strings:?}");
        assert!(
            strings[0].contains("ka") && strings[0].contains("ke") && strings[0].ends_with("≠ 0"),
            "{strings:?}"
        );
    }

    /// The *repeated* symbolic pole is a genuinely different answer, and it is
    /// produced without any hypothesis — nothing was divided by.
    #[test]
    fn repeated_symbolic_pole_is_the_other_branch() {
        let (pool, t, s) = setup();
        let ka = sym(&pool, "ka");
        let big_f = pool.pow(pool.add(vec![s, ka]), pool.integer(-2_i32));
        let (got, cs) = inverse_laplace_transform_with_conditions(big_f, s, t, &pool).unwrap();
        assert!(cs.is_empty(), "{:?}", conds(&cs, &pool));
        for (kav, tv) in [(1.5_f64, 2.0_f64), (0.25, 4.0)] {
            let env = HashMap::from([(ka, kav), (t, tv)]);
            let got_v = eval_env(got, &env, &pool).expect("finite");
            let want = tv * (-kav * tv).exp();
            assert!(
                (got_v - want).abs() < 1e-9,
                "{got_v} vs t·e^{{−ka t}} {want}"
            );
        }
    }

    /// A non-monic parametric denominator divides by its leading coefficient,
    /// and says so: at `a = 0` the input is a constant, whose inverse is a δ.
    #[test]
    fn non_monic_pole_reports_its_leading_coefficient() {
        let (pool, t, s) = setup();
        let (a, b) = (sym(&pool, "a"), sym(&pool, "b"));
        let big_f = recip(pool.add(vec![pool.mul(vec![a, s]), b]), &pool);
        let (got, cs) = inverse_laplace_transform_with_conditions(big_f, s, t, &pool).unwrap();
        assert_eq!(conds(&cs, &pool), vec!["a ≠ 0".to_string()]);
        for (av, bv, tv) in [(2.0_f64, 3.0_f64, 1.0_f64), (0.5, -1.0, 2.5)] {
            let env = HashMap::from([(a, av), (b, bv), (t, tv)]);
            let got_v = eval_env(got, &env, &pool).expect("finite");
            let want = (-bv / av * tv).exp() / av;
            assert!((got_v - want).abs() < 1e-9, "{got_v} vs {want}");
        }
    }

    /// A symbolic delay is a hypothesis: the unilateral shift rule needs
    /// `a ≥ 0`.
    #[test]
    fn symbolic_delay_reports_its_sign_hypothesis() {
        let (pool, t, s) = setup();
        let a = sym(&pool, "a");
        let big_f = pool.mul(vec![
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), a, s])]),
            recip(s, &pool),
        ]);
        let (_got, cs) = inverse_laplace_transform_with_conditions(big_f, s, t, &pool).unwrap();
        assert!(
            cs.iter()
                .any(|c| matches!(c, SideCondition::InDomain(id, Domain::NonNegative) if *id == a)),
            "{:?}",
            conds(&cs, &pool)
        );
    }

    /// A `Domain::Positive` symbol carries the fact with it, so the shift
    /// hypothesis is discharged rather than reported.
    #[test]
    fn a_positive_delay_symbol_discharges_the_shift_hypothesis() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let s = pool.symbol("s", Domain::Real);
        let a = pool.symbol("a", Domain::Positive);
        let big_f = pool.mul(vec![
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), a, s])]),
            recip(s, &pool),
        ]);
        let (_got, cs) = inverse_laplace_transform_with_conditions(big_f, s, t, &pool).unwrap();
        assert!(cs.is_empty(), "{:?}", conds(&cs, &pool));
    }

    /// Assumptions discharge rather than decorate: telling the engine
    /// `ka ≠ ke` removes the hypothesis without changing the answer.
    #[test]
    fn an_assumption_discharges_the_bateman_hypothesis() {
        let (pool, t, s) = setup();
        let (ka, ke) = (sym(&pool, "ka"), sym(&pool, "ke"));
        let den = pool.mul(vec![pool.add(vec![s, ka]), pool.add(vec![s, ke])]);
        let big_f = recip(den, &pool);

        let (plain, plain_cs) =
            inverse_laplace_transform_with_conditions(big_f, s, t, &pool).unwrap();
        assert_eq!(plain_cs.len(), 1);

        // The condition is reported on `ka − ke` (canonically signed); state
        // exactly that fact.
        let target = match &plain_cs[0] {
            SideCondition::NonZero(id) => *id,
            other => panic!("unexpected condition {other:?}"),
        };
        let mut ctx = AssumptionContext::new();
        ctx.refine(
            pool.predicate(
                crate::kernel::expr::PredicateKind::Ne,
                vec![target, pool.integer(0_i32)],
            ),
            &pool,
        )
        .unwrap();

        let (with, with_cs) =
            inverse_laplace_transform_with_assumptions(big_f, s, t, &pool, &ctx).unwrap();
        assert!(with_cs.is_empty(), "{:?}", conds(&with_cs, &pool));
        assert_eq!(with, plain, "the answer must not depend on the context");
    }

    /// The legacy signature keeps working and routes its hypotheses to the
    /// out-of-band channel — a consuming one, so a later read cannot be
    /// mistaken for this call's.
    #[test]
    fn the_legacy_signature_reports_out_of_band() {
        let (pool, t, s) = setup();
        let (ka, ke) = (sym(&pool, "ka"), sym(&pool, "ke"));
        let den = pool.mul(vec![pool.add(vec![s, ka]), pool.add(vec![s, ke])]);
        let _ = inverse_laplace_transform(recip(den, &pool), s, t, &pool).unwrap();
        let first = crate::transform::take_transform_side_conditions();
        assert_eq!(first.len(), 1, "{:?}", conds(&first, &pool));
        assert!(crate::transform::take_transform_side_conditions().is_empty());

        // A ℚ-coefficient inverse assumes nothing.
        let den2 = pool.mul(vec![
            pool.add(vec![s, pool.integer(1_i32)]),
            pool.add(vec![s, pool.integer(2_i32)]),
        ]);
        let _ = inverse_laplace_transform(recip(den2, &pool), s, t, &pool).unwrap();
        assert!(crate::transform::take_transform_side_conditions().is_empty());
    }

    /// A *failed* call must clear the channel, not leave the previous call's
    /// hypotheses on it for a caller to misread as its own.
    #[test]
    fn a_failed_call_does_not_leave_stale_hypotheses() {
        let (pool, t, s) = setup();
        let (ka, ke) = (sym(&pool, "ka"), sym(&pool, "ke"));
        let den = pool.mul(vec![pool.add(vec![s, ka]), pool.add(vec![s, ke])]);
        let _ = inverse_laplace_transform(recip(den, &pool), s, t, &pool).unwrap();

        // An improper rational: the polynomial part is a δ, which is declined.
        let bad = pool.mul(vec![
            pool.pow(s, pool.integer(3_i32)),
            recip(pool.add(vec![s, ka]), &pool),
        ]);
        assert!(inverse_laplace_transform(bad, s, t, &pool).is_err());
        assert!(
            crate::transform::take_transform_side_conditions().is_empty(),
            "a failed call must not report the previous call's hypotheses"
        );
    }

    /// Round trip: forward `L{e^{−at}·sin(bt)}`, then invert, and land back on
    /// the same function of `t` at concrete parameter values.
    #[test]
    fn round_trip_damped_sinusoid_with_symbolic_parameters() {
        let (pool, t, s) = setup();
        let (a, b) = (sym(&pool, "a"), sym(&pool, "b"));
        let f = pool.mul(vec![
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), a, t])]),
            pool.func("sin", vec![pool.mul(vec![b, t])]),
        ]);
        let big_f = laplace_transform(f, t, s, &pool).unwrap();
        let (back, _cs) = inverse_laplace_transform_with_conditions(big_f, s, t, &pool).unwrap();
        for (av, bv, tv) in [
            (0.5_f64, 3.0_f64, 1.1_f64),
            (2.0, 0.75, 0.4),
            (0.1, 1.0, 4.0),
        ] {
            let env = HashMap::from([(a, av), (b, bv), (t, tv)]);
            let lhs = eval_env(f, &env, &pool).expect("finite");
            let rhs = eval_env(back, &env, &pool).expect("finite");
            assert!(
                (lhs - rhs).abs() < 1e-9 * (1.0 + lhs.abs()),
                "a={av} b={bv} t={tv}: {lhs} vs {rhs} — {}",
                pool.display(back)
            );
        }
    }
    /// How far assumption discharge actually reaches — pinned, because the
    /// answer is "not as far as you might hope" and a caller needs to know.
    ///
    /// `0 < ζ < 1 ∧ ω > 0` entails all three second-order hypotheses. Only
    /// `ω ≠ 0` is discharged: the other two are nonlinear consequences, and the
    /// interval solver behind [`crate::logic::satisfiable`] answers `Unknown`
    /// rather than proving them. `Unknown` is not a proof, so they stay on the
    /// wire. Over-reporting, never under-reporting.
    #[test]
    fn assumption_discharge_stops_at_nonlinear_consequences() {
        let (pool, t, s) = setup();
        let (w, zeta) = (sym(&pool, "w"), sym(&pool, "zeta"));
        let den = pool.add(vec![
            pool.pow(s, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(2_i32), zeta, w, s]),
            pool.pow(w, pool.integer(2_i32)),
        ]);
        let big_f = recip(den, &pool);

        let zero = pool.integer(0_i32);
        let one = pool.integer(1_i32);
        let mut ctx = AssumptionContext::new();
        for p in [
            pool.predicate(crate::kernel::expr::PredicateKind::Gt, vec![zeta, zero]),
            pool.predicate(crate::kernel::expr::PredicateKind::Lt, vec![zeta, one]),
            pool.predicate(crate::kernel::expr::PredicateKind::Gt, vec![w, zero]),
        ] {
            ctx.refine(p, &pool).unwrap();
        }

        let (_out, cs) =
            inverse_laplace_transform_with_assumptions(big_f, s, t, &pool, &ctx).unwrap();
        let rendered = conds(&cs, &pool);
        assert!(
            !rendered.iter().any(|c| c == "w ≠ 0"),
            "ω > 0 should discharge ω ≠ 0: {rendered:?}"
        );
        assert_eq!(
            rendered.len(),
            3,
            "ζ ≠ ±1 and the positivity are nonlinear and stay: {rendered:?}"
        );

        // A `Domain::Positive` symbol carries the same fact without a context.
        let p2 = ExprPool::new();
        let s2 = p2.symbol("s", Domain::Real);
        let t2 = p2.symbol("t", Domain::Real);
        let w2 = p2.symbol("w", Domain::Positive);
        let den2 = p2.add(vec![
            p2.pow(s2, p2.integer(2_i32)),
            p2.pow(w2, p2.integer(2_i32)),
        ]);
        let (_o, cs2) =
            inverse_laplace_transform_with_conditions(recip(den2, &p2), s2, t2, &p2).unwrap();
        assert!(cs2.is_empty(), "{:?}", conds(&cs2, &p2));
    }
}
