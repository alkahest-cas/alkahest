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
