//! Batch evaluation, multi-output compiles and compile-time refusals (G7).
//!
//! The native batch path, its Rayon split, the shared power kernel and the
//! generic-primitive trampolines must all give exactly what the interpreter
//! gives; and an expression no evaluator can ever give a number for must be
//! refused when it is compiled, not turned into `NaN` at every point.

use super::*;
use crate::kernel::eval_const::{pow_f64_narrow, powi_with, F64Ops};
use crate::kernel::Domain;

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn grid(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * (i as f64 + 0.37) / n as f64)
        .collect()
}

// ---------------------------------------------------------------------------
// The power kernel
// ---------------------------------------------------------------------------

/// `x^n` rounded correctly, from MPFR at 200 bits.
fn powi_reference(x: f64, n: i32) -> f64 {
    use rug::ops::Pow;
    let big = rug::Float::with_val(200, x);
    let r: rug::Float = big.pow(n);
    r.to_f64()
}

fn ulps(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    let ia = a.to_bits() as i64;
    let ib = b.to_bits() as i64;
    let key = |i: i64| if i < 0 { i64::MIN - i } else { i };
    key(ia).abs_diff(key(ib))
}

#[test]
fn compensated_integer_power_is_within_one_ulp() {
    let mut worst = 0;
    // Deterministic pseudo-random bases across many binades, both signs.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..4000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let mant = 1.0 + (state >> 12) as f64 / (1u64 << 52) as f64;
        let exp = ((state >> 3) % 40) as i32 - 20;
        let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
        let x = sign * mant * 2f64.powi(exp);
        for n in -64..=64 {
            let got = powi_with(&mut F64Ops, x, n);
            let want = powi_reference(x, n);
            if !want.is_finite() || want == 0.0 || want.abs() < f64::MIN_POSITIVE {
                continue; // overflow / underflow: checked separately
            }
            let u = ulps(got, want);
            assert!(u <= 1, "{x:e}^{n}: got {got:e}, want {want:e} ({u} ulps)");
            worst = worst.max(u);
        }
    }
    assert!(worst <= 1);
}

/// The kernel's `fma` must be fused on every platform (MinGW's C `fma` is
/// not): `(1 + 2⁻³⁰)(1 − 2⁻³⁰) − 1` is `−2⁻⁶⁰` fused and `0` unfused.
#[test]
fn kernel_fma_is_fused_everywhere() {
    let e = 2f64.powi(-30);
    let want = -(2f64.powi(-60));
    assert_eq!(
        crate::kernel::eval_const::fma_exact(1.0 + e, 1.0 - e, -1.0),
        want
    );
    assert_eq!(libm::fma(1.0 + e, 1.0 - e, -1.0), want);
}

#[test]
fn integer_power_special_values_match_pow() {
    let specials = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0,
        -1.0,
        1e300,
        -1e300,
        1e-300,
        -1e-300,
        5e-324,
        -5e-324,
        f64::MAX,
        f64::MIN_POSITIVE,
    ];
    for &x in &specials {
        for n in -64..=64 {
            let got = powi_with(&mut F64Ops, x, n);
            let want = x.powf(n as f64);
            if want.is_nan() || !want.is_finite() || want == 0.0 {
                // NaN, infinities and zeros (with their sign) exactly as pow.
                assert!(
                    same(got, want),
                    "{x:e}^{n}: got {got:e}, pow gives {want:e}"
                );
            } else {
                assert!(
                    ulps(got, want) <= 2,
                    "{x:e}^{n}: got {got:e}, pow gives {want:e}"
                );
            }
        }
    }
}

#[test]
fn square_root_power_is_sqrt_with_pow_edge_values() {
    for &x in &[
        0.0,
        -0.0,
        2.0,
        0.25,
        1e-310,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -1.0,
        f64::NAN,
    ] {
        let got = pow_f64_narrow(x, 0.5);
        let want = x.powf(0.5);
        assert!(
            same(got, want),
            "{x:e}^(1/2): got {got:e}, pow gives {want:e}"
        );
    }
    // Correctly rounded everywhere else.
    for x in grid(1000, 1e-3, 1e3) {
        assert_eq!(pow_f64_narrow(x, 0.5), x.sqrt());
    }
}

// ---------------------------------------------------------------------------
// Native batch path ≡ interpreter
// ---------------------------------------------------------------------------

/// Expressions covering every code path of the batch kernel: integer powers
/// (compensated), `^(1/2)`, a run-time exponent, libm trampolines, generic
/// one- and two-argument primitives, and a primitive that fails at some
/// points (`lambert_w` below `-1/e`).
fn corpus(pool: &ExprPool, x: ExprId, y: ExprId) -> Vec<(&'static str, ExprId)> {
    let int = |n: i32| pool.integer(n);
    let r2 = pool.add(vec![pool.pow(x, int(2)), pool.pow(y, int(2))]);
    vec![
        ("poly10", {
            let terms = (0..=10)
                .map(|k| pool.mul(vec![int(k + 1), pool.pow(x, int(k))]))
                .collect();
            pool.add(terms)
        }),
        (
            "lennard_jones",
            pool.mul(vec![
                int(4),
                pool.add(vec![
                    pool.pow(r2, int(-6)),
                    pool.mul(vec![int(-1), pool.pow(r2, int(-3))]),
                ]),
            ]),
        ),
        ("sqrt_power", pool.pow(r2, pool.rational(1, 2))),
        ("neg_half_power", pool.pow(r2, pool.rational(-1, 2))),
        ("runtime_exponent", pool.pow(pool.func("abs", vec![x]), y)),
        (
            "transcendental",
            pool.mul(vec![
                pool.func("sin", vec![x]),
                pool.func("exp", vec![pool.mul(vec![int(-1), pool.pow(x, int(2))])]),
            ]),
        ),
        ("erf_generic", pool.func("erf", vec![pool.mul(vec![x, y])])),
        ("atan2_generic", pool.func("atan2", vec![y, x])),
        (
            "lambert_w_fails_below_minus_inv_e",
            pool.add(vec![pool.func("lambert_w", vec![x]), int(1)]),
        ),
    ]
}

#[cfg(feature = "cranelift")]
#[test]
fn native_batch_matches_interpreter_bit_for_bit() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let xs = grid(3000, -3.0, 3.0);
    let ys = grid(3000, -2.0, 2.5);
    for (name, expr) in corpus(&pool, x, y) {
        let native = compile_with(
            expr,
            &[x, y],
            &pool,
            CompileConfig {
                expected_evals: None,
                force_tier: Some(CompileTier::Cranelift),
            },
        )
        .unwrap();
        assert_eq!(native.compile_tier(), CompileTier::Cranelift, "{name}");
        let interp = compile_with(
            expr,
            &[x, y],
            &pool,
            CompileConfig {
                expected_evals: None,
                force_tier: Some(CompileTier::Interpreter),
            },
        )
        .unwrap();
        let mut a = vec![0.0; xs.len()];
        let mut b = vec![0.0; xs.len()];
        native.call_batch(&[&xs, &ys], &mut a);
        interp.call_batch(&[&xs, &ys], &mut b);
        for i in 0..xs.len() {
            assert!(
                same(a[i], b[i]),
                "{name} at ({}, {}): native {:?} vs interpreter {:?}",
                xs[i],
                ys[i],
                a[i],
                b[i]
            );
            let scalar = native.call(&[xs[i], ys[i]]);
            assert!(same(scalar, a[i]), "{name}: scalar vs batch at {i}");
        }
        // The failing primitive must actually fail somewhere in the grid.
        if name.starts_with("lambert_w") {
            assert!(a.iter().any(|v| v.is_nan()) && a.iter().any(|v| v.is_finite()));
        }
    }
}

/// The LLVM tier computes powers with the same shared kernel, so it agrees
/// with the interpreter exactly too. (Heads LLVM does not lower fall back to
/// another tier, which must agree as well.)
#[cfg(feature = "jit")]
#[test]
fn llvm_batch_matches_interpreter_bit_for_bit() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let xs = grid(2000, -3.0, 3.0);
    let ys = grid(2000, -2.0, 2.5);
    for (name, expr) in corpus(&pool, x, y) {
        let cfg = |tier| CompileConfig {
            expected_evals: None,
            force_tier: Some(tier),
        };
        let llvm = compile_with(expr, &[x, y], &pool, cfg(CompileTier::Llvm)).unwrap();
        let interp = compile_with(expr, &[x, y], &pool, cfg(CompileTier::Interpreter)).unwrap();
        let mut a = vec![0.0; xs.len()];
        let mut b = vec![0.0; xs.len()];
        llvm.call_batch(&[&xs, &ys], &mut a);
        interp.call_batch(&[&xs, &ys], &mut b);
        for i in 0..xs.len() {
            assert!(
                same(a[i], b[i]),
                "{name} ({:?}) at ({}, {}): {:?} vs interpreter {:?}",
                llvm.compile_tier(),
                xs[i],
                ys[i],
                a[i],
                b[i]
            );
        }
    }
}

#[cfg(all(feature = "cranelift", feature = "parallel"))]
#[test]
fn parallel_batch_matches_sequential() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    // Not a multiple of the chunk size, so the last chunk is partial.
    let n = 5 * PAR_CHUNK + 123;
    let xs = grid(n, -3.0, 3.0);
    let ys = grid(n, 0.1, 2.0);
    for (name, expr) in corpus(&pool, x, y) {
        for tier in [CompileTier::Cranelift, CompileTier::Interpreter] {
            let f = compile_with(
                expr,
                &[x, y],
                &pool,
                CompileConfig {
                    expected_evals: None,
                    force_tier: Some(tier),
                },
            )
            .unwrap();
            let mut seq = vec![0.0; n];
            let mut par = vec![0.0; n];
            f.call_batch(&[&xs, &ys], &mut seq);
            f.call_batch_par(&[&xs, &ys], &mut par);
            for i in 0..n {
                assert!(same(seq[i], par[i]), "{name} ({tier:?}) at {i}");
            }
        }
    }
}

#[cfg(feature = "cranelift")]
#[test]
fn call_bulk_uses_the_column_entry() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let expr = pool.add(vec![
        pool.func("sin", vec![x]),
        pool.mul(vec![pool.rational(2, 5), pool.pow(y, pool.integer(3))]),
    ]);
    let f = compile_jit_only(expr, &[x, y], &pool).unwrap();
    let xs = grid(97, -4.0, 4.0);
    let ys = grid(97, -2.0, 3.0);
    let mut flat = xs.clone();
    flat.extend_from_slice(&ys);
    let mut out = vec![0.0; 97];
    f.call_bulk(&flat, &mut out);
    for i in 0..97 {
        assert!(same(out[i], f.call(&[xs[i], ys[i]])));
    }
}

// ---------------------------------------------------------------------------
// compile_many
// ---------------------------------------------------------------------------

#[test]
fn compile_many_matches_each_output_compiled_alone() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let exprs: Vec<ExprId> = corpus(&pool, x, y).into_iter().map(|(_, e)| e).collect();
    let n = 2000;
    let xs = grid(n, -3.0, 3.0);
    let ys = grid(n, -2.0, 2.5);
    for force in [None, Some(CompileTier::Interpreter)] {
        let many = compile_many_with(
            &exprs,
            &[x, y],
            &pool,
            CompileConfig {
                expected_evals: None,
                force_tier: force,
            },
        )
        .unwrap();
        assert_eq!(many.n_inputs(), 2);
        assert_eq!(many.n_outputs(), exprs.len());
        if force.is_none() && cfg!(feature = "cranelift") {
            assert_ne!(many.compile_tier(), CompileTier::Interpreter);
        }
        let mut cols: Vec<Vec<f64>> = vec![vec![0.0; n]; exprs.len()];
        {
            let mut outs: Vec<&mut [f64]> = cols.iter_mut().map(|c| c.as_mut_slice()).collect();
            many.call_batch(&[&xs, &ys], &mut outs);
        }
        #[cfg(feature = "parallel")]
        {
            let mut par: Vec<Vec<f64>> = vec![vec![0.0; n]; exprs.len()];
            let mut outs: Vec<&mut [f64]> = par.iter_mut().map(|c| c.as_mut_slice()).collect();
            many.call_batch_par(&[&xs, &ys], &mut outs);
            for (a, b) in cols.iter().zip(&par) {
                assert!(a.iter().zip(b).all(|(&p, &q)| same(p, q)));
            }
        }
        let mut point = vec![0.0; exprs.len()];
        for (k, &e) in exprs.iter().enumerate() {
            let alone = compile(e, &[x, y], &pool).unwrap();
            for i in (0..n).step_by(7) {
                let want = alone.call(&[xs[i], ys[i]]);
                assert!(
                    same(cols[k][i], want),
                    "output {k} at {i}: many {:?} vs alone {want:?}",
                    cols[k][i]
                );
            }
        }
        many.call(&[0.3, -0.7], &mut point);
        for (k, &e) in exprs.iter().enumerate() {
            let alone = compile(e, &[x, y], &pool).unwrap();
            assert!(same(point[k], alone.call(&[0.3, -0.7])), "point output {k}");
        }
    }
}

#[test]
fn compile_many_failure_is_per_output() {
    // `lambert_w(x)` has no value at x = -1; the other output does.
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let w = pool.func("lambert_w", vec![x]);
    let other = pool.add(vec![x, pool.integer(1)]);
    let f = compile_many(&[w, other], &[x], &pool).unwrap();
    let mut out = [0.0; 2];
    f.call(&[-1.0], &mut out);
    assert!(out[0].is_nan());
    assert_eq!(out[1], 0.0);
}

#[test]
fn compile_many_refuses_like_compile() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    assert!(compile_many(&[x, pool.add(vec![x, y])], &[x], &pool).is_err());
    assert_eq!(take_unbound_symbols().unwrap().names(), ["y"]);
}

// ---------------------------------------------------------------------------
// E-JIT-006: functions with no numeric kernel
// ---------------------------------------------------------------------------

#[test]
fn undefined_function_is_refused_not_nan() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let f_x = pool.func("f", vec![x]);
    let expr = pool.add(vec![f_x, pool.pow(x, pool.integer(2))]);
    for tier in [None, Some(CompileTier::Interpreter)] {
        let err = compile_with(
            expr,
            &[x],
            &pool,
            CompileConfig {
                expected_evals: None,
                force_tier: tier,
            },
        )
        .err()
        .expect("an undefined function must be refused");
        assert!(matches!(err, JitError::UnsupportedNode(_)));
        let refusal = take_unevaluable_functions().expect("E-JIT-006 recorded");
        assert_eq!(refusal.names(), ["f"]);
        assert_eq!(crate::errors::AlkahestError::code(&refusal), "E-JIT-006");
    }
    // A derivative of an undefined function, however it is spelled.
    let d = pool.func("Derivative", vec![f_x, pool.integer(1)]);
    assert!(compile(d, &[x], &pool).is_err());
    let names = take_unevaluable_functions().unwrap().names().to_vec();
    assert!(names.contains(&"Derivative".to_string()), "{names:?}");
    assert!(compile_many(&[x, f_x], &[x], &pool).is_err());
    assert!(take_unevaluable_functions().is_some());
    // The unbound-symbol refusal is unchanged and is not misreported.
    let y = pool.symbol("y", Domain::Real);
    assert!(compile(pool.add(vec![x, y]), &[x], &pool).is_err());
    assert!(take_unevaluable_functions().is_none());
    assert!(take_unbound_symbols().is_some());
    // A registered function compiles as before.
    assert!(compile(pool.func("erf", vec![x]), &[x], &pool).is_ok());
    assert!(take_unevaluable_functions().is_none());
}

/// `RootOf` is not a registry primitive, but every evaluator gives it its
/// value: it is neither refused nor sent off the JIT, and the native code
/// folds it to the interpreter's constant.
#[test]
fn root_of_is_a_constant_not_a_refusal() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let t = pool.symbol("t", Domain::Real);
    let cubic = pool.add(vec![
        pool.pow(x, pool.integer(3)),
        pool.mul(vec![pool.integer(-1), x]),
        pool.integer(-1),
    ]);
    // No root isolation in this FLINT build: nothing to test.
    let Ok(r) = crate::poly::root_of::root_of(cubic, x, 0, &pool) else {
        return;
    };
    let e = pool.add(vec![pool.mul(vec![r, t]), pool.integer(1)]);
    assert!(check_functions_evaluable(e, &pool).is_ok());
    let interp = compile(e, &[t], &pool).unwrap();
    let want = interp.call(&[2.0]);
    assert!((want - 3.649435914489492).abs() < 1e-14);
    let many = compile_many(&[e, r], &[t], &pool).unwrap();
    let mut out = [0.0; 2];
    many.call(&[2.0], &mut out);
    assert_eq!(out[0], want);
    #[cfg(feature = "cranelift")]
    {
        let native = compile_jit_only(e, &[t], &pool).unwrap();
        assert_eq!(native.call(&[2.0]), want);
    }
}
