//! Every evaluator must agree on the same expression at the same point.
//!
//! Alkahest has several independent ways to turn an expression into a number —
//! the tree-walking interpreter behind `eval_expr`, the separate
//! [`crate::eval::eval_f64`] facade behind `evaluate(mode="f64")`, the
//! snapshot interpreter that `compile` falls back to, and the Cranelift JIT.
//! Verification gates are built on those evaluators, so a disagreement is not
//! a cosmetic difference: it means at least one gate is scoring answers with a
//! different arithmetic than the one that produced them.
//!
//! Every case here is a *bit-exact* comparison. Anything that legitimately
//! differs (a transcendental evaluated by two different libm implementations)
//! is deliberately not in the corpus; the shared numeric kernels make the
//! comparison exact for everything that is.

use crate::eval::eval_f64;
#[cfg(feature = "cranelift")]
use crate::jit::compile_jit_only;
use crate::jit::{compile, eval_interp};
use crate::kernel::{Domain, ExprId, ExprPool};
use rug::ops::Pow;
use std::collections::HashMap;

/// Every expression in the corpus, built against a fresh pool.
fn corpus(pool: &ExprPool, x: ExprId) -> Vec<(&'static str, ExprId)> {
    let int = |n: i32| pool.integer(n);
    let x2 = pool.pow(x, int(2));
    let x3 = pool.pow(x, int(3));
    let mut out: Vec<(&'static str, ExprId)> = vec![
        ("x", x),
        ("integer", int(-7)),
        ("rational_2_5", pool.rational(2, 5)),
        ("rational_neg_7_3", pool.rational(-7, 3)),
        ("rational_355_113", pool.rational(355, 113)),
        ("float", pool.float(1.234_567_890_123_45, 53)),
        ("x_squared", x2),
        ("x_cubed", x3),
        ("x_pow_neg1", pool.pow(x, int(-1))),
        ("x_pow_neg2", pool.pow(x, int(-2))),
        ("x_pow_zero", pool.pow(x, int(0))),
        ("x_pow_half", pool.pow(x, pool.rational(1, 2))),
        ("x_pow_x", pool.pow(x, x)),
    ];

    out.push((
        "poly_with_rational_coeffs",
        pool.add(vec![
            pool.mul(vec![pool.rational(2, 5), x3]),
            pool.mul(vec![pool.rational(-7, 3), x2]),
            pool.mul(vec![pool.rational(355, 113), x]),
            pool.rational(1, 7),
        ]),
    ));
    out.push((
        "big_integer_coefficient",
        pool.add(vec![
            pool.mul(vec![pool.integer(rug::Integer::from(10).pow(30)), x]),
            int(1),
        ]),
    ));
    out.push((
        "big_rational_coefficient",
        pool.mul(vec![
            pool.rational(
                rug::Integer::from(3) * rug::Integer::from(10).pow(400) + 1u32,
                rug::Integer::from(2) * rug::Integer::from(10).pow(400),
            ),
            x,
        ]),
    ));

    // The five heads `eval::eval_f64` implements, so the comparison covers it.
    for name in ["sin", "cos", "exp", "log", "sqrt"] {
        out.push((name, pool.func(name, vec![x])));
    }
    let sin_x = pool.func("sin", vec![x]);
    let cos_x = pool.func("cos", vec![x]);
    out.push((
        "nested_transcendental",
        pool.func("exp", vec![pool.mul(vec![sin_x, cos_x])]),
    ));
    out.push((
        "deep_sum",
        pool.add(vec![sin_x, cos_x, x2, x3, pool.rational(1, 3)]),
    ));
    out
}

/// Points chosen to include negatives, near-zero, exact halves and large
/// magnitudes; the domain-refusal cases are handled by the `is_finite` filter.
const POINTS: [f64; 15] = [
    -100.0, -8.0, -2.5, -1.0, -0.5, -1e-8, 0.0, 1e-8, 0.25, 0.5, 1.0, 2.0, 3.5, 100.0, 1e6,
];

#[test]
fn interpreter_and_eval_f64_facade_agree_bit_for_bit() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let mut compared = 0usize;
    for (name, expr) in corpus(&pool, x) {
        for &v in &POINTS {
            let mut env = HashMap::new();
            env.insert(x, v);
            let interp = eval_interp(expr, &env, &pool);
            let facade = eval_f64(expr, &pool, &env).ok();
            match (interp, facade) {
                // `eval_f64` rejects non-finite results; `eval_interp` returns
                // them. That asymmetry is deliberate, so only compare numbers
                // both produced.
                (Some(a), Some(b)) => {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{name} at x = {v}: eval_interp {a:?} vs eval::eval_f64 {b:?}"
                    );
                    compared += 1;
                }
                (Some(a), None) => assert!(
                    !a.is_finite(),
                    "{name} at x = {v}: eval_interp gave {a:?} but eval::eval_f64 refused"
                ),
                _ => {}
            }
        }
    }
    assert!(compared > 150, "corpus shrank to {compared} comparisons");
}

#[test]
fn interpreter_and_compiled_fn_agree_bit_for_bit() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    for (name, expr) in corpus(&pool, x) {
        let f = compile(expr, &[x], &pool).unwrap();
        for &v in &POINTS {
            let mut env = HashMap::new();
            env.insert(x, v);
            let Some(interp) = eval_interp(expr, &env, &pool) else {
                continue;
            };
            let compiled = f.call(&[v]);
            assert!(
                interp.to_bits() == compiled.to_bits() || (interp.is_nan() && compiled.is_nan()),
                "{name} at x = {v}: eval_interp {interp:?} vs compiled {compiled:?}"
            );
        }
    }
}

#[cfg(feature = "cranelift")]
#[test]
fn cranelift_and_interpreter_agree_bit_for_bit() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let mut jitted = 0usize;
    for (name, expr) in corpus(&pool, x) {
        // Cranelift refuses heads it has no trampoline for; those fall back to
        // the interpreter in `compile`, and there is nothing to compare.
        let Ok(f) = compile_jit_only(expr, &[x], &pool) else {
            continue;
        };
        jitted += 1;
        for &v in &POINTS {
            let mut env = HashMap::new();
            env.insert(x, v);
            let Some(interp) = eval_interp(expr, &env, &pool) else {
                continue;
            };
            let native = f.call(&[v]);
            assert!(
                interp.to_bits() == native.to_bits() || (interp.is_nan() && native.is_nan()),
                "{name} at x = {v}: eval_interp {interp:?} vs Cranelift {native:?}"
            );
        }
    }
    assert!(jitted > 10, "only {jitted} expressions reached Cranelift");
}

#[cfg(feature = "cranelift")]
#[test]
fn cranelift_bulk_entry_point_matches_its_own_scalar_entry_point() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let expr = pool.add(vec![
        pool.func("sin", vec![x]),
        pool.mul(vec![pool.rational(2, 5), pool.pow(y, pool.integer(2_i32))]),
        pool.mul(vec![x, y]),
    ]);
    let f = compile_jit_only(expr, &[x, y], &pool).unwrap();

    // `call_bulk` takes the variable-major layout `numpy_eval` uses.
    let xs: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) / 4.0).collect();
    let ys: Vec<f64> = (0..64).map(|i| (i as f64 - 17.0) / 3.0).collect();
    let mut flat = xs.clone();
    flat.extend_from_slice(&ys);
    let mut out = vec![0.0f64; 64];
    f.call_bulk(&flat, &mut out);

    for i in 0..64 {
        let scalar = f.call(&[xs[i], ys[i]]);
        assert_eq!(
            out[i].to_bits(),
            scalar.to_bits(),
            "bulk[{i}] {:?} vs scalar {scalar:?}",
            out[i]
        );
    }
}
