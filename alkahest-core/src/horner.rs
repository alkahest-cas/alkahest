//! Phase 24 — Horner-form code emission.
//!
//! Converts a univariate polynomial expression into numerically-stable Horner
//! form `a₀ + x*(a₁ + x*(a₂ + …))` and optionally emits a C function body.
//!
//! # Why Horner?
//!
//! A degree-*n* polynomial in naive pow/mul form requires O(n²) multiplications
//! (because `x^k` costs *k-1* multiplications).  Horner form uses exactly *n*
//! additions and *n* multiplications — the theoretical minimum.  It also
//! reduces floating-point cancellation by avoiding large intermediate powers.

use crate::kernel::{ExprId, ExprPool};
use crate::poly::{ConversionError, UniPoly};

// ---------------------------------------------------------------------------
// horner — symbolic rewrite to Horner form
// ---------------------------------------------------------------------------

/// Convert a polynomial expression into its Horner form.
///
/// `expr` must be expressible as a univariate polynomial in `var` with integer
/// coefficients (same restriction as [`UniPoly::from_symbolic`]).
///
/// Returns `Err` if `expr` is not a polynomial in `var`.
///
/// # Example (symbolic)
/// ```text
/// x² + 2x + 1  →  1 + x*(2 + x*1)
/// ```
pub fn horner(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<ExprId, ConversionError> {
    let poly = UniPoly::from_symbolic(expr, var, pool)?;
    let coeffs = poly.coefficients_i64(); // [a0, a1, …, an]
    Ok(build_horner(&coeffs, var, pool))
}

/// Build Horner form from a coefficient slice `[a0, a1, …, an]` where
/// `a_k` is the coefficient of `x^k`.
fn build_horner(coeffs: &[i64], var: ExprId, pool: &ExprPool) -> ExprId {
    if coeffs.is_empty() {
        return pool.integer(0_i32);
    }
    // Start from the highest-degree coefficient and fold inward:
    // result = a_n
    // result = a_{n-1} + x * result
    // …
    // result = a_0   + x * result
    let n = coeffs.len();
    let mut result = pool.integer(coeffs[n - 1]);
    for k in (0..n - 1).rev() {
        // result = coeffs[k] + var * result
        let xr = pool.mul(vec![var, result]);
        let ck = pool.integer(coeffs[k]);
        result = pool.add(vec![ck, xr]);
    }
    result
}

// ---------------------------------------------------------------------------
// emit_horner_c — C code emitter
// ---------------------------------------------------------------------------

/// Emit a C function body that evaluates the Horner form of `expr`.
///
/// `var_name` is the name of the C variable for the independent variable.
/// `fn_name` is the C function name.
///
/// Returns a complete C function as a `String`.
///
/// # Example output
/// For `x² + 2x + 1` with `var_name = "x"` and `fn_name = "eval_poly"`:
/// ```c
/// double eval_poly(double x) {
///     return 1.0 + x * (2.0 + x * 1.0);
/// }
/// ```
pub fn emit_horner_c(
    expr: ExprId,
    var: ExprId,
    var_name: &str,
    fn_name: &str,
    pool: &ExprPool,
) -> Result<String, ConversionError> {
    let poly = UniPoly::from_symbolic(expr, var, pool)?;
    let coeffs = poly.coefficients_i64();
    let body = build_c_horner(&coeffs, var_name);
    Ok(format!(
        "double {}(double {}) {{\n    return {};\n}}\n",
        fn_name, var_name, body
    ))
}

// Build a C expression string for the Horner evaluation.
// ---------------------------------------------------------------------------
// f64 Horner evaluation (scalar + SIMD batch)
// ---------------------------------------------------------------------------

/// Evaluate `a₀ + x·(a₁ + x·(a₂ + …))` for coefficients `[a₀, a₁, …, aₙ]`.
#[inline]
pub fn eval_horner_f64(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }
    let mut acc = coeffs[coeffs.len() - 1];
    for &c in coeffs[..coeffs.len() - 1].iter().rev() {
        acc = c + x * acc;
    }
    acc
}

/// Evaluate the same polynomial at many points, writing into `out`.
///
/// Uses 4-wide SIMD via the `wide` crate when `xs.len() == out.len()`; falls
/// back to scalar [`eval_horner_f64`] for tail elements.
pub fn eval_horner_f64_batch(coeffs: &[f64], xs: &[f64], out: &mut [f64]) {
    assert_eq!(xs.len(), out.len());
    let mut i = 0;
    while i + 4 <= xs.len() {
        let chunk = wide::f64x4::new([xs[i], xs[i + 1], xs[i + 2], xs[i + 3]]);
        let vals = eval_horner_f64x4(coeffs, chunk).to_array();
        out[i..i + 4].copy_from_slice(&vals);
        i += 4;
    }
    for (x, o) in xs[i..].iter().zip(out[i..].iter_mut()) {
        *o = eval_horner_f64(coeffs, *x);
    }
}

#[inline]
fn eval_horner_f64x4(coeffs: &[f64], x: wide::f64x4) -> wide::f64x4 {
    if coeffs.is_empty() {
        return wide::f64x4::splat(0.0);
    }
    let mut acc = wide::f64x4::splat(coeffs[coeffs.len() - 1]);
    for &c in coeffs[..coeffs.len() - 1].iter().rev() {
        acc = wide::f64x4::splat(c) + x * acc;
    }
    acc
}

fn build_c_horner(coeffs: &[i64], var: &str) -> String {
    if coeffs.is_empty() {
        return "0.0".to_string();
    }
    let n = coeffs.len();
    let mut result = format!("{}.0", coeffs[n - 1]);
    for k in (0..n - 1).rev() {
        let ck = format!("{}.0", coeffs[k]);
        result = format!("{} + {} * ({})", ck, var, result);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::{Domain, ExprPool};
    use std::collections::HashMap;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn horner_linear() {
        // 2x + 1 → Horner: 1 + x*2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let h = horner(expr, x, &pool).unwrap();
        // Verify at x=3: 2*3+1=7
        let mut env = HashMap::new();
        env.insert(x, 3.0f64);
        let val = eval_interp(h, &env, &pool).unwrap();
        assert!((val - 7.0).abs() < 1e-10, "expected 7.0, got {val}");
    }

    #[test]
    fn horner_quadratic() {
        // x² + 2x + 1 → Horner: 1 + x*(2 + x*1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let one = pool.integer(1_i32);
        let expr = pool.add(vec![x2, two_x, one]);
        let h = horner(expr, x, &pool).unwrap();
        // Verify at several points: (x+1)²
        let mut env = HashMap::new();
        for v in [-2.0f64, -1.0, 0.0, 1.0, 2.0, 3.0] {
            env.insert(x, v);
            let expected = (v + 1.0).powi(2);
            let actual = eval_interp(h, &env, &pool).unwrap();
            assert!(
                (actual - expected).abs() < 1e-9,
                "v={v}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn horner_degree_10_op_count() {
        // Degree-10 poly should have exactly 10 Mul nodes in Horner form
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // Build x^10 + x^9 + … + x + 1
        let mut expr = pool.integer(1_i32);
        for k in 1_i32..=10 {
            let xk = pool.pow(x, pool.integer(k));
            expr = pool.add(vec![expr, xk]);
        }
        let h = horner(expr, x, &pool).unwrap();
        // Count Mul nodes in the tree
        let muls = count_muls(h, &pool);
        assert!(
            muls <= 10,
            "Horner form should use ≤ 10 multiplications, got {muls}"
        );
    }

    fn count_muls(expr: ExprId, pool: &ExprPool) -> usize {
        use crate::kernel::ExprData;
        match pool.get(expr) {
            ExprData::Mul(args) => 1 + args.iter().map(|&a| count_muls(a, pool)).sum::<usize>(),
            ExprData::Add(args) => args.iter().map(|&a| count_muls(a, pool)).sum(),
            ExprData::Pow { base, exp } => count_muls(base, pool) + count_muls(exp, pool),
            _ => 0,
        }
    }

    #[test]
    fn emit_horner_c_quadratic() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let one = pool.integer(1_i32);
        let expr = pool.add(vec![x2, two_x, one]);
        let code = emit_horner_c(expr, x, "x", "eval_quad", &pool).unwrap();
        assert!(code.contains("eval_quad"), "function name not in output");
        assert!(code.contains("double"), "return type not in output");
    }

    #[test]
    fn horner_constant() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let five = pool.integer(5_i32);
        let h = horner(five, x, &pool).unwrap();
        let env = HashMap::new();
        let val = eval_interp(h, &env, &pool).unwrap();
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn eval_horner_f64_matches_interp() {
        let coeffs = [1.0, 2.0, 3.0]; // 1 + 2x + 3x²
        let xs = [-1.0, 0.0, 0.5, 2.0, 10.0];
        for &x in &xs {
            let scalar = eval_horner_f64(&coeffs, x);
            let expected = 1.0 + x * (2.0 + x * 3.0);
            assert!((scalar - expected).abs() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn eval_horner_f64_batch_matches_scalar() {
        let coeffs = [1.0, 2.0, 3.0];
        let xs = [-1.0, 0.0, 0.5, 2.0, 10.0, 3.0, 7.0];
        let mut out = vec![0.0; xs.len()];
        eval_horner_f64_batch(&coeffs, &xs, &mut out);
        for (i, &x) in xs.iter().enumerate() {
            assert!((out[i] - eval_horner_f64(&coeffs, x)).abs() < 1e-12);
        }
    }
}
