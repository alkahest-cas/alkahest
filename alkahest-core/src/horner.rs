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
//!
//! # Transcendental C emission
//!
//! [`emit_expr_c`] and [`emit_expr_c_vec`] walk the expression DAG and emit C
//! that calls `<math.h>` functions (sin, cos, exp, log, sqrt, tan, …) for
//! [`ExprData::Func`](crate::kernel::ExprData::Func) nodes.  This is the general entry point;
//! [`emit_horner_c`] remains available for the polynomial-only Horner form.
//!
//! Both share common sub-expressions across every output — see the `cemit`
//! submodule for the lowering, the rewrites, and where the emitted C and the
//! interpreter differ.

use crate::kernel::{integer_to_f64, ExprId, ExprPool};
use crate::poly::{ConversionError, UniPoly};

mod cemit;

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
    // `coefficients_i64` wraps modulo 2^64 without saying so, and this is a
    // *rewrite*: the result claims to be the same polynomial. It was not.
    // `(x+1)^80` has binomial coefficients past `i64`, so `horner` returned a
    // different polynomial — `2.12e20` instead of `2^80` at `x = 1` — with no
    // error and no flag. Exact coefficients throughout.
    let coeffs = poly.coefficients(); // [a0, a1, …, an]
    Ok(build_horner(&coeffs, var, pool))
}

/// Build Horner form from a coefficient slice `[a0, a1, …, an]` where
/// `a_k` is the coefficient of `x^k`.
fn build_horner(coeffs: &[rug::Integer], var: ExprId, pool: &ExprPool) -> ExprId {
    if coeffs.is_empty() {
        return pool.integer(0_i32);
    }
    // Start from the highest-degree coefficient and fold inward:
    // result = a_n
    // result = a_{n-1} + x * result
    // …
    // result = a_0   + x * result
    let n = coeffs.len();
    let mut result = pool.integer(coeffs[n - 1].clone());
    for k in (0..n - 1).rev() {
        // result = coeffs[k] + var * result
        let xr = pool.mul(vec![var, result]);
        let ck = pool.integer(coeffs[k].clone());
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
    // Exact, for the reason spelled out in `horner` above.
    let coeffs = poly.coefficients();
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

/// Render an exact integer as a C `double` literal.
///
/// Written out in full while it is exact in `f64`, so the C compiler's own
/// correctly-rounded parse agrees with the interpreter bit for bit. Past that
/// the exact decimal would be both misleading and, above `DBL_MAX`, outside
/// the range a C compiler will accept, so the rounded value is emitted
/// instead — the same one every evaluator uses.
fn c_integer_literal(n: &rug::Integer) -> String {
    if n.significant_bits() <= f64::MANTISSA_DIGITS {
        format!("{n}.0")
    } else {
        c_double_literal(integer_to_f64(n))
    }
}

fn build_c_horner(coeffs: &[rug::Integer], var: &str) -> String {
    if coeffs.is_empty() {
        return "0.0".to_string();
    }
    let n = coeffs.len();
    let mut result = c_integer_literal(&coeffs[n - 1]);
    for k in (0..n - 1).rev() {
        let ck = c_integer_literal(&coeffs[k]);
        result = format!("{} + {} * ({})", ck, var, result);
    }
    result
}

// ---------------------------------------------------------------------------
// Transcendental C emission — general DAG walker
// ---------------------------------------------------------------------------

/// Error type returned by [`emit_expr_c`] / [`emit_expr_c_vec`] when the
/// expression contains a construct that cannot be lowered to C.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitCError {
    /// A function call has no known C `<math.h>` equivalent.
    UnsupportedFunction(String),
    /// An expression node kind is not representable in straight-line C
    /// (e.g. a `Piecewise` condition or a `Forall`/`Exists` quantifier).
    UnsupportedNode(String),
    /// The `vars` slice is empty but the expression references a symbol.
    MissingVariable(String),
}

impl std::fmt::Display for EmitCError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitCError::UnsupportedFunction(name) => {
                write!(f, "function '{name}' has no C math.h equivalent")
            }
            EmitCError::UnsupportedNode(desc) => {
                write!(f, "expression node not supported in C emission: {desc}")
            }
            EmitCError::MissingVariable(name) => {
                write!(
                    f,
                    "symbol '{name}' is not listed in the vars/var_names parameter"
                )
            }
        }
    }
}

impl std::error::Error for EmitCError {}

impl crate::errors::AlkahestError for EmitCError {
    fn code(&self) -> &'static str {
        match self {
            EmitCError::UnsupportedFunction(_) => "E-CODEGEN-001",
            EmitCError::UnsupportedNode(_) => "E-CODEGEN-002",
            EmitCError::MissingVariable(_) => "E-CODEGEN-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            EmitCError::UnsupportedFunction(_) => {
                "rewrite the function in terms of <math.h> functions, or evaluate it \
                 with compile_expr / eval_expr instead of emitting C"
            }
            EmitCError::UnsupportedNode(_) => {
                "remove quantifiers, O() terms and root sums before emitting C; \
                 Piecewise conditions must be comparisons"
            }
            EmitCError::MissingVariable(_) => {
                "list every free symbol in vars (pi needs no binding)"
            }
        })
    }
}

/// Render an `f64` as a C `double` literal.
///
/// `{v:?}` alone produced the bare tokens `inf`, `-inf` and `NaN` for a
/// non-finite constant. None of the three is a C identifier, so `emit_c_expr`
/// reported success and handed back a function that does not compile. The
/// `<math.h>` macros the emitted code already requires say the same thing and
/// do compile.
fn c_double_literal(v: f64) -> String {
    if v.is_nan() {
        return "NAN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "INFINITY" } else { "-INFINITY" }.to_string();
    }
    format!("{v:?}")
}

/// Emit a C function that evaluates the symbolic expression `expr`.
///
/// Unlike [`emit_horner_c`], this function supports arbitrary expressions
/// including transcendental functions (`sin`, `cos`, `exp`, `log`, `sqrt`,
/// `tan`, `atan2`, `erf`, …) — not just polynomials.  The emitted code
/// `#include <math.h>` is required at the call site.
///
/// The arithmetic is real `double`. When the expression can have complex
/// intermediates (a square root, logarithm, `acos`/`asin` or non-integer
/// power, or the imaginary unit), a complex re-evaluation is emitted with it
/// and used at a point where the real code gives NaN: it returns the real
/// part when the imaginary part is zero within its rounding-error bound, so
/// `cos(t·√ω²)` at `ω² < 0` or Cardano's formula with three real roots give
/// their real value, and NaN only where the value is not real (`log(−1)`,
/// `√−2`) — the same rule `compile_expr` applies. An expression written with
/// the imaginary unit is emitted as the complex evaluation alone.
///
/// # Parameters
///
/// - `expr`: The symbolic expression to compile.
/// - `vars`: The symbolic variables (in argument order).  Each variable maps
///   to the corresponding name in `var_names`.
/// - `var_names`: C parameter names for each variable.  Must have the same
///   length as `vars`.
/// - `fn_name`: The C function name.
///
/// # Errors
///
/// Returns [`EmitCError::UnsupportedFunction`] if the expression calls a
/// function with no `<math.h>` equivalent (e.g. `diracdelta`, elliptic
/// integrals), or [`EmitCError::MissingVariable`] if the expression
/// references a symbol not listed in `vars`.
///
/// # Example
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::horner::emit_expr_c;
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let expr = pool.add(vec![
///     pool.func("sin", vec![x]),
///     pool.pow(x, pool.integer(2_i32)),
/// ]);
/// let code = emit_expr_c(expr, &[x], &["x"], "f", &pool).unwrap();
/// assert!(code.contains("sin("));
/// assert!(code.contains("double f"));
/// ```
pub fn emit_expr_c(
    expr: ExprId,
    vars: &[ExprId],
    var_names: &[&str],
    fn_name: &str,
    pool: &ExprPool,
) -> Result<String, EmitCError> {
    assert_eq!(
        vars.len(),
        var_names.len(),
        "vars and var_names must have the same length"
    );

    let fallback = crate::jit::complex_fallback::emit_c_fallbacks(
        &[expr],
        vars,
        var_names,
        &format!("{fn_name}__cfb"),
        pool,
    );
    let params: Vec<String> = var_names.iter().map(|n| format!("double {n}")).collect();
    let params_str = params.join(", ");
    let args = var_names.join(", ");
    let comma = if var_names.is_empty() { "" } else { ", " };

    if fallback.outputs[0] == Some(true) {
        // Written with the imaginary unit: the complex evaluation is the
        // function (the real lowering has no spelling for `I`).
        return Ok(format!(
            "{}double {fn_name}({params_str}) {{\n    return {fn_name}__cfb(0{comma}{args});\n}}\n",
            fallback.code
        ));
    }

    let low = cemit::lower_outputs(&[expr], vars, var_names, pool)?;
    let result = &low.outputs[0];
    let stmts = low.stmts;

    let ret = if fallback.outputs[0].is_some() {
        format!(
            "    double ak_r = {result};\n    if (isnan(ak_r)) ak_r = {fn_name}__cfb(0{comma}{args});\n    return ak_r;\n"
        )
    } else {
        format!("    return {result};\n")
    };
    // Build function body.
    let body = if stmts.is_empty() {
        ret
    } else {
        let mut body = stmts.join("\n");
        body.push('\n');
        body.push_str(&ret);
        body
    };

    Ok(format!(
        "{}{}double {fn_name}({params_str}) {{\n{body}}}\n",
        low.prelude, fallback.code
    ))
}

/// Emit a C function that writes multiple symbolic expressions into an output
/// array.
///
/// This is the vector-output path: a single C function computes `exprs.len()`
/// values and writes them into `double *out`.  All expressions share the same
/// set of input variables and the same `<math.h>` support as [`emit_expr_c`].
///
/// The generated signature is:
///
/// ```c
/// void fn_name(double var0, double var1, …, double *out);
/// ```
///
/// where `out[i]` receives the value of `exprs[i]`.
///
/// # Parameters
///
/// - `exprs`: Slice of expressions to evaluate (one per output component).
/// - `vars`: Symbolic variables (in argument order).
/// - `var_names`: C parameter names for each variable.
/// - `fn_name`: The C function name.
///
/// # Errors
///
/// Same as [`emit_expr_c`].
///
/// # Example
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::horner::emit_expr_c_vec;
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let y = pool.symbol("y", Domain::Real);
/// let f0 = pool.func("sin", vec![x]);
/// let f1 = pool.add(vec![pool.pow(x, pool.integer(2_i32)), y]);
/// let code = emit_expr_c_vec(&[f0, f1], &[x, y], &["x", "y"], "eval_vec", &pool).unwrap();
/// assert!(code.contains("double *out"));
/// assert!(code.contains("out[0]"));
/// assert!(code.contains("out[1]"));
/// assert!(code.contains("sin("));
/// ```
pub fn emit_expr_c_vec(
    exprs: &[ExprId],
    vars: &[ExprId],
    var_names: &[&str],
    fn_name: &str,
    pool: &ExprPool,
) -> Result<String, EmitCError> {
    assert_eq!(
        vars.len(),
        var_names.len(),
        "vars and var_names must have the same length"
    );

    let fallback = crate::jit::complex_fallback::emit_c_fallbacks(
        exprs,
        vars,
        var_names,
        &format!("{fn_name}__cfb"),
        pool,
    );
    let args = var_names.join(", ");
    let comma = if var_names.is_empty() { "" } else { ", " };
    // Outputs written with the imaginary unit are the complex evaluation
    // alone; the rest go through the real lowering together.
    let real_ix: Vec<usize> = (0..exprs.len())
        .filter(|&k| fallback.outputs[k] != Some(true))
        .collect();
    let real_exprs: Vec<ExprId> = real_ix.iter().map(|&k| exprs[k]).collect();
    let low = if real_exprs.is_empty() {
        cemit::Lowered {
            stmts: Vec::new(),
            outputs: Vec::new(),
            prelude: String::new(),
        }
    } else {
        cemit::lower_outputs(&real_exprs, vars, var_names, pool)?
    };
    let stmts = low.stmts;
    let mut result_exprs: Vec<Option<&String>> = vec![None; exprs.len()];
    for (&k, r) in real_ix.iter().zip(&low.outputs) {
        result_exprs[k] = Some(r);
    }

    // Build the parameter list: inputs then output pointer.
    let mut params: Vec<String> = var_names.iter().map(|n| format!("double {n}")).collect();
    params.push("double *out".to_string());
    let params_str = params.join(", ");

    // Build assignments into out[i].
    let assignments: Vec<String> = result_exprs
        .iter()
        .enumerate()
        .map(|(i, r)| match (r, &fallback.outputs[i]) {
            (Some(r), None) => format!("    out[{i}] = {r};"),
            (Some(r), Some(_)) => format!(
                "    out[{i}] = {r};\n    if (isnan(out[{i}])) out[{i}] = {fn_name}__cfb({i}{comma}{args});"
            ),
            (None, _) => format!("    out[{i}] = {fn_name}__cfb({i}{comma}{args});"),
        })
        .collect();

    // Build function body.
    let body = if stmts.is_empty() {
        format!("{}\n", assignments.join("\n"))
    } else {
        format!("{}\n{}\n", stmts.join("\n"), assignments.join("\n"))
    };

    Ok(format!(
        "{}{}void {fn_name}({params_str}) {{\n{body}}}\n",
        low.prelude, fallback.code
    ))
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
    fn emitted_c_keeps_the_parity_of_an_exponent_a_double_cannot_hold() {
        use rug::ops::Pow;
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let n = rug::Integer::from(10).pow(30) + 1u32;

        // The literal a C program can hold is `1e30`, which is *even*, so
        // `pow(x, 1e30)` is `+1` at `x = -1` and the answer is `-1`.
        let odd = emit_expr_c(
            pool.pow(x, pool.integer(n.clone())),
            &[x],
            &["x"],
            "f",
            &pool,
        )
        .unwrap();
        assert!(odd.contains("copysign(pow(fabs(x), 1e30), x)"), "{odd}");

        let even = emit_expr_c(
            pool.pow(x, pool.integer(n - 1u32)),
            &[x],
            &["x"],
            "g",
            &pool,
        )
        .unwrap();
        assert!(even.contains("pow(fabs(x), 1e30)"), "{even}");
        assert!(!even.contains("copysign"), "{even}");

        // Controls: every exponent a double holds exactly is emitted as before.
        for k in [5_i32, 19, -7] {
            let code = emit_expr_c(pool.pow(x, pool.integer(k)), &[x], &["x"], "h", &pool).unwrap();
            assert!(code.contains(&format!("pow(x, {k}.0)")), "k = {k}: {code}");
        }
        let wide_exact = pool.pow(x, pool.integer(rug::Integer::from(1u64 << 60)));
        let code = emit_expr_c(wide_exact, &[x], &["x"], "h", &pool).unwrap();
        assert!(code.contains("pow(x, 1.152921504606847e18)"), "{code}");
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

    // -----------------------------------------------------------------------
    // emit_expr_c tests
    // -----------------------------------------------------------------------

    #[test]
    fn emit_expr_c_sin_plus_x_squared() {
        // sin(x) + x² — the original failing case
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let expr = pool.add(vec![sin_x, x2]);
        let code = emit_expr_c(expr, &[x], &["x"], "f", &pool).unwrap();
        assert!(code.contains("sin("), "expected sin( in:\n{code}");
        assert!(
            code.contains("double f(double x)"),
            "expected signature:\n{code}"
        );
        assert!(code.contains("return "), "expected return:\n{code}");
    }

    #[test]
    fn emit_expr_c_transcendentals() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // cos(x) * exp(x)
        let expr = pool.mul(vec![pool.func("cos", vec![x]), pool.func("exp", vec![x])]);
        let code = emit_expr_c(expr, &[x], &["x"], "g", &pool).unwrap();
        assert!(code.contains("cos("), "expected cos(:\n{code}");
        assert!(code.contains("exp("), "expected exp(:\n{code}");
    }

    #[test]
    fn emit_expr_c_multivar() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // sqrt(x² + y²)
        let x2 = pool.pow(x, pool.integer(2_i32));
        let y2 = pool.pow(y, pool.integer(2_i32));
        let inner = pool.add(vec![x2, y2]);
        let expr = pool.func("sqrt", vec![inner]);
        let code = emit_expr_c(expr, &[x, y], &["x", "y"], "norm", &pool).unwrap();
        assert!(code.contains("sqrt("), "expected sqrt(:\n{code}");
        assert!(code.contains("double x"), "expected x param:\n{code}");
        assert!(code.contains("double y"), "expected y param:\n{code}");
    }

    #[test]
    fn emit_expr_c_unsupported_func_errors() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // diracdelta has no C math.h equivalent
        let expr = pool.func("diracdelta", vec![x]);
        let err = emit_expr_c(expr, &[x], &["x"], "f", &pool).unwrap_err();
        assert!(
            matches!(err, EmitCError::UnsupportedFunction(ref n) if n == "diracdelta"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn emit_expr_c_missing_var_errors() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // y is not listed in vars
        let expr = pool.add(vec![x, y]);
        let err = emit_expr_c(expr, &[x], &["x"], "f", &pool).unwrap_err();
        assert!(
            matches!(err, EmitCError::MissingVariable(ref n) if n == "y"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn emit_expr_c_vec_two_outputs() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let f0 = pool.func("sin", vec![x]);
        let f1 = pool.add(vec![pool.pow(x, pool.integer(2_i32)), y]);
        let code = emit_expr_c_vec(&[f0, f1], &[x, y], &["x", "y"], "eval_vec", &pool).unwrap();
        assert!(
            code.contains("double *out"),
            "expected out pointer:\n{code}"
        );
        assert!(code.contains("out[0]"), "expected out[0]:\n{code}");
        assert!(code.contains("out[1]"), "expected out[1]:\n{code}");
        assert!(code.contains("sin("), "expected sin(:\n{code}");
    }

    #[test]
    fn emit_expr_c_polynomial_still_works() {
        // Pure polynomial: x² + 2x + 1 — must still emit valid C
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let one = pool.integer(1_i32);
        let expr = pool.add(vec![x2, two_x, one]);
        let code = emit_expr_c(expr, &[x], &["x"], "eval_poly_new", &pool).unwrap();
        assert!(code.contains("double eval_poly_new"), "signature:\n{code}");
        assert!(code.contains("return "), "return:\n{code}");
    }
}

#[cfg(test)]
mod exact_coefficient_tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::{Domain, ExprPool};
    use rug::ops::Pow;
    use std::collections::HashMap;

    fn at(expr: ExprId, x: ExprId, v: f64, pool: &ExprPool) -> f64 {
        let mut env = HashMap::new();
        env.insert(x, v);
        eval_interp(expr, &env, pool).unwrap()
    }

    /// `horner` is a *rewrite*: it promises the same polynomial in a better
    /// shape. Going through `coefficients_i64` made that promise false —
    /// coefficients wrapped modulo 2^64, silently, so the returned expression
    /// was a different polynomial.
    #[test]
    fn horner_keeps_coefficients_past_i64() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 2^70 + 3 wraps to 3 in `i64`.
        let big = rug::Integer::from(2u32).pow(70) + 3u32;
        let expr = pool.add(vec![
            pool.mul(vec![
                pool.integer(big.clone()),
                pool.pow(x, pool.integer(2_i32)),
            ]),
            pool.mul(vec![pool.integer(5_i32), x]),
            pool.integer(7_i32),
        ]);
        let h = horner(expr, x, &pool).unwrap();
        for v in [0.0, 1.0, 2.0, -3.0] {
            assert_eq!(
                at(h, x, v, &pool),
                at(expr, x, v, &pool),
                "horner form disagrees with the input at x = {v}"
            );
        }
        // …and the value is the mathematically right one, not just consistent.
        // (2^70 + 3) + 5 + 7 = 1180591620717411303439.
        assert_eq!(at(h, x, 1.0, &pool), 1.1805916207174113e21);
    }

    /// The same defect on a polynomial nobody would call exotic: the binomial
    /// coefficients of `(x+1)^80` run past `i64` in the middle of the row.
    #[test]
    fn horner_of_an_ordinary_binomial_power_is_the_same_polynomial() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(pool.add(vec![x, pool.integer(1_i32)]), pool.integer(80_i32));
        let h = horner(expr, x, &pool).unwrap();
        // (1+1)^80 = 2^80.
        assert_eq!(at(h, x, 1.0, &pool), 2f64.powi(80));
        assert_eq!(at(h, x, 0.0, &pool), 1.0);
    }

    #[test]
    fn emit_horner_c_keeps_coefficients_past_i64() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let big = rug::Integer::from(2u32).pow(70) + 3u32;
        let expr = pool.add(vec![
            pool.mul(vec![pool.integer(big), x]),
            pool.integer(1_i32),
        ]);
        let c = emit_horner_c(expr, x, "x", "f", &pool).unwrap();
        assert!(
            c.contains("1.1805916207174113e21"),
            "emitted C lost the coefficient: {c}"
        );
        assert!(!c.contains(" 3.0"), "coefficient wrapped to 3: {c}");
    }

    /// A rational literal is emitted as the same `double` the interpreter
    /// uses, so the C and the interpreter cannot drift apart by an ulp.
    #[test]
    fn emit_expr_c_rounds_rationals_the_way_the_interpreter_does() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.rational(2, 5), x]);
        let c = emit_expr_c(expr, &[x], &["x"], "f", &pool).unwrap();
        assert!(c.contains("0.4"), "expected the nearest double 0.4: {c}");
        assert!(!c.contains("0.39999999999999997"), "truncated: {c}");
    }

    /// `{v:?}` produced the bare tokens `inf` / `NaN`, which are not C
    /// identifiers: `emit_expr_c` reported success and returned a function
    /// that does not compile.
    #[test]
    fn emit_expr_c_writes_math_h_macros_for_non_finite_literals() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for (v, want) in [
            (f64::INFINITY, "INFINITY"),
            (f64::NEG_INFINITY, "-INFINITY"),
            (f64::NAN, "NAN"),
        ] {
            let expr = pool.mul(vec![pool.float(v, 53), x]);
            let c = emit_expr_c(expr, &[x], &["x"], "f", &pool).unwrap();
            assert!(c.contains(want), "expected {want} in: {c}");
            assert!(!c.contains(" inf"), "bare `inf` is not valid C: {c}");
            assert!(!c.contains("NaN"), "bare `NaN` is not valid C: {c}");
        }
    }

    /// An integer past the `f64` grid must be emitted as the `double` every
    /// evaluator actually uses, not as an exact decimal a C compiler would
    /// have to round on its own (and would reject above `DBL_MAX`).
    #[test]
    fn emit_expr_c_emits_large_integers_as_in_range_doubles() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.integer(rug::Integer::from(10).pow(400)), x]);
        let c = emit_expr_c(expr, &[x], &["x"], "f", &pool).unwrap();
        assert!(c.contains("INFINITY"), "expected a saturating literal: {c}");

        let expr = pool.mul(vec![pool.integer(rug::Integer::from(10).pow(30)), x]);
        let c = emit_expr_c(expr, &[x], &["x"], "f", &pool).unwrap();
        assert!(c.contains("1e30"), "expected the rounded double: {c}");
    }

    /// N11: `cos(t·√w)` is `cosh(t·√−w)` at `w < 0`; the emitted C falls back
    /// to its complex evaluation where the real code gives NaN. An output
    /// written with `I` is the complex evaluation alone, and an expression
    /// whose NaN cannot hide a real value (`√x + 1`) gets no fallback.
    #[test]
    fn emitted_c_carries_the_complex_fallback_where_it_can_matter() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let w = pool.symbol("w", Domain::Real);
        let osc = pool.func(
            "cos",
            vec![pool.mul(vec![t, pool.pow(w, pool.rational(1, 2))])],
        );
        let c = emit_expr_c(osc, &[t, w], &["t", "w"], "f", &pool).unwrap();
        assert!(c.contains("static double f__cfb(int ak_out, double t, double w)"));
        assert!(
            c.contains("if (isnan(ak_r)) ak_r = f__cfb(0, t, w);"),
            "{c}"
        );

        let i = pool.imaginary_unit();
        let with_i = pool.mul(vec![i, pool.func("exp", vec![pool.mul(vec![i, t])])]);
        let c = emit_expr_c(with_i, &[t], &["t"], "g", &pool).unwrap();
        assert!(c.contains("return g__cfb(0, t);"), "{c}");

        let plain = pool.add(vec![pool.func("sqrt", vec![w]), pool.integer(1)]);
        let c = emit_expr_c(plain, &[w], &["w"], "h", &pool).unwrap();
        assert!(!c.contains("__cfb"), "{c}");

        let c = emit_expr_c_vec(&[plain, osc, with_i], &[t, w], &["t", "w"], "v", &pool).unwrap();
        assert!(c.contains("out[0] = ") && !c.contains("v__cfb(0,"), "{c}");
        assert!(
            c.contains("if (isnan(out[1])) out[1] = v__cfb(1, t, w);"),
            "{c}"
        );
        assert!(c.contains("out[2] = v__cfb(2, t, w);"), "{c}");
        // One runtime, one fallback function, shared by the outputs.
        assert_eq!(c.matches("static double v__cfb(").count(), 1, "{c}");
    }
}
