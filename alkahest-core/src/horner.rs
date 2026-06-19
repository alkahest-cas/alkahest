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
//! [`ExprData::Func`] nodes.  This is the general entry point;
//! [`emit_horner_c`] remains available for the polynomial-only Horner form.

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::{ConversionError, UniPoly};
use std::collections::HashMap;

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

/// Map an Alkahest function name to its C `<math.h>` equivalent.
///
/// Returns `None` for functions that cannot be expressed as a single `<math.h>`
/// call (e.g. `diracdelta`, elliptic integrals, `heaviside`).
fn func_to_c(name: &str, arg_exprs: &[String]) -> Option<String> {
    // Binary functions first
    if name == "atan2" && arg_exprs.len() == 2 {
        return Some(format!("atan2({}, {})", arg_exprs[0], arg_exprs[1]));
    }
    if name == "pow" && arg_exprs.len() == 2 {
        return Some(format!("pow({}, {})", arg_exprs[0], arg_exprs[1]));
    }
    if name == "min" && arg_exprs.len() == 2 {
        return Some(format!("fmin({}, {})", arg_exprs[0], arg_exprs[1]));
    }
    if name == "max" && arg_exprs.len() == 2 {
        return Some(format!("fmax({}, {})", arg_exprs[0], arg_exprs[1]));
    }
    // Unary functions
    if arg_exprs.len() == 1 {
        let a = &arg_exprs[0];
        let c_name = match name {
            "sin" => "sin",
            "cos" => "cos",
            "tan" => "tan",
            "asin" => "asin",
            "acos" => "acos",
            "atan" => "atan",
            "sinh" => "sinh",
            "cosh" => "cosh",
            "tanh" => "tanh",
            "exp" => "exp",
            "log" => "log",
            "sqrt" => "sqrt",
            "abs" => "fabs",
            "erf" => "erf",
            "erfc" => "erfc",
            "tgamma" | "gamma" => "tgamma",
            "floor" => "floor",
            "ceil" => "ceil",
            "round" => "round",
            "sign" => return Some(format!("(({a}) > 0.0 ? 1.0 : (({a}) < 0.0 ? -1.0 : 0.0))")),
            "log10" => "log10",
            "log2" => "log2",
            "exp2" => "exp2",
            "cbrt" => "cbrt",
            _ => return None,
        };
        return Some(format!("{c_name}({a})"));
    }
    None
}

/// Internal: emit a C expression for `expr`, collecting temporary variable
/// assignments into `stmts`.  Returns the C expression string (may be a temp
/// var name like `_t3` or an inline expression like `sin(_t2)`).
///
/// `var_map` maps ExprId → C variable name for the function's arguments.
/// `memo` caches already-emitted sub-expressions to avoid duplication in the
/// output (DAG sharing becomes shared temporaries).
fn emit_expr_inner(
    expr: ExprId,
    var_map: &HashMap<ExprId, &str>,
    pool: &ExprPool,
    stmts: &mut Vec<String>,
    memo: &mut HashMap<ExprId, String>,
    counter: &mut usize,
) -> Result<String, EmitCError> {
    if let Some(cached) = memo.get(&expr) {
        return Ok(cached.clone());
    }
    if let Some(&name) = var_map.get(&expr) {
        return Ok(name.to_string());
    }

    let result = match pool.get(expr) {
        ExprData::Integer(n) => {
            // Emit as double literal
            format!("{}.0", n.0)
        }
        ExprData::Rational(r) => {
            let v = r.0.numer().to_f64() / r.0.denom().to_f64();
            format!("{v:?}")
        }
        ExprData::Float(f) => {
            let v = f.inner.to_f64();
            format!("{v:?}")
        }
        ExprData::Symbol { name, .. } => {
            return Err(EmitCError::MissingVariable(name.clone()));
        }
        ExprData::Add(args) => {
            let mut parts = Vec::with_capacity(args.len());
            for &a in &args {
                parts.push(emit_expr_inner(a, var_map, pool, stmts, memo, counter)?);
            }
            format!("({})", parts.join(" + "))
        }
        ExprData::Mul(args) => {
            let mut parts = Vec::with_capacity(args.len());
            for &a in &args {
                parts.push(emit_expr_inner(a, var_map, pool, stmts, memo, counter)?);
            }
            format!("({})", parts.join(" * "))
        }
        ExprData::Pow { base, exp } => {
            let b = emit_expr_inner(base, var_map, pool, stmts, memo, counter)?;
            let e = emit_expr_inner(exp, var_map, pool, stmts, memo, counter)?;
            // Specialise integer exponents to repeated multiplication for
            // small n, otherwise use pow().
            match pool.get(exp) {
                ExprData::Integer(ref n) => {
                    if let Some(k) = n.0.to_i32() {
                        match k {
                            0 => "1.0".to_string(),
                            1 => b,
                            2 => format!("({b} * {b})"),
                            3 => format!("({b} * {b} * {b})"),
                            -1 => format!("(1.0 / {b})"),
                            _ => format!("pow({b}, {e})"),
                        }
                    } else {
                        format!("pow({b}, {e})")
                    }
                }
                _ => format!("pow({b}, {e})"),
            }
        }
        ExprData::Func { name, args } => {
            let mut arg_exprs = Vec::with_capacity(args.len());
            for &a in &args {
                arg_exprs.push(emit_expr_inner(a, var_map, pool, stmts, memo, counter)?);
            }
            func_to_c(&name, &arg_exprs).ok_or(EmitCError::UnsupportedFunction(name))?
        }
        ExprData::Piecewise { branches, default } => {
            // Emit as a chain of ternary operators.
            // Conditions must be predicates; we recursively emit them.
            let default_str = emit_expr_inner(default, var_map, pool, stmts, memo, counter)?;
            let mut result = default_str;
            // Iterate in reverse so we build right-to-left ternary nesting.
            for (cond, val) in branches.iter().rev() {
                let cond_str = emit_predicate_inner(*cond, var_map, pool, stmts, memo, counter)?;
                let val_str = emit_expr_inner(*val, var_map, pool, stmts, memo, counter)?;
                result = format!("(({cond_str}) ? ({val_str}) : ({result}))");
            }
            result
        }
        other => {
            return Err(EmitCError::UnsupportedNode(format!("{other:?}")));
        }
    };

    // If the result is longer than a simple identifier/literal, assign to a
    // temporary to keep nested expressions readable.
    let is_simple = result.len() <= 24
        || result.starts_with('(')
        || result
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '-');
    if !is_simple {
        let tmp = format!("_t{}", *counter);
        *counter += 1;
        stmts.push(format!("    double {tmp} = {result};"));
        memo.insert(expr, tmp.clone());
        return Ok(tmp);
    }
    memo.insert(expr, result.clone());
    Ok(result)
}

/// Internal: emit a C boolean expression for a `Predicate` node.
fn emit_predicate_inner(
    pred: ExprId,
    var_map: &HashMap<ExprId, &str>,
    pool: &ExprPool,
    stmts: &mut Vec<String>,
    memo: &mut HashMap<ExprId, String>,
    counter: &mut usize,
) -> Result<String, EmitCError> {
    use crate::kernel::expr::PredicateKind;
    match pool.get(pred) {
        ExprData::Predicate { kind, args } => match kind {
            PredicateKind::True => Ok("1".to_string()),
            PredicateKind::False => Ok("0".to_string()),
            PredicateKind::Lt => {
                let l = emit_expr_inner(args[0], var_map, pool, stmts, memo, counter)?;
                let r = emit_expr_inner(args[1], var_map, pool, stmts, memo, counter)?;
                Ok(format!("({l}) < ({r})"))
            }
            PredicateKind::Le => {
                let l = emit_expr_inner(args[0], var_map, pool, stmts, memo, counter)?;
                let r = emit_expr_inner(args[1], var_map, pool, stmts, memo, counter)?;
                Ok(format!("({l}) <= ({r})"))
            }
            PredicateKind::Gt => {
                let l = emit_expr_inner(args[0], var_map, pool, stmts, memo, counter)?;
                let r = emit_expr_inner(args[1], var_map, pool, stmts, memo, counter)?;
                Ok(format!("({l}) > ({r})"))
            }
            PredicateKind::Ge => {
                let l = emit_expr_inner(args[0], var_map, pool, stmts, memo, counter)?;
                let r = emit_expr_inner(args[1], var_map, pool, stmts, memo, counter)?;
                Ok(format!("({l}) >= ({r})"))
            }
            PredicateKind::Eq => {
                let l = emit_expr_inner(args[0], var_map, pool, stmts, memo, counter)?;
                let r = emit_expr_inner(args[1], var_map, pool, stmts, memo, counter)?;
                Ok(format!("({l}) == ({r})"))
            }
            PredicateKind::Ne => {
                let l = emit_expr_inner(args[0], var_map, pool, stmts, memo, counter)?;
                let r = emit_expr_inner(args[1], var_map, pool, stmts, memo, counter)?;
                Ok(format!("({l}) != ({r})"))
            }
            PredicateKind::Not => {
                let inner = emit_predicate_inner(args[0], var_map, pool, stmts, memo, counter)?;
                Ok(format!("!({inner})"))
            }
            PredicateKind::And => {
                let parts: Result<Vec<_>, _> = args
                    .iter()
                    .map(|&a| emit_predicate_inner(a, var_map, pool, stmts, memo, counter))
                    .collect();
                Ok(format!("({})", parts?.join(" && ")))
            }
            PredicateKind::Or => {
                let parts: Result<Vec<_>, _> = args
                    .iter()
                    .map(|&a| emit_predicate_inner(a, var_map, pool, stmts, memo, counter))
                    .collect();
                Ok(format!("({})", parts?.join(" || ")))
            }
        },
        _ => Err(EmitCError::UnsupportedNode(
            "expected a Predicate node in condition position".to_string(),
        )),
    }
}

/// Emit a C function that evaluates the symbolic expression `expr`.
///
/// Unlike [`emit_horner_c`], this function supports arbitrary expressions
/// including transcendental functions (`sin`, `cos`, `exp`, `log`, `sqrt`,
/// `tan`, `atan2`, `erf`, …) — not just polynomials.  The emitted code
/// `#include <math.h>` is required at the call site.
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

    let mut var_map: HashMap<ExprId, &str> = HashMap::new();
    for (&id, &name) in vars.iter().zip(var_names.iter()) {
        var_map.insert(id, name);
    }

    let mut stmts: Vec<String> = Vec::new();
    let mut memo: HashMap<ExprId, String> = HashMap::new();
    let mut counter = 0usize;

    let result = emit_expr_inner(expr, &var_map, pool, &mut stmts, &mut memo, &mut counter)?;

    // Build the parameter list.
    let params: Vec<String> = var_names.iter().map(|n| format!("double {n}")).collect();
    let params_str = params.join(", ");

    // Build function body.
    let body = if stmts.is_empty() {
        format!("    return {result};\n")
    } else {
        let mut body = stmts.join("\n");
        body.push('\n');
        body.push_str(&format!("    return {result};\n"));
        body
    };

    Ok(format!("double {fn_name}({params_str}) {{\n{body}}}\n"))
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

    let mut var_map: HashMap<ExprId, &str> = HashMap::new();
    for (&id, &name) in vars.iter().zip(var_names.iter()) {
        var_map.insert(id, name);
    }

    let mut stmts: Vec<String> = Vec::new();
    let mut memo: HashMap<ExprId, String> = HashMap::new();
    let mut counter = 0usize;

    // Emit each output component; collect result expressions.
    let mut result_exprs: Vec<String> = Vec::with_capacity(exprs.len());
    for &e in exprs {
        result_exprs.push(emit_expr_inner(
            e,
            &var_map,
            pool,
            &mut stmts,
            &mut memo,
            &mut counter,
        )?);
    }

    // Build the parameter list: inputs then output pointer.
    let mut params: Vec<String> = var_names.iter().map(|n| format!("double {n}")).collect();
    params.push("double *out".to_string());
    let params_str = params.join(", ");

    // Build assignments into out[i].
    let assignments: Vec<String> = result_exprs
        .iter()
        .enumerate()
        .map(|(i, r)| format!("    out[{i}] = {r};"))
        .collect();

    // Build function body.
    let body = if stmts.is_empty() {
        format!("{}\n", assignments.join("\n"))
    } else {
        format!("{}\n{}\n", stmts.join("\n"), assignments.join("\n"))
    };

    Ok(format!("void {fn_name}({params_str}) {{\n{body}}}\n"))
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
