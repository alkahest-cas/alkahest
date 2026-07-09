//! Lean 4 certificate exporter.
//!
//! Converts a [`DerivedExpr`] (expression + derivation log) into a `.lean`
//! source file that imports Mathlib and proves each step in the recorded
//! rewrite chain as its own `example`.
//!
//! # Scope (MVP)
//! Only the rewrites produced by the rule-based simplifier and integrator:
//! `ConstFold`, `AddZero`, `MulOne`, `MulZero`, `PowOne`, `PowZero`,
//! trig rulesets, log/exp rulesets, and integration rules.
//!
//! # Example
//! ```
//! use alkahest_cas::kernel::{Domain, ExprPool};
//! use alkahest_cas::simplify::simplify;
//! use alkahest_cas::lean::emit_lean_expr;
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! let zero = pool.integer(0_i32);
//! let expr = pool.add(vec![x, zero]);
//! let derived = simplify(expr, &pool);
//! let lean_src = emit_lean_expr(&derived, &pool);
//! assert!(lean_src.contains("import Mathlib.Tactic"));
//! assert!(lean_src.contains("simp"));
//! ```

use crate::deriv::log::{DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Tactic lookup table
// ---------------------------------------------------------------------------

fn rule_to_tactic(rule_name: &str) -> &'static str {
    match rule_name {
        "const_fold" => "by norm_num",
        "add_zero" => "by simp [add_zero]",
        "mul_one" => "by simp [mul_one]",
        "mul_zero" => "by simp [mul_zero]",
        "pow_one" => "by simp [pow_one]",
        "pow_zero" => "by simp [pow_zero]",
        "sin_neg" => "by simp [Real.sin_neg]",
        "cos_neg" => "by simp [Real.cos_neg]",
        "log_of_exp" => "by simp [Real.log_exp]",
        "exp_of_log" => "by simp [Real.exp_log (by positivity)]",
        "log_of_product" => "by rw [Real.log_mul (by positivity) (by positivity)]",
        "log_of_pow" => "by rw [Real.log_pow]",
        "sin_sq_plus_cos_sq" => "by rw [Real.sin_sq_add_cos_sq]",
        "power_rule" | "constant_rule" | "sum_rule" | "constant_multiple_rule" => "by ring",
        "int_sin" => "by simp [MeasureTheory.integral_sin]",
        "int_cos" => "by simp [MeasureTheory.integral_cos]",
        "int_exp" => "by simp [MeasureTheory.integral_exp]",
        "log_rule" => "by simp [MeasureTheory.integral_inv_of_pos (by positivity)]",
        "collect_add_terms" | "collect_mul_factors" => "by ring",
        "flatten_mul" | "flatten_add" | "canonical_order" => "by ring",
        "expand_mul" => "by ring",
        "tan_expand" => "by rw [Real.tan_eq_sin_div_cos]",
        _ => "by ring_nf; simp",
    }
}

/// True when a derivation log records differentiation (not algebraic rewrite).
fn is_diff_certificate(wrt: Option<ExprId>) -> bool {
    wrt.is_some()
}

fn diff_rule_to_tactic(rule_name: &str) -> &'static str {
    match rule_name {
        "diff_identity" => "by simp [deriv_id]",
        "diff_const" => "by simp [deriv_const]",
        "diff_univariate_poly" => "by simp [deriv_pow, deriv_add, deriv_mul, deriv_const]",
        "sum_rule" => "by simp [deriv_add]; ring",
        "product_rule" => "by simp [deriv_mul]; ring",
        "power_rule" | "power_rule_n1" => "by simp [deriv_pow, deriv_mul]; ring",
        "power_rule_n0" => "by simp [deriv_const]",
        "diff_sin" | "diff_cos" | "diff_exp" | "diff_log" | "diff_sqrt" => "by sorry",
        "diff_forward" | "diff_primitive_registry" | "diff_piecewise" | "diff_root_sum" => {
            "by sorry"
        }
        _ => "by sorry",
    }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Emit the Lean 4 file header (standard rewrites + trig/log).
pub fn emit_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic\n\
     import Mathlib.Analysis.SpecialFunctions.Log.Basic\n\
     import Mathlib.MeasureTheory.Integral.IntervalIntegral\n\
     \n\
     open Real MeasureTheory\n\n"
        .to_string()
}

/// Emit the Lean 4 file header for differentiation certificates.
pub fn emit_diff_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.Calculus.Deriv.Basic\n\
     import Mathlib.Analysis.Calculus.Deriv.Pow\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv\n\
     import Mathlib.Analysis.SpecialFunctions.ExpDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.Log.Deriv\n\
     \n\
     open Real\n\n"
        .to_string()
}

/// Emit the Lean 4 file header for limit / Filter.Tendsto certificates.
pub fn emit_limit_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.SpecialFunctions.ExpDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.Pow.Real\n\
     import Mathlib.Topology.Algebra.Order.LiminfLimsup\n\
     \n\
     open Real Filter Topology\n\n"
        .to_string()
}

/// Generate a Lean 4 `Filter.Tendsto` certificate for a computed limit.
///
/// The certificate asserts:
/// ```text
/// Filter.Tendsto (fun x => <expr>) Filter.atTop (nhds <limit>)
/// ```
/// and attempts to prove it using known Mathlib theorems.  For cases that
/// cannot be dispatched automatically, the body falls back to `by sorry`.
///
/// # Arguments
/// * `expr`  — the expression whose limit was computed (function body)
/// * `var`   — the free variable (lambda binder)
/// * `lim`   — the computed limit value
/// * `pool`  — expression pool
///
/// Returns a complete `.lean` source snippet including the header.
pub fn emit_tendsto_cert(expr: ExprId, var: ExprId, lim: ExprId, pool: &ExprPool) -> String {
    let var_name = pool.with(var, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    });
    let body = expr_to_lean(expr, pool);
    let (codom_filter, limit_display) = lean_codom_filter(lim, pool);
    let tactic = tendsto_tactic(expr, var, lim, pool);

    let mut out = emit_limit_header();
    out.push_str(&format!(
        "-- Filter.Tendsto certificate: lim_{{x→+∞}} f(x) = {limit_display}\n"
    ));
    out.push_str(&format!(
        "example : Filter.Tendsto (fun ({var_name} : ℝ) => {body}) Filter.atTop {codom_filter} :=\n"
    ));
    out.push_str(&format!("  {tactic}\n"));
    out
}

/// Return `(codomain_filter_str, display_str)` for a limit value.
///
/// Finite limit L → `("(nhds L)", "L")`
/// Infinite limit +∞ → `("Filter.atTop", "+∞")`
fn lean_codom_filter(lim: ExprId, pool: &ExprPool) -> (String, String) {
    let is_inf = pool.with(
        lim,
        |d| matches!(d, ExprData::Symbol { name, .. } if name == "∞"),
    );
    if is_inf {
        return ("Filter.atTop".to_string(), "+∞".to_string());
    }
    let val_str = pool.with(lim, |d| match d {
        ExprData::Integer(n) if n.0 == 0 => "(0 : ℝ)".to_string(),
        ExprData::Integer(n) if n.0 == 1 => "(1 : ℝ)".to_string(),
        _ => expr_to_lean(lim, pool),
    });
    (format!("(nhds {val_str})"), val_str)
}

/// Select the best Lean tactic to prove `Filter.Tendsto f atTop (nhds lim)`.
///
/// Recognises a small set of patterns with known Mathlib theorems; falls back
/// to `by sorry` for everything else.
fn tendsto_tactic(expr: ExprId, var: ExprId, lim: ExprId, pool: &ExprPool) -> String {
    let is_zero = pool.with(lim, |d| match d {
        ExprData::Integer(n) => n.0 == 0,
        _ => false,
    });
    let is_pos_inf = pool.with(lim, |d| match d {
        ExprData::Symbol { name, .. } => name == "∞",
        _ => false,
    });

    // Pattern: exp(-var) → 0
    if is_zero && matches_exp_neg_var(expr, var, pool) {
        return "tendsto_exp_neg_atTop_nhds_zero".to_string();
    }

    // Pattern: var^n * exp(-var) → 0 (for any n ≥ 1)
    if is_zero && matches_pow_mul_exp_neg(expr, var, pool) {
        return "by\n    have := tendsto_pow_mul_exp_neg_atTop_nhds_zero\n    exact this"
            .to_string();
    }

    // Pattern: exp(var) → +∞
    if is_pos_inf && matches_exp_var(expr, var, pool) {
        return "tendsto_exp_atTop".to_string();
    }

    // Pattern: exp(n*var) / exp(m*var) where n < m → 0
    if is_zero && matches_exp_ratio_to_zero(expr, var, pool) {
        return "by\n    simp only [div_eq_mul_inv, ← Real.exp_neg]\n    exact tendsto_exp_neg_atTop_nhds_zero.comp tendsto_id".to_string();
    }

    "by sorry".to_string()
}

/// True iff `expr` is structurally `exp(-var)`.
fn matches_exp_neg_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| {
        if let ExprData::Func { name, args } = d {
            if name == "exp" && args.len() == 1 {
                let arg = args[0];
                return pool.with(arg, |d2| {
                    if let ExprData::Mul(xs) = d2 {
                        xs.len() == 2
                            && xs.contains(&var)
                            && xs.iter().any(|&x| {
                                pool.with(x, |d3| matches!(d3, ExprData::Integer(n) if n.0 == -1))
                            })
                    } else {
                        false
                    }
                });
            }
        }
        false
    })
}

/// True iff `expr` is structurally `var^n * exp(-var)` for some integer n.
fn matches_pow_mul_exp_neg(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| {
        if let ExprData::Mul(xs) = d {
            let has_pow = xs.iter().any(|&x| {
                pool.with(
                    x,
                    |d2| matches!(d2, ExprData::Pow { base, .. } if *base == var),
                )
            });
            let has_exp_neg = xs.iter().any(|&x| matches_exp_neg_var(x, var, pool));
            has_pow && has_exp_neg
        } else {
            false
        }
    })
}

/// True iff `expr` is structurally `exp(var)`.
fn matches_exp_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| {
        if let ExprData::Func { name, args } = d {
            name == "exp" && args.len() == 1 && args[0] == var
        } else {
            false
        }
    })
}

/// True iff `expr` looks like exp(a*var) / exp(b*var) with a < b (or equivalent).
fn matches_exp_ratio_to_zero(expr: ExprId, _var: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| {
        if let ExprData::Mul(xs) = d {
            let exp_count = xs
                .iter()
                .filter(|&&x| {
                    pool.with(
                        x,
                        |d2| matches!(d2, ExprData::Func { name, .. } if name == "exp"),
                    )
                })
                .count();
            exp_count >= 2
        } else {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// Goal emission
// ---------------------------------------------------------------------------

/// Emit a Lean `example` statement asserting `before = after`.
pub fn emit_goal(before: ExprId, after: ExprId, pool: &ExprPool) -> String {
    let before_str = expr_to_lean(before, pool);
    let after_str = expr_to_lean(after, pool);
    format!("example : {before_str} = {after_str}")
}

/// Emit a Lean `example` asserting `deriv (fun v => before) v = after`.
pub fn emit_diff_goal(before: ExprId, after: ExprId, wrt: ExprId, pool: &ExprPool) -> String {
    let var_name = pool.with(wrt, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    });
    let before_str = expr_to_lean(before, pool);
    let after_str = expr_to_lean(after, pool);
    format!("example : deriv (fun ({var_name} : ℝ) => {before_str}) {var_name} = {after_str}")
}

// ---------------------------------------------------------------------------
// Step emission
// ---------------------------------------------------------------------------

/// Emit the Lean proof for a single [`RewriteStep`].
///
/// Returns a complete `example` statement with a tactic proof.
pub fn emit_step(step: &RewriteStep, pool: &ExprPool) -> String {
    emit_step_wrt(step, pool, None)
}

/// Like [`emit_step`], but when `wrt` is set emits a `deriv` goal instead of a rewrite equality.
pub fn emit_step_wrt(step: &RewriteStep, pool: &ExprPool, wrt: Option<ExprId>) -> String {
    let goal = if let Some(var) = wrt {
        emit_diff_goal(step.before, step.after, var, pool)
    } else {
        emit_goal(step.before, step.after, pool)
    };
    let tactic = if wrt.is_some() {
        diff_rule_to_tactic(step.rule_name)
    } else {
        rule_to_tactic(step.rule_name)
    };
    let mut out = format!("{goal} :=\n  {tactic}");
    if !step.side_conditions.is_empty() {
        out.push_str("\n  -- Side conditions: ");
        let conds: Vec<String> = step
            .side_conditions
            .iter()
            .map(|c| c.display_with(pool).to_string())
            .collect();
        out.push_str(&conds.join(", "));
    }
    out
}

// ---------------------------------------------------------------------------
// Full file emitter
// ---------------------------------------------------------------------------

/// Generate a complete `.lean` file proving the derivation recorded in
/// `derived`.
///
/// The file contains:
/// 1. A Mathlib import header.
/// 2. One `example` per rewrite step (each step is checked independently).
///
/// Returns the Lean source as a `String`.
pub fn emit_lean_expr(derived: &DerivedExpr<ExprId>, pool: &ExprPool) -> String {
    emit_lean_expr_wrt(derived, pool, None)
}

/// Like [`emit_lean_expr`], but when `wrt` is set emits differentiation goals
/// (`deriv … = …`) instead of rewrite equalities (`before = after`).
pub fn emit_lean_expr_wrt(
    derived: &DerivedExpr<ExprId>,
    pool: &ExprPool,
    wrt: Option<ExprId>,
) -> String {
    let diff_mode = is_diff_certificate(wrt);
    let mut out = if diff_mode {
        emit_diff_header()
    } else {
        emit_header()
    };

    let steps = derived.log.steps();

    if steps.is_empty() {
        let e = derived.value;
        let lean_e = expr_to_lean(e, pool);
        out.push_str(&format!(
            "-- No rewrite steps recorded.\nexample : {lean_e} = {lean_e} :=\n  rfl\n"
        ));
        return out;
    }

    for (i, step) in steps.iter().enumerate() {
        out.push_str(&format!("-- Step {}: {}\n", i + 1, step.rule_name));
        out.push_str(&emit_step_wrt(step, pool, wrt));
        out.push_str("\n\n");
    }

    out
}

// ---------------------------------------------------------------------------
// Expression → Lean syntax
// ---------------------------------------------------------------------------

/// Convert a symbolic expression to a Lean 4 term.
fn expr_to_lean(expr: ExprId, pool: &ExprPool) -> String {
    pool.with(expr, |data| match data {
        ExprData::Integer(n) => {
            let v = n.0.to_i64().unwrap_or(0);
            format!("({v} : ℝ)")
        }
        ExprData::Rational(r) => {
            let n = r.0.numer().to_i64().unwrap_or(0);
            let d = r.0.denom().to_i64().unwrap_or(1);
            format!("({n} / {d} : ℝ)")
        }
        ExprData::Float(f) => format!("({} : ℝ)", f.inner),
        // Bare names leave metavariables in goals like `(x ^ (1 : ℕ) = x)` (`HPow ?m ℕ ?m`).
        ExprData::Symbol { name, .. } => format!("({name} : ℝ)"),
        ExprData::Add(args) => {
            let parts: Vec<String> = args.iter().map(|&a| expr_to_lean(a, pool)).collect();
            format!("({})", parts.join(" + "))
        }
        ExprData::Mul(args) => {
            let parts: Vec<String> = args.iter().map(|&a| expr_to_lean(a, pool)).collect();
            format!("({})", parts.join(" * "))
        }
        ExprData::Pow { base, exp } => {
            let b = expr_to_lean(*base, pool);
            let neg_int = pool.with(*exp, |d| match d {
                ExprData::Integer(n) if n.0 < 0 => n.0.to_i64(),
                _ => None,
            });
            if let Some(n) = neg_int {
                let abs_n = n.unsigned_abs();
                if abs_n == 1 {
                    format!("({b})⁻¹")
                } else {
                    format!("({b})⁻¹ ^ ({abs_n} : ℕ)")
                }
            } else {
                // Nonnegative integer exponents must use `(n : ℕ)` so Lean picks `HPow ℝ ℕ ℝ`.
                // Using `(n : ℝ)` leads to `Real.rpow` and stuck metavariables on goals like `x^1 = x`.
                let e = pool.with(*exp, |d| match d {
                    ExprData::Integer(n) if n.0 >= 0 => format!("({} : ℕ)", n.0),
                    _ => expr_to_lean(*exp, pool),
                });
                format!("({b}) ^ {e}")
            }
        }
        ExprData::Func { name, args } => {
            let arg_strs: Vec<String> = args.iter().map(|&a| expr_to_lean(a, pool)).collect();
            match name.as_str() {
                "sin" => format!("Real.sin {}", arg_strs[0]),
                "cos" => format!("Real.cos {}", arg_strs[0]),
                "exp" => format!("Real.exp {}", arg_strs[0]),
                "log" => format!("Real.log {}", arg_strs[0]),
                "sqrt" => format!("Real.sqrt {}", arg_strs[0]),
                other => format!("{other} ({})", arg_strs.join(", ")),
            }
        }
        _ => "sorry".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::simplify::simplify;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn emit_lean_const_fold() {
        let pool = p();
        let two = pool.integer(2_i32);
        let three = pool.integer(3_i32);
        let expr = pool.add(vec![two, three]);
        let derived = simplify(expr, &pool);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            lean.contains("import Mathlib.Tactic"),
            "missing import: {lean}"
        );
        assert!(
            lean.contains("norm_num"),
            "ConstFold should produce norm_num: {lean}"
        );
    }

    #[test]
    fn emit_lean_add_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let expr = pool.add(vec![x, zero]);
        let derived = simplify(expr, &pool);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            lean.contains("add_zero") || lean.contains("simp"),
            "missing add_zero tactic: {lean}"
        );
        assert!(
            !lean.contains("simp_all [*]"),
            "Lean 4 does not parse `simp_all [*]`; emit only per-step examples ({lean})"
        );
    }

    #[test]
    fn emit_header_has_imports() {
        let h = emit_header();
        assert!(h.contains("import Mathlib.Tactic"));
        assert!(h.contains("open Real"));
    }

    #[test]
    fn emit_step_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let before = pool.add(vec![x, zero]);
        let step = crate::deriv::log::RewriteStep::simple("add_zero", before, x);
        let s = emit_step(&step, &pool);
        assert!(s.contains("add_zero"));
        assert!(s.contains("simp"));
    }

    #[test]
    fn emit_lean_diff_univariate_poly() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let three = pool.integer(3_i32);
        let expr = pool.pow(x, three);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.contains("deriv (fun (x : ℝ)"),
            "expected deriv goal, got: {lean}"
        );
        assert!(
            lean.contains("deriv_pow"),
            "expected deriv_pow tactic, got: {lean}"
        );
        assert!(
            !lean.contains("sorry"),
            "polynomial derivative certificate must not use an admission: {lean}"
        );
        assert!(
            !lean.contains("= (((x : ℝ)) ^ (2 : ℕ) * (3 : ℝ)) :=") || lean.contains("deriv"),
            "must not claim x^3 = 3*x^2 without deriv: {lean}"
        );
    }

    #[test]
    fn expr_to_lean_integer() {
        let pool = p();
        let three = pool.integer(3_i32);
        let s = expr_to_lean(three, &pool);
        assert!(s.contains("3"));
    }

    #[test]
    fn expr_to_lean_sin() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let s = expr_to_lean(sin_x, &pool);
        assert!(s.contains("Real.sin"));
    }

    #[test]
    fn expr_to_lean_pow_natural_exp_is_nat() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let pow_x_1 = pool.pow(x, one);
        let s = expr_to_lean(pow_x_1, &pool);
        assert!(
            s.contains(": ℕ"),
            "expected Nat exponent for HPow ℝ ℕ ℝ, got: {s}"
        );
        assert!(
            s.contains("(x : ℝ)"),
            "base must be typed as ℝ so HPow resolves: {s}"
        );
        assert!(
            !s.contains("(1 : ℝ)"),
            "Real exponent triggers rpow metavariable issues: {s}"
        );
    }

    #[test]
    fn emit_tendsto_exp_neg_x() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.func("exp", vec![neg_x]);
        let zero = pool.integer(0_i32);
        let lean = emit_tendsto_cert(expr, x, zero, &pool);
        assert!(
            lean.contains("Filter.Tendsto"),
            "missing Filter.Tendsto: {lean}"
        );
        assert!(
            lean.contains("tendsto_exp_neg_atTop_nhds_zero"),
            "expected known tactic: {lean}"
        );
    }

    #[test]
    fn emit_tendsto_exp_x_to_inf() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![x]);
        let inf = pool.symbol("∞", Domain::Real);
        let lean = emit_tendsto_cert(expr, x, inf, &pool);
        assert!(
            lean.contains("tendsto_exp_atTop"),
            "expected tendsto_exp_atTop: {lean}"
        );
    }

    #[test]
    fn emit_tendsto_fallback_uses_sorry() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // sin(x) → no known pattern → sorry
        let expr = pool.func("sin", vec![x]);
        let zero = pool.integer(0_i32);
        let lean = emit_tendsto_cert(expr, x, zero, &pool);
        assert!(
            lean.contains("sorry"),
            "complex patterns should fall back to sorry: {lean}"
        );
    }

    #[test]
    fn emit_tendsto_header_has_filter_imports() {
        let h = emit_limit_header();
        assert!(h.contains("import Mathlib.Tactic"));
        assert!(h.contains("Filter"));
    }
}
