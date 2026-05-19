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

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Emit the Lean 4 file header.
pub fn emit_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic\n\
     import Mathlib.Analysis.SpecialFunctions.Log.Basic\n\
     import Mathlib.MeasureTheory.Integral.IntervalIntegral\n\
     \n\
     open Real MeasureTheory\n\n"
        .to_string()
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

// ---------------------------------------------------------------------------
// Step emission
// ---------------------------------------------------------------------------

/// Emit the Lean proof for a single [`RewriteStep`].
///
/// Returns a complete `example` statement with a tactic proof.
pub fn emit_step(step: &RewriteStep, pool: &ExprPool) -> String {
    let goal = emit_goal(step.before, step.after, pool);
    let tactic = rule_to_tactic(step.rule_name);
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
    let mut out = emit_header();

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
        out.push_str(&emit_step(step, pool));
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
}
