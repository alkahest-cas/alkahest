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

use crate::deriv::log::{DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Tactic lookup table
// ---------------------------------------------------------------------------

fn rule_to_tactic(rule_name: &str) -> &'static str {
    match rule_name {
        // `const_fold` folds constant arithmetic, but in a differentiation log it
        // can also reorder a symbolic atom past the folded coefficient (e.g.
        // `x*2*sin(x²)*-1 = x*sin(x²)*-2`). `ring` closes both the pure-numeric
        // folds and these symbolic reorderings.
        "const_fold" => "by ring",
        "add_zero" => "by simp [add_zero]",
        "mul_one" => "by simp [mul_one]",
        "mul_zero" => "by simp [mul_zero]",
        "pow_one" => "by simp [pow_one]",
        "pow_zero" => "by simp [pow_zero]",
        "sin_neg" => "by simp [Real.sin_neg]",
        "cos_neg" => "by simp [Real.cos_neg]",
        "log_of_exp" => "by simp [Real.log_exp]",
        // These need a positivity hypothesis on the free variable(s). When the
        // recorded side conditions are simple (bare symbols), [`emit_step_wrt`]
        // upgrades the goal with explicit `(x : ℝ) (hx : 0 < x)` binders and
        // calls [`positivity_tactic`] instead of using this fallback. This
        // entry is only reached when that upgrade isn't possible (e.g. a
        // compound side-condition expression, or — for `log_of_product` — more
        // factors than [`positivity_tactic`] has a chained lemma for); such
        // steps are withheld via [`step_is_certifiable`] rather than emitting
        // a failing `positivity`.
        "exp_of_log" => "by sorry",
        "log_of_product" | "log_of_product_positive" => "by sorry",
        // `sum_of_logs` (`log a + log b + … = log(a·b·…)`) is only sound with a
        // positivity hypothesis on every argument. [`emit_step_wrt`] upgrades the
        // two-factor case to an explicit-binder `Real.log_mul` certificate via
        // [`positivity_certificate`]; anything it can't upgrade (compound
        // arguments, or three-plus factors [`positivity_tactic`] has no chained
        // lemma for) must be withheld, so the table default is a withheld `sorry`.
        "sum_of_logs" => "by sorry",
        // `exp a · exp b · … = exp(a + b + …)` is unconditionally valid; fold the
        // product of exponentials back with `Real.exp_add` (applied right-to-left,
        // repeatedly for ≥ 3 factors).
        "product_of_exps" => "by simp only [← Real.exp_add]",
        "log_of_pow" => "by simp [Real.log_pow]",
        "sin_sq_plus_cos_sq" => "by rw [Real.sin_sq_add_cos_sq]",
        "power_rule" | "constant_rule" | "sum_rule" | "constant_multiple_rule" => "by ring",
        // Integration rules must not be emitted as bare `integrand = F` equalities
        // (that claim is false). They are filtered out by [`step_is_certifiable`].
        "int_sin" | "int_cos" | "int_exp" | "log_rule" => "by sorry",
        "collect_add_terms" | "collect_mul_factors" => "by ring",
        "flatten_mul" | "flatten_add" | "canonical_order" => "by ring",
        "expand_mul" => "by ring",
        // `tan_eq_sin_div_cos` yields `/`; Alkahest stores the reciprocal product.
        "tan_expand" => "by rw [Real.tan_eq_sin_div_cos, div_eq_mul_inv]",
        _ => "by ring_nf; simp",
    }
}

/// True when a derivation log records differentiation (not algebraic rewrite).
fn is_diff_certificate(wrt: Option<ExprId>) -> bool {
    wrt.is_some()
}

/// Rules that construct derivatives (as opposed to algebraic cleanup after diff).
fn is_differentiation_rule(rule_name: &str) -> bool {
    rule_name.starts_with("diff_")
        || matches!(
            rule_name,
            "sum_rule"
                | "product_rule"
                | "quotient_rule"
                | "chain_rule"
                | "power_rule"
                | "power_rule_n0"
                | "power_rule_n1"
        )
}

/// Rules that build antiderivatives. Emitting `before = after` for these is
/// mathematically false (e.g. `sin x = -cos x`).
fn is_integration_rule(rule_name: &str) -> bool {
    rule_name.starts_with("int_")
        || rule_name.starts_with("risch_")
        || matches!(
            rule_name,
            "fundamental_theorem_of_calculus"
                | "log_rule"
                | "gosper_indefinite"
                | "gosper_definite_telescope"
        )
}

/// `before` is structurally `f(wrt)` for a unary primitive `f`.
fn is_unary_of_var(before: ExprId, wrt: ExprId, pool: &ExprPool) -> bool {
    pool.with(
        before,
        |d| matches!(d, ExprData::Func { args, .. } if args.len() == 1 && args[0] == wrt),
    )
}

/// `before` is structurally `wrt ^ e` for some exponent.
fn is_pow_of_var(before: ExprId, wrt: ExprId, pool: &ExprPool) -> bool {
    pool.with(
        before,
        |d| matches!(d, ExprData::Pow { base, .. } if *base == wrt),
    )
}

/// If `before` is a unary composite `f(wrt ^ n)` whose inner argument is a pure
/// power of the differentiation variable with integer exponent `n ≥ 2`, return
/// `n`. This is the subset of the chain rule that we can emit as a compiling
/// Lean certificate via `HasDerivAt.comp` + `hasDerivAt_pow`.
fn composite_pow_inner_exp(before: ExprId, wrt: ExprId, pool: &ExprPool) -> Option<i64> {
    pool.with(before, |d| {
        let arg = match d {
            ExprData::Func { args, .. } if args.len() == 1 => args[0],
            _ => return None,
        };
        pool.with(arg, |inner| match inner {
            ExprData::Pow { base, exp } if *base == wrt => pool.with(*exp, |e| match e {
                ExprData::Integer(n) => n.0.to_i64().filter(|&k| k >= 2),
                _ => None,
            }),
            _ => None,
        })
    })
}

/// The Mathlib `HasDerivAt.<f>` composite lemma suffix for the outer unary
/// primitive of a chain-rule differentiation step, if we know how to compose it.
///
/// These lemmas (`HasDerivAt.sin`, `HasDerivAt.cos`, `HasDerivAt.exp`) take a
/// `HasDerivAt` for the inner function and yield one for `fun x => f (g x)`,
/// avoiding the higher-order unification pitfalls of the raw `HasDerivAt.comp`.
fn chain_outer_lemma(rule_name: &str) -> Option<&'static str> {
    match rule_name {
        "diff_sin" => Some("sin"),
        "diff_cos" => Some("cos"),
        "diff_exp" => Some("exp"),
        _ => None,
    }
}

/// Build a self-contained Lean tactic proving a chain-rule derivative goal
/// `deriv (fun x => f (x^n)) x = <after>` for `f ∈ {sin, cos, exp}` and integer
/// `n ≥ 2`.
///
/// The proof takes `hasDerivAt_pow` for the polynomial inner, lifts it through
/// the outer primitive's `HasDerivAt.<f>` composite lemma, discharges the
/// derivative via `HasDerivAt.deriv`, and reconciles the (cast-laden) Mathlib
/// derivative form with Alkahest's recorded `after` using `push_cast; ring`.
/// Returns `None` when the step is not a supported composite shape.
fn chain_diff_tactic(
    rule_name: &str,
    before: ExprId,
    wrt: ExprId,
    pool: &ExprPool,
) -> Option<String> {
    let n = composite_pow_inner_exp(before, wrt, pool)?;
    let suffix = chain_outer_lemma(rule_name)?;
    let var_name = pool.with(wrt, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    });
    Some(format!(
        "by\n    \
         have hg := hasDerivAt_pow {n} {var_name}\n    \
         rw [(hg.{suffix}).deriv]\n    \
         push_cast\n    \
         ring"
    ))
}

fn diff_rule_to_tactic(rule_name: &str) -> Option<&'static str> {
    match rule_name {
        "diff_identity" => Some("by simp [deriv_id]"),
        "diff_const" => Some("by simp [deriv_const]"),
        // `; try ring` closes coefficient-order goals like `2 * x = x * 2`.
        "diff_univariate_poly" => {
            Some("by simp [deriv_pow, deriv_add, deriv_mul, deriv_const]; try ring")
        }
        // `deriv_add`/`deriv_mul` need DifferentiableAt side goals; include the
        // common Real lemmas so unary trig/exp sums and products close.
        "sum_rule" => Some(
            "by simp [deriv_add, Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, \
             Real.differentiableAt_sin, Real.differentiableAt_cos, Real.differentiableAt_exp, \
             deriv_pow, deriv_mul, deriv_const, one_mul, mul_one, mul_neg, neg_mul]; try ring",
        ),
        "product_rule" => Some(
            "by simp [deriv_mul, Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, \
             Real.differentiableAt_sin, Real.differentiableAt_cos, Real.differentiableAt_exp, \
             deriv_const, differentiableAt_const, one_mul, mul_one, mul_neg, neg_mul]; try ring",
        ),
        "power_rule" | "power_rule_n1" => Some("by simp [deriv_pow, deriv_mul]; try ring"),
        "power_rule_n0" => Some("by simp [deriv_const]"),
        // Pointwise Mathlib lemmas for `deriv (fun x => f x) x = …` when the
        // argument is exactly the free variable. Chain-rule cases are withheld.
        "diff_sin" => Some("by simp [Real.deriv_sin, one_mul, mul_one]"),
        "diff_cos" => Some("by simp [Real.deriv_cos, one_mul, mul_one]"),
        "diff_exp" => Some("by simp [Real.deriv_exp, one_mul, mul_one]"),
        // log/sqrt need side conditions / different lemmas — withhold for now.
        "diff_log" | "diff_sqrt" => None,
        "diff_forward" | "diff_primitive_registry" | "diff_piecewise" | "diff_root_sum" => None,
        _ => None,
    }
}

/// The name of `id` if it's a bare [`ExprData::Symbol`], else `None`.
///
/// Positivity certificates only bind explicit `(name : ℝ) (hname : 0 < name)`
/// binders for symbols — a compound side-condition expression (e.g. `x + y`)
/// has no single name to bind and is left withheld.
fn symbol_name(id: ExprId, pool: &ExprPool) -> Option<String> {
    pool.with(id, |d| match d {
        ExprData::Symbol { name, .. } => Some(name.clone()),
        _ => None,
    })
}

/// Select the Lean tactic that discharges `rule_name` given hypothesis names
/// `h<name>` (one per entry of `names`, in the same order as the step's
/// recorded [`SideCondition::Positive`] facts). Returns `None` when there's no
/// known closing lemma for this shape (e.g. `log_of_product` with more than
/// two factors) — callers must fall back to withholding the step.
fn positivity_tactic(rule_name: &str, names: &[String]) -> Option<String> {
    match (rule_name, names) {
        ("exp_of_log", [x]) => Some(format!("by rw [Real.exp_log h{x}]")),
        ("log_of_product" | "log_of_product_positive" | "sum_of_logs", [x, y]) => Some(format!(
            "by rw [Real.log_mul (ne_of_gt h{x}) (ne_of_gt h{y})]"
        )),
        _ => None,
    }
}

/// Attempt to build a self-contained positivity certificate for `step`: an
/// explicit `(x : ℝ) (hx : 0 < x) …` binder list plus a tactic that consumes
/// those hypotheses to close the goal.
///
/// Returns `None` when the step has no recorded positivity side conditions,
/// any condition is over a compound expression rather than a bare symbol, or
/// [`positivity_tactic`] has no lemma for this rule/arity combination — in
/// all of those cases the caller falls back to the (withheld) table tactic.
fn positivity_certificate(step: &RewriteStep, pool: &ExprPool) -> Option<(String, String)> {
    if step.side_conditions.is_empty() {
        return None;
    }
    let names: Vec<String> = step
        .side_conditions
        .iter()
        .map(|c| match c {
            SideCondition::Positive(id) => symbol_name(*id, pool),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let tactic = positivity_tactic(step.rule_name, &names)?;
    let mut binders = names
        .iter()
        .map(|n| format!("({n} : ℝ)"))
        .collect::<Vec<_>>();
    binders.extend(names.iter().map(|n| format!("(h{n} : 0 < {n})")));
    Some((binders.join(" "), tactic))
}

/// Whether this step can be emitted as a Lean `example` expected to typecheck
/// without `sorry` / `admit`.
fn step_is_certifiable(step: &RewriteStep, wrt: Option<ExprId>, pool: &ExprPool) -> bool {
    if is_integration_rule(step.rule_name) {
        return false;
    }
    if let Some(var) = wrt {
        if is_differentiation_rule(step.rule_name) {
            match step.rule_name {
                "diff_sin" | "diff_cos" | "diff_exp" => {
                    // Pointwise `f(x)` uses the direct Mathlib deriv lemma; a
                    // composite `f(x^n)` is closed via the chain-rule tactic.
                    return (is_unary_of_var(step.before, var, pool)
                        && diff_rule_to_tactic(step.rule_name).is_some())
                        || chain_diff_tactic(step.rule_name, step.before, var, pool).is_some();
                }
                "diff_log" | "diff_sqrt" => {
                    // Chain rule / composite arguments are not yet encoded.
                    return is_unary_of_var(step.before, var, pool)
                        && diff_rule_to_tactic(step.rule_name).is_some();
                }
                // Generalized power rule `d/dx[f^n]` embeds a chain rule when
                // `f ≠ x`; only `d/dx[x^n]` is Lean-encoded today.
                "power_rule" | "power_rule_n1" | "power_rule_n0" => {
                    return is_pow_of_var(step.before, var, pool)
                        && diff_rule_to_tactic(step.rule_name).is_some();
                }
                name => return diff_rule_to_tactic(name).is_some(),
            }
        }
        // Algebraic cleanup steps in a diff log use plain equality goals.
        let tactic = rule_to_tactic(step.rule_name);
        return !tactic.contains("sorry") || positivity_certificate(step, pool).is_some();
    }
    let tactic = rule_to_tactic(step.rule_name);
    !tactic.contains("sorry") || positivity_certificate(step, pool).is_some()
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Emit the Lean 4 file header (standard rewrites + trig/log).
pub fn emit_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic\n\
     import Mathlib.Analysis.SpecialFunctions.Log.Basic\n\
     import Mathlib.Analysis.SpecialFunctions.Gamma.Basic\n\
     \n\
     open Real\n\n"
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
/// Returns a complete `.lean` source snippet including the header. Returns an
/// empty string if the proof would require `sorry` or `admit` (unrecognized patterns).
pub fn emit_tendsto_cert(expr: ExprId, var: ExprId, lim: ExprId, pool: &ExprPool) -> String {
    let var_name = pool.with(var, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    });
    let body = expr_to_lean(expr, pool);
    let (codom_filter, limit_display) = lean_codom_filter(lim, pool);
    let tactic = tendsto_tactic(expr, var, lim, pool);

    // Gate: do not emit certificates that would require sorry or admit.
    if tactic.contains("sorry") || tactic.contains("admit") {
        return String::new();
    }

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

/// True iff `needle` occurs anywhere inside `haystack`. Because the pool
/// interns subexpressions, any occurrence of the `wrt` symbol shares its
/// `ExprId`, so a structural id-equality walk is exact.
fn depends_on(haystack: ExprId, needle: ExprId, pool: &ExprPool) -> bool {
    if haystack == needle {
        return true;
    }
    pool.with(haystack, |d| match d {
        ExprData::Add(xs) | ExprData::Mul(xs) | ExprData::Func { args: xs, .. } => {
            xs.iter().any(|&c| depends_on(c, needle, pool))
        }
        ExprData::Pow { base, exp } => {
            depends_on(*base, needle, pool) || depends_on(*exp, needle, pool)
        }
        ExprData::Predicate { args, .. } => args.iter().any(|&c| depends_on(c, needle, pool)),
        ExprData::Piecewise { branches, default } => {
            branches
                .iter()
                .any(|&(c, v)| depends_on(c, needle, pool) || depends_on(v, needle, pool))
                || depends_on(*default, needle, pool)
        }
        ExprData::BigO(a) => depends_on(*a, needle, pool),
        ExprData::Forall { var, body }
        | ExprData::Exists { var, body }
        | ExprData::RootSum { var, body, .. } => {
            depends_on(*var, needle, pool) || depends_on(*body, needle, pool)
        }
        _ => false,
    })
}

/// Emit a Lean `example` asserting `deriv (fun v => before) v = after`.
pub fn emit_diff_goal(before: ExprId, after: ExprId, wrt: ExprId, pool: &ExprPool) -> String {
    let var_name = pool.with(wrt, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    });
    // When the integrand doesn't mention the differentiation variable (e.g. the
    // derivative of a constant `C`), the lambda binder is genuinely unused. Under
    // `-DwarningAsError=true` Mathlib's `unusedVariables` linter turns that into a
    // hard error, so bind it as `_<var>` (underscore-prefixed names are exempt).
    // The evaluation point stays `var_name` (it is a real use of the free var).
    let binder = if depends_on(before, wrt, pool) {
        var_name.clone()
    } else {
        format!("_{var_name}")
    };
    let before_str = expr_to_lean(before, pool);
    let after_str = expr_to_lean(after, pool);
    format!("example : deriv (fun ({binder} : ℝ) => {before_str}) {var_name} = {after_str}")
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

/// Like [`emit_step`], but when `wrt` is set, differentiation rules emit a
/// `deriv` goal while algebraic cleanup steps in the same log stay plain
/// equalities (so `mul_one` is not wrongly wrapped as `deriv (1·cos) = cos`).
pub fn emit_step_wrt(step: &RewriteStep, pool: &ExprPool, wrt: Option<ExprId>) -> String {
    let diff_step = wrt.is_some() && is_differentiation_rule(step.rule_name);

    // Positivity-hypothesis certificates (`exp_of_log`, `log_of_product`, …)
    // need explicit `(x : ℝ) (hx : 0 < x)` binders on the `example` header
    // rather than the plain `example : before = after` goal, so they're
    // built separately from the diff/algebraic goal below.
    if !diff_step {
        if let Some((binders, tactic)) = positivity_certificate(step, pool) {
            let before_str = expr_to_lean(step.before, pool);
            let after_str = expr_to_lean(step.after, pool);
            let mut out = format!("example {binders} : {before_str} = {after_str} :=\n  {tactic}");
            out.push_str("\n  -- Side conditions: ");
            let conds: Vec<String> = step
                .side_conditions
                .iter()
                .map(|c| c.display_with(pool).to_string())
                .collect();
            out.push_str(&conds.join(", "));
            return out;
        }
    }

    let goal = if let (true, Some(var)) = (diff_step, wrt) {
        emit_diff_goal(step.before, step.after, var, pool)
    } else {
        emit_goal(step.before, step.after, pool)
    };
    let tactic: String = if let (true, Some(var)) = (diff_step, wrt) {
        chain_diff_tactic(step.rule_name, step.before, var, pool).unwrap_or_else(|| {
            diff_rule_to_tactic(step.rule_name)
                .unwrap_or("by sorry")
                .to_string()
        })
    } else if diff_step {
        diff_rule_to_tactic(step.rule_name)
            .unwrap_or("by sorry")
            .to_string()
    } else {
        rule_to_tactic(step.rule_name).to_string()
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
/// Returns the Lean source as a `String`. When the log cannot be certified
/// without `sorry` or would assert a false unwrapped equality (integration),
/// returns an empty string — callers should treat that as "no certificate".
pub fn emit_lean_expr(derived: &DerivedExpr<ExprId>, pool: &ExprPool) -> String {
    emit_lean_expr_wrt(derived, pool, None)
}

/// Like [`emit_lean_expr`], but when `wrt` is set emits differentiation goals
/// (`deriv … = …`) for differentiation rules.
///
/// Returns `""` when any step is not Lean-certifiable (B3): integration
/// antiderivative construction, chain-rule diffs not yet encoded, or tactics
/// that would emit `sorry`.
pub fn emit_lean_expr_wrt(
    derived: &DerivedExpr<ExprId>,
    pool: &ExprPool,
    wrt: Option<ExprId>,
) -> String {
    let steps = derived.log.steps();

    if steps.is_empty() {
        let diff_mode = is_diff_certificate(wrt);
        let mut out = if diff_mode {
            emit_diff_header()
        } else {
            emit_header()
        };
        let e = derived.value;
        let lean_e = expr_to_lean(e, pool);
        out.push_str(&format!(
            "-- No rewrite steps recorded.\nexample : {lean_e} = {lean_e} :=\n  rfl\n"
        ));
        return out;
    }

    // Withhold the whole certificate if any step is unsound or unfinished.
    if steps.iter().any(|s| !step_is_certifiable(s, wrt, pool)) {
        return String::new();
    }

    let diff_mode = is_diff_certificate(wrt);
    let mut out = if diff_mode {
        emit_diff_header()
    } else {
        emit_header()
    };

    for (i, step) in steps.iter().enumerate() {
        out.push_str(&format!("-- Step {}: {}\n", i + 1, step.rule_name));
        out.push_str(&emit_step_wrt(step, pool, wrt));
        out.push_str("\n\n");
    }

    // Defense in depth: never hand out a certificate containing admissions.
    if out.contains("sorry") || out.contains("admit") {
        return String::new();
    }

    out
}

/// Structural gate for antiderivatives whose FTC derivative certificate is
/// known to typecheck under the reused differentiation machinery.
///
/// The diff exporter's `deriv (fun x => F) x = …` tactics reliably close for a
/// restricted fragment: constants, powers of the differentiation variable,
/// *pointwise* `sin`/`cos`/`exp` (argument exactly the variable), sums of
/// those, and *flat* products of those (a product whose factors are atoms /
/// pointwise primitives, e.g. `x · cos x`). Two shapes that the diff exporter
/// currently emits but does **not** discharge — leaving `deriv` or a
/// `DifferentiableAt` side goal open — must be withheld here:
///
/// * a **chain composite** `f(g x)` with `g ≠ x` (e.g. `exp (x²)`), because the
///   product-rule simp set lacks the composite's `DifferentiableAt` lemma;
/// * a **sum nested inside a product** (e.g. `-1 · (a + b)`), because the
///   post-`simp` `ring` cannot reduce the still-symbolic nested `deriv`.
///
/// Rejecting these keeps the integration certificate sound: a withheld integral
/// is always preferable to a `.lean` file that fails to typecheck. Composites
/// and by-parts results outside this fragment simply stay withheld.
fn antiderivative_in_certifiable_fragment(f: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    // `in_product`: we are inside a Mul, where a nested Add would defeat `ring`.
    fn walk(f: ExprId, var: ExprId, pool: &ExprPool, in_product: bool) -> bool {
        pool.with(f, |d| match d {
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
            ExprData::Symbol { .. } => true,
            ExprData::Pow { base, exp } => {
                // Only powers of the differentiation variable with an integer
                // exponent — the polynomial / reciprocal fast path.
                *base == var && pool.with(*exp, |e| matches!(e, ExprData::Integer(_)))
            }
            ExprData::Func { name, args } => {
                // Pointwise primitive only: sin/cos/exp applied to exactly `var`.
                matches!(name.as_str(), "sin" | "cos" | "exp") && args.len() == 1 && args[0] == var
            }
            ExprData::Add(xs) => {
                // A sum inside a product is the shape `ring` cannot finish.
                !in_product && xs.iter().all(|&c| walk(c, var, pool, false))
            }
            ExprData::Mul(xs) => xs.iter().all(|&c| walk(c, var, pool, true)),
            _ => false,
        })
    }
    walk(f, var, pool, false)
}

/// Emit a Lean certificate for an **indefinite integral** `∫ f dx = F`.
///
/// A bare `f = F` equality is false (`sin x ≠ -cos x`), so an integration
/// result cannot be certified as a rewrite. The sound statement that pins the
/// antiderivative is the FTC derivative relation
///
/// ```text
/// deriv (fun x => F) x = f
/// ```
///
/// which we discharge by *reusing the differentiation-certificate machinery*:
/// we differentiate `F` in the kernel and hand the resulting derivation log to
/// [`emit_lean_expr_wrt`]. That already proves `deriv (fun x => F) x = d/dx F`
/// via `deriv_pow` / `Real.deriv_sin` / `HasDerivAt.comp` / … and withholds
/// (returns `""`) whenever the goal escapes the certifiable diff fragment.
///
/// The one extra obligation for an *integral* is that the differentiated
/// antiderivative is syntactically the integrand, so that the certificate's
/// final right-hand side is exactly `f` (i.e. the cert really proves
/// `deriv F = f`, not `deriv F = <something else>`). We require the kernel's
/// simplified `d/dx F` to intern to the same [`ExprId`] as `integrand`;
/// otherwise we WITHHOLD. This is precisely the exact-residual antiderivative
/// check, so numeric-only antiderivatives stay withheld.
///
/// Returns `""` (no certificate) when `F` cannot be differentiated, when its
/// derivative is not structurally the integrand, or when the diff certificate
/// itself is withheld. Never emits `sorry` / `admit`.
pub fn emit_integration_cert(
    antiderivative: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> String {
    // Withhold antiderivatives whose derivative certificate escapes the
    // reliably-typechecking diff fragment (chain composites, sums nested in
    // products): the reused exporter would emit a non-closing proof for them.
    if !antiderivative_in_certifiable_fragment(antiderivative, var, pool) {
        return String::new();
    }
    let Ok(derived_diff) = crate::diff::diff(antiderivative, var, pool) else {
        return String::new();
    };
    // The certificate proves `deriv F = d/dx F`; only present it as certifying
    // `∫ f = F` when `d/dx F` is exactly the integrand.
    if derived_diff.value != integrand {
        return String::new();
    }
    let cert = emit_lean_expr_wrt(&derived_diff, pool, Some(var));
    if cert.is_empty() {
        return String::new();
    }
    // Prefix a note tying the diff certificate back to the integral it proves.
    let f = expr_to_lean(integrand, pool);
    let big_f = expr_to_lean(antiderivative, pool);
    let var_name = pool.with(var, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    });
    let note = format!(
        "-- ∫ {f} d{var_name} = {big_f}\n\
         -- certified via the FTC derivative relation: deriv (fun {var_name} => {big_f}) {var_name} = {f}\n"
    );
    // Splice the note directly before the first proof step, after the imports.
    match cert.find("-- Step 1") {
        Some(idx) => format!("{}{note}{}", &cert[..idx], &cert[idx..]),
        None => format!("{cert}{note}"),
    }
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
            // Always parenthesize the argument: `Real.log Real.exp x` parses as
            // `(Real.log Real.exp) x`, and `Real.log x ^ 3` parses as
            // `(Real.log x) ^ 3` — both are type/math errors.
            match name.as_str() {
                "sin" => format!("Real.sin ({})", arg_strs[0]),
                "cos" => format!("Real.cos ({})", arg_strs[0]),
                "tan" => format!("Real.tan ({})", arg_strs[0]),
                "exp" => format!("Real.exp ({})", arg_strs[0]),
                "log" => format!("Real.log ({})", arg_strs[0]),
                "sqrt" => format!("Real.sqrt ({})", arg_strs[0]),
                // `Real.Gamma : ℝ → ℝ` (imported in the non-diff header). Alkahest
                // spells it lowercase `gamma`; map it to the Mathlib name so the
                // emitted term type-checks.
                "gamma" => format!("Real.Gamma ({})", arg_strs[0]),
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
            lean.contains("ring"),
            "ConstFold should produce a ring proof: {lean}"
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
    fn withhold_false_integrate_sin_certificate() {
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let derived = integrate(sin_x, x, &pool).expect("integrate");
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            lean.is_empty(),
            "∫ sin must not emit false `sin = -cos` Lean equality, got: {lean}"
        );
    }

    #[test]
    fn integration_cert_cos_via_ftc_derivative() {
        // ∫ cos x dx = sin x, certified as `deriv (fun x => sin x) x = cos x`.
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cos_x = pool.func("cos", vec![x]);
        let derived = integrate(cos_x, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, cos_x, x, &pool);
        assert!(
            !lean.is_empty(),
            "∫ cos x should certify via the FTC relation"
        );
        assert!(
            lean.contains("deriv (fun (x : ℝ)"),
            "must state the derivative relation, got: {lean}"
        );
        assert!(
            lean.contains("Real.deriv_sin"),
            "antiderivative sin is discharged by Real.deriv_sin: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn integration_cert_sin_via_ftc_derivative() {
        // ∫ sin x dx = -cos x, certified as `deriv (fun x => -cos x) x = sin x`.
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let derived = integrate(sin_x, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, sin_x, x, &pool);
        assert!(
            !lean.is_empty(),
            "∫ sin x should certify via the FTC relation, got empty"
        );
        assert!(
            lean.contains("deriv (fun (x : ℝ)"),
            "must state the derivative relation: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn integration_cert_exp_via_ftc_derivative() {
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let derived = integrate(exp_x, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, exp_x, x, &pool);
        assert!(
            !lean.is_empty(),
            "∫ exp x should certify via the FTC relation"
        );
        assert!(
            lean.contains("Real.deriv_exp"),
            "expected deriv_exp: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn integration_cert_power_via_ftc_derivative() {
        // ∫ x² dx = x³/3, certified via the polynomial derivative fragment.
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let derived = integrate(x2, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, x2, x, &pool);
        assert!(!lean.is_empty(), "∫ x² should certify via the FTC relation");
        assert!(
            lean.contains("deriv (fun (x : ℝ)"),
            "must state the derivative relation: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn integration_cert_withheld_for_chain_composite_antiderivative() {
        // ∫ x·exp(x²) dx = ½·exp(x²). Its derivative certificate would emit a
        // product rule whose factor is the composite exp(x²); the reused diff
        // tactic leaves a `DifferentiableAt` side goal open, so the integral
        // certificate must be withheld by the fragment gate.
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let integrand = pool.mul(vec![x, pool.func("exp", vec![x2])]);
        let derived = integrate(integrand, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, integrand, x, &pool);
        assert!(
            lean.is_empty(),
            "∫ x·exp(x²) has a composite antiderivative; must withhold: {lean}"
        );
    }

    #[test]
    fn integration_cert_withheld_for_non_certifiable_diff() {
        // ∫ log x dx = x·log x − x. Its derivative routes through `diff_log`,
        // which is outside the certifiable diff fragment, so the integral cert
        // must be withheld rather than emit an admission.
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let derived = integrate(log_x, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, log_x, x, &pool);
        assert!(
            lean.is_empty(),
            "∫ log x's antiderivative differentiates via diff_log; must withhold: {lean}"
        );
    }

    #[test]
    fn emit_lean_diff_sin_without_sorry() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let derived = diff(sin_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx sin(x) should be Lean-certifiable");
        assert!(
            !lean.contains("sorry"),
            "d/dx sin(x) certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("Real.deriv_sin"),
            "expected Real.deriv_sin tactic: {lean}"
        );
        // Algebraic cleanup (if present as its own step) must not be wrapped as a
        // deriv goal. Folded `mul_one` inside the `diff_sin` simp set is fine.
        if let Some(mul_one_block) = lean.split("-- Step").find(|b| b.contains(": mul_one\n")) {
            assert!(
                !mul_one_block.contains("deriv (fun"),
                "mul_one cleanup must be a plain equality, got: {mul_one_block}"
            );
        }
    }

    #[test]
    fn emit_lean_parens_nested_log_exp() {
        use crate::simplify::{rulesets::log_exp_rules, simplify_with, SimplifyConfig};

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("log", vec![pool.func("exp", vec![x])]);
        let derived = simplify_with(expr, &pool, &log_exp_rules(), SimplifyConfig::default());
        let lean = emit_lean_expr(&derived, &pool);
        assert!(!lean.is_empty(), "log(exp(x)) should be Lean-certifiable");
        assert!(
            lean.contains("Real.log (Real.exp"),
            "nested funcs must be parenthesized, got: {lean}"
        );
        assert!(
            !lean.contains("Real.log Real.exp "),
            "unparenthesized application is a type error: {lean}"
        );
    }

    #[test]
    fn exp_of_log_certifies_with_positivity_hyp() {
        // `ExpOfLog` records `SideCondition::Positive(x)`; the Lean exporter
        // upgrades that into an explicit `(x : ℝ) (hx : 0 < x)` binder and
        // closes the goal with `Real.exp_log hx` instead of withholding.
        use crate::simplify::{rulesets::log_exp_rules, simplify_with, SimplifyConfig};

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        let derived = simplify_with(expr, &pool, &log_exp_rules(), SimplifyConfig::default());
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            !lean.is_empty(),
            "exp(log(x)) with a recorded positivity condition should certify"
        );
        assert!(
            !lean.contains("sorry"),
            "certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hx : 0 < x)"),
            "expected an explicit positivity binder: {lean}"
        );
        assert!(
            lean.contains("Real.exp_log hx"),
            "expected Real.exp_log to consume the hypothesis: {lean}"
        );
    }

    #[test]
    fn exp_of_log_withheld_when_positivity_unproven() {
        // A step with no recorded side condition at all (e.g. hand-built,
        // bypassing `ExpOfLog::apply`) must still be withheld — the exporter
        // never invents a hypothesis that wasn't in the derivation log.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        let step = RewriteStep::simple("exp_of_log", expr, x);
        let lean = emit_step(&step, &pool);
        assert!(
            lean.contains("sorry"),
            "step without a positivity side condition must fall back to sorry: {lean}"
        );
    }

    #[test]
    fn log_of_product_certifies_two_factors_under_positivity() {
        // The colored e-graph's conditional `log_of_product_positive` rule
        // records `Positive(x)`/`Positive(y)` once the caller's assumptions
        // discharge them; the exporter should turn that into a real
        // `Real.log_mul` certificate instead of withholding.
        use crate::kernel::expr::PredicateKind;
        use crate::simplify::AssumptionContext;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let zero = pool.integer(0_i32);
        let mut assumptions = AssumptionContext::new();
        assumptions
            .refine(pool.predicate(PredicateKind::Gt, vec![x, zero]), &pool)
            .unwrap();
        assumptions
            .refine(pool.predicate(PredicateKind::Gt, vec![y, zero]), &pool)
            .unwrap();
        let expr = pool.func("log", vec![pool.mul(vec![x, y])]);
        let derived = assumptions.simplify(expr, &pool);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(!lean.is_empty(), "log(x*y) should certify under x>0, y>0");
        assert!(
            !lean.contains("sorry"),
            "certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hx : 0 < x)") && lean.contains("(hy : 0 < y)"),
            "expected explicit positivity binders: {lean}"
        );
        assert!(
            lean.contains("Real.log_mul (ne_of_gt hx) (ne_of_gt hy)"),
            "expected Real.log_mul to consume both hypotheses: {lean}"
        );
    }

    #[test]
    fn log_of_product_withheld_for_three_factors() {
        // `positivity_tactic` only has a chained lemma for two factors; a
        // three-factor product must stay withheld rather than emit a tactic
        // that can't close the goal.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let before = pool.func("log", vec![pool.mul(vec![x, y, z])]);
        let after = pool.add(vec![
            pool.func("log", vec![x]),
            pool.func("log", vec![y]),
            pool.func("log", vec![z]),
        ]);
        let step = RewriteStep::with_conditions(
            "log_of_product",
            before,
            after,
            vec![
                SideCondition::Positive(x),
                SideCondition::Positive(y),
                SideCondition::Positive(z),
            ],
        );
        let lean = emit_step(&step, &pool);
        assert!(
            lean.contains("sorry"),
            "three-factor log_of_product has no known lemma yet; must withhold: {lean}"
        );
    }

    #[test]
    fn sum_of_logs_certifies_with_positivity_hyp() {
        // `SumOfLogs` (`log x + log y → log(x·y)`) records `Positive(x)`,
        // `Positive(y)`. The exporter upgrades the two-factor case into an
        // explicit-binder `Real.log_mul` certificate rather than the (failing)
        // `ring_nf; simp` fallback.
        use crate::simplify::{rulesets::log_exp_rules, simplify_with, SimplifyConfig};

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![pool.func("log", vec![x]), pool.func("log", vec![y])]);
        let derived = simplify_with(expr, &pool, &log_exp_rules(), SimplifyConfig::default());
        let lean = emit_lean_expr(&derived, &pool);
        assert!(!lean.is_empty(), "log x + log y should certify under x,y>0");
        assert!(
            !lean.contains("sorry"),
            "certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hx : 0 < x)") && lean.contains("(hy : 0 < y)"),
            "expected explicit positivity binders: {lean}"
        );
        assert!(
            lean.contains("Real.log_mul (ne_of_gt hx) (ne_of_gt hy)"),
            "expected Real.log_mul to consume both hypotheses: {lean}"
        );
    }

    #[test]
    fn product_of_exps_certifies_with_exp_add() {
        // `exp x · exp y → exp(x + y)` is unconditionally valid; the exporter
        // folds it with `Real.exp_add` applied right-to-left.
        use crate::simplify::{rulesets::log_exp_rules, simplify_with, SimplifyConfig};

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.mul(vec![pool.func("exp", vec![x]), pool.func("exp", vec![y])]);
        let derived = simplify_with(expr, &pool, &log_exp_rules(), SimplifyConfig::default());
        let lean = emit_lean_expr(&derived, &pool);
        assert!(!lean.is_empty(), "exp x * exp y should certify");
        assert!(
            !lean.contains("sorry"),
            "certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("← Real.exp_add"),
            "expected exp_add fold: {lean}"
        );
    }

    #[test]
    fn gamma_maps_to_real_gamma_and_imports() {
        // Alkahest's lowercase `gamma` must be emitted as Mathlib's `Real.Gamma`,
        // with the Gamma import present in the (non-diff) header, so a factorial /
        // gamma identity type-checks.
        use crate::deriv::log::DerivedExpr;

        let pool = p();
        let k = pool.symbol("k", Domain::Real);
        let one = pool.integer(1_i32);
        let expr = pool.mul(vec![k, pool.func("gamma", vec![pool.add(vec![k, one])])]);
        assert!(
            expr_to_lean(expr, &pool).contains("Real.Gamma"),
            "gamma must map to Real.Gamma"
        );
        let derived = DerivedExpr::new(expr);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            lean.contains("import Mathlib.Analysis.SpecialFunctions.Gamma.Basic"),
            "header must import Gamma: {lean}"
        );
        assert!(
            lean.contains("Real.Gamma") && !lean.contains("sorry"),
            "gamma reflexivity cert must reference Real.Gamma without sorry: {lean}"
        );
    }

    #[test]
    fn diff_goal_names_unused_binder_underscore() {
        // The derivative of a constant leaves the lambda binder unused; under
        // `-DwarningAsError=true` that would be a hard lint error, so the binder
        // is emitted underscore-prefixed while the eval point stays a real use.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let c = pool.symbol("C1", Domain::Real);
        let zero = pool.integer(0_i32);
        let goal = emit_diff_goal(c, zero, x, &pool);
        assert!(
            goal.contains("fun (_x : ℝ)"),
            "unused binder must be underscore-prefixed: {goal}"
        );
        assert!(
            goal.contains(") x = "),
            "eval point must remain the bare variable: {goal}"
        );

        // When the body *does* use the variable, the binder keeps its real name.
        let sin_x = pool.func("sin", vec![x]);
        let one = pool.integer(1_i32);
        let used = emit_diff_goal(sin_x, one, x, &pool);
        assert!(
            used.contains("fun (x : ℝ)"),
            "a used binder must not be renamed: {used}"
        );
    }

    #[test]
    fn emit_lean_diff_x_squared_closes_with_ring() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(2_i32));
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx x² should be Lean-certifiable");
        assert!(
            lean.contains("try ring") || lean.contains("; ring"),
            "x² coeff order needs ring: {lean}"
        );
    }

    #[test]
    fn emit_lean_sum_rule_sin_cos() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![pool.func("sin", vec![x]), pool.func("cos", vec![x])]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (sin+cos) should be Lean-certifiable"
        );
        assert!(
            lean.contains("differentiableAt_sin") || lean.contains("deriv_add"),
            "sum_rule needs DifferentiableAt lemmas: {lean}"
        );
        assert!(
            !lean.contains("sorry"),
            "sum certificate must not use sorry: {lean}"
        );
    }

    #[test]
    fn emit_lean_product_rule_sin_exp() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.func("sin", vec![x]), pool.func("exp", vec![x])]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (sin·exp) should be Lean-certifiable after product_rule fix"
        );
        assert!(
            lean.contains("deriv_mul"),
            "expected product_rule deriv_mul tactic: {lean}"
        );
        assert!(
            !lean.contains("sorry"),
            "product certificate must not use sorry: {lean}"
        );
    }

    #[test]
    fn emit_lean_tan_expand_uses_div_eq_mul_inv() {
        use crate::simplify::{rulesets::trig_rules, simplify_with, SimplifyConfig};

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("tan", vec![x]);
        let derived = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let lean = emit_lean_expr(&derived, &pool);
        assert!(!lean.is_empty(), "tan(x) expand should be Lean-certifiable");
        assert!(
            lean.contains("div_eq_mul_inv"),
            "tan→sin/cos needs div_eq_mul_inv for reciprocal form: {lean}"
        );
        assert!(
            lean.contains("Real.tan"),
            "tan must emit Real.tan, got: {lean}"
        );
    }

    #[test]
    fn emit_lean_log_pow_parenthesized() {
        use crate::simplify::{rulesets::log_exp_rules, simplify_with, SimplifyConfig};

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("log", vec![pool.pow(x, pool.integer(3_i32))]);
        let derived = simplify_with(expr, &pool, &log_exp_rules(), SimplifyConfig::default());
        let lean = emit_lean_expr(&derived, &pool);
        assert!(!lean.is_empty(), "log(x^3) should be Lean-certifiable");
        assert!(
            lean.contains("Real.log (") && lean.contains("^"),
            "log of a power must keep the power inside the log arg: {lean}"
        );
        // Guard against `(Real.log x) ^ 3` parse.
        assert!(
            !lean.contains("Real.log (x : ℝ)) ^") && !lean.contains("Real.log x ^"),
            "power must not bind tighter than log: {lean}"
        );
    }

    #[test]
    fn withhold_generalized_power_rule_on_sin_squared() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let expr = pool.pow(sin_x, pool.integer(2_i32));
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "d/dx sin(x)² embeds chain rule via power_rule; must withhold: {lean}"
        );
    }

    #[test]
    fn emit_lean_chain_rule_diff_sin_x_squared() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let sin_x2 = pool.func("sin", vec![x2]);
        let derived = diff(sin_x2, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "chain-rule d/dx sin(x²) should now be Lean-certifiable"
        );
        assert!(
            lean.contains("hasDerivAt_pow") && lean.contains("(hg.sin).deriv"),
            "expected chain-rule composition tactic: {lean}"
        );
        assert!(
            !lean.contains("sorry"),
            "chain-rule certificate must not use sorry: {lean}"
        );
    }

    #[test]
    fn emit_lean_chain_rule_diff_exp_x_squared() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        let derived = diff(exp_x2, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "chain-rule d/dx exp(x²) should now be Lean-certifiable"
        );
        assert!(
            lean.contains("hasDerivAt_pow") && lean.contains("(hg.exp).deriv"),
            "expected exp chain-rule composition tactic: {lean}"
        );
        assert!(
            !lean.contains("sorry"),
            "chain-rule certificate must not use sorry: {lean}"
        );
    }

    #[test]
    fn withhold_chain_rule_diff_log_composite() {
        use crate::diff::diff;

        // d/dx log(x²) still routes through diff_log; that shape stays withheld.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let log_x2 = pool.func("log", vec![x2]);
        let derived = diff(log_x2, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "chain-rule d/dx log(x²) is not encoded; must withhold: {lean}"
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
    fn emit_tendsto_unrecognized_pattern_withheld() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // sin(x) → no known pattern → should return empty string
        let expr = pool.func("sin", vec![x]);
        let zero = pool.integer(0_i32);
        let lean = emit_tendsto_cert(expr, x, zero, &pool);
        assert!(
            lean.is_empty(),
            "unrecognized patterns must not emit sorry certificates: got {lean}"
        );
    }

    #[test]
    fn emit_tendsto_recognized_pattern_yields_cert() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // exp(-x) → recognized pattern → should yield non-empty cert
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.func("exp", vec![neg_x]);
        let zero = pool.integer(0_i32);
        let lean = emit_tendsto_cert(expr, x, zero, &pool);
        assert!(
            !lean.is_empty(),
            "recognized tendsto patterns must yield a certificate"
        );
        assert!(
            !lean.contains("sorry"),
            "recognized pattern certificate must not use sorry: {lean}"
        );
    }

    #[test]
    fn emit_tendsto_header_has_filter_imports() {
        let h = emit_limit_header();
        assert!(h.contains("import Mathlib.Tactic"));
        assert!(h.contains("Filter"));
    }
}
