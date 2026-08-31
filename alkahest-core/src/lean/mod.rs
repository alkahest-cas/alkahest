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
        // `abs_of_positive` (`abs(x) = x` under `x > 0`) is upgraded to a
        // `abs_of_pos` certificate by `positivity_certificate` for the bare-
        // symbol case; anything else must be withheld, so the table default
        // is a withheld `sorry` rather than the (non-compiling) generic
        // fallback.
        "abs_of_positive" => "by sorry",
        "log_of_product" | "log_of_product_positive" => "by sorry",
        // `sum_of_logs` (`log a + log b + … = log(a·b·…)`) is only sound with a
        // positivity hypothesis on every argument. [`emit_step_wrt`] upgrades the
        // two-factor case to an explicit-binder `Real.log_mul` certificate via
        // [`positivity_certificate`]; anything it can't upgrade (compound
        // arguments, or three-plus factors [`positivity_tactic`] has no chained
        // lemma for) must be withheld, so the table default is a withheld `sorry`.
        "sum_of_logs" => "by sorry",
        // `log(x·y⁻¹) = log x − log y` is only sound with `x > 0`, `y > 0`.
        // [`emit_step_wrt`] upgrades the two-symbol case to an explicit-binder
        // certificate via [`positivity_certificate`]/[`positivity_tactic`];
        // anything it can't upgrade must be withheld, so the table default is a
        // withheld `sorry` rather than the (non-compiling) `ring_nf; simp`.
        "log_of_quotient" => "by sorry",
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
                // `∫_{-∞}^{∞} P/Q = 2πi·Σ Res` — a value, not a rewrite.
                | "residue_theorem"
                | "log_rule"
                | "gosper_indefinite"
                | "gosper_definite_telescope"
                // Algorithmic Basel/ζ(2m) closed form — no Mathlib step proof yet.
                | "basel_zeta_even"
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

/// The printed name of a differentiation variable, falling back to `"x"` for
/// anything that isn't a bare [`ExprData::Symbol`] (shouldn't happen in
/// practice — `wrt` is always a symbol — but keeps this total).
fn wrt_name(wrt: ExprId, pool: &ExprPool) -> String {
    pool.with(wrt, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    })
}

/// The Mathlib pointwise `Real.hasDerivAt_<f>` fact for a primitive with a
/// known *unconditional* derivative at every real point — distinct from
/// [`chain_outer_lemma`], which composes `f` with an inner function rather
/// than supplying `f`'s own pointwise derivative.
fn pointwise_hasderivat_lemma(name: &str) -> Option<&'static str> {
    match name {
        "sin" => Some("Real.hasDerivAt_sin"),
        "cos" => Some("Real.hasDerivAt_cos"),
        "exp" => Some("Real.hasDerivAt_exp"),
        _ => None,
    }
}

/// `expr` is `name(wrt)` for some single-argument named function — i.e. a
/// unary primitive applied directly to the differentiation variable (not a
/// composite argument). Returns the function name.
fn unary_func_name(expr: ExprId, wrt: ExprId, pool: &ExprPool) -> Option<String> {
    pool.with(expr, |d| match d {
        ExprData::Func { name, args } if args.len() == 1 && args[0] == wrt => Some(name.clone()),
        _ => None,
    })
}

/// If `before` is `f(wrt) ^ n` for `f` a primitive with a known pointwise
/// `HasDerivAt` fact ([`pointwise_hasderivat_lemma`]) and integer exponent
/// `n`, return `(explicit_binders, tactic)` for the shapes this emitter knows
/// how to close:
///   - `n ≥ 2`: `HasDerivAt.pow`, unconditional (e.g. `d/dx sin(x)² = 2 sin x cos x`).
///   - `n == -1`: `HasDerivAt.inv`, needs `f(x) ≠ 0` (e.g. `d/dx (1 / sin x)`).
///
/// Other exponents (`0`, `1`, `≤ -2`) aren't encoded and return `None` — the
/// caller withholds. This is the "outer power, inner primitive" mirror of
/// [`composite_pow_inner_exp`]/[`chain_diff_tactic`], which instead handles
/// `f(x^n)` (power *inside* the primitive).
fn power_chain_certificate(
    before: ExprId,
    wrt: ExprId,
    pool: &ExprPool,
) -> Option<(Option<String>, String)> {
    let (base, exp_n) = pool.with(before, |d| match d {
        ExprData::Pow { base, exp } => pool.with(*exp, |e| match e {
            ExprData::Integer(n) => n.0.to_i64().map(|k| (*base, k)),
            _ => None,
        }),
        _ => None,
    })?;
    let name = unary_func_name(base, wrt, pool)?;
    let lemma = pointwise_hasderivat_lemma(&name)?;
    let var = wrt_name(wrt, pool);
    if exp_n >= 2 {
        let tactic = format!(
            "by\n    \
             have hf := {lemma} {var}\n    \
             rw [(hf.pow {exp_n}).deriv]\n    \
             push_cast\n    \
             ring"
        );
        return Some((None, tactic));
    }
    if exp_n == -1 {
        let binder = format!("({var} : ℝ) (hne : Real.{name} {var} ≠ 0)");
        let tactic = format!(
            "by\n    \
             have hf := {lemma} {var}\n    \
             rw [(hf.inv hne).deriv]\n    \
             ring"
        );
        return Some((Some(binder), tactic));
    }
    None
}

/// If `before` is `Mul([f(wrt), g(wrt)⁻¹])` (in either factor order) for
/// `f`, `g` primitives with known pointwise `HasDerivAt` facts, build a
/// certificate for the quotient's `product_rule` step directly via
/// `HasDerivAt.mul` + `HasDerivAt.inv`, given `g(x) ≠ 0`.
///
/// Alkahest has no explicit division node — `f(x)/g(x)` is represented as
/// `f(x) * g(x)⁻¹` — so this targets that shape rather than Mathlib's
/// `HasDerivAt.div` (whose LHS pattern is literally `c y / d y`, which would
/// not syntactically match our `rw` target without first rewriting `/` to
/// `* ⁻¹`). Note: `field_simp` (not `ring`) is required to close the
/// resulting goal — see [`inv_cancel_certificate`] for why bare `ring` can't
/// discharge the `g x * (g x)⁻¹` cancellation buried inside it.
fn quotient_chain_certificate(
    before: ExprId,
    wrt: ExprId,
    pool: &ExprPool,
) -> Option<(Option<String>, String)> {
    let factor_inv_base = |id: ExprId| -> Option<ExprId> {
        pool.with(id, |d| match d {
            ExprData::Pow { base, exp } => pool
                .with(*exp, |e| matches!(e, ExprData::Integer(n) if n.0 == -1))
                .then_some(*base),
            _ => None,
        })
    };
    let (num, den_base) = pool.with(before, |d| match d {
        ExprData::Mul(xs) if xs.len() == 2 => {
            if let Some(b) = factor_inv_base(xs[1]) {
                Some((xs[0], b))
            } else {
                factor_inv_base(xs[0]).map(|b| (xs[1], b))
            }
        }
        _ => None,
    })?;
    let fname = unary_func_name(num, wrt, pool)?;
    let gname = unary_func_name(den_base, wrt, pool)?;
    let flemma = pointwise_hasderivat_lemma(&fname)?;
    let glemma = pointwise_hasderivat_lemma(&gname)?;
    let var = wrt_name(wrt, pool);
    let binder = format!("({var} : ℝ) (hne : Real.{gname} {var} ≠ 0)");
    let tactic = format!(
        "by\n    \
         have hf := {flemma} {var}\n    \
         have hg := ({glemma} {var}).inv hne\n    \
         rw [(hf.mul hg).deriv]\n    \
         field_simp [hne]"
    );
    Some((Some(binder), tactic))
}

/// `deriv (fun x => Real.sqrt x) x = 1 / (2 * sqrt x)` needs `x ≠ 0`
/// (`Real.hasDerivAt_sqrt`), unlike `Real.deriv_log`'s unconditional identity.
/// Always succeeds once called — callers gate on
/// [`is_unary_of_var`]`(before, wrt, ..)` first.
fn diff_sqrt_certificate(wrt: ExprId, pool: &ExprPool) -> (Option<String>, String) {
    let var = wrt_name(wrt, pool);
    let binder = format!("({var} : ℝ) (hx : 0 < {var})");
    let tactic = "by\n    \
         have h := (Real.hasDerivAt_sqrt hx.ne').deriv\n    \
         rw [h]\n    \
         ring"
        .to_string();
    (Some(binder), tactic)
}

/// Pointwise `d/dx (wrt ^ n)` for a **negative** integer `n`, with an explicit
/// `x ≠ 0` binder. Pretty-print spells `x⁻¹` as `(x)⁻¹` and `x⁻ᵏ` (`k ≥ 2`) as
/// `(x)⁻¹ ^ (k : ℕ)` — never `x ^ (-k : ℤ)` — so the tactic must close that
/// spelling: `hasDerivAt_inv` for the inverse, then `HasDerivAt.pow` for a
/// further natural power of it.
///
/// Must not go through [`diff_body_unconditional`]/`deriv_pow`: that simp set
/// has no `x ≠ 0` and only discharges non-negative integer powers. Inverse of
/// a primitive (`(sin x)⁻¹`) is [`power_chain_certificate`]; this is only
/// `wrt ^ (-k)`.
fn neg_pow_of_var_certificate(
    before: ExprId,
    wrt: ExprId,
    pool: &ExprPool,
) -> Option<(Option<String>, String)> {
    let exp_n = pool.with(before, |d| match d {
        ExprData::Pow { base, exp } if *base == wrt => pool.with(*exp, |e| match e {
            ExprData::Integer(n) => n.0.to_i64().filter(|&k| k < 0),
            _ => None,
        }),
        _ => None,
    })?;
    let var = wrt_name(wrt, pool);
    let binder = format!("({var} : ℝ) (hx : {var} ≠ 0)");
    let k = exp_n.unsigned_abs();
    let tactic = if k == 1 {
        "by\n    \
         have h := (hasDerivAt_inv hx).deriv\n    \
         rw [h]\n    \
         field_simp [hx]\n    \
         try ring"
            .to_string()
    } else {
        format!(
            "by\n    \
             have hinv := hasDerivAt_inv hx\n    \
             rw [(hinv.pow {k}).deriv]\n    \
             field_simp [hx]\n    \
             try ring"
        )
    };
    Some((Some(binder), tactic))
}

/// Certificates for `diff_primitive_registry` steps: the generic rule name
/// `lean::diff_rule_to_tactic` never maps, so dispatch on the actual
/// primitive by re-inspecting `before` (mirroring how
/// [`PrimitiveRegistry::diff_forward`](crate::primitive::PrimitiveRegistry::diff_forward)
/// dispatched when building the derivative in the first place).
///
/// Encoded primitives:
/// * `tan` — Alkahest records `d/dx tan(x) = (1 + tan(x)²) · 1` (the `1 + tan²`
///   identity, not `1/cos²`), so closing the goal needs both
///   `Real.hasDerivAt_tan` (needs `cos x ≠ 0`) *and* `Real.inv_one_add_tan_sq`
///   to reconcile the two forms — a bare `rw [Real.deriv_tan]; ring` is not
///   enough since that equivalence itself depends on `cos x ≠ 0`.
/// * `sinh` / `cosh` — unconditional on `ℝ` (`Real.deriv_sinh` /
///   `Real.deriv_cosh`). Sums and products of these also certify via the
///   everywhere-differentiable fragment ([`diff_body_unconditional`]).
/// * `atan` — unconditional on `ℝ`. Alkahest records `(1+x²)⁻¹`; Mathlib's
///   `Real.hasDerivAt_arctan'` is already in that form (`hasDerivAt_arctan`
///   is the `1/(1+x²)` spelling).
/// * `asin` — needs `|x| < 1`. Mathlib's `Real.hasDerivAt_arcsin` asks for
///   `x ≠ -1` and `x ≠ 1`; the stricter open-interval binder implies those
///   and matches the domain where `1/√(1-x²)` is the genuine (non-junk)
///   derivative.
///
/// `tanh` is withheld: Mathlib v4.9.0 has no `hasDerivAt_tanh` and no
/// `1 - tanh² = 1/cosh²` identity analogous to `Real.inv_one_add_tan_sq`.
/// Do not sorry.
fn registry_diff_certificate(
    before: ExprId,
    wrt: ExprId,
    pool: &ExprPool,
) -> Option<(Option<String>, String)> {
    let name = unary_func_name(before, wrt, pool)?;
    match name.as_str() {
        "tan" => {
            let var = wrt_name(wrt, pool);
            let binder = format!("({var} : ℝ) (hne : Real.cos {var} ≠ 0)");
            let tactic = format!(
                "by\n    \
                 have hderiv := (Real.hasDerivAt_tan hne).deriv\n    \
                 have hinv : (1 + Real.tan {var} ^ 2)⁻¹ = Real.cos {var} ^ 2 := \
                 Real.inv_one_add_tan_sq hne\n    \
                 have hsq : 1 + Real.tan {var} ^ 2 = (Real.cos {var} ^ 2)⁻¹ := by \
                 rw [← hinv, inv_inv]\n    \
                 rw [hderiv, one_div, hsq]\n    \
                 ring"
            );
            Some((Some(binder), tactic))
        }
        "sinh" => Some((
            None,
            "by simp [Real.deriv_sinh, one_mul, mul_one]".to_string(),
        )),
        "cosh" => Some((
            None,
            "by simp [Real.deriv_cosh, one_mul, mul_one]".to_string(),
        )),
        "atan" => {
            let var = wrt_name(wrt, pool);
            let tactic = format!(
                "by\n    \
                 rw [(Real.hasDerivAt_arctan' {var}).deriv]\n    \
                 ring"
            );
            Some((None, tactic))
        }
        "asin" => {
            let var = wrt_name(wrt, pool);
            let binder = format!("({var} : ℝ) (hx : -1 < {var} ∧ {var} < 1)");
            let tactic = "by\n    \
                 have hderiv := (Real.hasDerivAt_arcsin hx.1.ne' hx.2.ne).deriv\n    \
                 rw [hderiv, one_div]\n    \
                 ring"
                .to_string();
            Some((Some(binder), tactic))
        }
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
    let var_name = wrt_name(wrt, pool);
    Some(format!(
        "by\n    \
         have hg := hasDerivAt_pow {n} {var_name}\n    \
         rw [(hg.{suffix}).deriv]\n    \
         push_cast\n    \
         ring"
    ))
}

/// The tactic that closes every "combine" differentiation step —
/// `diff_univariate_poly`, `sum_rule`, `product_rule` — over the
/// *everywhere-differentiable* fragment ([`diff_body_unconditional`]).
/// Negative powers of the variable go through [`neg_pow_combine_certificate`]
/// instead; they must not be added here.
///
/// `simp only` rewrites strictly at `deriv`/`DifferentiableAt` positions (no
/// algebraic normalization of the lambda body, which would desync the
/// structural `deriv_add`/`deriv_mul` match), and the raised
/// `maxDischargeDepth` lets simp's discharger recurse through
/// `DifferentiableAt.add`/`.mul`/`.pow` for deeply nested products and sums
/// (the default depth of 2 silently leaves the `deriv` unreduced — the exact
/// "green by luck" failure this gate hardening fixes). `try ring` then
/// reconciles Alkahest's coefficient ordering / `1 *` decorations with the
/// Mathlib derivative; it is a no-op (not a linter error) on the rare step
/// simp closes outright.
const UNCONDITIONAL_DIFF_TACTIC: &str = "by\n    \
     simp (config := { maxDischargeDepth := 8 }) only [deriv_add, deriv_mul, deriv_pow, \
     deriv_const, deriv_id'', Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, \
     Real.deriv_sinh, Real.deriv_cosh, \
     differentiableAt_pow, differentiableAt_id', differentiableAt_const, \
     Real.differentiableAt_sin, Real.differentiableAt_cos, Real.differentiableAt_exp, \
     Real.differentiableAt_sinh, Real.differentiableAt_cosh, \
     DifferentiableAt.add, DifferentiableAt.mul, DifferentiableAt.pow]\n    \
     try ring";

/// Combine tactic for bodies in the [`diff_body_unconditional`] fragment plus
/// pointwise `log(wrt)` / `sqrt(wrt)`. Unlike [`UNCONDITIONAL_DIFF_TACTIC`],
/// this consumes an explicit `(hx : 0 < x)` binder: `DifferentiableAt log`
/// needs `x ≠ 0` (`Real.differentiableAt_log` / `Real.hasDerivAt_log`) and
/// `sqrt` needs the same (`Real.hasDerivAt_sqrt`). `0 < x` implies both.
///
/// `Real.deriv_log` is deliberately *not* in [`UNCONDITIONAL_DIFF_TACTIC`]:
/// dumping it there would still leave the `DifferentiableAt log` side goal
/// of `deriv_mul`/`deriv_add` open. Here the hyp-gated `hasDerivAt` facts
/// discharge those side goals. `try field_simp` reconciles `x * x⁻¹` /
/// `1 / (2 * sqrt x)` against Alkahest's reciprocal form; `try ring` then
/// closes coefficient order the way the unconditional tactic does.
const LOG_SQRT_DIFF_TACTIC: &str = "by\n    \
     simp (config := { maxDischargeDepth := 8 }) only [deriv_add, deriv_mul, deriv_pow, \
     deriv_const, deriv_id'', Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, \
     Real.deriv_sinh, Real.deriv_cosh, \
     (Real.hasDerivAt_log hx.ne').deriv, (Real.hasDerivAt_sqrt hx.ne').deriv, \
     differentiableAt_pow, differentiableAt_id', differentiableAt_const, \
     Real.differentiableAt_sin, Real.differentiableAt_cos, Real.differentiableAt_exp, \
     Real.differentiableAt_sinh, Real.differentiableAt_cosh, \
     Real.differentiableAt_log hx.ne', (Real.hasDerivAt_sqrt hx.ne').differentiableAt, \
     DifferentiableAt.add, DifferentiableAt.mul, DifferentiableAt.pow]\n    \
     try field_simp [hx.ne']\n    \
     try ring";

/// Walk a differentiated body, accepting the unconditional fragment and
/// (when `allow_log_sqrt`) pointwise `log`/`sqrt` of exactly the variable.
fn diff_body_combine(before: ExprId, wrt: ExprId, pool: &ExprPool, allow_log_sqrt: bool) -> bool {
    fn walk(f: ExprId, wrt: ExprId, pool: &ExprPool, allow_log_sqrt: bool) -> bool {
        pool.with(f, |d| match d {
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
            // A bare symbol is either the differentiation variable or a free
            // constant (`fun x => C1` is constant in `x`); both are handled by
            // `differentiableAt_id'` / `differentiableAt_const`.
            ExprData::Symbol { .. } => true,
            ExprData::Pow { base, exp } => {
                *base == wrt
                    && pool.with(*exp, |e| match e {
                        ExprData::Integer(n) => n.0.to_i64().is_some_and(|k| k >= 0),
                        _ => false,
                    })
            }
            ExprData::Func { name, args } => {
                let ok = matches!(name.as_str(), "sin" | "cos" | "exp" | "sinh" | "cosh")
                    || (allow_log_sqrt && matches!(name.as_str(), "log" | "sqrt"));
                ok && args.len() == 1 && args[0] == wrt
            }
            ExprData::Add(xs) => xs.iter().all(|&c| walk(c, wrt, pool, allow_log_sqrt)),
            ExprData::Mul(xs) => xs.iter().all(|&c| walk(c, wrt, pool, allow_log_sqrt)),
            _ => false,
        })
    }
    walk(before, wrt, pool, allow_log_sqrt)
}

/// True when the differentiated body `before` is built only from atoms whose
/// derivative [`UNCONDITIONAL_DIFF_TACTIC`]'s simp set computes *without any
/// side condition* — i.e. the function is differentiable at every real point:
/// the differentiation variable, constant symbols (`C1`, `C2`, …) and numeric
/// literals, sums and products of those, non-negative integer powers of the
/// variable, and the pointwise primitives `sin`/`cos`/`exp`/`sinh`/`cosh`
/// applied to exactly the variable.
///
/// Everything else must be withheld by the caller of this predicate:
/// `log`/`sqrt` go through [`diff_body_log_sqrt`] (with a positivity binder),
/// `tan`/`asin` and any inverse or negative power need a side condition this
/// simp set cannot discharge, and a chain composite `f(g x)` with `g ≠ x`
/// lacks the composite's `DifferentiableAt` lemma (those go through
/// [`chain_diff_tactic`] instead, never this fragment). `atan` is everywhere
/// differentiable but its derivative is `(1+x²)⁻¹`, which this simp set does
/// not compute — it certifies pointwise via [`registry_diff_certificate`].
fn diff_body_unconditional(before: ExprId, wrt: ExprId, pool: &ExprPool) -> bool {
    diff_body_combine(before, wrt, pool, false)
}

/// The [`diff_body_unconditional`] fragment plus pointwise `log(wrt)` and/or
/// `sqrt(wrt)` (argument exactly the variable, not a composite). Callers emit
/// [`LOG_SQRT_DIFF_TACTIC`] with `(hx : 0 < x)` rather than the unconditional
/// tactic. Composites `log(g x)` / `sqrt(g x)` with `g ≠ x` stay withheld.
fn diff_body_log_sqrt(before: ExprId, wrt: ExprId, pool: &ExprPool) -> bool {
    diff_body_combine(before, wrt, pool, true)
}

/// Bind `(x : ℝ) (hx : 0 < x)` and close a combine step via
/// [`LOG_SQRT_DIFF_TACTIC`]. `0 < x` covers both `log` (`x ≠ 0`) and `sqrt`.
fn log_sqrt_diff_certificate(wrt: ExprId, pool: &ExprPool) -> (Option<String>, String) {
    let var = wrt_name(wrt, pool);
    let binder = format!("({var} : ℝ) (hx : 0 < {var})");
    (Some(binder), LOG_SQRT_DIFF_TACTIC.to_string())
}

/// True when `before` is built from `{wrt, constants, wrt⁻ⁿ}` under `Add`/`Mul`
/// and contains at least one negative integer power of the variable.
///
/// This is the second combine-step fragment: the derivative needs `x ≠ 0` to
/// discharge `DifferentiableAt` of the inverse, so it is *not* in
/// [`diff_body_unconditional`] and must not be dumped into that simp set.
/// Pointwise `wrt⁻ⁿ` itself is [`neg_pow_of_var_certificate`]; this gate is
/// for the `product_rule` / `sum_rule` bodies those atoms appear in
/// (e.g. `-x⁻¹`, `x + x⁻¹`). `log`/`sin`/`cos`/`exp` and non-negative powers
/// of `wrt` other than the bare variable stay out — those have their own
/// fragments.
fn diff_body_neg_pow_combine(before: ExprId, wrt: ExprId, pool: &ExprPool) -> bool {
    fn walk(f: ExprId, wrt: ExprId, pool: &ExprPool, saw_neg: &mut bool) -> bool {
        pool.with(f, |d| match d {
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
            ExprData::Symbol { .. } => true,
            ExprData::Pow { base, exp } if *base == wrt => pool.with(*exp, |e| match e {
                ExprData::Integer(n) => n.0.to_i64().is_some_and(|k| {
                    if k < 0 {
                        *saw_neg = true;
                    }
                    k < 0
                }),
                _ => false,
            }),
            ExprData::Add(xs) => xs.iter().all(|&c| walk(c, wrt, pool, saw_neg)),
            ExprData::Mul(xs) => xs.iter().all(|&c| walk(c, wrt, pool, saw_neg)),
            _ => false,
        })
    }
    let mut saw_neg = false;
    walk(before, wrt, pool, &mut saw_neg) && saw_neg
}

/// Combine-step certificate for [`diff_body_neg_pow_combine`]: bind
/// `(x : ℝ) (hx : x ≠ 0)` and close with the same inverse `HasDerivAt` facts
/// the pointwise `wrt⁻ⁿ` path already uses (`deriv_inv` / `differentiableAt_inv`),
/// plus `deriv_pow''` so a factor `(x)⁻¹ ^ (k : ℕ)` (pretty-printed `x⁻ᵏ`)
/// unfolds. `field_simp [hx]` reconciles `-(x^2)⁻¹` with Alkahest's
/// `(-1 * 1 * (x)⁻¹ ^ 2)` spelling. Must not be merged into
/// [`UNCONDITIONAL_DIFF_TACTIC`] — that simp set has no `x ≠ 0`.
fn neg_pow_combine_certificate(wrt: ExprId, pool: &ExprPool) -> (Option<String>, String) {
    let var = wrt_name(wrt, pool);
    let binder = format!("({var} : ℝ) (hx : {var} ≠ 0)");
    let tactic = "by\n    \
         simp (config := { maxDischargeDepth := 8 }) only [deriv_add, deriv_mul, deriv_pow, \
         deriv_pow'', deriv_const, deriv_id'', deriv_inv, \
         differentiableAt_pow, differentiableAt_id', differentiableAt_const, \
         DifferentiableAt.add, DifferentiableAt.mul, DifferentiableAt.pow, \
         differentiableAt_inv.mpr hx]\n    \
         field_simp [hx]\n    \
         try ring"
        .to_string();
    (Some(binder), tactic)
}

/// Unconditional combine first; then the negative-power fragment (`x ≠ 0`);
/// then the log/sqrt fragment (`0 < x`). Used by `sum_rule` /
/// `diff_univariate_poly` and (after the quotient-chain attempt) `product_rule`.
fn combine_step_certificate(
    before: ExprId,
    wrt: ExprId,
    pool: &ExprPool,
) -> Option<(Option<String>, String)> {
    if diff_body_unconditional(before, wrt, pool) {
        Some((None, UNCONDITIONAL_DIFF_TACTIC.to_string()))
    } else if diff_body_neg_pow_combine(before, wrt, pool) {
        Some(neg_pow_combine_certificate(wrt, pool))
    } else if diff_body_log_sqrt(before, wrt, pool) {
        Some(log_sqrt_diff_certificate(wrt, pool))
    } else {
        None
    }
}

fn diff_rule_to_tactic(rule_name: &str) -> Option<&'static str> {
    match rule_name {
        "diff_identity" => Some("by simp [deriv_id]"),
        "diff_const" => Some("by simp [deriv_const]"),
        // Combine steps over the everywhere-differentiable fragment. Callers in
        // [`diff_step_certificate`] go through [`combine_step_certificate`],
        // which uses this tactic only when [`diff_body_unconditional`] holds
        // and switches to a binder-carrying tactic for the neg-pow / log/sqrt
        // fragments.
        "diff_univariate_poly" => Some(UNCONDITIONAL_DIFF_TACTIC),
        "sum_rule" => Some(UNCONDITIONAL_DIFF_TACTIC),
        "product_rule" => Some(UNCONDITIONAL_DIFF_TACTIC),
        "power_rule" | "power_rule_n1" => Some("by simp [deriv_pow, deriv_mul]; try ring"),
        "power_rule_n0" => Some("by simp [deriv_const]"),
        // Pointwise Mathlib lemmas for `deriv (fun x => f x) x = …` when the
        // argument is exactly the free variable. Chain-rule cases are withheld.
        "diff_sin" => Some("by simp [Real.deriv_sin, one_mul, mul_one]"),
        "diff_cos" => Some("by simp [Real.deriv_cos, one_mul, mul_one]"),
        "diff_exp" => Some("by simp [Real.deriv_exp, one_mul, mul_one]"),
        // `Real.deriv_log (x : ℝ) : deriv log x = x⁻¹` holds unconditionally —
        // Mathlib extends `Real.log` to negatives via `log |x|` and to `0` via
        // the junk value `log 0 = 0`, and the derivative identity survives
        // both, so (unlike `diff_sqrt`) no positivity hypothesis is needed.
        "diff_log" => Some("by simp [Real.deriv_log, one_mul, mul_one]"),
        // `diff_sqrt` needs an explicit `x ≠ 0` side condition
        // (`Real.hasDerivAt_sqrt`) that this unconditional table can't
        // express; see [`diff_sqrt_certificate`].
        "diff_sqrt" => None,
        "diff_forward" | "diff_primitive_registry" | "diff_piecewise" | "diff_root_sum" => None,
        _ => None,
    }
}

/// Resolve the full certificate for a differentiation-rule step: an optional
/// explicit hypothesis-binder preamble plus the tactic that closes
/// `deriv (fun v => before) v = after`. Returns `None` when nothing this
/// emitter knows about applies — the caller must withhold.
///
/// Tries, in order: the `f(x^n)` chain rule ([`chain_diff_tactic`]), then a
/// per-rule dispatch that covers the pointwise cases (gated on
/// [`is_unary_of_var`]/[`is_pow_of_var`] so composites correctly fall
/// through to withholding), the `diff_sqrt` positivity certificate, the
/// `diff_primitive_registry` dispatch (`tan`/`sinh`/`cosh`/`atan`/`asin`),
/// negative integer powers of the variable ([`neg_pow_of_var_certificate`]),
/// the `f(x)^n` / quotient chain shapes ([`power_chain_certificate`],
/// [`quotient_chain_certificate`]), and the two combine-step fragments
/// ([`combine_step_certificate`]).
fn diff_step_certificate(
    step: &RewriteStep,
    wrt: ExprId,
    pool: &ExprPool,
) -> Option<(Option<String>, String)> {
    match step.rule_name {
        "diff_sin" | "diff_cos" | "diff_exp" => {
            if is_unary_of_var(step.before, wrt, pool) {
                return diff_rule_to_tactic(step.rule_name).map(|t| (None, t.to_string()));
            }
            chain_diff_tactic(step.rule_name, step.before, wrt, pool).map(|t| (None, t))
        }
        "diff_log" => {
            if is_unary_of_var(step.before, wrt, pool) {
                diff_rule_to_tactic("diff_log").map(|t| (None, t.to_string()))
            } else {
                None
            }
        }
        "diff_sqrt" => {
            if is_unary_of_var(step.before, wrt, pool) {
                Some(diff_sqrt_certificate(wrt, pool))
            } else {
                None
            }
        }
        "diff_primitive_registry" => registry_diff_certificate(step.before, wrt, pool),
        "power_rule" | "power_rule_n1" | "power_rule_n0" => {
            // `deriv (fun x => xⁿ)` closes via `deriv_pow` only for a
            // non-negative integer `n`. Negative powers of the variable
            // (`x^(-k)`) need `x ≠ 0` and go through
            // [`neg_pow_of_var_certificate`] rather than the unconditional
            // simp set. `is_pow_of_var` alone accepts `x^(-2)`, so the
            // non-negative path still requires the unconditional gate.
            if is_pow_of_var(step.before, wrt, pool)
                && diff_body_unconditional(step.before, wrt, pool)
            {
                diff_rule_to_tactic(step.rule_name).map(|t| (None, t.to_string()))
            } else if step.rule_name == "power_rule" {
                neg_pow_of_var_certificate(step.before, wrt, pool)
                    .or_else(|| power_chain_certificate(step.before, wrt, pool))
            } else {
                None
            }
        }
        "diff_univariate_poly" | "sum_rule" => combine_step_certificate(step.before, wrt, pool),
        // `product_rule` first tries the `f(x)/g(x)` quotient chain (which
        // carries its own `g x ≠ 0` binder), then the combine fragments
        // (unconditional, then `x ≠ 0` negative powers, then `0 < x` log/sqrt).
        "product_rule" => quotient_chain_certificate(step.before, wrt, pool)
            .or_else(|| combine_step_certificate(step.before, wrt, pool)),
        name => diff_rule_to_tactic(name).map(|t| (None, t.to_string())),
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
        ("abs_of_positive", [x]) => Some(format!("by rw [abs_of_pos h{x}]")),
        ("log_of_product" | "log_of_product_positive" | "sum_of_logs", [x, y]) => Some(format!(
            "by rw [Real.log_mul (ne_of_gt h{x}) (ne_of_gt h{y})]"
        )),
        // `log(x · y⁻¹) = log x + (-1)·log y`, sound under `x > 0`, `y > 0`.
        // `Real.log_inv` (the `log y⁻¹ = -log y` half) is unconditional; only
        // the `Real.log_mul` split needs the nonzero hypotheses.
        ("log_of_quotient", [x, y]) => Some(format!(
            "by rw [Real.log_mul (ne_of_gt h{x}) (inv_ne_zero (ne_of_gt h{y})), \
             Real.log_inv]; ring"
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

/// A bare factor `a` is `a^1`; `Pow{base, Integer(n)}` is `(base, n)`.
fn factor_base_exp(id: ExprId, pool: &ExprPool) -> (ExprId, i64) {
    pool.with(id, |d| match d {
        ExprData::Pow { base, exp } => pool
            .with(*exp, |e| match e {
                ExprData::Integer(n) => n.0.to_i64().map(|k| (*base, k)),
                _ => None,
            })
            .unwrap_or((id, 1)),
        _ => (id, 1),
    })
}

/// Numeric literal factors are spectators in inverse-cancellation detection
/// (`½` in `x² · x⁻¹ · ½`); they never participate in the exponent rewrite.
fn is_numeric_literal(id: ExprId, pool: &ExprPool) -> bool {
    pool.with(id, |d| {
        matches!(
            d,
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
        )
    })
}

/// If `before = a^m * a^n` for integer `m, n` with at least one negative
/// (i.e. the product genuinely routes through an inverse — same-sign
/// integer powers combine soundly via plain `ring` and are left alone), and
/// `after` is the fully-collapsed `a^(m+n)` (rendered as `1` when `m+n = 0`),
/// return `a`. This is the shape produced by `collect_mul_factors` (and
/// similar cleanup rules) — e.g. `cos x * (cos x)⁻¹ = 1`, `x² * x⁻² = 1`,
/// or `x⁻² * x⁵ = x³`.
///
/// Critically, this claim is only true when `a ≠ 0`: Lean's junk-value
/// convention (`0⁻¹ = 0`) makes `0^m` for `m < 0` evaluate to `0` rather than
/// diverging, so e.g. `0² * 0⁻² = 0 ≠ 1`. `ring`/`norm_num` — which only
/// prove identities that hold unconditionally — cannot close it. The static
/// per-rule-name tactic table (`"by ring"` for `collect_mul_factors`) cannot
/// be trusted whenever this shape is detected; see
/// [`inv_cancel_certificate`] for the actual closing tactic.
///
/// Only a **two-factor** product of like powers is encoded. An n-ary product
/// with a spectator coefficient — `x² · x⁻¹ · (1/2) = x · (1/2)` — is the
/// same cancellation mathematically, but [`inv_cancel_certificate`] does not
/// cover it; [`nary_inverse_cancelled`] makes the caller withhold rather
/// than emit a `by ring` that Lean cannot close (`Try this: ring_nf` under
/// `warningAsError`).
///
/// Returns `(base, net_exponent)`. `net_exponent` tells the caller whether
/// `field_simp` alone fully closes the goal: empirically, when the net
/// exponent is `0` (the `after = 1` case), `field_simp`'s own simp set
/// closes it outright, but for a nonzero net exponent (`after = a^k`,
/// `k ≠ 0`) it leaves a genuine commutative-ring rearrangement (e.g.
/// `x⁵ = x³ * x²`) that needs a following `ring`. Emitting an unconditional
/// `field_simp [hne]; ring` would trip Lean's `unreachableTactic` linter
/// (promoted to a hard error by `-DwarningAsError=true`) on the `net = 0`
/// cases, where `ring` would never run — so the caller must branch on this.
fn inv_cancel_base(before: ExprId, after: ExprId, pool: &ExprPool) -> Option<(ExprId, i64)> {
    let (a, b) = pool.with(before, |d| match d {
        ExprData::Mul(xs) if xs.len() == 2 => Some((xs[0], xs[1])),
        _ => None,
    })?;
    let (base_a, exp_a) = factor_base_exp(a, pool);
    let (base_b, exp_b) = factor_base_exp(b, pool);
    if base_a != base_b || (exp_a >= 0 && exp_b >= 0) {
        return None;
    }
    let net = exp_a + exp_b;
    let expected_after = if net == 0 {
        pool.integer(1_i32)
    } else {
        pool.pow(base_a, pool.integer(net))
    };
    (after == expected_after).then_some((base_a, net))
}

/// Build a nonzero-hypothesis certificate for an [`inv_cancel_base`] shape,
/// when the canceled base is a bare symbol or a known-total unary primitive
/// (`sin`/`cos`/`exp`) applied to one — the same primitive family covered
/// elsewhere in this module, all defined (and hence junk-value-safe) at
/// every real point. Returns `None` for anything else; callers must
/// withhold rather than trust the static "by ring" table entry.
fn inv_cancel_certificate(
    before: ExprId,
    after: ExprId,
    pool: &ExprPool,
) -> Option<(String, String)> {
    let (base, net) = inv_cancel_base(before, after, pool)?;
    // See `inv_cancel_base`'s doc: only append `ring` when it's actually
    // going to run (net ≠ 0), or the `unreachableTactic` linter fires.
    let tactic = if net == 0 {
        "by field_simp [hne]".to_string()
    } else {
        "by\n    field_simp [hne]\n    ring".to_string()
    };
    if let Some(sym) = symbol_name(base, pool) {
        let binder = format!("({sym} : ℝ) (hne : {sym} ≠ 0)");
        return Some((binder, tactic));
    }
    let name = pool.with(base, |d| match d {
        ExprData::Func { name, args } if args.len() == 1 => {
            symbol_name(args[0], pool).map(|sym| (name.clone(), sym))
        }
        _ => None,
    })?;
    let (name, sym) = name;
    if !matches!(name.as_str(), "sin" | "cos" | "exp") {
        return None;
    }
    let binder = format!("({sym} : ℝ) (hne : Real.{name} {sym} ≠ 0)");
    Some((binder, tactic))
}

/// Flat factors of a `Mul`, or the expression itself if it isn't a product.
fn mul_factors(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    pool.with(expr, |d| match d {
        ExprData::Mul(xs) => xs.clone(),
        _ => vec![expr],
    })
}

/// Sorted exponent list per non-literal base among the flat factors of `expr`.
fn exponent_multiset(
    expr: ExprId,
    pool: &ExprPool,
) -> std::collections::BTreeMap<ExprId, Vec<i64>> {
    let mut m: std::collections::BTreeMap<ExprId, Vec<i64>> = std::collections::BTreeMap::new();
    for fac in mul_factors(expr, pool) {
        let (base, exp) = factor_base_exp(fac, pool);
        if is_numeric_literal(base, pool) {
            continue;
        }
        m.entry(base).or_default().push(exp);
    }
    for v in m.values_mut() {
        v.sort_unstable();
    }
    m
}

/// True when an **n-ary** product (`≥ 3` factors) rewrites inverse powers of
/// some base so that the minimum exponent of that base *increases* (the
/// inverse was cancelled, not merely reordered or regrouped).
///
/// This is the shape `collect_mul_factors` produces for `d/dx (½ x² log x)`:
/// `x² · x⁻¹ · (1/2) = x · (1/2)`. `ring` cannot close it (`Try this:
/// ring_nf` under `warningAsError`), and [`inv_cancel_certificate`] only
/// encodes the two-factor collapse. Callers must withhold.
///
/// A two-factor product is left to [`inv_cancel_certificate`]. A regrouping
/// that *moves* the inverse onto a new compound base (e.g. `√x⁻¹ · ½ =
/// (2 √x)⁻¹`) is not cancellation and is left alone. A mere reorder
/// (`x⁻¹ · 2 = 2 · x⁻¹`) keeps the same min exponent.
fn nary_inverse_cancelled(before: ExprId, after: ExprId, pool: &ExprPool) -> bool {
    if mul_factors(before, pool).len() < 3 {
        return false;
    }
    let before_exps = exponent_multiset(before, pool);
    let after_exps = exponent_multiset(after, pool);
    before_exps.iter().any(|(&base, bexps)| {
        let Some(&bmin) = bexps.iter().min() else {
            return false;
        };
        if bmin >= 0 {
            return false;
        }
        match after_exps.get(&base).and_then(|v| v.iter().min().copied()) {
            Some(amin) => amin > bmin,
            None => {
                // Base vanished: cancellation, unless the inverse moved onto
                // a new compound base (a regrouping, not a collapse).
                !after_exps.iter().any(|(ab, aexps)| {
                    !before_exps.contains_key(ab) && aexps.iter().any(|&e| e < 0)
                })
            }
        }
    })
}

/// `before = (a⁻¹)⁻¹`, `after = a` or `a^1` — Mathlib's `inv_inv : a⁻¹⁻¹ = a`
/// holds *unconditionally* (at `a = 0`: `0⁻¹ = 0`, so `(0⁻¹)⁻¹ = 0⁻¹ = 0 =
/// a`), unlike [`inv_cancel_base`]'s shapes. `ring` still can't close it —
/// `ring` treats `⁻¹` as an opaque atom and doesn't know the involution law
/// — so this needs the dedicated `inv_inv` simp lemma instead. Returns
/// `true` when this shape matches.
fn is_double_inv_cancel(before: ExprId, after: ExprId, pool: &ExprPool) -> bool {
    let base = pool.with(before, |d| match d {
        ExprData::Pow { base, exp }
            if pool.with(*exp, |e| matches!(e, ExprData::Integer(n) if n.0 == -1)) =>
        {
            pool.with(*base, |bd| match bd {
                ExprData::Pow { base: b2, exp: e2 }
                    if pool.with(*e2, |e| matches!(e, ExprData::Integer(n) if n.0 == -1)) =>
                {
                    Some(*b2)
                }
                _ => None,
            })
        }
        _ => None,
    });
    match base {
        Some(a) => after == a || after == pool.pow(a, pool.integer(1_i32)),
        None => false,
    }
}

/// Certificate for a `sin_double_angle` fold `2 · sin u · cos u = sin(2u)`
/// (Alkahest stores the LHS as `sin u · 2 · cos u` and the RHS as `sin(u·2)`).
/// `Real.sin_two_mul u : sin (2·u) = 2 · sin u · cos u` is *unconditional*, so
/// no side condition is needed — we only have to reconcile the `u·2` vs `2·u`
/// argument order (a `mul_comm` rewrite confined to the sine's argument) and
/// the factor ordering (`ring`). Returns `None` (→ withhold) if `before` is
/// not literally a product containing a `sin` factor.
fn sin_double_angle_certificate(before: ExprId, pool: &ExprPool) -> Option<String> {
    let factors = pool.with(before, |d| match d {
        ExprData::Mul(xs) => Some(xs.clone()),
        _ => None,
    })?;
    let arg = factors.iter().find_map(|&fac| {
        pool.with(fac, |d| match d {
            ExprData::Func { name, args } if name == "sin" && args.len() == 1 => Some(args[0]),
            _ => None,
        })
    })?;
    let arg_str = expr_to_lean(arg, pool);
    Some(format!(
        "by rw [mul_comm ({arg_str} : ℝ) (2 : ℝ), Real.sin_two_mul]; ring"
    ))
}

/// Resolve `(explicit_binders, tactic)` for a non-differentiation ("plain
/// equality `before = after`") rewrite step. Tries, in order: the
/// [`inv_cancel_certificate`] soundness override and the
/// [`is_double_inv_cancel`] unconditional override (whenever either shape
/// matches, the static table tactic is never trusted, even if it doesn't
/// literally contain `"sorry"`), a withhold for n-ary inverse cancellation
/// beyond that two-factor encoding ([`nary_inverse_cancelled`]), the
/// [`positivity_certificate`] upgrade, and finally the static per-rule-name
/// tactic ([`rule_to_tactic`]) when it doesn't require `sorry`. Returns
/// `None` when the step must be withheld.
fn plain_step_certificate(step: &RewriteStep, pool: &ExprPool) -> Option<(Option<String>, String)> {
    // `sin_double_angle` is unconditional but needs a shape-specific rewrite
    // (`Real.sin_two_mul`); the static table's `ring_nf; simp` fallback cannot
    // close it. Handle it here (withholding if the product shape is unexpected)
    // so it never reaches — and wrongly trusts — that fallback.
    if step.rule_name == "sin_double_angle" {
        return sin_double_angle_certificate(step.before, pool).map(|t| (None, t));
    }
    if inv_cancel_base(step.before, step.after, pool).is_some() {
        return inv_cancel_certificate(step.before, step.after, pool).map(|(b, t)| (Some(b), t));
    }
    if is_double_inv_cancel(step.before, step.after, pool) {
        return Some((None, "by simp [inv_inv]".to_string()));
    }
    if nary_inverse_cancelled(step.before, step.after, pool) {
        // n-ary inverse cancellation (`x² · x⁻¹ · ½ = x · ½`) is beyond the
        // two-factor [`inv_cancel_certificate`]; `ring` cannot close it and
        // Lean suggests `ring_nf` as a warning. Withhold the whole certificate
        // rather than emit a non-typechecking file under `warningAsError`.
        return None;
    }
    if let Some((binders, tactic)) = positivity_certificate(step, pool) {
        return Some((Some(binders), tactic));
    }
    let tactic = rule_to_tactic(step.rule_name);
    if tactic.contains("sorry") {
        None
    } else {
        Some((None, tactic.to_string()))
    }
}

/// Whether this step can be emitted as a Lean `example` expected to typecheck
/// without `sorry` / `admit`.
///
/// Public so the certificate ledger (`alkahest.certifiable`) can report *which*
/// step blocks a withheld certificate instead of a bare "no". Note that a
/// derivation whose every step is certifiable is not automatically certifiable
/// as a whole: integration logs certify through the FTC relation rather than
/// step-by-step, so this predicate is a diagnostic, not the emission gate. The
/// emission gate is [`emit_lean_expr_wrt`] returning a non-empty string.
pub fn step_is_certifiable(step: &RewriteStep, wrt: Option<ExprId>, pool: &ExprPool) -> bool {
    if is_integration_rule(step.rule_name) {
        return false;
    }
    if let Some(var) = wrt {
        if is_differentiation_rule(step.rule_name) {
            return diff_step_certificate(step, var, pool).is_some();
        }
        // Algebraic cleanup steps in a diff log use plain equality goals.
        return plain_step_certificate(step, pool).is_some();
    }
    plain_step_certificate(step, pool).is_some()
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
     import Mathlib.Algebra.Order.Group.Abs\n\
     \n\
     open Real\n\n"
        .to_string()
}

/// Emit the Lean 4 file header for differentiation certificates.
pub fn emit_diff_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.Calculus.Deriv.Basic\n\
     import Mathlib.Analysis.Calculus.Deriv.Pow\n\
     import Mathlib.Analysis.Calculus.Deriv.Mul\n\
     import Mathlib.Analysis.Calculus.Deriv.Inv\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.InverseDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.ExpDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.Log.Deriv\n\
     import Mathlib.Analysis.SpecialFunctions.Sqrt\n\
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
    // The tactics below cite bare Mathlib lemmas stated about `-x`, so the
    // goal has to print the negation rather than the kernel's `x * -1`.
    let body = expr_to_lean_neg(expr, pool);
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

/// The `deriv (fun v => before) v = after` body of a differentiation goal,
/// without the leading `example [binders] :`. Shared by [`emit_diff_goal`]
/// (no binders) and [`emit_step_wrt`]'s hypothesis-gated certificates
/// (`diff_sqrt`, `tan`, the power/quotient chains — which prepend explicit
/// `(x : ℝ) (h... : ...)` binders before this same body).
fn diff_goal_body(before: ExprId, after: ExprId, wrt: ExprId, pool: &ExprPool) -> String {
    let var_name = wrt_name(wrt, pool);
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
    format!("deriv (fun ({binder} : ℝ) => {before_str}) {var_name} = {after_str}")
}

/// Emit a Lean `example` asserting `deriv (fun v => before) v = after`.
pub fn emit_diff_goal(before: ExprId, after: ExprId, wrt: ExprId, pool: &ExprPool) -> String {
    format!("example : {}", diff_goal_body(before, after, wrt, pool))
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

/// Append the `-- Side conditions: …` trailer (if any) recorded on `step`.
fn append_side_conditions(out: &mut String, step: &RewriteStep, pool: &ExprPool) {
    if step.side_conditions.is_empty() {
        return;
    }
    out.push_str("\n  -- Side conditions: ");
    let conds: Vec<String> = step
        .side_conditions
        .iter()
        .map(|c| c.display_with(pool).to_string())
        .collect();
    out.push_str(&conds.join(", "));
}

/// Like [`emit_step`], but when `wrt` is set, differentiation rules emit a
/// `deriv` goal while algebraic cleanup steps in the same log stay plain
/// equalities (so `mul_one` is not wrongly wrapped as `deriv (1·cos) = cos`).
pub fn emit_step_wrt(step: &RewriteStep, pool: &ExprPool, wrt: Option<ExprId>) -> String {
    let diff_step = wrt.is_some() && is_differentiation_rule(step.rule_name);

    if let (true, Some(var)) = (diff_step, wrt) {
        let (binders, tactic) =
            diff_step_certificate(step, var, pool).unwrap_or((None, "by sorry".to_string()));
        let body = diff_goal_body(step.before, step.after, var, pool);
        let mut out = match binders {
            Some(b) => format!("example {b} : {body} :=\n  {tactic}"),
            None => format!("example : {body} :=\n  {tactic}"),
        };
        append_side_conditions(&mut out, step, pool);
        return out;
    }

    // Plain equality goal: either `wrt` is unset entirely, or this is an
    // algebraic-cleanup step inside a diff log (e.g. `mul_one`,
    // `collect_mul_factors`) — both share the same `before = after` shape,
    // optionally upgraded with an explicit hypothesis binder.
    let (binders, tactic) =
        plain_step_certificate(step, pool).unwrap_or((None, "by sorry".to_string()));
    let before_str = expr_to_lean(step.before, pool);
    let after_str = expr_to_lean(step.after, pool);
    let mut out = match binders {
        Some(b) => format!("example {b} : {before_str} = {after_str} :=\n  {tactic}"),
        None => format!("example : {before_str} = {after_str} :=\n  {tactic}"),
    };
    append_side_conditions(&mut out, step, pool);
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
/// *pointwise* `sin`/`cos`/`exp`/`log`/`atan` (argument exactly the variable), sums of
/// those, and *flat* products of those (a product whose factors are atoms /
/// pointwise primitives, e.g. `x · cos x` or `x · log x`). Two shapes that the
/// diff exporter currently emits but does **not** discharge — leaving `deriv`
/// or a `DifferentiableAt` side goal open — must be withheld here:
///
/// * a **chain composite** `f(g x)` with `g ≠ x` (e.g. `exp (x²)`), because the
///   product-rule simp set lacks the composite's `DifferentiableAt` lemma;
/// * a **sum nested inside a product** (e.g. `-1 · (a + b)`), because the
///   post-`simp` `ring` cannot reduce the still-symbolic nested `deriv`.
///
/// `log` is included so that `∫ x⁻¹ dx = log x` certifies once `d/dx log(x)`
/// is known to close (`Real.deriv_log`). `atan` is included so that
/// `∫ (1+x²)⁻¹ dx = atan x` certifies once [`registry_diff_certificate`] can
/// close `d/dx atan(x)`. `sinh`/`cosh` are not needed here (no matching
/// integration rule). `asin` stays out: its derivative certificate carries an
/// `|x| < 1` binder that the reused FTC path does not thread. A *product* of
/// constants and a negative power of the variable (e.g. the antiderivative
/// `-x⁻¹` of `x⁻²`) is in the fragment: that derivative log's `product_rule`
/// step is closed by the second combine fragment ([`diff_body_neg_pow_combine`]),
/// not the unconditional simp set.
///
/// Rejecting these keeps the integration certificate sound: a withheld integral
/// is always preferable to a `.lean` file that fails to typecheck. Composites
/// and by-parts results outside this fragment simply stay withheld.
///
/// Passing this gate is **not** sufficient. The reused exporter still has to
/// certify every algebraic cleanup step of `d/dx F`. An n-ary inverse
/// cancellation such as `x² · x⁻¹ · (1/2) = x · (1/2)` (the residual of
/// `∫ x log x`) is beyond the two-factor [`inv_cancel_certificate`] and is
/// withheld there — intern-equality of `d/dx F` with the integrand must not
/// punch a hole in that discipline.
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
                // Pointwise primitive only: sin/cos/exp/log/atan applied to exactly `var`.
                // `log` is included so ∫ log x (antiderivative `x·log x − x`) can
                // reuse the log/sqrt combine certificate; composites stay out.
                matches!(name.as_str(), "sin" | "cos" | "exp" | "log" | "atan")
                    && args.len() == 1
                    && args[0] == var
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
/// simplified `d/dx F` to intern to the same [`ExprId`] as the simplified
/// `integrand` (so a `1 *` decoration on either side does not withhold a
/// mathematically identical residual); otherwise we WITHHOLD. This is
/// precisely the exact-residual antiderivative check, so numeric-only
/// antiderivatives stay withheld.
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
    // `∫ f = F` when `d/dx F` is the integrand. `diff` already runs `simplify`
    // on its result (stripping a `1 *` from `d/dx atan(x) = 1 · (1+x²)⁻¹`
    // and from `d/dx log(x) = 1 · x⁻¹`), so compare against the simplified
    // integrand — otherwise Python's `1/(1+x**2)` / `1/x` would fail
    // intern-equality against a mathematically identical derivative.
    let integrand_canon = crate::simplify::engine::simplify(integrand, pool).value;
    if derived_diff.value != integrand_canon {
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

/// Lean import header for **definite** interval-integral certificates.
///
/// Adds Mathlib's interval-integral and second FTC lemmas
/// (`intervalIntegral.integral_eq_sub_of_hasDerivAt`) on top of the derivative
/// lemmas the `HasDerivAt` obligations need (`Real.hasDerivAt_sin`,
/// `Real.hasDerivAt_exp`, `hasDerivAt_pow`, …).
fn emit_definite_integral_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv\n\
     import Mathlib.Analysis.SpecialFunctions.ExpDeriv\n\
     import Mathlib.Analysis.Calculus.Deriv.Pow\n\
     import Mathlib.MeasureTheory.Integral.IntervalIntegral\n\
     import Mathlib.MeasureTheory.Integral.FundThmCalculus\n\
     \n\
     open Real\n\n"
        .to_string()
}

/// The certifiable definite-integral fragment: an integrand class for which we
/// can emit a `HasDerivAt` witness on `Set.uIcc a b` and an
/// `IntervalIntegrable` side condition that Lean reliably discharges.
///
/// Every variant pins a concrete antiderivative `F` with `d/dx F = f` on all of
/// `ℝ`, so the resulting certificate proves the *sound* interval-FTC statement
/// `∫ x in a..b, f x = F b - F a` — never a false equality.
enum DefiniteIntegrandClass {
    /// `∫ cos x`, antiderivative `sin x`.
    Cos,
    /// `∫ sin x`, antiderivative `-cos x`.
    Sin,
    /// `∫ exp x`, antiderivative `exp x`.
    Exp,
    /// `∫ xⁿ` (`n ≥ 1`), antiderivative `x^(n+1) / (n+1)`.
    Pow(i64),
    /// `∫ (1+x²)⁻¹`, antiderivative `arctan x`.
    /// `one_left` records intern order: `1 + x²` vs `x² + 1`, so the
    /// `Continuous.inv₀` witness matches the printed lambda definitionally
    /// (addition is not defeq-commutative).
    InvOnePlusSq { one_left: bool },
}

/// True when `expr` is the integer (or 1/1 rational) `k`.
fn is_int_literal(expr: ExprId, k: i64, pool: &ExprPool) -> bool {
    pool.with(expr, |d| match d {
        ExprData::Integer(n) => n.0.to_i64() == Some(k),
        ExprData::Rational(r) => r.0.numer().to_i64() == Some(k) && r.0.denom().to_i64() == Some(1),
        _ => false,
    })
}

/// True when `expr` is structurally `var ^ 2`.
fn is_var_squared(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| match d {
        ExprData::Pow { base, exp } => {
            *base == var
                && pool.with(*exp, |e| match e {
                    ExprData::Integer(n) => n.0.to_i64() == Some(2),
                    _ => false,
                })
        }
        _ => false,
    })
}

/// If `expr` is `(1 + var²)⁻¹` or `(var² + 1)⁻¹`, return whether the `1` is
/// the left addend. Alkahest stores the reciprocal as `Pow(..., -1)`, never
/// as a `div` node. A surrounding `1 *` scalar is handled by
/// [`classify_definite_atom`]'s constant-multiple wrapper, not here.
fn inv_one_plus_var_sq_one_left(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<bool> {
    pool.with(expr, |d| match d {
        ExprData::Pow { base, exp } => {
            let is_inv = pool.with(*exp, |e| match e {
                ExprData::Integer(n) => n.0.to_i64() == Some(-1),
                _ => false,
            });
            if !is_inv {
                return None;
            }
            pool.with(*base, |b| match b {
                ExprData::Add(xs) if xs.len() == 2 => {
                    let (a, b) = (xs[0], xs[1]);
                    if is_int_literal(a, 1, pool) && is_var_squared(b, var, pool) {
                        Some(true)
                    } else if is_int_literal(b, 1, pool) && is_var_squared(a, var, pool) {
                        Some(false)
                    } else {
                        None
                    }
                }
                _ => None,
            })
        }
        _ => None,
    })
}

/// Classify `integrand` into the certifiable definite-integral *base* fragment
/// (a single pointwise `sin`/`cos`/`exp`, an integer power, the bare
/// variable, or `(1+x²)⁻¹`). [`classify_definite_atom`] extends this with an
/// optional constant-multiple wrapper, and [`build_definite_pieces`] extends
/// it further to finite sums — this function only recognises the un-scaled
/// building block.
///
/// Returns `None` (⇒ withhold) for anything outside the pointwise
/// `sin`/`cos`/`exp` of the integration variable, a positive integer power
/// of the variable, the bare variable itself (treated as `x¹`), or
/// `(1+x²)⁻¹`. Products, composites, and every other shape stay withheld —
/// a missing certificate is always preferable to an unsound or
/// non-compiling one.
fn classify_definite_integrand(
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DefiniteIntegrandClass> {
    if let Some(one_left) = inv_one_plus_var_sq_one_left(integrand, var, pool) {
        return Some(DefiniteIntegrandClass::InvOnePlusSq { one_left });
    }
    pool.with(integrand, |d| match d {
        ExprData::Symbol { .. } if integrand == var => Some(DefiniteIntegrandClass::Pow(1)),
        ExprData::Func { name, args } if args.len() == 1 && args[0] == var => match name.as_str() {
            "sin" => Some(DefiniteIntegrandClass::Sin),
            "cos" => Some(DefiniteIntegrandClass::Cos),
            "exp" => Some(DefiniteIntegrandClass::Exp),
            _ => None,
        },
        ExprData::Pow { base, exp } if *base == var => pool.with(*exp, |e| match e {
            ExprData::Integer(n) => {
                n.0.to_i64()
                    .filter(|&k| k >= 1)
                    .map(DefiniteIntegrandClass::Pow)
            }
            _ => None,
        }),
        _ => None,
    })
}

/// True for a numeric literal (`Integer` or `Rational`) — the only shapes
/// accepted as a constant-multiple coefficient in the definite-integral
/// linear-combination fragment. Such a literal never mentions the
/// integration variable, so no separate free-variable check is needed.
fn is_definite_coeff_literal(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| {
        matches!(d, ExprData::Integer(_) | ExprData::Rational(_))
    })
}

/// One additive term of the definite-integral **linear-combination**
/// fragment: a certifiable base class, optionally scaled by a numeric literal
/// coefficient. Alkahest interns `Mul` children sorted by raw [`ExprId`]
/// (commutative canonicalisation), *not* "coefficient first" — a coefficient
/// can end up on either side — so the two scaled variants record which side
/// the coefficient was found on. [`emit_definite_integration_cert`] uses this
/// to pick `HasDerivAt.const_mul`/`.mul_const` (and the `IntervalIntegrable`
/// analogues) so the certificate's derivative-value term stays syntactically
/// aligned with how the integrand's `Mul` node actually prints.
enum DefiniteAtom {
    /// A bare base term, coefficient 1 (e.g. `cos x`).
    Bare(DefiniteIntegrandClass),
    /// `coeff * base` as literally interned (e.g. `3 * cos x`).
    CoeffLeft(ExprId, DefiniteIntegrandClass),
    /// `base * coeff` as literally interned (e.g. `cos x * 3`).
    CoeffRight(DefiniteIntegrandClass, ExprId),
}

/// Classify one additive term (or a whole non-sum integrand) into the
/// definite-integral linear-combination fragment: a bare base class, or a
/// `Mul` of exactly two factors where one is a numeric literal
/// ([`is_definite_coeff_literal`]) and the other is a bare base class.
///
/// Returns `None` (⇒ withhold that term, and hence the whole certificate) for
/// anything else — in particular, three-or-more-factor products and
/// non-literal coefficients (which could depend on the integration variable)
/// are never accepted.
fn classify_definite_atom(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<DefiniteAtom> {
    if let Some(class) = classify_definite_integrand(expr, var, pool) {
        return Some(DefiniteAtom::Bare(class));
    }
    pool.with(expr, |d| match d {
        ExprData::Mul(xs) if xs.len() == 2 => {
            let (a, b) = (xs[0], xs[1]);
            if is_definite_coeff_literal(a, pool) {
                classify_definite_integrand(b, var, pool).map(|c| DefiniteAtom::CoeffLeft(a, c))
            } else if is_definite_coeff_literal(b, pool) {
                classify_definite_integrand(a, var, pool).map(|c| DefiniteAtom::CoeffRight(c, b))
            } else {
                None
            }
        }
        _ => None,
    })
}

/// `(antiderivative_at(t), HasDerivAt witness term, IntervalIntegrable witness
/// term)` for a bare base class, parameterised by the point/binder name
/// `var_name`. The two witness terms are bare Lean terms (no `exact`/`simpa`
/// wrapper) so [`atom_pieces`] and [`emit_definite_integration_cert`] can
/// compose them with `.const_mul` / `.mul_const` / `.add` before a single
/// top-level `simpa` closes the whole chain (bridging e.g. the double
/// negation `-(-sin x)` left over from [`DefiniteIntegrandClass::Sin`]'s
/// `.neg`, or the numeric division left over from
/// [`DefiniteIntegrandClass::Pow`]'s `.div_const`).
#[allow(clippy::type_complexity)]
fn base_pieces(
    class: &DefiniteIntegrandClass,
    var_name: &str,
) -> (Box<dyn Fn(&str) -> String>, String, String) {
    match class {
        DefiniteIntegrandClass::Cos => (
            Box::new(|t: &str| format!("Real.sin ({t})")),
            format!("Real.hasDerivAt_sin {var_name}"),
            "Real.continuous_cos.intervalIntegrable _ _".to_string(),
        ),
        DefiniteIntegrandClass::Sin => (
            Box::new(|t: &str| format!("-Real.cos ({t})")),
            format!("(Real.hasDerivAt_cos {var_name}).neg"),
            "Real.continuous_sin.intervalIntegrable _ _".to_string(),
        ),
        DefiniteIntegrandClass::Exp => (
            Box::new(|t: &str| format!("Real.exp ({t})")),
            format!("Real.hasDerivAt_exp {var_name}"),
            "Real.continuous_exp.intervalIntegrable _ _".to_string(),
        ),
        DefiniteIntegrandClass::Pow(n) => {
            let m = n + 1;
            let n = *n;
            (
                Box::new(move |t: &str| format!("({t}) ^ ({m} : ℕ) / ({m})")),
                format!("(hasDerivAt_pow {m} {var_name}).div_const {m}"),
                // `∫ xⁿ`: the integrand `fun x => xⁿ` is continuous. `x¹` is
                // written by Alkahest as the bare symbol, so its integrand is
                // `id`; higher powers use `continuous_pow`.
                if n == 1 {
                    "continuous_id.intervalIntegrable _ _".to_string()
                } else {
                    format!("(continuous_pow {n}).intervalIntegrable _ _")
                },
            )
        }
        DefiniteIntegrandClass::InvOnePlusSq { one_left } => {
            // Match intern order so the Continuous term is definitionally the
            // printed integrand (addition is not defeq-commutative). The
            // positivity witness has to follow the same order: `0 < 1 + x²`
            // uses `pos + nonneg`, `0 < x² + 1` uses `nonneg + pos`.
            let (add_cont, ne_proof) = if *one_left {
                (
                    "(continuous_const.add (continuous_pow 2))",
                    "(add_pos_of_pos_of_nonneg zero_lt_one (sq_nonneg t)).ne'",
                )
            } else {
                (
                    "((continuous_pow 2).add continuous_const)",
                    "(add_pos_of_nonneg_of_pos (sq_nonneg t) zero_lt_one).ne'",
                )
            };
            (
                Box::new(|t: &str| format!("Real.arctan ({t})")),
                format!("Real.hasDerivAt_arctan' {var_name}"),
                format!("({add_cont}.inv₀ (fun t => {ne_proof})).intervalIntegrable _ _"),
            )
        }
    }
}

/// Like [`base_pieces`], but for a full [`DefiniteAtom`] — wraps the base
/// class's witnesses with `HasDerivAt.const_mul`/`.mul_const` and
/// `IntervalIntegrable.const_mul`/`.mul_const` when the atom carries a
/// constant-multiple coefficient, on whichever side it was interned.
#[allow(clippy::type_complexity)]
fn atom_pieces(
    atom: &DefiniteAtom,
    var_name: &str,
    pool: &ExprPool,
) -> (Box<dyn Fn(&str) -> String>, String, String) {
    match atom {
        DefiniteAtom::Bare(class) => base_pieces(class, var_name),
        DefiniteAtom::CoeffLeft(coeff, class) => {
            let (base_anti, base_hderiv, base_int) = base_pieces(class, var_name);
            let c_lean = expr_to_lean(*coeff, pool);
            let anti_c = c_lean.clone();
            let anti: Box<dyn Fn(&str) -> String> =
                Box::new(move |t: &str| format!("({anti_c}) * ({})", base_anti(t)));
            (
                anti,
                format!("({base_hderiv}).const_mul ({c_lean})"),
                format!("({base_int}).const_mul ({c_lean})"),
            )
        }
        DefiniteAtom::CoeffRight(class, coeff) => {
            let (base_anti, base_hderiv, base_int) = base_pieces(class, var_name);
            let c_lean = expr_to_lean(*coeff, pool);
            let anti_c = c_lean.clone();
            let anti: Box<dyn Fn(&str) -> String> =
                Box::new(move |t: &str| format!("({}) * ({anti_c})", base_anti(t)));
            (
                anti,
                format!("({base_hderiv}).mul_const ({c_lean})"),
                format!("({base_int}).mul_const ({c_lean})"),
            )
        }
    }
}

/// Combine two already-built `(antiderivative, HasDerivAt term,
/// IntervalIntegrable term)` triples into the triple for their sum, via
/// `HasDerivAt.add` / `IntervalIntegrable.add`.
#[allow(clippy::type_complexity)]
fn combine_definite_add(
    a: (Box<dyn Fn(&str) -> String>, String, String),
    b: (Box<dyn Fn(&str) -> String>, String, String),
) -> (Box<dyn Fn(&str) -> String>, String, String) {
    let (a_anti, a_hderiv, a_int) = a;
    let (b_anti, b_hderiv, b_int) = b;
    // Explicit parens on both sides: the stated `have hderiv` goal's function
    // lambda must match — up to Lean's own parser, not just mathematically —
    // the exact nesting `HasDerivAt.add` gives the proof term (see the
    // doc-comment on `build_definite_pieces`), and a bare `a + b` reparsed
    // inside a larger flat join would silently re-associate.
    let anti: Box<dyn Fn(&str) -> String> =
        Box::new(move |t: &str| format!("({}) + ({})", a_anti(t), b_anti(t)));
    (
        anti,
        format!("({a_hderiv}).add ({b_hderiv})"),
        format!("({a_int}).add ({b_int})"),
    )
}

/// Classify `expr` into the definite-integral linear-combination fragment and
/// build its `(antiderivative, HasDerivAt term, IntervalIntegrable term)`
/// triple, recursing into `Add` nodes exactly as deep and in exactly the
/// order [`expr_to_lean`] does.
///
/// Recursing on the *actual* `Add` tree (rather than flattening to a term
/// list first) matters because Alkahest's raw expression builder does not
/// re-associate `+` at construction time: Python's chained `x**2 + sin(x) +
/// 3*cos(x)` interns as a 2-ary `Add` whose *first* child is itself an `Add`
/// (`Add([Add([x², sin x]), 3·cos x])`) — but commutative canonicalisation
/// (children sorted by raw [`ExprId`]) can just as easily put the nested
/// `Add` on the *right* (`Add([3·cos x, Add([sin x, x²])])`). Because
/// [`expr_to_lean`] parenthesizes every `Add` node's own rendering
/// (`"(" + parts.join(" + ") + ")"`), a right-nested tree like that prints as
/// `a + (b + c)`, which is a *different* parse tree from the left-associated
/// `(a + b) + c` a naive flatten-then-left-fold would produce — and a
/// certificate whose combinator chain doesn't structurally match the printed
/// derivative-value term fails to typecheck (mismatched associativity is not
/// bridged by `simpa`, since the mismatch sits under the `HasDerivAt` value
/// argument, not at the goal's outermost head). Recursing on the real tree
/// and combining each `Add` node's own (already-recursively-built) children
/// left-to-right reproduces the exact same grouping [`expr_to_lean`] prints,
/// for arbitrary nesting *and* arbitrary flat arity (an `n`-ary `Add` node
/// with `n ≥ 3` children prints as a single flat `a + b + … `, which Lean
/// itself parses left-associatively — matching a left-fold over that node's
/// direct children).
///
/// Returns `None` (⇒ withhold) as soon as any addend — at any depth — escapes
/// the fragment; a partially-certified sum is never emitted.
#[allow(clippy::type_complexity)]
fn build_definite_pieces(
    expr: ExprId,
    var: ExprId,
    var_name: &str,
    pool: &ExprPool,
) -> Option<(Box<dyn Fn(&str) -> String>, String, String)> {
    if let Some(atom) = classify_definite_atom(expr, var, pool) {
        return Some(atom_pieces(&atom, var_name, pool));
    }
    let xs = pool.with(expr, |d| match d {
        ExprData::Add(xs) if xs.len() >= 2 => Some(xs.clone()),
        _ => None,
    })?;
    let mut xs = xs.into_iter();
    let mut acc = build_definite_pieces(xs.next().expect("len >= 2"), var, var_name, pool)?;
    for child in xs {
        let piece = build_definite_pieces(child, var, var_name, pool)?;
        acc = combine_definite_add(acc, piece);
    }
    Some(acc)
}

/// Strictly positive numeric literal (`Integer` or `Rational`). Used to
/// discharge `0 < a` / `0 < b` with `norm_num` on a definite `∫ log` cert,
/// rather than asking for binders. Floats, zeros, and negatives are not
/// accepted — `∫_0^1 log` is singular and negative endpoints stay withheld.
fn bound_is_strictly_positive_literal(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| match d {
        ExprData::Integer(n) => n.0 > 0,
        ExprData::Rational(r) => r.0.numer().cmp0() == std::cmp::Ordering::Greater,
        _ => false,
    })
}

/// How a definite-`∫ log` certificate will pin `0 < a` and `0 < b`.
enum LogEndpointPositivity {
    /// Both endpoints are strictly positive literals; `norm_num` closes them.
    Literals,
    /// Bind `(a : ℝ) (b : ℝ) (ha : 0 < a) (hb : 0 < b)`. When the two names
    /// coincide (`∫_a^a`), a single binder is emitted and reused for both.
    Symbols { a: String, b: String },
}

/// Classify the endpoints of `∫_a^b log x` into a positivity encoding Lean
/// can discharge. Returns `None` (⇒ withhold) unless both bounds are strictly
/// positive literals, or both are symbols distinct from the integration
/// variable (so the extra binders cannot shadow `x`). Negative, zero, mixed,
/// compound, or infinite endpoints stay withheld — Mathlib's
/// `integral_log_of_neg` is deliberately unused.
fn classify_log_endpoints(
    lower: ExprId,
    upper: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<LogEndpointPositivity> {
    if bound_is_strictly_positive_literal(lower, pool)
        && bound_is_strictly_positive_literal(upper, pool)
    {
        return Some(LogEndpointPositivity::Literals);
    }
    let a = symbol_name(lower, pool)?;
    let b = symbol_name(upper, pool)?;
    let var_name = wrt_name(var, pool);
    if a == var_name || b == var_name {
        return None;
    }
    Some(LogEndpointPositivity::Symbols { a, b })
}

/// True when `expr` is pointwise `log(var)` — the only integrand the definite
/// log arm certifies. Composites (`log(x²)`) and linear combinations stay
/// out; a missing certificate is preferable to stretching `IntervalIntegrable`
/// lemmas past their hypotheses.
fn is_pointwise_log(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    pool.with(
        expr,
        |d| matches!(d, ExprData::Func { name, args } if name == "log" && args.len() == 1 && args[0] == var),
    )
}

/// Lean import header for the definite-`∫ log` arm. Adds Mathlib's
/// `intervalIntegrable_log` ([`Mathlib.Analysis.SpecialFunctions.Integrals`])
/// and `hasDerivAt_mul_log` ([`Mathlib.Analysis.SpecialFunctions.Log.NegMulLog`])
/// on top of the interval-FTC imports the shared definite header already has.
fn emit_definite_log_header() -> String {
    "import Mathlib.Tactic\n\
     import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv\n\
     import Mathlib.Analysis.SpecialFunctions.ExpDeriv\n\
     import Mathlib.Analysis.Calculus.Deriv.Pow\n\
     import Mathlib.MeasureTheory.Integral.IntervalIntegral\n\
     import Mathlib.MeasureTheory.Integral.FundThmCalculus\n\
     import Mathlib.Analysis.SpecialFunctions.Integrals\n\
     import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog\n\
     \n\
     open Real\n\n"
        .to_string()
}

/// Emit `∫ x in a..b, log x = (b log b − b) − (a log a − a)` via
/// `intervalIntegral.integral_eq_sub_of_hasDerivAt`, with `HasDerivAt` of
/// `F = x log x − x` from `hasDerivAt_mul_log.sub hasDerivAt_id` and
/// `IntervalIntegrable` from `intervalIntegrable_log` + `Set.not_mem_uIcc_of_lt`.
///
/// Requires [`classify_log_endpoints`] to succeed. Returns `""` otherwise.
fn emit_definite_log_cert(
    integrand: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> String {
    let Some(positivity) = classify_log_endpoints(lower, upper, var, pool) else {
        return String::new();
    };
    let var_name = wrt_name(var, pool);
    let a_lean = expr_to_lean(lower, pool);
    let b_lean = expr_to_lean(upper, pool);
    let f_lean = expr_to_lean(integrand, pool);
    let f_body = format!("({var_name}) * Real.log ({var_name}) - ({var_name})");
    let rhs = format!(
        "(({b_lean}) * Real.log ({b_lean}) - ({b_lean})) - (({a_lean}) * Real.log ({a_lean}) - ({a_lean}))"
    );

    let (example_binders, pos_hyps, hb_ident) = match &positivity {
        LogEndpointPositivity::Literals => (
            String::new(),
            format!(
                "\x20 have ha : (0 : ℝ) < {a_lean} := by norm_num\n\
                 \x20 have hb : (0 : ℝ) < {b_lean} := by norm_num\n"
            ),
            "hb",
        ),
        LogEndpointPositivity::Symbols { a, b } if a == b => {
            (format!("({a} : ℝ) (ha : 0 < {a}) "), String::new(), "ha")
        }
        LogEndpointPositivity::Symbols { a, b } => (
            format!("({a} : ℝ) ({b} : ℝ) (ha : 0 < {a}) (hb : 0 < {b}) "),
            String::new(),
            "hb",
        ),
    };

    let mut out = emit_definite_log_header();
    out.push_str(&format!(
        "-- ∫ {var_name} in {a_lean}..{b_lean}, {f_lean} = F {b_lean} - F {a_lean}   (F = fun {var_name} => {f_body})\n\
         -- certified via the second FTC for interval integrals, under 0 < a and 0 < b:\n\
         --   intervalIntegral.integral_eq_sub_of_hasDerivAt (deriv F = f on uIcc) (IntervalIntegrable f)\n\
         -- IntervalIntegrable log needs 0 ∉ uIcc a b (intervalIntegrable_log + Set.not_mem_uIcc_of_lt).\n"
    ));
    out.push_str(&format!(
        "example {example_binders}: ∫ {var_name} in ({a_lean})..({b_lean}), {f_lean} = {rhs} := by\n\
         {pos_hyps}\
         \x20 have hderiv : ∀ {var_name} ∈ Set.uIcc ({a_lean}) ({b_lean}),\n\
         \x20     HasDerivAt (fun ({var_name} : ℝ) => {f_body}) ({f_lean}) {var_name} := by\n\
         \x20   intro {var_name} hx\n\
         \x20   have hxpos : (0 : ℝ) < {var_name} := lt_of_lt_of_le (lt_min ha {hb_ident}) hx.1\n\
         \x20   simpa using (hasDerivAt_mul_log hxpos.ne').sub (hasDerivAt_id {var_name})\n\
         \x20 have hint : IntervalIntegrable (fun ({var_name} : ℝ) => {f_lean}) MeasureTheory.volume ({a_lean}) ({b_lean}) :=\n\
         \x20   intervalIntegral.intervalIntegrable_log (Set.not_mem_uIcc_of_lt ha {hb_ident})\n\
         \x20 exact intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint\n"
    ));

    if out.contains("sorry") || out.contains("admit") {
        return String::new();
    }
    out
}

/// True when `bound` is (or contains) the canonical `±∞` symbol — an improper
/// endpoint the finite interval-FTC lemma cannot certify.
fn bound_is_infinite(bound: ExprId, pool: &ExprPool) -> bool {
    if bound == pool.pos_infinity() {
        return true;
    }
    pool.with(bound, |d| match d {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&c| bound_is_infinite(c, pool)),
        ExprData::Pow { base, exp } => {
            bound_is_infinite(*base, pool) || bound_is_infinite(*exp, pool)
        }
        ExprData::Func { args, .. } => args.iter().any(|&a| bound_is_infinite(a, pool)),
        _ => false,
    })
}

/// Emit a Lean certificate for a **definite integral**
/// `∫ x in a..b, f x = F b - F a`.
///
/// Unlike an indefinite integral (which can only be certified via the FTC
/// *derivative* relation), a definite integral has a genuine equational
/// statement Mathlib can prove directly:
///
/// ```text
/// intervalIntegral.integral_eq_sub_of_hasDerivAt
///   (hderiv : ∀ x ∈ Set.uIcc a b, HasDerivAt F (f x) x)
///   (hint   : IntervalIntegrable f volume a b) :
///   ∫ x in a..b, f x = F b - F a
/// ```
///
/// The certificate states `∫ x in a..b, f x = F b - F a` with the endpoints
/// substituted textually (so the right-hand side is `sin b - sin a`,
/// `exp b - exp a`, …), and discharges it with a single `exact` of the lemma
/// above — the substituted right-hand side is definitionally equal to
/// `F b - F a` by β-reduction, so no fragile numeric/`ring` closer (and thus no
/// linter noise under `-DwarningAsError=true`) is ever needed.
///
/// Certified integrand shapes (the same base family the indefinite path
/// certifies, now closed under finite sums and constant multiples): `cos`,
/// `sin`, `exp` of the integration variable, integer powers `xⁿ` (`n ≥ 1`,
/// plus the bare variable as `x¹`), `(1+x²)⁻¹` (antiderivative `arctan x`),
/// any numeric-literal constant multiple of one of those (`3 * cos x`,
/// `cos x * 3`, `-sin x`, …), and any finite sum of such terms (`x² + sin x`,
/// `3 * cos x + exp x`, …). Pointwise `log` of the integration variable is a
/// separate arm: it needs `0 < a` and `0 < b` so `IntervalIntegrable log` is
/// discharged by `intervalIntegrable_log` + `Set.not_mem_uIcc_of_lt`, and
/// does **not** join the linear-combination fragment (a sum with one log
/// addend still withholds). Every other integrand — `∫_0^1 log` (singular
/// at 0), negative endpoints, and any improper (`±∞`) endpoint — is
/// **withheld** (returns `""`). Never emits `sorry` / `admit`, and never
/// asserts an unproven statement.
pub fn emit_definite_integration_cert(
    integrand: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> String {
    // Improper endpoints escape the finite interval-FTC lemma: withhold.
    if bound_is_infinite(lower, pool) || bound_is_infinite(upper, pool) {
        return String::new();
    }
    // Pointwise log needs positivity of the interval; handled separately so
    // the unconditional sin/cos/exp/pow fragment never has to thread binders.
    if is_pointwise_log(integrand, var, pool) {
        return emit_definite_log_cert(integrand, var, lower, upper, pool);
    }
    let var_name = pool.with(var, |d| match d {
        ExprData::Symbol { name, .. } => name.clone(),
        _ => "x".to_string(),
    });
    // Any residual mismatch between the assembled derivative-value term and
    // the literal integrand text (double negations from `Sin`'s `.neg`,
    // numeric division from `Pow`'s `.div_const`, …) is bridged by the single
    // top-level `simpa` in the proof below, rather than per-term closers.
    let Some((antideriv_body, hderiv_term, hint_term)) =
        build_definite_pieces(integrand, var, &var_name, pool)
    else {
        return String::new();
    };

    let a_lean = expr_to_lean(lower, pool);
    let b_lean = expr_to_lean(upper, pool);
    let f_lean = expr_to_lean(integrand, pool);

    // `antideriv_body(t)` renders the antiderivative `F` with the binder/endpoint
    // spelled `t`. `F(var_name)` is the lambda body; `F(a_lean)` / `F(b_lean)`
    // are the substituted endpoints (β-equal to `(fun x => F) a` / `… b`).
    let f_body = antideriv_body(&var_name);
    let rhs = format!(
        "({}) - ({})",
        antideriv_body(&b_lean),
        antideriv_body(&a_lean)
    );

    let mut out = emit_definite_integral_header();
    out.push_str(&format!(
        "-- ∫ {var_name} in {a_lean}..{b_lean}, {f_lean} = F {b_lean} - F {a_lean}   (F = fun {var_name} => {f_body})\n\
         -- certified via the second FTC for interval integrals:\n\
         --   intervalIntegral.integral_eq_sub_of_hasDerivAt (deriv F = f on uIcc) (IntervalIntegrable f)\n"
    ));
    out.push_str(&format!(
        "example : ∫ {var_name} in ({a_lean})..({b_lean}), {f_lean} = {rhs} := by\n\
         \x20 have hderiv : ∀ {var_name} ∈ Set.uIcc ({a_lean}) ({b_lean}),\n\
         \x20     HasDerivAt (fun ({var_name} : ℝ) => {f_body}) ({f_lean}) {var_name} := by\n\
         \x20   intro {var_name} _\n\
         \x20   simpa using {hderiv_term}\n\
         \x20 have hint : IntervalIntegrable (fun ({var_name} : ℝ) => {f_lean}) MeasureTheory.volume ({a_lean}) ({b_lean}) :=\n\
         \x20   {hint_term}\n\
         \x20 exact intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint\n"
    ));

    // Defense in depth: never hand out a certificate containing admissions.
    if out.contains("sorry") || out.contains("admit") {
        return String::new();
    }
    out
}

// ---------------------------------------------------------------------------
// Expression → Lean syntax
// ---------------------------------------------------------------------------

/// Convert a symbolic expression to a Lean 4 term.
fn expr_to_lean(expr: ExprId, pool: &ExprPool) -> String {
    expr_to_lean_opts(expr, pool, false)
}

/// As [`expr_to_lean`], but printing `Mul[e, -1]` as `-e`.
///
/// Use this only where the emitted goal is discharged by a bare Mathlib lemma
/// stated about `-e`: `tendsto_exp_neg_atTop_nhds_zero` proves
/// `Tendsto (fun x => rexp (-x)) atTop (nhds 0)`, and Lean rejects it against a
/// goal printed as `fun x => rexp (x * -1)`.
///
/// Do *not* use it for the `diff` or definite-integral emitters. Those pair the
/// goal with witness terms built as strings in the matching `c * f` shape — e.g.
/// `(Real.hasDerivAt_exp x).const_mul ((-1 : \u{211d}))`. Printing the goal as `-e`
/// there leaves Lean unifying `-rexp x` against `-1 * rexp x` by defeq, which
/// exhausts the `whnf` heartbeat budget on `int_def_neg_exp_0_1`.
fn expr_to_lean_neg(expr: ExprId, pool: &ExprPool) -> String {
    expr_to_lean_opts(expr, pool, true)
}

fn expr_to_lean_opts(expr: ExprId, pool: &ExprPool, neg_form: bool) -> String {
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
            let parts: Vec<String> = args
                .iter()
                .map(|&a| expr_to_lean_opts(a, pool, neg_form))
                .collect();
            format!("({})", parts.join(" + "))
        }
        ExprData::Mul(args) => {
            // The kernel folds `-e` into `Mul[e, -1]`. Most emitters here pair
            // the printed goal with hand-built witness terms stated in the same
            // `c * f` shape (`.const_mul ((-1 : \u{211d}))`), so the two agree and
            // printing the product is what keeps them matching. Only `neg_form`
            // callers want the negation; see `expr_to_lean_neg`.
            if neg_form {
                let neg_one =
                    |a: ExprId| pool.with(a, |d| matches!(d, ExprData::Integer(n) if n.0 == -1));
                let rest: Vec<ExprId> = args.iter().copied().filter(|&a| !neg_one(a)).collect();
                if args.iter().filter(|&&a| neg_one(a)).count() == 1 && !rest.is_empty() {
                    let inner = if rest.len() == 1 {
                        expr_to_lean_opts(rest[0], pool, neg_form)
                    } else {
                        let parts: Vec<String> = rest
                            .iter()
                            .map(|&a| expr_to_lean_opts(a, pool, neg_form))
                            .collect();
                        format!("({})", parts.join(" * "))
                    };
                    return format!("(-{inner})");
                }
            }
            let parts: Vec<String> = args
                .iter()
                .map(|&a| expr_to_lean_opts(a, pool, neg_form))
                .collect();
            format!("({})", parts.join(" * "))
        }
        ExprData::Pow { base, exp } => {
            let b = expr_to_lean_opts(*base, pool, neg_form);
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
                    _ => expr_to_lean_opts(*exp, pool, neg_form),
                });
                format!("({b}) ^ {e}")
            }
        }
        ExprData::Func { name, args } => {
            let arg_strs: Vec<String> = args
                .iter()
                .map(|&a| expr_to_lean_opts(a, pool, neg_form))
                .collect();
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
                "sinh" => format!("Real.sinh ({})", arg_strs[0]),
                "cosh" => format!("Real.cosh ({})", arg_strs[0]),
                "tanh" => format!("Real.tanh ({})", arg_strs[0]),
                // Mathlib spells these `arctan` / `arcsin`, not `atan` / `asin`.
                "atan" => format!("Real.arctan ({})", arg_strs[0]),
                "asin" => format!("Real.arcsin ({})", arg_strs[0]),
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
    fn withhold_basel_zeta_even_certificate() {
        // Σ 1/k² = π²/6 is a real closed form, but `basel_zeta_even` is not a
        // Mathlib-backed rewrite step. Emitting `1/k² = π²/6` (or a scaled
        // variant) as an equality example is false and must be withheld.
        use crate::sum::sum_definite;

        let pool = p();
        let k = pool.symbol("k", Domain::Real);
        let one = pool.integer(1_i32);
        let term = pool.pow(k, pool.integer(-2_i32));
        let derived = sum_definite(term, k, one, pool.pos_infinity(), &pool).expect("Basel sum");
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            lean.is_empty(),
            "Basel sum_definite must withhold Lean cert until Mathlib-backed, got: {lean}"
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
    fn definite_integration_cert_cos() {
        // ∫₀¹ cos x = sin 1 - sin 0, via the interval FTC.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cos_x = pool.func("cos", vec![x]);
        let lean = emit_definite_integration_cert(
            cos_x,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₀¹ cos must certify via the interval FTC"
        );
        assert!(
            lean.contains("intervalIntegral.integral_eq_sub_of_hasDerivAt"),
            "must invoke the interval FTC lemma: {lean}"
        );
        assert!(
            lean.contains("HasDerivAt (fun (x : ℝ) => Real.sin (x))"),
            "antiderivative is sin: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_sin"),
            "cos derivative witness: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_sin() {
        // ∫₀¹ sin x = (-cos 1) - (-cos 0), via the interval FTC.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let lean = emit_definite_integration_cert(
            sin_x,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₀¹ sin must certify via the interval FTC"
        );
        assert!(
            lean.contains("intervalIntegral.integral_eq_sub_of_hasDerivAt"),
            "must invoke the interval FTC lemma: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_cos"),
            "sin derivative witness comes from cos: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_power() {
        // ∫₀¹ x² = 1³/3 - 0³/3, via the interval FTC + hasDerivAt_pow.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let lean =
            emit_definite_integration_cert(x2, x, pool.integer(0_i32), pool.integer(1_i32), &pool);
        assert!(!lean.is_empty(), "∫₀¹ x² must certify via the interval FTC");
        assert!(
            lean.contains("hasDerivAt_pow"),
            "power derivative witness: {lean}"
        );
        assert!(
            lean.contains("continuous_pow"),
            "power integrability via continuous_pow: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_exp() {
        // ∫₀¹ exp x = exp 1 - exp 0, via the interval FTC.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let lean = emit_definite_integration_cert(
            exp_x,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₀¹ exp must certify via the interval FTC"
        );
        assert!(
            lean.contains("Real.hasDerivAt_exp"),
            "exp derivative witness: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_log_positive_bounds() {
        // ∫₁² log x = (2 log 2 − 2) − (1 log 1 − 1), via the interval FTC
        // under 0 < 1 and 0 < 2 (discharged by norm_num).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let lean = emit_definite_integration_cert(
            log_x,
            x,
            pool.integer(1_i32),
            pool.integer(2_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₁² log must certify via the interval FTC: {lean}"
        );
        assert!(
            lean.contains("intervalIntegral.integral_eq_sub_of_hasDerivAt"),
            "must invoke the interval FTC lemma: {lean}"
        );
        assert!(
            lean.contains("hasDerivAt_mul_log"),
            "antiderivative x log x − x via hasDerivAt_mul_log: {lean}"
        );
        assert!(
            lean.contains("intervalIntegrable_log"),
            "IntervalIntegrable log via intervalIntegrable_log: {lean}"
        );
        assert!(
            lean.contains("Set.not_mem_uIcc_of_lt"),
            "0 ∉ uIcc via not_mem_uIcc_of_lt: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_log_symbolic_binders() {
        // Symbolic endpoints emit (ha : 0 < a) (hb : 0 < b) rather than
        // claiming IntervalIntegrable on an interval that might contain 0.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let lean = emit_definite_integration_cert(log_x, x, a, b, &pool);
        assert!(
            !lean.is_empty(),
            "∫_a^b log with symbol endpoints must certify under positivity binders: {lean}"
        );
        assert!(
            lean.contains("(ha : 0 < a)") && lean.contains("(hb : 0 < b)"),
            "expected positivity binders: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_log_withheld_at_zero() {
        // ∫₀¹ log is singular at 0: IntervalIntegrable log needs 0 ∉ uIcc.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let lean = emit_definite_integration_cert(
            log_x,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            lean.is_empty(),
            "∫₀¹ log is singular at 0; must withhold: {lean}"
        );
    }

    #[test]
    fn definite_integration_cert_log_withheld_for_negative_endpoint() {
        // Negative endpoints stay withheld even though Mathlib has
        // integral_log_of_neg — Alkahest's log is not Real.log on (−∞, 0).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let lean = emit_definite_integration_cert(
            log_x,
            x,
            pool.integer(-2_i32),
            pool.integer(-1_i32),
            &pool,
        );
        assert!(
            lean.is_empty(),
            "∫ over negative endpoints must withhold: {lean}"
        );
    }

    #[test]
    fn definite_integration_cert_withheld_for_infinite_bound() {
        // Improper integral (∞ endpoint) must never claim the finite interval FTC.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_neg_x = pool.func("exp", vec![x]);
        let lean = emit_definite_integration_cert(
            exp_neg_x,
            x,
            pool.integer(0_i32),
            pool.pos_infinity(),
            &pool,
        );
        assert!(
            lean.is_empty(),
            "∫ with an infinite bound must withhold the interval-FTC cert: {lean}"
        );
    }

    #[test]
    fn definite_integration_cert_sum_of_sin_and_cos() {
        // ∫₀¹ (sin x + cos x) = (-cos 1 + sin 1) - (-cos 0 + sin 0), via
        // HasDerivAt.add / IntervalIntegrable.add composing the two base facts.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let sum = pool.add(vec![sin_x, cos_x]);
        let lean =
            emit_definite_integration_cert(sum, x, pool.integer(0_i32), pool.integer(1_i32), &pool);
        assert!(
            !lean.is_empty(),
            "∫₀¹ (sin x + cos x) must certify via the interval FTC linear combination"
        );
        assert!(
            lean.contains("intervalIntegral.integral_eq_sub_of_hasDerivAt"),
            "must invoke the interval FTC lemma: {lean}"
        );
        assert!(
            lean.contains(".add"),
            "must combine the two terms via HasDerivAt.add / IntervalIntegrable.add: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_constant_multiple_of_cos() {
        // ∫₀¹ 3*cos x = 3*sin 1 - 3*sin 0, via HasDerivAt.const_mul.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cos_x = pool.func("cos", vec![x]);
        let three_cos = pool.mul(vec![pool.integer(3_i32), cos_x]);
        let lean = emit_definite_integration_cert(
            three_cos,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₀¹ 3*cos x must certify via a constant-multiple interval FTC cert"
        );
        assert!(
            lean.contains("const_mul") || lean.contains("mul_const"),
            "must scale the base HasDerivAt fact: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_negative_coefficient_exp() {
        // ∫₀¹ -exp x = -(exp 1) - (-(exp 0)), via HasDerivAt.const_mul (-1).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let neg_exp = pool.mul(vec![pool.integer(-1_i32), exp_x]);
        let lean = emit_definite_integration_cert(
            neg_exp,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₀¹ -exp x must certify via a constant-multiple interval FTC cert"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_rational_coefficient() {
        // ∫₀¹ (1/2)*x² — a Rational literal coefficient must also certify.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let half_x2 = pool.mul(vec![pool.rational(1_i32, 2_i32), x2]);
        let lean = emit_definite_integration_cert(
            half_x2,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₀¹ (1/2)*x² must certify with a Rational coefficient"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_three_term_linear_combination() {
        // ∫₀¹ (x² + sin x + 3*cos x) — a three-term sum mixing a bare power,
        // a bare trig term, and a scaled trig term.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let three_cos = pool.mul(vec![pool.integer(3_i32), cos_x]);
        let combo = pool.add(vec![x2, sin_x, three_cos]);
        let lean = emit_definite_integration_cert(
            combo,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₀¹ (x² + sin x + 3*cos x) must certify via the linear-combination fragment"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_withheld_for_sum_with_unsupported_term() {
        // ∫ (cos x + log x): one term outside the fragment must withhold the
        // WHOLE certificate, not just skip that term.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cos_x = pool.func("cos", vec![x]);
        let log_x = pool.func("log", vec![x]);
        let sum = pool.add(vec![cos_x, log_x]);
        let lean =
            emit_definite_integration_cert(sum, x, pool.integer(1_i32), pool.integer(2_i32), &pool);
        assert!(
            lean.is_empty(),
            "a sum with one non-certifiable term must withhold entirely: {lean}"
        );
    }

    #[test]
    fn definite_integration_cert_withheld_for_variable_coefficient() {
        // ∫ y*cos x dx: `y` is not a numeric literal, so this must NOT be
        // (mis)classified as a constant multiple.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let cos_x = pool.func("cos", vec![x]);
        let y_cos = pool.mul(vec![y, cos_x]);
        let lean = emit_definite_integration_cert(
            y_cos,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            lean.is_empty(),
            "a symbolic (non-literal) coefficient must withhold: {lean}"
        );
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
    fn integration_cert_log_via_ftc_derivative() {
        // ∫ log x dx = x·log x − x. Differentiating the antiderivative intern-
        // equals the integrand, and the log/sqrt combine fragment now closes
        // the product/sum steps, so the FTC certificate emits.
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let derived = integrate(log_x, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, log_x, x, &pool);
        assert!(
            !lean.is_empty(),
            "∫ log x should certify via FTC once x·log x certifies: {lean}"
        );
        assert!(
            lean.contains("deriv (fun"),
            "FTC cert is the derivative relation: {lean}"
        );
        assert!(
            lean.contains("hasDerivAt_log") || lean.contains("differentiableAt_log"),
            "expected the hyp-gated log lemma in the reused diff cert: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn integration_cert_inv_x_via_log() {
        // ∫ x⁻¹ dx = log x, certified as `deriv (fun x => log x) x = x⁻¹`.
        // Intern-equality holds; pointwise log is in the FTC fragment.
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let inv_x = pool.pow(x, pool.integer(-1_i32));
        let derived = integrate(inv_x, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, inv_x, x, &pool);
        assert!(
            !lean.is_empty(),
            "∫ x⁻¹ should certify via d/dx log(x); antiderivative={}, integrand={}",
            pool.display(derived.value),
            pool.display(inv_x)
        );
        assert!(
            lean.contains("Real.deriv_log"),
            "expected Real.deriv_log FTC: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn integration_cert_x_neg_two_via_ftc() {
        // ∫ x⁻² dx = -x⁻¹ intern-equals `d/dx (-x⁻¹) = x⁻²`. The antiderivative
        // is a product; `product_rule` closes via the negative-power combine
        // fragment (`x ≠ 0`, `deriv_inv` / `differentiableAt_inv`), not the
        // unconditional simp set.
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x_neg2 = pool.pow(x, pool.integer(-2_i32));
        let derived = integrate(x_neg2, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, x_neg2, x, &pool);
        assert!(
            !lean.is_empty(),
            "∫ x⁻² should certify via FTC reuse of d/dx (-x⁻¹); antiderivative={}, integrand={}",
            pool.display(derived.value),
            pool.display(x_neg2)
        );
        assert!(
            lean.contains("(hx : x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean}"
        );
        assert!(
            lean.contains("deriv_inv") && lean.contains("differentiableAt_inv.mpr hx"),
            "expected the neg-pow combine tactic: {lean}"
        );
        assert!(
            lean.contains("hasDerivAt_inv"),
            "nested power_rule on x⁻¹ still uses hasDerivAt_inv: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn withhold_diff_of_x_log_x_antiderivative() {
        // Textbook-gate `test_int_x_log_x` diffs F = ∫ x log x
        // (`½ x² log x − x²/4`). The product/sum combine steps close with
        // `field_simp`, but a later `collect_mul_factors` rewrites
        // `x² · (1/2) · x⁻¹ = x · (1/2)` — an n-ary inverse cancellation
        // `ring` cannot close. Emitting that step made Lean CI red
        // (`Try this: ring_nf` under `warningAsError`). Withhold the whole
        // certificate; a missing cert beats a non-typechecking one.
        use crate::diff::diff;
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let integrand = pool.mul(vec![x, pool.func("log", vec![x])]);
        let derived = integrate(integrand, x, &pool).expect("integrate");
        let ftc = emit_integration_cert(derived.value, integrand, x, &pool);
        assert!(
            ftc.is_empty(),
            "∫ x log x must withhold the FTC cert until cleanup closes: {ftc}"
        );
        let d_f = diff(derived.value, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&d_f, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "d/dx of ∫ x log x's F must withhold (n-ary inv-cancel): {lean}"
        );
    }

    #[test]
    fn withhold_nary_inv_cancel_with_spectator_coeff() {
        // The exact cleanup shape that broke Lean CI: three-factor
        // `collect_mul_factors` is beyond the two-factor field_simp encoding.
        use crate::simplify::simplify;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.pow(x, pool.integer(-1_i32)),
            pool.rational(1_i32, 2_i32),
        ]);
        let derived = simplify(expr, &pool);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            lean.is_empty(),
            "x² · x⁻¹ · ½ must withhold until n-ary inv-cancel is encoded: {lean}"
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
        // Colored `exp_of_log` records `SideCondition::Positive(x)` once the
        // caller discharges it; the Lean exporter upgrades that into an
        // explicit `(x : ℝ) (hx : 0 < x)` binder closed by `Real.exp_log hx`.
        use crate::kernel::expr::PredicateKind;
        use crate::simplify::AssumptionContext;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let mut assumptions = AssumptionContext::new();
        assumptions
            .refine(pool.predicate(PredicateKind::Gt, vec![x, zero]), &pool)
            .unwrap();
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        let derived = assumptions.simplify(expr, &pool);
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
    fn abs_of_positive_certifies_with_positivity_hyp() {
        // Colored `abs_of_positive` records `SideCondition::Positive(x)` once
        // the caller discharges it; the exporter upgrades that into an
        // explicit `(x : ℝ) (hx : 0 < x)` binder closed by `abs_of_pos hx`.
        use crate::kernel::expr::PredicateKind;
        use crate::simplify::AssumptionContext;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let mut assumptions = AssumptionContext::new();
        assumptions
            .refine(pool.predicate(PredicateKind::Gt, vec![x, zero]), &pool)
            .unwrap();
        let expr = pool.func("abs", vec![x]);
        let derived = assumptions.simplify(expr, &pool);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            !lean.is_empty(),
            "abs(x) with a recorded positivity condition should certify"
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
            lean.contains("abs_of_pos hx"),
            "expected abs_of_pos to consume the hypothesis: {lean}"
        );
    }

    #[test]
    fn abs_of_positive_withheld_when_positivity_unproven() {
        // A step with no recorded side condition at all must still be
        // withheld — the exporter never invents a hypothesis that wasn't in
        // the derivation log.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("abs", vec![x]);
        let step = RewriteStep::simple("abs_of_positive", expr, x);
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
        // Colored `sum_of_logs` records `Positive(x)`, `Positive(y)` once the
        // caller discharges them; the exporter upgrades the two-factor case
        // into an explicit-binder `Real.log_mul` certificate.
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
        let expr = pool.add(vec![pool.func("log", vec![x]), pool.func("log", vec![y])]);
        let derived = assumptions.simplify(expr, &pool);
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
    fn inv_cancel_x_squared_times_x_neg_squared_certifies_with_nonzero_hyp() {
        // `x² * x⁻² = 1`: `ring` cannot prove this (false at `x = 0` under
        // Lean's `0⁻¹ = 0` junk value), so `collect_mul_factors`'s static
        // "by ring" table entry must not be trusted — the emitter upgrades
        // it to an explicit `(x : ℝ) (hne : x ≠ 0)` binder + `field_simp`.
        use crate::simplify::simplify;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.pow(x, pool.integer(-2_i32)),
        ]);
        let derived = simplify(expr, &pool);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(!lean.is_empty(), "x² * x⁻² = 1 should certify under x ≠ 0");
        assert!(
            !lean.contains("sorry"),
            "certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hne : x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean}"
        );
        assert!(
            lean.contains("field_simp [hne]"),
            "expected field_simp to consume the hypothesis: {lean}"
        );
    }

    #[test]
    fn inv_cancel_mixed_sign_exponents_needs_trailing_ring() {
        // `x⁻² * x⁵ = x³`: like the `net = 0` case above, `field_simp` needs
        // `x ≠ 0`, but here it leaves a genuine ring rearrangement
        // (`x⁵ = x³ * x²`) that a *following* `ring` must close — appending
        // `ring` unconditionally (even when unneeded, as in the `= 1` case)
        // would trip Lean's `unreachableTactic` lint under
        // `-DwarningAsError=true`, so the emitter must only add it here.
        use crate::simplify::simplify;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![
            pool.pow(x, pool.integer(-2_i32)),
            pool.pow(x, pool.integer(5_i32)),
        ]);
        let derived = simplify(expr, &pool);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(!lean.is_empty(), "x⁻² * x⁵ = x³ should certify under x ≠ 0");
        assert!(
            !lean.contains("sorry"),
            "certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hne : x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean}"
        );
        assert!(
            lean.contains("field_simp [hne]") && lean.contains("ring"),
            "expected field_simp followed by a closing ring: {lean}"
        );
    }

    #[test]
    fn double_inverse_certifies_unconditionally() {
        // `(x⁻¹)⁻¹ = x`: true for *every* real `x` including `0`
        // (`(0⁻¹)⁻¹ = 0⁻¹ = 0`), via Mathlib's `inv_inv`. `ring` can't close
        // it (it treats `⁻¹` as an opaque atom), but no hypothesis binder is
        // needed either — unlike the reciprocal-cancellation shapes above.
        use crate::simplify::simplify;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(pool.pow(x, pool.integer(-1_i32)), pool.integer(-1_i32));
        let derived = simplify(expr, &pool);
        let lean = emit_lean_expr(&derived, &pool);
        assert!(
            !lean.is_empty(),
            "(x⁻¹)⁻¹ = x should certify unconditionally"
        );
        assert!(
            !lean.contains("sorry"),
            "certificate must not use sorry: {lean}"
        );
        assert!(
            !lean.contains("(hne"),
            "double inverse needs no hypothesis binder: {lean}"
        );
        assert!(
            lean.contains("simp [inv_inv]"),
            "expected the inv_inv simp lemma: {lean}"
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
        assert!(
            !lean.contains("(hx : 0 < x)"),
            "sin·exp is everywhere differentiable; no extra binder: {lean}"
        );
    }

    #[test]
    fn multi_term_poly_combine_certifies_via_discharge_depth() {
        use crate::diff::diff;

        // A multi-term polynomial derivative combine (`diff_univariate_poly`)
        // whose nested `DifferentiableAt` discharge exceeds simp's default
        // depth of 2 — the exact "green by luck" shape. It must certify (not be
        // withheld) and use the raised `maxDischargeDepth` tactic.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![
            pool.mul(vec![pool.integer(3_i32), pool.pow(x, pool.integer(3_i32))]),
            pool.mul(vec![pool.integer(2_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.mul(vec![x, pool.integer(5_i32)]),
            pool.integer(7_i32),
        ]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "multi-term polynomial derivative must be certifiable: {lean}"
        );
        assert!(
            lean.contains("maxDischargeDepth"),
            "combine steps must use the raised discharge-depth tactic: {lean}"
        );
        assert!(!lean.contains("sorry"), "must not admit: {lean}");
    }

    #[test]
    fn emit_lean_product_rule_x_log() {
        use crate::diff::diff;

        // `d/dx (x · log x)` needs `x ≠ 0`; the combine path threads
        // `(hx : 0 < x)` rather than withholding.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.func("log", vec![x])]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (x·log x) should be Lean-certifiable: {lean}"
        );
        assert!(
            lean.contains("(hx : 0 < x)"),
            "expected an explicit positivity binder: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_log hx.ne'")
                || lean.contains("Real.differentiableAt_log hx.ne'"),
            "expected the hyp-gated log lemma: {lean}"
        );
        assert!(
            lean.contains("deriv_mul"),
            "expected product_rule deriv_mul tactic: {lean}"
        );
        assert!(!lean.contains("sorry"), "must not admit: {lean}");
    }

    #[test]
    fn emit_lean_product_rule_exp_log() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.func("exp", vec![x]), pool.func("log", vec![x])]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (exp·log) should be Lean-certifiable: {lean}"
        );
        assert!(
            lean.contains("(hx : 0 < x)"),
            "expected an explicit positivity binder: {lean}"
        );
        assert!(!lean.contains("sorry"), "must not admit: {lean}");
    }

    #[test]
    fn emit_lean_sum_rule_log_x() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![pool.func("log", vec![x]), x]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (log x + x) should be Lean-certifiable: {lean}"
        );
        assert!(
            lean.contains("(hx : 0 < x)"),
            "expected an explicit positivity binder: {lean}"
        );
        assert!(
            lean.contains("deriv_add"),
            "sum_rule needs deriv_add: {lean}"
        );
        assert!(!lean.contains("sorry"), "must not admit: {lean}");
    }

    #[test]
    fn emit_lean_product_rule_x_sqrt() {
        use crate::diff::diff;

        // Same `0 < x` binder covers sqrt (via `hasDerivAt_sqrt hx.ne'`).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.func("sqrt", vec![x])]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (x·sqrt x) should be Lean-certifiable: {lean}"
        );
        assert!(
            lean.contains("(hx : 0 < x)"),
            "expected an explicit positivity binder: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_sqrt hx.ne'"),
            "expected the hyp-gated sqrt lemma: {lean}"
        );
        assert!(!lean.contains("sorry"), "must not admit: {lean}");
    }

    #[test]
    fn withhold_chain_rule_log_of_nested_composite() {
        use crate::diff::diff;

        // d/dx log(sqrt(x²−1) + x) is a chain composite; another agent owns
        // general HasDerivAt.comp, so this stays withheld.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let inner = pool.add(vec![
            pool.func("sqrt", vec![pool.add(vec![x2, pool.integer(-1_i32)])]),
            x,
        ]);
        let expr = pool.func("log", vec![inner]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "chain-rule d/dx log(sqrt(x²−1)+x) must be withheld: {lean}"
        );
    }

    #[test]
    fn negative_power_of_var_diff_certifies_with_nonzero_hyp() {
        use crate::diff::diff;

        // `d/dx (x⁻²)` (stored as a negative power of the variable) needs
        // `x ≠ 0`; `deriv_pow` cannot discharge that, so the emitter upgrades
        // to an explicit binder + `hasDerivAt_inv` (then `HasDerivAt.pow`
        // for the remaining natural power of the inverse).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);

        let expr_inv = pool.pow(x, pool.integer(-1_i32));
        let derived_inv = diff(expr_inv, x, &pool).expect("diff");
        let lean_inv = emit_lean_expr_wrt(&derived_inv, &pool, Some(x));
        assert!(
            !lean_inv.is_empty(),
            "d/dx (x⁻¹) should be Lean-certifiable: {lean_inv}"
        );
        assert!(
            !lean_inv.contains("sorry"),
            "x⁻¹ certificate must not use sorry: {lean_inv}"
        );
        assert!(
            lean_inv.contains("(hx : x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean_inv}"
        );
        assert!(
            lean_inv.contains("hasDerivAt_inv"),
            "expected hasDerivAt_inv: {lean_inv}"
        );

        let expr_neg2 = pool.pow(x, pool.integer(-2_i32));
        let derived_neg2 = diff(expr_neg2, x, &pool).expect("diff");
        let lean_neg2 = emit_lean_expr_wrt(&derived_neg2, &pool, Some(x));
        assert!(
            !lean_neg2.is_empty(),
            "d/dx (x⁻²) should be Lean-certifiable: {lean_neg2}"
        );
        assert!(
            !lean_neg2.contains("sorry"),
            "x⁻² certificate must not use sorry: {lean_neg2}"
        );
        assert!(
            lean_neg2.contains("(hx : x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean_neg2}"
        );
        assert!(
            lean_neg2.contains("hasDerivAt_inv") && lean_neg2.contains("hinv.pow 2"),
            "expected hasDerivAt_inv then HasDerivAt.pow 2: {lean_neg2}"
        );
    }

    #[test]
    fn product_rule_neg_inv_combine_certifies_with_nonzero_hyp() {
        use crate::diff::diff;

        // `d/dx (-x⁻¹)` is a product whose `product_rule` step needs `x ≠ 0`.
        // The second combine fragment closes it; this is the FTC reuse path
        // for `∫ x⁻²`.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![
            pool.integer(-1_i32),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (-x⁻¹) should be Lean-certifiable: {lean}"
        );
        assert!(
            !lean.contains("sorry"),
            "(-x⁻¹) certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hx : x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean}"
        );
        assert!(
            lean.contains("deriv_inv") && lean.contains("differentiableAt_inv.mpr hx"),
            "expected the neg-pow product_rule tactic: {lean}"
        );
    }

    #[test]
    fn sum_rule_var_plus_inv_combine_certifies_with_nonzero_hyp() {
        use crate::diff::diff;

        // `d/dx (x + x⁻¹)` is a sum in the `{wrt, constants, wrt⁻ⁿ}` fragment.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.pow(x, pool.integer(-1_i32))]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (x + x⁻¹) should be Lean-certifiable: {lean}"
        );
        assert!(
            !lean.contains("sorry"),
            "(x + x⁻¹) certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hx : x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean}"
        );
        assert!(
            lean.contains("deriv_inv") && lean.contains("differentiableAt_inv.mpr hx"),
            "expected the neg-pow sum_rule tactic: {lean}"
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
    fn generalized_power_rule_on_sin_squared_certifies() {
        // `d/dx sin(x)² = 2 sin x cos x` via `HasDerivAt.pow` composed with
        // the pointwise `Real.hasDerivAt_sin` — unconditional, no side
        // condition needed (unlike the `1 / sin x` shape below).
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let expr = pool.pow(sin_x, pool.integer(2_i32));
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx sin(x)² should now be Lean-certifiable via HasDerivAt.pow"
        );
        assert!(
            !lean.contains("sorry"),
            "sin(x)² certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("hf.pow 2"),
            "expected HasDerivAt.pow composition: {lean}"
        );
    }

    #[test]
    fn inv_of_primitive_on_one_over_sin_certifies_with_nonzero_hyp() {
        // `d/dx (1 / sin x) = -cos x / sin²x` via `HasDerivAt.inv`, needs
        // `sin x ≠ 0` — unlike the `sin(x)²` shape above.
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let expr = pool.pow(sin_x, pool.integer(-1_i32));
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (1/sin x) should be Lean-certifiable via HasDerivAt.inv"
        );
        assert!(
            !lean.contains("sorry"),
            "1/sin(x) certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hne : Real.sin x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean}"
        );
        assert!(
            lean.contains("hf.inv hne"),
            "expected HasDerivAt.inv composition: {lean}"
        );
    }

    #[test]
    fn quotient_of_primitives_sin_over_cos_certifies() {
        // `d/dx (sin x / cos x)`, represented as `sin(x) * cos(x)⁻¹`, closes
        // via `HasDerivAt.mul` + `HasDerivAt.inv` given `cos x ≠ 0`. This
        // also exercises the `collect_mul_factors: cos x * (cos x)⁻¹ = 1`
        // cleanup step buried in the log, which needs the nonzero-hypothesis
        // upgrade (`inv_cancel_certificate`) rather than bare `ring`.
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let expr = pool.mul(vec![sin_x, pool.pow(cos_x, pool.integer(-1_i32))]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (sin x / cos x) should be Lean-certifiable via HasDerivAt.mul/.inv"
        );
        assert!(
            !lean.contains("sorry"),
            "sin/cos certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("hf.mul hg"),
            "expected the quotient-chain HasDerivAt composition: {lean}"
        );
        assert!(
            lean.contains("field_simp [hne]"),
            "expected the cos x * (cos x)⁻¹ cleanup to use field_simp, not bare ring: {lean}"
        );
    }

    #[test]
    fn withhold_power_of_primitive_with_unsupported_exponent() {
        // `n ≤ -2` (e.g. `sin(x)^-2`) isn't encoded by `power_chain_certificate`
        // (only `n ≥ 2` via `HasDerivAt.pow` and `n == -1` via `HasDerivAt.inv`
        // are); must withhold rather than emit a broken cert.
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let expr = pool.pow(sin_x, pool.integer(-2_i32));
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "d/dx sin(x)^-2 is not encoded; must withhold: {lean}"
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
    fn diff_log_certifies_unconditionally() {
        // `Real.deriv_log` holds for every real `x` (including `0` and
        // negatives, via Mathlib's `log |x|` extension and junk value at
        // `0`), so `d/dx log(x)` needs no positivity binder — unlike
        // `diff_sqrt` below.
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let derived = diff(log_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx log(x) should be Lean-certifiable");
        assert!(
            !lean.contains("sorry"),
            "log certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("Real.deriv_log"),
            "expected Real.deriv_log tactic: {lean}"
        );
        // No `example (x : ℝ) (hx : ...)` binder for `diff_log` — see `diff_sqrt`.
        assert!(
            lean.contains("example : deriv (fun (x : ℝ) => Real.log"),
            "diff_log needs no explicit hypothesis binder: {lean}"
        );
    }

    #[test]
    fn diff_sqrt_certifies_with_positivity_hyp() {
        // `Real.hasDerivAt_sqrt` needs `x ≠ 0`; the emitter upgrades this to
        // an explicit `(x : ℝ) (hx : 0 < x)` binder, mirroring #236's
        // positivity-binder mechanism but on a DIFFERENTIATION certificate.
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let derived = diff(sqrt_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx sqrt(x) should be Lean-certifiable");
        assert!(
            !lean.contains("sorry"),
            "sqrt certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hx : 0 < x)"),
            "expected an explicit positivity binder: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_sqrt hx.ne'"),
            "expected Real.hasDerivAt_sqrt to consume the hypothesis: {lean}"
        );
    }

    #[test]
    fn withhold_chain_rule_diff_sqrt_composite() {
        // d/dx sqrt(x²) still routes through diff_sqrt on a composite
        // argument; that shape stays withheld (same reasoning as diff_log).
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let sqrt_x2 = pool.func("sqrt", vec![x2]);
        let derived = diff(sqrt_x2, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "chain-rule d/dx sqrt(x²) is not encoded; must withhold: {lean}"
        );
    }

    #[test]
    fn diff_tan_certifies_via_primitive_registry() {
        // `tan` is dispatched through `diff_primitive_registry` (not a
        // dedicated `diff_*` rule name); the emitter maps it to
        // `Real.hasDerivAt_tan` + `Real.inv_one_add_tan_sq` to reconcile
        // Alkahest's `1 + tan²x` form with Mathlib's `1/cos²x` form.
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let tan_x = pool.func("tan", vec![x]);
        let derived = diff(tan_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx tan(x) should be Lean-certifiable");
        assert!(
            !lean.contains("sorry"),
            "tan certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hne : Real.cos x ≠ 0)"),
            "expected an explicit nonzero-hypothesis binder: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_tan hne") && lean.contains("Real.inv_one_add_tan_sq"),
            "expected the tan deriv + Pythagorean-identity reconciliation: {lean}"
        );
    }

    #[test]
    fn withhold_chain_rule_diff_tan_composite() {
        // d/dx tan(x²) routes through diff_primitive_registry on a composite
        // argument; must withhold (no chain-rule encoding for the registry
        // dispatch).
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let tan_x2 = pool.func("tan", vec![x2]);
        let derived = diff(tan_x2, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "chain-rule d/dx tan(x²) is not encoded; must withhold: {lean}"
        );
    }

    #[test]
    fn diff_sinh_certifies_unconditionally() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sinh_x = pool.func("sinh", vec![x]);
        let derived = diff(sinh_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx sinh(x) should be Lean-certifiable");
        assert!(
            !lean.contains("sorry"),
            "sinh certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("Real.deriv_sinh"),
            "expected Real.deriv_sinh tactic: {lean}"
        );
        assert!(
            lean.contains("example : deriv (fun (x : ℝ) => Real.sinh"),
            "diff_sinh needs no explicit hypothesis binder: {lean}"
        );
    }

    #[test]
    fn diff_cosh_certifies_unconditionally() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cosh_x = pool.func("cosh", vec![x]);
        let derived = diff(cosh_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx cosh(x) should be Lean-certifiable");
        assert!(
            !lean.contains("sorry"),
            "cosh certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("Real.deriv_cosh"),
            "expected Real.deriv_cosh tactic: {lean}"
        );
    }

    #[test]
    fn emit_lean_sum_rule_sinh_cosh() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![pool.func("sinh", vec![x]), pool.func("cosh", vec![x])]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (sinh+cosh) should be Lean-certifiable"
        );
        assert!(
            !lean.contains("sorry"),
            "sum certificate must not use sorry: {lean}"
        );
    }

    #[test]
    fn emit_lean_product_rule_exp_sinh() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.func("exp", vec![x]), pool.func("sinh", vec![x])]);
        let derived = diff(expr, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            !lean.is_empty(),
            "d/dx (exp·sinh) should be Lean-certifiable"
        );
        assert!(
            !lean.contains("sorry"),
            "product certificate must not use sorry: {lean}"
        );
    }

    #[test]
    fn diff_atan_certifies_unconditionally() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let atan_x = pool.func("atan", vec![x]);
        let derived = diff(atan_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx atan(x) should be Lean-certifiable");
        assert!(
            !lean.contains("sorry"),
            "atan certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_arctan'"),
            "expected hasDerivAt_arctan' (the ⁻¹ form): {lean}"
        );
        assert!(
            lean.contains("Real.arctan"),
            "Mathlib name is arctan, not atan: {lean}"
        );
        assert!(
            !lean.contains("example (x : ℝ) (h"),
            "diff_atan needs no explicit hypothesis binder: {lean}"
        );
    }

    #[test]
    fn diff_asin_certifies_with_open_interval_hyp() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let asin_x = pool.func("asin", vec![x]);
        let derived = diff(asin_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(!lean.is_empty(), "d/dx asin(x) should be Lean-certifiable");
        assert!(
            !lean.contains("sorry"),
            "asin certificate must not use sorry: {lean}"
        );
        assert!(
            lean.contains("(hx : -1 < x ∧ x < 1)"),
            "expected an explicit |x| < 1 binder: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_arcsin"),
            "expected Real.hasDerivAt_arcsin: {lean}"
        );
        assert!(
            lean.contains("Real.arcsin"),
            "Mathlib name is arcsin, not asin: {lean}"
        );
    }

    #[test]
    fn withhold_chain_rule_diff_asin_composite() {
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let asin_x2 = pool.func("asin", vec![x2]);
        let derived = diff(asin_x2, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "chain-rule d/dx asin(x²) is not encoded; must withhold: {lean}"
        );
    }

    #[test]
    fn withhold_diff_tanh() {
        // Mathlib v4.9.0 has no hasDerivAt_tanh / 1-tanh² identity.
        use crate::diff::diff;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let tanh_x = pool.func("tanh", vec![x]);
        let derived = diff(tanh_x, x, &pool).expect("diff");
        let lean = emit_lean_expr_wrt(&derived, &pool, Some(x));
        assert!(
            lean.is_empty(),
            "d/dx tanh(x) is withheld (no 4.9 identity lemma): {lean}"
        );
    }

    #[test]
    fn integration_cert_inv_one_plus_x_squared_via_atan() {
        use crate::integrate::integrate;

        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.integer(1_i32), pool.pow(x, pool.integer(2_i32))]);
        let inv = pool.pow(den, pool.integer(-1_i32));
        // Python `1/(1+x**2)` interns as `1 * (1+x²)⁻¹`; d/dx atan(x) is the
        // same intern, so the FTC intern-equality gate can fire.
        let integrand = pool.mul(vec![pool.integer(1_i32), inv]);
        let derived = integrate(integrand, x, &pool).expect("integrate");
        let lean = emit_integration_cert(derived.value, integrand, x, &pool);
        assert!(
            !lean.is_empty(),
            "∫ (1+x²)⁻¹ should certify via d/dx atan(x); antiderivative={}, integrand={}",
            pool.display(derived.value),
            pool.display(integrand)
        );
        assert!(
            lean.contains("Real.arctan") || lean.contains("hasDerivAt_arctan"),
            "expected atan FTC via arctan derivative: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
    }

    #[test]
    fn definite_integration_cert_inv_one_plus_x_squared() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.integer(1_i32), pool.pow(x, pool.integer(2_i32))]);
        let integrand = pool.pow(den, pool.integer(-1_i32));
        let lean = emit_definite_integration_cert(
            integrand,
            x,
            pool.integer(0_i32),
            pool.integer(1_i32),
            &pool,
        );
        assert!(
            !lean.is_empty(),
            "∫₀¹ (1+x²)⁻¹ must certify via the interval FTC"
        );
        assert!(
            lean.contains("intervalIntegral.integral_eq_sub_of_hasDerivAt"),
            "must invoke the interval FTC lemma: {lean}"
        );
        assert!(
            lean.contains("Real.hasDerivAt_arctan'"),
            "arctan derivative witness: {lean}"
        );
        assert!(
            lean.contains("Real.arctan"),
            "antiderivative is arctan: {lean}"
        );
        assert!(!lean.contains("sorry") && !lean.contains("admit"));
        // Do not claim = π/4; the certificate is F(b)-F(a) = arctan 1 - arctan 0.
        assert!(
            !lean.contains("π / 4") && !lean.contains("Real.pi / 4"),
            "must not claim the numeric π/4 evaluation: {lean}"
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
        // Naming the theorem is not enough: the goal we print has to *be* the
        // theorem's statement. `tendsto_exp_neg_atTop_nhds_zero` is about
        // `fun x => rexp (-x)`, and Lean rejects it against `rexp (x * -1)`,
        // which is what the kernel's `Mul[x, -1]` spelling used to render as.
        assert!(
            !lean.contains("* (-1 : \u{211d})"),
            "goal must print the negation, not the folded `* -1`: {lean}"
        );
        assert!(
            lean.contains("Real.exp ((-(x : \u{211d})))"),
            "goal must state `exp (-x)` to match the cited lemma: {lean}"
        );
    }

    #[test]
    fn negation_prints_as_negation_not_times_minus_one() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        assert_eq!(expr_to_lean_neg(neg_x, &pool), "(-(x : \u{211d}))");
        // The default printer keeps the product form, because the diff and
        // definite-integral emitters pair it with `.const_mul ((-1 : \u{211d}))`
        // witness terms stated the same way.
        assert_eq!(
            expr_to_lean(neg_x, &pool),
            "((x : \u{211d}) * (-1 : \u{211d}))"
        );

        // A `-1` among several factors negates the remaining product.
        let y = pool.symbol("y", Domain::Real);
        let neg_xy = pool.mul(vec![pool.integer(-1_i32), x, y]);
        assert_eq!(
            expr_to_lean_neg(neg_xy, &pool),
            "(-((x : \u{211d}) * (y : \u{211d})))"
        );

        // A coefficient that merely happens to be negative is not a negation
        // of the rest and must keep printing as a product.
        let neg_two_x = pool.mul(vec![pool.integer(-2_i32), x]);
        assert_eq!(
            expr_to_lean_neg(neg_two_x, &pool),
            "((x : \u{211d}) * (-2 : \u{211d}))"
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
