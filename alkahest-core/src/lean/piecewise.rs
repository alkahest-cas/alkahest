//! Branch-wise Lean 4 statements for the derivative of a `Piecewise`.
//!
//! `diff` distributes over the branches of a `Piecewise` and logs the step
//! `diff_piecewise: Piecewise(cᵢ, vᵢ; d) ↦ Piecewise(cᵢ, vᵢ'; d')`. As a
//! single equation `deriv (fun x => if … ) x = if … ` that step is *false*:
//! at a breakpoint the derivative of the piecewise function is not the
//! derivative of either branch (`d/dx |x|` at `0`). So it is never emitted as
//! one; it is emitted once per branch, on the open region where that branch is
//! selected:
//!
//! ```text
//! example (x : ℝ) (hc1 : 0 < x) :
//!     deriv (fun (x : ℝ) => if x > 0 then x ^ 2 else x * -1) x = x * 2
//! ```
//!
//! and for the default under the strict negation of every condition. The proof
//! shows that the piecewise function agrees with the branch on a neighbourhood
//! of `x` (`Filter.EventuallyEq.deriv_eq`, the neighbourhood coming from
//! `lt_mem_nhds` / `gt_mem_nhds`) and then reuses the branch's own derivative
//! certificate — so a branch is certified only when its derivative step is
//! certifiable on its own, and otherwise the whole certificate is withheld.
//!
//! Conditions are limited to `x ⋈ c` / `c ⋈ x` with `⋈ ∈ {<, ≤, >, ≥}` and `c`
//! free of `x`; anything else (`=`, `≠`, `∧`, a condition on a parameter only)
//! withholds.

use super::{
    depends_on, diff_goal_body, diff_step_certificate, expr_to_lean, is_differentiation_rule,
    join_binders, real_binders, wrt_name,
};
use crate::deriv::log::RewriteStep;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};

/// One branch condition `x ⋈ c`, normalised to the variable on the left.
struct Cond {
    /// The condition as written in the `if`, e.g. `(x : ℝ) > (0 : ℝ)`.
    text: String,
    /// `c` printed.
    other: String,
    /// `x < c` / `x ≤ c` (`true`) or `x > c` / `x ≥ c` (`false`).
    upper: bool,
    /// `<` / `>` (`true`) or `≤` / `≥` (`false`).
    strict: bool,
}

impl Cond {
    /// The open region where the condition holds: hypothesis text and the
    /// lemma turning it into an eventually-true fact at `x`.
    fn holds(&self, x: &str) -> (String, &'static str) {
        if self.upper {
            (format!("{x} < {}", self.other), "gt_mem_nhds")
        } else {
            (format!("{} < {x}", self.other), "lt_mem_nhds")
        }
    }

    /// The open region where the condition fails (the strict negation).
    fn fails(&self, x: &str) -> (String, &'static str) {
        if self.upper {
            (format!("{} < {x}", self.other), "lt_mem_nhds")
        } else {
            (format!("{x} < {}", self.other), "gt_mem_nhds")
        }
    }

    /// `rw` argument selecting (or skipping) this `if`, given the
    /// neighbourhood facts in context.
    fn rewrite(&self, taken: bool) -> String {
        if taken {
            format!("if_pos (show {} by linarith)", self.text)
        } else {
            let lemma = if self.strict { "not_lt" } else { "not_le" };
            format!(
                "if_neg (show ¬({}) from {lemma}.mpr (by linarith))",
                self.text
            )
        }
    }
}

fn parse_cond(cond: ExprId, var: ExprId, pool: &ExprPool) -> Option<Cond> {
    let (kind, a, b) = pool.with(cond, |d| match d {
        ExprData::Predicate { kind, args } if args.len() == 2 => {
            Some((kind.clone(), args[0], args[1]))
        }
        _ => None,
    })?;
    let (op, strict, var_lt_other_when_left) = match kind {
        PredicateKind::Lt => ("<", true, true),
        PredicateKind::Le => ("≤", false, true),
        PredicateKind::Gt => (">", true, false),
        PredicateKind::Ge => ("≥", false, false),
        _ => return None,
    };
    let (other, upper) = if a == var && !depends_on(b, var, pool) {
        (b, var_lt_other_when_left)
    } else if b == var && !depends_on(a, var, pool) {
        (a, !var_lt_other_when_left)
    } else {
        return None;
    };
    Some(Cond {
        text: format!("{} {op} {}", expr_to_lean(a, pool), expr_to_lean(b, pool)),
        other: expr_to_lean(other, pool),
        upper,
        strict,
    })
}

type Branches = (Vec<(ExprId, ExprId)>, ExprId);

fn piecewise_parts(e: ExprId, pool: &ExprPool) -> Option<Branches> {
    pool.with(e, |d| match d {
        ExprData::Piecewise { branches, default } => Some((branches.clone(), *default)),
        _ => None,
    })
}

/// `if c₁ then v₁ else if c₂ then v₂ else d`.
fn if_chain(conds: &[Cond], values: &[ExprId], default: ExprId, pool: &ExprPool) -> String {
    let mut out = String::new();
    for (c, &v) in conds.iter().zip(values) {
        out.push_str(&format!(
            "if {} then {} else ",
            c.text,
            expr_to_lean(v, pool)
        ));
    }
    out.push_str(&expr_to_lean(default, pool));
    out
}

/// The `example`s certifying a `diff_piecewise` step, one per branch plus the
/// default, or `None` when a condition is outside the supported shape or a
/// branch's own derivative step (found in `steps`) is not certifiable.
pub(super) fn piecewise_diff_examples(
    step: &RewriteStep,
    var: ExprId,
    steps: &[RewriteStep],
    pool: &ExprPool,
) -> Option<String> {
    let (branches, default) = piecewise_parts(step.before, pool)?;
    let (d_branches, d_default) = piecewise_parts(step.after, pool)?;
    if branches.is_empty() || branches.len() != d_branches.len() {
        return None;
    }
    let mut conds = Vec::with_capacity(branches.len());
    for ((c, _), (dc, _)) in branches.iter().zip(&d_branches) {
        if c != dc {
            return None;
        }
        conds.push(parse_cond(*c, var, pool)?);
    }
    let x = wrt_name(var, pool);
    // Names the proof introduces; a free variable spelled the same would be
    // captured.
    let reserved = |n: &str| {
        n == "hbr"
            || n == "hev"
            || ["hc", "hx"].iter().any(|p| {
                n.strip_prefix(p)
                    .is_some_and(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()))
            })
    };
    for v in crate::poly::collect_free_vars(step.before, pool) {
        let name = wrt_name(v, pool);
        if reserved(&name) {
            return None;
        }
    }

    let values: Vec<ExprId> = branches.iter().map(|&(_, v)| v).collect();
    let fun_text = if_chain(&conds, &values, default, pool);
    let mut targets: Vec<(usize, ExprId, ExprId)> = branches
        .iter()
        .zip(&d_branches)
        .enumerate()
        .map(|(i, (&(_, v), &(_, dv)))| (i, v, dv))
        .collect();
    targets.push((branches.len(), default, d_default));

    let mut out = Vec::new();
    for (i, value, derivative) in targets {
        // The branch's own derivative step: `deriv (fun x => vᵢ) x = vᵢ'`.
        let root = steps.iter().find(|s| {
            s.rule_name != "diff_piecewise"
                && is_differentiation_rule(s.rule_name)
                && s.before == value
                && s.after == derivative
        })?;
        let (own_binders, tactic) = diff_step_certificate(root, var, pool)?;
        // Region: conditions before `i` fail, condition `i` holds (none for
        // the default).
        let mut hyps = Vec::new();
        let mut facts = Vec::new();
        let mut rewrites = Vec::new();
        for (j, c) in conds.iter().enumerate().take(i + 1) {
            let taken = j == i;
            let (h, lemma) = if taken { c.holds(&x) } else { c.fails(&x) };
            hyps.push(format!("(hc{} : {h})", j + 1));
            facts.push(format!("{lemma} hc{}", j + 1));
            rewrites.push(c.rewrite(taken));
        }
        let mut binder_list = hyps.join(" ");
        if let Some(b) = own_binders.as_deref() {
            binder_list = format!("{b} {binder_list}");
        }
        // Only the branch certificate's own binders can bind a variable; the
        // region hypotheses merely mention `(a : ℝ)`.
        let free = real_binders(
            &[step.before, derivative, var],
            &[],
            own_binders.as_deref().unwrap_or(""),
            pool,
        )?;
        let branch_fun = if depends_on(value, var, pool) {
            format!("fun ({x} : ℝ) => {}", expr_to_lean(value, pool))
        } else {
            format!("fun (_{x} : ℝ) => {}", expr_to_lean(value, pool))
        };
        let which = if i < conds.len() {
            format!("branch {} ({})", i + 1, conds[i].text)
        } else {
            "default branch".to_string()
        };
        let hx: Vec<String> = (1..=facts.len()).map(|j| format!("hx{j}")).collect();
        out.push(format!(
            "-- diff_piecewise, {which}: on the open region where it is selected\n\
             example{}: deriv (fun ({x} : ℝ) => {fun_text}) {x} = {} := by\n  \
             have hbr : {} := {tactic}\n  \
             have hev : (fun ({x} : ℝ) => {fun_text}) =ᶠ[nhds {x}] ({branch_fun}) := by\n    \
             filter_upwards [{}] with {x} {}\n    \
             rw [{}]\n  \
             rw [hev.deriv_eq]\n  \
             exact hbr",
            join_binders(&free, Some(&binder_list)),
            expr_to_lean(derivative, pool),
            diff_goal_body(value, derivative, var, pool),
            facts.join(", "),
            hx.join(" "),
            rewrites.join(", "),
        ));
    }
    Some(out.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use crate::kernel::expr::PredicateKind;
    use crate::kernel::{Domain, ExprPool};
    use crate::lean::emit_lean_expr_wrt;

    #[test]
    fn piecewise_derivative_is_stated_per_branch_on_open_regions() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let cond = pool.predicate(PredicateKind::Gt, vec![x, zero]);
        let neg_x = pool.mul(vec![x, pool.integer(-1_i32)]);
        let pw = pool.piecewise(vec![(cond, pool.pow(x, pool.integer(2_i32)))], neg_x);
        let d = crate::diff::diff(pw, x, &pool).unwrap();
        let src = emit_lean_expr_wrt(&d, &pool, Some(x));
        assert!(!src.is_empty());
        assert!(src.contains("(hc1 : (0 : ℝ) < x)"), "{src}");
        assert!(src.contains("(hc1 : x < (0 : ℝ))"), "{src}");
        assert!(src.contains("=ᶠ[nhds x]"), "{src}");
        assert!(src.contains("rw [hev.deriv_eq]"), "{src}");
        // Never the whole-function equation `deriv (if …) = if …`.
        assert!(!src.contains("= if "), "{src}");
    }

    #[test]
    fn piecewise_derivative_binds_parameters_of_the_conditions() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let cond = pool.predicate(PredicateKind::Lt, vec![x, a]);
        let pw = pool.piecewise(vec![(cond, pool.func("sin", vec![x]))], pool.integer(3_i32));
        let d = crate::diff::diff(pw, x, &pool).unwrap();
        let src = emit_lean_expr_wrt(&d, &pool, Some(x));
        assert!(
            src.contains("example (a : ℝ) (x : ℝ) (hc1 : x < (a : ℝ))"),
            "{src}"
        );
    }

    #[test]
    fn piecewise_derivative_with_a_parameter_condition_is_withheld() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let cond = pool.predicate(PredicateKind::Gt, vec![a, pool.integer(0_i32)]);
        let pw = pool.piecewise(vec![(cond, pool.pow(x, pool.integer(2_i32)))], x);
        let d = crate::diff::diff(pw, x, &pool).unwrap();
        assert_eq!(emit_lean_expr_wrt(&d, &pool, Some(x)), "");
    }
}
