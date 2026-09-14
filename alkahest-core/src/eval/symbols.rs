//! Which symbols carry a value of their own, and which are free parameters a
//! numeric gate is allowed to bind to a sample.
//!
//! Every sampling gate in the crate — [`crate::matrix::spectrum`],
//! [`crate::ode::dsolve::verify`], [`crate::ode::dsolve::system`] — has to
//! answer the same question before it can build an environment: *which of the
//! symbols in this expression am I permitted to invent a value for?*
//!
//! Two of them must never be invented, because they already denote a number:
//!
//! * **`pi`**, which is an ordinary [`ExprData::Symbol`] in this crate rather
//!   than a distinguished constant node.  Collected as a free parameter it is
//!   bound to a sample such as `1.7`, and every expression containing it is
//!   then evaluated with `π = 1.7` — so a correct answer in the casus
//!   irreducibilis form `2√(−p/3)·cos((acos c + 2πk)/3)` disagrees at every
//!   sample and is refused.  That is a false refusal caused entirely by the
//!   collector.
//! * **the imaginary unit** `I`, which has no real value at all.  Binding it
//!   to a real sample turns `e^{iωt}` into a real exponential and reports a
//!   disagreement that is an artefact of the binding.
//!
//! Getting this wrong is silent in the safe direction — it refuses correct
//! answers rather than accepting wrong ones — which is why it survived in
//! three places at once.  The predicate lives here so a fourth gate inherits
//! it instead of rediscovering it.
//!
//! # Who resolves them
//!
//! The **evaluators** do, not their callers.  [`crate::eval::eval_f64`],
//! [`crate::eval::eval_interval`], [`crate::eval::eval_complex_f64`] and
//! [`crate::jit::eval_interp`] each resolve `π` (and, in complex mode, `I`)
//! without a binding, so no caller has to seed an environment with it and no
//! two callers can disagree about the value.  Five of them used to, in the same
//! one line each.
//!
//! An explicit binding still wins.  That is deliberate: a caller that wants
//! `pi` treated as an ordinary free parameter — a sensitivity sweep, a
//! deliberate perturbation — says so by binding it, and is obeyed.  The
//! default is a default, not a reservation of the name.
//!
//! One evaluator deliberately does **not** resolve `π`:
//! [`crate::eval::eval_exact_rational`], which returns exact rationals and has
//! no rational to return.  Nor does anything resolve the imaginary unit to a
//! *real* value; only the complex evaluator knows it.

use crate::kernel::{ExprData, ExprId, ExprPool};

/// The name `π` is interned under.
pub(crate) const PI_NAME: &str = "pi";

/// Is `expr` the symbol `π`?
pub(crate) fn is_pi(expr: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(expr), ExprData::Symbol { name, .. } if name == PI_NAME)
}

/// The symbol `π`, interned in `pool`.
///
/// One spelling, so a rule that *builds* `π` and a gate that *recognises* it
/// cannot disagree about its domain — `pool.symbol` interns by name **and**
/// domain, so a stray `Domain::Positive` π would be a different node that
/// [`is_pi`] still matched and `ExprId` equality did not.
pub(crate) fn pi_symbol(pool: &ExprPool) -> ExprId {
    pool.symbol(PI_NAME, crate::kernel::Domain::Real)
}

/// Is `expr` a symbol that already denotes a number — `π` or the imaginary
/// unit — and so must never be bound to a sampled value?
pub(crate) fn is_named_constant(expr: ExprId, pool: &ExprPool) -> bool {
    is_pi(expr, pool) || pool.is_imaginary_unit(expr)
}

/// Visit every [`ExprData::Symbol`] node reachable from `expr`.
pub(crate) fn walk_symbols(expr: ExprId, pool: &ExprPool, f: &mut impl FnMut(ExprId, &ExprPool)) {
    match pool.get(expr) {
        ExprData::Symbol { .. } => f(expr, pool),
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            for a in args {
                walk_symbols(a, pool, f);
            }
        }
        ExprData::Pow { base, exp } => {
            walk_symbols(base, pool, f);
            walk_symbols(exp, pool, f);
        }
        _ => {}
    }
}

/// Append the symbols of `expr` that are genuine free parameters — everything
/// [`is_named_constant`] rejects is left out.  Existing entries of `out` are
/// preserved and not duplicated, so several expressions can be accumulated.
pub(crate) fn collect_free_symbols(expr: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    walk_symbols(expr, pool, &mut |s, pool| {
        if !is_named_constant(s, pool) && !out.contains(&s) {
            out.push(s);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn pi_and_i_are_constants_and_everything_else_is_a_parameter() {
        let p = ExprPool::new();
        let pi = p.symbol(PI_NAME, Domain::Real);
        let i = p.imaginary_unit();
        let k = p.symbol("k", Domain::Real);
        assert!(is_named_constant(pi, &p));
        assert!(is_named_constant(i, &p));
        assert!(!is_named_constant(k, &p));

        // `cos(k + 2*pi) * I`
        let two_pi = p.mul(vec![p.integer(2_i32), pi]);
        let e = p.mul(vec![p.func("cos", vec![p.add(vec![k, two_pi])]), i]);
        let mut free = Vec::new();
        collect_free_symbols(e, &p, &mut free);
        assert_eq!(free, vec![k]);
    }

    /// The evaluators resolve `π` themselves, so no caller has to bind it —
    /// and one that *wants* it free still overrides.
    #[test]
    fn every_real_evaluator_resolves_pi_and_an_explicit_binding_still_wins() {
        use crate::ball::IntervalEval;
        use std::collections::HashMap;

        let p = ExprPool::new();
        let pi = pi_symbol(&p);
        let e = p.mul(vec![p.rational(1, 2), pi]);

        let empty: HashMap<ExprId, f64> = HashMap::new();
        let want = std::f64::consts::FRAC_PI_2;
        assert!((crate::eval::eval_f64(e, &p, &empty).unwrap() - want).abs() < 1e-15);
        assert!((crate::jit::eval_interp(e, &empty, &p).unwrap() - want).abs() < 1e-15);

        let ball = crate::eval::eval_interval(e, &p, &IntervalEval::new(128)).unwrap();
        assert!(ball.lo().to_f64() <= want && want <= ball.hi().to_f64());
        assert!(ball.rad_f64() < 1e-30, "π enclosure is not tight: {ball}");

        // A caller that deliberately wants `pi` to be a free parameter says so,
        // and is obeyed — the default is a default, not a seizure of the name.
        let bound: HashMap<ExprId, f64> = HashMap::from([(pi, 10.0)]);
        assert_eq!(crate::eval::eval_f64(e, &p, &bound).unwrap(), 5.0);
        assert_eq!(crate::jit::eval_interp(e, &bound, &p).unwrap(), 5.0);
        let mut iv = IntervalEval::new(64);
        iv.bind(pi, crate::ball::ArbBall::from_f64(10.0, 64));
        assert_eq!(
            crate::eval::eval_interval(e, &p, &iv).unwrap().mid_f64(),
            5.0
        );
    }
}
