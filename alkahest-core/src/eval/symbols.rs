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
    pool.with(
        expr,
        |d| matches!(d, ExprData::Symbol { name, .. } if name == PI_NAME),
    )
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

/// The name the Euler–Mascheroni constant `γ = 0.5772…` is interned under
/// (SymPy's and Mathematica's spelling).
pub(crate) const EULER_GAMMA_NAME: &str = "EulerGamma";
/// The name Catalan's constant `G = Σ (−1)ᵏ/(2k+1)² = 0.9159…` is interned
/// under.
pub(crate) const CATALAN_NAME: &str = "Catalan";
/// The name the Glaisher–Kinkelin constant `A = 1.2824…` is interned under.
pub(crate) const GLAISHER_NAME: &str = "Glaisher";

/// Prefix of the Stieltjes constants `γₙ` (`n ≥ 1`), interned as
/// `StieltjesGamma1`, `StieltjesGamma2`, … (`γ₀` is `EulerGamma`).
pub(crate) const STIELTJES_PREFIX: &str = "StieltjesGamma";
/// Largest `n` for which `γₙ` is a named constant (the `f64` table's length).
pub(crate) const MAX_STIELTJES: u32 = 30;

/// `n` when `name` is `StieltjesGamma{n}` with `1 <= n <=` [`MAX_STIELTJES`].
pub(crate) fn stieltjes_index(name: &str) -> Option<u32> {
    let n: u32 = name.strip_prefix(STIELTJES_PREFIX)?.parse().ok()?;
    ((1..=MAX_STIELTJES).contains(&n) && name == format!("{STIELTJES_PREFIX}{n}")).then_some(n)
}

/// `γₙ` for `1 <= n <= 30` (mpmath `stieltjes`).
#[allow(clippy::excessive_precision)]
const STIELTJES_F64: [f64; MAX_STIELTJES as usize] = [
    -0.07281584548367673,
    -0.00969036319287232,
    0.002053834420303346,
    0.0023253700654673,
    0.0007933238173010627,
    -0.0002387693454301996,
    -0.000527289567057751,
    -0.0003521233538030395,
    -3.439477441808805e-05,
    0.0002053328149090648,
    0.0002701844395439035,
    0.0001672729121051402,
    -2.7463806603760158e-05,
    -0.00020920926205929996,
    -0.0002834686553202414,
    -0.00019969685830896976,
    2.6277037109918338e-05,
    0.0003073684081492528,
    0.0005036054530473557,
    0.00046634356151155945,
    0.00010443776975600011,
    -0.0005415995822039977,
    -0.0012439620904082457,
    -0.0015885112789035616,
    -0.0010745919527384888,
    0.0006568035186371545,
    0.0034778369136185382,
    0.00640006853170063,
    0.007371151770472239,
    0.003557728855573161,
];

/// The `f64` value of a named real constant — `π`, `γ` (`EulerGamma`),
/// Catalan's `G` and Glaisher's `A` — or `None` for any other name.
///
/// Like `π`, the other three are ordinary [`ExprData::Symbol`] nodes: they
/// print, parse and substitute as symbols, and every evaluator resolves them
/// through this one table (an explicit binding still wins). They are **never**
/// folded into a float literal by a symbolic path — `ψ(1)` is `−EulerGamma`,
/// not `−0.5772…`.
pub(crate) fn constant_value_f64(name: &str) -> Option<f64> {
    match name {
        PI_NAME => Some(std::f64::consts::PI),
        EULER_GAMMA_NAME => Some(0.577_215_664_901_532_9),
        CATALAN_NAME => Some(0.915_965_594_177_219),
        GLAISHER_NAME => Some(1.282_427_129_100_622_6),
        other => stieltjes_index(other).map(|n| STIELTJES_F64[n as usize - 1]),
    }
}

/// The `f64` value of `expr` when it is one of the named real constants of
/// [`constant_value_f64`] (all of them are positive).
pub(crate) fn constant_value_of(expr: ExprId, pool: &ExprPool) -> Option<f64> {
    pool.with(expr, |d| match d {
        ExprData::Symbol { name, .. } => constant_value_f64(name),
        _ => None,
    })
}

/// Is `expr` one of the named real constants of [`constant_value_f64`]?
pub(crate) fn is_real_constant(expr: ExprId, pool: &ExprPool) -> bool {
    constant_value_of(expr, pool).is_some()
}

/// The Euler–Mascheroni constant `γ`, interned in `pool` (same domain as
/// [`pi_symbol`], for the same reason).
pub(crate) fn euler_gamma_symbol(pool: &ExprPool) -> ExprId {
    pool.symbol(EULER_GAMMA_NAME, crate::kernel::Domain::Real)
}

/// Is `expr` a symbol that already denotes a number — `π`, `γ`, Catalan's or
/// Glaisher's constant, or the imaginary unit — and so must never be bound to
/// a sampled value?
pub(crate) fn is_named_constant(expr: ExprId, pool: &ExprPool) -> bool {
    is_real_constant(expr, pool) || pool.is_imaginary_unit(expr)
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

    #[test]
    fn named_constants_resolve_in_every_real_evaluator() {
        use crate::ball::IntervalEval;
        use std::collections::HashMap;

        let p = ExprPool::new();
        let empty: HashMap<ExprId, f64> = HashMap::new();
        for (name, want) in [
            (EULER_GAMMA_NAME, 0.577_215_664_901_532_9_f64),
            (CATALAN_NAME, 0.915_965_594_177_219),
            (GLAISHER_NAME, 1.282_427_129_100_622_6),
        ] {
            let c = p.symbol(name, Domain::Real);
            assert!(is_named_constant(c, &p));
            let mut free = Vec::new();
            collect_free_symbols(c, &p, &mut free);
            assert!(free.is_empty());
            assert!((crate::eval::eval_f64(c, &p, &empty).unwrap() - want).abs() < 1e-15);
            assert!((crate::jit::eval_interp(c, &empty, &p).unwrap() - want).abs() < 1e-15);
            let ball = crate::eval::eval_interval(c, &p, &IntervalEval::new(200)).unwrap();
            assert!(ball.lo().to_f64() <= want && want <= ball.hi().to_f64());
            assert!(
                ball.rad_f64() < 1e-50,
                "{name} enclosure is not tight: {ball}"
            );
        }
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
