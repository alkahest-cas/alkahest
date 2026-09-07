//! `∫_0^∞ e^{-k·t} dt` and friends: an improper integral whose value is
//! decided by a *stated fact* about a free parameter.
//!
//! The engines involved are three deep — `integrate_definite` finds an
//! antiderivative, `eval_bound` asks for its limit at `∞`, and `limit` asks
//! for the sign of `−k` — and only the last one cares about the assumption.
//! These are end-to-end because that chain is the whole point: the unit tests
//! in `calculus::limits` pin the sign reasoning, and these pin that it is
//! actually reached from the call a user makes.

use alkahest_cas::calculus::limits::last_missing_assumption;
use alkahest_cas::integrate::integrate_definite;
use alkahest_cas::kernel::expr::PredicateKind;
use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use alkahest_cas::simplify::{enter_assumptions, simplify, AssumptionContext};
use std::collections::HashMap;

/// Numeric value of a closed form with the named symbols bound.
fn evaluate(expr: ExprId, bindings: &[(ExprId, f64)], pool: &ExprPool) -> f64 {
    let env: HashMap<ExprId, f64> = bindings.iter().copied().collect();
    alkahest_cas::jit::eval_interp(simplify(expr, pool).value, &env, pool)
        .unwrap_or_else(|| panic!("{} did not evaluate", pool.display(expr)))
}

fn decay(pool: &ExprPool, rate: ExprId, t: ExprId) -> ExprId {
    simplify(
        pool.func(
            "exp".to_string(),
            vec![pool.mul(vec![pool.integer(-1_i32), rate, t])],
        ),
        pool,
    )
    .value
}

#[test]
fn a_stated_positive_rate_makes_the_improper_integral_computable() {
    let pool = ExprPool::new();
    let t = pool.symbol("t", Domain::Real);
    let ke = pool.symbol("ke", Domain::Real);
    let integrand = decay(&pool, ke, t);

    let mut assumptions = AssumptionContext::new();
    assumptions
        .refine(
            pool.predicate(PredicateKind::Gt, vec![ke, pool.integer(0_i32)]),
            &pool,
        )
        .unwrap();
    let _scope = enter_assumptions(&assumptions, &pool);

    let value = integrate_definite(
        integrand,
        t,
        pool.integer(0_i32),
        pool.pos_infinity(),
        &pool,
    )
    .expect("∫_0^∞ e^{-ke·t} dt = 1/ke under ke > 0")
    .value;
    // 1/ke, whatever arrangement of signs and reciprocals it comes out in.
    for rate in [0.5_f64, 1.0, 3.0, 7.25] {
        let got = evaluate(value, &[(ke, rate)], &pool);
        assert!(
            (got - 1.0 / rate).abs() < 1e-12,
            "expected 1/{rate}, got {got} from {}",
            pool.display(value)
        );
    }
}

#[test]
fn the_pharmacokinetic_auc_integral_is_computable() {
    // AUC = ∫_0^∞ (D/V)·e^{-ke·t} dt = D/(V·ke).
    let pool = ExprPool::new();
    let t = pool.symbol("t", Domain::Real);
    let ke = pool.symbol("ke", Domain::Real);
    let dose = pool.symbol("D", Domain::Real);
    let volume = pool.symbol("V", Domain::Real);
    let integrand = simplify(
        pool.mul(vec![
            dose,
            pool.pow(volume, pool.integer(-1_i32)),
            decay(&pool, ke, t),
        ]),
        &pool,
    )
    .value;

    let mut assumptions = AssumptionContext::new();
    assumptions
        .refine(
            pool.predicate(PredicateKind::Gt, vec![ke, pool.integer(0_i32)]),
            &pool,
        )
        .unwrap();
    let _scope = enter_assumptions(&assumptions, &pool);

    let value = integrate_definite(
        integrand,
        t,
        pool.integer(0_i32),
        pool.pos_infinity(),
        &pool,
    )
    .expect("AUC must be computable once the elimination rate is known positive")
    .value;
    let got = evaluate(value, &[(ke, 0.25), (dose, 500.0), (volume, 40.0)], &pool);
    assert!(
        (got - 500.0 / (40.0 * 0.25)).abs() < 1e-9,
        "expected D/(V·ke) = 50, got {got} from {}",
        pool.display(value)
    );
}

#[test]
fn without_the_assumption_the_integral_is_refused_and_says_what_is_missing() {
    let pool = ExprPool::new();
    let t = pool.symbol("t", Domain::Real);
    let ke = pool.symbol("ke", Domain::Real);
    let integrand = decay(&pool, ke, t);

    let err = integrate_definite(
        integrand,
        t,
        pool.integer(0_i32),
        pool.pos_infinity(),
        &pool,
    )
    .expect_err("the value depends on the sign of ke, which nothing states");
    let message = err.to_string();
    assert!(
        message.contains("ke > 0") && message.contains("ke < 0"),
        "the refusal must name the missing fact; got: {message}"
    );
    let missing = last_missing_assumption().expect("recorded out of band for the bindings");
    assert_eq!(missing.parameter, "ke");
}

#[test]
fn a_stated_negative_rate_is_reported_as_divergent_not_refused_for_want_of_a_rule() {
    let pool = ExprPool::new();
    let t = pool.symbol("t", Domain::Real);
    let ke = pool.symbol("ke", Domain::Real);
    let integrand = decay(&pool, ke, t);

    let mut assumptions = AssumptionContext::new();
    assumptions
        .refine(
            pool.predicate(PredicateKind::Lt, vec![ke, pool.integer(0_i32)]),
            &pool,
        )
        .unwrap();
    let _scope = enter_assumptions(&assumptions, &pool);

    let err = integrate_definite(
        integrand,
        t,
        pool.integer(0_i32),
        pool.pos_infinity(),
        &pool,
    )
    .expect_err("∫_0^∞ e^{|ke|·t} dt diverges");
    assert!(
        err.to_string().contains("not finite"),
        "a divergent integral should say so; got: {err}"
    );
}

#[test]
fn a_positive_domain_rate_needs_no_assumption_context() {
    let pool = ExprPool::new();
    let t = pool.symbol("t", Domain::Real);
    let ke = pool.symbol("ke", Domain::Positive);
    let integrand = decay(&pool, ke, t);

    let value = integrate_definite(
        integrand,
        t,
        pool.integer(0_i32),
        pool.pos_infinity(),
        &pool,
    )
    .expect("Domain::Positive states the same fact")
    .value;
    let got = evaluate(value, &[(ke, 2.0)], &pool);
    assert!((got - 0.5).abs() < 1e-12, "expected 1/2, got {got}");
}

#[test]
fn a_numeric_rate_still_gives_the_same_answer() {
    // The control: this worked before and must be untouched.
    let pool = ExprPool::new();
    let t = pool.symbol("t", Domain::Real);
    let integrand = decay(&pool, pool.integer(2_i32), t);
    let value = integrate_definite(
        integrand,
        t,
        pool.integer(0_i32),
        pool.pos_infinity(),
        &pool,
    )
    .expect("∫_0^∞ e^{-2t} dt = 1/2")
    .value;
    assert_eq!(simplify(value, &pool).value, pool.rational(1, 2));
}
