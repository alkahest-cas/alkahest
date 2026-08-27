//! A runaway *allocation* must stop when the budget says so.
//!
//! `crate::budget` is a cooperative mechanism: it can only stop an algorithm
//! that comes back to ask. Until `check_growth` existed, one that left for a
//! single step and grew a `Vec` without bound could not be stopped at all —
//! the process died in `handle_alloc_error`, which aborts without unwinding,
//! so no `Budget`, no wall clock and no `catch_unwind` could intervene.
//!
//! The integrand below is `1/(x·log x·(1 + log²(log x)))`, the derivative of
//! `atan(log(log x))`: a two-level log tower with the outer generator in a
//! *denominator*. Integrating it made `risch::tower::decompose_log_inner` read
//! the exponent `-1` off `log(x)^-1`, cast it to `usize` (≈ 2⁶⁴) and push zero
//! coefficients until the allocator gave up — 26 GB of resident memory in ten
//! minutes on the machine this was written on, under a three-second budget.
//!
//! Making that integral *work* is a separate question; these tests are about
//! it stopping when asked, and about the refusal being legible as a resource
//! limit rather than as a mathematical verdict.

use alkahest_cas::budget::{self, Budget};
use alkahest_cas::errors::AlkahestError;
use alkahest_cas::kernel::{Domain, ExprId, ExprPool};
use std::time::Duration;

/// `d/dx atan(log(log x))` = `1/(x·log x·(1 + log²(log x)))`.
fn atan_log_log_derivative(pool: &ExprPool, x: ExprId) -> ExprId {
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), x);
    let e = alkahest_cas::parse::parse("atan(log(log(x)))", pool, &mut env).unwrap();
    let d = alkahest_cas::diff::diff(e, x, pool).unwrap();
    alkahest_cas::simplify::simplify(d.value, pool).value
}

/// Asserted on the error *kind*, not on elapsed seconds: a wall-clock
/// assertion here would be load-sensitive, and the guard under test is not a
/// deadline anyway — it refuses before the first allocation, so the call
/// returns in microseconds whatever the machine is doing.
#[test]
fn runaway_growth_stops_with_a_budget_error() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let f = atan_log_log_derivative(&pool, x);

    let _guard = budget::enter(Budget::new().with_wall(Duration::from_millis(3000)));
    let err = alkahest_cas::integrate::integrate(f, x, &pool)
        .expect_err("this integrand has no elementary form this engine can reach");

    assert!(
        err.is_budget(),
        "a resource refusal must not arrive as a mathematical verdict: {err}"
    );
    assert_eq!(err.budget_code(), Some("E-BUDGET-002"));
}

/// The refusal is not an artefact of having entered a budget. The abort it
/// replaces happened to callers who entered none, and they are the ones who
/// could not have opted into a limit for an allocation they did not know a
/// call would make.
#[test]
fn runaway_growth_stops_with_no_budget_entered() {
    assert!(!budget::is_active());
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let f = atan_log_log_derivative(&pool, x);

    let err = alkahest_cas::integrate::integrate(f, x, &pool).expect_err("must refuse");
    assert_eq!(err.budget_code(), Some("E-BUDGET-002"));
}

/// A *mathematical* decline must stay a mathematical decline — several tests
/// elsewhere separate `E-BUDGET-*` from `E-INT-*`, and a growth checkpoint
/// that turned every refusal into a budget trip would erase that distinction.
#[test]
fn an_ordinary_decline_is_still_not_a_budget_trip() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    // ∫ exp(x)/x dx — the exponential integral Ei, genuinely non-elementary.
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), x);
    let f = alkahest_cas::parse::parse("exp(x)/x", &pool, &mut env).unwrap();

    let _guard = budget::enter(Budget::new().with_wall(Duration::from_secs(60)));
    let err = alkahest_cas::integrate::integrate(f, x, &pool).expect_err("Ei is not elementary");
    assert!(
        !err.is_budget(),
        "a mathematical verdict must not arrive as a budget trip: {err}"
    );
    assert_eq!(err.budget_code(), None);
    assert!(
        err.code().starts_with("E-INT-"),
        "expected an integration code, got {}",
        err.code()
    );
}

/// The ceiling is a default, not a cap: a caller who genuinely wants a bigger
/// accumulator says so, and `E-BUDGET-002`'s remediation ("raise
/// `Budget(max_steps=...)`") is therefore true rather than merely plausible.
#[test]
fn the_growth_ceiling_is_raisable() {
    let over = budget::DEFAULT_MAX_GROWTH_UNITS + 1;
    assert!(budget::check_growth(over).is_err());
    let _guard = budget::enter(Budget::new().with_max_steps(over * 2));
    assert!(budget::check_growth(over).is_ok());
}
