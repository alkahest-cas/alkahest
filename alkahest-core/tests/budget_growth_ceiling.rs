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
//! **That integrand no longer reaches the runaway path, and these tests no
//! longer use it.** Two independent fixes landed since this file was written
//! and compose: `decompose_as_log_poly` now declines a negative degree via
//! `usize::try_from` instead of wrapping it (so the growth never starts), and
//! the router falls through on a sub-engine decline (so the integrand reaches
//! the derivative-divides substitution, which *solves* it — the integral now
//! returns `atan(log(log x))`).
//!
//! That is a better outcome than a clean refusal, but it means driving this
//! guard through `integrate` would leave the test hostage to routing changes
//! it is not about. So the ceiling is exercised **directly** below. The
//! integrator-level test that remains is the one that still has a stable
//! premise: that an ordinary mathematical decline is not reported as a budget
//! trip.

use alkahest_cas::budget::{self, Budget};
use alkahest_cas::errors::AlkahestError;
use alkahest_cas::kernel::{Domain, ExprPool};
use std::time::Duration;

/// The ceiling refuses a request past the limit, and does so *before* the
/// allocation — so the call returns in microseconds whatever the machine is
/// doing. Asserted on the error kind, never on elapsed seconds: a wall-clock
/// assertion here would be load-sensitive, and this guard is not a deadline.
#[test]
fn growth_past_the_ceiling_is_refused_as_a_budget_error() {
    let _guard = budget::enter(Budget::new().with_wall(Duration::from_millis(3000)));
    let err = budget::check_growth(u64::MAX).expect_err("a 2^64-unit request must be refused");
    assert_eq!(err.code(), "E-BUDGET-002");
}

/// The refusal is not an artefact of having entered a budget. The abort it
/// replaces happened to callers who entered none, and they are the ones who
/// could not have opted into a limit for an allocation they did not know a
/// call would make.
#[test]
fn growth_is_bounded_with_no_budget_entered() {
    assert!(!budget::is_active());
    let err =
        budget::check_growth(u64::MAX).expect_err("the default ceiling applies unconditionally");
    assert_eq!(err.code(), "E-BUDGET-002");
}

/// An ordinary-sized request is not refused — the ceiling must not fire on
/// legitimate work, or it would turn every large-but-honest computation into
/// a spurious resource error.
#[test]
fn an_ordinary_growth_request_is_allowed() {
    assert!(budget::check_growth(1024).is_ok());
    let _guard = budget::enter(Budget::new().with_wall(Duration::from_millis(3000)));
    assert!(budget::check_growth(1024).is_ok());
}

/// A *mathematical* decline must stay a mathematical decline — several tests
/// elsewhere separate `E-BUDGET-*` from `E-INT-*`, and a growth checkpoint
/// that turned every refusal into a budget trip would erase that distinction.
#[test]
fn an_ordinary_decline_is_still_not_a_budget_trip() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    // ∫ exp(x²) dx — genuinely non-elementary, and with no closed form over the
    // registered basis either: it is `erfi`, which is not a primitive here.
    // (`exp(x)/x` used to be the witness; it is answered as `Ei(x)` now, so it
    // no longer exercises the decline path at all.)
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), x);
    let f = alkahest_cas::parse::parse("exp(x^2)", &pool, &mut env).unwrap();

    let _guard = budget::enter(Budget::new().with_wall(Duration::from_secs(60)));
    let err =
        alkahest_cas::integrate::integrate(f, x, &pool).expect_err("exp(x²) is not elementary");
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
