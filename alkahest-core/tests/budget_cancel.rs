//! Cooperative cancellation (`request_cancel` / `clear_cancel`).
//!
//! These tests live in their own test binary, which means their own process,
//! and must stay here. The cancellation flag is a process-wide `AtomicBool`
//! by design: an orchestrator thread cancels a heavy call running on another
//! thread. Every test in the crate's unit-test binary reaches
//! `budget::check()`, because `simplify` calls it once per pass and stops
//! rewriting when it trips. When these tests ran in that binary, whatever
//! happened to be simplifying on another thread while the flag was set got
//! back a half-simplified result. `algebra::quaternion_tests::
//! the_hamilton_product_table` failed `k² = -1` that way, intermittently and
//! only in full runs.
//!
//! Within this binary the tests still share the flag, so they take a lock.

use alkahest_cas::budget::{
    check, check_growth, clear_cancel, enter, is_cancelled, request_cancel, Budget, BudgetError,
};
use alkahest_cas::errors::AlkahestError;
use alkahest_cas::kernel::{Domain, ExprPool};
use alkahest_cas::simplify::engine::simplify;
use std::sync::{Mutex, MutexGuard};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Clears cancellation on drop, so a failing assertion can't leave the flag
/// set for the next test that takes the lock.
struct CancelGuard;
impl Drop for CancelGuard {
    fn drop(&mut self) {
        clear_cancel();
    }
}

#[test]
fn cancel_flag_trips_check_and_clears() {
    let _serial = serial();
    let _cancel_guard = CancelGuard;
    assert!(!is_cancelled());
    request_cancel();
    assert!(is_cancelled());
    let err = check().unwrap_err();
    assert_eq!(err.code(), "E-BUDGET-003");
    assert_eq!(err, BudgetError::Cancelled);
    clear_cancel();
    assert!(!is_cancelled());
    assert!(check().is_ok());
}

#[test]
fn cancel_trips_even_with_a_generous_budget_active() {
    let _serial = serial();
    let _cancel_guard = CancelGuard;
    let _guard = enter(Budget::new().with_max_steps(1_000_000));
    request_cancel();
    assert_eq!(check().unwrap_err(), BudgetError::Cancelled);
}

/// A growth checkpoint is also an ordinary one: cancellation trips through it.
#[test]
fn growth_checkpoint_still_reports_cancel() {
    let _serial = serial();
    let _cancel_guard = CancelGuard;
    request_cancel();
    assert_eq!(check_growth(1).unwrap_err(), BudgetError::Cancelled);
}

/// Shows why these tests need their own process. A pending cancellation
/// affects `simplify` on *any* thread, which is the documented contract. It
/// stops rewriting, so `x + 0` comes back unsimplified. In the unit-test
/// binary, that is what a concurrently running test observed.
#[test]
fn a_pending_cancel_stops_simplify_on_another_thread() {
    let _serial = serial();
    let _cancel_guard = CancelGuard;
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let e = pool.add(vec![x, pool.integer(0_i32)]);
    assert_eq!(simplify(e, &pool).value, x);

    request_cancel();
    let pool2 = ExprPool::new();
    let y = pool2.symbol("y", Domain::Real);
    let e2 = pool2.add(vec![y, pool2.integer(0_i32)]);
    let got = std::thread::spawn(move || simplify(e2, &pool2).value == y)
        .join()
        .unwrap();
    assert!(!got, "simplify ran to completion under a pending cancel");
}
