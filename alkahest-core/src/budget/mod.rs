//! Per-call wall-clock and step budgets, cooperative cancellation, and a
//! deterministic seed for search-style workloads.
//!
//! # Motivation
//!
//! A fan-out loop trying 10k candidate rewrites/integrals/Gröbner bases
//! cannot afford one pathological candidate to hang the whole batch, and an
//! orchestrator that decides "this candidate isn't worth it" needs a way to
//! *stop it now* rather than waiting for an OS-level kill (`SIGKILL`, process
//! timeout). This module gives heavy engines a cheap, structured way to bail
//! out honestly — returning [`BudgetError`] — instead of running unbounded or
//! being killed with no diagnostic.
//!
//! # Model
//!
//! A [`Budget`] is *entered* with [`enter`], which pushes it onto a
//! thread-local stack and returns a [`BudgetGuard`]. The budget stays active
//! until the guard is dropped (including on panic-unwind), mirroring the
//! `try`/`finally` discipline of Python's `with alkahest.context(budget=...)`.
//! Budgets nest like `context(...)` blocks: only the *innermost* active frame
//! is consulted by [`check`] and [`seed`] — entering a new budget shadows the
//! outer one for the scope of the block rather than combining limits with it.
//!
//! Heavy call sites ([`crate::integrate::engine`]'s top-level entry and
//! recursion boundary, [`crate::simplify::engine`]'s per-pass loop) call
//! [`check`] at a handful of strategic points. `check` is cheap when no
//! budget is active and cancellation has not been requested — it is a single
//! atomic load plus (if a budget is active) an `Instant::now()` — so it is
//! safe to call unconditionally at those boundaries.
//!
//! # Time is not the only way to run away
//!
//! [`check`] bounds an algorithm that keeps *returning* to a checkpoint. It
//! cannot bound one that leaves for a single step which allocates without
//! bound: no deadline helps if the deadline is never consulted again, and the
//! process dies of `handle_alloc_error` — an abort, not an error any caller
//! can catch.
//!
//! [`check_growth`] is the size half of the same mechanism. A step that is
//! about to grow a structure by `units` says so *before* growing, and is
//! refused if `units` is more work than the budget allows. Because the
//! failure it prevents is an abort rather than a slow answer, it applies
//! [`DEFAULT_MAX_GROWTH_UNITS`] even when no [`Budget`] is active: a caller
//! cannot opt into a limit for an allocation it did not know a call would
//! make. `Budget { max_steps: Some(n), .. }` replaces that default, so the
//! `E-BUDGET-002` remediation ("raise `Budget(max_steps=...)`") is the true
//! remedy for both halves.
//!
//! Cancellation ([`request_cancel`] / [`is_cancelled`] / [`clear_cancel`]) is
//! a single process-wide flag, deliberately *not* scoped to a thread or a
//! [`Budget`] frame: it models "the orchestrator wants the current heavy
//! operation to stop right now", e.g. because a fan-out loop decided a
//! candidate has used enough wall time across every thread working on it.
//! Call [`clear_cancel`] before starting the next candidate.
//!
//! [`seed`] exposes the active budget's seed so RNG-consuming samplers (e.g.
//! randomized modular tests, homotopy continuation start systems) can be
//! seeded deterministically from the ambient budget instead of threading an
//! explicit seed parameter through every call — two runs entering the same
//! `Budget { seed: Some(7), .. }` observe the same [`seed`] at every call site
//! that consults it.
//!
//! # Errors
//!
//! [`BudgetError`] implements [`AlkahestError`] with stable codes:
//!
//! | Code           | Variant                | Cause             |
//! |----------------|-------------------------|-------------------|
//! | `E-BUDGET-001` | [`BudgetError::WallClock`] | wall-clock deadline elapsed |
//! | `E-BUDGET-002` | [`BudgetError::Steps`]     | step counter exceeded `max_steps`, or one step asked to grow past it |
//! | `E-BUDGET-003` | [`BudgetError::Cancelled`] | [`request_cancel`] was called |

use crate::errors::AlkahestError;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

/// A per-call resource budget: an optional wall-clock limit, an optional
/// cooperative step limit, and an optional determinism seed.
///
/// Every field is optional. `Budget::default()` never trips [`check`] on its
/// own — only [`request_cancel`] can stop a call entered with a default
/// budget. This is intentional: entering an (otherwise empty) budget is how
/// a caller opts a code path into consulting [`seed`] without also imposing
/// a wall/step limit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Budget {
    /// Wall-clock limit for the guarded block, measured from [`enter`].
    pub wall: Option<Duration>,
    /// Maximum number of work units the guarded block may spend.
    ///
    /// Counted by [`check`], which charges one unit per call, and by
    /// [`check_growth`], which charges a step for the size of the structure it
    /// is about to grow. When `None`, [`check`] is uncounted and
    /// [`check_growth`] falls back to [`DEFAULT_MAX_GROWTH_UNITS`].
    pub max_steps: Option<u64>,
    /// Determinism seed available to callers via [`seed`].
    pub seed: Option<u64>,
}

impl Budget {
    /// An empty budget — no wall/step limit, no seed. Equivalent to
    /// `Budget::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the wall-clock limit.
    pub fn with_wall(mut self, wall: Duration) -> Self {
        self.wall = Some(wall);
        self
    }

    /// Set the step limit.
    pub fn with_max_steps(mut self, max_steps: u64) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Set the determinism seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

struct Frame {
    start: Instant,
    wall: Option<Duration>,
    max_steps: Option<u64>,
    steps: Cell<u64>,
    seed: Option<u64>,
}

thread_local! {
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard returned by [`enter`].
///
/// Pops the corresponding [`Budget`] frame from the current thread's active-
/// budget stack on drop — on every exit path, including panic-unwind — so a
/// `?`-propagated error or an early `return` can never leak a stale budget
/// frame into unrelated code that runs later on the same thread.
///
/// Not [`Send`]: the stack it pops from is thread-local, so a guard created
/// on one thread must not be dropped from another.
pub struct BudgetGuard {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for BudgetGuard {
    fn drop(&mut self) {
        STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// Push `budget` onto the current thread's active-budget stack.
///
/// Returns a [`BudgetGuard`]; the budget stays active — visible to [`check`],
/// [`seed`], and [`is_active`] on this thread — until the guard is dropped.
/// Budgets nest like `with` blocks: entering a new one shadows the previous
/// one until it is popped, rather than combining limits with the outer
/// frame (matching the non-merging nesting semantics of
/// `alkahest.context(...)` on the Python side).
pub fn enter(budget: Budget) -> BudgetGuard {
    let frame = Frame {
        start: Instant::now(),
        wall: budget.wall,
        max_steps: budget.max_steps,
        steps: Cell::new(0),
        seed: budget.seed,
    };
    STACK.with(|s| s.borrow_mut().push(frame));
    BudgetGuard {
        _not_send: std::marker::PhantomData,
    }
}

/// Returns `true` if a [`Budget`] is currently active on this thread.
pub fn is_active() -> bool {
    STACK.with(|s| !s.borrow().is_empty())
}

/// The seed of the innermost active [`Budget`] on this thread, or `None` if
/// no budget is active or the active budget did not set one.
pub fn seed() -> Option<u64> {
    STACK.with(|s| s.borrow().last().and_then(|f| f.seed))
}

// ---------------------------------------------------------------------------
// Cancellation — process-wide, not scoped to a thread or Budget frame
// ---------------------------------------------------------------------------

static CANCELLED: AtomicBool = AtomicBool::new(false);

/// Request cancellation of the current cooperative operation(s).
///
/// Checked by every [`check`] call on every thread — regardless of whether a
/// [`Budget`] is active — until [`clear_cancel`] is called. Intended for an
/// orchestrator thread to stop a heavy call it decided is no longer worth
/// running, without waiting for an OS-level kill.
pub fn request_cancel() {
    CANCELLED.store(true, Ordering::SeqCst);
}

/// Clear a previously requested cancellation.
///
/// Call this before starting the next candidate in a fan-out loop — a
/// cancellation request left set would otherwise trip [`check`]
/// (`E-BUDGET-003`) immediately for every subsequent candidate.
pub fn clear_cancel() {
    CANCELLED.store(false, Ordering::SeqCst);
}

/// Returns `true` if [`request_cancel`] has been called and not yet cleared
/// by [`clear_cancel`].
pub fn is_cancelled() -> bool {
    CANCELLED.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Cooperative check
// ---------------------------------------------------------------------------

/// Cooperative checkpoint: call this at a natural short-circuit point in a
/// heavy algorithm — top-level entry, a recursion/depth-guard boundary, once
/// per major rewrite pass.
///
/// Checks, in order:
/// 1. [`is_cancelled`] — the process-wide cancellation flag.
/// 2. The innermost active [`Budget`]'s wall-clock limit, if any.
/// 3. The innermost active [`Budget`]'s step counter, if any — incremented on
///    every call, compared against `max_steps` after incrementing.
///
/// Returns `Ok(())` with no side effect (no counter increment) if no
/// [`Budget`] is active and cancellation has not been requested, so `check`
/// is cheap to call unconditionally at hot-loop boundaries even when no
/// caller has opted into a budget.
pub fn check() -> Result<(), BudgetError> {
    if is_cancelled() {
        return Err(BudgetError::Cancelled);
    }
    STACK.with(|s| {
        let stack = s.borrow();
        let Some(frame) = stack.last() else {
            return Ok(());
        };
        if let Some(wall) = frame.wall {
            let elapsed = frame.start.elapsed();
            if elapsed >= wall {
                return Err(BudgetError::WallClock {
                    limit: wall,
                    elapsed,
                });
            }
        }
        if let Some(max_steps) = frame.max_steps {
            let taken = frame.steps.get() + 1;
            frame.steps.set(taken);
            if taken > max_steps {
                return Err(BudgetError::Steps {
                    limit: max_steps,
                    taken,
                });
            }
        }
        Ok(())
    })
}

/// Work units a single growth step may ask for when the active [`Budget`]
/// sets no `max_steps`, and when no budget is active at all.
///
/// 2²⁶ units. Sized to be unreachable by any well-formed problem and cheap to
/// reach by a runaway one: the loop this was introduced for asks for 2⁶⁴
/// coefficients of a log-polynomial whose own documented degree bound is 20,
/// and every legitimate accumulator this crate builds is smaller than a
/// megabyte. A `Vec<ExprId>` of this many entries is already 256 MiB, so a
/// caller who genuinely wants more is asking for something worth having to
/// say out loud via `Budget::with_max_steps`.
///
/// It is deliberately a *default* rather than a hard cap. Growth is refused,
/// never truncated: a bound that silently returned a short answer would be the
/// wrong answer, where a refusal is an honest one.
pub const DEFAULT_MAX_GROWTH_UNITS: u64 = 1 << 26;

/// Cooperative checkpoint for a step that is about to do `units` of work *at
/// once* — typically grow a collection by that many elements.
///
/// [`check`] answers "may I keep going?", which bounds an algorithm that keeps
/// coming back to ask. This answers "may I spend this much before I come
/// back?", which is the only question a single unbounded allocation ever gets
/// to ask. Call it **before** the allocation, with the number of units it
/// would add; a refusal costs nothing, where discovering the same thing
/// afterwards costs the process.
///
/// Checks, in order:
/// 1. Everything [`check`] checks — cancellation, wall clock, step counter —
///    so a growth checkpoint is also an ordinary one and needs no companion
///    `check` call.
/// 2. `units` against the innermost active [`Budget`]'s `max_steps`, or
///    against [`DEFAULT_MAX_GROWTH_UNITS`] if it set none *or if no budget is
///    active at all*.
///
/// Point 2 is the one place in this module that does something without an
/// active budget, and it is deliberate: the failure mode is
/// `handle_alloc_error`, which aborts the process without unwinding, so there
/// is no later point at which any caller could have intervened. A caller who
/// wants more must ask for it (`Budget::with_max_steps`), which is exactly
/// what [`BudgetError::Steps`]'s remediation already tells them to do.
///
/// The comparison is per request and holds no state, so it is safe to call
/// from a loop condition and cannot latch a refusal onto unrelated later work.
pub fn check_growth(units: u64) -> Result<(), BudgetError> {
    check()?;
    let limit = STACK
        .with(|s| s.borrow().last().and_then(|f| f.max_steps))
        .unwrap_or(DEFAULT_MAX_GROWTH_UNITS);
    if units > limit {
        return Err(BudgetError::Steps {
            limit,
            taken: units,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// A [`Budget`] was exceeded, or cancellation was requested.
///
/// This is a fine, expected answer for a fan-out search loop — not a crash —
/// so it carries a stable code and remediation like every other
/// `alkahest-core` error, rather than surfacing as a panic or an OS kill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetError {
    /// The active budget's wall-clock limit elapsed.
    WallClock { limit: Duration, elapsed: Duration },
    /// A work limit was exceeded: either the active budget's step counter
    /// passed `max_steps`, or a single step asked [`check_growth`] to spend
    /// more units at once than the budget (or
    /// [`DEFAULT_MAX_GROWTH_UNITS`]) allows. Both are the same currency, so
    /// they carry the same code and the same remedy.
    Steps { limit: u64, taken: u64 },
    /// [`request_cancel`] was called and not yet cleared.
    Cancelled,
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetError::WallClock { limit, elapsed } => write!(
                f,
                "budget exceeded: wall-clock limit {limit:?} elapsed ({elapsed:?} elapsed)"
            ),
            BudgetError::Steps { limit, taken } => write!(
                f,
                "budget exceeded: step limit {limit} reached ({taken} steps taken)"
            ),
            BudgetError::Cancelled => write!(f, "budget: operation was cancelled"),
        }
    }
}

impl std::error::Error for BudgetError {}

impl AlkahestError for BudgetError {
    fn code(&self) -> &'static str {
        match self {
            BudgetError::WallClock { .. } => "E-BUDGET-001",
            BudgetError::Steps { .. } => "E-BUDGET-002",
            BudgetError::Cancelled => "E-BUDGET-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            BudgetError::WallClock { .. } => Some(
                "raise Budget(wall_ms=...), or accept a heuristic/numeric result for this \
                 candidate instead of an exact one",
            ),
            BudgetError::Steps { .. } => Some(
                "raise Budget(max_steps=...), or accept a partial/heuristic result for this \
                 candidate instead of an exact one",
            ),
            BudgetError::Cancelled => Some(
                "call alkahest.clear_cancel() (Python) or budget::clear_cancel() (Rust) before \
                 starting the next candidate",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// `CANCELLED` is a process-wide `AtomicBool` by design (see the module
    /// docs) so an orchestrator thread can cancel a heavy call running on a
    /// worker thread. That means tests which flip it must not run
    /// concurrently with *any* other test that calls [`check`] — including
    /// tests in this module that never touch cancellation themselves — or
    /// they'll intermittently observe a stale `true` from a racing test and
    /// fail with `Cancelled` instead of the error under test. Every test
    /// below acquires this lock for its whole body to serialize with the
    /// cancel-flipping tests; `cargo test` still runs them in parallel with
    /// unrelated tests elsewhere in the crate, but nothing else in the crate
    /// calls `request_cancel`, so those stay unaffected.
    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> MutexGuard<'static, ()> {
        TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Clears cancellation on drop so a panicking assertion mid-test doesn't
    /// leave `CANCELLED` set for the next test to acquire the lock.
    struct CancelGuard;
    impl Drop for CancelGuard {
        fn drop(&mut self) {
            clear_cancel();
        }
    }

    #[test]
    fn no_budget_active_never_trips() {
        let _serial = serial();
        assert!(!is_active());
        assert_eq!(seed(), None);
        for _ in 0..1000 {
            assert!(check().is_ok());
        }
    }

    #[test]
    fn step_budget_trips_after_limit() {
        let _serial = serial();
        let _guard = enter(Budget::new().with_max_steps(3));
        assert!(check().is_ok());
        assert!(check().is_ok());
        assert!(check().is_ok());
        let err = check().unwrap_err();
        assert_eq!(err.code(), "E-BUDGET-002");
        assert_eq!(err, BudgetError::Steps { limit: 3, taken: 4 });
    }

    #[test]
    fn wall_budget_trips_after_elapsed() {
        let _serial = serial();
        let _guard = enter(Budget::new().with_wall(Duration::from_millis(10)));
        assert!(check().is_ok());
        std::thread::sleep(Duration::from_millis(25));
        let err = check().unwrap_err();
        assert_eq!(err.code(), "E-BUDGET-001");
        assert!(matches!(err, BudgetError::WallClock { .. }));
    }

    #[test]
    fn seed_round_trips_through_active_budget() {
        let _serial = serial();
        assert_eq!(seed(), None);
        {
            let _guard = enter(Budget::new().with_seed(7));
            assert_eq!(seed(), Some(7));
        }
        // Popped: seed is no longer visible.
        assert_eq!(seed(), None);
    }

    #[test]
    fn nested_budgets_shadow_not_merge() {
        let _serial = serial();
        let _outer = enter(Budget::new().with_seed(1).with_max_steps(1000));
        assert_eq!(seed(), Some(1));
        {
            // Inner budget has no seed set — it does NOT inherit the outer
            // seed, matching alkahest.context(...)'s non-merging semantics.
            let _inner = enter(Budget::new().with_max_steps(2));
            assert_eq!(seed(), None);
            assert!(check().is_ok());
            assert!(check().is_ok());
            assert_eq!(check().unwrap_err().code(), "E-BUDGET-002");
        }
        // Back to the outer frame: its own step counter is untouched by the
        // inner frame's checks, and its seed is visible again.
        assert_eq!(seed(), Some(1));
        assert!(check().is_ok());
    }

    #[test]
    fn guard_pops_on_early_return_via_question_mark() {
        let _serial = serial();
        fn inner() -> Result<(), BudgetError> {
            let _guard = enter(Budget::new().with_max_steps(1));
            check()?;
            check()?; // trips — guard must still pop on this early return.
            unreachable!();
        }
        assert!(inner().is_err());
        assert!(!is_active());
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

    /// The whole point of the size half: a step that asks for more work than
    /// any budget allows is refused *before* it allocates, with no budget
    /// entered at all. There is no later checkpoint to refuse at — the
    /// allocation it prevents aborts the process.
    #[test]
    fn growth_is_refused_without_an_active_budget() {
        let _serial = serial();
        assert!(!is_active());
        // A modest request is fine and does not count as a step.
        assert!(check_growth(1_000).is_ok());
        let err = check_growth(DEFAULT_MAX_GROWTH_UNITS + 1).unwrap_err();
        assert_eq!(err.code(), "E-BUDGET-002");
        assert_eq!(
            err,
            BudgetError::Steps {
                limit: DEFAULT_MAX_GROWTH_UNITS,
                taken: DEFAULT_MAX_GROWTH_UNITS + 1,
            }
        );
    }

    /// `max_steps` is the tunable for both halves, which is what makes
    /// `E-BUDGET-002`'s "raise Budget(max_steps=...)" remediation true of a
    /// growth refusal and not just of a step-count one.
    #[test]
    fn max_steps_replaces_the_default_growth_ceiling() {
        let _serial = serial();
        {
            let _guard = enter(Budget::new().with_max_steps(10));
            assert!(check_growth(10).is_ok());
            assert_eq!(
                check_growth(11).unwrap_err(),
                BudgetError::Steps {
                    limit: 10,
                    taken: 11
                }
            );
        }
        {
            // Raised well past the default: the same request now passes.
            let _guard = enter(Budget::new().with_max_steps(DEFAULT_MAX_GROWTH_UNITS * 4));
            assert!(check_growth(DEFAULT_MAX_GROWTH_UNITS + 1).is_ok());
        }
    }

    /// A growth checkpoint is also an ordinary one, so a call site needs only
    /// one of them: wall clock and cancellation still trip through it.
    #[test]
    fn growth_checkpoint_still_reports_wall_and_cancel() {
        let _serial = serial();
        {
            let _guard = enter(Budget::new().with_wall(Duration::from_millis(10)));
            assert!(check_growth(1).is_ok());
            std::thread::sleep(Duration::from_millis(25));
            assert_eq!(check_growth(1).unwrap_err().code(), "E-BUDGET-001");
        }
        let _cancel_guard = CancelGuard;
        request_cancel();
        assert_eq!(check_growth(1).unwrap_err(), BudgetError::Cancelled);
    }

    /// The comparison is per request and holds no state, so a refusal cannot
    /// latch onto unrelated later work on the same thread.
    #[test]
    fn a_growth_refusal_does_not_poison_later_calls() {
        let _serial = serial();
        assert!(check_growth(DEFAULT_MAX_GROWTH_UNITS + 1).is_err());
        assert!(check_growth(1).is_ok());
        assert!(check().is_ok());
    }

    #[test]
    fn error_codes_have_remediation() {
        for err in [
            BudgetError::WallClock {
                limit: Duration::from_secs(1),
                elapsed: Duration::from_secs(2),
            },
            BudgetError::Steps { limit: 1, taken: 2 },
            BudgetError::Cancelled,
        ] {
            assert!(err.code().starts_with("E-BUDGET-"));
            assert!(err.remediation().is_some());
            // Display must not panic and should mention something useful.
            assert!(!err.to_string().is_empty());
        }
    }
}
