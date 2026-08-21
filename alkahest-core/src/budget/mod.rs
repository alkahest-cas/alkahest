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
//! | `E-BUDGET-002` | [`BudgetError::Steps`]     | step counter exceeded `max_steps` |
//! | `E-BUDGET-003` | [`BudgetError::Cancelled`] | [`request_cancel`] was called |
//!
//! Memory ceilings are reported through [`BudgetTrip`] rather than
//! [`BudgetError`]: `BudgetError` is a public *exhaustive* enum, so growing it
//! a `Memory` variant is a major semver break. [`check_all`] returns
//! [`BudgetTrip`], which wraps a [`BudgetError`] or carries one of the two
//! memory refusals:
//!
//! | Code           | Variant                       | Cause             |
//! |----------------|-------------------------------|-------------------|
//! | `E-BUDGET-004` | [`BudgetTrip::Memory`]        | the active budget's `max_bytes` ceiling |
//! | `E-BUDGET-005` | [`BudgetTrip::AddressSpace`]  | the process is about to exhaust `RLIMIT_AS` |
//!
//! See [`mod@memory`] for why the memory ceiling is enforced at these
//! checkpoints rather than inside a fallible allocator (GMP does not have
//! one, and a Rust `panic!` may not cross a C frame).

use crate::errors::AlkahestError;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub mod memory;

pub use memory::{gmp_live_bytes, install as install_memory_accounting};

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
    /// Maximum number of [`check`] calls the guarded block may make.
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
    /// Ceiling on GMP bytes held live by this block, from
    /// [`enter_with_memory`].
    max_bytes: Option<u64>,
    /// GMP live-byte total when this frame was pushed, so `max_bytes` measures
    /// *this block's* appetite rather than whatever the process was already
    /// holding.
    bytes_at_entry: u64,
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
    enter_with_memory(budget, None)
}

/// [`enter`], plus a ceiling on the GMP memory the guarded block may hold
/// live (see [`mod@memory`]); `None` imposes no ceiling.
///
/// # Why this is not a `Budget` field
///
/// [`Budget`] has only public fields and is not `#[non_exhaustive]`, so every
/// caller may build one with a struct literal — and adding a field to such a
/// struct is a major semver break (`cargo semver-checks`'
/// `constructible_struct_adds_field`). A free function is additive, so the
/// memory ceiling travels beside the budget instead of inside it. The Python
/// binding still spells it `Budget(max_bytes=...)`, because a `@dataclass`
/// with a defaulted trailing field *is* additive there.
pub fn enter_with_memory(budget: Budget, max_bytes: Option<u64>) -> BudgetGuard {
    if max_bytes.is_some() {
        memory::install();
    }
    let frame = Frame {
        start: Instant::now(),
        wall: budget.wall,
        max_steps: budget.max_steps,
        steps: Cell::new(0),
        seed: budget.seed,
        max_bytes,
        bytes_at_entry: memory::gmp_live_bytes(),
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

/// The `max_bytes` ceiling of the innermost active budget frame, or `None`.
pub fn max_bytes() -> Option<u64> {
    STACK.with(|s| s.borrow().last().and_then(|f| f.max_bytes))
}

/// GMP bytes held live *by the innermost active budget frame* — the total now,
/// less the total when that frame was entered.
///
/// Zero when no budget is active. Saturating: a block allocated before the
/// frame was entered and freed inside it would otherwise underflow.
pub fn bytes_used() -> u64 {
    STACK.with(|s| {
        s.borrow()
            .last()
            .map(|f| memory::gmp_live_bytes().saturating_sub(f.bytes_at_entry))
            .unwrap_or(0)
    })
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

// ---------------------------------------------------------------------------
// Memory checkpoints
// ---------------------------------------------------------------------------

/// How often [`check_memory`] pays for a `/proc/self/statm` read.
///
/// The per-frame `max_bytes` ceiling is a pair of atomic loads and is checked
/// every time; the address-space probe is a file read (a few microseconds), so
/// it runs on every `PROBE_INTERVAL`-th call — often enough that a pivot loop
/// cannot climb a whole [`memory::reserve_bytes`] between probes, rarely
/// enough not to show up in a profile.
const PROBE_INTERVAL: u32 = 16;

/// How many probe intervals' worth of *observed* growth the address-space
/// guard keeps in reserve, on top of [`memory::reserve_bytes`].
///
/// A flat reserve alone is a bet that no probe interval can cross it, and that
/// bet is wrong for a solve whose matrix entries double: under a 900 MB
/// `ulimit -v` the m = 4 multinomial was seen to add 26 MB between two
/// consecutive probes. Scaling the reserve by the growth actually observed
/// makes the guard tighten exactly when the workload accelerates, and stay out
/// of the way when it does not — which matters, because the import alone maps
/// ~600 MB of address space, so a large flat reserve would refuse work that
/// fits.
const GROWTH_RESERVE_FACTOR: u64 = 4;

thread_local! {
    static PROBE_TICK: Cell<u32> = const { Cell::new(0) };
    /// Address space mapped at the previous probe, for the growth term above.
    /// `0` means "no history yet on this thread".
    static LAST_VSZ: Cell<u64> = const { Cell::new(0) };
    /// The trip behind the engine-specific error the current thread is about
    /// to return — see [`record_trip`].
    static LAST_TRIP: Cell<Option<BudgetTrip>> = const { Cell::new(None) };
}

/// [`check`] *and* the memory ceilings — the checkpoint a heavy exact-
/// arithmetic loop should call.
///
/// Returns [`BudgetTrip::Budget`] for the wall/step/cancel cases (so an
/// existing `check` call site can be upgraded without changing what it
/// reports) and [`BudgetTrip::Memory`] / [`BudgetTrip::AddressSpace`] for the
/// memory ones.
pub fn check_all() -> Result<(), BudgetTrip> {
    check()?;
    check_memory()
}

/// The memory half of [`check_all`], without the wall/step/cancel checks.
pub fn check_memory() -> Result<(), BudgetTrip> {
    let framed = STACK.with(|s| {
        let stack = s.borrow();
        stack.last().and_then(|f| {
            f.max_bytes.map(|limit| {
                (
                    limit,
                    memory::gmp_live_bytes().saturating_sub(f.bytes_at_entry),
                )
            })
        })
    });
    if let Some((limit, used)) = framed {
        if used > limit {
            return Err(BudgetTrip::Memory { limit, used });
        }
    }
    if memory::address_space_limit().is_some() {
        let probe = PROBE_TICK.with(|t| {
            let n = t.get().wrapping_add(1);
            t.set(n);
            n % PROBE_INTERVAL == 1
        });
        if probe {
            if let (Some(limit), Some(used)) =
                (memory::address_space_limit(), memory::address_space_used())
            {
                let prev = LAST_VSZ.with(|c| c.replace(used));
                let growth = if prev == 0 {
                    0
                } else {
                    used.saturating_sub(prev)
                };
                let reserve =
                    memory::reserve_bytes(limit).max(growth.saturating_mul(GROWTH_RESERVE_FACTOR));
                if used.saturating_add(reserve) >= limit {
                    return Err(BudgetTrip::AddressSpace {
                        limit,
                        used,
                        reserve,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Pre-flight check for a caller that is about to allocate `bytes` in one go.
///
/// Unlike [`check_memory`] this always pays for the address-space probe: a
/// site that can name its allocation size up front is exactly the site where
/// one step can cross the whole reserve, so it is worth a syscall to refuse
/// *before* the allocation rather than after the next checkpoint.
pub fn check_alloc(bytes: u64) -> Result<(), BudgetTrip> {
    check()?;
    let framed = STACK.with(|s| {
        let stack = s.borrow();
        stack.last().and_then(|f| {
            f.max_bytes.map(|limit| {
                (
                    limit,
                    memory::gmp_live_bytes().saturating_sub(f.bytes_at_entry),
                )
            })
        })
    });
    if let Some((limit, used)) = framed {
        if used.saturating_add(bytes) > limit {
            return Err(BudgetTrip::Memory {
                limit,
                used: used.saturating_add(bytes),
            });
        }
    }
    if let (Some(limit), Some(used)) = (memory::address_space_limit(), memory::address_space_used())
    {
        let projected = used.saturating_add(bytes);
        let reserve = memory::reserve_bytes(limit);
        if projected.saturating_add(reserve) >= limit {
            return Err(BudgetTrip::AddressSpace {
                limit,
                used: projected,
                reserve,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Out-of-band trip reporting
// ---------------------------------------------------------------------------

/// Record the trip that is about to be reported as an engine-specific error.
///
/// Engines whose error enums are public and exhaustive (`HolonomicError`,
/// `Telescoping2dError`, `SosError`, …) cannot grow a `Budget` variant without
/// a major semver break, so they return their own "gave up" variant and leave
/// the real cause here for the bindings to pick up with [`take_trip`] — the
/// pattern [`crate::calculus::limits::last_budget_trip`] established for
/// wall-clock trips inside `LimitError::DepthExceeded`.
///
/// Call [`clear_trip`] at the outermost entry of such an engine so a stale
/// trip from an earlier call can never be attributed to this one.
pub fn record_trip(trip: BudgetTrip) {
    LAST_TRIP.with(|c| c.set(Some(trip)));
}

/// Take (and clear) the trip recorded by [`record_trip`] on this thread.
pub fn take_trip() -> Option<BudgetTrip> {
    LAST_TRIP.with(|c| c.take())
}

/// Clear any recorded trip. Call at the outermost entry of an engine that
/// reports trips out of band.
pub fn clear_trip() {
    LAST_TRIP.with(|c| c.set(None));
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
    /// The active budget's step counter exceeded `max_steps`.
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

/// Why a [`check_all`] checkpoint stopped a call.
///
/// # Why this is not a `BudgetError` variant
///
/// [`BudgetError`] is a public *exhaustive* enum: adding `Memory` to it is a
/// major semver break, and so is marking it `#[non_exhaustive]` to allow one
/// later. This enum is new, so it can be `#[non_exhaustive]` from birth and
/// grow without breaking anyone; [`From<BudgetError>`] keeps the two in one
/// `?`-chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BudgetTrip {
    /// A wall-clock, step, or cancellation trip — see [`BudgetError`].
    Budget(BudgetError),
    /// The active budget's `max_bytes` ceiling was reached
    /// ([`enter_with_memory`], `Budget(max_bytes=...)` in Python).
    Memory {
        /// The ceiling, in bytes.
        limit: u64,
        /// GMP bytes held live by the guarded block when it tripped.
        used: u64,
    },
    /// The process is within [`memory::reserve_bytes`] of its `RLIMIT_AS`, so
    /// the next large allocation would `abort()` inside GMP or the Rust
    /// allocator rather than fail.
    ///
    /// Unlike [`BudgetTrip::Memory`] this fires with **no budget active**: the
    /// operator who set `ulimit -v` (or a container memory limit) already said
    /// how much the process may have, and refusing inside that number is
    /// strictly better than dying at it.
    AddressSpace {
        /// The process's soft `RLIMIT_AS`, in bytes.
        limit: u64,
        /// Address space mapped by the process when it tripped, in bytes.
        used: u64,
        /// Headroom the guard was keeping below `limit` — the flat reserve of
        /// [`memory::reserve_bytes`], widened by the growth observed between
        /// the last two probes.
        reserve: u64,
    },
}

impl From<BudgetError> for BudgetTrip {
    fn from(e: BudgetError) -> Self {
        BudgetTrip::Budget(e)
    }
}

impl BudgetTrip {
    /// The [`BudgetError`] behind a wall/step/cancel trip, or `None` for a
    /// memory trip.
    pub fn budget_error(&self) -> Option<BudgetError> {
        match self {
            BudgetTrip::Budget(e) => Some(*e),
            _ => None,
        }
    }
}

impl fmt::Display for BudgetTrip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetTrip::Budget(e) => e.fmt(f),
            BudgetTrip::Memory { limit, used } => write!(
                f,
                "budget exceeded: exact-arithmetic memory limit {limit} bytes reached \
                 ({used} bytes held live)"
            ),
            BudgetTrip::AddressSpace {
                limit,
                used,
                reserve,
            } => write!(
                f,
                "budget exceeded: refusing before the process address-space limit \
                 ({used} of {limit} bytes mapped, reserve {reserve} bytes) — the allocation \
                 that would follow cannot fail safely, GMP and the Rust allocator both abort"
            ),
        }
    }
}

impl std::error::Error for BudgetTrip {}

impl AlkahestError for BudgetTrip {
    fn code(&self) -> &'static str {
        match self {
            BudgetTrip::Budget(e) => e.code(),
            BudgetTrip::Memory { .. } => "E-BUDGET-004",
            BudgetTrip::AddressSpace { .. } => "E-BUDGET-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            BudgetTrip::Budget(e) => e.remediation(),
            BudgetTrip::Memory { .. } => Some(
                "raise Budget(max_bytes=...), or ask for a smaller problem (fewer unknowns, \
                 a lower order/degree) — the exact coefficients, not the shape of the system, \
                 are what grew",
            ),
            BudgetTrip::AddressSpace { .. } => Some(
                "raise the process address-space limit (ulimit -v, or the container/cgroup \
                 memory limit), or ask for a smaller problem — this refusal replaces the \
                 uncatchable abort that would otherwise follow",
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

    /// ~8 MB of GMP limbs, well clear of the 1 MiB ceiling the memory tests
    /// set and of any noise from tests running in parallel (the GMP live-byte
    /// total is process-wide by necessity — see `memory::gmp_live_bytes`).
    fn big_gmp_integer() -> rug::Integer {
        let mut z = rug::Integer::from(1);
        z <<= 64_000_000;
        z
    }

    #[test]
    fn memory_budget_trips_when_gmp_memory_passes_max_bytes() {
        let _serial = serial();
        let _guard = enter_with_memory(Budget::new(), Some(1 << 20));
        assert!(
            check_all().is_ok(),
            "an empty frame must not trip before anything is allocated"
        );
        let z = big_gmp_integer();
        let err = check_all().unwrap_err();
        assert_eq!(err.code(), "E-BUDGET-004");
        assert!(
            matches!(err, BudgetTrip::Memory { limit, used } if limit == 1 << 20 && used > limit),
            "{err:?}"
        );
        drop(z);
    }

    #[test]
    fn a_generous_memory_budget_does_not_trip() {
        let _serial = serial();
        // 1 TiB: nothing this process can allocate reaches it, so this pins
        // that the ceiling is a ceiling and not an unconditional refusal.
        let _guard = enter_with_memory(Budget::new(), Some(1 << 40));
        let z = big_gmp_integer();
        assert!(check_all().is_ok());
        drop(z);
    }

    #[test]
    fn check_alloc_refuses_before_the_allocation_is_made() {
        let _serial = serial();
        let _guard = enter_with_memory(Budget::new(), Some(1 << 20));
        // The refusal is entirely pre-flight — this frame has allocated
        // nothing — which is the whole point: GMP has no failure path to take
        // once it has been asked for the memory.
        let err = check_alloc(4 << 20).unwrap_err();
        assert_eq!(err.code(), "E-BUDGET-004");
        // The reported `used` is the *projected* total, i.e. it accounts for
        // the allocation that has not happened. Not asserted exactly: GMP's
        // live-byte total is process-wide (its allocation hooks are), so a
        // test running in parallel can contribute to this frame's delta.
        assert!(
            matches!(err, BudgetTrip::Memory { used, .. } if used >= 4 << 20),
            "{err:?}"
        );
    }

    #[test]
    fn max_bytes_is_visible_and_scoped_to_its_frame() {
        let _serial = serial();
        assert_eq!(max_bytes(), None);
        {
            let _guard = enter_with_memory(Budget::new(), Some(4096));
            assert_eq!(max_bytes(), Some(4096));
            // A plain `enter` shadows it, matching the non-merging nesting of
            // every other budget field.
            let _inner = enter(Budget::new());
            assert_eq!(max_bytes(), None);
        }
        assert_eq!(max_bytes(), None);
    }

    #[test]
    fn trips_round_trip_through_the_out_of_band_carrier() {
        let _serial = serial();
        clear_trip();
        assert_eq!(take_trip(), None);
        record_trip(BudgetTrip::Memory { limit: 1, used: 2 });
        let taken = take_trip().expect("recorded");
        assert_eq!(taken.code(), "E-BUDGET-004");
        // Taking clears, so a later unrelated error cannot inherit it.
        assert_eq!(take_trip(), None);
    }

    #[test]
    fn budget_trip_codes_have_remediation() {
        for trip in [
            BudgetTrip::Budget(BudgetError::Cancelled),
            BudgetTrip::Memory { limit: 1, used: 2 },
            BudgetTrip::AddressSpace {
                limit: 3,
                used: 2,
                reserve: 1,
            },
        ] {
            assert!(trip.code().starts_with("E-BUDGET-"));
            assert!(trip.remediation().is_some());
            assert!(!trip.to_string().is_empty());
        }
        assert_eq!(
            BudgetTrip::from(BudgetError::Cancelled).budget_error(),
            Some(BudgetError::Cancelled)
        );
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
