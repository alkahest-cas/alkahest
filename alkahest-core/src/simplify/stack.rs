//! Segmented-stack trampoline shared by both simplification traversals.
//!
//! Both bottom-up traversals in this module — [`crate::simplify::engine`]'s
//! sequential `simplify_node` and [`crate::simplify::parallel`]'s
//! `simplify_node_par` — recurse once per level of the expression tree, with
//! no bound on how deep that goes.  A stack overflow is not a catchable error
//! on any platform this crate targets: it aborts the process, so no caller can
//! turn it into a refusal, and no `Result` can carry it.
//!
//! [`with_stack_segment`] removes the bound entirely rather than replacing it
//! with a smaller one.  Before the current *segment* (the run of levels
//! executing on one thread) spends its budget, the next level continues on a
//! freshly spawned thread with a stack this module sized.  Depth is therefore
//! limited by how many threads the OS will give us, not by any one stack, and
//! nothing is truncated or refused along the way.
//!
//! # This is not the same guard as `MAX_EXPR_DEPTH`
//!
//! [`crate::kernel::MAX_EXPR_DEPTH`] refuses an over-deep expression at the
//! **PyO3 boundary**, which covers the expressions a Python caller *hands in*.
//! It does not cover the ones this crate *builds*: `diff`, expansion and the
//! integrator all deepen an expression after it crossed that boundary, and
//! nothing re-checks. Nor does it cover Rust callers of
//! [`crate::simplify::simplify`], who never pass a boundary at all. And its
//! own docs note that it is calibrated for a release build on an 8 MiB main
//! thread — a `cargo test` worker or a rayon worker has 2 MiB and can overflow
//! well below it. The boundary check is a courtesy to the caller; this is what
//! makes the traversal itself safe.
//!
//! # Two budgets, either of which refills
//!
//! * **Levels** — an exact count of recursion levels entered since this
//!   segment began, converted from the byte budget by
//!   [`WORST_CASE_LEVEL_BYTES`].
//! * **Bytes** — how much stack the recursion has actually consumed, measured
//!   by [`probe_against_base`].
//!
//! Bytes alone used to be the whole mechanism, on the reasoning that depth is
//! a poor proxy for stack use because debug frames are several times larger
//! than release ones.  That reasoning is right about *tuning* and wrong about
//! *safety*: the byte probe can under-read to zero (see
//! [`probe_against_base`] — under AddressSanitizer it does so on every call),
//! a governor that under-reads does not refill, and a governor that does not
//! refill aborts the process.  The level count cannot under-read, so it is what actually bounds
//! the recursion; the byte budget is kept as the backstop for the case the
//! level count was mis-calibrated for, namely frames fatter than
//! [`WORST_CASE_LEVEL_BYTES`].

use std::cell::Cell;

/// Stack size handed to each refill thread by [`with_stack_segment`].
const SEGMENT_STACK_BYTES: usize = 16 * 1024 * 1024;

/// Stack the traversal may consume on a thread it did not create.  Rayon
/// workers get 2 MiB by default, so this leaves a wide margin for the frames
/// already below us and for whatever the rule engine needs.
const FOREIGN_STACK_BUDGET: usize = 512 * 1024;

/// Stack the traversal may consume on a segment thread it created itself.
/// Kept well under [`SEGMENT_STACK_BYTES`] so a single node visit can never
/// straddle the end of the segment.
const OWNED_STACK_BUDGET: usize = SEGMENT_STACK_BYTES - 4 * 1024 * 1024;

/// Upper bound on the stack one recursion level of a traversal occupies,
/// used to convert the two byte budgets above into level counts.
///
/// One level is `simplify_node` → `with_stack_segment` → `ExprPool::with`
/// → `simplify_children` → `simplify_args` and the `Vec` collection
/// machinery inlined into it — around nineteen frames in an unoptimised
/// build. Measured at **10 832 bytes** per level in the fattest configuration
/// this crate is built in (debug + AddressSanitizer, whose redzones inflate
/// every local); `16 * 1024` leaves ~50% headroom over that.
///
/// Over-estimating costs one extra thread spawn on a deep expression and
/// nothing at all on a shallow one, because the count only ever triggers the
/// same refill the byte budget already implements. Under-estimating would
/// cost a process abort, so the number is chosen high.
const WORST_CASE_LEVEL_BYTES: usize = 16 * 1024;

/// Recursion levels the traversal may enter on a thread it did not create.
///
/// The count-based twin of [`FOREIGN_STACK_BUDGET`]. It binds first in every
/// build (frames are smaller than [`WORST_CASE_LEVEL_BYTES`] everywhere the
/// crate is built), which is the intent: it is the condition that does not
/// depend on the byte probe being honest.
///
/// A tight foreign bound costs at most one extra thread spawn per top-level
/// call, not one per level — once the first segment exists, every deeper
/// refill uses the much larger [`OWNED_DEPTH_BUDGET`].
pub(crate) const FOREIGN_DEPTH_BUDGET: usize = FOREIGN_STACK_BUDGET / WORST_CASE_LEVEL_BYTES;

/// Recursion levels the traversal may enter on a segment thread it created
/// itself: the count-based twin of [`OWNED_STACK_BUDGET`].
pub(crate) const OWNED_DEPTH_BUDGET: usize = OWNED_STACK_BUDGET / WORST_CASE_LEVEL_BYTES;

// The budgets above are derived by division, so a future edit to any of the
// byte constants can silently produce a nonsensical one. Checked here rather
// than in a `#[test]` so a mis-calibration is a build failure, not a run.
const _: () = {
    // Zero would spawn a thread per node visit instead of per segment.
    assert!(FOREIGN_DEPTH_BUDGET > 0);
    // A segment thread has a bigger stack than a borrowed one, so it must be
    // allowed strictly more levels or spawning it buys nothing.
    assert!(OWNED_DEPTH_BUDGET > FOREIGN_DEPTH_BUDGET);
    // The point of the whole conversion: a full segment's worth of levels has
    // to fit in the stack that segment was given, with room left over.
    assert!(OWNED_DEPTH_BUDGET * WORST_CASE_LEVEL_BYTES < SEGMENT_STACK_BYTES);
};

/// The current thread's segment: where it started, how much it may spend, and
/// how much it has spent.
///
/// One `Cell` of four `usize`s rather than four separate `thread_local!`s
/// because this is read and written on **every node visit** of the sequential
/// simplifier, which is the hottest traversal in the crate. Bundling them
/// makes an entry/exit pair two thread-local lookups instead of eight.
#[derive(Clone, Copy)]
struct Segment {
    /// Stack address at which this segment began; 0 until the first probe.
    base: usize,
    /// How much stack this segment is allowed to consume.
    byte_budget: usize,
    /// Recursion levels currently entered on this segment.
    depth: usize,
    /// How many levels this segment is allowed to enter.
    depth_budget: usize,
}

impl Segment {
    /// The state a thread starts in: it belongs to someone else (rayon, the
    /// process main thread, a PyO3 callback), so assume the small budgets.
    const fn foreign() -> Self {
        Segment {
            base: 0,
            byte_budget: FOREIGN_STACK_BUDGET,
            depth: 0,
            depth_budget: FOREIGN_DEPTH_BUDGET,
        }
    }

    /// The state a freshly spawned segment thread starts in.
    const fn owned() -> Self {
        Segment {
            base: 0,
            byte_budget: OWNED_STACK_BUDGET,
            depth: 0,
            depth_budget: OWNED_DEPTH_BUDGET,
        }
    }
}

thread_local! {
    static SEGMENT: Cell<Segment> = const { Cell::new(Segment::foreign()) };
}

/// Holds one recursion level of the current [`Segment`] for as long as it lives.
///
/// A guard rather than a matched increment/decrement pair because the rule
/// engine and the pool can panic, and a leaked increment would make every
/// later traversal on that thread believe it was already at its depth budget
/// and refill immediately, forever.
struct DepthGuard;

impl Drop for DepthGuard {
    fn drop(&mut self) {
        SEGMENT.with(|c| {
            let mut s = c.get();
            s.depth = s.depth.saturating_sub(1);
            c.set(s);
        });
    }
}

/// Run `f`, moving it to a thread with a fresh [`SEGMENT_STACK_BYTES`] stack
/// once the current segment has spent either of its budgets.
///
/// The argument passed to `f` is `true` exactly when this call spawned the
/// segment `f` runs on, i.e. `f` is the *root* of that thread's segment and
/// will be the last thing to return on it. Callers that keep thread-local
/// side records — [`crate::simplify::rules`]'s declined-expansion log is the
/// one that matters — use it to drain their record before the thread dies,
/// since the caller's own thread will never see it. Callers with no such
/// record ignore it.
///
/// Segments are spawned scoped, so borrowed pools, rules and memos stay valid.
///
/// Inside a segment the ambient rayon pool is no longer installed, so nested
/// `par_iter` calls fall back to the global pool.  That only affects subtrees
/// deep enough to exhaust a whole segment, where returning an answer at all
/// matters more than which pool schedules the work.
///
/// # Panics
///
/// Panics if the OS refuses to create the segment thread — depth is bounded by
/// the thread limit, and past it there is no stack left to continue on. This
/// is an unwind, which a caller can catch and the PyO3 layer turns into an
/// exception; the stack overflow it replaces was an abort, which nothing can.
pub(crate) fn with_stack_segment<R: Send>(f: impl FnOnce(bool) -> R + Send) -> R {
    let probe = 0u8;
    let here = &probe as *const u8 as usize;
    let over_budget = SEGMENT.with(|c| {
        let mut s = c.get();
        let (rebased, used) = probe_against_base(s.base, here);
        s.base = rebased;
        let over = s.depth >= s.depth_budget || used >= s.byte_budget;
        if !over {
            s.depth += 1;
        }
        c.set(s);
        over
    });
    if !over_budget {
        let _level = DepthGuard;
        return f(false);
    }
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(SEGMENT_STACK_BYTES)
            .spawn_scoped(scope, || {
                // Fresh thread: its own thread-local, so the segment's base
                // address and level count both start from here.
                SEGMENT.with(|c| {
                    let mut s = Segment::owned();
                    s.depth = 1;
                    c.set(s);
                });
                let _level = DepthGuard;
                f(true)
            })
            .expect("failed to spawn stack segment for deep recursion")
            .join()
            .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
    })
}

/// Given the segment's current stack baseline and a fresh probe address,
/// return the baseline to keep and the bytes consumed since the segment began.
///
/// [`with_stack_segment`] takes the address of a local as a stack-depth probe
/// and feeds it here. Stacks grow downwards on every platform this crate
/// targets, so a *smaller* address means deeper.
///
/// The baseline is re-established whenever the probe lands at or above it.
/// That matters because Rayon reuses its workers: the baseline used to be
/// latched on a thread's first probe and never revisited, so a worker that
/// happened to take its first `simplify_par` task from deep inside a call
/// chain kept that deep address as its baseline forever.  Every later task on
/// that worker started *above* the stale baseline, the subtraction floored
/// the difference at 0, and the traversal read its own stack usage as zero no
/// matter how deep it went — so it never refilled, and ran off the end of the
/// worker's 2 MiB stack.  A stack overflow aborts the process, which is
/// precisely what this machinery exists to prevent.
///
/// Re-baselining upwards is always safe: an address above the current
/// baseline means the frames that baseline was measured against have already
/// returned, so it describes a stack that no longer exists.
///
/// # The probe can under-read, and does under AddressSanitizer
///
/// The probe assumes the local whose address it takes lives on the real
/// stack.  Under ASan's stack-use-after-return detection it does not: locals
/// whose address escapes are moved into a per-thread "fake stack" ring, whose
/// addresses *ascend* with recursion depth and then wrap.  Ascending
/// addresses hit the re-baseline branch on every call, so this reported `0`
/// however deep the traversal went; after a wrap it returned a bounded value
/// that never approached [`OWNED_STACK_BUDGET`].  The traversal therefore
/// never refilled and ran off the end of a real stack that was filling up all
/// along — the nightly `asan` shard died in
/// `par_survives_deep_chain_on_worker_thread` this way, and raising
/// `RUST_MIN_STACK` did not help because the probe reports the same `0` at
/// any stack size.
///
/// Reading the real stack pointer instead would need inline assembly, which
/// this crate does not use, and no `cfg` distinguishes "locals are being
/// relocated" from "they are not".  So the byte budget is advisory: it can
/// only make [`with_stack_segment`] refill *earlier* than the level count
/// does, never later.  Nothing may depend on it alone.
///
/// This half is split out from the probe because it is pure arithmetic that
/// can be checked against synthetic addresses on any target, while the probe
/// half is not observable under instrumentation that relocates locals.
fn probe_against_base(base: usize, here: usize) -> (usize, usize) {
    if base == 0 || here >= base {
        (here, 0)
    } else {
        (base, base - here)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stack baseline used to be latched on a thread's first probe and
    /// never revisited.  Rayon reuses its workers, so a worker whose first
    /// task probed from deep in a call chain kept that deep address forever;
    /// every later task started above it, the subtraction floored the result
    /// at 0, and the traversal believed it was using no stack however deep it
    /// went.
    ///
    /// Asserted on [`probe_against_base`] with synthetic addresses rather
    /// than by taking real ones. Real addresses are not a usable oracle here:
    /// under AddressSanitizer the locals this probe would take the address of
    /// live in a fake-stack ring whose addresses *ascend* with depth, so a
    /// "deeper probe reports more" assertion is false there by construction —
    /// which is precisely why nothing load-bearing may rest on the probe (see
    /// `probe_against_base` and `with_stack_segment`). Synthetic addresses check the
    /// same decision on every target, and the sibling test
    /// `stack_segments_are_bounded_by_level_count_alone` covers the bound
    /// that actually has to hold.
    ///
    /// Stacks grow downwards, so a *smaller* address means deeper.
    #[test]
    fn stack_probe_rebaselines_after_unwinding() {
        // A thread's first probe establishes the baseline and reports nothing.
        assert_eq!(probe_against_base(0, 0x9000), (0x9000, 0));

        // Descending from it is measured against it.
        assert_eq!(probe_against_base(0x9000, 0x8c00), (0x9000, 0x400));
        assert_eq!(probe_against_base(0x9000, 0x8000), (0x9000, 0x1000));

        // The regression: a probe at or above the baseline means the frames
        // it was measured against have returned, so the baseline moves up...
        assert_eq!(probe_against_base(0x9000, 0x9000), (0x9000, 0));
        assert_eq!(probe_against_base(0x8000, 0x9000), (0x9000, 0));

        // ...and a later descent is then measured from *there*, rather than
        // being floored at 0 by a stale, deeper baseline. Pre-fix the
        // baseline stayed at 0x8000 and this reported 0.
        assert_eq!(probe_against_base(0x9000, 0x8800), (0x9000, 0x800));
    }

    /// The refill must be bounded by the *level count* on its own, with no
    /// help at all from the byte probe.
    ///
    /// This is the property whose absence killed the nightly `asan` shard
    /// every night from 2026-08-19 on. Under AddressSanitizer the byte probe
    /// reads `0` however deep the traversal goes (locals live in ASan's fake
    /// stack, whose addresses ascend with depth, so every probe takes the
    /// re-baseline branch), the traversal never refilled, and
    /// `par_survives_deep_chain_on_worker_thread` overflowed a real stack that
    /// had been filling up all along. `RUST_MIN_STACK` did not help: the probe
    /// reports the same `0` at any stack size.
    ///
    /// Asserted with frames far too small for the byte budget to be what
    /// fires — a few hundred bytes a level against a 512 KiB foreign budget —
    /// so what is measured here really is the count. A regression must fail
    /// this test rather than abort the test process, which is why it counts
    /// levels per thread instead of trying to overflow something.
    #[test]
    fn stack_segments_are_bounded_by_level_count_alone() {
        use std::sync::Mutex;
        use std::thread::ThreadId;

        fn nest(levels: usize, seen: &Mutex<Vec<ThreadId>>) {
            seen.lock().unwrap().push(std::thread::current().id());
            if levels > 0 {
                with_stack_segment(|_| nest(levels - 1, seen));
            }
        }

        const LEVELS: usize = 2_000;
        let seen = Mutex::new(Vec::with_capacity(LEVELS + 1));
        nest(LEVELS, &seen);
        let seen = seen.into_inner().unwrap();

        assert_eq!(
            seen.len(),
            LEVELS + 1,
            "every level must run exactly once and the recursion must return"
        );

        // Maximal runs of consecutive levels executed on one thread: each is
        // one segment, and each must have stopped at its budget.
        let mut runs: Vec<(ThreadId, usize)> = Vec::new();
        for id in seen {
            match runs.last_mut() {
                Some((prev, n)) if *prev == id => *n += 1,
                _ => runs.push((id, 1)),
            }
        }
        assert!(
            runs.len() > 1,
            "the level budget never fired, so nothing here was tested: {} levels ran on one \
             thread",
            LEVELS + 1
        );
        // +1 for the level that observes it is over budget and does the spawn.
        let limit = OWNED_DEPTH_BUDGET.max(FOREIGN_DEPTH_BUDGET) + 1;
        for (i, (_, n)) in runs.iter().enumerate() {
            assert!(
                *n <= limit,
                "segment {i} ran {n} levels on one thread, over the {limit}-level budget"
            );
        }
    }

    /// Exactly the level that spawns a segment is told so, and it is told so
    /// only once per thread the recursion moves onto.
    #[test]
    fn only_the_segment_root_is_flagged_as_fresh() {
        use std::sync::Mutex;

        fn nest(levels: usize, fresh: bool, log: &Mutex<Vec<(usize, bool)>>) {
            log.lock().unwrap().push((levels, fresh));
            if levels > 0 {
                with_stack_segment(|f| nest(levels - 1, f, log));
            }
        }

        let levels = FOREIGN_DEPTH_BUDGET * 3;
        let log = Mutex::new(Vec::new());
        nest(levels, false, &log);
        let log = log.into_inner().unwrap();

        let fresh_count = log.iter().filter(|(_, f)| *f).count();
        assert!(
            fresh_count >= 1,
            "no segment was spawned, so the flag was never exercised"
        );
        // One spawn happens at the foreign budget; every later one needs a
        // whole owned segment, which `levels` is far too small to fill.
        assert_eq!(
            fresh_count, 1,
            "expected exactly one segment root over {levels} levels, got {fresh_count}"
        );
    }
}
